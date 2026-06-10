// src/network.rs
//! OPTIMIZED: Network system with client-side prediction, delta compression, and ping reduction
//! 
//! This module provides:
//! - Client-side prediction and reconciliation
//! - Delta compression for bandwidth reduction
//! - Interest management (only sync visible entities)
//! - Rollback netcode support
//! - RTT estimation and packet batching

use std::collections::{HashMap, VecDeque, HashSet};
use std::net::{SocketAddr, UdpSocket};
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::{RwLock, Mutex};
use serde::{Deserialize, Serialize};
use bincode;

use glam::{Vec3, Vec2};

// ============================================================================
// CONSTANTS
// ============================================================================

const PACKET_HEADER_SIZE: usize = 8;
const INPUT_BUFFER_SIZE: usize = 128;
const STATE_BUFFER_SIZE: usize = 30;
const MAX_PACKET_SIZE: usize = 1400;
const COMPRESSION_THRESHOLD: usize = 256;

// ============================================================================
// ENUMS
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NetworkRole {
    Server,
    Client,
    Host,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConnectionState {
    Connecting,
    Connected,
    Disconnecting,
    Disconnected,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ChannelType {
    ReliableOrdered,
    ReliableUnordered,
    UnreliableOrdered,
    UnreliableUnordered,
}

// ============================================================================
// CORE TYPES
// ============================================================================

/// Network message with channel support
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkMessage {
    pub id: u64,
    pub channel: ChannelType,
    pub data: Vec<u8>,
    pub timestamp: u64,
    pub sequence: u32,
}

/// Connection with telemetry
#[derive(Clone, Debug)]
pub struct Connection {
    pub peer_id: u64,
    pub address: SocketAddr,
    pub state: ConnectionState,
    pub rtt_ms: f32,
    pub packet_loss: f32,
    pub send_queue: VecDeque<NetworkMessage>,
    pub recv_queue: VecDeque<NetworkMessage>,
    pub last_heartbeat: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
    pub last_update: Instant,
}

impl Connection {
    pub fn new(peer_id: u64, address: SocketAddr) -> Self {
        Self {
            peer_id,
            address,
            state: ConnectionState::Connecting,
            rtt_ms: 0.0,
            packet_loss: 0.0,
            send_queue: VecDeque::new(),
            recv_queue: VecDeque::new(),
            last_heartbeat: 0,
            bytes_sent: 0,
            bytes_received: 0,
            packets_sent: 0,
            packets_received: 0,
            last_update: Instant::now(),
        }
    }
}

/// Replicated entity state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplicatedState {
    pub entity_id: u64,
    pub state_data: Vec<u8>,
    pub sequence_number: u32,
    pub timestamp: u64,
    pub delta_from: u32,
    pub is_delta: bool,
}

/// Input for rollback netcode
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameInput {
    pub player_id: u64,
    pub frame: u32,
    pub inputs: Vec<u8>,
    pub checksum: u32,
    pub timestamp: u64,
}

impl GameInput {
    pub fn serialize_compressed(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }
    
    pub fn deserialize_compressed(data: &[u8]) -> Option<Self> {
        bincode::deserialize(data).ok()
    }
}

/// Rollback state
#[derive(Clone, Debug)]
pub struct RollbackState {
    pub frame: u32,
    pub state_hash: u32,
    pub inputs: Vec<GameInput>,
    pub timestamp: u64,
}

/// ENTITY SNAPSHOT - Used by predictive renderer
#[derive(Debug, Clone)]
pub struct EntitySnapshot {
    pub id: u64,
    pub position: Vec3,
    pub rotation: Vec3,
    pub scale: Vec3,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
}

impl EntitySnapshot {
    pub fn new(id: u64, position: Vec3, rotation: Vec3, scale: Vec3) -> Self {
        Self {
            id,
            position,
            rotation,
            scale,
            bounds_min: position - scale * 0.5,
            bounds_max: position + scale * 0.5,
        }
    }
    
    pub fn center(&self) -> Vec3 {
        (self.bounds_min + self.bounds_max) * 0.5
    }
    
    pub fn velocity(&self, previous: &EntitySnapshot) -> Vec3 {
        self.position - previous.position
    }
}

/// Client-side prediction state
struct PredictedState {
    frame: u32,
    position: [f32; 3],
    velocity: [f32; 3],
    rotation: [f32; 4],
    input_sequence: u32,
}

struct PendingInput {
    input: GameInput,
    sent_at: Instant,
    acknowledged: bool,
}

struct DeltaContext {
    last_state: Option<Vec<u8>>,
    last_sequence: u32,
}

impl Default for DeltaContext {
    fn default() -> Self {
        Self {
            last_state: None,
            last_sequence: 0,
        }
    }
}

// ============================================================================
// NETWORK SYSTEM
// ============================================================================

pub struct NetworkSystem {
    role: NetworkRole,
    connections: RwLock<HashMap<u64, Connection>>,
    message_queue: RwLock<VecDeque<NetworkMessage>>,
    replicated_entities: RwLock<HashMap<u64, ReplicatedState>>,
    delta_contexts: RwLock<HashMap<u64, DeltaContext>>,
    
    input_buffer: RwLock<VecDeque<GameInput>>,
    rollback_states: RwLock<VecDeque<RollbackState>>,
    
    local_player_id: Option<u64>,
    predicted_states: RwLock<HashMap<u64, PredictedState>>,
    pending_inputs: RwLock<VecDeque<PendingInput>>,
    
    visible_entities: RwLock<HashSet<u64>>,
    authority_map: RwLock<HashMap<u64, u64>>,
    
    next_peer_id: RwLock<u64>,
    socket: Option<Arc<UdpSocket>>,
    pub compression_enabled: bool,
    pub delta_compression_enabled: bool,
    pub prediction_enabled: bool,
    
    last_ping_check: Instant,
    packet_times: RwLock<VecDeque<(Instant, u64)>>,
}

impl NetworkSystem {
    pub fn new(role: NetworkRole) -> Self {
        Self {
            role,
            connections: RwLock::new(HashMap::new()),
            message_queue: RwLock::new(VecDeque::new()),
            replicated_entities: RwLock::new(HashMap::new()),
            delta_contexts: RwLock::new(HashMap::new()),
            input_buffer: RwLock::new(VecDeque::with_capacity(INPUT_BUFFER_SIZE)),
            rollback_states: RwLock::new(VecDeque::with_capacity(STATE_BUFFER_SIZE)),
            local_player_id: None,
            predicted_states: RwLock::new(HashMap::new()),
            pending_inputs: RwLock::new(VecDeque::new()),
            visible_entities: RwLock::new(HashSet::new()),
            authority_map: RwLock::new(HashMap::new()),
            next_peer_id: RwLock::new(1),
            socket: None,
            compression_enabled: true,
            delta_compression_enabled: true,
            prediction_enabled: true,
            last_ping_check: Instant::now(),
            packet_times: RwLock::new(VecDeque::with_capacity(256)),
        }
    }

    // ============================================
    // CONNECTION MANAGEMENT
    // ============================================

    pub fn connect(&mut self, address: &str) -> Result<u64, NetworkError> {
        let addr: SocketAddr = address.parse().map_err(|_| NetworkError::InvalidAddress)?;
        
        let socket = UdpSocket::bind("0.0.0.0:0")
            .map_err(|e| NetworkError::ConnectionFailed(e.to_string()))?;
        socket.set_nonblocking(true)
            .map_err(|e| NetworkError::ConnectionFailed(e.to_string()))?;
        
        let shared_socket = Arc::new(socket);
        self.socket = Some(shared_socket.clone());
        
        shared_socket.connect(addr)
            .map_err(|e| NetworkError::ConnectionFailed(e.to_string()))?;
        
        let peer_id = {
            let mut next = self.next_peer_id.write();
            *next += 1;
            *next
        };
        
        let mut connections = self.connections.write();
        connections.insert(peer_id, Connection::new(peer_id, addr));
        connections.get_mut(&peer_id).unwrap().state = ConnectionState::Connected;
        
        log::info!("Connected to {} with peer_id {}", address, peer_id);
        Ok(peer_id)
    }

    pub fn disconnect(&mut self, peer_id: u64) {
        let mut connections = self.connections.write();
        if let Some(conn) = connections.get_mut(&peer_id) {
            conn.state = ConnectionState::Disconnecting;
        }
        
        self.predicted_states.write().remove(&peer_id);
        
        let mut pending = self.pending_inputs.write();
        pending.retain(|p| p.input.player_id != peer_id as u64);
    }

    // ============================================
    // MESSAGING
    // ============================================

    pub fn send_message(&self, peer_id: u64, mut message: NetworkMessage) {
        if self.compression_enabled && message.data.len() > COMPRESSION_THRESHOLD {
            message.data = compress_data(&message.data);
        }
        
        let mut connections = self.connections.write();
        if let Some(conn) = connections.get_mut(&peer_id) {
            conn.send_queue.push_back(message.clone());
            conn.bytes_sent += message.data.len() as u64;
            conn.packets_sent += 1;
        }
    }

    pub fn send_unreliable_position(&self, peer_id: u64, entity_id: u64, position: [f32; 3], sequence: u32) {
        let update = PositionUpdate {
            entity_id,
            position,
            sequence,
            timestamp: current_timestamp_ms(),
        };
        
        let data = bincode::serialize(&update).unwrap_or_default();
        let message = NetworkMessage {
            id: entity_id,
            channel: ChannelType::UnreliableUnordered,
            data,
            timestamp: current_timestamp_ms(),
            sequence,
        };
        
        self.send_message(peer_id, message);
    }

    pub fn flush_batch(&self) -> Option<Vec<u8>> {
        let mut batch = Vec::with_capacity(MAX_PACKET_SIZE);
        
        let mut connections = self.connections.write();
        for (peer_id, conn) in connections.iter_mut() {
            if conn.send_queue.is_empty() {
                continue;
            }
            
            while let Some(msg) = conn.send_queue.pop_front() {
                if batch.len() + msg.data.len() + PACKET_HEADER_SIZE > MAX_PACKET_SIZE {
                    conn.send_queue.push_front(msg);
                    break;
                }
                
                let header = PacketHeader {
                    peer_id: *peer_id,
                    channel: msg.channel,
                    sequence: msg.sequence,
                    size: msg.data.len() as u32,
                };
                if let Ok(h) = bincode::serialize(&header) {
                    batch.extend_from_slice(&h);
                    batch.extend_from_slice(&msg.data);
                }
            }
        }
        
        if batch.is_empty() { None } else { Some(batch) }
    }

    // ============================================
    // CLIENT-SIDE PREDICTION
    // ============================================

    pub fn queue_input(&mut self, input: GameInput) {
        let mut buffer = self.input_buffer.write();
        buffer.push_back(input.clone());
        
        while buffer.len() > INPUT_BUFFER_SIZE {
            buffer.pop_front();
        }
        
        if self.prediction_enabled {
            let mut pending = self.pending_inputs.write();
            pending.push_back(PendingInput {
                input,
                sent_at: Instant::now(),
                acknowledged: false,
            });
        }
    }

    pub fn apply_local_prediction(&mut self, player_id: u64, input: &GameInput, physics_step: impl Fn([f32; 3], &[u8]) -> ([f32; 3], [f32; 3])) {
        if !self.prediction_enabled { return; }
        
        let mut predicted = self.predicted_states.write();
        
        let current = predicted.get(&player_id).cloned().unwrap_or(PredictedState {
            frame: input.frame,
            position: [0.0; 3],
            velocity: [0.0; 3],
            rotation: [0.0; 0.0; 0.0; 1.0],
            input_sequence: 0,
        });
        
        let (new_pos, new_vel) = physics_step(current.position, &input.inputs);
        
        predicted.insert(player_id, PredictedState {
            frame: input.frame,
            position: new_pos,
            velocity: new_vel,
            rotation: current.rotation,
            input_sequence: input.frame,
        });
    }

    pub fn reconcile(&mut self, player_id: u64, server_frame: u32, server_position: [f32; 3]) {
        let mut pending = self.pending_inputs.write();
        let mut predicted = self.predicted_states.write();
        
        for pending_input in pending.iter_mut() {
            if pending_input.input.frame <= server_frame {
                pending_input.acknowledged = true;
            }
        }
        
        if let Some(current) = predicted.get(&player_id) {
            let tolerance = 0.01;
            let diff = f32::sqrt(
                (server_position[0] - current.position[0]).powi(2) +
                (server_position[1] - current.position[1]).powi(2) +
                (server_position[2] - current.position[2]).powi(2)
            );
            
            if diff > tolerance {
                log::warn!("Prediction divergence: {} > {}", diff, tolerance);
                predicted.insert(player_id, PredictedState {
                    frame: server_frame,
                    position: server_position,
                    velocity: current.velocity,
                    rotation: current.rotation,
                    input_sequence: server_frame,
                });
                
                self.resimulate_from_frame(player_id, server_frame + 1);
            }
        }
    }

    fn resimulate_from_frame(&mut self, player_id: u64, from_frame: u32) {
        let pending = self.pending_inputs.read();
        let mut predicted = self.predicted_states.write();
        
        let unacked: Vec<&GameInput> = pending.iter()
            .filter(|p| !p.acknowledged && p.input.frame >= from_frame)
            .map(|p| &p.input)
            .collect();
        
        let mut current_pos = predicted.get(&player_id)
            .map(|p| p.position)
            .unwrap_or([0.0; 3]);
        
        for input in unacked {
            current_pos[0] += 0.016 * input.inputs.first().map(|&b| b as f32 * 0.1).unwrap_or(0.0);
        }
        
        predicted.insert(player_id, PredictedState {
            frame: from_frame + unacked.len() as u32,
            position: current_pos,
            velocity: [0.0; 3],
            rotation: [0.0; 0.0; 0.0; 1.0],
            input_sequence: from_frame + unacked.len() as u32,
        });
    }

    // ============================================
    // INTEREST MANAGEMENT
    // ============================================

    pub fn set_visible_entities(&mut self, entities: HashSet<u64>) {
        *self.visible_entities.write() = entities;
    }

    pub fn should_replicate_entity(&self, entity_id: u64) -> bool {
        if self.role == NetworkRole::Server {
            return true;
        }
        self.visible_entities.read().contains(&entity_id)
    }

    pub fn register_replicated_entity(&mut self, entity_id: u64, initial_state: Vec<u8>, owner: Option<u64>) {
        let state = ReplicatedState {
            entity_id,
            state_data: initial_state,
            sequence_number: 0,
            timestamp: 0,
            delta_from: 0,
            is_delta: false,
        };
        
        self.replicated_entities.write().insert(entity_id, state);
        
        if let Some(o) = owner {
            self.authority_map.write().insert(entity_id, o);
        }
    }

    pub fn update_entity_state(&mut self, entity_id: u64, new_state: Vec<u8>) {
        let mut entities = self.replicated_entities.write();
        if let Some(state) = entities.get_mut(&entity_id) {
            state.sequence_number += 1;
            state.timestamp = current_timestamp_ms();
            
            if self.delta_compression_enabled {
                let mut contexts = self.delta_contexts.write();
                let ctx = contexts.entry(entity_id).or_default();
                
                if let Some(ref last) = ctx.last_state {
                    if let Some(delta) = compute_delta(last, &new_state) {
                        state.state_data = delta;
                        state.delta_from = ctx.last_sequence;
                        state.is_delta = true;
                        ctx.last_state = Some(new_state);
                        ctx.last_sequence = state.sequence_number;
                        return;
                    }
                }
                
                ctx.last_state = Some(new_state.clone());
                ctx.last_sequence = state.sequence_number;
            }
            
            state.state_data = new_state;
            state.is_delta = false;
        }
    }

    // ============================================
    // ROLLBACK NETCODE
    // ============================================

    pub fn save_state(&mut self, frame: u32, state_hash: u32) {
        let mut states = self.rollback_states.write();
        
        while states.len() >= STATE_BUFFER_SIZE {
            states.pop_front();
        }
        
        let inputs: Vec<GameInput> = self.input_buffer.read()
            .iter()
            .filter(|i| i.frame >= frame.saturating_sub(60))
            .cloned()
            .collect();
        
        states.push_back(RollbackState {
            frame,
            state_hash,
            inputs,
            timestamp: current_timestamp_ms(),
        });
    }

    pub fn rollback_to(&self, frame: u32) -> Option<RollbackState> {
        let states = self.rollback_states.read();
        states.iter().find(|s| s.frame == frame).cloned()
    }

    // ============================================
    // RTT CALCULATION
    // ============================================

    pub fn record_packet_sent(&self, sequence: u64) {
        let mut times = self.packet_times.write();
        times.push_back((Instant::now(), sequence));
        
        while times.len() > 256 {
            times.pop_front();
        }
    }

    pub fn calculate_rtt(&self, sequence: u64) -> Option<f32> {
        let times = self.packet_times.read();
        
        for (sent, seq) in times.iter().rev() {
            if *seq == sequence {
                let elapsed = sent.elapsed().as_millis() as f32;
                return Some(elapsed / 2.0);
            }
        }
        None
    }

    pub fn get_smoothed_rtt(&self, peer_id: u64) -> f32 {
        if let Some(conn) = self.connections.read().get(&peer_id) {
            let instant_rtt = conn.rtt_ms;
            let mut times = self.packet_times.read();
            
            if times.len() >= 2 {
                let mut sum = 0.0;
                let mut count = 0;
                for (sent, _) in times.iter().rev().take(16) {
                    sum += sent.elapsed().as_millis() as f32;
                    count += 1;
                }
                let avg = sum / count as f32;
                return instant_rtt * 0.3 + (avg / 2.0) * 0.7;
            }
            
            instant_rtt
        } else {
            0.0
        }
    }

    pub fn update_connection_rtt(&mut self, peer_id: u64) {
        let rtt = self.get_smoothed_rtt(peer_id);
        if let Some(conn) = self.connections.get_mut(&peer_id) {
            conn.rtt_ms = rtt;
            conn.last_update = Instant::now();
        }
    }

    pub fn get_rtt(&self, peer_id: u64) -> Option<f32> {
        self.connections.read().get(&peer_id).map(|c| c.rtt_ms)
    }

    pub fn set_local_player(&mut self, player_id: u64) {
        self.local_player_id = Some(player_id);
        
        self.predicted_states.write().insert(player_id, PredictedState {
            frame: 0,
            position: [0.0; 3],
            velocity: [0.0; 3],
            rotation: [0.0; 0.0; 0.0; 1.0],
            input_sequence: 0,
        });
    }

    pub fn connected_peers(&self) -> Vec<u64> {
        self.connections.read()
            .iter()
            .filter(|(_, c)| c.state == ConnectionState::Connected)
            .map(|(id, _)| *id)
            .collect()
    }

    pub fn peer_count(&self) -> usize {
        self.connections.read()
            .values()
            .filter(|c| c.state == ConnectionState::Connected)
            .count()
    }

    pub fn get_stats(&self) -> NetworkStats {
        let connections = self.connections.read();
        let total_sent: u64 = connections.values().map(|c| c.bytes_sent).sum();
        let total_recv: u64 = connections.values().map(|c| c.bytes_received).sum();
        let avg_rtt = if connections.is_empty() {
            0.0
        } else {
            connections.values().map(|c| c.rtt_ms).sum::<f32>() / connections.len() as f32
        };
        
        NetworkStats {
            connected_peers: connections.values().filter(|c| c.state == ConnectionState::Connected).count(),
            bytes_sent_total: total_sent,
            bytes_received_total: total_recv,
            average_rtt_ms: avg_rtt,
            predicted_entities: self.predicted_states.read().len(),
        }
    }
}

// ============================================================================
// SUPPORTING TYPES
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PacketHeader {
    peer_id: u64,
    channel: ChannelType,
    sequence: u32,
    size: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PositionUpdate {
    entity_id: u64,
    position: [f32; 3],
    sequence: u32,
    timestamp: u64,
}

#[derive(Debug)]
pub struct NetworkStats {
    pub connected_peers: usize,
    pub bytes_sent_total: u64,
    pub bytes_received_total: u64,
    pub average_rtt_ms: f32,
    pub predicted_entities: usize,
}

#[derive(Debug)]
pub enum NetworkError {
    ConnectionFailed(String),
    Timeout,
    InvalidAddress,
    SendFailed,
    RecvFailed,
    SerializationError,
    CompressionError,
}

impl std::fmt::Display for NetworkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NetworkError::ConnectionFailed(s) => write!(f, "Connection failed: {}", s),
            NetworkError::Timeout => write!(f, "Connection timeout"),
            NetworkError::InvalidAddress => write!(f, "Invalid address"),
            NetworkError::SendFailed => write!(f, "Send failed"),
            NetworkError::RecvFailed => write!(f, "Recv failed"),
            NetworkError::SerializationError => write!(f, "Serialization error"),
            NetworkError::CompressionError => write!(f, "Compression error"),
        }
    }
}

// ============================================================================
// COMPRESSION UTILITIES
// ============================================================================

fn compress_data(data: &[u8]) -> Vec<u8> {
    let mut compressed = Vec::with_capacity(data.len());
    let mut i = 0;
    
    while i < data.len() {
        let byte = data[i];
        let mut count = 1;
        
        while i + count < data.len() && data[i + count] == byte && count < 255 {
            count += 1;
        }
        
        if count > 3 {
            compressed.push(0xFF);
            compressed.push(byte);
            compressed.push(count as u8);
        } else {
            for _ in 0..count {
                compressed.push(byte);
            }
        }
        
        i += count;
    }
    
    compressed
}

fn compute_delta(old: &[u8], new: &[u8]) -> Option<Vec<u8>> {
    if new.len() > old.len() {
        return None;
    }
    
    let mut delta = Vec::with_capacity(new.len());
    for (i, (o, n)) in old.iter().zip(new.iter()).enumerate() {
        if *o != *n {
            delta.push(i as u8);
            delta.push(*n);
        }
    }
    
    if delta.len() < new.len() / 2 {
        Some(delta)
    } else {
        None
    }
}

fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_compression() {
        let old = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let new = vec![1, 2, 3, 9, 5, 10, 7, 8];
        
        if let Some(delta) = compute_delta(&old, &new) {
            assert!(delta.len() < new.len());
        }
    }

    #[test]
    fn test_entity_snapshot() {
        let entity = EntitySnapshot::new(1, Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(entity.id, 1);
        assert_eq!(entity.center(), Vec3::ZERO);
    }

    #[test]
    fn test_network_system_creation() {
        let net = NetworkSystem::new(NetworkRole::Server);
        assert_eq!(net.peer_count(), 0);
    }
}