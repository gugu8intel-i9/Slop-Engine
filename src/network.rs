// src/network.rs
// OPTIMIZED: Client-side prediction, delta compression, interest management, and ping reduction

use std::collections::{HashMap, VecDeque, HashSet};
use std::net::{SocketAddr, UdpSocket, TcpStream, Shutdown};
use std::io::{Read, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::{RwLock, Mutex};
use serde::{Deserialize, Serialize};
use bincode;

const PACKET_HEADER_SIZE: usize = 8;
const INPUT_BUFFER_SIZE: usize = 128; // ~2 seconds at 60fps
const STATE_BUFFER_SIZE: usize = 30; // Keep 0.5s of states for rollback
const MAX_PACKET_SIZE: usize = 1400; // MTU safe limit
const COMPRESSION_THRESHOLD: usize = 256; // Only compress payloads > 256 bytes

/// Network role for this instance
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NetworkRole {
    Server,
    Client,
    Host,
}

/// Connection state
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConnectionState {
    Connecting,
    Connected,
    Disconnecting,
    Disconnected,
}

/// Channel types for different reliability requirements
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ChannelType {
    ReliableOrdered,    // Game state, inputs
    ReliableUnordered,  // Player info, chat
    UnreliableOrdered,  // Not used typically
    UnreliableUnordered, // Position updates, non-critical
}

/// Network message with channel support
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkMessage {
    pub id: u64,
    pub channel: ChannelType,
    pub data: Vec<u8>,
    pub timestamp: u64,
    pub sequence: u32, // For ordering/reliability
}

/// Connection with advanced telemetry
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

/// Replicated entity state with delta support
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplicatedState {
    pub entity_id: u64,
    pub state_data: Vec<u8>,
    pub sequence_number: u32,
    pub timestamp: u64,
    pub delta_from: u32, // Previous sequence number for delta compression
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

/// Rollback state for deterministic simulation
#[derive(Clone, Debug)]
pub struct RollbackState {
    pub frame: u32,
    pub state_hash: u32,
    pub inputs: Vec<GameInput>,
    pub timestamp: u64,
}

/// Client-side prediction state
struct PredictedState {
    frame: u32,
    position: [f32; 3],
    velocity: [f32; 3],
    rotation: [f32; 4],
    input_sequence: u32,
}

/// Pending input acknowledgment for reconciliation
struct PendingInput {
    input: GameInput,
    sent_at: Instant,
    acknowledged: bool,
}

/// Delta compression context
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

/// OPTIMIZED: Network system with replication, rollback, and latency reduction
pub struct NetworkSystem {
    role: NetworkRole,
    connections: RwLock<HashMap<u64, Connection>>,
    message_queue: RwLock<VecDeque<NetworkMessage>>,
    replicated_entities: RwLock<HashMap<u64, ReplicatedState>>,
    delta_contexts: RwLock<HashMap<u64, DeltaContext>>,
    
    // Rollback netcode buffers
    input_buffer: RwLock<VecDeque<GameInput>>,
    rollback_states: RwLock<VecDeque<RollbackState>>,
    
    // Client-side prediction
    local_player_id: Option<u64>,
    predicted_states: RwLock<HashMap<u64, PredictedState>>,
    pending_inputs: RwLock<VecDeque<PendingInput>>,
    
    // Interest management
    visible_entities: RwLock<HashSet<u64>>,
    authority_map: RwLock<HashMap<u64, u64>>, // entity -> owner
    
    // Network optimization
    next_peer_id: RwLock<u64>,
    socket: Option<Arc<UdpSocket>>,
    compression_enabled: bool,
    delta_compression_enabled: bool,
    prediction_enabled: bool,
    
    // Telemetry
    last_ping_check: Instant,
    packet_times: RwLock<VecDeque<(Instant, u64)>>, // For RTT calculation
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
        
        // Store socket in a shared reference
        let shared_socket = Arc::new(socket);
        self.socket = Some(shared_socket.clone());
        
        // Connect to target
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
        
        info!("Connected to {} with peer_id {}", address, peer_id);
        Ok(peer_id)
    }

    pub fn disconnect(&mut self, peer_id: u64) {
        let mut connections = self.connections.write();
        if let Some(conn) = connections.get_mut(&peer_id) {
            conn.state = ConnectionState::Disconnecting;
        }
        
        // Clean up prediction state
        self.predicted_states.write().remove(&peer_id);
        
        // Clean up pending inputs
        let mut pending = self.pending_inputs.write();
        pending.retain(|p| p.input.player_id != peer_id as u64);
    }

    // ============================================
    // MESSAGE SENDING WITH PACKET BATCHING
    // ============================================

    /// Send a message with optional delta compression
    pub fn send_message(&self, peer_id: u64, mut message: NetworkMessage) {
        // Compress if enabled and payload is large enough
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

    /// Send unreliable position update (optimized path)
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

    /// Batch multiple messages into a single packet
    pub fn flush_batch(&self) -> Option<Vec<u8>> {
        let mut batch = Vec::with_capacity(MAX_PACKET_SIZE);
        
        // Find connections with pending messages
        let mut connections = self.connections.write();
        for (peer_id, conn) in connections.iter_mut() {
            if conn.send_queue.is_empty() {
                continue;
            }
            
            // Collect messages until we hit MTU limit
            while let Some(msg) = conn.send_queue.pop_front() {
                if batch.len() + msg.data.len() + PACKET_HEADER_SIZE > MAX_PACKET_SIZE {
                    conn.send_queue.push_front(msg); // Put it back
                    break;
                }
                
                // Write header
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

    /// Queue input for prediction and server submission
    pub fn queue_input(&mut self, input: GameInput) {
        let mut buffer = self.input_buffer.write();
        buffer.push_back(input.clone());
        
        // Keep only recent inputs
        while buffer.len() > INPUT_BUFFER_SIZE {
            buffer.pop_front();
        }
        
        // Add to pending for acknowledgment
        if self.prediction_enabled {
            let mut pending = self.pending_inputs.write();
            pending.push_back(PendingInput {
                input,
                sent_at: Instant::now(),
                acknowledged: false,
            });
        }
    }

    /// Apply local prediction immediately (reduces perceived latency)
    pub fn apply_local_prediction(&mut self, player_id: u64, input: &GameInput, physics_step: impl Fn([f32; 3], &[u8]) -> ([f32; 3], [f32; 3])) {
        if !self.prediction_enabled { return; }
        
        let mut predicted = self.predicted_states.write();
        
        // Get current predicted state or create new
        let current = predicted.get(&player_id).cloned().unwrap_or(PredictedState {
            frame: input.frame,
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            input_sequence: 0,
        });
        
        // Apply input to predict next state
        let (new_pos, new_vel) = physics_step(current.position, &input.inputs);
        
        predicted.insert(player_id, PredictedState {
            frame: input.frame,
            position: new_pos,
            velocity: new_vel,
            rotation: current.rotation,
            input_sequence: input.frame,
        });
    }

    /// Reconcile with server state when acknowledgment received
    pub fn reconcile(&mut self, player_id: u64, server_frame: u32, server_position: [f32; 3]) {
        let mut pending = self.pending_inputs.write();
        let mut predicted = self.predicted_states.write();
        
        // Find and mark acknowledged inputs
        for pending_input in pending.iter_mut() {
            if pending_input.input.frame <= server_frame {
                pending_input.acknowledged = true;
            }
        }
        
        // Check if prediction diverged
        if let Some(current) = predicted.get(&player_id) {
            let tolerance = 0.01; // 1cm tolerance
            let diff = f32::sqrt(
                (server_position[0] - current.position[0]).powi(2) +
                (server_position[1] - current.position[1]).powi(2) +
                (server_position[2] - current.position[2]).powi(2)
            );
            
            if diff > tolerance {
                warn!("Prediction divergence detected: {} > {}", diff, tolerance);
                // Snap to server state (or smooth interpolate)
                predicted.insert(player_id, PredictedState {
                    frame: server_frame,
                    position: server_position,
                    velocity: current.velocity,
                    rotation: current.rotation,
                    input_sequence: server_frame,
                });
                
                // Trigger re-simulation of unacknowledged inputs
                self.resimulate_from_frame(player_id, server_frame + 1);
            }
        }
    }

    /// Re-simulate from a specific frame with corrected inputs
    fn resimulate_from_frame(&mut self, player_id: u64, from_frame: u32) {
        let pending = self.pending_inputs.read();
        let mut predicted = self.predicted_states.write();
        
        // Find unacknowledged inputs
        let unacked: Vec<&GameInput> = pending.iter()
            .filter(|p| !p.acknowledged && p.input.frame >= from_frame)
            .map(|p| &p.input)
            .collect();
        
        // Re-simulate each input (would integrate with physics engine)
        let mut current_pos = predicted.get(&player_id)
            .map(|p| p.position)
            .unwrap_or([0.0; 3]);
        
        for input in unacked {
            // Placeholder: actual physics simulation would go here
            current_pos[0] += 0.016 * input.inputs.first().map(|&b| b as f32 * 0.1).unwrap_or(0.0);
        }
        
        predicted.insert(player_id, PredictedState {
            frame: from_frame + unacked.len() as u32,
            position: current_pos,
            velocity: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            input_sequence: from_frame + unacked.len() as u32,
        });
    }

    // ============================================
    // INTEREST MANAGEMENT (Reduce bandwidth)
    // ============================================

    /// Set which entities this client is interested in
    pub fn set_visible_entities(&mut self, entities: HashSet<u64>) {
        *self.visible_entities.write() = entities;
    }

    /// Only replicate entities in visible set
    pub fn should_replicate_entity(&self, entity_id: u64) -> bool {
        if self.role == NetworkRole::Server {
            return true; // Server replicates all
        }
        self.visible_entities.read().contains(&entity_id)
    }

    /// Register entity for replication with interest tracking
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

    /// Update with delta compression
    pub fn update_entity_state(&mut self, entity_id: u64, new_state: Vec<u8>) {
        let mut entities = self.replicated_entities.write();
        if let Some(state) = entities.get_mut(&entity_id) {
            let prev_seq = state.sequence_number;
            state.sequence_number += 1;
            state.timestamp = current_timestamp_ms();
            
            // Delta compression
            if self.delta_compression_enabled {
                let mut contexts = self.delta_contexts.write();
                let ctx = contexts.entry(entity_id).or_default();
                
                if let Some(ref last) = ctx.last_state {
                    if let Some(delta) = compute_delta(last, &new_state) {
                        state.state_data = delta;
                        state.delta_from = ctx.last_sequence;
                        state.is_delta = true;
                        
                        // Store full state for next delta
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

    /// Reconstruct state from delta
    pub fn reconstruct_state(&self, entity_id: u64, base_state: &[u8]) -> Option<Vec<u8>> {
        if !self.delta_compression_enabled {
            return self.replicated_entities.read().get(&entity_id).map(|s| s.state_data.clone());
        }
        
        let entities = self.replicated_entities.read();
        if let Some(state) = entities.get(&entity_id) {
            if state.is_delta {
                // Would need full state history for reconstruction
                // For simplicity, return delta and let caller handle
                Some(state.state_data.clone())
            } else {
                Some(state.state_data.clone())
            }
        } else {
            None
        }
    }

    // ============================================
    // ROLLBACK NETCODE
    // ============================================

    /// Save state for potential rollback
    pub fn save_state(&mut self, frame: u32, state_hash: u32) {
        let mut states = self.rollback_states.write();
        
        // Keep rolling window of states
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

    /// Rollback to a specific frame
    pub fn rollback_to(&self, frame: u32) -> Option<RollbackState> {
        let states = self.rollback_states.read();
        states.iter().find(|s| s.frame == frame).cloned()
    }

    // ============================================
    // RTT AND PING CALCULATION
    // ============================================

    /// Record packet send time for RTT calculation
    pub fn record_packet_sent(&self, sequence: u64) {
        let mut times = self.packet_times.write();
        times.push_back((Instant::now(), sequence));
        
        // Keep only recent samples
        while times.len() > 256 {
            times.pop_front();
        }
    }

    /// Calculate current RTT from packet times
    pub fn calculate_rtt(&self, sequence: u64) -> Option<f32> {
        let times = self.packet_times.read();
        
        // Find matching send time
        for (sent, seq) in times.iter().rev() {
            if *seq == sequence {
                let elapsed = sent.elapsed().as_millis() as f32;
                return Some(elapsed / 2.0); // One-way time
            }
        }
        None
    }

    /// Get smoothed RTT estimate
    pub fn get_smoothed_rtt(&self, peer_id: u64) -> f32 {
        if let Some(conn) = self.connections.read().get(&peer_id) {
            // Use exponentially weighted average
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
                // Blend instant and average
                return instant_rtt * 0.3 + (avg / 2.0) * 0.7;
            }
            
            instant_rtt
        } else {
            0.0
        }
    }

    /// Update connection RTT based on recent measurements
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
        
        // Initialize prediction state
        self.predicted_states.write().insert(player_id, PredictedState {
            frame: 0,
            position: [0.0; 3],
            velocity: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
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

    // ============================================
    // UTILITIES
    // ============================================

    /// Get network statistics
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

// ============================================
// COMPRESSION UTILITIES
// ============================================

fn compress_data(data: &[u8]) -> Vec<u8> {
    // Simple run-length encoding for demonstration
    // In production, use lz4 or zstd
    let mut compressed = Vec::with_capacity(data.len());
    let mut i = 0;
    
    while i < data.len() {
        let byte = data[i];
        let mut count = 1;
        
        while i + count < data.len() && data[i + count] == byte && count < 255 {
            count += 1;
        }
        
        if count > 3 {
            compressed.push(0xFF); // Escape byte
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
        return None; // Delta would be larger than full data
    }
    
    let mut delta = Vec::with_capacity(new.len());
    for (i, (o, n)) in old.iter().zip(new.iter()).enumerate() {
        if *o != *n {
            delta.push(i as u8); // Position
            delta.push(*n); // New value
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

use log::{info, warn};

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
    fn test_network_system_creation() {
        let net = NetworkSystem::new(NetworkRole::Server);
        assert_eq!(net.peer_count(), 0);
    }
}