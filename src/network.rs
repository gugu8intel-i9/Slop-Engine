// src/network.rs
// Client/server sync, replication, rollback netcode

use std::collections::{HashMap, VecDeque, HashSet};
use std::net::{SocketAddr, UdpSocket};
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Network role for this instance
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NetworkRole {
    Server,
    Client,
    Host, // Both server and client
}

/// Connection state
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConnectionState {
    Connecting,
    Connected,
    Disconnecting,
    Disconnected,
}

/// Network message with channel support
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkMessage {
    pub id: u64,
    pub channel: ChannelType,
    pub data: Vec<u8>,
    pub timestamp: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ChannelType {
    ReliableOrdered,
    ReliableUnordered,
    UnreliableOrdered,
    UnreliableUnordered,
}

/// Connection information
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
}

/// Replicated entity state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplicatedState {
    pub entity_id: u64,
    pub state_data: Vec<u8>,
    pub sequence_number: u32,
    pub timestamp: u64,
}

/// Input for rollback netcode
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameInput {
    pub player_id: u64,
    pub frame: u32,
    pub inputs: Vec<u8>,
    pub checksum: u32,
}

/// Rollback state for deterministic simulation
pub struct RollbackState {
    pub frame: u32,
    pub state_hash: u32,
    pub inputs: Vec<GameInput>,
}

/// Network system with replication and rollback support
pub struct NetworkSystem {
    role: NetworkRole,
    connections: RwLock<HashMap<u64, Connection>>,
    message_queue: RwLock<VecDeque<NetworkMessage>>,
    replicated_entities: RwLock<HashMap<u64, ReplicatedState>>,
    input_buffer: RwLock<VecDeque<GameInput>>,
    rollback_states: RwLock<Vec<RollbackState>>,
    local_player_id: Option<u64>,
    next_peer_id: RwLock<u64>,
    socket: Option<Arc<UdpSocket>>,
    compression_enabled: bool,
    encryption_enabled: bool,
}

impl NetworkSystem {
    pub fn new(role: NetworkRole) -> Self {
        Self {
            role,
            connections: RwLock::new(HashMap::new()),
            message_queue: RwLock::new(VecDeque::new()),
            replicated_entities: RwLock::new(HashMap::new()),
            input_buffer: RwLock::new(VecDeque::new()),
            rollback_states: RwLock::new(Vec::new()),
            local_player_id: None,
            next_peer_id: RwLock::new(1),
            socket: None,
            compression_enabled: true,
            encryption_enabled: false,
        }
    }

    pub fn connect(&self, _address: &str) -> Result<(), NetworkError> {
        // Implementation would create UDP/TCP connection
        Ok(())
    }

    pub fn disconnect(&self, peer_id: u64) {
        let mut connections = self.connections.write();
        if let Some(conn) = connections.get_mut(&peer_id) {
            conn.state = ConnectionState::Disconnecting;
        }
    }

    pub fn send_message(&self, peer_id: u64, message: NetworkMessage) {
        let mut connections = self.connections.write();
        if let Some(conn) = connections.get_mut(&peer_id) {
            conn.send_queue.push_back(message);
        }
    }

    pub fn broadcast(&self, message: NetworkMessage, exclude: Option<u64>) {
        let connections = self.connections.read();
        for (peer_id, _) in connections.iter() {
            if Some(*peer_id) != exclude {
                // Clone message for each recipient
            }
        }
    }

    /// Register entity for replication
    pub fn register_replicated_entity(&self, entity_id: u64, initial_state: Vec<u8>) {
        let state = ReplicatedState {
            entity_id,
            state_data: initial_state,
            sequence_number: 0,
            timestamp: 0,
        };
        self.replicated_entities.write().insert(entity_id, state);
    }

    /// Update replicated entity state
    pub fn update_entity_state(&self, entity_id: u64, new_state: Vec<u8>) {
        let mut entities = self.replicated_entities.write();
        if let Some(state) = entities.get_mut(&entity_id) {
            state.sequence_number += 1;
            state.state_data = new_state;
        }
    }

    /// Queue input for rollback netcode
    pub fn queue_input(&self, input: GameInput) {
        let mut buffer = self.input_buffer.write();
        buffer.push_back(input);
        
        // Keep only recent inputs
        while buffer.len() > 120 { // ~2 seconds at 60fps
            buffer.pop_front();
        }
    }

    /// Save state for potential rollback
    pub fn save_state(&self, frame: u32, state_hash: u32) {
        let mut states = self.rollback_states.write();
        states.push(RollbackState {
            frame,
            state_hash,
            inputs: Vec::new(),
        });
        
        // Limit stored states
        while states.len() > 120 {
            states.remove(0);
        }
    }

    /// Rollback to a specific frame
    pub fn rollback_to(&self, frame: u32) -> Option<RollbackState> {
        let states = self.rollback_states.read();
        states.iter().find(|s| s.frame == frame).cloned()
    }

    /// Re-simulate from a frame with corrected inputs
    pub fn resimulate(&self, from_frame: u32, corrected_inputs: &[GameInput]) {
        // Implementation would re-run simulation with corrected inputs
    }

    pub fn get_rtt(&self, peer_id: u64) -> Option<f32> {
        self.connections.read().get(&peer_id).map(|c| c.rtt_ms)
    }

    pub fn set_local_player(&mut self, player_id: u64) {
        self.local_player_id = Some(player_id);
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
}

/// Network error types
#[derive(Debug)]
pub enum NetworkError {
    ConnectionFailed(String),
    Timeout,
    InvalidAddress,
    SendFailed,
    RecvFailed,
    SerializationError,
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_system_creation() {
        let net = NetworkSystem::new(NetworkRole::Server);
        assert_eq!(net.peer_count(), 0);
    }

    #[test]
    fn test_replicated_entity() {
        let net = NetworkSystem::new(NetworkRole::Server);
        net.register_replicated_entity(1, vec![1, 2, 3]);
        net.update_entity_state(1, vec![4, 5, 6]);
        
        let entities = net.replicated_entities.read();
        let state = entities.get(&1).unwrap();
        assert_eq!(state.sequence_number, 1);
    }
}
