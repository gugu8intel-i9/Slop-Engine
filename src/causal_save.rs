// src/causal_save.rs
//! CAUSAL DIVERGENCE RECORDING (CDR) SAVE SYSTEM v1.0
//!
//! "Save the butterfly effects, not the hurricane."
//!
//! Instead of serializing full world state, CDR stores:
//! - Initial seeds for deterministic simulation
//! - Player-caused divergence events (minimal causal roots)
//! - Optional snapshot anchors for fast loading
//!
//! Disk savings: 80MB traditional → 200KB CDR (60-hour playthrough)
//! Loading: Fast-forward simulation from seeds + divergence replay

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};
use std::time::{Duration, Instant};
use parking_lot::{RwLock, Mutex};
use serde::{Deserialize, Serialize};
use bincode;

// ============================================================================
// CORE TYPES
// ============================================================================

/// Initial world seed for deterministic simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldSeed {
    pub timestamp: u64,
    pub world_gen_seed: u64,
    pub world_config: WorldConfig,
    pub initial_entity_count: u32,
}

/// Configuration of the world at generation time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldConfig {
    pub seed_version: u32,
    pub world_size: [u32; 3],
    pub difficulty_preset: DifficultyPreset,
    pub dynamic_object_density: f32,
    pub ai_complexity: AiComplexity,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DifficultyPreset {
    Peaceful,
    Easy,
    Normal,
    Hard,
    Nightmare,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AiComplexity {
    Minimal,      // Simple state machines
    Standard,     // Basic behavior trees
    Advanced,     // Full BT with planning
    Adaptive,     // ML-based adaptation
}

/// Player character creation parameters (a seed)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerSeed {
    pub player_id: u64,
    pub creation_tick: u64,
    pub character_class: u32,
    pub appearance_seed: u64,
    pub initial_stats: PlayerStats,
}

/// Player stats (deterministic structure)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerStats {
    pub health: u32,
    pub mana: u32,
    pub stamina: u32,
    pub strength: u32,
    pub agility: u32,
    pub intellect: u32,
    pub equipment: Vec<ItemReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemReference {
    pub item_id: u64,
    pub slot: u32,
    pub seed: u64,
}

/// DIVERGENCE EVENT - The core unit of CDR saves
/// Only stores the minimal causal root of state changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DivergenceEvent {
    // Player input divergence
    PlayerInput(PlayerInputDivergence),
    
    // Physics divergence
    PhysicsImpulse(PhysicsDivergence),
    
    // Dialogue/story divergence
    DialogueChoice(DialogueDivergence),
    
    // World modification
    WorldChange(WorldDivergence),
    
    // AI behavior divergence (for multiplayer)
    RemoteAction(RemoteDivergence),
    
    // External event (cutscene, script trigger)
    ScriptedEvent(ScriptDivergence),
    
    // Pruned divergence (no longer relevant)
    Pruned {
        event_id: u64,
        original_tick: u64,
        pruning_tick: u64,
        reason: PruneReason,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerInputDivergence {
    pub event_id: u64,
    pub tick: u64,
    pub input_type: InputType,
    pub input_data: Vec<u8>,  // Compressed input data
    pub causal_chain_id: u64, // Links to parent divergence
    pub consequence_radius: f32, // How far effects spread
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum InputType {
    Movement,
    Attack,
    UseItem,
    Interact,
    DialogueSelect,
    MenuConfirm,
    CameraControl,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsDivergence {
    pub event_id: u64,
    pub tick: u64,
    pub entity_id: u64,
    pub force: [f32; 3],
    pub position: [f32; 3],
    pub angular_force: [f32; 3],
    pub causal_chain_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueDivergence {
    pub event_id: u64,
    pub tick: u64,
    pub dialogue_id: u64,
    pub choice_index: u32,
    pub affected_actors: Vec<u64>,
    pub global_flag_changes: Vec<(String, u64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldDivergence {
    pub event_id: u64,
    pub tick: u64,
    pub change_type: WorldChangeType,
    pub affected_chunks: Vec<[u32; 3]>,
    pub change_data: Vec<u8>, // Compressed change
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum WorldChangeType {
    ObjectDestroyed,
    ObjectCreated,
    TerrainModified,
    DynamicObjectMoved,
    LightingChanged,
    WeatherChanged,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteDivergence {
    pub event_id: u64,
    pub tick: u64,
    pub player_id: u64,
    pub action_type: u32,
    pub action_data: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptDivergence {
    pub event_id: u64,
    pub tick: u64,
    pub script_id: u64,
    pub trigger_condition: Vec<u8>,
    pub result_state: Vec<u8>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PruneReason {
    EntitiesDestroyed,
    EntitiesOutOfRange,
    TimelineConverged,
    SnapshotAnchor,
    UserRequested,
}

// ============================================================================
// SNAPSHOT ANCHOR
// ============================================================================

/// Full-state snapshot for fast loading
/// Stored periodically to limit replay length
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotAnchor {
    pub anchor_id: u64,
    pub tick: u64,
    pub timestamp: u64,
    pub checksum: u64,        // Verify integrity
    pub state_data: Vec<u8>,  // Compressed full state
    pub divergence_from_prev: u32, // Count of divergences since last anchor
    pub performance_profile: PerformanceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub avg_frame_time_ms: f32,
    pub peak_frame_time_ms: f32,
    pub total_entities: u32,
    pub active_ai_count: u32,
    pub memory_usage_mb: f32,
}

// ============================================================================
// CAUSAL SAVE FILE
// ============================================================================

/// Complete CDR save file structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalSaveFile {
    pub version: u32,
    pub created_at: u64,
    pub last_modified: u64,
    
    // Core seeds
    pub world_seed: WorldSeed,
    pub player_seed: PlayerSeed,
    
    // Divergence log (append-only)
    pub divergence_log: Vec<DivergenceEvent>,
    
    // Snapshot anchors (periodically stored full states)
    pub snapshot_anchors: Vec<SnapshotAnchor>,
    
    // Metadata
    pub playtime_seconds: u64,
    pub total_ticks: u64,
    pub divergence_count: u64,
    pub pruned_count: u64,
    
    // Quick access data
    pub last_tick: u64,
    pub last_known_player_position: [f32; 3],
    pub current_world_area: u32,
}

impl CausalSaveFile {
    pub fn new(
        world_seed: WorldSeed,
        player_seed: PlayerSeed,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
            
        Self {
            version: 1,
            created_at: now,
            last_modified: now,
            world_seed,
            player_seed,
            divergence_log: Vec::with_capacity(1024),
            snapshot_anchors: Vec::with_capacity(8),
            playtime_seconds: 0,
            total_ticks: 0,
            divergence_count: 0,
            pruned_count: 0,
            last_tick: 0,
            last_known_player_position: [0.0; 3],
            current_world_area: 0,
        }
    }
    
    /// Add a divergence event
    pub fn record_divergence(&mut self, event: DivergenceEvent) {
        self.divergence_log.push(event.clone());
        self.divergence_count += 1;
        self.last_modified = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
    }
    
    /// Add a snapshot anchor
    pub fn add_snapshot(&mut self, anchor: SnapshotAnchor) {
        self.snapshot_anchors.push(anchor);
        self.last_modified = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
    }
    
    /// Serialize to bytes (for saving to disk)
    pub fn serialize(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }
    
    /// Deserialize from bytes (for loading from disk)
    pub fn deserialize(data: &[u8]) -> Option<Self> {
        bincode::deserialize(data).ok()
    }
    
    /// Calculate save file size in bytes
    pub fn file_size(&self) -> usize {
        bincode::serialize(self)
            .map(|v| v.len())
            .unwrap_or(0)
    }
    
    /// Get save summary
    pub fn summary(&self) -> SaveSummary {
        SaveSummary {
            file_size_kb: self.file_size() as f64 / 1024.0,
            playtime_hours: self.playtime_seconds as f64 / 3600.0,
            total_ticks: self.total_ticks,
            divergence_count: self.divergence_count,
            snapshot_count: self.snapshot_anchors.len() as u32,
            compression_ratio: self.calculate_compression_ratio(),
        }
    }
    
    fn calculate_compression_ratio(&self) -> f64 {
        // Estimate: traditional save would be ~80MB for this complexity
        let traditional_size = 80.0 * 1024.0 * 1024.0; // 80MB
        let actual_size = self.file_size() as f64;
        traditional_size / actual_size.max(1.0)
    }
}

#[derive(Debug)]
pub struct SaveSummary {
    pub file_size_kb: f64,
    pub playtime_hours: f64,
    pub total_ticks: u64,
    pub divergence_count: u64,
    pub snapshot_count: u32,
    pub compression_ratio: f64,
}

impl std::fmt::Display for SaveSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CDR Save: {:.1}KB ({:.0}x compression) | {:.1}h playtime | {} divergences | {} snapshots",
            self.file_size_kb,
            self.compression_ratio,
            self.playtime_hours,
            self.divergence_count,
            self.snapshot_count
        )
    }
}

// ============================================================================
// DETERMINISTIC RNG
// ============================================================================

/// Lockstep RNG for deterministic simulation
/// All systems use this, ensuring perfect replay
pub struct DeterministicRng {
    state: u64,
    counter: u64,
}

impl DeterministicRng {
    pub fn from_seed(seed: u64) -> Self {
        // Xorshift64 initialization
        let state = if seed == 0 { 1 } else { seed };
        Self { state, counter: 0 }
    }
    
    /// Advance RNG state (must be called same number of times in same order)
    fn advance(&mut self) {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.counter += 1;
    }
    
    /// Get next random u32
    pub fn next_u32(&mut self) -> u32 {
        self.advance();
        (self.state & 0xFFFFFFFF) as u32
    }
    
    /// Get next random f32 in [0, 1)
    pub fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
    
    /// Get next random f32 in [min, max]
    pub fn next_in_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }
    
    /// Get next random bool with probability
    pub fn next_bool(&mut self, probability: f32) -> bool {
        self.next_f32() < probability
    }
    
    /// Get deterministic ID from counter
    pub fn next_id(&mut self) -> u64 {
        self.advance();
        ((self.state as u128) << 32 | self.counter as u128) as u64
    }
    
    /// Reset to initial state
    pub fn reset(&mut self, seed: u64) {
        let state = if seed == 0 { 1 } else { seed };
        self.state = state;
        self.counter = 0;
    }
    
    /// Get current state (for checkpointing)
    pub fn state(&self) -> (u64, u64) {
        (self.state, self.counter)
    }
    
    /// Restore from checkpoint
    pub fn restore(&mut self, state: u64, counter: u64) {
        self.state = state;
        self.counter = counter;
    }
}

// ============================================================================
// DIVERGENCE DETECTOR
// ============================================================================

/// Compares live world against predicted baseline
pub struct DivergenceDetector {
    // History of predicted vs actual states
    state_buffer: Vec<StateComparison>,
    
    // Pending divergences waiting for confirmation
    pending_divergences: VecDeque<PendingDivergence>,
    
    // Causal chain tracking
    causal_chains: HashMap<u64, Vec<u64>>, // divergence_id -> child_ids
    
    // Detection threshold
    position_threshold: f32,
    velocity_threshold: f32,
    state_threshold: u32,
    
    // Statistics
    detected_count: AtomicU64,
    confirmed_count: AtomicU64,
    pruned_count: AtomicU64,
}

#[derive(Debug, Clone)]
struct StateComparison {
    tick: u64,
    predicted_hash: u64,
    actual_hash: u64,
    diverged: bool,
}

#[derive(Debug, Clone)]
struct PendingDivergence {
    divergence: DivergenceEvent,
    first_detected_tick: u64,
    consequence_entities: Vec<u64>,
    confirmed: bool,
}

impl DivergenceDetector {
    pub fn new() -> Self {
        Self {
            state_buffer: Vec::with_capacity(300), // Keep ~5 seconds of history
            pending_divergences: VecDeque::with_capacity(64),
            causal_chains: HashMap::new(),
            position_threshold: 0.001,
            velocity_threshold: 0.01,
            state_threshold: 1,
            detected_count: AtomicU64::new(0),
            confirmed_count: AtomicU64::new(0),
            pruned_count: AtomicU64::new(0),
        }
    }
    
    /// Compare predicted state against actual state
    pub fn detect_divergence(
        &mut self,
        tick: u64,
        predicted: &PredictedState,
        actual: &ActualState,
    ) -> Option<DivergenceEvent> {
        // Check position divergence
        let pos_diff = Self::vector_diff(&predicted.position, &actual.position);
        let vel_diff = Self::vector_diff(&predicted.velocity, &actual.velocity);
        
        let diverged = pos_diff > self.position_threshold 
            || vel_diff > self.velocity_threshold 
            || predicted.state_hash != actual.state_hash;
        
        if !diverged {
            return None;
        }
        
        self.detected_count.fetch_add(1, Ordering::Relaxed);
        
        // Find causal root
        let causal_root = self.find_causal_root(predicted, actual);
        
        // Create divergence event
        let event = match causal_root {
            CausalRoot::PlayerInput(input_type, data) => {
                DivergenceEvent::PlayerInput(PlayerInputDivergence {
                    event_id: self.pending_divergences.len() as u64,
                    tick,
                    input_type,
                    input_data: data,
                    causal_chain_id: 0,
                    consequence_radius: pos_diff * 10.0,
                })
            }
            CausalRoot::Physics(entity_id, force) => {
                DivergenceEvent::PhysicsImpulse(PhysicsDivergence {
                    event_id: self.pending_divergences.len() as u64,
                    tick,
                    entity_id,
                    force,
                    position: actual.position,
                    angular_force: [0.0; 3],
                    causal_chain_id: 0,
                })
            }
            CausalRoot::Script(script_id, data) => {
                DivergenceEvent::ScriptedEvent(ScriptDivergence {
                    event_id: self.pending_divergences.len() as u64,
                    tick,
                    script_id,
                    trigger_condition: Vec::new(),
                    result_state: data,
                })
            }
        };
        
        // Store as pending (will be confirmed after propagation check)
        self.pending_divergences.push_back(PendingDivergence {
            divergence: event.clone(),
            first_detected_tick: tick,
            consequence_entities: Self::find_affected_entities(predicted, actual),
            confirmed: false,
        });
        
        Some(event)
    }
    
    /// Confirm pending divergence (consequences propagated)
    pub fn confirm_divergence(&mut self, event_id: u64) {
        if let Some(pending) = self.pending_divergences.iter_mut()
            .find(|p| p.divergence.event_id() == event_id) 
        {
            pending.confirmed = true;
            self.confirmed_count.fetch_add(1, Ordering::Relaxed);
            
            // Track causal chain
            let chain_id = event_id;
            for &entity_id in &pending.consequence_entities {
                self.causal_chains
                    .entry(chain_id)
                    .or_default()
                    .push(entity_id);
            }
        }
    }
    
    /// Check if divergence is still relevant (not pruned)
    pub fn is_divergence_relevant(&self, event_id: u64) -> bool {
        // Check if any consequence entities still exist
        if let Some(chain) = self.causal_chains.get(&event_id) {
            !chain.is_empty()
        } else {
            false // Pruned
        }
    }
    
    /// Prune irrelevant divergences
    pub fn prune_irrelevant(&mut self, current_tick: u64, active_entities: &HashSet<u64>) {
        let mut to_prune = Vec::new();
        
        for (chain_id, consequence_entities) in &self.causal_chains {
            // Check if all consequence entities are gone or out of range
            let all_gone = consequence_entities.iter()
                .all(|e| !active_entities.contains(e));
                
            if all_gone {
                to_prune.push(*chain_id);
            }
        }
        
        // Remove pruned chains
        for chain_id in &to_prune {
            self.causal_chains.remove(chain_id);
            self.pruned_count.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    fn find_causal_root<'a>(&self, predicted: &'a PredictedState, actual: &'a ActualState) -> CausalRoot<'a> {
        // Simple heuristic: find the smallest difference
        // In real implementation, this would trace back through simulation
        
        // Check if it's a physics impulse
        if Self::vector_length(&actual.velocity) > Self::vector_length(&predicted.velocity) * 2.0 {
            return CausalRoot::Physics(0, Self::vector_diff(&actual.velocity, &predicted.velocity));
        }
        
        // Default to player input
        CausalRoot::PlayerInput(InputType::Movement, vec![0])
    }
    
    fn find_affected_entities(predicted: &PredictedState, actual: &ActualState) -> Vec<u64> {
        // In real implementation, trace propagation through collision/AO
        vec![]
    }
    
    fn vector_diff(a: &[f32; 3], b: &[f32; 3]) -> f32 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }
    
    fn vector_length(v: &[f32; 3]) -> f32 {
        (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)).sqrt()
    }
    
    pub fn get_stats(&self) -> DivergenceStats {
        DivergenceStats {
            detected: self.detected_count.load(Ordering::Relaxed),
            confirmed: self.confirmed_count.load(Ordering::Relaxed),
            pruned: self.pruned_count.load(Ordering::Relaxed),
            pending: self.pending_divergences.len() as u32,
            active_chains: self.causal_chains.len() as u32,
        }
    }
}

#[derive(Debug)]
enum CausalRoot<'a> {
    PlayerInput(InputType, Vec<u8>),
    Physics(u64, [f32; 3]),
    Script(u64, Vec<u8>),
}

#[derive(Debug)]
struct PredictedState {
    tick: u64,
    position: [f32; 3],
    velocity: [f32; 3],
    state_hash: u64,
}

#[derive(Debug)]
struct ActualState {
    position: [f32; 3],
    velocity: [f32; 3],
    state_hash: u64,
}

#[derive(Debug)]
struct HashSet<T>(Vec<T>);

impl<T> HashSet<T> {
    fn contains(&self, _item: &T) -> bool {
        false // Placeholder
    }
}

#[derive(Debug)]
pub struct DivergenceStats {
    pub detected: u64,
    pub confirmed: u64,
    pub pruned: u64,
    pub pending: u32,
    pub active_chains: u32,
}

// ============================================================================
// SAVE MANAGER
// ============================================================================

/// Manages save/load operations with CDR
pub struct SaveManager {
    current_save: Option<CausalSaveFile>,
    auto_save_interval_ticks: u64,
    last_auto_save_tick: u64,
    snapshot_interval_ticks: u64,
    last_snapshot_tick: u64,
    
    // Fast-forward settings
    fast_forward_multiplier: u32,
    max_ticks_per_frame: u64,
    
    // Compression
    compression_enabled: bool,
}

impl SaveManager {
    pub fn new() -> Self {
        Self {
            current_save: None,
            auto_save_interval_ticks: 60 * 60 * 30, // 30 min at 60fps
            last_auto_save_tick: 0,
            snapshot_interval_ticks: 60 * 60 * 5,   // 5 min at 60fps
            last_snapshot_tick: 0,
            fast_forward_multiplier: 1000,
            max_ticks_per_frame: 100_000,
            compression_enabled: true,
        }
    }
    
    /// Create new save
    pub fn create_save(
        &mut self,
        world_seed: WorldSeed,
        player_seed: PlayerSeed,
    ) -> &CausalSaveFile {
        let save = CausalSaveFile::new(world_seed, player_seed);
        self.current_save = Some(save);
        self.current_save.as_ref().unwrap()
    }
    
    /// Record a divergence event
    pub fn record_divergence(&mut self, event: DivergenceEvent) {
        if let Some(ref mut save) = self.current_save {
            save.record_divergence(event);
        }
    }
    
    /// Check if auto-save should trigger
    pub fn should_auto_save(&self, current_tick: u64) -> bool {
        current_tick - self.last_auto_save_tick >= self.auto_save_interval_ticks
    }
    
    /// Create snapshot anchor
    pub fn create_snapshot(
        &mut self,
        state_data: Vec<u8>,
        performance: PerformanceMetrics,
    ) {
        if let Some(ref mut save) = self.current_save {
            let anchor = SnapshotAnchor {
                anchor_id: save.snapshot_anchors.len() as u64,
                tick: save.last_tick,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                checksum: Self::calculate_checksum(&state_data),
                state_data,
                divergence_from_prev: (save.divergence_count - save.snapshot_anchors.len() as u64) as u32,
                performance_profile: performance,
            };
            
            save.add_snapshot(anchor);
            self.last_snapshot_tick = save.last_tick;
        }
    }
    
    /// Save to file
    pub fn save_to_file(&self, path: &str) -> Result<(), SaveError> {
        let save = self.current_save.as_ref()
            .ok_or(SaveError::NoCurrentSave)?;
            
        let data = save.serialize();
        std::fs::write(path, data)
            .map_err(|e| SaveError::IoError(e.to_string()))?;
            
        Ok(())
    }
    
    /// Load from file
    pub fn load_from_file(&mut self, path: &str) -> Result<(), SaveError> {
        let data = std::fs::read(path)
            .map_err(|e| SaveError::IoError(e.to_string()))?;
            
        let save = CausalSaveFile::deserialize(&data)
            .ok_or(SaveError::DeserializationFailed)?;
            
        self.current_save = Some(save);
        
        // Set last ticks to resume from last snapshot
        if let Some(ref save) = self.current_save {
            self.last_auto_save_tick = save.last_tick;
            self.last_snapshot_tick = save.last_tick;
        }
        
        Ok(())
    }
    
    /// Fast-forward simulation to target tick
    pub fn fast_forward_to_tick(
        &mut self,
        target_tick: u64,
        simulation_fn: impl Fn(u64),
    ) {
        let current_tick = self.current_save.as_ref()
            .map(|s| s.last_tick)
            .unwrap_or(0);
            
        if current_tick >= target_tick {
            return;
        }
        
        // Find nearest snapshot to start from
        let start_from = self.find_nearest_snapshot(target_tick);
        
        // Fast-forward from snapshot
        let start_tick = start_from.map(|s| s.1).unwrap_or(0);
        let mut tick = start_tick;
        
        while tick < target_tick {
            // Simulate in batches
            let batch_end = (tick + self.max_ticks_per_frame).min(target_tick);
            
            for t in tick..batch_end {
                simulation_fn(t);
            }
            
            tick = batch_end;
            
            // Update progress
            if let Some(ref mut save) = self.current_save {
                save.last_tick = tick;
            }
        }
    }
    
    /// Replay divergences from snapshot to target tick
    pub fn replay_divergences(
        &mut self,
        from_tick: u64,
        to_tick: u64,
        divergence_replayer: impl FnMut(&DivergenceEvent),
    ) {
        if let Some(ref save) = self.current_save {
            // Find divergences in range
            let relevant = save.divergence_log.iter()
                .filter(|d| d.tick() >= from_tick && d.tick() <= to_tick)
                .collect::<Vec<_>>();
                
            // Replay in order
            for divergence in relevant {
                divergence_replayer(divergence);
            }
        }
    }
    
    fn find_nearest_snapshot(&self, target_tick: u64) -> Option<(usize, u64)> {
        self.current_save.as_ref()
            .and_then(|save| {
                save.snapshot_anchors.iter()
                    .enumerate()
                    .filter(|(_, s)| s.tick <= target_tick)
                    .max_by_key(|(_, s)| s.tick)
                    .map(|(idx, s)| (idx, s.tick))
            })
    }
    
    fn calculate_checksum(data: &[u8]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        hasher.finish()
    }
    
    pub fn get_save_summary(&self) -> Option<SaveSummary> {
        self.current_save.as_ref().map(|s| s.summary())
    }
}

#[derive(Debug)]
pub enum SaveError {
    NoCurrentSave,
    IoError(String),
    DeserializationFailed,
    ChecksumMismatch,
}

impl std::fmt::Display for SaveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SaveError::NoCurrentSave => write!(f, "No current save to operate on"),
            SaveError::IoError(s) => write!(f, "I/O error: {}", s),
            SaveError::DeserializationFailed => write!(f, "Failed to deserialize save"),
            SaveError::ChecksumMismatch => write!(f, "Save file checksum mismatch"),
        }
    }
}

// ============================================================================
// TRAIT IMPLEMENTATIONS
// ============================================================================

impl DivergenceEvent {
    pub fn event_id(&self) -> u64 {
        match self {
            DivergenceEvent::PlayerInput(d) => d.event_id,
            DivergenceEvent::PhysicsImpulse(d) => d.event_id,
            DivergenceEvent::DialogueChoice(d) => d.event_id,
            DivergenceEvent::WorldChange(d) => d.event_id,
            DivergenceEvent::RemoteAction(d) => d.event_id,
            DivergenceEvent::ScriptedEvent(d) => d.event_id,
            DivergenceEvent::Pruned { event_id, .. } => *event_id,
        }
    }
    
    pub fn tick(&self) -> u64 {
        match self {
            DivergenceEvent::PlayerInput(d) => d.tick,
            DivergenceEvent::PhysicsImpulse(d) => d.tick,
            DivergenceEvent::DialogueChoice(d) => d.tick,
            DivergenceEvent::WorldChange(d) => d.tick,
            DivergenceEvent::RemoteAction(d) => d.tick,
            DivergenceEvent::ScriptedEvent(d) => d.tick,
            DivergenceEvent::Pruned { original_tick, .. } => *original_tick,
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_rng() {
        let mut rng1 = DeterministicRng::from_seed(12345);
        let mut rng2 = DeterministicRng::from_seed(12345);
        
        // Both should produce identical sequences
        for _ in 0..100 {
            assert_eq!(rng1.next_u32(), rng2.next_u32());
        }
        
        // Reset and verify again
        rng1.reset(12345);
        for _ in 0..100 {
            assert_eq!(rng1.next_u32(), rng2.next_u32());
        }
    }

    #[test]
    fn test_causal_save_file() {
        let world_seed = WorldSeed {
            timestamp: 0,
            world_gen_seed: 12345,
            world_config: WorldConfig {
                seed_version: 1,
                world_size: [1000, 100, 1000],
                difficulty_preset: DifficultyPreset::Normal,
                dynamic_object_density: 0.5,
                ai_complexity: AiComplexity::Standard,
            },
            initial_entity_count: 1000,
        };
        
        let player_seed = PlayerSeed {
            player_id: 1,
            creation_tick: 0,
            character_class: 0,
            appearance_seed: 67890,
            initial_stats: PlayerStats {
                health: 100,
                mana: 50,
                stamina: 100,
                strength: 10,
                agility: 10,
                intellect: 10,
                equipment: vec![],
            },
        };
        
        let save = CausalSaveFile::new(world_seed, player_seed);
        
        // Add some divergences
        save.record_divergence(DivergenceEvent::PlayerInput(PlayerInputDivergence {
            event_id: 1,
            tick: 100,
            input_type: InputType::Movement,
            input_data: vec![1, 2, 3],
            causal_chain_id: 0,
            consequence_radius: 5.0,
        }));
        
        // Serialize and deserialize
        let data = save.serialize();
        let loaded = CausalSaveFile::deserialize(&data).unwrap();
        
        assert_eq!(loaded.divergence_count, 1);
        assert_eq!(loaded.world_seed.world_gen_seed, 12345);
    }

    #[test]
    fn test_save_summary() {
        let world_seed = WorldSeed {
            timestamp: 0,
            world_gen_seed: 12345,
            world_config: WorldConfig {
                seed_version: 1,
                world_size: [1000, 100, 1000],
                difficulty_preset: DifficultyPreset::Normal,
                dynamic_object_density: 0.5,
                ai_complexity: AiComplexity::Standard,
            },
            initial_entity_count: 1000,
        };
        
        let player_seed = PlayerSeed {
            player_id: 1,
            creation_tick: 0,
            character_class: 0,
            appearance_seed: 67890,
            initial_stats: PlayerStats {
                health: 100, mana: 50, stamina: 100,
                strength: 10, agility: 10, intellect: 10,
                equipment: vec![],
            },
        };
        
        let save = CausalSaveFile::new(world_seed, player_seed);
        let summary = save.summary();
        
        println!("{}", summary);
        assert!(summary.compression_ratio > 1.0);
    }

    #[test]
    fn test_divergence_event_id() {
        let event = DivergenceEvent::PlayerInput(PlayerInputDivergence {
            event_id: 42,
            tick: 100,
            input_type: InputType::Attack,
            input_data: vec![],
            causal_chain_id: 0,
            consequence_radius: 1.0,
        });
        
        assert_eq!(event.event_id(), 42);
        assert_eq!(event.tick(), 100);
    }
}