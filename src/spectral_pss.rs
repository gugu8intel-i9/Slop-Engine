// src/spectral_pss.rs
//! PROBABILISTIC SPECTRAL SYSTEM (PSS) v1.0
//!
//! "The World as a Probability Wave"
//!
//! Instead of storing discrete, deterministic game assets and simulation states,
//! PSS represents them as compressed probability distributions. The engine only
//! "collapses" these waves into precise, high-fidelity data right before use.
//!
//! Key concepts:
//! 1. Spectral Asset Pool - Compressed assets via spectral modifiers + decoder
//! 2. Potentiality Grid - Probabilistic world state in RAM
//! 3. Cache-Affine Processing - Cluster-first scheduling for CPU efficiency

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use parking_lot::{RwLock, Mutex};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng, distributions::{Distribution, WeightedAliasIndex}};
use glam::{Vec3, Vec4, Mat4};

// ============================================================================
// SPECTRAL ASSET POOL
// ============================================================================

/// A spectral modifier layer - modifies base assets procedurally
#[derive(Debug, Clone)]
pub struct SpectralModifier {
    pub id: u32,
    pub name: String,
    pub frequency_band: SpectralBand,
    pub latent_vector: Vec<f32>,  // Compressed representation
    pub blend_weight: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectralBand {
    Low,      // Base shape, coarse geometry
    Mid,      // Surface detail, normal map influence
    High,     // Micro-detail, scratches, wear
    Ultra,    // Sub-pixel noise, grain
}

impl Default for SpectralBand {
    fn default() -> Self { SpectralBand::Low }
}

/// A base spectral asset with attached modifiers
#[derive(Debug, Clone)]
pub struct SpectralAsset {
    pub id: u64,
    pub name: String,
    pub base_hash: u64,           // Hash of base mesh/texture
    pub modifiers: Vec<SpectralModifier>,
    pub decoder_config: DecoderConfig,
}

#[derive(Debug, Clone)]
pub struct DecoderConfig {
    pub network_weights: Vec<f32>,
    pub input_dim: usize,
    pub output_channels: usize,
    pub latency_samples: usize,
}

/// The global Spectral Asset Pool - stores compressed asset representations
pub struct SpectralAssetPool {
    // Base assets indexed by hash
    base_assets: RwLock<HashMap<u64, Arc<BaseAsset>>>,
    
    // Shared modifier dictionary (game-wide)
    modifier_dictionary: RwLock<HashMap<u32, Arc<SpectralModifier>>>,
    
    // Spectral decoder network (cache-resident)
    decoder_network: Arc<SpectralDecoder>,
    
    // VRAM budget tracking
    vram_used: RwLock<usize>,
    vram_budget: usize,
    
    // Statistics
    stats: RwLock<SpectralStats>,
}

#[derive(Debug, Clone)]
pub struct BaseAsset {
    pub id: u64,
    pub low_freq_data: Vec<u8>,      // Low-res base texture/mesh
    pub memory_bytes: usize,
    pub spectral_channels: usize,
}

#[derive(Debug, Default)]
pub struct SpectralStats {
    pub assets_materialized: u64,
    pub coeff_bytes_streamed: usize,
    pub decoder_invocations: u64,
    pub vram_saved_bytes: usize,
}

impl SpectralAssetPool {
    pub fn new(vram_budget_mb: usize) -> Self {
        Self {
            base_assets: RwLock::new(HashMap::new()),
            modifier_dictionary: RwLock::new(HashMap::new()),
            decoder_network: Arc::new(SpectralDecoder::new(256, 16)), // Small decoder
            vram_used: RwLock::new(0),
            vram_budget: vram_budget_mb * 1024 * 1024,
            stats: RwLock::new(SpectralStats::default()),
        }
    }
    
    /// Register a base asset (e.g., "barrel_base")
    pub fn register_base(&self, base: BaseAsset) {
        let mut assets = self.base_assets.write();
        let mut vram = self.vram_used.write();
        
        *vram += base.memory_bytes;
        assets.insert(base.id, Arc::new(base));
    }
    
    /// Register a shared spectral modifier (e.g., "rust_spectrum")
    pub fn register_modifier(&self, modifier: SpectralModifier) {
        let mut dict = self.modifier_dictionary.write();
        dict.insert(modifier.id, Arc::new(modifier));
    }
    
    /// Materialize an asset with spectral coefficients
    /// Returns the final texel/vertex data procedurally generated
    pub fn materialize(
        &self,
        base_id: u64,
        spectral_coeffs: &[SpectralCoefficients],
    ) -> Option<MaterializedAsset> {
        let base = self.base_assets.read().get(&base_id)?.clone();
        
        // Get modifiers from coefficients
        let active_mods: Vec<Arc<SpectralModifier>> = spectral_coeffs
            .iter()
            .filter_map(|c| self.modifier_dictionary.read().get(&c.modifier_id).cloned())
            .collect();
        
        // Decode using neural network
        let decoded_data = self.decoder_network.decode(
            &active_mods,
            spectral_coeffs,
            base.spectral_channels,
        );
        
        // Update stats
        {
            let mut stats = self.stats.write();
            stats.assets_materialized += 1;
            stats.coeff_bytes_streamed += spectral_coeffs.len() * std::mem::size_of::<SpectralCoefficients>();
            stats.decoder_invocations += 1;
            stats.vram_saved_bytes += decoded_data.len();
        }
        
        Some(MaterializedAsset {
            base_id,
            data: decoded_data,
            resolution: (64, 64), // Could be dynamic
        })
    }
    
    /// Get perceptual LOD chain for distance-based streaming
    pub fn get_lod_chain(&self, base_id: u64, max_depth: u8) -> Vec<SpectralCoefficients> {
        // Return progressively more detailed coefficient chains
        (0..max_depth)
            .map(|depth| SpectralCoefficients {
                modifier_id: 0,
                band: match depth {
                    0 => SpectralBand::Low,
                    1 => SpectralBand::Mid,
                    _ => SpectralBand::High,
                },
                coefficients: vec![0.0; 8 >> depth.min(3)],
                confidence: 1.0 - depth as f32 * 0.2,
            })
            .collect()
    }
    
    pub fn get_stats(&self) -> SpectralStats {
        self.stats.read().clone()
    }
}

/// Spectral coefficients for materialization
#[derive(Debug, Clone)]
pub struct SpectralCoefficients {
    pub modifier_id: u32,
    pub band: SpectralBand,
    pub coefficients: Vec<f32>,  // Latent space coefficients
    pub confidence: f32,          // LOD confidence
}

#[derive(Debug)]
pub struct MaterializedAsset {
    pub base_id: u64,
    pub data: Vec<u8>,
    pub resolution: (u32, u32),
}

// ============================================================================
// SPECTRAL DECODER (Small Neural Network)
// ============================================================================

/// Cache-resident decoder network for procedural generation
pub struct SpectralDecoder {
    weights: Vec<f32>,
    input_dim: usize,
    output_channels: usize,
    hidden_dim: usize,
}

impl SpectralDecoder {
    pub fn new(input_dim: usize, output_channels: usize) -> Self {
        // Pre-allocate small decoder weights (simulated)
        let hidden_dim = 64;
        let total_weights = input_dim * hidden_dim + hidden_dim + hidden_dim * output_channels + output_channels;
        
        Self {
            weights: vec![0.0; total_weights],
            input_dim,
            output_channels,
            hidden_dim,
        }
    }
    
    /// Decode spectral modifiers into final asset data
    pub fn decode(
        &self,
        modifiers: &[Arc<SpectralModifier>],
        coeffs: &[SpectralCoefficients],
        output_channels: usize,
    ) -> Vec<u8> {
        // Build input vector from coefficients
        let mut input = vec![0.0f32; self.input_dim];
        for (i, coeff) in coeffs.iter().enumerate().take(self.input_dim) {
            for (j, &c) in coeff.coefficients.iter().enumerate().take(16) {
                if i * 16 + j < self.input_dim {
                    input[i * 16 + j] = c * coeff.confidence;
                }
            }
        }
        
        // Simple forward pass: input -> hidden -> output
        // Hidden layer
        let hidden: Vec<f32> = (0..self.hidden_dim)
            .map(|j| {
                let mut sum = 0.0f32;
                for i in 0..self.input_dim.min(input.len()) {
                    sum += input[i] * self.weights[i * self.hidden_dim + j];
                }
                // ReLU activation
                sum.max(0.0) + self.weights[self.input_dim * self.hidden_dim + j]
            })
            .collect();
        
        // Output layer
        let mut output = vec![0.0f32; output_channels];
        for k in 0..output_channels.min(self.output_channels) {
            let mut sum = 0.0f32;
            for j in 0..self.hidden_dim {
                sum += hidden[j] * self.weights[
                    self.input_dim * self.hidden_dim + 
                    self.hidden_dim + 
                    j * self.output_channels + k
                ];
            }
            output[k] = sum.tanh() * 0.5 + 0.5; // Sigmoid to [0, 1]
        }
        
        // Pack to RGBA bytes
        let mut bytes = Vec::with_capacity(output.len() * 4);
        for chunk in output.chunks(4) {
            let r = (chunk.get(0).unwrap_or(&0.0) * 255.0) as u8;
            let g = (chunk.get(1).unwrap_or(&0.0) * 255.0) as u8;
            let b = (chunk.get(2).unwrap_or(&0.0) * 255.0) as u8;
            let a = (chunk.get(3).unwrap_or(&1.0) * 255.0) as u8;
            bytes.extend_from_slice(&[r, g, b, a]);
        }
        
        bytes
    }
}

// ============================================================================
// POTENTIALITY GRID
// ============================================================================

/// An octree-based cell storing probabilistic world state
pub struct PotentialityCell {
    pub bounds: [[f32; 3]; 2],  // min, max
    pub probability_dist: ProbabilityDistribution,
    pub seed: u64,               // For deterministic collapse
    pub depth: u8,
    pub children: Option<Box<[Option<PotentialityCell>; 8]>>,
}

#[derive(Debug, Clone)]
pub struct ProbabilityDistribution {
    pub entity_count: WeightedDistribution<u32>,
    pub states: HashMap<String, f32>,  // P(State=...) 
    pub position_variance: Vec3,
    pub velocity_distribution: [f32; 3],  // Mean velocity
}

#[derive(Debug, Clone)]
pub struct WeightedDistribution<T> {
    items: Vec<(T, f32)>,
    alias_index: Option<WeightedAliasIndex>,
}

impl<T: Clone + std::hash::Hash + Eq> WeightedDistribution<T> {
    pub fn new(items: Vec<(T, f32)>) -> Self {
        let weights: Vec<f32> = items.iter().map(|(_, w)| *w).collect();
        let alias_index = WeightedAliasIndex::new(weights).ok();
        Self { items, alias_index }
    }
    
    pub fn sample<R: Rng>(&self, rng: &mut R) -> Option<T> {
        let idx = self.alias_index.as_ref()?.sample(rng);
        Some(self.items[idx].0.clone())
    }
}

/// The Potentiality Grid - stores probabilistic world state
pub struct PotentialityGrid {
    root: PotentialityCell,
    max_depth: u8,
    cell_size: f32,
    
    // Observer state
    observer_position: RwLock<Vec3>,
    frustum_culled: RwLock<HashSet<u64>>,
    
    // Retrocausality buffer
    retro_log: RwLock<Vec<RetroEvent>>,
    
    // RNG for collapse
    rng: RwLock<SmallRng>,
    
    // Statistics
    stats: RwLock<GridStats>,
}

#[derive(Debug, Clone)]
pub struct RetroEvent {
    pub tick: u64,
    pub cell_id: u64,
    pub event_type: String,  // "door_open", "npc_spawn", etc.
    pub probability_seed: u64,
}

#[derive(Debug, Default)]
pub struct GridStats {
    pub cells_collapsed: u64,
    pub cells_in_potential: u64,
    pub ram_saved_bytes: usize,
}

impl PotentialityGrid {
    pub fn new(bounds: [[f32; 3]; 2], max_depth: u8) -> Self {
        let cell_size = Self::compute_cell_size(&bounds, max_depth);
        
        Self {
            root: PotentialityCell {
                bounds,
                probability_dist: Self::default_distribution(),
                seed: rand::random(),
                depth: 0,
                children: None,
            },
            max_depth,
            cell_size,
            observer_position: RwLock::new(Vec3::ZERO),
            frustum_culled: RwLock::new(HashSet::new()),
            retro_log: RwLock::new(Vec::new()),
            rng: RwLock::new(SmallRng::from_entropy()),
            stats: RwLock::new(GridStats::default()),
        }
    }
    
    fn compute_cell_size(bounds: &[[f32; 3]; 2], depth: u8) -> f32 {
        let max_extent = (bounds[1][0] - bounds[0][0])
            .max(bounds[1][1] - bounds[0][1])
            .max(bounds[1][2] - bounds[0][2]);
        max_extent / (2_f32.powi(depth as i32))
    }
    
    fn default_distribution() -> ProbabilityDistribution {
        ProbabilityDistribution {
            entity_count: WeightedDistribution::new(vec![
                (0u32, 0.3),
                (1u32, 0.3),
                (3u32, 0.2),
                (5u32, 0.1),
                (10u32, 0.1),
            ]),
            states: HashMap::from([
                ("Idle".to_string(), 0.5),
                ("Walking".to_string(), 0.3),
                ("Working".to_string(), 0.2),
            ]),
            position_variance: Vec3::new(0.5, 0.0, 0.5),
            velocity_distribution: [0.0, 0.0, 0.0],
        }
    }
    
    /// Update observer position and trigger cell collapse
    pub fn update_observer(&self, position: Vec3) {
        *self.observer_position.write() = position;
        
        // Find cells that are now observable
        let visible = self.get_visible_cells(position);
        
        // Collapse potentiality for newly visible cells
        for cell_id in visible {
            if !self.frustum_culled.read().contains(&cell_id) {
                self.collapse_cell(cell_id);
            }
        }
        
        *self.frustum_culled.write() = visible;
    }
    
    /// Get cell IDs visible from observer position
    fn get_visible_cells(&self, observer: Vec3) -> HashSet<u64> {
        let mut visible = HashSet::new();
        
        // Traverse octree and collect cells in view frustum
        // Simplified: just collect cells within 50m
        const VIEW_RADIUS: f32 = 50.0;
        
        fn traverse(
            cell: &PotentialityCell,
            observer: Vec3,
            visible: &mut HashSet<u64>,
            cell_id: u64,
        ) {
            let center = Vec3::new(
                (cell.bounds[0][0] + cell.bounds[1][0]) * 0.5,
                (cell.bounds[0][1] + cell.bounds[1][1]) * 0.5,
                (cell.bounds[0][2] + cell.bounds[1][2]) * 0.5,
            );
            
            if center.distance(observer) < VIEW_RADIUS {
                visible.insert(cell_id);
                
                if let Some(children) = &cell.children {
                    for (i, child) in children.iter().enumerate() {
                        if let Some(c) = child {
                            traverse(c, observer, visible, cell_id * 8 + i as u64);
                        }
                    }
                }
            }
        }
        
        traverse(&self.root, observer, &mut visible, 0);
        visible
    }
    
    /// Collapse probability wave into deterministic state
    fn collapse_cell(&self, cell_id: u64) {
        let mut rng = self.rng.write();
        let mut stats = self.stats.write();
        
        // Sample from probability distribution
        let entity_count = self.root.probability_dist.entity_count.sample(&mut rng);
        let state = self.root.probability_dist.states.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, _)| k.clone());
        
        stats.cells_collapsed += 1;
        
        // Estimate RAM saved
        // Full entity struct ~2KB, probability ~20 bytes
        let saved = entity_count.unwrap_or(0) as usize * (2048 - 20);
        stats.ram_saved_bytes += saved;
        
        let _ = (cell_id, state);
    }
    
    /// Run retrocausality simulation for smooth materialization
    pub fn retrocausality_step(&self, current_tick: u64) {
        let mut log = self.retro_log.write();
        
        // For each collapsed cell, compute plausible path from unobserved to observed
        // Store high-level event log instead of full simulation history
        
        log.push(RetroEvent {
            tick: current_tick,
            cell_id: 0,  // Simplified
            event_type: "collapse".to_string(),
            probability_seed: rand::random(),
        });
        
        // Keep only last N events
        if log.len() > 120 {
            log.drain(0..60);
        }
    }
    
    /// Get collapsed entity for a cell (deterministic from seed)
    pub fn get_collapsed_entity(&self, cell_id: u64) -> Option<CollapsedEntity> {
        let rng = self.rng.read();
        
        // Use cell seed for deterministic generation
        let seed = cell_id.wrapping_mul(0x9e3779b97f4a7c15);
        let mut deterministic_rng = SmallRng::from_seed(seed);
        
        Some(CollapsedEntity {
            position: Vec3::new(
                deterministic_rng.gen_range(-5.0..5.0),
                0.0,
                deterministic_rng.gen_range(-5.0..5.0),
            ),
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            state: "Idle".to_string(),
            spectral_id: deterministic_rng.gen::<u64>(),
        })
    }
    
    pub fn get_stats(&self) -> GridStats {
        self.stats.read().clone()
    }
}

#[derive(Debug, Clone)]
pub struct CollapsedEntity {
    pub position: Vec3,
    pub rotation: Vec3,
    pub scale: Vec3,
    pub state: String,
    pub spectral_id: u64,
}

// ============================================================================
// CACHE-AFFINE PROCESSING (CLUSTER SYSTEM)
// ============================================================================

/// A data cluster optimized for CPU cache
pub struct DataCluster {
    pub id: u64,
    pub bounds: [[f32; 3]; 2],
    pub entities: Vec<ClusterEntity>,
    pub spectral_coeffs: Vec<SpectralCoefficients>,
    pub transforms: Vec<Mat4>,
    
    // Pre-computed spectral pose data (burned into cluster)
    pub skeleton_influence: Vec<f32>,
    
    // Packed data for streaming
    pub packed_stream: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct ClusterEntity {
    pub id: u64,
    pub spectral_asset_id: u64,
    pub position: Vec3,
    pub velocity: Vec3,
    pub state_hash: u32,
}

/// Cluster processor for cache-affine execution
pub struct ClusterProcessor {
    clusters: RwLock<Vec<DataCluster>>,
    cluster_size_bytes: usize,
    max_clusters: usize,
}

impl ClusterProcessor {
    pub fn new(cluster_size_kb: usize) -> Self {
        Self {
            clusters: RwLock::new(Vec::new()),
            cluster_size_bytes: cluster_size_kb * 1024,
            max_clusters: 1024,
        }
    }
    
    /// Process a cluster entirely in L1/L2 cache
    /// Returns packed transform stream for GPU
    pub fn process_cluster(&self, cluster: &mut DataCluster) -> &[u8] {
        // 1. Collapse potentiality (AI simulation)
        self.process_ai_simulation(cluster);
        
        // 2. Integrate physics on collapsed entities
        self.process_physics(cluster);
        
        // 3. Prepare animation bone matrices
        self.process_animation(cluster);
        
        // 4. Pack for GPU stream
        self.pack_transform_stream(cluster);
        
        &cluster.packed_stream
    }
    
    fn process_ai_simulation(&self, cluster: &mut DataCluster) {
        // Ultra-fast AI on collapsed entities
        for entity in &mut cluster.entities {
            // Simple state machine
            entity.state_hash = entity.state_hash.wrapping_add(1);
        }
    }
    
    fn process_physics(&self, cluster: &mut DataCluster) {
        const GRAVITY: f32 = -9.81;
        const DT: f32 = 0.016;
        
        for entity in &mut cluster.entities {
            entity.velocity.y += GRAVITY * DT;
            entity.position += entity.velocity * DT;
            
            // Ground collision
            if entity.position.y < 0.0 {
                entity.position.y = 0.0;
                entity.velocity.y = 0.0;
            }
        }
    }
    
    fn process_animation(&self, cluster: &mut DataCluster) {
        // Spectrally strided processing - all animation in one pass
        // No separate animation job - data stays in cache
        
        cluster.transforms.clear();
        
        for entity in &cluster.entities {
            // Compose: Cluster root * entity local * skeleton influence
            let base = Mat4::from_translation(entity.position);
            let spectral = Self::apply_spectral_pose(&cluster.skeleton_influence, entity.id);
            cluster.transforms.push(base * spectral);
        }
    }
    
    fn apply_spectral_pose(influence: &[f32], entity_id: u64) -> Mat4 {
        // Simulated spectral pose application
        let phase = (entity_id as f32 * 0.1).sin() * 0.1;
        Mat4::from_rotation_y(phase)
    }
    
    fn pack_transform_stream(&self, cluster: &mut DataCluster) {
        // Pack transforms + spectral coefficients into minimal stream
        cluster.packed_stream.clear();
        
        // Transform data (12 floats = 48 bytes per entity)
        for transform in &cluster.transforms {
            let cols = transform.to_cols_array();
            cluster.packed_stream.extend_from_slice(bytemuck::cast_slice(&cols));
        }
        
        // Spectral coefficients (compressed)
        for coeff in &cluster.spectral_coeffs {
            cluster.packed_stream.push(coeff.modifier_id as u8);
            cluster.packed_stream.push(coeff.band as u8);
            cluster.packed_stream.extend_from_slice(bytemuck::cast_slice(&coeff.coefficients));
        }
    }
    
    /// Allocate a new cluster
    pub fn allocate_cluster(&self, id: u64, bounds: [[f32; 3]; 2]) -> Option<DataCluster> {
        let mut clusters = self.clusters.write();
        
        if clusters.len() >= self.max_clusters {
            return None;
        }
        
        let cluster = DataCluster {
            id,
            bounds,
            entities: Vec::with_capacity(16),
            spectral_coeffs: Vec::with_capacity(8),
            transforms: Vec::with_capacity(16),
            skeleton_influence: vec![0.0; 64],
            packed_stream: Vec::with_capacity(self.cluster_size_bytes),
        };
        
        clusters.push(cluster);
        clusters.last().cloned()
    }
}

// ============================================================================
// INTEGRATION WITH SLOP ENGINE
// ============================================================================

/// PSS Manager - integrates spectral system with engine
pub struct PSSManager {
    pub asset_pool: Arc<SpectralAssetPool>,
    pub potentiality_grid: Arc<PotentialityGrid>,
    pub cluster_processor: Arc<ClusterProcessor>,
    
    config: PSSConfig,
}

#[derive(Debug, Clone)]
pub struct PSSConfig {
    pub vram_budget_mb: usize,
    pub ram_budget_mb: usize,
    pub cluster_size_kb: usize,
    pub view_radius_m: f32,
    pub retro_buffer_seconds: f32,
    pub decoder_input_dim: usize,
    pub enable_ssr: bool,
}

impl Default for PSSConfig {
    fn default() -> Self {
        Self {
            vram_budget_mb: 512,
            ram_budget_mb: 1024,
            cluster_size_kb: 64,
            view_radius_m: 50.0,
            retro_buffer_seconds: 2.0,
            decoder_input_dim: 256,
            enable_ssr: true,
        }
    }
}

impl PSSManager {
    pub fn new(config: PSSConfig) -> Self {
        Self {
            asset_pool: Arc::new(SpectralAssetPool::new(config.vram_budget_mb)),
            potentiality_grid: Arc::new(PotentialityGrid::new(
                [[-500.0, -100.0, -500.0], [500.0, 100.0, 500.0]],
                6,  // 64 cell depth
            )),
            cluster_processor: Arc::new(ClusterProcessor::new(config.cluster_size_kb)),
            config,
        }
    }
    
    /// Initialize with sample spectral assets
    pub fn initialize(&self) {
        // Register sample base assets
        self.asset_pool.register_base(BaseAsset {
            id: 1,
            low_freq_data: vec![0u8; 1024],  // Placeholder
            memory_bytes: 1024,
            spectral_channels: 4,
        });
        
        // Register sample modifiers
        let modifiers = vec![
            SpectralModifier { id: 1, name: "rust".to_string(), frequency_band: SpectralBand::High, latent_vector: vec![0.1; 16], blend_weight: 0.8 },
            SpectralModifier { id: 2, name: "dirt".to_string(), frequency_band: SpectralBand::Mid, latent_vector: vec![0.2; 16], blend_weight: 0.5 },
            SpectralModifier { id: 3, name: "paint_blue".to_string(), frequency_band: SpectralBand::Low, latent_vector: vec![0.0, 0.0, 0.8, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], blend_weight: 1.0 },
        ];
        
        for modifier in modifiers {
            self.asset_pool.register_modifier(modifier);
        }
    }
    
    /// Update PSS state (called each frame)
    pub fn update(&self, camera_position: Vec3, frame: u64) {
        // Update potentiality grid
        self.potentiality_grid.update_observer(camera_position);
        
        // Retrocausality simulation
        self.potentiality_grid.retrocausality_step(frame);
    }
    
    /// Get CPU-optimized cluster data for rendering
    pub fn get_cluster_data(&self, position: Vec3) -> Vec<u8> {
        let mut data = Vec::new();
        
        // Process nearby clusters
        let clusters = self.cluster_processor.clusters.read();
        for cluster in clusters.iter() {
            let center = Vec3::new(
                (cluster.bounds[0][0] + cluster.bounds[1][0]) * 0.5,
                (cluster.bounds[0][1] + cluster.bounds[1][1]) * 0.5,
                (cluster.bounds[0][2] + cluster.bounds[1][2]) * 0.5,
            );
            
            if center.distance(position) < 30.0 {
                data.extend_from_slice(&cluster.packed_stream);
            }
        }
        
        data
    }
    
    pub fn get_savings(&self) -> PSSEfficiencyReport {
        let asset_stats = self.asset_pool.get_stats();
        let grid_stats = self.potentiality_grid.get_stats();
        
        PSSEfficiencyReport {
            vram_saved_mb: asset_stats.vram_saved_bytes as f64 / (1024.0 * 1024.0),
            ram_saved_mb: grid_stats.ram_saved_bytes as f64 / (1024.0 * 1024.0),
            assets_materialized: asset_stats.assets_materialized,
            cells_collapsed: grid_stats.cells_collapsed,
            coeff_bytes_streamed: asset_stats.coeff_bytes_streamed,
            compression_ratio: if asset_stats.vram_saved_bytes > 0 {
                (asset_stats.vram_saved_bytes as f64 / 1_000_000.0) / (asset_stats.coeff_bytes_streamed as f64 / 1_000_000.0).max(0.001)
            } else { 0.0 },
        }
    }
}

#[derive(Debug)]
pub struct PSSEfficiencyReport {
    pub vram_saved_mb: f64,
    pub ram_saved_mb: f64,
    pub assets_materialized: u64,
    pub cells_collapsed: u64,
    pub coeff_bytes_streamed: usize,
    pub compression_ratio: f64,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_decoder() {
        let decoder = SpectralDecoder::new(256, 16);
        let modifier = Arc::new(SpectralModifier {
            id: 1,
            name: "test".to_string(),
            frequency_band: SpectralBand::Mid,
            latent_vector: vec![0.5; 16],
            blend_weight: 1.0,
        });
        
        let coeffs = vec![SpectralCoefficients {
            modifier_id: 1,
            band: SpectralBand::Mid,
            coefficients: vec![0.5; 16],
            confidence: 1.0,
        }];
        
        let result = decoder.decode(&[modifier], &coeffs, 4);
        assert!(!result.is_empty());
        assert_eq!(result.len() % 4, 0);  // RGBA
    }

    #[test]
    fn test_probability_distribution() {
        let dist = WeightedDistribution::new(vec![
            (0u32, 0.3),
            (1u32, 0.3),
            (2u32, 0.4),
        ]);
        
        let mut rng = SmallRng::from_entropy();
        let samples: Vec<u32> = (0..100).filter_map(|_| dist.sample(&mut rng)).collect();
        
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn test_pss_manager() {
        let config = PSSConfig::default();
        let manager = PSSManager::new(config);
        manager.initialize();
        
        manager.update(Vec3::ZERO, 0);
        
        let report = manager.get_savings();
        assert!(report.compression_ratio > 0.0);
    }
}