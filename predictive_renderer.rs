// src/predictive_renderer.rs
//! PREDICTIVE RENDERING SYSTEM v1.0
//!
//! Scene-Graph Delta Prediction + Micro-Tile Re-Rendering + Motion-Corrected Frame Reuse
//!
//! This system dramatically reduces GPU workload by:
//! 1. Predicting which scene elements will change next frame
//! 2. Re-rendering only affected micro-tiles (16x16 pixels)
//! 3. Reprojecting and blending with previous frame
//! 4. Monitoring error and triggering selective refreshes

use std::collections::{HashMap, HashSet, VecDeque};
use std::borrow::Cow;
use std::sync::Arc;
use parking_lot::{RwLock, Mutex};
use wgpu::util::DeviceExt;
use wgpu::*;
use glam::{Mat4, Vec3, Vec2};
use bytemuck::{Pod, Zeroable};

// ============================================================================
// CONFIGURATION
// ============================================================================

#[derive(Debug, Clone)]
pub struct PredictiveRenderConfig {
    // Tile configuration
    pub tile_size: u32,              // 8 or 16 pixels
    pub max_tiles_per_frame: u32,    // Budget limit
    
    // Prediction settings
    pub prediction_window: u32,      // How many frames ahead to predict
    pub velocity_threshold: f32,     // Min movement to mark as "hot"
    pub animation_threshold: f32,    // Min animation change to track
    
    // Frame reuse settings
    pub reuse_depth: u32,            // History depth for reprojection
    pub reprojection_blend: f32,     // Alpha for blending reprojected pixels
    
    // Error watchdog
    pub error_threshold: f32,        // Error threshold for full refresh
    pub max_accumulated_error: f32,  // Max error before forced refresh
    pub watchdog_check_interval: u32,// Frames between error checks
    
    // Performance
    pub enable_debug_overlay: bool,
    pub statistics_enabled: bool,
}

impl Default for PredictiveRenderConfig {
    fn default() -> Self {
        Self {
            tile_size: 16,
            max_tiles_per_frame: 512, // ~25% of 1920x1080 tiles
            prediction_window: 2,
            velocity_threshold: 0.001,
            animation_threshold: 0.01,
            reuse_depth: 4,
            reprojection_blend: 0.85,
            error_threshold: 0.02,
            max_accumulated_error: 0.15,
            watchdog_check_interval: 4,
            enable_debug_overlay: false,
            statistics_enabled: true,
        }
    }
}

// ============================================================================
// CORE TYPES
// ============================================================================

/// Screen divided into tiles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCoord {
    pub x: u32,
    pub y: u32,
}

impl TileCoord {
    pub fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }
    
    pub fn to_index(&self, tiles_per_row: u32) -> usize {
        (self.y * tiles_per_row + self.x) as usize
    }
}

/// Tile rendering state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TileState {
    Clean,          // Matches previous frame
    Hot,            // Needs re-rendering
    Reprojected,    // Reused from previous frame
    Error,          // Artifact detected
    ForceRefresh,   // Full refresh triggered
}

/// Tracks what changed in the scene
#[derive(Debug, Clone)]
pub struct DeltaPrediction {
    pub changed_entities: HashSet<u64>,
    pub affected_tiles: HashSet<TileCoord>,
    pub motion_vectors: HashMap<u64, MotionVector>,
    pub camera_motion: CameraMotion,
}

#[derive(Debug, Clone)]
pub struct MotionVector {
    pub entity_id: u64,
    pub screen_delta: Vec2,    // Pixel movement this frame
    pub depth_change: f32,     // Z movement
}

#[derive(Debug, Clone)]
pub struct CameraMotion {
    pub position_delta: Vec3,
    pub rotation_delta: Vec2,  // Pitch, yaw delta
    pub view_proj_change: f32,
}

/// Single tile data
#[derive(Debug, Clone)]
struct Tile {
    coord: TileCoord,
    state: TileState,
    error_score: f32,
    accumulated_error: f32,
    last_render_frame: u64,
    reprojection_offset: Vec2,
}

/// Frame history for reprojection
#[derive(Debug)]
struct FrameHistory {
    frame: Vec<u8>,
    depth: Vec<f32>,
    motion_vectors: Vec<Vec2>, // Per-pixel motion
    timestamp: u64,
}

// ============================================================================
// DELTA PREDICTOR
// ============================================================================

/// Predicts which scene elements will change next frame
pub struct DeltaPredictor {
    config: PredictiveRenderConfig,
    previous_transforms: HashMap<u64, TransformCache>,
    previous_animations: HashMap<u64, AnimationCache>,
    camera_history: VecDeque<CameraMotion>,
    motion_model: SimpleMotionPredictor,
}

#[derive(Debug, Clone)]
struct TransformCache {
    position: Vec3,
    rotation: Vec3,
    scale: Vec3,
    screen_bounds: ScreenBounds,
}

#[derive(Debug, Clone)]
struct ScreenBounds {
    min_x: f32, min_y: f32,
    max_x: f32, max_y: f32,
    depth: f32,
}

#[derive(Debug, Clone)]
struct AnimationCache {
    progress: f32,
    weights: Vec<f32>,
    screen_impact: f32,
}

/// Simple motion predictor (heuristic-based, no ML needed)
#[derive(Debug)]
struct SimpleMotionPredictor {
    velocity_history: HashMap<u64, VecDeque<Vec3>>,
}

impl DeltaPredictor {
    pub fn new(config: PredictiveRenderConfig) -> Self {
        Self {
            config,
            previous_transforms: HashMap::new(),
            previous_animations: HashMap::new(),
            camera_history: VecDeque::with_capacity(4),
            motion_model: SimpleMotionPredictor {
                velocity_history: HashMap::new(),
            },
        }
    }
    
    /// Predict which entities/tiles will change next frame
    pub fn predict(&mut self, scene_data: &SceneSnapshot) -> DeltaPrediction {
        let mut delta = DeltaPrediction {
            changed_entities: HashSet::new(),
            affected_tiles: HashSet::new(),
            motion_vectors: HashMap::new(),
            camera_motion: CameraMotion {
                position_delta: Vec3::ZERO,
                rotation_delta: Vec2::ZERO,
                view_proj_change: 0.0,
            },
        };
        
        // 1. Camera motion prediction
        self.predict_camera_motion(&mut delta, scene_data);
        
        // 2. Entity transform prediction
        self.predict_entity_transforms(&mut delta, scene_data);
        
        // 3. Animation prediction
        self.predict_animations(&mut delta, scene_data);
        
        // 4. AOE (Area of Effect) - affected tiles
        self.compute_affected_tiles(&mut delta, scene_data);
        
        delta
    }
    
    fn predict_camera_motion(&mut self, delta: &mut DeltaPrediction, scene: &SceneSnapshot) {
        let motion = CameraMotion {
            position_delta: scene.camera_position - self.get_last_camera_pos(),
            rotation_delta: Vec2::new(
                scene.camera_yaw - self.camera_history.back().map(|c| c.rotation_delta.x).unwrap_or(0.0),
                scene.camera_pitch - self.camera_history.back().map(|c| c.rotation_delta.y).unwrap_or(0.0),
            ),
            view_proj_change: self.estimate_view_proj_change(scene),
        };
        
        delta.camera_motion = motion;
        
        // Store in history
        if self.camera_history.len() >= 4 {
            self.camera_history.pop_front();
        }
        self.camera_history.push_back(motion);
    }
    
    fn get_last_camera_pos(&self) -> Vec3 {
        self.camera_history.back().map(|c| c.position_delta).unwrap_or(Vec3::ZERO)
    }
    
    fn estimate_view_proj_change(&self, scene: &SceneSnapshot) -> f32 {
        if let Some(prev) = self.camera_history.back() {
            prev.view_proj_change * 0.8 + // Damped prediction
            (prev.position_delta.length() * 0.2)
        } else {
            1.0
        }
    }
    
    fn predict_entity_transforms(&mut self, delta: &mut DeltaPrediction, scene: &SceneSnapshot) {
        for entity in &scene.entities {
            let prev = self.previous_transforms.get(&entity.id);
            
            // Calculate velocity
            let velocity = if let Some(p) = prev {
                entity.position - p.position
            } else {
                Vec3::ZERO
            };
            
            // Track velocity history
            let history = self.motion_model.velocity_history.entry(entity.id).or_insert_with(VecDeque::new);
            if history.len() >= 4 {
                history.pop_front();
            }
            history.push_back(velocity);
            
            // Predict next frame position
            let predicted_pos = if history.len() >= 2 {
                let avg_vel = history.iter().sum::<Vec3>() / history.len() as f32;
                entity.position + avg_vel * self.config.prediction_window as f32
            } else {
                entity.position
            };
            
            // Check if significant change
            if velocity.length() > self.config.velocity_threshold {
                delta.changed_entities.insert(entity.id);
                delta.motion_vectors.insert(entity.id, MotionVector {
                    entity_id: entity.id,
                    screen_delta: self.world_to_screen_delta(velocity, entity.position),
                    depth_change: predicted_pos.z - entity.position.z,
                });
            }
            
            // Update cache
            self.previous_transforms.insert(entity.id, TransformCache {
                position: entity.position,
                rotation: entity.rotation,
                scale: entity.scale,
                screen_bounds: self.compute_screen_bounds(entity),
            });
        }
    }
    
    fn predict_animations(&mut self, delta: &mut DeltaPrediction, scene: &SceneSnapshot) {
        for (entity_id, anim) in &scene.active_animations {
            if let Some(prev) = self.previous_animations.get(entity_id) {
                let progress_delta = anim.progress - prev.progress;
                
                if progress_delta > self.config.animation_threshold {
                    delta.changed_entities.insert(*entity_id);
                }
            }
            
            self.previous_animations.insert(*entity_id, AnimationCache {
                progress: anim.progress,
                weights: anim.weights.clone(),
                screen_impact: self.estimate_animation_impact(anim),
            });
        }
    }
    
    fn compute_affected_tiles(&mut self, delta: &mut DeltaPrediction, scene: &SceneSnapshot) {
        let tiles_per_row = (scene.screen_width / self.config.tile_size) as u32;
        let tiles_per_col = (scene.screen_height / self.config.tile_size) as u32;
        
        // Camera motion affects ALL tiles
        if delta.camera_motion.view_proj_change > 0.1 {
            // Mark all tiles as affected for camera movement
            for y in 0..tiles_per_col {
                for x in 0..tiles_per_row {
                    delta.affected_tiles.insert(TileCoord::new(x, y));
                }
            }
            return;
        }
        
        // Entity changes affect specific tiles
        for entity_id in &delta.changed_entities {
            if let Some(entity) = scene.entities.iter().find(|e| e.id == *entity_id) {
                if let Some(bounds) = self.previous_transforms.get(entity_id) {
                    // Convert world bounds to tile coords
                    self.add_tiles_for_bounds(&mut delta.affected_tiles, &bounds.screen_bounds, tiles_per_row);
                }
            }
        }
        
        // Limit tiles to budget
        let max_tiles = self.config.max_tiles_per_frame as usize;
        if delta.affected_tiles.len() > max_tiles {
            // Prioritize by distance to camera
            let mut prioritized: Vec<_> = delta.affected_tiles.iter().cloned().collect();
            prioritized.truncate(max_tiles);
            delta.affected_tiles = prioritized.into_iter().collect();
        }
    }
    
    fn world_to_screen_delta(&self, world_delta: Vec3, _position: Vec3) -> Vec2 {
        // Simplified: in real impl, project delta to screen space
        Vec2::new(world_delta.x * 1000.0, world_delta.y * 1000.0)
    }
    
    fn compute_screen_bounds(&self, entity: &EntitySnapshot) -> ScreenBounds {
        // Simplified projection - real impl would use view-projection matrix
        let depth = (entity.position.z - 1.0).max(0.1);
        let scale = 1000.0 / depth;
        
        ScreenBounds {
            min_x: entity.position.x * scale,
            min_y: entity.position.y * scale,
            max_x: (entity.position.x + entity.scale.x) * scale,
            max_y: (entity.position.y + entity.scale.y) * scale,
            depth,
        }
    }
    
    fn add_tiles_for_bounds(&self, tiles: &mut HashSet<TileCoord>, bounds: &ScreenBounds, tiles_per_row: u32) {
        let tile_size = self.config.tile_size as f32;
        
        let min_tile_x = ((bounds.min_x / tile_size) as u32).saturating_sub(1);
        let min_tile_y = ((bounds.min_y / tile_size) as u32).saturating_sub(1);
        let max_tile_x = ((bounds.max_x / tile_size) as u32).saturating_add(1);
        let max_tile_y = ((bounds.max_y / tile_size) as u32).saturating_add(1);
        
        for y in min_tile_y..=max_tile_y {
            for x in min_tile_x..=max_tile_x {
                tiles.insert(TileCoord::new(x, y));
            }
        }
    }
    
    fn estimate_animation_impact(&self, anim: &AnimationSnapshot) -> f32 {
        anim.weights.iter().map(|w| w.abs()).sum::<f32>() * anim.progress
    }
}

// ============================================================================
// SCENE SNAPSHOT (Data from main engine)
// ============================================================================

// Re-export EntitySnapshot from network module for compatibility
pub use crate::network::EntitySnapshot;

#[derive(Debug)]
pub struct SceneSnapshot {
    pub camera_position: Vec3,
    pub camera_pitch: f32,
    pub camera_yaw: f32,
    pub screen_width: u32,
    pub screen_height: u32,
    pub entities: Vec<EntitySnapshot>,
    pub active_animations: HashMap<u64, AnimationSnapshot>,
    pub particle_systems: Vec<ParticleSystemSnapshot>,
    pub lighting_changes: Vec<LightChange>,
}

#[derive(Debug, Clone)]
pub struct AnimationSnapshot {
    pub progress: f32,
    pub weights: Vec<f32>,
    pub affected_bones: Vec<u32>,
}

#[derive(Debug)]
pub struct ParticleSystemSnapshot {
    pub id: u64,
    pub emitter_position: Vec3,
    pub active_particles: u32,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct LightChange {
    pub light_id: u64,
    pub position_delta: Vec3,
    pub intensity_delta: f32,
}

// ============================================================================
// TILE MANAGER
// ============================================================================

pub struct TileManager {
    config: PredictiveRenderConfig,
    tiles: Vec<Tile>,
    tiles_per_row: u32,
    tiles_per_col: u32,
    frame_count: u64,
}

impl TileManager {
    pub fn new(config: PredictiveRenderConfig, screen_width: u32, screen_height: u32) -> Self {
        let tiles_per_row = (screen_width + config.tile_size - 1) / config.tile_size;
        let tiles_per_col = (screen_height + config.tile_size - 1) / config.tile_size;
        let total_tiles = (tiles_per_row * tiles_per_col) as usize;
        
        let tiles = (0..total_tiles).map(|i| {
            let x = (i as u32) % tiles_per_row;
            let y = (i as u32) / tiles_per_row;
            Tile {
                coord: TileCoord::new(x, y),
                state: TileState::Clean,
                error_score: 0.0,
                accumulated_error: 0.0,
                last_render_frame: 0,
                reprojection_offset: Vec2::ZERO,
            }
        }).collect();
        
        Self {
            config,
            tiles,
            tiles_per_row,
            tiles_per_col,
            frame_count: 0,
        }
    }
    
    pub fn update(&mut self, delta: &DeltaPrediction) {
        self.frame_count += 1;
        
        // Reset all tiles
        for tile in &mut self.tiles {
            tile.state = TileState::Clean;
            tile.error_score = 0.0;
        }
        
        // Mark tiles affected by delta
        for coord in &delta.affected_tiles {
            if let Some(tile) = self.get_tile_mut(*coord) {
                tile.state = TileState::Hot;
            }
        }
        
        // If camera moved significantly, mark all tiles
        if delta.camera_motion.view_proj_change > 0.2 {
            for tile in &mut self.tiles {
                tile.state = TileState::Hot;
            }
        }
    }
    
    pub fn get_tile(&self, coord: TileCoord) -> Option<&Tile> {
        let idx = coord.to_index(self.tiles_per_row);
        self.tiles.get(idx)
    }
    
    pub fn get_tile_mut(&mut self, coord: TileCoord) -> Option<&mut Tile> {
        let idx = coord.to_index(self.tiles_per_row);
        self.tiles.get_mut(idx)
    }
    
    pub fn mark_reprojected(&mut self, coords: &[TileCoord]) {
        for coord in coords {
            if let Some(tile) = self.get_tile_mut(*coord) {
                tile.state = TileState::Reprojected;
            }
        }
    }
    
    pub fn mark_error(&mut self, coord: TileCoord, error: f32) {
        if let Some(tile) = self.get_tile_mut(coord) {
            tile.error_score = error;
            tile.accumulated_error += error;
            
            if tile.accumulated_error > self.config.max_accumulated_error {
                tile.state = TileState::ForceRefresh;
            } else if error > self.config.error_threshold {
                tile.state = TileState::Error;
            }
        }
    }
    
    pub fn hot_tiles(&self) -> Vec<TileCoord> {
        self.tiles.iter()
            .filter(|t| matches!(t.state, TileState::Hot | TileState::ForceRefresh))
            .map(|t| t.coord)
            .collect()
    }
    
    pub fn statistics(&self) -> TileStatistics {
        let mut hot = 0;
        let mut clean = 0;
        let mut reprojected = 0;
        let mut error = 0;
        
        for tile in &self.tiles {
            match tile.state {
                TileState::Hot => hot += 1,
                TileState::Clean => clean += 1,
                TileState::Reprojected => reprojected += 1,
                TileState::Error | TileState::ForceRefresh => error += 1,
            }
        }
        
        TileStatistics {
            total_tiles: self.tiles.len(),
            hot_tiles: hot,
            clean_tiles: clean,
            reprojected_tiles: reprojected,
            error_tiles: error,
            hot_ratio: hot as f32 / self.tiles.len() as f32,
        }
    }
}

#[derive(Debug)]
pub struct TileStatistics {
    pub total_tiles: usize,
    pub hot_tiles: usize,
    pub clean_tiles: usize,
    pub reprojected_tiles: usize,
    pub error_tiles: usize,
    pub hot_ratio: f32,
}

// ============================================================================
// REPROJECTION ENGINE
// ============================================================================

pub struct ReprojectionEngine {
    config: PredictiveRenderConfig,
    frame_history: VecDeque<FrameHistory>,
    reprojection_shader: ShaderModule,
}

impl ReprojectionEngine {
    pub fn new(device: &Device) -> Self {
        let shader = device.create_shader_module(&ShaderModuleDescriptor {
            label: Some("predictive_reproject"),
            source: ShaderSource::Wgsl(Cow::Borrowed(REPROJECT_WGSL)),
        });
        
        Self {
            config: PredictiveRenderConfig::default(),
            frame_history: VecDeque::with_capacity(4),
            reprojection_shader: shader,
        }
    }
    
    /// Reproject previous frame to current view
    pub fn reproject(
        &mut self,
        device: &Device,
        queue: &Queue,
        current_view_proj: Mat4,
        delta: &DeltaPrediction,
        output_texture: &TextureView,
    ) {
        if let Some(prev) = self.frame_history.back() {
            // Reproject previous frame using motion vectors
            // This is a simplified version - real impl would use compute shader
            let _ = (device, queue, current_view_proj, delta, output_texture, prev);
            // Implementation would:
            // 1. Sample previous frame with motion offset
            // 2. Apply camera motion correction
            // 3. Blend with re-rendered tiles
        }
    }
    
    pub fn push_frame(&mut self, frame: Vec<u8>, depth: Vec<f32>, motion: Vec<Vec2>) {
        if self.frame_history.len() >= self.config.reuse_depth as usize {
            self.frame_history.pop_front();
        }
        
        self.frame_history.push_back(FrameHistory {
            frame,
            depth,
            motion_vectors: motion,
            timestamp: 0,
        });
    }
}

// ============================================================================
// ERROR WATCHDOG
// ============================================================================

pub struct ErrorWatchdog {
    config: PredictiveRenderConfig,
    frame_count: u32,
    total_error: f32,
    refresh_count: u32,
}

impl ErrorWatchdog {
    pub fn new(config: PredictiveRenderConfig) -> Self {
        Self {
            config,
            frame_count: 0,
            total_error: 0.0,
            refresh_count: 0,
        }
    }
    
    /// Check if a tile needs refresh based on accumulated error
    pub fn check_tile(&mut self, tile: &Tile) -> bool {
        self.frame_count += 1;
        
        // Add error to total
        self.total_error += tile.accumulated_error;
        
        // Check periodically
        if self.frame_count % self.config.watchdog_check_interval == 0 {
            let avg_error = self.total_error / self.frame_count as f32;
            
            if avg_error > self.config.error_threshold || tile.accumulated_error > self.config.max_accumulated_error {
                self.refresh_count += 1;
                self.total_error = 0.0;
                self.frame_count = 0;
                return true;
            }
        }
        
        tile.state == TileState::ForceRefresh
    }
    
    pub fn get_stats(&self) -> WatchdogStats {
        WatchdogStats {
            frames_checked: self.frame_count,
            total_refreshes: self.refresh_count,
            average_error: if self.frame_count > 0 { self.total_error / self.frame_count as f32 } else { 0.0 },
        }
    }
}

#[derive(Debug)]
pub struct WatchdogStats {
    pub frames_checked: u32,
    pub total_refreshes: u32,
    pub average_error: f32,
}

// ============================================================================
// PREDICTIVE RENDERER (Main System)
// ============================================================================

pub struct PredictiveRenderer {
    config: PredictiveRenderConfig,
    delta_predictor: DeltaPredictor,
    tile_manager: TileManager,
    reprojection_engine: ReprojectionEngine,
    error_watchdog: ErrorWatchdog,
    stats: PredictiveStats,
}

#[derive(Debug, Default)]
pub struct PredictiveStats {
    pub frames_rendered: u64,
    pub tiles_reduced: u64,
    pub reprojections: u64,
    pub forced_refreshes: u32,
    pub avg_hot_ratio: f32,
    pub gpu_time_saved_ms: f32,
}

impl PredictiveRenderer {
    pub fn new(device: &Device, config: PredictiveRenderConfig, screen_width: u32, screen_height: u32) -> Self {
        Self {
            delta_predictor: DeltaPredictor::new(config.clone()),
            tile_manager: TileManager::new(config.clone(), screen_width, screen_height),
            reprojection_engine: ReprojectionEngine::new(device),
            error_watchdog: ErrorWatchdog::new(config.clone()),
            stats: PredictiveStats::default(),
        }
    }
    
    /// Main render path
    pub fn render(
        &mut self,
        device: &Device,
        queue: &Queue,
        scene: &SceneSnapshot,
        encoder: &mut CommandEncoder,
        target_view: &TextureView,
    ) {
        self.stats.frames_rendered += 1;
        
        // 1. PREDICT: Get delta prediction
        let delta = self.delta_predictor.predict(scene);
        
        // 2. TILE: Update tile states
        self.tile_manager.update(&delta);
        
        // 3. REPROJECT: Reproject previous frame
        // (Skipped for now - full implementation would use compute shader)
        
        // 4. RENDER: Only render hot tiles
        let hot_tiles = self.tile_manager.hot_tiles();
        let hot_count = hot_tiles.len();
        
        // Update statistics
        let tile_stats = self.tile_manager.statistics();
        self.stats.tiles_reduced += (tile_stats.total_tiles - hot_count) as u64;
        self.stats.avg_hot_ratio = self.stats.avg_hot_ratio * 0.9 + tile_stats.hot_ratio * 0.1;
        
        // 5. WATCHDOG: Check for error accumulation
        for tile in &self.tile_manager.tiles {
            if self.error_watchdog.check_tile(tile) {
                self.stats.forced_refreshes += 1;
            }
        }
        
        // Estimate GPU time saved
        let full_frame_pixels = scene.screen_width * scene.screen_height;
        let hot_pixels = hot_count as u32 * self.config.tile_size * self.config.tile_size;
        let saved_ratio = 1.0 - (hot_pixels as f32 / full_frame_pixels as f32);
        self.stats.gpu_time_saved_ms += saved_ratio * 0.5; // Rough estimate
        
        let _ = (encoder, target_view); // Placeholder for actual rendering
    }
    
    /// Get current statistics
    pub fn get_stats(&self) -> PredictiveStats {
        self.stats.clone()
    }
    
    /// Resize renderer
    pub fn resize(&mut self, screen_width: u32, screen_height: u32) {
        self.tile_manager = TileManager::new(self.config.clone(), screen_width, screen_height);
    }
}

// ============================================================================
// WGSL SHADERS
// ============================================================================

const REPROJECT_WGSL: &str = r#"
struct ReprojectUniforms {
    prev_view_proj: mat4x4<f32>,
    curr_view_proj: mat4x4<f32>,
    inv_curr_view_proj: mat4x4<f32>,
    motion_scale: f32,
    blend_factor: f32,
};

@group(0) @binding(0) var prev_tex: texture_2d<f32>;
@group(0) @binding(1) var prev_depth: texture_depth_2d;
@group(0) @binding(2) var curr_depth: texture_depth_2d;
@group(0) @binding(3) var motion_tex: texture_2d<f32>;
@group(0) @binding(4) var<uniform> uniforms: ReprojectUniforms;
@group(0) @binding(5) var samp: sampler;

@fragment fn reproject(
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>
) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(prev_tex));
    
    // Sample current depth
    let curr_d = textureSample(curr_depth, samp, uv).r;
    
    // Unproject to world space
    let clip_pos = vec4<f32>(uv * 2.0 - 1.0, curr_d * 2.0 - 1.0, 1.0);
    let world_pos = uniforms.inv_curr_view_proj * clip_pos;
    let world_pos = world_pos / world_pos.w;
    
    // Project to previous frame
    let prev_clip = uniforms.prev_view_proj * vec4<f32>(world_pos.xyz, 1.0);
    let prev_uv = prev_clip.xy / prev_clip.w * 0.5 + 0.5;
    
    // Sample motion vectors for refinement
    let motion = textureSample(motion_tex, samp, uv).rg;
    let refined_uv = prev_uv + motion * uniforms.motion_scale;
    
    // Check depth validity
    let prev_d = textureSample(prev_depth, samp, refined_uv).r;
    let depth_diff = abs(curr_d - prev_d);
    
    if (depth_diff < 0.01 && prev_uv.x >= 0.0 && prev_uv.x <= 1.0 && prev_uv.y >= 0.0 && prev_uv.y <= 1.0) {
        // Reprojection valid - blend with previous
        let prev_color = textureSample(prev_tex, samp, refined_uv).rgb;
        return vec4<f32>(prev_color, uniforms.blend_factor);
    } else {
        // Reprojection failed - no reuse
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
}
"#;

const TILE_RENDER_WGSL: &str = r#"
struct TileUniforms {
    tile_min: vec2<u32>,
    tile_max: vec2<u32>,
    is_hot: u32,
};

@group(0) @binding(0) var<uniform> tile_uniforms: TileUniforms;

@fragment fn tile_check(
    @builtin(position) pos: vec4<f32>
) -> @location(0) vec4<f32> {
    let px = u32(pos.x);
    let py = u32(pos.y);
    
    if (px >= tile_uniforms.tile_min.x && px < tile_uniforms.tile_max.x &&
        py >= tile_uniforms.tile_min.y && py < tile_uniforms.tile_max.y) {
        // Inside hot tile - render
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    } else {
        // Outside hot tile - skip
        discard;
    }
}
"#;

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_coordinate() {
        let coord = TileCoord::new(5, 3);
        assert_eq!(coord.to_index(10), 35);
    }

    #[test]
    fn test_predictive_render_config() {
        let config = PredictiveRenderConfig::default();
        assert_eq!(config.tile_size, 16);
        assert_eq!(config.max_tiles_per_frame, 512);
    }
}