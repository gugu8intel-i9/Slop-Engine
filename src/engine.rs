// src/engine.rs
//! Core Engine orchestrator with all systems integrated
//! - Predictive Rendering
//! - Memory Management (W-TinyLFU)
//! - Network (Client-side prediction)
//! - Resource Management

use std::sync::Arc;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use crate::{
    predictive_renderer::{PredictiveRenderer, PredictiveRenderConfig, SceneSnapshot, DeltaPrediction},
    offload::{OffloadManager, OffloadConfig, ResourceTier, Priority},
    network::{NetworkSystem, NetworkRole, GameInput},
    resource_manager::{ResourceManager, ResourceConfig, MemoryStats},
    camera::Camera,
    camera_controller::CameraController,
    renderer::Renderer,
    scene::Scene,
    physics::PhysicsWorld,
    time::Time,
    fps_counter::FpsCounter,
    input_system::InputSystem,
    gui::GuiSystem,
    audio::AudioEngine,
};

/// Main engine configuration
#[derive(Debug, Clone)]
pub struct EngineSettings {
    pub predictive_rendering: PredictiveRenderConfig,
    pub offload: OffloadConfig,
    pub network: NetworkConfig,
    pub resource: ResourceConfig,
    pub target_fps: u32,
    pub enable_debug_overlay: bool,
}

#[derive(Debug, Clone)]
pub struct NetworkConfig {
    pub enabled: bool,
    pub role: NetworkRole,
    pub tick_rate: u32,
    pub client_prediction: bool,
    pub delta_compression: bool,
}

impl Default for EngineSettings {
    fn default() -> Self {
        Self {
            predictive_rendering: PredictiveRenderConfig::default(),
            offload: OffloadConfig::default(),
            network: NetworkConfig {
                enabled: false,
                role: NetworkRole::Client,
                tick_rate: 60,
                client_prediction: true,
                delta_compression: true,
            },
            resource: ResourceConfig::default(),
            target_fps: 60,
            enable_debug_overlay: false,
        }
    }
}

/// Engine performance statistics
#[derive(Debug, Default)]
pub struct EngineStats {
    pub fps: f32,
    pub frame_time_ms: f32,
    pub predictive_hot_ratio: f32,
    pub gpu_time_saved_ms: f32,
    pub vram_used_mb: f64,
    pub vram_budget_mb: f64,
    pub entities_culled: usize,
    pub network_rtt_ms: f32,
}

impl EngineStats {
    pub fn summary(&self) -> String {
        format!(
            "FPS: {:.1} | Frame: {:.2}ms | Predictive: {:.0}% saved | VRAM: {:.0}/{:.0}MB | RTT: {:.0}ms",
            self.fps,
            self.frame_time_ms,
            self.predictive_hot_ratio * 100.0,
            self.vram_used_mb,
            self.vram_budget_mb,
            self.network_rtt_ms
        )
    }
}

/// Main Engine orchestrator with all subsystems integrated
pub struct Engine {
    // Windowing & Context
    pub window: Arc<Window>,
    
    // Core Systems
    pub scene: Scene,
    pub physics: PhysicsWorld,
    pub time: Time,
    pub input: InputSystem,
    pub fps_counter: FpsCounter,
    
    // Rendering Systems
    pub camera: Camera,
    pub camera_controller: CameraController,
    pub renderer: Renderer,
    pub predictive_renderer: Option<PredictiveRenderer>,
    
    // Memory & Resources
    pub resource_manager: Arc<ResourceManager>,
    pub offload_manager: OffloadManager,
    
    // Network
    pub network: NetworkSystem,
    pub client_prediction_enabled: bool,
    
    // Audio & UI
    pub audio: AudioEngine,
    pub gui: GuiSystem,
    
    // Settings & Stats
    pub settings: EngineSettings,
    pub stats: EngineStats,
}

impl Engine {
    /// Initialize engine with settings
    pub async fn new(event_loop: &EventLoop<()>, settings: EngineSettings) -> Result<Self, crate::error::EngineError> {
        // 1. Create window
        let window = Arc::new(
            WindowBuilder::new()
                .with_title("Slop Engine v2.0")
                .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720))
                .build(event_loop)
                .unwrap(),
        );

        // 2. Initialize resource manager first (needed by other systems)
        let resource_manager = Arc::new(ResourceManager::new(
            todo!("Get device from context"),
            todo!("Get queue from context"),
            settings.resource.clone(),
        ));

        // 3. Initialize offload manager
        let offload_manager = OffloadManager::new(settings.offload.clone());

        // 4. Initialize network system
        let mut network = NetworkSystem::new(settings.network.role);
        network.prediction_enabled = settings.network.client_prediction;
        network.delta_compression_enabled = settings.network.delta_compression;

        // 5. Initialize core systems
        let scene = Scene::new();
        let physics = PhysicsWorld::new();
        let input = InputSystem::new();
        let time = Time::new();
        let fps_counter = FpsCounter::new();

        // 6. Initialize camera
        let camera = Camera::new(1280.0, 720.0);
        let camera_controller = CameraController::new(2.0, 0.5);

        // 7. Initialize renderer
        let renderer = Renderer::new(window.clone()).await;

        // 8. Initialize predictive renderer
        let predictive_renderer = Some(PredictiveRenderer::new(
            todo!("Get device"),
            settings.predictive_rendering.clone(),
            1280,
            720,
        ));

        // 9. Initialize audio and GUI
        let audio = AudioEngine::new()?;
        let gui = GuiSystem::new(&window);

        Ok(Self {
            window,
            scene,
            physics,
            time,
            input,
            fps_counter,
            camera,
            camera_controller,
            renderer,
            predictive_renderer,
            resource_manager,
            offload_manager,
            network,
            client_prediction_enabled: settings.network.client_prediction,
            audio,
            gui,
            settings,
            stats: EngineStats::default(),
        })
    }

    /// Standard initialization with default settings
    pub async fn with_defaults(event_loop: &EventLoop<()>) -> Result<Self, crate::error::EngineError> {
        Self::new(event_loop, EngineSettings::default()).await
    }

    /// Handle input events
    pub fn handle_event(&mut self, event: &Event<()>) {
        // GUI intercepts events first
        if self.gui.handle_event(event) {
            return;
        }

        // Pass to input system
        self.input.process_event(event);

        // Network receives inputs
        if let Event::WindowEvent { event: WindowEvent::KeyboardInput { input, .. }, .. } = event {
            if input.state == ElementState::Pressed {
                // Create input for network
                let input_data = GameInput {
                    player_id: 0,
                    frame: self.time.frame_count(),
                    inputs: vec![input.scancode as u8],
                    checksum: 0,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0),
                };
                
                if self.client_prediction_enabled {
                    self.network.queue_input(input_data);
                }
            }
        }
    }

    /// Main update loop
    pub fn update(&mut self) {
        self.time.tick();
        let dt = self.time.delta_seconds();
        
        // Update FPS counter
        self.fps_counter.update(dt);
        self.stats.fps = self.fps_counter.fps();
        self.stats.frame_time_ms = dt * 1000.0;

        // Update camera controller
        self.camera_controller.update_camera(&mut self.camera, &self.input, dt);

        // Update audio listener position
        self.audio.update(&self.camera);

        // Update GUI
        self.gui.update(dt);

        // Fixed timestep for physics and network
        while self.time.consume_fixed_timestep() {
            self.fixed_update();
        }

        // Clear input deltas
        self.input.clear_frame_state();
    }

    /// Fixed timestep updates (physics, networking)
    fn fixed_update(&mut self) {
        let dt = self.time.fixed_delta_seconds();

        // Step physics
        self.physics.step(dt);

        // Update scene
        self.scene.update(dt);

        // Update network
        if self.client_prediction_enabled {
            self.network.update_connection_rtt(0);
            self.stats.network_rtt_ms = self.network.get_rtt(0).unwrap_or(0.0);
        }
    }

    /// Prepare scene for rendering
    fn prepare_scene(&mut self) {
        // Get visible entities through culling
        let visible_entities = crate::culling::FrustumCuller::cull(&self.scene, &self.camera);
        self.stats.entities_culled = self.scene.entity_count() - visible_entities.len();

        // Generate prediction delta
        if let Some(ref mut pr) = self.predictive_renderer {
            let snapshot = self.create_scene_snapshot();
            let delta = pr.delta_predictor.predict(&snapshot);
            
            // Update tile manager with delta
            pr.tile_manager.update(&delta);
            
            // Collect statistics
            let tile_stats = pr.tile_manager.statistics();
            self.stats.predictive_hot_ratio = 1.0 - tile_stats.hot_ratio;
            self.stats.gpu_time_saved_ms = pr.stats.gpu_time_saved_ms;
        }
    }

    /// Create scene snapshot for predictive renderer
    fn create_scene_snapshot(&self) -> SceneSnapshot {
        SceneSnapshot {
            camera_position: self.camera.position,
            camera_pitch: self.camera.pitch,
            camera_yaw: self.camera.yaw,
            screen_width: self.camera.width as u32,
            screen_height: self.camera.height as u32,
            entities: self.scene.entities.iter().map(|e| crate::network::EntitySnapshot {
                id: e.id,
                position: e.position,
                rotation: e.rotation,
                scale: e.scale,
                bounds_min: e.aabb.min,
                bounds_max: e.aabb.max,
            }).collect(),
            active_animations: HashMap::new(),
            particle_systems: Vec::new(),
            lighting_changes: Vec::new(),
        }
    }

    /// Main render loop
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // Prepare scene with predictive rendering
        self.prepare_scene();

        // Get hot tiles from predictive renderer
        let hot_tiles = self.predictive_renderer
            .as_ref()
            .map(|pr| pr.tile_manager.hot_tiles())
            .unwrap_or_default();

        // Update offload manager
        self.offload_manager.tick();

        // Update resource manager
        self.resource_manager.tick();

        // Get VRAM stats
        let (vram_used, vram_budget) = self.offload_manager.vram_usage();
        self.stats.vram_used_mb = vram_used as f64 / (1024.0 * 1024.0);
        self.stats.vram_budget_mb = vram_budget as f64 / (1024.0 * 1024.0);

        // Render scene
        self.renderer.render()
    }

    /// Handle window resize
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.camera.resize(new_size.width as f32, new_size.height as f32);
            self.renderer.resize(new_size.width, new_size.height);
            
            // Resize predictive renderer
            if let Some(ref mut pr) = self.predictive_renderer {
                pr.resize(new_size.width, new_size.height);
            }
        }
    }

    /// Get current engine statistics
    pub fn get_stats(&self) -> EngineStats {
        self.stats.clone()
    }

    /// Register a resource for tracking
    pub fn register_resource(&mut self, id: u64, size_bytes: usize, tier: ResourceTier) {
        use crate::offload::ResourceId;
        let rid = ResourceId(id, 1);
        self.offload_manager.register(rid, size_bytes, tier, Priority::Normal);
    }

    /// Touch a resource (marks it as recently used)
    pub fn touch_resource(&mut self, id: u64) {
        use crate::offload::ResourceId;
        let rid = ResourceId(id, 1);
        self.offload_manager.touch(rid);
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Cleanup
        log::info!("Engine shutting down...");
    }
}

// Re-export for convenience
pub use crate::network::EntitySnapshot;
use std::collections::HashMap;