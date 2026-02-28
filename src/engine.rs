use std::sync::Arc;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

// Import all the amazing modules you've built
use crate::{
    anticheat::AntiCheatSystem,
    audio::AudioEngine,
    camera::Camera,
    camera_controller::CameraController,
    context::GraphicsContext,
    culling::FrustumCuller,
    fps_counter::FpsCounter,
    gui::GuiSystem,
    input_system::InputSystem,
    physics::PhysicsWorld,
    physics_integration::PhysicsSync,
    renderer::Renderer,
    resource_manager::ResourceManager,
    scene::Scene,
    shader_hot_reload::ShaderHotReloader,
    time::Time,
};

/// The Core Engine orchestrator.
/// Designed for high performance, utilizing fixed-timestep logic for physics/anticheat
/// and variable-timestep for rendering and input.
pub struct Engine {
    // Windowing & Context
    pub window: Arc<Window>,
    pub context: Arc<GraphicsContext>,

    // Core Systems
    pub renderer: Renderer,
    pub scene: Scene,
    pub physics: PhysicsWorld,
    pub resources: Arc<ResourceManager>,
    
    // Utilities & Features
    pub input: InputSystem,
    pub camera: Camera,
    pub camera_controller: CameraController,
    pub audio: AudioEngine,
    pub gui: GuiSystem,
    pub time: Time,
    pub fps_counter: FpsCounter,
    
    // Advanced/Security
    pub anticheat: AntiCheatSystem,
    pub shader_reloader: Option<ShaderHotReloader>,

    // Offloaded/Threaded tasks (e.g., async asset loading, background generation)
    pub task_pool: crate::offload::TaskPool,
}

impl Engine {
    /// Initializes the engine. Asynchronous to support WebGPU/WASM initialization.
    pub async fn new(event_loop: &EventLoop<()>) -> Result<Self, crate::error::EngineError> {
        // 1. Initialize Window & Context
        let window = Arc::new(
            WindowBuilder::new()
                .with_title("High-Performance Custom Engine")
                .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720))
                .build(event_loop)
                .unwrap(),
        );

        let context = Arc::new(GraphicsContext::new(window.clone()).await?);
        
        // 2. Initialize Shared Resource Manager (Assets, Textures, Meshes)
        let resources = Arc::new(ResourceManager::new(context.clone()));

        // 3. Initialize Core Systems
        let mut renderer = Renderer::new(context.clone(), resources.clone());
        let scene = Scene::new();
        let physics = PhysicsWorld::new();
        let input = InputSystem::new();
        let audio = AudioEngine::new()?;
        let gui = GuiSystem::new(&window, &context);
        
        // 4. Setup Camera
        let camera = Camera::new(context.config().width as f32, context.config().height as f32);
        let camera_controller = CameraController::new(2.0, 0.5); // Speed, Sensitivity

        // 5. Setup Utilities
        let time = Time::new();
        let fps_counter = FpsCounter::new();
        let task_pool = crate::offload::TaskPool::new(); // Thread pool for heavy tasks
        let anticheat = AntiCheatSystem::new();
        
        // Setup Hot Reloading (Only in debug/development builds)
        #[cfg(debug_assertions)]
        let shader_reloader = Some(ShaderHotReloader::new(context.clone())?);
        #[cfg(not(debug_assertions))]
        let shader_reloader = None;

        Ok(Self {
            window,
            context,
            renderer,
            scene,
            physics,
            resources,
            input,
            camera,
            camera_controller,
            audio,
            gui,
            time,
            fps_counter,
            anticheat,
            shader_reloader,
            task_pool,
        })
    }

    /// Handles OS-level window and input events.
    pub fn handle_event(&mut self, event: &Event<()>) {
        // Let GUI intercept events first (e.g., if user is typing in a UI box)
        if self.gui.handle_event(event) {
            return; 
        }

        // Pass events to the Input System
        self.input.process_event(event);
    }

    /// The main update loop. Separates Fixed-Update (Physics/Logic) and Variable-Update (Graphics).
    pub fn update(&mut self) {
        self.time.tick();
        self.fps_counter.update(self.time.delta_seconds());

        // 1. Hot Reloading (Dev only)
        if let Some(reloader) = &mut self.shader_reloader {
            reloader.check_and_reload(&mut self.renderer);
        }

        // 2. Variable Timestep Update (Runs every frame)
        self.camera_controller.update_camera(&mut self.camera, &self.input, self.time.delta_seconds());
        self.audio.update(&self.camera); // Update listener position
        self.gui.update(self.time.delta_seconds());

        // 3. Fixed Timestep Update (Physics, Anticheat, Gameplay Logic)
        // Ensures deterministic behavior regardless of framerate
        while self.time.consume_fixed_timestep() {
            self.fixed_update();
        }

        // 4. Interpolate physics state to rendering scene for ultra-smooth movement
        PhysicsSync::sync_to_scene(&self.physics, &mut self.scene, self.time.interpolation_alpha());
        
        // Clear input delta states (like mouse movement) at the end of the frame
        self.input.clear_frame_state();
    }

    /// Deterministic logic step (e.g., runs exactly 60 times a second)
    fn fixed_update(&mut self) {
        let dt = self.time.fixed_delta_seconds();

        // Security / Anticheat checks (Memory validation, speedhack checks)
        self.anticheat.verify_integrity(dt);

        // Step Physics Engine
        self.physics.step(dt);

        // Update Scene Logic / Animations / Entity scripts
        self.scene.update(dt);
    }

    /// The highly-optimized rendering pipeline.
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // 1. Frustum Culling (High-Performance Offloaded Task)
        // Only render what the camera can see.
        let visible_entities = FrustumCuller::cull(&self.scene, &self.camera);

        // 2. Prepare Render State
        self.renderer.prepare(&self.camera, &visible_entities, &self.time);

        // 3. Acquire Swapchain Texture
        let output = self.context.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // 4. Command Buffer Generation (Utilizing command_buffer.rs)
        let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Engine Render Encoder"),
        });

        // ==========================================
        // ADVANCED RENDER PIPELINE
        // ==========================================
        
        // A. Shadow Mapping (Cascaded / Depth Textures)
        self.renderer.render_shadows(&mut encoder, &self.scene, &visible_entities);

        // B. Main Opaque Geometry Pass (PBR / Materials)
        self.renderer.render_opaque(&mut encoder, &view, &self.scene, &visible_entities);

        // C. Skybox / Environment Prefiltering
        self.renderer.render_skybox(&mut encoder, &view, &self.camera);

        // D. Post Processing (SSR -> SSAO -> DoF -> Bloom -> TAA)
        // These utilize your dedicated compute/frag wgsl shaders.
        self.renderer.post_processing.apply(
            &mut encoder, 
            &view, 
            &self.context,
            &self.camera
        );

        // E. Render GUI on top of everything
        self.gui.render(&mut encoder, &view, &self.context);

        // 5. Submit to GPU Queue
        self.context.queue.submit(std::iter::once(encoder.finish()));
        
        // 6. Present to Screen
        output.present();

        Ok(())
    }

    /// Handles window resizing efficiently
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.context.resize(new_size);
            self.camera.resize(new_size.width as f32, new_size.height as f32);
            self.renderer.resize(new_size.width, new_size.height); // Resizes depth/post-process targets
        }
    }
}
