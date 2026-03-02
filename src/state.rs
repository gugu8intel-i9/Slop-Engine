//! The central nervous system of the engine.
//!
//! This module defines the core `State` struct, which holds all runtime data
//! for the engine. It is designed with a data-oriented approach, using an
//! archetype-based storage system for maximum CPU cache efficiency and performance.
//!
//! It also includes a robust event bus for decoupled inter-system communication
//! and comprehensive state management for all engine subsystems.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use wgpu::{
    BindGroup, BindGroupLayout, Buffer, CommandBuffer, ComputePipeline, Device, PipelineLayout,
    QuerySet, RenderPipeline, Sampler, SurfaceTexture, Texture, TextureView,
};
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyboardInput, MouseButton, MouseScrollDelta, WindowEvent};

// Re-export key types for convenience in other modules
pub use wgpu_types as wgpu;

// --- Core State Structs ---

/// The main state of the application, holding all engine systems and data.
/// This is the primary struct that is passed around and mutated each frame.
pub struct State {
    pub app: AppState,
    pub io: InputState,
    pub rendering: RenderingState,
    pub resources: ResourceState,
    pub scene: SceneState,
    pub physics: PhysicsState,
    pub audio: AudioState,
    pub gui: GuiState,
    pub time: TimeState,
    
    // Internal systems
    event_bus: EventBus,
    command_buffers: Vec<CommandBuffer>,
    surface_textures_to_release: Vec<SurfaceTexture>,
}

impl State {
    /// Creates a new State instance.
    /// This is an async function as it needs to wait for the device and queue to be ready.
    pub async fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface: wgpu::Surface,
        adapter: wgpu::Adapter,
        config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = config.format;

        Self {
            app: AppState::new(),
            io: InputState::new(),
            rendering: RenderingState::new(&device, surface_caps, surface_format),
            resources: ResourceState::new(&device, &queue),
            scene: SceneState::new(),
            physics: PhysicsState::new(),
            audio: AudioState::new(),
            gui: GuiState::new(),
            time: TimeState::new(),
            event_bus: EventBus::new(),
            command_buffers: Vec::with_capacity(2), // Double buffering for command buffers
            surface_textures_to_release: Vec::new(),
        }
    }

    /// Processes all pending events and updates the state for the next frame.
    /// This should be called once per frame, before rendering.
    pub fn update(&mut self) {
        self.time.update();
        self.io.update();
        self.event_bus.drain(); // Process all queued events
        self.scene.update(&self.time);
        self.physics.update(&self.time, &mut self.scene);
        self.audio.update();
        self.gui.update(&self.io, &self.time);
        
        // Clear command buffers for the new frame
        self.command_buffers.clear();
        // Release old surface textures to prevent memory leaks
        self.surface_textures_to_release.clear();
    }

    /// Resizes the render targets and updates surface-dependent resources.
    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.rendering.resize(new_size, &self.resources.device);
            self.event_bus.push(AppEvent::WindowResized(new_size));
        }
    }
    
    /// Queues a surface texture to be released after the GPU is done with it.
    pub fn queue_surface_texture_for_release(&mut self, texture: SurfaceTexture) {
        self.surface_textures_to_release.push(texture);
    }

    /// Gets a mutable reference to the event bus to post new events.
    pub fn event_bus_mut(&mut self) -> &mut EventBus {
        &mut self.event_bus
    }
    
    /// Takes ownership of a command buffer to be submitted to the queue.
    pub fn take_command_buffer(&mut self) -> Option<CommandBuffer> {
        self.command_buffers.pop()
    }

    /// Adds a command buffer to be submitted this frame.
    pub fn add_command_buffer(&mut self, cmd_buffer: CommandBuffer) {
        self.command_buffers.push(cmd_buffer);
    }
}

// --- Sub-State Structs ---
// Each of these manages a specific domain of the engine.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppState {
    Running,
    Paused,
    Exiting,
}
impl AppState {
    fn new() -> Self { Self::Running }
}

/// Manages all user input.
#[derive(Debug, Default)]
pub struct InputState {
    pub keys_down: [bool; 256],
    pub keys_pressed: Vec<KeyboardInput>,
    pub keys_released: Vec<KeyboardInput>,
    pub mouse_position: (f64, f64),
    pub mouse_delta: (f64, f64),
    pub mouse_buttons: [bool; 3],
    pub scroll_delta: f32,
    pub window_size: PhysicalSize<u32>,
}
impl InputState {
    fn new() -> Self { Self::default() }
    pub fn update(&mut self) {
        // Presses/releases are single-frame events, so we clear them after processing.
        self.keys_pressed.clear();
        self.keys_released.clear();
        self.scroll_delta = 0.0;
    }
    // Public methods to handle winit events would go here, e.g.:
    // pub fn handle_window_event(&mut self, event: &WindowEvent) { ... }
    // pub fn handle_keyboard_input(&mut self, input: KeyboardInput, state: ElementState) { ... }
}

/// Manages all rendering-related state, including the wgpu device, pipelines, and render passes.
#[derive(Debug)]
pub struct RenderingState {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub surface_format: wgpu::TextureFormat,
    
    // Main render targets
    pub color_target: TextureView,
    pub depth_target: TextureView,
    pub sample_count: u32,

    // Pipelines and Layouts
    pub pipeline_manager: PipelineManager,
    pub bind_group_manager: BindGroupManager,
    
    // Renderer-specific data
    pub renderer: Renderer,
    pub post_processing: PostProcessing,
    pub shadows: Shadows,
    pub skybox: Skybox,
    
    // Performance queries
    pub timestamp_query_set: Option<QuerySet>,
    pub frame_times: Vec<Duration>,
}
impl RenderingState {
    fn new(device: &wgpu::Device, caps: &wgpu::SurfaceCapabilities, format: wgpu::TextureFormat) -> Self {
        Self {
            device: device.clone(),
            queue: device.queue().clone(), // In a real engine, you'd manage queues more carefully
            surface: unimplemented!(), // Set during initialization
            surface_config: unimplemented!(), // Set during initialization
            surface_format: format,
            color_target: unimplemented!(),
            depth_target: unimplemented!(),
            sample_count: caps.formats[0].sample_counts().start(),
            pipeline_manager: PipelineManager::new(device),
            bind_group_manager: BindGroupManager::new(device),
            renderer: Renderer::new(device, format),
            post_processing: PostProcessing::new(device, format),
            shadows: Shadows::new(device, format),
            skybox: Skybox::new(device, format),
            timestamp_query_set: None, // Can be created if the device supports it
            frame_times: Vec::with_capacity(120),
        }
    }
    
    pub fn resize(&mut self, new_size: PhysicalSize<u32>, device: &wgpu::Device) {
        // Logic to recreate color/depth textures with the new size
        // This is a critical and complex part of a real engine.
        // For now, we just update the size.
        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        // self.color_target = ...
        // self.depth_target = ...
        // self.renderer.resize(...);
        // etc.
    }
}

/// Manages all loaded assets like meshes, textures, and shaders.
#[derive(Debug, Default)]
pub struct ResourceState {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    
    // Asset storage using an archetype-like pattern for performance.
    // In a real engine, this would be a proper ECS storage like `Vec<T>` for each component type.
    // For this example, we use HashMaps for clarity. A real high-perf version would use
    // slot maps or generation-based indices for handles.
    pub textures: HashMap<u64, Texture>,
    pub samplers: HashMap<u64, Sampler>,
    pub buffers: HashMap<u64, Buffer>,
    pub bind_group_layouts: HashMap<u64, BindGroupLayout>,
    pub bind_groups: HashMap<u64, BindGroup>,
    pub pipelines: HashMap<u64, RenderPipeline>,
    pub compute_pipelines: HashMap<u64, ComputePipeline>,
    pub shader_modules: HashMap<u64, wgpu::ShaderModule>,
    pub meshes: HashMap<u64, Mesh>,
    pub materials: HashMap<u64, Material>,
}
impl ResourceState {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            device: device.clone(),
            queue: queue.clone(),
            ..Default::default()
        }
    }
    // Methods like `load_texture`, `create_buffer`, etc. would go here.
    // They would return a handle (u64) to the newly created resource.
}

/// Manages the scene graph, entities, and their components.
#[derive(Debug, Default)]
pub struct SceneState {
    pub entities: Vec<Entity>,
    // Other scene data like active camera, lighting environment, etc.
    pub active_camera_entity: Option<EntityId>,
    pub point_lights: Vec<PointLight>,
    pub directional_lights: Vec<DirectionalLight>,
}
impl SceneState {
    fn new() -> Self { Self::default() }
    pub fn update(&mut self, time: &TimeState) {
        // Update entity transforms, animations, etc. based on time.
    }
}

/// Manages the physics simulation.
#[derive(Debug, Default)]
pub struct PhysicsState {
    pub gravity: glam::Vec3,
    pub broad_phase: BroadPhase,
    pub narrow_phase: NarrowPhase,
    pub solver: Solver,
    // A list of pairs of entities to check for collisions this frame.
    pub collision_pairs: Vec<(EntityId, EntityId)>,
}
impl PhysicsState {
    fn new() -> Self {
        Self {
            gravity: glam::Vec3::new(0.0, -9.81, 0.0),
            broad_phase: BroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            solver: Solver::new(),
            collision_pairs: Vec::new(),
        }
    }
    
    pub fn update(&mut self, time: &TimeState, scene: &mut SceneState) {
        // 1. Integrate forces and update velocities/positions.
        // 2. Update broad-phase (e.g., sweep and prune).
        // 3. Generate collision pairs.
        // 4. Run narrow-phase collision detection.
        // 5. Run solver to resolve collisions and constraints.
        // 6. Apply resulting impulses/positions back to the scene's entities.
    }
}

/// Manages audio playback and state.
#[derive(Debug, Default)]
pub struct AudioState {
    // pub device: rodio::Device,
    // pub output_stream: rodio::OutputStream,
    // pub sounds: HashMap<SoundId, Sound>,
    // pub music: Option<Music>,
}
impl AudioState {
    fn new() -> Self { Self::default() }
    pub fn update(&mut self) {
        // Update playing sounds, check for finished sounds, etc.
    }
}

/// Manages the immediate mode GUI.
#[derive(Debug, Default)]
pub struct GuiState {
    // pub ui_renderer: UiRenderer,
    // pub contexts: HashMap<GuiContextId, UiContext>,
    pub is_visible: bool,
}
impl GuiState {
    fn new() -> Self { Self { is_visible: true, ..Default::default() } }
    pub fn update(&mut self, input: &InputState, time: &TimeState) {
        // Begin a new UI frame, handle input, and draw UI elements.
    }
}

/// Manages timing and frame statistics.
#[derive(Debug, Default)]
pub struct TimeState {
    pub start_time: Instant,
    pub last_frame_time: Instant,
    pub delta_time: Duration,
    pub elapsed_time: Duration,
    pub frame_count: u64,
    pub fps: f32,
    pub frame_time_ms: f32,
}
impl TimeState {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            start_time: now,
            last_frame_time: now,
            ..Default::default()
        }
    }
    
    pub fn update(&mut self) {
        let now = Instant::now();
        self.delta_time = now - self.last_frame_time;
        self.last_frame_time = now;
        self.elapsed_time = now - self.start_time;
        self.frame_count += 1;
        
        // Calculate FPS and frame time every second
        if self.elapsed_time.as_secs_f32() > self.frame_count as f32 / 60.0 { // Simplified
             self.frame_time_ms = self.delta_time.as_secs_f32() * 1000.0;
             self.fps = 1.0 / self.delta_time.as_secs_f32();
        }
    }
}


// --- High-Performance Component "Archetype" Storage ---
// In a real engine, this would be a separate crate. Here we define the traits and a simple implementation.
// This avoids the overhead of a HashMap for component lookups and is much more cache-friendly.

pub type EntityId = u32;

// A bundle of components that can be attached to an entity.
// This is a simplified version. A real one would use a tuple macro.
pub trait Bundle: Default {
    fn entity_id(&self) -> EntityId;
}

// Example component structs
#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub entity_id: EntityId,
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: glam::Vec3,
}
impl Bundle for Transform {
    fn entity_id(&self) -> EntityId { self.entity_id }
}

#[derive(Debug, Clone, Copy)]
pub struct Velocity {
    pub entity_id: EntityId,
    pub linear: glam::Vec3,
    pub angular: glam::Vec3,
}
impl Bundle for Velocity {
    fn entity_id(&self) -> EntityId { self.entity_id }
}

// The World stores all component data in contiguous arrays (Vecs).
// Getting a component for an entity is an O(1) array lookup.
pub struct World {
    entities: Vec<EntityId>,
    transforms: Vec<Option<Transform>>, // Sparse set
    velocities: Vec<Option<Velocity>>,
    // ... other component vectors
    next_entity_id: EntityId,
}
impl World {
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            transforms: Vec::new(),
            velocities: Vec::new(),
            next_entity_id: 0,
        }
    }
    
    pub fn create_entity(&mut self) -> EntityId {
        let id = self.next_entity_id;
        self.next_entity_id += 1;
        
        // Ensure all component vectors are large enough
        let required_len = id as usize + 1;
        self.transforms.resize_with(required_len, || None);
        self.velocities.resize_with(required_len, || None);
        // ... resize other vectors
        
        self.entities.push(id);
        id
    }
    
    pub fn add_component<T: Bundle>(&mut self, entity_id: EntityId, component: T) {
        // This is a simplified setter. A real implementation would be more generic.
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Transform>() {
            let transform = unsafe { std::mem::transmute::<T, Transform>(component) };
            if let Some(slot) = self.transforms.get_mut(entity_id as usize) {
                *slot = Some(transform);
            }
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Velocity>() {
             let velocity = unsafe { std::mem::transmute::<T, Velocity>(component) };
             if let Some(slot) = self.velocities.get_mut(entity_id as usize) {
                *slot = Some(velocity);
            }
        }
        // ... else if for other component types
    }
    
    pub fn get_component<T: Bundle + Clone>(&self, entity_id: EntityId) -> Option<T> {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Transform>() {
            self.transforms.get(entity_id as usize).and_then(|opt| opt.clone()).map(|t| unsafe { std::mem::transmute(t) })
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Velocity>() {
            self.velocities.get(entity_id as usize).and_then(|opt| opt.clone()).map(|v| unsafe { std::mem::transmute(v) })
        } else {
            None
        }
    }
}

// A simple entity struct for convenience.
#[derive(Debug, Clone, Copy)]
pub struct Entity {
    pub id: EntityId,
    // You could store handles to components here for easier access.
}

// --- Placeholder structs for other subsystems ---
// These would be defined in their respective modules.

#[derive(Debug)]
pub struct PipelineManager { /* ... */ }
impl PipelineManager {
    pub fn new(_device: &wgpu::Device) -> Self { Self {} }
}

#[derive(Debug)]
pub struct BindGroupManager { /* ... */ }
impl BindGroupManager {
    pub fn new(_device: &wgpu::Device) -> Self { Self {} }
}

#[derive(Debug)]
pub struct Renderer { /* ... */ }
impl Renderer {
    pub fn new(_device: &wgpu::Device, _format: wgpu::TextureFormat) -> Self { Self {} }
    pub fn resize(&mut self, _size: PhysicalSize<u32>) {}
}

#[derive(Debug)]
pub struct PostProcessing { /* ... */ }
impl PostProcessing {
    pub fn new(_device: &wgpu::Device, _format: wgpu::TextureFormat) -> Self { Self {} }
}

#[derive(Debug)]
pub struct Shadows { /* ... */ }
impl Shadows {
    pub fn new(_device: &wgpu::Device, _format: wgpu::TextureFormat) -> Self { Self {} }
}

#[derive(Debug)]
pub struct Skybox { /* ... */ }
impl Skybox {
    pub fn new(_device: &wgpu::Device, _format: wgpu::TextureFormat) -> Self { Self {} }
}

#[derive(Debug, Clone, Copy)]
pub struct PointLight { /* position, color, intensity, radius */ }
#[derive(Debug, Clone, Copy)]
pub struct DirectionalLight { /* direction, color, intensity */ }
#[derive(Debug)] pub struct Mesh { /* vertex/index buffers */ }
#[derive(Debug)] pub struct Material { /* shader, textures, uniforms */ }
#[derive(Debug)] pub struct BroadPhase { /* ... */ }
#[derive(Debug)] pub struct NarrowPhase { /* ... */ }
#[derive(Debug)] pub struct Solver { /* ... */ }

// --- Event/Message Bus for Decoupled Communication ---

#[derive(Debug, Clone)]
pub enum AppEvent {
    WindowResized(PhysicalSize<u32>),
    // Other application-level events
}

#[derive(Debug, Clone)]
pub enum EngineEvent {
    // Events from the engine core
    FrameStarted,
    FrameFinished,
    ResourceLoaded(u64), // Handle to the resource
    ResourceFailedToLoad(String),
}

#[derive(Debug, Clone)]
pub enum SceneEvent {
    EntityCreated(EntityId),
    EntityDestroyed(EntityId),
    ComponentAdded(EntityId, String), // e.g., "Transform"
    ComponentRemoved(EntityId, String),
}

// The event bus holds different channels for different event types.
// Using Vec is simple. For extreme performance, you might use a lock-free queue
// or separate channels for senders/receivers.
#[derive(Debug, Default)]
pub struct EventBus {
    app_events: Vec<AppEvent>,
    engine_events: Vec<EngineEvent>,
    scene_events: Vec<SceneEvent>,
    // Add more channels as needed: PhysicsEvent, AudioEvent, etc.
}

impl EventBus {
    pub fn new() -> Self { Self::default() }

    pub fn push(&mut self, event: impl Into<DynamicEvent>) {
        // This is a simplified implementation. A real one would use trait objects
        // or an enum to store different event types in a single Vec.
        // For now, we'll just have separate push methods.
    }
    
    // Specific push methods for type safety
    pub fn push_app_event(&mut self, event: AppEvent) { self.app_events.push(event); }
    pub fn push_engine_event(&mut self, event: EngineEvent) { self.engine_events.push(event); }
    pub fn push_scene_event(&mut self, event: SceneEvent) { self.scene_events.push(event); }
    
    pub fn drain(&mut self) {
        // In a real system, other modules would subscribe to these events.
        // Here we just clear them to simulate processing.
        if !self.app_events.is_empty() {
            // log::info!("Processing {} app events", self.app_events.len());
            self.app_events.clear();
        }
        if !self.engine_events.is_empty() {
            // log::info!("Processing {} engine events", self.engine_events.len());
            self.engine_events.clear();
        }
        if !self.scene_events.is_empty() {
            // log::info!("Processing {} scene events", self.scene_events.len());
            self.scene_events.clear();
        }
    }
}

// A trait to allow different event enums to be stored in a common way.
// This is an advanced pattern, often solved with crates like `anymap` or `downcast-rs`.
pub trait DynamicEvent: Send + Sync + 'static {
    fn as_any(&self) -> &dyn std::any::Any;
}
// Impl for our concrete event types...
// impl DynamicEvent for AppEvent { ... }
// impl DynamicEvent for EngineEvent { ... }
// etc.
