use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use wgpu::{
    BindGroup, BindGroupLayout, Buffer, CommandBuffer, ComputePipeline, Device, PipelineLayout,
    QuerySet, RenderPipeline, Sampler, SurfaceTexture, Texture, TextureView,
};
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyboardInput, MouseButton, MouseScrollDelta, WindowEvent};
use slotmap::{SlotMap, new_key_type};
use parking_lot::RwLock as PLRwLock; // Faster RwLock than std::sync

// Re-export key types
pub use wgpu_types as wgpu;

// --- Type-Safe Handles (Strongly Typed IDs) ---
// Using new_key_type prevents mixing up Entity IDs and Asset IDs
new_key_type! { pub struct EntityId; }
new_key_type! { pub struct TextureId; }
new_key_type! { pub struct MeshId; }
new_key_type! { pub struct MaterialId; }
new_key_type! { pub struct PipelineId; }

/// The Master State struct. 
/// In a real engine, you might split this into `MainWorld` (App/IO/Physics) 
/// and `RenderWorld` (GPU resources) to allow multi-threading.
/// Here we keep them unified but use interior mutability for thread safety.
pub struct State {
    pub app: AppState,
    pub time: TimeState,
    pub io: InputState,
    
    // --- Core Data ---
    // Using SlotMap for O(1) access, generation-based safety (prevents dangling pointers),
    // and dense memory layout.
    pub entities: SlotMap<EntityId, Entity>,
    pub transforms: SlotMap<EntityId, Transform>,
    pub velocities: SlotMap<EntityId, Velocity>,
    
    // --- Resources ---
    // In production, this is often a ResourceServer that streams from disk.
    pub textures: SlotMap<TextureId, wgpu::Texture>,
    pub samplers: SlotMap<TextureId, wgpu::Sampler>, // Usually 1:1 with texture
    pub meshes: SlotMap<MeshId, Mesh>,
    pub materials: SlotMap<MaterialId, Material>,
    
    // --- Rendering (GPU Side) ---
    pub gpu: GpuState,
    
    // --- Systems ---
    pub scene_graph: SceneGraph,
    pub physics_broadphase: BroadPhase,
    pub event_bus: EventBus,
}

impl State {
    pub async fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface: wgpu::Surface,
        adapter: wgpu::Adapter,
        config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        let caps = surface.get_capabilities(&adapter);
        
        Self {
            app: AppState::Running,
            time: TimeState::new(),
            io: InputState::default(),
            entities: SlotMap::new(),
            transforms: SlotMap::new(),
            velocities: SlotMap::new(),
            textures: SlotMap::new(),
            samplers: SlotMap::new(),
            meshes: SlotMap::new(),
            materials: SlotMap::new(),
            gpu: GpuState::new(device, queue, surface, config, caps),
            scene_graph: SceneGraph::default(),
            physics_broadphase: BroadPhase::new(),
            event_bus: EventBus::new(),
        }
    }

    /// Main update tick. Called once per frame on the main thread.
    pub fn update(&mut self) {
        self.time.tick();
        self.io.update();
        
        // 1. Process Events (Decoupled systems)
        self.event_bus.drain();

        // 2. Update Systems (Order matters)
        self.update_physics();
        self.update_animations();
        self.update_culling();
        
        // 3. Prepare for Render (Main thread -> Render thread sync happens here implicitly in wgpu)
        self.gpu.frame_index = (self.gpu.frame_index + 1) % self.gpu.frames_in_flight;
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width > 0 && size.height > 0 {
            self.gpu.resize(size);
            self.event_bus.push(SystemEvent::WindowResized(size));
        }
    }

    // --- System Logic Implementations ---

    fn update_physics(&mut self) {
        // High-performance iteration: Iterate over (Entity, Transform, Velocity) tuples.
        // This is "Structure of Arrays" friendly and cache-coherent.
        let dt = self.time.delta_seconds();
        
        // Join iterators (pseudo-code for logic)
        for (entity, transform) in self.transforms.iter_mut() {
            if let Some(vel) = self.velocities.get(entity) {
                transform.position += vel.linear * dt;
                transform.rotation = transform.rotation * vel.angular_quat(dt);
            }
        }
    }

    fn update_animations(&mut self) {
        // Skeletal animation update logic here
    }

    fn update_culling(&mut self) {
        // Update Frustum Culling based on camera transform
        if let Some(cam_ent) = self.scene_graph.active_camera {
            if let Some(cam_tf) = self.transforms.get(cam_ent) {
                // self.culler.update_frustum(&cam_tf.position, &cam_tf.rotation);
            }
        }
    }
    
    // --- Entity Factory ---
    pub fn create_entity(&mut self) -> EntityId {
        let id = self.entities.insert(Entity { 
            id, 
            active: true,
            tags: Vec::new() 
        });
        id
    }
    
    pub fn add_transform(&mut self, id: EntityId) -> &mut Transform {
        self.transforms.insert(id, Transform::default())
    }
}

// --- Sub-Structs ---

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppState { Running, Paused, Exiting }

#[derive(Debug, Default)]
pub struct InputState {
    pub keys_down: [bool; 256],
    pub keys_pressed: Vec<KeyboardInput>,
    pub mouse_pos: (f64, f64),
    pub mouse_delta: (f64, f64),
    pub scroll: f32,
}
impl InputState {
    fn update(&mut self) {
        self.keys_pressed.clear();
        self.scroll = 0.0;
    }
}

#[derive(Debug, Default)]
pub struct TimeState {
    start: Instant,
    last_frame: Instant,
    pub delta: Duration,
    pub elapsed: f32,
    pub frame_count: u64,
    pub fps: f32,
}
impl TimeState {
    fn new() -> Self { Self::default() }
    pub fn tick(&mut self) {
        let now = Instant::now();
        self.delta = now - self.last_frame;
        self.elapsed = (now - self.start).as_secs_f32();
        self.last_frame = now;
        self.frame_count += 1;
        // Update FPS logic...
    }
    pub fn delta_seconds(&self) -> f32 { self.delta.as_secs_f32() }
}

// --- High Performance ECS Components ---

#[derive(Debug, Clone, Copy)]
pub struct Entity {
    pub id: EntityId,
    pub active: bool,
    pub tags: Vec<u64>, // For grouping (e.g., "Renderable", "Physics")
}

#[derive(Debug, Clone, Copy)]
#[repr(C)] // Force specific memory layout
pub struct Transform {
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: glam::Vec3,
    pub world_matrix: glam::Mat4, // Pre-calculated for rendering
    pub is_dirty: bool,
}
impl Transform {
    pub fn default() -> Self {
        Self {
            position: glam::Vec3::ZERO,
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
            world_matrix: glam::Mat4::IDENTITY,
            is_dirty: true,
        }
    }
    pub fn update_matrix(&mut self) {
        self.world_matrix = glam::Mat4::from_scale_rotation_translation(
            self.scale, self.rotation, self.position
        );
        self.is_dirty = false;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Velocity {
    pub linear: glam::Vec3,
    pub angular: glam::Vec3, // Euler for simplicity, or Quat for complex
}
impl Velocity {
    fn angular_quat(&self, dt: f32) -> glam::Quat {
        glam::Quat::from_axis_angle(glam::Vec3::Y, self.angular.y * dt)
    }
}

// --- GPU State ---
// This struct holds ONLY wgpu resources. It should ideally live on the render thread
// or be accessed via Arc<Mutex> if shared.
#[derive(Debug)]
pub struct GpuState {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface,
    pub config: wgpu::SurfaceConfiguration,
    pub format: wgpu::TextureFormat,
    
    // Frame Management
    pub frame_index: usize,
    pub frames_in_flight: usize,
    
    // Render Targets
    pub depth_texture: wgpu::Texture,
    pub msaa_texture: wgpu::Texture,
    
    // Pipelines & Layouts
    pub pipeline_layouts: SlotMap<PipelineId, wgpu::PipelineLayout>,
    pub render_pipelines: SlotMap<PipelineId, wgpu::RenderPipeline>,
    pub bind_group_layouts: SlotMap<u32, wgpu::BindGroupLayout>,
    
    // Sync
    pub fence: wgpu::Fence,
    pub fence_values: Vec<u64>,
    
    // Performance Queries
    pub timestamp_query_set: wgpu::QuerySet,
    
    // Global Bind Groups (e.g., Camera UBO)
    pub global_bind_group: wgpu::BindGroup,
    
    // Render Pass Descriptors (Cached)
    pub main_pass_desc: wgpu::RenderPassDescriptor<'static>,
    pub shadow_pass_desc: wgpu::RenderPassDescriptor<'static>,
    pub post_process_desc: wgpu::RenderPassDescriptor<'static>,
}

impl GpuState {
    fn new(
        device: wgpu::Device, 
        queue: wgpu::Queue, 
        surface: wgpu::Surface, 
        config: &wgpu::SurfaceConfiguration,
        caps: &wgpu::SurfaceCapabilities
    ) -> Self {
        let format = config.format;
        let size = config.size;
        
        let depth_texture = Self::create_depth_texture(&device, size, "Depth");
        let msaa_texture = Self::create_msaa_texture(&device, size, 4, format);
        
        let mut state = Self {
            device, queue, surface, config: config.clone(), format,
            frame_index: 0, frames_in_flight: 2,
            depth_texture, msaa_texture,
            pipeline_layouts: SlotMap::new(), render_pipelines: SlotMap::new(),
            bind_group_layouts: SlotMap::new(),
            fence: device.create_fence(&wgpu::FenceValue::INIT),
            fence_values: vec![0; 2],
            timestamp_query_set: device.create_query_set(&wgpu::QuerySetDescriptor {
                ty: wgpu::QueryType::Timestamp, count: 2, ..Default::default()
            }),
            global_bind_group: unimplemented!(), // Create actual UBO later
            main_pass_desc: unimplemented!(),
            shadow_pass_desc: unimplemented!(),
            post_process_desc: unimplemented!(),
        };
        
        // Initialize descriptors
        state.main_pass_desc = state.create_main_pass(None);
        state.shadow_pass_desc = state.create_shadow_pass(None);
        state.post_process_desc = state.create_post_process_pass(None);
        
        state
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        self.config.width = size.width;
        self.config.height = size.height;
        self.depth_texture = Self::create_depth_texture(&self.device, size, "Depth");
        self.msaa_texture = Self::create_msaa_texture(&self.device, size, 4, self.format);
        // Recreate pass descriptors
        self.main_pass_desc = self.create_main_pass(None);
        // etc...
    }

    fn create_depth_texture(device: &wgpu::Device, size: PhysicalSize<u32>, label: &str) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: size.width, height: size.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }
    
    fn create_msaa_texture(device: &wgpu::Device, size: PhysicalSize<u32>, samples: u32, format: wgpu::TextureFormat) -> wgpu::Texture {
         device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MSAA Texture"),
            size: wgpu::Extent3d { width: size.width, height: size.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }

    fn create_main_pass(&self, view: Option<&wgpu::TextureView>) -> wgpu::RenderPassDescriptor<'static> {
        wgpu::RenderPassDescriptor {
            label: Some("Main Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: view.unwrap_or(&self.msaa_texture.create_view()), // Placeholder
                resolve_target: Some(wgpu::TextureView::default()), // Will be swapchain
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture.create_view(),
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            ..Default::default()
        }
    }
    
    fn create_shadow_pass(&self, view: Option<&wgpu::TextureView>) -> wgpu::RenderPassDescriptor<'static> { 
        /* Implementation similar to above but for shadow maps */ 
        unimplemented!() 
    }
    
    fn create_post_process_pass(&self, view: Option<&wgpu::TextureView>) -> wgpu::RenderPassDescriptor<'static> { 
        /* Implementation for Bloom/TAA */ 
        unimplemented!() 
    }
}

// --- Scene & Physics ---

#[derive(Debug, Default)]
pub struct SceneGraph {
    pub root: EntityId,
    pub active_camera: Option<EntityId>,
    pub directional_light: Option<EntityId>,
    pub point_lights: Vec<EntityId>,
}

#[derive(Debug, Default)]
pub struct BroadPhase {
    // Axis Aligned Bounding Boxes
    pub aabbs: SlotMap<EntityId, AABB>,
    // Spatial partitioning grid or BVH
}

#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: glam::Vec3,
    pub max: glam::Vec3,
}

// --- Assets ---

#[derive(Debug)]
pub struct Mesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub vertex_layout: wgpu::VertexBufferLayout<'static>,
}

#[derive(Debug)]
pub struct Material {
    pub shader_id: PipelineId,
    pub bind_group: wgpu::BindGroup, // Per-material uniforms/textures
    pub albedo: [f32; 4],
    pub roughness: f32,
    pub metallic: f32,
}

// --- Event Bus ---

#[derive(Debug, Clone)]
pub enum SystemEvent {
    WindowResized(PhysicalSize<u32>),
    EntityCreated(EntityId),
    EntityDestroyed(EntityId),
}

#[derive(Debug, Default)]
pub struct EventBus {
    queue: crossbeam_channel::Sender<SystemEvent>,
    receiver: crossbeam_channel::Receiver<SystemEvent>,
}

impl EventBus {
    pub fn new() -> Self {
        let (tx, rx) = crossbeam_channel::unbounded();
        Self { queue: tx, receiver: rx }
    }
    
    pub fn push(&self, event: SystemEvent) {
        let _ = self.queue.send(event);
    }
    
    pub fn drain(&self) {
        while let Ok(evt) = self.receiver.try_recv() {
            // Process event (e.g., log it, trigger reactions)
            // In a real engine, this might trigger a system to run
            match evt {
                SystemEvent::WindowResized(s) => log::info!("Resized to {:?}", s),
                _ => {}
            }
        }
    }
}
