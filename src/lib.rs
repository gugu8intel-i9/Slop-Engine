// src/lib.rs
//! Slop Engine - High-Performance WebGPU Game Engine
//! 
//! Integrated subsystems:
//! - Predictive Rendering (micro-tile re-rendering, frame reuse)
//! - Client-Side Prediction (network latency reduction)
//! - W-TinyLFU Memory Management (VRAM/RAM optimization)
//! - Dynamic LOD and Culling (CPU/GPU optimization)

use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

pub mod predictive_renderer;
pub mod offload;
pub mod network;
pub mod resource_manager;

use predictive_renderer::*;
use offload::{OffloadManager, OffloadConfig};
use network::{NetworkSystem, NetworkRole, SceneSnapshot, EntitySnapshot, AnimationSnapshot};
use resource_manager::{ResourceManager, ResourceConfig};

use glam::{Vec3, vec3};

// ============================================================================
// ENGINE CONFIGURATION
// ============================================================================

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub predictive_rendering: PredictiveRenderConfig,
    pub offload: OffloadConfig,
    pub network: NetworkConfig,
    pub resource: ResourceConfig,
}

#[derive(Debug, Clone)]
pub struct NetworkConfig {
    pub enabled: bool,
    pub role: NetworkRole,
    pub tick_rate: u32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            predictive_rendering: PredictiveRenderConfig::default(),
            offload: OffloadConfig::default(),
            network: NetworkConfig {
                enabled: false,
                role: NetworkRole::Client,
                tick_rate: 60,
            },
            resource: ResourceConfig::default(),
        }
    }
}

// ============================================================================
// UNIFIED ENGINE STATE
// ============================================================================

pub struct EngineState {
    // Core systems
    predictive_renderer: Option<PredictiveRenderer>,
    offload_manager: OffloadManager,
    network_system: NetworkSystem,
    resource_manager: Option<Arc<ResourceManager>>,
    
    // Scene data
    entities: Vec<EntitySnapshot>,
    active_animations: HashMap<u64, AnimationSnapshot>,
    
    // Camera
    camera_position: Vec3,
    camera_pitch: f32,
    camera_yaw: f32,
    
    // Runtime
    frame_count: u64,
    config: EngineConfig,
}

impl EngineState {
    pub fn new(config: EngineConfig) -> Self {
        Self {
            predictive_renderer: None,
            offload_manager: OffloadManager::new(config.offload.clone()),
            network_system: NetworkSystem::new(config.network.role),
            resource_manager: None,
            entities: Vec::new(),
            active_animations: HashMap::new(),
            camera_position: Vec3::ZERO,
            camera_pitch: 0.0,
            camera_yaw: 0.0,
            frame_count: 0,
            config,
        }
    }
    
    pub fn init_predictive_renderer(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.predictive_renderer = Some(PredictiveRenderer::new(
            device,
            self.config.predictive_rendering.clone(),
            width,
            height,
        ));
    }
    
    pub fn init_resource_manager(&mut self, device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) {
        self.resource_manager = Some(Arc::new(ResourceManager::new(device, queue, self.config.resource.clone())));
    }
    
    pub fn get_scene_snapshot(&self, screen_width: u32, screen_height: u32) -> SceneSnapshot {
        SceneSnapshot {
            camera_position: self.camera_position,
            camera_pitch: self.camera_pitch,
            camera_yaw: self.camera_yaw,
            screen_width,
            screen_height,
            entities: self.entities.clone(),
            active_animations: self.active_animations.clone(),
            particle_systems: Vec::new(),
            lighting_changes: Vec::new(),
        }
    }
    
    pub fn update_camera(&mut self, position: Vec3, pitch: f32, yaw: f32) {
        self.camera_position = position;
        self.camera_pitch = pitch;
        self.camera_yaw = yaw;
    }
    
    pub fn add_entity(&mut self, id: u64, position: Vec3, rotation: Vec3, scale: Vec3) {
        self.entities.push(EntitySnapshot {
            id,
            position,
            rotation,
            scale,
            bounds_min: position - scale * 0.5,
            bounds_max: position + scale * 0.5,
        });
    }
    
    pub fn tick(&mut self) {
        self.frame_count += 1;
        
        // Update offload manager
        self.offload_manager.tick();
        
        // Update resource manager
        if let Some(rm) = &self.resource_manager {
            rm.tick();
        }
    }
}

// ============================================================================
// WINIT APPLICATION HANDLER
// ============================================================================

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    run_with_config(EngineConfig::default()).await;
}

pub async fn run_with_config(config: EngineConfig) {
    // 1. Setup logging
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        let _ = console_log::init_with_level(log::Level::Warn);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = env_logger::try_init();
    }

    log::info!("Initializing Slop Engine...");
    
    let event_loop = EventLoop::new().expect("Failed to create event loop");

    // 2. Initialize WGPU
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("Failed to request wgpu adapter");

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Slop_Device"),
                required_features: wgpu::Features::empty(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: Default::default(),
            },
            None,
        )
        .await
        .expect("Failed to request wgpu device");

    log::info!("GPU initialized: {:?}", adapter.get_info());

    // 3. Create engine state
    let mut engine_state = EngineState::new(config);

    // 4. Setup App
    let mut app = EngineApp {
        instance,
        device: Arc::new(device),
        queue: Arc::new(queue),
        adapter,
        engine_state: Some(engine_state),
        window: None,
        surface: None,
        config: None,
        render_pipeline: None,
        depth_texture_view: None,
        hdr_texture_view: None,
        predictive_enabled: true,
    };

    // 5. Start
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(app);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        event_loop.run_app(&mut app).expect("Event loop crashed");
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run_native() {
    pollster::block_on(run());
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run_native_with_config(config: EngineConfig) {
    pollster::block_on(run_with_config(config));
}

// ============================================================================
// APP STATE
// ============================================================================

struct EngineApp {
    instance: wgpu::Instance,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    adapter: wgpu::Adapter,
    engine_state: Option<EngineState>,
    
    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    config: Option<wgpu::SurfaceConfiguration>,
    render_pipeline: Option<wgpu::RenderPipeline>,
    depth_texture_view: Option<wgpu::TextureView>,
    hdr_texture_view: Option<wgpu::TextureView>,
    predictive_enabled: bool,
}

impl ApplicationHandler for EngineApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }

        log::info!("Creating window...");
        
        let attrs = Window::default_attributes()
            .with_title("Slop Engine v2.0")
            .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720));
        let window = Arc::new(event_loop.create_window(attrs).expect("Window creation failed"));
        self.window = Some(window.clone());

        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowExtWebSys;
            let canvas = window.canvas().expect("Failed to get Canvas");
            let canvas_element = web_sys::Element::from(canvas);
            web_sys::window()
                .and_then(|win| win.document())
                .and_then(|doc| doc.body())
                .and_then(|body| body.append_child(&canvas_element).ok())
                .expect("Couldn't append canvas");
        }

        let surface = self.instance.create_surface(window.clone()).expect("Surface creation failed");
        let size = window.inner_size();
        let surface_caps = surface.get_capabilities(&self.adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&self.device, &config);

        // Initialize engine systems
        if let Some(ref mut state) = self.engine_state {
            state.init_resource_manager(self.device.clone(), self.queue.clone());
            state.init_predictive_renderer(&self.device, size.width, size.height);
            
            // Add some demo entities
            for i in 0..10 {
                state.add_entity(
                    i as u64,
                    vec3((i as f32 - 5.0) * 2.0, 0.0, -10.0),
                    Vec3::ZERO,
                    vec3(1.0, 1.0, 1.0),
                );
            }
        }

        // Create depth and HDR textures
        let depth_view = self.create_depth_texture(&config);
        let hdr_view = self.create_hdr_texture(&config);

        // Create render pipeline
        let pipeline = self.build_pipeline(config.format);

        self.surface = Some(surface);
        self.config = Some(config);
        self.depth_texture_view = Some(depth_view);
        self.hdr_texture_view = Some(hdr_view);
        self.render_pipeline = Some(pipeline);

        log::info!("Engine initialized successfully!");
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        let window = self.window.as_ref().unwrap();
        if window.id() != window_id { return; }

        match event {
            WindowEvent::CloseRequested => {
                log::info!("Shutting down...");
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                if physical_size.width > 0 && physical_size.height > 0 {
                    if let (Some(surface), Some(config)) = (
                        self.surface.as_mut(),
                        self.config.as_mut()
                    ) {
                        config.width = physical_size.width;
                        config.height = physical_size.height;
                        surface.configure(&self.device, config);
                        
                        // Resize predictive renderer
                        if let Some(ref mut state) = self.engine_state {
                            if let Some(ref mut pr) = state.predictive_renderer {
                                pr.resize(physical_size.width, physical_size.height);
                            }
                        }
                        
                        self.depth_texture_view = Some(self.create_depth_texture(config));
                        self.hdr_texture_view = Some(self.create_hdr_texture(config));
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.update();
                match self.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => {
                        if let Some(surface) = self.surface.as_ref() {
                            if let Some(config) = self.config.as_ref() {
                                surface.configure(&self.device, config);
                            }
                        }
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(e) => log::error!("{:?}", e),
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

// ============================================================================
// RENDER METHODS
// ============================================================================

impl EngineApp {
    fn update(&mut self) {
        if let Some(ref mut state) = self.engine_state {
            state.tick();
            
            // Simulate camera movement
            state.update_camera(
                vec3((state.frame_count as f32 * 0.01).sin() * 5.0, 2.0, -5.0),
                0.0,
                state.frame_count as f32 * 0.001,
            );
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let surface = self.surface.as_ref().ok_or(wgpu::SurfaceError::OutOfMemory)?;
        let pipeline = self.render_pipeline.as_ref().ok_or(wgpu::SurfaceError::OutOfMemory)?;
        let depth_view = self.depth_texture_view.as_ref().ok_or(wgpu::SurfaceError::OutOfMemory)?;
        let config = self.config.as_ref().ok_or(wgpu::SurfaceError::OutOfMemory)?;

        let output = surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Main Encoder"),
        });

        // Use predictive rendering if enabled
        if self.predictive_enabled {
            if let Some(ref mut state) = self.engine_state {
                if let Some(ref mut pr) = state.predictive_renderer {
                    let scene = state.get_scene_snapshot(config.width, config.height);
                    
                    // Get hot tiles info
                    let hot_tiles = pr.tile_manager.hot_tiles();
                    let stats = pr.tile_manager.statistics();
                    
                    // Log performance stats
                    if state.frame_count % 60 == 0 {
                        log::debug!(
                            "Predictive: {}/{} tiles hot ({:.1}%) | Saved: {:.1}ms/frame",
                            stats.hot_tiles,
                            stats.total_tiles,
                            stats.hot_ratio * 100.0,
                            pr.stats.gpu_time_saved_ms / 60.0
                        );
                    }
                }
            }
        }

        // Main render pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.04, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(pipeline);
            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn build_pipeline(&self, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
        let shader = self.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Main Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(MAIN_WGSL)),
        });

        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Main Pipeline"),
            layout: None,
            cache: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        })
    }

    fn create_depth_texture(&self, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
        let size = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }).create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn create_hdr_texture(&self, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
        let size = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HDR"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }).create_view(&wgpu::TextureViewDescriptor::default())
    }
}

// ============================================================================
// SHADERS
// ============================================================================

const MAIN_WGSL: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.5),
        vec2<f32>(-0.5, -0.5),
        vec2<f32>(0.5, -0.5)
    );
    let idx = in_vertex_index % 3u;
    return vec4<f32>(positions[idx], 0.5, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.2, 0.4, 0.8, 1.0);
}
"#;

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_config_default() {
        let config = EngineConfig::default();
        assert_eq!(config.network.tick_rate, 60);
        assert!(config.predictive_rendering.enabled);
    }

    #[test]
    fn test_scene_snapshot() {
        let state = EngineState::new(EngineConfig::default());
        let snapshot = state.get_scene_snapshot(1920, 1080);
        assert_eq!(snapshot.screen_width, 1920);
        assert_eq!(snapshot.screen_height, 1080);
    }
}