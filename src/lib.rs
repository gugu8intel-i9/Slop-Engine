// src/lib.rs
//! Slop Engine - single-file, compatible, high-performance lib.rs shim
//! - Works on native and wasm32
//! - Minimal, fast renderer stub using wgpu + winit
//! - Internal modules provided inline to avoid missing-file errors
//! - Uses bytemuck derive macros; ensure bytemuck = { features = ["derive"] } in Cargo.toml

#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

use std::sync::Arc;

use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// ---------- Small utility types ----------
pub type Float3 = [f32; 3];
pub type Float4 = [f32; 4];

// ---------- Camera module (self-contained) ----------
pub mod camera {
    use bytemuck::{Pod, Zeroable};
    use glam::{Mat4, Vec3};

    /// Camera uniform layout for GPU (matches WGSL)
    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable)]
    pub struct CameraUniform {
        pub view_proj: [[f32; 4]; 4],
        pub view: [[f32; 4]; 4],
        pub proj: [[f32; 4]; 4],
        pub position: [f32; 4],
        pub forward: [f32; 4],
        pub up: [f32; 4],
        pub params: [f32; 4],
    }

    impl CameraUniform {
        pub fn from_matrices(view: Mat4, proj: Mat4, pos: Vec3, mode_id: f32) -> Self {
            let view_proj = proj * view;
            Self {
                view_proj: view_proj.to_cols_array_2d(),
                view: view.to_cols_array_2d(),
                proj: proj.to_cols_array_2d(),
                position: [pos.x, pos.y, pos.z, 0.0],
                forward: [-(view.z_axis.x), -(view.z_axis.y), -(view.z_axis.z), 0.0],
                up: [view.y_axis.x, view.y_axis.y, view.y_axis.z, 0.0],
                params: [mode_id, 0.0, 0.0, 0.0],
            }
        }
    }
}

// ---------- Minimal resource manager (small, fast) ----------
pub mod resources {
    use std::collections::HashMap;
    use std::sync::Arc;
    use parking_lot::RwLock;
    use wgpu::util::DeviceExt;

    /// Very small texture handle wrapper
    #[derive(Clone)]
    pub struct TextureHandle {
        pub view: wgpu::TextureView,
        pub sampler: wgpu::Sampler,
    }

    /// Lightweight resource manager: stores a few textures and allows lookup.
    /// This is intentionally tiny; replace with your full manager later.
    pub struct ResourceManager {
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        textures: RwLock<HashMap<String, TextureHandle>>,
    }

    impl ResourceManager {
        pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
            Self {
                device,
                queue,
                textures: RwLock::new(HashMap::new()),
            }
        }

        /// Create a 1x1 white texture and register under `name`.
        pub fn create_white(&self, name: &str) -> TextureHandle {
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("{}_white", name)),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let data = [255u8, 255u8, 255u8, 255u8];
            self.queue.write_texture(
                wgpu::ImageCopyTexture { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                &data,
                wgpu::ImageDataLayout { offset: 0, bytes_per_row: std::num::NonZeroU32::new(4), rows_per_image: std::num::NonZeroU32::new(1) },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor::default());
            let handle = TextureHandle { view: view.clone(), sampler: sampler.clone() };
            self.textures.write().insert(name.to_string(), handle.clone());
            handle
        }

        pub fn get(&self, name: &str) -> Option<TextureHandle> {
            self.textures.read().get(name).cloned()
        }
    }
}

// ---------- Minimal stubs for compute/network/physics ----------
pub mod compute {
    pub fn init() {
        // placeholder
    }
}
pub mod network {
    pub fn init() {
        // placeholder
    }
}
pub mod physics {
    pub fn init() {
        // placeholder
    }
}

// ---------- Renderer (async init, correct wgpu usage) ----------
pub mod renderer {
    use super::camera::CameraUniform;
    use super::resources::ResourceManager;
    use std::sync::Arc;
    use wgpu::util::DeviceExt;
    use winit::window::Window;
    use glam::{Mat4, Vec3};

    pub struct Renderer {
        pub device: Arc<wgpu::Device>,
        pub queue: Arc<wgpu::Queue>,
        pub surface: Option<wgpu::Surface>,
        pub config: wgpu::SurfaceConfiguration,
        pub size: (u32, u32),
        pub resource_manager: ResourceManager,
        // simple pipeline for a fullscreen triangle (demo)
        pub pipeline: wgpu::RenderPipeline,
        pub camera_buffer: wgpu::Buffer,
        pub camera_bind_group: wgpu::BindGroup,
    }

    impl Renderer {
        /// Async constructor: creates instance, adapter, device, queue, surface, and a tiny pipeline.
        pub async fn new(window: &Window) -> Self {
            // Create instance
            let backend = wgpu::Backends::all();
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: backend, dx12_shader_compiler: Default::default() });

            // Surface (if available)
            let surface = unsafe { instance.create_surface(window).ok() };

            // Request adapter
            let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface.as_ref(),
                force_fallback_adapter: false,
            }).await.expect("Failed to request adapter");

            // Request device and queue
            let (device, queue) = adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("device"),
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            ).await.expect("Failed to request device");

            let device = Arc::new(device);
            let queue = Arc::new(queue);

            // Surface config
            let size = window.inner_size();
            let surface_format = surface.as_ref()
                .and_then(|s| s.get_capabilities(&adapter).formats.iter().copied().find(|f| f.is_srgb()))
                .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb);

            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width: size.width.max(1),
                height: size.height.max(1),
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
            };

            if let Some(s) = &surface {
                s.configure(&device, &config);
            }

            // Resource manager
            let resource_manager = ResourceManager::new(device.clone(), queue.clone());
            let _white = resource_manager.create_white("dummy");

            // Camera uniform buffer and bind group layout
            let camera_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera_bind_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                }],
            });

            let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("camera_buffer"),
                size: std::mem::size_of::<CameraUniform>() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("camera_bind_group"),
                layout: &camera_bind_layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() }],
            });

            // Minimal shader (embedded WGSL)
            let shader_src = r#"
                @vertex fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
                    var pos = array<vec2<f32>, 3>(
                        vec2<f32>(-1.0, -1.0),
                        vec2<f32>(3.0, -1.0),
                        vec2<f32>(-1.0, 3.0)
                    );
                    return vec4<f32>(pos[idx], 0.0, 1.0);
                }
                @fragment fn fs_main() -> @location(0) vec4<f32> {
                    return vec4<f32>(0.2, 0.6, 0.9, 1.0);
                }
            "#;

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("minimal_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });

            // Pipeline layout
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pipeline_layout"),
                bind_group_layouts: &[&camera_bind_layout],
                push_constant_ranges: &[],
            });

            // Render pipeline (wgpu v22 requires `cache` and `compilation_options` fields)
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("minimal_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[],
                    compilation_options: None,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: None,
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

            Self {
                device,
                queue,
                surface,
                config,
                size: (config.width, config.height),
                resource_manager,
                pipeline,
                camera_buffer,
                camera_bind_group,
            }
        }

        /// Render a single frame (very small, fast path).
        pub fn render(&mut self) {
            // Acquire frame
            let frame = match &self.surface {
                Some(surface) => match surface.get_current_texture() {
                    Ok(frame) => Some(frame),
                    Err(e) => {
                        // Reconfigure surface on lost
                        surface.configure(&self.device, &self.config);
                        surface.get_current_texture().ok()
                    }
                },
                None => None,
            };

            // Create encoder
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame_encoder") });

            // If we have a frame, render to it; otherwise render to a dummy texture (headless)
            if let Some(frame) = frame {
                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("main_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                        })],
                        depth_stencil_attachment: None,
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    });
                    rpass.set_pipeline(&self.pipeline);
                    rpass.set_bind_group(0, &self.camera_bind_group, &[]);
                    rpass.draw(0..3, 0..1);
                }
                // submit and present
                self.queue.submit(Some(encoder.finish()));
                frame.present();
            } else {
                // Headless: render to a small temp texture to keep pipeline exercised
                let temp = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("headless_temp"),
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.config.format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                let view = temp.create_view(&wgpu::TextureViewDescriptor::default());
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("headless_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                        })],
                        depth_stencil_attachment: None,
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    });
                    rpass.set_pipeline(&self.pipeline);
                    rpass.set_bind_group(0, &self.camera_bind_group, &[]);
                    rpass.draw(0..3, 0..1);
                }
                self.queue.submit(Some(encoder.finish()));
            }
        }

        /// Resize surface (call on window resize)
        pub fn resize(&mut self, width: u32, height: u32) {
            if width == 0 || height == 0 { return; }
            self.config.width = width;
            self.config.height = height;
            self.size = (width, height);
            if let Some(surface) = &self.surface {
                surface.configure(&self.device, &self.config);
            }
        }

        /// Update camera uniform quickly
        pub fn update_camera(&self, view: Mat4, proj: Mat4, pos: Vec3) {
            let uniform = CameraUniform::from_matrices(view, proj, pos, 0.0);
            self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));
        }
    }
}

// ---------- wasm entrypoint and native runner ----------
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub async fn wasm_start() {
    // Better panic messages in browser console
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Info).ok();

    // Build event loop and window
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().with_title("Slop Engine (wasm)").build(&event_loop).expect("Failed to create window");

    // Attach canvas to DOM
    {
        use winit::platform::web::WindowExtWebSys;
        if let Some(canvas_el) = window.canvas() {
            if let Some(win) = web_sys::window() {
                if let Some(doc) = win.document() {
                    if let Some(dst) = doc.get_element_by_id("slop-container") {
                        let canvas: web_sys::Element = canvas_el.into();
                        let _ = dst.append_child(&canvas);
                    } else if let Some(body) = doc.body() {
                        let canvas: web_sys::Element = canvas_el.into();
                        let _ = body.append_child(&canvas);
                    }
                }
            }
        }
    }

    // Create renderer
    let mut renderer = renderer::Renderer::new(&window).await;

    // Run event loop (note: in wasm, EventLoop::run is supported)
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::RedrawRequested(_) => {
                renderer.render();
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                renderer.resize(size.width.max(1), size.height.max(1));
            }
            _ => {}
        }
    });
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run_native() {
    // Native entrypoint for quick testing
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().with_title("Slop Engine (native)").build(&event_loop).expect("Failed to create window");

    // Use pollster to block on async init
    let mut renderer = pollster::block_on(renderer::Renderer::new(&window));

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::RedrawRequested(_) => {
                renderer.render();
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                renderer.resize(size.width.max(1), size.height.max(1));
            }
            _ => {}
        }
    });
}

// ---------- Tests ----------
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn smoke_native_run() {
        // Ensure the library compiles and basic functions exist.
        // We don't run the event loop in tests; just construct a minimal CameraUniform.
        let u = camera::CameraUniform::from_matrices(glam::Mat4::IDENTITY, glam::Mat4::IDENTITY, glam::Vec3::ZERO, 0.0);
        assert_eq!(u.position[0], 0.0);
    }
}
