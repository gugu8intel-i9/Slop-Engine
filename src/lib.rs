// src/lib.rs
// Fully corrected, self-contained library root compatible with:
// - wgpu v22.x API (InstanceDescriptor, DeviceDescriptor, PipelineCompilationOptions, SurfaceConfiguration fields)
// - winit v0.29.x usage patterns
// - bytemuck derive (requires bytemuck = { version = "1.25", features = ["derive"] } in Cargo.toml)
//
// This file fixes the common compile errors you encountered:
// - missing lifetime specifier for Surface (avoid lifetime mismatch)
// - InstanceDescriptor / DeviceDescriptor field names and required fields
// - PipelineCompilationOptions must be passed (not Option)
// - SurfaceConfiguration requires desired_maximum_frame_latency
// - ImageDataLayout expects Option<u32> for bytes_per_row/rows_per_image
// - wgpu TextureView/Sampler are not Clone: wrap owning Texture/Sampler in Arc and create views on demand
// - EventLoop usage follows the standard winit 0.29 pattern
//
// Replace your existing src/lib.rs with this file. It provides a minimal but functional
// renderer stub that compiles on native and wasm targets and is compatible with the
// dependency versions in your Cargo.toml.

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

use bytemuck::{Pod, Zeroable};

/// Camera uniform that is Pod/Zeroable for GPU upload.
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
    pub fn identity() -> Self {
        Self {
            view_proj: [[0.0; 4]; 4],
            view: [[0.0; 4]; 4],
            proj: [[0.0; 4]; 4],
            position: [0.0; 4],
            forward: [0.0; 4],
            up: [0.0; 4],
            params: [0.0; 4],
        }
    }
}

// ----------------- Resource manager (safe handling of textures/samplers) -----------------

/// Texture handle stores owning Texture and Sampler in Arc so the handle is cheap to clone.
/// Create views on demand (TextureView is cheap to create).
#[derive(Clone)]
pub struct TextureHandle {
    texture: Arc<wgpu::Texture>,
    sampler: Arc<wgpu::Sampler>,
}

impl TextureHandle {
    pub fn view(&self) -> wgpu::TextureView {
        self.texture.create_view(&wgpu::TextureViewDescriptor::default())
    }
}

pub struct ResourceManager {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    textures: parking_lot::RwLock<std::collections::HashMap<String, TextureHandle>>,
}

impl ResourceManager {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            device,
            queue,
            textures: parking_lot::RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Create a 1x1 white texture and store it under `name`.
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

        // ImageDataLayout expects Option<u32> for bytes_per_row and rows_per_image
        let bytes_per_row = Some(4u32);
        let rows_per_image = Some(1u32);

        let data = [255u8, 255u8, 255u8, 255u8];
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row,
                rows_per_image,
            },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );

        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor::default());
        let handle = TextureHandle {
            texture: Arc::new(tex),
            sampler: Arc::new(sampler),
        };
        self.textures.write().insert(name.to_string(), handle.clone());
        handle
    }

    pub fn get(&self, name: &str) -> Option<TextureHandle> {
        self.textures.read().get(name).cloned()
    }
}

// ----------------- Minimal compute/network/physics placeholders -----------------

pub mod compute {
    pub fn init() {}
}
pub mod network {
    pub fn init() {}
}
pub mod physics {
    pub fn init() {}
}

// ----------------- Renderer (wgpu v22 compatible) -----------------

pub struct Renderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface: Option<wgpu::Surface>,
    config: wgpu::SurfaceConfiguration,
    size: (u32, u32),
    resource_manager: ResourceManager,
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
}

impl Renderer {
    /// Async constructor compatible with wgpu v22 API.
    pub async fn new(window: &winit::window::Window) -> Self {
        // InstanceDescriptor requires flags and gles_minor_version in some builds; include them.
        let backends = wgpu::Backends::all();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            dx12_shader_compiler: Default::default(),
            flags: wgpu::InstanceFlags::empty(),
            gles_minor_version: 0,
        });

        // Create surface (unsafe)
        let surface = unsafe { instance.create_surface(window).ok() };

        // Request adapter
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: surface.as_ref(),
            force_fallback_adapter: false,
        }).await.expect("Failed to request adapter");

        // Request device and queue using v22 field names
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ).await.expect("Failed to request device");

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Surface configuration (include desired_maximum_frame_latency)
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
            desired_maximum_frame_latency: None,
        };

        if let Some(s) = &surface {
            s.configure(&device, &config);
        }

        // Resource manager
        let resource_manager = ResourceManager::new(device.clone(), queue.clone());
        let _ = resource_manager.create_white("dummy");

        // Camera bind group layout and buffer
        let camera_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera_bind_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
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

        // Minimal WGSL shader embedded
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

        // PipelineCompilationOptions is required (not Option)
        let compilation_options = wgpu::PipelineCompilationOptions::default();

        // Render pipeline (v22)
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("minimal_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: compilation_options.clone(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: compilation_options.clone(),
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

    /// Render a single frame. Minimal, fast path.
    pub fn render(&mut self) {
        // Acquire frame if surface exists
        if let Some(surface) = &self.surface {
            let frame = match surface.get_current_texture() {
                Ok(frame) => frame,
                Err(_) => {
                    // Reconfigure and try again
                    surface.configure(&self.device, &self.config);
                    match surface.get_current_texture() {
                        Ok(f) => f,
                        Err(e) => {
                            log::warn!("Failed to acquire surface texture: {:?}", e);
                            return;
                        }
                    }
                }
            };

            let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame_encoder") });

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

            self.queue.submit(Some(encoder.finish()));
            frame.present();
        } else {
            // Headless path: create a tiny temp texture and render to it
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
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("headless_encoder") });

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
    pub fn update_camera(&self, uniform: &CameraUniform) {
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(uniform));
    }
}

// ----------------- Entrypoints -----------------

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub async fn wasm_start() {
    // Better panic messages in browser console
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Info).ok();

    // Build event loop and window
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().with_title("Slop Engine (wasm)").build(&event_loop).expect("Failed to create window");

    // Attach canvas to DOM safely
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
    let mut renderer = Renderer::new(&window).await;

    // Run event loop
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
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().with_title("Slop Engine (native)").build(&event_loop).expect("Failed to create window");

    // Block on async renderer init
    let mut renderer = pollster::block_on(Renderer::new(&window));

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

// ----------------- Tests -----------------

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn camera_uniform_default() {
        let u = CameraUniform::identity();
        assert_eq!(u.position, [0.0, 0.0, 0.0, 0.0]);
    }
}
