// src/lib.rs

use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// ----------------------------------------------------------------------------
// Async Runner (Works seamlessly on Native and Wasm)
// ----------------------------------------------------------------------------
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    // Setup logging and panic hooks for web and native
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
        } else {
            env_logger::init();
        }
    }

    // 1. Initialize Event Loop and Window
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let window = WindowBuilder::new()
        .with_title("Slop Engine")
        .build(&event_loop)
        .expect("Failed to create window");

    // 2. Wasm specific: Append canvas to the HTML document
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        let canvas = window.canvas().expect("Failed to get Canvas");
        let canvas_element = web_sys::Element::from(canvas);
        
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| body.append_child(&canvas_element).ok())
            .expect("Couldn't append canvas to document body.");
    }

    // 3. Initialize Engine State
    let mut state = EngineState::new(&window).await;

    // 4. Run standard Event Loop
    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent { ref event, window_id } if window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { .. } => {
                        // In winit 0.29, resize happens automatically when scale factor changes
                        // Just request a redraw
                        window.request_redraw();
                    }
                    WindowEvent::RedrawRequested => {
                        state.update();
                        match state.render() {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                            Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                            Err(e) => log::error!("{:?}", e),
                        }
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                // Request a redraw continuously for game loop performance
                window.request_redraw();
            }
            _ => {}
        }
    }).expect("Event loop failed to run");
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run_native() {
    pollster::block_on(run());
}

// ----------------------------------------------------------------------------
// Core Engine State
// ----------------------------------------------------------------------------
struct EngineState<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    
    // Performance Feature: Depth Texture for 3D capability
    depth_texture_view: wgpu::TextureView,
}

impl<'a> EngineState<'a> {
    async fn new(window: &'a Window) -> Self {
        let size = window.inner_size();

        // High-performance WGPU Instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // The surface MUST be created before the adapter to guarantee compatibility
        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

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
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
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
        surface.configure(&device, &config);

        // Feature: Depth Buffer setup
        let depth_texture_view = Self::create_depth_texture(&device, &config);

        // Feature: Support "Any WGSL File"
        // Here we provide a default simple shader to start.
        let default_wgsl = r#"
            @vertex
            fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
                let x = f32(1 - i32(in_vertex_index)) * 0.5;
                let y = f32(i32(in_vertex_index & 1u) * 2 - 1) * 0.5;
                return vec4<f32>(x, y, 0.0, 1.0);
            }

            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.5, 1.0);
            }
        "#;
        
        let render_pipeline = Self::build_pipeline(&device, config.format, default_wgsl);

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            depth_texture_view,
        }
    }

    /// Feature: Build Pipeline dynamically from ANY WGSL string.
    /// By setting `layout: None`, wgpu automatically deduces bind groups and layouts from the WGSL!
    pub fn build_pipeline(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        wgsl_source: &str,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Dynamic Shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Dynamic Render Pipeline"),
            layout: None, // <-- MAGIC: Auto-generates pipeline layout based on the WGSL file
            cache: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[], // Add your Vertex layouts here if needed later
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

    /// Helper for Depth Buffer generation
    fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> wgpu::TextureView {
        let size = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            // Must resize depth buffer when surface resizes
            self.depth_texture_view = Self::create_depth_texture(&self.device, &self.config);
        }
    }

    pub fn update(&mut self) {
        // Game logic goes here
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.05,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store, // v22 compatibility
                    },
                })],
                // Depth stencil enabled for 3D compatibility
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None, // v22 compatibility
                timestamp_writes: None,    // v22 compatibility
            });

            render_pass.set_pipeline(&self.render_pipeline);
            // Draw default hard-coded vertices from shader
            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
