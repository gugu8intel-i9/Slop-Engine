// src/lib.rs

use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// ----------------------------------------------------------------------------
// Async Engine Runner
// ----------------------------------------------------------------------------
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    // 1. Setup logging for Web or Native
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        let _ = console_log::init_with_level(log::Level::Warn);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = env_logger::try_init();
    }

    let event_loop = EventLoop::new().expect("Failed to create event loop");

    // 2. Initialize WGPU asynchronously BEFORE the window is created.
    // We pass `compatible_surface: None` here so we can get the device ready 
    // without blocking the main browser thread.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None, // No surface needed yet
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

    // 3. Setup the App state
    let mut app = EngineApp {
        instance,
        device: Arc::new(device),
        queue: Arc::new(queue),
        adapter,
        window: None,
        surface: None,
        config: None,
        render_pipeline: None,
        depth_texture_view: None,
    };

    // 4. Start the Application Handler
    #[cfg(target_arch = "wasm32")]
    {
        // On Web, winit 0.30 requires spawn_app
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(app);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        // On Native, winit 0.30 requires run_app
        event_loop.run_app(&mut app).expect("Event loop crashed");
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run_native() {
    pollster::block_on(run());
}

// ----------------------------------------------------------------------------
// Core App State (winit 0.30 ApplicationHandler)
// ----------------------------------------------------------------------------
struct EngineApp {
    instance: wgpu::Instance,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    adapter: wgpu::Adapter,
    
    // Created after resumed() is called
    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    config: Option<wgpu::SurfaceConfiguration>,
    render_pipeline: Option<wgpu::RenderPipeline>,
    depth_texture_view: Option<wgpu::TextureView>,
}

impl ApplicationHandler for EngineApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }

        // 1. Create Window (winit 0.30 replaced WindowBuilder with default_attributes)
        let attrs = Window::default_attributes().with_title("Slop Engine");
        let window = Arc::new(event_loop.create_window(attrs).expect("Window creation failed"));
        self.window = Some(window.clone());

        // 2. Attach to Web DOM
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

        // 3. Create Surface (Wrapped in Arc so surface can have 'static lifetime)
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

        // 4. Initialize performance features
        let depth_view = create_depth_texture(&self.device, &config);

        // 5. Default WGSL (layout: None automatically deduces uniforms for ANY wgsl code!)
        let default_wgsl = r#"
            @vertex
            fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
                let x = f32(1 - i32(in_vertex_index)) * 0.5;
                let y = f32(i32(in_vertex_index & 1u) * 2 - 1) * 0.5;
                return vec4<f32>(x, y, 0.5, 1.0);
            }
            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.3, 0.5, 1.0);
            }
        "#;
        
        let pipeline = build_pipeline(&self.device, config.format, default_wgsl);

        self.surface = Some(surface);
        self.config = Some(config);
        self.depth_texture_view = Some(depth_view);
        self.render_pipeline = Some(pipeline);

        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        let window = self.window.as_ref().unwrap();
        if window.id() != window_id { return; }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                if physical_size.width > 0 && physical_size.height > 0 {
                    if let (Some(surface), Some(config), Some(depth_view)) = (
                        self.surface.as_mut(),
                        self.config.as_mut(),
                        self.depth_texture_view.as_mut()
                    ) {
                        config.width = physical_size.width;
                        config.height = physical_size.height;
                        surface.configure(&self.device, config);
                        *depth_view = create_depth_texture(&self.device, config);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.update();
                match self.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => {
                        if let (Some(surface), Some(config)) = (self.surface.as_ref(), self.config.as_ref()) {
                            surface.configure(&self.device, config);
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
        // Run continuously
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

// ----------------------------------------------------------------------------
// Methods
// ----------------------------------------------------------------------------
impl EngineApp {
    fn update(&mut self) {
        // Game logic goes here
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let (Some(surface), Some(pipeline), Some(depth_view)) = (
            self.surface.as_ref(),
            self.render_pipeline.as_ref(),
            self.depth_texture_view.as_ref()
        ) else { return Ok(()) };

        let output = surface.get_current_texture()?;
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
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.05, b: 0.05, a: 1.0 }),
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
            render_pass.draw(0..3, 0..1); // Draw hardcoded triangle
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------
fn build_pipeline(device: &wgpu::Device, format: wgpu::TextureFormat, wgsl_source: &str) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Dynamic Shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Dynamic Render Pipeline"),
        layout: None, // <-- Automatically deduce uniforms/bindings from WGSL
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

fn create_depth_texture(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
    let size = wgpu::Extent3d {
        width: config.width,
        height: config.height,
        depth_or_array_layers: 1,
    };
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    }).create_view(&wgpu::TextureViewDescriptor::default())
}
