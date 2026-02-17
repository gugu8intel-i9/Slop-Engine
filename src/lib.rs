use std::sync::Arc;

use wasm_bindgen::prelude::*;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[wasm_bindgen(start)]
pub async fn run() {
    // Better panic messages in browser console
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));

    // Logging to browser console
    console_log::init_with_level(log::Level::Debug)
        .expect("Failed to initialize logger");

    // Create event loop + window (winit 0.29: EventLoop::new() -> Result)
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let window = WindowBuilder::new()
        .with_title("Slop Engine")
        .build(&event_loop)
        .expect("Failed to create window");

    // Attach canvas to HTML when running on wasm32
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;

        let canvas = window.canvas().expect("No canvas found");
        let canvas_el: web_sys::Element = canvas.into();

        let document = web_sys::window()
            .and_then(|w| w.document())
            .expect("No document");

        if let Some(dst) = document.get_element_by_id("slop-container") {
            dst.append_child(&canvas_el).unwrap();
        } else {
            document.body().unwrap().append_child(&canvas_el).unwrap();
        }
    }

    // === WGPU SETUP (wgpu 22) ===

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::default(),
        dx12_shader_compiler: Default::default(),
        gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
    });

    let surface = unsafe { instance.create_surface(&window) }
        .expect("Failed to create surface");

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("No suitable GPU adapters found");

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("SlopEngine Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .expect("Failed to create device");

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    // Surface config
    let size = window.inner_size();
    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(caps.formats[0]);

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: caps.present_modes[0],
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };

    surface.configure(&device, &config);

    // === MAIN EVENT LOOP (correct winit 0.29 API) ===

    event_loop
        .run(move |event, target| {
            target.set_control_flow(ControlFlow::Poll);

            match event {
                Event::WindowEvent { event, window_id } if window_id == window.id() => {
                    match event {
                        WindowEvent::CloseRequested => {
                            target.exit();
                        }

                        WindowEvent::Resized(new_size) => {
                            if new_size.width > 0 && new_size.height > 0 {
                                config.width = new_size.width;
                                config.height = new_size.height;
                                surface.configure(&device, &config);
                            }
                        }

                        WindowEvent::ScaleFactorChanged { scale_factor } => {
                            // No resizing here — winit will send Resized next.
                            log::debug!("Scale factor changed: {}", scale_factor);
                        }

                        WindowEvent::RedrawRequested => {
                            render_frame(&surface, &device, &queue, &config);
                        }

                        _ => {}
                    }
                }

                Event::AboutToWait => {
                    window.request_redraw();
                }

                _ => {}
            }
        })
        .expect("Event loop error");
}

fn render_frame(
    surface: &wgpu::Surface,
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    config: &wgpu::SurfaceConfiguration,
) {
    let frame = match surface.get_current_texture() {
        Ok(f) => f,
        Err(_) => {
            surface.configure(device, config);
            match surface.get_current_texture() {
                Ok(f) => f,
                Err(e) => {
                    log::error!("Surface error: {:?}", e);
                    return;
                }
            }
        }
    };

    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("SlopEngine Encoder"),
    });

    {
        let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SlopEngine Clear Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.3,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
    }

    queue.submit(Some(encoder.finish()));
    frame.present();
}
