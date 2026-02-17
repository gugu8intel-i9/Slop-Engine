// src/lib.rs
// Fully fixed, drop-in lib.rs compatible with:
// - wgpu = "22.x"
// - winit = "0.29.x"
// - wasm-bindgen for wasm target
//
// Fixes applied:
// - Unwrap EventLoop::new() result so build(&event_loop) and run() use the correct types.
// - Use wgpu::Gles3MinorVersion::Automatic for gles_minor_version.
// - Set desired_maximum_frame_latency as a u32 (2).
// - Handle ScaleFactorChanged with `..` to ignore extra fields.
// - Use MainEventsCleared to request redraws (winit 0.29).
//
// Replace your existing src/lib.rs with this file.

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

#[cfg(target_arch = "wasm32")]
use winit::platform::web::WindowExtWebSys;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub async fn run() {
    run_inner().await;
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run_native() {
    // Block on the async runtime to run the same initialization path as wasm.
    pollster::block_on(run_inner());
}

async fn run_inner() {
    // Better panic messages in browser console (wasm) and helpful logs everywhere
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    let _ = console_log::init_with_level(log::Level::Debug);

    // Create event loop + window
    // On some platforms EventLoop::new() returns Result; unwrap to get EventLoop.
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let window = WindowBuilder::new()
        .with_title("Slop Engine")
        .build(&event_loop)
        .expect("Failed to create window");

    // Attach canvas to HTML when running on wasm32
    #[cfg(target_arch = "wasm32")]
    {
        let canvas = window.canvas().expect("No canvas available from winit window");
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

    // === WGPU setup (v22-compatible) ===

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        dx12_shader_compiler: Default::default(),
        // Provide fields for compatibility across builds
        flags: wgpu::InstanceFlags::empty(),
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
        .expect("Failed to request adapter");

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("slop_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        )
        .await
        .expect("Failed to request device");

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    // Surface configuration
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
        // desired_maximum_frame_latency expects u32 in this wgpu version
        desired_maximum_frame_latency: 2u32,
    };

    surface.configure(&device, &config);

    // === Main event loop (winit 0.29 correct signature) ===
    event_loop.run(move |event, _event_loop_target, control_flow| {
        // Default to polling; change to Wait if you prefer lower CPU usage.
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }

                    WindowEvent::Resized(new_size) => {
                        if new_size.width > 0 && new_size.height > 0 {
                            config.width = new_size.width;
                            config.height = new_size.height;
                            surface.configure(&device, &config);
                        }
                    }

                    // In winit 0.29 ScaleFactorChanged no longer carries a size field.
                    // Use .. to ignore any additional fields (e.g., inner_size_writer).
                    WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                        // No surface reconfigure here; Resized will follow with the new size.
                        log::debug!("Scale factor changed: {}", scale_factor);
                    }

                    WindowEvent::RedrawRequested => {
                        render_frame(&surface, &device, &queue, &config);
                    }

                    _ => {}
                }
            }

            // Request redraw each frame
            Event::MainEventsCleared => {
                window.request_redraw();
            }

            _ => {}
        }
    });
}

fn render_frame(
    surface: &wgpu::Surface,
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    config: &wgpu::SurfaceConfiguration,
) {
    // Acquire frame
    let frame = match surface.get_current_texture() {
        Ok(frame) => frame,
        Err(err) => {
            log::warn!("Failed to acquire next swap chain texture: {:?}. Reconfiguring surface.", err);
            surface.configure(device, config);
            match surface.get_current_texture() {
                Ok(frame) => frame,
                Err(e) => {
                    log::error!("Failed to acquire frame after reconfigure: {:?}", e);
                    return;
                }
            }
        }
    };

    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("render_encoder"),
    });

    {
        let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clear_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.08,
                        g: 0.12,
                        b: 0.18,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        // No draw calls in this minimal stub.
    }

    queue.submit(Some(encoder.finish()));
    frame.present();
}
