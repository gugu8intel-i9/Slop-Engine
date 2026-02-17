use wasm_bindgen::prelude::*;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

// All modules that exist in src/
mod camera;
mod compat;
mod depth_texture;
mod renderer;
mod state;
mod compute;
mod network;
mod physics;

#[wasm_bindgen(start)]
pub async fn run() {
    // Better panic messages in browser console
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));

    // Logging to browser console
    console_log::init_with_level(log::Level::Debug)
        .expect("Failed to initialize logger");

    // Create event loop + window
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Slop Engine")
        .build(&event_loop)
        .expect("Failed to create window");

    // Attach canvas to HTML when running on wasm32
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;

        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("slop-container")?;
                let canvas = web_sys::Element::from(window.canvas());
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Failed to attach canvas to HTML");
    }

    // Create renderer
    let mut renderer = renderer::Renderer::new(&window).await;

    // Main event loop
    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::RedrawRequested(_) => {
                renderer.render();
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            _ => {
                *control_flow = ControlFlow::Poll;
            }
        }
    });
}
