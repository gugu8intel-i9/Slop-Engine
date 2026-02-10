use wasm_bindgen::prelude::*;
use winit::{event::*, event_loop::{ControlFlow, EventLoop}, window::WindowBuilder};
mod renderer;
mod state;

#[wasm_bindgen(start)]
pub async fn run() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Debug).expect("Logger failed");

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().with_title("Slop Engine").build(&event_loop).unwrap();

    // Attach to HTML
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        web_sys::window().and_then(|win| win.document()).and_then(|doc| {
            let dst = doc.get_element_by_id("slop-container")?;
            let canvas = web_sys::Element::from(window.canvas());
            dst.append_child(&canvas).ok()?;
            Some(())
        }).expect("Canvas attach failed");
    }

    let mut renderer = renderer::Renderer::new(&window).await;

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::RedrawRequested(_) => {
                renderer.render();
            }
            Event::MainEventsCleared => { window.request_redraw(); }
            _ => *control_flow = ControlFlow::Poll,
        }
    });
}