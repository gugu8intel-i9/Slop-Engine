// src/lib.rs

#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

// -------------------------------
// SECURITY NOTE
// -------------------------------
// Advisory: lru IterMut soundness issue (creates a temporary exclusive reference to keys
// while a shared pointer is still held by the internal HashMap). Upstream fix: upgrade
// `lru` to 0.16.3 or later.
//
// This file includes a safe helper behind the "lru_safe" feature that performs mutation
// without calling `iter_mut()`. Prefer upgrading the `lru` crate in Cargo.toml to >= 0.16.3.
// -------------------------------

use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
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
    pollster::block_on(run_inner());
}

// ----------------------------------------------------------------------------
// winit 0.30 + wgpu 22 App State
// ----------------------------------------------------------------------------
struct SlopApp {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    
    // Created inside the `resumed` event
    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    config: Option<wgpu::SurfaceConfiguration>,
}

impl ApplicationHandler for SlopApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);

        if self.window.is_some() {
            return;
        }

        // 1. Create Window
        let attrs = Window::default_attributes().with_title("Slop Engine");
        let window = Arc::new(event_loop.create_window(attrs).expect("Failed to create window"));
        self.window = Some(window.clone());

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

        // 2. Create and configure Surface
        let surface = self.instance.create_surface(window.clone()).expect("Failed to create surface");
        let size = window.inner_size();
        let caps = surface.get_capabilities(&self.adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: caps.present_modes[0],
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2u32,
        };

        surface.configure(&self.device, &config);

        self.surface = Some(surface);
        self.config = Some(config);

        window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.as_ref() else { return };
        if window.id() != window_id { return; }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    if let (Some(surface), Some(config)) = (self.surface.as_ref(), self.config.as_mut()) {
                        config.width = new_size.width;
                        config.height = new_size.height;
                        surface.configure(&self.device, config);
                    }
                }
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                log::debug!("Scale factor changed: {}", scale_factor);
            }
            WindowEvent::RedrawRequested => {
                if let (Some(surface), Some(config)) = (self.surface.as_ref(), self.config.as_ref()) {
                    render_frame(surface, &self.device, &self.queue, config);
                }
            }
            _ => {}
        }
    }
}

// ----------------------------------------------------------------------------
// Async Runner
// ----------------------------------------------------------------------------
async fn run_inner() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    let _ = console_log::init_with_level(log::Level::Debug);

    let event_loop = EventLoop::new().expect("Failed to create event loop");

    // === WGPU setup ===

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        dx12_shader_compiler: Default::default(),
        flags: wgpu::InstanceFlags::empty(),
        gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
    });

    // Request the adapter early. Setting `compatible_surface: None` allows us 
    // to handle WGPU's async init safely before winit's loop takes complete control.
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
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

    let mut app = SlopApp {
        instance,
        adapter,
        device: Arc::new(device),
        queue: Arc::new(queue),
        window: None,
        surface: None,
        config: None,
    };

    // === Main event loop ===
    
    // Web needs `.spawn_app()` instead of `.run_app()` so it doesn't block the browser loop
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(app);
    }

    // Native desktop apps use `.run_app()`
    #[cfg(not(target_arch = "wasm32"))]
    {
        event_loop.run_app(&mut app).expect("Event loop failed");
    }
}

fn render_frame(
    surface: &wgpu::Surface<'_>,
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    config: &wgpu::SurfaceConfiguration,
) {
    let frame = match surface.get_current_texture() {
        Ok(frame) => frame,
        Err(err) => {
            log::warn!(
                "Failed to acquire next swap chain texture: {:?}. Reconfiguring surface.",
                err
            );
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

    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

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
    }

    queue.submit(Some(encoder.finish()));
    frame.present();
}

// -------------------------------
// Optional safe Lru wrapper
// -------------------------------
#[cfg(feature = "lru_safe")]
pub mod safe_lru {
    use std::hash::Hash;
    use std::fmt;
    use std::ops::{Deref, DerefMut};
    use lru::LruCache;

    pub struct SafeLruCache<K, V> {
        inner: LruCache<K, V>,
    }

    impl<K, V> fmt::Debug for SafeLruCache<K, V>
    where
        K: fmt::Debug + Eq + Hash + Clone,
        V: fmt::Debug,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("SafeLruCache").finish()
        }
    }

    impl<K, V> SafeLruCache<K, V>
    where
        K: Eq + Hash + Clone,
    {
        pub fn new(capacity: usize) -> Self {
            SafeLruCache {
                inner: LruCache::new(capacity),
            }
        }

        pub fn put(&mut self, k: K, v: V) {
            self.inner.put(k, v);
        }

        pub fn get(&self, k: &K) -> Option<&V> {
            self.inner.get(k)
        }

        pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
            self.inner.get_mut(k)
        }

        pub fn pop(&mut self, k: &K) -> Option<V> {
            self.inner.pop(k)
        }

        pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
            self.inner.iter()
        }

        pub fn mutate_values<F>(&mut self, mut f: F)
        where
            F: FnMut(&mut V),
        {
            let keys: Vec<K> = self.inner.iter().map(|(k, _v)| k.clone()).collect();
            for k in keys {
                if let Some(v) = self.inner.get_mut(&k) {
                    f(v);
                }
            }
        }
    }

    impl<K, V> SafeLruCache<K, V> {
        pub fn into_inner(self) -> LruCache<K, V> {
            self.inner
        }
    }
}

#[cfg(not(feature = "lru_safe"))]
pub mod safe_lru {
    pub struct SafeLruCachePlaceholder;

    impl SafeLruCachePlaceholder {
        #[allow(dead_code)]
        pub fn new(_capacity: usize) -> Self {
            SafeLruCachePlaceholder
        }
    }
}
