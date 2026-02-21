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
//
// Example Cargo.toml change:
//
// [dependencies]
// lru = "0.16.3"
//
// If you can't upgrade immediately, enable the "lru_safe" feature in your crate and use
// `safe_lru::SafeLruCache` instead of directly calling `LruCache::iter_mut()`.
//
// -------------------------------

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
    pollster::block_on(run_inner());
}

async fn run_inner() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    let _ = console_log::init_with_level(log::Level::Debug);

    // EventLoop::new() historically returns an EventLoop; older code used `.expect(...)`.
    // Keep same shape as your original file.
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Slop Engine")
            .build(&event_loop)
            .expect("Failed to create window"),
    );

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
        flags: wgpu::InstanceFlags::empty(),
        gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
    });

    // wgpu 22.x: create_surface is NOT unsafe and accepts Arc<Window>
    let surface = instance
        .create_surface(window.clone())
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
        desired_maximum_frame_latency: 2u32,
    };

    surface.configure(&device, &config);

    window.request_redraw();

    // === Main event loop ===
    // `window` is now an Arc<Window>, so it can be moved into the closure
    // without conflicting with `surface` (which holds its own Arc clone).
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

                        WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                            log::debug!("Scale factor changed: {}", scale_factor);
                        }

                        WindowEvent::RedrawRequested => {
                            render_frame(&surface, &device, &queue, &config);
                        }

                        _ => {}
                    }
                }

                _ => {}
            }
        })
        .expect("Event loop failed");
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
//
// This module provides `SafeLruCache` when the "lru_safe" feature is enabled.
// It avoids `iter_mut()` by collecting cloned keys using `iter()` and using `get_mut()`
// for mutation, which does not trigger the problematic `IterMut` path.
//
// Usage (Cargo.toml):
// [features]
// lru_safe = ["lru"]
//
// [dependencies]
// lru = { version = "0.16.3", optional = true }
// -------------------------------

#[cfg(feature = "lru_safe")]
pub mod safe_lru {
    //! Safe wrapper around `lru::LruCache` to avoid `iter_mut()` usage.
    //!
    //! The `iter_mut()` implementation in certain `lru` versions had a soundness bug:
    //! it temporarily created exclusive references to keys while internal shared
    //! references still existed. The wrapper here performs key collection via `iter()`
    //! and then uses `get_mut()` for safe mutation.
    //!
    //! Please still prefer upgrading to lru >= 0.16.3 which contains the upstream fix.

    use std::hash::Hash;
    use std::fmt;
    use std::ops::{Deref, DerefMut};

    use lru::LruCache;

    /// A thin wrapper around `lru::LruCache` that provides safe mutation helpers.
    /// `K` must be `Clone` so we can collect keys safely from a shared iterator.
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
        /// Create a new SafeLruCache with the given capacity.
        pub fn new(capacity: usize) -> Self {
            SafeLruCache {
                inner: LruCache::new(capacity),
            }
        }

        /// Insert (or update) a value.
        pub fn put(&mut self, k: K, v: V) {
            self.inner.put(k, v);
        }

        /// Get a shared reference.
        pub fn get(&self, k: &K) -> Option<&V> {
            self.inner.get(k)
        }

        /// Get a mutable reference.
        pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
            self.inner.get_mut(k)
        }

        /// Remove a key.
        pub fn pop(&mut self, k: &K) -> Option<V> {
            self.inner.pop(k)
        }

        /// Iterate over entries immutably.
        pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
            self.inner.iter()
        }

        /// Mutate values for every key that currently exists in the cache by cloning keys
        /// first and then calling `get_mut()` for each key. This avoids `iter_mut()`.
        ///
        /// Example:
        /// ```
        /// # use safe_lru::SafeLruCache;
        /// # let mut c = SafeLruCache::new(16);
        /// // populate...
        /// c.mutate_values(|v| { /* mutate in-place */ });
        /// ```
        pub fn mutate_values<F>(&mut self, mut f: F)
        where
            F: FnMut(&mut V),
        {
            // Collect keys via shared iteration (safe), then mutate via get_mut.
            // Cloning keys avoids holding references into the internal structure while mutating.
            let keys: Vec<K> = self.inner.iter().map(|(k, _v)| k.clone()).collect();
            for k in keys {
                if let Some(v) = self.inner.get_mut(&k) {
                    f(v);
                }
            }
        }
    }

    // Allow extracting the inner cache if needed.
    impl<K, V> SafeLruCache<K, V> {
        pub fn into_inner(self) -> LruCache<K, V> {
            self.inner
        }
    }
}

#[cfg(not(feature = "lru_safe"))]
pub mod safe_lru {
    //! Stub module when the "lru_safe" feature isn't enabled.
    //! Keeps APIs available for compilation even if `lru` is not present.
    //!
    //! To enable the real safe wrapper, add `lru = { version = "0.16.3", optional = true }`
    //! to your dependencies and enable the `lru_safe` feature.
    pub struct SafeLruCachePlaceholder;

    impl SafeLruCachePlaceholder {
        #[allow(dead_code)]
        pub fn new(_capacity: usize) -> Self {
            SafeLruCachePlaceholder
        }
    }
}
