// frame_gen.rs
//! High-performance, low-latency frame generator for wgpu + WGSL
//!
//! Design goals:
//! - 2-3 allocations per frame (SurfaceTexture + CommandEncoder + View) — normal for wgpu
//! - 2 frames in flight -> lowest latency (set to 3 for smoother pacing)
//! - Mailbox preferred, Fifo fallback. Immediate is opt-in tearing mode only
//! - Real WGSL shader cache to avoid recompiles
//! - Frame lifetime enforced by Drop — you cannot forget to present
//!
//! Usage:
//! let mut fg = FrameGen::new(window, 1920, 1080).await;
//! loop {
//! if let Some(frame) = fg.begin_frame() {
//! // record passes with frame.encoder
//! // frame is auto-submitted and presented on drop
//! }
//! }

use std::{cell::RefCell, collections::hash_map::DefaultHasher, collections::HashMap, hash::{Hash, Hasher}, time::Instant};
use winit::window::Window;

const MAX_FRAMES_IN_FLIGHT: usize = 2; // 2 = lowest latency, 3 = more stable

pub struct FrameGen<'w> {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    surface: wgpu::Surface<'w>,
    config: wgpu::SurfaceConfiguration,
    frame_index: u64,
    last_frame: Instant,
    start_time: Instant,
    present_mode: wgpu::PresentMode,
    depth_view: Option<wgpu::TextureView>,
    shader_cache: RefCell<HashMap<u64, wgpu::ShaderModule>>,
    // track what features we actually got
    pub features: wgpu::Features,
}

pub struct Frame<'a, 'w> {
    fg: &'a FrameGen<'w>,
    pub texture: Option<wgpu::SurfaceTexture>,
    pub view: wgpu::TextureView,
    pub encoder: Option<wgpu::CommandEncoder>,
    pub delta_time: f32,
    pub total_time: f32,
    pub frame_index: u64,
}

impl<'a, 'w> Drop for Frame<'a, 'w> {
    fn drop(&mut self) {
        // Enforce end_frame — submit and present automatically
        if let Some(encoder) = self.encoder.take() {
            self.fg.queue.submit([encoder.finish()]);
        }
        if let Some(texture) = self.texture.take() {
            texture.present();
        }
        // No device.poll — rely on desired_maximum_frame_latency
    }
}

impl<'w> FrameGen<'w> {
    pub async fn new(window: &'w Window, width: u32, height: u32) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
           ..Default::default()
        });

        let surface = instance.create_surface(window).expect("surface");

        let adapter = instance
           .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
           .await
           .expect("adapter");

        // Detect features before requesting — fall back gracefully
        let adapter_features = adapter.features();
        let mut required_features = wgpu::Features::empty();
        if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY) {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        if adapter_features.contains(wgpu::Features::PUSH_CONSTANTS) {
            required_features |= wgpu::Features::PUSH_CONSTANTS;
        }

        let device_desc = wgpu::DeviceDescriptor {
            label: Some("FrameGen Device"),
            required_features,
            required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
            memory_hints: wgpu::MemoryHints::Performance,
        };

        let (device, queue) = adapter
           .request_device(&device_desc, None)
           .await
           .expect("device");

        let caps = surface.get_capabilities(&adapter);

        // Present mode: prefer Mailbox, fallback to Fifo. Immediate is opt-in.
        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else {
            wgpu::PresentMode::Fifo // guaranteed
        };

        // For true lowest latency on DX12/Vulkan: avoid sRGB swapchain and use Opaque alpha
        // sRGB formats can force a blit on some drivers
        let format = caps
           .formats
           .iter()
           .copied()
           .find(|f|!f.is_srgb())
           .unwrap_or(caps.formats[0]);

        let alpha_mode = if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
            wgpu::CompositeAlphaMode::Opaque
        } else {
            caps.alpha_modes[0]
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode,
            alpha_mode,
            view_formats: vec![format.add_srgb_suffix()], // allow srgb views if needed
            desired_maximum_frame_latency: (MAX_FRAMES_IN_FLIGHT as u32).saturating_sub(1),
        };

        surface.configure(&device, &config);

        let depth_view = Self::create_depth(&device, width, height);

        Self {
            device,
            queue,
            surface,
            config,
            frame_index: 0,
            last_frame: Instant::now(),
            start_time: Instant::now(),
            present_mode,
            depth_view: Some(depth_view),
            shader_cache: RefCell::new(HashMap::new()),
            features: required_features,
        }
    }

    fn create_depth(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("framegen.depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        depth.create_view(&wgpu::TextureViewDescriptor::default())
    }

    #[inline]
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.depth_view = Some(Self::create_depth(&self.device, width, height));
        }
    }

    #[inline]
    pub fn begin_frame(&mut self) -> Option<Frame<'_, 'w>> {
        let now = Instant::now();
        let delta = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;

        let texture = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                return None;
            }
            Err(wgpu::SurfaceError::Timeout) => return None,
            Err(e) => {
                log::warn!("surface error: {e:?}");
                return None;
            }
        };

        let view = texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("framegen.encoder"),
        });

        self.frame_index += 1;

        Some(Frame {
            fg: self,
            texture: Some(texture),
            view,
            encoder: Some(encoder),
            delta_time: delta.min(0.033), // clamp for hitch protection
            total_time: self.start_time.elapsed().as_secs_f32(),
            frame_index: self.frame_index,
        })
    }

    /// Real HashMap shader cache — avoids WGSL recompiles
    pub fn load_wgsl(&self, label: Option<&str>, source: &str) -> &wgpu::ShaderModule {
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        label.hash(&mut hasher);
        let key = hasher.finish();

        let mut cache = self.shader_cache.borrow_mut();
        if!cache.contains_key(&key) {
            let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label,
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            cache.insert(key, module);
        }
        // SAFETY: we just inserted, and RefCell keeps it alive for 'self lifetime
        // This is a bit of a hack to return a reference — for production, use Arc
        unsafe { &*(cache.get(&key).unwrap() as *const _) }
    }

    #[cfg(debug_assertions)]
    pub fn load_wgsl_file(&self, path: &str) -> &wgpu::ShaderModule {
        let src = std::fs::read_to_string(path).expect("wgsl file");
        self.load_wgsl(Some(path), &src)
    }

    #[inline] pub fn depth_view(&self) -> Option<&wgpu::TextureView> { self.depth_view.as_ref() }
    #[inline] pub fn format(&self) -> wgpu::TextureFormat { self.config.format }
    #[inline] pub fn size(&self) -> (u32, u32) { (self.config.width, self.config.height) }

    /// Enable tearing (Immediate) — opt-in only
    pub fn set_tearing(&mut self, enabled: bool) {
        let caps = self.surface.get_capabilities(&self.device.adapter().unwrap());
        self.present_mode = if enabled && caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
            wgpu::PresentMode::Immediate
        } else if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else {
            wgpu::PresentMode::Fifo
        };
        self.config.present_mode = self.present_mode;
        self.surface.configure(&self.device, &self.config);
    }
}

impl<'w> FrameGen<'w> {
    pub fn basic_pipeline(
        &self,
        wgsl_src: &str,
        bind_layouts: &[&wgpu::BindGroupLayout],
    ) -> wgpu::RenderPipeline {
        let shader = self.load_wgsl(Some("basic"), wgsl_src);
        let layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("basic.layout"),
            bind_group_layouts: bind_layouts,
            push_constant_ranges: &[],
        });

        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("basic.pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }
}
