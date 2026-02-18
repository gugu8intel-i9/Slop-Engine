// src/post_processing.rs
// High-performance Post-processing pipeline for wgpu
// Dependencies: wgpu, wgpu::util, bytemuck, glam
//
// Public API:
// - PostProcessor::new(device, format, options)
// - post.begin_frame(&mut encoder, &camera_uniform, &gbuffers)
// - post.execute(&mut encoder, &view, &output_view)
// - post.resize(width, height)
// - Toggle passes via PostOptions

use std::num::NonZeroU32;
use wgpu::util::DeviceExt;
use glam::{Vec2, Vec3, Mat4};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PostUniform {
    pub resolution: [f32; 2],
    pub inv_resolution: [f32; 2],
    pub time: f32,
    pub exposure: f32,
    pub bloom_threshold: f32,
    pub bloom_knee: f32,
    pub taa_alpha: f32,
    pub taa_history_feedback: f32,
}

pub struct PostOptions {
    pub enable_bloom: bool,
    pub bloom_threshold: f32,
    pub bloom_knee: f32,
    pub bloom_scales: u32,
    pub enable_taa: bool,
    pub enable_ssao: bool,
    pub enable_ssr: bool,
    pub enable_dof: bool,
    pub enable_motion_blur: bool,
    pub enable_color_grading: bool,
    pub use_half_res: bool,
}

impl Default for PostOptions {
    fn default() -> Self {
        Self {
            enable_bloom: true,
            bloom_threshold: 1.0,
            bloom_knee: 0.5,
            bloom_scales: 5,
            enable_taa: true,
            enable_ssao: true,
            enable_ssr: false,
            enable_dof: false,
            enable_motion_blur: false,
            enable_color_grading: false,
            use_half_res: true,
        }
    }
}

pub struct PostProcessor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    options: PostOptions,

    // Uniform buffer and bind group
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    uniform_layout: wgpu::BindGroupLayout,

    // Ping-pong render targets (full, half, quarter)
    rt_full: wgpu::TextureView,
    rt_half: wgpu::TextureView,
    rt_quarter: wgpu::TextureView,
    // history buffers for TAA and motion
    taa_history: Option<wgpu::TextureView>,
    velocity_view: Option<wgpu::TextureView>,

    // pipelines
    blit_pipeline: wgpu::RenderPipeline,
    bloom_prefilter_pipeline: wgpu::ComputePipeline,
    bloom_down_pipeline: wgpu::ComputePipeline,
    bloom_up_pipeline: wgpu::ComputePipeline,
    taa_pipeline: wgpu::ComputePipeline,
    ssao_pipeline: wgpu::ComputePipeline,
    ssr_pipeline: Option<wgpu::ComputePipeline>,
    dof_pipeline: Option<wgpu::ComputePipeline>,
    color_grade_pipeline: Option<wgpu::ComputePipeline>,

    // bind group caches
    bind_group_cache: Vec<wgpu::BindGroup>,

    // internal frame jitter for TAA
    taa_jitter: [f32; 2],
    frame_index: u64,
}

impl PostProcessor {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat, width: u32, height: u32, options: PostOptions) -> Self {
        let device = device.clone();
        let queue = queue.clone();

        // Uniform layout
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post_uniform_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                }
            ],
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post_uniform_buffer"),
            size: std::mem::size_of::<PostUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post_uniform_bind_group"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() }],
        });

        // Create render targets (full, half, quarter) as textures and views
        let (rt_full, rt_half, rt_quarter) = Self::create_render_targets(&device, width, height, format, options.use_half_res);

        // Load shader modules (WGSL strings below)
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post_blit"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/post_blit.wgsl").into()),
        });

        // Blit pipeline (fullscreen triangle)
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post_pipeline_layout"),
            bind_group_layouts: &[&uniform_layout],
            push_constant_ranges: &[],
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("post_blit_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: "fs_blit",
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Create compute pipelines for bloom, taa, ssao, etc. (shaders included below)
        let bloom_prefilter_cs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom_prefilter"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bloom_prefilter.wgsl").into()),
        });
        let bloom_prefilter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bloom_prefilter_pipeline"),
            layout: None,
            module: &bloom_prefilter_cs,
            entry_point: "main",
        });

        let bloom_down_cs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom_down"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bloom_down.wgsl").into()),
        });
        let bloom_down_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bloom_down_pipeline"),
            layout: None,
            module: &bloom_down_cs,
            entry_point: "main",
        });

        let bloom_up_cs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom_up"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bloom_up.wgsl").into()),
        });
        let bloom_up_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bloom_up_pipeline"),
            layout: None,
            module: &bloom_up_cs,
            entry_point: "main",
        });

        // TAA compute
        let taa_cs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("taa"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/taa.wgsl").into()),
        });
        let taa_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("taa_pipeline"),
            layout: None,
            module: &taa_cs,
            entry_point: "main",
        });

        // SSAO compute
        let ssao_cs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssao"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ssao.wgsl").into()),
        });
        let ssao_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ssao_pipeline"),
            layout: None,
            module: &ssao_cs,
            entry_point: "main",
        });

        // Optional pipelines (SSR, DoF, color grade) can be created similarly; omitted for brevity
        let ssr_pipeline = None;
        let dof_pipeline = None;
        let color_grade_pipeline = None;

        Self {
            device,
            queue,
            format,
            width,
            height,
            options,
            uniform_buffer,
            uniform_bind_group,
            uniform_layout,
            rt_full,
            rt_half,
            rt_quarter,
            taa_history: None,
            velocity_view: None,
            blit_pipeline,
            bloom_prefilter_pipeline,
            bloom_down_pipeline,
            bloom_up_pipeline,
            taa_pipeline,
            ssao_pipeline,
            ssr_pipeline,
            dof_pipeline,
            color_grade_pipeline,
            bind_group_cache: Vec::new(),
            taa_jitter: [0.0, 0.0],
            frame_index: 0,
        }
    }

    fn create_render_targets(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat, use_half: bool)
        -> (wgpu::TextureView, wgpu::TextureView, wgpu::TextureView)
    {
        let full = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pp_rt_full"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let full_view = full.create_view(&wgpu::TextureViewDescriptor::default());

        let half_w = if use_half { (width + 1) / 2 } else { width };
        let half_h = if use_half { (height + 1) / 2 } else { height };
        let half = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pp_rt_half"),
            size: wgpu::Extent3d { width: half_w, height: half_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let half_view = half.create_view(&wgpu::TextureViewDescriptor::default());

        let quarter = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pp_rt_quarter"),
            size: wgpu::Extent3d { width: (half_w + 1) / 2, height: (half_h + 1) / 2, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let quarter_view = quarter.create_view(&wgpu::TextureViewDescriptor::default());

        (full_view, half_view, quarter_view)
    }

    /// Begin frame: update uniforms, compute jitter for TAA, and prepare resources.
    /// `camera_proj` is used to compute jitter; `time` and `exposure` are passed to uniforms.
    pub fn begin_frame(&mut self, camera_proj: Mat4, time: f32, exposure: f32) {
        self.frame_index = self.frame_index.wrapping_add(1);
        // TAA jitter using Halton sequence (2,3)
        if self.options.enable_taa {
            let halton = |index: u64, base: u32| -> f32 {
                let mut f = 1.0f32;
                let mut r = 0.0f32;
                let mut i = index;
                while i > 0 {
                    f = f / base as f32;
                    r = r + f * (i % base as u64) as f32;
                    i = i / base as u64;
                }
                r
            };
            let jitter_x = halton(self.frame_index, 2) - 0.5;
            let jitter_y = halton(self.frame_index, 3) - 0.5;
            self.taa_jitter = [jitter_x, jitter_y];
        } else {
            self.taa_jitter = [0.0, 0.0];
        }

        // Update uniform buffer
        let uni = PostUniform {
            resolution: [self.width as f32, self.height as f32],
            inv_resolution: [1.0 / self.width as f32, 1.0 / self.height as f32],
            time,
            exposure,
            bloom_threshold: self.options.bloom_threshold,
            bloom_knee: self.options.bloom_knee,
            taa_alpha: 0.1,
            taa_history_feedback: 0.9,
        };
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uni));
    }

    /// Execute the post chain. `encoder` is the command encoder for the frame.
    /// `src_view` is the HDR color texture produced by the main pass.
    /// `depth_view` and `velocity_view` are optional inputs for SSAO/SSR/TAA.
    pub fn execute(&mut self, encoder: &mut wgpu::CommandEncoder, src_view: &wgpu::TextureView, depth_view: Option<&wgpu::TextureView>, velocity_view: Option<&wgpu::TextureView>, output_view: &wgpu::TextureView) {
        // 1) Bloom: prefilter -> downsample chain -> upsample composite
        if self.options.enable_bloom {
            // prefilter: threshold bright areas into half-res
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("bloom_prefilter") });
                cpass.set_pipeline(&self.bloom_prefilter_pipeline);
                // bind src_view and rt_half as storage; dispatch groups sized to half resolution
                // binding setup omitted for brevity; assume bind groups created and cached
                let groups_x = ((self.width + 1) / 2 + 7) / 8;
                let groups_y = ((self.height + 1) / 2 + 7) / 8;
                cpass.dispatch_workgroups(groups_x, groups_y, 1);
            }
            // downsample chain
            for level in 0..self.options.bloom_scales {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("bloom_down") });
                cpass.set_pipeline(&self.bloom_down_pipeline);
                // dispatch sized to current mip
                let w = (self.width >> (level + 1)).max(1);
                let h = (self.height >> (level + 1)).max(1);
                let gx = (w + 7) / 8;
                let gy = (h + 7) / 8;
                cpass.dispatch_workgroups(gx, gy, 1);
            }
            // upsample chain
            for level in (0..self.options.bloom_scales).rev() {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("bloom_up") });
                cpass.set_pipeline(&self.bloom_up_pipeline);
                let w = (self.width >> (level + 1)).max(1);
                let h = (self.height >> (level + 1)).max(1);
                let gx = (w + 7) / 8;
                let gy = (h + 7) / 8;
                cpass.dispatch_workgroups(gx, gy, 1);
            }
        }

        // 2) SSAO (if enabled)
        if self.options.enable_ssao {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ssao") });
            cpass.set_pipeline(&self.ssao_pipeline);
            // bind depth_view and rt_quarter as output; dispatch groups sized to quarter resolution
            let gx = ((self.width + 3) / 4 + 7) / 8;
            let gy = ((self.height + 3) / 4 + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        // 3) TAA (if enabled)
        if self.options.enable_taa {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("taa") });
            cpass.set_pipeline(&self.taa_pipeline);
            // bind src_view, history, velocity; dispatch full resolution groups
            let gx = (self.width + 7) / 8;
            let gy = (self.height + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        // 4) SSR, DoF, Motion blur, Color grading (omitted for brevity)...
        // 5) Final composite: blit pipeline reads HDR source, bloom, ssao, color grade and writes to output_view
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("post_final"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: true },
                })],
                depth_stencil_attachment: None,
            });
            rpass.set_pipeline(&self.blit_pipeline);
            rpass.set_bind_group(0, &self.uniform_bind_group, &[]);
            // bind other resources via cached bind groups if needed
            rpass.draw(0..3, 0..1); // fullscreen triangle
        }
    }

    /// Resize render targets
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == self.width && height == self.height { return; }
        self.width = width;
        self.height = height;
        let (full, half, quarter) = Self::create_render_targets(&self.device, width, height, self.format, self.options.use_half_res);
        self.rt_full = full;
        self.rt_half = half;
        self.rt_quarter = quarter;
    }
}
