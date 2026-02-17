// src/skybox.rs
// High-performance Skybox and IBL generator for wgpu
// Dependencies: wgpu, wgpu::util, image, bytemuck, anyhow, glam
//
// Usage summary:
// let mut sky = Skybox::new(&device, &queue, atlas_size, options);
// sky.load_hdr_from_bytes(&device, &queue, hdr_bytes).await?;
// sky.generate_pmrem(&device, &queue).await?;
// // In render loop:
// render_pass.set_pipeline(&sky.pipeline);
// render_pass.set_bind_group(0, &sky.bind_group, &[]);
// render_pass.draw(0..36, 0..1);

use std::sync::Arc;
use wgpu::util::DeviceExt;
use anyhow::{Result, Context};
use glam::{Mat4, Vec3};
use bytemuck::{Pod, Zeroable};

pub struct SkyboxOptions {
    pub cubemap_size: u32,        // base resolution for cubemap faces
    pub prefilter_mip_levels: u32,// number of mip levels for prefiltered specular
    pub irradiance_size: u32,     // resolution for irradiance cubemap
    pub brdf_lut_size: u32,       // resolution for BRDF LUT (2D)
    pub use_atmosphere: bool,     // enable atmospheric scattering procedural sky
    pub max_texture_bytes: u64,   // memory cap for sky textures
}

impl Default for SkyboxOptions {
    fn default() -> Self {
        Self {
            cubemap_size: 1024,
            prefilter_mip_levels: 5,
            irradiance_size: 64,
            brdf_lut_size: 512,
            use_atmosphere: false,
            max_texture_bytes: 256 * 1024 * 1024,
        }
    }
}

/// GPU resources for skybox and IBL
pub struct Skybox {
    // cubemap and derived maps
    pub env_cubemap: wgpu::Texture,         // original HDR cubemap (if provided)
    pub env_view: wgpu::TextureView,
    pub prefiltered: wgpu::Texture,         // prefiltered specular cubemap with mipmaps
    pub prefiltered_view: wgpu::TextureView,
    pub irradiance: wgpu::Texture,          // irradiance cubemap
    pub irradiance_view: wgpu::TextureView,
    pub brdf_lut: wgpu::Texture,            // 2D BRDF LUT
    pub brdf_view: wgpu::TextureView,

    // samplers
    pub linear_sampler: wgpu::Sampler,
    pub prefilter_sampler: wgpu::Sampler,

    // pipelines and bind groups
    pub sky_pipeline: wgpu::RenderPipeline,
    pub sky_bind_group: wgpu::BindGroup,
    pub sky_bind_layout: wgpu::BindGroupLayout,

    // compute pipelines for PMREM and irradiance
    pub prefilter_pipeline: wgpu::ComputePipeline,
    pub irradiance_pipeline: wgpu::ComputePipeline,
    pub brdf_pipeline: wgpu::ComputePipeline,

    // options and device handle
    pub options: SkyboxOptions,
    device: Arc<wgpu::Device>,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SkyUniform {
    sun_dir: [f32; 4],
    sun_color: [f32; 4],
    turbidity: f32,
    exposure: f32,
    _pad: [f32; 2],
}

impl Skybox {
    /// Create empty Skybox with default placeholder textures (1x1) and pipelines prepared.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, options: SkyboxOptions) -> Self {
        let device = Arc::new(device.clone());

        // create 1x1 white placeholder texture for env_cubemap (use texture view arrays for cubemap)
        let placeholder = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sky_placeholder"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        // write white pixel to all faces
        let white: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
        let bytes = bytemuck::cast_slice(&white);
        // queue.write_texture requires bytes; for float formats we upload via staging; for brevity skip here

        let env_view = placeholder.create_view(&wgpu::TextureViewDescriptor {
            label: Some("env_view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        // create prefiltered cubemap with mipmaps
        let prefiltered = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("prefiltered_cubemap"),
            size: wgpu::Extent3d { width: options.cubemap_size, height: options.cubemap_size, depth_or_array_layers: 6 },
            mip_level_count: options.prefilter_mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let prefiltered_view = prefiltered.create_view(&wgpu::TextureViewDescriptor {
            label: Some("prefiltered_view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            base_mip_level: 0,
            mip_level_count: Some(options.prefilter_mip_levels),
            ..Default::default()
        });

        // irradiance cubemap
        let irradiance = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("irradiance_cubemap"),
            size: wgpu::Extent3d { width: options.irradiance_size, height: options.irradiance_size, depth_or_array_layers: 6 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let irradiance_view = irradiance.create_view(&wgpu::TextureViewDescriptor {
            label: Some("irradiance_view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        // BRDF LUT 2D
        let brdf_lut = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("brdf_lut"),
            size: wgpu::Extent3d { width: options.brdf_lut_size, height: options.brdf_lut_size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let brdf_view = brdf_lut.create_view(&wgpu::TextureViewDescriptor::default());

        // samplers
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sky_linear_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let prefilter_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("prefilter_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: options.prefilter_mip_levels as f32 - 1.0,
            ..Default::default()
        });

        // create bind group layout for sky (group 3 in PBR shader)
        let sky_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sky_bind_layout"),
            entries: &[
                // 0: env cubemap
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::Cube, sample_type: wgpu::TextureSampleType::Float { filterable: true } },
                    count: None,
                },
                // 1: prefiltered cubemap
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::Cube, sample_type: wgpu::TextureSampleType::Float { filterable: true } },
                    count: None,
                },
                // 2: irradiance cubemap
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::Cube, sample_type: wgpu::TextureSampleType::Float { filterable: true } },
                    count: None,
                },
                // 3: brdf lut
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } },
                    count: None,
                },
                // 4: sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // create bind group
        let sky_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky_bind_group"),
            layout: &sky_bind_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&env_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&prefiltered_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&irradiance_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&brdf_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&linear_sampler) },
            ],
        });

        // create sky render pipeline (cube)
        let vs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sky_vert"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/skybox_vert.wgsl").into()),
        });
        let fs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sky_frag"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/skybox_frag.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sky_pipeline_layout"),
            bind_group_layouts: &[&sky_bind_layout],
            push_constant_ranges: &[],
        });

        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sky_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Front),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // compute pipelines for prefilter, irradiance, brdf
        let prefilter_cs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prefilter_cs"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/prefilter_env.wgsl").into()),
        });
        let prefilter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prefilter_pipeline"),
            layout: None,
            module: &prefilter_cs,
            entry_point: "main",
        });

        let irradiance_cs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("irradiance_cs"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/irradiance_conv.wgsl").into()),
        });
        let irradiance_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("irradiance_pipeline"),
            layout: None,
            module: &irradiance_cs,
            entry_point: "main",
        });

        let brdf_cs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brdf_cs"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/brdf_lut.wgsl").into()),
        });
        let brdf_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("brdf_pipeline"),
            layout: None,
            module: &brdf_cs,
            entry_point: "main",
        });

        Self {
            env_cubemap: placeholder,
            env_view,
            prefiltered,
            prefiltered_view,
            irradiance,
            irradiance_view,
            brdf_lut,
            brdf_view,
            linear_sampler,
            prefilter_sampler,
            sky_pipeline,
            sky_bind_group,
            sky_bind_layout,
            prefilter_pipeline,
            irradiance_pipeline,
            brdf_pipeline,
            options,
            device,
        }
    }

    /// Load an HDR equirectangular image and convert to cubemap on GPU.
    /// Accepts bytes of an HDR/EXR/PNG file. Decoding is done on CPU (image crate) then uploaded.
    pub async fn load_hdr_from_bytes(&mut self, queue: &wgpu::Queue, bytes: &[u8]) -> Result<()> {
        // decode using image crate (supports HDR via image-rs with feature)
        let dyn = image::load_from_memory(bytes).context("failed to decode image")?;
        let img = dyn.to_rgba32f();
        let (w, h) = img.dimensions();
        // create a 2D texture for equirectangular source
        let src = self.device.create_texture_with_data(queue, &wgpu::TextureDescriptor {
            label: Some("equirect_src"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        }, bytemuck::cast_slice(&img.into_vec()));

        // run compute shader to sample equirectangular -> cubemap faces into env_cubemap
        // For brevity, we assume a compute shader exists that performs this conversion (not included here).
        // After conversion, update self.env_cubemap and self.env_view to the generated cubemap.

        Ok(())
    }

    /// Generate PMREM prefiltered cubemap, irradiance cubemap, and BRDF LUT on GPU.
    /// This is the heavy GPU work but runs once per environment or on demand.
    pub async fn generate_pmrem(&mut self, queue: &wgpu::Queue) -> Result<()> {
        // 1) Generate BRDF LUT
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("brdf_lut_encoder") });
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("brdf_compute") });
            cpass.set_pipeline(&self.brdf_pipeline);
            // bind output as storage texture or via bind group; dispatch sized to brdf_lut_size/8
            let groups = (self.options.brdf_lut_size + 7) / 8;
            cpass.dispatch_workgroups(groups, groups, 1);
            queue.submit(Some(encoder.finish()));
        }

        // 2) Generate irradiance cubemap via convolution compute
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("irradiance_encoder") });
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("irradiance_compute") });
            cpass.set_pipeline(&self.irradiance_pipeline);
            // bind env_cubemap as input and irradiance as output; dispatch per face
            let groups = (self.options.irradiance_size + 7) / 8;
            for face in 0..6 {
                // set face index via push constants or uniform; omitted for brevity
                cpass.dispatch_workgroups(groups, groups, 1);
            }
            queue.submit(Some(encoder.finish()));
        }

        // 3) Prefilter specular into mipmapped cubemap using importance sampling compute
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("prefilter_encoder") });
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("prefilter_compute") });
            cpass.set_pipeline(&self.prefilter_pipeline);
            // for each mip level and face dispatch appropriate groups
            for mip in 0..self.options.prefilter_mip_levels {
                let mip_size = (self.options.cubemap_size >> mip).max(1);
                let groups = (mip_size + 7) / 8;
                for face in 0..6 {
                    // set uniforms: roughness = mip / (mips-1), face index, mip level
                    cpass.dispatch_workgroups(groups, groups, 1);
                }
            }
            queue.submit(Some(encoder.finish()));
        }

        Ok(())
    }

    /// Render skybox cube. Call inside render pass with pipeline set to sky_pipeline.
    /// This function sets bind group and draws a unit cube (36 vertices).
    pub fn render<'a>(&'a self, rpass: &mut wgpu::RenderPass<'a>) {
        rpass.set_pipeline(&self.sky_pipeline);
        rpass.set_bind_group(0, &self.sky_bind_group, &[]);
        // cube vertex buffer is assumed to be set by renderer or use draw call with no vertex buffer and a vertex shader that generates cube vertices by vertex_index
        rpass.draw(0..36, 0..1);
    }
}
