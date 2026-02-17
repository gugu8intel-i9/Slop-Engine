// src/mesh_material_system.rs
// Single-file Mesh + Material system optimized for performance and matching the dual 2D/3D WGSL shader.
// Dependencies: wgpu, wgpu::util, bytemuck, optionally image for texture loading.
// Add to Cargo.toml: wgpu, bytemuck = "1.13", image = { version = "0.24", optional = true }

use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use std::num::NonZeroU32;

/// Vertex format (matches WGSL shader: position, normal, uv, tangent)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub tangent: [f32; 3],
}

impl Vertex {
    pub const ATTRS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x3, // position
        1 => Float32x3, // normal
        2 => Float32x2, // uv
        3 => Float32x3, // tangent
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

/// Mesh: vertex + index buffers, index count, optional AABB for culling
pub struct Mesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
}

impl Mesh {
    pub fn new(device: &wgpu::Device, vertices: &[Vertex], indices: &[u32]) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_vertex_buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_index_buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Compute AABB
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for v in vertices {
            for i in 0..3 {
                min[i] = min[i].min(v.position[i]);
                max[i] = max[i].max(v.position[i]);
            }
        }

        Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            aabb_min: min,
            aabb_max: max,
        }
    }
}

/// Texture wrapper with view + sampler
pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl Texture {
    /// Create a 1x1 white dummy texture. Reuse this for missing textures to keep bind groups stable.
    pub fn dummy(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let rgba = [255u8, 255u8, 255u8, 255u8];
        let size = wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dummy_texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(4),
                rows_per_image: NonZeroU32::new(1),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("dummy_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self { texture, view, sampler }
    }

    /// Optional: load from bytes (PNG/JPEG) using the image crate
    #[allow(dead_code)]
    pub fn from_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        label: &str,
    ) -> anyhow::Result<Self> {
        let img = image::load_from_memory(bytes)?.to_rgba8();
        let (width, height) = img.dimensions();
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &img,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(4 * width),
                rows_per_image: NonZeroU32::new(height),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(&format!("{}_sampler", label)),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Ok(Self { texture, view, sampler })
    }
}

/// Material uniform layout (matches WGSL MaterialParams)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MaterialParams {
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub ao_factor: f32,
    pub flags: u32,
    pub _pad: [f32; 3],
}

/// Material: uniform buffer + bind group
pub struct Material {
    pub params_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl Material {
    /// Create a material. Provide optional textures; pass a dummy texture to avoid None in bind group.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        params: MaterialParams,
        base_color: Option<&Texture>,
        mr: Option<&Texture>,
        normal: Option<&Texture>,
        ao: Option<&Texture>,
        dummy: &Texture,
    ) -> Self {
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material_params_buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let base_view = base_color.map(|t| &t.view).unwrap_or(&dummy.view);
        let mr_view = mr.map(|t| &t.view).unwrap_or(&dummy.view);
        let normal_view = normal.map(|t| &t.view).unwrap_or(&dummy.view);
        let ao_view = ao.map(|t| &t.view).unwrap_or(&dummy.view);
        let sampler = base_color.map(|t| &t.sampler).unwrap_or(&dummy.sampler);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(base_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(mr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        Self {
            params_buffer,
            bind_group,
        }
    }

    /// Update material params (fast GPU update)
    pub fn update_params(&self, queue: &wgpu::Queue, params: &MaterialParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }
}

/// Model uniform (per-object model matrix)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ModelUniform {
    pub model: [[f32; 4]; 4],
}

pub struct Model {
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl Model {
    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, model_matrix: [[f32; 4]; 4]) -> Self {
        let uniform = ModelUniform { model: model_matrix };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("model_uniform_buffer"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("model_bind_group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        Self {
            uniform_buffer,
            bind_group,
        }
    }

    pub fn update(&self, queue: &wgpu::Queue, model_matrix: [[f32; 4]; 4]) {
        let uniform = ModelUniform { model: model_matrix };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    }
}

/// Create bind group layouts matching the WGSL shader
pub fn create_material_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("material_bind_group_layout"),
        entries: &[
            // 0: MaterialParams uniform
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 1: base_color_tex
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            // 2: mr_tex
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            // 3: normal_tex
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            // 4: ao_tex
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            // 5: sampler
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

pub fn create_camera_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("camera_bind_group_layout"),
        entries: &[
            // 0: CameraUniform (view_proj, view_pos, mode)
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 1: Light uniform
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

pub fn create_model_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("model_bind_group_layout"),
        entries: &[
            // 0: model matrix uniform
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Create pipeline layout combining camera, material, and model layouts
pub fn create_pipeline_layout(
    device: &wgpu::Device,
    camera_layout: &wgpu::BindGroupLayout,
    material_layout: &wgpu::BindGroupLayout,
    model_layout: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline_layout"),
        bind_group_layouts: &[camera_layout, material_layout, model_layout],
        push_constant_ranges: &[],
    })
}

/// Create render pipeline (2D or 3D). Use the same shader module for both; change depth/blend state.
pub fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader_module: &wgpu::ShaderModule,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    is_2d: bool,
) -> wgpu::RenderPipeline {
    let vertex_buffers = &[Vertex::layout()];
    let primitive = wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        strip_index_format: None,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: Some(wgpu::Face::Back),
        unclipped_depth: false,
        polygon_mode: wgpu::PolygonMode::Fill,
        conservative: false,
    };

    let depth_stencil = depth_format.map(|fmt| wgpu::DepthStencilState {
        format: fmt,
        depth_write_enabled: !is_2d,
        depth_compare: wgpu::CompareFunction::LessEqual,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(if is_2d { "pipeline_2d" } else { "pipeline_3d" }),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader_module,
            entry_point: "vs_main",
            buffers: vertex_buffers,
        },
        fragment: Some(wgpu::FragmentState {
            module: shader_module,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: if is_2d {
                    Some(wgpu::BlendState::ALPHA_BLENDING)
                } else {
                    Some(wgpu::BlendState::REPLACE)
                },
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive,
        depth_stencil,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    })
}

/// Minimal draw call: set buffers, bind groups, draw indexed
pub fn draw_mesh<'a>(
    render_pass: &mut wgpu::RenderPass<'a>,
    mesh: &'a Mesh,
    material: &'a Material,
    model: &'a Model,
) {
    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
    render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    render_pass.set_bind_group(1, &material.bind_group, &[]);
    render_pass.set_bind_group(2, &model.bind_group, &[]);
    render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
}

/// Example usage notes (not executed here):
///
/// 1. Create bind group layouts:
///    let camera_layout = create_camera_bind_group_layout(&device);
///    let material_layout = create_material_bind_group_layout(&device);
///    let model_layout = create_model_bind_group_layout(&device);
///
/// 2. Create pipeline layout and shader module:
///    let pipeline_layout = create_pipeline_layout(&device, &camera_layout, &material_layout, &model_layout);
///    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("shader"), source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()) });
///
/// 3. Create pipelines:
///    let pipeline_2d = create_render_pipeline(&device, &pipeline_layout, &shader_module, surface_format, None, true);
///    let pipeline_3d = create_render_pipeline(&device, &pipeline_layout, &shader_module, surface_format, Some(depth_format), false);
///
/// 4. Create dummy texture once and reuse:
///    let dummy = Texture::dummy(&device, &queue);
///
/// 5. Create materials and models:
///    let mat = Material::new(&device, &queue, &material_layout, params, Some(&base_tex), Some(&mr_tex), Some(&normal_tex), Some(&ao_tex), &dummy);
///    let model = Model::new(&device, &model_layout, model_matrix);
///
/// 6. In render pass:
///    render_pass.set_pipeline(&pipeline_3d);
///    render_pass.set_bind_group(0, &camera_bind_group, &[]);
///    draw_mesh(&mut render_pass, &mesh, &mat, &model);
///
/// Performance tips:
/// - Batch by pipeline -> material -> mesh.
/// - Use instancing for repeated draws.
/// - Keep bind group layouts stable; use dummy textures to avoid reallocation.
/// - Update uniform buffers only when necessary (use queue.write_buffer).
