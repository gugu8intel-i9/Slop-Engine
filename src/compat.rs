// src/compat.rs
// A small compatibility layer that ties together DepthTexture, camera uniform/bind group,
// and a basic pipeline + render helper so your existing files work together.
// Assumes `depth_texture.rs` exists and exposes `DepthTexture` as in earlier messages.
//
// Usage pattern (example):
//   let depth = DepthTexture::new(&device, &config);
//   let (cam_buf, cam_bg, cam_bgl) = compat::create_camera_bind_group(&device);
//   let pipeline = compat::create_basic_pipeline(&device, &config, &cam_bgl, Some(DepthTexture::DEPTH_FORMAT));
//   compat::render_with_depth(&mut encoder, &frame_view, &depth, &pipeline, &cam_bg, vertex_buffer, index_buffer, index_count);

use std::borrow::Cow;
use wgpu::util::DeviceExt;

pub mod prelude {
    pub use wgpu;
    pub use bytemuck::{Pod, Zeroable};
    pub use glam;
}

use prelude::*;
use crate::depth_texture::DepthTexture;

/// Camera uniform (view-proj)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
        }
    }

    pub fn from_mat4(m: glam::Mat4) -> Self {
        Self {
            view_proj: m.to_cols_array_2d(),
        }
    }
}

/// Create a camera buffer + bind group + bind group layout.
/// Returns (buffer, bind_group, bind_group_layout).
pub fn create_camera_bind_group(
    device: &wgpu::Device,
) -> (wgpu::Buffer, wgpu::BindGroup, wgpu::BindGroupLayout) {
    let camera_uniform = CameraUniform::new();
    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("compat_camera_buffer"),
        contents: bytemuck::cast_slice(&[camera_uniform]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("compat_camera_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compat_camera_bg"),
        layout: &camera_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: camera_buffer.as_entire_binding(),
        }],
    });

    (camera_buffer, camera_bg, camera_bgl)
}

/// Create a minimal shader module and a render pipeline that is compatible with DepthTexture.
/// If `depth_format` is None, depth_stencil will be disabled.
pub fn create_basic_pipeline(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    camera_bgl: &wgpu::BindGroupLayout,
    depth_format: Option<wgpu::TextureFormat>,
) -> wgpu::RenderPipeline {
    let shader_source = r#"
        struct Camera { view_proj: mat4x4<f32>; };
        @group(0) @binding(0) var<uniform> camera: Camera;

        struct VertexInput { @location(0) position: vec3<f32>; };
        struct VertexOutput {
            @builtin(position) clip_pos: vec4<f32>;
            @location(0) world_pos: vec3<f32>;
        };

        @vertex
        fn vs_main(in: VertexInput) -> VertexOutput {
            var out: VertexOutput;
            out.clip_pos = camera.view_proj * vec4<f32>(in.position, 1.0);
            out.world_pos = in.position;
            return out;
        }

        @fragment
        fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
            let c = (in.world_pos * 0.5) + vec3<f32>(0.5, 0.5, 0.5);
            return vec4<f32>(c, 1.0);
        }
    "#;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("compat_basic_shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("compat_pipeline_layout"),
        bind_group_layouts: &[camera_bgl],
        push_constant_ranges: &[],
    });

    let vertex_buffers = [wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x3,
        }],
    }];

    let depth_state = depth_format.map(|fmt| wgpu::DepthStencilState {
        format: fmt,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Less,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("compat_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &vertex_buffers,
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: depth_state,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    })
}

/// Helper render function that wires the pipeline, camera bind group and depth texture into a render pass.
/// - `frame_view` is the swapchain/frame texture view
/// - `depth` is your DepthTexture
/// - `vertex_buffer` is required; `index_buffer` is optional
/// - `index_count` is number of indices or vertices to draw
pub fn render_with_depth(
    encoder: &mut wgpu::CommandEncoder,
    frame_view: &wgpu::TextureView,
    depth: &DepthTexture,
    pipeline: &wgpu::RenderPipeline,
    camera_bind_group: &wgpu::BindGroup,
    vertex_buffer: &wgpu::Buffer,
    index_buffer: Option<&wgpu::Buffer>,
    index_count: u32,
) {
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("compat_main_pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: frame_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.05,
                    g: 0.05,
                    b: 0.1,
                    a: 1.0,
                }),
                store: true,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &depth.view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: true,
            }),
            stencil_ops: None,
        }),
    });

    rpass.set_pipeline(pipeline);
    rpass.set_bind_group(0, camera_bind_group, &[]);
    rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
    if let Some(ib) = index_buffer {
        rpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..index_count, 0, 0..1);
    } else {
        rpass.draw(0..index_count, 0..1);
    }
}
