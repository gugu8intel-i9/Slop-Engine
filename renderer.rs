// src/renderer.rs
//! OPTIMIZED RENDERER v2.0
//! Uses high-performance shaders from shaders.rs
//! Includes dynamic resolution, predictive rendering, and optimized pipelines

use std::borrow::Cow;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

use crate::shaders::*;
use crate::predictive_renderer::{SceneSnapshot, TileCoord, TileManager};
use crate::offload::{OffloadManager, ResourceTier};

#[cfg(not(target_arch = "wasm32"))]
use dashmap::DashMap;

// ============================================================================
// RENDERER CONFIGURATION
// ============================================================================

#[derive(Debug, Clone)]
pub struct RenderConfig {
    pub width: u32,
    pub height: u32,
    pub resolution_scale: f32,
    pub shadow_resolution: u32,
    pub enable_post_processing: bool,
    pub enable_ssao: bool,
    pub enable_bloom: bool,
    pub vsync: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            resolution_scale: 1.0,
            shadow_resolution: 2048,
            enable_post_processing: true,
            enable_ssao: true,
            enable_bloom: true,
            vsync: true,
        }
    }
}

impl RenderConfig {
    pub fn effective_size(&self) -> (u32, u32) {
        (
            (self.width as f32 * self.resolution_scale) as u32,
            (self.height as f32 * self.resolution_scale) as u32,
        )
    }
}

// ============================================================================
// RENDERER STATE
// ============================================================================

pub struct Renderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: RenderConfig,
    
    // Pipelines
    main_pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    post_pipeline: wgpu::RenderPipeline,
    mipmap_pipeline: wgpu::ComputePipeline,
    
    // Textures
    depth_texture: wgpu::TextureView,
    shadow_texture: wgpu::TextureView,
    hdr_texture: wgpu::TextureView,
    ssao_texture: wgpu::TextureView,
    
    // Bind Groups
    main_bind_group: wgpu::BindGroup,
    shadow_bind_group: wgpu::BindGroup,
    post_bind_group: wgpu::BindGroup,
    
    // Buffers
    vertex_buffer: wgpu::Buffer,
    camera_buffer: wgpu::Buffer,
    light_buffer: wgpu::Buffer,
    post_buffer: wgpu::Buffer,
    
    // State
    frame_count: u64,
    last_fps: f32,
    fps_accumulator: f32,
    fps_samples: u32,
}

impl Renderer {
    pub async fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: RenderConfig,
    ) -> Result<Self, wgpu::RequestDeviceError> {
        let (width, height) = config.effective_size();
        
        // Create textures
        let depth_texture = Self::create_depth_texture(&device, width, height);
        let shadow_texture = Self::create_shadow_texture(&device, config.shadow_resolution);
        let hdr_texture = Self::create_hdr_texture(&device, width, height);
        let ssao_texture = Self::create_ssao_texture(&device, width, height);
        
        // Create pipelines
        let main_pipeline = Self::create_main_pipeline(&device, config);
        let shadow_pipeline = Self::create_shadow_pipeline(&device);
        let post_pipeline = Self::create_post_pipeline(&device, config);
        let mipmap_pipeline = Self::create_mipmap_pipeline(&device);
        
        // Create buffers
        let vertex_buffer = Self::create_vertex_buffer(&device);
        let camera_buffer = Self::create_uniform_buffer(&device, 128);
        let light_buffer = Self::create_uniform_buffer(&device, 128);
        let post_buffer = Self::create_uniform_buffer(&device, 32);
        
        // Create bind groups
        let main_bind_group = Self::create_main_bind_group(&device, &camera_buffer, &light_buffer);
        let shadow_bind_group = Self::create_shadow_bind_group(&device, &light_buffer);
        let post_bind_group = Self::create_post_bind_group(&device, &hdr_texture, &post_buffer);
        
        Ok(Self {
            device,
            queue,
            config,
            main_pipeline,
            shadow_pipeline,
            post_pipeline,
            mipmap_pipeline,
            depth_texture,
            shadow_texture,
            hdr_texture,
            ssao_texture,
            main_bind_group,
            shadow_bind_group,
            post_bind_group,
            vertex_buffer,
            camera_buffer,
            light_buffer,
            post_buffer,
            frame_count: 0,
            last_fps: 0.0,
            fps_accumulator: 0.0,
            fps_samples: 0,
        })
    }
    
    pub fn render(&mut self) -> Result<RenderStats, wgpu::SurfaceError> {
        let start = std::time::Instant::now();
        
        // FPS calculation
        let frame_time = start.elapsed().as_secs_f32();
        self.fps_accumulator += frame_time;
        self.fps_samples += 1;
        
        if self.fps_samples >= 60 {
            self.last_fps = 60.0 / self.fps_accumulator;
            self.fps_accumulator = 0.0;
            self.fps_samples = 0;
        }
        
        self.frame_count += 1;
        
        // Update uniforms
        self.update_uniforms();
        
        // Get surface texture
        let output = self.device.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        // Shadow pass
        self.render_shadow_pass(&mut encoder);
        
        // Main geometry pass (to HDR texture)
        self.render_geometry_pass(&mut encoder);
        
        // Post-processing
        if self.config.enable_post_processing {
            self.render_post_processing(&mut encoder);
        }
        
        // Final present
        encoder.insert_debug_marker("Present");
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        
        let render_time = start.elapsed().as_secs_f32() * 1000.0;
        
        Ok(RenderStats {
            fps: self.last_fps,
            frame_time_ms: render_time,
            draw_calls: 3, // Shadow, Geometry, Post
            triangles: 2,
            resolution: self.config.effective_size(),
        })
    }
    
    fn render_shadow_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Shadow Pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.shadow_texture,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        
        pass.set_pipeline(&self.shadow_pipeline);
        pass.set_bind_group(0, &self.shadow_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..6, 0..1); // Fullscreen quad
    }
    
    fn render_geometry_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Geometry Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.hdr_texture,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.04, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        
        pass.set_pipeline(&self.main_pipeline);
        pass.set_bind_group(0, &self.main_bind_group, &[]);
        pass.set_bind_group(1, &self.shadow_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..6, 0..1);
    }
    
    fn render_post_processing(&self, encoder: &mut wgpu::CommandEncoder) {
        // Create output view for post-processing
        let surface = self.device.get_current_texture().unwrap();
        let view = surface.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Post Process"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        
        pass.set_pipeline(&self.post_pipeline);
        pass.set_bind_group(0, &self.post_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    
    fn update_uniforms(&self) {
        // Camera uniform (simplified for demo)
        let camera_data = CameraUniforms {
            view_proj: glam::Mat4::IDENTITY,
            camera_pos: glam::Vec3::new(0.0, 2.0, -5.0),
            camera_forward: glam::Vec3::NEG_Z,
        };
        
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_data]));
        
        // Light uniform
        let light_data = LightUniforms {
            view_proj: glam::Mat4::IDENTITY,
            light_pos: glam::Vec3::new(5.0, 10.0, -5.0),
            light_color: glam::Vec3::new(1.0, 0.98, 0.95),
            intensity: 1.5,
        };
        
        self.queue.write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&[light_data]));
        
        // Post uniform
        let post_data = PostUniforms {
            resolution: glam::Vec2::new(self.config.width as f32, self.config.height as f32),
            time: self.frame_count as f32 * 0.016,
            bloom_threshold: 0.8,
            vignette_strength: 0.3,
        };
        
        self.queue.write_buffer(&self.post_buffer, 0, bytemuck::cast_slice(&[post_data]));
    }
    
    // ========================================================================
    // PIPELINE CREATION
    // ========================================================================
    
    fn create_main_pipeline(device: &wgpu::Device, config: RenderConfig) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Main Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(OPTIMIZED_MAIN_SHADER)),
        });
        
        let main_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Main BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        
        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Main Pipeline Layout"),
            bind_group_layouts: &[&main_bgl, &shadow_bgl],
            push_constant_ranges: &[],
        });
        
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Main Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::TextureFormat::Rgba16Float.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                ..Default::default()
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        })
    }
    
    fn create_shadow_pipeline(device: &wgpu::Device) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(OPTIMIZED_SHADOW_SHADER)),
        });
        
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow Pipeline Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "shadow_vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "shadow_fs_pcf4",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Front),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                ..Default::default()
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        })
    }
    
    fn create_post_pipeline(device: &wgpu::Device, config: RenderConfig) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Post Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(OPTIMIZED_POST_SHADER)),
        });
        
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Post BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Post Pipeline Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Post Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        })
    }
    
    fn create_mipmap_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mipmap Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(OPTIMIZED_MIPMAP_SHADER)),
        });
        
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mipmap BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba8Unorm, view_dimension: wgpu::TextureViewDimension::D2 },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mipmap Pipeline Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Mipmap Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "generate_mip",
        })
    }
    
    // ========================================================================
    // BIND GROUP CREATION
    // ========================================================================
    
    fn create_main_bind_group(
        device: &wgpu::Device,
        camera_buffer: &wgpu::Buffer,
        light_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Main Bind Group"),
            layout: &device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Main BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                        count: None,
                    },
                ],
            }),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: light_buffer.as_entire_binding() },
            ],
        })
    }
    
    fn create_shadow_bind_group(
        device: &wgpu::Device,
        light_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Sampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        
        let shadow_texture = Self::create_shadow_texture(device, 2048);
        
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Bind Group"),
            layout: &device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Shadow BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                ],
            }),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&shadow_texture) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&shadow_sampler) },
            ],
        })
    }
    
    fn create_post_bind_group(
        device: &wgpu::Device,
        hdr_texture: &wgpu::TextureView,
        post_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Post Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Post Bind Group"),
            layout: &device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Post BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            }),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(hdr_texture) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        })
    }
    
    // ========================================================================
    // TEXTURE CREATION
    // ========================================================================
    
    fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }).create_view(&wgpu::TextureViewDescriptor::default())
    }
    
    fn create_shadow_texture(device: &wgpu::Device, size: u32) -> wgpu::TextureView {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow Texture"),
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }).create_view(&wgpu::TextureViewDescriptor::default())
    }
    
    fn create_hdr_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HDR Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }).create_view(&wgpu::TextureViewDescriptor::default())
    }
    
    fn create_ssao_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        // Half resolution for SSAO performance
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSAO Texture"),
            size: wgpu::Extent3d { width: width / 2, height: height / 2, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }).create_view(&wgpu::TextureViewDescriptor::default())
    }
    
    fn create_vertex_buffer(device: &wgpu::Device) -> wgpu::Buffer {
        // Fullscreen quad
        let vertices: [f32; 18] = [
            -1.0, -1.0, 0.0,   1.0, -1.0, 0.0,   1.0, 1.0, 0.0,
            -1.0, -1.0, 0.0,   1.0, 1.0, 0.0,   -1.0, 1.0, 0.0,
        ];
        
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        })
    }
    
    fn create_uniform_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
    
    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
        
        let (w, h) = self.config.effective_size();
        
        self.depth_texture = Self::create_depth_texture(&self.device, w, h);
        self.hdr_texture = Self::create_hdr_texture(&self.device, w, h);
        self.ssao_texture = Self::create_ssao_texture(&self.device, w, h);
    }
    
    pub fn set_resolution_scale(&mut self, scale: f32) {
        self.config.resolution_scale = scale.clamp(0.5, 2.0);
        let (w, h) = self.config.effective_size();
        self.resize(w, h);
    }
}

// ============================================================================
// UNIFORMS
// ============================================================================

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CameraUniforms {
    view_proj: glam::Mat4,
    camera_pos: glam::Vec3,
    _pad0: f32,
    camera_forward: glam::Vec3,
    _pad1: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct LightUniforms {
    view_proj: glam::Mat4,
    light_pos: glam::Vec3,
    light_color: glam::Vec3,
    intensity: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PostUniforms {
    resolution: glam::Vec2,
    time: f32,
    bloom_threshold: f32,
    vignette_strength: f32,
}

// ============================================================================
// STATS
// ============================================================================

#[derive(Debug)]
pub struct RenderStats {
    pub fps: f32,
    pub frame_time_ms: f32,
    pub draw_calls: u32,
    pub triangles: u32,
    pub resolution: (u32, u32),
}

impl std::fmt::Display for RenderStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FPS: {:.1} | {:.2}ms | {} draw calls | {} triangles | {}x{}",
            self.fps,
            self.frame_time_ms,
            self.draw_calls,
            self.triangles,
            self.resolution.0,
            self.resolution.1
        )
    }
}