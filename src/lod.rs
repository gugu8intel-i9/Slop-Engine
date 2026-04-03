use wgpu::*;
use glam::{Mat4, Vec3, Vec2};

/// Configuration for the LoD behavior.
/// Implements a "Screen-Space Error" metric strategy.
pub struct LodConfig {
    /// The base screen height used to calculate relative error pixels.
    pub reference_screen_height: f32,
    /// The maximum allowed screen-space error in pixels before switching LoD.
    pub max_screen_error_pixels: f32,
    /// Bias towards higher quality LoDs (0.0 = standard, 1.0 = force highest).
    pub quality_bias: f32,
}

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            reference_screen_height: 1080.0,
            max_screen_error_pixels: 4.0, // Adjust based on visual tolerance
            quality_bias: 0.0,
        }
    }
}

/// A single LoD level metadata stored on the CPU.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LodLevel {
    /// Index of the mesh buffer to use.
    pub mesh_index: u32,
    /// Geometric error of this mesh relative to the highest quality version.
    /// This is usually the Hausdorff distance to the original mesh.
    pub geometric_error: f32,
    /// Padding for alignment.
    pub _padding: [f32; 2],
}

/// The main LoD System.
pub struct LodSystem {
    config: LodConfig,
    compute_pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

impl LodSystem {
    pub fn new(device: &Device, config: LodConfig) -> Self {
        // WGSL Shader loading
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Lod Compute Shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/lod_compute.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Lod Bind Group Layout"),
            entries: &[
                // 0: Camera Uniforms
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: Instance Input Buffer (Read)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: Draw Command Output Buffer (Write)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: Hi-Z Depth Pyramid (Texture)
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Lod Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Lod Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            config,
            compute_pipeline,
            bind_group_layout,
        }
    }

    /// Dispatches the LoD selection compute shader.
    /// This should run before the main rendering pass.
    pub fn update(
        &self,
        encoder: &mut CommandEncoder,
        camera_buffer: &Buffer,
        instance_buffer: &Buffer,
        output_command_buffer: &Buffer,
        depth_pyramid_view: &TextureView,
        instance_count: u32,
    ) {
        let bind_group = encoder.device().create_bind_group(&BindGroupDescriptor {
            label: Some("Lod Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                Binding::Buffer(0, camera_buffer.as_entire_buffer_binding()),
                Binding::Buffer(1, instance_buffer.as_entire_buffer_binding()),
                Binding::Buffer(2, output_command_buffer.as_entire_buffer_binding()),
                Binding::Texture(3, depth_pyramid_view),
            ],
        });

        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("LoD Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        
        // Dispatch 1 thread per instance.
        // In a production engine, you might dispatch threadgroups based on chunk size.
        let workgroup_count = (instance_count + 63) / 64; 
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
}
