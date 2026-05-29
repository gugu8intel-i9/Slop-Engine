use glam::{Mat4, Vec2, Vec3, Vec4};
use std::num::NonZeroU64;
use wgpu::*;

// ============================================================================
// CONFIG & CONSTANTS
// ============================================================================

pub const LOD_WORKGROUP_SIZE: u32 = 64;
pub const MAX_LOD_LEVELS: u32 = 8;

// Bit flags for GpuInstance::lod_flags
pub const LOD_FLAG_DISABLE_OCCLUSION: u32 = 1 << 0;
pub const LOD_FLAG_DISABLE_FRUSTUM_CULL: u32 = 1 << 1;
pub const LOD_FLAG_FORCE_HIGHEST: u32 = 1 << 2;
pub const LOD_FLAG_FORCE_LOWEST: u32 = 1 << 3;
pub const LOD_FLAG_DISABLE_HYSTERESIS: u32 = 1 << 4;
pub const LOD_FLAG_USE_IMPOSTOR: u32 = 1 << 5;
pub const LOD_FLAG_DEBUG_OVERRIDE: u32 = 1 << 6;

// LOD type enum for GPU (packed in 2 bits)
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LodType {
    Mesh = 0,
    Impostor = 1,
    Billboard = 2,
    Fallback = 3,
}

// ============================================================================
// CPU-SIDE CONFIGURATION
// ============================================================================

#[derive(Copy, Clone, Debug)]
pub struct LodConfig {
    pub reference_screen_height: f32,
    pub max_screen_error_pixels: f32,
    pub quality_bias: f32,           // [-1, 1]: -1=performance, +1=quality
    pub hysteresis: f32,             // [0, 1]: LOD transition smoothing
    pub enable_occlusion_culling: bool,
    pub enable_frustum_culling: bool,
    pub enable_indirect_draws: bool, // Output to indirect buffer
    pub enable_metrics: bool,        // Collect LOD distribution stats
    pub debug_visualize: bool,       // Color-code LODs for debugging
}

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            reference_screen_height: 1080.0,
            max_screen_error_pixels: 3.5,
            quality_bias: 0.0,
            hysteresis: 0.2,
            enable_occlusion_culling: true,
            enable_frustum_culling: true,
            enable_indirect_draws: true,
            enable_metrics: true,
            debug_visualize: false,
        }
    }
}

// ============================================================================
// GPU DATA STRUCTURES (bytemuck compatible)
// ============================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LodLevel {
    pub mesh_index: u32,
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub geometric_error: f32,    // World-space error at 1m distance
    pub impostor_data: u32,      // packed: texture_index(16b) | atlas_uv(16b)
    pub _reserved: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuInstance {
    pub model: [[f32; 4]; 4],    // Column-major
    pub bounds: Vec4,            // xyz=center, w=radius
    pub lod_config: Vec4,        // x=min_lod, y=max_lod, z=debug_lod_override, w=lod_flags
    pub tuning: Vec4,            // x=geom_scale, y=quality_bias, z=impostor_scale, w=reserved
}

impl GpuInstance {
    #[inline]
    pub fn set_lod_range(&mut self, min: u32, max: u32) {
        self.lod_config.x = min as f32;
        self.lod_config.y = max as f32;
    }
    #[inline]
    pub fn set_flags(&mut self, flags: u32) {
        self.lod_config.w = f32::from_bits(flags);
    }
    #[inline]
    pub fn flags(&self) -> u32 {
        self.lod_config.w as u32
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuCamera {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4], // For reprojection/occlusion
    pub position: Vec4,
    pub viewport: Vec4,          // x=width, y=height, z=focal_length_px, w=frame_id
    pub lod_params: Vec4,        // x=max_error_px, y=quality_bias, z=hysteresis, w=occlusion_enabled
}

impl GpuCamera {
    pub fn new(
        view_proj: Mat4,
        position: Vec3,
        viewport: Vec2,
        focal_length_px: f32,
        frame_id: u32,
        config: &LodConfig,
    ) -> Self {
        let screen_scale = viewport.y / config.reference_screen_height.max(1.0);
        Self {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            position: Vec4::new(position.x, position.y, position.z, 0.0),
            viewport: Vec4::new(viewport.x, viewport.y, focal_length_px, frame_id as f32),
            lod_params: Vec4::new(
                config.max_screen_error_pixels * screen_scale,
                config.quality_bias.clamp(-1.0, 1.0),
                config.hysteresis.clamp(0.0, 1.0),
                if config.enable_occlusion_culling { 1.0 } else { 0.0 },
            ),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSelection {
    pub data: u32,               // packed: lod_level(8b) | lod_type(2b) | visible(1b) | reserved
    pub mesh_index: u32,
    pub draw_params: Vec4,       // x=first_index, y=base_vertex, z=index_count, w=impostor_data
    pub debug_color: u32,        // RGBA8 for debug visualization
}

impl GpuSelection {
    #[inline]
    pub fn pack(lod: u32, lod_type: LodType, visible: bool) -> u32 {
        ((lod & 0xFF) << 3) | ((lod_type as u32 & 0x3) << 1) | (visible as u32)
    }
    #[inline]
    pub fn lod_level(&self) -> u32 { (self.data >> 3) & 0xFF }
    #[inline]
    pub fn lod_type(&self) -> LodType { 
        match (self.data >> 1) & 0x3 {
            0 => LodType::Mesh, 1 => LodType::Impostor, 
            2 => LodType::Billboard, _ => LodType::Fallback,
        }
    }
    #[inline]
    pub fn visible(&self) -> bool { (self.data & 1) != 0 }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndirect {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LodMetrics {
    pub visible_count: u32,
    pub culled_frustum: u32,
    pub culled_occlusion: u32,
    pub lod_distribution: [u32; MAX_LOD_LEVELS as usize],
    pub _padding: [u32; 4],
}

// ============================================================================
// PUSH CONSTANTS (frame-coherent, low-latency)
// ============================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LodPushConstants {
    pub instance_count: u32,
    pub level_count: u32,
    pub indirect_offset: u32,    // Byte offset into indirect buffer
    pub debug_mode: u32,         // 0=off, 1=wireframe LOD, 2=heatmap
}

// ============================================================================
// MAIN LOD SYSTEM
// ============================================================================

pub struct LodSystem {
    config: LodConfig,
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    metrics_buffer_size: u64,
}

impl LodSystem {
    pub fn new(device: &Device, config: LodConfig) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("lod_compute_shader"),
            source: ShaderSource::Wgsl(include_str!("lod_compute.wgsl")),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("lod_bind_group_layout"),
            entries: &[
                // 0: Camera (uniform)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<GpuCamera>() as u64),
                    },
                    count: None,
                },
                // 1: Instances (storage read)
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
                // 2: LOD levels table (storage read)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: Selections output (storage read-write)
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: Depth pyramid (texture)
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 5: Indirect draw buffer (optional, storage write)
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 6: Metrics buffer (optional, storage write)
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<LodMetrics>() as u64),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("lod_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<LodPushConstants>() as u32,
            }],
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("lod_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions {
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
            cache: None,
        });

        Self {
            config,
            pipeline,
            bind_group_layout,
            metrics_buffer_size: std::mem::size_of::<LodMetrics>() as u64,
        }
    }

    pub fn bind_group_layout(&self) -> &BindGroupLayout { &self.bind_group_layout }

    pub fn create_bind_group(
        &self,
        device: &Device,
        camera: &Buffer,
        instances: &Buffer,
        levels: &Buffer,
        selections: &Buffer,
        depth_pyramid: &TextureView,
        indirect_buffer: Option<&Buffer>,
        metrics_buffer: Option<&Buffer>,
    ) -> BindGroup {
        let mut entries = vec![
            BindGroupEntry { binding: 0, resource: camera.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: instances.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: levels.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: selections.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: BindingResource::TextureView(depth_pyramid) },
        ];
        if self.config.enable_indirect_draws {
            entries.push(BindGroupEntry {
                binding: 5,
                resource: indirect_buffer.unwrap().as_entire_binding(),
            });
        }
        if self.config.enable_metrics {
            entries.push(BindGroupEntry {
                binding: 6,
                resource: metrics_buffer.unwrap().as_entire_binding(),
            });
        }
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("lod_bind_group"),
            layout: &self.bind_group_layout,
            entries: &entries,
        })
    }

    pub fn update(
        &self,
        encoder: &mut CommandEncoder,
        bind_group: &BindGroup,
        instance_count: u32,
        level_count: u32,
        indirect_byte_offset: u32,
    ) {
        if instance_count == 0 { return; }

        let push_constants = LodPushConstants {
            instance_count,
            level_count,
            indirect_offset: indirect_byte_offset,
            debug_mode: if self.config.debug_visualize { 2 } else { 0 },
        };

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("lod_compute_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.push_debug_group("LOD Selection");
        pass.set_push_constants(0, bytemuck::bytes_of(&push_constants));
        
        let workgroups = (instance_count + LOD_WORKGROUP_SIZE - 1) / LOD_WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups, 1, 1);
        
        pass.pop_debug_group();
    }

    // Buffer size helpers
    #[inline]
    pub const fn selection_buffer_size(instances: u32) -> u64 {
        (instances as u64) * std::mem::size_of::<GpuSelection>() as u64
    }
    #[inline]
    pub const fn instance_buffer_size(instances: u32) -> u64 {
        (instances as u64) * std::mem::size_of::<GpuInstance>() as u64
    }
    #[inline]
    pub const fn level_buffer_size(levels: u32) -> u64 {
        (levels as u64) * std::mem::size_of::<LodLevel>() as u64
    }
    #[inline]
    pub const fn indirect_buffer_size(instances: u32) -> u64 {
        (instances as u64) * std::mem::size_of::<DrawIndirect>() as u64
    }
    pub const fn metrics_buffer_size(&self) -> u64 { self.metrics_buffer_size }
}
