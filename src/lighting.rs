// src/lighting.rs
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use std::num::NonZeroU32;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuLight {
    pub position: [f32; 3],
    pub radius: f32,
    pub color: [f32; 3],
    pub intensity: f32,
    pub direction: [f32; 3],
    pub spot_angle_cos: f32,
    pub light_type: u32,
    pub shadow_index: u32,
    pub _pad: [u32; 1],
}

pub struct ShadowAtlas {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub size: u32,
    pub tile_size: u32,
    pub tiles_x: u32,
    pub tiles_y: u32,
}

pub struct IblMaps {
    pub env_view: wgpu::TextureView,
    pub irradiance_view: wgpu::TextureView,
    pub prefiltered_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

pub struct Lighting {
    pub cluster_dims: [u32; 3],
    pub max_lights_per_cluster: u32,
    pub max_lights: u32,
    pub light_buffer: wgpu::Buffer,
    pub cluster_head_buffer: wgpu::Buffer,
    pub cluster_light_index_buffer: wgpu::Buffer,
    pub cluster_bind_group: wgpu::BindGroup,
    pub cluster_pipeline: wgpu::ComputePipeline,
    pub cluster_bind_group_layout: wgpu::BindGroupLayout,
    pub light_bind_group_layout: wgpu::BindGroupLayout,
    pub shadow_atlas: ShadowAtlas,
    pub ibl: Option<IblMaps>,
}

impl Lighting {
    /// Create a new Lighting system
    /// cluster_dims: [x, y, z] number of clusters in screen x,y and depth slices z
    /// max_lights_per_cluster: maximum lights stored per cluster
    /// max_lights: maximum lights in the scene
    pub fn new(
        device: &wgpu::Device,
        cluster_dims: [u32; 3],
        max_lights_per_cluster: u32,
        max_lights: u32,
        shadow_atlas_size: u32,
    ) -> Self {
        // Buffers
        let light_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("light_buffer"),
            size: (std::mem::size_of::<GpuLight>() as u64) * (max_lights as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // cluster head buffer: one u32 per cluster (index into list or 0xFFFFFFFF)
        let cluster_count = (cluster_dims[0] * cluster_dims[1] * cluster_dims[2]) as u64;
        let cluster_head_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster_head_buffer"),
            size: 4 * cluster_count,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // cluster light index buffer: cluster_count * max_lights_per_cluster entries (u32)
        let cluster_light_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster_light_index_buffer"),
            size: 4 * (cluster_count * (max_lights_per_cluster as u64)),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group layouts
        let cluster_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cluster_bind_group_layout"),
            entries: &[
                // 0: camera uniform (view_proj + inverse proj etc) - set by renderer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // 1: light storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // 2: cluster head buffer (u32 per cluster)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // 3: cluster light index buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });

        // Light bind group layout for fragment shader to read lights if needed
        let light_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("light_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // 1: cluster head buffer (optional for shader)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // 2: cluster light index buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });

        // Create compute shader module for clustering
        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cluster_compute"),
            source: wgpu::ShaderSource::Wgsl(CLUSTER_COMPUTE_WGSL.into()),
        });

        // Pipeline
        let cluster_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cluster_pipeline"),
            layout: None,
            module: &cs_module,
            entry_point: "main",
        });

        // Create bind group with placeholder camera buffer (renderer must set real camera bind group at index 0)
        // For now create a dummy camera buffer of 64 bytes
        let dummy_camera = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dummy_camera"),
            contents: &[0u8; 64],
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let cluster_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cluster_bind_group"),
            layout: &cluster_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dummy_camera.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: light_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cluster_head_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cluster_light_index_buffer.as_entire_binding() },
            ],
        });

        // Shadow atlas creation
        let tile_size = 1024u32; // per shadow tile size; tune for quality/perf
        let tiles_x = shadow_atlas_size / tile_size;
        let tiles_y = shadow_atlas_size / tile_size;
        let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow_atlas"),
            size: wgpu::Extent3d { width: shadow_atlas_size, height: shadow_atlas_size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let shadow_atlas = ShadowAtlas {
            texture: shadow_texture,
            view: shadow_view,
            size: shadow_atlas_size,
            tile_size,
            tiles_x,
            tiles_y,
        };

        Self {
            cluster_dims,
            max_lights_per_cluster,
            max_lights,
            light_buffer,
            cluster_head_buffer,
            cluster_light_index_buffer,
            cluster_bind_group,
            cluster_pipeline,
            cluster_bind_group_layout,
            light_bind_group_layout,
            shadow_atlas,
            ibl: None,
        }
    }

    /// Update lights buffer from CPU array of GpuLight
    pub fn update_lights(&self, queue: &wgpu::Queue, lights: &[GpuLight]) {
        let bytes = bytemuck::cast_slice(lights);
        queue.write_buffer(&self.light_buffer, 0, bytes);
    }

    /// Dispatch compute to build cluster lists
    /// camera_buffer must be a buffer containing camera parameters expected by the compute shader
    pub fn cluster_lights(&self, encoder: &mut wgpu::CommandEncoder, camera_buffer: &wgpu::Buffer) {
        // Update bind group 0 entry (camera) by creating a new bind group referencing camera_buffer
        let cluster_bind_group = encoder.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cluster_bind_group_frame"),
            layout: &self.cluster_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.light_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.cluster_head_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.cluster_light_index_buffer.as_entire_binding() },
            ],
        });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("cluster_compute_pass") });
        cpass.set_pipeline(&self.cluster_pipeline);
        cpass.set_bind_group(0, &cluster_bind_group, &[]);
        // Dispatch groups: choose workgroup size 8x8 for XY and cluster_dims.z for Z slices
        let (cx, cy, cz) = (self.cluster_dims[0], self.cluster_dims[1], self.cluster_dims[2]);
        let gx = (cx + 7) / 8;
        let gy = (cy + 7) / 8;
        cpass.dispatch_workgroups(gx, gy, cz);
    }

    /// Return a bind group for fragment shader to read lights and cluster lists
    pub fn create_light_bind_group(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("light_bind_group"),
            layout: &self.light_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.light_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.cluster_head_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.cluster_light_index_buffer.as_entire_binding() },
            ],
        })
    }
}

// WGSL compute shader for clustering
// - expects camera uniform with inverse projection, screen dims, near/far, cluster dims
// - reads lights and writes cluster head and index lists
const CLUSTER_COMPUTE_WGSL: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>;
    inv_proj: mat4x4<f32>;
    screen_size: vec2<f32>;
    z_near: f32;
    z_far: f32;
    cluster_dims: vec3<u32>;
};

struct Light {
    position: vec3<f32>;
    radius: f32;
    color: vec3<f32>;
    intensity: f32;
    direction: vec3<f32>;
    spot_angle_cos: f32;
    light_type: u32;
    shadow_index: u32;
    _pad: u32;
};

@group(0) @binding(0) var<uniform> uCamera: Camera;
@group(0) @binding(1) var<storage, read> uLights: array<Light>;
@group(0) @binding(2) var<storage, read_write> uClusterHead: array<u32>;
@group(0) @binding(3) var<storage, read_write> uClusterLightIndices: array<u32>;

// constants tuned by host
const MAX_LIGHTS_PER_CLUSTER: u32 = 64u; // must match host
const INVALID_INDEX: u32 = 0xffffffffu;

fn cluster_index(x: u32, y: u32, z: u32) -> u32 {
    return x + y * uCamera.cluster_dims.x + z * (uCamera.cluster_dims.x * uCamera.cluster_dims.y);
}

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cx = gid.x;
    let cy = gid.y;
    let cz = gid.z;
    if (cx >= uCamera.cluster_dims.x || cy >= uCamera.cluster_dims.y || cz >= uCamera.cluster_dims.z) {
        return;
    }
    let idx = cluster_index(cx, cy, cz);
    // initialize head to 0xffffffff
    uClusterHead[idx] = INVALID_INDEX;

    // compute cluster frustum bounds in view space
    // compute screen rect for this cluster
    let sx = f32(uCamera.screen_size.x);
    let sy = f32(uCamera.screen_size.y);
    let x0 = f32(cx) * sx / f32(uCamera.cluster_dims.x);
    let x1 = f32(cx + 1u) * sx / f32(uCamera.cluster_dims.x);
    let y0 = f32(cy) * sy / f32(uCamera.cluster_dims.y);
    let y1 = f32(cy + 1u) * sy / f32(uCamera.cluster_dims.y);

    // compute depth slice bounds (logarithmic or linear)
    let zn = uCamera.z_near;
    let zf = uCamera.z_far;
    let slice = f32(cz) / f32(uCamera.cluster_dims.z);
    let slice_next = f32(cz + 1u) / f32(uCamera.cluster_dims.z);
    // use linear for simplicity; host can precompute slice planes for better distribution
    let z0 = mix(zn, zf, slice);
    let z1 = mix(zn, zf, slice_next);

    // For each light, test sphere-frustum overlap (cheap conservative test)
    let light_count = arrayLength(&uLights);
    var write_pos: u32 = idx * MAX_LIGHTS_PER_CLUSTER;
    var count: u32 = 0u;
    var i: u32 = 0u;
    loop {
        if (i >= light_count) { break; }
        let L = uLights[i];
        // project light position to view space by assuming camera view is identity in this compute (host can pass view matrix if needed)
        // For simplicity assume lights are in view space already; host should transform lights to view space before upload.
        let lp = L.position;
        // quick z overlap
        if (lp.z + L.radius < z0 || lp.z - L.radius > z1) {
            i = i + 1u;
            continue;
        }
        // project to screen space using projection (approx)
        // compute approximate screen x/y by dividing by -z (assuming right-handed view)
        let ndc_x = lp.x / -lp.z;
        let ndc_y = lp.y / -lp.z;
        let screen_x = (ndc_x * 0.5 + 0.5) * uCamera.screen_size.x;
        let screen_y = (ndc_y * 0.5 + 0.5) * uCamera.screen_size.y;
        // conservative test: if screen center is within cluster rect expanded by radius in pixels
        let rad_pixels = L.radius * (uCamera.screen_size.x / (2.0 * tan(0.5))); // placeholder; host should pass focal scale
        if (screen_x + rad_pixels < x0 || screen_x - rad_pixels > x1 || screen_y + rad_pixels < y0 || screen_y - rad_pixels > y1) {
            i = i + 1u;
            continue;
        }
        // add light index to cluster list if space
        if (count < MAX_LIGHTS_PER_CLUSTER) {
            uClusterLightIndices[write_pos + count] = i;
            count = count + 1u;
        }
        i = i + 1u;
    }
    // If fewer than MAX_LIGHTS_PER_CLUSTER, fill remaining with INVALID_INDEX
    var j: u32 = count;
    loop {
        if (j >= MAX_LIGHTS_PER_CLUSTER) { break; }
        uClusterLightIndices[write_pos + j] = INVALID_INDEX;
        j = j + 1u;
    }
    // Optionally store count in head buffer high bits; here we store count in head buffer for quick access
    uClusterHead[idx] = count;
}
"#;
