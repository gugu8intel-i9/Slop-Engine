// Engine specific structures
struct CameraData {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position: vec3<f32>,
    time: f32, // Used for temporal jitter
    screen_size: vec2<f32>,
    tan_half_fov: f32,
    _padding: f32,
};

struct InstanceInput {
    @location(0) world_position: vec3<f32>,
    @location(1) world_matrix: mat4x4<f32>,
    @location(2) bounding_sphere_radius: f32,
    @location(3) lod_levels: array<vec4<f32>, 4>, // x: mesh_idx, y: error, z: distance_start, w: padding
};

struct DrawCommand {
    vertex_buffer_index: u32,
    instance_index: u32, // Maps back to the instance buffer
    lod_level: u32,
    _padding: u32,
};

// Bind Groups
@group(0) @binding(0) var<uniform> camera: CameraData;
@group(0) @binding(1) var<storage, read> instances: array<InstanceInput>;
@group(0) @binding(2) var<storage, read_write> out_commands: array<DrawCommand>;
@group(0) @binding(3) var depth_pyramid: texture_depth_2d;

// Constants
const PI: f32 = 3.14159265359;
const LOD_ERROR_THRESHOLD: f32 = 4.0; // Pixels

// --- The Innovation: Stochastic LoD Selection ---
// Instead of hard boundaries, we use a probability function based on screen-space error.
// This removes popping and relies on TAA to blend the transition.
fn get_stochastic_lod(instance: InstanceInput) -> u32 {
    let world_pos = instance.world_position;
    let radius = instance.bounding_sphere_radius;

    // 1. Frustum Culling (Basic)
    let clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    if (abs(clip_pos.x) > clip_pos.w + radius || 
        abs(clip_pos.y) > clip_pos.w + radius || 
        clip_pos.z < -radius) {
        return 99u; // Culled (Invalid ID)
    }

    // 2. Hi-Z Occlusion Culling
    // Project bounding sphere to texture coordinates
    let uv = (clip_pos.xy / clip_pos.w) * 0.5 + 0.5;
    // Sample the depth pyramid (Assuming Mip 0 is generated)
    // Note: In a real engine, you'd sample the correct mip level based on bounding box projection.
    let depth_sample = textureSampleLevel(depth_pyramid, uv, 0.0).r; 
    
    // Simple occlusion check (linearizing depth usually required, simplified here)
    if (clip_pos.z > depth_sample * 100.0) { 
        // If behind depth, we might still draw it if it's large, but for now, we cull.
        // Conservative culling is safer: out_commands[idx].lod_level = 99u; return;
    }

    // 3. Screen-Space Error Calculation
    // Distance to camera
    let dist = length(camera.position - world_pos);
    
    // How many pixels does the radius take up on screen?
    // screen_diameter = (radius * 2.0) / tan(fov) * (screen_height / dist) 
    // Simplified heuristic for screen coverage:
    let screen_coverage = radius / dist;

    // Iterate through LoD levels
    // We assume instance.lod_levels is sorted by quality (High -> Low)
    // LOD 0: Error 0.0 (Reference)
    // LOD 1: Error 2.0 
    // LOD 2: Error 8.0
    
    // Calculate geometric error projected to screen pixels
    // Error_pixels = (Geometric_Error / Distance) * (Screen_Height / (2 * tan(FOV/2)))
    // Let's assume a constant denominator for simplicity: PROJ_CONST
    
    let num_lods = 4u;
    var selected_lod = num_lods - 1u; // Default to lowest LoD

    // Temporal Jitter for Stochastic Selection
    // We generate a noise value that is stable per object per frame but changes over time.
    // This effectively "dithers" the LoD boundary.
    let jitter = fract(sin(dot(world_pos.xz, vec2<f32>(12.9898, 78.233))) + camera.time);

    // Loop from high quality (0) to low quality
    for (var i = 0u; i < num_lods; i++) {
        let lod_error = instance.lod_levels[i].y; // Geometric error of this LOD
        if (lod_error <= 0.0) { 
            selected_lod = i; 
            break; 
        }

        // Project error to screen pixels
        let error_pixels = (lod_error / dist) * camera.screen_size.y;

        // Standard LoD: if error_pixels < threshold, pick this.
        // Innovative LoD: Use a threshold band.
        // If error_pixels is 6, and threshold is 4, standard LoD would skip.
        // Stochastic LoD: Add jitter.
        
        let lower_threshold = LOD_ERROR_THRESHOLD * 0.8;
        let upper_threshold = LOD_ERROR_THRESHOLD * 1.2;

        if (error_pixels < lower_threshold) {
            selected_lod = i;
            break;
        } else if (error_pixels < upper_threshold) {
            // Transition Zone
            // Probability to stay at higher quality = (upper - current) / (upper - lower)
            let prob = (upper_threshold - error_pixels) / (upper_threshold - lower_threshold);
            
            // Use jitter to decide
            if (jitter < prob) {
                selected_lod = i;
                break;
            }
            // If jitter check fails, we fall through to the next lower quality LoD
        }
    }

    return selected_lod;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Basic bounds check
    if (idx >= arrayLength(&instances)) {
        return;
    }

    let instance = instances[idx];
    let lod_index = get_stochastic_lod(instance);

    // Output Draw Command
    // If culled (99), we could write to a "culled" buffer, but here we just skip writing 
    // or write invalid data that the render pass ignores.
    
    out_commands[idx].instance_index = idx;
    out_commands[idx].lod_level = lod_index;
    
    // We retrieve the actual mesh index from the pre-baked LoD metadata
    if (lod_index < 4u) {
        out_commands[idx].vertex_buffer_index = u32(instance.lod_levels[lod_index].x);
    } else {
        out_commands[idx].vertex_buffer_index = 99u; // Marker for culled
    }
}
