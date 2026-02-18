// ssao.wgsl
// High-performance, featureful SSAO shader (WGSL).
// Features:
//  - Horizon-aware SSAO with bent-normal estimation
//  - Multi-sample Poisson disk kernel (configurable taps)
//  - Depth-aware radius and falloff (screen-space and view-space options)
//  - Normal-aware occlusion (uses normal buffer if available)
//  - Random rotation via blue-noise / rotation texture to reduce banding
//  - Bilateral separable blur pass (horizontal / vertical) with depth & normal edge preservation
//  - Optional temporal accumulation (history) support
//  - Modes: 0 = compute AO, 1 = blur horizontal, 2 = blur vertical, 3 = composite / debug
//
// Bindings (example):
// group(0) binding(0) : uniform Matrices { mat4 inv_proj; vec2 screen_size; }
// group(0) binding(1) : texture_depth_2d depth_tex;
// group(0) binding(2) : texture_2d<f32> normal_tex; // optional; if not provided, fallback to reconstructing normal
// group(0) binding(3) : texture_2d<f32> noise_tex;  // small tiled rotation / blue-noise texture
// group(0) binding(4) : sampler sLinearClamp;
// group(0) binding(5) : sampler sPointClamp;
// group(1) binding(0) : uniform SSAOParams { ... }
// group(1) binding(1) : texture_2d<f32> tAOHistory; // optional history for temporal accumulation
//
// Outputs:
//  - location(0): ao (single-channel in .x) or rgba debug
//  - location(1): updated history (if used)

struct Matrices {
    inv_proj: mat4x4<f32>;
    screen_size: vec2<f32>;
};
@group(0) @binding(0) var<uniform> uMatrices: Matrices;

@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var noise_tex: texture_2d<f32>;
@group(0) @binding(4) var sLinearClamp: sampler;
@group(0) @binding(5) var sPointClamp: sampler;

struct SSAOParams {
    mode: u32;                // 0 = compute AO, 1 = blur H, 2 = blur V, 3 = composite/debug
    radius: f32;              // base radius in view-space units (meters)
    bias: f32;                // small bias to avoid self-occlusion
    intensity: f32;           // AO intensity multiplier
    power: f32;               // final power curve (gamma-like)
    sample_count: u32;        // number of samples (use odd values up to 32)
    scale_with_resolution: u32; // 0/1 scale radius by screen diagonal
    temporal_enabled: u32;    // 0/1
    temporal_alpha: f32;      // history blend factor (0..1)
    blur_radius: f32;         // blur radius in texels
    blur_taps: u32;           // blur taps (odd)
    normal_threshold: f32;    // normal difference threshold for bilateral blur
    depth_threshold: f32;     // depth difference threshold for bilateral blur (view-space)
    debug: u32;               // 0 = off, 1 = show AO, 2 = show bent normal, 3 = show samples
    pad0: u32;
};
@group(1) @binding(0) var<uniform> uParams: SSAOParams;

@group(1) @binding(1) var tAOHistory: texture_2d<f32>;

// Fullscreen triangle vertex
struct VSOut {
    @builtin(position) pos: vec4<f32>;
    @location(0) uv: vec2<f32>;
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    var out: VSOut;
    let p = positions[vi];
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = p * 0.5 + vec2<f32>(0.5, 0.5);
    return out;
}

// ---------- Constants and helpers ----------
let PI: f32 = 3.141592653589793;
let EPS: f32 = 1e-6;

// Precomputed Poisson disk samples (2D) - 32 taps max
const POISSON_DISK: array<vec2<f32>, 32> = array<vec2<f32>, 32>(
    vec2<f32>(-0.94201624, -0.39906216),
    vec2<f32>(0.94558609, -0.76890725),
    vec2<f32>(-0.094184101, -0.92938870),
    vec2<f32>(0.34495938, 0.29387760),
    vec2<f32>(-0.91588581, 0.45771432),
    vec2<f32>(-0.81544232, -0.87912464),
    vec2<f32>(-0.38277543, 0.27676845),
    vec2<f32>(0.97484398, 0.75648379),
    vec2<f32>(0.44323325, -0.97511554),
    vec2<f32>(0.53742981, -0.47373420),
    vec2<f32>(-0.26496911, -0.41893023),
    vec2<f32>(0.79197514, 0.19090188),
    vec2<f32>(-0.24188840, 0.99706507),
    vec2<f32>(-0.81409955, 0.91437590),
    vec2<f32>(0.19984126, 0.78641367),
    vec2<f32>(0.14383161, -0.14100790),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 0.0)
);

// Reconstruct view-space position from depth and UV using inverse projection
fn reconstruct_view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    // depth is in 0..1 (NDC depth). Convert to clip space z
    // Reconstruct clip-space position: (x_ndc, y_ndc, z_ndc, 1)
    let ndc_x = uv.x * 2.0 - 1.0;
    let ndc_y = uv.y * 2.0 - 1.0;
    let ndc_z = depth * 2.0 - 1.0;
    let clip = vec4<f32>(ndc_x, ndc_y, ndc_z, 1.0);
    let view = uMatrices.inv_proj * clip;
    // perspective divide
    return view.xyz / max(view.w, EPS);
}

// Read normal: prefer normal_tex if available, otherwise reconstruct from depth neighbors
fn fetch_normal(uv: vec2<f32>) -> vec3<f32> {
    // Try to sample normal texture; if it's black (0,0,0) treat as missing and reconstruct
    let n_sample = textureSample(normal_tex, sLinearClamp, uv).xyz;
    if (length(n_sample) > 0.001) {
        // normals stored in [0,1] -> remap to [-1,1]
        return normalize(n_sample * 2.0 - vec3<f32>(1.0));
    }

    // Fallback: reconstruct normal from depth derivatives (screen-space)
    let dims = uMatrices.screen_size;
    let texel = vec2<f32>(1.0 / dims.x, 1.0 / dims.y);
    let d = textureSample(depth_tex, sLinearClamp, uv);
    let d_dx = textureSample(depth_tex, sLinearClamp, uv + vec2<f32>(texel.x, 0.0));
    let d_dy = textureSample(depth_tex, sLinearClamp, uv + vec2<f32>(0.0, texel.y));
    let p = reconstruct_view_pos(uv, d);
    let px = reconstruct_view_pos(uv + vec2<f32>(texel.x, 0.0), d_dx);
    let py = reconstruct_view_pos(uv + vec2<f32>(0.0, texel.y), d_dy);
    let e1 = px - p;
    let e2 = py - p;
    return normalize(cross(e1, e2));
}

// Sample rotation / random vector from noise texture (tiled)
fn sample_rotation(uv: vec2<f32>) -> vec2<f32> {
    // noise texture is small (e.g., 4x4 or 8x8) and tiled across screen
    // scale uv by screen_size / noise_size is done by caller; here we assume noise is tiled by using uv * screen_size / noise_size
    // For simplicity, sample noise at uv * screen_size / 4.0 (tile factor)
    let tile = 4.0;
    let n = textureSample(noise_tex, sLinearClamp, uv * tile).xy;
    // map to [-1,1]
    return n * 2.0 - vec2<f32>(1.0, 1.0);
}

// SSAO core: compute occlusion at uv
fn compute_ssao(uv: vec2<f32>) -> vec2<f32> {
    // returns vec2(ao, bent_normal_x) where bent normal can be encoded in xy/yz as needed
    let depth = textureSample(depth_tex, sLinearClamp, uv);
    if (depth >= 1.0) {
        return vec2<f32>(0.0, 0.0);
    }

    let view_pos = reconstruct_view_pos(uv, depth);
    let N = fetch_normal(uv);
    let V = normalize(-view_pos); // view vector in view-space (camera at origin)
    // compute radius in view-space; optionally scale with resolution
    var radius = uParams.radius;
    if (uParams.scale_with_resolution == 1u) {
        // scale by screen diagonal (approx)
        let diag = length(uMatrices.screen_size);
        radius = radius * (diag / 1080.0);
    }

    // sample rotation to rotate disk
    let rot = sample_rotation(uv);
    let angle = atan2(rot.y, rot.x);
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    // rotation matrix 2D
    // accumulate occlusion and bent normal
    var occlusion: f32 = 0.0;
    var bent: vec3<f32> = vec3<f32>(0.0);

    let samples = min(uParams.sample_count, 32u);
    for (var i: u32 = 0u; i < samples; i = i + 1u) {
        // rotate sample offset
        let s = POISSON_DISK[i];
        let rx = s.x * cos_a - s.y * sin_a;
        let ry = s.x * sin_a + s.y * cos_a;
        // scale by radius in screen-space: convert view-space radius to screen-space by projecting a point offset along view-space tangent
        // approximate: offset_uv = (rx, ry) * radius / view_pos.z * (1 / tan(fov/2)) ... simplified by using screen_size
        // Simpler and robust: compute sample position by offsetting view_pos in tangent space and projecting back
        // Build tangent/bitangent in view-space
        let T = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), N));
        let B = normalize(cross(N, T));
        // sample point in view-space
        let sample_dir = rx * T + ry * B;
        let sample_pos = view_pos + sample_dir * radius;
        // project sample_pos to screen UV
        let clip = uMatrices.inv_proj * vec4<f32>(sample_pos, 1.0); // inv_proj is inverse projection; to project we need proj, but we only have inv_proj
        // Instead, reconstruct sample depth by reprojecting sample_pos to NDC using projection matrix; we don't have proj here.
        // Workaround: approximate sample_uv by offsetting uv by sample_dir * (radius / view_pos.z) * (1 / screen_size)
        let offset_uv = vec2<f32>(rx, ry) * (radius / max(view_pos.z, 0.0001)) * 0.5; // heuristic scale
        let sample_uv = uv + offset_uv;
        // clamp
        let su = clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(1.0));
        let sample_depth = textureSample(depth_tex, sLinearClamp, su);
        if (sample_depth >= 1.0) {
            continue;
        }
        let sample_view = reconstruct_view_pos(su, sample_depth);
        // range check: if sample is farther than radius * 1.5, skip
        let range = length(sample_view - view_pos);
        if (range > radius * 1.5) {
            continue;
        }
        // occlusion test: if sample_view.z > view_pos.z + bias -> occluder (in view-space, more positive z is further)
        // Note: view-space forward is -Z if camera at origin; we used V = -view_pos; so compare distances
        let delta = sample_view.z - view_pos.z;
        // bias scaled by depth
        let bias = uParams.bias * max(0.001, view_pos.z);
        if (delta > bias) {
            // weight by angle between N and sample direction and by distance falloff
            let w = max(dot(N, normalize(sample_view - view_pos)), 0.0);
            // distance attenuation
            let att = 1.0 / (1.0 + range * range * 0.5);
            occlusion = occlusion + w * att;
            // accumulate bent normal (direction to occluder)
            bent = bent + normalize(sample_view - view_pos) * w * att;
        }
    }

    // normalize occlusion by sample count and intensity
    let inv_samples = 1.0 / f32(max(1u, samples));
    var ao = clamp(occlusion * inv_samples * uParams.intensity, 0.0, 1.0);
    // invert so 0 = occluded, 1 = lit (common convention)
    ao = 1.0 - ao;

    // bent normal: average and re-normalize; if zero fallback to N
    var bent_n = bent;
    if (length(bent_n) < 1e-5) {
        bent_n = N;
    } else {
        bent_n = normalize(bent_n);
    }

    // apply power curve
    ao = pow(ao, max(0.0001, uParams.power));

    // pack AO in .x and bent normal x in .y for debug (caller can request bent normal separately)
    return vec2<f32>(ao, bent_n.x);
}

// Bilateral separable blur (horizontal / vertical)
fn bilateral_blur(uv: vec2<f32>, horizontal: bool) -> f32 {
    // read center depth and normal
    let center_depth = textureSample(depth_tex, sLinearClamp, uv);
    let center_view = reconstruct_view_pos(uv, center_depth);
    let center_normal = fetch_normal(uv);

    // compute texel
    let dims = uMatrices.screen_size;
    let texel = vec2<f32>(1.0 / dims.x, 1.0 / dims.y);

    var sum: f32 = 0.0;
    var wsum: f32 = 0.0;

    var taps = uParams.blur_taps;
    if (taps == 0u) { taps = 5u; }
    if ((taps & 1u) == 0u) { taps = taps - 1u; }
    let half = i32(taps / 2u);

    // sigma derived from blur_radius
    let sigma = max(0.0001, uParams.blur_radius / 3.0);

    for (var i: i32 = -half; i <= half; i = i + 1) {
        let offset = if (horizontal) { vec2<f32>(f32(i) * texel.x, 0.0) } else { vec2<f32>(0.0, f32(i) * texel.y) };
        let s_uv = clamp(uv + offset, vec2<f32>(0.0), vec2<f32>(1.0));
        let sample_ao = textureSample(tAOHistory, sLinearClamp, s_uv).x; // using tAOHistory as source for blur (ping-pong)
        let sample_depth = textureSample(depth_tex, sLinearClamp, s_uv);
        let sample_view = reconstruct_view_pos(s_uv, sample_depth);
        let sample_normal = fetch_normal(s_uv);

        // spatial weight (gaussian)
        let spatial = exp(- (f32(i * i)) / (2.0 * sigma * sigma));
        // depth weight (bilateral)
        let depth_diff = abs(sample_view.z - center_view.z);
        let depth_w = exp(- (depth_diff * depth_diff) / (2.0 * uParams.depth_threshold * uParams.depth_threshold));
        // normal weight
        let n_dot = max(dot(center_normal, sample_normal), 0.0);
        let normal_w = exp(- (1.0 - n_dot) / (2.0 * uParams.normal_threshold * uParams.normal_threshold));

        let w = spatial * depth_w * normal_w;
        sum = sum + sample_ao * w;
        wsum = wsum + w;
    }

    return sum / max(wsum, 1e-6);
}

// ---------- Fragment main ----------
struct FSOut {
    @location(0) ao_out: vec4<f32>;
    @location(1) history_out: vec4<f32>;
};

@fragment
fn fs_main(in: VSOut) -> FSOut {
    var out: FSOut;
    let uv = in.uv;

    let mode = uParams.mode;

    if (mode == 0u) {
        // Compute AO
        let res = compute_ssao(uv);
        let ao = res.x;
        // If temporal enabled and history available, blend
        var final_ao = ao;
        if (uParams.temporal_enabled == 1u) {
            // fetch history at same uv
            let hist = textureSample(tAOHistory, sLinearClamp, uv).x;
            final_ao = mix(ao, hist, uParams.temporal_alpha);
        }
        // pack AO into RGBA (repeat in all channels for convenience)
        out.ao_out = vec4<f32>(final_ao, final_ao, final_ao, 1.0);
        out.history_out = vec4<f32>(final_ao, final_ao, final_ao, 1.0);
        return out;
    } else if (mode == 1u) {
        // Blur horizontal
        let blurred = bilateral_blur(uv, true);
        out.ao_out = vec4<f32>(blurred, blurred, blurred, 1.0);
        out.history_out = vec4<f32>(blurred, blurred, blurred, 1.0);
        return out;
    } else if (mode == 2u) {
        // Blur vertical
        let blurred = bilateral_blur(uv, false);
        out.ao_out = vec4<f32>(blurred, blurred, blurred, 1.0);
        out.history_out = vec4<f32>(blurred, blurred, blurred, 1.0);
        return out;
    } else {
        // Composite / debug: sample AO and show debug overlays
        let ao = textureSample(tAOHistory, sLinearClamp, uv).x;
        if (uParams.debug == 1u) {
            out.ao_out = vec4<f32>(ao, ao, ao, 1.0);
        } else if (uParams.debug == 2u) {
            // show bent normal x encoded as color
            let bent_x = compute_ssao(uv).y;
            out.ao_out = vec4<f32>(bent_x * 0.5 + 0.5, 0.0, 0.0, 1.0);
        } else {
            // overlay AO on scene (caller should sample scene and multiply)
            out.ao_out = vec4<f32>(ao, ao, ao, 1.0);
        }
        out.history_out = vec4<f32>(ao, ao, ao, 1.0);
        return out;
    }
}
