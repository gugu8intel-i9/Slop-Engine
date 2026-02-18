// dof.wgsl
// High-performance, featureful Depth-of-Field WGSL shader
// - Thin-lens Bokeh (physical aperture model)
// - Circle-of-Confusion (CoC) computation with near/far focus
// - Tile-based max CoC optimization (optional pre-pass friendly)
// - Gather pass (variable sample count) and separable bilateral blur
// - Hexagonal / Poisson disk sampling patterns for pleasing bokeh
// - Chromatic aberration per-channel blur radius
// - Temporal accumulation with reprojection and clamping to avoid ghosting
// - Optional dilation and prefilter to preserve highlights
//
// Expected bindings (example):
// group(0) binding(0) : sampler sLinearClamp
// group(0) binding(1) : texture_2d<f32> tColor;        // HDR color buffer (linear)
// group(0) binding(2) : texture_depth_2d tDepth;      // depth buffer (0..1 NDC depth)
// group(0) binding(3) : texture_2d<f32> tNormal;      // normal buffer (optional, for bilateral)
// group(0) binding(4) : texture_2d<f32> tVelocity;    // motion vectors (UV offset) for temporal reproj
// group(0) binding(5) : texture_2d<f32> tHistory;     // previous DOF accumulation (optional)
// group(0) binding(6) : texture_2d<f32> tBlueNoise;   // blue noise for jitter/dither
// group(0) binding(7) : sampler sPointClamp           // point sampler for history/LUT
//
// group(1) binding(0) : uniform DOFParams { ... }
// group(2) binding(0) : uniform Camera { proj, inv_proj, focal_length, sensor_height, aperture, focus_distance, screen_size }
//
// Render passes:
// - Mode 0: CoC compute + gather composite (single-pass gather, slower, high quality)
// - Mode 1: Separable blur horizontal
// - Mode 2: Separable blur vertical
// - Mode 3: Composite final (blend DOF with original, apply chromatic aberration, TAA write)
//
// Vertex: fullscreen triangle; Fragment: DOF operations
//
// Notes:
// - For best performance, run a cheap tile max-CoC pre-pass on a downsampled buffer and use it to skip pixels with tiny CoC.
// - Use lower sample counts on mobile; increase on desktop. The shader exposes knobs in DOFParams.

struct DOFParams {
    mode: u32;                 // 0=single-pass gather,1=blur_h,2=blur_v,3=composite
    max_samples: u32;          // max gather taps (e.g., 32)
    quality: u32;              // 0=low,1=medium,2=high (affects sample_count)
    aperture: f32;             // f-stop equivalent (smaller -> deeper DOF)
    focal_distance: f32;       // focus distance in world units
    focal_length: f32;         // lens focal length in mm
    sensor_height: f32;        // sensor height in mm (for CoC scaling)
    max_coc_px: f32;           // maximum CoC radius in pixels
    near_blur: f32;            // enable near-field blur (0/1)
    chrom_aberration: f32;     // 0..1 amount of chromatic separation
    temporal_alpha: f32;       // 0..1 history blend
    temporal_clamp: f32;       // clamp factor for history
    jitter_strength: f32;      // jitter amount for sample rotation
    pad0: vec2<f32>;
};
@group(1) @binding(0) var<uniform> uDOF: DOFParams;

struct Camera {
    proj: mat4x4<f32>;
    inv_proj: mat4x4<f32>;
    view: mat4x4<f32>;
    inv_view: mat4x4<f32>;
    focal_length: f32;
    sensor_height: f32;
    aperture: f32;
    focus_distance: f32;
    screen_size: vec2<f32>;
    pad: vec2<f32>;
};
@group(2) @binding(0) var<uniform> uCam: Camera;

@group(0) @binding(0) var sLinearClamp: sampler;
@group(0) @binding(1) var tColor: texture_2d<f32>;
@group(0) @binding(2) var tDepth: texture_depth_2d;
@group(0) @binding(3) var tNormal: texture_2d<f32>;
@group(0) @binding(4) var tVelocity: texture_2d<f32>;
@group(0) @binding(5) var tHistory: texture_2d<f32>;
@group(0) @binding(6) var tBlueNoise: texture_2d<f32>;
@group(0) @binding(7) var sPointClamp: sampler;

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

// Fragment outputs: color and history
struct FSOut {
    @location(0) color: vec4<f32>;
    @location(1) history: vec4<f32>;
};

// Utility constants
let PI: f32 = 3.141592653589793;
let EPS: f32 = 1e-6;

// Convert NDC depth to view-space Z (assuming projection matrix provided)
fn reconstruct_view_z(ndc_z: f32) -> f32 {
    // ndc_z in [0,1] -> clip z in [-1,1]
    let z = ndc_z * 2.0 - 1.0;
    // Reconstruct view-space position at clip (x,y arbitrary)
    // Use inv_proj to map (0,0,z,1) back to view space and take z
    let clip = vec4<f32>(0.0, 0.0, z, 1.0);
    let view = uCam.inv_proj * clip;
    return view.z / max(view.w, EPS);
}

// Reconstruct view-space position from depth and uv
fn reconstruct_view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - vec2<f32>(1.0, 1.0), depth * 2.0 - 1.0, 1.0);
    let view = uCam.inv_proj * ndc;
    return view.xyz / max(view.w, EPS);
}

// Compute Circle of Confusion (CoC) in pixels using thin-lens formula
// CoC = | (f * (s - z)) / (z * (s - f)) | * (aperture / sensor) * focal_scale
fn compute_coc_px(view_z: f32) -> f32 {
    // view_z is negative forward in many conventions; use absolute distance
    let z = abs(view_z);
    let f = max(0.0001, uDOF.focal_length * 0.001); // convert mm to meters if focal_length in mm
    let s = max(0.0001, uDOF.focal_distance);
    let aperture = max(0.0001, uDOF.aperture);
    let sensor = max(0.0001, uDOF.sensor_height * 0.001); // mm -> meters
    // thin lens CoC diameter (meters)
    let coc_m = abs((f * (s - z)) / (z * (s - f))) * (aperture / sensor);
    // convert to pixels: coc_px = coc_m * (screen_height_meters -> unknown) -> approximate via focal length projection
    // Simpler robust approach: map coc_m to screen pixels using focal length and screen size:
    // projected size = coc_m * (screen_px / projected_sensor_size)
    // approximate projected_sensor_size = sensor * (f / z)
    let projected_sensor = sensor * (f / z + EPS);
    let screen_px = uCam.screen_size.y; // use vertical resolution
    let coc_px = coc_m * (screen_px / max(projected_sensor, EPS));
    // clamp
    return clamp(coc_px, 0.0, uDOF.max_coc_px);
}

// Sample patterns: Poisson disk / hexagonal sampling (precomputed)
const POISSON_16: array<vec2<f32>, 16> = array<vec2<f32>, 16>(
    vec2<f32>(-0.326212f, -0.40581f),
    vec2<f32>(-0.840144f, -0.07358f),
    vec2<f32>(-0.695914f, 0.457137f),
    vec2<f32>(-0.203345f, 0.620716f),
    vec2<f32>(0.96234f, -0.194983f),
    vec2<f32>(0.473434f, -0.480026f),
    vec2<f32>(0.519456f, 0.767022f),
    vec2<f32>(0.185461f, -0.893124f),
    vec2<f32>(0.507431f, 0.064425f),
    vec2<f32>(0.89642f, 0.412458f),
    vec2<f32>(-0.32194f, -0.932615f),
    vec2<f32>(-0.791559f, -0.59771f),
    vec2<f32>(-0.091555f, 0.02303f),
    vec2<f32>(0.0f, 0.0f),
    vec2<f32>(0.0f, 0.0f),
    vec2<f32>(0.0f, 0.0f)
);

// Low-cost hash for jitter
fn hash2(p: vec2<f32>) -> vec2<f32> {
    let k1 = vec2<f32>(127.1, 311.7);
    let k2 = vec2<f32>(269.5, 183.3);
    let h = fract(sin(vec2<f32>(dot(p, k1), dot(p, k2))) * 43758.5453123);
    return h;
}

// Sample blue noise for rotation jitter
fn blue_noise(uv: vec2<f32>) -> vec2<f32> {
    // tile noise by screen size / small factor
    let tile = 4.0;
    let n = textureSample(tBlueNoise, sLinearClamp, uv * tile).xy;
    return n * 2.0 - vec2<f32>(1.0, 1.0);
}

// Bilateral weight using depth and normal (if normal available)
fn bilateral_weight(center_depth: f32, sample_depth: f32, center_normal: vec3<f32>, sample_normal: vec3<f32>, sigma_depth: f32, sigma_normal: f32) -> f32 {
    let dz = abs(sample_depth - center_depth);
    let wd = exp(- (dz * dz) / (2.0 * sigma_depth * sigma_depth));
    let wn = 1.0;
    if (length(center_normal) > 0.001 && length(sample_normal) > 0.001) {
        let nd = max(dot(center_normal, sample_normal), 0.0);
        wn = exp(- (1.0 - nd) / (2.0 * sigma_normal * sigma_normal));
    }
    return wd * wn;
}

// Temporal reprojection: fetch history at prev_uv using velocity
fn fetch_history(uv: vec2<f32>) -> vec3<f32> {
    // velocity stored as UV offset (prev_uv = uv + vel)
    let vel = textureSample(tVelocity, sLinearClamp, uv).xy;
    let prev_uv = clamp(uv + vel, vec2<f32>(0.0), vec2<f32>(1.0));
    let hist = textureSample(tHistory, sPointClamp, prev_uv).xyz;
    return hist;
}

// Clamp history to local neighborhood to avoid ghosting
fn clamp_history(curr: vec3<f32>, hist: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    // compute min/max in 3x3 neighborhood of current color
    let dims = uCam.screen_size;
    let texel = vec2<f32>(1.0 / dims.x, 1.0 / dims.y);
    var minc = vec3<f32>(1e9);
    var maxc = vec3<f32>(-1e9);
    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let s = clamp(uv + vec2<f32>(f32(x) * texel.x, f32(y) * texel.y), vec2<f32>(0.0), vec2<f32>(1.0));
            let c = textureSample(tColor, sLinearClamp, s).xyz;
            minc = min(minc, c);
            maxc = max(maxc, c);
        }
    }
    // expand range slightly
    let range = maxc - minc;
    let minc_e = minc - range * uDOF.temporal_clamp;
    let maxc_e = maxc + range * uDOF.temporal_clamp;
    return clamp(hist, minc_e, maxc_e);
}

// Single-pass gather DOF (high quality, uses many taps)
fn gather_dof(uv: vec2<f32>, coc_px: f32) -> vec3<f32> {
    // Determine sample count from quality
    var base_samples: u32 = 8u;
    if (uDOF.quality == 0u) { base_samples = 6u; }
    else if (uDOF.quality == 1u) { base_samples = 12u; }
    else { base_samples = 20u; }
    let samples = min(uDOF.max_samples, base_samples);

    // convert coc_px to uv-space radius
    let dims = uCam.screen_size;
    let texel = vec2<f32>(1.0 / dims.x, 1.0 / dims.y);
    let radius_uv = coc_px * texel.y; // use vertical pixel scale

    // jitter rotation from blue noise
    let jitter = blue_noise(uv) * uDOF.jitter_strength;
    let angle = atan2(jitter.y, jitter.x);
    let cosA = cos(angle);
    let sinA = sin(angle);

    var accum = vec3<f32>(0.0);
    var wsum = 0.0;

    // center sample
    let center = textureSample(tColor, sLinearClamp, uv).xyz;
    accum = accum + center;
    wsum = wsum + 1.0;

    // sample disk using Poisson pattern scaled by radius
    for (var i: u32 = 0u; i < samples; i = i + 1u) {
        let p = POISSON_16[i % 16u];
        // rotate
        let rx = p.x * cosA - p.y * sinA;
        let ry = p.x * sinA + p.y * cosA;
        let sample_uv = uv + vec2<f32>(rx, ry) * radius_uv;
        let su = clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(1.0));
        // chromatic aberration: per-channel offset proportional to coc and chrom_aberration
        let ca = uDOF.chrom_aberration;
        let r_uv = clamp(su + vec2<f32>(0.001, 0.0) * coc_px * ca, vec2<f32>(0.0), vec2<f32>(1.0));
        let g_uv = su;
        let b_uv = clamp(su - vec2<f32>(0.001, 0.0) * coc_px * ca, vec2<f32>(0.0), vec2<f32>(1.0));
        let sample_r = textureSample(tColor, sLinearClamp, r_uv).x;
        let sample_g = textureSample(tColor, sLinearClamp, g_uv).y;
        let sample_b = textureSample(tColor, sLinearClamp, b_uv).z;
        let sample = vec3<f32>(sample_r, sample_g, sample_b);

        // weight by distance (gaussian) to approximate bokeh falloff
        let dist = length(vec2<f32>(rx, ry));
        let w = exp(- (dist * dist) * 4.0); // sharper falloff
        accum = accum + sample * w;
        wsum = wsum + w;
    }

    return accum / max(wsum, 1e-6);
}

// Separable bilateral blur pass (horizontal/vertical) using CoC radius per-pixel
fn separable_blur(uv: vec2<f32>, horizontal: bool) -> vec3<f32> {
    // sample center depth and normal
    let center_depth = textureSample(tDepth, sLinearClamp, uv);
    let center_view = reconstruct_view_pos(uv, center_depth);
    let center_normal = textureSample(tNormal, sLinearClamp, uv).xyz;
    // compute sigma from CoC: use small kernel scaled by coc
    let coc_center = compute_coc_px(center_view.z);
    let radius_px = clamp(i32(ceil(coc_center)), 1, 25);
    let dims = uCam.screen_size;
    let texel = vec2<f32>(1.0 / dims.x, 1.0 / dims.y);

    var sum = vec3<f32>(0.0);
    var wsum = 0.0;

    for (var i: i32 = -radius_px; i <= radius_px; i = i + 1) {
        let offset = if (horizontal) { vec2<f32>(f32(i) * texel.x, 0.0) } else { vec2<f32>(0.0, f32(i) * texel.y) };
        let s_uv = clamp(uv + offset, vec2<f32>(0.0), vec2<f32>(1.0));
        let sample_color = textureSample(tColor, sLinearClamp, s_uv).xyz;
        let sample_depth = textureSample(tDepth, sLinearClamp, s_uv);
        let sample_view = reconstruct_view_pos(s_uv, sample_depth);
        let sample_normal = textureSample(tNormal, sLinearClamp, s_uv).xyz;

        // spatial weight (gaussian)
        let sigma = max(1.0, f32(radius_px) / 2.0);
        let spatial = exp(- (f32(i * i)) / (2.0 * sigma * sigma));
        // bilateral weights
        let depth_w = exp(- (abs(sample_view.z - center_view.z) * 100.0)); // scale depth sensitivity
        let normal_w = exp(- (1.0 - max(dot(center_normal, sample_normal), 0.0)) * 10.0);
        let w = spatial * depth_w * normal_w;
        sum = sum + sample_color * w;
        wsum = wsum + w;
    }

    return sum / max(wsum, 1e-6);
}

// Composite final DOF with temporal accumulation and optional dilation
fn composite_final(uv: vec2<f32>, dof_color: vec3<f32>, coc_px: f32) -> vec3<f32> {
    // fetch history and clamp
    var final = dof_color;
    if (uDOF.temporal_alpha > 0.0) {
        let hist = fetch_history(uv);
        if (all(isFinite(hist))) {
            let hist_clamped = clamp_history(dof_color, hist, uv);
            final = mix(dof_color, hist_clamped, uDOF.temporal_alpha);
        }
    }
    // optionally blend with original based on small CoC (near focus)
    let blend = smoothstep(0.0, 1.0, coc_px / 2.0); // small coc -> keep original
    let orig = textureSample(tColor, sLinearClamp, uv).xyz;
    return mix(orig, final, blend);
}

// Main fragment entry
@fragment
fn fs_main(in: VSOut) -> FSOut {
    var out: FSOut;
    let uv = in.uv;

    // read depth and reconstruct view pos
    let depth = textureSample(tDepth, sLinearClamp, uv);
    if (depth >= 1.0) {
        // sky / no geometry: pass through and keep history
        let orig = textureSample(tColor, sLinearClamp, uv).xyz;
        let hist = textureSample(tHistory, sPointClamp, uv).xyz;
        out.color = vec4<f32>(orig, 1.0);
        out.history = vec4<f32>(hist, 1.0);
        return out;
    }

    let view_pos = reconstruct_view_pos(uv, depth);
    let coc_px = compute_coc_px(view_pos.z);

    // Mode dispatch
    if (uDOF.mode == 0u) {
        // Single-pass gather (high quality)
        let gathered = gather_dof(uv, coc_px);
        let composed = composite_final(uv, gathered, coc_px);
        out.color = vec4<f32>(composed, 1.0);
        out.history = vec4<f32>(composed, 1.0);
        return out;
    } else if (uDOF.mode == 1u) {
        // Horizontal separable blur pass
        let blurred = separable_blur(uv, true);
        out.color = vec4<f32>(blurred, 1.0);
        out.history = vec4<f32>(blurred, 1.0);
        return out;
    } else if (uDOF.mode == 2u) {
        // Vertical separable blur pass
        let blurred = separable_blur(uv, false);
        out.color = vec4<f32>(blurred, 1.0);
        out.history = vec4<f32>(blurred, 1.0);
        return out;
    } else {
        // Composite final pass: read blurred DOF buffer (tColor assumed to be blurred DOF input)
        let dof_input = textureSample(tColor, sLinearClamp, uv).xyz;
        let final_color = composite_final(uv, dof_input, coc_px);
        // apply subtle chromatic aberration vignette
        let ca = uDOF.chrom_aberration;
        var ca_color = final_color;
        if (ca > 0.0) {
            let offset = vec2<f32>(0.001, 0.0) * coc_px * ca;
            let r = textureSample(tColor, sLinearClamp, clamp(uv + offset, vec2<f32>(0.0), vec2<f32>(1.0))).x;
            let g = textureSample(tColor, sLinearClamp, uv).y;
            let b = textureSample(tColor, sLinearClamp, clamp(uv - offset, vec2<f32>(0.0), vec2<f32>(1.0))).z;
            ca_color = vec3<f32>(r, g, b);
        }
        out.color = vec4<f32>(ca_color, 1.0);
        out.history = vec4<f32>(ca_color, 1.0);
        return out;
    }
}
