// taa.wgsl
// High-performance, featureful Temporal Anti-Aliasing (TAA) shader (WGSL).
// Features:
//  - Reprojection using motion vectors (screen-space UV velocity)
//  - History fetch, validity checks, and camera-cut / history-reset handling
//  - Adaptive blending (alpha) based on motion magnitude and history confidence
//  - Neighborhood clamping (min/max) to remove ghosting and light bleeding
//  - Reactive sharpening (unsharp mask) to recover detail after blending
//  - Optional stationary accumulation boost for static pixels
//  - Outputs: final color and updated history (ping-pong targets expected)
//
// Bindings expected:
// group(0) binding(0) : sampler sLinearClamp
// group(0) binding(1) : texture_2d<f32> tCurrHDR;      // current frame HDR color (linear)
// group(0) binding(2) : texture_2d<f32> tHistory;      // previous-frame history (linear)
// group(0) binding(3) : texture_2d<f32> tVelocity;     // motion vectors in UV space (RG: dx,dy) in texel units or normalized UV units
// group(0) binding(4) : texture_2d<f32> tDepth;        // current depth (optional, for occlusion checks)
// group(0) binding(5) : texture_2d<f32> tPrevDepth;    // previous depth (optional, for occlusion checks)
// group(0) binding(6) : sampler sPointClamp;           // point sampler for history (if needed)
// group(1) binding(0) : uniform TAAParams { ... }
//
// Render targets:
//  - location(0): final_color (to be displayed / further post-processing)
//  - location(1): new_history (to be used next frame as tHistory)

struct TAAParams {
    // blending
    history_alpha: f32;            // base history blend factor (0..1) (higher = more stable)
    min_alpha: f32;                // minimum alpha when motion is high
    max_alpha: f32;                // maximum alpha when motion is low
    // motion thresholds
    motion_scale: f32;             // scale to convert velocity to normalized [0..1] motion metric
    motion_threshold: f32;         // motion above this is considered fast
    // clamping
    clamp_radius: f32;             // neighborhood radius in texels for min/max clamp
    clamp_scale: f32;              // how aggressively to clamp (0..1)
    // sharpening
    sharpen_strength: f32;         // unsharp mask strength (0..1)
    sharpen_radius: f32;           // radius in texels for blur used in unsharp mask
    // history validity
    history_valid: u32;            // 0 = reset history (camera cut), 1 = use history
    // misc
    stationary_boost: f32;         // boost factor for stationary pixels (>=1.0)
    pad0: vec3<f32>;
};
@group(1) @binding(0) var<uniform> uParams: TAAParams;

@group(0) @binding(0) var sLinearClamp: sampler;
@group(0) @binding(6) var sPointClamp: sampler;
@group(0) @binding(1) var tCurrHDR: texture_2d<f32>;
@group(0) @binding(2) var tHistory: texture_2d<f32>;
@group(0) @binding(3) var tVelocity: texture_2d<f32>;
@group(0) @binding(4) var tDepth: texture_2d<f32>;
@group(0) @binding(5) var tPrevDepth: texture_2d<f32>;

// Vertex: fullscreen triangle
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

// Fragment outputs: final color and updated history
struct FSOut {
    @location(0) color: vec4<f32>;
    @location(1) history: vec4<f32>;
};

// Utility helpers
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn safe_pow(x: f32, e: f32) -> f32 {
    return pow(max(x, 0.0), e);
}

// Sample neighborhood min/max for clamping (box kernel)
struct NeighborhoodMinMax {
    minc: vec3<f32>,
    maxc: vec3<f32>,
};

fn neighborhood_minmax(uv: vec2<f32>, texel: vec2<f32>, radius: i32) -> NeighborhoodMinMax {
    var minc = vec3<f32>(1e9, 1e9, 1e9);
    var maxc = vec3<f32>(-1e9, -1e9, -1e9);
    for (var y: i32 = -radius; y <= radius; y = y + 1) {
        for (var x: i32 = -radius; x <= radius; x = x + 1) {
            let sample_uv = uv + vec2<f32>(f32(x) * texel.x, f32(y) * texel.y);
            let s = textureSample(tCurrHDR, sLinearClamp, sample_uv).xyz;
            minc = min(minc, s);
            maxc = max(maxc, s);
        }
    }
    return NeighborhoodMinMax(minc, maxc);
}

// Simple box blur for unsharp mask (small radius)
fn box_blur(uv: vec2<f32>, texel: vec2<f32>, radius: i32) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    var count: f32 = 0.0;
    for (var y: i32 = -radius; y <= radius; y = y + 1) {
        for (var x: i32 = -radius; x <= radius; x = x + 1) {
            let sample_uv = uv + vec2<f32>(f32(x) * texel.x, f32(y) * texel.y);
            sum = sum + textureSample(tCurrHDR, sLinearClamp, sample_uv).xyz;
            count = count + 1.0;
        }
    }
    return sum / max(count, 1.0);
}

// Main fragment: TAA
@fragment
fn fs_main(in: VSOut) -> FSOut {
    var out: FSOut;

    let uv = in.uv;

    // Query texture size (if supported). For portability, caller should ensure textures are same size.
    let dims = textureDimensions(tCurrHDR);
    let tex_w = f32(dims.x);
    let tex_h = f32(dims.y);
    let texel = vec2<f32>(1.0 / max(tex_w, 1.0), 1.0 / max(tex_h, 1.0));

    // Read current color
    let curr = textureSample(tCurrHDR, sLinearClamp, uv).xyz;

    // If history is invalid (camera cut), bypass TAA and write current as history
    if (uParams.history_valid == 0u) {
        out.color = vec4<f32>(curr, 1.0);
        out.history = vec4<f32>(curr, 1.0);
        return out;
    }

    // Read motion vector (assume stored as UV offset from current to previous: prev_uv = uv + velocity)
    // Convention: tVelocity stores motion in normalized UV units (i.e., [-1,1] range relative to screen)
    // If your pipeline stores motion in texel units, multiply by texel to convert.
    let vel = textureSample(tVelocity, sLinearClamp, uv).xy;
    // If velocity is in texel units, convert: vel_uv = vel * texel
    // Here we assume vel is in UV units already.
    let prev_uv = uv + vel;

    // Clamp prev_uv to [0,1] to avoid sampling outside
    let prev_uv_clamped = clamp(prev_uv, vec2<f32>(0.0), vec2<f32>(1.0));

    // Fetch history color from previous frame at prev_uv
    // Use point sampler for history to avoid filtering ghosting; but linear can be used too.
    let hist = textureSample(tHistory, sPointClamp, prev_uv_clamped).xyz;

    // Optional occlusion check using depth (if provided)
    var occluded: bool = false;
    // If both depth textures are provided, compare depths to detect disocclusion
    // depth values assumed in 0..1 (linear or non-linear depending on pipeline)
    if (uParams.history_valid == 1u) {
        let curr_depth = textureSample(tDepth, sLinearClamp, uv).x;
        let prev_depth = textureSample(tPrevDepth, sLinearClamp, prev_uv_clamped).x;
        // If depth difference is large (e.g., > threshold), mark occluded
        if (abs(curr_depth - prev_depth) > 0.01) {
            occluded = true;
        }
    }

    // Compute motion magnitude metric
    let motion_mag = length(vel) * uParams.motion_scale;

    // Adaptive alpha: reduce history weight for high motion or occlusion
    var alpha = uParams.history_alpha;
    // Map motion to [min_alpha..max_alpha] (higher motion -> lower alpha)
    let motion_t = clamp(motion_mag / max(uParams.motion_threshold, 1e-6), 0.0, 1.0);
    let motion_alpha = mix(uParams.max_alpha, uParams.min_alpha, motion_t);
    alpha = min(alpha, motion_alpha);
    if (occluded) {
        alpha = uParams.min_alpha; // force low history weight on occlusion
    }

    // History validity check: if history color is NaN or extreme, discard
    let hist_finite = all(hist == hist) && all(abs(hist) < vec3<f32>(65504.0));
    if (!hist_finite) {
        alpha = 0.0;
    }

    // Neighborhood clamping to avoid ghosting:
    // Compute min/max in a small neighborhood around current pixel
    let clamp_radius = i32(max(1.0, floor(uParams.clamp_radius)));
    let mm = neighborhood_minmax(uv, texel, clamp_radius);
    let minc = mm.minc;
    let maxc = mm.maxc;

    // Compute clamped history: clamp hist to [min - eps, max + eps]
    let eps = 1e-4;
    let hist_clamped = clamp(hist, minc - vec3<f32>(eps), maxc + vec3<f32>(eps));

    // Stationary boost: if motion is near zero, slightly favor history accumulation
    var stationary_boost = 1.0;
    if (motion_mag < 0.001) {
        stationary_boost = uParams.stationary_boost;
    }

    // Final blended color (linear)
    // blended = lerp(curr, hist_clamped, alpha * stationary_boost)
    let blend_factor = clamp(alpha * stationary_boost, 0.0, 1.0);
    var blended = mix(curr, hist_clamped, blend_factor);

    // Additional clamp: ensure blended within local min/max scaled by clamp_scale
    let clamp_scale = clamp(uParams.clamp_scale, 0.0, 1.0);
    let min_clamp = mix(curr, minc, clamp_scale);
    let max_clamp = mix(curr, maxc, clamp_scale);
    blended = clamp(blended, min_clamp, max_clamp);

    // Reactive sharpening (unsharp mask) to recover detail after temporal blur
    var final_color = blended;
    if (uParams.sharpen_strength > 0.0) {
        let blur_radius = i32(max(1.0, floor(uParams.sharpen_radius)));
        let blurred = box_blur(uv, texel, blur_radius);
        let mask = blended - blurred;
        final_color = blended + mask * uParams.sharpen_strength;
        // clamp to avoid overshoot
        final_color = clamp(final_color, min_clamp, max_clamp);
    }

    // Output final color and write new history (we store linear color)
    out.color = vec4<f32>(final_color, 1.0);
    // For history we may want to store additional metadata (e.g., alpha or velocity) in alpha channel.
    // Here we store color in RGB and blend_factor in A for debugging/analysis.
    out.history = vec4<f32>(final_color, blend_factor);

    return out;
}
