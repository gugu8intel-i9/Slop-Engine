// bloom_v1.wgsl
// High‑performance, featureful bloom shader (WGSL v1).
// Designed as a small set of flexible passes you can reuse in a bloom pipeline:
//  - Extract bright areas (threshold + soft knee)
//  - Separable Gaussian blur (horizontal / vertical) with configurable radius and sample count
//  - Composite (additive) bloom + tone-aware blend
//
// Typical pipeline:
//  1) Render bright parts: pass.mode = 0 -> writes bright color to downsampled target
//  2) Repeatedly blur: pass.mode = 1 (horizontal) and pass.mode = 2 (vertical) on ping/pong targets
//  3) Composite: pass.mode = 3 -> combine original HDR + bloom (tone-aware)
//
// Bindings (example):
// group(0) binding(0) : sampler sLinearClamp
// group(0) binding(1) : texture_2d<f32> tInputHDR   // source HDR (linear)
// group(0) binding(2) : texture_2d<f32> tBloomSrc   // ping/pong source for blur (same size as target)
// group(0) binding(3) : sampler sPointClamp         // for history/LUT if needed
// group(1) binding(0) : uniform BloomParams { ... }
//
// Vertex: fullscreen triangle; Fragment: does work based on mode
//
// Modes:
// 0 = extract bright (threshold + soft knee)
// 1 = blur horizontal (separable Gaussian)
// 2 = blur vertical
// 3 = composite (add bloom onto HDR)
//
// Notes:
// - Use multiple downsample levels for better quality and performance (render extract into smaller targets).
// - Use fewer samples for mobile; increase for high quality.
// - This shader uses textureSampleLevel for explicit LOD when sampling environment or prefiltered maps if needed.

struct BloomParams {
    mode: u32;               // 0=extract,1=blur_h,2=blur_v,3=composite
    threshold: f32;          // brightness threshold
    soft_knee: f32;          // soft knee (0..1)
    intensity: f32;          // bloom intensity (composite)
    radius: f32;             // blur radius in texels (used to compute weights)
    sample_count: u32;       // number of taps (odd: 1,3,5,7,9,11,...)
    downsample_factor: u32;  // 1 = same size, 2 = half, etc.
    pad0: u32;
    pad1: u32;
};
@group(1) @binding(0) var<uniform> uParams: BloomParams;

@group(0) @binding(0) var sLinearClamp: sampler;
@group(0) @binding(1) var tInputHDR: texture_2d<f32>;
@group(0) @binding(2) var tBloomSrc: texture_2d<f32>;
@group(0) @binding(3) var sPointClamp: sampler; // optional

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

// ---------- Utility math ----------

fn clamp01(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

// Soft threshold (Filmic soft knee)
// returns multiplier in [0,1] that indicates how much of the pixel contributes to bloom
fn soft_threshold(lum: f32, threshold: f32, knee: f32) -> f32 {
    // knee is soft_knee * threshold (user supplies soft_knee in 0..1)
    let k = threshold * knee;
    // if below threshold - knee -> 0
    if (lum <= threshold - k) {
        return 0.0;
    }
    // if above threshold + knee -> 1
    if (lum >= threshold + k) {
        return 1.0;
    }
    // smooth interpolation in the knee region
    // normalized t in [0,1]
    let t = (lum - (threshold - k)) / (2.0 * k + 1e-6);
    // use smoothstep-like curve (quadratic)
    return t * t * (3.0 - 2.0 * t);
}

// Compute luminance (Rec. 709)
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Gaussian weight function (unnormalized)
fn gaussian(x: f32, sigma: f32) -> f32 {
    let s2 = sigma * sigma;
    return exp(- (x * x) / (2.0 * s2));
}

// Precompute separable Gaussian weights and offsets on the CPU if possible.
// This shader computes them on the fly for flexibility.

// ---------- Fragment passes ----------

struct FSOut {
    @location(0) color: vec4<f32>;
};

@fragment
fn fs_main(in: VSOut) -> FSOut {
    var out: FSOut;
    let uv = in.uv;

    // Determine which pass to run
    let mode = uParams.mode;

    // For blur passes we need texel size of the bloom source (pass in via downsample factor or uniform)
    // For portability, compute texel size from textureDimensions (if supported). Use textureDimensions if available.
    // WGSL: textureDimensions returns vec2<i32> for 2D textures.
    let dims = textureDimensions(tBloomSrc);
    let tex_w = f32(dims.x);
    let tex_h = f32(dims.y);
    let texel = vec2<f32>(1.0 / max(tex_w, 1.0), 1.0 / max(tex_h, 1.0)); // conservative texel

    // ---------- Mode 0: Extract bright areas ----------
    if (mode == 0u) {
        // Sample HDR input
        let hdr = textureSample(tInputHDR, sLinearClamp, uv).xyz;
        let lum = luminance(hdr);

        // Soft threshold multiplier
        let knee = clamp01(uParams.soft_knee);
        let m = soft_threshold(lum, uParams.threshold, knee);

        // Multiply color by multiplier and optionally scale by (lum - threshold) to emphasize bright parts
        let bright = hdr * m;

        out.color = vec4<f32>(bright, 1.0);
        return out;
    }

    // ---------- Mode 1: Blur horizontal (separable) ----------
    if (mode == 1u) {
        // Horizontal blur: sample tBloomSrc with offsets along X
        // sample_count must be odd; if even, treat as next lower odd
        var taps = uParams.sample_count;
        if (taps == 0u) { taps = 5u; }
        if ((taps & 1u) == 0u) { taps = taps - 1u; } // make odd

        // sigma derived from radius
        let radius = max(0.0, uParams.radius);
        // map radius to sigma (approx): sigma = radius / 3
        let sigma = max(0.0001, radius / 3.0);

        // center index
        let half = i32(taps / 2u);

        var sum = vec3<f32>(0.0);
        var wsum = 0.0;

        // For performance, sample symmetric pairs and reuse weights
        for (var i: i32 = -half; i <= half; i = i + 1) {
            let offset = vec2<f32>(f32(i) * texel.x * uParams.downsample_factor, 0.0);
            let sample = textureSample(tBloomSrc, sLinearClamp, uv + offset).xyz;
            let weight = gaussian(f32(i), sigma);
            sum = sum + sample * weight;
            wsum = wsum + weight;
        }

        let color = sum / max(wsum, 1e-6);
        out.color = vec4<f32>(color, 1.0);
        return out;
    }

    // ---------- Mode 2: Blur vertical (separable) ----------
    if (mode == 2u) {
        var taps = uParams.sample_count;
        if (taps == 0u) { taps = 5u; }
        if ((taps & 1u) == 0u) { taps = taps - 1u; }

        let radius = max(0.0, uParams.radius);
        let sigma = max(0.0001, radius / 3.0);
        let half = i32(taps / 2u);

        var sum = vec3<f32>(0.0);
        var wsum = 0.0;

        for (var i: i32 = -half; i <= half; i = i + 1) {
            let offset = vec2<f32>(0.0, f32(i) * texel.y * uParams.downsample_factor);
            let sample = textureSample(tBloomSrc, sLinearClamp, uv + offset).xyz;
            let weight = gaussian(f32(i), sigma);
            sum = sum + sample * weight;
            wsum = wsum + weight;
        }

        let color = sum / max(wsum, 1e-6);
        out.color = vec4<f32>(color, 1.0);
        return out;
    }

    // ---------- Mode 3: Composite bloom onto HDR ----------
    if (mode == 3u) {
        // Read original HDR and bloom (bloom source contains blurred bloom)
        let hdr = textureSample(tInputHDR, sLinearClamp, uv).xyz;
        let bloom = textureSample(tBloomSrc, sLinearClamp, uv).xyz;

        // Tone-aware composite: scale bloom by intensity and modulate by HDR luminance to avoid overblooming dark areas
        let hdr_lum = luminance(hdr);
        // Use smoothstep to reduce bloom on very dark pixels
        let lum_factor = smoothstep(0.0, 0.5, hdr_lum); // tweak as needed
        let bloom_contrib = bloom * uParams.intensity * lum_factor;

        // Additive composite (HDR)
        var result = hdr + bloom_contrib;

        // Optional simple clamp to avoid NaNs
        result = max(result, vec3<f32>(0.0));

        out.color = vec4<f32>(result, 1.0);
        return out;
    }

    // Default: pass-through
    let pass = textureSample(tInputHDR, sLinearClamp, uv).xyz;
    out.color = vec4<f32>(pass, 1.0);
    return out;
}
