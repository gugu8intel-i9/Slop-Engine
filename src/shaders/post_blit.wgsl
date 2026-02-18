// post_blit.wgsl
// High-performance, feature-rich post-processing / blit shader (WGSL).
// Features:
//  - Tone mapping (ACES approximation) with exposure and white point
//  - Filmic tonemapping + gamma correction
//  - Bloom composite (additive) with configurable intensity and threshold
//  - Bloom soft knee and bloom tint
//  - Color grading via 3D LUT (sampled from 2D flattened LUT)
//  - FXAA anti-aliasing (fast approximate)
//  - Temporal Anti-Aliasing (TAA) blend with history buffer (simple exponential)
//  - Chromatic aberration, vignette, grain, and saturation controls
//  - Optional dithering (blue noise texture) and sRGB output
//
// Bindings expected:
// group(0) binding(0) : sampler sLinearClamp
// group(0) binding(1) : texture_2d<f32> tHDR;           // HDR color buffer (linear)
// group(0) binding(2) : texture_2d<f32> tBloom;         // Bloom prefiltered (same size or smaller)
// group(0) binding(3) : texture_2d<f32> tHistory;       // Previous frame color (for TAA) (optional)
// group(0) binding(4) : texture_2d<f32> tBlueNoise;     // Blue noise for dithering/grain (optional)
// group(0) binding(5) : texture_2d<f32> tLUT;           // 2D flattened 3D LUT (size: lutSize * lutSize tiles)
// group(0) binding(6) : sampler sPointClamp;            // point sampler for LUT and history
// group(1) binding(0) : uniform PostParams { ... }

struct PostParams {
    exposure: f32;
    white_point: f32;
    bloom_intensity: f32;
    bloom_threshold: f32;
    bloom_soft_knee: f32;
    bloom_tint: vec3<f32>;
    lut_size: u32;              // width/height of 3D LUT cube (e.g., 32)
    lut_tiles: u32;             // number of tiles per row in flattened LUT (e.g., lut_size)
    lut_strength: f32;          // 0..1 blend between original and graded
    fxaa_enabled: u32;          // 0/1
    taa_enabled: u32;           // 0/1
    taa_alpha: f32;             // 0..1 blend factor for TAA (higher = more history)
    chrom_aberration: f32;      // 0..1 offset amount
    vignette: f32;              // 0..1
    saturation: f32;            // 0..2
    contrast: f32;              // 0..2
    grain_amount: f32;          // 0..1
    dither_amount: f32;         // 0..1
    output_srgb: u32;           // 0 = linear, 1 = sRGB
};
@group(1) @binding(0) var<uniform> uParams: PostParams;

@group(0) @binding(0) var sLinearClamp: sampler;
@group(0) @binding(6) var sPointClamp: sampler;
@group(0) @binding(1) var tHDR: texture_2d<f32>;
@group(0) @binding(2) var tBloom: texture_2d<f32>;
@group(0) @binding(3) var tHistory: texture_2d<f32>;
@group(0) @binding(4) var tBlueNoise: texture_2d<f32>;
@group(0) @binding(5) var tLUT: texture_2d<f32>;

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

// ---------- Utility functions ----------

fn clamp01(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    // Approximate gamma 2.2
    return pow(c, vec3<f32>(1.0 / 2.2));
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return pow(c, vec3<f32>(2.2));
}

// ACES approximation (RRT + ODT fit) - fast filmic tone mapping
fn tone_map_aces(color: vec3<f32>) -> vec3<f32> {
    // ACES approximation constants
    let a: f32 = 2.51;
    let b: f32 = 0.03;
    let c: f32 = 2.43;
    let d: f32 = 0.59;
    let e: f32 = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Filmic exposure + white point
fn apply_exposure_and_white(color: vec3<f32>, exposure: f32, white_point: f32) -> vec3<f32> {
    let mapped = color * exposure;
    // simple white point scaling
    let wp = vec3<f32>(white_point);
    return mapped / (mapped + wp);
}

// Soft knee bloom curve
fn bloom_curve(x: f32, threshold: f32, soft_knee: f32) -> f32 {
    // from Filament / UE: smoothstep knee
    let knee = threshold * soft_knee;
    let soft = clamp((x - threshold + knee) / (knee + 1e-5), 0.0, 1.0);
    return max(x - threshold, 0.0) * (1.0 - soft) + soft * (x - threshold) * 0.5;
}

// 3D LUT sampling from flattened 2D LUT
fn sample_lut(color: vec3<f32>) -> vec3<f32> {
    // color components in [0,1]
    let lutSize = uParams.lut_size;
    let tiles = uParams.lut_tiles;
    if (lutSize == 0u) {
        return color;
    }
    // Convert to indices
    let scaled = color * f32(lutSize - 1u);
    let r = clamp(scaled.r, 0.0, f32(lutSize - 1u));
    let g = clamp(scaled.g, 0.0, f32(lutSize - 1u));
    let b = clamp(scaled.b, 0.0, f32(lutSize - 1u));

    // Compute integer and fractional parts
    let r_i = u32(r);
    let g_i = u32(g);
    let b_i = u32(b);
    let r_f = r - f32(r_i);
    let g_f = g - f32(g_i);
    let b_f = b - f32(b_i);

    // Flattened layout: each slice (b) is a tile in the 2D texture arranged row-major
    // tile_x = b % tiles, tile_y = b / tiles
    let tile_x = b_i % tiles;
    let tile_y = b_i / tiles;

    // UV within tile
    let invTile = 1.0 / f32(tiles);
    let texelSize = 1.0 / (f32(lutSize) * f32(tiles)); // assuming texture width = lutSize * tiles
    // base UV for texel (r_i, g_i)
    let u = (f32(r_i) + 0.5) * texelSize + f32(tile_x) * invTile;
    let v = (f32(g_i) + 0.5) * texelSize + f32(tile_y) * invTile;

    // Sample nearest texel (point sampling) and optionally trilinear via manual interpolation
    // We'll do trilinear-like interpolation across r and g and linear across b by sampling adjacent tile when b_f > 0
    let c000 = textureSample(tLUT, sPointClamp, vec2<f32>(u, v)).xyz;

    // r interpolation: sample r+1 within same tile if available
    let u_r1 = (f32(min(u32(r_i + 1u), lutSize - 1u)) + 0.5) * texelSize + f32(tile_x) * invTile;
    let c100 = textureSample(tLUT, sPointClamp, vec2<f32>(u_r1, v)).xyz;

    // g interpolation: sample g+1 row
    let v_g1 = (f32(min(u32(g_i + 1u), lutSize - 1u)) + 0.5) * texelSize + f32(tile_y) * invTile;
    let c010 = textureSample(tLUT, sPointClamp, vec2<f32>(u, v_g1)).xyz;
    let c110 = textureSample(tLUT, sPointClamp, vec2<f32>(u_r1, v_g1)).xyz;

    // Interpolate r/g plane
    let c00 = mix(c000, c100, r_f);
    let c01 = mix(c010, c110, r_f);
    let c0 = mix(c00, c01, g_f);

    // If b_f > 0, sample next tile (b+1) and interpolate between them
    var finalColor = c0;
    if (b_f > 0.0 && b_i + 1u < lutSize) {
        let tile_x2 = (b_i + 1u) % tiles;
        let tile_y2 = (b_i + 1u) / tiles;
        let u2 = (f32(r_i) + 0.5) * texelSize + f32(tile_x2) * invTile;
        let v2 = (f32(g_i) + 0.5) * texelSize + f32(tile_y2) * invTile;
        let c000b = textureSample(tLUT, sPointClamp, vec2<f32>(u2, v2)).xyz;
        let c100b = textureSample(tLUT, sPointClamp, vec2<f32>(u_r1, v2)).xyz;
        let c010b = textureSample(tLUT, sPointClamp, vec2<f32>(u2, v_g1)).xyz;
        let c110b = textureSample(tLUT, sPointClamp, vec2<f32>(u_r1, v_g1)).xyz;
        let cb0 = mix(mix(c000b, c100b, r_f), mix(c010b, c110b, r_f), g_f);
        finalColor = mix(c0, cb0, b_f);
    }

    return finalColor;
}

// FXAA implementation (approximate, single-pass)
// Reference: Jimenez / NVIDIA FXAA simplified
fn fxaa(uv: vec2<f32>, texSize: vec2<f32>) -> vec3<f32> {
    // Sample luminance at center and neighbors
    let rgbM = textureSample(tHDR, sLinearClamp, uv).xyz;
    let luma = dot(rgbM, vec3<f32>(0.299, 0.587, 0.114));

    // Offsets
    let rcpFrame = 1.0 / texSize;
    let uvN = uv + vec2<f32>(0.0, -1.0) * rcpFrame;
    let uvS = uv + vec2<f32>(0.0, 1.0) * rcpFrame;
    let uvW = uv + vec2<f32>(-1.0, 0.0) * rcpFrame;
    let uvE = uv + vec2<f32>(1.0, 0.0) * rcpFrame;

    let lumaN = dot(textureSample(tHDR, sLinearClamp, uvN).xyz, vec3<f32>(0.299, 0.587, 0.114));
    let lumaS = dot(textureSample(tHDR, sLinearClamp, uvS).xyz, vec3<f32>(0.299, 0.587, 0.114));
    let lumaW = dot(textureSample(tHDR, sLinearClamp, uvW).xyz, vec3<f32>(0.299, 0.587, 0.114));
    let lumaE = dot(textureSample(tHDR, sLinearClamp, uvE).xyz, vec3<f32>(0.299, 0.587, 0.114));

    let lumaMin = min(min(luma, lumaN), min(lumaS, min(lumaW, lumaE)));
    let lumaMax = max(max(luma, lumaN), max(lumaS, max(lumaW, lumaE)));

    // Early exit if low contrast
    if (lumaMax - lumaMin < 0.0312) {
        return rgbM;
    }

    // Compute edge direction
    let dir = vec2<f32>(
        -((lumaN + lumaS) - 2.0 * luma),
        ((lumaW + lumaE) - 2.0 * luma)
    );

    // Normalize and scale
    var dirN = dir * vec2<f32>(rcpFrame.x, rcpFrame.y);
    let dirReduce = max((lumaN + lumaS + lumaW + lumaE) * 0.25 * 0.5, 1e-6);
    let rcpDirMin = 1.0 / (min(abs(dirN.x), abs(dirN.y)) + dirReduce);
    dirN = clamp(dirN * rcpDirMin, vec2<f32>(-8.0), vec2<f32>(8.0));

    // Sample along edge
    let rgbA = 0.5 * (
        textureSample(tHDR, sLinearClamp, uv + dirN * (1.0 / 3.0 - 0.5)).xyz +
        textureSample(tHDR, sLinearClamp, uv + dirN * (2.0 / 3.0 - 0.5)).xyz
    );
    let rgbB = rgbA * 0.5 + 0.25 * (
        textureSample(tHDR, sLinearClamp, uv + dirN * -0.5).xyz +
        textureSample(tHDR, sLinearClamp, uv + dirN * 0.5).xyz
    );

    // Choose final based on luminance
    let lumaB = dot(rgbB, vec3<f32>(0.299, 0.587, 0.114));
    if (lumaB < lumaMin || lumaB > lumaMax) {
        return rgbA;
    } else {
        return rgbB;
    }
}

// Simple TAA blend: exponential moving average between current and history
fn taa_blend(curr: vec3<f32>, history: vec3<f32>, alpha: f32) -> vec3<f32> {
    return mix(curr, history, alpha);
}

// Chromatic aberration: offset R/G/B by small amounts radially
fn chromatic_aberration(uv: vec2<f32>, center: vec2<f32>, amount: f32) -> vec3<f32> {
    let dir = uv - center;
    let dist = length(dir);
    let norm = dir / (dist + 1e-6);
    // offsets scaled by dist and amount
    let r_uv = uv + norm * (amount * dist * 0.6);
    let g_uv = uv + norm * (amount * dist * 0.3);
    let b_uv = uv + norm * (amount * dist * 0.0);
    let r = textureSample(tHDR, sLinearClamp, r_uv).xyz;
    let g = textureSample(tHDR, sLinearClamp, g_uv).xyz;
    let b = textureSample(tHDR, sLinearClamp, b_uv).xyz;
    return vec3<f32>(r.r, g.g, b.b); // combine channels (approx)
}

// Grain using blue noise texture (if available)
fn apply_grain(uv: vec2<f32>, color: vec3<f32>, amount: f32) -> vec3<f32> {
    if (amount <= 0.0) { return color; }
    // sample blue noise at uv * noiseScale
    let noise = textureSample(tBlueNoise, sLinearClamp, uv * vec2<f32>(512.0, 512.0)).xyz; // assumes blue noise tiled
    let n = (noise.r - 0.5) * amount;
    return color + n;
}

// Vignette
fn apply_vignette(uv: vec2<f32>, color: vec3<f32>, strength: f32) -> vec3<f32> {
    if (strength <= 0.0) { return color; }
    let center = vec2<f32>(0.5, 0.5);
    let d = distance(uv, center);
    let vig = smoothstep(0.8, 0.5, d); // soft vignette
    return mix(color, color * vig, strength);
}

// Contrast and saturation
fn adjust_contrast_saturation(color: vec3<f32>, contrast: f32, saturation: f32) -> vec3<f32> {
    // contrast: 0..2 (1 = neutral)
    let avg = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let contrasted = mix(vec3<f32>(avg), color, contrast);
    // saturation: 0..2 (1 = neutral)
    let lum = vec3<f32>(avg);
    return mix(lum, contrasted, saturation);
}

// Dithering using blue noise
fn apply_dither(uv: vec2<f32>, color: vec3<f32>, amount: f32) -> vec3<f32> {
    if (amount <= 0.0) { return color; }
    let noise = textureSample(tBlueNoise, sLinearClamp, uv * vec2<f32>(512.0, 512.0)).xyz;
    let d = (noise.r - 0.5) * amount / 255.0;
    return color + vec3<f32>(d);
}

// ---------- Fragment main ----------

struct FSOut {
    @location(0) color: vec4<f32>;
};

@fragment
fn fs_main(in: VSOut) -> FSOut {
    var out: FSOut;
    let uv = in.uv;

    // Texture size query (approx) - not all backends support textureDimensions in fragment; pass as uniform if needed.
    // For portability, assume caller provides appropriate texel size via external uniform; here we approximate with 1/1024.
    let texSize = vec2<f32>(1024.0, 1024.0);
    let texel_uv = vec2<f32>(1.0 / texSize.x, 1.0 / texSize.y);

    // Read HDR color
    var hdr = textureSample(tHDR, sLinearClamp, uv).xyz;

    // Optionally apply FXAA before other effects (works on HDR buffer)
    if (uParams.fxaa_enabled == 1u) {
        hdr = fxaa(uv, texSize);
    }

    // Bloom: sample bloom texture and apply soft knee threshold + tint
    var bloom = textureSample(tBloom, sLinearClamp, uv).xyz;
    // Apply threshold curve
    let bright = max(dot(hdr, vec3<f32>(0.2126, 0.7152, 0.0722)) - uParams.bloom_threshold, 0.0);
    let knee = uParams.bloom_soft_knee;
    let bloomFactor = bloom_curve(bright, uParams.bloom_threshold, knee);
    bloom = bloom * uParams.bloom_intensity * bloomFactor * uParams.bloom_tint;

    // Composite bloom additively (pre-tone)
    var color_pre_tone = hdr + bloom;

    // Tone mapping
    // Apply exposure
    color_pre_tone = color_pre_tone * uParams.exposure;
    // ACES-like tonemap
    var tone = tone_map_aces(color_pre_tone);
    // White point scaling (optional)
    tone = tone / (tone + vec3<f32>(uParams.white_point));

    // Color grading via LUT
    let graded = sample_lut(clamp(tone, vec3<f32>(0.0), vec3<f32>(1.0)));
    let color_graded = mix(tone, graded, uParams.lut_strength);

    // Contrast & saturation
    var color_cs = adjust_contrast_saturation(color_graded, uParams.contrast, uParams.saturation);

    // Chromatic aberration (applied before vignette/grain)
    if (uParams.chrom_aberration > 0.0) {
        color_cs = chromatic_aberration(uv, vec2<f32>(0.5, 0.5), uParams.chrom_aberration);
    }

    // TAA: blend with history if enabled and history texture provided
    var final_color = color_cs;
    if (uParams.taa_enabled == 1u) {
        // Sample history at same UV (caller should jitter UVs for proper TAA)
        let history_color = textureSample(tHistory, sPointClamp, uv).xyz;
        final_color = taa_blend(final_color, history_color, uParams.taa_alpha);
    }

    // Grain
    final_color = apply_grain(uv, final_color, uParams.grain_amount);

    // Vignette
    final_color = apply_vignette(uv, final_color, uParams.vignette);

    // Dither
    final_color = apply_dither(uv, final_color, uParams.dither_amount);

    // Output conversion
    if (uParams.output_srgb == 1u) {
        final_color = linear_to_srgb(final_color);
    }

    out.color = vec4<f32>(final_color, 1.0);
    return out;
}
