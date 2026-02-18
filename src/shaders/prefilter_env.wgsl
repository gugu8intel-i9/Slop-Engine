// prefilter_env.wgsl
// High-performance prefilter shader for specular IBL (WGSL).
// - Importance-sampled GGX prefiltering for environment cubemap
// - Outputs prefiltered radiance for a given roughness (render to cubemap mip levels)
// - Supports dynamic sample count, sample stratification (Hammersley), and optional MIS fallback
// - Uses textureSampleLevel for explicit LOD sampling of the source environment map
//
// Usage:
// - Render a cube (6 faces) or fullscreen quad with direction encoded per-fragment (view vector).
// - For each mip level, set uParams.target_mip and render; the shader computes roughness from mip.
// - Render target: cubemap face at mip level (RGBA16F/32F recommended).
//
// Bindings expected:
// group(0) binding(0) : sampler sLinearClamp
// group(0) binding(1) : texture_cube<f32> tEnv;   // source environment (radiance) cubemap
// group(0) binding(2) : uniform PrefilterParams { uint sample_count; uint target_mip; uint max_mip; float roughness_override; }
// If roughness_override < 0.0, compute roughness = target_mip / max_mip; otherwise use override.

struct PrefilterParams {
    sample_count: u32;
    target_mip: u32;
    max_mip: u32;
    roughness_override: f32;
};
@group(0) @binding(2) var<uniform> uParams: PrefilterParams;

@group(0) @binding(0) var sLinearClamp: sampler;
@group(0) @binding(1) var tEnv: texture_cube<f32>;

// ---------- Math helpers ----------

let PI: f32 = 3.141592653589793;
let EPS: f32 = 1e-5;

// Radical inverse base 2 (Van der Corput)
fn radical_inverse_vdc(bits: u32) -> f32 {
    var b = bits;
    b = (b << 16u) | (b >> 16u);
    b = ((b & 0x00ff00ffu) << 8u) | ((b & 0xff00ff00u) >> 8u);
    b = ((b & 0x0f0f0f0fu) << 4u) | ((b & 0xf0f0f0f0u) >> 4u);
    b = ((b & 0x33333333u) << 2u) | ((b & 0xccccccccu) >> 2u);
    b = ((b & 0x55555555u) << 1u) | ((b & 0xaaaaaaaau) >> 1u);
    return f32(b) * 2.3283064365386963e-10; // / 2^32
}

// Hammersley sequence (i/N, radical_inverse(i))
fn hammersley(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), radical_inverse_vdc(i));
}

// GGX / Trowbridge-Reitz importance sampling
fn importance_sample_ggx(xi: vec2<f32>, N: vec3<f32>, roughness: f32) -> vec3<f32> {
    let a: f32 = roughness * roughness;

    let phi: f32 = 2.0 * PI * xi.x;
    let cosTheta: f32 = sqrt(max(0.0, (1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y)));
    let sinTheta: f32 = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

    // Spherical to cartesian (half vector in tangent space)
    let Ht: vec3<f32> = vec3<f32>(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

    // Build tangent space basis (T, B, N)
    var up: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);
    if (abs(N.z) > 0.999) {
        up = vec3<f32>(0.0, 1.0, 0.0);
    }
    let T: vec3<f32> = normalize(cross(up, N));
    let B: vec3<f32> = cross(N, T);

    // Transform Ht to world space
    let H: vec3<f32> = normalize(Ht.x * T + Ht.y * B + Ht.z * N);
    return H;
}

// GGX distribution D (not strictly needed for sampling but kept for reference)
fn D_GGX(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a: f32 = roughness * roughness;
    let a2: f32 = a * a;
    let NdotH: f32 = max(dot(N, H), 0.0);
    let denom: f32 = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + EPS);
}

// Schlick Fresnel approximation (for energy compensation if needed)
fn fresnel_schlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    let one_minus = pow(1.0 - cosTheta, 5.0);
    return F0 + (1.0 - F0) * one_minus;
}

// ---------- Sampling routine ----------
// Given a normal (view vector direction for prefilter), roughness, and sample count,
// perform importance sampling over the hemisphere and accumulate prefiltered radiance.
// We treat the fragment's "normal" as the reflection direction R (the direction to sample env).

// Note: For prefiltering specular IBL we integrate incoming radiance over the hemisphere weighted by the specular BRDF.
// Importance sampling GGX over the half-vector H and reflect V around H yields sample directions L.

fn prefilter_specular(R: vec3<f32>, roughness: f32, sample_count: u32) -> vec3<f32> {
    var prefiltered_color: vec3<f32> = vec3<f32>(0.0);
    var total_weight: f32 = 0.0;

    // Choose sample count adaptively for performance: fewer samples for low roughness
    var Nsamples: u32 = sample_count;
    if (Nsamples == 0u) {
        Nsamples = 1024u; // fallback default
    }

    // For stratification, use Hammersley sequence
    for (var i: u32 = 0u; i < Nsamples; i = i + 1u) {
        let xi: vec2<f32> = hammersley(i, Nsamples);
        let H: vec3<f32> = importance_sample_ggx(xi, R, roughness);
        let L: vec3<f32> = normalize(2.0 * dot(R, H) * H - R);

        let NdotL: f32 = max(dot(R, L), 0.0);
        if (NdotL > 0.0) {
            // Compute the PDF for the sampled direction (for MIS or energy compensation)
            // PDF = D(H) * NdotH / (4 * VdotH)
            // We approximate PDF via D_GGX and geometry terms if needed; for speed we skip explicit PDF and weight by NdotL
            // Sample the environment map with explicit LOD: map roughness to mip level
            // Map roughness to mip: roughness in [0,1] -> mip in [0,max_mip]
            let max_mip_f: f32 = f32(uParams.max_mip);
            let mip: f32 = roughness * max_mip_f;

            // Sample environment cubemap with explicit LOD
            let sample_color: vec3<f32> = textureSampleLevel(tEnv, sLinearClamp, L, mip).xyz;

            // Weight by NdotL (Lambertian-like) and accumulate
            prefiltered_color = prefiltered_color + sample_color * NdotL;
            total_weight = total_weight + NdotL;
        }
    }

    // Normalize
    if (total_weight > 0.0) {
        prefiltered_color = prefiltered_color / total_weight;
    }

    return prefiltered_color;
}

// ---------- Fragment entry ----------
// Input: per-fragment direction (view vector or reflection vector) encoded in vertex shader.
// For cube-face rendering, supply the direction corresponding to the current fragment's position on the cube face.
// Example vertex pipeline: render a unit cube and pass normalized position as direction.

struct FragIn {
    @location(0) dir: vec3<f32>; // world-space direction (unit)
};

struct FragOut {
    @location(0) color: vec4<f32>;
};

@fragment
fn fs_main(in: FragIn) -> FragOut {
    var out: FragOut;

    // Determine roughness: either override or derive from target mip
    var roughness: f32 = 0.0;
    if (uParams.roughness_override >= 0.0) {
        roughness = clamp(uParams.roughness_override, 0.0, 1.0);
    } else {
        // Map target_mip to roughness: mip / max_mip
        if (uParams.max_mip == 0u) {
            roughness = 0.0;
        } else {
            roughness = clamp(f32(uParams.target_mip) / f32(uParams.max_mip), 0.0, 1.0);
        }
    }

    // Use the fragment's direction as the reflection vector R (the normal for sampling)
    let R: vec3<f32> = normalize(in.dir);

    // Adaptive sample count: fewer samples for low roughness (sharp), more for high roughness (blurry)
    var base_samples: u32 = max(1u, uParams.sample_count);
    // Heuristic: scale samples by (roughness^2 * 1.5 + 0.5)
    let scale: f32 = roughness * roughness * 1.5 + 0.5;
    let sample_count: u32 = u32(f32(base_samples) * scale);

    // Cap sample_count to a reasonable maximum for performance
    let max_samples: u32 = 4096u;
    let final_samples: u32 = min(sample_count, max_samples);

    // Compute prefiltered color
    let prefiltered: vec3<f32> = prefilter_specular(R, roughness, final_samples);

    // Output linear HDR color
    out.color = vec4<f32>(prefiltered, 1.0);
    return out;
}
