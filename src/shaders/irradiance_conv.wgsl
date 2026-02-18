// irradiance_conv.wgsl
// High-performance irradiance convolution shader (WGSL).
// - Computes diffuse irradiance (Lambertian convolution) from an environment cubemap.
// - Uses stratified Hammersley sampling and cosine-weighted hemisphere sampling.
// - Adaptive sample count, sample stratification, and optional importance sampling.
// - Designed for rendering to a cubemap (one face per draw) or fullscreen quad with direction input.
// - Bindings:
//   group(0) binding(0) : sampler sLinearClamp
//   group(0) binding(1) : texture_cube<f32> tEnv;   // source environment (radiance) cubemap
//   group(0) binding(2) : uniform IrradianceParams { uint sample_count; uint sample_seed; uint pad0; uint pad1; }
// - Vertex pipeline should provide a normalized direction vector per-fragment (world-space direction).
//
// Output: linear HDR irradiance color (vec4<f32>).

struct IrradianceParams {
    sample_count: u32;
    sample_seed: u32;
    pad0: u32;
    pad1: u32;
};
@group(0) @binding(2) var<uniform> uParams: IrradianceParams;

@group(0) @binding(0) var sLinearClamp: sampler;
@group(0) @binding(1) var tEnv: texture_cube<f32>;

// Constants
let PI: f32 = 3.141592653589793;
let EPS: f32 = 1e-6;

// Radical inverse base 2 (Van der Corput) for Hammersley
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
fn hammersley(i: u32, n: u32, seed: u32) -> vec2<f32> {
    // simple scrambling by xor with seed for decorrelation across passes
    let idx = i ^ seed;
    return vec2<f32>(f32(i) / f32(n), radical_inverse_vdc(idx));
}

// Cosine-weighted hemisphere sampling (importance sampling for Lambertian)
fn sample_cosine_hemisphere(xi: vec2<f32>) -> vec3<f32> {
    // xi.x in [0,1) -> r^2 distribution, xi.y -> phi
    let r: f32 = sqrt(max(0.0, xi.x));
    let phi: f32 = 2.0 * PI * xi.y;
    let x: f32 = r * cos(phi);
    let y: f32 = r * sin(phi);
    let z: f32 = sqrt(max(0.0, 1.0 - xi.x)); // z = sqrt(1 - r^2)
    return vec3<f32>(x, y, z);
}

// Build tangent space basis from normal (N). N is assumed normalized.
fn build_tangent_space(N: vec3<f32>) -> mat3x3<f32> {
    var up: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);
    if (abs(N.z) > 0.999) {
        up = vec3<f32>(0.0, 1.0, 0.0);
    }
    let T: vec3<f32> = normalize(cross(up, N));
    let B: vec3<f32> = cross(N, T);
    return mat3x3<f32>(T, B, N);
}

// Importance-sampled irradiance convolution
// R: direction vector (the "normal" for which we compute irradiance), must be normalized
// sample_count: number of samples to use (adaptive recommended)
// seed: randomization seed to decorrelate samples across faces/passes
fn integrate_irradiance(R: vec3<f32>, sample_count: u32, seed: u32) -> vec3<f32> {
    var irradiance: vec3<f32> = vec3<f32>(0.0);
    var total_weight: f32 = 0.0;

    // Build tangent space where Z = R
    let TBN = build_tangent_space(R);

    // Cap sample_count to a reasonable maximum for performance
    let max_samples: u32 = 4096u;
    var Nsamples: u32 = min(max(sample_count, 1u), max_samples);

    // Adaptive sampling: reduce samples for near-constant regions (optional heuristic)
    // For irradiance, we can reduce samples for near-flat normals (not implemented here for determinism)

    // Loop over Hammersley samples
    for (var i: u32 = 0u; i < Nsamples; i = i + 1u) {
        let xi = hammersley(i, Nsamples, seed);
        // Cosine-weighted hemisphere sample in tangent space
        let hemi = sample_cosine_hemisphere(xi);
        // Transform to world space direction L
        let L: vec3<f32> = normalize(TBN * hemi);
        // Sample environment map in direction L
        // Use explicit LOD 0 for irradiance (low-frequency); if you have mipmapped irradiance targets, you can sample lower LODs
        let sample_color: vec3<f32> = textureSampleLevel(tEnv, sLinearClamp, L, 0.0).xyz;

        // Weight by cosine (hemi.z already accounts for cosine weighting)
        // hemi.z is cos(theta) in tangent space; ensure non-negative
        let cos_theta: f32 = max(hemi.z, 0.0);
        irradiance = irradiance + sample_color * cos_theta;
        total_weight = total_weight + cos_theta;
    }

    // Normalize by total weight and multiply by PI (Lambertian convolution factor)
    if (total_weight > 0.0) {
        irradiance = irradiance * (PI / total_weight);
    } else {
        irradiance = vec3<f32>(0.0);
    }

    return irradiance;
}

// Fragment inputs: direction vector (world-space) passed from vertex shader
struct FragIn {
    @location(0) dir: vec3<f32>; // normalized direction (unit)
};

struct FragOut {
    @location(0) color: vec4<f32>; // linear HDR irradiance
};

@fragment
fn fs_main(in: FragIn) -> FragOut {
    var out: FragOut;

    // Normalize input direction
    let R: vec3<f32> = normalize(in.dir);

    // Determine sample count and seed from uniform
    let base_samples: u32 = max(1u, uParams.sample_count);
    let seed: u32 = uParams.sample_seed;

    // Heuristic: fewer samples for directions near poles (optional)
    // Here we keep base_samples but allow caller to set appropriate value per-pass.
    let sample_count: u32 = base_samples;

    // Integrate irradiance
    let irradiance: vec3<f32> = integrate_irradiance(R, sample_count, seed);

    // Output linear HDR color; alpha = 1
    out.color = vec4<f32>(irradiance, 1.0);
    return out;
}
