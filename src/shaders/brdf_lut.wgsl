// brdf_lut.wgsl
// High-performance, featureful BRDF 2D LUT generator (WGSL).
// Produces a 2D lookup texture where:
//  - X axis = NdotV (0..1)
//  - Y axis = roughness (0..1)
// Each texel stores vec2 (scale, bias) used by the split-sum approximation for specular IBL:
//    prefilteredColor * (F0 * scale + bias)
// Implementation notes:
//  - Importance-sampled GGX (Trowbridge-Reitz) via Hammersley sequence
//  - Geometry Smith (Schlick-GGX) used to compute visibility term
//  - Accumulates two terms A and B as in common PBR implementations:
//      A += (1 - Fc) * G_Vis
//      B += Fc * G_Vis
//    where Fc = (1 - VdotH)^5
//  - Final LUT value = vec2(A / samples, B / samples)
//  - Vertex shader renders a full-screen triangle and passes UV to fragment shader
//
// Bindings:
//  group(0) binding(0) : uniform Params { uint sample_count; uint pad0; uint pad1; uint pad2; }
// (You can also hardcode sample_count in the shader if you prefer)

struct Params {
    sample_count: u32;
    pad0: u32;
    pad1: u32;
    pad2: u32;
};
@group(0) @binding(0) var<uniform> uParams: Params;

let PI: f32 = 3.141592653589793;
let EPS: f32 = 1e-6;

// -------------------- Sampling utilities --------------------

// Radical inverse (Van der Corput) base 2
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

// Importance sample GGX (half-vector) for N = (0,0,1) (tangent-space)
fn importance_sample_ggx(xi: vec2<f32>, roughness: f32) -> vec3<f32> {
    let a: f32 = roughness * roughness;

    let phi: f32 = 2.0 * PI * xi.x;
    let cosTheta: f32 = sqrt(max(0.0, (1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y)));
    let sinTheta: f32 = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

    // half vector in tangent space (TBN where N=(0,0,1))
    let Ht: vec3<f32> = vec3<f32>(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    return normalize(Ht);
}

// Schlick-GGX geometry term (per-direction)
fn G_SchlickGGX(NdotV: f32, k: f32) -> f32 {
    return NdotV / (NdotV * (1.0 - k) + k + EPS);
}

// Smith geometry (combined)
fn G_Smith(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    let r: f32 = roughness + 1.0;
    let k: f32 = (r * r) / 8.0; // UE4 remap for better fit
    let ggx1: f32 = G_SchlickGGX(NdotV, k);
    let ggx2: f32 = G_SchlickGGX(NdotL, k);
    return ggx1 * ggx2;
}

// -------------------- BRDF integration --------------------
// Integrate the specular BRDF for given NdotV and roughness.
// Returns vec2(A, B) where final specular = prefilteredColor * (F0 * A + B)
fn integrate_brdf(NdotV: f32, roughness: f32, sample_count: u32) -> vec2<f32> {
    // View vector in tangent space where N = (0,0,1)
    let V: vec3<f32> = vec3<f32>(sqrt(max(0.0, 1.0 - NdotV * NdotV)), 0.0, NdotV);

    var A: f32 = 0.0;
    var B: f32 = 0.0;

    let N: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);

    // Ensure at least 1 sample
    var samples: u32 = max(1u, sample_count);

    // Loop over Hammersley samples
    for (var i: u32 = 0u; i < samples; i = i + 1u) {
        let xi: vec2<f32> = hammersley(i, samples);
        let H: vec3<f32> = importance_sample_ggx(xi, roughness);
        let VdotH: f32 = max(dot(V, H), 0.0);
        // Compute L by reflecting V around H: L = 2*(V·H)*H - V
        let L: vec3<f32> = normalize(2.0 * VdotH * H - V);

        let NdotL: f32 = max(L.z, 0.0);
        let NdotH: f32 = max(H.z, 0.0);

        if (NdotL > 0.0) {
            // Geometry term
            let G: f32 = G_Smith(NdotV, NdotL, roughness);

            // Visibility term used in split-sum: G_Vis = (G * VdotH) / (NdotH * NdotV)
            let denom: f32 = max(NdotH * NdotV, EPS);
            let G_Vis: f32 = (G * VdotH) / denom;

            // Fresnel approximation factor Fc = (1 - VdotH)^5
            let Fc: f32 = pow(1.0 - VdotH, 5.0);

            // Accumulate A and B as in common implementations
            A = A + (1.0 - Fc) * G_Vis;
            B = B + Fc * G_Vis;
        }
    }

    let inv_samples: f32 = 1.0 / f32(samples);
    A = A * inv_samples;
    B = B * inv_samples;
    return vec2<f32>(A, B);
}

// -------------------- Fullscreen triangle vertex shader --------------------
struct VSOut {
    @builtin(position) pos: vec4<f32>;
    @location(0) uv: vec2<f32>;
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    // Fullscreen triangle (3 vertices)
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );

    var out: VSOut;
    let p: vec2<f32> = positions[vi];
    out.pos = vec4<f32>(p, 0.0, 1.0);
    // UV mapping from clip-space to 0..1
    out.uv = p * 0.5 + vec2<f32>(0.5, 0.5);
    return out;
}

// -------------------- Fragment shader (compute LUT) --------------------
struct FSOut {
    @location(0) color: vec4<f32>; // store vec2 in .xy, .zw unused (or used for debug)
};

@fragment
fn fs_main(in: VSOut) -> FSOut {
    var out: FSOut;

    // UV.x = NdotV, UV.y = roughness
    let NdotV: f32 = clamp(in.uv.x, 0.0, 1.0);
    let roughness: f32 = clamp(in.uv.y, 0.0, 1.0);

    // Determine sample count (allow uniform override)
    let base_samples: u32 = max(1u, uParams.sample_count);
    // Heuristic: more samples for mid/high roughness, fewer for very smooth surfaces
    let scale: f32 = 0.5 + roughness * 1.5; // between 0.5 and 2.0
    var samples: u32 = min(u32(f32(base_samples) * scale), 8192u);

    // Integrate BRDF
    let res: vec2<f32> = integrate_brdf(NdotV, roughness, samples);

    // Pack into RG channels; BA left zero for compatibility
    out.color = vec4<f32>(res.x, res.y, 0.0, 1.0);
    return out;
}
