// pbr_full.wgsl
// High-performance, feature-rich PBR shader (WGSL).
// Features:
//  - Metallic-Roughness PBR (Disney/UE style)
//  - Normal mapping (TBN, normal texture in tangent space)
//  - Ambient Occlusion, Emissive
//  - Image-Based Lighting (IBL): irradiance (diffuse), prefiltered environment (specular) with LOD
//  - BRDF 2D LUT sampling for split-sum approximation
//  - Clearcoat (single-layer clearcoat with its own roughness)
//  - Energy-conserving Fresnel (Schlick) and GGX microfacet model (NDF + Smith geometry)
//  - Optimized math and minimal branching for performance
//
// Bindings expected (example layout):
// group 0: camera / frame uniforms
//   binding 0: CameraUniform (viewProj, view, proj, position, etc.)
// group 1: material textures + samplers
//   binding 0: texture_2d<f32> albedo;
//   binding 1: texture_2d<f32> normal;
//   binding 2: texture_2d<f32> metallicRoughness; // R=metallic, G=roughness
//   binding 3: texture_2d<f32> ao;
//   binding 4: texture_2d<f32> emissive;
//   binding 5: sampler linear_sampler;
// group 2: IBL
//   binding 0: texture_cube<f32> env_specular; // prefiltered environment (mipmapped)
//   binding 1: texture_cube<f32> env_irradiance; // irradiance map (low freq)
//   binding 2: texture_2d<f32> brdf_lut; // 2D BRDF integration LUT
//   binding 3: sampler linear_clamp_sampler;
//
// Vertex inputs: position, normal, tangent, uv
// Vertex outputs to fragment: worldPos, uv, TBN, viewDir (world), normal (world)
//
// Notes:
//  - This shader assumes positions and normals are provided in model space and a model matrix is applied externally.
//  - For best performance, prefilter environment maps offline and generate BRDF LUT.
//  - Keep texture formats as float (RGBA16F/32F) for env maps and BRDF LUT.

struct CameraUniform {
    view_proj: mat4x4<f32>;
    view: mat4x4<f32>;
    proj: mat4x4<f32>;
    position: vec4<f32>;
    _pad0: vec4<f32>;
    _pad1: vec4<f32>;
    _pad2: vec4<f32>;
};
@group(0) @binding(0) var<uniform> uCamera: CameraUniform;

// Material textures
@group(1) @binding(0) var tAlbedo: texture_2d<f32>;
@group(1) @binding(1) var tNormal: texture_2d<f32>;
@group(1) @binding(2) var tMetallicRoughness: texture_2d<f32>;
@group(1) @binding(3) var tAO: texture_2d<f32>;
@group(1) @binding(4) var tEmissive: texture_2d<f32>;
@group(1) @binding(5) var sLinear: sampler;

// IBL
@group(2) @binding(0) var tEnvSpecular: texture_cube<f32>;
@group(2) @binding(1) var tEnvIrradiance: texture_cube<f32>;
@group(2) @binding(2) var tBRDFLUT: texture_2d<f32>;
@group(2) @binding(3) var sLinearClamp: sampler;

// Vertex attributes (match your vertex buffer layout)
struct VertexInput {
    @location(0) position: vec3<f32>;
    @location(1) normal: vec3<f32>;
    @location(2) tangent: vec4<f32>; // xyz = tangent, w = handedness
    @location(3) uv: vec2<f32>;
};

// Vertex -> Fragment payload
struct Varyings {
    @builtin(position) clip_pos: vec4<f32>;
    @location(0) world_pos: vec3<f32>;
    @location(1) uv: vec2<f32>;
    @location(2) tbn0: vec3<f32>;
    @location(3) tbn1: vec3<f32>;
    @location(4) tbn2: vec3<f32>;
    @location(5) view_dir: vec3<f32>;
    @location(6) normal_ws: vec3<f32>;
};

// Simple model matrix uniform (if you have one, otherwise assume identity)
// For this shader we assume positions are already in world space; if not, provide model matrix.
@vertex
fn vs_main(in: VertexInput) -> Varyings {
    var out: Varyings;

    // If your positions are in model space, multiply by model matrix here.
    // For portability we assume positions are already world-space.
    let world_pos: vec3<f32> = in.position;

    // Build TBN from vertex normal and tangent (tangent.w is handedness)
    let n: vec3<f32> = normalize(in.normal);
    let t: vec3<f32> = normalize(in.tangent.xyz);
    let b: vec3<f32> = cross(n, t) * in.tangent.w;

    out.world_pos = world_pos;
    out.uv = in.uv;
    out.tbn0 = t;
    out.tbn1 = b;
    out.tbn2 = n;
    out.normal_ws = n;

    // Compute view direction in world space
    let cam_pos = vec3<f32>(uCamera.position.x, uCamera.position.y, uCamera.position.z);
    out.view_dir = normalize(cam_pos - world_pos);

    // Transform to clip space using camera view-proj
    out.clip_pos = uCamera.view_proj * vec4<f32>(world_pos, 1.0);

    return out;
}

// ---------- PBR helper functions ----------

// Constants
let PI: f32 = 3.141592653589793;
let EPSILON: f32 = 1e-5;

// GGX / Trowbridge-Reitz normal distribution function
fn D_GGX(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a: f32 = roughness * roughness;
    let a2: f32 = a * a;
    let NdotH: f32 = max(dot(N, H), 0.0);
    let denom: f32 = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + EPSILON);
}

// Schlick-GGX geometry term (Smith's method)
fn G_SchlickGGX(NdotV: f32, k: f32) -> f32 {
    return NdotV / (NdotV * (1.0 - k) + k + EPSILON);
}

fn G_Smith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    let r: f32 = roughness + 1.0;
    let k: f32 = (r * r) / 8.0; // UE4 style remapping
    let NdotV: f32 = max(dot(N, V), 0.0);
    let NdotL: f32 = max(dot(N, L), 0.0);
    let ggx1: f32 = G_SchlickGGX(NdotV, k);
    let ggx2: f32 = G_SchlickGGX(NdotL, k);
    return ggx1 * ggx2;
}

// Fresnel Schlick approximation
fn Fresnel_Schlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    // Schlick: F = F0 + (1 - F0) * (1 - cosTheta)^5
    let one_minus = pow(1.0 - cosTheta, 5.0);
    return F0 + (1.0 - F0) * one_minus;
}

// Fresnel Schlick with roughness (for specular IBL)
fn Fresnel_Schlick_Roughness(cosTheta: f32, F0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return F0 + (max(vec3<f32>(1.0 - roughness), vec3<f32>(F0.x, F0.y, F0.z)) - F0) * pow(1.0 - cosTheta, 5.0);
}

// Convert sRGB to linear (approx)
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    // Use approximate gamma correction for speed
    return pow(c, vec3<f32>(2.2));
}

// Convert linear to sRGB (approx)
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    return pow(c, vec3<f32>(1.0 / 2.2));
}

// Sample normal map and compute perturbed normal in world space using TBN
fn getNormalFromMap(uv: vec2<f32>, tbn0: vec3<f32>, tbn1: vec3<f32>, tbn2: vec3<f32>) -> vec3<f32> {
    let nmap: vec3<f32> = textureSample(tNormal, sLinear, uv).xyz * 2.0 - vec3<f32>(1.0);
    // Transform from tangent space to world space
    let T: vec3<f32> = tbn0;
    let B: vec3<f32> = tbn1;
    let N: vec3<f32> = tbn2;
    let perturbed: vec3<f32> = normalize(nmap.x * T + nmap.y * B + nmap.z * N);
    return perturbed;
}

// Sample prefiltered environment (specular) using roughness -> LOD mapping
fn samplePrefilteredEnv(R: vec3<f32>, roughness: f32) -> vec3<f32> {
    // Map roughness to mip level: assume max mip ~ 8 (precomputed)
    // For performance, clamp and compute LOD directly
    let max_mip: f32 = 8.0;
    let mip: f32 = roughness * max_mip;
    // textureSampleLevel is not available for cube in all backends; use sample with bias via sampleLevel if supported.
    // WGSL provides textureSampleLevel for explicit LOD.
    return textureSampleLevel(tEnvSpecular, sLinearClamp, R, mip).xyz;
}

// Sample irradiance map (diffuse IBL)
fn sampleIrradiance(N: vec3<f32>) -> vec3<f32> {
    return textureSample(tEnvIrradiance, sLinearClamp, N).xyz;
}

// Sample BRDF LUT (preintegrated) using NdotV and roughness
fn sampleBRDF(NdotV: f32, roughness: f32) -> vec2<f32> {
    return textureSampleLevel(tBRDFLUT, sLinearClamp, vec2<f32>(NdotV, roughness), 0.0).xy;
}

// Tone mapping (ACES approximation, simple)
fn toneMapACES(color: vec3<f32>) -> vec3<f32> {
    // RRT + ODT fit (approx)
    let a: f32 = 2.51;
    let b: f32 = 0.03;
    let c: f32 = 2.43;
    let d: f32 = 0.59;
    let e: f32 = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ---------- Fragment stage ----------

struct FragmentInput {
    @location(0) world_pos: vec3<f32>;
    @location(1) uv: vec2<f32>;
    @location(2) tbn0: vec3<f32>;
    @location(3) tbn1: vec3<f32>;
    @location(4) tbn2: vec3<f32>;
    @location(5) view_dir: vec3<f32>;
    @location(6) normal_ws: vec3<f32>;
};

struct FragmentOutput {
    @location(0) out_color: vec4<f32>;
};

@fragment
fn fs_main(in: FragmentInput) -> FragmentOutput {
    var out: FragmentOutput;

    // Sample material textures
    let albedo_srgb: vec3<f32> = textureSample(tAlbedo, sLinear, in.uv).xyz;
    let albedo: vec3<f32> = srgb_to_linear(albedo_srgb);

    let normal_ws: vec3<f32> = getNormalFromMap(in.uv, in.tbn0, in.tbn1, in.tbn2);

    let mr: vec2<f32> = textureSample(tMetallicRoughness, sLinear, in.uv).xy;
    let metallic: f32 = clamp(mr.x, 0.0, 1.0);
    let roughness: f32 = clamp(mr.y, 0.04, 1.0); // avoid 0 roughness

    let ao: f32 = textureSample(tAO, sLinear, in.uv).x;
    let emissive_srgb: vec3<f32> = textureSample(tEmissive, sLinear, in.uv).xyz;
    let emissive: vec3<f32> = srgb_to_linear(emissive_srgb);

    // View and normal
    let V: vec3<f32> = normalize(in.view_dir);
    let N: vec3<f32> = normalize(normal_ws);

    // Compute reflectance at normal incidence (F0)
    let base_color: vec3<f32> = albedo;
    // Dielectric F0 ~ 0.04, metals use base color
    let F0_dielectric: vec3<f32> = vec3<f32>(0.04, 0.04, 0.04);
    let F0: vec3<f32> = mix(F0_dielectric, base_color, metallic);

    // Direct lighting (simple single directional light for demo; replace with your light system)
    // For performance, you can inject light via uniform or clustered lights; here we use a single directional light.
    let lightDir: vec3<f32> = normalize(vec3<f32>(0.5, 0.8, 0.6)); // world-space light direction
    let L: vec3<f32> = normalize(lightDir);
    let H: vec3<f32> = normalize(V + L);

    let NdotL: f32 = max(dot(N, L), 0.0);
    let NdotV: f32 = max(dot(N, V), 0.0);
    let NdotH: f32 = max(dot(N, H), 0.0);
    let VdotH: f32 = max(dot(V, H), 0.0);

    // Microfacet terms
    let D: f32 = D_GGX(N, H, roughness);
    let G: f32 = G_Smith(N, V, L, roughness);
    let F: vec3<f32> = Fresnel_Schlick(VdotH, F0);

    // Specular BRDF (Cook-Torrance)
    let numerator: vec3<f32> = D * G * F;
    let denominator: f32 = 4.0 * NdotV * NdotL + EPSILON;
    let specular: vec3<f32> = numerator / denominator;

    // kS is energy preserved specular, kD is diffuse component
    let kS: vec3<f32> = F;
    let kD: vec3<f32> = (vec3<f32>(1.0) - kS) * (1.0 - metallic);

    // Lambertian diffuse (energy-conserving)
    let diffuse: vec3<f32> = (kD * base_color / PI);

    // Direct lighting contribution (assume white directional light with intensity 1.0)
    let radiance: vec3<f32> = vec3<f32>(1.0); // directional light color/intensity
    let Lo_direct: vec3<f32> = (diffuse + specular) * radiance * NdotL;

    // IBL (Image-Based Lighting)
    // Sample irradiance for diffuse
    let irradiance: vec3<f32> = sampleIrradiance(N);
    let diffuse_IBL: vec3<f32> = irradiance * base_color;

    // Sample prefiltered environment for specular
    let R: vec3<f32> = reflect(-V, N);
    let prefilteredColor: vec3<f32> = samplePrefilteredEnv(R, roughness);

    // Sample BRDF LUT
    let brdf: vec2<f32> = sampleBRDF(NdotV, roughness);
    let specular_IBL: vec3<f32> = prefilteredColor * (F * brdf.x + brdf.y);

    // Ambient occlusion modulates diffuse IBL and ambient
    let ambient: vec3<f32> = (diffuse_IBL + specular_IBL) * ao;

    // Clearcoat (single-layer) - optional, simple approximation
    // For performance, clearcoat uses a separate F0 and roughness factor
    let clearcoat_factor: f32 = 0.0; // set >0 to enable (0..1)
    let clearcoat_roughness: f32 = 0.1;
    var clearcoat_spec: vec3<f32> = vec3<f32>(0.0);
    if (clearcoat_factor > 0.0) {
        let Fc: vec3<f32> = Fresnel_Schlick(VdotH, vec3<f32>(0.04));
        let Dc: f32 = D_GGX(N, H, clearcoat_roughness);
        let Gc: f32 = G_Smith(N, V, L, clearcoat_roughness);
        let numerator_c: vec3<f32> = Dc * Gc * Fc;
        let denom_c: f32 = 4.0 * NdotV * NdotL + EPSILON;
        clearcoat_spec = clearcoat_factor * (numerator_c / denom_c);
    }

    // Final color composition
    let Lo: vec3<f32> = Lo_direct + ambient + clearcoat_spec + emissive;

    // Apply tone mapping and gamma
    let color_linear: vec3<f32> = Lo;
    let color_tonemapped: vec3<f32> = toneMapACES(color_linear);
    let color_srgb: vec3<f32> = linear_to_srgb(color_tonemapped);

    out.out_color = vec4<f32>(color_srgb, 1.0);
    return out;
}
