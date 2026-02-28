// ---------- Bindings and uniforms ----------

// Group 0: per-frame camera / mode / light
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>, // xyz = camera position
    mode: u32,           // 0 = 2D, 1 = 3D
    _pad: vec3<f32>,
};
@group(0) @binding(0)
var<uniform> uCamera: CameraUniform;

// Light (only used in 3D)
struct Light {
    direction: vec4<f32>, // xyz normalized, w = intensity
    color: vec4<f32>,
};
@group(0) @binding(1)
var<uniform> uLight: Light;

// Group 1: material params + textures
struct MaterialParams {
    base_color_factor: vec4<f32>,
    metallic_factor: f32,
    roughness_factor: f32,
    ao_factor: f32,
    flags: u32, // bit0=base_color_tex, bit1=mr_tex, bit2=normal_tex, bit3=ao_tex
    _pad: vec3<f32>,
};
@group(1) @binding(0)
var<uniform> uMaterial: MaterialParams;

@group(1) @binding(1) var base_color_tex: texture_2d<f32>;
@group(1) @binding(2) var mr_tex: texture_2d<f32>;
@group(1) @binding(3) var normal_tex: texture_2d<f32>;
@group(1) @binding(4) var ao_tex: texture_2d<f32>;
@group(1) @binding(5) var linear_sampler: sampler;

// Group 2: per-object model matrix
@group(2) @binding(0)
var<uniform> uModel: mat4x4<f32>;

// ---------- Vertex input (single interleaved layout) ----------
struct VertexInput {
    @location(0) position: vec3<f32>, // for 2D: z ignored
    @location(1) normal: vec3<f32>,   // optional for 2D
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec3<f32>,  // optional for 2D
};

// ---------- VS -> FS payload ----------
struct VSOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec3<f32>,
};

// ---------- Vertex shader ----------
@vertex
fn vs_main(input: VertexInput) -> VSOut {
    var out: VSOut;

    // Compute world position (works for 2D and 3D)
    let world_pos4 = uModel * vec4<f32>(input.position, 1.0);
    out.world_pos = world_pos4.xyz;

    // For 2D mode we still produce a clip position using view_proj,
    // but keep other work minimal.
    out.clip_pos = uCamera.view_proj * world_pos4;

    // Transform normals/tangents (cheap if mode==2D they may be unused)
    let n4 = uModel * vec4<f32>(input.normal, 0.0);
    out.normal = normalize(n4.xyz);

    let t4 = uModel * vec4<f32>(input.tangent, 0.0);
    out.tangent = normalize(t4.xyz);

    out.uv = input.uv;
    return out;
}

// ---------- Helpers (PBR math) ----------
let PI: f32 = 3.141592653589793;

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

fn distribution_ggx(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 1e-6);
}

fn geometry_schlick_ggx(NdotV: f32, k: f32) -> f32 {
    return NdotV / (NdotV * (1.0 - k) + k + 1e-6);
}

fn geometry_smith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, k: f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    return geometry_schlick_ggx(NdotV, k) * geometry_schlick_ggx(NdotL, k);
}

fn apply_normal_map(nmap: vec3<f32>, N: vec3<f32>, T: vec3<f32>) -> vec3<f32> {
    let B = normalize(cross(N, T));
    let Tn = normalize(T);
    let tbn = mat3x3<f32>(Tn, B, N);
    let n = normalize(nmap * 2.0 - vec3<f32>(1.0, 1.0, 1.0));
    return normalize(tbn * n);
}

// ---------- Fragment input / output ----------
struct FSIn {
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec3<f32>,
};

struct FSOut {
    @location(0) color: vec4<f32>,
};

// ---------- Fragment shader (2D fast path vs 3D PBR path) ----------
@fragment
fn fs_main(input: FSIn) -> FSOut {
    // Mode check (0 = 2D, 1 = 3D)
    let mode3d = (uCamera.mode == 1u);

    // --- Base color (both modes) ---
    var base_color = uMaterial.base_color_factor.rgb;
    if ((uMaterial.flags & 1u) != 0u) {
        base_color = textureSample(base_color_tex, linear_sampler, input.uv).rgb * uMaterial.base_color_factor.rgb;
    }

    // Fast 2D path: minimal math, no lighting, no normal maps
    if (!mode3d) {
        return FSOut(vec4<f32>(base_color, uMaterial.base_color_factor.a));
    }

    // --- 3D PBR path (only executed when mode==1) ---
    // Metallic / Roughness
    var metallic = uMaterial.metallic_factor;
    var roughness = saturate(uMaterial.roughness_factor);
    if ((uMaterial.flags & 2u) != 0u) {
        let mr = textureSample(mr_tex, linear_sampler, input.uv).rgb;
        metallic = mr.b * metallic;
        roughness = saturate(mr.g * roughness);
    }

    // AO
    var ao = uMaterial.ao_factor;
    if ((uMaterial.flags & 8u) != 0u) {
        ao = ao * textureSample(ao_tex, linear_sampler, input.uv).r;
    }

    // Normal mapping
    var N = normalize(input.normal);
    if ((uMaterial.flags & 4u) != 0u) {
        let nmap = textureSample(normal_tex, linear_sampler, input.uv).rgb;
        N = apply_normal_map(nmap, N, input.tangent);
    }

    // View and light vectors
    let V = normalize(uCamera.view_pos.xyz - input.world_pos);
    let L = normalize(-uLight.direction.xyz);
    let H = normalize(V + L);

    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let VdotH = max(dot(V, H), 0.0);

    // Fresnel F0
    let dielectric_f0 = vec3<f32>(0.04, 0.04, 0.04);
    let f0 = mix(dielectric_f0, base_color, metallic);

    let D = distribution_ggx(N, H, roughness);
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let G = geometry_smith(N, V, L, k);
    let F = fresnel_schlick(VdotH, f0);

    let numerator = D * G * F;
    let denominator = 4.0 * max(NdotV * NdotL, 1e-6);
    let specular = numerator / denominator;

    let kd = (vec3<f32>(1.0, 1.0, 1.0) - F) * (1.0 - metallic);

    let radiance = uLight.color.rgb * uLight.direction.w;

    let Lo = (kd * base_color / PI + specular) * radiance * NdotL;
    let ambient = vec3<f32>(0.03, 0.03, 0.03) * base_color * ao;

    var color = ambient + Lo;

    // ACES-like tonemap + gamma
    color = color / (color + vec3<f32>(1.0, 1.0, 1.0));
    color = pow(color, vec3<f32>(1.0 / 2.2));

    return FSOut(vec4<f32>(color, 1.0));
}
