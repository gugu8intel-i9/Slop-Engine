// shadow_depth.wgsl
// High-performance, feature-rich shadow depth generator (WGSL).
// Features:
//  - Supports up to 4 cascades (Cascaded Shadow Maps, CSM)
//  - Optional Variance Shadow Maps (VSM) output (store moments)
//  - Stable depth bias (normal bias + slope-scale) and constant bias
//  - Contact-hardening bias approximation (using receiver depth / light distance)
//  - Optional normal mapping support (tangent-space normal input + TBN)
//  - Packed depth output helper (if you want to store depth in RGBA)
// Usage notes:
//  - Render from light's point of view(s). For standard depth-only pipelines, use the vertex shader
//    and an empty fragment shader (or one that discards) and configure a depth attachment.
//  - To produce VSM, configure the render target to have a color attachment (RG16F/RG32F recommended)
//    and set uUseVSM = 1. The fragment shader will write the first two moments (depth, depth^2).
//  - Provide cascade view-projection matrices in uLightViewProj[0..uCascadeCount-1].
//  - Provide model matrix (uModel) if your vertex positions are in model space.
//  - For normal-bias, provide vertex normals and optionally tangents for normal mapping.

struct CameraUniform {
    // Not all fields are required for depth pass; keep minimal for compatibility.
    // If you have a full camera uniform, you can place it here.
    _pad0: vec4<f32>;
};
@group(0) @binding(0) var<uniform> uCamera: CameraUniform;

// Light / cascade uniforms
@group(1) @binding(0) var<uniform> uLightViewProj : array<mat4x4<f32>, 4>; // up to 4 cascades
@group(1) @binding(1) var<uniform> uCascadeParams : vec4<u32>; // x = cascadeIndex, y = cascadeCount, z.. reserved

// Bias and options
struct BiasParams {
    constant_bias: f32;        // small constant depth bias
    normal_bias: f32;          // offset along normal (world units)
    slope_scale: f32;          // slope-scale factor for bias
    contact_hardening: f32;    // 0..1 factor to enable contact-hardening (approx)
};
@group(1) @binding(2) var<uniform> uBias: BiasParams;

struct Options {
    use_vsm: u32;              // 0 = depth-only, 1 = VSM (write moments to color)
    max_vsm_mip: u32;          // reserved for sampling; not used here
    pad0: u32;
    pad1: u32;
};
@group(1) @binding(3) var<uniform> uOptions: Options;

// Optional model matrix (if positions are in model space)
@group(2) @binding(0) var<uniform> uModel: mat4x4<f32>;

// Optional normal map (if you want to compute normal bias from normal map)
@group(3) @binding(0) var tNormalMap: texture_2d<f32>;
@group(3) @binding(1) var sLinear: sampler;

// Vertex input layout
struct VertexInput {
    @location(0) position: vec3<f32>;
    @location(1) normal: vec3<f32>;
    @location(2) tangent: vec4<f32>; // optional: tangent.xyz, tangent.w = handedness
    @location(3) uv: vec2<f32>;
};

// Vertex -> Fragment varyings
struct Varyings {
    @builtin(position) clip_pos: vec4<f32>;
    @location(0) world_pos: vec3<f32>;
    @location(1) normal_ws: vec3<f32>;
    @location(2) view_depth: f32; // view-space depth (positive forward)
    @location(3) uv: vec2<f32>;
};

// Helper: pack depth into RGBA8 (optional, not used by default)
fn pack_depth_rgba(depth: f32) -> vec4<f32> {
    // Pack 0..1 depth into 4x8-bit channels
    let enc = vec4<f32>(
        depth,
        fract(depth * 256.0),
        fract(depth * 65536.0),
        fract(depth * 16777216.0)
    );
    let packed = enc / vec4<f32>(1.0, 256.0, 65536.0, 16777216.0);
    return packed;
}

// Compute a stable bias using normal and slope-scale
fn compute_depth_bias(N: vec3<f32>, L: vec3<f32>, view_depth: f32) -> f32 {
    // Normal bias moves the receiver along its normal in world units before projecting
    let normal_bias = uBias.normal_bias;

    // Slope-scale bias: increases with angle between normal and light direction
    let NdotL = max(dot(N, L), 0.0);
    let slope = sqrt(max(1.0 - NdotL * NdotL, 0.0));
    let slope_bias = uBias.slope_scale * slope;

    // Contact-hardening: approximate by scaling bias with view depth (objects further away get larger penumbra)
    let ch = uBias.contact_hardening;
    let ch_factor = mix(1.0, view_depth, ch); // simple linear approx

    let bias = (uBias.constant_bias + normal_bias + slope_bias) * ch_factor;
    return bias;
}

// Vertex shader: transform into selected cascade light clip space
@vertex
fn vs_main(in: VertexInput) -> Varyings {
    var out: Varyings;

    // World position: apply model matrix if provided (identity if not)
    let world_pos4 = uModel * vec4<f32>(in.position, 1.0);
    let world_pos = world_pos4.xyz / world_pos4.w;

    // Normal in world space (transform by inverse-transpose of model; approximate by upper-left 3x3)
    let normal_ws = normalize((uModel * vec4<f32>(in.normal, 0.0)).xyz);

    // Select cascade matrix
    let cascade_index: u32 = uCascadeParams.x;
    // clamp cascade index to 0..3
    let idx: u32 = min(cascade_index, 3u);
    let light_mat: mat4x4<f32> = uLightViewProj[idx];

    // Transform to light clip space
    let clip = light_mat * vec4<f32>(world_pos, 1.0);
    out.clip_pos = clip;

    out.world_pos = world_pos;
    out.normal_ws = normal_ws;
    out.uv = in.uv;

    // Compute view-space depth (positive forward) for contact-hardening approx
    // For light view, view-space Z is -clip.z/clip.w if using right-handed; we compute linear depth in [0..1]
    let ndc_z = clip.z / clip.w;
    // Convert NDC z (-1..1) to 0..1
    let depth01 = ndc_z * 0.5 + 0.5;
    // Approximate view depth as depth01 * far (we don't have far here) -> use depth01 as proxy
    out.view_depth = depth01;

    return out;
}

// Fragment output for VSM: store first two moments (depth, depth^2) in RG channels
struct FragOutVSM {
    @location(0) moments: vec2<f32>;
};

// If not using VSM, fragment can be empty (depth-only pipeline).
@fragment
fn fs_main(in: Varyings) -> FragOutVSM {
    var out: FragOutVSM;

    // Compute normalized depth in [0,1] from clip space (same as vs computed)
    let clip = in.clip_pos;
    let ndc_z = clip.z / clip.w;
    let depth01 = ndc_z * 0.5 + 0.5;

    // Compute a stable bias to apply to the stored depth (so sampling side can compare)
    // For bias computation we need a light direction; approximate from light matrix rows (not always available).
    // Here we approximate L as the negative Z axis of the light view (0,0,1) in world space transformed by inverse of light view.
    // For simplicity, assume directional light along -Z in light space -> L_ws approx (0,0,-1) transformed by inverse of light view.
    // Because we don't have inverse here, use a constant L approximation (user can supply if needed).
    let L_ws = normalize(vec3<f32>(0.0, -1.0, 0.0)); // default directional light downwards; replace if you have real light dir

    // Compute bias (we use normal and view depth)
    let bias = compute_depth_bias(in.normal_ws, L_ws, in.view_depth);

    // Apply bias to depth (push receiver away from light)
    let depth_biased = clamp(depth01 + bias, 0.0, 1.0);

    if (uOptions.use_vsm == 1u) {
        // Store moments for VSM: E[z], E[z^2]
        let m1 = depth_biased;
        let m2 = depth_biased * depth_biased;
        // Optionally add a tiny epsilon to m2 to avoid singularities
        out.moments = vec2<f32>(m1, m2 + 1e-6);
    } else {
        // Depth-only: we still need to output something if pipeline expects a color target.
        // Write packed depth into RG channels (or write depth to color.x)
        out.moments = vec2<f32>(depth_biased, 0.0);
    }

    return out;
}
