// ssr.wgsl
// High-performance, featureful Screen-Space Reflections (SSR) shader (WGSL).
// Features:
//  - Ray-marched screen-space reflection with binary search refinement
//  - Roughness-aware max distance and cone/step adaptation
//  - Thickness-aware intersection and thickness compensation
//  - Temporal accumulation with reprojection (history) and clamping to avoid ghosting
//  - Multi-sample jitter + importance fallback to environment map (cubemap)
//  - Normal-aware bias and edge fade (to reduce leaking)
//  - Optional binary-search refinement iterations and performance knobs
// Bindings expected (example):
// group(0) binding(0) : sampler sLinearClamp
// group(0) binding(1) : texture_depth_2d tDepth
// group(0) binding(2) : texture_2d<f32> tNormalRoughness; // RGBA: normal.xy (encoded), roughness in B, metallic/A optional
// group(0) binding(3) : texture_2d<f32> tColor; // current HDR color buffer
// group(0) binding(4) : texture_2d<f32> tVelocity; // motion vectors (UV offset)
// group(0) binding(5) : texture_cube<f32> tEnv; // fallback environment cubemap
// group(0) binding(6) : sampler sEnv; // sampler for env
// group(1) binding(0) : uniform SSRParams { ... }
// group(1) binding(1) : texture_2d<f32> tHistory; // previous SSR accumulation (optional)
// group(1) binding(2) : sampler sPointClamp; // for history sampling
//
// Outputs:
//  - location(0): SSR color (linear HDR)
//  - location(1): SSR history (for next frame)

struct SSRParams {
    max_steps: u32;               // max ray-march steps
    binary_search_steps: u32;     // binary search refinement iterations
    max_distance: f32;            // max view-space distance for rays
    thickness: f32;               // thickness tolerance in view-space units
    stride: f32;                  // base step stride in view-space units
    roughness_fade: f32;          // fade factor for roughness (0..1)
    temporal_alpha: f32;          // history blend factor (0..1)
    temporal_clamp: f32;          // clamp factor for history (0..1)
    env_fallback_lod: f32;        // LOD for environment fallback
    jitter_strength: f32;         // jitter amount for multi-sample
    sample_count: u32;            // number of jitter samples (1..4)
    normal_bias: f32;             // bias along normal to avoid self-intersection
    edge_fade: f32;               // fade near screen edges (0..1)
    pad0: vec3<f32>;
};
@group(1) @binding(0) var<uniform> uSSR: SSRParams;

@group(0) @binding(0) var sLinearClamp: sampler;
@group(0) @binding(1) var tDepth: texture_depth_2d;
@group(0) @binding(2) var tNormalRoughness: texture_2d<f32>;
@group(0) @binding(3) var tColor: texture_2d<f32>;
@group(0) @binding(4) var tVelocity: texture_2d<f32>;
@group(0) @binding(5) var tEnv: texture_cube<f32>;
@group(0) @binding(6) var sEnv: sampler;
@group(1) @binding(1) var tHistory: texture_2d<f32>;
@group(1) @binding(2) var sPointClamp: sampler;

// Uniforms for projection matrices (caller must supply)
struct Proj {
    proj: mat4x4<f32>;
    inv_proj: mat4x4<f32>;
    view: mat4x4<f32>;
    inv_view: mat4x4<f32>;
    cam_pos: vec4<f32>;
    screen_size: vec2<f32>;
};
@group(2) @binding(0) var<uniform> uProj: Proj;

// Vertex -> Fragment
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

// Fragment outputs: SSR color and history
struct FSOut {
    @location(0) color: vec4<f32>;
    @location(1) history: vec4<f32>;
};

// Constants
let PI: f32 = 3.141592653589793;
let EPS: f32 = 1e-6;

// Helpers
fn ndc_from_uv(uv: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(uv * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
}

// Reconstruct view-space position from depth and UV
fn reconstruct_view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    // depth is in 0..1 (NDC depth). Convert to clip z
    let ndc = vec4<f32>(uv * 2.0 - vec2<f32>(1.0), depth * 2.0 - 1.0, 1.0);
    let view = uProj.inv_proj * ndc;
    return view.xyz / max(view.w, EPS);
}

// Project view-space position to screen UV
fn project_view_pos_to_uv(view_pos: vec3<f32>) -> vec2<f32> {
    let clip = uProj.proj * vec4<f32>(view_pos, 1.0);
    let ndc = clip.xyz / max(clip.w, EPS);
    return ndc.xy * 0.5 + vec2<f32>(0.5);
}

// Decode normal and roughness from packed texture
fn decode_normal_roughness(packed: vec4<f32>) -> (vec3<f32>, f32) {
    // assume normal.xy encoded in [-1,1] mapped to [0,1], roughness in b channel
    let nxy = packed.xy * 2.0 - vec2<f32>(1.0, 1.0);
    let nz = sqrt(max(1.0 - dot(nxy, nxy), 0.0));
    let normal = normalize(vec3<f32>(nxy.x, nxy.y, nz));
    let roughness = clamp(packed.z, 0.0, 1.0);
    return (normal, roughness);
}

// Fresnel Schlick (for energy compensation when sampling env)
fn fresnel_schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {
    let one_minus = pow(1.0 - cos_theta, 5.0);
    return F0 + (vec3<f32>(1.0) - F0) * one_minus;
}

// Edge fade factor to reduce leaking near screen edges
fn edge_fade(uv: vec2<f32>) -> f32 {
    let e = uSSR.edge_fade;
    if (e <= 0.0) { return 1.0; }
    let margin = vec2<f32>(e, e);
    let fade_x = smoothstep(0.0, margin.x, uv.x) * smoothstep(1.0, 1.0 - margin.x, uv.x);
    let fade_y = smoothstep(0.0, margin.y, uv.y) * smoothstep(1.0, 1.0 - margin.y, uv.y);
    return fade_x * fade_y;
}

// Jitter generator (low-cost hash)
fn hash2(p: vec2<f32>) -> vec2<f32> {
    let k1 = vec2<f32>(127.1, 311.7);
    let k2 = vec2<f32>(269.5, 183.3);
    let h = fract(sin(vec2<f32>(dot(p, k1), dot(p, k2))) * 43758.5453123);
    return h;
}

// Ray marching in view-space along reflection direction R starting from origin view_pos
// Returns (hit_uv, hit_depth, hit_view_pos, hit_found)
fn ray_march_ssr(origin: vec3<f32>, R: vec3<f32>, roughness: f32, uv0: vec2<f32>) -> (vec2<f32>, f32, vec3<f32>, u32) {
    // roughness influences max distance and stride
    let rough_factor = mix(1.0, 4.0, roughness * uSSR.roughness_fade); // rougher -> search farther but coarser
    let max_dist = uSSR.max_distance * rough_factor;
    let base_stride = uSSR.stride * rough_factor;

    var t: f32 = base_stride; // start a bit away from origin to avoid self-hit
    var prev_uv = uv0;
    var prev_depth = textureSample(tDepth, sLinearClamp, uv0);
    var found: u32 = 0u;
    var hit_uv = vec2<f32>(0.0);
    var hit_depth = 0.0;
    var hit_view = vec3<f32>(0.0);

    // March until max steps or distance
    for (var i: u32 = 0u; i < uSSR.max_steps; i = i + 1u) {
        let sample_pos = origin + R * t;
        // project to screen
        let sample_uv = project_view_pos_to_uv(sample_pos);
        // early exit if outside screen
        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            break;
        }
        // read depth at sample_uv
        let scene_depth = textureSample(tDepth, sLinearClamp, sample_uv);
        // reconstruct view pos of scene depth
        let scene_view = reconstruct_view_pos(sample_uv, scene_depth);
        // compute distance along ray to scene point
        // compare z (view-space forward) distances: if sample_pos.z >= scene_view.z - thickness -> intersection
        // Note: view-space forward is negative Z in many conventions; we compare distances via length along ray
        let delta = sample_pos.z - scene_view.z;
        // thickness compensation: allow small positive delta (ray behind surface) as hit
        if (delta > -uSSR.thickness && delta < uSSR.thickness) {
            // found intersection candidate
            found = 1u;
            hit_uv = sample_uv;
            hit_depth = scene_depth;
            hit_view = scene_view;
            break;
        }
        // advance t by adaptive stride: larger when far or rough
        t = t + base_stride * (1.0 + f32(i) * 0.05) * (1.0 + roughness * 2.0);
        if (t > max_dist) {
            break;
        }
        prev_uv = sample_uv;
        prev_depth = scene_depth;
    }

    // If found and binary search requested, refine along last segment
    if (found == 1u && uSSR.binary_search_steps > 0u) {
        // binary search between t - step and t
        var t0 = max(0.0, t - base_stride * 2.0);
        var t1 = t;
        for (var b: u32 = 0u; b < uSSR.binary_search_steps; b = b + 1u) {
            let tm = 0.5 * (t0 + t1);
            let sample_pos = origin + R * tm;
            let sample_uv = project_view_pos_to_uv(sample_pos);
            if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
                t1 = tm;
                continue;
            }
            let scene_depth = textureSample(tDepth, sLinearClamp, sample_uv);
            let scene_view = reconstruct_view_pos(sample_uv, scene_depth);
            let delta = sample_pos.z - scene_view.z;
            if (delta > -uSSR.thickness && delta < uSSR.thickness) {
                // hit in near half
                t1 = tm;
                hit_uv = sample_uv;
                hit_depth = scene_depth;
                hit_view = scene_view;
            } else {
                t0 = tm;
            }
        }
    }

    return (hit_uv, hit_depth, hit_view, found);
}

// Sample color at hit: read color buffer and optionally blend with environment fallback based on roughness & validity
fn sample_hit_color(hit_uv: vec2<f32>, hit_view: vec3<f32>, R: vec3<f32>, roughness: f32) -> vec3<f32> {
    // read scene color at hit_uv
    let scene_color = textureSample(tColor, sLinearClamp, hit_uv).xyz;
    // compute view-space normal at hit (sample normal texture)
    let packed = textureSample(tNormalRoughness, sLinearClamp, hit_uv);
    let (n_hit, _) = decode_normal_roughness(packed);
    // compute fresnel and blend with environment if rough
    let V = normalize(-hit_view);
    let cosVH = max(dot(V, R), 0.0);
    let F0 = vec3<f32>(0.04); // dielectric fallback; metallic surfaces should be handled by material system
    let F = fresnel_schlick(cosVH, F0);
    // environment fallback sample
    let env_color = textureSampleLevel(tEnv, sEnv, R, uSSR.env_fallback_lod).xyz;
    // blend factor based on roughness and edge fade
    let env_blend = clamp(roughness * 1.5, 0.0, 1.0);
    // final color: mix scene_color and env_color weighted by env_blend and Fresnel
    let color = mix(scene_color, env_color, env_blend);
    // apply a small fresnel boost to specular highlights
    return color * (1.0 + F * 0.2);
}

// Temporal reprojection and clamping
fn temporal_blend(curr: vec3<f32>, uv: vec2<f32>, prev_uv: vec2<f32>, alpha: f32, clamp_factor: f32) -> vec3<f32> {
    // fetch history at prev_uv
    let hist = textureSample(tHistory, sPointClamp, prev_uv).xyz;
    // if history invalid (NaN) or alpha == 0, return curr
    if (!all(isFinite(hist)) || alpha <= 0.0) {
        return curr;
    }
    // clamp history to neighborhood of current to avoid ghosting
    // compute min/max from small 3x3 neighborhood around uv
    var minc = vec3<f32>(1e9);
    var maxc = vec3<f32>(-1e9);
    let dims = uProj.screen_size;
    let texel = vec2<f32>(1.0 / dims.x, 1.0 / dims.y);
    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let s = clamp(uv + vec2<f32>(f32(x) * texel.x, f32(y) * texel.y), vec2<f32>(0.0), vec2<f32>(1.0));
            let c = textureSample(tColor, sLinearClamp, s).xyz;
            minc = min(minc, c);
            maxc = max(maxc, c);
        }
    }
    // expand min/max by clamp_factor
    let range = maxc - minc;
    minc = minc - range * clamp_factor;
    maxc = maxc + range * clamp_factor;
    let hist_clamped = clamp(hist, minc, maxc);
    // blend
    return mix(curr, hist_clamped, alpha);
}

// Main fragment
@fragment
fn fs_main(in: VSOut) -> FSOut {
    var out: FSOut;
    let uv = in.uv;

    // Early exit: if outside screen or depth is 1.0 (sky), return black and keep history
    let center_depth = textureSample(tDepth, sLinearClamp, uv);
    if (center_depth >= 1.0) {
        let prev = textureSample(tHistory, sPointClamp, uv).xyz;
        out.color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.history = vec4<f32>(prev, 1.0);
        return out;
    }

    // decode normal & roughness at current pixel
    let packed = textureSample(tNormalRoughness, sLinearClamp, uv);
    let (N, roughness) = decode_normal_roughness(packed);

    // reconstruct view-space position
    let view_pos = reconstruct_view_pos(uv, center_depth);

    // compute view vector
    let V = normalize(-view_pos);

    // compute reflection direction in view-space
    let R = normalize(reflect(-V, N));

    // apply normal bias to origin
    let origin = view_pos + N * uSSR.normal_bias;

    // multi-sample jitter loop (small number of samples)
    var accum_color = vec3<f32>(0.0);
    var accum_weight = 0.0;

    let samples = max(1u, uSSR.sample_count);
    // base jitter seed from uv
    let seed = hash2(uv);
    for (var s: u32 = 0u; s < samples; s = s + 1u) {
        // jitter in hemisphere around R to simulate glossy spread (small)
        var jitter = vec2<f32>(0.0);
        if (uSSR.jitter_strength > 0.0) {
            let h = hash2(uv + vec2<f32>(f32(s), seed.x));
            jitter = (h - vec2<f32>(0.5)) * uSSR.jitter_strength * roughness;
        }
        // convert jitter in screen-space to small angular perturbation: rotate R slightly around N
        let angle = (jitter.x + jitter.y) * 2.0 * PI * 0.5;
        let cosA = cos(angle);
        let sinA = sin(angle);
        // build tangent basis around R
        var up = vec3<f32>(0.0, 1.0, 0.0);
        if (abs(R.y) > 0.999) { up = vec3<f32>(1.0, 0.0, 0.0); }
        let T = normalize(cross(up, R));
        let B = cross(R, T);
        let Rj = normalize(R * cosA + T * sinA * jitter.x + B * sinA * jitter.y);

        // perform ray march
        let (hit_uv, hit_depth, hit_view, found) = ray_march_ssr(origin, Rj, roughness, uv);

        var sample_color = vec3<f32>(0.0);
        var weight = 1.0;
        if (found == 1u) {
            // sample color at hit
            sample_color = sample_hit_color(hit_uv, hit_view, Rj, roughness);
            // weight by N·L (approx) and fresnel
            let L = normalize(-hit_view); // direction from hit to camera
            let nDotL = max(dot(N, L), 0.0);
            weight = max(0.05, nDotL);
        } else {
            // fallback to environment map
            sample_color = textureSampleLevel(tEnv, sEnv, Rj, uSSR.env_fallback_lod).xyz;
            // reduce weight for fallback based on roughness
            weight = 0.25 + 0.75 * roughness;
        }

        accum_color = accum_color + sample_color * weight;
        accum_weight = accum_weight + weight;
    }

    var ssr_color = accum_color / max(accum_weight, 1e-6);

    // edge fade to reduce leaking near edges
    let ef = edge_fade(uv);
    ssr_color = mix(textureSample(tEnv, sEnv, R).xyz, ssr_color, ef);

    // Temporal reprojection: compute previous UV via velocity texture
    let vel = textureSample(tVelocity, sLinearClamp, uv).xy;
    // assume velocity is in UV units (prev_uv = uv + vel)
    let prev_uv = clamp(uv + vel, vec2<f32>(0.0), vec2<f32>(1.0));

    // blend with history
    let alpha = uSSR.temporal_alpha;
    let clamp_factor = uSSR.temporal_clamp;
    let blended = temporal_blend(ssr_color, uv, prev_uv, alpha, clamp_factor);

    // final output: apply roughness fade (reduce SSR for very rough surfaces)
    let rough_fade = 1.0 - smoothstep(0.0, 1.0, roughness * uSSR.roughness_fade);
    let final_color = mix(textureSample(tColor, sLinearClamp, uv).xyz, blended, rough_fade);

    out.color = vec4<f32>(final_color, 1.0);
    out.history = vec4<f32>(final_color, 1.0);
    return out;
}
