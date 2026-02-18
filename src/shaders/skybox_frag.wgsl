// Fragment shader for skybox sampling env cubemap or procedural sky
@group(0) @binding(0) var env_cubemap: texture_cube<f32>;
@group(0) @binding(1) var prefiltered: texture_cube<f32>;
@group(0) @binding(2) var irradiance: texture_cube<f32>;
@group(0) @binding(3) var brdf_lut: texture_2d<f32>;
@group(0) @binding(4) var samp: sampler;

struct FSIn {
    @location(0) v_dir: vec3<f32>;
};

@fragment
fn fs_main(in: FSIn) -> @location(0) vec4<f32> {
    let dir = normalize(in.v_dir);
    // sample environment directly for skybox display (use base level)
    let color = textureSampleLevel(env_cubemap, samp, dir, 0.0).rgb;
    // apply exposure/gamma if desired (handled by post)
    return vec4<f32>(color, 1.0);
}
