// Vertex shader for skybox cube.
// Compatibility/perf notes:
// - Uses a static 36-vertex cube list (no index buffer required).
// - Emits clip-space with z=w so the skybox always lands at the far plane.

struct VSOut {
    @builtin(position) clip: vec4<f32>;
    @location(0) v_dir: vec3<f32>;
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VSOut {
    let positions = array<vec3<f32>, 36>(
        // -Z
        vec3<f32>(-1.0, -1.0, -1.0), vec3<f32>( 1.0, -1.0, -1.0), vec3<f32>( 1.0,  1.0, -1.0),
        vec3<f32>( 1.0,  1.0, -1.0), vec3<f32>(-1.0,  1.0, -1.0), vec3<f32>(-1.0, -1.0, -1.0),
        // +Z
        vec3<f32>(-1.0, -1.0,  1.0), vec3<f32>( 1.0, -1.0,  1.0), vec3<f32>( 1.0,  1.0,  1.0),
        vec3<f32>( 1.0,  1.0,  1.0), vec3<f32>(-1.0,  1.0,  1.0), vec3<f32>(-1.0, -1.0,  1.0),
        // -X
        vec3<f32>(-1.0,  1.0,  1.0), vec3<f32>(-1.0,  1.0, -1.0), vec3<f32>(-1.0, -1.0, -1.0),
        vec3<f32>(-1.0, -1.0, -1.0), vec3<f32>(-1.0, -1.0,  1.0), vec3<f32>(-1.0,  1.0,  1.0),
        // +X
        vec3<f32>( 1.0,  1.0,  1.0), vec3<f32>( 1.0,  1.0, -1.0), vec3<f32>( 1.0, -1.0, -1.0),
        vec3<f32>( 1.0, -1.0, -1.0), vec3<f32>( 1.0, -1.0,  1.0), vec3<f32>( 1.0,  1.0,  1.0),
        // -Y
        vec3<f32>(-1.0, -1.0, -1.0), vec3<f32>( 1.0, -1.0, -1.0), vec3<f32>( 1.0, -1.0,  1.0),
        vec3<f32>( 1.0, -1.0,  1.0), vec3<f32>(-1.0, -1.0,  1.0), vec3<f32>(-1.0, -1.0, -1.0),
        // +Y
        vec3<f32>(-1.0,  1.0, -1.0), vec3<f32>( 1.0,  1.0, -1.0), vec3<f32>( 1.0,  1.0,  1.0),
        vec3<f32>( 1.0,  1.0,  1.0), vec3<f32>(-1.0,  1.0,  1.0), vec3<f32>(-1.0,  1.0, -1.0)
    );

    var out: VSOut;
    let dir = positions[idx];

    // Keep cube centered on the camera: project to far plane by forcing z = w.
    out.clip = vec4<f32>(dir.xy, 1.0, 1.0);
    out.v_dir = normalize(dir);
    return out;
}
