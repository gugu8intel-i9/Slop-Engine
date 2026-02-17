// Vertex shader for skybox cube
struct VSOut {
    @builtin(position) clip: vec4<f32>;
    @location(0) v_dir: vec3<f32>;
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VSOut {
    // Generate a cube by vertex index (36 vertices)
    var positions = array<vec3<f32>, 36>(
        vec3(-1.0, -1.0, -1.0), vec3(1.0, -1.0, -1.0), vec3(1.0, 1.0, -1.0),
        vec3(1.0, 1.0, -1.0), vec3(-1.0, 1.0, -1.0), vec3(-1.0, -1.0, -1.0),
        // ... remaining faces omitted for brevity; include full 36 positions in real file
        vec3(-1.0, -1.0, 1.0), vec3(1.0, -1.0, 1.0), vec3(1.0, 1.0, 1.0),
        vec3(1.0, 1.0, 1.0), vec3(-1.0, 1.0, 1.0), vec3(-1.0, -1.0, 1.0),
        // etc.
    );
    var out: VSOut;
    let pos = positions[idx];
    // Expand to far plane by using clip w = 1 and projecting with identity view (skybox rendered in view space)
    out.clip = vec4<f32>(pos, 1.0);
    out.v_dir = pos;
    return out;
}
