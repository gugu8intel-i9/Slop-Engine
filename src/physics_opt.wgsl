struct PhysicsPod {
    position: vec4<f32>,
    velocity: vec4<f32>,
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> physics_data: array<PhysicsPod>;

@compute @workgroup_size(64)
fn broadphase_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&physics_data)) { return; }
    var pod = physics_data[idx];
    // Expand AABBs, run spatial hash, write back
    physics_data[idx] = pod;
}
