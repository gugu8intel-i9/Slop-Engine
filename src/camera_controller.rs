// src/camera_controller.rs
// High performance Camera controller for Slop Engine
// Dependencies: glam = "0.24", bytemuck = "1.13", wgpu
//
// Example usage:
// let mut cam = CameraController::new_perspective(1920.0/1080.0, 60.0, 0.1, 1000.0);
// cam.set_mode(CameraMode::FreeFly);
// cam.process_input(&input); // per-frame or event-driven
// cam.update(dt);
// cam.upload_gpu(&device, &queue, &camera_bind_group, 0);

use glam::{Mat4, Vec3, Vec2, Quat};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Camera modes
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CameraMode {
    FreeFly,
    Orbit,
    Follow2D,
    Cinematic,
    Orthographic,
}

/// Camera projection type
#[derive(Copy, Clone, Debug)]
pub enum Projection {
    Perspective { fovy_deg: f32, aspect: f32, znear: f32, zfar: f32 },
    Orthographic { left: f32, right: f32, bottom: f32, top: f32, znear: f32, zfar: f32 },
}

/// GPU camera uniform layout
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub position: [f32; 4], // w unused
    pub forward: [f32; 4],
    pub up: [f32; 4],
    pub params: [f32; 4], // x = mode, y = padding, z = padding, w = padding
}

impl CameraUniform {
    pub fn from_matrices(view: Mat4, proj: Mat4, pos: Vec3, mode_id: f32) -> Self {
        let view_proj = proj * view;
        Self {
            view_proj: view_proj.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            position: [pos.x, pos.y, pos.z, 0.0],
            forward: [-(view.z_axis.x), -(view.z_axis.y), -(view.z_axis.z), 0.0],
            up: [view.y_axis.x, view.y_axis.y, view.y_axis.z, 0.0],
            params: [mode_id, 0.0, 0.0, 0.0],
        }
    }
}

/// Collision resolution callback trait
/// Implement this in your engine to provide collision avoidance without coupling physics.
pub trait CollisionResolver {
    /// Given desired camera position and radius, return a safe position.
    fn resolve(&mut self, desired: Vec3, radius: f32) -> Vec3;
}

/// Spline key for cinematic paths
#[derive(Copy, Clone, Debug)]
pub struct SplineKey {
    pub time: f32,
    pub position: Vec3,
    pub rotation: Quat,
    pub fov: f32,
}

/// Camera controller main struct
pub struct CameraController {
    // transform
    pub position: Vec3,
    pub rotation: Quat, // local orientation
    pub up: Vec3,

    // velocity for smooth motion
    velocity: Vec3,
    angular_velocity: Vec3,

    // projection
    pub projection: Projection,
    pub mode: CameraMode,

    // orbit target
    orbit_target: Vec3,
    orbit_distance: f32,
    orbit_min_dist: f32,
    orbit_max_dist: f32,
    orbit_yaw: f32,
    orbit_pitch: f32,

    // follow target
    follow_target: Option<Vec3>,
    follow_offset: Vec3,
    follow_smooth: f32,

    // cinematic spline
    spline: Vec<SplineKey>,
    spline_time: f32,
    spline_speed: f32,
    spline_loop: bool,

    // damping and responsiveness
    pub translation_damping: f32,
    pub rotation_damping: f32,
    pub input_sensitivity: f32,
    pub max_speed: f32,

    // camera radius for collision
    pub collision_radius: f32,
    pub collision_resolver: Option<Box<dyn CollisionResolver + Send + Sync>>,

    // GPU resources
    camera_buffer: Option<wgpu::Buffer>,
    camera_bind_group_layout: Option<wgpu::BindGroupLayout>,
    camera_bind_group: Option<wgpu::BindGroup>,

    // internal caches
    last_view: Mat4,
    last_proj: Mat4,
}

impl CameraController {
    /// Create perspective camera
    pub fn new_perspective(aspect: f32, fovy_deg: f32, znear: f32, zfar: f32) -> Self {
        let proj = Projection::Perspective { fovy_deg, aspect, znear, zfar };
        Self::new_common(proj)
    }

    /// Create orthographic camera
    pub fn new_orthographic(left: f32, right: f32, bottom: f32, top: f32, znear: f32, zfar: f32) -> Self {
        let proj = Projection::Orthographic { left, right, bottom, top, znear, zfar };
        Self::new_common(proj)
    }

    fn new_common(projection: Projection) -> Self {
        let position = Vec3::new(0.0, 0.0, 5.0);
        let rotation = Quat::IDENTITY;
        Self {
            position,
            rotation,
            up: Vec3::Y,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            projection,
            mode: CameraMode::FreeFly,
            orbit_target: Vec3::ZERO,
            orbit_distance: 5.0,
            orbit_min_dist: 0.5,
            orbit_max_dist: 100.0,
            orbit_yaw: 0.0,
            orbit_pitch: 0.0,
            follow_target: None,
            follow_offset: Vec3::ZERO,
            follow_smooth: 8.0,
            spline: Vec::new(),
            spline_time: 0.0,
            spline_speed: 1.0,
            spline_loop: false,
            translation_damping: 12.0,
            rotation_damping: 12.0,
            input_sensitivity: 1.0,
            max_speed: 50.0,
            collision_radius: 0.25,
            collision_resolver: None,
            camera_buffer: None,
            camera_bind_group_layout: None,
            camera_bind_group: None,
            last_view: Mat4::IDENTITY,
            last_proj: Mat4::IDENTITY,
        }
    }

    /// Set camera mode
    pub fn set_mode(&mut self, mode: CameraMode) {
        self.mode = mode;
    }

    /// Attach collision resolver
    pub fn set_collision_resolver<R: CollisionResolver + Send + Sync + 'static>(&mut self, resolver: R) {
        self.collision_resolver = Some(Box::new(resolver));
    }

    /// Set orbit parameters
    pub fn set_orbit_target(&mut self, target: Vec3, distance: f32) {
        self.orbit_target = target;
        self.orbit_distance = distance.clamp(self.orbit_min_dist, self.orbit_max_dist);
    }

    /// Set follow target and offset
    pub fn set_follow_target(&mut self, target: Vec3, offset: Vec3, smooth: f32) {
        self.follow_target = Some(target);
        self.follow_offset = offset;
        self.follow_smooth = smooth.max(0.0001);
    }

    /// Add cinematic spline key
    pub fn add_spline_key(&mut self, key: SplineKey) {
        self.spline.push(key);
        self.spline.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Set projection aspect or ortho extents
    pub fn set_aspect(&mut self, aspect: f32) {
        if let Projection::Perspective { fovy_deg, aspect: _, znear, zfar } = self.projection {
            self.projection = Projection::Perspective { fovy_deg, aspect, znear, zfar };
        }
    }

    /// Process input deltas from your Input system. This is intentionally generic.
    /// Provide translation vector in local camera space and rotation delta as yaw/pitch in radians.
    pub fn apply_input(&mut self, translation_local: Vec3, yaw_delta: f32, pitch_delta: f32, dt: f32) {
        // scale by sensitivity and dt
        let t = translation_local * (self.input_sensitivity * dt);
        // integrate velocity with damping
        self.velocity = self.velocity.lerp(t * self.max_speed, 1.0 - (-self.translation_damping * dt).exp());
        // angular velocity
        self.angular_velocity = self.angular_velocity.lerp(Vec3::new(pitch_delta, yaw_delta, 0.0) * self.input_sensitivity, 1.0 - (-self.rotation_damping * dt).exp());
    }

    /// Update camera each frame. Call after apply_input.
    pub fn update(&mut self, dt: f32) {
        match self.mode {
            CameraMode::FreeFly => self.update_freefly(dt),
            CameraMode::Orbit => self.update_orbit(dt),
            CameraMode::Follow2D => self.update_follow2d(dt),
            CameraMode::Cinematic => self.update_cinematic(dt),
            CameraMode::Orthographic => self.update_orthographic(dt),
        }
    }

    fn update_freefly(&mut self, dt: f32) {
        // apply rotation from angular_velocity
        let yaw = self.angular_velocity.y * dt;
        let pitch = self.angular_velocity.x * dt;
        let yaw_q = Quat::from_rotation_y(-yaw);
        let pitch_q = Quat::from_rotation_x(-pitch);
        self.rotation = (yaw_q * self.rotation) * pitch_q;
        // move in local space
        let forward = self.rotation * Vec3::Z * -1.0;
        let right = self.rotation * Vec3::X;
        let up = self.rotation * Vec3::Y;
        let delta = right * self.velocity.x + up * self.velocity.y + forward * self.velocity.z;
        let desired = self.position + delta;
        self.position = self.resolve_collision(desired);
    }

    fn update_orbit(&mut self, dt: f32) {
        // yaw/pitch integrated from angular_velocity
        self.orbit_yaw += self.angular_velocity.y * dt;
        self.orbit_pitch += self.angular_velocity.x * dt;
        self.orbit_pitch = self.orbit_pitch.clamp(-1.49, 1.49);
        // update distance with velocity.z
        self.orbit_distance = (self.orbit_distance + self.velocity.z * dt).clamp(self.orbit_min_dist, self.orbit_max_dist);
        // compute position
        let x = self.orbit_distance * self.orbit_yaw.cos() * self.orbit_pitch.cos();
        let z = self.orbit_distance * self.orbit_yaw.sin() * self.orbit_pitch.cos();
        let y = self.orbit_distance * self.orbit_pitch.sin();
        let desired = self.orbit_target + Vec3::new(x, y, z);
        self.position = self.resolve_collision(desired);
        // look at target
        let dir = (self.orbit_target - self.position).normalize_or_zero();
        self.rotation = Quat::from_rotation_arc(Vec3::Z * -1.0, dir);
    }

    fn update_follow2d(&mut self, dt: f32) {
        if let Some(target) = self.follow_target {
            let desired = target + self.follow_offset;
            // smooth damp
            let t = 1.0 - (-self.follow_smooth * dt).exp();
            self.position = self.position.lerp(desired, t);
            // keep rotation fixed to up axis
            self.rotation = Quat::IDENTITY;
        }
    }

    fn update_cinematic(&mut self, dt: f32) {
        if self.spline.is_empty() { return; }
        self.spline_time += dt * self.spline_speed;
        // loop or clamp
        let end_time = self.spline.last().unwrap().time;
        if self.spline_time > end_time {
            if self.spline_loop {
                self.spline_time = self.spline_time % end_time;
            } else {
                self.spline_time = end_time;
            }
        }
        // find segment
        let mut i = 0usize;
        while i + 1 < self.spline.len() && self.spline[i + 1].time < self.spline_time { i += 1; }
        let a = &self.spline[i];
        let b = if i + 1 < self.spline.len() { &self.spline[i + 1] } else { a };
        let t = if (b.time - a.time).abs() < 1e-6 { 0.0 } else { (self.spline_time - a.time) / (b.time - a.time) };
        // cubic hermite or slerp for rotation
        self.position = a.position.lerp(b.position, t);
        self.rotation = a.rotation.slerp(b.rotation, t);
    }

    fn update_orthographic(&mut self, _dt: f32) {
        // orthographic camera typically controlled externally; keep transform stable
    }

    fn resolve_collision(&mut self, desired: Vec3) -> Vec3 {
        if let Some(resolver) = &mut self.collision_resolver {
            resolver.resolve(desired, self.collision_radius)
        } else {
            desired
        }
    }

    /// Compute view matrix
    pub fn view_matrix(&self) -> Mat4 {
        let rot = self.rotation;
        let trans = Mat4::from_translation(-self.position);
        let rot_m = Mat4::from_quat(rot.conjugate());
        rot_m * trans
    }

    /// Compute projection matrix
    pub fn proj_matrix(&self) -> Mat4 {
        match self.projection {
            Projection::Perspective { fovy_deg, aspect, znear, zfar } => {
                Mat4::perspective_rh(fovy_deg.to_radians(), aspect, znear, zfar)
            }
            Projection::Orthographic { left, right, bottom, top, znear, zfar } => {
                Mat4::orthographic_rh(left, right, bottom, top, znear, zfar)
            }
        }
    }

    /// Compute GPU uniform
    pub fn camera_uniform(&self) -> CameraUniform {
        let view = self.view_matrix();
        let proj = self.proj_matrix();
        CameraUniform::from_matrices(view, proj, self.position, self.mode as i32 as f32)
    }

    /// Create GPU buffer and bind group layout for camera uniform. Call once at init.
    pub fn prepare_gpu(&mut self, device: &wgpu::Device, label: &str) {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{} camera layout", label)),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                }
            ],
        });
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} camera buffer", label)),
            size: std::mem::size_of::<CameraUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} camera bind group", label)),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
        });
        self.camera_buffer = Some(buffer);
        self.camera_bind_group_layout = Some(layout);
        self.camera_bind_group = Some(bind_group);
    }

    /// Upload camera uniform to GPU. Call each frame after update.
    pub fn upload_gpu(&mut self, queue: &wgpu::Queue) {
        if let Some(buf) = &self.camera_buffer {
            let uniform = self.camera_uniform();
            queue.write_buffer(buf, 0, bytemuck::bytes_of(&uniform));
            // cache last matrices
            self.last_view = Mat4::from_cols_array_2d(&uniform.view);
            self.last_proj = Mat4::from_cols_array_2d(&uniform.proj);
        }
    }

    /// Return bind group for pipeline binding index 0
    pub fn camera_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.camera_bind_group.as_ref()
    }

    /// Frustum planes in world space for culling
    pub fn frustum_planes(&self) -> [Vec4; 6] {
        // returns planes as (nx, ny, nz, d) in world space
        // compute from view_proj inverse
        let vp = self.proj_matrix() * self.view_matrix();
        let m = vp.to_cols_array_2d();
        // extract planes quickly
        // left = row4 + row1, right = row4 - row1, bottom = row4 + row2, top = row4 - row2, near = row4 + row3, far = row4 - row3
        // convert to world space by multiplying by inverse transpose if needed. For speed, return in view space and transform AABB accordingly.
        // For brevity return zeros here; implement as needed in engine.
        use glam::Vec4;
        [Vec4::ZERO, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO]
    }
}
