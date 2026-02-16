// src/camera.rs
use std::borrow::Cow;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

/// Perspective camera with position and Euler rotation (yaw, pitch).
pub struct Camera {
    pub position: Vec3,
    /// yaw: rotation around Y axis (radians). pitch: rotation around X axis (radians).
    pub yaw: f32,
    pub pitch: f32,

    pub fovy: f32,
    pub aspect: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    /// Create a new perspective camera.
    pub fn new(position: Vec3, yaw: f32, pitch: f32, fovy_radians: f32, aspect: f32, znear: f32, zfar: f32) -> Self {
        Self {
            position,
            yaw,
            pitch,
            fovy: fovy_radians,
            aspect,
            znear,
            zfar,
        }
    }

    /// Build view matrix from position + yaw/pitch (right-handed, Y up).
    pub fn view_matrix(&self) -> Mat4 {
        // Compute forward vector from yaw/pitch
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();

        let forward = Vec3::new(
            cos_pitch * sin_yaw,
            sin_pitch,
            cos_pitch * cos_yaw,
        )
        .normalize_or_zero();

        let target = self.position + forward;
        Mat4::look_at_rh(self.position, target, Vec3::Y)
    }

    /// Build projection matrix (perspective).
    pub fn proj_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar)
    }

    /// Combined view-projection matrix.
    pub fn view_proj_matrix(&self) -> Mat4 {
        self.proj_matrix() * self.view_matrix()
    }

    /// Update aspect ratio (call on resize).
    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
    }

    /// Convenience: set position.
    pub fn set_position(&mut self, pos: Vec3) {
        self.position = pos;
    }

    /// Convenience: set yaw/pitch.
    pub fn set_rotation(&mut self, yaw: f32, pitch: f32) {
        self.yaw = yaw;
        self.pitch = pitch;
    }
}

/// GPU camera uniform (matches shader layout).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform {
    /// Column-major 4x4 matrix
    pub view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
        }
    }

    pub fn from_matrix(m: Mat4) -> Self {
        Self {
            view_proj: m.to_cols_array_2d(),
        }
    }

    pub fn update_from_camera(&mut self, camera: &Camera) {
        self.view_proj = camera.view_proj_matrix().to_cols_array_2d();
    }
}

/// Small camera controller. Input-agnostic: call `process_keyboard` and `process_mouse` from your input layer.
pub struct CameraController {
    pub speed: f32,
    pub sensitivity: f32,

    // movement state
    pub forward: f32,
    pub right: f32,
    pub up: f32,

    // mouse deltas
    pub yaw_delta: f32,
    pub pitch_delta: f32,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            speed,
            sensitivity,
            forward: 0.0,
            right: 0.0,
            up: 0.0,
            yaw_delta: 0.0,
            pitch_delta: 0.0,
        }
    }

    /// Call when keyboard input changes. `fwd`, `right`, `up` are -1.0..1.0 values.
    pub fn process_keyboard(&mut self, fwd: f32, right: f32, up: f32) {
        self.forward = fwd;
        self.right = right;
        self.up = up;
    }

    /// Call when mouse moves. `dx`, `dy` are pixel deltas (or normalized deltas).
    pub fn process_mouse(&mut self, dx: f32, dy: f32) {
        self.yaw_delta += dx * self.sensitivity;
        self.pitch_delta += dy * self.sensitivity;
    }

    /// Apply controller to camera. `dt` is seconds since last update.
    pub fn update_camera(&mut self, camera: &mut Camera, dt: f32) {
        // Update rotation
        camera.yaw += self.yaw_delta;
        camera.pitch += self.pitch_delta;

        // clamp pitch to avoid gimbal flip
        let max_pitch = std::f32::consts::FRAC_PI_2 - 0.01;
        if camera.pitch > max_pitch {
            camera.pitch = max_pitch;
        } else if camera.pitch < -max_pitch {
            camera.pitch = -max_pitch;
        }

        // Reset deltas
        self.yaw_delta = 0.0;
        self.pitch_delta = 0.0;

        // Movement in camera space
        let (sin_yaw, cos_yaw) = camera.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = camera.pitch.sin_cos();

        // forward vector (same as Camera::view_matrix forward)
        let forward = Vec3::new(
            cos_pitch * sin_yaw,
            sin_pitch,
            cos_pitch * cos_yaw,
        )
        .normalize_or_zero();

        // right vector
        let right_vec = forward.cross(Vec3::Y).normalize_or_zero();

        // up vector (world up)
        let up_vec = Vec3::Y;

        let mut displacement = Vec3::ZERO;
        displacement += forward * (self.forward * self.speed * dt);
        displacement += right_vec * (self.right * self.speed * dt);
        displacement += up_vec * (self.up * self.speed * dt);

        camera.position += displacement;
    }
}

/// Create camera GPU resources: buffer, bind group layout, bind group.
/// Returns (camera_buffer, camera_bind_group, camera_bind_group_layout).
pub fn create_camera_gpu_resources(
    device: &wgpu::Device,
    initial_camera: &Camera,
) -> (wgpu::Buffer, wgpu::BindGroup, wgpu::BindGroupLayout) {
    let uniform = CameraUniform::from_matrix(initial_camera.view_proj_matrix());
    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("camera_buffer"),
        contents: bytemuck::cast_slice(&[uniform]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("camera_bind_group_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("camera_bind_group"),
        layout: &camera_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: camera_buffer.as_entire_binding(),
        }],
    });

    (camera_buffer, camera_bg, camera_bgl)
}

/// Update the GPU camera buffer from a Camera instance.
pub fn write_camera_buffer(queue: &wgpu::Queue, camera_buffer: &wgpu::Buffer, camera: &Camera) {
    let mut uniform = CameraUniform::new();
    uniform.update_from_camera(camera);
    queue.write_buffer(camera_buffer, 0, bytemuck::cast_slice(&[uniform]));
}

/// Optional helper: a minimal WGSL snippet for shaders that expect the camera uniform.
/// Use `@group(0) @binding(0) var<uniform> camera: Camera;` in your shader.
pub const CAMERA_WGSL: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>;
};
@group(0) @binding(0) var<uniform> camera: Camera;
"#;
