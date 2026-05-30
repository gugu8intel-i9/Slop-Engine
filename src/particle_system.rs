// src/particle_system.rs
// GPU/CPU particle emitters, curves, lifetime simulation

use std::collections::HashMap;
use glam::{Vec3, Vec4, Mat4};
use parking_lot::RwLock;
use rand::Rng;

/// Particle data structure
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub acceleration: [f32; 3],
    pub lifetime: f32,
    pub max_lifetime: f32,
    pub size: f32,
    pub rotation: f32,
    pub rotation_speed: f32,
    pub color: [f32; 4],
    pub pad: [f32; 3],
}

impl Default for Particle {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            velocity: [0.0; 3],
            acceleration: [0.0; 3],
            lifetime: 0.0,
            max_lifetime: 1.0,
            size: 1.0,
            rotation: 0.0,
            rotation_speed: 0.0,
            color: [1.0; 4],
            pad: [0.0; 3],
        }
    }
}

/// Curve for animating particle properties over time
#[derive(Clone, Debug)]
pub struct Curve {
    pub keys: Vec<CurveKey>,
}

#[derive(Clone, Debug)]
pub struct CurveKey {
    pub time: f32,
    pub value: f32,
    pub tangent_in: f32,
    pub tangent_out: f32,
}

impl Curve {
    pub fn new() -> Self {
        Self { keys: Vec::new() }
    }

    pub fn add_key(&mut self, time: f32, value: f32) {
        self.keys.push(CurveKey {
            time,
            value,
            tangent_in: 0.0,
            tangent_out: 0.0,
        });
    }

    pub fn evaluate(&self, t: f32) -> f32 {
        if self.keys.is_empty() {
            return 0.0;
        }

        if t <= self.keys[0].time {
            return self.keys[0].value;
        }

        if t >= self.keys.last().unwrap().time {
            return self.keys.last().unwrap().value;
        }

        for i in 0..self.keys.len() - 1 {
            let k1 = &self.keys[i];
            let k2 = &self.keys[i + 1];

            if t >= k1.time && t <= k2.time {
                let local_t = (t - k1.time) / (k2.time - k1.time);
                return self.lerp(k1.value, k2.value, local_t);
            }
        }

        0.0
    }

    fn lerp(&self, a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }
}

impl Default for Curve {
    fn default() -> Self {
        Self::new()
    }
}

/// Particle emission shape
#[derive(Clone, Copy, Debug)]
pub enum EmissionShape {
    Point,
    Sphere { radius: f32 },
    Box { size: Vec3 },
    Cone { angle: f32, height: f32 },
    Circle { radius: f32 },
}

/// Particle emitter configuration
#[derive(Clone, Debug)]
pub struct EmitterConfig {
    pub name: String,
    pub max_particles: u32,
    pub emission_rate: f32, // particles per second
    pub burst_count: u32,
    pub lifetime_min: f32,
    pub lifetime_max: f32,
    pub speed_min: f32,
    pub speed_max: f32,
    pub size_start: f32,
    pub size_end: f32,
    pub color_start: Vec4,
    pub color_end: Vec4,
    pub gravity: Vec3,
    pub drag: f32,
    pub emission_shape: EmissionShape,
    pub local_space: bool,
    pub looping: bool,
    pub prewarm: bool,
    pub size_curve: Option<Curve>,
    pub color_over_lifetime: Option<[Vec4; 3]>,
    pub speed_curve: Option<Curve>,
}

impl Default for EmitterConfig {
    fn default() -> Self {
        Self {
            name: "Emitter".to_string(),
            max_particles: 1000,
            emission_rate: 10.0,
            burst_count: 0,
            lifetime_min: 1.0,
            lifetime_max: 2.0,
            speed_min: 1.0,
            speed_max: 3.0,
            size_start: 0.5,
            size_end: 0.0,
            color_start: Vec4::ONE,
            color_end: Vec4::new(1.0, 1.0, 1.0, 0.0),
            gravity: Vec3::NEG_Y * 9.81,
            drag: 0.0,
            emission_shape: EmissionShape::Point,
            local_space: false,
            looping: true,
            prewarm: false,
            size_curve: None,
            color_over_lifetime: None,
            speed_curve: None,
        }
    }
}

/// Particle system instance
pub struct ParticleSystem {
    pub config: EmitterConfig,
    pub particles: Vec<Particle>,
    pub gpu_buffer: Option<wgpu::Buffer>,
    pub transform: Mat4,
    pub is_active: bool,
    pub time_since_emit: f32,
    pub total_time: f32,
    pub random_offset: f32,
}

impl ParticleSystem {
    pub fn new(device: &wgpu::Device, config: EmitterConfig) -> Self {
        let buffer_size = config.max_particles as u64 * std::mem::size_of::<Particle>() as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut rng = rand::thread_rng();

        Self {
            config,
            particles: Vec::with_capacity(config.max_particles as usize),
            gpu_buffer: Some(buffer),
            transform: Mat4::IDENTITY,
            is_active: true,
            time_since_emit: 0.0,
            total_time: 0.0,
            random_offset: rng.gen_range(0.0..1000.0),
        }
    }

    pub fn emit_particle(&mut self, spawn_pos: Vec3, spawn_vel: Vec3) {
        if self.particles.len() >= self.config.max_particles as usize {
            return;
        }

        let mut rng = rand::thread_rng();
        let lifetime = rng.gen_range(self.config.lifetime_min..=self.config.lifetime_max);

        let particle = Particle {
            position: spawn_pos.to_array(),
            velocity: spawn_vel.to_array(),
            acceleration: self.config.gravity.to_array(),
            lifetime: 0.0,
            max_lifetime: lifetime,
            size: self.config.size_start,
            rotation: rng.gen_range(0.0..360.0),
            rotation_speed: rng.gen_range(-180.0..180.0),
            color: self.config.color_start.to_array(),
            pad: [0.0; 3],
        };

        self.particles.push(particle);
    }

    pub fn spawn_position(&self) -> Vec3 {
        let mut rng = rand::thread_rng();

        match self.config.emission_shape {
            EmissionShape::Point => Vec3::ZERO,
            EmissionShape::Sphere { radius } => {
                let theta = rng.gen_range(0.0..360.0).to_radians();
                let phi = rng.gen_range(0.0..180.0).to_radians();
                let r = radius * rng.gen::<f32>().powf(1.0/3.0);
                Vec3::new(
                    r * phi.sin() * theta.cos(),
                    r * phi.sin() * theta.sin(),
                    r * phi.cos(),
                )
            }
            EmissionShape::Box { size } => {
                Vec3::new(
                    rng.gen_range(-size.x..size.x),
                    rng.gen_range(-size.y..size.y),
                    rng.gen_range(-size.z..size.z),
                )
            }
            EmissionShape::Cone { angle, height } => {
                let theta = rng.gen_range(0.0..360.0).to_radians();
                let h = rng.gen_range(0.0..height);
                let r = h * angle.tan() * rng.gen::<f32>().sqrt();
                Vec3::new(
                    r * theta.cos(),
                    -h,
                    r * theta.sin(),
                )
            }
            EmissionShape::Circle { radius } => {
                let theta = rng.gen_range(0.0..360.0).to_radians();
                let r = radius * rng.gen::<f32>().sqrt();
                Vec3::new(r * theta.cos(), 0.0, r * theta.sin())
            }
        }
    }

    pub fn spawn_velocity(&self, dir: Vec3) -> Vec3 {
        let mut rng = rand::thread_rng();
        let speed = rng.gen_range(self.config.speed_min..=self.config.speed_max);
        dir.normalize() * speed
    }

    pub fn update(&mut self, dt: f32, world_transform: Mat4) {
        if !self.is_active {
            return;
        }

        self.total_time += dt;
        self.transform = world_transform;

        // Emit new particles
        if self.config.emission_rate > 0.0 {
            self.time_since_emit += dt;
            let emit_interval = 1.0 / self.config.emission_rate;

            while self.time_since_emit >= emit_interval {
                self.time_since_emit -= emit_interval;

                if self.particles.len() < self.config.max_particles as usize {
                    let local_pos = self.spawn_position();
                    let local_dir = if self.config.emission_shape == EmissionShape::Point {
                        Vec3::Y
                    } else {
                        local_pos.normalize_or_zero()
                    };

                    let world_pos = world_transform * local_pos.extend(1.0);
                    let world_dir = world_transform * local_dir.extend(0.0);

                    self.emit_particle(world_pos.truncate(), self.spawn_velocity(world_dir.truncate()));
                }
            }
        }

        // Update existing particles
        let mut to_remove = Vec::new();
        let drag_factor = 1.0 - self.config.drag * dt;

        for (i, particle) in self.particles.iter_mut().enumerate() {
            particle.lifetime += dt;

            if particle.lifetime >= particle.max_lifetime {
                to_remove.push(i);
                continue;
            }

            // Apply forces
            particle.velocity[0] *= drag_factor;
            particle.velocity[1] *= drag_factor;
            particle.velocity[2] *= drag_factor;

            particle.velocity[0] += particle.acceleration[0] * dt;
            particle.velocity[1] += particle.acceleration[1] * dt;
            particle.velocity[2] += particle.acceleration[2] * dt;

            // Update position
            particle.position[0] += particle.velocity[0] * dt;
            particle.position[1] += particle.velocity[1] * dt;
            particle.position[2] += particle.velocity[2] * dt;

            // Update rotation
            particle.rotation += particle.rotation_speed * dt;

            // Update size from curve
            if let Some(ref curve) = self.config.size_curve {
                let normalized_time = particle.lifetime / particle.max_lifetime;
                particle.size = curve.evaluate(normalized_time);
            } else {
                let t = particle.lifetime / particle.max_lifetime;
                particle.size = self.config.size_start * (1.0 - t) + self.config.size_end * t;
            }

            // Update color
            if let Some(colors) = &self.config.color_over_lifetime {
                let t = particle.lifetime / particle.max_lifetime;
                let color = if t < 0.5 {
                    colors[0].lerp(colors[1], t * 2.0)
                } else {
                    colors[1].lerp(colors[2], (t - 0.5) * 2.0)
                };
                particle.color = color.to_array();
            } else {
                let t = particle.lifetime / particle.max_lifetime;
                particle.color = self.config.color_start.lerp(self.config.color_end, t).to_array();
            }
        }

        // Remove dead particles (swap with last for efficiency)
        for i in to_remove.into_iter().rev() {
            self.particles.swap_remove(i);
        }
    }

    pub fn upload_to_gpu(&self, queue: &wgpu::Queue) {
        if let Some(ref buffer) = self.gpu_buffer {
            queue.write_buffer(
                buffer,
                0,
                bytemuck::cast_slice(&self.particles),
            );
        }
    }

    pub fn clear(&mut self) {
        self.particles.clear();
    }

    pub fn stop(&mut self) {
        self.is_active = false;
    }

    pub fn play(&mut self) {
        self.is_active = true;
    }

    pub fn restart(&mut self) {
        self.clear();
        self.total_time = 0.0;
        self.time_since_emit = 0.0;
        self.is_active = true;
    }

    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }
}

/// Particle system manager
pub struct ParticleManager {
    pub systems: RwLock<HashMap<String, ParticleSystem>>,
    pub bind_group_layout: RwLock<Option<wgpu::BindGroupLayout>>,
    pub pipeline_layout: RwLock<Option<wgpu::PipelineLayout>>,
}

impl ParticleManager {
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Particle Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        Self {
            systems: RwLock::new(HashMap::new()),
            bind_group_layout: RwLock::new(Some(bind_layout)),
            pipeline_layout: RwLock::new(None),
        }
    }

    pub fn create_system(&self, name: &str, config: EmitterConfig, device: &wgpu::Device) {
        let system = ParticleSystem::new(device, config);
        self.systems.write().insert(name.to_string(), system);
    }

    pub fn get_system(&self, name: &str) -> Option<std::sync::MutexGuard<ParticleSystem>> {
        unimplemented!("Use direct access methods")
    }

    pub fn update_all(&self, dt: f32, transforms: &HashMap<String, Mat4>) {
        let mut systems = self.systems.write();
        
        for (name, system) in systems.iter_mut() {
            let transform = transforms.get(name).copied().unwrap_or(Mat4::IDENTITY);
            system.update(dt, transform);
        }
    }

    pub fn upload_all(&self, queue: &wgpu::Queue) {
        let systems = self.systems.read();
        
        for system in systems.values() {
            system.upload_to_gpu(queue);
        }
    }

    pub fn remove_system(&self, name: &str) {
        self.systems.write().remove(name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curve_evaluation() {
        let mut curve = Curve::new();
        curve.add_key(0.0, 0.0);
        curve.add_key(0.5, 1.0);
        curve.add_key(1.0, 0.0);

        assert!((curve.evaluate(0.0) - 0.0).abs() < 0.001);
        assert!((curve.evaluate(0.5) - 1.0).abs() < 0.001);
        assert!((curve.evaluate(1.0) - 0.0).abs() < 0.001);
        assert!((curve.evaluate(0.25) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_emission_shapes() {
        let config = EmitterConfig {
            emission_shape: EmissionShape::Sphere { radius: 5.0 },
            ..Default::default()
        };

        let mut system = ParticleSystem::new(
            &MockDevice,
            config,
        );

        // Test that spawn positions are within bounds
        for _ in 0..100 {
            let pos = system.spawn_position();
            assert!(pos.length() <= 5.0);
        }
    }

    // Mock device for testing
    struct MockDevice;
    impl MockDevice {
        fn create_buffer(&self, _: &wgpu::BufferDescriptor) -> wgpu::Buffer {
            unimplemented!()
        }
    }
}
