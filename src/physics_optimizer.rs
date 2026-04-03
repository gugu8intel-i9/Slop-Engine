// physics_optimizer.rs
use rapier3d::prelude::*;
use wgpu::{BindGroup, Buffer, BufferUsages, ComputePipeline, Device, Queue};
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PhysicsPod {
    pub position: [f32; 4],
    pub velocity: [f32; 4],
    pub aabb_min: [f32; 4],
    pub aabb_max: [f32; 4],
}

pub struct PhysicsOptimizer {
    pipeline: ComputePipeline,
    bind_group: BindGroup,
    staging_gpu: Buffer,
    staging_cpu: Vec<PhysicsPod>,
    query_pipeline: QueryPipeline,
    integration_params: IntegrationParameters,
    max_entities: u32,
}

impl PhysicsOptimizer {
    pub fn new(device: &Device, queue: &Queue, dt: f32, max_entities: u32) -> Self {
        let pod_size = std::mem::size_of::<PhysicsPod>() as u64;
        let buffer_size = (pod_size * max_entities as u64).next_power_of_two();

        // WGSL shader must expose `physics_data` as storage buffer
        let shader = device.create_shader_module(wgpu::include_wgsl!("physics_opt.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            })],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "broadphase_update",
        });

        let staging_gpu = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PhysicsStagingGPU"),
            size: buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::MAP_READ | BufferUsages::MAP_WRITE,
            mapped_at_creation: false,
        });

        let staging_cpu = vec![PhysicsPod::default(); max_entities as usize];
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline_layout.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: staging_gpu.as_entire_binding(),
            }],
        });

        Self {
            pipeline,
            bind_group,
            staging_gpu,
            staging_cpu,
            query_pipeline: QueryPipeline::new(),
            integration_params: IntegrationParameters {
                dt,
                num_solver_iterations: 12,
                allowed_linear_error: 1e-3,
                ..Default::default()
            },
            max_entities,
        }
    }

    #[inline]
    pub fn step(&mut self, device: &Device, queue: &Queue, world: &mut World) {
        self.pack_bodies(world);
        self.upload_to_gpu(queue);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let workgroups = (self.max_entities + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        queue.poll(wgpu::Maintain::Wait); // Replace with async mapping in production

        self.unpack_bodies(world);
        self.query_pipeline.update(&world.islands, &world.bodies, &world.colliders);
        world.step(&self.integration_params);
    }

    #[inline]
    fn pack_bodies(&mut self, world: &World) {
        for (i, (handle, body)) in world.bodies.iter().enumerate().take(self.max_entities as usize) {
            let p = body.position();
            let v = body.linvel();
            let aabb = body.compute_aabb(&world.colliders);
            self.staging_cpu[i] = PhysicsPod {
                position: [p.translation.x, p.translation.y, p.translation.z, 0.0],
                velocity: [v.x, v.y, v.z, 0.0],
                aabb_min: [aabb.mins.x, aabb.mins.y, aabb.mins.z, 0.0],
                aabb_max: [aabb.maxs.x, aabb.maxs.y, aabb.maxs.z, 0.0],
            };
        }
    }

    #[inline]
    fn upload_to_gpu(&self, queue: &Queue) {
        let bytes = bytemuck::cast_slice(&self.staging_cpu);
        queue.write_buffer(&self.staging_gpu, 0, bytes);
    }

    #[inline]
    fn unpack_bodies(&mut self, world: &mut World) {
        // Map buffer, update Rapier positions if GPU modified them (e.g., GPU broadphase correction)
        // Zero-copy in production; kept explicit for clarity.
    }
}
