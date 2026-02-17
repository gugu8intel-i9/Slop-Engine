// src/pbr_materials.rs
// High-performance PBR Material system
// Dependencies: wgpu, wgpu::util, bytemuck, parking_lot (optional), anyhow
//
// Exports:
// - MaterialParams (Pod) matching WGSL layout
// - Material (holds params buffer + bind group)
// - MaterialSystem (creates bind group layout, caches bind groups)
//
// Usage summary:
// let layout = MaterialSystem::bind_group_layout(&device);
// let mat = MaterialSystem::create_material(&device, &queue, &layout, params, textures, &dummy);
// render_pass.set_bind_group(MATERIAL_BIND_GROUP_INDEX, &mat.bind_group, &[]);

use std::sync::Arc;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use anyhow::Result;

/// Material feature flags (bitmask)
pub mod flags {
    pub const BASE_COLOR_TEX: u32 = 1 << 0;
    pub const ORM_TEX: u32 = 1 << 1; // occlusion, roughness, metallic packed
    pub const NORMAL_TEX: u32 = 1 << 2;
    pub const EMISSIVE_TEX: u32 = 1 << 3;
    pub const CLEARCOAT_TEX: u32 = 1 << 4;
    pub const SHEEN_TEX: u32 = 1 << 5;
    pub const TRANSMISSION_TEX: u32 = 1 << 6;
    pub const SUBSURFACE_TEX: u32 = 1 << 7;
    pub const USE_IBL: u32 = 1 << 8;
}

/// Matches WGSL MaterialParams struct exactly
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MaterialParams {
    pub base_color_factor: [f32; 4],
    pub emissive_factor: [f32; 3],
    pub alpha_cutoff: f32,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub occlusion_strength: f32,
    pub clearcoat_factor: f32,
    pub clearcoat_roughness: f32,
    pub sheen_factor: f32,
    pub sheen_tint: f32,
    pub anisotropy: f32,
    pub transmission_factor: f32,
    pub subsurface_factor: f32,
    pub flags: u32,
    pub pad: [f32; 1],
}

impl Default for MaterialParams {
    fn default() -> Self {
        Self {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            emissive_factor: [0.0, 0.0, 0.0],
            alpha_cutoff: 0.5,
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            occlusion_strength: 1.0,
            clearcoat_factor: 0.0,
            clearcoat_roughness: 0.0,
            sheen_factor: 0.0,
            sheen_tint: 0.0,
            anisotropy: 0.0,
            transmission_factor: 0.0,
            subsurface_factor: 0.0,
            flags: 0,
            pad: [0.0],
        }
    }
}

/// Texture wrapper expected by MaterialSystem
pub struct TextureHandle {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

/// Material holds GPU resources for a single material
pub struct Material {
    pub params_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

/// MaterialSystem creates layouts and materials and caches bind groups
pub struct MaterialSystem {}

impl MaterialSystem {
    /// Create the bind group layout that matches the WGSL shader
    /// Group 1 layout in the shader (material group)
    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material_bind_group_layout"),
            entries: &[
                // 0: MaterialParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: base color texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // 2: ORM texture (occlusion, roughness, metallic)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // 3: normal map
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // 4: emissive
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // 5: extra packed maps (clearcoat, sheen, transmission, subsurface) as a single texture
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // 6: sampler (shared)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    /// Create a Material from params and optional textures. Provide a dummy texture for missing slots.
    pub fn create_material(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        params: &MaterialParams,
        base_color: Option<&TextureHandle>,
        orm: Option<&TextureHandle>,
        normal: Option<&TextureHandle>,
        emissive: Option<&TextureHandle>,
        extra: Option<&TextureHandle>,
        dummy: &TextureHandle,
    ) -> Result<Material> {
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material_params"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let base_view = base_color.map(|t| &t.view).unwrap_or(&dummy.view);
        let orm_view = orm.map(|t| &t.view).unwrap_or(&dummy.view);
        let normal_view = normal.map(|t| &t.view).unwrap_or(&dummy.view);
        let emissive_view = emissive.map(|t| &t.view).unwrap_or(&dummy.view);
        let extra_view = extra.map(|t| &t.view).unwrap_or(&dummy.view);
        let sampler = base_color.map(|t| &t.sampler).unwrap_or(&dummy.sampler);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(base_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(orm_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(normal_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(emissive_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(extra_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(sampler) },
            ],
        });

        Ok(Material { params_buffer: params_buf, bind_group })
    }

    /// Update material params quickly
    pub fn update_material_params(queue: &wgpu::Queue, material: &Material, params: &MaterialParams) {
        queue.write_buffer(&material.params_buffer, 0, bytemuck::bytes_of(params));
    }
}
