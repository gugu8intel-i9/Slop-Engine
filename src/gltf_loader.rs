// src/gltf_loader.rs
// High-performance GLTF loader for wgpu
// Requires: gltf, image, bytemuck, wgpu, anyhow

use std::collections::HashMap;
use std::sync::Arc;
use anyhow::{Context, Result};
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub tangent: [f32; 3],
}

pub struct MeshGpu {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
}

pub struct TextureGpu {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    // keep texture alive
    _tex: wgpu::Texture,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MaterialParams {
    pub base_color_factor: [f32; 4], // 16 bytes
    pub metallic_factor: f32,        // 4 bytes
    pub roughness_factor: f32,       // 4 bytes
    pub ao_factor: f32,              // 4 bytes
    pub flags: u32,                  // 4 bytes
} // Total: 32 bytes (Perfectly aligned for WGSL, no padding needed)

pub struct MaterialGpu {
    pub params_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ModelUniform {
    pub model: [[f32; 4]; 4],
}

pub struct ModelGpu {
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

pub struct GltfScene {
    pub meshes: Vec<MeshGpu>,
    pub textures: Vec<Arc<TextureGpu>>,
    pub materials: Vec<MaterialGpu>,
    pub models: Vec<ModelGpu>,
    pub root_nodes: Vec<usize>,
}

pub struct GltfLoader {
    pub camera_layout: wgpu::BindGroupLayout,
    pub material_layout: wgpu::BindGroupLayout,
    pub model_layout: wgpu::BindGroupLayout,
}

impl GltfLoader {
    pub fn new(device: &wgpu::Device) -> Self {
        let camera_layout = create_camera_bind_group_layout(device);
        let material_layout = create_material_bind_group_layout(device);
        let model_layout = create_model_bind_group_layout(device);
        Self { camera_layout, material_layout, model_layout }
    }

    pub async fn load_from_bytes(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
    ) -> Result<GltfScene> {
        let (document, buffers, images) = parse_gltf_bytes(bytes)?;

        let dummy = create_dummy_texture(device, queue);

        let mut texture_gpu_cache: Vec<Arc<TextureGpu>> = Vec::with_capacity(images.len());
        for img in images.iter() {
            let tex = upload_image_to_gpu(device, queue, img).context("upload image")?;
            texture_gpu_cache.push(Arc::new(tex));
        }
        
        if texture_gpu_cache.is_empty() {
            texture_gpu_cache.push(Arc::new(dummy));
        }

        // Dummy texture mapped safely without cloning a TextureView
        let dummy_view = &texture_gpu_cache[0].view;

        let mut materials_gpu: Vec<MaterialGpu> = Vec::with_capacity(document.materials().len());
        for mat in document.materials() {
            let mg = self.create_material_gpu(device, &mat, &texture_gpu_cache, dummy_view)?;
            materials_gpu.push(mg);
        }

        let mut meshes_gpu: Vec<MeshGpu> = Vec::with_capacity(document.meshes().len());
        for mesh in document.meshes() {
            for prim in mesh.primitives() {
                let mg = create_mesh_gpu(device, &prim, &buffers)?;
                meshes_gpu.push(mg);
            }
        }

        let mut models_gpu: Vec<ModelGpu> = Vec::new();
        let mut node_to_model: HashMap<usize, usize> = HashMap::new();
        
        for (node_index, node) in document.nodes().enumerate() {
            let model_matrix = node_transform_to_matrix(&node);
            let model = ModelGpu::new(device, &self.model_layout, model_matrix);
            let model_idx = models_gpu.len();
            models_gpu.push(model);
            node_to_model.insert(node_index, model_idx);
        }

        let mut root_nodes = Vec::new();
        for scene in document.scenes() {
            for node in scene.nodes() {
                if let Some(&midx) = node_to_model.get(&node.index()) {
                    root_nodes.push(midx);
                }
            }
        }

        Ok(GltfScene {
            meshes: meshes_gpu,
            textures: texture_gpu_cache,
            materials: materials_gpu,
            models: models_gpu,
            root_nodes,
        })
    }

    fn create_material_gpu(
        &self,
        device: &wgpu::Device,
        mat: &gltf::Material,
        textures: &[Arc<TextureGpu>],
        dummy_view: &wgpu::TextureView,
    ) -> Result<MaterialGpu> {
        let pbr = mat.pbr_metallic_roughness();
        let base = pbr.base_color_factor();
        
        let mut flags: u32 = 0;
        if pbr.base_color_texture().is_some() { flags |= 1; }
        if pbr.metallic_roughness_texture().is_some() { flags |= 2; }
        if mat.normal_texture().is_some() { flags |= 4; }
        if mat.occlusion_texture().is_some() { flags |= 8; }

        let params = MaterialParams {
            base_color_factor: [base[0], base[1], base[2], base[3]],
            metallic_factor: pbr.metallic_factor(),
            roughness_factor: pbr.roughness_factor(),
            ao_factor: 1.0,
            flags,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gltf_material_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Safe View retrieval - avoiding illegal wgpu::TextureView cloning!
        let get_view = |tex: Option<gltf::texture::Info>| {
            tex.and_then(|t| textures.get(t.texture().index()))
               .map(|t| &t.view)
               .unwrap_or(dummy_view)
        };

        let base_tex_view = get_view(pbr.base_color_texture());
        let mr_tex_view = get_view(pbr.metallic_roughness_texture());
        let normal_tex_view = get_view(mat.normal_texture().map(|t| gltf::texture::Info::from(t)));
        let ao_tex_view = get_view(mat.occlusion_texture().map(|t| gltf::texture::Info::from(t)));
        
        let sampler = &textures[0].sampler;

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gltf_material_bind_group"),
            layout: &self.material_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(base_tex_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(mr_tex_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(normal_tex_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(ao_tex_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(sampler) },
            ],
        });

        Ok(MaterialGpu { params_buffer, bind_group })
    }
}

fn create_dummy_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> TextureGpu {
    let rgba = [255u8, 255u8, 255u8, 255u8];
    let size = wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 };
    
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gltf_dummy"),
        size, mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::ImageCopyTexture { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        &rgba,
        wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
        size,
    );

    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("gltf_dummy_sampler"),
        address_mode_u: wgpu::AddressMode::Repeat, address_mode_v: wgpu::AddressMode::Repeat, address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    TextureGpu { view, sampler, _tex: tex }
}

fn upload_image_to_gpu(device: &wgpu::Device, queue: &wgpu::Queue, img: &gltf::image::Data) -> Result<TextureGpu> {
    let width = img.width;
    let height = img.height;

    // Fix: GLTF images are ALREADY raw decoded pixels, so `image::load_from_memory()` will fail.
    // Instead, map the raw pixel arrays directly and convert RGB-24bit to RGBA-32bit for wgpu compatibility!
    let rgba_pixels: Vec<u8> = match img.format {
        gltf::image::Format::R8G8B8 => {
            let mut rgba = Vec::with_capacity((width * height * 4) as usize);
            for chunk in img.pixels.chunks_exact(3) {
                rgba.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
            }
            rgba
        },
        gltf::image::Format::R8G8B8A8 => img.pixels.clone(),
        gltf::image::Format::R8 => {
            let mut rgba = Vec::with_capacity((width * height * 4) as usize);
            for &p in &img.pixels {
                rgba.extend_from_slice(&[p, p, p, 255]);
            }
            rgba
        },
        gltf::image::Format::R8G8 => {
            let mut rgba = Vec::with_capacity((width * height * 4) as usize);
            for chunk in img.pixels.chunks_exact(2) {
                rgba.extend_from_slice(&[chunk[0], chunk[0], chunk[0], chunk[1]]);
            }
            rgba
        },
        _ => img.pixels.clone(),
    };

    let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gltf_texture"),
        size, mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::ImageCopyTexture { texture: &texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        &rgba_pixels,
        wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(4 * width), rows_per_image: Some(height) },
        size,
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("gltf_sampler"),
        address_mode_u: wgpu::AddressMode::Repeat, address_mode_v: wgpu::AddressMode::Repeat, address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    Ok(TextureGpu { view, sampler, _tex: texture })
}

fn create_mesh_gpu(device: &wgpu::Device, prim: &gltf::Primitive, buffers: &[gltf::buffer::Data]) -> Result<MeshGpu> {
    let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));

    let positions: Vec<[f32; 3]> = reader.read_positions().context("positions missing")?.collect();
    let vertex_count = positions.len();

    // Fix: Properly map the option iterator into a Vec before defaulting it!
    let normals: Vec<[f32; 3]> = reader.read_normals()
        .map(|iter| iter.collect())
        .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; vertex_count]);

    let uvs: Vec<[f32; 2]> = reader.read_tex_coords(0)
        .map(|iter| iter.into_f32().collect())
        .unwrap_or_else(|| vec![[0.0, 0.0]; vertex_count]);

    let tangents: Vec<[f32; 4]> = reader.read_tangents()
        .map(|iter| iter.collect())
        .unwrap_or_else(|| vec![[1.0, 0.0, 0.0, 1.0]; vertex_count]);

    let mut vertices: Vec<Vertex> = Vec::with_capacity(vertex_count);
    for i in 0..vertex_count {
        vertices.push(Vertex {
            position: positions[i],
            normal: normals[i],
            uv: uvs[i],
            tangent: [tangents[i][0], tangents[i][1], tangents[i][2]],
        });
    }

    let indices: Vec<u32> = if let Some(iter) = reader.read_indices() {
        iter.into_u32().collect()
    } else {
        (0u32..vertex_count as u32).collect()
    };

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gltf_vertex_buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gltf_index_buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for v in &vertices {
        for i in 0..3 {
            min[i] = min[i].min(v.position[i]);
            max[i] = max[i].max(v.position[i]);
        }
    }

    Ok(MeshGpu { vertex_buffer, index_buffer, index_count: indices.len() as u32, aabb_min: min, aabb_max: max })
}

fn parse_gltf_bytes(bytes: &[u8]) -> Result<(gltf::Document, Vec<gltf::buffer::Data>, Vec<gltf::image::Data>)> {
    let (document, buffers, images) = gltf::import_slice(bytes)?;
    Ok((document, buffers, images))
}

fn node_transform_to_matrix(node: &gltf::Node) -> [[f32; 4]; 4] {
    // Fix: `gltf` crate natively outputs a 4x4 nested array for transforms natively. No glam required.
    node.transform().matrix()
}

impl ModelGpu {
    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, model_matrix: [[f32; 4]; 4]) -> Self {
        let uniform = ModelUniform { model: model_matrix };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gltf_model_uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gltf_model_bind_group"),
            layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() }],
        });
        Self { uniform_buffer, bind_group }
    }

    pub fn update(&self, queue: &wgpu::Queue, model_matrix: [[f32; 4]; 4]) {
        let uniform = ModelUniform { model: model_matrix };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    }
}

pub fn create_material_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gltf_material_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
        ],
    })
}

pub fn create_camera_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gltf_camera_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    })
}

pub fn create_model_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gltf_model_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    })
}
