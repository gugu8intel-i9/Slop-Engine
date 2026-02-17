// src/gltf_loader.rs
// High-performance GLTF loader for wgpu
// Requires: gltf, image, bytemuck, wgpu, anyhow

use std::collections::HashMap;
use std::sync::Arc;
use anyhow::{Context, Result};
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

/// Reuse types from your mesh_material_system or duplicate minimal types here
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
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub ao_factor: f32,
    pub flags: u32,
    pub _pad: [f32; 3],
}

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

/// Scene container returned by loader
pub struct GltfScene {
    pub meshes: Vec<MeshGpu>,
    pub textures: Vec<Arc<TextureGpu>>,
    pub materials: Vec<MaterialGpu>,
    pub models: Vec<ModelGpu>,
    pub root_nodes: Vec<usize>, // indices into models
}

/// Loader struct with bind group layouts required by shader
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

    /// Load GLTF from bytes (supports .glb and .gltf with external buffers if you provide them)
    pub async fn load_from_bytes(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
    ) -> Result<GltfScene> {
        // Parse GLTF (auto-detect GLB)
        let (gltf, buffers, images) = parse_gltf_bytes(bytes)?;

        // Create a single dummy texture to use as fallback
        let dummy = create_dummy_texture(device, queue);

        // Load textures to GPU
        let mut texture_gpu_cache: Vec<Arc<TextureGpu>> = Vec::with_capacity(images.len());
        for img in images.iter() {
            let tex = upload_image_to_gpu(device, queue, img).context("upload image")?;
            texture_gpu_cache.push(Arc::new(tex));
        }
        // If no textures, keep at least one dummy
        if texture_gpu_cache.is_empty() {
            texture_gpu_cache.push(Arc::new(dummy));
        }

        // Build materials
        let mut materials_gpu: Vec<MaterialGpu> = Vec::with_capacity(gltf.materials().len());
        for mat in gltf.materials() {
            let mg = self.create_material_gpu(device, queue, &mat, &texture_gpu_cache, &dummy)?;
            materials_gpu.push(mg);
        }

        // Build meshes
        let mut meshes_gpu: Vec<MeshGpu> = Vec::with_capacity(gltf.meshes().len());
        for mesh in gltf.meshes() {
            for prim in mesh.primitives() {
                let mg = create_mesh_gpu(device, queue, &prim, &buffers)?;
                meshes_gpu.push(mg);
            }
        }

        // Build nodes/models
        let mut models_gpu: Vec<ModelGpu> = Vec::new();
        let mut node_to_model: HashMap<usize, usize> = HashMap::new();
        for (node_index, node) in gltf.nodes().enumerate() {
            // If node has a mesh, create a model for each primitive (we assume one primitive -> one mesh)
            if let Some(mesh) = node.mesh() {
                for prim in mesh.primitives() {
                    // find mesh index by primitive order: we created meshes in mesh->primitive order
                    // compute index: sum of previous mesh primitive counts
                    // For simplicity, map by name or by order: here we map by running index
                    // We'll use a simple approach: iterate meshes in same order as created above
                }
            }
            // Create model uniform from node transform
            let model_matrix = node_transform_to_matrix(&node);
            let model = ModelGpu::new(device, &self.model_layout, model_matrix);
            let model_idx = models_gpu.len();
            models_gpu.push(model);
            node_to_model.insert(node_index, model_idx);
        }

        // Root nodes
        let mut root_nodes = Vec::new();
        for scene in gltf.scenes() {
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
        queue: &wgpu::Queue,
        mat: &gltf::Material,
        textures: &[Arc<TextureGpu>],
        dummy: &TextureGpu,
    ) -> Result<MaterialGpu> {
        // Base color factor
        let pbr = mat.pbr_metallic_roughness();
        let base = pbr.base_color_factor();
        let base_color_factor = [base[0], base[1], base[2], base[3]];
        let metallic = pbr.metallic_factor();
        let roughness = pbr.roughness_factor();
        let ao = 1.0f32;

        // Flags and texture indices
        let mut flags: u32 = 0;
        let base_view = pbr.base_color_texture().map(|t| t.texture().index()).and_then(|i| textures.get(i)).map(|a| a.view.clone());
        if base_view.is_some() { flags |= 1; }
        let mr_view = pbr.metallic_roughness_texture().map(|t| t.texture().index()).and_then(|i| textures.get(i)).map(|a| a.view.clone());
        if mr_view.is_some() { flags |= 2; }
        let normal_view = mat.normal_texture().map(|t| t.texture().index()).and_then(|i| textures.get(i)).map(|a| a.view.clone());
        if normal_view.is_some() { flags |= 4; }
        let ao_view = mat.occlusion_texture().map(|t| t.texture().index()).and_then(|i| textures.get(i)).map(|a| a.view.clone());
        if ao_view.is_some() { flags |= 8; }

        // Create params buffer
        let params = MaterialParams {
            base_color_factor,
            metallic_factor: metallic,
            roughness_factor: roughness,
            ao_factor: ao,
            flags,
            _pad: [0.0; 3],
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gltf_material_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Build bind group entries using either real textures or dummy
        let base_tex_view = base_view.as_ref().map(|v| v).unwrap_or(&dummy.view);
        let mr_tex_view = mr_view.as_ref().map(|v| v).unwrap_or(&dummy.view);
        let normal_tex_view = normal_view.as_ref().map(|v| v).unwrap_or(&dummy.view);
        let ao_tex_view = ao_view.as_ref().map(|v| v).unwrap_or(&dummy.view);
        let sampler = &dummy.sampler;

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

/// Helper to create a dummy texture
fn create_dummy_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> TextureGpu {
    let rgba = [255u8, 255u8, 255u8, 255u8];
    let size = wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gltf_dummy"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        &rgba,
        wgpu::ImageDataLayout { offset: 0, bytes_per_row: NonZeroU32::new(4), rows_per_image: NonZeroU32::new(1) },
        size,
    );
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("gltf_dummy_sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });
    TextureGpu { view, sampler, _tex: tex }
}

/// Upload image bytes to GPU and return TextureGpu
fn upload_image_to_gpu(device: &wgpu::Device, queue: &wgpu::Queue, img: &gltf::image::Data) -> Result<TextureGpu> {
    // Decode image bytes using image crate
    let format = match img.format {
        gltf::image::Format::R8 => image::ColorType::L8,
        gltf::image::Format::R8G8 => image::ColorType::La8,
        gltf::image::Format::R8G8B8 => image::ColorType::Rgb8,
        gltf::image::Format::R8G8B8A8 => image::ColorType::Rgba8,
        _ => image::ColorType::Rgba8,
    };

    // image::load_from_memory expects full image bytes; gltf::image::Data may be raw or encoded
    // Try to decode using image::load_from_memory first
    let dyn_img = image::load_from_memory(&img.pixels)?;
    let rgba = dyn_img.to_rgba8();
    let (width, height) = rgba.dimensions();
    let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gltf_texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::ImageCopyTexture { texture: &texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        &rgba,
        wgpu::ImageDataLayout { offset: 0, bytes_per_row: NonZeroU32::new(4 * width), rows_per_image: NonZeroU32::new(height) },
        size,
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("gltf_sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    Ok(TextureGpu { view, sampler, _tex: texture })
}

/// Create GPU mesh from gltf primitive
fn create_mesh_gpu(device: &wgpu::Device, _queue: &wgpu::Queue, prim: &gltf::Primitive, buffers: &[gltf::buffer::Data]) -> Result<MeshGpu> {
    // Read positions, normals, uvs, tangents, indices
    let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));

    let positions: Vec<[f32; 3]> = reader.read_positions().context("positions missing")?.collect();
    let normals: Vec<[f32; 3]> = reader.read_normals().unwrap_or_else(|| vec![[0.0, 1.0, 0.0]].into_iter()).collect();
    let uvs: Vec<[f32; 2]> = reader.read_tex_coords(0).map(|t| t.into_f32().collect()).unwrap_or_else(|| vec![[0.0, 0.0]].into_iter()).collect();
    let tangents: Vec<[f32; 4]> = reader.read_tangents().map(|t| t.collect()).unwrap_or_else(|| vec![[1.0, 0.0, 0.0, 1.0]].into_iter()).collect();

    // Build interleaved vertices
    let vertex_count = positions.len();
    let mut vertices: Vec<Vertex> = Vec::with_capacity(vertex_count);
    for i in 0..vertex_count {
        let pos = positions[i];
        let n = normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]);
        let uv = uvs.get(i).copied().unwrap_or([0.0, 0.0]);
        let t4 = tangents.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 1.0]);
        let tangent = [t4[0], t4[1], t4[2]];
        vertices.push(Vertex { position: pos, normal: n, uv, tangent });
    }

    // Indices
    let indices: Vec<u32> = if let Some(iter) = reader.read_indices() {
        iter.into_u32().collect()
    } else {
        // non-indexed: build sequential indices
        (0u32..vertex_count as u32).collect()
    };

    // Create GPU buffers with DeviceExt::create_buffer_init for speed
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

    // Compute AABB
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for v in &vertices {
        for i in 0..3 {
            min[i] = min[i].min(v.position[i]);
            max[i] = max[i].max(v.position[i]);
        }
    }

    Ok(MeshGpu {
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
        aabb_min: min,
        aabb_max: max,
    })
}

/// Parse GLTF bytes into gltf::Gltf, buffer data, and image data
fn parse_gltf_bytes(bytes: &[u8]) -> Result<(gltf::Gltf, Vec<gltf::buffer::Data>, Vec<gltf::image::Data>)> {
    // Use gltf::import_slice to parse GLB or GLTF with embedded buffers
    let (gltf, buffers, images) = gltf::import_slice(bytes)?;
    Ok((gltf, buffers, images))
}

/// Convert node transform to 4x4 matrix
fn node_transform_to_matrix(node: &gltf::Node) -> [[f32; 4]; 4] {
    use glam::{Mat4, Vec3, Quat};
    if let Some(matrix) = node.transform().matrix() {
        // gltf returns [f32; 16] column-major
        let m = Mat4::from_cols_array_2d(&[
            [matrix[0], matrix[4], matrix[8], matrix[12]],
            [matrix[1], matrix[5], matrix[9], matrix[13]],
            [matrix[2], matrix[6], matrix[10], matrix[14]],
            [matrix[3], matrix[7], matrix[11], matrix[15]],
        ]);
        return m.to_cols_array_2d();
    }
    let (t, r, s) = node.transform().decomposed();
    let translation = Vec3::from_slice(&t);
    let rotation = Quat::from_array(r);
    let scale = Vec3::from_slice(&s);
    let m = Mat4::from_scale_rotation_translation(scale, rotation, translation);
    m.to_cols_array_2d()
}

/// ModelGpu constructor
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

/// Bind group layout helpers (same as earlier)
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
