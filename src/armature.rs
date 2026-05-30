// src/armature.rs
// Skeleton structure, bone transforms, skinning matrices

use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

/// A single bone in the skeleton hierarchy
#[derive(Clone, Debug)]
pub struct Bone {
    pub id: u32,
    pub name: String,
    pub parent_id: Option<u32>,
    pub children: Vec<u32>,
    
    // Local transform relative to parent
    pub local_position: Vec3,
    pub local_rotation: Quat,
    pub local_scale: Vec3,
    
    // Pre-computed matrices
    pub inverse_bind_pose: Mat4,
    pub world_transform: Mat4,
}

impl Bone {
    pub fn new(id: u32, name: String, parent_id: Option<u32>) -> Self {
        Self {
            id,
            name,
            parent_id,
            children: Vec::new(),
            local_position: Vec3::ZERO,
            local_rotation: Quat::IDENTITY,
            local_scale: Vec3::ONE,
            inverse_bind_pose: Mat4::IDENTITY,
            world_transform: Mat4::IDENTITY,
        }
    }

    pub fn local_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            self.local_scale,
            self.local_rotation,
            self.local_position,
        )
    }

    pub fn set_local_transform(&mut self, position: Vec3, rotation: Quat, scale: Vec3) {
        self.local_position = position;
        self.local_rotation = rotation;
        self.local_scale = scale;
    }
}

/// Complete skeleton/armature structure
#[derive(Clone, Debug)]
pub struct Skeleton {
    pub bones: HashMap<u32, Bone>,
    pub root_bone_id: Option<u32>,
    pub bone_names: HashMap<String, u32>,
    pub bind_pose_matrices: Vec<Mat4>,
}

impl Skeleton {
    pub fn new() -> Self {
        Self {
            bones: HashMap::new(),
            root_bone_id: None,
            bone_names: HashMap::new(),
            bind_pose_matrices: Vec::new(),
        }
    }

    /// Add a bone to the skeleton
    pub fn add_bone(&mut self, mut bone: Bone) -> u32 {
        let id = bone.id;
        let name = bone.name.clone();

        if let Some(parent_id) = bone.parent_id {
            if let Some(parent) = self.bones.get_mut(&parent_id) {
                parent.children.push(id);
            }
        } else {
            self.root_bone_id = Some(id);
        }

        self.bone_names.insert(name, id);
        self.bones.insert(id, bone);

        id
    }

    /// Build the skeleton hierarchy and compute bind poses
    pub fn build(&mut self) {
        self.bind_pose_matrices.clear();
        self.bind_pose_matrices.resize(self.bones.len(), Mat4::IDENTITY);

        // Compute bind pose for each bone
        if let Some(root_id) = self.root_bone_id {
            self.compute_bind_pose_recursive(root_id, Mat4::IDENTITY);
        }

        // Store inverse bind poses
        for (idx, bone) in self.bones.values().enumerate() {
            if idx < self.bind_pose_matrices.len() {
                self.bind_pose_matrices[idx] = bone.inverse_bind_pose;
            }
        }
    }

    fn compute_bind_pose_recursive(&mut self, bone_id: u32, parent_world: Mat4) {
        let bone = self.bones.get(&bone_id).cloned().unwrap();
        let local_matrix = bone.local_matrix();
        let world_matrix = parent_world * local_matrix;

        if let Some(bone_mut) = self.bones.get_mut(&bone_id) {
            bone_mut.world_transform = world_matrix;
            bone_mut.inverse_bind_pose = world_matrix.inverse();
        }

        let children = bone.children.clone();
        for child_id in children {
            self.compute_bind_pose_recursive(child_id, world_matrix);
        }
    }

    /// Update bone transforms based on animation poses
    pub fn update_poses(&mut self, animated_matrices: &[Mat4]) {
        if animated_matrices.len() != self.bones.len() {
            log::warn!("Animated matrix count mismatch: {} vs {}", 
                animated_matrices.len(), self.bones.len());
            return;
        }

        // Apply animated transforms
        for (idx, bone_id) in self.bones.keys().enumerate() {
            if idx < animated_matrices.len() {
                if let Some(bone) = self.bones.get_mut(bone_id) {
                    bone.world_transform = animated_matrices[idx];
                }
            }
        }

        // Recompute world transforms hierarchically
        if let Some(root_id) = self.root_bone_id {
            self.update_world_transforms_recursive(root_id, Mat4::IDENTITY);
        }
    }

    fn update_world_transforms_recursive(&mut self, bone_id: u32, parent_world: Mat4) {
        let bone = self.bones.get(&bone_id).cloned().unwrap();
        let local_matrix = bone.local_matrix();
        let world_matrix = parent_world * local_matrix;

        if let Some(bone_mut) = self.bones.get_mut(&bone_id) {
            bone_mut.world_transform = world_matrix;
        }

        let children = bone.children.clone();
        for child_id in children {
            self.update_world_transforms_recursive(child_id, world_matrix);
        }
    }

    /// Get the skinning matrices for GPU upload
    pub fn get_skinning_matrices(&self) -> Vec<Mat4> {
        let mut matrices = Vec::with_capacity(self.bones.len());

        for bone in self.bones.values() {
            let skinning_matrix = bone.world_transform * bone.inverse_bind_pose;
            matrices.push(skinning_matrix);
        }

        matrices
    }

    /// Find bone by name
    pub fn find_bone(&self, name: &str) -> Option<&Bone> {
        self.bone_names.get(name).and_then(|id| self.bones.get(id))
    }

    pub fn find_bone_mut(&mut self, name: &str) -> Option<&mut Bone> {
        self.bone_names.get(name).and_then(|id| {
            let bone_id = *id;
            self.bones.get_mut(&bone_id)
        })
    }

    pub fn bone_count(&self) -> usize {
        self.bones.len()
    }
}

/// GPU-ready skinning matrix data
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSkinningMatrix {
    pub matrix: [[f32; 4]; 4],
}

impl From<Mat4> for GpuSkinningMatrix {
    fn from(mat: Mat4) -> Self {
        Self {
            matrix: mat.to_cols_array_2d(),
        }
    }
}

/// Skinned mesh renderer component
#[derive(Clone)]
pub struct SkinnedMesh {
    pub skeleton: Arc<RwLock<Skeleton>>,
    pub vertex_buffer: Option<wgpu::Buffer>,
    pub index_buffer: Option<wgpu::Buffer>,
    pub skinning_buffer: Option<wgpu::Buffer>,
    pub vertex_count: u32,
    pub index_count: u32,
    pub max_bone_influence: u32,
}

impl SkinnedMesh {
    pub fn new(
        device: &wgpu::Device,
        skeleton: Arc<RwLock<Skeleton>>,
        vertices: &[SkinnedVertex],
        indices: &[u32],
        max_bone_influence: u32,
    ) -> Self {
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Skinned Mesh Vertex Buffer"),
            size: std::mem::size_of::<SkinnedVertex>() as u64 * vertices.len() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Skinned Mesh Index Buffer"),
            size: std::mem::size_of::<u32>() as u64 * indices.len() as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bone_count = skeleton.read().bone_count();
        let skinning_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Skinned Mesh Bone Buffer"),
            size: std::mem::size_of::<GpuSkinningMatrix>() as u64 * bone_count as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            skeleton,
            vertex_buffer: Some(vertex_buffer),
            index_buffer: Some(index_buffer),
            skinning_buffer: Some(skinning_buffer),
            vertex_count: vertices.len() as u32,
            index_count: indices.len() as u32,
            max_bone_influence,
        }
    }

    pub fn update_vertices(&self, queue: &wgpu::Queue, vertices: &[SkinnedVertex]) {
        if let Some(ref buffer) = self.vertex_buffer {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(vertices));
        }
    }

    pub fn update_indices(&self, queue: &wgpu::Queue, indices: &[u32]) {
        if let Some(ref buffer) = self.index_buffer {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(indices));
        }
    }

    pub fn update_bones(&self, queue: &wgpu::Queue, matrices: &[Mat4]) {
        if let Some(ref buffer) = self.skinning_buffer {
            let gpu_matrices: Vec<GpuSkinningMatrix> = matrices.iter()
                .map(|m| GpuSkinningMatrix::from(*m))
                .collect();
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&gpu_matrices));
        }
    }
}

/// Vertex with skinning data for GPU processing
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkinnedVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 4],
    pub uv: [f32; 2],
    pub bone_indices: [u32; 4],
    pub bone_weights: [f32; 4],
}

impl SkinnedVertex {
    pub fn new(
        position: Vec3,
        normal: Vec3,
        tangent: Vec3,
        uv: glam::Vec2,
        bone_indices: [u32; 4],
        bone_weights: [f32; 4],
    ) -> Self {
        Self {
            position: position.to_array(),
            normal: normal.to_array(),
            tangent: [tangent.x, tangent.y, tangent.z, 1.0],
            uv: uv.to_array(),
            bone_indices,
            bone_weights,
        }
    }

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SkinnedVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as u64,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as u64,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 10]>() as u64,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 12]>() as u64,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Uint32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[u32; 4]>() as u64 + std::mem::size_of::<[f32; 12]>() as u64,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

/// LOD (Level of Detail) for skinned meshes
#[derive(Clone)]
pub struct SkinnedMeshLod {
    pub lod_levels: Vec<SkinnedMesh>,
    pub lod_distances: Vec<f32>,
    pub current_lod: usize,
}

impl SkinnedMeshLod {
    pub fn new(lod_levels: Vec<SkinnedMesh>, lod_distances: Vec<f32>) -> Self {
        Self {
            lod_levels,
            lod_distances,
            current_lod: 0,
        }
    }

    pub fn update_lod(&mut self, camera_distance: f32) {
        for (i, &distance) in self.lod_distances.iter().enumerate() {
            if camera_distance <= distance {
                self.current_lod = i;
                return;
            }
        }
        self.current_lod = self.lod_levels.len() - 1;
    }

    pub fn current_mesh(&self) -> &SkinnedMesh {
        &self.lod_levels[self.current_lod]
    }
}

/// Armature system for managing all skeletons in the scene
pub struct ArmatureSystem {
    pub skeletons: RwLock<HashMap<u32, Arc<RwLock<Skeleton>>>>,
    pub skinned_meshes: RwLock<HashMap<u32, SkinnedMesh>>,
    pub default_bind_layout: RwLock<Option<wgpu::BindGroupLayout>>,
}

impl ArmatureSystem {
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Skeleton Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(std::num::NonZeroU64::new(64).unwrap()),
                    },
                    count: None,
                },
            ],
        });

        Self {
            skeletons: RwLock::new(HashMap::new()),
            skinned_meshes: RwLock::new(HashMap::new()),
            default_bind_layout: RwLock::new(Some(bind_layout)),
        }
    }

    pub fn create_skeleton(&self, entity_id: u32, skeleton: Skeleton) -> Arc<RwLock<Skeleton>> {
        let arc = Arc::new(RwLock::new(skeleton));
        self.skeletons.write().insert(entity_id, arc.clone());
        arc
    }

    pub fn get_skeleton(&self, entity_id: u32) -> Option<Arc<RwLock<Skeleton>>> {
        self.skeletons.read().get(&entity_id).cloned()
    }

    pub fn remove_skeleton(&self, entity_id: u32) {
        self.skeletons.write().remove(&entity_id);
    }

    pub fn add_skinned_mesh(&self, entity_id: u32, mesh: SkinnedMesh) {
        self.skinned_meshes.write().insert(entity_id, mesh);
    }

    pub fn update_all(&self, queue: &wgpu::Queue) {
        let skeletons = self.skeletons.read();
        let mut meshes = self.skinned_meshes.write();

        for mesh in meshes.values_mut() {
            let skeleton = mesh.skeleton.read();
            let matrices = skeleton.get_skinning_matrices();
            mesh.update_bones(queue, &matrices);
        }
    }

    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        mesh: &SkinnedMesh,
        bone_offset: u32,
    ) -> Option<wgpu::BindGroup> {
        let layout = self.default_bind_layout.read();
        layout.as_ref().and_then(|l| {
            mesh.skinning_buffer.as_ref().map(|buffer| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Skeleton Bind Group"),
                    layout: l,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: (bone_offset * std::mem::size_of::<GpuSkinningMatrix>() as u32) as u64,
                            size: Some(std::num::NonZeroU64::new(
                                std::mem::size_of::<GpuSkinningMatrix>() as u64 * 64
                            ).unwrap()),
                        }),
                    }],
                })
            })
        })
    }
}

impl Default for Skeleton {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bone_hierarchy() {
        let mut skeleton = Skeleton::new();

        let root = Bone::new(0, "root".to_string(), None);
        let child1 = Bone::new(1, "child1".to_string(), Some(0));
        let child2 = Bone::new(2, "child2".to_string(), Some(0));
        let grandchild = Bone::new(3, "grandchild".to_string(), Some(1));

        skeleton.add_bone(root);
        skeleton.add_bone(child1);
        skeleton.add_bone(child2);
        skeleton.add_bone(grandchild);
        skeleton.build();

        assert_eq!(skeleton.bone_count(), 4);
        assert!(skeleton.find_bone("root").is_some());
        assert!(skeleton.find_bone("grandchild").is_some());
    }

    #[test]
    fn test_skinning_matrices() {
        let mut skeleton = Skeleton::new();
        let root = Bone::new(0, "root".to_string(), None);
        skeleton.add_bone(root);
        skeleton.build();

        let matrices = skeleton.get_skinning_matrices();
        assert_eq!(matrices.len(), 1);
        assert_eq!(matrices[0], Mat4::IDENTITY);
    }
}
