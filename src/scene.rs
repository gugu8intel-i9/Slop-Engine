// src/scene.rs
// Hybrid Scene Graph + ECS optimized for performance.
// Dependencies: glam, wgpu, bytemuck
// Add to Cargo.toml: glam = "0.22", bytemuck = "1.13"

use std::collections::VecDeque;
use std::sync::Arc;
use glam::{Mat4, Vec3, Quat};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Maximums tuned for performance; adjust to your needs.
pub const MAX_ENTITIES: usize = 65536;
pub const MAX_INSTANCES: usize = 16384;

/// Entity id (index + generation)
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Entity(u32);

impl Entity {
    fn index(self) -> usize { (self.0 & 0xFFFF_FFFF) as usize }
    fn from_index(i: usize) -> Self { Entity(i as u32) }
}

/// Component bitflags for quick archetype checks
bitflags::bitflags! {
    pub struct CompMask: u32 {
        const TRANSFORM = 1 << 0;
        const SCENE_NODE = 1 << 1;
        const MESH = 1 << 2;
        const MATERIAL = 1 << 3;
        const INSTANCE = 1 << 4;
        const CAMERA = 1 << 5;
        const LIGHT = 1 << 6;
        const SKIN = 1 << 7;
        // add more as needed
    }
}

/// SparseSet storage for components (SoA friendly)
pub struct SparseSet<T: Copy + Pod> {
    dense: Vec<T>,
    dense_entities: Vec<u32>,
    sparse: Vec<i32>, // -1 = empty, else index into dense
    capacity: usize,
}

impl<T: Copy + Pod> SparseSet<T> {
    pub fn with_capacity(capacity: usize, default: T) -> Self {
        let mut sparse = vec![-1i32; capacity];
        Self {
            dense: Vec::with_capacity(capacity),
            dense_entities: Vec::with_capacity(capacity),
            sparse,
            capacity,
        }
    }

    pub fn insert(&mut self, ent: u32, value: T) {
        let idx = ent as usize;
        if idx >= self.capacity { panic!("SparseSet index overflow"); }
        if self.sparse[idx] != -1 {
            let di = self.sparse[idx] as usize;
            self.dense[di] = value;
            return;
        }
        let di = self.dense.len();
        self.dense.push(value);
        self.dense_entities.push(ent);
        self.sparse[idx] = di as i32;
    }

    pub fn remove(&mut self, ent: u32) {
        let idx = ent as usize;
        if idx >= self.capacity { return; }
        let si = self.sparse[idx];
        if si == -1 { return; }
        let di = si as usize;
        let last = self.dense.len() - 1;
        self.dense.swap_remove(di);
        let moved_ent = self.dense_entities.swap_remove(di);
        if di < self.dense.len() {
            self.sparse[moved_ent as usize] = di as i32;
        }
        self.sparse[idx] = -1;
    }

    pub fn get(&self, ent: u32) -> Option<&T> {
        let idx = ent as usize;
        if idx >= self.capacity { return None; }
        let si = self.sparse[idx];
        if si == -1 { return None; }
        Some(&self.dense[si as usize])
    }

    pub fn get_mut(&mut self, ent: u32) -> Option<&mut T> {
        let idx = ent as usize;
        if idx >= self.capacity { return None; }
        let si = self.sparse[idx];
        if si == -1 { return None; }
        Some(&mut self.dense[si as usize])
    }

    pub fn iter(&self) -> impl Iterator<Item = (u32, &T)> {
        self.dense_entities.iter().copied().zip(self.dense.iter())
    }
}

/// Transform component (SoA friendly)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Transform {
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub local_matrix: [[f32; 4]; 4],
    pub world_matrix: [[f32; 4]; 4],
    pub dirty: u32,
}

impl Transform {
    pub fn identity() -> Self {
        Self {
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            local_matrix: Mat4::IDENTITY.to_cols_array_2d(),
            world_matrix: Mat4::IDENTITY.to_cols_array_2d(),
            dirty: 1,
        }
    }

    pub fn update_local(&mut self) {
        let t = Vec3::from(self.translation);
        let r = Quat::from_array(self.rotation);
        let s = Vec3::from(self.scale);
        let m = Mat4::from_scale_rotation_translation(s, r, t);
        self.local_matrix = m.to_cols_array_2d();
        self.dirty = 1;
    }
}

/// Scene node for hierarchy and culling
pub struct SceneNode {
    pub parent: Option<u32>,
    pub first_child: Option<u32>,
    pub next_sibling: Option<u32>,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub world_aabb_min: [f32; 3],
    pub world_aabb_max: [f32; 3],
    pub dirty: bool,
}

impl SceneNode {
    pub fn new() -> Self {
        Self {
            parent: None,
            first_child: None,
            next_sibling: None,
            aabb_min: [f32::INFINITY; 3],
            aabb_max: [f32::NEG_INFINITY; 3],
            world_aabb_min: [f32::INFINITY; 3],
            world_aabb_max: [f32::NEG_INFINITY; 3],
            dirty: true,
        }
    }
}

/// MeshInstance component referencing Mesh and Material indices
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MeshInstance {
    pub mesh_index: u32,
    pub material_index: u32,
    pub visible: u32,
    pub pad: u32,
}

/// Camera component
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CameraComp {
    pub view_proj: [[f32; 4]; 4],
    pub pos: [f32; 3],
    pub mode: u32, // 0=2D,1=3D
}

/// Light component (small)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct LightComp {
    pub position: [f32; 3],
    pub radius: f32,
    pub color: [f32; 3],
    pub intensity: f32,
    pub light_type: u32,
    pub pad: [u32; 3],
}

/// Scene container
pub struct Scene {
    // entity pool
    free_list: VecDeque<u32>,
    pub entity_mask: Vec<CompMask>,

    // component stores
    pub transforms: SparseSet<Transform>,
    pub nodes: SparseSet<SceneNode>,
    pub mesh_instances: SparseSet<MeshInstance>,
    pub cameras: SparseSet<CameraComp>,
    pub lights: SparseSet<LightComp>,

    // resource references (indices into your Mesh/Material arrays)
    pub meshes: Vec<usize>,
    pub materials: Vec<usize>,

    // instance buffer for GPU instancing (per-frame updated)
    pub instance_data: Vec<[[f32; 4]; 4]>, // world matrices
    pub instance_buffer: Option<wgpu::Buffer>,

    // device/queue references for buffer updates
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    // capacity
    capacity: usize,
}

impl Scene {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, capacity: usize) -> Self {
        let mut free_list = VecDeque::with_capacity(capacity);
        for i in (0..capacity).rev() { free_list.push_back(i as u32); }
        Self {
            free_list,
            entity_mask: vec![CompMask::empty(); capacity],
            transforms: SparseSet::with_capacity(capacity, Transform::identity()),
            nodes: SparseSet::with_capacity(capacity, SceneNode::new()),
            mesh_instances: SparseSet::with_capacity(capacity, MeshInstance { mesh_index: 0, material_index: 0, visible: 1, pad: 0 }),
            cameras: SparseSet::with_capacity(capacity, CameraComp { view_proj: Mat4::IDENTITY.to_cols_array_2d(), pos: [0.0;3], mode: 1 }),
            lights: SparseSet::with_capacity(capacity, LightComp { position: [0.0;3], radius: 1.0, color: [1.0;3], intensity: 1.0, light_type: 1, pad: [0;3] }),
            meshes: Vec::new(),
            materials: Vec::new(),
            instance_data: Vec::with_capacity(MAX_INSTANCES),
            instance_buffer: None,
            device,
            queue,
            capacity,
        }
    }

    /// Spawn a new entity id
    pub fn spawn(&mut self) -> Entity {
        let idx = self.free_list.pop_front().expect("Entity pool exhausted");
        self.entity_mask[idx as usize] = CompMask::empty();
        Entity::from_index(idx as usize)
    }

    /// Despawn entity and remove components
    pub fn despawn(&mut self, e: Entity) {
        let i = e.index();
        if i >= self.capacity { return; }
        // remove components
        self.transforms.remove(i as u32);
        self.nodes.remove(i as u32);
        self.mesh_instances.remove(i as u32);
        self.cameras.remove(i as u32);
        self.lights.remove(i as u32);
        self.entity_mask[i] = CompMask::empty();
        self.free_list.push_back(i as u32);
    }

    /// Attach transform component
    pub fn add_transform(&mut self, e: Entity, t: Transform) {
        let i = e.index() as u32;
        self.transforms.insert(i, t);
        self.entity_mask[e.index()] |= CompMask::TRANSFORM;
    }

    /// Attach scene node
    pub fn add_node(&mut self, e: Entity) {
        let i = e.index() as u32;
        self.nodes.insert(i, SceneNode::new());
        self.entity_mask[e.index()] |= CompMask::SCENE_NODE;
    }

    /// Attach mesh instance
    pub fn add_mesh_instance(&mut self, e: Entity, mesh_index: u32, material_index: u32) {
        let i = e.index() as u32;
        self.mesh_instances.insert(i, MeshInstance { mesh_index, material_index, visible: 1, pad: 0 });
        self.entity_mask[e.index()] |= CompMask::MESH | CompMask::MATERIAL;
    }

    /// Attach camera
    pub fn add_camera(&mut self, e: Entity, cam: CameraComp) {
        let i = e.index() as u32;
        self.cameras.insert(i, cam);
        self.entity_mask[e.index()] |= CompMask::CAMERA;
    }

    /// Attach light
    pub fn add_light(&mut self, e: Entity, light: LightComp) {
        let i = e.index() as u32;
        self.lights.insert(i, light);
        self.entity_mask[e.index()] |= CompMask::LIGHT;
    }

    /// Update local matrices for dirty transforms
    pub fn update_transforms(&mut self) {
        // Update local matrices
        for (ent, t) in self.transforms.iter() {
            if t.dirty != 0 {
                // recompute local matrix
                let mut tmut = *t;
                let translation = Vec3::from(tmut.translation);
                let rotation = Quat::from_array(tmut.rotation);
                let scale = Vec3::from(tmut.scale);
                let m = Mat4::from_scale_rotation_translation(scale, rotation, translation);
                tmut.local_matrix = m.to_cols_array_2d();
                tmut.dirty = 0;
                // write back
                self.transforms.insert(ent, tmut);
            }
        }

        // Propagate world matrices using scene graph
        // For performance, iterate nodes and update world matrices only when needed.
        // Simple approach: for each node with no parent, traverse children.
        let mut stack: Vec<u32> = Vec::new();
        for (ent, node) in self.nodes.iter() {
            if node.parent.is_none() {
                stack.push(ent);
                while let Some(cur) = stack.pop() {
                    // compute world matrix
                    let mut world = Mat4::IDENTITY;
                    if let Some(t) = self.transforms.get(cur) {
                        world = Mat4::from_cols_array_2d(&t.local_matrix);
                    }
                    if let Some(parent) = self.nodes.get(cur).and_then(|n| n.parent) {
                        if let Some(pt) = self.transforms.get(parent) {
                            let pworld = Mat4::from_cols_array_2d(&pt.world_matrix);
                            world = pworld * world;
                        }
                    }
                    // write world matrix into transform
                    if let Some(mut tmut) = self.transforms.get_mut(cur) {
                        tmut.world_matrix = world.to_cols_array_2d();
                    }
                    // push children
                    if let Some(n) = self.nodes.get(cur) {
                        let mut child = n.first_child;
                        while let Some(c) = child {
                            stack.push(c);
                            child = self.nodes.get(c).and_then(|nc| nc.next_sibling);
                        }
                    }
                }
            }
        }
    }

    /// Simple frustum culling and instance collection
    /// camera_view_proj is camera.view_proj matrix
    pub fn cull_and_collect(&mut self, camera_view_proj: Mat4) {
        self.instance_data.clear();
        // For each mesh instance, test AABB in world space and collect visible instances
        for (ent, inst) in self.mesh_instances.iter() {
            if inst.visible == 0 { continue; }
            // get transform world matrix
            if let Some(t) = self.transforms.get(ent) {
                let world = Mat4::from_cols_array_2d(&t.world_matrix);
                // For now, assume mesh AABB is unit cube; in production use mesh AABB from Mesh
                // Transform AABB center and extents conservatively by max scale
                // Quick conservative frustum test: transform origin and test clip space
                let pos = world.transform_point3(Vec3::ZERO);
                let clip = camera_view_proj * world * Vec3::ZERO.extend(1.0);
                // simple depth check
                if clip.z < -1.0 || clip.z > 1.0 {
                    continue;
                }
                // collect instance matrix
                self.instance_data.push(world.to_cols_array_2d());
            }
        }
    }

    /// Upload instance buffer to GPU (call once per frame after cull_and_collect)
    pub fn upload_instances(&mut self) {
        if self.instance_data.is_empty() { return; }
        let bytes = bytemuck::cast_slice(&self.instance_data);
        if let Some(buf) = &self.instance_buffer {
            self.queue.write_buffer(buf, 0, bytes);
        } else {
            let buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("instance_buffer"),
                contents: bytes,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
            self.instance_buffer = Some(buf);
        }
    }

    /// Draw collected instances using provided draw callback
    /// draw_cb is a closure that receives (mesh_index, material_index, instance_range)
    pub fn draw_collected<F: FnMut(usize, usize, std::ops::Range<u32>)>(&self, mut draw_cb: F) {
        // For simplicity, we assume all collected instances are for the same mesh/material in this example.
        // In production, group instances by mesh/material.
        if self.instance_data.is_empty() { return; }
        let instance_count = self.instance_data.len() as u32;
        // Example: call draw callback with mesh 0, material 0
        draw_cb(0, 0, 0..instance_count);
    }
}
