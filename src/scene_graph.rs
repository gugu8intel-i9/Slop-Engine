// src/scene_graph.rs
// Parent/child transforms, hierarchical updates

use std::collections::HashMap;
use glam::{Mat4, Quat, Vec3};
use parking_lot::RwLock;
use smallvec::SmallVec;

/// Transform component for scene graph nodes
#[derive(Clone, Debug)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    pub local_matrix: Mat4,
    pub world_matrix: Mat4,
    pub dirty: bool,
}

impl Default for Transform {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform {
    pub fn new() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            local_matrix: Mat4::IDENTITY,
            world_matrix: Mat4::IDENTITY,
            dirty: true,
        }
    }

    pub fn from_matrix(matrix: Mat4) -> Self {
        let (scale, rotation, position) = matrix.to_scale_rotation_translation();
        Self {
            position,
            rotation,
            scale,
            local_matrix: matrix,
            world_matrix: matrix,
            dirty: false,
        }
    }

    pub fn compute_local_matrix(&mut self) -> Mat4 {
        self.local_matrix = Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position);
        self.dirty = true;
        self.local_matrix
    }

    pub fn set_position(&mut self, pos: Vec3) {
        self.position = pos;
        self.dirty = true;
    }

    pub fn set_rotation(&mut self, rot: Quat) {
        self.rotation = rot;
        self.dirty = true;
    }

    pub fn set_scale(&mut self, scale: Vec3) {
        self.scale = scale;
        self.dirty = true;
    }

    pub fn translate(&mut self, delta: Vec3) {
        self.position += delta;
        self.dirty = true;
    }

    pub fn rotate(&mut self, axis: Vec3, angle: f32) {
        let q = Quat::from_axis_angle(axis.normalize(), angle);
        self.rotation = q * self.rotation;
        self.dirty = true;
    }

    pub fn look_at(&mut self, target: Vec3, up: Vec3) {
        let direction = (target - self.position).normalize();
        self.rotation = Quat::from_mat4(&Mat4::look_to_rh(self.position, direction, up));
        self.dirty = true;
    }

    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Z
    }

    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    pub fn up(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }
}

/// Scene graph node
#[derive(Clone, Debug)]
pub struct SceneNode {
    pub id: u64,
    pub name: String,
    pub parent: Option<u64>,
    pub children: SmallVec<[u64; 4]>,
    pub transform: Transform,
    pub visible: bool,
    pub tags: Vec<String>,
    pub components: HashMap<String, Box<dyn Send + Sync>>,
}

impl SceneNode {
    pub fn new(id: u64, name: String) -> Self {
        Self {
            id,
            name,
            parent: None,
            children: SmallVec::new(),
            transform: Transform::new(),
            visible: true,
            tags: Vec::new(),
            components: HashMap::new(),
        }
    }

    pub fn with_parent(id: u64, name: String, parent_id: u64) -> Self {
        Self {
            id,
            name,
            parent: Some(parent_id),
            children: SmallVec::new(),
            transform: Transform::new(),
            visible: true,
            tags: Vec::new(),
            components: HashMap::new(),
        }
    }

    pub fn add_child(&mut self, child_id: u64) {
        if !self.children.contains(&child_id) {
            self.children.push(child_id);
        }
    }

    pub fn remove_child(&mut self, child_id: u64) {
        self.children.retain(|&id| id != child_id);
    }

    pub fn add_tag(&mut self, tag: &str) {
        if !self.tags.iter().any(|t| t == tag) {
            self.tags.push(tag.to_string());
        }
    }

    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }

    pub fn get_world_transform(&self) -> Mat4 {
        self.transform.world_matrix
    }
}

/// Scene graph for managing entity hierarchy
pub struct SceneGraph {
    pub nodes: RwLock<HashMap<u64, SceneNode>>,
    pub root_ids: RwLock<Vec<u64>>,
    pub next_id: RwLock<u64>,
    pub dirty_nodes: RwLock<Vec<u64>>,
}

impl SceneGraph {
    pub fn new() -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
            root_ids: RwLock::new(Vec::new()),
            next_id: RwLock::new(0),
            dirty_nodes: RwLock::new(Vec::new()),
        }
    }

    /// Create a new node and return its ID
    pub fn create_node(&self, name: &str, parent_id: Option<u64>) -> u64 {
        let mut next_id = self.next_id.write();
        let id = *next_id;
        *next_id += 1;

        let mut node = if let Some(pid) = parent_id {
            SceneNode::with_parent(id, name.to_string(), pid)
        } else {
            SceneNode::new(id, name.to_string())
        };

        node.transform.compute_local_matrix();

        let mut nodes = self.nodes.write();
        
        if let Some(pid) = parent_id {
            if let Some(parent) = nodes.get_mut(&pid) {
                parent.add_child(id);
            }
        } else {
            let mut root_ids = self.root_ids.write();
            root_ids.push(id);
        }

        nodes.insert(id, node);
        id
    }

    /// Remove a node and all its children
    pub fn remove_node(&self, id: u64) {
        let mut nodes = self.nodes.write();
        let mut to_remove = Vec::new();

        // Collect all descendants
        self.collect_descendants_recursive(id, &mut to_remove, &nodes);
        to_remove.push(id);

        // Remove from parent's children list
        for &node_id in &to_remove {
            if let Some(node) = nodes.get(&node_id) {
                if let Some(parent_id) = node.parent {
                    if let Some(parent) = nodes.get_mut(&parent_id) {
                        parent.remove_child(node_id);
                    }
                } else {
                    // Remove from root list
                    let mut root_ids = self.root_ids.write();
                    root_ids.retain(|&rid| rid != node_id);
                }
            }
        }

        // Remove all nodes
        for node_id in to_remove {
            nodes.remove(&node_id);
        }
    }

    fn collect_descendants_recursive(&self, id: u64, result: &mut Vec<u64>, nodes: &HashMap<u64, SceneNode>) {
        if let Some(node) = nodes.get(&id) {
            for &child_id in &node.children {
                result.push(child_id);
                self.collect_descendants_recursive(child_id, result, nodes);
            }
        }
    }

    /// Set the parent of a node
    pub fn set_parent(&self, node_id: u64, parent_id: Option<u64>) -> bool {
        let mut nodes = self.nodes.write();

        // Check for circular dependency
        if let Some(pid) = parent_id {
            if self.is_descendant(pid, node_id, &nodes) {
                log::error!("Cannot set parent: would create circular dependency");
                return false;
            }
        }

        if let Some(node) = nodes.get_mut(&node_id) {
            // Remove from old parent
            if let Some(old_parent_id) = node.parent.take() {
                if let Some(old_parent) = nodes.get_mut(&old_parent_id) {
                    old_parent.remove_child(node_id);
                } else {
                    let mut root_ids = self.root_ids.write();
                    root_ids.retain(|&rid| rid != node_id);
                }
            }

            // Add to new parent
            node.parent = parent_id;
            node.transform.dirty = true;

            if let Some(pid) = parent_id {
                if let Some(parent) = nodes.get_mut(&pid) {
                    parent.add_child(node_id);
                }
            } else {
                let mut root_ids = self.root_ids.write();
                root_ids.push(node_id);
            }

            // Mark as dirty
            let mut dirty_nodes = self.dirty_nodes.write();
            dirty_nodes.push(node_id);
        }

        true
    }

    fn is_descendant(&self, potential_ancestor: u64, node_id: u64, nodes: &HashMap<u64, SceneNode>) -> bool {
        let mut current = node_id;
        
        while let Some(node) = nodes.get(&current) {
            if let Some(parent_id) = node.parent {
                if parent_id == potential_ancestor {
                    return true;
                }
                current = parent_id;
            } else {
                break;
            }
        }

        false
    }

    /// Update all transforms hierarchically
    pub fn update_transforms(&self) {
        let mut nodes = self.nodes.write();
        let dirty_nodes = self.dirty_nodes.read();

        // First, update all dirty nodes' local matrices
        for &node_id in dirty_nodes.iter() {
            if let Some(node) = nodes.get_mut(&node_id) {
                node.transform.compute_local_matrix();
            }
        }
        drop(dirty_nodes);

        // Then compute world matrices from roots down
        let root_ids = self.root_ids.read().clone();
        drop(root_ids);

        let root_ids = self.root_ids.read();
        for &root_id in root_ids.iter() {
            self.update_world_matrix_recursive(root_id, Mat4::IDENTITY, &mut nodes);
        }

        self.dirty_nodes.write().clear();
    }

    fn update_world_matrix_recursive(
        &self,
        node_id: u64,
        parent_world: Mat4,
        nodes: &mut HashMap<u64, SceneNode>,
    ) {
        if let Some(node) = nodes.get_mut(&node_id) {
            node.transform.world_matrix = parent_world * node.transform.local_matrix;

            let children = node.children.clone();
            for child_id in children {
                self.update_world_matrix_recursive(child_id, node.transform.world_matrix, nodes);
            }
        }
    }

    /// Get a node by ID
    pub fn get_node(&self, id: u64) -> Option<SceneNode> {
        self.nodes.read().get(&id).cloned()
    }

    pub fn get_node_mut(&self, id: u64) -> Option<std::sync::MutexGuard<SceneNode>> {
        // This is a simplified version - in production you'd use parking_lot's mapped guards
        unimplemented!("Use get_node and update through methods")
    }

    /// Find nodes by name
    pub fn find_by_name(&self, name: &str) -> Vec<u64> {
        self.nodes.read()
            .iter()
            .filter(|(_, node)| node.name == name)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Find nodes by tag
    pub fn find_by_tag(&self, tag: &str) -> Vec<u64> {
        self.nodes.read()
            .iter()
            .filter(|(_, node)| node.has_tag(tag))
            .map(|(&id, _)| id)
            .collect()
    }

    /// Get all children of a node
    pub fn get_children(&self, parent_id: u64) -> Vec<u64> {
        self.nodes.read()
            .get(&parent_id)
            .map(|n| n.children.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Get the entire subtree rooted at a node
    pub fn get_subtree(&self, root_id: u64) -> Vec<u64> {
        let mut result = vec![root_id];
        self.collect_descendants_recursive(root_id, &mut result, &self.nodes.read());
        result
    }

    /// Get the path from root to a node
    pub fn get_path_to_root(&self, node_id: u64) -> Vec<u64> {
        let mut path = Vec::new();
        let nodes = self.nodes.read();

        let mut current = node_id;
        while let Some(node) = nodes.get(&current) {
            path.push(current);
            if let Some(parent_id) = node.parent {
                current = parent_id;
            } else {
                break;
            }
        }

        path.reverse();
        path
    }

    /// Mark a node as dirty (needs transform update)
    pub fn mark_dirty(&self, node_id: u64) {
        let mut dirty_nodes = self.dirty_nodes.write();
        if !dirty_nodes.contains(&node_id) {
            dirty_nodes.push(node_id);
        }
    }

    /// Get the number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.nodes.read().len()
    }

    /// Clear the entire scene graph
    pub fn clear(&self) {
        self.nodes.write().clear();
        self.root_ids.write().clear();
        self.dirty_nodes.write().clear();
        *self.next_id.write() = 0;
    }
}

impl Default for SceneGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility for batch transform operations
pub struct TransformBatch {
    operations: Vec<(u64, TransformOperation)>,
}

#[derive(Clone)]
pub enum TransformOperation {
    SetPosition(Vec3),
    SetRotation(Quat),
    SetScale(Vec3),
    Translate(Vec3),
    Rotate(Vec3, f32),
    LookAt(Vec3, Vec3),
}

impl TransformBatch {
    pub fn new() -> Self {
        Self { operations: Vec::new() }
    }

    pub fn set_position(&mut self, node_id: u64, pos: Vec3) {
        self.operations.push((node_id, TransformOperation::SetPosition(pos)));
    }

    pub fn set_rotation(&mut self, node_id: u64, rot: Quat) {
        self.operations.push((node_id, TransformOperation::SetRotation(rot)));
    }

    pub fn set_scale(&mut self, node_id: u64, scale: Vec3) {
        self.operations.push((node_id, TransformOperation::SetScale(scale)));
    }

    pub fn translate(&mut self, node_id: u64, delta: Vec3) {
        self.operations.push((node_id, TransformOperation::Translate(delta)));
    }

    pub fn rotate(&mut self, node_id: u64, axis: Vec3, angle: f32) {
        self.operations.push((node_id, TransformOperation::Rotate(axis, angle)));
    }

    pub fn execute(self, scene_graph: &SceneGraph) {
        for (node_id, op) in self.operations {
            if let Some(mut node) = scene_graph.get_node(node_id) {
                match op {
                    TransformOperation::SetPosition(pos) => node.transform.set_position(pos),
                    TransformOperation::SetRotation(rot) => node.transform.set_rotation(rot),
                    TransformOperation::SetScale(scale) => node.transform.set_scale(scale),
                    TransformOperation::Translate(delta) => node.transform.translate(delta),
                    TransformOperation::Rotate(axis, angle) => node.transform.rotate(axis, angle),
                    TransformOperation::LookAt(target, up) => node.transform.look_at(target, up),
                }
                scene_graph.mark_dirty(node_id);
            }
        }
        scene_graph.update_transforms();
    }
}

impl Default for TransformBatch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_graph_hierarchy() {
        let graph = SceneGraph::new();

        let root = graph.create_node("root", None);
        let child1 = graph.create_node("child1", Some(root));
        let child2 = graph.create_node("child2", Some(root));
        let grandchild = graph.create_node("grandchild", Some(child1));

        assert_eq!(graph.node_count(), 4);
        assert!(graph.get_children(root).contains(&child1));
        assert!(graph.get_children(root).contains(&child2));
        assert!(graph.get_children(child1).contains(&grandchild));
    }

    #[test]
    fn test_transform_propagation() {
        let graph = SceneGraph::new();

        let root = graph.create_node("root", None);
        let child = graph.create_node("child", Some(root));

        // Modify root transform
        {
            let mut nodes = graph.nodes.write();
            if let Some(r) = nodes.get_mut(&root) {
                r.transform.position = Vec3::new(10.0, 0.0, 0.0);
                r.transform.compute_local_matrix();
            }
        }

        graph.mark_dirty(root);
        graph.update_transforms();

        let child_node = graph.get_node(child).unwrap();
        assert!((child_node.transform.world_matrix.translation.x - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_circular_dependency_prevention() {
        let graph = SceneGraph::new();

        let root = graph.create_node("root", None);
        let child = graph.create_node("child", Some(root));

        // Try to make root a child of child (would create cycle)
        let result = graph.set_parent(root, Some(child));
        assert!(!result);
    }
}
