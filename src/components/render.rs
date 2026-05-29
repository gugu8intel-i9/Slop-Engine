//! Render components for graphics rendering.
//! Includes sprite, mesh, and material components.

use crate::components::{Component, StorageType};

/// Sprite component for 2D rendering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sprite {
    pub color: [f32; 4],
    pub uv_rect: [f32; 4], // [u, v, width, height]
    pub flip_x: bool,
    pub flip_y: bool,
}

impl Default for Sprite {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0, 1.0],
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            flip_x: false,
            flip_y: false,
        }
    }
}

impl Component for Sprite {
    const STORAGE_TYPE: StorageType = StorageType::Table;
}

/// Mesh component for 3D rendering.
#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    pub vertex_buffer_id: u64,
    pub index_buffer_id: u64,
    pub index_count: u32,
    pub bounding_radius: f32,
}

impl Component for Mesh {
    const STORAGE_TYPE: StorageType = StorageType::Table;
}

/// Material component referencing a material instance.
#[derive(Debug, Clone, PartialEq)]
pub struct Material {
    pub material_id: u64,
}

impl Component for Material {
    const STORAGE_TYPE: StorageType = StorageType::Table;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sprite_default() {
        let sprite = Sprite::default();
        assert_eq!(sprite.color, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(sprite.uv_rect, [0.0, 0.0, 1.0, 1.0]);
        assert!(!sprite.flip_x);
        assert!(!sprite.flip_y);
    }

    #[test]
    fn test_mesh_creation() {
        let mesh = Mesh {
            vertex_buffer_id: 1,
            index_buffer_id: 2,
            index_count: 100,
            bounding_radius: 5.0,
        };
        assert_eq!(mesh.vertex_buffer_id, 1);
        assert_eq!(mesh.index_count, 100);
    }
}
