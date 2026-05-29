//! Transform component for entity positioning, rotation, and scaling.
//! Optimized for cache coherence and SIMD operations.

use glam::{Mat4, Quat, Vec3};
use crate::components::{Component, ComponentId, StorageType};

/// Transform component representing position, rotation, and scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Component for Transform {
    const STORAGE_TYPE: StorageType = StorageType::Table;
}

impl Transform {
    /// Creates a new transform with the given position, rotation, and scale.
    #[inline(always)]
    pub fn new(position: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self { position, rotation, scale }
    }

    /// Returns the transformation matrix.
    #[inline(always)]
    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }

    /// Translates the transform by the given vector.
    #[inline(always)]
    pub fn translate(&mut self, translation: Vec3) {
        self.position += translation;
    }

    /// Rotates the transform by the given quaternion.
    #[inline(always)]
    pub fn rotate(&mut self, rotation: Quat) {
        self.rotation = (rotation * self.rotation).normalize();
    }

    /// Scales the transform by the given factor.
    #[inline(always)]
    pub fn scale(&mut self, scale: Vec3) {
        self.scale *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_default() {
        let transform = Transform::default();
        assert_eq!(transform.position, Vec3::ZERO);
        assert_eq!(transform.rotation, Quat::IDENTITY);
        assert_eq!(transform.scale, Vec3::ONE);
    }

    #[test]
    fn test_transform_to_matrix() {
        let transform = Transform::new(Vec3::X, Quat::IDENTITY, Vec3::ONE);
        let matrix = transform.to_matrix();
        assert!((matrix.col(3).xyz() - Vec3::X).length() < 1e-6);
    }
}
