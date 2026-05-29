//! Physics components for collision detection and rigid body simulation.

use crate::components::{Component, StorageType};

/// Collider component for collision detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Collider {
    pub shape: ColliderShape,
    pub density: f32,
    pub friction: f32,
    pub restitution: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColliderShape {
    Sphere { radius: f32 },
    Box { half_extents: [f32; 3] },
    Capsule { radius: f32, height: f32 },
    Cylinder { radius: f32, height: f32 },
}

impl Default for Collider {
    fn default() -> Self {
        Self {
            shape: ColliderShape::Sphere { radius: 0.5 },
            density: 1.0,
            friction: 0.5,
            restitution: 0.0,
        }
    }
}

impl Component for Collider {
    const STORAGE_TYPE: StorageType = StorageType::Table;
}

/// Rigidbody component for physics simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rigidbody {
    pub mass: f32,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub is_kinematic: bool,
}

impl Default for Rigidbody {
    fn default() -> Self {
        Self {
            mass: 1.0,
            linear_damping: 0.0,
            angular_damping: 0.0,
            is_kinematic: false,
        }
    }
}

impl Component for Rigidbody {
    const STORAGE_TYPE: StorageType = StorageType::Table;
}

/// Velocity component for movement.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Velocity {
    pub linear: [f32; 3],
    pub angular: [f32; 3],
}

impl Component for Velocity {
    const STORAGE_TYPE: StorageType = StorageType::Table;
}

impl Velocity {
    #[inline(always)]
    pub fn new(linear: [f32; 3], angular: [f32; 3]) -> Self {
        Self { linear, angular }
    }

    #[inline(always)]
    pub fn zero() -> Self {
        Self {
            linear: [0.0; 3],
            angular: [0.0; 3],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collider_default() {
        let collider = Collider::default();
        assert_eq!(collider.density, 1.0);
        assert_eq!(collider.friction, 0.5);
    }

    #[test]
    fn test_rigidbody_default() {
        let rb = Rigidbody::default();
        assert_eq!(rb.mass, 1.0);
        assert!(!rb.is_kinematic);
    }

    #[test]
    fn test_velocity_creation() {
        let vel = Velocity::new([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert_eq!(vel.linear, [1.0, 0.0, 0.0]);
    }
}
