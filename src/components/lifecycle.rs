//! Lifecycle components for entity state management.

use crate::components::{Component, StorageType};

/// Active component indicating an entity is active in the scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Active;

impl Component for Active {
    const STORAGE_TYPE: StorageType = StorageType::SparseSet;
}

/// Paused component indicating an entity's logic is paused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Paused;

impl Component for Paused {
    const STORAGE_TYPE: StorageType = StorageType::SparseSet;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_active_component() {
        let _active = Active;
        // Marker component, just needs to exist
    }

    #[test]
    fn test_paused_component() {
        let _paused = Paused;
        // Marker component, just needs to exist
    }
}
