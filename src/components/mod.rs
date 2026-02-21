//! Core component system root.
//! Optimized for ECS cache coherence and minimal indirection.

use std::any::TypeId;
use std::collections::HashMap;
use std::sync::Arc;

// Submodules
pub mod transform;
pub mod render;
pub mod physics;
pub mod input;
pub mod stats;
pub mod lifecycle;

// Re-exports
pub use transform::Transform;
pub use render::{Sprite, Mesh, Material};
pub use physics::{Collider, Rigidbody, Velocity};
pub use input::InputState;
pub use stats::{Health, Mana, Experience};
pub use lifecycle::{Active, Paused};

/// Unique identifier for component types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComponentId(pub u64);

/// High-performance component trait.
/// Enforces memory safety constraints for parallel iteration.
pub trait Component: Send + Sync + 'static {
    fn type_id() -> ComponentId;
    fn size() -> usize;
    fn alignment() -> usize;
}

/// Bitmask for component archetypes.
#[derive(Debug, Default, Clone, Copy)]
pub struct ComponentMask(pub u128);

impl ComponentMask {
    #[inline]
    pub fn set(&mut self, id: usize) {
        self.0 |= 1 << id;
    }

    #[inline]
    pub fn has(&self, id: usize) -> bool {
        (self.0 & (1 << id)) != 0
    }

    #[inline]
    pub fn intersects(&self, other: &ComponentMask) -> bool {
        (self.0 & other.0) != 0
    }
}

/// Registry for component metadata.
/// Singleton pattern avoided for testability; passed via World.
pub struct ComponentRegistry {
    map: HashMap<TypeId, ComponentId>,
    next_id: u64,
}

impl ComponentRegistry {
    #[inline]
    pub fn new() -> Self {
        Self {
            map: HashMap::with_capacity(64),
            next_id: 0,
        }
    }

    #[inline]
    pub fn register<T: Component>(&mut self) -> ComponentId {
        let type_id = TypeId::of::<T>();
        *self.map.entry(type_id).or_insert_with(|| {
            let id = ComponentId(self.next_id);
            self.next_id += 1;
            id
        })
    }

    #[inline]
    pub fn get_id<T: Component>(&self) -> Option<ComponentId> {
        self.map.get(&TypeId::of::<T>()).copied()
    }
}

/// Macro to reduce boilerplate for Component implementations.
#[macro_export]
macro_rules! impl_component {
    ($t:ty, $id:expr) => {
        impl $crate::components::Component for $t {
            #[inline]
            fn type_id() -> $crate::components::ComponentId {
                $crate::components::ComponentId($id)
            }
            #[inline]
            fn size() -> usize {
                std::mem::size_of::<$t>()
            }
            #[inline]
            fn alignment() -> usize {
                std::mem::align_of::<$t>()
            }
        }
    };
}

/// Batch registration helper.
#[inline]
pub fn register_all(registry: &mut ComponentRegistry) {
    registry.register::<Transform>();
    registry.register::<Collider>();
    registry.register::<Health>();
    registry.register::<Sprite>();
    registry.register::<Velocity>();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_operations() {
        let mut mask = ComponentMask::default();
        mask.set(0);
        assert!(mask.has(0));
        assert!(!mask.has(1));
    }
}
