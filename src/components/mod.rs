//! Core component system root.
//! Optimized for ECS cache coherence, zero-cost abstractions, and O(1) resolutions.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::ops::{BitAnd, BitOr, BitXor, Not};

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

/// Global atomic counter for generating sequential, zero-cost Component IDs.
static NEXT_COMPONENT_ID: AtomicUsize = AtomicUsize::new(0);

/// Unique identifier for component types.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComponentId(pub usize);

/// Defines how the ECS should store this component in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageType {
    /// Contiguous arrays. Best for components attached to many entities and iterated frequently.
    Table,
    /// Sparse sets. Best for components attached/removed frequently or to very few entities.
    SparseSet,
}

/// High-performance component trait.
/// Provides O(1) compile-time static ID generation. No macros required!
pub trait Component: Send + Sync + 'static {
    /// The preferred storage architecture for this component.
    const STORAGE_TYPE: StorageType = StorageType::Table;

    /// Retrieves a unique, sequential ID for this component type in O(1) time.
    #[inline(always)]
    fn id() -> ComponentId {
        static ID: OnceLock<ComponentId> = OnceLock::new();
        *ID.get_or_init(|| {
            let id = NEXT_COMPONENT_ID.fetch_add(1, Ordering::Relaxed);
            assert!(id < 256, "ECS limits exceeded: Maximum of 256 component types supported.");
            ComponentId(id)
        })
    }
}

/// A 256-bit mask for component archetypes, optimized for SIMD.
#[repr(transparent)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComponentMask(pub [u64; 4]);

impl ComponentMask {
    #[inline(always)]
    pub fn set(&mut self, id: ComponentId) {
        let idx = id.0 / 64;
        let bit = id.0 % 64;
        self.0[idx] |= 1 << bit;
    }

    #[inline(always)]
    pub fn remove(&mut self, id: ComponentId) {
        let idx = id.0 / 64;
        let bit = id.0 % 64;
        self.0[idx] &= !(1 << bit);
    }

    #[inline(always)]
    pub fn has(&self, id: ComponentId) -> bool {
        let idx = id.0 / 64;
        let bit = id.0 % 64;
        (self.0[idx] & (1 << bit)) != 0
    }

    /// Checks if this mask contains all components of `other` (Subset).
    #[inline(always)]
    pub fn contains_all(&self, other: &ComponentMask) -> bool {
        (self.0[0] & other.0[0]) == other.0[0] &&
        (self.0[1] & other.0[1]) == other.0[1] &&
        (self.0[2] & other.0[2]) == other.0[2] &&
        (self.0[3] & other.0[3]) == other.0[3]
    }
}

// Ergonomic bitwise operations for Component Masks
impl BitOr for ComponentMask {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self([
            self.0[0] | rhs.0[0],
            self.0[1] | rhs.0[1],
            self.0[2] | rhs.0[2],
            self.0[3] | rhs.0[3],
        ])
    }
}

impl BitAnd for ComponentMask {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self([
            self.0[0] & rhs.0[0],
            self.0[1] & rhs.0[1],
            self.0[2] & rhs.0[2],
            self.0[3] & rhs.0[3],
        ])
    }
}

impl BitXor for ComponentMask {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self([
            self.0[0] ^ rhs.0[0],
            self.0[1] ^ rhs.0[1],
            self.0[2] ^ rhs.0[2],
            self.0[3] ^ rhs.0[3],
        ])
    }
}

impl Not for ComponentMask {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self([!self.0[0], !self.0[1], !self.0[2], !self.0[3]])
    }
}

/// Metadata for a registered component. Useful for serialization and reflection.
#[derive(Debug, Clone)]
pub struct ComponentMetadata {
    pub id: ComponentId,
    pub name: &'static str,
    pub size: usize,
    pub alignment: usize,
    pub storage_type: StorageType,
}

/// Registry for component metadata.
/// O(1) performance using direct Array Indexing instead of a HashMap.
#[derive(Default)]
pub struct ComponentRegistry {
    /// Indexed directly by `ComponentId.0`. Blazing fast lookups.
    metadata: Vec<Option<ComponentMetadata>>,
}

impl ComponentRegistry {
    #[inline]
    pub fn new() -> Self {
        Self {
            metadata: Vec::with_capacity(64), // Pre-allocate typical component counts
        }
    }

    /// Registers a component and stores its reflection data.
    pub fn register<T: Component>(&mut self) -> ComponentId {
        let id = T::id();
        let index = id.0;

        // Ensure array is large enough
        if index >= self.metadata.len() {
            self.metadata.resize(index + 1, None);
        }

        self.metadata[index] = Some(ComponentMetadata {
            id,
            name: std::any::type_name::<T>(),
            size: std::mem::size_of::<T>(),
            alignment: std::mem::align_of::<T>(),
            storage_type: T::STORAGE_TYPE,
        });

        id
    }

    /// Retrieves O(1) metadata for a given component ID.
    #[inline(always)]
    pub fn get_metadata(&self, id: ComponentId) -> Option<&ComponentMetadata> {
        self.metadata.get(id.0)?.as_ref()
    }
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

    struct DummyA;
    impl Component for DummyA {}

    struct DummyB;
    impl Component for DummyB {}

    #[test]
    fn test_o1_id_generation() {
        let id_a1 = DummyA::id();
        let id_a2 = DummyA::id();
        let id_b = DummyB::id();

        assert_eq!(id_a1, id_a2); // Same type = Same ID
        assert_ne!(id_a1, id_b);  // Different types = Different ID
    }

    #[test]
    fn test_mask_operations() {
        let mut mask1 = ComponentMask::default();
        mask1.set(DummyA::id());

        let mut mask2 = ComponentMask::default();
        mask2.set(DummyB::id());

        assert!(mask1.has(DummyA::id()));
        assert!(!mask1.has(DummyB::id()));

        // Test bitwise combination
        let combined = mask1 | mask2;
        assert!(combined.has(DummyA::id()));
        assert!(combined.has(DummyB::id()));
        
        // Test subset tracking
        assert!(combined.contains_all(&mask1));
    }
}
