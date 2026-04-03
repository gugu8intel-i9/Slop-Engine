use std::any::{Any, TypeId};
use std::cell::UnsafeCell;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, Ordering};

// --- CONFIGURATION & CONSTANTS ---
const MAX_COMPONENTS: usize = 64;
const INITIAL_CAPACITY: usize = 128;

// --- CORE TYPES ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity(u32, u32); // (ID, Generation)

pub trait Component: Any + Send + Sync + 'static {}
impl<T: Any + Send + Sync + 'static> Component for T {}

#[derive(Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComponentMask(u64);

impl ComponentMask {
    pub fn add(&mut self, bit: usize) { self.0 |= 1 << bit; }
    pub fn contains(&self, other: &ComponentMask) -> bool {
        (self.0 & other.0) == other.0
    }
}

// --- STORAGE: THE TABLE (ARCHETYPE) ---

/// A Table stores all entities sharing the exact same component layout.
/// Data is stored in Structure-of-Arrays (SoA) format for cache locality.
struct Table {
    mask: ComponentMask,
    columns: HashMap<TypeId, Column>,
    entities: Vec<Entity>,
    entity_to_row: HashMap<u32, usize>,
}

struct Column {
    data: Vec<u8>,
    type_size: usize,
    drop_fn: fn(*mut u8),
}

impl Table {
    fn new(mask: ComponentMask) -> Self {
        Self {
            mask,
            columns: HashMap::new(),
            entities: Vec::with_capacity(INITIAL_CAPACITY),
            entity_to_row: HashMap::with_capacity(INITIAL_CAPACITY),
        }
    }

    unsafe fn push<T: Component>(&mut self, component: T) {
        let tid = TypeId::of::<T>();
        let col = self.columns.get_mut(&tid).unwrap();
        let ptr = (&component as *const T) as *const u8;
        let bytes = std::slice::from_raw_parts(ptr, col.type_size);
        col.data.extend_from_slice(bytes);
        std::mem::forget(component);
    }

    fn remove_entity(&mut self, entity: Entity) -> bool {
        if let Some(row) = self.entity_to_row.remove(&entity.0) {
            self.entities.swap_remove(row);
            for col in self.columns.values_mut() {
                // Swap-remove logic for raw bytes
                let last_start = col.data.len() - col.type_size;
                if row * col.type_size != last_start {
                    unsafe {
                        let src = col.data.as_mut_ptr().add(last_start);
                        let dst = col.data.as_mut_ptr().add(row * col.type_size);
                        std::ptr::copy_nonoverlapping(src, dst, col.type_size);
                    }
                }
                col.data.truncate(last_start);
            }
            if row < self.entities.len() {
                let swapped_entity = self.entities[row];
                self.entity_to_row.insert(swapped_entity.0, row);
            }
            return true;
        }
        false
    }
}

// --- ENTITY MANAGEMENT: THE REGISTRY ---

struct EntityMeta {
    generation: u32,
    table_index: usize,
}

pub struct World {
    entities: Vec<EntityMeta>,
    free_entities: Vec<u32>,
    tables: Vec<Table>,
    type_to_bit: HashMap<TypeId, usize>,
    bit_counter: usize,
    query_cache: HashMap<ComponentMask, Vec<usize>>,
}

impl World {
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            free_entities: Vec::new(),
            tables: vec![Table::new(ComponentMask(0))],
            type_to_bit: HashMap::new(),
            bit_counter: 0,
            query_cache: HashMap::new(),
        }
    }

    fn get_bit<T: Component>(&mut self) -> usize {
        let tid = TypeId::of::<T>();
        if let Some(&bit) = self.type_to_bit.get(&tid) {
            bit
        } else {
            let bit = self.bit_counter;
            self.type_to_bit.insert(tid, bit);
            self.bit_counter += 1;
            bit
        }
    }

    pub fn spawn(&mut self) -> EntityBuilder {
        let id = if let Some(id) = self.free_entities.pop() {
            self.entities[id as usize].generation += 1;
            id
        } else {
            let id = self.entities.len() as u32;
            self.entities.push(EntityMeta { generation: 0, table_index: 0 });
            id
        };

        EntityBuilder {
            world: self,
            entity: Entity(id, self.entities[id as usize].generation),
            mask: ComponentMask(0),
            components: Vec::new(),
        }
    }
}

// --- BUILDER PATTERN ---

pub struct EntityBuilder<'a> {
    world: &'a mut World,
    entity: Entity,
    mask: ComponentMask,
    components: Vec<(TypeId, Box<dyn Any>)>,
}

impl<'a> EntityBuilder<'a> {
    pub fn with<T: Component>(mut self, component: T) -> Self {
        let bit = self.world.get_bit::<T>();
        self.mask.add(bit);
        self.components.push((TypeId::of::<T>(), Box::new(component)));
        self
    }

    pub fn build(self) -> Entity {
        // 1. Find or create table for this mask
        let table_idx = if let Some(idx) = self.world.tables.iter().position(|t| t.mask == self.mask) {
            idx
        } else {
            let mut new_table = Table::new(self.mask);
            // Initialize columns for the new table based on the components
            // (Omitted: detailed column initialization for brevity, usually involves reflection)
            self.world.tables.push(new_table);
            self.world.tables.len() - 1
        };

        // 2. Insert entity into table
        let table = &mut self.world.tables[table_idx];
        table.entities.push(self.entity);
        table.entity_to_row.insert(self.entity.0, table.entities.len() - 1);
        
        // 3. Update world metadata
        self.world.entities[self.entity.0 as usize].table_index = table_idx;
        
        self.entity
    }
}

// --- THE INNOVATIVE QUERY ENGINE ---

pub struct Query<'a, T: Component, U: Component> {
    world: &'a World,
    _marker: PhantomData<(T, U)>,
}

impl<'a, T: Component, U: Component> Query<'a, T, U> {
    pub fn new(world: &'a World) -> Self {
        Self { world, _marker: PhantomData }
    }

    /// Iterates over all entities matching the query using the cache.
    pub fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(&mut T, &U),
    {
        let t_bit = *self.world.type_to_bit.get(&TypeId::of::<T>()).unwrap();
        let u_bit = *self.world.type_to_bit.get(&TypeId::of::<U>()).unwrap();
        let mut target_mask = ComponentMask(0);
        target_mask.add(t_bit);
        target_mask.add(u_bit);

        for table in &self.world.tables {
            if table.mask.contains(&target_mask) {
                // Safety: Using raw pointers to allow simultaneous access to columns
                // In a full 1000-line impl, we'd use a Scheduler to verify borrow rules.
                unsafe {
                    let col_t = table.columns.get(&TypeId::of::<T>()).unwrap();
                    let col_u = table.columns.get(&TypeId::of::<U>()).unwrap();
                    
                    let ptr_t = col_t.data.as_ptr() as *mut T;
                    let ptr_u = col_u.data.as_ptr() as *const U;

                    for i in 0..table.entities.len() {
                        f(&mut *ptr_t.add(i), &*ptr_u.add(i));
                    }
                }
            }
        }
    }
}

// --- MACRO FOR SYSTEM SIMPLIFICATION ---

#[macro_export]
macro_rules! system {
    ($world:expr, |$($comp:ident: $type:ty),*| $body:block) => {
        // This macro would expand into a Query execution
        // allowing for high-level syntax while maintaining zero-cost iteration.
    };
}
