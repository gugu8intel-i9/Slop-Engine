// src/prefab.rs
// Reusable entity templates with override rules

use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use parking_lot::RwLock;
use glam::{Vec3, Quat, Mat4};

/// A component value that can be overridden
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ComponentValue {
    Bool(bool),
    Int(i32),
    Float(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Quat([f32; 4]),
    String(String),
    Resource(String),
    Object(HashMap<String, ComponentValue>),
    Array(Vec<ComponentValue>),
}

impl ComponentValue {
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ComponentValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            ComponentValue::Float(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_vec3(&self) -> Option<Vec3> {
        match self {
            ComponentValue::Vec3(v) => Some(Vec3::from_array(*v)),
            _ => None,
        }
    }
}

/// Override rule for a specific component field
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OverrideRule {
    pub path: String,
    pub value: ComponentValue,
    pub operation: OverrideOperation,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum OverrideOperation {
    Replace,
    Add,
    Multiply,
    Append,
    Remove,
}

/// Entity definition within a prefab
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrefabEntity {
    pub id: String,
    pub name: String,
    pub parent: Option<String>,
    pub components: HashMap<String, HashMap<String, ComponentValue>>,
    pub children: Vec<String>,
}

/// Complete prefab template
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prefab {
    pub id: String,
    pub name: String,
    pub version: u32,
    pub entities: HashMap<String, PrefabEntity>,
    pub root_entity: String,
    pub metadata: HashMap<String, String>,
    pub dependencies: Vec<String>,
}

impl Prefab {
    pub fn new(id: String, name: String) -> Self {
        Self {
            id,
            name,
            version: 1,
            entities: HashMap::new(),
            root_entity: String::new(),
            metadata: HashMap::new(),
            dependencies: Vec::new(),
        }
    }

    pub fn add_entity(&mut self, entity: PrefabEntity) {
        if self.root_entity.is_empty() && entity.parent.is_none() {
            self.root_entity = entity.id.clone();
        }
        self.entities.insert(entity.id.clone(), entity);
    }

    pub fn get_entity(&self, id: &str) -> Option<&PrefabEntity> {
        self.entities.get(id)
    }

    pub fn apply_overrides(
        &self,
        overrides: &[OverrideRule],
    ) -> Result<Prefab, PrefabError> {
        let mut result = self.clone();

        for rule in overrides {
            let parts: Vec<&str> = rule.path.split('.').collect();
            if parts.len() < 2 {
                return Err(PrefabError::InvalidPath(rule.path.clone()));
            }

            let entity_id = parts[0];
            let component_name = parts[1];
            let field_path = if parts.len() > 2 {
                parts[2..].join(".")
            } else {
                String::new()
            };

            if let Some(entity) = result.entities.get_mut(entity_id) {
                if let Some(component) = entity.components.get_mut(component_name) {
                    Self::apply_value_to_component(component, &field_path, &rule.value, rule.operation)?;
                } else {
                    // Create new component if it doesn't exist
                    let mut new_component = HashMap::new();
                    Self::apply_value_to_component(&mut new_component, &field_path, &rule.value, rule.operation)?;
                    entity.components.insert(component_name.to_string(), new_component);
                }
            } else {
                return Err(PrefabError::EntityNotFound(entity_id.to_string()));
            }
        }

        Ok(result)
    }

    fn apply_value_to_component(
        component: &mut HashMap<String, ComponentValue>,
        field_path: &str,
        value: &ComponentValue,
        operation: OverrideOperation,
    ) -> Result<(), PrefabError> {
        if field_path.is_empty() {
            // Replace entire component or merge
            match operation {
                OverrideOperation::Replace => {
                    component.clear();
                    if let ComponentValue::Object(fields) = value {
                        for (k, v) in fields {
                            component.insert(k.clone(), v.clone());
                        }
                    }
                }
                OverrideOperation::Append => {
                    if let ComponentValue::Object(fields) = value {
                        for (k, v) in fields {
                            component.insert(k.clone(), v.clone());
                        }
                    }
                }
                _ => return Err(PrefabError::InvalidOperation(operation)),
            }
        } else {
            // Apply to specific field
            match operation {
                OverrideOperation::Replace => {
                    component.insert(field_path.to_string(), value.clone());
                }
                OverrideOperation::Add => {
                    if let Some(existing) = component.get_mut(field_path) {
                        match (existing, value) {
                            (ComponentValue::Float(a), ComponentValue::Float(b)) => {
                                *a += b;
                            }
                            (ComponentValue::Int(a), ComponentValue::Int(b)) => {
                                *a += b;
                            }
                            (ComponentValue::Vec3(a), ComponentValue::Vec3(b)) => {
                                a[0] += b[0];
                                a[1] += b[1];
                                a[2] += b[2];
                            }
                            _ => return Err(PrefabError::TypeMismatch),
                        }
                    } else {
                        component.insert(field_path.to_string(), value.clone());
                    }
                }
                OverrideOperation::Multiply => {
                    if let Some(existing) = component.get_mut(field_path) {
                        match (existing, value) {
                            (ComponentValue::Float(a), ComponentValue::Float(b)) => {
                                *a *= b;
                            }
                            (ComponentValue::Int(a), ComponentValue::Int(b)) => {
                                *a *= b;
                            }
                            (ComponentValue::Vec3(a), ComponentValue::Vec3(b)) => {
                                a[0] *= b[0];
                                a[1] *= b[1];
                                a[2] *= b[2];
                            }
                            _ => return Err(PrefabError::TypeMismatch),
                        }
                    } else {
                        component.insert(field_path.to_string(), value.clone());
                    }
                }
                OverrideOperation::Remove => {
                    component.remove(field_path);
                }
                OverrideOperation::Append => {
                    return Err(PrefabError::InvalidOperation(operation));
                }
            }
        }

        Ok(())
    }
}

/// Instance of a prefab in the scene
#[derive(Clone, Debug)]
pub struct PrefabInstance {
    pub prefab_id: String,
    pub instance_id: u64,
    pub transform: Mat4,
    pub overrides: Vec<OverrideRule>,
    pub instantiated_entities: HashMap<String, u64>, // prefab entity id -> scene entity id
    pub is_dirty: bool,
}

impl PrefabInstance {
    pub fn new(prefab_id: String, instance_id: u64) -> Self {
        Self {
            prefab_id,
            instance_id,
            transform: Mat4::IDENTITY,
            overrides: Vec::new(),
            instantiated_entities: HashMap::new(),
            is_dirty: true,
        }
    }

    pub fn set_transform(&mut self, position: Vec3, rotation: Quat, scale: Vec3) {
        self.transform = Mat4::from_scale_rotation_translation(scale, rotation, position);
        self.is_dirty = true;
    }

    pub fn add_override(&mut self, rule: OverrideRule) {
        self.overrides.push(rule);
        self.is_dirty = true;
    }

    pub fn remove_override(&mut self, path: &str) {
        self.overrides.retain(|r| r.path != path);
        self.is_dirty = true;
    }
}

/// Error types for prefab operations
#[derive(Debug, Clone)]
pub enum PrefabError {
    EntityNotFound(String),
    ComponentNotFound(String),
    InvalidPath(String),
    TypeMismatch,
    InvalidOperation(OverrideOperation),
    CircularDependency,
    VersionMismatch,
}

impl std::fmt::Display for PrefabError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrefabError::EntityNotFound(id) => write!(f, "Entity not found: {}", id),
            PrefabError::ComponentNotFound(name) => write!(f, "Component not found: {}", name),
            PrefabError::InvalidPath(path) => write!(f, "Invalid path: {}", path),
            PrefabError::TypeMismatch => write!(f, "Type mismatch in override"),
            PrefabError::InvalidOperation(op) => write!(f, "Invalid operation: {:?}", op),
            PrefabError::CircularDependency => write!(f, "Circular dependency detected"),
            PrefabError::VersionMismatch => write!(f, "Version mismatch"),
        }
    }
}

impl std::error::Error for PrefabError {}

/// Prefab system for managing prefab definitions and instances
pub struct PrefabSystem {
    pub prefabs: RwLock<HashMap<String, Arc<Prefab>>>,
    pub instances: RwLock<HashMap<u64, PrefabInstance>>,
    pub next_instance_id: RwLock<u64>,
}

impl PrefabSystem {
    pub fn new() -> Self {
        Self {
            prefabs: RwLock::new(HashMap::new()),
            instances: RwLock::new(HashMap::new()),
            next_instance_id: RwLock::new(0),
        }
    }

    pub fn register_prefab(&self, prefab: Prefab) -> Result<(), PrefabError> {
        let mut prefabs = self.prefabs.write();
        
        // Check for circular dependencies
        if self.has_circular_dependency(&prefab, &prefabs) {
            return Err(PrefabError::CircularDependency);
        }

        prefabs.insert(prefab.id.clone(), Arc::new(prefab));
        Ok(())
    }

    fn has_circular_dependency(&self, prefab: &Prefab, prefabs: &HashMap<String, Arc<Prefab>>) -> bool {
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![prefab.id.clone()];

        while let Some(id) = stack.pop() {
            if visited.contains(&id) {
                return true;
            }
            visited.insert(id.clone());

            if let Some(p) = prefabs.get(&id) {
                for dep in &p.dependencies {
                    stack.push(dep.clone());
                }
            }
        }

        false
    }

    pub fn get_prefab(&self, id: &str) -> Option<Arc<Prefab>> {
        self.prefabs.read().get(id).cloned()
    }

    pub fn instantiate(&self, prefab_id: &str, transform: Mat4) -> Result<PrefabInstance, PrefabError> {
        let prefab = self.prefabs.read()
            .get(prefab_id)
            .cloned()
            .ok_or_else(|| PrefabError::EntityNotFound(prefab_id.to_string()))?;

        let mut instance_id_counter = self.next_instance_id.write();
        let instance_id = *instance_id_counter;
        *instance_id_counter += 1;

        let mut instance = PrefabInstance::new(prefab_id.to_string(), instance_id);
        instance.transform = transform;

        // Generate entity IDs for all entities in the prefab
        for entity_id in prefab.entities.keys() {
            *instance_id_counter += 1;
            instance.instantiated_entities.insert(entity_id.clone(), *instance_id_counter);
        }

        let mut instances = self.instances.write();
        instances.insert(instance_id, instance.clone());

        Ok(instance)
    }

    pub fn get_instance(&self, instance_id: u64) -> Option<PrefabInstance> {
        self.instances.read().get(&instance_id).cloned()
    }

    pub fn update_instance(&self, instance_id: u64, instance: PrefabInstance) {
        self.instances.write().insert(instance_id, instance);
    }

    pub fn destroy_instance(&self, instance_id: u64) {
        self.instances.write().remove(&instance_id);
    }

    pub fn get_instances_by_prefab(&self, prefab_id: &str) -> Vec<PrefabInstance> {
        self.instances.read()
            .values()
            .filter(|i| i.prefab_id == prefab_id)
            .cloned()
            .collect()
    }

    /// Unload a prefab and all its instances
    pub fn unload_prefab(&self, prefab_id: &str) {
        let mut instances = self.instances.write();
        instances.retain(|_, instance| instance.prefab_id != prefab_id);
        
        let mut prefabs = self.prefabs.write();
        prefabs.remove(prefab_id);
    }
}

impl Default for PrefabSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefab_creation() {
        let mut prefab = Prefab::new("player".to_string(), "Player".to_string());
        
        let mut components = HashMap::new();
        let mut transform = HashMap::new();
        transform.insert("position".to_string(), ComponentValue::Vec3([0.0, 0.0, 0.0]));
        transform.insert("scale".to_string(), ComponentValue::Vec3([1.0, 1.0, 1.0]));
        components.insert("Transform".to_string(), transform);

        let entity = PrefabEntity {
            id: "root".to_string(),
            name: "PlayerRoot".to_string(),
            parent: None,
            components,
            children: Vec::new(),
        };

        prefab.add_entity(entity);

        assert_eq!(prefab.name, "Player");
        assert!(!prefab.root_entity.is_empty());
    }

    #[test]
    fn test_prefab_overrides() {
        let system = PrefabSystem::new();
        
        let mut prefab = Prefab::new("enemy".to_string(), "Enemy".to_string());
        
        let mut components = HashMap::new();
        let mut stats = HashMap::new();
        stats.insert("health".to_string(), ComponentValue::Float(100.0));
        stats.insert("speed".to_string(), ComponentValue::Float(5.0));
        components.insert("Stats".to_string(), stats);

        let entity = PrefabEntity {
            id: "enemy_root".to_string(),
            name: "Enemy".to_string(),
            parent: None,
            components,
            children: Vec::new(),
        };

        prefab.add_entity(entity);
        system.register_prefab(prefab).unwrap();

        let instance = system.instantiate("enemy", Mat4::IDENTITY).unwrap();
        assert_eq!(instance.prefab_id, "enemy");
    }
}
