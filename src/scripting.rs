// src/scripting.rs
// Lua/Python/WASM/JS/Rust scripting integration

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Script language type
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScriptLanguage {
    Lua,
    Python,
    JavaScript,
    WebAssembly,
    Rust,
}

/// Script execution result
#[derive(Debug, Clone)]
pub enum ScriptResult {
    Success(ScriptValue),
    Error(String),
    None,
}

/// Script value types
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ScriptValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Array(Vec<ScriptValue>),
    Object(HashMap<String, ScriptValue>),
    Function(String),
}

impl ScriptValue {
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ScriptValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            ScriptValue::Int(i) => Some(*i),
            ScriptValue::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ScriptValue::Float(f) => Some(*f),
            ScriptValue::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            ScriptValue::String(s) => Some(s),
            _ => None,
        }
    }
}

/// Script function signature
#[derive(Clone, Debug)]
pub struct ScriptFunction {
    pub name: String,
    pub parameters: Vec<String>,
    pub return_type: String,
}

/// Compiled script module
#[derive(Clone)]
pub struct ScriptModule {
    pub id: String,
    pub name: String,
    pub language: ScriptLanguage,
    pub source: String,
    pub compiled_data: Option<Vec<u8>>,
    pub functions: Vec<ScriptFunction>,
    pub exports: HashMap<String, ScriptValue>,
}

/// Script instance with runtime state
pub struct ScriptInstance {
    pub module_id: String,
    pub instance_id: u64,
    pub variables: HashMap<String, ScriptValue>,
    pub is_running: bool,
}

impl ScriptInstance {
    pub fn new(module_id: String, instance_id: u64) -> Self {
        Self {
            module_id,
            instance_id,
            variables: HashMap::new(),
            is_running: false,
        }
    }
}

/// Native function that can be called from scripts
pub type NativeFunction = Box<dyn Fn(&[ScriptValue]) -> ScriptResult + Send + Sync>;

/// Script host for managing script execution
pub struct ScriptHost {
    pub modules: RwLock<HashMap<String, Arc<ScriptModule>>>,
    pub instances: RwLock<HashMap<u64, ScriptInstance>>,
    pub native_functions: RwLock<HashMap<String, NativeFunction>>,
    pub next_instance_id: RwLock<u64>,
    pub default_language: ScriptLanguage,
}

impl ScriptHost {
    pub fn new() -> Self {
        Self {
            modules: RwLock::new(HashMap::new()),
            instances: RwLock::new(HashMap::new()),
            native_functions: RwLock::new(HashMap::new()),
            next_instance_id: RwLock::new(0),
            default_language: ScriptLanguage::Lua,
        }
    }

    /// Register a native function callable from scripts
    pub fn register_native_function<F>(&self, name: &str, func: F)
    where
        F: Fn(&[ScriptValue]) -> ScriptResult + Send + Sync + 'static,
    {
        self.native_functions.write().insert(name.to_string(), Box::new(func));
    }

    /// Load a script module from source
    pub fn load_module(&self, name: &str, source: &str, language: ScriptLanguage) -> Result<String, ScriptError> {
        let mut modules = self.modules.write();
        
        let module = ScriptModule {
            id: name.to_string(),
            name: name.to_string(),
            language,
            source: source.to_string(),
            compiled_data: None,
            functions: Vec::new(),
            exports: HashMap::new(),
        };

        modules.insert(name.to_string(), Arc::new(module));
        Ok(name.to_string())
    }

    /// Create an instance of a script module
    pub fn create_instance(&self, module_id: &str) -> Result<u64, ScriptError> {
        let modules = self.modules.read();
        let module = modules.get(module_id)
            .ok_or_else(|| ScriptError::ModuleNotFound(module_id.to_string()))?;

        let mut next_id = self.next_instance_id.write();
        let instance_id = *next_id;
        *next_id += 1;

        let instance = ScriptInstance::new(module_id.to_string(), instance_id);
        
        let mut instances = self.instances.write();
        instances.insert(instance_id, instance);

        Ok(instance_id)
    }

    /// Call a script function
    pub fn call_function(
        &self,
        instance_id: u64,
        function_name: &str,
        args: &[ScriptValue],
    ) -> ScriptResult {
        let instances = self.instances.read();
        let instance = instances.get(&instance_id)
            .ok_or(ScriptError::InstanceNotFound(instance_id))
            .map_err(|e| ScriptResult::Error(e.to_string()));

        if let Err(e) = instance {
            return e.into();
        }

        // In a real implementation, this would call the actual script runtime
        // For now, check if it's a native function
        let native_functions = self.native_functions.read();
        if let Some(native_func) = native_functions.get(function_name) {
            return native_func(args);
        }

        ScriptResult::Error(format!("Function '{}' not found", function_name))
    }

    /// Get a variable from a script instance
    pub fn get_variable(&self, instance_id: u64, name: &str) -> Option<ScriptValue> {
        self.instances.read()
            .get(&instance_id)
            .and_then(|inst| inst.variables.get(name).cloned())
    }

    /// Set a variable in a script instance
    pub fn set_variable(&self, instance_id: u64, name: &str, value: ScriptValue) -> bool {
        if let Some(instance) = self.instances.write().get_mut(&instance_id) {
            instance.variables.insert(name.to_string(), value);
            true
        } else {
            false
        }
    }

    /// Remove a script instance
    pub fn remove_instance(&self, instance_id: u64) {
        self.instances.write().remove(&instance_id);
    }

    /// Hot reload a script module
    pub fn hot_reload(&self, module_id: &str, new_source: &str) -> Result<(), ScriptError> {
        let mut modules = self.modules.write();
        let module = modules.get_mut(module_id)
            .ok_or_else(|| ScriptError::ModuleNotFound(module_id.to_string()))?;

        let mut new_module = (**module).clone();
        new_module.source = new_source.to_string();
        new_module.compiled_data = None; // Force recompilation

        *module = Arc::new(new_module);
        Ok(())
    }

    /// Get all loaded modules
    pub fn get_modules(&self) -> Vec<String> {
        self.modules.read().keys().cloned().collect()
    }

    /// Get module info
    pub fn get_module_info(&self, module_id: &str) -> Option<ScriptModule> {
        self.modules.read().get(module_id).map(|m| (**m).clone())
    }
}

impl Default for ScriptHost {
    fn default() -> Self {
        Self::new()
    }
}

/// Script error types
#[derive(Debug, Clone)]
pub enum ScriptError {
    ModuleNotFound(String),
    InstanceNotFound(u64),
    SyntaxError(String),
    RuntimeError(String),
    TypeMismatch(String),
    FunctionNotFound(String),
    CompilationFailed(String),
}

impl std::fmt::Display for ScriptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScriptError::ModuleNotFound(id) => write!(f, "Module not found: {}", id),
            ScriptError::InstanceNotFound(id) => write!(f, "Instance not found: {}", id),
            ScriptError::SyntaxError(msg) => write!(f, "Syntax error: {}", msg),
            ScriptError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
            ScriptError::TypeMismatch(msg) => write!(f, "Type mismatch: {}", msg),
            ScriptError::FunctionNotFound(name) => write!(f, "Function not found: {}", name),
            ScriptError::CompilationFailed(msg) => write!(f, "Compilation failed: {}", msg),
        }
    }
}

impl std::error::Error for ScriptError {}

impl From<ScriptError> for ScriptResult {
    fn from(err: ScriptError) -> Self {
        ScriptResult::Error(err.to_string())
    }
}

/// Script component for entity-based scripting
#[derive(Clone)]
pub struct ScriptComponent {
    pub instance_id: u64,
    pub update_function: Option<String>,
    pub fixed_update_function: Option<String>,
    pub on_start_function: Option<String>,
    pub on_destroy_function: Option<String>,
    pub on_collision_function: Option<String>,
}

impl ScriptComponent {
    pub fn new(instance_id: u64) -> Self {
        Self {
            instance_id,
            update_function: Some("update".to_string()),
            fixed_update_function: Some("fixed_update".to_string()),
            on_start_function: Some("on_start".to_string()),
            on_destroy_function: Some("on_destroy".to_string()),
            on_collision_function: Some("on_collision".to_string()),
        }
    }
}

/// Script system for updating all script instances
pub struct ScriptSystem {
    pub host: Arc<ScriptHost>,
    pub components: RwLock<HashMap<u64, ScriptComponent>>, // entity_id -> ScriptComponent
}

impl ScriptSystem {
    pub fn new(host: Arc<ScriptHost>) -> Self {
        Self {
            host,
            components: RwLock::new(HashMap::new()),
        }
    }

    pub fn add_script(&self, entity_id: u64, component: ScriptComponent) {
        self.components.write().insert(entity_id, component);
    }

    pub fn remove_script(&self, entity_id: u64) {
        self.components.write().remove(&entity_id);
    }

    /// Update all scripts
    pub fn update(&self, dt: f32) {
        let components = self.components.read();
        
        for (entity_id, component) in components.iter() {
            if let Some(ref func_name) = component.update_function {
                let args = vec![
                    ScriptValue::Float(dt as f64),
                    ScriptValue::Int(*entity_id as i64),
                ];
                let _ = self.host.call_function(component.instance_id, func_name, &args);
            }
        }
    }

    /// Fixed timestep update for physics-related scripts
    pub fn fixed_update(&self, dt: f32) {
        let components = self.components.read();
        
        for component in components.iter() {
            if let Some(ref func_name) = component.1.fixed_update_function {
                let args = vec![ScriptValue::Float(dt as f64)];
                let _ = self.host.call_function(component.1.instance_id, func_name, &args);
            }
        }
    }

    /// Call on_start for all scripts
    pub fn start_all(&self) {
        let components = self.components.read();
        
        for component in components.iter() {
            if let Some(ref func_name) = component.1.on_start_function {
                let _ = self.host.call_function(component.1.instance_id, func_name, &[]);
            }
        }
    }

    /// Call on_destroy for all scripts
    pub fn stop_all(&self) {
        let components = self.components.read();
        
        for component in components.iter() {
            if let Some(ref func_name) = component.1.on_destroy_function {
                let _ = self.host.call_function(component.1.instance_id, func_name, &[]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_script_host_creation() {
        let host = ScriptHost::new();
        assert_eq!(host.default_language, ScriptLanguage::Lua);
    }

    #[test]
    fn test_native_function_registration() {
        let host = ScriptHost::new();
        
        host.register_native_function("add", |args| {
            if args.len() >= 2 {
                if let (Some(a), Some(b)) = (args[0].as_f64(), args[1].as_f64()) {
                    return ScriptResult::Success(ScriptValue::Float(a + b));
                }
            }
            ScriptResult::Error("Invalid arguments".to_string())
        });

        let result = host.call_function(0, "add", &[
            ScriptValue::Float(2.0),
            ScriptValue::Float(3.0),
        ]);

        match result {
            ScriptResult::Success(ScriptValue::Float(v)) => assert!((v - 5.0).abs() < 0.001),
            _ => panic!("Expected success"),
        }
    }

    #[test]
    fn test_script_value_conversions() {
        let val = ScriptValue::Int(42);
        assert_eq!(val.as_i64(), Some(42));
        assert_eq!(val.as_f64(), Some(42.0));
        assert_eq!(val.as_bool(), None);
    }
}
