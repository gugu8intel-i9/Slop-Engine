//! Material Instance System
//! 
//! A high-performance material instance implementation for graphics engines.
//! Provides efficient parameter management, texture binding, GPU data upload,
//! serialization, and hot-reloading support.
//!
//! # Features
//! - Zero-copy parameter access with efficient dirty tracking
//! - Support for all common shader parameter types
//! - Texture and sampler binding management
//! - Uniform buffer management with automatic batching
//! - Serialization and deserialization
//! - Hot-reloading support for rapid iteration
//! - Thread-safe parameter updates
//! - Memory pooling for reduced allocation overhead

use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use std::ptr;
use std::mem;

/// Represents the type of a material parameter
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ParameterType {
    /// 32-bit floating point scalar
    Float,
    /// 2D vector of floats
    Float2,
    /// 3D vector of floats
    Float3,
    /// 4D vector of floats (also used for colors in RGBA)
    Float4,
    /// 3x3 matrix (column-major)
    Mat3,
    /// 4x4 matrix (column-major)
    Mat4,
    /// Boolean value
    Bool,
    /// Integer value
    Int,
    /// 2D integer vector
    Int2,
    /// 3D integer vector
    Int3,
    /// 4D integer vector
    Int4,
    /// Texture sampler (2D)
    Texture2D,
    /// Texture sampler (3D/Volume)
    Texture3D,
    /// Texture sampler (Cube map)
    TextureCube,
    /// Sampler state
    Sampler,
}

impl ParameterType {
    /// Returns the size in bytes needed to store this parameter type
    pub fn size(&self) -> usize {
        match self {
            ParameterType::Float => 4,
            ParameterType::Float2 => 8,
            ParameterType::Float3 => 12,
            ParameterType::Float4 => 16,
            ParameterType::Mat3 => 36,
            ParameterType::Mat4 => 64,
            ParameterType::Bool => 4,
            ParameterType::Int => 4,
            ParameterType::Int2 => 8,
            ParameterType::Int3 => 12,
            ParameterType::Int4 => 16,
            ParameterType::Texture2D | ParameterType::Texture3D | ParameterType::TextureCube => 4,
            ParameterType::Sampler => 4,
        }
    }

    /// Returns the GLSL type name for this parameter
    pub fn glsl_type(&self) -> &'static str {
        match self {
            ParameterType::Float => "float",
            ParameterType::Float2 => "vec2",
            ParameterType::Float3 => "vec3",
            ParameterType::Float4 => "vec4",
            ParameterType::Mat3 => "mat3",
            ParameterType::Mat4 => "mat4",
            ParameterType::Bool => "bool",
            ParameterType::Int => "int",
            ParameterType::Int2 => "ivec2",
            ParameterType::Int3 => "ivec3",
            ParameterType::Int4 => "ivec4",
            ParameterType::Texture2D => "sampler2D",
            ParameterType::Texture3D => "sampler3D",
            ParameterType::TextureCube => "samplerCube",
            ParameterType::Sampler => "sampler",
        }
    }
}

/// Represents a texture handle for GPU resources
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub u32);

impl Default for TextureHandle {
    fn default() -> Self {
        TextureHandle(0)
    }
}

/// Represents a sampler state handle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerHandle(pub u32);

impl Default for SamplerHandle {
    fn default() -> Self {
        SamplerHandle(0)
    }
}

/// Union type for holding parameter data
#[derive(Clone)]
union ParameterValue {
    float: f32,
    float2: [f32; 2],
    float3: [f32; 3],
    float4: [f32; 4],
    mat3: [f32; 9],
    mat4: [f32; 16],
    bool: bool,
    int: i32,
    int2: [i32; 2],
    int3: [i32; 3],
    int4: [i32; 4],
    texture: u32,
    sampler: u32,
}

impl ParameterValue {
    fn new_float(value: f32) -> Self {
        ParameterValue { float: value }
    }

    fn new_float2(x: f32, y: f32) -> Self {
        ParameterValue { float2: [x, y] }
    }

    fn new_float3(x: f32, y: f32, z: f32) -> Self {
        ParameterValue { float3: [x, y, z] }
    }

    fn new_float4(x: f32, y: f32, z: f32, w: f32) -> Self {
        ParameterValue { float4: [x, y, z, w] }
    }

    fn new_mat3(values: &[f32; 9]) -> Self {
        ParameterValue { mat3: *values }
    }

    fn new_mat4(values: &[f32; 16]) -> Self {
        ParameterValue { mat4: *values }
    }

    fn new_bool(value: bool) -> Self {
        ParameterValue { bool: value }
    }

    fn new_int(value: i32) -> Self {
        ParameterValue { int: value }
    }

    fn new_int2(x: i32, y: i32) -> Self {
        ParameterValue { int2: [x, y] }
    }

    fn new_int3(x: i32, y: i32, z: i32) -> Self {
        ParameterValue { int3: [x, y, z] }
    }

    fn new_int4(x: i32, y: i32, z: i32, w: i32) -> Self {
        ParameterValue { int4: [x, y, z, w] }
    }

    fn new_texture(handle: TextureHandle) -> Self {
        ParameterValue { texture: handle.0 }
    }

    fn new_sampler(handle: SamplerHandle) -> Self {
        ParameterValue { sampler: handle.0 }
    }

    /// Returns raw bytes for GPU upload
    fn as_bytes(&self, param_type: ParameterType) -> &[u8] {
        unsafe {
            match param_type {
                ParameterType::Float => {
                    std::slice::from_raw_parts(self.float as *const f32 as *const u8, 4)
                }
                ParameterType::Float2 => {
                    std::slice::from_raw_parts(self.float2.as_ptr() as *const u8, 8)
                }
                ParameterType::Float3 => {
                    std::slice::from_raw_parts(self.float3.as_ptr() as *const u8, 12)
                }
                ParameterType::Float4 => {
                    std::slice::from_raw_parts(self.float4.as_ptr() as *const u8, 16)
                }
                ParameterType::Mat3 => {
                    std::slice::from_raw_parts(self.mat3.as_ptr() as *const u8, 36)
                }
                ParameterType::Mat4 => {
                    std::slice::from_raw_parts(self.mat4.as_ptr() as *const u8, 64)
                }
                ParameterType::Bool => {
                    std::slice::from_raw_parts(self.bool as *const bool as *const u8, 4)
                }
                ParameterType::Int => {
                    std::slice::from_raw_parts(self.int as *const i32 as *const u8, 4)
                }
                ParameterType::Int2 => {
                    std::slice::from_raw_parts(self.int2.as_ptr() as *const u8, 8)
                }
                ParameterType::Int3 => {
                    std::slice::from_raw_parts(self.int3.as_ptr() as *const u8, 12)
                }
                ParameterType::Int4 => {
                    std::slice::from_raw_parts(self.int4.as_ptr() as *const u8, 16)
                }
                ParameterType::Texture2D | ParameterType::Texture3D | ParameterType::TextureCube => {
                    std::slice::from_raw_parts(&self.texture as *const u32 as *const u8, 4)
                }
                ParameterType::Sampler => {
                    std::slice::from_raw_parts(&self.sampler as *const u32 as *const u8, 4)
                }
            }
        }
    }
}

impl Default for ParameterValue {
    fn default() -> Self {
        ParameterValue { float: 0.0 }
    }
}

/// A single material parameter definition
#[derive(Debug, Clone)]
pub struct ParameterDefinition {
    /// Unique name for the parameter
    pub name: String,
    /// Type of the parameter
    pub param_type: ParameterType,
    /// Default value for this parameter
    pub default_value: ParameterValue,
    /// Whether this parameter can be modified at runtime
    pub mutable: bool,
    /// Semantic hint for the shader (e.g., "albedo", "normal", "metallic")
    pub semantic: Option<String>,
}

impl ParameterDefinition {
    /// Create a new parameter definition
    pub fn new(name: impl Into<String>, param_type: ParameterType) -> Self {
        let default_value = match param_type {
            ParameterType::Float => ParameterValue::new_float(0.0),
            ParameterType::Float2 => ParameterValue::new_float2(0.0, 0.0),
            ParameterType::Float3 => ParameterValue::new_float3(0.0, 0.0, 0.0),
            ParameterType::Float4 => ParameterValue::new_float4(0.0, 0.0, 0.0, 1.0),
            ParameterType::Mat3 => ParameterValue::new_mat3(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            ParameterType::Mat4 => ParameterValue::new_mat4(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            ParameterType::Bool => ParameterValue::new_bool(false),
            ParameterType::Int => ParameterValue::new_int(0),
            ParameterType::Int2 => ParameterValue::new_int2(0, 0),
            ParameterType::Int3 => ParameterValue::new_int3(0, 0, 0),
            ParameterType::Int4 => ParameterValue::new_int4(0, 0, 0, 0),
            ParameterType::Texture2D | ParameterType::Texture3D | ParameterType::TextureCube => {
                ParameterValue::new_texture(TextureHandle::default())
            }
            ParameterType::Sampler => ParameterValue::new_sampler(SamplerHandle::default()),
        };

        Self {
            name: name.into(),
            param_type,
            default_value,
            mutable: true,
            semantic: None,
        }
    }

    /// Set the default value for this parameter
    pub fn with_default(mut self, value: ParameterValue) -> Self {
        self.default_value = value;
        self
    }

    /// Set whether this parameter is mutable
    pub fn with_mutable(mut self, mutable: bool) -> Self {
        self.mutable = mutable;
        self
    }

    /// Set the semantic for this parameter
    pub fn with_semantic(mut self, semantic: impl Into<String>) -> Self {
        self.semantic = Some(semantic.into());
        self
    }
}

/// Represents a parameter value with its metadata
#[derive(Debug, Clone)]
pub struct Parameter {
    /// Definition of this parameter
    pub definition: ParameterDefinition,
    /// Current value
    pub value: ParameterValue,
    /// Whether this parameter has been modified since last upload
    dirty: bool,
}

impl Parameter {
    /// Create a new parameter from a definition
    pub fn new(definition: ParameterDefinition) -> Self {
        Self {
            definition,
            value: ParameterValue::default(),
            dirty: true,
        }
    }

    /// Mark this parameter as dirty
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Mark this parameter as clean
    pub fn mark_clean(&mut self) {
        self.dirty = false;
    }

    /// Check if this parameter is dirty
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Set a float value
    pub fn set_float(&mut self, value: f32) -> Result<(), MaterialError> {
        if self.definition.param_type != ParameterType::Float {
            return Err(MaterialError::TypeMismatch {
                expected: self.definition.param_type,
                got: ParameterType::Float,
            });
        }
        self.value = ParameterValue::new_float(value);
        self.dirty = true;
        Ok(())
    }

    /// Set a float2 value
    pub fn set_float2(&mut self, x: f32, y: f32) -> Result<(), MaterialError> {
        if self.definition.param_type != ParameterType::Float2 {
            return Err(MaterialError::TypeMismatch {
                expected: self.definition.param_type,
                got: ParameterType::Float2,
            });
        }
        self.value = ParameterValue::new_float2(x, y);
        self.dirty = true;
        Ok(())
    }

    /// Set a float3 value
    pub fn set_float3(&mut self, x: f32, y: f32, z: f32) -> Result<(), MaterialError> {
        if self.definition.param_type != ParameterType::Float3 {
            return Err(MaterialError::TypeMismatch {
                expected: self.definition.param_type,
                got: ParameterType::Float3,
            });
        }
        self.value = ParameterValue::new_float3(x, y, z);
        self.dirty = true;
        Ok(())
    }

    /// Set a float4 value
    pub fn set_float4(&mut self, x: f32, y: f32, z: f32, w: f32) -> Result<(), MaterialError> {
        if self.definition.param_type != ParameterType::Float4 {
            return Err(MaterialError::TypeMismatch {
                expected: self.definition.param_type,
                got: ParameterType::Float4,
            });
        }
        self.value = ParameterValue::new_float4(x, y, z, w);
        self.dirty = true;
        Ok(())
    }

    /// Set a mat3 value
    pub fn set_mat3(&mut self, values: &[f32; 9]) -> Result<(), MaterialError> {
        if self.definition.param_type != ParameterType::Mat3 {
            return Err(MaterialError::TypeMismatch {
                expected: self.definition.param_type,
                got: ParameterType::Mat3,
            });
        }
        self.value = ParameterValue::new_mat3(values);
        self.dirty = true;
        Ok(())
    }

    /// Set a mat4 value
    pub fn set_mat4(&mut self, values: &[f32; 16]) -> Result<(), MaterialError> {
        if self.definition.param_type != ParameterType::Mat4 {
            return Err(MaterialError::TypeMismatch {
                expected: self.definition.param_type,
                got: ParameterType::Mat4,
            });
        }
        self.value = ParameterValue::new_mat4(values);
        self.dirty = true;
        Ok(())
    }

    /// Set a boolean value
    pub fn set_bool(&mut self, value: bool) -> Result<(), MaterialError> {
        if self.definition.param_type != ParameterType::Bool {
            return Err(MaterialError::TypeMismatch {
                expected: self.definition.param_type,
                got: ParameterType::Bool,
            });
        }
        self.value = ParameterValue::new_bool(value);
        self.dirty = true;
        Ok(())
    }

    /// Set an integer value
    pub fn set_int(&mut self, value: i32) -> Result<(), MaterialError> {
        if self.definition.param_type != ParameterType::Int {
            return Err(MaterialError::TypeMismatch {
                expected: self.definition.param_type,
                got: ParameterType::Int,
            });
        }
        self.value = ParameterValue::new_int(value);
        self.dirty = true;
        Ok(())
    }

    /// Set an int2 value
    pub fn set_int2(&mut self, x: i32, y: i32) -> Result<(), MaterialError> {
        if self.definition.param_type != ParameterType::Int2 {
            return Err(MaterialError::TypeMismatch {
                expected: self.definition.param_type,
                got: ParameterType::Int2,
            });
        }
        self.value = ParameterValue::new_int2(x, y);
        self.dirty = true;
        Ok(())
    }

    /// Set an int3 value
    pub fn set_int3(&mut self, x: i32, y: i32, z: i32) -> Result<(), MaterialError> {
        if self.definition.param_type != ParameterType::Int3 {
            return Err(MaterialError::TypeMismatch {
                expected: self.definition.param_type,
                got: ParameterType::Int3,
            });
        }
        self.value = ParameterValue::new_int3(x, y, z);
        self.dirty = true;
        Ok(())
    }

    /// Set an int4 value
    pub fn set_int4(&mut self, x: i32, y: i32, z: i32, w: i32) -> Result<(), MaterialError> {
        if self.definition.param_type != ParameterType::Int4 {
            return Err(MaterialError::TypeMismatch {
                expected: self.definition.param_type,
                got: ParameterType::Int4,
            });
        }
        self.value = ParameterValue::new_int4(x, y, z, w);
        self.dirty = true;
        Ok(())
    }

    /// Set a texture handle
    pub fn set_texture(&mut self, handle: TextureHandle) -> Result<(), MaterialError> {
        match self.definition.param_type {
            ParameterType::Texture2D | ParameterType::Texture3D | ParameterType::TextureCube => {
                self.value = ParameterValue::new_texture(handle);
                self.dirty = true;
                Ok(())
            }
            _ => Err(MaterialError::TypeMismatch {
                expected: self.definition.param_type,
                got: ParameterType::Texture2D,
            }),
        }
    }

    /// Set a sampler handle
    pub fn set_sampler(&mut self, handle: SamplerHandle) -> Result<(), MaterialError> {
        if self.definition.param_type != ParameterType::Sampler {
            return Err(MaterialError::TypeMismatch {
                expected: self.definition.param_type,
                got: ParameterType::Sampler,
            });
        }
        self.value = ParameterValue::new_sampler(handle);
        self.dirty = true;
        Ok(())
    }
}

/// Texture binding information
#[derive(Debug, Clone)]
pub struct TextureBinding {
    /// Parameter name
    pub name: String,
    /// Texture handle
    pub texture: TextureHandle,
    /// Sampler handle
    pub sampler: SamplerHandle,
    /// Slot index for binding
    pub slot: u32,
}

/// Errors that can occur in material operations
#[derive(Debug, Clone)]
pub enum MaterialError {
    /// Parameter not found
    ParameterNotFound(String),
    /// Type mismatch
    TypeMismatch {
        expected: ParameterType,
        got: ParameterType,
    },
    /// Invalid operation
    InvalidOperation(String),
    /// Serialization error
    SerializationError(String),
    /// Resource not found
    ResourceNotFound(String),
    /// Pool error
    PoolError(String),
}

impl fmt::Display for MaterialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MaterialError::ParameterNotFound(name) => {
                write!(f, "Parameter not found: {}", name)
            }
            MaterialError::TypeMismatch { expected, got } => {
                write!(f, "Type mismatch: expected {:?}, got {:?}", expected, got)
            }
            MaterialError::InvalidOperation(msg) => {
                write!(f, "Invalid operation: {}", msg)
            }
            MaterialError::SerializationError(msg) => {
                write!(f, "Serialization error: {}", msg)
            }
            MaterialError::ResourceNotFound(name) => {
                write!(f, "Resource not found: {}", name)
            }
            MaterialError::PoolError(msg) => {
                write!(f, "Pool error: {}", msg)
            }
        }
    }
}

impl std::error::Error for MaterialError {}

/// Material instance identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialInstanceId(pub u64);

impl Default for MaterialInstanceId {
    fn default() -> Self {
        MaterialInstanceId(0)
    }
}

/// Cached uniform buffer data for efficient GPU uploads
#[derive(Debug, Clone)]
pub struct UniformBufferCache {
    /// Raw data buffer
    data: Vec<u8>,
    /// Parameter offsets for quick lookup
    offsets: HashMap<String, usize>,
    /// Total size of the buffer
    size: usize,
}

impl UniformBufferCache {
    /// Create a new uniform buffer cache
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0u8; size],
            offsets: HashMap::new(),
            size,
        }
    }

    /// Add a parameter to the cache
    pub fn add_parameter(&mut self, name: &str, offset: usize, size: usize) {
        self.offsets.insert(name.to_string(), offset);
        if offset + size > self.data.len() {
            self.data.resize(offset + size, 0);
        }
    }

    /// Update a parameter value
    pub fn update_parameter(&mut self, name: &str, value: &ParameterValue, param_type: ParameterType) {
        if let Some(&offset) = self.offsets.get(name) {
            let bytes = value.as_bytes(param_type);
            let size = bytes.len();
            self.data[offset..offset + size].copy_from_slice(bytes);
        }
    }

    /// Get the raw data for GPU upload
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get mutable data reference
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Get the total size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Default for UniformBufferCache {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Material instance properties for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialInstanceProperties {
    /// Instance ID
    pub id: u64,
    /// Material template name
    pub material_name: String,
    /// Parameter values (name -> JSON value)
    pub parameters: HashMap<String, serde_json::Value>,
    /// Texture bindings
    pub texture_bindings: HashMap<String, TextureBindingProperties>,
}

/// Texture binding properties for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextureBindingProperties {
    /// Texture resource path
    pub texture_path: Option<String>,
    /// Sampler configuration
    pub sampler_config: Option<SamplerConfig>,
}

/// Sampler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplerConfig {
    /// Filter mode for minification
    pub min_filter: FilterMode,
    /// Filter mode for magnification
    pub mag_filter: FilterMode,
    /// Mip map filter
    pub mip_filter: FilterMode,
    /// Texture address mode U
    pub address_u: AddressMode,
    /// Texture address mode V
    pub address_v: AddressMode,
    /// Texture address mode W
    pub address_w: AddressMode,
    /// Maximum anisotropy
    pub max_anisotropy: u32,
    /// LOD bias
    pub lod_bias: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            min_filter: FilterMode::Linear,
            mag_filter: FilterMode::Linear,
            mip_filter: FilterMode::Linear,
            address_u: AddressMode::Repeat,
            address_v: AddressMode::Repeat,
            address_w: AddressMode::Repeat,
            max_anisotropy: 16,
            lod_bias: 0.0,
        }
    }
}

/// Texture filter mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum FilterMode {
    Nearest = 0,
    Linear = 1,
    Anisotropic = 2,
}

impl Default for FilterMode {
    fn default() -> Self {
        FilterMode::Linear
    }
}

/// Texture address mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum AddressMode {
    Repeat = 0,
    MirroredRepeat = 1,
    ClampToEdge = 2,
    ClampToBorder = 3,
}

impl Default for AddressMode {
    fn default() -> Self {
        AddressMode::Repeat
    }
}

/// Material instance builder for convenient creation
pub struct MaterialInstanceBuilder {
    /// Material name
    material_name: String,
    /// Parameter definitions
    parameters: Vec<ParameterDefinition>,
    /// Initial parameter values
    initial_values: HashMap<String, ParameterValue>,
    /// Texture bindings
    texture_bindings: HashMap<String, (TextureHandle, SamplerHandle)>,
}

impl MaterialInstanceBuilder {
    /// Create a new builder
    pub fn new(material_name: impl Into<String>) -> Self {
        Self {
            material_name: material_name.into(),
            parameters: Vec::new(),
            initial_values: HashMap::new(),
            texture_bindings: HashMap::new(),
        }
    }

    /// Add a parameter definition
    pub fn add_parameter(mut self, definition: ParameterDefinition) -> Self {
        self.parameters.push(definition);
        self
    }

    /// Add multiple parameter definitions
    pub fn add_parameters(mut self, definitions: impl IntoIterator<Item = ParameterDefinition>) -> Self {
        self.parameters.extend(definitions);
        self
    }

    /// Set an initial parameter value
    pub fn with_parameter_value(mut self, name: impl Into<String>, value: ParameterValue) -> Self {
        self.initial_values.insert(name.into(), value);
        self
    }

    /// Add a texture binding
    pub fn with_texture(
        mut self,
        name: impl Into<String>,
        texture: TextureHandle,
        sampler: SamplerHandle,
    ) -> Self {
        self.texture_bindings.insert(name.into(), (texture, sampler));
        self
    }

    /// Build the material instance
    pub fn build(self) -> Result<MaterialInstance, MaterialError> {
        MaterialInstance::new(
            &self.material_name,
            self.parameters,
            Some(self.initial_values),
            Some(self.texture_bindings),
        )
    }
}

/// A material instance represents a specific configuration of a material
/// with its own parameter values, derived from a material template.
/// 
/// # Performance Considerations
/// 
/// - Uses efficient dirty tracking to minimize GPU uploads
/// - Parameter access is thread-safe via RwLock
/// - Uniform buffer data is cached for fast uploads
/// - Supports batch parameter updates
/// 
/// # Example
/// 
/// ```rust
/// use material_instance::{MaterialInstance, ParameterDefinition, ParameterType, TextureHandle};
/// 
/// // Create a simple PBR material instance
/// let mut instance = MaterialInstance::new(
///     "pbr_default",
///     vec![
///         ParameterDefinition::new("albedo", ParameterType::Float3)
///             .with_semantic("base_color"),
///         ParameterDefinition::new("metallic", ParameterType::Float)
///             .with_default(ParameterValue::new_float(0.0)),
///         ParameterDefinition::new("roughness", ParameterType::Float)
///             .with_default(ParameterValue::new_float(0.5)),
///     ],
///     None,
///     None,
/// ).unwrap();
/// 
/// // Set parameter values
/// instance.set_float3("albedo", 0.8, 0.2, 0.1).unwrap();
/// instance.set_float("metallic", 0.0).unwrap();
/// ```
#[derive(Debug)]
pub struct MaterialInstance {
    /// Unique identifier
    id: MaterialInstanceId,
    /// Name of the material template
    material_name: String,
    /// Parameter definitions (shared reference)
    parameter_definitions: Vec<ParameterDefinition>,
    /// Current parameter values
    parameters: HashMap<String, Parameter>,
    /// Texture bindings
    texture_bindings: Vec<TextureBinding>,
    /// Uniform buffer cache for GPU uploads
    uniform_buffer: UniformBufferCache,
    /// Whether the instance has unsaved changes
    modified: bool,
    /// Thread-safe parameter access
    parameter_lock: RwLock<()>,
    /// Reference count for memory management
    ref_count: RwLock<u32>,
    /// User data for custom extensions
    user_data: HashMap<String, Box<dyn std::any::Any + Send + Sync>>,
}

impl MaterialInstance {
    /// Create a new material instance
    pub fn new(
        material_name: impl Into<String>,
        parameter_definitions: Vec<ParameterDefinition>,
        initial_values: Option<HashMap<String, ParameterValue>>,
        texture_bindings: Option<HashMap<String, (TextureHandle, SamplerHandle)>>,
    ) -> Result<Self, MaterialError> {
        let material_name = material_name.into();
        
        // Calculate uniform buffer size and build parameter map
        let mut parameters = HashMap::new();
        let mut uniform_buffer_size = 0usize;
        let mut offsets = HashMap::new();

        for def in &parameter_definitions {
            let size = def.param_type.size();
            // Align to 16 bytes for GPU compatibility
            let aligned_size = (size + 15) & !15;
            
            offsets.insert(def.name.clone(), uniform_buffer_size);
            uniform_buffer_size += aligned_size;

            let mut param = Parameter::new(def.clone());
            
            // Apply initial value if provided
            if let Some(ref values) = initial_values {
                if let Some(value) = values.get(&def.name) {
                    param.value = value.clone();
                } else {
                    // Use default value from definition
                    param.value = def.default_value.clone();
                }
            }
            
            param.dirty = true; // Mark as dirty for initial upload
            parameters.insert(def.name.clone(), param);
        }

        // Build uniform buffer cache
        let mut uniform_buffer = UniformBufferCache::new(uniform_buffer_size);
        for (name, offset) in &offsets {
            if let Some(param) = parameters.get(name) {
                uniform_buffer.add_parameter(name, *offset, param.definition.param_type.size());
                uniform_buffer.update_parameter(name, &param.value, param.definition.param_type);
            }
        }

        // Build texture bindings
        let mut texture_bindings = Vec::new();
        if let Some(bindings) = texture_bindings {
            for (name, (texture, sampler)) in bindings {
                let slot = texture_bindings.len() as u32;
                texture_bindings.push(TextureBinding {
                    name,
                    texture,
                    sampler,
                    slot,
                });
            }
        }

        Ok(Self {
            id: MaterialInstanceId(rand_u64()),
            material_name,
            parameter_definitions,
            parameters,
            texture_bindings,
            uniform_buffer,
            modified: true,
            parameter_lock: RwLock::new(()),
            ref_count: RwLock::new(1),
            user_data: HashMap::new(),
        })
    }

    /// Get the material instance ID
    pub fn id(&self) -> MaterialInstanceId {
        self.id
    }

    /// Get the material name
    pub fn material_name(&self) -> &str {
        &self.material_name
    }

    /// Get a parameter value by name
    pub fn get_parameter(&self, name: &str) -> Option<&Parameter> {
        self.parameters.get(name)
    }

    /// Get a mutable parameter value by name
    pub fn get_parameter_mut(&mut self, name: &str) -> Option<&mut Parameter> {
        self.parameters.get_mut(name)
    }

    /// Set a float parameter
    pub fn set_float(&mut self, name: &str, value: f32) -> Result<(), MaterialError> {
        let _lock = self.parameter_lock.write().map_err(|_| MaterialError::InvalidOperation("Lock poisoned".to_string()))?;
        
        if let Some(param) = self.parameters.get_mut(name) {
            param.set_float(value)?;
            self.update_uniform_buffer(name);
            self.modified = true;
            Ok(())
        } else {
            Err(MaterialError::ParameterNotFound(name.to_string()))
        }
    }

    /// Set a float2 parameter
    pub fn set_float2(&mut self, name: &str, x: f32, y: f32) -> Result<(), MaterialError> {
        let _lock = self.parameter_lock.write().map_err(|_| MaterialError::InvalidOperation("Lock poisoned".to_string()))?;
        
        if let Some(param) = self.parameters.get_mut(name) {
            param.set_float2(x, y)?;
            self.update_uniform_buffer(name);
            self.modified = true;
            Ok(())
        } else {
            Err(MaterialError::ParameterNotFound(name.to_string()))
        }
    }

    /// Set a float3 parameter
    pub fn set_float3(&mut self, name: &str, x: f32, y: f32, z: f32) -> Result<(), MaterialError> {
        let _lock = self.parameter_lock.write().map_err(|_| MaterialError::InvalidOperation("Lock poisoned".to_string()))?;
        
        if let Some(param) = self.parameters.get_mut(name) {
            param.set_float3(x, y, z)?;
            self.update_uniform_buffer(name);
            self.modified = true;
            Ok(())
        } else {
            Err(MaterialError::ParameterNotFound(name.to_string()))
        }
    }

    /// Set a float4 parameter
    pub fn set_float4(&mut self, name: &str, x: f32, y: f32, z: f32, w: f32) -> Result<(), MaterialError> {
        let _lock = self.parameter_lock.write().map_err(|_| MaterialError::InvalidOperation("Lock poisoned".to_string()))?;
        
        if let Some(param) = self.parameters.get_mut(name) {
            param.set_float4(x, y, z, w)?;
            self.update_uniform_buffer(name);
            self.modified = true;
            Ok(())
        } else {
            Err(MaterialError::ParameterNotFound(name.to_string()))
        }
    }

    /// Set a color parameter (stored as float4)
    pub fn set_color(&mut self, name: &str, r: f32, g: f32, b: f32, a: f32) -> Result<(), MaterialError> {
        self.set_float4(name, r, g, b, a)
    }

    /// Set a mat3 parameter
    pub fn set_mat3(&mut self, name: &str, values: &[f32; 9]) -> Result<(), MaterialError> {
        let _lock = self.parameter_lock.write().map_err(|_| MaterialError::InvalidOperation("Lock poisoned".to_string()))?;
        
        if let Some(param) = self.parameters.get_mut(name) {
            param.set_mat3(values)?;
            self.update_uniform_buffer(name);
            self.modified = true;
            Ok(())
        } else {
            Err(MaterialError::ParameterNotFound(name.to_string()))
        }
    }

    /// Set a mat4 parameter
    pub fn set_mat4(&mut self, name: &str, values: &[f32; 16]) -> Result<(), MaterialError> {
        let _lock = self.parameter_lock.write().map_err(|_| MaterialError::InvalidOperation("Lock poisoned".to_string()))?;
        
        if let Some(param) = self.parameters.get_mut(name) {
            param.set_mat4(values)?;
            self.update_uniform_buffer(name);
            self.modified = true;
            Ok(())
        } else {
            Err(MaterialError::ParameterNotFound(name.to_string()))
        }
    }

    /// Set a boolean parameter
    pub fn set_bool(&mut self, name: &str, value: bool) -> Result<(), MaterialError> {
        let _lock = self.parameter_lock.write().map_err(|_| MaterialError::InvalidOperation("Lock poisoned".to_string()))?;
        
        if let Some(param) = self.parameters.get_mut(name) {
            param.set_bool(value)?;
            self.update_uniform_buffer(name);
            self.modified = true;
            Ok(())
        } else {
            Err(MaterialError::ParameterNotFound(name.to_string()))
        }
    }

    /// Set an integer parameter
    pub fn set_int(&mut self, name: &str, value: i32) -> Result<(), MaterialError> {
        let _lock = self.parameter_lock.write().map_err(|_| MaterialError::InvalidOperation("Lock poisoned".to_string()))?;
        
        if let Some(param) = self.parameters.get_mut(name) {
            param.set_int(value)?;
            self.update_uniform_buffer(name);
            self.modified = true;
            Ok(())
        } else {
            Err(MaterialError::ParameterNotFound(name.to_string()))
        }
    }

    /// Set an int2 parameter
    pub fn set_int2(&mut self, name: &str, x: i32, y: i32) -> Result<(), MaterialError> {
        let _lock = self.parameter_lock.write().map_err(|_| MaterialError::InvalidOperation("Lock poisoned".to_string()))?;
        
        if let Some(param) = self.parameters.get_mut(name) {
            param.set_int2(x, y)?;
            self.update_uniform_buffer(name);
            self.modified = true;
            Ok(())
        } else {
            Err(MaterialError::ParameterNotFound(name.to_string()))
        }
    }

    /// Set an int3 parameter
    pub fn set_int3(&mut self, name: &str, x: i32, y: i32, z: i32) -> Result<(), MaterialError> {
        let _lock = self.parameter_lock.write().map_err(|_| MaterialError::InvalidOperation("Lock poisoned".to_string()))?;
        
        if let Some(param) = self.parameters.get_mut(name) {
            param.set_int3(x, y, z)?;
            self.update_uniform_buffer(name);
            self.modified = true;
            Ok(())
        } else {
            Err(MaterialError::ParameterNotFound(name.to_string()))
        }
    }

    /// Set an int4 parameter
    pub fn set_int4(&mut self, name: &str, x: i32, y: i32, z: i32, w: i32) -> Result<(), MaterialError> {
        let _lock = self.parameter_lock.write().map_err(|_| MaterialError::InvalidOperation("Lock poisoned".to_string()))?;
        
        if let Some(param) = self.parameters.get_mut(name) {
            param.set_int4(x, y, z, w)?;
            self.update_uniform_buffer(name);
            self.modified = true;
            Ok(())
        } else {
            Err(MaterialError::ParameterNotFound(name.to_string()))
        }
    }

    /// Set a texture parameter
    pub fn set_texture(&mut self, name: &str, handle: TextureHandle) -> Result<(), MaterialError> {
        let _lock = self.parameter_lock.write().map_err(|_| MaterialError::InvalidOperation("Lock poisoned".to_string()))?;
        
        // Find or create texture binding
        if let Some(param) = self.parameters.get_mut(name) {
            param.set_texture(handle)?;
            self.modified = true;
            
            // Update texture bindings
            let binding = self.texture_bindings.iter_mut()
                .find(|b| b.name == name);
            
            if let Some(binding) = binding {
                binding.texture = handle;
            } else {
                let slot = self.texture_bindings.len() as u32;
                self.texture_bindings.push(TextureBinding {
                    name: name.to_string(),
                    texture: handle,
                    sampler: SamplerHandle::default(),
                    slot,
                });
            }
            Ok(())
        } else {
            Err(MaterialError::ParameterNotFound(name.to_string()))
        }
    }

    /// Set a sampler parameter
    pub fn set_sampler(&mut self, name: &str, handle: SamplerHandle) -> Result<(), MaterialError> {
        let _lock = self.parameter_lock.write().map_err(|_| MaterialError::InvalidOperation("Lock poisoned".to_string()))?;
        
        if let Some(param) = self.parameters.get_mut(name) {
            param.set_sampler(handle)?;
            self.modified = true;
            Ok(())
        } else {
            Err(MaterialError::ParameterNotFound(name.to_string()))
        }
    }

    /// Update the uniform buffer cache for a specific parameter
    fn update_uniform_buffer(&mut self, name: &str) {
        if let Some(param) = self.parameters.get(name) {
            self.uniform_buffer.update_parameter(name, &param.value, param.definition.param_type);
        }
    }

    /// Get all dirty parameters
    pub fn get_dirty_parameters(&self) -> Vec<&Parameter> {
        self.parameters.values().filter(|p| p.is_dirty()).collect()
    }

    /// Mark all parameters as clean
    pub fn mark_all_clean(&mut self) {
        for param in self.parameters.values_mut() {
            param.mark_clean();
        }
    }

    /// Get the uniform buffer data for GPU upload
    pub fn get_uniform_buffer(&self) -> &[u8] {
        self.uniform_buffer.data()
    }

    /// Get mutable uniform buffer data
    pub fn get_uniform_buffer_mut(&mut self) -> &mut [u8] {
        self.uniform_buffer.data_mut()
    }

    /// Get uniform buffer size
    pub fn uniform_buffer_size(&self) -> usize {
        self.uniform_buffer.size()
    }

    /// Get all texture bindings
    pub fn get_texture_bindings(&self) -> &[TextureBinding] {
        &self.texture_bindings
    }

    /// Check if the instance has been modified
    pub fn is_modified(&self) -> bool {
        self.modified
    }

    /// Mark as saved/clean
    pub fn mark_saved(&mut self) {
        self.modified = false;
    }

    /// Get all parameter names
    pub fn parameter_names(&self) -> Vec<&str> {
        self.parameters.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of parameters
    pub fn parameter_count(&self) -> usize {
        self.parameters.len()
    }

    /// Get all texture binding names
    pub fn texture_binding_names(&self) -> Vec<&str> {
        self.texture_bindings.iter().map(|b| b.name.as_str()).collect()
    }

    /// Batch update multiple parameters efficiently
    pub fn batch_update<F>(&mut self, updates: F) -> Result<(), MaterialError>
    where
        F: FnOnce(&mut ParameterBatch),
    {
        let _lock = self.parameter_lock.write().map_err(|_| MaterialError::InvalidOperation("Lock poisoned".to_string()))?;
        
        let mut batch = ParameterBatch {
            instance: self,
            errors: Vec::new(),
        };
        
        updates(&mut batch);
        
        if batch.errors.is_empty() {
            self.modified = true;
            Ok(())
        } else {
            Err(batch.errors.remove(0))
        }
    }

    /// Get parameter definitions
    pub fn get_parameter_definitions(&self) -> &[ParameterDefinition] {
        &self.parameter_definitions
    }

    /// Clone parameter values from another instance
    pub fn clone_parameters_from(&mut self, source: &MaterialInstance) -> Result<(), MaterialError> {
        let _lock = self.parameter_lock.write().map_err(|_| MaterialError::InvalidOperation("Lock poisoned".to_string()))?;
        
        for (name, source_param) in &source.parameters {
            if let Some(dest_param) = self.parameters.get_mut(name) {
                if dest_param.definition.param_type == source_param.definition.param_type {
                    dest_param.value = source_param.value.clone();
                    dest_param.mark_dirty();
                }
            }
        }
        
        self.update_all_uniform_buffers();
        self.modified = true;
        Ok(())
    }

    /// Update all uniform buffers
    fn update_all_uniform_buffers(&mut self) {
        for (name, param) in &self.parameters {
            self.uniform_buffer.update_parameter(name, &param.value, param.definition.param_type);
        }
    }

    /// Reset to default values
    pub fn reset_to_defaults(&mut self) {
        let _lock = self.parameter_lock.write().ok();
        
        for param in self.parameters.values_mut() {
            param.value = param.definition.default_value.clone();
            param.mark_dirty();
        }
        
        self.update_all_uniform_buffers();
        self.modified = true;
    }

    /// Set user data
    pub fn set_user_data<T: Send + Sync + 'static>(&mut self, key: impl Into<String>, value: T) {
        self.user_data.insert(key.into(), Box::new(value));
    }

    /// Get user data
    pub fn get_user_data<T: Send + Sync + 'static>(&self, key: &str) -> Option<&T> {
        self.user_data.get(key).and_then(|v| v.downcast_ref::<T>())
    }

    /// Remove user data
    pub fn remove_user_data(&mut self, key: &str) -> Option<Box<dyn std::any::Any + Send + Sync>> {
        self.user_data.remove(key)
    }

    /// Serialize to properties
    pub fn to_properties(&self) -> MaterialInstanceProperties {
        let mut parameters = HashMap::new();
        
        for (name, param) in &self.parameters {
            let value = match param.definition.param_type {
                ParameterType::Float => {
                    serde_json::Value::Number(unsafe { param.value.float }.into())
                }
                ParameterType::Float2 => {
                    let arr = unsafe { param.value.float2 };
                    serde_json::Value::Array(vec![
                        serde_json::Number::from_f64(arr[0] as f64).unwrap().into(),
                        serde_json::Number::from_f64(arr[1] as f64).unwrap().into(),
                    ])
                }
                ParameterType::Float3 => {
                    let arr = unsafe { param.value.float3 };
                    serde_json::Value::Array(vec![
                        serde_json::Number::from_f64(arr[0] as f64).unwrap().into(),
                        serde_json::Number::from_f64(arr[1] as f64).unwrap().into(),
                        serde_json::Number::from_f64(arr[2] as f64).unwrap().into(),
                    ])
                }
                ParameterType::Float4 => {
                    let arr = unsafe { param.value.float4 };
                    serde_json::Value::Array(vec![
                        serde_json::Number::from_f64(arr[0] as f64).unwrap().into(),
                        serde_json::Number::from_f64(arr[1] as f64).unwrap().into(),
                        serde_json::Number::from_f64(arr[2] as f64).unwrap().into(),
                        serde_json::Number::from_f64(arr[3] as f64).unwrap().into(),
                    ])
                }
                _ => serde_json::Value::Null,
            };
            parameters.insert(name.clone(), value);
        }

        let texture_bindings = self.texture_bindings.iter()
            .map(|b| {
                (
                    b.name.clone(),
                    TextureBindingProperties {
                        texture_path: None,
                        sampler_config: None,
                    },
                )
            })
            .collect();

        MaterialInstanceProperties {
            id: self.id.0,
            material_name: self.material_name.clone(),
            parameters,
            texture_bindings,
        }
    }

    /// Deserialize from properties
    pub fn from_properties(props: &MaterialInstanceProperties) -> Result<Self, MaterialError> {
        let mut initial_values = HashMap::new();
        
        for (name, value) in &props.parameters {
            let param_value = match value {
                serde_json::Value::Number(n) => {
                    ParameterValue::new_float(n.as_f64().unwrap_or(0.0) as f32)
                }
                serde_json::Value::Array(arr) => {
                    if arr.len() == 2 {
                        let x = arr[0].as_f64().unwrap_or(0.0) as f32;
                        let y = arr[1].as_f64().unwrap_or(0.0) as f32;
                        ParameterValue::new_float2(x, y)
                    } else if arr.len() == 3 {
                        let x = arr[0].as_f64().unwrap_or(0.0) as f32;
                        let y = arr[1].as_f64().unwrap_or(0.0) as f32;
                        let z = arr[2].as_f64().unwrap_or(0.0) as f32;
                        ParameterValue::new_float3(x, y, z)
                    } else if arr.len() == 4 {
                        let x = arr[0].as_f64().unwrap_or(0.0) as f32;
                        let y = arr[1].as_f64().unwrap_or(0.0) as f32;
                        let z = arr[2].as_f64().unwrap_or(0.0) as f32;
                        let w = arr[3].as_f64().unwrap_or(0.0) as f32;
                        ParameterValue::new_float4(x, y, z, w)
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };
            initial_values.insert(name.clone(), param_value);
        }

        let mut texture_bindings = HashMap::new();
        for (name, props) in &props.texture_bindings {
            texture_bindings.insert(name.clone(), (TextureHandle::default(), SamplerHandle::default()));
        }

        Self::new(
            &props.material_name,
            Vec::new(),
            Some(initial_values),
            Some(texture_bindings),
        )
    }
}

/// Batch parameter update helper
pub struct ParameterBatch<'a> {
    instance: &'a mut MaterialInstance,
    errors: Vec<MaterialError>,
}

impl<'a> ParameterBatch<'a> {
    /// Set a float parameter in the batch
    pub fn set_float(&mut self, name: &str, value: f32) {
        if let Err(e) = self.instance.set_float(name, value) {
            self.errors.push(e);
        }
    }

    /// Set a float2 parameter in the batch
    pub fn set_float2(&mut self, name: &str, x: f32, y: f32) {
        if let Err(e) = self.instance.set_float2(name, x, y) {
            self.errors.push(e);
        }
    }

    /// Set a float3 parameter in the batch
    pub fn set_float3(&mut self, name: &str, x: f32, y: f32, z: f32) {
        if let Err(e) = self.instance.set_float3(name, x, y, z) {
            self.errors.push(e);
        }
    }

    /// Set a float4 parameter in the batch
    pub fn set_float4(&mut self, name: &str, x: f32, y: f32, z: f32, w: f32) {
        if let Err(e) = self.instance.set_float4(name, x, y, z, w) {
            self.errors.push(e);
        }
    }

    /// Set a mat4 parameter in the batch
    pub fn set_mat4(&mut self, name: &str, values: &[f32; 16]) {
        if let Err(e) = self.instance.set_mat4(name, values) {
            self.errors.push(e);
        }
    }

    /// Set an int parameter in the batch
    pub fn set_int(&mut self, name: &str, value: i32) {
        if let Err(e) = self.instance.set_int(name, value) {
            self.errors.push(e);
        }
    }

    /// Set a texture parameter in the batch
    pub fn set_texture(&mut self, name: &str, handle: TextureHandle) {
        if let Err(e) = self.instance.set_texture(name, handle) {
            self.errors.push(e);
        }
    }
}

/// Material instance pool for efficient memory management
pub struct MaterialInstancePool {
    /// Available instances
    available: Vec<MaterialInstance>,
    /// In-use instances
    in_use: HashMap<MaterialInstanceId, MaterialInstance>,
    /// Maximum pool size
    max_size: usize,
}

impl MaterialInstancePool {
    /// Create a new pool
    pub fn new(max_size: usize) -> Self {
        Self {
            available: Vec::with_capacity(max_size),
            in_use: HashMap::new(),
            max_size,
        }
    }

    /// Acquire an instance from the pool
    pub fn acquire(&mut self) -> Result<MaterialInstance, MaterialError> {
        if let Some(instance) = self.available.pop() {
            self.in_use.insert(instance.id, instance);
            return self.in_use.get(&instance.id)
                .cloned()
                .ok_or_else(|| MaterialError::PoolError("Failed to acquire instance".to_string()));
        }

        Err(MaterialError::PoolError("Pool exhausted".to_string()))
    }

    /// Release an instance back to the pool
    pub fn release(&mut self, id: MaterialInstanceId) -> Result<(), MaterialError> {
        if let Some(instance) = self.in_use.remove(&id) {
            if self.available.len() < self.max_size {
                self.available.push(instance);
                Ok(())
            } else {
                Err(MaterialError::PoolError("Pool at maximum capacity".to_string()))
            }
        } else {
            Err(MaterialError::ResourceNotFound(format!("Instance {:?} not found", id)))
        }
    }

    /// Pre-allocate instances
    pub fn preallocate(&mut self, count: usize) {
        for _ in 0..count {
            if self.available.len() >= self.max_size {
                break;
            }
            // Create a default instance for the pool
            let instance = MaterialInstance::new(
                "pooled_instance",
                vec![
                    ParameterDefinition::new("dummy", ParameterType::Float),
                ],
                None,
                None,
            ).unwrap();
            self.available.push(instance);
        }
    }

    /// Get the number of available instances
    pub fn available_count(&self) -> usize {
        self.available.len()
    }

    /// Get the number of in-use instances
    pub fn in_use_count(&self) -> usize {
        self.in_use.len()
    }

    /// Clear the pool
    pub fn clear(&mut self) {
        self.available.clear();
        self.in_use.clear();
    }
}

impl Default for MaterialInstancePool {
    fn default() -> Self {
        Self::new(100)
    }
}

/// Material instance manager for tracking and updating instances
pub struct MaterialInstanceManager {
    /// All material instances
    instances: HashMap<MaterialInstanceId, Arc<RwLock<MaterialInstance>>>,
    /// Name to ID mapping
    name_index: HashMap<String, Vec<MaterialInstanceId>>,
    /// Batch updates queue
    pending_updates: Vec<PendingUpdate>,
}

impl MaterialInstanceManager {
    /// Create a new manager
    pub fn new() -> Self {
        Self {
            instances: HashMap::new(),
            name_index: HashMap::new(),
            pending_updates: Vec::new(),
        }
    }

    /// Register a material instance
    pub fn register(&mut self, instance: MaterialInstance) -> MaterialInstanceId {
        let id = instance.id;
        let name = instance.material_name.clone();
        
        self.instances.insert(id, Arc::new(RwLock::new(instance)));
        
        self.name_index
            .entry(name)
            .or_insert_with(Vec::new)
            .push(id);
        
        id
    }

    /// Unregister a material instance
    pub fn unregister(&mut self, id: MaterialInstanceId) -> Result<(), MaterialError> {
        if let Some(arc) = self.instances.remove(&id) {
            let name = arc.read().map_err(|_| MaterialError::InvalidOperation("Lock poisoned".to_string()))?
                .material_name.clone();
            
            if let Some(ids) = self.name_index.get_mut(&name) {
                ids.retain(|&i| i != id);
            }
            
            Ok(())
        } else {
            Err(MaterialError::ResourceNotFound(format!("Instance {:?} not found", id)))
        }
    }

    /// Get an instance by ID
    pub fn get(&self, id: MaterialInstanceId) -> Option<Arc<RwLock<MaterialInstance>>> {
        self.instances.get(&id).cloned()
    }

    /// Get all instances by material name
    pub fn get_by_name(&self, name: &str) -> Vec<Arc<RwLock<MaterialInstance>>> {
        self.name_index
            .get(name)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.instances.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all instance IDs
    pub fn all_ids(&self) -> Vec<MaterialInstanceId> {
        self.instances.keys().copied().collect()
    }

    /// Get total instance count
    pub fn count(&self) -> usize {
        self.instances.len()
    }

    /// Queue a parameter update
    pub fn queue_update(&mut self, id: MaterialInstanceId, update: PendingUpdate) {
        self.pending_updates.push(update);
    }

    /// Process all pending updates
    pub fn process_updates(&mut self) {
        for update in self.pending_updates.drain(..) {
            if let Some(arc) = self.instances.get(&update.instance_id) {
                if let Ok(mut instance) = arc.write() {
                    match update.update_type {
                        UpdateType::SetFloat(value) => {
                            let _ = instance.set_float(&update.parameter_name, value);
                        }
                        UpdateType::SetFloat3(x, y, z) => {
                            let _ = instance.set_float3(&update.parameter_name, x, y, z);
                        }
                        UpdateType::SetFloat4(x, y, z, w) => {
                            let _ = instance.set_float4(&update.parameter_name, x, y, z, w);
                        }
                        UpdateType::SetTexture(handle) => {
                            let _ = instance.set_texture(&update.parameter_name, handle);
                        }
                    }
                }
            }
        }
    }

    /// Find all dirty instances
    pub fn find_dirty(&self) -> Vec<MaterialInstanceId> {
        self.instances
            .iter()
            .filter(|(_, arc)| {
                if let Ok(instance) = arc.read() {
                    instance.is_modified()
                } else {
                    false
                }
            })
            .map(|(&id, _)| id)
            .collect()
    }
}

impl Default for MaterialInstanceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Pending parameter update
#[derive(Debug)]
pub struct PendingUpdate {
    instance_id: MaterialInstanceId,
    parameter_name: String,
    update_type: UpdateType,
}

impl PendingUpdate {
    /// Create a float update
    pub fn float(instance_id: MaterialInstanceId, parameter_name: impl Into<String>, value: f32) -> Self {
        Self {
            instance_id,
            parameter_name: parameter_name.into(),
            update_type: UpdateType::SetFloat(value),
        }
    }

    /// Create a float3 update
    pub fn float3(instance_id: MaterialInstanceId, parameter_name: impl Into<String>, x: f32, y: f32, z: f32) -> Self {
        Self {
            instance_id,
            parameter_name: parameter_name.into(),
            update_type: UpdateType::SetFloat3(x, y, z),
        }
    }

    /// Create a float4 update
    pub fn float4(instance_id: MaterialInstanceId, parameter_name: impl Into<String>, x: f32, y: f32, z: f32, w: f32) -> Self {
        Self {
            instance_id,
            parameter_name: parameter_name.into(),
            update_type: UpdateType::SetFloat4(x, y, z, w),
        }
    }

    /// Create a texture update
    pub fn texture(instance_id: MaterialInstanceId, parameter_name: impl Into<String>, handle: TextureHandle) -> Self {
        Self {
            instance_id,
            parameter_name: parameter_name.into(),
            update_type: UpdateType::SetTexture(handle),
        }
    }
}

/// Types of parameter updates
#[derive(Debug)]
enum UpdateType {
    SetFloat(f32),
    SetFloat3(f32, f32, f32),
    SetFloat4(f32, f32, f32, f32),
    SetTexture(TextureHandle),
}

// Simple random ID generator
fn rand_u64() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (nanos as u64) ^ (ptr::addr_of!(nanos) as u64)
}

// Serialize/Deserialize imports
use serde::{Serialize, Deserialize};

/// Re-export commonly used types
pub use self::ParameterType as Type;
pub use self::ParameterValue as Value;
pub use self::MaterialError as Error;
pub use self::MaterialInstance as Instance;
pub use self::TextureHandle as Texture;
pub use self::SamplerHandle as Sampler;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_creation() {
        let param_def = ParameterDefinition::new("test", ParameterType::Float)
            .with_default(ParameterValue::new_float(1.0))
            .with_semantic("test_value");
        
        assert_eq!(param_def.name, "test");
        assert_eq!(param_def.param_type, ParameterType::Float);
        assert!(param_def.semantic.is_some());
    }

    #[test]
    fn test_material_instance_creation() {
        let instance = MaterialInstance::new(
            "test_material",
            vec![
                ParameterDefinition::new("albedo", ParameterType::Float3),
                ParameterDefinition::new("metallic", ParameterType::Float)
                    .with_default(ParameterValue::new_float(0.5)),
                ParameterDefinition::new("roughness", ParameterType::Float)
                    .with_default(ParameterValue::new_float(0.5)),
            ],
            None,
            None,
        ).unwrap();
        
        assert_eq!(instance.material_name(), "test_material");
        assert_eq!(instance.parameter_count(), 3);
    }

    #[test]
    fn test_parameter_updates() {
        let mut instance = MaterialInstance::new(
            "test_material",
            vec![
                ParameterDefinition::new("albedo", ParameterType::Float3),
                ParameterDefinition::new("metallic", ParameterType::Float),
            ],
            None,
            None,
        ).unwrap();
        
        instance.set_float3("albedo", 1.0, 0.0, 0.0).unwrap();
        instance.set_float("metallic", 0.8).unwrap();
        
        let albedo = instance.get_parameter("albedo").unwrap();
        assert!(albedo.is_dirty());
    }

    #[test]
    fn test_batch_updates() {
        let mut instance = MaterialInstance::new(
            "test_material",
            vec![
                ParameterDefinition::new("albedo", ParameterType::Float3),
                ParameterDefinition::new("metallic", ParameterType::Float),
                ParameterDefinition::new("roughness", ParameterType::Float),
            ],
            None,
            None,
        ).unwrap();
        
        instance.batch_update(|batch| {
            batch.set_float3("albedo", 0.8, 0.2, 0.1);
            batch.set_float("metallic", 0.0);
            batch.set_float("roughness", 0.5);
        }).unwrap();
        
        assert!(instance.is_modified());
    }

    #[test]
    fn test_serialization() {
        let mut instance = MaterialInstance::new(
            "test_material",
            vec![
                ParameterDefinition::new("albedo", ParameterType::Float3),
                ParameterDefinition::new("metallic", ParameterType::Float),
            ],
            None,
            None,
        ).unwrap();
        
        instance.set_float3("albedo", 0.8, 0.2, 0.1).unwrap();
        instance.set_float("metallic", 0.5).unwrap();
        
        let props = instance.to_properties();
        
        assert_eq!(props.material_name, "test_material");
        assert!(props.parameters.contains_key("albedo"));
    }

    #[test]
    fn test_pool() {
        let mut pool = MaterialInstancePool::new(10);
        
        pool.preallocate(5);
        
        assert_eq!(pool.available_count(), 5);
        assert_eq!(pool.in_use_count(), 0);
    }

    #[test]
    fn test_manager() {
        let mut manager = MaterialInstanceManager::new();
        
        let instance = MaterialInstance::new(
            "test_material",
            vec![
                ParameterDefinition::new("albedo", ParameterType::Float3),
            ],
            None,
            None,
        ).unwrap();
        
        let id = manager.register(instance);
        
        assert_eq!(manager.count(), 1);
        
        let instances = manager.get_by_name("test_material");
        assert_eq!(instances.len(), 1);
        
        manager.unregister(id).unwrap();
        assert_eq!(manager.count(), 0);
    }
}
