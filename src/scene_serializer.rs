// src/scene_serializer.rs
// Save/load scenes to JSON/RON/binary

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use serde::{Deserialize, Serialize};
use glam::{Vec3, Quat, Mat4, Vec2, Vec4};
use parking_lot::RwLock;

/// Supported serialization formats
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SerializationFormat {
    Json,
    Ron,
    Binary,
}

/// Scene metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneMetadata {
    pub name: String,
    pub version: u32,
    pub created_at: u64,
    pub modified_at: u64,
    pub author: Option<String>,
    pub description: Option<String>,
    pub tags: Vec<String>,
}

/// Serialized entity data
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializedEntity {
    pub id: u64,
    pub name: String,
    pub parent: Option<u64>,
    pub children: Vec<u64>,
    pub transform: SerializedTransform,
    pub components: HashMap<String, SerializedComponent>,
    pub tags: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializedTransform {
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

impl From<Mat4> for SerializedTransform {
    fn from(mat: Mat4) -> Self {
        let (scale, rotation, position) = mat.to_scale_rotation_translation();
        Self {
            position: position.to_array(),
            rotation: rotation.to_array(),
            scale: scale.to_array(),
        }
    }
}

impl From<SerializedTransform> for Mat4 {
    fn from(st: SerializedTransform) -> Self {
        Mat4::from_scale_rotation_translation(
            Vec3::from_array(st.scale),
            Quat::from_array(st.rotation),
            Vec3::from_array(st.position),
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializedComponent {
    pub type_name: String,
    pub data: serde_json::Value,
}

/// Complete serialized scene
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializedScene {
    pub metadata: SceneMetadata,
    pub entities: Vec<SerializedEntity>,
    pub resources: HashMap<String, String>,
    pub settings: SceneSettings,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneSettings {
    pub ambient_light: [f32; 3],
    pub fog_enabled: bool,
    pub fog_color: [f32; 4],
    pub fog_density: f32,
    pub gravity: [f32; 3],
    pub time_of_day: f32,
}

impl Default for SceneSettings {
    fn default() -> Self {
        Self {
            ambient_light: [0.1, 0.1, 0.1],
            fog_enabled: false,
            fog_color: [0.5, 0.5, 0.5, 1.0],
            fog_density: 0.01,
            gravity: [0.0, -9.81, 0.0],
            time_of_day: 12.0,
        }
    }
}

/// Error types for serialization operations
#[derive(Debug)]
pub enum SerializationError {
    IoError(std::io::Error),
    JsonError(serde_json::Error),
    #[cfg(feature = "ron_support")]
    RonError(ron::error::SpannedError),
    BincodeError(bincode::Error),
    InvalidFormat,
    FileNotFound(String),
    ParseError(String),
}

impl From<std::io::Error> for SerializationError {
    fn from(err: std::io::Error) -> Self {
        SerializationError::IoError(err)
    }
}

impl From<serde_json::Error> for SerializationError {
    fn from(err: serde_json::Error) -> Self {
        SerializationError::JsonError(err)
    }
}

impl std::fmt::Display for SerializationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SerializationError::IoError(e) => write!(f, "IO error: {}", e),
            SerializationError::JsonError(e) => write!(f, "JSON error: {}", e),
            #[cfg(feature = "ron_support")]
            SerializationError::RonError(e) => write!(f, "RON error: {}", e),
            SerializationError::BincodeError(e) => write!(f, "Bincode error: {}", e),
            SerializationError::InvalidFormat => write!(f, "Invalid format"),
            SerializationError::FileNotFound(path) => write!(f, "File not found: {}", path),
            SerializationError::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for SerializationError {}

/// Scene serializer supporting multiple formats
pub struct SceneSerializer {
    pub default_format: SerializationFormat,
    pub compression_enabled: bool,
    pub pretty_print: bool,
}

impl SceneSerializer {
    pub fn new() -> Self {
        Self {
            default_format: SerializationFormat::Json,
            compression_enabled: false,
            pretty_print: true,
        }
    }

    /// Save a scene to file
    pub fn save<P: AsRef<Path>>(
        &self,
        scene: &SerializedScene,
        path: P,
        format: Option<SerializationFormat>,
    ) -> Result<(), SerializationError> {
        let format = format.unwrap_or(self.default_format);
        let path = path.as_ref();

        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        match format {
            SerializationFormat::Json => {
                if self.pretty_print {
                    serde_json::to_writer_pretty(&mut writer, scene)?;
                } else {
                    serde_json::to_writer(&mut writer, scene)?;
                }
            }
            SerializationFormat::Ron => {
                #[cfg(feature = "ron_support")]
                {
                    let config = ron::ser::PrettyConfig::default();
                    let output = ron::ser::to_string_pretty(scene, config)
                        .map_err(SerializationError::RonError)?;
                    writer.write_all(output.as_bytes())?;
                }
                #[cfg(not(feature = "ron_support"))]
                {
                    return Err(SerializationError::InvalidFormat);
                }
            }
            SerializationFormat::Binary => {
                #[cfg(feature = "bincode_support")]
                {
                    bincode::serialize_into(&mut writer, scene)
                        .map_err(SerializationError::BincodeError)?;
                }
                #[cfg(not(feature = "bincode_support"))]
                {
                    // Fallback to JSON binary
                    let json = serde_json::to_vec(scene)?;
                    if self.compression_enabled {
                        // Simple RLE compression could be added here
                        writer.write_all(&json)?;
                    } else {
                        writer.write_all(&json)?;
                    }
                }
            }
        }

        writer.flush()?;
        Ok(())
    }

    /// Load a scene from file
    pub fn load<P: AsRef<Path>>(
        &self,
        path: P,
        format: Option<SerializationFormat>,
    ) -> Result<SerializedScene, SerializationError> {
        let path = path.as_ref();
        
        if !path.exists() {
            return Err(SerializationError::FileNotFound(path.display().to_string()));
        }

        // Auto-detect format if not specified
        let format = format.unwrap_or_else(|| self.detect_format(path));

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        match format {
            SerializationFormat::Json => {
                let scene: SerializedScene = serde_json::from_reader(reader)?;
                Ok(scene)
            }
            SerializationFormat::Ron => {
                #[cfg(feature = "ron_support")]
                {
                    let mut contents = String::new();
                    reader.read_to_string(&mut contents)?;
                    let scene: SerializedScene = ron::from_str(&contents)
                        .map_err(SerializationError::RonError)?;
                    Ok(scene)
                }
                #[cfg(not(feature = "ron_support"))]
                {
                    Err(SerializationError::InvalidFormat)
                }
            }
            SerializationFormat::Binary => {
                #[cfg(feature = "bincode_support")]
                {
                    let scene: SerializedScene = bincode::deserialize_from(reader)
                        .map_err(SerializationError::BincodeError)?;
                    Ok(scene)
                }
                #[cfg(not(feature = "bincode_support"))]
                {
                    let mut contents = Vec::new();
                    reader.read_to_end(&mut contents)?;
                    let scene: SerializedScene = serde_json::from_slice(&contents)?;
                    Ok(scene)
                }
            }
        }
    }

    /// Detect format from file extension
    fn detect_format<P: AsRef<Path>>(&self, path: P) -> SerializationFormat {
        match path.as_ref().extension().and_then(|e| e.to_str()) {
            Some("json") | Some("scene") => SerializationFormat::Json,
            Some("ron") => SerializationFormat::Ron,
            Some("bin") => SerializationFormat::Binary,
            _ => self.default_format,
        }
    }

    /// Serialize to string
    pub fn to_string(&self, scene: &SerializedScene) -> Result<String, SerializationError> {
        match self.default_format {
            SerializationFormat::Json => {
                if self.pretty_print {
                    Ok(serde_json::to_string_pretty(scene)?)
                } else {
                    Ok(serde_json::to_string(scene)?)
                }
            }
            SerializationFormat::Ron => {
                #[cfg(feature = "ron_support")]
                {
                    let config = ron::ser::PrettyConfig::default();
                    ron::ser::to_string_pretty(scene, config)
                        .map_err(SerializationError::RonError)
                }
                #[cfg(not(feature = "ron_support"))]
                {
                    Err(SerializationError::InvalidFormat)
                }
            }
            SerializationFormat::Binary => {
                Err(SerializationError::InvalidFormat)
            }
        }
    }

    /// Deserialize from string
    pub fn from_string(&self, data: &str) -> Result<SerializedScene, SerializationError> {
        match self.default_format {
            SerializationFormat::Json => {
                Ok(serde_json::from_str(data)?)
            }
            SerializationFormat::Ron => {
                #[cfg(feature = "ron_support")]
                {
                    ron::from_str(data).map_err(SerializationError::RonError)
                }
                #[cfg(not(feature = "ron_support"))]
                {
                    Err(SerializationError::InvalidFormat)
                }
            }
            SerializationFormat::Binary => {
                Err(SerializationError::InvalidFormat)
            }
        }
    }

    /// Export scene to glTF
    pub fn export_gltf<P: AsRef<Path>>(
        &self,
        _scene: &SerializedScene,
        _path: P,
    ) -> Result<(), SerializationError> {
        // GLTF export implementation would go here
        // This requires the gltf crate and is more complex
        log::info!("GLTF export not yet implemented");
        Ok(())
    }

    /// Import scene from glTF
    pub fn import_gltf<P: AsRef<Path>>(
        &self,
        _path: P,
    ) -> Result<SerializedScene, SerializationError> {
        // GLTF import implementation would go here
        log::info!("GLTF import not yet implemented");
        Err(SerializationError::InvalidFormat)
    }
}

impl Default for SceneSerializer {
    fn default() -> Self {
        Self::new()
    }
}

/// Version control for scene files
pub struct SceneVersioning {
    pub history: RwLock<Vec<SceneVersion>>,
    pub max_history_size: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneVersion {
    pub version: u32,
    pub timestamp: u64,
    pub checksum: String,
    pub changes: Vec<String>,
    pub author: Option<String>,
}

impl SceneVersioning {
    pub fn new(max_history_size: usize) -> Self {
        Self {
            history: RwLock::new(Vec::new()),
            max_history_size,
        }
    }

    pub fn add_version(&self, version: SceneVersion) {
        let mut history = self.history.write();
        history.push(version);
        
        // Trim history if needed
        while history.len() > self.max_history_size {
            history.remove(0);
        }
    }

    pub fn get_version(&self, version: u32) -> Option<SceneVersion> {
        self.history.read()
            .iter()
            .find(|v| v.version == version)
            .cloned()
    }

    pub fn get_latest(&self) -> Option<SceneVersion> {
        self.history.read().last().cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization_roundtrip() {
        let serializer = SceneSerializer::new();
        
        let scene = SerializedScene {
            metadata: SceneMetadata {
                name: "Test Scene".to_string(),
                version: 1,
                created_at: 0,
                modified_at: 0,
                author: Some("Test".to_string()),
                description: None,
                tags: vec!["test".to_string()],
            },
            entities: vec![],
            resources: HashMap::new(),
            settings: SceneSettings::default(),
        };

        let json = serializer.to_string(&scene).unwrap();
        let loaded = serializer.from_string(&json).unwrap();

        assert_eq!(loaded.metadata.name, scene.metadata.name);
        assert_eq!(loaded.metadata.version, scene.metadata.version);
    }

    #[test]
    fn test_transform_conversion() {
        let original = Mat4::from_scale_rotation_translation(
            Vec3::new(2.0, 2.0, 2.0),
            Quat::from_rotation_y(std::f32::consts::PI / 4.0),
            Vec3::new(1.0, 2.0, 3.0),
        );

        let serialized: SerializedTransform = original.into();
        let restored: Mat4 = serialized.into();

        // Allow for floating point precision errors
        assert!((original.translation.x - restored.translation.x).abs() < 0.001);
        assert!((original.translation.y - restored.translation.y).abs() < 0.001);
        assert!((original.translation.z - restored.translation.z).abs() < 0.001);
    }
}
