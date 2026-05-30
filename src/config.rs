// src/config.rs
// User settings, graphics options, keybinds, saved to disk

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use parking_lot::RwLock;

/// Graphics quality preset
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum GraphicsPreset {
    Low,
    Medium,
    High,
    Ultra,
    Custom,
}

/// Fullscreen mode
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum FullscreenMode {
    Windowed,
    Borderless,
    Exclusive,
}

/// Graphics configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphicsConfig {
    pub resolution: (u32, u32),
    pub fullscreen: FullscreenMode,
    pub vsync: bool,
    pub max_fps: Option<u32>,
    pub preset: GraphicsPreset,
    pub shadow_quality: u32,
    pub texture_quality: u32,
    pub anti_aliasing: u32,
    pub anisotropic_filtering: u32,
    pub ambient_occlusion: bool,
    pub screen_space_reflections: bool,
    pub motion_blur: bool,
    pub depth_of_field: bool,
    pub bloom: bool,
    pub chromatic_aberration: bool,
    pub film_grain: bool,
    pub fov: f32,
    pub gamma: f32,
    pub brightness: f32,
}

impl Default for GraphicsConfig {
    fn default() -> Self {
        Self {
            resolution: (1920, 1080),
            fullscreen: FullscreenMode::Windowed,
            vsync: true,
            max_fps: Some(60),
            preset: GraphicsPreset::High,
            shadow_quality: 2,
            texture_quality: 3,
            anti_aliasing: 2,
            anisotropic_filtering: 4,
            ambient_occlusion: true,
            screen_space_reflections: true,
            motion_blur: false,
            depth_of_field: true,
            bloom: true,
            chromatic_aberration: false,
            film_grain: false,
            fov: 90.0,
            gamma: 2.2,
            brightness: 1.0,
        }
    }
}

/// Audio configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioConfig {
    pub master_volume: f32,
    pub music_volume: f32,
    pub sfx_volume: f32,
    pub voice_volume: f32,
    pub ambient_volume: f32,
    pub output_device: Option<String>,
    pub input_device: Option<String>,
    pub enable_spatial_audio: bool,
    pub hrtf_enabled: bool,
    pub mute_on_focus_loss: bool,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            master_volume: 1.0,
            music_volume: 0.8,
            sfx_volume: 1.0,
            voice_volume: 1.0,
            ambient_volume: 0.7,
            output_device: None,
            input_device: None,
            enable_spatial_audio: true,
            hrtf_enabled: false,
            mute_on_focus_loss: true,
        }
    }
}

/// Input configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputConfig {
    pub keybindings: HashMap<String, KeyBinding>,
    pub mouse_sensitivity: f32,
    pub invert_y_axis: bool,
    pub controller_enabled: bool,
    pub controller_vibration: bool,
    pub touch_controls: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeyBinding {
    pub primary: String,
    pub secondary: Option<String>,
    pub modifiers: Vec<String>,
}

impl Default for InputConfig {
    fn default() -> Self {
        let mut keybindings = HashMap::new();
        keybindings.insert("move_forward".to_string(), KeyBinding {
            primary: "W".to_string(),
            secondary: Some("Up".to_string()),
            modifiers: vec![],
        });
        keybindings.insert("move_backward".to_string(), KeyBinding {
            primary: "S".to_string(),
            secondary: Some("Down".to_string()),
            modifiers: vec![],
        });
        keybindings.insert("move_left".to_string(), KeyBinding {
            primary: "A".to_string(),
            secondary: Some("Left".to_string()),
            modifiers: vec![],
        });
        keybindings.insert("move_right".to_string(), KeyBinding {
            primary: "D".to_string(),
            secondary: Some("Right".to_string()),
            modifiers: vec![],
        });
        keybindings.insert("jump".to_string(), KeyBinding {
            primary: "Space".to_string(),
            secondary: None,
            modifiers: vec![],
        });

        Self {
            keybindings,
            mouse_sensitivity: 1.0,
            invert_y_axis: false,
            controller_enabled: true,
            controller_vibration: true,
            touch_controls: false,
        }
    }
}

/// Gameplay configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameplayConfig {
    pub difficulty: String,
    pub language: String,
    pub subtitles: bool,
    pub tutorial_enabled: bool,
    pub auto_save: bool,
    pub auto_save_interval: u32,
    pub show_damage_numbers: bool,
    pub show_minimap: bool,
    pub show_fps: bool,
    pub show_ping: bool,
}

impl Default for GameplayConfig {
    fn default() -> Self {
        Self {
            difficulty: "normal".to_string(),
            language: "en".to_string(),
            subtitles: true,
            tutorial_enabled: true,
            auto_save: true,
            auto_save_interval: 300, // 5 minutes
            show_damage_numbers: true,
            show_minimap: true,
            show_fps: false,
            show_ping: true,
        }
    }
}

/// Network configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub server_address: String,
    pub server_port: u16,
    pub max_connections: u32,
    pub tick_rate: u32,
    pub interpolation_delay: f32,
    pub prediction_enabled: bool,
    pub lag_compensation: bool,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            server_address: "localhost".to_string(),
            server_port: 7777,
            max_connections: 64,
            tick_rate: 60,
            interpolation_delay: 0.1,
            prediction_enabled: true,
            lag_compensation: true,
        }
    }
}

/// Complete configuration
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Config {
    pub version: u32,
    pub graphics: GraphicsConfig,
    pub audio: AudioConfig,
    pub input: InputConfig,
    pub gameplay: GameplayConfig,
    pub network: NetworkConfig,
    pub custom: HashMap<String, serde_json::Value>,
}

impl Config {
    pub fn new() -> Self {
        Self {
            version: 1,
            ..Default::default()
        }
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        
        if !path.exists() {
            return Ok(Self::new());
        }

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let config: Config = serde_json::from_reader(reader)?;
        
        Ok(config)
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), ConfigError> {
        let path = path.as_ref();
        
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        
        Ok(())
    }

    pub fn get_keybinding(&self, action: &str) -> Option<&KeyBinding> {
        self.input.keybindings.get(action)
    }

    pub fn set_keybinding(&mut self, action: &str, binding: KeyBinding) {
        self.input.keybindings.insert(action.to_string(), binding);
    }

    pub fn apply_graphics_preset(&mut self, preset: GraphicsPreset) {
        self.graphics.preset = preset;
        
        match preset {
            GraphicsPreset::Low => {
                self.graphics.shadow_quality = 0;
                self.graphics.texture_quality = 1;
                self.graphics.anti_aliasing = 0;
                self.graphics.anisotropic_filtering = 1;
                self.graphics.ambient_occlusion = false;
                self.graphics.screen_space_reflections = false;
                self.graphics.motion_blur = false;
                self.graphics.depth_of_field = false;
                self.graphics.bloom = false;
            }
            GraphicsPreset::Medium => {
                self.graphics.shadow_quality = 1;
                self.graphics.texture_quality = 2;
                self.graphics.anti_aliasing = 1;
                self.graphics.anisotropic_filtering = 2;
                self.graphics.ambient_occlusion = true;
                self.graphics.screen_space_reflections = false;
                self.graphics.motion_blur = false;
                self.graphics.depth_of_field = true;
                self.graphics.bloom = true;
            }
            GraphicsPreset::High => {
                self.graphics.shadow_quality = 2;
                self.graphics.texture_quality = 3;
                self.graphics.anti_aliasing = 2;
                self.graphics.anisotropic_filtering = 4;
                self.graphics.ambient_occlusion = true;
                self.graphics.screen_space_reflections = true;
                self.graphics.motion_blur = false;
                self.graphics.depth_of_field = true;
                self.graphics.bloom = true;
            }
            GraphicsPreset::Ultra => {
                self.graphics.shadow_quality = 3;
                self.graphics.texture_quality = 4;
                self.graphics.anti_aliasing = 4;
                self.graphics.anisotropic_filtering = 8;
                self.graphics.ambient_occlusion = true;
                self.graphics.screen_space_reflections = true;
                self.graphics.motion_blur = true;
                self.graphics.depth_of_field = true;
                self.graphics.bloom = true;
            }
            GraphicsPreset::Custom => {}
        }
    }
}

/// Configuration manager with hot-reload support
pub struct ConfigManager {
    config: RwLock<Config>,
    config_path: PathBuf,
    watchers: RwLock<Vec<Box<dyn Fn(&Config) + Send + Sync>>>,
}

impl ConfigManager {
    pub fn new(config_path: PathBuf) -> Result<Self, ConfigError> {
        let config = Config::load(&config_path).unwrap_or_default();
        
        Ok(Self {
            config: RwLock::new(config),
            config_path,
            watchers: RwLock::new(Vec::new()),
        })
    }

    pub fn get(&self) -> std::sync::MutexGuard<Config> {
        unimplemented!("Use read/write methods")
    }

    pub fn read(&self) -> parking_lot::RwLockReadGuard<Config> {
        self.config.read()
    }

    pub fn write(&self) -> parking_lot::RwLockWriteGuard<Config> {
        self.config.write()
    }

    pub fn save(&self) -> Result<(), ConfigError> {
        let config = self.config.read();
        config.save(&self.config_path)
    }

    pub fn reload(&self) -> Result<(), ConfigError> {
        let new_config = Config::load(&self.config_path)?;
        *self.config.write() = new_config;
        Ok(())
    }

    pub fn add_watcher<F>(&self, callback: F)
    where
        F: Fn(&Config) + Send + Sync + 'static,
    {
        self.watchers.write().push(Box::new(callback));
    }

    pub fn notify_watchers(&self) {
        let config = self.config.read();
        let watchers = self.watchers.read();
        
        for watcher in watchers.iter() {
            watcher(&config);
        }
    }
}

/// Configuration error types
#[derive(Debug)]
pub enum ConfigError {
    IoError(std::io::Error),
    JsonError(serde_json::Error),
    InvalidValue(String),
    NotFound(String),
}

impl From<std::io::Error> for ConfigError {
    fn from(err: std::io::Error) -> Self {
        ConfigError::IoError(err)
    }
}

impl From<serde_json::Error> for ConfigError {
    fn from(err: serde_json::Error) -> Self {
        ConfigError::JsonError(err)
    }
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::IoError(e) => write!(f, "IO error: {}", e),
            ConfigError::JsonError(e) => write!(f, "JSON error: {}", e),
            ConfigError::InvalidValue(msg) => write!(f, "Invalid value: {}", msg),
            ConfigError::NotFound(path) => write!(f, "Not found: {}", path),
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::new();
        assert_eq!(config.version, 1);
        assert_eq!(config.graphics.resolution, (1920, 1080));
        assert!((config.graphics.fov - 90.0).abs() < 0.001);
    }

    #[test]
    fn test_graphics_preset() {
        let mut config = Config::new();
        config.apply_graphics_preset(GraphicsPreset::Low);
        
        assert_eq!(config.graphics.shadow_quality, 0);
        assert!(!config.graphics.ambient_occlusion);
    }

    #[test]
    fn test_keybindings() {
        let config = Config::new();
        let jump = config.get_keybinding("jump").unwrap();
        assert_eq!(jump.primary, "Space");
    }
}
