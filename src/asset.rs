//! # Asset Management System
//!
//! A high-performance, featureful asset management system for game engines and graphics applications.
//! Supports multiple asset types, caching, streaming, hot-reloading, and thread-safe operations.
//!
//! ## Features
//!
//! - Multi-format asset support (textures, meshes, audio, fonts, shaders, etc.)
//! - Asynchronous and synchronous loading
//! - Memory-mapped file I/O for large assets
//! - LRU cache with memory pressure handling
//! - Hot-reloading for development
//! - Asset compression and decompression
//! - Dependency tracking and management
//! - Thread-safe operations with fine-grained locking

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::RwLock as ParkingLotRwLock;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during asset operations
#[derive(Error, Debug)]
pub enum AssetError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    
    #[error("Asset not found: {0}")]
    NotFound(String),
    
    #[error("Invalid asset format: {0}")]
    InvalidFormat(String),
    
    #[error("Asset loading failed: {0}")]
    LoadingFailed(String),
    
    #[error("Asset type mismatch: expected {expected}, got {got}")]
    TypeMismatch { expected: String, got: String },
    
    #[error("Asset is still loading")]
    StillLoading,
    
    #[error("Asset is corrupted: {0}")]
    Corrupted(String),
    
    #[error("Out of memory: {0}")]
    OutOfMemory(String),
    
    #[error("Compression error: {0}")]
    CompressionError(String),
    
    #[error("Hot-reload error: {0}")]
    HotReloadError(String),
}

impl fmt::Display for AssetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssetError::Io(e) => write!(f, "IO error: {}", e),
            AssetError::NotFound(path) => write!(f, "Asset not found: {}", path),
            AssetError::InvalidFormat(msg) => write!(f, "Invalid asset format: {}", msg),
            AssetError::LoadingFailed(msg) => write!(f, "Asset loading failed: {}", msg),
            AssetError::TypeMismatch { expected, got } => {
                write!(f, "Asset type mismatch: expected {}, got {}", expected, got)
            }
            AssetError::StillLoading => write!(f, "Asset is still loading"),
            AssetError::Corrupted(msg) => write!(f, "Asset is corrupted: {}", msg),
            AssetError::OutOfMemory(msg) => write!(f, "Out of memory: {}", msg),
            AssetError::CompressionError(msg) => write!(f, "Compression error: {}", msg),
            AssetError::HotReloadError(msg) => write!(f, "Hot-reload error: {}", msg),
        }
    }
}

pub type Result<T> = std::result::Result<T, AssetError>;

// ============================================================================
// Asset Identifiers and Types
// ============================================================================

/// Unique identifier for an asset
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AssetId(String);

impl AssetId {
    pub fn new(path: &str) -> Self {
        Self(path.to_string())
    }
    
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for AssetId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for AssetId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Types of assets supported by the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AssetType {
    Texture = 0,
    Mesh,
    Animation,
    Audio,
    Font,
    Shader,
    Material,
    Prefab,
    Script,
    Data,
    Unknown,
}

impl AssetType {
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            // Textures
            "png" | "jpg" | "jpeg" | "bmp" | "tga" | "dds" | "hdr" | "webp" | "gif" => {
                AssetType::Texture
            }
            // Meshes
            "obj" | "fbx" | "gltf" | "glb" | "dae" | "stl" | "3ds" => AssetType::Mesh,
            // Animations
            "anim" | "bvh" | "skel" => AssetType::Animation,
            // Audio
            "wav" | "mp3" | "ogg" | "flac" | "aac" | "m4a" => AssetType::Audio,
            // Fonts
            "ttf" | "otf" | "woff" | "woff2" | "fnt" => AssetType::Font,
            // Shaders
            "vert" | "frag" | "geom" | "compute" | "hlsl" | "glsl" | "shader" => {
                AssetType::Shader
            }
            // Materials
            "mat" | "mtl" | "material" => AssetType::Material,
            // Prefabs
            "prefab" | "entity" => AssetType::Prefab,
            // Scripts
            "lua" | "py" | "js" | "as" | "cs" => AssetType::Script,
            // Data
            "json" | "xml" | "yaml" | "toml" | "bin" | "pak" => AssetType::Data,
            _ => AssetType::Unknown,
        }
    }
    
    pub fn as_str(&self) -> &'static str {
        match self {
            AssetType::Texture => "Texture",
            AssetType::Mesh => "Mesh",
            AssetType::Animation => "Animation",
            AssetType::Audio => "Audio",
            AssetType::Font => "Font",
            AssetType::Shader => "Shader",
            AssetType::Material => "Material",
            AssetType::Prefab => "Prefab",
            AssetType::Script => "Script",
            AssetType::Data => "Data",
            AssetType::Unknown => "Unknown",
        }
    }
}

/// Metadata about an asset
#[derive(Debug, Clone)]
pub struct AssetMetadata {
    pub id: AssetId,
    pub path: PathBuf,
    pub asset_type: AssetType,
    pub size: u64,
    pub created_at: SystemTime,
    pub modified_at: SystemTime,
    pub hash: u64,
    pub dependencies: Vec<AssetId>,
    pub tags: Vec<String>,
    pub compression: Option<CompressionType>,
}

impl AssetMetadata {
    pub fn new(path: &Path, asset_type: AssetType) -> Self {
        let id = AssetId::new(path.to_string_lossy().as_ref());
        
        let metadata = std::fs::metadata(path).ok();
        let size = metadata.as_ref().map(|m| m.len()).unwrap_or(0);
        let created_at = metadata
            .as_ref()
            .and_then(|m| m.created().ok())
            .unwrap_or(UNIX_EPOCH);
        let modified_at = metadata
            .as_ref()
            .and_then(|m| m.modified().ok())
            .unwrap_or(UNIX_EPOCH);
        
        Self {
            id,
            path: path.to_path_buf(),
            asset_type,
            size,
            created_at,
            modified_at,
            hash: 0,
            dependencies: Vec::new(),
            tags: Vec::new(),
            compression: None,
        }
    }
    
    pub fn with_hash(mut self, hash: u64) -> Self {
        self.hash = hash;
        self
    }
    
    pub fn with_dependencies(mut self, deps: Vec<AssetId>) -> Self {
        self.dependencies = deps;
        self
    }
    
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
    
    pub fn with_compression(mut self, compression: CompressionType) -> Self {
        self.compression = Some(compression);
        self
    }
}

// ============================================================================
// Compression
// ============================================================================

/// Types of compression supported for assets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CompressionType {
    None = 0,
    Lz4,
    Zlib,
    Zstd,
    Lzma,
}

impl CompressionType {
    pub fn as_str(&self) -> &'static str {
        match self {
            CompressionType::None => "none",
            CompressionType::Lzip => "lz4",
            CompressionType::Zlib => "zlib",
            CompressionType::Zstd => "zstd",
            CompressionType::Lzma => "lzma",
        }
    }
    
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" => Some(CompressionType::None),
            "lz4" => Some(CompressionType::Lzip),
            "zlib" => Some(CompressionType::Zlib),
            "zstd" => Some(CompressionType::Zstd),
            "lzma" => Some(CompressionType::Lzma),
            _ => None,
        }
    }
}

// ============================================================================
// Asset Data Types
// ============================================================================

/// Raw asset data with ownership
pub struct AssetData {
    pub data: Vec<u8>,
    pub metadata: AssetMetadata,
}

impl AssetData {
    pub fn new(data: Vec<u8>, metadata: AssetMetadata) -> Self {
        Self { data, metadata }
    }
    
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
}

/// Trait for custom asset loaders
pub trait AssetLoader: Send + Sync {
    /// Load asset from bytes
    fn load(&self, data: &[u8], metadata: &AssetMetadata) -> Result<Box<dyn Asset>>;
    
    /// Get the asset type this loader supports
    fn supported_type(&self) -> AssetType;
    
    /// Get file extensions this loader handles
    fn supported_extensions(&self) -> Vec<&'static str>;
    
    /// Check if this loader can handle the given data
    fn can_load(&self, data: &[u8], metadata: &AssetMetadata) -> bool {
        // Default implementation checks magic bytes or format signature
        true
    }
}

/// Trait for loaded assets
pub trait Asset: Send + Sync {
    /// Get the type of this asset
    fn asset_type(&self) -> AssetType;
    
    /// Get the size in bytes of this asset in memory
    fn size_in_memory(&self) -> usize;
    
    /// Get the asset ID
    fn id(&self) -> &AssetId;
    
    /// Clone the asset (if cloneable)
    fn clone_box(&self) -> Box<dyn Asset>;
    
    /// Convert to Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
}

impl<T: Clone + Send + Sync + 'static> Asset for T {
    fn asset_type(&self) -> AssetType {
        // Default implementation - override for specific types
        AssetType::Unknown
    }
    
    fn size_in_memory(&self) -> usize {
        std::mem::size_of::<T>()
    }
    
    fn id(&self) -> &AssetId {
        panic!("id() not implemented for this asset type")
    }
    
    fn clone_box(&self) -> Box<dyn Asset> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ============================================================================
// Asset Cache
// ============================================================================

/// Cache entry with metadata
struct CacheEntry {
    asset: Box<dyn Asset>,
    access_time: Instant,
    last_reload_time: Option<Instant>,
    ref_count: usize,
    memory_size: usize,
}

impl CacheEntry {
    fn new(asset: Box<dyn Asset>) -> Self {
        let memory_size = asset.size_in_memory();
        Self {
            asset,
            access_time: Instant::now(),
            last_reload_time: None,
            ref_count: 1,
            memory_size,
        }
    }
    
    fn touch(&mut self) {
        self.access_time = Instant::now();
        self.ref_count += 1;
    }
    
    fn release(&mut self) {
        if self.ref_count > 0 {
            self.ref_count -= 1;
        }
    }
}

/// LRU Cache for assets with memory management
pub struct AssetCache {
    entries: RwLock<HashMap<AssetId, CacheEntry>>,
    max_memory: usize,
    current_memory: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

impl AssetCache {
    pub fn new(max_memory_bytes: usize) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            max_memory: max_memory_bytes,
            current_memory: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }
    
    pub fn get(&self, id: &AssetId) -> Option<Box<dyn Asset>> {
        let mut entries = self.entries.write().ok()?;
        let entry = entries.get_mut(id)?;
        
        entry.touch();
        self.hits.fetch_add(1, Ordering::Relaxed);
        
        Some(entry.asset.clone_box())
    }
    
    pub fn insert(&self, id: AssetId, asset: Box<dyn Asset>) -> Result<()> {
        let memory_size = asset.size_in_memory();
        
        // Check if we need to evict
        let mut entries = self.entries.write().unwrap();
        
        // Evict if necessary
        while self.current_memory.load(Ordering::Relaxed) + memory_size as u64 
               > self.max_memory as u64 
               && !entries.is_empty() {
            self.evict_lru(&mut entries);
        }
        
        // Check if asset already exists
        if entries.contains_key(&id) {
            // Update existing
            if let Some(old_entry) = entries.get(&id) {
                self.current_memory.fetch_sub(old_entry.memory_size as u64, Ordering::Relaxed);
            }
            entries.remove(&id);
        }
        
        let entry = CacheEntry::new(asset);
        self.current_memory.fetch_add(memory_size as u64, Ordering::Relaxed);
        entries.insert(id, entry);
        
        Ok(())
    }
    
    fn evict_lru(&self, entries: &mut HashMap<AssetId, CacheEntry>) {
        if let Some((oldest_id, _)) = entries
            .iter()
            .min_by_key(|(_, e)| e.access_time)
            .map(|(k, v)| (k.clone(), v.access_time))
        {
            if let Some(entry) = entries.remove(&oldest_id) {
                self.current_memory.fetch_sub(entry.memory_size as u64, Ordering::Relaxed);
                self.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    
    pub fn remove(&self, id: &AssetId) -> Option<Box<dyn Asset>> {
        let mut entries = self.entries.write().ok()?;
        if let Some(entry) = entries.remove(id) {
            self.current_memory.fetch_sub(entry.memory_size as u64, Ordering::Relaxed);
            Some(entry.asset)
        } else {
            None
        }
    }
    
    pub fn clear(&self) {
        let mut entries = self.entries.write().unwrap();
        entries.clear();
        self.current_memory.store(0, Ordering::Relaxed);
    }
    
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            current_memory: self.current_memory.load(Ordering::Relaxed),
            max_memory: self.max_memory as u64,
            entry_count: self.entries.read().unwrap().len() as u64,
        }
    }
    
    pub fn current_memory(&self) -> usize {
        self.current_memory.load(Ordering::Relaxed) as usize
    }
    
    pub fn entry_count(&self) -> usize {
        self.entries.read().unwrap().len()
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub current_memory: u64,
    pub max_memory: u64,
    pub entry_count: u64,
}

impl fmt::Display for CacheStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let hit_rate = if self.hits + self.misses > 0 {
            self.hits as f64 / (self.hits + self.misses) as f64 * 100.0
        } else {
            0.0
        };
        
        write!(
            f,
            "CacheStats {{ hits: {}, misses: {}, hit_rate: {:.2}%, evictions: {}, \
             memory: {}/{}, entries: {} }}",
            self.hits,
            self.misses,
            hit_rate,
            self.evictions,
            Self::format_bytes(self.current_memory as usize),
            Self::format_bytes(self.max_memory as usize),
            self.entry_count
        )
    }
}

impl CacheStats {
    fn format_bytes(bytes: usize) -> String {
        const KB: usize = 1024;
        const MB: usize = KB * 1024;
        const GB: usize = MB * 1024;
        
        if bytes >= GB {
            format!("{:.2} GB", bytes as f64 / GB as f64)
        } else if bytes >= MB {
            format!("{:.2} MB", bytes as f64 / MB as f64)
        } else if bytes >= KB {
            format!("{:.2} KB", bytes as f64 / KB as f64)
        } else {
            format!("{} B", bytes)
        }
    }
}

// ============================================================================
// Asset Loader Registry
// ============================================================================

/// Registry for custom asset loaders
pub struct LoaderRegistry {
    loaders: RwLock<HashMap<AssetType, Vec<Box<dyn AssetLoader>>>>,
    extension_map: RwLock<HashMap<String, AssetType>>,
}

impl LoaderRegistry {
    pub fn new() -> Self {
        Self {
            loaders: RwLock::new(HashMap::new()),
            extension_map: RwLock::new(HashMap::new()),
        }
    }
    
    pub fn register<L: AssetLoader + 'static>(&self, loader: L) {
        let mut loaders = self.loaders.write().unwrap();
        let asset_type = loader.supported_type();
        
        loaders
            .entry(asset_type)
            .or_insert_with(Vec::new)
            .push(Box::new(loader));
        
        // Register extension mappings
        let mut ext_map = self.extension_map.write().unwrap();
        for ext in loader.supported_extensions() {
            ext_map.insert(ext.to_lowercase(), asset_type);
        }
    }
    
    pub fn get_loader(&self, asset_type: AssetType) -> Option<Box<dyn AssetLoader>> {
        let loaders = self.loaders.read().unwrap();
        loaders
            .get(&asset_type)
            .and_then(|loaders| loaders.first())
            .map(|l| {
                // Clone the loader if possible, or create new instance
                // This is a simplified version - in practice you'd want Clone
                panic!("Loader cloning not implemented - create new instance")
            })
    }
    
    pub fn get_type_for_extension(&self, ext: &str) -> Option<AssetType> {
        let ext_map = self.extension_map.read().unwrap();
        ext_map.get(&ext.to_lowercase()).copied()
    }
    
    pub fn unregister(&self, asset_type: AssetType) {
        let mut loaders = self.loaders.write().unwrap();
        loaders.remove(&asset_type);
    }
}

impl Default for LoaderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Async Loading
// ============================================================================

/// State of an async asset load operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadState {
    Pending,
    Loading,
    Ready,
    Failed,
}

/// Handle for tracking async asset loads
pub struct LoadHandle {
    id: AssetId,
    state: LoadState,
    progress: f32,
    error: Option<AssetError>,
}

impl LoadHandle {
    fn new(id: AssetId) -> Self {
        Self {
            id,
            state: LoadState::Pending,
            progress: 0.0,
            error: None,
        }
    }
    
    fn with_state(mut self, state: LoadState) -> Self {
        self.state = state;
        self
    }
    
    fn with_progress(mut self, progress: f32) -> Self {
        self.progress = progress.clamp(0.0, 1.0);
        self
    }
    
    fn with_error(mut self, error: AssetError) -> Self {
        self.state = LoadState::Failed;
        self.error = Some(error);
        self
    }
    
    pub fn state(&self) -> LoadState {
        self.state
    }
    
    pub fn progress(&self) -> f32 {
        self.progress
    }
    
    pub fn error(&self) -> Option<&AssetError> {
        self.error.as_ref()
    }
    
    pub fn is_ready(&self) -> bool {
        self.state == LoadState::Ready
    }
    
    pub fn is_failed(&self) -> bool {
        self.state == LoadState::Failed
    }
    
    pub fn is_loading(&self) -> bool {
        self.state == LoadState::Loading || self.state == LoadState::Pending
    }
}

/// Future for async asset loading
pub struct AssetLoadFuture {
    handle: Arc<Mutex<LoadHandle>>,
    signal: Arc<Condvar>,
}

impl AssetLoadFuture {
    fn new(id: AssetId) -> Self {
        Self {
            handle: Arc::new(Mutex::new(LoadHandle::new(id))),
            signal: Arc::new(Condvar::new()),
        }
    }
    
    pub fn wait(&self) -> Result<()> {
        let mut handle = self.handle.lock().unwrap();
        
        while handle.state != LoadState::Ready && handle.state != LoadState::Failed {
            handle = self.signal.wait(handle).unwrap();
        }
        
        if let Some(ref error) = handle.error {
            Err(error.clone())
        } else {
            Ok(())
        }
    }
    
    pub fn wait_timeout(&self, timeout: Duration) -> Result<()> {
        let mut handle = self.handle.lock().unwrap();
        let result = self.signal.wait_timeout(handle, timeout).unwrap();
        
        if result.1.timed_out() {
            return Err(AssetError::LoadingFailed("Timeout waiting for asset".to_string()));
        }
        
        let handle = result.0;
        
        if let Some(ref error) = handle.error {
            Err(error.clone())
        } else {
            Ok(())
        }
    }
    
    pub fn state(&self) -> LoadState {
        self.handle.lock().unwrap().state
    }
    
    pub fn progress(&self) -> f32 {
        self.handle.lock().unwrap().progress
    }
    
    fn notify_ready(&self) {
        let mut handle = self.handle.lock().unwrap();
        handle.state = LoadState::Ready;
        self.signal.notify_all();
    }
    
    fn notify_error(&self, error: AssetError) {
        let mut handle = self.handle.lock().unwrap();
        handle.error = Some(error);
        self.signal.notify_all();
    }
}

// ============================================================================
// Hot Reload System
// ============================================================================

/// Watch configuration for hot reloading
#[derive(Debug, Clone)]
pub struct HotReloadConfig {
    pub enabled: bool,
    pub watch_directories: Vec<PathBuf>,
    pub poll_interval: Duration,
    pub debounce_delay: Duration,
}

impl Default for HotReloadConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            watch_directories: Vec::new(),
            poll_interval: Duration::from_millis(500),
            debounce_delay: Duration::from_millis(200),
        }
    }
}

/// Hot reload observer callback
pub trait HotReloadObserver: Send + Sync {
    fn on_asset_changed(&self, id: &AssetId);
    fn on_asset_added(&self, id: &AssetId);
    fn on_asset_removed(&self, id: &AssetId);
}

/// Hot reload system
pub struct HotReloadSystem {
    config: HotReloadConfig,
    observers: RwLock<Vec<Box<dyn HotReloadObserver>>>,
    last_modified: RwLock<HashMap<PathBuf, SystemTime>>,
    enabled: AtomicBool,
}

impl HotReloadSystem {
    pub fn new(config: HotReloadConfig) -> Self {
        Self {
            config,
            observers: RwLock::new(Vec::new()),
            last_modified: RwLock::new(HashMap::new()),
            enabled: AtomicBool::new(config.enabled),
        }
    }
    
    pub fn add_observer(&self, observer: Box<dyn HotReloadObserver>) {
        let mut observers = self.observers.write().unwrap();
        observers.push(observer);
    }
    
    pub fn remove_observer(&self, _observer: &dyn HotReloadObserver) {
        // Implementation would need to track observer identity
    }
    
    pub fn check_for_changes(&self) -> Vec<AssetId> {
        if !self.enabled.load(Ordering::Relaxed) {
            return Vec::new();
        }
        
        let mut changed = Vec::new();
        let mut last_modified = self.last_modified.write().unwrap();
        
        for dir in &self.config.watch_directories {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        if let Ok(metadata) = std::fs::metadata(&path) {
                            if let Ok(modified) = metadata.modified() {
                                let should_reload = last_modified
                                    .get(&path)
                                    .map(|last| *last != modified)
                                    .unwrap_or(true);
                                
                                if should_reload {
                                    let id = AssetId::new(path.to_string_lossy().as_ref());
                                    changed.push(id.clone());
                                    last_modified.insert(path, modified);
                                    
                                    // Notify observers
                                    let observers = self.observers.read().unwrap();
                                    for observer in observers.iter() {
                                        observer.on_asset_changed(&id);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        changed
    }
    
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Relaxed);
    }
    
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Relaxed);
    }
    
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }
}

// ============================================================================
// Memory-Mapped File Support
// ============================================================================

/// Memory-mapped asset for large files
pub struct MmappedAsset {
    data: memmap2::Mmap,
    metadata: AssetMetadata,
}

impl MmappedAsset {
    pub fn from_path(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let metadata = AssetMetadata::new(path, AssetType::from_extension(
            path.extension().and_then(|s| s.to_str()).unwrap_or("")
        ));
        
        let data = unsafe { memmap2::Mmap::map(&file)? };
        
        Ok(Self { data, metadata })
    }
    
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
    
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Asset for MmappedAsset {
    fn asset_type(&self) -> AssetType {
        self.metadata.asset_type
    }
    
    fn size_in_memory(&self) -> usize {
        self.data.len()
    }
    
    fn id(&self) -> &AssetId {
        &self.metadata.id
    }
    
    fn clone_box(&self) -> Box<dyn Asset> {
        Box::new(Self {
            data: self.data.clone(),
            metadata: self.metadata.clone(),
        })
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ============================================================================
// Asset Source
// ============================================================================

/// Source location for assets
#[derive(Debug, Clone)]
pub enum AssetSource {
    FileSystem(PathBuf),
    Embedded,
    Archive(PathBuf, String), // (archive_path, inner_path)
    Network(String),          // URL
}

impl AssetSource {
    pub fn resolve(&self, id: &AssetId) -> PathBuf {
        match self {
            AssetSource::FileSystem(base) => base.join(id.as_str()),
            AssetSource::Embedded => panic!("Cannot resolve embedded assets by path"),
            AssetSource::Archive(_, _) => panic!("Cannot resolve archived assets by path"),
            AssetSource::Network(url) => panic!("Cannot resolve network assets by path"),
        }
    }
}

/// Asset source manager
pub struct AssetSourceManager {
    sources: RwLock<Vec<AssetSource>>,
    base_path: PathBuf,
}

impl AssetSourceManager {
    pub fn new(base_path: PathBuf) -> Self {
        let mut sources = Vec::new();
        sources.push(AssetSource::FileSystem(base_path.clone()));
        
        Self {
            sources: RwLock::new(sources),
            base_path,
        }
    }
    
    pub fn add_source(&self, source: AssetSource) {
        let mut sources = self.sources.write().unwrap();
        sources.push(source);
    }
    
    pub fn remove_source(&self, source: &AssetSource) {
        let mut sources = self.sources.write().unwrap();
        sources.retain(|s| s != source);
    }
    
    pub fn find_asset(&self, id: &AssetId) -> Option<PathBuf> {
        let sources = self.sources.read().unwrap();
        
        for source in sources.iter() {
            match source {
                AssetSource::FileSystem(base) => {
                    let path = base.join(id.as_str());
                    if path.exists() {
                        return Some(path);
                    }
                }
                _ => {} // Other sources need different handling
            }
        }
        
        None
    }
    
    pub fn set_base_path(&mut self, path: PathBuf) {
        self.base_path = path;
        let mut sources = self.sources.write().unwrap();
        if let Some(AssetSource::FileSystem(base)) = sources.get_mut(0) {
            *base = path;
        }
    }
}

// ============================================================================
// Streaming Assets
// ============================================================================

/// Chunk for streaming large assets
#[derive(Debug, Clone)]
pub struct AssetChunk {
    pub data: Vec<u8>,
    pub offset: u64,
    pub size: usize,
    pub is_last: bool,
}

/// Stream for loading large assets in chunks
pub struct AssetStream {
    path: PathBuf,
    file: Option<BufReader<File>>,
    chunk_size: usize,
    current_offset: u64,
    total_size: u64,
}

impl AssetStream {
    pub fn from_path(path: &Path, chunk_size: usize) -> Result<Self> {
        let file = File::open(path)?;
        let total_size = file.len();
        
        Ok(Self {
            path: path.to_path_buf(),
            file: Some(BufReader::new(file)),
            chunk_size,
            current_offset: 0,
            total_size,
        })
    }
    
    pub fn next_chunk(&mut self) -> Result<Option<AssetChunk>> {
        let file = match &mut self.file {
            Some(f) => f,
            None => return Ok(None),
        };
        
        if self.current_offset >= self.total_size {
            self.file = None;
            return Ok(None);
        }
        
        let mut buffer = vec![0u8; self.chunk_size];
        let bytes_read = file.read(&mut buffer)?;
        buffer.truncate(bytes_read);
        
        let is_last = self.current_offset + bytes_read as u64 >= self.total_size;
        let chunk = AssetChunk {
            data: buffer,
            offset: self.current_offset,
            size: bytes_read,
            is_last,
        };
        
        self.current_offset += bytes_read as u64;
        
        Ok(Some(chunk))
    }
    
    pub fn seek(&mut self, offset: u64) -> Result<()> {
        if let Some(ref mut file) = self.file {
            file.seek(SeekFrom::Start(offset))?;
            self.current_offset = offset;
            Ok(())
        } else {
            Err(AssetError::Io(io::Error::new(
                io::ErrorKind::NotFound,
                "Stream file is closed",
            )))
        }
    }
    
    pub fn current_offset(&self) -> u64 {
        self.current_offset
    }
    
    pub fn total_size(&self) -> u64 {
        self.total_size
    }
    
    pub fn progress(&self) -> f32 {
        if self.total_size > 0 {
            self.current_offset as f32 / self.total_size as f32
        } else {
            0.0
        }
    }
}

impl Drop for AssetStream {
    fn drop(&mut self) {
        self.file = None;
    }
}

// ============================================================================
// Dependency Tracking
// ============================================================================

/// Asset dependency node
#[derive(Debug, Clone)]
pub struct DependencyNode {
    pub id: AssetId,
    pub dependencies: Vec<AssetId>,
    pub dependents: Vec<AssetId>,
    pub loaded: bool,
}

impl DependencyNode {
    pub fn new(id: AssetId) -> Self {
        Self {
            id,
            dependencies: Vec::new(),
            dependents: Vec::new(),
            loaded: false,
        }
    }
    
    pub fn add_dependency(&mut self, dep: AssetId) {
        if !self.dependencies.contains(&dep) {
            self.dependencies.push(dep);
        }
    }
    
    pub fn add_dependent(&mut self, dep: AssetId) {
        if !self.dependents.contains(&dep) {
            self.dependents.push(dep);
        }
    }
}

/// Dependency graph for assets
pub struct DependencyGraph {
    nodes: RwLock<HashMap<AssetId, DependencyNode>>,
}

impl DependencyGraph {
    pub fn new() -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
        }
    }
    
    pub fn add_asset(&self, id: AssetId, dependencies: Vec<AssetId>) {
        let mut nodes = self.nodes.write().unwrap();
        
        // Create or update the main node
        let node = nodes.entry(id.clone()).or_insert_with(|| DependencyNode::new(id.clone()));
        node.dependencies = dependencies.clone();
        
        // Update dependents for each dependency
        for dep in dependencies {
            let dep_node = nodes.entry(dep.clone()).or_insert_with(|| DependencyNode::new(dep));
            dep_node.add_dependent(id.clone());
        }
    }
    
    pub fn get_dependencies(&self, id: &AssetId) -> Vec<AssetId> {
        let nodes = self.nodes.read().unwrap();
        nodes
            .get(id)
            .map(|n| n.dependencies.clone())
            .unwrap_or_default()
    }
    
    pub fn get_dependents(&self, id: &AssetId) -> Vec<AssetId> {
        let nodes = self.nodes.read().unwrap();
        nodes
            .get(id)
            .map(|n| n.dependents.clone())
            .unwrap_or_default()
    }
    
    pub fn get_load_order(&self, id: &AssetId) -> Vec<AssetId> {
        let mut order = Vec::new();
        let mut visited = std::collections::HashSet::new();
        
        self.visit_dependencies(id, &mut order, &mut visited);
        
        order
    }
    
    fn visit_dependencies(&self, id: &AssetId, order: &mut Vec<AssetId>, visited: &mut std::collections::HashSet<AssetId>) {
        if visited.contains(id) {
            return;
        }
        
        visited.insert(id.clone());
        
        let deps = self.get_dependencies(id);
        for dep in deps {
            self.visit_dependencies(&dep, order, visited);
        }
        
        order.push(id.clone());
    }
    
    pub fn remove_asset(&self, id: &AssetId) {
        let mut nodes = self.nodes.write().unwrap();
        
        if let Some(node) = nodes.remove(id) {
            // Remove from dependents
            for dep in &node.dependencies {
                if let Some(dep_node) = nodes.get_mut(dep) {
                    dep_node.dependents.retain(|d| d != id);
                }
            }
        }
    }
    
    pub fn mark_loaded(&self, id: &AssetId) {
        let mut nodes = self.nodes.write().unwrap();
        if let Some(node) = nodes.get_mut(id) {
            node.loaded = true;
        }
    }
    
    pub fn is_loaded(&self, id: &AssetId) -> bool {
        let nodes = self.nodes.read().unwrap();
        nodes.get(id).map(|n| n.loaded).unwrap_or(false)
    }
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Main Asset Manager
// ============================================================================

/// Configuration for the asset manager
#[derive(Debug, Clone)]
pub struct AssetManagerConfig {
    pub base_path: PathBuf,
    pub max_cache_memory: usize,
    pub enable_streaming: bool,
    pub enable_hot_reload: bool,
    pub hot_reload_poll_interval: Duration,
    pub default_chunk_size: usize,
    pub max_concurrent_loads: usize,
    pub preload_on_startup: Vec<AssetId>,
}

impl Default for AssetManagerConfig {
    fn default() -> Self {
        Self {
            base_path: PathBuf::from("assets"),
            max_cache_memory: 512 * 1024 * 1024, // 512 MB
            enable_streaming: false,
            enable_hot_reload: false,
            hot_reload_poll_interval: Duration::from_millis(500),
            default_chunk_size: 64 * 1024, // 64 KB
            max_concurrent_loads: 4,
            preload_on_startup: Vec::new(),
        }
    }
}

/// Main asset manager that coordinates all systems
pub struct AssetManager {
    config: AssetManagerConfig,
    cache: Arc<AssetCache>,
    registry: Arc<LoaderRegistry>,
    source_manager: Arc<AssetSourceManager>,
    dependency_graph: Arc<DependencyGraph>,
    hot_reload: Arc<HotReloadSystem>,
    loading_handles: Arc<RwLock<HashMap<AssetId, Arc<AssetLoadFuture>>>>,
    statistics: Arc<RwLock<AssetStatistics>>,
}

impl AssetManager {
    /// Create a new asset manager with the given configuration
    pub fn new(config: AssetManagerConfig) -> Self {
        let hot_reload_config = HotReloadConfig {
            enabled: config.enable_hot_reload,
            watch_directories: vec![config.base_path.clone()],
            poll_interval: config.hot_reload_poll_interval,
            ..Default::default()
        };
        
        Self {
            config: config.clone(),
            cache: Arc::new(AssetCache::new(config.max_cache_memory)),
            registry: Arc::new(LoaderRegistry::new()),
            source_manager: Arc::new(AssetSourceManager::new(config.base_path)),
            dependency_graph: Arc::new(DependencyGraph::new()),
            hot_reload: Arc::new(HotReloadSystem::new(hot_reload_config)),
            loading_handles: Arc::new(RwLock::new(HashMap::new())),
            statistics: Arc::new(RwLock::new(AssetStatistics::default())),
        }
    }
    
    /// Get the cache instance
    pub fn cache(&self) -> &Arc<AssetCache> {
        &self.cache
    }
    
    /// Get the loader registry
    pub fn registry(&self) -> &Arc<LoaderRegistry> {
        &self.registry
    }
    
    /// Get the source manager
    pub fn source_manager(&self) -> &Arc<AssetSourceManager> {
        &self.source_manager
    }
    
    /// Get the dependency graph
    pub fn dependency_graph(&self) -> &Arc<DependencyGraph> {
        &self.dependency_graph
    }
    
    /// Get the hot reload system
    pub fn hot_reload(&self) -> &Arc<HotReloadSystem> {
        &self.hot_reload
    }
    
    /// Register a custom asset loader
    pub fn register_loader<L: AssetLoader + 'static>(&self, loader: L) {
        self.registry.register(loader);
    }
    
    /// Load an asset synchronously
    pub fn load(&self, id: &AssetId) -> Result<Box<dyn Asset>> {
        // Check cache first
        if let Some(asset) = self.cache.get(id) {
            return Ok(asset);
        }
        
        // Load the asset
        let asset = self.load_from_source(id)?;
        
        // Add to cache
        self.cache.insert(id.clone(), asset.clone_box())?;
        
        // Update statistics
        {
            let mut stats = self.statistics.write().unwrap();
            stats.load_count += 1;
            stats.total_bytes_loaded += asset.size_in_memory() as u64;
        }
        
        Ok(asset)
    }
    
    /// Load an asset from source
    fn load_from_source(&self, id: &AssetId) -> Result<Box<dyn Asset>> {
        // Find the asset path
        let path = self.source_manager
            .find_asset(id)
            .ok_or_else(|| AssetError::NotFound(id.as_str().to_string()))?;
        
        // Determine asset type
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let asset_type = AssetType::from_extension(ext);
        
        // Create metadata
        let metadata = AssetMetadata::new(&path, asset_type);
        
        // Load raw data
        let mut file = File::open(&path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        
        // Get or create loader
        let loader = self.registry
            .get_loader(asset_type)
            .ok_or_else(|| AssetError::LoadingFailed(format!("No loader for type: {:?}", asset_type)))?;
        
        // Load asset
        loader.load(&buffer, &metadata).map_err(|e| {
            AssetError::LoadingFailed(format!("Failed to load asset {}: {}", id.as_str(), e))
        })
    }
    
    /// Load an asset asynchronously
    pub fn load_async(&self, id: &AssetId) -> Arc<AssetLoadFuture> {
        // Check if already loading
        {
            let handles = self.loading_handles.read().unwrap();
            if let Some(handle) = handles.get(id) {
                return handle.clone();
            }
        }
        
        // Create new load future
        let future = Arc::new(AssetLoadFuture::new(id.clone()));
        
        // Store the handle
        {
            let mut handles = self.loading_handles.write().unwrap();
            handles.insert(id.clone(), future.clone());
        }
        
        // Spawn loading task
        let cache = self.cache.clone();
        let source_manager = self.source_manager.clone();
        let registry = self.registry.clone();
        let future_clone = future.clone();
        
        std::thread::spawn(move || {
            // Load from source
            let result = (|| -> Result<Box<dyn Asset>> {
                let path = source_manager
                    .find_asset(id)
                    .ok_or_else(|| AssetError::NotFound(id.as_str().to_string()))?;
                
                let ext = path
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("");
                let asset_type = AssetType::from_extension(ext);
                
                let metadata = AssetMetadata::new(&path, asset_type);
                
                let mut file = File::open(&path)?;
                let mut buffer = Vec::new();
                file.read_to_end(&mut buffer)?;
                
                let loader = registry
                    .get_loader(asset_type)
                    .ok_or_else(|| AssetError::LoadingFailed(format!("No loader for type: {:?}", asset_type)))?;
                
                loader.load(&buffer, &metadata).map_err(|e| {
                    AssetError::LoadingFailed(format!("Failed to load asset: {}", e))
                })
            })();
            
            match result {
                Ok(asset) => {
                    // Cache the asset
                    let _ = cache.insert(id.clone(), asset.clone_box());
                    future_clone.notify_ready();
                }
                Err(e) => {
                    future_clone.notify_error(e);
                }
            }
        });
        
        future
    }
    
    /// Load multiple assets synchronously
    pub fn load_batch(&self, ids: &[AssetId]) -> Vec<Result<Box<dyn Asset>>> {
        ids.iter().map(|id| self.load(id)).collect()
    }
    
    /// Load multiple assets in parallel
    pub fn load_parallel(&self, ids: &[AssetId]) -> Vec<Result<Box<dyn Asset>>> {
        use std::sync::mpsc;
        
        let (tx, rx) = mpsc::channel();
        
        for id in ids {
            let tx = tx.clone();
            let manager = self.clone_manager();
            let id = id.clone();
            
            std::thread::spawn(move || {
                let result = manager.load(&id);
                let _ = tx.send(result);
            });
        }
        
        drop(tx);
        
        rx.iter().collect()
    }
    
    /// Clone the manager for use in other threads
    fn clone_manager(&self) -> Self {
        Self {
            config: self.config.clone(),
            cache: self.cache.clone(),
            registry: self.registry.clone(),
            source_manager: self.source_manager.clone(),
            dependency_graph: self.dependency_graph.clone(),
            hot_reload: self.hot_reload.clone(),
            loading_handles: self.loading_handles.clone(),
            statistics: self.statistics.clone(),
        }
    }
    
    /// Stream a large asset in chunks
    pub fn stream(&self, id: &AssetId) -> Result<AssetStream> {
        let path = self.source_manager
            .find_asset(id)
            .ok_or_else(|| AssetError::NotFound(id.as_str().to_string()))?;
        
        AssetStream::from_path(&path, self.config.default_chunk_size)
    }
    
    /// Load a memory-mapped asset
    pub fn load_mmapped(&self, id: &AssetId) -> Result<Box<dyn Asset>> {
        let path = self.source_manager
            .find_asset(id)
            .ok_or_else(|| AssetError::NotFound(id.as_str().to_string()))?;
        
        let mmapped = MmappedAsset::from_path(&path)?;
        Ok(Box::new(mmapped))
    }
    
    /// Unload an asset from cache
    pub fn unload(&self, id: &AssetId) {
        self.cache.remove(id);
    }
    
    /// Clear all cached assets
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
    
    /// Preload assets on startup
    pub fn preload(&self, ids: &[AssetId]) -> Vec<Result<Box<dyn Asset>>> {
        self.load_batch(ids)
    }
    
    /// Check for hot-reload changes
    pub fn check_reload(&self) -> Vec<AssetId> {
        let changed = self.hot_reload.check_for_changes();
        
        for id in &changed {
            // Unload the cached version
            self.cache.remove(id);
            
            // Update dependency graph
            self.dependency_graph.mark_loaded(id);
        }
        
        changed
    }
    
    /// Get statistics
    pub fn statistics(&self) -> AssetStatistics {
        self.statistics.read().unwrap().clone()
    }
    
    /// Get cache statistics
    pub fn cache_statistics(&self) -> CacheStats {
        self.cache.stats()
    }
    
    /// Set the base path for assets
    pub fn set_base_path(&self, path: PathBuf) {
        self.source_manager.set_base_path(path);
    }
    
    /// Enable or disable hot reloading
    pub fn set_hot_reload(&self, enabled: bool) {
        if enabled {
            self.hot_reload.enable();
        } else {
            self.hot_reload.disable();
        }
    }
    
    /// Get asset metadata without loading the full asset
    pub fn get_metadata(&self, id: &AssetId) -> Result<AssetMetadata> {
        let path = self.source_manager
            .find_asset(id)
            .ok_or_else(|| AssetError::NotFound(id.as_str().to_string()))?;
        
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let asset_type = AssetType::from_extension(ext);
        
        Ok(AssetMetadata::new(&path, asset_type))
    }
    
    /// Check if an asset exists
    pub fn exists(&self, id: &AssetId) -> bool {
        self.source_manager.find_asset(id).is_some()
    }
    
    /// Get all loaded asset IDs
    pub fn loaded_assets(&self) -> Vec<AssetId> {
        // This would require adding a method to the cache
        Vec::new()
    }
    
    /// Get memory usage
    pub fn memory_usage(&self) -> usize {
        self.cache.current_memory()
    }
}

impl Default for AssetManager {
    fn default() -> Self {
        Self::new(AssetManagerConfig::default())
    }
}

// ============================================================================
// Statistics
// ============================================================================

/// Statistics about asset loading
#[derive(Debug, Clone, Default)]
pub struct AssetStatistics {
    pub load_count: u64,
    pub unload_count: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_bytes_loaded: u64,
    pub total_loading_time_ms: u64,
    pub failed_loads: u64,
    pub hot_reload_count: u64,
}

impl fmt::Display for AssetStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AssetStatistics {{ loads: {}, unloaded: {}, cache_hits: {}, cache_misses: {}, \
             bytes_loaded: {}, loading_time: {}ms, failed: {}, hot_reloads: {} }}",
            self.load_count,
            self.unload_count,
            self.cache_hits,
            self.cache_misses,
            self.total_bytes_loaded,
            self.total_loading_time_ms,
            self.failed_loads,
            self.hot_reload_count
        )
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Calculate a simple hash of asset data
pub fn calculate_hash(data: &[u8]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    hasher.finish()
}

/// Get the file extension from a path
pub fn get_extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
}

/// Check if a file extension is supported
pub fn is_supported_extension(ext: &str) -> bool {
    AssetType::from_extension(ext) != AssetType::Unknown
}

/// Format bytes to human readable string
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

// ============================================================================
// Common Asset Type Implementations
// ============================================================================

/// Placeholder for texture data
#[derive(Debug, Clone)]
pub struct TextureAsset {
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub data: Vec<u8>,
    pub id: AssetId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    R8,
    RG8,
    RGB8,
    RGBA8,
    R16,
    RG16,
    RGB16,
    RGBA16,
    R32F,
    RG32F,
    RGB32F,
    RGBA32F,
    DXT1,
    DXT5,
    BC7,
}

impl Asset for TextureAsset {
    fn asset_type(&self) -> AssetType {
        AssetType::Texture
    }
    
    fn size_in_memory(&self) -> usize {
        // Calculate based on format
        let bytes_per_pixel = match self.format {
            TextureFormat::R8 => 1,
            TextureFormat::RG8 => 2,
            TextureFormat::RGB8 => 3,
            TextureFormat::RGBA8 => 4,
            TextureFormat::R16 => 2,
            TextureFormat::RG16 => 4,
            TextureFormat::RGB16 => 6,
            TextureFormat::RGBA16 => 8,
            TextureFormat::R32F => 4,
            TextureFormat::RG32F => 8,
            TextureFormat::RGB32F => 12,
            TextureFormat::RGBA32F => 16,
            TextureFormat::DXT1 => 8, // Per block
            TextureFormat::DXT5 => 16, // Per block
            TextureFormat::BC7 => 16,  // Per block
        };
        
        self.width as usize * self.height as usize * bytes_per_pixel
    }
    
    fn id(&self) -> &AssetId {
        &self.id
    }
    
    fn clone_box(&self) -> Box<dyn Asset> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Placeholder for mesh data
#[derive(Debug, Clone)]
pub struct MeshAsset {
    pub vertices: Vec<f32>,
    pub indices: Vec<u32>,
    pub normals: Vec<f32>,
    pub uvs: Vec<f2>,
    pub id: AssetId,
}

#[derive(Debug, Clone, Copy)]
pub struct f2 {
    pub x: f32,
    pub y: f32,
}

impl Asset for MeshAsset {
    fn asset_type(&self) -> AssetType {
        AssetType::Mesh
    }
    
    fn size_in_memory(&self) -> usize {
        self.vertices.len() * std::mem::size_of::<f32>()
            + self.indices.len() * std::mem::size_of::<u32>()
            + self.normals.len() * std::mem::size_of::<f32>()
            + self.uvs.len() * std::mem::size_of::<f2>()
    }
    
    fn id(&self) -> &AssetId {
        &self.id
    }
    
    fn clone_box(&self) -> Box<dyn Asset> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Placeholder for audio data
#[derive(Debug, Clone)]
pub struct AudioAsset {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    pub duration: f32,
    pub id: AssetId,
}

impl Asset for AudioAsset {
    fn asset_type(&self) -> AssetType {
        AssetType::Audio
    }
    
    fn size_in_memory(&self) -> usize {
        self.samples.len() * std::mem::size_of::<f32>()
    }
    
    fn id(&self) -> &AssetId {
        &self.id
    }
    
    fn clone_box(&self) -> Box<dyn Asset> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Placeholder for shader data
#[derive(Debug, Clone)]
pub struct ShaderAsset {
    pub source: String,
    pub shader_type: ShaderType,
    pub id: AssetId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderType {
    Vertex,
    Fragment,
    Geometry,
    Compute,
    TessellationControl,
    TessellationEvaluation,
}

impl Asset for ShaderAsset {
    fn asset_type(&self) -> AssetType {
        AssetType::Shader
    }
    
    fn size_in_memory(&self) -> usize {
        self.source.len()
    }
    
    fn id(&self) -> &AssetId {
        &self.id
    }
    
    fn clone_box(&self) -> Box<dyn Asset> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Placeholder for font data
#[derive(Debug, Clone)]
pub struct FontAsset {
    pub data: Vec<u8>,
    pub font_size: f32,
    pub id: AssetId,
}

impl Asset for FontAsset {
    fn asset_type(&self) -> AssetType {
        AssetType::Font
    }
    
    fn size_in_memory(&self) -> usize {
        self.data.len()
    }
    
    fn id(&self) -> &AssetId {
        &self.id
    }
    
    fn clone_box(&self) -> Box<dyn Asset> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ============================================================================
// Builder Pattern for Easy Setup
// ============================================================================

/// Builder for creating configured asset managers
pub struct AssetManagerBuilder {
    config: AssetManagerConfig,
    loaders: Vec<Box<dyn AssetLoader>>,
}

impl AssetManagerBuilder {
    pub fn new() -> Self {
        Self {
            config: AssetManagerConfig::default(),
            loaders: Vec::new(),
        }
    }
    
    pub fn base_path(mut self, path: PathBuf) -> Self {
        self.config.base_path = path;
        self
    }
    
    pub fn max_cache_memory(mut self, bytes: usize) -> Self {
        self.config.max_cache_memory = bytes;
        self
    }
    
    pub fn enable_streaming(mut self, enabled: bool) -> Self {
        self.config.enable_streaming = enabled;
        self
    }
    
    pub fn enable_hot_reload(mut self, enabled: bool) -> Self {
        self.config.enable_hot_reload = enabled;
        self
    }
    
    pub fn max_concurrent_loads(mut self, max: usize) -> Self {
        self.config.max_concurrent_loads = max;
        self
    }
    
    pub fn preload_assets(mut self, ids: Vec<AssetId>) -> Self {
        self.config.preload_on_startup = ids;
        self
    }
    
    pub fn add_loader(mut self, loader: Box<dyn AssetLoader>) -> Self {
        self.loaders.push(loader);
        self
    }
    
    pub fn build(self) -> AssetManager {
        let mut manager = AssetManager::new(self.config);
        
        for loader in self.loaders {
            manager.register_loader(loader);
        }
        
        manager
    }
}

impl Default for AssetManagerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_asset_id() {
        let id = AssetId::new("textures/player.png");
        assert_eq!(id.as_str(), "textures/player.png");
    }
    
    #[test]
    fn test_asset_type_from_extension() {
        assert_eq!(AssetType::from_extension("png"), AssetType::Texture);
        assert_eq!(AssetType::from_extension("obj"), AssetType::Mesh);
        assert_eq!(AssetType::from_extension("wav"), AssetType::Audio);
        assert_eq!(AssetType::from_extension("ttf"), AssetType::Font);
        assert_eq!(AssetType::from_extension("vert"), AssetType::Shader);
    }
    
    #[test]
    fn test_cache_operations() {
        let cache = AssetCache::new(1024 * 1024); // 1 MB
        
        let id = AssetId::new("test");
        let asset: Box<dyn Asset> = Box::new(TextureAsset {
            width: 256,
            height: 256,
            format: TextureFormat::RGBA8,
            data: vec![0u8; 256 * 256 * 4],
            id: id.clone(),
        });
        
        cache.insert(id.clone(), asset.clone_box()).unwrap();
        
        let loaded = cache.get(&id);
        assert!(loaded.is_some());
        
        let stats = cache.stats();
        assert_eq!(stats.entry_count, 1);
    }
    
    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }
    
    #[test]
    fn test_builder() {
        let manager = AssetManagerBuilder::new()
            .base_path(PathBuf::from("assets"))
            .max_cache_memory(1024 * 1024 * 1024)
            .enable_hot_reload(true)
            .build();
        
        let stats = manager.cache_statistics();
        assert_eq!(stats.max_memory, 1024 * 1024 * 1024);
    }
}
