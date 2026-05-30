// src/virtual_fs.rs - Mount asset packs, override layers, mod support
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use parking_lot::RwLock;

pub struct VirtualFileSystem {
    mounts: RwLock<HashMap<String, MountPoint>>,
    overlays: RwLock<Vec<PathBuf>>,
}

struct MountPoint { path: PathBuf, priority: u32, pack_type: PackType }
#[derive(Clone)] enum PackType { Directory, Zip, Pak }

impl VirtualFileSystem {
    pub fn new() -> Self { Self { mounts: RwLock::new(HashMap::new()), overlays: RwLock::new(Vec::new()) } }
    pub fn mount<P: AsRef<Path>>(&self, virtual_path: &str, real_path: P, priority: u32) {
        self.mounts.write().insert(virtual_path.to_string(), MountPoint { path: real_path.as_ref().to_path_buf(), priority, pack_type: PackType::Directory });
    }
}
