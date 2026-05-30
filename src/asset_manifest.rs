// src/asset_manifest.rs - Tracks asset dependencies + versioning
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AssetManifest { version: u32, assets: HashMap<String, AssetEntry> }

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AssetEntry { path: String, hash: String, size: u64, dependencies: Vec<String>, load_order: u32 }

impl AssetManifest {
    pub fn new() -> Self { Self { version: 1, assets: HashMap::new() } }
    pub fn add_asset(&mut self, path: &str, hash: &str, size: u64) {
        self.assets.insert(path.to_string(), AssetEntry { path: path.to_string(), hash: hash.to_string(), size, dependencies: vec![], load_order: 0 });
    }
}
