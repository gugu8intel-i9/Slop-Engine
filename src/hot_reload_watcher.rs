// src/hot_reload_watcher.rs - File watcher for assets, scripts, shaders
use std::collections::HashMap;
use parking_lot::RwLock;
use notify::{Watcher, RecursiveMode, watcher};

pub struct HotReloadWatcher {
    watcher: RwLock<Option<notify::RecommendedWatcher>>,
    watched_paths: RwLock<HashMap<String, Box<dyn Fn(&str) + Send + Sync>>>,
}

impl HotReloadWatcher {
    pub fn new() -> Self { Self { watcher: RwLock::new(None), watched_paths: RwLock::new(HashMap::new()) } }
    pub fn watch<F>(&self, path: &str, callback: F) where F: Fn(&str) + Send + Sync + 'static {
        self.watched_paths.write().insert(path.to_string(), Box::new(callback));
    }
}
