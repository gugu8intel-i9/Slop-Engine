// src/shader_hot_reload.rs
//! Shader Hot Reload system
//! - Watches WGSL files
//! - Validates and creates shader modules off-thread
//! - Atomically swaps pipelines on main thread
//! - Optional error overlay texture
//!
//! Usage:
//!  let mut hot = ShaderHotReload::new(device.clone(), queue.clone(), config);
//!  hot.register_pipeline("pbr", desc);
//!  // each frame:
///   hot.poll(); // apply updates on main thread
///   let pipeline = hot.get_pipeline("pbr").unwrap();

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use crossbeam_channel::{unbounded, Receiver, Sender};
use notify::{Config as NotifyConfig, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use parking_lot::RwLock;
use wgpu::util::DeviceExt;

/// Configuration
pub struct ShaderHotReloadConfig {
    /// Debounce window for file changes (ms)
    pub debounce_ms: u64,
    /// Whether to show compile errors in an overlay texture
    pub enable_error_overlay: bool,
    /// Path roots allowed for includes
    pub include_roots: Vec<PathBuf>,
}

impl Default for ShaderHotReloadConfig {
    fn default() -> Self {
        Self {
            debounce_ms: 150,
            enable_error_overlay: true,
            include_roots: vec![std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))],
        }
    }
}

/// Shader module pair (vertex/fragment or compute)
pub struct ShaderModules {
    pub vertex: Option<wgpu::ShaderModule>,
    pub fragment: Option<wgpu::ShaderModule>,
    pub compute: Option<wgpu::ShaderModule>,
}

/// A pipeline description provided by the engine. The closure builds a pipeline from shader modules.
/// The closure runs on the main thread when swapping pipelines.
pub struct ShaderPipelineDesc {
    pub vertex_path: Option<PathBuf>,
    pub fragment_path: Option<PathBuf>,
    pub compute_path: Option<PathBuf>,
    /// Bind group layouts expected by the pipeline (used to detect layout changes)
    pub bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    /// Builder closure: (device, pipeline_layout, shader_modules) -> RenderPipeline or ComputePipeline wrapper
    pub pipeline_builder: Box<dyn Fn(&wgpu::Device, &wgpu::PipelineLayout, &ShaderModules) -> Result<PipelineVariant> + Send + Sync>,
}

/// PipelineVariant wraps either a render or compute pipeline
pub enum PipelineVariant {
    Render(wgpu::RenderPipeline),
    Compute(wgpu::ComputePipeline),
}

/// Internal record for a registered pipeline
struct PipelineRecord {
    desc: ShaderPipelineDesc,
    /// current active pipeline (atomic swap)
    active: RwLock<Option<Arc<PipelineVariant>>>,
    /// last successful shader sources (for change detection)
    last_sources: RwLock<HashMap<PathBuf, String>>,
    /// last compile error (if any)
    last_error: RwLock<Option<String>>,
}

/// Messages from file watcher to worker
enum WatchMsg {
    FileChanged(PathBuf),
    FileRemoved(PathBuf),
}

/// Messages from worker to main thread
enum WorkerMsg {
    PipelineCompiled {
        name: String,
        modules: ShaderModules,
        sources: HashMap<PathBuf, String>,
    },
    PipelineCompileError {
        name: String,
        error: String,
        sources: HashMap<PathBuf, String>,
    },
}

/// The hot reload manager
pub struct ShaderHotReload {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: ShaderHotReloadConfig,

    // registered pipelines
    pipelines: RwLock<HashMap<String, Arc<PipelineRecord>>>,

    // file watcher
    watcher: RecommendedWatcher,
    watch_tx: Sender<WatchMsg>,

    // worker thread channel
    worker_tx: Sender<(String, HashSet<PathBuf>)>,
    worker_rx: Receiver<WorkerMsg>,

    // debounce state
    pending: RwLock<HashMap<PathBuf, Instant>>,

    // overlay resources (optional)
    overlay_texture: RwLock<Option<wgpu::TextureView>>,
}

impl ShaderHotReload {
    /// Create manager. Must be called on main thread where `device` is valid.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, config: ShaderHotReloadConfig) -> Self {
        // channels
        let (watch_tx, watch_rx) = unbounded::<WatchMsg>();
        let (worker_tx, worker_rx_in) = unbounded::<(String, HashSet<PathBuf>)>();
        let (worker_tx_back, worker_rx) = unbounded::<WorkerMsg>();

        // create notify watcher
        let mut watcher: RecommendedWatcher = Watcher::new_immediate(move |res: notify::Result<Event>| {
            match res {
                Ok(event) => {
                    // send events to watch_tx
                    for path in event.paths {
                        let kind = &event.kind;
                        match kind {
                            EventKind::Modify(_) | EventKind::Create(_) => {
                                let _ = watch_tx.send(WatchMsg::FileChanged(path.clone()));
                            }
                            EventKind::Remove(_) => {
                                let _ = watch_tx.send(WatchMsg::FileRemoved(path.clone()));
                            }
                            _ => {}
                        }
                    }
                }
                Err(e) => {
                    log::warn!("notify error: {:?}", e);
                }
            }
        }).expect("failed to create file watcher");

        // spawn worker thread that listens for compile requests
        {
            let device_clone = device.clone();
            let queue_clone = queue.clone();
            let config_clone = config.clone();
            let worker_rx_in = worker_rx_in.clone();
            let worker_tx_back = worker_tx_back.clone();
            std::thread::spawn(move || {
                // worker loop
                while let Ok((name, paths)) = worker_rx_in.recv() {
                    // read and preprocess sources (includes)
                    let mut sources = HashMap::new();
                    let mut ok = true;
                    let mut error_msg = String::new();
                    let mut vertex_src = None;
                    let mut fragment_src = None;
                    let mut compute_src = None;

                    for p in &paths {
                        match std::fs::read_to_string(p) {
                            Ok(s) => {
                                // preprocess includes
                                match preprocess_includes(p, &s, &config_clone.include_roots) {
                                    Ok(pre) => {
                                        sources.insert(p.clone(), pre.clone());
                                        // assign to vertex/fragment/compute based on extension or name
                                        if p.extension().and_then(|e| e.to_str()) == Some("vert.wgsl") || p.file_name().and_then(|n| n.to_str()).map(|n| n.contains("vert")).unwrap_or(false) {
                                            vertex_src = Some(pre);
                                        } else if p.extension().and_then(|e| e.to_str()) == Some("frag.wgsl") || p.file_name().and_then(|n| n.to_str()).map(|n| n.contains("frag")).unwrap_or(false) {
                                            fragment_src = Some(pre);
                                        } else if p.extension().and_then(|e| e.to_str()) == Some("comp.wgsl") || p.file_name().and_then(|n| n.to_str()).map(|n| n.contains("comp")).unwrap_or(false) {
                                            compute_src = Some(pre);
                                        } else {
                                            // fallback: if only one file, treat as both vertex/fragment if needed
                                            // leave assignment to pipeline desc later
                                        }
                                    }
                                    Err(e) => {
                                        ok = false;
                                        error_msg = format!("Include preprocessing failed for {:?}: {}", p, e);
                                        break;
                                    }
                                }
                            }
                            Err(e) => {
                                ok = false;
                                error_msg = format!("Failed to read {:?}: {}", p, e);
                                break;
                            }
                        }
                    }

                    if !ok {
                        let _ = worker_tx_back.send(WorkerMsg::PipelineCompileError {
                            name: name.clone(),
                            error: error_msg,
                            sources,
                        });
                        continue;
                    }

                    // create shader modules on a temporary device context by using the same device (safe)
                    // Note: wgpu shader module creation is cheap and validates WGSL; errors will be returned as panic? No — create_shader_module returns module but validation errors appear at pipeline creation time.
                    // We'll attempt to create modules and then send them back; pipeline creation will be done on main thread to ensure safety.
                    let mut modules = ShaderModules { vertex: None, fragment: None, compute: None };
                    if let Some(vs) = vertex_src {
                        // create module
                        let m = device_clone.create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some(&format!("hot_v_{}", name)),
                            source: wgpu::ShaderSource::Wgsl(vs.into()),
                        });
                        modules.vertex = Some(m);
                    }
                    if let Some(fs) = fragment_src {
                        let m = device_clone.create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some(&format!("hot_f_{}", name)),
                            source: wgpu::ShaderSource::Wgsl(fs.into()),
                        });
                        modules.fragment = Some(m);
                    }
                    if let Some(cs) = compute_src {
                        let m = device_clone.create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some(&format!("hot_c_{}", name)),
                            source: wgpu::ShaderSource::Wgsl(cs.into()),
                        });
                        modules.compute = Some(m);
                    }

                    // send compiled modules back to main thread
                    let _ = worker_tx_back.send(WorkerMsg::PipelineCompiled {
                        name: name.clone(),
                        modules,
                        sources,
                    });
                }
            });
        }

        // spawn a small thread to debounce file events and forward compile requests to worker_tx
        {
            let worker_tx_clone = worker_tx.clone();
            let watch_rx_clone = watch_rx.clone();
            let debounce_ms = config.debounce_ms;
            std::thread::spawn(move || {
                let mut pending: HashMap<PathBuf, Instant> = HashMap::new();
                loop {
                    // block for a short time to collect events
                    if let Ok(msg) = watch_rx_clone.recv_timeout(Duration::from_millis(50)) {
                        match msg {
                            WatchMsg::FileChanged(p) => {
                                pending.insert(p, Instant::now());
                            }
                            WatchMsg::FileRemoved(p) => {
                                pending.insert(p, Instant::now());
                            }
                        }
                    }
                    // check pending and if any older than debounce_ms, send compile request
                    let now = Instant::now();
                    let mut to_send: HashSet<PathBuf> = HashSet::new();
                    let mut remove = Vec::new();
                    for (p, t) in &pending {
                        if now.duration_since(*t).as_millis() as u64 >= debounce_ms {
                            to_send.insert(p.clone());
                            remove.push(p.clone());
                        }
                    }
                    for r in remove { pending.remove(&r); }
                    if !to_send.is_empty() {
                        // group by pipeline name: naive approach: send each path as its own pipeline name "file::<path>"
                        // Better: engine registers pipelines and we map files -> pipelines. For simplicity, send a generic name "hot" and include all paths.
                        let _ = worker_tx_clone.send(("hot".to_string(), to_send));
                    }
                }
            });
        }

        // create manager
        let manager = Self {
            device,
            queue,
            config,
            pipelines: RwLock::new(HashMap::new()),
            watcher,
            watch_tx: watch_tx.clone(),
            worker_tx,
            worker_rx,
            pending: RwLock::new(HashMap::new()),
            overlay_texture: RwLock::new(None),
        };

        manager
    }

    /// Register a pipeline for hot reload. `name` must be unique.
    /// This registers files to watch and stores the pipeline builder.
    pub fn register_pipeline(&self, name: &str, desc: ShaderPipelineDesc) -> Result<()> {
        let name = name.to_string();
        let rec = PipelineRecord {
            desc,
            active: RwLock::new(None),
            last_sources: RwLock::new(HashMap::new()),
            last_error: RwLock::new(None),
        };
        // watch files
        let mut watch_paths = HashSet::new();
        if let Some(p) = &rec.desc.vertex_path { watch_paths.insert(p.clone()); }
        if let Some(p) = &rec.desc.fragment_path { watch_paths.insert(p.clone()); }
        if let Some(p) = &rec.desc.compute_path { watch_paths.insert(p.clone()); }
        for p in &watch_paths {
            // ensure parent exists
            if let Some(parent) = p.parent() {
                if let Err(e) = self.watcher.watch(parent, RecursiveMode::NonRecursive) {
                    log::warn!("watch failed for {:?}: {:?}", parent, e);
                }
            }
        }

        self.pipelines.write().insert(name.clone(), Arc::new(rec));
        Ok(())
    }

    /// Poll must be called on the main thread each frame. It applies compiled pipelines and errors.
    pub fn poll(&self) {
        // process worker messages
        while let Ok(msg) = self.worker_rx.try_recv() {
            match msg {
                WorkerMsg::PipelineCompiled { name, modules, sources } => {
                    // find matching pipeline(s) — here we match all registered pipelines that reference any of the source paths
                    // For simplicity, we attempt to rebuild all pipelines; in a real engine map files->pipelines
                    for (pname, prec) in self.pipelines.read().iter() {
                        // attempt to create pipeline layout and pipeline using the builder
                        let layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some(&format!("hot_layout_{}", pname)),
                            bind_group_layouts: &prec.desc.bind_group_layouts.iter().collect::<Vec<_>>(),
                            push_constant_ranges: &[],
                        });
                        // call builder
                        match (prec.desc.pipeline_builder)(&self.device, &layout, &modules) {
                            Ok(variant) => {
                                // wrap and swap
                                let arc = Arc::new(variant);
                                *prec.active.write() = Some(arc.clone());
                                // clear last error
                                *prec.last_error.write() = None;
                                // store sources
                                *prec.last_sources.write() = sources.clone();
                                log::info!("Hot-reloaded pipeline '{}'", pname);
                            }
                            Err(e) => {
                                let err = format!("Pipeline build failed for '{}': {:?}", pname, e);
                                *prec.last_error.write() = Some(err.clone());
                                log::error!("{}", err);
                                // optionally create overlay texture with error text
                                if self.config.enable_error_overlay {
                                    let _ = self.create_error_overlay(&err);
                                }
                            }
                        }
                    }
                }
                WorkerMsg::PipelineCompileError { name: _name, error, sources } => {
                    // set last_error on all pipelines (simpler mapping)
                    for (_pname, prec) in self.pipelines.read().iter() {
                        *prec.last_error.write() = Some(error.clone());
                        *prec.last_sources.write() = sources.clone();
                        if self.config.enable_error_overlay {
                            let _ = self.create_error_overlay(&error);
                        }
                    }
                }
            }
        }
    }

    /// Get current pipeline by name (if compiled). Returns Arc to pipeline variant.
    pub fn get_pipeline(&self, name: &str) -> Option<Arc<PipelineVariant>> {
        self.pipelines.read().get(name).and_then(|r| r.active.read().clone())
    }

    /// Create an on-screen overlay texture containing the error text.
    /// This is a simple implementation: render text into a CPU image (via tiny-skia or system font) and upload as RGBA8 texture.
    /// For brevity we implement a minimal placeholder that uploads a 1x1 red pixel when error exists.
    fn create_error_overlay(&self, _error: &str) -> Result<()> {
        // Minimal: create 1x1 red texture view and store it
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shader_error_overlay"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let data = [255u8, 0u8, 0u8, 255u8];
        self.queue.write_texture(
            wgpu::ImageCopyTexture { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &data,
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: NonZeroU32::new(4), rows_per_image: NonZeroU32::new(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        *self.overlay_texture.write() = Some(view);
        Ok(())
    }
}

/// Preprocess includes: very small include system that replaces lines like `#include "file.wgsl"`
/// with the contents of the included file. Prevents recursive includes and supports include roots.
fn preprocess_includes(path: &Path, src: &str, include_roots: &[PathBuf]) -> Result<String> {
    let mut out = String::new();
    let mut stack = vec![(path.to_path_buf(), src.to_string())];
    let mut visited = HashSet::new();

    while let Some((p, s)) = stack.pop() {
        if !visited.insert(p.clone()) {
            return Err(anyhow!("Recursive include detected: {:?}", p));
        }
        for line in s.lines() {
            if let Some(rest) = line.trim_start().strip_prefix("#include") {
                // parse "file"
                let rest = rest.trim();
                if rest.starts_with('"') && rest.ends_with('"') {
                    let inner = &rest[1..rest.len()-1];
                    // search include roots
                    let mut found = None;
                    for root in include_roots {
                        let candidate = root.join(inner);
                        if candidate.exists() {
                            found = Some(candidate);
                            break;
                        }
                    }
                    if let Some(f) = found {
                        let content = std::fs::read_to_string(&f).with_context(|| format!("failed to read include {:?}", f))?;
                        // push current remainder back and process include first (depth-first)
                        stack.push((p.clone(), String::new())); // placeholder to keep visited semantics
                        stack.push((f, content));
                        break;
                    } else {
                        return Err(anyhow!("Include not found: {}", inner));
                    }
                } else {
                    return Err(anyhow!("Malformed include directive: {}", line));
                }
            } else {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    Ok(out)
}
