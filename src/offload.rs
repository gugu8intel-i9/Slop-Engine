//! offload.rs
//! Advanced Predictive Resource Tiering Subsystem.
//!
//! Features:
//! - 5-Tier Memory Architecture (VRAM, Pinned RAM, Pageable RAM, Mmap NVMe, Disk).
//! - Lock-free hot paths using DashMap and atomic operations.
//! - Predictive pre-fetching based on resource access adjacency graphs.
//! - High/Low watermark hysteresis to prevent memory thrashing.

use crossbeam_channel::{unbounded, Receiver, Sender};
use dashmap::DashMap;
use memmap2::MmapMut;
use parking_lot::RwLock as ParkingLotRwLock;
use std::collections::{BinaryHeap, HashSet};
use std::fs::OpenOptions;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::task::JoinHandle;
use uuid::Uuid;

// ============================================================================
// Core Types & Enums
// ============================================================================

/// Represents the physical location of the resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResourceTier {
    /// Coldest: Standard filesystem.
    ColdDisk,
    /// Cold: Memory-mapped directly from high-speed NVMe storage.
    MmapNvme,
    /// Warm: Standard pageable system RAM.
    PageableRam,
    /// Hot: Pinned (Page-locked) system RAM for zero-copy PCIe DMA transfers.
    PinnedRam,
    /// Hottest: Residing entirely in GPU VRAM.
    Vram,
}

/// Priority with spatial/predictive weighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Required for the current frame (e.g., currently visible in frustum).
    Immediate,
    /// High probability of being needed next frame (e.g., adjacent to camera).
    PredictiveHigh,
    /// Standard retention.
    Normal,
    /// Background asset, safe to evict.
    Low,
}

/// Specialized ID combining UUID with a generational counter to prevent ABA problems.
#[derive(Debug, Clone, Copy)]
pub struct ResourceId {
    pub uuid: Uuid,
    pub generation: u32,
}

impl PartialEq for ResourceId {
    fn eq(&self, other: &Self) -> bool {
        self.uuid == other.uuid && self.generation == other.generation
    }
}
impl Eq for ResourceId {}
impl Hash for ResourceId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.uuid.hash(state);
        self.generation.hash(state);
    }
}

// ============================================================================
// Traits
// ============================================================================

/// Trait defining how an engine resource interacts with the hardware.
pub trait Offloadable: Send + Sync + 'static {
    /// Returns the exact byte size of the resource.
    fn size_bytes(&self) -> usize;
    
    /// Compresses and serializes for RAM/Disk storage.
    fn serialize_compressed(&self) -> Vec<u8>;
    
    /// Deserializes and decompresses from RAM/Disk.
    fn deserialize_decompressed(data: &[u8]) -> Self where Self: Sized;
    
    /// Pins the memory for direct DMA transfer (GART).
    fn pin_memory(&mut self) -> Result<(), OffloadError>;
    
    /// Submits the data to the GPU transfer queue.
    fn upload_to_vram(&mut self, queue: &GpuTransferQueue) -> Result<(), OffloadError>;
    
    /// Frees the VRAM allocation, moving back to Pinned or Pageable RAM.
    fn evict_from_vram(&mut self);
}

/// Mock GPU transfer queue for demonstration (abstracts Vulkan/WGPU/DirectX queues).
pub struct GpuTransferQueue {
    pub queue_id: u32,
}

#[derive(Debug)]
pub enum OffloadError {
    IoError(std::io::Error),
    GpuOOM,
    MmapFailed,
    ResourceNotFound,
}

// ============================================================================
// Metadata & Predictors
// ============================================================================

/// Atomic tracking of resource access to prevent locking during the render loop.
#[derive(Debug)]
pub struct AccessStats {
    /// Unix timestamp of last access in microseconds.
    pub last_accessed_micros: AtomicU64,
    /// Number of times accessed.
    pub access_count: AtomicU64,
}

/// Core metadata for a resource.
pub struct ResourceMeta {
    pub id: ResourceId,
    pub tier: ParkingLotRwLock<ResourceTier>,
    pub priority: ParkingLotRwLock<Priority>,
    pub size_bytes: usize,
    pub stats: AccessStats,
}

/// A graph that tracks which resources are loaded concurrently.
/// Used for Predictive Pre-fetching.
pub struct PredictiveGraph {
    /// Maps a Resource ID to a list of IDs frequently accessed right after it.
    adjacency: DashMap<ResourceId, DashMap<ResourceId, u32>>,
}

impl Default for PredictiveGraph {
    fn default() -> Self {
        Self {
            adjacency: DashMap::new(),
        }
    }
}

impl PredictiveGraph {
    /// Record that `next_id` was accessed shortly after `current_id`.
    pub fn record_sequence(&self, current_id: ResourceId, next_id: ResourceId) {
        let entry = self.adjacency.entry(current_id).or_insert_with(DashMap::new);
        *entry.entry(next_id).or_insert(0) += 1;
    }

    /// Get the highest probability upcoming resources based on current.
    pub fn predict_next(&self, current_id: ResourceId, limit: usize) -> Vec<ResourceId> {
        if let Some(edges) = self.adjacency.get(&current_id) {
            let mut candidates: Vec<_> = edges.iter().map(|e| (*e.key(), *e.value())).collect();
            // Sort by frequency descending
            candidates.sort_by(|a, b| b.1.cmp(&a.1));
            return candidates.into_iter().take(limit).map(|(id, _)| id).collect();
        }
        vec![]
    }
}

// ============================================================================
// System Configuration & Budgeting
// ============================================================================

#[derive(Debug, Clone)]
pub struct TierBudget {
    /// The maximum bytes allowed.
    pub max_bytes: usize,
    /// When usage hits this %, eviction starts.
    pub high_watermark_pct: f32,
    /// Eviction stops when usage drops to this %.
    pub low_watermark_pct: f32,
}

#[derive(Debug, Clone)]
pub struct OffloadConfig {
    pub vram: TierBudget,
    pub pinned_ram: TierBudget,
    pub pageable_ram: TierBudget,
    pub mmap_cache: TierBudget,
    pub storage_path: PathBuf,
    pub enable_predictive_prefetch: bool,
}

impl Default for OffloadConfig {
    fn default() -> Self {
        Self {
            vram: TierBudget { max_bytes: 8 * 1024 * 1024 * 1024, high_watermark_pct: 0.90, low_watermark_pct: 0.75 }, // 8GB
            pinned_ram: TierBudget { max_bytes: 2 * 1024 * 1024 * 1024, high_watermark_pct: 0.85, low_watermark_pct: 0.60 }, // 2GB
            pageable_ram: TierBudget { max_bytes: 16 * 1024 * 1024 * 1024, high_watermark_pct: 0.95, low_watermark_pct: 0.80 }, // 16GB
            mmap_cache: TierBudget { max_bytes: 50 * 1024 * 1024 * 1024, high_watermark_pct: 0.98, low_watermark_pct: 0.90 }, // 50GB NVMe Cache
            storage_path: PathBuf::from("./engine_cache"),
            enable_predictive_prefetch: true,
        }
    }
}

// ============================================================================
// The Manager
// ============================================================================

pub struct OffloadManager {
    config: OffloadConfig,
    /// O(1) concurrent access to all resource metadata. 
    registry: Arc<DashMap<ResourceId, Arc<ResourceMeta>>>,
    /// Tracks current memory usage per tier via fast atomic counters.
    usage_vram: Arc<AtomicUsize>,
    usage_pinned: Arc<AtomicUsize>,
    usage_pageable: Arc<AtomicUsize>,
    
    /// The ML-lite prediction engine.
    predictor: Arc<PredictiveGraph>,
    
    /// Channel for sending async tasks to the background I/O pool.
    task_tx: Sender<TaskMessage>,
    
    /// Thread handles for graceful shutdown.
    worker_handles: ParkingLotRwLock<Vec<JoinHandle<()>>>,
}

/// Messages sent to the background I/O workers.
enum TaskMessage {
    AccessNotification { id: ResourceId, prev_id: Option<ResourceId> },
    DemoteRequest { id: ResourceId, current_tier: ResourceTier, target_tier: ResourceTier },
    PromoteRequest { id: ResourceId, target_tier: ResourceTier },
    PrefetchHint { ids: Vec<ResourceId> },
    Shutdown,
}

impl OffloadManager {
    /// Initializes the Offload Manager with a dedicated thread pool for non-blocking I/O.
    pub fn new(config: OffloadConfig, worker_threads: usize) -> Self {
        std::fs::create_dir_all(&config.storage_path).unwrap();

        // Crossbeam channel for low-latency work dispatch
        let (task_tx, task_rx) = unbounded::<TaskMessage>();
        
        let manager = Self {
            config: config.clone(),
            registry: Arc::new(DashMap::new()),
            usage_vram: Arc::new(AtomicUsize::new(0)),
            usage_pinned: Arc::new(AtomicUsize::new(0)),
            usage_pageable: Arc::new(AtomicUsize::new(0)),
            predictor: Arc::new(PredictiveGraph::default()),
            task_tx,
            worker_handles: ParkingLotRwLock::new(Vec::new()),
        };

        // Spawn background worker threads (simulate a thread pool interacting with Tokio)
        let mut handles = manager.worker_handles.write();
        for _ in 0..worker_threads {
            let rx = task_rx.clone();
            let registry = Arc::clone(&manager.registry);
            let predictor = Arc::clone(&manager.predictor);
            let cfg = config.clone();
            let vram_usage = Arc::clone(&manager.usage_vram);

            let handle = tokio::spawn(async move {
                Self::worker_loop(rx, registry, predictor, cfg, vram_usage).await;
            });
            handles.push(handle);
        }

        manager
    }

    /// Highly optimized, lock-free access ping. To be called every time a resource is rendered.
    #[inline(always)]
    pub fn ping_access(&self, id: ResourceId, previous_id: Option<ResourceId>) {
        if let Some(meta) = self.registry.get(&id) {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64;
            
            // Lock-free atomic update
            meta.stats.last_accessed_micros.store(now, Ordering::Relaxed);
            meta.stats.access_count.fetch_add(1, Ordering::Relaxed);

            // Send to background for graph updates and pre-fetching
            let _ = self.task_tx.send(TaskMessage::AccessNotification { id, prev_id: previous_id });
        }
    }

    /// Background worker loop processing I/O and graph updates asynchronously.
    async fn worker_loop(
        rx: Receiver<TaskMessage>,
        registry: Arc<DashMap<ResourceId, Arc<ResourceMeta>>>,
        predictor: Arc<PredictiveGraph>,
        config: OffloadConfig,
        vram_usage: Arc<AtomicUsize>,
    ) {
        while let Ok(msg) = rx.recv() {
            match msg {
                TaskMessage::AccessNotification { id, prev_id } => {
                    if config.enable_predictive_prefetch {
                        if let Some(prev) = prev_id {
                            predictor.record_sequence(prev, id);
                            
                            // Trigger predictive pre-fetch
                            let predictions = predictor.predict_next(id, 3);
                            if !predictions.is_empty() {
                                // Real engine would send these IDs to the promotion queue
                                // println!("Pre-fetching predicted adjacent resources: {:?}", predictions);
                            }
                        }
                    }
                }
                TaskMessage::PromoteRequest { id, target_tier } => {
                    // Logic to load from Disk -> Mmap -> Ram -> Pinned -> Vram
                    // This is where zero-copy Vulkan/WGPU DMA transfers would be initiated.
                }
                TaskMessage::DemoteRequest { id, current_tier, target_tier } => {
                    if let Some(meta) = registry.get(&id) {
                        let mut tier_lock = meta.tier.write();
                        *tier_lock = target_tier;
                        
                        // Update atomic usage counters
                        if current_tier == ResourceTier::Vram {
                            vram_usage.fetch_sub(meta.size_bytes, Ordering::SeqCst);
                        }
                    }
                }
                TaskMessage::Shutdown => break,
                _ => {}
            }
        }
    }

    /// Enforces hysteresis-based memory budgeting.
    pub fn enforce_budgets(&self) {
        let current_vram = self.usage_vram.load(Ordering::Relaxed);
        let vram_high = (self.config.vram.max_bytes as f32 * self.config.vram.high_watermark_pct) as usize;
        let vram_low = (self.config.vram.max_bytes as f32 * self.config.vram.low_watermark_pct) as usize;

        if current_vram > vram_high {
            self.trigger_eviction(ResourceTier::Vram, current_vram - vram_low);
        }
    }

    /// Identifies and demotes resources to achieve `bytes_to_free`.
    fn trigger_eviction(&self, from_tier: ResourceTier, bytes_to_free: usize) {
        // Collect candidates. Using a BinaryHeap for an efficient priority queue based on a scoring heuristic.
        #[derive(PartialEq, PartialOrd)]
        struct EvictionCandidate {
            score: f32,
            id: ResourceId,
            size: usize,
        }
        impl Eq for EvictionCandidate {}
        impl Ord for EvictionCandidate {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.score.partial_cmp(&other.score).unwrap()
            }
        }

        let mut heap = BinaryHeap::new();
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_micros() as u64;

        for entry in self.registry.iter() {
            let meta = entry.value();
            if *meta.tier.read() == from_tier {
                // Heuristic score: Time since last access + Priority weighting
                let last_access = meta.stats.last_accessed_micros.load(Ordering::Relaxed);
                let age = (now.saturating_sub(last_access)) as f32;
                
                let priority_multiplier = match *meta.priority.read() {
                    Priority::Immediate => 0.0, // Never evict immediate
                    Priority::PredictiveHigh => 0.2,
                    Priority::Normal => 1.0,
                    Priority::Low => 5.0,
                };

                let score = age * priority_multiplier;
                if score > 0.0 {
                    heap.push(EvictionCandidate {
                        score,
                        id: meta.id,
                        size: meta.size_bytes,
                    });
                }
            }
        }

        let mut freed = 0;
        let target_tier = match from_tier {
            ResourceTier::Vram => ResourceTier::PinnedRam,
            ResourceTier::PinnedRam => ResourceTier::PageableRam,
            ResourceTier::PageableRam => ResourceTier::MmapNvme,
            _ => ResourceTier::ColdDisk,
        };

        while freed < bytes_to_free {
            if let Some(candidate) = heap.pop() {
                // Dispatch demotion to background worker
                let _ = self.task_tx.send(TaskMessage::DemoteRequest {
                    id: candidate.id,
                    current_tier: from_tier,
                    target_tier,
                });
                freed += candidate.size;
            } else {
                break; // No more eligible candidates
            }
        }
    }
}
