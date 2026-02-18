//! # High‑Performance GPU Resource Pool (wgpu)
//!
//! This is a single‑file, production‑ready implementation that provides:
//!
//! * **Texture pool** – LRU eviction, per‑format memory budgets, reference‑counted handles.
//! * **Buffer pool** – sub‑allocation, three size classes (small / medium / large),
//!   fast free‑list, optional defragmentation.
//! * **Staging manager** – ring of mapped buffers, double‑buffered upload,
//!   async‑friendly frame‑tracking.
//! * **Automatic cleanup** – per‑frame garbage collection, background
//!   de‑allocation, safety against GPU‑still‑in‑flight resources.
//! * **Extended statistics** – hits / misses, memory usage, eviction counts,
//!   upload throughput, etc.
//!
//! The implementation uses fine‑grained locking (RwLock + atomic counters)
//! to maximise concurrency and minimise lock contention.

// -----------------------------------------------------------------------------
// Imports
// -----------------------------------------------------------------------------

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    num::NonZeroU64,
    sync::{
        atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
        mpsc::{self, Receiver, Sender, TryRecvError},
        Arc, Mutex, RwLock,
    },
    time::{Duration, Instant},
};

use wgpu::{
    util::{BufferInitDescriptor as WgpuBufferInitDescriptor, DeviceExt},
    Buffer, BufferUsages, CommandEncoder, Device, Extent3d, ImageCopyBuffer, ImageCopyTexture,
    ImageDataLayout, Queue, Texture, TextureDescriptor, TextureFormat, TextureUsages,
    TextureView, TextureViewDescriptor,
};

// -----------------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------------

/// Opaque handle to a pooled texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureHandle {
    id: u64,
}

/// Opaque handle to a pooled buffer allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferHandle {
    id: u64,
    // valid only for dedicated allocations
    block_id: Option<u64>,
}

/// Token returned from a staging upload. It can be used to query when the
/// upload is no longer in‑flight.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UploadToken {
    id: u64,
    frame: u64,
}

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

/// Global configuration for the resource pool.
#[derive(Clone, Copy)]
pub struct PoolConfig {
    // --- Staging --------------------------------------------------------------
    /// Size of each staging buffer (bytes).
    pub staging_buffer_size: u64,
    /// Number of staging buffers in the ring.
    pub staging_buffer_count: usize,
    /// Maximum number of frames a staging slice may stay in‑flight.
    pub max_frames_in_flight: usize,

    // --- Buffer pool ---------------------------------------------------------
    /// Small‑block size (bytes). Good for uniform / small vertex buffers.
    pub buffer_small_block: u64,
    /// Medium‑block size (bytes). Good for index / large vertex buffers.
    pub buffer_medium_block: u64,
    /// Large‑block size (bytes). For big data (e.g. large SSBOs, texture‑upload buffers).
    pub buffer_large_block: u64,

    // --- Texture pool --------------------------------------------------------
    /// Soft cap for total pooled texture memory (bytes).
    pub max_texture_memory: u64,
    /// Target fraction of `max_texture_memory` to evict to when over the cap.
    pub texture_eviction_target: f32,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            staging_buffer_size: 8 * 1024 * 1024,          // 8 MiB
            staging_buffer_count: 3,
            max_frames_in_flight: 3,

            buffer_small_block: 64 * 1024,               // 64 KiB
            buffer_medium_block: 1 * 1024 * 1024,        // 1 MiB
            buffer_large_block: 8 * 1024 * 1024,         // 8 MiB

            max_texture_memory: 1024 * 1024 * 1024,      // 1 GiB
            texture_eviction_target: 0.7,
        }
    }
}

// -----------------------------------------------------------------------------
// Statistics
// -----------------------------------------------------------------------------

/// Global statistics for the whole pool.
#[derive(Default, Clone, Debug)]
pub struct PoolStats {
    // Texture pool
    pub texture_allocation_hits: u64,
    pub texture_allocation_misses: u64,
    pub texture_eviction_count: u64,
    pub texture_total_bytes: u64,
    pub texture_pooled_bytes: u64,
    pub texture_count: usize,

    // Buffer pool
    pub buffer_allocation_hits: u64,
    pub buffer_allocation_misses: u64,
    pub buffer_eviction_count: u64,
    pub buffer_total_bytes: u64,
    pub buffer_pooled_bytes: u64,
    pub buffer_block_count: usize,
    pub buffer_dedicated_count: usize,

    // Staging
    pub staging_upload_bytes: u64,
    pub staging_in_flight: usize,
    pub staging_recycle_count: u64,

    // GC
    pub gc_textures_collected: u64,
    pub gc_buffers_collected: u64,
    pub last_gc_duration: Duration,
}

// -----------------------------------------------------------------------------
// Texture Pool
// -----------------------------------------------------------------------------

/// Key that uniquely identifies a texture layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TextureKey {
    width: u32,
    height: u32,
    mip_level_count: u32,
    sample_count: u32,
    format: TextureFormat,
    usage: TextureUsages,
}

impl TextureKey {
    fn from_desc(desc: &TextureDescriptor) -> Self {
        Self {
            width: desc.size.width,
            height: desc.size.height,
            mip_level_count: desc.mip_level_count,
            sample_count: desc.sample_count,
            format: desc.format,
            usage: desc.usage,
        }
    }

    /// Approximate size in bytes (conservative estimate).
    fn size_bytes(&self) -> u64 {
        let bpp = match self.format {
            TextureFormat::R8Unorm
            | TextureFormat::R8Snorm
            | TextureFormat::R8Uint
            | TextureFormat::R8Sint => 1,
            TextureFormat::R16Unorm
            | TextureFormat::R16Snorm
            | TextureFormat::R16Uint
            | TextureFormat::R16Sint
            | TextureFormat::R16Float => 2,
            TextureFormat::Rg8Unorm
            | TextureFormat::Rg8Snorm
            | TextureFormat::Rg8Uint
            | TextureFormat::Rg8Sint => 2,
            TextureFormat::Rg16Unorm
            | TextureFormat::Rg16Snorm
            | TextureFormat::Rg16Uint
            | TextureFormat::Rg16Sint
            | TextureFormat::Rg16Float => 4,
            TextureFormat::Rgb8Unorm
            | TextureFormat::Rgb8Snorm
            | TextureFormat::Rgb8Uint
            | TextureFormat::Rgb8Sint => 3,
            TextureFormat::Rgba8Unorm
            | TextureFormat::Rgba8UnormSrgb
            | TextureFormat::Rgba8Snorm
            | TextureFormat::Rgba8Uint
            | TextureFormat::Rgba8Sint => 4,
            TextureFormat::Bgra8Unorm
            | TextureFormat::Bgra8UnormSrgb
            | TextureFormat::Rgba16Unorm
            | TextureFormat::Rgba16Snorm
            | TextureFormat::Rgba16Uint
            | TextureFormat::Rgba16Sint
            | TextureFormat::Rgba16Float => 8,
            TextureFormat::Rgba32Float => 16,
            _ => 4, // fall‑back
        } as u64;

        let mut size = 0u64;
        let mut w = self.width as u64;
        let mut h = self.height as u64;
        for _ in 0..self.mip_level_count {
            size += w * h * bpp;
            w = (w / 2).max(1);
            h = (h / 2).max(1);
        }
        size
    }
}

/// Internal representation of a pooled texture.
struct PooledTexture {
    texture: Texture,
    view: TextureView,
    key: TextureKey,
    size_bytes: u64,

    // Concurrency helpers
    ref_count: AtomicU32,                // how many handles are using it
    last_used_frame: AtomicU64,          // last frame it was touched
    resident: AtomicBool,                // true if still in GPU memory
}

/// Internals of `TexturePool`. The lock granularity is:
/// * `items`: RwLock – read‑only for `get_view`, `touch`, etc.
/// * `map`:   RwLock – fast look‑up of free candidates.
/// * `lru`:   Mutex – only mutated during eviction or promotion.
struct TexturePoolInner {
    items: RwLock<HashMap<u64, PooledTexture>>,
    map: RwLock<HashMap<TextureKey, Vec<u64>>>,
    lru: Mutex<VecDeque<u64>>,            // front = least‑recently used
    next_id: AtomicU64,
    total_bytes: AtomicU64,
}

/// High‑performance texture pool.
pub struct TexturePool {
    device: Arc<Device>,
    inner: Arc<TexturePoolInner>,
    config: PoolConfig,
    stats: Mutex<PoolStats>,
}

impl TexturePool {
    /// Create a new texture pool.
    pub fn new(device: Arc<Device>, config: PoolConfig) -> Self {
        Self {
            device,
            inner: Arc::new(TexturePoolInner {
                items: RwLock::new(HashMap::new()),
                map: RwLock::new(HashMap::new()),
                lru: Mutex::new(VecDeque::new()),
                next_id: AtomicU64::new(1),
                total_bytes: AtomicU64::new(0),
            }),
            config,
            stats: Mutex::new(PoolStats::default()),
        }
    }

    /// Request a texture that matches `desc`. Returns a handle that can later
    /// be released with `release_texture`.
    pub fn request_texture(&self, desc: &TextureDescriptor) -> TextureHandle {
        let key = TextureKey::from_desc(desc);
        let mut handle = TextureHandle { id: 0 };

        // Fast path – try to recycle an existing entry.
        {
            let map = self.inner.map.read().unwrap();
            if let Some(ids) = map.get(&key) {
                for &id in ids.iter() {
                    let items = self.inner.items.read().unwrap();
                    if let Some(tex) = items.get(&id) {
                        if tex.ref_count.load(Ordering::Acquire) == 0
                            && tex.resident.load(Ordering::Acquire)
                        {
                            // Claim it.
                            tex.ref_count.store(1, Ordering::Release);
                            tex.last_used_frame
                                .store(0, Ordering::Release); // will be set by `touch`
                            handle.id = id;
                            break;
                        }
                    }
                }
            }
        }

        if handle.id != 0 {
            // Update statistics.
            let mut s = self.stats.lock().unwrap();
            s.texture_allocation_hits += 1;
            return handle;
        }

        // Slow path – create a new texture.
        let texture = self.device.create_texture(desc);
        let view = texture.create_view(&TextureViewDescriptor::default {});
        let size_bytes = key.size_bytes();
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);

        let pooled = PooledTexture {
            texture,
            view,
            key,
            size_bytes,
            ref_count: AtomicU32::new(1),
            last_used_frame: AtomicU64::new(0),
            resident: AtomicBool::new(true),
        };

  
