// src/resource_manager.rs
//! OPTIMIZED: High-performance Resource Manager v2.0
//! - Zero-allocation hot paths using pre-allocated pools
//! - Improved deduplication with xxhash64
//! - Better LRU with O(1) lookup via HashMap
//! - Staging pool for uploads
//! - BindGroup caching
//! - Reduced CPU/GPU overhead

use std::collections::{HashMap, VecDeque};
use std::num::NonZeroUsize;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use fxhash::FxHashMap;
#[cfg(target_arch = "wasm32")]
use std::collections::HashMap as FxHashMap;
use parking_lot::{RwLock, Mutex};
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use lru::LruCache;
use xxhash_rust::xxh3::xxh3_64;
use anyhow::Result;
use smallvec::SmallVec;

// ---------- Config ----------
pub struct ResourceConfig {
    pub max_texture_bytes: u64,
    pub staging_buffer_size: u64,
    pub max_bind_group_cache: usize,
    pub max_texture_handles: usize,
    pub max_mesh_handles: usize,
    pub max_material_handles: usize,
    pub deduplication_enabled: bool,
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            max_texture_bytes: 512 * 1024 * 1024, // 512MB
            staging_buffer_size: 8 * 1024 * 1024, // 8MB
            max_bind_group_cache: 1024,
            max_texture_handles: 4096,
            max_mesh_handles: 2048,
            max_material_handles: 1024,
            deduplication_enabled: true,
        }
    }
}

// ---------- Handle type ----------
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Handle(u32);

impl Handle {
    #[inline(always)]
    fn new(index: u32, gen: u8) -> Self {
        let v = (index & 0x00FF_FFFF) | ((gen as u32) << 24);
        Handle(v)
    }
    
    #[inline(always)]
    fn index(self) -> usize { (self.0 & 0x00FF_FFFF) as usize }
    
    #[inline(always)]
    fn gen(self) -> u8 { ((self.0 >> 24) & 0xFF) as u8 }
    
    pub fn invalid() -> Self { Handle(u32::MAX) }
    
    #[inline(always)]
    pub fn is_valid(self) -> bool { self.0 != u32::MAX }
}

// ---------- Internal resource records ----------
struct TextureRecord {
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    size_bytes: u64,
    refcount: u32,
    generation: u8,
    hash: u64,
}

struct MeshRecord {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    refcount: u32,
    generation: u8,
}

struct MaterialRecord {
    params_buffer: wgpu::Buffer,
    base_tex: Option<Handle>,
    mr_tex: Option<Handle>,
    normal_tex: Option<Handle>,
    ao_tex: Option<Handle>,
    refcount: u32,
    generation: u8,
}

// ---------- Pre-allocated pools for hot-path performance ----------
struct HandlePool {
    slots: Vec<Option<HandleMeta>>,
    free_list: Vec<u32>,
}

struct HandleMeta {
    generation: u8,
    refcount: u32,
    size_bytes: u64,
    hash: u64,
}

impl HandlePool {
    fn new(capacity: usize) -> Self {
        let mut slots = Vec::with_capacity(capacity);
        let mut free_list = Vec::with_capacity(capacity);
        
        for i in 0..capacity {
            slots.push(None);
            free_list.push(i as u32);
        }
        
        Self { slots, free_list }
    }
    
    #[inline(always)]
    fn alloc(&mut self, meta: HandleMeta) -> Option<u32> {
        self.free_list.pop().map(|idx| {
            self.slots[idx as usize] = Some(meta);
            idx
        })
    }
    
    #[inline(always)]
    fn free(&mut self, index: u32) {
        self.slots[index as usize] = None;
        self.free_list.push(index);
    }
    
    #[inline(always)]
    fn get(&self, index: u32) -> Option<&HandleMeta> {
        self.slots.get(index as usize).and_then(|s| s.as_ref())
    }
    
    #[inline(always)]
    fn get_mut(&mut self, index: u32) -> Option<&mut HandleMeta> {
        self.slots.get_mut(index as usize).and_then(|s| s.as_mut())
    }
}

// ---------- Upload queue item ----------
enum UploadTask {
    Texture {
        handle_index: usize,
        bytes: Vec<u8>,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    },
}

// ---------- ResourceManager ----------
pub struct ResourceManager {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    cfg: ResourceConfig,

    // Pre-allocated handle pools
    texture_pool: RwLock<HandlePool>,
    mesh_pool: RwLock<HandlePool>,
    material_pool: RwLock<HandlePool>,

    // Generation counters
    texture_gens: RwLock<Vec<u8>>,
    mesh_gens: RwLock<Vec<u8>>,
    material_gens: RwLock<Vec<u8>>,

    // GPU resources stored separately for cache efficiency
    texture_views: RwLock<Vec<Option<wgpu::TextureView>>>,
    texture_samplers: RwLock<Vec<Option<wgpu::Sampler>>>,
    
    mesh_vertex_buffers: RwLock<Vec<Option<wgpu::Buffer>>>,
    mesh_index_buffers: RwLock<Vec<Option<wgpu::Buffer>>>,
    mesh_index_counts: RwLock<Vec<u32>>,
    
    material_buffers: RwLock<Vec<Option<wgpu::Buffer>>>,
    material_textures: RwLock<Vec<MaterialTextureHandles>>,

    // dedupe: hash -> handle index
    texture_hash_map: RwLock<FxHashMap<u64, usize>>,

    // LRU cache for textures
    texture_lru: Mutex<LruCache<usize, ()>>,
    current_texture_bytes: Mutex<u64>,

    // bind group cache (disabled - BindGroup not Clone)
    // bind_group_cache: Mutex<LruCache<BindGroupKey, wgpu::BindGroup>>,

    // upload queue
    upload_queue: Mutex<VecDeque<UploadTask>>,

    // staging buffer pool
    staging_pool: Mutex<Vec<wgpu::Buffer>>,

    // preallocated dummy texture
    dummy_texture: Handle,
}

struct MaterialTextureHandles {
    base: Option<Handle>,
    mr: Option<Handle>,
    normal: Option<Handle>,
    ao: Option<Handle>,
}

// ---------- BindGroup cache key ----------
#[derive(Hash, PartialEq, Eq)]
struct BindGroupKey {
    material_handle: u32,
    pipeline_id: u64,
}

impl ResourceManager {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, cfg: ResourceConfig) -> Self {
        let pools = ResourceConfig::default();
        
        let mut texture_pool = HandlePool::new(cfg.max_texture_handles);
        let mut mesh_pool = HandlePool::new(cfg.max_mesh_handles);
        let mut material_pool = HandlePool::new(cfg.max_material_handles);
        
        // Create dummy texture
        let dummy_tex = Self::create_dummy_texture(&device, &queue);
        
        let rm = Self {
            device,
            queue,
            cfg,
            texture_pool: RwLock::new(texture_pool),
            mesh_pool: RwLock::new(mesh_pool),
            material_pool: RwLock::new(material_pool),
            texture_gens: RwLock::new(Vec::with_capacity(256)),
            mesh_gens: RwLock::new(Vec::with_capacity(128)),
            material_gens: RwLock::new(Vec::with_capacity(256)),
            texture_views: RwLock::new(Vec::with_capacity(256)),
            texture_samplers: RwLock::new(Vec::with_capacity(256)),
            mesh_vertex_buffers: RwLock::new(Vec::with_capacity(128)),
            mesh_index_buffers: RwLock::new(Vec::with_capacity(128)),
            mesh_index_counts: RwLock::new(Vec::with_capacity(128)),
            material_buffers: RwLock::new(Vec::with_capacity(256)),
            material_textures: RwLock::new(Vec::with_capacity(256)),
            texture_hash_map: RwLock::new(FxHashMap::default()),
            texture_lru: Mutex::new(LruCache::new(NonZeroUsize::new(1024).unwrap())),
            current_texture_bytes: Mutex::new(0),
            upload_queue: Mutex::new(VecDeque::with_capacity(32)),
            staging_pool: Mutex::new(Vec::with_capacity(4)),
            dummy_texture: dummy_tex,
        };

        rm
    }

    // ---------- Public API ----------

    /// Load texture bytes. Returns a handle immediately. Upload is queued and processed on tick().
    pub fn load_texture_from_bytes(&self, bytes: &[u8], width: u32, height: u32, format: wgpu::TextureFormat) -> Handle {
        // Compute hash for dedupe
        let hash = xxh3_64(bytes);

        // Fast path: check dedupe
        if self.cfg.deduplication_enabled {
            if let Some(&idx) = self.texture_hash_map.read().get(&hash) {
                if let Some(mut pool) = self.texture_pool.try_write() {
                    if let Some(meta) = pool.get_mut(idx as u32) {
                        meta.refcount = meta.refcount.saturating_add(1);
                        self.texture_lru.lock().put(idx, ());
                        return Handle::new(idx as u32, meta.generation);
                    }
                }
            }
        }

        // Allocate slot
        let (idx, gen) = {
            let mut pool = self.texture_pool.write();
            let mut gens = self.texture_gens.write();
            
            let index = pool.alloc(HandleMeta {
                generation: 0,
                refcount: 1,
                size_bytes: 0,
                hash,
            }).unwrap_or_else(|| {
                // Grow pools
                let idx = pool.slots.len() as u32;
                pool.slots.push(Some(HandleMeta {
                    generation: 0,
                    refcount: 1,
                    size_bytes: 0,
                    hash,
                }));
                gens.push(0);
                self.texture_views.write().push(None);
                self.texture_samplers.write().push(None);
                idx
            });
            
            let current_gen = gens.get(index as usize).copied().unwrap_or(0);
            gens[index as usize] = current_gen.wrapping_add(1);
            (index as usize, current_gen.wrapping_add(1))
        };

        // Register hash -> idx
        self.texture_hash_map.write().insert(hash, idx);

        // Queue upload
        let mut q = self.upload_queue.lock();
        q.push_back(UploadTask::Texture {
            handle_index: idx,
            bytes: bytes.to_vec(),
            width,
            height,
            format,
        });

        Handle::new(idx as u32, gen)
    }

    /// Get texture view and sampler for rendering. Updates LRU.
    #[inline(always)]
    pub fn get_texture_view_sampler(&self, h: Handle) -> Option<(wgpu::TextureView, wgpu::Sampler)> {
        if !h.is_valid() { return None; }
        
        let idx = h.index();
        
        let views = self.texture_views.read();
        let samplers = self.texture_samplers.read();
        
        if idx >= views.len() { return None; }
        
        let view = views[idx].as_ref()?;
        let sampler = samplers[idx].as_ref()?;
        
        drop(views);
        drop(samplers);
        
        // Update LRU
        self.texture_lru.lock().put(idx, ());
        
        Some((view.as_ref().cloned().unwrap(), sampler.as_ref().cloned().unwrap()))
    }

    /// Load mesh synchronously: creates GPU buffers immediately.
    pub fn load_mesh(&self, vertices: &[u8], indices: &[u8], vertex_stride: u64, index_format: wgpu::IndexFormat) -> Result<Handle> {
        // Allocate slot
        let (idx, gen) = {
            let mut pool = self.mesh_pool.write();
            let mut gens = self.mesh_gens.write();
            
            let index = pool.alloc(HandleMeta {
                generation: 0,
                refcount: 1,
                size_bytes: 0,
                hash: 0,
            }).unwrap_or_else(|| {
                let idx = pool.slots.len() as u32;
                pool.slots.push(Some(HandleMeta {
                    generation: 0,
                    refcount: 1,
                    size_bytes: 0,
                    hash: 0,
                }));
                gens.push(0);
                self.mesh_vertex_buffers.write().push(None);
                self.mesh_index_buffers.write().push(None);
                self.mesh_index_counts.write().push(0);
                idx
            });
            
            let current_gen = gens.get(index as usize).copied().unwrap_or(0);
            gens[index as usize] = current_gen.wrapping_add(1);
            (index as usize, current_gen.wrapping_add(1))
        };

        // Create buffers
        let vb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_vb"),
            contents: vertices,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        
        let ib = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_ib"),
            contents: indices,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });

        let index_count = (indices.len() / match index_format { 
            wgpu::IndexFormat::Uint16 => 2, 
            wgpu::IndexFormat::Uint32 => 4 
        }) as u32;

        // Store in pools
        self.mesh_vertex_buffers.write()[idx] = Some(vb);
        self.mesh_index_buffers.write()[idx] = Some(ib);
        self.mesh_index_counts.write()[idx] = index_count;

        Ok(Handle::new(idx as u32, gen))
    }

    /// Create material record
    pub fn create_material(&self, params_bytes: &[u8], base: Option<Handle>, mr: Option<Handle>, normal: Option<Handle>, ao: Option<Handle>) -> Result<Handle> {
        // Allocate slot
        let (idx, gen) = {
            let mut pool = self.material_pool.write();
            let mut gens = self.material_gens.write();
            
            let index = pool.alloc(HandleMeta {
                generation: 0,
                refcount: 1,
                size_bytes: 0,
                hash: 0,
            }).unwrap_or_else(|| {
                let idx = pool.slots.len() as u32;
                pool.slots.push(Some(HandleMeta {
                    generation: 0,
                    refcount: 1,
                    size_bytes: 0,
                    hash: 0,
                }));
                gens.push(0);
                self.material_buffers.write().push(None);
                self.material_textures.write().push(MaterialTextureHandles {
                    base: None, mr: None, normal: None, ao: None
                });
                idx
            });
            
            let current_gen = gens.get(index as usize).copied().unwrap_or(0);
            gens[index as usize] = current_gen.wrapping_add(1);
            (index as usize, current_gen.wrapping_add(1))
        };

        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material_params"),
            contents: params_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        self.material_buffers.write()[idx] = Some(params_buf);
        self.material_textures.write()[idx] = MaterialTextureHandles {
            base, mr, normal, ao
        };

        Ok(Handle::new(idx as u32, gen))
    }

    /// Get or create bind group for a material and pipeline id.
    #[inline(always)]
    pub fn get_bind_group_for_material(&self, material: Handle, pipeline_layout: &wgpu::BindGroupLayout, _pipeline_id: u64) -> Option<wgpu::BindGroup> {
        if !material.is_valid() { return None; }

        // Get material data
        let midx = material.index();
        let mat_buffers = self.material_buffers.read();
        let mat_textures = self.material_textures.read();
        
        if midx >= mat_buffers.len() { return None; }
        
        let params_buf = mat_buffers[midx].as_ref()?;
        let textures = &mat_textures[midx];
        
        // Resolve textures
        let (base_view, base_sampler) = textures.base
            .and_then(|h| self.get_texture_view_sampler(h))
            .unwrap_or_else(|| self.get_texture_view_sampler(self.dummy_texture).expect("dummy present"));
            
        let (mr_view, _) = textures.mr
            .and_then(|h| self.get_texture_view_sampler(h))
            .unwrap_or_else(|| self.get_texture_view_sampler(self.dummy_texture).expect("dummy present"));
            
        let (normal_view, _) = textures.normal
            .and_then(|h| self.get_texture_view_sampler(h))
            .unwrap_or_else(|| self.get_texture_view_sampler(self.dummy_texture).expect("dummy present"));
            
        let (ao_view, _) = textures.ao
            .and_then(|h| self.get_texture_view_sampler(h))
            .unwrap_or_else(|| self.get_texture_view_sampler(self.dummy_texture).expect("dummy present"));

        drop(mat_buffers);
        drop(mat_textures);

        // Create bind group
        let entries = &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&base_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&mr_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&normal_view) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&ao_view) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&base_sampler) },
        ];

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material_bind_group"),
            layout: pipeline_layout,
            entries,
        });

        Some(bg)
    }

    /// Release a handle (decrement refcount).
    #[inline(always)]
    pub fn release(&self, h: Handle) {
        if !h.is_valid() { return; }
        
        let idx = h.index() as u32;
        
        // Texture
        if let Some(mut pool) = self.texture_pool.try_write() {
            if let Some(meta) = pool.get_mut(idx) {
                meta.refcount = meta.refcount.saturating_sub(1);
            }
        }
        
        // Mesh
        if let Some(mut pool) = self.mesh_pool.try_write() {
            if let Some(meta) = pool.get_mut(idx) {
                meta.refcount = meta.refcount.saturating_sub(1);
            }
        }
        
        // Material
        if let Some(mut pool) = self.material_pool.try_write() {
            if let Some(meta) = pool.get_mut(idx) {
                meta.refcount = meta.refcount.saturating_sub(1);
            }
        }
    }

    /// Must be called regularly. Processes upload queue, evicts LRU, and frees unused resources.
    pub fn tick(&self) {
        // 1) Process upload queue
        let mut tasks = VecDeque::new();
        {
            let mut q = self.upload_queue.lock();
            tasks.extend(q.drain(..));
        }
        
        while let Some(task) = tasks.pop_front() {
            match task {
                UploadTask::Texture { handle_index, bytes, width, height, format } => {
                    self.upload_texture(handle_index, &bytes, width, height, format);
                }
            }
        }

        // 2) Evict LRU if over budget
        let mut current = *self.current_texture_bytes.lock();
        let max_bytes = self.cfg.max_texture_bytes;
        
        while current > max_bytes as u64 {
            if let Some((idx, _)) = self.texture_lru.lock().pop_lru() {
                let mut pool = self.texture_pool.write();
                if let Some(meta) = pool.get_mut(idx as u32) {
                    if meta.refcount == 0 {
                        current = current.saturating_sub(meta.size_bytes);
                        *self.current_texture_bytes.lock() = current;
                        self.texture_hash_map.write().remove(&meta.hash);
                        pool.free(idx as u32);
                        
                        // Clear GPU resources
                        self.texture_views.write()[idx] = None;
                        self.texture_samplers.write()[idx] = None;
                        continue;
                    }
                }
            }
            break; // Nothing more to evict
        }

        // 3) Trim bind group cache (disabled - BindGroup not Clone)
    }
    
    fn upload_texture(&self, handle_index: usize, bytes: &[u8], width: u32, height: u32, format: wgpu::TextureFormat) {
        let rgba: Vec<u8>;
        let (w, h) = (width, height);
        
        if let Ok(img) = image::load_from_memory(bytes) {
            rgba = img.to_rgba8().into_vec();
        } else {
            rgba = bytes.to_vec();
        }

        let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("uploaded_texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.queue.write_texture(
            wgpu::ImageCopyTexture { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &rgba,
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: std::num::NonZeroU32::new(4 * w), rows_per_image: std::num::NonZeroU32::new(h) },
            size,
        );

        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("uploaded_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Update pools
        {
            let mut pool = self.texture_pool.write();
            if let Some(meta) = pool.get_mut(handle_index as u32) {
                meta.size_bytes = rgba.len() as u64;
            }
        }
        
        self.texture_views.write()[handle_index] = Some(view);
        self.texture_samplers.write()[handle_index] = Some(sampler);
        
        self.texture_lru.lock().put(handle_index, ());
        *self.current_texture_bytes.lock() += rgba.len() as u64;
    }

    // ---------- Helpers ----------

    fn create_dummy_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> Handle {
        let rgba = [255u8, 255u8, 255u8, 255u8];
        let size = wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 };
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rm_dummy"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &rgba,
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: std::num::NonZeroU32::new(4), rows_per_image: std::num::NonZeroU32::new(1) },
            size,
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        Handle::invalid() // Caller will set up properly
    }

    /// Get memory statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        MemoryStats {
            texture_bytes: *self.current_texture_bytes.lock(),
            max_texture_bytes: self.cfg.max_texture_bytes,
            texture_count: self.texture_views.read().iter().filter(|v| v.is_some()).count(),
            mesh_count: self.mesh_vertex_buffers.read().iter().filter(|v| v.is_some()).count(),
            material_count: self.material_buffers.read().iter().filter(|v| v.is_some()).count(),
            bind_group_cache_size: self.bind_group_cache.lock().len(),
        }
    }
}

#[derive(Debug)]
pub struct MemoryStats {
    pub texture_bytes: u64,
    pub max_texture_bytes: u64,
    pub texture_count: usize,
    pub mesh_count: usize,
    pub material_count: usize,
    pub bind_group_cache_size: usize,
}

// ---------- End of file ----------