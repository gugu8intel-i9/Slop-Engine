// src/resource_manager.rs
//! High-performance Resource Manager
//! - Handles: compact u32 handles with generation
//! - Deduplication: xxhash64 on bytes
//! - LRU cache for textures
//! - Staging pool for uploads
//! - BindGroup caching
//! - Minimal allocations and low CPU/GPU overhead

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use lru::LruCache;
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use xxhash_rust::xxh3::xxh3_64;

// ---------- Config ----------
pub struct ResourceConfig {
    pub max_texture_bytes: u64,
    pub staging_buffer_size: u64,
    pub max_bind_group_cache: usize,
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            max_texture_bytes: 512 * 1024 * 1024, // 512MB
            staging_buffer_size: 8 * 1024 * 1024, // 8MB
            max_bind_group_cache: 1024,
        }
    }
}

// ---------- Handle type ----------
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Handle(u32);

impl Handle {
    fn new(index: u32, gen: u8) -> Self {
        let v = (index & 0x00FF_FFFF) | ((gen as u32) << 24);
        Handle(v)
    }
    fn index(self) -> usize {
        (self.0 & 0x00FF_FFFF) as usize
    }
    fn gen(self) -> u8 {
        ((self.0 >> 24) & 0xFF) as u8
    }
    pub fn invalid() -> Self {
        Handle(u32::MAX)
    }
    pub fn is_valid(self) -> bool {
        self.0 != u32::MAX
    }
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
    // store texture handles used by material for rebinds
    base_tex: Option<Handle>,
    mr_tex: Option<Handle>,
    normal_tex: Option<Handle>,
    ao_tex: Option<Handle>,
    refcount: u32,
    generation: u8,
}

// BindGroup cache key
#[derive(Hash, PartialEq, Eq)]
struct BindGroupKey {
    material_handle: u32,
    pipeline_id: u64,
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

    // slabs and generations
    texture_slab: RwLock<Vec<Option<TextureRecord>>>,
    texture_gens: RwLock<Vec<u8>>,
    mesh_slab: RwLock<Vec<Option<MeshRecord>>>,
    mesh_gens: RwLock<Vec<u8>>,
    material_slab: RwLock<Vec<Option<MaterialRecord>>>,
    material_gens: RwLock<Vec<u8>>,

    // dedupe: hash -> handle index
    texture_hash_map: RwLock<HashMap<u64, usize>>,

    // LRU cache for textures by handle index
    texture_lru: Mutex<LruCache<usize, ()>>,
    current_texture_bytes: Mutex<u64>,

    // bind group cache
    bind_group_cache: Mutex<LruCache<BindGroupKey, wgpu::BindGroup>>,

    // upload queue
    upload_queue: Mutex<Vec<UploadTask>>,

    // staging buffer pool
    staging_pool: Mutex<Vec<wgpu::Buffer>>,

    // preallocated dummy texture
    dummy_texture: Handle,
}

impl ResourceManager {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, cfg: ResourceConfig) -> Self {
        let lru = LruCache::new(1024);

        let mut rm = Self {
            device,
            queue,
            cfg,
            texture_slab: RwLock::new(Vec::new()),
            texture_gens: RwLock::new(Vec::new()),
            mesh_slab: RwLock::new(Vec::new()),
            mesh_gens: RwLock::new(Vec::new()),
            material_slab: RwLock::new(Vec::new()),
            material_gens: RwLock::new(Vec::new()),
            texture_hash_map: RwLock::new(HashMap::new()),
            texture_lru: Mutex::new(lru),
            current_texture_bytes: Mutex::new(0),
            bind_group_cache: Mutex::new(LruCache::new(cfg.max_bind_group_cache)),
            upload_queue: Mutex::new(Vec::new()),
            staging_pool: Mutex::new(Vec::new()),
            dummy_texture: Handle::invalid(),
        };

        // register dummy texture into slab
        rm.dummy_texture = rm.register_dummy_in_slab();
        rm
    }

    // ---------- Public API ----------

    /// Load texture bytes. Returns a handle immediately. Upload is queued and processed on tick().
    pub fn load_texture_from_bytes(
        &self,
        bytes: &[u8],
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Handle {
        // compute hash for dedupe
        let hash = xxh3_64(bytes);

        // fast path: existing texture
        if let Some(&idx) = self.texture_hash_map.read().get(&hash) {
            // bump refcount and LRU
            let mut slab = self.texture_slab.write();
            if let Some(Some(tr)) = slab.get_mut(idx) {
                tr.refcount = tr.refcount.saturating_add(1);
                self.texture_lru.lock().put(idx, ());
                return Handle::new(idx as u32, tr.generation);
            }
        }

        // allocate new slab slot
        let idx = {
            let mut slab = self.texture_slab.write();
            let mut gens = self.texture_gens.write();
            // find free slot
            let mut slot = None;
            for (i, s) in slab.iter().enumerate() {
                if s.is_none() {
                    slot = Some(i);
                    break;
                }
            }
            let index = if let Some(i) = slot {
                i
            } else {
                slab.push(None);
                gens.push(0u8);
                slab.len() - 1
            };
            // increment generation
            gens[index] = gens[index].wrapping_add(1);
            index
        };

        // create placeholder record with zeroed view; will be replaced on upload completion
        {
            let mut slab = self.texture_slab.write();
            let mut gens = self.texture_gens.write();
            let gen = gens[idx];
            // create placeholder texture view using dummy texture view to keep bind groups stable
            let dummy = self
                .get_texture_record(self.dummy_texture)
                .expect("dummy present");
            let rec = TextureRecord {
                view: dummy.view.clone(),
                sampler: dummy.sampler.clone(),
                size_bytes: 0,
                refcount: 1,
                generation: gen,
                hash,
            };
            slab[idx] = Some(rec);
        }

        // register hash -> idx
        self.texture_hash_map.write().insert(hash, idx);

        // queue upload
        let mut q = self.upload_queue.lock();
        q.push(UploadTask::Texture {
            handle_index: idx,
            bytes: bytes.to_vec(),
            width,
            height,
            format,
        });

        Handle::new(idx as u32, self.texture_gens.read()[idx])
    }

    /// Get texture view and sampler for rendering. Updates LRU.
    pub fn get_texture_view_sampler(
        &self,
        h: Handle,
    ) -> Option<(wgpu::TextureView, wgpu::Sampler)> {
        if !h.is_valid() {
            return None;
        }
        let idx = h.index();
        let slab = self.texture_slab.read();
        if idx >= slab.len() {
            return None;
        }
        if let Some(rec) = &slab[idx] {
            // update LRU
            self.texture_lru.lock().put(idx, ());
            Some((rec.view.clone(), rec.sampler.clone()))
        } else {
            None
        }
    }

    /// Load mesh synchronously: creates GPU buffers immediately.
    pub fn load_mesh(
        &self,
        vertices: &[u8],
        indices: &[u8],
        vertex_stride: u64,
        index_format: wgpu::IndexFormat,
    ) -> Result<Handle> {
        if vertex_stride == 0 {
            anyhow::bail!("vertex_stride must be > 0");
        }
        if (vertices.len() as u64) % vertex_stride != 0 {
            anyhow::bail!("vertex buffer length must be divisible by vertex_stride");
        }

        let index_stride = match index_format {
            wgpu::IndexFormat::Uint16 => 2usize,
            wgpu::IndexFormat::Uint32 => 4usize,
        };
        if !indices.len().is_multiple_of(index_stride) {
            anyhow::bail!("index buffer length is not aligned with index format");
        }

        // allocate slot
        let idx = {
            let mut slab = self.mesh_slab.write();
            let mut gens = self.mesh_gens.write();
            let mut slot = None;
            for (i, s) in slab.iter().enumerate() {
                if s.is_none() {
                    slot = Some(i);
                    break;
                }
            }
            let index = if let Some(i) = slot {
                i
            } else {
                slab.push(None);
                gens.push(0u8);
                slab.len() - 1
            };
            gens[index] = gens[index].wrapping_add(1);
            index
        };

        // create buffers
        let vb = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh_vb"),
                contents: vertices,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
        let ib = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh_ib"),
                contents: indices,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            });

        let index_count = (indices.len() / index_stride) as u32;

        let rec = MeshRecord {
            vertex_buffer: vb,
            index_buffer: ib,
            index_count,
            refcount: 1,
            generation: self.mesh_gens.read()[idx],
        };

        self.mesh_slab.write()[idx] = Some(rec);
        Ok(Handle::new(idx as u32, self.mesh_gens.read()[idx]))
    }

    /// Create material record (params buffer + store texture handles). BindGroup created lazily and cached.
    pub fn create_material(
        &self,
        params_bytes: &[u8],
        base: Option<Handle>,
        mr: Option<Handle>,
        normal: Option<Handle>,
        ao: Option<Handle>,
    ) -> Result<Handle> {
        // allocate slot
        let idx = {
            let mut slab = self.material_slab.write();
            let mut gens = self.material_gens.write();
            let mut slot = None;
            for (i, s) in slab.iter().enumerate() {
                if s.is_none() {
                    slot = Some(i);
                    break;
                }
            }
            let index = if let Some(i) = slot {
                i
            } else {
                slab.push(None);
                gens.push(0u8);
                slab.len() - 1
            };
            gens[index] = gens[index].wrapping_add(1);
            index
        };

        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("material_params"),
                contents: params_bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let rec = MaterialRecord {
            params_buffer: params_buf,
            base_tex: base,
            mr_tex: mr,
            normal_tex: normal,
            ao_tex: ao,
            refcount: 1,
            generation: self.material_gens.read()[idx],
        };

        self.material_slab.write()[idx] = Some(rec);
        Ok(Handle::new(idx as u32, self.material_gens.read()[idx]))
    }

    /// Get or create bind group for a material and pipeline id. Pipeline id is user-defined stable id for pipeline layout.
    pub fn get_bind_group_for_material(
        &self,
        material: Handle,
        pipeline_layout: &wgpu::BindGroupLayout,
        pipeline_id: u64,
    ) -> Option<wgpu::BindGroup> {
        if !material.is_valid() {
            return None;
        }
        let key = BindGroupKey {
            material_handle: material.0,
            pipeline_id,
        };
        // check cache
        if let Some(bg) = self.bind_group_cache.lock().get(&key) {
            return Some(bg.clone());
        }

        // build bind group entries from material record
        let midx = material.index();
        let mat_slab = self.material_slab.read();
        if midx >= mat_slab.len() {
            return None;
        }
        let mat_rec = mat_slab[midx].as_ref()?;
        // resolve textures (use dummy if missing)
        let (base_view, base_sampler) = mat_rec
            .base_tex
            .and_then(|h| self.get_texture_view_sampler(h))
            .unwrap_or_else(|| {
                self.get_texture_view_sampler(self.dummy_texture)
                    .expect("dummy present")
            });
        let (mr_view, _) = mat_rec
            .mr_tex
            .and_then(|h| self.get_texture_view_sampler(h))
            .unwrap_or_else(|| {
                self.get_texture_view_sampler(self.dummy_texture)
                    .expect("dummy present")
            });
        let (normal_view, _) = mat_rec
            .normal_tex
            .and_then(|h| self.get_texture_view_sampler(h))
            .unwrap_or_else(|| {
                self.get_texture_view_sampler(self.dummy_texture)
                    .expect("dummy present")
            });
        let (ao_view, _) = mat_rec
            .ao_tex
            .and_then(|h| self.get_texture_view_sampler(h))
            .unwrap_or_else(|| {
                self.get_texture_view_sampler(self.dummy_texture)
                    .expect("dummy present")
            });

        // create bind group
        let entries = &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: mat_rec.params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&base_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&mr_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&normal_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&ao_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(&base_sampler),
            },
        ];

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material_bind_group_cached"),
            layout: pipeline_layout,
            entries,
        });

        self.bind_group_cache.lock().put(key, bg.clone());
        Some(bg)
    }

    /// Release a handle (decrement refcount). Actual free happens in tick() GC.
    pub fn release(&self, h: Handle) {
        if !h.is_valid() {
            return;
        }
        // try textures
        {
            let mut slab = self.texture_slab.write();
            let idx = h.index();
            if idx < slab.len() {
                if let Some(rec) = slab[idx].as_mut() {
                    rec.refcount = rec.refcount.saturating_sub(1);
                    return;
                }
            }
        }
        // meshes
        {
            let mut slab = self.mesh_slab.write();
            let idx = h.index();
            if idx < slab.len() {
                if let Some(rec) = slab[idx].as_mut() {
                    rec.refcount = rec.refcount.saturating_sub(1);
                    return;
                }
            }
        }
        // materials
        {
            let mut slab = self.material_slab.write();
            let idx = h.index();
            if idx < slab.len() {
                if let Some(rec) = slab[idx].as_mut() {
                    rec.refcount = rec.refcount.saturating_sub(1);
                    return;
                }
            }
        }
    }

    /// Must be called regularly on main thread. Processes upload queue, evicts LRU, and frees unused resources.
    pub fn tick(&self) {
        // 1) process upload queue
        let mut tasks = Vec::new();
        {
            let mut q = self.upload_queue.lock();
            if !q.is_empty() {
                tasks.append(&mut *q);
            }
        }
        for task in tasks {
            match task {
                UploadTask::Texture {
                    handle_index,
                    bytes,
                    width,
                    height,
                    format,
                } => {
                    // decode bytes if necessary (assume bytes are raw RGBA8 if format provided)
                    // For generality, try to decode via image crate if bytes are encoded
                    let rgba: Vec<u8>;
                    let (w, h) = (width, height);
                    if let Ok(img) = image::load_from_memory(&bytes) {
                        let img = img.to_rgba8();
                        rgba = img.into_vec();
                    } else {
                        // assume bytes are already RGBA8
                        let expected = (w as usize).saturating_mul(h as usize).saturating_mul(4);
                        if bytes.len() != expected {
                            log::warn!(
                                "Skipping texture upload: raw byte length {} does not match expected RGBA8 size {} for {}x{}",
                                bytes.len(),
                                expected,
                                w,
                                h
                            );
                            continue;
                        }
                        rgba = bytes;
                    }

                    // create texture
                    let size = wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    };
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

                    // write texture
                    self.queue.write_texture(
                        wgpu::ImageCopyTexture {
                            texture: &tex,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        &rgba,
                        wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: std::num::NonZeroU32::new(4 * w),
                            rows_per_image: std::num::NonZeroU32::new(h),
                        },
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

                    // update slab record
                    {
                        let mut slab = self.texture_slab.write();
                        if let Some(rec_opt) = slab.get_mut(handle_index) {
                            if let Some(rec) = rec_opt {
                                // update view and sampler
                                rec.view = view;
                                rec.sampler = sampler;
                                let size_bytes = (rgba.len() as u64);
                                rec.size_bytes = size_bytes;
                                // update LRU and total bytes
                                self.texture_lru.lock().put(handle_index, ());
                                let mut current_bytes = self.current_texture_bytes.lock();
                                *current_bytes = current_bytes.saturating_add(size_bytes);
                            }
                        }
                    }
                }
            }
        }

        // 2) Evict LRU if over budget
        let mut current = *self.current_texture_bytes.lock();
        let max_bytes = self.cfg.max_texture_bytes;
        while current > max_bytes {
            // evict least recently used
            if let Some((idx, _)) = self.texture_lru.lock().pop_lru() {
                // free texture if refcount == 0
                let mut slab = self.texture_slab.write();
                if let Some(rec) = slab.get_mut(idx) {
                    if let Some(tr) = rec {
                        if tr.refcount == 0 {
                            current = current.saturating_sub(tr.size_bytes);
                            *self.current_texture_bytes.lock() = current;
                            // remove from hash map
                            self.texture_hash_map.write().remove(&tr.hash);
                            // drop record
                            *rec = None;
                            continue;
                        } else {
                            // still referenced; reinsert to LRU to avoid busy loop
                            self.texture_lru.lock().put(idx, ());
                        }
                    }
                }
                break;
            } else {
                break;
            }
        }

        // 3) GC for meshes and materials with refcount == 0
        {
            let mut slab = self.mesh_slab.write();
            for rec_opt in slab.iter_mut() {
                if let Some(rec) = rec_opt {
                    if rec.refcount == 0 {
                        *rec_opt = None;
                    }
                }
            }
        }
        {
            let mut slab = self.material_slab.write();
            for rec_opt in slab.iter_mut() {
                if let Some(rec) = rec_opt {
                    if rec.refcount == 0 {
                        *rec_opt = None;
                    }
                }
            }
        }

        // 4) Trim bind group cache
        self.bind_group_cache
            .lock()
            .resize(self.cfg.max_bind_group_cache);
    }

    // ---------- Helpers ----------

    fn create_dummy_record(device: &wgpu::Device, queue: &wgpu::Queue) -> TextureRecord {
        // 1x1 white RGBA8
        let rgba = [255u8, 255u8, 255u8, 255u8];
        let size = wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };
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
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(4),
                rows_per_image: std::num::NonZeroU32::new(1),
            },
            size,
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        TextureRecord {
            view,
            sampler,
            size_bytes: 4,
            refcount: 1,
            generation: 1,
            hash: 0,
        }
    }

    // helper to get texture record by handle
    fn get_texture_record(&self, h: Handle) -> Option<TextureRecord> {
        if !h.is_valid() {
            return None;
        }
        let idx = h.index();
        let slab = self.texture_slab.read();
        if idx >= slab.len() {
            return None;
        }
        slab[idx].as_ref().map(|r| TextureRecord {
            view: r.view.clone(),
            sampler: r.sampler.clone(),
            size_bytes: r.size_bytes,
            refcount: r.refcount,
            generation: r.generation,
            hash: r.hash,
        })
    }

    // register dummy into slab after creation (called in new)
    fn register_dummy_in_slab(&self) -> Handle {
        let dummy = Self::create_dummy_record(&self.device, &self.queue);

        // allocate slot
        let idx = {
            let mut slab = self.texture_slab.write();
            let mut gens = self.texture_gens.write();
            slab.push(Some(TextureRecord {
                view: dummy.view.clone(),
                sampler: dummy.sampler.clone(),
                size_bytes: dummy.size_bytes,
                refcount: dummy.refcount,
                generation: dummy.generation,
                hash: dummy.hash,
            }));
            gens.push(1u8);
            slab.len() - 1
        };

        // register in LRU and hash map
        self.texture_lru.lock().put(idx, ());
        *self.current_texture_bytes.lock() += 4;
        self.texture_hash_map.write().insert(0, idx);

        Handle::new(idx as u32, self.texture_gens.read()[idx])
    }
}

// ---------- End of file ----------
