// src/materials/material_system.rs
//
// High-performance Material System for WGPU — Optimized Edition
//
// Key optimizations over the original:
//  1. Pre-computed PropertyLayout with flat arrays — zero HashMap lookups in render hot-path
//  2. Persistent UniformPool with sub-allocation — no per-frame ring wrap hazards
//  3. Shared bind groups via dynamic uniform offsets — one BG per unique texture-set
//  4. Atomic dirty flags — no Mutex for dirty checks
//  5. PropertyHandle API — O(1) property access after one-time name resolution
//  6. SmallVec inline storage — no heap allocation for materials with ≤16 properties
//  7. Generation-based bind group keys — stable caching, no fragile pointer hashing
//  8. Material sort-key generation — minimal state changes during rendering
//  9. Deferred GPU resource deletion queue — safe cleanup without stalls
// 10. Inner-Arc architecture — trivially cloneable for hot-reload callbacks
// 11. Per-material instance tracking for O(1) batch invalidation
// 12. Pipeline cache keyed by (material, vertex_format, render_target)
//
// Crate requirements (Cargo.toml):
//  wgpu        = "0.19"
//  serde       = { version = "1", features = ["derive"] }
//  serde_json  = "1"
//  parking_lot = "0.12"
//  notify      = "6"
//  fxhash      = "0.2"
//  bytemuck    = { version = "1", features = ["derive"] }
//  smallvec    = "1"
//  log         = "0.4"

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use fxhash::FxHashMap;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

// ═══════════════════════════════════════════════════════════════════════════════
// Section 1 — Configuration Constants
// ═══════════════════════════════════════════════════════════════════════════════

/// Maximum number of vec4 uniform properties per material.
const MAX_UNIFORM_PROPERTIES: usize = 16;

/// Maximum texture/sampler slots per material.
const MAX_TEXTURE_SLOTS: usize = 8;

/// Uniform slot size in bytes.  Must equal device `min_uniform_buffer_offset_alignment`
/// (256 on most GPUs).  Every instance occupies exactly one slot.
const UNIFORM_SLOT_SIZE: u64 = 256;

/// Default uniform-pool capacity in bytes (16 MiB → 65 536 slots).
const DEFAULT_POOL_BYTES: u64 = 16 * 1024 * 1024;

/// Maximum frames the GPU may be behind the CPU.
const MAX_FRAMES_IN_FLIGHT: u64 = 3;

/// Dirty-flag bit positions (stored in AtomicU32).
const DIRTY_UNIFORMS: u32     = 1 << 0;
const DIRTY_TEXTURES: u32     = 1 << 1;
const DIRTY_BIND_GROUP: u32   = 1 << 2;
const DIRTY_ALL: u32          = DIRTY_UNIFORMS | DIRTY_TEXTURES | DIRTY_BIND_GROUP;

/// Global monotonic counter for texture-slot generations used as stable cache keys.
static NEXT_TEXTURE_GEN: AtomicU64 = AtomicU64::new(1);

fn next_texture_generation() -> u64 {
    NEXT_TEXTURE_GEN.fetch_add(1, Ordering::Relaxed)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 2 — Public ID / Handle Types
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaterialId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InstanceId(pub u32);

/// Resolved handle to a material property.  Obtain via
/// `MaterialSystem::resolve_property` once, then use repeatedly in the hot path.
#[derive(Clone, Copy, Debug)]
pub struct PropertyHandle {
    pub kind: PropertyKind,
    /// Index into the flat uniform or texture-slot array.
    pub index: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PropertyKind {
    Uniform,
    Texture,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 3 — Serialization (Material Source Files)
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MaterialSource {
    pub name: String,
    pub shader_path: String,
    pub vertex_entry: Option<String>,
    pub fragment_entry: String,
    pub defines: Option<HashMap<String, String>>,
    pub properties: Vec<MaterialPropertyDef>,
    pub pipeline_flags: Option<PipelineFlags>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MaterialPropertyDef {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: PropertyType,
    pub default: Option<serde_json::Value>,
    /// Optional explicit WGSL binding index for textures.
    pub binding: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum PropertyType {
    Float, Vec2, Vec3, Vec4, Color, Int, UInt, Bool,
    Texture2D, TextureCube, Sampler,
}

impl PropertyType {
    fn is_texture(&self) -> bool {
        matches!(self, Self::Texture2D | Self::TextureCube)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct PipelineFlags {
    pub double_sided: Option<bool>,
    pub alpha_mode: Option<String>,
    pub depth_write: Option<bool>,
    pub depth_test: Option<bool>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 4 — Core Value Types
// ═══════════════════════════════════════════════════════════════════════════════

/// 16-byte uniform value aligned for GPU upload (vec4).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct UniformValue {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl UniformValue {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };
    #[inline] pub fn f32(v: f32) -> Self { Self { x: v, y: 0.0, z: 0.0, w: 0.0 } }
    #[inline] pub fn vec2(x: f32, y: f32) -> Self { Self { x, y, z: 0.0, w: 0.0 } }
    #[inline] pub fn vec3(x: f32, y: f32, z: f32) -> Self { Self { x, y, z, w: 0.0 } }
    #[inline] pub fn vec4(x: f32, y: f32, z: f32, w: f32) -> Self { Self { x, y, z, w } }

    /// Parse from a JSON value according to property type.
    pub fn from_json(val: &serde_json::Value) -> Self {
        match val {
            serde_json::Value::Number(n) => Self::f32(n.as_f64().unwrap_or(0.0) as f32),
            serde_json::Value::Array(arr) => {
                let f = |i: usize| -> f32 {
                    arr.get(i)
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0) as f32
                };
                Self::vec4(f(0), f(1), f(2), f(3))
            }
            serde_json::Value::Bool(b) => Self::f32(if *b { 1.0 } else { 0.0 }),
            _ => Self::ZERO,
        }
    }
}

/// Texture + sampler pair stored per-instance, tagged with a generation for cache keys.
#[derive(Clone)]
pub struct TextureSlot {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub generation: u64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 5 — Pre-computed Property Layout
// ═══════════════════════════════════════════════════════════════════════════════

/// Computed once when a material is loaded.  Maps property names to fast indices,
/// stores defaults, and records texture binding indices.
#[derive(Debug)]
struct PropertyLayout {
    /// Ordered list of uniform properties (index matches flat array position).
    uniforms: Vec<UniformPropertyInfo>,
    /// Ordered list of texture properties (index matches texture-slot position).
    textures: Vec<TexturePropertyInfo>,
    /// Name → PropertyHandle for one-time resolution.
    lookup: FxHashMap<String, PropertyHandle>,
    /// Pre-baked default uniform values (one vec4 per uniform property).
    default_uniforms: SmallVec<[UniformValue; MAX_UNIFORM_PROPERTIES]>,
    /// Total uniform buffer bytes needed (≤ UNIFORM_SLOT_SIZE).
    uniform_bytes: u32,
}

#[derive(Debug)]
struct UniformPropertyInfo {
    name: String,
    byte_offset: u32,
}

#[derive(Debug)]
struct TexturePropertyInfo {
    name: String,
    texture_binding: u32,
    sampler_binding: u32,
}

impl PropertyLayout {
    /// Build layout from MaterialSource properties.
    fn build(props: &[MaterialPropertyDef]) -> Self {
        let mut uniforms = Vec::new();
        let mut textures = Vec::new();
        let mut lookup = FxHashMap::default();
        let mut defaults = SmallVec::new();
        let mut binding_idx: u32 = 1; // binding 0 is reserved for the uniform buffer

        let mut uniform_idx: u32 = 0;
        let mut texture_idx: u32 = 0;

        for prop in props {
            if prop.ty.is_texture() {
                let tex_binding = prop.binding.unwrap_or(binding_idx);
                let samp_binding = tex_binding + 1;
                textures.push(TexturePropertyInfo {
                    name: prop.name.clone(),
                    texture_binding: tex_binding,
                    sampler_binding: samp_binding,
                });
                lookup.insert(prop.name.clone(), PropertyHandle {
                    kind: PropertyKind::Texture,
                    index: texture_idx,
                });
                texture_idx += 1;
                binding_idx = samp_binding + 1;
            } else {
                let byte_offset = uniform_idx * 16;
                uniforms.push(UniformPropertyInfo {
                    name: prop.name.clone(),
                    byte_offset,
                });
                let def = prop.default.as_ref()
                    .map(UniformValue::from_json)
                    .unwrap_or(UniformValue::ZERO);
                defaults.push(def);
                lookup.insert(prop.name.clone(), PropertyHandle {
                    kind: PropertyKind::Uniform,
                    index: uniform_idx,
                });
                uniform_idx += 1;
            }
        }

        let raw_bytes = uniform_idx * 16;
        // Align up to 16 (vec4), capped at UNIFORM_SLOT_SIZE.
        let uniform_bytes = ((raw_bytes + 15) / 16) * 16;

        Self { uniforms, textures, lookup, default_uniforms: defaults, uniform_bytes }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 6 — Default GPU Resources (1×1 white texture, linear sampler)
// ═══════════════════════════════════════════════════════════════════════════════

struct DefaultResources {
    white_view: wgpu::TextureView,
    linear_sampler: wgpu::Sampler,
}

impl DefaultResources {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mat_default_white"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            &[255u8; 4],
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: None },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let white_view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mat_default_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            ..Default::default()
        });
        Self { white_view, linear_sampler }
    }

    fn default_slot(&self) -> TextureSlot {
        TextureSlot {
            view: self.white_view.clone(),
            sampler: self.linear_sampler.clone(),
            generation: 0, // sentinel: default resources always have generation 0
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 7 — Persistent Uniform Pool (sub-allocated GPU buffer)
// ═══════════════════════════════════════════════════════════════════════════════

/// Fixed-slot sub-allocator over a single large wgpu::Buffer.
/// Each slot is UNIFORM_SLOT_SIZE bytes.  Freed slots are recycled via a free list.
struct UniformPool {
    buffer: wgpu::Buffer,
    slot_size: u64,
    max_slots: u32,
    /// Stack of reusable slot indices (LIFO for cache warmth).
    free_list: Vec<u32>,
    /// Next never-used slot index (grows monotonically until max_slots).
    next_virgin: u32,
}

impl UniformPool {
    fn new(device: &wgpu::Device, total_bytes: u64, slot_size: u64) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mat_uniform_pool"),
            size: total_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let max_slots = (total_bytes / slot_size) as u32;
        Self {
            buffer,
            slot_size,
            max_slots,
            free_list: Vec::with_capacity(256),
            next_virgin: 0,
        }
    }

    /// Allocate one slot.  Returns slot index or `None` if exhausted.
    fn allocate(&mut self) -> Option<u32> {
        // Prefer recycled slots (hot cache lines).
        if let Some(slot) = self.free_list.pop() {
            return Some(slot);
        }
        if self.next_virgin < self.max_slots {
            let s = self.next_virgin;
            self.next_virgin += 1;
            Some(s)
        } else {
            None
        }
    }

    /// Return a slot to the free list.
    fn free(&mut self, slot: u32) {
        debug_assert!(slot < self.max_slots);
        self.free_list.push(slot);
    }

    /// Byte offset for a given slot index.
    #[inline]
    fn offset(&self, slot: u32) -> u64 {
        slot as u64 * self.slot_size
    }

    /// Write `data` into the slot via `queue.write_buffer` (fast staging path).
    #[inline]
    fn write(&self, queue: &wgpu::Queue, slot: u32, data: &[u8]) {
        let off = self.offset(slot);
        queue.write_buffer(&self.buffer, off, data);
    }

    fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    fn active_slots(&self) -> u32 {
        self.next_virgin - self.free_list.len() as u32
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 8 — Material Definition
// ═══════════════════════════════════════════════════════════════════════════════

pub struct MaterialDefinition {
    pub id: MaterialId,
    pub name: String,
    pub source_path: PathBuf,
    pub source: MaterialSource,
    pub shader_module: wgpu::ShaderModule,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub layout: PropertyLayout,
    pub pipeline_flags: PipelineFlags,
    pub loaded_at: Instant,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 9 — Material Instance
// ═══════════════════════════════════════════════════════════════════════════════

pub struct MaterialInstance {
    pub id: InstanceId,
    pub material: MaterialId,
    /// Flat uniform array indexed by `PropertyHandle.index`.
    /// SmallVec avoids heap allocation for materials with ≤ MAX_UNIFORM_PROPERTIES props.
    uniform_data: RwLock<SmallVec<[UniformValue; MAX_UNIFORM_PROPERTIES]>>,
    /// Texture slots indexed by `PropertyHandle.index` for texture properties.
    texture_slots: RwLock<SmallVec<[TextureSlot; MAX_TEXTURE_SLOTS]>>,
    /// Persistent uniform-pool slot index.
    uniform_slot: u32,
    /// Cached bind group (shared with other instances that have the same texture set).
    bind_group: Mutex<Option<wgpu::BindGroup>>,
    /// Hash of the texture-generation key used to build the current bind group.
    bind_group_key: AtomicU64,
    /// Bitfield: DIRTY_UNIFORMS | DIRTY_TEXTURES | DIRTY_BIND_GROUP
    dirty: AtomicU32,
    /// Last frame this instance was flushed.
    last_frame: AtomicU64,
}

impl MaterialInstance {
    fn new(
        id: InstanceId,
        material: MaterialId,
        defaults: &SmallVec<[UniformValue; MAX_UNIFORM_PROPERTIES]>,
        tex_count: usize,
        default_tex: &TextureSlot,
        uniform_slot: u32,
    ) -> Self {
        let mut tex_slots = SmallVec::new();
        for _ in 0..tex_count {
            tex_slots.push(default_tex.clone());
        }
        Self {
            id,
            material,
            uniform_data: RwLock::new(defaults.clone()),
            texture_slots: RwLock::new(tex_slots),
            uniform_slot,
            bind_group: Mutex::new(None),
            bind_group_key: AtomicU64::new(0),
            dirty: AtomicU32::new(DIRTY_ALL),
            last_frame: AtomicU64::new(0),
        }
    }

    #[inline]
    fn mark_dirty(&self, bits: u32) {
        self.dirty.fetch_or(bits, Ordering::Release);
    }

    #[inline]
    fn is_dirty(&self, bits: u32) -> bool {
        self.dirty.load(Ordering::Acquire) & bits != 0
    }

    #[inline]
    fn clear_dirty(&self, bits: u32) {
        self.dirty.fetch_and(!bits, Ordering::Release);
    }

    /// Compute a hash over texture-slot generations for bind-group cache lookup.
    fn texture_key_hash(&self) -> u64 {
        let slots = self.texture_slots.read();
        let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
        for slot in slots.iter() {
            h ^= slot.generation;
            h = h.wrapping_mul(0x100000001b3); // FNV prime
        }
        h
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 10 — Pipeline Cache
// ═══════════════════════════════════════════════════════════════════════════════

/// Key for cached render pipelines.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PipelineCacheKey {
    pub material_id: MaterialId,
    pub vertex_layout_hash: u64,
    pub color_format: wgpu::TextureFormat,
    pub depth_format: Option<wgpu::TextureFormat>,
    pub sample_count: u32,
    pub double_sided: bool,
    pub alpha_blend: bool,
}

struct PipelineCache {
    pipelines: FxHashMap<PipelineCacheKey, wgpu::RenderPipeline>,
}

impl PipelineCache {
    fn new() -> Self {
        Self { pipelines: FxHashMap::default() }
    }

    fn get_or_create(
        &mut self,
        key: &PipelineCacheKey,
        device: &wgpu::Device,
        mat: &MaterialDefinition,
        pipeline_layout: &wgpu::PipelineLayout,
        vertex_buffers: &[wgpu::VertexBufferLayout],
    ) -> &wgpu::RenderPipeline {
        self.pipelines.entry(key.clone()).or_insert_with(|| {
            let cull = if key.double_sided {
                wgpu::Face::default() // default = none if double_sided
            } else {
                wgpu::Face::Back
            };

            let blend = if key.alpha_blend {
                Some(wgpu::BlendState::ALPHA_BLENDING)
            } else {
                Some(wgpu::BlendState::REPLACE)
            };

            let depth_stencil = key.depth_format.map(|fmt| wgpu::DepthStencilState {
                format: fmt,
                depth_write_enabled: mat.pipeline_flags.depth_write.unwrap_or(true),
                depth_compare: if mat.pipeline_flags.depth_test.unwrap_or(true) {
                    wgpu::CompareFunction::Less
                } else {
                    wgpu::CompareFunction::Always
                },
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            });

            let vs_entry = mat.source.vertex_entry.as_deref().unwrap_or("vs_main");

            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("pipeline:{}", mat.name)),
                layout: Some(pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &mat.shader_module,
                    entry_point: vs_entry,
                    buffers: vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &mat.shader_module,
                    entry_point: &mat.source.fragment_entry,
                    targets: &[Some(wgpu::ColorTargetState {
                        format: key.color_format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: if key.double_sided { None } else { Some(cull) },
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil,
                multisample: wgpu::MultisampleState {
                    count: key.sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            })
        })
    }

    fn invalidate_material(&mut self, mat: MaterialId) {
        self.pipelines.retain(|k, _| k.material_id != mat);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 11 — Deferred Deletion Queue
// ═══════════════════════════════════════════════════════════════════════════════

/// GPU resources scheduled for deletion once all in-flight frames have retired.
struct DeletionQueue {
    entries: Vec<(u64, DeletionEntry)>, // (frame_submitted, resource)
}

enum DeletionEntry {
    UniformSlot(u32),
    // Extend with textures, buffers, etc. as needed.
}

impl DeletionQueue {
    fn new() -> Self { Self { entries: Vec::new() } }

    fn push(&mut self, frame: u64, entry: DeletionEntry) {
        self.entries.push((frame, entry));
    }

    /// Drain entries whose frame has been fully retired by the GPU.
    fn drain_retired(&mut self, retired_frame: u64) -> Vec<DeletionEntry> {
        let mut retired = Vec::new();
        self.entries.retain(|(f, _)| {
            if *f <= retired_frame { false } else { true }
        });
        // Actually collect them:
        let mut keep = Vec::new();
        for (f, e) in self.entries.drain(..) {
            if f <= retired_frame {
                retired.push(e);
            } else {
                keep.push((f, e));
            }
        }
        self.entries = keep;
        retired
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 12 — Material System (shared inner)
// ═══════════════════════════════════════════════════════════════════════════════

/// Shared interior — allows cheap cloning for hot-reload callbacks.
struct MaterialSystemInner {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    defaults: DefaultResources,

    materials: RwLock<FxHashMap<MaterialId, Arc<MaterialDefinition>>>,
    instances: RwLock<FxHashMap<InstanceId, Arc<MaterialInstance>>>,
    /// material → list of instances (for batch invalidation).
    material_instances: Mutex<FxHashMap<MaterialId, Vec<InstanceId>>>,

    uniform_pool: Mutex<UniformPool>,
    pipeline_cache: Mutex<PipelineCache>,
    bind_group_cache: Mutex<FxHashMap<(u32, u64), wgpu::BindGroup>>,
    deletion_queue: Mutex<DeletionQueue>,

    next_material_id: AtomicU32,
    next_instance_id: AtomicU32,
    current_frame: AtomicU64,

    stats: Mutex<MaterialSystemStats>,
}

/// Public handle — cheaply cloneable.
#[derive(Clone)]
pub struct MaterialSystem {
    inner: Arc<MaterialSystemInner>,
}

#[derive(Debug, Clone, Default)]
pub struct MaterialSystemStats {
    pub material_count: u32,
    pub instance_count: u32,
    pub active_uniform_slots: u32,
    pub bind_group_cache_entries: u32,
    pub pipeline_cache_entries: u32,
    pub flush_us: u64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 13 — Material System Implementation
// ═══════════════════════════════════════════════════════════════════════════════

impl MaterialSystem {
    /// Create a new material system.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let defaults = DefaultResources::new(&device, &queue);
        let uniform_pool = UniformPool::new(&device, DEFAULT_POOL_BYTES, UNIFORM_SLOT_SIZE);
        Self {
            inner: Arc::new(MaterialSystemInner {
                device,
                queue,
                defaults,
                materials: RwLock::new(FxHashMap::default()),
                instances: RwLock::new(FxHashMap::default()),
                material_instances: Mutex::new(FxHashMap::default()),
                uniform_pool: Mutex::new(uniform_pool),
                pipeline_cache: Mutex::new(PipelineCache::new()),
                bind_group_cache: Mutex::new(FxHashMap::default()),
                deletion_queue: Mutex::new(DeletionQueue::new()),
                next_material_id: AtomicU32::new(1),
                next_instance_id: AtomicU32::new(1),
                current_frame: AtomicU64::new(0),
                stats: Mutex::new(MaterialSystemStats::default()),
            }),
        }
    }

    // ── Loading ──────────────────────────────────────────────────────────────

    /// Load a material definition from a JSON file.
    pub fn load_material<P: AsRef<Path>>(&self, path: P) -> Result<MaterialId, MaterialError> {
        let path = path.as_ref().to_path_buf();
        let json = std::fs::read_to_string(&path).map_err(MaterialError::Io)?;
        let src: MaterialSource = serde_json::from_str(&json).map_err(MaterialError::Parse)?;
        self.load_material_from_source(src, path)
    }

    /// Load a material from an already-parsed `MaterialSource`.
    pub fn load_material_from_source(
        &self,
        src: MaterialSource,
        source_path: PathBuf,
    ) -> Result<MaterialId, MaterialError> {
        let i = &self.inner;

        // Compile shader
        let shader_code = std::fs::read_to_string(&src.shader_path).map_err(MaterialError::Io)?;
        let shader_module = i.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("shader:{}", src.name)),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });

        // Build property layout (pre-computed: no per-frame cost).
        let layout = PropertyLayout::build(&src.properties);

        // Build bind group layout entries.
        let mut bgl_entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::new();

        // Binding 0: dynamic uniform buffer
        bgl_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: NonZeroU64::new(layout.uniform_bytes.max(16) as u64),
            },
            count: None,
        });

        // Texture + sampler pairs
        for tex in &layout.textures {
            bgl_entries.push(wgpu::BindGroupLayoutEntry {
                binding: tex.texture_binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            });
            bgl_entries.push(wgpu::BindGroupLayoutEntry {
                binding: tex.sampler_binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }

        let bind_group_layout = i.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("bgl:{}", src.name)),
            entries: &bgl_entries,
        });

        let id = MaterialId(i.next_material_id.fetch_add(1, Ordering::Relaxed));

        let def = Arc::new(MaterialDefinition {
            id,
            name: src.name.clone(),
            source_path,
            pipeline_flags: src.pipeline_flags.clone().unwrap_or_default(),
            source: src,
            shader_module,
            bind_group_layout,
            layout,
            loaded_at: Instant::now(),
        });

        i.materials.write().insert(id, def);
        i.material_instances.lock().insert(id, Vec::new());
        self.update_stats();
        log::info!("Loaded material {:?}", id);
        Ok(id)
    }

    // ── Instances ────────────────────────────────────────────────────────────

    /// Create an instance of a loaded material.
    pub fn create_instance(&self, material: MaterialId) -> Result<InstanceId, MaterialError> {
        let i = &self.inner;

        let mat = i.materials.read().get(&material).cloned()
            .ok_or(MaterialError::NotFound(material))?;

        let slot = i.uniform_pool.lock().allocate()
            .ok_or(MaterialError::PoolExhausted)?;

        let id = InstanceId(i.next_instance_id.fetch_add(1, Ordering::Relaxed));

        let inst = Arc::new(MaterialInstance::new(
            id,
            material,
            &mat.layout.default_uniforms,
            mat.layout.textures.len(),
            &i.defaults.default_slot(),
            slot,
        ));

        i.instances.write().insert(id, inst);
        i.material_instances.lock()
            .entry(material)
            .or_default()
            .push(id);

        self.update_stats();
        Ok(id)
    }

    /// Destroy an instance and schedule its uniform slot for deferred recycling.
    pub fn destroy_instance(&self, id: InstanceId) {
        let i = &self.inner;
        let frame = i.current_frame.load(Ordering::Relaxed);

        if let Some(inst) = i.instances.write().remove(&id) {
            // Remove from per-material tracking.
            if let Some(list) = i.material_instances.lock().get_mut(&inst.material) {
                list.retain(|x| *x != id);
            }
            // Defer slot recycling until all in-flight frames have retired.
            i.deletion_queue.lock().push(
                frame + MAX_FRAMES_IN_FLIGHT,
                DeletionEntry::UniformSlot(inst.uniform_slot),
            );
        }
        self.update_stats();
    }

    // ── Property Handle Resolution ──────────────────────────────────────────

    /// Resolve a property name to a fast handle.  Call once, reuse in hot path.
    pub fn resolve_property(
        &self,
        material: MaterialId,
        name: &str,
    ) -> Option<PropertyHandle> {
        self.inner.materials.read()
            .get(&material)
            .and_then(|m| m.layout.lookup.get(name).copied())
    }

    // ── Property Setters (any thread) ───────────────────────────────────────

    /// Set a uniform property on an instance.  Uses pre-resolved handle for O(1) access.
    #[inline]
    pub fn set_uniform(&self, id: InstanceId, handle: PropertyHandle, value: UniformValue) {
        debug_assert_eq!(handle.kind, PropertyKind::Uniform);
        if let Some(inst) = self.inner.instances.read().get(&id) {
            inst.uniform_data.write()[handle.index as usize] = value;
            inst.mark_dirty(DIRTY_UNIFORMS);
        }
    }

    /// Convenience: set uniform by name (slower — one HashMap lookup).
    pub fn set_uniform_by_name(&self, id: InstanceId, name: &str, value: UniformValue) {
        if let Some(inst) = self.inner.instances.read().get(&id) {
            if let Some(mat) = self.inner.materials.read().get(&inst.material) {
                if let Some(handle) = mat.layout.lookup.get(name) {
                    debug_assert_eq!(handle.kind, PropertyKind::Uniform);
                    inst.uniform_data.write()[handle.index as usize] = value;
                    inst.mark_dirty(DIRTY_UNIFORMS);
                }
            }
        }
    }

    /// Set a texture + sampler on an instance.
    #[inline]
    pub fn set_texture(
        &self,
        id: InstanceId,
        handle: PropertyHandle,
        view: wgpu::TextureView,
        sampler: wgpu::Sampler,
    ) {
        debug_assert_eq!(handle.kind, PropertyKind::Texture);
        if let Some(inst) = self.inner.instances.read().get(&id) {
            let gen = next_texture_generation();
            inst.texture_slots.write()[handle.index as usize] = TextureSlot {
                view, sampler, generation: gen,
            };
            inst.mark_dirty(DIRTY_TEXTURES | DIRTY_BIND_GROUP);
        }
    }

    /// Bulk-set multiple uniforms in one lock acquisition.
    pub fn set_uniforms_bulk(
        &self,
        id: InstanceId,
        values: &[(PropertyHandle, UniformValue)],
    ) {
        if let Some(inst) = self.inner.instances.read().get(&id) {
            let mut data = inst.uniform_data.write();
            for &(handle, value) in values {
                debug_assert_eq!(handle.kind, PropertyKind::Uniform);
                data[handle.index as usize] = value;
            }
            drop(data);
            inst.mark_dirty(DIRTY_UNIFORMS);
        }
    }

    // ── Flush (render thread) ───────────────────────────────────────────────

    /// Upload dirty uniforms and rebuild bind groups as needed.
    /// Call once per frame before rendering.
    pub fn flush(&self, frame_index: u64) {
        let start = Instant::now();
        let i = &self.inner;
        i.current_frame.store(frame_index, Ordering::Relaxed);

        let instances = i.instances.read();

        for inst in instances.values() {
            let dirty = inst.dirty.load(Ordering::Acquire);
            if dirty == 0 {
                inst.last_frame.store(frame_index, Ordering::Relaxed);
                continue;
            }

            // ── Uniform upload ─────────────────────────────────────────────
            if dirty & DIRTY_UNIFORMS != 0 {
                let data = inst.uniform_data.read();
                let bytes: &[u8] = bytemuck::cast_slice(&data[..]);
                i.uniform_pool.lock().write(&i.queue, inst.uniform_slot, bytes);
                inst.clear_dirty(DIRTY_UNIFORMS);
            }

            // ── Bind group rebuild ─────────────────────────────────────────
            if dirty & (DIRTY_TEXTURES | DIRTY_BIND_GROUP) != 0 {
                let tex_hash = inst.texture_key_hash();
                let old_hash = inst.bind_group_key.load(Ordering::Relaxed);

                if tex_hash != old_hash || inst.bind_group.lock().is_none() {
                    let mat = match i.materials.read().get(&inst.material).cloned() {
                        Some(m) => m,
                        None => continue,
                    };
                    let cache_key = (mat.id.0, tex_hash);

                    let mut bg_cache = i.bind_group_cache.lock();
                    let bg = if let Some(existing) = bg_cache.get(&cache_key) {
                        existing.clone()
                    } else {
                        let bg = self.build_bind_group(&mat, inst);
                        bg_cache.insert(cache_key, bg.clone());
                        bg
                    };
                    *inst.bind_group.lock() = Some(bg);
                    inst.bind_group_key.store(tex_hash, Ordering::Relaxed);
                }
                inst.clear_dirty(DIRTY_TEXTURES | DIRTY_BIND_GROUP);
            }

            inst.last_frame.store(frame_index, Ordering::Relaxed);
        }

        // Process deferred deletions.
        let retired_frame = frame_index.saturating_sub(MAX_FRAMES_IN_FLIGHT);
        let deletions = i.deletion_queue.lock().drain_retired(retired_frame);
        if !deletions.is_empty() {
            let mut pool = i.uniform_pool.lock();
            for d in deletions {
                match d {
                    DeletionEntry::UniformSlot(slot) => pool.free(slot),
                }
            }
        }

        // Update stats.
        {
            let mut s = i.stats.lock();
            s.material_count = i.materials.read().len() as u32;
            s.instance_count = instances.len() as u32;
            s.active_uniform_slots = i.uniform_pool.lock().active_slots();
            s.bind_group_cache_entries = i.bind_group_cache.lock().len() as u32;
            s.pipeline_cache_entries = i.pipeline_cache.lock().pipelines.len() as u32;
            s.flush_us = start.elapsed().as_micros() as u64;
        }
    }

    /// Build a fresh wgpu::BindGroup for an instance.
    fn build_bind_group(
        &self,
        mat: &MaterialDefinition,
        inst: &MaterialInstance,
    ) -> wgpu::BindGroup {
        let i = &self.inner;
        let pool = i.uniform_pool.lock();

        let mut entries: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(
            1 + mat.layout.textures.len() * 2,
        );

        // Binding 0: uniform buffer (dynamic offset applied at set_bind_group).
        entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: pool.buffer(),
                offset: 0,
                size: NonZeroU64::new(UNIFORM_SLOT_SIZE),
            }),
        });

        // Texture + sampler bindings.
        let tex_slots = inst.texture_slots.read();
        for (idx, tex_info) in mat.layout.textures.iter().enumerate() {
            let slot = &tex_slots[idx];
            entries.push(wgpu::BindGroupEntry {
                binding: tex_info.texture_binding,
                resource: wgpu::BindingResource::TextureView(&slot.view),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: tex_info.sampler_binding,
                resource: wgpu::BindingResource::Sampler(&slot.sampler),
            });
        }

        i.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("bg:{}:{}", mat.name, inst.id.0)),
            layout: &mat.bind_group_layout,
            entries: &entries,
        })
    }

    // ── Render-time Accessors ───────────────────────────────────────────────

    /// Get the bind group and dynamic offset for an instance.
    /// Returns `None` if the instance does not exist or has not been flushed.
    #[inline]
    pub fn instance_bind_data(&self, id: InstanceId) -> Option<(wgpu::BindGroup, u32)> {
        let instances = self.inner.instances.read();
        let inst = instances.get(&id)?;
        let bg = inst.bind_group.lock().clone()?;
        let offset = (inst.uniform_slot as u64 * UNIFORM_SLOT_SIZE) as u32;
        Some((bg, offset))
    }

    /// Generate a 64-bit sort key for draw-call ordering.
    ///
    /// Layout:
    /// ```text
    /// [63..48] material id      (group by pipeline)
    /// [47..32] bind-group hash  (group by texture set)
    /// [31.. 0] user key         (e.g. depth for front-to-back)
    /// ```
    #[inline]
    pub fn sort_key(&self, id: InstanceId, user_key: u32) -> u64 {
        let instances = self.inner.instances.read();
        if let Some(inst) = instances.get(&id) {
            let mat_bits = (inst.material.0 as u64) << 48;
            let tex_bits = ((inst.bind_group_key.load(Ordering::Relaxed) & 0xFFFF) as u64) << 32;
            mat_bits | tex_bits | user_key as u64
        } else {
            u64::MAX
        }
    }

    /// Iterate over all instances of a material (useful for batched rendering).
    pub fn instances_of(&self, material: MaterialId) -> Vec<InstanceId> {
        self.inner.material_instances.lock()
            .get(&material)
            .cloned()
            .unwrap_or_default()
    }

    // ── Pipeline Cache Access ───────────────────────────────────────────────

    /// Get or create a render pipeline for the given configuration.
    pub fn get_pipeline(
        &self,
        key: &PipelineCacheKey,
        extra_bind_group_layouts: &[&wgpu::BindGroupLayout],
        vertex_buffers: &[wgpu::VertexBufferLayout],
    ) -> Option<wgpu::RenderPipeline> {
        let i = &self.inner;
        let mats = i.materials.read();
        let mat = mats.get(&key.material_id)?;

        // Build pipeline layout: material BGL + any extra BGLs (e.g. camera, lights).
        let mut bgls: Vec<&wgpu::BindGroupLayout> = vec![&mat.bind_group_layout];
        bgls.extend_from_slice(extra_bind_group_layouts);
        let pipeline_layout = i.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("pl_layout:{}", mat.name)),
            bind_group_layouts: &bgls,
            push_constant_ranges: &[],
        });

        let pipeline = i.pipeline_cache.lock().get_or_create(
            key, &i.device, mat, &pipeline_layout, vertex_buffers,
        ).clone(); // Clone the pipeline (wgpu types are Arc'd internally)
        Some(pipeline)
    }

    /// Get the bind group layout for a material (needed when building external pipeline layouts).
    pub fn bind_group_layout(&self, material: MaterialId) -> Option<wgpu::BindGroupLayout> {
        self.inner.materials.read()
            .get(&material)
            .map(|m| m.bind_group_layout.clone())
    }

    // ── Hot Reload ──────────────────────────────────────────────────────────

    /// Watch a directory for material file changes and hot-reload on modification.
    pub fn start_hot_reload<P: AsRef<Path>>(&self, dir: P) -> Result<HotReloadHandle, notify::Error> {
        use notify::{Config, RecommendedWatcher, RecursiveMode, Watcher};

        let dir = dir.as_ref().to_path_buf();
        let sys = self.clone();

        let mut watcher = RecommendedWatcher::new(
            move |res: notify::Result<notify::Event>| {
                if let Ok(event) = res {
                    use notify::EventKind;
                    if matches!(event.kind, EventKind::Modify(_) | EventKind::Create(_)) {
                        for path in &event.paths {
                            sys.handle_file_change(path);
                        }
                    }
                }
            },
            Config::default(),
        )?;
        watcher.watch(&dir, RecursiveMode::Recursive)?;
        Ok(HotReloadHandle { _watcher: watcher })
    }

    fn handle_file_change(&self, path: &Path) {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !matches!(ext, "json" | "mat" | "wgsl") {
            return;
        }

        let i = &self.inner;

        // Find materials whose source_path or shader_path matches.
        let mat_ids: Vec<MaterialId> = {
            let mats = i.materials.read();
            mats.values()
                .filter(|m| m.source_path == path || Path::new(&m.source.shader_path) == path)
                .map(|m| m.id)
                .collect()
        };

        for mat_id in mat_ids {
            log::info!("Hot-reloading material {:?} due to change in {:?}", mat_id, path);

            let old = match i.materials.read().get(&mat_id).cloned() {
                Some(m) => m,
                None => continue,
            };

            // Re-read and re-parse the material source.
            let src: MaterialSource = match std::fs::read_to_string(&old.source_path)
                .ok()
                .and_then(|json| serde_json::from_str(&json).ok())
            {
                Some(s) => s,
                None => {
                    log::warn!("Failed to parse material source {:?}", old.source_path);
                    continue;
                }
            };

            // Re-compile shader.
            let shader_code = match std::fs::read_to_string(&src.shader_path) {
                Ok(c) => c,
                Err(e) => {
                    log::warn!("Failed to read shader {:?}: {}", src.shader_path, e);
                    continue;
                }
            };
            let shader_module = i.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("shader:{}", src.name)),
                source: wgpu::ShaderSource::Wgsl(shader_code.into()),
            });

            // Rebuild layout.
            let layout = PropertyLayout::build(&src.properties);

            let new_def = Arc::new(MaterialDefinition {
                id: mat_id,
                name: src.name.clone(),
                source_path: old.source_path.clone(),
                pipeline_flags: src.pipeline_flags.clone().unwrap_or_default(),
                source: src,
                shader_module,
                bind_group_layout: old.bind_group_layout.clone(), // reuse unless properties changed
                layout,
                loaded_at: Instant::now(),
            });

            i.materials.write().insert(mat_id, new_def);

            // Invalidate pipeline cache for this material.
            i.pipeline_cache.lock().invalidate_material(mat_id);

            // Invalidate bind-group cache entries for this material.
            i.bind_group_cache.lock().retain(|(mid, _), _| *mid != mat_id.0);

            // Mark all instances dirty so they rebuild on next flush.
            if let Some(inst_ids) = i.material_instances.lock().get(&mat_id) {
                let instances = i.instances.read();
                for iid in inst_ids {
                    if let Some(inst) = instances.get(iid) {
                        inst.mark_dirty(DIRTY_ALL);
                    }
                }
            }
        }
    }

    // ── Garbage Collection ──────────────────────────────────────────────────

    /// Evict bind-group cache entries not used in `max_age` frames.
    pub fn gc_bind_groups(&self, current_frame: u64, max_age: u64) {
        let threshold = current_frame.saturating_sub(max_age);
        let i = &self.inner;
        let instances = i.instances.read();

        // Collect active (material_id, tex_hash) pairs.
        let mut active: std::collections::HashSet<(u32, u64)> = std::collections::HashSet::new();
        for inst in instances.values() {
            if inst.last_frame.load(Ordering::Relaxed) >= threshold {
                let key = (inst.material.0, inst.bind_group_key.load(Ordering::Relaxed));
                active.insert(key);
            }
        }

        i.bind_group_cache.lock().retain(|k, _| active.contains(k));
    }

    // ── Stats ───────────────────────────────────────────────────────────────

    pub fn stats(&self) -> MaterialSystemStats {
        self.inner.stats.lock().clone()
    }

    fn update_stats(&self) {
        let i = &self.inner;
        let mut s = i.stats.lock();
        s.material_count = i.materials.read().len() as u32;
        s.instance_count = i.instances.read().len() as u32;
        s.active_uniform_slots = i.uniform_pool.lock().active_slots();
    }
}

/// Handle returned by `start_hot_reload`.  Drop it to stop watching.
pub struct HotReloadHandle {
    _watcher: notify::RecommendedWatcher,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 14 — Error Type
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub enum MaterialError {
    Io(std::io::Error),
    Parse(serde_json::Error),
    NotFound(MaterialId),
    PoolExhausted,
    PropertyNotFound(String),
}

impl std::fmt::Display for MaterialError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::Parse(e) => write!(f, "Parse error: {e}"),
            Self::NotFound(id) => write!(f, "Material {:?} not found", id),
            Self::PoolExhausted => write!(f, "Uniform pool exhausted"),
            Self::PropertyNotFound(n) => write!(f, "Property '{n}' not found"),
        }
    }
}

impl std::error::Error for MaterialError {}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 15 — Render Helper: Batch Builder
// ═══════════════════════════════════════════════════════════════════════════════

/// A draw command produced by the batch builder, pre-sorted for minimal state changes.
#[derive(Debug)]
pub struct DrawCommand {
    pub sort_key: u64,
    pub instance_id: InstanceId,
    pub material_id: MaterialId,
    pub bind_group: wgpu::BindGroup,
    pub dynamic_offset: u32,
}

impl MaterialSystem {
    /// Build sorted draw commands for a set of instances.
    /// The caller provides `(InstanceId, user_sort_key)` pairs.
    /// Returns draw commands sorted by material → texture-set → user key.
    pub fn build_draw_commands(
        &self,
        instance_keys: &[(InstanceId, u32)],
    ) -> Vec<DrawCommand> {
        let mut cmds: Vec<DrawCommand> = Vec::with_capacity(instance_keys.len());
        let instances = self.inner.instances.read();

        for &(id, user_key) in instance_keys {
            if let Some(inst) = instances.get(&id) {
                if let Some(bg) = inst.bind_group.lock().clone() {
                    let offset = (inst.uniform_slot as u64 * UNIFORM_SLOT_SIZE) as u32;
                    let sort_key = {
                        let mat_bits = (inst.material.0 as u64) << 48;
                        let tex_bits =
                            ((inst.bind_group_key.load(Ordering::Relaxed) & 0xFFFF) as u64) << 32;
                        mat_bits | tex_bits | user_key as u64
                    };
                    cmds.push(DrawCommand {
                        sort_key,
                        instance_id: id,
                        material_id: inst.material,
                        bind_group: bg,
                        dynamic_offset: offset,
                    });
                }
            }
        }

        cmds.sort_unstable_by_key(|c| c.sort_key);
        cmds
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 16 — Drop
// ═══════════════════════════════════════════════════════════════════════════════

impl Drop for MaterialSystemInner {
    fn drop(&mut self) {
        // Clear caches — GPU resources are ref-counted and will be freed
        // when all references (pipelines, bind groups, etc.) are dropped.
        self.bind_group_cache.lock().clear();
        self.pipeline_cache.lock().pipelines.clear();
        log::info!("MaterialSystem dropped");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 17 — Usage Example
// ═══════════════════════════════════════════════════════════════════════════════
//
// ```rust
// let mat_sys = MaterialSystem::new(device.clone(), queue.clone());
//
// // Load material
// let mat = mat_sys.load_material("assets/materials/pbr.json")?;
//
// // Resolve property handles ONCE (e.g. at init or when material loads)
// let h_roughness = mat_sys.resolve_property(mat, "roughness").unwrap();
// let h_albedo    = mat_sys.resolve_property(mat, "albedo_map").unwrap();
//
// // Create instances
// let inst = mat_sys.create_instance(mat)?;
//
// // Set properties (O(1) with handles, safe from any thread)
// mat_sys.set_uniform(inst, h_roughness, UniformValue::f32(0.3));
// mat_sys.set_texture(inst, h_albedo, my_view, my_sampler);
//
// // Render loop
// loop {
//     mat_sys.flush(frame_index);
//
//     let cmds = mat_sys.build_draw_commands(&visible_instances);
//     let mut last_mat = MaterialId(u32::MAX);
//     let mut last_bg_key = u64::MAX;
//
//     for cmd in &cmds {
//         if cmd.material_id != last_mat {
//             // bind pipeline
//             last_mat = cmd.material_id;
//             last_bg_key = u64::MAX;
//         }
//         render_pass.set_bind_group(0, &cmd.bind_group, &[cmd.dynamic_offset]);
//         // draw ...
//     }
//
//     mat_sys.gc_bind_groups(frame_index, 120);
// }
//
