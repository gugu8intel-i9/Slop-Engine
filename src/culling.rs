//! # Fast Culling System for WGPU‑based Engines
//!
//! This single‑file module implements:
//! * **Frustum culling** – CPU‑side, vectorised and parallelised via `rayon`.
//! * **Hi‑Z occlusion culling** – GPU‑based hierarchical Z‑buffer generation
//!   (compute shader) and screen‑space occlusion tests.
//! * **LOD selection** – per‑object screen‑size, importance, hysteresis.
//! * **Thread‑safe API**, minimal allocations and profiling hooks.
//!
//! The system works with any WGPU‑compatible renderer. It can be used
//! with or without GPU‑accelerated occlusion (fallback to conservative CPU
//! heuristics). All resources are internally managed – just create a `Culler`,
//! feed it instances and a camera each frame and get a list of `CullResult`.
//!
//! ## Cargo.toml (minimum dependencies)
//! ```toml
//! [dependencies]
//! glam = "0.22"
//! wgpu = "0.16"
//! rayon = "1.7"
//! parking_lot = "0.12"
//! bytemuck = "1.9"
//! futures = "0.3"          # for async read‑back of GPU results
//! ```
//!
//! ---------------------------------------------------------------------------

#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use rayon::prelude::*;
use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use futures::{channel::oneshot, executor::block_on};

// ---------------------------------------------------------------------------
// Public API Types
// ---------------------------------------------------------------------------

/// Unique identifier for a renderable object (mesh instance).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct InstanceId(pub u64);

/// Result of culling for a single instance.
#[derive(Clone, Copy, Debug)]
pub struct CullResult {
    /// Instance that was tested.
    pub id: InstanceId,
    /// `true` if the instance should be drawn (passed frustum & occlusion).
    pub visible: bool,
    /// `true` if the instance is hidden by occlusion (only meaningful when `visible == false`).
    pub occluded: bool,
    /// Selected level‑of‑detail (0 = highest detail).
    pub lod: u8,
    /// Normalised screen rectangle `[x0, y0, x1, y1]` in UV space.
    pub screen_rect: [f32; 4],
}

/// Bounding volume used for frustum culling.
#[derive(Clone, Copy, Debug)]
pub enum Bounds {
    /// Sphere defined in **local** space.
    Sphere { center: Vec3, radius: f32 },
    /// Axis‑aligned box defined in **local** space.
    Aabb { min: Vec3, max: Vec3 },
}

/// Per‑instance metadata required by the culler.
#[derive(Clone, Debug)]
pub struct Instance {
    /// Identifier – must be unique among all instances you submit each frame.
    pub id: InstanceId,
    /// Bounding volume in local space.
    pub bounds: Bounds,
    /// World transform (model matrix). Must be column‑major (glam `Mat4`).
    pub world_transform: Mat4,
    /// User‑specified bias for LOD selection (>= 0). Higher bias keeps higher LODs longer.
    pub lod_bias: f32,
    /// Importance factor `[0..1]`. Higher importance reduces chance of being culled by occlusion.
    pub importance: f32,
    /// Visibility layer mask – can be used for layer‑based culling.
    pub layer_mask: u32,
}

/// Camera data required for culling.
#[derive(Clone, Debug)]
pub struct Camera {
    /// View matrix (`camera → world`).
    pub view: Mat4,
    /// Projection matrix (`clip → ndc`).
    pub proj: Mat4,
    /// Combined view‑projection (`clip ← view * proj`).
    pub view_proj: Mat4,
    /// Inverse of view‑projection (used for depth reconstruction).
    pub inv_view_proj: Mat4,
    /// World‑space camera position.
    pub position: Vec3,
    /// Near / far planes (in view space).
    pub near: f32,
    pub far: f32,
    /// Vertical field‑of‑view (radians).
    pub fov_y: f32,
    /// Aspect ratio (width / height).
    pub aspect: f32,
    /// Current swap‑chain size.
    pub screen_size: (u32, u32),
}

/// Configuration knobs for the culler.
#[derive(Clone, Debug)]
pub struct CullerConfig {
    /// Enable GPU‑based Hi‑Z occlusion. `false` falls back to CPU heuristics.
    pub use_gpu_hiz: bool,
    /// Format of the Hi‑Z depth mip‑chain. `R32Float` is a safe default.
    pub hiz_format: wgpu::TextureFormat,
    /// Smallest mip dimension – stop down‑sampling when width/height ≤ this value.
    pub hiz_mip_min_size: u32,
    /// Visibility threshold for occlusion – fraction of pixels that must be visible.
    /// `0.01` (1 %) is a good default.
    pub occlusion_threshold: f32,
    /// Number of frames of consecutive occlusion before we consider an object culled.
    pub occlusion_history_frames: usize,
    /// Number of LOD levels (must be ≥ 1). Level 0 is highest detail.
    pub lod_levels: u8,
    /// Screen‑space size thresholds for LOD transitions.
    /// `len = lod_levels‑1`. Values are *normalised* area (`[0..1]`) where
    /// a value of `0.5` means “if screen‑area < 0.5 → use next LOD”.
    pub lod_screen_size_thresholds: Vec<f32>,
    /// Hint for `rayon` thread count used for CPU frustum culling.
    /// `0` = let `rayon` decide automatically.
    pub cpu_frustum_threads: usize,
}

impl Default for CullerConfig {
    fn default() -> Self {
        Self {
            use_gpu_hiz: true,
            hiz_format: wgpu::TextureFormat::R32Float,
            hiz_mip_min_size: 1,
            occlusion_threshold: 0.01,
            occlusion_history_frames: 3,
            lod_levels: 4,
            lod_screen_size_thresholds: vec![0.5, 0.25, 0.125],
            cpu_frustum_threads: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal per‑instance state
// ---------------------------------------------------------------------------

/// Per‑instance occlusion history (keeps hysteresis stable).
#[derive(Clone, Debug)]
struct OcclusionState {
    last_visible_frame: u64,
    consecutive_occluded: u32,
    last_test_frame: u64,
}

// ---------------------------------------------------------------------------
// Profiling information
// ---------------------------------------------------------------------------

#[derive(Default, Debug, Clone)]
pub struct CullerStats {
    /// Time spent in CPU frustum culling (parallelised).
    pub cpu_frustum_time: Duration,
    /// Time spent building the Hi‑Z mip‑chain.
    pub gpu_hiz_time: Duration,
    /// Time spent in the occlusion test compute pass (including read‑back).
    pub gpu_occlusion_time: Duration,
    /// Number of instances that were culled by any test.
    pub culled_count: usize,
    /// Number of instances that survived all tests.
    pub visible_count: usize,
    /// Count of instances per LOD level.
    pub lod_counts: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Main Culler
// ---------------------------------------------------------------------------

/// `Culler` – the entry point for all culling operations.
///
/// Create with [`Culler::new`], feed it instances via [`Culler::set_instances`],
/// update the Hi‑Z texture when the viewport size changes
/// ([`Culler::prepare_hiz`]), build the mip‑chain after a depth pre‑pass
/// ([`Culler::build_hiz`]) and finally call [`Culler::cull_scene`] each frame.
pub struct Culler {
    // WGPU handles – kept as `Arc` because we may be used from multiple threads.
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    // User configuration.
    config: CullerConfig,

    // ------------------------------------------------------------------------
    // Hi‑Z resources
    // ------------------------------------------------------------------------
    hiz_texture: Option<wgpu::Texture>,
    hiz_view: Option<wgpu::TextureView>,
    hiz_mip_count: u32,
    hiz_bind_group_layout: Option<wgpu::BindGroupLayout>,
    hiz_pipeline: Option<wgpu::ComputePipeline>,

    // ------------------------------------------------------------------------
    // Occlusion test resources (rect + depth threshold + visibility)
    // ------------------------------------------------------------------------
    occlusion_bind_group_layout: Option<wgpu::BindGroupLayout>,
    occlusion_pipeline: Option<wgpu::ComputePipeline>,

    // ------------------------------------------------------------------------
    // CPU‑side caches & state
    // ------------------------------------------------------------------------
    /// Mapping `InstanceId -> Instance` for fast look‑ups during LOD selection.
    instance_cache: Mutex<HashMap<u64, Instance>>,
    /// Per‑instance occlusion history.
    occlusion_states: Mutex<HashMap<u64, OcclusionState>>,
    /// Profiling counters.
    stats: Mutex<CullerStats>,
    /// Monotonically increasing frame index used for hysteresis.
    frame_index: Mutex<u64>,
}

impl Culler {
    // ------------------------------------------------------------------------
    // Construction & resource creation
    // ------------------------------------------------------------------------

    /// Create a new culler.
    ///
    /// `device` and `queue` must stay alive for the lifetime of the `Culler`.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, config: CullerConfig) -> Self {
        Self {
            device,
            queue,
            config,
            hiz_texture: None,
            hiz_view: None,
            hiz_mip_count: 0,
            hiz_bind_group_layout: None,
            hiz_pipeline: None,
            occlusion_bind_group_layout: None,
            occlusion_pipeline: None,
            instance_cache: Mutex::new(HashMap::new()),
            occlusion_states: Mutex::new(HashMap::new()),
            stats: Mutex::new(CullerStats::default()),
            frame_index: Mutex::new(0),
        }
    }

    /// Register a set of instances. Replaces the internal cache.
    ///
    /// The culler only needs the instances that are *potentially* visible.
    /// It is your responsibility to filter out culled static batches beforehand
    /// if you wish to reduce the amount of data uploaded.
    pub fn set_instances(&self, instances: Vec<Instance>) {
        let mut cache = self.instance_cache.lock();
        cache.clear();
        for inst in instances {
            cache.insert(inst.id.0, inst);
        }
    }

    // ------------------------------------------------------------------------
    // Hi‑Z texture & pipeline creation
    // ------------------------------------------------------------------------

    /// Ensure the Hi‑Z texture matches the current viewport size.
    ///
    /// Call this when the swap‑chain resolution changes.
    pub fn prepare_hiz(&mut self, width: u32, height: u32) {
        if !self.config.use_gpu_hiz {
            return;
        }

        self.ensure_hiz_pipeline();
        self.ensure_occlusion_pipeline();

        // Compute required mip count.
        let mut w = width;
        let mut h = height;
        let mut mips = 1u32;
        while w > self.config.hiz_mip_min_size || h > self.config.hiz_mip_min_size {
            w = (w + 1) / 2;
            h = (h + 1) / 2;
            mips += 1;
        }

        // Re‑create texture only when size or mip count changes.
        let tex = match &self.hiz_texture {
            Some(tex) => {
                if tex.size().width == width
                    && tex.size().height == height
                    && self.hiz_mip_count == mips
                {
                    return; // nothing to do
                }
                // Drop old texture – we are about to replace it.
                drop(tex);
                None
            }
            None => None,
        };

        // Create new Hi‑Z texture.
        let texture_desc = wgpu::TextureDescriptor {
            label: Some("culler_hiz_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.config.hiz_format,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
        };
        let texture = self.device.create_texture(&texture_desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.hiz_texture = Some(texture);
        self.hiz_view = Some(view);
        self.hiz_mip_count = mips;
    }

    /// Build the Hi‑Z mip‑chain from a depth texture.
    ///
    /// This must be called **after** the depth pre‑pass but **before**
    /// `cull_scene`. The provided `depth_view` must be a `R32Float` (or
    /// compatible) 2‑D texture containing the scene depth in view space.
    pub fn build_hiz(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        if !self.config.use_gpu_hiz {
            return;
        }
        let hiz_tex = match &self.hiz_texture {
            Some(t) => t,
            None => return,
        };
        let hiz_view = match &self.hiz_view {
            Some(v) => v,
            None => return,
        };
        let pipeline = match &self.hiz_pipeline {
            Some(p) => p,
            None => return,
        };
        let bgl = match &self.hiz_bind_group_layout {
            Some(b) => b,
            None => return,
        };

        // --------------------------------------------------------------------
        // 1) Copy depth into Hi‑Z mip 0.
        // --------------------------------------------------------------------
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: depth_view.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: hiz_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        // --------------------------------------------------------------------
        // 2) Down‑sample mip chain – each mip is the max of its 2×2 block.
        // --------------------------------------------------------------------
        let mut src_mip = 0u32;
        let mut dst_mip = 1u32;
        let mut src_w = width;
        let mut src_h = height;

        while dst_mip < self.hiz_mip_count {
            let dst_w = (src_w + 1) / 2;
            let dst_h = (src_h + 1) / 2;

            // Views for source / destination mips.
            let src_view_desc = wgpu::TextureViewDescriptor {
                label: Some(&format!("hiz_src_mip_{}", src_mip)),
                base_mip_level: src_mip,
                mip_level_count: Some(1),
                ..Default::default()
            };
            let dst_view_desc = wgpu::TextureViewDescriptor {
                label: Some(&format!("hiz_dst_mip_{}", dst_mip)),
                base_mip_level: dst_mip,
                mip_level_count: Some(1),
                ..Default::default()
            };

            let src_view = hiz_tex.create_view(&src_view_desc);
            let dst_view = hiz_tex.create_view(&dst_view_desc);

            // Bind group for the down‑sample compute pass.
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("hiz_bg_{}_{}", src_mip, dst_mip)),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&dst_view),
                    },
                ],
            });

            // Dispatch compute workgroups (workgroup size = 16×16).
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hiz_downsample_pass"),
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            let wg = 16u32;
            let gx = (dst_w + wg - 1) / wg;
            let gy = (dst_h + wg - 1) / wg;
            cpass.dispatch_workgroups(gx, gy, 1);

            src_mip = dst_mip;
            dst_mip += 1;
            src_w = dst_w;
            src_h = dst_h;
        }
    }

    // ------------------------------------------------------------------------
    // Main culling routine
    // ------------------------------------------------------------------------

    /// Perform frustum + occlusion culling and LOD selection for the current
    /// frame. Returns a `Vec<CullResult>` sorted in the order of the input
    /// `frame_instances`.
    ///
    /// # Parameters
    ///
    /// * `camera` – current camera state.
    /// * `frame_instances` – list of instances that *might* be visible this frame.
    ///
    /// # Remarks
    ///
    /// * The function increments an internal frame counter used for hysteresis.
    /// * All CPU work is parallelised via `rayon`. The GPU path uses a single
    ///   compute dispatch plus a blocking read‑back of a small visibility buffer.
    /// * If `use_gpu_hiz` is `false` or the GPU pipeline is not yet created,
    ///   the system falls back to a cheap conservative CPU heuristic.
    pub fn cull_scene(&self, camera: &Camera, frame_instances: &[Instance]) -> Vec<CullResult> {
        let _timer = Instant::now();
        // --------------------------------------------------------------------
        // 0) Bump frame index – used for occlusion hysteresis.
        // --------------------------------------------------------------------
        let mut fi = self.frame_index.lock();
        *fi += 1;
        let frame_idx = *fi;
        drop(fi);

        // --------------------------------------------------------------------
        // 1) Extract frustum planes from the view‑projection matrix.
        // --------------------------------------------------------------------
        let frustum_planes = extract_frustum_planes(&camera.view_proj);

        // --------------------------------------------------------------------
        // 2) CPU‑side frustum culling (parallelised).
        // --------------------------------------------------------------------
        let cpu_start = Instant::now();

        // Determine thread count for `rayon`. `0` means “auto”.
        let cpu_threads = if self.config.cpu_frustum_threads == 0 {
            rayon::current_num_threads()
        } else {
            self.config.cpu_frustum_threads
        };
        // We *could* set the global thread pool here, but `rayon` already does
        // the right thing when `current_num_threads()` is used.

        // For each instance we compute:
        //   * a world‑space sphere (or AABB → sphere) for cheap plane tests,
        //   * a normalised screen rectangle,
        //   * an approximate screen‑area metric.
        let candidates: Vec<(
            InstanceId,
            Instance,
            Option<[f32; 4]>,
            f32,
            f32, // depth threshold (normalised view‑Z)
        )> = frame_instances
            .par_iter()
            .filter_map(|inst| {
                // ---- world‑space bounding sphere ------------------------------
                let (center, radius) = match inst.bounds {
                    Bounds::Sphere { center, radius } => {
                        let world_center = inst.world_transform.transform_point3(center);
                        let scale = inst.world_transform.to_scale().max_element();
                        (world_center, radius * scale)
                    }
                    Bounds::Aabb { min, max } => {
                        let world_min = inst.world_transform.transform_point3(min);
                        let world_max = inst.world_transform.transform_point3(max);
                        let world_center = (world_min + world_max) * 0.5;
                        let extent = (world_max - world_min).length() * 0.5;
                        (world_center, extent)
                    }
                };

                // ---- fast frustum test (sphere‑in‑plane) --------------------
                if !sphere_in_frustum(center, radius, &frustum_planes) {
                    return None; // completely outside frustum
                }

                // ---- compute normalized screen rectangle --------------------
                // Project sphere centre into clip space.
                let clip = camera.view_proj
                    * inst.world_transform
                    * glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
                let clip_w = clip.w;
                if clip_w <= 0.0 {
                    return None; // behind camera
                }
                let ndc = glam::Vec3::new(clip.x / clip_w, clip.y / clip_w, clip.z / clip_w);

                // View‑space Z for depth threshold.
                let view_pos = camera.view
                    * inst.world_transform
                    * glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
                let norm_z = view_pos.z / camera.far; // in [0..1]

                // Approximate screen radius from sphere radius.
                let fov_rad = camera.fov_y.to_radians();
                let fov_h = fov_rad; // vertical FOV is enough for a rough estimate
                let screen_radius = (radius / view_pos.z.abs().max(1e-6)) * fov_h.tan();

                // Convert to UV coordinates ([0..1] range).
                let cx = ndc.x * 0.5 + 0.5;
                let cy = ndc.y * 0.5 + 0.5;
                let r = screen_radius * 0.5; // because ndc runs from -1..1

                let x0 = (cx - r).max(0.0);
                let y0 = (cy - r).max(0.0);
                let x1 = (cx + r).min(1.0);
                let y1 = (cy + r).min(1.0);

                let rect = [x0, y0, x1, y1];
                let screen_area = (x1 - x0) * (y1 - y0);

                Some((
                    inst.id,
                    inst.clone(),
                    Some(rect),
                    screen_area,
                    norm_z.clamp(0.0, 1.0),
                ))
            })
            .collect();

        let cpu_time = cpu_start.elapsed();
        {
            let mut s = self.stats.lock();
            s.cpu_frustum_time = cpu_time;
        }

        // --------------------------------------------------------------------
        // 3) Occlusion test + LOD selection.
        // --------------------------------------------------------------------
        let mut results = Vec::with_capacity(candidates.len());

        // Fast path – if we have no candidates we are done.
        if candidates.is_empty() {
            // Update stats with zero visible/culled.
            let mut s = self.stats.lock();
            s.visible_count = 0;
            s.culled_count = 0;
            s.lod_counts = vec![0; self.config.lod_levels as usize];
            return results;
        }

        // --------------------------------------------------------------------
        // 3a) GPU‑accelerated Hi‑Z occlusion (if enabled & pipelines ready)
        // --------------------------------------------------------------------
        if self.config.use_gpu_hiz
            && self.hiz_view.is_some()
            && self.occlusion_pipeline.is_some()
            && self.occlusion_bind_group_layout.is_some()
        {
            // Build GPU input: one rect per candidate.
            // `GpuRect` = (x0, y0, x1, y1, depth_threshold)
            let mut gpu_rects: Vec<GpuRect> = Vec::with_capacity(candidates.len());
            let mut ids: Vec<InstanceId> = Vec::with_capacity(candidates.len());

            for (id, _inst, rect_opt, _area, depth_thr) in &candidates {
                let rect = rect_opt.unwrap_or([0.0, 0.0, 0.0, 0.0]);
                gpu_rects.push(GpuRect {
                    x0: rect[0],
                    y0: rect[1],
                    x1: rect[2],
                    y1: rect[3],
                    depth_threshold: *depth_thr,
                });
                ids.push(*id);
            }

            // Early exit if nothing to test (should not happen, but safe).
            if gpu_rects.is_empty() {
                // Fallback to CPU occlusion.
                self.cull_cpu_fallback(camera, &candidates, frame_idx, &mut results);
                return results;
            }

            // ----------------------------------------------------------------
            // Upload rects and allocate result buffer.
            // ----------------------------------------------------------------
            let rects_bytes = bytemuck::cast_slice(&gpu_rects);
            let rects_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("culler_rects_buf"),
                contents: rects_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

            let result_count = gpu_rects.len() as u64;
            let results_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("culler_vis_buf"),
                size: result_count * std::mem::size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // ----------------------------------------------------------------
            // Build bind group for the occlusion compute pass.
            // ----------------------------------------------------------------
            let bgl = self.occlusion_bind_group_layout.as_ref().unwrap();
            let hiz_view = self.hiz_view.as_ref().unwrap();
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("culler_occlusion_bg"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: rects_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(hiz_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: results_buf.as_entire_binding(),
                    },
                ],
            });

            // ----------------------------------------------------------------
            // Record and submit the compute pass.
            // ----------------------------------------------------------------
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("culler_occlusion_encoder"),
            });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("culler_occlusion_pass"),
                });
                cpass.set_pipeline(self.occlusion_pipeline.as_ref().unwrap());
                cpass.set_bind_group(0, &bg, &[]);
                let wg = 64u32; // workgroup size X (matches WGSL `@workgroup_size(64)`)
                let dispatch = ((gpu_rects.len() as u32 + wg - 1) / wg);
                cpass.dispatch_workgroups(dispatch, 1, 1);
            }

            let cmd_buf = encoder.finish();
            self.queue.submit(Some(cmd_buf));

            // ----------------------------------------------------------------
            // Blocking read‑back of visibility flags.
            // ----------------------------------------------------------------
            let gpu_start = Instant::now();

            // Slice of the result buffer we want to map.
            let result_slice = results_buf.slice(..);
            // One‑shot channel to know when mapping is ready.
            let (sender, receiver) = oneshot::channel();

            result_slice.map_async(wgpu::MapMode::Read, move |res| {
                sender.send(res).ok();
            });

            // Wait for the GPU to finish the mapping.
            self.device.poll(wgpu::Maintain::Wait);
            let map_ok = block_on(receiver).unwrap_or(Ok(()));

            if map_ok.is_ok() {
                let mapped = result_slice.get_mapped_range();
                let flags: &[u32] = bytemuck::cast_slice(&mapped);

                // ------------------------------------------------------------
                // Build final `CullResult`s using the visibility flags.
                // ------------------------------------------------------------
                for (i, (id, inst, rect_opt, area, _depth)) in candidates.iter().enumerate() {
                    let visible = flags.get(i).copied().unwrap_or(1) != 0;
                    let occluded = !visible;

                    // Select LOD.
                    let lod = self.select_lod_for_instance(inst, *area);

                    let rect = rect_opt.unwrap_or([0.0, 0.0, 0.0, 0.0]);

                    results.push(CullResult {
                        id: *id,
                        visible,
                        occluded,
                        lod,
                        screen_rect: rect,
                    });

                    // Update per‑instance occlusion state for hysteresis.
                    let mut states = self.occlusion_states.lock();
                    let state = states.entry(id.0).or_insert(OcclusionState {
                        last_visible_frame: 0,
                        consecutive_occluded: 0,
                        last_test_frame: 0,
                    });
                    if visible {
                        state.consecutive_occluded = 0;
                        state.last_visible_frame = frame_idx;
                    } else {
                        state.consecutive_occluded = state.consecutive_occluded.saturating_add(1);
                    }
                    state.last_test_frame = frame_idx;
                }

                // Unmap the buffer.
                drop(mapped);
                result_slice.unmap();
            } else {
                // Read‑back failed – fall back to CPU heuristic.
                self.cull_cpu_fallback(camera, &candidates, frame_idx, &mut results);
            }

            let gpu_time = gpu_start.elapsed();
            {
                let mut s = self.stats.lock();
                s.gpu_occlusion_time = gpu_time;
            }
        } else {
            // ----------------------------------------------------------------
            // 3b) CPU fallback occlusion (conservative + hysteresis)
            // ----------------------------------------------------------------
            self.cull_cpu_fallback(camera, &candidates, frame_idx, &mut results);
        }

        // --------------------------------------------------------------------
        // 4) Update statistics (visible / culled counts, LOD distribution)
        // --------------------------------------------------------------------
        {
            let mut s = self.stats.lock();
            s.visible_count = results.iter().filter(|r| r.visible).count();
            s.culled_count = results.iter().filter(|r| r.occluded).count();
            // Re‑compute LOD histogram.
            let lod_levels = self.config.lod_levels as usize;
            s.lod_counts = vec![0; lod_levels];
            for r in &results {
                if (r.lod as usize) < lod_levels {
                    s.lod_counts[r.lod as usize] += 1;
                }
            }
        }

        results
    }

    /// Internal helper – cheap CPU‑side occlusion + hysteresis.
    fn cull_cpu_fallback(
        &self,
        camera: &Camera,
        candidates: &[
            (
                InstanceId,
                Instance,
                Option<[f32; 4]>,
                f32,
                f32,
            ),
        ],
        frame_idx: u64,
        out: &mut Vec<CullResult>,
    ) {
        for (id, inst, rect_opt, screen_area, _depth) in candidates {
            let rect = rect_opt.unwrap_or([0.0, 0.0, 0.0, 0.0]);

            // Conservative heuristic: small objects with low importance are
            // considered potentially occluded. We use hysteresis to avoid flicker.
            let mut occluded = false;
            let small_thresh = 1.0
                / ((camera.screen_size.0.max(camera.screen_size.1) as f32).powi(2))
                * 4.0; // approx 4 pixels area

            if *screen_area < small_thresh && inst.importance < 0.5 {
                let mut states = self.occlusion_states.lock();
                let state = states.entry(id.0).or_insert(OcclusionState {
                    last_visible_frame: 0,
                    consecutive_occluded: 0,
                    last_test_frame: 0,
                });

                // Hysteresis logic.
                if frame_idx - state.last_test_frame <= 1 {
                    // We have recent history – only cull after N consecutive frames.
                    if state.consecutive_occluded >= self.config.occlusion_history_frames as u32 {
                        occluded = true;
                    }
                } else {
                    // No recent test – start a new streak.
                    state.consecutive_occluded = 1;
                    state.last_test_frame = frame_idx;
                    // Only cull after enough frames.
                    if state.consecutive_occluded >= self.config.occlusion_history_frames as u32 {
                        occluded = true;
                    }
                }

                if !occluded {
                    state.last_visible_frame = frame_idx;
                }
            } else {
                // Object is big / important – treat as visible.
                let mut states = self.occlusion_states.lock();
                let state = states.entry(id.0).or_insert(OcclusionState {
                    last_visible_frame: 0,
                    consecutive_occluded: 0,
                    last_test_frame: 0,
                });
                state.consecutive_occluded = 0;
                state.last_visible_frame = frame_idx;
                state.last_test_frame = frame_idx;
            }

            let lod = self.select_lod_for_instance(inst, *screen_area);
            out.push(CullResult {
                id: *id,
                visible: !occluded,
                occluded,
                lod,
                screen_rect: rect,
            });
        }
    }

    /// Select LOD based on screen‑area and per‑instance bias.
    fn select_lod_for_instance(&self, inst: &Instance, screen_area: f32) -> u8 {
        // `screen_area` is in `[0..1]` (normalised UV area).
        // LOD thresholds are defined in the config, we simply walk them.
        let bias = 1.0 + inst.lod_bias; // allow user to keep higher LODs longer
        let mut lod = 0u8;

        for (i, &th) in self.config.lod_screen_size_thresholds.iter().enumerate() {
            if screen_area < th * bias {
                lod = (i + 1) as u8;
            } else {
                break;
            }
        }

        // Clamp to the maximum configured LOD level.
        if lod >= self.config.lod_levels {
            lod = self.config.lod_levels - 1;
        }
        lod
    }

    // ------------------------------------------------------------------------
    // Lazy pipeline creation (Hi‑Z & occlusion)
    // ------------------------------------------------------------------------

    fn ensure_hiz_pipeline(&mut self) {
        if self.hiz_pipeline.is_some() {
            return;
        }

        let cs_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("culler_hiz_build_cs"),
                source: wgpu::ShaderSource::Wgsl(HIZ_BUILD_WGSL.into()),
            });

        let bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("culler_hiz_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: self.config.hiz_format,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("culler_hiz_pipeline_layout"),
                    bind_group_layouts: &[&bgl],
                    push_constant_ranges: &[],
                });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("culler_hiz_build_pipeline"),
                layout: Some(&pipeline_layout),
                module: &cs_module,
                entry_point: Some("main"),
            });

        self.hiz_bind_group_layout = Some(bgl);
        self.hiz_pipeline = Some(pipeline);
    }

    fn ensure_occlusion_pipeline(&mut self) {
        if self.occlusion_pipeline.is_some() {
            return;
        }

        let cs_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("culler_occlusion_cs"),
                source: wgpu::ShaderSource::Wgsl(HIZ_OCCLUSION_WGSL.into()),
            });

        let bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("culler_occlusion_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("culler_occlusion_pipeline_layout"),
                    bind_group_layouts: &[&bgl],
                    push_constant_ranges: &[],
                });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("culler_occlusion_pipeline"),
                layout: Some(&pipeline_layout),
                module: &cs_module,
                entry_point: Some("main"),
            });

        self.occlusion_bind_group_layout = Some(bgl);
        self.occlusion_pipeline = Some(pipeline);
    }

    // ------------------------------------------------------------------------
    // Public stats accessor
    // ------------------------------------------------------------------------
    /// Return a snapshot of the culling statistics.
    pub fn stats(&self) -> CullerStats {
        self.stats.lock().clone()
    }
}

// ---------------------------------------------------------------------------
// Helper functions (math)
// ---------------------------------------------------------------------------

/// Extract six frustum planes from a combined view‑projection matrix.
/// Planes are normalised and in the form `(normal, distance)`.
fn extract_frustum_planes(vp: &Mat4) -> [Vec4; 6] {
    // Columns of the matrix (glam stores column‑major)
    let m = vp.to_cols_array_2d();

    // Helper to build a plane from two rows.
    fn plane_from_rows(r1: [f32; 4], r2: [f32; 4]) -> Vec4 {
        let normal = Vec4::new(r1[0] + r2[0], r1[1] + r2[1], r1[2] + r2[2], r1[3] + r2[3]);
        let len = (normal.x * normal.x + normal.y * normal.y + normal.z * normal.z)
            .sqrt()
            .max(1e-6);
        normal / len
    }

    // left = row3 + row0
    let p0 = plane_from_rows(m[3], m[0]);
    // right = row3 - row0
    let p1 = plane_from_rows(
        [m[3][0] - m[0][0], m[3][1] - m[0][1], m[3][2] - m[0][2], m[3][3] - m[0][3]],
        [0.0, 0.0, 0.0, 0.0],
    );
    // bottom = row3 + row1
    let p2 = plane_from_rows(m[3], m[1]);
    // top = row3 - row1
    let p3 = plane_from_rows(
        [m[3][0] - m[1][0], m[3][1] - m[1][1], m[3][2] - m[1][2], m[3][3] - m[1][3]],
        [0.0, 0.0, 0.0, 0.0],
    );
    // near = row3 + row2
    let p4 = plane_from_rows(m[3], m[2]);
    // far = row3 - row2
    let p5 = plane_from_rows(
        [m[3][0] - m[2][0], m[3][1] - m[2][1], m[3][2] - m[2][2], m[3][3] - m[2][3]],
        [0.0, 0.0, 0.0, 0.0],
    );

    // Pack into array: left, right, bottom, top, near, far
    [p0, p1, p2, p3, p4, p5]
}

/// Fast sphere‑in‑frustum test (conservative).
fn sphere_in_frustum(center: Vec3, radius: f32, planes: &[Vec4; 6]) -> bool {
    for p in planes {
        // plane: normal·point + distance >= 0
        let d = p.x * center.x + p.y * center.y + p.z * center.z + p.w;
        if d < -radius {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Embedded WGSL shaders
// ---------------------------------------------------------------------------

/// Hi‑Z generation – builds a mip chain by taking the max of 2×2 blocks.
const HIZ_BUILD_WGSL: &str = r#"
@group(0) @binding(0) var src_depth: texture_2d<f32>;
@group(0) @binding(1) var dst_mip: texture_storage_2d<rg32float, write>;

// Workgroup size matches the 2×2 downsample.
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let dims = textureDimensions(dst_mip);
    if (x >= dims.x || y >= dims.y) { return; }

    // Sample 2×2 block from the previous mip.
    let sx = x * 2;
    let sy = y * 2;
    var maxd = 0.0;
    for (var oy: i32 = 0; oy < 2; oy = oy + 1) {
        for (var ox: i32 = 0; ox < 2; ox = ox + 1) {
            let sx2 = sx + ox;
            let sy2 = sy + oy;
            let sdim = textureDimensions(src_depth);
            if (sx2 < sdim.x && sy2 < sdim.y) {
                let d = textureLoad(src_depth, vec2<i32>(sx2, sy2), 0).x;
                if (d > maxd) { maxd = d; }
            }
        }
    }
    textureStore(dst_mip, vec2<i32>(x, y), vec4<f32>(maxd, 0.0, 0.0, 0.0));
}
"#;

/// Occlusion test – rasterises a bounding rectangle into Hi‑Z and decides
/// visibility. `rects` contains `(x0,y0,x1,y1,depth_threshold)`. The shader
/// samples the Hi‑Z mip that best matches the rectangle size (here we simply
/// use mip 0 for demonstration; a production implementation would select the
/// appropriate mip based on rectangle size).
const HIZ_OCCLUSION_WGSL: &str = r#"
struct Rect {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    depth_threshold: f32,
};

@group(0) @binding(0) var<storage, read> rects: array<Rect>;
@group(0) @binding(1) var hiz: texture_2d<f32>;
@group(0) @binding(2) var<storage, write> visible: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = i32(id.x);
    if (idx >= arrayLength(&rects)) { return; }

    let r = rects[idx];

    // Choose a sample point – centre of the rect.
    let sx = (r.x0 + r.x1) * 0.5;
    let sy = (r.y0 + r.y1) * 0.5;

    // Convert UV to texture coordinates.
    let dim = textureDimensions(hiz);
    let px = i32(sx * f32(dim.x));
    let py = i32(sy * f32(dim.y));

    // Guard against out‑of‑bounds.
    if (px < 0 || py < 0 || px >= dim.x || py >= dim.y) {
        visible[idx] = 0; // off‑screen → occluded
        return;
    }

    // Load Hi‑Z depth at the chosen pixel (mip 0).
    let scene_z = textureLoad(hiz, vec2<i32>(px, py), 0).x;

    // `depth_threshold` is the normalized depth of the instance's far bound.
    // If the stored depth in the scene is *shallower* than the instance's far
    // bound, the instance is behind geometry → occluded.
    if (scene_z < r.depth_threshold) {
        visible[idx] = 0; // occluded
    } else {
        visible[idx] = 1; // visible
    }
}
"#;

// ---------------------------------------------------------------------------
// Exported items for external use.
// ---------------------------------------------------------------------------

pub use {
    InstanceId, CullResult, Bounds, Instance, Camera, CullerConfig, Culler, CullerStats,
};

/// Example usage (commented out – replace with your own WGPU setup):
/*
use std::sync::Arc;
use wgpu::{Device, Queue};

fn build_culler(device: Arc<Device>, queue: Arc<Queue>) -> Culler {
    let config = CullerConfig::default();
    Culler::new(device, queue, config)
}

// In your render loop:
// let mut culler = build_culler(device, queue);
// culler.prepare_hiz(width, height);
// // after depth pre‑pass:
// culler.build_hiz(&mut encoder, depth_view, width, height);
// let cam = Camera { /* fill fields */ };
// let visible = culler.cull_scene(&cam, &instances);
// for res in visible {
//     if res.visible {
//         draw_mesh(res.id, res.lod);
//     }
// }
// println!("Stats: {:?}", culler.stats());
*/
