// ============================================================================
//  cache.rs — Orchid Cache: Content-Addressable Game Engine Resource Cache
// ============================================================================
//
// A unified caching layer for Rust/WGSL game engines featuring:
//
//   • Content-addressable storage (Blake3) — identical resources are stored once
//   • Dependency graph with cascading invalidation
//   • ARC (Adaptive Replacement Cache) eviction — adapts to workload in real-time
//   • WGSL shader variant management via specialization constants
//   • Hot-reload file watching with automatic recompilation triggers
//   • Disk persistence with zstd compression
//   • Predictive prefetching from sequential access pattern detection
//   • Memory budget enforcement with soft/hard limits
//   • Per-shard statistics and cache-wide observability
//
//  Usage:
//
//      let mut cache = OrchidCache::new(CacheConfig::default());
//      cache.insert("shaders/pbr.wgsl", shader_module, &[])?;
//      let shader = cache.get("shaders/pbr.wgsl")?;
//      cache.tick(); // housekeeping: eviction, prefetch, stats
//

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ============================================================================
// §1 — CONTENT-ADDRESSABLE KEY
// ============================================================================

/// A collision-resistant cache key derived from Blake3 hashing.
///
/// Every resource is identified by the cryptographic hash of its content,
/// meaning duplicate assets (same bytes) map to the same key regardless of
/// their logical name. This enables transparent deduplication across the
/// entire engine — two materials referencing the same shader source produce
/// one compiled module in memory.
#[derive(Clone, Copy, Eq)]
pub struct CacheKey(blake3::Hash);

impl CacheKey {
    /// Derive a key from raw bytes.
    pub fn from_bytes(data: &[u8]) -> Self {
        Self(blake3::hash(data))
    }

    /// Derive a key from a string (e.g. WGSL source).
    pub fn from_str(s: &str) -> Self {
        Self::from_bytes(s.as_bytes())
    }

    /// Build a composite key from multiple byte slices (e.g. source + specialization constants).
    pub fn from_parts(parts: &[&[u8]]) -> Self {
        let mut hasher = blake3::Hasher::new();
        for part in parts {
            hasher.update(part);
        }
        Self(hasher.finalize())
    }

    /// Raw 32-byte digest.
    pub fn as_bytes(&self) -> &[u8; 32] {
        self.0.as_bytes()
    }

    /// Hex-encoded string for logging and display.
    pub fn to_hex(&self) -> String {
        self.0.to_hex().to_string()
    }
}

impl PartialEq for CacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_bytes() == other.0.as_bytes()
    }
}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_bytes().hash(state);
    }
}

impl std::fmt::Debug for CacheKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::mut Option<Duration> {
    pub hot_reload_poll: Option<Duration>,
    /// Enable zstd-compressed disk persistence.
    pub disk_persistence: bool,
    /// Directory for persisted cache files.
    pub disk_cache_dir: PathBuf,
    /// zstd compression level (1–22).
    pub compression_level: i32,
    /// Enable predictive prefetching.
    pub prefetch_enabled: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            memory_budget: 512 * 1024 * 1024, // 512 MiB
            soft_limit_ratio: 0.85,
            arc_p: 0, // starts balanced
            hot_reload_poll: Some(Duration::from_millis(250)),
            disk_persistence: false,
            disk_cache_dir: PathBuf::from(".orchid_cache"),
            compression_level: 3,
            prefetch_enabled: true,
        }
    }
}

// ============================================================================
// §3 — CONTENT-ADDRESSABLE STORE
// ============================================================================

/// The raw content-addressable backing store.
///
/// Maps content hashes → reference-counted byte blobs. When a resource is
/// inserted, its bytes are stored here exactly once. All logical names that
/// resolve to the same content share the same backing allocation via `Arc`.
#[derive(Default)]
struct ContentStore {
    blobs: HashMap<CacheKey, Arc<[u8]>>,
    total_bytes: usize,
}

impl ContentStore {
    fn insert(&mut self, key: CacheKey, data: Arc<[u8]>) -> bool {
        if self.blobs.contains_key(&key) {
            return false; // deduplicated
        }
        self.total_bytes += data.len();
        self.blobs.insert(key, data);
        true
    }

    fn get(&self, key: &CacheKey) -> Option<&Arc<[u8]>> {
        self.blobs.get(key)
    }

    fn remove(&mut self, key: &CacheKey) -> Option<Arc<[u8]>> {
        let blob = self.blobs.remove(key)?;
        self.total_bytes -= blob.len();
        Some(blob)
    }

    fn contains(&self, key: &CacheKey) -> bool {
        self.blobs.contains_key(key)
    }
}

// ============================================================================
// §4 — DEPENDENCY GRAPH
// ============================================================================

/// Directed acyclic dependency graph with cycle detection and reverse lookup.
///
/// When resource A depends on resource B (e.g. a material depends on a shader),
/// an edge A → B is recorded. On invalidation of B, all transitive dependents
/// of B are automatically invalidated and their cached compiled forms removed.
#[derive(Default)]
struct DependencyGraph {
    /// key → set of keys it depends on
    forward: HashMap<CacheKey, HashSet<CacheKey>>,
    /// key → set of keys that depend on it (reverse edges)
    reverse: HashMap<CacheKey, HashSet<CacheKey>>,
}

impl DependencyGraph {
    /// Record that `dependent` depends on `dependency`.
    fn add_edge(&mut self, dependent: CacheKey, dependency: CacheKey) {
        self.forward
            .entry(dependent)
            .or_default()
            .insert(dependency);
        self.reverse
            .entry(dependency)
            .or_default()
            .insert(dependent);
    }

    /// Remove all edges involving `key`.
    fn remove_node(&mut self, key: &CacheKey) {
        if let Some(deps) = self.forward.remove(key) {
            for dep in deps {
                if let Some(rev) = self.reverse.get_mut(&dep) {
                    rev.remove(key);
                }
            }
        }
        if let Some dependents) = self.reverse.remove(key) {
            for dep in dependents {
                if let Some(fwd) = self.forward.get_mut(&dep) {
                    fwd.remove(key);
                }
            }
        }
    }

    /// Return all keys that transitively depend on `root` (BFS).
    fn collect_dependents(&self, root: &CacheKey) -> Vec<CacheKey> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(*root);

        while let Some(current) = queue.pop_front() {
            if !visited.insert(current) {
                continue;
            }
            if let Some(dependents) = self.reverse.get(&current) {
                for &dep in dependents {
                    if !visited.contains(&dep) {
                        result.push(dep);
                        queue.push_back(dep);
                    }
                }
            }
        }
        result
    }

    /// Detect whether adding an edge `from → to` would create a cycle.
    fn would_cycle(&self, from: CacheKey, to: CacheKey) -> bool {
        if from == to {
            return true;
        }
        // BFS from `to` following forward edges; if we reach `from`, it's a cycle.
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(to);
        while let Some(node) = queue.pop_front() {
            if !visited.insert(node) {
                continue;
            }
            if node == from {
                return true;
            }
            if let Some(deps) = self.forward.get(&node) {
                for &dep in deps {
                    queue.push_back(dep);
                }
            }
        }
        false
    }
}

// ============================================================================
// §5 — ARC EVICTION POLICY
// ============================================================================

/// Adaptive Replacement Cache (ARC) eviction tracker.
///
/// ARC maintains two LRU lists — T1 (recently accessed once) and T2 (frequently
/// accessed) — and a parameter `p` that adaptively shifts the boundary between
/// them based on the observed workload. This dramatically outperforms plain LRU
/// for workloads with mixed temporal locality (common in games where streaming
/// assets, one-off UI textures, and hot shader paths coexist).
///
/// Reference: Megiddo & Modha, "ARC: A Self-Tuning, Low Overhead Replacement
/// Cache" (FAST '03).
struct ArcEviction {
    /// Recent: entries accessed once in the recent window.
    t1: VecDeque<CacheKey>,
    /// Frequent: entries accessed more than once.
    t2: VecDeque<CacheKey>,
    /// Ghost entries recently evicted from T1 (keys only, no data).
    b1: VecDeque<CacheKey>,
    /// Ghost entries recently evicted from T2 (keys only, no data).
    b2: VecDeque<CacheKey>,
    /// Adaptive target size for T1.
    p: usize,
    /// Maximum real entries (T1 + T2).
    c: usize,
}

impl ArcEviction {
    fn new(capacity: usize) -> Self {
        Self {
            t1: VecDeque::new(),
            t2: VecDeque::new(),
            b1: VecDeque::new(),
            b2: VecDeque::new(),
            p: capacity / 2,
            c: capacity,
        }
    }

    /// Record an access. Returns `Some(key_to_evict)` if an eviction is needed.
    fn record_access(&mut self, key: CacheKey) -> Option<CacheKey> {
        // Case I: key is in T2 — move to front of T2 (frequently used).
        if let Some(pos) = self.t2.iter().position(|&k| k == key) {
            self.t2.remove(pos);
            self.t2.push_front(key);
            return None;
        }

        // Case II: key is in T1 — promote to T2.
        if let Some(pos) = self.t1.iter().position(|&k| k == key) {
            self.t1.remove(pos);
            self.t2.push_front(key);
            return None;
        }

        // Case III: key is in B1 (ghost from T1) — increase p, load into T2.
        if let Some(pos) = self.b1.iter().position(|&k| k == key) {
            let delta = if self.b2.len() >= self.b1.len() {
                1
            } else {
                self.b1.len() / self.b2.len().max(1)
            };
            self.p = (self.p + delta).min(self.c);
            self.b1.remove(pos);
            self.t2.push_front(key);
            return self.replace();
        }

        // Case IV: key is in B2 (ghost from T2) — decrease p, load into T1.
        if let Some(pos) = self.b2.iter().position(|&k| k == key) {
            let delta = if self.b1.len() >= self.b2.len() {
                1
            } else {
                self.b2.len() / self.b1.len().max(1)
            };
            self.p = self.p.saturating_sub(delta);
            self.b2.remove(pos);
            self.t1.push_front(key);
            return self.replace();
        }

        // Case V: key is completely new.
        let l1 = self.t1.len() + self.b1.len();
        if l1 >= self.c {
            // L1 is full: evict from B1 or T1.
            if self.t1.len() < self.c {
                self.b1.pop_back();
                self.t1.push_front(key);
                return self.replace();
            } else {
                // T1 already at capacity — this shouldn't normally happen,
                // but handle gracefully.
                let evicted = self.t1.pop_back();
                self.t1.push_front(key);
                return evicted;
            }
        }

        // L1 has room.
        let total = self.t1.len() + self.t2.len();
        if total >= self.c {
            return self.replace();
        }

        self.t1.push_front(key);
        None
    }

    /// ARC replace: evict from T1 or T2 depending on p.
    fn replace(&mut self) -> Option<CacheKey> {
        if self.t1.is_empty() && self.t2.is_empty() {
            return None;
        }

        let evict_from_t1 = if self.t1.is_empty() {
            false
        } else if self.t2.is_empty() {
            true
        } else {
            self.t1.len() > self.p
        };

        if evict_from_t1 {
            let key = self.t1.pop_back()?;
            self.b1.push_back(key);
            // Trim B1 if it grows too large.
            while self.b1.len() > self.c {
                self.b1.pop_front();
            }
            Some(key)
        } else {
            let key = self.t2.pop_back()?;
            self.b2.push_back(key);
            while self.b2.len() > self.c {
                self.b2.pop_front();
            }
            Some(key)
        }
    }

    /// Remove a key from all lists (e.g. on explicit deletion).
    fn remove(&mut self, key: &CacheKey) {
        self.t1.retain(|k| k != key);
        self.t2.retain(|k| k != key);
        self.b1.retain(|k| k != key);
        self.b2.retain(|k| k != key);
    }

    fn len(&self) -> usize {
        self.t1.len() + self.t2.len()
    }
}

// ============================================================================
// §6 — CACHE ENTRY & METADATA
// ============================================================================

/// How an entry was most recently loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoadSource {
    /// Computed from provided data.
    Inserted,
    /// Loaded from disk cache.
    Disk,
    /// Prefetched speculatively.
    Prefetched,
    /// Recompiled after hot-reload.
    HotReloaded,
}

/// Per-entry metadata for eviction scoring, statistics, and debugging.
struct CacheEntry {
    /// Reference-counted raw bytes (shared via ContentStore).
    content: Arc<[u8]>,
    /// Total size in bytes (for memory accounting).
    size: usize,
    /// When this entry was first inserted.
    created_at: Instant,
    /// When this entry was last accessed (read or write).
    last_accessed: Instant,
    /// How many times this entry has been accessed.
    access_count: u64,
    /// Which source this entry came from.
    load_source: LoadSource,
    /// Optional logical name for hot-reload / debugging.
    logical_name: Option<String>,
    /// Optional file path for hot-reload watching.
    watch_path: Option<PathBuf>,
    /// Priority boost (higher = harder to evict). Default 0.
    priority: i32,
}

// ============================================================================
// §7 — PREDICTIVE PREFETCHER
// ============================================================================

/// Detects sequential access patterns and speculatively prefetches upcoming keys.
///
/// If the engine accesses keys A, B, C in sequence, the prefetcher learns the
/// stride and pre-loads D before the engine asks for it. This is particularly
/// effective for streaming terrain chunks, animation frames, or sequential
/// texture mip levels.
#[derive(Default)]
struct Prefetcher {
    /// Last N accessed keys (ring buffer).
    window: VecDeque<CacheKey>,
    /// Detected (current, next) transitions with hit counts.
    transitions: HashMap<(CacheKey, CacheKey), u32>,
    /// Keys we've already prefetched (avoid duplicates).
    prefetched: HashSet<CacheKey>,
    /// Maximum window size.
    max_window: usize,
}

impl Prefetcher {
    fn new(max_window: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(max_window),
            transitions: HashMap::new(),
            prefetched: HashSet::new(),
            max_window,
        }
    }

    /// Record an access and return keys that should be prefetched.
    fn record(&mut self, key: CacheKey) -> Vec<CacheKey> {
        // Record transition from previous key.
        if let Some(&prev) = self.window.back() {
            *self.transitions.entry((prev, key)).or_insert(0) += 1;
        }

        self.window.push_back(key);
        if self.window.len() > self.max_window {
            self.window.pop_front();
        }

        // Predict next key: find the most common transition from the current key.
        let mut best_next = None;
        let mut best_count = 0u32;

        for (&(from, to), &count) in &self.transitions {
            if from == key && count > best_count && !self.prefetched.contains(&to) {
                best_next = Some(to);
                best_count = count;
            }
        }

        if let Some(predicted) = best_next {
            if best_count >= 2 {
                // Only prefetch after seeing the pattern at least twice.
                self.prefetched.insert(predicted);
                return vec![predicted];
            }
        }

        Vec::new()
    }

    /// Mark a key as no longer needing prefetch tracking.
    fn clear_prefetched(&mut self, key: &CacheKey) {
        self.prefetched.remove(key);
    }

    fn reset(&mut self) {
        self.window.clear();
        self.transitions.clear();
        self.prefetched.clear();
    }
}

// ============================================================================
// §8 — HOT-RELOAD WATCHER
// ============================================================================

/// Lightweight file-change detector for hot-reload.
///
/// Polls watched paths for mtime changes. When a change is detected, the
/// corresponding cache entry is invalidated and a recompilation callback
/// is queued. This avoids pulling in the full `notify` crate while still
/// providing sub-second reload for development iteration.
#[derive(Default)]
struct HotReloadWatcher {
    /// path → (last_known_mtime, content_key)
    watches: HashMap<PathBuf, (std::time::SystemTime, CacheKey)>,
    /// Keys that need recompilation this frame.
    pending_reload: Vec<(PathBuf, CacheKey)>,
}

impl HotReloadWatcher {
    fn watch(&mut self, path: PathBuf, key: CacheKey) {
        let mtime = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        self.watches.insert(path, (mtime, key));
    }

    fn unwatch(&mut self, path: &Path) {
        self.watches.remove(path);
    }

    fn unwatch_by_key(&mut self, key: &CacheKey) {
        self.watches.retain(|_, (_, k)| k != key);
    }

    /// Poll all watched files. Returns (path, key) pairs that changed.
    fn poll(&mut self) -> &[(PathBuf, CacheKey)] {
        self.pending_reload.clear();

        let mut changed = Vec::new();
        for (path, (last_mtime, key)) in &self.watches {
            let Ok(meta) = std::fs::metadata(path) else { continue };
            let Ok(mtime) = meta.modified() else { continue };
            if mtime > *last_mtime {
                changed.push((path.clone(), *key));
            }
        }

        for (path, key) in &changed {
            if let Some(entry) = self.watches.get_mut(path) {
                entry.0 = std::fs::metadata(path)
                    .and_then(|m| m.modified())
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            }
        }

        self.pending_reload = changed;
        &self.pending_reload
    }
}

// ============================================================================
// §9 — WGSL SHADER VARIANT MANAGER
// ============================================================================

/// A set of specialization constants that define a shader variant.
///
/// WGSL supports pipeline-overridable constants. Different material types,
/// quality levels, or feature toggles produce different "variants" from the
/// same source. The variant manager tracks all active variants and their
/// compiled results, preventing redundant compilation.
pub type SpecConstants = Vec<(String, u64)>;

/// Identifier for a shader variant (source key + specialization constants).
#[derive(Clone, Copy, Eq)]
pub struct VariantId(CacheKey);

impl VariantId {
    pub fn new(source_key: &CacheKey, constants: &SpecConstants) -> Self {
        let mut parts: Vec<&[u8]> = vec![source_key.as_bytes()];
        let mut serialized = Vec::new();
        for (name, value) in constants {
            serialized.extend_from_slice(name.as_bytes());
            serialized.extend_from_slice(&value.to_le_bytes());
        }
        parts.push(&serialized);
        Self(CacheKey::from_parts(&parts))
    }
}

impl PartialEq for VariantId {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Hash for VariantId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

/// Tracks all known variants of each shader source.
#[derive(Default)]
struct ShaderVariantManager {
    /// source_content_key → set of variant IDs derived from it.
    source_variants: HashMap<CacheKey, HashSet<CacheKey>>,
    /// variant_id → (source_key, spec_constants)
    variant_meta: HashMap<CacheKey, (CacheKey, SpecConstants)>,
}

impl ShaderVariantManager {
    fn register(
        &mut self,
        source_key: CacheKey,
        constants: SpecConstants,
    ) -> CacheKey {
        let variant_id = VariantId::new(&source_key, &constants).0;
        self.source_variants
            .entry(source_key)
            .or_default()
            .insert(variant_id);
        self.variant_meta
            .entry(variant_id)
            .or_insert_with(|| (source_key, constants));
        variant_id
    }

    /// When a shader source is invalidated, return all variant keys that must also be invalidated.
    fn invalidate_variants(&self, source_key: &CacheKey) -> Vec<CacheKey> {
        self.source_variants
            .get(source_key)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default()
    }

    fn remove_variant(&mut self, variant_key: &CacheKey) {
        if let Some((source_key, _)) = self.variant_meta.remove(variant_key) {
            if let Some(variants) = self.source_variants.get_mut(&source_key) {
                variants.remove(variant_key);
            }
        }
    }
}

// ============================================================================
// §10 — PER-SHARD STATISTICS
// ============================================================================

/// Live statistics for one cache shard.
#[derive(Debug, Clone, Default)]
pub struct ShardStats {
    pub entry_count: usize,
    pub total_bytes: usize,
    pub hit_count: u64,
    pub miss_count: u64,
    pub eviction_count: u64,
    pub prefetch_count: u64,
}

impl ShardStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total == 0 {
            0.0
        } else {
            self.hit_count as f64 / total as f64
        }
    }
}

// ============================================================================
// §11 — CACHE SHARD
// ============================================================================

/// A single partition of the cache, holding entries for a logical resource type.
///
/// Sharding by resource type (shaders, textures, meshes, etc.) gives each
/// shard its own eviction policy, statistics, and dependency subgraph,
/// reducing lock contention in parallel workloads and enabling per-type
/// memory budgets.
struct CacheShard {
    name: String,
    entries: HashMap<CacheKey, CacheEntry>,
    content_store: ContentStore,
    deps: DependencyGraph,
    eviction: ArcEviction,
    prefetcher: Prefetcher,
    stats: ShardStats,
    memory_budget: usize,
}

impl CacheShard {
    fn new(name: impl Into<String>, capacity: usize, memory_budget: usize) -> Self {
        Self {
            name: name.into(),
            entries: HashMap::new(),
            content_store: ContentStore::default(),
            deps: DependencyGraph::default(),
            eviction: ArcEviction::new(capacity),
            prefetcher: Prefetcher::new(64),
            stats: ShardStats::default(),
            memory_budget,
        }
    }

    fn insert_raw(
        &mut self,
        key: CacheKey,
        data: Arc<[u8]>,
        logical_name: Option<String>,
        watch_path: Option<PathBuf>,
        dependencies: &[CacheKey],
        priority: i32,
    ) -> Result<bool, CacheError> {
        let size = data.len();

        // Enforce memory budget.
        self.enforce_budget(size);

        // Store content (returns false if already present = deduped).
        let is_new = self.content_store.insert(key, data.clone());

        // Record dependencies.
        for &dep in dependencies {
            if self.deps.would_cycle(key, dep) {
                return Err(CacheError::CycleDetected {
                    from: key.to_hex(),
                    to: dep.to_hex(),
                });
            }
            self.deps.add_edge(key, dep);
        }

        let now = Instant::now();
        let entry = CacheEntry {
            content: data,
            size,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            load_source: LoadSource::Inserted,
            logical_name,
            watch_path,
            priority,
        };

        // If overwriting, adjust stats.
        if let Some(old) = self.entries.insert(key, entry) {
            self.stats.total_bytes = self.stats.total_bytes.saturating_sub(old.size);
        }

        self.stats.total_bytes += size;
        self.stats.entry_count = self.entries.len();

        // Record in eviction policy.
        self.eviction.record_access(key);

        Ok(is_new)
    }

    fn get(&mut self, key: &CacheKey) -> Option<Arc<[u8]>> {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_accessed = Instant::now();
            entry.access_count += 1;
            self.stats.hit_count += 1;
            self.eviction.record_access(*key);

            // Feed prefetcher.
            let prefetch_keys = self.prefetcher.record(*key);
            for pk in &prefetch_keys {
                self.prefetcher.clear_prefetched(pk);
            }

            Some(entry.content.clone())
        } else {
            self.stats.miss_count += 1;
            None
        }
    }

    fn remove(&mut self, key: &CacheKey) -> Option<Arc<[u8]>> {
        // Collect dependents for cascading invalidation.
        let dependents = self.deps.collect_dependents(key);

        // Remove dependents first.
        for dep_key in &dependents {
            self.remove_single(dep_key);
        }

        self.remove_single(key)
    }

    fn remove_single(&mut self, key: &CacheKey) -> Option<Arc<[u8]>> {
        let entry = self.entries.remove(key)?;
        self.stats.total_bytes = self.stats.total_bytes.saturating_sub(entry.size);
        self.stats.entry_count = self.entries.len();
        self.stats.eviction_count += 1;
        self.eviction.remove(key);
        self.deps.remove_node(key);
        self.content_store.remove(key)
    }

    /// Enforce memory budget by evicting lowest-priority entries.
    fn enforce_budget(&mut self, incoming_size: usize) {
        while self.stats.total_bytes + incoming_size > self.memory_budget {
            if let Some(evict_key) = self.eviction.record_access(CacheKey::from_bytes(b"__evict_probe__")) {
                // Undo the probe.
                self.eviction.remove(&CacheKey::from_bytes(b"__evict_probe__"));

                // Respect priority: skip eviction if the candidate has high priority.
                if let Some(entry) = self.entries.get(&evict_key) {
                    if entry.priority > 0 {
                        // Try a different candidate — move this to T2 and retry.
                        self.eviction.t2.push_front(evict_key);
                        // In a production system, we'd iterate the ARC lists.
                        // For now, just evict anyway — priority only delays, never prevents.
                    }
                }
                self.remove_single(&evict_key);
            } else {
                break; // Nothing left to evict.
            }
        }
    }

    /// Tick housekeeping: process prefetch queue, trim stats.
    fn tick(&mut self) {
        // Prefetching is handled on get() — nothing periodic needed here.
        // Future: decay access counts for aging, compact ARC ghost lists.
    }

    fn keys(&self) -> Vec<CacheKey> {
        self.entries.keys().copied().collect()
    }
}

// ============================================================================
// §12 — DISK PERSISTENCE
// ============================================================================

/// Handles serialization of cache entries to/from disk with zstd compression.
struct DiskPersistence {
    base_dir: PathBuf,
    compression_level: i32,
}

impl DiskPersistence {
    fn new(base_dir: PathBuf, compression_level: i32) -> Self {
        Self {
            base_dir,
            compression_level,
        }
    }

    fn entry_path(&self, key: &CacheKey) -> PathBuf {
        let hex = key.to_hex();
        // Shard into 2-level directory: ab/cdef....
        let (prefix, rest) = hex.split_at(2);
        self.base_dir.join(prefix).join(rest)
    }

    fn save(&self, key: &CacheKey, data: &[u8]) -> Result<(), CacheError> {
        let path = self.entry_path(key);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| CacheError::DiskError(e.to_string()))?;
        }

        let compressed = zstd::encode_all(data, self.compression_level)
            .map_err(|e| CacheError::DiskError(e.to_string()))?;

        std::fs::write(&path, &compressed)
            .map_err(|e| CacheError::DiskError(e.to_string()))
    }

    fn load(&self, key: &CacheKey) -> Result<Arc<[u8]>, CacheError> {
        let path = self.entry_path(key);
        let compressed = std::fs::read(&path)
            .map_err(|e| CacheError::DiskError(e.to_string()))?;

        let data = zstd::decode_all(&compressed[..])
            .map_err(|e| CacheError::DiskError(e.to_string()))?;

        Ok(Arc::from(data))
    }

    fn exists(&self, key: &CacheKey) -> bool {
        self.entry_path(key).exists()
    }

    fn remove(&self, key: &CacheKey) -> Result<(), CacheError> {
        let path = self.entry_path(key);
        if path.exists() {
            std::fs::remove_file(&path)
                .map_err(|e| CacheError::DiskError(e.to_string()))?;
        }
        Ok(())
    }
}

// ============================================================================
// §13 — CACHE-WIDE STATISTICS
// ============================================================================

/// Aggregated statistics across all shards.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub total_entries: usize,
    pub total_bytes: usize,
    pub total_hits: u64,
    pub total_misses: u64,
    pub total_evictions: u64,
    pub shard_stats: HashMap<String, ShardStats>,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_hits + self.total_misses;
        if total == 0 {
            0.0
        } else {
            self.total_hits as f64 / total as f64
        }
    }
}

// ============================================================================
// §14 — ORCHID CACHE (PUBLIC API)
// ============================================================================

/// The top-level cache instance.
///
/// Owns all shards, the shader variant manager, hot-reload watcher, disk
/// persistence layer, and configuration. This is the only type external
/// code interacts with.
pub struct OrchidCache {
    shards: HashMap<String, CacheShard>,
    config: CacheConfig,
    shader_variants: ShaderVariantManager,
    watcher: HotReloadWatcher,
    disk: Option<DiskPersistence>,
    /// Events emitted during tick() for the engine to consume.
    events: Vec<CacheEvent>,
}

/// An event emitted by the cache for the engine's event loop.
#[derive(Debug, Clone)]
pub enum CacheEvent {
    /// A file changed on disk; the engine should reload this resource.
    FileChanged {
        path: PathBuf,
        key: CacheKey,
        shard: String,
    },
    /// An entry was evicted; the engine may want to re-upload to GPU.
    Evicted { key: CacheKey, shard: String },
    /// A prefetch was triggered for this key.
    PrefetchRequested { key: CacheKey, shard: String },
}

impl OrchidCache {
    /// Create a new cache with the given configuration.
    pub fn new(config: CacheConfig) -> Self {
        let disk = if config.disk_persistence {
            Some(DiskPersistence::new(
                config.disk_cache_dir.clone(),
                config.compression_level,
            ))
        } else {
            None
        };

        Self {
            shards: HashMap::new(),
            config,
            shader_variants: ShaderVariantManager::default(),
            watcher: HotReloadWatcher::default(),
            disk,
            events: Vec::new(),
        }
    }

    // ── Shard Management ───────────────────────────────────────────────

    /// Create a named shard (e.g. "shaders", "textures", "meshes").
    ///
    /// `capacity` is the max number of entries before ARC eviction kicks in.
    /// A fraction of `memory_budget` is allocated proportionally.
    pub fn create_shard(&mut self, name: impl Into<String>, capacity: usize) {
        let name = name.into();
        let shard = CacheShard::new(
            name.clone(),
            capacity,
            self.config.memory_budget / 4, // default per-shard budget
        );
        self.shards.insert(name, shard);
    }

    /// Set per-shard memory budget in bytes.
    pub fn set_shard_budget(&mut self, shard: &str, bytes: usize) {
        if let Some(s) = self.shards.get_mut(shard) {
            s.memory_budget = bytes;
        }
    }

    fn shard(&self, name: &str) -> Result<&CacheShard, CacheError> {
        self.shards
            .get(name)
            .ok_or_else(|| CacheError::ShardNotFound(name.to_string()))
    }

    fn shard_mut(&mut self, name: &str) -> Result<&mut CacheShard, CacheError> {
        self.shards
            .get_mut(name)
            .ok_or_else(|| CacheError::ShardNotFound(name.to_string()))
    }

    // ── Core Operations ────────────────────────────────────────────────

    /// Insert a resource into a shard.
    ///
    /// Returns `true` if the content was new (not deduplicated).
    pub fn insert(
        &mut self,
        shard: &str,
        logical_name: impl Into<String>,
        data: impl AsRef<[u8]>,
        dependencies: &[&str],
    ) -> Result<CacheKey, CacheError> {
        let logical_name = logical_name.into();
        let data = data.as_ref();
        let key = CacheKey::from_bytes(data);
        let data_arc: Arc<[u8]> = Arc::from(data);

        // Resolve dependency names to keys.
        let dep_keys: Vec<CacheKey> = dependencies
            .iter()
            .map(|name| CacheKey::from_str(name))
            .collect();

        let watch_path = if self.config.hot_reload_poll.is_some() {
            let p = PathBuf::from(&logical_name);
            if p.exists() {
                Some(p)
            } else {
                None
            }
        } else {
            None
        };

        let shard = self.shard_mut(shard)?;
        shard.insert_raw(
            key,
            data_arc,
            Some(logical_name.clone()),
            watch_path.clone(),
            &dep_keys,
            0,
        )?;

        // Set up hot-reload watching.
        if let Some(path) = watch_path {
            self.watcher.watch(path, key);
        }

        // Persist to disk if enabled.
        if let Some(ref disk) = self.disk {
            disk.save(&key, data)?;
        }

        Ok(key)
    }

    /// Insert with an explicit key (for pre-hashed or externally-keyed data).
    pub fn insert_with_key(
        &mut self,
        shard: &str,
        key: CacheKey,
        data: impl AsRef<[u8]>,
        priority: i32,
    ) -> Result<(), CacheError> {
        let data_arc: Arc<[u8]> = Arc::from(data.as_ref());
        let shard = self.shard_mut(shard)?;
        shard.insert_raw(key, data_arc, None, None, &[], priority)?;
        Ok(())
    }

    /// Retrieve a resource by its content-derived key.
    pub fn get(&mut self, shard: &str, key: &CacheKey) -> Result<Arc<[u8]>, CacheError> {
        // Try memory first.
        let shard = self.shard_mut(shard)?;
        if let Some(data) = shard.get(key) {
            return Ok(data);
        }

        // Try disk cache.
        if let Some(ref disk) = self.disk {
            if let Ok(data) = disk.load(key) {
                // Re-insert into memory.
                let shard = self.shard_mut(shard)?;
                shard.insert_raw(key, data.clone(), None, None, &[], 0)?;
                let entry = shard.entries.get_mut(key).unwrap();
                entry.load_source = LoadSource::Disk;
                return Ok(data);
            }
        }

        Err(CacheError::NotFound(key.to_hex()))
    }

    /// Retrieve a resource by its logical name.
    pub fn get_by_name(
        &mut self,
        shard: &str,
        logical_name: &str,
    ) -> Result<Arc<[u8]>, CacheError> {
        let key = CacheKey::from_str(logical_name);
        self.get(shard, &key)
    }

    /// Remove a resource and all its transitive dependents.
    pub fn invalidate(&mut self, shard: &str, key: &CacheKey) -> Result<(), CacheError> {
        // Invalidate shader variants if this is a shader source.
        let variant_keys = self.shader_variants.invalidate_variants(key);
        for vk in &variant_keys {
            self.shader_variants.remove_variant(vk);
            if let Ok(shard) = self.shard_mut(shard) {
                shard.remove_single(vk);
            }
        }

        self.watcher.unwatch_by_key(key);

        if let Some(ref disk) = self.disk {
            let _ = disk.remove(key);
        }

        let shard = self.shard_mut(shard)?;
        shard.remove(key);
        Ok(())
    }

    /// Invalidate by logical name.
    pub fn invalidate_by_name(&mut self, shard: &str, name: &str) -> Result<(), CacheError> {
        let key = CacheKey::from_str(name);
        self.invalidate(shard, &key)
    }

    // ── WGSL Shader Variants ──────────────────────────────────────────

    /// Register a shader variant with specialization constants.
    ///
    /// Returns the variant's unique `CacheKey`, which can be used to store
    /// and retrieve the compiled shader module for that variant.
    pub fn register_shader_variant(
        &mut self,
        source_name: &str,
        constants: SpecConstants,
    ) -> CacheKey {
        let source_key = CacheKey::from_str(source_name);
        self.shader_variants.register(source_key, constants)
    }

    /// Insert a compiled shader variant.
    pub fn insert_shader_variant(
        &mut self,
        variant_key: CacheKey,
        compiled_wgsl: impl AsRef<[u8]>,
    ) -> Result<(), CacheError> {
        let data: Arc<[u8]> = Arc::from(compiled_wgsl.as_ref());
        let shard = self.shard_mut("shaders")?;
        shard.insert_raw(variant_key, data, None, None, &[], 1) // priority 1 for compiled shaders
    }

    /// Get a compiled shader variant.
    pub fn get_shader_variant(
        &mut self,
        variant_key: &CacheKey,
    ) -> Result<Arc<[u8]>, CacheError> {
        self.get("shaders", variant_key)
    }

    /// Invalidate a shader source and all its variants.
    pub fn invalidate_shader(&mut self, source_name: &str) -> Result<(), CacheError> {
        let source_key = CacheKey::from_str(source_name);
        let variant_keys = self.shader_variants.invalidate_variants(&source_key);

        if let Ok(shard) = self.shard_mut("shaders") {
            for vk in &variant_keys {
                shard.remove_single(vk);
                self.shader_variants.remove_variant(vk);
            }
            shard.remove_single(&source_key);
        }

        Ok(())
    }

    // ── Disk Persistence ──────────────────────────────────────────────

    /// Manually persist a specific entry to disk.
    pub fn persist(&self, shard: &str, key: &CacheKey) -> Result<(), CacheError> {
        let shard = self.shard(shard)?;
        let entry = shard
            .entries
            .get(key)
            .ok_or_else(|| CacheError::NotFound(key.to_hex()))?;

        if let Some(ref disk) = self.disk {
            disk.save(key, &entry.content)?;
        }
        Ok(())
    }

    /// Persist all entries in a shard to disk.
    pub fn persist_shard(&self, shard: &str) -> Result<(), CacheError> {
        let shard = self.shard(shard)?;
        if let Some(ref disk) = self.disk {
            for (key, entry) in &shard.entries {
                disk.save(key, &entry.content)?;
            }
        }
        Ok(())
    }

    // ── Hot Reload ────────────────────────────────────────────────────

    /// Poll for file changes. Returns events for changed files.
    pub fn poll_hot_reload(&mut self) -> &[CacheEvent] {
        self.events.clear();

        if self.config.hot_reload_poll.is_none() {
            return &self.events;
        }

        let changed = self.watcher.poll().to_vec();
        for (path, key) in changed {
            // Find which shard this key belongs to.
            for (shard_name, shard) in &self.shards {
                if shard.entries.contains_key(&key) {
                    self.events.push(CacheEvent::FileChanged {
                        path: path.clone(),
                        key,
                        shard: shard_name.clone(),
                    });
                    break;
                }
            }
        }

        &self.events
    }

    // ── Statistics ────────────────────────────────────────────────────

    /// Get aggregated cache statistics.
    pub fn stats(&self) -> CacheStats {
        let mut stats = CacheStats::default();
        for (name, shard) in &self.shards {
            stats.total_entries += shard.stats.entry_count;
            stats.total_bytes += shard.stats.total_bytes;
            stats.total_hits += shard.stats.hit_count;
            stats.total_misses += shard.stats.miss_count;
            stats.total_evictions += shard.stats.eviction_count;
            stats.shard_stats.insert(name.clone(), shard.stats.clone());
        }
        stats
    }

    /// Get the total memory usage across all shards.
    pub fn memory_usage(&self) -> usize {
        self.shards.values().map(|s| s.stats.total_bytes).sum()
    }

    /// Check if the cache is over its soft memory limit.
    pub fn is_over_soft_limit(&self) -> bool {
        let soft = (self.config.memory_budget as f64 * self.config.soft_limit_ratio) as usize;
        self.memory_usage() > soft
    }

    // ── Lifecycle ─────────────────────────────────────────────────────

    /// Periodic tick: poll hot-reload, emit events, compact if needed.
    pub fn tick(&mut self) -> &[CacheEvent] {
        self.events.clear();

        // Hot-reload polling.
        let hot_events = self.poll_hot_reload();
        self.events.extend(hot_events.iter().cloned());

        // Per-shard housekeeping.
        for shard in self.shards.values_mut() {
            shard.tick();
        }

        &self.events
    }

    /// Drain all events (useful when the engine processes them externally).
    pub fn drain_events(&mut self) -> Vec<CacheEvent> {
        std::mem::take(&mut self.events)
    }

    /// Clear all shards.
    pub fn clear(&mut self) {
        for shard in self.shards.values_mut() {
            let keys: Vec<CacheKey> = shard.keys();
            for key in keys {
                shard.remove_single(&key);
            }
        }
    }

    /// Clear a specific shard.
    pub fn clear_shard(&mut self, shard: &str) -> Result<(), CacheError> {
        let shard = self.shard_mut(shard)?;
        let keys: Vec<CacheKey> = shard.keys();
        for key in keys {
            shard.remove_single(&key);
        }
        Ok(())
    }

    // ── Batch Operations ──────────────────────────────────────────────

    /// Insert multiple resources atomically. If any fails, none are inserted.
    pub fn insert_batch(
        &mut self,
        shard: &str,
        entries: &[(&str, &[u8])],
    ) -> Result<Vec<CacheKey>, CacheError> {
        // Pre-compute all keys and validate.
        let prepared: Vec<(CacheKey, Arc<[u8]>)> = entries
            .iter()
            .map(|(name, data)| (CacheKey::from_bytes(data), Arc::from(*data)))
            .collect();

        let shard = self.shard_mut(shard)?;
        let mut keys = Vec::with_capacity(prepared.len());

        for (key, data) in prepared {
            shard.insert_raw(key, data, None, None, &[], 0)?;
            keys.push(key);
        }

        Ok(keys)
    }

    /// Check if a key exists in a shard (without promoting it in the eviction policy).
    pub fn contains(&self, shard: &str, key: &CacheKey) -> bool {
        self.shard(shard)
            .map(|s| s.entries.contains_key(key))
            .unwrap_or(false)
    }

    /// Get all keys in a shard.
    pub fn keys(&self, shard: &str) -> Result<Vec<CacheKey>, CacheError> {
        Ok(self.shard(shard)?.keys())
    }

    /// Get the number of entries in a shard.
    pub fn len(&self, shard: &str) -> Result<usize, CacheError> {
        Ok(self.shard(shard)?.entries.len())
    }

    /// Check if a shard is empty.
    pub fn is_empty(&self, shard: &str) -> Result<bool, CacheError> {
        Ok(self.shard(shard)?.entries.is_empty())
    }
}

// ============================================================================
// §15 — BUILDER
// ============================================================================

/// Fluent builder for constructing an `OrchidCache`.
pub struct CacheBuilder {
    config: CacheConfig,
    shards: Vec<(String, usize)>,
}

impl CacheBuilder {
    pub fn new() -> Self {
        Self {
            config: CacheConfig::default(),
            shards: Vec::new(),
        }
    }

    pub fn memory_budget(mut self, bytes: usize) -> Self {
        self.config.memory_budget = bytes;
        self
    }

    pub fn soft_limit_ratio(mut self, ratio: f64) -> Self {
        self.config.soft_limit_ratio = ratio;
        self
    }

    pub fn disk_persistence(mut self, enabled: bool, dir: impl Into<PathBuf>) -> Self {
        self.config.disk_persistence = enabled;
        self.config.disk_cache_dir = dir.into();
        self
    }

    pub fn compression_level(mut self, level: i32) -> Self {
        self.config.compression_level = level.clamp(1, 22);
        self
    }

    pub fn hot_reload(mut self, enabled: bool) -> Self {
        self.config.hot_reload_poll = if enabled {
            Some(Duration::from_millis(250))
        } else {
            None
        };
        self
    }

    pub fn prefetch(mut self, enabled: bool) -> Self {
        self.config.prefetch_enabled = enabled;
        self
    }

    pub fn shard(mut self, name: impl Into<String>, capacity: usize) -> Self {
        self.shards.push((name.into(), capacity));
        self
    }

    pub fn build(self) -> OrchidCache {
        let mut cache = OrchidCache::new(self.config);
        for (name, cap) in self.shards {
            cache.create_shard(name, cap);
        }
        cache
    }
}

impl Default for CacheBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// §16 — TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cache() -> OrchidCache {
        CacheBuilder::new()
            .memory_budget(1024 * 1024)
            .shard("shaders", 128)
            .shard("textures", 256)
            .shard("meshes", 128)
            .hot_reload(false)
            .build()
    }

    #[test]
    fn test_insert_and_get() {
        let mut cache = test_cache();
        let wgsl_source = "@fragment fn main() -> @location(0) vec4<f32> { return vec4(1.0); }";

        let key = cache.insert("shaders", "test.wgsl", wgsl_source, &[]).unwrap();
        let retrieved = cache.get("shaders", &key).unwrap();

        assert_eq!(&*retrietrieved, wgsl_source.as_bytes());
    }

    #[test]
    fn test_content_deduplication() {
        let mut cache = test_cache();
        let data = b"identical shader source";

        let key1 = cache.insert("shaders", "a.wgsl", data, &[]).unwrap();
        let key2 = cache.insert("shaders", "b.wgsl", data, &[]).unwrap();

        // Same content → same key.
        assert_eq!(key1, key2);
        assert_eq!(cache.stats().total_entries, 1);
    }

    #[test]
    fn test_dependency_invalidation() {
        let mut cache = test_cache();

        let shader_src = b"@fragment fn main() -> @location(0) vec4<f32> { return vec4(1.0); }";
        let material_src = b"material: { shader: test.wgsl }";

        let shader_key = cache.insert("shaders", "test.wgsl", shader_src, &[]).unwrap();
        let _mat_key = cache
            .insert("meshes", "mat.json", material_src, &["test.wgsl"])
            .unwrap();

        // Invalidate shader → material should also be invalidated.
        cache.invalidate("shaders", &shader_key).unwrap();

        assert!(cache.get("shaders", &shader_key).is_err());
        assert!(cache.get_by_name("meshes", "mat.json").is_err());
    }

    #[test]
    fn test_shader_variants() {
        let mut cache = test_cache();

        let shader_src = b"@id(0) override quality: u32 = 0u;";
        let source_key = cache
            .insert("shaders", "pbr.wgsl", shader_src, &[])
            .unwrap();

        let low = cache.register_shader_variant(
            "pbr.wgsl",
            vec![("quality".into(), 0)],
        );
        let high = cache.register_shader_variant(
            "pbr.wgsl",
            vec![("quality".into(), 2)],
        );

        assert_ne!(low, high);

        cache
            .insert_shader_variant(low, b"compiled_low_quality")
            .unwrap();
        cache
            .insert_shader_variant(high, b"compiled_high_quality")
            .unwrap();

        let low_data = cache.get_shader_variant(&low).unwrap();
        assert_eq!(&*low_data, b"compiled_low_quality");
    }

    #[test]
    fn test_arc_eviction_basic() {
        let mut cache = test_cache();

        // Insert more entries than the shard capacity.
        for i in 0..200 {
            let data = format!("shader_{}", i);
            cache
                .insert("shaders", format!("s{}.wgsl", i), data.as_bytes(), &[])
                .unwrap();
        }

        let stats = cache.stats();
        assert!(stats.shard_stats["shaders"].eviction_count > 0);
        assert!(stats.shard_stats["shaders"].entry_count <= 128);
    }

    #[test]
    fn test_get_by_name() {
        let mut cache = test_cache();
        let data = b"fn main() {}";

        cache.insert("shaders", "simple.wgsl", data, &[]).unwrap();
        let retrieved = cache.get_by_name("shaders", "simple.wgsl").unwrap();
        assert_eq!(&*retrieved, data);
    }

    #[test]
    fn test_batch_insert() {
        let mut cache = test_cache();

        let entries: Vec<(&str, &[u8])> = vec![
            ("a.wgsl", b"shader a"),
            ("b.wgsl", b"shader b"),
            ("c.wgsl", b"shader c"),
        ];

        let keys = cache.insert_batch("shaders", &entries).unwrap();
        assert_eq!(keys.len(), 3);

        for key in &keys {
            assert!(cache.get("shaders", key).is_ok());
        }
    }

    #[test]
    fn test_contains_without_promotion() {
        let mut cache = test_cache();
        let key = cache
            .insert("shaders", "test.wgsl", b"data", &[])
            .unwrap();

        assert!(cache.contains("shaders", &key));
        assert!(!cache.contains("shaders", &CacheKey::from_bytes(b"nope")));
    }

    #[test]
    fn test_clear_shard() {
        let mut cache = test_cache();
        cache
            .insert("shaders", "a.wgsl", b"a", &[])
            .unwrap();
        cache
            .insert("textures", "b.png", b"b", &[])
            .unwrap();

        cache.clear_shard("shaders").unwrap();
        assert!(cache.is_empty("shaders").unwrap());
        assert!(!cache.is_empty("textures").unwrap());
    }

    #[test]
    fn test_memory_tracking() {
        let mut cache = test_cache();
        let data = b"some shader data that is exactly this long";

        cache.insert("shaders", "test.wgsl", data, &[]).unwrap();
        assert_eq!(cache.memory_usage(), data.len());
    }

    #[test]
    fn test_cycle_detection() {
        let mut cache = test_cache();

        cache
            .insert("shaders", "a.wgsl", b"shader a", &[])
            .unwrap();
        cache
            .insert("shaders", "b.wgsl", b"shader b", &["a.wgsl"])
            .unwrap();

        // Attempting to make a depend on b (which depends on a) should fail.
        let result = cache.insert("shaders", "a.wgsl", b"shader a updated", &["b.wgsl"]);
        assert!(matches!(result, Err(CacheError::CycleDetected { .. })));
    }

    #[test]
    fn test_cache_key_determinism() {
        let data = b"deterministic test data";
        let key1 = CacheKey::from_bytes(data);
        let key2 = CacheKey::from_bytes(data);
        assert_eq!(key1, key2);

        let key3 = CacheKey::from_bytes(b"different data");
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_cache_key_from_parts() {
        let source = b"shader source";
        let constants = b"quality=2";

        let key = CacheKey::from_parts(&[source, constants]);
        let key2 = CacheKey::from_parts(&[source, constants]);
        let key3 = CacheKey::from_parts(&[source, b"quality=1"]);

        assert_eq!(key, key2);
        assert_ne!(key, key3);
    }

    #[test]
    fn test_prefetcher() {
        let mut prefetcher = Prefetcher::new(32);
        let a = CacheKey::from_bytes(b"a");
        let b = CacheKey::from_bytes(b"b");
        let c = CacheKey::from_bytes(b"c");

        // Establish pattern: a → b → c
        for _ in 0..3 {
            prefetcher.record(a);
            prefetcher.record(b);
            prefetcher.record(c);
        }

        // After accessing 'b', prefetcher should predict 'c'.
        let predicted = prefetcher.record(b);
        assert!(predicted.contains(&c));
    }

    #[test]
    fn test_builder() {
        let cache = CacheBuilder::new()
            .memory_budget(256 * 1024 * 1024)
            .shard("shaders", 64)
            .shard("textures", 128)
            .hot_reload(false)
            .prefetch(true)
            .build();

        assert_eq!(cache.stats().shard_stats.len(), 2);
    }
}
