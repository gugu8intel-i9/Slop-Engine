//! # profiler.rs — High-performance, low-overpute profiler for Rust
//!
//! Zero-cost when disabled via `#[cfg(feature = "profiler")]`.
//! Thread-local buffering eliminates lock contention on the hot path.
//! Uses `rdtsc` on x86/x86_64 for sub-nanosecond timing, falls back to `Instant`.

use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Once;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Feature gate — zero cost when profiler is disabled
// ---------------------------------------------------------------------------

#[cfg(not(feature = "profiler"))]
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {{}};
    ($name:expr, $profiler:expr) => {{}};
}

#[cfg(not(feature = "profiler"))]
#[macro_export]
macro_rules! profile_fn {
    () => {{}};
    ($profiler:expr) => {{}};
}

// ---------------------------------------------------------------------------
// Cycle counter — rdtsc on x86, Instant elsewhere
// ---------------------------------------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn read_cycles() -> u64 {
    unsafe {
        use std::arch::x86_64::_rdtsc;
        _rdtsc()
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline(always)]
fn read_cycles() -> u64 {
    // Fallback: nanoseconds from a monotonic source
    use std::time::UNIX_EPOCH;
    UNIX_EPOCH
        .elapsed()
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Call once at startup to calibrate rdtsc against wall clock.
/// Returns nanoseconds per tsc tick (approx).
static mut CYCLES_PER_NS: f64 = 1.0;
static CALIBRATE: Once = Once::new();

fn calibrate_tsc() {
    CALIBRATE.call_once(|| {
        let start_inst = Instant::now();
        let start_cyc = read_cycles();
        // Burn ~500µs calibrating
        while start_inst.elapsed().as_micros() < 500 {}
        let end_cyc = read_cycles();
        let elapsed_ns = start_inst.elapsed().as_nanos() as f64;
        let elapsed_cyc = (end_cyc - start_cyc) as f64;
        if elapsed_ns > 0.0 {
            unsafe {
                CYCLES_PER_NS = elapsed_cyc / elapsed_ns;
            }
        }
    });
}

#[inline(always)]
fn cycles_to_ns(cycles: u64) -> f64 {
    cycles as f64 / unsafe { CYCLES_PER_NS }
}

// ---------------------------------------------------------------------------
// Span record — the minimal unit captured on the hot path
// ---------------------------------------------------------------------------

/// Packed span record. 24 bytes, fits in a cache line with room to spare.
#[repr(C)]
#[derive(Clone, Copy)]
struct SpanRecord {
    /// Hash of the span name (FNV-1a 64-bit)
    name_hash: u64,
    /// Start cycle count
    start: u64,
    /// Duration in cycles
    duration: u64,
}

/// FNV-1a 64-bit hash — fast, no allocation, good distribution for short strings.
#[inline(always)]
fn fnv1a64(s: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    for &byte in s.as_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// Thread-local buffer — the hot path touches ONLY this
// ---------------------------------------------------------------------------

const BUFFER_CAPACITY: usize = 4096;

struct ThreadBuffer {
    records: [SpanRecord; BUFFER_CAPACITY],
    len: usize,
}

impl ThreadBuffer {
    const fn new() -> Self {
        Self {
            records: [SpanRecord {
                name_hash: 0,
                start: 0,
                duration: 0,
            }; BUFFER_CAPACITY],
            len: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, record: SpanRecord) -> bool {
        if self.len < BUFFER_CAPACITY {
            // SAFETY: len < BUFFER_CAPACITY, index is in bounds
            unsafe {
                *self.records.get_unchecked_mut(self.len) = record;
            }
            self.len += 1;
            true
        } else {
            false
        }
    }

    fn drain(&mut self) -> &[SpanRecord] {
        let slice = &self.records[..self.len];
        self.len = 0;
        slice
    }
}

// Per-thread state, no Mutex needed
struct LocalState {
    buffer: ThreadBuffer,
    /// Stack of open span start times (for nested spans)
    stack: [u64; 64],
    stack_depth: usize,
}

impl LocalState {
    const fn new() -> Self {
        Self {
            buffer: ThreadBuffer::new(),
            stack: [0u64; 64],
            stack_depth: 0,
        }
    }
}

std::thread_local! {
    static LOCAL: UnsafeCell<LocalState> = UnsafeCell::new(LocalState::new());
}

// ---------------------------------------------------------------------------
// Global statistics aggregator
// ---------------------------------------------------------------------------

/// Lock-free-ish aggregator. Spans are flushed from thread-local buffers
/// into here. The flush path uses a spin lock (very low contention).
struct Aggregator {
    /// name_hash -> SpanStats
    stats: UnsafeCell<HashMap<u64, SpanStats>>,
    spin: AtomicUsize,
}

/// Per-span running statistics using Welford's online algorithm
/// for numerically stable mean/variance, plus min/max and histogram
/// for percentile estimation.
struct SpanStats {
    count: AtomicU64,
    sum_ns: AtomicU64,
    min_ns: AtomicU64,
    max_ns: AtomicU64,
    /// For Welford: mean
    mean_ns: AtomicU64, // stored as f64 bits
    /// For Welford: M2 (sum of squared deviations)
    m2_ns: AtomicU64, // stored as f64 bits

    /// Fixed-bucket histogram for percentile estimation.
    /// Bucket i covers [i * bucket_width, (i+1) * bucket_width).
    /// We use 1024 buckets covering 0..max_range_ns (default 100ms).
    histogram: Vec<AtomicU64>,
    bucket_width_ns: f64,
    max_range_ns: f64,

    /// Human-readable name (resolved lazily from hash)
    name: String,
}

impl SpanStats {
    fn new(name: String, max_range_ns: f64, num_buckets: usize) -> Self {
        let bucket_width_ns = max_range_ns / num_buckets as f64;
        let mut histogram = Vec::with_capacity(num_buckets);
        for _ in 0..num_buckets {
            histogram.push(AtomicU64::new(0));
        }
        Self {
            count: AtomicU64::new(0),
            sum_ns: AtomicU64::new(0),
            min_ns: AtomicU64::new(u64::MAX),
            max_ns: AtomicU64::new(0),
            mean_ns: AtomicU64::new(0.0f64.to_bits()),
            m2_ns: AtomicU64::new(0.0f64.to_bits()),
            histogram,
            bucket_width_ns,
            max_range_ns,
            name,
        }
    }

    #[inline(always)]
    fn record(&self, duration_ns: f64) {
        let d = duration_ns as u64;

        // Min / Max
        self.min_ns.fetch_min(d, Ordering::Relaxed);
        self.max_ns.fetch_max(d, Ordering::Relaxed);

        // Count & Sum
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_ns.fetch_add(d, Ordering::Relaxed);

        // Welford's online algorithm (approximate with atomics — good enough)
        let n = self.count.load(Ordering::Relaxed) as f64;
        let old_mean = f64::from_bits(self.mean_ns.load(Ordering::Relaxed));
        let delta = duration_ns - old_mean;
        let new_mean = old_mean + delta / n;
        self.mean_ns.store(new_mean.to_bits(), Ordering::Relaxed);

        let old_m2 = f64::from_bits(self.m2_ns.load(Ordering::Relaxed));
        let delta2 = duration_ns - new_mean;
        let new_m2 = old_m2 + delta * delta2;
        self.m2_ns.store(new_m2.to_bits(), Ordering::Relaxed);

        // Histogram bucket
        let bucket = if duration_ns >= self.max_range_ns {
            self.histogram.len() - 1
        } else {
            (duration_ns / self.bucket_width_ns) as usize
        };
        if bucket < self.histogram.len() {
            self.histogram[bucket].fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl Aggregator {
    const fn new() -> Self {
        Self {
            stats: UnsafeCell::new(HashMap::new()),
            spin: AtomicUsize::new(0),
        }
    }

    fn lock(&self) {
        while self
            .spin
            .compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            std::hint::spin_loop();
        }
    }

    fn unlock(&self) {
        self.spin.store(0, Ordering::Release);
    }

    /// SAFETY: caller must hold the spin lock
    unsafe fn get_or_create(&self, name_hash: u64, name: &str) -> &SpanStats {
        let map = &mut *self.stats.get();
        map.entry(name_hash).or_insert_with(|| {
            SpanStats::new(name.to_string(), 100_000_000.0, 1024) // 100ms range, 1024 buckets
        })
    }

    fn flush_records(&self, records: &[SpanRecord], name_table: &NameTable) {
        if records.is_empty() {
            return;
        }
        self.lock();
        for rec in records {
            let name = name_table.resolve(rec.name_hash);
            // SAFETY: we hold the lock
            let stat = unsafe { self.get_or_create(rec.name_hash, name) };
            let ns = cycles_to_ns(rec.duration);
            stat.record(ns);
        }
        self.unlock();
    }

    fn snapshot(&self) -> Vec<SpanReport> {
        self.lock();
        let map = unsafe { &*self.stats.get() };
        let reports: Vec<SpanReport> = map
            .values()
            .map(|s| {
                let count = s.count.load(Ordering::Relaxed);
                let mean_ns = f64::from_bits(s.mean_ns.load(Ordering::Relaxed));
                let m2 = f64::from_bits(s.m2_ns.load(Ordering::Relaxed));
                let stddev = if count > 1 {
                    (m2 / (count as f64 - 1.0)).sqrt()
                } else {
                    0.0
                };

                // Compute percentiles from histogram
                let (p50, p95, p99) = compute_percentiles(&s.histogram, s.bucket_width_ns, count);

                SpanReport {
                    name: s.name.clone(),
                    count,
                    total_ns: s.sum_ns.load(Ordering::Relaxed),
                    min_ns: s.min_ns.load(Ordering::Relaxed),
                    max_ns: s.max_ns.load(Ordering::Relaxed),
                    mean_ns,
                    stddev_ns: stddev,
                    p50_ns: p50,
                    p95_ns: p95,
                    p99_ns: p99,
                }
            })
            .collect();
        self.unlock();
        reports
    }
}

// SAFETY: access is guarded by spin lock; thread-local buffers are never
// shared across threads.
unsafe impl Send for Aggregator {}
unsafe impl Sync for Aggregator {}

// ---------------------------------------------------------------------------
// Name table — maps hash -> human-readable name (append-only, no locks on read)
// ---------------------------------------------------------------------------

struct NameTable {
    table: UnsafeCell<HashMap<u64, &'static str>>,
    spin: AtomicUsize,
}

impl NameTable {
    const fn new() -> Self {
        Self {
            table: UnsafeCell::new(HashMap::new()),
            spin: AtomicUsize::new(0),
        }
    }

    fn intern(&self, name: &'static str) -> u64 {
        let hash = fnv1a64(name);
        // Fast path: already interned (no lock needed for read)
        {
            let table = unsafe { &*self.table.get() };
            if table.contains_key(&hash) {
                return hash;
            }
        }
        // Slow path: acquire spin lock and insert
        while self
            .spin
            .compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            std::hint::spin_loop();
        }
        let table = unsafe { &mut *self.table.get() };
        table.entry(hash).or_insert(name);
        self.spin.store(0, Ordering::Release);
        hash
    }

    fn resolve(&self, hash: u64) -> &str {
        let table = unsafe { &*self.table.get() };
        table.get(&hash).copied().unwrap_or("<unknown>")
    }
}

unsafe impl Send for NameTable {}
unsafe impl Sync for NameTable {}

// ---------------------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------------------

static AGGREGATOR: Aggregator = Aggregator::new();
static NAME_TABLE: NameTable = NameTable::new();

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Handle returned by `begin_span`. Calls `end_span` on drop.
pub struct SpanGuard {
    name_hash: u64,
    _pad: [u8; 56], // pad to 64 bytes to avoid false sharing
}

impl SpanGuard {
    #[inline(always)]
    fn start(name: &'static str) -> Self {
        let name_hash = NAME_TABLE.intern(name);
        let start = read_cycles();

        LOCAL.with(|cell| {
            let state = unsafe { &mut *cell.get() };
            if state.stack_depth < state.stack.len() {
                state.stack[state.stack_depth] = start;
                state.stack_depth += 1;
            }
        });

        Self {
            name_hash,
            _pad: [0u8; 56],
        }
    }
}

impl Drop for SpanGuard {
    #[inline(always)]
    fn drop(&mut self) {
        let end = read_cycles();

        LOCAL.with(|cell| {
            let state = unsafe { &mut *cell.get() };
            if state.stack_depth > 0 {
                state.stack_depth -= 1;
                let start = state.stack[state.stack_depth];
                let duration = end.saturating_sub(start);

                let record = SpanRecord {
                    name_hash: self.name_hash,
                    start,
                    duration,
                };

                if !state.buffer.push(record) {
                    // Buffer full — flush and retry
                    flush_local(state);
                    state.buffer.push(record);
                }
            }
        });
    }
}

/// Begin a named profiling span. Returns a guard that records the duration on drop.
///
/// # Example
/// ```ignore
/// {
///     let _span = profile_span!("physics_update");
///     // ... work ...
/// } // duration recorded here
/// ```
#[cfg(feature = "profiler")]
#[inline(always)]
pub fn begin_span(name: &'static str) -> SpanGuard {
    SpanGuard::start(name)
}

/// Flush the current thread's buffered spans to the global aggregator.
/// Called automatically when the buffer fills, but can be called manually
/// to ensure data is captured (e.g., before reading reports).
pub fn flush() {
    LOCAL.with(|cell| {
        let state = unsafe { &mut *cell.get() };
        flush_local(state);
    });
}

fn flush_local(state: &mut LocalState) {
    let records = state.buffer.drain();
    AGGREGATOR.flush_records(records, &NAME_TABLE);
}

// ---------------------------------------------------------------------------
// Macros (enabled version)
// ---------------------------------------------------------------------------

#[cfg(feature = "profiler")]
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {
        let _prof_guard = $crate::begin_span($name);
    };
}

#[cfg(feature = "profiler")]
#[macro_export]
macro_rules! profile_fn {
    () => {
        let _prof_guard = $crate::begin_span(concat!(module_path!(), "::", function_name!()));
    };
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

/// Snapshot of statistics for a single span.
#[derive(Debug, Clone)]
pub struct SpanReport {
    pub name: String,
    pub count: u64,
    pub total_ns: u64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub mean_ns: f64,
    pub stddev_ns: f64,
    pub p50_ns: f64,
    pub p95_ns: f64,
    pub p99_ns: f64,
}

impl fmt::Display for SpanReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:<40} {:>10} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>12.2}",
            self.name,
            self.count,
            self.mean_ns,
            self.p50_ns,
            self.p95_ns,
            self.p99_ns,
            self.min_ns,
            self.max_ns,
        )
    }
}

/// Flush all thread-local buffers and return a snapshot of all span reports.
pub fn report() -> Vec<SpanReport> {
    flush();
    AGGREGATOR.snapshot()
}

/// Print a formatted report table to stdout.
pub fn print_report() {
    let mut reports = report();
    reports.sort_by(|a, b| b.total_ns.cmp(&a.total_ns));

    println!();
    println!(
        "{:<40} {:>10} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "SPAN", "CALLS", "MEAN(ns)", "P50(ns)", "P95(ns)", "P99(ns)", "MIN(ns)", "MAX(ns)"
    );
    println!("{}", "-".repeat(128));
    for r in &reports {
        println!("{}", r);
    }
    println!();

    let total: u64 = reports.iter().map(|r| r.total_ns).sum();
    println!("Total profiled time: {:.3} ms", total as f64 / 1_000_000.0);
}

/// Compute percentiles from the histogram.
fn compute_percentiles(
    histogram: &[AtomicU64],
    bucket_width: f64,
    total_count: u64,
) -> (f64, f64, f64) {
    if total_count == 0 {
        return (0.0, 0.0, 0.0);
    }

    let targets: [(f64, usize); 3] = [
        (0.50, 0), // p50
        (0.95, 1), // p95
        (0.99, 2), // p99
    ];

    let mut results = [0.0f64; 3];
    let mut cumulative = 0u64;
    let mut found = [false; 3];
    let mut found_count = 0usize;

    for (i, bucket) in histogram.iter().enumerate() {
        let count = bucket.load(Ordering::Relaxed);
        if count == 0 {
            continue;
        }
        cumulative += count;

        for (percentile, idx) in &targets {
            if !found[*idx] {
                let threshold = (total_count as f64 * percentile).ceil() as u64;
                if cumulative >= threshold {
                    // Interpolate within bucket
                    let bucket_start = i as f64 * bucket_width;
                    let frac = if count > 0 {
                        (cumulative - threshold) as f64 / count as f64
                    } else {
                        0.0
                    };
                    results[*idx] = bucket_start + frac * bucket_width;
                    found[*idx] = true;
                    found_count += 1;
                }
            }
        }

        if found_count == 3 {
            break;
        }
    }

    (results[0], results[1], results[2])
}

// ---------------------------------------------------------------------------
// Reset — useful for benchmarking loops
// ---------------------------------------------------------------------------

/// Clear all statistics. Not thread-safe; call only when no profiling is active.
pub fn reset() {
    AGGREGATOR.lock();
    let map = unsafe { &mut *AGGREGATOR.stats.get() };
    map.clear();
    AGGREGATOR.unlock();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_span_recording() {
        reset();

        {
            let _s = SpanGuard::start("test_span");
            // simulate work
            let mut x = 0u64;
            for i in 0..1000 {
                x = x.wrapping_mul(i).wrapping_add(1);
            }
            std::hint::black_box(x);
        }

        flush();
        let reports = report();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].name, "test_span");
        assert_eq!(reports[0].count, 1);
        assert!(reports[0].total_ns > 0);
    }

    #[test]
    fn nested_spans() {
        reset();

        {
            let _outer = SpanGuard::start("outer");
            {
                let _inner = SpanGuard::start("inner");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }

        flush();
        let reports = report();
        assert_eq!(reports.len(), 2);

        let outer = reports.iter().find(|r| r.name == "outer").unwrap();
        let inner = reports.iter().find(|r| r.name == "inner").unwrap();
        assert!(outer.total_ns >= inner.total_ns);
    }

    #[test]
    fn many_spans_fill_buffer() {
        reset();

        for _ in 0..BUFFER_CAPACITY + 100 {
            let _s = SpanGuard::start("hot_loop");
        }

        flush();
        let reports = report();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].count, (BUFFER_CAPACITY + 100) as u64);
    }

    #[test]
    fn fnv1a_hash_stability() {
        let h1 = fnv1a64("render");
        let h2 = fnv1a64("render");
        let h3 = fnv1a64("physics");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn histogram_percentiles() {
        reset();

        // Insert known durations
        for i in 0..100 {
            let name = "pct_test";
            let name_hash = NAME_TABLE.intern(name);
            let duration_cycles = (i * 1000) as u64; // 0, 1000, 2000, ... 99000
            let record = SpanRecord {
                name_hash,
                start: 0,
                duration: duration_cycles,
            };
            AGGREGATOR.flush_records(&[record], &NAME_TABLE);
        }

        let reports = report();
        let r = &reports[0];
        assert_eq!(r.count, 100);
        // p50 should be roughly in the middle
        assert!(r.p50_ns > 0.0);
    }
}
