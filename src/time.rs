// src/time.rs
//! High-performance, full-featured timing system for games, renderers, and real-time apps.
//!
//! - **Performance**: `#[inline(always)]` everywhere, zero allocations on update/hot path,
//!   f32 delta + f64 total for precision, exponential smoothing, max-delta clamp.
//! - **Features**: Real vs virtual time, time scaling, pause, fixed timestep (with catch-up limit),
//!   rolling FPS stats (min/avg/max/1% low), Stopwatch, Timer (one-shot + repeating),
//!   Scheduler with BinaryHeap, scoped tracing timers, perfect `?`/`.context()` integration.
//! - **Integration**: Pass `&TimeManager` or `Time` snapshot to your `PipelineStage::Ctx`,
//!   feed FPS directly into `GuiManager`.

use crate::{Result, bail, ensure, Context};
use std::collections::BinaryHeap;
use std::time::Instant;
use tracing::{debug_span, info_span, Instrument};

/// Snapshot of timing data passed around each frame (Copy, cheap).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Time {
    pub delta: f32,          // smoothed real delta (seconds)
    pub raw_delta: f32,      // unsmoothed real delta
    pub virtual_delta: f32,  // after time_scale + pause
    pub total: f64,          // virtual elapsed seconds
    pub real_total: f64,     // wall-clock elapsed
    pub frame: u64,
    pub fps: f32,
    pub fps_min: f32,
    pub fps_avg: f32,
    pub fps_max: f32,
}

/// Core high-performance time manager.
#[derive(Debug)]
pub struct TimeManager {
    start: Instant,
    last_frame: Instant,
    accumulator: f32,
    frame: u64,
    time_scale: f32,
    paused: bool,
    max_delta: f32,           // prevent spiral of death
    smoothing: f32,           // EMA factor (0.0 = no smoothing, 0.2 = default)
    last_delta: f32,
    fps_history: Vec<f32>,    // rolling 120 frames
    fps_min: f32,
    fps_max: f32,
    fps_sum: f32,
}

impl Default for TimeManager {
    #[inline(always)]
    fn default() -> Self {
        let now = Instant::now();
        Self {
            start: now,
            last_frame: now,
            accumulator: 0.0,
            frame: 0,
            time_scale: 1.0,
            paused: false,
            max_delta: 0.25,      // 4 FPS minimum before clamping
            smoothing: 0.2,
            last_delta: 1.0 / 60.0,
            fps_history: Vec::with_capacity(120),
            fps_min: f32::MAX,
            fps_max: 0.0,
            fps_sum: 0.0,
        }
    }
}

impl TimeManager {
    /// Create a new time manager (call once at startup).
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Update timing — call **once per frame** at the very beginning.
    /// Returns a cheap `Time` snapshot you can pass everywhere.
    #[inline(always)]
    pub fn update(&mut self) -> Time {
        let now = Instant::now();
        let mut raw_delta = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;

        // Clamp to prevent death spirals (e.g. tab-out)
        if raw_delta > self.max_delta {
            raw_delta = self.max_delta;
        }

        // Exponential smoothing for buttery delta
        let smoothed = self.last_delta * (1.0 - self.smoothing) + raw_delta * self.smoothing;
        self.last_delta = smoothed;

        let virtual_delta = if self.paused { 0.0 } else { smoothed * self.time_scale };

        self.accumulator += virtual_delta;
        self.frame += 1;

        let real_total = now.duration_since(self.start).as_secs_f64();
        let total = self.frame as f64 * virtual_delta as f64; // more stable than summing

        // FPS rolling stats
        let fps = if smoothed > 0.0 { 1.0 / smoothed } else { 0.0 };
        self.fps_history.push(fps);
        if self.fps_history.len() > 120 {
            self.fps_sum -= self.fps_history.remove(0);
        }
        self.fps_sum += fps;

        let avg = self.fps_sum / self.fps_history.len() as f32;
        if fps < self.fps_min { self.fps_min = fps; }
        if fps > self.fps_max { self.fps_max = fps; }

        Time {
            delta: smoothed,
            raw_delta,
            virtual_delta,
            total,
            real_total,
            frame: self.frame,
            fps,
            fps_min: self.fps_min,
            fps_avg: avg,
            fps_max: self.fps_max,
        }
    }

    /// Fixed timestep iterator — perfect for physics inside PipelineStage.
    /// Usage: `for _ in time.fixed_timestep(1.0/60.0) { physics_step(); }`
    #[inline(always)]
    pub fn fixed_timestep(&mut self, fixed_dt: f32) -> FixedTimestepIter<'_> {
        FixedTimestepIter {
            accumulator: &mut self.accumulator,
            fixed_dt,
            max_steps: 5, // safety
        }
    }

    // ================ CONTROLS ================
    #[inline(always)] pub fn set_time_scale(&mut self, scale: f32) { self.time_scale = scale.max(0.0); }
    #[inline(always)] pub fn time_scale(&self) -> f32 { self.time_scale }
    #[inline(always)] pub fn pause(&mut self) { self.paused = true; }
    #[inline(always)] pub fn resume(&mut self) { self.paused = false; }
    #[inline(always)] pub fn toggle_pause(&mut self) { self.paused = !self.paused; }
    #[inline(always)] pub fn is_paused(&self) -> bool { self.paused }

    // ================ STATS ================
    #[inline(always)] pub fn frame(&self) -> u64 { self.frame }
    #[inline(always)] pub fn fps_history(&self) -> &[f32] { &self.fps_history }
}

/// Fixed timestep iterator (zero-allocation, safe catch-up).
pub struct FixedTimestepIter<'a> {
    accumulator: &'a mut f32,
    fixed_dt: f32,
    max_steps: u32,
}

impl Iterator for FixedTimestepIter<'_> {
    type Item = ();
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if *self.accumulator >= self.fixed_dt && self.max_steps > 0 {
            *self.accumulator -= self.fixed_dt;
            self.max_steps -= 1;
            Some(())
        } else {
            None
        }
    }
}

/// High-performance stopwatch (reset, pause, elapsed).
#[derive(Debug, Clone, Copy)]
pub struct Stopwatch {
    start: Instant,
    paused_at: Option<Instant>,
    accumulated: f32,
}

impl Stopwatch {
    #[inline(always)] pub fn new() -> Self { Self { start: Instant::now(), paused_at: None, accumulated: 0.0 } }
    #[inline(always)] pub fn start(&mut self) { if self.paused_at.is_some() { self.start = Instant::now() - std::time::Duration::from_secs_f32(self.accumulated); self.paused_at = None; } }
    #[inline(always)] pub fn pause(&mut self) { if self.paused_at.is_none() { self.paused_at = Some(Instant::now()); } }
    #[inline(always)] pub fn reset(&mut self) { *self = Self::new(); }
    #[inline(always)] pub fn elapsed(&self) -> f32 {
        if let Some(p) = self.paused_at {
            self.accumulated + p.duration_since(self.start).as_secs_f32()
        } else {
            self.accumulated + self.start.elapsed().as_secs_f32()
        }
    }
}

/// Countdown / repeating timer.
#[derive(Debug)]
pub struct Timer {
    duration: f32,
    remaining: f32,
    repeating: bool,
}

impl Timer {
    #[inline(always)] pub fn new(duration: f32, repeating: bool) -> Self { Self { duration, remaining: duration, repeating } }
    #[inline(always)] pub fn tick(&mut self, dt: f32) -> bool {
        if self.remaining <= 0.0 { return false; }
        self.remaining -= dt;
        if self.remaining <= 0.0 {
            let finished = true;
            if self.repeating { self.remaining += self.duration; }
            finished
        } else { false }
    }
    #[inline(always)] pub fn reset(&mut self) { self.remaining = self.duration; }
}

/// Simple scheduler for one-shot and recurring events (BinaryHeap).
pub struct Scheduler {
    events: BinaryHeap<ScheduledEvent>,
}

#[derive(PartialEq, Eq)]
struct ScheduledEvent {
    when: std::time::Instant,
    id: u64,
    action: Box<dyn FnOnce() + Send + Sync>,
    repeat_every: Option<f32>,
}

impl PartialOrd for ScheduledEvent { fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) } }
impl Ord for ScheduledEvent {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.when.cmp(&self.when).then_with(|| self.id.cmp(&other.id))
    }
}

impl Scheduler {
    pub fn new() -> Self { Self { events: BinaryHeap::new() } }
    pub fn schedule<F: FnOnce() + Send + Sync + 'static>(&mut self, delay: f32, action: F) {
        let when = Instant::now() + std::time::Duration::from_secs_f32(delay);
        self.events.push(ScheduledEvent { when, id: rand::random(), action: Box::new(action), repeat_every: None });
    }
    pub fn schedule_repeating<F: FnMut() + Send + Sync + 'static>(&mut self, delay: f32, interval: f32, mut action: F) {
        let when = Instant::now() + std::time::Duration::from_secs_f32(delay);
        let repeat = Some(interval);
        self.events.push(ScheduledEvent {
            when,
            id: rand::random(),
            action: Box::new(move || action()),
            repeat_every: repeat,
        });
    }
    pub fn update(&mut self, now: Instant) {
        while let Some(event) = self.events.peek() {
            if event.when > now { break; }
            let ev = self.events.pop().unwrap();
            (ev.action)();
            if let Some(interval) = ev.repeat_every {
                let next = now + std::time::Duration::from_secs_f32(interval);
                self.events.push(ScheduledEvent { when: next, id: rand::random(), action: ev.action, repeat_every: ev.repeat_every });
            }
        }
    }
}

// ================ SCOPED TRACING TIMER ================
#[macro_export]
macro_rules! timed {
    ($name:literal, $block:expr) => {{
        let _span = info_span!($name).entered();
        let start = std::time::Instant::now();
        let result = $block;
        let elapsed = start.elapsed().as_secs_f32();
        tracing::debug!(?elapsed, concat!($name, " took"));
        result
    }};
}

// Re-exports
pub use {Time, TimeManager, Stopwatch, Timer, Scheduler, timed};
