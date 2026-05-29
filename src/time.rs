#![allow(unused)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::{f32, f64, time::Duration};

#[cfg(all(target_arch = "wasm32", feature = "web_sys"))]
use web_sys::window;

/// High resolution timestamp in nanoseconds
pub type Timestamp = u64;

/// Default fixed timestep: 60 ticks per second
pub const DEFAULT_FIXED_TIMESTEP: Duration = Duration::from_nanos(16666667);
/// Default maximum delta time to prevent spiral of death
pub const DEFAULT_MAX_DELTA_TIME: Duration = Duration::from_millis(100);
/// Threshold where we assume the process was suspended
const SUSPEND_THRESHOLD: Duration = Duration::from_secs(2);

#[inline(always)]
pub fn now() -> Timestamp {
    #[cfg(windows)]
    {
        use core::arch::x86_64::_mm_lfence;
        extern "system" {
            fn QueryPerformanceCounter(lpPerformanceCount: *mut i64) -> i32;
        }
        static mut FREQ: i64 = 0;

        // SAFETY: Race safe, idempotent
        unsafe {
            if FREQ == 0 {
                extern "system" {
                    fn QueryPerformanceFrequency(lpFrequency: *mut i64) -> i32;
                }
                QueryPerformanceFrequency(&mut FREQ);
            }

            let mut t = 0i64;
            _mm_lfence();
            QueryPerformanceCounter(&mut t);
            _mm_lfence();

            ((t as u128) * 1_000_000_000 / FREQ as u128) as u64
        }
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd", target_os = "openbsd"))]
    {
        use core::mem::MaybeUninit;
        libc::timespec;
        const CLOCK_MONOTONIC_RAW: libc::clockid_t = 4;

        let mut ts = MaybeUninit::uninit();
        unsafe {
            libc::clock_gettime(CLOCK_MONOTONIC_RAW, ts.as_mut_ptr());
            let ts = ts.assume_init();
            (ts.tv_sec as u64) * 1_000_000_000 + ts.tv_nsec as u64
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        extern "C" {
            fn mach_absolute_time() -> u64;
            fn mach_timebase_info(info: *mut u32) -> i32;
        }

        static mut NUM: u32 = 0;
        static mut DEN: u32 = 0;

        unsafe {
            if NUM == 0 {
                mach_timebase_info(&mut NUM);
            }
            mach_absolute_time() * NUM as u64 / DEN as u64
        }
    }

    #[cfg(all(target_arch = "wasm32", feature = "web_sys"))]
    {
        window()
            .unwrap()
            .performance()
            .unwrap()
            .now() * 1000000.0 as u64
    }

    #[cfg(feature = "std")]
    {
        use std::time::Instant;
        static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        START.get_or_init(Instant::now).elapsed().as_nanos() as u64
    }
}

/// Master time source for the entire engine
#[derive(Debug, Clone)]
pub struct Time {
    last_timestamp: Timestamp,

    /// Real time, always advances at normal speed
    real_time: Timestamp,
    real_delta: Timestamp,

    /// Game time, affected by pause and time scale
    game_time: Timestamp,
    game_delta: Timestamp,

    fixed_timestep: Timestamp,
    max_delta_time: Timestamp,

    accumulator: Timestamp,
    alpha: f32,

    timescale: f32,
    paused: bool,

    smoothed_fps: f32,
    smoothed_frame_time: f32,

    frame_count: u64,
    fixed_tick_count: u64,
}

impl Default for Time {
    fn default() -> Self {
        Self::new()
    }
}

impl Time {
    /// Create a new time instance
    pub fn new() -> Self {
        let now = now();
        Self {
            last_timestamp: now,
            real_time: 0,
            real_delta: 0,
            game_time: 0,
            game_delta: 0,
            fixed_timestep: DEFAULT_FIXED_TIMESTEP.as_nanos() as u64,
            max_delta_time: DEFAULT_MAX_DELTA_TIME.as_nanos() as u64,
            accumulator: 0,
            alpha: 1.0,
            timescale: 1.0,
            paused: false,
            smoothed_fps: 60.0,
            smoothed_frame_time: 16.666,
            frame_count: 0,
            fixed_tick_count: 0,
        }
    }

    /// Call this EXACTLY ONCE at the very start of every frame
    #[inline]
    pub fn start_of_frame(&mut self) {
        let now = now();
        let mut raw_delta = now.saturating_sub(self.last_timestamp);
        self.last_timestamp = now;

        // Protect against backwards time
        if raw_delta == 0 {
            raw_delta = 1;
        }

        // Protect against suspend / time jumps
        if raw_delta > SUSPEND_THRESHOLD.as_nanos() as u64 {
            raw_delta = self.max_delta_time;
        }

        // Clamp delta time to prevent spiral of death
        raw_delta = raw_delta.min(self.max_delta_time);

        self.real_delta = raw_delta;
        self.real_time += raw_delta;

        if !self.paused {
            self.game_delta = ((raw_delta as f64) * self.timescale as f64) as u64;
            self.game_time += self.game_delta;
            self.accumulator += self.game_delta;
        } else {
            self.game_delta = 0;
        }

        let dt = raw_delta as f64 / 1e9;
        let ewma = f64::exp(-dt / 0.5);
        let fps = 1.0 / dt;

        self.smoothed_fps = (self.smoothed_fps as f64 * ewma + fps * (1.0 - ewma)) as f32;
        self.smoothed_frame_time = (self.smoothed_frame_time as f64 * ewma + dt * 1000.0 * (1.0 - ewma)) as f32;

        self.frame_count += 1;
    }

    /// Call this in a loop at the start of your frame.
    /// Returns true if you should run another fixed timestep tick.
    #[inline]
    pub fn tick_fixed(&mut self) -> bool {
        if self.accumulator >= self.fixed_timestep {
            self.accumulator -= self.fixed_timestep;
            self.fixed_tick_count += 1;
            self.alpha = self.accumulator as f32 / self.fixed_timestep as f32;
            true
        } else {
            false
        }
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    /// Delta time for this frame, in seconds. Affected by timescale and pause.
    #[inline(always)]
    pub fn delta(&self) -> f32 {
        self.game_delta as f32 / 1e9
    }

    /// Total elapsed game time in seconds. Affected by timescale and pause.
    #[inline(always)]
    pub fn time(&self) -> f64 {
        self.game_time as f64 / 1e9
    }

    /// Real delta time, not affected by pause or timescale. Use for UI, profilers etc.
    #[inline(always)]
    pub fn real_delta(&self) -> f32 {
        self.real_delta as f32 / 1e9
    }

    /// Total real elapsed time. Never paused or scaled.
    #[inline(always)]
    pub fn real_time(&self) -> f64 {
        self.real_time as f64 / 1e9
    }

    /// Interpolation factor between the last and next fixed tick. 0.0 .. 1.0
    #[inline(always)]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    #[inline(always)]
    pub fn fps(&self) -> f32 {
        self.smoothed_fps
    }

    #[inline(always)]
    pub fn frame_time_ms(&self) -> f32 {
        self.smoothed_frame_time
    }

    #[inline(always)]
    pub fn frame_id(&self) -> u64 {
        self.frame_count
    }

    #[inline(always)]
    pub fn fixed_tick_count(&self) -> u64 {
        self.fixed_tick_count
    }

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    #[inline]
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    #[inline]
    pub fn set_timescale(&mut self, timescale: f32) {
        self.timescale = timescale.max(0.0);
    }

    #[inline]
    pub fn set_fixed_timestep(&mut self, timestep: Duration) {
        self.fixed_timestep = timestep.as_nanos() as u64;
    }

    #[inline]
    pub fn set_max_delta_time(&mut self, max_delta: Duration) {
        self.max_delta_time = max_delta.as_nanos() as u64;
    }

    /// Fully reset all time state
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Ultra low overhead scoped benchmark timer
pub struct ScopedTimer {
    start: Timestamp,
    name: &'static str,
}

impl ScopedTimer {
    #[inline]
    pub fn new(name: &'static str) -> Self {
        Self {
            start: now(),
            name,
        }
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        let ns = now() - self.start;
        println!("[{}] {:.2}µs", self.name, ns as f32 / 1000.0);
    }
}
