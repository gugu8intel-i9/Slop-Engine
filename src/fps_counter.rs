// fps_counter.rs

use web_sys::Performance;

/// A browser‑safe FPS counter using `performance.now()`.
pub struct FpsCounter {
    perf: Performance,
    last_time: f64,
    frame_count: u32,
    accum_time: f64,
    fps: f32,
    update_interval: f64,
    smoothing: f32,
}

impl FpsCounter {
    /// Create a new FPS counter.
    /// `update_interval_secs` – how often to update FPS (e.g. 0.5 or 1.0)
    /// `smoothing` – exponential smoothing factor [0.0, 1.0]
    pub fn new(update_interval_secs: f32, smoothing: f32) -> Self {
        let perf = web_sys::window()
            .expect("no window")
            .performance()
            .expect("performance API unavailable");

        let now = perf.now();

        Self {
            perf,
            last_time: now,
            frame_count: 0,
            accum_time: 0.0,
            fps: 0.0,
            update_interval: update_interval_secs.max(0.001) as f64 * 1000.0,
            smoothing: smoothing.clamp(0.0, 1.0),
        }
    }

    /// Default: update every 1s, light smoothing.
    pub fn default() -> Self {
        Self::new(1.0, 0.3)
    }

    /// Call once per frame.
    pub fn tick(&mut self) {
        let now = self.perf.now();
        let dt = now - self.last_time;
        self.last_time = now;

        self.frame_count += 1;
        self.accum_time += dt;

        if self.accum_time >= self.update_interval {
            let seconds = self.accum_time / 1000.0;
            let raw_fps = self.frame_count as f64 / seconds;

            if self.fps == 0.0 {
                self.fps = raw_fps as f32;
            } else {
                self.fps = self.fps * self.smoothing
                    + (raw_fps as f32) * (1.0 - self.smoothing);
            }

            self.frame_count = 0;
            self.accum_time = 0.0;
        }
    }

    /// Current FPS estimate.
    pub fn fps(&self) -> f32 {
        self.fps
    }
}
