use std::time::{Duration, Instant};

pub struct FpsCounter {
    frame_times: [f32; 128], // store last N frame times (ms)
    index: usize,
    last_instant: Instant,
    frames: usize,
}

impl FpsCounter {
    pub fn new() -> Self {
        Self {
            frame_times: [0.0; 128],
            index: 0,
            last_instant: Instant::now(),
            frames: 0,
        }
    }

    /// Call at the start of a frame (or end) to record timing.
    pub fn tick(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_instant);
        self.last_instant = now;
        let ms = dt.as_secs_f32() * 1000.0;
        self.frame_times[self.index] = ms;
        self.index = (self.index + 1) % self.frame_times.len();
        self.frames += 1;
    }

    /// Returns averaged FPS and frame time in ms over the buffer.
    pub fn averaged(&self) -> (f32, f32) {
        let mut sum = 0.0f32;
        let mut count = 0;
        for &v in &self.frame_times {
            if v > 0.0 {
                sum += v;
                count += 1;
            }
        }
        if count == 0 {
            return (0.0, 0.0);
        }
        let avg_ms = sum / count as f32;
        (1000.0 / avg_ms, avg_ms)
    }

    /// Print a periodic log every N frames (useful while running).
    pub fn log_every(&self, every: usize) {
        if self.frames % every == 0 {
            let (fps, ms) = self.averaged();
            println!("FPS: {:.1}, Frame time: {:.3} ms", fps, ms);
        }
    }
}
