// src/profiler_ui.rs - On-screen profiler graphs
use std::collections::VecDeque;
use parking_lot::RwLock;

pub struct ProfilerUI {
    frames: RwLock<VecDeque<FrameData>>,
    max_frames: usize,
    visible: bool,
}

#[derive(Clone)] struct FrameData { frame_number: u64, frame_time_ms: f32, systems: Vec<SystemProfile> }
#[derive(Clone)] struct SystemProfile { name: String, time_ms: f32, percentage: f32 }

impl ProfilerUI {
    pub fn new(max_frames: usize) -> Self { Self { frames: RwLock::new(VecDeque::with_capacity(max_frames)), max_frames, visible: false } }
    pub fn record_frame(&self, data: FrameData) {
        let mut frames = self.frames.write();
        if frames.len() >= self.max_frames { frames.pop_front(); }
        frames.push_back(data);
    }
    pub fn toggle(&mut self) { self.visible = !self.visible; }
}
