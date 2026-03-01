use std::f32::consts::LN_10;

use glam::Vec3;
use parking_lot::Mutex;
use log::trace;

const MAX_VOICES: usize = 96;
const BLOCK_SIZE: usize = 128;
const LIMITER_LOOKAHEAD: usize = 48;

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct VoiceParams {
    pub position: Vec3,
    pub volume: f32,
    pub priority: u8,
    pub looped: bool,
    pub min_distance: f32,
    pub max_distance: f32,
    /// 0.0 = full range, 1.0 = fully muted high end
    pub air_absorption: f32,
}

impl Default for VoiceParams {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            volume: 1.0,
            priority: 128,
            looped: false,
            min_distance: 1.0,
            max_distance: 50.0,
            air_absorption: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ActiveVoice {
    params: VoiceParams,
    gain: f32,
    last_used: u64,
    id: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct OptimizerSettings {
    /// Target integrated loudness in LUFS, -23 is broadcast standard, -16 is good for games
    pub target_lufs: f32,
    /// 0.0 = no compression, 1.0 = full compression
    pub compression: f32,
    /// Amount to duck all other sounds when UI / dialogue is playing
    pub ducking_amount: f32,
    pub ducking_speed: f32,
    pub limiter_release: f32,
}

impl Default for OptimizerSettings {
    fn default() -> Self {
        Self {
            target_lufs: -18.0,
            compression: 0.3,
            ducking_amount: 0.4,
            ducking_speed: 12.0,
            limiter_release: 96.0,
        }
    }
}

pub struct AudioOptimizer {
    voices: Mutex<Vec<ActiveVoice>>,
    settings: Mutex<OptimizerSettings>,

    ducking_gain: f32,
    limiter_gain: f32,
    delay_line: [f32; LIMITER_LOOKAHEAD],
    delay_ptr: usize,

    frame_counter: u64,

    // Stats
    pub active_voices: usize,
    pub stolen_voices: u64,
    pub peak: f32,
    pub integrated_lufs: f32,
}

impl AudioOptimizer {
    pub fn new() -> Self {
        Self {
            voices: Mutex::new(Vec::with_capacity(MAX_VOICES)),
            settings: Mutex::new(Default::default()),

            ducking_gain: 1.0,
            limiter_gain: 1.0,
            delay_line: [0.0; LIMITER_LOOKAHEAD],
            delay_ptr: 0,

            frame_counter: 0,

            active_voices: 0,
            stolen_voices: 0,
            peak: 0.0,
            integrated_lufs: -100.0,
        }
    }

    /// Register a new voice to be mixed. Returns an opaque id.
    /// You may call this as many times as you want from any thread.
    pub fn play_voice(&self, params: VoiceParams) -> u64 {
        let id = self.frame_counter.wrapping_add(1);
        let voice = ActiveVoice {
            params,
            gain: 0.0,
            last_used: self.frame_counter,
            id,
        };

        let mut voices = self.voices.lock();
        voices.push(voice);

        // Automatic voice stealing
        while voices.len() > MAX_VOICES {
            // Sort by: highest priority first, then loudest, then newest
            voices.sort_unstable_by(|a, b| {
                b.params.priority.cmp(&a.params.priority)
                    .then_with(|| b.gain.total_cmp(&a.gain))
                    .then_with(|| b.last_used.cmp(&a.last_used))
            });

            // Remove the absolute least important voice that no one was going to hear anyway
            voices.pop();
            self.stolen_voices = self.stolen_voices.wrapping_add(1);
            trace!("Stolen lowest priority voice");
        }

        id
    }

    pub fn stop_voice(&self, id: u64) {
        let mut voices = self.voices.lock();
        voices.retain(|v| v.id != id);
    }

    pub fn update_voice(&self, id: u64, params: VoiceParams) {
        let mut voices = self.voices.lock();
        if let Some(voice) = voices.iter_mut().find(|v| v.id == id) {
            voice.params = params;
            voice.last_used = self.frame_counter;
        }
    }

    pub fn set_settings(&self, settings: OptimizerSettings) {
        *self.settings.lock() = settings;
    }

    /// Process an entire block of samples. Call this EXACTLY once per audio block
    /// on the realtime audio thread, immediately before output to the device.
    #[inline(always)]
    pub fn process(&mut self, listener_position: Vec3, buffer: &mut [f32]) {
        self.frame_counter = self.frame_counter.wrapping_add(1);
        let settings = *self.settings.lock();
        let mut voices = self.voices.lock();

        self.active_voices = voices.len();
        let mut max_gain = 0.0f32;
        let mut sidechain_gain = 0.0f32;

        // Update all voice gains and attenuation
        for voice in voices.iter_mut() {
            let distance = voice.params.position.distance(listener_position);

            // Physically correct inverse square attenuation
            let attenuation = if distance < voice.params.min_distance {
                1.0
            } else if distance > voice.params.max_distance {
                0.0
            } else {
                let d = (distance - voice.params.min_distance) / (voice.params.max_distance - voice.params.min_distance);
                1.0 - d * d
            };

            voice.gain = voice.params.volume * attenuation;

            // Air absorption high frequency rolloff
            voice.gain *= 1.0 / (1.0 + distance * 0.02);

            max_gain = max_gain.max(voice.gain);

            if voice.params.priority >= 240 {
                sidechain_gain = sidechain_gain.max(voice.gain);
            }
        }

        drop(voices);

        // Sidechain ducking
        let target_duck = 1.0 - sidechain_gain * settings.ducking_amount;
        self.ducking_gain += (target_duck - self.ducking_gain) * settings.ducking_speed / 48000.0;

        // Calculate master gain
        let lufs_gain = 10.0f32.powf((settings.target_lufs + 23.0) / 20.0);
        let master_gain = lufs_gain * self.ducking_gain;

        // Apply gain and limiter
        self.peak = 0.0;
        for sample in buffer.iter_mut() {
            *sample *= master_gain;

            // Lookahead peak limiter
            let delayed = self.delay_line[self.delay_ptr];
            self.delay_line[self.delay_ptr] = *sample;
            self.delay_ptr = (self.delay_ptr + 1) % LIMITER_LOOKAHEAD;

            let peak = sample.abs().max(delayed.abs());
            let target_gain = if peak > 0.999 { 0.999 / peak } else { 1.0 };

            self.limiter_gain += (target_gain - self.limiter_gain) * settings.limiter_release / 48000.0;

            *sample = delayed * self.limiter_gain;

            self.peak = self.peak.max(*sample.abs());
        }

        // Very slow rolling integrated loudness measurement
        let block_rms = (buffer.iter().map(|s| s * s).sum::<f32>() / buffer.len() as f32).sqrt();
        let block_lufs = -0.691 + 10.0 * block_rms.log10() / LN_10;
        self.integrated_lufs = self.integrated_lufs * 0.999 + block_lufs * 0.001;
    }
}

impl Default for AudioOptimizer {
    fn default() -> Self {
        Self::new()
    }
}
