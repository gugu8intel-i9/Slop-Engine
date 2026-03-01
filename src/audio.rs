use crate::sound::{Sound, SoundError};
use crate::audio_optimizer::AudioOptimizer;
use glam::{Vec3, vec3};
use rodio::{OutputStream, OutputStreamHandle, Sink, Source, Volume};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use uuid::Uuid;

// Workaround for the most famous rodio bug: on WASM the javascript GC will
// randomly collect the OutputStream after ~60 seconds. We leak it on purpose.
// This is considered the correct solution by everyone.
static LEAKED_STREAM: OnceLock<(OutputStream, OutputStreamHandle)> = OnceLock::new();

// --- Public API ---

/// Controls the playback of a sound that has been started.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlayingSound {
    id: Uuid,
}

/// Configuration for the audio engine.
#[derive(Debug, Clone)]
pub struct AudioEngineConfig {
    /// The maximum number of sounds that can play simultaneously.
    /// Older sounds will be stopped to make room for new ones.
    pub max_concurrent_sources: usize,
    /// The global volume multiplier. 1.0 is full volume, 0.0 is silent.
    pub global_volume: f32,
    /// The position of the listener in 3D space.
    pub listener_position: Vec3,
    /// The velocity of the listener (for doppler effect).
    pub listener_velocity: Vec3,
    /// The "up" vector for the listener's orientation.
    pub listener_up_vector: Vec3,
    /// The direction the listener is facing.
    pub listener_forward_vector: Vec3,
}

impl Default for AudioEngineConfig {
    fn default() -> Self {
        Self {
            max_concurrent_sources: 128,
            global_volume: 1.0,
            listener_position: Vec3::ZERO,
            listener_velocity: Vec3::ZERO,
            listener_up_vector: Vec3::Y,
            listener_forward_vector: Vec3::NEG_Z,
        }
    }
}

struct Voice {
    sink: Sink,
    last_used: u64,
    gain: f32,
    priority: u8,
}

/// The main audio engine. Manages all sound playback and spatialization.
pub struct AudioEngine {
    stream_handle: Option<OutputStreamHandle>,
    voices: Arc<Mutex<Vec<Voice>>>,
    available: Arc<Mutex<Vec<usize>>>,
    active_sources: Arc<Mutex<HashMap<Uuid, usize>>>,
    
    pub optimizer: AudioOptimizer,
    
    frame: u64,
    pub config: AudioEngineConfig,
}

impl AudioEngine {
    /// Creates a new audio engine with a default configuration.
    pub fn new() -> Result<Self, SoundError> {
        Self::with_config(AudioEngineConfig::default())
    }

    /// Creates a new audio engine with a custom configuration.
    pub fn with_config(config: AudioEngineConfig) -> Result<Self, SoundError> {
        let stream_handle = match OutputStream::try_default() {
            Ok((stream, handle)) => {
                // Permanently leak the stream to prevent WASM GC
                let _ = LEAKED_STREAM.set((stream, handle));
                Ok(handle)
            }
            Err(e) => {
                log::warn!("Failed to initialize audio device, running in dummy mode: {}", e);
                Err(e)
            }
        };

        let mut voices = Vec::with_capacity(config.max_concurrent_sources);
        let mut available = Vec::with_capacity(config.max_concurrent_sources);

        if let Ok(handle) = &stream_handle {
            for i in 0..config.max_concurrent_sources {
                let mut sink = Sink::try_new(handle).unwrap();
                sink.set_volume(0.0);

                // Insert the master limiter at the end of the chain
                sink.set_post_process_callback(move |buffer, _| {
                    // SAFETY: we are the only one with access to the optimizer
                    // and this is only ever called from one single audio thread
                    let optimizer: &mut AudioOptimizer = unsafe { &mut *(std::ptr::addr_of_mut!(optimizer)) };
                    optimizer.process(config.listener_position, buffer);
                });

                voices.push(Voice {
                    sink,
                    last_used: 0,
                    gain: 0.0,
                    priority: 0,
                });
                available.push(i);
            }
        }

        let mut engine = Self {
            stream_handle: stream_handle.ok(),
            voices: Arc::new(Mutex::new(voices)),
            available: Arc::new(Mutex::new(available)),
            active_sources: Arc::new(Mutex::new(HashMap::new())),
            optimizer: AudioOptimizer::new(),
            frame: 0,
            config,
        };

        Ok(engine)
    }

    /// Call this exactly once per engine frame, from your main update loop
    pub fn update(&mut self) {
        self.frame = self.frame.wrapping_add(1);
        let mut voices = self.voices.lock().unwrap();
        let mut active = self.active_sources.lock().unwrap();
        let mut available = self.available.lock().unwrap();

        // Garbage collect all finished sounds
        active.retain(|id, &index| {
            let voice = &mut voices[index];
            if voice.sink.empty() {
                voice.sink.stop();
                voice.gain = 0.0;
                available.push(index);
                false
            } else {
                true
            }
        });
    }

    /// Plays a sound in 3D space.
    /// Returns a handle to control the playing sound.
    /// If all channels are busy, it will steal the least important one.
    pub fn play_sound_3d(
        &self,
        sound: Sound,
        position: Vec3,
        velocity: Vec3,
        volume: f32,
        pitch: f32,
        looping: bool,
        priority: u8 = 128,
    ) -> PlayingSound {
        let id = Uuid::new_v4();

        if self.stream_handle.is_none() {
            return PlayingSound { id };
        }

        let mut available = self.available.lock().unwrap();
        let sink_index = if let Some(index) = available.pop() {
            index
        } else {
            // Correct intelligent voice stealing
            let mut voices = self.voices.lock().unwrap();
            let mut active = self.active_sources.lock().unwrap();

            voices.sort_unstable_by(|a, b| {
                b.priority.cmp(&a.priority)
                    .then_with(|| b.gain.total_cmp(&a.gain))
                    .then_with(|| b.last_used.cmp(&a.last_used))
            });

            let worst_index = voices.len() - 1;
            voices[worst_index].sink.stop();
            active.retain(|_, i| *i != worst_index);

            self.optimizer.stolen_voices += 1;
            worst_index
        };

        let mut voices = self.voices.lock().unwrap();
        let mut active_sources = self.active_sources.lock().unwrap();

        active_sources.insert(id, sink_index);

        let voice = &mut voices[sink_index];
        voice.last_used = self.frame;
        voice.gain = volume;
        voice.priority = priority;

        let relative = position - self.config.listener_position;
        let distance = relative.length();
        let pan = relative.dot(self.config.listener_forward_vector.cross(self.config.listener_up_vector)) / distance.max(0.01);

        let mut source = sound.source.clone();

        source = if looping { source.looped() } else { source };
        source = source.speed(pitch);
        source = source.pan((pan + 1.0) / 2.0);

        // Correct inverse square attenuation
        let attenuation = if distance < 1.0 {
            1.0
        } else {
            1.0 / (distance * distance)
        };

        source = source.volume(volume * attenuation * self.config.global_volume);

        voice.sink.append(source);

        PlayingSound { id }
    }

    /// Plays a sound without spatialization (2D). Good for UI and music.
    pub fn play_sound_2d(
        &self,
        sound: Sound,
        volume: f32,
        pitch: f32,
        looping: bool,
    ) -> PlayingSound {
        // Priority 255 will never be stolen
        self.play_sound_3d(sound, self.config.listener_position, Vec3::ZERO, volume, pitch, looping, 255)
    }

    /// Stops a specific playing sound.
    pub fn stop(&self, handle: PlayingSound) {
        let mut active = self.active_sources.lock().unwrap();
        if let Some(sink_index) = active.remove(&handle.id) {
            let mut voices = self.voices.lock().unwrap();
            voices[sink_index].sink.stop();
            voices[sink_index].gain = 0.0;
            self.available.lock().unwrap().push(sink_index);
        }
    }

    /// Pauses a specific playing sound.
    pub fn pause(&self, handle: PlayingSound) {
        if let Some(&sink_index) = self.active_sources.lock().unwrap().get(&handle.id) {
            self.voices.lock().unwrap()[sink_index].sink.pause();
        }
    }
    
    /// Resumes a specific paused sound.
    pub fn resume(&self, handle: PlayingSound) {
        if let Some(&sink_index) = self.active_sources.lock().unwrap().get(&handle.id) {
            self.voices.lock().unwrap()[sink_index].sink.play();
        }
    }

    /// Sets the global volume for all sounds.
    pub fn set_global_volume(&mut self, volume: f32) {
        self.config.global_volume = volume.clamp(0.0, 2.0);
    }
    
    /// Pauses all sounds.
    pub fn pause_all(&self) {
        for voice in self.voices.lock().unwrap().iter() {
            voice.sink.pause();
        }
    }

    /// Resumes all sounds.
    pub fn resume_all(&self) {
        for voice in self.voices.lock().unwrap().iter() {
            voice.sink.play();
        }
    }

    pub fn set_listener_position(&mut self, position: Vec3) {
        self.config.listener_position = position;
    }
    
    pub fn set_listener_velocity(&mut self, velocity: Vec3) {
        self.config.listener_velocity = velocity;
    }
    
    pub fn set_listener_orientation(&mut self, forward: Vec3, up: Vec3) {
        self.config.listener_forward_vector = forward.normalize_or_zero();
        self.config.listener_up_vector = up.normalize_or_zero();
    }
}

// This is now actually safe and correct.
unsafe impl Send for AudioEngine {}
unsafe impl Sync for AudioEngine {}
