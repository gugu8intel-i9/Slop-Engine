// src/audio.rs

use crate::sound::{Sound, SoundError};
use glam::{Vec3, vec3};
use rodio::{OutputStream, Sink, Source, SpatialSink, Volume};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use uuid::Uuid;

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
            max_concurrent_sources: 64,
            global_volume: 1.0,
            listener_position: Vec3::ZERO,
            listener_velocity: Vec3::ZERO,
            listener_up_vector: Vec3::Y,
            listener_forward_vector: Vec3::Z,
        }
    }
}

/// The main audio engine. Manages all sound playback and spatialization.
pub struct AudioEngine {
    // The main audio output stream. Must be kept alive.
    _stream: OutputStream,
    // The main sink for non-spatial sounds (e.g., UI, music).
    main_sink: Sink,
    // A pool of sinks for spatial (3D) sounds.
    spatial_sinks: Arc<Mutex<Vec<SpatialSink>>>,
    // A map to track which sink is playing which sound.
    active_sources: Arc<Mutex<HashMap<Uuid, usize>>>,
    // A map of available sink indices.
    available_sinks: Arc<Mutex<Vec<usize>>>,
    
    config: AudioEngineConfig,
}

impl AudioEngine {
    /// Creates a new audio engine with a default configuration.
    pub fn new() -> Result<Self, SoundError> {
        Self::with_config(AudioEngineConfig::default())
    }

    /// Creates a new audio engine with a custom configuration.
    pub fn with_config(config: AudioEngineConfig) -> Result<Self, SoundError> {
        let (_stream, stream_handle) = OutputStream::try_default()?;
        let main_sink = Sink::try_new(&stream_handle)?;
        
        let max_sources = config.max_concurrent_sources;
        let mut spatial_sinks = Vec::with_capacity(max_sources);
        let mut available_sinks = Vec::with_capacity(max_sources);

        for i in 0..max_sources {
            let spatial_sink = SpatialSink::try_new(&stream_handle)?;
            spatial_sinks.push(spatial_sink);
            available_sinks.push(i);
        }

        let engine = Self {
            _stream: _stream,
            main_sink,
            spatial_sinks: Arc::new(Mutex::new(spatial_sinks)),
            active_sources: Arc::new(Mutex::new(HashMap::new())),
            available_sinks: Arc::new(Mutex::new(available_sinks)),
            config,
        };

        // Set initial listener properties
        engine.update_listener_properties();

        Ok(engine)
    }

    /// Plays a sound in 3D space.
    /// Returns a handle to control the playing sound.
    /// If all channels are busy, it will steal the oldest one.
    pub fn play_sound_3d(
        &self,
        sound: Sound,
        position: Vec3,
        velocity: Vec3,
        volume: f32,
        pitch: f32,
        looping: bool,
    ) -> PlayingSound {
        let id = Uuid::new_v4();

        let mut available_sinks = self.available_sinks.lock().unwrap();
        let sink_index = if let Some(index) = available_sinks.pop() {
            index
        } else {
            // All sinks are busy, steal the oldest one.
            // A more sophisticated policy could be implemented here (e.g., quietest sound).
            let mut active = self.active_sources.lock().unwrap();
            if let Some((_, &oldest_index)) = active.iter().next() {
                let sink = &mut self.spatial_sinks.lock().unwrap()[oldest_index];
                sink.stop();
                active.remove(&id); // Remove the old ID if it exists for some reason
                oldest_index
            } else {
                // This should not happen if logic is correct, but as a fallback
                0
            }
        };

        let mut active_sources = self.active_sources.lock().unwrap();
        active_sources.insert(id, sink_index);

        let sink = &self.spatial_sinks.lock().unwrap()[sink_index];
        
        let source = sound.source.clone();
        let source = source.periodic_access(move |mut source, time| {
            // This is a bit of a hack to make looping work with spatial sinks
            // as rodio's spatial sinks don't have a built-in loop method.
            if looping && time.as_secs_f64() > source.total_duration().unwrap_or_default().as_secs_f64() {
                source.seek(Duration::from_secs(0)).ok();
            }
            source.next()
        });

        let source = source.set_position(position.x, position.y, position.z);
        let source = source.set_velocity(velocity.x, velocity.y, velocity.z);
        let source = source.set_pitch(pitch);
        let source = source.set_volume(volume * self.config.global_volume);
        
        sink.append(source);
        
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
        let id = Uuid::new_v4();
        
        let source = sound.source.clone();
        let source = if looping {
            source.looped()
        } else {
            source
        };
        
        let source = source.set_pitch(pitch);
        let source = source.set_volume(volume * self.config.global_volume);
        
        self.main_sink.append(source);
        
        PlayingSound { id }
    }

    /// Stops a specific playing sound.
    pub fn stop(&self, handle: PlayingSound) {
        if let Some(sink_index) = self.active_sources.lock().unwrap().remove(&handle.id) {
            let sink = &self.spatial_sinks.lock().unwrap()[sink_index];
            // This is a bit blunt, it stops the whole sink.
            // A more advanced system would track individual sources per sink.
            sink.stop();
            self.available_sinks.lock().unwrap().push(sink_index);
        }
    }

    /// Pauses a specific playing sound.
    pub fn pause(&self, handle: PlayingSound) {
        if let Some(&sink_index) = self.active_sources.lock().unwrap().get(&handle.id) {
            let sink = &self.spatial_sinks.lock().unwrap()[sink_index];
            sink.pause();
        }
    }
    
    /// Resumes a specific paused sound.
    pub fn resume(&self, handle: PlayingSound) {
        if let Some(&sink_index) = self.active_sources.lock().unwrap().get(&handle.id) {
            let sink = &self.spatial_sinks.lock().unwrap()[sink_index];
            sink.play();
        }
    }

    /// Sets the global volume for all sounds.
    pub fn set_global_volume(&mut self, volume: f32) {
        self.config.global_volume = volume.clamp(0.0, 1.0);
        // This is a simplification. A real engine would need to update all active sources.
        // For now, we just update the main sink. Spatial sinks are updated per-source.
        self.main_sink.set_volume(self.config.global_volume);
    }
    
    /// Pauses all sounds.
    pub fn pause_all(&self) {
        self.main_sink.pause();
        for sink in self.spatial_sinks.lock().unwrap().iter() {
            sink.pause();
        }
    }

    /// Resumes all sounds.
    pub fn resume_all(&self) {
        self.main_sink.play();
        for sink in self.spatial_sinks.lock().unwrap().iter() {
            sink.play();
        }
    }

    /// Updates the 3D position and orientation of the listener.
    /// This should be called every frame from your game's update loop.
    pub fn set_listener_position(&mut self, position: Vec3) {
        self.config.listener_position = position;
        self.update_listener_properties();
    }
    
    /// Updates the velocity of the listener (for doppler effect).
    pub fn set_listener_velocity(&mut self, velocity: Vec3) {
        self.config.listener_velocity = velocity;
        self.update_listener_properties();
    }
    
    /// Sets the orientation of the listener.
    pub fn set_listener_orientation(&mut self, forward: Vec3, up: Vec3) {
        self.config.listener_forward_vector = forward.normalize_or_zero();
        self.config.listener_up_vector = up.normalize_or_zero();
        self.update_listener_properties();
    }

    fn update_listener_properties(&self) {
        // Apply settings to all spatial sinks, as they each have their own listener emitter.
        for sink in self.spatial_sinks.lock().unwrap().iter() {
            sink.set_emitter_position(
                self.config.listener_position.x,
                self.config.listener_position.y,
                self.config.listener_position.z,
            );
            sink.set_emitter_velocity(
                self.config.listener_velocity.x,
                self.config.listener_velocity.y,
                self.config.listener_velocity.z,
            );
            sink.set_emitter_orientation(
                self.config.listener_forward_vector.x,
                self.config.listener_forward_vector.y,
                self.config.listener_forward_vector.z,
                self.config.listener_up_vector.x,
                self.config.listener_up_vector.y,
                self.config.listener_up_vector.z,
            );
        }
    }
}

// Make the engine thread-safe for use in a multi-threaded game engine.
// This is a simple implementation. A more robust one might use message passing.
unsafe impl Send for AudioEngine {}
unsafe impl Sync for AudioEngine {}
