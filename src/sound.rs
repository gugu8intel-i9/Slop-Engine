// src/sound.rs

use rodio::{Decoder, OutputStream, Sink, Source};
use std::fs::File;
use std::io::{BufReader, Cursor};
use std::sync::Arc;
use thiserror::Error;

/// Represents an error that can occur when loading a sound.
#[derive(Debug, Error)]
pub enum SoundError {
    #[error("I/O Error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Failed to decode audio data: {0}")]
    Decode(#[from] rodio::decoder::DecoderError),
    #[error("Unsupported audio format")]
    UnsupportedFormat,
}

/// The raw audio data for a sound effect or music track.
/// This is designed to be loaded once and played many times.
/// It's cheap to clone as it's just an Arc to the data.
#[derive(Clone)]
pub struct Sound {
    pub(crate) source: Arc<dyn Source<Item = i16> + Send + Sync>,
    pub(crate) channels: u32,
    pub(crate) sample_rate: u32,
    pub(crate) total_duration: Option<std::time::Duration>,
}

impl Sound {
    /// Loads a sound from a file path.
    /// This is the primary way to load user-uploaded sounds.
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, SoundError> {
        let file = File::open(path)?;
        let buffered = BufReader::new(file);
        
        // Try to decode the file. Symphonia (via rodio's feature) handles various formats.
        let decoder = Decoder::new(buffered)?;
        let source = decoder.convert_i16();

        let channels = source.channels();
        let sample_rate = source.sample_rate();
        let total_duration = source.total_duration();

        Ok(Self {
            source: Arc::new(source),
            channels,
            sample_rate,
            total_duration,
        })
    }

    /// Loads a sound from an in-memory byte slice.
    /// Useful for loading from a network, a zip file, or embedded assets.
    pub fn from_bytes(data: &[u8]) -> Result<Self, SoundError> {
        let cursor = Cursor::new(data);
        let decoder = Decoder::new(cursor)?;
        let source = decoder.convert_i16();

        let channels = source.channels();
        let sample_rate = source.sample_rate();
        let total_duration = source.total_duration();

        Ok(Self {
            source: Arc::new(source),
            channels,
            sample_rate,
            total_duration,
        })
    }
}

// We can't automatically derive Debug for Sound because the source is a trait object.
impl std::fmt::Debug for Sound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sound")
            .field("channels", &self.channels)
            .field("sample_rate", &self.sample_rate)
            .field("total_duration", &self.total_duration)
            .finish()
    }
}
