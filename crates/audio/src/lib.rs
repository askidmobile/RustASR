//! # audio
//!
//! Audio processing module for RustASR.
//!
//! This crate handles:
//! - WAV file loading and saving
//! - Audio resampling to target sample rate (16kHz)
//! - Mel spectrogram extraction

pub mod loader;
pub mod mel;
pub mod resample;

pub use loader::load_wav;
pub use mel::MelSpectrogramExtractor;
pub use resample::Resampler;
