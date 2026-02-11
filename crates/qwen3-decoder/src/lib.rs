//! Qwen3 Decoder crate for speech recognition.
//!
//! This crate provides the Qwen3 language model decoder for
//! converting audio embeddings into text transcriptions.

pub mod cache;
pub mod config;
pub mod layers;
pub mod model;

pub use config::Qwen3Config;
pub use layers::{Attention, DecoderLayer, MLP, RmsNorm, RotaryEmbedding};
pub use model::Qwen3Decoder;
