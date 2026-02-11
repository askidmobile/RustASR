//! # aut-encoder
//!
//! AuT (Attention-based) Audio Encoder for RustASR.
//!
//! This crate implements the audio encoder architecture from Qwen3-ASR:
//! - Conv2D downsampling block (8x temporal compression)
//! - 32-layer Transformer encoder with self-attention
//! - Position embeddings
//!
//! The encoder converts Mel spectrograms to audio embeddings at 12.5 Hz token rate.

pub mod config;
pub mod layers;
pub mod model;
pub mod position;

pub use config::AuTConfig;
pub use model::AuTEncoder;
