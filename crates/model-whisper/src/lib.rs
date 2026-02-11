//! # model-whisper
//!
//! Реализация Whisper ASR для RustASR (через candle-transformers).
//!
//! Поддержка: Whisper tiny / base / small / medium / large-v3 / large-v3-turbo.
//! Бэкенд: candle (чистый Rust). safetensors + GGUF квантизация.

pub mod decoder;
pub mod model;

pub use model::WhisperModel;
