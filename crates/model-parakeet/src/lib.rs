//! NVIDIA Parakeet-TDT v3 — чистая Rust-реализация на Candle.
//!
//! FastConformer-энкодер (627M параметров) + TDT-декодер для распознавания речи.
//! 25 европейских языков включая русский (WER 5.51% на FLEURS).

pub mod config;
pub mod decoder;
pub mod encoder;
pub mod joint;
pub mod mel;
pub mod model;
pub mod tdt;

pub use model::ParakeetModel;
