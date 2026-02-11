//! GigaAM v3 E2E CTC — чистая Rust-реализация на Candle.
//!
//! Conformer-энкодер (220M параметров) + CTC-голова для распознавания речи.
//! SOTA-качество для русского языка.

pub mod config;
pub mod conformer;
pub mod ctc;
pub mod mel;
pub mod model;

pub use model::GigaAmModel;
