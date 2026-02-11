//! # asr-engine
//!
//! Единый фасад для всех ASR-моделей в RustASR.
//!
//! `AsrEngine` позволяет загружать любую поддерживаемую модель через единый
//! интерфейс и выполнять транскрибацию, не привязываясь к конкретной реализации.
//!
//! # Пример
//!
//! ```ignore
//! use asr_engine::AsrEngine;
//! use asr_core::{ModelType, TranscribeOptions};
//!
//! let mut engine = AsrEngine::load(
//!     ModelType::Whisper,
//!     "models/whisper-large-v3-turbo",
//!     &candle_core::Device::Cpu,
//! )?;
//!
//! let result = engine.transcribe(&samples, &TranscribeOptions::default())?;
//! println!("{}", result.text);
//! ```

mod engine;

pub use engine::AsrEngine;
