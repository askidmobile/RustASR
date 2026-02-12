//! # asr-core
//!
//! Базовые типы, трейты и определения ошибок для RustASR Engine.
//!
//! Этот крейт предоставляет фундаментальные абстракции для всех остальных
//! крейтов в workspace:
//!
//! - Общие типы данных (`AudioBuffer`, `MelSpectrum`, `TranscriptionResult`)
//! - Конфигурационные структуры для моделей
//! - Унифицированная обработка ошибок через `AsrError`
//! - Trait [`AsrModel`] — единый интерфейс для всех ASR-моделей
//! - Реестр моделей [`ModelType`]

pub mod config;
pub mod debug;
pub mod error;
pub mod metal_utils;
pub mod model_files;
pub mod model_registry;
pub mod traits;
pub mod types;

// Обратная совместимость: старые экспорты
pub use config::{AsrModelConfig, AuTEncoderConfig, FeatureExtractorConfig};
pub use config::{LogType, MelNormalization, MelScale};
pub use error::{AsrError, AsrResult};
pub use types::{AudioBuffer, MelSpectrum, Transcription};

// Новые экспорты для мульти-модельного API
pub use model_registry::{ModelType, QuantizationType};
pub use traits::AsrModel;
pub use types::{ModelInfo, Segment, TranscribeOptions, TranscriptionResult};
