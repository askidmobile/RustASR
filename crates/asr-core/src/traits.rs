//! Унифицированный trait для всех ASR-моделей.
//!
//! Каждая модель (Whisper, GigaAM, Parakeet, Qwen3-ASR) реализует
//! [`AsrModel`], обеспечивая единый интерфейс для загрузки и транскрибации.

use crate::error::AsrResult;
use crate::model_registry::ModelType;
use crate::types::{ModelInfo, TranscribeOptions, TranscriptionResult};

/// Унифицированный trait для всех ASR-моделей.
///
/// # Пример
/// ```ignore
/// let mut model = WhisperModel::load("models/whisper-large-v3-turbo", &device)?;
/// let result = model.transcribe(&samples, &TranscribeOptions::default())?;
/// println!("{}", result.text);
/// ```
pub trait AsrModel: Send {
    /// Уникальное имя загруженной модели (например, "whisper-large-v3-turbo").
    fn name(&self) -> &str;

    /// Тип модели для реестра.
    fn model_type(&self) -> ModelType;

    /// Ожидаемая частота дискретизации входного аудио (обычно 16000).
    fn sample_rate(&self) -> u32 {
        16_000
    }

    /// Список поддерживаемых языков (ISO 639-1 коды).
    ///
    /// Пустой слайс означает «язык не фиксирован» (мультиязычная модель).
    fn supported_languages(&self) -> &[&str];

    /// Информация о загруженной модели (параметры, размер, квантизация).
    fn model_info(&self) -> ModelInfo;

    /// Транскрибация аудио-сэмплов.
    ///
    /// # Аргументы
    /// * `samples` — моно аудио, `f32`, нормализованное к [-1.0, 1.0],
    ///   с частотой дискретизации [`Self::sample_rate()`].
    /// * `options` — параметры транскрибации (язык, max_tokens и пр.).
    ///
    /// # Ошибки
    /// Возвращает `AsrError` при проблемах с инференсом.
    fn transcribe(
        &mut self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> AsrResult<TranscriptionResult>;
}
