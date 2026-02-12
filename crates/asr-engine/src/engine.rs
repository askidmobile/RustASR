//! Диспетчеризация по типу модели.
//!
//! `AsrEngine` — единая точка входа для загрузки и использования
//! любой из поддерживаемых ASR-моделей.

use std::path::Path;

use candle_core::Device;
use tracing::info;

use asr_core::{
    AsrModel, AsrResult, ModelInfo, ModelType, TranscribeOptions, TranscriptionResult,
};

/// Единый движок ASR, абстрагирующий конкретную модель.
///
/// Под капотом хранит `Box<dyn AsrModel>` и делегирует вызовы.
pub struct AsrEngine {
    /// Внутренняя модель.
    inner: Box<dyn AsrModel>,
}

impl AsrEngine {
    /// Загрузить модель по типу и пути к директории.
    ///
    /// # Аргументы
    /// * `model_type` — тип модели (Whisper, GigaAm, Parakeet, Qwen3Asr).
    /// * `model_dir` — путь к директории с файлами модели.
    /// * `device` — устройство (CPU, Metal, CUDA).
    ///
    /// # Ошибки
    /// Возвращает ошибку, если:
    /// - Тип модели не скомпилирован (feature gate отключен).
    /// - Файлы модели не найдены или повреждены.
    pub fn load(
        model_type: ModelType,
        model_dir: impl AsRef<Path>,
        device: &Device,
    ) -> AsrResult<Self> {
        Self::load_inner(model_type, model_dir.as_ref(), device, false)
    }

    /// Загрузить квантизированную модель (GGUF).
    pub fn load_quantized(
        model_type: ModelType,
        model_dir: impl AsRef<Path>,
        device: &Device,
    ) -> AsrResult<Self> {
        Self::load_inner(model_type, model_dir.as_ref(), device, true)
    }

    fn load_inner(
        model_type: ModelType,
        model_dir: &Path,
        device: &Device,
        quantized: bool,
    ) -> AsrResult<Self> {
        info!(
            "AsrEngine: загрузка модели {} из {:?} (quantized={})",
            model_type, model_dir, quantized
        );

        let inner: Box<dyn AsrModel> = match model_type {
            #[cfg(feature = "whisper")]
            ModelType::Whisper => {
                if quantized {
                    Box::new(model_whisper::WhisperModel::load_quantized(model_dir, device)?)
                } else {
                    Box::new(model_whisper::WhisperModel::load(model_dir, device)?)
                }
            }

            #[cfg(not(feature = "whisper"))]
            ModelType::Whisper => {
                return Err(asr_core::AsrError::Model(
                    "Whisper не скомпилирован. Включите feature 'whisper' в asr-engine.".into(),
                ));
            }

            #[cfg(feature = "gigaam")]
            ModelType::GigaAm => {
                if quantized {
                    return Err(asr_core::AsrError::Model(
                        "GigaAM: квантизированные модели пока не поддерживаются.".into(),
                    ));
                }
                // Metal safety: проверяем работоспособность GPU перед загрузкой
                // тяжёлой модели. Краш в AGXMetalG16X::fillBuffer на M4/macOS 26.x
                // происходит при определённых паттернах буферного пула.
                if device.is_metal() {
                    asr_core::metal_utils::metal_probe(device)?;
                }
                Box::new(model_gigaam::GigaAmModel::load(model_dir, device)?)
            }

            #[cfg(not(feature = "gigaam"))]
            ModelType::GigaAm => {
                return Err(asr_core::AsrError::Model(
                    "GigaAM не скомпилирован. Включите feature 'gigaam' в asr-engine.".into(),
                ));
            }

            #[cfg(feature = "parakeet")]
            ModelType::Parakeet => {
                if quantized {
                    return Err(asr_core::AsrError::Model(
                        "Parakeet: квантизированные модели пока не поддерживаются.".into(),
                    ));
                }
                Box::new(model_parakeet::ParakeetModel::load(model_dir, device)?)
            }

            #[cfg(not(feature = "parakeet"))]
            ModelType::Parakeet => {
                return Err(asr_core::AsrError::Model(
                    "Parakeet не скомпилирован. Включите feature 'parakeet' в asr-engine.".into(),
                ));
            }

            #[cfg(feature = "qwen3")]
            ModelType::Qwen3Asr => {
                if quantized {
                    Box::new(model_qwen3::Qwen3AsrModel::load_quantized(model_dir, device)?)
                } else {
                    Box::new(model_qwen3::Qwen3AsrModel::load(model_dir, device)?)
                }
            }

            #[cfg(not(feature = "qwen3"))]
            ModelType::Qwen3Asr => {
                return Err(asr_core::AsrError::Model(
                    "Qwen3-ASR не скомпилирован. Включите feature 'qwen3' в asr-engine.".into(),
                ));
            }
        };

        info!(
            "AsrEngine: модель '{}' загружена ({})",
            inner.name(),
            inner.model_info().quantization
        );

        Ok(Self { inner })
    }

    /// Создать движок из уже загруженной модели.
    pub fn from_model(model: Box<dyn AsrModel>) -> Self {
        Self { inner: model }
    }

    // -----------------------------------------------------------------------
    // Делегация AsrModel
    // -----------------------------------------------------------------------

    /// Имя загруженной модели.
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    /// Тип модели.
    pub fn model_type(&self) -> ModelType {
        self.inner.model_type()
    }

    /// Ожидаемая частота дискретизации.
    pub fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }

    /// Поддерживаемые языки.
    pub fn supported_languages(&self) -> &[&str] {
        self.inner.supported_languages()
    }

    /// Метаданные модели.
    pub fn model_info(&self) -> ModelInfo {
        self.inner.model_info()
    }

    /// Транскрибация аудио.
    ///
    /// Передайте моно-аудио с частотой [`Self::sample_rate()`],
    /// нормализованное к диапазону [-1.0, 1.0].
    pub fn transcribe(
        &mut self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> AsrResult<TranscriptionResult> {
        self.inner.transcribe(samples, options)
    }

    /// Список скомпилированных моделей.
    pub fn available_models() -> Vec<ModelType> {
        let mut models = Vec::new();

        #[cfg(feature = "whisper")]
        models.push(ModelType::Whisper);

        #[cfg(feature = "gigaam")]
        models.push(ModelType::GigaAm);

        #[cfg(feature = "parakeet")]
        models.push(ModelType::Parakeet);

        #[cfg(feature = "qwen3")]
        models.push(ModelType::Qwen3Asr);

        models
    }
}
