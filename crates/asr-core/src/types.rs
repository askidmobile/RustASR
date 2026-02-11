//! Общие типы для ASR-операций.
//!
//! Содержит базовые структуры данных, используемые всеми крейтами workspace:
//! буферы аудио, mel-спектрограммы, результаты транскрибации и опции.

use candle_core::Tensor;
use serde::{Deserialize, Serialize};

use crate::model_registry::{ModelType, QuantizationType};

// ---------------------------------------------------------------------------
// Аудио-буфер
// ---------------------------------------------------------------------------

/// Буфер необработанного аудио.
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Аудио-сэмплы (нормализованы к [-1.0, 1.0]).
    pub samples: Vec<f32>,

    /// Частота дискретизации в Гц.
    pub sample_rate: usize,

    /// Количество каналов.
    pub channels: usize,
}

impl AudioBuffer {
    /// Создать новый буфер аудио.
    pub fn new(samples: Vec<f32>, sample_rate: usize, channels: usize) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
        }
    }

    /// Длительность в секундах.
    pub fn duration(&self) -> f32 {
        self.samples.len() as f32 / (self.sample_rate * self.channels) as f32
    }

    /// Количество сэмплов на канал.
    pub fn num_samples(&self) -> usize {
        self.samples.len() / self.channels
    }
}

// ---------------------------------------------------------------------------
// Mel-спектрограмма
// ---------------------------------------------------------------------------

/// Представление mel-спектрограммы.
#[derive(Debug, Clone)]
pub struct MelSpectrum {
    /// Тензор формы [batch, time, n_mels].
    pub tensor: Tensor,

    /// Количество временных фреймов.
    pub num_frames: usize,

    /// Количество mel-бинов.
    pub num_mels: usize,
}

impl MelSpectrum {
    /// Создать новый mel-спектр.
    pub fn new(tensor: Tensor, num_frames: usize, num_mels: usize) -> Self {
        Self {
            tensor,
            num_frames,
            num_mels,
        }
    }
}

// ---------------------------------------------------------------------------
// Результат транскрибации (старый, для обратной совместимости)
// ---------------------------------------------------------------------------

/// Устаревший результат транскрибации.
///
/// Используется существующим Qwen3-ASR пайплайном.
/// Новый код должен использовать [`TranscriptionResult`].
#[derive(Debug, Clone)]
pub struct Transcription {
    /// Распознанный текст.
    pub text: String,

    /// Детектированный язык (если доступен).
    pub language: Option<String>,

    /// Время обработки в секундах.
    pub processing_time: f32,

    /// Real-time factor (processing_time / audio_duration).
    pub rtf: f32,
}

impl Transcription {
    /// Создать новый результат транскрибации.
    pub fn new(text: String, processing_time: f32, audio_duration: f32) -> Self {
        let rtf = if audio_duration > 0.0 {
            processing_time / audio_duration
        } else {
            0.0
        };

        Self {
            text,
            language: None,
            processing_time,
            rtf,
        }
    }

    /// Установить детектированный язык.
    pub fn with_language(mut self, language: String) -> Self {
        self.language = Some(language);
        self
    }
}

// ---------------------------------------------------------------------------
// Новые унифицированные типы (мульти-модельный API)
// ---------------------------------------------------------------------------

/// Результат транскрибации — унифицированный для всех моделей.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    /// Полный распознанный текст.
    pub text: String,

    /// Время инференса в секундах.
    pub inference_time_secs: f64,

    /// Длительность аудио в секундах.
    pub audio_duration_secs: f64,

    /// Real-Time Factor (inference_time / audio_duration).
    /// Значение < 1.0 означает «быстрее реального времени».
    pub rtf: f64,

    /// Название использованной модели.
    pub model_name: String,

    /// Сегменты с временными метками (если модель поддерживает).
    pub segments: Vec<Segment>,

    /// Детектированный или заданный язык (ISO 639-1).
    pub language: Option<String>,
}

impl TranscriptionResult {
    /// Создать результат из текста и метрик производительности.
    pub fn new(
        text: String,
        model_name: String,
        inference_time_secs: f64,
        audio_duration_secs: f64,
    ) -> Self {
        let rtf = if audio_duration_secs > 0.0 {
            inference_time_secs / audio_duration_secs
        } else {
            0.0
        };
        Self {
            text,
            inference_time_secs,
            audio_duration_secs,
            rtf,
            model_name,
            segments: Vec::new(),
            language: None,
        }
    }

    /// Добавить информацию о языке.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Добавить сегменты.
    pub fn with_segments(mut self, segments: Vec<Segment>) -> Self {
        self.segments = segments;
        self
    }
}

/// Сегмент транскрибации с временными метками.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    /// Начало сегмента в секундах.
    pub start: f64,
    /// Конец сегмента в секундах.
    pub end: f64,
    /// Текст сегмента.
    pub text: String,
    /// Уверенность (0.0–1.0), если доступна.
    pub confidence: Option<f64>,
}

/// Опции транскрибации — передаются в [`AsrModel::transcribe`](crate::traits::AsrModel::transcribe).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribeOptions {
    /// Форсировать язык вывода (ISO 639-1, например "ru", "en").
    /// `None` — автоопределение.
    pub language: Option<String>,

    /// Максимальное количество токенов для генерации.
    /// `None` — определяется автоматически по длительности аудио.
    pub max_tokens: Option<usize>,

    /// Включить генерацию временных меток.
    pub timestamps: bool,

    /// Температура сэмплирования (0.0 = greedy, > 0 = sampling).
    pub temperature: f32,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            language: None,
            max_tokens: None,
            timestamps: false,
            temperature: 0.0,
        }
    }
}

impl TranscribeOptions {
    /// Создать опции с заданным языком.
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    /// Создать опции с заданным лимитом токенов.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

/// Метаданные загруженной модели.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Тип модели.
    pub model_type: ModelType,
    /// Человекочитаемое название.
    pub display_name: String,
    /// Приблизительное количество параметров.
    pub parameters: Option<u64>,
    /// Размер весов в байтах.
    pub weights_size_bytes: Option<u64>,
    /// Тип квантизации.
    pub quantization: QuantizationType,
    /// Поддерживаемые языки.
    pub languages: Vec<String>,
    /// Бэкенд инференса.
    pub backend: String,
}

impl ModelInfo {
    /// Создать метаданные модели.
    pub fn new(model_type: ModelType) -> Self {
        Self {
            display_name: model_type.display_name().to_string(),
            backend: model_type.backend().to_string(),
            model_type,
            parameters: None,
            weights_size_bytes: None,
            quantization: QuantizationType::None,
            languages: Vec::new(),
        }
    }

    /// Задать количество параметров.
    pub fn with_parameters(mut self, params: u64) -> Self {
        self.parameters = Some(params);
        self
    }

    /// Задать тип квантизации.
    pub fn with_quantization(mut self, q: QuantizationType) -> Self {
        self.quantization = q;
        self
    }

    /// Задать список языков.
    pub fn with_languages(mut self, langs: Vec<String>) -> Self {
        self.languages = langs;
        self
    }
}
