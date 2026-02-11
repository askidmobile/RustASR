//! Реестр поддерживаемых ASR-моделей.
//!
//! Содержит перечисление типов моделей и метаданные о каждой.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Тип ASR-модели.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelType {
    /// OpenAI Whisper (Large v3 Turbo и другие варианты).
    Whisper,
    /// Сбер GigaAM v3 E2E CTC — специализирована на русский.
    GigaAm,
    /// NVIDIA Parakeet TDT — FastConformer + Token-and-Duration Transducer.
    Parakeet,
    /// Alibaba Qwen3-ASR — AuT encoder + Qwen3 LLM decoder.
    Qwen3Asr,
}

impl ModelType {
    /// Все поддерживаемые типы моделей.
    pub fn all() -> &'static [ModelType] {
        &[
            ModelType::Whisper,
            ModelType::GigaAm,
            ModelType::Parakeet,
            ModelType::Qwen3Asr,
        ]
    }

    /// Строковый идентификатор для CLI.
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelType::Whisper => "whisper",
            ModelType::GigaAm => "gigaam",
            ModelType::Parakeet => "parakeet",
            ModelType::Qwen3Asr => "qwen3-asr",
        }
    }

    /// Полное человекочитаемое название.
    pub fn display_name(&self) -> &'static str {
        match self {
            ModelType::Whisper => "Whisper Large v3 Turbo",
            ModelType::GigaAm => "GigaAM v3 E2E CTC",
            ModelType::Parakeet => "Parakeet TDT v3",
            ModelType::Qwen3Asr => "Qwen3-ASR-0.6B",
        }
    }

    /// Бэкенд инференса.
    pub fn backend(&self) -> &'static str {
        // Все модели на Candle (чистый Rust)
        "candle"
    }

    /// Парсинг из строки (CLI-совместимо).
    pub fn from_str_loose(s: &str) -> Option<ModelType> {
        match s.to_lowercase().as_str() {
            "whisper" | "whisper-large-v3-turbo" | "whisper-large" => Some(ModelType::Whisper),
            "gigaam" | "gigaam-ctc" | "gigaam-v3" => Some(ModelType::GigaAm),
            "parakeet" | "parakeet-tdt" | "parakeet-v3" => Some(ModelType::Parakeet),
            "qwen3-asr" | "qwen3" | "qwen" => Some(ModelType::Qwen3Asr),
            _ => None,
        }
    }
}

impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Формат квантизации весов.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationType {
    /// Полные веса (f32/f16/bf16).
    None,
    /// GGUF Q4_0.
    GgufQ4_0,
    /// GGUF Q8_0.
    GgufQ8_0,
    /// GGUF Q6_K.
    GgufQ6K,
}

impl fmt::Display for QuantizationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantizationType::None => write!(f, "f32/f16"),
            QuantizationType::GgufQ4_0 => write!(f, "GGUF Q4_0"),
            QuantizationType::GgufQ8_0 => write!(f, "GGUF Q8_0"),
            QuantizationType::GgufQ6K => write!(f, "GGUF Q6_K"),
        }
    }
}
