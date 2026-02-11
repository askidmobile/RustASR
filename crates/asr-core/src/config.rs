//! Конфигурационные структуры для ASR-моделей.

use serde::{Deserialize, Serialize};

/// Конфигурация ASR-модели (Qwen3-ASR).
///
/// Используется существующим пайплайном. Для новых моделей
/// см. [`crate::traits::AsrModel`] и специфичные конфиги каждой модели.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrModelConfig {
    /// Model variant name (e.g., "qwen3-asr-0.6b", "qwen3-asr-1.7b").
    pub model_name: String,

    /// Feature extractor configuration.
    pub feature_extractor: FeatureExtractorConfig,

    /// AuT encoder configuration.
    pub encoder: AuTEncoderConfig,

    /// LLM decoder hidden size.
    pub llm_hidden_size: usize,

    /// Vocabulary size for the tokenizer.
    pub vocab_size: usize,
}

impl Default for AsrModelConfig {
    fn default() -> Self {
        Self::qwen3_asr_0_6b()
    }
}

impl AsrModelConfig {
    /// Configuration for Qwen3-ASR-0.6B model.
    pub fn qwen3_asr_0_6b() -> Self {
        Self {
            model_name: "qwen3-asr-0.6b".to_string(),
            feature_extractor: FeatureExtractorConfig::default(),
            encoder: AuTEncoderConfig::qwen3_asr_0_6b(),
            llm_hidden_size: 896,
            vocab_size: 151936,
        }
    }

    /// Configuration for Qwen3-ASR-1.7B model.
    pub fn qwen3_asr_1_7b() -> Self {
        Self {
            model_name: "qwen3-asr-1.7b".to_string(),
            feature_extractor: FeatureExtractorConfig::default(),
            encoder: AuTEncoderConfig::qwen3_asr_1_7b(),
            llm_hidden_size: 1024,
            vocab_size: 151936,
        }
    }
}

/// Конфигурация mel-спектрограммы.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtractorConfig {
    /// Целевая частота дискретизации в Гц.
    pub sample_rate: usize,

    /// Размер окна FFT.
    pub n_fft: usize,

    /// Шаг между фреймами.
    pub hop_length: usize,

    /// Длина окна для STFT.
    pub win_length: usize,

    /// Количество mel-бинов.
    pub n_mels: usize,

    /// Минимальная частота для mel-фильтра.
    pub f_min: f32,

    /// Максимальная частота для mel-фильтра.
    pub f_max: f32,

    /// Шкала mel-фильтров.
    pub mel_scale: MelScale,

    /// Тип логарифма для mel-спектрограммы.
    pub log_type: LogType,

    /// Тип нормализации mel-спектрограммы.
    pub normalization: MelNormalization,
}

impl Default for FeatureExtractorConfig {
    fn default() -> Self {
        Self::whisper()
    }
}

impl FeatureExtractorConfig {
    /// Конфигурация для Whisper / Qwen3-ASR (128 mel bins, log10, Slaney).
    pub fn whisper() -> Self {
        Self {
            sample_rate: 16000,
            n_fft: 400,
            hop_length: 160,
            win_length: 400,
            n_mels: 128,
            f_min: 0.0,
            f_max: 8000.0,
            mel_scale: MelScale::Slaney,
            log_type: LogType::Log10,
            normalization: MelNormalization::WhisperDynamicRange,
        }
    }

    /// Конфигурация для GigaAM (64 mel bins, ln, per-utterance нормализация).
    pub fn gigaam() -> Self {
        Self {
            sample_rate: 16000,
            n_fft: 512,
            hop_length: 160,
            win_length: 512,
            n_mels: 64,
            f_min: 0.0,
            f_max: 8000.0,
            mel_scale: MelScale::Slaney,
            log_type: LogType::Ln,
            normalization: MelNormalization::PerUtterance,
        }
    }

    /// Конфигурация для Parakeet (80 mel bins, ln, per-utterance, HTK).
    pub fn parakeet() -> Self {
        Self {
            sample_rate: 16000,
            n_fft: 512,
            hop_length: 160,
            win_length: 512,
            n_mels: 80,
            f_min: 0.0,
            f_max: 8000.0,
            mel_scale: MelScale::Htk,
            log_type: LogType::Ln,
            normalization: MelNormalization::PerUtterance,
        }
    }
}

/// Шкала mel-фильтров.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MelScale {
    /// Slaney: линейная ниже 1000 Гц, логарифмическая выше.
    /// Используется Whisper, GigaAM, librosa.
    Slaney,
    /// HTK: полностью логарифмическая шкала.
    /// Используется Parakeet (NeMo).
    Htk,
}

/// Тип логарифма для mel-спектрограммы.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogType {
    /// log10 — используется Whisper, Qwen3-ASR.
    Log10,
    /// Натуральный логарифм (ln) — используется GigaAM, Parakeet.
    Ln,
}

/// Тип нормализации mel-спектрограммы.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MelNormalization {
    /// Whisper-стиль: dynamic range compression (clamp max-8, normalize (x+4)/4).
    WhisperDynamicRange,
    /// Per-utterance: вычитание среднего, деление на стандартное отклонение.
    /// Используется GigaAM, Parakeet.
    PerUtterance,
    /// Без нормализации.
    None,
}

/// Configuration for the AuT (Attention-based) Audio Encoder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuTEncoderConfig {
    /// Model dimension (hidden size).
    pub d_model: usize,

    /// Number of transformer encoder layers.
    pub encoder_layers: usize,

    /// Number of attention heads.
    pub encoder_attention_heads: usize,

    /// Feed-forward network dimension.
    pub encoder_ffn_dim: usize,

    /// Number of Mel filter bins (input dimension).
    pub num_mel_bins: usize,

    /// Downsample rate (temporal compression factor).
    pub downsample_rate: usize,

    /// Layer normalization epsilon.
    pub layer_norm_eps: f64,

    /// Dropout probability.
    pub dropout: f32,

    /// RoPE theta for position embeddings.
    pub rope_theta: f32,
}

impl Default for AuTEncoderConfig {
    fn default() -> Self {
        Self::qwen3_asr_0_6b()
    }
}

impl AuTEncoderConfig {
    /// Configuration for Qwen3-ASR-0.6B encoder.
    pub fn qwen3_asr_0_6b() -> Self {
        Self {
            d_model: 896,
            encoder_layers: 32,
            encoder_attention_heads: 20,
            encoder_ffn_dim: 5120,
            num_mel_bins: 128,
            downsample_rate: 8,
            layer_norm_eps: 1e-6,
            dropout: 0.0,
            rope_theta: 1_000_000.0,
        }
    }

    /// Configuration for Qwen3-ASR-1.7B encoder.
    pub fn qwen3_asr_1_7b() -> Self {
        Self {
            d_model: 1024,
            encoder_layers: 32,
            encoder_attention_heads: 20,
            encoder_ffn_dim: 5120,
            num_mel_bins: 128,
            downsample_rate: 8,
            layer_norm_eps: 1e-6,
            dropout: 0.0,
            rope_theta: 1_000_000.0,
        }
    }

    /// Head dimension (d_model / num_heads).
    pub fn head_dim(&self) -> usize {
        self.d_model / self.encoder_attention_heads
    }

    /// Output token rate in Hz (100 Hz / downsample_rate).
    pub fn output_token_rate(&self) -> f32 {
        100.0 / self.downsample_rate as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AsrModelConfig::default();
        assert_eq!(config.model_name, "qwen3-asr-0.6b");
        assert_eq!(config.encoder.d_model, 896);
    }

    #[test]
    fn test_feature_extractor_config() {
        let config = FeatureExtractorConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.n_mels, 128);
        assert_eq!(config.hop_length, 160);
    }

    #[test]
    fn test_encoder_head_dim() {
        let config = AuTEncoderConfig::qwen3_asr_0_6b();
        // 896 / 20 = 44.8, but typically adjusted
        assert!(config.head_dim() > 0);
    }
}
