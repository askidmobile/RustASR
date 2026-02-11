//! Конфигурация модели Parakeet-TDT v3.

use serde::{Deserialize, Serialize};

/// Корневая конфигурация модели Parakeet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParakeetConfig {
    /// Название модели.
    pub model_name: String,

    /// Тип декодера: "tdt".
    pub model_class: String,

    /// Частота дискретизации аудио.
    pub sample_rate: usize,

    /// Конфигурация препроцессора (mel-спектрограмма).
    pub preprocessor: PreprocessorConfig,

    /// Конфигурация FastConformer-энкодера.
    pub encoder: EncoderConfig,

    /// Конфигурация LSTM-декодера (prediction network).
    pub decoder: DecoderConfig,

    /// Конфигурация Joint Network.
    pub joint: JointConfig,

    /// Конфигурация TDT-декодирования.
    pub tdt: TdtConfig,
}

/// Конфигурация mel-спектрограммы (NeMo-стиль).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessorConfig {
    /// Частота дискретизации (16000).
    pub sample_rate: usize,

    /// Количество mel-бинов (128).
    pub features: usize,

    /// Размер окна в секундах (0.025 → 400 отсчётов).
    pub window_size: f64,

    /// Шаг окна в секундах (0.01 → 160 отсчётов).
    pub window_stride: f64,

    /// Размер FFT (512).
    pub n_fft: usize,

    /// Тип нормализации: "per_feature" или "per_utterance".
    pub normalize: String,

    /// Амплитуда дизеринга для предотвращения log(0).
    pub dither: f64,

    /// Коэффициент предварительного усиления (0.97).
    pub preemph: f64,

    /// Дополнение до кратности (0 = без дополнения).
    pub pad_to: usize,
}

/// Конфигурация FastConformer-энкодера.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    /// Количество слоёв Conformer (24).
    pub n_layers: usize,

    /// Размерность модели (1024).
    pub d_model: usize,

    /// Количество голов внимания (8).
    pub n_heads: usize,

    /// Размерность одной головы (128).
    pub d_k: usize,

    /// Размерность feed-forward (4096).
    pub d_ff: usize,

    /// Размер ядра свёртки в ConformerConvolution (9).
    pub conv_kernel_size: usize,

    /// Тип субдискретизации: "dw_striding".
    pub subsampling: String,

    /// Фактор субдискретизации (8).
    pub subsampling_factor: usize,

    /// Количество каналов субдискретизации (256).
    pub subsampling_conv_channels: usize,

    /// Количество входных mel-бинов (128).
    pub feat_in: usize,
}

/// Конфигурация LSTM-декодера (prediction network).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderConfig {
    /// Тип декодера: "lstm".
    pub decoder_type: String,

    /// Размер словаря включая blank (8193).
    pub vocab_size: usize,

    /// Размерность скрытого состояния (640).
    pub pred_hidden: usize,

    /// Размерность эмбеддинга (640).
    pub embed_dim: usize,

    /// Количество слоёв LSTM (2).
    pub num_lstm_layers: usize,

    /// Индекс blank-токена (8192).
    pub blank_idx: usize,
}

/// Конфигурация Joint Network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointConfig {
    /// Размерность скрытого слоя (640).
    pub joint_hidden: usize,

    /// Размерность входа от энкодера (1024).
    pub encoder_hidden: usize,

    /// Размерность входа от декодера (640).
    pub pred_hidden: usize,

    /// Размерность выхода (8198 = 8193 токенов + 5 длительностей).
    pub output_dim: usize,

    /// Количество классов (8193).
    pub num_classes: usize,

    /// Количество длительностей (5).
    pub num_durations: usize,
}

/// Конфигурация TDT-декодирования.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TdtConfig {
    /// Возможные значения длительностей: [0, 1, 2, 3, 4].
    pub durations: Vec<usize>,

    /// Максимальное количество символов на один фрейм.
    pub max_symbols_per_step: usize,
}

impl ParakeetConfig {
    /// Конфигурация по умолчанию для Parakeet-TDT 0.6B v3.
    pub fn default_v3() -> Self {
        Self {
            model_name: "parakeet-tdt-0.6b-v3".to_string(),
            model_class: "tdt".to_string(),
            sample_rate: 16000,
            preprocessor: PreprocessorConfig {
                sample_rate: 16000,
                features: 128,
                window_size: 0.025,
                window_stride: 0.01,
                n_fft: 512,
                normalize: "per_feature".to_string(),
                dither: 1e-5,
                preemph: 0.97,
                pad_to: 0,
            },
            encoder: EncoderConfig {
                n_layers: 24,
                d_model: 1024,
                n_heads: 8,
                d_k: 128,
                d_ff: 4096,
                conv_kernel_size: 9,
                subsampling: "dw_striding".to_string(),
                subsampling_factor: 8,
                subsampling_conv_channels: 256,
                feat_in: 128,
            },
            decoder: DecoderConfig {
                decoder_type: "lstm".to_string(),
                vocab_size: 8193,
                pred_hidden: 640,
                embed_dim: 640,
                num_lstm_layers: 2,
                blank_idx: 8192,
            },
            joint: JointConfig {
                joint_hidden: 640,
                encoder_hidden: 1024,
                pred_hidden: 640,
                output_dim: 8198,
                num_classes: 8193,
                num_durations: 5,
            },
            tdt: TdtConfig {
                durations: vec![0, 1, 2, 3, 4],
                max_symbols_per_step: 10,
            },
        }
    }

    /// Длина окна в отсчётах.
    pub fn win_length(&self) -> usize {
        (self.preprocessor.window_size * self.preprocessor.sample_rate as f64) as usize
    }

    /// Шаг фрейма в отсчётах.
    pub fn hop_length(&self) -> usize {
        (self.preprocessor.window_stride * self.preprocessor.sample_rate as f64) as usize
    }
}
