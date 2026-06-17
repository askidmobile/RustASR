//! Конфигурация Nemotron 3.5 ASR — десериализуется из `config.json`,
//! сгенерированного `scripts/convert_nemotron.py`.
//!
//! Архитектура (ground truth из .nemo): FastConformer-CacheAware-RNNT + Lang prompt.

use serde::Deserialize;

/// Препроцессор: лог-мел спектрограмма (NeMo AudioToMelSpectrogramPreprocessor).
#[derive(Debug, Clone, Deserialize)]
pub struct PreprocessorConfig {
    pub sample_rate: u32,
    /// Число мел-бинов (128).
    pub features: usize,
    pub n_fft: usize,
    pub window_size: f64,
    pub window_stride: f64,
    pub window: String,
    /// "NA" — без нормализации (важно для совпадения с обучением).
    pub normalize: String,
    pub log: bool,
    pub preemph: f64,
    pub dither: f64,
    pub pad_to: i64,
}

impl PreprocessorConfig {
    /// win_length в сэмплах (window_size * sample_rate).
    pub fn win_length(&self) -> usize {
        (self.window_size * self.sample_rate as f64).round() as usize
    }
    /// hop_length в сэмплах (window_stride * sample_rate).
    pub fn hop_length(&self) -> usize {
        (self.window_stride * self.sample_rate as f64).round() as usize
    }
}

/// FastConformer cache-aware энкодер.
#[derive(Debug, Clone, Deserialize)]
pub struct EncoderConfig {
    pub feat_in: usize,
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub ff_expansion_factor: usize,
    pub d_ff: usize,
    pub conv_kernel_size: usize,
    pub conv_norm_type: String,
    /// "causal" — причинная свёртка (cache-aware streaming).
    pub conv_context_size: String,
    pub causal_downsampling: bool,
    pub subsampling: String,
    pub subsampling_factor: usize,
    pub subsampling_conv_channels: usize,
    pub self_attention_model: String,
    pub att_context_style: String,
    /// Список вариантов [[left,right],...] (в 80мс-фреймах).
    pub att_context_size: Option<Vec<Vec<i64>>>,
    /// Какой контекст использовать в offline (лучший WER): [56,13].
    pub att_context_for_offline: Vec<i64>,
    pub use_bias: bool,
    pub xscaling: bool,
    pub untie_biases: bool,
    pub pos_emb_max_len: usize,
}

/// RNN-T prediction network (embed + 2-layer LSTM).
#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    /// Реальных токенов без blank (13087).
    pub vocab_size: usize,
    /// Строк в embed.weight = vocab + blank (13088).
    pub embed_rows: usize,
    pub pred_hidden: usize,
    pub pred_rnn_layers: usize,
    /// blank = последний индекс (13087).
    pub blank_idx: usize,
}

/// RNN-T joint network (без TDT durations).
#[derive(Debug, Clone, Deserialize)]
pub struct JointConfig {
    pub encoder_hidden: usize,
    pub pred_hidden: usize,
    pub joint_hidden: usize,
    pub num_classes: usize,
    /// num_classes + blank (13088).
    pub output_dim: usize,
    pub activation: String,
}

/// Language-ID prompt fusion (prompt_kernel + one-hot).
#[derive(Debug, Clone, Deserialize)]
pub struct PromptConfig {
    pub num_prompts: usize,
    pub hidden: usize,
    /// d_model + num_prompts (1024 + 128 = 1152).
    pub in_features: usize,
    pub out_features: usize,
    pub prompt_field: String,
    /// Маппинг кода языка → индекс one-hot (ru-RU = 11).
    pub lang_index: std::collections::HashMap<String, usize>,
    #[serde(default)]
    pub prompt_dictionary: std::collections::HashMap<String, usize>,
}

impl PromptConfig {
    /// Индекс языка для one-hot (по умолчанию ru-RU=11, fallback на prompt_dictionary).
    pub fn lang_idx(&self, lang: &str) -> Option<usize> {
        self.lang_index
            .get(lang)
            .or_else(|| self.prompt_dictionary.get(lang))
            .copied()
    }
}

/// Полная конфигурация модели.
#[derive(Debug, Clone, Deserialize)]
pub struct NemotronConfig {
    pub model_name: String,
    pub model_class: String,
    pub sample_rate: u32,
    pub preprocessor: PreprocessorConfig,
    pub encoder: EncoderConfig,
    pub decoder: DecoderConfig,
    pub joint: JointConfig,
    pub prompt: PromptConfig,
}

impl NemotronConfig {
    /// Загрузить из config.json.
    pub fn from_json_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }
}
