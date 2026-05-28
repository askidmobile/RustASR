//! GGUF loader для Parakeet TDT 0.6B v3 (cstr/parakeet-tdt-0.6b-v3-gguf).
//!
//! GGUF структура (Q8_0 file):
//! - Metadata: parakeet.* keys (config), tokenizer.ggml.tokens (vocab)
//! - Tensors: mixed precision
//!   - Q8_0 (Linear FFN/MHA/joint/embedding) — 221 тензор, 544M params
//!   - F16 (Conv depthwise/pointwise + subsampling) — 81 тензор, 82.4M params
//!   - F32 (BatchNorm/LayerNorm/biases + preprocessor.fb/window) — 423 тензора
//!
//! Naming отличается от safetensors:
//! - encoder.pre_encode.conv.0.* → encoder.pre.conv.0.*
//! - encoder.layers.0.feed_forward1.* → encoder.layers.0.ff1.*
//! - encoder.layers.0.conv.depthwise_conv.* → encoder.layers.0.conv.dw.*
//! - encoder.layers.0.conv.pointwise_conv1.* → encoder.layers.0.conv.pw1.*
//! - encoder.layers.0.conv.batch_norm.* → encoder.layers.0.conv.bn.*
//! - encoder.layers.0.self_attn.linear_q.* → encoder.layers.0.attn.q.*
//! - decoder.prediction.embed.* → decoder.embed.*
//! - decoder.prediction.dec_rnn.lstm.weight_ih_l0 → decoder.lstm.0.w_ih
//! - joint.joint_net.2.* → joint.out.*

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Result, Tensor};
use std::sync::Arc;

use asr_core::AsrError;

use crate::config::{
    DecoderConfig, EncoderConfig, JointConfig, ParakeetConfig, PreprocessorConfig, TdtConfig,
};

/// Контейнер для одного тензора из GGUF — может быть quantized (Q8/Q4/...) или regular (F16/F32).
pub enum GgufTensor {
    Quantized(Arc<QTensor>),
    Regular(Tensor),
}

impl GgufTensor {
    /// Вернуть как regular Tensor (для quantized — dequantize).
    pub fn dequantize(&self, device: &Device) -> Result<Tensor> {
        match self {
            Self::Quantized(qt) => qt.dequantize(device),
            Self::Regular(t) => Ok(t.clone()),
        }
    }

    /// Вернуть как dequantized F16 (быстрее dequantize() который F32).
    pub fn dequantize_f16(&self, device: &Device) -> Result<Tensor> {
        match self {
            Self::Quantized(qt) => qt.dequantize_f16(device),
            Self::Regular(t) => t.to_dtype(DType::F16),
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized(_))
    }

    pub fn as_quantized(&self) -> Option<&Arc<QTensor>> {
        match self {
            Self::Quantized(qt) => Some(qt),
            _ => None,
        }
    }

    pub fn as_regular(&self) -> Option<&Tensor> {
        match self {
            Self::Regular(t) => Some(t),
            _ => None,
        }
    }
}

/// Загруженный Parakeet GGUF: тензоры + config из метаданных + tokenizer.
pub struct ParakeetGguf {
    /// Map: tensor_name → GgufTensor
    pub tensors: HashMap<String, GgufTensor>,
    /// Config извлечённый из parakeet.* метадаты
    pub config: ParakeetConfig,
    /// SentencePiece vocab (id → piece)
    pub vocab: Vec<String>,
    /// Mel filterbank [n_fft/2+1, n_mels] = [257, 128] из preprocessor.fb
    pub mel_filterbank: Tensor,
    /// Hann window [win_length=400] из preprocessor.window
    pub mel_window: Tensor,
}

impl ParakeetGguf {
    /// Загрузить GGUF файл и распарсить метаданные/тензоры.
    pub fn from_file(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;

        // Распарсить config из metadata
        let config = parse_config(&content.metadata)
            .map_err(|e| candle_core::Error::Msg(format!("parse config: {e}")))?;

        // Распарсить vocab из tokenizer.ggml.tokens
        let vocab = parse_vocab(&content.metadata)
            .map_err(|e| candle_core::Error::Msg(format!("parse vocab: {e}")))?;

        // Загрузить все тензоры
        let mut tensors = HashMap::with_capacity(content.tensor_infos.len());
        for (name, info) in &content.tensor_infos {
            let qt = content.tensor(&mut file, name, device)?;
            let gt = if qt.dtype() == candle_core::quantized::GgmlDType::F32 {
                GgufTensor::Regular(qt.dequantize(device)?)
            } else if qt.dtype() == candle_core::quantized::GgmlDType::F16 {
                GgufTensor::Regular(qt.dequantize(device)?.to_dtype(DType::F16)?)
            } else {
                // Q8_0, Q4_K, etc.
                GgufTensor::Quantized(Arc::new(qt))
            };
            let _ = info;
            tensors.insert(name.clone(), gt);
        }

        // Извлечь preprocessor.fb и preprocessor.window для mel extractor
        let mel_filterbank = match tensors.remove("preprocessor.fb") {
            Some(t) => t.dequantize(device)?,
            None => return Err(candle_core::Error::Msg("preprocessor.fb missing".into())),
        };
        let mel_window = match tensors.remove("preprocessor.window") {
            Some(t) => t.dequantize(device)?,
            None => return Err(candle_core::Error::Msg("preprocessor.window missing".into())),
        };

        Ok(Self {
            tensors,
            config,
            vocab,
            mel_filterbank,
            mel_window,
        })
    }

    /// Получить тензор по имени (для quantized → возвращает QTensor wrapper).
    pub fn get(&self, name: &str) -> Result<&GgufTensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("GGUF tensor not found: {name}")))
    }

    /// Получить как Regular Tensor (dequantize если нужно).
    pub fn get_tensor(&self, name: &str, device: &Device) -> Result<Tensor> {
        self.get(name)?.dequantize(device)
    }

    /// Получить как QTensor (только для quantized; ошибка если F16/F32).
    pub fn get_qtensor(&self, name: &str) -> Result<Arc<QTensor>> {
        match self.get(name)? {
            GgufTensor::Quantized(qt) => Ok(qt.clone()),
            GgufTensor::Regular(_) => Err(candle_core::Error::Msg(format!(
                "GGUF tensor {name} is not quantized"
            ))),
        }
    }

    /// Список tensor names (для debug).
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }
}

fn parse_config(
    meta: &HashMap<String, gguf_file::Value>,
) -> std::result::Result<ParakeetConfig, AsrError> {
    use gguf_file::Value;

    fn get_u32(m: &HashMap<String, Value>, k: &str) -> std::result::Result<u32, AsrError> {
        m.get(k)
            .ok_or_else(|| AsrError::Config(format!("missing meta {k}")))?
            .to_u32()
            .map_err(|e| AsrError::Config(format!("bad meta {k}: {e}")))
    }
    fn get_usize(m: &HashMap<String, Value>, k: &str) -> std::result::Result<usize, AsrError> {
        get_u32(m, k).map(|v| v as usize)
    }

    let sample_rate = get_usize(meta, "parakeet.sample_rate")?;
    let n_mels = get_usize(meta, "parakeet.n_mels")?;
    let n_fft = get_usize(meta, "parakeet.n_fft")?;
    let win_length = get_usize(meta, "parakeet.win_length")?;
    let hop_length = get_usize(meta, "parakeet.hop_length")?;
    let d_model = get_usize(meta, "parakeet.d_model")?;
    let n_layers = get_usize(meta, "parakeet.n_layers")?;
    let n_heads = get_usize(meta, "parakeet.n_heads")?;
    let head_dim = get_usize(meta, "parakeet.head_dim")?;
    let ff_dim = get_usize(meta, "parakeet.ff_dim")?;
    let subsampling_factor = get_usize(meta, "parakeet.subsampling_factor")?;
    let subsampling_channels = get_usize(meta, "parakeet.subsampling_channels")?;
    let conv_kernel = get_usize(meta, "parakeet.conv_kernel")?;
    let pred_hidden = get_usize(meta, "parakeet.pred_hidden")?;
    let pred_layers = get_usize(meta, "parakeet.pred_layers")?;
    let joint_hidden = get_usize(meta, "parakeet.joint_hidden")?;
    let vocab_size = get_usize(meta, "parakeet.vocab_size")?;
    let blank_id = get_usize(meta, "parakeet.blank_id")?;

    // TDT durations: array of i32/u32 → Vec<usize>
    let durations: Vec<usize> = match meta.get("parakeet.tdt_durations") {
        Some(Value::Array(arr)) => arr
            .iter()
            .map(|v| {
                // Поддерживаем и i32 и u32 элементы
                v.to_i32()
                    .map(|n| n as usize)
                    .or_else(|_| v.to_u32().map(|n| n as usize))
                    .map_err(|e| {
                        AsrError::Config(format!("bad parakeet.tdt_durations entry: {e}"))
                    })
            })
            .collect::<std::result::Result<_, _>>()?,
        _ => vec![0, 1, 2, 3, 4],
    };

    // Сборка ParakeetConfig из частей.
    Ok(ParakeetConfig {
        model_name: "parakeet-tdt-0.6b-v3".to_string(),
        model_class: "tdt".to_string(),
        sample_rate,
        preprocessor: PreprocessorConfig {
            sample_rate,
            features: n_mels,
            window_size: win_length as f64 / sample_rate as f64,
            window_stride: hop_length as f64 / sample_rate as f64,
            n_fft,
            normalize: "per_feature".to_string(),
            dither: 1e-5,
            preemph: 0.97,
            pad_to: 0,
        },
        encoder: EncoderConfig {
            n_layers,
            d_model,
            n_heads,
            d_k: head_dim,
            d_ff: ff_dim,
            conv_kernel_size: conv_kernel,
            subsampling: "dw_striding".to_string(),
            subsampling_factor,
            subsampling_conv_channels: subsampling_channels,
            feat_in: n_mels,
        },
        decoder: DecoderConfig {
            decoder_type: "lstm".to_string(),
            vocab_size: vocab_size + 1, // +1 for blank
            embed_dim: pred_hidden,
            pred_hidden,
            num_lstm_layers: pred_layers,
            blank_idx: blank_id,
        },
        joint: JointConfig {
            encoder_hidden: d_model,
            pred_hidden,
            joint_hidden,
            output_dim: vocab_size + 1 + durations.len(), // vocab + blank + durations
            num_classes: vocab_size + 1,                  // vocab + blank
            num_durations: durations.len(),
        },
        tdt: TdtConfig {
            durations,
            max_symbols_per_step: 10,
        },
    })
}

fn parse_vocab(
    meta: &HashMap<String, gguf_file::Value>,
) -> std::result::Result<Vec<String>, AsrError> {
    use gguf_file::Value;
    match meta.get("tokenizer.ggml.tokens") {
        Some(Value::Array(arr)) => arr
            .iter()
            .map(|v| {
                v.to_string()
                    .cloned()
                    .map_err(|e| AsrError::Config(format!("bad vocab entry: {e}")))
            })
            .collect(),
        _ => Err(AsrError::Config(
            "tokenizer.ggml.tokens missing or wrong type".into(),
        )),
    }
}
