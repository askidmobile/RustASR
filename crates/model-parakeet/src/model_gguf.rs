//! GGUF-based loader для Parakeet TDT 0.6B v3.
//!
//! Phase 3b: всё через единый quantized_var_builder. Q8 linears (FFN/MHA/joint)
//! идут в `quantized_nn::Linear` через `QMatMul` без dequant в RAM. Regular
//! tensors (conv/norm/embed/lstm/biases) хранятся как QTensor F32/F16 wrap —
//! dequantize zero-cost на доступе.
//!
//! Memory:
//! - Phase 2 (Q8→F32 dequant): ~2.5 ГБ working set
//! - Phase 3a (Q8→F16 dequant): ~1.1 ГБ working set
//! - Phase 3b (real Q8 via QMatMul): ~544 МБ working set (target)

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use tracing::{debug, info};

use asr_core::AsrResult;
use candle_transformers::quantized_var_builder as qvb;

use crate::config::ParakeetConfig;
use crate::decoder::PredictionNet;
use crate::encoder::{FastConformerEncoder, Weights};
use crate::gguf::{GgufTensor, ParakeetGguf};
use crate::joint::JointNetwork;
use crate::mel::ParakeetMelExtractor;
use crate::tdt::TdtGreedyDecoder;

/// GGUF-loaded Parakeet model.
pub struct ParakeetModelGguf {
    pub encoder: FastConformerEncoder,
    pub prediction_net: PredictionNet,
    pub joint: JointNetwork,
    pub tdt_decoder: TdtGreedyDecoder,
    pub mel_extractor: ParakeetMelExtractor,
    pub vocab: Vec<String>,
    pub config: ParakeetConfig,
    pub device: Device,
}

impl ParakeetModelGguf {
    /// Загрузить Parakeet Q8 GGUF.
    /// Default: F16 dequant (memory win ~1GB vs F32). На CPU использует F32.
    pub fn load(path: impl AsRef<Path>, device: &Device) -> AsrResult<Self> {
        let dtype = if device.is_metal() || device.is_cuda() {
            DType::F16
        } else {
            DType::F32
        };
        Self::load_with_dtype(path, device, dtype)
    }

    /// Загрузить с указанным `target_dtype` для НЕ-quantized тензоров (norms/biases).
    /// Q8 веса остаются Q8 в RAM (Phase 3b — real QMatMul).
    ///
    /// - F32: norms/biases в F32 (numerically safer)
    /// - F16: norms/biases в F16 (memory win, ops уже F16-compatible)
    pub fn load_with_dtype(
        path: impl AsRef<Path>,
        device: &Device,
        target_dtype: DType,
    ) -> AsrResult<Self> {
        let path = path.as_ref();
        info!(
            "Loading Parakeet Q8 GGUF: {:?} (target_dtype={:?})",
            path, target_dtype
        );
        let t0 = Instant::now();

        let gguf = ParakeetGguf::from_file(path, device)
            .map_err(|e| asr_core::AsrError::Model(format!("ParakeetGguf::from_file: {e}")))?;
        let gguf_loaded_ms = t0.elapsed().as_millis();
        debug!("GGUF загружен за {} мс", gguf_loaded_ms);

        let config = gguf.config.clone();
        let vocab = gguf.vocab.clone();

        // Mel extractor из preprocessor.fb + preprocessor.window (CPU FFT)
        let mel_extractor = build_mel_extractor(&gguf, &config)
            .map_err(|e| asr_core::AsrError::Model(format!("build_mel_extractor: {e}")))?;

        // Split GGUF tensors на Q8 (для quantized backend) и regular (standard backend).
        // Q8 weights → quantized_nn::Linear с QMatMul (стают ~544 МБ Q8 в RAM).
        // Regular tensors (conv/norm/embed/lstm/biases) → dequant в target_dtype.
        let t1 = Instant::now();
        let (q_map, std_map) = split_gguf_tensors(gguf, device, target_dtype)
            .map_err(|e| asr_core::AsrError::Model(format!("split_gguf_tensors: {e}")))?;
        let remap_ms = t1.elapsed().as_millis();
        debug!(
            "Split: {} Q8 tensors + {} regular tensors за {} мс",
            q_map.len(),
            std_map.len(),
            remap_ms
        );

        let qvb = qvb::VarBuilder::from_tensors_map(q_map, device);
        let std_vb = VarBuilder::from_tensors(std_map, target_dtype, device);
        let weights = Weights::Hybrid {
            quantized: qvb,
            standard: std_vb,
        };

        let encoder = FastConformerEncoder::load_from_weights(&config.encoder, weights.pp("encoder"))
            .map_err(|e| asr_core::AsrError::Model(format!("encoder load: {e}")))?;
        let prediction_net =
            PredictionNet::load_from_weights(&config.decoder, weights.pp("decoder"))
                .map_err(|e| asr_core::AsrError::Model(format!("prediction_net load: {e}")))?;
        let joint = JointNetwork::load_from_weights(&config.joint, weights.pp("joint"))
            .map_err(|e| asr_core::AsrError::Model(format!("joint load: {e}")))?;
        let tdt_decoder = TdtGreedyDecoder::new(&config.tdt, config.decoder.blank_idx);

        let total_ms = t0.elapsed().as_millis();
        info!(
            "Parakeet GGUF загружен полностью за {} мс (gguf={}мс, remap={}мс)",
            total_ms, gguf_loaded_ms, remap_ms
        );

        Ok(Self {
            encoder,
            prediction_net,
            joint,
            tdt_decoder,
            mel_extractor,
            vocab,
            config,
            device: device.clone(),
        })
    }

    /// Декодирует token IDs в текст через SentencePiece vocab.
    pub fn decode_tokens(&self, tokens: &[u32]) -> String {
        let mut text = String::new();
        for &tok in tokens {
            let idx = tok as usize;
            if idx < self.vocab.len() {
                let piece = &self.vocab[idx];
                let decoded = piece.replace('▁', " ");
                text.push_str(&decoded);
            }
        }
        text.trim().to_string()
    }
}

/// Построить mel extractor используя preprocessor.fb и preprocessor.window из GGUF.
fn build_mel_extractor(
    gguf: &ParakeetGguf,
    config: &ParakeetConfig,
) -> Result<ParakeetMelExtractor> {
    // preprocessor.fb shape: [n_mels, n_fft/2+1] = [128, 257]
    // preprocessor.window shape: [win_length] = [400]
    // Существующий ParakeetMelExtractor::from_tensors принимает [1, n_mels, n_fft/2+1]
    let fb = gguf.mel_filterbank.unsqueeze(0)?; // [1, 128, 257]
    let window = gguf.mel_window.clone();
    ParakeetMelExtractor::from_tensors(config, &fb, Some(&window))
        .map_err(|e| candle_core::Error::Msg(e.to_string()))
}

/// Map GGUF tensor name (cstr naming) → safetensors-style name (HF naming наш encoder ожидает).
///
/// Возвращает None для тензоров которые не должны попасть в encoder/decoder/joint
/// (preprocessor.* уже извлечены отдельно).
fn map_gguf_name_to_safetensors(name: &str) -> Option<String> {
    // preprocessor.fb и .window обрабатываются отдельно
    if name == "preprocessor.fb" || name == "preprocessor.window" {
        return None;
    }

    // encoder.pre.* → encoder.pre_encode.* (наш DwStridingSubsampling)
    if let Some(rest) = name.strip_prefix("encoder.pre.") {
        return Some(format!("encoder.pre_encode.{rest}"));
    }

    // encoder.layers.N.* — нужны множественные подстановки
    if let Some(rest) = name.strip_prefix("encoder.layers.") {
        // Split layer_idx.rest
        let (layer_idx, suffix) = match rest.split_once('.') {
            Some((a, b)) => (a, b),
            None => return Some(format!("encoder.layers.{rest}")),
        };

        // Suffix mapping (order matters: longer prefixes first чтобы не было double-replace)
        let mapped = suffix
            // LayerNorm
            .replace("norm_ff1.", "norm_feed_forward1.")
            .replace("norm_ff2.", "norm_feed_forward2.")
            .replace("norm_attn.", "norm_self_att.")
            // FFN linears
            .replace("ff1.linear1", "feed_forward1.linear1")
            .replace("ff1.linear2", "feed_forward1.linear2")
            .replace("ff2.linear1", "feed_forward2.linear1")
            .replace("ff2.linear2", "feed_forward2.linear2")
            // Conv module
            .replace("conv.dw.", "conv.depthwise_conv.")
            .replace("conv.pw1.", "conv.pointwise_conv1.")
            .replace("conv.pw2.", "conv.pointwise_conv2.")
            .replace("conv.bn.", "conv.batch_norm.")
            // Attention
            .replace("attn.q.", "self_attn.linear_q.")
            .replace("attn.k.", "self_attn.linear_k.")
            .replace("attn.v.", "self_attn.linear_v.")
            .replace("attn.out.", "self_attn.linear_out.")
            .replace("attn.pos.", "self_attn.linear_pos.")
            .replace("attn.pos_bias_u", "self_attn.pos_bias_u")
            .replace("attn.pos_bias_v", "self_attn.pos_bias_v");

        return Some(format!("encoder.layers.{layer_idx}.{mapped}"));
    }

    // decoder.embed.weight → decoder.prediction.embed.weight
    if name == "decoder.embed.weight" {
        return Some("decoder.prediction.embed.weight".to_string());
    }

    // decoder.lstm.N.{w_ih,w_hh,b_ih,b_hh} → decoder.prediction.dec_rnn.lstm.{weight_ih_lN, ...}
    if let Some(rest) = name.strip_prefix("decoder.lstm.") {
        // Format: N.w_ih | N.w_hh | N.b_ih | N.b_hh
        let parts: Vec<&str> = rest.splitn(2, '.').collect();
        if parts.len() == 2 {
            let layer_idx = parts[0];
            let suffix = parts[1];
            let mapped_suffix = match suffix {
                "w_ih" => format!("weight_ih_l{layer_idx}"),
                "w_hh" => format!("weight_hh_l{layer_idx}"),
                "b_ih" => format!("bias_ih_l{layer_idx}"),
                "b_hh" => format!("bias_hh_l{layer_idx}"),
                other => return Some(format!("decoder.prediction.dec_rnn.lstm.{other}")),
            };
            return Some(format!(
                "decoder.prediction.dec_rnn.lstm.{mapped_suffix}"
            ));
        }
    }

    // joint.enc, joint.pred, joint.out → joint.enc, joint.pred, joint.joint_net.2
    if name.starts_with("joint.enc.") {
        return Some(name.to_string()); // совпадает
    }
    if name.starts_with("joint.pred.") {
        return Some(name.to_string()); // совпадает
    }
    if let Some(rest) = name.strip_prefix("joint.out.") {
        return Some(format!("joint.joint_net.2.{rest}"));
    }

    // Unknown tensor — оставляем как есть, может в загрузке мы его пропустим
    Some(name.to_string())
}

/// Split GGUF tensors на (Q8 map, regular map) с remap'ом имён.
///
/// - Q8/Q4/etc. → quantized backend (Arc<QTensor>) для use в `quantized_nn::Linear`
/// - F32/F16 → regular backend (Tensor, dequant в target_dtype или F32 если needs_f32)
fn split_gguf_tensors(
    gguf: ParakeetGguf,
    device: &Device,
    target_dtype: DType,
) -> Result<(HashMap<String, Arc<QTensor>>, HashMap<String, Tensor>)> {
    let mut q_map: HashMap<String, Arc<QTensor>> = HashMap::new();
    let mut std_map: HashMap<String, Tensor> = HashMap::new();

    for (gguf_name, gt) in gguf.tensors.into_iter() {
        let mapped = match map_gguf_name_to_safetensors(&gguf_name) {
            Some(m) => m,
            None => continue,
        };
        let qt = match gt {
            GgufTensor::Quantized(qt) => qt,
            GgufTensor::Regular(_) => continue, // gguf.rs больше не использует Regular
        };

        match qt.dtype() {
            GgmlDType::F32 | GgmlDType::F16 => {
                // Regular tensor — dequant в target_dtype (или F32 для norms/stats)
                let t = qt.dequantize(device)?;
                let final_dtype = if needs_f32(&mapped) {
                    DType::F32
                } else {
                    target_dtype
                };
                let t = if t.dtype() != final_dtype {
                    t.to_dtype(final_dtype)?
                } else {
                    t
                };
                std_map.insert(mapped, t);
            }
            _ => {
                // Q8_0, Q4_K, etc. — оставляем как QTensor
                q_map.insert(mapped, qt);
            }
        }
    }

    Ok((q_map, std_map))
}

/// Тензоры обязанные остаться F32 (numerical stability в norms/stats).
fn needs_f32(name: &str) -> bool {
    name.contains("batch_norm.running_mean")
        || name.contains("batch_norm.running_var")
        || name.contains("batch_norm.weight")
        || name.contains("batch_norm.bias")
        || (name.contains(".norm_") && (name.ends_with(".weight") || name.ends_with(".bias")))
}
