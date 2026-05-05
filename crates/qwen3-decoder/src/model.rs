//! Qwen3 Decoder model.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use candle_transformers::quantized_var_builder as quantized_vb;
use std::path::Path;

use crate::cache::KvCache;
use crate::config::Qwen3Config;
use crate::layers::{DecoderLayer, RmsNorm, RotaryEmbedding, Weights};

/// Silent-вариант flush для Metal — без `tracing` warnings (вызывается часто).
fn flush_metal_pool_silent(device: &Device) {
    if device.is_metal() {
        let _ = device.synchronize();
        let _ = device.flush_buffers();
    }
}

/// Qwen3 Language Model Decoder.
///
/// Converts audio embeddings (from AuT encoder) to text tokens.
#[derive(Debug, Clone)]
pub struct Qwen3Decoder {
    config: Qwen3Config,
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: LmHead,
}

#[derive(Debug, Clone)]
struct LmHead {
    weight: Tensor, // [vocab, hidden]
}

impl LmHead {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, h) = x.dims3()?;
        let x2 = x.reshape((b * s, h))?;
        let w_t = self.weight.transpose(0, 1)?; // [hidden, vocab]
        let y = x2.matmul(&w_t)?;
        y.reshape((b, s, self.weight.dim(0)?))
    }
}

impl Qwen3Decoder {
    /// Create a new Qwen3 decoder from weights.
    ///
    /// Memory-engineering (по образцу Yttri Local LLM):
    /// - **`dequantize_f16(device)`** для embed_tokens (большой) — пишет в F16
    ///   Metal буфер напрямую (×2 экономия vs F32 dequantize → to_dtype).
    /// - **CPU dequant + to_device** для RMS norm weights (маленькие) — обходит
    ///   Metal blit pipeline, 1 буфер вместо 3.
    /// - **`flush_buffers()` после каждого слоя** — освобождает intermediate
    ///   Metal буферы (intermediate dequantize, blit copies) которые иначе
    ///   накапливаются в pool. Без этого Q8 0.6B даёт +2972 МБ peak вместо
    ///   ожидаемых ~700 МБ.
    pub fn new(config: Qwen3Config, weights: Weights<'_>, device: &Device) -> Result<Self> {
        let target_dtype = if device.is_metal() || device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        // embed_tokens.weight — самый большой тензор (vocab_size × hidden_size).
        // Для Qwen3-ASR-0.6B: 151936 × 1024 = 156М весов.
        // F32 dequantize = 622 МБ, F16 dequantize = 311 МБ — ×2 экономия пикового.
        let embed_weight = match &weights {
            Weights::Standard(vb) => vb
                .pp("embed_tokens")
                .get((config.vocab_size, config.hidden_size), "weight")?,
            Weights::Quantized(vb) => vb
                .pp("embed_tokens")
                .get((config.vocab_size, config.hidden_size), "weight")?
                .dequantize_f16(device)?,
        };

        let embed_weight = if embed_weight.dtype() != target_dtype {
            embed_weight.to_dtype(target_dtype)?
        } else {
            embed_weight
        };
        flush_metal_pool_silent(device);

        let embed_tokens = Embedding::new(embed_weight.clone(), config.hidden_size);

        // RoPE одинаковый для всех слоев, поэтому строим таблицы один раз и делимся ими.
        // Это критично по памяти: иначе каждый слой дублирует cos/sin.
        let rope = RotaryEmbedding::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            device,
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer =
                DecoderLayer::new(&config, weights.pp(format!("layers.{}", i)), rope.clone())?;
            layers.push(layer);
            // Per-layer flush — каждый слой создаёт ~10 intermediate dequantize
            // буферов (Q/K/V/O proj + 3 MLP + 2 RmsNorm). Без flush они
            // аккумулируются в Metal pool до конца loop.
            flush_metal_pool_silent(device);
        }

        let norm = match weights.pp("norm") {
            Weights::Standard(vb) => RmsNorm::new(config.hidden_size, config.rms_norm_eps, vb)?,
            Weights::Quantized(vb) => {
                // RMS norm — маленький (hidden_size = 1024), dequantize на CPU
                // и to_device(Metal) даёт 1 буфер вместо blit pipeline overhead
                // (см. Yttri rms_norm_cpu в quantized_qwen35.rs).
                let mut w = vb
                    .get((config.hidden_size,), "weight")?
                    .dequantize(&Device::Cpu)?
                    .to_device(device)?;
                if w.dtype() != target_dtype {
                    w = w.to_dtype(target_dtype)?;
                }
                RmsNorm::from_weight(w, config.rms_norm_eps)?
            }
        };

        let lm_head = LmHead {
            weight: embed_weight,
        };

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    /// Load decoder from safetensors file.
    pub fn from_safetensors(
        config: Qwen3Config,
        path: impl AsRef<Path>,
        device: &Device,
    ) -> Result<Self> {
        let paths = [path.as_ref()];
        Self::from_safetensors_files(config, &paths, device)
    }

    /// Load decoder from one or multiple safetensors files (шарды).
    pub fn from_safetensors_files(
        config: Qwen3Config,
        paths: &[&Path],
        device: &Device,
    ) -> Result<Self> {
        // Use F32 for CPU (BF16 not supported for matmul on CPU)
        // Use BF16 for GPU for better performance
        let dtype = if device.is_metal() || device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(paths, dtype, device)? };
        Self::new(config, Weights::Standard(vb.pp("thinker.model")), device)
    }

    /// Load decoder from gguf file.
    pub fn from_gguf(config: Qwen3Config, path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let vb = quantized_vb::VarBuilder::from_gguf(path.as_ref(), device)?;
        Self::new(config, Weights::Quantized(vb.pp("thinker.model")), device)
    }

    /// Forward pass for language modeling.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `start_pos` - Starting position for KV cache
    ///
    /// # Returns
    /// Logits [batch, seq_len, vocab_size]
    pub fn forward(&self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, start_pos)?;
        }

        let hidden_states = self.norm.forward(&hidden_states)?;
        self.lm_head.forward(&hidden_states)
    }

    /// Forward pass with audio embeddings instead of token embeddings.
    ///
    /// # Arguments
    /// * `audio_embeds` - Audio embeddings from AuT encoder [batch, audio_len, hidden_size]
    /// * `input_ids` - Optional text token IDs [batch, text_len]
    /// * `start_pos` - Starting position
    ///
    /// # Returns
    /// Logits for next token prediction
    pub fn forward_with_audio(
        &self,
        audio_embeds: &Tensor,
        input_ids: Option<&Tensor>,
        start_pos: usize,
    ) -> Result<Tensor> {
        // Combine audio embeddings with optional text embeddings
        let hidden_states = if let Some(ids) = input_ids {
            let text_embeds = self.embed_tokens.forward(ids)?;
            Tensor::cat(&[audio_embeds, &text_embeds], 1)?
        } else {
            audio_embeds.clone()
        };

        // Pass through decoder layers
        let mut hidden_states = hidden_states;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, start_pos)?;
        }

        let hidden_states = self.norm.forward(&hidden_states)?;
        self.lm_head.forward(&hidden_states)
    }

    /// Get the configuration.
    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Get the token embedding layer (for building prompts externally).
    pub fn get_embed_tokens(&self) -> &Embedding {
        &self.embed_tokens
    }

    /// Forward pass with pre-computed embeddings (for autoregressive generation).
    ///
    /// # Arguments
    /// * `embeds` - Pre-computed embeddings [batch, seq_len, hidden_size]
    /// * `start_pos` - Starting position for positional encoding
    ///
    /// # Returns
    /// Logits [batch, seq_len, vocab_size]
    pub fn forward_embeds(&self, embeds: &Tensor, start_pos: usize) -> Result<Tensor> {
        let debug = asr_core::debug::enabled();
        if debug {
            eprintln!(
                "DEBUG decoder forward_embeds: embeds dtype={:?}",
                embeds.dtype()
            );
        }
        let mut hidden_states = embeds.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            if debug {
                eprintln!(
                    "DEBUG decoder layer {}: hidden_states dtype={:?}",
                    i,
                    hidden_states.dtype()
                );
            }
            hidden_states = layer.forward(&hidden_states, start_pos)?;
            if debug {
                eprintln!(
                    "DEBUG decoder layer {}: hidden_states dtype={:?}",
                    i,
                    hidden_states.dtype()
                );
            }
        }

        let hidden_states = self.norm.forward(&hidden_states)?;
        if debug {
            eprintln!(
                "DEBUG decoder norm: hidden_states dtype={:?}",
                hidden_states.dtype()
            );
        }
        self.lm_head.forward(&hidden_states)
    }

    /// Forward pass с KV-кешем (prefill + decode).
    ///
    /// - В режиме prefill (кеш пустой) корректно применяет causal mask и заполнит кеш K/V.
    /// - В режиме decode ожидает `embeds` формы `[batch, 1, hidden]` и делает один шаг с дописыванием кеша.
    pub fn forward_embeds_with_cache(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let debug = asr_core::debug::enabled();
        let mut hidden_states = embeds.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            if debug {
                eprintln!(
                    "DEBUG decoder(cache) layer {}: hidden_states dtype={:?}",
                    i,
                    hidden_states.dtype()
                );
            }
            let layer_cache = cache.layer_mut(i);
            hidden_states =
                layer.forward_with_cache(&hidden_states, start_pos, Some(layer_cache))?;
        }

        let hidden_states = self.norm.forward(&hidden_states)?;
        self.lm_head.forward(&hidden_states)
    }
}
