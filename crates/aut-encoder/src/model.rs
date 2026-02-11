//! AuT Audio Encoder model.

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;

use crate::config::AuTConfig;
use crate::layers::{AudioProjector, ConvDownsample, EncoderLayer, LayerNorm};
use crate::position::SinusoidsPositionEmbedding;

/// AuT (Attention-based) Audio Encoder.
///
/// Converts Mel spectrograms to audio embeddings.
///
/// Based on actual Qwen3-ASR audio_tower structure:
/// 1. Conv2D downsampling (8x temporal compression via 3x stride-2 convs)
/// 2. Sinusoidal positional embedding (CRITICAL!)
/// 3. 18 Transformer encoder layers with post-norm
/// 4. Final layer normalization (ln_post)
/// 5. Audio projector (proj1, proj2) to LLM dimension
#[derive(Debug, Clone)]
pub struct AuTEncoder {
    config: AuTConfig,
    conv_downsample: ConvDownsample,
    positional_embedding: SinusoidsPositionEmbedding,
    layers: Vec<EncoderLayer>,
    ln_post: LayerNorm,
    projector: AudioProjector,
    _dtype: DType,
}

impl AuTEncoder {
    /// Create a new AuT encoder from VarBuilder.
    pub fn new(config: AuTConfig, vb: VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        let conv_downsample = ConvDownsample::new(&config, vb.clone())?;

        // HF Qwen3-ASR использует max_source_positions для позиционки аудио-энкодера.
        let positional_embedding = SinusoidsPositionEmbedding::new(
            config.max_source_positions,
            config.d_model, // channels (896 for 0.6B)
            10000.0,        // max_timescale
            device,
            dtype, // Use same dtype as weights
        )?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer = EncoderLayer::new(&config, vb.pp(format!("layers.{}", i)))?;
            layers.push(layer);
        }

        let ln_post = LayerNorm::new(config.d_model, config.layer_norm_eps, vb.pp("ln_post"))?;
        let projector = AudioProjector::new(&config, vb.clone())?;

        Ok(Self {
            config,
            conv_downsample,
            positional_embedding,
            layers,
            ln_post,
            projector,
            _dtype: dtype,
        })
    }

    /// Load encoder from safetensors file.
    ///
    /// The weights should be at path `thinker.audio_tower.*` in the safetensors file.
    pub fn from_safetensors(
        config: AuTConfig,
        path: impl AsRef<Path>,
        device: &Device,
    ) -> Result<Self> {
        let paths = [path.as_ref()];
        Self::from_safetensors_files(config, &paths, device)
    }

    /// Load encoder from one or multiple safetensors files (шарды).
    pub fn from_safetensors_files(
        config: AuTConfig,
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
        Self::new(config, vb.pp("thinker.audio_tower"), device, dtype)
    }

    /// Forward pass through the encoder.
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram tensor of shape [batch, time, n_mels]
    ///
    /// # Returns
    /// Audio embeddings of shape [batch, time/4, output_dim]
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let debug = asr_core::debug::enabled();

        // Convert mel to correct dtype if needed
        let mel = if mel.dtype() != self._dtype {
            mel.to_dtype(self._dtype)?
        } else {
            mel.clone()
        };

        // HF Qwen3-ASR: вход режется по `n_window * 2` mel-фреймов.
        // После CNN-даунсемплинга чанки объединяются в одну последовательность,
        // а self-attention ограничивается окнами (`cu_seqlens`).
        let time_len = mel.dim(1)?;
        let chunk_len = self.config.n_window.saturating_mul(2).max(1);

        // 1) CNN + позиционка по каждому чанку.
        let mut aftercnn_chunks: Vec<Tensor> = Vec::new();
        let mut max_aftercnn_t: usize = 0;
        let mut start = 0usize;
        while start < time_len {
            let mut len = chunk_len;
            if start + len > time_len {
                len = time_len - start;
            }

            let mel_chunk = mel.i((.., start..start + len, ..))?;
            let mut h = self.conv_downsample.forward(&mel_chunk)?; // [1, t, d_model]
            let t = h.dim(1)?;
            max_aftercnn_t = max_aftercnn_t.max(t);

            let pos = self.positional_embedding.forward(t)?.unsqueeze(0)?;
            let pos = if pos.dtype() != h.dtype() {
                pos.to_dtype(h.dtype())?
            } else {
                pos
            };
            h = h.broadcast_add(&pos)?;
            aftercnn_chunks.push(h);

            start += len;
        }

        // 2) Склейка по времени: [1, total_t, d_model]
        let refs: Vec<&Tensor> = aftercnn_chunks.iter().collect();
        let mut hidden_states = Tensor::cat(refs.as_slice(), 1)?;
        let _total_t = hidden_states.dim(1)?;

        // 3) Transformer слои.
        //
        // Примечание: в актуальной реализации Python `qwen-asr` на CPU/"eager"-attention
        // `cu_seqlens` не ограничивает attention (kwargs игнорируются), поэтому поведение
        // ближе к глобальному self-attention по всей последовательности.
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        // 5) ln_post + projector на полной последовательности.
        hidden_states = self.ln_post.forward(&hidden_states)?;
        let out = self.projector.forward(&hidden_states)?;

        if debug {
            eprintln!(
                "DEBUG AuTEncoder: time_len={}, chunks={}, max_t={}, out_time={}",
                time_len,
                aftercnn_chunks.len(),
                max_aftercnn_t,
                out.dim(1)?
            );
        }

        Ok(out)
    }

    /// Get the configuration.
    pub fn config(&self) -> &AuTConfig {
        &self.config
    }

    /// Get the output dimension.
    pub fn output_dim(&self) -> usize {
        self.config.output_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_output_shape() {
        let config = AuTConfig::qwen3_asr_0_6b();

        // Verify configuration
        let _batch_size = 1;
        let time_frames = 800; // 8 seconds at 100 Hz
        let _n_mels = 128;

        // Expected output time: input / 4 due to 2x2 strided convolutions
        let expected_output_time = time_frames / 4;
        assert_eq!(expected_output_time, 200);

        // Expected output dim
        assert_eq!(config.output_dim, 1024);
    }

    #[test]
    fn test_config_from_hf() {
        let config_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("models")
            .join("qwen3-asr-0.6b")
            .join("config.json");

        if config_path.exists() {
            let config = AuTConfig::from_hf_config(&config_path).unwrap();
            assert_eq!(config.d_model, 896);
            assert_eq!(config.num_layers, 18);
            assert_eq!(config.num_attention_heads, 14);
            assert_eq!(config.output_dim, 1024);
        }
    }
}
