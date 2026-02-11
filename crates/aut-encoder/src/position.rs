//! Sinusoidal Position Embedding for AuT encoder.

use candle_core::{DType, Device, Result, Tensor};

/// Sinusoidal Position Embedding.
///
/// Generates sin/cos positional embeddings with log-spaced timescales.
/// Used in audio encoder after conv downsampling.
#[derive(Debug, Clone)]
pub struct SinusoidsPositionEmbedding {
    embedding: Tensor,
    dtype: DType,
}

impl SinusoidsPositionEmbedding {
    /// Create a new sinusoidal position embedding.
    ///
    /// # Arguments
    /// * `max_length` - Maximum sequence length
    /// * `channels` - Embedding dimension (must be even)
    /// * `max_timescale` - Maximum timescale for frequencies (default: 10000.0)
    /// * `device` - Device to create tensor on
    /// * `dtype` - Data type for the embedding
    pub fn new(
        max_length: usize,
        channels: usize,
        max_timescale: f32,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        if channels % 2 != 0 {
            return Err(candle_core::Error::Msg(
                "SinusoidsPositionEmbedding needs even channels".to_string(),
            ));
        }

        let half_channels = channels / 2;
        let log_timescale_increment = (max_timescale as f64).ln() / (half_channels as f64 - 1.0);

        // inv_timescales = exp(-log_timescale_increment * arange(half_channels))
        let inv_timescales: Vec<f32> = (0..half_channels)
            .map(|i| (-log_timescale_increment * i as f64).exp() as f32)
            .collect();

        // scaled_time[pos, i] = pos * inv_timescales[i]
        // Result shape: [max_length, channels]
        let mut embedding_data = vec![0.0_f32; max_length * channels];

        for pos in 0..max_length {
            for i in 0..half_channels {
                let scaled = pos as f32 * inv_timescales[i];
                embedding_data[pos * channels + i] = scaled.sin();
                embedding_data[pos * channels + half_channels + i] = scaled.cos();
            }
        }

        let embedding = Tensor::from_vec(embedding_data, (max_length, channels), device)?;

        // Convert to target dtype
        let embedding = embedding.to_dtype(dtype)?;

        Ok(Self { embedding, dtype })
    }

    /// Get positional embedding for a given sequence length.
    ///
    /// # Arguments
    /// * `seq_len` - Sequence length to get embeddings for
    ///
    /// # Returns
    /// Tensor of shape [seq_len, channels]
    pub fn forward(&self, seq_len: usize) -> Result<Tensor> {
        let emb = self.embedding.narrow(0, 0, seq_len)?;
        let debug = asr_core::debug::enabled();
        if debug {
            eprintln!(
                "DEBUG pos_emb: emb dtype={:?}, self.dtype={:?}",
                emb.dtype(),
                self.dtype
            );
        }
        // Convert to target dtype if needed
        if emb.dtype() != self.dtype {
            let result = emb.to_dtype(self.dtype)?;
            if debug {
                eprintln!("DEBUG pos_emb: converted to {:?}", result.dtype());
            }
            Ok(result)
        } else {
            Ok(emb)
        }
    }

    /// Get positional embedding as BF16 (for GPU).
    pub fn forward_bf16(&self, seq_len: usize) -> Result<Tensor> {
        self.forward(seq_len)?.to_dtype(DType::BF16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinusoidal_embedding() {
        let device = Device::Cpu;
        let emb = SinusoidsPositionEmbedding::new(100, 128, 10000.0, &device, DType::F32).unwrap();

        let out = emb.forward(50).unwrap();
        assert_eq!(out.dims(), &[50, 128]);
    }
}
