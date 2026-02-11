//! KV-кеш для авторегресcивной генерации.
//!
//! Реализован минимально необходимый кеш для Qwen3-ASR:
//! хранит K/V для каждого слоя в формате `[batch, kv_heads, seq, head_dim]`.

use candle_core::Tensor;

#[derive(Debug, Clone, Default)]
pub struct LayerKvCache {
    pub k: Option<Tensor>,
    pub v: Option<Tensor>,
}

impl LayerKvCache {
    pub fn seq_len(&self) -> usize {
        self.k.as_ref().map(|t| t.dim(2).unwrap_or(0)).unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
pub struct KvCache {
    pub layers: Vec<LayerKvCache>,
}

impl KvCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| LayerKvCache::default()).collect(),
        }
    }

    pub fn layer_mut(&mut self, idx: usize) -> &mut LayerKvCache {
        &mut self.layers[idx]
    }
}
