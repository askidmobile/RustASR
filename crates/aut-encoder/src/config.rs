//! Configuration for AuT encoder.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Configuration for the AuT Audio Encoder.
///
/// Based on the actual Qwen3-ASR model structure from config.json:
/// - audio_config with d_model, encoder_layers, encoder_attention_heads, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuTConfig {
    /// Model dimension (hidden size).
    pub d_model: usize,

    /// Number of transformer encoder layers.
    pub num_layers: usize,

    /// Number of attention heads.
    pub num_attention_heads: usize,

    /// Feed-forward network intermediate dimension.
    pub intermediate_size: usize,

    /// Number of Mel filter bins (input dimension).
    pub num_mel_bins: usize,

    /// Downsampling hidden size for Conv1D compression.
    pub downsample_hidden_size: usize,

    /// Output dimension (projection to LLM dimension).
    pub output_dim: usize,

    /// Layer normalization epsilon.
    pub layer_norm_eps: f64,

    /// Dropout probability (0.0 for inference).
    pub dropout: f32,

    /// Maximum source positions (for Conv1D chunking).
    pub max_source_positions: usize,

    /// Базовый размер окна (в mel-фреймах) для разбиения на чанки.
    /// В HF-референсе используется `n_window * 2`.
    pub n_window: usize,

    /// Окно для инференса (используется для внутренней логики attention в HF).
    /// В нашей реализации пока хранится для совместимости конфигов.
    pub n_window_infer: usize,

    /// Размер чанка при прогоне свёрток (в HF нужно для экономии памяти).
    /// В текущей реализации (по чанкам) это значение не критично.
    pub conv_chunksize: usize,

    /// Activation function ("gelu").
    pub activation_function: String,
}

impl Default for AuTConfig {
    fn default() -> Self {
        Self::qwen3_asr_0_6b()
    }
}

impl AuTConfig {
    /// Configuration for Qwen3-ASR-0.6B encoder.
    ///
    /// From actual model config.json:
    /// - d_model: 896
    /// - encoder_layers: 18
    /// - encoder_attention_heads: 14
    /// - encoder_ffn_dim: 3584
    /// - num_mel_bins: 128
    /// - output_dim: 1024
    pub fn qwen3_asr_0_6b() -> Self {
        Self {
            d_model: 896,
            num_layers: 18,
            num_attention_heads: 14,
            intermediate_size: 3584,
            num_mel_bins: 128,
            downsample_hidden_size: 480,
            output_dim: 1024,
            layer_norm_eps: 1e-5,
            dropout: 0.0,
            max_source_positions: 1500,
            n_window: 50,
            n_window_infer: 800,
            conv_chunksize: 500,
            activation_function: "gelu".to_string(),
        }
    }

    /// Configuration for Qwen3-ASR-1.7B encoder.
    pub fn qwen3_asr_1_7b() -> Self {
        Self {
            d_model: 1024,
            num_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            num_mel_bins: 128,
            downsample_hidden_size: 512,
            output_dim: 1536,
            layer_norm_eps: 1e-5,
            dropout: 0.0,
            max_source_positions: 1500,
            n_window: 50,
            n_window_infer: 800,
            conv_chunksize: 500,
            activation_function: "gelu".to_string(),
        }
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_attention_heads
    }

    /// Load configuration from a JSON file (config.json from HuggingFace).
    pub fn from_hf_config(path: impl AsRef<Path>) -> Result<Self, String> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| format!("Failed to read config file: {}", e))?;

        // Parse the HuggingFace config format
        let value: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| format!("Failed to parse JSON: {}", e))?;

        // Navigate to audio_config
        let audio_config = value
            .get("thinker_config")
            .and_then(|v| v.get("audio_config"))
            .ok_or("Missing thinker_config.audio_config")?;

        let text_config = value
            .get("thinker_config")
            .and_then(|v| v.get("text_config"))
            .ok_or("Missing thinker_config.text_config")?;

        Ok(Self {
            d_model: audio_config
                .get("d_model")
                .and_then(|v| v.as_u64())
                .unwrap_or(896) as usize,
            num_layers: audio_config
                .get("encoder_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(18) as usize,
            num_attention_heads: audio_config
                .get("encoder_attention_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(14) as usize,
            intermediate_size: audio_config
                .get("encoder_ffn_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(3584) as usize,
            num_mel_bins: audio_config
                .get("num_mel_bins")
                .and_then(|v| v.as_u64())
                .unwrap_or(128) as usize,
            downsample_hidden_size: audio_config
                .get("downsample_hidden_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(480) as usize,
            output_dim: audio_config
                .get("output_dim")
                .and_then(|v| v.as_u64())
                .or_else(|| text_config.get("hidden_size").and_then(|v| v.as_u64()))
                .unwrap_or(1024) as usize,
            layer_norm_eps: 1e-5,
            dropout: 0.0,
            max_source_positions: audio_config
                .get("max_source_positions")
                .and_then(|v| v.as_u64())
                .unwrap_or(1500) as usize,
            n_window: audio_config
                .get("n_window")
                .and_then(|v| v.as_u64())
                .unwrap_or(50) as usize,
            n_window_infer: audio_config
                .get("n_window_infer")
                .and_then(|v| v.as_u64())
                .unwrap_or(800) as usize,
            conv_chunksize: audio_config
                .get("conv_chunksize")
                .and_then(|v| v.as_u64())
                .unwrap_or(500) as usize,
            activation_function: audio_config
                .get("activation_function")
                .and_then(|v| v.as_str())
                .unwrap_or("gelu")
                .to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AuTConfig::default();
        assert_eq!(config.d_model, 896);
        assert_eq!(config.num_layers, 18);
        assert_eq!(config.num_attention_heads, 14);
        assert_eq!(config.output_dim, 1024);
    }

    #[test]
    fn test_head_dim() {
        let config = AuTConfig::qwen3_asr_0_6b();
        assert_eq!(config.head_dim(), 64); // 896 / 14 = 64
    }
}
