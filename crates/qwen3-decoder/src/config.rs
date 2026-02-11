//! Configuration for Qwen3 decoder.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Configuration for the Qwen3 LLM Decoder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3Config {
    /// Hidden size.
    pub hidden_size: usize,

    /// Number of hidden layers.
    pub num_hidden_layers: usize,

    /// Number of attention heads.
    pub num_attention_heads: usize,

    /// Number of key-value heads (for GQA).
    pub num_key_value_heads: usize,

    /// Intermediate size for MLP.
    pub intermediate_size: usize,

    /// Vocabulary size.
    pub vocab_size: usize,

    /// Maximum sequence length.
    pub max_position_embeddings: usize,

    /// RMS norm epsilon.
    pub rms_norm_eps: f64,

    /// RoPE theta.
    pub rope_theta: f64,

    /// Head dimension.
    pub head_dim: usize,

    /// Hidden activation (usually "silu").
    pub hidden_act: String,
}

impl Default for Qwen3Config {
    fn default() -> Self {
        Self::qwen3_asr_0_6b()
    }
}

impl Qwen3Config {
    /// Configuration for Qwen3-ASR-0.6B decoder.
    pub fn qwen3_asr_0_6b() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            intermediate_size: 3072,
            vocab_size: 151936,
            max_position_embeddings: 65536,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            head_dim: 128,
            hidden_act: "silu".to_string(),
        }
    }

    /// Load configuration from HuggingFace config.json.
    pub fn from_hf_config(path: impl AsRef<Path>) -> Result<Self, String> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| format!("Failed to read config: {}", e))?;

        let value: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| format!("Failed to parse JSON: {}", e))?;

        let text_config = value
            .get("thinker_config")
            .and_then(|v| v.get("text_config"))
            .ok_or("Missing thinker_config.text_config")?;

        Ok(Self {
            hidden_size: text_config
                .get("hidden_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(1024) as usize,
            num_hidden_layers: text_config
                .get("num_hidden_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(28) as usize,
            num_attention_heads: text_config
                .get("num_attention_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(16) as usize,
            num_key_value_heads: text_config
                .get("num_key_value_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(8) as usize,
            intermediate_size: text_config
                .get("intermediate_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(3072) as usize,
            vocab_size: text_config
                .get("vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(151936) as usize,
            max_position_embeddings: text_config
                .get("max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(65536) as usize,
            rms_norm_eps: text_config
                .get("rms_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-6),
            rope_theta: text_config
                .get("rope_theta")
                .and_then(|v| v.as_f64())
                .unwrap_or(1_000_000.0),
            head_dim: text_config
                .get("head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(128) as usize,
            hidden_act: text_config
                .get("hidden_act")
                .and_then(|v| v.as_str())
                .unwrap_or("silu")
                .to_string(),
        })
    }
}
