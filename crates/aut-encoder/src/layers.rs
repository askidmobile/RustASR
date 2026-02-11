//! Neural network layers for the AuT encoder.
//!
//! Based on the actual Qwen3-ASR audio_tower structure:
//! - LayerNorm (not RMSNorm) with bias
//! - Self-attention with Q/K/V/O projections with bias  
//! - FFN with fc1/fc2 (GELU activation)
//! - Post-norm architecture

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear};

use crate::config::AuTConfig;

/// LayerNorm layer with learnable weight and bias.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    /// Create a new LayerNorm layer.
    pub fn new(hidden_size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((hidden_size,), "weight")?;
        let bias = vb.get((hidden_size,), "bias")?;
        Ok(Self { weight, bias, eps })
    }

    /// Apply LayerNorm to the input tensor.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // HF LayerNorm: вычисления в float32 для стабильности, затем каст обратно.
        let input_dtype = x.dtype();
        let x_f32 = x.to_dtype(candle_core::DType::F32)?;

        let mean = x_f32.mean_keepdim(candle_core::D::Minus1)?;
        let x_centered = x_f32.broadcast_sub(&mean)?;
        let variance = x_centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x_centered.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let x_normed = x_normed.to_dtype(input_dtype)?;

        let w = if self.weight.dtype() != input_dtype {
            self.weight.to_dtype(input_dtype)?
        } else {
            self.weight.clone()
        };
        let b = if self.bias.dtype() != input_dtype {
            self.bias.to_dtype(input_dtype)?
        } else {
            self.bias.clone()
        };
        Ok(x_normed.broadcast_mul(&w)?.broadcast_add(&b)?)
    }
}

/// Multi-Head Self-Attention layer.
///
/// Structure from the model:
/// - q_proj, k_proj, v_proj, out_proj (all with bias)
/// - self_attn_layer_norm (pre-norm applied before attention)
#[derive(Debug, Clone)]
pub struct SelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl SelfAttention {
    /// Create a new attention layer.
    pub fn new(config: &AuTConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads;
        let hidden_size = config.d_model;

        let q_proj = linear(hidden_size, hidden_size, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_size, hidden_size, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_size, hidden_size, vb.pp("v_proj"))?;
        let out_proj = linear(hidden_size, hidden_size, vb.pp("out_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
        })
    }

    /// Forward pass.
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape to [batch, num_heads, seq_len, head_dim]
        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? / scale)?;

        // HF делает softmax в float32 для стабильности.
        let attn_f32 = attn_weights.to_dtype(candle_core::DType::F32)?;
        let attn_weights =
            candle_nn::ops::softmax_last_dim(&attn_f32)?.to_dtype(attn_weights.dtype())?;

        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back to [batch, seq_len, hidden_size]
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        self.out_proj.forward(&attn_output)
    }
}

/// Feed-Forward Network with GELU activation.
///
/// Structure: fc1 -> GELU -> fc2
#[derive(Debug, Clone)]
pub struct FeedForward {
    fc1: Linear,
    fc2: Linear,
}

impl FeedForward {
    /// Create a new FFN layer.
    pub fn new(config: &AuTConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.d_model;
        let intermediate_size = config.intermediate_size;

        let fc1 = linear(hidden_size, intermediate_size, vb.pp("fc1"))?;
        let fc2 = linear(intermediate_size, hidden_size, vb.pp("fc2"))?;

        Ok(Self { fc1, fc2 })
    }

    /// Forward pass with GELU activation.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.fc1.forward(x)?;
        let hidden = hidden.gelu_erf()?;
        self.fc2.forward(&hidden)
    }
}

/// Transformer Encoder Layer.
///
/// Structure from the model (post-norm):
/// - self_attn_layer_norm -> self_attn -> residual
/// - final_layer_norm -> mlp -> residual
#[derive(Debug, Clone)]
pub struct EncoderLayer {
    self_attn: SelfAttention,
    mlp: FeedForward,
    self_attn_layer_norm: LayerNorm,
    final_layer_norm: LayerNorm,
}

impl EncoderLayer {
    /// Create a new encoder layer.
    pub fn new(config: &AuTConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = SelfAttention::new(config, vb.pp("self_attn"))?;
        let mlp = FeedForward::new(config, vb.clone())?;
        let self_attn_layer_norm = LayerNorm::new(
            config.d_model,
            config.layer_norm_eps,
            vb.pp("self_attn_layer_norm"),
        )?;
        let final_layer_norm = LayerNorm::new(
            config.d_model,
            config.layer_norm_eps,
            vb.pp("final_layer_norm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            self_attn_layer_norm,
            final_layer_norm,
        })
    }

    /// Forward pass with post-norm and residual connections.
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Self-attention block
        let residual = hidden_states;
        let hidden_states = self.self_attn_layer_norm.forward(hidden_states)?;
        let hidden_states = self.self_attn.forward(&hidden_states)?;
        let hidden_states = (residual + hidden_states)?;

        // FFN block
        let residual = &hidden_states;
        let hidden_states = self.final_layer_norm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        residual + hidden_states
    }
}

/// Conv2D Downsampling Block.
///
/// Compresses Mel spectrogram using 2D convolutions.
///
/// Based on actual model structure:
/// - conv2d1: (480, 1, 3, 3) - input 1 channel (mel as 2D image)
/// - conv2d2: (480, 480, 3, 3) - intermediate
/// - conv2d3: (480, 480, 3, 3) - intermediate
/// - conv_out: Linear(7680, 896) - flatten and project
///
/// Input is treated as [batch, 1, n_mels, time] (mel spectrogram as 2D image)
#[derive(Debug, Clone)]
pub struct ConvDownsample {
    conv2d1: candle_nn::Conv2d,
    conv2d2: candle_nn::Conv2d,
    conv2d3: candle_nn::Conv2d,
    conv_out: Linear,
}

impl ConvDownsample {
    /// Create a new conv downsampling block.
    pub fn new(config: &AuTConfig, vb: VarBuilder) -> Result<Self> {
        use candle_nn::Conv2dConfig;

        let downsample_hidden_size = config.downsample_hidden_size;

        // conv2d1: 1 -> 480, kernel=3x3, stride=2, padding=1
        let conv2d1_cfg = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let conv2d1 =
            candle_nn::conv2d(1, downsample_hidden_size, 3, conv2d1_cfg, vb.pp("conv2d1"))?;

        // conv2d2: 480 -> 480, kernel=3x3, stride=2, padding=1
        let conv2d2_cfg = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let conv2d2 = candle_nn::conv2d(
            downsample_hidden_size,
            downsample_hidden_size,
            3,
            conv2d2_cfg,
            vb.pp("conv2d2"),
        )?;

        // conv2d3: 480 -> 480, kernel=3x3, stride=2, padding=1
        let conv2d3_cfg = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let conv2d3 = candle_nn::conv2d(
            downsample_hidden_size,
            downsample_hidden_size,
            3,
            conv2d3_cfg,
            vb.pp("conv2d3"),
        )?;

        // conv_out: Linear(480 * (128/8), 896) = (7680, 896)
        // Note: The weight shape is [896, 7680], so we need to transpose
        // Using linear_no_bias since we only have weight tensor
        let conv_out_dim = config.downsample_hidden_size * (config.num_mel_bins / 8);
        let conv_out = candle_nn::linear_no_bias(conv_out_dim, config.d_model, vb.pp("conv_out"))?;

        Ok(Self {
            conv2d1,
            conv2d2,
            conv2d3,
            conv_out,
        })
    }

    /// Forward pass: [batch, time, mels] -> [batch, time/8, d_model]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, _time, _n_mels) = x.dims3()?;

        // Input: [batch, time, n_mels] -> [batch, 1, n_mels, time]
        let x = x.transpose(1, 2)?; // [batch, n_mels, time]
        let x = x.unsqueeze(1)?; // [batch, 1, n_mels, time]

        // Conv layers with GELU
        let x = self.conv2d1.forward(&x)?;
        let x = x.gelu_erf()?;

        let x = self.conv2d2.forward(&x)?;
        let x = x.gelu_erf()?;

        let x = self.conv2d3.forward(&x)?;
        let x = x.gelu_erf()?;

        // After 3x stride-2 convs: [batch, 480, n_mels/8, time/8]
        // Reshape for linear: [batch, time/8, 480 * (n_mels/8)]
        let (_, channels, h, w) = x.dims4()?;
        let x = x.permute((0, 3, 1, 2))?; // [batch, time/8, 480, n_mels/8]
        let x = x.contiguous()?; // Ensure contiguous layout
        let x = x.reshape((batch_size, w, channels * h))?;

        // Linear projection to d_model
        self.conv_out.forward(&x)
    }
}

/// Audio Projector layers.
/// Maps encoder output to LLM input dimension.
#[derive(Debug, Clone)]
pub struct AudioProjector {
    proj1: Linear,
    proj2: Linear,
}

impl AudioProjector {
    /// Create audio projector.
    pub fn new(config: &AuTConfig, vb: VarBuilder) -> Result<Self> {
        let proj1 = linear(config.d_model, config.d_model, vb.pp("proj1"))?;
        let proj2 = linear(config.d_model, config.output_dim, vb.pp("proj2"))?;

        Ok(Self { proj1, proj2 })
    }

    /// Forward pass: [batch, seq, d_model] -> [batch, seq, output_dim]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.proj1.forward(x)?;
        let x = x.gelu_erf()?;
        self.proj2.forward(&x)
    }
}
