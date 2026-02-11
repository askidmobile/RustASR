//! FastConformer-энкодер для Parakeet-TDT.
//!
//! Архитектура:
//! - DwStriding субдискретизация (×8): 3 стадии Conv2d → Linear проекция
//! - RelPositionalEncoding: синусоидальное PE в стиле Transformer-XL
//! - 24 × ConformerLayer (Macaron): FFN₁(×0.5) → MHSA → Conv → FFN₂(×0.5) → LN
//! - RelPositionMultiHeadAttention: pos_bias_u/v + rel_shift
//! - ConformerConvolution: pointwise → GLU → depthwise → BN → SiLU → pointwise

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::VarBuilder;
use tracing::debug;

use crate::config::EncoderConfig;

// ============================================================================
// Вспомогательные функции
// ============================================================================

/// Загрузка Linear без bias (используя candle_nn для правильного batched broadcasting).
fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<candle_nn::Linear> {
    candle_nn::linear_no_bias(in_dim, out_dim, vb)
}

/// Загрузка Linear с bias.
fn linear_with_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<candle_nn::Linear> {
    candle_nn::linear(in_dim, out_dim, vb)
}

// ============================================================================
// Layer Normalization
// ============================================================================

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        let bias = vb.get(dim, "bias")?;
        Ok(Self {
            weight,
            bias,
            eps: 1e-5,
        })
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(candle_core::D::Minus1)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let var = x_centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let std = (var + self.eps)?.sqrt()?;
        let norm = x_centered.broadcast_div(&std)?;
        norm.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)
    }
}

// ============================================================================
// Batch Normalization (инференс-режим: running_mean/var)
// ============================================================================

struct BatchNorm1d {
    weight: Tensor,
    bias: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    eps: f64,
}

impl BatchNorm1d {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        let bias = vb.get(dim, "bias")?;
        let running_mean = vb.get(dim, "running_mean")?;
        let running_var = vb.get(dim, "running_var")?;
        Ok(Self {
            weight,
            bias,
            running_mean,
            running_var,
            eps: 1e-5,
        })
    }

    /// Forward для инференса: x [batch, channels, time].
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = self.running_mean.reshape((1, (), 1))?;
        let var = self.running_var.reshape((1, (), 1))?;
        let w = self.weight.reshape((1, (), 1))?;
        let b = self.bias.reshape((1, (), 1))?;

        let std = (var + self.eps)?.sqrt()?;
        x.broadcast_sub(&mean)?
            .broadcast_div(&std)?
            .broadcast_mul(&w)?
            .broadcast_add(&b)
    }
}

// ============================================================================
// DwStriding Subsampling (×8)
// ============================================================================

/// Субдискретизация dw_striding: 3 стадии Conv2d со stride=2.
///
/// Фактическая структура nn.Sequential (из весов модели):
/// - conv.0: Conv2d(1, C, 3, stride=2, pad=1) + bias  — channel expansion
/// - (index 1: ReLU — без весов)
/// - conv.2: Conv2d(C, C, 3, groups=C, stride=2, pad=1) + bias  — depthwise
/// - conv.3: Conv2d(C, C, 1) + bias  — pointwise
/// - (index 4: ReLU — без весов)
/// - conv.5: Conv2d(C, C, 3, groups=C, stride=2, pad=1) + bias  — depthwise
/// - conv.6: Conv2d(C, C, 1) + bias  — pointwise
/// - out: Linear(C * freq_out, d_model) + bias
struct DwStridingSubsampling {
    conv0_w: Tensor,
    conv0_b: Tensor,
    conv2_w: Tensor,
    conv2_b: Tensor,
    conv3_w: Tensor,
    conv3_b: Tensor,
    conv5_w: Tensor,
    conv5_b: Tensor,
    conv6_w: Tensor,
    conv6_b: Tensor,
    out: candle_nn::Linear,
    channels: usize,
}

impl DwStridingSubsampling {
    fn load(config: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let c = config.subsampling_conv_channels; // 256

        let conv_vb = vb.pp("conv");

        // Stage 0: Conv2d(1, C, 3, stride=2)
        let conv0_w = conv_vb.get((c, 1, 3, 3), "0.weight")?;
        let conv0_b = conv_vb.get(c, "0.bias")?;

        // Stage 1: depthwise Conv2d(C, C, 3, groups=C, stride=2)
        let conv2_w = conv_vb.get((c, 1, 3, 3), "2.weight")?;
        let conv2_b = conv_vb.get(c, "2.bias")?;

        // Stage 1: pointwise Conv2d(C, C, 1)
        let conv3_w = conv_vb.get((c, c, 1, 1), "3.weight")?;
        let conv3_b = conv_vb.get(c, "3.bias")?;

        // Stage 2: depthwise Conv2d(C, C, 3, groups=C, stride=2)
        let conv5_w = conv_vb.get((c, 1, 3, 3), "5.weight")?;
        let conv5_b = conv_vb.get(c, "5.bias")?;

        // Stage 2: pointwise Conv2d(C, C, 1)
        let conv6_w = conv_vb.get((c, c, 1, 1), "6.weight")?;
        let conv6_b = conv_vb.get(c, "6.bias")?;

        // Проекция: Linear(C * freq_out, d_model)
        // freq_out = feat_in / 8 = 128 / 8 = 16
        let freq_out = config.feat_in / config.subsampling_factor;
        let proj_in = c * freq_out;
        let out = linear_with_bias(proj_in, config.d_model, vb.pp("out"))?;

        Ok(Self {
            conv0_w,
            conv0_b,
            conv2_w,
            conv2_b,
            conv3_w,
            conv3_b,
            conv5_w,
            conv5_b,
            conv6_w,
            conv6_b,
            out,
            channels: c,
        })
    }

    /// Forward: [batch, 1, T, D] → [batch, T/8, d_model]
    ///
    /// Вход: mel transposed [B, 1, T, D] (NeMo convention: dim2=time, dim3=freq)
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, 1, T, D] где D=128 (mel bins)
        debug!(
            "  Sub input: {:?}, [{:.4}, {:.4}]",
            x.shape(),
            x.flatten_all()?.min(0)?.to_scalar::<f32>().unwrap_or(0.0),
            x.flatten_all()?.max(0)?.to_scalar::<f32>().unwrap_or(0.0),
        );

        // Stage 0: Conv2d(1→C, 3×3, stride=2, pad=1) + ReLU
        let x = conv2d_manual(x, &self.conv0_w, Some(&self.conv0_b), 2, 1, 1)?;
        let x = x.relu()?;
        debug!(
            "  After stage0: {:?}, [{:.4}, {:.4}]",
            x.shape(),
            x.flatten_all()?.min(0)?.to_scalar::<f32>().unwrap_or(0.0),
            x.flatten_all()?.max(0)?.to_scalar::<f32>().unwrap_or(0.0),
        );

        // Stage 1: Depthwise Conv2d(C, 3×3, groups=C, stride=2, pad=1)
        let x = conv2d_manual(&x, &self.conv2_w, Some(&self.conv2_b), 2, 1, self.channels)?;
        // Pointwise Conv2d(C→C, 1×1)
        let x = conv2d_manual(&x, &self.conv3_w, Some(&self.conv3_b), 1, 0, 1)?;
        let x = x.relu()?;
        debug!(
            "  After stage1: {:?}, [{:.4}, {:.4}]",
            x.shape(),
            x.flatten_all()?.min(0)?.to_scalar::<f32>().unwrap_or(0.0),
            x.flatten_all()?.max(0)?.to_scalar::<f32>().unwrap_or(0.0),
        );

        // Stage 2: Depthwise Conv2d(C, 3×3, groups=C, stride=2, pad=1)
        let x = conv2d_manual(&x, &self.conv5_w, Some(&self.conv5_b), 2, 1, self.channels)?;
        // Pointwise Conv2d(C→C, 1×1)
        let x = conv2d_manual(&x, &self.conv6_w, Some(&self.conv6_b), 1, 0, 1)?;
        debug!(
            "  After stage2: {:?}, [{:.4}, {:.4}]",
            x.shape(),
            x.flatten_all()?.min(0)?.to_scalar::<f32>().unwrap_or(0.0),
            x.flatten_all()?.max(0)?.to_scalar::<f32>().unwrap_or(0.0),
        );

        // x: [B, C, T/8, D/8] (convention: dim2=time, dim3=freq)
        let (b, _c, time, _freq) = x.dims4()?;
        // Transpose + flatten: [B, C, T, D] → [B, T, C, D] → [B, T, C*D]
        let x = x.permute((0, 2, 1, 3))?; // [B, T, C, F]
        let x = x.contiguous()?.reshape((b, time, ()))?;
        debug!(
            "  Pre-proj: {:?}, [{:.4}, {:.4}]",
            x.shape(),
            x.flatten_all()?.min(0)?.to_scalar::<f32>().unwrap_or(0.0),
            x.flatten_all()?.max(0)?.to_scalar::<f32>().unwrap_or(0.0),
        );

        // Линейная проекция → [B, T, d_model]
        let x = self.out.forward(&x)?;
        debug!(
            "  After proj: {:?}, [{:.4}, {:.4}]",
            x.shape(),
            x.flatten_all()?.min(0)?.to_scalar::<f32>().unwrap_or(0.0),
            x.flatten_all()?.max(0)?.to_scalar::<f32>().unwrap_or(0.0),
        );
        Ok(x)
    }
}

/// Ручная 2D свёртка через Candle conv2d.
fn conv2d_manual(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    groups: usize,
) -> Result<Tensor> {
    let _cfg = candle_nn::Conv2dConfig {
        stride,
        padding,
        dilation: 1,
        groups,
    };
    let out = x.conv2d(weight, padding, stride, 1, groups)?;
    if let Some(b) = bias {
        let b = b.reshape((1, (), 1, 1))?;
        out.broadcast_add(&b)
    } else {
        Ok(out)
    }
}

// ============================================================================
// Relative Positional Encoding (Transformer-XL-стиль)
// ============================================================================

/// Синусоидальное позиционное кодирование для relative attention.
///
/// Генерирует PE для позиций от -(T-1) до +(T-1).
///
/// NB: xscaling отключено (xscaling=False в model_config.yaml).
/// NeMo при xscaling=False устанавливает self.xscale = None и не масштабирует вход.
struct RelPositionalEncoding {
    d_model: usize,
}

impl RelPositionalEncoding {
    fn new(d_model: usize) -> Self {
        Self { d_model }
    }

    /// Сгенерировать позиционные эмбеддинги (без масштабирования входа).
    ///
    /// Вход: x [B, T, D]
    /// Выход: (x, pos_emb [1, 2T-1, D])
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let t = x.dim(1)?;
        let pos_emb = self.create_pe(t, x.device())?;
        Ok((x.clone(), pos_emb))
    }

    /// Создать позиционные эмбеддинги [1, 2T-1, D].
    fn create_pe(&self, t: usize, device: &Device) -> Result<Tensor> {
        let pe_len = 2 * t - 1;
        let d = self.d_model;

        let mut pe = vec![0.0f32; pe_len * d];

        for pos_idx in 0..pe_len {
            // Позиция: от -(T-1) до +(T-1)
            let pos = pos_idx as f32 - (t - 1) as f32;
            for i in 0..d / 2 {
                let freq = 1.0 / 10000.0f32.powf(2.0 * i as f32 / d as f32);
                let angle = pos * freq;
                pe[pos_idx * d + 2 * i] = angle.sin();
                pe[pos_idx * d + 2 * i + 1] = angle.cos();
            }
        }

        Tensor::from_vec(pe, (1, pe_len, d), device)
    }
}

// ============================================================================
// RelPositionMultiHeadAttention
// ============================================================================

/// Multi-head attention с relative positional bias (Transformer-XL style).
///
/// score = (q + bias_u) @ K^T + rel_shift((q + bias_v) @ P^T)
struct RelPositionMultiHeadAttention {
    linear_q: candle_nn::Linear,
    linear_k: candle_nn::Linear,
    linear_v: candle_nn::Linear,
    linear_out: candle_nn::Linear,
    linear_pos: candle_nn::Linear,
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
    n_heads: usize,
    d_k: usize,
}

impl RelPositionMultiHeadAttention {
    fn load(d_model: usize, n_heads: usize, d_k: usize, vb: VarBuilder) -> Result<Self> {
        let linear_q = linear_no_bias(d_model, d_model, vb.pp("linear_q"))?;
        let linear_k = linear_no_bias(d_model, d_model, vb.pp("linear_k"))?;
        let linear_v = linear_no_bias(d_model, d_model, vb.pp("linear_v"))?;
        let linear_out = linear_no_bias(d_model, d_model, vb.pp("linear_out"))?;
        let linear_pos = linear_no_bias(d_model, d_model, vb.pp("linear_pos"))?;

        let pos_bias_u = vb.get((n_heads, d_k), "pos_bias_u")?;
        let pos_bias_v = vb.get((n_heads, d_k), "pos_bias_v")?;

        Ok(Self {
            linear_q,
            linear_k,
            linear_v,
            linear_out,
            linear_pos,
            pos_bias_u,
            pos_bias_v,
            n_heads,
            d_k,
        })
    }

    /// Forward: x [B, T, D], pos_emb [1, 2T-1, D] → [B, T, D]
    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;
        let h = self.n_heads;
        let dk = self.d_k;

        // Q, K, V проекции: [B, T, D] → [B, H, T, dk]
        let q = self.linear_q.forward(x)?.reshape((b, t, h, dk))?.permute((0, 2, 1, 3))?;
        let k = self.linear_k.forward(x)?.reshape((b, t, h, dk))?.permute((0, 2, 1, 3))?;
        let v = self.linear_v.forward(x)?.reshape((b, t, h, dk))?.permute((0, 2, 1, 3))?;

        // Позиционные эмбеддинги: [1, 2T-1, D] → [1, H, 2T-1, dk]
        let pe_len = pos_emb.dim(1)?;
        let p = self.linear_pos.forward(pos_emb)?
            .reshape((1, pe_len, h, dk))?
            .permute((0, 2, 1, 3))?;

        // Content score: (q + bias_u) @ K^T
        // bias_u: [H, dk] → [1, H, 1, dk]
        let bias_u = self.pos_bias_u.reshape((1, h, 1, dk))?;
        let q_with_u = q.broadcast_add(&bias_u)?;
        let content_score = q_with_u.contiguous()?.matmul(&k.contiguous()?.t()?)?;
        // [B, H, T, T]

        // Position score: (q + bias_v) @ P^T → rel_shift
        let bias_v = self.pos_bias_v.reshape((1, h, 1, dk))?;
        let q_with_v = q.broadcast_add(&bias_v)?;
        let pos_score_full = q_with_v.contiguous()?.matmul(&p.contiguous()?.t()?)?;
        // [B, H, T, 2T-1] → rel_shift → [B, H, T, T]
        let pos_score = self.rel_shift(&pos_score_full, t)?;

        // Суммарный score
        let scale = (dk as f64).sqrt();
        let scores = (content_score + pos_score)? / scale;

        // Softmax + Attention
        let attn = candle_nn::ops::softmax_last_dim(&scores?)?;
        let context = attn.matmul(&v.contiguous()?)?;

        // Reshape: [B, H, T, dk] → [B, T, D]
        let context = context.permute((0, 2, 1, 3))?.reshape((b, t, h * dk))?;

        self.linear_out.forward(&context)
    }

    /// Relative shift: [B, H, T, 2T-1] → [B, H, T, T]
    ///
    /// Алгоритм NeMo:
    /// 1. Pad left: [B, H, T, 2T-1] → [B, H, T, 2T]
    /// 2. Reshape: [B, H, T, 2T] → [B, H, 2T, T]
    /// 3. Drop first row: [B, H, 2T-1, T]
    /// 4. Reshape back: [B, H, 2T-1, T] → [B, H, T, 2T-1]
    /// 5. Take first T columns: [B, H, T, T]
    fn rel_shift(&self, x: &Tensor, _t: usize) -> Result<Tensor> {
        let (b, h, t_q, pe_len) = x.dims4()?; // pe_len = 2T-1

        // 1. Pad left: [B, H, T, 2T-1] → [B, H, T, 2T]
        let pad = Tensor::zeros((b, h, t_q, 1), x.dtype(), x.device())?;
        let x_padded = Tensor::cat(&[&pad, x], 3)?;

        // 2. Reshape: [B, H, T, 2T] → [B, H, 2T, T]
        let x_reshaped = x_padded.contiguous()?.reshape((b, h, pe_len + 1, t_q))?;

        // 3. Drop first row: [B, H, 2T, T] → [B, H, 2T-1, T]
        let x_sliced = x_reshaped.narrow(2, 1, pe_len)?;

        // 4. Reshape back: [B, H, 2T-1, T] → [B, H, T, 2T-1]
        let x_shifted = x_sliced.contiguous()?.reshape((b, h, t_q, pe_len))?;

        // 5. Take first T columns: [B, H, T, 2T-1] → [B, H, T, T]
        let t = pe_len / 2 + 1; // T = (2T-1)/2 + 1
        let out = x_shifted.narrow(3, 0, t)?;

        Ok(out)
    }
}

// ============================================================================
// Conformer Feed-Forward
// ============================================================================

/// Feed-forward модуль: Linear → SiLU → Linear.
/// Все linear без bias.
struct ConformerFeedForward {
    linear1: candle_nn::Linear,
    linear2: candle_nn::Linear,
}

impl ConformerFeedForward {
    fn load(d_model: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        let linear1 = linear_no_bias(d_model, d_ff, vb.pp("linear1"))?;
        let linear2 = linear_no_bias(d_ff, d_model, vb.pp("linear2"))?;
        Ok(Self { linear1, linear2 })
    }
}

impl Module for ConformerFeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.linear1.forward(x)?;
        let h = candle_nn::Activation::Silu.forward(&h)?;
        self.linear2.forward(&h)
    }
}

// ============================================================================
// Conformer Convolution
// ============================================================================

/// ConformerConvolution: pointwise → GLU → depthwise → BatchNorm → SiLU → pointwise.
///
/// Все conv1d без bias, batch_norm с running stats.
struct ConformerConvolution {
    pointwise_conv1_w: Tensor, // [2*D, D, 1]
    depthwise_conv_w: Tensor,  // [D, 1, K]
    batch_norm: BatchNorm1d,
    pointwise_conv2_w: Tensor, // [D, D, 1]
    d_model: usize,
    kernel_size: usize,
}

impl ConformerConvolution {
    fn load(d_model: usize, kernel_size: usize, vb: VarBuilder) -> Result<Self> {
        let pointwise_conv1_w = vb.get((2 * d_model, d_model, 1), "pointwise_conv1.weight")?;
        let depthwise_conv_w = vb.get((d_model, 1, kernel_size), "depthwise_conv.weight")?;
        let batch_norm = BatchNorm1d::load(d_model, vb.pp("batch_norm"))?;
        let pointwise_conv2_w = vb.get((d_model, d_model, 1), "pointwise_conv2.weight")?;

        Ok(Self {
            pointwise_conv1_w,
            depthwise_conv_w,
            batch_norm,
            pointwise_conv2_w,
            d_model,
            kernel_size,
        })
    }

    /// Forward: x [B, T, D] → [B, T, D]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Transpose: [B, T, D] → [B, D, T]
        let x = x.permute((0, 2, 1))?;

        // Pointwise Conv1d: [B, D, T] → [B, 2D, T]
        let x = x.conv1d(&self.pointwise_conv1_w, 0, 1, 1, 1)?;

        // GLU: разбиваем пополам по dim=1, gate = sigmoid(b)
        let half = self.d_model;
        let a = x.narrow(1, 0, half)?;
        let b = x.narrow(1, half, half)?;
        let x = (a * candle_nn::Activation::Sigmoid.forward(&b)?)?;

        // Depthwise Conv1d: padding = kernel_size / 2, groups = D
        let pad = self.kernel_size / 2;
        let x = x.conv1d(&self.depthwise_conv_w, pad, 1, 1, self.d_model)?;

        // BatchNorm1d
        let x = self.batch_norm.forward(&x)?;

        // SiLU
        let x = candle_nn::Activation::Silu.forward(&x)?;

        // Pointwise Conv1d: [B, D, T] → [B, D, T]
        let x = x.conv1d(&self.pointwise_conv2_w, 0, 1, 1, 1)?;

        // Transpose обратно: [B, D, T] → [B, T, D]
        x.permute((0, 2, 1))
    }
}

// ============================================================================
// ConformerLayer (Macaron-net)
// ============================================================================

/// Один слой Conformer (Macaron-net архитектура):
/// residual + 0.5 * FFN₁(LN(x))  →
/// residual + MHSA(LN(x))        →
/// residual + Conv(LN(x))         →
/// residual + 0.5 * FFN₂(LN(x))  →
/// LN(x)
struct ConformerLayer {
    norm_ff1: LayerNorm,
    ff1: ConformerFeedForward,
    norm_attn: LayerNorm,
    self_attn: RelPositionMultiHeadAttention,
    norm_conv: LayerNorm,
    conv: ConformerConvolution,
    norm_ff2: LayerNorm,
    ff2: ConformerFeedForward,
    norm_out: LayerNorm,
}

impl ConformerLayer {
    fn load(config: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let d = config.d_model;
        let d_ff = config.d_ff;

        Ok(Self {
            norm_ff1: LayerNorm::load(d, vb.pp("norm_feed_forward1"))?,
            ff1: ConformerFeedForward::load(d, d_ff, vb.pp("feed_forward1"))?,
            norm_attn: LayerNorm::load(d, vb.pp("norm_self_att"))?,
            self_attn: RelPositionMultiHeadAttention::load(d, config.n_heads, config.d_k, vb.pp("self_attn"))?,
            norm_conv: LayerNorm::load(d, vb.pp("norm_conv"))?,
            conv: ConformerConvolution::load(d, config.conv_kernel_size, vb.pp("conv"))?,
            norm_ff2: LayerNorm::load(d, vb.pp("norm_feed_forward2"))?,
            ff2: ConformerFeedForward::load(d, d_ff, vb.pp("feed_forward2"))?,
            norm_out: LayerNorm::load(d, vb.pp("norm_out"))?,
        })
    }

    /// Forward: x [B, T, D], pos_emb [1, 2T-1, D] → [B, T, D]
    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        // 1. FFN₁ (×0.5)
        let residual = x;
        let x = (residual + (self.ff1.forward(&self.norm_ff1.forward(x)?)? * 0.5)?)?;

        // 2. Self-Attention
        let residual = &x;
        let x = (residual + self.self_attn.forward(&self.norm_attn.forward(&x)?, pos_emb)?)?;

        // 3. Convolution
        let residual = &x;
        let x = (residual + self.conv.forward(&self.norm_conv.forward(&x)?)?)?;

        // 4. FFN₂ (×0.5)
        let residual = &x;
        let x = (residual + (self.ff2.forward(&self.norm_ff2.forward(&x)?)? * 0.5)?)?;

        // 5. Final LayerNorm
        self.norm_out.forward(&x)
    }
}

// ============================================================================
// FastConformerEncoder
// ============================================================================

/// Полный FastConformer-энкодер.
pub struct FastConformerEncoder {
    subsampling: DwStridingSubsampling,
    pos_enc: RelPositionalEncoding,
    layers: Vec<ConformerLayer>,
}

impl FastConformerEncoder {
    /// Загрузка весов из safetensors через VarBuilder.
    pub fn load(config: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let subsampling = DwStridingSubsampling::load(config, vb.pp("pre_encode"))?;
        let pos_enc = RelPositionalEncoding::new(config.d_model);

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let layer = ConformerLayer::load(config, vb.pp(format!("layers.{i}")))?;
            layers.push(layer);
        }

        debug!(
            "FastConformerEncoder загружен: {} слоёв, d_model={}, n_heads={}",
            config.n_layers, config.d_model, config.n_heads
        );

        Ok(Self {
            subsampling,
            pos_enc,
            layers,
        })
    }

    /// Forward: mel [1, n_mels, time] → encoder_output [1, T/8, d_model].
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // NeMo convention: mel [B, D, T] → transpose → [B, T, D] → unsqueeze → [B, 1, T, D]
        // Это важно: Conv2d обрабатывает dim2 как time, dim3 как freq
        let x = mel.permute((0, 2, 1))?.unsqueeze(1)?; // [B, 1, T, D]

        // Subsampling: [B, 1, T, D] → [B, T/8, d_model]
        let x = self.subsampling.forward(&x)?;
        debug!(
            "Subsampling output: {:?}, [{:.4}, {:.4}]",
            x.shape(),
            x.flatten_all()?.min(0)?.to_scalar::<f32>().unwrap_or(0.0),
            x.flatten_all()?.max(0)?.to_scalar::<f32>().unwrap_or(0.0),
        );

        // Positional encoding: масштабирование + PE
        let (x, pos_emb) = self.pos_enc.forward(&x)?;

        // N × ConformerLayer
        let mut x = x;
        debug!(
            "Pre-conformer (after xscale): [{:.4}, {:.4}]",
            x.flatten_all()?.min(0)?.to_scalar::<f32>().unwrap_or(0.0),
            x.flatten_all()?.max(0)?.to_scalar::<f32>().unwrap_or(0.0),
        );
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &pos_emb)?;
            debug!(
                "  Layer {}: [{:.4}, {:.4}]",
                i,
                x.flatten_all()?.min(0)?.to_scalar::<f32>().unwrap_or(0.0),
                x.flatten_all()?.max(0)?.to_scalar::<f32>().unwrap_or(0.0),
            );
        }

        Ok(x)
    }
}
