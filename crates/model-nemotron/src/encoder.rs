//! FastConformer cache-aware энкодер Nemotron 3.5 ASR.
//!
//! Структура как Parakeet FastConformer, КРОМЕ:
//! - **causal subsampling**: dw_striding, асимметричный паддинг (left=k-1=2, right=s-1=1)
//!   на оба измерения → time 301→39, freq 128→17.
//! - **causal depthwise conv** в conformer conv-модуле (pad left=k-1=8, right=0).
//! - **conv-norm = LayerNorm** (ключ `conv.batch_norm.{weight,bias}`, без running stats).
//! - **chunked_limited attention** [56,13]: chunk=14, left_chunks=4; i видит j ⇔ 0≤i//14−j//14≤4.
//!
//! Ф3 parity (tmp/parity/nemotron_*.npy): subsampling→pre_encode_out ✓,
//! layers→layer0_out/layer1_out, full→encoded_pre_prompt.

use candle_core::{DType, Result, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};

use crate::config::EncoderConfig;

// ============================================================================
// Subsampling (causal dw_striding ×8)
// ============================================================================

pub struct Subsampling {
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
    out: Linear,
    channels: usize,
}

fn causal_conv2d_s2(x: &Tensor, w: &Tensor, b: &Tensor, groups: usize) -> Result<Tensor> {
    // CausalConv2D: F.pad(x, (left, right, left, right)), left=k-1=2, right=s-1=1
    let x = x.pad_with_zeros(2, 2, 1)?; // time
    let x = x.pad_with_zeros(3, 2, 1)?; // freq
    let y = x.conv2d(w, 0, 2, 1, groups)?;
    y.broadcast_add(&b.reshape((1, (), 1, 1))?)
}

fn pointwise(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    let y = x.conv2d(w, 0, 1, 1, 1)?;
    y.broadcast_add(&b.reshape((1, (), 1, 1))?)
}

impl Subsampling {
    pub fn load(config: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let c = config.subsampling_conv_channels;
        let conv = vb.pp("conv");
        let out = candle_nn::linear(c * subsampled_len(config.feat_in), config.d_model, vb.pp("out"))?;
        Ok(Self {
            conv0_w: conv.get((c, 1, 3, 3), "0.weight")?,
            conv0_b: conv.get(c, "0.bias")?,
            conv2_w: conv.get((c, 1, 3, 3), "2.weight")?,
            conv2_b: conv.get(c, "2.bias")?,
            conv3_w: conv.get((c, c, 1, 1), "3.weight")?,
            conv3_b: conv.get(c, "3.bias")?,
            conv5_w: conv.get((c, 1, 3, 3), "5.weight")?,
            conv5_b: conv.get(c, "5.bias")?,
            conv6_w: conv.get((c, c, 1, 1), "6.weight")?,
            conv6_b: conv.get(c, "6.bias")?,
            out,
            channels: c,
        })
    }

    /// Вход: [B, 1, T, F]. Выход: [B, T', d_model].
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = causal_conv2d_s2(x, &self.conv0_w, &self.conv0_b, 1)?.relu()?;
        let x = causal_conv2d_s2(&x, &self.conv2_w, &self.conv2_b, self.channels)?;
        let x = pointwise(&x, &self.conv3_w, &self.conv3_b)?.relu()?;
        let x = causal_conv2d_s2(&x, &self.conv5_w, &self.conv5_b, self.channels)?;
        let x = pointwise(&x, &self.conv6_w, &self.conv6_b)?.relu()?;
        let (b, _c, t, _f) = x.dims4()?;
        let x = x.permute((0, 2, 1, 3))?.contiguous()?.reshape((b, t, ()))?;
        self.out.forward(&x)
    }
}

pub fn subsampled_len(n: usize) -> usize {
    let mut x = n;
    for _ in 0..3 {
        x = x / 2 + 1;
    }
    x
}

// ============================================================================
// Conformer building blocks
// ============================================================================

/// Feed-forward: Linear → SiLU → Linear (без bias).
struct FeedForward {
    l1: Linear,
    l2: Linear,
}

impl FeedForward {
    fn load(d: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            l1: candle_nn::linear_no_bias(d, d_ff, vb.pp("linear1"))?,
            l2: candle_nn::linear_no_bias(d_ff, d, vb.pp("linear2"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.l2.forward(&self.l1.forward(x)?.silu()?)
    }
}

/// Синусоидальное rel-pos кодирование (Transformer-XL).
/// NeMo RelPositionalEncoding: позиции УБЫВАЮТ от +(T-1) до -(T-1)
/// (pos_emb[0]=+(T-1), середина=0). Подтверждено parity к NeMo (dist 5.6e-08).
fn rel_pos_emb(t: usize, d: usize, device: &candle_core::Device) -> Result<Tensor> {
    let pe_len = 2 * t - 1;
    let mut pe = vec![0.0f32; pe_len * d];
    for p in 0..pe_len {
        let pos = (t - 1) as f32 - p as f32;
        for i in 0..d / 2 {
            let freq = 1.0 / 10000.0f32.powf(2.0 * i as f32 / d as f32);
            let a = pos * freq;
            pe[p * d + 2 * i] = a.sin();
            pe[p * d + 2 * i + 1] = a.cos();
        }
    }
    Tensor::from_vec(pe, (1, pe_len, d), device)
}

/// Rel-pos multi-head attention с pos_bias_u/v + rel_shift + опциональная маска.
struct RelPosMHA {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    pos: Linear,
    bias_u: Tensor,
    bias_v: Tensor,
    h: usize,
    dk: usize,
}

impl RelPosMHA {
    fn load(d: usize, h: usize, dk: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q: candle_nn::linear_no_bias(d, d, vb.pp("linear_q"))?,
            k: candle_nn::linear_no_bias(d, d, vb.pp("linear_k"))?,
            v: candle_nn::linear_no_bias(d, d, vb.pp("linear_v"))?,
            o: candle_nn::linear_no_bias(d, d, vb.pp("linear_out"))?,
            pos: candle_nn::linear_no_bias(d, d, vb.pp("linear_pos"))?,
            bias_u: vb.get((h, dk), "pos_bias_u")?,
            bias_v: vb.get((h, dk), "pos_bias_v")?,
            h,
            dk,
        })
    }

    fn rel_shift(&self, x: &Tensor) -> Result<Tensor> {
        let (b, h, tq, pe) = x.dims4()?;
        let pad = Tensor::zeros((b, h, tq, 1), x.dtype(), x.device())?;
        let x = Tensor::cat(&[&pad, x], 3)?;
        let x = x.contiguous()?.reshape((b, h, pe + 1, tq))?;
        let x = x.narrow(2, 1, pe)?;
        let x = x.contiguous()?.reshape((b, h, tq, pe))?;
        x.narrow(3, 0, pe / 2 + 1)
    }

    /// x [B,T,D], pos_emb [1,2T-1,D], add_mask [1,1,T,T] (0/-inf) → [B,T,D].
    fn forward(&self, x: &Tensor, pos_emb: &Tensor, add_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;
        let (h, dk) = (self.h, self.dk);
        let q = self.q.forward(x)?.reshape((b, t, h, dk))?.permute((0, 2, 1, 3))?;
        let k = self.k.forward(x)?.reshape((b, t, h, dk))?.permute((0, 2, 1, 3))?;
        let v = self.v.forward(x)?.reshape((b, t, h, dk))?.permute((0, 2, 1, 3))?;
        let pe_len = pos_emb.dim(1)?;
        let p = self.pos.forward(pos_emb)?.reshape((1, pe_len, h, dk))?.permute((0, 2, 1, 3))?;

        let bu = self.bias_u.reshape((1, h, 1, dk))?;
        let bv = self.bias_v.reshape((1, h, 1, dk))?;
        let content = q.broadcast_add(&bu)?.contiguous()?.matmul(&k.contiguous()?.t()?)?;
        let pos_full = q.broadcast_add(&bv)?.contiguous()?.matmul(&p.contiguous()?.t()?)?;
        let pos = self.rel_shift(&pos_full)?;
        let scale = (dk as f64).sqrt();
        let mut scores = (content + pos)?.affine(1.0 / scale, 0.0)?;
        if let Some(m) = add_mask {
            scores = scores.broadcast_add(m)?;
        }
        let attn = candle_nn::ops::softmax(&scores.contiguous()?, D::Minus1)?;
        let ctx = attn.matmul(&v.contiguous()?)?;
        let ctx = ctx.permute((0, 2, 1, 3))?.reshape((b, t, h * dk))?;
        self.o.forward(&ctx)
    }
}

/// Conformer conv: pointwise → GLU → causal depthwise → LayerNorm(каналы) → SiLU → pointwise.
struct ConvModule {
    pw1: Tensor, // [2D, D, 1]
    dw: Tensor,  // [D, 1, K]
    norm: candle_nn::LayerNorm,
    pw2: Tensor, // [D, D, 1]
    d: usize,
    k: usize,
}

impl ConvModule {
    fn load(d: usize, k: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            pw1: vb.get((2 * d, d, 1), "pointwise_conv1.weight")?,
            dw: vb.get((d, 1, k), "depthwise_conv.weight")?,
            // conv_norm_type=layer_norm: NeMo хранит как `batch_norm`, но это LayerNorm(d)
            norm: candle_nn::layer_norm(d, 1e-5, vb.pp("batch_norm"))?,
            pw2: vb.get((d, d, 1), "pointwise_conv2.weight")?,
            d,
            k,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x [B,T,D] → [B,D,T]
        let x = x.permute((0, 2, 1))?;
        // pointwise1 → [B,2D,T]
        let x = x.conv1d(&self.pw1, 0, 1, 1, 1)?;
        // GLU по dim=1
        let a = x.narrow(1, 0, self.d)?;
        let g = x.narrow(1, self.d, self.d)?;
        let g = g.neg()?.exp()?.affine(1.0, 1.0)?.recip()?; // sigmoid
        let x = (a * g)?;
        // causal depthwise: pad left=k-1, right=0
        let x = x.pad_with_zeros(2, self.k - 1, 0)?;
        let x = x.conv1d(&self.dw, 0, 1, 1, self.d)?;
        // LayerNorm по каналам: [B,D,T] → [B,T,D] → LN → [B,D,T]
        let x = x.permute((0, 2, 1))?;
        let x = self.norm.forward(&x)?;
        let x = x.permute((0, 2, 1))?;
        // SiLU → pointwise2
        let x = x.silu()?;
        let x = x.conv1d(&self.pw2, 0, 1, 1, 1)?;
        x.permute((0, 2, 1)) // [B,T,D]
    }
}

/// Один Conformer-слой (Macaron).
struct ConformerLayer {
    norm_ff1: candle_nn::LayerNorm,
    ff1: FeedForward,
    norm_attn: candle_nn::LayerNorm,
    attn: RelPosMHA,
    norm_conv: candle_nn::LayerNorm,
    conv: ConvModule,
    norm_ff2: candle_nn::LayerNorm,
    ff2: FeedForward,
    norm_out: candle_nn::LayerNorm,
}

impl ConformerLayer {
    fn load(c: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let d = c.d_model;
        Ok(Self {
            norm_ff1: candle_nn::layer_norm(d, 1e-5, vb.pp("norm_feed_forward1"))?,
            ff1: FeedForward::load(d, c.d_ff, vb.pp("feed_forward1"))?,
            norm_attn: candle_nn::layer_norm(d, 1e-5, vb.pp("norm_self_att"))?,
            attn: RelPosMHA::load(d, c.n_heads, c.d_k(), vb.pp("self_attn"))?,
            norm_conv: candle_nn::layer_norm(d, 1e-5, vb.pp("norm_conv"))?,
            conv: ConvModule::load(d, c.conv_kernel_size, vb.pp("conv"))?,
            norm_ff2: candle_nn::layer_norm(d, 1e-5, vb.pp("norm_feed_forward2"))?,
            ff2: FeedForward::load(d, c.d_ff, vb.pp("feed_forward2"))?,
            norm_out: candle_nn::layer_norm(d, 1e-5, vb.pp("norm_out"))?,
        })
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let x = (x + self.ff1.forward(&self.norm_ff1.forward(x)?)?.affine(0.5, 0.0)?)?;
        let x = (&x + self.attn.forward(&self.norm_attn.forward(&x)?, pos_emb, mask)?)?;
        let x = (&x + self.conv.forward(&self.norm_conv.forward(&x)?)?)?;
        let x = (&x + self.ff2.forward(&self.norm_ff2.forward(&x)?)?.affine(0.5, 0.0)?)?;
        self.norm_out.forward(&x)
    }
}

/// Аддитивная chunked_limited маска [1,1,T,T]: 0 где разрешено, -inf где нет.
/// i видит j ⇔ 0 ≤ chunk(i)-chunk(j) ≤ left_chunks, chunk(k)=k/(right+1).
fn chunk_mask(t: usize, left: i64, right: i64, device: &candle_core::Device) -> Result<Tensor> {
    let chunk = (right + 1).max(1);
    let left_chunks = if left >= 0 { left / chunk } else { i64::MAX };
    let mut m = vec![0.0f32; t * t];
    for i in 0..t {
        let ci = (i as i64) / chunk;
        for j in 0..t {
            let cj = (j as i64) / chunk;
            let diff = ci - cj;
            let allowed = diff >= 0 && diff <= left_chunks;
            if !allowed {
                m[i * t + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(m, (1, 1, t, t), device)
}

// ============================================================================
// FastConformerEncoder
// ============================================================================

pub struct NemotronEncoder {
    subsampling: Subsampling,
    layers: Vec<ConformerLayer>,
    d_model: usize,
    offline_context: (i64, i64),
}

impl NemotronEncoder {
    pub fn load(config: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let subsampling = Subsampling::load(config, vb.pp("pre_encode"))?;
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            layers.push(ConformerLayer::load(config, vb.pp(format!("layers.{i}")))?);
        }
        Ok(Self {
            subsampling,
            layers,
            d_model: config.d_model,
            offline_context: config.offline_context(),
        })
    }

    /// mel [B, n_mels, T] → subsampled [B, T', d_model] (выход pre_encode).
    pub fn forward_subsampling(&self, mel: &Tensor) -> Result<Tensor> {
        let x = mel.permute((0, 2, 1))?.unsqueeze(1)?;
        self.subsampling.forward(&x)
    }

    /// Прогон первых `n` conformer-слоёв (для parity). n=0 → только subsampling.
    pub fn forward_through(&self, mel: &Tensor, n: usize) -> Result<Tensor> {
        let mut x = self.forward_subsampling(mel)?; // [B,T,D]
        let t = x.dim(1)?;
        let pos = rel_pos_emb(t, self.d_model, x.device())?;
        let (l, r) = self.offline_context;
        let mask = chunk_mask(t, l, r, x.device())?;
        for layer in self.layers.iter().take(n) {
            x = layer.forward(&x, &pos, Some(&mask))?;
        }
        Ok(x)
    }

    /// Полный энкодер, выход [B, T', d_model] (для prompt_kernel/joint).
    pub fn encode(&self, mel: &Tensor) -> Result<Tensor> {
        self.forward_through(mel, self.layers.len())
    }

    /// Полный энкодер: mel [B,n_mels,T] → encoded [B, d_model, T'] (как NeMo выход, D×T).
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let x = self.forward_through(mel, self.layers.len())?; // [B,T,D]
        x.permute((0, 2, 1)) // → [B,D,T]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subsampled_len_matches() {
        assert_eq!(subsampled_len(301), 39);
        assert_eq!(subsampled_len(128), 17);
    }
}
