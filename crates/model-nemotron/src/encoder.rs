//! FastConformer cache-aware энкодер Nemotron 3.5 ASR.
//!
//! Структура идентична Parakeet FastConformer, КРОМЕ:
//! - **causal subsampling**: dw_striding с асимметричным паддингом (left=k-1=2, right=s-1=1)
//!   на ОБА измерения (time, freq) → 301→39 (time), 128→17 (freq).
//! - **causal depthwise conv** в conformer conv-модуле (pad left=k-1=8, right=0).
//! - **conv-norm = LayerNorm** (ключ `conv.batch_norm.{weight,bias}`, без running stats).
//! - **chunked_limited attention** [56,13] (маска) — TODO.
//!
//! Ф3, поэтапно с parity против tmp/parity/nemotron_*.npy:
//!   [x] subsampling → pre_encode_out
//!   [ ] conformer layers (+ attention mask) → layer0_out / encoded_pre_prompt

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::EncoderConfig;

/// Causal dw_striding субдискретизация (×8).
///
/// nn.Sequential из весов:
/// - conv.0: CausalConv2d(1→C, 3×3, stride 2)        + ReLU
/// - conv.2: CausalConv2d(C, 3×3, stride 2, dw)
/// - conv.3: Conv2d(C→C, 1×1)                         + ReLU
/// - conv.5: CausalConv2d(C, 3×3, stride 2, dw)
/// - conv.6: Conv2d(C→C, 1×1)                         + ReLU
/// - out:    Linear(C * freq_out, d_model)
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
    out: candle_nn::Linear,
    channels: usize,
}

/// Асимметричный causal-паддинг (left=2, right=1) на dims time(2) и freq(3), затем conv2d stride 2.
fn causal_conv2d_s2(
    x: &Tensor,
    w: &Tensor,
    b: &Tensor,
    groups: usize,
) -> Result<Tensor> {
    // CausalConv2D: F.pad(x, (left, right, left, right)) с left=k-1=2, right=s-1=1
    let x = x.pad_with_zeros(2, 2, 1)?; // time
    let x = x.pad_with_zeros(3, 2, 1)?; // freq
    let y = x.conv2d(w, 0, 2, 1, groups)?;
    let b = b.reshape((1, (), 1, 1))?;
    y.broadcast_add(&b)
}

/// Pointwise 1×1 conv2d (stride 1, no pad) + bias.
fn pointwise(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    let y = x.conv2d(w, 0, 1, 1, 1)?;
    let b = b.reshape((1, (), 1, 1))?;
    y.broadcast_add(&b)
}

impl Subsampling {
    pub fn load(config: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let c = config.subsampling_conv_channels; // 256
        let conv = vb.pp("conv");
        let conv0_w = conv.get((c, 1, 3, 3), "0.weight")?;
        let conv0_b = conv.get(c, "0.bias")?;
        let conv2_w = conv.get((c, 1, 3, 3), "2.weight")?;
        let conv2_b = conv.get(c, "2.bias")?;
        let conv3_w = conv.get((c, c, 1, 1), "3.weight")?;
        let conv3_b = conv.get(c, "3.bias")?;
        let conv5_w = conv.get((c, 1, 3, 3), "5.weight")?;
        let conv5_b = conv.get(c, "5.bias")?;
        let conv6_w = conv.get((c, c, 1, 1), "6.weight")?;
        let conv6_b = conv.get(c, "6.bias")?;

        // freq_out: 128 → 65 → 33 → 17 (причинный паддинг 2,1 на каждом stride-2)
        let freq_out = subsampled_len(config.feat_in);
        let proj_in = c * freq_out;
        let out = candle_nn::linear(proj_in, config.d_model, vb.pp("out"))?;

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

    /// Вход: [B, 1, T, F] (time=H, freq=W). Выход: [B, T', d_model].
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = causal_conv2d_s2(x, &self.conv0_w, &self.conv0_b, 1)?.relu()?;
        let x = causal_conv2d_s2(&x, &self.conv2_w, &self.conv2_b, self.channels)?;
        let x = pointwise(&x, &self.conv3_w, &self.conv3_b)?.relu()?;
        let x = causal_conv2d_s2(&x, &self.conv5_w, &self.conv5_b, self.channels)?;
        let x = pointwise(&x, &self.conv6_w, &self.conv6_b)?.relu()?;
        // [B, C, T', F'] → [B, T', C, F'] → [B, T', C*F']
        let (b, _c, t, _f) = x.dims4()?;
        let x = x.permute((0, 2, 1, 3))?.contiguous()?.reshape((b, t, ()))?;
        self.out.forward(&x)
    }
}

/// Длина после 3× causal stride-2 conv (kernel 3, pad left=2 right=1):
/// out = floor((in + 3 - 3)/2) + 1 = floor(in/2) + 1, трижды.
pub fn subsampled_len(n: usize) -> usize {
    let mut x = n;
    for _ in 0..3 {
        x = x / 2 + 1;
    }
    x
}

/// FastConformer-энкодер (пока только subsampling; conformer-слои — следующий шаг Ф3).
pub struct NemotronEncoder {
    subsampling: Subsampling,
}

impl NemotronEncoder {
    pub fn load(config: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let subsampling = Subsampling::load(config, vb.pp("pre_encode"))?;
        Ok(Self { subsampling })
    }

    /// mel [B, n_mels, T] → subsampled [B, T', d_model] (выход pre_encode).
    pub fn forward_subsampling(&self, mel: &Tensor) -> Result<Tensor> {
        // [B, F, T] → [B, T, F] → [B, 1, T, F]
        let x = mel.permute((0, 2, 1))?.unsqueeze(1)?;
        self.subsampling.forward(&x)
    }
}

use candle_nn::Module;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subsampled_len_matches() {
        assert_eq!(subsampled_len(301), 39);
        assert_eq!(subsampled_len(128), 17);
    }
}
