//! Language-ID prompt fusion (prompt_kernel) — НОВЫЙ компонент vs Parakeet.
//!
//! NeMo: concat(encoder[B,T,D], lang_onehot[B,T,num_prompts]) → prompt_kernel → [B,T,D].
//! prompt_kernel = Sequential(Linear(1152→2048), ReLU, Linear(2048→1024)).
//! Язык ru-RU → one-hot позиция 11 (num_prompts=128).

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::config::PromptConfig;

pub struct PromptKernel {
    l0: Linear, // [2048, 1152]
    l2: Linear, // [1024, 2048]
    num_prompts: usize,
    out_features: usize,
}

impl PromptKernel {
    pub fn load(cfg: &PromptConfig, vb: VarBuilder) -> Result<Self> {
        // prompt_kernel.0 (Linear+bias), prompt_kernel.2 (Linear+bias)
        let l0 = candle_nn::linear(cfg.in_features, cfg.hidden, vb.pp("0"))?;
        let l2 = candle_nn::linear(cfg.hidden, cfg.out_features, vb.pp("2"))?;
        Ok(Self {
            l0,
            l2,
            num_prompts: cfg.num_prompts,
            out_features: cfg.out_features,
        })
    }

    /// Прогон уже сконкатенированного входа [B,T,1152] → [B,T,1024] (для parity).
    pub fn forward_fused(&self, fused: &Tensor) -> Result<Tensor> {
        let h = self.l0.forward(fused)?.relu()?;
        self.l2.forward(&h)
    }

    /// Слить encoder-выход [B,T,D] с one-hot языка и прогнать kernel → [B,T,D].
    pub fn forward(&self, encoded: &Tensor, lang_idx: usize) -> Result<Tensor> {
        let (b, t, _d) = encoded.dims3()?;
        let dev = encoded.device();
        // one-hot [num_prompts], позиция lang_idx = 1; dtype = как у encoded (F16 в проде),
        // иначе Tensor::cat упрётся в dtype-mismatch F32(one-hot) vs F16(encoded).
        let mut oh = vec![0.0f32; self.num_prompts];
        oh[lang_idx] = 1.0;
        let oh = Tensor::from_vec(oh, (1, 1, self.num_prompts), dev)?
            .to_dtype(encoded.dtype())?
            .broadcast_as((b, t, self.num_prompts))?;
        let fused = Tensor::cat(&[encoded, &oh], 2)?; // [B,T,D+num_prompts]
        let out = self.forward_fused(&fused)?;
        debug_assert_eq!(out.dim(2).unwrap_or(self.out_features), self.out_features);
        Ok(out)
    }
}
