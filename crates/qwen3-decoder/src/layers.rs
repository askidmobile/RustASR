//! Neural network layers for Qwen3 decoder.

use candle_core::{D, DType, IndexOp, Result, Tensor};
use candle_nn::{Module, VarBuilder};
use candle_transformers::quantized_nn;
use candle_transformers::quantized_var_builder as quantized_vb;

use crate::config::Qwen3Config;

#[derive(Clone)]
pub enum Weights<'a> {
    Standard(VarBuilder<'a>),
    Quantized(quantized_vb::VarBuilder),
}

impl<'a> Weights<'a> {
    pub fn pp<S: ToString>(&self, s: S) -> Self {
        match self {
            Self::Standard(vb) => Self::Standard(vb.pp(s)),
            Self::Quantized(vb) => Self::Quantized(vb.pp(s)),
        }
    }
}

#[derive(Debug, Clone)]
pub enum LinearLayer {
    Standard(candle_nn::Linear),
    Quantized(quantized_nn::Linear),
}

impl LinearLayer {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Standard(l) => l.forward(x),
            Self::Quantized(l) => {
                // Квантованные матмулы в candle часто работают в f32.
                // Для совместимости с остальным графом приводим вход/выход к dtype x.
                let x_dtype = x.dtype();
                let x_f32 = if x_dtype != DType::F32 {
                    x.to_dtype(DType::F32)?
                } else {
                    x.clone()
                };
                let y = l.forward(&x_f32)?;
                if y.dtype() != x_dtype {
                    y.to_dtype(x_dtype)
                } else {
                    Ok(y)
                }
            }
        }
    }
}

fn linear_no_bias(in_dim: usize, out_dim: usize, vb: Weights<'_>) -> Result<LinearLayer> {
    match vb {
        Weights::Standard(vb) => Ok(LinearLayer::Standard(candle_nn::linear_no_bias(
            in_dim, out_dim, vb,
        )?)),
        Weights::Quantized(vb) => Ok(LinearLayer::Quantized(quantized_nn::linear_no_bias(
            in_dim, out_dim, vb,
        )?)),
    }
}

/// RMS Normalization layer.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((size,), "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn from_weight(weight: Tensor, eps: f64) -> Result<Self> {
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // HF-совместимо: вычисления в f32, затем каст обратно.
        let input_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let x_normed = x_normed.to_dtype(input_dtype)?;
        let w = if self.weight.dtype() != input_dtype {
            self.weight.to_dtype(input_dtype)?
        } else {
            self.weight.clone()
        };
        x_normed.broadcast_mul(&w)
    }
}

/// Rotary Position Embedding.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        theta: f64,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / (theta.powf(i as f64 / head_dim as f64) as f32))
            .collect();

        let inv_freq = Tensor::new(inv_freq, device)?;
        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let positions = Tensor::new(positions, device)?.unsqueeze(1)?;

        let freqs = positions.matmul(&inv_freq.unsqueeze(0)?)?; // [seq, head_dim/2]

        // HF Qwen3-ASR использует rotate_half по половинам вектора,
        // поэтому cos/sin должны быть размера [seq, head_dim].
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        let cos = emb.cos()?;
        let sin = emb.sin()?;

        Ok(Self { cos, sin })
    }

    pub fn apply(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        let seq_len = x.dim(2)?;
        let x_dtype = x.dtype();

        // cos/sin: [seq, head_dim] -> [1, 1, seq, head_dim]
        let cos = self
            .cos
            .i(start_pos..start_pos + seq_len)?
            .to_dtype(x_dtype)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let sin = self
            .sin
            .i(start_pos..start_pos + seq_len)?
            .to_dtype(x_dtype)?
            .unsqueeze(0)?
            .unsqueeze(0)?;

        // rotate_half: cat(-x2, x1)
        let head_dim = x.dim(3)?;
        let half = head_dim / 2;
        let x1 = x.i((.., .., .., 0..half))?;
        let x2 = x.i((.., .., .., half..head_dim))?;
        let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;

        x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?
    }
}

/// Grouped Query Attention layer.
#[derive(Debug, Clone)]
pub struct Attention {
    q_proj: LinearLayer,
    k_proj: LinearLayer,
    v_proj: LinearLayer,
    o_proj: LinearLayer,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: RotaryEmbedding,
}

impl Attention {
    pub fn new(config: &Qwen3Config, vb: Weights<'_>, rope: RotaryEmbedding) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

        let q_proj = linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        let q_norm = match vb.pp("q_norm") {
            Weights::Standard(vb) => RmsNorm::new(head_dim, config.rms_norm_eps, vb)?,
            Weights::Quantized(vb) => {
                let target_dtype = if vb.device().is_metal() || vb.device().is_cuda() {
                    DType::BF16
                } else {
                    DType::F32
                };
                let mut w = vb.get((head_dim,), "weight")?.dequantize(vb.device())?;
                if w.dtype() != target_dtype {
                    w = w.to_dtype(target_dtype)?;
                }
                RmsNorm::from_weight(w, config.rms_norm_eps)?
            }
        };
        let k_norm = match vb.pp("k_norm") {
            Weights::Standard(vb) => RmsNorm::new(head_dim, config.rms_norm_eps, vb)?,
            Weights::Quantized(vb) => {
                let target_dtype = if vb.device().is_metal() || vb.device().is_cuda() {
                    DType::BF16
                } else {
                    DType::F32
                };
                let mut w = vb.get((head_dim,), "weight")?.dequantize(vb.device())?;
                if w.dtype() != target_dtype {
                    w = w.to_dtype(target_dtype)?;
                }
                RmsNorm::from_weight(w, config.rms_norm_eps)?
            }
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            rope,
        })
    }

    pub fn forward(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        self.forward_with_cache(x, start_pos, None)
    }

    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        start_pos: usize,
        mut cache: Option<&mut crate::cache::LayerKvCache>,
    ) -> Result<Tensor> {
        let debug = asr_core::debug::enabled();
        let (batch_size, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;

        // Apply Q/K normalization
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Apply RoPE
        let q = self.rope.apply(&q, start_pos)?;
        let k = self.rope.apply(&k, start_pos)?;

        if debug {
            eprintln!(
                "DEBUG attention: q dtype={:?}, k dtype={:?}, v dtype={:?}",
                q.dtype(),
                k.dtype(),
                v.dtype()
            );
        }

        // KV-cache: если передан, обновляем и используем накопленные K/V.
        let (k, v, use_causal_mask) = if let Some(layer_cache) = cache.as_deref_mut() {
            if layer_cache.k.is_none() {
                // Префилл (кеш пустой). Сохраняем K/V всего префикса.
                layer_cache.k = Some(k.clone());
                layer_cache.v = Some(v.clone());
                (k, v, true)
            } else {
                // Декод: добавляем новые K/V в конец.
                let ck = layer_cache.k.as_ref().unwrap();
                let cv = layer_cache.v.as_ref().unwrap();
                let new_k = Tensor::cat(&[ck, &k], 2)?;
                let new_v = Tensor::cat(&[cv, &v], 2)?;
                layer_cache.k = Some(new_k);
                layer_cache.v = Some(new_v);
                (
                    layer_cache.k.as_ref().unwrap().clone(),
                    layer_cache.v.as_ref().unwrap().clone(),
                    false,
                )
            }
        } else {
            (k, v, true)
        };

        // GQA: repeat k/v heads so that each KV head is repeated `n_rep` times.
        // Нужен порядок: [kv0,kv0, kv1,kv1, ...], как в HF `repeat_kv`.
        let kv_repeat = self.num_heads / self.num_kv_heads;
        let k = Self::repeat_kv(&k, kv_repeat)?;
        let v = Self::repeat_kv(&v, kv_repeat)?;

        // Attention
        let scale = (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.transpose(2, 3)?)? / scale)?;
        if debug {
            eprintln!("DEBUG attention: attn dtype={:?}", attn.dtype());
        }

        // Префилл требует causal mask. В decode (seq_len=1 с KV-cache) он не нужен.
        let attn = if use_causal_mask {
            let causal_mask = Self::create_causal_mask(seq_len, attn.device(), attn.dtype())?;
            if debug {
                eprintln!(
                    "DEBUG attention: causal_mask dtype={:?}",
                    causal_mask.dtype()
                );
            }
            attn.broadcast_add(&causal_mask)?
        } else {
            attn
        };

        // HF делает softmax в float32 для стабильности.
        let attn_f32 = attn.to_dtype(DType::F32)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn_f32)?.to_dtype(attn.dtype())?;
        let out = attn.matmul(&v)?;

        // Reshape back
        let out = out.transpose(1, 2)?.contiguous()?;
        let out = out.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&out)
    }

    /// Create causal attention mask.
    /// Returns mask of shape [1, 1, seq_len, seq_len] with -inf for future positions.
    fn create_causal_mask(
        seq_len: usize,
        device: &candle_core::Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
            .collect();

        let mask = Tensor::from_vec(mask, (seq_len, seq_len), device)?;
        mask.unsqueeze(0)?.unsqueeze(0)?.to_dtype(dtype)
    }

    fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            return Ok(x.clone());
        }
        let (b, kv, s, d) = x.dims4()?;
        let mut parts: Vec<Tensor> = Vec::with_capacity(kv * n_rep);
        for i in 0..kv {
            let head = x.i((.., i..i + 1, .., ..))?;
            for _ in 0..n_rep {
                parts.push(head.clone());
            }
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        let out = Tensor::cat(refs.as_slice(), 1)?;
        // [b, kv*n_rep, s, d]
        let _ = b;
        let _ = s;
        let _ = d;
        Ok(out)
    }
}

/// SwiGLU MLP layer.
#[derive(Debug, Clone)]
pub struct MLP {
    gate_proj: LinearLayer,
    up_proj: LinearLayer,
    down_proj: LinearLayer,
}

impl MLP {
    pub fn new(config: &Qwen3Config, vb: Weights<'_>) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// Transformer decoder layer.
#[derive(Debug, Clone)]
pub struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    pub fn new(config: &Qwen3Config, vb: Weights<'_>, rope: RotaryEmbedding) -> Result<Self> {
        let self_attn = Attention::new(config, vb.pp("self_attn"), rope)?;
        let mlp = MLP::new(config, vb.pp("mlp"))?;
        let input_layernorm = match vb.pp("input_layernorm") {
            Weights::Standard(vb) => RmsNorm::new(config.hidden_size, config.rms_norm_eps, vb)?,
            Weights::Quantized(vb) => {
                let target_dtype = if vb.device().is_metal() || vb.device().is_cuda() {
                    DType::BF16
                } else {
                    DType::F32
                };
                let mut w = vb
                    .get((config.hidden_size,), "weight")?
                    .dequantize(vb.device())?;
                if w.dtype() != target_dtype {
                    w = w.to_dtype(target_dtype)?;
                }
                RmsNorm::from_weight(w, config.rms_norm_eps)?
            }
        };
        let post_attention_layernorm = match vb.pp("post_attention_layernorm") {
            Weights::Standard(vb) => RmsNorm::new(config.hidden_size, config.rms_norm_eps, vb)?,
            Weights::Quantized(vb) => {
                let target_dtype = if vb.device().is_metal() || vb.device().is_cuda() {
                    DType::BF16
                } else {
                    DType::F32
                };
                let mut w = vb
                    .get((config.hidden_size,), "weight")?
                    .dequantize(vb.device())?;
                if w.dtype() != target_dtype {
                    w = w.to_dtype(target_dtype)?;
                }
                RmsNorm::from_weight(w, config.rms_norm_eps)?
            }
        };

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        self.forward_with_cache(x, start_pos, None)
    }

    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        start_pos: usize,
        cache: Option<&mut crate::cache::LayerKvCache>,
    ) -> Result<Tensor> {
        // Pre-norm attention
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward_with_cache(&x, start_pos, cache)?;
        let x = (residual + x)?;

        // Pre-norm MLP
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        residual + x
    }
}
