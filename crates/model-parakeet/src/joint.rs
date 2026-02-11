//! Joint Network для TDT.
//!
//! Объединяет выходы энкодера и предсказательной сети:
//! joint = ReLU(enc_proj(enc_out) + pred_proj(pred_out))
//! logits = output_linear(joint)
//!
//! Выход: [num_classes + num_durations] = [8193 + 5] = 8198
//!
//! Весовые ключи:
//! - joint.enc.weight: [640, 1024], joint.enc.bias: [640]
//! - joint.pred.weight: [640, 640], joint.pred.bias: [640]
//! - joint.joint_net.2.weight: [8198, 640], joint.joint_net.2.bias: [8198]

use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;
use tracing::debug;

use crate::config::JointConfig;

/// Линейный слой с bias.
struct Linear {
    weight: Tensor,
    bias: Tensor,
}

impl Linear {
    fn load(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((out_dim, in_dim), "weight")?;
        let bias = vb.get(out_dim, "bias")?;
        Ok(Self { weight, bias })
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.matmul(&self.weight.t()?)?.broadcast_add(&self.bias)
    }
}

/// Joint Network: enc_proj + pred_proj → ReLU → output_linear.
pub struct JointNetwork {
    enc_proj: Linear,
    pred_proj: Linear,
    output_linear: Linear,
    num_classes: usize,
    num_durations: usize,
}

impl JointNetwork {
    /// Загрузка из safetensors.
    pub fn load(config: &JointConfig, vb: VarBuilder) -> Result<Self> {
        let enc_proj = Linear::load(config.encoder_hidden, config.joint_hidden, vb.pp("enc"))?;
        let pred_proj = Linear::load(config.pred_hidden, config.joint_hidden, vb.pp("pred"))?;

        // output: joint_net.2 (после ReLU[0] и Dropout[1])
        // Пытаемся загрузить с индексом 2, иначе с индексом 1
        let output_linear = Linear::load(
            config.joint_hidden,
            config.output_dim,
            vb.pp("joint_net").pp("2"),
        )
        .or_else(|_| {
            Linear::load(
                config.joint_hidden,
                config.output_dim,
                vb.pp("joint_net").pp("1"),
            )
        })?;

        debug!(
            "JointNetwork загружен: enc_proj({} → {}), pred_proj({} → {}), output({} → {})",
            config.encoder_hidden,
            config.joint_hidden,
            config.pred_hidden,
            config.joint_hidden,
            config.joint_hidden,
            config.output_dim,
        );

        Ok(Self {
            enc_proj,
            pred_proj,
            output_linear,
            num_classes: config.num_classes,
            num_durations: config.num_durations,
        })
    }

    /// Forward: enc_out [D_enc], pred_out [D_pred] → (token_logits [num_classes], dur_logits [num_durations]).
    ///
    /// Поддерживает как 1D [D], так и 2D [1, D] входы.
    pub fn forward(&self, enc_out: &Tensor, pred_out: &Tensor) -> Result<(Tensor, Tensor)> {
        // Гарантируем 2D вход для candle_nn::Linear
        let enc_in = if enc_out.dims().len() == 1 { enc_out.unsqueeze(0)? } else { enc_out.clone() };
        let pred_in = if pred_out.dims().len() == 1 { pred_out.unsqueeze(0)? } else { pred_out.clone() };

        let enc_h = self.enc_proj.forward(&enc_in)?;
        let pred_h = self.pred_proj.forward(&pred_in)?;

        // ReLU(enc_h + pred_h)
        let joint = (enc_h + pred_h)?.relu()?;

        // Output logits
        let logits = self.output_linear.forward(&joint)?;

        // Squeeze обратно в 1D если вход был 1D
        let logits = logits.squeeze(0)?;

        // Разделить на token logits и duration logits
        let token_logits = logits.narrow(candle_core::D::Minus1, 0, self.num_classes)?;
        let dur_logits = logits.narrow(candle_core::D::Minus1, self.num_classes, self.num_durations)?;

        Ok((token_logits, dur_logits))
    }
}
