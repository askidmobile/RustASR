//! RNN-T joint network (без TDT durations).
//!
//! logits = joint_net.2( relu( enc(f) + pred(g) ) )
//! enc: Linear(1024→640), pred: Linear(640→640), joint_net.2: Linear(640→13088).

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::config::JointConfig;

pub struct JointNet {
    enc: Linear,
    pred: Linear,
    out: Linear,
}

impl JointNet {
    pub fn load(cfg: &JointConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            enc: candle_nn::linear(cfg.encoder_hidden, cfg.joint_hidden, vb.pp("enc"))?,
            pred: candle_nn::linear(cfg.pred_hidden, cfg.joint_hidden, vb.pp("pred"))?,
            out: candle_nn::linear(cfg.joint_hidden, cfg.output_dim, vb.pp("joint_net.2"))?,
        })
    }

    /// Спроецировать энкодер заранее: f [.., enc_hidden] → [.., joint_hidden].
    pub fn project_enc(&self, f: &Tensor) -> Result<Tensor> {
        self.enc.forward(f)
    }

    /// logits для одного шага: enc_proj [joint_hidden] (уже спроецирован joint.enc),
    /// g [pred_hidden] → [output_dim]. Принимает 1D или 2D.
    pub fn step_logits(&self, enc_proj: &Tensor, g: &Tensor) -> Result<Tensor> {
        let to2d = |t: &Tensor| -> Result<Tensor> {
            if t.dims().len() == 1 { t.unsqueeze(0) } else { Ok(t.clone()) }
        };
        let e = to2d(enc_proj)?; // [1, joint_hidden]
        let g = to2d(g)?; // [1, pred_hidden]
        let p = self.pred.forward(&g)?; // [1, joint_hidden]
        let h = e.broadcast_add(&p)?.relu()?;
        let out = self.out.forward(&h)?; // [1, output_dim]
        out.squeeze(0) // [output_dim]
    }
}
