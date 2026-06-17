//! RNN-T prediction network: Embedding + 2-layer LSTM (F32).
//! Адаптация Parakeet PredictionNet под VarBuilder (без quantization).
//!
//! Ключи: decoder.prediction.embed.weight [13088,640],
//!        decoder.prediction.dec_rnn.lstm.{weight,bias}_{ih,hh}_l{0,1}.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::VarBuilder;

use crate::config::DecoderConfig;

struct LstmLayer {
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Tensor,
    bias_hh: Tensor,
    hidden_size: usize,
}

impl LstmLayer {
    fn load(input_size: usize, hidden_size: usize, i: usize, vb: &VarBuilder) -> Result<Self> {
        let g = 4 * hidden_size;
        Ok(Self {
            weight_ih: vb.get((g, input_size), &format!("weight_ih_l{i}"))?,
            weight_hh: vb.get((g, hidden_size), &format!("weight_hh_l{i}"))?,
            bias_ih: vb.get(g, &format!("bias_ih_l{i}"))?,
            bias_hh: vb.get(g, &format!("bias_hh_l{i}"))?,
            hidden_size,
        })
    }

    /// (x [hidden], h [hidden], c [hidden]) → (h_new, c_new).
    fn step(&self, x: &Tensor, h: &Tensor, c: &Tensor) -> Result<(Tensor, Tensor)> {
        let x2 = if x.dims().len() == 1 { x.unsqueeze(0)? } else { x.clone() };
        let h2 = if h.dims().len() == 1 { h.unsqueeze(0)? } else { h.clone() };
        let gates = x2
            .matmul(&self.weight_ih.t()?)?
            .broadcast_add(&self.bias_ih)?
            .broadcast_add(&h2.matmul(&self.weight_hh.t()?)?)?
            .broadcast_add(&self.bias_hh)?
            .squeeze(0)?;
        let hs = self.hidden_size;
        let i = gates.narrow(D::Minus1, 0, hs)?;
        let f = gates.narrow(D::Minus1, hs, hs)?;
        let g = gates.narrow(D::Minus1, 2 * hs, hs)?;
        let o = gates.narrow(D::Minus1, 3 * hs, hs)?;
        let i = i.neg()?.exp()?.affine(1.0, 1.0)?.recip()?;
        let f = f.neg()?.exp()?.affine(1.0, 1.0)?.recip()?;
        let g = g.tanh()?;
        let o = o.neg()?.exp()?.affine(1.0, 1.0)?.recip()?;
        let c_new = (f * c)?.broadcast_add(&(i * g)?)?;
        let h_new = (o * c_new.tanh()?)?;
        Ok((h_new, c_new))
    }
}

#[derive(Clone)]
pub struct LstmState {
    pub h: Vec<Tensor>,
    pub c: Vec<Tensor>,
}

impl LstmState {
    pub fn zeros(num_layers: usize, hidden: usize, device: &Device) -> Result<Self> {
        let mut h = Vec::with_capacity(num_layers);
        let mut c = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            h.push(Tensor::zeros(hidden, DType::F32, device)?);
            c.push(Tensor::zeros(hidden, DType::F32, device)?);
        }
        Ok(Self { h, c })
    }
}

pub struct PredictionNet {
    embedding: Tensor, // [embed_rows, pred_hidden]
    lstm: Vec<LstmLayer>,
    hidden_size: usize,
    num_layers: usize,
}

impl PredictionNet {
    pub fn load(config: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let pred = vb.pp("prediction");
        let embedding = pred.get((config.embed_rows, config.pred_hidden), "embed.weight")?;
        let lstm_vb = pred.pp("dec_rnn").pp("lstm");
        let mut lstm = Vec::with_capacity(config.pred_rnn_layers);
        for i in 0..config.pred_rnn_layers {
            let input = config.pred_hidden; // embed_dim == pred_hidden == 640
            lstm.push(LstmLayer::load(input, config.pred_hidden, i, &lstm_vb)?);
        }
        Ok(Self {
            embedding,
            lstm,
            hidden_size: config.pred_hidden,
            num_layers: config.pred_rnn_layers,
        })
    }

    pub fn initial_state(&self, device: &Device) -> Result<LstmState> {
        LstmState::zeros(self.num_layers, self.hidden_size, device)
    }

    /// token_id → (pred_out [hidden], new_state). blank → embedding row (padding_idx, нули).
    pub fn step(&self, token_id: u32, state: &LstmState) -> Result<(Tensor, LstmState)> {
        let device = self.embedding.device();
        let idx = Tensor::new(&[token_id], device)?;
        let embed = self.embedding.embedding(&idx)?.squeeze(0)?;
        let mut x = embed;
        let mut new_h = Vec::with_capacity(self.num_layers);
        let mut new_c = Vec::with_capacity(self.num_layers);
        for (i, layer) in self.lstm.iter().enumerate() {
            let (h, c) = layer.step(&x, &state.h[i], &state.c[i])?;
            x = h.clone();
            new_h.push(h);
            new_c.push(c);
        }
        Ok((x, LstmState { h: new_h, c: new_c }))
    }
}
