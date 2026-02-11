//! LSTM Prediction Network для TDT-декодера.
//!
//! Архитектура:
//! - Embedding(vocab_size, embed_dim)
//! - 2-layer LSTM(embed_dim, hidden_size)
//!
//! Весовые ключи:
//! - decoder.prediction.embed.weight: [8193, 640]
//! - decoder.prediction.dec_rnn.lstm.weight_ih_l{i}: [4*hidden, input]
//! - decoder.prediction.dec_rnn.lstm.weight_hh_l{i}: [4*hidden, hidden]
//! - decoder.prediction.dec_rnn.lstm.bias_ih_l{i}: [4*hidden]
//! - decoder.prediction.dec_rnn.lstm.bias_hh_l{i}: [4*hidden]

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::VarBuilder;
use tracing::debug;

use crate::config::DecoderConfig;

/// Один слой LSTM.
///
/// Формулы:
/// gates = x @ W_ih^T + h @ W_hh^T + b_ih + b_hh
/// i, f, g, o = gates.chunk(4)
/// c = sigmoid(f) * c_prev + sigmoid(i) * tanh(g)
/// h = sigmoid(o) * tanh(c)
struct LstmLayer {
    weight_ih: Tensor, // [4*hidden, input_size]
    weight_hh: Tensor, // [4*hidden, hidden_size]
    bias_ih: Tensor,   // [4*hidden]
    bias_hh: Tensor,   // [4*hidden]
    hidden_size: usize,
}

impl LstmLayer {
    fn load(input_size: usize, hidden_size: usize, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let gate_size = 4 * hidden_size;
        let weight_ih = vb.get((gate_size, input_size), &format!("weight_ih_l{layer_idx}"))?;
        let weight_hh = vb.get((gate_size, hidden_size), &format!("weight_hh_l{layer_idx}"))?;
        let bias_ih = vb.get(gate_size, &format!("bias_ih_l{layer_idx}"))?;
        let bias_hh = vb.get(gate_size, &format!("bias_hh_l{layer_idx}"))?;
        Ok(Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            hidden_size,
        })
    }

    /// Forward одного шага: (x [hidden], h [hidden], c [hidden]) → (h_new, c_new).
    fn step(&self, x: &Tensor, h: &Tensor, c: &Tensor) -> Result<(Tensor, Tensor)> {
        // Добавляем batch dim для matmul: [hidden] → [1, hidden]
        let x_2d = if x.dims().len() == 1 { x.unsqueeze(0)? } else { x.clone() };
        let h_2d = if h.dims().len() == 1 { h.unsqueeze(0)? } else { h.clone() };

        // gates = x @ W_ih^T + h @ W_hh^T + b_ih + b_hh
        let gates = x_2d
            .matmul(&self.weight_ih.t()?)?
            .broadcast_add(&self.bias_ih)?
            .broadcast_add(&h_2d.matmul(&self.weight_hh.t()?)?)?
            .broadcast_add(&self.bias_hh)?;

        // Убираем batch dim для последующих операций
        let gates = gates.squeeze(0)?;

        let hs = self.hidden_size;

        // Разбиваем на 4 части: input, forget, cell, output gates
        let i_gate = gates.narrow(D::Minus1, 0, hs)?;
        let f_gate = gates.narrow(D::Minus1, hs, hs)?;
        let g_gate = gates.narrow(D::Minus1, 2 * hs, hs)?;
        let o_gate = gates.narrow(D::Minus1, 3 * hs, hs)?;

        let i_gate = candle_nn::Activation::Sigmoid.forward(&i_gate)?;
        let f_gate = candle_nn::Activation::Sigmoid.forward(&f_gate)?;
        let g_gate = g_gate.tanh()?;
        let o_gate = candle_nn::Activation::Sigmoid.forward(&o_gate)?;

        // c_new = f * c + i * g
        let c_new = (f_gate * c)?.broadcast_add(&(i_gate * g_gate)?)?;
        // h_new = o * tanh(c_new)
        let h_new = (o_gate * c_new.tanh()?)?;

        Ok((h_new, c_new))
    }
}

/// Состояние LSTM (скрытое состояние и ячейка для каждого слоя).
pub struct LstmState {
    /// h[i]: [hidden_size] для каждого слоя.
    pub h: Vec<Tensor>,
    /// c[i]: [hidden_size] для каждого слоя.
    pub c: Vec<Tensor>,
}

impl LstmState {
    /// Создать нулевое начальное состояние.
    pub fn zeros(num_layers: usize, hidden_size: usize, device: &Device) -> Result<Self> {
        let mut h = Vec::with_capacity(num_layers);
        let mut c = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            h.push(Tensor::zeros(hidden_size, DType::F32, device)?);
            c.push(Tensor::zeros(hidden_size, DType::F32, device)?);
        }
        Ok(Self { h, c })
    }
}

/// Prediction Network: Embedding + N-layer LSTM.
pub struct PredictionNet {
    embedding: Tensor, // [vocab_size, embed_dim]
    lstm_layers: Vec<LstmLayer>,
    hidden_size: usize,
    num_layers: usize,
    blank_idx: usize,
}

impl PredictionNet {
    /// Загрузка из safetensors.
    ///
    /// Ключи:
    /// - prediction.embed.weight
    /// - prediction.dec_rnn.lstm.{weight_ih_l0, weight_hh_l0, ...}
    pub fn load(config: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let pred_vb = vb.pp("prediction");

        // Embedding
        let embedding = pred_vb.get((config.vocab_size, config.embed_dim), "embed.weight")?;

        // LSTM layers
        let lstm_vb = pred_vb.pp("dec_rnn").pp("lstm");
        let mut lstm_layers = Vec::with_capacity(config.num_lstm_layers);
        for i in 0..config.num_lstm_layers {
            let input_size = if i == 0 { config.embed_dim } else { config.pred_hidden };
            let layer = LstmLayer::load(input_size, config.pred_hidden, i, lstm_vb.clone())?;
            lstm_layers.push(layer);
        }

        debug!(
            "PredictionNet загружен: vocab={}, embed={}, LSTM {}×{}",
            config.vocab_size, config.embed_dim, config.num_lstm_layers, config.pred_hidden
        );

        Ok(Self {
            embedding,
            lstm_layers,
            hidden_size: config.pred_hidden,
            num_layers: config.num_lstm_layers,
            blank_idx: config.blank_idx,
        })
    }

    /// Начальное состояние LSTM.
    pub fn initial_state(&self, device: &Device) -> Result<LstmState> {
        LstmState::zeros(self.num_layers, self.hidden_size, device)
    }

    /// Forward одного шага: token_id → (output [hidden], new_state).
    ///
    /// При blank-токене эмбеддинг — нулевой вектор.
    pub fn step(&self, token_id: u32, state: &LstmState) -> Result<(Tensor, LstmState)> {
        let device = self.embedding.device();

        // Embedding lookup
        let embed = if (token_id as usize) == self.blank_idx {
            // Для blank используем нулевой вектор (как при инициализации)
            Tensor::zeros(self.embedding.dim(1)?, DType::F32, device)?
        } else {
            let idx = Tensor::new(&[token_id], device)?;
            self.embedding.embedding(&idx)?.squeeze(0)?
        };

        // Прогнать через LSTM слои
        let mut x = embed;
        let mut new_h = Vec::with_capacity(self.num_layers);
        let mut new_c = Vec::with_capacity(self.num_layers);

        for (i, layer) in self.lstm_layers.iter().enumerate() {
            let (h_new, c_new) = layer.step(&x, &state.h[i], &state.c[i])?;
            x = h_new.clone();
            new_h.push(h_new);
            new_c.push(c_new);
        }

        let new_state = LstmState {
            h: new_h,
            c: new_c,
        };

        Ok((x, new_state))
    }
}
