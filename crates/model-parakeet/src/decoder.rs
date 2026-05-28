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

use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use tracing::debug;

use crate::config::DecoderConfig;
use crate::encoder::Weights;

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
    fn load(
        input_size: usize,
        hidden_size: usize,
        layer_idx: usize,
        target_dtype: DType,
        vb: Weights<'_>,
    ) -> Result<Self> {
        let gate_size = 4 * hidden_size;
        // LSTM веса — regular tensors. Принудительно cast в target_dtype чтобы
        // matmul state × weight были одного dtype (на Hybrid path LSTM weights
        // могут прийти из quantized backend через dequant F32, а state — F16).
        let weight_ih = vb
            .get_tensor((gate_size, input_size), &format!("weight_ih_l{layer_idx}"))?
            .to_dtype(target_dtype)?;
        let weight_hh = vb
            .get_tensor((gate_size, hidden_size), &format!("weight_hh_l{layer_idx}"))?
            .to_dtype(target_dtype)?;
        let bias_ih = vb
            .get_tensor(gate_size, &format!("bias_ih_l{layer_idx}"))?
            .to_dtype(target_dtype)?;
        let bias_hh = vb
            .get_tensor(gate_size, &format!("bias_hh_l{layer_idx}"))?
            .to_dtype(target_dtype)?;
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
        let x_2d = if x.dims().len() == 1 {
            x.unsqueeze(0)?
        } else {
            x.clone()
        };
        let h_2d = if h.dims().len() == 1 {
            h.unsqueeze(0)?
        } else {
            h.clone()
        };

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

        // sigmoid через базовые ops + .affine(1.0, 1.0) сохраняет dtype (vs `+ 1.0`
        // который промотит F16 в F32 mismatch).
        let i_gate = i_gate.neg()?.exp()?.affine(1.0, 1.0)?.recip()?;
        let f_gate = f_gate.neg()?.exp()?.affine(1.0, 1.0)?.recip()?;
        let g_gate = g_gate.tanh()?;
        let o_gate = o_gate.neg()?.exp()?.affine(1.0, 1.0)?.recip()?;

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
    /// Создать нулевое начальное состояние с указанным dtype (должен совпадать с весами).
    pub fn zeros(
        num_layers: usize,
        hidden_size: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let mut h = Vec::with_capacity(num_layers);
        let mut c = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            h.push(Tensor::zeros(hidden_size, dtype, device)?);
            c.push(Tensor::zeros(hidden_size, dtype, device)?);
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
}

impl PredictionNet {
    /// Загрузка из safetensors (legacy F16/F32 path).
    pub fn load(config: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        Self::load_from_weights(config, Weights::Standard(vb))
    }

    /// Универсальная загрузка из Standard или Quantized backend.
    /// LSTM работает в фиксированном dtype = F32 (numerical stability в gates,
    /// embedding/weights cast'нутся в F32). pred_out возвращается в F32,
    /// joint cast'нет в encoder dtype перед добавлением.
    pub fn load_from_weights(config: &DecoderConfig, vb: Weights<'_>) -> Result<Self> {
        Self::load_with_dtype(config, vb, DType::F32)
    }

    pub fn load_with_dtype(
        config: &DecoderConfig,
        vb: Weights<'_>,
        target_dtype: DType,
    ) -> Result<Self> {
        let pred_vb = vb.pp("prediction");

        // Embedding cast'нут в target_dtype для match с LSTM weights
        let embedding = pred_vb
            .get_tensor((config.vocab_size, config.embed_dim), "embed.weight")?
            .to_dtype(target_dtype)?;

        // LSTM layers
        let lstm_vb = pred_vb.pp("dec_rnn").pp("lstm");
        let mut lstm_layers = Vec::with_capacity(config.num_lstm_layers);
        for i in 0..config.num_lstm_layers {
            let input_size = if i == 0 {
                config.embed_dim
            } else {
                config.pred_hidden
            };
            let layer =
                LstmLayer::load(input_size, config.pred_hidden, i, target_dtype, lstm_vb.clone())?;
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
        })
    }

    /// Начальное состояние LSTM (dtype совпадает с весами для matmul).
    pub fn initial_state(&self, device: &Device) -> Result<LstmState> {
        let dtype = self.embedding.dtype();
        LstmState::zeros(self.num_layers, self.hidden_size, dtype, device)
    }
    // Note: dtype-aware initial_state нужен на случай если когда-то добавим
    // F16 path. Сейчас всё F32, но код безопасен и для будущих изменений.

    /// Forward одного шага: token_id → (output [hidden], new_state).
    ///
    /// При blank-токене эмбеддинг — нулевой вектор.
    pub fn step(&self, token_id: u32, state: &LstmState) -> Result<(Tensor, LstmState)> {
        let device = self.embedding.device();

        // Embedding lookup (включая blank — у него свой выученный вектор)
        let idx = Tensor::new(&[token_id], device)?;
        let embed = self.embedding.embedding(&idx)?.squeeze(0)?;

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

        let new_state = LstmState { h: new_h, c: new_c };

        Ok((x, new_state))
    }
}
