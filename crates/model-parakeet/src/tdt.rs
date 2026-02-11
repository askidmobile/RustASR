//! TDT (Token-and-Duration Transducer) greedy decoding.
//!
//! Алгоритм:
//! 1. Инициализировать LSTM-состояние нулями, token = blank
//! 2. Для каждого временного фрейма энкодера:
//!    a. Получить pred_out от prediction network
//!    b. Вычислить joint(enc_out[t], pred_out) → token_logits, dur_logits
//!    c. k = argmax(token_logits), dur_idx = argmax(dur_logits)
//!    d. skip = durations[dur_idx]
//!    e. Если k == blank: advance по skip (minimum 1)
//!    f. Если k != blank: добавить к гипотезе, обновить состояние
//!       - Если skip == 0: остаёмся на фрейме (до max_symbols_per_step)
//!       - Если skip > 0: advance на skip фреймов

use candle_core::{IndexOp, Result, Tensor, D};
use tracing::debug;

use crate::config::TdtConfig;
use crate::decoder::PredictionNet;
use crate::joint::JointNetwork;

/// Результат TDT-декодирования.
pub struct TdtResult {
    /// Список token ID (без blank).
    pub tokens: Vec<u32>,
}

/// TDT greedy decoder.
pub struct TdtGreedyDecoder {
    durations: Vec<usize>,
    max_symbols_per_step: usize,
    blank_idx: usize,
}

impl TdtGreedyDecoder {
    pub fn new(config: &TdtConfig, blank_idx: usize) -> Self {
        Self {
            durations: config.durations.clone(),
            max_symbols_per_step: config.max_symbols_per_step,
            blank_idx,
        }
    }

    /// Greedy TDT декодирование.
    ///
    /// `encoder_output`: [T, d_model] — выход энкодера (без batch dim).
    pub fn decode(
        &self,
        encoder_output: &Tensor,
        prediction_net: &PredictionNet,
        joint: &JointNetwork,
    ) -> Result<TdtResult> {
        let t_total = encoder_output.dim(0)?;
        let device = encoder_output.device();

        debug!("TDT decode: {} фреймов энкодера", t_total);

        let mut hypothesis: Vec<u32> = Vec::new();
        let mut state = prediction_net.initial_state(device)?;
        let mut last_token: u32 = self.blank_idx as u32;
        let mut time_idx: usize = 0;
        let mut step_count: usize = 0;

        while time_idx < t_total {
            let enc_frame = encoder_output.i(time_idx)?; // [d_model]

            // Prediction network: пропустить текущий токен
            let (pred_out, state_next) = prediction_net.step(last_token, &state)?;

            // Joint network: получить logits
            let (token_logits, dur_logits) = joint.forward(&enc_frame, &pred_out)?;

            // Argmax для токена и длительности
            let k = token_logits
                .argmax(D::Minus1)?
                .to_scalar::<u32>()?;
            let dur_idx = dur_logits
                .argmax(D::Minus1)?
                .to_scalar::<u32>()? as usize;

            // Debug: показать первые 5 шагов
            if step_count < 5 {
                let _token_max = token_logits.max(D::Minus1)?.to_scalar::<f32>()?;
                let blank_score = token_logits.i(self.blank_idx)?.to_scalar::<f32>()?;
                // Top-5 tokens
                let logits_vec: Vec<f32> = token_logits.to_vec1()?;
                let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let top5: Vec<String> = indexed[..5].iter().map(|(i, v)| format!("{}:{:.3}", i, v)).collect();
                debug!(
                    "TDT step {}: t={}/{}, k={}, dur_idx={}, blank={:.3}, top=[{}]",
                    step_count, time_idx, t_total, k, dur_idx, blank_score, top5.join(", ")
                );
            }
            step_count += 1;

            let skip = if dur_idx < self.durations.len() {
                self.durations[dur_idx]
            } else {
                1
            };

            if k as usize == self.blank_idx {
                // Blank: продвигаемся по времени (minimum 1)
                time_idx += skip.max(1);
                // Состояние НЕ обновляем (blank не меняет prediction net)
            } else {
                // Non-blank: добавляем токен
                hypothesis.push(k);
                last_token = k;
                state = state_next;

                if skip == 0 {
                    // Остаёмся на том же фрейме, можем выдать ещё символы
                    let mut symbols_added = 1;
                    while symbols_added < self.max_symbols_per_step {
                        let (pred_out2, state_next2) =
                            prediction_net.step(last_token, &state)?;
                        let (token_logits2, dur_logits2) =
                            joint.forward(&enc_frame, &pred_out2)?;

                        let k2 = token_logits2.argmax(D::Minus1)?.to_scalar::<u32>()?;
                        let dur_idx2 = dur_logits2.argmax(D::Minus1)?.to_scalar::<u32>()? as usize;
                        let skip2 = if dur_idx2 < self.durations.len() {
                            self.durations[dur_idx2]
                        } else {
                            1
                        };

                        if k2 as usize == self.blank_idx {
                            // Blank → advance
                            time_idx += skip2.max(1);
                            break;
                        }

                        hypothesis.push(k2);
                        last_token = k2;
                        state = state_next2;
                        symbols_added += 1;

                        if skip2 > 0 {
                            time_idx += skip2;
                            break;
                        }
                    }
                    // Если вышли по max_symbols — принудительно сдвигаемся на 1
                    if symbols_added >= self.max_symbols_per_step {
                        time_idx += 1;
                    }
                } else {
                    // skip > 0: advance по времени
                    time_idx += skip;
                }
            }
        }

        debug!("TDT decode: {} токенов гипотезы", hypothesis.len());

        Ok(TdtResult { tokens: hypothesis })
    }
}
