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

use candle_core::{D, IndexOp, Result, Tensor};
use tracing::debug;

use crate::config::TdtConfig;
use crate::decoder::PredictionNet;
use crate::joint::JointNetwork;

/// Результат TDT-декодирования.
pub struct TdtResult {
    /// Список token ID (без blank).
    pub tokens: Vec<u32>,
    /// Timestamps: (token_id, timestamp_ms) для каждого non-blank токена.
    pub timestamps: Vec<(u32, u64)>,
}

/// Гипотеза TDT beam-поиска (несёт собственное время из-за длительностей).
#[derive(Clone)]
struct TdtBeamHyp {
    tokens: Vec<u32>,
    timestamps: Vec<(u32, u64)>,
    state: crate::decoder::LstmState,
    pred_out: Tensor,
    time: usize,
    symbols_at_t: usize,
    score: f32,
}

fn log_softmax_vec(v: &[f32]) -> Vec<f32> {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f32;
    for &x in v {
        s += (x - m).exp();
    }
    let lse = m + s.ln();
    v.iter().map(|&x| x - lse).collect()
}

fn argmax(v: &[f32]) -> usize {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi
}

/// top-K индексов по убыванию значения, исключая `exclude` (blank).
fn top_k_indices(v: &[f32], k: usize, exclude: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).filter(|&i| i != exclude).collect();
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
    idx.truncate(k);
    idx
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
        self.decode_inner(encoder_output, prediction_net, joint, None)
    }

    /// TDT beam-search (token-beam + greedy-duration, per-hyp время).
    ///
    /// В отличие от RNN-T, длительность TDT (skip 0..4) рассинхронизирует гипотезы по
    /// времени, поэтому каждая гипотеза несёт собственный `time`. Расширяем по top-K
    /// токенам (+ blank), длительность берём argmax (greedy, она не зависит от токена).
    /// beam_width<=1 эквивалентно greedy.
    pub fn beam_decode(
        &self,
        encoder_output: &Tensor,
        prediction_net: &PredictionNet,
        joint: &JointNetwork,
        beam_width: usize,
    ) -> Result<TdtResult> {
        if beam_width <= 1 {
            return self.decode(encoder_output, prediction_net, joint);
        }
        let t_total = encoder_output.dim(0)?;
        let device = encoder_output.device();
        let enc_h_all = joint.project_encoder_all(encoder_output)?; // [T, joint_hidden]
        let blank = self.blank_idx;

        let init_state = prediction_net.initial_state(device)?;
        let (pred0, state0) = prediction_net.step(blank as u32, &init_state)?;
        let mut beam = vec![TdtBeamHyp {
            tokens: Vec::new(),
            timestamps: Vec::new(),
            state: state0,
            pred_out: pred0,
            time: 0,
            symbols_at_t: 0,
            score: 0.0,
        }];
        let mut finished: Vec<TdtBeamHyp> = Vec::new();

        // Каждый шаг либо двигает время, либо ограничен max_symbols → терминируется.
        let max_iters = (t_total + 1) * (self.max_symbols_per_step + 2) * 4 + 64;
        let mut iters = 0usize;
        while !beam.is_empty() {
            iters += 1;
            if iters > max_iters {
                debug!("TDT beam: max_iters достигнут");
                break;
            }
            let mut active: Vec<TdtBeamHyp> = Vec::new();
            for h in beam.drain(..) {
                if h.time >= t_total {
                    finished.push(h);
                } else {
                    active.push(h);
                }
            }
            if active.is_empty() {
                break;
            }
            let mut cand: Vec<TdtBeamHyp> = Vec::new();
            for h in &active {
                let enc_h = enc_h_all.i(h.time)?;
                let (tok_logits, dur_logits) = joint.forward_with_cached_enc(&enc_h, &h.pred_out)?;
                let tlog = log_softmax_vec(
                    &tok_logits.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                );
                let dlog = log_softmax_vec(
                    &dur_logits.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                );
                let dur_idx = argmax(&dlog);
                let skip = self.durations.get(dur_idx).copied().unwrap_or(1);
                let dlp = dlog[dur_idx];

                // blank-ветка: время вперёд на skip(>=1), эмиссии нет
                let mut nb = h.clone();
                nb.score += tlog[blank] + dlp;
                nb.time += skip.max(1);
                nb.symbols_at_t = 0;
                cand.push(nb);

                // top-K не-blank токенов
                for &k in top_k_indices(&tlog, beam_width, blank).iter() {
                    let (po, sn) = prediction_net.step(k as u32, &h.state)?;
                    let mut nb = h.clone();
                    nb.tokens.push(k as u32);
                    nb.timestamps.push((k as u32, (h.time as u64) * 80));
                    nb.state = sn;
                    nb.pred_out = po;
                    nb.score += tlog[k] + dlp;
                    if skip == 0 {
                        nb.symbols_at_t = h.symbols_at_t + 1;
                        if nb.symbols_at_t >= self.max_symbols_per_step {
                            nb.time += 1;
                            nb.symbols_at_t = 0;
                        }
                    } else {
                        nb.time += skip;
                        nb.symbols_at_t = 0;
                    }
                    cand.push(nb);
                }
            }
            cand.sort_by(|a, b| b.score.total_cmp(&a.score));
            cand.truncate(beam_width);
            beam = cand;
        }
        finished.extend(beam.drain(..));
        finished.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(finished
            .into_iter()
            .next()
            .map(|h| TdtResult { tokens: h.tokens, timestamps: h.timestamps })
            .unwrap_or(TdtResult { tokens: Vec::new(), timestamps: Vec::new() }))
    }

    /// Streaming TDT декодирование: callback вызывается на каждый non-blank токен.
    ///
    /// `callback(token_id, timestamp_ms)` — timestamp из frame_index × 80мс.
    pub fn decode_streaming(
        &self,
        encoder_output: &Tensor,
        prediction_net: &PredictionNet,
        joint: &JointNetwork,
        callback: impl Fn(u32, u64),
    ) -> Result<TdtResult> {
        self.decode_inner(encoder_output, prediction_net, joint, Some(&callback))
    }

    fn decode_inner(
        &self,
        encoder_output: &Tensor,
        prediction_net: &PredictionNet,
        joint: &JointNetwork,
        callback: Option<&dyn Fn(u32, u64)>,
    ) -> Result<TdtResult> {
        let t_total = encoder_output.dim(0)?;
        let device = encoder_output.device();

        debug!("TDT decode: {} фреймов энкодера, blank_idx={}", t_total, self.blank_idx);

        // P1 — enc_proj cache: проецируем весь encoder output ОДИН раз.
        // enc_h_all: [T, joint_hidden=640] вместо повторного матмула на каждом фрейме.
        let enc_h_all = joint.project_encoder_all(encoder_output)?;

        let mut hypothesis: Vec<u32> = Vec::new();
        let mut timestamps: Vec<(u32, u64)> = Vec::new();
        let mut state = prediction_net.initial_state(device)?;
        let mut last_token: u32 = self.blank_idx as u32;
        let mut time_idx: usize = 0;

        // P0a — predictor state cache.
        // pred_dirty=true означает: нужно пересчитать pred_out перед следующим joint.
        // Пересчёт нужен только после non-blank emission (last_token изменился).
        let mut pred_dirty = true;
        let mut pred_out_cached: Option<Tensor> = None;

        while time_idx < t_total {
            // P0a: пересчитываем predictor только при изменении last_token.
            if pred_dirty {
                let (po, sn) = prediction_net.step(last_token, &state)?;
                pred_out_cached = Some(po);
                state = sn;
                pred_dirty = false;
            }
            let pred_out = pred_out_cached.as_ref().unwrap();

            // P1: используем закешированный enc_h для текущего фрейма.
            let enc_h_frame = enc_h_all.i(time_idx)?;
            let (token_logits, dur_logits) =
                joint.forward_with_cached_enc(&enc_h_frame, pred_out)?;

            // P4 — combined sync: вместо 2× to_scalar (2 GPU→CPU блокирующих transfer'а)
            // делаем 1 transfer через stack([token_argmax, dur_argmax]) + to_vec1.
            // Bit-exact, но в 2 раза меньше syncs per frame.
            let token_argmax = token_logits.argmax_keepdim(D::Minus1)?;
            let dur_argmax = dur_logits.argmax_keepdim(D::Minus1)?;
            let combined = Tensor::cat(&[&token_argmax, &dur_argmax], 0)?;
            let v: Vec<u32> = combined.flatten_all()?.to_vec1()?;
            let k = v[0];
            let dur_idx = v[1] as usize;

            // Убран per-iter flush_buffers — он замусоривал hot path. Pool flush
            // делается callsite декодера (см. parakeet_engine.rs::transcribe).

            let skip = if dur_idx < self.durations.len() {
                self.durations[dur_idx]
            } else {
                1
            };

            if k as usize == self.blank_idx {
                // blank — pred остаётся валидным, кеш не сбрасываем
                time_idx += skip.max(1);
            } else {
                // subsampling_factor=8, window_stride=0.01 → 1 frame = 80мс
                let ts_ms = (time_idx as u64) * 80;
                hypothesis.push(k);
                timestamps.push((k, ts_ms));
                if let Some(cb) = &callback {
                    cb(k, ts_ms);
                }
                last_token = k;
                // Non-blank emission: state уже обновлён в step() выше (sn → state),
                // pred_dirty=true заставит пересчитать с новым last_token на следующей итерации.
                pred_dirty = true;

                if skip == 0 {
                    let mut symbols_added = 1;
                    while symbols_added < self.max_symbols_per_step {
                        // Каждый inner шаг требует свежего pred (last_token только что изменился).
                        if pred_dirty {
                            let (po2, sn2) = prediction_net.step(last_token, &state)?;
                            pred_out_cached = Some(po2);
                            state = sn2;
                            pred_dirty = false;
                        }
                        let pred_out2 = pred_out_cached.as_ref().unwrap();
                        let (token_logits2, dur_logits2) =
                            joint.forward_with_cached_enc(&enc_h_frame, pred_out2)?;

                        // Combined sync (см. главный loop выше)
                        let token_argmax2 = token_logits2.argmax_keepdim(D::Minus1)?;
                        let dur_argmax2 = dur_logits2.argmax_keepdim(D::Minus1)?;
                        let combined2 = Tensor::cat(&[&token_argmax2, &dur_argmax2], 0)?;
                        let v2: Vec<u32> = combined2.flatten_all()?.to_vec1()?;
                        let k2 = v2[0];
                        let dur_idx2 = v2[1] as usize;
                        let skip2 = if dur_idx2 < self.durations.len() {
                            self.durations[dur_idx2]
                        } else {
                            1
                        };

                        if k2 as usize == self.blank_idx {
                            time_idx += skip2.max(1);
                            break;
                        }

                        hypothesis.push(k2);
                        timestamps.push((k2, ts_ms));
                        if let Some(cb) = &callback {
                            cb(k2, ts_ms);
                        }
                        last_token = k2;
                        pred_dirty = true;
                        symbols_added += 1;

                        if skip2 > 0 {
                            time_idx += skip2;
                            break;
                        }
                    }
                    if symbols_added >= self.max_symbols_per_step {
                        time_idx += 1;
                    }
                } else {
                    time_idx += skip;
                }
            }
        }

        debug!("TDT decode: {} токенов гипотезы", hypothesis.len());

        Ok(TdtResult {
            tokens: hypothesis,
            timestamps,
        })
    }
}
