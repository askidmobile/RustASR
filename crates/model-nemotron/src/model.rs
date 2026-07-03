//! NemotronModel — полная сборка пайплайна Nemotron 3.5 ASR (offline, F32).
//!
//! mel → encoder(causal FastConformer) → prompt_kernel(lang fusion) →
//! RNN-T greedy (LSTM pred + joint) → SentencePiece detok.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use tracing::info;

use std::time::Instant;

use asr_core::{
    AsrError, AsrModel, AsrResult, ModelInfo, ModelType, QuantizationType, TranscribeOptions,
    TranscriptionResult,
};

use crate::config::NemotronConfig;
use crate::decoder::PredictionNet;
use crate::encoder::NemotronEncoder;
use crate::joint::JointNet;
use crate::mel::NemotronMelExtractor;
use crate::prompt::PromptKernel;
use crate::tokenizer::SpVocab;

/// Максимум символов на один кадр энкодера (RNN-T greedy).
const MAX_SYMBOLS_PER_STEP: usize = 10;

/// Ширина beam по умолчанию (beam-search RNN-T). 1 = greedy.
/// Замеры: beam=4 восстанавливает пропущенные слова на трудном RU; >4 — diminishing.
/// Override: env NEMOTRON_BEAM.
const DEFAULT_BEAM_WIDTH: usize = 4;

pub struct NemotronModel {
    pub config: NemotronConfig,
    device: Device,
    #[allow(dead_code)]
    model_dir: PathBuf,
    mel: NemotronMelExtractor,
    encoder: NemotronEncoder,
    prompt: PromptKernel,
    pred: PredictionNet,
    joint: JointNet,
    vocab: SpVocab,
    blank_idx: u32,
    lang_idx: usize,
    /// Compute-dtype весов (F16 в проде). Нужен, чтобы приводить внешне поданный
    /// `fused` (например, F32-эталон в parity-тесте) к dtype весов перед joint.
    compute_dtype: DType,
}

/// Compute-dtype весов Nemotron после dequant. По умолчанию **F16**: вдвое меньше RAM
/// (~2.4ГБ→~1.2ГБ) практически без потери качества — joint/decoder/prompt в GGUF уже
/// хранятся как F16 (round-trip F16→dequant F32→F16 бит-в-бит), а encoder Q8→F16
/// покрывается 10-битной мантиссой F16 с запасом относительно ошибки самого Q8.
/// Override: env `NEMOTRON_DTYPE=f32|f16|bf16` (диагностика / A-B качества).
fn resolve_compute_dtype() -> DType {
    match std::env::var("NEMOTRON_DTYPE").ok().as_deref() {
        Some("f32") => DType::F32,
        Some("bf16") => DType::BF16,
        None | Some("f16") | Some("") => DType::F16,
        Some(other) => {
            tracing::warn!("NEMOTRON_DTYPE='{other}' не распознан — использую f16");
            DType::F16
        }
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best
}

/// log-softmax по вектору логитов (numerically stable).
fn log_softmax_vec(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for &x in logits {
        sum += (x - m).exp();
    }
    let lse = m + sum.ln();
    logits.iter().map(|&x| x - lse).collect()
}

/// Гипотеза beam-поиска RNN-T.
#[derive(Clone)]
struct BeamHyp {
    tokens: Vec<u32>,
    state: crate::decoder::LstmState,
    g: Tensor,
    score: f32,
}

impl NemotronModel {
    /// Загрузить модель из директории. Приоритет: `model-q8_0.gguf` (Q8, ~755МБ),
    /// fallback `model.safetensors` (F32, ~2.4ГБ). + config.json + vocab.json.
    pub fn load(model_dir: impl AsRef<Path>, device: &Device) -> AsrResult<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let config = NemotronConfig::from_json_file(&model_dir.join("config.json"))
            .map_err(|e| AsrError::Config(format!("config.json: {e}")))?;
        let vocab = SpVocab::from_json(&model_dir.join("vocab.json"))
            .map_err(|e| AsrError::Model(format!("vocab.json: {e}")))?;

        // NEMOTRON_FORCE_F32 / NEMOTRON_GGUF=<file> — для диагностики качества квантования.
        let force_f32 = std::env::var("NEMOTRON_FORCE_F32").is_ok();
        // Compute-dtype весов из GGUF: F16 по умолчанию (вдвое меньше RAM). Полный F32
        // остаётся через NEMOTRON_FORCE_F32 (safetensors-путь) или NEMOTRON_DTYPE=f32.
        let compute_dtype = if force_f32 { DType::F32 } else { resolve_compute_dtype() };
        let gguf_name = std::env::var("NEMOTRON_GGUF").unwrap_or_else(|_| "model-q8_0.gguf".into());
        let gguf = model_dir.join(&gguf_name);
        let (vb, fb, window) = if !force_f32 && gguf.exists() {
            Self::weights_from_gguf(&gguf, device, compute_dtype)?
        } else {
            // Fallback F32 safetensors (диагностика/отсутствие GGUF) — всегда полный F32.
            Self::weights_from_safetensors(&model_dir.join("model.safetensors"), device)?
        };

        let mel = NemotronMelExtractor::from_tensors(&config, &fb, &window)?;
        let encoder = NemotronEncoder::load(&config.encoder, vb.pp("encoder"))
            .map_err(|e| AsrError::Model(format!("encoder: {e}")))?;
        let prompt = PromptKernel::load(&config.prompt, vb.pp("prompt_kernel"))
            .map_err(|e| AsrError::Model(format!("prompt_kernel: {e}")))?;
        let pred = PredictionNet::load(&config.decoder, vb.pp("decoder"))
            .map_err(|e| AsrError::Model(format!("decoder: {e}")))?;
        let joint = JointNet::load(&config.joint, vb.pp("joint"))
            .map_err(|e| AsrError::Model(format!("joint: {e}")))?;

        let lang_idx = config.prompt.lang_idx("ru-RU").unwrap_or(11);
        let blank_idx = config.decoder.blank_idx as u32;
        info!(
            "Nemotron загружен ({}, compute={:?}): encoder {}×{}d, vocab {}, blank {}, ru-RU {}",
            if !force_f32 && gguf.exists() { "Q8 GGUF" } else { "F32 safetensors" },
            compute_dtype,
            config.encoder.n_layers, config.encoder.d_model, config.decoder.vocab_size, blank_idx, lang_idx
        );

        Ok(Self {
            config, device: device.clone(), model_dir,
            mel, encoder, prompt, pred, joint, vocab, blank_idx, lang_idx,
            compute_dtype,
        })
    }

    /// VarBuilder + fb/window из F32 safetensors.
    fn weights_from_safetensors(
        st: &Path,
        device: &Device,
    ) -> AsrResult<(VarBuilder<'static>, Tensor, Tensor)> {
        if !st.exists() {
            return Err(AsrError::Model(format!("нет {st:?}")));
        }
        let w = candle_core::safetensors::load(st, device)
            .map_err(|e| AsrError::Model(format!("load weights: {e}")))?;
        let fb = w.get("preprocessor.featurizer.fb").cloned()
            .ok_or_else(|| AsrError::Model("нет fb".into()))?;
        let window = w.get("preprocessor.featurizer.window").cloned()
            .ok_or_else(|| AsrError::Model("нет window".into()))?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[st.to_path_buf()], DType::F32, device)
                .map_err(|e| AsrError::Model(format!("safetensors: {e}")))?
        };
        Ok((vb, fb, window))
    }

    /// VarBuilder + fb/window из Q8 GGUF. Веса dequant → `dtype` (F16 по умолчанию —
    /// вдвое меньше RAM). Исключение: `fb`/`window` всегда остаются F32, т.к. идут в
    /// CPU-mel (rustfft) и определяют точность мел-фильтрбанка/окна — их понижать нельзя.
    ///
    /// **Dequant идёт на CPU, на устройство переносится только финальный (F16) тензор.**
    /// Иначе на Metal `dequantize`→F32 создаёт крупные временные буферы, которые
    /// pool-аллокатор candle (round-up до pow2, «первый подходящий ≥ size») затем
    /// переиспользует под F16-результат — F16-веса оседают в F32-размерных буферах, и
    /// экономия RAM почти обнуляется (замер: 2024 МБ вместо 1233). CPU-путь держит на
    /// устройстве ровно один буфер нужного размера на тензор.
    fn weights_from_gguf(
        path: &Path,
        device: &Device,
        dtype: DType,
    ) -> AsrResult<(VarBuilder<'static>, Tensor, Tensor)> {
        use candle_core::quantized::gguf_file;
        let cpu = Device::Cpu;
        let mut f = std::fs::File::open(path)
            .map_err(|e| AsrError::Model(format!("open gguf: {e}")))?;
        let content = gguf_file::Content::read(&mut f)
            .map_err(|e| AsrError::Model(format!("read gguf: {e}")))?;
        let mut map: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        for name in content.tensor_infos.keys() {
            // Читаем и dequant-им на CPU: крупные F32-временные не попадают в Metal-пул.
            let qt = content.tensor(&mut f, name, &cpu)
                .map_err(|e| AsrError::Model(format!("gguf tensor {name}: {e}")))?;
            let dq = qt.dequantize(&cpu)
                .map_err(|e| AsrError::Model(format!("dequant {name}: {e}")))?;
            let keep_f32 = name == "preprocessor.featurizer.fb"
                || name == "preprocessor.featurizer.window";
            let host = if keep_f32 || dtype == DType::F32 {
                dq
            } else {
                dq.to_dtype(dtype)
                    .map_err(|e| AsrError::Model(format!("cast {name}→{dtype:?}: {e}")))?
            };
            // На устройство переносим уже узкий (F16) тензор — один буфер нужного размера.
            let t = host.to_device(device)
                .map_err(|e| AsrError::Model(format!("to_device {name}: {e}")))?;
            map.insert(name.clone(), t);
        }
        let fb = map.get("preprocessor.featurizer.fb").cloned()
            .ok_or_else(|| AsrError::Model("нет fb в gguf".into()))?;
        let window = map.get("preprocessor.featurizer.window").cloned()
            .ok_or_else(|| AsrError::Model("нет window в gguf".into()))?;
        // dtype VarBuilder = compute-dtype: encoder/decoder читают его через vb.dtype().
        let vb = VarBuilder::from_tensors(map, dtype, device);
        Ok((vb, fb, window))
    }

    /// Транскрибировать аудио (16kHz mono f32) → текст (язык по lang_idx, ru-RU).
    pub fn transcribe(&self, samples: &[f32]) -> AsrResult<String> {
        let tokens = self
            .transcribe_tokens(samples)
            .map_err(|e| AsrError::Model(format!("inference: {e}")))?;
        Ok(self.vocab.decode(&tokens))
    }

    /// Детокенизация token id → текст (для тестов/изоляции).
    pub fn decode_text(&self, tokens: &[u32]) -> String {
        self.vocab.decode(tokens)
    }

    /// Текущий язык prompt'а (индекс one-hot).
    pub fn lang_idx(&self) -> usize {
        self.lang_idx
    }

    /// Сменить язык prompt'а по коду (`ru`/`ru-RU`/`en`/`auto`/…). Для смешанного
    /// RU/EN полезен `auto`. Возвращает true если код найден в prompt_dictionary.
    pub fn set_language(&mut self, code: &str) -> bool {
        match self.config.prompt.lang_idx(code) {
            Some(idx) => {
                self.lang_idx = idx;
                true
            }
            None => false,
        }
    }

    /// Внутренний инференс → последовательность token id (без blank).
    fn transcribe_tokens(&self, samples: &[f32]) -> CandleResult<Vec<u32>> {
        let mel = self
            .mel
            .extract(samples, &self.device)
            .map_err(|e| candle_core::Error::Msg(format!("mel: {e}")))?; // [1,128,T]
        let encoded = self.encoder.encode(&mel)?; // [1,T',1024]
        let fused = self.prompt.forward(&encoded, self.lang_idx)?; // [1,T',1024]
        // По умолчанию beam-search (ширина DEFAULT_BEAM_WIDTH); env NEMOTRON_BEAM=1 → greedy.
        let w = std::env::var("NEMOTRON_BEAM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_BEAM_WIDTH);
        if w > 1 {
            self.beam_decode(&fused, w)
        } else {
            self.greedy_decode(&fused)
        }
    }

    /// RNN-T beam-search (Graves, time-synchronous) по сфьюженному выходу.
    /// Держит beam_width гипотез; на каждом кадре blank-расширение → след. кадр,
    /// token-расширения (top-K) остаются в кадре. Скор = сумма log-prob.
    pub fn beam_decode(&self, fused: &Tensor, beam_width: usize) -> CandleResult<Vec<u32>> {
        let fused = if fused.dims().len() == 3 { fused.squeeze(0)? } else { fused.clone() };
        // Внешне поданный fused (parity-тест) может быть F32 — приводим к dtype весов.
        let fused = fused.to_dtype(self.compute_dtype)?;
        let enc_proj = self.joint.project_enc(&fused)?; // [T,640]
        let t_frames = enc_proj.dim(0)?;
        let blank = self.blank_idx as usize;

        let init = self.pred.initial_state(&self.device)?;
        let (g0, s0) = self.pred.step(self.blank_idx, &init)?;
        let mut beam = vec![BeamHyp { tokens: vec![], state: s0, g: g0, score: 0.0 }];

        for t in 0..t_frames {
            let enc_t = enc_proj.narrow(0, t, 1)?.squeeze(0)?; // [640]
            let mut a = std::mem::take(&mut beam);
            let mut b: Vec<BeamHyp> = Vec::new();
            let mut guard = 0usize;
            let max_expansions = beam_width * (MAX_SYMBOLS_PER_STEP + 1) + 4;
            while !a.is_empty() && guard < max_expansions {
                guard += 1;
                // достать лучшую гипотезу из A
                let bi = a
                    .iter()
                    .enumerate()
                    .max_by(|x, y| x.1.score.total_cmp(&y.1.score))
                    .map(|(i, _)| i)
                    .unwrap();
                let y = a.swap_remove(bi);
                let logits = self.joint.step_logits(&enc_t, &y.g)?.to_vec1::<f32>()?;
                let logp = log_softmax_vec(&logits);
                // blank → переносим в B (продвинуть время), токены не меняются
                let mut yb = y.clone();
                yb.score += logp[blank];
                b.push(yb);
                // top-K не-blank токенов → расширяем, остаются в A
                let mut idx: Vec<usize> = (0..logp.len()).filter(|&i| i != blank).collect();
                idx.sort_by(|&i, &j| logp[j].total_cmp(&logp[i]));
                for &k in idx.iter().take(beam_width) {
                    let (g_new, s_new) = self.pred.step(k as u32, &y.state)?;
                    let mut tk = y.tokens.clone();
                    tk.push(k as u32);
                    a.push(BeamHyp { tokens: tk, state: s_new, g: g_new, score: y.score + logp[k] });
                }
                if a.len() > beam_width {
                    a.sort_by(|x, y| y.score.total_cmp(&x.score));
                    a.truncate(beam_width);
                }
                // ранний выход: лучшая в A уже не попадёт в top-W из B
                if b.len() >= beam_width {
                    b.sort_by(|x, y| y.score.total_cmp(&x.score));
                    let wth = b[beam_width - 1].score;
                    let amax = a.iter().map(|h| h.score).fold(f32::NEG_INFINITY, f32::max);
                    if amax <= wth {
                        break;
                    }
                }
            }
            b.sort_by(|x, y| y.score.total_cmp(&x.score));
            b.truncate(beam_width.max(1));
            beam = b;
        }
        beam.sort_by(|x, y| y.score.total_cmp(&x.score));
        Ok(beam.into_iter().next().map(|h| h.tokens).unwrap_or_default())
    }

    /// RNN-T greedy по уже сфьюженному (post-prompt_kernel) выходу [.,T,1024] или [T,1024].
    /// Тест может подать эталонный fused (prompt_kernel_out) для изоляции joint/greedy.
    pub fn greedy_decode(&self, fused: &Tensor) -> CandleResult<Vec<u32>> {
        let fused = if fused.dims().len() == 3 { fused.squeeze(0)? } else { fused.clone() }; // [T',1024]
        // Внешне поданный fused (parity-тест изоляции) может быть F32 — приводим к dtype весов.
        let fused = fused.to_dtype(self.compute_dtype)?;
        let enc_proj = self.joint.project_enc(&fused)?; // [T',640]
        let t_frames = enc_proj.dim(0)?;

        let mut state = self.pred.initial_state(&self.device)?;
        // SOS = blank (padding_idx, нулевой embedding)
        let (mut g, s0) = self.pred.step(self.blank_idx, &state)?;
        state = s0;

        let dbg = std::env::var("NEMOTRON_DEBUG").is_ok();
        if dbg {
            let fstat = |t: &Tensor| -> String {
                let v = t.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
                let n = v.len() as f32;
                let mean = v.iter().sum::<f32>() / n;
                let (mn, mx) = v.iter().fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
                let nan = v.iter().any(|x| x.is_nan());
                format!("mean={mean:.3} min={mn:.3} max={mx:.3} nan={nan}")
            };
            eprintln!("[dbg] fused {:?} {}", fused.dims(), fstat(&fused));
            eprintln!("[dbg] enc_proj {:?} {}", enc_proj.dims(), fstat(&enc_proj));
            eprintln!("[dbg] g(SOS) {:?} {}", g.dims(), fstat(&g));
            // По кадрам с фиксированным SOS g: что предсказывает joint?
            let mut per_frame = Vec::new();
            for t in 0..t_frames {
                let e = enc_proj.narrow(0, t, 1)?.squeeze(0)?;
                let lv = self.joint.step_logits(&e, &g)?.to_vec1::<f32>()?;
                let tok = argmax(&lv);
                let blk = lv[self.blank_idx as usize];
                per_frame.push((tok, lv[tok], blk));
            }
            let nonblank: Vec<_> = per_frame.iter().enumerate().filter(|(_, (t, _, _))| *t != self.blank_idx as usize).collect();
            eprintln!("[dbg] {} frames, {} non-blank argmax (с SOS g)", t_frames, nonblank.len());
            for (t, (tok, lg, blk)) in per_frame.iter().enumerate().take(6) {
                eprintln!("[dbg]  t={t} argmax={tok} ({:.2}) blank={:.2}", lg, blk);
            }
        }
        let mut tokens = Vec::new();
        for t in 0..t_frames {
            let enc_t = enc_proj.narrow(0, t, 1)?.squeeze(0)?; // [640]
            let mut sym = 0;
            loop {
                let logits = self.joint.step_logits(&enc_t, &g)?; // [13088]
                let lv = logits.to_vec1::<f32>()?;
                let tok = argmax(&lv) as u32;
                if tok == self.blank_idx || sym >= MAX_SYMBOLS_PER_STEP {
                    break;
                }
                tokens.push(tok);
                let (g_new, s_new) = self.pred.step(tok, &state)?;
                g = g_new;
                state = s_new;
                sym += 1;
            }
        }
        Ok(tokens)
    }
}

impl AsrModel for NemotronModel {
    fn name(&self) -> &str {
        &self.config.model_name
    }

    fn model_type(&self) -> ModelType {
        ModelType::Nemotron
    }

    fn supported_languages(&self) -> &[&str] {
        // Мультиязычная (40 локалей); сейчас сконфигурирована на ru-RU prompt.
        &[]
    }

    fn model_info(&self) -> ModelInfo {
        let gguf = self.model_dir.join("model-q8_0.gguf");
        let (size, quant) = if gguf.exists() {
            (std::fs::metadata(&gguf).map(|m| m.len()).ok(), QuantizationType::GgufQ8_0)
        } else {
            (
                std::fs::metadata(self.model_dir.join("model.safetensors")).map(|m| m.len()).ok(),
                QuantizationType::None,
            )
        };
        ModelInfo {
            model_type: ModelType::Nemotron,
            display_name: "Nemotron 3.5 ASR (0.6B)".to_string(),
            parameters: Some(648_000_000),
            weights_size_bytes: size,
            quantization: quant,
            languages: vec!["ru".to_string()],
            backend: "candle".to_string(),
        }
    }

    fn transcribe(
        &mut self,
        samples: &[f32],
        _options: &TranscribeOptions,
    ) -> AsrResult<TranscriptionResult> {
        let t0 = Instant::now();
        let tokens = self
            .transcribe_tokens(samples)
            .map_err(|e| AsrError::Model(format!("inference: {e}")))?;
        let text = self.vocab.decode(&tokens);
        let dt = t0.elapsed().as_secs_f64();
        let dur = samples.len() as f64 / self.config.sample_rate as f64;
        let mut res = TranscriptionResult::new(text, self.name().to_string(), dt, dur);
        res.language = Some("ru".to_string());
        Ok(res)
    }
}
