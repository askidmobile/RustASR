//! NemotronModel — полная сборка пайплайна Nemotron 3.5 ASR (offline, F32).
//!
//! mel → encoder(causal FastConformer) → prompt_kernel(lang fusion) →
//! RNN-T greedy (LSTM pred + joint) → SentencePiece detok.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use tracing::info;

use asr_core::{AsrError, AsrResult};

use crate::config::NemotronConfig;
use crate::decoder::PredictionNet;
use crate::encoder::NemotronEncoder;
use crate::joint::JointNet;
use crate::mel::NemotronMelExtractor;
use crate::prompt::PromptKernel;
use crate::tokenizer::SpVocab;

/// Максимум символов на один кадр энкодера (RNN-T greedy).
const MAX_SYMBOLS_PER_STEP: usize = 10;

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

impl NemotronModel {
    /// Загрузить модель из директории. Приоритет: `model-q8_0.gguf` (Q8, ~755МБ),
    /// fallback `model.safetensors` (F32, ~2.4ГБ). + config.json + vocab.json.
    pub fn load(model_dir: impl AsRef<Path>, device: &Device) -> AsrResult<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let config = NemotronConfig::from_json_file(&model_dir.join("config.json"))
            .map_err(|e| AsrError::Config(format!("config.json: {e}")))?;
        let vocab = SpVocab::from_json(&model_dir.join("vocab.json"))
            .map_err(|e| AsrError::Model(format!("vocab.json: {e}")))?;

        let gguf = model_dir.join("model-q8_0.gguf");
        let (vb, fb, window) = if gguf.exists() {
            Self::weights_from_gguf(&gguf, device)?
        } else {
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
            "Nemotron загружен ({}): encoder {}×{}d, vocab {}, blank {}, ru-RU {}",
            if gguf.exists() { "Q8 GGUF" } else { "F32 safetensors" },
            config.encoder.n_layers, config.encoder.d_model, config.decoder.vocab_size, blank_idx, lang_idx
        );

        Ok(Self {
            config, device: device.clone(), model_dir,
            mel, encoder, prompt, pred, joint, vocab, blank_idx, lang_idx,
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

    /// VarBuilder + fb/window из Q8 GGUF (все тензоры dequant → F32).
    /// Корректность важнее RAM (Q8 в RAM через QMatMul — отдельная оптимизация Ф4b).
    fn weights_from_gguf(
        path: &Path,
        device: &Device,
    ) -> AsrResult<(VarBuilder<'static>, Tensor, Tensor)> {
        use candle_core::quantized::gguf_file;
        let mut f = std::fs::File::open(path)
            .map_err(|e| AsrError::Model(format!("open gguf: {e}")))?;
        let content = gguf_file::Content::read(&mut f)
            .map_err(|e| AsrError::Model(format!("read gguf: {e}")))?;
        let mut map: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        for name in content.tensor_infos.keys() {
            let qt = content.tensor(&mut f, name, device)
                .map_err(|e| AsrError::Model(format!("gguf tensor {name}: {e}")))?;
            let t = qt.dequantize(device)
                .map_err(|e| AsrError::Model(format!("dequant {name}: {e}")))?;
            map.insert(name.clone(), t);
        }
        let fb = map.get("preprocessor.featurizer.fb").cloned()
            .ok_or_else(|| AsrError::Model("нет fb в gguf".into()))?;
        let window = map.get("preprocessor.featurizer.window").cloned()
            .ok_or_else(|| AsrError::Model("нет window в gguf".into()))?;
        let vb = VarBuilder::from_tensors(map, DType::F32, device);
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

    /// Внутренний инференс → последовательность token id (без blank).
    fn transcribe_tokens(&self, samples: &[f32]) -> CandleResult<Vec<u32>> {
        let mel = self
            .mel
            .extract(samples, &self.device)
            .map_err(|e| candle_core::Error::Msg(format!("mel: {e}")))?; // [1,128,T]
        let encoded = self.encoder.encode(&mel)?; // [1,T',1024]
        let fused = self.prompt.forward(&encoded, self.lang_idx)?; // [1,T',1024]
        self.greedy_decode(&fused)
    }

    /// RNN-T greedy по уже сфьюженному (post-prompt_kernel) выходу [.,T,1024] или [T,1024].
    /// Тест может подать эталонный fused (prompt_kernel_out) для изоляции joint/greedy.
    pub fn greedy_decode(&self, fused: &Tensor) -> CandleResult<Vec<u32>> {
        let fused = if fused.dims().len() == 3 { fused.squeeze(0)? } else { fused.clone() }; // [T',1024]
        let enc_proj = self.joint.project_enc(&fused)?; // [T',640]
        let t_frames = enc_proj.dim(0)?;

        let mut state = self.pred.initial_state(&self.device)?;
        // SOS = blank (padding_idx, нулевой embedding)
        let (mut g, s0) = self.pred.step(self.blank_idx, &state)?;
        state = s0;

        let dbg = std::env::var("NEMOTRON_DEBUG").is_ok();
        if dbg {
            let fstat = |t: &Tensor| -> String {
                let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
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
