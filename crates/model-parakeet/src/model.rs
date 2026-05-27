//! ParakeetModel — основной модуль, реализующий AsrModel trait.
//!
//! Загружает всю модель из safetensors + config.json + tokenizer.model,
//! объединяет mel → encoder → TDT decode → SentencePiece detokenize.

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use tracing::{debug, info, warn};

use asr_core::{
    AsrError, AsrModel, AsrResult, ModelInfo, ModelType, QuantizationType, TranscribeOptions,
    TranscriptionResult,
};

use crate::config::ParakeetConfig;
use crate::decoder::PredictionNet;
use crate::encoder::FastConformerEncoder;
use crate::joint::JointNetwork;
use crate::mel::ParakeetMelExtractor;
use crate::tdt::TdtGreedyDecoder;

/// Максимальная длительность одного чанка в секундах.
const CHUNK_DURATION_SECS: f64 = 30.0;

/// Модель Parakeet-TDT v3.
pub struct ParakeetModel {
    encoder: FastConformerEncoder,
    prediction_net: PredictionNet,
    joint: JointNetwork,
    tdt_decoder: TdtGreedyDecoder,
    mel_extractor: ParakeetMelExtractor,
    tokenizer: SentencePieceTokenizer,
    device: Device,
    config: ParakeetConfig,
    model_dir: PathBuf,
}

/// Обёртка SentencePiece токенизатора.
struct SentencePieceTokenizer {
    /// Массив кусочков (pieces): index → string.
    pub(crate) vocab: Vec<String>,
}

impl SentencePieceTokenizer {
    /// Загрузить токенизатор из vocab.json.
    fn from_vocab_json(path: &Path) -> AsrResult<Self> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| AsrError::Model(format!("Не удалось прочитать vocab.json: {e}")))?;
        let map: serde_json::Value = serde_json::from_str(&data)
            .map_err(|e| AsrError::Model(format!("Не удалось распарсить vocab.json: {e}")))?;

        // vocab.json: { "piece": {"id": N, "score": F}, ... }
        let obj = map
            .as_object()
            .ok_or_else(|| AsrError::Model("vocab.json должен быть JSON-объектом".into()))?;

        let mut vocab = vec![String::new(); obj.len()];
        for (piece, info) in obj {
            if let Some(id) = info.get("id").and_then(|v| v.as_u64()) {
                let idx = id as usize;
                if idx < vocab.len() {
                    vocab[idx] = piece.clone();
                }
            }
        }

        info!("SentencePiece токенизатор: {} токенов", vocab.len());
        Ok(Self { vocab })
    }

    /// Декодировать последовательность token ID в текст.
    fn decode(&self, tokens: &[u32]) -> String {
        let mut text = String::new();
        for &tok in tokens {
            let idx = tok as usize;
            if idx < self.vocab.len() {
                let piece = &self.vocab[idx];
                // SentencePiece: ▁ → пробел
                let decoded = piece.replace('▁', " ");
                text.push_str(&decoded);
            }
        }
        // Убрать ведущий пробел
        text.trim().to_string()
    }
}

impl ParakeetModel {
    /// Загрузить модель из директории.
    ///
    /// Ожидаемые файлы:
    /// - config.json
    /// - model.safetensors
    /// - vocab.json (или tokenizer.model)
    pub fn load(model_dir: impl AsRef<Path>, device: &Device) -> AsrResult<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        info!("Загрузка Parakeet-TDT из {:?}", model_dir);

        // 1. Загрузить конфигурацию
        let config_path = model_dir.join("config.json");
        let config: ParakeetConfig = if config_path.exists() {
            let data = std::fs::read_to_string(&config_path)?;
            serde_json::from_str(&data)
                .map_err(|e| AsrError::Config(format!("Ошибка парсинга config.json: {e}")))?
        } else {
            warn!("config.json не найден, используем дефолтные значения");
            ParakeetConfig::default_v3()
        };

        // 2. Загрузить веса
        let safetensors_path = model_dir.join("model.safetensors");
        if !safetensors_path.exists() {
            return Err(AsrError::Model(format!(
                "model.safetensors не найден в {:?}",
                model_dir
            )));
        }

        // F32 на Metal/CPU — mel extraction (FFT) требует F32.
        // Candle автоматически upcast'ит F16 safetensors → F32 при загрузке.
        // BF16 только на CUDA.
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&safetensors_path], dtype, device)? };

        // 3. Mel-экстрактор: загружаем фильтры из весов если доступны
        let mel_extractor = {
            let pp_vb = vb.pp("preprocessor").pp("featurizer");
            let mel_fb = pp_vb.get(
                (
                    1,
                    config.preprocessor.features,
                    config.preprocessor.n_fft / 2 + 1,
                ),
                "fb",
            );
            let window = pp_vb.get(config.win_length(), "window");

            match (mel_fb, window) {
                (Ok(fb), Ok(w)) => {
                    info!("Mel-фильтры загружены из весов модели");
                    ParakeetMelExtractor::from_tensors(&config, &fb, Some(&w))?
                }
                (Ok(fb), Err(_)) => {
                    info!("Mel-фильтры из весов, окно сгенерировано");
                    ParakeetMelExtractor::from_tensors(&config, &fb, None)?
                }
                _ => {
                    warn!("Mel-фильтры не найдены в весах, генерируем Slaney");
                    ParakeetMelExtractor::new(&config)
                }
            }
        };

        // 4. Encoder
        let encoder = FastConformerEncoder::load(&config.encoder, vb.pp("encoder"))?;
        info!(
            "FastConformer encoder загружен: {} слоёв",
            config.encoder.n_layers
        );

        // 5. Decoder (prediction network)
        let prediction_net = PredictionNet::load(&config.decoder, vb.pp("decoder"))?;
        info!(
            "Prediction network загружен: LSTM {}×{}",
            config.decoder.num_lstm_layers, config.decoder.pred_hidden
        );

        // 6. Joint network
        let joint = JointNetwork::load(&config.joint, vb.pp("joint"))?;
        info!(
            "Joint network загружен: output_dim={}",
            config.joint.output_dim
        );

        // 7. TDT decoder
        let tdt_decoder = TdtGreedyDecoder::new(&config.tdt, config.decoder.blank_idx);

        // 8. Tokenizer
        let vocab_path = model_dir.join("vocab.json");
        let tokenizer = if vocab_path.exists() {
            SentencePieceTokenizer::from_vocab_json(&vocab_path)?
        } else {
            return Err(AsrError::Model(
                "vocab.json не найден для SentencePiece токенизатора".into(),
            ));
        };

        Ok(Self {
            encoder,
            prediction_net,
            joint,
            tdt_decoder,
            mel_extractor,
            tokenizer,
            device: device.clone(),
            config,
            model_dir,
        })
    }

    /// Encode audio → encoder frames.
    fn encode_chunk(&self, samples: &[f32]) -> AsrResult<candle_core::Tensor> {
        let mel = self.mel_extractor.extract(samples, &self.device)?;
        debug!("Mel-спектрограмма: {:?}", mel.shape());
        let encoder_output = self.encoder.forward(&mel)?;
        debug!("Encoder output: {:?}", encoder_output.shape());
        encoder_output
            .squeeze(0)
            .map_err(|e| AsrError::Inference(e.to_string()))
    }

    /// Транскрибация одного чанка аудио.
    fn transcribe_chunk(&self, samples: &[f32]) -> AsrResult<(String, Vec<(u32, u64)>)> {
        let encoder_output = self.encode_chunk(samples)?;
        let result = self
            .tdt_decoder
            .decode(&encoder_output, &self.prediction_net, &self.joint)
            .map_err(|e| AsrError::Inference(e.to_string()))?;
        let text = self.tokenizer.decode(&result.tokens);
        debug!("Transcript ({}): {}", result.tokens.len(), &text);
        Ok((text, result.timestamps))
    }

    /// Streaming транскрибация чанка: callback на каждый non-blank токен.
    fn transcribe_chunk_streaming(
        &self,
        samples: &[f32],
        callback: &dyn Fn(u32, u64),
    ) -> AsrResult<(String, Vec<(u32, u64)>)> {
        let encoder_output = self.encode_chunk(samples)?;
        let result = self
            .tdt_decoder
            .decode_streaming(
                &encoder_output,
                &self.prediction_net,
                &self.joint,
                callback,
            )
            .map_err(|e| AsrError::Inference(e.to_string()))?;
        let text = self.tokenizer.decode(&result.tokens);
        Ok((text, result.timestamps))
    }

    /// Проверяет word boundary по SentencePiece ▁ prefix.
    fn is_word_boundary(&self, token_id: u32) -> bool {
        let idx = token_id as usize;
        if idx < self.tokenizer.vocab.len() {
            self.tokenizer.vocab[idx].starts_with('▁')
        } else {
            false
        }
    }

    /// Декодирует один token ID в строку.
    fn decode_token(&self, token_id: u32) -> String {
        self.tokenizer.decode(&[token_id])
    }
}

impl AsrModel for ParakeetModel {
    fn name(&self) -> &str {
        &self.config.model_name
    }

    fn model_type(&self) -> ModelType {
        ModelType::Parakeet
    }

    fn sample_rate(&self) -> u32 {
        self.config.sample_rate as u32
    }

    fn supported_languages(&self) -> &[&str] {
        // Parakeet-TDT v3: 25 европейских языков
        &[
            "en", "de", "es", "fr", "it", "pt", "nl", "pl", "ro", "hu", "cs", "sk", "bg", "hr",
            "sl", "uk", "ru", "sv", "da", "fi", "nb", "el", "ca", "eu", "gl",
        ]
    }

    fn model_info(&self) -> ModelInfo {
        // Размер файла
        let weights_size = std::fs::metadata(self.model_dir.join("model.safetensors"))
            .map(|m| m.len())
            .ok();

        ModelInfo {
            model_type: ModelType::Parakeet,
            display_name: "Parakeet-TDT v3 (0.6B)".to_string(),
            parameters: Some(627_090_606),
            weights_size_bytes: weights_size,
            quantization: QuantizationType::None,
            languages: self
                .supported_languages()
                .iter()
                .map(|s| s.to_string())
                .collect(),
            backend: if self.device.is_metal() {
                "Metal".to_string()
            } else if self.device.is_cuda() {
                "CUDA".to_string()
            } else {
                "CPU".to_string()
            },
        }
    }

    fn transcribe(
        &mut self,
        samples: &[f32],
        _options: &TranscribeOptions,
    ) -> AsrResult<TranscriptionResult> {
        let start = Instant::now();
        let audio_duration = samples.len() as f64 / self.config.sample_rate as f64;

        info!(
            "Parakeet transcribe: {:.1}с аудио ({} сэмплов)",
            audio_duration,
            samples.len()
        );

        let chunk_samples = (CHUNK_DURATION_SECS * self.config.sample_rate as f64) as usize;
        let (text, all_timestamps) = if samples.len() <= chunk_samples {
            self.transcribe_chunk(samples)?
        } else {
            let mut parts = Vec::new();
            let mut all_ts = Vec::new();
            let mut offset = 0;
            while offset < samples.len() {
                let end = (offset + chunk_samples).min(samples.len());
                let chunk = &samples[offset..end];
                let (chunk_text, ts) = self.transcribe_chunk(chunk)?;
                let offset_ms = (offset as u64 * 1000) / self.config.sample_rate as u64;
                for (tok, ms) in &ts {
                    all_ts.push((*tok, ms + offset_ms));
                }
                if !chunk_text.is_empty() {
                    parts.push(chunk_text);
                }
                offset = end;
            }
            (parts.join(" "), all_ts)
        };

        let segments = self.timestamps_to_segments(&all_timestamps);

        let inference_time = start.elapsed().as_secs_f64();
        let rtf = if audio_duration > 0.0 {
            inference_time / audio_duration
        } else {
            0.0
        };

        info!(
            "Parakeet: {:.1}с инференса, RTF={:.3} | {}",
            inference_time, rtf, &text
        );

        Ok(TranscriptionResult {
            text,
            inference_time_secs: inference_time,
            audio_duration_secs: audio_duration,
            rtf,
            model_name: self.config.model_name.clone(),
            segments,
            language: None,
        })
    }
}

impl ParakeetModel {
    /// Streaming транскрибация: callback(text, is_word_boundary, timestamp_ms).
    pub fn transcribe_streaming(
        &mut self,
        samples: &[f32],
        _options: &TranscribeOptions,
        word_callback: impl Fn(&str, bool, u64),
    ) -> AsrResult<TranscriptionResult> {
        let start = Instant::now();
        let audio_duration = samples.len() as f64 / self.config.sample_rate as f64;

        info!(
            "Parakeet streaming transcribe: {:.1}с аудио ({} сэмплов)",
            audio_duration,
            samples.len()
        );

        let callback = |token_id: u32, ts_ms: u64| {
            let text = self.decode_token(token_id);
            let is_wb = self.is_word_boundary(token_id);
            if !text.trim().is_empty() {
                word_callback(&text, is_wb, ts_ms);
            }
        };

        let chunk_samples = (CHUNK_DURATION_SECS * self.config.sample_rate as f64) as usize;
        let (text, all_timestamps) = if samples.len() <= chunk_samples {
            self.transcribe_chunk_streaming(samples, &callback)?
        } else {
            let mut parts = Vec::new();
            let mut all_ts = Vec::new();
            let mut offset = 0;
            while offset < samples.len() {
                let end = (offset + chunk_samples).min(samples.len());
                let chunk = &samples[offset..end];
                let (chunk_text, ts) = self.transcribe_chunk_streaming(chunk, &callback)?;
                let offset_ms = (offset as u64 * 1000) / self.config.sample_rate as u64;
                for (tok, ms) in &ts {
                    all_ts.push((*tok, ms + offset_ms));
                }
                if !chunk_text.is_empty() {
                    parts.push(chunk_text);
                }
                offset = end;
            }
            (parts.join(" "), all_ts)
        };

        let segments = self.timestamps_to_segments(&all_timestamps);

        let inference_time = start.elapsed().as_secs_f64();
        let rtf = if audio_duration > 0.0 {
            inference_time / audio_duration
        } else {
            0.0
        };

        debug!(
            "Parakeet streaming: {:.1}с инференса, RTF={:.3}",
            inference_time, rtf,
        );

        Ok(TranscriptionResult {
            text,
            inference_time_secs: inference_time,
            audio_duration_secs: audio_duration,
            rtf,
            model_name: self.config.model_name.clone(),
            segments,
            language: None,
        })
    }

    /// Конвертация raw token timestamps в word-level Segments.
    fn timestamps_to_segments(
        &self,
        timestamps: &[(u32, u64)],
    ) -> Vec<asr_core::Segment> {
        use asr_core::Segment;
        let mut segments = Vec::new();
        let mut word_tokens: Vec<u32> = Vec::new();
        let mut word_start_ms: u64 = 0;

        for &(tok, ts_ms) in timestamps {
            let is_wb = self.is_word_boundary(tok);
            if is_wb && !word_tokens.is_empty() {
                let text = self.tokenizer.decode(&word_tokens);
                let text = text.trim().to_string();
                if !text.is_empty() {
                    segments.push(Segment {
                        start: word_start_ms as f64 / 1000.0,
                        end: ts_ms as f64 / 1000.0,
                        text,
                        confidence: None,
                    });
                }
                word_tokens.clear();
            }
            if word_tokens.is_empty() {
                word_start_ms = ts_ms;
            }
            word_tokens.push(tok);
        }

        if !word_tokens.is_empty() {
            let text = self.tokenizer.decode(&word_tokens);
            let text = text.trim().to_string();
            let end_ms = timestamps.last().map(|(_, ms)| *ms + 80).unwrap_or(word_start_ms);
            if !text.is_empty() {
                segments.push(Segment {
                    start: word_start_ms as f64 / 1000.0,
                    end: end_ms as f64 / 1000.0,
                    text,
                    confidence: None,
                });
            }
        }

        segments
    }
}
