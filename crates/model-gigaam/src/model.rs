//! GigaAmModel — обёртка над Conformer-энкодером с CTC-декодером.
//!
//! Реализует [`AsrModel`] trait для единого интерфейса со всеми моделями.

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use tracing::{debug, info};

use asr_core::{
    AsrError, AsrModel, AsrResult, ModelInfo, ModelType, QuantizationType,
    Segment, TranscribeOptions, TranscriptionResult,
    metal_utils,
};

use crate::config::GigaAmConfig;
use crate::conformer::ConformerEncoder;
use crate::ctc::{CtcGreedyDecoder, CtcHead};
use crate::mel::GigaAmMelExtractor;

/// GigaAM v3 E2E CTC — Conformer ASR-модель.
pub struct GigaAmModel {
    /// Conformer-энкодер (16 слоёв, 768 dim).
    encoder: ConformerEncoder,
    /// CTC-голова (проекция на словарь).
    head: CtcHead,
    /// CTC greedy-декодер.
    decoder: CtcGreedyDecoder,
    /// Mel-спектрограмма (GigaAM-совместимая).
    mel_extractor: GigaAmMelExtractor,
    /// Устройство (CPU, Metal, CUDA).
    device: Device,
    /// Конфигурация модели.
    config: GigaAmConfig,
    /// Путь к директории модели.
    #[allow(dead_code)]
    model_dir: PathBuf,
}

impl GigaAmModel {
    /// Загрузить модель из директории.
    ///
    /// Ожидаемые файлы:
    /// - `model.safetensors` — веса (encoder + head)
    /// - `config.json` — конфигурация
    /// - `vocab.json` — словарь CTC
    pub fn load(model_dir: impl AsRef<Path>, device: &Device) -> AsrResult<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        info!("GigaAM: загрузка модели из {:?}", model_dir);

        // 1. Загрузить конфигурацию
        let config = Self::load_config(&model_dir)?;
        info!(
            "GigaAM: {} ({} слоёв, d_model={}, {} голов)",
            config.model_name,
            config.encoder.n_layers,
            config.encoder.d_model,
            config.encoder.n_heads,
        );

        // 2. Загрузить веса
        let safetensors_path = model_dir.join("model.safetensors");
        if !safetensors_path.exists() {
            return Err(AsrError::Model(format!(
                "Файл model.safetensors не найден в {:?}. \
                 Используйте scripts/convert_gigaam.py для конвертации.",
                model_dir
            )));
        }

        let start = Instant::now();

        // Определить dtype: f16 на GPU, f32 на CPU
        // Metal не поддерживает conv1d в F16, используем F32.
        // CUDA может работать в F16/BF16.
        let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[safetensors_path],
                dtype,
                device,
            )?
        };

        // 3. Создать энкодер
        let encoder = ConformerEncoder::load(&config.encoder, vb.pp("encoder"))?;
        debug!("GigaAM: энкодер загружен");

        // 4. Создать CTC-голову
        let head = CtcHead::load(
            config.head.feat_in,
            config.head.num_classes,
            vb.pp("head"),
        )?;
        debug!("GigaAM: CTC-голова загружена");

        // 5. Создать CTC-декодер (загрузить словарь)
        let vocab_path = model_dir.join("vocab.json");
        if !vocab_path.exists() {
            return Err(AsrError::Model(format!(
                "Файл vocab.json не найден в {:?}. \
                 Используйте scripts/convert_gigaam.py для конвертации.",
                model_dir
            )));
        }
        let blank_id = config.head.num_classes - 1;
        let decoder = CtcGreedyDecoder::from_vocab_file(&vocab_path, blank_id)?;
        debug!("GigaAM: CTC-декодер загружен (vocab_size={})", decoder.vocab_size());

        // 6. Создать mel-экстрактор
        let mel_extractor = GigaAmMelExtractor::new(config.preprocessor.clone());

        let elapsed = start.elapsed();
        info!("GigaAM: модель загружена за {:.2}с", elapsed.as_secs_f64());

        Ok(Self {
            encoder,
            head,
            decoder,
            mel_extractor,
            device: device.clone(),
            config,
            model_dir,
        })
    }

    /// Загрузить конфигурацию из config.json.
    fn load_config(model_dir: &Path) -> AsrResult<GigaAmConfig> {
        let config_path = model_dir.join("config.json");
        if config_path.exists() {
            let data = std::fs::read_to_string(&config_path)?;
            let config: GigaAmConfig = serde_json::from_str(&data)?;
            Ok(config)
        } else {
            info!("GigaAM: config.json не найден, использую конфигурацию по умолчанию");
            Ok(GigaAmConfig::v3_e2e_ctc())
        }
    }
}

impl AsrModel for GigaAmModel {
    fn name(&self) -> &str {
        &self.config.model_name
    }

    fn model_type(&self) -> ModelType {
        ModelType::GigaAm
    }

    fn sample_rate(&self) -> u32 {
        self.config.sample_rate as u32
    }

    fn supported_languages(&self) -> &[&str] {
        // GigaAM — модель для русского языка
        &["ru"]
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            model_type: ModelType::GigaAm,
            display_name: format!("GigaAM {}", self.config.model_name),
            parameters: Some(220_000_000), // ~220M параметров
            weights_size_bytes: Some(442_000_000), // ~442MB (f16)
            quantization: QuantizationType::None,
            languages: vec!["ru".to_string()],
            backend: "candle".to_string(),
        }
    }

    fn transcribe(
        &mut self,
        samples: &[f32],
        _options: &TranscribeOptions,
    ) -> AsrResult<TranscriptionResult> {
        let start = Instant::now();
        let sample_rate = self.config.sample_rate;
        let audio_duration = samples.len() as f64 / sample_rate as f64;

        info!(
            "GigaAM: транскрибация {:.1}с аудио ({} сэмплов)",
            audio_duration,
            samples.len()
        );

        // Максимальная длина чанка (~30с).
        // Self-attention O(n²) — для длинных аудио необходима чанковая обработка.
        let chunk_secs = 30.0;
        let chunk_samples = (chunk_secs * sample_rate as f64) as usize;

        if samples.len() <= chunk_samples {
            // Короткий аудиофрагмент — обрабатываем целиком.
            let text = self.transcribe_chunk(samples)?;
            let inference_time = start.elapsed().as_secs_f64();
            let rtf = if audio_duration > 0.0 {
                inference_time / audio_duration
            } else {
                0.0
            };

            info!(
                "GigaAM: '{}' (RTF={:.3}, {:.2}с)",
                &text[..text.len().min(80)],
                rtf,
                inference_time,
            );

            return Ok(TranscriptionResult {
                text,
                inference_time_secs: inference_time,
                audio_duration_secs: audio_duration,
                rtf,
                model_name: self.config.model_name.clone(),
                segments: vec![Segment {
                    start: 0.0,
                    end: audio_duration,
                    text: String::new(),
                    confidence: None,
                }],
                language: Some("ru".to_string()),
            });
        }

        // Длинное аудио — разбиваем на чанки.
        let mut all_text = String::new();
        let mut segments = Vec::new();
        let mut offset = 0usize;
        let mut chunk_idx = 0usize;

        while offset < samples.len() {
            let end = (offset + chunk_samples).min(samples.len());
            let chunk = &samples[offset..end];
            let chunk_start_secs = offset as f64 / sample_rate as f64;
            let chunk_end_secs = end as f64 / sample_rate as f64;

            debug!(
                "GigaAM: чанк {} ({:.1}с - {:.1}с, {} сэмплов)",
                chunk_idx,
                chunk_start_secs,
                chunk_end_secs,
                chunk.len()
            );

            let chunk_text = self.transcribe_chunk(chunk)?;

            if !chunk_text.is_empty() {
                if !all_text.is_empty() {
                    all_text.push(' ');
                }
                all_text.push_str(&chunk_text);
                segments.push(Segment {
                    start: chunk_start_secs,
                    end: chunk_end_secs,
                    text: chunk_text,
                    confidence: None,
                });
            }

            offset = end;
            chunk_idx += 1;
        }

        let inference_time = start.elapsed().as_secs_f64();
        let rtf = if audio_duration > 0.0 {
            inference_time / audio_duration
        } else {
            0.0
        };

        info!(
            "GigaAM: {} чанков, RTF={:.3}, {:.2}с инференс",
            chunk_idx, rtf, inference_time,
        );

        Ok(TranscriptionResult {
            text: all_text,
            inference_time_secs: inference_time,
            audio_duration_secs: audio_duration,
            rtf,
            model_name: self.config.model_name.clone(),
            segments,
            language: Some("ru".to_string()),
        })
    }
}

impl GigaAmModel {
    /// Транскрибировать один чанк аудио.
    fn transcribe_chunk(&self, samples: &[f32]) -> AsrResult<String> {
        // 1. Mel-спектрограмма (CPU via rustfft → загрузка на device)
        let mel = self.mel_extractor.extract(samples, &self.device)?;
        let mel = mel.to_dtype(DType::F32)?;

        // Metal sync: ждём завершения загрузки данных на GPU.
        // Предотвращает конфликт буферов при агрессивном переиспользовании
        // в Metal command buffer pool (workaround для AGXMetalG16X fillBuffer bug).
        metal_utils::metal_sync(&self.device)?;

        // 2. Encoder
        let encoded = self.encoder.forward(&mel)?;

        // Metal sync: ждём завершения encoder'а перед CTC head.
        metal_utils::metal_sync(&self.device)?;

        // 3. CTC head → log_probs
        let log_probs = self.head.forward(&encoded)?;

        // 4. CTC greedy decode
        let log_probs = log_probs.squeeze(0)?;
        let log_probs_f32 = log_probs.to_dtype(DType::F32)?;
        Ok(self.decoder.decode(&log_probs_f32)?)
    }
}