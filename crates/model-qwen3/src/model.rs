//! Qwen3AsrModel — обёртка `AsrPipeline` через `AsrModel` trait.
//!
//! Qwen3-ASR — мультиязычная модель (AuT encoder + Qwen3 LLM decoder).
//! Поддерживает safetensors и GGUF квантизацию для декодера.

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::Device;
use tracing::{debug, info};

use asr_core::{
    AsrError, AsrModel, AsrResult, ModelInfo, ModelType, QuantizationType, TranscribeOptions,
    TranscriptionResult,
};
use asr_pipeline::{AsrPipeline, DecoderWeights};

/// Qwen3-ASR модель.
///
/// Обёртка вокруг [`AsrPipeline`], реализующая [`AsrModel`] trait для
/// единого интерфейса через `AsrEngine`.
pub struct Qwen3AsrModel {
    pipeline: AsrPipeline,
    model_dir: PathBuf,
    quantization: QuantizationType,
    model_name: String,
}

impl Qwen3AsrModel {
    /// Загрузить модель из директории (safetensors, предпочитая GGUF если есть).
    ///
    /// # Аргументы
    /// * `model_dir` — директория с файлами модели:
    ///   - `config.json` (обязательно)
    ///   - `model.safetensors` или `model-*.safetensors` (веса)
    ///   - `vocab.json` + `merges.txt` (токенайзер)
    ///   - `model-*.gguf` (опционально, для квантизированного декодера)
    /// * `device` — устройство (CPU, Metal, CUDA)
    pub fn load(model_dir: impl AsRef<Path>, device: &Device) -> AsrResult<Self> {
        Self::load_inner(model_dir.as_ref(), device, DecoderWeights::Auto, None)
    }

    /// Загрузить квантизированную модель (GGUF для декодера).
    pub fn load_quantized(model_dir: impl AsRef<Path>, device: &Device) -> AsrResult<Self> {
        Self::load_inner(model_dir.as_ref(), device, DecoderWeights::Gguf, None)
    }

    /// Загрузить с явным выбором формата весов декодера и опциональным GGUF-файлом.
    pub fn load_with_options(
        model_dir: impl AsRef<Path>,
        device: &Device,
        decoder_weights: DecoderWeights,
        decoder_gguf: Option<PathBuf>,
    ) -> AsrResult<Self> {
        Self::load_inner(model_dir.as_ref(), device, decoder_weights, decoder_gguf)
    }

    fn load_inner(
        model_dir: &Path,
        device: &Device,
        decoder_weights: DecoderWeights,
        decoder_gguf: Option<PathBuf>,
    ) -> AsrResult<Self> {
        info!("Загрузка Qwen3-ASR из {:?}", model_dir);

        let pipeline = AsrPipeline::from_model_dir_with_decoder_weights_and_gguf(
            model_dir,
            device,
            decoder_weights,
            decoder_gguf,
        )
        .map_err(|e| AsrError::Model(format!("Ошибка загрузки Qwen3-ASR: {e}")))?;

        // Определяем квантизацию по наличию GGUF-файлов
        let quantization = Self::detect_quantization(model_dir, decoder_weights);

        // Извлекаем имя модели из имени директории
        let model_name = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("qwen3-asr")
            .to_string();

        info!("Qwen3-ASR загружена: {}, квантизация: {}", model_name, quantization);

        Ok(Self {
            pipeline,
            model_dir: model_dir.to_path_buf(),
            quantization,
            model_name,
        })
    }

    /// Определить тип квантизации по файлам в директории.
    fn detect_quantization(model_dir: &Path, decoder_weights: DecoderWeights) -> QuantizationType {
        match decoder_weights {
            DecoderWeights::Safetensors => return QuantizationType::None,
            _ => {}
        }

        // Проверяем наличие GGUF-файлов
        if let Ok(entries) = std::fs::read_dir(model_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.ends_with(".gguf") {
                    if name.contains("q4") || name.contains("Q4") {
                        return QuantizationType::GgufQ4_0;
                    } else if name.contains("q6k") || name.contains("Q6K") || name.contains("q6_k") {
                        return QuantizationType::GgufQ6K;
                    } else if name.contains("q8") || name.contains("Q8") {
                        return QuantizationType::GgufQ8_0;
                    }
                    // GGUF есть, но тип не определён — считаем Q8
                    return QuantizationType::GgufQ8_0;
                }
            }
        }

        QuantizationType::None
    }

    /// Оценить max_tokens по длительности аудио.
    fn estimate_max_tokens(duration_secs: f64) -> usize {
        // ~10 токенов/секунду для типичной речи, с запасом ×2.5
        let estimate = (duration_secs * 25.0) as usize;
        estimate.clamp(64, 4096)
    }
}

impl AsrModel for Qwen3AsrModel {
    fn name(&self) -> &str {
        &self.model_name
    }

    fn model_type(&self) -> ModelType {
        ModelType::Qwen3Asr
    }

    fn sample_rate(&self) -> u32 {
        16_000
    }

    fn supported_languages(&self) -> &[&str] {
        // Qwen3-ASR: мультиязычная (основные)
        &[
            "en", "ru", "zh", "de", "fr", "es", "it", "ja", "ko", "pt",
            "nl", "pl", "ar", "hi", "th", "vi", "tr", "id", "ms", "sv",
        ]
    }

    fn model_info(&self) -> ModelInfo {
        let weights_size = self.weights_total_size();

        // Определяем количество параметров по имени модели
        let parameters = if self.model_name.contains("1.7b") || self.model_name.contains("1.7B") {
            Some(1_700_000_000u64)
        } else {
            Some(600_000_000u64) // 0.6B по умолчанию
        };

        ModelInfo {
            model_type: ModelType::Qwen3Asr,
            display_name: format!("Qwen3-ASR ({})", self.model_name),
            parameters,
            weights_size_bytes: weights_size,
            quantization: self.quantization,
            languages: self
                .supported_languages()
                .iter()
                .map(|s| s.to_string())
                .collect(),
            backend: "Candle".to_string(),
        }
    }

    fn transcribe(
        &mut self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> AsrResult<TranscriptionResult> {
        let start = Instant::now();
        let audio_duration_secs = samples.len() as f64 / self.sample_rate() as f64;

        info!(
            "Qwen3-ASR transcribe: {:.1}с аудио ({} сэмплов)",
            audio_duration_secs,
            samples.len()
        );

        let max_tokens = options
            .max_tokens
            .unwrap_or_else(|| Self::estimate_max_tokens(audio_duration_secs));

        let language = options.language.as_deref();

        let result = self
            .pipeline
            .transcribe_to_result_with_max_tokens_and_language(samples, max_tokens, language)
            .map_err(|e| AsrError::Inference(format!("Qwen3-ASR ошибка инференса: {e}")))?;

        let inference_time = start.elapsed().as_secs_f64();
        let rtf = if audio_duration_secs > 0.0 {
            inference_time / audio_duration_secs
        } else {
            0.0
        };

        debug!(
            "Qwen3-ASR: {:.1}с инференса, RTF={:.3}, lang={}, tokens={}, stop={:?}",
            inference_time, rtf, result.language, result.generated_tokens, result.stop_reason,
        );

        let detected_language = if result.language.is_empty() {
            None
        } else {
            Some(result.language)
        };

        Ok(TranscriptionResult {
            text: result.text,
            inference_time_secs: inference_time,
            audio_duration_secs,
            rtf,
            model_name: self.model_name.clone(),
            segments: vec![],
            language: detected_language,
        })
    }
}

impl Qwen3AsrModel {
    /// Суммарный размер файлов весов.
    fn weights_total_size(&self) -> Option<u64> {
        let mut total = 0u64;
        if let Ok(entries) = std::fs::read_dir(&self.model_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.ends_with(".safetensors") || name.ends_with(".gguf") {
                    if let Ok(meta) = entry.metadata() {
                        total += meta.len();
                    }
                }
            }
        }
        if total > 0 { Some(total) } else { None }
    }
}
