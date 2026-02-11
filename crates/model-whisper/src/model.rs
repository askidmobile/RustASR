//! WhisperModel — обёртка над candle-transformers Whisper
//! с реализацией `AsrModel` trait.

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, Config as WhisperConfig};
use tokenizers::Tokenizer;
use tracing::{debug, info};

use asr_core::{
    AsrError, AsrModel, AsrResult, FeatureExtractorConfig, ModelInfo, ModelType, QuantizationType,
    Segment, TranscribeOptions, TranscriptionResult,
};
use audio::MelSpectrogramExtractor;

use crate::decoder::{Task, WhisperDecoder};

/// Внутреннее представление модели (fp16/fp32 или квантизированная).
enum InnerModel {
    Normal(m::model::Whisper),
    Quantized(m::quantized_model::Whisper),
}

impl InnerModel {
    fn config(&self) -> &WhisperConfig {
        match self {
            Self::Normal(m) => &m.config,
            Self::Quantized(m) => &m.config,
        }
    }

    fn encoder_forward(&mut self, x: &Tensor, flush: bool) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(m) => m.encoder.forward(x, flush),
            Self::Quantized(m) => m.encoder.forward(x, flush),
        }
    }

    fn decoder_forward(
        &mut self,
        x: &Tensor,
        xa: &Tensor,
        flush: bool,
    ) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.forward(x, xa, flush),
            Self::Quantized(m) => m.decoder.forward(x, xa, flush),
        }
    }

    fn decoder_final_linear(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.final_linear(x),
            Self::Quantized(m) => m.decoder.final_linear(x),
        }
    }

    fn reset_kv_cache(&mut self) {
        match self {
            Self::Normal(m) => m.reset_kv_cache(),
            Self::Quantized(m) => m.reset_kv_cache(),
        }
    }
}

/// Whisper ASR модель.
///
/// Поддерживает все варианты Whisper (tiny, base, small, medium, large-v3, large-v3-turbo)
/// в форматах safetensors (f16/f32) и GGUF (квантизация Q4/Q8).
pub struct WhisperModel {
    model: InnerModel,
    tokenizer: Tokenizer,
    mel_extractor: MelSpectrogramExtractor,
    device: Device,
    model_name: String,
    #[allow(dead_code)]
    model_dir: PathBuf,
    quantization: QuantizationType,
    /// Mel-фильтры из safetensors (если загружены из файла модели).
    /// Будет использоваться при точном воспроизведении HF mel-фильтров.
    #[allow(dead_code)]
    mel_filters: Vec<f32>,
    /// Мультиязычная ли модель (large-v3, turbo — да; tiny.en — нет).
    is_multilingual: bool,
}

impl WhisperModel {
    /// Загрузить модель Whisper из директории.
    ///
    /// # Аргументы
    /// * `model_dir` — директория с файлами модели:
    ///   - `config.json` (обязательно)
    ///   - `model.safetensors` или `model.gguf` (веса)
    ///   - `tokenizer.json` (токенайзер)
    ///   - `mel_filters.safetensors` (опционально, для точных mel-фильтров)
    /// * `device` — устройство (CPU, Metal, CUDA)
    pub fn load(model_dir: impl AsRef<Path>, device: &Device) -> AsrResult<Self> {
        Self::load_with_options(model_dir, device, false)
    }

    /// Загрузить квантизированную модель (GGUF).
    pub fn load_quantized(model_dir: impl AsRef<Path>, device: &Device) -> AsrResult<Self> {
        Self::load_with_options(model_dir, device, true)
    }

    fn load_with_options(
        model_dir: impl AsRef<Path>,
        device: &Device,
        quantized: bool,
    ) -> AsrResult<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        info!("Загрузка Whisper из {:?}, quantized={}", model_dir, quantized);

        // 1. Загрузка конфигурации
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| AsrError::Model(format!("Cannot read config.json: {e}")))?;
        let config: WhisperConfig = serde_json::from_str(&config_str)?;
        debug!("Whisper config: {:?}", config);

        // 2. Загрузка токенайзера
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| AsrError::Model(format!("Cannot load tokenizer: {e}")))?;

        // 3. Загрузка mel-фильтров
        let mel_filters = Self::load_mel_filters(&model_dir, config.num_mel_bins)?;

        // 4. Определяем мультиязычность
        let is_multilingual = config.vocab_size >= 51865;

        // 5. Загрузка весов
        let (model, quantization) = if quantized {
            let gguf_path = Self::find_gguf_file(&model_dir)?;
            info!("Загрузка GGUF весов из {:?}", gguf_path);
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                &gguf_path,
                device,
            )?;
            let whisper = m::quantized_model::Whisper::load(&vb, config.clone())?;
            (InnerModel::Quantized(whisper), QuantizationType::GgufQ8_0)
        } else {
            let safetensors_path = Self::find_safetensors_file(&model_dir)?;
            info!("Загрузка safetensors весов из {:?}", safetensors_path);
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[safetensors_path],
                    m::DTYPE,
                    device,
                )?
            };
            let whisper = m::model::Whisper::load(&vb, config.clone())?;
            (InnerModel::Normal(whisper), QuantizationType::None)
        };

        // 6. Mel-экстрактор с Whisper-конфигурацией
        let mel_config = FeatureExtractorConfig::whisper();
        let mel_extractor = MelSpectrogramExtractor::new(mel_config);

        // 7. Определяем имя модели
        let model_name = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("whisper")
            .to_string();

        info!(
            "Whisper загружен: {}, {} слоёв encoder, {} слоёв decoder, vocab={}",
            model_name,
            model.config().encoder_layers,
            model.config().decoder_layers,
            model.config().vocab_size,
        );

        Ok(Self {
            model,
            tokenizer,
            mel_extractor,
            device: device.clone(),
            model_name,
            model_dir,
            quantization,
            mel_filters,
            is_multilingual,
        })
    }

    /// Загрузить mel-фильтры из safetensors файла (mel_filters.safetensors).
    fn load_mel_filters(model_dir: &Path, num_mel_bins: usize) -> AsrResult<Vec<f32>> {
        // Ищем mel_filters.safetensors (стандартный HF формат)
        let mel_path = model_dir.join("mel_filters.safetensors");
        if mel_path.exists() {
            let data = std::fs::read(&mel_path)?;
            let tensors = safetensors::SafeTensors::deserialize(&data)
                .map_err(|e| AsrError::Model(format!("Cannot load mel_filters: {e}")))?;

            // Пробуем ключи mel_80 или mel_128
            let key = format!("mel_{num_mel_bins}");
            let tensor = tensors
                .tensor(&key)
                .or_else(|_| tensors.tensor("mel_80"))
                .or_else(|_| tensors.tensor("mel_128"))
                .map_err(|_| {
                    AsrError::Model(format!("No mel filter tensor found in mel_filters.safetensors"))
                })?;

            let mel_tensor = Tensor::from_raw_buffer(
                tensor.data(),
                DType::F32,
                tensor.shape(),
                &Device::Cpu,
            )?;
            let mel_filters: Vec<f32> = mel_tensor.flatten_all()?.to_vec1()?;
            debug!("Загружено {} mel-фильтров из файла", mel_filters.len());
            return Ok(mel_filters);
        }

        // Если файла нет — генерируем программно (для совместимости)
        debug!("mel_filters.safetensors не найден, генерируем программно");
        Ok(Vec::new())
    }

    /// Найти .gguf файл в директории модели.
    fn find_gguf_file(model_dir: &Path) -> AsrResult<PathBuf> {
        for entry in std::fs::read_dir(model_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "gguf") {
                return Ok(path);
            }
        }
        Err(AsrError::Model(format!(
            "No .gguf file found in {:?}",
            model_dir
        )))
    }

    /// Найти .safetensors файл(ы) в директории модели.
    fn find_safetensors_file(model_dir: &Path) -> AsrResult<PathBuf> {
        let single = model_dir.join("model.safetensors");
        if single.exists() {
            return Ok(single);
        }
        // Ищем первый safetensors файл
        for entry in std::fs::read_dir(model_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "safetensors")
                && path
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("model"))
            {
                return Ok(path);
            }
        }
        Err(AsrError::Model(format!(
            "No model.safetensors found in {:?}",
            model_dir
        )))
    }

    /// Получить токен языка по ISO 639-1 коду.
    fn language_token(&self, language: &str) -> Option<u32> {
        // Whisper использует токены вида <|ru|>, <|en|> и т.д.
        let token_str = format!("<|{language}|>");
        self.tokenizer
            .token_to_id(&token_str)
    }

    /// Специальные токены Whisper.
    fn special_tokens(&self) -> AsrResult<(u32, u32, u32, u32, u32, u32)> {
        let tok = |s: &str| -> AsrResult<u32> {
            self.tokenizer.token_to_id(s).ok_or_else(|| {
                AsrError::Model(format!("Special token '{s}' not found in tokenizer"))
            })
        };

        Ok((
            tok("<|startoftranscript|>")?,
            tok("<|endoftext|>")?,
            tok("<|transcribe|>")?,
            tok("<|translate|>")?,
            tok("<|nospeech|>").unwrap_or(tok("<|nocaptions|>").unwrap_or(50362)),
            tok("<|notimestamps|>")?,
        ))
    }

    /// Mel-спектрограмма из аудио-сэмплов.
    ///
    /// Whisper ожидает 30-секундные сегменты, дополненные нулями.
    fn compute_mel(&self, samples: &[f32]) -> AsrResult<Tensor> {
        let mel = self.mel_extractor.extract(samples, &self.device)?;
        // Whisper ожидает формат [1, n_mels, time] (канал первый)
        let mel_t = mel.tensor.transpose(1, 2)?;
        Ok(mel_t)
    }

    /// Паддинг/обрезка аудио до 30 секунд.
    fn pad_or_trim_audio(samples: &[f32], target_len: usize) -> Vec<f32> {
        if samples.len() >= target_len {
            samples[..target_len].to_vec()
        } else {
            let mut padded = samples.to_vec();
            padded.resize(target_len, 0.0);
            padded
        }
    }
}

impl AsrModel for WhisperModel {
    fn name(&self) -> &str {
        &self.model_name
    }

    fn model_type(&self) -> ModelType {
        ModelType::Whisper
    }

    fn sample_rate(&self) -> u32 {
        16_000
    }

    fn supported_languages(&self) -> &[&str] {
        if self.is_multilingual {
            // Whisper large-v3 поддерживает 99+ языков, показываем основные
            &["en", "ru", "zh", "de", "fr", "es", "it", "ja", "ko", "pt"]
        } else {
            &["en"]
        }
    }

    fn model_info(&self) -> ModelInfo {
        let params = match self.model.config().encoder_layers {
            4 => 39_000_000,   // tiny
            6 => 74_000_000,   // base
            12 => 244_000_000, // small
            24 => 769_000_000, // medium
            32 => {
                if self.model.config().decoder_layers <= 4 {
                    809_000_000 // large-v3-turbo
                } else {
                    1_550_000_000 // large-v3
                }
            }
            _ => 0,
        };

        ModelInfo::new(ModelType::Whisper)
            .with_parameters(params)
            .with_quantization(self.quantization)
            .with_languages(
                self.supported_languages()
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            )
    }

    fn transcribe(
        &mut self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> AsrResult<TranscriptionResult> {
        let start = Instant::now();
        let audio_duration_secs = samples.len() as f64 / self.sample_rate() as f64;

        // Специальные токены
        let (sot, eot, transcribe, translate, no_speech, no_timestamps) =
            self.special_tokens()?;

        // Токен языка
        let language_token = options
            .language
            .as_ref()
            .and_then(|lang| self.language_token(lang));

        // Suppress tokens из конфига
        let suppress = self.model.config().suppress_tokens.clone();
        let suppress_tokens = Tensor::new(suppress.as_slice(), &self.device)?;

        // Сегменты по 30 секунд
        let chunk_samples = 30 * self.sample_rate() as usize; // 480000
        let sample_len = self.model.config().max_target_positions / 2;
        let mut all_text = String::new();
        let mut all_segments = Vec::new();

        let chunks: Vec<&[f32]> = if samples.len() <= chunk_samples {
            vec![samples]
        } else {
            samples.chunks(chunk_samples).collect()
        };

        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            let chunk_start_secs = chunk_idx as f64 * 30.0;

            // Паддинг до 30 секунд
            let padded = Self::pad_or_trim_audio(chunk, chunk_samples);

            // Mel-спектрограмма
            let mel = self.compute_mel(&padded)?;

            // Кэш энкодера
            self.model.reset_kv_cache();

            // Прогон через аудио-энкодер
            let audio_features = self.model.encoder_forward(&mel, true)?;

            // Создаём декодер для этого сегмента
            let mut decoder = WhisperDecoder::new(
                sot,
                eot,
                transcribe,
                translate,
                no_speech,
                no_timestamps,
                language_token,
                Task::Transcribe,
                options.timestamps,
                suppress_tokens.clone(),
                42 + chunk_idx as u64,
            );

            // Декодирование
            let temperature = options.temperature as f64;
            let mut result = if temperature == 0.0 {
                decoder.decode_with_fallback(&audio_features, sample_len, |tokens, xa, flush| {
                    let hidden = self.model.decoder_forward(tokens, xa, flush)?;
                    self.model.decoder_final_linear(&hidden)
                })?
            } else {
                decoder.decode_segment(
                    &audio_features,
                    sample_len,
                    temperature,
                    |tokens, xa, flush| {
                        let hidden = self.model.decoder_forward(tokens, xa, flush)?;
                        self.model.decoder_final_linear(&hidden)
                    },
                )?
            };

            // Декодирование токенов в текст
            let text = self
                .tokenizer
                .decode(&result.tokens, true)
                .map_err(|e| AsrError::Inference(format!("Tokenizer decode error: {e}")))?;
            result.text = text.clone();

            // Добавляем сегмент
            let chunk_duration = chunk.len() as f64 / self.sample_rate() as f64;
            all_segments.push(Segment {
                start: chunk_start_secs,
                end: chunk_start_secs + chunk_duration,
                text: text.clone(),
                confidence: Some(result.avg_logprob.exp()),
            });

            if !all_text.is_empty() {
                all_text.push(' ');
            }
            all_text.push_str(&text);
        }

        let inference_time = start.elapsed().as_secs_f64();
        let mut result =
            TranscriptionResult::new(all_text, self.model_name.clone(), inference_time, audio_duration_secs);

        if !all_segments.is_empty() {
            result = result.with_segments(all_segments);
        }
        if let Some(lang) = &options.language {
            result = result.with_language(lang.clone());
        }

        info!(
            "Whisper: транскрибация {:.1}с аудио за {:.2}с (RTF={:.3})",
            audio_duration_secs, inference_time, result.rtf
        );

        Ok(result)
    }
}
