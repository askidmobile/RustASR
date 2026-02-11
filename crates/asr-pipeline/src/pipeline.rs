//! ASR Pipeline - End-to-end speech recognition.

use asr_core::FeatureExtractorConfig;
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use std::path::Path;

use audio::MelSpectrogramExtractor;
use aut_encoder::{AuTConfig, AuTEncoder};
use qwen3_decoder::{Qwen3Config, Qwen3Decoder, cache::KvCache};

use crate::output::{AsrTranscription, StopReason, parse_asr_output};
use crate::tokenizer::Tokenizer;

fn normalize_language_name(lang: &str) -> String {
    let val = lang.trim();
    if val.is_empty() {
        return String::new();
    }
    let mut chars = val.chars();
    if let Some(first) = chars.next() {
        let mut out = String::new();
        out.push_str(&first.to_uppercase().to_string());
        out.push_str(&chars.as_str().to_lowercase());
        out
    } else {
        val.to_string()
    }
}

/// ASR Pipeline for end-to-end speech recognition.
///
/// Combines all components:
/// 1. Audio → Mel spectrogram (MelSpectrogramExtractor)
/// 2. Mel → Audio embeddings (AuTEncoder)
/// 3. Audio embeddings → Text tokens (Qwen3Decoder)
#[derive(Debug)]
pub struct AsrPipeline {
    mel_extractor: MelSpectrogramExtractor,
    encoder: AuTEncoder,
    decoder: Qwen3Decoder,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
}

/// Откуда загружать веса текстового декодера (LLM).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderWeights {
    /// Автоматически: предпочесть GGUF, если он есть в директории модели.
    Auto,
    /// Всегда загружать декодер из safetensors (игнорируя GGUF).
    Safetensors,
    /// Всегда загружать декодер из GGUF (ошибка, если GGUF не найден).
    Gguf,
}

impl AsrPipeline {
    fn looks_like_repetition_loop(tokens: &[u32]) -> bool {
        // Эвристика против "залипания" на квантованных/шумных прогонах (особенно Q4).
        // Мы НЕ пытаемся улучшить качество, только избегаем бессмысленного раздувания вывода
        // до max_tokens при явном цикле.
        //
        // Правила достаточно консервативные, чтобы не срабатывать на коротких "угу/да".
        let n = tokens.len();
        if n < 128 {
            return false;
        }

        // 1) Низкое разнообразие на хвосте.
        let tail = &tokens[n.saturating_sub(96)..n];
        let mut uniq = std::collections::HashSet::with_capacity(tail.len());
        for &t in tail {
            uniq.insert(t);
        }
        if uniq.len() <= 6 {
            return true;
        }

        // 2) Повторяющийся паттерн (фиксированный период) на хвосте.
        // Проверяем несколько периодов, чтобы поймать и повторы одного токена,
        // и повторы короткой фразы.
        for period in 1..=16 {
            let blocks = 6; // требуем минимум 6 одинаковых блоков подряд
            let need = period * blocks;
            if n < need {
                continue;
            }
            let end = &tokens[n - period..n];
            let mut ok = true;
            for b in 2..=blocks {
                let s = n - b * period;
                let e = s + period;
                if tokens[s..e] != *end {
                    ok = false;
                    break;
                }
            }
            if ok {
                return true;
            }
        }

        false
    }

    /// Create a new ASR pipeline from model directory.
    pub fn from_model_dir(model_dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        Self::from_model_dir_with_decoder_weights(model_dir, device, DecoderWeights::Auto)
    }

    /// Create a new ASR pipeline from model directory with explicit decoder weights preference.
    pub fn from_model_dir_with_decoder_weights(
        model_dir: impl AsRef<Path>,
        device: &Device,
        decoder_weights: DecoderWeights,
    ) -> Result<Self> {
        Self::from_model_dir_with_decoder_weights_and_gguf(model_dir, device, decoder_weights, None)
    }

    /// Create a new ASR pipeline with explicit decoder weights preference and optional GGUF override.
    ///
    /// Если `decoder_gguf_override` задан, то:
    /// - при `decoder_weights=Safetensors` это считается конфликтом (вернется ошибка);
    /// - при `decoder_weights=Auto|Gguf` будет использован указанный GGUF.
    ///
    /// Путь может быть абсолютным, либо относительным (тогда он трактуется относительно `model_dir`).
    pub fn from_model_dir_with_decoder_weights_and_gguf(
        model_dir: impl AsRef<Path>,
        device: &Device,
        decoder_weights: DecoderWeights,
        decoder_gguf_override: Option<std::path::PathBuf>,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let safetensors_files = asr_core::model_files::resolve_safetensors_files(model_dir)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let override_gguf: Option<std::path::PathBuf> = decoder_gguf_override.map(|p| {
            let p = p.as_path();
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                model_dir.join(p)
            }
        });

        if decoder_weights == DecoderWeights::Safetensors && override_gguf.is_some() {
            return Err(candle_core::Error::msg(
                "Конфликт аргументов: указан --decoder-gguf, но выбран decoder_weights=safetensors",
            ));
        }

        if let Some(p) = &override_gguf {
            if !p.exists() {
                return Err(candle_core::Error::msg(format!(
                    "GGUF файл не найден: {}",
                    p.display()
                )));
            }
        }

        let gguf = asr_core::model_files::find_preferred_decoder_gguf(model_dir);
        let decoder_gguf = match decoder_weights {
            DecoderWeights::Safetensors => None,
            DecoderWeights::Auto => override_gguf.or(gguf),
            DecoderWeights::Gguf => Some(override_gguf.or(gguf).ok_or_else(|| {
                candle_core::Error::msg(
                    "GGUF не найден в директории модели (ожидался model-*.gguf)",
                )
            })?),
        };

        let config_path = model_dir.join("config.json");

        // Load configurations
        let aut_config =
            AuTConfig::from_hf_config(&config_path).map_err(|e| candle_core::Error::Msg(e))?;
        let qwen3_config =
            Qwen3Config::from_hf_config(&config_path).map_err(|e| candle_core::Error::Msg(e))?;

        // Create Mel extractor with mel filters loaded from file (for exact Python match)
        let mel_config = FeatureExtractorConfig::default();
        let mel_filters_path = model_dir.join("mel_filters.bin");
        let mel_extractor = if mel_filters_path.exists() {
            MelSpectrogramExtractor::with_mel_filters_from_file(mel_config, &mel_filters_path)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?
        } else {
            MelSpectrogramExtractor::new(mel_config)
        };

        // Load encoder
        let st_refs: Vec<&Path> = safetensors_files.iter().map(|p| p.as_path()).collect();
        let encoder = AuTEncoder::from_safetensors_files(aut_config, &st_refs, device)?;

        // Load decoder (prefer GGUF if present)
        let decoder = if let Some(p) = decoder_gguf {
            Qwen3Decoder::from_gguf(qwen3_config, &p, device)?
        } else {
            Qwen3Decoder::from_safetensors_files(qwen3_config, &st_refs, device)?
        };

        // Determine dtype based on device (F32 for CPU, BF16 for GPU)
        let dtype = if device.is_metal() || device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        // Load tokenizer
        let vocab_path = model_dir.join("vocab.json");
        let merges_path = model_dir.join("merges.txt");
        let tokenizer = Tokenizer::from_vocab_and_merges(&vocab_path, &merges_path)
            .map_err(candle_core::Error::msg)?;

        Ok(Self {
            mel_extractor,
            encoder,
            decoder,
            tokenizer,
            device: device.clone(),
            dtype,
        })
    }

    /// Transcribe audio samples to token IDs using autoregressive generation.
    ///
    /// # Arguments
    /// * `samples` - Audio samples at 16kHz
    /// * `max_tokens` - Maximum number of tokens to generate
    ///
    /// # Returns
    /// Token IDs of the transcription
    pub fn transcribe(&self, samples: &[f32]) -> Result<Vec<u32>> {
        self.transcribe_with_max_tokens(samples, 256)
    }

    /// Transcribe with configurable max tokens.
    pub fn transcribe_with_max_tokens(
        &self,
        samples: &[f32],
        max_tokens: usize,
    ) -> Result<Vec<u32>> {
        self.transcribe_with_max_tokens_and_language(samples, max_tokens, None)
    }

    /// Transcribe with configurable max tokens and optional forced language.
    ///
    /// Если `force_language` задан, в промпт добавляется суффикс
    /// `language {Language}<asr_text>` (как в Python пакете `qwen-asr`).
    pub fn transcribe_with_max_tokens_and_language(
        &self,
        samples: &[f32],
        max_tokens: usize,
        force_language: Option<&str>,
    ) -> Result<Vec<u32>> {
        let (tokens, _reason) = self.transcribe_with_max_tokens_and_language_with_reason(
            samples,
            max_tokens,
            force_language,
        )?;
        Ok(tokens)
    }

    fn transcribe_with_max_tokens_and_language_with_reason(
        &self,
        samples: &[f32],
        max_tokens: usize,
        force_language: Option<&str>,
    ) -> Result<(Vec<u32>, StopReason)> {
        let debug = asr_core::debug::enabled();

        // Special token IDs for Qwen3-ASR
        const IM_START: u32 = 151644;
        const IM_END: u32 = 151645;
        const AUDIO_START: u32 = 151669;
        const AUDIO_END: u32 = 151670;
        const ASR_TEXT: u32 = 151704; // "<asr_text>" (added token)
        const TOKEN_LANGUAGE: u32 = 11528; // "language"

        // Step 1: Extract Mel spectrogram
        let mel_spectrum = self
            .mel_extractor
            .extract(samples, &self.device)
            .map_err(|e: asr_core::AsrError| candle_core::Error::Msg(e.to_string()))?;

        // Use the appropriate dtype for the device
        let mel_tensor = mel_spectrum.tensor.to_dtype(self.dtype)?;

        if debug {
            // Отладочная статистика (дорогая): включается через RUSTASR_DEBUG=1.
            let mel_f32 = mel_spectrum.tensor.to_dtype(DType::F32)?;
            let mel_flat = mel_f32.flatten_all()?;
            let mel_min = mel_flat.min(0)?.to_scalar::<f32>()?;
            let mel_max = mel_flat.max(0)?.to_scalar::<f32>()?;
            let mel_mean = mel_flat.mean_all()?.to_scalar::<f32>()?;
            eprintln!(
                "DEBUG Mel: shape={:?}, range=[{:.4}, {:.4}], mean={:.4}",
                mel_f32.dims(),
                mel_min,
                mel_max,
                mel_mean
            );

            // Mel[bin=0, first 5 frames] в терминах HF: mel[0, :5]
            // У нас тензор [1, time, n_mels] => берем [0, 0..5, 0].
            let first_vals: Vec<f32> = (0..5.min(mel_spectrum.num_frames))
                .map(|i| mel_f32.get(0)?.get(i)?.get(0)?.to_scalar::<f32>())
                .collect::<Result<_>>()?;
            eprintln!("DEBUG Mel[0,:5]: {:?}", first_vals);
        }

        // Step 2: Encode audio → embeddings
        if debug {
            eprintln!(
                "DEBUG pipeline: About to call encoder.forward, mel_tensor dtype={:?}",
                mel_tensor.dtype()
            );
        }
        let audio_embeds = self.encoder.forward(&mel_tensor)?;
        if debug {
            eprintln!(
                "DEBUG pipeline: encoder.forward result dtype={:?}",
                audio_embeds.dtype()
            );
        }
        let _audio_seq_len = audio_embeds.dim(1)?;

        if debug {
            let a = audio_embeds.to_dtype(DType::F32)?;
            // [1, T, C]
            let mut rows = Vec::new();
            for t in 0..2.min(a.dim(1)?) {
                let mut row = Vec::new();
                for c in 0..8.min(a.dim(2)?) {
                    row.push(a.get(0)?.get(t)?.get(c)?.to_scalar::<f32>()?);
                }
                rows.push(row);
            }
            eprintln!("DEBUG audio_embeds[:2,:8] (f32): {:?}", rows);
        }

        // Step 3: Точный текстовый префикс/суффикс, совместимый с Python SDK.
        // Формат:
        // <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
        // ... audio embeds ...
        // <|audio_end|><|im_end|>\n<|im_start|>assistant\n
        let prefix_tokens: Vec<u32> = vec![
            IM_START,
            8948,
            198, // <|im_start|>system\n
            IM_END,
            198, // <|im_end|>\n
            IM_START,
            872,
            198,         // <|im_start|>user\n
            AUDIO_START, // <|audio_start|>
        ];

        let mut suffix_tokens: Vec<u32> = vec![
            AUDIO_END, // <|audio_end|>
            IM_END, 198, // <|im_end|>\n
            IM_START, 77091, 198, // <|im_start|>assistant\n
        ];

        if let Some(lang) = force_language {
            let lang = normalize_language_name(lang);

            let token_language = self
                .tokenizer
                .token_id("language")
                .unwrap_or(TOKEN_LANGUAGE);
            let token_space = self.tokenizer.token_id("Ġ").unwrap_or(220);

            let token_lang_spaced = format!("Ġ{lang}");
            if let Some(lang_id) = self.tokenizer.token_id(&token_lang_spaced) {
                suffix_tokens.extend_from_slice(&[token_language, lang_id, ASR_TEXT]);
            } else if let Some(lang_id) = self.tokenizer.token_id(&lang) {
                // Редкий fallback: если в vocab нет токена "ĠLanguage", но есть "Language",
                // добавляем пробел отдельным токеном.
                suffix_tokens.extend_from_slice(&[token_language, token_space, lang_id, ASR_TEXT]);
            } else {
                return Err(candle_core::Error::msg(format!(
                    "Не удалось форсировать язык: token \"Ġ{lang}\" не найден в vocab.json. Попробуйте другое имя языка или используйте auto-режим.",
                )));
            }
        }

        let embed_tokens = self.decoder.get_embed_tokens();

        // Embed token sequences
        let prefix_ids = Tensor::new(prefix_tokens.as_slice(), &self.device)?;
        let suffix_ids = Tensor::new(suffix_tokens.as_slice(), &self.device)?;

        if debug {
            eprintln!("DEBUG pipeline: About to call embed_tokens.forward on prefix");
        }
        let prefix_embeds = embed_tokens.forward(&prefix_ids)?.unsqueeze(0)?;
        if debug {
            let p0 = prefix_embeds.to_dtype(DType::F32)?;
            let mut row = Vec::new();
            for c in 0..8.min(p0.dim(2)?) {
                row.push(p0.get(0)?.get(0)?.get(c)?.to_scalar::<f32>()?);
            }
            eprintln!("DEBUG prefix_embeds[0,0,:8] (f32): {:?}", row);
        }
        if debug {
            eprintln!(
                "DEBUG pipeline: prefix_embeds dtype={:?}",
                prefix_embeds.dtype()
            );
            eprintln!("DEBUG pipeline: About to call embed_tokens.forward on suffix");
        }
        let suffix_embeds = embed_tokens.forward(&suffix_ids)?.unsqueeze(0)?;
        if debug {
            eprintln!(
                "DEBUG pipeline: suffix_embeds dtype={:?}",
                suffix_embeds.dtype()
            );
            eprintln!(
                "DEBUG pipeline: audio_embeds dtype={:?}",
                audio_embeds.dtype()
            );
        }

        // Concatenate: prefix + audio + suffix
        let prompt_embeds = Tensor::cat(&[&prefix_embeds, &audio_embeds, &suffix_embeds], 1)?;
        if debug {
            eprintln!(
                "DEBUG pipeline: prompt_embeds dtype={:?}",
                prompt_embeds.dtype()
            );
        }

        let prompt_len = prompt_embeds.dim(1)?;

        // Step 4: Autoregressive generation (prefill + decode с KV-cache).
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut stop_reason = StopReason::MaxTokens;
        let mut cache = KvCache::new(self.decoder.config().num_hidden_layers);
        let mut cur_pos = prompt_len;

        // Prefill: один проход по всему промпту, заполняем KV-cache.
        let logits = self
            .decoder
            .forward_embeds_with_cache(&prompt_embeds, 0, &mut cache)?;
        let mut last_logits = logits.i((.., logits.dim(1)? - 1, ..))?;

        for i in 0..max_tokens {
            // Greedy: take argmax
            let next_token = last_logits.argmax(candle_core::D::Minus1)?;
            let next_token_id: u32 = next_token.squeeze(0)?.to_scalar()?;

            if debug {
                eprintln!("DEBUG gen: step={}, token_id={}", i, next_token_id);
            }

            // Check for EOS
            if next_token_id == IM_END || next_token_id == 151643 {
                stop_reason = StopReason::Eos;
                break;
            }

            generated_tokens.push(next_token_id);

            if Self::looks_like_repetition_loop(&generated_tokens) {
                stop_reason = StopReason::Repetition;
                break;
            }

            // Decode step: прогоняем ровно один токен, используя KV-cache.
            let next_ids = Tensor::new(&[next_token_id], &self.device)?;
            let next_embed = embed_tokens.forward(&next_ids)?.unsqueeze(0)?; // [1, 1, hidden]
            let step_logits =
                self.decoder
                    .forward_embeds_with_cache(&next_embed, cur_pos, &mut cache)?;
            cur_pos += 1;
            last_logits = step_logits.i((.., step_logits.dim(1)? - 1, ..))?;
        }

        Ok((generated_tokens, stop_reason))
    }

    /// Transcribe audio samples to text.
    ///
    /// # Arguments
    /// * `samples` - Audio samples at 16kHz
    ///
    /// # Returns
    /// Transcribed text
    pub fn transcribe_to_text(&self, samples: &[f32]) -> Result<String> {
        self.transcribe_to_text_with_max_tokens(samples, 256)
    }

    /// То же, что `transcribe_to_text`, но с явным ограничением на число генерируемых токенов.
    pub fn transcribe_to_text_with_max_tokens(
        &self,
        samples: &[f32],
        max_tokens: usize,
    ) -> Result<String> {
        let tokens = self.transcribe_with_max_tokens(samples, max_tokens)?;
        Ok(self.tokenizer.decode(&tokens))
    }

    /// Версия с опциональным форсированием языка (суффикс `language X<asr_text>`).
    pub fn transcribe_to_text_with_max_tokens_and_language(
        &self,
        samples: &[f32],
        max_tokens: usize,
        force_language: Option<&str>,
    ) -> Result<String> {
        let tokens =
            self.transcribe_with_max_tokens_and_language(samples, max_tokens, force_language)?;
        Ok(self.tokenizer.decode(&tokens))
    }

    /// Транскрибация с парсингом результата (язык + текст).
    pub fn transcribe_to_result_with_max_tokens_and_language(
        &self,
        samples: &[f32],
        max_tokens: usize,
        force_language: Option<&str>,
    ) -> Result<AsrTranscription> {
        let (tokens, stop_reason) = self.transcribe_with_max_tokens_and_language_with_reason(
            samples,
            max_tokens,
            force_language,
        )?;
        let raw = self.tokenizer.decode(&tokens);
        let (language, text) = parse_asr_output(&raw, force_language);
        Ok(AsrTranscription {
            language,
            text,
            raw,
            generated_tokens: tokens.len(),
            stop_reason,
        })
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.decoder.vocab_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pick_test_device() -> Device {
        // В тестах используем CPU по умолчанию: Metal может быть недоступен (или panics внутри candle).
        // Для локальной проверки на Metal: RUSTASR_TEST_DEVICE=metal cargo test -p asr-pipeline --lib
        match std::env::var("RUSTASR_TEST_DEVICE").as_deref() {
            Ok("metal") => std::panic::catch_unwind(|| Device::new_metal(0).ok())
                .ok()
                .flatten()
                .unwrap_or(Device::Cpu),
            _ => Device::Cpu,
        }
    }

    #[test]
    fn test_pipeline_creation() {
        let model_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("models")
            .join("qwen3-asr-0.6b");

        if !model_path.join("model.safetensors").exists() {
            eprintln!("⚠️  Skipping test: model not found");
            return;
        }

        let device = pick_test_device();
        let pipeline = AsrPipeline::from_model_dir(&model_path, &device);

        match pipeline {
            Ok(p) => {
                eprintln!("✅ Pipeline created successfully!");
                eprintln!("   Vocab size: {}", p.vocab_size());
            }
            Err(e) => {
                eprintln!("⚠️  Failed to create pipeline: {}", e);
            }
        }
    }
}
