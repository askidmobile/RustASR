//! Декодер для Whisper: greedy + temperature fallback.
//!
//! Реализует авторегрессивную генерацию текста из аудио-фичей энкодера.

use candle_core::{IndexOp, Tensor};
use rand::distributions::Distribution;
use rand::SeedableRng;

/// Результат декодирования одного 30-секундного сегмента.
#[derive(Debug, Clone)]
pub struct DecodingResult {
    /// Список сгенерированных токенов.
    pub tokens: Vec<u32>,
    /// Декодированный текст.
    pub text: String,
    /// Средний log-probability.
    pub avg_logprob: f64,
    /// Вероятность отсутствия речи.
    pub no_speech_prob: f64,
    /// Температура, при которой было принято решение.
    pub temperature: f64,
    /// Степень сжатия (для детекции зацикливания).
    pub compression_ratio: f64,
}

/// Задача декодирования.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    /// Транскрибация (STT на исходном языке).
    Transcribe,
    /// Перевод на английский.
    Translate,
}

/// Whisper-декодер: greedy + temperature fallback.
pub struct WhisperDecoder {
    /// Генератор случайных чисел (для sampling с температурой > 0).
    rng: rand::rngs::StdRng,
    /// Токены для подавления (suppress_tokens из конфига).
    suppress_tokens: Tensor,
    /// Специальные токены.
    pub sot_token: u32,
    pub eot_token: u32,
    pub transcribe_token: u32,
    pub translate_token: u32,
    pub no_speech_token: u32,
    pub no_timestamps_token: u32,
    /// Токен языка (например, <|ru|>).
    pub language_token: Option<u32>,
    /// Задача.
    pub task: Task,
    /// Генерировать ли временные метки.
    pub timestamps: bool,
}

impl WhisperDecoder {
    /// Создать декодер с заданными специальными токенами.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sot_token: u32,
        eot_token: u32,
        transcribe_token: u32,
        translate_token: u32,
        no_speech_token: u32,
        no_timestamps_token: u32,
        language_token: Option<u32>,
        task: Task,
        timestamps: bool,
        suppress_tokens: Tensor,
        seed: u64,
    ) -> Self {
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            suppress_tokens,
            sot_token,
            eot_token,
            transcribe_token,
            translate_token,
            no_speech_token,
            no_timestamps_token,
            language_token,
            task,
            timestamps,
        }
    }

    /// Начальные токены промпта для декодирования.
    pub fn prompt_tokens(&self) -> Vec<u32> {
        let mut tokens = vec![self.sot_token];
        if let Some(lang_token) = self.language_token {
            tokens.push(lang_token);
        }
        match self.task {
            Task::Transcribe => tokens.push(self.transcribe_token),
            Task::Translate => tokens.push(self.translate_token),
        }
        if !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }
        tokens
    }

    /// Декодировать один 30-секундный сегмент.
    ///
    /// # Аргументы
    /// * `audio_features` — выход аудио-энкодера, shape [1, T, d_model]
    /// * `sample_len` — максимальное количество генерируемых токенов
    /// * `temperature` — температура сэмплирования (0.0 = greedy)
    /// * `forward_fn` — функция для forward pass через text decoder:
    ///   `(token_ids: &Tensor, audio_features: &Tensor, flush: bool) -> logits: Tensor`
    pub fn decode_segment<F>(
        &mut self,
        audio_features: &Tensor,
        sample_len: usize,
        temperature: f64,
        mut forward_fn: F,
    ) -> candle_core::Result<DecodingResult>
    where
        F: FnMut(&Tensor, &Tensor, bool) -> candle_core::Result<Tensor>,
    {
        let device = audio_features.device();
        let mut tokens = self.prompt_tokens();
        let initial_tokens_len = tokens.len();

        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;

        for i in 0..sample_len {
            let tokens_t =
                Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
            let flush = i == 0;
            let ys = forward_fn(&tokens_t, audio_features, flush)?;

            // Берём логиты последнего токена
            let (_, seq_len, _) = ys.dims3()?;
            let logits = ys.i((0, seq_len - 1))?;

            // На первом шаге проверяем no_speech
            if i == 0 {
                let logits_no_speech = candle_nn::ops::softmax(&logits, 0)?;
                no_speech_prob =
                    logits_no_speech.i(self.no_speech_token as usize)?.to_scalar::<f32>()? as f64;
            }

            // Подавление запрещённых токенов
            let logits = {
                let suppress = &self.suppress_tokens;
                let on_token = Tensor::new(&[f32::NEG_INFINITY], device)?
                    .broadcast_as(suppress.shape())?;
                logits.scatter_add(suppress, &on_token, 0)?
            };

            // Выбор следующего токена
            let next_token = if temperature > 0.0 {
                self.sample_with_temperature(&logits, temperature)?
            } else {
                logits.argmax(0)?.to_scalar::<u32>()?
            };

            // Суммируем log-probability
            let logprobs = candle_nn::ops::log_softmax(&logits, 0)?;
            sum_logprob += logprobs.i(next_token as usize)?.to_scalar::<f32>()? as f64;

            // Проверка EOT
            if next_token == self.eot_token {
                break;
            }

            tokens.push(next_token);
        }

        let generated_tokens: Vec<u32> =
            tokens[initial_tokens_len..].to_vec();
        let n_generated = generated_tokens.len().max(1);
        let avg_logprob = sum_logprob / n_generated as f64;
        let compression_ratio = Self::compression_ratio(&generated_tokens);

        Ok(DecodingResult {
            tokens: generated_tokens,
            text: String::new(), // Заполняется вызывающим кодом через tokenizer
            avg_logprob,
            no_speech_prob,
            temperature,
            compression_ratio,
        })
    }

    /// Декодирование с temperature fallback.
    ///
    /// Пробует температуры [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], пока не получит
    /// результат с приемлемым compression_ratio и avg_logprob.
    pub fn decode_with_fallback<F>(
        &mut self,
        audio_features: &Tensor,
        sample_len: usize,
        mut forward_fn: F,
    ) -> candle_core::Result<DecodingResult>
    where
        F: FnMut(&Tensor, &Tensor, bool) -> candle_core::Result<Tensor>,
    {
        let temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let mut last_result = None;

        for &t in &temperatures {
            let result = self.decode_segment(audio_features, sample_len, t, &mut forward_fn)?;

            // Хороший результат: не слишком зацикленный и не слишком шумный
            if result.compression_ratio < 2.4 && result.avg_logprob > -1.0 {
                return Ok(result);
            }

            last_result = Some(result);
        }

        // Если все температуры провалились, возвращаем последний результат
        Ok(last_result.unwrap())
    }

    /// Сэмплирование с температурой.
    fn sample_with_temperature(
        &mut self,
        logits: &Tensor,
        temperature: f64,
    ) -> candle_core::Result<u32> {
        let logits = (logits / temperature)?;
        let probs = candle_nn::ops::softmax(&logits, 0)?;
        let probs_vec: Vec<f32> = probs.to_vec1()?;

        let distr =
            rand::distributions::WeightedIndex::new(&probs_vec).map_err(candle_core::Error::wrap)?;
        let next_token = distr.sample(&mut self.rng) as u32;
        Ok(next_token)
    }

    /// Вычислить compression ratio для детекции зацикливания.
    fn compression_ratio(tokens: &[u32]) -> f64 {
        if tokens.is_empty() {
            return 0.0;
        }
        let text_bytes: Vec<u8> = tokens
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect();
        let _original_len = text_bytes.len();

        // Простая эвристика: считаем уникальные n-граммы
        let mut unique_bigrams = std::collections::HashSet::new();
        for window in tokens.windows(2) {
            unique_bigrams.insert((window[0], window[1]));
        }
        let unique_ratio = if tokens.len() > 1 {
            unique_bigrams.len() as f64 / (tokens.len() - 1) as f64
        } else {
            1.0
        };

        // Инвертируем: больше уникальность = меньше compression ratio
        1.0 / unique_ratio.max(0.01)
    }
}
