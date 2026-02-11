//! CTC-голова и greedy-декодирование.
//!
//! CTCHead: Conv1d(d_model → num_classes, kernel=1) → log_softmax
//! CTC Greedy: argmax → удаление blanks и дублей → декодирование через словарь.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};
use tracing::debug;

/// CTC-голова: проекция на словарь + log_softmax.
pub struct CtcHead {
    /// Conv1d(feat_in, num_classes, kernel=1)
    conv: Conv1d,
    /// Количество классов (включая blank).
    num_classes: usize,
}

impl CtcHead {
    pub fn load(feat_in: usize, num_classes: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        let conv = candle_nn::conv1d(feat_in, num_classes, 1, cfg, vb.pp("decoder_layers.0"))?;
        Ok(Self { conv, num_classes })
    }

    /// Прямой проход: (batch, d_model, seq) → log_softmax (batch, seq, num_classes).
    pub fn forward(&self, encoder_output: &Tensor) -> Result<Tensor> {
        // Conv1d: (batch, d_model, seq) → (batch, num_classes, seq)
        let logits = self.conv.forward(encoder_output)?;

        // Транспонируем: (batch, num_classes, seq) → (batch, seq, num_classes)
        let logits = logits.transpose(1, 2)?;

        // log_softmax по последнему измерению
        candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)
    }

    /// ID blank-токена (последний класс).
    pub fn blank_id(&self) -> usize {
        self.num_classes - 1
    }
}

/// Простой CTC greedy-декодер.
pub struct CtcGreedyDecoder {
    /// Словарь: id → строка (SentencePiece piece).
    vocab: HashMap<usize, String>,
    /// ID blank-токена.
    blank_id: usize,
}

impl CtcGreedyDecoder {
    /// Создать декодер из файла vocab.json.
    pub fn from_vocab_file(path: impl AsRef<Path>, blank_id: usize) -> asr_core::AsrResult<Self> {
        let path = path.as_ref();
        let data = std::fs::read_to_string(path).map_err(|e| {
            asr_core::AsrError::Model(format!(
                "Не удалось прочитать vocab.json из {:?}: {e}",
                path
            ))
        })?;

        let raw: HashMap<String, String> = serde_json::from_str(&data).map_err(|e| {
            asr_core::AsrError::Model(format!("Ошибка парсинга vocab.json: {e}"))
        })?;

        let vocab: HashMap<usize, String> = raw
            .into_iter()
            .filter_map(|(k, v)| k.parse::<usize>().ok().map(|id| (id, v)))
            .collect();

        debug!("CTC словарь загружен: {} токенов, blank_id={}", vocab.len(), blank_id);

        Ok(Self { vocab, blank_id })
    }

    /// Декодировать log_probs в текст (greedy).
    ///
    /// `log_probs` — (seq_len, num_classes) - log-вероятности.
    pub fn decode(&self, log_probs: &Tensor) -> Result<String> {
        // argmax по последнему измерению → (seq_len,)
        let predictions = log_probs.argmax(candle_core::D::Minus1)?;
        let token_ids: Vec<u32> = predictions.to_vec1()?;

        // CTC: убрать blanks и последовательные дубликаты
        let mut decoded_ids = Vec::new();
        let mut prev_token = self.blank_id as u32;

        for &tok in &token_ids {
            if tok != self.blank_id as u32 && (tok != prev_token || prev_token == self.blank_id as u32) {
                decoded_ids.push(tok as usize);
            }
            prev_token = tok;
        }

        // Преобразовать ID в строку через SentencePiece словарь
        let text = self.decode_ids(&decoded_ids);
        Ok(text)
    }

    /// Преобразовать список ID в текст.
    fn decode_ids(&self, ids: &[usize]) -> String {
        let mut pieces: Vec<&str> = Vec::with_capacity(ids.len());
        for &id in ids {
            if let Some(piece) = self.vocab.get(&id) {
                pieces.push(piece);
            }
        }

        // SentencePiece использует ▁ вместо пробелов
        let text = pieces.join("");
        let text = text.replace('▁', " ");

        // Убрать начальный пробел
        text.trim_start().to_string()
    }

    /// Размер словаря.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}
