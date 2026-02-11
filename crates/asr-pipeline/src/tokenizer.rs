//! Декодер токенов Qwen3-ASR.
//!
//! В Qwen3-ASR базовый словарь хранится в `vocab.json` (id 0..151642),
//! а спец-токены (включая `<|im_start|>`, `<|audio_start|>`, `<asr_text>`, ...)
//! описаны в `tokenizer_config.json` в `added_tokens_decoder` и занимают id >= 151643.
//!
//! Словарь использует GPT2-style byte-level BPE: строковые представления токенов
//! содержат псевдосимволы (например, `Ġ`), которые нужно декодировать через
//! обратное отображение byte->unicode (как в оригинальном GPT-2).

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde::Deserialize;

#[derive(Debug)]
pub struct Tokenizer {
    id_to_token: Vec<Option<String>>,
    token_to_id: HashMap<String, u32>,
    special_ids: HashSet<u32>,
    eos_token_ids: HashSet<u32>,
    byte_decoder: HashMap<char, u8>,
}

#[derive(Debug, Deserialize)]
struct AddedTokenEntry {
    content: String,
    special: bool,
}

impl Tokenizer {
    /// Загружает `vocab.json` + `tokenizer_config.json`.
    ///
    /// `merges.txt` оставлен в сигнатуре для совместимости, но для декодирования
    /// в этой реализации не используется.
    pub fn from_vocab_and_merges(
        vocab_path: impl AsRef<Path>,
        _merges_path: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let vocab_path = vocab_path.as_ref();
        let config_path = vocab_path
            .parent()
            .ok_or("Не удалось определить директорию модели")?
            .join("tokenizer_config.json");

        // vocab.json: token -> id
        let vocab_str = std::fs::read_to_string(vocab_path)
            .map_err(|e| format!("Не удалось прочитать vocab.json: {e}"))?;
        let vocab: HashMap<String, u32> = serde_json::from_str(&vocab_str)
            .map_err(|e| format!("Не удалось распарсить vocab.json: {e}"))?;

        let max_vocab_id = vocab.values().copied().max().unwrap_or(0);

        // tokenizer_config.json: added_tokens_decoder
        let cfg_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Не удалось прочитать tokenizer_config.json: {e}"))?;
        let cfg_val: serde_json::Value = serde_json::from_str(&cfg_str)
            .map_err(|e| format!("Не удалось распарсить tokenizer_config.json: {e}"))?;

        let mut added: HashMap<u32, AddedTokenEntry> = HashMap::new();
        if let Some(obj) = cfg_val
            .get("added_tokens_decoder")
            .and_then(|v| v.as_object())
        {
            for (k, v) in obj {
                let id: u32 = k
                    .parse()
                    .map_err(|_| format!("Некорректный id в added_tokens_decoder: {k}"))?;
                let entry: AddedTokenEntry = serde_json::from_value(v.clone())
                    .map_err(|e| format!("Некорректная запись added_tokens_decoder[{k}]: {e}"))?;
                added.insert(id, entry);
            }
        }

        let max_added_id = added.keys().copied().max().unwrap_or(0);
        let max_id = max_vocab_id.max(max_added_id);
        let mut id_to_token: Vec<Option<String>> = vec![None; (max_id as usize) + 1];
        let mut token_to_id: HashMap<String, u32> = HashMap::with_capacity(id_to_token.len());

        for (tok, id) in vocab {
            id_to_token[id as usize] = Some(tok);
            // Если вдруг встречаются дубликаты, сохраняем первое значение.
            // Для форсирования языка это безопаснее, чем "переопределить".
            token_to_id
                .entry(id_to_token[id as usize].as_ref().unwrap().clone())
                .or_insert(id);
        }

        let mut special_ids = HashSet::new();
        for (id, entry) in &added {
            if (*id as usize) >= id_to_token.len() {
                continue;
            }
            id_to_token[*id as usize] = Some(entry.content.clone());
            token_to_id.entry(entry.content.clone()).or_insert(*id);
            if entry.special {
                special_ids.insert(*id);
            }
        }

        // generation_config.json: eos_token_id=[151643,151645]
        let eos_token_ids: HashSet<u32> = [151643_u32, 151645_u32].into_iter().collect();

        Ok(Self {
            id_to_token,
            token_to_id,
            special_ids,
            eos_token_ids,
            byte_decoder: gpt2_byte_decoder(),
        })
    }

    /// Декодирует id токенов в строку (HF-поведение: skip_special_tokens=true).
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut raw = String::new();
        for &id in token_ids {
            if self.eos_token_ids.contains(&id) {
                break;
            }
            if self.special_ids.contains(&id) {
                continue;
            }
            let tok = self.id_to_token.get(id as usize).and_then(|x| x.as_ref());
            if let Some(t) = tok {
                raw.push_str(t);
            }
        }
        byte_level_decode(&raw, &self.byte_decoder)
            .trim()
            .to_string()
    }

    /// Вернуть id токена по строковому представлению из `vocab.json`.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }
}

fn byte_level_decode(s: &str, byte_decoder: &HashMap<char, u8>) -> String {
    let mut bytes: Vec<u8> = Vec::with_capacity(s.len());
    for ch in s.chars() {
        if let Some(&b) = byte_decoder.get(&ch) {
            bytes.push(b);
        } else {
            let mut buf = [0u8; 4];
            let enc = ch.encode_utf8(&mut buf);
            bytes.extend_from_slice(enc.as_bytes());
        }
    }
    String::from_utf8_lossy(&bytes).to_string()
}

fn gpt2_byte_decoder() -> HashMap<char, u8> {
    // Портировано из GPT-2 encoder.py (byte_to_unicode).
    let mut bs: Vec<u32> = Vec::new();
    bs.extend((b'!' as u32)..=(b'~' as u32));
    bs.extend((0x00A1_u32)..=(0x00AC_u32));
    bs.extend((0x00AE_u32)..=(0x00FF_u32));

    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0u32..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    let mut map = HashMap::with_capacity(256);
    for (&b, &c) in bs.iter().zip(cs.iter()) {
        let ch = char::from_u32(c).unwrap();
        map.insert(ch, b as u8);
    }
    map
}
