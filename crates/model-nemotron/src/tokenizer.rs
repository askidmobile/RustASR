//! SentencePiece-детокенизатор из vocab.json (как Parakeet).
//! vocab.json: { "piece": {"id": N}, ... }. Детокен: ▁→пробел, trim.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;

pub struct SpVocab {
    vocab: Vec<String>,
}

impl SpVocab {
    pub fn from_json(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let map: HashMap<String, serde_json::Value> = serde_json::from_str(&data)?;
        let mut vocab = vec![String::new(); map.len()];
        for (piece, info) in map {
            if let Some(id) = info.get("id").and_then(|v| v.as_u64()) {
                let idx = id as usize;
                if idx < vocab.len() {
                    vocab[idx] = piece;
                }
            }
        }
        Ok(Self { vocab })
    }

    pub fn len(&self) -> usize {
        self.vocab.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vocab.is_empty()
    }

    /// Детокенизация: ▁ → пробел, конкатенация, trim.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut s = String::new();
        for &t in tokens {
            let i = t as usize;
            if i < self.vocab.len() {
                s.push_str(&self.vocab[i].replace('\u{2581}', " "));
            }
        }
        s.trim().to_string()
    }
}
