//! Утилиты для работы с файлами модели (safetensors/GGUF) на диске.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use crate::{AsrError, AsrResult};

/// Имена GGUF-файлов, которые пайплайн пробует в порядке убывания предпочтительности.
///
/// Важно: это относится только к весам декодера (LLM), аудио-энкодер пока грузится из safetensors.
pub const GGUF_DECODER_CANDIDATES: &[&str] = &[
    "model-q8_0.gguf",
    "model-q6k.gguf",
    "model-q6_k.gguf",
    "model-q4_0.gguf",
    "model.gguf",
];

/// Найти предпочтительный GGUF-файл для декодера в директории модели.
pub fn find_preferred_decoder_gguf(model_dir: impl AsRef<Path>) -> Option<PathBuf> {
    let model_dir = model_dir.as_ref();
    for n in GGUF_DECODER_CANDIDATES {
        let p = model_dir.join(n);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

#[derive(serde::Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Разрешить список safetensors-файлов в директории модели.
///
/// Поддерживает:
/// - `model.safetensors` (один файл)
/// - `model.safetensors.index.json` + шардированные `model-00001-of-0000N.safetensors`
/// - fallback: поиск `model-*-of-*.safetensors` без index.json
pub fn resolve_safetensors_files(model_dir: impl AsRef<Path>) -> AsrResult<Vec<PathBuf>> {
    let model_dir = model_dir.as_ref();

    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let data = std::fs::read(&index_path)?;
        let idx: SafetensorsIndex = serde_json::from_slice(&data)?;

        // weight_map: tensor_name -> shard_filename
        // Порядок не важен для VarBuilder, но для детерминизма сортируем.
        let mut uniq: BTreeSet<String> = BTreeSet::new();
        for shard in idx.weight_map.values() {
            uniq.insert(shard.clone());
        }

        let mut out = Vec::with_capacity(uniq.len());
        for shard in uniq {
            let p = model_dir.join(&shard);
            if !p.exists() {
                return Err(AsrError::Model(format!(
                    "В index.json указан шард, но файл не найден: {}",
                    p.display()
                )));
            }
            out.push(p);
        }

        if out.is_empty() {
            return Err(AsrError::Model(format!(
                "Пустой weight_map в {}",
                index_path.display()
            )));
        }

        return Ok(out);
    }

    // Fallback: если index.json отсутствует, но шардированные файлы лежат рядом.
    let mut shards: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(model_dir)? {
        let entry = entry?;
        let p = entry.path();
        if !p.is_file() {
            continue;
        }
        let Some(name) = p.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if name.starts_with("model-") && name.ends_with(".safetensors") && name.contains("-of-") {
            shards.push(p);
        }
    }
    shards.sort();
    if !shards.is_empty() {
        return Ok(shards);
    }

    Err(AsrError::Model(format!(
        "В директории модели не найден ни model.safetensors, ни model.safetensors.index.json, ни шардов model-*-of-*.safetensors: {}",
        model_dir.display()
    )))
}
