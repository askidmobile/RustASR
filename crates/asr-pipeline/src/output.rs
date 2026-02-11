//! Парсинг вывода Qwen3-ASR.
//!
//! Модель обычно генерирует строку вида:
//! `language Russian<asr_text>...текст...`
//!
//! Эта логика соответствует `qwen_asr.inference.utils.parse_asr_output`.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsrTranscription {
    /// Определенный (или форсированный) язык, например "Russian".
    /// Пустая строка, если язык неизвестен или аудио пустое.
    pub language: String,
    /// Распознанный текст.
    pub text: String,
    /// Исходная строка модели после декодирования.
    pub raw: String,

    /// Сколько новых токенов было сгенерировано.
    pub generated_tokens: usize,

    /// Причина остановки генерации.
    pub stop_reason: StopReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Сработал EOS (`<|endoftext|>` или `<|im_end|>`).
    Eos,
    /// Достигнут лимит `max_tokens`.
    MaxTokens,
    /// Обнаружено зацикливание по повторяющимся токенам/фразам, остановлено эвристикой.
    Repetition,
}

pub fn parse_asr_output(raw: &str, forced_language: Option<&str>) -> (String, String) {
    let s = raw.trim();
    if s.is_empty() {
        return (String::new(), String::new());
    }

    if let Some(lang) = forced_language {
        return (lang.trim().to_string(), s.to_string());
    }

    const TAG: &str = "<asr_text>";
    if let Some((meta, text)) = s.split_once(TAG) {
        let meta_trim = meta.trim();
        let text_trim = text.trim();

        // empty-audio heuristic
        if meta_trim.to_ascii_lowercase().contains("language none") {
            if text_trim.is_empty() {
                return (String::new(), String::new());
            }
            return (String::new(), text_trim.to_string());
        }

        // extract language
        let mut lang_out = String::new();
        for line in meta_trim.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let lower = line.to_ascii_lowercase();
            if let Some(rest) = lower.strip_prefix("language ") {
                // Берем язык из оригинальной строки (чтобы сохранить регистр), но
                // позицию вычисляем через длину префикса.
                let orig_rest = &line["language ".len()..];
                let val = orig_rest.trim();
                if !val.is_empty() {
                    // Нормализация как в python: первая буква uppercase.
                    let mut chars = val.chars();
                    if let Some(first) = chars.next() {
                        lang_out.push_str(&first.to_uppercase().to_string());
                        lang_out.push_str(&chars.as_str().to_lowercase());
                    } else {
                        lang_out.push_str(val);
                    }
                }
                let _ = rest;
                break;
            }
        }

        return (lang_out, text_trim.to_string());
    }

    // no tag => treat as pure text
    (String::new(), s.to_string())
}
