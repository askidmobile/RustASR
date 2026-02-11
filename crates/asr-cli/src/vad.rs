//! VAD (voice activity detection) и сегментация аудио для покадровой транскрибации.

use std::path::PathBuf;

use anyhow::{Context, Result};
use webrtc_vad::{SampleRate, Vad, VadMode};

/// Настройки сегментации по VAD.
#[derive(Debug, Clone)]
pub struct VadSegmentationConfig {
    /// Агрессивность VAD (0..3).
    pub mode: u8,
    /// Длина фрейма в миллисекундах (10/20/30).
    pub frame_ms: usize,
    /// Минимальная длительность речи, чтобы открыть сегмент.
    pub min_speech_ms: usize,
    /// Минимальная длительность тишины, чтобы закрыть сегмент.
    pub min_silence_ms: usize,
    /// Паддинг (добавка) к началу и концу сегмента.
    pub pad_ms: usize,
    /// Максимальная длительность сегмента. Длинные сегменты режем на куски.
    pub max_segment_ms: usize,
}

impl Default for VadSegmentationConfig {
    fn default() -> Self {
        Self {
            mode: 2,
            frame_ms: 30,
            min_speech_ms: 300,
            min_silence_ms: 200,
            pad_ms: 150,
            max_segment_ms: 30_000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpeechSegment {
    pub start_sample: usize,
    pub end_sample: usize,
}

impl SpeechSegment {
    // (пока пусто) оставляем struct простым контейнером start/end.
}

fn round_up_div(a: usize, b: usize) -> usize {
    if b == 0 {
        return 0;
    }
    (a + b - 1) / b
}

fn sample_rate_to_webrtc(sample_rate: usize) -> Result<SampleRate> {
    let sr: i32 = sample_rate
        .try_into()
        .context("sample_rate out of i32 range")?;
    SampleRate::try_from(sr).map_err(|e| anyhow::anyhow!(e))
}

fn vad_mode_from_u8(v: u8) -> Result<VadMode> {
    Ok(match v {
        0 => VadMode::Quality,
        1 => VadMode::LowBitrate,
        2 => VadMode::Aggressive,
        3 => VadMode::VeryAggressive,
        _ => anyhow::bail!("Неподдерживаемый vad-mode={v} (ожидается 0..3)"),
    })
}

/// Сегментировать моно-сигнал (16-bit PCM по сути) на отрезки речи.
///
/// `samples` должны быть в диапазоне [-1; 1], sample_rate: 8000/16000/32000/48000.
pub fn split_mono_by_vad(
    samples: &[f32],
    sample_rate: usize,
    cfg: &VadSegmentationConfig,
) -> Result<Vec<SpeechSegment>> {
    if !matches!(cfg.frame_ms, 10 | 20 | 30) {
        anyhow::bail!(
            "Неподдерживаемый frame_ms={} (ожидается 10/20/30)",
            cfg.frame_ms
        );
    }

    let sr = sample_rate_to_webrtc(sample_rate)?;
    let mut vad = Vad::new_with_rate_and_mode(sr, vad_mode_from_u8(cfg.mode)?);

    let frame_len = sample_rate * cfg.frame_ms / 1000;
    if frame_len == 0 {
        anyhow::bail!(
            "Некорректный frame_len=0 (sample_rate={sample_rate}, frame_ms={})",
            cfg.frame_ms
        );
    }

    // Приводим к i16 один раз, чтобы не делать конвертацию на каждом фрейме.
    let pcm: Vec<i16> = samples
        .iter()
        .map(|&v| {
            let v = v.clamp(-1.0, 1.0);
            (v * i16::MAX as f32) as i16
        })
        .collect();

    let n_frames = pcm.len() / frame_len;
    if n_frames == 0 {
        return Ok(Vec::new());
    }

    let min_speech_frames = round_up_div(cfg.min_speech_ms, cfg.frame_ms).max(1);
    let min_silence_frames = round_up_div(cfg.min_silence_ms, cfg.frame_ms).max(1);
    let pad_frames = round_up_div(cfg.pad_ms, cfg.frame_ms);

    // 1) Получаем бинарную разметку voice/non-voice по фреймам.
    let mut voice: Vec<bool> = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let start = i * frame_len;
        let end = start + frame_len;
        let is_voice = vad.is_voice_segment(&pcm[start..end]).unwrap_or(false);
        voice.push(is_voice);
    }

    // 2) Конвертируем voice-фреймы в сырые сегменты (по фрейм-индексам) с гистерезисом.
    let mut raw: Vec<(usize, usize)> = Vec::new();
    let mut in_speech = false;
    let mut speech_start_frame = 0usize;
    let mut speech_streak = 0usize;
    let mut silence_streak = 0usize;

    for (i, &v) in voice.iter().enumerate() {
        if v {
            silence_streak = 0;
            if !in_speech {
                speech_streak += 1;
                if speech_streak >= min_speech_frames {
                    in_speech = true;
                    speech_start_frame = i + 1 - speech_streak;
                }
            }
        } else {
            speech_streak = 0;
            if in_speech {
                silence_streak += 1;
                if silence_streak >= min_silence_frames {
                    let end_frame = i + 1 - silence_streak;
                    raw.push((speech_start_frame, end_frame));
                    in_speech = false;
                    silence_streak = 0;
                }
            }
        }
    }
    if in_speech {
        raw.push((speech_start_frame, voice.len()));
    }

    if raw.is_empty() {
        return Ok(Vec::new());
    }

    // 3) Переводим во временные отрезки (по sample-индексам), добавляем padding, мерджим пересечения.
    let total_samples = pcm.len();
    let mut padded: Vec<(usize, usize)> = Vec::with_capacity(raw.len());
    for (sf, ef) in raw {
        let start_frame = sf.saturating_sub(pad_frames);
        let end_frame = (ef + pad_frames).min(voice.len());
        let start = start_frame * frame_len;
        let end = (end_frame * frame_len).min(total_samples);
        if end > start {
            padded.push((start, end));
        }
    }

    padded.sort_by_key(|(s, _)| *s);
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (s, e) in padded {
        if let Some(last) = merged.last_mut() {
            if s <= last.1 {
                last.1 = last.1.max(e);
                continue;
            }
        }
        merged.push((s, e));
    }

    // 4) Режем слишком длинные сегменты.
    let max_seg_samples = cfg.max_segment_ms * sample_rate / 1000;
    let mut out: Vec<SpeechSegment> = Vec::new();
    for (s, e) in merged {
        if max_seg_samples == 0 || e - s <= max_seg_samples {
            out.push(SpeechSegment {
                start_sample: s,
                end_sample: e,
            });
            continue;
        }
        let mut cur = s;
        while cur < e {
            let next = (cur + max_seg_samples).min(e);
            out.push(SpeechSegment {
                start_sample: cur,
                end_sample: next,
            });
            cur = next;
        }
    }

    Ok(out)
}

pub fn format_hhmmss_millis(t: f32) -> String {
    let total_ms = (t * 1000.0).round().max(0.0) as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

pub fn default_out_dir(prefix: &str) -> Result<PathBuf> {
    use std::time::{SystemTime, UNIX_EPOCH};

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("SystemTime before UNIX_EPOCH")?
        .as_secs();
    Ok(PathBuf::from(format!("{prefix}-{ts}")))
}
