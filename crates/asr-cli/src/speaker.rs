//! Простая диаризация по акустическим признакам (кластеризация говорящих).
//!
//! Важно: это базовый (и намеренно простой) вариант, который не требует отдельной
//! модели speaker-embedding. Для более точной диаризации в будущем можно заменить
//! embedding на специализированную модель (ECAPA/x-vector и т.п.).

use std::path::Path;

use anyhow::{Context, Result};
use asr_core::FeatureExtractorConfig;
use audio::MelSpectrogramExtractor;
use candle_core::Device;

/// Кластеризовать сегменты речи на N говорящих по среднему log-mel вектору.
///
/// Возвращает `speaker_id` (0..num_speakers-1) для каждого сегмента в исходном порядке.
pub fn cluster_segments_mel_kmeans(
    model_dir: &Path,
    sample_rate: usize,
    samples: &[f32],
    segments: &[(usize, usize)],
    num_speakers: usize,
) -> Result<Vec<usize>> {
    if num_speakers == 0 {
        anyhow::bail!("num_speakers=0 недопустим");
    }
    if segments.is_empty() {
        return Ok(Vec::new());
    }
    if num_speakers == 1 || segments.len() == 1 {
        return Ok(vec![0; segments.len()]);
    }

    let extractor = build_mel_extractor(model_dir, sample_rate)?;
    let device = Device::Cpu;

    let mut embs: Vec<Vec<f32>> = Vec::with_capacity(segments.len());
    for &(start, end) in segments {
        let s = start.min(samples.len());
        let e = end.min(samples.len());
        if e <= s {
            // Для корректного VAD это не должно происходить, но держим безопасный fallback.
            embs.push(vec![0.0; extractor_n_mels(&extractor)]);
            continue;
        }

        let seg = &samples[s..e];
        let mel = extractor
            .extract(seg, &device)
            .context("mel extraction failed for diarization segment")?;
        let flat: Vec<f32> = mel
            .tensor
            .flatten_all()
            .context("mel flatten failed")?
            .to_vec1()
            .context("mel to_vec failed")?;

        let t = mel.num_frames.max(1);
        let m = mel.num_mels;
        let mut emb = vec![0f32; m];
        for i in 0..t {
            let base = i * m;
            for j in 0..m {
                emb[j] += flat[base + j];
            }
        }
        let inv = 1.0 / (t as f32);
        for v in &mut emb {
            *v *= inv;
        }
        l2_normalize(&mut emb);
        embs.push(emb);
    }

    let k = num_speakers.min(embs.len());
    let mut assign = kmeans_cosine(&embs, k, 25);
    remap_clusters_by_first_segment(segments, &mut assign, k);
    Ok(assign)
}

fn extractor_n_mels(extractor: &MelSpectrogramExtractor) -> usize {
    // В текущей реализации MelSpectrogramExtractor не экспонирует config,
    // поэтому используем дефолтный размерность как fallback.
    // Если сегмент пустой (e<=s) мы всё равно не используем embedding по смыслу.
    //
    // Дефолт по FeatureExtractorConfig: 128 mel bins.
    let _ = extractor;
    128
}

fn build_mel_extractor(model_dir: &Path, sample_rate: usize) -> Result<MelSpectrogramExtractor> {
    let mut cfg = FeatureExtractorConfig::default();
    cfg.sample_rate = sample_rate;
    cfg.f_max = (sample_rate as f32) / 2.0;

    let mel_filters = model_dir.join("mel_filters.bin");
    if mel_filters.exists() && sample_rate == 16000 {
        let ex = MelSpectrogramExtractor::with_mel_filters_from_file(cfg, &mel_filters)
            .with_context(|| {
                format!(
                    "Не удалось загрузить mel_filters.bin: {}",
                    mel_filters.display()
                )
            })?;
        Ok(ex)
    } else {
        Ok(MelSpectrogramExtractor::new(cfg))
    }
}

fn l2_normalize(v: &mut [f32]) {
    let mut norm2 = 0.0f32;
    for &x in v.iter() {
        norm2 += x * x;
    }
    let norm = norm2.sqrt();
    let norm = if norm.is_finite() && norm > 1e-6 {
        norm
    } else {
        1e-6
    };
    for x in v {
        *x /= norm;
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        s += x * y;
    }
    s
}

fn kmeans_cosine(embs: &[Vec<f32>], k: usize, max_iter: usize) -> Vec<usize> {
    let n = embs.len();
    if n == 0 || k <= 1 {
        return vec![0; n];
    }

    let d = embs[0].len();

    // Детерминированная инициализация "farthest point" по cosine distance.
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    centroids.push(embs[0].clone());
    while centroids.len() < k {
        let mut best_i = 0usize;
        let mut best_dist = -1.0f32;
        for (i, e) in embs.iter().enumerate() {
            let best_sim = centroids.iter().map(|c| dot(e, c)).fold(-1.0f32, f32::max);
            let dist = 1.0 - best_sim;
            if dist > best_dist {
                best_dist = dist;
                best_i = i;
            }
        }
        centroids.push(embs[best_i].clone());
    }

    let mut assign = vec![0usize; n];
    let mut prev_assign = vec![usize::MAX; n];

    for _ in 0..max_iter {
        // 1) Assignment step.
        for (i, e) in embs.iter().enumerate() {
            let mut best_c = 0usize;
            let mut best_sim = f32::NEG_INFINITY;
            for (c, cent) in centroids.iter().enumerate() {
                let sim = dot(e, cent);
                if sim > best_sim {
                    best_sim = sim;
                    best_c = c;
                }
            }
            assign[i] = best_c;
        }

        if assign == prev_assign {
            break;
        }
        prev_assign.clone_from(&assign);

        // 2) Update step.
        let mut sums: Vec<Vec<f32>> = vec![vec![0.0; d]; k];
        let mut counts: Vec<usize> = vec![0; k];
        for (e, &c) in embs.iter().zip(assign.iter()) {
            counts[c] += 1;
            let sum = &mut sums[c];
            for j in 0..d {
                sum[j] += e[j];
            }
        }

        for c in 0..k {
            if counts[c] == 0 {
                // Пустой кластер: берём точку, которая дальше всего от текущих центроидов.
                let mut best_i = 0usize;
                let mut best_dist = -1.0f32;
                for (i, e) in embs.iter().enumerate() {
                    let best_sim = centroids
                        .iter()
                        .map(|cent| dot(e, cent))
                        .fold(-1.0f32, f32::max);
                    let dist = 1.0 - best_sim;
                    if dist > best_dist {
                        best_dist = dist;
                        best_i = i;
                    }
                }
                centroids[c] = embs[best_i].clone();
                continue;
            }

            let inv = 1.0 / (counts[c] as f32);
            let cent = &mut centroids[c];
            for j in 0..d {
                cent[j] = sums[c][j] * inv;
            }
            l2_normalize(cent);
        }
    }

    assign
}

fn remap_clusters_by_first_segment(segments: &[(usize, usize)], assign: &mut [usize], k: usize) {
    if k <= 1 {
        return;
    }

    let mut min_start: Vec<usize> = vec![usize::MAX; k];
    for (&(start, _), &c) in segments.iter().zip(assign.iter()) {
        min_start[c] = min_start[c].min(start);
    }

    let mut order: Vec<(usize, usize)> = (0..k).map(|c| (min_start[c], c)).collect();
    order.sort_by_key(|(s, c)| (*s, *c));

    let mut map = vec![0usize; k];
    for (new_id, (_, old_id)) in order.into_iter().enumerate() {
        map[old_id] = new_id;
    }

    for c in assign {
        *c = map[*c];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_splits_two_blobs() {
        // Два "говорящих": около (1,0) и около (0,1).
        let embs = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.95, 0.05],
            vec![0.0, 1.0],
            vec![0.1, 0.9],
            vec![0.05, 0.95],
        ];

        let mut normed = embs.clone();
        for v in &mut normed {
            l2_normalize(v);
        }

        let assign = kmeans_cosine(&normed, 2, 25);
        // Проверяем, что первые три в одном кластере, последние три в другом.
        assert_eq!(assign[0], assign[1]);
        assert_eq!(assign[1], assign[2]);
        assert_eq!(assign[3], assign[4]);
        assert_eq!(assign[4], assign[5]);
        assert_ne!(assign[0], assign[3]);
    }
}
