//! Mel-спектрограмма Nemotron 3.5 ASR (NeMo AudioToMelSpectrogramPreprocessor).
//!
//! Идентична Parakeet-препроцессору, КРОМЕ `normalize="NA"` (без нормализации).
//! - pre-emphasis 0.97
//! - mel-фильтры `preprocessor.featurizer.fb` [1,128,257] из весов
//! - окно `preprocessor.featurizer.window` [400] из весов (hann)
//! - center=True (pad n_fft/2), power spectrum, log с guard 2^-24
//! - normalize="NA": нормализация НЕ применяется

use candle_core::{DType, Device, Tensor};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

use asr_core::AsrResult;

use crate::config::NemotronConfig;

/// Экстрактор mel-спектрограммы для Nemotron.
pub struct NemotronMelExtractor {
    mel_filters: Vec<Vec<f32>>, // [n_mels][n_fft/2+1]
    window: Vec<f32>,           // [win_length]
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    n_mels: usize,
    preemph: f32,
    normalize: String,
}

impl NemotronMelExtractor {
    /// Создать из тензоров весов: `fb` [1,n_mels,n_fft/2+1], `window` [win_length].
    pub fn from_tensors(
        config: &NemotronConfig,
        mel_fb: &Tensor,
        window_tensor: &Tensor,
    ) -> AsrResult<Self> {
        let n_mels = config.preprocessor.features;
        let fb_data = mel_fb.squeeze(0)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        assert_eq!(fb_data.len(), n_mels, "fb: ожидается {n_mels} мел-бинов");
        let window = window_tensor.to_dtype(DType::F32)?.to_vec1::<f32>()?;

        Ok(Self {
            mel_filters: fb_data,
            window,
            n_fft: config.preprocessor.n_fft,
            hop_length: config.preprocessor.hop_length(),
            win_length: config.preprocessor.win_length(),
            n_mels,
            preemph: config.preprocessor.preemph as f32,
            normalize: config.preprocessor.normalize.clone(),
        })
    }

    /// Число валидных фреймов по соглашению NeMo: floor((n + n_fft - n_fft)/hop) = floor(n/hop).
    /// Кадры >= valid маскируются нулём (pad_value=0) в препроцессоре NeMo.
    pub fn valid_frames(&self, n_samples: usize) -> usize {
        n_samples / self.hop_length
    }

    /// Извлечь mel [1, n_mels, time] на устройстве.
    ///
    /// Кадры за пределами `valid_frames` обнуляются (как маскирование NeMo).
    pub fn extract(&self, samples: &[f32], device: &Device) -> AsrResult<Tensor> {
        let pe = self.apply_preemphasis(samples);
        let power = self.compute_stft(&pe);
        let mel = self.apply_mel_filterbank(&power);
        let log_mel = self.apply_log(&mel);
        let mut norm_mel = self.apply_normalization(&log_mel);

        // NeMo маскирует кадры >= seq_len (=floor(n/hop)) нулём.
        let valid = self.valid_frames(samples.len());
        let n_frames = norm_mel.first().map_or(0, |b| b.len());
        for bin in norm_mel.iter_mut() {
            for f in valid..n_frames {
                bin[f] = 0.0;
            }
        }

        let mut flat = Vec::with_capacity(self.n_mels * n_frames);
        for bin in &norm_mel {
            flat.extend_from_slice(bin);
        }
        Ok(Tensor::from_vec(flat, (1, self.n_mels, n_frames), device)?)
    }

    fn apply_preemphasis(&self, samples: &[f32]) -> Vec<f32> {
        if self.preemph == 0.0 || samples.is_empty() {
            return samples.to_vec();
        }
        let mut out = Vec::with_capacity(samples.len());
        out.push(samples[0]);
        for i in 1..samples.len() {
            out.push(samples[i] - self.preemph * samples[i - 1]);
        }
        out
    }

    /// STFT с center padding (pad n_fft/2). Возвращает power spectrum [n_fft/2+1][n_frames].
    fn compute_stft(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let n_bins = self.n_fft / 2 + 1;
        let pad = self.n_fft / 2;
        let mut padded = vec![0.0f32; pad];
        padded.extend_from_slice(samples);
        padded.resize(padded.len() + pad, 0.0);

        let n_frames = if padded.len() >= self.n_fft {
            (padded.len() - self.n_fft) / self.hop_length + 1
        } else {
            0
        };

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.n_fft);
        let mut power = vec![vec![0.0f32; n_frames]; n_bins];
        let mut buf = vec![Complex::new(0.0f32, 0.0); self.n_fft];

        // torch.stft при win_length < n_fft центрирует окно в кадре n_fft.
        let left = (self.n_fft - self.win_length) / 2;

        #[allow(clippy::needless_range_loop)]
        for f in 0..n_frames {
            let start = f * self.hop_length;
            for i in 0..self.n_fft {
                let s = if start + i < padded.len() {
                    padded[start + i]
                } else {
                    0.0
                };
                // окно центрировано: window[i-left] для i ∈ [left, left+win_length)
                let w = if i >= left && i - left < self.window.len() {
                    self.window[i - left]
                } else {
                    0.0
                };
                buf[i] = Complex::new(s * w, 0.0);
            }
            fft.process(&mut buf);
            for k in 0..n_bins {
                power[k][f] = buf[k].norm_sqr();
            }
        }
        power
    }

    fn apply_mel_filterbank(&self, power: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n_frames = power.first().map_or(0, |b| b.len());
        let n_bins = power.len();
        let mut mel = vec![vec![0.0f32; n_frames]; self.n_mels];
        #[allow(clippy::needless_range_loop)]
        for m in 0..self.n_mels {
            let filter = &self.mel_filters[m];
            let flen = filter.len().min(n_bins);
            for t in 0..n_frames {
                let mut sum = 0.0f32;
                for k in 0..flen {
                    sum += filter[k] * power[k][t];
                }
                mel[m][t] = sum;
            }
        }
        mel
    }

    fn apply_log(&self, mel: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // NeMo log_zero_guard_type="add": log(x + 2^-24)
        let guard: f32 = 2.0f32.powi(-24);
        mel.iter()
            .map(|bin| bin.iter().map(|&v| (v + guard).ln()).collect())
            .collect()
    }

    fn apply_normalization(&self, mel: &[Vec<f32>]) -> Vec<Vec<f32>> {
        match self.normalize.as_str() {
            "per_feature" => mel
                .iter()
                .map(|bin| {
                    let n = bin.len() as f32;
                    if n < 1.0 {
                        return bin.clone();
                    }
                    let mean = bin.iter().sum::<f32>() / n;
                    let var = bin.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
                    let std = var.sqrt().max(1e-5);
                    bin.iter().map(|&x| (x - mean) / std).collect()
                })
                .collect(),
            // "NA" / прочее — без нормализации (как в Nemotron)
            _ => mel.to_vec(),
        }
    }
}
