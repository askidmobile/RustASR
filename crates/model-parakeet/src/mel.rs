//! Mel-спектрограмма в стиле NeMo для Parakeet.
//!
//! Особенности по сравнению с другими моделями:
//! - Pre-emphasis фильтр (0.97)
//! - Per-feature нормализация (каждый mel-бин отдельно)
//! - Mel-фильтры загружаются из весов модели (`preprocessor.featurizer.fb`)
//! - Окно FFT загружается из весов (`preprocessor.featurizer.window`)
//! - center=True (отступ win_length/2 перед STFT)

use std::f32::consts::PI;

use candle_core::{Device, Tensor};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use asr_core::AsrResult;

use crate::config::ParakeetConfig;

/// Экстрактор mel-спектрограммы для Parakeet (NeMo-стиль).
pub struct ParakeetMelExtractor {
    /// Mel-фильтры [n_mels, n_fft/2+1].
    mel_filters: Vec<Vec<f32>>,
    /// Оконная функция (Hann, загружена из весов или создана).
    window: Vec<f32>,
    /// Размер FFT.
    n_fft: usize,
    /// Шаг между фреймами.
    hop_length: usize,
    /// Длина окна.
    win_length: usize,
    /// Количество mel-бинов.
    n_mels: usize,
    /// Коэффициент предварительного усиления.
    preemph: f32,
    /// Тип нормализации ("per_feature" или "per_utterance").
    normalize: String,
}

impl ParakeetMelExtractor {
    /// Создать экстрактор с mel-фильтрами и окном из тензоров модели.
    ///
    /// `mel_fb` — тензор [1, n_mels, n_fft/2+1] из весов
    /// `window` — тензор [win_length] из весов (или None для генерации Hann)
    pub fn from_tensors(
        config: &ParakeetConfig,
        mel_fb: &Tensor,
        window_tensor: Option<&Tensor>,
    ) -> AsrResult<Self> {
        let n_mels = config.preprocessor.features;
        let n_fft = config.preprocessor.n_fft;
        let hop_length = config.hop_length();
        let win_length = config.win_length();

        // Извлечь mel-фильтры: [1, n_mels, n_fft/2+1] → [n_mels][n_fft/2+1]
        let fb_data = mel_fb.squeeze(0)?.to_vec2::<f32>()?;
        assert_eq!(fb_data.len(), n_mels, "mel_filters: ожидается {n_mels} бинов");

        // Извлечь окно или сгенерировать Hann
        let window = if let Some(w) = window_tensor {
            w.to_vec1::<f32>()?
        } else {
            hann_window(win_length)
        };

        Ok(Self {
            mel_filters: fb_data,
            window,
            n_fft,
            hop_length,
            win_length,
            n_mels,
            preemph: config.preprocessor.preemph as f32,
            normalize: config.preprocessor.normalize.clone(),
        })
    }

    /// Создать экстрактор со сгенерированными Slaney mel-фильтрами.
    pub fn new(config: &ParakeetConfig) -> Self {
        let n_mels = config.preprocessor.features;
        let n_fft = config.preprocessor.n_fft;
        let hop_length = config.hop_length();
        let win_length = config.win_length();
        let sample_rate = config.preprocessor.sample_rate;

        let mel_filters = create_slaney_mel_filterbank(
            n_fft,
            n_mels,
            0.0,
            sample_rate as f32 / 2.0,
            sample_rate,
        );

        Self {
            mel_filters,
            window: hann_window(win_length),
            n_fft,
            hop_length,
            win_length,
            n_mels,
            preemph: config.preprocessor.preemph as f32,
            normalize: config.preprocessor.normalize.clone(),
        }
    }

    /// Извлечь mel-спектрограмму из аудио-сэмплов.
    ///
    /// Возвращает тензор [1, n_mels, time] на указанном устройстве.
    pub fn extract(&self, samples: &[f32], device: &Device) -> AsrResult<Tensor> {
        // 1. Pre-emphasis: y[n] = x[n] - preemph * x[n-1]
        let preemph_samples = self.apply_preemphasis(samples);

        // 2. STFT с center padding
        let stft = self.compute_stft(&preemph_samples);

        // 3. Power spectrum → mel filterbank
        let mel = self.apply_mel_filterbank(&stft);

        // 4. Log
        let log_mel = self.apply_log(&mel);

        // 5. Normalization
        let norm_mel = self.apply_normalization(&log_mel);

        // 6. Создать тензор [1, n_mels, time]
        let n_frames = norm_mel[0].len();
        let mut flat = Vec::with_capacity(self.n_mels * n_frames);
        for bin in &norm_mel {
            flat.extend_from_slice(bin);
        }

        let tensor = Tensor::from_vec(flat, (1, self.n_mels, n_frames), device)?;
        Ok(tensor)
    }

    /// Pre-emphasis: y[n] = x[n] - α * x[n-1].
    fn apply_preemphasis(&self, samples: &[f32]) -> Vec<f32> {
        if self.preemph == 0.0 || samples.is_empty() {
            return samples.to_vec();
        }
        let mut out = Vec::with_capacity(samples.len());
        out.push(samples[0]); // первый отсчёт без изменений
        for i in 1..samples.len() {
            out.push(samples[i] - self.preemph * samples[i - 1]);
        }
        out
    }

    /// STFT с center padding и Hann-окном.
    ///
    /// Возвращает `[n_fft/2+1][n_frames]` — амплитудный спектр².
    fn compute_stft(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let n_fft_bins = self.n_fft / 2 + 1;

        // Center padding: добавить win_length/2 нулей с обоих сторон
        let pad = self.win_length / 2;
        let mut padded = vec![0.0f32; pad];
        padded.extend_from_slice(samples);
        padded.resize(padded.len() + pad, 0.0);

        // Количество фреймов
        let n_frames = if padded.len() >= self.n_fft {
            (padded.len() - self.n_fft) / self.hop_length + 1
        } else {
            0
        };

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.n_fft);

        // Результат: [n_fft_bins][n_frames]
        let mut power_spectrum = vec![vec![0.0f32; n_frames]; n_fft_bins];

        let mut fft_buffer = vec![Complex::new(0.0f32, 0.0); self.n_fft];

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;

            // Применить окно
            for i in 0..self.n_fft {
                let sample = if i < self.win_length && start + i < padded.len() {
                    padded[start + i]
                } else {
                    0.0
                };
                let w = if i < self.window.len() {
                    self.window[i]
                } else {
                    0.0
                };
                fft_buffer[i] = Complex::new(sample * w, 0.0);
            }

            // FFT
            fft.process(&mut fft_buffer);

            // Power spectrum: |X[k]|²
            for k in 0..n_fft_bins {
                power_spectrum[k][frame_idx] = fft_buffer[k].norm_sqr();
            }
        }

        power_spectrum
    }

    /// Применить mel-фильтры к спектру мощности.
    ///
    /// Вход: [n_fft_bins][n_frames], выход: [n_mels][n_frames].
    fn apply_mel_filterbank(&self, power_spectrum: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n_frames = if power_spectrum.is_empty() {
            0
        } else {
            power_spectrum[0].len()
        };
        let n_fft_bins = power_spectrum.len();

        let mut mel = vec![vec![0.0f32; n_frames]; self.n_mels];

        for m in 0..self.n_mels {
            let filter = &self.mel_filters[m];
            let filter_len = filter.len().min(n_fft_bins);
            for t in 0..n_frames {
                let mut sum = 0.0f32;
                for k in 0..filter_len {
                    sum += filter[k] * power_spectrum[k][t];
                }
                mel[m][t] = sum;
            }
        }

        mel
    }

    /// Применить логарифм с защитой от нуля.
    fn apply_log(&self, mel: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // NeMo использует log_zero_guard = 2^-24 ≈ 5.96e-8
        let guard: f32 = 2.0f32.powi(-24);

        mel.iter()
            .map(|bin| bin.iter().map(|&v| (v.max(guard)).ln()).collect())
            .collect()
    }

    /// Per-feature нормализация: для каждого mel-бина вычитаем среднее и делим на std.
    fn apply_normalization(&self, mel: &[Vec<f32>]) -> Vec<Vec<f32>> {
        match self.normalize.as_str() {
            "per_feature" => {
                // Для каждого mel-бина: (x - mean) / max(std, 1e-5)
                mel.iter()
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
                    .collect()
            }
            "per_utterance" => {
                // Глобальная нормализация по всему спектрограмме
                let total: f32 = mel.iter().flat_map(|b| b.iter()).sum();
                let count = mel.iter().map(|b| b.len()).sum::<usize>() as f32;
                let mean = total / count.max(1.0);
                let var: f32 = mel
                    .iter()
                    .flat_map(|b| b.iter())
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f32>()
                    / count.max(1.0);
                let std = var.sqrt().max(1e-5);
                mel.iter()
                    .map(|bin| bin.iter().map(|&x| (x - mean) / std).collect())
                    .collect()
            }
            _ => mel.to_vec(),
        }
    }
}

/// Генерация Hann-окна длиной `length`.
fn hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / length as f32).cos()))
        .collect()
}

/// Создать Slaney mel-фильтры (треугольные, slaney-нормализованные).
fn create_slaney_mel_filterbank(
    n_fft: usize,
    n_mels: usize,
    f_min: f32,
    f_max: f32,
    sample_rate: usize,
) -> Vec<Vec<f32>> {
    let n_fft_bins = n_fft / 2 + 1;

    // Slaney mel scale: линейная < 1000 Гц, логарифмическая > 1000 Гц
    let mel_min = hz_to_mel_slaney(f_min);
    let mel_max = hz_to_mel_slaney(f_max);

    // n_mels + 2 точек для треугольных фильтров
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz_slaney(m)).collect();

    // Частотные бины FFT
    let fft_freqs: Vec<f32> = (0..n_fft_bins)
        .map(|k| k as f32 * sample_rate as f32 / n_fft as f32)
        .collect();

    let mut filters = vec![vec![0.0f32; n_fft_bins]; n_mels];

    for m in 0..n_mels {
        let f_left = hz_points[m];
        let f_center = hz_points[m + 1];
        let f_right = hz_points[m + 2];

        // Slaney нормализация: 2 / (f_right - f_left)
        let norm = 2.0 / (f_right - f_left);

        for k in 0..n_fft_bins {
            let freq = fft_freqs[k];
            let val = if freq >= f_left && freq < f_center {
                (freq - f_left) / (f_center - f_left)
            } else if freq >= f_center && freq <= f_right {
                (f_right - freq) / (f_right - f_center)
            } else {
                0.0
            };
            filters[m][k] = val * norm;
        }
    }

    filters
}

fn hz_to_mel_slaney(freq: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = 1000.0 / F_SP; // = 15.0
    const LOG_STEP: f32 = 0.068_751_78; // ln(6.4) / 27.0

    if freq < MIN_LOG_HZ {
        freq / F_SP
    } else {
        MIN_LOG_MEL + (freq / MIN_LOG_HZ).ln() / LOG_STEP
    }
}

fn mel_to_hz_slaney(mel: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = 15.0;
    const LOG_STEP: f32 = 0.068_751_78;

    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * LOG_STEP).exp()
    }
}
