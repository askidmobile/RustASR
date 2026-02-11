//! Mel-спектрограмма для GigaAM.
//!
//! Отдельная реализация, совместимая с torchaudio.MelSpectrogram:
//! - HTK mel scale, mel_norm=None
//! - STFT с center=false
//! - SpecScaler: log(clamp(x, 1e-9, 1e9))
//!
//! Параметры GigaAM v3 E2E CTC:
//!   sample_rate=16000, n_mels=64, n_fft=320,
//!   hop_length=160, win_length=320, center=false

use std::f32::consts::PI;

use candle_core::{Device, Result, Tensor};
use rustfft::{num_complex::Complex, FftPlanner};

use crate::config::PreprocessorConfig;

/// Mel-спектрограмма, совместимая с GigaAM (torchaudio).
pub struct GigaAmMelExtractor {
    config: PreprocessorConfig,
    /// Оконная функция Hann.
    window: Vec<f32>,
    /// Mel-фильтрбанк: (n_fft/2+1, n_mels).
    filterbank: Vec<Vec<f32>>,
}

impl GigaAmMelExtractor {
    pub fn new(config: PreprocessorConfig) -> Self {
        let window = hann_window(config.win_length);
        let n_freqs = config.n_fft / 2 + 1;
        let filterbank = create_htk_mel_filterbank(
            n_freqs,
            config.features,
            0.0,
            config.sample_rate as f32 / 2.0,
            config.sample_rate,
        );
        Self { config, window, filterbank }
    }

    /// Извлечь mel-спектрограмму для подачи в Conformer.
    ///
    /// Возвращает тензор (1, n_mels, num_frames) — формат для Conv1d.
    pub fn extract(&self, samples: &[f32], device: &Device) -> Result<Tensor> {
        // 1. STFT (power spectrum)
        let spectrogram = self.stft(samples);

        // 2. Применить mel-фильтрбанк
        let mel_spec = self.apply_mel_filters(&spectrogram);

        // 3. Log (SpecScaler: log(clamp(x, 1e-9, 1e9)))
        let log_mel = self.log_mel(&mel_spec);

        let num_frames = log_mel.len();
        let n_mels = self.config.features;

        // Формат для ConformerEncoder: (1, n_mels, num_frames)
        // Каждый фрейм — это столбец из n_mels значений
        let mut flat = Vec::with_capacity(n_mels * num_frames);
        for mel_bin in 0..n_mels {
            for frame in &log_mel {
                flat.push(frame[mel_bin]);
            }
        }

        Tensor::from_vec(flat, (1, n_mels, num_frames), device)
    }

    /// STFT (center=false) → power spectrum.
    fn stft(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let n_fft = self.config.n_fft;
        let hop_length = self.config.hop_length;
        let win_length = self.config.win_length;
        let n = samples.len();

        // center=false: num_frames = (n - win_length) / hop_length + 1
        if n < win_length {
            return vec![];
        }
        let num_frames = (n - win_length) / hop_length + 1;

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);

        let mut spectrogram = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_length;

            // Применить окно и создать входной буфер для FFT
            let mut buffer: Vec<Complex<f32>> = (0..n_fft)
                .map(|i| {
                    let sample = if i < win_length && start + i < n {
                        samples[start + i] * self.window[i]
                    } else {
                        0.0
                    };
                    Complex::new(sample, 0.0)
                })
                .collect();

            fft.process(&mut buffer);

            // Power spectrum (magnitude²), только положительные частоты
            let power: Vec<f32> = buffer
                .iter()
                .take(n_fft / 2 + 1)
                .map(|c| c.re * c.re + c.im * c.im)
                .collect();

            spectrogram.push(power);
        }

        spectrogram
    }

    /// Применить mel-фильтрбанк к power spectrum.
    fn apply_mel_filters(&self, spectrogram: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n_mels = self.config.features;
        let n_freqs = self.config.n_fft / 2 + 1;

        spectrogram
            .iter()
            .map(|frame| {
                let mut mel_frame = Vec::with_capacity(n_mels);
                for mel_bin in 0..n_mels {
                    let mut energy = 0.0f32;
                    for (k, frame_val) in frame.iter().enumerate().take(n_freqs) {
                        energy += frame_val * self.filterbank[k][mel_bin];
                    }
                    mel_frame.push(energy);
                }
                mel_frame
            })
            .collect()
    }

    /// SpecScaler: log(clamp(x, 1e-9, 1e9)).
    fn log_mel(&self, mel_spec: &[Vec<f32>]) -> Vec<Vec<f32>> {
        mel_spec
            .iter()
            .map(|frame| {
                frame.iter().map(|&x| x.clamp(1e-9, 1e9).ln()).collect()
            })
            .collect()
    }

    /// Количество фреймов для заданной длины аудио.
    pub fn num_frames(&self, num_samples: usize) -> usize {
        if num_samples < self.config.win_length {
            return 0;
        }
        (num_samples - self.config.win_length) / self.config.hop_length + 1
    }
}

// -----------------------------------------------------------------------
// Вспомогательные функции
// -----------------------------------------------------------------------

/// Оконная функция Hann.
fn hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| {
            let phase = 2.0 * PI * i as f32 / length as f32;
            0.5 * (1.0 - phase.cos())
        })
        .collect()
}

/// Конвертация Гц → mel (HTK шкала).
fn hz_to_mel_htk(freq: f32) -> f32 {
    2595.0 * (1.0 + freq / 700.0).log10()
}

/// Конвертация mel → Гц (HTK шкала).
fn mel_to_hz_htk(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

/// Создать HTK mel-фильтрбанк (совместимо с torchaudio, norm=None).
///
/// Возвращает матрицу (n_freqs, n_mels) — для умножения на power spectrum.
fn create_htk_mel_filterbank(
    n_freqs: usize,
    n_mels: usize,
    f_min: f32,
    f_max: f32,
    sample_rate: usize,
) -> Vec<Vec<f32>> {
    let mel_min = hz_to_mel_htk(f_min);
    let mel_max = hz_to_mel_htk(f_max);

    // n_mels + 2 точек в mel-пространстве (включая границы)
    let n_points = n_mels + 2;
    let mel_points: Vec<f32> = (0..n_points)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_points - 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz_htk(m)).collect();

    // Частоты для каждого FFT-бина
    let all_freqs: Vec<f32> = (0..n_freqs)
        .map(|k| k as f32 * sample_rate as f32 / ((n_freqs - 1) * 2) as f32)
        .collect();

    // Разности между соседними mel-точками в Гц
    let f_diff: Vec<f32> = hz_points.windows(2).map(|w| w[1] - w[0]).collect();

    // Создать фильтрбанк
    let mut filterbank = vec![vec![0.0f32; n_mels]; n_freqs];

    for (k, &freq) in all_freqs.iter().enumerate() {
        for m in 0..n_mels {
            let f_left = hz_points[m];
            let _f_center = hz_points[m + 1];
            let f_right = hz_points[m + 2];

            // Нарастающий склон
            let down_slope = (freq - f_left) / f_diff[m];
            // Убывающий склон
            let up_slope = (f_right - freq) / f_diff[m + 1];

            filterbank[k][m] = 0.0f32.max(down_slope.min(up_slope));
        }
    }

    filterbank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window() {
        let w = hann_window(320);
        assert_eq!(w.len(), 320);
        // Первый элемент ≈ 0
        assert!(w[0].abs() < 1e-6);
        // Середина ≈ 1
        assert!((w[160] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let fb = create_htk_mel_filterbank(161, 64, 0.0, 8000.0, 16000);
        assert_eq!(fb.len(), 161);        // n_freqs
        assert_eq!(fb[0].len(), 64);      // n_mels
    }

    #[test]
    fn test_hz_mel_roundtrip() {
        let hz = 1000.0;
        let mel = hz_to_mel_htk(hz);
        let hz2 = mel_to_hz_htk(mel);
        assert!((hz - hz2).abs() < 0.01);
    }

    #[test]
    fn test_stft_center_false() {
        let config = PreprocessorConfig {
            sample_rate: 16000,
            features: 64,
            win_length: 320,
            hop_length: 160,
            n_fft: 320,
            mel_scale: "htk".to_string(),
            center: false,
        };
        let extractor = GigaAmMelExtractor::new(config);

        // 1 секунда @ 16kHz = 16000 сэмплов
        // Фреймы: (16000 - 320) / 160 + 1 = 99
        let samples = vec![0.0f32; 16000];
        let num_frames = extractor.num_frames(samples.len());
        assert_eq!(num_frames, 99);
    }
}
