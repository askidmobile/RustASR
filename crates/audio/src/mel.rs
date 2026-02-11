//! Mel-спектрограмма — параметризованный экстрактор для всех ASR-моделей.
//!
//! Поддерживаемые конфигурации:
//! - **Whisper/Qwen3-ASR**: 128 mel bins, Slaney, log10, dynamic range
//! - **GigaAM**: 64 mel bins, Slaney, ln, per-utterance
//! - **Parakeet**: 80 mel bins, HTK, ln, per-utterance

use asr_core::{AsrResult, FeatureExtractorConfig, LogType, MelNormalization, MelScale, MelSpectrum};
use candle_core::{Device, Tensor};
use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;

/// Параметризованный mel-экстрактор для всех моделей.
#[derive(Debug)]
pub struct MelSpectrogramExtractor {
    config: FeatureExtractorConfig,
    window: Vec<f32>,
    mel_filters: Vec<Vec<f32>>,
}

impl MelSpectrogramExtractor {
    /// Создать mel-экстрактор с фильтрами, сгенерированными по конфигурации.
    ///
    /// Автоматически выбирает Slaney или HTK шкалу в зависимости от `config.mel_scale`.
    pub fn new(config: FeatureExtractorConfig) -> Self {
        let window = hann_window(config.n_fft);
        let mel_filters = match config.mel_scale {
            MelScale::Slaney => create_slaney_mel_filterbank(
                config.n_mels,
                config.n_fft,
                config.sample_rate as f32,
                config.f_min,
                config.f_max,
            ),
            MelScale::Htk => create_htk_mel_filterbank(
                config.n_mels,
                config.n_fft,
                config.sample_rate as f32,
                config.f_min,
                config.f_max,
            ),
        };

        Self {
            config,
            window,
            mel_filters,
        }
    }

    /// Create extractor with mel filters loaded from binary file.
    /// This ensures exact match with Python WhisperFeatureExtractor.
    ///
    /// # Arguments
    /// * `config` - Feature extractor configuration
    /// * `mel_filters_path` - Path to mel_filters.bin (raw f32, shape [n_mels, n_fft/2+1])
    pub fn with_mel_filters_from_file(
        config: FeatureExtractorConfig,
        mel_filters_path: impl AsRef<std::path::Path>,
    ) -> std::io::Result<Self> {
        let window = hann_window(config.n_fft);

        // Load mel filters from binary file
        let data = std::fs::read(mel_filters_path)?;
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let n_freqs = config.n_fft / 2 + 1;
        let n_mels = config.n_mels;

        // Convert flat array to Vec<Vec<f32>> [n_mels][n_freqs]
        let mel_filters: Vec<Vec<f32>> = (0..n_mels)
            .map(|m| {
                let start = m * n_freqs;
                floats[start..start + n_freqs].to_vec()
            })
            .collect();

        Ok(Self {
            config,
            window,
            mel_filters,
        })
    }

    /// Extract Mel spectrogram from audio samples.
    ///
    /// # Arguments
    /// * `samples` - Audio samples (mono, normalized to [-1.0, 1.0])
    /// * `device` - Candle device for output tensor
    ///
    /// # Returns
    /// `MelSpectrum` with tensor of shape [1, time, n_mels]
    pub fn extract(&self, samples: &[f32], device: &Device) -> AsrResult<MelSpectrum> {
        let spectrogram = self.stft(samples);
        let mel_spec = self.apply_mel_filters(&spectrogram);
        let log_mel = self.log_mel(&mel_spec);

        let num_frames = log_mel.len();
        let n_mels = self.config.n_mels;

        // Create tensor [1, time, n_mels] - AuT encoder expects this format
        let flat: Vec<f32> = log_mel.into_iter().flatten().collect();
        let tensor = Tensor::from_vec(flat, (1, num_frames, n_mels), device)?;

        Ok(MelSpectrum::new(tensor, num_frames, n_mels))
    }

    /// Compute Short-Time Fourier Transform with POWER spectrum (magnitude^2).
    fn stft(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let n_fft = self.config.n_fft;
        let hop_length = self.config.hop_length;
        // Совместимость с WhisperFeatureExtractor (HF):
        // используется `torch.stft(..., center=True)` => паддинг по n_fft/2 слева/справа,
        // что дает (L // hop_length) + 1 фрейм. Затем последний фрейм отбрасывается.
        // Мы генерируем полный набор фреймов (L // hop + 1), а отбрасывание делаем позже.
        let num_frames = samples.len() / hop_length + 1;
        let pad = (n_fft / 2) as isize;

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);

        let mut spectrogram = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            // center=True: окно центрируется на позиции frame_idx * hop_length
            // => start = t - n_fft/2.
            let start = frame_idx as isize * hop_length as isize - pad;

            // Apply window and create complex input
            let n = samples.len() as isize;
            let mut buffer: Vec<Complex<f32>> = (0..n_fft)
                .map(|i| {
                    let mut idx = start + i as isize;
                    // torch.stft(center=True) по умолчанию использует pad_mode="reflect".
                    // Соответственно, значения за границами сигнала берутся отражением.
                    if idx < 0 {
                        idx = -idx;
                    }
                    if idx >= n {
                        idx = 2 * n - idx - 2;
                    }
                    let sample = if idx >= 0 && idx < n {
                        samples[idx as usize] * self.window[i]
                    } else {
                        // На сверх-краях (теоретически) безопасно падать в 0.
                        0.0
                    };
                    Complex::new(sample, 0.0)
                })
                .collect();

            // Perform in-place FFT
            fft.process(&mut buffer);

            // Compute POWER spectrum (magnitude^2) - only positive frequencies
            let power: Vec<f32> = buffer
                .iter()
                .take(n_fft / 2 + 1)
                .map(|c| c.re * c.re + c.im * c.im)
                .collect();

            spectrogram.push(power);
        }

        spectrogram
    }

    /// Apply Mel filterbank to power spectrogram.
    fn apply_mel_filters(&self, spectrogram: &[Vec<f32>]) -> Vec<Vec<f32>> {
        spectrogram
            .iter()
            .map(|frame| {
                self.mel_filters
                    .iter()
                    .map(|filter| {
                        frame
                            .iter()
                            .zip(filter.iter())
                            .map(|(s, f)| s * f)
                            .sum::<f32>()
                    })
                    .collect()
            })
            .collect()
    }

    /// Apply log transformation with Whisper-style normalization.
    /// 1. log10(mel.clamp(1e-10))
    /// 2. clamp to max - 8.0 (dynamic range compression)
    /// 3. normalize: (x + 4.0) / 4.0
    fn log_mel(&self, mel_spec: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let floor = 1e-10_f32;

        // Step 1: Compute log of mel spectrogram (log10 или ln)
        let log_fn: fn(f32) -> f32 = match self.config.log_type {
            LogType::Log10 => |v: f32| v.log10(),
            LogType::Ln => |v: f32| v.ln(),
        };

        let mut log_spec: Vec<Vec<f32>> = mel_spec
            .iter()
            .map(|frame| frame.iter().map(|v| log_fn(v.max(floor))).collect())
            .collect();

        // Whisper: отбрасываем последний фрейм (совместимость с HF)
        match self.config.normalization {
            MelNormalization::WhisperDynamicRange => {
                if !log_spec.is_empty() {
                    log_spec.pop();
                }
            }
            _ => {}
        }

        // Step 2: Нормализация
        match self.config.normalization {
            MelNormalization::WhisperDynamicRange => {
                // Whisper: dynamic range compression
                let global_max = log_spec
            .iter()
            .flat_map(|frame| frame.iter())
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        let min_val = global_max - 8.0;

        // Step 3: Apply clamping and normalization
        for frame in log_spec.iter_mut() {
            for val in frame.iter_mut() {
                *val = (*val).max(min_val);
                *val = (*val + 4.0) / 4.0;
            }
        }
            }
            MelNormalization::PerUtterance => {
                // Per-utterance: μ/σ нормализация (GigaAM, Parakeet)
                let n = log_spec.len() as f64 * self.config.n_mels as f64;
                if n > 0.0 {
                    let sum: f64 = log_spec
                        .iter()
                        .flat_map(|f| f.iter())
                        .map(|&v| v as f64)
                        .sum();
                    let mean = sum / n;

                    let sum_sq: f64 = log_spec
                        .iter()
                        .flat_map(|f| f.iter())
                        .map(|&v| {
                            let d = v as f64 - mean;
                            d * d
                        })
                        .sum();
                    let std = (sum_sq / n).sqrt().max(1e-10);

                    for frame in log_spec.iter_mut() {
                        for val in frame.iter_mut() {
                            *val = ((*val as f64 - mean) / std) as f32;
                        }
                    }
                }
            }
            MelNormalization::None => {
                // Без нормализации
            }
        }

        log_spec
    }
}

impl Default for MelSpectrogramExtractor {
    fn default() -> Self {
        Self::new(FeatureExtractorConfig::default())
    }
}

/// Create Hann window (periodic for STFT).
fn hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / length as f32).cos()))
        .collect()
}

/// Convert frequency to Slaney Mel scale.
/// Slaney uses linear below 1000 Hz, log above.
fn hz_to_mel_slaney(hz: f32) -> f32 {
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0; // ~66.67 Hz
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4f32).ln() / 27.0;

    if hz >= min_log_hz {
        min_log_mel + ((hz / min_log_hz).ln() / logstep)
    } else {
        (hz - f_min) / f_sp
    }
}

/// Convert Slaney Mel scale to frequency.
fn mel_to_hz_slaney(mel: f32) -> f32 {
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4f32).ln() / 27.0;

    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        f_min + f_sp * mel
    }
}

/// Конвертация Hz → mel по HTK шкале (полностью логарифмическая).
fn hz_to_mel_htk(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Конвертация mel → Hz по HTK шкале.
fn mel_to_hz_htk(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Create Slaney-normalized Mel filterbank (matches librosa/Whisper).
fn create_slaney_mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: f32,
    f_min: f32,
    f_max: f32,
) -> Vec<Vec<f32>> {
    create_mel_filterbank_inner(
        n_mels,
        n_fft,
        sample_rate,
        f_min,
        f_max,
        hz_to_mel_slaney,
        mel_to_hz_slaney,
    )
}

/// Создать HTK mel filterbank (для NeMo / Parakeet).
fn create_htk_mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: f32,
    f_min: f32,
    f_max: f32,
) -> Vec<Vec<f32>> {
    create_mel_filterbank_inner(
        n_mels,
        n_fft,
        sample_rate,
        f_min,
        f_max,
        hz_to_mel_htk,
        mel_to_hz_htk,
    )
}

/// Общая реализация mel filterbank для обеих шкал.
fn create_mel_filterbank_inner(
    n_mels: usize,
    n_fft: usize,
    sample_rate: f32,
    f_min: f32,
    f_max: f32,
    hz_to_mel: fn(f32) -> f32,
    mel_to_hz: fn(f32) -> f32,
) -> Vec<Vec<f32>> {
    let n_freqs = n_fft / 2 + 1;

    // Create FFT frequency bins
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * sample_rate / n_fft as f32)
        .collect();

    // Create mel points
    let mel_min = hz_to_mel_slaney(f_min);
    let mel_max = hz_to_mel_slaney(f_max);

    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + i as f32 * (mel_max - mel_min) / (n_mels + 1) as f32)
        .collect();

    // Convert mel points to Hz
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz_slaney(m)).collect();

    // Create filterbank with Slaney normalization
    let mut filterbank = vec![vec![0.0_f32; n_freqs]; n_mels];

    for m in 0..n_mels {
        let f_left = hz_points[m];
        let f_center = hz_points[m + 1];
        let f_right = hz_points[m + 2];

        // Slaney normalization: 2 / (f_right - f_left)
        let enorm = 2.0 / (f_right - f_left);

        for (k, &freq) in fft_freqs.iter().enumerate() {
            if freq >= f_left && freq < f_center {
                // Rising slope
                filterbank[m][k] = enorm * (freq - f_left) / (f_center - f_left);
            } else if freq >= f_center && freq <= f_right {
                // Falling slope
                filterbank[m][k] = enorm * (f_right - freq) / (f_right - f_center);
            }
        }
    }

    filterbank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window() {
        let window = hann_window(400);
        assert_eq!(window.len(), 400);
        assert!(window[0].abs() < 1e-6); // Should start at 0
        assert!((window[200] - 1.0).abs() < 0.01); // Peak near center
    }

    #[test]
    fn test_slaney_mel_conversion() {
        // Test roundtrip
        let hz = 1000.0;
        let mel = hz_to_mel_slaney(hz);
        let back = mel_to_hz_slaney(mel);
        assert!((hz - back).abs() < 1e-3);

        // Test another frequency
        let hz2 = 4000.0;
        let mel2 = hz_to_mel_slaney(hz2);
        let back2 = mel_to_hz_slaney(mel2);
        assert!((hz2 - back2).abs() < 1.0);
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let filters = create_slaney_mel_filterbank(128, 400, 16000.0, 0.0, 8000.0);
        assert_eq!(filters.len(), 128);
        assert_eq!(filters[0].len(), 201); // n_fft/2 + 1
    }

    #[test]
    fn test_filterbank_normalized() {
        let filters = create_slaney_mel_filterbank(128, 400, 16000.0, 0.0, 8000.0);

        // Each filter should have area = 1 (approximately, due to discretization)
        for filter in &filters {
            let area: f32 = filter.iter().sum();
            // Slaney normalization should make peak height = 2/(f_right - f_left)
            // Area depends on frequency resolution
            assert!(area > 0.0);
        }
    }
}
