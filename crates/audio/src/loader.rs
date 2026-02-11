//! WAV file loading.

use asr_core::{AsrError, AsrResult, AudioBuffer};
use hound::WavReader;
use std::path::Path;

/// Load a WAV file and return an AudioBuffer.
pub fn load_wav(path: impl AsRef<Path>) -> AsrResult<AudioBuffer> {
    let path = path.as_ref();
    let reader =
        WavReader::open(path).map_err(|e| AsrError::Audio(format!("Failed to open WAV: {}", e)))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate as usize;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| AsrError::Audio(format!("Failed to read samples: {}", e)))?,
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1u32 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| AsrError::Audio(format!("Failed to read samples: {}", e)))?
        }
    };

    Ok(AudioBuffer::new(samples, sample_rate, channels))
}

/// Convert stereo audio to mono by averaging channels.
pub fn to_mono(buffer: &AudioBuffer) -> AudioBuffer {
    if buffer.channels == 1 {
        return buffer.clone();
    }

    let mono_samples: Vec<f32> = buffer
        .samples
        .chunks(buffer.channels)
        .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
        .collect();

    AudioBuffer::new(mono_samples, buffer.sample_rate, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_mono() {
        // Stereo buffer: L=1.0, R=0.0, L=0.5, R=0.5
        let stereo = AudioBuffer::new(vec![1.0, 0.0, 0.5, 0.5], 16000, 2);
        let mono = to_mono(&stereo);

        assert_eq!(mono.channels, 1);
        assert_eq!(mono.samples.len(), 2);
        assert!((mono.samples[0] - 0.5).abs() < 1e-6);
        assert!((mono.samples[1] - 0.5).abs() < 1e-6);
    }
}
