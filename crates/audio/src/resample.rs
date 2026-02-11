//! Audio resampling.

use asr_core::{AsrError, AsrResult, AudioBuffer};
use rubato::{FftFixedInOut, Resampler as RubatoResampler};

/// Audio resampler for converting sample rates.
pub struct Resampler {
    target_sample_rate: usize,
}

impl Resampler {
    /// Create a new resampler with target sample rate.
    pub fn new(target_sample_rate: usize) -> Self {
        Self { target_sample_rate }
    }

    /// Resample audio buffer to target sample rate.
    pub fn resample(&self, buffer: &AudioBuffer) -> AsrResult<AudioBuffer> {
        if buffer.sample_rate == self.target_sample_rate {
            return Ok(buffer.clone());
        }

        // Ensure mono audio
        if buffer.channels != 1 {
            return Err(AsrError::Audio(
                "Resampling requires mono audio. Use to_mono() first.".to_string(),
            ));
        }

        let ratio = self.target_sample_rate as f64 / buffer.sample_rate as f64;

        // Calculate chunk sizes for FftFixedInOut
        let chunk_size = 1024;
        let output_chunk_size = (chunk_size as f64 * ratio).ceil() as usize;

        let mut resampler = FftFixedInOut::<f32>::new(
            buffer.sample_rate,
            self.target_sample_rate,
            chunk_size,
            1, // mono
        )
        .map_err(|e| AsrError::Audio(format!("Failed to create resampler: {}", e)))?;

        let mut output = Vec::with_capacity((buffer.samples.len() as f64 * ratio) as usize);

        // Process in chunks
        let mut pos = 0;
        while pos + chunk_size <= buffer.samples.len() {
            let input_chunk = vec![buffer.samples[pos..pos + chunk_size].to_vec()];
            let output_chunk = resampler
                .process(&input_chunk, None)
                .map_err(|e| AsrError::Audio(format!("Resampling failed: {}", e)))?;
            output.extend_from_slice(&output_chunk[0]);
            pos += chunk_size;
        }

        // Handle remaining samples (pad with zeros if needed)
        if pos < buffer.samples.len() {
            let mut remaining = buffer.samples[pos..].to_vec();
            remaining.resize(chunk_size, 0.0);
            let input_chunk = vec![remaining];
            let output_chunk = resampler
                .process(&input_chunk, None)
                .map_err(|e| AsrError::Audio(format!("Resampling failed: {}", e)))?;

            // Only take the proportional amount of output
            let remaining_ratio = (buffer.samples.len() - pos) as f64 / chunk_size as f64;
            let take = (output_chunk_size as f64 * remaining_ratio) as usize;
            output.extend_from_slice(&output_chunk[0][..take.min(output_chunk[0].len())]);
        }

        Ok(AudioBuffer::new(output, self.target_sample_rate, 1))
    }
}

impl Default for Resampler {
    fn default() -> Self {
        Self::new(16000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resampler_no_change() {
        let buffer = AudioBuffer::new(vec![0.0; 1024], 16000, 1);
        let resampler = Resampler::new(16000);
        let result = resampler.resample(&buffer).unwrap();

        assert_eq!(result.sample_rate, 16000);
        assert_eq!(result.samples.len(), buffer.samples.len());
    }
}
