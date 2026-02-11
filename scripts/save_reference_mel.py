#!/usr/bin/env python3
"""Save reference mel spectrogram from WhisperFeatureExtractor for comparison."""

import numpy as np
import soundfile as sf
from transformers import WhisperFeatureExtractor

# Load audio
audio, sr = sf.read('test_30sec.wav')
if len(audio.shape) > 1:
    audio = audio[:, 0]
audio = audio[:sr*2]  # First 2 seconds

print(f"Audio samples: {len(audio)}")
print(f"Audio min: {audio.min():.6f}")
print(f"Audio max: {audio.max():.6f}")

# Extract features using WhisperFeatureExtractor
fe = WhisperFeatureExtractor.from_pretrained('models/qwen3-asr-0.6b')
result = fe(audio, return_tensors='np', sampling_rate=sr, padding=False, return_attention_mask=True)

features = result['input_features']  # [batch, n_mels, time]
print(f"\nMel shape: {features.shape}")
print(f"Mel min: {features.min():.4f}")
print(f"Mel max: {features.max():.4f}")
print(f"Mel mean: {features.mean():.4f}")

# Print first few values for comparison
print("\nFirst 5 values of mel bin 0:")
print(features[0, 0, :5])

print("\nFirst 5 values of mel bin 64:")
print(features[0, 64, :5])

# Save to file for Rust comparison
np.save('reference_mel.npy', features)
print("\nSaved to reference_mel.npy")
