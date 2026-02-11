#!/usr/bin/env python3
"""Compare Rust mel output with Python reference."""

import numpy as np
import soundfile as sf
from transformers import WhisperFeatureExtractor

# Load audio - first 2 seconds (same as Python reference)
audio, sr = sf.read("test_30sec.wav")
if len(audio.shape) > 1:
    audio = audio[:, 0]
audio = audio[:sr * 2].astype(np.float32)

print(f"Audio: {len(audio)} samples")

# Get Python mel
fe = WhisperFeatureExtractor.from_pretrained("models/qwen3-asr-0.6b")
mel_result = fe(audio, return_tensors="np", sampling_rate=sr, padding=False)
mel_python = mel_result["input_features"][0]  # [n_mels, time]

print(f"\n=== Python Mel Spectrogram ===")
print(f"Shape: {mel_python.shape}")
print(f"Range: [{mel_python.min():.4f}, {mel_python.max():.4f}]")
print(f"Mean: {mel_python.mean():.4f}")
print(f"\nExpected values (for Rust comparison):")
print(f"Mel[0, :5] = {mel_python[0, :5]}")
print(f"Mel[64, :5] = {mel_python[64, :5]}")
print(f"Mel[127, :5] = {mel_python[127, :5]}")

# Save for later
np.save("reference_mel_2sec.npy", mel_python)
print("\nSaved to reference_mel_2sec.npy")
