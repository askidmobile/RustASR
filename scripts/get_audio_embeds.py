#!/usr/bin/env python3
"""Get reference audio embeddings from audio_tower."""

import numpy as np
import soundfile as sf
import torch

# Load audio
audio, sr = sf.read("test_30sec.wav")
if len(audio.shape) > 1:
    audio = audio[:, 0]
audio = audio[:sr * 2].astype(np.float32)

print(f"Audio: {len(audio)} samples")

# Get mel
from transformers import WhisperFeatureExtractor
fe = WhisperFeatureExtractor.from_pretrained("models/qwen3-asr-0.6b")
mel_result = fe(audio, return_tensors="pt", sampling_rate=sr, padding=False)
mel = mel_result["input_features"]

print(f"Mel: shape={mel.shape}, range=[{mel.min():.4f}, {mel.max():.4f}], mean={mel.mean():.4f}")

# Load model and get audio_tower
from qwen_asr import Qwen3ASRModel
model = Qwen3ASRModel.from_pretrained("models/qwen3-asr-0.6b")
audio_tower = model.model.thinker.audio_tower
print(f"Audio tower: {type(audio_tower)}")

# Forward pass
with torch.no_grad():
    device = next(audio_tower.parameters()).device
    dtype = next(audio_tower.parameters()).dtype
    mel_input = mel.to(device=device, dtype=dtype)
    
    time_len = mel_input.shape[2]
    feature_lens = torch.tensor([time_len], device=device)
    aftercnn_lens = torch.tensor([time_len // 8], device=device)
    
    print(f"\nInput: mel={mel_input.shape}, feature_lens={feature_lens}, aftercnn_lens={aftercnn_lens}")
    
    result = audio_tower(mel_input, feature_lens, aftercnn_lens)
    audio_embeds = result[0] if isinstance(result, tuple) else result
    
    print(f"\n=== Reference Audio Embeddings ===")
    print(f"Shape: {audio_embeds.shape}")
    print(f"Range: [{audio_embeds.min().item():.4f}, {audio_embeds.max().item():.4f}]")
    print(f"Mean: {audio_embeds.mean().item():.4f}")
    print(f"Std: {audio_embeds.std().item():.4f}")
    print(f"\nFirst 10 values [0,:10]:")
    print(audio_embeds[0, :10].float().cpu().numpy())
    
    np.save("reference_audio_embeds.npy", audio_embeds.float().cpu().numpy())
    print("\nSaved to reference_audio_embeds.npy")
