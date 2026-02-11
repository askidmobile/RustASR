#!/usr/bin/env python3
"""Debug: сравнить subsampling Rust vs PyTorch."""

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from safetensors import safe_open

# 1. Load audio
audio, sr = sf.read('tests/fixtures/test_speech_en_16k.wav')
audio = audio.astype(np.float32)
print(f"Audio: {len(audio)} samples, {sr}Hz")

# 2. Load mel filters and window from model
f = safe_open('models/parakeet-tdt-0.6b-v3/model.safetensors', framework='pt')
mel_fb = f.get_tensor('preprocessor.featurizer.fb')  # [1, 128, 257]
window = f.get_tensor('preprocessor.featurizer.window')  # [400]

# 3. Compute mel (matching our Rust implementation)
n_fft = 512
hop_length = 160
win_length = 400
n_mels = 128
preemph = 0.97

# Pre-emphasis
audio_pe = np.zeros_like(audio)
audio_pe[0] = audio[0]
for i in range(1, len(audio)):
    audio_pe[i] = audio[i] - preemph * audio[i-1]

# STFT with center padding
audio_t = torch.from_numpy(audio_pe)
stft = torch.stft(
    audio_t, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
    window=window, center=True, return_complex=True
)
power = stft.abs().pow(2)  # [n_fft/2+1, T]
print(f"STFT power: {power.shape}")

# Mel filterbank
mel_fb_2d = mel_fb.squeeze(0)  # [128, 257]
mel = torch.matmul(mel_fb_2d, power)  # [128, T]
print(f"Mel: {mel.shape}, [{mel.min():.4f}, {mel.max():.4f}]")

# Log
guard = 2**-24
log_mel = torch.log(mel + guard)
print(f"Log mel: [{log_mel.min():.4f}, {log_mel.max():.4f}]")

# Per-feature normalization
mean = log_mel.mean(dim=1, keepdim=True)
std = log_mel.std(dim=1, keepdim=True).clamp(min=1e-5)
norm_mel = (log_mel - mean) / std
print(f"Normalized mel: [{norm_mel.min():.4f}, {norm_mel.max():.4f}], mean={norm_mel.mean():.6f}")

# 4. Run subsampling in PyTorch
conv0_w = f.get_tensor('encoder.pre_encode.conv.0.weight')
conv0_b = f.get_tensor('encoder.pre_encode.conv.0.bias')
conv2_w = f.get_tensor('encoder.pre_encode.conv.2.weight')
conv2_b = f.get_tensor('encoder.pre_encode.conv.2.bias')
conv3_w = f.get_tensor('encoder.pre_encode.conv.3.weight')
conv3_b = f.get_tensor('encoder.pre_encode.conv.3.bias')
conv5_w = f.get_tensor('encoder.pre_encode.conv.5.weight')
conv5_b = f.get_tensor('encoder.pre_encode.conv.5.bias')
conv6_w = f.get_tensor('encoder.pre_encode.conv.6.weight')
conv6_b = f.get_tensor('encoder.pre_encode.conv.6.bias')
out_w = f.get_tensor('encoder.pre_encode.out.weight')
out_b = f.get_tensor('encoder.pre_encode.out.bias')

# NeMo convention: mel [D, T] → transpose → [T, D] → unsqueeze → [1, 1, T, D]
x = norm_mel.unsqueeze(0)  # [1, 128, T]
x = x.permute(0, 2, 1).unsqueeze(1)  # [1, 1, T, D]
print(f"\nSub input: {x.shape}, [{x.min():.4f}, {x.max():.4f}]")

# Stage 0
x = F.conv2d(x, conv0_w, conv0_b, stride=2, padding=1)
x = F.relu(x)
print(f"After stage0: {x.shape}, [{x.min():.4f}, {x.max():.4f}]")

# Stage 1
x = F.conv2d(x, conv2_w, conv2_b, stride=2, padding=1, groups=256)
x = F.conv2d(x, conv3_w, conv3_b, stride=1, padding=0)
x = F.relu(x)
print(f"After stage1: {x.shape}, [{x.min():.4f}, {x.max():.4f}]")

# Stage 2
x = F.conv2d(x, conv5_w, conv5_b, stride=2, padding=1, groups=256)
x = F.conv2d(x, conv6_w, conv6_b, stride=1, padding=0)
print(f"After stage2: {x.shape}, [{x.min():.4f}, {x.max():.4f}]")

# NeMo reshape: [B, C, T, D] → transpose(1,2) → [B, T, C, D] → reshape [B, T, C*D]
b_s, c_s, t_s, d_s = x.shape
x = x.transpose(1, 2).reshape(b_s, t_s, c_s * d_s)
print(f"Pre-proj: {x.shape}, [{x.min():.4f}, {x.max():.4f}]")

# Linear projection
x = F.linear(x, out_w, out_b)
print(f"After proj: {x.shape}, [{x.min():.4f}, {x.max():.4f}]")
