#!/usr/bin/env python3
"""Run full Parakeet TDT encoder via PyTorch using safetensors weights.
Compare encoder output with expected values.
"""
import torch
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open
import soundfile as sf

MODEL_DIR = 'models/parakeet-tdt-0.6b-v3'
AUDIO_PATH = 'tests/fixtures/test_speech_en_16k.wav'

# Load audio
audio, sr = sf.read(AUDIO_PATH)
audio = audio.astype(np.float32)
print(f"Audio: {len(audio)} samples, sr={sr}")

# Load weights
f = safe_open(f'{MODEL_DIR}/model.safetensors', framework='pt')

# ============= MEL EXTRACTION (NeMo style) =============
# Pre-emphasis
preemph = 0.97
audio_pe = np.zeros_like(audio)
audio_pe[0] = audio[0]
for i in range(1, len(audio)):
    audio_pe[i] = audio[i] - preemph * audio[i-1]

# STFT with center=True (pad n_fft//2)
n_fft = 512
hop_length = 160
win_length = 400

audio_t = torch.from_numpy(audio_pe).unsqueeze(0)  # [1, T]
window = torch.hann_window(win_length)
# Center padding with n_fft//2
pad = n_fft // 2
audio_padded = F.pad(audio_t, (pad, pad))
stft = torch.stft(audio_padded, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                   window=window, center=False, return_complex=True)
# stft: [1, n_fft/2+1, T_frames]
mag = stft.abs()
power = mag ** 2  # mag_power=2.0

# Mel filterbank from model weights
mel_fb = f.get_tensor('preprocessor.featurizer.fb')  # [1, 128, 257]
mel_fb = mel_fb.squeeze(0)  # [128, 257]

mel = torch.matmul(mel_fb, power.squeeze(0))  # [128, T_frames]

# Log
guard = 2**-24
mel_log = torch.log(mel + guard)

# Per-feature normalization
mean = mel_log.mean(dim=1, keepdim=True)
std = mel_log.std(dim=1, keepdim=True).clamp(min=1e-5)
mel_norm = (mel_log - mean) / std

mel_norm = mel_norm.unsqueeze(0)  # [1, 128, T]
print(f"Mel: shape={mel_norm.shape}, range=[{mel_norm.min():.4f}, {mel_norm.max():.4f}]")

# ============= SUBSAMPLING =============
# NeMo convention: [B, D, T] -> transpose -> [B, T, D] -> unsqueeze -> [B, 1, T, D]
x = mel_norm.permute(0, 2, 1).unsqueeze(1)  # [1, 1, T, D=128]
print(f"Sub input: {x.shape}")

# Conv2d stage 0: (1->256, 3x3, stride=2, pad=1)
conv0_w = f.get_tensor('encoder.pre_encode.conv.0.weight')
conv0_b = f.get_tensor('encoder.pre_encode.conv.0.bias')
x = F.conv2d(x, conv0_w, conv0_b, stride=2, padding=1)
x = F.relu(x)
print(f"After stage0: {x.shape}, [{x.min():.4f}, {x.max():.4f}]")

# Conv2d stage 1: dw+pw
conv2_w = f.get_tensor('encoder.pre_encode.conv.2.weight')
conv2_b = f.get_tensor('encoder.pre_encode.conv.2.bias')
conv3_w = f.get_tensor('encoder.pre_encode.conv.3.weight')
conv3_b = f.get_tensor('encoder.pre_encode.conv.3.bias')
x = F.conv2d(x, conv2_w, conv2_b, stride=2, padding=1, groups=256)
x = F.conv2d(x, conv3_w, conv3_b)
x = F.relu(x)
print(f"After stage1: {x.shape}, [{x.min():.4f}, {x.max():.4f}]")

# Conv2d stage 2: dw+pw
conv5_w = f.get_tensor('encoder.pre_encode.conv.5.weight')
conv5_b = f.get_tensor('encoder.pre_encode.conv.5.bias')
conv6_w = f.get_tensor('encoder.pre_encode.conv.6.weight')
conv6_b = f.get_tensor('encoder.pre_encode.conv.6.bias')
x = F.conv2d(x, conv5_w, conv5_b, stride=2, padding=1, groups=256)
x = F.conv2d(x, conv6_w, conv6_b)
print(f"After stage2: {x.shape}, [{x.min():.4f}, {x.max():.4f}]")

# Flatten: [B, C, T/8, D/8] -> [B, T/8, C*D/8]
B, C, T, D = x.shape
x = x.permute(0, 2, 1, 3)  # [B, T, C, D]
x = x.reshape(B, T, C * D)
print(f"Pre-proj: {x.shape}, [{x.min():.4f}, {x.max():.4f}]")

# Linear projection
proj_w = f.get_tensor('encoder.pre_encode.out.weight')
proj_b = f.get_tensor('encoder.pre_encode.out.bias')
x = F.linear(x, proj_w, proj_b)
print(f"After proj: {x.shape}, [{x.min():.4f}, {x.max():.4f}]")

# xscale
xscale = np.sqrt(1024.0)
x = x * xscale
print(f"After xscale: [{x.min():.4f}, {x.max():.4f}]")

# ============= CONFORMER LAYERS =============
def load_layer_norm(prefix):
    w = f.get_tensor(f'{prefix}.weight')
    b = f.get_tensor(f'{prefix}.bias')
    return w, b

def apply_layer_norm(x, w, b, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm * w + b

def load_linear_no_bias(prefix):
    return f.get_tensor(f'{prefix}.weight')

def load_linear_with_bias(prefix):
    return f.get_tensor(f'{prefix}.weight'), f.get_tensor(f'{prefix}.bias')

# Positional encoding
def create_pe(t, d_model):
    pe_len = 2 * t - 1
    pe = torch.zeros(pe_len, d_model)
    for pos_idx in range(pe_len):
        pos = pos_idx - (t - 1)
        for i in range(d_model // 2):
            freq = 1.0 / (10000.0 ** (2 * i / d_model))
            angle = pos * freq
            pe[pos_idx, 2*i] = np.sin(angle)
            pe[pos_idx, 2*i+1] = np.cos(angle)
    return pe.unsqueeze(0)  # [1, 2T-1, D]

T_enc = x.shape[1]
pos_emb = create_pe(T_enc, 1024)

def rel_shift(x):
    """NeMo-style rel_shift."""
    B, H, T, PE = x.shape
    # Pad left
    x_padded = F.pad(x, (1, 0))  # [B, H, T, PE+1]
    # Reshape
    x_reshaped = x_padded.reshape(B, H, PE+1, T)
    # Drop first row
    x_sliced = x_reshaped[:, :, 1:, :]  # [B, H, PE, T]
    # Reshape back
    x_shifted = x_sliced.reshape(B, H, T, PE)
    # Take first T columns
    t = PE // 2 + 1
    return x_shifted[:, :, :, :t]

for layer_idx in range(24):
    prefix = f'encoder.layers.{layer_idx}'
    
    # FF1
    ff1_ln_w, ff1_ln_b = load_layer_norm(f'{prefix}.norm_feed_forward1')
    ff1_w1 = load_linear_no_bias(f'{prefix}.feed_forward1.linear1')
    ff1_w2 = load_linear_no_bias(f'{prefix}.feed_forward1.linear2')
    
    residual = x
    x_norm = apply_layer_norm(x, ff1_ln_w, ff1_ln_b)
    ff1_out = F.silu(F.linear(x_norm, ff1_w1))
    ff1_out = F.linear(ff1_out, ff1_w2)
    x = residual + 0.5 * ff1_out
    
    # Self-Attention
    attn_ln_w, attn_ln_b = load_layer_norm(f'{prefix}.norm_self_att')
    q_w = load_linear_no_bias(f'{prefix}.self_attn.linear_q')
    k_w = load_linear_no_bias(f'{prefix}.self_attn.linear_k')
    v_w = load_linear_no_bias(f'{prefix}.self_attn.linear_v')
    out_w = load_linear_no_bias(f'{prefix}.self_attn.linear_out')
    pos_w = load_linear_no_bias(f'{prefix}.self_attn.linear_pos')
    bias_u = f.get_tensor(f'{prefix}.self_attn.pos_bias_u')  # [8, 128]
    bias_v = f.get_tensor(f'{prefix}.self_attn.pos_bias_v')  # [8, 128]
    
    residual = x
    x_norm = apply_layer_norm(x, attn_ln_w, attn_ln_b)
    
    B_a, T_a, D_a = x_norm.shape
    H, dk = 8, 128
    
    q = F.linear(x_norm, q_w).reshape(B_a, T_a, H, dk).permute(0, 2, 1, 3)
    k = F.linear(x_norm, k_w).reshape(B_a, T_a, H, dk).permute(0, 2, 1, 3)
    v = F.linear(x_norm, v_w).reshape(B_a, T_a, H, dk).permute(0, 2, 1, 3)
    
    pe_len_a = pos_emb.shape[1]
    p = F.linear(pos_emb, pos_w).reshape(1, pe_len_a, H, dk).permute(0, 2, 1, 3)
    
    # Content score
    q_u = q + bias_u.reshape(1, H, 1, dk)
    content_score = torch.matmul(q_u, k.transpose(-2, -1))
    
    # Position score
    q_v = q + bias_v.reshape(1, H, 1, dk)
    pos_score_full = torch.matmul(q_v, p.transpose(-2, -1))
    pos_score = rel_shift(pos_score_full)
    
    scores = (content_score + pos_score) / np.sqrt(dk)
    attn = F.softmax(scores, dim=-1)
    context = torch.matmul(attn, v)
    context = context.permute(0, 2, 1, 3).reshape(B_a, T_a, H * dk)
    attn_out = F.linear(context, out_w)
    
    x = residual + attn_out
    
    # Conv
    conv_ln_w, conv_ln_b = load_layer_norm(f'{prefix}.norm_conv')
    pw1_w = f.get_tensor(f'{prefix}.conv.pointwise_conv1.weight')
    dw_w = f.get_tensor(f'{prefix}.conv.depthwise_conv.weight')
    bn_w = f.get_tensor(f'{prefix}.conv.batch_norm.weight')
    bn_b = f.get_tensor(f'{prefix}.conv.batch_norm.bias')
    bn_rm = f.get_tensor(f'{prefix}.conv.batch_norm.running_mean')
    bn_rv = f.get_tensor(f'{prefix}.conv.batch_norm.running_var')
    pw2_w = f.get_tensor(f'{prefix}.conv.pointwise_conv2.weight')
    
    residual = x
    x_norm = apply_layer_norm(x, conv_ln_w, conv_ln_b)
    
    # Conv: [B, T, D] -> [B, D, T]
    x_conv = x_norm.permute(0, 2, 1)
    x_conv = F.conv1d(x_conv, pw1_w)  # [B, 2D, T]
    # GLU
    a, b_gate = x_conv.chunk(2, dim=1)
    x_conv = a * torch.sigmoid(b_gate)
    # Depthwise
    pad_conv = 9 // 2
    x_conv = F.conv1d(x_conv, dw_w, padding=pad_conv, groups=1024)
    # BatchNorm1d (inference)
    bn_std = torch.sqrt(bn_rv + 1e-5)
    x_conv = (x_conv - bn_rm.reshape(1, -1, 1)) / bn_std.reshape(1, -1, 1) * bn_w.reshape(1, -1, 1) + bn_b.reshape(1, -1, 1)
    # SiLU
    x_conv = F.silu(x_conv)
    # Pointwise 2
    x_conv = F.conv1d(x_conv, pw2_w)
    # Back to [B, T, D]
    conv_out = x_conv.permute(0, 2, 1)
    
    x = residual + conv_out
    
    # FF2
    ff2_ln_w, ff2_ln_b = load_layer_norm(f'{prefix}.norm_feed_forward2')
    ff2_w1 = load_linear_no_bias(f'{prefix}.feed_forward2.linear1')
    ff2_w2 = load_linear_no_bias(f'{prefix}.feed_forward2.linear2')
    
    residual = x
    x_norm = apply_layer_norm(x, ff2_ln_w, ff2_ln_b)
    ff2_out = F.silu(F.linear(x_norm, ff2_w1))
    ff2_out = F.linear(ff2_out, ff2_w2)
    x = residual + 0.5 * ff2_out
    
    # Final norm
    out_ln_w, out_ln_b = load_layer_norm(f'{prefix}.norm_out')
    x = apply_layer_norm(x, out_ln_w, out_ln_b)
    
    if layer_idx % 6 == 5 or layer_idx == 23:
        print(f"Layer {layer_idx}: [{x.min():.4f}, {x.max():.4f}], std={x.std():.6f}")

print(f"\nFinal encoder output: shape={x.shape}")
print(f"  range: [{x.min():.6f}, {x.max():.6f}]")
print(f"  mean: {x.mean():.6f}, std: {x.std():.6f}")

# Now run through joint network with initial LSTM state
# LSTM state = zeros, input = blank (zero embedding)
pred_out = torch.zeros(640)

# joint
enc_proj_w, enc_proj_b = load_linear_with_bias('joint.enc')
pred_proj_w, pred_proj_b = load_linear_with_bias('joint.pred')
out_joint_w, out_joint_b = load_linear_with_bias('joint.joint_net.2')

# For first frame
enc_frame = x[0, 0]  # [1024]
enc_h = F.linear(enc_frame, enc_proj_w, enc_proj_b)
pred_h = F.linear(pred_out, pred_proj_w, pred_proj_b)
joint_h = F.relu(enc_h + pred_h)
logits = F.linear(joint_h, out_joint_w, out_joint_b)

token_logits = logits[:8193]
dur_logits = logits[8193:]

print(f"\nJoint (frame 0):")
print(f"  enc_h: [{enc_h.min():.4f}, {enc_h.max():.4f}]")
print(f"  joint_h: [{joint_h.min():.4f}, {joint_h.max():.4f}], nonzero={joint_h.count_nonzero()}/{len(joint_h)}")
top5 = torch.topk(token_logits, 5)
print(f"  top5 tokens: {list(zip(top5.indices.tolist(), [f'{v:.4f}' for v in top5.values.tolist()]))}")
print(f"  blank (8192): {token_logits[8192]:.4f}")
