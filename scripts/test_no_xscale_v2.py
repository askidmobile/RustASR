#!/usr/bin/env python3
"""Full pipeline: encoder output + LSTM + joint - compare with Rust."""
import torch
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open
import soundfile as sf

f = safe_open('models/parakeet-tdt-0.6b-v3/model.safetensors', framework='pt')

# Load audio
audio, sr = sf.read('tests/fixtures/test_speech_en_16k.wav')
audio = audio.astype(np.float32)

# ===== MEL =====
preemph = 0.97
audio_pe = np.zeros_like(audio)
audio_pe[0] = audio[0]
for i in range(1, len(audio)):
    audio_pe[i] = audio[i] - preemph * audio[i-1]

n_fft, hop_length, win_length = 512, 160, 400
audio_t = torch.from_numpy(audio_pe).unsqueeze(0)
window = torch.hann_window(win_length)
pad = n_fft // 2
audio_padded = F.pad(audio_t, (pad, pad))
stft = torch.stft(audio_padded, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                   window=window, center=False, return_complex=True)
power = stft.abs() ** 2
mel_fb = f.get_tensor('preprocessor.featurizer.fb').squeeze(0)
mel = torch.matmul(mel_fb, power.squeeze(0))
guard = 2**-24
mel_log = torch.log(mel + guard)
mean = mel_log.mean(dim=1, keepdim=True)
std = mel_log.std(dim=1, keepdim=True).clamp(min=1e-5)
mel_norm = (mel_log - mean) / std
mel_norm = mel_norm.unsqueeze(0)

# ===== SUBSAMPLING =====
x = mel_norm.permute(0, 2, 1).unsqueeze(1)
for stage in [(0, 2, 1, 256, 1), (2, 2, 1, 256, 256), (3, 1, 0, 1, 1), (5, 2, 1, 256, 256), (6, 1, 0, 1, 1)]:
    idx, stride, padding, _, groups = stage
    w = f.get_tensor(f'encoder.pre_encode.conv.{idx}.weight')
    b = f.get_tensor(f'encoder.pre_encode.conv.{idx}.bias')
    x = F.conv2d(x, w, b, stride=stride, padding=padding, groups=groups)
    if idx in (0, 3):
        x = F.relu(x)

B, C, T, D = x.shape
x = x.permute(0, 2, 1, 3).reshape(B, T, C * D)
proj_w = f.get_tensor('encoder.pre_encode.out.weight')
proj_b = f.get_tensor('encoder.pre_encode.out.bias')
x = F.linear(x, proj_w, proj_b)
# x = x * np.sqrt(1024.0)  # DISABLED: xscaling=False in config

# ===== CONFORMER LAYERS =====
def ln(x, prefix):
    w, b = f.get_tensor(f'{prefix}.weight'), f.get_tensor(f'{prefix}.bias')
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + 1e-5) * w + b

def create_pe(t, d_model):
    pe_len = 2 * t - 1
    pe = torch.zeros(pe_len, d_model)
    for pos_idx in range(pe_len):
        pos = pos_idx - (t - 1)
        for i in range(d_model // 2):
            freq = 1.0 / (10000.0 ** (2 * i / d_model))
            pe[pos_idx, 2*i] = np.sin(pos * freq)
            pe[pos_idx, 2*i+1] = np.cos(pos * freq)
    return pe.unsqueeze(0)

def rel_shift(x):
    B, H, T, PE = x.shape
    x_padded = F.pad(x, (1, 0))
    x_reshaped = x_padded.reshape(B, H, PE+1, T)
    x_sliced = x_reshaped[:, :, 1:, :]
    x_shifted = x_sliced.reshape(B, H, T, PE)
    return x_shifted[:, :, :, :PE//2+1]

T_enc = x.shape[1]
pos_emb = create_pe(T_enc, 1024)

for layer_idx in range(24):
    p = f'encoder.layers.{layer_idx}'

    # FF1
    r = x
    x_n = ln(x, f'{p}.norm_feed_forward1')
    x_n = F.linear(F.silu(F.linear(x_n, f.get_tensor(f'{p}.feed_forward1.linear1.weight'))),
                    f.get_tensor(f'{p}.feed_forward1.linear2.weight'))
    x = r + 0.5 * x_n

    # MHSA
    r = x
    x_n = ln(x, f'{p}.norm_self_att')
    B_a, T_a, D_a = x_n.shape
    H, dk = 8, 128
    q = F.linear(x_n, f.get_tensor(f'{p}.self_attn.linear_q.weight')).reshape(B_a, T_a, H, dk).permute(0, 2, 1, 3)
    k = F.linear(x_n, f.get_tensor(f'{p}.self_attn.linear_k.weight')).reshape(B_a, T_a, H, dk).permute(0, 2, 1, 3)
    v = F.linear(x_n, f.get_tensor(f'{p}.self_attn.linear_v.weight')).reshape(B_a, T_a, H, dk).permute(0, 2, 1, 3)
    pe_len_a = pos_emb.shape[1]
    pp = F.linear(pos_emb, f.get_tensor(f'{p}.self_attn.linear_pos.weight')).reshape(1, pe_len_a, H, dk).permute(0, 2, 1, 3)
    bu = f.get_tensor(f'{p}.self_attn.pos_bias_u').reshape(1, H, 1, dk)
    bv = f.get_tensor(f'{p}.self_attn.pos_bias_v').reshape(1, H, 1, dk)
    cs = (q + bu) @ k.transpose(-2, -1)
    ps = rel_shift((q + bv) @ pp.transpose(-2, -1))
    scores = (cs + ps) / np.sqrt(dk)
    attn = F.softmax(scores, dim=-1)
    ctx = (attn @ v).permute(0, 2, 1, 3).reshape(B_a, T_a, H * dk)
    x = r + F.linear(ctx, f.get_tensor(f'{p}.self_attn.linear_out.weight'))

    # Conv
    r = x
    x_n = ln(x, f'{p}.norm_conv').permute(0, 2, 1)
    x_n = F.conv1d(x_n, f.get_tensor(f'{p}.conv.pointwise_conv1.weight'))
    a, b = x_n.chunk(2, dim=1)
    x_n = a * torch.sigmoid(b)
    x_n = F.conv1d(x_n, f.get_tensor(f'{p}.conv.depthwise_conv.weight'), padding=4, groups=1024)
    bn_rm = f.get_tensor(f'{p}.conv.batch_norm.running_mean')
    bn_rv = f.get_tensor(f'{p}.conv.batch_norm.running_var')
    bn_w = f.get_tensor(f'{p}.conv.batch_norm.weight')
    bn_b = f.get_tensor(f'{p}.conv.batch_norm.bias')
    x_n = (x_n - bn_rm.reshape(1,-1,1)) / (bn_rv + 1e-5).sqrt().reshape(1,-1,1) * bn_w.reshape(1,-1,1) + bn_b.reshape(1,-1,1)
    x_n = F.silu(x_n)
    x_n = F.conv1d(x_n, f.get_tensor(f'{p}.conv.pointwise_conv2.weight'))
    x = r + x_n.permute(0, 2, 1)

    # FF2
    r = x
    x_n = ln(x, f'{p}.norm_feed_forward2')
    x_n = F.linear(F.silu(F.linear(x_n, f.get_tensor(f'{p}.feed_forward2.linear1.weight'))),
                    f.get_tensor(f'{p}.feed_forward2.linear2.weight'))
    x = r + 0.5 * x_n

    # norm_out
    x = ln(x, f'{p}.norm_out')

enc_output = x  # [1, 34, 1024]
print(f"Encoder output: [{enc_output.min():.6f}, {enc_output.max():.6f}], std={enc_output.std():.6f}")

# ===== LSTM PREDICTION NET (initial step with blank) =====
def lstm_step(inp, h, c, layer_idx):
    wih = f.get_tensor(f'decoder.prediction.dec_rnn.lstm.weight_ih_l{layer_idx}')
    whh = f.get_tensor(f'decoder.prediction.dec_rnn.lstm.weight_hh_l{layer_idx}')
    bih = f.get_tensor(f'decoder.prediction.dec_rnn.lstm.bias_ih_l{layer_idx}')
    bhh = f.get_tensor(f'decoder.prediction.dec_rnn.lstm.bias_hh_l{layer_idx}')
    gates = inp @ wih.t() + bih + h @ whh.t() + bhh
    hs = 640
    i_g = torch.sigmoid(gates[:hs])
    f_g = torch.sigmoid(gates[hs:2*hs])
    g_g = torch.tanh(gates[2*hs:3*hs])
    o_g = torch.sigmoid(gates[3*hs:])
    c_new = f_g * c + i_g * g_g
    h_new = o_g * torch.tanh(c_new)
    return h_new, c_new

embed_blank = torch.zeros(640)  # blank embedding is all zeros
h0, c0 = torch.zeros(640), torch.zeros(640)
h1, c1 = torch.zeros(640), torch.zeros(640)

h0_new, c0_new = lstm_step(embed_blank, h0, c0, 0)
h1_new, c1_new = lstm_step(h0_new, h1, c1, 1)
pred_out = h1_new
print(f"LSTM pred_out: [{pred_out.min():.6f}, {pred_out.max():.6f}], norm={pred_out.norm():.4f}")

# ===== JOINT NETWORK =====
enc_w, enc_b = f.get_tensor('joint.enc.weight'), f.get_tensor('joint.enc.bias')
pred_w, pred_b = f.get_tensor('joint.pred.weight'), f.get_tensor('joint.pred.bias')
out_w, out_b = f.get_tensor('joint.joint_net.2.weight'), f.get_tensor('joint.joint_net.2.bias')

# Process first 5 frames
for frame in range(5):
    enc_frame = enc_output[0, frame]  # [1024]
    enc_h = F.linear(enc_frame, enc_w, enc_b)
    pred_h = F.linear(pred_out, pred_w, pred_b)
    joint_h = F.relu(enc_h + pred_h)
    logits = F.linear(joint_h, out_w, out_b)
    
    token_logits = logits[:8193]
    dur_logits = logits[8193:]
    
    top5 = torch.topk(token_logits, 5)
    best_idx = top5.indices[0].item()
    blank_val = token_logits[8192].item()
    
    print(f"\nFrame {frame}:")
    print(f"  enc_h: [{enc_h.min():.4f}, {enc_h.max():.4f}]")
    print(f"  pred_h: [{pred_h.min():.4f}, {pred_h.max():.4f}]")
    print(f"  joint_h: [{joint_h.min():.4f}, {joint_h.max():.4f}], nz={joint_h.count_nonzero()}/{len(joint_h)}")
    print(f"  top5: {list(zip(top5.indices.tolist(), [f'{v:.4f}' for v in top5.values.tolist()]))}")
    print(f"  blank={blank_val:.4f}, dur_logits={dur_logits.tolist()}")
