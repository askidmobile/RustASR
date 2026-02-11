#!/usr/bin/env python3
"""Compare mel with correct NeMo params: reflect padding, n_fft//2 pad."""
import torch
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open
import soundfile as sf

f = safe_open('models/parakeet-tdt-0.6b-v3/model.safetensors', framework='pt')

audio, sr = sf.read('tests/fixtures/test_real_speech_5s.wav')
audio = audio.astype(np.float32)
print(f"Audio: {len(audio)} samples, sr={sr}")

# Pre-emphasis
preemph = 0.97
audio_pe = np.zeros_like(audio)
audio_pe[0] = audio[0]
for i in range(1, len(audio)):
    audio_pe[i] = audio[i] - preemph * audio[i-1]

n_fft, hop_length, win_length = 512, 160, 400
audio_t = torch.from_numpy(audio_pe).unsqueeze(0)
window = torch.hann_window(win_length)

# Method 1: Zero padding with win_length//2 (our Rust code)
pad1 = win_length // 2  # 200
audio_padded1 = F.pad(audio_t, (pad1, pad1), mode='constant', value=0)
stft1 = torch.stft(audio_padded1, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    window=window, center=False, return_complex=True)
print(f"Method 1 (zero pad, win//2={pad1}): stft shape={stft1.shape}")

# Method 2: Zero padding with n_fft//2 (my Python script)
pad2 = n_fft // 2  # 256
audio_padded2 = F.pad(audio_t, (pad2, pad2), mode='constant', value=0)
stft2 = torch.stft(audio_padded2, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    window=window, center=False, return_complex=True)
print(f"Method 2 (zero pad, n_fft//2={pad2}): stft shape={stft2.shape}")

# Method 3: Reflect padding with n_fft//2 (NeMo default: center=True)
stft3 = torch.stft(audio_t, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    window=window, center=True, return_complex=True)
print(f"Method 3 (reflect pad, center=True): stft shape={stft3.shape}")

# Compare mel spectrograms
mel_fb = f.get_tensor('preprocessor.featurizer.fb').squeeze(0)
guard = 2**-24

def compute_mel(stft_result, label):
    power = stft_result.abs() ** 2
    mel = torch.matmul(mel_fb, power.squeeze(0))
    mel_log = torch.log(mel + guard)
    mean = mel_log.mean(dim=1, keepdim=True)
    std = mel_log.std(dim=1, keepdim=True).clamp(min=1e-5)
    mel_norm = (mel_log - mean) / std
    print(f"  {label}: mel shape={mel_norm.shape}, range=[{mel_norm.min():.4f}, {mel_norm.max():.4f}]")
    return mel_norm.unsqueeze(0)

mel1 = compute_mel(stft1, "zero+win//2")
mel2 = compute_mel(stft2, "zero+n_fft//2")
mel3 = compute_mel(stft3, "reflect+center")

# Compare frame-by-frame
t_min = min(mel1.shape[2], mel2.shape[2], mel3.shape[2])
diff12 = (mel1[:,:,:t_min] - mel2[:,:,:t_min]).abs().mean().item()
diff13 = (mel1[:,:,:t_min] - mel3[:,:,:t_min]).abs().mean().item()
diff23 = (mel2[:,:,:t_min] - mel3[:,:,:t_min]).abs().mean().item()
print(f"\nMean abs difference:")
print(f"  zero+win//2 vs zero+n_fft//2: {diff12:.6f}")
print(f"  zero+win//2 vs reflect+center: {diff13:.6f}")
print(f"  zero+n_fft//2 vs reflect+center: {diff23:.6f}")

# Run full pipeline with reflect padding (NeMo-correct)
print("\n=== Full pipeline with NeMo-correct mel (reflect padding) ===")
mel_correct = mel3  # [1, 128, T]
x = mel_correct.permute(0, 2, 1).unsqueeze(1)

# Subsampling
for idx, stride, padding, groups in [(0, 2, 1, 1), (2, 2, 1, 256), (3, 1, 0, 1), (5, 2, 1, 256), (6, 1, 0, 1)]:
    w = f.get_tensor(f'encoder.pre_encode.conv.{idx}.weight')
    b = f.get_tensor(f'encoder.pre_encode.conv.{idx}.bias')
    x = F.conv2d(x, w, b, stride=stride, padding=padding, groups=groups)
    if idx in (0, 3):
        x = F.relu(x)

B, C, T, D = x.shape
x = x.permute(0, 2, 1, 3).reshape(B, T, C * D)
x = F.linear(x, f.get_tensor('encoder.pre_encode.out.weight'), f.get_tensor('encoder.pre_encode.out.bias'))
x = x * np.sqrt(1024.0)

# Conformer (abbreviated - just run all 24 layers)
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
    x = F.pad(x, (1, 0)).reshape(B, H, PE+1, T)[:, :, 1:, :].reshape(B, H, T, PE)
    return x[:, :, :, :PE//2+1]

T_enc = x.shape[1]
pos_emb = create_pe(T_enc, 1024)

for layer_idx in range(24):
    p = f'encoder.layers.{layer_idx}'
    # FF1
    r = x; x_n = ln(x, f'{p}.norm_feed_forward1')
    x_n = F.linear(F.silu(F.linear(x_n, f.get_tensor(f'{p}.feed_forward1.linear1.weight'))), f.get_tensor(f'{p}.feed_forward1.linear2.weight'))
    x = r + 0.5 * x_n
    # MHSA
    r = x; x_n = ln(x, f'{p}.norm_self_att')
    B_a, T_a, D_a = x_n.shape; H, dk = 8, 128
    q = F.linear(x_n, f.get_tensor(f'{p}.self_attn.linear_q.weight')).reshape(B_a,T_a,H,dk).permute(0,2,1,3)
    k = F.linear(x_n, f.get_tensor(f'{p}.self_attn.linear_k.weight')).reshape(B_a,T_a,H,dk).permute(0,2,1,3)
    v = F.linear(x_n, f.get_tensor(f'{p}.self_attn.linear_v.weight')).reshape(B_a,T_a,H,dk).permute(0,2,1,3)
    pe_len_a = pos_emb.shape[1]
    pp = F.linear(pos_emb, f.get_tensor(f'{p}.self_attn.linear_pos.weight')).reshape(1,pe_len_a,H,dk).permute(0,2,1,3)
    bu = f.get_tensor(f'{p}.self_attn.pos_bias_u').reshape(1,H,1,dk)
    bv = f.get_tensor(f'{p}.self_attn.pos_bias_v').reshape(1,H,1,dk)
    cs = (q+bu) @ k.transpose(-2,-1)
    ps = rel_shift((q+bv) @ pp.transpose(-2,-1))
    attn = F.softmax((cs+ps)/np.sqrt(dk), dim=-1)
    ctx = (attn @ v).permute(0,2,1,3).reshape(B_a,T_a,H*dk)
    x = r + F.linear(ctx, f.get_tensor(f'{p}.self_attn.linear_out.weight'))
    # Conv
    r = x; x_n = ln(x, f'{p}.norm_conv').permute(0,2,1)
    x_n = F.conv1d(x_n, f.get_tensor(f'{p}.conv.pointwise_conv1.weight'))
    a, b = x_n.chunk(2, dim=1); x_n = a * torch.sigmoid(b)
    x_n = F.conv1d(x_n, f.get_tensor(f'{p}.conv.depthwise_conv.weight'), padding=4, groups=1024)
    bn_rv = f.get_tensor(f'{p}.conv.batch_norm.running_var')
    x_n = (x_n - f.get_tensor(f'{p}.conv.batch_norm.running_mean').reshape(1,-1,1)) / (bn_rv+1e-5).sqrt().reshape(1,-1,1) * f.get_tensor(f'{p}.conv.batch_norm.weight').reshape(1,-1,1) + f.get_tensor(f'{p}.conv.batch_norm.bias').reshape(1,-1,1)
    x_n = F.silu(x_n)
    x_n = F.conv1d(x_n, f.get_tensor(f'{p}.conv.pointwise_conv2.weight'))
    x = r + x_n.permute(0,2,1)
    # FF2
    r = x; x_n = ln(x, f'{p}.norm_feed_forward2')
    x_n = F.linear(F.silu(F.linear(x_n, f.get_tensor(f'{p}.feed_forward2.linear1.weight'))), f.get_tensor(f'{p}.feed_forward2.linear2.weight'))
    x = r + 0.5 * x_n
    # norm_out
    x = ln(x, f'{p}.norm_out')

enc_output = x
print(f"Encoder output: [{enc_output.min():.6f}, {enc_output.max():.6f}], std={enc_output.std():.6f}")

# LSTM
def lstm_step(inp, h, c, li):
    gates = inp @ f.get_tensor(f'decoder.prediction.dec_rnn.lstm.weight_ih_l{li}').t() + f.get_tensor(f'decoder.prediction.dec_rnn.lstm.bias_ih_l{li}') + h @ f.get_tensor(f'decoder.prediction.dec_rnn.lstm.weight_hh_l{li}').t() + f.get_tensor(f'decoder.prediction.dec_rnn.lstm.bias_hh_l{li}')
    hs=640; ig=torch.sigmoid(gates[:hs]); fg=torch.sigmoid(gates[hs:2*hs]); gg=torch.tanh(gates[2*hs:3*hs]); og=torch.sigmoid(gates[3*hs:])
    c_new=fg*c+ig*gg; h_new=og*torch.tanh(c_new)
    return h_new, c_new

h0,c0,h1,c1 = torch.zeros(640),torch.zeros(640),torch.zeros(640),torch.zeros(640)
h0_new,c0_new = lstm_step(torch.zeros(640), h0, c0, 0)
h1_new,c1_new = lstm_step(h0_new, h1, c1, 1)
pred_out = h1_new

# Joint
enc_w,enc_b = f.get_tensor('joint.enc.weight'),f.get_tensor('joint.enc.bias')
pred_w,pred_b = f.get_tensor('joint.pred.weight'),f.get_tensor('joint.pred.bias')
out_w,out_b = f.get_tensor('joint.joint_net.2.weight'),f.get_tensor('joint.joint_net.2.bias')

# TDT greedy decode
durations = [0,1,2,3,4]
blank_idx = 8192
hypothesis = []
h0,c0,h1,c1 = torch.zeros(640),torch.zeros(640),torch.zeros(640),torch.zeros(640)
h0_new,c0_new = lstm_step(torch.zeros(640), h0, c0, 0)
h1_new,c1_new = lstm_step(h0_new, h1, c1, 1)
pred_out = h1_new
state = (h0_new,c0_new,h1_new,c1_new)
last_token = blank_idx
t_total = enc_output.shape[1]
time_idx = 0
step = 0

while time_idx < t_total:
    enc_frame = enc_output[0, time_idx]
    enc_h = F.linear(enc_frame, enc_w, enc_b)
    pred_h = F.linear(pred_out, pred_w, pred_b)
    joint_h = F.relu(enc_h + pred_h)
    logits = F.linear(joint_h, out_w, out_b)
    token_logits = logits[:8193]
    dur_logits = logits[8193:]
    
    k = token_logits.argmax().item()
    dur_idx = dur_logits.argmax().item()
    skip = durations[dur_idx] if dur_idx < len(durations) else 1

    if step < 10:
        top5 = torch.topk(token_logits, 5)
        print(f"Step {step}: t={time_idx}/{t_total}, k={k}, dur={dur_idx}, blank={token_logits[8192]:.3f}, top5={list(zip(top5.indices.tolist(), [f'{v:.3f}' for v in top5.values.tolist()]))}")
    
    if k == blank_idx:
        time_idx += max(skip, 1)
    else:
        hypothesis.append(k)
        last_token = k
        # Update LSTM state with this token
        embed_w = f.get_tensor('decoder.prediction.embed.weight')
        embed = embed_w[k]
        h0_new,c0_new = lstm_step(embed, state[0], state[1], 0)
        h1_new,c1_new = lstm_step(h0_new, state[2], state[3], 1)
        pred_out = h1_new
        state = (h0_new,c0_new,h1_new,c1_new)
        time_idx += max(skip, 1)
    
    step += 1

# Decode tokens
import json
with open('models/parakeet-tdt-0.6b-v3/vocab.json') as vf:
    vocab_data = json.load(vf)
vocab = [''] * len(vocab_data)
for piece, info in vocab_data.items():
    vocab[info['id']] = piece

text = ''.join(vocab[t].replace('▁', ' ') for t in hypothesis).strip()
print(f"\nTranscription ({len(hypothesis)} tokens): '{text}'")
