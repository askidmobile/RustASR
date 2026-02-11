#!/usr/bin/env python3
"""Test the full pipeline WITHOUT xscaling (xscaling=False in model config).

The model config says xscaling: False, which means NeMo does NOT multiply
the subsampled output by sqrt(d_model)=32.0 before the conformer layers.

Our previous Python+Rust code was applying xscale=32, which is WRONG.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
import math
import json

MODEL_DIR = "models/parakeet-tdt-0.6b-v3"

def load_tensor(f, key):
    return torch.from_numpy(f.get_tensor(key).copy())

def layer_norm(x, weight, bias, eps=1e-5):
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)

def main():
    f = safe_open(f"{MODEL_DIR}/model.safetensors", framework="numpy")
    
    # Load audio
    audio = np.load("test_30sec_audio.npy")
    if audio.ndim == 1:
        audio = audio[:16000*5]  # first 5 seconds
    audio_t = torch.from_numpy(audio.copy()).float()
    
    print(f"Audio: shape={audio_t.shape}, range=[{audio_t.min():.4f}, {audio_t.max():.4f}]")
    
    # ========== MEL EXTRACTION ==========
    # Use same mel extraction as before (simplified)
    n_fft = 512
    hop_length = 160
    win_length = 400
    n_mels = 128
    
    # Pre-emphasis
    pre_emph = 0.97
    audio_pe = torch.cat([audio_t[:1], audio_t[1:] - pre_emph * audio_t[:-1]])
    
    # STFT
    window = torch.hann_window(win_length)
    padded = F.pad(audio_pe, (win_length // 2, win_length // 2))
    stft = torch.stft(padded, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                       window=window, return_complex=True, center=False)
    power = stft.abs().pow(2)
    
    # Mel filterbank from model weights  
    mel_basis_np = f.get_tensor("preprocessor.featurizer.fb")  # [1, 128, 257]
    mel_basis = torch.from_numpy(mel_basis_np.copy()).float().squeeze(0)  # [128, 257]
    mel = torch.matmul(mel_basis, power)
    
    # Log
    guard = 2**(-24)
    mel = torch.log(mel + guard)
    
    # Per-feature normalize
    mel_mean = mel.mean(dim=-1, keepdim=True)
    mel_std = mel.std(dim=-1, keepdim=True)
    mel = (mel - mel_mean) / (mel_std + 1e-5)
    
    print(f"Mel: shape={mel.shape}, range=[{mel.min():.4f}, {mel.max():.4f}]")
    
    # ========== SUBSAMPLING ==========
    # Reshape for conv2d: [B, 1, T, F] where T=time, F=features
    mel_input = mel.T.unsqueeze(0).unsqueeze(0)  # [1, 1, T, 128]
    
    def conv2d_forward(x, w, b, stride, padding, groups=1):
        return F.conv2d(x, w, b, stride=stride, padding=padding, groups=groups)
    
    # Stage 0: conv.0 (stride 2)
    w0 = load_tensor(f, "encoder.pre_encode.conv.0.weight")
    b0 = load_tensor(f, "encoder.pre_encode.conv.0.bias")
    x = conv2d_forward(mel_input, w0, b0, stride=(2,2), padding=(1,1))
    x = F.relu(x)
    
    # Stage 1: depthwise conv.2 (stride 2) + pointwise conv.3
    w2 = load_tensor(f, "encoder.pre_encode.conv.2.weight")
    b2 = load_tensor(f, "encoder.pre_encode.conv.2.bias")
    x = conv2d_forward(x, w2, b2, stride=(2,2), padding=(1,1), groups=w2.shape[0])
    x = F.relu(x)
    
    w3 = load_tensor(f, "encoder.pre_encode.conv.3.weight")
    b3 = load_tensor(f, "encoder.pre_encode.conv.3.bias")
    x = conv2d_forward(x, w3, b3, stride=(1,1), padding=(0,0))
    x = F.relu(x)
    
    # Stage 2: depthwise conv.5 (stride 2) + pointwise conv.6
    w5 = load_tensor(f, "encoder.pre_encode.conv.5.weight")
    b5 = load_tensor(f, "encoder.pre_encode.conv.5.bias")
    x = conv2d_forward(x, w5, b5, stride=(2,2), padding=(1,1), groups=w5.shape[0])
    x = F.relu(x)
    
    w6 = load_tensor(f, "encoder.pre_encode.conv.6.weight")
    b6 = load_tensor(f, "encoder.pre_encode.conv.6.bias")
    x = conv2d_forward(x, w6, b6, stride=(1,1), padding=(0,0))
    x = F.relu(x)
    
    # Reshape: [B, C, T', F'] -> [B, T', C*F']
    B, C, T_sub, F_sub = x.shape
    x = x.permute(0, 2, 1, 3).reshape(B, T_sub, C * F_sub)
    
    # Linear projection
    out_w = load_tensor(f, "encoder.pre_encode.out.weight")
    out_b = load_tensor(f, "encoder.pre_encode.out.bias")
    x = F.linear(x, out_w, out_b)
    
    print(f"After subsampling: shape={x.shape}, range=[{x.min():.4f}, {x.max():.4f}]")
    
    # ========== NO XSCALING! ==========
    # xscaling: False means we do NOT multiply by sqrt(d_model)
    # x_scaled = x * math.sqrt(1024)  # OLD - WRONG!
    x_scaled = x  # NEW - CORRECT: no scaling
    
    print(f"After xscale (NONE - xscaling=False): range=[{x_scaled.min():.4f}, {x_scaled.max():.4f}]")
    
    # ========== POSITIONAL ENCODING ==========
    d_model = 1024
    T = x_scaled.shape[1]
    max_len = 2 * T - 1
    positions = torch.arange(T - 1, -T, -1, dtype=torch.float32).unsqueeze(1)
    pe = torch.zeros(max_len, d_model)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    pe = pe.unsqueeze(0)
    
    center = max_len // 2 + 1
    start = center - T
    end = center + T - 1
    pos_emb = pe[:, start:end, :]
    
    # For RelPositionalEncoding, pos_emb is NOT added to x — they're separate
    # pos_emb is used in attention layers directly
    x_enc = x_scaled  # just pass through (dropout is no-op in eval mode)
    
    print(f"Encoder input (no xscale): range=[{x_enc.min():.4f}, {x_enc.max():.4f}]")
    
    # ========== CONFORMER LAYERS (all 24) ==========
    # Full encoder forward pass through all 24 conformer layers
    hidden = x_enc  # [1, T, 1024]
    
    for layer_idx in range(24):
        prefix = f"encoder.layers.{layer_idx}"
        
        # === FFN1 (half-step) ===
        res = hidden
        # norm_feed_forward1
        ln1_w = load_tensor(f, f"{prefix}.norm_feed_forward1.weight")
        ln1_b = load_tensor(f, f"{prefix}.norm_feed_forward1.bias")
        h = layer_norm(hidden, ln1_w, ln1_b)
        # feed_forward1: linear1 -> SiLU -> dropout -> linear2 -> dropout
        ff1_w1 = load_tensor(f, f"{prefix}.feed_forward1.0.weight")
        ff1_b1 = load_tensor(f, f"{prefix}.feed_forward1.0.bias")
        h = F.linear(h, ff1_w1, ff1_b1)
        h = F.silu(h)
        ff1_w2 = load_tensor(f, f"{prefix}.feed_forward1.3.weight")
        ff1_b2 = load_tensor(f, f"{prefix}.feed_forward1.3.bias")
        h = F.linear(h, ff1_w2, ff1_b2)
        hidden = res + 0.5 * h
        
        # === MHSA (RelPositionMultiHeadAttention) ===
        res = hidden
        ln_mha_w = load_tensor(f, f"{prefix}.norm_self_att.weight")
        ln_mha_b = load_tensor(f, f"{prefix}.norm_self_att.bias")
        h = layer_norm(hidden, ln_mha_w, ln_mha_b)
        
        n_heads = 8
        d_k = d_model // n_heads  # 128
        
        # Q, K, V projections
        q_w = load_tensor(f, f"{prefix}.self_attn.linear_q.weight")
        k_w = load_tensor(f, f"{prefix}.self_attn.linear_k.weight")
        v_w = load_tensor(f, f"{prefix}.self_attn.linear_v.weight")
        
        Q = F.linear(h, q_w).view(1, T, n_heads, d_k)  # [1, T, 8, 128]
        K = F.linear(h, k_w).view(1, T, n_heads, d_k)
        V = F.linear(h, v_w).view(1, T, n_heads, d_k).transpose(1, 2)  # [1, 8, T, 128]
        
        # Positional encoding projection
        pos_w = load_tensor(f, f"{prefix}.self_attn.linear_pos.weight")
        P = F.linear(pos_emb, pos_w).view(1, -1, n_heads, d_k).transpose(1, 2)  # [1, 8, 2T-1, 128]
        
        # Biases
        bias_u = load_tensor(f, f"{prefix}.self_attn.pos_bias_u")  # [8, 128]
        bias_v = load_tensor(f, f"{prefix}.self_attn.pos_bias_v")  # [8, 128]
        
        # Q + bias_u, Q + bias_v
        q_u = (Q + bias_u).transpose(1, 2)  # [1, 8, T, 128]
        q_v = (Q + bias_v).transpose(1, 2)  # [1, 8, T, 128]
        
        K_t = K.transpose(1, 2)  # [1, 8, T, 128]
        
        # Matrix AC: (q + bias_u) @ K^T
        matrix_ac = torch.matmul(q_u, K_t.transpose(-2, -1))  # [1, 8, T, T]
        
        # Matrix BD: (q + bias_v) @ P^T then rel_shift
        matrix_bd = torch.matmul(q_v, P.transpose(-2, -1))  # [1, 8, T, 2T-1]
        
        # rel_shift
        b, hh, qlen, pos_len = matrix_bd.shape
        matrix_bd = F.pad(matrix_bd, (1, 0))
        matrix_bd = matrix_bd.view(b, hh, -1, qlen)
        matrix_bd = matrix_bd[:, :, 1:].view(b, hh, qlen, pos_len)
        matrix_bd = matrix_bd[:, :, :, :T]
        
        s_d_k = math.sqrt(d_k)
        scores = (matrix_ac + matrix_bd) / s_d_k
        
        attn = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, V)  # [1, 8, T, 128]
        attn_out = attn_out.transpose(1, 2).reshape(1, T, d_model)
        
        out_w = load_tensor(f, f"{prefix}.self_attn.linear_out.weight")
        attn_out = F.linear(attn_out, out_w)
        
        hidden = res + attn_out
        
        # === Convolution module ===
        res = hidden
        ln_conv_w = load_tensor(f, f"{prefix}.norm_conv.weight")
        ln_conv_b = load_tensor(f, f"{prefix}.norm_conv.bias")
        h = layer_norm(hidden, ln_conv_w, ln_conv_b)
        
        # Pointwise conv1 (expand to 2*d_model for GLU)
        pw1_w = load_tensor(f, f"{prefix}.conv.pointwise_conv1.weight")
        pw1_b = load_tensor(f, f"{prefix}.conv.pointwise_conv1.bias")
        # Shape: h is [1, T, 1024], transpose to [1, 1024, T] for conv1d
        h_t = h.transpose(1, 2)
        h_t = F.conv1d(h_t, pw1_w, pw1_b)  # [1, 2048, T]
        
        # GLU
        h_t = F.glu(h_t, dim=1)  # [1, 1024, T]
        
        # Depthwise conv (kernel=9, padding=4, groups=d_model)
        dw_w = load_tensor(f, f"{prefix}.conv.depthwise_conv.weight")
        dw_b = load_tensor(f, f"{prefix}.conv.depthwise_conv.bias")
        h_t = F.conv1d(h_t, dw_w, dw_b, padding=4, groups=d_model)
        
        # BatchNorm
        bn_w = load_tensor(f, f"{prefix}.conv.batch_norm.weight")
        bn_b = load_tensor(f, f"{prefix}.conv.batch_norm.bias")
        bn_rm = load_tensor(f, f"{prefix}.conv.batch_norm.running_mean")
        bn_rv = load_tensor(f, f"{prefix}.conv.batch_norm.running_var")
        h_t = F.batch_norm(h_t, bn_rm, bn_rv, bn_w, bn_b, training=False)
        
        # SiLU activation
        h_t = F.silu(h_t)
        
        # Pointwise conv2
        pw2_w = load_tensor(f, f"{prefix}.conv.pointwise_conv2.weight")
        pw2_b = load_tensor(f, f"{prefix}.conv.pointwise_conv2.bias")
        h_t = F.conv1d(h_t, pw2_w, pw2_b)
        
        h = h_t.transpose(1, 2)  # back to [1, T, 1024]
        hidden = res + h
        
        # === FFN2 (half-step) ===
        res = hidden
        ln2_w = load_tensor(f, f"{prefix}.norm_feed_forward2.weight")
        ln2_b = load_tensor(f, f"{prefix}.norm_feed_forward2.bias")
        h = layer_norm(hidden, ln2_w, ln2_b)
        ff2_w1 = load_tensor(f, f"{prefix}.feed_forward2.0.weight")
        ff2_b1 = load_tensor(f, f"{prefix}.feed_forward2.0.bias")
        h = F.linear(h, ff2_w1, ff2_b1)
        h = F.silu(h)
        ff2_w2 = load_tensor(f, f"{prefix}.feed_forward2.3.weight")
        ff2_b2 = load_tensor(f, f"{prefix}.feed_forward2.3.bias")
        h = F.linear(h, ff2_w2, ff2_b2)
        hidden = res + 0.5 * h
        
        # === Final LayerNorm ===
        norm_out_w = load_tensor(f, f"{prefix}.norm_out.weight")
        norm_out_b = load_tensor(f, f"{prefix}.norm_out.bias")
        hidden = layer_norm(hidden, norm_out_w, norm_out_b)
        
        if layer_idx == 0 or layer_idx == 23:
            print(f"  Layer {layer_idx}: range=[{hidden.min():.6f}, {hidden.max():.6f}], std={hidden.std():.6f}")
    
    encoder_out = hidden
    print(f"\nEncoder output: range=[{encoder_out.min():.6f}, {encoder_out.max():.6f}], std={encoder_out.std():.6f}")
    
    # ========== DECODER (LSTM) ==========
    # Start with blank token (index 8192)
    embed_w = load_tensor(f, "decoder.prediction.embed.weight")  # [8193, 640]
    blank_embed = embed_w[8192]  # should be zeros
    
    # LSTM Layer 0
    lstm0_wih = load_tensor(f, "decoder.prediction.lstm.lstm.weight_ih_l0")
    lstm0_whh = load_tensor(f, "decoder.prediction.lstm.lstm.weight_hh_l0")
    lstm0_bih = load_tensor(f, "decoder.prediction.lstm.lstm.bias_ih_l0")
    lstm0_bhh = load_tensor(f, "decoder.prediction.lstm.lstm.bias_hh_l0")
    
    lstm1_wih = load_tensor(f, "decoder.prediction.lstm.lstm.weight_ih_l1")
    lstm1_whh = load_tensor(f, "decoder.prediction.lstm.lstm.weight_hh_l1")
    lstm1_bih = load_tensor(f, "decoder.prediction.lstm.lstm.bias_ih_l1")
    lstm1_bhh = load_tensor(f, "decoder.prediction.lstm.lstm.bias_hh_l1")
    
    # Initialize LSTM state
    h0 = torch.zeros(2, 1, 640)
    c0 = torch.zeros(2, 1, 640)
    
    # Create LSTM module and load weights
    lstm = nn.LSTM(input_size=640, hidden_size=640, num_layers=2, batch_first=True)
    lstm.weight_ih_l0.data = lstm0_wih
    lstm.weight_hh_l0.data = lstm0_whh
    lstm.bias_ih_l0.data = lstm0_bih
    lstm.bias_hh_l0.data = lstm0_bhh
    lstm.weight_ih_l1.data = lstm1_wih
    lstm.weight_hh_l1.data = lstm1_whh
    lstm.bias_ih_l1.data = lstm1_bih
    lstm.bias_hh_l1.data = lstm1_bhh
    lstm.eval()
    
    pred_input = blank_embed.unsqueeze(0).unsqueeze(0)  # [1, 1, 640]
    with torch.no_grad():
        pred_out, (h_n, c_n) = lstm(pred_input, (h0, c0))
    pred_out = pred_out.squeeze(0).squeeze(0)  # [640]
    
    # ========== JOINT NETWORK ==========
    enc_w = load_tensor(f, "joint.enc.weight")
    enc_b = load_tensor(f, "joint.enc.bias")
    pred_w = load_tensor(f, "joint.pred.weight")
    pred_b = load_tensor(f, "joint.pred.bias")
    out_w_j = load_tensor(f, "joint.joint_net.2.weight")
    out_b_j = load_tensor(f, "joint.joint_net.2.bias")
    
    # Process first 10 frames
    print("\n=== GREEDY DECODE (first 10 frames) ===")
    
    # Load vocab
    with open(f"{MODEL_DIR}/vocab.json") as vf:
        vocab = json.load(vf)
    
    tokens = []
    h_state = h0.clone()
    c_state = c0.clone()
    last_token = 8192  # blank
    
    for t in range(min(10, encoder_out.shape[1])):
        enc_frame = encoder_out[0, t, :]  # [1024]
        
        # Encoder projection
        enc_h = F.linear(enc_frame, enc_w, enc_b)  # [640]
        
        # Prediction projection  
        pred_h = F.linear(pred_out, pred_w, pred_b)  # [640]
        
        # Joint
        joint_h = F.relu(enc_h + pred_h)  # [640]
        logits = F.linear(joint_h, out_w_j, out_b_j)  # [8198]
        
        # Only consider token logits [0:8193] (vocab + blank), not duration logits [8193:8198]
        token_logits = logits[:8193]
        
        best_idx = token_logits.argmax().item()
        best_val = token_logits[best_idx].item()
        blank_val = token_logits[8192].item()
        
        # Top 5
        top5_vals, top5_idx = token_logits.topk(5)
        top5_str = [(idx.item(), f"{val.item():.4f}") for idx, val in zip(top5_idx, top5_vals)]
        
        is_blank = best_idx == 8192
        tok_name = "<blank>" if is_blank else vocab.get(str(best_idx), f"?{best_idx}")
        print(f"  Frame {t}: best={best_idx}({tok_name}) val={best_val:.4f}, blank={blank_val:.4f}, top5={top5_str}")
        
        if not is_blank:
            tokens.append(best_idx)
            # Update LSTM with the new token
            tok_embed = embed_w[best_idx].unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                pred_out_new, (h_state, c_state) = lstm(tok_embed, (h_state, c_state))
            pred_out = pred_out_new.squeeze(0).squeeze(0)
            
            # Re-check same frame with updated prediction (RNNT inner loop)
            pred_h = F.linear(pred_out, pred_w, pred_b)
            joint_h = F.relu(enc_h + pred_h)
            logits = F.linear(joint_h, out_w_j, out_b_j)
            token_logits = logits[:8193]
            best_idx2 = token_logits.argmax().item()
            if best_idx2 != 8192:
                tok_name2 = vocab.get(str(best_idx2), f"?{best_idx2}")
                print(f"    -> next symbol: {best_idx2}({tok_name2})")
    
    if tokens:
        text = "".join([vocab.get(str(t), "?") for t in tokens]).replace("▁", " ").strip()
        print(f"\nDecoded text: '{text}'")
    else:
        print("\nNo tokens decoded (all blank)")

if __name__ == "__main__":
    main()
