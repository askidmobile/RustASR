#!/usr/bin/env python3
"""Compare LSTM prediction network + joint network output between PyTorch and Rust."""
import torch
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open

f = safe_open('models/parakeet-tdt-0.6b-v3/model.safetensors', framework='pt')

# LSTM layers
def lstm_step(x, h, c, layer_idx):
    """Single LSTM step."""
    wih = f.get_tensor(f'decoder.prediction.dec_rnn.lstm.weight_ih_l{layer_idx}')
    whh = f.get_tensor(f'decoder.prediction.dec_rnn.lstm.weight_hh_l{layer_idx}')
    bih = f.get_tensor(f'decoder.prediction.dec_rnn.lstm.bias_ih_l{layer_idx}')
    bhh = f.get_tensor(f'decoder.prediction.dec_rnn.lstm.bias_hh_l{layer_idx}')
    
    gates = x @ wih.t() + bih + h @ whh.t() + bhh
    hs = h.shape[0]
    
    i_gate = torch.sigmoid(gates[:hs])
    f_gate = torch.sigmoid(gates[hs:2*hs])
    g_gate = torch.tanh(gates[2*hs:3*hs])
    o_gate = torch.sigmoid(gates[3*hs:])
    
    c_new = f_gate * c + i_gate * g_gate
    h_new = o_gate * torch.tanh(c_new)
    return h_new, c_new

# Initial state: blank embedding (zeros) + zero LSTM state
embed = torch.zeros(640)
h0, c0 = torch.zeros(640), torch.zeros(640)
h1, c1 = torch.zeros(640), torch.zeros(640)

# LSTM layer 0
h0_new, c0_new = lstm_step(embed, h0, c0, 0)
print(f"LSTM L0 output: [{h0_new.min():.6f}, {h0_new.max():.6f}], norm={h0_new.norm():.4f}")

# LSTM layer 1
h1_new, c1_new = lstm_step(h0_new, h1, c1, 1)
print(f"LSTM L1 output (pred_out): [{h1_new.min():.6f}, {h1_new.max():.6f}], norm={h1_new.norm():.4f}")

pred_out = h1_new

# Now load encoder output from file (or simulate)
# For a real comparison, let's use the actual encoder output from the python full_encoder_comparison
# But for now, simulate with small values
enc_frame = torch.randn(1024) * 0.01  # approximate encoder output std

# Joint network
enc_w, enc_b = f.get_tensor('joint.enc.weight'), f.get_tensor('joint.enc.bias')
pred_w, pred_b = f.get_tensor('joint.pred.weight'), f.get_tensor('joint.pred.bias')
out_w, out_b = f.get_tensor('joint.joint_net.2.weight'), f.get_tensor('joint.joint_net.2.bias')

enc_h = F.linear(enc_frame, enc_w, enc_b)
pred_h = F.linear(pred_out, pred_w, pred_b)
joint_h = F.relu(enc_h + pred_h)
logits = F.linear(joint_h, out_w, out_b)
token_logits = logits[:8193]

print(f"\nWith LSTM pred_out (random enc_frame std=0.01):")
print(f"  enc_h: [{enc_h.min():.4f}, {enc_h.max():.4f}]")
print(f"  pred_h: [{pred_h.min():.4f}, {pred_h.max():.4f}]")
print(f"  joint_h: [{joint_h.min():.4f}, {joint_h.max():.4f}], nonzero={joint_h.count_nonzero()}/{len(joint_h)}")
top5 = torch.topk(token_logits, 5)
print(f"  top5: {list(zip(top5.indices.tolist(), [f'{v:.4f}' for v in top5.values.tolist()]))}")
print(f"  blank (8192): {token_logits[8192]:.4f}")

# Now with pred_out = zeros (no LSTM)
enc_h2 = F.linear(enc_frame, enc_w, enc_b)
pred_h2 = F.linear(torch.zeros(640), pred_w, pred_b)
joint_h2 = F.relu(enc_h2 + pred_h2)
logits2 = F.linear(joint_h2, out_w, out_b)
token_logits2 = logits2[:8193]

print(f"\nWith ZERO pred_out (same enc_frame):")
print(f"  enc_h: [{enc_h2.min():.4f}, {enc_h2.max():.4f}]")
print(f"  pred_h: [{pred_h2.min():.4f}, {pred_h2.max():.4f}]")
print(f"  joint_h: [{joint_h2.min():.4f}, {joint_h2.max():.4f}], nonzero={joint_h2.count_nonzero()}/{len(joint_h2)}")
top52 = torch.topk(token_logits2, 5)
print(f"  top5: {list(zip(top52.indices.tolist(), [f'{v:.4f}' for v in top52.values.tolist()]))}")
print(f"  blank (8192): {token_logits2[8192]:.4f}")

# Compare: what's the difference due to LSTM?
diff = (pred_h - pred_h2).abs()
print(f"\nDifference in pred_h: max={diff.max():.4f}, mean={diff.mean():.4f}")
print(f"LSTM pred_out vs zeros: norm_diff = {(pred_out - torch.zeros(640)).norm():.4f}")
