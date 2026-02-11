#!/usr/bin/env python3
"""Analyze joint network weights and simulate inference."""
import numpy as np
from safetensors import safe_open

f = safe_open('models/parakeet-tdt-0.6b-v3/model.safetensors', framework='numpy')

# Check enc_proj and output layer
enc_w = f.get_tensor('joint.enc.weight')
enc_b = f.get_tensor('joint.enc.bias')
print(f'enc_proj weight: shape={enc_w.shape}, range=[{enc_w.min():.4f}, {enc_w.max():.4f}], std={enc_w.std():.6f}')
print(f'enc_proj bias: range=[{enc_b.min():.4f}, {enc_b.max():.4f}], mean={enc_b.mean():.6f}')

pred_w = f.get_tensor('joint.pred.weight')
pred_b = f.get_tensor('joint.pred.bias')
print(f'pred_proj weight: shape={pred_w.shape}, range=[{pred_w.min():.4f}, {pred_w.max():.4f}]')
print(f'pred_proj bias: range=[{pred_b.min():.4f}, {pred_b.max():.4f}]')

out_w = f.get_tensor('joint.joint_net.2.weight')
out_b = f.get_tensor('joint.joint_net.2.bias')
print(f'output weight: shape={out_w.shape}, range=[{out_w.min():.4f}, {out_w.max():.4f}]')
print(f'output bias: shape={out_b.shape}, range=[{out_b.min():.4f}, {out_b.max():.4f}]')
print(f'output bias[8192] (blank): {out_b[8192]:.6f}')
print(f'output bias[:5] (first tokens): {out_b[:5]}')
print(f'output bias[8190:8198]: {out_b[8190:8198]}')

# Simulate: what if enc_frame ≈ 0.05 and pred_out ≈ 0?
enc_frame_sim = np.ones(1024) * 0.05
pred_out_sim = np.zeros(640)
enc_h = enc_frame_sim @ enc_w.T + enc_b
pred_h = pred_out_sim @ pred_w.T + pred_b
joint_h = np.maximum(enc_h + pred_h, 0)  # ReLU
logits = joint_h @ out_w.T + out_b
token_logits = logits[:8193]

blank_logit = token_logits[8192]
best_token = int(np.argmax(token_logits))
print(f'\nSimulated (enc=0.05*ones, pred=zeros):')
print(f'  enc_h range: [{enc_h.min():.4f}, {enc_h.max():.4f}]')
print(f'  pred_h range: [{pred_h.min():.4f}, {pred_h.max():.4f}]')
print(f'  joint_h range: [{joint_h.min():.4f}, {joint_h.max():.4f}], nonzero={np.count_nonzero(joint_h)}/{len(joint_h)}')
print(f'  best_token={best_token}, blank_logit={blank_logit:.4f}, best_logit={token_logits[best_token]:.4f}')
top5 = sorted(enumerate(token_logits), key=lambda x: -x[1])[:5]
print(f'  top5: {[(idx, f"{val:.4f}") for idx, val in top5]}')

# Also simulate with more realistic encoder output (random small values)
np.random.seed(42)
enc_frame_real = np.random.randn(1024) * 0.05  # std=0.05
enc_h2 = enc_frame_real @ enc_w.T + enc_b
pred_h2 = pred_out_sim @ pred_w.T + pred_b
joint_h2 = np.maximum(enc_h2 + pred_h2, 0)
logits2 = joint_h2 @ out_w.T + out_b
token_logits2 = logits2[:8193]
best_token2 = int(np.argmax(token_logits2))
print(f'\nSimulated (enc=randn*0.05, pred=zeros):')
print(f'  best_token={best_token2}, blank={token_logits2[8192]:.4f}, best={token_logits2[best_token2]:.4f}')
top52 = sorted(enumerate(token_logits2), key=lambda x: -x[1])[:5]
print(f'  top5: {[(idx, f"{val:.4f}") for idx, val in top52]}')
