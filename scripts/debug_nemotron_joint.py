#!/usr/bin/env python3
"""Дамп внутренностей RNN-T joint NeMo для frame-0 + greedy token ids — сравнение с Rust."""
import numpy as np
import torch
from safetensors.numpy import load_file
import nemo.collections.asr as nemo_asr

NEMO = "models/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo"
model = nemo_asr.models.ASRModel.restore_from(NEMO, map_location="cpu")
model.eval()
model.encoder.set_default_att_context_size([56, 13])

refs = load_file("tmp/parity/nemotron_parity.safetensors")
# prompt_kernel_out = encoded post-fusion, [1,39,1024] (B,T,H)
enc = torch.tensor(refs["prompt_kernel_out"])  # [1,39,1024]
print("encoded(fused) [B,T,H]:", enc.shape, "range", float(enc.min()), float(enc.max()))

joint = model.joint
dec = model.decoder
print("joint type:", type(joint).__name__, "| has enc/pred:", hasattr(joint, "enc"), hasattr(joint, "pred"))

with torch.inference_mode():
    # enc projection
    f_proj = joint.enc(enc)  # [1,39,640]
    print("f_proj[0,0] mean/min/max:", float(f_proj[0, 0].mean()), float(f_proj[0, 0].min()), float(f_proj[0, 0].max()))

    # SOS prediction
    try:
        g, states = dec.predict(None, None, add_sos=False, batch_size=1)
        print("SOS dec_out shape:", g.shape, "mean/min/max", float(g.mean()), float(g.min()), float(g.max()))
        g_proj = joint.pred(g)  # [1,1,640]
        # frame 0 joint
        finp = f_proj[:, 0:1].unsqueeze(2) + g_proj.unsqueeze(1)  # [1,1,1,640]
        logits0 = joint.joint_net(finp)[0, 0, 0]  # [13088]
        top = torch.topk(logits0, 5)
        print("frame0 top5 ids:", top.indices.tolist(), "vals:", [round(float(v), 2) for v in top.values])
        print("frame0 blank(13087) logit:", float(logits0[13087]))
    except Exception as e:
        print("predict/joint dump failed:", type(e).__name__, e)

    # Полный greedy через decoding
    enc_bdt = enc.transpose(1, 2).contiguous()  # [1,1024,39] (B,H,T)
    elen = torch.tensor([enc.shape[1]])
    hyps = model.decoding.rnnt_decoder_predictions_tensor(enc_bdt, elen, return_hypotheses=True)
    h = hyps[0][0] if isinstance(hyps, tuple) else hyps[0]
    ids = h.y_sequence.tolist() if hasattr(h, "y_sequence") else "?"
    print("GREEDY token ids:", ids)
    print("GREEDY text:", h.text if hasattr(h, "text") else "?")
