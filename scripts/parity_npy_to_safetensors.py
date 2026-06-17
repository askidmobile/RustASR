#!/usr/bin/env python3
"""Сконвертировать parity-дампы tmp/parity/nemotron_*.npy → один safetensors
(candle читает нативно) для Rust parity-тестов model-nemotron."""

import glob
import os

import numpy as np
import torch
from safetensors.torch import save_file

ddir = "tmp/parity"
tensors = {}
for path in sorted(glob.glob(os.path.join(ddir, "nemotron_*.npy"))):
    name = os.path.splitext(os.path.basename(path))[0].replace("nemotron_", "")
    arr = np.load(path).astype(np.float32)
    tensors[name] = torch.from_numpy(np.ascontiguousarray(arr))
    print(f"  {name}: {list(arr.shape)}")

out = os.path.join(ddir, "nemotron_parity.safetensors")
save_file(tensors, out)
print(f"saved: {out} ({os.path.getsize(out)/1024:.0f} KB)")
