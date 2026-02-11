#!/usr/bin/env python3
"""Test Parakeet TDT v3 using NeMo directly."""
import torch
import nemo.collections.asr as nemo_asr

# Load model from .nemo file
model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
    'models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo',
    map_location='cpu'
)
model.eval()

# Transcribe
files = [
    'tests/fixtures/test_speech_en_16k.wav',
    'tests/fixtures/test_real_speech_5s.wav',
]

for f in files:
    result = model.transcribe([f])
    print(f'{f}: "{result[0]}"')
