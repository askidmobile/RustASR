#!/usr/bin/env python3
"""Run full transcription using qwen_asr."""

from qwen_asr import Qwen3ASRModel

print("Loading model...")
model = Qwen3ASRModel.from_pretrained("models/qwen3-asr-0.6b")
print("Model loaded")

# Transcribe using file path
print("\n=== Python Transcription ===")
result = model.transcribe("test_30sec.wav")
print(f"Result: {result}")
