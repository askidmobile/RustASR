#!/usr/bin/env python3
"""Дампит промежуточные признаки Python-референса (qwen-asr).

Запуск:
  .venv312/bin/python scripts/dump_python_features.py --audio test_30sec.wav
"""

from __future__ import annotations

import argparse

import numpy as np
import soundfile as sf
import torch

from qwen_asr import Qwen3ASRModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen3-asr-0.6b")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--max-new", type=int, default=32)
    args = ap.parse_args()

    model = Qwen3ASRModel.from_pretrained(
        args.model,
        device_map="cpu",
        dtype=torch.float32,
        max_new_tokens=args.max_new,
    )

    wav, sr = sf.read(args.audio, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(-1)
    if sr != 16000:
        raise SystemExit(f"Ожидается 16kHz WAV, got sr={sr}")
    wav = np.asarray(wav, dtype=np.float32)

    prompt = model.processor.apply_chat_template(
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = model.processor(
        text=[prompt], audio=[wav], return_tensors="pt", padding=True
    )
    print("processor keys:", list(inputs.keys()))
    print("input_ids shape:", tuple(inputs["input_ids"].shape))
    print("input_features shape:", tuple(inputs["input_features"].shape))

    ids = inputs["input_ids"][0].tolist()
    print("first 24 input_ids:", ids[:24])
    thinker = model.model.thinker
    audio_pad = int(thinker.config.audio_token_id)
    print("audio_token_id:", audio_pad)
    print("audio_pad_count:", sum(1 for x in ids if x == audio_pad))

    feats = inputs["input_features"]
    # feats: [B, n_mels, T]
    mel0_5 = feats[0, 0, :5].detach().cpu().numpy().tolist()
    print("python input_features[0, mel=0, :5]:", [float(x) for x in mel0_5])

    with torch.no_grad():
        audio_features = thinker.get_audio_features(
            input_features=inputs["input_features"],
            feature_attention_mask=inputs.get("feature_attention_mask"),
            audio_feature_lengths=None,
        )
    print(
        "python audio_features shape:",
        tuple(audio_features.shape),
        "dtype:",
        audio_features.dtype,
    )
    a = audio_features[:2, :8].detach().cpu().numpy()
    print("python audio_features[:2, :8]:\n", a)

    # Быстрая генерация (для ориентира)
    gen = model.model.generate(
        **{
            k: v
            for k, v in inputs.items()
            if k
            in [
                "input_ids",
                "attention_mask",
                "input_features",
                "feature_attention_mask",
            ]
        },
        max_new_tokens=args.max_new,
    )
    # В qwen-asr они декодируют только суффикс
    start = inputs["input_ids"].shape[1]
    out_ids = gen.sequences[:, start:]
    out_list = out_ids[0].tolist()
    print("python gen token_ids (first 16):", out_list[:16])
    txt = model.processor.batch_decode(
        out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print("python decoded:", txt)


if __name__ == "__main__":
    main()
