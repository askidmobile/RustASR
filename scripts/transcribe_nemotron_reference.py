#!/usr/bin/env python3
"""
Эталонная транскрипция Nemotron 3.5 ASR (NeMo, offline) — золотой результат
для последующей parity-проверки нативного Rust-порта.

Модель: EncDecRNNTBPEModelWithPrompt (FastConformer-CacheAware-RNNT + Lang prompt).
Язык: target_lang=ru-RU (индекс prompt = 11).

Использование:
    python scripts/transcribe_nemotron_reference.py \
        --nemo models/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo \
        --lang ru-RU \
        --out tmp/nemotron_reference_ru.json \
        --audio test_prod_3s.wav test_30sec.wav \
                "/Volumes/Askid Dev/Projects/Yttri/.claude/worktrees/beautiful-cerf-f0ad99/frontend/src-tauri/tests/fixtures/test_ru.wav"
"""

import argparse
import inspect
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--lang", default="ru-RU")
    ap.add_argument("--audio", nargs="+", required=True)
    ap.add_argument("--out", default="tmp/nemotron_reference_ru.json")
    args = ap.parse_args()

    import torch
    import nemo.collections.asr as nemo_asr

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[load] device={dev} restoring {args.nemo}")
    t0 = time.time()
    model = nemo_asr.models.ASRModel.restore_from(args.nemo, map_location="cpu")
    model.eval()
    print(f"[load] class={type(model).__name__} in {time.time()-t0:.1f}s")

    # Введение в API: что принимает transcribe + prompt-поля
    try:
        sig = inspect.signature(model.transcribe)
        print(f"[api] transcribe{sig}")
    except (ValueError, TypeError):
        print("[api] transcribe signature unavailable")
    for attr in ("prompt_field", "prompt_format", "default_prompt"):
        if hasattr(model, attr):
            print(f"[api] model.{attr} = {getattr(model, attr)}")

    paths = list(args.audio)
    results = {}

    def run(label, **kw):
        print(f"\n[try:{label}] kwargs={list(kw)}")
        t = time.time()
        out = model.transcribe(paths, batch_size=1, **kw)
        dt = time.time() - t
        texts = []
        for o in out:
            texts.append(o.text if hasattr(o, "text") else (o[0] if isinstance(o, (list, tuple)) else str(o)))
        for p, tx in zip(paths, texts):
            print(f"   {os.path.basename(p)} :: {tx!r}")
        print(f"[try:{label}] OK in {dt:.1f}s")
        return texts

    texts = None
    attempts = [
        ("target_lang_kwarg", dict(target_lang=args.lang)),
        ("prompt_list", dict(prompt=[{"target_lang": args.lang}] * len(paths))),
        ("plain", dict()),
    ]
    last_err = None
    for label, kw in attempts:
        try:
            texts = run(label, **kw)
            used = label
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[try:{label}] FAILED: {type(e).__name__}: {str(e)[:300]}")
    if texts is None:
        print("\nВСЕ попытки провалились:", last_err)
        raise SystemExit(1)

    for p, tx in zip(paths, texts):
        results[p] = tx
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"lang": args.lang, "api": used, "results": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n[done] api={used}; сохранено: {args.out}")


if __name__ == "__main__":
    main()
