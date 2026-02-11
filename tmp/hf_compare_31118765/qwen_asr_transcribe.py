#!/usr/bin/env python3
"""
Запуск эталонного Python-инференса Qwen3-ASR через пакет qwen-asr.

Нужен для сравнения качества HF (Python) vs RustASR.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Путь к директории модели (HF format).")
    ap.add_argument("--audio", required=True, help="Путь к аудиофайлу (wav/mp3/...).")
    ap.add_argument(
        "--language",
        default="auto",
        help='Язык ("Russian") или "auto" для автоопределения.',
    )
    ap.add_argument("--device", default="mps", help='Устройство ("mps" или "cpu").')
    ap.add_argument("--dtype", default="f16", choices=["f16", "bf16"], help="Тип весов.")
    ap.add_argument("--max-new-tokens", type=int, default=1024, help="Лимит генерации.")
    ap.add_argument("--out-text", required=True, help="Куда сохранить распознанный текст.")
    args = ap.parse_args()

    import torch
    from qwen_asr import Qwen3ASRModel

    dtype = torch.float16 if args.dtype == "f16" else torch.bfloat16
    language = None if args.language == "auto" else args.language

    out_text = Path(args.out_text)
    out_text.parent.mkdir(parents=True, exist_ok=True)

    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device,
        max_inference_batch_size=1,
        max_new_tokens=args.max_new_tokens,
    )

    results = model.transcribe(audio=args.audio, language=language)
    if not results:
        raise SystemExit("qwen-asr вернул пустой результат")

    r = results[0]
    text = (r.text or "").strip()
    lang = (r.language or "").strip()

    out_text.write_text(text + "\n", encoding="utf-8")

    print(f"language: {lang}")
    print(text)


if __name__ == "__main__":
    main()

