#!/usr/bin/env python3
"""
Эталон Nemotron 3.5 ASR (NeMo, offline, manual forward) + дамп референс-тензоров
для parity-проверки нативного Rust-порта.

Обходит lhotse/prompt dataloader (он требует per-cut language). Вместо .transcribe()
вызываем model.forward(input_signal, input_signal_length, prompt_indices=[11]) напрямую
(11 = ru-RU) и затем model.decoding.rnnt_decoder_predictions_tensor.

Дампит (для короткого файла): processed_signal (mel), encoded_pre_prompt,
encoded_post_prompt → tmp/parity/*.npy + meta.json.

Использование:
    python scripts/nemotron_reference_and_dump.py \
        --nemo models/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo \
        --lang-index 11 \
        --out tmp/nemotron_reference_ru.json \
        --dump-audio test_prod_3s.wav \
        --audio test_prod_3s.wav test_30sec.wav <yttri test_ru.wav>
"""

import argparse
import json
import os
import time

import numpy as np
import soundfile as sf
import torch


def load_audio(path: str) -> np.ndarray:
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    assert sr == 16000, f"{path}: ожидается 16kHz, got {sr}"
    return a


def decode_text(model, encoded, encoded_len):
    out = model.decoding.rnnt_decoder_predictions_tensor(
        encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=False
    )
    # API возвращает list[str] или list[Hypothesis] или tuple
    if isinstance(out, tuple):
        out = out[0]
    h = out[0]
    return h.text if hasattr(h, "text") else str(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--lang-index", type=int, default=11)  # ru-RU
    ap.add_argument("--audio", nargs="+", required=True)
    ap.add_argument("--dump-audio", default=None)
    ap.add_argument("--out", default="tmp/nemotron_reference_ru.json")
    ap.add_argument("--att-context", default="56,13")
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr

    t0 = time.time()
    model = nemo_asr.models.ASRModel.restore_from(args.nemo, map_location="cpu")
    model.eval()
    print(f"[load] {type(model).__name__} in {time.time()-t0:.1f}s; concat={getattr(model,'concat',None)} "
          f"num_prompts={getattr(model,'num_prompts',None)}")

    # offline: максимальный контекст для лучшего WER
    try:
        lc, rc = (int(x) for x in args.att_context.split(","))
        model.encoder.set_default_att_context_size([lc, rc])
        print(f"[cfg] att_context_size=[{lc},{rc}]")
    except Exception as e:  # noqa: BLE001
        print(f"[cfg] set_default_att_context_size недоступен: {e}")

    # hooks для дампа
    captured = {}
    if args.dump_audio:
        def enc_hook(_m, _inp, out):
            captured["encoded_pre_prompt"] = (out[0] if isinstance(out, tuple) else out).detach().cpu().numpy()
        def pk_hook(_m, inp, out):
            captured["prompt_kernel_in"] = inp[0].detach().cpu().numpy()
            captured["prompt_kernel_out"] = out.detach().cpu().numpy()
        model.encoder.register_forward_hook(enc_hook)
        model.prompt_kernel.register_forward_hook(pk_hook)

    results = {}
    for path in args.audio:
        a = load_audio(path)
        sig = torch.from_numpy(a).unsqueeze(0)
        sig_len = torch.tensor([sig.shape[1]], dtype=torch.long)
        prompt_idx = torch.tensor([args.lang_index], dtype=torch.long)
        t = time.time()
        with torch.inference_mode():
            # mel для дампа
            proc, proc_len = model.preprocessor(input_signal=sig, length=sig_len)
            if args.dump_audio and os.path.abspath(path) == os.path.abspath(args.dump_audio):
                captured["mel"] = proc.detach().cpu().numpy()
            encoded, encoded_len = model.forward(
                input_signal=sig, input_signal_length=sig_len, prompt_indices=prompt_idx
            )
            text = decode_text(model, encoded, encoded_len)
        dt = time.time() - t
        dur = len(a) / 16000.0
        rtf = dt / dur if dur else 0
        results[path] = {"text": text, "dur_s": round(dur, 2), "infer_s": round(dt, 2), "rtf": round(rtf, 3)}
        print(f"  {os.path.basename(path)} [{dur:.1f}s, rtf={rtf:.3f}] :: {text!r}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"lang_index": args.lang_index, "att_context": args.att_context, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"[done] golden → {args.out}")

    if args.dump_audio and captured:
        ddir = "tmp/parity"
        os.makedirs(ddir, exist_ok=True)
        meta = {}
        for name, arr in captured.items():
            np.save(os.path.join(ddir, f"nemotron_{name}.npy"), arr)
            meta[name] = list(arr.shape)
        with open(os.path.join(ddir, "nemotron_meta.json"), "w") as f:
            json.dump({"dump_audio": args.dump_audio, "lang_index": args.lang_index, "shapes": meta}, f, indent=2)
        print(f"[dump] parity тензоры → {ddir}: {meta}")


if __name__ == "__main__":
    main()
