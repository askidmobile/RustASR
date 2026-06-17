#!/usr/bin/env python3
"""
Конвертация Nemotron 3.5 ASR (.nemo) → safetensors + config.json + vocab.json + tokenizer.model
для нативного Rust-загрузчика (crates/model-nemotron).

Архитектура (из model_config.yaml, подтверждено инспекцией весов):
- preprocessor.featurizer: fb [1,128,257] (мел-фильтрбанк), window [400] (hann)
- encoder: FastConformer dw_striding 8×, 24 слоя, d_model 1024, n_heads 8, conv_kernel 9,
           causal (conv_context_size=causal, causal_downsampling=true), use_bias=false
- decoder.prediction: embed [13088,640] + 2-layer LSTM (hidden 640)  [RNN-T, НЕ TDT]
- joint: enc [640,1024], pred [640,640], joint_net.2 [13088,640]  (num_classes 13087 + blank)
- prompt_kernel: Linear(1152→2048) → act → Linear(2048→1024); 1152 = 1024 акустика + 128 lang one-hot
- язык: prompt_dictionary[ru-RU] = 11 (128-dim one-hot, позиция 11)

Использование:
    python scripts/convert_nemotron.py \
        --nemo-file models/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo \
        --output models/nemotron-3.5-asr-streaming-0.6b
"""

import argparse
import glob
import json
import os
import shutil
import sys
import tarfile
import tempfile

import torch
from safetensors.torch import save_file


def extract_nemo(nemo_path: str, tmp: str):
    with tarfile.open(nemo_path, "r:*") as tar:
        tar.extractall(path=tmp)
    cfg = glob.glob(os.path.join(tmp, "**", "model_config.yaml"), recursive=True)
    ckpt = glob.glob(os.path.join(tmp, "**", "*.ckpt"), recursive=True)
    toks = glob.glob(os.path.join(tmp, "**", "tokenizer.model"), recursive=True)
    if not cfg or not ckpt:
        print("ОШИБКА: не найдены model_config.yaml / .ckpt в .nemo")
        sys.exit(1)
    return cfg[0], ckpt[0], (toks[0] if toks else None)


def filter_weights(sd: dict):
    """Оставляем только инференс-веса. Drop: num_batches_tracked, любые ctc/aux."""
    out, dropped = {}, []
    for k, v in sd.items():
        if k.endswith(".num_batches_tracked"):
            dropped.append(k); continue
        if k.startswith(("ctc_decoder.", "aux_ctc.", "decoder_losses")):
            dropped.append(k); continue
        out[k] = v.contiguous()
    if dropped:
        print(f"Отброшено {len(dropped)} тензоров: {dropped[:6]}{'...' if len(dropped)>6 else ''}")
    return out


def build_config(cfg) -> dict:
    pp = cfg.preprocessor
    enc = cfg.encoder
    # att_context_size: список вариантов [[56,3],[56,0],[56,6],[56,13]];
    # для offline-инференса берём максимальный правый контекст (лучший WER): [56,13].
    att = None
    try:
        att = [list(x) for x in enc.att_context_size]
    except Exception:
        att = None
    prompt_dict = {}
    for path in ("model_defaults.prompt_dictionary", "prompt_format_fn.prompt_dictionary"):
        try:
            node = cfg
            for k in path.split("."):
                node = node[k]
            from omegaconf import OmegaConf as _OC
            d = _OC.to_container(node, resolve=True)
            if d:
                prompt_dict = d
                break
        except Exception:
            pass

    return {
        "model_name": "nemotron-3.5-asr-streaming-0.6b",
        "model_class": "rnnt_prompt",  # FastConformer-CacheAware-RNNT + Lang prompt
        "sample_rate": 16000,
        "preprocessor": {
            "sample_rate": int(getattr(pp, "sample_rate", 16000)),
            "features": int(getattr(pp, "features", 128)),
            "n_fft": int(getattr(pp, "n_fft", 512)),
            "window_size": float(getattr(pp, "window_size", 0.025)),
            "window_stride": float(getattr(pp, "window_stride", 0.01)),
            "window": str(getattr(pp, "window", "hann")),
            "normalize": str(getattr(pp, "normalize", "NA")),
            "log": bool(getattr(pp, "log", True)),
            "preemph": float(getattr(pp, "preemph", 0.97)) if getattr(pp, "preemph", 0.97) is not None else 0.97,
            "dither": 0.0,  # инференс — без dither
            "pad_to": 0,
        },
        "encoder": {
            "feat_in": int(getattr(enc, "feat_in", 128)),
            "n_layers": int(getattr(enc, "n_layers", 24)),
            "d_model": int(getattr(enc, "d_model", 1024)),
            "n_heads": int(getattr(enc, "n_heads", 8)),
            "ff_expansion_factor": int(getattr(enc, "ff_expansion_factor", 4)),
            "d_ff": int(getattr(enc, "d_model", 1024)) * int(getattr(enc, "ff_expansion_factor", 4)),
            "conv_kernel_size": int(getattr(enc, "conv_kernel_size", 9)),
            "conv_norm_type": str(getattr(enc, "conv_norm_type", "layer_norm")),
            "conv_context_size": str(getattr(enc, "conv_context_size", "causal")),
            "causal_downsampling": bool(getattr(enc, "causal_downsampling", True)),
            "subsampling": str(getattr(enc, "subsampling", "dw_striding")),
            "subsampling_factor": int(getattr(enc, "subsampling_factor", 8)),
            "subsampling_conv_channels": int(getattr(enc, "subsampling_conv_channels", 256)),
            "self_attention_model": str(getattr(enc, "self_attention_model", "rel_pos")),
            "att_context_style": str(getattr(enc, "att_context_style", "chunked_limited")),
            "att_context_size": att,
            "att_context_for_offline": [56, 13],
            "use_bias": bool(getattr(enc, "use_bias", False)),
            "xscaling": bool(getattr(enc, "xscaling", False)),
            "untie_biases": bool(getattr(enc, "untie_biases", True)),
            "pos_emb_max_len": int(getattr(enc, "pos_emb_max_len", 5000)),
        },
        "decoder": {
            "vocab_size": 13087,          # реальных токенов (без blank)
            "embed_rows": 13088,          # embed.weight rows = vocab + blank
            "pred_hidden": 640,
            "pred_rnn_layers": 2,
            "blank_idx": 13087,           # blank = последний индекс
        },
        "joint": {
            "encoder_hidden": 1024,
            "pred_hidden": 640,
            "joint_hidden": 640,
            "num_classes": 13087,
            "output_dim": 13088,          # num_classes + blank
            "activation": "relu",
        },
        "prompt": {
            "num_prompts": 128,
            "hidden": 2048,
            "in_features": 1152,          # 1024 + 128
            "out_features": 1024,
            "prompt_field": "target_lang",
            "lang_index": {"ru-RU": 11},
            "prompt_dictionary": prompt_dict,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo-file", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    from omegaconf import OmegaConf

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path, ckpt_path, tok_path = extract_nemo(args.nemo_file, tmp)
        cfg = OmegaConf.load(cfg_path)

        print(f"Загрузка весов: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        filtered = filter_weights(sd)

        st_path = os.path.join(args.output, "model.safetensors")
        save_file(filtered, st_path, metadata={"format": "pt"})
        print(f"model.safetensors: {os.path.getsize(st_path)/1024**2:.1f} MB, {len(filtered)} тензоров")

        # config.json
        config = build_config(cfg)
        with open(os.path.join(args.output, "config.json"), "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print("config.json записан")

        # vocab.json из joint.vocabulary (13087 pieces, blank=13087)
        vocab_list = list(cfg.joint.vocabulary)
        vocab = {piece: {"id": i} for i, piece in enumerate(vocab_list)}
        with open(os.path.join(args.output, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=1)
        print(f"vocab.json: {len(vocab_list)} токенов (blank idx={len(vocab_list)})")

        # tokenizer.model (SentencePiece) — для эталонной детокенизации
        if tok_path:
            shutil.copy2(tok_path, os.path.join(args.output, "tokenizer.model"))
            print("tokenizer.model скопирован")
        else:
            print("WARN: tokenizer.model не найден в .nemo")

    print("\nГОТОВО. Файлы:")
    for f in sorted(os.listdir(args.output)):
        p = os.path.join(args.output, f)
        if os.path.isfile(p):
            print(f"  {f}: {os.path.getsize(p)/1024**2:.2f} MB")


if __name__ == "__main__":
    main()
