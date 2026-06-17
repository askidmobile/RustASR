#!/usr/bin/env python3
"""
Инспекция .nemo архива Nemotron 3.5 ASR (FastConformer-CacheAware-RNNT + Lang prompt).

Извлекает из .nemo:
- model_config.yaml → ключевые секции (preprocessor / encoder / decoder / joint / prompt)
- model_weights.ckpt → имена и формы всех тензоров, сгруппированные по компоненту

Цель: получить ground-truth для конвертации (.nemo → safetensors + config.json)
и для нативного Rust-порта (crates/model-nemotron).

Использование:
    python scripts/inspect_nemotron.py \
        --nemo-file models/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo \
        [--dump-config out/model_config.yaml]
"""

import argparse
import glob
import os
import sys
import tarfile
import tempfile


def extract_nemo(nemo_path: str, tmpdir: str) -> str:
    """Распаковывает .nemo (tar) и возвращает prefix имён внутри архива."""
    with tarfile.open(nemo_path, "r:*") as tar:
        names = tar.getnames()
        print(f"Файлы в .nemo архиве ({len(names)}):")
        for n in sorted(names):
            print(f"  {n}")
        tar.extractall(path=tmpdir)
    cfgs = glob.glob(os.path.join(tmpdir, "**", "model_config.yaml"), recursive=True)
    if not cfgs:
        print("ОШИБКА: model_config.yaml не найден")
        sys.exit(1)
    return cfgs[0]


def dump_config(config_path: str, dump_to: str | None):
    print("\n" + "=" * 70)
    print("model_config.yaml — ключевые секции")
    print("=" * 70)
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    for section in [
        "preprocessor",
        "encoder",
        "decoder",
        "joint",
        "decoding",
        "model_defaults",
        "prompt",
        "prompt_format",
        "aux_ctc",
        "tokenizer",
        "labels",
        "target_lang",
        "spec_augment",
    ]:
        if section in cfg:
            print(f"\n--- {section} ---")
            try:
                print(OmegaConf.to_yaml(cfg[section]))
            except Exception:
                print(cfg[section])
    # Топ-уровневые скаляры
    print("\n--- top-level keys ---")
    for k in cfg.keys():
        v = cfg[k]
        if not hasattr(v, "keys"):
            print(f"  {k}: {v}")
    if dump_to:
        os.makedirs(os.path.dirname(dump_to) or ".", exist_ok=True)
        OmegaConf.save(cfg, dump_to)
        print(f"\nПолный конфиг сохранён: {dump_to}")


def dump_weights(tmpdir: str):
    import torch

    ckpts = glob.glob(os.path.join(tmpdir, "**", "*.ckpt"), recursive=True)
    if not ckpts:
        ckpts = glob.glob(os.path.join(tmpdir, "**", "*weights*"), recursive=True)
    if not ckpts:
        print("ОШИБКА: файл весов (.ckpt) не найден")
        sys.exit(1)
    weight_file = ckpts[0]
    print("\n" + "=" * 70)
    print(f"Веса: {weight_file}")
    print("=" * 70)

    sd = torch.load(weight_file, map_location="cpu", weights_only=True)
    if not isinstance(sd, dict):
        print("Неожиданный формат state_dict:", type(sd))
        return

    comps: dict[str, list] = {}
    total = 0
    for key, t in sorted(sd.items()):
        comp = key.split(".")[0]
        try:
            shape = list(t.shape)
            dtype = str(t.dtype)
            n = t.numel()
        except Exception:
            shape, dtype, n = ["?"], "?", 0
        comps.setdefault(comp, []).append((key, shape, dtype, n))
        total += n

    print(f"\nВсего тензоров: {len(sd)}, параметров: {total:,} "
          f"(FP32 ≈ {total*4/1024**2:.0f} MB)\n")
    for comp, items in comps.items():
        cp = sum(x[3] for x in items)
        print(f"\n########## {comp}  ({len(items)} тензоров, {cp:,} параметров) ##########")
        for key, shape, dtype, n in items:
            print(f"  {key}: {shape} ({dtype})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo-file", required=True)
    ap.add_argument("--dump-config", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.nemo_file):
        print("Не найден:", args.nemo_file)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = extract_nemo(args.nemo_file, tmp)
        dump_config(cfg_path, args.dump_config)
        dump_weights(tmp)


if __name__ == "__main__":
    main()
