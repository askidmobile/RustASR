#!/usr/bin/env python3
"""
Конвертация весов GigaAM v3 E2E CTC из формата PyTorch в safetensors.

Скрипт скачивает модель с CDN SberDevices (или HuggingFace),
удаляет веса препроцессора (mel вычисляется в Rust),
сохраняет encoder + CTC head как safetensors,
и извлекает словарь SentencePiece токенизатора в vocab.json.

Использование:
    python scripts/convert_gigaam.py --output models/gigaam-v3-e2e-ctc

    # Или из уже скачанного чекпоинта:
    python scripts/convert_gigaam.py --checkpoint ~/.cache/gigaam/v3_e2e_ctc.pt \
        --tokenizer ~/.cache/gigaam/v3_e2e_ctc_tokenizer.model \
        --output models/gigaam-v3-e2e-ctc

    # Или из HuggingFace:
    python scripts/convert_gigaam.py --hf ai-sage/GigaAM-v3 --hf-revision e2e_ctc \
        --output models/gigaam-v3-e2e-ctc
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

import torch

try:
    from safetensors.torch import save_file
except ImportError:
    print("Установите safetensors: pip install safetensors")
    sys.exit(1)

try:
    import sentencepiece as spm
except ImportError:
    print("Установите sentencepiece: pip install sentencepiece")
    sys.exit(1)


# CDN URL для моделей GigaAM
CDN_BASE = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM"
MODEL_HASHES = {
    "v3_e2e_ctc": "367074d6498f426d960b25f49531cf68",
}

# Конфигурация модели v3_e2e_ctc (из HuggingFace config.json)
GIGAAM_V3_E2E_CTC_CONFIG = {
    "model_name": "gigaam-v3-e2e-ctc",
    "model_class": "ctc",
    "sample_rate": 16000,
    "preprocessor": {
        "sample_rate": 16000,
        "features": 64,
        "win_length": 320,
        "hop_length": 160,
        "n_fft": 320,
        "mel_scale": "htk",
        "mel_norm": None,
        "center": False,
    },
    "encoder": {
        "feat_in": 64,
        "n_layers": 16,
        "d_model": 768,
        "subsampling": "conv1d",
        "subs_kernel_size": 5,
        "subsampling_factor": 4,
        "ff_expansion_factor": 4,
        "self_attention_model": "rotary",
        "pos_emb_max_len": 5000,
        "n_heads": 16,
        "conv_kernel_size": 5,
        "conv_norm_type": "layer_norm",
    },
    "head": {
        "feat_in": 768,
        "num_classes": 257,
    },
}


def download_file(url: str, dest: str, expected_md5: str = None) -> str:
    """Скачать файл с прогресс-баром."""
    if os.path.exists(dest):
        if expected_md5:
            md5 = hashlib.md5(open(dest, "rb").read()).hexdigest()
            if md5 == expected_md5:
                print(f"  Файл уже скачан и MD5 совпадает: {dest}")
                return dest
            else:
                print(f"  MD5 не совпадает, повторная загрузка...")
        else:
            print(f"  Файл уже существует: {dest}")
            return dest

    print(f"  Скачиваю: {url}")
    print(f"  Сохраняю в: {dest}")

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = count * block_size * 100 // total_size
            mb = count * block_size / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {pct}% ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()  # newline

    if expected_md5:
        md5 = hashlib.md5(open(dest, "rb").read()).hexdigest()
        if md5 != expected_md5:
            print(f"  ВНИМАНИЕ: MD5 не совпадает! Ожидалось {expected_md5}, получено {md5}")

    return dest


def download_from_cdn(output_dir: str) -> tuple:
    """Скачать модель и токенизатор с CDN SberDevices."""
    model_name = "v3_e2e_ctc"
    cache_dir = os.path.expanduser("~/.cache/gigaam")
    os.makedirs(cache_dir, exist_ok=True)

    # Скачать чекпоинт
    checkpoint_url = f"{CDN_BASE}/{model_name}.pt"
    checkpoint_path = os.path.join(cache_dir, f"{model_name}.pt")
    download_file(checkpoint_url, checkpoint_path, MODEL_HASHES.get(model_name))

    # Скачать токенизатор
    tokenizer_url = f"{CDN_BASE}/{model_name}_tokenizer.model"
    tokenizer_path = os.path.join(cache_dir, f"{model_name}_tokenizer.model")
    download_file(tokenizer_url, tokenizer_path)

    return checkpoint_path, tokenizer_path


def download_from_hf(repo_id: str, revision: str, output_dir: str) -> tuple:
    """Скачать модель с HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Установите huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    print(f"Скачиваю с HuggingFace: {repo_id} (revision={revision})")
    model_path = hf_hub_download(repo_id, "pytorch_model.bin", revision=revision)
    tokenizer_path = hf_hub_download(repo_id, "tokenizer.model", revision=revision)

    return model_path, tokenizer_path


def load_state_dict(checkpoint_path: str, is_hf: bool = False) -> dict:
    """Загрузить state_dict из чекпоинта."""
    print(f"Загружаю чекпоинт: {checkpoint_path}")
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if is_hf:
        # HuggingFace pytorch_model.bin — это state_dict напрямую,
        # но ключи могут иметь префикс 'model.'
        state_dict = data if isinstance(data, dict) and "state_dict" not in data else data.get("state_dict", data)
    else:
        # CDN формат: {"state_dict": ..., "cfg": ...}
        state_dict = data.get("state_dict", data)

    return state_dict


def filter_and_rename_keys(state_dict: dict) -> dict:
    """Фильтрация и переименование ключей для Candle."""
    filtered = {}
    skip_prefixes = ("preprocessor.", "decoding.")

    for key, tensor in state_dict.items():
        # Убрать префикс 'model.' если есть (HuggingFace wrapper)
        clean_key = key
        if clean_key.startswith("model."):
            clean_key = clean_key[len("model."):]

        # Пропустить веса препроцессора (mel вычисляется в Rust)
        if any(clean_key.startswith(p) for p in skip_prefixes):
            continue

        filtered[clean_key] = tensor

    return filtered


def extract_vocab(tokenizer_path: str, output_dir: str) -> int:
    """Извлечь словарь из SentencePiece tokenizer.model в vocab.json."""
    print(f"Извлекаю словарь из: {tokenizer_path}")
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)

    vocab_size = sp.GetPieceSize()
    print(f"  Размер словаря: {vocab_size}")

    vocab = {}
    for i in range(vocab_size):
        piece = sp.IdToPiece(i)
        vocab[str(i)] = piece

    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"  Словарь сохранён: {vocab_path}")
    return vocab_size


def save_config(output_dir: str, vocab_size: int):
    """Сохранить конфигурацию модели."""
    config = GIGAAM_V3_E2E_CTC_CONFIG.copy()
    config["head"]["num_classes"] = vocab_size + 1  # +1 для blank токена

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"  Конфигурация сохранена: {config_path}")


def convert(args):
    """Основная функция конвертации."""
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # 1. Получить файлы модели
    if args.checkpoint and args.tokenizer:
        checkpoint_path = args.checkpoint
        tokenizer_path = args.tokenizer
        is_hf = False
    elif args.hf:
        checkpoint_path, tokenizer_path = download_from_hf(
            args.hf, args.hf_revision, output_dir
        )
        is_hf = True
    else:
        checkpoint_path, tokenizer_path = download_from_cdn(output_dir)
        is_hf = False

    # 2. Загрузить state_dict
    state_dict = load_state_dict(checkpoint_path, is_hf)
    print(f"  Всего параметров: {len(state_dict)}")

    # 3. Фильтрация ключей
    filtered = filter_and_rename_keys(state_dict)
    print(f"  Параметров после фильтрации: {len(filtered)}")

    # Напечатать ключи для отладки
    print("\n  Ключи тензоров:")
    total_params = 0
    for key in sorted(filtered.keys()):
        shape = list(filtered[key].shape)
        n = filtered[key].numel()
        total_params += n
        dtype = filtered[key].dtype
        print(f"    {key}: {shape} ({dtype})")
    print(f"\n  Всего параметров (числ.): {total_params:,}")
    print(f"  Примерный размер (f32): {total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"  Примерный размер (f16): {total_params * 2 / 1024 / 1024:.1f} MB")

    # 4. Конвертировать в f32 для safetensors
    # (Candle загрузит и при необходимости конвертирует в f16)
    converted = {}
    for key, tensor in filtered.items():
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        elif tensor.dtype == torch.float16:
            tensor = tensor.float()
        converted[key] = tensor.contiguous()

    # 5. Сохранить как safetensors
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(converted, safetensors_path)
    file_size_mb = os.path.getsize(safetensors_path) / 1024 / 1024
    print(f"\n  Safetensors сохранён: {safetensors_path} ({file_size_mb:.1f} MB)")

    # 6. Извлечь словарь
    vocab_size = extract_vocab(tokenizer_path, output_dir)

    # 7. Сохранить конфиг
    save_config(output_dir, vocab_size)

    # 8. Скопировать tokenizer.model
    import shutil
    tokenizer_dest = os.path.join(output_dir, "tokenizer.model")
    if not os.path.exists(tokenizer_dest) or not os.path.samefile(tokenizer_path, tokenizer_dest):
        shutil.copy2(tokenizer_path, tokenizer_dest)
        print(f"  Токенизатор скопирован: {tokenizer_dest}")

    print(f"\n✅ Конвертация завершена! Модель сохранена в: {output_dir}")
    print(f"   Файлы:")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        size = os.path.getsize(fpath)
        if size > 1024 * 1024:
            print(f"     {f} ({size / 1024 / 1024:.1f} MB)")
        else:
            print(f"     {f} ({size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Конвертация весов GigaAM в safetensors для RustASR"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Директория для сохранения конвертированной модели",
    )
    parser.add_argument(
        "--checkpoint",
        help="Путь к локальному чекпоинту (.pt или pytorch_model.bin)",
    )
    parser.add_argument(
        "--tokenizer",
        help="Путь к локальному SentencePiece tokenizer.model",
    )
    parser.add_argument(
        "--hf",
        help="HuggingFace repo ID (например, ai-sage/GigaAM-v3)",
    )
    parser.add_argument(
        "--hf-revision",
        default="e2e_ctc",
        help="HuggingFace revision/branch (default: e2e_ctc)",
    )

    args = parser.parse_args()

    if not args.checkpoint and not args.hf:
        print("Скачиваю модель с CDN SberDevices...")

    convert(args)


if __name__ == "__main__":
    main()
