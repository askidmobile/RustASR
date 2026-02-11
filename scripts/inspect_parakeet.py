#!/usr/bin/env python3
"""
Скрипт для инспекции модели parakeet-tdt-0.6b-v3.

Извлекает:
1. model_config.yaml из .nemo файла
2. Ключи и формы весов (state_dict)
3. Конвертирует веса в safetensors

Использование:
    # Скачать .nemo файл:
    python scripts/download_model.py --model nvidia/parakeet-tdt-0.6b-v3 --output models/parakeet-tdt-0.6b-v3

    # Инспекция из HuggingFace (через NeMo):
    python scripts/inspect_parakeet.py --from-hf nvidia/parakeet-tdt-0.6b-v3

    # Инспекция из локального .nemo файла:
    python scripts/inspect_parakeet.py --nemo-file models/parakeet-tdt-0.6b-v3.nemo

    # Конвертация весов в safetensors:
    python scripts/inspect_parakeet.py --nemo-file models/parakeet-tdt-0.6b-v3.nemo --convert-safetensors models/parakeet-tdt-0.6b-v3/
"""

import argparse
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path


def inspect_from_nemo(nemo_path: str, convert_dir: str = None):
    """Извлекает и выводит информацию из .nemo файла."""
    import torch

    print(f"\n{'='*60}")
    print(f"Инспекция .nemo файла: {nemo_path}")
    print(f"{'='*60}\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(nemo_path, "r:") as tar:
            names = tar.getnames()

            # Найти prefix
            config_files = [n for n in names if n.endswith("model_config.yaml")]
            assert len(config_files) == 1, f"Ожидается один model_config.yaml, найдено: {config_files}"
            prefix = config_files[0].replace("model_config.yaml", "")

            print(f"Prefix: '{prefix}'")
            print(f"Файлы в архиве ({len(names)}):")
            for name in sorted(names):
                print(f"  {name}")

            # Извлечь конфигурацию
            tar.extract(f"{prefix}model_config.yaml", path=tmpdir)
            config_path = os.path.join(tmpdir, f"{prefix}model_config.yaml" if prefix else "model_config.yaml")

            try:
                from omegaconf import OmegaConf
                config = OmegaConf.load(config_path)
                print(f"\n{'='*60}")
                print("MODEL CONFIG (ключевые параметры):")
                print(f"{'='*60}\n")

                # Препроцессор
                if hasattr(config, 'preprocessor'):
                    pp = config.preprocessor
                    print("Preprocessor:")
                    for key in ['sample_rate', 'window_size', 'window_stride', 'features',
                                'n_fft', 'normalize', 'dither', 'pad_to', 'preemph',
                                'window', 'log', 'log_zero_guard_type', 'log_zero_guard_value',
                                'mag_power', 'mel_norm', 'exact_pad']:
                        if hasattr(pp, key):
                            print(f"  {key}: {getattr(pp, key)}")

                # Энкодер
                if hasattr(config, 'encoder'):
                    enc = config.encoder
                    print("\nEncoder:")
                    for key in ['feat_in', 'feat_out', 'n_layers', 'd_model',
                                'subsampling', 'subsampling_factor', 'subsampling_conv_channels',
                                'ff_expansion_factor', 'self_attention_model', 'n_heads',
                                'conv_kernel_size', 'conv_norm_type', 'att_context_size',
                                'xscaling', 'untie_biases', 'pos_emb_max_len',
                                'dropout', 'dropout_pre_encoder', 'dropout_emb', 'dropout_att',
                                'reduction', 'reduction_factor', 'reduction_position']:
                        if hasattr(enc, key):
                            print(f"  {key}: {getattr(enc, key)}")

                # Model defaults
                if hasattr(config, 'model_defaults'):
                    md = config.model_defaults
                    print("\nModel Defaults:")
                    for key in ['enc_hidden', 'pred_hidden', 'joint_hidden',
                                'tdt_durations', 'num_tdt_durations']:
                        if hasattr(md, key):
                            print(f"  {key}: {getattr(md, key)}")

                # Декодер
                if hasattr(config, 'decoder'):
                    dec = config.decoder
                    print("\nDecoder:")
                    for key in ['_target_', 'normalization_mode', 'random_state_sampling',
                                'blank_as_pad']:
                        if hasattr(dec, key):
                            print(f"  {key}: {getattr(dec, key)}")
                    if hasattr(dec, 'prednet'):
                        pn = dec.prednet
                        print("  Prednet:")
                        for key in ['pred_hidden', 'pred_rnn_layers', 'dropout',
                                    'context_size']:
                            if hasattr(pn, key):
                                print(f"    {key}: {getattr(pn, key)}")

                # Joint
                if hasattr(config, 'joint'):
                    jt = config.joint
                    print("\nJoint:")
                    for key in ['_target_', 'log_softmax', 'preserve_memory',
                                'num_extra_outputs', 'fuse_loss_wer', 'fused_batch_size']:
                        if hasattr(jt, key):
                            print(f"  {key}: {getattr(jt, key)}")
                    if hasattr(jt, 'jointnet'):
                        jn = jt.jointnet
                        print("  Jointnet:")
                        for key in ['joint_hidden', 'activation', 'dropout',
                                    'encoder_hidden', 'pred_hidden']:
                            if hasattr(jn, key):
                                print(f"    {key}: {getattr(jn, key)}")

                # Декодирование
                if hasattr(config, 'decoding'):
                    d = config.decoding
                    print("\nDecoding:")
                    for key in ['strategy', 'model_type', 'durations']:
                        if hasattr(d, key):
                            print(f"  {key}: {getattr(d, key)}")

                # Полный конфиг в файл
                full_config_path = os.path.join(
                    convert_dir if convert_dir else ".",
                    "model_config.yaml"
                )
                if convert_dir:
                    os.makedirs(convert_dir, exist_ok=True)
                    OmegaConf.save(config, full_config_path)
                    print(f"\nПолный конфиг сохранён: {full_config_path}")

            except ImportError:
                print("WARN: omegaconf не установлен, показываем raw YAML")
                with open(config_path) as f:
                    print(f.read())

            # Извлечь веса
            weight_files = [n for n in names if n.endswith(".ckpt")]
            if weight_files:
                weight_file = weight_files[0]
                print(f"\n{'='*60}")
                print(f"Весовой файл: {weight_file}")
                print(f"{'='*60}\n")

                tar.extract(weight_file, path=tmpdir)
                weight_path = os.path.join(tmpdir, weight_file)

                state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)

                print(f"Всего тензоров: {len(state_dict)}")
                total_params = 0
                total_bytes = 0

                # Группировка по компонентам
                components = {}
                for key, tensor in sorted(state_dict.items()):
                    component = key.split(".")[0]
                    if component not in components:
                        components[component] = []
                    components[component].append((key, tensor))
                    total_params += tensor.numel()
                    total_bytes += tensor.numel() * tensor.element_size()

                print(f"Всего параметров: {total_params:,}")
                print(f"Размер в памяти: {total_bytes / 1024**2:.1f} MB")

                for comp_name, tensors in components.items():
                    comp_params = sum(t.numel() for _, t in tensors)
                    print(f"\n--- {comp_name} ({comp_params:,} параметров, {len(tensors)} тензоров) ---")
                    for key, tensor in tensors:
                        print(f"  {key}: {list(tensor.shape)} ({tensor.dtype})")

                # Конвертация в safetensors
                if convert_dir:
                    try:
                        from safetensors.torch import save_file
                        os.makedirs(convert_dir, exist_ok=True)
                        safetensors_path = os.path.join(convert_dir, "model.safetensors")
                        save_file(state_dict, safetensors_path)
                        print(f"\nВеса сохранены в safetensors: {safetensors_path}")
                        print(f"Размер файла: {os.path.getsize(safetensors_path) / 1024**2:.1f} MB")
                    except ImportError:
                        print("\nWARN: safetensors не установлен. pip install safetensors")

            # Извлечь токенизатор
            tokenizer_files = [n for n in names if "tokenizer" in n.lower() or n.endswith(".model")]
            if tokenizer_files and convert_dir:
                os.makedirs(convert_dir, exist_ok=True)
                for tf in tokenizer_files:
                    tar.extract(tf, path=tmpdir)
                    src = os.path.join(tmpdir, tf)
                    dst = os.path.join(convert_dir, os.path.basename(tf))
                    import shutil
                    shutil.copy2(src, dst)
                    print(f"Токенизатор скопирован: {dst}")


def inspect_from_hf(model_name: str, convert_dir: str = None):
    """Загружает модель из HuggingFace и инспектирует."""
    import nemo.collections.asr as nemo_asr

    print(f"\n{'='*60}")
    print(f"Загрузка модели из HuggingFace: {model_name}")
    print(f"{'='*60}\n")

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

    # Конфигурация
    from omegaconf import OmegaConf
    print("MODEL CONFIG:")
    print(OmegaConf.to_yaml(model.cfg))

    # Параметры
    print(f"\n{'='*60}")
    print("ПАРАМЕТРЫ МОДЕЛИ:")
    print(f"{'='*60}\n")

    total_params = 0
    for name, param in model.named_parameters():
        print(f"  {name}: {list(param.shape)} ({param.dtype})")
        total_params += param.numel()

    print(f"\nВсего параметров: {total_params:,}")

    # Буферы
    print(f"\n{'='*60}")
    print("БУФЕРЫ МОДЕЛИ:")
    print(f"{'='*60}\n")

    for name, buf in model.named_buffers():
        print(f"  {name}: {list(buf.shape)} ({buf.dtype})")

    # Конвертация
    if convert_dir:
        import torch
        from safetensors.torch import save_file

        os.makedirs(convert_dir, exist_ok=True)

        state_dict = model.state_dict()
        safetensors_path = os.path.join(convert_dir, "model.safetensors")
        save_file(state_dict, safetensors_path)
        print(f"\nВеса сохранены: {safetensors_path}")

        config_path = os.path.join(convert_dir, "model_config.yaml")
        OmegaConf.save(model.cfg, config_path)
        print(f"Конфиг сохранён: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Инспекция модели Parakeet-TDT")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--nemo-file", type=str, help="Путь к .nemo файлу")
    group.add_argument("--from-hf", type=str, help="Имя модели на HuggingFace")
    parser.add_argument("--convert-safetensors", type=str, default=None,
                        help="Директория для конвертации весов в safetensors")

    args = parser.parse_args()

    if args.nemo_file:
        if not os.path.exists(args.nemo_file):
            print(f"Файл не найден: {args.nemo_file}")
            sys.exit(1)
        inspect_from_nemo(args.nemo_file, args.convert_safetensors)
    else:
        inspect_from_hf(args.from_hf, args.convert_safetensors)


if __name__ == "__main__":
    main()
