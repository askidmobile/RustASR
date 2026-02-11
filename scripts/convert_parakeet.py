#!/usr/bin/env python3
"""
Конвертация модели Parakeet-TDT из формата NeMo (.nemo) в safetensors.

Скрипт:
1. Скачивает .nemo файл с HuggingFace (или использует локальный)
2. Извлекает model_config.yaml, model_weights.ckpt, tokenizer
3. Определяет точные размерности модели по весам
4. Конвертирует веса в safetensors
5. Создаёт config.json для Rust загрузчика
6. Извлекает SentencePiece токенизатор

Использование:
    # Из HuggingFace (скачивает .nemo):
    python scripts/convert_parakeet.py --model nvidia/parakeet-tdt-0.6b-v3 \
        --output models/parakeet-tdt-0.6b-v3

    # Из локального .nemo файла:
    python scripts/convert_parakeet.py --nemo-file parakeet.nemo \
        --output models/parakeet-tdt-0.6b-v3

    # Только инспекция (без конвертации):
    python scripts/convert_parakeet.py --nemo-file parakeet.nemo --inspect-only
"""

import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

try:
    import torch
except ImportError:
    print("Установите PyTorch: pip install torch")
    sys.exit(1)

try:
    from safetensors.torch import save_file
except ImportError:
    print("Установите safetensors: pip install safetensors")
    sys.exit(1)


def download_nemo_from_hf(model_name: str, output_dir: str) -> str:
    """Скачивает .nemo файл с HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Установите huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    print(f"Ищем .nemo файл для модели: {model_name}")

    # Сначала попробуем найти .nemo файл
    from huggingface_hub import list_repo_files

    files = list(list_repo_files(model_name))
    nemo_files = [f for f in files if f.endswith(".nemo")]

    if not nemo_files:
        print(f"Ошибка: .nemo файл не найден в репозитории {model_name}")
        print(f"Доступные файлы: {files[:20]}...")
        sys.exit(1)

    nemo_file = nemo_files[0]
    print(f"Найден .nemo файл: {nemo_file}")

    os.makedirs(output_dir, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=model_name,
        filename=nemo_file,
        local_dir=output_dir,
    )
    print(f"Скачан: {local_path}")
    return local_path


def extract_nemo(nemo_path: str, tmpdir: str):
    """Извлекает содержимое .nemo архива."""
    print(f"\nИзвлекаем .nemo: {nemo_path}")

    with tarfile.open(nemo_path, "r:") as tar:
        names = tar.getnames()
        print(f"Файлы в архиве ({len(names)}):")
        for name in sorted(names):
            print(f"  {name}")

        # Определяем prefix
        config_files = [n for n in names if n.endswith("model_config.yaml")]
        if not config_files:
            print("Ошибка: model_config.yaml не найден в архиве!")
            sys.exit(1)

        prefix = config_files[0].replace("model_config.yaml", "")

        # Извлекаем всё
        tar.extractall(path=tmpdir)

    return prefix


def parse_config(config_path: str) -> dict:
    """Парсит model_config.yaml и извлекает ключевые параметры."""
    try:
        from omegaconf import OmegaConf
        config = OmegaConf.load(config_path)

        info = {}

        # Препроцессор
        if hasattr(config, 'preprocessor'):
            pp = config.preprocessor
            info['preprocessor'] = {
                'sample_rate': getattr(pp, 'sample_rate', 16000),
                'features': getattr(pp, 'features', 128),
                'window_size': getattr(pp, 'window_size', 0.025),
                'window_stride': getattr(pp, 'window_stride', 0.01),
                'n_fft': getattr(pp, 'n_fft', 512),
                'normalize': str(getattr(pp, 'normalize', 'per_feature')),
                'dither': float(getattr(pp, 'dither', 1e-5)),
                'preemph': float(getattr(pp, 'preemph', 0.97)),
                'pad_to': int(getattr(pp, 'pad_to', 0)),
            }

        # Энкодер
        if hasattr(config, 'encoder'):
            enc = config.encoder
            info['encoder'] = {
                'n_layers': getattr(enc, 'n_layers', 17),
                'd_model': getattr(enc, 'd_model', 512),
                'n_heads': getattr(enc, 'n_heads', 8),
                'ff_expansion_factor': getattr(enc, 'ff_expansion_factor', 4),
                'conv_kernel_size': getattr(enc, 'conv_kernel_size', 9),
                'subsampling': getattr(enc, 'subsampling', 'dw_striding'),
                'subsampling_factor': getattr(enc, 'subsampling_factor', 8),
                'subsampling_conv_channels': getattr(enc, 'subsampling_conv_channels', 256),
                'feat_in': getattr(enc, 'feat_in', 128),
                'self_attention_model': getattr(enc, 'self_attention_model', 'rel_pos'),
                'untie_biases': getattr(enc, 'untie_biases', True),
                'pos_emb_max_len': getattr(enc, 'pos_emb_max_len', 5000),
            }
            # Reduction
            if hasattr(enc, 'reduction') and enc.reduction is not None:
                info['encoder']['reduction'] = str(enc.reduction)
                info['encoder']['reduction_position'] = getattr(enc, 'reduction_position', None)
                info['encoder']['reduction_factor'] = getattr(enc, 'reduction_factor', 2)

        # Декодер
        if hasattr(config, 'decoder'):
            dec = config.decoder
            info['decoder'] = {
                'normalization_mode': str(getattr(dec, 'normalization_mode', 'null')),
            }
            if hasattr(dec, 'prednet'):
                pn = dec.prednet
                info['decoder']['pred_hidden'] = getattr(pn, 'pred_hidden', 640)
                info['decoder']['context_size'] = getattr(pn, 'context_size', 2)

        # Joint
        if hasattr(config, 'joint'):
            jt = config.joint
            info['joint'] = {
                'num_extra_outputs': getattr(jt, 'num_extra_outputs', 5),
            }
            if hasattr(jt, 'jointnet'):
                jn = jt.jointnet
                info['joint']['joint_hidden'] = getattr(jn, 'joint_hidden', 640)
                info['joint']['encoder_hidden'] = getattr(jn, 'encoder_hidden', 512)
                info['joint']['pred_hidden'] = getattr(jn, 'pred_hidden', 640)

        # Decoding
        if hasattr(config, 'decoding'):
            d = config.decoding
            durations = getattr(d, 'durations', [0, 1, 2, 3, 4])
            if hasattr(durations, '__iter__'):
                durations = list(durations)
            info['decoding'] = {
                'durations': durations,
            }

        # Model defaults
        if hasattr(config, 'model_defaults'):
            md = config.model_defaults
            info['model_defaults'] = {}
            for key in ['enc_hidden', 'pred_hidden', 'joint_hidden',
                        'tdt_durations', 'num_tdt_durations']:
                if hasattr(md, key):
                    val = getattr(md, key)
                    if hasattr(val, '__iter__') and not isinstance(val, str):
                        val = list(val)
                    info['model_defaults'][key] = val

        return info

    except ImportError:
        print("WARN: omegaconf не установлен, парсю вручную через PyYAML")
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f) or {}


def infer_dimensions(state_dict: dict) -> dict:
    """Определяет реальные размерности по формам тензоров."""
    dims = {}

    # Encoder d_model — по liner_q первого слоя
    key = "encoder.layers.0.self_attn.linear_q.weight"
    if key in state_dict:
        shape = state_dict[key].shape
        dims['d_model'] = shape[1]  # Linear(d_model, d_model) → [d_model, d_model]
        dims['n_heads_x_d_k'] = shape[0]
        # head_dim по pos_bias_u
        bias_key = "encoder.layers.0.self_attn.pos_bias_u"
        if bias_key in state_dict:
            dims['n_heads'] = state_dict[bias_key].shape[0]
            dims['d_k'] = state_dict[bias_key].shape[1]

    # Число слоёв
    layer_keys = [k for k in state_dict if k.startswith("encoder.layers.") and ".self_attn.linear_q.weight" in k]
    dims['n_layers'] = len(layer_keys)

    # FF expansion
    ff_key = "encoder.layers.0.feed_forward1.linear1.weight"
    if ff_key in state_dict:
        dims['d_ff'] = state_dict[ff_key].shape[0]
        if dims.get('d_model'):
            dims['ff_expansion_factor'] = dims['d_ff'] // dims['d_model']

    # Conv kernel size
    conv_key = "encoder.layers.0.conv.depthwise_conv.weight"
    if conv_key in state_dict:
        dims['conv_kernel_size'] = state_dict[conv_key].shape[2]

    # Subsampling
    sub_key = "encoder.pre_encode.conv.0.weight"
    if sub_key in state_dict:
        dims['subsampling_conv_channels'] = state_dict[sub_key].shape[0]

    # Projection out
    proj_key = "encoder.pre_encode.out.weight"
    if proj_key in state_dict:
        dims['subsampling_feat_in'] = state_dict[proj_key].shape[1]

    # Decoder embeddings
    emb0_key = "decoder.prediction.embeds.0.weight"
    emb1_key = "decoder.prediction.embeds.1.weight"
    if emb0_key in state_dict:
        dims['vocab_size'] = state_dict[emb0_key].shape[0]
        dims['embed_0_dim'] = state_dict[emb0_key].shape[1]
    if emb1_key in state_dict:
        dims['embed_1_dim'] = state_dict[emb1_key].shape[1]
    if dims.get('embed_0_dim') and dims.get('embed_1_dim'):
        dims['pred_hidden'] = dims['embed_0_dim'] + dims['embed_1_dim']

    # Joint
    enc_proj_key = "joint.enc.weight"
    if enc_proj_key in state_dict:
        dims['joint_hidden'] = state_dict[enc_proj_key].shape[0]
        dims['joint_enc_in'] = state_dict[enc_proj_key].shape[1]

    pred_proj_key = "joint.pred.weight"
    if pred_proj_key in state_dict:
        dims['joint_pred_in'] = state_dict[pred_proj_key].shape[1]

    # Joint output
    # Может быть joint.joint_net.1.weight или joint.joint_net.2.weight
    for suffix in ["joint.joint_net.1.weight", "joint.joint_net.2.weight"]:
        if suffix in state_dict:
            dims['joint_output_dim'] = state_dict[suffix].shape[0]
            break

    # CTC head (вспомогательный, не нужен для инференса)
    ctc_key = "ctc_decoder.decoder_layers.0.weight"
    if ctc_key in state_dict:
        dims['ctc_output_dim'] = state_dict[ctc_key].shape[0]

    return dims


def print_weight_summary(state_dict: dict):
    """Выводит сводку весов по компонентам."""
    components = {}
    total_params = 0

    for key, tensor in sorted(state_dict.items()):
        component = key.split(".")[0]
        if component not in components:
            components[component] = {'params': 0, 'count': 0, 'keys': []}
        components[component]['params'] += tensor.numel()
        components[component]['count'] += 1
        components[component]['keys'].append((key, list(tensor.shape), str(tensor.dtype)))
        total_params += tensor.numel()

    print(f"\nВсего параметров: {total_params:,}")
    print(f"Размер в памяти (FP32): {total_params * 4 / 1024**2:.1f} MB\n")

    for comp_name, info in components.items():
        print(f"--- {comp_name} ({info['params']:,} параметров, {info['count']} тензоров) ---")
        for key, shape, dtype in info['keys']:
            print(f"  {key}: {shape} ({dtype})")
        print()


def filter_weights(state_dict: dict) -> dict:
    """Фильтрует веса для инференса: убирает CTC head, буферы BN num_batches."""
    filtered = {}
    skipped = []

    for key, tensor in state_dict.items():
        # Пропускаем CTC head — не нужен для TDT инференса
        if key.startswith("ctc_decoder."):
            skipped.append(key)
            continue

        # Пропускаем num_batches_tracked для BatchNorm
        if key.endswith(".num_batches_tracked"):
            skipped.append(key)
            continue

        filtered[key] = tensor

    if skipped:
        print(f"\nПропущено {len(skipped)} тензоров:")
        for k in skipped:
            print(f"  {k}")

    return filtered


def create_config_json(config_info: dict, dims: dict, output_path: str):
    """Создаёт config.json для загрузки из Rust."""
    # Берём параметры из конфига, перезаписываем реальными из dims
    config = {
        "model_name": "parakeet-tdt-0.6b-v3",
        "model_class": "tdt",
        "sample_rate": 16000,
        "preprocessor": config_info.get('preprocessor', {
            "sample_rate": 16000,
            "features": 128,
            "window_size": 0.025,
            "window_stride": 0.01,
            "n_fft": 512,
            "normalize": "per_feature",
            "dither": 1e-5,
            "preemph": 0.97,
            "pad_to": 0,
        }),
        "encoder": {
            "n_layers": dims.get('n_layers', 17),
            "d_model": dims.get('d_model', 512),
            "n_heads": dims.get('n_heads', 8),
            "d_k": dims.get('d_k', 64),
            "d_ff": dims.get('d_ff', 2048),
            "conv_kernel_size": dims.get('conv_kernel_size', 9),
            "subsampling": "dw_striding",
            "subsampling_factor": 8,
            "subsampling_conv_channels": dims.get('subsampling_conv_channels', 256),
            "feat_in": dims.get('subsampling_feat_in', 4096) // 16 if dims.get('subsampling_feat_in') else 128,
        },
        "decoder": {
            "vocab_size": dims.get('vocab_size', 8193),
            "pred_hidden": dims.get('pred_hidden', 640),
            "embed_0_dim": dims.get('embed_0_dim', 480),
            "embed_1_dim": dims.get('embed_1_dim', 160),
            "context_size": config_info.get('decoder', {}).get('context_size', 2),
            "blank_idx": dims.get('vocab_size', 8193) - 1,
        },
        "joint": {
            "joint_hidden": dims.get('joint_hidden', 640),
            "encoder_hidden": dims.get('joint_enc_in', dims.get('d_model', 512)),
            "pred_hidden": dims.get('joint_pred_in', dims.get('pred_hidden', 640)),
            "output_dim": dims.get('joint_output_dim', 8198),
            "num_classes": dims.get('vocab_size', 8193),
            "num_durations": 5,
        },
        "tdt": {
            "durations": [0, 1, 2, 3, 4],
            "max_symbols_per_step": 10,
        },
    }

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nconfig.json сохранён: {output_path}")

    return config


def convert_nemo(nemo_path: str, output_dir: str, inspect_only: bool = False):
    """Полная конвертация .nemo → safetensors + config.json + tokenizer."""
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = extract_nemo(nemo_path, tmpdir)

        # 1. Парсим конфиг
        config_path = os.path.join(tmpdir, f"{prefix}model_config.yaml")
        config_info = parse_config(config_path)

        print("\n" + "=" * 60)
        print("КОНФИГУРАЦИЯ ИЗ model_config.yaml:")
        print("=" * 60)
        print(json.dumps(config_info, indent=2, default=str))

        # Копируем original config
        shutil.copy2(config_path, os.path.join(output_dir, "model_config.yaml"))

        # 2. Загружаем веса
        weight_file = os.path.join(tmpdir, f"{prefix}model_weights.ckpt")
        if not os.path.exists(weight_file):
            # Ищем любой .ckpt
            import glob
            ckpt_files = glob.glob(os.path.join(tmpdir, "**/*.ckpt"), recursive=True)
            if ckpt_files:
                weight_file = ckpt_files[0]
            else:
                print("Ошибка: .ckpt файл с весами не найден!")
                sys.exit(1)

        print(f"\nЗагружаем веса: {weight_file}")
        state_dict = torch.load(weight_file, map_location="cpu", weights_only=True)

        # 3. Анализ
        print_weight_summary(state_dict)

        dims = infer_dimensions(state_dict)
        print("\n" + "=" * 60)
        print("ОПРЕДЕЛЁННЫЕ РАЗМЕРНОСТИ:")
        print("=" * 60)
        for k, v in sorted(dims.items()):
            print(f"  {k}: {v}")

        if inspect_only:
            print("\n--- Режим инспекции, конвертация не выполняется ---")
            return

        # 4. Фильтруем и конвертируем
        filtered = filter_weights(state_dict)
        safetensors_path = os.path.join(output_dir, "model.safetensors")
        save_file(filtered, safetensors_path)
        file_size = os.path.getsize(safetensors_path) / 1024 ** 2
        print(f"\nmodel.safetensors сохранён: {safetensors_path} ({file_size:.1f} MB)")

        # 5. config.json
        config = create_config_json(config_info, dims, os.path.join(output_dir, "config.json"))

        # 6. Токенизатор
        tokenizer_src = None
        for candidate in [
            os.path.join(tmpdir, f"{prefix}tokenizer.model"),
            os.path.join(tmpdir, f"{prefix}tokenizer_spe_bpe_v128/tokenizer.model"),
        ]:
            if os.path.exists(candidate):
                tokenizer_src = candidate
                break

        # Поиск рекурсивно
        if tokenizer_src is None:
            import glob
            tok_files = glob.glob(os.path.join(tmpdir, "**/tokenizer.model"), recursive=True)
            if tok_files:
                tokenizer_src = tok_files[0]

        if tokenizer_src:
            dst = os.path.join(output_dir, "tokenizer.model")
            shutil.copy2(tokenizer_src, dst)
            print(f"Токенизатор скопирован: {dst}")

            # Извлекаем vocab если sentencepiece доступен
            try:
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor()
                sp.Load(tokenizer_src)

                vocab = {}
                for i in range(sp.GetPieceSize()):
                    piece = sp.IdToPiece(i)
                    score = sp.GetScore(i)
                    vocab[piece] = {"id": i, "score": score}

                vocab_path = os.path.join(output_dir, "vocab.json")
                with open(vocab_path, 'w', encoding='utf-8') as f:
                    json.dump(vocab, f, ensure_ascii=False, indent=1)
                print(f"vocab.json сохранён: {vocab_path} ({sp.GetPieceSize()} токенов)")

            except ImportError:
                print("WARN: sentencepiece не установлен — vocab.json не создан")
        else:
            print("WARN: tokenizer.model не найден в архиве")

        # 7. Сводка
        print("\n" + "=" * 60)
        print("ГОТОВО! Файлы в", output_dir + ":")
        print("=" * 60)
        for f in sorted(os.listdir(output_dir)):
            fpath = os.path.join(output_dir, f)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath) / 1024 ** 2
                print(f"  {f}: {size:.1f} MB" if size > 1 else f"  {f}: {os.path.getsize(fpath)} bytes")


def main():
    parser = argparse.ArgumentParser(
        description="Конвертация Parakeet-TDT из NeMo (.nemo) в safetensors"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Имя модели на HuggingFace (например, nvidia/parakeet-tdt-0.6b-v3)"
    )
    parser.add_argument(
        "--nemo-file", type=str, default=None,
        help="Путь к локальному .nemo файлу"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Директория для выходных файлов"
    )
    parser.add_argument(
        "--inspect-only", action="store_true",
        help="Только инспекция, без конвертации"
    )

    args = parser.parse_args()

    if not args.model and not args.nemo_file:
        parser.error("Укажите --model или --nemo-file")

    # Скачиваем .nemo если нужно
    if args.model:
        nemo_path = download_nemo_from_hf(args.model, args.output)
    else:
        nemo_path = args.nemo_file
        if not os.path.exists(nemo_path):
            print(f"Файл не найден: {nemo_path}")
            sys.exit(1)

    convert_nemo(nemo_path, args.output, args.inspect_only)


if __name__ == "__main__":
    main()
