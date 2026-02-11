#!/usr/bin/env python3
"""
Инспекция safetensors файлов для понимания структуры весов.

Использование:
    python inspect_safetensors.py ../models/qwen3-asr-0.6b/model.safetensors
"""

import argparse
import json
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    print("❌ Необходимо установить safetensors:")
    print("   pip install safetensors")
    exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Инспекция safetensors файлов"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Путь к .safetensors файлу или директории с файлами"
    )
    parser.add_argument(
        "--filter", "-f",
        type=str,
        default=None,
        help="Фильтр по имени тензора (substring match)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Сохранить JSON со структурой в файл"
    )
    
    args = parser.parse_args()
    path = Path(args.path)
    
    # Собираем все safetensors файлы
    if path.is_dir():
        files = sorted(path.glob("*.safetensors"))
    else:
        files = [path]
    
    if not files:
        print(f"❌ Safetensors файлы не найдены: {path}")
        return 1
    
    all_tensors = {}
    
    for file_path in files:
        print(f"\n📦 Файл: {file_path.name}")
        print("=" * 60)
        
        with safe_open(str(file_path), framework="pt", device="cpu") as f:
            for name in sorted(f.keys()):
                if args.filter and args.filter not in name:
                    continue
                    
                tensor = f.get_tensor(name)
                shape = list(tensor.shape)
                dtype = str(tensor.dtype)
                size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
                
                print(f"  {name}")
                print(f"    Shape: {shape}, Dtype: {dtype}, Size: {size_mb:.2f} MB")
                
                all_tensors[name] = {
                    "shape": shape,
                    "dtype": dtype,
                    "file": file_path.name,
                }
    
    # Анализ структуры
    print("\n" + "=" * 60)
    print("📊 Статистика:")
    
    # Группировка по префиксам
    prefixes = {}
    for name in all_tensors:
        prefix = name.split(".")[0]
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(name)
    
    for prefix, names in sorted(prefixes.items()):
        print(f"  {prefix}: {len(names)} тензоров")
    
    print(f"\n  Всего: {len(all_tensors)} тензоров")
    
    # Сохранение JSON
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_tensors, f, indent=2)
        print(f"\n💾 Структура сохранена: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
