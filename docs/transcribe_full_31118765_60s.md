# Сравнение: Qwen3-ASR (HF `qwen-asr`, Python) vs RustASR (Rust) на 60 сек

Исходный файл:
- MP3: `/Users/askid/Library/Application Support/com.yttri.app/users/askid.mobile_at_gmail.com/recordings/31118765-48e9-449b-ac59-e742205225da/full.mp3`
- Вырезка: первые 60 секунд

## Подготовка аудио

```bash
ffmpeg -y -i "<...>/full.mp3" -t 60 -ac 1 -ar 16000 -c:a pcm_s16le \
  tmp/hf_compare_31118765/full_16k_mono_60s.wav
```

## Артефакты

Тексты распознавания:
- HF (Python `qwen-asr`): `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_31118765_60s/hf_1.7b_auto.txt`
- RustASR (1.7B, safetensors, auto): `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_31118765_60s/rust_1.7b_fp_auto.txt`
- RustASR (1.7B, GGUF q8_0, forced Russian): `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_31118765_60s/rust_1.7b_q8_ru.txt`

Логи прогонов (stdout+stderr + `/usr/bin/time -l`):
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_31118765_60s/logs/hf_1.7b_auto.log`
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_31118765_60s/logs/rust_1.7b_fp_auto.log`
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_31118765_60s/logs/rust_1.7b_q8_ru.log`

## Команды прогонов

### HF (Python, `qwen-asr`)

Использовался локальный модельный каталог `models/qwen3-asr-1.7b` (без скачивания).

```bash
/usr/bin/time -l tmp/hf_compare_31118765/.venv-qwen/bin/python \
  tmp/hf_compare_31118765/qwen_asr_transcribe.py \
  --model models/qwen3-asr-1.7b \
  --audio tmp/hf_compare_31118765/full_16k_mono_60s.wav \
  --language auto \
  --device mps \
  --dtype f16 \
  --max-new-tokens 1024 \
  --out-text docs/transcriptions/full_31118765_60s/hf_1.7b_auto.txt
```

### RustASR (Rust)

```bash
# 1) safetensors (FP baseline), auto language
/usr/bin/time -l cargo run -p asr-cli --release -- transcribe \
  --model models/qwen3-asr-1.7b \
  --audio tmp/hf_compare_31118765/full_16k_mono_60s.wav \
  --device metal \
  --max-tokens 1024 \
  --decoder-weights safetensors \
  --out-text docs/transcriptions/full_31118765_60s/rust_1.7b_fp_auto.txt

# 2) GGUF q8_0, forced Russian
/usr/bin/time -l cargo run -p asr-cli --release -- transcribe \
  --model models/qwen3-asr-1.7b \
  --audio tmp/hf_compare_31118765/full_16k_mono_60s.wav \
  --device metal \
  --max-tokens 1024 \
  --language Russian \
  --decoder-weights gguf --decoder-gguf model-q8_0.gguf \
  --out-text docs/transcriptions/full_31118765_60s/rust_1.7b_q8_ru.txt
```

## Таблица сравнения (время/память)

Примечания:
- `Time real (s)` и `Max RSS (MB)` — из `/usr/bin/time -l`.
- Для `Max RSS (MB)` использовано деление `bytes / 1024 / 1024`.

| Run | Реализация | Model | Weights | Device | Time real (s) | Max RSS (MB) | Output |
|---|---|---|---|---|---:|---:|---|
| hf-auto | HF `qwen-asr` | 1.7B | transformers FP16 | mps | 47.96 | 3752 | `docs/transcriptions/full_31118765_60s/hf_1.7b_auto.txt` |
| rust-fp-auto | RustASR | 1.7B | safetensors (BF16) | metal | 19.51 | 5943 | `docs/transcriptions/full_31118765_60s/rust_1.7b_fp_auto.txt` |
| rust-q8-ru | RustASR | 1.7B | gguf q8_0 | metal | 22.74 | 4703 | `docs/transcriptions/full_31118765_60s/rust_1.7b_q8_ru.txt` |

## Вывод по качеству (этот 60s клип)

HF vs RustASR совпадают очень близко по смыслу и ошибкам (различия точечные, типа “пятьничных встреч” vs “пять нечисленных требований”).

Грубая метрика похожести (SequenceMatcher по словам):
- HF vs RustASR FP: ~0.962
- HF vs RustASR Q8: ~0.954

То есть **Rust-реализация в целом воспроизводит эталонный Python-инференс `qwen-asr`** на этом клипе; если качество модели в целом не устраивает по сравнению с Whisper/GigaAM/Parakeet, это уже вопрос выбора модели/архитектуры, а не “поломки” Rust пайплайна.
