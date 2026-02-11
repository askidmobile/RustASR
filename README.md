# RustASR

**Мульти-модельное распознавание речи на Rust** — локально, быстро, без Python-зависимостей.

[![Rust](https://img.shields.io/badge/Rust-1.85+-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg)](LICENSE)

---

## Обзор

RustASR — библиотека и CLI-инструмент для автоматического распознавания речи (ASR, Automatic Speech Recognition), написанные на чистом Rust.
Все модели работают через [Candle](https://github.com/huggingface/candle) — zero C++/Python зависимостей в runtime.

### Поддерживаемые модели

| Модель | Архитектура | Языки | Параметры |
|--------|-------------|-------|-----------|
| **[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)** | AuT Encoder + Qwen3 LLM Decoder | Мультиязычная | 0.6B / 1.7B |
| **[Whisper Large v3 Turbo](https://huggingface.co/openai/whisper-large-v3-turbo)** | Encoder-Decoder (Transformer) | Мультиязычная | ~809M |
| **[GigaAM v3 E2E CTC](https://huggingface.co/salute-developers/GigaAM-v2-CTC)** | Conformer + CTC | Русский | ~220M |
| **[Parakeet TDT v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)** | FastConformer + TDT | 25 языков | ~627M |

### Ключевые возможности

- **4 модели** с единым API (trait `AsrModel`)
- **GPU-ускорение**: Metal (macOS), CUDA (Linux/Windows)
- **Квантизация**: GGUF (Q4_0, Q6K, Q8_0) для уменьшения потребления памяти
- **Диаризация**: определение говорящих по каналам или кластеризацией
- **VAD**: сегментация речи через WebRTC VAD
- **Форматы аудио**: WAV (16-bit PCM), автоматический ресемплинг до 16 кГц

---

## Быстрый старт

### Требования

- Rust 1.85+ (stable)
- macOS 13+ (для Metal) или Linux 5.0+

### Сборка

```bash
# Клонирование
git clone https://github.com/example/rustasr.git
cd rustasr

# Сборка (debug)
cargo build --workspace

# Сборка (release, рекомендуется для инференса)
cargo build --workspace --release
```

### Загрузка моделей

Qwen3 и Whisper загружаются напрямую с HuggingFace, GigaAM и Parakeet требуют конвертации весов.

```bash
# Qwen3-ASR 0.6B
python scripts/download_model.py \
    --model Qwen/Qwen3-ASR-0.6B \
    --output models/qwen3-asr-0.6b

# Qwen3-ASR 1.7B
python scripts/download_model.py \
    --model Qwen/Qwen3-ASR-1.7B \
    --output models/qwen3-asr-1.7b

# Whisper Large v3 Turbo
python scripts/download_model.py \
    --model openai/whisper-large-v3-turbo \
    --output models/whisper-large-v3-turbo

# GigaAM v3 E2E CTC (загрузка + конвертация PyTorch → safetensors)
python scripts/convert_gigaam.py --output models/gigaam-v3-e2e-ctc

# Parakeet TDT 0.6b v3 (загрузка + конвертация NeMo → safetensors)
python scripts/convert_parakeet.py \
    --model nvidia/parakeet-tdt-0.6b-v3 \
    --output models/parakeet-tdt-0.6b-v3
```

> **Python-зависимости для конвертации**: `torch`, `safetensors`, `sentencepiece`, `huggingface_hub`.
> Для Parakeet дополнительно нужен `omegaconf`.

### Транскрибация

```bash
# Qwen3-ASR (по умолчанию)
cargo run -p asr-cli --release -- transcribe \
    --model models/qwen3-asr-0.6b \
    --audio recording.wav \
    --device metal

# Whisper
cargo run -p asr-cli --release -- transcribe \
    --model models/whisper-large-v3-turbo \
    --model-type whisper \
    --audio recording.wav \
    --device metal

# GigaAM (русский)
cargo run -p asr-cli --release -- transcribe \
    --model models/gigaam-v3-e2e-ctc \
    --model-type gigaam \
    --audio recording.wav \
    --device cpu

# Parakeet TDT
cargo run -p asr-cli --release -- transcribe \
    --model models/parakeet-tdt-0.6b-v3 \
    --model-type parakeet \
    --audio recording.wav \
    --device cpu
```

---

## Использование как библиотеки

```rust
use asr_engine::AsrEngine;
use asr_core::{ModelType, TranscribeOptions};

// Загрузка модели
let mut engine = AsrEngine::load(
    ModelType::Whisper,
    "models/whisper-large-v3-turbo",
    &device,
)?;

// Транскрибация
let result = engine.transcribe(&samples, &TranscribeOptions {
    language: Some("ru".into()),
    ..Default::default()
})?;

println!("{}", result.text);
for seg in &result.segments {
    println!("[{:.1}s–{:.1}s] {}", seg.start, seg.end, seg.text);
}
```

---

## CLI-команды

### `transcribe` — распознавание речи

```bash
rustasr transcribe \
    --model <путь> \
    --model-type <qwen3|whisper|gigaam|parakeet> \
    --audio <файл.wav> \
    --device <cpu|metal|cuda> \
    --language <язык> \
    --decoder-weights <auto|safetensors|gguf>
```

### `diarize` — диаризация + транскрибация

Работает с **любой** моделью (Qwen3, Whisper, GigaAM, Parakeet).

```bash
rustasr diarize \
    --model <путь> \
    --model-type <qwen3|whisper|gigaam|parakeet> \
    --audio <файл.wav> \
    --device <cpu|metal|cuda> \
    --speaker-mode <auto|channel|cluster> \
    --num-speakers 2 \
    --out-dir output/
```

### `quantize` — квантизация модели

```bash
rustasr quantize \
    --input models/qwen3-asr-0.6b/model.safetensors \
    --output models/qwen3-asr-0.6b/model-q8_0.gguf \
    --qtype q8_0
```

### `models list` — список локальных моделей

```bash
rustasr models list --root models/
```

### `test` — проверка окружения

```bash
rustasr test --device metal
```

---

## Архитектура проекта

```
rustasr/
├── crates/
│   ├── asr-core/          # Базовые типы, ошибки, trait AsrModel
│   ├── audio/             # WAV-загрузка, ресемплинг, mel-спектрограмма
│   ├── aut-encoder/       # AuT аудио-энкодер (Qwen3-ASR)
│   ├── qwen3-decoder/     # Qwen3 LLM декодер (Qwen3-ASR)
│   ├── asr-pipeline/      # End-to-end пайплайн Qwen3-ASR
│   ├── model-qwen3/       # Обёртка Qwen3-ASR → AsrModel
│   ├── model-whisper/     # Whisper Large v3 Turbo → AsrModel
│   ├── model-gigaam/      # GigaAM v3 CTC → AsrModel
│   ├── model-parakeet/    # Parakeet TDT v3 → AsrModel
│   ├── asr-engine/        # Единый фасад AsrEngine (диспетчеризация)
│   └── asr-cli/           # CLI-приложение rustasr
├── models/                # Локальные модели (не в git)
├── scripts/               # Утилиты: загрузка, конвертация, сравнение
└── docs/                  # Документация и PRD
```

### Граф зависимостей

```
asr-core  ←── audio  ←── aut-encoder
    ↑              ↑         ↑
    │              └── asr-pipeline ←── qwen3-decoder
    │                       ↑
    ├── model-whisper       │
    ├── model-gigaam        │
    ├── model-parakeet      │
    ├── model-qwen3 ────────┘
    │
    └── asr-engine ←── asr-cli
```

### Единый trait для моделей

Все модели реализуют `AsrModel` из `asr-core`:

```rust
pub trait AsrModel: Send + Sync {
    fn name(&self) -> &str;
    fn model_type(&self) -> ModelType;
    fn sample_rate(&self) -> u32 { 16_000 }
    fn supported_languages(&self) -> &[&str];
    fn model_info(&self) -> ModelInfo;
    fn transcribe(
        &mut self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> AsrResult<TranscriptionResult>;
}
```

---

## Разработка

### Форматирование и линт

```bash
# Форматирование
cargo fmt --all

# Линт (clippy)
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

### Тесты

```bash
# Все тесты
cargo test --workspace

# Конкретный крейт
cargo test -p audio

# Конкретный тест с выводом
cargo test -p asr-pipeline test_pipeline_creation -- --nocapture
```

> Часть интеграционных тестов пропускается, если модель не загружена (проверяется наличие файлов в `models/`).

### Feature gates

Модели подключаются через feature gates в `asr-engine`:

```toml
[features]
default = ["whisper", "gigaam", "parakeet", "qwen3"]
whisper = ["dep:model-whisper"]
gigaam  = ["dep:model-gigaam"]
parakeet = ["dep:model-parakeet"]
qwen3   = ["dep:model-qwen3"]
```

---

## Целевые метрики

| Метрика | Значение | Платформа |
|---------|----------|-----------|
| RTF (Real-Time Factor) | < 0.3 | M1 Mac (Metal), 0.6B |
| RTF | < 1.0 | CPU (8 cores), 0.6B |
| Cold Start | < 5 секунд | Любая |
| Peak Memory | < 2 GB | 0.6B модели |
| Peak Memory | < 3 GB | 1.7B модели |

---

## Результаты тестирования

> Подробный сравнительный тест: [docs/tests/SUMMARY.md](docs/tests/SUMMARY.md)

### Производительность на Metal GPU (60 с русской речи)

| Модель | RTF | Transcribe | Peak RSS | Качество (рус.) |
|--------|-----|------------|----------|-----------------|
| **GigaAM v3 CTC** | **0.017** | 1.02 с | 1 719 МБ | ★★★★☆ |
| **Parakeet TDT v3** | 0.038 | 2.30 с | 4 672 МБ | ★☆☆☆☆ |
| **Whisper v3 Turbo** | 0.110 | 6.60 с | 1 711 МБ | ★★★★★ |
| **Qwen3-ASR 0.6B** | 0.114 | 6.84 с | 1 932 МБ | ★★★☆☆ |

### Рекомендации

| Сценарий | Модель |
|----------|--------|
| Русский, лучшее качество | Whisper v3 Turbo |
| Русский, макс. скорость | GigaAM v3 CTC (Metal) |
| Мультиязычный контент | Whisper v3 Turbo |
| VAD/диаризация | Любая (`--model-type`) |

Также доступны [результаты квантизации GGUF](docs/tests/quantization.md) (Qwen3, Whisper).

---

## Стек технологий

| Компонент | Библиотека |
|-----------|-----------|
| ML-бэкенд | [Candle](https://github.com/huggingface/candle) (`candle-core`, `candle-nn`, `candle-transformers`) |
| Формат весов | SafeTensors, GGUF |
| Токенизация | [tokenizers](https://github.com/huggingface/tokenizers) |
| Аудио I/O | [hound](https://crates.io/crates/hound) (WAV) |
| Ресемплинг | [rubato](https://crates.io/crates/rubato) |
| FFT | [rustfft](https://crates.io/crates/rustfft) |
| VAD | [webrtc-vad](https://crates.io/crates/webrtc-vad) |
| CLI | [clap](https://crates.io/crates/clap) |
| Логирование | [tracing](https://crates.io/crates/tracing) |

---

## Документация

- [Product Requirements Document](docs/PRD.md)
- [Мульти-модельный PRD](docs/MULTI_MODEL_PRD.md)
- [Фаза 1: Инфраструктура](docs/PHASE_1_INFRASTRUCTURE.md)
- [Фаза 2: Обработка аудио](docs/PHASE_2_FEATURE_EXTRACTOR.md)
- [Фаза 3: AuT Энкодер](docs/PHASE_3_AUT_ENCODER.md)
- [Фаза 4: Интеграция](docs/PHASE_4_INTEGRATION.md)
- [Исследование архитектур ASR](docs/research_asr_architectures.md)

## Лицензия

MIT OR Apache-2.0
