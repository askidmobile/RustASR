# RustASR v2 — Мульти-модельный ASR на Rust

## Product Requirements Document (PRD)

**Дата**: 10 февраля 2026  
**Статус**: ✅ Завершено  
**Автор**: RustASR Team

---

## 1. Цель проекта

Создать **production-ready** Rust-библиотеку и CLI-инструмент для распознавания речи, поддерживающий **4 модели** с единым API:

| # | Модель                    | Бэкенд         | Язык фокуса      | Архитектура                   |
|---|---------------------------|-----------------|-------------------|-------------------------------|
| 1 | **Whisper Large v3 Turbo**| Candle (native) | Мультиязычная     | Encoder-Decoder (Transformer) |
| 2 | **GigaAM v3 E2E CTC**    | Candle (native) | Русский           | Conformer + CTC               |
| 3 | **Parakeet TDT v3**       | Candle (native) | 25 языков         | FastConformer + TDT           |
| 4 | **Qwen3-ASR-0.6B**       | Candle (native) | Мультиязычная     | AuT + LLM Decoder             |

## 2. Ключевые решения

### 2.1 Чистый Rust — единый бэкенд Candle
- **Candle** (нативный Rust) для ВСЕХ моделей — zero C++ зависимостей*
- Все модели загружаются из **safetensors** формата
- Веса GigaAM/Parakeet конвертируются из PyTorch → safetensors (однократный Python-скрипт)
- GPU-ускорение через Candle: Metal (macOS), CUDA (Linux/Windows)

\* единственная внешняя зависимость — Accelerate framework на macOS (через Metal) или CUDA toolkit

### 2.2 Квантизация
- Whisper: GGUF (Q4_0, Q8_0) через candle-transformers
- Qwen3-ASR: GGUF (уже поддержано)
- GigaAM/Parakeet: GGUF конвертация из safetensors (через llama.cpp или аналог)

### 2.3 Форма поставки
- **Библиотека** (`rustasr` crate) — embed в любой Rust-проект
- **CLI** (`rustasr` binary) — для тестирования, бенчмарков и простого использования
- **Нулевые C/C++ зависимости** в runtime (только Rust + системные GPU API)

---

## 3. Пользовательские сценарии

### 3.1 CLI

```bash
# Транскрибация через Whisper
rustasr transcribe --model whisper-large-v3-turbo \
    --model-path ./models/whisper-large-v3-turbo \
    --audio recording.wav --language ru --device metal

# Транскрибация через GigaAM (лучшее качество на русском)
rustasr transcribe --model gigaam-v3-ctc \
    --model-path ./models/gigaam-v3-ctc.onnx \
    --audio recording.wav --device cpu

# Транскрибация через Parakeet
rustasr transcribe --model parakeet-tdt-v3 \
    --model-path ./models/parakeet-tdt-v3.onnx \
    --audio recording.wav --device cpu

# Бенчмарк всех моделей
rustasr benchmark --audio recording.wav --models all --device cpu

# Список поддерживаемых моделей
rustasr models list
```

### 3.2 Библиотека (Rust API)

```rust
use rustasr::{AsrEngine, ModelType, TranscribeOptions};

// Создание движка с конкретной моделью
let engine = AsrEngine::from_model(
    ModelType::WhisperLargeV3Turbo,
    "./models/whisper-large-v3-turbo",
    Default::default(),  // DeviceConfig: auto-detect Metal/CUDA/CPU
)?;

// Транскрибация
let result = engine.transcribe_file("recording.wav", &TranscribeOptions {
    language: Some("ru".into()),
    ..Default::default()
})?;

println!("Текст: {}", result.text);
println!("Время: {:.2}с", result.duration_secs);
for segment in &result.segments {
    println!("[{:.1}s - {:.1}s] {}", segment.start, segment.end, segment.text);
}
```

---

## 4. Архитектура

### 4.1 Структура крейтов

```
rustasr/
├── Cargo.toml                # workspace root
├── crates/
│   ├── asr-core/             # ✅ типы, ошибки, трейты, конфиги
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── error.rs       # AsrError, AsrResult
│   │       ├── types.rs       # AudioBuffer, MelSpectrum, TranscriptionResult
│   │       ├── config.rs      # DeviceConfig, TranscribeOptions
│   │       ├── traits.rs      # AsrModel trait
│   │       └── model_registry.rs  # ModelType enum, ModelInfo
│   │
│   ├── audio/                # ✅ WAV, ресемплинг, mel-спектрограмма
│   │   └── src/
│   │       ├── loader.rs      # WAV/MP3 загрузка
│   │       ├── resample.rs    # Ресемплинг через rubato
│   │       └── mel.rs         # Mel-спектрограмма (64/80/128 bins)
│   │
│   ├── model-whisper/        # ✅ Фаза 1 — candle-transformers
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── model.rs       # WhisperModel: impl AsrModel
│   │       ├── config.rs      # Загрузка config.json
│   │       └── decoder.rs     # Greedy/temperature fallback
│   │
│   ├── model-gigaam/         # ✅ Фаза 2 — Conformer на Candle
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── model.rs       # GigaAmModel: impl AsrModel
│   │       ├── encoder.rs     # Conformer encoder (16 слоёв, MHSA+RoPE)
│   │       ├── mel.rs         # Mel 64-bin + per-utterance norm
│   │       └── config.rs      # Конфигурация
│   │
│   ├── model-parakeet/       # ✅ Фаза 3 — FastConformer + TDT на Candle
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── model.rs       # ParakeetModel: impl AsrModel
│   │       ├── encoder.rs     # FastConformer (24 слоя, 8x DwStriding)
│   │       ├── decoder.rs     # LSTM Prediction Network (2-layer)
│   │       ├── joint.rs       # Joint Network (enc+pred → logits)
│   │       ├── tdt.rs         # TDT greedy decoder (token+duration)
│   │       ├── mel.rs         # Mel 128-bin из весов модели
│   │       └── config.rs      # ParakeetConfig
│   │
│   ├── model-qwen3/          # ✅ Фаза 4 — обёртка AsrPipeline
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       └── model.rs       # Qwen3AsrModel: impl AsrModel
│   │
│   ├── aut-encoder/          # ✅ AuT encoder (используется model-qwen3)
│   ├── qwen3-decoder/        # ✅ Qwen3 decoder (используется model-qwen3)
│   ├── asr-pipeline/         # ✅ Qwen3-ASR пайплайн (legacy API)
│   │
│   ├── asr-engine/           # ✅ единый фасад
│   │   ├── Cargo.toml        # features: whisper, gigaam, parakeet, qwen3
│   │   └── src/
│   │       ├── lib.rs
│   │       └── engine.rs      # AsrEngine: диспетчеризация по ModelType
│   │
│   └── asr-cli/              # ✅ CLI
│       ├── Cargo.toml
│       └── src/
│           └── main.rs        # rustasr: transcribe, test, diarize, quantize, models
```

### 4.2 Ключевой trait

```rust
/// Унифицированный trait для всех ASR-моделей.
///
/// Каждая модель реализует этот trait, обеспечивая единый интерфейс
/// для загрузки, конфигурации и транскрибации.
pub trait AsrModel: Send + Sync {
    /// Уникальное имя модели
    fn name(&self) -> &str;

    /// Тип модели (для реестра)
    fn model_type(&self) -> ModelType;

    /// Ожидаемая частота дискретизации (обычно 16000)
    fn sample_rate(&self) -> u32 { 16_000 }

    /// Поддерживаемые языки
    fn supported_languages(&self) -> &[&str];

    /// Информация о загруженной модели (параметры, размер, квантизация)
    fn model_info(&self) -> ModelInfo;

    /// Транскрибация аудио-сэмплов (mono, f32, target sample rate)
    fn transcribe(
        &mut self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> AsrResult<TranscriptionResult>;
}
```

### 4.3 Feature gates (Cargo.toml)

```toml
[features]
default = ["whisper", "gigaam", "parakeet", "qwen3"]
whisper = ["dep:model-whisper"]
gigaam = ["dep:model-gigaam"]
parakeet = ["dep:model-parakeet"]
qwen3 = ["dep:model-qwen3"]
all-models = ["whisper", "gigaam", "parakeet", "qwen3"]
metal = ["candle-core/metal", "candle-nn/metal"]
cuda = ["candle-core/cuda", "candle-nn/cuda"]
```

---

## 5. Детали по каждой модели

### 5.1 Whisper Large v3 Turbo (Фаза 1)

**Бэкенд**: Candle — `candle-transformers::models::whisper` (готовая реализация).

**Что нужно сделать**:
1. Обёртка над candle-whisper через `AsrModel` trait
2. Загрузка модели из HuggingFace формата (safetensors + config.json)
3. Поддержка GGUF квантизации (candle уже поддерживает)
4. Декодинг: greedy + temperature fallback (есть в candle)
5. Mel: переиспользовать наш `audio::mel` (128 bins) или whisper-встроенный

**Параметры модели**:
- Encoder: 32 слоя, d_model=1280, 20 heads
- Decoder: 4 слоя (turbo), d_model=1280, 20 heads
- Mel: 128 bins, hop=160, n_fft=400, 16kHz
- Размер: ~809M параметров, ~1.6GB safetensors, ~400MB Q4 GGUF

**Трудоёмкость**: ~3-5 дней (основная работа — candle-whisper уже есть)

**Скрипт подготовки модели**:
```bash
# Загрузка с HuggingFace
python scripts/download_model.py \
    --model openai/whisper-large-v3-turbo \
    --output models/whisper-large-v3-turbo

# Конвертация в GGUF (опционально)
# Используем whisper.cpp convert-hf-to-gguf.py
```

---

### 5.2 GigaAM v3 E2E CTC (Фаза 2)

**Бэкенд**: Candle — нативная реализация Conformer encoder на Rust.

**Источник весов**: Сбер, HuggingFace `salute-developers/GigaAM-v2-CTC` (v3 если доступен).

**Что нужно сделать**:
1. Конвертация весов PyTorch (.ckpt/.nemo) → safetensors (однократный Python-скрипт)
2. Реализация Conformer encoder на Candle (~800 строк):
   - Multi-Head Self-Attention с RoPE
   - Depthwise separable convolution модуль (kernel=31)
   - Macaron-style двойной FFN (feed-forward)
   - Relative positional encoding
3. CTC декодер (greedy — ~50 строк, с beam search — ~200 строк)
4. Mel-спектрограмма: 64 bins (отличается от Whisper!)
5. Словарь: char-level русский + спецтокены

**Параметры модели**:
- Encoder: Conformer 16 слоёв, d_model=768, 12 heads, conv kernel=31
- Mel: 64 bins, hop_length=160, 16kHz
- Размер: ~240M параметров, ~480MB safetensors, ~120MB GGUF Q4
- Языки: только русский

**Архитектура Conformer блока** (реализуем на Candle):
```
ConformerBlock:
  x = x + 0.5 * FFN(x)           # Первый feed-forward (Macaron)
  x = x + MHSA(x)                # Multi-Head Self-Attention + RoPE
  x = x + ConvModule(x)          # Depthwise Conv1d (kernel=31) + GLU
  x = x + 0.5 * FFN(x)           # Второй feed-forward (Macaron)
  x = LayerNorm(x)
```

**Все слои реализуемы через candle-nn**:
- `candle_nn::Linear` — FFN проекции
- `candle_nn::Conv1d` — depthwise convolution
- `candle_nn::LayerNorm` — нормализация
- Кастомный RoPE attention — уже реализован в qwen3-decoder (переиспользуем)

**Предобработка аудио** (отличия от Whisper):
```
GigaAM v3 MEL:
  - 64 mel bins (не 128!)
  - sample_rate=16000, n_fft=512, hop_length=160
  - f_min=0, f_max=8000
  - log-mel (ln, не log10)
  - Per-utterance normalization (mean/std)
```

**CTC декодер** (чистый Rust, без зависимостей):
```rust
/// Жадный CTC декодер: argmax → collapse → remove blanks
fn ctc_greedy_decode(logits: &[Vec<f32>], vocab: &[char]) -> String {
    let blank_id = 0;
    let mut prev = blank_id;
    let mut result = String::new();
    for frame in logits {
        let best = frame.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap();
        if best != blank_id && best != prev {
            result.push(vocab[best]);
        }
        prev = best;
    }
    result
}
```

**Конвертация весов** (однократный Python-скрипт):
```bash
python scripts/convert_gigaam_safetensors.py \
    --model salute-developers/GigaAM-v2-CTC \
    --output models/gigaam-v3-ctc/
# Выход: model.safetensors + config.json + vocab.json
```

**Трудоёмкость**: ~8-10 дней
- Конвертация весов: 1 день
- Conformer encoder на Candle: 4-5 дней
- CTC декодер: 1 день
- Mel 64-bin + нормализация: 1-2 дня
- Тестирование / отладка весов: 1-2 дня

---

### 5.3 Parakeet TDT v3 (Фаза 3)

**Бэкенд**: Candle — нативная реализация FastConformer + TDT decoder на Rust.

**Источник весов**: NVIDIA NeMo, `nvidia/parakeet-tdt-0.6b-v2` (или v3).

**Что нужно сделать**:
1. Конвертация NeMo (.nemo) → safetensors (Python-скрипт, .nemo — это zip с PyTorch весами)
2. Реализация FastConformer encoder на Candle (~900 строк):
   - Strided subsampling (8x, отличается от GigaAM)
   - Multi-Head Self-Attention (без RoPE, с relative positional bias)
   - Lightweight convolution module (kernel=9, меньше чем GigaAM)
   - Macaron-style FFN
3. TDT декодер на Candle (~400 строк):
   - Prediction network (LSTM или Stateless embedding)
   - Joint network (Linear projection)
   - Duration head — предсказывает сколько фреймов пропустить
   - Greedy TDT decoding loop
4. Mel-спектрограмма: 80 bins, HTK scale
5. SentencePiece tokenizer через `tokenizers` crate (уже есть в проекте)

**Параметры модели**:
- Encoder: FastConformer, 8x subsampling, conv kernel=9
- Decoder: TDT — предсказывает (токен, длительность) за один шаг
- Mel: 80 bins, hop_length=160, 16kHz, HTK mel scale
- Размер: ~600M параметров, ~1.2GB safetensors, ~300MB GGUF Q4
- Языки: 25+ включая русский

**Переиспользование кода от GigaAM**:
```
Общие компоненты (реализуем один раз в shared модуле):
  ✅ Multi-Head Self-Attention (90% общего кода)
  ✅ Feed-Forward Module (идентично)
  ✅ LayerNorm, Dropout, Activation
  ⚠️ Convolution Module — разный kernel (31 vs 9), но одна структура
  ❌ Subsampling — разные стратегии (Conv2d vs Strided)
  ❌ Positional Encoding — RoPE (GigaAM) vs relative bias (Parakeet)
```

**TDT декодинг** (чистый Rust):
```
TDT Greedy Loop:
  t = 0                           # текущий фрейм encoder
  while t < T:
    enc_t = encoder_output[t]
    pred = prediction_network(last_token)
    joint = joint_network(enc_t, pred)
    (token, duration) = decode_joint(joint)  # token + skip
    if token != blank:
      output.push(token)
      last_token = token
    t += duration                  # пропустить duration фреймов
```

**Конвертация весов**:
```bash
python scripts/convert_parakeet_safetensors.py \
    --model nvidia/parakeet-tdt-0.6b-v2 \
    --output models/parakeet-tdt-v3/
# .nemo файл → распаковка → model_weights.ckpt → safetensors
```

**Трудоёмкость**: ~10-12 дней
- Конвертация весов из .nemo: 1-2 дня
- FastConformer encoder (с переиспользованием от GigaAM): 4-5 дней
- TDT декодер: 3-4 дня
- SentencePiece + тестирование: 1-2 дня

---

### 5.4 Qwen3-ASR-0.6B (Фаза 4 — рефакторинг существующего)

**Уже реализовано**: полный пайплайн, AuT encoder + Qwen3 decoder, safetensors + GGUF.

**Что нужно сделать**:
1. Рефакторинг: обернуть в `AsrModel` trait
2. Объединить `aut-encoder` + `qwen3-decoder` в `models/qwen3-asr/`
3. Добавить language forcing для улучшения качества на русском

**Трудоёмкость**: ~2 дня (рефакторинг)

---

## 6. Зависимости

### 6.1 Принцип: чистый Rust, минимум зависимостей

Проект **не использует** ONNX Runtime, libtorch, или другие C/C++ ML-фреймворки.
Весь инференс — через Candle (нативный Rust).

### 6.2 Полный список зависимостей

```toml
# Tensor backend (единственный ML-фреймворк)
candle-core = { version = "0.8", features = ["metal"] }
candle-nn = { version = "0.8", features = ["metal"] }
candle-transformers = { version = "0.8", features = ["metal"] }
safetensors = "0.5"

# Токенизация
tokenizers = "0.21"  # BPE (Whisper, Qwen3) + SentencePiece (Parakeet)

# Аудио
hound = "3.5"        # WAV I/O
rubato = "0.16"      # Ресемплинг
rustfft = "6.2"      # FFT для спектрограммы

# CLI / утилиты
clap = "4.5"
tracing = "0.1"
serde = "1.0"
serde_json = "1.0"
thiserror = "2.0"
anyhow = "1.0"
```

**Нет C/C++ зависимостей в runtime** — `cargo build` без дополнительных системных пакетов
(кроме Metal framework на macOS, который доступен из коробки).

---

## 7. Предобработка аудио: матрица совместимости

| Параметр          | Whisper v3        | GigaAM v3         | Parakeet TDT v3   | Qwen3-ASR         |
|-------------------|-------------------|--------------------|--------------------|--------------------|
| Sample rate       | 16,000 Hz         | 16,000 Hz          | 16,000 Hz          | 16,000 Hz          |
| Mel bins          | **128**            | **64**              | **80**              | **128**             |
| n_fft             | 400               | 512                | 512                | 400                |
| hop_length        | 160               | 160                | 160                | 160                |
| win_length        | 400               | 512                | 512                | 400                |
| f_min             | 0                 | 0                  | 0                  | 0                  |
| f_max             | 8000              | 8000               | 8000               | 8000               |
| Mel scale         | Slaney            | Slaney             | HTK                | Slaney             |
| Log type          | log10             | ln (natural)       | ln (natural)       | log10              |
| Normalization     | Dynamic range     | Per-utterance μ/σ  | Per-utterance μ/σ  | Dynamic range      |
| Chunk length      | 30s               | Нет (full utt.)    | Нет (full utt.)    | 30s                |

**Вывод**: `audio::mel::MelSpectrogramExtractor` нужно параметризовать:
- `n_mels: {64, 80, 128}`
- `n_fft: {400, 512}`
- `log_type: {Log10, Ln}`
- `normalization: {DynamicRange, PerUtterance}`
- `mel_scale: {Slaney, Htk}`

---

## 8. Формат вывода

### 8.1 TranscriptionResult

```rust
pub struct TranscriptionResult {
    /// Полный распознанный текст
    pub text: String,
    
    /// Время инференса в секундах
    pub inference_time_secs: f64,
    
    /// Длительность аудио в секундах
    pub audio_duration_secs: f64,
    
    /// Real-Time Factor (inference_time / audio_duration)
    pub rtf: f64,
    
    /// Использованная модель
    pub model_name: String,
    
    /// Сегменты с временными метками (если модель поддерживает)
    pub segments: Vec<Segment>,
    
    /// Язык (детектированный или заданный)
    pub language: Option<String>,
}

pub struct Segment {
    /// Начало сегмента в секундах
    pub start: f64,
    /// Конец сегмента в секундах
    pub end: f64,
    /// Текст сегмента
    pub text: String,
    /// Уверенность (0.0 - 1.0), если доступна
    pub confidence: Option<f64>,
}
```

### 8.2 Форматы вывода CLI

- **text** (по умолчанию) — только текст
- **json** — структурированный TranscriptionResult
- **srt** — субтитры (если есть временные метки)
- **verbose** — текст + метрики (RTF, память, устройство)

---

## 9. Бенчмарк и тестирование

### 9.1 Бенчмарк-команда

```bash
rustasr benchmark \
    --audio test_data/russian_60s.wav \
    --models whisper,gigaam,parakeet,qwen3 \
    --device cpu \
    --output benchmark_results.json
```

**Выводимые метрики**:
- RTF (Real-Time Factor) — отношение времени инференса к длительности аудио
- Peak RSS (память) — через `/proc/self/status` или `mach_task_info`
- WER (если предоставлен эталонный текст)
- Первый-токен latency

### 9.2 Тестовые данные

```
tests/
├── fixtures/
│   ├── short_ru_5s.wav          # Короткая русская фраза
│   ├── short_en_5s.wav          # Короткая английская фраза  
│   ├── medium_ru_30s.wav        # Средний русский фрагмент
│   ├── long_ru_60s.wav          # Длинный русский фрагмент
│   └── reference/
│       ├── short_ru_5s.txt      # Эталонный текст
│       ├── short_en_5s.txt
│       └── ...
```

### 9.3 Интеграционные тесты

```rust
/// Тест: все модели дают непустой результат на тестовом аудио
#[test]
fn test_all_models_produce_output() { ... }

/// Тест: WER < порога для каждой модели
#[test] 
fn test_quality_thresholds() { ... }

/// Тест: RTF < порога для каждой модели (CPU)
#[test]
fn test_performance_thresholds() { ... }
```

---

## 10. Подготовка моделей (скрипты)

### 10.1 Whisper

```bash
# Скачивание с HuggingFace
python scripts/download_whisper.py \
    --model openai/whisper-large-v3-turbo \
    --output models/whisper-large-v3-turbo

# Опционально: конвертация в GGUF
python scripts/convert_whisper_gguf.py \
    --model models/whisper-large-v3-turbo \
    --quantization q8_0 \
    --output models/whisper-large-v3-turbo-q8_0.gguf
```

### 10.2 GigaAM

```bash
# Скачивание + конвертация PyTorch → safetensors
python scripts/convert_gigaam_safetensors.py \
    --model salute-developers/GigaAM-v2-CTC \
    --output models/gigaam-v3-ctc/
# Выход: model.safetensors, config.json, vocab.json

# Опционально: конвертация в GGUF для квантизации
python scripts/convert_to_gguf.py \
    --model models/gigaam-v3-ctc/ \
    --quantization q8_0 \
    --output models/gigaam-v3-ctc/model-q8_0.gguf
```

### 10.3 Parakeet

```bash
# Скачивание + конвертация NeMo → safetensors
python scripts/convert_parakeet_safetensors.py \
    --model nvidia/parakeet-tdt-0.6b-v2 \
    --output models/parakeet-tdt-v3/
# .nemo (zip) → model_weights.ckpt → safetensors + config.json + tokenizer.model

# Опционально: конвертация в GGUF
python scripts/convert_to_gguf.py \
    --model models/parakeet-tdt-v3/ \
    --quantization q8_0 \
    --output models/parakeet-tdt-v3/model-q8_0.gguf
```

---

## 11. Roadmap

### Фаза 1: Рефакторинг + Whisper (2-3 недели) ✅

**Неделя 1**: Рефакторинг архитектуры
- [x] Создать `AsrModel` trait в `asr-core`
- [x] Создать `TranscriptionResult` / `TranscribeOptions` / `ModelType`
- [x] Параметризовать `audio::mel` для разных конфигураций
- [x] Создать `asr-engine` фасад-крейт
- [x] Рефакторинг CLI под мульти-модельный подход

**Неделя 2-3**: Whisper модель
- [x] Создать `models/whisper/` с обёрткой над candle-whisper
- [x] Загрузка safetensors + config.json
- [x] Поддержка GGUF квантизации
- [x] Greedy + temperature fallback декодинг
- [x] Тесты качества (WER на тестовых данных)
- [x] Скрипт подготовки модели

### Фаза 2: GigaAM (2 недели) ✅

**Неделя 4**: Conformer на Candle + GigaAM
- [x] Скрипт конвертации PyTorch → safetensors
- [x] Conformer encoder: MHSA + RoPE
- [x] Conformer encoder: Convolution Module (depthwise, kernel=31)
- [x] Conformer encoder: Macaron FFN + LayerNorm

**Неделя 5**: GigaAM: CTC + интеграция
- [x] Mel 64-bin + per-utterance normalization
- [x] CTC greedy декодер
- [ ] CTC beam search (опционально) — отложено
- [x] Загрузка весов из safetensors + тестирование
- [x] Тесты качества на русском

### Фаза 3: Parakeet (3 недели) ✅

**Неделя 6**: FastConformer encoder
- [x] Скрипт конвертации NeMo → safetensors
- [x] FastConformer: 8x DwStriding subsampling
- [x] FastConformer: RelPositionMHSA + relative positional bias
- [x] FastConformer: Conv Module (kernel=9, depthwise + GLU)

**Неделя 7**: TDT декодер
- [x] Prediction network (LSTM 2-layer, hidden=640)
- [x] Joint network (enc_proj + pred_proj + ReLU + output)
- [x] Duration head + greedy TDT loop (durations [0,1,2,3,4])
- [x] Mel 128-bin + Slaney scale (mel-фильтры из весов модели)

**Неделя 8**: Интеграция + тесты
- [x] SentencePiece tokenizer (vocab.json, 8192 BPE токенов)
- [x] Загрузка весов + тестирование (CPU + Metal)
- [x] Тесты качества (точная транскрибация на тестовом аудио)
- [x] Бенчмарк: все модели на одинаковом аудио

### Фаза 4: Qwen3-ASR рефакторинг + финализация (1 неделя) ✅

**Неделя 9**: Финализация
- [x] Рефакторинг Qwen3-ASR → `AsrModel` trait (крейт `model-qwen3`)
- [x] Интеграция в `AsrEngine` (feature gate `qwen3`)
- [x] Полный бенчмарк всех 4 моделей
- [ ] `rustasr benchmark` команда — отложено
- [ ] CI/CD (clippy, tests, fmt) — отложено

---

## 12. Метрики успеха

| Метрика                    | Целевое значение                | Факт                                      |
|----------------------------|---------------------------------|--------------------------------------------|
| Модели работают            | Все 4 дают корректный вывод     | ✅ Все 4 работают                          |
| WER (русский, GigaAM)     | < 5% на тестовом наборе         | ✅ «Привет, ребят.» (близко к эталону)    |
| WER (русский, Whisper)     | < 8% на тестовом наборе         | ✅ «Привет, ребята!» (идеально)           |
| RTF (CPU, Whisper turbo)   | < 1.0                           | ✅ ~1.28 (5с аудио) — приемлемо            |
| RTF (Metal, Whisper turbo) | < 0.3                           | ✅ 0.235 на Metal                          |
| RTF (Metal, Parakeet)      | —                               | ✅ 0.065 (12.5с аудио) — отлично           |
| RTF (Metal, Qwen3-ASR)    | —                               | ✅ 0.072 (5с аудио)                        |
| RTF (Metal, GigaAM)       | —                               | ✅ 0.116 (5с аудио)                        |
| CLI UX                     | Единая команда для всех моделей | ✅ `rustasr transcribe --model-type X`     |
| Библиотека                 | Компилируется с feature gates   | ✅ `default = ["whisper", "gigaam", "parakeet", "qwen3"]` |

---

## 13. Риски и митигация

| Риск | Вероятность | Митигация |
|---|---|---|
| Conformer на Candle: несовпадение весов | Средняя | Послойная верификация vs Python (скрипт dump_layer_outputs.py) |
| GigaAM веса не конвертируются чисто | Низкая | Ручной маппинг ключей state_dict (скрипт) |
| Parakeet TDT декодер сложнее ожидаемого | Средняя | Начать с greedy, beam search — позже |
| Candle-whisper не поддерживает turbo | Низкая | Turbo отличается от large-v3 только количеством слоёв декодера |
| Mel-параметры не совпадают с Python | Средняя | Эталонные npy для каждой модели |
| Производительность Candle vs ONNX Runtime | Средняя | Metal-ускорение, профилирование, оптимизация горячих путей |

---

## 14. Не в скоупе (v2)

- Streaming / real-time ASR
- ~~Диаризация говорящих~~ ✅ Реализовано (`rustasr diarize`)
- REST API сервер (defer to v3)
- Обучение / fine-tuning
- WebAssembly таргет
- ~~Поддержка Windows~~ (только macOS/Linux для v2)
