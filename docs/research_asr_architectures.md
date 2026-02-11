# Исследование: ASR-архитектуры, Candle Whisper и Rust-экосистема

Дата: Июль 2025

---

## Содержание

1. [Candle-transformers: реализация Whisper](#1-candle-transformers-реализация-whisper)
2. [Rust-экосистема ASR: крейты и реализации](#2-rust-экосистема-asr-крейты-и-реализации)
3. [Архитектуры GigaAM v3 и Parakeet TDT v3](#3-архитектуры-gigaam-v3-и-parakeet-tdt-v3)

---

## 1. Candle-transformers: реализация Whisper

### 1.1 Обзор

Whisper реализован в `candle-transformers` как полноценная модель ASR (encoder-decoder transformer). Код расположен в `candle-transformers/models/whisper/` с модулями:

- `mod.rs` — конфигурация, аудио-константы
- `model.rs` — полная модель (FP32/FP16/BF16)
- `quantized_model.rs` — квантованная модель (GGUF)
- `audio.rs` — mel-спектрограмма, FFT

Пример использования — `candle-examples/examples/whisper/main.rs`.

### 1.2 Поддерживаемые модели

Enum `WhichModel` определяет все поддерживаемые варианты:

| Модель | Параметры | Слои enc/dec | d_model | Heads |
|--------|-----------|-------------|---------|-------|
| Tiny / Tiny-En | 39M | 4/4 | 384 | 6 |
| Base / Base-En | 74M | 6/6 | 512 | 8 |
| Small / Small-En | 244M | 12/12 | 768 | 12 |
| Medium / Medium-En | 769M | 24/24 | 1024 | 16 |
| Large | 1.55B | 32/32 | 1280 | 20 |
| Large-V2 | 1.55B | 32/32 | 1280 | 20 |
| Large-V3 | 1.55B | 32/32 | 1280 | 20 |
| Large-V3 Turbo | 809M | 32/4 | 1280 | 20 |
| Distil-Medium-En | ~394M | 24/2 | 1024 | 16 |
| Distil-Large-V2 | ~756M | 32/2 | 1280 | 20 |
| Distil-Large-V3 | ~756M | 32/2 | 1280 | 20 |

**Квантованные модели** (GGUF) — поддерживается загрузка через `VarBuilder::from_gguf()`, но в HuggingFace-репозиториях pre-quantized доступны только tiny/tiny-en. Для остальных нужна конвертация.

### 1.3 Аудио-препроцессинг

Константы определены в `mod.rs`:

```rust
pub const SAMPLE_RATE: usize = 16000;
pub const N_FFT: usize = 400;          // 25мс окно при 16kHz
pub const HOP_LENGTH: usize = 160;     // 10мс шаг
pub const CHUNK_LENGTH: usize = 30;    // 30 сек — один чанк
pub const N_SAMPLES: usize = 480000;   // CHUNK_LENGTH * SAMPLE_RATE
pub const N_FRAMES: usize = 3000;      // N_SAMPLES / HOP_LENGTH
```

- Mel-фильтры: **80 бинов** (standard) или **128 бинов** (large-v3, large-v3-turbo)
- FFT: собственная реализация (не FFTW), recursive FFT
- Окно: Hann (`hann_window`)
- Масштаб: log10, clamp max, нормализация `(mel + 4.0) / 4.0`

```rust
// audio.rs — ключевые функции
pub fn pcm_to_mel(cfg: &Config, samples: &[f32], filters: &[f32]) -> Result<Vec<f32>>
fn log_mel_spectrogram_(samples: &[f32], filters: &[f32], fft_size: usize, ...) -> Vec<f32>
```

### 1.4 Архитектура модели

**AudioEncoder:**
```
Input mel → Conv1d(n_mels, n_audio_state, kernel=3, pad=1)
         → GELU
         → Conv1d(n_audio_state, n_audio_state, kernel=3, stride=2, pad=1)  // stride 2!
         → GELU
         → positional_embedding (sinusoid)
         → N × ResidualAttentionBlock (self-attn + MLP)
         → layer_norm
```

**TextDecoder:**
```
token_embedding + positional_embedding
→ N × ResidualAttentionBlock (self-attn + cross-attn + MLP)
→ layer_norm
→ final_linear (→ vocab_size) [опционально, иначе weight-tying с token_embedding]
```

**Config struct:**
```rust
pub struct Config {
    pub num_mel_bins: usize,
    pub max_source_positions: usize,  // 1500 (= N_FRAMES/2 после stride)
    pub d_model: usize,               // n_audio_state / n_text_state
    pub encoder_attention_heads: usize,
    pub decoder_attention_heads: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub vocab_size: usize,
    pub suppress_tokens: Vec<u32>,
}
```

### 1.5 Декодирование

Реализовано в `main.rs` (example), **НЕ** в библиотеке:

- **Greedy decoding** (основной режим, temperature = 0.0)
- **Temperature fallback**: если декодирование "плохое" (по сжатию / повторениям), повторяется с temperature 0.2, 0.4, 0.6, 0.8, 1.0
- **Нет beam search** в candle (в отличие от оригинала OpenAI)
- Определение языка: 99 языков, автоматическое определение по softmax на language tokens
- Timestamps: поддержка token-level timestamps через suppress/force token логику

**DecodingResult / Segment:**
```rust
struct DecodingResult {
    tokens: Vec<u32>,
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
}
struct Segment {
    start: f64,
    duration: f64,
    dr: DecodingResult,
}
```

### 1.6 Загрузка моделей

- **SafeTensors**: `Whisper::load(&VarBuilder, config)` — стандартный путь
- **GGUF**: `quantized_model::Whisper::load(vb)` через `VarBuilder::from_gguf(path)`
- Веса скачиваются через `hf_hub` из HuggingFace репозиториев (openai/whisper-* или distil-whisper/*)
- KV-cache: `reset_kv_cache()` для очистки между чанками

### 1.7 Применимость к RustASR

**Что можно переиспользовать:**
- Формат Config / аудио-константы — стандартные для ASR
- Подход к загрузке через VarBuilder (SafeTensors + GGUF)
- Структура encoder (Conv+Transformer) аналогична Conformer
- KV-cache паттерн в decoder
- Temperature fallback для robustness

**Ключевые отличия от Qwen3-ASR:**
- Whisper: encoder-decoder (transformer), 30с chunking
- Qwen3-ASR: AuT encoder + Qwen3 decoder (LLM), нет фиксированных чанков
- Mel bins: Whisper 80/128, Qwen3-ASR другие
- Whisper decoder: авторегрессивный с suppress tokens; Qwen3: LLM prompt-based

---

## 2. Rust-экосистема ASR: крейты и реализации

### 2.1 parakeet-rs (⭐ НАИБОЛЕЕ РЕЛЕВАНТНЫЙ)

**Версия:** 0.3.2 | **Лицензия:** MIT/Apache-2.0 | **Stars:** 161 | **Язык:** 100% Rust

Поддерживает 5 типов моделей NVIDIA через ONNX Runtime:

| Тип | Потоковый | Языки | API |
|-----|-----------|-------|-----|
| **CTC** | Нет | English | `Parakeet::from_pretrained()` |
| **TDT** | Нет | 25 языков | `ParakeetTDT::from_pretrained()` |
| **EOU** | Да (160мс / 2560 сэмплов) | English | `ParakeetEOU::from_pretrained()` |
| **Nemotron** | Да (560мс / 8960 сэмплов) | English | `Nemotron::from_pretrained()` |
| **Sortformer** | Нет | — | Диаризация (до 4 спикеров) |

**TDT модель** скачивается из `istupakov/parakeet-tdt-0.6b-v3-onnx` (HuggingFace):
- `encoder-model.onnx` + `decoder_joint-model.onnx` + `vocab.txt`
- Есть int8 квантованные варианты

**GPU поддержка** через features: `cuda`, `tensorrt`, `webgpu`, `directml`, `migraphx`

**Ограничения:**
- Аудио: только 16kHz mono WAV
- CTC/TDT: ~4-5 минут максимум (full attention)
- Зависимость от ONNX Runtime (не нативные веса)

```rust
// Пример использования TDT
use parakeet_rs::ParakeetTDT;
let model = ParakeetTDT::from_pretrained(None)?;
let result = model.transcribe("audio.wav")?;
println!("{}", result.text);
// Timestamps доступны через result.segments
```

### 2.2 canary-rs

**Версия:** 0.2.0 | **Лицензия:** MIT | **Stars:** 0 (новый, ~3 недели)

Поддержка NVIDIA Canary моделей (ASR + AST — перевод речи):
- **Canary-1b-v2** и **Canary-180m-flash** через ONNX
- ONNX-экспорты от `istupakov` (тот же автор что и parakeet-rs ONNX)
- int8 квантование

```rust
let session = Canary::from_pretrained(CanaryModel::Canary1bV2, None)?;
let result = session.transcribe_file("audio.wav", "en", "en")?;
```

**Ограничения:**
- Timestamps — dummy (Canary не генерирует timestamp-токены; NeMo использует forced alignment с вспомогательной CTC-моделью)
- Windowed streaming helper для живого аудио (нарезка на окна)

### 2.3 whisper-apr

**Версия:** 0.2.2 | **Лицензия:** MIT | **Stars:** 16

Полностью нативная (чистый Rust) реализация Whisper, **WASM-ориентированная**:

- Кастомный формат `.apr` (magic `APR\0`, zstd-сжатие, zero-copy loading, streaming-оптимизация)
- **Int4/Int8 квантование** (4x-8x уменьшение размера модели)
- Поддержка всех размеров Whisper (tiny → large)
- 99 языков, авто-определение
- Greedy + beam search декодирование
- WASM SIMD ускорение, KV-cache оптимизация

**Производительность:**
- Whisper tiny, 30с аудио: **9.2с** (native M1) / **18.5с** (WASM Chrome)

**Масштаб проекта:** ~145K SLoC, 2273 теста

```rust
let model = WhisperModel::load(model_bytes)?;
let result = model.transcribe(&audio_samples, TranscribeOptions::default())?;
```

**Ключевое отличие от candle-whisper:** собственный формат моделей (.apr), заточен под WASM и браузер, не использует candle.

### 2.4 CTC-декодеры на Rust

| Крейт | Версия | Загрузки | Описание |
|--------|--------|----------|----------|
| `ctclib` | 0.1.0 | 1.6K | CTC утилиты (prefix search, beam search). 4+ года, не обновлялся |
| `ctclib-pp` | 0.2.0 | 2.1K | Форк ctclib с поддержкой KenLM perplexity scoring |

**ctclib API:**
```rust
// Greedy (best path)
ctclib::decode_best_path(&probs, &labels, blank_id);
// Beam search
ctclib::decode_beam_search(&probs, &labels, blank_id, beam_width);
```

**Вывод:** CTC-декодирование достаточно простое для своей реализации. Существующие крейты малоактивны и ограничены. Для RustASR имеет смысл реализовать CTC greedy + beam search самостоятельно.

### 2.5 Другие Rust ASR крейты

| Крейт | Версия | Загрузки | Описание |
|--------|--------|----------|----------|
| `april_asr` | 0.1.31 | 10.7K | Оффлайн ASR на основе next-gen Kaldi |
| `sherpa-transducers` | 0.5.5 | 8.7K | Streaming ASR через sherpa-onnx (zipformer-transducer). Low-latency |
| `nvidia_riva` | 0.2.0 | 15K | gRPC клиент к NVIDIA Riva ASR серверу |
| `voirs-recognizer` | 0.1.0-α3 | — | Voice recognition для VoiRS framework |

### 2.6 Сводная таблица Rust ASR

| Крейт | Нативный Rust | Backend | Модели | Потоковый | Русский |
|-------|---------------|---------|--------|-----------|---------|
| candle-whisper | ✅ | Candle | Whisper (все) | Нет | ✅ (99 яз.) |
| whisper-apr | ✅ | Свой | Whisper (.apr) | Нет | ✅ (99 яз.) |
| parakeet-rs | ✅ | ONNX RT | Parakeet TDT/CTC/EOU/Nemotron | Частично | ✅ (TDT, 25 яз.) |
| canary-rs | ✅ | ONNX RT | Canary 1b-v2, 180m-flash | Windowed | ✅ |
| sherpa-transducers | FFI | sherpa-onnx | Zipformer | ✅ | ? |
| april_asr | FFI | Next-gen Kaldi | Kaldi models | ✅ | ? |
| **RustASR (наш)** | **✅** | **Candle** | **Qwen3-ASR** | **Нет** | **✅** |

**Ключевой вывод:** Нативных Rust-реализаций Conformer / GigaAM / NeMo-формата **не существует**. `parakeet-rs` и `canary-rs` используют ONNX Runtime, а не нативный вычислительный граф. Для поддержки GigaAM в RustASR потребуется реализация Conformer encoder + CTC/RNNT decoder с нуля на Candle.

---

## 3. Архитектуры GigaAM v3 и Parakeet TDT v3

### 3.1 GigaAM v3 — Общая информация

**Автор:** Sber / salute-developers | **Лицензия:** MIT | **Бумага:** arxiv:2506.01192 (InterSpeech 2025)

**Версии моделей:**

| Версия | Варианты | Параметры | Предобучение |
|--------|----------|-----------|--------------|
| v1 | ssl, ctc, rnnt | ~220M | SSL |
| v2 | ssl, ctc, rnnt | ~220M | SSL v2 |
| v3 | ssl, ctc, rnnt, **e2e_ctc**, **e2e_rnnt** | ~240M | HuBERT-CTC SSL на 700K часов |

- **E2E варианты** (v3_e2e_ctc, v3_e2e_rnnt) поддерживают пунктуацию и нормализацию текста
- SSL предобучение: masked language modeling + CTC-derived targets от teacher модели
- Chunkwise attention с динамическим размером чанка для поддержки стриминга
- **Превосходит Whisper-large-v3 на 50%** на русском языке (по метрикам из бумаги)

### 3.2 GigaAM v3 — Препроцессинг (из исходного кода)

```python
# preprocess.py — FeatureExtractor defaults
SAMPLE_RATE = 16000

class FeatureExtractor:
    def __init__(self, sample_rate=16000, features=64, **kwargs):
        self.hop_length = sample_rate // 100    # = 160 (10мс)
        self.win_length = sample_rate // 40     # = 400 (25мс)
        self.n_fft = sample_rate // 40          # = 400
        self.center = True
        # features = количество mel-бинов (feat_in из конфига энкодера)
```

| Параметр | Значение | Примечание |
|----------|----------|------------|
| Sample rate | **16000 Hz** | Моно |
| n_mels (mel bins) | **64** | `feat_in` в ConformerEncoder |
| n_fft | **400** | = sample_rate / 40 |
| hop_length | **160** | = sample_rate / 100 (10мс) |
| win_length | **400** | = sample_rate / 40 (25мс) |
| center | **True** | Pad сигнала перед STFT |
| Масштаб | **log** | SpecScaler: `log(mel)` |

**Загрузка аудио:**
```python
# Через ffmpeg subprocess:
ffmpeg -i <input> -f s16le -acodec pcm_s16le -ac 1 -ar 16000 pipe:1
# Затем: float32 / 32768.0
```

**Mel-спектрограмма:** `torchaudio.transforms.MelSpectrogram` → `SpecScaler` (log-масштабирование)

### 3.3 GigaAM v3 — Архитектура Conformer Encoder

```python
class ConformerEncoder:
    feat_in = 64              # Число mel-бинов
    n_layers = 16             # Количество Conformer-слоёв
    d_model = 768             # Размерность скрытого состояния
    n_heads = 16              # Количество голов внимания
    ff_expansion_factor = 4   # Множитель FeedForward (768*4 = 3072)
    conv_kernel_size = 31     # Размер ядра свёртки в ConformerConvolution
    conv_norm_type = "batch_norm"
    self_attention_model = "rotary"  # Rotary Position Embeddings (RoPE)
    subsampling = "conv2d"    # Тип субдискретизации
    subs_kernel_size = 3      # Ядро в субдискретизации
    subsampling_factor = 4    # Суммарное уменьшение длины последовательности
    pos_emb_max_len = 5000    # Максимальная длина для позиционных эмбеддингов
    flash_attn = False        # Поддержка Flash Attention (опционально)
```

**Архитектура слоя (Macaron-style Conformer):**

```
Input
  ├─ FFN1 (Linear → SiLU → Linear) × fc_factor(0.5) + residual
  ├─ Self-Attention (Rotary Multi-Head Attention + residual)
  │   └─ Поддержка: flash_attn, torch.nn.functional.scaled_dot_product_attention
  ├─ ConformerConvolution (pointwise → GLU → depthwise(k=31) → BatchNorm → SiLU → pointwise) + residual
  ├─ FFN2 (Linear → SiLU → Linear) × fc_factor(0.5) + residual
  └─ LayerNorm → Output
```

**Субдискретизация (StridingSubsampling):**
- `log2(subsampling_factor) = 2` слоя Conv2d со stride 2
- Каждый слой: Conv2d → BatchNorm → активация
- Итого: временная ось уменьшается в 4 раза
- Финальный линейный слой проецирует в d_model (768)

**Self-Attention (RotaryPositionMultiHeadAttention):**
- Rotary Position Embeddings (не sinusoidal, не learnable)
- Multi-head: 16 голов, head_dim = 768/16 = 48
- xpos = False (стандартные RoPE)
- Поддержка Flash Attention и SDPA

### 3.4 GigaAM v3 — Декодеры

**CTC Head:**
```python
class CTCHead:
    # Conv1d(feat_in, num_classes, kernel_size=1) → log_softmax
    # feat_in = d_model (768), num_classes = размер словаря
```

**RNNT Decoder:**
```python
class RNNTDecoder:
    # Prediction network: Embedding → LSTM
    # Joint network: Linear(encoder + decoder) → activation → Linear(vocab)
```

**Модельная иерархия:**
```
GigaAM (base)
  ├─ preprocessor (FeatureExtractor)
  └─ encoder (ConformerEncoder)

GigaAMASR(GigaAM)
  ├─ head (CTCHead или RNNTDecoder + RNNTJoint)
  ├─ decoding
  ├─ transcribe()        — для аудио ≤25 секунд
  └─ transcribe_longform() — с pyannote VAD для длинных записей

GigaAMEmo(GigaAM)
  ├─ avg_pool1d → head → softmax
  └─ Классификация эмоций
```

**Загрузка конфигурации модели:**
```python
checkpoint = torch.load(path)
cfg = checkpoint["cfg"]
# Конфиг содержит параметры preprocessor, encoder, head, decoding
# Модель инстанцируется через hydra.utils.instantiate(cfg.preprocessor), и т.д.
```

### 3.5 Parakeet TDT v3 — Общая информация

**Модель:** `nvidia/parakeet-tdt-0.6b-v3` | **Параметры:** 600M | **Лицензия:** CC-BY-4.0

**Основные характеристики:**
- 25 европейских языков (bg, hr, cs, da, nl, **en**, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, sk, sl, es, sv, **ru**, uk)
- Автоматическое определение языка
- Пунктуация и капитализация
- Word-level timestamps
- Технический отчёт: arxiv:2509.14128

### 3.6 Parakeet TDT v3 — Архитектура

**FastConformer Encoder:**

| Параметр | Значение |
|----------|----------|
| Тип энкодера | FastConformer |
| Параметры | 600M (всего) |
| Субдискретизация | 8x depthwise Conv (вместо стандартных 4x strided Conv) |
| Conv kernel size | 9 (вместо 31 у GigaAM) |
| Скорость | ~2.4x быстрее стандартного Conformer |

**TDT Decoder (Token-and-Duration Transducer):**

TDT — расширение RNN-Transducer (RNNT), предложенное в arxiv:2304.06795:

```
Стандартный RNNT:
  Каждый фрейм → предсказание одного токена ИЛИ blank

TDT:
  Каждый фрейм → предсказание (токен, duration)
  - token: какой символ/subword выдать
  - duration: НА СКОЛЬКО ФРЕЙМОВ СДВИНУТЬ указатель (1, 2, ...N)
  - Blank тоже имеет duration
```

**Преимущества TDT:**
- **2.82x ускорение** инференса по сравнению со стандартным RNNT
- Меньше шагов декодирования (пропуск фреймов)
- Естественные timestamps (duration напрямую кодирует время)
- Без потери качества WER

**Формально:**
$$P(y, d | x) = \prod_{t} P(y_t, d_t | x_t, h_t)$$
где $d_t \in \{1, 2, ..., D_{max}\}$ — количество фреймов до следующего шага.

### 3.7 Parakeet TDT v3 — Препроцессинг

| Параметр | Значение |
|----------|----------|
| Sample rate | **16000 Hz** |
| Входной формат | WAV, FLAC (mono) |
| Mel bins | **80** (стандарт NeMo FastConformer) |
| Субдискретизация | **8x** (depthwise Conv) |
| Токенизатор | SentencePiece, unified, **8192 токена** |
| Максимальная длина | 24 мин (full attention, A100 80GB) / до 3 часов (local attention) |
| Декодирование | Greedy TDT (без внешнего LM) |

### 3.8 Parakeet TDT v3 — Обучение

**Данные:**
- **660K часов** pseudo-labeled аудио из Granary корпуса
- **~10K часов** human-transcribed NeMo ASR Set 3.0
- **~7.5K часов** human-transcribed для Stage 2 fine-tuning

**Процесс обучения:**
1. Инициализация из CTC multilingual checkpoint, предобученного на Granary
2. **Stage 1:** 150K шагов на 128 × A100 GPU
3. **Stage 2:** Fine-tuning 5K шагов на 4 × A100 с human-transcribed данными

### 3.9 Parakeet TDT v3 — Производительность (WER)

| Бенчмарк | WER |
|----------|-----|
| LibriSpeech test-clean | 1.93% |
| LibriSpeech test-other | 3.59% |
| Мультиязычные бенчмарки | См. техотчёт arxiv:2509.14128 |

---

## 4. Сравнительная таблица архитектур

| Характеристика | Qwen3-ASR | GigaAM v3 (CTC) | Parakeet TDT v3 | Whisper (large-v3) |
|----------------|-----------|-----------------|-----------------|-------------------|
| **Параметры** | 0.6B / 1.7B | ~240M | 600M | 1.55B |
| **Энкодер** | AuT (Audio Transformer) | Conformer (16 layers) | FastConformer | Transformer |
| **Декодер** | Qwen3 LLM (авторег.) | CTC (Conv1d→log_softmax) | TDT (token+duration) | Transformer (авторег.) |
| **Sample rate** | 16kHz | 16kHz | 16kHz | 16kHz |
| **Mel bins** | 128 | **64** | **80** | 128 |
| **n_fft** | 400 | 400 | N/A (NeMo FilterBank) | 400 |
| **hop_length** | 160 (10мс) | 160 (10мс) | 160 (10мс) | 160 (10мс) |
| **win_length** | 400 (25мс) | 400 (25мс) | 400 (25мс)* | 400 (25мс) |
| **Субдискрет.** | Conv (зав. от AuT) | Conv2d × 2 (4x) | Depthwise Conv (8x) | Conv1d stride 2 (2x) |
| **Attention** | LLM-style | RoPE | Стандартный + Local | Sinusoidal |
| **Conv kernel** | — | 31 | 9 | — |
| **Русский** | ✅ | ✅ (основной) | ✅ (1 из 25) | ✅ (1 из 99) |
| **Стриминг** | Нет | Chunkwise attn | EOU/Nemotron варианты | Нет (30с чанки) |
| **Лицензия** | Apache 2.0 | MIT | CC-BY-4.0 | MIT |
| **Runtime** | PyTorch / Candle | PyTorch / ONNX | NeMo 2.4 / ONNX | PyTorch / Candle / ONNX |

*\* NeMo использует 80-mel FilterBankFeatures с аналогичными параметрами окна*

---

## 5. Выводы для RustASR

### 5.1 Поддержка GigaAM в RustASR

Для нативной (без ONNX) поддержки GigaAM потребуется реализовать:

1. **Mel-спектрограмма** (64 бина) — отличается от текущей (128 бинов для Qwen3-ASR)
   - Можно параметризовать существующий `audio` крейт
   - n_fft, hop_length, win_length — совпадают с Whisper/Qwen3

2. **Conformer Encoder** — новый крейт:
   - StridingSubsampling (Conv2d × 2, stride 2 каждый)
   - ConformerLayer: FFN → SelfAttn(RoPE) → Conv(k=31, BatchNorm) → FFN → LayerNorm
   - 16 слоёв, d_model=768, 16 heads

3. **CTC Decoder** — простой:
   - Conv1d(768, vocab_size, k=1) → log_softmax
   - Greedy: argmax + collapse repeats + remove blank
   - Beam search: опционально (ctclib или свой)

4. **Загрузка весов** — формат PyTorch checkpoint (не SafeTensors):
   - `checkpoint["cfg"]` содержит конфиг
   - `checkpoint["state_dict"]` содержит веса
   - Нужен конвертер PyTorch → SafeTensors или загрузка через pickle/GGUF

### 5.2 Потенциальная архитектура multi-model RustASR

```
asr-core          — общие типы, ошибки
audio             — mel-спектрограмма (параметризованная: 64/80/128 бинов)
conformer         — Conformer/FastConformer энкодер (для GigaAM, Parakeet-like)
ctc-decoder       — CTC greedy/beam search
aut-encoder       — AuT энкодер (для Qwen3-ASR)
qwen3-decoder     — Qwen3 LLM декодер
tdt-decoder       — TDT декодер (для Parakeet-like)
asr-pipeline      — Унифицированный пайплайн
asr-cli           — CLI
```

### 5.3 Приоритеты

1. **Краткосрочные:** Завершить Qwen3-ASR пайплайн (текущая работа)
2. **Среднесрочные:** Параметризовать mel-спектрограмму + добавить CTC decoder → поддержка GigaAM
3. **Долгосрочные:** FastConformer + TDT decoder → поддержка Parakeet-like моделей

### 5.4 Альтернативный путь: ONNX

Вместо нативной реализации можно использовать подход `parakeet-rs`:
- Конвертировать GigaAM в ONNX
- Использовать `ort` (ONNX Runtime для Rust) — зрелый and активный крейт
- **Плюсы:** быстрее разработка, проверенный инференс
- **Минусы:** зависимость от ONNX RT, потеря контроля над оптимизациями, сложнее интегрировать с Candle-экосистемой
