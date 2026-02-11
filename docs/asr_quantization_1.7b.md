# Квантование Qwen3-ASR 1.7B (план + протокол измерений)

Цель: уменьшить потребление unified memory на Apple Silicon для модели Qwen3-ASR 1.7B, сохранив качество распознавания на уровне FP16/BF16.

В RustTTS мы решали эту задачу через GGUF-квантование (Candle `QTensor`) и загрузку квантованных весов. Здесь повторяем тот же путь.

## Что именно будем квантовать

Основной потребитель памяти в 1.7B — текстовый декодер (LLM). Квантование даёт максимальную отдачу на больших 2D матрицах.

Рекомендуемая стратегия (минимальный риск деградации):

- Квантовать (rank==2, большие веса):
  - `thinker.model.layers.*.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight`
  - `thinker.model.layers.*.mlp.{gate_proj,up_proj,down_proj}.weight`
- Не квантовать:
  - `*.norm*` (все нормы)
  - мелкие 1D тензоры

Эмбеддинги (`thinker.model.embed_tokens.weight`):
- На первом этапе: оставить FP16/BF16 (самый безопасный путь).
- При необходимости максимального сжатия: добавить отдельный режим квантования эмбеддингов (это может требовать более сложной реализации embedding lookup).

## Схемы квантования

- `q8_0` — основной режим (качество максимально близко к FP16)
- `q6k` — компромисс (если хотим ещё меньше памяти)
- `q4_0` — максимальная экономия памяти, но риск деградации качества выше

## Реализация (шаги)

### Шаг 1 — CLI: конвертация safetensors -> gguf

Добавить команду:

```bash
cargo run -p asr-cli --release -- quantize \
  --input models/qwen3-asr-1.7b \
  --output models/qwen3-asr-1.7b/model-q8_0.gguf \
  --qtype q8_0
```

Пример для `q6k`:

```bash
cargo run -p asr-cli --release -- quantize \
  --input models/qwen3-asr-1.7b \
  --output models/qwen3-asr-1.7b/model-q6k.gguf \
  --qtype q6k
```

Примечание: если веса сохранены шардированно (`model-00001-of-0000N.safetensors` + `model.safetensors.index.json`),
команда автоматически подхватит все шарды.

Параметры:
- `--qtype`: `q8_0|q6k|q4_0`
- `--scope`: какие ключи включать в gguf (по умолчанию `thinker.model.`)
- `--quantize-embeddings`: опционально квантовать `embed_tokens.weight`

### Шаг 2 — загрузка GGUF в декодере

Добавить поддержку загрузки весов декодера из `.gguf` (Candle `quantized_var_builder`).
Использовать квантованные `Linear` для больших матриц.

Пайплайн:
- аудио-энкодер может оставаться на safetensors;
- декодер выбирает GGUF, если он есть в директории модели.

### Шаг 3 — тесты

Минимальный набор:
- smoke: загрузка модели из GGUF
- smoke: транскрибация короткого wav
- регрессия: сравнение первых N токенов (FP vs Q8_0) на одном и том же аудио

### Шаг 4 — измерения памяти/скорости/качества

На macOS удобно использовать `/usr/bin/time -l` (пиковый RSS).

Сценарии:
- FP baseline (safetensors)
- GGUF q8_0
- (опционально) GGUF q6k
- (опционально) GGUF q4_0

Команды:

```bash
/usr/bin/time -l cargo run -p asr-cli --release -- transcribe \
  --model models/qwen3-asr-1.7b --audio <wav> --device metal --decoder-weights safetensors
```

Для замера квантованного декодера (GGUF):

```bash
/usr/bin/time -l cargo run -p asr-cli --release -- transcribe \
  --model models/qwen3-asr-1.7b --audio <wav> --device metal --decoder-weights gguf
```

Если в директории модели лежит несколько `.gguf`, можно явно выбрать файл:

```bash
/usr/bin/time -l cargo run -p asr-cli --release -- transcribe \
  --model models/qwen3-asr-1.7b --audio <wav> --device metal \
  --decoder-weights gguf --decoder-gguf model-q6k.gguf
```

Качество:
- ориентир: WER/CER относительно baseline FP (или относительно эталонной разметки, если есть).

## Таблица результатов (заполнить после прогонов)

| Variant | Weights | Quant scheme | Scope | Embeddings | Device | Peak RSS (MB) | Load (s) | 30s transcribe (s) | CER/WER vs FP | Notes |
|---|---|---|---|---|---|---:|---:|---:|---:|---|
| FP | safetensors | F32 | full | fp16/bf16 | cpu | ~8991 | 8.97 | 27.61 |  | `--decoder-weights safetensors` |
| Q8 | gguf | q8_0 | thinker.model | fp16/bf16 | cpu | ~6173 | 4.67 | 15.44 |  | `--decoder-weights gguf` |
| Q6K | gguf | q6k | thinker.model | fp16/bf16 | cpu | ~5840 | 2.88 | 17.04 |  | `--decoder-weights gguf --decoder-gguf model-q6k.gguf` |
| Q4 | gguf | q4_0 | thinker.model | fp16/bf16 | cpu | ~4471 | 1.09 | 17.37 |  | `--decoder-weights gguf --decoder-gguf model-q4_0.gguf` |
| FP | safetensors | BF16 | full | fp16/bf16 | metal | ~7554 | 6.25 | 5.41 |  | `--decoder-weights safetensors` |
| Q8 | gguf | q8_0 | thinker.model | fp16/bf16 | metal | ~4777 | 3.33 | 5.33 |  | `--decoder-weights gguf` |
| Q6K | gguf | q6k | thinker.model | fp16/bf16 | metal | ~5163 | 1.60 | 6.47 |  | `--decoder-weights gguf --decoder-gguf model-q6k.gguf` |
| Q4 | gguf | q4_0 | thinker.model | fp16/bf16 | metal | ~4899 | 3.92 | 4.08 |  | `--decoder-weights gguf --decoder-gguf model-q4_0.gguf` |

## Пилотный прогон на 0.6B (чтобы проверить пайплайн)

Модель: `models/qwen3-asr-0.6b`.
GGUF собран нашей командой `rustasr quantize` со scope `thinker.model.` и qtype `q8_0`.
Измерение peak RSS выполнено через `/usr/bin/time -l`.

| Variant | Device | Weights | Peak RSS (MB) | Load (s) | 30s transcribe (s) | Notes |
|---|---|---|---:|---:|---:|---|
| FP baseline | cpu | safetensors | ~4959 | 3.61 | 6.43 | самый большой RSS, но быстрее чем q8 на CPU в этом тесте |
| Q8 | cpu | gguf q8_0 | ~3136 | 1.04 | 7.62 | RSS заметно ниже, загрузка быстрее |
| FP baseline | metal | safetensors | ~2792 | 3.30 | 1.38 | baseline |
| Q8 | metal | gguf q8_0 | ~3056 | 0.63 | 1.73 | RSS чуть выше baseline; требуется дальнейшая оптимизация под Metal |

Вывод пилота:
- на CPU квантование уже даёт ощутимую экономию памяти;
- на Metal в текущем виде квантованные линейные слои вероятно работают через f32 и не дают выигрыша по peak RSS.

## Примечания

- На Apple Silicon память единая, поэтому приоритет — Peak RSS и стабильность.
- Слишком агрессивная квантовка может ухудшить language detection и punctuation.
