# Транскрибация `full.mp3`: сравнение 0.6B vs 1.7B и квантований (GGUF)

Исходный файл:
- MP3: `/Users/askid/Library/Application Support/com.yttri.app/users/askid.mobile_at_gmail.com/recordings/2ba4863a-32b8-4e25-8125-eb8aa990a0f4/full.mp3`
- Длительность: ~265 сек (4:25)

Подготовка входа (конвертация в WAV 16 kHz mono):

```bash
ffmpeg -y -i "<...>/full.mp3" -ac 1 -ar 16000 -c:a pcm_s16le \
  tmp/transcribe_full_mp3/full_16k_mono.wav
```

В этом отчете распознавание делалось CLI `rustasr` на `--device metal`.

## Артефакты

Тексты распознавания (по одному файлу на прогон):
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/full_1.7b_q8_0.txt`
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/full_1.7b_q4_0.txt`
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/full_0.6b_q8_0.txt`
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/full_0.6b_q4_0.txt`

Логи прогонов (stdout+stderr + `/usr/bin/time -l`):
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/logs/full_1.7b_q8_0.log`
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/logs/full_1.7b_q4_0.log`
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/logs/full_0.6b_q8_0.log`
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/logs/full_0.6b_q4_0.log`

## Таблица прогонов (скорость/память)

Примечания:
- `Load (s)` и `Transcribe (s)` взяты из вывода CLI.
- `Max RSS (MB)` взят из `/usr/bin/time -l` (строка `maximum resident set size`).
- Для Q4 видно деградацию: распознавание «зацикливается» и упирается в `max-tokens`.

| Run | Model | Decoder GGUF | Device | Max tokens | Load (s) | Transcribe (s) | Total (s) | Max RSS (MB) | Output |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| 1.7b-q8 | `models/qwen3-asr-1.7b` | `model-q8_0.gguf` | metal | 6000 | 5.70 | 89.78 | 95.49 | 3213 | `docs/transcriptions/full_mp3/full_1.7b_q8_0.txt` |
| 1.7b-q4 | `models/qwen3-asr-1.7b` | `model-q4_0.gguf` | metal | 4096 | 4.27 | 317.16 | 321.44 | 5878 | `docs/transcriptions/full_mp3/full_1.7b_q4_0.txt` |
| 0.6b-q8 | `models/qwen3-asr-0.6b` | `model-q8_0.gguf` | metal | 4096 | 2.27 | 51.09 | 53.37 | 3200 | `docs/transcriptions/full_mp3/full_0.6b_q8_0.txt` |
| 0.6b-q4 | `models/qwen3-asr-0.6b` | `model-q4_0.gguf` | metal | 4096 | 2.03 | 295.57 | 297.61 | 3368 | `docs/transcriptions/full_mp3/full_0.6b_q4_0.txt` |

## Сравнение качества распознавания

Эталонной разметки (ground truth) нет, поэтому сравнение сделано относительно наиболее «здорового» варианта: **1.7B + Q8_0**.

Метрики:
- `Words`: количество слов.
- `Uniq ratio`: доля уникальных слов (грубый индикатор зацикливания/повторов).
- `Word sim`: похожесть по словам (SequenceMatcher) относительно `1.7B Q8_0`.

| Run | Words | Uniq ratio | Word sim vs 1.7B Q8_0 | Итог по качеству |
|---|---:|---:|---:|---|
| 1.7b-q8 | 398 | 0.477 | 1.000 | Лучший/базовый результат, текст связный |
| 0.6b-q8 | 326 | 0.454 | 0.450 | Похоже на смысл, но больше «каши» и пропусков |
| 1.7b-q4 | 975 | 0.155 | 0.287 | Сильно деградирует, уходит в повтор «двадцать…», достигает `max-tokens` |
| 0.6b-q4 | 2042 | 0.001 | 0.002 | Почти сразу зацикливается на «Что говорили?», достигает `max-tokens` |

Вывод по качеству:
- Для этой записи **Q8_0 (GGUF)** выглядит работоспособно и дает осмысленный текст (1.7B заметно лучше 0.6B).
- **Q4_0 в режиме `transcribe` целиком (без VAD) непригоден**: модель уходит в повторяющиеся фразы и не останавливается по смыслу, упираясь в `max-tokens`.
- Для длинных записей лучше использовать VAD + каналы (`rustasr diarize`, см. ниже): в этом режиме `q4_0` становится существенно стабильнее.
- Ранее (до исправления RoPE cache в `qwen3-decoder`) для `1.7B Q4_0` при `--max-tokens 6000` наблюдался runtime-краш (`narrow invalid args ... start: 8192`).

## Команды прогонов (для воспроизведения)

```bash
# 1.7B Q8_0
/usr/bin/time -l cargo run -p asr-cli --release -- transcribe \
  --model models/qwen3-asr-1.7b --audio tmp/transcribe_full_mp3/full_16k_mono.wav \
  --device metal --max-tokens 6000 --decoder-weights gguf --decoder-gguf model-q8_0.gguf \
  --out-text docs/transcriptions/full_mp3/full_1.7b_q8_0.txt

# 1.7B Q4_0
/usr/bin/time -l cargo run -p asr-cli --release -- transcribe \
  --model models/qwen3-asr-1.7b --audio tmp/transcribe_full_mp3/full_16k_mono.wav \
  --device metal --max-tokens 4096 --decoder-weights gguf --decoder-gguf model-q4_0.gguf \
  --out-text docs/transcriptions/full_mp3/full_1.7b_q4_0.txt

# 0.6B Q8_0
/usr/bin/time -l cargo run -p asr-cli --release -- transcribe \
  --model models/qwen3-asr-0.6b --audio tmp/transcribe_full_mp3/full_16k_mono.wav \
  --device metal --max-tokens 4096 --decoder-weights gguf --decoder-gguf model-q8_0.gguf \
  --out-text docs/transcriptions/full_mp3/full_0.6b_q8_0.txt

# 0.6B Q4_0
/usr/bin/time -l cargo run -p asr-cli --release -- transcribe \
  --model models/qwen3-asr-0.6b --audio tmp/transcribe_full_mp3/full_16k_mono.wav \
  --device metal --max-tokens 4096 --decoder-weights gguf --decoder-gguf model-q4_0.gguf \
  --out-text docs/transcriptions/full_mp3/full_0.6b_q4_0.txt
```

## Режим VAD + каналы (рекомендуемый для длинных записей)

Подробное описание режимов диаризации и опций: `docs/diarization.md`.

Ниже тот же `full.mp3`, но распознавание делается через:
- **VAD** (нарезка на сегменты),
- **разделение по каналам**: левый=`mic`, правый=`system`,
- один загруженный пайплайн на все сегменты.

Это снимает главную проблему “распознать всё целиком”: в таком режиме даже `q4_0` перестаёт “уходить в цикл” на всей записи и чаще останавливается на `eos`.

### Подготовка входа (конвертация в WAV 16 kHz stereo)

```bash
ffmpeg -y -i "<...>/full.mp3" -ac 2 -ar 16000 -c:a pcm_s16le \
  tmp/transcribe_full_mp3/full_16k_stereo.wav
```

### Артефакты (VAD/каналы)

Папки с результатами (внутри: `report.md`, `segments.json`, `combined.txt`, `mic.txt`, `system.txt`):
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/diarize_1.7b_q8/`
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/diarize_1.7b_q4/`
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/diarize_0.6b_q8/`
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/diarize_0.6b_q4/`

Логи прогонов (stdout+stderr + `/usr/bin/time -l`):
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/logs/diarize_1.7b_q8.log`
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/logs/diarize_1.7b_q4.log`
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/logs/diarize_0.6b_q8.log`
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/logs/diarize_0.6b_q4.log`

### Таблица прогонов (VAD/каналы)

Примечания:
- `Time real (s)` и `Max RSS (MB)` взяты из `/usr/bin/time -l`.
- `Segments max_tokens` показывает, в скольких сегментах модель не остановилась по `eos` и упёрлась в лимит токенов.

| Run | Model | Decoder GGUF | Device | Segments | Segments max_tokens | Time real (s) | Max RSS (MB) | Output |
|---|---|---|---|---:|---:|---:|---:|---|
| 1.7b-q8 | `models/qwen3-asr-1.7b` | `model-q8_0.gguf` | metal | 102 | 0 | 93.19 | 3259 | `docs/transcriptions/full_mp3/diarize_1.7b_q8/combined.txt` |
| 1.7b-q4 | `models/qwen3-asr-1.7b` | `model-q4_0.gguf` | metal | 102 | 0 | 98.07 | 4088 | `docs/transcriptions/full_mp3/diarize_1.7b_q4/combined.txt` |
| 0.6b-q8 | `models/qwen3-asr-0.6b` | `model-q8_0.gguf` | metal | 102 | 0 | 51.10 | 2917 | `docs/transcriptions/full_mp3/diarize_0.6b_q8/combined.txt` |
| 0.6b-q4 | `models/qwen3-asr-0.6b` | `model-q4_0.gguf` | metal | 102 | 1 | 45.33 | 2708 | `docs/transcriptions/full_mp3/diarize_0.6b_q4/combined.txt` |

### Сравнение качества (VAD/каналы)

Эталонной разметки (ground truth) нет, поэтому:
- сравнение идёт относительно **1.7B + Q8_0** как наиболее “здорового” варианта;
- метрики ниже служат только как грубые индикаторы (не WER).

Метрики:
- `Words`: количество слов в `combined.txt`.
- `Uniq ratio`: доля уникальных слов (индикатор повторов).
- `Word sim`: похожесть по словам (SequenceMatcher) относительно `1.7B Q8_0`.

| Run | Words | Uniq ratio | Word sim vs 1.7B Q8_0 | Итог по качеству |
|---|---:|---:|---:|---|
| 1.7b-q8 | 1530 | 0.288 | 1.000 | Самый читаемый текст, минимальная “каша” |
| 1.7b-q4 | 1541 | 0.289 | 0.517 | Качество ниже Q8, но без зацикливания на всей записи |
| 0.6b-q8 | 1526 | 0.290 | 0.399 | Заметно хуже 1.7B, но стабильно по остановке |
| 0.6b-q4 | 1563 | 0.283 | 0.442 | В целом работает, но есть сегмент, упёршийся в `max_tokens` |

Вывод по VAD/каналам:
- VAD-сегментация существенно повышает “продуктовую” стабильность: **Q4_0 больше не уходит в бесконечный цикл на всей записи**.
- По качеству распознавания **1.7B Q8_0** остаётся лучшим вариантом.
- По памяти на Metal `q4_0` не гарантирует выигрыш (на 1.7B пик RSS выше, чем у `q8_0`).

## VAD/каналы + форсирование языка (важно для качества)

Для `full.mp3` разговор на русском, но в auto-режиме на коротких/шумных сегментах модель иногда ошибается с языком (вплоть до “случайных” языков в `system.txt`).

Рекомендация для этого файла:
- `--speaker-language mic=Russian`
- `--speaker-language system=Russian`

Артефакты прогонов (RU-forced):
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/diarize_ru_1.7b_q8/`
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/diarize_ru_1.7b_q4/`
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/diarize_ru_0.6b_q8/`
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/diarize_ru_0.6b_q4/`

Логи (`/usr/bin/time -l`):
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/logs/diarize_ru_1.7b_q8.log`
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/logs/diarize_ru_1.7b_q4.log`
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/logs/diarize_ru_0.6b_q8.log`
- `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/logs/diarize_ru_0.6b_q4.log`

Таблица прогонов (VAD/каналы, RU-forced):

| Run | Model | Decoder GGUF | Device | Segments | Segments max_tokens | Time real (s) | Max RSS (MB) | Output |
|---|---|---|---|---:|---:|---:|---:|---|
| 1.7b-q8-ru | `models/qwen3-asr-1.7b` | `model-q8_0.gguf` | metal | 102 | 0 | 89.77 | 2949 | `docs/transcriptions/full_mp3/diarize_ru_1.7b_q8/combined.txt` |
| 1.7b-q4-ru | `models/qwen3-asr-1.7b` | `model-q4_0.gguf` | metal | 102 | 2 | 79.27 | 4194 | `docs/transcriptions/full_mp3/diarize_ru_1.7b_q4/combined.txt` |
| 0.6b-q8-ru | `models/qwen3-asr-0.6b` | `model-q8_0.gguf` | metal | 102 | 0 | 52.69 | 2918 | `docs/transcriptions/full_mp3/diarize_ru_0.6b_q8/combined.txt` |
| 0.6b-q4-ru | `models/qwen3-asr-0.6b` | `model-q4_0.gguf` | metal | 102 | 1 | 46.19 | 2706 | `docs/transcriptions/full_mp3/diarize_ru_0.6b_q4/combined.txt` |

Сравнение качества (VAD/каналы, RU-forced):
- `Words`: количество слов в `combined.txt` (без таймкодов).
- `Uniq ratio`: доля уникальных слов (индикатор повторов/зацикливания).
- `Word sim`: похожесть по словам (SequenceMatcher) относительно `1.7B Q8_0 RU-forced`.

| Run | Words | Uniq ratio | Word sim vs 1.7B Q8_0 RU-forced | Итог по качеству |
|---|---:|---:|---:|---|
| 1.7b-q8-ru | 623 | 0.427 | 1.000 | Лучший вариант на этой записи |
| 1.7b-q4-ru | 689 | 0.398 | 0.782 | Хуже Q8, иногда “залипает” на отдельных сегментах |
| 0.6b-q8-ru | 627 | 0.451 | 0.691 | Работает, но заметно больше ошибок чем 1.7B |
| 0.6b-q4-ru | 645 | 0.423 | 0.648 | Самый слабый вариант из RU-forced; Q4 деградирует качество |

Сравнение с “эталоном” из другой модели:
- файл: `/Volumes/Dev/Projects/RustASR/docs/transcriptions/full_mp3/reference_other_model.txt`
- метрика: Word sim vs ref (SequenceMatcher по словам)

| Run | Word sim vs ref |
|---|---:|
| 1.7b-q8-ru | 0.701 |
| 1.7b-q4-ru | 0.649 |
| 0.6b-q8-ru | 0.644 |
| 0.6b-q4-ru | 0.599 |

### Команды прогонов (VAD/каналы)

```bash
# 1.7B Q8_0
/usr/bin/time -l cargo run -p asr-cli --release -- diarize \
  --model models/qwen3-asr-1.7b --audio tmp/transcribe_full_mp3/full_16k_stereo.wav \
  --device metal --decoder-weights gguf --decoder-gguf model-q8_0.gguf \
  --out-dir docs/transcriptions/full_mp3/diarize_1.7b_q8

# 1.7B Q4_0
/usr/bin/time -l cargo run -p asr-cli --release -- diarize \
  --model models/qwen3-asr-1.7b --audio tmp/transcribe_full_mp3/full_16k_stereo.wav \
  --device metal --decoder-weights gguf --decoder-gguf model-q4_0.gguf \
  --out-dir docs/transcriptions/full_mp3/diarize_1.7b_q4

# 0.6B Q8_0
/usr/bin/time -l cargo run -p asr-cli --release -- diarize \
  --model models/qwen3-asr-0.6b --audio tmp/transcribe_full_mp3/full_16k_stereo.wav \
  --device metal --decoder-weights gguf --decoder-gguf model-q8_0.gguf \
  --out-dir docs/transcriptions/full_mp3/diarize_0.6b_q8

# 0.6B Q4_0
/usr/bin/time -l cargo run -p asr-cli --release -- diarize \
  --model models/qwen3-asr-0.6b --audio tmp/transcribe_full_mp3/full_16k_stereo.wav \
  --device metal --decoder-weights gguf --decoder-gguf model-q4_0.gguf \
  --out-dir docs/transcriptions/full_mp3/diarize_0.6b_q4

# Для этой записи (RU-only) рекомендуется форсировать язык по спикерам:
# --speaker-language mic=Russian --speaker-language system=Russian
```
