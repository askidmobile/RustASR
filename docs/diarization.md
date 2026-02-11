# Диаризация в RustASR (VAD + разметка «кто говорит»)

В RustASR диаризация реализована как:
1) VAD-сегментация (нарезка записи на фрагменты речи),
2) назначение «говорящего» (speaker label) каждому сегменту,
3) транскрибация сегментов одним загруженным пайплайном.

Команда: `rustasr diarize`.

## Сценарии

### 1) Stereo: левый канал = микрофон (пользователь), правый = системный звук

Базовый режим: `--speaker-mode channel` (по умолчанию для stereo при `--speaker-mode auto`).

```bash
cargo run -p asr-cli --release -- diarize \
  --model models/qwen3-asr-1.7b \
  --audio tmp/transcribe_full_mp3/full_16k_stereo.wav \
  --device metal \
  --decoder-weights gguf --decoder-gguf model-q8_0.gguf \
  --speaker-mode channel \
  --left-speaker mic --right-speaker system \
  --out-dir docs/transcriptions/example_channel
```

Выход:
- `combined.txt` — склейка по времени
- `report.md` — таблица сегментов (Start/End/Speaker/Lang/Stop/Text)
- `segments.json` — структурированные данные по сегментам
- `mic.txt`, `system.txt` — тексты по каждому speaker label

#### Если в правом канале несколько собеседников (несколько голосов в system)

Включите дополнительную диаризацию внутри правого канала:
- `--right-num-speakers N` — кластеризовать сегменты правого канала на N спикеров
- `--right-speaker-names "alice,bob"` — (опционально) человекочитаемые имена

```bash
cargo run -p asr-cli --release -- diarize \
  --model models/qwen3-asr-1.7b \
  --audio tmp/transcribe_full_mp3/full_16k_stereo.wav \
  --device metal \
  --decoder-weights gguf --decoder-gguf model-q8_0.gguf \
  --speaker-mode channel \
  --left-speaker mic --right-speaker system \
  --right-num-speakers 2 \
  --right-speaker-names "remote0,remote1" \
  --out-dir docs/transcriptions/example_right_cluster
```

В результате вместо `system.txt` появятся, например:
- `system-remote0.txt`
- `system-remote1.txt`

Примечание: это **разделение на кластеры** (например, `system-spk0`, `system-spk1`), а не «узнавание личности» по имени.

### 2) Mono или «смешанные» каналы (голоса в одном канале)

Используйте `--speaker-mode cluster`:

```bash
cargo run -p asr-cli --release -- diarize \
  --model models/qwen3-asr-1.7b \
  --audio tmp/transcribe_full_mp3/full_16k_mono.wav \
  --device metal \
  --decoder-weights gguf --decoder-gguf model-q8_0.gguf \
  --speaker-mode cluster --num-speakers 2 \
  --speaker-names "spk0,spk1" \
  --out-dir docs/transcriptions/example_cluster
```

## Язык (RU+ENG и т.п.)

Модель может ошибаться в автоопределении языка на коротких сегментах/шуме.
В `diarize` доступны:
- `--language <LANG>` — форсировать язык для всех сегментов (пример: `Russian`)
- `--speaker-language SPEAKER=LANG` — форсировать язык для конкретного speaker label

Примеры:

```bash
# Форсируем русский только для микрофона
--speaker-language mic=Russian
```

```bash
# Форсируем английский для всей группы system-* (system-spk0/system-spk1/...)
--speaker-language system=English
```

## Ограничения текущей реализации

- Кластеризация speaker’ов сделана по простому акустическому эмбеддингу (средний log-mel вектор) + k-means.
  Это может путать похожие голоса, короткие сегменты, шумные участки.
- Одновременная речь двух людей **в одном канале** не разделяется (для этого нужен source separation).
- Если вы хотите «узнавание персоналий» (Алиса/Боб) устойчиво, следующий шаг — режим *enrollment*:
  дать эталонные куски голоса для каждого человека и классифицировать сегменты по близости к voiceprint-эмбеддингу.

