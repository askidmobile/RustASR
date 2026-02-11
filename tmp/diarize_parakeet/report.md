# Диаризация (VAD) + транскрибация по сегментам

- Model: `models/parakeet-tdt-0.6b-v3`
- Audio: `tmp/hf_compare_31118765/full_16k_mono_60s.wav`
- Device: `metal`
- Decoder: weights=`auto` gguf=`-`
- Language: `auto`
- Speaker language overrides: `-`
- Speaker mode: `cluster`
- Right channel clustering: `-`
- Num speakers: 2
- Audio info: 16000 Hz, channels=1, duration=60.00s
- VAD: mode=2, frame_ms=30, min_speech_ms=300, min_silence_ms=200, pad_ms=150, max_segment_s=30
- Segments: 6
- Outputs: `tmp/diarize_parakeet/report.md`, `tmp/diarize_parakeet/combined.txt`, `tmp/diarize_parakeet/segments.json`

## Таблица сегментов

| # | Start | End | Speaker | Lang | Stop | Text |
|---:|---|---|---|---|---|---|
| 0 | 00:00:00.120 | 00:00:02.160 | spk0 |  | eos |  |
| 1 | 00:00:02.250 | 00:00:32.250 | spk0 |  | eos |  |
| 2 | 00:00:32.250 | 00:00:33.900 | spk0 |  | eos | Работа и есть на ждем. |
| 3 | 00:00:34.050 | 00:00:46.680 | spk0 |  | eos | Примерно этих типов обслуживание как сделано для можно ли мы используем, тоже самое справочник или нет, и определиться, как из этих сценарий потребуется на тренинг. |
| 4 | 00:00:46.860 | 00:00:55.590 | spk1 |  | eos |  |
| 5 | 00:00:56.070 | 00:01:00.000 | spk1 |  | eos |  |

## Склейка (по времени)

См. также `combined.txt`.

```text
[00:00:00.120 - 00:00:02.160] spk0: 
[00:00:02.250 - 00:00:32.250] spk0: 
[00:00:32.250 - 00:00:33.900] spk0: Работа и есть на ждем.
[00:00:34.050 - 00:00:46.680] spk0: Примерно этих типов обслуживание как сделано для можно ли мы используем, тоже самое справочник или нет, и определиться, как из этих сценарий потребуется на тренинг.
[00:00:46.860 - 00:00:55.590] spk1: 
[00:00:56.070 - 00:01:00.000] spk1: 
```

Время выполнения: 3.19s
