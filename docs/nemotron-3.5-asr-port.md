# Порт Nemotron 3.5 ASR Streaming 0.6B → RustASR/candle + Yttri

> Спека + план + прогресс. Модель: `nvidia/nemotron-3.5-asr-streaming-0.6b`
> (FastConformer-CacheAware-RNNT + Lang-ID prompt). Цель: нативный Rust/candle-порт
> (как Parakeet/Qwen3-ASR), квантование Q8, S3-доставка, сравнение RU ASR.

## 1. Что это за модель (ground truth из .nemo)

- **Класс NeMo:** `EncDecRNNTBPEModelWithPrompt` (только в NeMo `main`, не в PyPI 2.7.x).
- **Параметры:** ~648M. Файл `.nemo` 2.37 ГБ (F32). Лицензия — см. README (NVIDIA OpenMDW).
- **RU:** Tier-1 (FLEURS WER ~9.2–10.8% LangID, ~10–12.5% auto). `target_lang=ru-RU`.

### Поток данных
```
audio(16k mono)
 → preprocessor: mel 128 бинов, n_fft 512, win 400(0.025s), hop 160(0.01s),
                 hann, log=true, preemph 0.97, normalize=NA (без норм!), dither=0 (infer)
                 [fb [1,128,257] и window [400] лежат в весах → точная репродукция]
 → encoder.pre_encode: dw_striding 8× (conv2d стек), out Linear[1024,4352] → d_model 1024
 → encoder.layers.0..23: FastConformer (rel_pos attn, n_heads 8, conv_kernel 9 causal,
                 conv_norm layer_norm, use_bias=false, ff_expansion 4 → d_ff 4096)
                 [att_context_style=chunked_limited; offline берём [56,13]]
 → prompt_kernel: concat(acoustic 1024, lang_onehot 128)=1152
                 → Linear[2048,1152] → act → Linear[1024,2048] → 1024   ← НОВЫЙ компонент
 → joint.enc: Linear[640,1024];  prediction → joint.pred: Linear[640,640]
 → joint: relu(enc+pred) → joint_net.2 Linear[13088,640]  (RNN-T, БЕЗ durations)
 → RNN-T greedy decode (blank=13087) → токены
 → SentencePiece detok (vocab.json, ▁→пробел) → текст
```

### Decoder (prediction net)
- `decoder.prediction.embed` [13088,640] (vocab+blank), `dec_rnn.lstm` — **2-layer LSTM** hidden 640.
- Язык ru-RU: 128-dim one-hot, **позиция 11** (`prompt_dictionary[ru-RU]=11`, `num_prompts=128`).

## 2. Карта переиспользования (crates/model-parakeet → model-nemotron)

| Компонент | Parakeet | Для Nemotron |
|---|---|---|
| `mel.rs` ParakeetMelExtractor | 128 бинов, n_fft 512 | **reuse**, fb/window из весов Nemotron |
| `encoder.rs` FastConformerEncoder | dw_striding 8×, rel_pos, 24сл | **reuse + adapt**: causal conv + chunked_limited маска, use_bias=false |
| `decoder.rs` PredictionNet | embed + 2-layer LSTM 640 | **reuse**, vocab 13088 |
| `joint.rs` JointNetwork | TDT (durations) | **adapt**: чистый RNN-T (enc1024→640, pred640→640, →13088) |
| `tdt.rs` TdtGreedyDecoder | token+duration greedy | **replace**: RNN-T greedy (проще, без duration) |
| SentencePieceTokenizer | vocab.json | **reuse** (vocab.json готов) |
| — | — | **NEW** `prompt.rs`: prompt_kernel + lang one-hot fusion |

Вывод: ~70% кода Parakeet переиспользуется. Основная новая работа — `prompt_kernel`,
RNN-T (вместо TDT) greedy, причинная маска энкодера, валидация parity.

## 3. Артефакты конвертации (готово)

`models/nemotron-3.5-asr-streaming-0.6b/`:
- `model.safetensors` (2434 MB, F32, 657 тензоров)
- `config.json` (все размерности + prompt + lang_index ru-RU=11)
- `vocab.json` (13087 pieces, blank idx 13087), `tokenizer.model` (SPE), `tokenizer_vocab.txt`
- `model_config.yaml` (полный оригинал)

Скрипты: `scripts/inspect_nemotron.py`, `scripts/convert_nemotron.py`,
`scripts/transcribe_nemotron_reference.py`. venv: `.venv-nemo` (Python 3.12, NeMo main 3.1.0).

## 4. Фазы и прогресс

- [x] **Ф0 Фундамент:** загрузка .nemo, NeMo main на 3.12, инспекция архитектуры.
- [~] **Ф1 Эталон:** NeMo транскрипция RU-фикстур (golden text) + WER; дамп референс-тензоров
      (mel/encoder/prompt_kernel/joint) для parity Rust-порта.
- [x] **Ф2 Конвертация:** .nemo → safetensors + config.json + vocab.json + tokenizer.
- [ ] **Ф3 Rust-порт** `crates/model-nemotron` (адаптация Parakeet): mel → encoder(causal) →
      prompt_kernel → RNN-T joint+greedy → detok; parity против Ф1 (tol mel<0.05, logits).
- [ ] **Ф4 Квантование:** энкодер safetensors(BF16) + decoder/joint Q8_0 GGUF (RNN-T loop F32);
      WER Q8 vs F32.
- [ ] **Ф5 Интеграция Yttri:** path-dep model-nemotron; `EngineType::Nemotron` (engine_manager);
      `nemotron_engine.rs`; bundled_models; models_downloader; provisioning; mel_common.
- [ ] **Ф6 S3 + публикация:** залить квант. файлы в S3 (`models/nemotron-3.5-asr-streaming-0.6b/`) +
      `storage_files`(modelId) + presigned. ⚠️ наружу — подтвердить перед заливкой.
- [ ] **Ф7 Сравнение:** Nemotron в `asr_ru_real_audio_compare.rs`; прогон всех RU ASR:
      WER/RTF/RAM/размер.

## 4b. Эталон Ф1 (NeMo, ru-RU idx 11, att_context [56,13], CPU)

Получен через manual forward (обход lhotse prompt-dataloader):
`model.forward(input_signal, length, prompt_indices=[11])` → `decoding.rnnt_decoder_predictions_tensor`.

| Аудио | dur | RTF | Текст (golden) |
|---|---|---|---|
| `frontend/.../fixtures/test_ru.wav` | 6.0s | 0.18 | «Проверка записи» |
| `test_prod_3s.wav` | 3.0s | 0.39 | «Примерно в двадцать третьем году в двадцать четыре» |
| `test_30sec.wav` | 30.0s | 0.07 | «Привет, ребят! Да я пока понял, что ты, оказывается, в этом, как он называется, в скайпе поставил… `<ru-RU>`» |

Качество RU — Tier-1 (пунктуация + капитализация). `<ru-RU>` — language-tag (strip_lang_tags).

**Parity-тензоры** (`tmp/parity/nemotron_*.npy`, для `test_prod_3s.wav`):
`mel [1,128,301]`, `encoded_pre_prompt [1,1024,76]`, `prompt_kernel_in [1,76,1152]`,
`prompt_kernel_out [1,76,1024]` — формы точно совпадают с config → модель архитектуры верна.
Эти тензоры — reference для Ф3 (mel → encoder → prompt_kernel parity).

## 5. Открытые риски

- **Causal/chunked_limited энкодер:** offline-parity требует точной маски внимания и причинной
  свёртки. Если расходится — снять att_context на full (xscaling=false) и сверить с NeMo full-context.
- **prompt_kernel активация:** уточнить (relu/gelu) по NeMo (rnnt_bpe_models_prompt).
- **MPS vs CPU в NeMo:** эталон считаем на CPU (стабильно); Rust-порт — Metal.
- **Квантование RNN-T:** держать LSTM/joint loop в F32 (как Qwen ASR decoder Q8, но stateful loop чувствителен).
