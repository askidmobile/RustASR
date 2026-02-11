# Исследование: FastConformer-TDT (parakeet-tdt-0.6b-v3)

> Архитектура NVIDIA FastConformer + TDT декодер для реализации на чистом Rust (Candle).
> Источник: NVIDIA/NeMo (GitHub), HuggingFace model card nvidia/parakeet-tdt-0.6b-v3.

---

## Оглавление

1. [Общие сведения о модели](#1-общие-сведения-о-модели)
2. [Препроцессор (Mel-спектрограмма)](#2-препроцессор-mel-спектрограмма)
3. [FastConformer энкодер](#3-fastconformer-энкодер)
4. [TDT декодер (Prediction Network + Joint Network)](#4-tdt-декодер)
5. [TDT жадное декодирование (Greedy Decoding)](#5-tdt-жадное-декодирование)
6. [Формат .nemo файла и извлечение весов](#6-формат-nemo-файла)
7. [Именование ключей весов (state_dict)](#7-именование-ключей-весов)
8. [Сводная таблица гиперпараметров](#8-сводная-таблица-гиперпараметров)
9. [План реализации на Rust/Candle](#9-план-реализации)

---

## 1. Общие сведения о модели

| Параметр | Значение |
|---|---|
| Имя модели | `nvidia/parakeet-tdt-0.6b-v3` |
| Параметры | ~600M |
| Архитектура | FastConformer encoder + TDT decoder (Hybrid RNNT+CTC) |
| Частота дискретизации | 16 kHz |
| Языки | 25 европейских (en, ru, de, fr, es, ...) |
| Токенизатор | SentencePiece BPE, 8192 токена |
| TDT durations | `[0, 1, 2, 3, 4]` (5 значений) |
| Лицензия | CC-BY-4.0 |
| Размер .nemo | ~2.51 GB |
| Статьи | [Fast Conformer](https://arxiv.org/abs/2305.05084), [TDT](https://arxiv.org/abs/2304.06795) |

### Высокоуровневый pipeline

```
Аудио WAV (16kHz, mono)
    ↓
[Препроцессор] → Mel-спектрограмма [B, 128, T]
    ↓
[FastConformer Encoder] → Encoded features [B, D_enc, T']
    ↓                              ↓
    ↓                    [CTC Head] (вспомогательный, игнорируем для инференса)
    ↓
[TDT Decoder] = Prediction Net + Joint Net → Greedy TDT Decoding → Текст
```

---

## 2. Препроцессор (Mel-спектрограмма)

Класс: `AudioToMelSpectrogramPreprocessor` → использует `FilterbankFeatures`.

### Параметры из обучающего конфига

| Параметр | Значение | Описание |
|---|---|---|
| `sample_rate` | 16000 | Частота дискретизации |
| `window_size` | 0.025 | Размер окна в секундах = 400 сэмплов |
| `window_stride` | 0.01 | Шаг окна в секундах = 160 сэмплов |
| `window` | `"hann"` | Оконная функция |
| `features` | 128 | Количество mel-бинов |
| `n_fft` | 512 | Размер FFT |
| `normalize` | `"per_feature"` | Нормализация по каждому mel-бину |
| `dither` | 0.00001 | Добавление белого шума |
| `pad_to` | 0 | Padding выхода (0 = без паддинга) |
| `frame_splicing` | 1 | Без склеивания фреймов |

### Параметры по умолчанию (дополнительные)

| Параметр | Значение |
|---|---|
| `preemph` | 0.97 |
| `lowfreq` | 0 |
| `highfreq` | None (= sample_rate / 2 = 8000) |
| `log` | True |
| `log_zero_guard_type` | `"add"` |
| `log_zero_guard_value` | 2^(-24) ≈ 5.96e-8 |
| `mag_power` | 2.0 (power spectrum) |
| `mel_norm` | `"slaney"` (area normalization) |
| `exact_pad` | False |

### Алгоритм вычисления mel-спектрограммы

```
1. Preemphasis: signal[n] = signal[n] - 0.97 * signal[n-1]
2. Dither: signal += white_noise * 0.00001
3. STFT:
   - window = hann(400)
   - n_fft = 512
   - hop_length = 160
   - center = True (по умолчанию, exact_pad=False)
4. Power spectrum: |STFT|^2
5. Mel filterbank: 128 mel-бинов, [0 Hz, 8000 Hz], slaney norm
6. Log: log(mel + 2^-24)
7. Normalize per_feature: для каждого mel-бина вычитать mean и делить на std
```

### Вычисление длин

```python
n_window_size = int(0.025 * 16000) = 400
n_window_stride = int(0.01 * 16000) = 160
# Выходная длина:
T_mel = floor((audio_len - n_window_size) / n_window_stride) + 1
# Или при center=True (по умолчанию STFT):
T_mel = floor(audio_len / n_window_stride) + 1
```

### SpecAugment (только при обучении, не нужен для инференса)

```yaml
freq_masks: 2, time_masks: 10, freq_width: 27, time_width: 0.05
```

---

## 3. FastConformer энкодер

Класс: `ConformerEncoder` (`nemo.collections.asr.modules.ConformerEncoder`)

### Гиперпараметры из обучающего конфига

| Параметр | Значение | Описание |
|---|---|---|
| `feat_in` | 128 | = features из preprocessor |
| `feat_out` | -1 | = d_model (нет дополнительной проекции) |
| `n_layers` | 17 | Количество ConformerLayer |
| `d_model` | 512 | Размерность модели |
| `subsampling` | `"dw_striding"` | Тип субдискретизации |
| `subsampling_factor` | 8 | Коэффициент субдискретизации (log2(8) = 3 стадии) |
| `subsampling_conv_channels` | 256 | Каналы конволюций субдискретизации |
| `ff_expansion_factor` | 4 | d_ff = d_model × 4 = 2048 |
| `self_attention_model` | `"rel_pos"` | Relative positional (Transformer-XL) |
| `n_heads` | 8 | Количество голов внимания |
| `conv_kernel_size` | 9 | Размер ядра свёрточного модуля |
| `conv_norm_type` | `"batch_norm"` | BatchNorm1d в conv модуле |
| `att_context_size` | `[-1, -1]` | Полный контекст (не ограничен) |
| `xscaling` | True | Масштабирование на sqrt(d_model) |
| `untie_biases` | True | Развязанные biases в Transformer-XL |
| `pos_emb_max_len` | 5000 | |
| `dropout` | 0.1 | |
| `dropout_pre_encoder` | 0.1 | |
| `dropout_emb` | 0.0 | |
| `dropout_att` | 0.1 | |
| `use_bias` | True (по умолчанию) | |
| `reduction` | null | Без дополнительной редукции |

### Архитектура: Pre-encode (Subsampling)

Для `subsampling = "dw_striding"` с `subsampling_factor = 8`:

```
sampling_num = log2(8) = 3  → три стадии свёрток

Стадия 0 (i=0):
  Conv2d(1, 256, kernel=(3,3), stride=(2,2), padding=(1,1))  # обычная свёртка
  BatchNorm2d(256)
  ReLU

Стадии 1, 2 (i=1,2): depthwise separable
  Conv2d(256, 256, kernel=(3,3), stride=(2,2), padding=(1,1), groups=256)  # depthwise
  Conv2d(256, 256, kernel=(1,1), stride=(1,1))  # pointwise
  BatchNorm2d(256)
  ReLU

После конволюций:
  Transpose + Flatten: [B, C, T', F'] → [B, T', C × F']
  Linear(256 × out_freq, 512)  # проекция в d_model
```

Размер по частоте после 3 стадий:
```
F0 = 128 (mel bins)
F1 = floor((128 + 2×1 - 3) / 2 + 1) = 64
F2 = floor((64 + 2×1 - 3) / 2 + 1) = 32
F3 = floor((32 + 2×1 - 3) / 2 + 1) = 16
out_freq = 16
Linear input dim = 256 × 16 = 4096 → Linear(4096, 512)
```

Размер по времени:
```
T' = T_mel ÷ 8  (после 3 стадий stride=2)
```

### Архитектура: Positional Encoding

Для `self_attention_model = "rel_pos"` используется `RelPositionalEncoding`:

```python
class RelPositionalEncoding:
    # xscale = sqrt(512) ≈ 22.627
    # Позиции от (length-1) до -(length-1), итого 2*length - 1 позиций
    # PE[pos, 2i] = sin(pos / 10000^(2i/d_model))  
    # PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
    
    def forward(x):
        x = x * xscale  # масштабирование
        pos_emb = pe[center-input_len : center+input_len-1]  # [1, 2L-1, D]
        x = dropout(x)
        return x, pos_emb
```

### Архитектура: ConformerLayer

Каждый из 17 слоёв имеет структуру «Macaron-net»:

```
input x [B, T, 512]
    ↓
[LayerNorm] → [FeedForward1] → dropout × 0.5 + residual
    ↓
[LayerNorm] → [RelPositionMultiHeadAttention] → dropout + residual
    ↓
[LayerNorm] → [ConformerConvolution] → dropout + residual
    ↓
[LayerNorm] → [FeedForward2] → dropout × 0.5 + residual
    ↓
[LayerNorm] → output
```

#### FeedForward модуль

```
Linear(512, 2048) → Swish → Dropout(0.1) → Linear(2048, 512)
```

#### RelPositionMultiHeadAttention

```
n_heads = 8, d_k = 512/8 = 64

Параметры:
  linear_q: Linear(512, 512)
  linear_k: Linear(512, 512)
  linear_v: Linear(512, 512)
  linear_out: Linear(512, 512)
  linear_pos: Linear(512, 512, bias=False)  # для позиционного кодирования
  pos_bias_u: Parameter(8, 64)  # обучаемый bias для матрицы AC
  pos_bias_v: Parameter(8, 64)  # обучаемый bias для матрицы BD

Forward:
  1. q, k, v = linear_{q,k,v}(x), reshape → [B, H, T, d_k]
  2. p = linear_pos(pos_emb), reshape → [1, H, 2T-1, d_k]
  3. q_u = q + pos_bias_u  # [B, H, T, d_k]
  4. q_v = q + pos_bias_v
  5. matrix_ac = q_u @ k^T        # content-based attention
  6. matrix_bd = q_v @ p^T        # position-based attention  
  7. matrix_bd = rel_shift(matrix_bd)   # скос для выравнивания позиций
  8. scores = (matrix_ac + matrix_bd) / sqrt(64)
  9. attn = softmax(scores, mask)
  10. out = attn @ v → reshape → linear_out
```

**rel_shift** — преобразование для относительных позиций:
```python
def rel_shift(x):  # x: [B, H, T, 2T-1]
    x = F.pad(x, (1, 0))  # [B, H, T, 2T]
    x = x.view(B, H, 2T, T)  # reshape
    x = x[:, :, 1:].view(B, H, T, 2T-1)  # drop first row
    return x
```

#### ConformerConvolution модуль

```
input x [B, T, 512]
  ↓ transpose → [B, 512, T]
  ↓
Conv1d(512, 1024, kernel=1, stride=1)  # pointwise_conv1, расширение ×2
  ↓
GLU(dim=1)  # → [B, 512, T]
  ↓ (masked_fill если есть pad_mask)
CausalConv1d(512, 512, kernel=9, padding=4, groups=512)  # depthwise_conv
  ↓
BatchNorm1d(512)
  ↓
Swish()
  ↓
Conv1d(512, 512, kernel=1, stride=1)  # pointwise_conv2
  ↓ transpose → [B, T, 512]
```

**CausalConv1D** при `padding = (kernel_size-1)//2 = 4`:
- `_left_padding = 4`
- `_right_padding = 4`
- Это стандартная (не каузальная) свёртка — симметричный padding

### Выход энкодера

```
Вход: [B, 128, T_mel]
После subsampling: [B, T', 512]  где T' ≈ T_mel / 8
Через 17 ConformerLayer: [B, T', 512]
feat_out = -1 → нет out_proj
Transpose: [B, 512, T']  → это encoder output

enc_hidden = d_model = 512
```

---

## 4. TDT декодер

### 4.1. Prediction Network (StatelessTransducerDecoder)

Класс: `StatelessTransducerDecoder` → использует `StatelessNet`.

#### Параметры

| Параметр | Значение | Описание |
|---|---|---|
| `pred_hidden` | 640 | Размерность эмбеддинга |
| `pred_rnn_layers` | 1 | (не используется для stateless) |
| `context_size` | 2 (по умолчанию) | Размер истории |
| `vocab_size` | 8192 | + 1 blank = 8193 |
| `normalization_mode` | null | (в конфиге null для экспорта) |
| `blank_as_pad` | true | blank = padding |
| `dropout` | 0.2 | |

#### blank_idx

```python
blank_idx = vocab_size  # = 8192 (индекс blank — последний)
```

#### StatelessNet архитектура

```python
context_size = 2  # (рекомендуемое значение)
vocab_size = 8192
emb_dim = 640
blank_idx = 8192

# Создаём 2 эмбеддинга с разными размерами:
# i=0 (самый последний токен): получает больше измерений
embed_0_size = emb_dim - (emb_dim // 2 // context_size) * (context_size - 1)
            = 640 - (320 // 2) * 1 = 640 - 160 = 480
# i=1 (предпоследний токен):
embed_1_size = emb_dim // 2 // context_size = 320 // 2 = 160

# Итого:
embeds[0] = Embedding(8193, 480, padding_idx=8192)  # последний токен
embeds[1] = Embedding(8193, 160, padding_idx=8192)  # предпоследний токен
# 480 + 160 = 640 = emb_dim ✓

# Нормализация:
norm = Identity()  # normalization_mode=null

# Dropout:
dropout = Dropout(0.2)
```

#### Forward pass StatelessNet

```python
def predict(y, state):
    """
    y: [B, 1] — текущий токен (или несколько)
    state: [tensor of shape [B, context_size-1]] — история из предыдущих меток
    
    1. Конкатенация: appended_y = [state[0], y] → [B, context_size]
    2. Для каждой позиции i контекста:
       out_i = embeds[i](appended_y[:, context_size-1-i : context_size-i])
    3. out = concat(out_0, out_1, dim=-1) → [B, 1, 640]
    4. out = dropout(out)
    5. out = norm(out)  # Identity при null
    6. Обновить state: state = [appended_y[:, -context_size+1:]]
    
    return out [B, 1, 640], state
    """
```

При начале декодирования:
```python
# Инициализация state:
state = [torch.full([B, context_size-1], fill_value=blank_idx)]  
# = [tensor([8192])]  для context_size=2

# add_sos = True → prepend нулевой вектор как SOS
```

### 4.2. Joint Network (RNNTJoint)

Класс: `RNNTJoint`

#### Параметры

| Параметр | Значение | Описание |
|---|---|---|
| `joint_hidden` | 640 | Скрытый размер joint |
| `encoder_hidden` | 512 (= enc_hidden) | Размер выхода энкодера |
| `pred_hidden` | 640 | Размер выхода prediction net |
| `num_classes` | 8192 | = vocab_size (без blank) |
| `activation` | `"relu"` | Функция активации |
| `dropout` | 0.2 | |
| `num_extra_outputs` | 5 | = num_tdt_durations (для длительностей) |

#### Архитектура

```python
# Проекции
enc_proj  = Linear(512, 640)    # проекция энкодера
pred_proj = Linear(640, 640)    # проекция предсказательной сети

# Joint network
joint_net = Sequential(
    ReLU(),
    Dropout(0.2),
    Linear(640, 8192 + 1 + 5)   # = Linear(640, 8198)
)
# num_classes_with_blank = 8193
# total output = 8193 + 5 = 8198
#   [0..8192] = token logits (включая blank на индексе 8192)
#   [8193..8197] = duration logits (5 значений для durations [0,1,2,3,4])
```

#### Forward pass Joint

```python
def joint(encoder_out, decoder_out):
    """
    encoder_out: [B, T', 512]  (после транспонирования)
    decoder_out: [B, U, 640]
    
    1. f = enc_proj(encoder_out)   → [B, T', 640]
    2. g = pred_proj(decoder_out)  → [B, U, 640]
    3. h = f.unsqueeze(2) + g.unsqueeze(1)  → [B, T', U, 640]
    4. out = joint_net(h)          → [B, T', U, 8198]
    
    return out
    """
```

---

## 5. TDT жадное декодирование

Класс: `GreedyTDTInfer` / `GreedyBatchedTDTInfer`

### Конфигурация

```yaml
strategy: "greedy_batch"
model_type: "tdt"
durations: [0, 1, 2, 3, 4]   # возможные значения длительностей
max_symbols: 10               # макс. символов на один фрейм
```

### Алгоритм (пошагово)

```
ВХОД:
  encoder_output: [T', D_enc]   — выход энкодера для одного аудио
  durations = [0, 1, 2, 3, 4]   — список допустимых длительностей
  blank_index = 8192             — индекс blank-токена

ИНИЦИАЛИЗАЦИЯ:
  last_label = blank_index
  state = initialize_state()     — [blank_idx] для context_size=2
  time_idx = 0
  hypothesis = []

ЦИКЛ:
  while time_idx < T':
      f = encoder_output[time_idx]           # [1, 1, D_enc]
      need_loop = True
      symbols_added = 0
      
      while need_loop AND symbols_added < max_symbols:
          # 1. Prediction step
          g, state_next = prediction_net(last_label, state)  # [1, 1, pred_hidden]
          
          # 2. Joint step  
          logits = joint_net(f, g)           # [1, 1, 1, 8198]
          logits = logits.squeeze()          # [8198]
          
          # 3. Разделение logits
          token_logits = logits[:-5]         # [8193]  — первые 8193 значения
          duration_logits = logits[-5:]      # [5]     — последние 5 значений
          
          # 4. Log-softmax для длительностей  
          duration_logp = log_softmax(duration_logits)
          
          # 5. Находим лучший токен
          v, k = token_logits.max(dim=0)     # k = индекс лучшего токена
          
          # 6. Находим лучшую длительность
          d_v, d_k = duration_logp.max(dim=0)  # d_k = индекс лучшей длительности
          skip = durations[d_k]              # длительность в фреймах
          
          # 7. Решение
          if k != blank_index:
              # Эмиссия токена
              hypothesis.append(k)
              last_label = k
              state = state_next
              symbols_added += 1
          # endif
          
          # 8. Продвижение по времени
          if skip == 0:
              need_loop = True   # длительность 0 → повторить для того же фрейма
          else:
              time_idx += skip
              need_loop = False  # перейти к следующему фрейму
      
      # Если skip=0 и мы вышли по max_symbols → принудительно сдвигаемся на 1
      if time_idx не изменился:  
          time_idx += 1

ВЫХОД: hypothesis — список токен-индексов, декодируемых через SentencePiece
```

### Ключевые особенности TDT

1. **Переменный шаг по времени**: В отличие от vanilla RNN-T, где time_idx всегда +1 при blank, TDT предсказывает `skip` — сколько фреймов пропустить.
2. **Durations [0,1,2,3,4]**: Модель может предсказать пропуск от 0 до 4 фреймов. `skip=0` означает «остаться на том же фрейме и предсказать ещё один токен».
3. **Эффективность**: При длинных аудио TDT значительно быстрее RNN-T, так как может пропускать по 4 фрейма за раз.
4. **Joint output**: Последние `len(durations)=5` выходов joint network — это logits для длительностей, остальные — для токенов.

---

## 6. Формат .nemo файла

### Структура

Файл `.nemo` — это **tar-архив** (без сжатия, `"r:"` mode):

```
model.nemo (tar archive)
├── model_config.yaml        # Полная конфигурация модели (OmegaConf/Hydra)
├── model_weights.ckpt       # PyTorch state_dict (torch.save)
├── tokenizer.model          # SentencePiece модель (если BPE)
├── vocab.json               # Словарь (если есть)
└── ... (другие артефакты)
```

### Извлечение

```python
import tarfile
import torch
from omegaconf import OmegaConf

# 1. Извлечь конфигурацию
with tarfile.open("parakeet-tdt-0.6b-v3.nemo", "r:") as tar:
    # Найти model_config.yaml
    names = tar.getnames()
    prefix = ""  # или detect_prefix(names)
    
    # Извлечь конфиг
    tar.extract(f"{prefix}model_config.yaml", path="/tmp")
    config = OmegaConf.load("/tmp/model_config.yaml")
    
    # Извлечь веса
    tar.extract(f"{prefix}model_weights.ckpt", path="/tmp")
    state_dict = torch.load("/tmp/model_weights.ckpt", map_location="cpu")
    
    # Извлечь токенизатор
    tar.extract(f"{prefix}tokenizer.model", path="/tmp")
```

### Для Rust/Candle

Веса в `.ckpt` — формат `torch.save` (pickle). Для загрузки в Candle:

**Вариант 1**: Конвертировать в safetensors через Python-скрипт:
```python
import torch
from safetensors.torch import save_file

state_dict = torch.load("model_weights.ckpt", map_location="cpu")
save_file(state_dict, "model.safetensors")
```

**Вариант 2**: Скачать с HuggingFace (если доступна `model.safetensors` версия).

---

## 7. Именование ключей весов

### Encoder (ConformerEncoder)

```
encoder.pre_encode.conv.0.weight                    # Conv2d(1, 256, 3, 3) — первая свёртка
encoder.pre_encode.conv.0.bias
encoder.pre_encode.conv.1.weight                    # BatchNorm2d(256)
encoder.pre_encode.conv.1.bias
encoder.pre_encode.conv.1.running_mean
encoder.pre_encode.conv.1.running_var
encoder.pre_encode.conv.3.weight                    # Conv2d depthwise (stage 1)
encoder.pre_encode.conv.3.bias
encoder.pre_encode.conv.4.weight                    # Conv2d pointwise (stage 1)
encoder.pre_encode.conv.4.bias
encoder.pre_encode.conv.5.weight                    # BatchNorm2d
encoder.pre_encode.conv.5.bias
...                                                  # stage 2 аналогично
encoder.pre_encode.out.weight                        # Linear(4096, 512) проекция
encoder.pre_encode.out.bias

encoder.pos_enc.pe                                   # buffer, не обучаемый

encoder.pos_emb_max_len                              # buffer

# Для каждого слоя i (0..16):
encoder.layers.{i}.norm_feed_forward1.weight         # LayerNorm
encoder.layers.{i}.norm_feed_forward1.bias
encoder.layers.{i}.feed_forward1.linear1.weight      # Linear(512, 2048)
encoder.layers.{i}.feed_forward1.linear1.bias
encoder.layers.{i}.feed_forward1.linear2.weight      # Linear(2048, 512)
encoder.layers.{i}.feed_forward1.linear2.bias

encoder.layers.{i}.norm_self_att.weight              # LayerNorm
encoder.layers.{i}.norm_self_att.bias
encoder.layers.{i}.self_attn.linear_q.weight         # Linear(512, 512)
encoder.layers.{i}.self_attn.linear_q.bias
encoder.layers.{i}.self_attn.linear_k.weight
encoder.layers.{i}.self_attn.linear_k.bias
encoder.layers.{i}.self_attn.linear_v.weight
encoder.layers.{i}.self_attn.linear_v.bias
encoder.layers.{i}.self_attn.linear_out.weight
encoder.layers.{i}.self_attn.linear_out.bias
encoder.layers.{i}.self_attn.linear_pos.weight       # Linear(512, 512, bias=False)
encoder.layers.{i}.self_attn.pos_bias_u              # Parameter(8, 64)
encoder.layers.{i}.self_attn.pos_bias_v              # Parameter(8, 64)

encoder.layers.{i}.norm_conv.weight                  # LayerNorm
encoder.layers.{i}.norm_conv.bias
encoder.layers.{i}.conv.pointwise_conv1.weight       # Conv1d(512, 1024, 1)
encoder.layers.{i}.conv.pointwise_conv1.bias
encoder.layers.{i}.conv.depthwise_conv.weight        # Conv1d(512, 512, 9, groups=512)
encoder.layers.{i}.conv.depthwise_conv.bias
encoder.layers.{i}.conv.batch_norm.weight            # BatchNorm1d(512)
encoder.layers.{i}.conv.batch_norm.bias
encoder.layers.{i}.conv.batch_norm.running_mean
encoder.layers.{i}.conv.batch_norm.running_var
encoder.layers.{i}.conv.pointwise_conv2.weight       # Conv1d(512, 512, 1)
encoder.layers.{i}.conv.pointwise_conv2.bias

encoder.layers.{i}.norm_feed_forward2.weight
encoder.layers.{i}.norm_feed_forward2.bias
encoder.layers.{i}.feed_forward2.linear1.weight
encoder.layers.{i}.feed_forward2.linear1.bias
encoder.layers.{i}.feed_forward2.linear2.weight
encoder.layers.{i}.feed_forward2.linear2.bias

encoder.layers.{i}.norm_out.weight                   # LayerNorm
encoder.layers.{i}.norm_out.bias
```

### При untie_biases=True

Когда `untie_biases=True`, каждый слой имеет **свои** `pos_bias_u` и `pos_bias_v`, переданные в конструктор как параметры энкодера:

```
# Biases создаются в ConformerEncoder и передаются в каждый слой:
# Если untie_biases=True, biases уникальны для каждого слоя
# Они хранятся внутри self_attn каждого слоя
encoder.layers.{i}.self_attn.pos_bias_u              # Parameter(8, 64)
encoder.layers.{i}.self_attn.pos_bias_v              # Parameter(8, 64)
```

### Decoder (StatelessTransducerDecoder)

```
decoder.prediction.embeds.0.weight                   # Embedding(8193, 480)
decoder.prediction.embeds.1.weight                   # Embedding(8193, 160)
decoder.prediction.norm.weight                       # (если normalization_mode='layer')
decoder.prediction.norm.bias                         # (если normalization_mode='layer')
```

При `normalization_mode=null` → norm = Identity → нет обучаемых параметров для norm.

### Joint Network (RNNTJoint)

```
joint.enc.weight                                     # Linear(512, 640)
joint.enc.bias
joint.pred.weight                                    # Linear(640, 640)
joint.pred.bias
joint.joint_net.1.weight                             # Linear(640, 8198)    [после ReLU + Dropout]
joint.joint_net.1.bias                               # или joint.joint_net.2.weight если с dropout
```

Примечание: нумерация в `joint_net` зависит от наличия Dropout:
```python
joint_net = Sequential(
    ReLU(),         # [0]
    Dropout(0.2),   # [1] (только при обучении, может быть в state_dict)
    Linear(640, 8198)  # [2] или [1] 
)
```

### CTC Head (вспомогательный, не нужен для TDT инференса)

```
ctc_decoder.decoder_layers.0.weight                  # Linear(512, 8193)
ctc_decoder.decoder_layers.0.bias
```

---

## 8. Сводная таблица гиперпараметров

| Компонент | Параметр | Значение |
|---|---|---|
| **Audio** | sample_rate | 16000 |
| **Audio** | n_fft | 512 |
| **Audio** | win_length | 400 (0.025s) |
| **Audio** | hop_length | 160 (0.01s) |
| **Audio** | mel_bins | 128 |
| **Audio** | preemph | 0.97 |
| **Encoder** | n_layers | 17 |
| **Encoder** | d_model | 512 |
| **Encoder** | n_heads | 8 |
| **Encoder** | d_k (head dim) | 64 |
| **Encoder** | d_ff | 2048 (512×4) |
| **Encoder** | conv_kernel | 9 |
| **Encoder** | subsampling | dw_striding ×8 |
| **Encoder** | subsampling_channels | 256 |
| **Encoder** | pos_encoding | rel_pos (Transformer-XL) |
| **Decoder** | pred_hidden | 640 |
| **Decoder** | context_size | 2 |
| **Decoder** | embed_0_dim | 480 |
| **Decoder** | embed_1_dim | 160 |
| **Joint** | joint_hidden | 640 |
| **Joint** | output_dim | 8198 (8193 + 5) |
| **TDT** | durations | [0, 1, 2, 3, 4] |
| **TDT** | blank_idx | 8192 |
| **Tokenizer** | type | SentencePiece BPE |
| **Tokenizer** | vocab_size | 8192 |

---

## 9. План реализации на Rust/Candle

### Крейт `model-parakeet`

```
crates/model-parakeet/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── config.rs              # ParakeetConfig, все гиперпараметры
    ├── preprocessor.rs        # Mel-спектрограмма (повторно использовать из audio)
    ├── subsampling.rs         # DwStridingSubsampling (Conv2d ×3)
    ├── pos_encoding.rs        # RelPositionalEncoding
    ├── attention.rs           # RelPositionMultiHeadAttention
    ├── conformer_conv.rs      # ConformerConvolution (pointwise+depthwise+BN)
    ├── feed_forward.rs        # ConformerFeedForward
    ├── conformer_layer.rs     # ConformerLayer (Macaron-net)
    ├── encoder.rs             # FastConformerEncoder (subsampling + N × layer)
    ├── prediction_net.rs      # StatelessNet (embedding concatenation)
    ├── joint_net.rs           # RNNTJoint (encoder proj + decoder proj + joint)
    ├── tdt_decoder.rs         # TDT greedy decoding algorithm
    ├── tokenizer.rs           # SentencePiece wrapper
    └── weights.rs             # Загрузка и маппинг весов из safetensors
```

### Порядок реализации

1. **Конвертация весов**: Python-скрипт для извлечения из `.nemo` → safetensors
2. **Mel-спектрограмма**: Адаптировать существующий `audio` крейт (window=0.025, stride=0.01, 128 mel bins)
3. **Subsampling**: 3 стадии Conv2d (dw_striding), затем Linear проекция
4. **Positional Encoding**: RelPositionalEncoding с xscale=sqrt(512)
5. **ConformerLayer**: FeedForward, RelPosMHA, ConvModule, residual connections
6. **Encoder**: Собрать pre_encode + pos_enc + 17 × ConformerLayer
7. **Prediction Net**: StatelessNet с 2 эмбеддингами (480 + 160)
8. **Joint Net**: Enc проекция + Dec проекция + ReLU + Linear(640, 8198)
9. **TDT Decoding**: Greedy loop с предсказанием длительностей
10. **Pipeline**: Аудио → Mel → Encoder → TDT Decode → SentencePiece detokenize

### Особенности для Candle

- **BatchNorm**: В Candle есть `batch_norm` — использовать running_mean/running_var для инференса
- **GLU**: `x.chunk(2, dim)` → `a * sigmoid(b)`, в Candle вручную через split
- **Conv2d groups**: Candle поддерживает groups в Conv2d
- **rel_shift**: Реализовать через pad + reshape
- **Тензоры**: Все вычисления в f32, при GPU можно bf16 для encoder

---

## Приложение А: Параметры по слоям (подсчёт)

### Encoder (~320M параметров)

```
Pre-encode (subsampling):
  Conv2d(1, 256, 3×3)     = 2,304 + 256 bias
  BN(256)                 = 512
  Conv2d(256, 256, 3×3, g=256) = 2,304 + 256  (depthwise stage 1)
  Conv2d(256, 256, 1×1)   = 65,536 + 256      (pointwise stage 1)
  BN(256)                 = 512
  Conv2d(256, 256, 3×3, g=256) = 2,304 + 256  (depthwise stage 2)
  Conv2d(256, 256, 1×1)   = 65,536 + 256
  BN(256)                 = 512
  Linear(4096, 512)       = 2,097,152 + 512
  ≈ 2.2M параметров

Один ConformerLayer:
  2× FeedForward:  2 × (512×2048 + 2048 + 2048×512 + 512)    = 2 × 2,099,712 = 4,199,424
  MHA:             4 × (512×512 + 512) + 512×512 + 8×64 + 8×64 = 1,574,400
  Conv:            512×1024 + 1024 + 512×9 + 512 + 512 + 512×512 + 512 = 794,112
  5× LayerNorm:    5 × (512 + 512) = 5,120
  ≈ 6.57M на слой

17 слоёв: 17 × 6.57M ≈ 111.7M

Итого encoder ≈ 114M
```

### Decoder (~5M параметров)

```
Embedding(8193, 480)  = 3,932,640
Embedding(8193, 160)  = 1,310,880
≈ 5.2M
```

### Joint (~5.8M параметров)

```
Linear(512, 640) + bias  = 328,320
Linear(640, 640) + bias  = 410,240
Linear(640, 8198) + bias = 5,254,918
≈ 5.99M
```

### CTC Head (~4.2M, не используется в TDT)

```
Linear(512, 8193) + bias = 4,194,944
```

> **Примечание**: Реальные цифры для 0.6b модели (600M параметров) могут отличаться
> от стандартных значений из обучающего конфига. Конфиг описывает ~115M "large" архитектуру,
> но parakeet-tdt-0.6b имеет увеличенные размерности. Точные размерности нужно проверить
> по model_config.yaml из .nemo файла.
>
> Исходя из 600M параметров, вероятные изменения:
> - d_model может быть 1024 вместо 512
> - n_layers может быть ~24
> - n_heads может быть 8 или 16
> - subsampling_conv_channels может быть 256 или 512

---

## Приложение Б: Точные параметры из .nemo файла

> **ВАЖНО**: Для получения точных параметров конкретной модели parakeet-tdt-0.6b-v3:
>
> ```python
> import nemo.collections.asr as nemo_asr
> model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
> print(model.cfg)
> # или
> for name, param in model.named_parameters():
>     print(f"{name}: {param.shape}")
> ```
>
> Или извлечь `model_config.yaml` из `.nemo` файла (см. раздел 6).
>
> Рекомендуется создать Python-скрипт `scripts/inspect_parakeet.py` для
> автоматического извлечения всех параметров и ключей весов.
