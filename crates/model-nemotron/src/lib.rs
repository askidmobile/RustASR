//! # model-nemotron
//!
//! Нативный Rust/Candle порт **NVIDIA Nemotron 3.5 ASR Streaming 0.6B**
//! (`nvidia/nemotron-3.5-asr-streaming-0.6b`).
//!
//! Архитектура: **FastConformer-CacheAware-RNNT + Lang-ID prompt**.
//! Подробности — `docs/nemotron-3.5-asr-port.md`.
//!
//! ## Пайплайн (план Ф3)
//! ```text
//! audio(16k) → mel(128, n_fft512, win400, hop160, log, preemph0.97, normalize=NA)
//!   → pre_encode dw_striding 8× → 24× FastConformer (rel_pos, causal conv, d_model1024)
//!   → prompt_kernel: concat(acoustic1024, lang_onehot128) → Linear[2048,1152] → act → Linear[1024,2048]
//!   → joint(enc1024→640, pred640→640) → joint_net.2 → 13088
//!   → RNN-T greedy (blank=13087, 2-layer LSTM pred) → SentencePiece detok
//! ```
//!
//! ## Статус
//! - [x] `config` — десериализация config.json (готово).
//! - [ ] `mel`     — переиспользовать ParakeetMelExtractor (fb/window из весов).
//! - [ ] `encoder` — адаптация FastConformerEncoder (causal conv + chunked_limited маска).
//! - [ ] `prompt`  — prompt_kernel + lang one-hot (НОВЫЙ компонент).
//! - [ ] `decoder` — PredictionNet (2-layer LSTM, переиспользуемо).
//! - [ ] `joint`   — RNN-T joint (без TDT durations).
//! - [ ] `rnnt`    — RNN-T greedy decode.
//! - [ ] `model`   — сборка + impl AsrModel + parity против tmp/parity/nemotron_*.npy.

pub mod config;
pub mod encoder;
pub mod mel;
pub mod model;

pub use config::NemotronConfig;
pub use encoder::NemotronEncoder;
pub use mel::NemotronMelExtractor;
pub use model::NemotronModel;
