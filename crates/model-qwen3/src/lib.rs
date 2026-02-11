//! Qwen3-ASR модель — обёртка `asr_pipeline::AsrPipeline` через `AsrModel` trait.
//!
//! Этот крейт адаптирует существующий Qwen3-ASR пайплайн (AuT encoder + Qwen3 LLM decoder)
//! к единому интерфейсу [`AsrModel`], позволяя использовать его через `AsrEngine`.

mod model;

pub use model::Qwen3AsrModel;
