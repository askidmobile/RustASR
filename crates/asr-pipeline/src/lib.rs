//! ASR Pipeline - End-to-end speech recognition.
//!
//! This crate provides the main pipeline for speech recognition,
//! combining audio processing, neural network encoding, and text decoding.

mod output;
mod pipeline;
mod tokenizer;

pub use output::{AsrTranscription, StopReason, parse_asr_output};
pub use pipeline::{AsrPipeline, DecoderWeights};
pub use tokenizer::Tokenizer;
