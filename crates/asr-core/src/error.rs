//! Error types for RustASR.

use thiserror::Error;

/// Main error type for ASR operations.
#[derive(Error, Debug)]
pub enum AsrError {
    /// Audio processing errors.
    #[error("Audio error: {0}")]
    Audio(String),

    /// Model loading errors.
    #[error("Model error: {0}")]
    Model(String),

    /// Inference errors.
    #[error("Inference error: {0}")]
    Inference(String),

    /// Configuration errors.
    #[error("Config error: {0}")]
    Config(String),

    /// I/O errors.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Candle tensor errors.
    #[error("Tensor error: {0}")]
    Candle(#[from] candle_core::Error),

    /// JSON parsing errors.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Result type alias for ASR operations.
pub type AsrResult<T> = Result<T, AsrError>;
