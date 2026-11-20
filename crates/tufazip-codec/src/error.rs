use thiserror::Error;
use tufazip_arith::ArithError;
use tufazip_prior::PriorError;

/// Errors returned by the neural codec, model initialization, optimizer, or checkpoints.
#[derive(Debug, Error)]
pub enum CodecError {
    /// Underlying Candle tensor or autograd error.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// Arithmetic coding or bit stream error.
    #[error(transparent)]
    Arith(#[from] ArithError),

    /// Quantized prior archive or validation error.
    #[error(transparent)]
    Prior(#[from] PriorError),

    /// Underlying filesystem I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// JSON configuration or checkpoint metadata error.
    #[error(transparent)]
    Json(#[from] serde_json::Error),

    /// Invalid configuration or CLI parameter combination.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Resumed checkpoint does not match the current invocation or failed verification.
    #[error("checkpoint mismatch: {0}")]
    CheckpointMismatch(String),

    /// Input token file or compressed payload is malformed.
    #[error("invalid codec data: {0}")]
    InvalidData(String),
}
