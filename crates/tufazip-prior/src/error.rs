use thiserror::Error;

/// Errors returned when reading, writing, or validating quantized prior archives.
#[derive(Debug, Error)]
pub enum PriorError {
    /// Archive header magic is not one of `QPR1`, `QPI1`, `LRA1`, or `QLA1`.
    #[error("unknown tensor archive magic: {0:?}")]
    UnknownMagic([u8; 4]),

    /// Archive ended before header, metadata, or tensor payload finished.
    #[error("truncated tensor archive while reading {0}")]
    Truncated(String),

    /// Archive metadata version is unsupported.
    #[error("unsupported tensor archive version {0}")]
    UnsupportedVersion(u32),

    /// Archive metadata contains invalid tensor descriptors.
    #[error("invalid archive metadata: {0}")]
    InvalidMetadata(String),

    /// Archive defines the same tensor name twice.
    #[error("duplicate archive tensor: {0}")]
    DuplicateTensor(String),

    /// Tensor shape is empty or contains a zero dimension.
    #[error("invalid tensor shape for {name}: {shape:?}")]
    InvalidShape {
        /// Tensor parameter name.
        name: String,
        /// Invalid shape dimensions.
        shape: Vec<usize>,
    },

    /// Bit width is not one of `1`, `2`, `4`, `8`, or `16`.
    #[error("unsupported bit width {bits} for {name}")]
    UnsupportedBitWidth {
        /// Tensor parameter name.
        name: String,
        /// Unsupported bit width.
        bits: u8,
    },

    /// Entry byte count or scale count does not match its shape and bit width.
    #[error("size mismatch for {name}: {detail}")]
    SizeMismatch {
        /// Tensor parameter name.
        name: String,
        /// Explanation of the mismatch.
        detail: String,
    },

    /// Unexpected data found after the final tensor entry.
    #[error("tensor archive has trailing bytes")]
    TrailingBytes,

    /// Prior initialization metadata does not match the target model or vocabulary.
    #[error("prior validation failed: {0}")]
    Validation(String),

    /// Underlying I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// JSON metadata parsing or serialization error.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
