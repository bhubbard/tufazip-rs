use thiserror::Error;

/// Errors returned by arithmetic coding and bit stream operations.
#[derive(Debug, Error)]
pub enum ArithError {
    /// State bit width is outside the supported `[1, 62]` range.
    #[error("arithmetic coder state bit width {0} must be in 1..=62")]
    InvalidStateBits(u32),

    /// Attempted to write a bit value other than `0` or `1`.
    #[error("bit value must be 0 or 1, got {0}")]
    InvalidBit(u8),

    /// Frequency table is empty.
    #[error("cumulative frequency table cannot be empty")]
    EmptyFreqTable,

    /// Symbol index is out of bounds for the frequency table.
    #[error("symbol index {symbol} out of bounds for table of length {len}")]
    InvalidSymbol {
        /// Requested symbol index.
        symbol: usize,
        /// Number of symbols in the frequency table.
        len: usize,
    },

    /// Symbol has zero probability mass in the cumulative frequency table.
    #[error("symbol {0} has zero frequency")]
    ZeroFrequencySymbol(usize),

    /// Total frequency count exceeds the coder's maximum allowed total.
    #[error("frequency total {total} exceeds maximum allowed {max}")]
    TotalTooLarge {
        /// Observed cumulative frequency total.
        total: u64,
        /// Coder maximum allowed total (`minimum_range`).
        max: u64,
    },

    /// Restored checkpoint state does not match the coder's configured bit width.
    #[error("arithmetic coder state width mismatch: expected {expected}, found {found}")]
    StateWidthMismatch {
        /// Configured state width.
        expected: u32,
        /// Restored state width.
        found: u32,
    },

    /// Restored bit stream state contains invalid bit counts.
    #[error("invalid bit stream state: {0}")]
    InvalidStreamState(String),

    /// Unexpected end of input bit stream when a non-EOF bit was requested.
    #[error("unexpected end of bit stream")]
    UnexpectedEof,

    /// Underlying I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
