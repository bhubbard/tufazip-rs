use thiserror::Error;

/// Errors returned by dictionary preprocessing, loading, encoding, or decoding.
#[derive(Debug, Error)]
pub enum PreprocessError {
    /// Dictionary file contains an unrecognized backslash escape sequence.
    #[error("invalid escape sequence in dictionary at byte {0:#04x}")]
    InvalidEscape(u8),

    /// Dictionary entry exceeds the maximum supported word length (`4096` bytes).
    #[error("dictionary word exceeds 4096-byte limit")]
    WordTooLong,

    /// Token stream contains a symbol ID outside the dictionary bounds.
    #[error("invalid symbol {symbol} for dictionary of size {vocab_size}")]
    InvalidSymbol {
        /// Encountered token ID.
        symbol: u16,
        /// Number of entries in the dictionary.
        vocab_size: usize,
    },

    /// Maximum vocabulary size is invalid (must be at least 256).
    #[error("max_words ({0}) must be between 256 and 65536")]
    InvalidMaxWords(usize),

    /// Token stream length is not a multiple of 2 bytes.
    #[error("preprocessed token stream has odd byte length ({0} bytes)")]
    OddTokenStreamLength(u64),

    /// Provided dictionary lacks a single-byte entry needed to encode an arbitrary byte.
    #[error("dictionary is missing single-byte entry for byte {0:#04x}")]
    MissingByteSymbol(u8),

    /// Underlying I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
