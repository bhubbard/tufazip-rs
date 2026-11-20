//! Pure-Rust port of Fabrice Bellard's NNCP dictionary preprocessor (`preprocess.c`).
//!
//! Performs reversible case/space encoding, iterative bigram dictionary induction,
//! big-endian `u16` token serialization, and streaming decoding back to exact source bytes.

mod case_space;
mod codec;
mod dictionary;
mod error;

pub use case_space::{
    case_space_decode, case_space_encode, CaseSpaceDecoder, CH_FIRST_UPPER, CH_NO_SPACE,
    CH_TO_UPPER,
};
pub use codec::{
    decode_file, decode_tokens, encode_bytes, encode_file, encode_with_dictionary,
    read_tokens_be16, write_tokens_be16, EncodeConfig, EncodeReport, SortOrder,
};
pub use dictionary::Dictionary;
pub use error::PreprocessError;
