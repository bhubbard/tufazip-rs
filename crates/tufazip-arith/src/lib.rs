//! Reference arithmetic coder, big-endian bit streams, and frequency tables for `tufazip`.
//!
//! Ported from `arithmetic_coder.py` (Project Nayuki) and `make_freq_table` in `nn.py`
//! with exact resumable checkpoint state snapshots.

mod bitstream;
mod coder;
mod error;

pub use bitstream::{BitInputState, BitInputStream, BitOutputState, BitOutputStream};
pub use coder::{
    make_freq_table, make_freq_table_into, ArithmeticDecoder, ArithmeticEncoder, DecoderState,
    EncoderState, DEFAULT_STATE_BITS, FREQ_TOTAL,
};
pub use error::ArithError;
