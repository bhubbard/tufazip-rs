use std::io::{Read, Write};

use serde::{Deserialize, Serialize};

use crate::bitstream::{BitInputStream, BitOutputStream};
use crate::error::ArithError;

/// Default state bit width used by `tufazip`.
pub const DEFAULT_STATE_BITS: u32 = 32;

/// Frequency-table resolution used when quantizing softmax probabilities.
pub const FREQ_TOTAL: u32 = 10_000_000;

/// Exact serializable state of an [`ArithmeticEncoder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncoderState {
    /// Bit width of the coder state registers.
    pub num_state_bits: u32,
    /// Lower bound of the active coding interval.
    pub low: u64,
    /// Upper bound (inclusive) of the active coding interval.
    pub high: u64,
    /// Number of pending underflow bits.
    pub num_underflow: u64,
}

/// Exact serializable state of an [`ArithmeticDecoder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderState {
    /// Bit width of the coder state registers.
    pub num_state_bits: u32,
    /// Lower bound of the active coding interval.
    pub low: u64,
    /// Upper bound (inclusive) of the active coding interval.
    pub high: u64,
    /// Current buffered code word.
    pub code: u64,
}

#[derive(Debug, Clone, Copy)]
struct CoderCore {
    num_state_bits: u32,
    half_range: u64,
    quarter_range: u64,
    maximum_total: u64,
    state_mask: u64,
    low: u64,
    high: u64,
}

impl CoderCore {
    fn new(num_state_bits: u32) -> Result<Self, ArithError> {
        if !(1..=62).contains(&num_state_bits) {
            return Err(ArithError::InvalidStateBits(num_state_bits));
        }
        let full_range = 1u64 << num_state_bits;
        let half_range = full_range >> 1;
        let quarter_range = half_range >> 1;
        let minimum_range = quarter_range + 2;
        let state_mask = full_range - 1;
        Ok(Self {
            num_state_bits,
            half_range,
            quarter_range,
            maximum_total: minimum_range,
            state_mask,
            low: 0,
            high: state_mask,
        })
    }

    #[inline]
    fn validate_table(&self, freqs: &[u32]) -> Result<u64, ArithError> {
        let total = *freqs.last().ok_or(ArithError::EmptyFreqTable)?;
        let total_u64 = u64::from(total);
        if total_u64 == 0 || total_u64 > self.maximum_total {
            return Err(ArithError::TotalTooLarge {
                total: total_u64,
                max: self.maximum_total,
            });
        }
        Ok(total_u64)
    }

    #[inline]
    fn update<F>(&mut self, freqs: &[u32], symbol: usize, mut on_event: F) -> Result<(), ArithError>
    where
        F: FnMut(UpdateEvent) -> Result<(), ArithError>,
    {
        if symbol >= freqs.len() {
            return Err(ArithError::InvalidSymbol {
                symbol,
                len: freqs.len(),
            });
        }
        let total = self.validate_table(freqs)?;
        let sym_low = if symbol > 0 { freqs[symbol - 1] } else { 0 };
        let sym_high = freqs[symbol];
        if sym_low >= sym_high {
            return Err(ArithError::ZeroFrequencySymbol(symbol));
        }

        let (new_low, new_high) = if self.num_state_bits <= 32 {
            let range = self.high - self.low + 1;
            let new_low = self.low + (u64::from(sym_low) * range) / total;
            let new_high = self.low + (u64::from(sym_high) * range) / total - 1;
            (new_low, new_high)
        } else {
            let low_u128 = u128::from(self.low);
            let range_u128 = u128::from(self.high - self.low + 1);
            let total_u128 = u128::from(total);
            let new_low = (low_u128 + (u128::from(sym_low) * range_u128) / total_u128) as u64;
            let new_high = (low_u128 + (u128::from(sym_high) * range_u128) / total_u128 - 1) as u64;
            (new_low, new_high)
        };
        self.low = new_low;
        self.high = new_high;

        while ((self.low ^ self.high) & self.half_range) == 0 {
            on_event(UpdateEvent::Shift(self.low))?;
            self.low = (self.low << 1) & self.state_mask;
            self.high = ((self.high << 1) & self.state_mask) | 1;
        }

        while (self.low & !self.high & self.quarter_range) != 0 {
            on_event(UpdateEvent::Underflow)?;
            self.low = ((self.low << 1) ^ self.half_range) & self.state_mask;
            self.high =
                (((self.high ^ self.half_range) << 1) | self.half_range | 1) & self.state_mask;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum UpdateEvent {
    Shift(u64),
    Underflow,
}

/// Range arithmetic encoder writing big-endian bits to a [`BitOutputStream`].
#[derive(Debug)]
pub struct ArithmeticEncoder<W> {
    core: CoderCore,
    output: BitOutputStream<W>,
    num_underflow: u64,
}

impl<W: Write> ArithmeticEncoder<W> {
    /// Constructs an encoder using `num_state_bits` (typically [`DEFAULT_STATE_BITS`]).
    ///
    /// # Errors
    ///
    /// Returns [`ArithError::InvalidStateBits`] if `num_state_bits` is not in `1..=62`.
    pub fn new(num_state_bits: u32, output: BitOutputStream<W>) -> Result<Self, ArithError> {
        Ok(Self {
            core: CoderCore::new(num_state_bits)?,
            output,
            num_underflow: 0,
        })
    }

    /// Encodes `symbol` using the cumulative frequency table `freqs`.
    ///
    /// `freqs[i]` must hold the inclusive cumulative frequency of symbols `0..=i`,
    /// with `freqs.last()` equal to the total frequency mass (`<= minimum_range`).
    ///
    /// # Errors
    ///
    /// Returns [`ArithError`] if `freqs` or `symbol` is invalid or writing fails.
    #[inline]
    pub fn write(&mut self, freqs: &[u32], symbol: usize) -> Result<(), ArithError> {
        let num_state_bits = self.core.num_state_bits;
        let output = &mut self.output;
        let num_underflow = &mut self.num_underflow;
        self.core.update(freqs, symbol, |event| match event {
            UpdateEvent::Shift(low) => {
                let bit = ((low >> (num_state_bits - 1)) & 1) as u8;
                output.write_bit(bit)?;
                for _ in 0..*num_underflow {
                    output.write_bit(bit ^ 1)?;
                }
                *num_underflow = 0;
                Ok(())
            }
            UpdateEvent::Underflow => {
                *num_underflow += 1;
                Ok(())
            }
        })
    }

    /// Terminates the arithmetic stream by emitting the final disambiguating bit.
    ///
    /// Call [`BitOutputStream::finish_padding`] on [`Self::bit_stream_mut`] afterwards to pad
    /// the final partial byte, or call [`Self::into_inner`] directly instead of calling `finish`.
    ///
    /// # Errors
    ///
    /// Returns [`ArithError::Io`] if writing to the underlying bit stream fails.
    pub fn finish(&mut self) -> Result<(), ArithError> {
        self.output.write_bit(1)
    }

    /// Captures the minimal exact encoder state for checkpointing.
    pub fn state(&self) -> EncoderState {
        EncoderState {
            num_state_bits: self.core.num_state_bits,
            low: self.core.low,
            high: self.core.high,
            num_underflow: self.num_underflow,
        }
    }

    /// Restores a previously captured [`EncoderState`].
    ///
    /// Note that the underlying [`BitOutputStream`] must also be restored separately.
    ///
    /// # Errors
    ///
    /// Returns [`ArithError::StateWidthMismatch`] if `num_state_bits` differs.
    pub fn restore_state(&mut self, state: EncoderState) -> Result<(), ArithError> {
        if state.num_state_bits != self.core.num_state_bits {
            return Err(ArithError::StateWidthMismatch {
                expected: self.core.num_state_bits,
                found: state.num_state_bits,
            });
        }
        self.core.low = state.low;
        self.core.high = state.high;
        self.num_underflow = state.num_underflow;
        Ok(())
    }

    /// Returns a shared reference to the underlying [`BitOutputStream`].
    pub fn bit_stream(&self) -> &BitOutputStream<W> {
        &self.output
    }

    /// Returns a mutable reference to the underlying [`BitOutputStream`].
    pub fn bit_stream_mut(&mut self) -> &mut BitOutputStream<W> {
        &mut self.output
    }

    /// Finishes arithmetic coding, pads the bit stream, and returns the underlying writer.
    ///
    /// # Errors
    ///
    /// Returns [`ArithError`] if finishing or flushing fails.
    pub fn into_inner(mut self) -> Result<W, ArithError> {
        self.finish()?;
        self.output.into_inner()
    }
}

/// Range arithmetic decoder reading big-endian bits from a [`BitInputStream`].
#[derive(Debug)]
pub struct ArithmeticDecoder<R> {
    core: CoderCore,
    input: BitInputStream<R>,
    code: u64,
}

impl<R: Read> ArithmeticDecoder<R> {
    /// Constructs a decoder using `num_state_bits` and reads the initial code word.
    ///
    /// # Errors
    ///
    /// Returns [`ArithError::InvalidStateBits`] or [`ArithError::Io`] on initial reads.
    pub fn new(num_state_bits: u32, mut input: BitInputStream<R>) -> Result<Self, ArithError> {
        let core = CoderCore::new(num_state_bits)?;
        let mut code = 0u64;
        for _ in 0..num_state_bits {
            let bit = u64::from(input.read_bit()?.unwrap_or(0));
            code = (code << 1) | bit;
        }
        Ok(Self { core, input, code })
    }

    /// Decodes the next symbol using the cumulative frequency table `freqs`.
    ///
    /// # Errors
    ///
    /// Returns [`ArithError`] if `freqs` is invalid or reading from the bit stream fails.
    #[inline]
    pub fn read(&mut self, freqs: &[u32]) -> Result<usize, ArithError> {
        let total = self.core.validate_table(freqs)?;
        let value = if self.core.num_state_bits <= 32 {
            let range = self.core.high - self.core.low + 1;
            let offset = self.code - self.core.low;
            ((offset + 1) * total - 1) / range
        } else {
            let range = u128::from(self.core.high - self.core.low + 1);
            let offset = u128::from(self.code - self.core.low);
            (((offset + 1) * u128::from(total) - 1) / range) as u64
        };

        let value_u32 = value as u32;
        let symbol = freqs.partition_point(|&cum| cum <= value_u32);

        let half_range = self.core.half_range;
        let state_mask = self.core.state_mask;
        let input = &mut self.input;
        let code = &mut self.code;
        self.core.update(freqs, symbol, |event| {
            let bit = u64::from(input.read_bit()?.unwrap_or(0));
            match event {
                UpdateEvent::Shift(_) => {
                    *code = ((*code << 1) & state_mask) | bit;
                }
                UpdateEvent::Underflow => {
                    *code = (*code & half_range) | ((*code << 1) & (state_mask >> 1)) | bit;
                }
            }
            Ok(())
        })?;

        Ok(symbol)
    }

    /// Captures the minimal exact decoder state for checkpointing.
    pub fn state(&self) -> DecoderState {
        DecoderState {
            num_state_bits: self.core.num_state_bits,
            low: self.core.low,
            high: self.core.high,
            code: self.code,
        }
    }

    /// Restores a previously captured [`DecoderState`].
    ///
    /// Note that the underlying [`BitInputStream`] must also be restored separately.
    ///
    /// # Errors
    ///
    /// Returns [`ArithError::StateWidthMismatch`] if `num_state_bits` differs.
    pub fn restore_state(&mut self, state: DecoderState) -> Result<(), ArithError> {
        if state.num_state_bits != self.core.num_state_bits {
            return Err(ArithError::StateWidthMismatch {
                expected: self.core.num_state_bits,
                found: state.num_state_bits,
            });
        }
        self.core.low = state.low;
        self.core.high = state.high;
        self.code = state.code;
        Ok(())
    }

    /// Returns a shared reference to the underlying [`BitInputStream`].
    pub fn bit_stream(&self) -> &BitInputStream<R> {
        &self.input
    }

    /// Returns a mutable reference to the underlying [`BitInputStream`].
    pub fn bit_stream_mut(&mut self) -> &mut BitInputStream<R> {
        &mut self.input
    }

    /// Consumes this decoder and returns the underlying [`BitInputStream`].
    pub fn into_bit_stream(self) -> BitInputStream<R> {
        self.input
    }
}

/// Converts raw logits into a cumulative integer frequency table matching `nn.py`.
///
/// Computes `softmax(logits)`, scales by [`FREQ_TOTAL`], rounds ties to even,
/// clamps each symbol frequency to at least `1`, and returns the cumulative sum.
#[must_use]
pub fn make_freq_table(logits: &[f32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(logits.len());
    make_freq_table_into(logits, &mut out);
    out
}

/// Converts raw logits into a cumulative integer frequency table, reusing `out`.
pub fn make_freq_table_into(logits: &[f32], out: &mut Vec<u32>) {
    out.clear();
    if logits.is_empty() {
        return;
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    out.reserve(logits.len());
    // Pass 1: store positive exp(x - max_logit) bit patterns directly in `out`
    // so we avoid computing `f32::exp` twice with zero extra allocations.
    for &x in logits {
        let e = (x - max_logit).exp();
        sum_exp += e;
        out.push(e.to_bits());
    }
    let inv_sum = 1.0f32 / sum_exp;
    let scale = FREQ_TOTAL as f32;
    let mut cum = 0u32;
    for slot in out.iter_mut() {
        let prob = f32::from_bits(*slot) * inv_sum;
        let rounded = (prob * scale).round_ties_even();
        let freq = (rounded as i64).max(1) as u32;
        cum = cum.saturating_add(freq);
        *slot = cum;
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn roundtrip_uniform_and_skewed_tables() {
        let tables: &[&[u32]] = &[
            &[1, 2, 3, 4],
            &[999_997, 999_998, 999_999, 1_000_000],
            &[1, 500_000, 500_001, 1_000_000],
            &[10_000_000],
        ];
        let symbols = [0usize, 3, 0, 2, 1, 0, 0, 1, 2, 3, 0, 0];

        let mut encoded = Vec::new();
        {
            let bit_out = BitOutputStream::new(&mut encoded);
            let mut enc = ArithmeticEncoder::new(DEFAULT_STATE_BITS, bit_out).unwrap();
            for (i, &sym) in symbols.iter().enumerate() {
                let table = tables[i % (tables.len() - 1)];
                enc.write(table, sym).unwrap();
            }
            enc.write(tables[3], 0).unwrap();
            enc.finish().unwrap();
            enc.bit_stream_mut().finish_padding().unwrap();
        }

        let bit_in = BitInputStream::new(Cursor::new(&encoded));
        let mut dec = ArithmeticDecoder::new(DEFAULT_STATE_BITS, bit_in).unwrap();
        for (i, &want) in symbols.iter().enumerate() {
            let table = tables[i % (tables.len() - 1)];
            let got = dec.read(table).unwrap();
            assert_eq!(got, want, "mismatch at symbol index {i}");
        }
        assert_eq!(dec.read(tables[3]).unwrap(), 0);
    }

    #[test]
    fn roundtrip_make_freq_table_4097_vocab() {
        let mut logits = vec![-12.0f32; 4097];
        logits[0] = 4.5;
        logits[42] = 6.25;
        logits[4096] = 2.0;
        let freqs = make_freq_table(&logits);
        assert_eq!(freqs.len(), 4097);
        assert!(*freqs.last().unwrap() >= FREQ_TOTAL);

        let symbols = [42usize, 0, 4096, 1234, 42, 42, 999, 1, 4095];
        let mut buf = Vec::new();
        {
            let mut enc =
                ArithmeticEncoder::new(DEFAULT_STATE_BITS, BitOutputStream::new(&mut buf)).unwrap();
            for &sym in &symbols {
                enc.write(&freqs, sym).unwrap();
            }
            enc.into_inner().unwrap();
        }

        let mut dec =
            ArithmeticDecoder::new(DEFAULT_STATE_BITS, BitInputStream::new(Cursor::new(&buf)))
                .unwrap();
        let decoded: Vec<usize> = (0..symbols.len())
            .map(|_| dec.read(&freqs).unwrap())
            .collect();
        assert_eq!(decoded, symbols);
    }

    #[test]
    fn checkpoint_resume_encoder_and_decoder_bit_exact() {
        let mut logits_a = vec![-3.0f32; 64];
        logits_a[5] = 3.0;
        logits_a[17] = 1.5;
        let mut logits_b = vec![-4.0f32; 64];
        logits_b[63] = 2.5;
        logits_b[0] = 2.2;
        let table_a = make_freq_table(&logits_a);
        let table_b = make_freq_table(&logits_b);

        let stream_syms: Vec<usize> = (0..120).map(|i| (i * 19 + 5) % 64).collect();
        let split = 57;

        // Uninterrupted encode
        let mut full_bytes = Vec::new();
        {
            let mut enc =
                ArithmeticEncoder::new(DEFAULT_STATE_BITS, BitOutputStream::new(&mut full_bytes))
                    .unwrap();
            for (i, &sym) in stream_syms.iter().enumerate() {
                let tbl = if i % 2 == 0 { &table_a } else { &table_b };
                enc.write(tbl, sym).unwrap();
            }
            enc.into_inner().unwrap();
        }

        // Split encode with state snapshot + resume on temporary file
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let (enc_state, bit_out_state) = {
            let file = tmp.reopen().unwrap();
            let mut enc =
                ArithmeticEncoder::new(DEFAULT_STATE_BITS, BitOutputStream::new(file)).unwrap();
            for (i, &sym) in stream_syms[..split].iter().enumerate() {
                let tbl = if i % 2 == 0 { &table_a } else { &table_b };
                enc.write(tbl, sym).unwrap();
            }
            let es = enc.state();
            let bs = enc.bit_stream_mut().state().unwrap();
            // Simulate extra trailing bytes written before crash
            enc.write(&table_a, 11).unwrap();
            enc.write(&table_b, 22).unwrap();
            enc.bit_stream_mut().flush().unwrap();
            (es, bs)
        };

        {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(tmp.path())
                .unwrap();
            let mut bit_out = BitOutputStream::new(file);
            bit_out.restore_file_state(bit_out_state).unwrap();
            let mut enc = ArithmeticEncoder::new(DEFAULT_STATE_BITS, bit_out).unwrap();
            enc.restore_state(enc_state).unwrap();
            for (offset, &sym) in stream_syms[split..].iter().enumerate() {
                let i = split + offset;
                let tbl = if i % 2 == 0 { &table_a } else { &table_b };
                enc.write(tbl, sym).unwrap();
            }
            enc.into_inner().unwrap();
        }

        let resumed_bytes = std::fs::read(tmp.path()).unwrap();
        assert_eq!(resumed_bytes, full_bytes);

        // Split decode with state snapshot + resume
        let mut dec = ArithmeticDecoder::new(
            DEFAULT_STATE_BITS,
            BitInputStream::new(Cursor::new(&full_bytes)),
        )
        .unwrap();
        let mut decoded = Vec::new();
        for i in 0..split {
            let tbl = if i % 2 == 0 { &table_a } else { &table_b };
            decoded.push(dec.read(tbl).unwrap());
        }
        let dec_state = dec.state();
        let bit_in_state = dec.bit_stream_mut().state().unwrap();

        let mut dec2 = ArithmeticDecoder::new(
            DEFAULT_STATE_BITS,
            BitInputStream::new(Cursor::new(&full_bytes)),
        )
        .unwrap();
        dec2.bit_stream_mut().restore_state(bit_in_state).unwrap();
        dec2.restore_state(dec_state).unwrap();
        for i in split..stream_syms.len() {
            let tbl = if i % 2 == 0 { &table_a } else { &table_b };
            decoded.push(dec2.read(tbl).unwrap());
        }
        assert_eq!(decoded, stream_syms);
    }

    #[test]
    fn error_handling_edges() {
        assert!(matches!(
            CoderCore::new(0),
            Err(ArithError::InvalidStateBits(0))
        ));
        assert!(matches!(
            CoderCore::new(63),
            Err(ArithError::InvalidStateBits(63))
        ));

        let mut buf = Vec::new();
        let mut enc =
            ArithmeticEncoder::new(DEFAULT_STATE_BITS, BitOutputStream::new(&mut buf)).unwrap();
        assert!(matches!(
            enc.write(&[], 0),
            Err(ArithError::InvalidSymbol { .. })
        ));
        assert!(matches!(
            enc.write(&[10, 10], 1),
            Err(ArithError::ZeroFrequencySymbol(1))
        ));
        assert!(matches!(
            enc.bit_stream_mut().write_bit(2),
            Err(ArithError::InvalidBit(2))
        ));
    }

    #[test]
    fn single_pass_freq_table_matches_two_pass_reference_bit_for_bit() {
        fn ref_two_pass(logits: &[f32]) -> Vec<u32> {
            if logits.is_empty() {
                return Vec::new();
            }
            let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for &x in logits {
                sum_exp += (x - max_logit).exp();
            }
            let inv_sum = 1.0f32 / sum_exp;
            let scale = FREQ_TOTAL as f32;
            let mut cum = 0u32;
            let mut out = Vec::with_capacity(logits.len());
            for &x in logits {
                let prob = (x - max_logit).exp() * inv_sum;
                let rounded = (prob * scale).round_ties_even();
                let freq = (rounded as i64).max(1) as u32;
                cum = cum.saturating_add(freq);
                out.push(cum);
            }
            out
        }

        let cases: &[&[f32]] = &[&[], &[0.0], &[-100.0, 0.0, 100.0, -50.0, 99.5], &[1.0; 128]];
        for case in cases {
            assert_eq!(make_freq_table(case), ref_two_pass(case));
        }

        let mut large = vec![-12.0f32; 4097];
        for (i, slot) in large.iter_mut().enumerate() {
            *slot = -6.0 + ((i * 37) % 101) as f32 * 0.11;
        }
        large[42] = 9.75;
        let mut reused = Vec::with_capacity(4097);
        make_freq_table_into(&large, &mut reused);
        assert_eq!(reused, ref_two_pass(&large));
    }

    #[test]
    fn u64_fast_path_matches_32bit_and_wide_state_roundtrip() {
        // Test 32-bit u64 fast path with FREQ_TOTAL ~10M logits and sparse zero-frequency CDFs,
        // plus 40-bit u128 fallback path.
        let mut logits = vec![-12.0f32; 512];
        logits[0] = 6.0;
        logits[255] = 4.5;
        logits[511] = 3.0;
        let dense_cdf = make_freq_table(&logits);
        // Construct a sparse CDF containing zero-frequency intervals between active symbols.
        let sparse_cdf = vec![10, 10, 10, 1_000_000, 1_000_000, FREQ_TOTAL, FREQ_TOTAL];

        let dense_syms = [0usize, 1, 255, 511, 0, 510, 511, 42, 255, 0];
        let sparse_syms = [0usize, 3, 5, 0, 5, 3];

        for bits in [DEFAULT_STATE_BITS, 40u32] {
            let mut bytes = Vec::new();
            let mut enc = ArithmeticEncoder::new(bits, BitOutputStream::new(&mut bytes)).unwrap();
            for &s in &dense_syms {
                enc.write(&dense_cdf, s).unwrap();
            }
            for &s in &sparse_syms {
                enc.write(&sparse_cdf, s).unwrap();
            }
            let bytes = enc.into_inner().unwrap();

            let mut dec =
                ArithmeticDecoder::new(bits, BitInputStream::new(Cursor::new(&bytes))).unwrap();
            let mut got_dense = Vec::with_capacity(dense_syms.len());
            for _ in 0..dense_syms.len() {
                got_dense.push(dec.read(&dense_cdf).unwrap());
            }
            let mut got_sparse = Vec::with_capacity(sparse_syms.len());
            for _ in 0..sparse_syms.len() {
                got_sparse.push(dec.read(&sparse_cdf).unwrap());
            }
            assert_eq!(got_dense, dense_syms, "bits={bits}");
            assert_eq!(got_sparse, sparse_syms, "bits={bits}");
        }
    }
}
