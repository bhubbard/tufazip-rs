//! Port of Project Nayuki's Reference arithmetic coding JUnit test suite
//! (`ArithmeticCodingTest.java`, `ArithmeticCompressTest.java`,
//! `AdaptiveArithmeticCompressTest.java`) plus bit-stream and state-width checks.

use std::io::Cursor;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tufazip_arith::{
    ArithError, ArithmeticDecoder, ArithmeticEncoder, BitInputState, BitInputStream,
    BitOutputState, BitOutputStream, DecoderState, EncoderState, DEFAULT_STATE_BITS,
};

const EOF_SYM: usize = 256;
const SYMBOL_LIMIT: usize = 257;

fn build_cumulative(counts: &[u32; SYMBOL_LIMIT]) -> [u32; SYMBOL_LIMIT] {
    let mut cum = [0u32; SYMBOL_LIMIT];
    let mut running = 0u32;
    for (dst, &count) in cum.iter_mut().zip(counts.iter()) {
        running += count;
        *dst = running;
    }
    cum
}

fn static_compress(input: &[u8], num_state_bits: u32) -> Vec<u8> {
    let mut counts = [0u32; SYMBOL_LIMIT];
    for &b in input {
        counts[usize::from(b)] += 1;
    }
    counts[EOF_SYM] = 1;
    let cum = build_cumulative(&counts);

    let mut out = Vec::with_capacity(256 * 4 + input.len());
    for &c in &counts[..256] {
        out.extend_from_slice(&c.to_be_bytes());
    }

    let bit_out = BitOutputStream::new(&mut out);
    let mut enc = ArithmeticEncoder::new(num_state_bits, bit_out).unwrap();
    for &b in input {
        enc.write(&cum, usize::from(b)).unwrap();
    }
    enc.write(&cum, EOF_SYM).unwrap();
    enc.into_inner().unwrap();
    out
}

fn static_decompress(compressed: &[u8], num_state_bits: u32) -> Vec<u8> {
    let mut counts = [0u32; SYMBOL_LIMIT];
    for (i, chunk) in compressed[..256 * 4].chunks_exact(4).enumerate() {
        counts[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    counts[EOF_SYM] = 1;
    let cum = build_cumulative(&counts);

    let bit_in = BitInputStream::new(Cursor::new(&compressed[256 * 4..]));
    let mut dec = ArithmeticDecoder::new(num_state_bits, bit_in).unwrap();
    let mut out = Vec::new();
    loop {
        let sym = dec.read(&cum).unwrap();
        if sym == EOF_SYM {
            break;
        }
        out.push(sym as u8);
    }
    out
}

fn adaptive_compress(input: &[u8], num_state_bits: u32) -> Vec<u8> {
    let mut counts = [1u32; SYMBOL_LIMIT];
    let mut cum = build_cumulative(&counts);
    let mut out = Vec::with_capacity(input.len() + 8);
    let bit_out = BitOutputStream::new(&mut out);
    let mut enc = ArithmeticEncoder::new(num_state_bits, bit_out).unwrap();
    for &b in input {
        let sym = usize::from(b);
        enc.write(&cum, sym).unwrap();
        counts[sym] += 1;
        for c in &mut cum[sym..] {
            *c += 1;
        }
    }
    enc.write(&cum, EOF_SYM).unwrap();
    enc.into_inner().unwrap();
    out
}

fn adaptive_decompress(compressed: &[u8], num_state_bits: u32) -> Vec<u8> {
    let mut counts = [1u32; SYMBOL_LIMIT];
    let mut cum = build_cumulative(&counts);
    let bit_in = BitInputStream::new(Cursor::new(compressed));
    let mut dec = ArithmeticDecoder::new(num_state_bits, bit_in).unwrap();
    let mut out = Vec::new();
    loop {
        let sym = dec.read(&cum).unwrap();
        if sym == EOF_SYM {
            break;
        }
        out.push(sym as u8);
        counts[sym] += 1;
        for c in &mut cum[sym..] {
            *c += 1;
        }
    }
    out
}

fn assert_both_roundtrip(msg: &[u8]) {
    let static_enc = static_compress(msg, DEFAULT_STATE_BITS);
    let static_dec = static_decompress(&static_enc, DEFAULT_STATE_BITS);
    assert_eq!(
        static_dec,
        msg,
        "static roundtrip failed (len={})",
        msg.len()
    );

    let adaptive_enc = adaptive_compress(msg, DEFAULT_STATE_BITS);
    let adaptive_dec = adaptive_decompress(&adaptive_enc, DEFAULT_STATE_BITS);
    assert_eq!(
        adaptive_dec,
        msg,
        "adaptive roundtrip failed (len={})",
        msg.len()
    );
}

#[test]
fn nayuki_test_empty() {
    assert_both_roundtrip(&[]);
}

#[test]
fn nayuki_test_one_symbol() {
    assert_both_roundtrip(&[0u8; 10]);
}

#[test]
fn nayuki_test_simple() {
    assert_both_roundtrip(&[0, 3, 1, 2]);
}

#[test]
fn nayuki_test_every_byte_value() {
    let all_bytes: Vec<u8> = (0u8..=255).collect();
    assert_both_roundtrip(&all_bytes);
}

#[test]
fn nayuki_test_underflow() {
    assert_both_roundtrip(&[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]);
}

#[test]
fn nayuki_test_uniform_random() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x5EED_0001);
    for _ in 0..100 {
        let len = rng.random_range(0..1000);
        let mut buf = vec![0u8; len];
        rng.fill(&mut buf[..]);
        assert_both_roundtrip(&buf);
    }
}

#[test]
fn nayuki_test_random_distribution() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x5EED_0002);
    for _ in 0..1000 {
        let m = rng.random_range(1..=255usize);
        let n = rng.random_range(0..1000usize).max(m);

        let mut freqs = vec![0usize; m];
        let mut total = 0usize;
        for f in &mut freqs {
            *f = rng.random_range(1..=10_000);
            total += *f;
        }

        let mut sum = 0usize;
        let mut index = 0usize;
        for (j, f) in freqs.iter_mut().enumerate() {
            let new_sum = sum + *f;
            let new_index = (n - m) * new_sum / total + j + 1;
            *f = new_index - index;
            sum = new_sum;
            index = new_index;
        }
        assert_eq!(index, n);

        let mut message = Vec::with_capacity(n);
        for (k, &count) in freqs.iter().enumerate() {
            message.extend(std::iter::repeat_n(k as u8, count));
        }

        for j in 0..message.len() {
            let k = rng.random_range(j..message.len());
            message.swap(j, k);
        }

        assert_both_roundtrip(&message);
    }
}

#[test]
fn bit_stream_eof_and_state_width_sweep() {
    let mut bit_in = BitInputStream::new(Cursor::new([0b1010_0000u8]));
    assert_eq!(bit_in.read_bit().unwrap(), Some(1));
    assert_eq!(bit_in.read_bit().unwrap(), Some(0));
    assert_eq!(bit_in.read_bit().unwrap(), Some(1));
    for _ in 0..5 {
        assert_eq!(bit_in.read_bit().unwrap(), Some(0));
    }
    assert_eq!(bit_in.read_bit().unwrap(), None);
    assert_eq!(bit_in.read_bit().unwrap(), None);
    assert!(matches!(
        bit_in.read_bit_no_eof(),
        Err(ArithError::UnexpectedEof)
    ));

    let bad_in_state = BitInputState {
        file_position: 0,
        current_byte: Some(0),
        num_bits_remaining: 9,
    };
    let mut seek_in = BitInputStream::new(Cursor::new(Vec::<u8>::new()));
    assert!(matches!(
        seek_in.restore_state(bad_in_state),
        Err(ArithError::InvalidStreamState(_))
    ));

    let bad_out_state = BitOutputState {
        file_position: 0,
        current_byte: 0,
        num_bits_filled: 8,
    };
    let mut seek_out = BitOutputStream::new(Cursor::new(Vec::<u8>::new()));
    assert!(matches!(
        seek_out.restore_state(bad_out_state),
        Err(ArithError::InvalidStreamState(_))
    ));

    let mut enc = ArithmeticEncoder::new(32, BitOutputStream::new(Vec::<u8>::new())).unwrap();
    assert!(matches!(
        enc.restore_state(EncoderState {
            num_state_bits: 24,
            low: 0,
            high: 0,
            num_underflow: 0,
        }),
        Err(ArithError::StateWidthMismatch {
            expected: 32,
            found: 24
        })
    ));

    let mut dec =
        ArithmeticDecoder::new(32, BitInputStream::new(Cursor::new(Vec::<u8>::new()))).unwrap();
    assert!(matches!(
        dec.restore_state(DecoderState {
            num_state_bits: 16,
            low: 0,
            high: 0,
            code: 0,
        }),
        Err(ArithError::StateWidthMismatch {
            expected: 32,
            found: 16
        })
    ));

    let msg = b"Testing arithmetic coding across 16, 24, 32, and 48 state bits!";
    for bits in [16u32, 24, 32, 48] {
        let s_enc = static_compress(msg, bits);
        assert_eq!(static_decompress(&s_enc, bits), msg);
        let a_enc = adaptive_compress(msg, bits);
        assert_eq!(adaptive_decompress(&a_enc, bits), msg);
    }
}
