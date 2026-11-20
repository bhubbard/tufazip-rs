use std::io::Write;

/// Control byte suppressing the implicit leading space before the next word.
pub const CH_NO_SPACE: u8 = 1;

/// Control byte marking the next word as all uppercase.
pub const CH_TO_UPPER: u8 = 2;

/// Control byte capitalizing the first ASCII letter of the next word.
pub const CH_FIRST_UPPER: u8 = 3;

#[inline]
fn is_word_char(c: u8) -> bool {
    c.is_ascii_lowercase() || c.is_ascii_uppercase() || c >= 128
}

#[inline]
fn is_upper(c: u8) -> bool {
    c.is_ascii_uppercase()
}

#[inline]
fn is_lower(c: u8) -> bool {
    c.is_ascii_lowercase() || c >= 128
}

/// Applies Fabrice Bellard's reversible case-and-space encoding to raw input bytes.
///
/// Word runs are lowercased with space-elision and case markers
/// ([`CH_NO_SPACE`], [`CH_TO_UPPER`], [`CH_FIRST_UPPER`]), returning a `u16`
/// symbol buffer whose values are in `0..=255`.
///
/// Matches `preprocess.c` verbatim (including its quirk that a single leading ASCII
/// space at byte 0 followed by a word is absorbed by the decoder's initial `has_space = false`).
#[must_use]
pub fn case_space_encode(input: &[u8]) -> Vec<u16> {
    let mut obuf: Vec<u16> = Vec::with_capacity(input.len());
    let mut i = 0usize;
    while i < input.len() {
        if is_word_char(input[i]) {
            let mut j = i + 1;
            let ch_type = if is_lower(input[i]) {
                while j < input.len() && is_lower(input[j]) {
                    j += 1;
                }
                0u8
            } else if j < input.len() && is_upper(input[j]) {
                while j < input.len() && is_upper(input[j]) {
                    j += 1;
                }
                CH_TO_UPPER
            } else {
                while j < input.len() && is_lower(input[j]) {
                    j += 1;
                }
                CH_FIRST_UPPER
            };
            let len = j - i;
            let has_space = if obuf.is_empty() {
                true
            } else if obuf.last() == Some(&(u16::from(b' '))) {
                obuf.pop();
                true
            } else {
                false
            };

            if !has_space {
                obuf.push(u16::from(CH_NO_SPACE));
            }
            if ch_type != 0 {
                obuf.push(u16::from(ch_type));
            }
            obuf.push(u16::from(b' '));
            for &b in &input[i..i + len] {
                let lowered = if b.is_ascii_uppercase() {
                    b - b'A' + b'a'
                } else {
                    b
                };
                obuf.push(u16::from(lowered));
            }
            i += len;
        } else {
            obuf.push(u16::from(input[i]));
            i += 1;
        }
    }
    obuf
}

/// Stateful decoder inverting [`case_space_encode`] across streamed dictionary entries.
#[derive(Debug, Clone, Default)]
pub struct CaseSpaceDecoder {
    ch_type: u8,
    ch_type1: u8,
    has_space: bool,
}

impl CaseSpaceDecoder {
    /// Creates a decoder initialized at the start of a stream.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds one dictionary entry's raw bytes and writes the decoded output bytes to `out`.
    ///
    /// # Errors
    ///
    /// Returns any `std::io::Error` produced by `out`.
    pub fn push_bytes(&mut self, bytes: &[u8], out: &mut impl Write) -> std::io::Result<()> {
        for &raw in bytes {
            if raw == CH_TO_UPPER || raw == CH_FIRST_UPPER {
                self.ch_type = raw;
            } else if raw == CH_NO_SPACE {
                self.has_space = false;
            } else if raw == b' ' {
                self.ch_type1 = self.ch_type;
                self.ch_type = 0;
                if self.has_space {
                    out.write_all(b" ")?;
                }
                self.has_space = true;
            } else {
                let mut c = raw;
                if self.ch_type1 == CH_TO_UPPER || self.ch_type1 == CH_FIRST_UPPER {
                    if c.is_ascii_lowercase() {
                        c = c - b'a' + b'A';
                    }
                    if self.ch_type1 == CH_FIRST_UPPER {
                        self.ch_type1 = 0;
                    }
                }
                out.write_all(&[c])?;
            }
        }
        Ok(())
    }
}

/// Decodes a byte buffer produced by [`case_space_encode`] back to original bytes.
#[must_use]
pub fn case_space_decode(encoded: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(encoded.len());
    let mut decoder = CaseSpaceDecoder::new();
    // Writing to Vec<u8> is infallible.
    let _ = decoder.push_bytes(encoded, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_various_case_and_space_patterns() {
        let samples: &[&[u8]] = &[
            b"",
            b"Hello, world!",
            b"USA and NASA launched 42 rockets.",
            b"camelCase PascalCase snake_case kebab-case",
            b"leading   multiple  spaces\tand\nnewlines\n\n",
            b"Mixed Case: XML <title>Wikipedia</title> [[Link]]!",
            "UTF-8 café naïve 日本語 — em-dash".as_bytes(),
        ];

        for sample in samples {
            let encoded_u16 = case_space_encode(sample);
            let encoded_u8: Vec<u8> = encoded_u16.iter().map(|&v| v as u8).collect();
            let decoded = case_space_decode(&encoded_u8);
            assert_eq!(
                decoded.as_slice(),
                *sample,
                "failed roundtrip for {:?}",
                String::from_utf8_lossy(sample)
            );
        }
    }
}
