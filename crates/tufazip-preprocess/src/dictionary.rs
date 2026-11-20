use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::error::PreprocessError;

const MAX_WORD_BYTES: usize = 4096;

/// NNCP token dictionary mapping 16-bit symbol IDs (`0..vocab_size`) to raw word byte slices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dictionary {
    words: Vec<Vec<u8>>,
}

impl Dictionary {
    /// Creates a dictionary directly from its ordered word byte entries.
    #[must_use]
    pub fn from_words(words: Vec<Vec<u8>>) -> Self {
        Self { words }
    }

    /// Loads a `.voc` dictionary file in Bellard's escaped line format.
    ///
    /// # Errors
    ///
    /// Returns [`PreprocessError`] on I/O failure, invalid escape sequences, or entries
    /// longer than 4096 bytes.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, PreprocessError> {
        let file = File::open(path)?;
        Self::from_reader(BufReader::new(file))
    }

    /// Parses a `.voc` dictionary from any [`Read`] source.
    ///
    /// # Errors
    ///
    /// Returns [`PreprocessError`] on read errors, invalid escapes, or oversized entries.
    pub fn from_reader<R: Read>(reader: R) -> Result<Self, PreprocessError> {
        let mut bytes = BufReader::new(reader).bytes();
        let mut words = Vec::new();
        let mut current = Vec::new();

        while let Some(res) = bytes.next() {
            let mut b = res?;
            if b == b'\n' {
                if !current.is_empty() {
                    words.push(std::mem::take(&mut current));
                }
            } else {
                if b == b'\\' {
                    let Some(esc_res) = bytes.next() else {
                        break;
                    };
                    let esc = esc_res?;
                    b = match esc {
                        b'n' => b'\n',
                        b'\\' => b'\\',
                        other => return Err(PreprocessError::InvalidEscape(other)),
                    };
                }
                if current.len() >= MAX_WORD_BYTES {
                    return Err(PreprocessError::WordTooLong);
                }
                current.push(b);
            }
        }

        Ok(Self { words })
    }

    /// Writes this dictionary to `path` in Bellard's escaped line format.
    ///
    /// # Errors
    ///
    /// Returns [`PreprocessError::Io`] on write failure.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), PreprocessError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        self.write_to(&mut writer)?;
        writer.flush()?;
        Ok(())
    }

    /// Serializes this dictionary to `writer` in Bellard's escaped line format.
    ///
    /// # Errors
    ///
    /// Returns [`PreprocessError::Io`] on write failure.
    pub fn write_to(&self, writer: &mut impl Write) -> Result<(), PreprocessError> {
        for word in &self.words {
            for &b in word {
                match b {
                    b'\n' => writer.write_all(b"\\n")?,
                    b'\\' => writer.write_all(b"\\\\")?,
                    other => writer.write_all(&[other])?,
                }
            }
            writer.write_all(b"\n")?;
        }
        Ok(())
    }

    /// Returns all dictionary entries in symbol-ID order.
    #[must_use]
    pub fn words(&self) -> &[Vec<u8>] {
        &self.words
    }

    /// Returns the number of entries in the dictionary.
    #[must_use]
    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// Returns `true` if the dictionary contains zero entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Looks up the raw byte sequence for symbol ID `symbol`.
    #[must_use]
    pub fn get(&self, symbol: u16) -> Option<&[u8]> {
        self.words.get(usize::from(symbol)).map(Vec::as_slice)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use sha2::{Digest, Sha256};

    use super::*;

    #[test]
    fn enwik9_voc_roundtrip_matches_exact_bytes_and_sha256() {
        let voc_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/enwik9.voc");
        let original = std::fs::read(&voc_path).expect("assets/enwik9.voc should exist");
        let dict = Dictionary::load(&voc_path).unwrap();
        assert_eq!(dict.len(), 4097);
        assert_eq!(dict.get(0), Some(&[0x00u8][..]));
        assert_eq!(dict.get(10), Some(&[0x0Au8][..]));
        assert_eq!(dict.get(92), Some(&[0x5Cu8][..]));

        let mut reserialized = Vec::with_capacity(original.len());
        dict.write_to(&mut reserialized).unwrap();
        assert_eq!(reserialized.len(), 34_421);
        assert_eq!(reserialized, original);

        let digest = format!("{:x}", Sha256::digest(&reserialized));
        assert_eq!(
            digest,
            "a63e517783e84a778f3f08c1f2321f3e2035e64d34c8e9206f03591460e44629"
        );
    }

    #[test]
    fn dictionary_rejects_invalid_escape_and_oversized_word() {
        let bad_escape = b"abc\\xdef\n";
        assert!(matches!(
            Dictionary::from_reader(&bad_escape[..]),
            Err(PreprocessError::InvalidEscape(b'x'))
        ));

        let mut too_long = vec![b'a'; 4097];
        too_long.push(b'\n');
        assert!(matches!(
            Dictionary::from_reader(&too_long[..]),
            Err(PreprocessError::WordTooLong)
        ));
    }
}
