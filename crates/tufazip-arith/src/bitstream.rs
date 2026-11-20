use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};

use serde::{Deserialize, Serialize};

use crate::error::ArithError;

/// Exact serializable state of a [`BitInputStream`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BitInputState {
    /// Byte offset of the underlying reader.
    pub file_position: u64,
    /// Currently buffered byte, or `None` once EOF has been reached.
    pub current_byte: Option<u8>,
    /// Number of unread bits remaining in `current_byte` (`0..=8`).
    pub num_bits_remaining: u8,
}

/// Exact serializable state of a [`BitOutputStream`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BitOutputState {
    /// Byte offset of the underlying writer.
    pub file_position: u64,
    /// Bits currently accumulated for the next output byte.
    pub current_byte: u8,
    /// Number of bits accumulated in `current_byte` (`0..=7`).
    pub num_bits_filled: u8,
}

/// Big-endian bit reader backed by a byte stream.
#[derive(Debug)]
pub struct BitInputStream<R> {
    input: R,
    current_byte: Option<u8>,
    num_bits_remaining: u8,
}

impl<R: Read> BitInputStream<R> {
    /// Constructs a bit input stream reading big-endian bits from `input`.
    pub fn new(input: R) -> Self {
        Self {
            input,
            current_byte: Some(0),
            num_bits_remaining: 0,
        }
    }

    /// Reads the next bit (`0` or `1`), or `None` once EOF is reached.
    ///
    /// # Errors
    ///
    /// Returns [`ArithError::Io`] if reading from the underlying byte stream fails.
    #[inline]
    pub fn read_bit(&mut self) -> Result<Option<u8>, ArithError> {
        let Some(mut byte) = self.current_byte else {
            return Ok(None);
        };
        if self.num_bits_remaining == 0 {
            let mut buf = [0u8; 1];
            let n = read_one_byte(&mut self.input, &mut buf)?;
            if n == 0 {
                self.current_byte = None;
                return Ok(None);
            }
            byte = buf[0];
            self.current_byte = Some(byte);
            self.num_bits_remaining = 8;
        }
        self.num_bits_remaining -= 1;
        Ok(Some((byte >> self.num_bits_remaining) & 1))
    }

    /// Reads the next bit (`0` or `1`), returning [`ArithError::UnexpectedEof`] at EOF.
    ///
    /// # Errors
    ///
    /// Returns [`ArithError::UnexpectedEof`] at EOF or [`ArithError::Io`] on read failure.
    pub fn read_bit_no_eof(&mut self) -> Result<u8, ArithError> {
        self.read_bit()?.ok_or(ArithError::UnexpectedEof)
    }

    /// Returns a shared reference to the underlying reader.
    pub fn get_ref(&self) -> &R {
        &self.input
    }

    /// Returns a mutable reference to the underlying reader.
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.input
    }

    /// Consumes this stream and returns the underlying reader.
    pub fn into_inner(self) -> R {
        self.input
    }
}

impl<R: Read + Seek> BitInputStream<R> {
    /// Captures the exact stream position and buffered byte state for checkpointing.
    ///
    /// # Errors
    ///
    /// Returns [`ArithError::Io`] if querying the reader's stream position fails.
    pub fn state(&mut self) -> Result<BitInputState, ArithError> {
        let file_position = self.input.stream_position()?;
        Ok(BitInputState {
            file_position,
            current_byte: self.current_byte,
            num_bits_remaining: self.num_bits_remaining,
        })
    }

    /// Restores a previously captured [`BitInputState`].
    ///
    /// # Errors
    ///
    /// Returns [`ArithError::InvalidStreamState`] if `num_bits_remaining > 8` or
    /// [`ArithError::Io`] if seeking fails.
    pub fn restore_state(&mut self, state: BitInputState) -> Result<(), ArithError> {
        if state.num_bits_remaining > 8 {
            return Err(ArithError::InvalidStreamState(format!(
                "num_bits_remaining {} exceeds 8",
                state.num_bits_remaining
            )));
        }
        self.input.seek(SeekFrom::Start(state.file_position))?;
        self.current_byte = state.current_byte;
        self.num_bits_remaining = state.num_bits_remaining;
        Ok(())
    }
}

/// Big-endian bit writer backed by a byte stream.
#[derive(Debug)]
pub struct BitOutputStream<W> {
    output: W,
    current_byte: u8,
    num_bits_filled: u8,
}

impl<W: Write> BitOutputStream<W> {
    /// Constructs a bit output stream writing big-endian bits to `output`.
    pub fn new(output: W) -> Self {
        Self {
            output,
            current_byte: 0,
            num_bits_filled: 0,
        }
    }

    /// Writes a single bit (`0` or `1`) to the stream.
    ///
    /// # Errors
    ///
    /// Returns [`ArithError::InvalidBit`] if `bit > 1`, or [`ArithError::Io`] on write failure.
    #[inline]
    pub fn write_bit(&mut self, bit: u8) -> Result<(), ArithError> {
        if bit > 1 {
            return Err(ArithError::InvalidBit(bit));
        }
        self.current_byte = (self.current_byte << 1) | bit;
        self.num_bits_filled += 1;
        if self.num_bits_filled == 8 {
            self.output.write_all(&[self.current_byte])?;
            self.current_byte = 0;
            self.num_bits_filled = 0;
        }
        Ok(())
    }

    /// Zero-pads any partial byte up to the next byte boundary and flushes the underlying writer.
    ///
    /// # Errors
    ///
    /// Returns [`ArithError::Io`] if writing or flushing fails.
    pub fn finish_padding(&mut self) -> Result<(), ArithError> {
        while self.num_bits_filled != 0 {
            self.write_bit(0)?;
        }
        self.output.flush()?;
        Ok(())
    }

    /// Flushes the underlying writer without padding the current partial byte.
    ///
    /// # Errors
    ///
    /// Returns [`ArithError::Io`] if flushing fails.
    pub fn flush(&mut self) -> Result<(), ArithError> {
        self.output.flush()?;
        Ok(())
    }

    /// Returns a shared reference to the underlying writer.
    pub fn get_ref(&self) -> &W {
        &self.output
    }

    /// Returns a mutable reference to the underlying writer.
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.output
    }

    /// Finishes padding, flushes, and returns the underlying writer.
    ///
    /// # Errors
    ///
    /// Returns [`ArithError::Io`] if padding or flushing fails.
    pub fn into_inner(mut self) -> Result<W, ArithError> {
        self.finish_padding()?;
        Ok(self.output)
    }
}

impl<W: Write + Seek> BitOutputStream<W> {
    /// Flushes buffered full bytes and captures the exact output stream state for checkpointing.
    ///
    /// # Errors
    ///
    /// Returns [`ArithError::Io`] if flushing or querying stream position fails.
    pub fn state(&mut self) -> Result<BitOutputState, ArithError> {
        self.output.flush()?;
        let file_position = self.output.stream_position()?;
        Ok(BitOutputState {
            file_position,
            current_byte: self.current_byte,
            num_bits_filled: self.num_bits_filled,
        })
    }

    /// Restores a previously captured [`BitOutputState`] by seeking to `state.file_position`.
    ///
    /// When resuming onto an existing file on disk, prefer [`Self::restore_file_state`]
    /// so any bytes written after the checkpoint are truncated.
    ///
    /// # Errors
    ///
    /// Returns [`ArithError::InvalidStreamState`] if `num_bits_filled >= 8`, or
    /// [`ArithError::Io`] on flush/seek errors.
    pub fn restore_state(&mut self, state: BitOutputState) -> Result<(), ArithError> {
        if state.num_bits_filled >= 8 {
            return Err(ArithError::InvalidStreamState(format!(
                "num_bits_filled {} must be < 8",
                state.num_bits_filled
            )));
        }
        self.output.flush()?;
        self.output.seek(SeekFrom::Start(state.file_position))?;
        self.current_byte = state.current_byte;
        self.num_bits_filled = state.num_bits_filled;
        Ok(())
    }
}

impl BitOutputStream<File> {
    /// Restores a [`BitOutputState`] on a file, truncating any bytes beyond `file_position`.
    ///
    /// # Errors
    ///
    /// Returns [`ArithError::InvalidStreamState`] or [`ArithError::Io`].
    pub fn restore_file_state(&mut self, state: BitOutputState) -> Result<(), ArithError> {
        if state.num_bits_filled >= 8 {
            return Err(ArithError::InvalidStreamState(format!(
                "num_bits_filled {} must be < 8",
                state.num_bits_filled
            )));
        }
        self.output.flush()?;
        self.output.set_len(state.file_position)?;
        self.output.seek(SeekFrom::Start(state.file_position))?;
        self.current_byte = state.current_byte;
        self.num_bits_filled = state.num_bits_filled;
        Ok(())
    }
}

impl BitOutputStream<BufWriter<File>> {
    /// Restores a [`BitOutputState`] on a buffered file, truncating any bytes beyond `file_position`.
    ///
    /// # Errors
    ///
    /// Returns [`ArithError::InvalidStreamState`] or [`ArithError::Io`].
    pub fn restore_file_state(&mut self, state: BitOutputState) -> Result<(), ArithError> {
        if state.num_bits_filled >= 8 {
            return Err(ArithError::InvalidStreamState(format!(
                "num_bits_filled {} must be < 8",
                state.num_bits_filled
            )));
        }
        self.output.flush()?;
        self.output.get_mut().set_len(state.file_position)?;
        self.output.seek(SeekFrom::Start(state.file_position))?;
        self.current_byte = state.current_byte;
        self.num_bits_filled = state.num_bits_filled;
        Ok(())
    }
}

fn read_one_byte<R: Read>(reader: &mut R, buf: &mut [u8; 1]) -> Result<usize, std::io::Error> {
    loop {
        match reader.read(buf) {
            Ok(n) => return Ok(n),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
}
