use candle_core::{DType, Device, Shape, Tensor};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::error::CodecError;

/// Exact serializable state of [`DeterministicRng`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RngState {
    /// 256-bit ChaCha8 seed.
    pub seed: [u8; 32],
    /// Stream identifier.
    pub stream: u64,
    /// Lower 64 bits of the 128-bit ChaCha word position.
    pub word_pos_lo: u64,
    /// Upper 64 bits of the 128-bit ChaCha word position.
    pub word_pos_hi: u64,
}

/// Deterministic, checkpointable PRNG for weight initialization and replay dropout.
#[derive(Debug, Clone)]
pub struct DeterministicRng {
    inner: ChaCha8Rng,
}

impl DeterministicRng {
    /// Constructs a PRNG deterministically from a 64-bit seed.
    #[must_use]
    pub fn from_seed_u64(seed: u64) -> Self {
        Self {
            inner: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Captures the exact internal PRNG state for resumable checkpoints.
    #[must_use]
    pub fn state(&self) -> RngState {
        let word_pos = self.inner.get_word_pos();
        RngState {
            seed: self.inner.get_seed(),
            stream: self.inner.get_stream(),
            word_pos_lo: word_pos as u64,
            word_pos_hi: (word_pos >> 64) as u64,
        }
    }

    /// Restores a previously captured [`RngState`].
    pub fn restore_state(&mut self, state: &RngState) {
        let mut inner = ChaCha8Rng::from_seed(state.seed);
        inner.set_stream(state.stream);
        let word_pos = (u128::from(state.word_pos_hi) << 64) | u128::from(state.word_pos_lo);
        inner.set_word_pos(word_pos);
        self.inner = inner;
    }

    /// Samples `len` i.i.d. normal values `N(mean, std^2)`.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::InvalidConfig`] if `std` is negative or non-finite.
    pub fn sample_normal_vec(
        &mut self,
        len: usize,
        mean: f32,
        std: f32,
    ) -> Result<Vec<f32>, CodecError> {
        let dist = Normal::new(mean, std).map_err(|e| {
            CodecError::InvalidConfig(format!("invalid normal distribution parameters: {e}"))
        })?;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(dist.sample(&mut self.inner));
        }
        Ok(out)
    }

    /// Generates an inverted-dropout scaling mask of `shape` on `device` in `dtype`.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError`] on invalid `drop_p` or tensor transfer failure.
    pub fn dropout_mask(
        &mut self,
        shape: &Shape,
        drop_p: f32,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor, CodecError> {
        if !(0.0..1.0).contains(&drop_p) {
            return Err(CodecError::InvalidConfig(format!(
                "dropout probability must be in [0.0, 1.0), got {drop_p}"
            )));
        }
        let n = shape.elem_count();
        let keep_prob = 1.0f32 - drop_p;
        let scale = 1.0f32 / keep_prob;
        let mut mask = Vec::with_capacity(n);
        for _ in 0..n {
            let draw: f32 = self.inner.random();
            mask.push(if draw >= drop_p { scale } else { 0.0 });
        }
        let tensor = Tensor::from_vec(mask, shape.clone(), device)?;
        if dtype == DType::F32 {
            Ok(tensor)
        } else {
            Ok(tensor.to_dtype(dtype)?)
        }
    }
}
