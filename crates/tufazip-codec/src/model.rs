use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use candle_core::{CpuStorage, CustomOp2, DType, Device, Layout, Shape, Storage, Tensor, Var, D};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tufazip_prior::{plan_depth_prior, read_archive, InitializationKind, PriorArchive};

use crate::config::{ModelConfig, NormStyle, PositionType};
use crate::error::CodecError;
use crate::rng::DeterministicRng;

const RMS_NORM_EPS: f64 = 1e-5;

/// Forward pass execution mode.
#[derive(Debug)]
pub enum ForwardMode<'a> {
    /// Deterministic evaluation mode (dropout disabled; used for online coding).
    Eval,
    /// Training/replay mode using the given deterministic PRNG for dropout.
    Train(&'a mut DeterministicRng),
}

/// Selects which sequence positions are projected through the vocabulary head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputSlice {
    /// Project all sequence positions `[0..seq]`.
    All,
    /// Project only the final sequence position (`seq - 1..seq`).
    Last,
    /// Project a contiguous sub-range `start..start + len`.
    Range {
        /// Starting token position index.
        start: usize,
        /// Number of positions to project.
        len: usize,
    },
}

/// Summary of loading a `QPI1` prior archive into a [`Transformer`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PriorLoadInfo {
    /// Archive size in bytes.
    pub archive_bytes: u64,
    /// SHA-256 hex digest of the archive file.
    pub archive_sha256: String,
    /// SHA-256 hex digest of the vocabulary file.
    pub vocab_sha256: String,
    /// Prior expansion kind (`depth-growth` or `loop-untying`).
    pub initialization_kind: InitializationKind,
    /// Number of physical source layers in the archive.
    pub source_depth: usize,
    /// Effective target-to-source block mapping.
    pub block_mapping: Vec<usize>,
    /// Optional recurrent schedule for loop-untying.
    pub block_schedule: Option<Vec<usize>>,
    /// Canonical SHA-256 digest of the initialized model state.
    pub initial_state_sha256: String,
    /// Optional quantization policy label from archive metadata.
    pub policy: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct FusedCpuRmsNorm {
    eps: f64,
}

impl FusedCpuRmsNorm {
    fn compute_row_fwd(x_row: &[f32], w: &[f32], out_row: &mut [f32], eps: f64) {
        let mut sq_acc = [0.0f32; 8];
        let x_chunks = x_row.chunks_exact(8);
        let x_rem = x_chunks.remainder();
        for xc in x_chunks {
            for i in 0..8 {
                let x_v = xc[i];
                sq_acc[i] += x_v * x_v;
            }
        }
        let mut sum_sq = 0.0f64;
        for &lane in &sq_acc {
            sum_sq += f64::from(lane);
        }
        for &v in x_rem {
            let fv = f64::from(v);
            sum_sq += fv * fv;
        }
        let inv_rms = (1.0 / (sum_sq / (x_row.len() as f64) + eps).sqrt()) as f32;
        for ((dst, &x_val), &w_val) in out_row.iter_mut().zip(x_row.iter()).zip(w.iter()) {
            *dst = x_val * inv_rms * w_val;
        }
    }

    fn compute_row_bwd(
        x_row: &[f32],
        dy_row: &[f32],
        w: &[f32],
        gx_row: &mut [f32],
        gw_acc: &mut [f64],
        eps: f64,
    ) {
        let dim = x_row.len();
        let mut sq_acc = [0.0f32; 8];
        let mut dot_acc = [0.0f32; 8];
        let x_chunks = x_row.chunks_exact(8);
        let dy_chunks = dy_row.chunks_exact(8);
        let w_chunks = w.chunks_exact(8);
        let x_rem = x_chunks.remainder();
        let dy_rem = dy_chunks.remainder();
        let w_rem = w_chunks.remainder();
        for ((xc, dyc), wc) in x_chunks.zip(dy_chunks).zip(w_chunks) {
            for i in 0..8 {
                let x_v = xc[i];
                sq_acc[i] += x_v * x_v;
                dot_acc[i] += dyc[i] * wc[i] * x_v;
            }
        }
        let mut sum_sq = 0.0f64;
        let mut dot = 0.0f64;
        for i in 0..8 {
            sum_sq += f64::from(sq_acc[i]);
            dot += f64::from(dot_acc[i]);
        }
        for ((&x_v, &dy_v), &w_v) in x_rem.iter().zip(dy_rem.iter()).zip(w_rem.iter()) {
            let x_f = f64::from(x_v);
            sum_sq += x_f * x_f;
            dot += f64::from(dy_v) * f64::from(w_v) * x_f;
        }
        let inv_rms = 1.0 / (sum_sq / (dim as f64) + eps).sqrt();
        let coeff = dot * inv_rms * inv_rms / (dim as f64);
        let inv_rms_f32 = inv_rms as f32;
        let coeff_f32 = coeff as f32;
        for ((((gx_slot, &dy_val), &w_val), &x_val), gw_slot) in gx_row
            .iter_mut()
            .zip(dy_row.iter())
            .zip(w.iter())
            .zip(x_row.iter())
            .zip(gw_acc.iter_mut())
        {
            *gx_slot = inv_rms_f32 * (dy_val * w_val - x_val * coeff_f32);
            let dy_f = f64::from(dy_val);
            let x_f = f64::from(x_val);
            *gw_slot += dy_f * x_f * inv_rms;
        }
    }
}

impl CustomOp2 for FusedCpuRmsNorm {
    fn name(&self) -> &'static str {
        "fused-cpu-rms-norm"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let (x_start, x_end) = l1.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("fused-cpu-rms-norm requires contiguous x".to_owned())
        })?;
        let (w_start, w_end) = l2.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("fused-cpu-rms-norm requires contiguous weight".to_owned())
        })?;
        let x_all = match s1 {
            CpuStorage::F32(v) => &v[x_start..x_end],
            _ => {
                return Err(candle_core::Error::Msg(
                    "fused-cpu-rms-norm expects F32 x".to_owned(),
                ))
            }
        };
        let w = match s2 {
            CpuStorage::F32(v) => &v[w_start..w_end],
            _ => {
                return Err(candle_core::Error::Msg(
                    "fused-cpu-rms-norm expects F32 weight".to_owned(),
                ))
            }
        };
        let dim = w.len();
        if dim == 0 || l1.dims().last() != Some(&dim) || x_all.len() % dim != 0 {
            return Err(candle_core::Error::Msg(format!(
                "fused-cpu-rms-norm shape mismatch: x={:?}, w={:?}",
                l1.shape(),
                l2.shape()
            )));
        }
        let mut out = vec![0.0f32; x_all.len()];
        let eps = self.eps;
        for (out_row, x_row) in out.chunks_exact_mut(dim).zip(x_all.chunks_exact(dim)) {
            Self::compute_row_fwd(x_row, w, out_row, eps);
        }
        Ok((CpuStorage::F32(out), l1.shape().clone()))
    }

    fn bwd(
        &self,
        arg1: &Tensor,
        arg2: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> candle_core::Result<(Option<Tensor>, Option<Tensor>)> {
        let x_c = if arg1.is_contiguous() && arg1.dtype() == DType::F32 {
            arg1.clone()
        } else {
            arg1.to_dtype(DType::F32)?.contiguous()?
        };
        let w_c = if arg2.is_contiguous() && arg2.dtype() == DType::F32 {
            arg2.clone()
        } else {
            arg2.to_dtype(DType::F32)?.contiguous()?
        };
        let dy_c = if grad_res.is_contiguous() && grad_res.dtype() == DType::F32 {
            grad_res.clone()
        } else {
            grad_res.to_dtype(DType::F32)?.contiguous()?
        };

        let (x_st, x_l) = x_c.storage_and_layout();
        let (w_st, w_l) = w_c.storage_and_layout();
        let (dy_st, dy_l) = dy_c.storage_and_layout();
        let (x_s, x_e) = x_l
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("non-contiguous x in RmsNorm bwd".to_owned()))?;
        let (w_s, w_e) = w_l
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("non-contiguous w in RmsNorm bwd".to_owned()))?;
        let (dy_s, dy_e) = dy_l.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("non-contiguous dy in RmsNorm bwd".to_owned())
        })?;

        let (x_slice, w_slice, dy_slice) = match (&*x_st, &*w_st, &*dy_st) {
            (
                Storage::Cpu(CpuStorage::F32(xv)),
                Storage::Cpu(CpuStorage::F32(wv)),
                Storage::Cpu(CpuStorage::F32(dyv)),
            ) => (&xv[x_s..x_e], &wv[w_s..w_e], &dyv[dy_s..dy_e]),
            _ => {
                return Err(candle_core::Error::Msg(
                    "fused-cpu-rms-norm bwd expects F32 CPU storage".to_owned(),
                ))
            }
        };

        let dim = w_slice.len();
        if dim == 0 || x_slice.is_empty() {
            return Ok((
                Some(Tensor::zeros(arg1.shape(), DType::F32, arg1.device())?),
                Some(Tensor::zeros(arg2.shape(), DType::F32, arg2.device())?),
            ));
        }

        let mut gx = vec![0.0f32; x_slice.len()];
        let mut gw_acc = vec![0.0f64; dim];
        let eps = self.eps;

        for ((gx_row, x_row), dy_row) in gx
            .chunks_exact_mut(dim)
            .zip(x_slice.chunks_exact(dim))
            .zip(dy_slice.chunks_exact(dim))
        {
            Self::compute_row_bwd(x_row, dy_row, w_slice, gx_row, &mut gw_acc, eps);
        }

        let gw: Vec<f32> = gw_acc.into_iter().map(|v| v as f32).collect();
        let gx_t = Tensor::from_vec(gx, arg1.shape().clone(), arg1.device())?;
        let gw_t = Tensor::from_vec(gw, arg2.shape().clone(), arg2.device())?;
        Ok((Some(gx_t), Some(gw_t)))
    }
}

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Option<Var>,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, enabled: bool, device: &Device) -> Result<Self, CodecError> {
        let weight = if enabled {
            Some(Var::ones(dim, DType::F32, device)?)
        } else {
            None
        };
        Ok(Self {
            weight,
            eps: RMS_NORM_EPS,
        })
    }

    fn forward_unfused(&self, x: &Tensor, weight: &Var) -> Result<Tensor, CodecError> {
        let x_dtype = x.dtype();
        let x_f32 = if x_dtype == DType::F32 {
            x.clone()
        } else {
            x.to_dtype(DType::F32)?
        };
        let var = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let inv_std = (var + self.eps)?.sqrt()?.recip()?;
        let normed = x_f32.broadcast_mul(&inv_std)?;
        let normed_cast = if x_dtype == DType::F32 {
            normed
        } else {
            normed.to_dtype(x_dtype)?
        };
        let w = if x_dtype == DType::F32 {
            weight.as_tensor().clone()
        } else {
            weight.as_tensor().to_dtype(x_dtype)?
        };
        Ok(normed_cast.broadcast_mul(&w)?)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, CodecError> {
        let Some(weight) = &self.weight else {
            return Ok(x.clone());
        };
        if x.device().is_cpu() && x.dtype() == DType::F32 && weight.dtype() == DType::F32 {
            let x_c = if x.is_contiguous() {
                x.clone()
            } else {
                x.contiguous()?
            };
            return Ok(x_c.apply_op2(weight.as_tensor(), FusedCpuRmsNorm { eps: self.eps })?);
        }
        self.forward_unfused(x, weight)
    }
}

fn linear_forward(x: &Tensor, weight: &Var, compute_dtype: DType) -> Result<Tensor, CodecError> {
    let w = if compute_dtype == DType::F32 {
        weight.as_tensor().clone()
    } else {
        weight.as_tensor().to_dtype(compute_dtype)?
    };
    let out_dim = w.dim(0)?;
    match *x.dims() {
        [batch, seq, hidden] => {
            let x_flat = x.contiguous()?.reshape((batch * seq, hidden))?;
            let out = x_flat.matmul(&w.t()?)?;
            Ok(out.reshape((batch, seq, out_dim))?)
        }
        _ => Ok(x.matmul(&w.t()?)?),
    }
}

fn apply_dropout(
    x: &Tensor,
    drop_p: f32,
    mode: &mut ForwardMode<'_>,
) -> Result<Tensor, CodecError> {
    if drop_p == 0.0 {
        return Ok(x.clone());
    }
    match mode {
        ForwardMode::Eval => Ok(x.clone()),
        ForwardMode::Train(rng) => {
            let mask = rng.dropout_mask(x.shape(), drop_p, x.dtype(), x.device())?;
            Ok(x.mul(&mask)?)
        }
    }
}

fn build_rope_cache(
    ctx_len: usize,
    head_dim: usize,
    base: f64,
    device: &Device,
) -> Result<(Tensor, Tensor), CodecError> {
    if head_dim % 2 != 0 {
        return Err(CodecError::InvalidConfig(format!(
            "head_dim ({head_dim}) must be even for RoPE"
        )));
    }
    let half = head_dim / 2;
    let mut cos_data = vec![0.0f32; ctx_len * head_dim];
    let mut sin_data = vec![0.0f32; ctx_len * head_dim];
    let inv_freq: Vec<f64> = (0..half)
        .map(|i| 1.0f64 / base.powf((2 * i) as f64 / head_dim as f64))
        .collect();

    for pos in 0..ctx_len {
        let row = pos * head_dim;
        for (i, &freq) in inv_freq.iter().enumerate() {
            let angle = (pos as f64) * freq;
            let c = angle.cos() as f32;
            let s = angle.sin() as f32;
            cos_data[row + i] = c;
            cos_data[row + half + i] = c;
            sin_data[row + i] = s;
            sin_data[row + half + i] = s;
        }
    }

    let cos = Tensor::from_vec(cos_data, (1, 1, ctx_len, head_dim), device)?;
    let sin = Tensor::from_vec(sin_data, (1, 1, ctx_len, head_dim), device)?;
    Ok((cos, sin))
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor, CodecError> {
    let head_dim = x.dim(D::Minus1)?;
    let half = head_dim / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    let term1 = x.broadcast_mul(cos)?;
    let term2 = rotated.broadcast_mul(sin)?;
    Ok((term1 + term2)?)
}

fn build_causal_mask(ctx_len: usize, device: &Device) -> Result<Tensor, CodecError> {
    let mut data = vec![0.0f32; ctx_len * ctx_len];
    for i in 0..ctx_len {
        for j in (i + 1)..ctx_len {
            data[i * ctx_len + j] = f32::NEG_INFINITY;
        }
    }
    Ok(Tensor::from_vec(data, (1, 1, ctx_len, ctx_len), device)?)
}

#[derive(Debug, Clone)]
struct RelativePosParams {
    r_emb: Var,
    r_w_bias: Var,
    r_bias: Var,
}

fn init_normal_var(
    shape: impl Into<Shape>,
    std: f32,
    device: &Device,
    rng: Option<&mut DeterministicRng>,
) -> Result<Var, CodecError> {
    let shape = shape.into();
    match rng {
        Some(rng) => {
            let data = rng.sample_normal_vec(shape.elem_count(), 0.0, std)?;
            Ok(Var::from_vec(data, shape, device)?)
        }
        None => Ok(Var::zeros(shape, DType::F32, device)?),
    }
}

#[derive(Debug, Clone)]
struct Attention {
    n_head: usize,
    d_head: usize,
    d_model: usize,
    dropout: f32,
    position_type: PositionType,
    qkv: Var,
    proj: Var,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rel_pos: Option<RelativePosParams>,
}

impl Attention {
    fn new(
        cfg: &ModelConfig,
        residual_std: f32,
        device: &Device,
        mut rng: Option<&mut DeterministicRng>,
    ) -> Result<Self, CodecError> {
        let d_model = cfg.d_model;
        let d_head = d_model / cfg.n_head;
        let qkv = init_normal_var((3 * d_model, d_model), 0.02, device, rng.as_deref_mut())?;
        let proj = init_normal_var((d_model, d_model), residual_std, device, rng.as_deref_mut())?;
        let q_norm = RmsNorm::new(d_head, cfg.qk_norm, device)?;
        let k_norm = RmsNorm::new(d_head, cfg.qk_norm, device)?;

        let rel_pos = if cfg.position_type == PositionType::LearnedRelative {
            let r_emb = init_normal_var(
                (cfg.ctx_len, cfg.n_head, d_head),
                0.02,
                device,
                rng.as_deref_mut(),
            )?;
            let r_w_bias = init_normal_var((cfg.n_head, d_head), 0.02, device, rng)?;
            let r_bias = Var::zeros((cfg.ctx_len, cfg.n_head), DType::F32, device)?;
            Some(RelativePosParams {
                r_emb,
                r_w_bias,
                r_bias,
            })
        } else {
            None
        };

        Ok(Self {
            n_head: cfg.n_head,
            d_head,
            d_model,
            dropout: cfg.dropout,
            position_type: cfg.position_type,
            qkv,
            proj,
            q_norm,
            k_norm,
            rel_pos,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rope_cos: Option<&Tensor>,
        rope_sin: Option<&Tensor>,
        causal_mask: Option<&Tensor>,
        mode: &mut ForwardMode<'_>,
        compute_dtype: DType,
    ) -> Result<Tensor, CodecError> {
        let (batch, seq, d_model) = x.dims3()?;
        let qkv = linear_forward(x, &self.qkv, compute_dtype)?;
        let qkv_split = qkv
            .reshape((batch, seq, 3, self.n_head, self.d_head))?
            .permute((2, 0, 3, 1, 4))?
            .contiguous()?;
        let q = qkv_split.narrow(0, 0, 1)?.squeeze(0)?;
        let k = qkv_split.narrow(0, 1, 1)?.squeeze(0)?;
        let v = qkv_split.narrow(0, 2, 1)?.squeeze(0)?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let scores = if self.position_type == PositionType::Rope {
            let cos = rope_cos
                .ok_or_else(|| CodecError::InvalidConfig("RoPE cosine cache missing".to_owned()))?;
            let sin = rope_sin
                .ok_or_else(|| CodecError::InvalidConfig("RoPE sine cache missing".to_owned()))?;
            let q = apply_rope(&q, cos, sin)?;
            let k = apply_rope(&k, cos, sin)?;
            let q_f32 = q.to_dtype(DType::F32)?;
            let k_f32 = k.to_dtype(DType::F32)?.transpose(2, 3)?.contiguous()?;
            let scale = 1.0 / (self.d_head as f64).sqrt();
            q_f32.matmul(&k_f32)?.affine(scale, 0.0)?
        } else {
            let rel = self.rel_pos.as_ref().ok_or_else(|| {
                CodecError::InvalidConfig("learned_relative parameters missing".to_owned())
            })?;
            let q_f32 = q.to_dtype(DType::F32)?;
            let k_f32 = k.to_dtype(DType::F32)?.transpose(2, 3)?.contiguous()?;
            let r_w_bias = rel.r_w_bias.as_tensor().unsqueeze(0)?.unsqueeze(2)?;
            let q_biased = q_f32.broadcast_add(&r_w_bias)?;
            let content_score = q_biased.matmul(&k_f32)?;

            let r_emb_seq = rel
                .r_emb
                .as_tensor()
                .narrow(0, 0, seq)?
                .permute((1, 2, 0))?
                .contiguous()?
                .broadcast_as((batch, self.n_head, self.d_head, seq))?
                .contiguous()?;
            let raw_pos = q_f32.matmul(&r_emb_seq)?;

            let mut dist_vec = Vec::with_capacity(seq * seq);
            for i in 0..seq {
                for j in 0..seq {
                    dist_vec.push(i.saturating_sub(j) as u32);
                }
            }
            let dist_flat = Tensor::from_vec(dist_vec.clone(), seq * seq, x.device())?;
            let dist_idx = Tensor::from_vec(dist_vec, (1, 1, seq, seq), x.device())?
                .broadcast_as((batch, self.n_head, seq, seq))?
                .contiguous()?;
            let pos_gathered = raw_pos.gather(&dist_idx, D::Minus1)?;

            let r_bias_term = rel
                .r_bias
                .as_tensor()
                .index_select(&dist_flat, 0)?
                .reshape((seq, seq, self.n_head))?
                .permute((2, 0, 1))?
                .contiguous()?
                .affine((self.d_model as f64).sqrt(), 0.0)?
                .unsqueeze(0)?;
            let pos_score = pos_gathered.broadcast_add(&r_bias_term)?;
            let scale = 1.0 / (self.d_head as f64).sqrt();
            (content_score + pos_score)?.affine(scale, 0.0)?
        };

        let masked = if let Some(mask) = causal_mask {
            scores.broadcast_add(mask)?
        } else {
            scores
        };

        let prob = candle_nn::ops::softmax(&masked, D::Minus1)?;
        let prob = apply_dropout(&prob, self.dropout, mode)?;
        let prob = if prob.dtype() == v.dtype() {
            prob
        } else {
            prob.to_dtype(v.dtype())?
        };
        let out = prob
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq, d_model))?;
        linear_forward(&out, &self.proj, compute_dtype)
    }
}

#[derive(Debug, Clone)]
struct SwiGlu {
    gate: Var,
    up: Var,
    down: Var,
}

impl SwiGlu {
    fn new(
        d_model: usize,
        d_inner: usize,
        residual_std: f32,
        device: &Device,
        mut rng: Option<&mut DeterministicRng>,
    ) -> Result<Self, CodecError> {
        let gate = init_normal_var((d_inner, d_model), 0.02, device, rng.as_deref_mut())?;
        let up = init_normal_var((d_inner, d_model), 0.02, device, rng.as_deref_mut())?;
        let down = init_normal_var((d_model, d_inner), residual_std, device, rng)?;
        Ok(Self { gate, up, down })
    }

    fn forward(&self, x: &Tensor, compute_dtype: DType) -> Result<Tensor, CodecError> {
        let gate = linear_forward(x, &self.gate, compute_dtype)?.silu()?;
        let up = linear_forward(x, &self.up, compute_dtype)?;
        linear_forward(&(gate * up)?, &self.down, compute_dtype)
    }
}

#[derive(Debug, Clone)]
struct Block {
    norm_style: NormStyle,
    dropout: f32,
    attn_pre_norm: RmsNorm,
    attn: Attention,
    attn_post_norm: RmsNorm,
    ffn_pre_norm: RmsNorm,
    ffn: SwiGlu,
    ffn_post_norm: RmsNorm,
}

impl Block {
    fn new(
        cfg: &ModelConfig,
        residual_std: f32,
        device: &Device,
        mut rng: Option<&mut DeterministicRng>,
    ) -> Result<Self, CodecError> {
        let peri = cfg.norm_style == NormStyle::Peri;
        Ok(Self {
            norm_style: cfg.norm_style,
            dropout: cfg.dropout,
            attn_pre_norm: RmsNorm::new(cfg.d_model, peri, device)?,
            attn: Attention::new(cfg, residual_std, device, rng.as_deref_mut())?,
            attn_post_norm: RmsNorm::new(cfg.d_model, true, device)?,
            ffn_pre_norm: RmsNorm::new(cfg.d_model, peri, device)?,
            ffn: SwiGlu::new(cfg.d_model, cfg.d_inner, residual_std, device, rng)?,
            ffn_post_norm: RmsNorm::new(cfg.d_model, true, device)?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rope_cos: Option<&Tensor>,
        rope_sin: Option<&Tensor>,
        causal_mask: Option<&Tensor>,
        mode: &mut ForwardMode<'_>,
        compute_dtype: DType,
    ) -> Result<Tensor, CodecError> {
        match self.norm_style {
            NormStyle::Peri => {
                let normed_attn_in = self.attn_pre_norm.forward(x)?;
                let attn_out = self.attn.forward(
                    &normed_attn_in,
                    rope_cos,
                    rope_sin,
                    causal_mask,
                    mode,
                    compute_dtype,
                )?;
                let attn_out = self.attn_post_norm.forward(&attn_out)?;
                let attn_out = apply_dropout(&attn_out, self.dropout, mode)?;
                let x = (x + attn_out)?;

                let normed_ffn_in = self.ffn_pre_norm.forward(&x)?;
                let ffn_out = self.ffn.forward(&normed_ffn_in, compute_dtype)?;
                let ffn_out = self.ffn_post_norm.forward(&ffn_out)?;
                let ffn_out = apply_dropout(&ffn_out, self.dropout, mode)?;
                Ok((x + ffn_out)?)
            }
            NormStyle::Post => {
                let attn_out =
                    self.attn
                        .forward(x, rope_cos, rope_sin, causal_mask, mode, compute_dtype)?;
                let attn_out = apply_dropout(&attn_out, self.dropout, mode)?;
                let x = self.attn_post_norm.forward(&(x + attn_out)?)?;

                let ffn_out = self.ffn.forward(&x, compute_dtype)?;
                let ffn_out = apply_dropout(&ffn_out, self.dropout, mode)?;
                self.ffn_post_norm.forward(&(x + ffn_out)?)
            }
        }
    }
}

/// Neural Transformer language model matching `nn.py`.
#[derive(Debug)]
pub struct Transformer {
    config: ModelConfig,
    block_schedule: Vec<usize>,
    tok_emb: Var,
    emb_norm: RmsNorm,
    blocks: Vec<Block>,
    norm: RmsNorm,
    head: Var,
    rope_cos: Option<Tensor>,
    rope_sin: Option<Tensor>,
    causal_mask: Tensor,
    named_vars: Vec<(String, Var)>,
    state_vars: BTreeMap<String, Var>,
}

impl Transformer {
    /// Initializes a new [`Transformer`] with deterministic weights on `device`.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError`] on invalid model configuration or tensor allocation error.
    pub fn new(
        config: &ModelConfig,
        device: &Device,
        rng: &mut DeterministicRng,
    ) -> Result<Self, CodecError> {
        Self::build(config, device, Some(rng))
    }

    /// Constructs a [`Transformer`] and immediately loads a `QPI1` prior archive.
    ///
    /// For `loop-untying` priors (where every target block is copied from the archive),
    /// parameter buffers are allocated zeroed instead of sampling discarded normal
    /// initialization weights. For `depth-growth` priors (where unmapped layers retain
    /// random initialization), full deterministic initialization is performed via `rng`.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError`] if reading, validating, or loading the prior archive fails.
    pub fn from_prior(
        config: &ModelConfig,
        archive_path: impl AsRef<Path>,
        vocab_path: impl AsRef<Path>,
        seed: u64,
        device: &Device,
        rng: &mut DeterministicRng,
    ) -> Result<(Self, PriorLoadInfo), CodecError> {
        let archive = read_archive(archive_path)?;
        let is_loop_untying = archive
            .metadata
            .initialization
            .as_ref()
            .is_some_and(|init| init.kind == InitializationKind::LoopUntying);
        let mut model = if is_loop_untying {
            Self::build(config, device, None)?
        } else {
            Self::build(config, device, Some(rng))?
        };
        let info = model.apply_prior_archive(&archive, vocab_path, seed)?;
        Ok((model, info))
    }

    fn build(
        config: &ModelConfig,
        device: &Device,
        mut rng: Option<&mut DeterministicRng>,
    ) -> Result<Self, CodecError> {
        config.validate()?;
        let block_schedule = config
            .block_schedule
            .clone()
            .unwrap_or_else(|| (0..config.n_layer).collect());
        let executed_depth = block_schedule.len();
        let residual_std = 0.02f32 / ((2.0 * executed_depth as f64).sqrt() as f32);

        let tok_emb = init_normal_var(
            (config.vocab_size, config.d_model),
            0.02,
            device,
            rng.as_deref_mut(),
        )?;
        let peri = config.norm_style == NormStyle::Peri;
        let emb_norm = RmsNorm::new(config.d_model, peri, device)?;

        let mut blocks = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            blocks.push(Block::new(
                config,
                residual_std,
                device,
                rng.as_deref_mut(),
            )?);
        }

        let norm = RmsNorm::new(config.d_model, peri, device)?;
        let head = if config.tie_embeddings {
            tok_emb.clone()
        } else {
            init_normal_var((config.vocab_size, config.d_model), 0.02, device, rng)?
        };

        let (rope_cos, rope_sin) = if config.position_type == PositionType::Rope {
            let (c, s) = build_rope_cache(
                config.ctx_len,
                config.d_model / config.n_head,
                config.rope_base,
                device,
            )?;
            (Some(c), Some(s))
        } else {
            (None, None)
        };

        let mut named_vars = Vec::new();
        let mut state_vars = BTreeMap::new();

        let mut register = |name: String, var: Var, trainable: bool| {
            state_vars.insert(name.clone(), var.clone());
            if trainable {
                named_vars.push((name, var));
            }
        };

        register("tok_emb.weight".to_owned(), tok_emb.clone(), true);
        if let Some(w) = &emb_norm.weight {
            register("emb_norm.weight".to_owned(), w.clone(), true);
        }
        for (i, block) in blocks.iter().enumerate() {
            let p = format!("blocks.{i}");
            if let Some(w) = &block.attn_pre_norm.weight {
                register(format!("{p}.attn_pre_norm.weight"), w.clone(), true);
            }
            register(format!("{p}.attn.qkv.weight"), block.attn.qkv.clone(), true);
            if let Some(w) = &block.attn.q_norm.weight {
                register(format!("{p}.attn.q_norm.weight"), w.clone(), true);
            }
            if let Some(w) = &block.attn.k_norm.weight {
                register(format!("{p}.attn.k_norm.weight"), w.clone(), true);
            }
            if let Some(rel) = &block.attn.rel_pos {
                register(format!("{p}.attn.r_emb"), rel.r_emb.clone(), true);
                register(format!("{p}.attn.r_w_bias"), rel.r_w_bias.clone(), true);
                register(format!("{p}.attn.r_bias"), rel.r_bias.clone(), true);
            }
            register(
                format!("{p}.attn.proj.weight"),
                block.attn.proj.clone(),
                true,
            );
            if let Some(w) = &block.attn_post_norm.weight {
                register(format!("{p}.attn_post_norm.weight"), w.clone(), true);
            }
            if let Some(w) = &block.ffn_pre_norm.weight {
                register(format!("{p}.ffn_pre_norm.weight"), w.clone(), true);
            }
            register(format!("{p}.ffn.gate.weight"), block.ffn.gate.clone(), true);
            register(format!("{p}.ffn.up.weight"), block.ffn.up.clone(), true);
            register(format!("{p}.ffn.down.weight"), block.ffn.down.clone(), true);
            if let Some(w) = &block.ffn_post_norm.weight {
                register(format!("{p}.ffn_post_norm.weight"), w.clone(), true);
            }
        }
        if let Some(w) = &norm.weight {
            register("norm.weight".to_owned(), w.clone(), true);
        }
        register(
            "head.weight".to_owned(),
            head.clone(),
            !config.tie_embeddings,
        );

        let causal_mask = build_causal_mask(config.ctx_len, device)?;

        Ok(Self {
            config: config.clone(),
            block_schedule,
            tok_emb,
            emb_norm,
            blocks,
            norm,
            head,
            rope_cos,
            rope_sin,
            causal_mask,
            named_vars,
            state_vars,
        })
    }

    /// Runs a forward pass over token IDs `idx` (`[batch, seq]`, `U32`), returning logits.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError`] if tensor shapes, sequence lengths, or Candle operations fail.
    pub fn forward(
        &self,
        idx: &Tensor,
        output_slice: OutputSlice,
        mode: &mut ForwardMode<'_>,
        compute_dtype: DType,
    ) -> Result<Tensor, CodecError> {
        let (batch, seq) = idx.dims2()?;
        if seq == 0 || seq > self.config.ctx_len {
            return Err(CodecError::InvalidData(format!(
                "sequence length {seq} must be in 1..={}",
                self.config.ctx_len
            )));
        }

        let (cos_slice, sin_slice) = match (&self.rope_cos, &self.rope_sin) {
            (Some(c), Some(s)) => {
                let cn = c.narrow(2, 0, seq)?;
                let sn = s.narrow(2, 0, seq)?;
                let cn = if cn.dtype() == compute_dtype {
                    cn
                } else {
                    cn.to_dtype(compute_dtype)?
                };
                let sn = if sn.dtype() == compute_dtype {
                    sn
                } else {
                    sn.to_dtype(compute_dtype)?
                };
                (Some(cn), Some(sn))
            }
            _ => (None, None),
        };
        let causal_mask = if seq > 1 {
            Some(self.causal_mask.narrow(2, 0, seq)?.narrow(3, 0, seq)?)
        } else {
            None
        };

        let flat_idx = idx.flatten_all()?;
        let mut x = self
            .tok_emb
            .as_tensor()
            .index_select(&flat_idx, 0)?
            .reshape((batch, seq, self.config.d_model))?;
        if x.dtype() != compute_dtype {
            x = x.to_dtype(compute_dtype)?;
        }

        x = self.emb_norm.forward(&x)?;
        x = apply_dropout(&x, self.config.dropout, mode)?;

        for &block_idx in &self.block_schedule {
            x = self.blocks[block_idx].forward(
                &x,
                cos_slice.as_ref(),
                sin_slice.as_ref(),
                causal_mask.as_ref(),
                mode,
                compute_dtype,
            )?;
        }

        x = self.norm.forward(&x)?;

        let x = match output_slice {
            OutputSlice::All => x,
            OutputSlice::Last => x.narrow(1, seq - 1, 1)?,
            OutputSlice::Range { start, len } => x.narrow(1, start, len)?,
        };

        linear_forward(&x, &self.head, compute_dtype)
    }

    /// Returns all unique trainable `(name, Var)` pairs in canonical order.
    #[must_use]
    pub fn named_vars(&self) -> &[(String, Var)] {
        &self.named_vars
    }

    /// Returns `(total_params, non_embedding_params)`.
    #[must_use]
    pub fn param_counts(&self) -> (usize, usize) {
        let total: usize = self.named_vars.iter().map(|(_, v)| v.elem_count()).sum();
        let mut emb = self.tok_emb.elem_count();
        if !self.config.tie_embeddings {
            emb += self.head.elem_count();
        }
        (total, total.saturating_sub(emb))
    }

    /// Returns the expected parameter shape map matching PyTorch `state_dict()`.
    #[must_use]
    pub fn state_shapes(&self) -> BTreeMap<String, Vec<usize>> {
        self.state_vars
            .iter()
            .map(|(name, var)| (name.clone(), var.dims().to_vec()))
            .collect()
    }

    /// Computes a deterministic SHA-256 digest over all parameter names, shapes, dtypes, and `f32` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::Candle`] if reading parameter tensors from device fails.
    pub fn state_sha256(&self) -> Result<String, CodecError> {
        let mut hasher = Sha256::new();
        for (name, var) in &self.state_vars {
            let cpu_t = var
                .as_tensor()
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?;
            let dims = cpu_t.dims();
            let shape_str = match dims.len() {
                1 => format!("({},)", dims[0]),
                _ => {
                    let joined: Vec<String> = dims.iter().map(usize::to_string).collect();
                    format!("({})", joined.join(", "))
                }
            };
            hasher.update(name.as_bytes());
            hasher.update(b"\0");
            hasher.update(shape_str.as_bytes());
            hasher.update(b"\0");
            hasher.update(b"torch.float32\0");

            let flat = cpu_t.flatten_all()?.to_vec1::<f32>()?;
            let mut buf = [0u8; 4096];
            let mut off = 0usize;
            for val in flat {
                buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
                off += 4;
                if off == buf.len() {
                    hasher.update(buf);
                    off = 0;
                }
            }
            if off > 0 {
                hasher.update(&buf[..off]);
            }
        }
        Ok(hex::encode(hasher.finalize()))
    }

    /// Loads and validates a `QPI1` depth-growth or loop-untying prior archive.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError`] if reading, validating, or transferring tensors fails.
    pub fn load_prior(
        &mut self,
        archive_path: impl AsRef<Path>,
        vocab_path: impl AsRef<Path>,
        seed: u64,
    ) -> Result<PriorLoadInfo, CodecError> {
        let archive = read_archive(archive_path)?;
        self.apply_prior_archive(&archive, vocab_path, seed)
    }

    fn apply_prior_archive(
        &mut self,
        archive: &PriorArchive,
        vocab_path: impl AsRef<Path>,
        seed: u64,
    ) -> Result<PriorLoadInfo, CodecError> {
        let expected_geometry = self.config.geometry();
        let expected_shapes = self.state_shapes();
        let plan = plan_depth_prior(
            archive,
            vocab_path,
            &expected_geometry,
            seed,
            &expected_shapes,
        )?;
        let device = self.tok_emb.device().clone();

        for name in &plan.root_names {
            let src = &archive.tensors[name];
            let tensor = Tensor::from_slice(&src.data, Shape::from_dims(&src.shape), &device)?;
            self.state_vars[name].set(&tensor)?;
        }

        for (&target_idx, &source_idx) in &plan.target_sources {
            for suffix in &plan.block_suffixes {
                let source_name = format!("blocks.{source_idx}.{suffix}");
                let target_name = format!("blocks.{target_idx}.{suffix}");
                let src = &archive.tensors[&source_name];
                let tensor = Tensor::from_slice(&src.data, Shape::from_dims(&src.shape), &device)?;
                self.state_vars[&target_name].set(&tensor)?;
            }
        }

        for target_idx in &plan.zero_unmapped_targets {
            let proj_name = format!("blocks.{target_idx}.attn.proj.weight");
            let down_name = format!("blocks.{target_idx}.ffn.down.weight");
            for key in [&proj_name, &down_name] {
                let var = &self.state_vars[key];
                let zeros = Tensor::zeros(var.shape(), DType::F32, &device)?;
                var.set(&zeros)?;
            }
        }

        let initial_state_sha256 = self.state_sha256()?;
        Ok(PriorLoadInfo {
            archive_bytes: plan.archive_bytes,
            archive_sha256: plan.archive_sha256,
            vocab_sha256: plan.vocab_sha256,
            initialization_kind: plan.initialization_kind,
            source_depth: plan.source_depth,
            block_mapping: plan.block_mapping,
            block_schedule: plan.block_schedule,
            initial_state_sha256,
            policy: plan.policy,
        })
    }

    /// Exports all trainable parameters into `out` under `"model.<name>"` keys for safetensors checkpoints.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::Candle`] if converting tensors to contiguous CPU tensors fails.
    pub fn export_tensors(&self, out: &mut HashMap<String, Tensor>) -> Result<(), CodecError> {
        for (name, var) in &self.named_vars {
            let tensor = var.as_tensor().to_device(&Device::Cpu)?.contiguous()?;
            out.insert(format!("model.{name}"), tensor);
        }
        Ok(())
    }

    /// Restores trainable parameters from checkpoint tensors loaded by `load_checkpoint`.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError`] if any expected parameter is missing or has an incompatible shape.
    pub fn import_tensors(&mut self, tensors: &HashMap<String, Tensor>) -> Result<(), CodecError> {
        let device = self.tok_emb.device().clone();
        for (name, var) in &self.named_vars {
            let key = format!("model.{name}");
            let src = tensors.get(&key).ok_or_else(|| {
                CodecError::CheckpointMismatch(format!("checkpoint missing model parameter {key}"))
            })?;
            if src.dims() != var.dims() {
                return Err(CodecError::CheckpointMismatch(format!(
                    "checkpoint parameter {key} shape {:?} != expected {:?}",
                    src.dims(),
                    var.dims()
                )));
            }
            let aligned = src.to_device(&device)?.to_dtype(DType::F32)?.contiguous()?;
            var.set(&aligned)?;
        }
        Ok(())
    }

    /// Computes global L2 parameter norm across all trainable parameters.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::Candle`] on tensor read failures.
    pub fn weight_norm(&self) -> Result<f64, CodecError> {
        let mut sum_sq = 0.0f64;
        for (_, var) in &self.named_vars {
            let sq = var.as_tensor().to_dtype(DType::F32)?.sqr()?.sum_all()?;
            sum_sq += f64::from(sq.to_scalar::<f32>()?);
        }
        Ok(sum_sq.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_cpu_rms_norm_matches_unfused_fwd_and_bwd() {
        for (rows, dim) in [(4usize, 16usize), (96usize, 32usize)] {
            let mut x_data = Vec::with_capacity(rows * dim);
            for i in 0..(rows * dim) {
                x_data.push(((i % 23) as f32 - 11.0) * 0.13);
            }
            let w_data: Vec<f32> = (0..dim).map(|j| 0.7 + (j % 7) as f32 * 0.11).collect();
            let dy_data: Vec<f32> = (0..(rows * dim))
                .map(|i| ((i * 5 % 19) as f32 - 9.0) * 0.05)
                .collect();

            let dy = Tensor::from_slice(&dy_data, (rows, dim), &Device::Cpu).unwrap();

            let x_ref = Var::from_slice(&x_data, (rows, dim), &Device::Cpu).unwrap();
            let w_ref = Var::from_slice(&w_data, dim, &Device::Cpu).unwrap();
            let norm_ref = RmsNorm {
                weight: Some(w_ref.clone()),
                eps: RMS_NORM_EPS,
            };
            let out_ref = norm_ref.forward_unfused(x_ref.as_tensor(), &w_ref).unwrap();
            let loss_ref = (&out_ref * &dy).unwrap().sum_all().unwrap();
            let grads_ref = loss_ref.backward().unwrap();

            let x_opt = Var::from_slice(&x_data, (rows, dim), &Device::Cpu).unwrap();
            let w_opt = Var::from_slice(&w_data, dim, &Device::Cpu).unwrap();
            let norm_opt = RmsNorm {
                weight: Some(w_opt.clone()),
                eps: RMS_NORM_EPS,
            };
            let out_opt = norm_opt.forward(x_opt.as_tensor()).unwrap();
            let loss_opt = (&out_opt * &dy).unwrap().sum_all().unwrap();
            let grads_opt = loss_opt.backward().unwrap();

            let out_r = out_ref.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let out_o = out_opt.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (got, want) in out_o.iter().zip(out_r.iter()) {
                assert!((got - want).abs() < 1e-5, "fwd mismatch: {got} vs {want}");
            }

            let gx_r = grads_ref
                .get(x_ref.as_tensor())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let gx_o = grads_opt
                .get(x_opt.as_tensor())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for (got, want) in gx_o.iter().zip(gx_r.iter()) {
                assert!((got - want).abs() < 1e-4, "gx mismatch: {got} vs {want}");
            }

            let gw_r = grads_ref
                .get(w_ref.as_tensor())
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let gw_o = grads_opt
                .get(w_opt.as_tensor())
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for (got, want) in gw_o.iter().zip(gw_r.iter()) {
                assert!((got - want).abs() < 1e-4, "gw mismatch: {got} vs {want}");
            }
        }
    }
}
