use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};

use candle_core::backprop::GradStore;
use candle_core::{CpuStorage, DType, Device, InplaceOp1, Layout, Storage, Tensor, Var};
use rayon::prelude::*;

use crate::error::CodecError;

const PAR_ELEM_THRESHOLD: usize = 8_000_000;

fn cpu_f32_tensor_slice(tensor: &Tensor) -> Result<(Option<Tensor>, usize, usize), CodecError> {
    if tensor.dtype() == DType::F32 && tensor.is_contiguous() {
        if let Some((start, end)) = tensor.layout().contiguous_offsets() {
            return Ok((None, start, end));
        }
    }
    let owned = tensor.to_dtype(DType::F32)?.contiguous()?;
    let (start, end) = owned
        .layout()
        .contiguous_offsets()
        .ok_or_else(|| CodecError::InvalidData("expected contiguous tensor".to_owned()))?;
    Ok((Some(owned), start, end))
}

fn read_cpu_f32_to_vec(tensor: &Tensor) -> Result<Vec<f32>, CodecError> {
    let (owned, start, end) = cpu_f32_tensor_slice(tensor)?;
    let target = owned.as_ref().unwrap_or(tensor);
    let (storage, _) = target.storage_and_layout();
    match &*storage {
        Storage::Cpu(CpuStorage::F32(slice)) => Ok(slice[start..end].to_vec()),
        _ => Err(CodecError::InvalidData(
            "expected F32 CPU tensor storage".to_owned(),
        )),
    }
}

fn cpu_grad_sq_norm(grad: &Tensor) -> Result<f64, CodecError> {
    let (owned, start, end) = cpu_f32_tensor_slice(grad)?;
    let target = owned.as_ref().unwrap_or(grad);
    let (storage, _) = target.storage_and_layout();
    match &*storage {
        Storage::Cpu(CpuStorage::F32(slice)) => {
            let data = &slice[start..end];
            let mut sq_acc = [0.0f32; 8];
            let chunks = data.chunks_exact(8);
            let rem = chunks.remainder();
            for c in chunks {
                for i in 0..8 {
                    let v = c[i];
                    sq_acc[i] += v * v;
                }
            }
            let mut acc = 0.0f64;
            for &lane in &sq_acc {
                acc += f64::from(lane);
            }
            for &v in rem {
                let fv = f64::from(v);
                acc += fv * fv;
            }
            Ok(acc)
        }
        _ => Err(CodecError::InvalidData(
            "expected F32 CPU gradient storage".to_owned(),
        )),
    }
}

/// Computes the global L2 gradient norm across `vars` using a single host sync on GPU devices
/// and zero intermediate allocations on CPU.
fn compute_global_grad_norm(vars: &[(String, Var)], grads: &GradStore) -> Result<f64, CodecError> {
    let Some((_, first)) = vars.first() else {
        return Ok(0.0);
    };
    if first.device().is_cpu() {
        let total_elems: usize = vars.iter().map(|(_, v)| v.elem_count()).sum();
        let sum_sq: f64 = if total_elems >= PAR_ELEM_THRESHOLD {
            let per_var: Vec<f64> = vars
                .par_iter()
                .map(|(_, var)| match grads.get(var) {
                    Some(grad) => cpu_grad_sq_norm(grad),
                    None => Ok(0.0),
                })
                .collect::<Result<Vec<_>, _>>()?;
            per_var.into_iter().sum()
        } else {
            let mut acc = 0.0f64;
            for (_, var) in vars {
                if let Some(grad) = grads.get(var) {
                    acc += cpu_grad_sq_norm(grad)?;
                }
            }
            acc
        };
        return Ok(sum_sq.sqrt());
    }

    let mut sq_terms = Vec::with_capacity(vars.len());
    for (_, var) in vars {
        if let Some(grad) = grads.get(var) {
            sq_terms.push(grad.to_dtype(DType::F32)?.sqr()?.sum_all()?);
        }
    }
    if sq_terms.is_empty() {
        return Ok(0.0);
    }
    let total_sq = if sq_terms.len() == 1 {
        sq_terms[0].to_scalar::<f32>()?
    } else {
        Tensor::stack(&sq_terms, 0)?.sum_all()?.to_scalar::<f32>()?
    };
    Ok(f64::from(total_sq).sqrt())
}

/// Computes the global L2 norm across `vars`' gradients and rescales `grads` in place when above `max_norm`.
///
/// Matches PyTorch's `torch.nn.utils.clip_grad_norm_` (`norm_type=2.0`, `eps=1e-6`).
///
/// # Errors
///
/// Returns [`CodecError::Candle`] if gradient reduction or scaling fails.
pub fn clip_grad_norm(
    vars: &[(String, Var)],
    grads: &mut GradStore,
    max_norm: f64,
) -> Result<f64, CodecError> {
    let total_norm = compute_global_grad_norm(vars, grads)?;
    if max_norm > 0.0 {
        let clip_coef = max_norm / (total_norm + 1e-6);
        if clip_coef < 1.0 {
            for (_, var) in vars {
                if let Some(grad) = grads.remove(var) {
                    let scaled = grad.to_dtype(DType::F32)?.affine(clip_coef, 0.0)?;
                    grads.insert(var, scaled);
                }
            }
        }
    }
    Ok(total_norm)
}

struct CpuAdamInplaceOp<'a> {
    grad: &'a [f32],
    m: RefCell<&'a mut [f32]>,
    v: RefCell<&'a mut [f32]>,
    clip_scale: f32,
    beta1: f32,
    one_minus_beta1: f32,
    beta2: f32,
    one_minus_beta2: f32,
    inv_bias_corr2: f32,
    step_size: f32,
    eps: f32,
}

impl InplaceOp1 for CpuAdamInplaceOp<'_> {
    fn name(&self) -> &'static str {
        "cpu-fused-adam"
    }

    fn cpu_fwd(&self, storage: &mut CpuStorage, layout: &Layout) -> candle_core::Result<()> {
        let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("Adam parameter variable must be contiguous".to_owned())
        })?;
        let slice = match storage {
            CpuStorage::F32(data) => &mut data[start..end],
            _ => {
                return Err(candle_core::Error::Msg(
                    "Adam parameter variable must have F32 storage".to_owned(),
                ))
            }
        };
        let mut m = self.m.borrow_mut();
        let mut v = self.v.borrow_mut();
        if slice.len() != self.grad.len() || slice.len() != m.len() || slice.len() != v.len() {
            return Err(candle_core::Error::Msg(format!(
                "Adam length mismatch: param={} grad={} m={} v={}",
                slice.len(),
                self.grad.len(),
                m.len(),
                v.len()
            )));
        }
        for (((p_val, m_val), v_val), &g_raw) in slice
            .iter_mut()
            .zip(m.iter_mut())
            .zip(v.iter_mut())
            .zip(self.grad.iter())
        {
            let g = g_raw * self.clip_scale;
            let next_m = *m_val * self.beta1 + g * self.one_minus_beta1;
            let next_v = *v_val * self.beta2 + (g * g) * self.one_minus_beta2;
            let denom = (next_v * self.inv_bias_corr2).sqrt() + self.eps;
            *p_val -= (next_m / denom) * self.step_size;
            *m_val = next_m;
            *v_val = next_v;
        }
        Ok(())
    }
}

#[derive(Debug)]
enum MomentBuffer {
    Cpu { m: Vec<f32>, v: Vec<f32> },
    Device { m: Tensor, v: Tensor },
}

#[derive(Debug)]
struct AdamParamState {
    name: String,
    var: Var,
    step: u64,
    moments: MomentBuffer,
}

impl AdamParamState {
    #[allow(clippy::too_many_arguments)]
    fn apply_update(
        &mut self,
        raw_grad: &Tensor,
        clip_scale: f64,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
    ) -> Result<(), CodecError> {
        self.step += 1;
        let t = self.step as i32;
        let bias_corr1 = 1.0 - beta1.powi(t);
        let bias_corr2 = 1.0 - beta2.powi(t);
        let step_size = lr / bias_corr1;

        match &mut self.moments {
            MomentBuffer::Cpu { m, v } => {
                let (owned_grad, start, end) = cpu_f32_tensor_slice(raw_grad)?;
                let grad_tensor = owned_grad.as_ref().unwrap_or(raw_grad);
                let (storage, _) = grad_tensor.storage_and_layout();
                let grad_slice = match &*storage {
                    Storage::Cpu(CpuStorage::F32(data)) => &data[start..end],
                    _ => {
                        return Err(CodecError::InvalidData(
                            "expected F32 CPU gradient storage".to_owned(),
                        ))
                    }
                };
                let op = CpuAdamInplaceOp {
                    grad: grad_slice,
                    m: RefCell::new(m.as_mut_slice()),
                    v: RefCell::new(v.as_mut_slice()),
                    clip_scale: clip_scale as f32,
                    beta1: beta1 as f32,
                    one_minus_beta1: (1.0 - beta1) as f32,
                    beta2: beta2 as f32,
                    one_minus_beta2: (1.0 - beta2) as f32,
                    inv_bias_corr2: (1.0 / bias_corr2) as f32,
                    step_size: step_size as f32,
                    eps: eps as f32,
                };
                self.var.as_tensor().inplace_op1(&op)?;
            }
            MomentBuffer::Device { m, v } => {
                let g = if raw_grad.dtype() == DType::F32 {
                    raw_grad.detach()
                } else {
                    raw_grad.to_dtype(DType::F32)?.detach()
                };
                let g_b1 = (1.0 - beta1) * clip_scale;
                let g_b2 = (1.0 - beta2) * clip_scale * clip_scale;
                let next_m = ((&*m * beta1)? + (&g * g_b1)?)?.detach();
                let next_v = ((&*v * beta2)? + (g.sqr()? * g_b2)?)?.detach();
                let denom = ((&next_v / bias_corr2)?.sqrt()? + eps)?;
                let update = (&next_m / denom)?.affine(step_size, 0.0)?;
                let next_param = (self.var.as_tensor() - update)?;
                self.var.set(&next_param)?;
                *m = next_m;
                *v = next_v;
            }
        }
        Ok(())
    }
}

/// Pure-Rust PyTorch-compatible Adam optimizer (without decoupled weight decay) with state export/import.
#[derive(Debug)]
pub struct Adam {
    params: Vec<AdamParamState>,
    total_elems: usize,
    is_cpu: bool,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
}

impl Adam {
    /// Creates an [`Adam`] optimizer tracking `vars`.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::Candle`] if moment buffer allocation fails.
    pub fn new(
        vars: &[(String, Var)],
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
    ) -> Result<Self, CodecError> {
        let mut params = Vec::with_capacity(vars.len());
        let mut total_elems = 0usize;
        let is_cpu = vars.first().map_or(true, |(_, v)| v.device().is_cpu());

        for (name, var) in vars {
            let n = var.elem_count();
            total_elems += n;
            let moments = if var.device().is_cpu() {
                MomentBuffer::Cpu {
                    m: vec![0.0f32; n],
                    v: vec![0.0f32; n],
                }
            } else {
                MomentBuffer::Device {
                    m: Tensor::zeros(var.shape(), DType::F32, var.device())?,
                    v: Tensor::zeros(var.shape(), DType::F32, var.device())?,
                }
            };
            params.push(AdamParamState {
                name: name.clone(),
                var: var.clone(),
                step: 0,
                moments,
            });
        }
        Ok(Self {
            params,
            total_elems,
            is_cpu,
            lr,
            beta1,
            beta2,
            eps,
        })
    }

    /// Returns the active learning rate.
    #[must_use]
    pub fn learning_rate(&self) -> f64 {
        self.lr
    }

    /// Updates the optimizer learning rate.
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// Performs one Adam update step over all tracked parameters present in `grads`.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::Candle`] if tensor updates fail.
    pub fn step(&mut self, grads: &GradStore) -> Result<(), CodecError> {
        self.apply_step(grads, 1.0)
    }

    /// Computes global gradient L2 norm, applies global norm clipping up to `max_norm`
    /// (when `max_norm > 0.0`), and performs one fused Adam update step.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError`] if gradient norm reduction or parameter updates fail.
    pub fn step_clipped(&mut self, grads: &GradStore, max_norm: f64) -> Result<f64, CodecError> {
        let total_norm = self.global_grad_norm(grads)?;
        let clip_scale = if max_norm > 0.0 {
            (max_norm / (total_norm + 1e-6)).min(1.0)
        } else {
            1.0
        };
        self.apply_step(grads, clip_scale)?;
        Ok(total_norm)
    }

    fn global_grad_norm(&self, grads: &GradStore) -> Result<f64, CodecError> {
        if self.params.is_empty() {
            return Ok(0.0);
        }
        if self.is_cpu {
            let sum_sq: f64 = if self.total_elems >= PAR_ELEM_THRESHOLD {
                let per_param: Vec<f64> = self
                    .params
                    .par_iter()
                    .map(|p| match grads.get(&p.var) {
                        Some(grad) => cpu_grad_sq_norm(grad),
                        None => Ok(0.0),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                per_param.into_iter().sum()
            } else {
                let mut acc = 0.0f64;
                for p in &self.params {
                    if let Some(grad) = grads.get(&p.var) {
                        acc += cpu_grad_sq_norm(grad)?;
                    }
                }
                acc
            };
            return Ok(sum_sq.sqrt());
        }

        let mut sq_terms = Vec::with_capacity(self.params.len());
        for p in &self.params {
            if let Some(grad) = grads.get(&p.var) {
                sq_terms.push(grad.to_dtype(DType::F32)?.sqr()?.sum_all()?);
            }
        }
        if sq_terms.is_empty() {
            return Ok(0.0);
        }
        let total_sq = if sq_terms.len() == 1 {
            sq_terms[0].to_scalar::<f32>()?
        } else {
            Tensor::stack(&sq_terms, 0)?.sum_all()?.to_scalar::<f32>()?
        };
        Ok(f64::from(total_sq).sqrt())
    }

    fn apply_step(&mut self, grads: &GradStore, clip_scale: f64) -> Result<(), CodecError> {
        let lr = self.lr;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;

        if self.is_cpu && self.total_elems >= PAR_ELEM_THRESHOLD {
            self.params
                .par_iter_mut()
                .try_for_each(|p| -> Result<(), CodecError> {
                    let Some(raw_grad) = grads.get(&p.var) else {
                        return Ok(());
                    };
                    p.apply_update(raw_grad, clip_scale, lr, beta1, beta2, eps)
                })?;
        } else {
            for p in &mut self.params {
                let Some(raw_grad) = grads.get(&p.var) else {
                    continue;
                };
                p.apply_update(raw_grad, clip_scale, lr, beta1, beta2, eps)?;
            }
        }
        Ok(())
    }

    /// Exports moment tensors into `out_tensors` and returns per-parameter step counts for checkpointing.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::Candle`] if transferring moments to CPU fails.
    pub fn export_state(
        &self,
        prefix: &str,
        out_tensors: &mut HashMap<String, Tensor>,
    ) -> Result<BTreeMap<String, u64>, CodecError> {
        let mut steps = BTreeMap::new();
        for p in &self.params {
            steps.insert(p.name.clone(), p.step);
            let (m_t, v_t) = match &p.moments {
                MomentBuffer::Cpu { m, v } => (
                    Tensor::from_slice(m, p.var.shape(), &Device::Cpu)?,
                    Tensor::from_slice(v, p.var.shape(), &Device::Cpu)?,
                ),
                MomentBuffer::Device { m, v } => (
                    m.to_device(&Device::Cpu)?.contiguous()?,
                    v.to_device(&Device::Cpu)?.contiguous()?,
                ),
            };
            out_tensors.insert(format!("{prefix}.{}.m", p.name), m_t);
            out_tensors.insert(format!("{prefix}.{}.v", p.name), v_t);
        }
        Ok(steps)
    }

    /// Restores step counts and moment buffers from a loaded checkpoint.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError`] if any parameter state is missing or mismatched in shape.
    pub fn import_state(
        &mut self,
        prefix: &str,
        steps: &BTreeMap<String, u64>,
        tensors: &HashMap<String, Tensor>,
    ) -> Result<(), CodecError> {
        for p in &mut self.params {
            let step = steps.get(&p.name).copied().ok_or_else(|| {
                CodecError::CheckpointMismatch(format!(
                    "checkpoint missing {prefix} step count for {}",
                    p.name
                ))
            })?;
            let m_key = format!("{prefix}.{}.m", p.name);
            let v_key = format!("{prefix}.{}.v", p.name);
            let m_src = tensors.get(&m_key).ok_or_else(|| {
                CodecError::CheckpointMismatch(format!("checkpoint missing tensor {m_key}"))
            })?;
            let v_src = tensors.get(&v_key).ok_or_else(|| {
                CodecError::CheckpointMismatch(format!("checkpoint missing tensor {v_key}"))
            })?;
            if m_src.dims() != p.var.dims() || v_src.dims() != p.var.dims() {
                return Err(CodecError::CheckpointMismatch(format!(
                    "checkpoint optimizer moment shape mismatch for {}",
                    p.name
                )));
            }
            p.step = step;
            match &mut p.moments {
                MomentBuffer::Cpu { m, v } => {
                    *m = read_cpu_f32_to_vec(m_src)?;
                    *v = read_cpu_f32_to_vec(v_src)?;
                }
                MomentBuffer::Device { m, v } => {
                    *m = m_src
                        .to_device(p.var.device())?
                        .to_dtype(DType::F32)?
                        .contiguous()?;
                    *v = v_src
                        .to_device(p.var.device())?
                        .to_dtype(DType::F32)?
                        .contiguous()?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_cpu_adam_step_clipped_matches_unfused_reference() {
        let init_a = vec![0.3f32, -0.7, 1.2, -0.05, 2.0, 0.11, -1.4, 0.8];
        let init_b = vec![-0.4f32, 0.9, -1.1, 0.6];
        let x = Tensor::from_slice(
            &[1.5f32, -0.5, 0.25, 0.75, -1.0, 0.2, 0.8, -0.3],
            (1, 8),
            &Device::Cpu,
        )
        .unwrap();

        let ref_a = Var::from_slice(&init_a, (1, 8), &Device::Cpu).unwrap();
        let ref_b = Var::from_slice(&init_b, (4, 1), &Device::Cpu).unwrap();
        let opt_a = Var::from_slice(&init_a, (1, 8), &Device::Cpu).unwrap();
        let opt_b = Var::from_slice(&init_b, (4, 1), &Device::Cpu).unwrap();

        let mut opt = Adam::new(
            &[
                ("a".to_owned(), opt_a.clone()),
                ("b".to_owned(), opt_b.clone()),
            ],
            1e-2,
            0.9,
            0.999,
            1e-8,
        )
        .unwrap();

        let mut m_a = Tensor::zeros(ref_a.shape(), DType::F32, &Device::Cpu).unwrap();
        let mut v_a = Tensor::zeros(ref_a.shape(), DType::F32, &Device::Cpu).unwrap();
        let mut m_b = Tensor::zeros(ref_b.shape(), DType::F32, &Device::Cpu).unwrap();
        let mut v_b = Tensor::zeros(ref_b.shape(), DType::F32, &Device::Cpu).unwrap();

        for step in 1usize..=4 {
            let loss_ref = ((ref_a.as_tensor() * &x).unwrap().sum_all().unwrap()
                + ref_b.as_tensor().sqr().unwrap().sum_all().unwrap())
            .unwrap();
            let mut grads_ref = loss_ref.backward().unwrap();
            let ref_vars = [
                ("a".to_owned(), ref_a.clone()),
                ("b".to_owned(), ref_b.clone()),
            ];
            let norm_ref = clip_grad_norm(&ref_vars, &mut grads_ref, 0.5).unwrap();

            let loss_opt = ((opt_a.as_tensor() * &x).unwrap().sum_all().unwrap()
                + opt_b.as_tensor().sqr().unwrap().sum_all().unwrap())
            .unwrap();
            let grads_opt = loss_opt.backward().unwrap();
            let norm_opt = opt.step_clipped(&grads_opt, 0.5).unwrap();
            assert!((norm_ref - norm_opt).abs() < 1e-6);

            let bc1 = 1.0 - 0.9f64.powi(step as i32);
            let bc2 = 1.0 - 0.999f64.powi(step as i32);
            for (var, m, v) in [(&ref_a, &mut m_a, &mut v_a), (&ref_b, &mut m_b, &mut v_b)] {
                let g = grads_ref.get(var).unwrap().detach();
                let nm = ((&*m * 0.9).unwrap() + (&g * 0.1).unwrap()).unwrap();
                let nv = ((&*v * 0.999).unwrap() + (g.sqr().unwrap() * 0.001).unwrap()).unwrap();
                let denom = ((&nv / bc2).unwrap().sqrt().unwrap() + 1e-8).unwrap();
                let upd = (&nm / denom).unwrap().affine(1e-2 / bc1, 0.0).unwrap();
                var.set(&(var.as_tensor() - upd).unwrap()).unwrap();
                *m = nm;
                *v = nv;
            }
        }

        let got_a = opt_a
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let want_a = ref_a
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let got_b = opt_b
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let want_b = ref_b
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (g, w) in got_a
            .iter()
            .zip(want_a.iter())
            .chain(got_b.iter().zip(want_b.iter()))
        {
            assert!((g - w).abs() < 1e-5, "mismatch: got {g} vs want {w}");
        }
    }
}
