use std::collections::{BTreeMap, HashMap};

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};

use crate::error::CodecError;

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
    let mut total_sq = 0.0f64;
    for (_, var) in vars {
        if let Some(grad) = grads.get(var) {
            let sq = grad.to_dtype(DType::F32)?.sqr()?.sum_all()?;
            total_sq += f64::from(sq.to_scalar::<f32>()?);
        }
    }
    let total_norm = total_sq.sqrt();
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

#[derive(Debug)]
struct AdamParamState {
    name: String,
    var: Var,
    step: u64,
    m: Tensor,
    v: Tensor,
}

/// Pure-Rust PyTorch-compatible Adam optimizer (without decoupled weight decay) with state export/import.
#[derive(Debug)]
pub struct Adam {
    params: Vec<AdamParamState>,
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
        for (name, var) in vars {
            let m = Tensor::zeros(var.shape(), DType::F32, var.device())?;
            let v = Tensor::zeros(var.shape(), DType::F32, var.device())?;
            params.push(AdamParamState {
                name: name.clone(),
                var: var.clone(),
                step: 0,
                m,
                v,
            });
        }
        Ok(Self {
            params,
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
        for p in &mut self.params {
            let Some(raw_grad) = grads.get(&p.var) else {
                continue;
            };
            let g = if raw_grad.dtype() == DType::F32 {
                raw_grad.detach()
            } else {
                raw_grad.to_dtype(DType::F32)?.detach()
            };
            p.step += 1;
            let t = p.step as i32;

            let next_m = ((&p.m * self.beta1)? + (&g * (1.0 - self.beta1))?)?.detach();
            let next_v = ((&p.v * self.beta2)? + (g.sqr()? * (1.0 - self.beta2))?)?.detach();

            let bias_corr1 = 1.0 - self.beta1.powi(t);
            let bias_corr2 = 1.0 - self.beta2.powi(t);
            let step_size = self.lr / bias_corr1;

            let denom = ((&next_v / bias_corr2)?.sqrt()? + self.eps)?;
            let update = (&next_m / denom)?.affine(step_size, 0.0)?;
            let next_param = (p.var.as_tensor() - update)?;
            p.var.set(&next_param)?;
            p.m = next_m;
            p.v = next_v;
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
            out_tensors.insert(
                format!("{prefix}.{}.m", p.name),
                p.m.to_device(&Device::Cpu)?.contiguous()?,
            );
            out_tensors.insert(
                format!("{prefix}.{}.v", p.name),
                p.v.to_device(&Device::Cpu)?.contiguous()?,
            );
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
            p.m = m_src
                .to_device(p.var.device())?
                .to_dtype(DType::F32)?
                .contiguous()?;
            p.v = v_src
                .to_device(p.var.device())?
                .to_dtype(DType::F32)?
                .contiguous()?;
        }
        Ok(())
    }
}
