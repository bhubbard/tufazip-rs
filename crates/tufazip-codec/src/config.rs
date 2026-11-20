use std::path::PathBuf;

use candle_core::{DType, Device};
use serde::{Deserialize, Serialize};
use tufazip_prior::{validate_block_schedule, TransformerGeometry};

use crate::error::CodecError;

/// Positional attention mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PositionType {
    /// Rotary Position Embeddings (RoPE).
    Rope,
    /// Bellard/Transformer-XL-style learned relative distance biases.
    LearnedRelative,
}

impl PositionType {
    /// Returns the canonical string identifier used in `QPI1` metadata.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Rope => "rope",
            Self::LearnedRelative => "learned_relative",
        }
    }
}

/// Residual block normalization placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormStyle {
    /// Peri-RMSNorm (normalization before and after attention/FFN branches, plus embeddings).
    Peri,
    /// Post-RMSNorm (normalization after residual addition).
    Post,
}

impl NormStyle {
    /// Returns the canonical string identifier used in `QPI1` metadata.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Peri => "peri",
            Self::Post => "post",
        }
    }
}

/// Online supervision target mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnlineLoss {
    /// Train only on the newly coded token per stream (`V2_CONFIG` default).
    Code,
    /// Train on every target position across the full context window.
    Window,
}

/// Model activation precision during forward/backward passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeDType {
    /// Full IEEE 754 single precision (`f32`).
    Fp32,
    /// Brain float16 mixed precision (`bf16`) with `f32` master weights and loss.
    Bf16,
}

/// Controls whether `ComputeDType::Bf16` activates on CPU or only GPU accelerators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutocastPolicy {
    /// Matches `nn.py`: enables `bf16` on GPU devices (`cuda`/`metal`) and stays `f32` on CPU.
    #[default]
    Auto,
    /// Forces `bf16` activations even on CPU (useful for testing `bf16` paths without GPU).
    Force,
}

/// Hardware execution backend preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceChoice {
    /// Automatically prefer CUDA, then Metal, then CPU.
    #[default]
    Auto,
    /// Force CPU execution.
    Cpu,
    /// Require Apple Metal GPU execution (`metal` feature).
    Metal,
    /// Require NVIDIA CUDA GPU execution (`cuda` feature).
    Cuda,
}

/// Operating mode for [`crate::run_codec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecMode {
    /// Encode input tokens into an arithmetic-coded payload.
    Compress,
    /// Decode an arithmetic-coded payload back into tokens.
    Decompress,
    /// Run online learning and report modeled bits without writing a coded stream.
    EstimateOnly,
}

/// Linearly interpolated piecewise learning-rate schedule (`lr0[,step1,lr1,...]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PiecewiseLrSchedule {
    points: Vec<(usize, f64)>,
}

impl PiecewiseLrSchedule {
    /// Creates a constant learning-rate schedule.
    #[must_use]
    pub fn constant(lr: f64) -> Self {
        Self {
            points: vec![(0, lr)],
        }
    }

    /// Parses a comma-separated piecewise LR schedule (`lr0[,step1,lr1,...]`).
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::InvalidConfig`] if the syntax is invalid or steps are not strictly increasing.
    pub fn parse(spec: &str) -> Result<Self, CodecError> {
        let parts: Vec<&str> = spec.split(',').map(str::trim).collect();
        if parts.is_empty() || parts.len() % 2 == 0 {
            return Err(CodecError::InvalidConfig(format!(
                "LR schedule must have syntax lr0[,step1,lr1,...], got {spec:?}"
            )));
        }
        let lr0: f64 = parts[0].parse().map_err(|_| {
            CodecError::InvalidConfig(format!("invalid initial LR {:?} in {spec:?}", parts[0]))
        })?;
        let mut points = vec![(0usize, lr0)];
        let mut prev_step = 0usize;
        for pair in parts[1..].chunks_exact(2) {
            let step: usize = pair[0].parse().map_err(|_| {
                CodecError::InvalidConfig(format!("invalid step {:?} in {spec:?}", pair[0]))
            })?;
            let lr: f64 = pair[1].parse().map_err(|_| {
                CodecError::InvalidConfig(format!("invalid LR {:?} in {spec:?}", pair[1]))
            })?;
            if step <= prev_step {
                return Err(CodecError::InvalidConfig(
                    "LR schedule steps must be strictly increasing".to_owned(),
                ));
            }
            points.push((step, lr));
            prev_step = step;
        }
        Ok(Self { points })
    }

    /// Evaluates the schedule at integer optimizer `step` with linear interpolation.
    #[must_use]
    pub fn lr_at(&self, step: usize) -> f64 {
        if step <= self.points[0].0 {
            return self.points[0].1;
        }
        for window in self.points.windows(2) {
            let (start_step, start_lr) = window[0];
            let (end_step, end_lr) = window[1];
            if step <= end_step {
                let fraction = (step - start_step) as f64 / (end_step - start_step) as f64;
                return start_lr + fraction * (end_lr - start_lr);
            }
        }
        self.points.last().map_or(0.0, |p| p.1)
    }

    /// Returns the initial learning rate at step 0.
    #[must_use]
    pub fn initial_lr(&self) -> f64 {
        self.points[0].1
    }

    /// Formats the schedule back into canonical `lr0[,step1,lr1,...]` syntax.
    #[must_use]
    pub fn to_spec(&self) -> String {
        let mut out = format!("{}", self.points[0].1);
        for &(step, lr) in &self.points[1..] {
            out.push_str(&format!(",{step},{lr}"));
        }
        out
    }
}

/// Linear warmup followed by cosine decay to `min_lr`.
#[must_use]
pub fn lr_at(step: usize, peak_lr: f64, min_lr: f64, warmup: usize, total_steps: usize) -> f64 {
    if warmup > 0 && step < warmup {
        return peak_lr * ((step + 1) as f64) / (warmup as f64);
    }
    if step >= total_steps {
        return min_lr;
    }
    let denom = total_steps.saturating_sub(warmup).max(1) as f64;
    let progress = (step.saturating_sub(warmup) as f64) / denom;
    min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + (std::f64::consts::PI * progress).cos())
}

/// Transformer architectural configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Vocabulary token count.
    pub vocab_size: usize,
    /// Number of physical transformer blocks.
    pub n_layer: usize,
    /// Residual stream width.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_head: usize,
    /// SwiGLU hidden width.
    pub d_inner: usize,
    /// Maximum context window length.
    pub ctx_len: usize,
    /// Dropout probability used only during prefix replay.
    pub dropout: f32,
    /// Rotary embedding frequency base (`10000.0`).
    pub rope_base: f64,
    /// Whether input and output embeddings share weights (`false` in v2).
    pub tie_embeddings: bool,
    /// Positional encoding strategy.
    pub position_type: PositionType,
    /// Per-head QK RMSNorm toggle (`true` in v2).
    pub qk_norm: bool,
    /// Residual normalization placement (`Peri` in v2).
    pub norm_style: NormStyle,
    /// Optional recurrent block execution order over `0..n_layer`.
    pub block_schedule: Option<Vec<usize>>,
}

impl ModelConfig {
    /// Returns the exact Transformer architecture of `tufazip` v2 (`V2_CONFIG`).
    #[must_use]
    pub fn v2() -> Self {
        Self {
            vocab_size: 4097,
            n_layer: 12,
            d_model: 1024,
            n_head: 8,
            d_inner: 2816,
            ctx_len: 192,
            dropout: 0.25,
            rope_base: 10_000.0,
            tie_embeddings: false,
            position_type: PositionType::Rope,
            qk_norm: true,
            norm_style: NormStyle::Peri,
            block_schedule: None,
        }
    }

    /// Returns a fast, tiny Transformer preset suited for smoke tests and small files.
    #[must_use]
    pub fn tiny(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            n_layer: 2,
            d_model: 64,
            n_head: 2,
            d_inner: 128,
            ctx_len: 32,
            dropout: 0.1,
            rope_base: 10_000.0,
            tie_embeddings: false,
            position_type: PositionType::Rope,
            qk_norm: true,
            norm_style: NormStyle::Peri,
            block_schedule: None,
        }
    }

    /// Validates architectural invariants.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::InvalidConfig`] if dimensions or schedules are invalid.
    pub fn validate(&self) -> Result<(), CodecError> {
        if self.vocab_size == 0 || self.n_layer == 0 || self.d_model == 0 || self.n_head == 0 {
            return Err(CodecError::InvalidConfig(
                "vocab_size, n_layer, d_model, and n_head must be positive".to_owned(),
            ));
        }
        if self.d_model % self.n_head != 0 {
            return Err(CodecError::InvalidConfig(format!(
                "d_model ({}) must be divisible by n_head ({})",
                self.d_model, self.n_head
            )));
        }
        if self.ctx_len == 0 || self.d_inner == 0 {
            return Err(CodecError::InvalidConfig(
                "ctx_len and d_inner must be positive".to_owned(),
            ));
        }
        if !(0.0..1.0).contains(&self.dropout) {
            return Err(CodecError::InvalidConfig(format!(
                "dropout ({}) must be in [0.0, 1.0)",
                self.dropout
            )));
        }
        if let Some(schedule) = &self.block_schedule {
            validate_block_schedule(schedule, self.n_layer)?;
        }
        Ok(())
    }

    /// Converts this model config to a `QPI1`-compatible [`TransformerGeometry`].
    #[must_use]
    pub fn geometry(&self) -> TransformerGeometry {
        TransformerGeometry {
            vocab_size: self.vocab_size,
            n_layer: self.n_layer,
            d_model: self.d_model,
            n_head: self.n_head,
            d_inner: self.d_inner,
            ctx_len: self.ctx_len,
            tie_embeddings: self.tie_embeddings,
            position_type: self.position_type.as_str().to_owned(),
            qk_norm: self.qk_norm,
            norm_style: self.norm_style.as_str().to_owned(),
        }
    }
}

/// Online Adam optimizer hyperparameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimConfig {
    /// Peak learning rate.
    pub lr: f64,
    /// Cosine decay minimum ratio relative to `lr`.
    pub min_lr_ratio: f64,
    /// Linear warmup steps.
    pub warmup: usize,
    /// Global gradient norm clipping threshold.
    pub clip: f64,
    /// Adam `beta1` (`0.0` in v2).
    pub beta1: f64,
    /// Adam `beta2` (`0.9999` in v2).
    pub beta2: f64,
    /// Adam numerical stability epsilon (`1e-9` in v2).
    pub adam_eps: f64,
}

impl OptimConfig {
    /// Returns the exact optimizer settings of `V2_CONFIG`.
    #[must_use]
    pub fn v2() -> Self {
        Self {
            lr: 4e-5,
            min_lr_ratio: 0.25,
            warmup: 200,
            clip: 0.25,
            beta1: 0.0,
            beta2: 0.9999,
            adam_eps: 1e-9,
        }
    }

    /// Returns faster default optimizer settings for small/tiny runs.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            lr: 5e-4,
            min_lr_ratio: 0.1,
            warmup: 16,
            clip: 0.25,
            beta1: 0.0,
            beta2: 0.999,
            adam_eps: 1e-8,
        }
    }
}

/// Periodic prefix retraining (replay) hyperparameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrainConfig {
    /// Retrain every `period` symbols (checked at block boundaries; `0` disables).
    pub period: usize,
    /// Maximum trailing prefix tokens to replay.
    pub len: usize,
    /// Number of replay epochs per retrain invocation.
    pub epochs: usize,
    /// Optional override epoch count when `file_pos <= early_until` (`0` disables).
    pub early_epochs: usize,
    /// Symbol prefix limit for `early_epochs` (`0` disables).
    pub early_until: usize,
    /// Piecewise learning-rate schedule across cumulative replay optimizer steps.
    pub lr_schedule: PiecewiseLrSchedule,
    /// New retraining targets per optimizer step (`0` uses `ctx_len`).
    pub update_len: usize,
    /// Independent stream count during replay (`0` uses online `batch_size`).
    pub batch_size: usize,
}

impl RetrainConfig {
    /// Returns the exact prefix-retraining settings of `V2_CONFIG`.
    ///
    /// # Panics
    ///
    /// Never panics because the hard-coded schedule string is statically valid.
    #[must_use]
    pub fn v2() -> Self {
        Self {
            period: 1,
            len: 35_000_000,
            epochs: 1,
            early_epochs: 0,
            early_until: 0,
            lr_schedule: PiecewiseLrSchedule::parse(
                "1.6e-4,13000,1.6e-4,93000,7.9e-5,163000,4.0e-5,1911300,1.3e-5",
            )
            .expect("v2 retrain LR schedule literal is valid"),
            update_len: 64,
            batch_size: 32,
        }
    }

    /// Returns fast default replay settings suited for tiny tests.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            period: 0,
            len: 2048,
            epochs: 1,
            early_epochs: 0,
            early_until: 0,
            lr_schedule: PiecewiseLrSchedule::constant(3e-4),
            update_len: 16,
            batch_size: 4,
        }
    }
}

/// Complete configuration for running the neural compressor or decompressor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodecConfig {
    /// Transformer architecture configuration.
    pub model: ModelConfig,
    /// Online Adam configuration.
    pub optim: OptimConfig,
    /// Prefix replay configuration.
    pub retrain: RetrainConfig,
    /// Number of parallel streams per block.
    pub batch_size: usize,
    /// Target token count per block (rounded down to multiple of `batch_size`).
    pub block_len: usize,
    /// Supervision loss mode (`Code` vs `Window`).
    pub online_loss: OnlineLoss,
    /// Project only supervised positions through `head`.
    pub project_only_targets: bool,
    /// Deterministic PRNG seed.
    pub seed: u64,
    /// Activation precision (`Fp32` or `Bf16`).
    pub dtype: ComputeDType,
    /// Controls when `Bf16` autocast is active.
    pub autocast_policy: AutocastPolicy,
    /// Execution device choice.
    pub device: DeviceChoice,
    /// Optional cap on input tokens during compression.
    pub max_tokens: Option<usize>,
    /// Original byte length before preprocessing (for bits/byte metrics).
    pub raw_bytes: Option<u64>,
    /// Vocabulary file byte size (for benchmark accounting).
    pub vocab_bytes: u64,
    /// Path to vocabulary file required when `initial_prior` is set.
    pub vocab_file: Option<PathBuf>,
    /// Path to `QPI1` quantized prior archive (`weights.bin`).
    pub initial_prior: Option<PathBuf>,
    /// Progress logging interval in symbols.
    pub log_interval: usize,
    /// Optional directory for durable resumable checkpoints.
    pub checkpoint_dir: Option<PathBuf>,
    /// Periodic checkpoint interval in completed blocks.
    pub checkpoint_every_blocks: usize,
    /// Optional block limit to trigger clean checkpoint & requeue exit (for testing).
    pub stop_after_blocks: Option<usize>,
    /// Optional directory where `log.txt` is appended.
    pub run_dir: Option<PathBuf>,
}

impl CodecConfig {
    /// Returns the exact configuration used to produce and decode `tufazip` v2.
    #[must_use]
    pub fn v2() -> Self {
        Self {
            model: ModelConfig::v2(),
            optim: OptimConfig::v2(),
            retrain: RetrainConfig::v2(),
            batch_size: 64,
            block_len: 500_000,
            online_loss: OnlineLoss::Code,
            project_only_targets: true,
            seed: 1111,
            dtype: ComputeDType::Bf16,
            autocast_policy: AutocastPolicy::Auto,
            device: DeviceChoice::Auto,
            max_tokens: None,
            raw_bytes: Some(1_000_000_000),
            vocab_bytes: 34_421,
            vocab_file: None,
            initial_prior: None,
            log_interval: 100_000,
            checkpoint_dir: None,
            checkpoint_every_blocks: 4,
            stop_after_blocks: None,
            run_dir: None,
        }
    }

    /// Returns a fast, tiny preset suitable for unit/integration tests and quick local runs.
    #[must_use]
    pub fn tiny(vocab_size: usize) -> Self {
        Self {
            model: ModelConfig::tiny(vocab_size),
            optim: OptimConfig::tiny(),
            retrain: RetrainConfig::tiny(),
            batch_size: 4,
            block_len: 64,
            online_loss: OnlineLoss::Code,
            project_only_targets: true,
            seed: 1111,
            dtype: ComputeDType::Fp32,
            autocast_policy: AutocastPolicy::Auto,
            device: DeviceChoice::Cpu,
            max_tokens: None,
            raw_bytes: None,
            vocab_bytes: 0,
            vocab_file: None,
            initial_prior: None,
            log_interval: 64,
            checkpoint_dir: None,
            checkpoint_every_blocks: 2,
            stop_after_blocks: None,
            run_dir: None,
        }
    }

    /// Validates all invariants before starting compression or decompression.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::InvalidConfig`] on invalid settings.
    pub fn validate(&self, mode: CodecMode) -> Result<(), CodecError> {
        self.model.validate()?;
        if self.batch_size == 0 {
            return Err(CodecError::InvalidConfig(
                "batch_size must be positive".to_owned(),
            ));
        }
        if self.checkpoint_every_blocks < 1 {
            return Err(CodecError::InvalidConfig(
                "checkpoint_every_blocks must be at least 1".to_owned(),
            ));
        }
        if self.project_only_targets && self.online_loss != OnlineLoss::Code {
            return Err(CodecError::InvalidConfig(
                "project_only_targets requires online_loss = Code".to_owned(),
            ));
        }
        if self.retrain.epochs < 1 {
            return Err(CodecError::InvalidConfig(
                "retrain.epochs must be at least 1".to_owned(),
            ));
        }
        if (self.retrain.early_epochs == 0) != (self.retrain.early_until == 0) {
            return Err(CodecError::InvalidConfig(
                "retrain.early_epochs and retrain.early_until must either both be set or both be 0"
                    .to_owned(),
            ));
        }
        if self.retrain.update_len > self.model.ctx_len {
            return Err(CodecError::InvalidConfig(format!(
                "retrain.update_len ({}) must be <= ctx_len ({})",
                self.retrain.update_len, self.model.ctx_len
            )));
        }
        if mode == CodecMode::Decompress && self.max_tokens.is_some() {
            return Err(CodecError::InvalidConfig(
                "max_tokens applies only when compressing".to_owned(),
            ));
        }
        if let Some(prior) = &self.initial_prior {
            if !prior.is_file() {
                return Err(CodecError::InvalidConfig(format!(
                    "initial_prior does not exist: {}",
                    prior.display()
                )));
            }
            if self.vocab_file.is_none() {
                return Err(CodecError::InvalidConfig(
                    "vocab_file is required when initial_prior is specified".to_owned(),
                ));
            }
        }
        Ok(())
    }

    /// Computes a deterministic JSON configuration signature used to reject incompatible resumes.
    #[must_use]
    pub fn checkpoint_signature(&self) -> String {
        #[derive(Serialize)]
        struct Sig<'a> {
            adam_eps: f64,
            batch_size: usize,
            beta1: f64,
            beta2: f64,
            block_len: usize,
            block_schedule: &'a Option<Vec<usize>>,
            clip: f64,
            ctx_len: usize,
            d_inner: usize,
            d_model: usize,
            dropout: f32,
            dtype: ComputeDType,
            lr: f64,
            min_lr_ratio: f64,
            n_head: usize,
            n_layer: usize,
            norm_style: NormStyle,
            online_loss: OnlineLoss,
            position_type: PositionType,
            qk_norm: bool,
            retrain_batch_size: usize,
            retrain_epochs: usize,
            retrain_len: usize,
            retrain_lr: String,
            retrain_period: usize,
            retrain_update_len: usize,
            rope_base: f64,
            seed: u64,
            tie_embeddings: bool,
            vocab_size: usize,
            warmup: usize,
        }

        let sig = Sig {
            adam_eps: self.optim.adam_eps,
            batch_size: self.batch_size,
            beta1: self.optim.beta1,
            beta2: self.optim.beta2,
            block_len: self.block_len,
            block_schedule: &self.model.block_schedule,
            clip: self.optim.clip,
            ctx_len: self.model.ctx_len,
            d_inner: self.model.d_inner,
            d_model: self.model.d_model,
            dropout: self.model.dropout,
            dtype: self.dtype,
            lr: self.optim.lr,
            min_lr_ratio: self.optim.min_lr_ratio,
            n_head: self.model.n_head,
            n_layer: self.model.n_layer,
            norm_style: self.model.norm_style,
            online_loss: self.online_loss,
            position_type: self.model.position_type,
            qk_norm: self.model.qk_norm,
            retrain_batch_size: self.retrain.batch_size,
            retrain_epochs: self.retrain.epochs,
            retrain_len: self.retrain.len,
            retrain_lr: self.retrain.lr_schedule.to_spec(),
            retrain_period: self.retrain.period,
            retrain_update_len: self.retrain.update_len,
            rope_base: self.model.rope_base,
            seed: self.seed,
            tie_embeddings: self.model.tie_embeddings,
            vocab_size: self.model.vocab_size,
            warmup: self.optim.warmup,
        };
        serde_json::to_string(&sig).unwrap_or_default()
    }

    /// Resolves the effective activation `DType` for `device`.
    #[must_use]
    pub fn effective_compute_dtype(&self, device: &Device) -> DType {
        match self.dtype {
            ComputeDType::Fp32 => DType::F32,
            ComputeDType::Bf16 => match self.autocast_policy {
                AutocastPolicy::Force => DType::BF16,
                AutocastPolicy::Auto => {
                    if matches!(device, Device::Cpu) {
                        DType::F32
                    } else {
                        DType::BF16
                    }
                }
            },
        }
    }
}

/// Resolves a Candle [`Device`] matching the requested [`DeviceChoice`].
///
/// # Errors
///
/// Returns [`CodecError::InvalidConfig`] if a specific accelerator was requested
/// without its Cargo feature compiled in or without available hardware.
pub fn resolve_device(choice: DeviceChoice) -> Result<Device, CodecError> {
    match choice {
        DeviceChoice::Cpu => Ok(Device::Cpu),
        DeviceChoice::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Ok(Device::new_cuda(0)?)
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(CodecError::InvalidConfig(
                    "CUDA device requested, but tufazip-codec was built without the `cuda` feature"
                        .to_owned(),
                ))
            }
        }
        DeviceChoice::Metal => {
            #[cfg(feature = "metal")]
            {
                Ok(Device::new_metal(0)?)
            }
            #[cfg(not(feature = "metal"))]
            {
                Err(CodecError::InvalidConfig(
                    "Metal device requested, but tufazip-codec was built without the `metal` feature"
                        .to_owned(),
                ))
            }
        }
        DeviceChoice::Auto => {
            #[cfg(feature = "cuda")]
            if let Ok(dev) = Device::new_cuda(0) {
                return Ok(dev);
            }
            #[cfg(feature = "metal")]
            if let Ok(dev) = Device::new_metal(0) {
                return Ok(dev);
            }
            Ok(Device::Cpu)
        }
    }
}
