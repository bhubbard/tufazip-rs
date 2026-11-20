use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use tufazip_arith::{
    make_freq_table_into, ArithmeticDecoder, ArithmeticEncoder, BitInputStream, BitOutputStream,
    DEFAULT_STATE_BITS,
};

use crate::checkpoint::{
    load_checkpoint, mark_done, save_checkpoint, stop_requested, CheckpointMetadata,
    CHECKPOINT_VERSION,
};
use crate::config::{lr_at, resolve_device, CodecConfig, CodecMode, OnlineLoss};
use crate::error::CodecError;
use crate::model::{ForwardMode, OutputSlice, PriorLoadInfo, Transformer};
use crate::optim::Adam;
use crate::rng::DeterministicRng;

const LOG2: f64 = std::f64::consts::LN_2;

/// Metrics reported upon completion of a neural coding or estimation run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunSummary {
    /// Total tokens processed (including batch padding).
    pub symbols: usize,
    /// Unpadded original token count.
    pub original_tokens: usize,
    /// Cumulative modeled cross-entropy bits.
    pub modeled_bits: f64,
    /// Theoretical payload bytes (`modeled_bits / 8.0`).
    pub modeled_payload_bytes: f64,
    /// Average modeled bits per token.
    pub modeled_bits_per_token: f64,
    /// Average modeled bits per original source byte (if `raw_bytes` was supplied).
    pub modeled_bits_per_byte: Option<f64>,
    /// Actual compressed artifact size in bytes (`None` in `EstimateOnly` mode).
    pub actual_compressed_bytes: Option<u64>,
    /// Actual compressed bits per original source byte (`None` unless coding and `raw_bytes` set).
    pub actual_bits_per_byte: Option<f64>,
    /// Configured vocabulary size in bytes.
    pub vocab_bytes: u64,
    /// Prior archive file size in bytes (`0` if uninitialized).
    pub prior_archive_bytes: u64,
    /// Prior archive SHA-256 hex digest if loaded.
    pub initial_prior_sha256: Option<String>,
    /// Prior source layer count (`0` if uninitialized).
    pub prior_source_depth: usize,
    /// Prior initialization kind string (`"depth-growth"` / `"loop-untying"`).
    pub prior_initialization_kind: Option<String>,
    /// Prior block mapping if loaded.
    pub prior_block_mapping: Option<Vec<usize>>,
    /// Total wall-clock seconds elapsed across all resumed segments.
    pub elapsed_seconds: f64,
}

/// Outcome returned by [`run_codec`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RunOutcome {
    /// Stream finished compressing or decompressing completely.
    Completed(RunSummary),
    /// Stopped cleanly at a block boundary after persisting a resumable checkpoint.
    RequeueRequested {
        /// Token index of the next block to execute upon resume.
        block_start: usize,
        /// Cumulative coded symbol count so far.
        n_symbols: usize,
    },
}

struct RunLogger {
    file: Option<BufWriter<File>>,
}

impl RunLogger {
    fn open(run_dir: Option<&Path>) -> Result<Self, CodecError> {
        let file = match run_dir {
            Some(dir) => {
                fs::create_dir_all(dir)?;
                let f = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(dir.join("log.txt"))?;
                Some(BufWriter::new(f))
            }
            None => None,
        };
        Ok(Self { file })
    }

    fn line(&mut self, msg: &str) {
        tracing::info!("{msg}");
        if let Some(f) = &mut self.file {
            let _ = writeln!(f, "{msg}");
            let _ = f.flush();
        }
    }
}

fn read_be16_tokens(path: &Path) -> Result<Vec<u16>, CodecError> {
    let bytes = fs::read(path)?;
    if bytes.len() % 2 != 0 {
        return Err(CodecError::InvalidData(format!(
            "token file {} has odd byte count ({})",
            path.display(),
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        out.push(u16::from_be_bytes([chunk[0], chunk[1]]));
    }
    Ok(out)
}

fn validate_resume_state(
    meta: &CheckpointMetadata,
    config: &CodecConfig,
    checkpoint_sig: &str,
    input_size: u64,
    prior_info: Option<&PriorLoadInfo>,
) -> Result<(), CodecError> {
    if meta.version != CHECKPOINT_VERSION {
        return Err(CodecError::CheckpointMismatch(format!(
            "unsupported checkpoint version {}",
            meta.version
        )));
    }
    if meta.config_signature != checkpoint_sig {
        return Err(CodecError::CheckpointMismatch(
            "checkpoint configuration differs from this invocation".to_owned(),
        ));
    }
    if meta.project_only_targets != config.project_only_targets {
        return Err(CodecError::CheckpointMismatch(
            "checkpoint project_only_targets differs from this invocation".to_owned(),
        ));
    }
    if meta.retrain_early_epochs != config.retrain.early_epochs
        || meta.retrain_early_until != config.retrain.early_until
    {
        return Err(CodecError::CheckpointMismatch(
            "checkpoint early replay policy differs from this invocation".to_owned(),
        ));
    }
    if meta.input_size != input_size {
        return Err(CodecError::CheckpointMismatch(format!(
            "checkpoint input size {} differs from current input {input_size}",
            meta.input_size
        )));
    }
    let expected_prior_sha = prior_info.map(|p| p.archive_sha256.as_str());
    if meta.initial_prior_sha256.as_deref() != expected_prior_sha {
        return Err(CodecError::CheckpointMismatch(
            "checkpoint initial_prior SHA-256 differs from this invocation".to_owned(),
        ));
    }
    let expected_vocab_sha = prior_info.map(|p| p.vocab_sha256.as_str());
    if meta.initial_prior_vocab_sha256.as_deref() != expected_vocab_sha {
        return Err(CodecError::CheckpointMismatch(
            "checkpoint prior vocab SHA-256 differs from this invocation".to_owned(),
        ));
    }
    Ok(())
}

fn cross_entropy_sum_cpu(rows: &[Vec<f32>], targets: &[u32]) -> f64 {
    let mut total_nats = 0.0f64;
    for (row, &tgt) in rows.iter().zip(targets.iter()) {
        let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f64;
        for &v in row {
            sum_exp += f64::from(v - max_logit).exp();
        }
        let target_val = f64::from(row[tgt as usize] - max_logit);
        total_nats += sum_exp.ln() - target_val;
    }
    total_nats
}

fn cross_entropy_sum_detached(logits: &Tensor, targets: &Tensor) -> Result<f64, CodecError> {
    let n = targets.elem_count();
    if n == 0 {
        return Ok(0.0);
    }
    let mean = candle_nn::loss::cross_entropy(&logits.detach().to_dtype(DType::F32)?, targets)?;
    Ok(f64::from(mean.to_scalar::<f32>()?) * (n as f64))
}

struct RetrainState<'a> {
    model: &'a Transformer,
    optimizer: &'a mut Adam,
    rng: &'a mut DeterministicRng,
    config: &'a CodecConfig,
    device: &'a Device,
    compute_dtype: DType,
    retrain_train_step: &'a mut usize,
    logger: &'a mut RunLogger,
}

fn run_prefix_retrain(
    state: &mut RetrainState<'_>,
    file_data: &[u16],
    file_pos: usize,
) -> Result<(), CodecError> {
    let retrain_batch = if state.config.retrain.batch_size == 0 {
        state.config.batch_size
    } else {
        state.config.retrain.batch_size
    };
    if retrain_batch == 0 {
        return Err(CodecError::InvalidConfig(
            "retrain_batch_size must be positive".to_owned(),
        ));
    }
    let lo = file_pos.saturating_sub(state.config.retrain.len);
    let stride = (file_pos - lo) / retrain_batch;
    if stride < 2 {
        return Ok(());
    }

    let n_targets = stride - 1;
    let ctx_len = state.config.model.ctx_len;
    let update_len = if state.config.retrain.update_len == 0 {
        ctx_len
    } else {
        state.config.retrain.update_len
    };
    let retrain_epochs =
        if state.config.retrain.early_until > 0 && file_pos <= state.config.retrain.early_until {
            state.config.retrain.early_epochs
        } else {
            state.config.retrain.epochs
        };

    let vocab_size = state.config.model.vocab_size;
    let mut window_bits = 0.0f64;
    let mut current_lr = state.config.retrain.lr_schedule.initial_lr();
    let stream_bases: Vec<usize> = (0..retrain_batch).map(|b| lo + b * stride).collect();

    for _epoch in 0..retrain_epochs {
        let mut epoch_nats = 0.0f64;
        let mut start = 0usize;
        while start < n_targets {
            let length = update_len.min(n_targets - start);
            let context_start = if state.config.retrain.update_len > 0 {
                start.saturating_sub(ctx_len - length)
            } else {
                start
            };
            let prefix_len = start - context_start;
            let seq_len = prefix_len + length;

            let mut inp_vec = Vec::with_capacity(retrain_batch * seq_len);
            let mut tgt_vec = Vec::with_capacity(retrain_batch * length);
            for &base in &stream_bases {
                for col in context_start..(context_start + seq_len) {
                    inp_vec.push(u32::from(file_data[base + col]));
                }
                for col in start..(start + length) {
                    tgt_vec.push(u32::from(file_data[base + col + 1]));
                }
            }

            let inp_t = Tensor::from_vec(inp_vec, (retrain_batch, seq_len), state.device)?;
            let tgt_t = Tensor::from_vec(tgt_vec, retrain_batch * length, state.device)?;

            let target_logits = {
                let mut fwd_mode = ForwardMode::Train(state.rng);
                if state.config.project_only_targets {
                    state.model.forward(
                        &inp_t,
                        OutputSlice::Range {
                            start: prefix_len,
                            len: length,
                        },
                        &mut fwd_mode,
                        state.compute_dtype,
                    )?
                } else {
                    state
                        .model
                        .forward(&inp_t, OutputSlice::All, &mut fwd_mode, state.compute_dtype)?
                        .narrow(1, prefix_len, length)?
                }
            };

            let flat_logits = target_logits
                .to_dtype(DType::F32)?
                .reshape((retrain_batch * length, vocab_size))?;
            let loss = candle_nn::loss::cross_entropy(&flat_logits, &tgt_t)?;
            let grads = loss.backward()?;

            current_lr = state
                .config
                .retrain
                .lr_schedule
                .lr_at(*state.retrain_train_step);
            state.optimizer.set_learning_rate(current_lr);
            state
                .optimizer
                .step_clipped(&grads, state.config.optim.clip)?;
            *state.retrain_train_step += 1;

            let loss_val = f64::from(loss.to_scalar::<f32>()?);
            epoch_nats += loss_val * (length as f64);
            start += update_len;
        }
        window_bits = epoch_nats / (n_targets as f64) / LOG2;
    }

    state.logger.line(&format!(
        "R {:8}  window={:8}  epochs={:2}  bits/tok={:6.3}  step={:7}  lr={:.3e}",
        file_pos,
        file_pos - lo,
        retrain_epochs,
        window_bits,
        *state.retrain_train_step,
        current_lr
    ));

    Ok(())
}

/// Runs online neural compression, decompression, or entropy estimation.
///
/// Supports deterministic checkpoint-resume via `config.checkpoint_dir` and graceful
/// stop-at-next-block-boundary via either `stop_flag` or `<checkpoint_dir>/STOP_REQUESTED`.
///
/// # Errors
///
/// Returns [`CodecError`] on invalid configuration, checkpoint mismatch, prior validation
/// failure, arithmetic coder errors, or tensor/I/O failures.
pub fn run_codec(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    config: &CodecConfig,
    mode: CodecMode,
    stop_flag: Arc<AtomicBool>,
) -> Result<RunOutcome, CodecError> {
    config.validate(mode)?;
    let input_path = input_path.as_ref();
    let output_path = output_path.as_ref();
    let device = resolve_device(config.device)?;
    let compute_dtype = config.effective_compute_dtype(&device);

    let mut logger = RunLogger::open(config.run_dir.as_deref())?;
    let mut rng = DeterministicRng::from_seed_u64(config.seed);

    let (mut model, prior_info) = match (&config.initial_prior, &config.vocab_file) {
        (Some(prior_path), Some(vocab_path)) => {
            let (m, info) = Transformer::from_prior(
                &config.model,
                prior_path,
                vocab_path,
                config.seed,
                &device,
                &mut rng,
            )?;
            (m, Some(info))
        }
        (Some(_), None) => {
            return Err(CodecError::InvalidConfig(
                "vocab_file is required when initial_prior is set".to_owned(),
            ));
        }
        (None, _) => (Transformer::new(&config.model, &device, &mut rng)?, None),
    };

    let mut optimizer = Adam::new(
        model.named_vars(),
        config.optim.lr,
        config.optim.beta1,
        config.optim.beta2,
        config.optim.adam_eps,
    )?;
    let mut retrain_optimizer = Adam::new(
        model.named_vars(),
        config.retrain.lr_schedule.initial_lr(),
        config.optim.beta1,
        config.optim.beta2,
        config.optim.adam_eps,
    )?;

    let (n_params, n_params_nonemb) = model.param_counts();
    logger.line(&format!(
        "tufazip-codec mode={mode:?} device={device:?} dtype={compute_dtype:?} params={n_params} (non-emb={n_params_nonemb})"
    ));

    let checkpoint_sig = config.checkpoint_signature();
    let input_size = fs::metadata(input_path)?.len();
    let loaded_checkpoint = match &config.checkpoint_dir {
        Some(dir) => load_checkpoint(dir, &device)?,
        None => None,
    };

    if let Some(ref ckpt) = loaded_checkpoint {
        let meta = &ckpt.metadata;
        let tensors = &ckpt.tensors;
        validate_resume_state(
            meta,
            config,
            &checkpoint_sig,
            input_size,
            prior_info.as_ref(),
        )?;
        model.import_tensors(tensors)?;
        optimizer.import_state("online_adam", &meta.online_adam_steps, tensors)?;
        retrain_optimizer.import_state("retrain_adam", &meta.retrain_adam_steps, tensors)?;
        rng.restore_state(&meta.rng_state);
        logger.line(&format!(
            "resuming checkpoint at block_start={} symbols={}",
            meta.block_start, meta.n_symbols
        ));
    }

    let batch = config.batch_size;
    let ctx_len = config.model.ctx_len;
    let vocab_size = config.model.vocab_size;

    let original_len: usize;
    let mut file_data: Vec<u16>;
    let mut encoder: Option<ArithmeticEncoder<BufWriter<File>>> = None;
    let mut decoder: Option<ArithmeticDecoder<BufReader<File>>> = None;
    let mut decode_out_file: Option<BufWriter<File>> = None;

    match mode {
        CodecMode::Compress | CodecMode::EstimateOnly => {
            file_data = read_be16_tokens(input_path)?;
            if let Some(max_tok) = config.max_tokens {
                file_data.truncate(max_tok);
            }
            original_len = file_data.len();

            if mode == CodecMode::Compress {
                if let Some(parent) = output_path.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                if let Some(ref ckpt) = loaded_checkpoint {
                    let meta = &ckpt.metadata;
                    let file = OpenOptions::new()
                        .read(true)
                        .write(true)
                        .open(output_path)?;
                    let mut bit_out = BitOutputStream::new(BufWriter::new(file));
                    let bit_state = meta.bit_out_state.ok_or_else(|| {
                        CodecError::CheckpointMismatch("missing bit_out_state".to_owned())
                    })?;
                    let enc_state = meta.encoder_state.ok_or_else(|| {
                        CodecError::CheckpointMismatch("missing encoder_state".to_owned())
                    })?;
                    bit_out.restore_file_state(bit_state)?;
                    let mut enc = ArithmeticEncoder::new(DEFAULT_STATE_BITS, bit_out)?;
                    enc.restore_state(enc_state)?;
                    encoder = Some(enc);
                } else {
                    let mut file = File::create(output_path)?;
                    let len_u32 = u32::try_from(original_len).map_err(|_| {
                        CodecError::InvalidData(format!(
                            "token count {original_len} exceeds u32 header limit"
                        ))
                    })?;
                    file.write_all(&len_u32.to_be_bytes())?;
                    file.flush()?;
                    let bit_out = BitOutputStream::new(BufWriter::new(file));
                    encoder = Some(ArithmeticEncoder::new(DEFAULT_STATE_BITS, bit_out)?);
                }
            }
        }
        CodecMode::Decompress => {
            if let Some(parent) = output_path.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent)?;
                }
            }
            let mut in_file = File::open(input_path)?;
            let mut len_buf = [0u8; 4];
            in_file.read_exact(&mut len_buf)?;
            original_len = u32::from_be_bytes(len_buf) as usize;
            file_data = vec![0u16; original_len];

            let bit_in = BitInputStream::new(BufReader::new(in_file));
            let mut dec = ArithmeticDecoder::new(DEFAULT_STATE_BITS, bit_in)?;

            if let Some(ref ckpt) = loaded_checkpoint {
                let meta = &ckpt.metadata;
                let bit_state = meta.bit_in_state.ok_or_else(|| {
                    CodecError::CheckpointMismatch("missing bit_in_state".to_owned())
                })?;
                let dec_state = meta.decoder_state.ok_or_else(|| {
                    CodecError::CheckpointMismatch("missing decoder_state".to_owned())
                })?;
                let output_pos = meta.output_position.ok_or_else(|| {
                    CodecError::CheckpointMismatch("missing output_position".to_owned())
                })?;
                dec.bit_stream_mut().restore_state(bit_state)?;
                dec.restore_state(dec_state)?;

                let mut out_file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(output_path)?;
                out_file.seek(SeekFrom::Start(0))?;
                let mut prefix_bytes = vec![0u8; output_pos as usize];
                out_file.read_exact(&mut prefix_bytes)?;
                if prefix_bytes.len() % 2 != 0 {
                    return Err(CodecError::CheckpointMismatch(
                        "decoded checkpoint prefix has odd byte count".to_owned(),
                    ));
                }
                for (dst, pair) in file_data.iter_mut().zip(prefix_bytes.chunks_exact(2)) {
                    *dst = u16::from_be_bytes([pair[0], pair[1]]);
                }
                out_file.set_len(output_pos)?;
                out_file.seek(SeekFrom::Start(output_pos))?;
                decode_out_file = Some(BufWriter::new(out_file));
            } else {
                let out_file = File::create(output_path)?;
                decode_out_file = Some(BufWriter::new(out_file));
            }
            decoder = Some(dec);
        }
    }

    if let Some(ref ckpt) = loaded_checkpoint {
        let meta = &ckpt.metadata;
        if meta.original_len != original_len {
            return Err(CodecError::CheckpointMismatch(format!(
                "checkpoint token count {} differs from input ({original_len})",
                meta.original_len
            )));
        }
    }

    let file_len = original_len.div_ceil(batch) * batch;
    if file_data.len() < file_len {
        file_data.resize(file_len, 0);
    }
    let block_len = batch.max((config.block_len / batch) * batch);
    let total_steps = file_len / batch;
    let min_lr = config.optim.lr * config.optim.min_lr_ratio;
    let bytes_per_token = config
        .raw_bytes
        .filter(|_| original_len > 0)
        .map(|raw| (raw as f64) / (original_len as f64));

    let (
        mut n_symbols,
        mut n_bits,
        mut train_step,
        mut retrain_train_step,
        mut block_start,
        mut last_retrain_pos,
        elapsed_before,
    ) = match loaded_checkpoint {
        Some(ref ckpt) => {
            let meta = &ckpt.metadata;
            (
                meta.n_symbols,
                meta.n_bits,
                meta.train_step,
                meta.retrain_train_step,
                meta.block_start,
                meta.last_retrain_pos,
                meta.elapsed_seconds,
            )
        }
        None => (0, 0.0, 0, 0, 0, 0, 0.0),
    };

    let mut completed_blocks = block_start / block_len;
    let session_start = Instant::now();
    let mut last_symbols = n_symbols;
    let mut last_bits = n_bits;
    let mut last_time = Instant::now();
    let mut freq_buf = vec![0u32; vocab_size];

    logger.line(&format!(
        "{:>8} {:>11} {:>11} {:>7} {:>7} {:>8} {:>8} {:>7} {:>9}",
        "step", "symbols", "comp_bytes", "bpt_loc", "bpt_cum", "bpb_loc", "bpb_cum", "kS/s", "lr"
    ));

    while block_start < file_len {
        if config.retrain.period > 0
            && block_start.saturating_sub(last_retrain_pos) >= config.retrain.period
        {
            let mut rt_state = RetrainState {
                model: &model,
                optimizer: &mut retrain_optimizer,
                rng: &mut rng,
                config,
                device: &device,
                compute_dtype,
                retrain_train_step: &mut retrain_train_step,
                logger: &mut logger,
            };
            run_prefix_retrain(&mut rt_state, &file_data, block_start)?;
            last_retrain_pos = block_start;
        }

        let cur_block_len = (file_len - block_start).min(block_len);
        let block_stride = cur_block_len / batch;
        let stream_base: Vec<usize> = (0..batch).map(|b| block_start + b * block_stride).collect();

        for stream_pos in 0..block_stride {
            let window = stream_pos.min(ctx_len);
            let seq_len = window.max(1);
            let mut inp_vec = Vec::with_capacity(batch * seq_len);
            if window == 0 {
                inp_vec.resize(batch, 0u32);
            } else {
                let start_off = stream_pos - window;
                for &base in &stream_base {
                    for off in start_off..(start_off + window) {
                        inp_vec.push(u32::from(file_data[base + off]));
                    }
                }
            }
            let inp_t = Tensor::from_vec(inp_vec, (batch, seq_len), &device)?;

            let lr = lr_at(
                train_step,
                config.optim.lr,
                min_lr,
                config.optim.warmup,
                total_steps,
            );
            optimizer.set_learning_rate(lr);

            let logits = {
                let mut eval_mode = ForwardMode::Eval;
                let slice = if config.project_only_targets {
                    OutputSlice::Last
                } else {
                    OutputSlice::All
                };
                model.forward(&inp_t, slice, &mut eval_mode, compute_dtype)?
            };

            let code_logits = logits
                .narrow(1, logits.dim(1)? - 1, 1)?
                .squeeze(1)?
                .to_dtype(DType::F32)?;

            let cached_cpu_logits = if encoder.is_some() || decoder.is_some() {
                let cpu_logits = code_logits
                    .detach()
                    .to_device(&Device::Cpu)?
                    .contiguous()?
                    .to_vec2::<f32>()?;
                for (i, row_logits) in cpu_logits.iter().enumerate() {
                    make_freq_table_into(row_logits, &mut freq_buf);
                    let pos = stream_base[i] + stream_pos;
                    if let Some(dec) = &mut decoder {
                        let sym = dec.read(&freq_buf)?;
                        file_data[pos] = sym as u16;
                    } else if let Some(enc) = &mut encoder {
                        enc.write(&freq_buf, usize::from(file_data[pos]))?;
                    }
                }
                Some(cpu_logits)
            } else {
                None
            };

            let target_vec: Vec<u32> = stream_base
                .iter()
                .map(|&base| u32::from(file_data[base + stream_pos]))
                .collect();
            let code_targets = Tensor::from_vec(target_vec.clone(), batch, &device)?;

            let (loss, code_nats) = match config.online_loss {
                OnlineLoss::Code => {
                    let loss = candle_nn::loss::cross_entropy(&code_logits, &code_targets)?;
                    let nats = if let Some(rows) = cached_cpu_logits.as_ref() {
                        cross_entropy_sum_cpu(rows, &target_vec)
                    } else {
                        f64::from(loss.to_scalar::<f32>()?) * (batch as f64)
                    };
                    (loss, nats)
                }
                OnlineLoss::Window => {
                    let nats = if let Some(rows) = cached_cpu_logits.as_ref() {
                        cross_entropy_sum_cpu(rows, &target_vec)
                    } else {
                        cross_entropy_sum_detached(&code_logits, &code_targets)?
                    };
                    let mut win_tgts = Vec::with_capacity(batch * seq_len);
                    if window == 0 {
                        for &base in &stream_base {
                            win_tgts.push(u32::from(file_data[base + stream_pos]));
                        }
                    } else {
                        let start_off = stream_pos - window + 1;
                        for &base in &stream_base {
                            for off in start_off..(start_off + window) {
                                win_tgts.push(u32::from(file_data[base + off]));
                            }
                        }
                    }
                    let tgt_t = Tensor::from_vec(win_tgts, batch * seq_len, &device)?;
                    let flat_logits = logits
                        .to_dtype(DType::F32)?
                        .reshape((batch * seq_len, vocab_size))?;
                    let loss = candle_nn::loss::cross_entropy(&flat_logits, &tgt_t)?;
                    (loss, nats)
                }
            };

            let grads = loss.backward()?;
            let _grad_norm = optimizer.step_clipped(&grads, config.optim.clip)?;

            n_bits += code_nats / LOG2;
            n_symbols += batch;
            train_step += 1;

            let is_last = block_start + cur_block_len >= file_len && stream_pos + 1 == block_stride;
            if (config.log_interval > 0 && n_symbols - last_symbols >= config.log_interval)
                || is_last
            {
                let now = Instant::now();
                let dt = now.duration_since(last_time).as_secs_f64().max(1e-6);
                let delta_syms = (n_symbols - last_symbols) as f64;
                let local_bpt = (n_bits - last_bits) / delta_syms;
                let cum_bpt = n_bits / (n_symbols as f64);
                let local_bpb = bytes_per_token.map_or(f64::NAN, |b| local_bpt / b);
                let cum_bpb = bytes_per_token.map_or(f64::NAN, |b| cum_bpt / b);
                let ks_per_s = delta_syms / (dt * 1000.0);
                logger.line(&format!(
                    "{train_step:8} {n_symbols:11} {:11.0} {local_bpt:7.3} {cum_bpt:7.3} {local_bpb:8.4} {cum_bpb:8.4} {ks_per_s:7.2} {lr:9.3e}",
                    n_bits / 8.0
                ));
                last_symbols = n_symbols;
                last_bits = n_bits;
                last_time = now;
            }
        }

        if mode == CodecMode::Decompress {
            if let Some(out_file) = &mut decode_out_file {
                let end = (block_start + cur_block_len).min(original_len);
                if block_start < end {
                    for &tok in &file_data[block_start..end] {
                        out_file.write_all(&tok.to_be_bytes())?;
                    }
                    out_file.flush()?;
                }
            }
        }

        block_start += cur_block_len;
        completed_blocks += 1;

        let requested_stop = stop_flag.load(Ordering::Relaxed)
            || config.checkpoint_dir.as_deref().is_some_and(stop_requested);
        let test_stop = config
            .stop_after_blocks
            .is_some_and(|limit| completed_blocks >= limit);
        let periodic = config.checkpoint_dir.is_some()
            && completed_blocks % config.checkpoint_every_blocks == 0;

        if block_start < file_len && (periodic || requested_stop || test_stop) {
            let reason = if requested_stop {
                "stop-request"
            } else if test_stop {
                "test-stop"
            } else {
                "periodic"
            };
            if let Some(dir) = &config.checkpoint_dir {
                let mut tensors = HashMap::new();
                model.export_tensors(&mut tensors)?;
                let online_adam_steps = optimizer.export_state("online_adam", &mut tensors)?;
                let retrain_adam_steps =
                    retrain_optimizer.export_state("retrain_adam", &mut tensors)?;

                let (encoder_state, bit_out_state) = match &mut encoder {
                    Some(enc) => (Some(enc.state()), Some(enc.bit_stream_mut().state()?)),
                    None => (None, None),
                };
                let (decoder_state, bit_in_state) = match &mut decoder {
                    Some(dec) => (Some(dec.state()), Some(dec.bit_stream_mut().state()?)),
                    None => (None, None),
                };
                let output_position = match &mut decode_out_file {
                    Some(out) => {
                        out.flush()?;
                        out.get_ref().sync_all()?;
                        Some(out.stream_position()?)
                    }
                    None => None,
                };

                let mut meta = CheckpointMetadata {
                    version: CHECKPOINT_VERSION,
                    config_signature: checkpoint_sig.clone(),
                    project_only_targets: config.project_only_targets,
                    retrain_early_epochs: config.retrain.early_epochs,
                    retrain_early_until: config.retrain.early_until,
                    input_size,
                    original_len,
                    file_len,
                    initial_prior_sha256: prior_info.as_ref().map(|p| p.archive_sha256.clone()),
                    initial_prior_vocab_sha256: prior_info.as_ref().map(|p| p.vocab_sha256.clone()),
                    tensors_sha256: String::new(),
                    online_adam_steps,
                    retrain_adam_steps,
                    rng_state: rng.state(),
                    encoder_state,
                    decoder_state,
                    bit_out_state,
                    bit_in_state,
                    output_position,
                    n_symbols,
                    n_bits,
                    train_step,
                    retrain_train_step,
                    block_start,
                    last_retrain_pos,
                    elapsed_seconds: elapsed_before + session_start.elapsed().as_secs_f64(),
                    reason: reason.to_owned(),
                };
                save_checkpoint(dir, &mut meta, &tensors)?;
                logger.line(&format!(
                    "checkpoint saved at block_start={block_start} ({reason})"
                ));
            }
            if requested_stop || test_stop {
                if let Some(out) = &mut decode_out_file {
                    out.flush()?;
                    out.get_ref().sync_all()?;
                }
                if let Some(enc) = &mut encoder {
                    enc.bit_stream_mut().flush()?;
                    enc.bit_stream_mut().get_ref().get_ref().sync_all()?;
                }
                return Ok(RunOutcome::RequeueRequested {
                    block_start,
                    n_symbols,
                });
            }
        }
    }

    if let Some(enc) = encoder.take() {
        let mut out_writer = enc.into_inner()?;
        out_writer.flush()?;
        out_writer.get_ref().sync_all()?;
    }
    if let Some(mut out_file) = decode_out_file.take() {
        out_file.flush()?;
        out_file.get_ref().sync_all()?;
    }

    let elapsed_seconds = elapsed_before + session_start.elapsed().as_secs_f64();
    let actual_compressed_bytes = match mode {
        CodecMode::Compress => Some(fs::metadata(output_path)?.len()),
        CodecMode::Decompress => Some(fs::metadata(input_path)?.len()),
        CodecMode::EstimateOnly => None,
    };
    let actual_bits_per_byte = match (actual_compressed_bytes, config.raw_bytes) {
        (Some(bytes), Some(raw)) if raw > 0 => Some((bytes as f64) * 8.0 / (raw as f64)),
        _ => None,
    };
    let modeled_bits_per_token = if n_symbols > 0 {
        n_bits / (n_symbols as f64)
    } else {
        0.0
    };
    let modeled_bits_per_byte = bytes_per_token.map(|b| modeled_bits_per_token / b);

    let summary = RunSummary {
        symbols: n_symbols,
        original_tokens: original_len,
        modeled_bits: n_bits,
        modeled_payload_bytes: n_bits / 8.0,
        modeled_bits_per_token,
        modeled_bits_per_byte,
        actual_compressed_bytes,
        actual_bits_per_byte,
        vocab_bytes: config.vocab_bytes,
        prior_archive_bytes: prior_info.as_ref().map_or(0, |p| p.archive_bytes),
        initial_prior_sha256: prior_info.as_ref().map(|p| p.archive_sha256.clone()),
        prior_source_depth: prior_info.as_ref().map_or(0, |p| p.source_depth),
        prior_initialization_kind: prior_info.as_ref().map(|p| {
            match p.initialization_kind {
                tufazip_prior::InitializationKind::DepthGrowth => "depth-growth",
                tufazip_prior::InitializationKind::LoopUntying => "loop-untying",
            }
            .to_owned()
        }),
        prior_block_mapping: prior_info.as_ref().map(|p| p.block_mapping.clone()),
        elapsed_seconds,
    };

    if let Ok(json) = serde_json::to_string(&summary) {
        logger.line(&format!("FINAL_RESULT {json}"));
    }

    if let Some(dir) = &config.checkpoint_dir {
        mark_done(dir)?;
    }

    Ok(RunOutcome::Completed(summary))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tufazip_preprocess::write_tokens_be16;

    use super::*;
    use crate::clip_grad_norm;
    use crate::config::{
        AutocastPolicy, ComputeDType, DeviceChoice, ModelConfig, NormStyle, PositionType,
    };

    fn tiny_tokens(len: usize, vocab_size: usize) -> Vec<u16> {
        (0..len)
            .map(|i| ((i * 7 + (i / 5) * 3) % vocab_size) as u16)
            .collect()
    }

    #[test]
    fn forward_backward_gradient_variants_and_bf16() {
        let variants = [
            (
                PositionType::Rope,
                NormStyle::Peri,
                false,
                true,
                OutputSlice::Last,
                DType::F32,
            ),
            (
                PositionType::LearnedRelative,
                NormStyle::Post,
                true,
                false,
                OutputSlice::All,
                DType::F32,
            ),
            (
                PositionType::Rope,
                NormStyle::Peri,
                false,
                true,
                OutputSlice::Last,
                DType::BF16,
            ),
        ];

        for (pos, norm, tie, qk_norm, slice, dt) in variants {
            let mut rng = DeterministicRng::from_seed_u64(42);
            let cfg = ModelConfig {
                vocab_size: 32,
                n_layer: 2,
                d_model: 16,
                n_head: 2,
                d_inner: 32,
                ctx_len: 8,
                dropout: 0.25,
                rope_base: 10_000.0,
                tie_embeddings: tie,
                position_type: pos,
                qk_norm,
                norm_style: norm,
                block_schedule: None,
            };
            let model = Transformer::new(&cfg, &Device::Cpu, &mut rng).unwrap();
            let mut adam = Adam::new(model.named_vars(), 1e-3, 0.0, 0.999, 1e-8).unwrap();

            let inp =
                Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6, 7, 8], (2, 4), &Device::Cpu).unwrap();
            let logits = {
                let mut mode = ForwardMode::Train(&mut rng);
                model.forward(&inp, slice, &mut mode, dt).unwrap()
            };
            let out_seq = logits.dim(1).unwrap();
            let tgt_len = 2 * out_seq;
            let tgts: Vec<u32> = (0..tgt_len).map(|i| (i % 32) as u32).collect();
            let tgt_t = Tensor::from_vec(tgts, tgt_len, &Device::Cpu).unwrap();
            let flat = logits
                .to_dtype(DType::F32)
                .unwrap()
                .reshape((tgt_len, 32))
                .unwrap();
            let loss = candle_nn::loss::cross_entropy(&flat, &tgt_t).unwrap();
            let mut grads = loss.backward().unwrap();
            let norm_val = clip_grad_norm(model.named_vars(), &mut grads, 0.25).unwrap();
            assert!(
                norm_val.is_finite(),
                "non-finite grad norm for {pos:?} {dt:?}"
            );
            adam.step(&grads).unwrap();
            assert!(model.weight_norm().unwrap().is_finite());
        }
    }

    #[test]
    fn lossless_compress_decompress_roundtrip_code_and_window() {
        for (online_loss, project_only) in [(OnlineLoss::Code, true), (OnlineLoss::Window, false)] {
            let dir = tempfile::tempdir().unwrap();
            let in_pp = dir.path().join("input.pp");
            let comp_bin = dir.path().join("comp.bin");
            let out_pp = dir.path().join("recovered.pp");

            let tokens = tiny_tokens(77, 64);
            write_tokens_be16(&in_pp, &tokens).unwrap();

            let mut cfg = CodecConfig::tiny(64);
            cfg.model.n_layer = 2;
            cfg.model.d_model = 32;
            cfg.model.n_head = 2;
            cfg.model.d_inner = 64;
            cfg.model.ctx_len = 16;
            cfg.model.dropout = 0.25;
            cfg.batch_size = 4;
            cfg.block_len = 32;
            cfg.retrain.period = 32;
            cfg.retrain.len = 64;
            cfg.retrain.batch_size = 4;
            cfg.retrain.update_len = 8;
            cfg.online_loss = online_loss;
            cfg.project_only_targets = project_only;
            cfg.autocast_policy = AutocastPolicy::Auto;
            cfg.dtype = ComputeDType::Fp32;
            cfg.device = DeviceChoice::Cpu;

            let stop = Arc::new(AtomicBool::new(false));
            let res_enc =
                run_codec(&in_pp, &comp_bin, &cfg, CodecMode::Compress, stop.clone()).unwrap();
            assert!(matches!(res_enc, RunOutcome::Completed(_)));

            let res_dec = run_codec(&comp_bin, &out_pp, &cfg, CodecMode::Decompress, stop).unwrap();
            assert!(matches!(res_dec, RunOutcome::Completed(_)));

            assert_eq!(
                std::fs::read(&out_pp).unwrap(),
                std::fs::read(&in_pp).unwrap(),
                "roundtrip mismatch for {online_loss:?}"
            );
        }
    }

    #[test]
    fn checkpoint_resume_bit_exact_compress_and_decompress() {
        let dir = tempfile::tempdir().unwrap();
        let in_pp = dir.path().join("input.pp");
        let full_bin = dir.path().join("full.bin");
        let resumed_bin = dir.path().join("resumed.bin");
        let resumed_pp = dir.path().join("resumed.pp");
        let enc_ckpt = dir.path().join("enc_ckpt");
        let dec_ckpt = dir.path().join("dec_ckpt");

        let tokens = tiny_tokens(113, 48);
        write_tokens_be16(&in_pp, &tokens).unwrap();

        let mut base_cfg = CodecConfig::tiny(48);
        base_cfg.model.n_layer = 2;
        base_cfg.model.d_model = 32;
        base_cfg.model.n_head = 2;
        base_cfg.model.d_inner = 64;
        base_cfg.model.ctx_len = 12;
        base_cfg.model.dropout = 0.25;
        base_cfg.batch_size = 4;
        base_cfg.block_len = 32;
        base_cfg.retrain.period = 32;
        base_cfg.retrain.len = 96;
        base_cfg.retrain.batch_size = 4;
        base_cfg.retrain.update_len = 8;

        let stop = Arc::new(AtomicBool::new(false));
        let full_out = run_codec(
            &in_pp,
            &full_bin,
            &base_cfg,
            CodecMode::Compress,
            stop.clone(),
        )
        .unwrap();
        assert!(matches!(full_out, RunOutcome::Completed(_)));

        // Stop after 1 block, then resume compression
        let mut stop_cfg = base_cfg.clone();
        stop_cfg.checkpoint_dir = Some(enc_ckpt.clone());
        stop_cfg.stop_after_blocks = Some(1);
        let paused = run_codec(
            &in_pp,
            &resumed_bin,
            &stop_cfg,
            CodecMode::Compress,
            stop.clone(),
        )
        .unwrap();
        assert!(matches!(paused, RunOutcome::RequeueRequested { .. }));

        stop_cfg.stop_after_blocks = None;
        let finished = run_codec(
            &in_pp,
            &resumed_bin,
            &stop_cfg,
            CodecMode::Compress,
            stop.clone(),
        )
        .unwrap();
        assert!(matches!(finished, RunOutcome::Completed(_)));
        assert!(crate::checkpoint::is_done(&enc_ckpt));
        assert_eq!(
            std::fs::read(&resumed_bin).unwrap(),
            std::fs::read(&full_bin).unwrap()
        );

        // Stop after 1 block, then resume decompression
        let mut dec_cfg = base_cfg;
        dec_cfg.checkpoint_dir = Some(dec_ckpt.clone());
        dec_cfg.stop_after_blocks = Some(1);
        let dec_paused = run_codec(
            &resumed_bin,
            &resumed_pp,
            &dec_cfg,
            CodecMode::Decompress,
            stop.clone(),
        )
        .unwrap();
        assert!(matches!(dec_paused, RunOutcome::RequeueRequested { .. }));

        dec_cfg.stop_after_blocks = None;
        let dec_done = run_codec(
            &resumed_bin,
            &resumed_pp,
            &dec_cfg,
            CodecMode::Decompress,
            stop,
        )
        .unwrap();
        assert!(matches!(dec_done, RunOutcome::Completed(_)));
        assert_eq!(
            std::fs::read(&resumed_pp).unwrap(),
            std::fs::read(&in_pp).unwrap()
        );
    }

    #[test]
    fn v2_model_loads_real_weights_bin_and_runs_forward() {
        let assets = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets");
        let weights_path = assets.join("weights.bin");
        let voc_path = assets.join("enwik9.voc");

        let cfg = ModelConfig::v2();
        let mut rng = DeterministicRng::from_seed_u64(1111);
        let (model, info) =
            Transformer::from_prior(&cfg, &weights_path, &voc_path, 1111, &Device::Cpu, &mut rng)
                .unwrap();
        let (total, non_emb) = model.param_counts();
        assert_eq!(total, 162_585_600);
        assert_eq!(non_emb, 154_194_944);
        assert_eq!(info.archive_bytes, 4_334_741);
        assert!(!info.initial_state_sha256.is_empty());

        let idx = Tensor::from_vec(vec![0u32, 32, 100, 200], (1, 4), &Device::Cpu).unwrap();
        let mut eval = ForwardMode::Eval;
        let logits = model
            .forward(&idx, OutputSlice::Last, &mut eval, DType::F32)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 1, 4097]);
        let vals = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()));
    }
}
