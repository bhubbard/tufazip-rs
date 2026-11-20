use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use tufazip_arith::{BitInputState, BitOutputState, DecoderState, EncoderState};
use tufazip_prior::sha256_file;

use crate::error::CodecError;
use crate::rng::RngState;

/// Checkpoint schema version.
pub const CHECKPOINT_VERSION: u32 = 1;

/// Exit code returned when stopping cleanly at a block boundary for requeue/resume.
pub const REQUEUE_EXIT_CODE: i32 = 75;

/// Sentinel filename marking successful codec completion.
pub const DONE_NAME: &str = "DONE";

/// Sentinel filename requesting clean checkpoint-and-stop at the next block boundary.
pub const STOP_NAME: &str = "STOP_REQUESTED";

const LATEST_META_NAME: &str = "latest.json";
const LATEST_TENSORS_NAME: &str = "latest.safetensors";

/// Serializable metadata written alongside `latest.safetensors` for resumable execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Checkpoint format version (`1`).
    pub version: u32,
    /// Stable JSON signature of architectural and coding parameters.
    pub config_signature: String,
    /// Whether only target positions were projected through `head`.
    pub project_only_targets: bool,
    /// Early replay epoch override setting.
    pub retrain_early_epochs: usize,
    /// Early replay prefix threshold setting.
    pub retrain_early_until: usize,
    /// Input file size in bytes at start of run.
    pub input_size: u64,
    /// Unpadded original token count.
    pub original_len: usize,
    /// Padded token buffer length (multiple of `batch_size`).
    pub file_len: usize,
    /// SHA-256 digest of `--initial_prior` if used.
    pub initial_prior_sha256: Option<String>,
    /// SHA-256 digest of `--vocab_file` if used with `--initial_prior`.
    pub initial_prior_vocab_sha256: Option<String>,
    /// SHA-256 digest of the paired `latest.safetensors` file.
    pub tensors_sha256: String,
    /// Per-parameter Adam step counters for the online optimizer.
    pub online_adam_steps: BTreeMap<String, u64>,
    /// Per-parameter Adam step counters for the replay optimizer.
    pub retrain_adam_steps: BTreeMap<String, u64>,
    /// Exact PRNG state.
    pub rng_state: RngState,
    /// Encoder state (`Some` when compressing with coding enabled).
    pub encoder_state: Option<EncoderState>,
    /// Decoder state (`Some` when decompressing).
    pub decoder_state: Option<DecoderState>,
    /// Output bit stream state (`Some` when compressing with coding enabled).
    pub bit_out_state: Option<BitOutputState>,
    /// Input bit stream state (`Some` when decompressing).
    pub bit_in_state: Option<BitInputState>,
    /// Byte offset of the decoded token output file (`Some` when decompressing).
    pub output_position: Option<u64>,
    /// Cumulative coded symbol count.
    pub n_symbols: usize,
    /// Cumulative modeled cross-entropy bits.
    pub n_bits: f64,
    /// Cumulative online optimizer steps.
    pub train_step: usize,
    /// Cumulative prefix-replay optimizer steps.
    pub retrain_train_step: usize,
    /// Token index at the start of the next block.
    pub block_start: usize,
    /// Token index where prefix retraining last ran.
    pub last_retrain_pos: usize,
    /// Elapsed active compute seconds before this checkpoint.
    pub elapsed_seconds: f64,
    /// Reason this checkpoint was saved (`"periodic"`, `"stop-request"`, `"test-stop"`).
    pub reason: String,
}

fn sync_parent_dir(path: &Path) -> Result<(), CodecError> {
    if let Some(parent) = path.parent() {
        #[cfg(unix)]
        if let Ok(dir) = File::open(parent) {
            let _ = dir.sync_all();
        }
        #[cfg(not(unix))]
        let _ = parent;
    }
    Ok(())
}

/// Atomically writes `bytes` to `path` using a PID-scoped temporary file, fsync, and rename.
///
/// # Errors
///
/// Returns [`CodecError::Io`] on filesystem failure.
pub fn write_atomic_bytes(path: &Path, bytes: &[u8]) -> Result<(), CodecError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_path = tmp_path_for(path);
    let write_res = (|| -> Result<(), CodecError> {
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&tmp_path)?;
        file.write_all(bytes)?;
        file.flush()?;
        file.sync_all()?;
        drop(file);
        fs::rename(&tmp_path, path)?;
        sync_parent_dir(path)?;
        Ok(())
    })();
    if write_res.is_err() {
        let _ = fs::remove_file(&tmp_path);
    }
    write_res
}

fn tmp_path_for(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("checkpoint");
    path.with_file_name(format!("{name}.tmp.{}", std::process::id()))
}

/// Atomically saves a tensor map in `safetensors` format to `path`.
///
/// # Errors
///
/// Returns [`CodecError`] on serialization or filesystem failure.
pub fn write_atomic_safetensors(
    path: &Path,
    tensors: &HashMap<String, Tensor>,
) -> Result<(), CodecError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_path = tmp_path_for(path);
    let save_res = (|| -> Result<(), CodecError> {
        candle_core::safetensors::save(tensors, &tmp_path)?;
        if let Ok(file) = File::open(&tmp_path) {
            let _ = file.sync_all();
        }
        fs::rename(&tmp_path, path)?;
        sync_parent_dir(path)?;
        Ok(())
    })();
    if save_res.is_err() {
        let _ = fs::remove_file(&tmp_path);
    }
    save_res
}

/// Saves a resumable checkpoint (`latest.safetensors` + `latest.json`) atomically under `dir`.
///
/// # Errors
///
/// Returns [`CodecError`] on serialization or filesystem failure.
pub fn save_checkpoint(
    dir: &Path,
    meta: &mut CheckpointMetadata,
    tensors: &HashMap<String, Tensor>,
) -> Result<(), CodecError> {
    fs::create_dir_all(dir)?;
    let tensors_path = dir.join(LATEST_TENSORS_NAME);
    write_atomic_safetensors(&tensors_path, tensors)?;
    meta.tensors_sha256 = sha256_file(&tensors_path)?;
    let json = serde_json::to_vec_pretty(meta)?;
    write_atomic_bytes(&dir.join(LATEST_META_NAME), &json)?;
    Ok(())
}

/// Loaded checkpoint metadata and tensor map.
#[derive(Debug)]
pub struct LoadedCheckpoint {
    /// Parsed checkpoint metadata.
    pub metadata: CheckpointMetadata,
    /// Loaded model and optimizer tensors on the target device.
    pub tensors: HashMap<String, Tensor>,
}

/// Loads an existing checkpoint from `dir`, verifying `latest.safetensors` against `tensors_sha256`.
///
/// Returns `Ok(None)` if no checkpoint exists yet.
///
/// # Errors
///
/// Returns [`CodecError`] if files are malformed, incomplete, or fail integrity checks.
pub fn load_checkpoint(
    dir: &Path,
    device: &Device,
) -> Result<Option<LoadedCheckpoint>, CodecError> {
    let meta_path = dir.join(LATEST_META_NAME);
    if !meta_path.exists() {
        return Ok(None);
    }
    let tensors_path = dir.join(LATEST_TENSORS_NAME);
    if !tensors_path.exists() {
        return Err(CodecError::CheckpointMismatch(format!(
            "found {} without paired {}",
            meta_path.display(),
            tensors_path.display()
        )));
    }
    let meta_bytes = fs::read(&meta_path)?;
    let meta: CheckpointMetadata = serde_json::from_slice(&meta_bytes)?;
    let actual_sha = sha256_file(&tensors_path)?;
    if actual_sha != meta.tensors_sha256 {
        return Err(CodecError::CheckpointMismatch(format!(
            "checkpoint tensor digest mismatch: expected {}, found {actual_sha}",
            meta.tensors_sha256
        )));
    }
    let tensors = candle_core::safetensors::load(&tensors_path, device)?;
    Ok(Some(LoadedCheckpoint {
        metadata: meta,
        tensors,
    }))
}

/// Atomically writes the `DONE` completion marker in `dir`.
///
/// # Errors
///
/// Returns [`CodecError::Io`] on filesystem failure.
pub fn mark_done(dir: &Path) -> Result<(), CodecError> {
    write_atomic_bytes(&dir.join(DONE_NAME), b"complete\n")
}

/// Returns `true` if `dir/DONE` exists.
#[must_use]
pub fn is_done(dir: &Path) -> bool {
    dir.join(DONE_NAME).exists()
}

/// Writes `STOP_REQUESTED` in `dir`.
///
/// # Errors
///
/// Returns [`CodecError::Io`] on filesystem failure.
pub fn request_stop(dir: &Path) -> Result<(), CodecError> {
    write_atomic_bytes(&dir.join(STOP_NAME), b"stop\n")
}

/// Removes `STOP_REQUESTED` in `dir` if present.
///
/// # Errors
///
/// Returns [`CodecError::Io`] on filesystem errors other than `NotFound`.
pub fn clear_stop(dir: &Path) -> Result<(), CodecError> {
    match fs::remove_file(dir.join(STOP_NAME)) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// Returns `true` if `dir/STOP_REQUESTED` exists.
#[must_use]
pub fn stop_requested(dir: &Path) -> bool {
    dir.join(STOP_NAME).exists()
}
