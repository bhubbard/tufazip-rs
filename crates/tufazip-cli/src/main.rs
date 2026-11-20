//! Pure-Rust CLI for `tufazip` neural compression, decompression, preprocessing, and prior inspection.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use tracing_subscriber::EnvFilter;
use tufazip_codec::{
    clear_stop, is_done, run_codec, AutocastPolicy, CodecConfig, CodecMode, ComputeDType,
    DeviceChoice, NormStyle, OnlineLoss, PiecewiseLrSchedule, PositionType, RunOutcome,
    REQUEUE_EXIT_CODE,
};
use tufazip_preprocess::{
    decode_file, encode_bytes, encode_file, encode_with_dictionary, write_tokens_be16, Dictionary,
    EncodeConfig, SortOrder,
};
use tufazip_prior::{read_archive, sha256_file, InitializationKind};

const PAYLOAD_SHA256: &str = "198451a21f64386b487b8ea97e141562db355eec9cc9af34e7bbb6a022bcce77";
const PAYLOAD_BYTES: u64 = 100_958_324;
const VOCAB_SHA256: &str = "a63e517783e84a778f3f08c1f2321f3e2035e64d34c8e9206f03591460e44629";
const VOCAB_BYTES: u64 = 34_421;
const WEIGHTS_SHA256: &str = "e94c089f8eb985a87307dca3a7702385fcaccd14d95743811f8788f7da24448d";
const WEIGHTS_BYTES: u64 = 4_334_741;
const TOKENS_SHA256: &str = "102fa82b3f8932b37a1ec0d9262e555d23d17852685fe5fb33680dbee19dd694";
const TOKENS_BYTES: u64 = 570_405_660;
const RAW_SHA256: &str = "159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc";
const RAW_BYTES: u64 = 1_000_000_000;

#[derive(Debug, Parser)]
#[command(
    name = "tufazip",
    version = "0.2.0",
    about = "Pure-Rust port of tufazip v2 neural lossless compressor and decompressor"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// End-to-end compression of a raw file (preprocess + neural arithmetic coding).
    Compress {
        /// Raw input file path.
        input: PathBuf,
        /// Compressed output artifact path.
        output: PathBuf,
        /// Path to write the induced vocabulary (or read from with --use-existing-vocab).
        #[arg(long)]
        vocab: PathBuf,
        /// Reuse an existing vocabulary file (e.g. assets/enwik9.voc) instead of inducing a new one.
        #[arg(long)]
        use_existing_vocab: bool,
        /// Maximum dictionary size including the 256 base byte symbols.
        #[arg(long, default_value_t = 4097)]
        max_words: usize,
        /// Minimum bigram frequency for dictionary induction.
        #[arg(long, default_value_t = 5)]
        min_freq: u32,
        /// Sort learned vocabulary by decreasing frequency.
        #[arg(short = 's', long)]
        sort_by_freq: bool,
        /// Optional path to keep the intermediate .pp big-endian token file.
        #[arg(long)]
        keep_tokens: Option<PathBuf>,
        /// Neural model and training overrides.
        #[command(flatten)]
        codec: CodecArgs,
    },
    /// End-to-end decompression of a compressed tufazip artifact back to raw bytes.
    Decompress {
        /// Compressed input artifact path.
        input: PathBuf,
        /// Decoded raw output file path.
        output: PathBuf,
        /// Vocabulary (.voc) file used during compression.
        #[arg(long)]
        vocab: PathBuf,
        /// Optional path to keep the intermediate .pp big-endian token file.
        #[arg(long)]
        keep_tokens: Option<PathBuf>,
        /// Neural model and training overrides.
        #[command(flatten)]
        codec: CodecArgs,
    },
    /// Low-level neural codec operating directly on .pp token files (mirrors nn.py).
    Codec {
        /// Input .pp token stream (when compressing) or compressed .bin stream (when decompressing).
        #[arg(long)]
        input: PathBuf,
        /// Output file path.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Decompress instead of compressing.
        #[arg(long)]
        decompress: bool,
        /// Skip arithmetic coding and only train/report modeled bits.
        #[arg(long)]
        no_coding: bool,
        /// Neural model and training overrides.
        #[command(flatten)]
        codec: CodecArgs,
    },
    /// Standalone NNCP dictionary preprocessor (mirrors preprocess.c).
    Preprocess {
        #[command(subcommand)]
        action: PreprocessAction,
    },
    /// Deterministic verifier/decoder wrapper for the tufazip v2 enwik9 artifact.
    V2Decode {
        /// Path to enwik9-v2.bin payload.
        payload: PathBuf,
        /// Output path for decoded enwik9.
        output: PathBuf,
        /// Resumable state directory.
        #[arg(default_value = "tufazip-v2-state")]
        state: PathBuf,
        /// Path to enwik9.voc (defaults to ./assets/enwik9.voc).
        #[arg(long)]
        vocab: Option<PathBuf>,
        /// Path to weights.bin (defaults to ./assets/weights.bin).
        #[arg(long)]
        weights: Option<PathBuf>,
        /// Execution device backend.
        #[arg(long, value_enum, default_value_t = CliDevice::Auto)]
        device: CliDevice,
        /// Skip strict v2 SHA-256 verification checks (for testing on non-B200 hardware).
        #[arg(long)]
        skip_verify: bool,
    },
    /// Inspect and optionally verify a QPI1/QPR1/LRA1/QLA1 prior archive.
    InspectPrior {
        /// Path to packed prior file (e.g. assets/weights.bin).
        archive: PathBuf,
        /// Optional vocabulary file to verify against the prior's vocab_sha256.
        #[arg(long)]
        vocab: Option<PathBuf>,
        /// Print machine-readable JSON instead of human-readable summary.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
enum PreprocessAction {
    /// Induce dictionary from raw file and output big-endian u16 tokens (`preprocess c`).
    Encode {
        /// Output dictionary path (.voc).
        dict: PathBuf,
        /// Raw input file path.
        input: PathBuf,
        /// Output big-endian u16 token stream (.pp).
        output: PathBuf,
        /// Approximate maximum vocabulary size (including 256 base bytes).
        n_words: usize,
        /// Minimum bigram frequency.
        min_freq: u32,
        /// Sort dictionary entries by decreasing frequency (`-s`).
        #[arg(short = 's', long)]
        sort_by_freq: bool,
        /// Write debug frequency table (`-D`).
        #[arg(short = 'D', long)]
        debug_dict: Option<PathBuf>,
    },
    /// Decode big-endian u16 tokens back into original raw bytes (`preprocess d`).
    Decode {
        /// Dictionary path (.voc).
        dict: PathBuf,
        /// Input big-endian u16 token stream (.pp).
        input: PathBuf,
        /// Decoded raw output file path.
        output: PathBuf,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Preset {
    Tiny,
    V2,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliPositionType {
    Rope,
    LearnedRelative,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliNormStyle {
    Peri,
    Post,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliOnlineLoss {
    Code,
    Window,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliDType {
    Fp32,
    Bf16,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliDevice {
    Auto,
    Cpu,
    Metal,
    Cuda,
}

#[derive(Debug, Args)]
struct CodecArgs {
    /// Configuration preset (`tiny` for fast local runs, `v2` for full enwik9 settings).
    #[arg(long, value_enum, default_value_t = Preset::Tiny)]
    preset: Preset,
    #[arg(long)]
    vocab_size: Option<usize>,
    #[arg(long)]
    n_layer: Option<usize>,
    #[arg(long)]
    d_model: Option<usize>,
    #[arg(long)]
    n_head: Option<usize>,
    #[arg(long)]
    d_inner: Option<usize>,
    #[arg(long)]
    ctx_len: Option<usize>,
    #[arg(long)]
    dropout: Option<f32>,
    #[arg(long)]
    tie_embeddings: Option<bool>,
    #[arg(long, value_enum)]
    position_type: Option<CliPositionType>,
    #[arg(long)]
    no_qk_norm: bool,
    #[arg(long, value_enum)]
    norm_style: Option<CliNormStyle>,
    #[arg(long)]
    batch_size: Option<usize>,
    #[arg(long)]
    block_len: Option<usize>,
    #[arg(long, value_enum)]
    online_loss: Option<CliOnlineLoss>,
    #[arg(long)]
    project_only_targets: Option<bool>,
    #[arg(long)]
    lr: Option<f64>,
    #[arg(long)]
    min_lr_ratio: Option<f64>,
    #[arg(long)]
    warmup: Option<usize>,
    #[arg(long)]
    clip: Option<f64>,
    #[arg(long)]
    adam_eps: Option<f64>,
    #[arg(long)]
    beta1: Option<f64>,
    #[arg(long)]
    beta2: Option<f64>,
    #[arg(long)]
    retrain_period: Option<usize>,
    #[arg(long)]
    retrain_len: Option<usize>,
    #[arg(long)]
    retrain_epochs: Option<usize>,
    #[arg(long)]
    retrain_early_epochs: Option<usize>,
    #[arg(long)]
    retrain_early_until: Option<usize>,
    #[arg(long)]
    retrain_lr: Option<String>,
    #[arg(long)]
    retrain_update_len: Option<usize>,
    #[arg(long)]
    retrain_batch_size: Option<usize>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, value_enum)]
    dtype: Option<CliDType>,
    #[arg(long)]
    force_bf16: bool,
    #[arg(long, value_enum)]
    device: Option<CliDevice>,
    #[arg(long)]
    max_tokens: Option<usize>,
    #[arg(long)]
    raw_bytes: Option<u64>,
    #[arg(long)]
    vocab_bytes: Option<u64>,
    #[arg(long)]
    vocab_file: Option<PathBuf>,
    #[arg(long)]
    initial_prior: Option<PathBuf>,
    #[arg(long)]
    log_interval: Option<usize>,
    #[arg(long)]
    checkpoint_dir: Option<PathBuf>,
    #[arg(long)]
    checkpoint_every_blocks: Option<usize>,
    #[arg(long)]
    stop_after_blocks: Option<usize>,
    #[arg(long)]
    run_dir: Option<PathBuf>,
}

impl CodecArgs {
    fn build_config(&self, inferred_vocab_size: Option<usize>) -> Result<CodecConfig> {
        let mut cfg = match self.preset {
            Preset::V2 => CodecConfig::v2(),
            Preset::Tiny => CodecConfig::tiny(inferred_vocab_size.unwrap_or(4097)),
        };
        if let Some(v) = self.vocab_size {
            cfg.model.vocab_size = v;
        } else if let Some(v) = inferred_vocab_size {
            if matches!(self.preset, Preset::Tiny) {
                cfg.model.vocab_size = v;
            }
        }
        if let Some(v) = self.n_layer {
            cfg.model.n_layer = v;
        }
        if let Some(v) = self.d_model {
            cfg.model.d_model = v;
        }
        if let Some(v) = self.n_head {
            cfg.model.n_head = v;
        }
        if let Some(v) = self.d_inner {
            cfg.model.d_inner = v;
        }
        if let Some(v) = self.ctx_len {
            cfg.model.ctx_len = v;
        }
        if let Some(v) = self.dropout {
            cfg.model.dropout = v;
        }
        if let Some(v) = self.tie_embeddings {
            cfg.model.tie_embeddings = v;
        }
        if let Some(v) = self.position_type {
            cfg.model.position_type = match v {
                CliPositionType::Rope => PositionType::Rope,
                CliPositionType::LearnedRelative => PositionType::LearnedRelative,
            };
        }
        if self.no_qk_norm {
            cfg.model.qk_norm = false;
        }
        if let Some(v) = self.norm_style {
            cfg.model.norm_style = match v {
                CliNormStyle::Peri => NormStyle::Peri,
                CliNormStyle::Post => NormStyle::Post,
            };
        }
        if let Some(v) = self.batch_size {
            cfg.batch_size = v;
        }
        if let Some(v) = self.block_len {
            cfg.block_len = v;
        }
        if let Some(v) = self.online_loss {
            cfg.online_loss = match v {
                CliOnlineLoss::Code => OnlineLoss::Code,
                CliOnlineLoss::Window => OnlineLoss::Window,
            };
        }
        if let Some(v) = self.project_only_targets {
            cfg.project_only_targets = v;
        }
        if let Some(v) = self.lr {
            cfg.optim.lr = v;
        }
        if let Some(v) = self.min_lr_ratio {
            cfg.optim.min_lr_ratio = v;
        }
        if let Some(v) = self.warmup {
            cfg.optim.warmup = v;
        }
        if let Some(v) = self.clip {
            cfg.optim.clip = v;
        }
        if let Some(v) = self.adam_eps {
            cfg.optim.adam_eps = v;
        }
        if let Some(v) = self.beta1 {
            cfg.optim.beta1 = v;
        }
        if let Some(v) = self.beta2 {
            cfg.optim.beta2 = v;
        }
        if let Some(v) = self.retrain_period {
            cfg.retrain.period = v;
        }
        if let Some(v) = self.retrain_len {
            cfg.retrain.len = v;
        }
        if let Some(v) = self.retrain_epochs {
            cfg.retrain.epochs = v;
        }
        if let Some(v) = self.retrain_early_epochs {
            cfg.retrain.early_epochs = v;
        }
        if let Some(v) = self.retrain_early_until {
            cfg.retrain.early_until = v;
        }
        if let Some(ref spec) = self.retrain_lr {
            cfg.retrain.lr_schedule = PiecewiseLrSchedule::parse(spec)?;
        }
        if let Some(v) = self.retrain_update_len {
            cfg.retrain.update_len = v;
        }
        if let Some(v) = self.retrain_batch_size {
            cfg.retrain.batch_size = v;
        }
        if let Some(v) = self.seed {
            cfg.seed = v;
        }
        if let Some(v) = self.dtype {
            cfg.dtype = match v {
                CliDType::Fp32 => ComputeDType::Fp32,
                CliDType::Bf16 => ComputeDType::Bf16,
            };
        }
        if self.force_bf16 {
            cfg.autocast_policy = AutocastPolicy::Force;
        }
        if let Some(v) = self.device {
            cfg.device = map_device(v);
        }
        if let Some(v) = self.max_tokens {
            cfg.max_tokens = Some(v);
        }
        if let Some(v) = self.raw_bytes {
            cfg.raw_bytes = Some(v);
        }
        if let Some(v) = self.vocab_bytes {
            cfg.vocab_bytes = v;
        }
        if let Some(ref p) = self.vocab_file {
            cfg.vocab_file = Some(p.clone());
        }
        if let Some(ref p) = self.initial_prior {
            cfg.initial_prior = Some(p.clone());
        }
        if let Some(v) = self.log_interval {
            cfg.log_interval = v;
        }
        if let Some(ref p) = self.checkpoint_dir {
            cfg.checkpoint_dir = Some(p.clone());
        }
        if let Some(v) = self.checkpoint_every_blocks {
            cfg.checkpoint_every_blocks = v;
        }
        if let Some(v) = self.stop_after_blocks {
            cfg.stop_after_blocks = Some(v);
        }
        if let Some(ref p) = self.run_dir {
            cfg.run_dir = Some(p.clone());
        }
        Ok(cfg)
    }
}

fn map_device(d: CliDevice) -> DeviceChoice {
    match d {
        CliDevice::Auto => DeviceChoice::Auto,
        CliDevice::Cpu => DeviceChoice::Cpu,
        CliDevice::Metal => DeviceChoice::Metal,
        CliDevice::Cuda => DeviceChoice::Cuda,
    }
}

fn check_file(path: &Path, expected_size: u64, expected_sha256: &str) -> Result<()> {
    if !path.is_file() {
        bail!("missing file: {}", path.display());
    }
    let size = fs::metadata(path)?.len();
    if size != expected_size {
        bail!(
            "wrong size for {}: expected {expected_size} bytes, found {size}",
            path.display()
        );
    }
    let digest = sha256_file(path)?;
    if digest != expected_sha256 {
        bail!(
            "wrong SHA-256 for {}: expected {expected_sha256}, found {digest}",
            path.display()
        );
    }
    Ok(())
}

fn resolve_default_asset(explicit: Option<PathBuf>, filename: &str) -> PathBuf {
    if let Some(p) = explicit {
        return p;
    }
    let cwd_asset = PathBuf::from("assets").join(filename);
    if cwd_asset.is_file() {
        return cwd_asset;
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let sibling = dir.join(filename);
            if sibling.is_file() {
                return sibling;
            }
            let sibling_assets = dir.join("assets").join(filename);
            if sibling_assets.is_file() {
                return sibling_assets;
            }
        }
    }
    PathBuf::from(filename)
}

fn install_stop_handlers(stop_flag: &Arc<AtomicBool>) {
    let _ = signal_hook::flag::register(signal_hook::consts::SIGINT, Arc::clone(stop_flag));
    #[cfg(unix)]
    {
        let _ = signal_hook::flag::register(signal_hook::consts::SIGTERM, Arc::clone(stop_flag));
        let _ = signal_hook::flag::register(signal_hook::consts::SIGUSR1, Arc::clone(stop_flag));
    }
}

fn run(cli: Cli, stop_flag: Arc<AtomicBool>) -> Result<ExitCode> {
    match cli.command {
        Command::Compress {
            input,
            output,
            vocab,
            use_existing_vocab,
            max_words,
            min_freq,
            sort_by_freq,
            keep_tokens,
            codec,
        } => {
            let raw_bytes = fs::metadata(&input)
                .with_context(|| format!("reading {}", input.display()))?
                .len();
            let raw = fs::read(&input)?;
            let tokens = if use_existing_vocab {
                let dict = Dictionary::load(&vocab)?;
                encode_with_dictionary(&raw, &dict)?
            } else {
                let order = if sort_by_freq {
                    SortOrder::ByFrequency
                } else {
                    SortOrder::Discovery
                };
                let (dict, tokens, _report) = encode_bytes(
                    &raw,
                    &EncodeConfig {
                        max_words,
                        min_freq,
                        sort_order: order,
                        debug_dict_path: None,
                    },
                )?;
                if let Some(parent) = vocab.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                dict.save(&vocab)?;
                tokens
            };

            let dict = Dictionary::load(&vocab)?;
            let vocab_bytes = fs::metadata(&vocab)?.len();
            let mut cfg = codec.build_config(Some(dict.len()))?;
            if cfg.raw_bytes.is_none() {
                cfg.raw_bytes = Some(raw_bytes);
            }
            if cfg.vocab_bytes == 0 {
                cfg.vocab_bytes = vocab_bytes;
            }

            let tmp_dir;
            let pp_path = match keep_tokens {
                Some(p) => {
                    write_tokens_be16(&p, &tokens)?;
                    p
                }
                None => {
                    tmp_dir = tempfile::tempdir()?;
                    let p = tmp_dir.path().join("tokens.pp");
                    write_tokens_be16(&p, &tokens)?;
                    p
                }
            };

            match run_codec(&pp_path, &output, &cfg, CodecMode::Compress, stop_flag)? {
                RunOutcome::Completed(summary) => {
                    println!(
                        "Compressed {} ({} bytes) -> {} ({} bytes, vocab {} bytes, {:.4} bpb)",
                        input.display(),
                        raw_bytes,
                        output.display(),
                        summary.actual_compressed_bytes.unwrap_or(0),
                        vocab_bytes,
                        summary.actual_bits_per_byte.unwrap_or(0.0),
                    );
                    Ok(ExitCode::SUCCESS)
                }
                RunOutcome::RequeueRequested {
                    block_start,
                    n_symbols,
                } => {
                    eprintln!(
                        "Stopped at block_start={block_start} (symbols={n_symbols}); repeat command to resume."
                    );
                    Ok(ExitCode::from(REQUEUE_EXIT_CODE as u8))
                }
            }
        }
        Command::Decompress {
            input,
            output,
            vocab,
            keep_tokens,
            codec,
        } => {
            let dict = Dictionary::load(&vocab)?;
            let cfg = codec.build_config(Some(dict.len()))?;

            let tmp_dir;
            let pp_path = match keep_tokens {
                Some(p) => p,
                None => {
                    tmp_dir = tempfile::tempdir()?;
                    tmp_dir.path().join("tokens.pp")
                }
            };

            match run_codec(&input, &pp_path, &cfg, CodecMode::Decompress, stop_flag)? {
                RunOutcome::Completed(_) => {
                    if let Some(parent) = output.parent() {
                        if !parent.as_os_str().is_empty() {
                            fs::create_dir_all(parent)?;
                        }
                    }
                    let token_count = decode_file(&pp_path, &output, &vocab)?;
                    let out_bytes = fs::metadata(&output)?.len();
                    println!(
                        "Decompressed {} ({token_count} tokens) -> {} ({out_bytes} bytes)",
                        input.display(),
                        output.display()
                    );
                    Ok(ExitCode::SUCCESS)
                }
                RunOutcome::RequeueRequested {
                    block_start,
                    n_symbols,
                } => {
                    eprintln!(
                        "Stopped at block_start={block_start} (symbols={n_symbols}); repeat command to resume."
                    );
                    Ok(ExitCode::from(REQUEUE_EXIT_CODE as u8))
                }
            }
        }
        Command::Codec {
            input,
            output,
            decompress,
            no_coding,
            codec,
        } => {
            if no_coding && decompress {
                bail!("--no-coding cannot be combined with --decompress");
            }
            let mode = if decompress {
                CodecMode::Decompress
            } else if no_coding {
                CodecMode::EstimateOnly
            } else {
                CodecMode::Compress
            };
            let cfg = codec.build_config(None)?;
            let out_path = output.unwrap_or_else(|| {
                cfg.run_dir
                    .clone()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("out.bin")
            });
            match run_codec(&input, &out_path, &cfg, mode, stop_flag)? {
                RunOutcome::Completed(summary) => {
                    println!("{}", serde_json::to_string_pretty(&summary)?);
                    Ok(ExitCode::SUCCESS)
                }
                RunOutcome::RequeueRequested {
                    block_start,
                    n_symbols,
                } => {
                    eprintln!(
                        "Stopped at block_start={block_start} (symbols={n_symbols}); repeat command to resume."
                    );
                    Ok(ExitCode::from(REQUEUE_EXIT_CODE as u8))
                }
            }
        }
        Command::Preprocess { action } => match action {
            PreprocessAction::Encode {
                dict,
                input,
                output,
                n_words,
                min_freq,
                sort_by_freq,
                debug_dict,
            } => {
                let order = if sort_by_freq {
                    SortOrder::ByFrequency
                } else {
                    SortOrder::Discovery
                };
                let report = encode_file(
                    &input,
                    &output,
                    &dict,
                    &EncodeConfig {
                        max_words: n_words,
                        min_freq,
                        sort_order: order,
                        debug_dict_path: debug_dict,
                    },
                )?;
                println!(
                    "Number of words={} Final length={} (entropy={:.0} bytes)",
                    report.vocab_size,
                    report.token_count,
                    report.entropy_bits / 8.0
                );
                Ok(ExitCode::SUCCESS)
            }
            PreprocessAction::Decode {
                dict,
                input,
                output,
            } => {
                let tokens = decode_file(&input, &output, &dict)?;
                println!("Decoded {tokens} tokens -> {}", output.display());
                Ok(ExitCode::SUCCESS)
            }
        },
        Command::V2Decode {
            payload,
            output,
            state,
            vocab,
            weights,
            device,
            skip_verify,
        } => {
            let vocab_path = resolve_default_asset(vocab, "enwik9.voc");
            let weights_path = resolve_default_asset(weights, "weights.bin");
            if !skip_verify {
                check_file(&payload, PAYLOAD_BYTES, PAYLOAD_SHA256)?;
                check_file(&vocab_path, VOCAB_BYTES, VOCAB_SHA256)?;
                check_file(&weights_path, WEIGHTS_BYTES, WEIGHTS_SHA256)?;
            }

            let ckpt_dir = state.join("checkpoint");
            let run_dir = state.join("run");
            let pp_path = state.join("enwik9.pp");
            fs::create_dir_all(&ckpt_dir)?;
            fs::create_dir_all(&run_dir)?;
            if let Some(parent) = output.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent)?;
                }
            }
            clear_stop(&ckpt_dir)?;

            if !is_done(&ckpt_dir) {
                let mut cfg = CodecConfig::v2();
                cfg.device = map_device(device);
                cfg.vocab_file = Some(vocab_path.clone());
                cfg.initial_prior = Some(weights_path);
                cfg.checkpoint_dir = Some(ckpt_dir.clone());
                cfg.run_dir = Some(run_dir);
                match run_codec(&payload, &pp_path, &cfg, CodecMode::Decompress, stop_flag)? {
                    RunOutcome::Completed(_) => {}
                    RunOutcome::RequeueRequested { .. } => {
                        eprintln!(
                            "Stopping at checkpoint; repeat the identical command to resume."
                        );
                        return Ok(ExitCode::from(REQUEUE_EXIT_CODE as u8));
                    }
                }
            }

            if !skip_verify {
                check_file(&pp_path, TOKENS_BYTES, TOKENS_SHA256)?;
            }
            decode_file(&pp_path, &output, &vocab_path)?;
            if !skip_verify {
                check_file(&output, RAW_BYTES, RAW_SHA256)?;
            }
            println!("Decoded and verified: {}", output.display());
            Ok(ExitCode::SUCCESS)
        }
        Command::InspectPrior {
            archive,
            vocab,
            json,
        } => {
            let parsed = read_archive(&archive)?;
            let mut bit_counts: BTreeMap<u8, usize> = BTreeMap::new();
            for entry in &parsed.metadata.entries {
                *bit_counts.entry(entry.bits).or_default() += 1;
            }
            let vocab_match = match (&parsed.metadata.initialization, &vocab) {
                (Some(init), Some(voc)) => Some(sha256_file(voc)? == init.vocab_sha256),
                _ => None,
            };

            if json {
                let report = serde_json::json!({
                    "magic": parsed.magic.as_str(),
                    "archive_bytes": parsed.archive_bytes,
                    "archive_sha256": parsed.archive_sha256,
                    "version": parsed.metadata.version,
                    "policy": parsed.metadata.policy,
                    "initialization": parsed.metadata.initialization,
                    "tensor_count": parsed.tensors.len(),
                    "bit_width_counts": bit_counts,
                    "vocab_sha256_matches": vocab_match,
                });
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!("Magic:        {}", parsed.magic.as_str());
                println!("Size:         {} bytes", parsed.archive_bytes);
                println!("SHA-256:      {}", parsed.archive_sha256);
                println!("Policy:       {:?}", parsed.metadata.policy);
                println!(
                    "Tensors:      {} (bits: {bit_counts:?})",
                    parsed.tensors.len()
                );
                if let Some(init) = &parsed.metadata.initialization {
                    let kind = match init.kind {
                        InitializationKind::DepthGrowth => "depth-growth",
                        InitializationKind::LoopUntying => "loop-untying",
                    };
                    println!("Init kind:    {kind}");
                    println!("Seed:         {}", init.initialization_seed);
                    println!(
                        "Source layers: {} -> Target layers: {} (d_model={}, n_head={}, d_inner={}, ctx_len={}, vocab={})",
                        init.source_geometry.n_layer,
                        init.target_geometry.n_layer,
                        init.target_geometry.d_model,
                        init.target_geometry.n_head,
                        init.target_geometry.d_inner,
                        init.target_geometry.ctx_len,
                        init.target_geometry.vocab_size
                    );
                    println!("Vocab SHA256: {}", init.vocab_sha256);
                }
                if let Some(matched) = vocab_match {
                    println!(
                        "Vocab check:  {}",
                        if matched { "MATCH" } else { "MISMATCH" }
                    );
                }
            }
            Ok(ExitCode::SUCCESS)
        }
    }
}

fn main() -> ExitCode {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .without_time()
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let stop_flag = Arc::new(AtomicBool::new(false));
    install_stop_handlers(&stop_flag);

    match run(cli, stop_flag) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("tufazip: {err:#}");
            ExitCode::FAILURE
        }
    }
}
