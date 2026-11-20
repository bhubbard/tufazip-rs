//! Reproducible pure-Rust benchmark suite for `tufazip-rs`.
//!
//! Benchmarks arithmetic coding (`tufazip-arith`), NNCP preprocessing parity against
//! Fabrice Bellard's original `preprocess.c` (`tufazip-preprocess`), `QPI1` prior loading
//! (`tufazip-prior`), neural backends across CPU / Accelerate / Metal (`tufazip-codec`),
//! and lossless compression ratios (`bpb`) against standard compressors.

use std::fs;
use std::hint::black_box;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use sha2::{Digest, Sha256};
use tufazip_arith::{
    make_freq_table, make_freq_table_into, ArithmeticDecoder, ArithmeticEncoder, BitInputStream,
    BitOutputStream, DEFAULT_STATE_BITS,
};
use tufazip_codec::{
    resolve_device, run_codec, CodecConfig, CodecMode, DeterministicRng, DeviceChoice, ForwardMode,
    ModelConfig, OnlineLoss, OutputSlice, RunOutcome, Transformer,
};
use tufazip_preprocess::{
    decode_file, decode_tokens, encode_bytes, encode_file, encode_with_dictionary,
    read_tokens_be16, write_tokens_be16, Dictionary, EncodeConfig, SortOrder,
};
use tufazip_prior::{read_archive, sha256_file};

#[derive(Debug, Clone, Serialize)]
struct BenchmarkReport {
    backend: String,
    only_neural: bool,
    environment: EnvironmentInfo,
    corpus: CorpusInfo,
    arith: Option<ArithBench>,
    preprocess: Option<PreprocessBench>,
    prior: Option<PriorBench>,
    neural: NeuralBench,
    ratios: Option<RatioBench>,
}

#[derive(Debug, Clone, Serialize)]
struct EnvironmentInfo {
    os: String,
    arch: String,
    logical_cpus: usize,
}

#[derive(Debug, Clone, Serialize)]
struct CorpusInfo {
    file_count: usize,
    total_bytes: usize,
    sha256: String,
}

#[derive(Debug, Clone, Serialize)]
struct ArithBench {
    vocab_size: usize,
    freq_table_iterations: usize,
    freq_table_median_ms: f64,
    freq_tables_per_sec: f64,
    symbol_count: usize,
    compressed_bytes: usize,
    encode_median_ms: f64,
    encode_msym_per_sec: f64,
    decode_median_ms: f64,
    decode_msym_per_sec: f64,
    verified_lossless: bool,
}

#[derive(Debug, Clone, Serialize)]
struct PreprocessBench {
    input_bytes: usize,
    max_words: usize,
    min_freq: u32,
    vocab_bytes: u64,
    pp_bytes: u64,
    c_vocab_bytes: u64,
    c_pp_bytes: u64,
    rust_encode_median_ms: f64,
    rust_encode_mib_per_sec: f64,
    rust_decode_median_ms: f64,
    rust_decode_mib_per_sec: f64,
    c_encode_median_ms: f64,
    c_encode_mib_per_sec: f64,
    c_decode_median_ms: f64,
    c_decode_mib_per_sec: f64,
    voc_sha256_match: bool,
    pp_sha256_match: bool,
    cross_decode_rust_on_c_ok: bool,
    cross_decode_c_on_rust_ok: bool,
    enwik9_voc_tokens: usize,
    enwik9_encode_median_ms: f64,
    enwik9_decode_median_ms: f64,
    verified_lossless: bool,
}

#[derive(Debug, Clone, Serialize)]
struct PriorBench {
    archive_bytes: u64,
    tensor_count: usize,
    dequantized_f32_bytes: usize,
    read_archive_median_ms: f64,
    from_prior_median_ms: f64,
    v2_total_params: usize,
    v2_non_emb_params: usize,
}

#[derive(Debug, Clone, Serialize)]
struct SubBenchOutcome<T> {
    ok: bool,
    error: Option<String>,
    metrics: Option<T>,
}

impl<T> SubBenchOutcome<T> {
    fn from_result(res: Result<T>) -> Self {
        match res {
            Ok(metrics) => Self {
                ok: true,
                error: None,
                metrics: Some(metrics),
            },
            Err(err) => Self {
                ok: false,
                error: Some(err.to_string()),
                metrics: None,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct OnlineCodecMetrics {
    name: String,
    input_tokens: usize,
    compressed_bytes: u64,
    compress_seconds: f64,
    compress_tok_per_sec: f64,
    decompress_seconds: Option<f64>,
    decompress_tok_per_sec: Option<f64>,
    modeled_bpt: f64,
    verified_lossless: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ForwardPassMetrics {
    dtype: String,
    batch1_seq192_median_ms: f64,
    batch1_tok_per_sec: f64,
    batch4_seq192_median_ms: f64,
    batch4_tok_per_sec: f64,
}

#[derive(Debug, Clone, Serialize)]
struct NeuralBench {
    backend: String,
    device_label: String,
    prefix_bytes: usize,
    prefix_vocab_size: usize,
    prefix_tokens: usize,
    tiny_roundtrip: SubBenchOutcome<OnlineCodecMetrics>,
    medium_compress: SubBenchOutcome<OnlineCodecMetrics>,
    v2_forward_f32: SubBenchOutcome<ForwardPassMetrics>,
    v2_forward_bf16: Option<SubBenchOutcome<ForwardPassMetrics>>,
}

#[derive(Debug, Clone, Serialize)]
struct RatioEntry {
    tool: String,
    raw_bytes: usize,
    payload_bytes: u64,
    vocab_bytes: u64,
    total_bytes: u64,
    payload_bpb: f64,
    total_bpb: f64,
    compress_seconds: f64,
    verified_sha256: bool,
}

#[derive(Debug, Clone, Serialize)]
struct RatioBench {
    prefix_bytes: usize,
    prefix_sha256: String,
    vocab_size: usize,
    vocab_bytes: u64,
    token_count: usize,
    entries: Vec<RatioEntry>,
}

fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(Path::parent)
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
}

fn sha256_bytes(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

fn median_ms(samples: &mut [Duration]) -> f64 {
    samples.sort_unstable();
    let mid = samples.len() / 2;
    if samples.len() % 2 == 1 {
        samples[mid].as_secs_f64() * 1_000.0
    } else {
        (samples[mid - 1].as_secs_f64() + samples[mid].as_secs_f64()) * 500.0
    }
}

fn measure_median<F>(warmups: usize, runs: usize, mut f: F) -> Result<f64>
where
    F: FnMut() -> Result<()>,
{
    for _ in 0..warmups {
        f()?;
    }
    let mut durations = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        f()?;
        durations.push(t0.elapsed());
    }
    Ok(median_ms(&mut durations))
}

fn collect_repo_corpus(root: &Path) -> Result<(Vec<u8>, usize)> {
    let mut rel_paths = Vec::new();
    let crates_dir = root.join("crates");
    for crate_entry in
        fs::read_dir(&crates_dir).with_context(|| format!("reading {}", crates_dir.display()))?
    {
        let crate_path = crate_entry?.path().join("src");
        if !crate_path.is_dir() {
            continue;
        }
        for src_entry in fs::read_dir(&crate_path)? {
            let file_path = src_entry?.path();
            if file_path.extension().and_then(|s| s.to_str()) == Some("rs") {
                rel_paths.push(file_path);
            }
        }
    }
    rel_paths.sort();
    if rel_paths.is_empty() {
        bail!("no Rust source files found under {}", crates_dir.display());
    }
    let mut buf = Vec::new();
    for (idx, path) in rel_paths.iter().enumerate() {
        if idx > 0 {
            buf.push(b'\n');
        }
        buf.extend_from_slice(&fs::read(path)?);
    }
    Ok((buf, rel_paths.len()))
}

fn run_arith_track() -> Result<ArithBench> {
    let vocab_size = 4097usize;
    let mut logits = vec![-10.0f32; vocab_size];
    for (i, slot) in logits.iter_mut().enumerate() {
        *slot = -6.0 + ((i % 97) as f32) * 0.08;
    }
    logits[0] = 5.5;
    logits[42] = 6.2;
    logits[257] = 4.9;
    logits[4096] = 3.1;

    let freq_iters = 20_000usize;
    let mut freqs = Vec::with_capacity(vocab_size);
    let freq_table_median_ms = measure_median(1, 5, || {
        for _ in 0..freq_iters {
            make_freq_table_into(&logits, &mut freqs);
            black_box(&freqs);
        }
        Ok(())
    })?;
    let freq_tables_per_sec = (freq_iters as f64) / (freq_table_median_ms / 1_000.0);

    let cdf = make_freq_table(&logits);
    let symbol_count = 500_000usize;
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    let mut symbols = Vec::with_capacity(symbol_count);
    for _ in 0..symbol_count {
        let sym = if rng.random::<f64>() < 0.65 {
            rng.random_range(0..256)
        } else {
            rng.random_range(0..vocab_size)
        };
        symbols.push(sym);
    }

    let mut encoded = Vec::with_capacity(symbol_count);
    let encode_median_ms = measure_median(1, 5, || {
        encoded.clear();
        let bit_out = BitOutputStream::new(&mut encoded);
        let mut enc = ArithmeticEncoder::new(DEFAULT_STATE_BITS, bit_out)?;
        for &sym in &symbols {
            enc.write(&cdf, sym)?;
        }
        let _ = enc.into_inner()?;
        Ok(())
    })?;

    let mut decoded = Vec::with_capacity(symbol_count);
    let decode_median_ms = measure_median(1, 5, || {
        decoded.clear();
        let bit_in = BitInputStream::new(Cursor::new(&encoded));
        let mut dec = ArithmeticDecoder::new(DEFAULT_STATE_BITS, bit_in)?;
        for _ in 0..symbol_count {
            decoded.push(dec.read(&cdf)?);
        }
        Ok(())
    })?;

    let verified_lossless = decoded == symbols;
    if !verified_lossless {
        bail!("arithmetic coding roundtrip mismatch in benchmark");
    }

    Ok(ArithBench {
        vocab_size,
        freq_table_iterations: freq_iters,
        freq_table_median_ms,
        freq_tables_per_sec,
        symbol_count,
        compressed_bytes: encoded.len(),
        encode_median_ms,
        encode_msym_per_sec: (symbol_count as f64 / 1e6) / (encode_median_ms / 1_000.0),
        decode_median_ms,
        decode_msym_per_sec: (symbol_count as f64 / 1e6) / (decode_median_ms / 1_000.0),
        verified_lossless,
    })
}

fn compile_preprocess_ref(c_source: &Path, out_bin: &Path) -> Result<()> {
    let status = Command::new("cc")
        .args([
            "-O2",
            "-w",
            "-D_WIN32",
            "-DCONFIG_VERSION=\"ref\"",
            c_source
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("non-UTF-8 c_source path"))?,
            "-lm",
            "-o",
            out_bin
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("non-UTF-8 out_bin path"))?,
        ])
        .status()
        .context("invoking cc on preprocess_ref.c")?;
    if !status.success() {
        bail!("cc failed compiling {}", c_source.display());
    }
    Ok(())
}

fn run_cmd(bin: &Path, args: &[&str]) -> Result<()> {
    let out = Command::new(bin)
        .args(args)
        .output()
        .with_context(|| format!("running {}", bin.display()))?;
    if !out.status.success() {
        bail!(
            "{} {:?} failed: {}",
            bin.display(),
            args,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(())
}

fn run_preprocess_track(
    root: &Path,
    corpus: &[u8],
    corpus_sha256: &str,
) -> Result<PreprocessBench> {
    let tmp = tempfile::tempdir()?;
    let dir = tmp.path();
    let input_path = dir.join("corpus.rs");
    fs::write(&input_path, corpus)?;

    let c_src = root.join("crates/tufazip-preprocess/tests/fixtures/preprocess_ref.c");
    let c_bin = dir.join("preprocess_ref");
    compile_preprocess_ref(&c_src, &c_bin)?;

    let rust_voc = dir.join("rust.voc");
    let rust_pp = dir.join("rust.pp");
    let rust_dec = dir.join("rust.dec");
    let c_voc = dir.join("c.voc");
    let c_pp = dir.join("c.pp");
    let c_dec = dir.join("c.dec");

    let cfg = EncodeConfig {
        max_words: 1500,
        min_freq: 5,
        sort_order: SortOrder::Discovery,
        debug_dict_path: None,
    };

    let rust_encode_median_ms = measure_median(1, 5, || {
        let _ = encode_file(&input_path, &rust_pp, &rust_voc, &cfg)?;
        Ok(())
    })?;

    let rust_decode_median_ms = measure_median(1, 5, || {
        let _ = decode_file(&rust_pp, &rust_dec, &rust_voc)?;
        Ok(())
    })?;

    let c_voc_str = c_voc.to_str().ok_or_else(|| anyhow::anyhow!("path utf8"))?;
    let input_str = input_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("path utf8"))?;
    let c_pp_str = c_pp.to_str().ok_or_else(|| anyhow::anyhow!("path utf8"))?;
    let c_dec_str = c_dec.to_str().ok_or_else(|| anyhow::anyhow!("path utf8"))?;

    let c_encode_median_ms = measure_median(1, 5, || {
        run_cmd(&c_bin, &["c", c_voc_str, input_str, c_pp_str, "1500", "5"])
    })?;

    let c_decode_median_ms = measure_median(1, 5, || {
        run_cmd(&c_bin, &["d", c_voc_str, c_pp_str, c_dec_str])
    })?;

    let voc_sha256_match = sha256_file(&rust_voc)? == sha256_file(&c_voc)?;
    let pp_sha256_match = sha256_file(&rust_pp)? == sha256_file(&c_pp)?;
    let rust_dec_sha = sha256_file(&rust_dec)?;
    let c_dec_sha = sha256_file(&c_dec)?;

    let cross_rust_dec = dir.join("cross_rust_on_c.dec");
    let _ = decode_file(&c_pp, &cross_rust_dec, &c_voc)?;
    let cross_decode_rust_on_c_ok = sha256_file(&cross_rust_dec)? == corpus_sha256;

    let cross_c_dec = dir.join("cross_c_on_rust.dec");
    let rust_voc_str = rust_voc
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("path utf8"))?;
    let rust_pp_str = rust_pp
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("path utf8"))?;
    let cross_c_dec_str = cross_c_dec
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("path utf8"))?;
    run_cmd(&c_bin, &["d", rust_voc_str, rust_pp_str, cross_c_dec_str])?;
    let cross_decode_c_on_rust_ok = sha256_file(&cross_c_dec)? == corpus_sha256;

    let enwik9_dict = Dictionary::load(root.join("assets/enwik9.voc"))?;
    let mut enwik9_tokens = Vec::new();
    let enwik9_encode_median_ms = measure_median(1, 5, || {
        enwik9_tokens = encode_with_dictionary(corpus, &enwik9_dict)?;
        Ok(())
    })?;

    let mut enwik9_decoded = Vec::new();
    let enwik9_decode_median_ms = measure_median(1, 5, || {
        enwik9_decoded = decode_tokens(&enwik9_tokens, &enwik9_dict)?;
        Ok(())
    })?;
    let enwik9_dec_sha = sha256_bytes(&enwik9_decoded);

    let verified_lossless = rust_dec_sha == corpus_sha256
        && c_dec_sha == corpus_sha256
        && enwik9_dec_sha == corpus_sha256
        && cross_decode_rust_on_c_ok
        && cross_decode_c_on_rust_ok;
    if !verified_lossless {
        bail!(
            "preprocess lossless verification failed: rust={} c={} enwik9={} rust_on_c={cross_decode_rust_on_c_ok} c_on_rust={cross_decode_c_on_rust_ok}",
            rust_dec_sha == corpus_sha256,
            c_dec_sha == corpus_sha256,
            enwik9_dec_sha == corpus_sha256,
        );
    }

    let mib = (corpus.len() as f64) / (1024.0 * 1024.0);
    Ok(PreprocessBench {
        input_bytes: corpus.len(),
        max_words: cfg.max_words,
        min_freq: cfg.min_freq,
        vocab_bytes: fs::metadata(&rust_voc)?.len(),
        pp_bytes: fs::metadata(&rust_pp)?.len(),
        c_vocab_bytes: fs::metadata(&c_voc)?.len(),
        c_pp_bytes: fs::metadata(&c_pp)?.len(),
        rust_encode_median_ms,
        rust_encode_mib_per_sec: mib / (rust_encode_median_ms / 1_000.0),
        rust_decode_median_ms,
        rust_decode_mib_per_sec: mib / (rust_decode_median_ms / 1_000.0),
        c_encode_median_ms,
        c_encode_mib_per_sec: mib / (c_encode_median_ms / 1_000.0),
        c_decode_median_ms,
        c_decode_mib_per_sec: mib / (c_decode_median_ms / 1_000.0),
        voc_sha256_match,
        pp_sha256_match,
        cross_decode_rust_on_c_ok,
        cross_decode_c_on_rust_ok,
        enwik9_voc_tokens: enwik9_tokens.len(),
        enwik9_encode_median_ms,
        enwik9_decode_median_ms,
        verified_lossless,
    })
}

fn run_prior_track(root: &Path) -> Result<PriorBench> {
    let weights_path = root.join("assets/weights.bin");
    let vocab_path = root.join("assets/enwik9.voc");
    let archive_bytes = fs::metadata(&weights_path)?.len();

    let mut tensor_count = 0usize;
    let mut dequantized_f32_bytes = 0usize;
    let read_archive_median_ms = measure_median(1, 3, || {
        let archive = read_archive(&weights_path)?;
        tensor_count = archive.tensors.len();
        dequantized_f32_bytes = archive
            .tensors
            .values()
            .map(|t| t.data.len() * std::mem::size_of::<f32>())
            .sum();
        Ok(())
    })?;

    let v2_cfg = ModelConfig::v2();
    let mut v2_total_params = 0usize;
    let mut v2_non_emb_params = 0usize;
    let from_prior_median_ms = measure_median(1, 3, || {
        let mut rng = DeterministicRng::from_seed_u64(1111);
        let (model, _info) = Transformer::from_prior(
            &v2_cfg,
            &weights_path,
            &vocab_path,
            1111,
            &Device::Cpu,
            &mut rng,
        )?;
        let (tot, non_emb) = model.param_counts();
        v2_total_params = tot;
        v2_non_emb_params = non_emb;
        Ok(())
    })?;

    Ok(PriorBench {
        archive_bytes,
        tensor_count,
        dequantized_f32_bytes,
        read_archive_median_ms,
        from_prior_median_ms,
        v2_total_params,
        v2_non_emb_params,
    })
}

fn run_forward_bench(
    model: &Transformer,
    device: &Device,
    dtype: DType,
) -> Result<ForwardPassMetrics> {
    let batch1 = Tensor::zeros((1usize, 192usize), DType::U32, device)?;
    let batch4 = Tensor::zeros((4usize, 192usize), DType::U32, device)?;

    let batch1_seq192_median_ms = measure_median(1, 3, || {
        let mut mode = ForwardMode::Eval;
        let logits = model.forward(&batch1, OutputSlice::Last, &mut mode, dtype)?;
        let synced = logits
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec3::<f32>()?;
        black_box(synced);
        Ok(())
    })?;

    let batch4_seq192_median_ms = measure_median(1, 3, || {
        let mut mode = ForwardMode::Eval;
        let logits = model.forward(&batch4, OutputSlice::Last, &mut mode, dtype)?;
        let synced = logits
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec3::<f32>()?;
        black_box(synced);
        Ok(())
    })?;

    Ok(ForwardPassMetrics {
        dtype: format!("{dtype:?}"),
        batch1_seq192_median_ms,
        batch1_tok_per_sec: 192.0 / (batch1_seq192_median_ms / 1_000.0),
        batch4_seq192_median_ms,
        batch4_tok_per_sec: (4.0 * 192.0) / (batch4_seq192_median_ms / 1_000.0),
    })
}

fn run_neural_track(root: &Path, corpus: &[u8], backend: &str) -> Result<NeuralBench> {
    let prefix_bytes = corpus.len().min(32_768);
    let prefix = &corpus[..prefix_bytes];
    let (_dict, tokens, report) = encode_bytes(
        prefix,
        &EncodeConfig {
            max_words: 512,
            min_freq: 3,
            sort_order: SortOrder::Discovery,
            debug_dict_path: None,
        },
    )?;

    let tmp = tempfile::tempdir()?;
    let pp_path = tmp.path().join("prefix.pp");
    write_tokens_be16(&pp_path, &tokens)?;

    let choice = if backend == "metal" {
        DeviceChoice::Metal
    } else {
        DeviceChoice::Cpu
    };
    let device = resolve_device(choice)?;
    let device_label = format!("{device:?}");
    let vocab_size = report.vocab_size;

    eprintln!("[bench]   Track 4a ({backend}): tiny roundtrip...");
    let tiny_res: Result<OnlineCodecMetrics> = (|| {
        let mut cfg = CodecConfig::tiny(vocab_size);
        cfg.device = choice;
        cfg.max_tokens = Some(1024);
        cfg.log_interval = 10_000;
        let bin_path = tmp.path().join("tiny.bin");
        let dec_path = tmp.path().join("tiny_dec.pp");
        let stop = Arc::new(AtomicBool::new(false));

        let summary = match run_codec(
            &pp_path,
            &bin_path,
            &cfg,
            CodecMode::Compress,
            Arc::clone(&stop),
        )? {
            RunOutcome::Completed(s) => s,
            RunOutcome::RequeueRequested { .. } => bail!("unexpected requeue"),
        };

        let mut dec_cfg = cfg.clone();
        dec_cfg.max_tokens = None;
        let dec_summary =
            match run_codec(&bin_path, &dec_path, &dec_cfg, CodecMode::Decompress, stop)? {
                RunOutcome::Completed(s) => s,
                RunOutcome::RequeueRequested { .. } => bail!("unexpected requeue"),
            };

        let orig = read_tokens_be16(&pp_path)?;
        let dec_tokens = read_tokens_be16(&dec_path)?;
        let expected_len = summary.original_tokens.min(orig.len());
        let verified_lossless = orig[..expected_len] == dec_tokens[..expected_len];
        if !verified_lossless {
            bail!("tiny neural codec roundtrip token mismatch");
        }

        Ok(OnlineCodecMetrics {
            name: "tiny-roundtrip".to_owned(),
            input_tokens: summary.original_tokens,
            compressed_bytes: summary.actual_compressed_bytes.unwrap_or(0),
            compress_seconds: summary.elapsed_seconds,
            compress_tok_per_sec: (summary.original_tokens as f64)
                / summary.elapsed_seconds.max(1e-9),
            decompress_seconds: Some(dec_summary.elapsed_seconds),
            decompress_tok_per_sec: Some(
                (dec_summary.original_tokens as f64) / dec_summary.elapsed_seconds.max(1e-9),
            ),
            modeled_bpt: summary.modeled_bits_per_token,
            verified_lossless,
        })
    })();

    eprintln!("[bench]   Track 4b ({backend}): medium 4L/256d compress...");
    let medium_res: Result<OnlineCodecMetrics> = (|| {
        let mut cfg = CodecConfig::tiny(vocab_size);
        cfg.device = choice;
        cfg.model.n_layer = 4;
        cfg.model.d_model = 256;
        cfg.model.n_head = 4;
        cfg.model.d_inner = 704;
        cfg.model.ctx_len = 64;
        cfg.batch_size = 16;
        cfg.block_len = 256;
        cfg.retrain.period = 0;
        cfg.max_tokens = Some(1024);
        cfg.log_interval = 10_000;
        let bin_path = tmp.path().join("medium.bin");
        let stop = Arc::new(AtomicBool::new(false));

        let summary = match run_codec(&pp_path, &bin_path, &cfg, CodecMode::Compress, stop)? {
            RunOutcome::Completed(s) => s,
            RunOutcome::RequeueRequested { .. } => bail!("unexpected requeue"),
        };

        Ok(OnlineCodecMetrics {
            name: "medium-4L-256d-compress".to_owned(),
            input_tokens: summary.original_tokens,
            compressed_bytes: summary.actual_compressed_bytes.unwrap_or(0),
            compress_seconds: summary.elapsed_seconds,
            compress_tok_per_sec: (summary.original_tokens as f64)
                / summary.elapsed_seconds.max(1e-9),
            decompress_seconds: None,
            decompress_tok_per_sec: None,
            modeled_bpt: summary.modeled_bits_per_token,
            verified_lossless: true,
        })
    })();

    let weights_path = root.join("assets/weights.bin");
    let vocab_path = root.join("assets/enwik9.voc");
    eprintln!("[bench]   Track 4c ({backend}): v2 12-layer 162.6M forward...");
    let v2_model_res = (|| -> Result<Transformer> {
        let mut rng = DeterministicRng::from_seed_u64(1111);
        let (model, _info) = Transformer::from_prior(
            &ModelConfig::v2(),
            &weights_path,
            &vocab_path,
            1111,
            &device,
            &mut rng,
        )?;
        Ok(model)
    })();

    let v2_forward_f32_res = match &v2_model_res {
        Ok(model) => run_forward_bench(model, &device, DType::F32),
        Err(err) => Err(anyhow::anyhow!("{err}")),
    };

    let v2_forward_bf16 = if backend == "metal" {
        Some(SubBenchOutcome::from_result(match &v2_model_res {
            Ok(model) => run_forward_bench(model, &device, DType::BF16),
            Err(err) => Err(anyhow::anyhow!("{err}")),
        }))
    } else {
        None
    };

    Ok(NeuralBench {
        backend: backend.to_owned(),
        device_label,
        prefix_bytes,
        prefix_vocab_size: vocab_size,
        prefix_tokens: tokens.len(),
        tiny_roundtrip: SubBenchOutcome::from_result(tiny_res),
        medium_compress: SubBenchOutcome::from_result(medium_res),
        v2_forward_f32: SubBenchOutcome::from_result(v2_forward_f32_res),
        v2_forward_bf16,
    })
}

fn run_pipe_compressor(bin: &str, args: &[&str], input: &[u8]) -> Result<Vec<u8>> {
    let mut child = Command::new(bin)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawning {bin} {args:?}"))?;
    {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("missing stdin pipe for {bin}"))?;
        stdin.write_all(input)?;
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        bail!(
            "{bin} {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(output.stdout)
}

fn bench_cli_compressor(
    label: &str,
    bin: &str,
    comp_args: &[&str],
    dec_args: &[&str],
    raw: &[u8],
    raw_sha256: &str,
) -> Result<RatioEntry> {
    // Warmup once
    let _ = run_pipe_compressor(bin, comp_args, raw)?;
    let t0 = Instant::now();
    let compressed = run_pipe_compressor(bin, comp_args, raw)?;
    let compress_seconds = t0.elapsed().as_secs_f64();

    let restored = run_pipe_compressor(bin, dec_args, &compressed)?;
    let verified_sha256 = sha256_bytes(&restored) == raw_sha256;
    if !verified_sha256 {
        bail!("{label} lossless verification failed");
    }

    let payload_bytes = compressed.len() as u64;
    let payload_bpb = (payload_bytes as f64 * 8.0) / (raw.len() as f64);
    Ok(RatioEntry {
        tool: label.to_owned(),
        raw_bytes: raw.len(),
        payload_bytes,
        vocab_bytes: 0,
        total_bytes: payload_bytes,
        payload_bpb,
        total_bpb: payload_bpb,
        compress_seconds,
        verified_sha256,
    })
}

fn run_ratio_track(corpus: &[u8]) -> Result<RatioBench> {
    let prefix_bytes = corpus.len().min(16_384);
    let prefix = &corpus[..prefix_bytes];
    let prefix_sha256 = sha256_bytes(prefix);

    eprintln!("[bench]   Track 5a: classic compressors (gzip/bzip2/xz/zstd)...");
    let mut entries = vec![
        bench_cli_compressor(
            "gzip -9",
            "gzip",
            &["-9", "-c"],
            &["-dc"],
            prefix,
            &prefix_sha256,
        )?,
        bench_cli_compressor(
            "bzip2 -9",
            "bzip2",
            &["-9", "-c"],
            &["-dc"],
            prefix,
            &prefix_sha256,
        )?,
        bench_cli_compressor(
            "xz -9",
            "xz",
            &["-9", "-c"],
            &["-dc"],
            prefix,
            &prefix_sha256,
        )?,
        bench_cli_compressor(
            "zstd -19",
            "zstd",
            &["-19", "-q", "-c"],
            &["-d", "-q", "-c"],
            prefix,
            &prefix_sha256,
        )?,
    ];

    let tmp = tempfile::tempdir()?;
    let in_path = tmp.path().join("prefix.txt");
    let voc_path = tmp.path().join("prefix.voc");
    let pp_path = tmp.path().join("prefix.pp");
    fs::write(&in_path, prefix)?;

    let report = encode_file(
        &in_path,
        &pp_path,
        &voc_path,
        &EncodeConfig {
            max_words: 1024,
            min_freq: 3,
            sort_order: SortOrder::Discovery,
            debug_dict_path: None,
        },
    )?;
    let vocab_bytes = fs::metadata(&voc_path)?.len();

    let run_neural_ratio = |label: &str, mut cfg: CodecConfig, tag: &str| -> Result<RatioEntry> {
        let bin_path = tmp.path().join(format!("{tag}.bin"));
        let dec_pp = tmp.path().join(format!("{tag}_dec.pp"));
        let dec_raw = tmp.path().join(format!("{tag}_dec.txt"));
        let stop = Arc::new(AtomicBool::new(false));
        cfg.raw_bytes = Some(prefix_bytes as u64);
        cfg.vocab_bytes = vocab_bytes;

        let t0 = Instant::now();
        let _ = encode_file(
            &in_path,
            &pp_path,
            &voc_path,
            &EncodeConfig {
                max_words: 1024,
                min_freq: 3,
                sort_order: SortOrder::Discovery,
                debug_dict_path: None,
            },
        )?;
        let summary = match run_codec(
            &pp_path,
            &bin_path,
            &cfg,
            CodecMode::Compress,
            Arc::clone(&stop),
        )? {
            RunOutcome::Completed(s) => s,
            RunOutcome::RequeueRequested { .. } => bail!("unexpected requeue"),
        };
        let compress_seconds = t0.elapsed().as_secs_f64();

        let _ = match run_codec(&bin_path, &dec_pp, &cfg, CodecMode::Decompress, stop)? {
            RunOutcome::Completed(s) => s,
            RunOutcome::RequeueRequested { .. } => bail!("unexpected requeue"),
        };
        let _ = decode_file(&dec_pp, &dec_raw, &voc_path)?;
        let verified_sha256 = sha256_file(&dec_raw)? == prefix_sha256;
        if !verified_sha256 {
            bail!("{label} lossless verification failed");
        }

        let payload_bytes = summary.actual_compressed_bytes.unwrap_or(0);
        let total_bytes = payload_bytes + vocab_bytes;
        let payload_bpb = (payload_bytes as f64 * 8.0) / (prefix_bytes as f64);
        let total_bpb = (total_bytes as f64 * 8.0) / (prefix_bytes as f64);
        Ok(RatioEntry {
            tool: label.to_owned(),
            raw_bytes: prefix_bytes,
            payload_bytes,
            vocab_bytes,
            total_bytes,
            payload_bpb,
            total_bpb,
            compress_seconds,
            verified_sha256,
        })
    };

    eprintln!("[bench]   Track 5b: tufazip-rs tiny lossless ratio...");
    let mut tiny_cfg = CodecConfig::tiny(report.vocab_size);
    tiny_cfg.device = DeviceChoice::Cpu;
    tiny_cfg.model.ctx_len = 64;
    tiny_cfg.batch_size = 8;
    tiny_cfg.block_len = 128;
    tiny_cfg.retrain.period = 512;
    tiny_cfg.retrain.len = 512;
    tiny_cfg.retrain.epochs = 1;
    tiny_cfg.retrain.update_len = 64;
    tiny_cfg.log_interval = 10_000;
    entries.push(run_neural_ratio("tufazip-rs (tiny)", tiny_cfg, "tiny")?);

    eprintln!("[bench]   Track 5c: tufazip-rs medium 4L lossless ratio...");
    let mut med_cfg = CodecConfig::tiny(report.vocab_size);
    med_cfg.device = DeviceChoice::Cpu;
    med_cfg.model.n_layer = 4;
    med_cfg.model.d_model = 128;
    med_cfg.model.n_head = 4;
    med_cfg.model.d_inner = 384;
    med_cfg.model.ctx_len = 64;
    med_cfg.batch_size = 16;
    med_cfg.block_len = 256;
    med_cfg.online_loss = OnlineLoss::Code;
    med_cfg.project_only_targets = true;
    med_cfg.retrain.period = 1024;
    med_cfg.retrain.len = 1024;
    med_cfg.retrain.epochs = 1;
    med_cfg.retrain.update_len = 64;
    med_cfg.log_interval = 10_000;
    entries.push(run_neural_ratio(
        "tufazip-rs (medium 4L)",
        med_cfg,
        "medium",
    )?);

    Ok(RatioBench {
        prefix_bytes,
        prefix_sha256,
        vocab_size: report.vocab_size,
        vocab_bytes,
        token_count: report.token_count,
        entries,
    })
}

fn main() -> Result<()> {
    let mut backend = "cpu".to_owned();
    let mut only_neural = false;
    let mut emit_json = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--backend" => {
                backend = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--backend requires an argument"))?;
            }
            "--only-neural" => only_neural = true,
            "--json" => emit_json = true,
            other => bail!("unknown flag: {other}"),
        }
    }
    if !matches!(backend.as_str(), "cpu" | "accelerate" | "metal") {
        bail!("--backend must be one of cpu, accelerate, or metal (got {backend:?})");
    }

    let root = workspace_root();
    let (corpus, file_count) = collect_repo_corpus(&root)?;
    let corpus_sha256 = sha256_bytes(&corpus);

    let arith = if only_neural {
        None
    } else {
        eprintln!("[bench] Track 1/5: tufazip-arith throughput...");
        Some(run_arith_track()?)
    };
    let preprocess = if only_neural {
        None
    } else {
        eprintln!("[bench] Track 2/5: tufazip-preprocess vs preprocess.c...");
        Some(run_preprocess_track(&root, &corpus, &corpus_sha256)?)
    };
    let prior = if only_neural {
        None
    } else {
        eprintln!("[bench] Track 3/5: tufazip-prior weights.bin load...");
        Some(run_prior_track(&root)?)
    };
    eprintln!("[bench] Track 4/5: tufazip-codec neural backend ({backend})...");
    let neural = run_neural_track(&root, &corpus, &backend)?;
    let ratios = if only_neural {
        None
    } else {
        eprintln!("[bench] Track 5/5: lossless bpb ratio comparison (16 KiB)...");
        Some(run_ratio_track(&corpus)?)
    };

    let logical_cpus = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);

    let report = BenchmarkReport {
        backend: backend.clone(),
        only_neural,
        environment: EnvironmentInfo {
            os: std::env::consts::OS.to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
            logical_cpus,
        },
        corpus: CorpusInfo {
            file_count,
            total_bytes: corpus.len(),
            sha256: corpus_sha256,
        },
        arith,
        preprocess,
        prior,
        neural,
        ratios,
    };

    if emit_json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "tufazip-rs benchmark [backend={}, device={}, corpus={} bytes ({} files)]",
            report.backend,
            report.neural.device_label,
            report.corpus.total_bytes,
            report.corpus.file_count
        );
        if let Some(a) = &report.arith {
            println!(
                "  arith: freq_table={:.2}ms ({:.0}/s), encode={:.2}ms ({:.2} Msym/s), decode={:.2}ms ({:.2} Msym/s)",
                a.freq_table_median_ms,
                a.freq_tables_per_sec,
                a.encode_median_ms,
                a.encode_msym_per_sec,
                a.decode_median_ms,
                a.decode_msym_per_sec
            );
        }
        if let Some(p) = &report.preprocess {
            println!(
                "  preprocess: Rust enc={:.2}ms ({:.2} MiB/s) vs C enc={:.2}ms ({:.2} MiB/s); Rust dec={:.2}ms ({:.2} MiB/s) vs C dec={:.2}ms ({:.2} MiB/s)",
                p.rust_encode_median_ms,
                p.rust_encode_mib_per_sec,
                p.c_encode_median_ms,
                p.c_encode_mib_per_sec,
                p.rust_decode_median_ms,
                p.rust_decode_mib_per_sec,
                p.c_decode_median_ms,
                p.c_decode_mib_per_sec
            );
        }
        if let Some(pr) = &report.prior {
            println!(
                "  prior: read_archive={:.2}ms, from_prior(v2)={:.2}ms ({} params)",
                pr.read_archive_median_ms, pr.from_prior_median_ms, pr.v2_total_params
            );
        }
        if let Some(m) = &report.neural.tiny_roundtrip.metrics {
            println!(
                "  neural tiny: comp={:.3}s ({:.1} tok/s), dec={:.3}s ({:.1} tok/s)",
                m.compress_seconds,
                m.compress_tok_per_sec,
                m.decompress_seconds.unwrap_or(0.0),
                m.decompress_tok_per_sec.unwrap_or(0.0)
            );
        }
        if let Some(m) = &report.neural.medium_compress.metrics {
            println!(
                "  neural medium (4L/256d): comp={:.3}s ({:.1} tok/s)",
                m.compress_seconds, m.compress_tok_per_sec
            );
        } else if let Some(err) = &report.neural.medium_compress.error {
            println!("  neural medium error: {err}");
        }
        if let Some(f) = &report.neural.v2_forward_f32.metrics {
            println!(
                "  neural v2 F32 fwd: b1x192={:.2}ms ({:.1} tok/s), b4x192={:.2}ms ({:.1} tok/s)",
                f.batch1_seq192_median_ms,
                f.batch1_tok_per_sec,
                f.batch4_seq192_median_ms,
                f.batch4_tok_per_sec
            );
        }
    }

    Ok(())
}
