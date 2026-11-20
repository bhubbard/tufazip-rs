# tufazip-rs

Pure-Rust port of [Tufalabs/tufazip](https://github.com/Tufalabs/tufazip) (`tufazip v2`), a neural lossless data compressor and decompressor combining Fabrice Bellard's dictionary preprocessor, a Peri-RMSNorm SwiGLU Transformer with RoPE/learned-relative attention, online Adam adaptation with prefix replay, and a 32-bit resumable arithmetic coder.

- **GitHub Pages:** <https://code.brandonhubbard.com/tufazip-rs/>
- **Rustdoc API Reference:** <https://code.brandonhubbard.com/tufazip-rs/api/>
- **Benchmarks:** [`BENCHMARKS.md`](BENCHMARKS.md) &amp; [Live Benchmarks](https://code.brandonhubbard.com/tufazip-rs/#benchmarks)

## Headline Benchmarks (Apple M3 Pro, `--release`)

Measured via paired back-to-back runs against `fdf2b21` (`lowpowermode 1`, load avg `16.89–17.41`):

- **`tufazip-arith`:** **64,401 tables/s** (`4,097`-symbol `make_freq_table_into`, **1.67&times;** faster), **25.74 Msym/s** encode (**1.08&times;**), **12.24 Msym/s** decode (**1.14&times;**).
- **`tufazip-preprocess` vs `preprocess.c` (`cc -O2`):** Safe pure-Rust bigram induction now runs at **3.05 MiB/s** (**2.56&times;** over baseline; **1.53&times; faster** than `preprocess.c` at `1.99 MiB/s`), and streaming decode runs at **151.52 MiB/s** (**11.84&times;** faster than the C CLI) with cycle-detection hardening.
- **`tufazip-codec` Neural Backends:** Online coding improved **1.39&times;–1.64&times;** on CPU (`2,490.9 tok/s` tiny), **1.22&times;–1.32&times;** on **Apple Accelerate BLAS** (`3,178–3,396 tok/s` tiny, `524.3 tok/s` medium `4L`), and **1.41&times;–2.31&times;** on **Apple Metal GPU**, while Metal reaches **9.29&times;** (`51.31 ms` vs `476.41 ms` on CPU) on the **`v2` 12-layer 162.6M-parameter** forward pass.

See [`BENCHMARKS.md`](BENCHMARKS.md) for the full paired A/B table, all 5 tracks, compression ratio (`bpb`) comparison, and reproduction commands (48 workspace tests passing).

## Workspace Layout

| Crate | Purpose | Upstream Source |
| :--- | :--- | :--- |
| `crates/tufazip-arith` | 32-bit range arithmetic encoder/decoder, big-endian bit streams, softmax-to-CDF table quantization, and exact checkpoint state | `arithmetic_coder.py`, `make_freq_table` in `nn.py` |
| `crates/tufazip-preprocess` | Pure-Rust NNCP case/space transform, iterative bigram dictionary induction, and streaming `.voc` / `.pp` decoder | `preprocess.c` |
| `crates/tufazip-prior` | Binary `QPI1` / `QPR1` / `LRA1` / `QLA1` quantized prior reader, writer, and depth-growth / loop-untying validator | `read_tensor_archive`, `load_depth_prior` in `nn.py` |
| `crates/tufazip-codec` | Candle-backed Transformer, online Adam & prefix replay, gradient clipping, and atomic `safetensors` + JSON checkpoint resume | `nn.py`, `checkpoint_utils.py` |
| `crates/tufazip-cli` | Unified `tufazip` binary (`compress`, `decompress`, `codec`, `preprocess`, `v2-decode`, `inspect-prior`) | `tufazip`, `nn.py`, `preprocess.c` |

## Included Assets

Verified upstream artifacts in `assets/`:

| File | Bytes | SHA-256 |
| :--- | ---: | :--- |
| `assets/enwik9.voc` | 34,421 | `a63e517783e84a778f3f08c1f2321f3e2035e64d34c8e9206f03591460e44629` |
| `assets/weights.bin` | 4,334,741 | `e94c089f8eb985a87307dca3a7702385fcaccd14d95743811f8788f7da24448d` |

## Building

```bash
# Default pure-Rust CPU backend
cargo build --release --bin tufazip

# Apple Silicon Accelerate BLAS backend (macOS)
cargo build --release --bin tufazip --features tufazip-codec/accelerate

# Apple Silicon Metal GPU backend (macOS)
cargo build --release --bin tufazip --features tufazip-codec/metal

# NVIDIA CUDA backend (Linux/Windows with CUDA toolkit)
cargo build --release --bin tufazip --features tufazip-codec/cuda
```

## Usage

### Inspect the v2 `weights.bin` prior
```bash
cargo run --bin tufazip -- inspect-prior assets/weights.bin --vocab assets/enwik9.voc
```

### Standalone NNCP Dictionary Preprocessor (`preprocess.c` equivalent)
```bash
# Induce vocabulary & encode raw text to big-endian u16 tokens
cargo run --release --bin tufazip -- preprocess encode out.voc input.txt tokens.pp 4097 5

# Decode tokens back to exact original bytes
cargo run --release --bin tufazip -- preprocess decode out.voc tokens.pp recovered.txt
```

### End-to-End Neural Compression & Decompression
```bash
# Compress using fast tiny preset (or --preset v2 for full 12-layer v2 geometry)
cargo run --release --bin tufazip -- compress input.txt archive.tz --vocab out.voc --preset tiny

# Decompress back to exact original bytes
cargo run --release --bin tufazip -- decompress archive.tz recovered.txt --vocab out.voc --preset tiny
```

### Low-Level Token Codec (`nn.py` equivalent)
```bash
cargo run --release --bin tufazip -- codec --input tokens.pp --output payload.bin --preset tiny --checkpoint-dir ./ckpt
cargo run --release --bin tufazip -- codec --decompress --input payload.bin --output decoded.pp --preset tiny --checkpoint-dir ./ckpt
```

### Upstream `v2-decode` Wrapper
```bash
cargo run --release --bin tufazip -- v2-decode /path/to/enwik9-v2.bin /path/to/enwik9 ./tufazip-v2-state
```

## Determinism Note

`tufazip-rs` provides **bit-exact self-consistent compression, decompression, and checkpoint-resume** across CPU and supported backends using fixed PRNG streams (`ChaCha8Rng`), deterministic `safetensors` + arithmetic-coder checkpoints, and exact integer frequency tables.

> [!NOTE]
> Upstream's precomputed `enwik9-v2.bin` artifact (`100,958,324` bytes) was generated with a pinned 2× NVIDIA B200 CUDA 13.0 / cuDNN 9.20 BF16 FlashAttention + NCCL DDP floating-point rounding trajectory (`README.md` in `Tufalabs/tufazip`). Single-device or non-B200 hardware (such as Apple Silicon CPU/Metal) differs at the last ULP of BF16/F32 softmax/GEMM reduction, which will diverge over hundreds of millions of online gradient steps against that precomputed bitstream; pass `--skip-verify` to `v2-decode` if experimenting on other hardware.

## Verification

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo doc --workspace --no-deps
```

## License & Attributions

MIT License — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
