# `tufazip-rs` Benchmarks

Measured performance, parity, backend scaling, and lossless compression benchmarks for [`tufazip-rs`](https://github.com/bhubbard/tufazip-rs) on Apple Silicon. Every measurement on this page comes from [`crates/tufazip-cli/examples/benchmark.rs`](crates/tufazip-cli/examples/benchmark.rs) compiled in `--release` mode (`lto = "thin"`, `codegen-units = 1`).

---

## Test Environment & Corpus Provenance

| Property | Value |
| :--- | :--- |
| **Hardware** | Apple M3 Pro (11 physical / 11 logical CPUs, `aarch64`), 18 GiB unified memory |
| **Operating System** | macOS 27.2.1 (`BuildVersion 26B131`) |
| **Rust Toolchain** | `rustc 1.100.0 (0ef1bc386 2026-11-09)` / `cargo 1.100.0` |
| **C Reference Compiler** | Apple Clang (`cc -O2 -w -D_WIN32 -DCONFIG_VERSION="ref"`) compiling unmodified `preprocess.c` |
| **In-Repo Corpus** | All 21 sorted `crates/*/src/*.rs` source files concatenated (`266,014` bytes) |
| **Corpus SHA-256** | `15591fb8e68858fb5511d7ca3de4193a650fea1b22e4e3bdac2e5b7b1b92aed2` |
| **Methodology** | 1 warmup + 3–5 timed iterations (median reported); GPU timings force CPU host sync (`to_device(&Device::Cpu)?.to_vec3::<f32>()`) before stopping the timer; backends run sequentially without contention |

### Reproduction Commands

```bash
# Full 5-track suite using Apple Accelerate BLAS (CPU + Accelerate + Classic Compressors)
cargo run --release --example benchmark --features tufazip-codec/accelerate -- --backend accelerate

# Neural comparison on pure-Rust CPU backend
cargo run --release --example benchmark -- --backend cpu --only-neural

# Neural comparison on Apple Metal GPU backend
cargo run --release --example benchmark --features tufazip-codec/metal -- --backend metal --only-neural
```

---

## 1. `tufazip-arith` — 32-Bit Arithmetic Coder & CDF Table Throughput

Measured over `4,097`-symbol vocabularies (`FREQ_TOTAL = 10_000_000`, round-half-to-even quantization) and `500,000` deterministic `ChaCha8Rng` symbols (`954,384` compressed output bytes, lossless roundtrip verified):

| Operation | Workload | Median Time | Throughput | Verification |
| :--- | :--- | ---: | ---: | :--- |
| `make_freq_table_into` | 20,000 tables &times; 4,097 logits | 425.32 ms | **47,024 tables/s** | `FREQ_TOTAL >= 10M` |
| `ArithmeticEncoder::write` | 500,000 symbols &rarr; 954,384 B | 19.47 ms | **25.68 Msym/s** | Exact bitstream |
| `ArithmeticDecoder::read` | 500,000 symbols decoded | 45.21 ms | **11.06 Msym/s** | 500,000/500,000 exact match |

---

## 2. `tufazip-preprocess` (Safe Pure Rust) vs Upstream `preprocess.c` (`cc -O2`)

Benchmarked on the full `266,014`-byte source corpus (`max_words = 1500`, `min_freq = 5`, `SortOrder::Discovery`, producing `10,573`-byte `.voc` and `261,716`-byte `.pp` streams in both Rust and C):

| Pipeline / Mode | Median Time | Throughput | Output Size | Verification |
| :--- | ---: | ---: | :--- | :--- |
| **Pure Rust Encode** (`tufazip-preprocess`) | 187.17 ms | 1.36 MiB/s | `10,573 B` `.voc` / `261,716 B` `.pp` | Bidirectional C&#8651;Rust lossless SHA-256 |
| **Upstream C Encode** (`preprocess.c`, `cc -O2`) | 99.65 ms | 2.55 MiB/s | `10,573 B` `.voc` / `261,716 B` `.pp` | Bidirectional C&#8651;Rust lossless SHA-256 |
| **Pure Rust Decode** (`tufazip-preprocess`) | **2.35 ms** | **107.93 MiB/s** | `266,014 B` raw | Exact corpus SHA-256 |
| **Upstream C Decode** (`preprocess.c`, `cc -O2`) | 17.51 ms | 14.49 MiB/s | `266,014 B` raw | Exact corpus SHA-256 |
| **Fixed `assets/enwik9.voc` Encode** (`encode_with_dictionary`) | 8.25 ms | 30.74 MiB/s | 175,442 tokens | Exact roundtrip SHA-256 |
| **Fixed `assets/enwik9.voc` Decode** (`decode_tokens`) | 1.82 ms | 139.58 MiB/s | `266,014 B` raw | Exact corpus SHA-256 |

### Parity & Cycle-Detection Notes

1. **In-process decode speedup:** Pure-Rust streaming decoding (`2.35 ms`, `107.93 MiB/s`) runs **7.45&times; faster** than invoking the compiled C binary (`17.51 ms`), while Rust induction (`187.17 ms`) runs within `1.88&times;` of C&rsquo;s custom open-addressed bigram pointer table (`99.65 ms`).
2. **Equal-score `qsort` tie-breaking:** On the full 21-file `266,014`-byte corpus (`max_words = 1500`), unstable libc `qsort` in `preprocess.c` and Rust `slice::sort_by` break equal `f32` bigram score ties differently, so `.voc` and `.pp` digests differ (`voc_sha256_match = false`, `pp_sha256_match = false`) while yielding identical file sizes (`10,573 B` / `261,716 B`) and passing **bidirectional C&#8651;Rust lossless cross-decoding** (`cross_decode_rust_on_c_ok = true`, `cross_decode_c_on_rust_ok = true`). Single-file inputs without tie divergence still match bit-for-bit (`c_differential.rs`).
3. **Upstream oscillation bug fixed in `tufazip-preprocess`:** Unmodified `preprocess.c` enters an infinite period-6 bigram oscillation at `max_words = 4097, min_freq = 5` on the Rust source corpus (`word_count` cycling `762 -> 770 -> 763 -> 769 -> 763 -> 770 -> 762`), because Bellard&rsquo;s original loop only checks `word_count == word_count_prev`. `tufazip-preprocess` tracks seen loop states (`fnv1a_u16` + `HashSet`) to terminate cleanly on oscillating inputs without altering non-cyclic outputs.

---

## 3. `tufazip-prior` — `QPI1` Archive & 12-Layer `v2` Expansion (`assets/weights.bin`)

| Operation | Artifact / Geometry | Median Time | Notes |
| :--- | :--- | ---: | :--- |
| `read_archive` | `4,334,741 B` `QPI1`, 26 tensors &rarr; `136,366,080 B` dequantized `f32` | **89.88 ms** | ~46.0 MiB/s compressed / **1,447 MiB/s** dequantized `f32` |
| `Transformer::from_prior` (`v2`) | 2 &rarr; 12 layers (`[0,1]&times;6`), `162,585,600` total (`154,194,944` non-emb) params | **2,574.41 ms** | Zero-init fast path avoids sampling 162.6M discarded normals |

---

## 4. `tufazip-codec` — Neural Compute Backend Scaling

Measured on a deterministic `32,768`-byte prefix of the in-repo corpus (`511`-symbol vocab, `19,290` tokens; online coding capped at `1,024` tokens, outputting bit-exact payloads of `806 B` for `tiny` and `777 B` for `medium`, plus full `v2` `162.6M`-parameter `OutputSlice::Last` forward passes):

| Workload | Pure-Rust CPU (Baseline) | Apple Accelerate BLAS | Apple Metal GPU | Fastest vs CPU |
| :--- | ---: | ---: | ---: | :--- |
| **Online `tiny` Compress** (`2L`, `d=64`, `batch=4`, `1,024` tok &rarr; `806 B`) | 0.565 s (1,813.7 tok/s) | **0.401 s (2,554.7 tok/s)** | 21.000 s (48.8 tok/s)&dagger; | **Accelerate 1.41&times;** |
| **Online `tiny` Decompress** (`1,024` tok from `806 B`) | 0.578 s (1,772.6 tok/s) | **0.416 s (2,461.8 tok/s)** | 7.161 s (143.0 tok/s)&dagger; | **Accelerate 1.39&times;** |
| **Online `medium` Compress** (`4L`, `d=256`, `d_inner=704`, `batch=16`, `1,024` tok &rarr; `777 B`) | 4.129 s (248.0 tok/s) | **1.951 s (524.8 tok/s)** | 4.731 s (216.4 tok/s)&dagger; | **Accelerate 2.12&times;** |
| **`v2` 12L 162.6M Forward (`F32`, `batch=1`, `seq=192`)** | 441.52 ms (434.9 tok/s) | 157.89 ms (1,216.0 tok/s) | **54.90 ms (3,497.4 tok/s)** | **Metal 8.04&times;** |
| **`v2` 12L 162.6M Forward (`BF16`, `batch=1`, `seq=192`)** | &mdash; | &mdash; | **61.59 ms (3,117.4 tok/s)** | **Metal 7.17&times;** |
| **`v2` 12L 162.6M Forward (`F32`, `batch=4`, `seq=192`)** | 1,623.47 ms (473.1 tok/s) | 1,094.22 ms (701.9 tok/s) | **268.92 ms (2,855.8 tok/s)** | **Metal 6.04&times;** |
| **`v2` 12L 162.6M Forward (`BF16`, `batch=4`, `seq=192`)** | &mdash; | &mdash; | **274.02 ms (2,802.7 tok/s)** | **Metal 5.92&times;** |

&dagger; *Online arithmetic coding synchronizes logits to CPU after every coding step. For tiny/medium matrices (`d=64..256`, `batch=4..16`), first-run Metal shader compilation and per-step command-buffer synchronization dominate runtime, whereas **Apple Accelerate BLAS** delivers `1.39&times;`–`2.12&times;` speedups with zero GPU dispatch overhead. Once tensors scale to **`v2` 12-layer 162.6M parameters**, **Apple Metal GPU** jumps to **8.04&times;** (`54.90 ms`) at `batch=1` and **6.04&times;** (`268.92 ms`) at `batch=4`.*

---

## 5. Lossless Compression Ratio (`bpb`) vs Classic Compressors

Measured on a `16,384`-byte prefix of the in-repo Rust corpus (SHA-256 `678cee66e893dff8cdffcd3285cf9a13ec6bf6dd81c766ca68761a9a73675cd4`; `tufazip-rs` runs executed end-to-end with Apple Accelerate BLAS, inducing `553` vocab symbols / `3,272 B` `.voc` / `7,092` `.pp` tokens, cold-start with no pretrained prior):

| Tool / Preset | Raw Bytes | Payload Bytes | Vocab Bytes | Total Bytes | Payload `bpb` | Total `bpb` | Compress Time | Lossless Check |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **`zstd -19`** | 16,384 | 3,524 | 0 | 3,524 | 1.721 | **1.721** | 0.028 s | SHA-256 verified |
| **`bzip2 -9`** | 16,384 | 3,576 | 0 | 3,576 | 1.746 | **1.746** | 0.013 s | SHA-256 verified |
| **`xz -9`** | 16,384 | 3,608 | 0 | 3,608 | 1.762 | **1.762** | 0.015 s | SHA-256 verified |
| **`gzip -9`** | 16,384 | 3,614 | 0 | 3,614 | 1.765 | **1.765** | 0.009 s | SHA-256 verified |
| **`tufazip-rs` (`medium 4L`, cold-start)** | 16,384 | 3,995 | 3,272 | 7,267 | **1.951** | 3.548 | 6.424 s | SHA-256 verified |
| **`tufazip-rs` (`tiny`, cold-start)** | 16,384 | 4,077 | 3,272 | 7,349 | **1.991** | 3.588 | 2.103 s | SHA-256 verified |

### Upstream Reference (`tufazip v2` on `enwik9`, 2&times; NVIDIA B200)

Online Transformer compression does not amortize its fixed `.voc` header (`3,272 B` on `16 KiB`) or warm up Adam weights on short cold-start samples. On full-scale `enwik9` (`1,000,000,000 B`) with a pretrained prior, `Tufalabs/tufazip` achieves:

| Artifact | Raw Bytes | Payload Bytes | Prior + Vocab Overhead | Total Archive Bytes | Payload `bpb` | Total `bpb` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`tufazip v2` (`enwik9-v2.bin`)** | 1,000,000,000 | 100,958,324 | `4,334,741` (`weights.bin`) + `34,421` (`enwik9.voc`) | 105,327,486 | **0.808** | **0.843** |
