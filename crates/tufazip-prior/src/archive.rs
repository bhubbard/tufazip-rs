use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use half::f16;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::PriorError;

/// Version of the `QPI1`/`QPR1`/`LRA1`/`QLA1` binary archive format.
pub const ARCHIVE_VERSION: u32 = 1;

/// Four-byte archive magic identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArchiveMagic {
    /// Quantized prior initialization archive (`QPI1`).
    Qpi1,
    /// Quantized prior runtime archive (`QPR1`).
    Qpr1,
    /// Low-rank adapter archive (`LRA1`).
    Lra1,
    /// Quantized low-rank adapter archive (`QLA1`).
    Qla1,
}

impl ArchiveMagic {
    /// Parses a 4-byte magic prefix.
    ///
    /// # Errors
    ///
    /// Returns [`PriorError::UnknownMagic`] if `bytes` is not recognized.
    pub fn from_bytes(bytes: [u8; 4]) -> Result<Self, PriorError> {
        match &bytes {
            b"QPI1" => Ok(Self::Qpi1),
            b"QPR1" => Ok(Self::Qpr1),
            b"LRA1" => Ok(Self::Lra1),
            b"QLA1" => Ok(Self::Qla1),
            _ => Err(PriorError::UnknownMagic(bytes)),
        }
    }

    /// Returns the 4-byte ASCII representation of this magic header.
    #[must_use]
    pub const fn as_bytes(self) -> &'static [u8; 4] {
        match self {
            Self::Qpi1 => b"QPI1",
            Self::Qpr1 => b"QPR1",
            Self::Lra1 => b"LRA1",
            Self::Qla1 => b"QLA1",
        }
    }

    /// Returns the ASCII string representation (e.g. `"QPI1"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Qpi1 => "QPI1",
            Self::Qpr1 => "QPR1",
            Self::Lra1 => "LRA1",
            Self::Qla1 => "QLA1",
        }
    }
}

/// Supported per-tensor quantization bit widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum BitWidth {
    /// 1-bit binary sign quantization (`{-1, +1} * row_scale`).
    B1 = 1,
    /// 2-bit symmetric quantization (`code - 1`).
    B2 = 2,
    /// 4-bit symmetric quantization (`code - 7`).
    B4 = 4,
    /// 8-bit symmetric quantization (`code - 127`).
    B8 = 8,
    /// Unscaled IEEE 754 float16 storage.
    B16 = 16,
}

impl BitWidth {
    /// Converts a raw integer bit width to [`BitWidth`].
    ///
    /// # Errors
    ///
    /// Returns [`PriorError::UnsupportedBitWidth`] for unsupported values.
    pub fn from_u8(name: &str, bits: u8) -> Result<Self, PriorError> {
        match bits {
            1 => Ok(Self::B1),
            2 => Ok(Self::B2),
            4 => Ok(Self::B4),
            8 => Ok(Self::B8),
            16 => Ok(Self::B16),
            _ => Err(PriorError::UnsupportedBitWidth {
                name: name.to_owned(),
                bits,
            }),
        }
    }

    /// Returns the numeric bit width (`1`, `2`, `4`, `8`, or `16`).
    #[must_use]
    pub const fn bits(self) -> u8 {
        self as u8
    }
}

/// Prior initialization expansion strategy recorded in `QPI1` metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InitializationKind {
    /// Evenly spaces source blocks into a deeper model, zeroing inserted residual outputs.
    DepthGrowth,
    /// Unties a recurrent loop schedule into distinct physical blocks.
    LoopUntying,
}

/// Architectural geometry used to verify source/target compatibility for `QPI1` priors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformerGeometry {
    /// Vocabulary token count.
    pub vocab_size: usize,
    /// Number of physical transformer blocks.
    pub n_layer: usize,
    /// Hidden state width.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_head: usize,
    /// SwiGLU intermediate dimension.
    pub d_inner: usize,
    /// Maximum context window length.
    pub ctx_len: usize,
    /// Whether input and output embeddings share weights.
    pub tie_embeddings: bool,
    /// Attention positional mechanism (`"rope"` or `"learned_relative"`).
    pub position_type: String,
    /// Whether per-head QK RMSNorm is active.
    pub qk_norm: bool,
    /// Block normalization style (`"peri"` or `"post"`).
    pub norm_style: String,
}

impl TransformerGeometry {
    /// Compares all fields except `n_layer`, returning the first mismatched field name.
    #[must_use]
    pub fn mismatch_excluding_n_layer(&self, other: &Self) -> Option<&'static str> {
        if self.vocab_size != other.vocab_size {
            Some("vocab_size")
        } else if self.d_model != other.d_model {
            Some("d_model")
        } else if self.n_head != other.n_head {
            Some("n_head")
        } else if self.d_inner != other.d_inner {
            Some("d_inner")
        } else if self.ctx_len != other.ctx_len {
            Some("ctx_len")
        } else if self.tie_embeddings != other.tie_embeddings {
            Some("tie_embeddings")
        } else if self.position_type != other.position_type {
            Some("position_type")
        } else if self.qk_norm != other.qk_norm {
            Some("qk_norm")
        } else if self.norm_style != other.norm_style {
            Some("norm_style")
        } else {
            None
        }
    }
}

/// Initialization metadata stored inside `QPI1` archives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InitializationMeta {
    /// Expansion strategy (`"depth-growth"` or `"loop-untying"`).
    pub kind: InitializationKind,
    /// Geometry of the source model weights stored in the archive.
    pub source_geometry: TransformerGeometry,
    /// Target model geometry the archive was calibrated for.
    pub target_geometry: TransformerGeometry,
    /// Canonical block mapping (for `depth-growth`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_mapping: Option<Vec<usize>>,
    /// Recurrent loop block schedule (for `loop-untying`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_schedule: Option<Vec<usize>>,
    /// Deterministic initialization seed pinned by the prior.
    pub initialization_seed: u64,
    /// SHA-256 hex digest of the vocabulary file pinned by the prior.
    pub vocab_sha256: String,
    /// Optional SHA-256 digest of the source training checkpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_checkpoint_sha256: Option<String>,
}

/// Metadata header for a single tensor stored in an archive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorEntryMeta {
    /// Canonical parameter name.
    pub name: String,
    /// Tensor dimensions.
    pub shape: Vec<usize>,
    /// Quantization bit width (`1`, `2`, `4`, `8`, or `16`).
    pub bits: u8,
    /// Number of float16 row/tensor scales (`0` when `bits == 16`).
    pub scale_count: usize,
    /// Byte length of the serialized scale section (`scale_count * 2`).
    pub scale_bytes: usize,
    /// Scale storage dtype (typically `"float16"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale_dtype: Option<String>,
    /// Byte length of the packed quantized values.
    pub value_bytes: usize,
}

/// Top-level JSON metadata header inside a tensor archive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchiveMetadata {
    /// Archive format version (must be `1`).
    pub version: u32,
    /// Optional quantization policy label (e.g. `"uniform1"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy: Option<String>,
    /// Optional depth-growth or loop-untying initialization metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initialization: Option<InitializationMeta>,
    /// Ordered list of packed tensor payloads following the JSON header.
    pub entries: Vec<TensorEntryMeta>,
}

/// Dequantized float32 tensor decoded from an archive.
#[derive(Debug, Clone, PartialEq)]
pub struct DequantizedTensor {
    /// Tensor shape dimensions.
    pub shape: Vec<usize>,
    /// Row-major dequantized `f32` elements.
    pub data: Vec<f32>,
}

/// Complete decoded quantized prior archive.
#[derive(Debug, Clone, PartialEq)]
pub struct PriorArchive {
    /// Four-byte magic kind.
    pub magic: ArchiveMagic,
    /// Total file size on disk in bytes.
    pub archive_bytes: u64,
    /// SHA-256 hex digest of the raw archive file.
    pub archive_sha256: String,
    /// Parsed archive JSON metadata.
    pub metadata: ArchiveMetadata,
    /// Dequantized `f32` tensors keyed by parameter name.
    pub tensors: BTreeMap<String, DequantizedTensor>,
}

/// Validated plan for applying a `QPI1` prior to a target `Transformer`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PriorPlan {
    /// Size of the archive file in bytes.
    pub archive_bytes: u64,
    /// SHA-256 hex digest of the archive file.
    pub archive_sha256: String,
    /// SHA-256 hex digest of the pinned vocabulary file.
    pub vocab_sha256: String,
    /// Initialization strategy used by the prior.
    pub initialization_kind: InitializationKind,
    /// Number of physical blocks stored in the archive.
    pub source_depth: usize,
    /// Effective block mapping (`Vec<usize>`) recorded for diagnostics.
    pub block_mapping: Vec<usize>,
    /// Recurrent block schedule if `loop-untying`.
    pub block_schedule: Option<Vec<usize>>,
    /// Map from target block index to archive source block index.
    pub target_sources: BTreeMap<usize, usize>,
    /// Root (non-block) parameter names verified present in both prior and target.
    pub root_names: Vec<String>,
    /// Per-block suffix names (after `"blocks.{i}."`).
    pub block_suffixes: Vec<String>,
    /// Unmapped target block indices whose residual output projections must be zeroed (`depth-growth`).
    pub zero_unmapped_targets: Vec<usize>,
    /// Optional quantization policy name.
    pub policy: Option<String>,
}

/// Computes the canonical byte size for packing `num_values` elements at `bits`.
#[must_use]
pub fn packed_byte_len(num_values: usize, bits: BitWidth) -> usize {
    let b = usize::from(bits.bits());
    if b == 16 {
        num_values * 2
    } else {
        (num_values * b).div_ceil(8)
    }
}

fn read_exact_or_truncated<R: Read>(
    reader: &mut R,
    buf: &mut [u8],
    what: &str,
) -> Result<(), PriorError> {
    let mut read = 0usize;
    while read < buf.len() {
        match reader.read(&mut buf[read..]) {
            Ok(0) => return Err(PriorError::Truncated(what.to_owned())),
            Ok(n) => read += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

fn decode_f16_le_slice(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|pair| f16::from_le_bytes([pair[0], pair[1]]).to_f32())
        .collect()
}

fn unpack_and_dequantize(
    value_payload: &[u8],
    scales: &[f32],
    count: usize,
    bits: BitWidth,
) -> Vec<f32> {
    let bits_u8 = bits.bits();
    let mut out = vec![0.0f32; count];
    let scale_count = scales.len().max(1);
    let row_stride = count / scale_count;

    if bits == BitWidth::B8 {
        for (idx, (&raw, dst)) in value_payload
            .iter()
            .take(count)
            .zip(out.iter_mut())
            .enumerate()
        {
            let signed = i32::from(raw) - 127;
            let scale = scales[idx / row_stride];
            *dst = (signed as f32) * scale;
        }
        return out;
    }

    let per_byte = 8usize / usize::from(bits_u8);
    let mask = (1u8 << bits_u8) - 1;
    let offset = if bits == BitWidth::B1 {
        0i32
    } else {
        (1i32 << (bits_u8 - 1)) - 1
    };

    for (idx, dst) in out.iter_mut().enumerate() {
        let byte = value_payload[idx / per_byte];
        let shift = (idx % per_byte) * usize::from(bits_u8);
        let code = (byte >> shift) & mask;
        let signed = if bits == BitWidth::B1 {
            i32::from(code) * 2 - 1
        } else {
            i32::from(code) - offset
        };
        *dst = (signed as f32) * scales[idx / row_stride];
    }

    out
}

/// Computes the lowercase hex SHA-256 digest of a file on disk.
///
/// # Errors
///
/// Returns [`PriorError::Io`] if opening or reading `path` fails.
pub fn sha256_file(path: impl AsRef<Path>) -> Result<String, PriorError> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Reads and dequantizes a `QPI1`/`QPR1`/`LRA1`/`QLA1` archive into `f32` tensors.
///
/// # Errors
///
/// Returns [`PriorError`] if the file cannot be read, the magic/version/metadata is invalid,
/// any payload is truncated, or trailing bytes remain.
pub fn read_archive(path: impl AsRef<Path>) -> Result<PriorArchive, PriorError> {
    let path = path.as_ref();
    let archive_bytes = std::fs::metadata(path)?.len();
    let archive_sha256 = sha256_file(path)?;

    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);

    let mut magic_buf = [0u8; 4];
    read_exact_or_truncated(&mut reader, &mut magic_buf, "archive magic")?;
    let magic = ArchiveMagic::from_bytes(magic_buf)?;

    let mut len_buf = [0u8; 8];
    read_exact_or_truncated(&mut reader, &mut len_buf, "metadata length")?;
    let metadata_len = u64::from_le_bytes(len_buf);
    if metadata_len > 64 * 1024 * 1024 {
        return Err(PriorError::InvalidMetadata(format!(
            "metadata length {metadata_len} exceeds 64 MiB safety limit"
        )));
    }
    let mut metadata_payload = vec![0u8; metadata_len as usize];
    read_exact_or_truncated(&mut reader, &mut metadata_payload, "archive metadata")?;
    let metadata: ArchiveMetadata = serde_json::from_slice(&metadata_payload)?;
    if metadata.version != ARCHIVE_VERSION {
        return Err(PriorError::UnsupportedVersion(metadata.version));
    }

    let mut tensors = BTreeMap::new();
    for entry in &metadata.entries {
        if entry.name.is_empty() {
            return Err(PriorError::InvalidMetadata(
                "archive tensor name must be a non-empty string".to_owned(),
            ));
        }
        if tensors.contains_key(&entry.name) {
            return Err(PriorError::DuplicateTensor(entry.name.clone()));
        }
        if entry.shape.is_empty() || entry.shape.contains(&0) {
            return Err(PriorError::InvalidShape {
                name: entry.name.clone(),
                shape: entry.shape.clone(),
            });
        }
        let count = entry.shape.iter().try_fold(1usize, |acc, &d| {
            acc.checked_mul(d).ok_or_else(|| PriorError::InvalidShape {
                name: entry.name.clone(),
                shape: entry.shape.clone(),
            })
        })?;
        let bits = BitWidth::from_u8(&entry.name, entry.bits)?;
        let expected_value_bytes = packed_byte_len(count, bits);
        if entry.value_bytes != expected_value_bytes {
            return Err(PriorError::SizeMismatch {
                name: entry.name.clone(),
                detail: format!(
                    "expected value_bytes={expected_value_bytes}, found {}",
                    entry.value_bytes
                ),
            });
        }

        let expected_scale_count = if bits == BitWidth::B16 {
            0
        } else if entry.shape.len() >= 2 {
            entry.shape[0]
        } else {
            1
        };
        if entry.scale_count != expected_scale_count {
            return Err(PriorError::SizeMismatch {
                name: entry.name.clone(),
                detail: format!(
                    "expected scale_count={expected_scale_count}, found {}",
                    entry.scale_count
                ),
            });
        }
        if entry.scale_bytes != expected_scale_count * 2 {
            return Err(PriorError::SizeMismatch {
                name: entry.name.clone(),
                detail: format!(
                    "expected scale_bytes={}, found {}",
                    expected_scale_count * 2,
                    entry.scale_bytes
                ),
            });
        }

        let data = if bits == BitWidth::B16 {
            let mut payload = vec![0u8; entry.value_bytes];
            read_exact_or_truncated(&mut reader, &mut payload, &entry.name)?;
            decode_f16_le_slice(&payload)
        } else {
            let mut scale_payload = vec![0u8; entry.scale_bytes];
            read_exact_or_truncated(
                &mut reader,
                &mut scale_payload,
                &format!("scales for {}", entry.name),
            )?;
            let scales = decode_f16_le_slice(&scale_payload);

            let mut value_payload = vec![0u8; entry.value_bytes];
            read_exact_or_truncated(
                &mut reader,
                &mut value_payload,
                &format!("values for {}", entry.name),
            )?;
            unpack_and_dequantize(&value_payload, &scales, count, bits)
        };

        tensors.insert(
            entry.name.clone(),
            DequantizedTensor {
                shape: entry.shape.clone(),
                data,
            },
        );
    }

    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        return Err(PriorError::TrailingBytes);
    }

    Ok(PriorArchive {
        magic,
        archive_bytes,
        archive_sha256,
        metadata,
        tensors,
    })
}

/// Writes quantized tensors into a `QPI1`/`QPR1`/`LRA1`/`QLA1` binary archive.
///
/// Used primarily for testing and creating quantized prior checkpoints.
///
/// # Errors
///
/// Returns [`PriorError`] on invalid shapes/counts or I/O failures.
pub fn write_archive(
    path: impl AsRef<Path>,
    magic: ArchiveMagic,
    policy: Option<String>,
    initialization: Option<InitializationMeta>,
    tensors: &[(&str, &[usize], &[f32], BitWidth)],
) -> Result<(), PriorError> {
    let mut entries = Vec::with_capacity(tensors.len());
    let mut payloads: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(tensors.len());
    let mut seen = BTreeSet::new();

    for &(name, shape, data, bits) in tensors {
        if name.is_empty() {
            return Err(PriorError::InvalidMetadata(
                "archive tensor name must be non-empty".to_owned(),
            ));
        }
        if !seen.insert(name) {
            return Err(PriorError::DuplicateTensor(name.to_owned()));
        }
        if shape.is_empty() || shape.contains(&0) {
            return Err(PriorError::InvalidShape {
                name: name.to_owned(),
                shape: shape.to_vec(),
            });
        }
        let count: usize = shape.iter().product();
        if data.len() != count {
            return Err(PriorError::SizeMismatch {
                name: name.to_owned(),
                detail: format!("data length {} != product of shape {count}", data.len()),
            });
        }
        let value_bytes = packed_byte_len(count, bits);
        let scale_count = if bits == BitWidth::B16 {
            0
        } else if shape.len() >= 2 {
            shape[0]
        } else {
            1
        };
        let scale_bytes = scale_count * 2;
        let scale_dtype = if bits == BitWidth::B16 {
            None
        } else {
            Some("float16".to_owned())
        };

        entries.push(TensorEntryMeta {
            name: name.to_owned(),
            shape: shape.to_vec(),
            bits: bits.bits(),
            scale_count,
            scale_bytes,
            scale_dtype,
            value_bytes,
        });

        if bits == BitWidth::B16 {
            let mut val_bytes = Vec::with_capacity(value_bytes);
            for &v in data {
                val_bytes.extend_from_slice(&f16::from_f32(v).to_le_bytes());
            }
            payloads.push((Vec::new(), val_bytes));
        } else {
            let row_stride = count / scale_count;
            let mut scale_buf = Vec::with_capacity(scale_bytes);
            let mut scales_f32 = Vec::with_capacity(scale_count);
            let max_code_mag: i32 = match bits {
                BitWidth::B1 => 1,
                BitWidth::B2 => 1,
                BitWidth::B4 => 7,
                BitWidth::B8 => 127,
                BitWidth::B16 => unreachable!(),
            };

            for row in 0..scale_count {
                let row_slice = &data[row * row_stride..(row + 1) * row_stride];
                let max_abs = row_slice
                    .iter()
                    .copied()
                    .map(f32::abs)
                    .fold(0.0f32, f32::max);
                let raw_scale = if max_abs == 0.0 {
                    1.0f32
                } else {
                    max_abs / (max_code_mag as f32)
                };
                let scale_f16 = f16::from_f32(raw_scale);
                scale_buf.extend_from_slice(&scale_f16.to_le_bytes());
                scales_f32.push(scale_f16.to_f32());
            }

            let bits_u8 = bits.bits();
            let per_byte = 8usize / usize::from(bits_u8);
            let mut val_buf = vec![0u8; value_bytes];
            for (idx, &v) in data.iter().enumerate() {
                let scale = scales_f32[idx / row_stride];
                let code: u8 = if bits == BitWidth::B1 {
                    if v >= 0.0 {
                        1
                    } else {
                        0
                    }
                } else {
                    let q = if scale == 0.0 {
                        0
                    } else {
                        (v / scale).round_ties_even() as i32
                    };
                    let clamped = q.clamp(-max_code_mag, max_code_mag);
                    (clamped + max_code_mag) as u8
                };
                let byte_idx = idx / per_byte;
                let shift = (idx % per_byte) * usize::from(bits_u8);
                val_buf[byte_idx] |= code << shift;
            }
            payloads.push((scale_buf, val_buf));
        }
    }

    let meta = ArchiveMetadata {
        version: ARCHIVE_VERSION,
        policy,
        initialization,
        entries,
    };
    let meta_json = serde_json::to_vec(&meta)?;

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(magic.as_bytes())?;
    writer.write_all(&(meta_json.len() as u64).to_le_bytes())?;
    writer.write_all(&meta_json)?;
    for (scales, values) in payloads {
        writer.write_all(&scales)?;
        writer.write_all(&values)?;
    }
    writer.flush()?;
    Ok(())
}

/// Computes the canonical even placement of `source_depth` blocks inside `target_depth`.
///
/// # Errors
///
/// Returns [`PriorError::Validation`] if `source_depth == 0` or `source_depth > target_depth`.
pub fn depth_block_mapping(
    source_depth: usize,
    target_depth: usize,
) -> Result<Vec<usize>, PriorError> {
    if source_depth < 1 || source_depth > target_depth {
        return Err(PriorError::Validation(format!(
            "source_depth ({source_depth}) must be between 1 and target_depth ({target_depth})"
        )));
    }
    Ok((0..source_depth)
        .map(|i| i * target_depth / source_depth)
        .collect())
}

/// Parses a comma-separated recurrent-depth block schedule string.
///
/// # Errors
///
/// Returns [`PriorError::Validation`] if the schedule is empty, contains non-integers,
/// exceeds `n_layer`, or fails to use every physical block `0..n_layer`.
pub fn parse_block_schedule(spec: &str, n_layer: usize) -> Result<Vec<usize>, PriorError> {
    if spec.trim().is_empty() {
        return Err(PriorError::Validation(
            "block_schedule cannot be empty".to_owned(),
        ));
    }
    let schedule: Vec<usize> = spec
        .split(',')
        .map(|part| {
            part.trim().parse::<usize>().map_err(|_| {
                PriorError::Validation(format!(
                    "block_schedule must contain comma-separated non-negative integers, got {spec:?}"
                ))
            })
        })
        .collect::<Result<_, _>>()?;
    validate_block_schedule(&schedule, n_layer)?;
    Ok(schedule)
}

/// Validates that a block schedule covers every physical layer index `0..n_layer`.
///
/// # Errors
///
/// Returns [`PriorError::Validation`] if `schedule` is empty, out of range, or omits a block.
pub fn validate_block_schedule(schedule: &[usize], n_layer: usize) -> Result<(), PriorError> {
    if schedule.is_empty() {
        return Err(PriorError::Validation(
            "block_schedule cannot be empty".to_owned(),
        ));
    }
    let mut seen = vec![false; n_layer];
    for &idx in schedule {
        if idx >= n_layer {
            return Err(PriorError::Validation(format!(
                "block_schedule index {idx} is outside n_layer ({n_layer})"
            )));
        }
        seen[idx] = true;
    }
    if seen.iter().any(|&present| !present) {
        return Err(PriorError::Validation(
            "block_schedule must use every physical block".to_owned(),
        ));
    }
    Ok(())
}

/// Validates a loaded `QPI1` prior archive against the target model geometry, vocabulary, and seed.
///
/// Mirrors `load_depth_prior` validation in `nn.py` without loading tensors into an ML backend.
///
/// # Errors
///
/// Returns [`PriorError`] if any magic, geometry, vocab SHA-256, seed, tensor set, or shape
/// verification fails.
pub fn plan_depth_prior(
    archive: &PriorArchive,
    vocab_path: impl AsRef<Path>,
    expected_geometry: &TransformerGeometry,
    seed: u64,
    expected_target_shapes: &BTreeMap<String, Vec<usize>>,
) -> Result<PriorPlan, PriorError> {
    if archive.magic != ArchiveMagic::Qpi1 {
        return Err(PriorError::Validation(
            "initial prior requires a QPI1 archive".to_owned(),
        ));
    }
    let init = archive.metadata.initialization.as_ref().ok_or_else(|| {
        PriorError::Validation("QPI1 archive lacks initialization metadata".to_owned())
    })?;
    if &init.target_geometry != expected_geometry {
        return Err(PriorError::Validation(
            "prior target geometry differs from invocation geometry".to_owned(),
        ));
    }
    if let Some(field) = init
        .source_geometry
        .mismatch_excluding_n_layer(expected_geometry)
    {
        return Err(PriorError::Validation(format!(
            "prior source geometry differs at {field}"
        )));
    }
    let source_depth = init.source_geometry.n_layer;
    let target_depth = expected_geometry.n_layer;

    let (block_mapping, block_schedule, target_sources) = match init.kind {
        InitializationKind::DepthGrowth => {
            let mapping = depth_block_mapping(source_depth, target_depth)?;
            if init.block_mapping.as_deref() != Some(mapping.as_slice()) {
                return Err(PriorError::Validation(
                    "prior block mapping is not the canonical depth mapping".to_owned(),
                ));
            }
            let target_sources = mapping
                .iter()
                .enumerate()
                .map(|(source_idx, &target_idx)| (target_idx, source_idx))
                .collect::<BTreeMap<_, _>>();
            (mapping, None, target_sources)
        }
        InitializationKind::LoopUntying => {
            let schedule = init.block_schedule.as_ref().ok_or_else(|| {
                PriorError::Validation("loop prior lacks a block_schedule".to_owned())
            })?;
            validate_block_schedule(schedule, source_depth)?;
            if schedule.len() != target_depth {
                return Err(PriorError::Validation(format!(
                    "loop prior schedule length ({}) differs from target depth ({target_depth})",
                    schedule.len()
                )));
            }
            let target_sources = schedule
                .iter()
                .enumerate()
                .map(|(target_idx, &source_idx)| (target_idx, source_idx))
                .collect::<BTreeMap<_, _>>();
            (schedule.clone(), Some(schedule.clone()), target_sources)
        }
    };

    if init.initialization_seed != seed {
        return Err(PriorError::Validation(format!(
            "prior initialization seed ({}) differs from configured seed ({seed})",
            init.initialization_seed
        )));
    }
    let vocab_sha256 = sha256_file(vocab_path)?;
    if init.vocab_sha256 != vocab_sha256 {
        return Err(PriorError::Validation(format!(
            "prior vocabulary digest ({}) differs from vocab_file ({vocab_sha256})",
            init.vocab_sha256
        )));
    }

    let mut root_names = Vec::new();
    let mut block_suffixes = Vec::new();
    for name in expected_target_shapes.keys() {
        if !name.starts_with("blocks.") {
            root_names.push(name.clone());
        } else if let Some(suffix) = name.strip_prefix("blocks.0.") {
            block_suffixes.push(suffix.to_owned());
        }
    }

    let mut expected_source_names = BTreeSet::new();
    for name in &root_names {
        expected_source_names.insert(name.clone());
    }
    for source_idx in 0..source_depth {
        for suffix in &block_suffixes {
            expected_source_names.insert(format!("blocks.{source_idx}.{suffix}"));
        }
    }

    let actual_source_names: BTreeSet<String> = archive.tensors.keys().cloned().collect();
    if actual_source_names != expected_source_names {
        let missing: Vec<_> = expected_source_names
            .difference(&actual_source_names)
            .cloned()
            .collect();
        let extra: Vec<_> = actual_source_names
            .difference(&expected_source_names)
            .cloned()
            .collect();
        return Err(PriorError::Validation(format!(
            "prior tensor set mismatch: missing={missing:?} extra={extra:?}"
        )));
    }

    for name in &root_names {
        let src = &archive.tensors[name];
        let dst_shape = &expected_target_shapes[name];
        if &src.shape != dst_shape {
            return Err(PriorError::Validation(format!(
                "prior tensor shape differs for {name}: {:?} vs {:?}",
                src.shape, dst_shape
            )));
        }
    }
    for (&target_idx, &source_idx) in &target_sources {
        for suffix in &block_suffixes {
            let source_name = format!("blocks.{source_idx}.{suffix}");
            let target_name = format!("blocks.{target_idx}.{suffix}");
            let src = &archive.tensors[&source_name];
            let dst_shape = &expected_target_shapes[&target_name];
            if &src.shape != dst_shape {
                return Err(PriorError::Validation(format!(
                    "prior tensor shape differs for {source_name}: {:?} vs {:?}",
                    src.shape, dst_shape
                )));
            }
        }
    }

    let zero_unmapped_targets = if init.kind == InitializationKind::DepthGrowth {
        (0..target_depth)
            .filter(|idx| !target_sources.contains_key(idx))
            .collect()
    } else {
        Vec::new()
    };

    Ok(PriorPlan {
        archive_bytes: archive.archive_bytes,
        archive_sha256: archive.archive_sha256.clone(),
        vocab_sha256,
        initialization_kind: init.kind,
        source_depth,
        block_mapping,
        block_schedule,
        target_sources,
        root_names,
        block_suffixes,
        zero_unmapped_targets,
        policy: archive.metadata.policy.clone(),
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn pack_unpack_roundtrip_all_bit_widths() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let vec_1d: Vec<f32> = vec![-2.0, -0.5, 0.0, 0.75, 2.0, 1.25];
        let mat_2d_b1: Vec<f32> = vec![-0.8, 0.8, -0.4, 0.4, 1.5, -1.5, 1.2, -1.2];
        let mat_2d_b2: Vec<f32> = vec![-1.0, 0.0, 1.0, -0.9, 0.0, 0.9];
        let mat_2d_b4: Vec<f32> = (-7..=7).map(|x| x as f32 * 0.25).chain([0.0]).collect();
        let mat_2d_b8: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();

        write_archive(
            tmp.path(),
            ArchiveMagic::Qpr1,
            Some("test".to_owned()),
            None,
            &[
                ("t16", &[6], &vec_1d, BitWidth::B16),
                ("t1", &[2, 4], &mat_2d_b1, BitWidth::B1),
                ("t2", &[2, 3], &mat_2d_b2, BitWidth::B2),
                ("t4", &[2, 8], &mat_2d_b4, BitWidth::B4),
                ("t8", &[2, 8], &mat_2d_b8, BitWidth::B8),
            ],
        )
        .unwrap();

        let archive = read_archive(tmp.path()).unwrap();
        assert_eq!(archive.magic, ArchiveMagic::Qpr1);
        assert_eq!(archive.tensors.len(), 5);

        // Verify 16-bit precision
        for (got, want) in archive.tensors["t16"].data.iter().zip(&vec_1d) {
            assert!((got - want).abs() < 1e-2);
        }
        // Verify 1-bit sign reconstruction
        let t1 = &archive.tensors["t1"].data;
        assert!(t1[0] < 0.0 && t1[1] > 0.0 && (t1[0].abs() - 0.8).abs() < 0.02);
        assert!(t1[4] > 0.0 && t1[5] < 0.0 && (t1[4].abs() - 1.5).abs() < 0.02);
        // Verify 8-bit reconstruction accuracy
        for (got, want) in archive.tensors["t8"].data.iter().zip(&mat_2d_b8) {
            assert!((got - want).abs() < 0.02, "got={got} want={want}");
        }
    }

    #[test]
    fn real_weights_bin_and_enwik9_voc_validate() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets");
        let weights_path = root.join("weights.bin");
        let voc_path = root.join("enwik9.voc");

        let archive = read_archive(&weights_path).expect("assets/weights.bin should load");
        assert_eq!(archive.magic, ArchiveMagic::Qpi1);
        assert_eq!(archive.archive_bytes, 4_334_741);
        assert_eq!(
            archive.archive_sha256,
            "e94c089f8eb985a87307dca3a7702385fcaccd14d95743811f8788f7da24448d"
        );
        assert_eq!(archive.metadata.policy.as_deref(), Some("uniform1"));
        assert_eq!(archive.tensors.len(), 26);

        for (name, tensor) in &archive.tensors {
            assert!(
                tensor.data.iter().all(|v| v.is_finite()),
                "non-finite value in {name}"
            );
        }

        let mut expected_shapes = BTreeMap::new();
        expected_shapes.insert("tok_emb.weight".to_owned(), vec![4097, 1024]);
        expected_shapes.insert("emb_norm.weight".to_owned(), vec![1024]);
        expected_shapes.insert("norm.weight".to_owned(), vec![1024]);
        expected_shapes.insert("head.weight".to_owned(), vec![4097, 1024]);
        for layer in 0..12 {
            let p = format!("blocks.{layer}");
            expected_shapes.insert(format!("{p}.attn_pre_norm.weight"), vec![1024]);
            expected_shapes.insert(format!("{p}.attn.qkv.weight"), vec![3072, 1024]);
            expected_shapes.insert(format!("{p}.attn.q_norm.weight"), vec![128]);
            expected_shapes.insert(format!("{p}.attn.k_norm.weight"), vec![128]);
            expected_shapes.insert(format!("{p}.attn.proj.weight"), vec![1024, 1024]);
            expected_shapes.insert(format!("{p}.attn_post_norm.weight"), vec![1024]);
            expected_shapes.insert(format!("{p}.ffn_pre_norm.weight"), vec![1024]);
            expected_shapes.insert(format!("{p}.ffn.gate.weight"), vec![2816, 1024]);
            expected_shapes.insert(format!("{p}.ffn.up.weight"), vec![2816, 1024]);
            expected_shapes.insert(format!("{p}.ffn.down.weight"), vec![1024, 2816]);
            expected_shapes.insert(format!("{p}.ffn_post_norm.weight"), vec![1024]);
        }

        let geom = TransformerGeometry {
            vocab_size: 4097,
            n_layer: 12,
            d_model: 1024,
            n_head: 8,
            d_inner: 2816,
            ctx_len: 192,
            tie_embeddings: false,
            position_type: "rope".to_owned(),
            qk_norm: true,
            norm_style: "peri".to_owned(),
        };

        let plan = plan_depth_prior(&archive, &voc_path, &geom, 1111, &expected_shapes).unwrap();
        assert_eq!(plan.initialization_kind, InitializationKind::LoopUntying);
        assert_eq!(plan.source_depth, 2);
        assert_eq!(plan.block_mapping, vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
        assert!(plan.zero_unmapped_targets.is_empty());
    }

    #[test]
    fn depth_mapping_and_schedule_helpers() {
        assert_eq!(depth_block_mapping(2, 6).unwrap(), vec![0, 3]);
        assert_eq!(depth_block_mapping(3, 6).unwrap(), vec![0, 2, 4]);
        assert!(depth_block_mapping(0, 6).is_err());
        assert!(depth_block_mapping(7, 6).is_err());

        assert_eq!(
            parse_block_schedule("0,1,0,1", 2).unwrap(),
            vec![0, 1, 0, 1]
        );
        assert!(parse_block_schedule("", 2).is_err());
        assert!(parse_block_schedule("0,0", 2).is_err());
        assert!(parse_block_schedule("0,2", 2).is_err());
    }
}
