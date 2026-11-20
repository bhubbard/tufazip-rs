//! Quantized prior weight archive (`QPI1`, `QPR1`, `LRA1`, `QLA1`) reader, writer, and validator.
//!
//! Ported from `read_tensor_archive`, `load_depth_prior`, and helper functions in `nn.py`.

mod archive;
mod error;

pub use archive::{
    depth_block_mapping, parse_block_schedule, plan_depth_prior, read_archive, sha256_file,
    validate_block_schedule, write_archive, ArchiveMagic, ArchiveMetadata, BitWidth,
    DequantizedTensor, InitializationKind, InitializationMeta, PriorArchive, PriorPlan,
    TensorEntryMeta, TransformerGeometry, ARCHIVE_VERSION,
};
pub use error::PriorError;
