//! Neural Transformer compression and decompression codec for `tufazip` backed by Candle.
//!
//! Ported from `nn.py` and `checkpoint_utils.py`: includes the Peri-RMSNorm / Post-RMSNorm
//! SwiGLU Transformer with RoPE or learned-relative attention, online Adam updates,
//! periodic prefix retraining, `QPI1` prior loading, and atomic checkpoint resume.

mod checkpoint;
mod config;
mod engine;
mod error;
mod model;
mod optim;
mod rng;

pub use checkpoint::{
    clear_stop, is_done, load_checkpoint, mark_done, request_stop, save_checkpoint, stop_requested,
    CheckpointMetadata, LoadedCheckpoint, CHECKPOINT_VERSION, DONE_NAME, REQUEUE_EXIT_CODE,
    STOP_NAME,
};
pub use config::{
    lr_at, resolve_device, AutocastPolicy, CodecConfig, CodecMode, ComputeDType, DeviceChoice,
    ModelConfig, NormStyle, OnlineLoss, OptimConfig, PiecewiseLrSchedule, PositionType,
    RetrainConfig,
};
pub use engine::{run_codec, RunOutcome, RunSummary};
pub use error::CodecError;
pub use model::{ForwardMode, OutputSlice, PriorLoadInfo, Transformer};
pub use optim::{clip_grad_norm, Adam};
pub use rng::{DeterministicRng, RngState};
