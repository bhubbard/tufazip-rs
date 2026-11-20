//! Exhaustive tests for `nn.py` and `checkpoint_utils.py` edge cases, prior modes,
//! recurrent block schedules, LR schedules, early replay, checkpoint integrity, and stream boundaries.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use candle_core::{DType, Device};
use sha2::{Digest, Sha256};
use tufazip_codec::{
    clear_stop, lr_at, request_stop, run_codec, stop_requested, CodecConfig, CodecError, CodecMode,
    ComputeDType, DeterministicRng, DeviceChoice, ModelConfig, NormStyle, OnlineLoss,
    PiecewiseLrSchedule, PositionType, RunOutcome, Transformer,
};
use tufazip_preprocess::{read_tokens_be16, write_tokens_be16};
use tufazip_prior::{
    write_archive, ArchiveMagic, BitWidth, InitializationKind, InitializationMeta,
};

fn build_synthetic_tokens(len: usize, vocab: usize) -> Vec<u16> {
    (0..len)
        .map(|i| ((i * 11 + (i / 7) * 5) % vocab) as u16)
        .collect()
}

fn tiny_cfg(vocab_size: usize) -> CodecConfig {
    let mut cfg = CodecConfig::tiny(vocab_size);
    cfg.model.n_layer = 2;
    cfg.model.d_model = 32;
    cfg.model.n_head = 2;
    cfg.model.d_inner = 64;
    cfg.model.ctx_len = 16;
    cfg.model.dropout = 0.2;
    cfg.batch_size = 4;
    cfg.block_len = 32;
    cfg.retrain.period = 32;
    cfg.retrain.len = 96;
    cfg.retrain.batch_size = 4;
    cfg.retrain.update_len = 8;
    cfg.device = DeviceChoice::Cpu;
    cfg.dtype = ComputeDType::Fp32;
    cfg
}

#[test]
fn depth_growth_from_prior_zeros_unmapped_residuals_and_matches_load_prior() {
    let dir = tempfile::tempdir().unwrap();
    let voc_path = dir.path().join("depth.voc");
    let voc_bytes: Vec<u8> = (0u8..32).flat_map(|i| [b'a' + (i % 26), b'\n']).collect();
    std::fs::write(&voc_path, &voc_bytes).unwrap();
    let voc_sha = format!("{:x}", Sha256::digest(&voc_bytes));

    let src_cfg = ModelConfig {
        vocab_size: 32,
        n_layer: 2,
        d_model: 16,
        n_head: 2,
        d_inner: 32,
        ctx_len: 8,
        dropout: 0.0,
        rope_base: 10_000.0,
        tie_embeddings: false,
        position_type: PositionType::Rope,
        qk_norm: true,
        norm_style: NormStyle::Peri,
        block_schedule: None,
    };
    let mut tgt_cfg = src_cfg.clone();
    tgt_cfg.n_layer = 4;

    let init_meta = InitializationMeta {
        kind: InitializationKind::DepthGrowth,
        source_geometry: src_cfg.geometry(),
        target_geometry: tgt_cfg.geometry(),
        block_mapping: Some(vec![0, 2]),
        block_schedule: None,
        initialization_seed: 777,
        vocab_sha256: voc_sha,
        source_checkpoint_sha256: None,
    };

    let head_dim = src_cfg.d_model / src_cfg.n_head;
    let mut owned_tensors: Vec<(String, Vec<usize>, Vec<f32>, BitWidth)> = vec![
        (
            "tok_emb.weight".to_owned(),
            vec![32, 16],
            vec![0.125; 32 * 16],
            BitWidth::B8,
        ),
        (
            "emb_norm.weight".to_owned(),
            vec![16],
            vec![1.0; 16],
            BitWidth::B16,
        ),
        (
            "norm.weight".to_owned(),
            vec![16],
            vec![1.0; 16],
            BitWidth::B16,
        ),
        (
            "head.weight".to_owned(),
            vec![32, 16],
            vec![-0.125; 32 * 16],
            BitWidth::B8,
        ),
    ];
    for l in 0..2 {
        let p = format!("blocks.{l}");
        let v = (l as f32 + 1.0) * 0.5;
        owned_tensors.push((
            format!("{p}.attn_pre_norm.weight"),
            vec![16],
            vec![1.0; 16],
            BitWidth::B16,
        ));
        owned_tensors.push((
            format!("{p}.attn.qkv.weight"),
            vec![48, 16],
            vec![v; 48 * 16],
            BitWidth::B8,
        ));
        owned_tensors.push((
            format!("{p}.attn.q_norm.weight"),
            vec![head_dim],
            vec![1.0; head_dim],
            BitWidth::B16,
        ));
        owned_tensors.push((
            format!("{p}.attn.k_norm.weight"),
            vec![head_dim],
            vec![1.0; head_dim],
            BitWidth::B16,
        ));
        owned_tensors.push((
            format!("{p}.attn.proj.weight"),
            vec![16, 16],
            vec![v; 16 * 16],
            BitWidth::B8,
        ));
        owned_tensors.push((
            format!("{p}.attn_post_norm.weight"),
            vec![16],
            vec![1.0; 16],
            BitWidth::B16,
        ));
        owned_tensors.push((
            format!("{p}.ffn_pre_norm.weight"),
            vec![16],
            vec![1.0; 16],
            BitWidth::B16,
        ));
        owned_tensors.push((
            format!("{p}.ffn.gate.weight"),
            vec![32, 16],
            vec![v; 32 * 16],
            BitWidth::B8,
        ));
        owned_tensors.push((
            format!("{p}.ffn.up.weight"),
            vec![32, 16],
            vec![v; 32 * 16],
            BitWidth::B8,
        ));
        owned_tensors.push((
            format!("{p}.ffn.down.weight"),
            vec![16, 32],
            vec![v; 16 * 32],
            BitWidth::B8,
        ));
        owned_tensors.push((
            format!("{p}.ffn_post_norm.weight"),
            vec![16],
            vec![1.0; 16],
            BitWidth::B16,
        ));
    }

    let prior_path = dir.path().join("depth_prior.bin");
    let refs: Vec<(&str, &[usize], &[f32], BitWidth)> = owned_tensors
        .iter()
        .map(|(n, s, d, b)| (n.as_str(), s.as_slice(), d.as_slice(), *b))
        .collect();
    write_archive(
        &prior_path,
        ArchiveMagic::Qpi1,
        Some("depth-test".to_owned()),
        Some(init_meta),
        &refs,
    )
    .unwrap();

    let mut rng_a = DeterministicRng::from_seed_u64(777);
    let (model_a, info_a) = Transformer::from_prior(
        &tgt_cfg,
        &prior_path,
        &voc_path,
        777,
        &Device::Cpu,
        &mut rng_a,
    )
    .unwrap();

    let mut rng_b = DeterministicRng::from_seed_u64(777);
    let mut model_b = Transformer::new(&tgt_cfg, &Device::Cpu, &mut rng_b).unwrap();
    let info_b = model_b.load_prior(&prior_path, &voc_path, 777).unwrap();

    assert_eq!(info_a.initial_state_sha256, info_b.initial_state_sha256);

    let vars: std::collections::HashMap<_, _> = model_a
        .named_vars()
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect();
    for unmapped in [1, 3] {
        let proj = vars[format!("blocks.{unmapped}.attn.proj.weight").as_str()]
            .as_tensor()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let down = vars[format!("blocks.{unmapped}.ffn.down.weight").as_str()]
            .as_tensor()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let qkv = vars[format!("blocks.{unmapped}.attn.qkv.weight").as_str()]
            .as_tensor()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(proj.iter().all(|&x| x == 0.0));
        assert!(down.iter().all(|&x| x == 0.0));
        assert!(qkv.iter().any(|&x| x != 0.0));
    }
}

#[test]
fn recurrent_block_schedule_lossless_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let in_pp = dir.path().join("in.pp");
    let comp_bin = dir.path().join("comp.bin");
    let out_pp = dir.path().join("out.pp");

    let tokens = build_synthetic_tokens(72, 48);
    write_tokens_be16(&in_pp, &tokens).unwrap();

    let mut cfg = tiny_cfg(48);
    cfg.model.n_layer = 2;
    cfg.model.block_schedule = Some(vec![0, 1, 0, 1]);

    let mut rng = DeterministicRng::from_seed_u64(1);
    let m = Transformer::new(&cfg.model, &Device::Cpu, &mut rng).unwrap();
    assert_eq!(m.param_counts().0, {
        let mut plain = cfg.model.clone();
        plain.block_schedule = None;
        Transformer::new(&plain, &Device::Cpu, &mut rng)
            .unwrap()
            .param_counts()
            .0
    });

    let stop = Arc::new(AtomicBool::new(false));
    assert!(matches!(
        run_codec(&in_pp, &comp_bin, &cfg, CodecMode::Compress, stop.clone()).unwrap(),
        RunOutcome::Completed(_)
    ));
    assert!(matches!(
        run_codec(&comp_bin, &out_pp, &cfg, CodecMode::Decompress, stop).unwrap(),
        RunOutcome::Completed(_)
    ));
    assert_eq!(read_tokens_be16(&out_pp).unwrap(), tokens);
}

#[test]
fn lr_schedules_and_early_replay_roundtrip() {
    // Cosine warmup schedule checks
    assert!((lr_at(0, 1e-3, 1e-4, 10, 100) - 1e-4).abs() < 1e-12);
    assert!((lr_at(9, 1e-3, 1e-4, 10, 100) - 1e-3).abs() < 1e-12);
    assert!((lr_at(100, 1e-3, 1e-4, 10, 100) - 1e-4).abs() < 1e-12);
    assert!((lr_at(250, 1e-3, 1e-4, 10, 100) - 1e-4).abs() < 1e-12);
    let mid = lr_at(55, 1e-3, 1e-4, 10, 100);
    assert!((mid - 5.5e-4).abs() < 1e-12);

    // Piecewise LR schedule checks including exact V2_CONFIG schedule
    let v2_sched =
        PiecewiseLrSchedule::parse("1.6e-4,13000,1.6e-4,93000,7.9e-5,163000,4.0e-5,1911300,1.3e-5")
            .unwrap();
    assert!((v2_sched.lr_at(0) - 1.6e-4).abs() < 1e-12);
    assert!((v2_sched.lr_at(13_000) - 1.6e-4).abs() < 1e-12);
    assert!((v2_sched.lr_at(53_000) - 1.195e-4).abs() < 1e-12);
    assert!((v2_sched.lr_at(9_999_999) - 1.3e-5).abs() < 1e-12);

    assert!(PiecewiseLrSchedule::parse("").is_err());
    assert!(PiecewiseLrSchedule::parse("1e-4,100").is_err());
    assert!(PiecewiseLrSchedule::parse("1e-4,100,5e-5,100,1e-5").is_err());

    // Early replay roundtrip
    let dir = tempfile::tempdir().unwrap();
    let in_pp = dir.path().join("early.pp");
    let comp_bin = dir.path().join("early.bin");
    let out_pp = dir.path().join("early_out.pp");

    let tokens = build_synthetic_tokens(96, 48);
    write_tokens_be16(&in_pp, &tokens).unwrap();

    let mut cfg = tiny_cfg(48);
    cfg.retrain.early_epochs = 2;
    cfg.retrain.early_until = 64;
    cfg.retrain.epochs = 1;
    cfg.retrain.lr_schedule = PiecewiseLrSchedule::parse("2e-4,8,1e-4,24,5e-5").unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    run_codec(&in_pp, &comp_bin, &cfg, CodecMode::Compress, stop.clone()).unwrap();
    run_codec(&comp_bin, &out_pp, &cfg, CodecMode::Decompress, stop).unwrap();
    assert_eq!(read_tokens_be16(&out_pp).unwrap(), tokens);
}

#[test]
fn learned_relative_post_norm_tied_embeddings_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let in_pp = dir.path().join("rel.pp");
    let comp_bin = dir.path().join("rel.bin");
    let out_pp = dir.path().join("rel_out.pp");

    let tokens = build_synthetic_tokens(68, 40);
    write_tokens_be16(&in_pp, &tokens).unwrap();

    let mut cfg = tiny_cfg(40);
    cfg.model.position_type = PositionType::LearnedRelative;
    cfg.model.norm_style = NormStyle::Post;
    cfg.model.tie_embeddings = true;
    cfg.model.qk_norm = false;

    let stop = Arc::new(AtomicBool::new(false));
    run_codec(&in_pp, &comp_bin, &cfg, CodecMode::Compress, stop.clone()).unwrap();
    run_codec(&comp_bin, &out_pp, &cfg, CodecMode::Decompress, stop).unwrap();
    assert_eq!(read_tokens_be16(&out_pp).unwrap(), tokens);
}

#[test]
fn stop_requested_file_and_atomic_flag_resume_bit_exact() {
    let dir = tempfile::tempdir().unwrap();
    let in_pp = dir.path().join("stop.pp");
    let full_bin = dir.path().join("full.bin");
    let stop_file_bin = dir.path().join("stop_file.bin");
    let flag_bin = dir.path().join("flag.bin");
    let ckpt_a = dir.path().join("ckpt_a");
    let ckpt_b = dir.path().join("ckpt_b");

    let tokens = build_synthetic_tokens(100, 48);
    write_tokens_be16(&in_pp, &tokens).unwrap();

    let base_cfg = tiny_cfg(48);
    let no_stop = Arc::new(AtomicBool::new(false));
    run_codec(
        &in_pp,
        &full_bin,
        &base_cfg,
        CodecMode::Compress,
        no_stop.clone(),
    )
    .unwrap();
    let want = std::fs::read(&full_bin).unwrap();

    // 1. Pre-create STOP_REQUESTED file before block 1 finishes
    let mut cfg_a = base_cfg.clone();
    cfg_a.checkpoint_dir = Some(ckpt_a.clone());
    request_stop(&ckpt_a).unwrap();
    assert!(stop_requested(&ckpt_a));
    let paused = run_codec(
        &in_pp,
        &stop_file_bin,
        &cfg_a,
        CodecMode::Compress,
        no_stop.clone(),
    )
    .unwrap();
    assert!(matches!(paused, RunOutcome::RequeueRequested { .. }));
    let meta_json = std::fs::read_to_string(ckpt_a.join("latest.json")).unwrap();
    assert!(meta_json.contains("\"stop-request\""));

    clear_stop(&ckpt_a).unwrap();
    assert!(!stop_requested(&ckpt_a));
    let resumed = run_codec(
        &in_pp,
        &stop_file_bin,
        &cfg_a,
        CodecMode::Compress,
        no_stop.clone(),
    )
    .unwrap();
    assert!(matches!(resumed, RunOutcome::Completed(_)));
    assert_eq!(std::fs::read(&stop_file_bin).unwrap(), want);

    // 2. AtomicBool stop_flag triggers clean checkpoint stop
    let mut cfg_b = base_cfg;
    cfg_b.checkpoint_dir = Some(ckpt_b);
    let triggered = Arc::new(AtomicBool::new(true));
    let paused_b = run_codec(&in_pp, &flag_bin, &cfg_b, CodecMode::Compress, triggered).unwrap();
    assert!(matches!(paused_b, RunOutcome::RequeueRequested { .. }));
    let resumed_b = run_codec(&in_pp, &flag_bin, &cfg_b, CodecMode::Compress, no_stop).unwrap();
    assert!(matches!(resumed_b, RunOutcome::Completed(_)));
    assert_eq!(std::fs::read(&flag_bin).unwrap(), want);
}

#[test]
fn checkpoint_integrity_and_mismatch_rejections() {
    let dir = tempfile::tempdir().unwrap();
    let in_pp = dir.path().join("ckpt_in.pp");
    let out_bin = dir.path().join("ckpt_out.bin");
    let ckpt = dir.path().join("ckpt");

    let tokens = build_synthetic_tokens(96, 48);
    write_tokens_be16(&in_pp, &tokens).unwrap();

    let mut cfg = tiny_cfg(48);
    cfg.checkpoint_dir = Some(ckpt.clone());
    cfg.stop_after_blocks = Some(1);

    let stop = Arc::new(AtomicBool::new(false));
    let paused = run_codec(&in_pp, &out_bin, &cfg, CodecMode::Compress, stop.clone()).unwrap();
    assert!(matches!(paused, RunOutcome::RequeueRequested { .. }));

    cfg.stop_after_blocks = None;

    // Seed mismatch
    let mut bad_seed = cfg.clone();
    bad_seed.seed = 9999;
    assert!(matches!(
        run_codec(
            &in_pp,
            &out_bin,
            &bad_seed,
            CodecMode::Compress,
            stop.clone()
        ),
        Err(CodecError::CheckpointMismatch(_))
    ));

    // project_only_targets mismatch
    let mut bad_proj = cfg.clone();
    bad_proj.project_only_targets = false;
    assert!(matches!(
        run_codec(
            &in_pp,
            &out_bin,
            &bad_proj,
            CodecMode::Compress,
            stop.clone()
        ),
        Err(CodecError::CheckpointMismatch(_))
    ));

    // early replay mismatch
    let mut bad_early = cfg.clone();
    bad_early.retrain.early_epochs = 2;
    bad_early.retrain.early_until = 64;
    assert!(matches!(
        run_codec(
            &in_pp,
            &out_bin,
            &bad_early,
            CodecMode::Compress,
            stop.clone()
        ),
        Err(CodecError::CheckpointMismatch(_))
    ));

    // Input file size mismatch
    let other_pp = dir.path().join("other.pp");
    write_tokens_be16(&other_pp, &tokens[..80]).unwrap();
    assert!(matches!(
        run_codec(&other_pp, &out_bin, &cfg, CodecMode::Compress, stop.clone()),
        Err(CodecError::CheckpointMismatch(_))
    ));

    // Corrupt latest.safetensors -> SHA-256 mismatch
    let tensors_path = ckpt.join("latest.safetensors");
    let original_safetensors = std::fs::read(&tensors_path).unwrap();
    let mut corrupted = original_safetensors.clone();
    let last_idx = corrupted.len() - 1;
    corrupted[last_idx] ^= 0xFF;
    std::fs::write(&tensors_path, &corrupted).unwrap();
    assert!(matches!(
        run_codec(&in_pp, &out_bin, &cfg, CodecMode::Compress, stop.clone()),
        Err(CodecError::CheckpointMismatch(_))
    ));

    // Missing latest.safetensors
    std::fs::remove_file(&tensors_path).unwrap();
    assert!(matches!(
        run_codec(&in_pp, &out_bin, &cfg, CodecMode::Compress, stop),
        Err(CodecError::CheckpointMismatch(_))
    ));
}

#[test]
fn config_validation_rejections_and_edge_stream_lengths() {
    let mut cfg = tiny_cfg(32);
    cfg.checkpoint_every_blocks = 0;
    assert!(matches!(
        cfg.validate(CodecMode::Compress),
        Err(CodecError::InvalidConfig(_))
    ));

    let mut cfg = tiny_cfg(32);
    cfg.online_loss = OnlineLoss::Window;
    cfg.project_only_targets = true;
    assert!(matches!(
        cfg.validate(CodecMode::Compress),
        Err(CodecError::InvalidConfig(_))
    ));

    let mut cfg = tiny_cfg(32);
    cfg.retrain.epochs = 0;
    assert!(matches!(
        cfg.validate(CodecMode::Compress),
        Err(CodecError::InvalidConfig(_))
    ));

    let mut cfg = tiny_cfg(32);
    cfg.retrain.early_epochs = 2;
    cfg.retrain.early_until = 0;
    assert!(matches!(
        cfg.validate(CodecMode::Compress),
        Err(CodecError::InvalidConfig(_))
    ));

    let mut cfg = tiny_cfg(32);
    cfg.retrain.update_len = cfg.model.ctx_len + 1;
    assert!(matches!(
        cfg.validate(CodecMode::Compress),
        Err(CodecError::InvalidConfig(_))
    ));

    let mut cfg = tiny_cfg(32);
    cfg.max_tokens = Some(10);
    assert!(matches!(
        cfg.validate(CodecMode::Decompress),
        Err(CodecError::InvalidConfig(_))
    ));

    let mut cfg = tiny_cfg(32);
    cfg.model.n_head = 3;
    assert!(matches!(
        cfg.validate(CodecMode::Compress),
        Err(CodecError::InvalidConfig(_))
    ));

    let mut cfg = tiny_cfg(32);
    cfg.model.dropout = 1.0;
    assert!(matches!(
        cfg.validate(CodecMode::Compress),
        Err(CodecError::InvalidConfig(_))
    ));

    let dir = tempfile::tempdir().unwrap();
    let dummy_prior = dir.path().join("p.bin");
    std::fs::write(&dummy_prior, b"QPI1").unwrap();
    let mut cfg = tiny_cfg(32);
    cfg.initial_prior = Some(dummy_prior);
    cfg.vocab_file = None;
    assert!(matches!(
        cfg.validate(CodecMode::Compress),
        Err(CodecError::InvalidConfig(_))
    ));

    // EstimateOnly mode + 0-token empty stream + exact batch-multiple stream (64 tokens)
    let stop = Arc::new(AtomicBool::new(false));
    let est_cfg = tiny_cfg(48);

    let empty_pp = dir.path().join("empty.pp");
    let empty_bin = dir.path().join("empty.bin");
    let empty_out = dir.path().join("empty_out.pp");
    write_tokens_be16(&empty_pp, &[]).unwrap();
    run_codec(
        &empty_pp,
        &empty_bin,
        &est_cfg,
        CodecMode::Compress,
        stop.clone(),
    )
    .unwrap();
    run_codec(
        &empty_bin,
        &empty_out,
        &est_cfg,
        CodecMode::Decompress,
        stop.clone(),
    )
    .unwrap();
    assert!(read_tokens_be16(&empty_out).unwrap().is_empty());

    let exact_pp = dir.path().join("exact64.pp");
    let exact_bin = dir.path().join("exact64.bin");
    let exact_out = dir.path().join("exact64_out.pp");
    let exact_tokens = build_synthetic_tokens(64, 48);
    write_tokens_be16(&exact_pp, &exact_tokens).unwrap();

    match run_codec(
        &exact_pp,
        &exact_bin,
        &est_cfg,
        CodecMode::EstimateOnly,
        stop.clone(),
    )
    .unwrap()
    {
        RunOutcome::Completed(summary) => {
            assert_eq!(summary.actual_compressed_bytes, None);
            assert!(summary.modeled_bits.is_finite() && summary.modeled_bits > 0.0);
        }
        RunOutcome::RequeueRequested { .. } => panic!("unexpected requeue"),
    }

    run_codec(
        &exact_pp,
        &exact_bin,
        &est_cfg,
        CodecMode::Compress,
        stop.clone(),
    )
    .unwrap();
    run_codec(
        &exact_bin,
        &exact_out,
        &est_cfg,
        CodecMode::Decompress,
        stop,
    )
    .unwrap();
    assert_eq!(read_tokens_be16(&exact_out).unwrap(), exact_tokens);
}
