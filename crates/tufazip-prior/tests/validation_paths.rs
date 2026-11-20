//! Exhaustive validation and error-path tests for `QPI1`/`QPR1`/`LRA1`/`QLA1` archives
//! and `plan_depth_prior`.

use std::collections::BTreeMap;
use std::path::Path;

use sha2::{Digest, Sha256};
use tufazip_prior::{
    plan_depth_prior, read_archive, write_archive, ArchiveMagic, BitWidth, InitializationKind,
    InitializationMeta, PriorError, TransformerGeometry,
};

fn small_geom(n_layer: usize) -> TransformerGeometry {
    TransformerGeometry {
        vocab_size: 32,
        n_layer,
        d_model: 16,
        n_head: 2,
        d_inner: 32,
        ctx_len: 8,
        tie_embeddings: false,
        position_type: "rope".to_owned(),
        qk_norm: true,
        norm_style: "peri".to_owned(),
    }
}

fn target_shapes(geom: &TransformerGeometry) -> BTreeMap<String, Vec<usize>> {
    let mut shapes = BTreeMap::new();
    shapes.insert(
        "tok_emb.weight".to_owned(),
        vec![geom.vocab_size, geom.d_model],
    );
    shapes.insert("emb_norm.weight".to_owned(), vec![geom.d_model]);
    shapes.insert("norm.weight".to_owned(), vec![geom.d_model]);
    shapes.insert(
        "head.weight".to_owned(),
        vec![geom.vocab_size, geom.d_model],
    );
    let head_dim = geom.d_model / geom.n_head;
    for l in 0..geom.n_layer {
        let p = format!("blocks.{l}");
        shapes.insert(format!("{p}.attn_pre_norm.weight"), vec![geom.d_model]);
        shapes.insert(
            format!("{p}.attn.qkv.weight"),
            vec![3 * geom.d_model, geom.d_model],
        );
        shapes.insert(format!("{p}.attn.q_norm.weight"), vec![head_dim]);
        shapes.insert(format!("{p}.attn.k_norm.weight"), vec![head_dim]);
        shapes.insert(
            format!("{p}.attn.proj.weight"),
            vec![geom.d_model, geom.d_model],
        );
        shapes.insert(format!("{p}.attn_post_norm.weight"), vec![geom.d_model]);
        shapes.insert(format!("{p}.ffn_pre_norm.weight"), vec![geom.d_model]);
        shapes.insert(
            format!("{p}.ffn.gate.weight"),
            vec![geom.d_inner, geom.d_model],
        );
        shapes.insert(
            format!("{p}.ffn.up.weight"),
            vec![geom.d_inner, geom.d_model],
        );
        shapes.insert(
            format!("{p}.ffn.down.weight"),
            vec![geom.d_model, geom.d_inner],
        );
        shapes.insert(format!("{p}.ffn_post_norm.weight"), vec![geom.d_model]);
    }
    shapes
}

fn build_source_tensors(
    geom: &TransformerGeometry,
) -> Vec<(String, Vec<usize>, Vec<f32>, BitWidth)> {
    let head_dim = geom.d_model / geom.n_head;
    let mut out = vec![
        (
            "tok_emb.weight".to_owned(),
            vec![geom.vocab_size, geom.d_model],
            vec![0.1f32; geom.vocab_size * geom.d_model],
            BitWidth::B8,
        ),
        (
            "emb_norm.weight".to_owned(),
            vec![geom.d_model],
            vec![1.0f32; geom.d_model],
            BitWidth::B16,
        ),
        (
            "norm.weight".to_owned(),
            vec![geom.d_model],
            vec![1.0f32; geom.d_model],
            BitWidth::B16,
        ),
        (
            "head.weight".to_owned(),
            vec![geom.vocab_size, geom.d_model],
            vec![-0.1f32; geom.vocab_size * geom.d_model],
            BitWidth::B8,
        ),
    ];

    for l in 0..geom.n_layer {
        let p = format!("blocks.{l}");
        let val = (l as f32 + 1.0) * 0.25;
        out.push((
            format!("{p}.attn_pre_norm.weight"),
            vec![geom.d_model],
            vec![1.0f32; geom.d_model],
            BitWidth::B16,
        ));
        out.push((
            format!("{p}.attn.qkv.weight"),
            vec![3 * geom.d_model, geom.d_model],
            vec![val; 3 * geom.d_model * geom.d_model],
            BitWidth::B8,
        ));
        out.push((
            format!("{p}.attn.q_norm.weight"),
            vec![head_dim],
            vec![1.0f32; head_dim],
            BitWidth::B16,
        ));
        out.push((
            format!("{p}.attn.k_norm.weight"),
            vec![head_dim],
            vec![1.0f32; head_dim],
            BitWidth::B16,
        ));
        out.push((
            format!("{p}.attn.proj.weight"),
            vec![geom.d_model, geom.d_model],
            vec![val; geom.d_model * geom.d_model],
            BitWidth::B8,
        ));
        out.push((
            format!("{p}.attn_post_norm.weight"),
            vec![geom.d_model],
            vec![1.0f32; geom.d_model],
            BitWidth::B16,
        ));
        out.push((
            format!("{p}.ffn_pre_norm.weight"),
            vec![geom.d_model],
            vec![1.0f32; geom.d_model],
            BitWidth::B16,
        ));
        out.push((
            format!("{p}.ffn.gate.weight"),
            vec![geom.d_inner, geom.d_model],
            vec![val; geom.d_inner * geom.d_model],
            BitWidth::B8,
        ));
        out.push((
            format!("{p}.ffn.up.weight"),
            vec![geom.d_inner, geom.d_model],
            vec![val; geom.d_inner * geom.d_model],
            BitWidth::B8,
        ));
        out.push((
            format!("{p}.ffn.down.weight"),
            vec![geom.d_model, geom.d_inner],
            vec![val; geom.d_model * geom.d_inner],
            BitWidth::B8,
        ));
        out.push((
            format!("{p}.ffn_post_norm.weight"),
            vec![geom.d_model],
            vec![1.0f32; geom.d_model],
            BitWidth::B16,
        ));
    }
    out
}

fn write_test_vocab(dir: &Path) -> (std::path::PathBuf, String) {
    let path = dir.join("test.voc");
    let mut bytes = Vec::new();
    for b in 0..32u8 {
        bytes.push(b'a' + (b % 26));
        bytes.push(b'\n');
    }
    std::fs::write(&path, &bytes).unwrap();
    let sha = format!("{:x}", Sha256::digest(&bytes));
    (path, sha)
}

fn write_qpi_with_init(
    path: &Path,
    magic: ArchiveMagic,
    init: Option<InitializationMeta>,
    tensors: &[(String, Vec<usize>, Vec<f32>, BitWidth)],
) {
    let refs: Vec<(&str, &[usize], &[f32], BitWidth)> = tensors
        .iter()
        .map(|(n, s, d, b)| (n.as_str(), s.as_slice(), d.as_slice(), *b))
        .collect();
    write_archive(path, magic, Some("test-policy".to_owned()), init, &refs).unwrap();
}

#[test]
fn depth_growth_valid_and_validation_rejections() {
    let dir = tempfile::tempdir().unwrap();
    let (voc_path, voc_sha) = write_test_vocab(dir.path());
    let src_geom = small_geom(2);
    let tgt_geom = small_geom(4);
    let tgt_shapes = target_shapes(&tgt_geom);
    let tensors = build_source_tensors(&src_geom);

    // 1. Valid depth-growth (2 -> 4, mapping [0, 2], zeroing [1, 3])
    let valid_path = dir.path().join("depth_valid.bin");
    let valid_init = InitializationMeta {
        kind: InitializationKind::DepthGrowth,
        source_geometry: src_geom.clone(),
        target_geometry: tgt_geom.clone(),
        block_mapping: Some(vec![0, 2]),
        block_schedule: None,
        initialization_seed: 1111,
        vocab_sha256: voc_sha.clone(),
        source_checkpoint_sha256: None,
    };
    write_qpi_with_init(
        &valid_path,
        ArchiveMagic::Qpi1,
        Some(valid_init.clone()),
        &tensors,
    );
    let archive = read_archive(&valid_path).unwrap();
    let plan = plan_depth_prior(&archive, &voc_path, &tgt_geom, 1111, &tgt_shapes).unwrap();
    assert_eq!(plan.initialization_kind, InitializationKind::DepthGrowth);
    assert_eq!(plan.target_sources.get(&0), Some(&0));
    assert_eq!(plan.target_sources.get(&2), Some(&1));
    assert_eq!(plan.zero_unmapped_targets, vec![1, 3]);

    // 2. Non-canonical depth-growth mapping [0, 1] rejected
    let bad_map_path = dir.path().join("depth_bad_map.bin");
    let mut bad_map_init = valid_init.clone();
    bad_map_init.block_mapping = Some(vec![0, 1]);
    write_qpi_with_init(
        &bad_map_path,
        ArchiveMagic::Qpi1,
        Some(bad_map_init),
        &tensors,
    );
    assert!(matches!(
        plan_depth_prior(
            &read_archive(&bad_map_path).unwrap(),
            &voc_path,
            &tgt_geom,
            1111,
            &tgt_shapes
        ),
        Err(PriorError::Validation(_))
    ));

    // 3. Missing block_mapping rejected
    let no_map_path = dir.path().join("depth_no_map.bin");
    let mut no_map_init = valid_init.clone();
    no_map_init.block_mapping = None;
    write_qpi_with_init(
        &no_map_path,
        ArchiveMagic::Qpi1,
        Some(no_map_init),
        &tensors,
    );
    assert!(matches!(
        plan_depth_prior(
            &read_archive(&no_map_path).unwrap(),
            &voc_path,
            &tgt_geom,
            1111,
            &tgt_shapes
        ),
        Err(PriorError::Validation(_))
    ));

    // 4. Loop-untying invalid schedules
    for bad_sched in [
        None,
        Some(vec![0, 1]),
        Some(vec![0, 0, 0, 0]),
        Some(vec![0, 1, 2, 0]),
    ] {
        let p = dir.path().join("bad_loop.bin");
        let init = InitializationMeta {
            kind: InitializationKind::LoopUntying,
            source_geometry: src_geom.clone(),
            target_geometry: tgt_geom.clone(),
            block_mapping: None,
            block_schedule: bad_sched,
            initialization_seed: 1111,
            vocab_sha256: voc_sha.clone(),
            source_checkpoint_sha256: None,
        };
        write_qpi_with_init(&p, ArchiveMagic::Qpi1, Some(init), &tensors);
        assert!(matches!(
            plan_depth_prior(
                &read_archive(&p).unwrap(),
                &voc_path,
                &tgt_geom,
                1111,
                &tgt_shapes
            ),
            Err(PriorError::Validation(_))
        ));
    }

    // 5. Non-QPI1 magic rejected by plan_depth_prior
    let qpr_path = dir.path().join("qpr.bin");
    write_qpi_with_init(
        &qpr_path,
        ArchiveMagic::Qpr1,
        Some(valid_init.clone()),
        &tensors,
    );
    assert!(matches!(
        plan_depth_prior(
            &read_archive(&qpr_path).unwrap(),
            &voc_path,
            &tgt_geom,
            1111,
            &tgt_shapes
        ),
        Err(PriorError::Validation(_))
    ));

    // 6. Missing initialization metadata rejected
    let no_init_path = dir.path().join("no_init.bin");
    write_qpi_with_init(&no_init_path, ArchiveMagic::Qpi1, None, &tensors);
    assert!(matches!(
        plan_depth_prior(
            &read_archive(&no_init_path).unwrap(),
            &voc_path,
            &tgt_geom,
            1111,
            &tgt_shapes
        ),
        Err(PriorError::Validation(_))
    ));

    // 7. Mismatched seed, vocab digest, target geometry, source geometry
    assert!(matches!(
        plan_depth_prior(&archive, &voc_path, &tgt_geom, 9999, &tgt_shapes),
        Err(PriorError::Validation(_))
    ));

    let wrong_voc = dir.path().join("wrong.voc");
    std::fs::write(&wrong_voc, b"different-vocab\n").unwrap();
    assert!(matches!(
        plan_depth_prior(&archive, &wrong_voc, &tgt_geom, 1111, &tgt_shapes),
        Err(PriorError::Validation(_))
    ));

    let mut diff_tgt = tgt_geom.clone();
    diff_tgt.d_model = 64;
    assert!(matches!(
        plan_depth_prior(&archive, &voc_path, &diff_tgt, 1111, &tgt_shapes),
        Err(PriorError::Validation(_))
    ));

    let bad_src_path = dir.path().join("bad_src_geom.bin");
    let mut bad_src_init = valid_init.clone();
    bad_src_init.source_geometry.qk_norm = false;
    write_qpi_with_init(
        &bad_src_path,
        ArchiveMagic::Qpi1,
        Some(bad_src_init),
        &tensors,
    );
    assert!(matches!(
        plan_depth_prior(
            &read_archive(&bad_src_path).unwrap(),
            &voc_path,
            &tgt_geom,
            1111,
            &tgt_shapes
        ),
        Err(PriorError::Validation(_))
    ));

    // 8. Incomplete tensor set & shape mismatch
    let short_path = dir.path().join("short.bin");
    write_qpi_with_init(
        &short_path,
        ArchiveMagic::Qpi1,
        Some(valid_init),
        &tensors[1..],
    );
    assert!(matches!(
        plan_depth_prior(
            &read_archive(&short_path).unwrap(),
            &voc_path,
            &tgt_geom,
            1111,
            &tgt_shapes
        ),
        Err(PriorError::Validation(_))
    ));

    let mut wrong_shapes = tgt_shapes;
    wrong_shapes.insert("norm.weight".to_owned(), vec![999]);
    assert!(matches!(
        plan_depth_prior(&archive, &voc_path, &tgt_geom, 1111, &wrong_shapes),
        Err(PriorError::Validation(_))
    ));
}

fn write_raw_archive(path: &Path, magic: &[u8; 4], meta: serde_json::Value, payload: &[u8]) {
    let json = serde_json::to_vec(&meta).unwrap();
    let mut raw = Vec::new();
    raw.extend_from_slice(magic);
    raw.extend_from_slice(&(json.len() as u64).to_le_bytes());
    raw.extend_from_slice(&json);
    raw.extend_from_slice(payload);
    std::fs::write(path, raw).unwrap();
}

#[test]
fn malformed_binary_archives_return_structured_errors() {
    let dir = tempfile::tempdir().unwrap();

    // Unknown magic
    let p_magic = dir.path().join("magic.bin");
    std::fs::write(&p_magic, b"NOPE\x00\x00\x00\x00\x00\x00\x00\x00").unwrap();
    assert!(matches!(
        read_archive(&p_magic),
        Err(PriorError::UnknownMagic([b'N', b'O', b'P', b'E']))
    ));

    // Truncated header
    let p_trunc = dir.path().join("trunc.bin");
    std::fs::write(&p_trunc, b"QPI1\x04\x00").unwrap();
    assert!(matches!(
        read_archive(&p_trunc),
        Err(PriorError::Truncated(_))
    ));

    // Unsupported version
    let p_ver = dir.path().join("ver.bin");
    write_raw_archive(
        &p_ver,
        b"QPR1",
        serde_json::json!({ "version": 2, "entries": [] }),
        &[],
    );
    assert!(matches!(
        read_archive(&p_ver),
        Err(PriorError::UnsupportedVersion(2))
    ));

    // Duplicate tensor name
    let p_dup = dir.path().join("dup.bin");
    let entry = serde_json::json!({
        "name": "w", "shape": [2], "bits": 16, "scale_count": 0, "scale_bytes": 0, "value_bytes": 4
    });
    write_raw_archive(
        &p_dup,
        b"QPR1",
        serde_json::json!({ "version": 1, "entries": [entry, entry] }),
        &[0u8; 8],
    );
    assert!(matches!(
        read_archive(&p_dup),
        Err(PriorError::DuplicateTensor(name)) if name == "w"
    ));

    // Zero shape dimension
    let p_shape = dir.path().join("shape.bin");
    write_raw_archive(
        &p_shape,
        b"QPR1",
        serde_json::json!({
            "version": 1,
            "entries": [{ "name": "z", "shape": [0, 4], "bits": 16, "scale_count": 0, "scale_bytes": 0, "value_bytes": 0 }]
        }),
        &[],
    );
    assert!(matches!(
        read_archive(&p_shape),
        Err(PriorError::InvalidShape { .. })
    ));

    // Unsupported bit width
    let p_bits = dir.path().join("bits.bin");
    write_raw_archive(
        &p_bits,
        b"QPR1",
        serde_json::json!({
            "version": 1,
            "entries": [{ "name": "b", "shape": [4], "bits": 3, "scale_count": 1, "scale_bytes": 2, "value_bytes": 2 }]
        }),
        &[],
    );
    assert!(matches!(
        read_archive(&p_bits),
        Err(PriorError::UnsupportedBitWidth { bits: 3, .. })
    ));

    // Value bytes mismatch
    let p_val = dir.path().join("val_sz.bin");
    write_raw_archive(
        &p_val,
        b"QPR1",
        serde_json::json!({
            "version": 1,
            "entries": [{ "name": "v", "shape": [4], "bits": 16, "scale_count": 0, "scale_bytes": 0, "value_bytes": 99 }]
        }),
        &[],
    );
    assert!(matches!(
        read_archive(&p_val),
        Err(PriorError::SizeMismatch { .. })
    ));

    // Scale count mismatch
    let p_sc = dir.path().join("sc_sz.bin");
    write_raw_archive(
        &p_sc,
        b"QPR1",
        serde_json::json!({
            "version": 1,
            "entries": [{ "name": "s", "shape": [2, 4], "bits": 8, "scale_count": 1, "scale_bytes": 2, "value_bytes": 8 }]
        }),
        &[],
    );
    assert!(matches!(
        read_archive(&p_sc),
        Err(PriorError::SizeMismatch { .. })
    ));

    // Trailing bytes after valid archive
    let p_trail = dir.path().join("trail.bin");
    write_archive(
        &p_trail,
        ArchiveMagic::Qpr1,
        None,
        None,
        &[("x", &[2], &[1.0, -1.0], BitWidth::B16)],
    )
    .unwrap();
    let mut bytes = std::fs::read(&p_trail).unwrap();
    bytes.push(0xAA);
    std::fs::write(&p_trail, bytes).unwrap();
    assert!(matches!(
        read_archive(&p_trail),
        Err(PriorError::TrailingBytes)
    ));
}
