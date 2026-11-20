//! End-to-end integration tests for the `tufazip` CLI binary.

use std::path::PathBuf;
use std::process::Command;

fn bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_tufazip"))
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn cli_inspect_prior_verifies_real_assets() {
    let assets = workspace_root().join("assets");
    let output = Command::new(bin_path())
        .args([
            "inspect-prior",
            assets.join("weights.bin").to_str().unwrap(),
            "--vocab",
            assets.join("enwik9.voc").to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("QPI1"));
    assert!(stdout.contains("e94c089f8eb985a87307dca3a7702385fcaccd14d95743811f8788f7da24448d"));
    assert!(stdout.contains("MATCH"));
}

#[test]
fn cli_preprocess_and_full_compress_decompress_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let raw_in = dir.path().join("sample.txt");
    let voc_file = dir.path().join("sample.voc");
    let pp_file = dir.path().join("sample.pp");
    let pp_decoded = dir.path().join("sample_pp_decoded.txt");
    let compressed = dir.path().join("sample.tz");
    let full_voc = dir.path().join("full.voc");
    let final_out = dir.path().join("sample_roundtrip.txt");

    let sample = concat!(
        "<page><title>Tufazip Rust Port</title><text>\n",
        "Deterministic online neural lossless compression in pure Rust using Candle.\n",
        "Case handling: ALLCAPS, TitleCase, lower-case, UTF-8 café 日本語!\n",
        "</text></page>\n"
    )
    .repeat(6);
    std::fs::write(&raw_in, sample.as_bytes()).unwrap();

    // 1. Standalone preprocess encode + decode
    let enc_status = Command::new(bin_path())
        .args([
            "preprocess",
            "encode",
            voc_file.to_str().unwrap(),
            raw_in.to_str().unwrap(),
            pp_file.to_str().unwrap(),
            "320",
            "2",
        ])
        .status()
        .unwrap();
    assert!(enc_status.success());

    let dec_status = Command::new(bin_path())
        .args([
            "preprocess",
            "decode",
            voc_file.to_str().unwrap(),
            pp_file.to_str().unwrap(),
            pp_decoded.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(dec_status.success());
    assert_eq!(std::fs::read(&pp_decoded).unwrap(), sample.as_bytes());

    // 2. Full end-to-end neural compress -> decompress roundtrip
    let comp_out = Command::new(bin_path())
        .args([
            "compress",
            raw_in.to_str().unwrap(),
            compressed.to_str().unwrap(),
            "--vocab",
            full_voc.to_str().unwrap(),
            "--max-words",
            "320",
            "--min-freq",
            "2",
            "--preset",
            "tiny",
            "--n-layer",
            "2",
            "--d-model",
            "32",
            "--n-head",
            "2",
            "--d-inner",
            "64",
            "--ctx-len",
            "16",
            "--batch-size",
            "4",
            "--block-len",
            "64",
            "--retrain-period",
            "64",
            "--retrain-len",
            "128",
            "--retrain-batch-size",
            "4",
            "--retrain-update-len",
            "8",
        ])
        .output()
        .unwrap();
    assert!(
        comp_out.status.success(),
        "compress failed: {}",
        String::from_utf8_lossy(&comp_out.stderr)
    );

    let decomp_out = Command::new(bin_path())
        .args([
            "decompress",
            compressed.to_str().unwrap(),
            final_out.to_str().unwrap(),
            "--vocab",
            full_voc.to_str().unwrap(),
            "--preset",
            "tiny",
            "--n-layer",
            "2",
            "--d-model",
            "32",
            "--n-head",
            "2",
            "--d-inner",
            "64",
            "--ctx-len",
            "16",
            "--batch-size",
            "4",
            "--block-len",
            "64",
            "--retrain-period",
            "64",
            "--retrain-len",
            "128",
            "--retrain-batch-size",
            "4",
            "--retrain-update-len",
            "8",
        ])
        .output()
        .unwrap();
    assert!(
        decomp_out.status.success(),
        "decompress failed: {}",
        String::from_utf8_lossy(&decomp_out.stderr)
    );

    assert_eq!(std::fs::read(&final_out).unwrap(), sample.as_bytes());
}
