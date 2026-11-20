//! Differential integration tests compiling Fabrice Bellard's original `preprocess.c`
//! reference implementation and verifying exact parity with `tufazip-preprocess`.

use std::path::{Path, PathBuf};
use std::process::Command;

use tufazip_preprocess::{
    decode_file, encode_file, encode_with_dictionary, write_tokens_be16, Dictionary, EncodeConfig,
    SortOrder,
};

fn fixture_c_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/preprocess_ref.c")
}

fn compile_ref_binary(out_dir: &Path) -> PathBuf {
    let bin = out_dir.join("preprocess_ref");
    let status = Command::new("cc")
        .args([
            "-O2",
            "-w",
            "-D_WIN32",
            "-DCONFIG_VERSION=\"ref\"",
            fixture_c_path().to_str().unwrap(),
            "-lm",
            "-o",
            bin.to_str().unwrap(),
        ])
        .status()
        .expect("system C compiler (`cc`) must be available for differential tests");
    assert!(status.success(), "failed to compile preprocess_ref.c");
    bin
}

fn run_ref(bin: &Path, args: &[&str]) {
    let out = Command::new(bin).args(args).output().unwrap();
    assert!(
        out.status.success(),
        "preprocess_ref {:?} failed:\nstdout: {}\nstderr: {}",
        args,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn c_differential_exact_voc_and_pp_match_on_real_text() {
    let dir = tempfile::tempdir().unwrap();
    let c_bin = compile_ref_binary(dir.path());
    let sample_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tufazip-codec/src/engine.rs");
    let sample_bytes = std::fs::read(&sample_path).unwrap();

    for (max_words, min_freq) in [(400usize, 5u32), (512usize, 3u32)] {
        let tag = format!("{max_words}_{min_freq}");
        let c_voc = dir.path().join(format!("c_{tag}.voc"));
        let c_pp = dir.path().join(format!("c_{tag}.pp"));
        let rs_voc = dir.path().join(format!("rs_{tag}.voc"));
        let rs_pp = dir.path().join(format!("rs_{tag}.pp"));
        let rs_from_c = dir.path().join(format!("rs_from_c_{tag}.txt"));
        let c_from_rs = dir.path().join(format!("c_from_rs_{tag}.txt"));

        run_ref(
            &c_bin,
            &[
                "c",
                c_voc.to_str().unwrap(),
                sample_path.to_str().unwrap(),
                c_pp.to_str().unwrap(),
                &max_words.to_string(),
                &min_freq.to_string(),
            ],
        );

        encode_file(
            &sample_path,
            &rs_pp,
            &rs_voc,
            &EncodeConfig {
                max_words,
                min_freq,
                sort_order: SortOrder::Discovery,
                debug_dict_path: None,
            },
        )
        .unwrap();

        assert_eq!(
            std::fs::read(&rs_voc).unwrap(),
            std::fs::read(&c_voc).unwrap(),
            "dictionary mismatch for {tag}"
        );
        assert_eq!(
            std::fs::read(&rs_pp).unwrap(),
            std::fs::read(&c_pp).unwrap(),
            "token stream mismatch for {tag}"
        );

        decode_file(&c_pp, &rs_from_c, &c_voc).unwrap();
        assert_eq!(std::fs::read(&rs_from_c).unwrap(), sample_bytes);

        run_ref(
            &c_bin,
            &[
                "d",
                rs_voc.to_str().unwrap(),
                rs_pp.to_str().unwrap(),
                c_from_rs.to_str().unwrap(),
            ],
        );
        assert_eq!(std::fs::read(&c_from_rs).unwrap(), sample_bytes);
    }
}

#[test]
fn c_differential_cross_decode_edge_cases_and_sorted_vocab() {
    let dir = tempfile::tempdir().unwrap();
    let c_bin = compile_ref_binary(dir.path());
    let in_path = dir.path().join("edge_corpus.txt");

    let corpus = concat!(
        "<page><title>Edge Case Corpus</title><text>\n",
        "ALLCAPS TitleCase lower-case camelCase PascalCase snake_case.\n",
        "Backslashes \\\\ and single \\ escapes, tabs\t, double  spaces, newlines\n\n",
        "UTF-8 multibyte text: café naïve résumé 日本語 — € 42!\n",
        "</text></page>\n"
    )
    .repeat(16);
    std::fs::write(&in_path, corpus.as_bytes()).unwrap();

    for (sort_order, flag) in [
        (SortOrder::Discovery, None),
        (SortOrder::ByFrequency, Some("-s")),
    ] {
        let tag = format!("{sort_order:?}");
        let c_voc = dir.path().join(format!("c_{tag}.voc"));
        let c_pp = dir.path().join(format!("c_{tag}.pp"));
        let rs_voc = dir.path().join(format!("rs_{tag}.voc"));
        let rs_pp = dir.path().join(format!("rs_{tag}.pp"));
        let rs_dbg = dir.path().join(format!("rs_{tag}.dbg"));
        let rs_from_c = dir.path().join(format!("rs_from_c_{tag}.txt"));
        let c_from_rs = dir.path().join(format!("c_from_rs_{tag}.txt"));

        let mut args = Vec::new();
        if let Some(f) = flag {
            args.push(f);
        }
        args.extend_from_slice(&[
            "c",
            c_voc.to_str().unwrap(),
            in_path.to_str().unwrap(),
            c_pp.to_str().unwrap(),
            "350",
            "3",
        ]);
        run_ref(&c_bin, &args);

        encode_file(
            &in_path,
            &rs_pp,
            &rs_voc,
            &EncodeConfig {
                max_words: 350,
                min_freq: 3,
                sort_order,
                debug_dict_path: Some(rs_dbg.clone()),
            },
        )
        .unwrap();
        assert!(rs_dbg.is_file());

        decode_file(&c_pp, &rs_from_c, &c_voc).unwrap();
        assert_eq!(std::fs::read(&rs_from_c).unwrap(), corpus.as_bytes());

        run_ref(
            &c_bin,
            &[
                "d",
                rs_voc.to_str().unwrap(),
                rs_pp.to_str().unwrap(),
                c_from_rs.to_str().unwrap(),
            ],
        );
        assert_eq!(std::fs::read(&c_from_rs).unwrap(), corpus.as_bytes());
    }
}

#[test]
fn c_differential_enwik9_voc_cross_decode() {
    let dir = tempfile::tempdir().unwrap();
    let c_bin = compile_ref_binary(dir.path());
    let voc_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/enwik9.voc");
    let dict = Dictionary::load(&voc_path).unwrap();

    let sample = concat!(
        "<mediawiki><page><title>Rust (programming language)</title>\n",
        "<text>Rust is a systems programming language focused on safety, speed, and concurrency.\n",
        "Wikipedia UTF-8 check: naïve café — 2026.</text></page></mediawiki>\n"
    )
    .as_bytes();

    let tokens = encode_with_dictionary(sample, &dict).unwrap();
    let pp_path = dir.path().join("enwik_sample.pp");
    let c_out = dir.path().join("c_out.txt");
    let rs_out = dir.path().join("rs_out.txt");
    write_tokens_be16(&pp_path, &tokens).unwrap();

    run_ref(
        &c_bin,
        &[
            "d",
            voc_path.to_str().unwrap(),
            pp_path.to_str().unwrap(),
            c_out.to_str().unwrap(),
        ],
    );
    decode_file(&pp_path, &rs_out, &voc_path).unwrap();

    assert_eq!(std::fs::read(&c_out).unwrap(), sample);
    assert_eq!(std::fs::read(&rs_out).unwrap(), sample);
}
