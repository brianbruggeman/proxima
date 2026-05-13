//! Mechanically re-provable (P16) proof that `src/pipe/part.rs` compiles clean
//! under a bare `#![no_std]` crate root with zero `extern crate`s — the
//! tier-3 claim for the `Part` primitive, independent of whether the rest of
//! `proxima-primitives` is no_std-ready (it isn't yet; `cargo build -p
//! proxima-primitives --no-default-features --features part-source` fails on
//! unrelated modules — see `docs/proxima-pipe/discipline.md`).
//!
//! This shells out to `rustc` against the literal bytes of the source file
//! (prefixed with `#![no_std]`), so the proof is over the actual shipped
//! module, not a hand-copied approximation, and re-runs identically in CI.
//! (`include!` can't be used here — inner doc comments / inner attributes
//! are only legal at the syntactic start of the file the compiler parses
//! directly, not through a textual `include!` splice — so the wrapper
//! concatenates the real source's bytes into one literal crate-root file
//! instead.)

#![cfg(feature = "part-source")]
// test code — expect() failures are the assertion, per rust.md ("unwrap OK in
// tests but prefer expect() with a good message").
#![allow(clippy::expect_used)]

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[test]
fn part_module_compiles_under_bare_no_std_zero_extern_crates() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let part_source = manifest_dir.join("src").join("pipe").join("part.rs");
    assert!(part_source.is_file(), "expected {part_source:?} to exist");

    let out_dir = manifest_dir.join("target").join("part_tier3_check");
    fs::create_dir_all(&out_dir).expect("create scratch out dir");

    let part_body = fs::read_to_string(&part_source).expect("read src/part.rs");
    let wrapper_path = out_dir.join("wrapper.rs");
    let wrapper_body = format!("#![no_std]\n{part_body}");
    fs::write(&wrapper_path, wrapper_body).expect("write no_std wrapper");

    let metadata_out = out_dir.join("part_tier3_check.rmeta");
    let output = Command::new("rustc")
        .args(["--edition", "2024", "--crate-type", "lib"])
        .args(["--crate-name", "part_tier3_check"])
        .arg("--emit=metadata")
        .arg("-o")
        .arg(&metadata_out)
        .arg(&wrapper_path)
        .output()
        .expect("invoke rustc directly (must be on PATH)");

    assert!(
        output.status.success(),
        "part.rs must compile under bare #![no_std] with zero extern crates \
         (core-only tier-3 claim):\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}
