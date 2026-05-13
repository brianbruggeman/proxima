//! Tokio-elimination verifier — concrete `#[test]`s that assert the
//! proxima library stays runtime-agnostic where it claims to.
//!
//! Two checks, deliberately simple `read_to_string + grep` shape:
//!
//! 1. `tokio::select!` / `tokio::join!` / `tokio::pin!` macro paths
//!    must not appear anywhere under `rust/src/`. These were swept
//!    in P1.4 of the tokio-elimination plan; this test prevents
//!    regression.
//!
//! 2. `tokio::` must not appear on any line that defines a public
//!    item signature (`pub fn`, `pub async fn`, `pub trait`, `pub
//!    struct`, `pub enum`, `pub type`). This is a heuristic — won't
//!    catch multi-line signatures perfectly — but it catches the
//!    common drift where someone adds `pub fn foo(s: tokio::sync::
//!    Mutex<T>) -> ...` and ships it.
//!
//! Out of scope: tokio types used in private impl detail behind the
//! runtime trait (e.g. `tokio::sync::Notify` in `listener.rs:152`,
//! `tokio::time::sleep_until` inside listener accept loops). Those
//! are documented as acceptable in `rust/docs/runtime-coupling.md`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    // post-Phase-A: umbrella package is at workspace root; CARGO_MANIFEST_DIR
    // IS the repo root (was rust/ subdir pre-collapse).
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn rust_src() -> PathBuf {
    repo_root().join("src")
}

fn walk_rust_files(root: &Path, files: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_rust_files(&path, files);
        } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            files.push(path);
        }
    }
}

fn collect_rust_src_files() -> Vec<PathBuf> {
    let mut files = Vec::new();
    walk_rust_files(&rust_src(), &mut files);
    files
}

/// P1.4 verifier — no `tokio::{select,join,pin}!` macro paths in
/// `rust/src/`. These were eliminated in the listener-select
/// migration; this test prevents regression.
#[test]
fn no_tokio_macro_paths_in_src() {
    let macros = ["tokio::select!", "tokio::join!", "tokio::pin!"];
    let mut leaks: Vec<(PathBuf, usize, String)> = Vec::new();
    for path in collect_rust_src_files() {
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        for (index, line) in text.lines().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                continue;
            }
            for needle in &macros {
                if line.contains(needle) {
                    leaks.push((path.clone(), index + 1, (*needle).to_string()));
                }
            }
        }
    }
    assert!(
        leaks.is_empty(),
        "tokio-elimination regression: {} macro-path leak(s) in rust/src/. \
         Use futures::select! / select_biased! instead. Leaks: {:#?}",
        leaks.len(),
        leaks
            .iter()
            .map(|(path, line, needle)| format!(
                "{}:{} ({})",
                path.strip_prefix(repo_root())
                    .unwrap_or(path.as_path())
                    .display(),
                line,
                needle
            ))
            .collect::<Vec<_>>(),
    );
}

/// Public-API leak heuristic — `tokio::` must not appear on the same
/// line as a `pub fn` / `pub async fn` / `pub trait` / `pub struct`
/// / `pub enum` / `pub type` signature opener. Catches the common
/// drift where someone exposes a tokio type in a public signature.
///
/// Heuristic: only checks single-line signatures. Multi-line traits
/// with `pub trait Foo<T: tokio::io::AsyncRead>` would slip through
/// if the bound is on a separate line — accept the gap.
#[test]
fn no_tokio_in_public_signatures() {
    let pub_openers = [
        "pub fn ",
        "pub async fn ",
        "pub unsafe fn ",
        "pub const fn ",
        "pub trait ",
        "pub struct ",
        "pub enum ",
        "pub type ",
    ];
    let mut leaks: Vec<(PathBuf, usize, String)> = Vec::new();
    for path in collect_rust_src_files() {
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        for (index, line) in text.lines().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                continue;
            }
            let opens_pub_item = pub_openers.iter().any(|opener| trimmed.starts_with(opener));
            if !opens_pub_item {
                continue;
            }
            if line.contains("tokio::") {
                leaks.push((path.clone(), index + 1, line.trim().to_string()));
            }
        }
    }
    assert!(
        leaks.is_empty(),
        "tokio public-API leak: {} signature(s) reference tokio:: \
         types. Library public API must stay runtime-agnostic — use \
         proxima::sync::* / proxima::time::* / futures::io::* / \
         crate Runtime trait instead. Leaks: {:#?}",
        leaks.len(),
        leaks
            .iter()
            .map(|(path, line, sig)| format!(
                "{}:{} — {sig}",
                path.strip_prefix(repo_root())
                    .unwrap_or(path.as_path())
                    .display(),
                line
            ))
            .collect::<Vec<_>>(),
    );
}
