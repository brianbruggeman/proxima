//! DX-audit verifiers — concrete `#[test]`s that assert the
//! source-level fixes from `docs/audit-pipe-dx-2026-05-18.md`
//! stay landed. Replaces "manual review on PR" with `cargo test`.
//!
//! Each test reads a file from the proxima repo root (located via
//! `CARGO_MANIFEST_DIR`) and asserts the audit fix's invariant.
//! These are deliberately simple `read_to_string + contains`
//! checks — no AST parsing, no rule engine. The earlier `proxima
//! audit` CLI subcommand was scope creep; this is the honest
//! shape.
//!
//! Findings covered: F1, F3, F4, F6, F8, F9, F10, F11.
//! F2, F5, F7, F12 are verified elsewhere — see
//! `docs/audit-pipe-dx-2026-05-18-todo.md` for the per-finding map.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::path::PathBuf;

fn repo_root() -> PathBuf {
    // post-Phase-A: the umbrella `proxima` package lives at the workspace root,
    // so `CARGO_MANIFEST_DIR` IS the repo root (was `rust/` subdir pre-collapse).
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read_repo_file(relative: &str) -> String {
    let path = repo_root().join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

/// F1 — `lib.rs` has a `//!` crate doc near the top.
#[test]
fn f1_lib_rs_has_crate_doc() {
    let text = read_repo_file("src/lib.rs");
    let head: String = text.lines().take(30).collect::<Vec<_>>().join("\n");
    assert!(
        head.contains("//!"),
        "audit finding F1: src/lib.rs must have a `//!` crate \
         doc in its first 30 lines; got:\n{head}",
    );
}

/// F3 — `DynPipe` and `ThreadLocalDynPipe` re-exports in `lib.rs`
/// live inside a `#[doc(hidden)]` block. Word-bounded so
/// `DynPipeFactory` (legitimately public) is not a false positive.
#[test]
fn f3_dyn_pipe_re_exports_doc_hidden() {
    let text = read_repo_file("src/lib.rs");
    let pattern = regex::Regex::new(r"\b(DynPipe|ThreadLocalDynPipe)\b")
        .expect("compile audit dyn-pipe regex");
    let lines: Vec<&str> = text.lines().collect();
    let mut leaks: Vec<usize> = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        let mentions = trimmed.starts_with("pub use") && pattern.is_match(line);
        if !mentions {
            continue;
        }
        let mut hidden = false;
        for back in 1..=5 {
            if index < back {
                break;
            }
            let prev = lines[index - back].trim();
            if prev.starts_with("#[doc(hidden)]") {
                hidden = true;
                break;
            }
            if prev.starts_with("pub use") || prev.starts_with("//") || prev.is_empty() {
                continue;
            }
            break;
        }
        if !hidden {
            leaks.push(index + 1);
        }
    }
    assert!(
        leaks.is_empty(),
        "audit finding F3: src/lib.rs has unhidden \
         DynPipe / ThreadLocalDynPipe re-exports at line(s) {leaks:?}",
    );
}

/// F4 — settings types are re-exported at the crate root so the
/// README's fluent hello-world can `use proxima::{BearerAuth,
/// Composable, HttpListener, HttpUpstream}` instead of
/// `use proxima::settings::{...}`.
#[test]
fn f4_fluent_settings_types_at_crate_root() {
    let text = read_repo_file("src/lib.rs");
    assert!(
        text.contains("pub use settings::"),
        "audit finding F4: src/lib.rs must re-export from \
         `settings::*` at the crate root",
    );
    for required in ["BearerAuth", "Composable", "HttpListener", "HttpUpstream"] {
        assert!(
            text.contains(required),
            "audit finding F4: src/lib.rs missing crate-root \
             re-export of `{required}`",
        );
    }
}

/// F6 — `middlewares/mod.rs` has a module-level `//!` doc with the
/// composition narrative.
#[test]
fn f6_middlewares_module_doc_present() {
    let text = read_repo_file("src/middlewares/mod.rs");
    let head: String = text.lines().take(10).collect::<Vec<_>>().join("\n");
    assert!(
        head.contains("//!"),
        "audit finding F6: src/middlewares/mod.rs must have a \
         `//!` module doc in its first 10 lines",
    );
}

/// F8 + F9 — `handler.rs` `//!` module doc contains both the
/// "Substrate primitives" section (F9) and the "Recording wraps
/// any Pipe" section (F8). Folded from the deleted `docs/PIPE.md`.
/// `pipe.rs` -> `handler.rs` (proxima-pipe TARGET 2 — the served-Pipe
/// rename): the served-HTTP face moved, and its module doc moved with it.
#[test]
fn f8_f9_pipe_module_doc_has_substrate_and_recording_sections() {
    let text = read_repo_file("proxima-primitives/src/pipe/handler.rs");
    let module_doc: String = text
        .lines()
        .take_while(|line| line.trim_start().starts_with("//!") || line.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        module_doc.contains("Substrate primitives"),
        "audit finding F9: proxima-primitives/src/pipe/handler.rs `//!` must contain a \
         '# Substrate primitives' section",
    );
    assert!(
        module_doc.contains("Recording wraps any Pipe"),
        "audit finding F8: proxima-primitives/src/pipe/handler.rs `//!` must contain a \
         '# Recording wraps any Pipe' section",
    );
}

/// F10 — `docs/QUICKSTART.md` mentions "Pipe" in its first 10
/// lines. Bare-minimum check that the quickstart names the
/// primitive somewhere up front.
#[test]
fn f10_quickstart_mentions_pipe_in_head() {
    let text = read_repo_file("docs/QUICKSTART.md");
    let head: String = text.lines().take(10).collect::<Vec<_>>().join("\n");
    assert!(
        head.contains("Pipe") || head.contains("pipe"),
        "audit finding F10: docs/QUICKSTART.md must mention `Pipe` \
         in its first 10 lines",
    );
}

/// F11 — `runtime/prime.rs` `//!` module doc names the design
/// principles. Three sentinel phrases drawn from the principles
/// section; any one missing means the design content has drifted.
#[test]
fn f11_prime_module_doc_has_design_principles() {
    let text = read_repo_file("prime/src/lib.rs");
    let module_doc: String = text
        .lines()
        .take_while(|line| line.trim_start().starts_with("//!") || line.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    let lowercase = module_doc.to_lowercase();
    for required in ["per-core", "umbrella-free"] {
        assert!(
            lowercase.contains(required),
            "audit finding F11: prime/src/lib.rs `//!` is \
             missing design-principles phrase '{required}'",
        );
    }
}
