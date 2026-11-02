//! Records the cached-attention kernel goldens under
//! `tests/fixtures/attn_split_decode/` that `attn_split_tests` compares
//! against. Run it on an unpatched export of `main`, once with no attention
//! feature and once with `metal-attn-split-decode`:
//!
//! `cargo run -p proxima-omega --features test-support --example record_attn_goldens`
//!
//! Rendering goes through `omega::msl::attn_golden_tests`, the same functions
//! the gate tests call.

use std::path::Path;
use std::process::ExitCode;

#[cfg(feature = "metal-wide-cooperative-reduce")]
use omega::msl::attn_golden_tests::{softmax_weights_golden_cases, softmax_weights_rendered};

use omega::msl::attn_golden_tests::{
    GOLDEN_PREFIX, golden_cases, golden_dir, preamble_of, rendered,
};

fn write_golden(path: &Path, text: &str) -> Result<(), String> {
    std::fs::write(path, text)
        .map_err(|error| format!("cannot write golden {}: {error}", path.display()))
}

fn record_main_goldens(directory: &Path) -> Result<(), String> {
    let cases = golden_cases();
    let Some((_, first_op, first_policy)) = cases.first() else {
        return Err("zero golden cases would record nothing".into());
    };
    let preamble = preamble_of(first_op, *first_policy);
    write_golden(
        &directory.join(format!("{GOLDEN_PREFIX}preamble.msl")),
        &preamble,
    )?;
    for (name, op, policy) in &cases {
        if preamble_of(op, *policy) != preamble {
            return Err(format!("{name}: one preamble must serve every golden case"));
        }
        write_golden(
            &directory.join(format!("{GOLDEN_PREFIX}{name}.msl")),
            &rendered(op, *policy),
        )?;
    }
    Ok(())
}

#[cfg(feature = "metal-wide-cooperative-reduce")]
fn record_softmax_weights_goldens(directory: &Path) -> Result<(), String> {
    let cases = softmax_weights_golden_cases();
    if cases.is_empty() {
        return Err("zero softmax weights golden cases would record nothing".into());
    }
    for (name, op) in &cases {
        write_golden(
            &directory.join(format!("main_{name}.msl")),
            &softmax_weights_rendered(op),
        )?;
    }
    Ok(())
}

#[cfg(not(feature = "metal-wide-cooperative-reduce"))]
fn record_softmax_weights_goldens(_directory: &Path) -> Result<(), String> {
    Ok(())
}

fn record(directory: &Path) -> Result<(), String> {
    std::fs::create_dir_all(directory)
        .map_err(|error| format!("cannot create {}: {error}", directory.display()))?;
    record_main_goldens(directory)?;
    record_softmax_weights_goldens(directory)
}

fn main() -> ExitCode {
    match record(&golden_dir()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}
