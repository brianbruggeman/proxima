//! Stages a second immediate prefault, an `MADV_WILLNEED` and an `mlock`
//! against one real checkpoint mapping in sequence, timing and
//! `mincore`-checking each rung. A near-zero elapsed time for a stage over a
//! multi-GB mapping (a real touch is >= 500 ms) means that stage's touch was
//! a no-op the optimizer elided.
//!
//! `PROXIMA_QWEN35MOE_GGUF` names the checkpoint.
//!
//! `cargo run -p proxima-model-interop --release --features std,metal --example residency_ladder`

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[cfg(all(feature = "metal", target_os = "macos"))]
#[path = "real_model_probe/mod.rs"]
mod real_model_probe;

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run() -> Result<(), String> {
    use proxima_model_interop::probes;

    let path = std::env::var("PROXIMA_QWEN35MOE_GGUF").unwrap_or_else(|_| {
        "/Users/brianbruggeman/.ollama/models/blobs/sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d".to_string()
    });
    let mapping = real_model_probe::map_checkpoint(&path, "PROXIMA_QWEN35MOE_GGUF")?;
    let bytes: &[u8] = &mapping;
    let stages = probes::residency_ladder(bytes)
        .map_err(|error| format!("the prefault pool builds and every chunk reports: {error}"))?;
    for stage in stages {
        println!(
            "{}: {:.1} ms, missing_pages={} of {}, contiguous_runs={}, first_missing_offset={:?}, last_missing_offset={:?}",
            stage.label,
            stage.elapsed.as_secs_f64() * 1000.0,
            stage.missing_pages,
            stage.total_pages,
            stage.contiguous_runs,
            stage.first_missing_offset,
            stage.last_missing_offset,
        );
        if let Some(note) = stage.note {
            eprintln!("{}: {note}", stage.label);
        }
    }
    Ok(())
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("residency_ladder: {message}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    println!("residency_ladder requires --features std,metal on macOS; skipped");
}
