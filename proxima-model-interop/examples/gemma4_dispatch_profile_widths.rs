//! Attributes ONE real gemma4-E2B forward's GPU time onto op/kernel types
//! using the PRODUCTION per-dispatch GPU-timestamp path
//! (`execute_plan_with_placements_dispatch_timed`,
//! `omega/src/metal/dispatch_timed_and_classify.rs:713`) against the SAME
//! single batched command buffer `run_decode_loop` submits in production --
//! never an isolated per-op command buffer (that shape's own floor-doubling
//! defect is `docs/microbench/discipline.md`'s SLICE 3 finding, "Two honest
//! reads" section).
//!
//! Reuses `decode.rs`'s own `PROXIMA_METAL_DISPATCH_PROFILE_STEP` env-var
//! convention (already wired, `generate/decode.rs:4126-4157`) and
//! `report_op_timings`'s own `info!`/`eprintln!` emission
//! (`generate/load_model.rs:74`) -- this file adds zero new instrumentation,
//! only a runnable entry point plus a console telemetry sink so the
//! existing `info!` events are visible (the same `install_stdout_telemetry`
//! pattern `proxima-model-interop/src/bind.rs:5103`'s test-only helper
//! already uses, duplicated here because that helper is `#[cfg(test)]`
//! private to the crate and unreachable from an external `examples/`
//! binary).
//!
//! Profiles step 0 (prefill -- real observed `new_count`, BOS-inflated to 2
//! for a one-word prompt, `docs/microbench/discipline.md`'s own SLICE 1
//! finding) and step 1 (the first genuine decode step, `new_count=1`) via
//! two SEPARATE `generate_with_serving_config` calls, each naming a
//! different step to instrument so only that step pays for
//! `MTLCounterSampleBuffer` sampling -- every other step runs the
//! unmodified production path.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::env;
use std::fs::File;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use memmap2::{Mmap, MmapOptions};
use proxima_gguf::parse_complete;
use proxima_gguf::types::GgmlType;
use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel, ServingConfig};
use proxima_telemetry::export::Exporter;
use proxima_telemetry::recorder::Recorder;

const MODEL_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/\
sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

fn install_console_telemetry() -> (Arc<Recorder>, Arc<AtomicUsize>) {
    proxima_telemetry::emit::global::install(proxima_telemetry::emit::EnvFilter::parse("debug"));
    let recorder = Recorder::builder()
        .ring_capacity(65536)
        .export(Exporter::std())
        .expect("console exporter installs")
        .install()
        .expect("telemetry recorder installs");
    let drained_total = Arc::new(AtomicUsize::new(0));
    let pump_recorder = Arc::clone(&recorder);
    let pump_total = Arc::clone(&drained_total);
    thread::Builder::new()
        .name("dispatch-profile-drain".to_string())
        .spawn(move || {
            loop {
                let drained = pump_recorder.drain();
                pump_total.fetch_add(drained, Ordering::Relaxed);
                thread::sleep(Duration::from_millis(5));
            }
        })
        .expect("spawn telemetry drain thread");
    (recorder, drained_total)
}

fn profile_step(model: &LoadedModel<'_>, step: usize, max_tokens: usize, prompt: &str) {
    let serving_config = ServingConfig {
        context_length: 64,
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        gpu_layers: GPU_LAYERS_ALL,
        reasoning_budget: 0,
        ..ServingConfig::default()
    };
    // SAFETY: single-threaded example, no concurrent reader of this var --
    // same justification `proxima-model-interop/src/bind.rs`'s own
    // `profiles_one_real_decode_step_by_per_op_gpu_time` test gives.
    unsafe {
        env::set_var("PROXIMA_METAL_DISPATCH_PROFILE_STEP", step.to_string());
    }
    let generated = model
        .generate_with_serving_config(prompt, max_tokens, serving_config)
        .expect("generate through the metal backend");
    unsafe {
        env::remove_var("PROXIMA_METAL_DISPATCH_PROFILE_STEP");
    }
    eprintln!(
        "gemma4_dispatch_profile_widths step={step} max_tokens={max_tokens} tokens_generated={} stopped_by_eos={}",
        generated.0.len(),
        generated.2,
    );
}

fn main() {
    let (_recorder, drained_total) = install_console_telemetry();

    let file = File::open(MODEL_PATH).expect("open gemma4-E2B blob");
    // SAFETY: `file` is dropped at the end of this scope, but the mapping
    // stays valid past that -- POSIX `mmap`/`munmap` semantics, same
    // pattern every real-checkpoint example in this crate already uses.
    let bytes: Mmap = unsafe { MmapOptions::new().map(&file) }.expect("map gemma4-E2B blob");
    let parsed = parse_complete(&bytes).expect("parse gemma4-E2B header");
    let model = LoadedModel::load(&parsed, &bytes).expect("bind gemma4-E2B");

    // BOS-inflated one-word prompt (SLICE 1's own `prompt_for_width`
    // finding: a real minimal prompt tokenizes to width=2, never width=1)
    // -- this is `new_count=2` at step 0, the WIDTH-2 arm.
    let prompt = "Paris";

    eprintln!("gemma4_dispatch_profile_widths run=prefill_width step=0 (new_count observed, BOS-inflated)");
    profile_step(&model, 0, 1, prompt);

    eprintln!("gemma4_dispatch_profile_widths run=decode_width1 step=1 (new_count=1, first real decode step)");
    profile_step(&model, 1, 2, prompt);

    let total = drained_total.load(Ordering::Relaxed) + _recorder.drain();
    eprintln!("gemma4_dispatch_profile_widths telemetry_records_drained={total}");
    assert!(total > 0, "degenerate control: no telemetry emitted");
}
