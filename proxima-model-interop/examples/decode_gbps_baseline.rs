//! MEASUREMENT 0a/0b baseline: real gemma4-E2B decode, steady-state
//! wall-clock vs the CLEAN aggregate `metal_stage_totals().gpu_exec_ticks`
//! counter (never the per-dispatch stage-boundary path, which is known to
//! inflate 50-100x on this box -- `gemma4_dispatch_profile_widths.rs`'s own
//! doc names that floor-doubling defect).
//!
//! Reuses the ALREADY-WIRED `PROXIMA_DEBUG_METAL_STAGES` env-var convention
//! (`generate/load_model.rs::emit_token_breakdown`/`emit_token_breakdown_metal`)
//! -- this file adds zero new instrumentation, only a runnable entry point
//! over `LoadedModel::generate_streaming`, the same decode loop
//! `generate_with_serving_config`/`gemma4_correctness_gate.rs` already
//! exercise on the real checkpoint, given a real `on_token` instead of
//! `&mut |_| Continue` so `TokenEvent::elapsed_ms` (Instant-based, compiled
//! on every build) is readable without the diagnostic-only `instrument`
//! feature. Run with `PROXIMA_DEBUG_METAL_STAGES=1` set in the environment;
//! every decode step then emits one `token_breakdown_wall` and one
//! `token_breakdown_metal` line to stderr, parsed by the caller.
//! `PROXIMA_RUNS=<n>` (default `1`) repeats the same generation `n` times in
//! this one process, reusing the loaded model/runtime so runs after the
//! first skip step-0 pipeline compilation.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use core::ops::ControlFlow;
use std::fs::File;
#[cfg(feature = "instrument")]
use std::sync::Arc;
#[cfg(feature = "instrument")]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "instrument")]
use std::thread;
#[cfg(feature = "instrument")]
use std::time::Duration;
use std::time::Instant;

use memmap2::{Mmap, MmapOptions};
use proxima_gguf::parse_complete;
use proxima_gguf::types::GgmlType;
use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel, Phase, ServingConfig, TokenEvent};
#[cfg(feature = "instrument")]
use proxima_telemetry::export::Exporter;
#[cfg(feature = "instrument")]
use proxima_telemetry::recorder::Recorder;

/// FNV-1a 64-bit over `text`'s own UTF-8 bytes -- a cheap, dependency-free
/// content fingerprint so `PROXIMA_RUNS` arms and on/off `PROXIMA_COMMAND_BUFFER_CHUNKS`
/// arms can assert byte-identical generated text without diffing full strings
/// in every log line.
fn fnv64(text: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

const MODEL_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/\
sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

const DEFAULT_MAX_TOKENS: usize = 48;

// mirrors `gemma4_dispatch_profile_widths.rs`'s `install_console_telemetry` --
// `report_encoder_split`/`report_op_timings` emit via `info!`, which is
// otherwise silent: this example had no console sink, so those events never
// reached stderr even with PROXIMA_METAL_ENCODER_SPLIT_AT/DISPATCH_PROFILE_STEP set.
// `proxima-telemetry` is only pulled in by `instrument`
// (`dep:proxima-telemetry`, this crate's own `Cargo.toml`) -- a build
// without it has no recorder to install, and no `token_breakdown`/
// `report_*` events compiled anywhere in this crate to drain.
#[cfg(feature = "instrument")]
fn install_console_telemetry() -> Arc<AtomicUsize> {
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
    drained_total
}

/// Owner brief item (3): one `capture_binary` line, printed once at process
/// start -- the running binary's own identity (path + content md5, not a
/// version string that could drift from the bytes actually executing) and
/// the exact feature set this build compiled with, read from `cfg!` rather
/// than restated by hand so it can never disagree with what actually built.
#[cfg(feature = "instrument")]
fn print_capture_binary() {
    use md5::{Digest, Md5};
    let argv0 = std::env::current_exe().expect("resolve running binary path");
    let bytes = std::fs::read(&argv0).expect("read running binary for md5");
    let digest = Md5::digest(&bytes);
    let md5_hex = digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
    eprintln!(
        "capture_binary argv0={argv0:?} md5={md5_hex} features=[metal={}, instrument={}, metal-fuse-attn-decode={}, identity-copy-alias={}, metal-output-placement={}, reduce-epilogue-fusion={}]",
        cfg!(feature = "metal"),
        cfg!(feature = "instrument"),
        cfg!(feature = "metal-fuse-attn-decode"),
        cfg!(feature = "identity-copy-alias"),
        cfg!(feature = "metal-output-placement"),
        cfg!(feature = "reduce-epilogue-fusion"),
    );
}

fn main() {
    #[cfg(feature = "instrument")]
    print_capture_binary();
    // attribution slice (2026-09-22, OWNER_BRIEF_dominant_cost): example-only
    // knob so the console sink's per-event format/write can be isolated from
    // the macro's own field evaluation; unset keeps today's behavior. No-op
    // without `instrument` -- there is no recorder to install and no
    // `token_breakdown`/`report_*` event compiled anywhere in this crate.
    #[cfg(feature = "instrument")]
    let _drained_total = if std::env::var_os("PROXIMA_CONSOLE_TELEMETRY").as_deref() == Some(std::ffi::OsStr::new("0"))
    {
        None
    } else {
        Some(install_console_telemetry())
    };
    if std::env::var_os("PROXIMA_DEBUG_METAL_STAGES").is_none() {
        eprintln!(
            "decode_gbps_baseline: PROXIMA_DEBUG_METAL_STAGES not set -- \
             per-step token_breakdown_wall/token_breakdown_metal lines will \
             not be emitted; set it in the environment before running."
        );
    }

    let file = File::open(MODEL_PATH).expect("open gemma4-E2B blob");
    // SAFETY: `file` is dropped at the end of this scope, but the mapping
    // stays valid past that -- POSIX `mmap`/`munmap` semantics, same
    // pattern every real-checkpoint example in this crate already uses.
    let bytes: Mmap = unsafe { MmapOptions::new().map(&file) }.expect("map gemma4-E2B blob");
    let parsed = parse_complete(&bytes).expect("parse gemma4-E2B header");
    let model = LoadedModel::load(&parsed, &bytes).expect("bind gemma4-E2B");

    // attribution3 followon, intervention 3 (2026-09-22, owner redirect):
    // `ServingConfig::default()` carries `dispatch_type: DispatchType::Serial`
    // since 3afb3db37 (qwen35moe residency boundary), a bundled side effect
    // of that commit, not a gemma4-measured choice -- the comment at
    // `residency_caches.rs`'s `evaluate_with_expert_sources` names the real
    // reason (`concurrent`'s hazard schedule was "proven only for the
    // placed single-range program, not hybrid recurrent graphs" like
    // qwen35's GDN path), which does not describe gemma4's decode graph.
    // `PROXIMA_DISPATCH=concurrent` flips this one example's arm without
    // touching the shared default.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let dispatch_type = match std::env::var("PROXIMA_DISPATCH").as_deref() {
        Ok("concurrent") => omega::DispatchType::Concurrent,
        Ok("serial") | Err(_) => omega::DispatchType::Serial,
        Ok(other) => panic!("PROXIMA_DISPATCH={other}: expected `serial` or `concurrent`"),
    };
    let serving_config = ServingConfig {
        gpu_layers: GPU_LAYERS_ALL,
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        reasoning_budget: 0,
        #[cfg(all(feature = "metal", target_os = "macos"))]
        dispatch_type,
        ..ServingConfig::default()
    };

    // Chat-template prompt (same convention as `gemma4_correctness_gate.rs`)
    // chosen because its locked greedy completion runs the full 46-token
    // budget without an early EOS -- a short factual completion like
    // "The capital of France is" hits EOS after ~5 tokens, too few steps
    // for a steady-state decode average. `PROXIMA_PROMPT` overrides it
    // verbatim -- the caller supplies any chat-template markers, this
    // example does not add or infer any.
    let default_prompt = "<|turn>user\nWhich of these is smaller in size: a hippopotamus or a large office building?<turn|>\n<|turn>model\n";
    let prompt_override = std::env::var("PROXIMA_PROMPT").ok();
    let prompt: &str = prompt_override.as_deref().unwrap_or(default_prompt);
    let max_tokens: usize = std::env::var("PROXIMA_MAX_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_MAX_TOKENS);
    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)
        .expect("builds vocab via the current gemma4 dispatch");
    let prompt_token_count = proxima_tokenizer::encode(prompt, &vocab)
        .expect("tokenize prompt for cached_len accounting")
        .len();
    eprintln!(
        "decode_gbps_baseline run=start prompt={prompt:?} prompt_token_count={prompt_token_count} max_tokens={max_tokens}"
    );

    // owner rollout check (2026-09-22, intervention6): K=1 vs K=8 chunked
    // command buffers, WITHOUT stage logging and WITHOUT the `instrument`
    // feature -- `PROXIMA_RUNS` reuses this one process's already-loaded
    // model and runtime across `n` generations so run `1..n` skip step 0's
    // pipeline-compilation cost (`generate_streaming`'s plan cache is warm
    // by run 2), giving a serving-shaped measure instead of a whole-run
    // time that bundles one-time compilation into steady-state decode.
    let runs: usize = std::env::var("PROXIMA_RUNS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);

    for run_index in 0..runs {
        // `generate_streaming`'s own `TokenEvent::elapsed_ms` is Instant-based
        // and compiled on every build that reaches this call, never gated
        // behind `instrument` -- see `TokenEvent`'s own field doc on why.
        let mut prefill_elapsed_ms: u64 = 0;
        let mut on_token = |event: TokenEvent<'_>| {
            if matches!(event.phase, Phase::Prefill { .. }) {
                prefill_elapsed_ms = event.elapsed_ms;
            }
            ControlFlow::Continue(())
        };
        let start = Instant::now();
        let (token_ids, text, stopped_by_eos) = model
            .generate_streaming(prompt, max_tokens, serving_config, &mut on_token)
            .expect("greedy decode on the real gemma4-E2B checkpoint");
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        let tokens_generated = token_ids.len();
        let text_hash = fnv64(&text);

        // `(wall - first-step time) / (tokens - 1)`: prefill (step 0, 26
        // prompt tokens here) is included identically in both K=1 and K=8
        // arms, so subtracting its own measured `elapsed_ms` isolates the
        // per-decode-step cost the K sweep is actually meant to move.
        if tokens_generated > 1 {
            let decode_ms_per_token =
                (wall_ms - prefill_elapsed_ms as f64) / (tokens_generated - 1) as f64;
            eprintln!(
                "decode_gbps_baseline run=done run_index={run_index} wall_ms={wall_ms:.3} \
                 tokens_generated={tokens_generated} prompt_token_count={prompt_token_count} \
                 text_hash={text_hash:016x} stopped_by_eos={stopped_by_eos} \
                 decode_ms_per_token={decode_ms_per_token:.3} text={text:?}"
            );
        } else {
            eprintln!(
                "decode_gbps_baseline run=done run_index={run_index} wall_ms={wall_ms:.3} \
                 tokens_generated={tokens_generated} prompt_token_count={prompt_token_count} \
                 text_hash={text_hash:016x} stopped_by_eos={stopped_by_eos} text={text:?}"
            );
        }
    }
}
