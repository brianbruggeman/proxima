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
//! every decode EVALUATION then emits one `token_breakdown_wall` and one
//! `token_breakdown_metal` line to stderr, parsed by the caller. Speculation
//! runs at the production default, so a verify evaluation commits several
//! tokens and the following steps evaluate nothing: lines are per evaluation,
//! `step_time` and `decode_ms_per_token` are per token.
//! `PROXIMA_RUNS=<n>` (default `1`) repeats the same generation `n` times in
//! this one process, reusing the loaded model/runtime so runs after the
//! first skip step-0 pipeline compilation.
//! `PROXIMA_DECODE_SPECULATIVE=none` turns speculative decoding off
//! (`SpeculativeConfig::none()`); unset keeps the serving default, ngram-simple.
//! A half-width KV cache (`PROXIMA_KV_CACHE_TYPE=f16`) requires it, because a
//! verify step reads the cache through a kernel that only reads f32.
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
use proxima_model_interop::{
    GPU_LAYERS_ALL, LoadedModel, Phase, PromptCacheConfig, ServingConfig, SpeculativeConfig,
    TokenEvent,
};
#[cfg(feature = "instrument")]
use proxima_telemetry::export::Exporter;
#[cfg(feature = "instrument")]
use proxima_telemetry::recorder::Recorder;

/// User and system CPU milliseconds the whole process has used so far
/// (`getrusage(RUSAGE_SELF)`), so the difference across one generation is the
/// CPU that generation cost every thread of this process.
fn process_cpu_ms() -> (f64, f64) {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `usage` is a valid, writable `rusage`; `RUSAGE_SELF` takes no other pointer.
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    assert_eq!(status, 0, "getrusage(RUSAGE_SELF) failed");
    // SAFETY: `getrusage` returned 0, so it filled the struct.
    let usage = unsafe { usage.assume_init() };
    let millis = |time: libc::timeval| time.tv_sec as f64 * 1000.0 + time.tv_usec as f64 / 1000.0;
    (millis(usage.ru_utime), millis(usage.ru_stime))
}

/// Resident set and physical footprint of this process right now, in bytes
/// (`proc_pid_rusage`): the current values, not the peaks `/usr/bin/time -l`
/// reports at exit. Footprint is what `footprint(1)` and Activity Monitor's
/// memory column read. `(0, 0)` where the call is unavailable.
fn process_memory_bytes() -> (u64, u64) {
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v2>::zeroed();
    // SAFETY: `RUSAGE_INFO_V2` fills a `rusage_info_v2`; the C signature takes the struct's address cast to `rusage_info_t *`.
    let status = unsafe {
        libc::proc_pid_rusage(std::process::id() as libc::c_int, libc::RUSAGE_INFO_V2, info.as_mut_ptr().cast())
    };
    if status != 0 {
        return (0, 0);
    }
    // SAFETY: the call returned 0, so the struct is filled.
    let info = unsafe { info.assume_init() };
    (info.ri_resident_size, info.ri_phys_footprint)
}

/// Generated tokens before the steady-state memory samples start: the first
/// tokens still carry the prefill transient.
const STEADY_FROM_TOKEN: usize = 10;

fn median_bytes(samples: &[u64]) -> u64 {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    sorted.get(sorted.len() / 2).copied().unwrap_or(0)
}

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

/// `PROXIMA_DECODE_MODEL_GGUF` names the checkpoint to decode; unset keeps the
/// gemma4 E2B blob every earlier baseline binary measured.
fn model_path() -> String {
    std::env::var("PROXIMA_DECODE_MODEL_GGUF").unwrap_or_else(|_| MODEL_PATH.to_string())
}

/// Bytes Metal has allocated on the system device right now; zero where there
/// is no Metal device. Sampled at every token, the maximum is the run's peak.
fn gpu_allocated_bytes() -> u64 {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        omega::metal::current_allocated_size().unwrap_or(0)
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        0
    }
}

const DEFAULT_MAX_TOKENS: usize = 48;

// proxima-telemetry is only a dependency under `instrument`, so events
// otherwise had no console sink to reach.
#[cfg(feature = "instrument")]
fn install_console_telemetry() -> (
    Arc<AtomicUsize>,
    Arc<Recorder<proxima_telemetry::clock::GlobalClock>>,
) {
    // result lines stay at info even when RUST_LOG is unset
    let rust_log = std::env::var("RUST_LOG").unwrap_or_default();
    let filter = if rust_log.is_empty() {
        "decode_gbps_baseline=info".to_string()
    } else {
        format!("{rust_log},decode_gbps_baseline=info")
    };
    proxima_telemetry::emit::global::install(proxima_telemetry::emit::EnvFilter::parse(&filter));
    let exporter = match std::env::var("PROXIMA_TELEMETRY_FILE") {
        Ok(path) => Exporter::fan(vec![Exporter::std(), Exporter::file(path)])
            .expect("console+file fan composes"),
        Err(_) => Exporter::std(),
    };
    let footprint_before_bytes = process_memory_bytes().1;
    let recorder = Recorder::builder()
        .ring_capacity(65536)
        .export(exporter)
        .expect("console exporter installs")
        .install()
        .expect("telemetry recorder installs");
    proxima_telemetry::info!(
        footprint_before_bytes,
        footprint_after_bytes = process_memory_bytes().1,
        "telemetry_recorder_footprint"
    );
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
    (drained_total, recorder)
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
    let md5_hex = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    proxima_telemetry::info!(
        argv0 = ?argv0,
        md5 = %md5_hex,
        metal = cfg!(feature = "metal"),
        instrument = cfg!(feature = "instrument"),
        metal_fuse_attn_decode = cfg!(feature = "metal-fuse-attn-decode"),
        identity_copy_alias = cfg!(feature = "identity-copy-alias"),
        metal_output_placement = cfg!(feature = "metal-output-placement"),
        reduce_epilogue_fusion = cfg!(feature = "reduce-epilogue-fusion"),
        "capture_binary"
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
    // the pump is never joined, so drain to empty or the last result line is lost
    #[cfg(feature = "instrument")]
    let _telemetry = if std::env::var_os("PROXIMA_CONSOLE_TELEMETRY").as_deref()
        == Some(std::ffi::OsStr::new("0"))
    {
        None
    } else {
        Some(install_console_telemetry())
    };
    // `PROXIMA_DEBUG_METAL_STAGES` no longer gates whether
    // `token_breakdown_wall`/`token_breakdown_metal` fire -- they are
    // unconditional `info!`/`debug!` events now; RUST_LOG (raised via
    // `install_console_telemetry`'s `EnvFilter::parse("debug")`) is what
    // controls visibility.

    let model_file = model_path();
    let file = File::open(&model_file).unwrap_or_else(|error| panic!("open {model_file}: {error}"));
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
    // 2026-09-25 bucket-width sweep (owner brief prefill599/bucket): default
    // off, so every other caller of this example keeps `ServingConfig::
    // default()`'s measured `kv_bucket_tokens: 32` (comment at that field's
    // own default, CARD 6.3's quiet-round table). Unset keeps 32; set but
    // unparseable or non-positive is an explicit panic naming the value,
    // matching `PROXIMA_DISPATCH`'s own invalid-arm handling above --
    // `apply_serving_config`'s own `kv_bucket_tokens < 1` rejection
    // (`serving.rs`) would otherwise surface as a less legible interop error.
    let kv_bucket_tokens: usize = match std::env::var("PROXIMA_KV_BUCKET_TOKENS") {
        Ok(value) => value.parse().unwrap_or_else(|_| {
            panic!("PROXIMA_KV_BUCKET_TOKENS={value}: expected a positive integer")
        }),
        Err(_) => ServingConfig::default().kv_bucket_tokens,
    };
    let numeric_policy = match std::env::var("PROXIMA_EPILOGUE_SOURCES").as_deref() {
        Ok("0") => ServingConfig::default().numeric_policy.with_epilogue_sources(false),
        Ok("1") => ServingConfig::default().numeric_policy.with_epilogue_sources(true),
        Ok(other) => panic!("PROXIMA_EPILOGUE_SOURCES={other}: expected `0` or `1`"),
        Err(_) => ServingConfig::default().numeric_policy,
    };
    let resident_prefill_plan_bytes: usize = match std::env::var("PROXIMA_RESIDENT_PREFILL_PLAN_BYTES") {
        Ok(value) => value.parse().unwrap_or_else(|_| {
            panic!("PROXIMA_RESIDENT_PREFILL_PLAN_BYTES={value}: expected a byte count")
        }),
        Err(_) => ServingConfig::default().resident_prefill_plan_bytes,
    };
    let kv_cache_type = match std::env::var("PROXIMA_KV_CACHE_TYPE").as_deref() {
        Ok("f16") => GgmlType::F16,
        Ok("f32") | Err(_) => GgmlType::F32,
        Ok(other) => panic!("PROXIMA_KV_CACHE_TYPE={other}: expected `f32` or `f16`"),
    };
    let speculative = match std::env::var("PROXIMA_DECODE_SPECULATIVE").as_deref() {
        Ok("none") => SpeculativeConfig::none(),
        Ok(other) => panic!("PROXIMA_DECODE_SPECULATIVE={other}: expected `none` or unset"),
        Err(_) => SpeculativeConfig::default(),
    };
    let warm_model_buffers_at_load = match std::env::var("PROXIMA_SERVING_WARM_MODEL_BUFFERS_AT_LOAD").as_deref() {
        Ok("true") => true,
        Ok("false") => false,
        Ok(other) => panic!("PROXIMA_SERVING_WARM_MODEL_BUFFERS_AT_LOAD={other}: expected `true` or `false`"),
        Err(_) => ServingConfig::default().warm_model_buffers_at_load,
    };
    let serving_config = ServingConfig {
        warm_model_buffers_at_load,
        speculative,
        gpu_layers: GPU_LAYERS_ALL,
        numeric_policy,
        kv_cache_key_quant: kv_cache_type,
        kv_cache_value_quant: kv_cache_type,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        reasoning_budget: 0,
        kv_bucket_tokens,
        resident_prefill_plan_bytes,
        prompt_cache: PromptCacheConfig::off(),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        dispatch_type,
        ..ServingConfig::default()
    };

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        let warm_started = Instant::now();
        let warmed_buffers = model
            .warm_resident_buffers(&serving_config)
            .expect("declare the model buffers to the driver");
        eprintln!(
            "decode_gbps_baseline warm_model_buffers={warm_model_buffers_at_load} warmed_buffers={warmed_buffers} warm_ms={:.3}",
            warm_started.elapsed().as_secs_f64() * 1000.0
        );
    }

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
    // `proxima_telemetry` is only a dependency under `instrument`
    // (`proxima-model-interop/Cargo.toml`'s own `metal` feature list does not
    // pull `dep:proxima-telemetry`) -- this example's `required-features =
    // ["std", "metal"]` means it must build WITHOUT `instrument`, so every
    // converted event here keeps the pre-existing `eprintln!` as the
    // `not(instrument)` fallback, matching `decode.rs`'s own established
    // `prefill_batch`/`token_stages` split.
    #[cfg(feature = "instrument")]
    proxima_telemetry::info!(
        run = "start",
        prompt = ?prompt,
        prompt_token_count = prompt_token_count as u64,
        max_tokens = max_tokens as u64,
        "decode_gbps_baseline"
    );
    #[cfg(not(feature = "instrument"))]
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

    // OWNER_BRIEF_prefill_correction (2026-09-23): per-step deltas from the
    // SAME generation's own `TokenEvent::elapsed_ms` stream, so a caller can
    // read the first-decode-step cost separately from later steps within one
    // run instead of fitting it across prompts of different lengths. A step
    // is a token, not an evaluation: tokens a verify evaluation committed
    // arrive back to back, so those later steps read ~0 ms.
    let step_times = std::env::var_os("PROXIMA_STEP_TIMES").is_some();

    let mut gpu_peak_bytes: u64 = 0;
    for run_index in 0..runs {
        // `generate_streaming`'s own `TokenEvent::elapsed_ms` is Instant-based
        // and compiled on every build that reaches this call, never gated
        // behind `instrument` -- see `TokenEvent`'s own field doc on why.
        let mut prefill_elapsed_ms: u64 = 0;
        let mut previous_elapsed_ms: u64 = 0;
        let mut token_events: usize = 0;
        let mut steady_rss: Vec<u64> = Vec::new();
        let mut steady_footprint: Vec<u64> = Vec::new();
        let mut steady_gpu: Vec<u64> = Vec::new();
        let mut on_token = |event: TokenEvent<'_>| {
            if matches!(event.phase, Phase::Prefill { .. }) {
                prefill_elapsed_ms = event.elapsed_ms;
            }
            token_events += 1;
            let gpu_now = gpu_allocated_bytes();
            gpu_peak_bytes = gpu_peak_bytes.max(gpu_now);
            if token_events > STEADY_FROM_TOKEN {
                let (rss_now, footprint_now) = process_memory_bytes();
                steady_rss.push(rss_now);
                steady_footprint.push(footprint_now);
                steady_gpu.push(gpu_now);
            }
            if step_times {
                let phase = if matches!(event.phase, Phase::Prefill { .. }) {
                    "prefill"
                } else {
                    "decode"
                };
                let step_ms = event.elapsed_ms.saturating_sub(previous_elapsed_ms);
                #[cfg(feature = "instrument")]
                proxima_telemetry::info!(
                    run_index = run_index as u64,
                    step = event.step as u64,
                    step_ms,
                    phase,
                    "step_time"
                );
                #[cfg(not(feature = "instrument"))]
                eprintln!(
                    "step_time run_index={run_index} step={} step_ms={step_ms} phase={phase}",
                    event.step,
                );
                previous_elapsed_ms = event.elapsed_ms;
            }
            ControlFlow::Continue(())
        };
        let cpu_before = process_cpu_ms();
        let start = Instant::now();
        let (token_ids, text, stopped_by_eos) = model
            .generate_streaming(prompt, max_tokens, serving_config, &mut on_token)
            .expect("greedy decode on the real gemma4-E2B checkpoint");
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        let cpu_after = process_cpu_ms();
        let cpu_user_ms = cpu_after.0 - cpu_before.0;
        let cpu_sys_ms = cpu_after.1 - cpu_before.1;
        let cpu_ms = cpu_user_ms + cpu_sys_ms;
        let cpu_pct = cpu_ms / wall_ms * 100.0;
        gpu_peak_bytes = gpu_peak_bytes.max(gpu_allocated_bytes());
        let steady_rss_bytes = median_bytes(&steady_rss);
        let steady_footprint_bytes = median_bytes(&steady_footprint);
        let steady_gpu_bytes = median_bytes(&steady_gpu);
        let steady_samples = steady_rss.len();
        let tokens_generated = token_ids.len();
        let text_hash = fnv64(&text);
        eprintln!("decode_gbps_baseline token_ids run_index={run_index} ids={token_ids:?}");

        // `(wall - first-step time) / (tokens - 1)`: prefill (step 0, 26
        // prompt tokens here) is included identically in both K=1 and K=8
        // arms, so subtracting its own measured `elapsed_ms` isolates the
        // per-committed-token decode cost the K sweep is actually meant to
        // move; with speculation it is wall per token, not per evaluation.
        if tokens_generated > 1 {
            let decode_ms_per_token =
                (wall_ms - prefill_elapsed_ms as f64) / (tokens_generated - 1) as f64;
            #[cfg(feature = "instrument")]
            proxima_telemetry::info!(
                run = "done",
                run_index = run_index as u64,
                wall_ms,
                cpu_user_ms,
                cpu_sys_ms,
                cpu_ms,
                cpu_pct,
                tokens_generated = tokens_generated as u64,
                prompt_token_count = prompt_token_count as u64,
                text_hash = %format!("{text_hash:016x}"),
                stopped_by_eos,
                decode_ms_per_token,
                ttft_ms = prefill_elapsed_ms,
                gpu_peak_bytes,
                steady_rss_bytes,
                steady_footprint_bytes,
                steady_gpu_bytes,
                steady_samples = steady_samples as u64,
                text = %text,
                "decode_gbps_baseline"
            );
            #[cfg(not(feature = "instrument"))]
            eprintln!(
                "decode_gbps_baseline run=done run_index={run_index} wall_ms={wall_ms:.3} \
                 cpu_user_ms={cpu_user_ms:.3} cpu_sys_ms={cpu_sys_ms:.3} cpu_ms={cpu_ms:.3} cpu_pct={cpu_pct:.2} \
                 tokens_generated={tokens_generated} prompt_token_count={prompt_token_count} \
                 text_hash={text_hash:016x} stopped_by_eos={stopped_by_eos} \
                 decode_ms_per_token={decode_ms_per_token:.3} ttft_ms={prefill_elapsed_ms} \
                 gpu_peak_bytes={gpu_peak_bytes} steady_rss_bytes={steady_rss_bytes} \
                 steady_footprint_bytes={steady_footprint_bytes} steady_gpu_bytes={steady_gpu_bytes} \
                 steady_samples={steady_samples} text={text:?}"
            );
        } else {
            #[cfg(feature = "instrument")]
            proxima_telemetry::info!(
                run = "done",
                run_index = run_index as u64,
                wall_ms,
                cpu_user_ms,
                cpu_sys_ms,
                cpu_ms,
                cpu_pct,
                tokens_generated = tokens_generated as u64,
                prompt_token_count = prompt_token_count as u64,
                text_hash = %format!("{text_hash:016x}"),
                stopped_by_eos,
                ttft_ms = prefill_elapsed_ms,
                gpu_peak_bytes,
                steady_rss_bytes,
                steady_footprint_bytes,
                steady_gpu_bytes,
                steady_samples = steady_samples as u64,
                text = %text,
                "decode_gbps_baseline"
            );
            #[cfg(not(feature = "instrument"))]
            eprintln!(
                "decode_gbps_baseline run=done run_index={run_index} wall_ms={wall_ms:.3} \
                 cpu_user_ms={cpu_user_ms:.3} cpu_sys_ms={cpu_sys_ms:.3} cpu_ms={cpu_ms:.3} cpu_pct={cpu_pct:.2} \
                 tokens_generated={tokens_generated} prompt_token_count={prompt_token_count} \
                 text_hash={text_hash:016x} stopped_by_eos={stopped_by_eos} ttft_ms={prefill_elapsed_ms} \
                 gpu_peak_bytes={gpu_peak_bytes} steady_rss_bytes={steady_rss_bytes} \
                 steady_footprint_bytes={steady_footprint_bytes} steady_gpu_bytes={steady_gpu_bytes} \
                 steady_samples={steady_samples} text={text:?}"
            );
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            let (archive_hits, archive_stores) = omega::metal::pipeline_disk_cache_counts();
            eprintln!(
                "decode_gbps_baseline pipeline_disk_cache run_index={run_index} hits={archive_hits} stores={archive_stores}"
            );
        }
        // synchronous flush point: the background pump thread above only
        // wakes every 5ms, and one `drain()` call is one bounded batch pass
        // (`drain.batch` per ring, not "until empty") -- looping until it
        // returns 0 is every other example's own drain-to-shutdown idiom.
        #[cfg(feature = "instrument")]
        if let Some((_, recorder)) = _telemetry.as_ref() {
            while recorder.drain() > 0 {}
        }
    }
}
