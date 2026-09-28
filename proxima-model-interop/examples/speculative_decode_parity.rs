//! P14 hard parity probe for gemma4-E2B's default-off speculative decode
//! loop: for each of TWO `ServingConfig`s -- plain greedy, and a genuinely
//! sampling config (temperature, top-k/top-p/min-p, repeat penalty, a fixed
//! seed) -- runs the SAME prompt with `PROXIMA_SPECULATIVE_DECODE` off then
//! on, and reports whether the two token id streams are byte-identical --
//! not library surface, a one-shot diagnostic (same convention as
//! `gemma4_real_weight_parity.rs`). The sampled config is the load-bearing
//! case: it is only a parity proof at all once `decode.rs`'s speculative
//! verify branch selects every row through the SAME `select_decoded_token`
//! path (temperature, penalties, and the shared seeded `rng`) the
//! non-speculative branch uses, rather than a raw greedy argmax.
//!
//! `identical = true` alone is a degenerate control: it cannot distinguish
//! "drafts were proposed, verified, and matched the target model" from
//! "speculation never actually fired, so the ON run just re-ran the OFF
//! path." To rule the second case out, this probe installs a telemetry
//! recorder (the same `Recorder::builder().export(..).install()` convention
//! `decode_gbps_baseline.rs` uses) and captures `decode.rs`'s own
//! `speculative_verify` `debug!` event in memory, then asserts the ON run
//! actually emitted at least one, for BOTH configs.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use memmap2::{Mmap, MmapOptions};
use proxima_gguf::parse_complete;
use proxima_gguf::types::GgmlType;
use proxima_model_interop::{LoadedModel, ServingConfig};
use proxima_telemetry::emit::{EnvFilter, global};
use proxima_telemetry::export::Exporter;
use proxima_telemetry::log::LogBody;
use proxima_telemetry::pipes::{InMemoryPipe, into_telemetry_handle};
use proxima_telemetry::recorder::Recorder;
use proxima_telemetry::tag::ScalarValue;

/// Installs the process-default recorder, fanned to a file sink (the
/// `decode.rs` `speculative_verify` line is human-readable there too) and an
/// in-memory capture this probe reads back. A bare `"debug"` filter floor
/// applies globally -- `decode.rs`'s event lives under the library's own
/// module path, not this binary's, so a target-scoped rule (as
/// `decode_gbps_baseline.rs` uses for its own `info!` lines) would miss it.
fn install_telemetry(log_path: &std::path::Path) -> (InMemoryPipe, std::sync::Arc<Recorder>) {
    let filter = env::var("RUST_LOG").unwrap_or_else(|_| "debug".to_string());
    global::install(EnvFilter::parse(&filter));
    let capture = InMemoryPipe::new();
    let exporter = Exporter::fan(vec![
        Exporter::pipe(into_telemetry_handle(capture.clone())),
        Exporter::file(log_path),
    ])
    .expect("fan of capture + file sink composes");
    let recorder = Recorder::builder()
        .ring_capacity(65536)
        .export(exporter)
        .expect("recorder export installs")
        .install()
        .expect("recorder installs as process default");
    (capture, recorder)
}

/// `verified.accepted` from every captured `speculative_verify` event --
/// `decode.rs` tags it `accepted = verified.accepted as u64`.
fn accepted_field(record: &proxima_telemetry::log::LogRecord) -> Option<u64> {
    record.attrs.iter().find_map(|tag| match tag {
        proxima_telemetry::tag::Tag::Scalar {
            key: "accepted",
            value: ScalarValue::U64(accepted),
        } => Some(*accepted),
        _ => None,
    })
}

fn speculative_verify_events(capture: &InMemoryPipe) -> Vec<proxima_telemetry::log::LogRecord> {
    capture
        .logs()
        .into_iter()
        .filter(|record| record.body == LogBody::Text("speculative_verify"))
        .collect()
}

/// One config's own OFF-then-ON pair, run against the same model and
/// prompt: [`ServingConfig`] is `Copy`, so OFF and ON see byte-identical
/// input other than the env var this probe itself toggles.
struct PairResult {
    off_ids: Vec<u32>,
    on_ids: Vec<u32>,
    off_elapsed_ms: f64,
    on_elapsed_ms: f64,
    speculative_verify_steps: usize,
    accepted_total: u64,
}

/// `off_serving_config` and `on_serving_config` are equal for every caller
/// except `--seed-mismatch-control`'s sampled block, where the ON run is
/// deliberately reseeded -- the point of that control is that the two runs
/// must NOT reproduce each other.
fn run_pair(
    model: &LoadedModel,
    prompt: &str,
    max_tokens: usize,
    off_serving_config: ServingConfig,
    on_serving_config: ServingConfig,
    capture: &InMemoryPipe,
    recorder: &Recorder,
) -> PairResult {
    unsafe {
        env::remove_var("PROXIMA_SPECULATIVE_DECODE");
    }
    let off_started = Instant::now();
    let (off_ids, _off_text, _off_eos) = model
        .generate_with_serving_config(prompt, max_tokens, off_serving_config)
        .expect("OFF decode");
    let off_elapsed_ms = off_started.elapsed().as_secs_f64() * 1000.0;

    // `install()` registers the recorder as the process default but spawns
    // no background drain thread (that is `install_console_recorder_with`'s
    // job, not the plain builder's) -- without `lossless-backpressure`'s
    // managed drainer either, nothing ever moves a record out of its
    // per-core ring into `capture`/the file sink until something calls
    // `Recorder::drain()`. A probe run proved this: an unconditional
    // per-step `debug!` produced zero captured records and a zero-byte
    // file across every step of an OFF+ON pair.
    recorder.drain();

    // the OFF run's decode path never reaches the `speculative_step` branch,
    // so it cannot emit `speculative_verify` -- clearing here means every
    // event `speculative_verify_events` sees afterward came from the ON run.
    capture.clear();

    unsafe {
        env::set_var("PROXIMA_SPECULATIVE_DECODE", "1");
    }
    let on_started = Instant::now();
    let (on_ids, _on_text, _on_eos) = model
        .generate_with_serving_config(prompt, max_tokens, on_serving_config)
        .expect("ON decode");
    let on_elapsed_ms = on_started.elapsed().as_secs_f64() * 1000.0;
    unsafe {
        env::remove_var("PROXIMA_SPECULATIVE_DECODE");
    }
    recorder.drain();

    let verify_events = speculative_verify_events(capture);
    let speculative_verify_steps = verify_events.len();
    let accepted_total: u64 = verify_events.iter().filter_map(accepted_field).sum();

    PairResult {
        off_ids,
        on_ids,
        off_elapsed_ms,
        on_elapsed_ms,
        speculative_verify_steps,
        accepted_total,
    }
}

/// `expect_divergence` flips the pass rule for `--seed-mismatch-control`'s
/// sampled block: that block is a control proving the sampled comparison is
/// rng-sensitive, so passing means `identical = false`, not `true`.
fn report_pair(label: &str, result: &PairResult, expect_divergence: bool) -> bool {
    let first_divergence = result
        .off_ids
        .iter()
        .zip(result.on_ids.iter())
        .position(|(left, right)| left != right);
    let identical = result.off_ids == result.on_ids;

    println!("== {label} ==");
    println!("off_ids = {:?}", result.off_ids);
    println!("on_ids  = {:?}", result.on_ids);
    println!("identical = {identical}");
    println!("first_divergence = {first_divergence:?}");
    println!(
        "off_ms_total = {:.3} on_ms_total = {:.3}",
        result.off_elapsed_ms, result.on_elapsed_ms
    );
    println!(
        "off_ms_per_tok = {:.3} on_ms_per_tok = {:.3}",
        result.off_elapsed_ms / result.off_ids.len().max(1) as f64,
        result.on_elapsed_ms / result.on_ids.len().max(1) as f64
    );
    println!(
        "speculative_verify_steps = {}",
        result.speculative_verify_steps
    );
    println!("accepted_total = {}", result.accepted_total);

    // the non-control pass rule additionally requires speculation to have
    // fired at all (AC1's degenerate-control guard); the mismatch control's
    // pass rule is `identical == false` alone -- a reseeded ON run that hits
    // EOS after its first, already-diverged token is still a valid control.
    if expect_divergence {
        !identical
    } else {
        identical && result.speculative_verify_steps > 0
    }
}

/// AC2's control seed for the sampled block's ON run: distinct from the OFF
/// run's `seed: 7` below. Both seeds feed the same active top-k/top-p/min-p
/// filter chain and repeat penalty at `temperature = 0.8`, so a reseeded ON
/// run samples a different token at the very first step -- verified by the
/// gate run, not assumed (AC2 requires a pair that provably diverges).
const SEED_MISMATCH_CONTROL_ON_SEED: u64 = 4242;

/// A real two-sentence paragraph (Pangrams -- every letter of the alphabet
/// appears in each, real English, not filler), repeated four times.
/// `ngram_simple_draft`'s own `history.len() > size_n + size_m + 1 = 61`
/// gate is necessary but not sufficient: llama.cpp's scan only tries the
/// MOST RECENT earlier occurrence of the trailing `size_n`-gram (this
/// module's own doc on `ngram_simple::ngram_simple_draft`, "the MOST RECENT
/// earlier occurrence... a later match always wins"), and gives up entirely
/// if fewer than `size_n` tokens remain after that occurrence
/// (`copy_max < size_n` in the port) -- it never falls back to an earlier,
/// longer-tailed occurrence. So a draft can only fire when the repeated
/// unit's own token period exceeds `2 * size_n = 24`: a shorter period (a
/// first attempt at this fixture used the ~20-token "quick brown fox"
/// pangram alone and drafted nothing for the entire run, confirmed by a
/// zero-`speculative_verify_steps` gate failure) always finds its nearest
/// recurrence too close to have `size_n` tokens left to copy. This
/// paragraph tokenizes to 37 ids (`tokenize_local` against the real
/// checkpoint), comfortably past the 24-token floor, and four repeats
/// (148 ids) clears the 61-token gate from the very first generated token.
fn default_prompt() -> String {
    const PARAGRAPH: &str = "The quick brown fox jumps over the lazy dog while a curious cat \
         watches quietly from the garden wall. Pack my box with five dozen liquor jugs before \
         the delivery truck arrives at noon. ";
    PARAGRAPH.repeat(4)
}

fn main() {
    let raw_args: Vec<String> = env::args().skip(1).collect();
    let seed_mismatch_control = raw_args
        .iter()
        .any(|arg| arg == "--seed-mismatch-control");
    let mut args = raw_args
        .into_iter()
        .filter(|arg| arg != "--seed-mismatch-control");
    let model_path = args
        .next()
        .unwrap_or_else(|| "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd".to_string());
    let prompt = args.next().unwrap_or_else(default_prompt);
    let max_tokens: usize = args
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(40);

    let log_path: PathBuf = env::var("PROXIMA_TELEMETRY_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("speculative_decode_parity_telemetry.log"));
    let (capture, recorder) = install_telemetry(&log_path);

    let file = File::open(&model_path).expect("open model");
    // SAFETY: `file` remains alive while the read-only mapping is borrowed.
    let bytes: Mmap = unsafe { MmapOptions::new().map(&file) }.expect("map model");
    let parsed = parse_complete(&bytes).expect("parse model");
    let model = LoadedModel::load(&parsed, &bytes).expect("bind model");

    let base_config = ServingConfig {
        gpu_layers: 0,
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        reasoning_budget: 0,
        ..ServingConfig::default()
    };

    // greedy, explicitly: every field a "plain argmax" classification would
    // check is pinned rather than relying on `ServingConfig::default()`
    // staying at these values.
    let greedy_config = ServingConfig {
        temperature: 0.0,
        repeat_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        ..base_config
    };

    // the load-bearing case: a genuinely sampling config (nonzero
    // temperature, an active top-k/top-p/min-p filter chain, and an active
    // repeat penalty) at a fixed seed. Speculative decode's verify branch
    // must select every row through this exact config for the ON run to
    // reproduce the OFF run's own seeded draws in order. `seed: 7` -- of
    // 12 seeds `{1,2,3,5,7,11,13,17,19,23,42,99}` this fixture's own
    // selection swept (scratchpad probe, `speculative-decode-llama-parity`
    // slice 3), most draw this repeated-paragraph prompt's own first token
    // as `<end_of_turn>` (id 107) at `temperature = 0.8`, ending generation
    // after one token, before a decode step ever reaches `cached_len > 0` --
    // structurally unable to exercise the speculative branch at all. Of the
    // seeds that DO run past one token, most sample genuinely novel
    // continuations with no repeated n-gram anywhere in `history` for
    // `ngram_simple_draft` to find (an n-gram drafter is definitionally
    // blind to non-repeating text, `bind.rs`'s own `draft_acceptance`
    // harness measured the same shape on real prose). `seed: 7` is the one
    // swept seed whose sampled draws happen to re-enter this prompt's own
    // repeated paragraph early (its first 15 tokens match the greedy block
    // above exactly), giving the drafter a real recurring pattern to find --
    // the same structural requirement `default_prompt`'s own doc argues for
    // the prompt itself, now also true of what gets SAMPLED from it.
    let sampled_config = ServingConfig {
        temperature: 0.8,
        top_k: 40,
        top_p: 0.9,
        min_p: 0.05,
        repeat_penalty: 1.1,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        seed: 7,
        ..base_config
    };

    let greedy_result = run_pair(
        &model,
        &prompt,
        max_tokens,
        greedy_config,
        greedy_config,
        &capture,
        &recorder,
    );
    let greedy_ok = report_pair("greedy", &greedy_result, false);

    // `--seed-mismatch-control` reseeds only the sampled ON run: greedy's
    // argmax selection ignores the rng entirely, so its OFF/ON pair stays
    // pinned to the same config with or without the flag.
    let sampled_on_config = if seed_mismatch_control {
        ServingConfig {
            seed: SEED_MISMATCH_CONTROL_ON_SEED,
            ..sampled_config
        }
    } else {
        sampled_config
    };
    let sampled_result = run_pair(
        &model,
        &prompt,
        max_tokens,
        sampled_config,
        sampled_on_config,
        &capture,
        &recorder,
    );
    let sampled_ok = report_pair("sampled", &sampled_result, seed_mismatch_control);

    if !greedy_ok || !sampled_ok {
        eprintln!(
            "speculative_decode_parity: greedy_ok={greedy_ok} sampled_ok={sampled_ok} -- either \
             the OFF/ON token streams diverged or the ON run emitted zero speculative_verify \
             events (speculation never fired, so identical=true would be a degenerate control, \
             not evidence the speculative path ran). telemetry log: {}",
            log_path.display()
        );
        std::process::exit(1);
    }
}
