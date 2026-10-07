//! P14 hard parity probe for gemma4-E2B's speculative decode
//! loop (on by default; this probe pins OFF explicitly): for each of TWO `ServingConfig`s -- plain greedy, and a genuinely
//! sampling config (temperature, top-k/top-p/min-p, repeat penalty, a fixed
//! seed) -- runs the SAME prompt with `ServingConfig::speculative` off then
//! on (`SpeculativeType::None` vs `SpeculativeType::NgramSimple`), and
//! reports whether the two token id streams are byte-identical --
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

use core::ops::ControlFlow;
use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use memmap2::{Mmap, MmapOptions};
use proxima_gguf::parse_complete;
use proxima_gguf::types::GgmlType;
use proxima_model_interop::{
    GPU_LAYERS_ALL, LoadedModel, ServingConfig, SpeculativeConfig, SpeculativeDecodeStats,
    SpeculativeType, SpeculativeTypeSet,
};
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
/// `omega=warn` narrows the one target that would otherwise drown the sparse
/// `speculative_verify` event out of the recorder's bounded ring on the metal
/// path: `omega::metal::execute_and_hazards`/`pipeline_buffers_upload` emit a
/// per-op `debug!` for every dispatch/upload (measured: ~85 events per decode
/// step against gemma4-E2B's real graph), so a 146-token prefill plus even a
/// handful of decode steps fills a 65536-capacity ring before the ONE
/// `speculative_verify` event this probe actually needs to count ever gets
/// drained -- a probe run with the bare `"debug"` floor measured
/// `speculative_verify_steps = 0` on `--gpu-layers all` despite the drafter
/// itself firing (confirmed by instrumenting the draft call directly: a
/// real 48-token draft was produced and accepted at that step). The CPU path
/// never hit this ceiling because it has no metal per-op tracing to compete
/// with. `omega`'s own `warn`-and-above events (none emitted in this probe)
/// still reach both sinks; only its `debug`/`trace`/`info` volume is cut.
fn install_telemetry(log_path: &std::path::Path) -> (InMemoryPipe, std::sync::Arc<Recorder>) {
    let filter = env::var("RUST_LOG").unwrap_or_else(|_| "debug,omega=warn".to_string());
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
/// input other than `speculative.speculative_types` this probe itself sets.
struct PairResult {
    off_ids: Vec<u32>,
    on_ids: Vec<u32>,
    off_elapsed_ms: f64,
    on_elapsed_ms: f64,
    speculative_verify_steps: usize,
    accepted_total: u64,
    on_stats: SpeculativeDecodeStats,
}

/// `off_serving_config` and `on_serving_config` differ only in
/// `speculative.speculative_types` (`None` vs `NgramSimple`) for every caller
/// except `--seed-mismatch-control`'s sampled block, where the ON run is
/// ALSO deliberately reseeded -- the point of that control is that the two
/// runs must NOT reproduce each other.
fn run_pair(
    model: &LoadedModel,
    prompt: &str,
    max_tokens: usize,
    off_serving_config: ServingConfig,
    on_serving_config: ServingConfig,
    capture: &InMemoryPipe,
    recorder: &Recorder,
) -> PairResult {
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

    // `generate_streaming_with_speculative_stats` instead of the OFF run's
    // plain `generate_with_serving_config`: its own `on_stats` out parameter
    // (`SpeculativeDecodeStats`'s own doc) reads verify/drafted/accepted
    // counts, including the PER-TYPE breakdown, directly off the decode
    // loop's own bookkeeping -- the multi-type demonstration run needs to
    // attribute drafted/accepted tokens to whichever enabled type actually
    // won each step, which the telemetry ring below cannot do (it only
    // records the pooled `accepted` field, not `drafting_type`).
    let mut on_stats = SpeculativeDecodeStats::default();
    let mut on_token = |_event: proxima_model_interop::TokenEvent<'_>| ControlFlow::Continue(());
    let on_started = Instant::now();
    let (on_ids, _on_text, _on_eos) = model
        .generate_streaming_with_speculative_stats(
            prompt,
            max_tokens,
            on_serving_config,
            &mut on_token,
            &mut on_stats,
            None,
        )
        .expect("ON decode");
    let on_elapsed_ms = on_started.elapsed().as_secs_f64() * 1000.0;
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
        on_stats,
    }
}

/// Prints one line per enabled drafter type -- `SpeculativeType::llama_name`
/// keeps this in the same vocabulary `--drafter` itself accepts. Skipped
/// entirely for the common single-type case (nothing to attribute), matching
/// this file's own `--drafter` doc: attribution only matters once more than
/// one type can win a given verify step.
fn print_per_type_stats(drafter_types: SpeculativeTypeSet, stats: &SpeculativeDecodeStats) {
    if drafter_types.iter_priority_order().count() <= 1 {
        return;
    }
    for type_id in drafter_types.iter_priority_order() {
        let per_type = stats.per_type_stats(type_id);
        println!(
            "per_type {} drafted={} accepted={}",
            type_id.llama_name(),
            per_type.drafted,
            per_type.accepted
        );
    }
}

/// `expect_divergence` flips the pass rule for `--seed-mismatch-control`'s
/// sampled block: that block is a control proving the sampled comparison is
/// rng-sensitive, so passing means `identical = false`, not `true`.
fn report_pair(
    label: &str,
    result: &PairResult,
    drafter_types: SpeculativeTypeSet,
    expect_divergence: bool,
) -> bool {
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
    print_per_type_stats(drafter_types, &result.on_stats);

    // the non-control pass rule additionally requires speculation to have
    // fired at all (the degenerate-control guard); the mismatch control's
    // pass rule is `identical == false` alone -- a reseeded ON run that hits
    // EOS after its first, already-diverged token is still a valid control.
    if expect_divergence {
        !identical
    } else {
        identical && result.speculative_verify_steps > 0
    }
}

/// The control seed for the sampled block's ON run: distinct from the OFF
/// run's `seed: 7` below. Both seeds feed the same active top-k/top-p/min-p
/// filter chain and repeat penalty at `temperature = 0.8`, so a reseeded ON
/// run samples a different token at the very first step -- verified by the
/// gate run, not assumed (the gate requires a pair that provably diverges).
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

/// `ngram-mod`'s own default prompt (the spec's own): [`default_prompt`]'s
/// four full repeats make greedy (`temperature = 0.0`) draft well (a real
/// run measured `speculative_verify_steps = 2 accepted_total = 44`), but the
/// sampled block (`temperature = 0.8`, `seed: 7`) samples ~15 tokens that
/// match greedy, diverges, then reaches `<end_of_turn><eos>` at ~22 tokens --
/// before `ngram_mod_draft`'s own `n_min = 48`-token chain-keep floor is
/// ever reachable (`ngram_mod.rs`'s own doc on that gate), so speculation
/// never fires there (real runs measured `speculative_verify_steps = 0`).
/// The fix is NOT more repeats: five or six full repeats (real runs on this
/// checkpoint) push BOTH temperatures off the literal-repeat completion
/// entirely -- greedy either samples `<eos>` on the very first token or
/// paraphrases the paragraph into a `summary` instead of repeating it
/// (ngram-mod's hash table trained on the literal text then finds nothing to
/// match), which is worse, not better. What works is ending the prompt
/// MID-SENTENCE, one word short of the fourth repeat's final clause ("...the
/// delivery truck arrives at", no trailing "noon."): both temperatures are
/// then forced to complete the interrupted clause before doing anything
/// else, which is grammatically almost the only continuation an LM will
/// assign real probability mass to, so greedy and sampled land on the exact
/// same completion at that step regardless of temperature. A real run on
/// this checkpoint (`--gpu-layers all`) measured `speculative_verify_steps =
/// 1 accepted_total = 1` for BOTH blocks, byte-identical OFF/ON, at
/// `max_tokens = 64` (`ngram_mod_default_max_tokens`) -- the small accepted
/// count is expected: this prompt's job is to prove speculation FIRES on
/// this type, not to maximise its acceptance rate (that is AC17/R14's own
/// corpus-level measurement).
fn ngram_mod_default_prompt() -> String {
    const PARAGRAPH: &str = "The quick brown fox jumps over the lazy dog while a curious cat \
         watches quietly from the garden wall. Pack my box with five dozen liquor jugs before \
         the delivery truck arrives at noon. ";
    const PARTIAL: &str = "The quick brown fox jumps over the lazy dog while a curious cat \
         watches quietly from the garden wall. Pack my box with five dozen liquor jugs before \
         the delivery truck arrives at";
    format!("{}{PARTIAL}", PARAGRAPH.repeat(3))
}

/// Picks [`default_prompt`] for every drafter set except the single-member
/// `{ngram-mod}` set, where [`ngram_mod_default_prompt`]'s own doc explains
/// why a different prompt (and a larger `max_tokens`, to give the
/// mid-sentence completion room to run) is needed for speculation to fire
/// in both the greedy and sampled blocks. A caller-supplied positional
/// `prompt`/`max_tokens` argument always overrides this choice (`main`'s own
/// `args.next()` calls happen before this is consulted).
fn default_prompt_and_max_tokens(drafter_types: SpeculativeTypeSet) -> (String, usize) {
    if drafter_types == SpeculativeTypeSet::single(SpeculativeType::NgramMod) {
        (ngram_mod_default_prompt(), 64)
    } else {
        (default_prompt(), 40)
    }
}

/// `--gpu-layers <n|all>` -- llama's own `-ngl` sentinel convention
/// ([`GPU_LAYERS_ALL`]'s own doc): `all` selects [`GPU_LAYERS_ALL`],
/// anything else parses as a literal layer count. `apply_serving_config`
/// (`serving.rs`) is the sole validator: only `0` (CPU) and
/// [`GPU_LAYERS_ALL`] (whole-model metal offload, requires the `metal`
/// feature) are supported today, so an out-of-range value surfaces as that
/// function's own typed error rather than a second check here.
fn parse_gpu_layers(value: &str) -> i32 {
    if value.eq_ignore_ascii_case("all") {
        GPU_LAYERS_ALL
    } else {
        value
            .parse()
            .unwrap_or_else(|err| panic!("--gpu-layers {value}: not `all` or an integer: {err}"))
    }
}

/// `--drafter <type>[,<type>...]` -- a comma list of llama's own `--spec-type`
/// names ([`SpeculativeType::from_llama_name`]), folded into one
/// [`SpeculativeTypeSet`] the same way llama's own `--spec-type` (repeatable)
/// folds into `common_params_speculative::types`
/// ([`SpeculativeTypeSet`]'s own doc). Panics on an unrecognised name --
/// this CLI has no typed-error surface of its own, matching every other
/// `parse_*` helper in this file.
fn parse_drafter_flag(value: &str) -> SpeculativeTypeSet {
    value
        .split(',')
        .map(str::trim)
        .fold(SpeculativeTypeSet::empty(), |set, name| {
            let type_id = SpeculativeType::from_llama_name(name)
                .unwrap_or_else(|| panic!("--drafter {value}: unknown speculation type {name:?}"));
            set.insert(type_id)
        })
}

/// The compiled-in cargo features that move timing numbers for this
/// binary (`proxima-model-interop/Cargo.toml` lines 128-227), rendered
/// `key=true/false` via `cfg!` -- the SAME shape `speculative_bench.rs`
/// prints, so a parity divergence and a bench timing line are always
/// attributable to the same binary configuration.
fn compiled_perf_features_summary() -> String {
    format!(
        "metal_feature={} metal_fuse_attn_decode_feature={} identity_copy_alias_feature={} metal_tiled_gemm_feature={}",
        cfg!(feature = "metal"),
        cfg!(feature = "metal-fuse-attn-decode"),
        cfg!(feature = "identity-copy-alias"),
        cfg!(feature = "metal-tiled-gemm"),
    )
}

/// The short commit this binary was built from, read at startup rather
/// than baked in by a build script (no build.rs exists in this crate) --
/// `unknown` when `git` is unavailable or this tree is not a git checkout.
fn git_commit_at_startup() -> String {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map_or_else(|| "unknown".to_string(), |commit| commit.trim().to_string())
}

/// Prints the same `lever_config` shape `speculative_bench.rs` prints, so a
/// parity result is equally attributable to compiled features and commit --
/// this probe carries no `PROXIMA_*` runtime levers of its own to report.
fn print_lever_config() {
    println!(
        "lever_config {} git_commit={}",
        compiled_perf_features_summary(),
        git_commit_at_startup(),
    );
}

fn main() {
    print_lever_config();
    let raw_args: Vec<String> = env::args().skip(1).collect();
    let seed_mismatch_control = raw_args
        .iter()
        .any(|arg| arg == "--seed-mismatch-control");
    let gpu_layers_flag_index = raw_args.iter().position(|arg| arg == "--gpu-layers");
    let gpu_layers = gpu_layers_flag_index
        .and_then(|flag_index| raw_args.get(flag_index + 1))
        .map_or(0, |value| parse_gpu_layers(value));
    let gpu_layers_value_index = gpu_layers_flag_index.map(|flag_index| flag_index + 1);
    let drafter_flag_index = raw_args.iter().position(|arg| arg == "--drafter");
    let drafter_types = drafter_flag_index
        .and_then(|flag_index| raw_args.get(flag_index + 1))
        .map_or_else(
            || SpeculativeTypeSet::single(SpeculativeType::NgramSimple),
            |value| parse_drafter_flag(value),
        );
    let drafter_value_index = drafter_flag_index.map(|flag_index| flag_index + 1);
    let telemetry_file_flag_index = raw_args.iter().position(|arg| arg == "--telemetry-file");
    let telemetry_file_flag = telemetry_file_flag_index
        .and_then(|flag_index| raw_args.get(flag_index + 1))
        .map(PathBuf::from);
    let telemetry_file_value_index = telemetry_file_flag_index.map(|flag_index| flag_index + 1);
    let mut args = raw_args
        .into_iter()
        .enumerate()
        .filter(move |(index, arg)| {
            arg != "--seed-mismatch-control"
                && Some(*index) != gpu_layers_flag_index
                && Some(*index) != gpu_layers_value_index
                && Some(*index) != drafter_flag_index
                && Some(*index) != drafter_value_index
                && Some(*index) != telemetry_file_flag_index
                && Some(*index) != telemetry_file_value_index
        })
        .map(|(_, arg)| arg);
    let model_path = args
        .next()
        .unwrap_or_else(|| "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd".to_string());
    let explicit_prompt = args.next();
    let explicit_max_tokens: Option<usize> = args.next().and_then(|value| value.parse().ok());
    let (chosen_prompt, chosen_max_tokens) = default_prompt_and_max_tokens(drafter_types);
    let prompt = explicit_prompt.unwrap_or(chosen_prompt);
    let max_tokens = explicit_max_tokens.unwrap_or(chosen_max_tokens);
    println!(
        "speculative_decode_parity: drafter={:?} max_tokens={max_tokens} prompt={:?}",
        drafter_types.iter_priority_order().collect::<Vec<_>>(),
        &prompt[..prompt.len().min(80)]
    );

    let log_path: PathBuf = telemetry_file_flag
        .or_else(|| env::var("PROXIMA_TELEMETRY_FILE").ok().map(PathBuf::from))
        .unwrap_or_else(|| std::env::temp_dir().join("speculative_decode_parity_telemetry.log"));
    let (capture, recorder) = install_telemetry(&log_path);

    let file = File::open(&model_path).expect("open model");
    // SAFETY: `file` remains alive while the read-only mapping is borrowed.
    let bytes: Mmap = unsafe { MmapOptions::new().map(&file) }.expect("map model");
    let parsed = parse_complete(&bytes).expect("parse model");
    let model = LoadedModel::load(&parsed, &bytes).expect("bind model");

    let base_config = ServingConfig {
        gpu_layers,
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        reasoning_budget: 0,
        speculative: SpeculativeConfig::none(),
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
    // selection swept (scratchpad probe, `speculative-decode-llama-parity`),
    // most draw this repeated-paragraph prompt's own first token
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

    let speculative_on = SpeculativeConfig {
        speculative_types: drafter_types,
        ..SpeculativeConfig::none()
    };

    let greedy_result = run_pair(
        &model,
        &prompt,
        max_tokens,
        greedy_config,
        greedy_config.with_speculative(speculative_on),
        &capture,
        &recorder,
    );
    let greedy_ok = report_pair("greedy", &greedy_result, drafter_types, false);

    // `--seed-mismatch-control` reseeds only the sampled ON run: greedy's
    // argmax selection ignores the rng entirely, so its OFF/ON pair stays
    // pinned to the same config (other than speculation itself) with or
    // without the flag.
    let sampled_on_config = if seed_mismatch_control {
        ServingConfig {
            seed: SEED_MISMATCH_CONTROL_ON_SEED,
            ..sampled_config
        }
    } else {
        sampled_config
    }
    .with_speculative(speculative_on);
    let sampled_result = run_pair(
        &model,
        &prompt,
        max_tokens,
        sampled_config,
        sampled_on_config,
        &capture,
        &recorder,
    );
    let sampled_ok = report_pair(
        "sampled",
        &sampled_result,
        drafter_types,
        seed_mismatch_control,
    );

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

#[cfg(test)]
mod tests {
    use super::{compiled_perf_features_summary, git_commit_at_startup};

    #[test]
    fn compiled_perf_features_summary_names_every_performance_relevant_feature() {
        let summary = compiled_perf_features_summary();

        for key in [
            "metal_feature=",
            "metal_fuse_attn_decode_feature=",
            "identity_copy_alias_feature=",
            "metal_tiled_gemm_feature=",
        ] {
            assert!(
                summary.contains(key),
                "lever_config must attribute {key} on the same line, got: {summary}"
            );
        }
        assert!(
            summary.split(' ').all(|field| field.ends_with("=true") || field.ends_with("=false")),
            "every feature field must render a bool, got: {summary}"
        );
    }

    #[test]
    fn git_commit_at_startup_never_panics_and_is_never_empty() {
        let commit = git_commit_at_startup();

        assert!(
            !commit.is_empty(),
            "git_commit_at_startup must fall back to \"unknown\", never an empty string"
        );
        assert!(
            !commit.contains(char::is_whitespace),
            "git_commit_at_startup must trim to a bare token, got: {commit:?}"
        );
    }
}
