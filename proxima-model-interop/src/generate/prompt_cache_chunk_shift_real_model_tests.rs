#![allow(clippy::expect_used)]

use core::ops::ControlFlow;
use std::sync::PoisonError;

use super::chunk_shift::{ChunkRun, extract_run};
use super::prompt_cache_real_model_tests::{
    TurnOutcome, cached_config, cached_tokens_after, encode_continuation, encode_opening,
    long_document, uncached_config, with_model,
};
use super::ring_checkpoint::LayerRows;
use super::*;
use crate::generate::CachePath;
use crate::serving::{PromptCacheConfig, ServingConfig, SpeculativeConfig};

const GENERATED_TOKENS: usize = 16;
const REUSE_MIN: u32 = 64;

fn shifting_config(cache_reuse_min: u32) -> ServingConfig<'static> {
    let base = cached_config(SpeculativeConfig::none());
    ServingConfig {
        prompt_cache: PromptCacheConfig {
            cache_reuse_min,
            ..base.prompt_cache
        },
        ..base
    }
}

fn generate(
    model: &LoadedModel<'_>,
    config: &ServingConfig<'_>,
    ids: &[u32],
    turn_ends: &[usize],
) -> TurnOutcome {
    let (generated, _text, stopped_by_eos) = model
        .generate_from_ids_with_turn_ends(ids, turn_ends, GENERATED_TOKENS, config, &mut |_event| {
            ControlFlow::Continue(())
        })
        .expect("generate from the prompt ids");
    let report = model
        .last_prompt_cache_report()
        .expect("a cached request records its report");
    TurnOutcome {
        generated,
        stopped_by_eos,
        report,
    }
}

fn prefill_state(model: &LoadedModel<'_>, ids: &[u32]) -> PrefixState {
    let config = uncached_config(SpeculativeConfig::none());
    let effective = model
        .effective_serving_config(&config)
        .expect("resolve the serving config");
    let mut runtime = BackendRuntime::new(&effective);
    let (.., state) = model
        .run_decode_loop_from_ids(
            ids.to_vec(),
            1,
            &effective,
            &mut runtime,
            None,
            &mut LogitsSink::Discard,
            &mut NodeValuesSink::Discard,
            &mut |_event| ControlFlow::Continue(()),
            None,
            true,
            None,
            None,
        )
        .expect("prefill the ids");
    state
}

/// Per row of a moved chunk, how far the moved rows are from the rows a fresh
/// prefill stores at the same positions.
struct LayerDiffs {
    layer: usize,
    ring: bool,
    first_position: usize,
    key_diff: Vec<f32>,
    key_scale: Vec<f32>,
    value_diff: Vec<f32>,
}

fn row_maxima(moved: &[f32], fresh: &[f32], width: usize) -> (Vec<f32>, Vec<f32>) {
    let diff = moved
        .chunks_exact(width)
        .zip(fresh.chunks_exact(width))
        .map(|(moved_row, fresh_row)| {
            moved_row
                .iter()
                .zip(fresh_row)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f32, f32::max)
        })
        .collect();
    let scale = fresh
        .chunks_exact(width)
        .map(|row| row.iter().map(|value| value.abs()).fold(0.0_f32, f32::max))
        .collect();
    (diff, scale)
}

/// Moves `run` out of a fresh prefill of `old_prefix` (which ends where the
/// run ends) with the model's own rotation for the position delta, and
/// compares every moved row to the same position in a fresh prefill of
/// `new_prefix` (which ends where the run ends in the new prompt).
fn measure_shift(
    model: &LoadedModel<'_>,
    old_prefix: &[u32],
    new_prefix: &[u32],
    run: ChunkRun,
) -> Vec<LayerDiffs> {
    assert_eq!(old_prefix.len(), run.old_end());
    assert_eq!(new_prefix.len(), run.new_end());
    assert_eq!(
        old_prefix[run.old_start..],
        new_prefix[run.new_start..],
        "the chunk must be the same ids in both prompts"
    );
    let (_, widths) = model
        .declared_layer_cache_names_and_widths()
        .expect("declared layer widths");
    let config = uncached_config(SpeculativeConfig::none());
    let old_state = prefill_state(model, old_prefix);
    let new_state = prefill_state(model, new_prefix);
    let delta = isize::try_from(run.new_start).expect("fits")
        - isize::try_from(run.old_start).expect("fits");
    let rotations = model.delta_rotations(&config, delta);
    let moved = extract_run(&old_state, run, &widths, &rotations)
        .expect("every attention layer has rows and a rotation");
    let in_place = ChunkRun {
        old_start: run.new_start,
        new_start: run.new_start,
        len: run.len,
    };
    moved
        .layers
        .iter()
        .map(|moved_rows| {
            let LayerPadRowWidths::Attention {
                even_odd_row,
                v_row,
            } = widths[moved_rows.layer]
            else {
                panic!("layer {} owns rows", moved_rows.layer);
            };
            let LayerCacheState::Attention(cache) = &new_state.layer_caches[moved_rows.layer]
            else {
                panic!("layer {} owns a cache", moved_rows.layer);
            };
            let fresh = LayerRows::of_run(moved_rows.layer, cache, &in_place, even_odd_row, v_row)
                .expect("the fresh state holds the rows");
            let (even_diff, even_scale) =
                row_maxima(&moved_rows.k_even, &fresh.k_even, even_odd_row);
            let (odd_diff, odd_scale) = row_maxima(&moved_rows.k_odd, &fresh.k_odd, even_odd_row);
            let (value_diff, _) = row_maxima(&moved_rows.v, &fresh.v, v_row);
            let rows = even_diff.len();
            LayerDiffs {
                layer: moved_rows.layer,
                ring: cache.ring_geometry().is_some(),
                first_position: run.new_end() - rows,
                key_diff: even_diff
                    .iter()
                    .zip(&odd_diff)
                    .map(|(a, b)| a.max(*b))
                    .collect(),
                key_scale: even_scale
                    .iter()
                    .zip(&odd_scale)
                    .map(|(a, b)| a.max(*b))
                    .collect(),
                value_diff,
            }
        })
        .collect()
}

fn max_of(values: &[f32]) -> f32 {
    values.iter().copied().fold(0.0_f32, f32::max)
}

fn report_diffs(label: &str, run: ChunkRun, diffs: &[LayerDiffs]) {
    for layer in diffs {
        let depth = |row: usize| layer.first_position + row - run.new_start;
        let from_depth = |floor: usize| -> Vec<usize> {
            (0..layer.key_diff.len())
                .filter(|row| depth(*row) >= floor)
                .collect()
        };
        let worst_over = |rows: &[usize]| {
            rows.iter()
                .map(|row| layer.key_diff[*row])
                .fold(0.0_f32, f32::max)
        };
        eprintln!(
            "SHIFT_ROPE {label} layer={} ring={} rows={} first_position={} \
             max_key_diff={:.6e} max_key_abs={:.4} max_value_diff={:.6e} \
             max_key_diff_depth_lt_256={:.6e} max_key_diff_depth_ge_2200={:.6e}",
            layer.layer,
            layer.ring,
            layer.key_diff.len(),
            layer.first_position,
            max_of(&layer.key_diff),
            max_of(&layer.key_scale),
            max_of(&layer.value_diff),
            worst_over(
                &(0..layer.key_diff.len())
                    .filter(|row| depth(*row) < 256)
                    .collect::<Vec<_>>()
            ),
            worst_over(&from_depth(2200)),
        );
    }
}

fn pool() -> String {
    long_document(38_000)
}

fn slice(pool: &str, from: usize, chars: usize) -> String {
    pool.chars().skip(from).take(chars).collect()
}

/// A chunk of text 2,400 tokens deep or more, preceded by two different
/// openings of different lengths, so the same ids sit at two positions.
/// Every sliding layer before the first full-attention layer (0 to 3) sees
/// only the last 511 positions through each layer, so a key those layers
/// store 2,200 or more tokens into the chunk, and the first full layer's
/// (layer 4) key, depend on the chunk alone: rotating the first prefill's rows
/// by the position delta must reproduce the second prefill's rows. Layers
/// 5 and later read the earlier text through full-attention layer 4, so their
/// rows differ by what the two openings contribute; the line for each is
/// printed, not asserted.
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn a_chunk_moved_by_the_models_own_rotation_matches_the_rows_a_fresh_prefill_stores() {
    with_model(|model| {
        let pool = pool();
        let long_opening = encode_opening(
            model,
            &format!(
                "<|turn>user\n{}<turn|>\n<|turn>model\n",
                slice(&pool, 0, 1500)
            ),
        );
        let short_opening = encode_opening(
            model,
            &format!(
                "<|turn>user\n{}<turn|>\n<|turn>model\n",
                slice(&pool, 0, 300)
            ),
        );
        let chunk = encode_continuation(model, &slice(&pool, 1500, 15_000));
        assert!(
            chunk.len() > 2600,
            "the chunk must reach 2,200 tokens deep, got {}",
            chunk.len()
        );
        let old_prefix: Vec<u32> = long_opening.iter().chain(&chunk).copied().collect();
        let new_prefix: Vec<u32> = short_opening.iter().chain(&chunk).copied().collect();
        let run = ChunkRun {
            old_start: long_opening.len(),
            new_start: short_opening.len(),
            len: chunk.len(),
        };
        assert_ne!(run.old_start, run.new_start);

        let diffs = measure_shift(model, &old_prefix, &new_prefix, run);

        report_diffs("exact_context", run, &diffs);
        assert_eq!(diffs.len(), 15, "gemma4-E2B owns 15 KV layers");
        for layer in diffs.iter().filter(|layer| layer.layer <= 4) {
            let scale = max_of(&layer.key_scale);
            let deep: Vec<usize> = (0..layer.key_diff.len())
                .filter(|row| layer.first_position + row - run.new_start >= 2200)
                .collect();
            assert!(
                !deep.is_empty(),
                "layer {} has no row 2,200 deep",
                layer.layer
            );
            let worst = deep
                .iter()
                .map(|row| layer.key_diff[*row])
                .fold(0.0_f32, f32::max);
            assert!(
                worst <= 2e-3 * scale,
                "layer {} key diff {worst:e} against scale {scale}",
                layer.layer
            );
        }
    });
}

struct Squash {
    system_end: usize,
    kept_start: usize,
    old_prompt: Vec<u32>,
    turn_ends: Vec<usize>,
    summary: Vec<u32>,
    kept_end: usize,
    closing: Vec<u32>,
}

fn turn(user: &str, answer: &str) -> String {
    format!("<|turn>user\n{user}<turn|>\n<|turn>model\n{answer}<turn|>\n")
}

fn squash_case(model: &LoadedModel<'_>) -> Squash {
    let pool = pool();
    let system = encode_opening(
        model,
        &format!("<|turn>system\n{}<turn|>\n", slice(&pool, 0, 2400)),
    );
    let mut old_prompt = system.clone();
    let mut turn_ends = vec![system.len()];
    let mut kept_start = 0;
    let mut kept_end = 0;
    for index in 0..6 {
        let from = 2400 + index * 2000;
        if index == 4 {
            kept_start = old_prompt.len();
        }
        old_prompt.extend(encode_continuation(
            model,
            &turn(&slice(&pool, from, 1200), &slice(&pool, from + 1200, 800)),
        ));
        turn_ends.push(old_prompt.len());
        kept_end = old_prompt.len();
    }
    old_prompt.extend(encode_continuation(
        model,
        &format!(
            "<|turn>user\n{}<turn|>\n<|turn>model\n",
            slice(&pool, 14_400, 1400)
        ),
    ));
    let summary = encode_continuation(
        model,
        &turn(
            "Summary of the conversation so far: the user shared several documents and asked about each.",
            "Understood.",
        ),
    );
    let closing = encode_continuation(
        model,
        &format!(
            "<turn|>\n<|turn>user\n{}<turn|>\n<|turn>model\n",
            slice(&pool, 15_800, 400)
        ),
    );
    Squash {
        system_end: system.len(),
        kept_start,
        old_prompt,
        turn_ends,
        summary,
        kept_end,
        closing,
    }
}

struct SquashRun {
    shifted: TurnOutcome,
    fresh: Vec<u32>,
    stored_after_first: usize,
    case: Squash,
    new_prompt: Vec<u32>,
}

/// AC5: a transcript whose middle turns were replaced by a short summary,
/// the system prompt in front and the last turns (and the answer the cache
/// holds) byte-identical. `cache_reuse_min` is the config under test.
fn squashed_history(model: &LoadedModel<'_>, cache_reuse_min: u32) -> SquashRun {
    let case = squash_case(model);
    let config = shifting_config(cache_reuse_min);
    let first = generate(model, &config, &case.old_prompt, &case.turn_ends);
    model
        .run_pending_prewarm(&config)
        .expect("run the queued end-of-answer prewarm");
    let stored = stored_lengths(model);
    assert_eq!(stored.len(), 1, "the first request stores one entry");
    let mut new_prompt = case.old_prompt[..case.system_end].to_vec();
    new_prompt.extend_from_slice(&case.summary);
    new_prompt.extend_from_slice(&case.old_prompt[case.kept_start..]);
    new_prompt.extend_from_slice(&first.generated);
    new_prompt.extend_from_slice(&case.closing);
    let shifted = generate(model, &config, &new_prompt, &[]);
    let fresh = generate(
        model,
        &uncached_config(SpeculativeConfig::none()),
        &new_prompt,
        &[],
    )
    .generated;
    SquashRun {
        shifted,
        fresh,
        stored_after_first: stored[0],
        case,
        new_prompt,
    }
}

fn describe(label: &str, run: &SquashRun) {
    let report = &run.shifted.report;
    eprintln!(
        "SQUASH {label} old_prompt_tokens={} new_prompt_tokens={} system_end={} kept_start={} \
         stored_after_first={} path={} lcp={} reused={} shifted={} prefilled={} ids_equal_fresh={} \
         shifted_ids={:?} fresh_ids={:?}",
        run.case.old_prompt.len(),
        run.new_prompt.len(),
        run.case.system_end,
        run.case.kept_start,
        run.stored_after_first,
        report.path.as_str(),
        report.lcp,
        report.reused_tokens,
        report.shifted_tokens,
        report.prefilled_tokens,
        run.shifted.generated == run.fresh,
        run.shifted.generated,
        run.fresh,
    );
}

fn stored_lengths(model: &LoadedModel<'_>) -> Vec<usize> {
    model
        .prompt_cache
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .entry_states()
        .map(|state| state.cached_len)
        .collect()
}

/// AC5's counts: the system prompt is restored from its checkpoint, the kept
/// turns, the last user turn and the answer the entry holds are moved (with
/// the end-of-turn tokens that close both the summary and the turn before the
/// kept ones, which match), and only the rest of the summary, the unforwarded
/// tail and the new user turn are prefilled. The ids are printed against a full prefill's, not asserted
/// equal: the moved rows were computed under the middle turns the summary
/// replaced, so the layers past the first full-attention layer hold
/// different rows than a fresh prefill of the squashed prompt, and the ids
/// generated from them can differ
/// ([`the_squash_stores_the_lifted_rows_in_every_layer_not_the_fresh_ones`]
/// measures how far).
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn a_squashed_history_shifts_the_kept_turns_and_the_system_prompt_is_restored() {
    with_model(|model| {
        let run = squashed_history(model, REUSE_MIN);
        describe("shift", &run);
        let report = run.shifted.report;
        let closing_shared_with_the_summary = run
            .case
            .summary
            .iter()
            .rev()
            .zip(run.case.old_prompt[..run.case.kept_start].iter().rev())
            .take_while(|(summary, old)| summary == old)
            .count();
        let shifted =
            run.stored_after_first - run.case.kept_start + closing_shared_with_the_summary;

        assert!(
            run.case.old_prompt.len() > 3000,
            "the transcript must have wrapped the sliding rings"
        );
        assert_eq!(report.path, CachePath::Shift, "{report:?}");
        assert_eq!(report.shifted_tokens, shifted, "{report:?}");
        assert_eq!(
            report.reused_tokens,
            run.case.system_end + shifted,
            "{report:?}"
        );
        assert_eq!(
            report.prefilled_tokens,
            run.new_prompt.len() - report.reused_tokens
        );
    });
}

#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn the_same_squashed_history_with_reuse_off_keeps_only_the_system_prompt() {
    with_model(|model| {
        let run = squashed_history(model, 0);
        describe("no_shift", &run);
        let report = run.shifted.report;

        assert_eq!(report.path, CachePath::Checkpoint, "{report:?}");
        assert_eq!(report.shifted_tokens, 0);
        assert_eq!(report.reused_tokens, run.case.system_end, "{report:?}");
        assert_eq!(run.shifted.generated, run.fresh);
    });
}

/// The kept turns end 370 tokens before the end of the cached sequence, past
/// the ring's 256 rows of slack, so the rows the window after them reads are
/// overwritten: the chunk is prefilled like any other gap, never moved from
/// rows that are gone.
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn a_kept_turn_whose_ring_rows_are_gone_is_prefilled_and_still_matches_a_full_prefill() {
    with_model(|model| {
        let case = squash_case(model);
        let config = shifting_config(REUSE_MIN);
        let first = generate(model, &config, &case.old_prompt, &case.turn_ends);
        let held = cached_tokens_after(case.old_prompt.len(), &first);
        let mut new_prompt = case.old_prompt[..case.system_end].to_vec();
        new_prompt.extend_from_slice(&case.summary);
        new_prompt.extend_from_slice(&case.old_prompt[case.kept_start..case.kept_end]);
        new_prompt.extend_from_slice(&case.closing);
        assert!(
            held - case.kept_end > 256,
            "the kept turns must end past the slack"
        );

        let shifted = generate(model, &config, &new_prompt, &[]);
        let fresh = generate(
            model,
            &uncached_config(SpeculativeConfig::none()),
            &new_prompt,
            &[],
        )
        .generated;

        eprintln!("SQUASH ring_gone {:?}", shifted.report);
        assert_ne!(
            shifted.report.path,
            CachePath::Shift,
            "{:?}",
            shifted.report
        );
        assert_eq!(shifted.report.shifted_tokens, 0);
        assert_eq!(shifted.generated, fresh);
    });
}

/// How far the rows the squash moves are from the rows a fresh prefill of the
/// squashed prompt stores, layer by layer: the cost of reusing rows computed
/// under the old context.
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn the_squash_reports_how_far_the_moved_rows_are_from_a_fresh_prefill() {
    with_model(|model| {
        let case = squash_case(model);
        let new_start = case.system_end + case.summary.len();
        let mut new_prefix = case.old_prompt[..case.system_end].to_vec();
        new_prefix.extend_from_slice(&case.summary);
        new_prefix.extend_from_slice(&case.old_prompt[case.kept_start..]);
        let run = ChunkRun {
            old_start: case.kept_start,
            new_start,
            len: case.old_prompt.len() - case.kept_start,
        };

        let diffs = measure_shift(model, &case.old_prompt, &new_prefix, run);

        report_diffs("squash", run, &diffs);
        assert_eq!(diffs.len(), 15);
    });
}

fn rows_of(
    state: &PrefixState,
    widths: &[LayerPadRowWidths],
    layer: usize,
    run: &ChunkRun,
) -> LayerRows {
    let LayerPadRowWidths::Attention {
        even_odd_row,
        v_row,
    } = widths[layer]
    else {
        panic!("layer {layer} owns rows");
    };
    let LayerCacheState::Attention(cache) = &state.layer_caches[layer] else {
        panic!("layer {layer} owns a cache");
    };
    LayerRows::of_run(layer, cache, run, even_odd_row, v_row).expect("the state holds the rows")
}

/// Largest key difference between two sets of rows and the largest key
/// magnitude among the second.
fn key_difference(left: &LayerRows, right: &LayerRows) -> (f32, f32) {
    let worst = |moved: &[f32], fresh: &[f32]| {
        moved
            .iter()
            .zip(fresh)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max)
    };
    let scale = right
        .k_even
        .iter()
        .chain(&right.k_odd)
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    (
        worst(&left.k_even, &right.k_even).max(worst(&left.k_odd, &right.k_odd)),
        scale,
    )
}

/// The pipeline end to end, in the layers where the moved rows are exact. A
/// request whose prompt is the cached one with a shorter opening moves the
/// whole 3,600-token chunk; layers 0 to 3 (the last window) and layer 4 (rows
/// 2,200 and more into the chunk) depend on the chunk alone, so the entry the
/// request leaves behind must hold the rows a fresh prefill of the same
/// prompt stores there.
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn the_rows_a_shifted_request_stores_match_a_fresh_prefill_where_the_context_is_the_same() {
    with_model(|model| {
        let pool = pool();
        let opening = |chars: usize| {
            encode_opening(
                model,
                &format!(
                    "<|turn>user\n{}<turn|>\n<|turn>model\n",
                    slice(&pool, 0, chars)
                ),
            )
        };
        let (long_opening, short_opening) = (opening(1500), opening(300));
        let chunk = encode_continuation(model, &slice(&pool, 1500, 15_000));
        let old_prompt: Vec<u32> = long_opening.iter().chain(&chunk).copied().collect();
        let new_prompt: Vec<u32> = short_opening.iter().chain(&chunk).copied().collect();
        let base = shifting_config(REUSE_MIN);
        let config = ServingConfig {
            prompt_cache: PromptCacheConfig {
                min_similarity_milli: 0,
                ..base.prompt_cache
            },
            ..base
        };

        generate(model, &config, &old_prompt, &[64]);
        let shifted = generate(model, &config, &new_prompt, &[]);
        let fresh = generate(
            model,
            &uncached_config(SpeculativeConfig::none()),
            &new_prompt,
            &[],
        )
        .generated;

        let report = shifted.report;
        eprintln!(
            "SHIFT_PIPELINE {report:?} ids_equal_fresh={} shifted_ids={:?} fresh_ids={fresh:?}",
            shifted.generated == fresh,
            shifted.generated
        );
        assert_eq!(report.path, CachePath::Shift, "{report:?}");
        let shared_closing = long_opening
            .iter()
            .rev()
            .zip(short_opening.iter().rev())
            .take_while(|(long, short)| long == short)
            .count();
        assert_eq!(
            report.shifted_tokens,
            shared_closing + chunk.len() - 1,
            "the run starts at the closing tokens both openings end with and stops one short of the prompt: {report:?}"
        );
        let (_, widths) = model
            .declared_layer_cache_names_and_widths()
            .expect("declared layer widths");
        let fresh_state = prefill_state(model, &new_prompt);
        let length = new_prompt.len();
        let window = ChunkRun {
            old_start: length - 512,
            new_start: length - 512,
            len: 512,
        };
        let deep = ChunkRun {
            old_start: short_opening.len() + 2200,
            new_start: short_opening.len() + 2200,
            len: length - short_opening.len() - 2200,
        };
        let stored_rows: Vec<(usize, f32, f32)> = {
            let cache = model
                .prompt_cache
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            let state = cache
                .entry_states()
                .next()
                .expect("the request stored its entry");
            (0..=4)
                .map(|layer| {
                    let run = if layer == 4 { &deep } else { &window };
                    let stored = rows_of(state, &widths, layer, run);
                    let fresh = rows_of(&fresh_state, &widths, layer, run);
                    let (worst, scale) = key_difference(&stored, &fresh);
                    (layer, worst, scale)
                })
                .collect()
        };

        for (layer, worst, scale) in &stored_rows {
            eprintln!(
                "SHIFT_PIPELINE_ROWS layer={layer} max_key_diff={worst:.6e} max_key_abs={scale:.4}"
            );
            assert!(
                *worst <= 2e-3 * scale,
                "layer {layer}: stored rows differ from a fresh prefill by {worst:e} against scale {scale}"
            );
        }
    });
}

/// The squash through the pipeline: every layer's rows over the kept turns, in
/// the entry the shifted request leaves behind, are the rows lifted from the
/// old prefill (to within f32 noise) and are not the rows a fresh prefill of
/// the squashed prompt stores. This is what separates a wrong write from the
/// rows' own dependence on the context they were computed in.
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn the_squash_stores_the_lifted_rows_in_every_layer_not_the_fresh_ones() {
    with_model(|model| {
        let run = squashed_history(model, REUSE_MIN);
        let case = &run.case;
        let (_, widths) = model
            .declared_layer_cache_names_and_widths()
            .expect("declared layer widths");
        let new_start = case.system_end + case.summary.len();
        let old_state = prefill_state(model, &case.old_prompt);
        let compared = run
            .shifted
            .report
            .shifted_tokens
            .min(old_state.cached_len - case.kept_start);
        let lifted_run = ChunkRun {
            old_start: case.kept_start,
            new_start,
            len: compared,
        };
        let fresh_state = prefill_state(model, &run.new_prompt[..lifted_run.new_end()]);
        let delta = isize::try_from(new_start).expect("fits")
            - isize::try_from(case.kept_start).expect("fits");
        let rotations = model.delta_rotations(&shifting_config(REUSE_MIN), delta);
        let lifted = extract_run(&old_state, lifted_run, &widths, &rotations)
            .expect("the old state lifts the run");
        let comparisons: Vec<(usize, bool, f32, f32, f32)> = {
            let cache = model
                .prompt_cache
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            let state = cache
                .entry_states()
                .next()
                .expect("the request stored its entry");
            lifted
                .layers
                .iter()
                .map(|lifted_rows| {
                    let layer = lifted_rows.layer;
                    let LayerPadRowWidths::Attention { even_odd_row, .. } = widths[layer] else {
                        panic!("layer {layer} owns rows");
                    };
                    let ring = matches!(
                        &state.layer_caches[layer],
                        LayerCacheState::Attention(cache) if cache.ring_geometry().is_some()
                    );
                    let skipped = if ring {
                        run.shifted.report.shifted_tokens - compared
                    } else {
                        0
                    };
                    let count = lifted_rows.k_even.len() / even_odd_row - skipped;
                    let lifted_rows = &LayerRows {
                        layer,
                        k_even: lifted_rows.k_even[skipped * even_odd_row..].to_vec(),
                        k_odd: lifted_rows.k_odd[skipped * even_odd_row..].to_vec(),
                        v: Vec::new(),
                    };
                    let window = ChunkRun {
                        old_start: lifted_run.new_end() - count,
                        new_start: lifted_run.new_end() - count,
                        len: count,
                    };
                    let stored = rows_of(state, &widths, layer, &window);
                    let fresh = rows_of(&fresh_state, &widths, layer, &window);
                    let (to_lifted, scale) = key_difference(&stored, lifted_rows);
                    let (to_fresh, _) = key_difference(&stored, &fresh);
                    (layer, ring, to_lifted, to_fresh, scale)
                })
                .collect()
        };

        for (layer, ring, to_lifted, to_fresh, scale) in &comparisons {
            eprintln!(
                "SQUASH_ROWS layer={layer} ring={ring} stored_vs_lifted={to_lifted:.6e} stored_vs_fresh={to_fresh:.6e} max_key_abs={scale:.4}"
            );
            assert!(
                *to_lifted <= 2e-3 * scale,
                "layer {layer}: the stored rows are not the lifted rows ({to_lifted:e} against scale {scale})"
            );
        }
        assert_eq!(comparisons.len(), 15);
    });
}
