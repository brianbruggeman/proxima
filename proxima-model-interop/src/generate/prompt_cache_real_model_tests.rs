#![allow(clippy::expect_used)]

use core::ops::ControlFlow;
use std::fs::File;

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use proxima_gguf::types::GgmlType;

use super::prefix_resume_long_prompt_tests::{chat_prompt, corpus_document, greedy_config};
use super::prompt_cache::longest_common_prefix;
use super::ring_checkpoint::RingCheckpoint;
use super::wants_bos;
use crate::InteropError;
use crate::LoadedModel;
use crate::RopeScaling;
use crate::generate::{CachePath, CacheReport, MissReason};
use crate::serving::{PromptCacheConfig, ServingConfig, SpeculativeConfig};

const GENERATED_TOKENS: usize = 16;
const REWRITTEN_TOKENS: usize = 50;
const BEYOND_SLACK_TOKENS: usize = 300;

pub(super) fn cached_config(speculative: SpeculativeConfig<'static>) -> ServingConfig<'static> {
    checkpointed_config(
        speculative,
        PromptCacheConfig::standard().checkpoint_interval,
    )
}

fn checkpointed_config(
    speculative: SpeculativeConfig<'static>,
    checkpoint_interval: u32,
) -> ServingConfig<'static> {
    ServingConfig {
        speculative,
        prompt_cache: PromptCacheConfig {
            checkpoint_interval,
            max_checkpoints: 8,
            ..PromptCacheConfig::standard()
        },
        ..greedy_config()
    }
}

pub(super) fn uncached_config(speculative: SpeculativeConfig<'static>) -> ServingConfig<'static> {
    ServingConfig {
        speculative,
        prompt_cache: PromptCacheConfig::off(),
        ..greedy_config()
    }
}

pub(super) fn with_model<T>(body: impl FnOnce(&LoadedModel<'_>) -> T) -> T {
    let model_path = crate::test_support::gemma4_e2b_gguf_path();
    crate::test_support::require_fixture(&model_path, Some("PROXIMA_GEMMA4_E2B_GGUF"));
    let file = File::open(&model_path).expect("open the real gemma4-E2B checkpoint");
    // SAFETY: read-only mapping of a file nothing else writes during the test.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the real gemma4-E2B checkpoint");
    let bytes: &[u8] = &mapping;
    let parsed = parse_complete(bytes).expect("parse the real gemma4-E2B header");
    let model = LoadedModel::load(&parsed, bytes).expect("bind the real gemma4-E2B checkpoint");
    body(&model)
}

pub(super) fn encode_opening(model: &LoadedModel<'_>, text: &str) -> Vec<u32> {
    proxima_tokenizer::encode_with_bos_eos(
        text,
        &model.vocab,
        wants_bos(&model.vocab),
        model.vocab.add_eos_token().unwrap_or(false),
    )
    .expect("tokenize the opening prompt")
}

pub(super) fn encode_continuation(model: &LoadedModel<'_>, text: &str) -> Vec<u32> {
    proxima_tokenizer::encode_with_bos_eos(text, &model.vocab, false, false)
        .expect("tokenize a continuation")
}

fn next_user_turn(document: &str, from_char: usize, chars: usize) -> String {
    let excerpt: String = document.chars().skip(from_char).take(chars).collect();
    format!("<turn|>\n<|turn>user\n{excerpt}<turn|>\n<|turn>model\n")
}

pub(super) struct TurnOutcome {
    pub(super) generated: Vec<u32>,
    pub(super) stopped_by_eos: bool,
    pub(super) report: CacheReport,
}

pub(super) fn run_cached(
    model: &LoadedModel<'_>,
    config: ServingConfig<'_>,
    ids: &[u32],
) -> TurnOutcome {
    let (generated, _text, stopped_by_eos) = model
        .generate_from_ids(ids, GENERATED_TOKENS, &config, &mut |_event| {
            ControlFlow::Continue(())
        })
        .expect("generate through the prompt cache");
    let report = model
        .last_prompt_cache_report()
        .expect("a cached request records its report");
    TurnOutcome {
        generated,
        stopped_by_eos,
        report,
    }
}

pub(super) fn run_fresh(
    model: &LoadedModel<'_>,
    config: ServingConfig<'_>,
    ids: &[u32],
) -> Vec<u32> {
    let (generated, ..) = model
        .generate_from_ids(ids, GENERATED_TOKENS, &config, &mut |_event| {
            ControlFlow::Continue(())
        })
        .expect("generate from a full prefill");
    generated
}

/// Tokens the cache holds after a request: the prompt plus every generated
/// token the forward pass consumed (the last sampled one is not forwarded
/// unless the model stopped on its end token).
pub(super) fn cached_tokens_after(prompt_len: usize, outcome: &TurnOutcome) -> usize {
    let unforwarded = usize::from(!outcome.stopped_by_eos);
    prompt_len + outcome.generated.len() - unforwarded.min(outcome.generated.len())
}

/// AC2: a 3-turn transcript where each turn's prompt is the previous turn's
/// prompt plus its generated tokens plus new tokens. Every turn's greedy ids
/// equal the ids of a full prefill of the same prompt, the path is `extend`
/// from turn 2, and the prefilled tokens are the tokens past what the cache
/// already held.
fn three_turn_extension(speculative: SpeculativeConfig<'static>) {
    with_model(|model| {
        let document = corpus_document("rag004");
        let mut prompt_ids = encode_opening(model, &chat_prompt(1200));
        let mut held_tokens = 0;

        for turn in 0..3_usize {
            let cached = run_cached(model, cached_config(speculative), &prompt_ids);
            let fresh = run_fresh(model, uncached_config(speculative), &prompt_ids);
            assert_eq!(
                cached.generated, fresh,
                "turn {turn}: ids through the cache must equal ids from a full prefill"
            );
            if turn == 0 {
                assert_eq!(cached.report.path, CachePath::Miss);
                assert_eq!(cached.report.prefilled_tokens, prompt_ids.len());
            } else {
                assert_eq!(
                    cached.report.path,
                    CachePath::Extend,
                    "turn {turn}: {:?}",
                    cached.report
                );
                assert_eq!(cached.report.lcp, held_tokens);
                assert_eq!(
                    cached.report.prefilled_tokens,
                    prompt_ids.len() - held_tokens
                );
            }
            held_tokens = cached_tokens_after(prompt_ids.len(), &cached);
            prompt_ids.extend_from_slice(&cached.generated);
            let next_turn = next_user_turn(&document, 1200 + turn * 500, 500);
            prompt_ids.extend(encode_continuation(model, &next_turn));
        }
    });
}

#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn three_turn_extension_matches_fresh_prefill_with_speculation_off() {
    three_turn_extension(SpeculativeConfig::none());
}

#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn three_turn_extension_matches_fresh_prefill_with_speculation_on() {
    three_turn_extension(SpeculativeConfig::default());
}

struct Rewrite {
    report: CacheReport,
    opening_tokens: usize,
    kept_tokens: usize,
    tail_tokens: usize,
}

/// Turn 2 keeps turn 1's prompt and generated tokens except the last
/// `rewritten_tokens`, which it replaces with new text. Turn 1 is long enough
/// (about 1,000 tokens) that the sliding-window rings, 512 rows wide, have
/// wrapped. Whatever path the cache takes, the ids must equal a full prefill's.
fn rewrite_last_tokens(
    speculative: SpeculativeConfig<'static>,
    rewritten_tokens: usize,
    opening: &str,
    cached: ServingConfig<'static>,
) -> Rewrite {
    with_model(|model| {
        let document = corpus_document("rag004");
        let opening_ids = encode_opening(model, opening);
        let first = run_cached(model, cached, &opening_ids);
        let held_tokens = cached_tokens_after(opening_ids.len(), &first);
        let mut stored_ids = opening_ids.clone();
        stored_ids.extend_from_slice(&first.generated);
        stored_ids.truncate(held_tokens);

        let kept_tokens = held_tokens - rewritten_tokens;
        let rewritten_tail = encode_continuation(
            model,
            &format!(
                "Disregard the passage above and answer in one sentence. {}",
                next_user_turn(&document, 3200, 1500)
            ),
        );
        assert_ne!(
            rewritten_tail[0], stored_ids[kept_tokens],
            "the rewrite must diverge at its first token"
        );
        let mut rewritten_ids = stored_ids[..kept_tokens].to_vec();
        rewritten_ids.extend_from_slice(&rewritten_tail);
        assert_eq!(
            longest_common_prefix(&stored_ids, &rewritten_ids),
            kept_tokens
        );

        let rewritten = run_cached(model, cached, &rewritten_ids);
        let fresh = run_fresh(model, uncached_config(speculative), &rewritten_ids);

        assert_eq!(
            rewritten.generated, fresh,
            "ids after rewriting {rewritten_tokens} tokens must equal a full prefill"
        );
        Rewrite {
            report: rewritten.report,
            opening_tokens: opening_ids.len(),
            kept_tokens,
            tail_tokens: rewritten_tail.len(),
        }
    })
}

/// AC3: a 50-token rewrite sits inside the ring's 256-row slack, so the path
/// is `rewind` and the prefilled tokens are the 50 replaced ones plus the new
/// tokens.
fn rewrite_within_slack(speculative: SpeculativeConfig<'static>) {
    let rewrite = rewrite_last_tokens(
        speculative,
        REWRITTEN_TOKENS,
        &chat_prompt(3000),
        cached_config(speculative),
    );

    assert_eq!(
        rewrite.report.path,
        CachePath::Rewind,
        "{:?}",
        rewrite.report
    );
    assert_eq!(rewrite.report.lcp, rewrite.kept_tokens);
    assert!(
        rewrite.tail_tokens > REWRITTEN_TOKENS,
        "the tail must carry new tokens past the 50 it replaces"
    );
    assert_eq!(rewrite.report.prefilled_tokens, rewrite.tail_tokens);
}

/// R3's other half, now served by R4: a 300-token rewrite on a roughly
/// 1,000-token transcript is past the 256-row slack, but turn 1 stopped its
/// prefill every 256 tokens to snapshot the rings, so the request restores the
/// checkpoint at or before the shared prefix instead of prefilling in full.
fn rewrite_beyond_slack(speculative: SpeculativeConfig<'static>) {
    const INTERVAL: u32 = 256;
    let rewrite = rewrite_last_tokens(
        speculative,
        BEYOND_SLACK_TOKENS,
        &chat_prompt(3000),
        checkpointed_config(speculative, INTERVAL),
    );

    assert_restored_nearest_checkpoint(&rewrite, INTERVAL as usize);
}

/// AC4: a 2,000-token rewrite on a transcript of more than 3,000 tokens is
/// far past the ring's slack. Turn 1 snapshotted the rings every 512 tokens,
/// so the request restores the newest snapshot at or before the shared prefix
/// and prefills the gap to the prefix plus the new tokens. Whatever it does,
/// its ids equal a full prefill's (asserted inside `rewrite_last_tokens`).
fn rewrite_two_thousand_tokens_back(speculative: SpeculativeConfig<'static>) {
    const INTERVAL: u32 = 512;
    const REWRITTEN: usize = 2000;
    let rewrite = rewrite_last_tokens(
        speculative,
        REWRITTEN,
        &long_chat_prompt(12_000),
        checkpointed_config(speculative, INTERVAL),
    );

    assert!(
        rewrite.opening_tokens > 3000,
        "the transcript must exceed 3,000 tokens, got {}",
        rewrite.opening_tokens
    );
    assert_restored_nearest_checkpoint(&rewrite, INTERVAL as usize);
}

fn assert_restored_nearest_checkpoint(rewrite: &Rewrite, interval: usize) {
    let report = &rewrite.report;
    eprintln!(
        "CHECKPOINT_RESTORE opening_tokens={} kept_tokens={} tail_tokens={} {report:?}",
        rewrite.opening_tokens, rewrite.kept_tokens, rewrite.tail_tokens
    );
    let checkpoint = rewrite.kept_tokens / interval * interval;

    assert_eq!(report.path, CachePath::Checkpoint, "{report:?}");
    assert_eq!(report.lcp, rewrite.kept_tokens, "{report:?}");
    assert_eq!(report.reused_tokens, checkpoint, "{report:?}");
    assert_eq!(
        report.prefilled_tokens,
        (rewrite.kept_tokens - checkpoint) + rewrite.tail_tokens,
        "prefilled tokens are the gap from the checkpoint to the shared prefix plus the new tokens: {report:?}"
    );
}

#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn rewriting_two_thousand_tokens_back_restores_a_checkpoint_with_speculation_off() {
    rewrite_two_thousand_tokens_back(SpeculativeConfig::none());
}

#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn rewriting_two_thousand_tokens_back_restores_a_checkpoint_with_speculation_on() {
    rewrite_two_thousand_tokens_back(SpeculativeConfig::default());
}

#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn rewriting_the_last_fifty_tokens_rewinds_with_speculation_off() {
    rewrite_within_slack(SpeculativeConfig::none());
}

#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn rewriting_the_last_fifty_tokens_rewinds_with_speculation_on() {
    rewrite_within_slack(SpeculativeConfig::default());
}

#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn rewriting_past_the_ring_slack_restores_a_checkpoint_with_speculation_off() {
    rewrite_beyond_slack(SpeculativeConfig::none());
}

#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn rewriting_past_the_ring_slack_restores_a_checkpoint_with_speculation_on() {
    rewrite_beyond_slack(SpeculativeConfig::default());
}

fn long_document(chars: usize) -> String {
    ["rag011", "rag016", "rag013", "rag008", "rag012", "rag004"]
        .iter()
        .map(|id| corpus_document(id))
        .collect::<Vec<_>>()
        .join("\n\n")
        .chars()
        .take(chars)
        .collect()
}

pub(super) fn long_chat_prompt(chars: usize) -> String {
    format!(
        "<|turn>user\n{}<turn|>\n<|turn>model\n",
        long_document(chars)
    )
}

fn stored_bytes_after_prefill(model: &LoadedModel<'_>, tokens: usize) -> (usize, usize) {
    let mut ids = encode_opening(model, &long_chat_prompt(40_000));
    ids.truncate(tokens);
    let outcome = run_cached(model, cached_config(SpeculativeConfig::none()), &ids);
    (
        cached_tokens_after(ids.len(), &outcome),
        model.prompt_cache_bytes(),
    )
}

/// The default byte budget is justified by what one gemma4-E2B entry costs:
/// the host bytes the cache holds after a 2,048- and an 8,192-token request.
/// The default has to hold the four entries `max_entries` allows at 8k
/// tokens, with the checkpoints the default config takes inside each.
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn default_byte_budget_holds_four_8k_conversations_with_their_checkpoints() {
    with_model(|model| {
        let (tokens_2k, bytes_2k) = stored_bytes_after_prefill(model, 2048);
        let (tokens_8k, bytes_8k) = stored_bytes_after_prefill(model, 8192);
        eprintln!("PREFIX_STATE_BYTES cached_tokens={tokens_2k} bytes={bytes_2k}");
        eprintln!("PREFIX_STATE_BYTES cached_tokens={tokens_8k} bytes={bytes_8k}");
        let config = PromptCacheConfig::standard();

        assert!(
            bytes_8k > bytes_2k,
            "an entry grows with the tokens it holds"
        );
        assert!(
            4 * bytes_8k <= config.byte_budget as usize,
            "four 8k entries are {} bytes, budget {}",
            4 * bytes_8k,
            config.byte_budget
        );
    });
}

/// Sizing a checkpoint: the twelve sliding layers of gemma4-E2B each hold
/// their 512-row window of `k_even` (128), `k_odd` (128) and `v` (256) floats,
/// 2,048 bytes a row, 12,582,912 bytes in all, whatever the prompt length.
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn a_gemma4_checkpoint_is_twelve_mebibytes() {
    with_model(|model| {
        let state = model
            .prefill_prefix(
                &long_chat_prompt(6000),
                &uncached_config(SpeculativeConfig::none()),
            )
            .expect("prefill a prompt past the 512-row window");

        let checkpoint = RingCheckpoint::capture(&state).expect("a gemma4 state is captured");

        eprintln!(
            "CHECKPOINT_BYTES position={} bytes={}",
            checkpoint.position(),
            checkpoint.byte_len()
        );
        assert!(checkpoint.position() > 512);
        assert_eq!(checkpoint.byte_len(), 12 * 512 * 2048);
    });
}

/// A request under another rope scaling on the same `LoadedModel` and prompt
/// finds the first arm's entry, whose key rows were rotated under the first
/// arm's scaling. It must miss with `ConfigMismatch`, generate what a fresh
/// prefill under its own scaling generates, and leave the first arm's entry
/// for the first arm to reuse.
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn another_rope_scaling_on_the_same_model_misses_and_never_reuses_the_other_arms_rows() {
    with_model(|model| {
        let scaling_b = RopeScaling::Linear { factor: 2.0 };
        assert_ne!(
            model.rope_scaling, scaling_b,
            "arm B must differ from the checkpoint's own scaling"
        );
        let arm_a = cached_config(SpeculativeConfig::none());
        let arm_b = ServingConfig {
            rope_scaling: Some(scaling_b),
            ..arm_a
        };
        let fresh_b = ServingConfig {
            prompt_cache: PromptCacheConfig::off(),
            ..arm_b
        };
        let prompt_ids = encode_opening(model, &chat_prompt(1200));

        let first_a = run_cached(model, arm_a, &prompt_ids);
        let first_b = run_cached(model, arm_b, &prompt_ids);
        let expected_b = run_fresh(model, fresh_b, &prompt_ids);
        let second_a = run_cached(model, arm_a, &prompt_ids);

        assert_eq!(first_a.report.miss, Some(MissReason::Empty));
        assert_eq!(
            first_b.generated, expected_b,
            "arm B's ids must equal a fresh prefill under arm B's scaling"
        );
        assert_eq!(first_b.report.path, CachePath::Miss);
        assert_eq!(
            first_b.report.miss,
            Some(MissReason::ConfigMismatch),
            "arm B found arm A's entry"
        );
        assert_ne!(
            second_a.report.path,
            CachePath::Miss,
            "arm A must reuse its own entry after arm B stored one"
        );
        assert!(second_a.report.reused_tokens > 0);
        assert_eq!(second_a.generated, first_a.generated);
    });
}

/// A request whose serving config `apply_serving_config` rejects errors
/// before any forward, so the entry a prior request stored is still there for
/// the next valid request.
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo"]
fn a_request_rejected_by_the_serving_config_leaves_the_stored_entry_for_the_next_one() {
    with_model(|model| {
        let valid = ServingConfig {
            gpu_layers: 0,
            ..cached_config(SpeculativeConfig::none())
        };
        let rejected = ServingConfig {
            kv_cache_key_quant: GgmlType::Q8_0,
            kv_cache_value_quant: GgmlType::Q8_0,
            ..valid
        };
        let mut prompt_ids = encode_opening(model, "The quick brown fox jumps over the lazy dog. ");
        let first = run_cached(model, valid, &prompt_ids);
        let stored_bytes = model.prompt_cache_bytes();
        prompt_ids.extend(&first.generated);

        let refusal = model.generate_from_ids(&prompt_ids, 1, &rejected, &mut |_event| {
            ControlFlow::Continue(())
        });
        let second = run_cached(model, valid, &prompt_ids);

        assert!(stored_bytes > 0, "the first request must store an entry");
        assert!(
            matches!(refusal, Err(InteropError::UnsupportedServingConfig(_))),
            "Q8 KV must be rejected, got {refusal:?}"
        );
        assert_ne!(
            second.report.path,
            CachePath::Miss,
            "the rejected request took the entry"
        );
        assert!(second.report.reused_tokens > 0);
    });
}
