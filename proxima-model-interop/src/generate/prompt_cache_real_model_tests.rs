#![allow(clippy::expect_used)]

use core::ops::ControlFlow;
use std::fs::File;

use memmap2::Mmap;
use proxima_gguf::parse_complete;

use super::prefix_resume_long_prompt_tests::{chat_prompt, corpus_document, greedy_config};
use super::prompt_cache::longest_common_prefix;
use super::wants_bos;
use crate::LoadedModel;
use crate::generate::{CachePath, CacheReport, MissReason};
use crate::serving::{PromptCacheConfig, ServingConfig, SpeculativeConfig};

const GENERATED_TOKENS: usize = 16;
const REWRITTEN_TOKENS: usize = 50;
const BEYOND_SLACK_TOKENS: usize = 300;

fn cached_config(speculative: SpeculativeConfig<'static>) -> ServingConfig<'static> {
    ServingConfig {
        speculative,
        prompt_cache: PromptCacheConfig::standard(),
        ..greedy_config()
    }
}

fn uncached_config(speculative: SpeculativeConfig<'static>) -> ServingConfig<'static> {
    ServingConfig {
        speculative,
        prompt_cache: PromptCacheConfig::off(),
        ..greedy_config()
    }
}

fn with_model<T>(body: impl FnOnce(&LoadedModel<'_>) -> T) -> T {
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

fn encode_opening(model: &LoadedModel<'_>, text: &str) -> Vec<u32> {
    proxima_tokenizer::encode_with_bos_eos(
        text,
        &model.vocab,
        wants_bos(&model.vocab),
        model.vocab.add_eos_token().unwrap_or(false),
    )
    .expect("tokenize the opening prompt")
}

fn encode_continuation(model: &LoadedModel<'_>, text: &str) -> Vec<u32> {
    proxima_tokenizer::encode_with_bos_eos(text, &model.vocab, false, false)
        .expect("tokenize a continuation")
}

fn next_user_turn(document: &str, from_char: usize, chars: usize) -> String {
    let excerpt: String = document.chars().skip(from_char).take(chars).collect();
    format!("<turn|>\n<|turn>user\n{excerpt}<turn|>\n<|turn>model\n")
}

struct TurnOutcome {
    generated: Vec<u32>,
    stopped_by_eos: bool,
    report: CacheReport,
}

fn run_cached(model: &LoadedModel<'_>, config: ServingConfig<'_>, ids: &[u32]) -> TurnOutcome {
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

fn run_fresh(model: &LoadedModel<'_>, config: ServingConfig<'_>, ids: &[u32]) -> Vec<u32> {
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
fn cached_tokens_after(prompt_len: usize, outcome: &TurnOutcome) -> usize {
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
) -> Rewrite {
    with_model(|model| {
        let document = corpus_document("rag004");
        let opening_ids = encode_opening(model, &chat_prompt(3000));
        let first = run_cached(model, cached_config(speculative), &opening_ids);
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

        let cached = run_cached(model, cached_config(speculative), &rewritten_ids);
        let fresh = run_fresh(model, uncached_config(speculative), &rewritten_ids);

        assert_eq!(
            cached.generated, fresh,
            "ids after rewriting {rewritten_tokens} tokens must equal a full prefill"
        );
        Rewrite {
            report: cached.report,
            kept_tokens,
            tail_tokens: rewritten_tail.len(),
        }
    })
}

/// AC3: a 50-token rewrite sits inside the ring's 256-row slack, so the path
/// is `rewind` and the prefilled tokens are the 50 replaced ones plus the new
/// tokens.
fn rewrite_within_slack(speculative: SpeculativeConfig<'static>) {
    let rewrite = rewrite_last_tokens(speculative, REWRITTEN_TOKENS);

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

/// R3's other half: a rewrite past the slack cannot reuse the wrapped rings,
/// so the request prefills in full, says why, and still matches a full prefill.
fn rewrite_beyond_slack(speculative: SpeculativeConfig<'static>) {
    let rewritten_tokens = BEYOND_SLACK_TOKENS;
    let rewrite = rewrite_last_tokens(speculative, rewritten_tokens);

    assert_eq!(rewrite.report.path, CachePath::Miss, "{:?}", rewrite.report);
    assert_eq!(
        rewrite.report.miss,
        Some(MissReason::RingSlackExceeded {
            rewind_rows: rewritten_tokens,
            slack_rows: PromptCacheConfig::off().ring_rewind_slack as usize,
        })
    );
    assert_eq!(rewrite.report.lcp, 0);
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
fn rewriting_past_the_ring_slack_prefills_in_full_with_speculation_off() {
    rewrite_beyond_slack(SpeculativeConfig::none());
}

#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn rewriting_past_the_ring_slack_prefills_in_full_with_speculation_on() {
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
/// tokens, each with room for the same bytes again in checkpoints.
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn default_byte_budget_holds_four_8k_conversations_with_room_for_checkpoints() {
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
            4 * 2 * bytes_8k <= config.byte_budget as usize,
            "four 8k entries and as many checkpoint bytes are {} bytes, budget {}",
            4 * 2 * bytes_8k,
            config.byte_budget
        );
    });
}
