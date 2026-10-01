#![allow(clippy::expect_used)]

use core::ops::ControlFlow;
use std::fs::File;

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use proxima_gguf::types::GgmlType;

use crate::LoadedModel;
use crate::serving::{GPU_LAYERS_ALL, ServingConfig, SpeculativeConfig};

const SPECULATIVE_CORPUS: &str = include_str!("../../examples/data/speculative_corpus.jsonl");

const GREEDY_TOKENS: usize = 32;

/// gemma4-E2B's per-layer-embedding projection is `[rows, 1536] x [1536, 8960]`
/// dispatched as one 256-lane threadgroup per output: `rows * 8960 * 256`
/// threads must stay below `u32::MAX`, which holds through 1872 rows and not
/// one row further.
const LAST_ROW_COUNT_AT_FULL_LANE_WIDTH: usize = 1872;

pub(super) fn greedy_config() -> ServingConfig<'static> {
    ServingConfig {
        gpu_layers: GPU_LAYERS_ALL,
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        reasoning_budget: 0,
        temperature: 0.0,
        repeat_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        speculative: SpeculativeConfig::none(),
        ..ServingConfig::default()
    }
}

pub(super) fn corpus_document(id: &str) -> String {
    SPECULATIVE_CORPUS
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("corpus line is json"))
        .find(|record| record["id"] == id)
        .and_then(|record| record["prompt"].as_str().map(str::to_owned))
        .unwrap_or_else(|| panic!("corpus has no record with id {id}"))
}

pub(super) fn chat_prompt(document_chars: usize) -> String {
    let document: String = corpus_document("rag004")
        .chars()
        .take(document_chars)
        .collect();
    format!("<|turn>user\n{document}<turn|>\n<|turn>model\n")
}

struct Outcome {
    prefix_rows: usize,
    fresh: Vec<u32>,
    resumed: Vec<u32>,
}

fn split_at_model_turn_opener(templated: &str) -> (&str, &str) {
    let boundary = templated
        .rfind('\n')
        .expect("the chat template ends in a newline");
    templated.split_at(boundary)
}

/// Rows `prefill_prefix` caches for `document_chars` characters of the
/// document: the token count of the templated prompt up to the model-turn
/// opener, with the vocabulary's own BOS policy.
fn prefix_rows(model: &LoadedModel<'_>, document_chars: usize) -> usize {
    let templated = chat_prompt(document_chars);
    let (prefix_text, _) = split_at_model_turn_opener(&templated);
    proxima_tokenizer::encode_with_bos_eos(
        prefix_text,
        &model.vocab,
        super::wants_bos(&model.vocab),
        model.vocab.add_eos_token().unwrap_or(false),
    )
    .expect("tokenize the prefix text")
    .len()
}

/// The smallest `document_chars` whose prefix reaches
/// [`LAST_ROW_COUNT_AT_FULL_LANE_WIDTH`] rows, found by bisection over the
/// tokenizer so the case pair straddles the overflow row count whatever the
/// vocabulary's characters-per-token ratio is.
fn first_document_chars_past_the_overflow(model: &LoadedModel<'_>) -> usize {
    let mut below = 0;
    let mut at_or_past = 8192;
    assert!(
        prefix_rows(model, at_or_past) >= LAST_ROW_COUNT_AT_FULL_LANE_WIDTH,
        "the corpus document must be long enough to pass the overflow row count"
    );
    while at_or_past - below > 1 {
        let middle = below + (at_or_past - below) / 2;
        if prefix_rows(model, middle) >= LAST_ROW_COUNT_AT_FULL_LANE_WIDTH {
            at_or_past = middle;
        } else {
            below = middle;
        }
    }
    at_or_past
}

fn decode_fresh_and_resumed(model: &LoadedModel<'_>, document_chars: usize) -> Outcome {
    let templated = chat_prompt(document_chars);
    let (prefix_text, suffix) = split_at_model_turn_opener(&templated);

    let (fresh, ..) = model
        .generate_with_serving_config(&templated, GREEDY_TOKENS, greedy_config())
        .expect("fresh full-prompt greedy decode");
    let prefix = model
        .prefill_prefix(prefix_text, &greedy_config())
        .expect("prefill the templated prompt through the model-turn opener");
    let (resumed, ..) = model
        .generate_from_prefix(
            &prefix,
            suffix,
            GREEDY_TOKENS,
            &greedy_config(),
            &mut |_event| ControlFlow::Continue(()),
        )
        .expect("resume greedy decode from the cached prefix");
    Outcome {
        prefix_rows: prefix.len(),
        fresh,
        resumed,
    }
}

/// `prefill_prefix` + `generate_from_prefix` against a fresh full-prompt
/// decode on real RAG text (`speculative_corpus.jsonl` `rag004`, truncated at
/// three lengths). Below the threshold both paths dispatch the per-layer
/// projection at full lane width; above it the prefix prefill (one row
/// fewer than the fresh prompt) and the fresh prefill both overflowed the
/// projection's 32-bit thread index, so every output past the first
/// `(threads - 2^32) / 256` came back zero, and the resumed decode -- whose
/// single suffix row is projected in its own small dispatch -- diverged from
/// the fresh one from the very first token.
///
/// A consistency check between two device paths, NOT a correctness oracle:
/// the FFN gate/up reduces overflow from ~1,365 rows, so between 1,365 and
/// 1,872 rows both paths share the same overflowed prefill and agree while
/// both wrong. The oracle is llama.cpp on the same token ids:
/// `speculative_bench --llama-parity`.
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn resumed_decode_matches_fresh_decode_across_the_thread_index_overflow_row_count() {
    let model_path = crate::test_support::gemma4_e2b_gguf_path();
    crate::test_support::require_fixture(&model_path, Some("PROXIMA_GEMMA4_E2B_GGUF"));
    let file = File::open(&model_path).expect("open the real gemma4-E2B checkpoint");
    // SAFETY: read-only mapping of a file nothing else writes during the test.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the real gemma4-E2B checkpoint");
    let bytes: &[u8] = &mapping;
    let parsed = parse_complete(bytes).expect("parse the real gemma4-E2B header");
    let model = LoadedModel::load(&parsed, bytes).expect("bind the real gemma4-E2B checkpoint");

    let first_past = first_document_chars_past_the_overflow(&model);
    let outcomes: Vec<(usize, Outcome)> = [first_past - 1, first_past, first_past + 900]
        .into_iter()
        .map(|document_chars| {
            (
                document_chars,
                decode_fresh_and_resumed(&model, document_chars),
            )
        })
        .collect();

    let (_, below) = &outcomes[0];
    assert!(
        below.prefix_rows < LAST_ROW_COUNT_AT_FULL_LANE_WIDTH,
        "the shortest case must sit below the overflow row count, got {} prefix rows",
        below.prefix_rows
    );
    for (document_chars, outcome) in &outcomes[1..] {
        assert!(
            outcome.prefix_rows >= LAST_ROW_COUNT_AT_FULL_LANE_WIDTH,
            "document_chars={document_chars} must reach the overflow row count, got {} prefix rows",
            outcome.prefix_rows
        );
    }
    for (document_chars, outcome) in &outcomes {
        assert!(
            !outcome.fresh.is_empty(),
            "document_chars={document_chars}: the fresh decode must produce tokens"
        );
        assert_eq!(
            outcome.resumed, outcome.fresh,
            "document_chars={document_chars} prefix_rows={}: resuming from the cached prefix must \
             sample the same greedy tokens as decoding the whole prompt fresh",
            outcome.prefix_rows
        );
    }
}
