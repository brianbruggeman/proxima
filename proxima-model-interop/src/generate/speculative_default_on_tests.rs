#![allow(clippy::expect_used)]

use core::ops::ControlFlow;
use std::fs::File;

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use proxima_gguf::types::GgmlType;

use crate::serving::{GPU_LAYERS_ALL, ServingConfig, SpeculativeConfig};
use crate::{LoadedModel, SpeculativeDecodeStats, TokenEvent};

const REAL_GEMMA4_E2B_GGUF_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

const MAX_TOKENS: usize = 40;

/// `examples/speculative_decode_parity.rs`'s own default prompt: four repeats
/// of a real two-sentence paragraph, so greedy and seeded-sampled decodes
/// both re-enter the repeated text and give `ngram-simple` something to draft.
fn repeated_paragraph_prompt() -> String {
    const PARAGRAPH: &str = "The quick brown fox jumps over the lazy dog while a curious cat \
         watches quietly from the garden wall. Pack my box with five dozen liquor jugs before \
         the delivery truck arrives at noon. ";
    PARAGRAPH.repeat(4)
}

fn shipped_default_config() -> ServingConfig<'static> {
    ServingConfig {
        gpu_layers: GPU_LAYERS_ALL,
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        reasoning_budget: 0,
        ..ServingConfig::default()
    }
}

fn greedy_config() -> ServingConfig<'static> {
    ServingConfig {
        temperature: 0.0,
        repeat_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        ..shipped_default_config()
    }
}

fn sampled_config() -> ServingConfig<'static> {
    ServingConfig {
        temperature: 0.8,
        top_k: 40,
        top_p: 0.9,
        min_p: 0.05,
        repeat_penalty: 1.1,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        seed: 7,
        ..shipped_default_config()
    }
}

fn decode(
    model: &LoadedModel<'_>,
    config: ServingConfig<'_>,
) -> (Vec<u32>, SpeculativeDecodeStats) {
    let mut stats = SpeculativeDecodeStats::default();
    let mut on_token = |_event: TokenEvent<'_>| ControlFlow::Continue(());
    let (ids, _text, _eos) = model
        .generate_streaming_with_speculative_stats(
            &repeated_paragraph_prompt(),
            MAX_TOKENS,
            config,
            &mut on_token,
            &mut stats,
            None,
        )
        .expect("decode on the real gemma4-E2B checkpoint");
    (ids, stats)
}

fn assert_default_matches_off(config: ServingConfig<'_>, label: &str) {
    crate::test_support::require_fixture(REAL_GEMMA4_E2B_GGUF_PATH, None);
    let file = File::open(REAL_GEMMA4_E2B_GGUF_PATH).expect("open the real gemma4-E2B checkpoint");
    // SAFETY: read-only mapping of a file nothing else writes during the test.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the real gemma4-E2B checkpoint");
    let bytes: &[u8] = &mapping;
    let parsed = parse_complete(bytes).expect("parse the real gemma4-E2B header");
    let model = LoadedModel::load(&parsed, bytes).expect("bind the real gemma4-E2B checkpoint");

    let (on_ids, on_stats) = decode(&model, config);
    let (off_ids, off_stats) = decode(&model, config.with_speculative(SpeculativeConfig::none()));

    assert_eq!(on_ids, off_ids, "{label}: default speculation must not change the output");
    assert!(!on_ids.is_empty(), "{label}: the decode must produce tokens");
    assert!(
        on_stats.verify_steps >= 1 && on_stats.drafted_total >= 1,
        "{label}: the default (ngram-simple) must fire on this repeated prompt: {on_stats:?}"
    );
    assert_eq!(
        off_stats,
        SpeculativeDecodeStats::default(),
        "{label}: off must take the non-speculative path (0 verify steps, 0 drafted)"
    );
}

#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn default_speculation_matches_off_greedy_on_real_gemma4_e2b() {
    assert_default_matches_off(greedy_config(), "greedy");
}

#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn default_speculation_matches_off_sampled_on_real_gemma4_e2b() {
    assert_default_matches_off(sampled_config(), "sampled seed 7");
}
