#![allow(clippy::expect_used)]

use core::ops::ControlFlow;

use omega::metal::PIPELINE_MISSES;
use proxima_gguf::parse_complete;
use proxima_numeric::NumericFormat;

use super::chunked_prefill_tests::dense_checkpoint;
use super::{LoadedModel, ServingConfig};
use crate::serving::{GPU_LAYERS_ALL, PromptCacheConfig};

const CHUNK_ROWS: usize = 24;
const PROMPT_ROWS: usize = 3 * CHUNK_ROWS / 2 + 5;
const NEW_TOKENS: usize = 6;

fn config() -> ServingConfig<'static> {
    ServingConfig {
        gpu_layers: GPU_LAYERS_ALL,
        kv_cache_key_quant: NumericFormat::Float32,
        kv_cache_value_quant: NumericFormat::Float32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: CHUNK_ROWS as u32,
        reasoning_budget: 0,
        temperature: 0.0,
        prompt_cache: PromptCacheConfig::off(),
        ..ServingConfig::default()
    }
}

fn prompt() -> Vec<u32> {
    (0..PROMPT_ROWS).map(|row| ((row * 11 + 5) % 250 + 1) as u32).collect()
}

fn request(model: &LoadedModel<'_>) -> Vec<u32> {
    model
        .generate_from_ids(&prompt(), NEW_TOKENS, &config(), &mut |_| ControlFlow::Continue(()))
        .expect("request on the tiny dense checkpoint")
        .0
}

#[test]
fn a_request_after_warm_prefill_matches_the_same_request_on_a_fresh_model() {
    let bytes = dense_checkpoint();
    let parsed = parse_complete(&bytes).expect("parse the dense fixture");
    let fresh = LoadedModel::load(&parsed, &bytes).expect("load a fresh model");
    let warmed = LoadedModel::load(&parsed, &bytes).expect("load the warmed model");

    let rows = warmed.warm_prefill(&config(), CHUNK_ROWS).expect("warm prefill");
    let expected = request(&fresh);
    let actual = request(&warmed);

    assert_eq!(rows, CHUNK_ROWS + CHUNK_ROWS / 2, "the throwaway prompt is one chunk plus a half-chunk tail");
    assert_eq!(expected.len(), NEW_TOKENS, "the reference request generated every token");
    assert_eq!(actual, expected, "warm_prefill changed a later request's token ids");
}

#[test]
fn warm_prefill_leaves_the_first_request_no_pipeline_to_compile() {
    let bytes = dense_checkpoint();
    let parsed = parse_complete(&bytes).expect("parse the dense fixture");
    let model = LoadedModel::load(&parsed, &bytes).expect("load the model");

    let (rows, warm_misses) = PIPELINE_MISSES.scoped(|| model.warm_prefill(&config(), CHUNK_ROWS).expect("warm prefill"));
    let (ids, request_misses) = PIPELINE_MISSES.scoped(|| request(&model));

    assert_eq!(rows, CHUNK_ROWS + CHUNK_ROWS / 2);
    assert_eq!(ids.len(), NEW_TOKENS);
    assert_eq!(request_misses, 0, "the request compiled a pipeline after warm_prefill (warm compiled {warm_misses})");
}

#[test]
fn warm_prefill_stores_no_prompt_cache_entry() {
    let bytes = dense_checkpoint();
    let parsed = parse_complete(&bytes).expect("parse the dense fixture");
    let model = LoadedModel::load(&parsed, &bytes).expect("load the model");
    let cached = ServingConfig { prompt_cache: PromptCacheConfig::default(), ..config() };

    model.warm_prefill(&cached, CHUNK_ROWS).expect("warm prefill");

    assert_eq!(model.prompt_cache.lock().entry_count(), 0, "warm_prefill left a prompt-cache entry");
}
