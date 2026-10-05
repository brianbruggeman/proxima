// workspace denies expect_used; a failed fixture must abort the test with its message
#![allow(clippy::expect_used)]

use super::chunked_prefill_tests::{config, gemma4_checkpoint, prefill_with, prompt_of};
use super::{LayerCacheState, LoadedModel, ServingConfig};
use crate::PromptCacheConfig;

#[test]
fn decode_seals_full_attention_layers_at_commit() {
    let bytes = gemma4_checkpoint();
    let parsed = proxima_gguf::pipe::parse_complete(&bytes).expect("parses the gemma4 fixture");
    let model = LoadedModel::load(&parsed, &bytes).expect("loads the gemma4 fixture");
    let serving_config = ServingConfig {
        prompt_cache: PromptCacheConfig { block_tokens: 8, seal_horizon_rows: 4, ..PromptCacheConfig::standard() },
        ..config(7)
    };

    let prefilled = prefill_with(&model, &prompt_of(40, '3'), &serving_config);

    assert_eq!(prefilled.state.len(), 40, "the prompt is cached in full");
    let (full, ring): (Vec<_>, Vec<_>) = prefilled
        .state
        .layer_caches
        .iter()
        .filter_map(|state| match state {
            LayerCacheState::Attention(cache) => Some(cache),
            _ => None,
        })
        .partition(|cache| cache.ring_geometry().is_none());
    assert_eq!(full.len(), 1, "one full-attention layer");
    assert_eq!(ring.len(), 2, "two sliding ring layers");
    assert_eq!(full[0].sealed_end, 32, "(40 - 4) / 8 * 8 rows seal");
    assert!(ring.iter().all(|cache| cache.sealed_end == 0), "a ring layer never seals");
}
