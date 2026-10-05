// workspace denies expect_used; a failed fixture must abort the test with its message
#![allow(clippy::expect_used)]

use super::chunked_prefill_tests::{config, gemma4_checkpoint, prefill_with, prompt_of};
use super::residency_caches::LayerCache;
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

pub(super) fn key_minmax(k_even: &[f32], k_odd: &[f32], _value: &[f32], even_odd_row: usize) -> Vec<f32> {
    let width = 2 * even_odd_row;
    let mut low = vec![f32::INFINITY; width];
    let mut high = vec![f32::NEG_INFINITY; width];
    for (even_row, odd_row) in k_even.chunks_exact(even_odd_row).zip(k_odd.chunks_exact(even_odd_row)) {
        for (slot, key) in even_row.iter().chain(odd_row).enumerate() {
            low[slot] = low[slot].min(*key);
            high[slot] = high[slot].max(*key);
        }
    }
    [low, high].concat()
}

fn prefill_full_layer(seal_horizon_rows: u32) -> (LayerCache, usize) {
    let bytes = gemma4_checkpoint();
    let parsed = proxima_gguf::pipe::parse_complete(&bytes).expect("parses the gemma4 fixture");
    let model = LoadedModel::load(&parsed, &bytes).expect("loads the gemma4 fixture").with_block_summarizer(key_minmax);
    let serving_config = ServingConfig {
        prompt_cache: PromptCacheConfig { block_tokens: 8, seal_horizon_rows, ..PromptCacheConfig::standard() },
        ..config(7)
    };
    let prefilled = prefill_with(&model, &prompt_of(40, '3'), &serving_config);
    assert_eq!(prefilled.state.len(), 40, "the prompt is cached in full");
    let rows = prefilled.state.len();
    let full = prefilled
        .state
        .layer_caches
        .into_iter()
        .find_map(|state| match state {
            LayerCacheState::Attention(cache) if cache.ring_geometry().is_none() => Some(cache),
            _ => None,
        })
        .expect("one full-attention layer");
    (full, rows)
}

#[test]
fn key_minmax_summaries_equal_an_independent_fold_of_the_decoded_rows() {
    let (full, rows) = prefill_full_layer(4);

    assert_eq!(full.k_even.len() % rows, 0, "the key rows divide evenly by the cached rows");
    let even_odd_row = full.k_even.len() / rows;
    assert_eq!(full.block_summaries.len(), 4, "sealed_end 32 is four blocks of eight rows");
    let mut some_slot_spans_a_range = false;
    for (block, record) in full.block_summaries.iter().enumerate() {
        assert_eq!(record.len(), 4 * even_odd_row, "a record is the minimum then the maximum of the concatenated key");
        let span = block * 8 * even_odd_row..(block + 1) * 8 * even_odd_row;
        let key_rows: Vec<Vec<f32>> = full.k_even[span.clone()]
            .chunks_exact(even_odd_row)
            .zip(full.k_odd[span].chunks_exact(even_odd_row))
            .map(|(even_row, odd_row)| even_row.iter().chain(odd_row).copied().collect())
            .collect();
        for slot in 0..2 * even_odd_row {
            let low = key_rows.iter().map(|row| row[slot]).fold(f32::INFINITY, f32::min);
            let high = key_rows.iter().map(|row| row[slot]).fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(record[slot].to_bits(), low.to_bits(), "block {block} slot {slot} minimum");
            assert_eq!(record[2 * even_odd_row + slot].to_bits(), high.to_bits(), "block {block} slot {slot} maximum");
            some_slot_spans_a_range |= low < high;
        }
    }
    assert!(some_slot_spans_a_range, "the rows are not all equal, so the fold is not degenerate");
}

#[test]
fn a_horizon_past_the_prompt_seals_and_summarizes_nothing() {
    let (full, _rows) = prefill_full_layer(1000);

    assert_eq!(full.sealed_end, 0, "no row is old enough to seal");
    assert!(full.block_summaries.is_empty(), "nothing sealed, nothing summarized");
}
