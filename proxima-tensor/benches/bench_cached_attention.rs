//! Cached-attention physical stream against the materialized online-softmax
//! reference it replaces. The reference keeps score, weight, and output
//! intermediates caller-owned so the timed comparison isolates arithmetic and
//! memory traffic rather than allocator setup.
//!
//! Re-prove with:
//! `CARGO_TARGET_DIR=<scratch> cargo bench -p proxima-tensor --bench bench_cached_attention`

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use proxima_tensor::physical::{AttentionExtents, CausalBand, stream_cached_attention_split_gqa};

const KV_HEADS: usize = 8;
const QUERY_GROUPS: usize = 4;
const HEAD_DIM: usize = 64;
const QUERY_ROWS: usize = 1;
const SCALE: f32 = 0.125;

fn data(length: usize, width: usize, seed: f32) -> Vec<f32> {
    (0..length * width)
        .map(|index| (index as f32 + seed).sin() * 0.5)
        .collect()
}

struct MaterializedScratch<'buffer> {
    scores: &'buffer mut [f32],
    weights: &'buffer mut [f32],
}

fn materialized_attention(
    queries: [&[f32]; 2],
    keys: [[&[f32]; 2]; 2],
    values: [&[f32]; 2],
    output: &mut [f32],
    cached_key_rows: usize,
    new_key_rows: usize,
    scratch: MaterializedScratch<'_>,
) {
    let MaterializedScratch { scores, weights } = scratch;
    for query_row in 0..QUERY_ROWS {
        for kv_head in 0..KV_HEADS {
            for query_group in 0..QUERY_GROUPS {
                let query_head = kv_head * QUERY_GROUPS + query_group;
                let query_start =
                    (query_row * KV_HEADS * QUERY_GROUPS + query_head) * (HEAD_DIM / 2);
                let output_start = (query_row * KV_HEADS * QUERY_GROUPS + query_head) * HEAD_DIM;
                let score_start = (query_row * KV_HEADS * QUERY_GROUPS + query_head)
                    * (cached_key_rows + new_key_rows);
                let mut maximum = f32::NEG_INFINITY;
                for key in 0..cached_key_rows + new_key_rows {
                    let range = usize::from(key >= cached_key_rows);
                    let range_row = if range == 0 {
                        key
                    } else {
                        key - cached_key_rows
                    };
                    if range != 0 && range_row > query_row {
                        continue;
                    }
                    let key_start = (range_row * KV_HEADS + kv_head) * (HEAD_DIM / 2);
                    let mut score = 0.0;
                    for pair in 0..HEAD_DIM / 2 {
                        score += queries[0][query_start + pair] * keys[range][0][key_start + pair]
                            + queries[1][query_start + pair] * keys[range][1][key_start + pair];
                    }
                    let slot = score_start + key;
                    scores[slot] = score * SCALE;
                    maximum = maximum.max(scores[slot]);
                }
                let mut sum = 0.0;
                for key in 0..cached_key_rows + new_key_rows {
                    let slot = score_start + key;
                    weights[slot] = if scores[slot].is_finite() {
                        (scores[slot] - maximum).exp()
                    } else {
                        0.0
                    };
                    sum += weights[slot];
                }
                for dimension in 0..HEAD_DIM {
                    let mut value = 0.0;
                    for key in 0..cached_key_rows + new_key_rows {
                        let range = usize::from(key >= cached_key_rows);
                        let range_row = if range == 0 {
                            key
                        } else {
                            key - cached_key_rows
                        };
                        let value_start = (range_row * KV_HEADS + kv_head) * HEAD_DIM;
                        value +=
                            weights[score_start + key] * values[range][value_start + dimension];
                    }
                    output[output_start + dimension] = if sum == 0.0 { 0.0 } else { value / sum };
                }
            }
        }
    }
}

fn bench_cached_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("cached_attention_cpu");
    for cached_key_rows in [0usize, 1, 16, 128] {
        let query_even = data(QUERY_ROWS * KV_HEADS * QUERY_GROUPS, HEAD_DIM / 2, 1.0);
        let query_odd = data(QUERY_ROWS * KV_HEADS * QUERY_GROUPS, HEAD_DIM / 2, 2.0);
        let cached_key_even = data(cached_key_rows * KV_HEADS, HEAD_DIM / 2, 3.0);
        let cached_key_odd = data(cached_key_rows * KV_HEADS, HEAD_DIM / 2, 4.0);
        let new_key_even = data(KV_HEADS, HEAD_DIM / 2, 5.0);
        let new_key_odd = data(KV_HEADS, HEAD_DIM / 2, 6.0);
        let cached_value = data(cached_key_rows * KV_HEADS, HEAD_DIM, 7.0);
        let new_value = data(KV_HEADS, HEAD_DIM, 8.0);
        let mut streamed_output = vec![0.0; QUERY_ROWS * KV_HEADS * QUERY_GROUPS * HEAD_DIM];
        let mut materialized_output = streamed_output.clone();
        let mut scores = vec![0.0; QUERY_ROWS * KV_HEADS * QUERY_GROUPS * (cached_key_rows + 1)];
        let mut weights = scores.clone();
        let extents = AttentionExtents {
            query_rows: QUERY_ROWS as u64,
            cached_key_rows: cached_key_rows as u64,
            new_key_rows: 1,
            kv_heads: KV_HEADS as u64,
            query_groups: QUERY_GROUPS as u64,
            head_dim: HEAD_DIM as u64,
        };
        assert!(stream_cached_attention_split_gqa(
            [&query_even, &query_odd],
            [
                [&cached_key_even, &cached_key_odd],
                [&new_key_even, &new_key_odd],
            ],
            [&cached_value, &new_value],
            &mut streamed_output,
            extents.clone(),
            SCALE,
            [
                CausalBand {
                    lower_inclusive: i64::MIN,
                    upper_inclusive: i64::MAX,
                },
                CausalBand {
                    lower_inclusive: i64::MIN,
                    upper_inclusive: 0,
                },
            ],
        ));
        materialized_attention(
            [&query_even, &query_odd],
            [
                [&cached_key_even, &cached_key_odd],
                [&new_key_even, &new_key_odd],
            ],
            [&cached_value, &new_value],
            &mut materialized_output,
            cached_key_rows,
            1,
            MaterializedScratch {
                scores: &mut scores,
                weights: &mut weights,
            },
        );
        let maximum_difference = streamed_output
            .iter()
            .zip(&materialized_output)
            .map(|(streamed, materialized)| (streamed - materialized).abs())
            .fold(0.0, f32::max);
        assert!(
            maximum_difference < 1e-5,
            "stream/materialized max difference={maximum_difference}"
        );
        group.bench_with_input(
            BenchmarkId::new("stream", cached_key_rows),
            &cached_key_rows,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(stream_cached_attention_split_gqa(
                        [&query_even, &query_odd],
                        [
                            [&cached_key_even, &cached_key_odd],
                            [&new_key_even, &new_key_odd],
                        ],
                        [&cached_value, &new_value],
                        &mut streamed_output,
                        extents.clone(),
                        SCALE,
                        [
                            CausalBand {
                                lower_inclusive: i64::MIN,
                                upper_inclusive: i64::MAX,
                            },
                            CausalBand {
                                lower_inclusive: i64::MIN,
                                upper_inclusive: 0,
                            },
                        ],
                    ));
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("materialized", cached_key_rows),
            &cached_key_rows,
            |bencher, _| {
                bencher.iter(|| {
                    materialized_attention(
                        [&query_even, &query_odd],
                        [
                            [&cached_key_even, &cached_key_odd],
                            [&new_key_even, &new_key_odd],
                        ],
                        [&cached_value, &new_value],
                        &mut materialized_output,
                        cached_key_rows,
                        1,
                        MaterializedScratch {
                            scores: &mut scores,
                            weights: &mut weights,
                        },
                    );
                    black_box(&materialized_output);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_cached_attention);
criterion_main!(benches);
