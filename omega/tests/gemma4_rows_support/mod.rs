//! The gemma4-shaped two-layer two-range cached program the row-tiled
//! attention gates share: a sliding layer (head_dim 256, window 512) and a
//! global layer (head_dim 512, unwindowed), eight query heads on one kv head
//! each, the attention geometry of gemma4-E2B at the dimensions the row-tiled
//! kernel is sized for. Every named input is generated from the program's own
//! inferred shapes, RoPE tables are real rotary angles at the absolute
//! positions of the new rows, and the cache holds `cached_len` live rows
//! followed by the zero padding of its capacity bucket.
//!
//! Each `omega/tests/*.rs` file is its own crate that `mod`s this directory,
//! and no gate calls every item here.

#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::spec::{
    AttentionScoreScale, EmbeddingScale, KeySourceKind, LayerAttentionConfig, LayerFfnConfig,
    LayerKind, LayerSchedule, RopePairing, RopeTableSel, ValueSourceKind,
    lfm2_two_range_cached_forward_program_with_experts,
};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{NodeId, Op, infer};

pub const VOCAB: u32 = 64;
pub const EMBEDDING: u32 = 256;
pub const FEED_FORWARD: u32 = 128;
pub const QUERY_HEADS: u32 = 8;
pub const BLOCK_COUNT: u32 = 2;
pub const KV_HEADS: u32 = 1;
pub const SLIDING_HEAD_DIM: u32 = 256;
pub const GLOBAL_HEAD_DIM: u32 = 512;
pub const SLIDING_WINDOW: u32 = 512;
pub const KV_BUCKET_TOKENS: usize = 32;
pub const SLIDING_ROPE_BASE: f32 = 10_000.0;
pub const GLOBAL_ROPE_BASE: f32 = 1_000_000.0;

pub struct Fixture {
    pub program: Vec<Op>,
    pub logits: NodeId,
    pub symbols: Vec<u64>,
    pub named: Vec<(String, Vec<f32>)>,
}

fn attention(
    head_dim: u32,
    window: Option<u32>,
    value: ValueSourceKind,
    full: bool,
) -> LayerAttentionConfig {
    LayerAttentionConfig {
        head_dim,
        kv_heads: KV_HEADS,
        mask_window: window,
        value_source_kind: value,
        key_source_kind: KeySourceKind::ProjectedK,
        rope_table: if full {
            RopeTableSel {
                cos_name: "rope_cos",
                sin_name: "rope_sin",
            }
        } else {
            RopeTableSel {
                cos_name: "rope_cos_swa",
                sin_name: "rope_sin_swa",
            }
        },
        rope_pairing: RopePairing::SplitHalf {
            pairs: head_dim / 2,
        },
        score_scale: AttentionScoreScale::Unscaled,
        value_norm: true,
    }
}

fn schedule() -> Vec<LayerSchedule> {
    vec![
        LayerSchedule {
            kind: LayerKind::Attention,
            attention: attention(
                SLIDING_HEAD_DIM,
                Some(SLIDING_WINDOW),
                ValueSourceKind::ProjectedV,
                false,
            ),
            ffn: LayerFfnConfig::exclusive(),
        },
        LayerSchedule {
            kind: LayerKind::Attention,
            attention: attention(GLOBAL_HEAD_DIM, None, ValueSourceKind::SharedWithKey, true),
            ffn: LayerFfnConfig::exclusive(),
        },
    ]
}

/// The capacity bucket a cache of `cached_len` live rows binds at: the next
/// multiple of [`KV_BUCKET_TOKENS`], as the serving loop rounds it.
#[must_use]
pub fn bucket_for(cached_len: usize) -> usize {
    cached_len.div_ceil(KV_BUCKET_TOKENS) * KV_BUCKET_TOKENS
}

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// RoPE angles for `rows` new positions starting at `first_position`:
/// `[rows, pairs]` cosines and sines of `position * base^(-pair / pairs)`.
fn rope_table(first_position: usize, rows: usize, pairs: usize, base: f32, sine: bool) -> Vec<f32> {
    let mut table = Vec::with_capacity(rows * pairs);
    for row in 0..rows {
        let position = (first_position + row) as f32;
        for pair in 0..pairs {
            let frequency = base.powf(-(pair as f32) / pairs as f32);
            let angle = position * frequency;
            table.push(if sine { angle.sin() } else { angle.cos() });
        }
    }
    table
}

fn named_block(
    name: &str,
    count: usize,
    last_extent: usize,
    rows: usize,
    cached_len: usize,
    seed: u64,
) -> Vec<f32> {
    match name {
        "ids" => (0..count)
            .map(|index| ((index * 7 + 3) % VOCAB as usize) as f32)
            .collect(),
        "eps" => vec![1e-6; count],
        "cached_len" => vec![cached_len as f32; count],
        "rope_cos_swa" => rope_table(cached_len, rows, last_extent, SLIDING_ROPE_BASE, false),
        "rope_sin_swa" => rope_table(cached_len, rows, last_extent, SLIDING_ROPE_BASE, true),
        "rope_cos" => rope_table(cached_len, rows, last_extent, GLOBAL_ROPE_BASE, false),
        "rope_sin" => rope_table(cached_len, rows, last_extent, GLOBAL_ROPE_BASE, true),
        _ if name.starts_with("kv_cache.") => {
            let row_width = last_extent;
            let live = cached_len * row_width;
            let mut data = random_vec(seed, live.min(count));
            data.resize(count, 0.0);
            data
        }
        _ if name.ends_with("norm.weight") => random_vec(seed, count)
            .into_iter()
            .map(|value| 1.0 + 0.5 * value)
            .collect(),
        _ if name.ends_with("layer_output_scale.weight") => vec![1.0; count],
        _ => random_vec(seed, count)
            .into_iter()
            .map(|value| 0.1 * value)
            .collect(),
    }
}

/// The program at `rows` new rows over `cached_len` live cached rows, bound
/// at the capacity bucket `bucket_for(cached_len)`.
#[must_use]
pub fn fixture(rows: usize, cached_len: usize) -> Fixture {
    let (program, logits, _cache_roots, _moe_sites, _head_repeats) =
        lfm2_two_range_cached_forward_program_with_experts(
            VOCAB,
            EMBEDDING,
            FEED_FORWARD,
            FEED_FORWARD,
            QUERY_HEADS,
            BLOCK_COUNT,
            0,
            0,
            BLOCK_COUNT,
            &schedule(),
            Some(EmbeddingScale::Sqrt),
            None,
            false,
            None,
            false,
        )
        .expect("the gemma4-shaped two-range program lowers");
    let symbols = vec![rows as u64, bucket_for(cached_len) as u64];
    let shapes = infer(&program, &symbols).expect("the gemma4-shaped program infers");

    let mut named = Vec::new();
    for (position, op) in program.iter().enumerate() {
        let Op::Input { name, .. } = op else { continue };
        let name = name
            .clone()
            .expect("every block input in this program is named");
        let extents = shapes.of(NodeId(position as u32));
        let count: usize = extents.iter().map(|extent| *extent as usize).product();
        let last_extent = extents.last().map_or(1, |extent| *extent as usize);
        let width = if name.starts_with("kv_cache.") {
            extents
                .iter()
                .skip(1)
                .map(|extent| *extent as usize)
                .product()
        } else {
            last_extent
        };
        let data = named_block(&name, count, width, rows, cached_len, position as u64 + 1);
        named.push((name, data));
    }
    Fixture {
        program,
        logits,
        symbols,
        named,
    }
}
