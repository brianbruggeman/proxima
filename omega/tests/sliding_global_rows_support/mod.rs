//! The sliding-and-global two-layer two-range cached program the row-tiled
//! attention gates share: a sliding layer (head_dim 256, window 512) and a
//! global layer (head_dim 512, unwindowed), eight query heads on one kv head,
//! at the dimensions the row-tiled kernel is sized for. Every named input is generated from the program's own
//! inferred shapes, RoPE tables are real rotary angles at the absolute
//! positions of the new rows, and the cache holds `cached_len` live rows
//! followed by the zero padding of its capacity bucket.
//!
//! Each `omega/tests/*.rs` file is its own crate that `mod`s this directory,
//! and no gate calls every item here.

#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::spec::{
    AttentionScoreScale, EmbeddingScale, KeySourceKind, LayerAttentionConfig, LayerFfnConfig,
    LayerKind, LayerSchedule, RopePairing, RopeTableSel, SharedExpertWidth, ValueSourceKind,
    scheduled_two_range_cached_forward_program_with_experts,
};
use proxima_numeric::{NumericFormat, PackedRows};
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

pub struct Fixture {
    pub program: Vec<Op>,
    pub logits: NodeId,
    pub symbols: Vec<u64>,
    pub named: Vec<(String, PackedRows)>,
}

/// The attention geometry of the two layers: query heads, kv heads, the head
/// dim of the sliding layer and of the global layer, and the sliding window.
#[derive(Clone, Copy)]
pub struct Geometry {
    pub query_heads: u32,
    pub kv_heads: u32,
    pub sliding_head_dim: u32,
    pub global_head_dim: u32,
    pub window: u32,
}

impl Geometry {
    /// Eight query heads on one kv head, head dims 256 and 512,
    /// a 512-key window.
    pub const ONE_KV_HEAD: Self = Self {
        query_heads: QUERY_HEADS,
        kv_heads: KV_HEADS,
        sliding_head_dim: SLIDING_HEAD_DIM,
        global_head_dim: GLOBAL_HEAD_DIM,
        window: SLIDING_WINDOW,
    };

    /// Sixteen query heads on eight kv heads, two query
    /// groups of head dim 64 in both layers.
    pub const TWO_QUERY_GROUPS: Self = Self {
        query_heads: 16,
        kv_heads: 8,
        sliding_head_dim: 64,
        global_head_dim: 64,
        window: SLIDING_WINDOW,
    };

    #[must_use]
    pub fn with_window(self, window: u32) -> Self {
        Self { window, ..self }
    }
}

fn attention(
    head_dim: u32,
    kv_heads: u32,
    window: Option<u32>,
    value: ValueSourceKind,
    full: bool,
) -> LayerAttentionConfig {
    LayerAttentionConfig {
        head_dim,
        kv_heads,
        mask_window: window,
        value_source_kind: value,
        key_source_kind: KeySourceKind::ProjectedK,
        rope_table: if full {
            RopeTableSel {
                cos_name: "rope_cos".into(),
                sin_name: "rope_sin".into(),
            }
        } else {
            RopeTableSel {
                cos_name: "rope_cos_swa".into(),
                sin_name: "rope_sin_swa".into(),
            }
        },
        rope_pairing: RopePairing::SplitHalf {
            pairs: head_dim / 2,
        },
        score_scale: AttentionScoreScale::Unscaled,
        value_norm: true,
        qk_norm: true,
        attention_sinks: false,
        rope: true,
    }
}

fn schedule(geometry: Geometry) -> Vec<LayerSchedule> {
    vec![
        LayerSchedule {
            kind: LayerKind::Attention,
            attention: attention(
                geometry.sliding_head_dim,
                geometry.kv_heads,
                Some(geometry.window),
                ValueSourceKind::ProjectedV,
                false,
            ),
            ffn: LayerFfnConfig::exclusive(),
        },
        LayerSchedule {
            kind: LayerKind::Attention,
            attention: attention(
                geometry.global_head_dim,
                geometry.kv_heads,
                None,
                ValueSourceKind::SharedWithKey,
                true,
            ),
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

fn random_rows(seed: u64, count: usize) -> PackedRows {
    let mut lcg = Lcg(seed);
    let values: Vec<_> = (0..count).map(|_| lcg.next_unit()).collect();
    PackedRows::from_values(&NumericFormat::host_element(), &values)
        .expect("random values encode in the host element")
}

/// RoPE angles for `rows` new positions starting at `first_position`:
/// `[rows, pairs]` cosines and sines of `position * base^(-pair / pairs)`,
/// the base read from `rope.toml` under `base_key`.
fn rope_table(
    first_position: usize,
    rows: usize,
    pairs: usize,
    base_key: &str,
    sine: bool,
) -> PackedRows {
    let host = NumericFormat::host_element();
    let decode = |values: &[_]| {
        let mut decoded = Vec::new();
        PackedRows::from_values(&host, values)
            .expect("the host element encodes")
            .decode_all(&mut decoded)
            .expect("the host element decodes");
        decoded
    };
    let bases: toml::Table = toml::from_str(include_str!("rope.toml")).expect("rope.toml parses");
    let configured = bases[base_key]
        .as_float()
        .expect("each rope base is a number");
    let base = decode(&[configured as _])[0];
    let width = decode(&[pairs as _])[0];
    let positions = decode(
        &(0..rows)
            .map(|row| (first_position + row) as _)
            .collect::<Vec<_>>(),
    );
    let indices = decode(&(0..pairs).map(|pair| pair as _).collect::<Vec<_>>());
    let mut table = Vec::with_capacity(rows * pairs);
    for position in &positions {
        for pair in &indices {
            let frequency = base.powf(-pair / width);
            let angle = position * frequency;
            table.push(if sine { angle.sin() } else { angle.cos() });
        }
    }
    PackedRows::from_values(&host, &table).expect("the rope table encodes in the host element")
}

fn named_block(
    name: &str,
    count: usize,
    last_extent: usize,
    rows: usize,
    cached_len: usize,
    seed: u64,
) -> PackedRows {
    let host = NumericFormat::host_element();
    let pack = |values: &[_]| {
        PackedRows::from_values(&host, values).expect("the values encode in the host element")
    };
    let drawn = |len: usize| {
        let mut values = Vec::new();
        random_rows(seed, len)
            .decode_all(&mut values)
            .expect("the host element decodes");
        values
    };
    match name {
        "ids" => pack(
            &(0..count)
                .map(|index| ((index * 7 + 3) % VOCAB as usize) as _)
                .collect::<Vec<_>>(),
        ),
        "eps" => pack(&vec![1e-6; count]),
        "cached_len" => pack(&vec![cached_len as _; count]),
        "rope_cos_swa" => rope_table(cached_len, rows, last_extent, "sliding_base", false),
        "rope_sin_swa" => rope_table(cached_len, rows, last_extent, "sliding_base", true),
        "rope_cos" => rope_table(cached_len, rows, last_extent, "global_base", false),
        "rope_sin" => rope_table(cached_len, rows, last_extent, "global_base", true),
        _ if name.starts_with("kv_cache.") => {
            let row_width = last_extent;
            let live = cached_len * row_width;
            let mut data = drawn(live.min(count));
            data.resize(count, 0.0);
            pack(&data)
        }
        _ if name.ends_with("norm.weight") => pack(
            &drawn(count)
                .into_iter()
                .map(|value| 1.0 + 0.5 * value)
                .collect::<Vec<_>>(),
        ),
        _ if name.ends_with("layer_output_scale.weight") => pack(&vec![1.0; count]),
        _ => pack(
            &drawn(count)
                .into_iter()
                .map(|value| 0.1 * value)
                .collect::<Vec<_>>(),
        ),
    }
}

/// The program at `rows` new rows over `cached_len` live cached rows, bound
/// at the capacity bucket `bucket_for(cached_len)`.
#[must_use]
pub fn fixture(rows: usize, cached_len: usize) -> Fixture {
    fixture_with(rows, cached_len, Geometry::ONE_KV_HEAD)
}

/// [`fixture`] at another attention geometry: a window under `rows` is one the
/// new range itself crosses, the prefill shape.
#[must_use]
pub fn fixture_with(rows: usize, cached_len: usize, geometry: Geometry) -> Fixture {
    let (program, logits, _cache_roots, _moe_sites, _head_repeats) =
        scheduled_two_range_cached_forward_program_with_experts(
            VOCAB,
            EMBEDDING,
            FEED_FORWARD,
            FEED_FORWARD,
            geometry.query_heads,
            BLOCK_COUNT,
            0,
            0,
            BLOCK_COUNT,
            &schedule(geometry),
            Some(EmbeddingScale::Sqrt),
            None,
            false,
            None,
            false,
            SharedExpertWidth::NONE,
            None,
        )
        .expect("the sliding-and-global two-range program lowers");
    let symbols = vec![rows as u64, bucket_for(cached_len) as u64];
    let shapes = infer(&program, &symbols).expect("the sliding-and-global program infers");

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
