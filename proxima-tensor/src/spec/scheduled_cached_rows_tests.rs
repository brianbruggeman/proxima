//! `append_two_range_cached_attention` over a one-row key cache against the same
//! keys held in two planes: every score is the same products summed in the same
//! `e0,o0,e1,o1` order, so the layer outputs must agree bit for bit.

use super::*;
use crate::test_support::{Lcg, evaluate_named_online_f32};

const EMBEDDING: u32 = 8;
const KV_HEADS: u32 = 2;
const GROUP: u32 = 2;
const HEAD_DIM: u32 = 4;
const PAIRS: u32 = HEAD_DIM / 2;
const NEW_TOKENS: u32 = 2;
const CACHED_TOKENS: u32 = 3;
const LAYERS: u32 = 2;
const THETA: f32 = 10000.0;

struct Fixture {
    program: Vec<Op>,
    inputs: Vec<(String, Vec<f32>)>,
}

impl Fixture {
    fn leaf(&mut self, name: &str, shape: &[u32], values: Vec<f32>) -> NodeId {
        let extents = shape.iter().map(|&extent| Extent::Static(extent)).collect();
        self.inputs.push((String::from(name), values));
        input_leaf(&mut self.program, DType::FLOAT32, extents, name)
    }

    fn random_leaf(&mut self, name: &str, shape: &[u32]) -> NodeId {
        let seed = name.bytes().fold(0x6212_0001u64, |acc, byte| {
            acc.wrapping_mul(31).wrapping_add(u64::from(byte))
        });
        let mut lcg = Lcg(seed);
        let count = shape.iter().product::<u32>() as usize;
        let values = (0..count).map(|_| lcg.next_unit()).collect();
        self.leaf(name, shape, values)
    }
}

fn rope_tables(first_position: f32) -> (Vec<f32>, Vec<f32>) {
    let positions: Vec<f32> = (0..NEW_TOKENS)
        .map(|row| first_position + row as f32)
        .collect();
    let (cos, sin) = multi_axis_rope_tables(&positions, &[HEAD_DIM], &[THETA], DType::FLOAT32)
        .expect("rope tables build from real positions and theta");
    let decode = |bytes: &[u8]| -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect()
    };
    (decode(&cos), decode(&sin))
}

fn cached_keys(layer: u32) -> Vec<f32> {
    let mut lcg = Lcg(0x6212_1000 + u64::from(layer));
    (0..CACHED_TOKENS * KV_HEADS * 2 * PAIRS)
        .map(|_| lcg.next_unit())
        .collect()
}

fn half_of_rows(rows: &[f32], half: usize) -> Vec<f32> {
    let pairs = PAIRS as usize;
    rows.chunks(2 * pairs)
        .flat_map(|row| row[half * pairs..(half + 1) * pairs].to_vec())
        .collect()
}

fn key_leaves(fixture: &mut Fixture, layer: u32, layout: KeyCacheLayout) -> (NodeId, NodeId) {
    let keys = cached_keys(layer);
    let rows = [CACHED_TOKENS, KV_HEADS];
    match layout {
        KeyCacheLayout::Planes => {
            let shape = [rows[0], rows[1], PAIRS];
            let even = fixture.leaf(
                &alloc::format!("kv_cache.{layer}.k_even"),
                &shape,
                half_of_rows(&keys, 0),
            );
            let odd = fixture.leaf(
                &alloc::format!("kv_cache.{layer}.k_odd"),
                &shape,
                half_of_rows(&keys, 1),
            );
            (even, odd)
        }
        KeyCacheLayout::Rows => {
            let shape = [rows[0], rows[1], 2 * PAIRS];
            let both = fixture.leaf(&alloc::format!("kv_cache.{layer}.k"), &shape, keys);
            (both, both)
        }
    }
}

fn build(layout: KeyCacheLayout, pairing: RopePairing) -> (Fixture, NodeId, NodeId) {
    let mut fixture = Fixture {
        program: Vec::new(),
        inputs: Vec::new(),
    };
    let heads = KV_HEADS * GROUP;
    let (cos_values, sin_values) = rope_tables(CACHED_TOKENS as f32);
    let cos = fixture.leaf("rope_cos", &[NEW_TOKENS, PAIRS], cos_values);
    let sin = fixture.leaf("rope_sin", &[NEW_TOKENS, PAIRS], sin_values);
    let group_ones = fixture.leaf(
        "group_ones",
        &[KV_HEADS, GROUP],
        alloc::vec![1.0; (KV_HEADS * GROUP) as usize],
    );
    let local_future = (0..NEW_TOKENS * NEW_TOKENS)
        .map(|cell| f32::from(u8::from(cell % NEW_TOKENS > cell / NEW_TOKENS)))
        .collect();
    let is_future_local = fixture.leaf("is_future_local", &[NEW_TOKENS, NEW_TOKENS], local_future);
    let is_future_cached = fixture.leaf(
        "is_future_cached",
        &[NEW_TOKENS, CACHED_TOKENS],
        alloc::vec![0.0; (NEW_TOKENS * CACHED_TOKENS) as usize],
    );
    let inv_dim = count_constant(&mut fixture.program, EMBEDDING);
    let eps = scalar_constant(&mut fixture.program, 1e-6);
    let inv_sqrt = scalar_constant(&mut fixture.program, 1.0 / (HEAD_DIM as f32).sqrt());
    let inv_head = count_constant(&mut fixture.program, HEAD_DIM);
    let neg_infinity = scalar_constant(&mut fixture.program, f32::NEG_INFINITY);
    let mut x = fixture.random_leaf("x", &[NEW_TOKENS, EMBEDDING]);
    let mut first_layer_output = x;
    for layer in 0..LAYERS {
        let norm = fixture.random_leaf(
            &alloc::format!("blk.{layer}.attn_norm.weight"),
            &[EMBEDDING],
        );
        let wq = fixture.random_leaf(
            &alloc::format!("blk.{layer}.attn_q.weight"),
            &[EMBEDDING, heads, HEAD_DIM],
        );
        let wk = fixture.random_leaf(
            &alloc::format!("blk.{layer}.attn_k.weight"),
            &[EMBEDDING, KV_HEADS, HEAD_DIM],
        );
        let wv = fixture.random_leaf(
            &alloc::format!("blk.{layer}.attn_v.weight"),
            &[EMBEDDING, KV_HEADS, HEAD_DIM],
        );
        let wo = fixture.random_leaf(
            &alloc::format!("blk.{layer}.attn_output.weight"),
            &[KV_HEADS, GROUP, HEAD_DIM, EMBEDDING],
        );
        let v_cache = fixture.random_leaf(
            &alloc::format!("kv_cache.{layer}.v"),
            &[CACHED_TOKENS, KV_HEADS, HEAD_DIM],
        );
        let (k_even, k_odd) = key_leaves(&mut fixture, layer, layout);
        let (out, _) = append_two_range_cached_attention(
            &mut fixture.program,
            x,
            inv_dim,
            eps,
            inv_sqrt,
            inv_head,
            Some((cos, sin)),
            PAIRS,
            group_ones,
            is_future_local,
            neg_infinity,
            is_future_cached,
            neg_infinity,
            GROUP,
            None,
            norm,
            None,
            KeySource::Projected {
                wk,
                k_norm_weight: None,
            },
            wq,
            ValueSource::Projected(wv),
            wo,
            pairing,
            None,
            false,
            k_even,
            k_odd,
            v_cache,
            layout,
        )
        .expect("one cached attention layer builds");
        if layer == 0 {
            first_layer_output = out;
        }
        x = out;
    }
    (fixture, first_layer_output, x)
}

fn run(layout: KeyCacheLayout, pairing: RopePairing) -> (Vec<f32>, Vec<f32>) {
    let (fixture, first, last) = build(layout, pairing);
    let named: Vec<(&str, &[f32])> = fixture
        .inputs
        .iter()
        .map(|(name, values)| (name.as_str(), values.as_slice()))
        .collect();
    let evaluated = evaluate_named_online_f32(&fixture.program, &[], &named, &[first, last])
        .expect("the cached attention stack evaluates");
    let read = |node| evaluated.get(node).expect("output requested").0.to_vec();
    (read(first), read(last))
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[proxima::test]
#[case::split_half(RopePairing::SplitHalf { pairs: PAIRS })]
#[case::adjacent_pairs(RopePairing::Interleaved)]
async fn one_row_key_cache_gives_the_two_plane_outputs_bit_for_bit(#[case] pairing: RopePairing) {
    let (planes_first, planes_last) = run(KeyCacheLayout::Planes, pairing);
    let (rows_first, rows_last) = run(KeyCacheLayout::Rows, pairing);

    assert_eq!(planes_last.len(), (NEW_TOKENS * EMBEDDING) as usize);
    assert!(planes_last.iter().all(|value| value.is_finite()));
    assert_eq!(bits(&rows_first), bits(&planes_first), "layer 0 output");
    assert_eq!(bits(&rows_last), bits(&planes_last), "layer 1 output");
    std::println!(
        "{pairing:?} rows bits {:?} planes bits {:?}",
        &bits(&rows_last)[..4],
        &bits(&planes_last)[..4]
    );
}
