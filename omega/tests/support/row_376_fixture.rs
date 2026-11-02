use proxima_tensor::spec::{DuplicateHeadPosition, gqa_single_range_cached_forward_program};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{NodeId, Op, block_node_ids, infer};

pub const QUERY_HEADS: u32 = 32;
pub const KV_HEADS: u32 = 8;
pub const HEAD_DIM: u32 = 128;
pub const EMBEDDING: u32 = QUERY_HEADS * HEAD_DIM;
pub const FEED_FORWARD: u32 = 256;
pub const VOCAB: u32 = 64;
pub const LAYERS: u32 = 32;

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

pub type Row376Fixture = (Vec<Op>, Vec<u64>, Vec<NodeId>, Vec<(String, Vec<f32>)>);

/// Builds the real-shape openchat single-range forward at one context
/// length. `context_length` is `cached_len + 1` (one new query row, the
/// decode shape) -- this row never exercises prefill (`new_count > 1`).
pub fn build_fixture(context_length: u64) -> Row376Fixture {
    let cached_len = context_length - 1;
    let new_count = 1u64;

    let (program, logits_root, cache_roots, _) = gqa_single_range_cached_forward_program(
        VOCAB,
        EMBEDDING,
        FEED_FORWARD,
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
        LAYERS,
        false,
        DuplicateHeadPosition::None,
        false,
    )
    .expect("the real-shape single-range forward program builds");

    let mut roots = vec![logits_root];
    for (even, odd, value) in &cache_roots {
        roots.push(*even);
        roots.push(*odd);
        roots.push(*value);
    }

    let symbols = vec![new_count, context_length];
    let shapes = infer(&program, &symbols).expect("the real-shape single-range forward infers");

    let mut named: Vec<(String, Vec<f32>)> = Vec::new();
    for node in block_node_ids(&program) {
        let Op::Input { name, .. } = &program[node.0 as usize] else {
            unreachable!("block_node_ids only ever returns Op::Input nodes")
        };
        let name = name
            .clone()
            .expect("every block input in this program is named");
        let count: usize = shapes
            .of(node)
            .iter()
            .map(|extent| *extent as usize)
            .product();
        let data = if name == "ids" {
            vec![3.0f32; count]
        } else if name == "eps" {
            vec![1e-5f32; count]
        } else if name == "cached_len" {
            vec![cached_len as f32]
        } else {
            random_vec(node.0 as u64 + 1, count)
        };
        named.push((name, data));
    }

    (program, symbols, roots, named)
}
