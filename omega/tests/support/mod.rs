//! The real cached-forward graph fixture, shared by `metal_real_forward.rs`
//! (CPU vs the raw Metal driver) and `backend_parity.rs` (CPU vs Metal
//! through `omega::backend`'s wrapper) — lifted out of the former so the
//! SAME program, roots and named block data feed both gates rather than two
//! copies that can drift on which named block gets which random seed.

// fixture construction is hand-built to succeed; an expect failure here IS
// the fixture being broken, same convention as every other `omega/tests/*.rs` file.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::spec::mistral_cached_forward_program;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{NodeId, Op, QuantizedBlock, infer};

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// [`real_forward_fixture`]'s return shape: the program, its decode-step
/// symbols, every root a caller should request as output, and deterministic
/// `(name, data)` pairs for every named block input.
pub type RealForwardFixture = (Vec<Op>, Vec<u64>, Vec<NodeId>, Vec<(String, Vec<f32>)>);

/// A small (2-layer, 64-wide) instance of the real cached-forward graph:
/// production op set and graph shape, scaled down only so it runs in a
/// test. Returns the program, the decode-step symbols (one step, no cached
/// history), every root a caller should request as output (the logits, plus
/// every layer's KV-cache write), and deterministic f32 data for every named
/// block input -- sized from the graph's own inferred shapes, so this needs
/// no checkpoint on disk.
pub fn real_forward_fixture() -> RealForwardFixture {
    const VOCAB: u32 = 64;
    const EMBEDDING: u32 = 64;
    const FEED_FORWARD: u32 = 128;
    const QUERY_HEADS: u32 = 4;
    const KV_HEADS: u32 = 2;
    const HEAD_DIM: u32 = 16;
    const LAYERS: u32 = 2;

    let (program, logits_root, cache_roots) = mistral_cached_forward_program(
        VOCAB,
        EMBEDDING,
        FEED_FORWARD,
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
        LAYERS,
    )
    .expect("the real forward program builds");

    let mut roots = vec![logits_root];
    for (even, odd, value) in &cache_roots {
        roots.push(*even);
        roots.push(*odd);
        roots.push(*value);
    }

    let symbols = vec![1u64, 0u64];
    let shapes = infer(&program, &symbols).expect("the real forward infers");

    let mut named: Vec<(String, Vec<f32>)> = Vec::new();
    for (position, op) in program.iter().enumerate() {
        let Op::Input { name, .. } = op else { continue };
        let node = NodeId(position as u32);
        let count: usize = shapes.of(node).iter().map(|extent| *extent as usize).product();
        let name = name.clone().expect("every block input in this program is named");
        // an empty block is legitimate here: a KV-cache input is genuinely
        // zero-length at `cached_len == 0`, and padding it to one element is
        // an invented value the shape check correctly rejects.
        let data = if name == "ids" {
            // a token id, not a weight: must be an in-range integer
            vec![3.0f32; count]
        } else if name == "eps" {
            vec![1e-5f32; count]
        } else {
            random_vec(position as u64 + 1, count)
        };
        named.push((name, data));
    }

    (program, symbols, roots, named)
}

/// Borrows [`real_forward_fixture`]'s owned `(name, data)` pairs into the
/// `&[(&str, QuantizedBlock<'_>)]` shape both evaluators bind against.
pub fn as_named_blocks(owned: &[(String, Vec<f32>)]) -> Vec<(&str, QuantizedBlock<'_>)> {
    owned
        .iter()
        .map(|(name, data)| (name.as_str(), QuantizedBlock::Float32(data.as_slice())))
        .collect()
}
