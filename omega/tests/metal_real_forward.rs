//! The gate that ROWS 69-78 never had: the REAL forward graph, end to end,
//! CPU against device.
//!
//! Every other GPU number in this workspace comes from a synthetic matvec.
//! This binds `mistral_cached_forward_program` — the same builder
//! `proxima-model-interop` uses for a real token — and runs it through both
//! evaluators on identical named blocks.
//!
//! The architecture is scaled down (2 layers, 64-wide) so it runs in a test,
//! but the OP SET and the graph shape are the production ones: embedding
//! gather, RMSNorm, RoPE, grouped-query attention with a KV cache, SwiGLU,
//! and the output projection. That is the coverage that matters here; the
//! full-size numbers are what the probe next to this measures.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::spec::mistral_cached_forward_program;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::cpu::evaluate_quantized_named_with_scratch;
use proxima_tensor::{NodeId, Op, QuantizedBlock, infer};

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

#[test]
fn metal_runs_the_real_forward_graph_and_agrees_with_the_cpu() {
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

    // one decode step, no cached history
    let symbols = [1u64, 0u64];
    let shapes = infer(&program, &symbols).expect("the real forward infers");

    // every block input gets deterministic f32 data sized from the graph
    // itself, so this needs no checkpoint on disk.
    let mut owned: Vec<(String, Vec<f32>)> = Vec::new();
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
        owned.push((name, data));
    }
    let named: Vec<(&str, QuantizedBlock<'_>)> = owned
        .iter()
        .map(|(name, data)| (name.as_str(), QuantizedBlock::Float32(data.as_slice())))
        .collect();

    let mut free_buffers: Vec<Vec<f32>> = Vec::new();
    let mut validated = None;
    let cpu = evaluate_quantized_named_with_scratch(
        &program,
        &symbols,
        &named,
        &roots,
        &mut free_buffers,
        &mut validated,
    )
    .expect("cpu runs the real forward");

    let plan = omega::plan_named(&program, &symbols, &named, &roots)
        .expect("metal plans the real forward");
    let metal = omega::execute_plan_named(&plan, &named)
        .expect("metal runs the real forward on a real device");

    let expected = cpu.root();
    let actual = metal.root();
    assert_eq!(
        actual.len(),
        VOCAB as usize,
        "degenerate gate: logits must be one row of the vocabulary"
    );
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(got.is_finite(), "metal produced a non-finite logit: {got}");
        max_diff = max_diff.max((got - want).abs());
    }
    let max_magnitude = expected.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude.max(f32::MIN_POSITIVE);
    eprintln!(
        "real forward, {LAYERS} layers: max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}"
    );
    assert!(
        relative < 1e-4,
        "metal disagrees with the cpu on the real forward: relative={relative} max_diff={max_diff}"
    );
}
