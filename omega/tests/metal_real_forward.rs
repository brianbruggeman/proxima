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

use proxima_tensor::cpu::evaluate_quantized_named_with_scratch;

mod support;
use support::{as_named_blocks, real_forward_fixture};

#[test]
fn metal_runs_the_real_forward_graph_and_agrees_with_the_cpu() {
    const VOCAB: usize = 64;

    let (program, symbols, roots, owned) = real_forward_fixture();
    let named = as_named_blocks(&owned);

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
        VOCAB,
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
        "real forward: max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}"
    );
    assert!(
        relative < 1e-4,
        "metal disagrees with the cpu on the real forward: relative={relative} max_diff={max_diff}"
    );
}
