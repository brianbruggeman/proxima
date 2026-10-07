//! Metal-vs-CPU parity for the stacked routed feed-forward at decode and prefill
//! widths: `MoeProjectionStrategy::Stacked` lowered through the fused
//! `BoundOpKind::MoeTopK` (one threadgroup per token row, route and weight
//! stacks written by the same dispatch) and the gather operations that read
//! those stacks. The oracle for the routing is an independent top-k over each
//! token's scores; the oracle for the layer output is the CPU evaluation of the
//! per-route lowering of the same layer, which shares no node with the stacked
//! program past the inputs.

#![cfg(all(feature = "metal", feature = "moe-topk-fusion", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_lines)]

use proxima_tensor::spec::{
    Activation, ExpertGatingFunc, MoeFfnSpec, MoeProjectionStrategy, MoeRouter, append_moe_ffn,
    input_leaf, scalar_constant,
};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{DType, Extent, NodeId, NumericPolicy, Op, QuantizedBlock};

const EXPERT_COUNT: u32 = 32;
const EXPERT_USED_COUNT: u32 = 8;
const EMBEDDING: u32 = 64;
const FEED_FORWARD: u32 = 64;

struct Fixture {
    program: Vec<Op>,
    output: NodeId,
    selected: Vec<NodeId>,
    weights: Vec<NodeId>,
}

fn build(strategy: MoeProjectionStrategy, token_count: u32) -> Fixture {
    let mut program = Vec::new();
    let x = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(token_count), Extent::Static(EMBEDDING)],
        "x",
    );
    let logits = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(token_count), Extent::Static(EXPERT_COUNT)],
        "logits",
    );
    let stack = |program: &mut Vec<Op>, rows: u32, columns: u32, name: &str| {
        input_leaf(
            program,
            DType::Float32,
            vec![Extent::Static(EXPERT_COUNT), Extent::Static(rows), Extent::Static(columns)],
            name,
        )
    };
    let expert_w_gate = stack(&mut program, EMBEDDING, FEED_FORWARD, "expert_w_gate");
    let expert_w_up = stack(&mut program, EMBEDDING, FEED_FORWARD, "expert_w_up");
    let expert_w_down = stack(&mut program, FEED_FORWARD, EMBEDDING, "expert_w_down");
    let ones = scalar_constant(&mut program, 1.0);
    let moe_spec = MoeFfnSpec {
        router: MoeRouter::Logits(logits),
        expert_w_gate,
        expert_w_up,
        expert_w_down,
        expert_count: EXPERT_COUNT,
        expert_used_count: EXPERT_USED_COUNT,
        ones,
        gating: ExpertGatingFunc::Softmax,
        expert_bias: None,
        expert_scale: None,
        activation: Activation::Silu,
        strategy,
    };
    let (output, site) = append_moe_ffn(&mut program, 0, x, &moe_spec).expect("the routed block lowers");
    Fixture { program, output, selected: site.selected, weights: site.weights }
}

fn unit_values(seed: u64, count: usize, scale: f32) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit() * scale).collect()
}

fn reference_topk(scores: &[f32], top_k: usize) -> (Vec<f32>, Vec<f32>, f32) {
    let mut live = scores.to_vec();
    let (mut routes, mut weights) = (Vec::new(), Vec::new());
    let (mut first_max, mut total) = (0.0_f32, 0.0_f32);
    for round in 0..top_k {
        let mut best_index = 0;
        let mut best_value = f32::NEG_INFINITY;
        for (index, value) in live.iter().enumerate() {
            if *value >= best_value {
                best_value = *value;
                best_index = index;
            }
        }
        if round == 0 {
            first_max = best_value;
        }
        let weight = (best_value - first_max).exp();
        routes.push(best_index as f32);
        weights.push(weight);
        total += weight;
        for value in live.iter_mut().filter(|value| **value == best_value) {
            *value = f32::NEG_INFINITY;
        }
    }
    (routes, weights, total)
}

fn assert_stacked_matches(token_count: u32) {
    let tokens = token_count as usize;
    let per_route = build(MoeProjectionStrategy::PerRoute, token_count);
    let stacked = build(MoeProjectionStrategy::Stacked, token_count);
    let stack_len = (EXPERT_COUNT * EMBEDDING * FEED_FORWARD) as usize;
    let x = unit_values(1, tokens * EMBEDDING as usize, 1.0);
    let logits = unit_values(2, tokens * EXPERT_COUNT as usize, 10.0);
    let gate = unit_values(3, stack_len, 0.2);
    let up = unit_values(4, stack_len, 0.2);
    let down = unit_values(5, stack_len, 0.2);
    let named: Vec<(&str, QuantizedBlock)> = vec![
        ("x", QuantizedBlock::Float32(&x)),
        ("logits", QuantizedBlock::Float32(&logits)),
        ("expert_w_gate", QuantizedBlock::Float32(&gate)),
        ("expert_w_up", QuantizedBlock::Float32(&up)),
        ("expert_w_down", QuantizedBlock::Float32(&down)),
    ];

    let mut free_buffers = Vec::new();
    let mut validated = None;
    let cpu = proxima_tensor::cpu::evaluate_quantized_named_with_scratch(
        &per_route.program,
        &[],
        &named,
        &[per_route.output],
        &mut free_buffers,
        &mut validated,
    )
    .expect("cpu evaluates the per-route layer");
    let expected = cpu.root().to_vec();

    let mut outputs = vec![stacked.output];
    outputs.extend(stacked.selected.iter().copied());
    outputs.extend(stacked.weights.iter().copied());
    let plan = omega::plan_named(&stacked.program, &[], &named, &outputs, NumericPolicy::default())
        .expect("metal plans the stacked layer");
    let kernel_keys = plan.kernel_keys().expect("plan exposes its kernel keys");
    assert!(
        kernel_keys.iter().any(|key| key.contains("moe_topk_e32_k8_stacked")),
        "{token_count} tokens: the stacked top-k kernel was never selected: {kernel_keys:?}"
    );
    let metal = omega::execute_plan_named(&plan, &named).expect("metal runs the stacked layer");

    let actual = metal.get(stacked.output).map(|(values, _)| values.to_vec()).expect("layer output");
    assert_eq!(actual.len(), expected.len(), "{token_count} tokens: output length");
    let norm = expected.iter().map(|value| value * value).sum::<f32>().sqrt().max(1.0e-6);
    let worst = expected
        .iter()
        .zip(&actual)
        .map(|(want, got)| (want - got).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("tokens={token_count} worst_abs_over_norm={:e}", worst / norm);
    assert!(worst / norm <= 1.0e-4, "{token_count} tokens: layer output drifted by {:e} of the norm", worst / norm);

    for token in 0..tokens {
        let row = &logits[token * EXPERT_COUNT as usize..(token + 1) * EXPERT_COUNT as usize];
        let (routes, weights, total) = reference_topk(row, EXPERT_USED_COUNT as usize);
        for (round, &route_node) in stacked.selected.iter().enumerate() {
            let values = metal.get(route_node).map(|(values, _)| values.to_vec()).expect("route");
            assert_eq!(values[token], routes[round], "{token_count} tokens, token {token}, round {round}: route");
        }
        for (round, &weight_node) in stacked.weights.iter().take(EXPERT_USED_COUNT as usize).enumerate() {
            let values = metal.get(weight_node).map(|(values, _)| values.to_vec()).expect("weight");
            assert!((values[token] - weights[round]).abs() <= 1.0e-6, "{token_count} tokens, token {token}, round {round}: weight");
        }
        let total_node = *stacked.weights.last().expect("weight_total closes the weights");
        let values = metal.get(total_node).map(|(values, _)| values.to_vec()).expect("weight_total");
        assert!((values[token] - total).abs() <= 1.0e-5, "{token_count} tokens, token {token}: weight_total");
    }
}

#[test]
fn a_decode_step_of_one_token_matches_the_per_route_layer() {
    assert_stacked_matches(1);
}

#[test]
fn a_short_prefill_matches_the_per_route_layer() {
    assert_stacked_matches(7);
}

#[test]
fn a_prefill_chunk_wider_than_the_tile_minimum_matches_the_per_route_layer() {
    assert_stacked_matches(200);
}
