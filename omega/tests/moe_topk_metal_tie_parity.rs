//! ROW 569 (`docs/discipline.md`): CPU-vs-Metal parity for
//! `BoundOpKind::MoeTopK` at qwen35moe's real routing shape (256 experts,
//! `expert_used_count = 8`, `ExpertGatingFunc::Softmax`, no `expert_bias`)
//! over 50 random logit vectors plus the exact-tie fixtures
//! `proxima_tensor::bind::tests::moe_routing_census`'s own CPU parity test
//! already proved -- routes exact, weights within `1e-6`, `weight_total`
//! within `1e-5` (looser: it accumulates `top_k` per-round weights, so its
//! own floating error budget is `top_k` times a single weight's).

#![cfg(all(feature = "metal", feature = "moe-topk-fusion", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_lines)]

use proxima_tensor::spec::{
    Activation, ExpertGatingFunc, MoeFfnSpec, MoeProjectionStrategy, MoeRouter, append_moe_ffn,
    input_leaf, scalar_constant,
};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{DType, Extent, NumericPolicy};

const EXPERT_COUNT: u32 = 256;
const EXPERT_USED_COUNT: u32 = 8;
const EMBEDDING: u32 = 8;
const FEED_FORWARD: u32 = 8;

struct RoutingFixture {
    program: Vec<proxima_tensor::Op>,
    output: proxima_tensor::NodeId,
    selected: Vec<proxima_tensor::NodeId>,
    weights: Vec<proxima_tensor::NodeId>,
}

fn build_fixture() -> RoutingFixture {
    let mut program = Vec::new();
    let x = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(1), Extent::Static(EMBEDDING)],
        "x",
    );
    let logits = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(1), Extent::Static(EXPERT_COUNT)],
        "logits",
    );
    let expert_w_gate = input_leaf(
        &mut program,
        DType::Float32,
        vec![
            Extent::Static(EXPERT_COUNT),
            Extent::Static(EMBEDDING),
            Extent::Static(FEED_FORWARD),
        ],
        "expert_w_gate",
    );
    let expert_w_up = input_leaf(
        &mut program,
        DType::Float32,
        vec![
            Extent::Static(EXPERT_COUNT),
            Extent::Static(EMBEDDING),
            Extent::Static(FEED_FORWARD),
        ],
        "expert_w_up",
    );
    let expert_w_down = input_leaf(
        &mut program,
        DType::Float32,
        vec![
            Extent::Static(EXPERT_COUNT),
            Extent::Static(FEED_FORWARD),
            Extent::Static(EMBEDDING),
        ],
        "expert_w_down",
    );
    let one = scalar_constant(&mut program, 1.0);
    let moe_spec = MoeFfnSpec {
        router: MoeRouter::Logits(logits),
        expert_w_gate,
        expert_w_up,
        expert_w_down,
        expert_count: EXPERT_COUNT,
        expert_used_count: EXPERT_USED_COUNT,
        ones: one,
        gating: ExpertGatingFunc::Softmax,
        expert_bias: None,
        expert_scale: None,
        activation: Activation::Silu,
        strategy: MoeProjectionStrategy::PerRoute,
    };
    let (output, site) = append_moe_ffn(&mut program, 0, x, &moe_spec)
        .expect("real-shape qwen35moe routing block lowers");
    RoutingFixture {
        program,
        output,
        selected: site.selected,
        weights: site.weights,
    }
}

fn run_case(fixture: &RoutingFixture, logits_data: &[f32], case: usize, failures: &mut Vec<String>) {
    let mut outputs = vec![fixture.output];
    outputs.extend(fixture.selected.iter().copied());
    outputs.extend(fixture.weights.iter().copied());

    let x_data = vec![0.1_f32; EMBEDDING as usize];
    let expert_w_gate_data =
        vec![0.0_f32; EXPERT_COUNT as usize * EMBEDDING as usize * FEED_FORWARD as usize];
    let expert_w_up_data = expert_w_gate_data.clone();
    let expert_w_down_data = expert_w_gate_data.clone();

    let symbols: Vec<u64> = Vec::new();
    let named: Vec<(&str, proxima_tensor::QuantizedBlock)> = vec![
        ("x", proxima_tensor::QuantizedBlock::Float32(&x_data)),
        ("logits", proxima_tensor::QuantizedBlock::Float32(logits_data)),
        (
            "expert_w_gate",
            proxima_tensor::QuantizedBlock::Float32(&expert_w_gate_data),
        ),
        (
            "expert_w_up",
            proxima_tensor::QuantizedBlock::Float32(&expert_w_up_data),
        ),
        (
            "expert_w_down",
            proxima_tensor::QuantizedBlock::Float32(&expert_w_down_data),
        ),
    ];

    let mut free_buffers = Vec::new();
    let mut validated = None;
    let cpu = proxima_tensor::cpu::evaluate_quantized_named_with_scratch(
        &fixture.program,
        &symbols,
        &named,
        &outputs,
        &mut free_buffers,
        &mut validated,
    )
    .expect("cpu unfused-or-fused routing evaluates");

    let plan = omega::plan_named(
        &fixture.program,
        &symbols,
        &named,
        &outputs,
        NumericPolicy::default(),
    )
    .expect("metal plan builds");
    let metal = omega::execute_plan_named(&plan, &named).expect("metal routing evaluates");

    for (round, &route_node) in fixture.selected.iter().enumerate() {
        let cpu_route = cpu.get(route_node).map(|(values, _)| values[0]);
        let metal_route = metal.get(route_node).map(|(values, _)| values[0]);
        if cpu_route != metal_route {
            failures.push(format!(
                "case {case} round {round}: route cpu={cpu_route:?} metal={metal_route:?}"
            ));
        }
    }
    for (round, &weight_node) in fixture.weights.iter().take(EXPERT_USED_COUNT as usize).enumerate() {
        let cpu_weight = cpu.get(weight_node).map(|(values, _)| values[0]);
        let metal_weight = metal.get(weight_node).map(|(values, _)| values[0]);
        match (cpu_weight, metal_weight) {
            (Some(cpu_weight), Some(metal_weight)) if (cpu_weight - metal_weight).abs() <= 1e-6 => {}
            (cpu_weight, metal_weight) => failures.push(format!(
                "case {case} round {round}: weight cpu={cpu_weight:?} metal={metal_weight:?}"
            )),
        }
    }
    let weight_total_node = *fixture
        .weights
        .last()
        .expect("weights carries weight_total as its last entry");
    let cpu_total = cpu.get(weight_total_node).map(|(values, _)| values[0]);
    let metal_total = metal.get(weight_total_node).map(|(values, _)| values[0]);
    match (cpu_total, metal_total) {
        (Some(cpu_total), Some(metal_total)) if (cpu_total - metal_total).abs() <= 1e-5 => {}
        (cpu_total, metal_total) => failures.push(format!(
            "case {case}: weight_total cpu={cpu_total:?} metal={metal_total:?}"
        )),
    }
}

#[test]
fn moe_topk_metal_matches_cpu_at_real_shape_over_random_scores_and_exact_ties() {
    let fixture = build_fixture();
    let mut failures = Vec::new();

    let mut lcg = Lcg(4_242);
    for case in 0..50 {
        let logits: Vec<f32> = (0..EXPERT_COUNT as usize)
            .map(|_| lcg.next_unit() * 10.0)
            .collect();
        run_case(&fixture, &logits, case, &mut failures);
    }

    let mut top_tie = (0..EXPERT_COUNT as usize)
        .map(|index| index as f32 * 0.01)
        .collect::<Vec<f32>>();
    top_tie[12] = 9.0;
    top_tie[200] = 9.0;
    run_case(&fixture, &top_tie, 50, &mut failures);

    let mut later_tie = (0..EXPERT_COUNT as usize)
        .map(|index| index as f32 * 0.01)
        .collect::<Vec<f32>>();
    later_tie[5] = 20.0;
    later_tie[40] = 7.0;
    later_tie[220] = 7.0;
    run_case(&fixture, &later_tie, 51, &mut failures);

    assert!(
        failures.is_empty(),
        "moe topk metal tie parity failures:\n{}",
        failures.join("\n")
    );
}
