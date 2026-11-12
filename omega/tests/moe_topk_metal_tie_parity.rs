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
use proxima_tensor::{Extent};
use proxima_numeric::{DType, NumericPolicy};

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
    build_gated_fixture(ExpertGatingFunc::Softmax, false)
}

fn build_gated_fixture(gating: ExpertGatingFunc, with_bias: bool) -> RoutingFixture {
    build_token_fixture(gating, with_bias, MoeProjectionStrategy::PerRoute, 1)
}

fn build_token_fixture(
    gating: ExpertGatingFunc,
    with_bias: bool,
    strategy: MoeProjectionStrategy,
    tokens: u32,
) -> RoutingFixture {
    let mut program = Vec::new();
    let x = input_leaf(
        &mut program,
        DType::FLOAT32,
        vec![Extent::Static(tokens), Extent::Static(EMBEDDING)],
        "x",
    );
    let logits = input_leaf(
        &mut program,
        DType::FLOAT32,
        vec![Extent::Static(tokens), Extent::Static(EXPERT_COUNT)],
        "logits",
    );
    let expert_w_gate = input_leaf(
        &mut program,
        DType::FLOAT32,
        vec![
            Extent::Static(EXPERT_COUNT),
            Extent::Static(EMBEDDING),
            Extent::Static(FEED_FORWARD),
        ],
        "expert_w_gate",
    );
    let expert_w_up = input_leaf(
        &mut program,
        DType::FLOAT32,
        vec![
            Extent::Static(EXPERT_COUNT),
            Extent::Static(EMBEDDING),
            Extent::Static(FEED_FORWARD),
        ],
        "expert_w_up",
    );
    let expert_w_down = input_leaf(
        &mut program,
        DType::FLOAT32,
        vec![
            Extent::Static(EXPERT_COUNT),
            Extent::Static(FEED_FORWARD),
            Extent::Static(EMBEDDING),
        ],
        "expert_w_down",
    );
    let one = scalar_constant(&mut program, 1.0);
    let expert_bias = with_bias.then(|| {
        input_leaf(
            &mut program,
            DType::FLOAT32,
            vec![Extent::Static(EXPERT_COUNT)],
            "expert_bias",
        )
    });
    let moe_spec = MoeFfnSpec {
        router: MoeRouter::Logits(logits),
        expert_w_gate,
        expert_w_up,
        expert_w_down,
        expert_count: EXPERT_COUNT,
        expert_used_count: EXPERT_USED_COUNT,
        ones: one,
        gating,
        expert_bias,
        expert_scale: None,
        activation: Activation::Silu,
        strategy,
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

fn run_case(
    fixture: &RoutingFixture,
    logits_data: &[f32],
    case: usize,
    failures: &mut Vec<String>,
) {
    run_biased_case(fixture, logits_data, None, case, failures);
}

fn run_biased_case(
    fixture: &RoutingFixture,
    logits_data: &[f32],
    bias_data: Option<&[f32]>,
    case: usize,
    failures: &mut Vec<String>,
) {
    let mut outputs = vec![fixture.output];
    outputs.extend(fixture.selected.iter().copied());
    outputs.extend(fixture.weights.iter().copied());

    let x_data = vec![0.1_f32; EMBEDDING as usize];
    let expert_w_gate_data =
        vec![0.0_f32; EXPERT_COUNT as usize * EMBEDDING as usize * FEED_FORWARD as usize];
    let expert_w_up_data = expert_w_gate_data.clone();
    let expert_w_down_data = expert_w_gate_data.clone();

    let symbols: Vec<u64> = Vec::new();
    let mut named: Vec<(&str, proxima_tensor::QuantizedBlock)> = vec![
        ("x", proxima_tensor::QuantizedBlock::Float32(&x_data)),
        (
            "logits",
            proxima_tensor::QuantizedBlock::Float32(logits_data),
        ),
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

    if let Some(bias_data) = bias_data {
        named.push(("expert_bias", proxima_tensor::QuantizedBlock::Float32(bias_data)));
    }

    let mut free_buffers = proxima_tensor::cpu::BufferPools::default();
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
    for (round, &weight_node) in fixture
        .weights
        .iter()
        .take(EXPERT_USED_COUNT as usize)
        .enumerate()
    {
        let cpu_weight = cpu.get(weight_node).map(|(values, _)| values[0]);
        let metal_weight = metal.get(weight_node).map(|(values, _)| values[0]);
        match (cpu_weight, metal_weight) {
            (Some(cpu_weight), Some(metal_weight)) if (cpu_weight - metal_weight).abs() <= 1e-6 => {
            }
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

/// flint's router: sigmoid gating with the selection bias. The Metal kernel selects on
/// `sigmoid(logit) + bias` and weighs each chosen expert by the unbiased `sigmoid(logit)`
/// (`MoeWeighting::SelectedScore`); the CPU evaluator is the oracle.
#[test]
fn moe_topk_metal_sigmoid_biased_matches_cpu_over_random_scores_and_exact_ties() {
    let fixture = build_gated_fixture(ExpertGatingFunc::Sigmoid, true);
    let mut failures = Vec::new();

    let mut lcg = Lcg(7_331);
    for case in 0..50 {
        let logits: Vec<f32> = (0..EXPERT_COUNT as usize)
            .map(|_| lcg.next_unit() * 8.0 - 4.0)
            .collect();
        let bias: Vec<f32> = (0..EXPERT_COUNT as usize)
            .map(|_| lcg.next_unit() * 0.5)
            .collect();
        run_biased_case(&fixture, &logits, Some(&bias), case, &mut failures);
    }

    let ramp: Vec<f32> = (0..EXPERT_COUNT as usize)
        .map(|index| index as f32 * 0.01 - 1.0)
        .collect();
    let mut top_tie_logits = ramp.clone();
    top_tie_logits[200] = top_tie_logits[12];
    let mut top_tie_bias = vec![0.0_f32; EXPERT_COUNT as usize];
    top_tie_bias[12] = 1.0;
    top_tie_bias[200] = 1.0;
    run_biased_case(&fixture, &top_tie_logits, Some(&top_tie_bias), 50, &mut failures);

    let mut later_tie_logits = ramp;
    later_tie_logits[220] = later_tie_logits[40];
    let mut later_tie_bias = vec![0.0_f32; EXPERT_COUNT as usize];
    later_tie_bias[5] = 2.0;
    later_tie_bias[40] = 1.0;
    later_tie_bias[220] = 1.0;
    run_biased_case(&fixture, &later_tie_logits, Some(&later_tie_bias), 51, &mut failures);

    assert!(
        failures.is_empty(),
        "moe topk metal sigmoid parity failures:\n{}",
        failures.join("\n")
    );
}

/// A prefill-width batch: every token row of the fused routing must agree with the CPU oracle,
/// not only row zero, under each gating function and projection strategy.
#[test]
fn moe_topk_metal_matches_cpu_for_every_row_of_a_prefill_batch() {
    const TOKENS: u32 = 84;
    let mut failures = Vec::new();
    for (gating, with_bias) in [
        (ExpertGatingFunc::Softmax, false),
        (ExpertGatingFunc::Sigmoid, true),
    ] {
        for strategy in [MoeProjectionStrategy::PerRoute, MoeProjectionStrategy::Stacked] {
            let fixture = build_token_fixture(gating, with_bias, strategy, TOKENS);
            let mut outputs = vec![fixture.output];
            outputs.extend(fixture.selected.iter().copied());
            outputs.extend(fixture.weights.iter().copied());

            let mut lcg = Lcg(99);
            let logits: Vec<f32> = (0..(TOKENS * EXPERT_COUNT) as usize)
                .map(|_| lcg.next_unit() * 8.0 - 4.0)
                .collect();
            let bias: Vec<f32> = (0..EXPERT_COUNT as usize)
                .map(|_| lcg.next_unit() * 0.5)
                .collect();
            let x_data = vec![0.1_f32; (TOKENS * EMBEDDING) as usize];
            let weight_data =
                vec![0.0_f32; (EXPERT_COUNT * EMBEDDING * FEED_FORWARD) as usize];
            let symbols: Vec<u64> = Vec::new();
            let mut named: Vec<(&str, proxima_tensor::QuantizedBlock)> = vec![
                ("x", proxima_tensor::QuantizedBlock::Float32(&x_data)),
                ("logits", proxima_tensor::QuantizedBlock::Float32(&logits)),
                ("expert_w_gate", proxima_tensor::QuantizedBlock::Float32(&weight_data)),
                ("expert_w_up", proxima_tensor::QuantizedBlock::Float32(&weight_data)),
                ("expert_w_down", proxima_tensor::QuantizedBlock::Float32(&weight_data)),
            ];
            if with_bias {
                named.push(("expert_bias", proxima_tensor::QuantizedBlock::Float32(&bias)));
            }
            let mut free_buffers = proxima_tensor::cpu::BufferPools::default();
            let mut validated = None;
            let cpu = proxima_tensor::cpu::evaluate_quantized_named_with_scratch(
                &fixture.program,
                &symbols,
                &named,
                &outputs,
                &mut free_buffers,
                &mut validated,
            )
            .expect("cpu routing evaluates");
            let plan = omega::plan_named(
                &fixture.program,
                &symbols,
                &named,
                &outputs,
                NumericPolicy::default(),
            )
            .expect("metal plan builds");
            let metal = match omega::execute_plan_named(&plan, &named) {
                Ok(metal) => metal,
                Err(error) => {
                    failures.push(format!("{gating:?} {strategy:?}: metal failed: {error:?}"));
                    continue;
                }
            };
            for node in fixture.selected.iter().chain(fixture.weights.iter()) {
                let cpu_values = cpu.get(*node).map(|(values, _)| values.to_vec());
                let metal_values = metal.get(*node).map(|(values, _)| values.to_vec());
                let agrees = match (&cpu_values, &metal_values) {
                    (Some(left), Some(right)) => {
                        left.len() == right.len()
                            && left.iter().zip(right).all(|(a, b)| (a - b).abs() <= 1e-5)
                    }
                    _ => false,
                };
                if !agrees {
                    failures.push(format!("{gating:?} {strategy:?}: node {} differs", node.0));
                }
            }
        }
    }
    assert!(failures.is_empty(), "prefill batch parity failures:\n{}", failures.join("\n"));
}

/// A row whose scores are all NaN has no lane equal to its maximum. The unfused chain
/// (`reduce(Maximum, ReduceInit::Zero)` over `mask * expert_index`) routes it to expert 0 in every
/// round; the fused kernel must do the same instead of emitting a negative expert index, which a
/// downstream gather reports as `GatherIndexOutOfRange`.
#[test]
fn moe_topk_metal_routes_a_nan_row_to_expert_zero_like_the_unfused_chain() {
    const TOKENS: u32 = 3;
    for (gating, with_bias) in [
        (ExpertGatingFunc::Softmax, false),
        (ExpertGatingFunc::Sigmoid, true),
    ] {
        let fixture =
            build_token_fixture(gating, with_bias, MoeProjectionStrategy::Stacked, TOKENS);
        let mut outputs = vec![fixture.output];
        outputs.extend(fixture.selected.iter().copied());

        let mut lcg = Lcg(5);
        let mut logits: Vec<f32> = (0..(TOKENS * EXPERT_COUNT) as usize)
            .map(|_| lcg.next_unit() * 8.0 - 4.0)
            .collect();
        logits[EXPERT_COUNT as usize..2 * EXPERT_COUNT as usize].fill(f32::NAN);
        let bias = vec![0.1_f32; EXPERT_COUNT as usize];
        let x_data = vec![0.1_f32; (TOKENS * EMBEDDING) as usize];
        let weight_data = vec![0.0_f32; (EXPERT_COUNT * EMBEDDING * FEED_FORWARD) as usize];
        let mut named: Vec<(&str, proxima_tensor::QuantizedBlock)> = vec![
            ("x", proxima_tensor::QuantizedBlock::Float32(&x_data)),
            ("logits", proxima_tensor::QuantizedBlock::Float32(&logits)),
            ("expert_w_gate", proxima_tensor::QuantizedBlock::Float32(&weight_data)),
            ("expert_w_up", proxima_tensor::QuantizedBlock::Float32(&weight_data)),
            ("expert_w_down", proxima_tensor::QuantizedBlock::Float32(&weight_data)),
        ];
        if with_bias {
            named.push(("expert_bias", proxima_tensor::QuantizedBlock::Float32(&bias)));
        }
        let symbols: Vec<u64> = Vec::new();
        let plan = omega::plan_named(
            &fixture.program,
            &symbols,
            &named,
            &outputs,
            NumericPolicy::default(),
        )
        .expect("metal plan builds");
        let metal = omega::execute_plan_named(&plan, &named)
            .unwrap_or_else(|error| panic!("{gating:?}: a NaN row must not fault: {error:?}"));
        for (round, node) in fixture.selected.iter().enumerate() {
            let (values, _) = metal.get(*node).expect("route node resolves");
            assert_eq!(values[1], 0.0, "{gating:?} round {round}: the NaN row routes to expert 0");
            assert!(
                values[0] >= 0.0 && values[2] >= 0.0,
                "{gating:?} round {round}: finite rows keep valid experts"
            );
        }
    }
}
