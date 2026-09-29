//! Internal consistency of the round-batched MoE fold on a real Metal device
//! (this is the SAME backend run two ways, not parity with an outside oracle).
//!
//! `MoeTopK` writes every round's route buffer on the GPU, and the collapsed
//! `RoundBatchedReduce` reads them through its own bound buffers in the SAME
//! command buffer, so the ordering between the two dispatches is what this
//! file exercises. The program is built routes-first -- every round's argmax
//! chain, then every round's gate and up projection -- because
//! `moe_round_group_is_contiguous` only admits a group whose round routes all
//! precede round 0's reduce in program order, and `append_moe_ffn`'s per-route
//! unrolling interleaves each round's projections with the next round's
//! routing, so it never admits one.
//!
//! Each shape runs twice: collapsed, and with the collapse switched off
//! (`PROXIMA_DISABLE_MOE_ROUND_GROUP_FUSION`), which dispatches the k reduces
//! one round at a time. `MoeTopK` fires in both, so the only difference is the
//! round-batched kernel; every output element must be bit-for-bit equal.

#![cfg(all(
    feature = "metal",
    feature = "metal-moe-mul-mat-id",
    feature = "moe-topk-fusion",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_lines)]

use proxima_tensor::spec::{elementwise, gathered_expert_product, input_leaf, reduce, scalar_constant};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    BoundOpKind, DType, Extent, NodeId, NumericPolicy, Op, QuantizedBlock, ReduceInit, ScalarOp,
    append,
};

const EMBEDDING: u32 = 32;
const EXPERT_HIDDEN: u32 = 16;
const DISABLE_ROUND_GROUP_FUSION: &str = "PROXIMA_DISABLE_MOE_ROUND_GROUP_FUSION";

struct RoutesFirstProgram {
    program: Vec<Op>,
    outputs: Vec<NodeId>,
    named: Vec<(&'static str, Vec<f32>)>,
}

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit() * 2.0 - 1.0).collect()
}

fn stack_leaf(program: &mut Vec<Op>, name: &str, expert_count: u32, rows: u32, columns: u32) -> NodeId {
    input_leaf(
        program,
        DType::Float32,
        vec![
            Extent::Static(expert_count),
            Extent::Static(rows),
            Extent::Static(columns),
        ],
        name,
    )
}

/// `append_moe_ffn`'s softmax argmax-with-exclusion chain for `top_k` rounds,
/// node for node (the shape `match_moe_topk` anchors on), returning each
/// round's route and weight.
fn routing_rounds(
    program: &mut Vec<Op>,
    logits: NodeId,
    expert_count: u32,
    top_k: u32,
) -> (Vec<NodeId>, Vec<NodeId>) {
    let expert_index = append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(expert_count),
        },
    );
    let neg_infinity = scalar_constant(program, f32::NEG_INFINITY);
    let mut selection_scores = logits;
    let mut first_max = None;
    let mut routes = Vec::new();
    let mut weights = Vec::new();
    let mut weight_total: Option<NodeId> = None;
    for round in 0..top_k {
        let max_selection = reduce(
            program,
            DType::Float32,
            ScalarOp::Maximum,
            ReduceInit::NegativeInfinity,
            selection_scores,
            "se->se",
            "s->se",
        )
        .expect("max selection reduce");
        let mask = elementwise(
            program,
            DType::Float32,
            ScalarOp::Equal,
            &[(selection_scores, "se->se"), (max_selection, "s->se")],
        )
        .expect("selection mask");
        let candidate = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(mask, "se->se"), (expert_index, "e->se")],
        )
        .expect("candidate index");
        let route = reduce(
            program,
            DType::Int32,
            ScalarOp::Maximum,
            ReduceInit::Zero,
            candidate,
            "se->se",
            "s->se",
        )
        .expect("route reduce");
        let anchor = *first_max.get_or_insert(max_selection);
        let shifted = elementwise(
            program,
            DType::Float32,
            ScalarOp::Subtract,
            &[(max_selection, "s->s"), (anchor, "s->s")],
        )
        .expect("shifted score");
        let weight = elementwise(
            program,
            DType::Float32,
            ScalarOp::Exponential,
            &[(shifted, "s->s")],
        )
        .expect("round weight");
        routes.push(route);
        weights.push(weight);
        weight_total = Some(match weight_total {
            Some(running) => elementwise(
                program,
                DType::Float32,
                ScalarOp::Add,
                &[(running, "s->s"), (weight, "s->s")],
            )
            .expect("weight total"),
            None => weight,
        });
        if round + 1 < top_k {
            selection_scores = elementwise(
                program,
                DType::Float32,
                ScalarOp::Select,
                &[
                    (mask, "se->se"),
                    (neg_infinity, "->se"),
                    (selection_scores, "se->se"),
                ],
            )
            .expect("exclusion select");
        }
    }
    (routes, weights)
}

fn projection_round(program: &mut Vec<Op>, stack: NodeId, route: NodeId, x: NodeId) -> NodeId {
    let product = gathered_expert_product(program, stack, route, x);
    reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "sio->sio",
        "so->sio",
    )
    .expect("projection reduce")
}

fn routes_first_program(expert_count: u32, top_k: u32) -> RoutesFirstProgram {
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
        vec![Extent::Static(1), Extent::Static(expert_count)],
        "logits",
    );
    let expert_w_gate = stack_leaf(&mut program, "expert_w_gate", expert_count, EMBEDDING, EXPERT_HIDDEN);
    let expert_w_up = stack_leaf(&mut program, "expert_w_up", expert_count, EMBEDDING, EXPERT_HIDDEN);

    let (routes, weights) = routing_rounds(&mut program, logits, expert_count, top_k);
    let gates: Vec<NodeId> = routes
        .iter()
        .map(|&route| projection_round(&mut program, expert_w_gate, route, x))
        .collect();
    let ups: Vec<NodeId> = routes
        .iter()
        .map(|&route| projection_round(&mut program, expert_w_up, route, x))
        .collect();

    let mut outputs = gates;
    outputs.extend(ups);
    outputs.extend(routes.iter().skip(1));
    outputs.extend(weights);
    let stack_len = (expert_count * EMBEDDING * EXPERT_HIDDEN) as usize;
    let named = vec![
        ("x", random_vec(2_001, EMBEDDING as usize)),
        ("logits", random_vec(2_002, expert_count as usize)),
        ("expert_w_gate", random_vec(2_003, stack_len)),
        ("expert_w_up", random_vec(2_004, stack_len)),
    ];
    RoutesFirstProgram {
        program,
        outputs,
        named,
    }
}

fn round_batched_and_topk_counts(fixture: &RoutesFirstProgram) -> (usize, usize) {
    let shapes = proxima_tensor::infer(&fixture.program, &[]).expect("fixture infers");
    let bound = proxima_tensor::bind(
        &fixture.program,
        &shapes,
        &fixture.outputs,
        NumericPolicy::default(),
    )
    .expect("fixture binds");
    let count = |wanted: fn(&BoundOpKind) -> bool| bound.iter().filter(|op| wanted(&op.kind)).count();
    (
        count(|kind| matches!(kind, BoundOpKind::RoundBatchedReduce { .. })),
        count(|kind| matches!(kind, BoundOpKind::MoeTopK { .. })),
    )
}

fn run_on_metal(fixture: &RoutesFirstProgram) -> proxima_tensor::cpu::Evaluated {
    let named: Vec<(&str, QuantizedBlock)> = fixture
        .named
        .iter()
        .map(|(name, data)| (*name, QuantizedBlock::Float32(data.as_slice())))
        .collect();
    let plan = omega::plan_named(
        &fixture.program,
        &[],
        &named,
        &fixture.outputs,
        NumericPolicy::default(),
    )
    .expect("metal plans the routes-first moe projections");
    omega::execute_plan_named(&plan, &named).expect("metal runs the moe projections on a real device")
}

fn output_bits(evaluated: &proxima_tensor::cpu::Evaluated, node: NodeId) -> Option<Vec<u32>> {
    evaluated
        .get(node)
        .map(|(values, _shape)| values.iter().map(|value| value.to_bits()).collect())
}

#[test]
fn round_batched_moe_matches_the_per_round_dispatches_on_metal() {
    let mut failures = Vec::new();

    for &expert_count in &[8u32, 32, 128] {
        for &top_k in &[2u32, 4, 8] {
            let fixture = routes_first_program(expert_count, top_k);
            let label = format!("experts={expert_count} top_k={top_k}");

            let (batched_ops, topk_ops) = round_batched_and_topk_counts(&fixture);
            let (per_round_ops, per_round_topk_ops) = temp_env::with_var(
                DISABLE_ROUND_GROUP_FUSION,
                Some("1"),
                || round_batched_and_topk_counts(&fixture),
            );
            if batched_ops != 2 || topk_ops != 1 || per_round_ops != 0 || per_round_topk_ops != 1 {
                failures.push(format!(
                    "{label} degenerate comparison: collapsed round_batched={batched_ops} \
                     moe_topk={topk_ops}, per-round round_batched={per_round_ops} \
                     moe_topk={per_round_topk_ops} (want 2/1 and 0/1: one gate and one up group)"
                ));
                continue;
            }

            let batched = run_on_metal(&fixture);
            let per_round = temp_env::with_var(
                DISABLE_ROUND_GROUP_FUSION,
                Some("1"),
                || run_on_metal(&fixture),
            );
            for (slot, &node) in fixture.outputs.iter().enumerate() {
                let batched_bits = output_bits(&batched, node);
                if batched_bits.is_none() || batched_bits != output_bits(&per_round, node) {
                    failures.push(format!(
                        "{label} output_slot={slot} node={node:?} batched={batched_bits:?} \
                         per_round={:?}",
                        output_bits(&per_round, node)
                    ));
                }
            }
            eprintln!(
                "{label}: round_batched_ops={batched_ops} moe_topk_ops={topk_ops} outputs_compared={}",
                fixture.outputs.len()
            );
        }
    }

    assert!(
        failures.is_empty(),
        "round-batched vs per-round moe differences:\n{}",
        failures.join("\n")
    );
}
