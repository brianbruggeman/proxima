//! The adjoint of a read through an affine operand map with more than one
//! term per axis is the scatter-add of the upstream gradient along that map
//! (`proxima-autograd/src/affine_scatter.rs`). Every case here builds
//! `loss = sum((x[m(i)] * w[i])^2)`, so `d loss / d x[j]` is
//! `sum over i with m(i) = j of 2 * x[j] * w[i]^2`: nonlinear in `x`, so the
//! gradient depends on the operand value as well as on the scatter, and
//! different `i` that land on one `j` must add.
//!
//! The maps are the ones the granite census refused: the kv-head group merge
//! `[i0, 2*i1 + i2, i3]` and the rope even/odd interleave
//! `[i0, i1, 2*i2 + 0]` / `[i0, i1, 2*i2 + 1]`, plus a three-term axis
//! (`3*i + 2*j + 1`) no granite node has.
//!
//! Two checks per element of `x`. The f64 loop enumerates the iteration space
//! and evaluates the pattern itself, with no call into the adjoint, and the
//! adjoint (f32) must match it within `1e-5 * (1 + abs(expected))`, a few
//! f32 additions of terms of order 1. The central difference (step `1e-2`;
//! the loss is quadratic in `x`, so the difference is exact up to f32
//! rounding of a loss of order 10) uses `1e-2 + 1e-2 * abs(numeric)`, the
//! combined tolerance `adjoint_laws.rs` uses for every numeric reference.
//!
//! The `adjoint_affine_map_` tests are the five `proxima-tensor/specs/
//! fp6-qat-ablation` AC1a names; the rest pin the dispatch (pure projections
//! keep the direct `Reduce`), granite's own head counts
//! (`proxima-model-interop/tests/fixtures/model-configs/granite_moe.toml`: 16
//! query heads over 8 kv heads), and the refusals that remain.
#![allow(clippy::unwrap_used, clippy::expect_used)]

extern crate alloc;

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use proxima_autograd::AutogradError;
use proxima_autograd::adjoint::differentiate;
use proxima_tensor::cpu::evaluate_named;
use proxima_tensor::dtype::DType;
use proxima_tensor::map::{self, AxisTerm, IndexMap, IndexPattern};
use proxima_tensor::op::{self, Extent, NodeId, Op, Reduce, ReduceInit, ScalarOp};

struct Case {
    name: &'static str,
    iteration: Vec<u32>,
    operand: Vec<u32>,
    pattern: IndexPattern,
}

fn kv_head_merge(sequence: u32, kv_heads: u32, head_width: u32) -> Case {
    Case {
        name: "kv_head_merge",
        iteration: vec![sequence, kv_heads, 2, head_width],
        operand: vec![sequence, 2 * kv_heads, head_width],
        pattern: map::affine(
            4,
            &[
                (&[AxisTerm::projection(0)], 0),
                (&[AxisTerm::scaled(1, 2), AxisTerm::projection(2)], 0),
                (&[AxisTerm::projection(3)], 0),
            ],
        ),
    }
}

fn rope_interleave(sequence: u32, heads: u32, head_width: u32, parity: i32) -> Case {
    Case {
        name: if parity == 0 { "rope_even" } else { "rope_odd" },
        iteration: vec![sequence, heads, head_width / 2],
        operand: vec![sequence, heads, head_width],
        pattern: map::affine(
            3,
            &[
                (&[AxisTerm::projection(0)], 0),
                (&[AxisTerm::projection(1)], 0),
                (&[AxisTerm::scaled(2, 2)], parity),
            ],
        ),
    }
}

fn three_term_with_collisions() -> Case {
    Case {
        name: "three_term",
        iteration: vec![3, 4],
        operand: vec![14],
        pattern: map::affine(2, &[(&[AxisTerm::scaled(0, 3), AxisTerm::scaled(1, 2)], 1)]),
    }
}

fn input(program: &mut Vec<Op>, name: &str, shape: &[u32]) -> NodeId {
    op::append(
        program,
        Op::Input {
            dtype: DType::Float32,
            shape: shape.iter().map(|&extent| Extent::Static(extent)).collect(),
            name: Some(name.into()),
        },
    )
}

fn elementwise(program: &mut Vec<Op>, body: ScalarOp, operands: Vec<(NodeId, IndexMap)>) -> NodeId {
    op::append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body,
            operands,
            name: None,
        },
    )
}

fn identity(rank: u16) -> IndexMap {
    IndexMap::Affine(map::projection(rank, &(0..rank).collect::<Vec<u16>>()))
}

fn sum_to_scalar(program: &mut Vec<Op>, operand: NodeId, rank: u16) -> NodeId {
    op::append(
        program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand,
            in_map: identity(rank),
            out_map: IndexMap::Affine(map::projection(rank, &[])),
            keep: op::Keep::Reduce,
            name: None,
        }),
    )
}

fn squared_product_loss(case: &Case) -> (Vec<Op>, NodeId) {
    let rank = u16::try_from(case.iteration.len()).unwrap();
    let mut program = Vec::new();
    let operand = input(&mut program, "x", &case.operand);
    let weight = input(&mut program, "w", &case.iteration);
    let product = elementwise(
        &mut program,
        ScalarOp::Multiply,
        vec![
            (operand, IndexMap::Affine(case.pattern.clone())),
            (weight, identity(rank)),
        ],
    );
    let squared = elementwise(
        &mut program,
        ScalarOp::Multiply,
        vec![(product, identity(rank)), (product, identity(rank))],
    );
    let loss = sum_to_scalar(&mut program, squared, rank);
    (program, loss)
}

fn element_count(shape: &[u32]) -> usize {
    shape.iter().map(|&extent| extent as usize).product()
}

fn pseudo_random(seed: u64, count: usize) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn unravel(mut flat: usize, shape: &[u32]) -> Vec<usize> {
    let mut coordinates = vec![0; shape.len()];
    for (slot, &extent) in coordinates.iter_mut().zip(shape).rev() {
        *slot = flat % extent as usize;
        flat /= extent as usize;
    }
    coordinates
}

fn ravel(coordinates: &[usize], shape: &[u32]) -> usize {
    coordinates
        .iter()
        .zip(shape)
        .fold(0, |flat, (&coordinate, &extent)| {
            flat * extent as usize + coordinate
        })
}

fn operand_flat_index(case: &Case, point: &[usize]) -> usize {
    let coordinates: Vec<usize> = case
        .pattern
        .axes
        .iter()
        .map(|axis| {
            let value: i64 = axis
                .terms
                .iter()
                .map(|term| {
                    i64::from(term.coeff) * i64::try_from(point[usize::from(term.axis)]).unwrap()
                })
                .sum::<i64>()
                + i64::from(axis.offset);
            usize::try_from(value).expect("the map stays inside the operand")
        })
        .collect();
    ravel(&coordinates, &case.operand)
}

fn f64_loop_gradient(case: &Case, operand: &[f32], weight: &[f32]) -> Vec<f64> {
    let mut gradient = vec![0.0_f64; operand.len()];
    for (flat, weight_value) in weight.iter().enumerate() {
        let target = operand_flat_index(case, &unravel(flat, &case.iteration));
        let (x_value, w_value) = (f64::from(operand[target]), f64::from(*weight_value));
        gradient[target] += 2.0 * x_value * w_value * w_value;
    }
    gradient
}

fn adjoint_gradient(case: &Case, operand: &[f32], weight: &[f32]) -> Vec<f32> {
    let (program, loss) = squared_product_loss(case);
    let differentiated =
        differentiate(&program, loss).expect("affine multi-term map differentiates");
    let gradient_node = differentiated
        .gradient_of_named("x")
        .expect("x feeds the loss");
    let evaluated = evaluate_named(
        &differentiated.program,
        &[],
        &[("x", operand), ("w", weight)],
        &[gradient_node],
    )
    .expect("adjoint program evaluates");
    evaluated
        .get(gradient_node)
        .expect("gradient evaluated")
        .0
        .to_vec()
}

fn loss_value(program: &[Op], loss: NodeId, operand: &[f32], weight: &[f32]) -> f32 {
    let evaluated = evaluate_named(program, &[], &[("x", operand), ("w", weight)], &[loss])
        .expect("forward program evaluates");
    evaluated.get(loss).expect("loss evaluated").0[0]
}

fn compare_to_loop(name: &str, analytic: &[f32], expected: &[f64]) -> Result<(), String> {
    if analytic.len() != expected.len() {
        return Err(format!(
            "{name}: gradient has {} elements, operand has {}",
            analytic.len(),
            expected.len()
        ));
    }
    for (index, (got, want)) in analytic.iter().zip(expected).enumerate() {
        if (f64::from(*got) - want).abs() > 1e-5 * (1.0 + want.abs()) {
            return Err(format!(
                "{name} x[{index}]: adjoint {got} vs f64 loop {want}"
            ));
        }
    }
    Ok(())
}

fn compare_to_central_difference(
    case: &Case,
    analytic: &[f32],
    operand: &[f32],
    weight: &[f32],
) -> Result<(), String> {
    let (program, loss) = squared_product_loss(case);
    let step = 1e-2_f32;
    for (index, got) in analytic.iter().enumerate() {
        let (mut up, mut down) = (operand.to_vec(), operand.to_vec());
        up[index] += step;
        down[index] -= step;
        let numeric = (loss_value(&program, loss, &up, weight)
            - loss_value(&program, loss, &down, weight))
            / (2.0 * step);
        if (got - numeric).abs() > 1e-2 + 1e-2 * numeric.abs() {
            return Err(format!(
                "{} x[{index}]: adjoint {got} vs central difference {numeric}",
                case.name
            ));
        }
    }
    Ok(())
}

fn check_case(case: &Case) -> Result<(), String> {
    let operand = pseudo_random(7, element_count(&case.operand));
    let weight = pseudo_random(11, element_count(&case.iteration));
    let analytic = adjoint_gradient(case, &operand, &weight);
    let expected = f64_loop_gradient(case, &operand, &weight);
    if expected.iter().all(|value| *value == 0.0) {
        return Err(format!("{}: the oracle gradient is all zero", case.name));
    }
    compare_to_loop(case.name, &analytic, &expected)?;
    compare_to_central_difference(case, &analytic, &operand, &weight)
}

#[proxima::test]
async fn adjoint_affine_map_kv_head_merge_equals_f64_loop() {
    check_case(&kv_head_merge(2, 4, 3)).expect("kv head merge");
}

#[proxima::test]
async fn adjoint_affine_map_rope_even_equals_f64_loop() {
    check_case(&rope_interleave(2, 3, 6, 0)).expect("rope even");
}

#[proxima::test]
async fn adjoint_affine_map_rope_odd_equals_f64_loop() {
    check_case(&rope_interleave(2, 3, 6, 1)).expect("rope odd");
}

#[proxima::test]
async fn adjoint_affine_map_three_term_not_in_granite_equals_f64_loop() {
    check_case(&three_term_with_collisions()).expect("three term map with collisions");
}

#[proxima::test]
async fn adjoint_affine_map_control_zeroed_scatter_fails_the_oracle() {
    let case = three_term_with_collisions();
    let operand = pseudo_random(7, element_count(&case.operand));
    let weight = pseudo_random(11, element_count(&case.iteration));
    let expected = f64_loop_gradient(&case, &operand, &weight);

    let zeroed = vec![0.0_f32; operand.len()];

    assert!(compare_to_loop(case.name, &zeroed, &expected).is_err());
    assert!(compare_to_central_difference(&case, &zeroed, &operand, &weight).is_err());
}

#[proxima::test]
#[case::kv_head_merge_16_query_heads_over_8_kv_heads(kv_head_merge(3, 8, 8))]
#[case::rope_even_16_heads(rope_interleave(3, 16, 8, 0))]
#[case::rope_odd_16_heads(rope_interleave(3, 16, 8, 1))]
async fn granite_attention_head_counts_match_the_f64_loop_and_central_difference(
    #[case] case: Case,
) {
    check_case(&case).expect("granite head counts");
}

#[proxima::test]
async fn the_rope_even_and_odd_maps_each_write_only_their_own_parity() {
    for parity in 0..2_i32 {
        let case = rope_interleave(2, 3, 6, parity);
        let operand = vec![1.0_f32; element_count(&case.operand)];
        let weight = vec![1.0_f32; element_count(&case.iteration)];

        let gradient = adjoint_gradient(&case, &operand, &weight);

        for (flat, value) in gradient.iter().enumerate() {
            let expected = if flat % 2 == parity as usize {
                2.0
            } else {
                0.0
            };
            assert_eq!(*value, expected, "{} operand element {flat}", case.name);
        }
    }
}

fn iota_count(program: &[Op]) -> usize {
    program
        .iter()
        .filter(|op| matches!(op, Op::Iota { .. }))
        .count()
}

#[proxima::test]
async fn a_pure_projection_keeps_the_direct_reduce_and_adds_no_mask() {
    let rank = 2_u16;
    let mut program = Vec::new();
    let operand = input(&mut program, "x", &[4, 3]);
    let weight = input(&mut program, "w", &[3, 4]);
    let transposed = IndexMap::Affine(map::projection(rank, &[1, 0]));
    let product = elementwise(
        &mut program,
        ScalarOp::Multiply,
        vec![(operand, transposed.clone()), (weight, identity(rank))],
    );
    let loss = sum_to_scalar(&mut program, product, rank);

    let differentiated = differentiate(&program, loss).expect("pure projection differentiates");

    assert_eq!(
        iota_count(&differentiated.program),
        0,
        "no mask nodes on the projection path"
    );
    let routed = differentiated.program[program.len()..]
        .iter()
        .filter_map(|op| match op {
            Op::Reduce(reduce) if reduce.out_map == transposed => Some(reduce),
            _ => None,
        })
        .count();
    assert_eq!(
        routed, 1,
        "the operand map itself is the backward Reduce out_map"
    );
}

#[proxima::test]
async fn a_multi_term_map_routes_through_one_mask_per_irregular_axis() {
    let case = kv_head_merge(2, 4, 3);
    let (program, loss) = squared_product_loss(&case);

    let differentiated = differentiate(&program, loss).expect("kv merge differentiates");

    assert_eq!(
        iota_count(&differentiated.program[program.len()..]),
        3,
        "one position Iota, one iteration-axis Iota per term (2) for the one irregular axis"
    );
}

#[proxima::test]
async fn a_gather_with_a_non_projection_index_map_is_still_refused() {
    let mut program = Vec::new();
    let table = input(&mut program, "table", &[8, 4]);
    let ids = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Int32,
            shape: vec![Extent::Static(6)],
            name: Some("ids".into()),
        },
    );
    let weight = input(&mut program, "w", &[3, 4]);
    let gathered = IndexMap::Computed {
        indices: ids,
        index_map: map::affine(2, &[(&[AxisTerm::scaled(0, 2)], 0)]),
        base: map::affine(2, &[(&[], 0), (&[AxisTerm::projection(1)], 0)]),
        gathered_dim: 0,
    };
    let product = elementwise(
        &mut program,
        ScalarOp::Multiply,
        vec![(table, gathered), (weight, identity(2))],
    );
    let loss = sum_to_scalar(&mut program, product, 2);

    let refused = differentiate(&program, loss)
        .err()
        .expect("strided index map is refused");

    assert!(
        matches!(refused, AutogradError::NonProjectionIndexMap { .. }),
        "got {refused:?}"
    );
}

#[proxima::test]
async fn a_reduce_over_a_multi_term_in_map_is_still_refused() {
    let mut program = Vec::new();
    let operand = input(&mut program, "x", &[6]);
    let mut tiled = map::affine(
        2,
        &[(&[AxisTerm::scaled(0, 2), AxisTerm::projection(1)], 0)],
    );
    tiled.axes[0].len = Some(Extent::Static(2));
    let windowed = op::append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand,
            in_map: IndexMap::Affine(tiled),
            out_map: IndexMap::Affine(map::projection(2, &[0])),
            keep: op::Keep::Reduce,
            name: None,
        }),
    );
    let loss = sum_to_scalar(&mut program, windowed, 1);

    let refused = differentiate(&program, loss)
        .err()
        .expect("windowed reduce is refused");

    assert!(
        matches!(refused, AutogradError::NonProjectionOperandMap { .. }),
        "got {refused:?}"
    );
}
