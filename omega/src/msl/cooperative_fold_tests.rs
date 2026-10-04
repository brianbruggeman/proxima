use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use proxima_tensor::{
    DType, Extent, IndexMap, Keep, Op, Reduce, ReduceInit, ScalarOp, append, bind, infer, map,
};

use super::*;

fn sum_of_squares_op(width: u32) -> BoundOp {
    let mut program = Vec::new();
    let row = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1), Extent::Static(width)],
            name: None,
        },
    );
    let squared = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (row, IndexMap::Affine(map::projection(2, &[0, 1]))),
                (row, IndexMap::Affine(map::projection(2, &[0, 1]))),
            ],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: squared,
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[0])),
            keep: Keep::Reduce,
            name: Some("sum_of_squares".into()),
        }),
    );
    let root = NodeId((program.len() - 1) as u32);
    let shapes = infer(&program, &[]).expect("sum of squares infers");
    bind(&program, &shapes, &[root], NumericPolicy::default())
        .expect("sum of squares lowers")
        .into_iter()
        .next()
        .expect("one fused bound emitted")
}

#[test]
fn cooperative_fold_issues_every_slot_load_before_the_first_fold() {
    let source = emit(
        &sum_of_squares_op(1536),
        &BTreeMap::new(),
        NumericPolicy::llama_relaxed(),
    )
    .expect("hidden-width sum of squares emits")
    .source;
    let unroll = crate::sized::COOPERATIVE_REDUCE_UNROLL;
    let load_loop = source
        .find(&format!("float batch[{unroll}]["))
        .expect("loads land in a batch array");
    let fold = source
        .find("accumulator = seeded ? (accumulator + value) : value;")
        .expect("the fold still runs per slot");
    assert!(
        load_loop < fold,
        "every slot is loaded before slot 0 is folded"
    );
    assert!(source.contains("batch[slot][0] = in_range ? in0[walk0 + slot * advance0]"));
    assert!(source.contains(&format!("walk0 += {unroll} * advance0;")));
}

#[test]
fn cooperative_fold_keeps_slot_order_so_the_sum_is_the_serial_sum() {
    let source = emit(
        &sum_of_squares_op(1536),
        &BTreeMap::new(),
        NumericPolicy::llama_relaxed(),
    )
    .expect("hidden-width sum of squares emits")
    .source;
    let fold_loop = source
        .find("if ((r + slot * 256) < total_r) {")
        .expect("the fold loop walks slots under the same range guard as the loads");
    assert!(
        source[fold_loop..].contains("scratch[0] = batch[slot][0];"),
        "each slot's operands come from its own batch row"
    );
    assert!(
        !source.contains("batch[slot + 1]"),
        "no slot is folded out of order"
    );
}

#[test]
fn bit_exact_policy_keeps_the_serial_fold_and_never_batches() {
    let source = emit(
        &sum_of_squares_op(1536),
        &BTreeMap::new(),
        NumericPolicy::bit_exact(),
    )
    .expect("hidden-width sum of squares emits")
    .source;
    assert!(
        !source.contains("batch["),
        "no cooperative fold means no batched loads"
    );
    assert!(source.contains("for (long r = 0; r < u.reduction_total; r++)"));
}
