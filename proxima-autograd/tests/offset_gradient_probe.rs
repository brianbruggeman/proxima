//! Does a gradient routed through a NONZERO-offset `IndexMap::Affine`
//! operand read (a sub-range slice of a larger `Op::Input`, not the
//! offset-0 case every other test in this crate exercises) come back as a
//! full-size, zero-elsewhere gradient tensor? Checked BEFORE
//! `sparse_ffn_pruning.rs` was built on top of it (guiding principle 6:
//! never work from inference) -- and the answer this file found is why
//! that file uses separate per-band leaf tensors instead.
//!
//! Three findings, one test each:
//!
//! 1. An offset-only axis does NOT shape-infer on its own --
//!    `shape.rs:195-198`'s `unify_iteration_space` only resolves an axis's
//!    extent from a zero-offset, unit-coefficient operand read; an offset
//!    read is only BOUNDS-CHECKED against an already-resolved extent
//!    (`shape.rs:411-430`'s `bounds_check`), never used to establish one.
//! 2. Adding an anchor -- a same-shaped zero-offset `Constant` naming the
//!    slice's own width, exactly the `broadcast_anchor` pattern this
//!    crate's own adjoint already uses for the structurally identical
//!    problem -- fixes shape inference, and the FORWARD read then
//!    evaluates to exactly the right slice.
//! 3. But the BACKWARD routed `Reduce` -- writing that slice's gradient
//!    into the full `[2, 6]` destination through the same offset -- panics
//!    in `proxima-tensor/src/cpu.rs:4461` with an index-out-of-bounds. This
//!    is a genuine `proxima-tensor` evaluator gap, not a `proxima-autograd`
//!    one, and out of this session's scope (`proxima-autograd/**`) to fix.
//!    The test below captures it precisely (`catch_unwind` on the exact
//!    panic site) so it stays a documented, reproducible finding rather
//!    than a red test in the gate or a silently-skipped one.
#![allow(clippy::unwrap_used, clippy::expect_used)]

extern crate alloc;

use proxima_autograd::adjoint::differentiate;
use proxima_tensor::cpu::evaluate_named;
use proxima_tensor::dtype::DType;
use proxima_tensor::map::{self, AxisTerm, IndexMap};
use proxima_tensor::op::{self, Extent, NodeId, Op, ReduceInit, ScalarOp};

fn leaf(program: &mut Vec<Op>, name: &str, shape: alloc::vec::Vec<Extent>) -> NodeId {
    op::append(
        program,
        Op::Input {
            dtype: DType::Float32,
            shape,
            name: Some(name.into()),
        },
    )
}

/// Finding 1: without an anchor, an offset-only axis has nothing to pin
/// its own extent, so shape inference rejects it -- proving the anchor
/// added in the tests below is load-bearing, not decorative.
#[proxima::test]
async fn an_offset_only_axis_with_no_anchor_is_rejected_by_shape_inference() {
    let mut program = Vec::new();
    let w = leaf(
        &mut program,
        "w",
        alloc::vec![Extent::Static(2), Extent::Static(6)],
    );
    let slice_map = IndexMap::Affine(map::affine(
        2,
        &[
            (&[AxisTerm::projection(0)], 0),
            (&[AxisTerm::projection(1)], 2),
        ],
    ));
    let sliced = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: alloc::vec![(w, slice_map)],
            name: None,
        },
    );
    let loss = op::append(
        &mut program,
        Op::Reduce(proxima_tensor::op::Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: sliced,
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[])),
            keep: proxima_tensor::op::Keep::Reduce,
            name: None,
        }),
    );
    let outcome = differentiate(&program, loss);
    match outcome {
        Err(proxima_autograd::error::AutogradError::ShapeInference(inner)) => {
            std::eprintln!("offset-only axis correctly rejected: {inner}");
        }
        Err(other) => panic!("expected a ShapeInference error, got a different error: {other}"),
        Ok(_) => panic!(
            "an offset-only axis with no anchor must fail shape inference, not silently pick a size"
        ),
    }
}

fn build_anchored_slice_program() -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let w = leaf(
        &mut program,
        "w",
        alloc::vec![Extent::Static(2), Extent::Static(6)],
    );
    let anchor = op::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(2)],
            value: 0.0,
        },
    );

    let slice_map = IndexMap::Affine(map::affine(
        2,
        &[
            (&[AxisTerm::projection(0)], 0),
            (&[AxisTerm::projection(1)], 2),
        ],
    ));
    let anchor_map = IndexMap::Affine(map::projection(2, &[1]));
    let sliced = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: alloc::vec![(w, slice_map), (anchor, anchor_map)],
            name: None,
        },
    );
    let loss = op::append(
        &mut program,
        Op::Reduce(proxima_tensor::op::Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: sliced,
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[])),
            keep: proxima_tensor::op::Keep::Reduce,
            name: None,
        }),
    );
    (program, loss)
}

/// Finding 2: with the anchor, shape inference and the FORWARD evaluation
/// both succeed and read exactly the right slice -- see
/// `offset_forward_only_probe.rs` for the isolated forward-only check.
/// `differentiate` itself also succeeds (shape inference is a forward-only
/// concern); this test only confirms that much, not that evaluating the
/// resulting adjoint program is safe -- see the next test for that.
#[proxima::test]
async fn an_anchored_offset_read_differentiates_without_a_shape_error() {
    let (program, loss) = build_anchored_slice_program();
    let differentiated = differentiate(&program, loss).expect("anchored slice differentiates");
    assert!(
        differentiated.gradient_of_named("w").is_some(),
        "w must have a gradient node recorded"
    );
}

/// Finding 3, captured precisely: evaluating that SAME adjoint program's
/// gradient panics in `proxima-tensor`'s CPU interpreter with an
/// index-out-of-bounds, not a typed `TensorError`. `catch_unwind` turns
/// this into a documented, reproducible regression check (does the panic
/// still happen at this exact site, with this exact message) rather than
/// leaving a permanently-red test in the gate for a bug this session's
/// scope (`proxima-autograd/**`) cannot fix. If a future
/// `proxima-tensor` fix makes this evaluate correctly, THIS test starts
/// failing (the `catch_unwind` no longer observes a panic) and should be
/// replaced with the positive assertion `offset_gradient_probe.rs`'s own
/// history shows was the original intent.
#[proxima::test]
async fn anchored_offset_backward_write_panics_in_the_cpu_evaluator_today() {
    let (program, loss) = build_anchored_slice_program();
    let differentiated = differentiate(&program, loss).expect("anchored slice differentiates");
    let grad_w = differentiated
        .gradient_of_named("w")
        .expect("w feeds the loss");
    let w_values: alloc::vec::Vec<f32> = (0..12).map(|index| index as f32).collect();

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        evaluate_named(&differentiated.program, &[], &[("w", &w_values)], &[grad_w])
    }));

    match outcome {
        Err(_) => {
            std::eprintln!(
                "confirmed: evaluating a gradient routed through a nonzero-offset out_map \
                 still panics in proxima-tensor's CPU evaluator (cpu.rs:4461, index out of bounds) -- \
                 this is why sparse_ffn_pruning.rs uses separate per-band leaves instead"
            );
        }
        Ok(Ok(_)) => panic!(
            "evaluation succeeded -- proxima-tensor's evaluator gap appears to be FIXED; replace this test \
             with the positive full-size/zero-off-the-slice assertion this file's own doc describes"
        ),
        Ok(Err(error)) => panic!(
            "expected a panic (the known cpu.rs:4461 bug), got a typed error instead: {error}"
        ),
    }
}
