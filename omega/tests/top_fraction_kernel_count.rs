#![cfg(feature = "top-fraction-fusion")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use omega::sized::SELECTION_TOP_FRACTION_MIN_ROWS;
use omega::{Binding, PackedOperands, emit};
use proxima_tensor::spec::{input_leaf, top_fraction_mask};
use proxima_tensor::{
    BoundOp, DType, Extent, NumericPolicy, bind_with_fusion, bind_with_top_fraction, infer,
};

fn bound_lists(rows: u64) -> (Vec<BoundOp>, Vec<BoundOp>) {
    let mut program = Vec::new();
    let scores = input_leaf(&mut program, DType::Float32, vec![Extent::Symbolic(0)], "scores");
    let keep_count = input_leaf(&mut program, DType::Float32, Vec::new(), "keep_count");
    let mask = top_fraction_mask(&mut program, scores, keep_count, None).expect("mask lowers");
    let shapes = infer(&program, &[rows]).expect("program infers");
    let policy = NumericPolicy::default();
    let plain = bind_with_fusion(&program, &shapes, &[mask], true, policy).expect("plain binds");
    let fused = bind_with_top_fraction(
        &program,
        &shapes,
        &[mask],
        true,
        policy,
        SELECTION_TOP_FRACTION_MIN_ROWS,
    )
    .expect("fused binds");
    (plain, fused)
}

fn selection_kernels(bound: &[BoundOp]) -> Vec<omega::Kernel> {
    bound
        .iter()
        .map(|bound_op| {
            emit(bound_op, &PackedOperands::new(), NumericPolicy::default()).expect("bound op emits")
        })
        .filter(|kernel| kernel.entry.starts_with("omega_top_fraction_select_"))
        .collect()
}

#[test]
fn no_selection_kernel_below_the_configured_row_threshold() {
    let (plain, fused) = bound_lists(SELECTION_TOP_FRACTION_MIN_ROWS - 1);
    assert_eq!(selection_kernels(&fused).len(), 0);
    let kinds = |bound: &[BoundOp]| {
        bound
            .iter()
            .map(|bound_op| bound_op.kind.name())
            .collect::<Vec<_>>()
    };
    assert_eq!(kinds(&fused), kinds(&plain));
}

#[test]
fn one_selection_kernel_at_the_configured_row_threshold() {
    let rows = SELECTION_TOP_FRACTION_MIN_ROWS;
    let (plain, fused) = bound_lists(rows);
    let kernels = selection_kernels(&fused);
    assert_eq!(kernels.len(), 1);
    assert_eq!(kernels[0].entry, format!("omega_top_fraction_select_r{rows}_k0"));
    assert_eq!(kernels[0].grid.threads, 1024);
    assert_eq!(kernels[0].bindings.len(), 4);
    assert!(matches!(kernels[0].bindings[2], Binding::Output(_)));
    assert!(fused.len() < plain.len());
}
