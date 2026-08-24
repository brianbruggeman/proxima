//! Isolates whether the panic in `offset_gradient_probe.rs` is in the
//! FORWARD anchored-offset read itself, or specific to the BACKWARD routed
//! reduce writing into a full-size destination via an offset out_map.
#![allow(clippy::unwrap_used, clippy::expect_used)]

extern crate alloc;

use proxima_tensor::cpu::evaluate_named;
use proxima_tensor::dtype::DType;
use proxima_tensor::map::{self, AxisTerm, IndexMap};
use proxima_tensor::op::{self, Extent, Op, ScalarOp};

fn leaf(program: &mut Vec<Op>, name: &str, shape: alloc::vec::Vec<Extent>) -> proxima_tensor::op::NodeId {
    op::append(program, Op::Input { dtype: DType::Float32, shape, name: Some(name.into()) })
}

#[proxima::test]
async fn forward_only_anchored_offset_read_evaluates_the_right_slice() {
    let mut program = Vec::new();
    let w = leaf(&mut program, "w", alloc::vec![Extent::Static(2), Extent::Static(6)]);
    let anchor = op::append(&mut program, Op::Constant { dtype: DType::Float32, shape: alloc::vec![Extent::Static(2)], value: 0.0 });

    let slice_map = IndexMap::Affine(map::affine(2, &[(&[AxisTerm::projection(0)], 0), (&[AxisTerm::projection(1)], 2)]));
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

    let w_values: alloc::vec::Vec<f32> = (0..12).map(|index| index as f32).collect();
    let evaluated = evaluate_named(&program, &[], &[("w", &w_values)], &[sliced]).expect("forward-only anchored slice evaluates");
    let (values, shape) = evaluated.get(sliced).expect("sliced requested");
    std::eprintln!("sliced forward shape={shape:?} values={values:?}");
    assert_eq!(shape, &[2u64, 2u64]);
    assert_eq!(values, &[2.0, 3.0, 8.0, 9.0], "row0 cols[2,4)=[2,3], row1 cols[2,4)=[8,9]");
}
