#![cfg(all(feature = "alloc-count", feature = "metal", target_os = "macos"))]
#![allow(clippy::expect_used)]

use std::collections::BTreeMap;

use proxima_tensor::{
    DType, Extent, IndexMap, Keep, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp, append,
    projection,
};
use proxima_test::alloc_count::{CountingAllocator, recorded_sizes, reset};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn a_named_placed_input_does_not_allocate_a_tensor_sized_placeholder() {
    const STATE_ELEMENTS: usize = 128 * 128 * 16 * 2;
    const STATE_BYTES: usize = STATE_ELEMENTS * size_of::<f32>();

    let mut program = Vec::new();
    let state_input = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(STATE_ELEMENTS as u32)],
            name: Some(String::from("state")),
        },
    );
    let state_sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: state_input,
            in_map: IndexMap::Affine(projection(1, &[0])),
            out_map: IndexMap::Affine(projection(1, &[])),
            keep: Keep::Reduce,
            name: Some(String::from("state_sum")),
        }),
    );
    let state = vec![0.0_f32; STATE_ELEMENTS];
    let plan = omega::plan_named(
        &program,
        &[],
        &[("state", QuantizedBlock::Float32(&state))],
        &[state_sum],
        proxima_tensor::NumericPolicy::default(),
    )
    .expect("plans a production-sized recurrent-state reduction");
    let state_buffer = omega::allocate_placed_buffer(STATE_BYTES)
        .expect("allocates the recurrent-state placement");

    omega::execute_plan_named_with_placements_and_expert_sources(
        &plan,
        &[],
        &[(state_input, &state_buffer, 0)],
        &[],
        &BTreeMap::new(),
    )
    .expect("cold execution resolves the placed-input plan");

    reset();
    omega::execute_plan_named_with_placements_and_expert_sources(
        &plan,
        &[],
        &[(state_input, &state_buffer, 0)],
        &[],
        &BTreeMap::new(),
    )
    .expect("warm execution accepts the placed input without host data");
    let warm_sizes = recorded_sizes();

    assert!(
        !warm_sizes.contains(&STATE_BYTES),
        "the warm named-placement path must not allocate the placed input's {STATE_BYTES}-byte \
         host mirror; recorded sizes: {warm_sizes:?}"
    );
}
