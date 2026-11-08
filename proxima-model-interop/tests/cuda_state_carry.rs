#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::string::String;

use omega::CudaDriver;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::{Extent, NodeId, Op, QuantizedBlock, ScalarOp};
use proxima_numeric::{DType, NumericPolicy};

const STATE: NodeId = NodeId(0);
const NEXT_STATE: NodeId = NodeId(2);
const WIDTH: usize = 786_432;

fn accumulate_program() -> Vec<Op> {
    let input = |name: &str| Op::Input {
        dtype: DType::FLOAT32,
        shape: vec![Extent::Symbolic(0)],
        name: Some(String::from(name)),
    };
    vec![
        input("state"),
        input("delta"),
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            operands: vec![
                (STATE, IndexMap::Affine(map::projection(1, &[0]))),
                (NodeId(1), IndexMap::Affine(map::projection(1, &[0]))),
            ],
            name: Some(String::from("next_state")),
        },
    ]
}

fn delta_values() -> Vec<f32> {
    (0..WIDTH).map(|index| (index % 97) as f32 * 0.25).collect()
}

#[proxima::test]
#[case::three_steps(3)]
#[case::one_step(1)]
async fn carried_state_matches_round_trip_and_skips_host_copies(#[case] steps: usize) {
    let driver = CudaDriver::new(0).expect("a CUDA device at ordinal 0 is required");
    let program = accumulate_program();
    let delta = delta_values();
    let zeros = vec![0.0_f32; WIDTH];
    let planning_blocks = [
        ("state", QuantizedBlock::Float32(&zeros)),
        ("delta", QuantizedBlock::Float32(&delta)),
    ];
    let mut round_trip_plan = driver
        .plan(&program, &[WIDTH as u64], &[NEXT_STATE], NumericPolicy::default(), &planning_blocks)
        .expect("round trip plan should build");
    let mut host_state = zeros.clone();
    for _ in 0..steps {
        let blocks = [
            ("state", QuantizedBlock::Float32(&host_state)),
            ("delta", QuantizedBlock::Float32(&delta)),
        ];
        let evaluated = round_trip_plan
            .execute_named(&blocks, None)
            .expect("round trip step should execute");
        host_state = evaluated.get(NEXT_STATE).expect("round trip returns the state").0.to_vec();
    }

    let mut carry_plan = driver
        .plan(&program, &[WIDTH as u64], &[NEXT_STATE], NumericPolicy::default(), &planning_blocks)
        .expect("carry plan should build");
    assert!(carry_plan.carry_state(&[(STATE, NEXT_STATE)]), "pair is swappable");
    for step in 0..steps {
        let blocks = [
            ("state", QuantizedBlock::Float32(&zeros)),
            ("delta", QuantizedBlock::Float32(&delta)),
        ];
        let evaluated = carry_plan
            .execute_named(&blocks, None)
            .expect("carried step should execute");
        assert!(evaluated.get(NEXT_STATE).is_none(), "step {step}: carried state is not read back");
    }
    let state_bytes = (WIDTH * std::mem::size_of::<f32>()) as u64;
    assert_eq!(carry_plan.carried_host_bytes(), state_bytes, "only the first upload crosses the host inside the loop");

    let drained = carry_plan.drain_carried().expect("drain should read the device state");
    assert_eq!(carry_plan.carried_host_bytes(), 2 * state_bytes, "flush adds exactly one readback");
    assert_eq!(drained.len(), 1);
    assert_eq!(drained[0].0, NEXT_STATE);
    assert_eq!(drained[0].1, host_state, "flushed state equals the round trip state");
}
