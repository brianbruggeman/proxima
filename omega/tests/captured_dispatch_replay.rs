//! The live-capture replay a kernel-variant harness drives: a dispatch captured
//! from a real plan run replays through its own pipeline, through a sibling's
//! pipeline (`with_pipeline_of`), and describes the buffers it is bound to.
//! `replay_output` poisons only the bytes the op writes (`output_total` f32
//! values), never the iteration-space-sized region a reduce's reduction axis
//! would otherwise extend over a pooled buffer's neighbours.

#![cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_gguf::quant::q4_0;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, append, projection,
};

const IN_DIM: usize = 1536;
const OUT_DIM: usize = 12;

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

fn matvec_program() -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::UInt8,
            shape: vec![Extent::Static(OUT_DIM as u32), Extent::Static(IN_DIM as u32)],
            name: None,
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1), Extent::Static(IN_DIM as u32)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(3, &[1, 2]))),
                (activation, IndexMap::Affine(projection(3, &[0, 2]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, sum)
}

fn capture_one_matvec() -> (Vec<omega::CapturedDispatch>, Vec<f32>) {
    let row_bytes = IN_DIM / q4_0::QK4_0 * q4_0::BLOCK_BYTES;
    let mut packed = vec![0u8; OUT_DIM * row_bytes];
    for (row, chunk) in packed.chunks_exact_mut(row_bytes).enumerate() {
        q4_0::quantize(&random_vec(7 + row as u64, IN_DIM), chunk).unwrap();
    }
    let activation = random_vec(99, IN_DIM);
    let (program, root) = matvec_program();
    let blocks = [
        QuantizedBlock::Packed { codec: omega::Codec::Q4_0, bytes: &packed },
        QuantizedBlock::Float32(&activation),
    ];
    let plan = omega::plan(&program, &[], &blocks, &[root], NumericPolicy::default())
        .expect("metal plans the matvec");
    temp_env::with_vars(
        [
            ("PROXIMA_CAPTURE_NODES", Some("all")),
            ("PROXIMA_CAPTURE_LIVE", Some("1")),
        ],
        || {
            let output = omega::execute_plan(&plan, &blocks).expect("metal runs the matvec");
            (omega::take_captured_dispatches(), output.root().to_vec())
        },
    )
}

#[test]
fn a_captured_matvec_replays_the_values_the_plan_wrote() {
    let (dispatches, written) = capture_one_matvec();
    let matvec = dispatches
        .iter()
        .find(|dispatch| dispatch.operands.iter().any(|(_, codec)| codec == "Q4_0"))
        .expect("the packed matvec was captured");

    let replayed = matvec.replay_output().expect("the captured dispatch replays");

    assert_eq!(
        replayed.len(),
        OUT_DIM * core::mem::size_of::<f32>(),
        "the poisoned and returned span is the output, not the iteration space"
    );
    let values: Vec<f32> = replayed
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    assert_eq!(values, written, "a replay writes what the plan wrote");
}

#[test]
fn a_dispatch_rebound_to_a_siblings_pipeline_writes_the_same_values() {
    let (dispatches, _) = capture_one_matvec();
    let matvec = dispatches
        .iter()
        .find(|dispatch| dispatch.operands.iter().any(|(_, codec)| codec == "Q4_0"))
        .expect("the packed matvec was captured");

    let rebound = matvec.with_pipeline_of(matvec);

    assert_eq!(
        rebound.replay_output().expect("rebound replays"),
        matvec.replay_output().expect("original replays"),
    );
}

#[test]
fn every_bound_buffer_is_described_once() {
    let (dispatches, _) = capture_one_matvec();
    let matvec = dispatches
        .iter()
        .find(|dispatch| dispatch.operands.iter().any(|(_, codec)| codec == "Q4_0"))
        .expect("the packed matvec was captured");

    let lines = matvec.describe_buffers();

    assert!(lines.len() >= 3, "weights, activation and output are bound: {lines:?}");
    assert!(
        lines.iter().all(|line| line.starts_with("binding=")),
        "{lines:?}"
    );
}
