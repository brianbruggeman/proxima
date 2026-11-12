//! An output node registered with `Plan::set_encoded_outputs` lands in the caller's buffer already
//! in the node's numeric format: the producing op writes binary32, a dispatch rendered from the
//! format's description encodes it in the same command buffer, and no host codec runs.

#![cfg(all(feature = "metal-output-placement", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use omega::DispatchType;
use proxima_numeric::{DType, FormatCodec, NumericFormat, NumericPolicy};
use proxima_tensor::{Extent, IndexMap, Op, QuantizedBlock, ScalarOp, append, projection};

const WIDTH: usize = 256;
const ROWS_BEFORE: usize = 3;

fn identity_program() -> (Vec<Op>, proxima_tensor::NodeId) {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input { dtype: DType::FLOAT32, shape: vec![Extent::Static(WIDTH as u32)], name: None },
    );
    let identity = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Identity,
            operands: vec![(source, IndexMap::Affine(projection(1, &[0])))],
            name: None,
        },
    );
    (program, identity)
}

fn attention_row() -> Vec<f32> {
    (0..WIDTH)
        .map(|index| {
            let phase = index as f32 * 0.37;
            (phase.sin() * 3.0 + (phase * 0.11).cos()) * if index % 61 == 7 { 12.0 } else { 0.5 }
        })
        .collect()
}

fn stored_rows_equal_the_host_codec(format: NumericFormat, dispatch: DispatchType) {
    let (program, node) = identity_program();
    let row = attention_row();
    let mut plan = omega::plan(&program, &[], &[QuantizedBlock::Float32(&row)], &[], NumericPolicy::default())
        .expect("plans the identity program");
    plan.set_dispatch_type(dispatch);
    plan.set_encoded_outputs([(node, format)]).expect("the format renders");
    let row_bytes = format.packed_len(WIDTH).expect("whole blocks");
    let buffer = omega::allocate_placed_buffer(row_bytes * (ROWS_BEFORE + 2)).expect("allocates the cache");
    omega::zero_placed_buffer(&buffer, row_bytes * (ROWS_BEFORE + 2)).expect("zeroes the cache");
    omega::execute_plan_with_placements(
        &plan,
        &[QuantizedBlock::Float32(&row)],
        &[],
        &[(node, &buffer, ROWS_BEFORE * row_bytes)],
        &mut Vec::new(),
    )
    .expect("executes with the encoded output");
    let codec = FormatCodec::for_format(&format).expect("host codec");
    let mut host = vec![0u8; row_bytes];
    codec.encode(&row, &mut host).expect("host encodes");
    let mut expected = vec![0.0f32; WIDTH];
    let mut spare: Vec<core::mem::MaybeUninit<f32>> = vec![core::mem::MaybeUninit::uninit(); WIDTH];
    codec.decode(&host, &mut spare).expect("host decodes");
    for (slot, value) in spare.iter().zip(expected.iter_mut()) {
        *value = unsafe { slot.assume_init() };
    }
    let stored = omega::read_placed_buffer(&buffer, ROWS_BEFORE * row_bytes, &codec, WIDTH).expect("reads the row");
    assert_eq!(stored, expected, "{format:?} {dispatch:?}: the stored row decodes to the host encode");
    let before = omega::read_placed_buffer(&buffer, 0, &codec, WIDTH).expect("reads the untouched row");
    assert!(before.iter().all(|value| *value == 0.0), "{format:?}: rows before the offset are untouched");
}

#[test]
fn an_encoded_output_lands_in_the_cache_format_for_every_dispatch_type() {
    for format in [NumericFormat::Float16, NumericFormat::BFloat16, NumericFormat::Q8_0, NumericFormat::Q4_0] {
        for dispatch in [DispatchType::Serial, DispatchType::Concurrent] {
            stored_rows_equal_the_host_codec(format, dispatch);
        }
    }
}

#[test]
fn a_format_with_no_rendered_encode_is_refused_at_registration() {
    let (program, node) = identity_program();
    let row = attention_row();
    let mut plan = omega::plan(&program, &[], &[QuantizedBlock::Float32(&row)], &[], NumericPolicy::default())
        .expect("plans");
    assert!(plan.set_encoded_outputs([(node, NumericFormat::Q4_1)]).is_err());
}
