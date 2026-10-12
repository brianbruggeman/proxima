#![cfg(all(feature = "metal", feature = "float-layout-dtype", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

//! The packed matvec arms (`omega/src/msl/packed_row_blocked_ggml.rs`) against an f64 dot-product
//! loop on real granite attention rows, and `Codec::layout()` against the layout constants.
//!
//! The weights are the granite 1024x512 attn_q rows (Q8_0 dequantized) re-encoded into each
//! float codec by `proxima_gguf::quant`. The reference decodes the packed bytes through the CPU
//! codec (itself checked against independent references by `float_layout_codec_*`) and
//! accumulates in f64; the bound is the worst-case f32 dot rounding bound `K u / (1 - K u) *
//! sum(|term|)` with `u = 2^-24`, which holds for any association of the sum.

#[path = "float_layout_parity/rows.rs"]
mod rows;

use proxima_gguf::quant::{bf4_e2m1, bf8_e5m2, bf16, scaled_float};
use proxima_primitives::Codec;
use proxima_primitives::float_layout::FloatLayout;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, append, projection,
};

const ACTIVATION_SEED: u64 = 0xac71_f10a;

struct Packed {
    bytes: Vec<u8>,
    decoded: Vec<f64>,
}

struct Reference {
    sum: f64,
    magnitude: f64,
}

fn pack(codec: Codec, values: &[f32]) -> Packed {
    match codec {
        Codec::BFloat16 => pack_bf16(values),
        Codec::Bf8E5M2 => pack_bf8(values),
        Codec::Bf4E2M1 => pack_bf4(values),
        Codec::Fp6E3M2 => pack_fp6(values),
        other => panic!("{other:?} is not a float codec of this test"),
    }
}

fn pack_bf16(values: &[f32]) -> Packed {
    let mut bytes = vec![0u8; values.len() * bf16::BLOCK_BYTES];
    bf16::quantize(values, &mut bytes).expect("bf16 quantizes finite weights");
    let mut decoded = vec![0.0f32; values.len()];
    bf16::dequantize(&bytes, &mut decoded).expect("bf16 dequantizes its own bytes");
    Packed {
        bytes,
        decoded: decoded.into_iter().map(f64::from).collect(),
    }
}

fn pack_bf8(values: &[f32]) -> Packed {
    let bytes: Vec<u8> = values
        .iter()
        .map(|&value| bf8_e5m2::encode(value))
        .collect();
    let decoded = bytes
        .iter()
        .map(|&byte| f64::from(bf8_e5m2::decode(byte)))
        .collect();
    Packed { bytes, decoded }
}

fn pack_bf4(values: &[f32]) -> Packed {
    let bytes: Vec<u8> = values
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| bf4_e2m1::pack_pair(pair[0], pair[1]).expect("finite weights pack"))
        .collect();
    let decoded = bytes
        .iter()
        .flat_map(|&byte| bf4_e2m1::unpack_pair(byte))
        .map(f64::from)
        .collect();
    Packed { bytes, decoded }
}

fn pack_fp6(values: &[f32]) -> Packed {
    let mut bytes =
        vec![0u8; values.len() / scaled_float::BLOCK_ELEMENTS * scaled_float::BLOCK_BYTES];
    let mut decoded = Vec::with_capacity(values.len());
    for (source, block) in values
        .as_chunks::<{ scaled_float::BLOCK_ELEMENTS }>()
        .0
        .iter()
        .zip(bytes.as_chunks_mut::<{ scaled_float::BLOCK_BYTES }>().0)
    {
        scaled_float::quantize_block(source, FloatLayout::E3M2, block).expect("finite weights");
        let mut elements = [0.0f32; scaled_float::BLOCK_ELEMENTS];
        scaled_float::dequantize_block(block, FloatLayout::E3M2, &mut elements)
            .expect("an e3m2 block dequantizes");
        decoded.extend(elements.into_iter().map(f64::from));
    }
    Packed { bytes, decoded }
}

fn references(weights: &[f64], activation: &[f32]) -> Vec<Reference> {
    let (row_chunks, _) = weights.as_chunks::<{ rows::COLUMNS }>();
    row_chunks
        .iter()
        .map(|row| {
            row.iter().zip(activation).fold(
                Reference {
                    sum: 0.0,
                    magnitude: 0.0,
                },
                |reference, (&weight, &value)| {
                    let term = weight * f64::from(value);
                    Reference {
                        sum: reference.sum + term,
                        magnitude: reference.magnitude + term.abs(),
                    }
                },
            )
        })
        .collect()
}

fn matvec_program() -> (Vec<Op>, NodeId) {
    let (depth, rows) = (rows::COLUMNS as u32, rows::ROWS as u32);
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::UInt8,
            shape: vec![Extent::Static(rows), Extent::Static(depth)],
            name: None,
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1), Extent::Static(depth)],
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

fn metal_matvec(codec: Codec, bytes: &[u8], activation: &[f32]) -> Vec<f32> {
    let (program, sum) = matvec_program();
    let blocks = [
        QuantizedBlock::Packed { codec, bytes },
        QuantizedBlock::Float32(activation),
    ];
    let plan = omega::plan(
        &program,
        &[],
        &blocks,
        &[sum],
        NumericPolicy::llama_relaxed(),
    )
    .expect("metal plans a packed matvec");
    omega::execute_plan(&plan, &blocks)
        .expect("metal runs the packed matvec on a real device")
        .root()
        .to_vec()
}

fn dot_rounding_bound() -> f64 {
    let accumulated = rows::COLUMNS as f64 * 2f64.powi(-24);
    accumulated / (1.0 - accumulated)
}

fn worst_ratio(observed: &[f32], reference: &[Reference]) -> (usize, f64) {
    assert_eq!(
        observed.len(),
        reference.len(),
        "every output row is compared"
    );
    observed
        .iter()
        .zip(reference)
        .map(|(&value, expected)| (f64::from(value) - expected.sum).abs() / expected.magnitude)
        .enumerate()
        .fold((0, 0.0), |worst, (index, ratio)| {
            if ratio > worst.1 {
                (index, ratio)
            } else {
                worst
            }
        })
}

fn activation() -> Vec<f32> {
    let mut lcg = Lcg(ACTIVATION_SEED);
    (0..rows::COLUMNS).map(|_| lcg.next_unit()).collect()
}

#[proxima::test]
#[case::brain16(Codec::BFloat16)]
#[case::bf8_e5m2(Codec::Bf8E5M2)]
#[case::bf4_e2m1(Codec::Bf4E2M1)]
#[case::fp6_e3m2(Codec::Fp6E3M2)]
async fn float_layout_packed_matvec_equals_f64_loop(#[case] codec: Codec) {
    let packed = pack(codec, &rows::granite_attention_rows());
    let activation = activation();
    let reference = references(&packed.decoded, &activation);

    let observed = metal_matvec(codec, &packed.bytes, &activation);

    let (row, ratio) = worst_ratio(&observed, &reference);
    println!(
        "packed {codec:?} compared={} worst_ratio={ratio:e} row={row} bound={:e}",
        observed.len(),
        dot_rounding_bound()
    );
    assert!(
        ratio <= dot_rounding_bound(),
        "{codec:?} row {row}: {ratio:e} exceeds the bound"
    );
}

#[test]
fn float_layout_packed_codec_layout_table_matches_constants() {
    let expected = [
        (Codec::Float16, FloatLayout::F16),
        (Codec::BFloat16, FloatLayout::BF16),
        (Codec::BFloat8, FloatLayout::E5M2),
        (Codec::Bf8E5M2, FloatLayout::E5M2_SATURATING),
        (Codec::Bf4E2M1, FloatLayout::E2M1),
        (Codec::Fp6E3M2, FloatLayout::E3M2),
    ];

    for (codec, layout) in expected {
        assert_eq!(codec.layout(), Some(layout), "{codec:?}");
    }
    let with_layout = (0..=u8::MAX)
        .filter_map(Codec::from_tag)
        .filter(|codec| codec.layout().is_some())
        .count();
    assert_eq!(
        with_layout,
        expected.len(),
        "no other codec names a float layout"
    );
}

#[test]
fn float_layout_packed_control_known_good_matvec_passes_bound() {
    let packed = pack(Codec::BFloat16, &rows::granite_attention_rows());
    let activation = activation();
    let reference = references(&packed.decoded, &activation);
    let known_good: Vec<f32> = reference
        .iter()
        .map(|expected| expected.sum as f32)
        .collect();

    let (_, ratio) = worst_ratio(&known_good, &reference);

    assert!(ratio <= dot_rounding_bound(), "{ratio:e}");
}

#[test]
fn float_layout_packed_control_perturbed_weight_fails_bound() {
    let packed = pack(Codec::BFloat16, &rows::granite_attention_rows());
    let activation = activation();
    let reference = references(&packed.decoded, &activation);
    let mut perturbed = packed.bytes.clone();
    perturbed[..2].copy_from_slice(&[0x00, 0x42]);

    let observed = metal_matvec(Codec::BFloat16, &perturbed, &activation);

    let (row, ratio) = worst_ratio(&observed, &reference);
    assert_eq!(row, 0, "the failure lands on the perturbed row");
    assert!(ratio > dot_rounding_bound(), "{ratio:e}");
}
