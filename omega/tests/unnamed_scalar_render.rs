//! A Metal read of an unnamed scalar split (E3M0, E1M2) decodes its own bytes to what the host
//! `FormatCodec::scalar` decode gives, in one process, with no per-name arm anywhere (PROXNUM-118).
//! A split whose width neither divides nor fills a byte (E2M3, six bits) stays refused at plan time.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::mem::MaybeUninit;

use proxima_numeric::float_layout::FloatLayout;
use proxima_numeric::format_codec::FormatCodec;
use proxima_numeric::{DType, NumericFormat, NumericPolicy};
use proxima_tensor::cpu::QuantizedBlock;
use proxima_tensor::spec::input_leaf;
use proxima_tensor::{Extent, IndexMap, NodeId, Op, ScalarOp, append, map};

const REPEATS: usize = 8;

fn unnamed_scalar(exponent_bits: u8, mantissa_bits: u8) -> (FloatLayout, NumericFormat) {
    let layout = FloatLayout {
        exponent_bits,
        mantissa_bits,
        ..FloatLayout::E2M1
    };
    (layout, NumericFormat::Scalar(layout))
}

fn times_one_program(elements: usize) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let packed = input_leaf(&mut program, DType::FLOAT32, vec![Extent::Static(elements as u32)], "packed");
    let one = append(&mut program, Op::Constant { dtype: DType::FLOAT32, shape: Vec::new(), value: 1.0 });
    let output = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (packed, IndexMap::Affine(map::projection(1, &[0]))),
                (one, IndexMap::Affine(map::projection(1, &[]))),
            ],
            name: Some("unnamed_scalar_read".into()),
        },
    );
    (program, output)
}

fn every_code_repeated(width: u32) -> Vec<u32> {
    (0..REPEATS).flat_map(|_| 0..(1_u32 << width)).collect()
}

fn pack_low_bits_first(codes: &[u32], width: u32) -> Vec<u8> {
    let per_byte = (8 / width) as usize;
    codes
        .chunks(per_byte)
        .map(|group| group.iter().enumerate().fold(0_u8, |byte, (slot, code)| byte | ((*code as u8) << (slot as u32 * width))))
        .collect()
}

fn host_decode(layout: &FloatLayout, codes: &[u32]) -> Vec<f32> {
    let codec = FormatCodec::scalar(layout).expect("an unnamed scalar of at most 32 bits has a host codec");
    let one_code_per_byte: Vec<u8> = codes.iter().map(|code| *code as u8).collect();
    let mut decoded = vec![MaybeUninit::<f32>::uninit(); codes.len()];
    codec.decode(&one_code_per_byte, &mut decoded).expect("host decode of every code");
    decoded.iter().map(|slot| unsafe { slot.assume_init() }).collect()
}

fn metal_decode(format: NumericFormat, bytes: &[u8], elements: usize) -> Result<Vec<f32>, omega::MetalError> {
    let (program, output) = times_one_program(elements);
    omega::execute(
        &program,
        &[],
        &[QuantizedBlock::Packed { codec: format, bytes }],
        &[output],
        NumericPolicy::default(),
    )
    .map(|outputs| outputs.get(output).expect("Metal output exists").0.to_vec())
}

fn differing(host: &[f32], metal: &[f32]) -> usize {
    assert_eq!(host.len(), metal.len(), "degenerate gate: output length");
    host.iter()
        .zip(metal)
        .filter(|(want, got)| want.to_bits() != got.to_bits() && !(want.is_nan() && got.is_nan()))
        .count()
}

#[test]
fn e3m0_and_e1m2_read_decode_their_own_bytes_in_one_process() {
    let mut distinct_per_split = Vec::new();
    for (exponent_bits, mantissa_bits) in [(3, 0), (1, 2)] {
        let (layout, format) = unnamed_scalar(exponent_bits, mantissa_bits);
        let width = u32::from(layout.width());
        let codes = every_code_repeated(width);
        let host = host_decode(&layout, &codes);
        let metal = metal_decode(format, &pack_low_bits_first(&codes, width), codes.len())
            .expect("Metal reads an unnamed scalar whose width divides a byte");

        let first_sixteen: Vec<(u32, u32, u32)> = (0..16)
            .map(|code| (code as u32, host[code].to_bits(), metal[code].to_bits()))
            .collect();
        assert_eq!(
            differing(&host, &metal),
            0,
            "E{exponent_bits}M{mantissa_bits}: Metal differs from host decode; (code, host bits, metal bits) = {first_sixteen:x?}"
        );
        let mut distinct: Vec<u32> = host.iter().map(|value| value.to_bits()).collect();
        distinct.sort_unstable();
        distinct.dedup();
        distinct_per_split.push(distinct);
    }
    assert_ne!(
        distinct_per_split[0], distinct_per_split[1],
        "degenerate gate: the two splits decode to the same value set, so reuse of one kernel would pass"
    );
}

#[test]
fn control_wrong_split_fails_the_host_comparison() {
    let (e3m0_layout, _) = unnamed_scalar(3, 0);
    let (_, e1m2_format) = unnamed_scalar(1, 2);
    let codes = every_code_repeated(4);
    let host = host_decode(&e3m0_layout, &codes);
    let metal = metal_decode(e1m2_format, &pack_low_bits_first(&codes, 4), codes.len()).expect("E1M2 reads");

    assert!(differing(&host, &metal) > 0, "E1M2 bytes matched E3M0 host decode; the comparison cannot discriminate");
}

#[test]
fn e2m3_six_bit_split_is_refused_at_plan_time() {
    let (_, format) = unnamed_scalar(2, 3);
    let refused = metal_decode(format, &[0_u8; 6], 8);

    assert!(refused.is_err(), "a width that neither divides nor fills a byte has no flat address");
}
