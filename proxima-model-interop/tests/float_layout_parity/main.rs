#![cfg(all(feature = "metal", feature = "float-layout-dtype", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Parity of the one MSL emitter (`omega::msl::layout_render`) against the CPU codec
//! (`proxima_gguf::quant::float_layout`) on real granite attention rows, one probe kernel per
//! check. The CPU codec is the oracle here only because `float_layout_codec_*` checks it against
//! independent f64 and `half` references.

mod compare;
mod gpu;
mod probes;
mod rows;

use std::sync::OnceLock;

use proxima_gguf::quant::float_layout::{decode, encode};
use proxima_primitives::float_layout::{FloatLayout, SignEncoding, Specials};

use compare::{compare_bits, compare_codes};
use gpu::{Gpu, floats, read_words, words};
use probes::{narrow, widen};

const E4M3_FROM_FIELDS: FloatLayout = FloatLayout {
    exponent_bits: 4,
    mantissa_bits: 3,
    specials: Specials::None,
    bias_override: None,
    sign: SignEncoding::Magnitude,
};

const MATMUL_ROWS: usize = 64;
const MATMUL_REDUCTION: usize = 512;
const MATMUL_COLUMNS: usize = 32;

fn granite() -> &'static [f32] {
    static ROWS: OnceLock<Vec<f32>> = OnceLock::new();
    ROWS.get_or_init(rows::granite_attention_rows)
}

fn gpu_codes(gpu: &Gpu, source: &str, inputs: &[&[u8]], count: usize) -> Vec<u32> {
    read_words(&gpu.run(source, "probe", inputs, count * 4, count))
}

fn gpu_decode(gpu: &Gpu, layout: FloatLayout, codes: &[u32]) -> Vec<f32> {
    let bytes = gpu.run(
        &probes::decode_source(layout),
        "probe",
        &[&words(codes)],
        codes.len() * 4,
        codes.len(),
    );
    read_words(&bytes).into_iter().map(f32::from_bits).collect()
}

fn gpu_encode(gpu: &Gpu, layout: FloatLayout, values: &[f32]) -> Vec<u32> {
    gpu_codes(
        gpu,
        &probes::encode_source(layout),
        &[&floats(values)],
        values.len(),
    )
}

fn assert_decode_parity(layout: FloatLayout) {
    let codes = narrow(layout, granite());
    let expected = widen(layout, &codes);

    let observed = gpu_decode(&Gpu::open(), layout, &codes);

    let mismatches = compare_bits(&expected, &observed);
    println!(
        "decode layout={layout:?} elements={} mismatches={mismatches}",
        codes.len()
    );
    assert_eq!(
        mismatches, 0,
        "{layout:?} decode differs from the CPU codec"
    );
}

fn assert_encode_parity(layout: FloatLayout) {
    let gpu = Gpu::open();
    let expected = narrow(layout, granite());

    let observed = gpu_encode(&gpu, layout, granite());

    let comparison = compare_codes(layout, &expected, &observed);
    println!("encode layout={layout:?} comparison={comparison:?}");
    assert_eq!(
        comparison.mismatches, 0,
        "{layout:?} encode differs from the CPU codec"
    );
}

fn representable_codes(layout: FloatLayout) -> Vec<u32> {
    let width = u32::from(layout.width());
    let candidates: Vec<u32> = if width <= 16 {
        (0..1u32 << width).collect()
    } else {
        (0..1u32 << 20)
            .map(|index| index.wrapping_mul(4099))
            .collect()
    };
    candidates
        .into_iter()
        .filter(|&code| decode(code, layout).is_finite())
        .collect()
}

fn assert_round_trip(layout: FloatLayout) {
    let gpu = Gpu::open();
    let codes = representable_codes(layout);
    let values = gpu_decode(&gpu, layout, &codes);

    let stored = gpu_encode(&gpu, layout, &values);

    let comparison = compare_codes(layout, &codes, &stored);
    println!(
        "round_trip layout={layout:?} codes={} comparison={comparison:?}",
        codes.len()
    );
    assert_eq!(
        comparison.mismatches, 0,
        "{layout:?} store(load(code)) changed a code"
    );
}

#[proxima::test]
#[case::binary32(FloatLayout::BF32)]
#[case::binary16(FloatLayout::F16)]
#[case::brain16(FloatLayout::BF16)]
#[case::e5m2_saturating(FloatLayout::E5M2_SATURATING)]
#[case::e2m1(FloatLayout::E2M1)]
#[case::e3m2(FloatLayout::E3M2)]
#[case::e2m3(FloatLayout::E2M3)]
#[case::e4m3_from_fields(E4M3_FROM_FIELDS)]
async fn float_layout_parity_decode_equals_cpu_codec_on_granite_rows(#[case] layout: FloatLayout) {
    assert_decode_parity(layout);
}

#[proxima::test]
#[case::binary32(FloatLayout::BF32)]
#[case::binary16(FloatLayout::F16)]
#[case::brain16(FloatLayout::BF16)]
#[case::e5m2_saturating(FloatLayout::E5M2_SATURATING)]
#[case::e2m1(FloatLayout::E2M1)]
#[case::e3m2(FloatLayout::E3M2)]
#[case::e2m3(FloatLayout::E2M3)]
#[case::e4m3_from_fields(E4M3_FROM_FIELDS)]
async fn float_layout_parity_encode_equals_cpu_codec_and_round_trips(#[case] layout: FloatLayout) {
    assert_encode_parity(layout);
    assert_round_trip(layout);
}

fn kernel_report(name: &str, layout: FloatLayout, expected: &[u32], observed: &[u32]) -> u64 {
    let comparison = compare_codes(layout, expected, observed);
    println!(
        "kernel={name} layout={layout:?} mismatches={} max_ulp={}",
        comparison.mismatches, comparison.max_ulp
    );
    comparison.max_ulp
}

fn assert_kernels(layout: FloatLayout) {
    let gpu = Gpu::open();
    let codes = narrow(layout, granite());
    let rotated: Vec<u32> = codes
        .iter()
        .cycle()
        .skip(1)
        .take(codes.len())
        .copied()
        .collect();

    let negated = gpu_codes(
        &gpu,
        &probes::unary_source(layout, "-value"),
        &[&words(&codes)],
        codes.len(),
    );
    let doubled = gpu_codes(
        &gpu,
        &probes::unary_source(layout, "value * 2.0f"),
        &[&words(&codes)],
        codes.len(),
    );
    let added = gpu_codes(
        &gpu,
        &probes::add_source(layout),
        &[&words(&codes), &words(&rotated)],
        codes.len(),
    );
    let summed = gpu_codes(
        &gpu,
        &probes::row_sum_source(layout, rows::COLUMNS),
        &[&words(&codes)],
        rows::ROWS,
    );
    let left = &codes[..MATMUL_ROWS * MATMUL_REDUCTION];
    let right = &codes[MATMUL_ROWS * MATMUL_REDUCTION..][..MATMUL_REDUCTION * MATMUL_COLUMNS];
    let product = gpu_codes(
        &gpu,
        &probes::matmul_source(layout, MATMUL_REDUCTION, MATMUL_COLUMNS),
        &[&words(left), &words(right)],
        MATMUL_ROWS * MATMUL_COLUMNS,
    );

    let negate_ulp = kernel_report(
        "negate",
        layout,
        &probes::cpu_negate(layout, &codes),
        &negated,
    );
    let double_ulp = kernel_report(
        "double",
        layout,
        &probes::cpu_double(layout, &codes),
        &doubled,
    );
    let add_ulp = kernel_report(
        "add",
        layout,
        &probes::cpu_add(layout, &codes, &rotated),
        &added,
    );
    let sum_ulp = kernel_report(
        "row_sum",
        layout,
        &probes::cpu_row_sum(layout, &codes, rows::COLUMNS),
        &summed,
    );
    let matmul_ulp = kernel_report(
        "matmul",
        layout,
        &probes::cpu_matmul(layout, left, right, MATMUL_REDUCTION, MATMUL_COLUMNS),
        &product,
    );

    assert_eq!(negate_ulp, 0, "negate is exact");
    assert_eq!(double_ulp, 0, "multiplying by 2.0 is exact");
    assert!(
        add_ulp <= 1,
        "add within 1 ulp of the layout, got {add_ulp}"
    );
    assert!(
        sum_ulp <= 1,
        "row sum within 1 ulp of the layout, got {sum_ulp}"
    );
    assert!(
        matmul_ulp <= 1,
        "matmul within 1 ulp of the layout, got {matmul_ulp}"
    );
}

#[proxima::test]
#[case::binary32(FloatLayout::BF32)]
#[case::binary16(FloatLayout::F16)]
#[case::brain16(FloatLayout::BF16)]
#[case::e5m2_saturating(FloatLayout::E5M2_SATURATING)]
#[case::e2m1(FloatLayout::E2M1)]
#[case::e3m2(FloatLayout::E3M2)]
#[case::e2m3(FloatLayout::E2M3)]
#[case::e4m3_from_fields(E4M3_FROM_FIELDS)]
async fn float_layout_parity_kernels_match_oracle(#[case] layout: FloatLayout) {
    assert_kernels(layout);
}

#[test]
fn float_layout_parity_unnamed_e4m3_runs_with_zero_new_code() {
    let emitter = include_str!("../../../omega/src/msl/layout_render.rs");
    let codec = include_str!("../../../proxima-gguf/src/quant/float_layout.rs");

    let occurrences = emitter.to_ascii_lowercase().matches("e4m3").count()
        + codec.to_ascii_lowercase().matches("e4m3").count();

    assert_eq!(occurrences, 0, "the emitter and the codec name no E4M3");
    assert_decode_parity(E4M3_FROM_FIELDS);
    assert_encode_parity(E4M3_FROM_FIELDS);
}

#[test]
fn float_layout_parity_bf8_e5m2_exponent_31_specials_msl_equals_cpu() {
    let layout = FloatLayout::E5M2_SATURATING;
    let codes: Vec<u32> = (0x7c..=0x7f).chain(0xfc..=0xff).collect();
    let expected = widen(layout, &codes);

    let observed = gpu_decode(&Gpu::open(), layout, &codes);

    for ((code, want), got) in codes.iter().zip(&expected).zip(&observed) {
        println!("code={code:#04x} cpu={want:?} msl={got:?}");
    }
    assert_eq!(compare_bits(&expected, &observed), 0);
}

#[test]
fn float_layout_parity_control_cpu_output_through_the_comparator_passes() {
    let layout = FloatLayout::BF16;
    let codes = narrow(layout, granite());
    let values = widen(layout, &codes);

    assert_eq!(compare_bits(&values, &widen(layout, &codes)), 0);
    assert_eq!(
        compare_codes(layout, &codes, &narrow(layout, &values)).mismatches,
        0
    );
}

#[test]
fn float_layout_parity_control_two_ulp_perturbation_fails() {
    let layout = FloatLayout::BF16;
    let codes = narrow(layout, granite());
    let mut perturbed = codes.clone();
    let target = codes
        .iter()
        .position(|&code| code & 0x7fff > 0x100 && code & 0x7fff < 0x7000)
        .expect("granite holds a mid-range finite value");
    perturbed[target] += 2;

    let comparison = compare_codes(layout, &codes, &perturbed);

    assert_eq!(
        comparison,
        compare::Comparison {
            mismatches: 1,
            max_ulp: 2
        }
    );
}

fn truncating_encode(value: f32, layout: FloatLayout) -> u32 {
    let nearest = encode(value, layout);
    let sign_shift = u32::from(layout.exponent_bits) + u32::from(layout.mantissa_bits);
    let magnitude_mask = (1u32 << sign_shift) - 1;
    let overshoots = decode(nearest, layout).abs() > value.abs();
    if overshoots && nearest & magnitude_mask > 0 {
        nearest - 1
    } else {
        nearest
    }
}

#[test]
fn float_layout_parity_control_truncating_store_differs() {
    let layout = FloatLayout::E5M2_SATURATING;
    let truncated: Vec<u32> = granite()
        .iter()
        .map(|&value| truncating_encode(value, layout))
        .collect();

    let observed = gpu_encode(&Gpu::open(), layout, granite());

    let comparison = compare_codes(layout, &truncated, &observed);
    println!("truncating control comparison={comparison:?}");
    assert!(
        comparison.mismatches > 0,
        "round-to-nearest must differ from truncation"
    );
}

#[test]
fn float_layout_parity_control_wrong_bias_decode_differs() {
    let layout = FloatLayout::E3M2;
    let shifted = FloatLayout {
        bias_override: Some(layout.bias() + 1),
        ..layout
    };
    let codes = narrow(layout, granite());
    let expected = widen(layout, &codes);

    let observed = gpu_decode(&Gpu::open(), shifted, &codes);

    assert!(
        compare_bits(&expected, &observed) > 0,
        "a shifted bias must change the decode"
    );
}

#[test]
fn float_layout_parity_control_int32_operand_still_rejected() {
    use proxima_tensor::{
        DType, Extent, IndexMap, NodeId, NumericPolicy, Op, QuantizedBlock, ScalarOp, append, map,
    };

    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype: DType::Int32,
            shape: vec![Extent::Static(8)],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Negate,
            operands: vec![(source, IndexMap::Affine(map::projection(1, &[0])))],
            name: None,
        },
    );
    let data = [0i32; 8];
    let blocks = [QuantizedBlock::Int32(&data)];

    let rejection = omega::plan(
        &program,
        &[],
        &blocks,
        &[NodeId(1)],
        NumericPolicy::default(),
    );

    let message = format!(
        "{:?}",
        rejection
            .err()
            .expect("Int32 as a non-index operand is rejected")
    );
    assert!(message.contains("NotLowerable"), "{message}");
}
