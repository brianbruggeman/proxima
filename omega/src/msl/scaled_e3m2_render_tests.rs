use alloc::collections::BTreeMap;

use proxima_primitives::float_layout::FloatLayout;

use super::tests::matmul_op;
use super::*;

const ROW_BLOCKED_1024X512: &str =
    include_str!("../../tests/fixtures/scaled_e3m2_row_blocked_1024x512.msl");
const ROW_BLOCKED_1536X6144: &str =
    include_str!("../../tests/fixtures/scaled_e3m2_row_blocked_1536x6144.msl");
const DECODER_START: &str = "static inline float scaled_e3m2_field";

fn matvec_source(rows: u32, reduction: u32) -> String {
    let bound = matmul_op(rows, reduction, 1);
    let packed = BTreeMap::from([(bound.operands()[0].0, Codec::Fp6E3M2)]);
    let kernel = emit(&bound, &packed, NumericPolicy::default()).expect("an e3m2 matvec emits");
    let start = kernel
        .source
        .find(DECODER_START)
        .expect("the e3m2 decoders are spliced in");
    kernel.source[start..].to_string()
}

fn assert_carries_the_six_bit_unpack(source: &str) {
    for shift in [0, 6, 12, 18] {
        assert!(
            source.contains(&format!("(word >> {shift}u) & 0x3fu")),
            "group shift {shift} with the 0x3f mask:\n{source}"
        );
    }
    assert!(
        source.contains("payload + group * 3u"),
        "3-byte stride per group of 4 weights:\n{source}"
    );
    assert!(
        source.contains("sumf[q] + scaled_e3m2_quad_dot(blk, slot, acts)"),
        "the quad dot is the row-blocked body:\n{source}"
    );
}

#[test]
fn scaled_e3m2_render_row_blocked_1024x512() {
    let source = matvec_source(1024, 512);

    assert_eq!(source, ROW_BLOCKED_1024X512);
    assert_carries_the_six_bit_unpack(&source);
}

#[test]
fn scaled_e3m2_render_row_blocked_1536x6144() {
    let source = matvec_source(1536, 6144);

    assert_eq!(source, ROW_BLOCKED_1536X6144);
    assert_carries_the_six_bit_unpack(&source);
}

#[test]
fn scaled_e3m2_render_decode_carries_bias_3_and_field_masks() {
    let layout = FloatLayout::E3M2;
    let shifted = FloatLayout {
        bias_override: Some(4),
        ..layout
    };

    let source = scaled_e3m2_unpack_msl(layout);
    let rebiased = scaled_e3m2_unpack_msl(shifted);

    assert_eq!(layout.bias(), 3);
    assert!(source.contains("const int bias = 3;"));
    assert!(
        source.contains("(field & 0x1cu) >> 2u"),
        "exponent mask 0x1c over 2 mantissa bits:\n{source}"
    );
    assert!(
        source.contains("field & 0x03u"),
        "mantissa mask 0x03:\n{source}"
    );
    assert!(rebiased.contains("const int bias = 4;") && !rebiased.contains("const int bias = 3;"));
    assert_ne!(source, rebiased);
}
