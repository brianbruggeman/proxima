use proxima_primitives::float_layout::FloatLayout;

use super::golden_support::{e3m2_kernel_source, full_matvec_source, matvec_source};
#[cfg(feature = "metal-tiled-gemm")]
use super::tests::{TILED_ADMITTED_TOKENS, gathered_matmul_op, tiled_gemm_op};
use super::*;

const ROW_BLOCKED_1024X512: &str =
    include_str!("../../tests/fixtures/scaled_e3m2_row_blocked_1024x512.msl");
const ROW_BLOCKED_1536X6144: &str =
    include_str!("../../tests/fixtures/scaled_e3m2_row_blocked_1536x6144.msl");

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
    assert!(source.contains("(int)exponent - (3) - 2"));
    assert!(
        source.contains("(code >> 2u) & 0x7u"),
        "exponent field of 3 bits over 2 mantissa bits:\n{source}"
    );
    assert!(
        source.contains("code & 0x3u"),
        "mantissa mask 0x3:\n{source}"
    );
    assert!(
        rebiased.contains("(int)exponent - (4) - 2")
            && !rebiased.contains("(int)exponent - (3) - 2")
    );
    assert_ne!(source, rebiased);
}

#[cfg(feature = "metal-tiled-gemm")]
const GROUPED_EXPERTS: u32 = 8;
#[cfg(feature = "metal-tiled-gemm")]
const GROUPED_ROWS: u32 = 192;
#[cfg(feature = "metal-tiled-gemm")]
const GROUPED_REDUCTION: u32 = 512;

#[cfg(feature = "metal-tiled-gemm")]
fn tiled_source() -> String {
    e3m2_kernel_source(&tiled_gemm_op(TILED_ADMITTED_TOKENS, 512, 192))
}

#[cfg(feature = "metal-tiled-gemm")]
fn grouped_source() -> String {
    e3m2_kernel_source(&gathered_matmul_op(
        300,
        GROUPED_EXPERTS,
        GROUPED_ROWS,
        GROUPED_REDUCTION,
    ))
}

#[cfg(feature = "metal-tiled-gemm")]
fn e3m2_decode_description() -> TiledDecode {
    tiled_decode(Codec::Fp6E3M2).expect("the e3m2 codec has a tiled decode description")
}

#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn scaled_e3m2_render_tiled_decode_elements_32_bytes_26() {
    let decode = e3m2_decode_description();
    let source = tiled_source();

    assert_eq!(decode.block_elements, 32);
    assert_eq!(decode.block_bytes, 26);
    assert_eq!(decode.function, "scaled_e3m2_decode");
    assert!(
        source.contains(&format!("{}(wws_blk", decode.function)),
        "the tiled body stages through the description:\n{source}"
    );
    assert!(source.contains(decode.msl));
}

#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn scaled_e3m2_render_grouped_gemm_source_uses_the_tiled_decode() {
    let decode = e3m2_decode_description();
    let source = grouped_source();

    assert!(source.contains("scaled_e3m2_decode"));
    assert!(source.contains(decode.msl));
    assert_eq!(
        source
            .matches("static inline float scaled_e3m2_field")
            .count(),
        1
    );
    assert!(source.contains(&format!("+= 1u * {};", decode.block_bytes)));
}

#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn scaled_e3m2_render_model_name_free() {
    let sources = [
        full_matvec_source(1024, 512),
        tiled_source(),
        grouped_source(),
    ];

    assert_eq!(sources.len(), 3);
    for source in sources.map(|text| text.to_lowercase()) {
        for name in ["gemma", "granite", "qwen", "llama"] {
            assert!(!source.contains(name), "{name} in a rendered source");
        }
    }
}

#[cfg(all(
    feature = "std",
    any(all(feature = "metal", target_os = "macos"), feature = "wgpu-backend")
))]
#[test]
fn scaled_e3m2_bind_maps_a_packed_operand_to_its_codec_and_a_float_operand_to_none() {
    let packed_bytes = [0u8; 4 * Codec::FP6_E3M2_BLOCK_BYTES];
    let activation = [0.5f32; 32];
    let packed = QuantizedBlock::Packed {
        codec: Codec::Fp6E3M2,
        bytes: &packed_bytes,
    };

    assert_eq!(codec_from_quantized_block(&packed), Some(Codec::Fp6E3M2));
    assert_eq!(
        codec_from_quantized_block(&QuantizedBlock::Float32(&activation)),
        None
    );
}
