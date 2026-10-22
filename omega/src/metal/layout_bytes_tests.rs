use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use proxima_primitives::float_layout::{FloatLayout, SignEncoding, Specials};

use super::device_buffers_arena_plan::device_and_queue;
use crate::Codec;
use crate::msl::{codec_block_bytes, codec_block_elements};

const ROWS: u64 = 1024;
const COLUMNS: u64 = 512;
const F16_SCALE_PER_32: Option<(FloatLayout, u64)> = Some((FloatLayout::F16, 32));

const E4M3_FROM_FIELDS: FloatLayout = FloatLayout {
    exponent_bits: 4,
    mantissa_bits: 3,
    specials: Specials::NanOnly,
    bias_override: None,
    sign: SignEncoding::Magnitude,
};

fn bits_per_element(layout: FloatLayout, scale_block: Option<(FloatLayout, u64)>) -> f64 {
    let elements = 1 << 12;
    layout.packed_bytes(elements, scale_block) as f64 * 8.0 / elements as f64
}

#[proxima::test]
#[case::binary32(FloatLayout::BF32, None, 32.0)]
#[case::binary16(FloatLayout::F16, None, 16.0)]
#[case::brain16(FloatLayout::BF16, None, 16.0)]
#[case::e5m2_saturating(FloatLayout::E5M2_SATURATING, None, 8.0)]
#[case::e2m1(FloatLayout::E2M1, None, 4.0)]
#[case::e3m2_under_f16_scale(FloatLayout::E3M2, F16_SCALE_PER_32, 6.5)]
#[case::e2m3_under_f16_scale(FloatLayout::E2M3, F16_SCALE_PER_32, 6.5)]
#[case::e4m3_from_fields(E4M3_FROM_FIELDS, None, 8.0)]
async fn float_layout_bytes_per_element_asserted(
    #[case] layout: FloatLayout,
    #[case] scale_block: Option<(FloatLayout, u64)>,
    #[case] expected_bits: f64,
) {
    assert_eq!(bits_per_element(layout, scale_block), expected_bits);
}

fn codec_bytes(codec: Codec, elements: u64) -> u64 {
    elements / codec_block_elements(codec) as u64 * codec_block_bytes(codec) as u64
}

#[test]
fn float_layout_bytes_metal_buffer_equals_formula_1024x512() {
    let Ok((device, _queue)) = device_and_queue() else {
        panic!("this box has a Metal device; set up the lock and run on the Mac");
    };
    let elements = ROWS * COLUMNS;
    let layouts = [
        (FloatLayout::BF16, None, Codec::BFloat16),
        (FloatLayout::E5M2_SATURATING, None, Codec::Bf8E5M2),
        (FloatLayout::E2M1, None, Codec::Bf4E2M1),
        (FloatLayout::E3M2, F16_SCALE_PER_32, Codec::Fp6E3M2),
    ];

    for (layout, scale_block, codec) in layouts {
        let packed = vec![0u8; codec_bytes(codec, elements) as usize];
        let buffer = device
            .newBufferWithLength_options(packed.len(), MTLResourceOptions::StorageModeShared)
            .expect("the device allocates a packed operand buffer");

        assert_eq!(
            buffer.length() as u64,
            layout.packed_bytes(elements, scale_block),
            "{layout:?} operand buffer at {ROWS}x{COLUMNS}"
        );
    }
}

#[test]
fn float_layout_bytes_control_scale_bytes_omitted_disagrees() {
    let elements = ROWS * COLUMNS;

    let without_scale = FloatLayout::E3M2.packed_bytes(elements, None);
    let with_scale = FloatLayout::E3M2.packed_bytes(elements, F16_SCALE_PER_32);

    assert_eq!(with_scale, codec_bytes(Codec::Fp6E3M2, elements));
    assert_ne!(without_scale, codec_bytes(Codec::Fp6E3M2, elements));
    assert_eq!(with_scale - without_scale, elements / 32 * 2);
}
