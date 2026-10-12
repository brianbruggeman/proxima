#![cfg(all(feature = "metal", feature = "float-layout-dtype", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

//! The K family (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K) as nested `BlockLayout`s: the CPU block
//! walker and the MSL block emitter against the decoder output recorded from the incumbent
//! decoders, the descriptor fields of the classification table, and the llama-ids agreement of
//! a model whose block weights were re-encoded into the family.

#[path = "../float_layout_family_linear/agreement.rs"]
mod agreement;
#[path = "../float_layout_family_linear/fixtures.rs"]
mod fixtures;
#[path = "../float_layout_parity/gpu.rs"]
mod gpu;
mod k_targets;
#[path = "../float_layout_family_linear/reencode.rs"]
mod reencode;

use proxima_primitives::Codec;
use proxima_primitives::block_layout::BlockLayout;
use proxima_primitives::float_layout::FloatLayout;

use fixtures::{cpu_decode, layout_of, metal_decode, mismatches, recorded};

#[proxima::test]
#[case::q2_k(Codec::Q2K)]
#[case::q3_k(Codec::Q3K)]
#[case::q4_k(Codec::Q4K)]
#[case::q5_k(Codec::Q5K)]
#[case::q6_k(Codec::Q6K)]
#[case::q8_k(Codec::Q8K)]
async fn float_layout_family_k_cpu_decode_equals_recorded_incumbent(#[case] codec: Codec) {
    let fixture = recorded(codec);

    let observed = cpu_decode(&layout_of(codec), &fixture.blocks);

    assert_eq!(mismatches(&fixture.decoded, &observed), 0, "{codec:?}");
}

#[proxima::test]
#[case::q2_k(Codec::Q2K)]
#[case::q3_k(Codec::Q3K)]
#[case::q4_k(Codec::Q4K)]
#[case::q5_k(Codec::Q5K)]
#[case::q6_k(Codec::Q6K)]
#[case::q8_k(Codec::Q8K)]
async fn float_layout_family_k_metal_decode_equals_recorded_incumbent(#[case] codec: Codec) {
    let fixture = recorded(codec);

    let observed = metal_decode(&layout_of(codec), &fixture.blocks);

    assert_eq!(mismatches(&fixture.decoded, &observed), 0, "{codec:?}");
}

struct Expected {
    codec: Codec,
    sub_blocks: Option<u16>,
    scale_code: Option<(FloatLayout, i16)>,
    min_code: Option<FloatLayout>,
    element_bits: u8,
    scale_layout: FloatLayout,
    min_multiplier: i8,
}

#[test]
fn float_layout_family_k_descriptors_state_sub_block_fields() {
    let uint = FloatLayout::UINT6;
    let table = [
        Expected {
            codec: Codec::Q2K,
            sub_blocks: Some(16),
            scale_code: Some((FloatLayout::UINT4, 0)),
            min_code: Some(FloatLayout::UINT4),
            element_bits: 2,
            scale_layout: FloatLayout::F16,
            min_multiplier: -1,
        },
        Expected {
            codec: Codec::Q3K,
            sub_blocks: Some(16),
            scale_code: Some((uint, 32)),
            min_code: None,
            element_bits: 3,
            scale_layout: FloatLayout::F16,
            min_multiplier: 1,
        },
        Expected {
            codec: Codec::Q4K,
            sub_blocks: Some(8),
            scale_code: Some((uint, 0)),
            min_code: Some(uint),
            element_bits: 4,
            scale_layout: FloatLayout::F16,
            min_multiplier: -1,
        },
        Expected {
            codec: Codec::Q5K,
            sub_blocks: Some(8),
            scale_code: Some((uint, 0)),
            min_code: Some(uint),
            element_bits: 5,
            scale_layout: FloatLayout::F16,
            min_multiplier: -1,
        },
        Expected {
            codec: Codec::Q6K,
            sub_blocks: Some(16),
            scale_code: Some((FloatLayout::INT8, 0)),
            min_code: None,
            element_bits: 6,
            scale_layout: FloatLayout::F16,
            min_multiplier: 1,
        },
        Expected {
            codec: Codec::Q8K,
            sub_blocks: None,
            scale_code: None,
            min_code: None,
            element_bits: 8,
            scale_layout: FloatLayout::BF32,
            min_multiplier: 1,
        },
    ];

    for expected in table {
        let layout = layout_of(expected.codec);
        let sub = layout.sub;
        let element_bits =
            layout.element.low.bits + layout.element.high.map_or(0, |plane| plane.bits);
        assert_eq!(layout.elements, 256, "{:?}", expected.codec);
        assert_eq!(
            sub.map(|sub| sub.count),
            expected.sub_blocks,
            "{:?}",
            expected.codec
        );
        assert_eq!(
            sub.map(|sub| (sub.scale.code, sub.scale.bias)),
            expected.scale_code,
            "{:?}",
            expected.codec
        );
        assert_eq!(
            sub.and_then(|sub| sub.min).map(|min| min.code),
            expected.min_code,
            "{:?}",
            expected.codec
        );
        assert_eq!(element_bits, expected.element_bits, "{:?}", expected.codec);
        assert_eq!(
            layout.scale.layout, expected.scale_layout,
            "{:?}",
            expected.codec
        );
        assert_eq!(
            layout.min_multiplier, expected.min_multiplier,
            "{:?}",
            expected.codec
        );
    }
}

#[test]
fn float_layout_family_k_llama_agreement_gemma4_e2b() {
    let rate = agreement::run(
        "PROXIMA_ARCH_GEMMA4_E2B_GGUF",
        "gemma4_e2b",
        k_targets::k_target,
    );

    assert!(rate >= 0.95, "agreement rate {rate:.4} is below 0.95");
}

#[test]
fn float_layout_family_k_control_recorded_fixture_through_the_comparator_passes() {
    let fixture = recorded(Codec::Q4K);
    let observed = cpu_decode(&layout_of(Codec::Q4K), &fixture.blocks);
    let cases = agreement::cases("granite_moe");

    let recorded_rate = agreement::agree(&cases, &agreement::recorded_generations(&cases));
    let shifted_rate = agreement::agree(&cases, &agreement::incremented_generations(&cases));

    assert_eq!(mismatches(&fixture.decoded, &observed), 0);
    assert_eq!(
        recorded_rate.0, recorded_rate.1,
        "the recorded ids agree with themselves"
    );
    assert!((shifted_rate.0 as f64) < 0.95 * shifted_rate.1 as f64);
}

#[test]
fn float_layout_family_k_control_swapped_sub_scale_layout_differs() {
    let fixture = recorded(Codec::Q4K);
    let mut swapped: BlockLayout = layout_of(Codec::Q4K);
    let sub = swapped.sub.as_mut().expect("Q4_K has sub-blocks");
    let min = sub.min.as_mut().expect("Q4_K has a sub-min");
    std::mem::swap(&mut sub.scale, min);

    let observed = cpu_decode(&swapped, &fixture.blocks);

    assert!(mismatches(&fixture.decoded, &observed) > 0);
}
