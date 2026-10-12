#![cfg(all(feature = "metal", feature = "float-layout-dtype", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

//! The linear block family (Q8_0, Q4_0, Q5_0, Q4_1, Q5_1, Q8_1, Q1_0, Q2_0) as configured
//! `BlockLayout`s: the CPU block walker and the MSL block emitter against the decoder output
//! recorded from the incumbent decoders (`proxima-tensor/specs/float-layout-dtype/fixtures`), and
//! the llama-ids agreement of a model whose linear weights were re-encoded into the family.

mod agreement;
mod fixtures;
#[path = "../float_layout_parity/gpu.rs"]
mod gpu;
mod linear_targets;
mod reencode;

use proxima_primitives::Codec;
use proxima_primitives::float_layout::FloatLayout;

use fixtures::{FIXTURE_ELEMENTS, cpu_decode, layout_of, metal_decode, mismatches, recorded};

#[proxima::test]
#[case::q8_0(Codec::Q8_0)]
#[case::q4_0(Codec::Q4_0)]
#[case::q5_0(Codec::Q5_0)]
#[case::q4_1(Codec::Q4_1)]
#[case::q5_1(Codec::Q5_1)]
#[case::q8_1(Codec::Q8_1)]
#[case::q1_0(Codec::Q1_0)]
#[case::q2_0(Codec::Q2_0)]
async fn float_layout_family_linear_cpu_decode_equals_recorded_incumbent(#[case] codec: Codec) {
    let fixture = recorded(codec);

    let observed = cpu_decode(&layout_of(codec), &fixture.blocks);

    assert_eq!(mismatches(&fixture.decoded, &observed), 0, "{codec:?}");
}

#[proxima::test]
#[case::q8_0(Codec::Q8_0)]
#[case::q4_0(Codec::Q4_0)]
#[case::q5_0(Codec::Q5_0)]
#[case::q4_1(Codec::Q4_1)]
#[case::q5_1(Codec::Q5_1)]
#[case::q8_1(Codec::Q8_1)]
#[case::q1_0(Codec::Q1_0)]
#[case::q2_0(Codec::Q2_0)]
async fn float_layout_family_linear_metal_decode_equals_recorded_incumbent(#[case] codec: Codec) {
    let fixture = recorded(codec);

    let observed = metal_decode(&layout_of(codec), &fixture.blocks);

    assert_eq!(mismatches(&fixture.decoded, &observed), 0, "{codec:?}");
}

#[test]
fn float_layout_family_linear_control_wrong_scale_layout_differs() {
    let fixture = recorded(Codec::Q4_0);
    let mut wrong = layout_of(Codec::Q4_0);
    wrong.scale.layout = FloatLayout::BF16;

    let observed = cpu_decode(&wrong, &fixture.blocks);

    assert!(mismatches(&fixture.decoded, &observed) > 0);
}

#[test]
fn float_layout_family_linear_llama_agreement_granite() {
    let rate = agreement::run(
        "PROXIMA_ARCH_GRANITE_MOE_GGUF",
        "granite_moe",
        linear_targets::linear_target,
    );

    assert!(rate >= 0.95, "agreement rate {rate:.4} is below 0.95");
}

#[test]
fn float_layout_family_linear_llama_agreement_gemma4_e2b() {
    let rate = agreement::run(
        "PROXIMA_ARCH_GEMMA4_E2B_GGUF",
        "gemma4_e2b",
        linear_targets::linear_target,
    );

    assert!(rate >= 0.95, "agreement rate {rate:.4} is below 0.95");
}

#[test]
fn float_layout_family_linear_control_recorded_fixture_through_the_comparator_passes() {
    let fixture = recorded(Codec::Q8_0);
    let observed = cpu_decode(&layout_of(Codec::Q8_0), &fixture.blocks);
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
fn fit_encoder_round_trips_within_half_a_step() {
    for codec in [Codec::Q5_1, Codec::Q8_1, Codec::Q5_0, Codec::Q4_1] {
        let layout = layout_of(codec);
        let values: Vec<f32> = (0..FIXTURE_ELEMENTS)
            .map(|index| ((index * 37 % 101) as f32 - 50.0) * 0.013)
            .collect();
        let mut blocks = vec![
            0u8;
            FIXTURE_ELEMENTS / usize::from(layout.elements)
                * usize::from(layout.block_bytes)
        ];
        for (chunk, block) in values
            .chunks_exact(usize::from(layout.elements))
            .zip(blocks.chunks_exact_mut(usize::from(layout.block_bytes)))
        {
            linear_targets::encode_block(&layout, chunk, None, block);
        }

        let decoded = cpu_decode(&layout, &blocks);

        let worst = values
            .iter()
            .zip(&decoded)
            .map(|(want, got)| (want - got).abs())
            .fold(0.0f32, f32::max);
        println!("{codec:?} worst round-trip error {worst}");
        assert!(worst < 0.1, "{codec:?} worst error {worst}");
    }
}
