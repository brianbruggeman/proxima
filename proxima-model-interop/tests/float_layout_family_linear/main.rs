#![cfg(all(feature = "metal", feature = "float-layout-dtype", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

//! The linear block family (Q8_0, Q4_0, Q5_0, Q4_1, Q5_1, Q8_1, Q1_0, Q2_0) as configured
//! `BlockLayout`s: the CPU block walker and the MSL block emitter against the decoder output
//! recorded from the incumbent decoders (`proxima-tensor/specs/float-layout-dtype/fixtures`), and
//! the llama-ids agreement of a model whose linear weights were re-encoded into the family.

mod agreement;
#[path = "../float_layout_parity/gpu.rs"]
mod gpu;
mod reencode;

use std::path::PathBuf;

use omega::msl::layout_render::block_unpack_msl;
use proxima_gguf::quant::block_layout;
use proxima_primitives::Codec;
use proxima_primitives::block_layout::BlockLayout;
use proxima_primitives::float_layout::FloatLayout;

use gpu::{Gpu, read_words};

const FIXTURE_ELEMENTS: usize = 4096;

struct Recorded {
    blocks: Vec<u8>,
    decoded: Vec<f32>,
}

fn fixture_directory(codec: Codec) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../proxima-tensor/specs/float-layout-dtype/fixtures")
        .join(format!("{codec:?}"))
}

fn recorded(codec: Codec) -> Recorded {
    let directory = fixture_directory(codec);
    let blocks = std::fs::read(directory.join("blocks.bin"))
        .unwrap_or_else(|error| panic!("fixture blocks for {codec:?}: {error}"));
    let decoded = std::fs::read(directory.join("decoded_f32.bin"))
        .unwrap_or_else(|error| panic!("fixture decode for {codec:?}: {error}"))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    assert_eq!(
        decoded.len(),
        FIXTURE_ELEMENTS,
        "{codec:?} fixture element count"
    );
    Recorded { blocks, decoded }
}

fn layout_of(codec: Codec) -> BlockLayout {
    codec
        .block_layout()
        .unwrap_or_else(|| panic!("{codec:?} is in the linear family"))
}

fn mismatches(expected: &[f32], observed: &[f32]) -> usize {
    assert_eq!(
        expected.len(),
        observed.len(),
        "compared sequences differ in length"
    );
    expected
        .iter()
        .zip(observed)
        .filter(|(want, got)| want.to_bits() != got.to_bits())
        .count()
}

fn cpu_decode(layout: &BlockLayout, blocks: &[u8]) -> Vec<f32> {
    let mut output = vec![0.0f32; FIXTURE_ELEMENTS];
    block_layout::dequantize(layout, blocks, &mut output).expect("whole blocks decode");
    output
}

fn metal_decode(layout: &BlockLayout, blocks: &[u8]) -> Vec<f32> {
    let (elements, block_bytes) = (layout.elements, layout.block_bytes);
    let source = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{}
kernel void probe(device const uchar *blocks [[buffer(0)]], device float *out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{
    device const uchar *block = blocks + (gid / {elements}u) * {block_bytes}u;
    out[gid] = probe_block(block, gid % {elements}u);
}}
",
        block_unpack_msl(layout, "probe_block")
    );
    let bytes = Gpu::open().run(
        &source,
        "probe",
        &[blocks],
        FIXTURE_ELEMENTS * 4,
        FIXTURE_ELEMENTS,
    );
    read_words(&bytes).into_iter().map(f32::from_bits).collect()
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
    let rate = agreement::run("PROXIMA_ARCH_GRANITE_MOE_GGUF", "granite_moe");

    assert!(rate >= 0.95, "agreement rate {rate:.4} is below 0.95");
}

#[test]
fn float_layout_family_linear_llama_agreement_gemma4_e2b() {
    let rate = agreement::run("PROXIMA_ARCH_GEMMA4_E2B_GGUF", "gemma4_e2b");

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
            reencode::encode_block(&layout, chunk, None, block);
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
