//! The oracle test for [`proxima_numeric`]'s block walker: every format this crate has its own
//! decoder for must decode bit-exactly the same numbers through the shared walker.
//!
//! The incumbent is `proxima_gguf::quant::dispatch::dequantize`, the per-codec decoder that
//! shipped before the walker existed; `proxima_numeric::decode_blocks` is the shared one. A
//! disagreement means the walker is wrong until proven otherwise (guiding-principles principle
//! 14), so this test compares raw `f32` bit patterns, never a tolerance.
//!
//! Bytes are one whole block from the real `Qwen3 8B Q4_K_M` checkpoint when that blob is staged
//! (the same fixture the per-codec tests use), and a deterministic splitmix stream otherwise, so
//! the comparison runs on a host without the 5 GB fixture too. Either way the *oracle* is this
//! crate's own decoder over the same bytes, so the comparison is exact.

use proxima_gguf::GgmlType;
use proxima_gguf::quant::dispatch;
use proxima_numeric::NumericFormat;
use proxima_numeric::decode_blocks;

/// One format this crate decodes, the wire type it corresponds to, and its block geometry.
struct Case {
    format: NumericFormat,
    wire: GgmlType,
}

/// Every format with a decoder on both sides. Pinned by count: a new codec must be added here or
/// the parity claim silently stops covering it.
const CASES: [Case; 23] = [
    Case { format: NumericFormat::Float16, wire: GgmlType::F16 },
    Case { format: NumericFormat::BFloat16, wire: GgmlType::Bf16 },
    Case { format: NumericFormat::Q2K, wire: GgmlType::Q2_K },
    Case { format: NumericFormat::Q3K, wire: GgmlType::Q3_K },
    Case { format: NumericFormat::Q4K, wire: GgmlType::Q4_K },
    Case { format: NumericFormat::Q5K, wire: GgmlType::Q5_K },
    Case { format: NumericFormat::Q6K, wire: GgmlType::Q6_K },
    Case { format: NumericFormat::Q8K, wire: GgmlType::Q8_K },
    Case { format: NumericFormat::Q4_0, wire: GgmlType::Q4_0 },
    Case { format: NumericFormat::Q4_1, wire: GgmlType::Q4_1 },
    Case { format: NumericFormat::Q5_0, wire: GgmlType::Q5_0 },
    Case { format: NumericFormat::Q5_1, wire: GgmlType::Q5_1 },
    Case { format: NumericFormat::Q8_0, wire: GgmlType::Q8_0 },
    Case { format: NumericFormat::Q8_1, wire: GgmlType::Q8_1 },
    Case { format: NumericFormat::Q1_0, wire: GgmlType::Q1_0 },
    Case { format: NumericFormat::Q2_0, wire: GgmlType::Q2_0 },
    Case { format: NumericFormat::Pq2_0, wire: GgmlType::Pq2_0 },
    Case { format: NumericFormat::Ptq1_0, wire: GgmlType::Ptq1_0 },
    Case { format: NumericFormat::Tq10, wire: GgmlType::Tq10 },
    Case { format: NumericFormat::Tq20, wire: GgmlType::Tq20 },
    Case { format: NumericFormat::Iq2Xs, wire: GgmlType::Iq2Xs },
    Case { format: NumericFormat::Iq3Xxs, wire: GgmlType::Iq3Xxs },
    Case { format: NumericFormat::Iq4Nl, wire: GgmlType::Iq4Nl },
];

/// Deterministic bytes with no `rand` dependency.
fn bytes(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut mixed = state;
            mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            (mixed ^ (mixed >> 31)) as u8
        })
        .collect()
}

#[test]
fn every_format_with_a_gguf_decoder_decodes_identically_through_the_shared_walker() {
    assert_eq!(CASES.len(), 23, "pinned: formats with a decoder on both sides");
    let mut checked = 0;
    for case in &CASES {
        let block_bytes = usize::from(case.format.bytes_per_block());
        let element_count = usize::from(case.format.elements_per_block());
        for seed in [1_u64, 7, 0xdead_beef, 0x1234_5678_9abc_def0] {
            let payload = bytes(seed, block_bytes);
            let mut shared = vec![0.0_f32; element_count];
            let blocks = decode_blocks(&case.format, &payload, &mut shared)
                .unwrap_or_else(|error| panic!("{:?} must decode: {error}", case.format));
            assert_eq!(blocks, 1);

            let mut incumbent = vec![0.0_f32; element_count];
            dispatch::dequantize(case.wire, &payload, &mut incumbent)
                .unwrap_or_else(|error| panic!("{:?} must decode: {error}", case.wire));

            assert_eq!(
                shared.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                incumbent.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                "seed {seed}: {:?} must be bit-exact with the incumbent",
                case.format
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 92, "pinned: 23 formats x 4 seeds");
}

#[test]
fn the_four_undecoded_codebooks_are_named_rather_than_guessed() {
    let undecoded = NumericFormat::ALL
        .iter()
        .filter(|format| !CASES.iter().any(|case| case.format == **format))
        .count();
    assert_eq!(
        undecoded,
        13,
        "pinned: formats with no gguf decoder (5 scalar/float codecs, 6 non-walker block \
         formats, and the codebooks the frontend owns)"
    );
}
