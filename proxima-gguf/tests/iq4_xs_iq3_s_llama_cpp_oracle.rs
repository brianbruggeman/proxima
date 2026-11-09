#![allow(clippy::expect_used, clippy::unwrap_used)]

//! Parity against llama.cpp's own dequantizer, over the flint checkpoint's own blocks.
//!
//! The bytes are four real super-blocks lifted straight from
//! `flint-UD-Q4_K_XL.gguf` (sha256 `63c2ecc4…`) by
//! `tests/fixtures/iq4_xs_iq3_s/iql_extract.py`, and the expected values are what
//! `dequantize_row_iq4_xs`/`dequantize_row_iq3_s` in llama.cpp print when `iql_oracle.py`
//! compiles those two functions out of `ggml-quants.c` verbatim. The tree on this host is
//! `f1ea20621`, and that is the revision every line number in the port and in `README.md`
//! names; the capture itself records `llama_build` b12110 / commit `cd959bae9`, which is
//! provenance for the ids, not a tree anything here can read. The comparison is raw `f32`
//! bit patterns, never a tolerance: a one-ulp disagreement is a port defect until proven
//! otherwise (guiding-principles principle 14).
//!
//! Why this test exists rather than a hand-worked literal: `IQ4_XS`'s 6-bit scale is
//! split across a high-bit plane and four low nibbles, and `IQ3_S`'s grid address is 9
//! bits split across `qs` and `qh` with two independent ninth bits per 8-element group.
//! A hand-derived literal can agree with a mistranslation of either. The incumbent's
//! own decoder over the incumbent's own bytes cannot.

use proxima_gguf::GgmlType;
use proxima_gguf::quant::dispatch;

/// One codec, its fixture block bytes, and the incumbent's expected `f32` bit patterns.
struct Case {
    name: &'static str,
    wire: GgmlType,
    bytes: &'static [u8],
    expected_bits: &'static str,
    block_bytes: usize,
    blocks: usize,
}

const IQ4_XS_BYTES: &[u8] = include_bytes!("fixtures/iq4_xs_iq3_s/iq4_xs.bin");
const IQ4_XS_EXPECTED: &str = include_str!("fixtures/iq4_xs_iq3_s/iq4_xs.expected.txt");
const IQ3_S_BYTES: &[u8] = include_bytes!("fixtures/iq4_xs_iq3_s/iq3_s.bin");
const IQ3_S_EXPECTED: &str = include_str!("fixtures/iq4_xs_iq3_s/iq3_s.expected.txt");

const CASES: [Case; 2] = [
    Case {
        name: "iq4_xs",
        wire: GgmlType::Iq4Xs,
        bytes: IQ4_XS_BYTES,
        expected_bits: IQ4_XS_EXPECTED,
        block_bytes: 136,
        blocks: 4,
    },
    Case {
        name: "iq3_s",
        wire: GgmlType::Iq3S,
        bytes: IQ3_S_BYTES,
        expected_bits: IQ3_S_EXPECTED,
        block_bytes: 110,
        blocks: 4,
    },
];

fn expected(case: &Case) -> Vec<u32> {
    case.expected_bits
        .split_whitespace()
        .map(|word| u32::from_str_radix(word, 16).expect("a hexadecimal f32 bit pattern"))
        .collect()
}

fn decode(case: &Case, bytes: &[u8]) -> Vec<u32> {
    let mut values = vec![0.0f32; bytes.len() / case.block_bytes * 256];
    dispatch::dequantize(case.wire, bytes, &mut values).expect("the fixture is a whole block run");
    values.into_iter().map(f32::to_bits).collect()
}

/// Compares `case`'s decoded values against the incumbent's, reporting the
/// first element that differs rather than a count of differences.
fn assert_matches_the_incumbent(case: &Case) {
    assert_eq!(
        case.bytes.len(),
        case.block_bytes * case.blocks,
        "{} fixture is not {} whole blocks",
        case.name,
        case.blocks
    );
    let expected_bits = expected(case);
    assert_eq!(
        expected_bits.len(),
        case.blocks * 256,
        "{} oracle produced the wrong element count",
        case.name
    );

    let decoded = decode(case, case.bytes);
    let first_mismatch = decoded
        .iter()
        .zip(&expected_bits)
        .position(|(ours, theirs)| ours != theirs);
    assert_eq!(
        first_mismatch,
        None,
        "{}: element {:?} differs (ours {:?}, llama.cpp {:?})",
        case.name,
        first_mismatch,
        first_mismatch.map(|index| decoded[index]),
        first_mismatch.map(|index| expected_bits[index]),
    );
}

#[test]
fn iq4_xs_dequantize_matches_the_llama_cpp_reference() {
    assert_matches_the_incumbent(&CASES[0]);
}

#[test]
fn iq3_s_dequantize_matches_the_llama_cpp_reference() {
    assert_matches_the_incumbent(&CASES[1]);
}

/// The control: the same comparison over the same bytes is not vacuous -- a single
/// flipped nibble in the first block moves at least one decoded element, so a decoder
/// that ignored the byte it corrupted would pass the parity test above by accident.
#[test]
fn control_a_flipped_byte_moves_the_decoded_values() {
    for case in &CASES {
        let expected_bits = expected(case);
        let mut mutated = case.bytes.to_vec();
        mutated[case.block_bytes / 2] ^= 0x01;
        let decoded = decode(case, &mutated);
        assert_ne!(
            decoded, expected_bits,
            "{}: flipping a packed byte left every decoded value unchanged",
            case.name
        );
    }
}

/// The fixture's own bytes, pinned. Without this the parity test compares the
/// decoder to whatever the fixture happens to hold: a regenerated or edited
/// `.expected.txt` would satisfy it, and a regenerated `.bin` would silently
/// re-pin the decoder to a different capture.
///
/// The digests are the bytes committed with this test; `README.md` names the
/// commands that rebuild them from the checkpoint and llama.cpp, so a
/// deliberate regeneration updates both the files and these constants in one
/// diff.
const FIXTURE_DIGESTS: [(&str, &str); 4] = [
    (
        "iq4_xs.bin",
        "bdedb5ff738f1bb9dffe3e21eab719c6831a864498923da5b02cceb9262440fc",
    ),
    (
        "iq4_xs.expected.txt",
        "02c1d3d1e5668afa2d5d32c66067aca1c8f1d8efc6c35f694691903ab72f139c",
    ),
    (
        "iq3_s.bin",
        "5eee1838cd144aa3891eae754e580f106e00141cf52b694a468d05c5a84fb42d",
    ),
    (
        "iq3_s.expected.txt",
        "ed31e0dadc2ee8d547f50143331e66c9306403c27248b95bb0f343c824e077bb",
    ),
];

#[test]
fn the_fixture_bytes_are_the_ones_this_test_was_measured_on() {
    use sha2::{Digest, Sha256};

    for (name, expected) in FIXTURE_DIGESTS {
        let bytes: &[u8] = match name {
            "iq4_xs.bin" => IQ4_XS_BYTES,
            "iq4_xs.expected.txt" => IQ4_XS_EXPECTED.as_bytes(),
            "iq3_s.bin" => IQ3_S_BYTES,
            "iq3_s.expected.txt" => IQ3_S_EXPECTED.as_bytes(),
            other => panic!("{other} is not one of this test's fixtures"),
        };
        let actual: String = Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert_eq!(actual, expected, "{name} changed since it was captured");
    }
}
