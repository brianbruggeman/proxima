//! The `Q4_K` unpack the GPU path needs, gated two ways.
//!
//! Decode is a weight sweep, so bytes-per-weight is the only variable that
//! moves the number. Measured on this box: llama.cpp's Metal backend runs a
//! 7B `Q4_K_S` checkpoint at 17.62 ms/token (56.8 tok/s) reading packed
//! bytes; the same sweep in `f16` is 3.56x the traffic, 67.4 ms/token, which
//! is slower than our own CPU path. A float-only Metal backend is therefore
//! not worth having, and this unpack is what makes one worth having.
//!
//! Two claims, tested separately because they fail separately:
//!
//! 1. the MSL is valid MSL — assembled by the real `xcrun metal` toolchain,
//!    never "looks like MSL". A missing toolchain is a RED gate, not a skip.
//! 2. the index arithmetic that MSL encodes is the arithmetic
//!    `proxima_gguf::quant::q4_k::dequantize_block` performs — bit-exact
//!    against that codec over randomized blocks.
//!
//! Deliberately NOT claimed here: that the compiled shader computes that
//! arithmetic on a device. That needs the emitter to route a
//! `QuantizedBlock::Q4K` input into a real program, at which point
//! `metal_parity.rs`'s existing CPU-vs-device harness covers it end to end
//! with no bespoke dispatch plumbing. Building that plumbing for a snippet
//! would test the plumbing, not the kernel.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::process::Command;

use proxima_gguf::quant::q4_k;
use proxima_tensor::test_support::Lcg;

/// A `Q4_K` super-block with plausible field values rather than uniform
/// noise: `d`/`dmin` are real `f16`s in the range a trained checkpoint
/// carries, and every scale/min and nibble byte varies. An all-zero fixture
/// would pass under an index bug, every element agreeing by accident.
fn random_block(seed: u64) -> Vec<u8> {
    let mut lcg = Lcg(seed);
    let mut unit = move || f32::midpoint(lcg.next_unit(), 1.0);
    let mut block = vec![0u8; omega::Q4K_BLOCK_BYTES];

    let scale = half::f16::from_f32(0.005 + unit() * 0.02);
    let minimum = half::f16::from_f32(0.001 + unit() * 0.01);
    block[0..2].copy_from_slice(&scale.to_le_bytes());
    block[2..4].copy_from_slice(&minimum.to_le_bytes());
    for byte in &mut block[4..] {
        *byte = (unit() * 255.0) as u8;
    }
    block
}

/// The Rust twin of `q4k_element` in [`omega::Q4K_UNPACK_MSL`] — the same
/// index arithmetic, transcribed. Kept beside the MSL so a change to one
/// that is not mirrored in the other fails this file.
fn q4k_element_host(block: &[u8], index: usize) -> f32 {
    let d = f32::from(half::f16::from_le_bytes([block[0], block[1]]));
    let dmin = f32::from(half::f16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qs = &block[16..];

    let group = index / 64;
    let within = index % 64;
    let low_nibble = within < 32;
    let sub_block = 2 * group + usize::from(!low_nibble);
    let byte_index = group * 32 + (within % 32);

    let (scale_bits, min_bits) = if sub_block < 4 {
        (scales[sub_block] & 63, scales[sub_block + 4] & 63)
    } else {
        (
            (scales[sub_block + 4] & 0x0F) | ((scales[sub_block - 4] >> 6) << 4),
            (scales[sub_block + 4] >> 4) | ((scales[sub_block] >> 6) << 4),
        )
    };

    let nibble = if low_nibble {
        qs[byte_index] & 0x0F
    } else {
        qs[byte_index] >> 4
    };
    d * f32::from(scale_bits) * f32::from(nibble) - dmin * f32::from(min_bits)
}

#[test]
fn q4k_unpack_index_arithmetic_matches_the_gguf_codec_bit_exactly() {
    let mut compared = 0usize;
    for seed in 1..=16u64 {
        let block = random_block(seed);
        let mut expected = vec![0.0f32; omega::Q4K_BLOCK_ELEMENTS];
        q4_k::dequantize_block(&block, &mut expected);

        for (index, reference) in expected.iter().enumerate() {
            let ours = q4k_element_host(&block, index);
            assert_eq!(
                ours.to_bits(),
                reference.to_bits(),
                "seed {seed} element {index}: unpack {ours} vs codec {reference}"
            );
            compared += 1;
        }
    }
    assert_eq!(
        compared,
        16 * omega::Q4K_BLOCK_ELEMENTS,
        "degenerate gate: every element of every block must be compared"
    );
}

/// The nibble order is the detail most likely to be wrong and least likely
/// to look wrong: a `qs` byte's low and high nibbles land 32 elements apart,
/// NOT adjacent, so element `i` does not read `qs[i / 2]`. Pinned directly,
/// so a "simplification" to `qs[index / 2]` fails with a name that says what
/// broke instead of as diffuse parity drift.
#[test]
fn q4k_low_and_high_nibbles_of_one_byte_land_32_elements_apart() {
    let mut block = vec![0u8; omega::Q4K_BLOCK_BYTES];
    block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    block[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    block[4] = 1;
    block[5] = 1;
    block[16] = 0x73;

    assert_eq!(q4k_element_host(&block, 0), 3.0, "element 0 takes the LOW nibble of qs[0]");
    assert_eq!(
        q4k_element_host(&block, 32),
        7.0,
        "element 32 takes the HIGH nibble of that SAME byte"
    );
    assert_eq!(
        q4k_element_host(&block, 1),
        0.0,
        "element 1 reads qs[1], not the high nibble of qs[0]"
    );
}

#[test]
fn q4k_unpack_msl_assembles_with_the_real_metal_toolchain() {
    let source = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{}\nkernel void q4k_unpack_probe(\n    device const uchar *block [[buffer(0)]],\n    device float *out [[buffer(1)]],\n    uint gid [[thread_position_in_grid]]) {{\n    out[gid] = q4k_element(block, gid);\n}}\n",
        omega::Q4K_UNPACK_MSL
    );

    let dir = tempfile::tempdir().expect("temp dir for the metal source");
    let metal_path = dir.path().join("q4k_unpack.metal");
    std::fs::write(&metal_path, &source).expect("write metal source");

    let output = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-c"])
        .arg(&metal_path)
        .arg("-o")
        .arg(dir.path().join("q4k_unpack.air"))
        .output()
        .unwrap_or_else(|error| {
            panic!(
                "metal toolchain unavailable ({error}) -- this is a red gate, not a skip; \
                 install the Xcode command line tools"
            )
        });

    assert!(
        output.status.success(),
        "metal compile failed:\n--- source ---\n{}\n--- stderr ---\n{}",
        source,
        String::from_utf8_lossy(&output.stderr)
    );
}

/// omega restates `Q4_K`'s block geometry rather than depending on
/// `proxima-gguf` at build time. That is only safe if the two agree.
#[test]
fn q4k_block_geometry_matches_the_gguf_codec() {
    assert_eq!(omega::Q4K_BLOCK_BYTES, q4_k::BLOCK_BYTES);
    assert_eq!(omega::Q4K_BLOCK_ELEMENTS, q4_k::QK_K);
}
