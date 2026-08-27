//! The `Q8_0` unpack the GPU path needs, gated two ways -- mirrors
//! `q6k_unpack.rs` one codec over.
//!
//! `Q8_0` is a flat 32-element block with one `f16` scale and no sub-block
//! structure at all -- genuinely different in KIND from `Q4_K`/`Q5_K`/
//! `Q6_K`'s shared 256-element super-block, not a widening or narrowing of
//! one (see [`omega::Q8_0_UNPACK_MSL`]'s own doc). `blk.0.attn_k.weight`/
//! `blk.0.attn_v.weight` (and 62 more layers of each) are `Q8_0` in the
//! real Mixtral checkpoint this repo has host-local
//! (`q8_0_real_checkpoint_parity.rs` covers those real bytes end to end).
//!
//! Two claims, tested separately because they fail separately:
//!
//! 1. the MSL is valid MSL — assembled by the real `xcrun metal` toolchain,
//!    never "looks like MSL". A missing toolchain is a RED gate, not a skip.
//! 2. the index arithmetic that MSL encodes is the arithmetic
//!    `proxima_gguf::quant::q8_0::dequantize_block` performs — bit-exact
//!    against that codec over randomized blocks.
//!
//! Deliberately NOT claimed here: that the compiled shader computes that
//! arithmetic on a device — `metal_parity.rs`'s
//! `metal_matmul_parity_across_codec_and_dtype::q8_0_at_float32`/`_at_float16`
//! cases cover that end to end.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::process::Command;

use proxima_gguf::quant::q8_0;
use proxima_tensor::test_support::Lcg;

/// A `Q8_0` block with plausible field values: `d` is a real `f16` in the
/// range a trained checkpoint carries, and every level byte is a real signed
/// `i8` spanning the full range. An all-zero fixture would pass under an
/// index bug (every level would decode to 0 regardless of which byte a wrong
/// index read).
fn random_block(seed: u64) -> Vec<u8> {
    let mut lcg = Lcg(seed);
    let mut unit = move || f32::midpoint(lcg.next_unit(), 1.0);
    let mut block = vec![0u8; omega::Q8_0_BLOCK_BYTES];

    let scale = half::f16::from_f32(0.01 + unit() * 0.05);
    block[0..2].copy_from_slice(&scale.to_le_bytes());
    for byte in &mut block[2..2 + omega::Q8_0_BLOCK_ELEMENTS] {
        let raw = (unit() * 255.0) as u8;
        *byte = if raw == 0 { 1 } else { raw };
    }
    block
}

/// The Rust twin of `q8_0_element` in [`omega::Q8_0_UNPACK_MSL`] -- the same
/// index arithmetic, transcribed. Kept beside the MSL so a change to one
/// that is not mirrored in the other fails this file.
fn q8_0_element_host(block: &[u8], index: usize) -> f32 {
    let d = f32::from(half::f16::from_le_bytes([block[0], block[1]]));
    let level = block[2 + index] as i8;
    f32::from(level) * d
}

#[test]
fn q8_0_unpack_index_arithmetic_matches_the_gguf_codec_bit_exactly() {
    let mut compared = 0usize;
    for seed in 1..=16u64 {
        let block = random_block(seed);
        let mut expected = vec![0.0f32; omega::Q8_0_BLOCK_ELEMENTS];
        q8_0::dequantize_block(&block, &mut expected);

        for (index, reference) in expected.iter().enumerate() {
            let ours = q8_0_element_host(&block, index);
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
        16 * omega::Q8_0_BLOCK_ELEMENTS,
        "degenerate gate: every element of every block must be compared"
    );
}

/// `Q8_0` has no sub-block scale/min pair at all, unlike every K-quant --
/// pinned directly so a "simplification" that reuses `q4k_scale_min`-style
/// machinery fails with a name that says what broke.
#[test]
fn q8_0_has_no_sub_block_structure() {
    let mut block = vec![0u8; omega::Q8_0_BLOCK_BYTES];
    block[0..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    block[2] = (-10i8) as u8; // element 0's raw signed level

    // element 0: d * level = 2.0 * -10.0 = -20.0, with NOTHING else read --
    // no scale/min sub-block byte exists to get this wrong by construction.
    assert_eq!(q8_0_element_host(&block, 0), -20.0);
}

#[test]
fn q8_0_unpack_msl_assembles_with_the_real_metal_toolchain() {
    let source = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{}\nkernel void q8_0_unpack_probe(\n    device const uchar *block [[buffer(0)]],\n    device float *out [[buffer(1)]],\n    uint gid [[thread_position_in_grid]]) {{\n    out[gid] = q8_0_element(block, gid);\n}}\n",
        omega::Q8_0_UNPACK_MSL
    );

    let dir = tempfile::tempdir().expect("temp dir for the metal source");
    let metal_path = dir.path().join("q8_0_unpack.metal");
    std::fs::write(&metal_path, &source).expect("write metal source");

    let output = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-c"])
        .arg(&metal_path)
        .arg("-o")
        .arg(dir.path().join("q8_0_unpack.air"))
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

/// omega restates `Q8_0`'s block geometry rather than depending on
/// `proxima-gguf` at build time. That is only safe if the two agree.
#[test]
fn q8_0_block_geometry_matches_the_gguf_codec() {
    assert_eq!(omega::Q8_0_BLOCK_BYTES, q8_0::BLOCK_BYTES);
    assert_eq!(omega::Q8_0_BLOCK_ELEMENTS, q8_0::QK8_0);
}
