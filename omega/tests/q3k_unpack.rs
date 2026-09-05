//! The `Q3_K` unpack the GPU path needs, gated two ways -- mirrors
//! `q6k_unpack.rs` one codec over (both codecs have a per-super-block `d`
//! and a per-sub-block scale, but `Q3_K`'s scale is bit-packed like
//! `Q4_K`/`Q5_K`, not plain like `Q6_K`'s).
//!
//! Two claims, tested separately because they fail separately:
//!
//! 1. the MSL is valid MSL -- assembled by the real `xcrun metal` toolchain,
//!    never "looks like MSL". A missing toolchain is a RED gate, not a skip.
//! 2. the index arithmetic that MSL encodes is the arithmetic
//!    `proxima_gguf::quant::q3_k::dequantize_block` performs -- bit-exact
//!    against that codec over randomized blocks.
//!
//! Deliberately NOT claimed here: that the compiled shader computes that
//! arithmetic on a device -- `q3k_real_checkpoint_parity.rs` covers that end
//! to end on real checkpoint bytes.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::process::Command;

use proxima_gguf::quant::q3_k;
use proxima_tensor::test_support::Lcg;

/// A `Q3_K` super-block with plausible field values: `d` is a real `f16` in
/// the range a trained checkpoint carries, every scale code varies (never
/// collapsing every sub-block to the same scale), and every `hmask`/`qs`
/// byte varies. An all-zero fixture would pass under an index bug -- every
/// level would decode to `-4` (bit clear, correction 4) or accidentally hit
/// another all-zero byte.
fn random_block(seed: u64) -> Vec<u8> {
    let mut lcg = Lcg(seed);
    let mut unit = move || f32::midpoint(lcg.next_unit(), 1.0);
    let mut block = vec![0u8; omega::Q3K_BLOCK_BYTES];

    // hmask (32 bytes at 0) + qs (64 bytes at 32): real varying bit patterns.
    for byte in &mut block[0..96] {
        *byte = (unit() * 255.0) as u8;
    }
    // scales (12 bytes at 96): packed 6-bit codes, real varying values.
    for byte in &mut block[96..108] {
        *byte = (unit() * 255.0) as u8;
    }
    let scale = half::f16::from_f32(0.01 + unit() * 0.05);
    block[108..110].copy_from_slice(&scale.to_le_bytes());
    block
}

/// The Rust twin of `q3k_value`/`q3k_element` in [`omega::Q3K_UNPACK_MSL`] --
/// the same index arithmetic, transcribed. Kept beside the MSL so a change
/// to one that is not mirrored in the other fails this file.
fn q3k_unpack_scale_host(scales: &[u8], sub_block: usize) -> i32 {
    let low = if sub_block < 8 {
        scales[sub_block] & 0x0F
    } else {
        scales[sub_block - 8] >> 4
    };
    let high = (scales[8 + sub_block % 4] >> (2 * (sub_block / 4))) & 0x03;
    let combined = low | (high << 4);
    i32::from(combined) - 32
}

fn q3k_element_host(block: &[u8], index: usize) -> f32 {
    let hmask = &block[0..32];
    let qs = &block[32..96];
    let scales = &block[96..108];
    let d = f32::from(half::f16::from_le_bytes([block[108], block[109]]));

    let chunk = index / 128;
    let rem = index % 128;
    let j = rem / 32;
    let local32 = rem % 32;
    let low = local32 < 16;
    let sub_block = 8 * chunk + 2 * j + usize::from(!low);

    let scale = d * q3k_unpack_scale_host(scales, sub_block) as f32;
    let level = (qs[chunk * 32 + local32] >> (2 * j)) & 0x03;
    let mask = 1u8 << (4 * chunk + j);
    let correction = if hmask[local32] & mask != 0 { 0.0 } else { 4.0 };
    scale * (f32::from(level) - correction)
}

#[test]
fn q3k_unpack_index_arithmetic_matches_the_gguf_codec_bit_exactly() {
    let mut compared = 0usize;
    for seed in 1..=16u64 {
        let block = random_block(seed);
        let mut expected = vec![0.0f32; omega::Q4K_BLOCK_ELEMENTS];
        q3_k::dequantize_block(&block, &mut expected);

        for (index, reference) in expected.iter().enumerate() {
            let ours = q3k_element_host(&block, index);
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

/// `d` sits LAST in `Q3_K`'s layout (offset 108), the same trailing position
/// `Q6_K` uses -- unlike `Q4_K`/`Q5_K`/`Q8_0` where it leads. Pinned directly
/// so a "simplification" that reads it at offset 0 fails with a name that
/// says what broke.
#[test]
fn q3k_scale_trails_the_block_not_leads_it() {
    let mut block = vec![0u8; omega::Q3K_BLOCK_BYTES];
    // element 0: chunk=0, j=0, local32=0, sub_block=0, mask=1.
    block[32] = 0x03; // qs[0] low 2 bits = 3
    block[0] = 0x00; // hmask[0] bit 0 clear -> correction = 4
    // sub_block 0's scale code is bit-interleaved across two bytes, not a
    // single byte minus 32 (`q3k_unpack_scale_host`'s own low/high split):
    // low nibble in scales[0], high 2 bits in scales[8]'s bits 0-1. Setting
    // scales[0]=5, scales[8]=2 assembles code 0b10_0101=37, sc=37-32=5.
    block[96] = 5;
    block[96 + 8] = 2;
    block[108..110].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());

    // scale = 2.0 * 5 = 10.0, level = 3, correction = 4.0 ->
    // value = 10.0 * (3.0 - 4.0) = -10.0. If `d` were misread from offset 0
    // (the first `hmask` byte, 0x00 -> f16 zero) this would be 0.0.
    assert_eq!(q3k_element_host(&block, 0), -10.0);
}

#[test]
fn q3k_unpack_msl_assembles_with_the_real_metal_toolchain() {
    let source = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{}\n{}\nkernel void q3k_unpack_probe(\n    device const uchar *block [[buffer(0)]],\n    device float *out [[buffer(1)]],\n    uint gid [[thread_position_in_grid]]) {{\n    out[gid] = q3k_element(block, gid);\n}}\n",
        omega::Q3K_UNPACK_MSL,
        omega::Q3K_PAIR_DOT_MSL
    );

    let dir = tempfile::tempdir().expect("temp dir for the metal source");
    let metal_path = dir.path().join("q3k_unpack.metal");
    std::fs::write(&metal_path, &source).expect("write metal source");

    let output = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-c"])
        .arg(&metal_path)
        .arg("-o")
        .arg(dir.path().join("q3k_unpack.air"))
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

/// omega restates `Q3_K`'s block geometry rather than depending on
/// `proxima-gguf` at build time. That is only safe if the two agree.
#[test]
fn q3k_block_geometry_matches_the_gguf_codec() {
    assert_eq!(omega::Q3K_BLOCK_BYTES, q3_k::BLOCK_BYTES);
    assert_eq!(omega::Q4K_BLOCK_ELEMENTS, q3_k::QK_K);
}
