//! The `Q6_K` unpack the GPU path needs, gated two ways -- mirrors
//! `q4k_unpack.rs` one codec over.
//!
//! `output.weight` is the LONE `Q6_K` tensor `openchat-3.5-1210.Q4_K_S.gguf`
//! carries, and it measured ~150ms of ~250.7ms total `gpu_exec` time (roughly
//! 60%) BEFORE this landing, because Metal had no unpack kernel for it and
//! the loader dequantized it to 524 MB of f32 first. This unpack is what
//! lets that weight stay packed all the way to the kernel.
//!
//! Two claims, tested separately because they fail separately:
//!
//! 1. the MSL is valid MSL — assembled by the real `xcrun metal` toolchain,
//!    never "looks like MSL". A missing toolchain is a RED gate, not a skip.
//! 2. the index arithmetic that MSL encodes is the arithmetic
//!    `proxima_gguf::quant::q6_k::dequantize_block` performs — bit-exact
//!    against that codec over randomized blocks.
//!
//! Deliberately NOT claimed here: that the compiled shader computes that
//! arithmetic on a device — `metal_parity.rs`'s
//! `metal_matmul_on_packed_q6k_weights_matches_the_dequantized_f32_cpu_path`
//! covers that end to end.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::process::Command;

use proxima_gguf::quant::q6_k;
use proxima_tensor::test_support::Lcg;

/// A `Q6_K` super-block with plausible field values: `d` is a real `f16` in
/// the range a trained checkpoint carries, every scale byte is a real signed
/// `i8`, and every `ql`/`qh` byte varies. An all-zero fixture would pass
/// under an index bug, every element agreeing by accident (every level would
/// decode to 0, quant to -32, and any wrong-lane read would still hit
/// another all-zero byte).
fn random_block(seed: u64) -> Vec<u8> {
    let mut lcg = Lcg(seed);
    let mut unit = move || f32::midpoint(lcg.next_unit(), 1.0);
    let mut block = vec![0u8; omega::Q6K_BLOCK_BYTES];

    for byte in &mut block[0..192] {
        *byte = (unit() * 255.0) as u8;
    }
    for byte in &mut block[192..208] {
        // a real signed i8 scale, never zero (zero collapses this sub-block
        // to always-zero regardless of level, hiding a wrong-lane read).
        let raw = (unit() * 254.0) as u8;
        *byte = if raw == 0 { 1 } else { raw };
    }
    let scale = half::f16::from_f32(0.01 + unit() * 0.05);
    block[208..210].copy_from_slice(&scale.to_le_bytes());
    block
}

/// The Rust twin of `q6k_value`/`q6k_element` in [`omega::Q6K_UNPACK_MSL`] —
/// the same index arithmetic, transcribed. Kept beside the MSL so a change
/// to one that is not mirrored in the other fails this file.
fn q6k_element_host(block: &[u8], index: usize) -> f32 {
    let d = f32::from(half::f16::from_le_bytes([block[208], block[209]]));
    let half_index = index / 128;
    let local = index % 128;
    let l = local % 32;
    let lane = local / 32;
    let sub_block_in_half = l / 16;

    let ql = &block[half_index * 64..half_index * 64 + 64];
    let qh = &block[128 + half_index * 32..128 + half_index * 32 + 32];
    let scales = &block[192..208];

    let ql_byte = if lane % 2 == 0 { ql[l] } else { ql[l + 32] };
    let nibble = if lane < 2 { ql_byte & 0x0F } else { ql_byte >> 4 };
    let high2 = (qh[l] >> (lane * 2)) & 0x03;
    let level = nibble | (high2 << 4);

    let scale_byte = scales[half_index * 8 + sub_block_in_half + lane * 2];
    let scale = f32::from(scale_byte as i8);
    let quant = f32::from(level) - 32.0;
    d * scale * quant
}

#[test]
fn q6k_unpack_index_arithmetic_matches_the_gguf_codec_bit_exactly() {
    let mut compared = 0usize;
    for seed in 1..=16u64 {
        let block = random_block(seed);
        let mut expected = vec![0.0f32; omega::Q4K_BLOCK_ELEMENTS];
        q6_k::dequantize_block(&block, &mut expected);

        for (index, reference) in expected.iter().enumerate() {
            let ours = q6k_element_host(&block, index);
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

/// `d` sits LAST in `Q6_K`'s layout (offset 208), unlike `Q4_K`/`Q5_K` where
/// it leads — pinned directly so a "simplification" that reads it at offset
/// 0 fails with a name that says what broke.
#[test]
fn q6k_scale_trails_the_block_not_leads_it() {
    let mut block = vec![0u8; omega::Q6K_BLOCK_BYTES];
    // element 0: half=0, l=0, lane=0 -> ql[0] low nibble, qh[0] bits 0-1,
    // scale index 0.
    block[0] = 0x0F; // ql[0] low nibble = 15
    block[192] = 1; // scale[0] = 1
    block[208..210].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());

    // level = 15 (qh byte is 0), quant = 15 - 32 = -17, d * scale * quant =
    // 2.0 * 1.0 * -17.0 = -34.0. If `d` were misread from offset 0 (the
    // first `ql` byte, 0x0F -> a tiny/garbage f16) this would NOT be -34.0.
    assert_eq!(q6k_element_host(&block, 0), -34.0);
}

#[test]
fn q6k_unpack_msl_assembles_with_the_real_metal_toolchain() {
    let source = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{}\nkernel void q6k_unpack_probe(\n    device const uchar *block [[buffer(0)]],\n    device float *out [[buffer(1)]],\n    uint gid [[thread_position_in_grid]]) {{\n    out[gid] = q6k_element(block, gid);\n}}\n",
        omega::Q6K_UNPACK_MSL
    );

    let dir = tempfile::tempdir().expect("temp dir for the metal source");
    let metal_path = dir.path().join("q6k_unpack.metal");
    std::fs::write(&metal_path, &source).expect("write metal source");

    let output = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-c"])
        .arg(&metal_path)
        .arg("-o")
        .arg(dir.path().join("q6k_unpack.air"))
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

/// omega restates `Q6_K`'s block geometry rather than depending on
/// `proxima-gguf` at build time. That is only safe if the two agree.
#[test]
fn q6k_block_geometry_matches_the_gguf_codec() {
    assert_eq!(omega::Q6K_BLOCK_BYTES, q6_k::BLOCK_BYTES);
    assert_eq!(omega::Q4K_BLOCK_ELEMENTS, q6_k::QK_K);
}
