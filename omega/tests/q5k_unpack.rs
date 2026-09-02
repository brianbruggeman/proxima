//! The `Q5_K` unpack the GPU path needs, gated two ways -- mirrors
//! `q6k_unpack.rs` one codec over.
//!
//! `blk.{n}.attn_v.weight` and `blk.{n}.ffn_down.weight` (4 layers each) are
//! the only `Q5_K` tensors `openchat-3.5-1210.Q4_K_S.gguf` carries. Isolated
//! per-op profiling (ROW 92) measured those 8 ops at ~56.86ms of a
//! ~150ms diagnostic-scale decode step -- over half the whole step's GPU
//! time, once they are told apart from the 56 already-fast `Q4_K` ops
//! sharing their two families -- because the loader still dequantized them
//! to f32 before Metal ever saw them. This unpack is what lets them stay
//! packed all the way to the kernel.
//!
//! Two claims, tested separately because they fail separately:
//!
//! 1. the MSL is valid MSL — assembled by the real `xcrun metal` toolchain,
//!    never "looks like MSL". A missing toolchain is a RED gate, not a skip.
//! 2. the index arithmetic that MSL encodes is the arithmetic
//!    `proxima_gguf::quant::q5_k::dequantize_block` performs — bit-exact
//!    against that codec over randomized blocks.
//!
//! Deliberately NOT claimed here: that the compiled shader computes that
//! arithmetic on a device — `metal_parity.rs`'s
//! `metal_matmul_on_packed_q5k_weights_matches_the_dequantized_f32_cpu_path`
//! covers that end to end.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::process::Command;

use proxima_gguf::quant::q5_k;
use proxima_tensor::test_support::Lcg;

/// A `Q5_K` super-block with plausible field values: `d`/`dmin` are real
/// `f16`s in the range a trained checkpoint carries, every scale byte
/// varies, and `qh`/`qs` both vary bit-for-bit. An all-zero fixture would
/// pass under an index bug (every element would decode to `-dmin*min`
/// regardless of which byte/bit a wrong-lane read actually touched).
fn random_block(seed: u64) -> Vec<u8> {
    let mut lcg = Lcg(seed);
    let mut unit = move || f32::midpoint(lcg.next_unit(), 1.0);
    let mut block = vec![0u8; omega::Q5K_BLOCK_BYTES];

    let d = half::f16::from_f32(0.01 + unit() * 0.05);
    let dmin = half::f16::from_f32(0.005 + unit() * 0.02);
    block[0..2].copy_from_slice(&d.to_le_bytes());
    block[2..4].copy_from_slice(&dmin.to_le_bytes());
    for byte in &mut block[4..176] {
        *byte = (unit() * 255.0) as u8;
    }
    block
}

/// The Rust twin of `q5k_value`/`q5k_element` in [`omega::Q5K_UNPACK_MSL`] —
/// the same index arithmetic, transcribed. Kept beside the MSL so a change
/// to one that is not mirrored in the other fails this file.
fn q5k_scale_min_host(scales: &[u8], sub_block: usize) -> (u8, u8) {
    if sub_block < 4 {
        (scales[sub_block] & 63, scales[sub_block + 4] & 63)
    } else {
        let scale = (scales[sub_block + 4] & 0x0F) | ((scales[sub_block - 4] >> 6) << 4);
        let min = (scales[sub_block + 4] >> 4) | ((scales[sub_block] >> 6) << 4);
        (scale, min)
    }
}

fn q5k_element_host(block: &[u8], index: usize) -> f32 {
    let d = f32::from(half::f16::from_le_bytes([block[0], block[1]]));
    let dmin = f32::from(half::f16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];

    let chunk = index / 64;
    let within = index % 64;
    let low = within < 32;
    let offset = within % 32;
    let sub_block = 2 * chunk + usize::from(!low);

    let (scale_code, min_code) = q5k_scale_min_host(scales, sub_block);
    let scale = d * f32::from(scale_code);
    let minimum = dmin * f32::from(min_code);
    let mask: u8 = if low {
        1u8 << (2 * chunk)
    } else {
        2u8 << (2 * chunk)
    };

    let qs_byte = qs[chunk * 32 + offset];
    let nibble = if low { qs_byte & 0x0F } else { qs_byte >> 4 };
    let high = if qh[offset] & mask != 0 { 16.0 } else { 0.0 };
    scale * (f32::from(nibble) + high) - minimum
}

#[test]
fn q5k_unpack_index_arithmetic_matches_the_gguf_codec_bit_exactly() {
    let mut compared = 0usize;
    for seed in 1..=16u64 {
        let block = random_block(seed);
        let mut expected = vec![0.0f32; omega::Q4K_BLOCK_ELEMENTS];
        q5_k::dequantize_block(&block, &mut expected);

        for (index, reference) in expected.iter().enumerate() {
            let ours = q5k_element_host(&block, index);
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

/// The trap `Q5_K`'s own module doc calls out as easiest to get silently
/// wrong: `qh` is indexed by within-sub-block `offset` alone (0..32),
/// never by `chunk` -- two elements in DIFFERENT chunks but the SAME local
/// offset read different BITS of the SAME `qh` byte. Pinned directly with
/// a fixture where `qh[0]` carries a bit for BOTH chunk 0's low half
/// (element 0) and chunk 1's low half (element 128), and only one of the
/// two bits is set.
#[test]
fn q5k_qh_bit_is_selected_by_mask_not_by_a_second_qh_index() {
    let mut block = vec![0u8; omega::Q5K_BLOCK_BYTES];
    block[0..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes()); // d
    block[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes()); // dmin=0
    // sub_block 0 (element 0's, direct branch): scale code = 1.
    block[4] = 1;
    // sub_block 4 (element 128's, interleaved branch): scale code = 5 --
    // deliberately NONZERO, so a wrong (incorrectly-set) high bit at
    // element 128 would show up as 10.0*16.0=160.0, not silently vanish
    // into a zero-scale degenerate case.
    block[12] = 5;
    // qh[0] bit 0 set (chunk 0 low mask = 1<<0 = 1) -> element 0 gets +16.
    // bit 4 (chunk 2 low mask = 1<<4 = 16) NOT set -> element 128 stays +0.
    block[16] = 0b0000_0001;

    let element_0 = q5k_element_host(&block, 0);
    let element_128 = q5k_element_host(&block, 128);
    assert_eq!(
        element_0,
        2.0 * 1.0 * 16.0,
        "element 0 must read qh[0] bit 0 as set"
    );
    assert_eq!(
        element_128, 0.0,
        "element 128 must read qh[0]'s bit 4 (via its own mask), NOT a second qh byte, and see it unset"
    );
}

#[test]
fn q5k_unpack_msl_assembles_with_the_real_metal_toolchain() {
    let source = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{}\nkernel void q5k_unpack_probe(\n    device const uchar *block [[buffer(0)]],\n    device float *out [[buffer(1)]],\n    uint gid [[thread_position_in_grid]]) {{\n    out[gid] = q5k_element(block, gid);\n}}\n",
        omega::Q5K_UNPACK_MSL
    );

    let dir = tempfile::tempdir().expect("temp dir for the metal source");
    let metal_path = dir.path().join("q5k_unpack.metal");
    std::fs::write(&metal_path, &source).expect("write metal source");

    let output = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-c"])
        .arg(&metal_path)
        .arg("-o")
        .arg(dir.path().join("q5k_unpack.air"))
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

/// omega restates `Q5_K`'s block geometry rather than depending on
/// `proxima-gguf` at build time. That is only safe if the two agree.
#[test]
fn q5k_block_geometry_matches_the_gguf_codec() {
    assert_eq!(omega::Q5K_BLOCK_BYTES, q5_k::BLOCK_BYTES);
    assert_eq!(omega::Q4K_BLOCK_ELEMENTS, q5_k::QK_K);
}
