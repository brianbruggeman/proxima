//! `Q2_K` Metal unpack evidence: the emitted helper assembles with Apple's
//! compiler and its element addressing matches the GGUF codec byte-for-byte.

#![cfg(target_os = "macos")]
#![allow(clippy::expect_used)]

use std::process::Command;

use proxima_gguf::quant::q2_k;
use proxima_tensor::test_support::Lcg;

fn q2k_element_host(block: &[u8], index: usize) -> f32 {
    let d = f32::from(half::f16::from_le_bytes([block[80], block[81]]));
    let dmin = f32::from(half::f16::from_le_bytes([block[82], block[83]]));
    let chunk = index / 128;
    let within = index % 128;
    let group = within / 32;
    let local = within % 32;
    let sub_block = chunk * 8 + group * 2 + usize::from(local >= 16);
    let scale_min = block[sub_block];
    let scale = d * f32::from(scale_min & 0x0f);
    let minimum = dmin * f32::from(scale_min >> 4);
    let level = (block[16 + chunk * 32 + local] >> (2 * group)) & 0x03;
    scale * f32::from(level) - minimum
}

#[test]
fn q2k_unpack_index_arithmetic_matches_the_gguf_codec_bit_exactly() {
    let mut compared = 0usize;
    for seed in 1..=16u64 {
        let mut generator = Lcg(seed);
        let mut block = vec![0u8; q2_k::BLOCK_BYTES];
        for byte in &mut block[..80] {
            *byte = (generator.next_unit() * 255.0) as u8;
        }
        block[80..82].copy_from_slice(&half::f16::from_f32(0.03125).to_le_bytes());
        block[82..84].copy_from_slice(&half::f16::from_f32(0.015625).to_le_bytes());
        let mut expected = vec![0.0f32; q2_k::QK_K];
        q2_k::dequantize_block(&block, &mut expected);

        for (index, reference) in expected.iter().enumerate() {
            let actual = q2k_element_host(&block, index);
            assert_eq!(
                actual.to_bits(),
                reference.to_bits(),
                "seed {seed} element {index}: Metal arithmetic twin {actual} vs codec {reference}"
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 16 * q2_k::QK_K);
}

#[test]
fn q2k_unpack_msl_assembles_with_the_real_metal_toolchain() {
    let source = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{}\nkernel void q2k_unpack_probe(device const uchar *block [[buffer(0)]], device float *out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{ out[gid] = q2k_element(block, gid); }}\n",
        omega::Q2K_UNPACK_MSL
    );
    let directory = tempfile::tempdir().expect("temp dir for Q2_K Metal source");
    let source_path = directory.path().join("q2k_unpack.metal");
    std::fs::write(&source_path, &source).expect("write Q2_K Metal source");
    let output = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-c"])
        .arg(&source_path)
        .arg("-o")
        .arg(directory.path().join("q2k_unpack.air"))
        .output()
        .expect("run the installed Metal compiler");
    assert!(
        output.status.success(),
        "Metal compile failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn q2k_block_geometry_matches_the_gguf_codec() {
    assert_eq!(omega::Q2K_BLOCK_BYTES, q2_k::BLOCK_BYTES);
    assert_eq!(omega::Q2K_BLOCK_ELEMENTS, q2_k::QK_K);
}
