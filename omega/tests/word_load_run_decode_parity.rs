//! The rendered run decode reads a run's plane fields as `uint` words, derived from the format's
//! `Plane` description (`layout_render::run_words`), and agrees bit for bit with
//! `proxima_numeric::decode_blocks` on real-valued encoded blocks.
//!
//! Worked example (IQ4_XS, `BlockLayout::IQ4_XS`, 136-byte block, code plane
//! `plane(4, 64, [digit(16, 8), digit(2, 4), digit(8, 128)])`, run of 16 elements at `slot`):
//!
//! - slot 0: `address = 64 + 0*8 + 0*4 + 0*128 = 64`, so first byte 8, shift 0. The 16 codes are the
//!   16 bytes 8..24 (the low nibbles of one 32-element sub-block's `qs` bytes).
//! - slot 16: `address = 64 + 0*8 + 1*4 = 68`, so first byte 8, shift 4. The same 16 bytes, high nibbles.
//! - slot 32: `address = 64 + 0 + 0 + 1*128 = 192`, first byte 24, shift 0.
//!
//! The first non-trivial digit is 16 wide, a multiple of the run (16), so no run crosses a carry and
//! lane `r` sits at bit `address + 8 * r`. The run needs `ceil((max_shift 4 + 15*8 + 4) / 32) = 4`
//! words `w0..w3` (`w_k` = bytes `first + 4k .. first + 4k + 3`, little endian, one `uint` load each
//! when the pointer is 4-aligned). Lane `r` is `(w[r / 4] >> (8 * (r % 4) + shift)) & 0xf`: for
//! slot 16 (shift 4) lane 5 is `(w1 >> 12) & 0xf`, the high nibble of byte 13. The shift is a run
//! time value here (0 or 4 across the block's runs) and the stride is whole bytes, so every lane
//! stays inside its word at the largest shift (`8*(r%4) + 4 + 4 <= 32`).
//!
//! A six-bit plane (`run(6, 16, 32)`, FP6_E3M2) has a constant shift 0 at every run start, so
//! lane `r` is at the constant bit `6 * r`: lane 1 at bits 6..12 straddles bytes 0 and 1 and lane 5
//! at bits 30..36 straddles words 0 and 1, joined as `((w0 >> 30) | (w1 << 2)) & 0x3f`.

#![cfg(all(feature = "metal", target_os = "macos"))]


use omega::msl::layout_render::{block_unpack_msl, level_block_element_msl, level_run_decode_msl};
use proxima_numeric::block_layout::BlockLayout;
use proxima_numeric::codec::Codec;
use proxima_numeric::{NumericFormat, decode_blocks, encode_blocks};

use probe::run_words;

const RUN: u64 = 16;
const BLOCKS: usize = 8;

type Check = Result<(), String>;

fn real_values(count: usize) -> Vec<f32> {
    (0..count)
        .map(|index| {
            let phase = index as f32;
            0.9 * (phase * 0.37).sin() + 0.4 * (phase * 1.31).cos() + 0.002 * (phase % 17.0)
        })
        .collect()
}

struct Encoded {
    bytes: Vec<u8>,
    decoded: Vec<f32>,
    block_bytes: usize,
    elements: usize,
}

fn encode(format: &NumericFormat) -> Option<Encoded> {
    let elements = usize::from(format.elements_per_block());
    let block_bytes = usize::from(format.bytes_per_block());
    let values = real_values(BLOCKS * elements);
    let mut bytes = vec![0u8; BLOCKS * block_bytes];
    if encode_blocks(format, &values, &mut bytes).is_err() {
        bytes = synthesized_blocks(format, block_bytes)?;
    }
    let mut decoded = vec![0f32; BLOCKS * elements];
    if decode_blocks(format, &bytes, &mut decoded).is_err() {
        gguf_decode(format, &bytes, &mut decoded)?;
    }
    Some(Encoded { bytes, decoded, block_bytes, elements })
}

/// Blocks for a format the encoder does not cover (a trained codebook, K-quant two-level fit):
/// every byte is a xorshift draw, so the code planes hold the full spread of codes a real tensor
/// does, and the f16 super-scale is a real weight-sized value, as a GGUF writer stores it. `None`
/// when the scale is not a whole f16 field, which a random byte could not make a real number.
fn synthesized_blocks(format: &NumericFormat, block_bytes: usize) -> Option<Vec<u8>> {
    let scale = format.scale()?;
    if scale.layout.width() != 16 || scale.pieces.is_some() {
        return None;
    }
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut bytes = Vec::with_capacity(BLOCKS * block_bytes);
    for block in 0..BLOCKS {
        let start = bytes.len();
        for _ in 0..block_bytes {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            bytes.push((state >> 32) as u8);
        }
        let value = half::f16::from_f32(0.004 + 0.0007 * block as f32).to_le_bytes();
        let at = start + usize::from(scale.byte_offset);
        bytes[at..at + 2].copy_from_slice(&value);
    }
    Some(bytes)
}

/// The block bytes at `pad` into the buffer, so `pad = 2` puts every block off a 4-byte boundary.
fn padded(bytes: &[u8], pad: usize) -> Vec<u8> {
    let mut out = vec![0u8; pad];
    out.extend_from_slice(bytes);
    out
}

fn element_probe(source: &str, name: &str, encoded: &Encoded, pad: usize) -> Result<Vec<u32>, String> {
    let total = BLOCKS * encoded.elements;
    let program = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{source}\nkernel void probe(device const uchar *blocks [[buffer(0)]], device float *out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{\n    device const uchar *block = blocks + {pad}u + (gid / {}u) * {}u;\n    out[gid] = {name}(block, gid % {}u);\n}}\n",
        encoded.elements, encoded.block_bytes, encoded.elements
    );
    run_words(&program, "probe", &[&padded(&encoded.bytes, pad)], total * 4, total)
}

fn run_probe(source: &str, function: &str, encoded: &Encoded, pad: usize) -> Result<Vec<u32>, String> {
    let total = BLOCKS * encoded.elements;
    let runs = total / RUN as usize;
    let program = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{source}\nkernel void probe(device const uchar *blocks [[buffer(0)]], device float *out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{\n    uint first = gid * {RUN}u;\n    device const uchar *block = blocks + {pad}u + (first / {}u) * {}u;\n    thread half tile[{RUN}];\n    {function}(block, first % {}u, tile);\n    for (uint lane = 0u; lane < {RUN}u; ++lane) {{ out[first + lane] = (float)tile[lane]; }}\n}}\n",
        encoded.elements, encoded.block_bytes, encoded.elements
    );
    run_words(&program, "probe", &[&padded(&encoded.bytes, pad)], total * 4, runs)
}

fn compare(label: &str, got: &[u32], want: impl Iterator<Item = u32>, pad: usize) -> Check {
    let want: Vec<u32> = want.collect();
    if got.len() != want.len() || want.is_empty() {
        return Err(format!("{label}: degenerate lengths got {} want {}", got.len(), want.len()));
    }
    let same = |left: &u32, right: &u32| left == right || (f32::from_bits(*left).is_nan() && f32::from_bits(*right).is_nan());
    let wrong = got.iter().zip(&want).filter(|(left, right)| !same(left, right)).count();
    if wrong > 0 {
        let first = got.iter().zip(&want).position(|(left, right)| !same(left, right)).unwrap_or(0);
        return Err(format!(
            "{label} pad {pad}: {wrong} of {} differ, first at {first}: got {:#010x} want {:#010x}",
            want.len(),
            got[first],
            want[first]
        ));
    }
    Ok(())
}

fn half_bits(value: f32) -> u32 {
    f32::from(half::f16::from_f32(value)).to_bits()
}

fn check_levels(name: &str, format: &NumericFormat) -> Result<Option<usize>, String> {
    let NumericFormat::Levels(levels) = format else {
        return Ok(None);
    };
    let Ok(element) = level_block_element_msl(levels, "el") else {
        return Ok(None);
    };
    let Some(encoded) = encode(format) else {
        return Ok(None);
    };
    let run_decode = level_run_decode_msl(levels, "run_decode", "el", RUN).ok();
    let run_source = run_decode.map(|decode| format!("{element}{decode}"));
    for pad in [0usize, 2] {
        let scalar = element_probe(&element, "el", &encoded, pad)?;
        compare(&format!("{name} element"), &scalar, encoded.decoded.iter().map(|value| value.to_bits()), pad)?;
        let Some(source) = &run_source else { continue };
        let runs = run_probe(source, "run_decode", &encoded, pad)?;
        compare(&format!("{name} run"), &runs, encoded.decoded.iter().map(|value| half_bits(*value)), pad)?;
    }
    Ok(Some(encoded.decoded.len()))
}

fn check_block(name: &str, format: &NumericFormat) -> Result<Option<usize>, String> {
    let NumericFormat::Block(layout) = format else {
        return Ok(None);
    };
    if layout.scale_runs.is_some() {
        return Ok(None);
    }
    let Some(encoded) = encode(format) else {
        return Ok(None);
    };
    let source = block_unpack_msl(layout, "bl");
    for pad in [0usize, 2] {
        let scalar = element_probe(&source, "bl", &encoded, pad)?;
        compare(&format!("{name} element"), &scalar, encoded.decoded.iter().map(|value| value.to_bits()), pad)?;
    }
    Ok(Some(encoded.decoded.len()))
}

#[test]
fn rendered_decode_equals_numeric_decode_for_every_covered_format() -> Check {
    let mut level_checked = Vec::new();
    let mut block_checked = Vec::new();
    let mut skipped = Vec::new();
    for format in &NumericFormat::ALL {
        let name = format.debug_name().unwrap_or("unnamed");
        match (check_levels(name, format)?, check_block(name, format)?) {
            (Some(_), _) => level_checked.push(name),
            (_, Some(_)) => block_checked.push(name),
            (None, None) => skipped.push(name),
        }
    }
    println!("level formats checked ({}): {level_checked:?}", level_checked.len());
    println!("block formats checked ({}): {block_checked:?}", block_checked.len());
    println!("not covered by the renderer or the encoder ({}): {skipped:?}", skipped.len());
    println!("formats checked: {}", level_checked.len() + block_checked.len());
    if level_checked.is_empty() || block_checked.is_empty() {
        return Err("degenerate: a family checked zero formats".to_string());
    }
    Ok(())
}

#[test]
fn iq4_xs_run_decode_is_word_loads_with_a_per_lane_shift_and_mask() -> Check {
    let NumericFormat::Levels(levels) = Codec::Iq4Xs else {
        return Err("IQ4_XS is a level format".to_string());
    };
    let text = level_run_decode_msl(&levels, "run_decode", "el", RUN).map_err(|refusal| refusal.to_string())?;
    let codes: Vec<&str> = text.lines().filter(|line| line.trim_start().starts_with("low_codes[")).collect();
    let words = text.matches("device const uint *").count();
    println!("IQ4_XS run decode: {words} word-load sites, {} code lines", codes.len());
    if words != 4 || !text.contains("low_w3") || text.contains("low_w4") {
        return Err(format!("expected exactly 4 word loads, found {words}:\n{text}"));
    }
    if codes.len() != RUN as usize || codes.iter().any(|line| line.contains("block[")) {
        return Err(format!("lane lines must shift and mask a word, not load a byte:\n{}", codes.join("\n")));
    }
    Ok(())
}

#[test]
fn fp6_e3m2_straddling_fields_decode_with_distinct_codes() -> Check {
    let format = NumericFormat::Block(BlockLayout::FP6_E3M2);
    let Some(encoded) = encode(&format) else {
        return Err("FP6_E3M2 encodes".to_string());
    };
    let plane = BlockLayout::FP6_E3M2.element.low;
    let straddling: Vec<usize> = (0..encoded.elements).filter(|index| plane.address(*index) % 8 + 6 > 8).collect();
    let block = &encoded.bytes[..encoded.block_bytes];
    let codes: std::collections::BTreeSet<u32> = straddling
        .iter()
        .map(|index| {
            let address = plane.address(*index);
            let pair = u32::from(block[address / 8]) | (u32::from(block[address / 8 + 1]) << 8);
            (pair >> (address % 8)) & 0x3f
        })
        .collect();
    println!("FP6_E3M2: {} straddling positions, {} distinct codes", straddling.len(), codes.len());
    if straddling.len() < 8 || codes.len() < 6 {
        return Err("degenerate fixture: too few distinct codes at straddling positions".to_string());
    }
    let source = block_unpack_msl(&BlockLayout::FP6_E3M2, "bl");
    for pad in [0usize, 2] {
        let got = element_probe(&source, "bl", &encoded, pad)?;
        compare("FP6_E3M2 element", &got, encoded.decoded.iter().map(|value| value.to_bits()), pad)?;
    }
    Ok(())
}

/// The incumbent dequantizer for the two formats `decode_blocks` has no decoder for (a codebook
/// grid and a signed grid); `proxima_gguf` is the gguf reader's own decode of the same bytes.
fn gguf_decode(format: &NumericFormat, bytes: &[u8], out: &mut [f32]) -> Option<()> {
    match format.name()? {
        "iq4_xs" => proxima_gguf::quant::iq4_xs::dequantize(bytes, out).ok(),
        "iq3_s" => proxima_gguf::quant::iq3_s::dequantize(bytes, out).ok(),
        _ => None,
    }
}

/// One compute dispatch on the default Metal device, every failure a `Err` string (the shared
/// `float_layout_parity/gpu.rs` harness panics through `expect`, which the workspace denies).
mod probe {
    use core::ffi::c_void;
    use core::ptr::NonNull;

    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
        MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
        MTLLanguageVersion, MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
    };

    /// Runs kernel `probe` of `source` over `threads` threads, `input` at buffer 0 and a
    /// `threads * 4`-byte word buffer at buffer 1, and returns the output words.
    pub fn run_words(source: &str, entry: &str, inputs: &[&[u8]], output_bytes: usize, threads: usize) -> Result<Vec<u32>, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let queue = device.newCommandQueue().ok_or("no command queue")?;
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version3_1);
        options.setMathMode(MTLMathMode::Safe);
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
            .map_err(|error| format!("metal rejected the source: {}\n{source}", error.localizedDescription()))?;
        let function = library.newFunctionWithName(&NSString::from_str(entry)).ok_or("kernel entry missing")?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| format!("no pipeline: {}", error.localizedDescription()))?;
        let mut buffers = Vec::new();
        for bytes in inputs {
            let pointer = NonNull::new(bytes.as_ptr().cast_mut().cast::<c_void>()).ok_or("empty input")?;
            // SAFETY: `bytes` is a live non-empty slice for the whole call and the device copies it once.
            let buffer = unsafe { device.newBufferWithBytes_length_options(pointer, bytes.len(), MTLResourceOptions::StorageModeShared) };
            buffers.push(buffer.ok_or("input buffer")?);
        }
        let output = device
            .newBufferWithLength_options(output_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or("output buffer")?;
        let command_buffer = queue.commandBuffer().ok_or("command buffer")?;
        let encoder = command_buffer.computeCommandEncoder().ok_or("compute encoder")?;
        encoder.setComputePipelineState(&pipeline);
        for (index, buffer) in buffers.iter().enumerate() {
            // SAFETY: each buffer outlives the command buffer's completion below.
            unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, index) };
        }
        // SAFETY: as above, for the output buffer.
        unsafe { encoder.setBuffer_offset_atIndex(Some(&output), 0, buffers.len()) };
        let width = threads.clamp(1, pipeline.maxTotalThreadsPerThreadgroup().min(256));
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: threads, height: 1, depth: 1 },
            MTLSize { width, height: 1, depth: 1 },
        );
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let mut bytes = vec![0u8; output_bytes];
        // SAFETY: the shared output buffer holds `output_bytes` initialized bytes and the command buffer completed.
        unsafe { core::ptr::copy_nonoverlapping(output.contents().as_ptr().cast::<u8>(), bytes.as_mut_ptr(), output_bytes) };
        Ok(bytes.as_chunks::<4>().0.iter().map(|chunk| u32::from_le_bytes(*chunk)).collect())
    }
}
