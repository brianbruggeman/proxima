//! The grouped IQ3_S run decode (one grid word per four elements, the code and sign bytes read
//! once per run, signs XORed into the float sign bit) equals the per-element run decode bit for bit.
//!
//! The blocks are synthetic and exhaustive: run `n` of the corpus carries the sixteen sign bits
//! `n` (all 65536 patterns), its four grid indices are `4n..4n+4 mod 512` (every index, both
//! `qh` bit values, many times each), and the f16 scale and the eight sub-block scale nibbles cycle
//! through several values.

#![cfg(all(feature = "metal", target_os = "macos"))]

use omega::msl::layout_render::{level_block_element_msl, level_run_decode_msl, level_run_decode_per_element_msl};
use proxima_numeric::codec::Codec;
use proxima_numeric::NumericFormat;

use probe::run_words;

const RUN: u64 = 16;
const BLOCK_BYTES: usize = 110;
const BLOCK_ELEMENTS: usize = 256;
const RUNS_PER_BLOCK: usize = BLOCK_ELEMENTS / RUN as usize;
const BLOCKS: usize = 65536 / RUNS_PER_BLOCK;

type Check = Result<(), String>;

fn corpus() -> Vec<u8> {
    let scales = [0.0031f32, 0.017, 0.25, 1.5, 0.0009];
    let mut bytes = vec![0u8; BLOCKS * BLOCK_BYTES];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for block in 0..BLOCKS {
        let at = block * BLOCK_BYTES;
        bytes[at..at + 2].copy_from_slice(&half::f16::from_f32(scales[block % scales.len()]).to_le_bytes());
        for run in 0..RUNS_PER_BLOCK {
            let number = block * RUNS_PER_BLOCK + run;
            for group in 0..4 {
                let index = (number * 4 + group) % 512;
                let slot = run * 4 + group;
                bytes[at + 2 + slot] = (index & 0xff) as u8;
                bytes[at + 66 + slot / 8] &= !(1 << (slot % 8));
                bytes[at + 66 + slot / 8] |= ((index >> 8) as u8) << (slot % 8);
            }
            bytes[at + 74 + run * 2..at + 76 + run * 2].copy_from_slice(&(number as u16).to_le_bytes());
        }
        for byte in &mut bytes[at + 106..at + 110] {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = (state >> 32) as u8;
        }
    }
    bytes
}

fn padded(bytes: &[u8], pad: usize) -> Vec<u8> {
    let mut out = vec![0u8; pad];
    out.extend_from_slice(bytes);
    out
}

fn run_probe(source: &str, function: &str, bytes: &[u8], pad: usize) -> Result<Vec<u32>, String> {
    let total = BLOCKS * BLOCK_ELEMENTS;
    let program = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{source}\nkernel void probe(device const uchar *blocks [[buffer(0)]], device float *out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{\n    uint first = gid * {RUN}u;\n    device const uchar *block = blocks + {pad}u + (first / {BLOCK_ELEMENTS}u) * {BLOCK_BYTES}u;\n    thread half tile[{RUN}];\n    {function}(block, first % {BLOCK_ELEMENTS}u, tile);\n    for (uint lane = 0u; lane < {RUN}u; ++lane) {{ out[first + lane] = (float)tile[lane]; }}\n}}\n"
    );
    run_words(&program, "probe", &[&padded(bytes, pad)], total * 4, total / RUN as usize)
}

#[test]
fn iq3_s_grouped_run_decode_is_bit_identical_to_the_per_element_decode() -> Check {
    let NumericFormat::Levels(levels) = Codec::Iq3S else {
        return Err("IQ3_S is a level format".to_string());
    };
    let element = level_block_element_msl(&levels, "el").map_err(|refusal| refusal.to_string())?;
    let grouped = level_run_decode_msl(&levels, "run_new", "el", RUN).map_err(|refusal| refusal.to_string())?;
    let old = level_run_decode_per_element_msl(&levels, "run_old", "el", RUN).map_err(|refusal| refusal.to_string())?;
    if grouped == old || grouped.contains("for (uint run") || grouped.matches("el_grid[").count() != 4 {
        return Err(format!("the new decode is not the grouped form:\n{grouped}"));
    }
    let bytes = corpus();
    let source = format!("{element}{grouped}{old}");
    let mut compared = 0usize;
    let mut negative = 0usize;
    for pad in [0usize, 2] {
        let new = run_probe(&source, "run_new", &bytes, pad)?;
        let reference = run_probe(&source, "run_old", &bytes, pad)?;
        if new.len() != BLOCKS * BLOCK_ELEMENTS || new.len() != reference.len() {
            return Err(format!("degenerate lengths {} {}", new.len(), reference.len()));
        }
        if let Some(first) = new.iter().zip(&reference).position(|(left, right)| left != right) {
            return Err(format!("pad {pad}: differ at element {first}: new {:#010x} old {:#010x}", new[first], reference[first]));
        }
        compared += new.len();
        negative = new.iter().filter(|bits| **bits >> 31 == 1).count();
    }
    println!("IQ3_S grouped vs per-element: {compared} elements compared (2 paddings), {negative} negative per padding");
    if compared == 0 || negative == 0 || negative == BLOCKS * BLOCK_ELEMENTS {
        return Err("degenerate corpus".to_string());
    }
    Ok(())
}

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
