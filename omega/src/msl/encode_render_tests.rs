use alloc::string::String;
use alloc::vec::Vec;

use proxima_numeric::{FormatCodec, NumericFormat};

use super::encode_render::{encode_kernel_refusal, format_encode_kernel_msl};

#[cfg(all(feature = "metal", target_os = "macos"))]
use core::ffi::c_void;
#[cfg(all(feature = "metal", target_os = "macos"))]
use core::ptr::NonNull;
#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_foundation::NSString;
#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLCreateSystemDefaultDevice, MTLDevice, MTLLanguageVersion, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

const EMITTER_SOURCE: &str = include_str!("encode_render.rs");

/// K/V-shaped rows: per-head magnitudes that differ by orders of magnitude, a few outlier channels,
/// sign-symmetric spread, the shape of a projected and rotated attention row.
fn attention_rows(seed: u64, rows: usize, width: usize) -> Vec<f32> {
    let mut state = seed;
    let mut next = move || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    };
    let mut values = Vec::with_capacity(rows * width);
    for row in 0..rows {
        let head_scale = [0.04f32, 0.5, 3.0, 24.0][row % 4];
        for column in 0..width {
            let outlier = if column % 61 == 7 { 12.0 } else { 1.0 };
            values.push(next() * head_scale * outlier);
        }
    }
    values
}

/// Rows a cache must survive: all zeros, f16's largest finite values and the first overflow,
/// f16 and f32 subnormals, negative zero, a block whose scale underflows the stored width.
fn edge_rows(width: usize) -> Vec<f32> {
    let patterns: [&dyn Fn(usize) -> f32; 9] = [
        &|_| 0.0,
        &|index| if index % 2 == 0 { 65504.0 } else { -65504.0 },
        &|index| if index % 2 == 0 { 65520.0 } else { -70000.0 },
        &|index| (index as f32 + 1.0) * 5.960_464_5e-8,
        &|index| (index as f32 + 1.0) * 1.0e-40,
        &|index| if index == 3 { -0.0 } else { 0.0 },
        &|index| if index == 0 { 1.0e-30 } else { 1.0e-33 },
        &|index| if index == 5 { 3.0e38 } else { -1.0 },
        &|index| (index as f32 - 16.0) * 0.5,
    ];
    patterns.iter().flat_map(|pattern| (0..width).map(|index| pattern(index))).collect()
}

fn host_encode(format: &NumericFormat, values: &[f32]) -> Vec<u8> {
    let codec = FormatCodec::for_format(format).expect("the format has a host codec");
    let bytes = format.packed_len(values.len()).expect("whole blocks");
    let mut out = alloc::vec![0u8; bytes];
    codec.encode(values, &mut out).expect("the host encodes the rows");
    out
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn device_encode(format: &NumericFormat, values: &[f32]) -> Vec<u8> {
    let rendered = format_encode_kernel_msl(format, "probe_encode").expect("the format renders");
    let device = MTLCreateSystemDefaultDevice().expect("this box has a Metal device");
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version3_1);
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(&rendered.source), Some(&options))
        .unwrap_or_else(|error| panic!("metal rejected the source: {error:?}\n{}", rendered.source));
    let function = library
        .newFunctionWithName(&NSString::from_str(&rendered.entry))
        .expect("the encode kernel is in the library");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .unwrap_or_else(|error| panic!("metal could not build a pipeline: {error:?}"));
    let threads = values.len() / rendered.elements_per_thread;
    let out_bytes = threads * rendered.bytes_per_thread;
    // SAFETY: `values` outlives the call and the length is its own byte length.
    let input = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(values.as_ptr().cast_mut().cast::<c_void>()).expect("a slice pointer is never null"),
            values.len() * 4,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("the device allocates the input");
    let output = device
        .newBufferWithLength_options(out_bytes, MTLResourceOptions::StorageModeShared)
        .expect("the device allocates the output");
    let queue = device.newCommandQueue().expect("the device makes a queue");
    let command_buffer = queue.commandBuffer().expect("the queue makes a command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("an encoder");
    encoder.setComputePipelineState(&pipeline);
    // SAFETY: both buffers are live through the encode; 0 and 1 are the kernel's buffer slots.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&output), 0, 1);
    }
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize { width: threads, height: 1, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    // SAFETY: the command buffer completed and the buffer holds `out_bytes` bytes.
    unsafe { core::slice::from_raw_parts(output.contents().as_ptr().cast::<u8>(), out_bytes) }.to_vec()
}

fn cache_formats() -> [(&'static str, NumericFormat); 5] {
    [
        ("f16", NumericFormat::Float16),
        ("bf16", NumericFormat::BFloat16),
        ("bf8", NumericFormat::BFloat8),
        ("q8_0", NumericFormat::Q8_0),
        ("q4_0", NumericFormat::Q4_0),
    ]
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn a_rendered_encode_equals_the_host_codec_byte_for_byte_on_attention_rows_and_edge_rows() {
    let mut report = String::new();
    let mut differing_total = 0usize;
    for (label, format) in cache_formats() {
        let mut values = attention_rows(0x5eed, 512, 256);
        values.extend(attention_rows(0xface, 64, 128));
        values.extend(edge_rows(256));
        let host = host_encode(&format, &values);
        let device = device_encode(&format, &values);
        assert_eq!(host.len(), device.len(), "{label}: packed length");
        let differing = host.iter().zip(&device).filter(|(left, right)| left != right).count();
        let first = host.iter().zip(&device).position(|(left, right)| left != right);
        report.push_str(&format!(
            "{label}: {} values, {} bytes, byte-differing {differing}, first {first:?}\n",
            values.len(),
            host.len()
        ));
        differing_total += differing;
    }
    eprintln!("{report}");
    assert_eq!(differing_total, 0, "{report}");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn control_a_perturbed_input_row_changes_the_rendered_block_bytes() {
    let format = NumericFormat::Q4_0;
    let mut values = attention_rows(1, 4, 256);
    let before = device_encode(&format, &values);
    values[40] += 5.0;
    let after = device_encode(&format, &values);
    assert_ne!(before, after, "the rendered encode reads its input");
}

#[test]
fn a_format_with_no_rule_or_an_odd_width_is_refused_naming_the_property() {
    assert!(encode_kernel_refusal(&NumericFormat::Q8_0).is_ok());
    assert!(encode_kernel_refusal(&NumericFormat::Q4_1).is_err());
    assert!(encode_kernel_refusal(&NumericFormat::Bf4E2M1).is_err());
}

#[test]
fn the_encode_emitter_names_no_format() {
    for name in ["q8_0", "q4_0", "q8_1", "f16", "bf16", "bf8", "e5m2", "ggml", "llama"] {
        assert!(
            !EMITTER_SOURCE.to_ascii_lowercase().replace("floatlayout::f16", "").lines().filter(|line| !line.trim_start().starts_with("//")).any(|line| line.contains(name)),
            "the emitter source names {name}"
        );
    }
}

#[test]
fn a_rendered_block_encode_is_the_rule_with_its_constants_substituted() {
    let rendered = format_encode_kernel_msl(&NumericFormat::Q4_0, "probe_encode").expect("q4_0 renders");
    assert_eq!(rendered.elements_per_thread, 32);
    assert_eq!(rendered.bytes_per_thread, 18);
    assert!(rendered.source.contains("-8.0f"), "the signed-peak divisor is the rule's");
}
