#[cfg(all(feature = "metal", target_os = "macos"))]
use alloc::string::String;
#[cfg(all(feature = "metal", target_os = "macos"))]
use alloc::vec::Vec;

#[cfg(all(feature = "metal", target_os = "macos"))]
use proxima_numeric::encode_rule::{CodeRounding, EncodeRule, ScaleRounding};
#[cfg(all(feature = "metal", target_os = "macos"))]
use proxima_numeric::BlockLayout;
use proxima_numeric::NumericFormat;
#[cfg(all(feature = "metal", target_os = "macos"))]
use proxima_numeric::FormatCodec;

#[cfg(all(feature = "metal", target_os = "macos"))]
use super::encode_render::{RenderedEncode, block_encode_kernel_by_rule};
use super::encode_render::{encode_kernel_refusal, format_encode_kernel_msl};

#[expect(dead_code, reason = "the numeric layout tests use the dequantizers this crate's tests do not")]
mod llama {
    include!("../../../proxima-numeric/tests/support/llama_fp4_reference.rs");
}

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
const NEUTRAL_SOURCE: &str = include_str!("../encode_render.rs");

/// K/V-shaped rows: per-head magnitudes that differ by orders of magnitude, a few outlier channels,
/// sign-symmetric spread, the shape of a projected and rotated attention row.
#[cfg(all(feature = "metal", target_os = "macos"))]
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
#[cfg(all(feature = "metal", target_os = "macos"))]
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
    patterns.iter().flat_map(|pattern| (0..width).map(pattern)).collect()
}

#[cfg(all(feature = "metal", target_os = "macos"))]
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
    device_run(&rendered, values)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn device_run(rendered: &RenderedEncode, values: &[f32]) -> Vec<u8> {
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

#[cfg(all(feature = "metal", target_os = "macos"))]
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
            ![EMITTER_SOURCE, NEUTRAL_SOURCE].iter().any(|source| source.to_ascii_lowercase().replace("floatlayout::f16", "").lines().filter(|line| !line.trim_start().starts_with("//")).any(|line| line.contains(name))),
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

fn fp4_formats() -> [(&'static str, NumericFormat); 2] {
    [("block of 32 under a shared exponent", NumericFormat::Mxfp4), ("runs of 16 under an E4M3 scale", NumericFormat::Nvfp4)]
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn kv_shaped_values() -> Vec<f32> {
    let mut values = attention_rows(0x5eed, 512, 256);
    values.extend(attention_rows(0xface, 64, 128));
    values.extend(edge_rows(256));
    values
}

#[test]
fn a_float_coded_block_format_has_a_rendered_encode_whether_it_packs_scale_runs_or_not() {
    for (label, format) in fp4_formats() {
        assert!(encode_kernel_refusal(&format).is_ok(), "{label}");
    }
}

/// The smallest peak whose fitted scale is a normal `f32`: below it (an all-zero block included,
/// whose exponent-only scale stores as the lowest power of two) the scale is a subnormal, which the
/// device flushes to zero where the host keeps it, so those blocks (a peak under 1e-37, zero for
/// attention) encode as all-zero codes on the device, and only the sign of a negative zero differs.
#[cfg(all(feature = "metal", target_os = "macos"))]
const DEVICE_SUBNORMAL_SCALE_PEAK: f32 = 8.0 * f32::MIN_POSITIVE;

/// Bytes that differ between a host and a device encode, split into those in blocks whose peak the
/// device flushes and the rest: `(differing in flushed blocks, differing elsewhere, flushed blocks)`.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn differing_bytes(host: &[u8], device: &[u8], values: &[f32], format: &NumericFormat) -> (usize, usize, usize) {
    let elements = usize::from(format.elements_per_block());
    let bytes = usize::from(format.bytes_per_block());
    let run_elements = format.block_layout().map_or(elements, |layout| usize::from(layout.run_elements()));
    let (mut flushed_differing, mut other_differing, mut flushed_blocks) = (0usize, 0usize, 0usize);
    for ((block_values, left), right) in values.chunks(elements).zip(host.chunks(bytes)).zip(device.chunks(bytes)) {
        let differing = left.iter().zip(right).filter(|(a, b)| a != b).count();
        let flushed = block_values.chunks(run_elements).any(|run| {
            let peak = run.iter().fold(0.0f32, |peak, value| peak.max(value.abs()));
            peak < DEVICE_SUBNORMAL_SCALE_PEAK
        });
        flushed_blocks += usize::from(flushed && differing > 0);
        if flushed {
            flushed_differing += differing;
        } else {
            other_differing += differing;
        }
    }
    (flushed_differing, other_differing, flushed_blocks)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn a_rendered_fp4_encode_equals_the_host_codec_byte_for_byte_on_kv_rows_and_edge_rows() {
    let values = kv_shaped_values();
    let mut report = String::new();
    let mut other_total = 0usize;
    for (label, format) in fp4_formats() {
        let host = host_encode(&format, &values);
        let device = device_encode(&format, &values);
        assert_eq!(host.len(), device.len(), "{label}: packed length");
        let (flushed, other, flushed_blocks) = differing_bytes(&host, &device, &values, &format);
        report.push_str(&format!(
            "{label}: {} values, {} bytes, byte-differing {other} outside subnormal-scale blocks, {flushed} inside {flushed_blocks} of them\n",
            values.len(),
            host.len()
        ));
        other_total += other;
    }
    eprintln!("{report}");
    assert_eq!(other_total, 0, "{report}");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn a_rendered_mxfp4_encode_equals_llama_quantize_row_mxfp4_ref_on_kv_rows() {
    let mut values = attention_rows(0x5eed, 512, 256);
    values.extend(attention_rows(0xface, 64, 128));
    let reference: Vec<u8> = values.chunks(32).flat_map(llama::llama_quantize_block_mxfp4).collect();

    let device = device_encode(&NumericFormat::Mxfp4, &values);

    let differing = device.iter().zip(&reference).filter(|(left, right)| left != right).count();
    eprintln!("mxfp4 device vs llama: {} values, {} bytes, {differing} differing", values.len(), device.len());
    assert_eq!((device.len(), differing), (reference.len(), 0));
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn a_rendered_away_from_zero_mxfp4_encode_equals_the_host_codec_with_the_same_rule() {
    let rule = EncodeRule {
        scale_rounding: ScaleRounding::PowerOfTwoCeil,
        rounding: CodeRounding::NearestEvenQuotient,
        ..BlockLayout::MXFP4.encode_rule().expect("mxfp4 has a rule")
    };
    let values = kv_shaped_values();
    let host = {
        let codec = FormatCodec::coded(&BlockLayout::MXFP4, rule).expect("the rule fits the layout");
        let mut out = alloc::vec![0u8; codec.packed_len(values.len()).expect("whole blocks")];
        codec.encode(&values, &mut out).expect("the host encodes the rows");
        out
    };
    let rendered = block_encode_kernel_by_rule(&BlockLayout::MXFP4, &rule, "probe_encode").expect("the rule renders");

    let device = device_run(&rendered, &values);

    let (flushed, other, flushed_blocks) = differing_bytes(&host, &device, &values, &NumericFormat::Mxfp4);
    eprintln!(
        "mxfp4 away-from-zero device vs host: {} values, {} bytes, {other} differing outside subnormal-scale blocks, {flushed} inside {flushed_blocks} of them",
        values.len(),
        host.len()
    );
    assert_eq!((device.len(), other), (host.len(), 0));
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn control_the_ocp_and_the_away_from_zero_rules_render_different_bytes() {
    let values = attention_rows(0x5eed, 16, 256);
    let ocp = device_encode(&NumericFormat::Mxfp4, &values);
    let rule = EncodeRule {
        scale_rounding: ScaleRounding::PowerOfTwoCeil,
        rounding: CodeRounding::NearestEvenQuotient,
        ..BlockLayout::MXFP4.encode_rule().expect("mxfp4 has a rule")
    };
    let rendered = block_encode_kernel_by_rule(&BlockLayout::MXFP4, &rule, "probe_encode").expect("the rule renders");

    let away = device_run(&rendered, &values);

    assert_ne!(ocp, away, "the scale rounding is read from the rule");
}

#[test]
fn a_rendered_float_encode_is_the_rule_with_its_constants_substituted() {
    let rendered = format_encode_kernel_msl(&NumericFormat::Mxfp4, "probe_encode").expect("mxfp4 renders");
    assert_eq!((rendered.elements_per_thread, rendered.bytes_per_thread), (32, 17));
    assert!(rendered.source.contains("probe_encode_grid[16]"), "the code grid is the layout's decoded values");
    let runs = format_encode_kernel_msl(&NumericFormat::Nvfp4, "probe_encode").expect("nvfp4 renders");
    assert_eq!((runs.elements_per_thread, runs.bytes_per_thread), (64, 36));
    assert!(runs.source.contains("run < 4u"), "the four scale runs are the layout's");
}
