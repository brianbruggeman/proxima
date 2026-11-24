use super::pipeline_disk_cache::{pipeline_from_function, probe_function};
use super::*;

use crate::msl::layout_render::hardware_type;
use crate::msl::{DecodeSource, EmitKnobs, GridSpec, tiled_decode};
use proxima_numeric::float_layout::FloatLayout;
use proxima_numeric::{NumericFormat, PackedRows};

#[derive(Default, Debug)]
struct ConversionMismatches {
    subnormal_range: usize,
    overflow: usize,
    nan: usize,
    other: usize,
    first_other: Option<(u16, u16, u16)>,
}

impl ConversionMismatches {
    fn total(&self) -> usize {
        self.subnormal_range + self.overflow + self.nan + self.other
    }
}

fn widening_decode_kernel(decode_source: &str, function: &str, scalar: &str) -> Kernel {
    let source = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{decode_source}\n\
        kernel void widen_every_code(device const uchar *codes [[buffer(0)]], device {scalar} *out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{\n\
            thread {scalar} run[16];\n\
            {function}(codes, gid * 16u, run);\n\
            for (uint i = 0u; i < 16u; ++i) {{ out[gid * 16u + i] = run[i]; }}\n\
        }}\n"
    );
    Kernel {
        source,
        entry: String::from("widen_every_code"),
        bindings: Vec::new(),
        grid: GridSpec {
            threads: 65_536 / 16,
            threadgroup_width: None,
            depth: 1,
            grid2d: None,
        },
    }
}

fn run_widening_decode(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    stored_bytes: &[u8],
) -> Vec<u16> {
    let options = MTLResourceOptions::StorageModeShared;
    let input = device
        .newBufferWithLength_options(stored_bytes.len(), options)
        .expect("device must allocate the stored codes");
    unsafe {
        core::ptr::copy_nonoverlapping(
            stored_bytes.as_ptr(),
            input.contents().as_ptr().cast::<u8>(),
            stored_bytes.len(),
        );
    }
    let output = device
        .newBufferWithLength_options(65_536 * size_of::<u16>(), options)
        .expect("device must allocate the decoded codes");
    let command_buffer = queue
        .commandBuffer()
        .expect("queue must hand out a command buffer");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("command buffer must open a compute encoder");
    encoder.setComputePipelineState(pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&output), 0, 1);
    }
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize { width: 65_536 / 16, height: 1, depth: 1 },
        MTLSize { width: 64, height: 1, depth: 1 },
    );
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    let contents = output.contents().as_ptr().cast::<u16>();
    (0..65_536)
        .map(|index| unsafe { contents.add(index).read() })
        .collect()
}

#[test]
fn the_widening_decode_matches_the_host_conversion_for_every_stored_code() {
    let Ok((device, queue)) = device_and_queue() else {
        return;
    };
    let target = EmitKnobs::default().grouped_gemm_formats().dequant_out;
    let (decode, stored) = crate::msl::Codec::ALL
        .into_iter()
        .filter_map(tiled_decode)
        .find_map(|decode| match decode.source {
            DecodeSource::Scalar(stored) if !stored.same_format(&target) => Some((decode, stored)),
            _ => None,
        })
        .expect("a flat scalar codec other than the dequant target has a tiled decode description");
    assert_eq!(stored.width(), 16, "the exhaustive input is every 16-bit stored code");
    assert_eq!(target.width(), 16, "the output buffer holds one 16-bit code per element");
    let scalar = hardware_type(&target).expect("the dequant target has a metal type");
    let stored_codes: Vec<u16> = (0..=u16::MAX).collect();
    let stored_bytes: Vec<u8> = stored_codes.iter().flat_map(|code| code.to_le_bytes()).collect();
    let mut host_values = Vec::new();
    PackedRows::from_bytes(&NumericFormat::Scalar(stored), stored_bytes.clone())
        .expect("stored codes are whole elements")
        .decode_all(&mut host_values)
        .expect("the stored layout decodes to the host element");
    let host_packed = PackedRows::from_values(&NumericFormat::Scalar(target), &host_values)
        .expect("the host converts into the target layout");
    let host_codes: Vec<u16> = host_packed
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect();
    let mut host_target_values = Vec::new();
    host_packed
        .decode_all(&mut host_target_values)
        .expect("the converted run decodes back");
    let kernel = widening_decode_kernel(&decode.msl(target), decode.function, scalar);
    let is_nan_code = |code: u16| host_target_nan(&target, code);

    let mut report = Vec::new();
    for math_mode in [MathMode::Safe, MathMode::Relaxed, MathMode::Fast] {
        let function = probe_function(&device, &kernel, math_mode);
        let pipeline = pipeline_from_function(&device, &function).expect("the widening kernel compiles");
        let gpu_codes = run_widening_decode(&device, &queue, &pipeline, &stored_bytes);
        let mut mismatches = ConversionMismatches::default();
        let mut compared = 0usize;
        for index in 0..stored_codes.len() {
            compared += 1;
            let (host, gpu) = (host_codes[index], gpu_codes[index]);
            if host == gpu || (is_nan_code(host) && is_nan_code(gpu)) {
                continue;
            }
            let source = host_values[index];
            if source.is_nan() {
                mismatches.nan += 1;
            } else if host_target_values[index].is_infinite() && source.is_finite() {
                mismatches.overflow += 1;
            } else if source != 0.0 && source.abs() < target.min_normal() {
                mismatches.subnormal_range += 1;
            } else {
                mismatches.other += 1;
                mismatches.first_other.get_or_insert((stored_codes[index], host, gpu));
            }
        }
        assert_eq!(compared, 65_536, "every stored code is compared in {math_mode:?}");
        if mismatches.total() != 0 {
            report.push(format!("{math_mode:?}: {} of {compared} differ: {mismatches:?}", mismatches.total()));
        }
    }

    assert!(
        report.is_empty(),
        "the in-kernel widening decode diverges from the host conversion (subnormal_range = target subnormal inputs, overflow = host rounds to infinity, nan, other = (stored, host, gpu) first example): {report:#?}"
    );
}

fn host_target_nan(target: &FloatLayout, code: u16) -> bool {
    let mut value = Vec::new();
    PackedRows::from_bytes(&NumericFormat::Scalar(*target), code.to_le_bytes().to_vec())
        .and_then(|rows| rows.decode_all(&mut value))
        .expect("one target code decodes");
    value[0].is_nan()
}
