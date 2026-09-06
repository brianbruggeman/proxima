//! Contention generator for `proxima-model-interop::bind::real_openchat_file::
//! decode_text_is_deterministic_across_repeated_runs` (race-brief ROW 317):
//! streams a 2 GB buffer through the same `streaming_reduce` kernel
//! `device_streaming_ceiling.rs` uses, one command buffer per iteration, in
//! a loop for `PROXIMA_LOAD_SECONDS` (default 120) -- keeping the GPU busy
//! long enough for a concurrent decode run to overlap with it. Its process
//! name (`gpu_load_generat` under macOS's 15-character truncation) is an
//! EXPECTED quiet-gate occupant for this slice only.
//!
//! Not a measurement: no bake-off, no CoV, no incumbent arm. It exists only
//! to hold the device busy while a separate process (the determinism
//! harness) is timed.
//!
//! `#[ignore]`d: depends on a real Metal device, same convention as
//! `device_streaming_ceiling.rs`.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use core::ffi::c_void;
use core::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};

const LOAD_BUFFER_BYTES: usize = 2_000_000_000;
const THREADGROUP: usize = 256;

const STREAMING_REDUCE_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void streaming_reduce(
    device const uint4* data [[buffer(0)]],
    device float* partial_sums [[buffer(1)]],
    constant uint64_t& vec_count [[buffer(2)]],
    constant uint64_t& stride [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup float partials[256];
    uint4 accumulator = uint4(0u);
    for (uint64_t index = (uint64_t)gid; index < vec_count; index += stride) {
        accumulator += data[index];
    }
    float lane_sum = float(accumulator.x) + float(accumulator.y) + float(accumulator.z) + float(accumulator.w);
    partials[tid] = lane_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partials[tid] += partials[tid + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        partial_sums[tgid] = partials[0];
    }
}
"#;

/// Test-local copy of `omega::metal::compile_pipeline` (private to that
/// module) -- same convention `device_streaming_ceiling.rs`'s own copy
/// documents: this bench compiles through the identical
/// `MTLCompileOptions`/`MTLMathMode::Safe`/`newLibraryWithSource_options_error`
/// sequence `execute` does, rather than a parallel one.
fn compile_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
    entry: &str,
) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
    let options = MTLCompileOptions::new();
    options.setMathMode(MTLMathMode::Safe);
    let source = NSString::from_str(source);
    let library = device
        .newLibraryWithSource_options_error(&source, Some(&options))
        .unwrap_or_else(|error| {
            panic!(
                "compiles the kernel `{entry}`: {}",
                error.localizedDescription()
            )
        });
    let entry_name = NSString::from_str(entry);
    let function = library
        .newFunctionWithName(&entry_name)
        .unwrap_or_else(|| panic!("kernel entry `{entry}` missing from its own compiled library"));
    device
        .newComputePipelineStateWithFunction_error(&function)
        .unwrap_or_else(|error| {
            panic!(
                "creates the pipeline for `{entry}`: {}",
                error.localizedDescription()
            )
        })
}

fn zero_filled_shared_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    length: usize,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    device
        .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
        .expect("device allocates a fresh shared load buffer")
}

fn uniform_u64_pair(
    device: &ProtocolObject<dyn MTLDevice>,
    first: u64,
    second: u64,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let words = [first, second];
    // SAFETY: `words` is a live, non-empty local for the duration of this
    // call; `newBufferWithBytes_length_options` copies from it once and
    // never retains the pointer past this call.
    let pointer = unsafe { NonNull::new_unchecked(words.as_ptr() as *mut c_void) };
    unsafe {
        device.newBufferWithBytes_length_options(
            pointer,
            size_of_val(&words),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("device allocates a uniform buffer")
}

fn load_seconds() -> u64 {
    std::env::var("PROXIMA_LOAD_SECONDS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120)
}

fn dispatch_once(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    data: &ProtocolObject<dyn MTLBuffer>,
    partial_sums: &ProtocolObject<dyn MTLBuffer>,
    uniforms: &ProtocolObject<dyn MTLBuffer>,
    total_threads: usize,
) {
    let command_buffer = queue.commandBuffer().expect("command buffer");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("command buffer refused to hand out a compute encoder");
    encoder.setComputePipelineState(pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(data), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(partial_sums), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(uniforms), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(uniforms), 8, 3);
    }
    let threadgroup_width = total_threads.clamp(1, THREADGROUP);
    let grid = MTLSize {
        width: total_threads,
        height: 1,
        depth: 1,
    };
    let threadgroup = MTLSize {
        width: threadgroup_width,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
}

/// Run this alongside `decode_text_is_deterministic_across_repeated_runs`
/// (that test's own doc comment names the exact companion invocation):
///
/// ```text
/// cargo test -p omega --release --features metal --no-default-features \
///   --features metal gpu_load_generator -- --ignored --nocapture
/// ```
#[test]
#[ignore = "holds a real Metal device busy for PROXIMA_LOAD_SECONDS; run only as the race slice's contention generator"]
fn gpu_load_generator() {
    let device = MTLCreateSystemDefaultDevice().expect("a Metal device is available on this host");
    let queue = device.newCommandQueue().expect("device creates a command queue");
    let pipeline = compile_pipeline(&device, STREAMING_REDUCE_SOURCE, "streaming_reduce");

    let data = zero_filled_shared_buffer(&device, LOAD_BUFFER_BYTES);
    let vec_count = (LOAD_BUFFER_BYTES / 16) as u64;
    let total_threads = usize::try_from(vec_count).expect("vec_count fits in usize on a 64-bit host");
    let threadgroups = total_threads.div_ceil(THREADGROUP).max(1);
    let partial_sums = zero_filled_shared_buffer(&device, threadgroups * size_of::<f32>());
    let uniforms = uniform_u64_pair(&device, vec_count, total_threads as u64);

    let budget = std::time::Duration::from_secs(load_seconds());
    let started = std::time::Instant::now();
    let mut iterations = 0u64;
    while started.elapsed() < budget {
        dispatch_once(
            &queue,
            &pipeline,
            &data,
            &partial_sums,
            &uniforms,
            total_threads,
        );
        iterations += 1;
    }

    std::println!(
        "gpu_load_generator iterations={iterations} elapsed_s={:.1} buffer_bytes={LOAD_BUFFER_BYTES}",
        started.elapsed().as_secs_f64()
    );
}
