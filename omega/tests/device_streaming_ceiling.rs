//! The physical read-bandwidth floor for decode on this host: how fast can
//! the GPU actually pull bytes out of each of the three buffer shapes decode
//! itself uses or could use, independent of any kernel's compute cost.
//!
//! Three sources, matching decode's own upload paths
//! (`omega/src/metal.rs::create_no_copy_buffer`/`upload_block_copy`):
//!
//! - **no-copy resident** — `newBufferWithBytesNoCopy_length_options_deallocator`
//!   over the REAL openchat-3.5 GGUF checkpoint's own `mmap`, page-aligned,
//!   `StorageModeShared` — the exact FFI call and options
//!   `create_no_copy_buffer` makes, applied to the exact bytes decode binds
//!   its weight tensors out of. One buffer spans the whole (page-rounded)
//!   mapping, addressed at offset 0 for both read sizes below — mirroring
//!   `checkpoint_mapping_offset`'s one-buffer-many-offsets shape (the offset
//!   itself does not change DRAM/GPU read bandwidth, which is what this file
//!   measures, so every read here binds the same buffer at offset 0 and
//!   varies only how many bytes of it a dispatch reads).
//! - **fresh shared** — `newBufferWithBytes_length_options` (copies once at
//!   creation) into a brand-new `StorageModeShared` buffer, filled from the
//!   same real checkpoint bytes so content is identical across all three
//!   arms and the only variable is the buffer's own backing.
//! - **private via blit** — `newBufferWithLength_options(StorageModePrivate)`
//!   filled once, untimed, via `MTLBlitCommandEncoder::copyFromBuffer_
//!   sourceOffset_toBuffer_destinationOffset_size` from the fresh-shared
//!   buffer above.
//!
//! Kernel: a streaming reduce, `float4`/`uint4` loads strided across the
//! whole read range so every load is live (the accumulator feeds a real
//! per-threadgroup output write, so the compiler cannot elide the loads).
//! Grid width is swept three ways per the task brief -- LOW (4
//! threadgroups/core), HIGH (8 threadgroups/core), WIDE (one thread per 16
//! bytes, no per-thread loop) -- and the best of the three is what this
//! file's discipline-log row reports as the ceiling.
//!
//! Timed by `commit()` -> `waitUntilCompleted()` around exactly one dispatch,
//! nothing subtracted. A separate empty-dispatch pipeline reports the fixed
//! per-dispatch cost this floor already includes.
//!
//! `#[ignore]`d: depends on a host-local openchat GGUF checkout outside this
//! repo, same convention as `proxima-model-interop`'s
//! `bind::real_openchat_file::*` tests.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use core::ffi::c_void;
use core::ptr::NonNull;
use std::os::fd::AsFd;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};

/// Same literal path `proxima-model-interop::serving::ServingConfig::
/// DEFAULT_MODEL_PATH` carries -- duplicated rather than depended on, since
/// `omega` has no (and must not gain) a dependency on `proxima-model-interop`
/// (that dependency runs the other way). Overridable so this test is not
/// nailed to one machine's home directory.
fn checkpoint_path() -> std::path::PathBuf {
    std::env::var("PROXIMA_OPENCHAT_GGUF")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(
                "/Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf",
            )
        })
}

/// A read-only `mmap` of the fixture file -- the same `rustix::mm::mmap`
/// call, same flags, as `proxima-model-interop/src/bind.rs`'s
/// `real_openchat_file::MappedGguf`, duplicated here (not depended on: that
/// module is private, and a bench harness keeping its own copy of a two-line
/// mmap wrapper is the existing convention, see `omega/benches/
/// wide_cooperative_reduce.rs`'s own doc on this exact tradeoff).
struct MappedFile {
    base: *mut u8,
    len: usize,
    _file: std::fs::File,
}

impl MappedFile {
    fn open(path: &std::path::Path) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let len = usize::try_from(file.metadata()?.len()).expect("fixture length fits in usize");
        // SAFETY: `len` matches the just-opened file's own length; `file` is
        // kept alive in `_file` for as long as `base` is used; the mapping
        // is read-only/private so no writer can observe or race it.
        let base = unsafe {
            rustix::mm::mmap(
                core::ptr::null_mut(),
                len,
                rustix::mm::ProtFlags::READ,
                rustix::mm::MapFlags::PRIVATE,
                file.as_fd(),
                0,
            )
        }
        .expect("mmap host-local openchat gguf fixture")
        .cast::<u8>();
        Ok(Self {
            base,
            len,
            _file: file,
        })
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: `base` points at `len` bytes mapped for `self`'s whole
        // lifetime; this borrows `self` immutably, so nothing can unmap the
        // region while the returned slice is alive.
        unsafe { core::slice::from_raw_parts(self.base, self.len) }
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        // SAFETY: `base`/`len` are exactly what `open`'s `mmap` call
        // returned; nothing else unmaps this region.
        let _ = unsafe { rustix::mm::munmap(self.base.cast::<c_void>(), self.len) };
    }
}

fn round_up_to_page(value: usize, page: usize) -> usize {
    value.div_ceil(page) * page
}

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

const EMPTY_DISPATCH_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void empty_dispatch(device float* sink [[buffer(0)]], uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        sink[0] = 0.0f;
    }
}
"#;

/// Test-local copy of `omega::metal::compile_pipeline` (private to that
/// module) -- same `MTLCompileOptions`/`MTLMathMode::Safe`/
/// `newLibraryWithSource_options_error`/`newComputePipelineStateWithFunction_error`
/// sequence, so this bench compiles through the identical path `execute`
/// does rather than a parallel one.
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

fn shared_buffer_from_bytes(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    // SAFETY: `bytes` is a live, non-empty slice for the duration of this
    // call; `newBufferWithBytes_length_options` copies from it once and
    // never retains the pointer past this call.
    let pointer = unsafe { NonNull::new_unchecked(bytes.as_ptr() as *mut c_void) };
    unsafe {
        device.newBufferWithBytes_length_options(
            pointer,
            bytes.len(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("device allocates a fresh shared buffer copied from real checkpoint bytes")
}

fn private_buffer_via_blit(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    source: &ProtocolObject<dyn MTLBuffer>,
    length: usize,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let private_buffer = device
        .newBufferWithLength_options(length, MTLResourceOptions::StorageModePrivate)
        .expect("device allocates a private buffer");
    let command_buffer = queue.commandBuffer().expect("command buffer for the fill blit");
    let blit = command_buffer
        .blitCommandEncoder()
        .expect("command buffer refused to hand out a blit encoder");
    unsafe {
        blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
            source,
            0,
            &private_buffer,
            0,
            length,
        );
    }
    blit.endEncoding();
    // untimed: this is the one-time fill, not part of any measured read.
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    private_buffer
}

fn no_copy_buffer_over_whole_mapping(
    device: &ProtocolObject<dyn MTLDevice>,
    mapped: &MappedFile,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let page = omega::metal::page_size();
    let rounded_length = round_up_to_page(mapped.len, page);
    // SAFETY: `mapped.base` is page-aligned (every `mmap` return is) and
    // stays valid for `mapped`'s whole lifetime, which outlives this test's
    // single-threaded use of the returned buffer; `rounded_length` extends
    // only up to the next page boundary past the file's real length, which
    // -- per `mmap`'s own contract, matching `checkpoint_mapping_offset`'s
    // doc in `omega/src/metal.rs` -- is already resident, zero-filled, mapped
    // memory. `None` deallocator: Metal never owns or frees this memory.
    let pointer = unsafe { NonNull::new_unchecked(mapped.base.cast::<c_void>()) };
    unsafe {
        device.newBufferWithBytesNoCopy_length_options_deallocator(
            pointer,
            rounded_length,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }
    .expect("device wraps the real checkpoint mapping no-copy")
}

fn uniform_u64_pair(
    device: &ProtocolObject<dyn MTLDevice>,
    first: u64,
    second: u64,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let words = [first, second];
    let bytes: &[u8] =
        unsafe { core::slice::from_raw_parts(words.as_ptr().cast::<u8>(), size_of_val(&words)) };
    shared_buffer_from_bytes(device, bytes)
}

/// One `commit()` -> `waitUntilCompleted()` timing around exactly one
/// dispatch of `streaming_reduce`, nothing subtracted.
fn time_streaming_reduce(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    data: &ProtocolObject<dyn MTLBuffer>,
    partial_sums: &ProtocolObject<dyn MTLBuffer>,
    uniforms: &ProtocolObject<dyn MTLBuffer>,
    total_threads: usize,
) -> std::time::Duration {
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
    let threadgroup_width = total_threads.clamp(1, 256);
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
    let started = std::time::Instant::now();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    started.elapsed()
}

fn time_empty_dispatch(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    sink: &ProtocolObject<dyn MTLBuffer>,
) -> std::time::Duration {
    let command_buffer = queue.commandBuffer().expect("command buffer");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("command buffer refused to hand out a compute encoder");
    encoder.setComputePipelineState(pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(sink), 0, 0);
    }
    let one = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(one, one);
    encoder.endEncoding();
    let started = std::time::Instant::now();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    started.elapsed()
}

fn mean_and_cov(samples: &[f64]) -> (f64, f64) {
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples.iter().map(|value| (value - mean).powi(2)).sum::<f64>()
        / samples.len() as f64;
    let stddev = variance.sqrt();
    let cov = if mean.abs() > f64::MIN_POSITIVE {
        stddev / mean * 100.0
    } else {
        0.0
    };
    (mean, cov)
}

const GPU_CORES: usize = 32;
const THREADGROUP: usize = 256;
const REPEATS: usize = 5;

#[derive(Clone, Copy)]
struct GridConfig {
    name: &'static str,
    total_threads_fn: fn(vec_count: u64) -> usize,
}

fn low_grid(_vec_count: u64) -> usize {
    4 * GPU_CORES * THREADGROUP
}

fn high_grid(_vec_count: u64) -> usize {
    8 * GPU_CORES * THREADGROUP
}

fn wide_grid(vec_count: u64) -> usize {
    usize::try_from(vec_count).expect("vec_count fits in usize on a 64-bit host")
}

const GRID_CONFIGS: [GridConfig; 3] = [
    GridConfig {
        name: "low_4tg_per_core",
        total_threads_fn: low_grid,
    },
    GridConfig {
        name: "high_8tg_per_core",
        total_threads_fn: high_grid,
    },
    GridConfig {
        name: "wide_one_thread_per_16b",
        total_threads_fn: wide_grid,
    },
];

struct SourceArm<'buffer> {
    name: &'static str,
    buffer: &'buffer ProtocolObject<dyn MTLBuffer>,
}

/// Measures every `(grid config, repeat)` cell for one `(source, size)`
/// pair, arms interleaved within each repeat (never all of one arm's
/// repeats back to back) so a thermal or scheduler drift across the whole
/// sweep cannot land on one arm more than another.
#[allow(clippy::too_many_arguments)]
fn sweep_one_size(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    read_bytes: u64,
    arms: &[SourceArm<'_>],
) -> Vec<(&'static str, &'static str, f64, f64)> {
    let vec_count = read_bytes / 16;
    let mut results: Vec<(&'static str, &'static str, f64, f64)> = Vec::new();

    for config in GRID_CONFIGS {
        let total_threads = (config.total_threads_fn)(vec_count);
        let threadgroups = total_threads.div_ceil(THREADGROUP).max(1);
        let uniforms = uniform_u64_pair(device, vec_count, total_threads as u64);
        let mut per_arm_samples: Vec<Vec<f64>> = vec![Vec::with_capacity(REPEATS); arms.len()];
        let partial_sums: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> = arms
            .iter()
            .map(|_| {
                device
                    .newBufferWithLength_options(
                        threadgroups * size_of::<f32>(),
                        MTLResourceOptions::StorageModeShared,
                    )
                    .expect("device allocates the per-threadgroup output buffer")
            })
            .collect();

        for _repeat in 0..REPEATS {
            for (arm_index, arm) in arms.iter().enumerate() {
                let elapsed = time_streaming_reduce(
                    queue,
                    pipeline,
                    arm.buffer,
                    &partial_sums[arm_index],
                    &uniforms,
                    total_threads,
                );
                let gbps = read_bytes as f64 / elapsed.as_secs_f64() / 1e9;
                per_arm_samples[arm_index].push(gbps);
            }
        }

        for (arm, samples) in arms.iter().zip(per_arm_samples.iter()) {
            let (mean, cov) = mean_and_cov(samples);
            results.push((arm.name, config.name, mean, cov));
            println!(
                "size_bytes={read_bytes} source={:<16} grid={:<24} mean_gbps={mean:.2} cov_pct={cov:.2} samples={samples:?}",
                arm.name, config.name
            );
        }
    }

    results
}

#[test]
#[ignore = "depends on a host-local openchat gguf checkout outside this repo"]
fn device_streaming_ceiling_across_three_sources_and_two_sizes() {
    let checkpoint = checkpoint_path();
    if !checkpoint.exists() {
        eprintln!(
            "skipping: no host-local openchat gguf fixture at {}",
            checkpoint.display()
        );
        return;
    }

    let device = MTLCreateSystemDefaultDevice().expect("a Metal device is available on this host");
    let queue = device.newCommandQueue().expect("device creates a command queue");

    let mapped = MappedFile::open(&checkpoint).expect("mmap the real openchat checkpoint");
    let file_bytes = mapped.as_slice();

    const ONE_GB_BYTES: u64 = 1_000_000_000;
    const FOUR_GB_BYTES: u64 = 4_000_000_000;
    assert!(
        mapped.len as u64 >= FOUR_GB_BYTES,
        "fixture must be at least 4,000,000,000 bytes to cover the 4 GB read arm; got {}",
        mapped.len
    );
    assert_eq!(ONE_GB_BYTES % 16, 0, "kernel reads whole uint4 (16B) lanes");
    assert_eq!(FOUR_GB_BYTES % 16, 0, "kernel reads whole uint4 (16B) lanes");

    let reduce_pipeline = compile_pipeline(&device, STREAMING_REDUCE_SOURCE, "streaming_reduce");
    let empty_pipeline = compile_pipeline(&device, EMPTY_DISPATCH_SOURCE, "empty_dispatch");

    // fixed per-dispatch cost this floor already includes.
    let empty_sink = device
        .newBufferWithLength_options(size_of::<f32>(), MTLResourceOptions::StorageModeShared)
        .expect("device allocates the empty-dispatch sink");
    let mut empty_samples = Vec::with_capacity(10);
    for _ in 0..10 {
        let elapsed = time_empty_dispatch(&queue, &empty_pipeline, &empty_sink);
        empty_samples.push(elapsed.as_secs_f64() * 1e3);
    }
    let (empty_mean_ms, empty_cov) = mean_and_cov(&empty_samples);
    println!(
        "empty_dispatch_fixed_cost_ms mean={empty_mean_ms:.4} cov_pct={empty_cov:.2} samples={empty_samples:?}"
    );

    let no_copy_buffer = no_copy_buffer_over_whole_mapping(&device, &mapped);

    let mut all_results: Vec<(u64, &'static str, &'static str, f64, f64)> = Vec::new();

    for read_bytes in [ONE_GB_BYTES, FOUR_GB_BYTES] {
        let byte_range = &file_bytes[..usize::try_from(read_bytes).expect("fits usize")];
        let fresh_shared_buffer = shared_buffer_from_bytes(&device, byte_range);
        let private_buffer = private_buffer_via_blit(
            &device,
            &queue,
            &fresh_shared_buffer,
            usize::try_from(read_bytes).expect("fits usize"),
        );

        let arms = [
            SourceArm {
                name: "nocopy_resident",
                buffer: &no_copy_buffer,
            },
            SourceArm {
                name: "fresh_shared",
                buffer: &fresh_shared_buffer,
            },
            SourceArm {
                name: "private_blit",
                buffer: &private_buffer,
            },
        ];

        let results = sweep_one_size(&device, &queue, &reduce_pipeline, read_bytes, &arms);
        for (source, grid, mean, cov) in results {
            all_results.push((read_bytes, source, grid, mean, cov));
        }
    }

    println!("=== best grid config per (source, size) ===");
    let mut best_ceiling_gbps = 0.0f64;
    for read_bytes in [ONE_GB_BYTES, FOUR_GB_BYTES] {
        for source in ["nocopy_resident", "fresh_shared", "private_blit"] {
            let best = all_results
                .iter()
                .filter(|(bytes, name, _, _, _)| *bytes == read_bytes && *name == source)
                .max_by(|left, right| left.3.total_cmp(&right.3))
                .expect("every (source, size) pair has at least one grid config result");
            println!(
                "size_bytes={read_bytes} source={source:<16} BEST grid={:<24} mean_gbps={:.2} cov_pct={:.2}",
                best.2, best.3, best.4
            );
            best_ceiling_gbps = best_ceiling_gbps.max(best.3);
            assert!(best.3.is_finite() && best.3 > 0.0, "measured a non-positive bandwidth");
        }
    }

    let floor_ms = 4.169e9 / (best_ceiling_gbps * 1e9) * 1e3;
    println!(
        "device streaming ceiling (best across all sources/sizes/grids): {best_ceiling_gbps:.2} GB/s -> floor_ms for 4.169 GB/token = {floor_ms:.3}"
    );
}
