//! Why does the packed-row `Q4_K` matvec cap at ~174-178 GB/s in-buffer when
//! a streaming kernel reaches the device ceiling (`omega/tests/
//! device_streaming_ceiling.rs`)? This file builds a ladder of hand-written
//! Metal kernels that each add exactly one layer of production's own work on
//! top of the last, over the SAME real weight bytes and the SAME production
//! dispatch shape (32-thread threadgroups, 4 rows per simdgroup
//! [`PACKED_ROW_ROWS_PER_GROUP`], `dispatchThreads`), so a rung-to-rung
//! bandwidth drop can be attributed to the specific thing that rung added
//! rather than guessed at:
//!
//! - **L0 (`q4k_matvec_l0_streaming`)** -- production's exact addressing
//!   (`ix`/`it`/`iq`/`ir`/`byte_base`/`low_index`, the same lane-spread
//!   `push_packed_row_blocked_body`'s `plain_product` arm computes,
//!   `omega/src/msl.rs:3345-3347,3386-3398`) reading the same byte ranges
//!   `q4k_pair_dot` reads per `(row, ib)`, summed as `uint`. No half->float,
//!   no mask, no shift, no fma -- isolates the READ PATTERN alone.
//! - **L1 (`q4k_matvec_l1_header_decode`)** -- L0's reads, PLUS the real
//!   `q4k_header_for` (lifted verbatim from the production
//!   [`omega::msl::Q4K_UNPACK_MSL`] constant, not restated) decoding each of
//!   the four per-lane headers into `(scale, minimum)` floats. Still no
//!   nibble extraction.
//! - **L2 (`q4k_matvec_l2_dequant`)** -- L1, PLUS the full nibble unpack and
//!   dequant (`scale*nibble - minimum` for all 32 elements a lane owns,
//!   `q4k_pair_dot`'s own arithmetic minus the activation multiply). No
//!   activation buffer touched at all.
//! - **L3 baseline** -- the REAL production kernel, dispatched through
//!   [`omega::metal::plan`]/[`omega::metal::execute_plan`] (the same public
//!   entry point `q4k_real_checkpoint_parity.rs` uses), so its MSL text is
//!   whatever [`omega::msl::emit`] renders today -- byte-identical to
//!   production by construction, not by copy-paste. This is the ladder's
//!   ground truth.
//! - **L3 shape-sweep (`q4k_matvec_l3_shape`)** -- calls the SAME
//!   `q4k_pair_dot` MSL function (again lifted verbatim from
//!   [`omega::msl::Q4K_UNPACK_MSL`], not restated), but through a
//!   hand-written dispatch wrapper this file controls directly, because
//!   `emit`'s own dispatch geometry (simdgroups per threadgroup, math mode)
//!   is a compile-time Cargo-feature decision (`metal-q4k-nsg2` and
//!   friends) that cannot be varied at runtime inside one test binary. This
//!   sweep isolates the SHAPE axis (simdgroups/threadgroup, `dispatchThreads`
//!   vs `dispatchThreadgroups`, `MTLMathMode`) while holding the per-element
//!   compute path textually identical to production. Its own default arm (1
//!   simdgroup/threadgroup, `dispatchThreads`, `MTLMathMode::Safe`) is
//!   checked against the L3 baseline's own output for agreement, so a
//!   divergence between the two L3 paths reads as a bug in this harness, not
//!   a shape effect.
//!
//! Real weight bytes only (guiding-principles §9): `blk.0.ffn_up.weight`
//! (14336 x 4096, `Q4_K`) from the same openchat-3.5-1210 checkpoint
//! `q4k_real_checkpoint_parity.rs`/`device_streaming_ceiling.rs` use, bound
//! with the SAME no-copy mapping technique
//! (`newBufferWithBytesNoCopy_length_options_deallocator` over the whole
//! page-rounded `mmap`, `StorageModeShared`, offset-addressed per tensor --
//! `device_streaming_ceiling.rs`'s own doc on `checkpoint_mapping_offset`'s
//! one-buffer-many-offsets shape) for every hand-dispatched arm (L0/L1/L2/L3
//! shape-sweep); the L3 baseline arm borrows the identical mmap'd byte slice
//! as a [`proxima_tensor::QuantizedBlock::Q4K`] operand, which
//! `omega::metal::upload_packed_bytes`'s own `checkpoint_mapping_offset` arm
//! resolves to the same no-copy technique when the slice is not itself
//! page-aligned (real GGUF tensor offsets never are).
//!
//! GB/s for every arm (including the L3 baseline) is the full row-major byte
//! extent of the swept tensor (`ROWS * ROW_BYTES`, all 144 bytes per Q4_K
//! super-block) divided by measured wall time -- the same accounting
//! `device_streaming_ceiling.rs` and production's own decode-throughput
//! number use, NOT a per-lane distinct-byte count (a lane's OWN issued loads
//! only cover a narrower slice of each 144-byte block; a per-lane count
//! would read smaller and would not be comparable to production's own GB/s
//! figures). This keeps every rung directly comparable on the same axis.
//!
//! Timed by `commit()` -> `waitUntilCompleted()` around exactly one dispatch
//! per repeat, 5 repeats per arm (`REPEATS`), nothing subtracted -- same
//! convention as `device_streaming_ceiling.rs`. The L3 baseline arm is the
//! one exception: it times `omega::metal::execute_plan` end to end (upload
//! resolution + dispatch + readback) because assembling the emitted kernel's
//! general uniform buffer by hand would risk diverging from the exact bytes
//! this arm exists to be faithful to; the plan is built once and marked
//! resident so repeats amortize upload as much as the public API allows.
//! This is a documented scope limitation, not an oversight -- refining it
//! (an `instrument`-gated per-op GPU timestamp, `execute_plan_op_timed`) is
//! a candidate for the measurement slice, not this build slice.
//!
//! `#[ignore]`d: depends on a host-local openchat GGUF checkout outside this
//! repo, same convention as `q4k_real_checkpoint_parity.rs`/
//! `device_streaming_ceiling.rs`.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use core::ffi::c_void;
use core::ptr::NonNull;
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::os::fd::AsFd;
use std::path::Path;
use std::time::{Duration, Instant};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};

use proxima_gguf::parser::{GgufEvent, GgufParser};
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::quant::q4_k;
use proxima_gguf::types::GgmlType;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp,
    append, map,
};

/// Real GGUF checkpoint path, overridable via `PROXIMA_BENCH_GGUF_PATH` --
/// same env var and same default host path `q4k_real_checkpoint_parity.rs`
/// reads, since this file is closest kin to that one (a `Q4_K` matvec over
/// real checkpoint bytes), restated per-binary per that file's own doc on
/// why (each `omega/tests/*.rs` is a separate integration-test crate).
fn checkpoint_path() -> String {
    std::env::var("PROXIMA_BENCH_GGUF_PATH").unwrap_or_else(|_| {
        "/Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf"
            .to_string()
    })
}

/// A read-only `mmap` of the fixture file -- same `rustix::mm::mmap` call,
/// flags, and per-binary-copy posture as `device_streaming_ceiling.rs`'s own
/// `MappedFile` (see that file's doc for why this is restated rather than
/// shared).
struct MappedFile {
    base: *mut u8,
    len: usize,
    _file: std::fs::File,
}

impl MappedFile {
    fn open(path: &Path) -> std::io::Result<Self> {
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

/// Byte size and shape of a real tensor located in the checkpoint's own
/// header -- restated from `q4k_real_checkpoint_parity.rs`'s
/// `real_tensor_bytes`, but returning the byte RANGE rather than a copied
/// `Vec`: every hand-dispatched arm in this file binds the tensor no-copy
/// straight off the `mmap`, so copying its bytes here would defeat the
/// point.
struct RealTensor {
    byte_offset: u64,
    byte_len: u64,
    in_dim: usize,
    out_dim: usize,
}

fn real_gguf_header(path: &Path) -> Option<(ParsedGguf, u64, std::fs::File)> {
    let mut file = std::fs::File::open(path).ok()?;
    let file_len = file.metadata().ok()?.len();

    let mut prefix_len = 1usize << 20;
    loop {
        let mut buf = vec![0u8; prefix_len];
        file.seek(SeekFrom::Start(0)).expect("seek to start");
        let read = file.read(&mut buf).expect("read gguf prefix");
        buf.truncate(read);

        if let Ok((parser, events)) = GgufParser::new().push(&buf) {
            let mut version = None;
            let mut metadata = Vec::new();
            let mut tensors = Vec::new();
            let mut completion = None;
            for event in events {
                match event {
                    GgufEvent::Header {
                        version: version_value,
                        ..
                    } => version = Some(version_value),
                    GgufEvent::Metadata { key, value } => metadata.push((key, value)),
                    GgufEvent::Tensor(tensor) => tensors.push(tensor),
                    GgufEvent::Complete {
                        data_offset,
                        alignment,
                    } => {
                        completion = Some((data_offset, alignment));
                    }
                }
            }
            if let (Some(version), Some((data_offset, alignment))) = (version, completion) {
                parser.finish().expect("parser reports complete and clean");
                let parsed = ParsedGguf {
                    version,
                    tensor_count: tensors.len() as u64,
                    kv_count: metadata.len() as u64,
                    metadata,
                    tensors,
                    data_offset,
                    alignment,
                };
                return Some((parsed, file_len, file));
            }
        }
        if prefix_len as u64 >= file_len {
            return None;
        }
        prefix_len *= 2;
    }
}

fn locate_real_tensor(
    parsed: &ParsedGguf,
    file_len: u64,
    name: &str,
    expect_type: GgmlType,
) -> Option<RealTensor> {
    let tensor = parsed.tensors.iter().find(|candidate| candidate.name == name)?;
    if tensor.ggml_type != expect_type {
        eprintln!(
            "locate_real_tensor: {name} is {:?} in this file, not {expect_type:?} -- test skipped, not faked",
            tensor.ggml_type
        );
        return None;
    }
    let in_dim = tensor.dims[0] as usize;
    let out_dim = tensor.dims[1] as usize;
    let range = parsed
        .tensor_data_range(tensor, file_len)
        .expect("tensor byte range within file bounds");
    Some(RealTensor {
        byte_offset: range.start,
        byte_len: range.end - range.start,
        in_dim,
        out_dim,
    })
}

// ---- production shape constants, restated per the same posture
// `omega::msl::Q4K_BLOCK_BYTES`'s own doc establishes (this crate's own
// public consts pinned by a dedicated unpack test; the two private
// dispatch-shape facts below restated here since they are not `pub` in
// `omega::msl`). ----

const Q4K_BLOCK_BYTES: usize = omega::msl::Q4K_BLOCK_BYTES;
const Q4K_BLOCK_ELEMENTS: usize = omega::msl::Q4K_BLOCK_ELEMENTS;
/// `omega/src/msl.rs`'s own private `PACKED_ROWS_PER_GROUP` -- ggml's
/// `N_R0_Q4_K 4`, restated here since it is not `pub`.
const PACKED_ROWS_PER_GROUP: usize = 4;

const ROWS: usize = 14336;
const IN_DIM: usize = 4096;
const BLOCKS_PER_ROW: usize = IN_DIM / Q4K_BLOCK_ELEMENTS;
const ROW_BYTES: usize = BLOCKS_PER_ROW * Q4K_BLOCK_BYTES;
const TOTAL_WEIGHT_BYTES: u64 = (ROWS * ROW_BYTES) as u64;
const TOTAL_SIMDGROUPS: usize = ROWS / PACKED_ROWS_PER_GROUP;

const REPEATS: usize = 5;
const PARITY_ROWS: usize = 64;
const PARITY_MAX_ABS_ERROR: f32 = 1e-4;

// ---- L0: pure streaming, production's addressing, no dequant ----

const L0_STREAMING_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void q4k_matvec_l0_streaming(
    device const uchar* weight [[buffer(0)]],
    device uint* row_sums [[buffer(1)]],
    constant uint64_t& blocks_per_row [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    uint lane = tid;
    uint ix = lane / 8u;
    uint it = lane % 8u;
    uint iq = it / 4u;
    uint ir = it % 4u;
    uint byte_base = 32u * iq + 8u * ir;
    uint low_index = 64u * iq + 8u * ir;
    uint group_first = tgid * 4u;
    uint ib_first = ix;
    uint ib_step = 4u;
    uint indices[4] = { low_index, low_index + 32u, low_index + 128u, low_index + 160u };

    ulong blk_step = (ulong)ib_step * 144ul;
    device const uchar* blk_ptr[4];
    for (uint q = 0u; q < 4u; ++q) {
        blk_ptr[q] = weight + (ulong)(group_first + q) * (ulong)blocks_per_row * 144ul
            + (ulong)ib_first * 144ul;
    }

    uint acc[4] = { 0u, 0u, 0u, 0u };
    for (uint ib = ib_first; ib < (uint)blocks_per_row; ib += ib_step) {
        for (uint q = 0u; q < 4u; ++q) {
            device const uchar* block = blk_ptr[q];
            device const uchar* qs = block + 16u;
            device const ushort* word_low = (device const ushort*)(qs + byte_base);
            device const ushort* word_high = (device const ushort*)(qs + byte_base + 64u);
            uint local = 0u;
            for (uint i = 0u; i < 4u; ++i) {
                local += (uint)word_low[i];
                local += (uint)word_high[i];
            }
            local += (uint)block[0] + (uint)block[1] + (uint)block[2] + (uint)block[3];
            device const uchar* scales = block + 4u;
            for (uint h = 0u; h < 4u; ++h) {
                uint index = indices[h];
                uint group = index / 64u;
                uint within = index % 64u;
                uint sub_block = 2u * group + (within < 32u ? 0u : 1u);
                local += (uint)scales[sub_block] + (uint)scales[sub_block + 4u];
            }
            acc[q] += local;
            blk_ptr[q] += blk_step;
        }
    }
    for (uint q = 0u; q < 4u; ++q) {
        uint total = simd_sum(acc[q]);
        if (lane == 0u) {
            row_sums[group_first + q] = total;
        }
    }
}
"#;

const L1_KERNEL_BODY: &str = r#"
kernel void q4k_matvec_l1_header_decode(
    device const uchar* weight [[buffer(0)]],
    device float* row_sums [[buffer(1)]],
    constant uint64_t& blocks_per_row [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    uint lane = tid;
    uint ix = lane / 8u;
    uint it = lane % 8u;
    uint iq = it / 4u;
    uint ir = it % 4u;
    uint byte_base = 32u * iq + 8u * ir;
    uint low_index = 64u * iq + 8u * ir;
    uint group_first = tgid * 4u;
    uint ib_first = ix;
    uint ib_step = 4u;

    ulong blk_step = (ulong)ib_step * 144ul;
    device const uchar* blk_ptr[4];
    for (uint q = 0u; q < 4u; ++q) {
        blk_ptr[q] = weight + (ulong)(group_first + q) * (ulong)blocks_per_row * 144ul
            + (ulong)ib_first * 144ul;
    }

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (uint ib = ib_first; ib < (uint)blocks_per_row; ib += ib_step) {
        for (uint q = 0u; q < 4u; ++q) {
            device const uchar* block = blk_ptr[q];
            device const uchar* qs = block + 16u;
            device const ushort* word_low = (device const ushort*)(qs + byte_base);
            device const ushort* word_high = (device const ushort*)(qs + byte_base + 64u);
            uint byte_sum = 0u;
            for (uint i = 0u; i < 4u; ++i) {
                byte_sum += (uint)word_low[i];
                byte_sum += (uint)word_high[i];
            }
            q4k_header h0 = q4k_header_for(block, low_index);
            q4k_header h1 = q4k_header_for(block, low_index + 32u);
            q4k_header h2 = q4k_header_for(block, low_index + 128u);
            q4k_header h3 = q4k_header_for(block, low_index + 160u);
            acc[q] += float(byte_sum) + h0.scale + h0.minimum + h1.scale + h1.minimum
                + h2.scale + h2.minimum + h3.scale + h3.minimum;
            blk_ptr[q] += blk_step;
        }
    }
    for (uint q = 0u; q < 4u; ++q) {
        float total = simd_sum(acc[q]);
        if (lane == 0u) {
            row_sums[group_first + q] = total;
        }
    }
}
"#;

const L2_KERNEL_BODY: &str = r#"
kernel void q4k_matvec_l2_dequant(
    device const uchar* weight [[buffer(0)]],
    device float* row_sums [[buffer(1)]],
    constant uint64_t& blocks_per_row [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    uint lane = tid;
    uint ix = lane / 8u;
    uint it = lane % 8u;
    uint iq = it / 4u;
    uint ir = it % 4u;
    uint byte_base = 32u * iq + 8u * ir;
    uint low_index = 64u * iq + 8u * ir;
    uint group_first = tgid * 4u;
    uint ib_first = ix;
    uint ib_step = 4u;

    ulong blk_step = (ulong)ib_step * 144ul;
    device const uchar* blk_ptr[4];
    for (uint q = 0u; q < 4u; ++q) {
        blk_ptr[q] = weight + (ulong)(group_first + q) * (ulong)blocks_per_row * 144ul
            + (ulong)ib_first * 144ul;
    }

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (uint ib = ib_first; ib < (uint)blocks_per_row; ib += ib_step) {
        for (uint q = 0u; q < 4u; ++q) {
            device const uchar* block = blk_ptr[q];
            device const uchar* qs = block + 16u;
            device const ushort* word_low = (device const ushort*)(qs + byte_base);
            device const ushort* word_high = (device const ushort*)(qs + byte_base + 64u);
            q4k_header h0 = q4k_header_for(block, low_index);
            q4k_header h1 = q4k_header_for(block, low_index + 32u);
            q4k_header h2 = q4k_header_for(block, low_index + 128u);
            q4k_header h3 = q4k_header_for(block, low_index + 160u);
            for (uint i = 0u; i < 4u; ++i) {
                uint word1 = (uint)word_low[i];
                uint word2 = (uint)word_high[i];
                acc[q] += h0.scale * float(word1 & 0x0Fu) - h0.minimum;
                acc[q] += h0.scale * float((word1 >> 8) & 0x0Fu) - h0.minimum;
                acc[q] += h1.scale * float((word1 >> 4) & 0x0Fu) - h1.minimum;
                acc[q] += h1.scale * float((word1 >> 12) & 0x0Fu) - h1.minimum;
                acc[q] += h2.scale * float(word2 & 0x0Fu) - h2.minimum;
                acc[q] += h2.scale * float((word2 >> 8) & 0x0Fu) - h2.minimum;
                acc[q] += h3.scale * float((word2 >> 4) & 0x0Fu) - h3.minimum;
                acc[q] += h3.scale * float((word2 >> 12) & 0x0Fu) - h3.minimum;
            }
            blk_ptr[q] += blk_step;
        }
    }
    for (uint q = 0u; q < 4u; ++q) {
        float total = simd_sum(acc[q]);
        if (lane == 0u) {
            row_sums[group_first + q] = total;
        }
    }
}
"#;

/// L3's shape-sweep body -- calls the REAL `q4k_pair_dot` (from
/// [`omega::msl::Q4K_UNPACK_MSL`], not restated), through a dispatch this
/// file controls so `threads_per_threadgroup` (and therefore simdgroups per
/// threadgroup) can vary at runtime -- see this file's module doc for why
/// `emit`'s own geometry cannot.
const L3_SHAPE_KERNEL_BODY: &str = r#"
kernel void q4k_matvec_l3_shape(
    device const uchar* weight [[buffer(0)]],
    device const float* activation [[buffer(1)]],
    device float* row_sums [[buffer(2)]],
    constant uint64_t& blocks_per_row [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint threads_per_tg [[threads_per_threadgroup]])
{
    uint simdgroups_per_tg = threads_per_tg / 32u;
    uint sgitg = tid / 32u;
    uint lane = tid % 32u;
    uint simdgroup_index = tgid * simdgroups_per_tg + sgitg;
    uint group_first = simdgroup_index * 4u;

    uint ix = lane / 8u;
    uint it = lane % 8u;
    uint iq = it / 4u;
    uint ir = it % 4u;
    uint low_index = 64u * iq + 8u * ir;
    uint ib_first = ix;
    uint ib_step = 4u;

    ulong blk_step = (ulong)ib_step * 144ul;
    device const uchar* blk_ptr[4];
    for (uint q = 0u; q < 4u; ++q) {
        blk_ptr[q] = weight + (ulong)(group_first + q) * (ulong)blocks_per_row * 144ul
            + (ulong)ib_first * 144ul;
    }
    device const float* y4 = activation + (ulong)ib_first * 256ul + (ulong)(64u * iq + 8u * ir);
    ulong y4_step = (ulong)ib_step * 256ul;

    float sumf[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float yl[16];
    float yh[16];
    for (uint ib = ib_first; ib < (uint)blocks_per_row; ib += ib_step) {
        for (uint i = 0u; i < 8u; ++i) {
            yl[i] = y4[i];
            yl[i + 8u] = y4[i + 32u];
            yh[i] = y4[i + 128u];
            yh[i + 8u] = y4[i + 160u];
        }
        for (uint q = 0u; q < 4u; ++q) {
            sumf[q] += q4k_pair_dot(blk_ptr[q], iq, ir, yl, yh);
            blk_ptr[q] += blk_step;
        }
        y4 += y4_step;
    }
    for (uint q = 0u; q < 4u; ++q) {
        float total = simd_sum(sumf[q]);
        if (lane == 0u) {
            row_sums[group_first + q] = total;
        }
    }
}
"#;

/// same `#include`/`using namespace` preamble `omega::msl::emit` itself
/// prepends before `Q4K_UNPACK_MSL` (`omega/src/msl.rs:2355-2356`) --
/// `Q4K_UNPACK_MSL` is bare body text, not a compilable translation unit on
/// its own, so every hand-assembled arm here restates the same preamble.
const METAL_PREAMBLE: &str = "#include <metal_stdlib>\nusing namespace metal;\n\n";

fn l1_source() -> String {
    format!("{METAL_PREAMBLE}{}\n{L1_KERNEL_BODY}", omega::msl::Q4K_UNPACK_MSL)
}

fn l2_source() -> String {
    format!("{METAL_PREAMBLE}{}\n{L2_KERNEL_BODY}", omega::msl::Q4K_UNPACK_MSL)
}

fn l3_shape_source() -> String {
    format!("{METAL_PREAMBLE}{}\n{L3_SHAPE_KERNEL_BODY}", omega::msl::Q4K_UNPACK_MSL)
}

// ---- device plumbing (per-binary copies, same posture as
// `device_streaming_ceiling.rs`'s own `compile_pipeline`/
// `shared_buffer_from_bytes`) ----

fn compile_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
    entry: &str,
    math_mode: MTLMathMode,
) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
    let options = MTLCompileOptions::new();
    options.setMathMode(math_mode);
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

fn uniform_u64(device: &ProtocolObject<dyn MTLDevice>, value: u64) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let bytes = value.to_ne_bytes();
    shared_buffer_from_bytes(device, &bytes)
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
    // -- per `mmap`'s own contract -- is already resident, zero-filled,
    // mapped memory. `None` deallocator: Metal never owns or frees this
    // memory.
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

fn bind_buffers(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buffers: &[(&ProtocolObject<dyn MTLBuffer>, usize)],
) {
    for (index, (buffer, offset)) in buffers.iter().enumerate() {
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(*buffer), *offset, index);
        }
    }
}

fn time_dispatch_threads(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: &[(&ProtocolObject<dyn MTLBuffer>, usize)],
    grid_threads: usize,
    threadgroup_width: usize,
) -> Duration {
    let command_buffer = queue.commandBuffer().expect("command buffer");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("command buffer refused to hand out a compute encoder");
    encoder.setComputePipelineState(pipeline);
    bind_buffers(&encoder, buffers);
    let grid = MTLSize {
        width: grid_threads,
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
    let started = Instant::now();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    started.elapsed()
}

fn time_dispatch_threadgroups(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: &[(&ProtocolObject<dyn MTLBuffer>, usize)],
    threadgroup_count: usize,
    threadgroup_width: usize,
) -> Duration {
    let command_buffer = queue.commandBuffer().expect("command buffer");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("command buffer refused to hand out a compute encoder");
    encoder.setComputePipelineState(pipeline);
    bind_buffers(&encoder, buffers);
    let grid = MTLSize {
        width: threadgroup_count,
        height: 1,
        depth: 1,
    };
    let threadgroup = MTLSize {
        width: threadgroup_width,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, threadgroup);
    encoder.endEncoding();
    let started = Instant::now();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    started.elapsed()
}

fn mean_and_cov(samples: &[f64]) -> (f64, f64) {
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance =
        samples.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / samples.len() as f64;
    let stddev = variance.sqrt();
    let cov = if mean.abs() > f64::MIN_POSITIVE {
        stddev / mean * 100.0
    } else {
        0.0
    };
    (mean, cov)
}

fn gbps_samples(elapsed_samples: &[Duration]) -> Vec<f64> {
    elapsed_samples
        .iter()
        .map(|elapsed| TOTAL_WEIGHT_BYTES as f64 / elapsed.as_secs_f64() / 1e9)
        .collect()
}

/// `[rows, k] x [k, 1] -> [rows, 1]`, same shape
/// `q4k_real_checkpoint_parity.rs`'s own `matmul_program` builds -- named
/// inputs here (`mark_resident` matches by name) since this file's L3
/// baseline arm reuses one [`omega::metal::Plan`] across repeats.
fn matmul_program(rows: u32, k: u32, weight_dtype: DType) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: weight_dtype,
            shape: vec![Extent::Static(rows), Extent::Static(k)],
            name: Some("weight".into()),
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(k), Extent::Static(1)],
            name: Some("activation".into()),
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (activation, IndexMap::Affine(map::projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("q4k_ladder_matmul".into()),
        }),
    );
    (program, sum)
}

fn cpu_reference_first_rows(weight_bytes: &[u8], activation: &[f32]) -> Vec<f32> {
    let mut reference = Vec::with_capacity(PARITY_ROWS);
    let mut dequantized_row = vec![0.0f32; IN_DIM];
    let (row_chunks, _remainder) =
        weight_bytes[..PARITY_ROWS * ROW_BYTES].as_chunks::<ROW_BYTES>();
    for row_blocks in row_chunks {
        q4_k::dequantize(row_blocks, &mut dequantized_row)
            .expect("a whole number of q4_k super-blocks per row");
        let dot: f32 = dequantized_row
            .iter()
            .zip(activation.iter())
            .map(|(weight, act)| weight * act)
            .sum();
        reference.push(dot);
    }
    reference
}

fn assert_parity(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: degenerate gate, row counts differ");
    let mut max_abs_error = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(got.is_finite(), "{label}: produced a non-finite value: {got}");
        max_abs_error = max_abs_error.max((got - want).abs());
    }
    eprintln!("{label}: max_abs_error={max_abs_error}");
    assert!(
        max_abs_error <= PARITY_MAX_ABS_ERROR,
        "{label}: max_abs_error={max_abs_error} exceeds {PARITY_MAX_ABS_ERROR}"
    );
}

fn read_f32_buffer(buffer: &ProtocolObject<dyn MTLBuffer>, count: usize) -> Vec<f32> {
    let pointer = buffer.contents();
    // SAFETY: `buffer` is `StorageModeShared` and `count` `f32`s is within
    // the length every caller of this helper allocated it with.
    let slice = unsafe { core::slice::from_raw_parts(pointer.as_ptr().cast::<f32>(), count) };
    slice.to_vec()
}

#[derive(Clone, Copy)]
enum DispatchMode {
    Threads,
    Threadgroups,
}

#[derive(Clone, Copy)]
struct MathModeArm {
    name: &'static str,
    mode: MTLMathMode,
}

const MATH_MODE_ARMS: [MathModeArm; 3] = [
    MathModeArm {
        name: "safe",
        mode: MTLMathMode::Safe,
    },
    MathModeArm {
        name: "relaxed",
        mode: MTLMathMode::Relaxed,
    },
    MathModeArm {
        name: "fast",
        mode: MTLMathMode::Fast,
    },
];

const SIMDGROUPS_PER_THREADGROUP: [usize; 3] = [1, 2, 4];

#[test]
#[ignore = "depends on a host-local openchat gguf checkout outside this repo"]
fn matvec_roofline_ladder_l0_through_l3_and_shape_sweep() {
    let path_string = checkpoint_path();
    let path = Path::new(&path_string);
    let Some((parsed, file_len, _file)) = real_gguf_header(path) else {
        eprintln!("real gguf file not found at {path_string}; test skipped");
        return;
    };
    let Some(tensor) = locate_real_tensor(&parsed, file_len, "blk.0.ffn_up.weight", GgmlType::Q4_K)
    else {
        return;
    };
    assert_eq!(
        tensor.in_dim, IN_DIM,
        "blk.0.ffn_up.weight in_dim must match the {IN_DIM} this ladder is sized for"
    );
    assert_eq!(
        tensor.out_dim, ROWS,
        "blk.0.ffn_up.weight out_dim must match the {ROWS} this ladder is sized for"
    );
    assert_eq!(
        tensor.byte_len as usize,
        ROWS * ROW_BYTES,
        "declared tensor byte length matches rows*row_bytes"
    );

    let mapped = MappedFile::open(path).expect("mmap the real openchat checkpoint");
    let weight_offset = tensor.byte_offset as usize;
    let weight_bytes_for_cpu_reference =
        &mapped.as_slice()[weight_offset..weight_offset + PARITY_ROWS * ROW_BYTES];

    let device = MTLCreateSystemDefaultDevice().expect("a Metal device is available on this host");
    let queue = device.newCommandQueue().expect("device creates a command queue");

    let no_copy_weight = no_copy_buffer_over_whole_mapping(&device, &mapped);

    let mut lcg = Lcg(2026);
    let activation: Vec<f32> = (0..IN_DIM).map(|_| lcg.next_unit() * 4.0 - 2.0).collect();
    // SAFETY: `activation` is a live `Vec<f32>` for the duration of this
    // call; the byte view is read-only and never outlives `activation`.
    let activation_bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(
            activation.as_ptr().cast::<u8>(),
            std::mem::size_of_val(activation.as_slice()),
        )
    };
    let activation_buffer = shared_buffer_from_bytes(&device, activation_bytes);
    let blocks_per_row_uniform = uniform_u64(&device, BLOCKS_PER_ROW as u64);

    let cpu_reference = cpu_reference_first_rows(weight_bytes_for_cpu_reference, &activation);

    let default_threadgroup_width = 32usize;
    let default_grid_threads = TOTAL_SIMDGROUPS * default_threadgroup_width;

    println!(
        "=== bandwidth ladder: production dispatch shape ({default_threadgroup_width}-thread \
         threadgroups, {PACKED_ROWS_PER_GROUP} rows/simdgroup, dispatchThreads) ==="
    );

    let mut ladder_gbps: BTreeMap<&'static str, f64> = BTreeMap::new();

    // ---- L0 ----
    let l0_pipeline = compile_pipeline(
        &device,
        L0_STREAMING_SOURCE,
        "q4k_matvec_l0_streaming",
        MTLMathMode::Safe,
    );
    let l0_output = device
        .newBufferWithLength_options(ROWS * size_of::<u32>(), MTLResourceOptions::StorageModeShared)
        .expect("device allocates L0's output buffer");
    {
        let mut elapsed_samples = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let elapsed = time_dispatch_threads(
                &queue,
                &l0_pipeline,
                &[
                    (&no_copy_weight, weight_offset),
                    (&l0_output, 0),
                    (&blocks_per_row_uniform, 0),
                ],
                default_grid_threads,
                default_threadgroup_width,
            );
            elapsed_samples.push(elapsed);
        }
        let samples = gbps_samples(&elapsed_samples);
        let (mean, cov) = mean_and_cov(&samples);
        println!("arm=L0_streaming mean_gbps={mean:.2} cov_pct={cov:.2} samples={samples:?}");
        ladder_gbps.insert("L0_streaming", mean);
    }

    // ---- L1 ----
    let l1_source_text = l1_source();
    let l1_pipeline = compile_pipeline(
        &device,
        &l1_source_text,
        "q4k_matvec_l1_header_decode",
        MTLMathMode::Safe,
    );
    let l1_output = device
        .newBufferWithLength_options(ROWS * size_of::<f32>(), MTLResourceOptions::StorageModeShared)
        .expect("device allocates L1's output buffer");
    {
        let mut elapsed_samples = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let elapsed = time_dispatch_threads(
                &queue,
                &l1_pipeline,
                &[
                    (&no_copy_weight, weight_offset),
                    (&l1_output, 0),
                    (&blocks_per_row_uniform, 0),
                ],
                default_grid_threads,
                default_threadgroup_width,
            );
            elapsed_samples.push(elapsed);
        }
        let samples = gbps_samples(&elapsed_samples);
        let (mean, cov) = mean_and_cov(&samples);
        println!("arm=L1_header_decode mean_gbps={mean:.2} cov_pct={cov:.2} samples={samples:?}");
        ladder_gbps.insert("L1_header_decode", mean);
    }

    // ---- L2 ----
    let l2_source_text = l2_source();
    let l2_pipeline = compile_pipeline(
        &device,
        &l2_source_text,
        "q4k_matvec_l2_dequant",
        MTLMathMode::Safe,
    );
    let l2_output = device
        .newBufferWithLength_options(ROWS * size_of::<f32>(), MTLResourceOptions::StorageModeShared)
        .expect("device allocates L2's output buffer");
    {
        let mut elapsed_samples = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let elapsed = time_dispatch_threads(
                &queue,
                &l2_pipeline,
                &[
                    (&no_copy_weight, weight_offset),
                    (&l2_output, 0),
                    (&blocks_per_row_uniform, 0),
                ],
                default_grid_threads,
                default_threadgroup_width,
            );
            elapsed_samples.push(elapsed);
        }
        let samples = gbps_samples(&elapsed_samples);
        let (mean, cov) = mean_and_cov(&samples);
        println!("arm=L2_dequant mean_gbps={mean:.2} cov_pct={cov:.2} samples={samples:?}");
        ladder_gbps.insert("L2_dequant", mean);
    }

    // ---- L3 baseline: the REAL production kernel, through the public
    // plan()/execute_plan() entry point -- byte-identical emitted MSL by
    // construction. ----
    let (program, sum) = matmul_program(ROWS as u32, IN_DIM as u32, DType::UInt8);
    let weight_slice = &mapped.as_slice()[weight_offset..weight_offset + tensor.byte_len as usize];
    let blocks = [
        QuantizedBlock::Q4K(weight_slice),
        QuantizedBlock::Float32(&activation),
    ];
    let mut plan = omega::metal::plan(&program, &[], &blocks, &[sum])
        .expect("plan resolves the real production q4_k matmul");
    let resident_names: std::collections::BTreeSet<&str> = ["weight", "activation"].into();
    plan.mark_resident(&resident_names);

    let mut l3_baseline_output: Vec<f32> = Vec::new();
    {
        let mut elapsed_samples = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let started = Instant::now();
            let evaluated = omega::metal::execute_plan(&plan, &blocks)
                .expect("execute_plan runs the real production q4_k matmul");
            elapsed_samples.push(started.elapsed());
            l3_baseline_output = evaluated.root().to_vec();
        }
        let samples = gbps_samples(&elapsed_samples);
        let (mean, cov) = mean_and_cov(&samples);
        println!(
            "arm=L3_baseline_production_execute_plan mean_gbps={mean:.2} cov_pct={cov:.2} \
             samples={samples:?} (end-to-end plan+dispatch+readback, see module doc)"
        );
        ladder_gbps.insert("L3_baseline", mean);
    }
    assert_parity(
        "L3_baseline vs cpu_reference",
        &l3_baseline_output[..PARITY_ROWS],
        &cpu_reference,
    );

    // ---- L3 shape sweep: same q4k_pair_dot body, hand-dispatched, varying
    // simdgroups/threadgroup, dispatchThreads vs dispatchThreadgroups, and
    // MTLMathMode across three compiled libraries. ----
    println!("=== L3 shape sweep (q4k_pair_dot, hand-dispatched) ===");
    let l3_shape_source_text = l3_shape_source();
    let l3_output = device
        .newBufferWithLength_options(ROWS * size_of::<f32>(), MTLResourceOptions::StorageModeShared)
        .expect("device allocates the L3 shape-sweep output buffer");

    let mut default_shape_output: Vec<f32> = Vec::new();
    for math_arm in MATH_MODE_ARMS {
        let pipeline = compile_pipeline(
            &device,
            &l3_shape_source_text,
            "q4k_matvec_l3_shape",
            math_arm.mode,
        );
        for simdgroups_per_tg in SIMDGROUPS_PER_THREADGROUP {
            let threadgroup_width = simdgroups_per_tg * 32;
            let threadgroup_count = TOTAL_SIMDGROUPS / simdgroups_per_tg;
            assert_eq!(
                threadgroup_count * simdgroups_per_tg,
                TOTAL_SIMDGROUPS,
                "row count must divide evenly by simdgroups_per_tg for this sweep"
            );
            let grid_threads = threadgroup_count * threadgroup_width;

            for dispatch_mode in [DispatchMode::Threads, DispatchMode::Threadgroups] {
                let dispatch_name = match dispatch_mode {
                    DispatchMode::Threads => "dispatchThreads",
                    DispatchMode::Threadgroups => "dispatchThreadgroups",
                };
                let mut elapsed_samples = Vec::with_capacity(REPEATS);
                for _ in 0..REPEATS {
                    let buffers: [(&ProtocolObject<dyn MTLBuffer>, usize); 4] = [
                        (&no_copy_weight, weight_offset),
                        (&activation_buffer, 0),
                        (&l3_output, 0),
                        (&blocks_per_row_uniform, 0),
                    ];
                    let elapsed = match dispatch_mode {
                        DispatchMode::Threads => time_dispatch_threads(
                            &queue,
                            &pipeline,
                            &buffers,
                            grid_threads,
                            threadgroup_width,
                        ),
                        DispatchMode::Threadgroups => time_dispatch_threadgroups(
                            &queue,
                            &pipeline,
                            &buffers,
                            threadgroup_count,
                            threadgroup_width,
                        ),
                    };
                    elapsed_samples.push(elapsed);
                }
                let samples = gbps_samples(&elapsed_samples);
                let (mean, cov) = mean_and_cov(&samples);
                println!(
                    "arm=L3_shape math_mode={:<8} simdgroups_per_tg={simdgroups_per_tg} \
                     dispatch={dispatch_name:<20} mean_gbps={mean:.2} cov_pct={cov:.2} \
                     samples={samples:?}",
                    math_arm.name
                );

                let is_production_default =
                    matches!(math_arm.name, "safe") && simdgroups_per_tg == 1 && matches!(dispatch_mode, DispatchMode::Threads);
                if is_production_default {
                    default_shape_output = read_f32_buffer(&l3_output, ROWS);
                    ladder_gbps.insert("L3_shape_default", mean);
                }
            }
        }
    }
    assert_parity(
        "L3_shape default arm vs cpu_reference",
        &default_shape_output[..PARITY_ROWS],
        &cpu_reference,
    );
    assert_parity(
        "L3_shape default arm vs L3_baseline",
        &default_shape_output[..PARITY_ROWS],
        &l3_baseline_output[..PARITY_ROWS],
    );

    println!("=== ladder ratios (limiter class = where the drop lands) ===");
    let l0 = ladder_gbps["L0_streaming"];
    let l1 = ladder_gbps["L1_header_decode"];
    let l2 = ladder_gbps["L2_dequant"];
    let l3_shape = ladder_gbps["L3_shape_default"];
    let l3_base = ladder_gbps["L3_baseline"];
    println!(
        "L0={l0:.2} L1={l1:.2} L2={l2:.2} L3_shape={l3_shape:.2} L3_baseline={l3_base:.2} GB/s"
    );
    println!("L1/L0={:.3} L2/L1={:.3} L3_shape/L2={:.3}", l1 / l0, l2 / l1, l3_shape / l2);
}
