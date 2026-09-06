//! Why does the packed-row `Q4_K` matvec cap at ~174-178 GB/s in-buffer when
//! a streaming kernel reaches the device ceiling (`omega/tests/
//! device_streaming_ceiling.rs`)? This file builds a ladder of hand-written
//! Metal kernels that each add exactly one layer of production's own work on
//! top of the last, over the SAME real weight bytes and the SAME production
//! dispatch shape (32-thread threadgroups, 4 rows per simdgroup
//! [`PACKED_ROW_ROWS_PER_GROUP`], `dispatchThreads`), so a rung-to-rung
//! bandwidth drop can be attributed to the specific thing that rung added
//! rather than guessed at.
//!
//! **Amortization fix (this landing).** ROW 289 ran this ladder's first cut
//! -- ONE dispatch over ONE `blk.0.ffn_up.weight` (33 MB) per timed command
//! buffer -- and found every arm's GB/s figure was noise: at production's own
//! ~180 GB/s ceiling, 33 MB takes ~0.18 ms of GPU time, BELOW the ~0.5 ms
//! empty-dispatch fixed cost that same row measured, so every timed buffer
//! was fixed-cost-and-scheduling-noise-dominated, not bandwidth-dominated
//! (CoV 35-120%, 4x run-to-run swings, no rung nameable as a limiter). The
//! fix is the same one production itself relies on: batch many independent
//! dispatches into ONE timed command buffer so the fixed per-dispatch cost
//! amortizes across real bandwidth. Every arm below now encodes
//! [`WEIGHT_TENSOR_COUNT`] dispatches of its kernel -- one per real,
//! DISTINCT `blk.{layer}.{ffn_up,ffn_gate}.weight` tensor across all
//! [`FFN_LAYERS`] layers of the SAME real checkpoint -- into a single
//! `computeCommandEncoder`, `endEncoding`s once, and times only the
//! `commit()`-`waitUntilCompleted()` span around all of them, moving
//! [`TOTAL_TIMED_BYTES`] per timed buffer (`MIN_TIMED_BYTES`'s own doc: two
//! full orders of magnitude past the fixed-cost floor). An `empty` arm
//! (`WEIGHT_TENSOR_COUNT` no-op dispatches, same batching) reports that fixed
//! per-dispatch cost directly, alongside every bandwidth number, rather than
//! leaving it to be inferred. `ffn_down.weight` is deliberately excluded from
//! the per-layer sweep -- see [`FFN_TENSOR_KINDS`]'s own doc for the two
//! independent reasons (a transposed shape and a mixed real quant codec in
//! this checkpoint) neither of which this ladder's kernels can absorb without
//! a second dispatch geometry, which is out of scope for this fix.
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
//!   ground truth. [`multi_tensor_matmul_program`] appends
//!   [`WEIGHT_TENSOR_COUNT`] independent multiply+reduce chains, all sharing
//!   ONE `activation` input node, into a SINGLE program -- `plan`/
//!   `execute_plan` already encode a whole program's ops into one command
//!   buffer (`omega::metal::execute_plan`'s own doc, `omega/src/
//!   metal.rs:2000-2145`: one `computeCommandEncoder`, `endEncoding`d once
//!   after every op), so this is the SAME batching every hand-dispatched arm
//!   below does, through the public API instead of a hand-rolled encoder.
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
//! Real weight bytes only (guiding-principles §9): every layer's
//! `blk.{0..31}.ffn_up.weight` and `blk.{0..31}.ffn_gate.weight` (`Q4_K`,
//! `[ROWS, IN_DIM]` each) from the same openchat-3.5-1210 checkpoint
//! `q4k_real_checkpoint_parity.rs`/`device_streaming_ceiling.rs` use, bound
//! with the SAME no-copy mapping technique
//! (`newBufferWithBytesNoCopy_length_options_deallocator` over the whole
//! page-rounded `mmap`, `StorageModeShared`, offset-addressed per tensor --
//! `device_streaming_ceiling.rs`'s own doc on `checkpoint_mapping_offset`'s
//! one-buffer-many-offsets shape) for every hand-dispatched arm (L0/L1/L2/L3
//! shape-sweep); the L3 baseline arm borrows the identical mmap'd byte
//! slices as [`proxima_tensor::QuantizedBlock::Q4K`] operands, which
//! `omega::metal::upload_packed_bytes`'s own `checkpoint_mapping_offset` arm
//! resolves to the same no-copy technique when a slice is not itself
//! page-aligned (real GGUF tensor offsets never are).
//!
//! GB/s for every non-`empty` arm is [`TOTAL_TIMED_BYTES`] -- the full
//! row-major byte extent of ALL [`WEIGHT_TENSOR_COUNT`] swept tensors, all
//! 144 bytes per Q4_K super-block, `ROWS * ROW_BYTES` each -- divided by
//! measured wall time; the same per-tensor-byte accounting
//! `device_streaming_ceiling.rs` and production's own decode-throughput
//! number use, summed across every tensor a timed buffer actually moved, NOT
//! a per-lane distinct-byte count (a lane's OWN issued loads only cover a
//! narrower slice of each 144-byte block; a per-lane count would read
//! smaller and would not be comparable to production's own GB/s figures).
//! This keeps every rung directly comparable on the same axis. The `empty`
//! arm reports ns/dispatch instead -- it moves no weight bytes by design, so
//! a GB/s figure for it would be meaningless.
//!
//! Timed by `commit()` -> `waitUntilCompleted()` around exactly
//! [`WEIGHT_TENSOR_COUNT`] dispatches per repeat (one dispatch per real,
//! distinct weight tensor, all encoded into ONE command buffer before that
//! buffer is ever committed), 5 repeats per arm (`REPEATS`), nothing
//! subtracted -- the encode loop itself (binding each dispatch's own weight
//! offset and output offset) happens entirely BEFORE the timer starts, same
//! convention `device_streaming_ceiling.rs` uses around its own single
//! dispatch. The L3 baseline arm is the one exception: it times
//! `omega::metal::execute_plan` end to end (upload resolution + all
//! [`WEIGHT_TENSOR_COUNT`] dispatches + readback of every requested output)
//! because assembling the emitted kernel's general uniform buffer by hand
//! would risk diverging from the exact bytes this arm exists to be faithful
//! to; the plan is built once and marked resident so repeats amortize upload
//! as much as the public API allows. This is a documented scope limitation,
//! not an oversight -- refining it (an `instrument`-gated per-op GPU
//! timestamp, `execute_plan_op_timed`) is a candidate for the measurement
//! slice, not this build slice.
//!
//! `#[ignore]`d: depends on a host-local openchat GGUF checkout outside this
//! repo, same convention as `q4k_real_checkpoint_parity.rs`/
//! `device_streaming_ceiling.rs`.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use core::ffi::c_void;
use core::ptr::NonNull;
use std::collections::{BTreeMap, BTreeSet};
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
use proxima_gguf::quant::q6_k;
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
/// `Vec`: every hand-dispatched arm in this file binds every tensor no-copy
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

/// Every real `blk.{layer}.{kind}.weight` tensor this ladder sweeps, in
/// dispatch order (layer-major: `blk.0.ffn_up`, `blk.0.ffn_gate`,
/// `blk.1.ffn_up`, ...) -- the SAME order every arm below encodes its
/// dispatches in, so index 0 always names `blk.0.ffn_up.weight` (the tensor
/// this file's parity checks are anchored to) in every arm.
fn locate_ffn_weight_tensors(parsed: &ParsedGguf, file_len: u64) -> Option<Vec<(String, RealTensor)>> {
    let mut tensors = Vec::with_capacity(WEIGHT_TENSOR_COUNT);
    for layer in 0..FFN_LAYERS {
        for kind in FFN_TENSOR_KINDS {
            let name = format!("blk.{layer}.{kind}.weight");
            let tensor = locate_real_tensor(parsed, file_len, &name, GgmlType::Q4_K)?;
            tensors.push((name, tensor));
        }
    }
    Some(tensors)
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
/// Bytes moved by dispatching one arm's kernel over ONE `[ROWS, IN_DIM]`
/// `Q4_K` tensor.
const TOTAL_WEIGHT_BYTES: u64 = (ROWS * ROW_BYTES) as u64;
const TOTAL_SIMDGROUPS: usize = ROWS / PACKED_ROWS_PER_GROUP;

/// This checkpoint's real layer count (`blk.0` through `blk.31`, confirmed
/// against the real file: `strings` over its header lists exactly 32
/// `blk.{n}.ffn_gate.weight` names) -- Mistral-7B shape, dense SwiGLU FFN
/// per layer (`ffn_gate`/`ffn_up`/`ffn_down`), no MoE router.
const FFN_LAYERS: usize = 32;

/// `ffn_down.weight` is deliberately EXCLUDED from this ladder's per-layer
/// sweep, for two independent reasons, either one sufficient on its own:
///
/// 1. **Transposed shape.** `blk.0.ffn_down.weight` is `in_dim=14336,
///    out_dim=4096` in this checkpoint (`proxima-tensor/docs/discipline.md`
///    ROW 63's own doc on that exact tensor) -- the reverse of `ffn_up`/
///    `ffn_gate`'s `in_dim=4096, out_dim=14336` this ladder's `ROWS`/
///    `IN_DIM`/`TOTAL_SIMDGROUPS` constants are sized for. Sweeping it would
///    need a SECOND dispatch geometry (`ROWS=4096`, different
///    `BLOCKS_PER_ROW`) this file's hand-written L0/L1/L2/L3-shape kernels
///    do not carry.
/// 2. **Mixed real quant codec.** 4 of this checkpoint's 32 `ffn_down`
///    tensors are `Q5_K`, not `Q4_K` (same ROW 63 inventory: `Q4_K x217,
///    F32 x65, Q5_K x8 [4 attn_v + 4 ffn_down], Q6_K x1`) -- and Metal has
///    no `Q5_K` unpack kernel at all (`omega::metal`'s own `"metal has no
///    q5_k/q6_k unpack kernel yet"` error string). Running those 4 layers'
///    real bytes through this file's `Q4_K`-only kernels would silently
///    mislabel their codec rather than measure real `Q4_K` bandwidth.
///
/// `ffn_up` + `ffn_gate` alone, both real, uniformly-shaped
/// (`in_dim=4096, out_dim=14336`), uniformly-`Q4_K` across all 32 layers,
/// already clear [`MIN_TIMED_BYTES`] -- the amortization floor this fix
/// exists for -- without either compromise.
const FFN_TENSOR_KINDS: [&str; 2] = ["ffn_up", "ffn_gate"];

/// One real, distinct weight tensor swept per timed command buffer, per arm
/// -- `FFN_LAYERS * FFN_TENSOR_KINDS.len()`.
const WEIGHT_TENSOR_COUNT: usize = FFN_LAYERS * FFN_TENSOR_KINDS.len();

/// Bytes moved by ONE timed command buffer once every [`WEIGHT_TENSOR_COUNT`]
/// dispatch is encoded into it -- this is the number [`gbps_samples`]
/// divides wall time by for every non-`empty` arm.
const TOTAL_TIMED_BYTES: u64 = TOTAL_WEIGHT_BYTES * WEIGHT_TENSOR_COUNT as u64;

/// The floor this whole fix exists to clear (ROW 289's own honest read): at
/// production's own ~180 GB/s ceiling a buffer this large takes
/// `MIN_TIMED_BYTES / 180e9` ~= 11.1 ms, two orders of magnitude past the
/// ~0.5 ms empty-dispatch fixed cost that row measured -- GB/s is
/// bandwidth-dominated, not fixed-cost-dominated, by construction. 2 GB
/// decimal (`1e9`, this file's own [`gbps_samples`] unit), NOT 2 GiB.
const MIN_TIMED_BYTES: u64 = 2_000_000_000;

const _: () = assert!(
    TOTAL_TIMED_BYTES >= MIN_TIMED_BYTES,
    "WEIGHT_TENSOR_COUNT must move at least MIN_TIMED_BYTES per timed command buffer"
);

// ---- Q6_K shape arms: the output-head shape (rows=32000, k=4096) ROW
// 319/output-head-read singles out (llama's own Q6_K kernel measures
// 45 GB/s in-program on `output.weight`, ours 41, both well under the
// 183-237 GB/s the Q4_K families reach and the 381.24 GB/s device ceiling
// ROW 296 measured), plus a layer-sized sibling (rows=4096) so shape
// scaling is visible. `IN_DIM` (4096) is shared with the Q4_K ladder above
// -- Q6_K's `output.weight` in this checkpoint has the same `k`. ----

/// `Q4_K`/`Q5_K`/`Q6_K` share one super-block element count (`QK_K == 256`,
/// each codec's own module doc, restated at `metal_parity.rs:2068-2071`).
const Q6K_BLOCKS_PER_ROW: usize = IN_DIM / q6_k::QK_K;
const Q6K_ROW_BYTES: usize = Q6K_BLOCKS_PER_ROW * q6_k::BLOCK_BYTES;

/// The output head's own shape: `output.weight` is `[32002, 4096]` in the
/// real openchat checkpoint (`q6k_real_checkpoint_parity.rs`'s own
/// `real_tensor_bytes` call); 32000 is the round shape this arm sweeps,
/// per the read this arm exists to follow up (`output-head-read.md`).
const Q6K_HEAD_ROWS: usize = 32_000;
/// A layer-sized `Q6_K` matrix -- `Q4_K`'s own `ffn_up`/`ffn_gate` shape's
/// row count would be `ROWS` (14336), but no real layer tensor in this
/// checkpoint is `Q6_K`-encoded (`output.weight` is the LONE one), so 4096
/// is chosen to keep `Q6K_LAYER_TENSOR_BYTES` an easy comparison point
/// against `TOTAL_WEIGHT_BYTES / (ROWS / IN_DIM)`-scale reasoning, not
/// because a real 4096-row `Q6_K` tensor exists in this file.
const Q6K_LAYER_ROWS: usize = 4_096;

const Q6K_HEAD_TENSOR_BYTES: u64 = (Q6K_HEAD_ROWS * Q6K_ROW_BYTES) as u64;
const Q6K_LAYER_TENSOR_BYTES: u64 = (Q6K_LAYER_ROWS * Q6K_ROW_BYTES) as u64;

/// One real `output.weight`-shaped `Q6_K` tensor is 107.5 MB
/// (`32000 * 16 * 210` bytes) -- `WEIGHT_TENSOR_COUNT`'s own 64-tensor
/// convention would move 6.9 GB per timed buffer for this shape, so this
/// arm instead uses the smallest tensor count that still clears
/// [`MIN_TIMED_BYTES`] (2 GB decimal), same amortization-floor reasoning as
/// this file's module doc: `ceil(2e9 / 107_520_000) = 19`.
const Q6K_HEAD_TENSOR_COUNT: usize = MIN_TIMED_BYTES.div_ceil(Q6K_HEAD_TENSOR_BYTES) as usize;
/// Same reasoning at the layer shape: `ceil(2e9 / 13_762_560) = 146`.
const Q6K_LAYER_TENSOR_COUNT: usize = MIN_TIMED_BYTES.div_ceil(Q6K_LAYER_TENSOR_BYTES) as usize;

const _: () = assert!(
    Q6K_HEAD_TENSOR_BYTES * Q6K_HEAD_TENSOR_COUNT as u64 >= MIN_TIMED_BYTES,
    "Q6K_HEAD_TENSOR_COUNT must move at least MIN_TIMED_BYTES per timed command buffer"
);
const _: () = assert!(
    Q6K_LAYER_TENSOR_BYTES * Q6K_LAYER_TENSOR_COUNT as u64 >= MIN_TIMED_BYTES,
    "Q6K_LAYER_TENSOR_COUNT must move at least MIN_TIMED_BYTES per timed command buffer"
);

/// llama's own device streaming ceiling on this M1 Max (ROW 296,
/// `proxima-tensor/docs/discipline.md`), decimal GB/s -- reported alongside
/// every Q6_K shape arm's own GB/s as a ratio, not re-measured here.
const DEVICE_CEILING_GBPS: f64 = 381.24;
/// ROW 319's in-program measurement of llama's ported `Q6_K` kernel on the
/// real `output.weight` dispatch (2.38 ms for 107.5 MB) -- the number this
/// arm's isolated measurement is checked against.
const ROW_319_IN_PROGRAM_GBPS: f64 = 45.0;

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

/// The `empty` arm's kernel -- reads no weight bytes at all, writes one word
/// so the dispatch is not dead-code-eliminated. Same 32-thread threadgroup
/// convention every other arm uses. See this file's module doc for why this
/// arm exists (reporting the fixed per-dispatch cost ROW 289's floor bug was
/// hiding).
const EMPTY_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void q4k_matvec_empty(
    device uint* out [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid == 0u) {
        out[0] = 1u;
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

/// Encodes `weight_offsets.len()` dispatches of `pipeline` into ONE command
/// buffer -- L0/L1/L2's buffer layout (`weight@0`, per-dispatch `output@1`
/// at `dispatch_index * output_dispatch_stride_bytes`, fixed `uniform@2`) --
/// then times exactly the `commit()`-`waitUntilCompleted()` span around all
/// of them (module doc: the encode loop above it is never timed).
// each argument is a distinct real Metal buffer/geometry parameter this
// batched dispatch needs bound per-call; splitting them into a struct would
// not reduce what a caller has to supply.
#[allow(clippy::too_many_arguments)]
fn time_batch_l0l1l2(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    weight: &ProtocolObject<dyn MTLBuffer>,
    weight_offsets: &[usize],
    output: &ProtocolObject<dyn MTLBuffer>,
    output_dispatch_stride_bytes: usize,
    uniform: &ProtocolObject<dyn MTLBuffer>,
    grid_threads: usize,
    threadgroup_width: usize,
) -> Duration {
    let command_buffer = queue.commandBuffer().expect("command buffer");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("command buffer refused to hand out a compute encoder");
    encoder.setComputePipelineState(pipeline);
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
    for (dispatch_index, &weight_offset) in weight_offsets.iter().enumerate() {
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weight), weight_offset, 0);
            encoder.setBuffer_offset_atIndex(
                Some(output),
                dispatch_index * output_dispatch_stride_bytes,
                1,
            );
            encoder.setBuffer_offset_atIndex(Some(uniform), 0, 2);
        }
        encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
    }
    encoder.endEncoding();
    let started = Instant::now();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    started.elapsed()
}

/// L3 shape-sweep's buffer layout (`weight@0`, fixed `activation@1`,
/// per-dispatch `output@2`, fixed `uniform@3`), `dispatchThreads` variant.
// each argument is a distinct real Metal buffer/geometry parameter this
// batched dispatch needs bound per-call; splitting them into a struct would
// not reduce what a caller has to supply.
#[allow(clippy::too_many_arguments)]
fn time_batch_l3_shape_threads(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    weight: &ProtocolObject<dyn MTLBuffer>,
    weight_offsets: &[usize],
    activation: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    output_dispatch_stride_bytes: usize,
    uniform: &ProtocolObject<dyn MTLBuffer>,
    grid_threads: usize,
    threadgroup_width: usize,
) -> Duration {
    let command_buffer = queue.commandBuffer().expect("command buffer");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("command buffer refused to hand out a compute encoder");
    encoder.setComputePipelineState(pipeline);
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
    for (dispatch_index, &weight_offset) in weight_offsets.iter().enumerate() {
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weight), weight_offset, 0);
            encoder.setBuffer_offset_atIndex(Some(activation), 0, 1);
            encoder.setBuffer_offset_atIndex(
                Some(output),
                dispatch_index * output_dispatch_stride_bytes,
                2,
            );
            encoder.setBuffer_offset_atIndex(Some(uniform), 0, 3);
        }
        encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
    }
    encoder.endEncoding();
    let started = Instant::now();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    started.elapsed()
}

/// Same layout as [`time_batch_l3_shape_threads`], `dispatchThreadgroups`
/// variant.
// each argument is a distinct real Metal buffer/geometry parameter this
// batched dispatch needs bound per-call; splitting them into a struct would
// not reduce what a caller has to supply.
#[allow(clippy::too_many_arguments)]
fn time_batch_l3_shape_threadgroups(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    weight: &ProtocolObject<dyn MTLBuffer>,
    weight_offsets: &[usize],
    activation: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    output_dispatch_stride_bytes: usize,
    uniform: &ProtocolObject<dyn MTLBuffer>,
    threadgroup_count: usize,
    threadgroup_width: usize,
) -> Duration {
    let command_buffer = queue.commandBuffer().expect("command buffer");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("command buffer refused to hand out a compute encoder");
    encoder.setComputePipelineState(pipeline);
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
    for (dispatch_index, &weight_offset) in weight_offsets.iter().enumerate() {
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weight), weight_offset, 0);
            encoder.setBuffer_offset_atIndex(Some(activation), 0, 1);
            encoder.setBuffer_offset_atIndex(
                Some(output),
                dispatch_index * output_dispatch_stride_bytes,
                2,
            );
            encoder.setBuffer_offset_atIndex(Some(uniform), 0, 3);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, threadgroup);
    }
    encoder.endEncoding();
    let started = Instant::now();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    started.elapsed()
}

/// The `empty` arm: `count` dispatches of a kernel that reads no weight
/// bytes at all, encoded into ONE command buffer the same way every other
/// arm is -- reports the fixed per-dispatch floor ROW 289 found dominating
/// every arm's GB/s before this fix (module doc).
fn time_batch_empty(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    output: &ProtocolObject<dyn MTLBuffer>,
    count: usize,
    grid_threads: usize,
    threadgroup_width: usize,
) -> Duration {
    let command_buffer = queue.commandBuffer().expect("command buffer");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("command buffer refused to hand out a compute encoder");
    encoder.setComputePipelineState(pipeline);
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
    for dispatch_index in 0..count {
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(output), dispatch_index * size_of::<u32>(), 0);
        }
        encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
    }
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

fn gbps_samples(elapsed_samples: &[Duration], total_bytes: u64) -> Vec<f64> {
    elapsed_samples
        .iter()
        .map(|elapsed| total_bytes as f64 / elapsed.as_secs_f64() / 1e9)
        .collect()
}

fn ns_per_dispatch_samples(elapsed_samples: &[Duration], dispatch_count: usize) -> Vec<f64> {
    elapsed_samples
        .iter()
        .map(|elapsed| elapsed.as_secs_f64() * 1e9 / dispatch_count as f64)
        .collect()
}

/// The multi-tensor sibling of the single-tensor `[rows,k]x[k,1]->[rows,1]`
/// program every prior real-checkpoint parity test in this crate builds
/// (`q4k_real_checkpoint_parity.rs`'s own `matmul_program`):
/// `weight_names.len()` independent `Op::Input`+multiply+reduce chains, ALL
/// sharing the ONE `activation` `Op::Input` node, appended into a SINGLE
/// program so `omega::metal::plan`/`execute_plan` encode every one of them
/// into the SAME command buffer (this file's module doc). Named inputs here
/// (`Plan::mark_resident` matches by name) so the caller can mark every
/// weight AND the shared activation resident across repeats.
fn multi_tensor_matmul_program(
    weight_names: &[String],
    rows: u32,
    k: u32,
    weight_dtype: DType,
) -> (Vec<Op>, Vec<NodeId>) {
    let mut program = Vec::new();
    let weight_nodes: Vec<NodeId> = weight_names
        .iter()
        .map(|name| {
            append(
                &mut program,
                Op::Input {
                    dtype: weight_dtype,
                    shape: vec![Extent::Static(rows), Extent::Static(k)],
                    name: Some(name.clone()),
                },
            )
        })
        .collect();
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(k), Extent::Static(1)],
            name: Some("activation".into()),
        },
    );
    let mut sums = Vec::with_capacity(weight_nodes.len());
    for (index, weight) in weight_nodes.into_iter().enumerate() {
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
                name: Some(format!("q4k_ladder_matmul_{index}")),
            }),
        );
        sums.push(sum);
    }
    (program, sums)
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
    let Some(ffn_tensors) = locate_ffn_weight_tensors(&parsed, file_len) else {
        return;
    };
    assert_eq!(ffn_tensors.len(), WEIGHT_TENSOR_COUNT, "every layer's ffn_up/ffn_gate must resolve");
    for (name, tensor) in &ffn_tensors {
        assert_eq!(
            tensor.in_dim, IN_DIM,
            "{name} in_dim must match the {IN_DIM} this ladder is sized for"
        );
        assert_eq!(
            tensor.out_dim, ROWS,
            "{name} out_dim must match the {ROWS} this ladder is sized for"
        );
        assert_eq!(
            tensor.byte_len as usize,
            ROWS * ROW_BYTES,
            "{name} declared tensor byte length matches rows*row_bytes"
        );
    }
    let weight_names: Vec<String> = ffn_tensors.iter().map(|(name, _)| name.clone()).collect();
    let weight_offsets: Vec<usize> = ffn_tensors.iter().map(|(_, tensor)| tensor.byte_offset as usize).collect();

    let mapped = MappedFile::open(path).expect("mmap the real openchat checkpoint");
    let parity_weight_offset = ffn_tensors[0].1.byte_offset as usize;
    let weight_bytes_for_cpu_reference =
        &mapped.as_slice()[parity_weight_offset..parity_weight_offset + PARITY_ROWS * ROW_BYTES];

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
         threadgroups, {PACKED_ROWS_PER_GROUP} rows/simdgroup, dispatchThreads), \
         {WEIGHT_TENSOR_COUNT} real distinct tensors ({TOTAL_TIMED_BYTES} bytes) per timed \
         command buffer ==="
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
        .newBufferWithLength_options(
            WEIGHT_TENSOR_COUNT * ROWS * size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
        .expect("device allocates L0's output buffer");
    {
        let mut elapsed_samples = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let elapsed = time_batch_l0l1l2(
                &queue,
                &l0_pipeline,
                &no_copy_weight,
                &weight_offsets,
                &l0_output,
                ROWS * size_of::<u32>(),
                &blocks_per_row_uniform,
                default_grid_threads,
                default_threadgroup_width,
            );
            elapsed_samples.push(elapsed);
        }
        let samples = gbps_samples(&elapsed_samples, TOTAL_TIMED_BYTES);
        let (mean, cov) = mean_and_cov(&samples);
        println!(
            "arm=L0_streaming mean_gbps={mean:.2} cov_pct={cov:.2} samples={samples:?} \
             dispatches={WEIGHT_TENSOR_COUNT} bytes={TOTAL_TIMED_BYTES}"
        );
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
        .newBufferWithLength_options(
            WEIGHT_TENSOR_COUNT * ROWS * size_of::<f32>(),
            MTLResourceOptions::StorageModeShared,
        )
        .expect("device allocates L1's output buffer");
    {
        let mut elapsed_samples = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let elapsed = time_batch_l0l1l2(
                &queue,
                &l1_pipeline,
                &no_copy_weight,
                &weight_offsets,
                &l1_output,
                ROWS * size_of::<f32>(),
                &blocks_per_row_uniform,
                default_grid_threads,
                default_threadgroup_width,
            );
            elapsed_samples.push(elapsed);
        }
        let samples = gbps_samples(&elapsed_samples, TOTAL_TIMED_BYTES);
        let (mean, cov) = mean_and_cov(&samples);
        println!(
            "arm=L1_header_decode mean_gbps={mean:.2} cov_pct={cov:.2} samples={samples:?} \
             dispatches={WEIGHT_TENSOR_COUNT} bytes={TOTAL_TIMED_BYTES}"
        );
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
        .newBufferWithLength_options(
            WEIGHT_TENSOR_COUNT * ROWS * size_of::<f32>(),
            MTLResourceOptions::StorageModeShared,
        )
        .expect("device allocates L2's output buffer");
    {
        let mut elapsed_samples = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let elapsed = time_batch_l0l1l2(
                &queue,
                &l2_pipeline,
                &no_copy_weight,
                &weight_offsets,
                &l2_output,
                ROWS * size_of::<f32>(),
                &blocks_per_row_uniform,
                default_grid_threads,
                default_threadgroup_width,
            );
            elapsed_samples.push(elapsed);
        }
        let samples = gbps_samples(&elapsed_samples, TOTAL_TIMED_BYTES);
        let (mean, cov) = mean_and_cov(&samples);
        println!(
            "arm=L2_dequant mean_gbps={mean:.2} cov_pct={cov:.2} samples={samples:?} \
             dispatches={WEIGHT_TENSOR_COUNT} bytes={TOTAL_TIMED_BYTES}"
        );
        ladder_gbps.insert("L2_dequant", mean);
    }

    // ---- L3 baseline: the REAL production kernel, through the public
    // plan()/execute_plan() entry point -- byte-identical emitted MSL by
    // construction, all WEIGHT_TENSOR_COUNT tensors in ONE program so
    // execute_plan encodes all of them into ONE command buffer. ----
    let (program, sums) = multi_tensor_matmul_program(&weight_names, ROWS as u32, IN_DIM as u32, DType::UInt8);
    let weight_slices: Vec<&[u8]> = ffn_tensors
        .iter()
        .map(|(_, tensor)| {
            let offset = tensor.byte_offset as usize;
            &mapped.as_slice()[offset..offset + tensor.byte_len as usize]
        })
        .collect();
    let mut blocks: Vec<QuantizedBlock<'_>> =
        weight_slices.iter().map(|slice| QuantizedBlock::Q4K(slice)).collect();
    blocks.push(QuantizedBlock::Float32(&activation));

    let mut plan = omega::metal::plan(&program, &[], &blocks, &sums)
        .expect("plan resolves the real production q4_k matmul over every ffn tensor");
    let mut resident_names: BTreeSet<&str> = weight_names.iter().map(String::as_str).collect();
    resident_names.insert("activation");
    plan.mark_resident(&resident_names);

    let mut l3_baseline_layer0_up: Vec<f32> = Vec::new();
    {
        let mut elapsed_samples = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let started = Instant::now();
            let evaluated = omega::metal::execute_plan(&plan, &blocks)
                .expect("execute_plan runs the real production q4_k matmul over every ffn tensor");
            elapsed_samples.push(started.elapsed());
            l3_baseline_layer0_up = evaluated
                .get(sums[0])
                .map(|(data, _)| data.to_vec())
                .expect("blk.0.ffn_up's own reduce node is a requested output");
        }
        let samples = gbps_samples(&elapsed_samples, TOTAL_TIMED_BYTES);
        let (mean, cov) = mean_and_cov(&samples);
        println!(
            "arm=L3_baseline_production_execute_plan mean_gbps={mean:.2} cov_pct={cov:.2} \
             samples={samples:?} dispatches={WEIGHT_TENSOR_COUNT} bytes={TOTAL_TIMED_BYTES} \
             (end-to-end plan+dispatch+readback, see module doc)"
        );
        ladder_gbps.insert("L3_baseline", mean);
    }
    assert_parity(
        "L3_baseline vs cpu_reference",
        &l3_baseline_layer0_up[..PARITY_ROWS],
        &cpu_reference,
    );

    // ---- L3 shape sweep: same q4k_pair_dot body, hand-dispatched, varying
    // simdgroups/threadgroup, dispatchThreads vs dispatchThreadgroups, and
    // MTLMathMode across three compiled libraries -- WEIGHT_TENSOR_COUNT
    // dispatches per cell, one command buffer per cell. ----
    println!("=== L3 shape sweep (q4k_pair_dot, hand-dispatched) ===");
    let l3_shape_source_text = l3_shape_source();
    let l3_output = device
        .newBufferWithLength_options(
            WEIGHT_TENSOR_COUNT * ROWS * size_of::<f32>(),
            MTLResourceOptions::StorageModeShared,
        )
        .expect("device allocates the L3 shape-sweep output buffer");

    let mut default_shape_layer0_up: Vec<f32> = Vec::new();
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
                    let elapsed = match dispatch_mode {
                        DispatchMode::Threads => time_batch_l3_shape_threads(
                            &queue,
                            &pipeline,
                            &no_copy_weight,
                            &weight_offsets,
                            &activation_buffer,
                            &l3_output,
                            ROWS * size_of::<f32>(),
                            &blocks_per_row_uniform,
                            grid_threads,
                            threadgroup_width,
                        ),
                        DispatchMode::Threadgroups => time_batch_l3_shape_threadgroups(
                            &queue,
                            &pipeline,
                            &no_copy_weight,
                            &weight_offsets,
                            &activation_buffer,
                            &l3_output,
                            ROWS * size_of::<f32>(),
                            &blocks_per_row_uniform,
                            threadgroup_count,
                            threadgroup_width,
                        ),
                    };
                    elapsed_samples.push(elapsed);
                }
                let samples = gbps_samples(&elapsed_samples, TOTAL_TIMED_BYTES);
                let (mean, cov) = mean_and_cov(&samples);
                println!(
                    "arm=L3_shape math_mode={:<8} simdgroups_per_tg={simdgroups_per_tg} \
                     dispatch={dispatch_name:<20} mean_gbps={mean:.2} cov_pct={cov:.2} \
                     samples={samples:?} dispatches={WEIGHT_TENSOR_COUNT} bytes={TOTAL_TIMED_BYTES}",
                    math_arm.name
                );

                let is_production_default = matches!(math_arm.name, "safe")
                    && simdgroups_per_tg == 1
                    && matches!(dispatch_mode, DispatchMode::Threads);
                if is_production_default {
                    let full = read_f32_buffer(&l3_output, WEIGHT_TENSOR_COUNT * ROWS);
                    default_shape_layer0_up = full[..ROWS].to_vec();
                    ladder_gbps.insert("L3_shape_default", mean);
                }
            }
        }
    }
    assert_parity(
        "L3_shape default arm vs cpu_reference",
        &default_shape_layer0_up[..PARITY_ROWS],
        &cpu_reference,
    );
    assert_parity(
        "L3_shape default arm vs L3_baseline",
        &default_shape_layer0_up[..PARITY_ROWS],
        &l3_baseline_layer0_up[..PARITY_ROWS],
    );

    // ---- empty: WEIGHT_TENSOR_COUNT no-op dispatches, one command buffer
    // -- the fixed per-dispatch cost every bandwidth arm above pays
    // WEIGHT_TENSOR_COUNT times, reported directly instead of inferred. ----
    println!("=== empty arm ({WEIGHT_TENSOR_COUNT} no-op dispatches, one command buffer) ===");
    let empty_pipeline = compile_pipeline(&device, EMPTY_KERNEL_SOURCE, "q4k_matvec_empty", MTLMathMode::Safe);
    let empty_output = device
        .newBufferWithLength_options(
            WEIGHT_TENSOR_COUNT * size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
        .expect("device allocates the empty arm's output buffer");
    {
        let mut elapsed_samples = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let elapsed = time_batch_empty(&queue, &empty_pipeline, &empty_output, WEIGHT_TENSOR_COUNT, 32, 32);
            elapsed_samples.push(elapsed);
        }
        let ns_samples = ns_per_dispatch_samples(&elapsed_samples, WEIGHT_TENSOR_COUNT);
        let (mean, cov) = mean_and_cov(&ns_samples);
        println!(
            "arm=empty mean_ns_per_dispatch={mean:.1} cov_pct={cov:.2} samples={ns_samples:?} \
             dispatches={WEIGHT_TENSOR_COUNT}"
        );
    }

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

/// Synthesizes `tensor_count` distinct `Q6_K`-encoded tensors, each
/// `[rows, k]`, through the SAME encoder production's own real `Q6_K`
/// tensors are quantized with (`proxima_gguf::quant::q6_k::quantize`) --
/// guiding-principles §9's fallback for the one case in this file where the
/// real artifact (`output.weight`) exists only ONCE in the checkpoint, not
/// in the quantity (19/146) this harness's amortization floor needs. Each
/// tensor gets its own [`Lcg`] seed (`seed_base + tensor_index`) so no two
/// tensors are bit-identical and a GPU cache cannot serve one tensor's read
/// from another's -- the same "distinct tensors" posture the real `Q4_K`
/// arms above get for free from 32 real, distinct checkpoint tensors. One
/// row's worth of `f32` (`k` elements) is reused per row rather than
/// materializing the whole tensor in `f32` first, so peak host memory stays
/// at the quantized output size, not `4x` that.
fn synth_q6k_weight_bytes(seed_base: u64, tensor_count: usize, rows: usize, row_bytes: usize, k: usize) -> Vec<u8> {
    let tensor_bytes = rows * row_bytes;
    let mut bytes = vec![0u8; tensor_bytes * tensor_count];
    let mut row_f32 = vec![0.0f32; k];
    for (tensor_index, tensor_slice) in bytes.chunks_exact_mut(tensor_bytes).enumerate() {
        let mut lcg = Lcg(seed_base + tensor_index as u64);
        for row_blocks in tensor_slice.chunks_exact_mut(row_bytes) {
            for value in row_f32.iter_mut() {
                *value = lcg.next_unit() * 4.0 - 2.0;
            }
            q6_k::quantize(&row_f32, row_blocks).expect("row length is a whole multiple of QK_K");
        }
    }
    bytes
}

/// CPU oracle for the first [`PARITY_ROWS`] rows of the FIRST synthesized
/// tensor -- same posture as [`cpu_reference_first_rows`], `Q6_K`'s own
/// `dequantize`/dot-product instead of `Q4_K`'s.
fn q6k_cpu_reference_first_rows(first_tensor_bytes: &[u8], row_bytes: usize, k: usize, activation: &[f32]) -> Vec<f32> {
    let mut reference = Vec::with_capacity(PARITY_ROWS);
    let mut dequantized_row = vec![0.0f32; k];
    for row_blocks in first_tensor_bytes[..PARITY_ROWS * row_bytes].chunks_exact(row_bytes) {
        q6_k::dequantize(row_blocks, &mut dequantized_row).expect("a whole number of q6_k super-blocks per row");
        let dot: f32 = dequantized_row.iter().zip(activation.iter()).map(|(weight, act)| weight * act).sum();
        reference.push(dot);
    }
    reference
}

/// Runs one `Q6_K` shape arm: synthesizes `tensor_count` distinct
/// `[rows, IN_DIM]` tensors, dispatches PRODUCTION's own emitted `Q6_K`
/// kernel over all of them through `omega::metal::plan`/`execute_plan` --
/// the SAME public entry point `L3_baseline` uses for `Q4_K` above, so the
/// MSL text is whatever `omega::msl::emit` renders today (byte-identical to
/// production by construction) and the dispatch geometry
/// (`packed_row_dispatch`, `Q6K`'s `rows_per_simdgroup() == 1`,
/// `omega/src/msl.rs:1148`, `nsg2` default-on) is production's, never
/// hand-assembled here -- and reports GB/s plus CoV over [`REPEATS`], with
/// ratios to ROW 319's in-program 45 GB/s and to the 381.24 GB/s device
/// ceiling (ROW 296), so an in-isolation vs in-program gap reads directly
/// off this arm's own printed line.
fn run_q6k_shape_arm(label: &str, rows: usize, tensor_count: usize, seed_base: u64) {
    let row_bytes = Q6K_ROW_BYTES;
    let tensor_bytes = (rows * row_bytes) as u64;
    let total_timed_bytes = tensor_bytes * tensor_count as u64;
    println!(
        "=== q6_k shape arm {label}: rows={rows} k={IN_DIM} tensors={tensor_count} \
         tensor_bytes={tensor_bytes} total_timed_bytes={total_timed_bytes} \
         (production Q6_K kernel via omega::metal::plan/execute_plan) ==="
    );

    let weight_bytes = synth_q6k_weight_bytes(seed_base, tensor_count, rows, row_bytes, IN_DIM);
    let weight_names: Vec<String> = (0..tensor_count).map(|index| format!("q6k_{label}_{index}")).collect();

    let mut activation_lcg = Lcg(seed_base + 999);
    let activation: Vec<f32> = (0..IN_DIM).map(|_| activation_lcg.next_unit() * 4.0 - 2.0).collect();

    let cpu_reference =
        q6k_cpu_reference_first_rows(&weight_bytes[..rows * row_bytes], row_bytes, IN_DIM, &activation);

    let (program, sums) =
        multi_tensor_matmul_program(&weight_names, rows as u32, IN_DIM as u32, DType::UInt8);
    let weight_slices: Vec<&[u8]> = weight_bytes.chunks_exact(rows * row_bytes).collect();
    let mut blocks: Vec<QuantizedBlock<'_>> =
        weight_slices.iter().map(|slice| QuantizedBlock::Q6K(slice)).collect();
    blocks.push(QuantizedBlock::Float32(&activation));

    let mut plan = omega::metal::plan(&program, &[], &blocks, &sums)
        .expect("plan resolves the synthetic q6_k matmul over every tensor");
    let mut resident_names: BTreeSet<&str> = weight_names.iter().map(String::as_str).collect();
    resident_names.insert("activation");
    plan.mark_resident(&resident_names);

    let mut first_tensor_output: Vec<f32> = Vec::new();
    let mut elapsed_samples = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let started = Instant::now();
        let evaluated = omega::metal::execute_plan(&plan, &blocks)
            .expect("execute_plan runs the synthetic q6_k matmul over every tensor");
        elapsed_samples.push(started.elapsed());
        first_tensor_output = evaluated
            .get(sums[0])
            .map(|(data, _)| data.to_vec())
            .expect("first tensor's own reduce node is a requested output");
    }
    let samples = gbps_samples(&elapsed_samples, total_timed_bytes);
    let (mean, cov) = mean_and_cov(&samples);
    println!(
        "arm=Q6K_{label} rows={rows} tensors={tensor_count} mean_gbps={mean:.2} cov_pct={cov:.2} \
         samples={samples:?} bytes={total_timed_bytes} ratio_to_row319_in_program={:.3} \
         ratio_to_device_ceiling={:.3}",
        mean / ROW_319_IN_PROGRAM_GBPS,
        mean / DEVICE_CEILING_GBPS,
    );
    // ROW 322: the shared PARITY_MAX_ABS_ERROR (1e-4) is an absolute bound --
    // this print makes the batch's own output magnitude visible so an
    // absolute-vs-relative read is a fact, not a guess.
    let batch_peak = cpu_reference.iter().fold(0.0f32, |peak, value| peak.max(value.abs()));
    let max_abs_error = first_tensor_output[..PARITY_ROWS]
        .iter()
        .zip(cpu_reference.iter())
        .fold(0.0f32, |max_error, (got, want)| max_error.max((got - want).abs()));
    println!(
        "Q6K_{label} batch_peak_abs={batch_peak:.4} max_abs_error={max_abs_error:.4} \
         relative_to_peak={:.6}",
        max_abs_error / batch_peak.max(f32::EPSILON),
    );
    assert_parity(
        &format!("Q6K_{label} vs cpu_reference"),
        &first_tensor_output[..PARITY_ROWS],
        &cpu_reference,
    );
}

/// Isolates the output head's `Q6_K` shape (rows=32000, k=4096) from any
/// in-program effect -- terminal dispatch position, unamortized
/// per-dispatch floor at count=1 -- that ROW 310/314's in-buffer ablation
/// could not separate from the kernel's own bandwidth
/// (`output-head-read.md`'s candidates 2/3). Also runs the layer-sized
/// sibling (rows=4096) so shape scaling is visible on the same axis.
///
/// `#[ignore]`d: synthesizes ~2 GB of `Q6_K` bytes per arm (19 tensors at
/// the head shape, 146 at the layer shape) through the real quantize
/// encoder -- CPU-bound minutes of work per arm -- and depends on a real
/// Metal device, same posture as this file's GGUF-dependent arms above,
/// though this one needs no external fixture file.
#[test]
#[ignore = "synthesizes ~2 GB of q6_k bytes per arm through the real encoder and needs a real metal device"]
fn q6k_head_and_layer_shape_roofline_ladder() {
    run_q6k_shape_arm("head", Q6K_HEAD_ROWS, Q6K_HEAD_TENSOR_COUNT, 4096);
    run_q6k_shape_arm("layer", Q6K_LAYER_ROWS, Q6K_LAYER_TENSOR_COUNT, 8192);
}
