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
//! buffer is ever committed). Every arm runs one UNTIMED warm-up dispatch of
//! its own command buffer first, immediately discarded, then [`REPEATS`]
//! timed repeats (ROW 335: ROW 296/302/331 all saw the same cold-first-
//! dispatch signature, so the warm-up removes that outlier from the timed set
//! rather than leaving it to be averaged around; [`warmed_up_samples`] is the
//! one function every arm below calls for this). Nothing is subtracted from
//! a timed repeat itself -- the encode loop (binding each dispatch's own
//! weight offset and output offset) happens entirely BEFORE the timer
//! starts, same convention `device_streaming_ceiling.rs` uses around its own
//! single dispatch. Every arm reports the MEDIAN of its timed repeats as the
//! headline number (robust to a single outlier repeat), alongside min, max,
//! and CoV, and keeps the mean for continuity with every prior ladder row's
//! own printed number. The L3 baseline arm is the one exception: it times
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

use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;

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
use proxima_gguf::quant::{QuantError, q4_k, q5_k, q6_k};
use proxima_gguf::types::GgmlType;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    BoundOp, BoundOpKind, DType, Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce,
    ReduceInit, ScalarOp, append, bind, correct_packed_matmul_layouts, infer, map,
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
/// ROW 310's own in-program per-family bandwidth table
/// (`proxima-tensor/docs/discipline.md:23058-23068`, the in-buffer ablation
/// `cost = FULL - arm` / `bytes` / `GB/s`), restated here as a small lookup
/// so [`run_shape_arm`] can print each isolated arm's GB/s beside the
/// in-program number for the SAME family on the SAME line -- never averaged
/// or combined into one derived number (guiding-principles §18/§19: these
/// are two independent MEASUREMENTs of two different dispatch contexts,
/// concurrent-overlapped-in-program vs isolated-in-this-file). `attn_q`/
/// `attn_k`/`attn_v` are `None` -- ROW 310 measured them noise-negative
/// (`attn_q`'s own CoV 12.75% blew through this campaign's 5% bound), not
/// zero throughput; a `None` here reads as "not comparable", never as "0
/// GB/s". `head` names `output.weight`'s own row (`Q6_K`, 41.63 GB/s).
const ROW_310_IN_PROGRAM_GBPS: &[(&str, Option<f64>)] = &[
    ("attn_q", None),
    ("attn_k", None),
    ("attn_v", None),
    ("head", Some(41.63)),
    ("attn_output", Some(144.94)),
    ("ffn_up", Some(183.48)),
    ("ffn_down", Some(200.46)),
    ("ffn_gate", Some(236.55)),
];

const REPEATS: usize = 7;
const PARITY_ROWS: usize = 64;

/// Parity is measured relative to the batch's own peak reference magnitude,
/// never as a fixed absolute bound -- a per-row-relative check explodes near
/// zero crossings, and a fixed absolute check breaks the other way once
/// output magnitude climbs (workspace rule: normalize by the BATCH PEAK).
/// ROW 321/322 hit exactly that failure mode: the Q6_K head arm's reference
/// output peaks near 17379 (a 4096-term dot product), its `max_abs_error`
/// (0.029) is `2e-6` relative to that peak -- below float32's own
/// representable ULP at that magnitude (`17379 * 1.19e-7 ~= 0.002`) -- yet
/// the old fixed `1e-4` absolute gate panicked on it regardless, aborting
/// the test before the `layer` arm ever ran. `1e-5` is the tightest
/// power-of-ten bound above every arm's own measured relative error in this
/// file: the Q4_K arms (`L3_baseline`/`L3_shape` vs `cpu_reference`) measure
/// `max_abs_error=1.9073486e-6` against a real-checkpoint reference peak of
/// `1.5504022` -- `1.23e-6` relative -- and the Q6_K head arm measures
/// `1.69e-6` relative (`0.029296875 / 17379.0879`); both sit an order of
/// magnitude below this bound.
const PARITY_MAX_REL_ERROR: f32 = 1e-5;

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

/// `L3_SHAPE_KERNEL_BODY`'s own `Q6_K` counterpart, for
/// [`whole_token_matvec_sequence_bare`]'s output-head dispatch: identical
/// lane/simdgroup assignment and per-block loop shape, `144ul` (`Q4_K`'s own
/// [`omega::msl::Q4K_BLOCK_BYTES`]) swapped for `210ul`
/// (`omega::msl::Q6K_BLOCK_BYTES`) and `q4k_pair_dot` swapped for
/// `q6k_pair_dot` (`omega::msl::Q6K_PAIR_DOT_MSL`, same `(block, iq, ir, yl,
/// yh)` signature per that constant's own doc) -- not restated body text,
/// only the block stride and the callee name differ.
const Q6K_L3_SHAPE_KERNEL_BODY: &str = r#"
kernel void q6k_matvec_l3_shape(
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

    ulong blk_step = (ulong)ib_step * 210ul;
    device const uchar* blk_ptr[4];
    for (uint q = 0u; q < 4u; ++q) {
        blk_ptr[q] = weight + (ulong)(group_first + q) * (ulong)blocks_per_row * 210ul
            + (ulong)ib_first * 210ul;
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
            sumf[q] += q6k_pair_dot(blk_ptr[q], iq, ir, yl, yh);
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

fn q6k_l3_shape_source() -> String {
    format!(
        "{METAL_PREAMBLE}{}\n{}\n{Q6K_L3_SHAPE_KERNEL_BODY}",
        omega::msl::Q6K_UNPACK_MSL,
        omega::msl::Q6K_PAIR_DOT_MSL
    )
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

/// One arm's own timed-repeat statistics (ROW 335). `median` is the arm's
/// reported headline number -- robust to a single cold-outlier repeat, the
/// exact failure ROW 296/302/331 all saw (repeat 1 at 4-13 GB/s, repeats 2-5
/// converged; ROW 331's Q6_K head sample list `[3.95, 43.17, 21.48, 65.45,
/// 56.60]` never converged at all in 5). `mean` travels alongside it for
/// continuity with every prior ladder row's own printed number; `min`/`max`/
/// `cov_pct` show the spread the median alone hides.
struct SampleStats {
    mean: f64,
    median: f64,
    min: f64,
    max: f64,
    cov_pct: f64,
}

fn sample_stats(samples: &[f64]) -> SampleStats {
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance =
        samples.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / samples.len() as f64;
    let stddev = variance.sqrt();
    let cov_pct = if mean.abs() > f64::MIN_POSITIVE { stddev / mean * 100.0 } else { 0.0 };

    let mut sorted = samples.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).expect("bandwidth/latency samples are never NaN"));
    let len = sorted.len();
    let median = if len % 2 == 1 {
        sorted[len / 2]
    } else {
        (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
    };

    SampleStats {
        mean,
        median,
        min: sorted[0],
        max: sorted[len - 1],
        cov_pct,
    }
}

/// Every arm's own timing discipline (ROW 335): one untimed warm-up dispatch
/// of the arm's own command buffer, immediately discarded, then [`REPEATS`]
/// timed repeats. ROW 296/302/331 all saw the same signature -- a cold first
/// dispatch reading 4-13 GB/s while later repeats converged -- so the warm-up
/// removes that outlier from the timed set itself, rather than leaving it in
/// and averaging (or medianing) around it. `dispatch_once` is whatever one
/// arm's own command-buffer-commit-and-wait closure is; every call site below
/// supplies its own.
fn warmed_up_samples<F: FnMut() -> Duration>(mut dispatch_once: F) -> Vec<Duration> {
    dispatch_once();
    (0..REPEATS).map(|_| dispatch_once()).collect()
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
    activation_name: &str,
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
            name: Some(activation_name.to_string()),
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

/// Checks one arm's parity against its reference batch and reports, never
/// panics -- so a failing arm does not stop every arm after it from
/// dispatching and reporting its own bandwidth (module doc, [`PARITY_MAX_REL_ERROR`]).
/// Returns `Some(failure message)` when the batch-peak-relative error exceeds
/// [`PARITY_MAX_REL_ERROR`]; callers collect every `Some` across all their
/// arms and fail the test once, at the end, listing all of them.
fn check_parity(label: &str, actual: &[f32], expected: &[f32]) -> Option<String> {
    assert_eq!(actual.len(), expected.len(), "{label}: degenerate gate, row counts differ");
    let mut max_abs_error = 0.0f32;
    let mut batch_peak = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(got.is_finite(), "{label}: produced a non-finite value: {got}");
        max_abs_error = max_abs_error.max((got - want).abs());
        batch_peak = batch_peak.max(want.abs());
    }
    let relative_error = max_abs_error / batch_peak.max(f32::EPSILON);
    println!(
        "{label}: max_abs_error={max_abs_error} batch_peak_abs={batch_peak} relative_error={relative_error}"
    );
    (relative_error > PARITY_MAX_REL_ERROR).then(|| {
        format!(
            "{label}: relative_error={relative_error} (max_abs_error={max_abs_error}, \
             batch_peak_abs={batch_peak}) exceeds PARITY_MAX_REL_ERROR={PARITY_MAX_REL_ERROR}"
        )
    })
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
    let mut parity_failures: Vec<String> = Vec::new();

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
        let elapsed_samples = warmed_up_samples(|| {
            time_batch_l0l1l2(
                &queue,
                &l0_pipeline,
                &no_copy_weight,
                &weight_offsets,
                &l0_output,
                ROWS * size_of::<u32>(),
                &blocks_per_row_uniform,
                default_grid_threads,
                default_threadgroup_width,
            )
        });
        let samples = gbps_samples(&elapsed_samples, TOTAL_TIMED_BYTES);
        let stats = sample_stats(&samples);
        println!(
            "arm=L0_streaming median_gbps={:.2} mean_gbps={:.2} min_gbps={:.2} max_gbps={:.2} \
             cov_pct={:.2} samples={samples:?} dispatches={WEIGHT_TENSOR_COUNT} bytes={TOTAL_TIMED_BYTES}",
            stats.median, stats.mean, stats.min, stats.max, stats.cov_pct
        );
        ladder_gbps.insert("L0_streaming", stats.median);
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
        let elapsed_samples = warmed_up_samples(|| {
            time_batch_l0l1l2(
                &queue,
                &l1_pipeline,
                &no_copy_weight,
                &weight_offsets,
                &l1_output,
                ROWS * size_of::<f32>(),
                &blocks_per_row_uniform,
                default_grid_threads,
                default_threadgroup_width,
            )
        });
        let samples = gbps_samples(&elapsed_samples, TOTAL_TIMED_BYTES);
        let stats = sample_stats(&samples);
        println!(
            "arm=L1_header_decode median_gbps={:.2} mean_gbps={:.2} min_gbps={:.2} max_gbps={:.2} \
             cov_pct={:.2} samples={samples:?} dispatches={WEIGHT_TENSOR_COUNT} bytes={TOTAL_TIMED_BYTES}",
            stats.median, stats.mean, stats.min, stats.max, stats.cov_pct
        );
        ladder_gbps.insert("L1_header_decode", stats.median);
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
        let elapsed_samples = warmed_up_samples(|| {
            time_batch_l0l1l2(
                &queue,
                &l2_pipeline,
                &no_copy_weight,
                &weight_offsets,
                &l2_output,
                ROWS * size_of::<f32>(),
                &blocks_per_row_uniform,
                default_grid_threads,
                default_threadgroup_width,
            )
        });
        let samples = gbps_samples(&elapsed_samples, TOTAL_TIMED_BYTES);
        let stats = sample_stats(&samples);
        println!(
            "arm=L2_dequant median_gbps={:.2} mean_gbps={:.2} min_gbps={:.2} max_gbps={:.2} \
             cov_pct={:.2} samples={samples:?} dispatches={WEIGHT_TENSOR_COUNT} bytes={TOTAL_TIMED_BYTES}",
            stats.median, stats.mean, stats.min, stats.max, stats.cov_pct
        );
        ladder_gbps.insert("L2_dequant", stats.median);
    }

    // ---- L3 baseline: the REAL production kernel, through the public
    // plan()/execute_plan() entry point -- byte-identical emitted MSL by
    // construction, all WEIGHT_TENSOR_COUNT tensors in ONE program so
    // execute_plan encodes all of them into ONE command buffer. ----
    let (program, sums) =
        multi_tensor_matmul_program(&weight_names, ROWS as u32, IN_DIM as u32, DType::UInt8, "activation");
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
        let elapsed_samples = warmed_up_samples(|| {
            let started = Instant::now();
            let evaluated = omega::metal::execute_plan(&plan, &blocks)
                .expect("execute_plan runs the real production q4_k matmul over every ffn tensor");
            let elapsed = started.elapsed();
            l3_baseline_layer0_up = evaluated
                .get(sums[0])
                .map(|(data, _)| data.to_vec())
                .expect("blk.0.ffn_up's own reduce node is a requested output");
            elapsed
        });
        let samples = gbps_samples(&elapsed_samples, TOTAL_TIMED_BYTES);
        let stats = sample_stats(&samples);
        println!(
            "arm=L3_baseline_production_execute_plan median_gbps={:.2} mean_gbps={:.2} \
             min_gbps={:.2} max_gbps={:.2} cov_pct={:.2} samples={samples:?} \
             dispatches={WEIGHT_TENSOR_COUNT} bytes={TOTAL_TIMED_BYTES} \
             (end-to-end plan+dispatch+readback, see module doc)",
            stats.median, stats.mean, stats.min, stats.max, stats.cov_pct
        );
        ladder_gbps.insert("L3_baseline", stats.median);
    }
    parity_failures.extend(check_parity(
        "L3_baseline vs cpu_reference",
        &l3_baseline_layer0_up[..PARITY_ROWS],
        &cpu_reference,
    ));

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
                let elapsed_samples = warmed_up_samples(|| match dispatch_mode {
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
                });
                let samples = gbps_samples(&elapsed_samples, TOTAL_TIMED_BYTES);
                let stats = sample_stats(&samples);
                println!(
                    "arm=L3_shape math_mode={:<8} simdgroups_per_tg={simdgroups_per_tg} \
                     dispatch={dispatch_name:<20} median_gbps={:.2} mean_gbps={:.2} \
                     min_gbps={:.2} max_gbps={:.2} cov_pct={:.2} samples={samples:?} \
                     dispatches={WEIGHT_TENSOR_COUNT} bytes={TOTAL_TIMED_BYTES}",
                    math_arm.name, stats.median, stats.mean, stats.min, stats.max, stats.cov_pct
                );

                let is_production_default = matches!(math_arm.name, "safe")
                    && simdgroups_per_tg == 1
                    && matches!(dispatch_mode, DispatchMode::Threads);
                if is_production_default {
                    let full = read_f32_buffer(&l3_output, WEIGHT_TENSOR_COUNT * ROWS);
                    default_shape_layer0_up = full[..ROWS].to_vec();
                    ladder_gbps.insert("L3_shape_default", stats.median);
                }
            }
        }
    }
    parity_failures.extend(check_parity(
        "L3_shape default arm vs cpu_reference",
        &default_shape_layer0_up[..PARITY_ROWS],
        &cpu_reference,
    ));
    parity_failures.extend(check_parity(
        "L3_shape default arm vs L3_baseline",
        &default_shape_layer0_up[..PARITY_ROWS],
        &l3_baseline_layer0_up[..PARITY_ROWS],
    ));

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
        let elapsed_samples = warmed_up_samples(|| {
            time_batch_empty(&queue, &empty_pipeline, &empty_output, WEIGHT_TENSOR_COUNT, 32, 32)
        });
        let ns_samples = ns_per_dispatch_samples(&elapsed_samples, WEIGHT_TENSOR_COUNT);
        let stats = sample_stats(&ns_samples);
        println!(
            "arm=empty median_ns_per_dispatch={:.1} mean_ns_per_dispatch={:.1} \
             min_ns_per_dispatch={:.1} max_ns_per_dispatch={:.1} cov_pct={:.2} samples={ns_samples:?} \
             dispatches={WEIGHT_TENSOR_COUNT}",
            stats.median, stats.mean, stats.min, stats.max, stats.cov_pct
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

    assert!(
        parity_failures.is_empty(),
        "parity failed for {} arm(s): {parity_failures:#?}",
        parity_failures.len()
    );
}

/// The three K-quant codecs a real decode step actually dispatches through
/// Metal on this checkpoint (`Q4_K` for every uniformly-quantized family,
/// `Q5_K` for the llama.cpp-inserted per-layer quality bump on 4 of 32
/// `attn_v`/`ffn_down` tensors, `Q6_K` for the lone `output.weight` head --
/// `proxima-tensor/docs/discipline.md` ROW 87's own printed inventory,
/// cross-checked live against `omega::msl::emit`'s `PackedCodec::Q5K` arm
/// (`omega/src/msl.rs:4516-4527`, `q5k_pair_dot` selected the same way
/// `q4k_pair_dot`/`q6k_pair_dot` are) -- Metal has a real production kernel
/// for all three today, superseding the stale "no q5_k/q6_k unpack kernel"
/// claim `FFN_TENSOR_KINDS`'s own doc above restates from an earlier row).
/// One enum dispatching each codec's own `proxima_gguf::quant` module and
/// `QuantizedBlock` variant, instead of three near-identical copies of every
/// synth/reference/arm function below (guiding-principles §1: the shape is
/// the same, only the codec's own block size and functions differ).
#[derive(Clone, Copy)]
enum ShapeCodec {
    Q4K,
    Q5K,
    Q6K,
}

impl ShapeCodec {
    const fn name(self) -> &'static str {
        match self {
            ShapeCodec::Q4K => "Q4_K",
            ShapeCodec::Q5K => "Q5_K",
            ShapeCodec::Q6K => "Q6_K",
        }
    }

    const fn qk_k(self) -> usize {
        match self {
            ShapeCodec::Q4K => q4_k::QK_K,
            ShapeCodec::Q5K => q5_k::QK_K,
            ShapeCodec::Q6K => q6_k::QK_K,
        }
    }

    const fn block_bytes(self) -> usize {
        match self {
            ShapeCodec::Q4K => q4_k::BLOCK_BYTES,
            ShapeCodec::Q5K => q5_k::BLOCK_BYTES,
            ShapeCodec::Q6K => q6_k::BLOCK_BYTES,
        }
    }

    fn row_bytes(self, k: usize) -> usize {
        (k / self.qk_k()) * self.block_bytes()
    }

    fn quantize(self, input: &[f32], output: &mut [u8]) -> Result<(), QuantError> {
        match self {
            ShapeCodec::Q4K => q4_k::quantize(input, output),
            ShapeCodec::Q5K => q5_k::quantize(input, output),
            ShapeCodec::Q6K => q6_k::quantize(input, output),
        }
    }

    fn dequantize(self, data: &[u8], output: &mut [f32]) -> Result<(), QuantError> {
        match self {
            ShapeCodec::Q4K => q4_k::dequantize(data, output),
            ShapeCodec::Q5K => q5_k::dequantize(data, output),
            ShapeCodec::Q6K => q6_k::dequantize(data, output),
        }
    }

    fn quantized_block(self, bytes: &[u8]) -> QuantizedBlock<'_> {
        match self {
            ShapeCodec::Q4K => QuantizedBlock::Q4K(bytes),
            ShapeCodec::Q5K => QuantizedBlock::Q5K(bytes),
            ShapeCodec::Q6K => QuantizedBlock::Q6K(bytes),
        }
    }
}

/// Synthesizes `tensor_count` distinct `codec`-encoded tensors, each
/// `[rows, k]`, through the SAME encoder production's own real weights of
/// that codec are quantized with (`proxima_gguf::quant::{q4_k,q5_k,q6_k}::quantize`)
/// -- guiding-principles §9's fallback for shapes this checkpoint carries in
/// a quantity (1 `output.weight`, or a `Q5_K`-mixed 4/32 per family) below
/// this harness's own amortization floor. Each tensor gets its own [`Lcg`]
/// seed (`seed_base + tensor_index`) so no two tensors are bit-identical and
/// a GPU cache cannot serve one tensor's read from another's -- the same
/// "distinct tensors" posture the real `Q4_K` ffn_up/ffn_gate arms above get
/// for free from 32 real, distinct checkpoint tensors. One row's worth of
/// `f32` (`k` elements) is reused per row rather than materializing the
/// whole tensor in `f32` first, so peak host memory stays at the quantized
/// output size, not `4x` that.
fn synth_weight_bytes(codec: ShapeCodec, seed_base: u64, tensor_count: usize, rows: usize, k: usize) -> Vec<u8> {
    let row_bytes = codec.row_bytes(k);
    let tensor_bytes = rows * row_bytes;
    let mut bytes = vec![0u8; tensor_bytes * tensor_count];
    let mut row_f32 = vec![0.0f32; k];
    for (tensor_index, tensor_slice) in bytes.chunks_exact_mut(tensor_bytes).enumerate() {
        let mut lcg = Lcg(seed_base + tensor_index as u64);
        for row_blocks in tensor_slice.chunks_exact_mut(row_bytes) {
            for value in row_f32.iter_mut() {
                *value = lcg.next_unit() * 4.0 - 2.0;
            }
            codec
                .quantize(&row_f32, row_blocks)
                .expect("row length is a whole multiple of the codec's own QK_K");
        }
    }
    bytes
}

/// CPU oracle for the first [`PARITY_ROWS`] rows of the FIRST synthesized
/// tensor -- same posture as [`cpu_reference_first_rows`], `codec`'s own
/// `dequantize`/dot-product instead of a fixed `Q4_K`/`Q6_K` call.
fn codec_cpu_reference_first_rows(
    codec: ShapeCodec,
    first_tensor_bytes: &[u8],
    k: usize,
    activation: &[f32],
) -> Vec<f32> {
    let row_bytes = codec.row_bytes(k);
    let mut reference = Vec::with_capacity(PARITY_ROWS);
    let mut dequantized_row = vec![0.0f32; k];
    for row_blocks in first_tensor_bytes[..PARITY_ROWS * row_bytes].chunks_exact(row_bytes) {
        codec
            .dequantize(row_blocks, &mut dequantized_row)
            .expect("a whole number of the codec's own super-blocks per row");
        let dot: f32 = dequantized_row.iter().zip(activation.iter()).map(|(weight, act)| weight * act).sum();
        reference.push(dot);
    }
    reference
}

/// Runs one isolated decode-matvec shape arm: synthesizes `tensor_count`
/// distinct `[rows, k]` tensors of `codec`, dispatches PRODUCTION's own
/// emitted kernel for that codec over all of them through
/// `omega::metal::plan`/`execute_plan` -- the SAME public entry point
/// `L3_baseline` uses for `Q4_K` above, so the MSL text is whatever
/// `omega::msl::emit` renders today (byte-identical to production by
/// construction) and the dispatch geometry is production's, never
/// hand-assembled here -- and reports rows/k/codec/tensors/bytes/GB/s/CoV
/// plus the ratio to the 381.24 GB/s device ceiling (ROW 296), so an
/// in-isolation vs in-program gap reads directly off this arm's own printed
/// line next to [`ROW_310_IN_PROGRAM_GBPS`]'s number for the same family.
/// Reports its parity failure, if any, as a return value rather than
/// panicking, so the caller can run every shape arm and fail once at the
/// end listing all of them ([`check_parity`]'s own doc). Generalizes the
/// prior `run_q6k_shape_arm` (one codec parameter instead of a second copy
/// per codec, guiding-principles §1).
/// `weight_keepalive` (ROW 336) is the caller's own accumulator that this
/// call's synthesized weight bytes are moved into rather than dropped --
/// see the `resident_names` comment below for why an arm's weight buffer
/// must never be freed while the process (and therefore `NOCOPY_BUFFERS`,
/// `omega::metal`'s own address-keyed cache) is still alive.
#[must_use]
fn run_shape_arm(
    label: &str,
    codec: ShapeCodec,
    rows: usize,
    k: usize,
    tensor_count: usize,
    seed_base: u64,
    weight_keepalive: &mut Vec<Vec<u8>>,
) -> Option<String> {
    let row_bytes = codec.row_bytes(k);
    let tensor_bytes = (rows * row_bytes) as u64;
    let total_timed_bytes = tensor_bytes * tensor_count as u64;
    let codec_name = codec.name();
    println!(
        "=== decode shape arm {label}: rows={rows} k={k} codec={codec_name} tensors={tensor_count} \
         tensor_bytes={tensor_bytes} total_timed_bytes={total_timed_bytes} \
         (production kernel via omega::metal::plan/execute_plan) ==="
    );

    let weight_bytes = synth_weight_bytes(codec, seed_base, tensor_count, rows, k);
    let weight_names: Vec<String> = (0..tensor_count).map(|index| format!("{label}_{index}")).collect();

    let mut activation_lcg = Lcg(seed_base + 999);
    let activation: Vec<f32> = (0..k).map(|_| activation_lcg.next_unit() * 4.0 - 2.0).collect();

    let cpu_reference =
        codec_cpu_reference_first_rows(codec, &weight_bytes[..rows * row_bytes], k, &activation);

    // ROW 332: `RESIDENT_BUFFERS` (`omega/src/metal.rs:4604-4622`) is a
    // thread-local cache keyed by NAME plus a byte-length check, never by
    // content -- sound only for the caller's OWN static weights
    // (`Plan::mark_resident`'s own doc). `decode_shape_roofline_ladder` runs
    // every [`DecodeShape`] arm in ONE test thread; a shared literal
    // "activation" name across arms with DIFFERENT activation content but
    // the SAME byte length (every k=4096 arm) silently served an earlier
    // arm's stale activation bytes to a later arm's kernel. Naming the
    // activation node per-`label` makes each arm's own resident entry
    // distinct, so no arm can ever be served another arm's activation.
    let activation_name = format!("{label}_activation");
    let (program, sums) =
        multi_tensor_matmul_program(&weight_names, rows as u32, k as u32, DType::UInt8, &activation_name);
    let weight_slices: Vec<&[u8]> = weight_bytes.chunks_exact(rows * row_bytes).collect();
    let mut blocks: Vec<QuantizedBlock<'_>> =
        weight_slices.iter().map(|slice| codec.quantized_block(slice)).collect();
    blocks.push(QuantizedBlock::Float32(&activation));

    let mut plan = omega::metal::plan(&program, &[], &blocks, &sums)
        .expect("plan resolves the synthetic matmul over every tensor");
    // ROW 334 found that marking these weight nodes resident bought no real
    // saving and was UNSOUND: this shape's own byte size (a multiple of the
    // host page size for every arm in `DECODE_SHAPES`) sends every weight
    // chunk through the page-aligned no-copy path either way
    // (`omega::metal::upload_packed_bytes`), which keys its cache
    // (`NOCOPY_BUFFERS`) on `(pointer, byte_length)` alone, NEVER on the name
    // `mark_resident` proved unique -- so when one arm's `Vec` was freed and
    // the NEXT arm's same-byte-size `Vec` landed at the identical address,
    // that cache served the PRIOR arm's stale weight bytes under the new
    // arm's own fresh, uniquely-named plan. ROW 334's fix left weight nodes
    // unmarked, which is sound but re-pays `upload_block_no_copy_uncached`'s
    // own device-buffer-wrapper cost on EVERY timed repeat, not just once --
    // ROW 335's own defect (every shape arm reading 20-25 GB/s in a quiet
    // session where the L3 rung reads 186-247 with the identical kernel).
    //
    // ROW 336 fix: mark every weight node resident TOO (each already carries
    // a name unique per arm AND tensor, `{label}_{index}`), and -- the part
    // that actually restores ROW 334's soundness argument -- never free this
    // arm's `weight_bytes` while the process is alive: `weight_keepalive`
    // (this function's own last statement) moves it into an accumulator the
    // caller holds for the whole test run. `NOCOPY_BUFFERS`'s hazard was
    // never residency itself, only a FREED address being handed to a LATER,
    // different allocation; a `Vec` this function never frees can never
    // donate its address to a later arm, so the exact collision ROW 334
    // reproduced (`nocopy_reuses` equal to `nocopy_uploads`, every touch
    // served from a DIFFERENT arm's addresses) is now impossible by
    // construction, not by avoidance. The activation node stays resident
    // (label-unique since ROW 332) for the same reason it always was.
    let mut resident_names: BTreeSet<&str> =
        weight_names.iter().map(String::as_str).collect();
    resident_names.insert(activation_name.as_str());
    plan.mark_resident(&resident_names);

    // ROW 336: reset every stage counter right before this arm's own timed
    // repeats, then read `metal_stage_totals()` again inside the closure
    // itself, once per call (`warmed_up_samples`'s own doc: index 0 is the
    // untimed warm-up, indices 1.. are the timed repeats) -- a per-repeat
    // delta, not one aggregate over the whole arm, is the only way to prove
    // a TIMED repeat never re-uploads: an aggregate could still hide one
    // stray upload inside seven repeats' worth of otherwise-free reuses.
    #[cfg(feature = "instrument")]
    let _ = omega::metal::metal_stage_totals();
    #[cfg(feature = "instrument")]
    let mut per_call_totals: Vec<omega::metal::MetalStageTotals> = Vec::new();

    let mut first_tensor_output: Vec<f32> = Vec::new();
    let elapsed_samples = warmed_up_samples(|| {
        let started = Instant::now();
        let evaluated = omega::metal::execute_plan(&plan, &blocks)
            .expect("execute_plan runs the synthetic matmul over every tensor");
        let elapsed = started.elapsed();
        first_tensor_output = evaluated
            .get(sums[0])
            .map(|(data, _)| data.to_vec())
            .expect("first tensor's own reduce node is a requested output");
        #[cfg(feature = "instrument")]
        per_call_totals.push(omega::metal::metal_stage_totals());
        elapsed
    });
    // `resident_uploads`/`resident_reuses` witness the NAME-keyed
    // `RESIDENT_BUFFERS` copy path (ROW 332's own fix); `nocopy_uploads`/
    // `nocopy_reuses` witness the ADDRESS-keyed `NOCOPY_BUFFERS` cache this
    // ROW 336 fix now routes every weight node through.
    //
    // `nocopy_uploads` is NOT "how many fresh buffers were created" -- it
    // fires once per call routed through the no-copy path REGARDLESS of hit
    // or miss (`upload_block_as_float`/`upload_packed_bytes`'s own counter
    // site sits before the hit/miss branch); `nocopy_reuses` is the ONLY
    // counter that fires exclusively on a cache HIT. So the per-repeat
    // no-copy invariant is `nocopy_uploads == nocopy_reuses` (every no-copy
    // touch this repeat was a reuse of the warm-up's own wrapper, zero
    // fresh creations), never `nocopy_uploads == 0` -- confirmed the hard
    // way: the naive `== 0` form false-failed on `attn_q`'s very first timed
    // repeat, which this comment exists so nobody repeats. `resident_uploads`
    // has the opposite shape (`upload_resident_copy`'s own counter site sits
    // INSIDE the miss branch only), so `== 0` is exactly right there.
    #[cfg(feature = "instrument")]
    {
        for (repeat_index, totals) in per_call_totals.iter().enumerate().skip(1) {
            assert_eq!(
                totals.nocopy_uploads, totals.nocopy_reuses,
                "arm={label} timed repeat {repeat_index}: {} of {} no-copy weight touches were \
                 a FRESH upload, not a cache reuse of the warm-up's own wrapper",
                totals.nocopy_uploads - totals.nocopy_reuses,
                totals.nocopy_uploads
            );
            assert_eq!(
                totals.resident_uploads, 0,
                "arm={label} timed repeat {repeat_index} re-uploaded {} buffer(s) via the \
                 resident-copy path instead of reusing the warm-up's own upload",
                totals.resident_uploads
            );
        }
        let total_nocopy_uploads: u64 = per_call_totals.iter().map(|totals| totals.nocopy_uploads).sum();
        let total_nocopy_reuses: u64 = per_call_totals.iter().map(|totals| totals.nocopy_reuses).sum();
        let total_resident_uploads: u64 =
            per_call_totals.iter().map(|totals| totals.resident_uploads).sum();
        let total_resident_reuses: u64 =
            per_call_totals.iter().map(|totals| totals.resident_reuses).sum();
        let total_copying_uploads: u64 =
            per_call_totals.iter().map(|totals| totals.copying_uploads).sum();
        println!(
            "arm={label} stage_totals: nocopy_uploads={total_nocopy_uploads} \
             nocopy_reuses={total_nocopy_reuses} resident_uploads={total_resident_uploads} \
             resident_reuses={total_resident_reuses} copying_uploads={total_copying_uploads} \
             per_call_uploads={:?}",
            per_call_totals
                .iter()
                .map(|totals| (totals.nocopy_uploads, totals.resident_uploads))
                .collect::<Vec<_>>()
        );
    }
    let samples = gbps_samples(&elapsed_samples, total_timed_bytes);
    let stats = sample_stats(&samples);
    let ratio_to_ceiling = stats.median / DEVICE_CEILING_GBPS;
    println!(
        "arm={label} rows={rows} k={k} codec={codec_name} tensors={tensor_count} bytes={total_timed_bytes} \
         median_gbps={:.2} mean_gbps={:.2} min_gbps={:.2} max_gbps={:.2} cov_pct={:.2} samples={samples:?} \
         ratio_to_device_ceiling={ratio_to_ceiling:.3}",
        stats.median, stats.mean, stats.min, stats.max, stats.cov_pct
    );
    if let Some((_, row_310_gbps)) =
        ROW_310_IN_PROGRAM_GBPS.iter().find(|(family, _)| *family == label)
    {
        match row_310_gbps {
            Some(in_program) => println!(
                "summary family={label}: isolated={:.2} GB/s vs ROW 310 in-program={in_program:.2} \
                 GB/s (ratio isolated/in_program={:.3})",
                stats.median,
                stats.median / in_program
            ),
            None => println!(
                "summary family={label}: isolated={:.2} GB/s vs ROW 310 in-program=noise-negative \
                 (not comparable, see ROW 310's own CoV note)",
                stats.median
            ),
        }
    }
    if let Ok(raw_repeat_count) = std::env::var("PROXIMA_LADDER_REPEAT") {
        let repeat_count: usize = raw_repeat_count
            .parse()
            .expect("PROXIMA_LADDER_REPEAT must be a positive integer");
        let failures =
            repeat_dump_divergences(label, &plan, &blocks, 0, sums[0], &cpu_reference, repeat_count);
        println!(
            "PROXIMA_LADDER_REPEAT summary: label={label} repeats={repeat_count} failures={failures}"
        );
    }

    // ROW 336: moved, never dropped -- see this function's own doc and the
    // `resident_names` comment above for why `weight_bytes`' address must
    // outlive this arm.
    weight_keepalive.push(weight_bytes);

    check_parity(
        &format!("{label} ({codec_name}) vs cpu_reference"),
        &first_tensor_output[..PARITY_ROWS],
        &cpu_reference,
    )
}

/// One repeat's divergent row ([`PROXIMA_LADDER_REPEAT`], `ladder-parity-
/// flake-read.md` candidate 2) -- which row, and by how much, so the diff of
/// divergent rows across repeats names the mechanism (a whole-row garbage
/// read vs a single SIMD lane's partial contribution) instead of leaving it
/// to [`check_parity`]'s own aggregate `relative_error`.
struct DivergentRow {
    row: usize,
    reference: f32,
    observed: f32,
    relative_error: f32,
}

/// `PROXIMA_LADDER_REPEAT=<n>` re-dispatches the SAME resolved `plan` `n`
/// extra times beyond the bandwidth timing loop above and checks EVERY
/// repeat's per-row parity individually against `cpu_reference` (rather than
/// [`check_parity`]'s single aggregate `relative_error` over the last repeat
/// only), printing the repeat index, `tensor_index`, and the first 8
/// divergent rows (row, reference, observed, relative_error) for any repeat
/// that diverges. No-op when the env var is unset (callers only reach this
/// when it parsed), so normal bandwidth runs never pay for it.
fn repeat_dump_divergences(
    label: &str,
    plan: &omega::metal::Plan,
    blocks: &[QuantizedBlock<'_>],
    tensor_index: usize,
    sum_node: NodeId,
    cpu_reference: &[f32],
    repeat_count: usize,
) -> usize {
    let batch_peak = cpu_reference.iter().fold(0.0f32, |peak, value| peak.max(value.abs()));
    let mut failing_repeats = 0usize;
    for repeat in 0..repeat_count {
        let evaluated = omega::metal::execute_plan(plan, blocks)
            .expect("execute_plan runs the synthetic matmul over every tensor");
        let observed = evaluated
            .get(sum_node)
            .map(|(data, _)| data.to_vec())
            .expect("tensor's own reduce node is a requested output");
        let divergent: Vec<DivergentRow> = cpu_reference
            .iter()
            .zip(observed.iter())
            .enumerate()
            .filter_map(|(row, (&reference, &observed))| {
                let relative_error = (observed - reference).abs() / batch_peak.max(f32::EPSILON);
                (relative_error > PARITY_MAX_REL_ERROR)
                    .then_some(DivergentRow { row, reference, observed, relative_error })
            })
            .collect();
        if divergent.is_empty() {
            continue;
        }
        failing_repeats += 1;
        println!(
            "PROXIMA_LADDER_REPEAT flake: label={label} repeat={repeat} tensor_index={tensor_index} \
             divergent_rows={} (showing first 8)",
            divergent.len()
        );
        for entry in divergent.iter().take(8) {
            println!(
                "  row={} reference={} observed={} relative_error={}",
                entry.row, entry.reference, entry.observed, entry.relative_error
            );
        }
    }
    failing_repeats
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
    let mut weight_keepalive: Vec<Vec<u8>> = Vec::new();
    let parity_failures: Vec<String> = [
        run_shape_arm(
            "head",
            ShapeCodec::Q6K,
            Q6K_HEAD_ROWS,
            IN_DIM,
            Q6K_HEAD_TENSOR_COUNT,
            4096,
            &mut weight_keepalive,
        ),
        run_shape_arm(
            "layer",
            ShapeCodec::Q6K,
            Q6K_LAYER_ROWS,
            IN_DIM,
            Q6K_LAYER_TENSOR_COUNT,
            8192,
            &mut weight_keepalive,
        ),
    ]
    .into_iter()
    .flatten()
    .collect();

    assert!(
        parity_failures.is_empty(),
        "parity failed for {} arm(s): {parity_failures:#?}",
        parity_failures.len()
    );
}

// ---- decode-shape ladder: one arm per real matvec shape a decode step
// actually dispatches on this checkpoint (openchat-3.5-1210.Q4_K_S.gguf,
// Mistral-7B GQA architecture: n_heads=32, n_kv_heads=8, head_dim=128,
// hidden=4096, ffn_dim=14336), isolated from ROW 310's concurrent-dispatch
// overlap and, for `attn_v`/`ffn_down`, from the shared-buffer measurement
// this checkpoint's own mixed codec (`Q4_K` x28 + `Q5_K` x4 of 32 layers,
// `proxima-tensor/docs/discipline.md` ROW 87's own printed inventory) would
// otherwise average into one number. ROW 296/322 measured only the `Q6_K`
// head shape in isolation before this; every other family below is new. ----

/// One (family, codec, rows, k) shape this checkpoint's decode step
/// dispatches for real, per `proxima-model-interop`'s own weight binding
/// (`bind.rs`) and this checkpoint's GQA shape (`k`/`v` project to
/// `n_kv_heads * head_dim = 8 * 128 = 1024`, not the full 4096 `q`/`o` use).
/// `attn_v`/`ffn_down` each appear twice -- once per real codec this
/// checkpoint mixes for that family -- so neither codec's isolated
/// bandwidth is silently averaged away.
struct DecodeShape {
    family: &'static str,
    codec: ShapeCodec,
    rows: usize,
    k: usize,
    seed: u64,
}

const DECODE_SHAPES: &[DecodeShape] = &[
    DecodeShape { family: "attn_q", codec: ShapeCodec::Q4K, rows: 4096, k: 4096, seed: 10_000 },
    DecodeShape { family: "attn_k", codec: ShapeCodec::Q4K, rows: 1024, k: 4096, seed: 20_000 },
    DecodeShape { family: "attn_v", codec: ShapeCodec::Q4K, rows: 1024, k: 4096, seed: 30_000 },
    DecodeShape { family: "attn_v_q5k", codec: ShapeCodec::Q5K, rows: 1024, k: 4096, seed: 30_500 },
    DecodeShape { family: "attn_output", codec: ShapeCodec::Q4K, rows: 4096, k: 4096, seed: 40_000 },
    DecodeShape { family: "ffn_gate", codec: ShapeCodec::Q4K, rows: 14_336, k: 4096, seed: 50_000 },
    DecodeShape { family: "ffn_up", codec: ShapeCodec::Q4K, rows: 14_336, k: 4096, seed: 60_000 },
    DecodeShape { family: "ffn_down", codec: ShapeCodec::Q4K, rows: 4096, k: 14_336, seed: 70_000 },
    DecodeShape { family: "ffn_down_q5k", codec: ShapeCodec::Q5K, rows: 4096, k: 14_336, seed: 70_500 },
    DecodeShape { family: "head", codec: ShapeCodec::Q6K, rows: 32_000, k: 4096, seed: 80_000 },
];

/// Same amortization-floor reasoning as [`Q6K_HEAD_TENSOR_COUNT`]: the
/// smallest tensor count that still clears [`MIN_TIMED_BYTES`] (2 GB
/// decimal) for this shape's own byte size, per shape rather than one fixed
/// count for every family (a `ffn_gate` tensor is ~117 MB, an `attn_k`
/// tensor is ~9 MB -- a shared count would either starve the small shapes
/// below the floor or move tens of GB for the large ones).
fn tensor_count_for_shape(shape: &DecodeShape) -> usize {
    // ROW 334's own repro tool -- overrides every shape's real per-family
    // tensor count with a single small value so the resident-cache
    // collision this row root-causes reproduces in seconds instead of the
    // ~25 minutes/arm a real tensor count (61-848) costs. Test-edge only:
    // no production path reads this, and `decode_shape_roofline_ladder`
    // itself (the real-scale gate) never sets it.
    if let Ok(raw) = std::env::var("PROXIMA_LADDER_TENSORS") {
        return raw.parse().expect("PROXIMA_LADDER_TENSORS must be a positive integer");
    }
    let tensor_bytes = (shape.rows * shape.codec.row_bytes(shape.k)) as u64;
    MIN_TIMED_BYTES.div_ceil(tensor_bytes) as usize
}

/// Isolated, per-shape bandwidth for every real decode matvec this
/// checkpoint dispatches -- the KERNEL's own speed per shape, quiet and
/// isolated, as a follow-up to ROW 310's in-buffer ablation (confounded by
/// concurrent-dispatch overlap) and ROW 296/322 (measured only the `Q6_K`
/// head shape in isolation). Each arm calls the SAME [`run_shape_arm`] the
/// `Q6_K`-only ladder above now also calls -- one function generalized over
/// codec and shape, not one per family.
///
/// `#[ignore]`d: synthesizes real quantized bytes through the real encoder
/// for every one of [`DECODE_SHAPES`] (CPU-bound minutes of work) and needs
/// a real Metal device, same posture as this file's other synthetic-data
/// arms above.
#[test]
#[ignore = "synthesizes real quantized bytes per shape through the real encoder and needs a real metal device"]
fn decode_shape_roofline_ladder() {
    println!(
        "=== decode-shape roofline ladder: {} real per-family matvec shapes, isolated, quiet \
         (device ceiling {DEVICE_CEILING_GBPS:.2} GB/s, ROW 296) ===",
        DECODE_SHAPES.len()
    );

    let mut weight_keepalive: Vec<Vec<u8>> = Vec::new();
    let parity_failures: Vec<String> = DECODE_SHAPES
        .iter()
        .filter_map(|shape| {
            let tensor_count = tensor_count_for_shape(shape);
            run_shape_arm(
                shape.family,
                shape.codec,
                shape.rows,
                shape.k,
                tensor_count,
                shape.seed,
                &mut weight_keepalive,
            )
        })
        .collect();

    assert!(
        parity_failures.is_empty(),
        "parity failed for {} arm(s): {parity_failures:#?}",
        parity_failures.len()
    );
}

// ---- ROW 338: per-decode-shape simdgroups/math sweep, production's own
// dispatch mode -- [`run_shape_arm`] cannot answer this (it goes through
// `omega::metal::plan`/`execute_plan`, whose dispatch geometry is a
// compile-time Cargo feature, this file's own module doc). This sweep reuses
// the L3 shape-sweep's mechanism (`q4k_matvec_l3_shape`, `l3_shape_source`,
// `compile_pipeline`, `time_batch_l3_shape_threads`) -- the SAME `q4k_pair_dot`
// body production emits, hand-dispatched so `threads_per_threadgroup` and
// `MTLMathMode` vary at runtime -- generalized over (rows, k) the way ROW
// 336's `run_shape_arm` generalized the production path over shape. Dispatch
// mode is held at `dispatchThreads` throughout (production's own dispatch
// mode, ROW 335's checked default), so only the two axes the brief asks
// about move. ----

/// One representative decode shape for this sweep -- `attn_k`/`attn_q` share
/// this checkpoint's GQA `k`=4096 attention width, `ffn_up`/`ffn_down` are
/// each other's transpose (`ffn_up` projects 4096->14336, `ffn_down`
/// 14336->4096), all four real `Q4_K` shapes this checkpoint's decode step
/// dispatches per [`DECODE_SHAPES`].
struct ShapeArmSweepSpec {
    family: &'static str,
    rows: usize,
    k: usize,
    seed: u64,
}

const SHAPE_ARM_SWEEP_SPECS: &[ShapeArmSweepSpec] = &[
    ShapeArmSweepSpec { family: "attn_k", rows: 1024, k: 4096, seed: 210_000 },
    ShapeArmSweepSpec { family: "attn_q", rows: 4096, k: 4096, seed: 220_000 },
    ShapeArmSweepSpec { family: "ffn_up", rows: 14_336, k: 4096, seed: 230_000 },
    ShapeArmSweepSpec { family: "ffn_down", rows: 4096, k: 14_336, seed: 240_000 },
];

/// The two axes this sweep varies: `simdgroups_per_tg` (`threads_per_tg / 32`,
/// `q4k_matvec_l3_shape`'s own runtime parameter) and `MTLMathMode`.
/// Production's own default cell is `simdgroups_per_tg=2` (ROW 311),
/// `MTLMathMode::Relaxed` (ROW 297) -- both present here so the table can mark
/// it. `Safe` is excluded: the brief's question is what NSG/math buy over
/// production's own variant, not a third math mode nobody ships.
const SWEEP_SIMDGROUPS_PER_THREADGROUP: [usize; 3] = [2, 4, 8];
const SWEEP_MATH_MODE_ARMS: [MathModeArm; 2] = [
    MathModeArm {
        name: "relaxed",
        mode: MTLMathMode::Relaxed,
    },
    MathModeArm {
        name: "fast",
        mode: MTLMathMode::Fast,
    },
];

/// Runs every (`simdgroups_per_tg`, math mode) cell for one shape, printing
/// one line per cell (`gbps_samples`/`sample_stats`, same discipline every
/// other arm in this file uses) and checking that cell's own parity against a
/// single CPU reference computed once per shape (the per-element compute path
/// is textually identical across cells -- only dispatch geometry and rounding
/// mode move, so one reference suffices; [`PARITY_MAX_REL_ERROR`]'s own doc
/// covers why `fast` is allowed to, but not required to, sit further from
/// that reference than `relaxed`). Weight and activation bytes are copied
/// into fresh `StorageModeShared` buffers once per shape, before any cell
/// runs (`shared_buffer_from_bytes` copies at creation and is never touched
/// again), so every timed repeat of every cell dispatches over buffers
/// already resident on the GPU -- zero uploads on a timed repeat, the same
/// invariant [`run_shape_arm`]'s own `nocopy_uploads == nocopy_reuses`
/// check exists to prove for the production path, satisfied here by
/// construction instead: nothing this function calls after buffer creation
/// can trigger a fresh upload.
/// [`synth_weight_bytes`]'s own body, parallelized across tensors with
/// `rayon::par_chunks_mut` (workspace performance philosophy: rayon where
/// applicable). Each tensor's real `q4_k::quantize` call is an independent,
/// CPU-bound RD search with no shared state -- `synth_weight_bytes`'s own
/// doc notes each tensor already gets its own `Lcg` seeded from `seed_base +
/// tensor_index`, so splitting the outer loop across threads changes nothing
/// about the bytes produced, only the wall time. This sweep needs
/// `MIN_TIMED_BYTES`'-scale tensor counts per shape (ROW 336's own real
/// per-family counts, restated here) for four shapes in one test run; the
/// serial version measured ~40 minutes for a single 848-tensor shape on this
/// host, which the sweep's own 40-minute budget cannot absorb four times
/// over. Kept local to this sweep rather than folded into
/// [`synth_weight_bytes`] itself so every OTHER caller's timing (and its own
/// single-threaded posture) is unchanged.
fn synth_weight_bytes_parallel(
    codec: ShapeCodec,
    seed_base: u64,
    tensor_count: usize,
    k: usize,
    tensor_bytes: usize,
) -> Vec<u8> {
    let row_bytes = codec.row_bytes(k);
    let mut bytes = vec![0u8; tensor_bytes * tensor_count];
    bytes.par_chunks_mut(tensor_bytes).enumerate().for_each(|(tensor_index, tensor_slice)| {
        let mut lcg = Lcg(seed_base + tensor_index as u64);
        let mut row_f32 = vec![0.0f32; k];
        for row_blocks in tensor_slice.chunks_exact_mut(row_bytes) {
            for value in row_f32.iter_mut() {
                *value = lcg.next_unit() * 4.0 - 2.0;
            }
            codec
                .quantize(&row_f32, row_blocks)
                .expect("row length is a whole multiple of the codec's own QK_K");
        }
    });
    bytes
}

fn run_shape_arm_sweep(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    spec: &ShapeArmSweepSpec,
) -> Vec<String> {
    let row_bytes = ShapeCodec::Q4K.row_bytes(spec.k);
    let tensor_bytes = (spec.rows * row_bytes) as u64;
    let tensor_count = MIN_TIMED_BYTES.div_ceil(tensor_bytes) as usize;
    let total_timed_bytes = tensor_bytes * tensor_count as u64;
    println!(
        "=== shape-arm sweep {}: rows={} k={} tensors={tensor_count} total_timed_bytes={total_timed_bytes} \
         (q4k_matvec_l3_shape, dispatchThreads, production's own read path) ===",
        spec.family, spec.rows, spec.k
    );

    let weight_bytes =
        synth_weight_bytes_parallel(ShapeCodec::Q4K, spec.seed, tensor_count, spec.k, tensor_bytes as usize);
    let weight_offsets: Vec<usize> = (0..tensor_count).map(|index| index * tensor_bytes as usize).collect();
    let weight_buffer = shared_buffer_from_bytes(device, &weight_bytes);

    let mut activation_lcg = Lcg(spec.seed + 999);
    let activation: Vec<f32> = (0..spec.k).map(|_| activation_lcg.next_unit() * 4.0 - 2.0).collect();
    // SAFETY: `activation` is a live `Vec<f32>` for the duration of this
    // call; the byte view is read-only and never outlives `activation`.
    let activation_bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(activation.as_ptr().cast::<u8>(), std::mem::size_of_val(activation.as_slice()))
    };
    let activation_buffer = shared_buffer_from_bytes(device, activation_bytes);
    let blocks_per_row_uniform = uniform_u64(device, (spec.k / Q4K_BLOCK_ELEMENTS) as u64);

    let cpu_reference =
        codec_cpu_reference_first_rows(ShapeCodec::Q4K, &weight_bytes[..spec.rows * row_bytes], spec.k, &activation);

    let output = device
        .newBufferWithLength_options(tensor_count * spec.rows * size_of::<f32>(), MTLResourceOptions::StorageModeShared)
        .expect("device allocates the sweep's output buffer");

    let total_simdgroups = spec.rows / PACKED_ROWS_PER_GROUP;
    let l3_shape_source_text = l3_shape_source();
    let mut parity_failures = Vec::new();

    for math_arm in SWEEP_MATH_MODE_ARMS {
        let pipeline = compile_pipeline(device, &l3_shape_source_text, "q4k_matvec_l3_shape", math_arm.mode);
        for simdgroups_per_tg in SWEEP_SIMDGROUPS_PER_THREADGROUP {
            let threadgroup_width = simdgroups_per_tg * 32;
            let threadgroup_count = total_simdgroups / simdgroups_per_tg;
            assert_eq!(
                threadgroup_count * simdgroups_per_tg,
                total_simdgroups,
                "{}: rows must divide evenly by simdgroups_per_tg for this sweep",
                spec.family
            );
            let grid_threads = threadgroup_count * threadgroup_width;

            let elapsed_samples = warmed_up_samples(|| {
                time_batch_l3_shape_threads(
                    queue,
                    &pipeline,
                    &weight_buffer,
                    &weight_offsets,
                    &activation_buffer,
                    &output,
                    spec.rows * size_of::<f32>(),
                    &blocks_per_row_uniform,
                    grid_threads,
                    threadgroup_width,
                )
            });
            let samples = gbps_samples(&elapsed_samples, total_timed_bytes);
            let stats = sample_stats(&samples);
            let is_production_default = simdgroups_per_tg == 2 && math_arm.name == "relaxed";
            let first_tensor_rows = read_f32_buffer(&output, PARITY_ROWS);
            let label = format!("{} nsg={simdgroups_per_tg} math={}", spec.family, math_arm.name);
            let parity_failure = check_parity(&label, &first_tensor_rows, &cpu_reference);
            println!(
                "arm=shape_sweep family={} nsg={simdgroups_per_tg} math={:<8} production_default={is_production_default} \
                 median_gbps={:.2} mean_gbps={:.2} min_gbps={:.2} max_gbps={:.2} cov_pct={:.2} samples={samples:?} \
                 parity_ok={}",
                spec.family, math_arm.name, stats.median, stats.mean, stats.min, stats.max, stats.cov_pct,
                parity_failure.is_none()
            );
            parity_failures.extend(parity_failure);
        }
    }
    parity_failures
}

/// ROW 338: what do simdgroups-per-threadgroup {2, 4, 8} x math mode
/// {relaxed, fast} buy, per decode shape, on PRODUCTION's own op variant
/// (`q4k_pair_dot`, `dispatchThreads`)? Follows ROW 335's single-shape (ffn)
/// L3 sweep and ROW 336's per-shape isolation with the axes ROW 335 already
/// has, generalized across every representative shape instead of one.
///
/// `PROXIMA_SWEEP_SHAPE` (read here, at the test edge only) restricts the run
/// to one [`SHAPE_ARM_SWEEP_SPECS`] family by name -- ROW 337 found the full
/// 4-shape x 6-cell run does not finish inside a 25-minute quiet-box window,
/// so a per-shape landing needs to name the shape it is actually timing.
///
/// `#[ignore]`d: synthesizes real quantized bytes per shape through the real
/// encoder (CPU-bound minutes of work) and needs a real Metal device, same
/// posture as this file's other synthetic-data arms.
#[test]
#[ignore = "synthesizes real quantized bytes per shape through the real encoder and needs a real metal device"]
fn decode_shape_nsg_math_sweep() {
    let device = MTLCreateSystemDefaultDevice().expect("a Metal device is available on this host");
    let queue = device.newCommandQueue().expect("device creates a command queue");

    let requested_shape = std::env::var("PROXIMA_SWEEP_SHAPE").ok();
    let specs: Vec<&ShapeArmSweepSpec> = SHAPE_ARM_SWEEP_SPECS
        .iter()
        .filter(|spec| requested_shape.as_deref().is_none_or(|shape| shape == spec.family))
        .collect();
    assert!(
        !specs.is_empty(),
        "PROXIMA_SWEEP_SHAPE={:?} matched no family in SHAPE_ARM_SWEEP_SPECS",
        requested_shape
    );

    let failures: Vec<String> =
        specs.iter().flat_map(|spec| run_shape_arm_sweep(&device, &queue, spec)).collect();

    assert!(
        failures.is_empty(),
        "parity failed for {} cell(s): {failures:#?}",
        failures.len()
    );
}

/// ROW 334's own repro tool -- `attn_q` (PASS) then `attn_output` (FAIL,
/// ROW 333), same shape (rows=4096, k=4096, Q4_K), REAL tensor count (212,
/// `tensor_count_for_shape`'s own value for this shape), the second-of-pair
/// signature ROW 333's table shows on every same-byte-size arm pair. Two
/// arms only -- not the full 10-arm, ~25-minute `decode_shape_roofline_ladder`
/// -- so this reproduces or clears in the time one arm's own real-scale run
/// costs.
#[test]
#[ignore = "synthesizes real quantized bytes through the real encoder and needs a real metal device"]
fn row334_attn_q_then_attn_output_real_scale() {
    let attn_q = DECODE_SHAPES.iter().find(|shape| shape.family == "attn_q").expect("attn_q shape exists");
    let attn_output = DECODE_SHAPES
        .iter()
        .find(|shape| shape.family == "attn_output")
        .expect("attn_output shape exists");
    let mut weight_keepalive: Vec<Vec<u8>> = Vec::new();
    let failures: Vec<String> = [attn_q, attn_output]
        .into_iter()
        .filter_map(|shape| {
            let tensor_count = tensor_count_for_shape(shape);
            run_shape_arm(
                shape.family,
                shape.codec,
                shape.rows,
                shape.k,
                tensor_count,
                shape.seed,
                &mut weight_keepalive,
            )
        })
        .collect();
    assert!(failures.is_empty(), "parity failed for {} arm(s): {failures:#?}", failures.len());
}

/// ROW 334's own verification pass: the exact four arms ROW 333's full-scale
/// run found failing (`attn_v`, `attn_output`, `ffn_up`, `ffn_down`), at
/// their REAL per-family tensor counts, run as ONE pass (not the full
/// 10-arm ladder -- the other six arms already passed at full scale and this
/// fix touches no path any of them exercise differently). `ffn_up` directly
/// precedes `ffn_down` here, the SAME same-byte-size adjacency
/// (`rows*row_bytes(k)` is identical for `ffn_gate`/`ffn_up`/`ffn_down`, this
/// file's own `tensor_count_for_shape` doc) that exposed the bug this row
/// fixes, so this run still exercises the adversarial ordering, not just the
/// arms in isolation.
#[test]
#[ignore = "synthesizes real quantized bytes through the real encoder and needs a real metal device"]
fn row334_four_failing_arms_real_scale() {
    let families = ["attn_v", "attn_output", "ffn_up", "ffn_down"];
    let shapes: Vec<&DecodeShape> = families
        .iter()
        .map(|family| {
            DECODE_SHAPES.iter().find(|shape| shape.family == *family).expect("family exists in DECODE_SHAPES")
        })
        .collect();
    let mut weight_keepalive: Vec<Vec<u8>> = Vec::new();
    let failures: Vec<String> = shapes
        .into_iter()
        .filter_map(|shape| {
            let tensor_count = tensor_count_for_shape(shape);
            run_shape_arm(
                shape.family,
                shape.codec,
                shape.rows,
                shape.k,
                tensor_count,
                shape.seed,
                &mut weight_keepalive,
            )
        })
        .collect();

    assert!(failures.is_empty(), "parity failed for {} arm(s): {failures:#?}", failures.len());
}

// ---- ROW 327: output-head buffer-kind vs kernel-variant ladder ----
//
// ROW 326 found two variables confounded between the in-program head
// dispatch (45 GB/s) and this file's own head arm (231 GB/s): the compiled
// kernel VARIANT (production reduces the MIDDLE axis of a `[seq,emb,vocab]`
// intermediate, activation operand first -- `..._floatf6B1_ax_0_2_w64`; this
// file's [`multi_tensor_matmul_program`] reduces the LAST axis, weight
// operand first -- `..._float6fB1_ax_0_1_w64`) and the weight BUFFER kind
// (production's weight buffer is a no-copy `MTLBuffer` over the mmap'd real
// gguf; this file's own head arm synthesizes fresh `StorageModeShared`
// bytes). This 2x2 isolates each axis independently over the SAME real
// `output.weight` shape (`Q6_K`, rows=32000ish, k=4096).

/// Which axis order/operand order [`production_head_program`] and
/// [`multi_tensor_matmul_program`] each bake into the reduce -- the KERNEL
/// VARIANT half of ROW 327's 2x2. `LadderReduceLast` delegates straight to
/// [`multi_tensor_matmul_program`] (weight operand first, reduces the LAST
/// axis, `_ax_0_1` in `kernel_cache_key`); `ProductionReduceMiddle` mirrors
/// `mistral_forward_program`'s own output-head `elementwise`/`reduce` calls
/// (`proxima-tensor/src/spec.rs:4181-4195`: `(normed_final, "sd->sdv")` then
/// `(lm_head, "dv->sdv")`, `reduce(.., "sdv->sdv", "sv->sdv")`) exactly --
/// activation operand first, reduces the MIDDLE axis, `_ax_0_2`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum OpShape {
    LadderReduceLast,
    ProductionReduceMiddle,
}

impl OpShape {
    const fn label(self) -> &'static str {
        match self {
            OpShape::LadderReduceLast => "ladder_reduce_last",
            OpShape::ProductionReduceMiddle => "production_reduce_middle",
        }
    }
}

/// Mirrors `mistral_forward_program`'s output head exactly
/// (`proxima-tensor/src/spec.rs:4181-4195`), generalized over
/// `weight_names.len()` independent weight nodes sharing ONE `activation`
/// input so [`omega::metal::plan`]/`execute_plan` batch every dispatch into
/// one command buffer, same posture as [`multi_tensor_matmul_program`].
/// Iteration space is rank 3, `(s, d, v)` = (seq=1, embedding=`k`,
/// vocab=`rows`) -- the SAME letters spec.rs's own notation strings name.
/// `activation` is operand 0 of the elementwise product (spec.rs's
/// `(normed_final, "sd->sdv")` comes first); the reduce's `out_map` keeps
/// axes `[0, 2]` (`s`, `v`), reducing axis 1 (`d`, the MIDDLE axis) -- spec.rs's
/// `reduce(.., "sdv->sdv", "sv->sdv")` resolves to exactly this.
fn production_head_program(
    weight_names: &[String],
    rows: u32,
    k: u32,
    weight_dtype: DType,
) -> (Vec<Op>, Vec<NodeId>) {
    // Weight nodes declared BEFORE `activation` -- `packed_operands_of`
    // (`omega/src/metal.rs:574`) zips `block_nodes` (program declaration
    // order) against the caller's own `blocks` slice POSITIONALLY, and
    // [`run_head_arm`] binds every weight's block before `activation`'s
    // (the SAME convention [`multi_tensor_matmul_program`] declares in).
    // Declaration order is independent of OPERAND order inside the
    // elementwise product below -- `activation` still comes first there,
    // matching spec.rs's own `(normed_final, "sd->sdv")` before
    // `(lm_head, "dv->sdv")` -- so this stays a faithful mirror of
    // production's BoundOp while satisfying this driver's own block-binding
    // contract.
    let mut program = Vec::new();
    let weight_nodes: Vec<NodeId> = weight_names
        .iter()
        .map(|name| {
            append(
                &mut program,
                Op::Input {
                    dtype: weight_dtype,
                    shape: vec![Extent::Static(k), Extent::Static(rows)],
                    name: Some(name.clone()),
                },
            )
        })
        .collect();
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1), Extent::Static(k)],
            name: Some("activation".into()),
        },
    );
    let mut sums = Vec::with_capacity(weight_names.len());
    for (index, weight) in weight_nodes.into_iter().enumerate() {
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (activation, IndexMap::Affine(map::projection(3, &[0, 1]))),
                    (weight, IndexMap::Affine(map::projection(3, &[1, 2]))),
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
                out_map: IndexMap::Affine(map::projection(3, &[0, 2])),
                keep: Keep::Reduce,
                name: Some(format!("production_head_{index}")),
            }),
        );
        sums.push(sum);
    }
    (program, sums)
}

/// Builds either op shape over `weight_names`/`rows`/`k` -- the single
/// dispatch point [`run_head_arm`] calls so its own body never branches on
/// [`OpShape`] directly.
fn head_program(
    op_shape: OpShape,
    weight_names: &[String],
    rows: u32,
    k: u32,
    weight_dtype: DType,
) -> (Vec<Op>, Vec<NodeId>) {
    match op_shape {
        OpShape::LadderReduceLast => {
            multi_tensor_matmul_program(weight_names, rows, k, weight_dtype, "activation")
        }
        OpShape::ProductionReduceMiddle => production_head_program(weight_names, rows, k, weight_dtype),
    }
}

/// Which weight BYTES [`run_head_arm`] binds every named weight input to --
/// the BUFFER KIND half of ROW 327's 2x2. `Synthesized` generates
/// `tensor_count` DISTINCT `Q6_K` tensors through the real encoder, exactly
/// [`run_shape_arm`]'s own posture. `RealNoCopy` binds every one of the
/// `tensor_count` named weight inputs to the SAME slice, borrowed straight
/// out of the real gguf's `mmap` -- 19 (or 1) dispatches of the ONE real
/// `output.weight` tensor this checkpoint has, not 19 distinct ones (ROW
/// 327's own doc: "19 distinct tensors are not available, one real head").
/// [`omega::backend::register_checkpoint_mapping`] must be called on the
/// SAME mapping before [`omega::metal::plan`]/`execute_plan` runs, so
/// `upload_packed_bytes`'s `checkpoint_mapping_offset` fallback (this
/// tensor's own byte offset is essentially never page-aligned by itself)
/// hands the GPU a no-copy `MTLBuffer` over the WHOLE mapping plus this
/// tensor's byte offset into it -- production's own exact upload path
/// (`omega/src/metal.rs:4230-4269`), not a hand-rolled Metal call.
enum HeadBufferKind<'a> {
    Synthesized,
    RealNoCopy { tensor_bytes: &'a [u8] },
}

impl HeadBufferKind<'_> {
    const fn label(&self) -> &'static str {
        match self {
            HeadBufferKind::Synthesized => "synthesized",
            HeadBufferKind::RealNoCopy { .. } => "real_no_copy",
        }
    }
}

/// One measured (op_shape, buffer_kind) cell of ROW 327's 2x2, at one
/// `tensor_count` -- called once at `tensor_count=1` (isolates the
/// first-dispatch cost per command buffer, [`REPEATS`] separate 1-dispatch
/// buffers) and once at `tensor_count` large enough to clear
/// [`MIN_TIMED_BYTES`] (steady per-dispatch mean). Runs parity against the
/// REAL codec's own dequantize-and-dot CPU oracle only when `check_parity`
/// is `true` -- the two calls per arm bind bit-identical weight bytes for
/// their first tensor, so checking it twice would be redundant, not a
/// stronger proof.
#[allow(clippy::too_many_arguments)]
fn run_head_arm(
    label: &str,
    op_shape: OpShape,
    buffer_kind: &HeadBufferKind<'_>,
    rows: u32,
    k: u32,
    tensor_count: usize,
    seed_base: u64,
    assert_parity: bool,
) -> (String, f64, f64, Option<String>) {
    let row_bytes = ShapeCodec::Q6K.row_bytes(k as usize);
    let tensor_bytes = (rows as usize * row_bytes) as u64;
    let total_timed_bytes = tensor_bytes * tensor_count as u64;
    let weight_names: Vec<String> = (0..tensor_count).map(|index| format!("{label}_{index}")).collect();

    let synth_bytes = match buffer_kind {
        HeadBufferKind::Synthesized => {
            Some(synth_weight_bytes(ShapeCodec::Q6K, seed_base, tensor_count, rows as usize, k as usize))
        }
        HeadBufferKind::RealNoCopy { .. } => None,
    };

    let mut activation_lcg = Lcg(seed_base + 999);
    let activation: Vec<f32> = (0..k).map(|_| activation_lcg.next_unit() * 4.0 - 2.0).collect();

    let (program, sums) = head_program(op_shape, &weight_names, rows, k, DType::UInt8);
    let blocks_weights: Vec<QuantizedBlock<'_>> = match (buffer_kind, &synth_bytes) {
        (HeadBufferKind::Synthesized, Some(bytes)) => bytes
            .chunks_exact(rows as usize * row_bytes)
            .map(|slice| ShapeCodec::Q6K.quantized_block(slice))
            .collect(),
        (HeadBufferKind::RealNoCopy { tensor_bytes }, _) => {
            (0..tensor_count).map(|_| ShapeCodec::Q6K.quantized_block(tensor_bytes)).collect()
        }
        _ => unreachable!("synth_bytes is Some exactly when buffer_kind is Synthesized"),
    };
    let mut blocks = blocks_weights;
    blocks.push(QuantizedBlock::Float32(&activation));

    let mut plan = omega::metal::plan(&program, &[], &blocks, &sums)
        .expect("plan resolves the output-head matmul over every weight input");
    // ROW 334: weight nodes are resident-eligible ONLY for `RealNoCopy` --
    // that branch aliases the SAME real, mmap'd, process-lifetime tensor
    // bytes on every call (`ShapeCodec::Q6K.quantized_block(tensor_bytes)`
    // repeated `tensor_count` times), exactly `Plan::mark_resident`'s own
    // documented precondition. `Synthesized` allocates a fresh `Vec<u8>`
    // this function itself drops on return -- the same shape of bug
    // `run_shape_arm` had (see that function's own ROW 334 comment): a
    // later call at the SAME `rows`/`k`/`tensor_count` can land its own
    // fresh `Vec` at the address `NOCOPY_BUFFERS` still has cached under,
    // and get served the PRIOR call's stale weight bytes.
    let mut resident_names: BTreeSet<&str> = match buffer_kind {
        HeadBufferKind::RealNoCopy { .. } => weight_names.iter().map(String::as_str).collect(),
        HeadBufferKind::Synthesized => BTreeSet::new(),
    };
    resident_names.insert("activation");
    plan.mark_resident(&resident_names);
    let keys = plan.kernel_keys().expect("every resolved position emits a kernel key");
    let key = keys
        .last()
        .cloned()
        .expect("the reduce position's own key is the plan's last resolved position");

    let mut first_tensor_output: Vec<f32> = Vec::new();
    let elapsed_samples = warmed_up_samples(|| {
        let started = Instant::now();
        let evaluated = omega::metal::execute_plan(&plan, &blocks)
            .expect("execute_plan runs the output-head matmul over every weight input");
        let elapsed = started.elapsed();
        first_tensor_output = evaluated
            .get(sums[0])
            .map(|(data, _)| data.to_vec())
            .expect("first weight's own reduce node is a requested output");
        elapsed
    });
    let ns_per_dispatch = ns_per_dispatch_samples(&elapsed_samples, tensor_count);
    let ns_stats = sample_stats(&ns_per_dispatch);
    let samples = gbps_samples(&elapsed_samples, total_timed_bytes);
    let gbps_stats = sample_stats(&samples);
    let median_ns_per_dispatch = ns_stats.median;
    let median_gbps = gbps_stats.median;

    let parity_failure = if assert_parity {
        let first_tensor_bytes: &[u8] = match (buffer_kind, &synth_bytes) {
            (HeadBufferKind::Synthesized, Some(bytes)) => &bytes[..rows as usize * row_bytes],
            (HeadBufferKind::RealNoCopy { tensor_bytes }, _) => tensor_bytes,
            _ => unreachable!("synth_bytes is Some exactly when buffer_kind is Synthesized"),
        };
        let cpu_reference =
            codec_cpu_reference_first_rows(ShapeCodec::Q6K, first_tensor_bytes, k as usize, &activation);
        check_parity(
            &format!("{label} vs cpu_reference"),
            &first_tensor_output[..PARITY_ROWS],
            &cpu_reference,
        )
    } else {
        None
    };

    println!(
        "arm={label} op_shape={} buffer_kind={} key={key} tensor_count={tensor_count} \
         median_ns_per_dispatch={:.0} mean_ns_per_dispatch={:.0} median_gbps={:.2} mean_gbps={:.2} \
         min_gbps={:.2} max_gbps={:.2} cov_pct={:.2}",
        op_shape.label(),
        buffer_kind.label(),
        median_ns_per_dispatch,
        ns_stats.mean,
        median_gbps,
        gbps_stats.mean,
        gbps_stats.min,
        gbps_stats.max,
        gbps_stats.cov_pct
    );

    (key, median_ns_per_dispatch, median_gbps, parity_failure)
}

/// ROW 327: isolates the two variables ROW 326 found confounded between
/// production's own in-program output-head dispatch (2.19 ms first dispatch,
/// 0.41 ms a duplicate; 45 GB/s in-program) and this file's own head arm
/// (231 GB/s) -- the compiled kernel VARIANT
/// ([`OpShape::ProductionReduceMiddle`] vs [`OpShape::LadderReduceLast`]) and
/// the weight BUFFER kind ([`HeadBufferKind::RealNoCopy`] vs
/// [`HeadBufferKind::Synthesized`]) -- as an honest 2x2 over the SAME real
/// `output.weight` shape.
///
/// `#[ignore]`d: depends on the same host-local openchat GGUF checkout as
/// this file's other real-checkpoint arms, and needs a real Metal device.
#[test]
#[ignore = "depends on a host-local openchat gguf checkout and a real metal device"]
fn head_buffer_kind_by_kernel_variant() {
    let checkpoint = checkpoint_path();
    let path = Path::new(&checkpoint);
    let (parsed, file_len, _file) =
        real_gguf_header(path).expect("real openchat gguf checkpoint header parses");
    let real_tensor = locate_real_tensor(&parsed, file_len, "output.weight", GgmlType::Q6_K)
        .expect("output.weight is Q6_K in this checkpoint");
    assert_eq!(real_tensor.in_dim, IN_DIM, "output.weight's own k must match this ladder's IN_DIM");

    let mapped = MappedFile::open(path).expect("mmap the real openchat gguf checkpoint");
    omega::backend::register_checkpoint_mapping(mapped.as_slice());
    let real_tensor_bytes = &mapped.as_slice()
        [real_tensor.byte_offset as usize..(real_tensor.byte_offset + real_tensor.byte_len) as usize];

    let arms: [(OpShape, HeadBufferKind<'_>); 4] = [
        (OpShape::LadderReduceLast, HeadBufferKind::Synthesized),
        (OpShape::ProductionReduceMiddle, HeadBufferKind::Synthesized),
        (OpShape::LadderReduceLast, HeadBufferKind::RealNoCopy { tensor_bytes: real_tensor_bytes }),
        (OpShape::ProductionReduceMiddle, HeadBufferKind::RealNoCopy { tensor_bytes: real_tensor_bytes }),
    ];

    let mut parity_failures = Vec::new();
    println!(
        "=== ROW 327: output-head buffer-kind x kernel-variant, real output.weight \
         (rows={} k={IN_DIM}, device ceiling {DEVICE_CEILING_GBPS:.2} GB/s) ===",
        real_tensor.out_dim
    );
    for (op_shape, buffer_kind) in &arms {
        let label = format!("{}_{}", op_shape.label(), buffer_kind.label());
        let rows = match buffer_kind {
            HeadBufferKind::Synthesized => Q6K_HEAD_ROWS as u32,
            HeadBufferKind::RealNoCopy { .. } => real_tensor.out_dim as u32,
        };

        let (_, first_dispatch_median_ns, _, _) =
            run_head_arm(&label, *op_shape, buffer_kind, rows, IN_DIM as u32, 1, 90_000, false);

        let (key, steady_dispatch_median_ns, steady_median_gbps, parity_failure) = run_head_arm(
            &label,
            *op_shape,
            buffer_kind,
            rows,
            IN_DIM as u32,
            Q6K_HEAD_TENSOR_COUNT,
            91_000,
            true,
        );
        if let Some(failure) = parity_failure {
            parity_failures.push(failure);
        }

        println!(
            "summary arm={label} key={key} first_dispatch_median_ms={:.4} steady_dispatch_median_ms={:.4} \
             steady_median_gbps={steady_median_gbps:.2} vs_row_326_in_program_first_ms=2.19 \
             vs_row_326_in_program_dup_ms=0.41",
            first_dispatch_median_ns / 1e6,
            steady_dispatch_median_ns / 1e6
        );
    }

    assert!(
        parity_failures.is_empty(),
        "parity failed for {} arm(s): {parity_failures:#?}",
        parity_failures.len()
    );
}

/// One dispatch inside [`time_bare_dispatch_sequence`]'s own timed command
/// buffer -- everything [`time_batch_l3_shape_threads`] binds per repeat
/// (`weight@0`, `activation@1`, `output@2`, `uniform@3`), but per-DISPATCH
/// rather than per-BATCH-of-the-same-shape, so a heterogeneous program
/// (varying `rows`/`k`/codec/pipeline across dispatches, ROW 339's own whole
/// decode token) can be expressed. Each field is a distinct real Metal
/// binding or geometry parameter this dispatch needs; splitting them further
/// would not reduce what the caller supplies.
struct BareDispatchOp<'a> {
    pipeline: &'a ProtocolObject<dyn MTLComputePipelineState>,
    weight: &'a ProtocolObject<dyn MTLBuffer>,
    weight_offset: usize,
    activation: &'a ProtocolObject<dyn MTLBuffer>,
    uniform: &'a ProtocolObject<dyn MTLBuffer>,
    grid_threads: usize,
    threadgroup_width: usize,
    /// Buffer-index this op's own compiled kernel expects each binding at --
    /// `[0, 1, 2, 3]` (weight/activation/output/uniform) for every hand-
    /// written `q4k_pair_dot`/`q6k_pair_dot` body arm (A-D), but PRODUCTION's
    /// own emitted reduce declares operands activation-first
    /// (`production_head_program`'s own doc), so arm E's [`Kernel::bindings`]
    /// order differs and must be read off that kernel rather than assumed --
    /// see [`production_reduce_kernel`]'s own doc.
    weight_index: usize,
    activation_index: usize,
    output_index: usize,
    uniform_index: usize,
}

/// Encodes `ops` in program order into ONE plain `computeCommandEncoder()`
/// -- no `HazardTrackingModeUntracked` buffers, no `memoryBarrierWithScope`,
/// no plan cache, no uniform arena: llama.cpp's own default serial encoder,
/// the same posture [`time_batch_l3_shape_threads`] already uses for a
/// homogeneous batch, generalized here to a per-dispatch pipeline/shape so a
/// whole token's mixed-shape op sequence can be timed as ONE command buffer
/// (ROW 339's module doc). `setComputePipelineState` is called on every
/// dispatch, matching `omega::metal::encode_op`'s own per-op call
/// (`omega/src/metal.rs:6047`) regardless of whether the previous dispatch
/// used the same pipeline.
fn time_bare_dispatch_sequence(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    output: &ProtocolObject<dyn MTLBuffer>,
    ops: &[BareDispatchOp<'_>],
) -> Duration {
    let command_buffer = queue.commandBuffer().expect("command buffer");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("command buffer refused to hand out a compute encoder");
    for op in ops {
        encoder.setComputePipelineState(op.pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(op.weight), op.weight_offset, op.weight_index);
            encoder.setBuffer_offset_atIndex(Some(op.activation), 0, op.activation_index);
            encoder.setBuffer_offset_atIndex(Some(output), 0, op.output_index);
            encoder.setBuffer_offset_atIndex(Some(op.uniform), 0, op.uniform_index);
        }
        let grid = MTLSize {
            width: op.grid_threads,
            height: 1,
            depth: 1,
        };
        let threadgroup = MTLSize {
            width: op.threadgroup_width,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
    }
    encoder.endEncoding();
    let started = Instant::now();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    started.elapsed()
}

/// `reduction_dims` restated over public [`BoundOp`] fields --
/// `omega::msl::reduction_dims` is `pub(crate)`, not exported, so
/// [`pack_production_reduce_uniforms`] (this crate never sees the private
/// packer either, ROW 344's own question) re-derives the identical axis set
/// its own doc defines: every axis NOT in `output_axes`, ascending.
fn reduction_dims_port(rank: usize, output_axes: &[u16]) -> Vec<u16> {
    (0..rank as u16).filter(|axis| !output_axes.contains(axis)).collect()
}

fn push_i64(bytes: &mut Vec<u8>, value: i64) {
    bytes.extend_from_slice(&value.to_ne_bytes());
}

fn push_i64_row(bytes: &mut Vec<u8>, values: &[i64], width: usize) {
    for slot in 0..width {
        push_i64(bytes, values.get(slot).copied().unwrap_or(0));
    }
}

fn push_gathered_extent_row(bytes: &mut Vec<u8>, extents: &[u64], axes: &[u16], width: usize) {
    for slot in 0..width {
        let value = axes.get(slot).map(|axis| extents[*axis as usize] as i64).unwrap_or(0);
        push_i64(bytes, value);
    }
}

fn gathered_extent_product(extents: &[u64], axes: &[u16]) -> i64 {
    axes.iter().map(|axis| extents[*axis as usize] as i64).product()
}

/// Ports `omega::metal::pack_reduce_uniforms` field-for-field over public
/// [`BoundOp`] accessors -- that function is `omega`-private (this ladder is
/// a separate integration-test crate, so it genuinely cannot call it), and
/// ROW 344's own question needs the identical bytes production's own driver
/// would pack for this fold, not an approximation. Mirrors the `Uniforms`
/// struct `omega::msl::render_reduce` declares (`omega/src/msl.rs:3827-3833`'s
/// own doc): `output_total`, `reduction_total`, `output_extents[..]`,
/// `reduction_extents[..]`, `operand_base[..]`, `operand_strides[..][..]`,
/// `out_base`, `out_strides[..]`. Sound only for the gather-free,
/// epilogue-free, non-scatter fold [`production_reduce_kernel`] resolves --
/// [`production_head_program`]'s own reduce never gathers, scatters, or
/// fuses an epilogue, so every `assert!` below holds by construction, not by
/// luck.
fn pack_production_reduce_uniforms(bound: &BoundOp) -> Vec<u8> {
    let BoundOpKind::Reduce {
        output_axes,
        out_layout,
        epilogue_operands,
        out_scatter,
        ..
    } = &bound.kind
    else {
        panic!("production reduce fold expected, found {}", bound.kind.name());
    };
    assert!(epilogue_operands.is_empty(), "epilogue-free fold expected for ROW 344's own head program");
    assert!(out_scatter.is_none(), "affine (non-scatter) fold expected for ROW 344's own head program");
    assert!(
        bound.operands().iter().all(|(_, _, gather)| gather.is_none()),
        "gather-free operands expected for ROW 344's own head program"
    );

    let rank_len = bound.extents.len().max(1);
    let output_rank_len = output_axes.len().max(1);
    let reduce_axes = reduction_dims_port(bound.extents.len(), output_axes);
    let reduce_rank_len = reduce_axes.len().max(1);

    let mut bytes = Vec::new();
    push_i64(&mut bytes, gathered_extent_product(&bound.extents, output_axes));
    push_i64(&mut bytes, gathered_extent_product(&bound.extents, &reduce_axes));
    push_gathered_extent_row(&mut bytes, &bound.extents, output_axes, output_rank_len);
    push_gathered_extent_row(&mut bytes, &bound.extents, &reduce_axes, reduce_rank_len);
    for (_, layout, _) in bound.operands() {
        push_i64(&mut bytes, layout.base);
    }
    for (_, layout, _) in bound.operands() {
        push_i64_row(&mut bytes, &layout.strides, rank_len);
    }
    push_i64(&mut bytes, out_layout.base);
    push_i64_row(&mut bytes, &out_layout.strides, rank_len);
    bytes
}

/// PRODUCTION's own emitted kernel for one whole-token matvec shape, ready
/// for [`time_bare_dispatch_sequence`]'s bare per-op encoding -- arm E's
/// answer to arm C, the SAME `q4k_pair_dot`/`q6k_pair_dot`-free path
/// [`run_shape_arm`]/[`run_head_arm`] already exercise through
/// `omega::metal::plan`/`execute_plan`, but resolved and emitted directly
/// (`proxima_tensor::{infer, bind}` then [`omega::emit`]) so THIS caller
/// controls the dispatch (no hazard tracker, no arena, no plan cache) the
/// same way arms A-D already do for the ladder's own hand-written body.
/// Builds the exact `sd->sdv`/`dv->sdv` reduce-middle op
/// [`production_head_program`] mirrors from `mistral_forward_program`'s own
/// output head (activation operand first, weight second -- that function's
/// own doc), which is why [`Kernel::bindings`] is read back rather than
/// assumed: production's own operand order differs from this ladder's
/// weight-first `q4k_pair_dot`/`q6k_pair_dot` convention.
struct ProductionKernel {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    uniform: Retained<ProtocolObject<dyn MTLBuffer>>,
    weight_index: usize,
    activation_index: usize,
    output_index: usize,
    uniform_index: usize,
    grid_threads: usize,
    threadgroup_width: usize,
}

/// ROW 350 bisection: the emitted kernel's generic N-D coordinate
/// decomposition (`coord_q_cache`, mod/div against `u.output_extents`) is
/// provably a no-op for this test's shape -- the output is rank-2 `[1,
/// rows]`, so the leading "sequence" axis is always extent 1 and every
/// `% output_extents[0]` / `/ output_extents[0]` in the generated text
/// yields `0`/no-change for every thread, and the FIRST mod/div pair
/// (`% output_extents[1]` / `/ output_extents[1]`) always yields
/// `flat`/`0` for the same reason (`flat < rows` always). This rewrite
/// replaces that decomposition with the direct single-axis addressing the
/// hand-written ladder kernel already uses, to measure whether the 16
/// division-class ops/thread it removes is where the 5.3ms gap lives.
/// Applied as a POST-EMIT string rewrite (not a change to `omega/src`) so
/// this stays a measurement, not a landed fix.
fn row350_direct_addressing_rewrite(source: &str) -> String {
    let setup_old = "    long weight_base[4];\n    long other_base[4];\n    long coord_q_cache[4][3];\n    for (int q = 0; q < 4; ++q) {\n        long flat = group_first + q;\n        long remaining_q = flat;\n        for (int d = 0; d < 3; ++d) { coord_q_cache[q][d] = 0; }\n        coord_q_cache[q][2] = remaining_q % u.output_extents[1]; remaining_q /= u.output_extents[1];\n        coord_q_cache[q][0] = remaining_q % u.output_extents[0]; remaining_q /= u.output_extents[0];\n        long wb = u.operand_base[1];\n        long ob = u.operand_base[0];\n        wb += coord_q_cache[q][0] * u.operand_strides[1][0];\n        ob += coord_q_cache[q][0] * u.operand_strides[0][0];\n        wb += coord_q_cache[q][2] * u.operand_strides[1][2];\n        ob += coord_q_cache[q][2] * u.operand_strides[0][2];\n        weight_base[q] = wb;\n        other_base[q] = ob;\n    }\n";
    let setup_new = "    long weight_base[4];\n    long other_base[4];\n    for (int q = 0; q < 4; ++q) {\n        long flat = group_first + q;\n        weight_base[q] = u.operand_base[1] + flat * u.operand_strides[1][2];\n        other_base[q] = u.operand_base[0] + flat * u.operand_strides[0][2];\n    }\n";
    let output_old = "    for (int q = 0; q < 4; ++q) {\n        float reduced = simd_sum(sumf[q]);\n        long flat = group_first + q;\n        if (lane == 0u && flat < u.output_total) {\n            long out_offset = u.out_base;\n            out_offset += coord_q_cache[q][0] * u.out_strides[0];\n            out_offset += coord_q_cache[q][1] * u.out_strides[1];\n            out_offset += coord_q_cache[q][2] * u.out_strides[2];\n            out[out_offset] = reduced;\n        }\n    }\n}\n";
    let output_new = "    for (int q = 0; q < 4; ++q) {\n        float reduced = simd_sum(sumf[q]);\n        long flat = group_first + q;\n        if (lane == 0u && flat < u.output_total) {\n            long out_offset = u.out_base + flat * u.out_strides[2];\n            out[out_offset] = reduced;\n        }\n    }\n}\n";
    assert!(source.contains(setup_old), "row350 setup pattern not found in emitted source");
    assert!(source.contains(output_old), "row350 output pattern not found in emitted source");
    source.replace(setup_old, setup_new).replace(output_old, output_new)
}

/// ROW 350 cell selector: `0` = baseline (byte-identical to production's own
/// `omega::emit` output, no rewrite -- the default so this test's normal runs
/// are unaffected), `1` = [`row350_direct_addressing_rewrite`] applied to
/// every Q4K family kernel's source before compiling (measured 22.71ms ->
/// 17.67ms, see `docs/discipline.md` ROW 350). Flipped by hand between timed
/// bisection runs -- this is a measurement knob, not a runtime config
/// surface, and ships default-`0` (inert).
const ROW350_CELL: u32 = 0;

fn production_reduce_kernel(
    device: &ProtocolObject<dyn MTLDevice>,
    codec: omega::PackedCodec,
    rows: u32,
    k: u32,
) -> ProductionKernel {
    let (program, sums) = production_head_program(&["weight".to_string()], rows, k, DType::UInt8);
    // `production_head_program` declares the weight node(s) before
    // `activation` (its own doc), so with exactly one weight name the input
    // nodes are `NodeId(0)` (weight) then `NodeId(1)` (activation) --
    // `proxima_tensor::append`'s own doc: a `NodeId` IS the node's position
    // in `program`.
    let weight_node = NodeId(0);
    let activation_node = NodeId(1);

    let shapes = infer(&program, &[]).expect("production reduce program's shapes infer");
    let mut bound_ops = bind(&program, &shapes, &sums).expect("production reduce program binds");
    // `bind()` alone lays out the weight operand row-major over its DECLARED
    // axis order (`correct_packed_matmul_layouts`'s own doc) -- wrong for a
    // packed `Q4_K`/`Q6_K` weight's real on-disk bytes. `omega::metal::prepare`
    // (`omega/src/metal.rs:3253`) always runs this correction between `bind`
    // and `emit`; skipping it here made the emitted kernel fall off the
    // packed-row fast path entirely (a from-scratch reproduction measured
    // ~1.24s/token instead of ~17ms -- this call is what `run_shape_arm`/
    // `run_head_arm` get for free through `omega::metal::plan`).
    let packed_node_set: BTreeSet<NodeId> = [NodeId(0)].into_iter().collect();
    correct_packed_matmul_layouts(&mut bound_ops, &packed_node_set);
    let bound = bound_ops
        .into_iter()
        .find(|op| op.node == sums[0])
        .expect("the head sum's own fused reduce is present in the bound program");

    let packed_operands: omega::PackedOperands = BTreeMap::from([(weight_node, codec)]);
    let kernel =
        omega::emit(&bound, &packed_operands).expect("production reduce fold emits an MSL kernel");

    let weight_index = kernel
        .bindings
        .iter()
        .position(|binding| matches!(binding, omega::Binding::Input(node) if *node == weight_node))
        .expect("weight binding present in the emitted kernel");
    let activation_index = kernel
        .bindings
        .iter()
        .position(|binding| matches!(binding, omega::Binding::Input(node) if *node == activation_node))
        .expect("activation binding present in the emitted kernel");
    let output_index = kernel
        .bindings
        .iter()
        .position(|binding| matches!(binding, omega::Binding::Output(_)))
        .expect("output binding present in the emitted kernel");
    let uniform_index = kernel
        .bindings
        .iter()
        .position(|binding| matches!(binding, omega::Binding::Uniforms))
        .expect("uniforms binding present in the emitted kernel");

    let uniform_bytes = pack_production_reduce_uniforms(&bound);
    let uniform = shared_buffer_from_bytes(device, &uniform_bytes);
    // Q6K's head kernel does not share this wrapper's exact text (its own
    // preamble differs enough that the byte-exact `setup_old`/`output_old`
    // patterns below do not match it) -- the rewrite targets the Q4K FFN
    // shapes only, which is 224 of this test's 225 dispatches anyway.
    let source = if ROW350_CELL == 1 && matches!(codec, omega::PackedCodec::Q4K) {
        row350_direct_addressing_rewrite(&kernel.source)
    } else {
        kernel.source.clone()
    };
    let pipeline = compile_pipeline(device, &source, &kernel.entry, MTLMathMode::Relaxed);
    let threadgroup_width = kernel.grid.threadgroup_width.unwrap_or(64) as usize;

    ProductionKernel {
        pipeline,
        uniform,
        weight_index,
        activation_index,
        output_index,
        uniform_index,
        grid_threads: kernel.grid.threads as usize,
        threadgroup_width,
    }
}

/// One of the seven per-layer `Q4_K` matvec shapes a real decode token
/// dispatches, in llama's own program order (`q`, `k`, `v`, `o`, `gate`,
/// `up`, `down`) -- rows/k restated from [`DECODE_SHAPES`]'s own Mistral-7B
/// GQA shapes, `Q4_K` for every one of them (this ladder's own doc: the
/// 4-of-32 real `Q5_K` `attn_v`/`ffn_down` layers are a detail this bare
/// dispatch count does not need to absorb).
struct WholeTokenLayerFamily {
    name: &'static str,
    rows: usize,
    k: usize,
    seed: u64,
}

const WHOLE_TOKEN_LAYER_FAMILIES: [WholeTokenLayerFamily; 7] = [
    WholeTokenLayerFamily { name: "q", rows: 4096, k: 4096, seed: 100_000 },
    WholeTokenLayerFamily { name: "k", rows: 1024, k: 4096, seed: 200_000 },
    WholeTokenLayerFamily { name: "v", rows: 1024, k: 4096, seed: 300_000 },
    WholeTokenLayerFamily { name: "o", rows: 4096, k: 4096, seed: 400_000 },
    WholeTokenLayerFamily { name: "gate", rows: 14_336, k: 4096, seed: 500_000 },
    WholeTokenLayerFamily { name: "up", rows: 14_336, k: 4096, seed: 600_000 },
    WholeTokenLayerFamily { name: "down", rows: 4096, k: 14_336, seed: 700_000 },
];

const WHOLE_TOKEN_HEAD_ROWS: usize = 32_000;
const WHOLE_TOKEN_HEAD_K: usize = 4096;

/// One dispatch-geometry arm for [`whole_token_matvec_sequence_bare`]: Arm A
/// is production's own DEFAULT compiled shape (1 simdgroup/threadgroup,
/// `MTLMathMode::Safe` -- the same default this file's L3 shape sweep names
/// `is_production_default`, line 1610-1612 above); Arm B is ROW 337/338's
/// packed-row sweep mechanism (`nsg=4`, `MTLMathMode::Fast`) applied to the
/// SAME whole-token sequence, so the two arms differ ONLY in compiled
/// pipeline/dispatch geometry, never in which ops run or how they are
/// encoded. ROW 340: Arm A was mislabeled "production's default kernel" --
/// production's actual default cell is `nsg=2`/`MTLMathMode::Relaxed` (ROW
/// 297/311), not `nsg=1`/`Safe`. Arm C is that ACTUAL production cell, timed
/// with the same bare llama-style encoding as A/B, so ROW 339's "9 ms is the
/// kernel" conclusion can be checked against the cell it should have used.
/// Arm D holds `nsg=2` fixed and swaps only the math mode to `Fast`,
/// isolating math from `nsg` on top of Arm C (ROW 338 found `nsg=2`/relaxed
/// and `nsg=4`/fast land within ~2% of each other on the FFN shape alone;
/// D checks whether that holds across the whole token).
struct WholeTokenArm {
    name: &'static str,
    simdgroups_per_tg: usize,
    math: MTLMathMode,
}

const WHOLE_TOKEN_ARMS: [WholeTokenArm; 4] = [
    WholeTokenArm { name: "A_llama_encoding_production_kernel", simdgroups_per_tg: 1, math: MTLMathMode::Safe },
    WholeTokenArm { name: "B_llama_encoding_nsg4_fast", simdgroups_per_tg: 4, math: MTLMathMode::Fast },
    WholeTokenArm { name: "C_llama_encoding_nsg2_relaxed_production_cell", simdgroups_per_tg: 2, math: MTLMathMode::Relaxed },
    WholeTokenArm { name: "D_llama_encoding_nsg2_fast", simdgroups_per_tg: 2, math: MTLMathMode::Fast },
];

/// ROW 339: does llama.cpp's own SERIAL encoding (one plain
/// `computeCommandEncoder()`, no hazard tracker, no `memoryBarrier`, no
/// uniform arena, no plan cache -- [`time_bare_dispatch_sequence`]'s own
/// doc) turn our matching-in-isolation kernels (ROW 336/338: `ffn_up` 236
/// GB/s, head 232 GB/s) into a WHOLE decode token near llama's 17.5 ms, or
/// does it stay near the in-program 22.64 ms (ROW 308) even with none of
/// production's own machinery in the way? Encodes the full per-layer
/// dispatch sequence (`q`, `k`, `v`, `o`, `gate`, `up`, `down` x 32 layers,
/// then `head` once -- 225 dispatches, this module doc's own count) into
/// ONE command buffer per token and times `commit()`-`waitUntilCompleted()`
/// around it, same discipline every other arm in this file uses
/// ([`warmed_up_samples`]): one untimed warm-up token, then [`REPEATS`]
/// timed tokens.
///
/// Every layer's `q`/`k`/`v`/`o`/`gate`/`up`/`down` weight is a genuinely
/// DISTINCT synthesized tensor (`seed_base + layer`,
/// [`synth_weight_bytes_parallel`]'s own per-tensor `Lcg` seed), packed
/// layer-major into ONE `StorageModeShared` buffer per family and uploaded
/// exactly once (`shared_buffer_from_bytes` copies at creation and is never
/// touched again -- [`run_shape_arm_sweep`]'s own zero-uploads-on-a-timed-
/// repeat invariant, satisfied here by construction the same way). Output is
/// never read back -- this arm times dispatch cost only, not correctness
/// (this function's own doc), so every dispatch's `row_sums` write lands in
/// the SAME shared scratch buffer regardless of the writing shape's own row
/// count.
///
/// `#[ignore]`d: CPU-bound minutes synthesizing real quantized bytes for a
/// full 7B-parameter-shaped program, and needs a real Metal device, same
/// posture as every other synthetic-data arm in this file.
/// ROW 361: llama's own `test-backend-ops perf` number per shape is
/// AMORTIZED -- `n_runs` copies of the SAME op duplicated into one graph
/// (`tests/test-backend-ops.cpp:675-677`), one `ggml_backend_graph_compute`
/// call per timed sample (`:704`), `us/run = total_time_us / total_runs`
/// (`:718`). ROW 360's per-shape ladder (`decode_shape_roofline_ladder`) is
/// ISOLATED -- one dispatch per timed sample, its own command-buffer
/// commit/wait -- which is why its per-shape sum overshoots our own
/// measured whole-token sequence (ROW 354). This function restricts
/// [`whole_token_matvec_sequence_bare`]'s own 225-dispatch, one-command-
/// buffer construction to ONE family's dispatches (its 32 per-layer
/// dispatches, or the head's 1), so the resulting number is AMORTIZED the
/// same way llama's is -- comparable to ROW 360's llama column, not to ROW
/// 360's isolated "ours" column. `attn_v`/`ffn_down` are synthesized as
/// `Q5_K` for every one of their dispatches (this checkpoint's real codec
/// for those two families, ROW 63's inventory: 4 of 32 layers; this cell
/// reports a clean single-codec number rather than mixing Q4_K/Q5_K bytes).
fn run_row361_family_amortized(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    family_name: &str,
) {
    let (label, codec, packed_codec, rows, k, seed, dispatch_count) = if family_name == "head" {
        (
            "head",
            ShapeCodec::Q6K,
            omega::PackedCodec::Q6K,
            WHOLE_TOKEN_HEAD_ROWS,
            WHOLE_TOKEN_HEAD_K,
            900_000,
            1_usize,
        )
    } else {
        let family = WHOLE_TOKEN_LAYER_FAMILIES
            .iter()
            .find(|family| family.name == family_name)
            .unwrap_or_else(|| panic!("ROW361_FAMILY={family_name} is not a known family name"));
        let uses_q5k = family_name == "v" || family_name == "down";
        let codec = if uses_q5k { ShapeCodec::Q5K } else { ShapeCodec::Q4K };
        let packed_codec = if uses_q5k { omega::PackedCodec::Q5K } else { omega::PackedCodec::Q4K };
        (family.name, codec, packed_codec, family.rows, family.k, family.seed, FFN_LAYERS)
    };

    let row_bytes = codec.row_bytes(k);
    let tensor_bytes = rows * row_bytes;
    let total_timed_bytes = (tensor_bytes * dispatch_count) as u64;
    println!(
        "=== ROW 361 amortized family={label} codec={} rows={rows} k={k} dispatches={dispatch_count} \
         bytes={total_timed_bytes} (one command buffer, production's own emitted kernel) ===",
        codec.name()
    );

    let weight_bytes = synth_weight_bytes_parallel(codec, seed, dispatch_count, k, tensor_bytes);
    let weight_buffer = shared_buffer_from_bytes(device, &weight_bytes);
    let offsets: Vec<usize> = (0..dispatch_count).map(|index| index * tensor_bytes).collect();

    let mut activation_lcg = Lcg(seed + 999);
    let activation: Vec<f32> = (0..k).map(|_| activation_lcg.next_unit() * 4.0 - 2.0).collect();
    // SAFETY: `activation` is a live `Vec<f32>` for the duration of this
    // call; the byte view is read-only and never outlives `activation`.
    let activation_bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(activation.as_ptr().cast::<u8>(), std::mem::size_of_val(activation.as_slice()))
    };
    let activation_buffer = shared_buffer_from_bytes(device, activation_bytes);

    let kernel = production_reduce_kernel(device, packed_codec, rows as u32, k as u32);

    let output = device
        .newBufferWithLength_options(rows * size_of::<f32>(), MTLResourceOptions::StorageModeShared)
        .expect("device allocates ROW 361's own scratch output buffer");

    let ops: Vec<BareDispatchOp<'_>> = offsets
        .iter()
        .map(|offset| BareDispatchOp {
            pipeline: &kernel.pipeline,
            weight: &weight_buffer,
            weight_offset: *offset,
            activation: &activation_buffer,
            uniform: &kernel.uniform,
            grid_threads: kernel.grid_threads,
            threadgroup_width: kernel.threadgroup_width,
            weight_index: kernel.weight_index,
            activation_index: kernel.activation_index,
            output_index: kernel.output_index,
            uniform_index: kernel.uniform_index,
        })
        .collect();

    let elapsed_samples = warmed_up_samples(|| time_bare_dispatch_sequence(queue, &output, &ops));
    let ms_samples: Vec<f64> = elapsed_samples.iter().map(Duration::as_secs_f64).map(|s| s * 1e3).collect();
    let ms_stats = sample_stats(&ms_samples);
    let gbps_samples_vec = gbps_samples(&elapsed_samples, total_timed_bytes);
    let gbps_stats = sample_stats(&gbps_samples_vec);
    let us_per_dispatch = ms_stats.median * 1000.0 / dispatch_count as f64;
    println!(
        "arm=ROW361_family_{label} codec={} dispatches={dispatch_count} bytes={total_timed_bytes} \
         median_ms={:.3} min_ms={:.3} max_ms={:.3} cov_pct={:.2} ms_samples={ms_samples:?} \
         median_gbps={:.2} mean_gbps={:.2} us_per_dispatch={us_per_dispatch:.2}",
        codec.name(),
        ms_stats.median,
        ms_stats.min,
        ms_stats.max,
        ms_stats.cov_pct,
        gbps_stats.median,
        gbps_stats.mean,
    );
}

#[test]
#[ignore = "synthesizes real quantized bytes for a whole decode token and needs a real metal device"]
fn whole_token_matvec_sequence_bare() {
    let device = MTLCreateSystemDefaultDevice().expect("a Metal device is available on this host");
    let queue = device.newCommandQueue().expect("device creates a command queue");

    // ROW 361: `ROW361_FAMILY=<name>` (`q`/`k`/`v`/`o`/`gate`/`up`/`down`/
    // `head`) restricts this whole-token construction to ONE family's
    // dispatches, amortized the same way llama's own `test-backend-ops
    // perf` amortizes (n_runs copies in one graph, one dispatch), so the
    // resulting per-dispatch number is comparable to ROW 360's llama
    // column rather than its isolated "ours" column. Additive: unset,
    // this function's behavior is byte-for-byte the same as before.
    if let Some(family_name) = std::env::var("ROW361_FAMILY").ok().filter(|value| !value.is_empty()) {
        run_row361_family_amortized(&device, &queue, &family_name);
        return;
    }

    type FamilyBuffer = (Retained<ProtocolObject<dyn MTLBuffer>>, Vec<usize>);
    let family_buffers: Vec<FamilyBuffer> =
        WHOLE_TOKEN_LAYER_FAMILIES
            .iter()
            .map(|family| {
                let row_bytes = ShapeCodec::Q4K.row_bytes(family.k);
                let tensor_bytes = family.rows * row_bytes;
                let bytes = synth_weight_bytes_parallel(
                    ShapeCodec::Q4K,
                    family.seed,
                    FFN_LAYERS,
                    family.k,
                    tensor_bytes,
                );
                let offsets: Vec<usize> = (0..FFN_LAYERS).map(|layer| layer * tensor_bytes).collect();
                (shared_buffer_from_bytes(&device, &bytes), offsets)
            })
            .collect();

    let head_row_bytes = ShapeCodec::Q6K.row_bytes(WHOLE_TOKEN_HEAD_K);
    let head_tensor_bytes = WHOLE_TOKEN_HEAD_ROWS * head_row_bytes;
    let head_bytes =
        synth_weight_bytes_parallel(ShapeCodec::Q6K, 900_000, 1, WHOLE_TOKEN_HEAD_K, head_tensor_bytes);
    let head_buffer = shared_buffer_from_bytes(&device, &head_bytes);

    let mut activation_lcg_4096 = Lcg(1);
    let activation_4096: Vec<f32> = (0..4096).map(|_| activation_lcg_4096.next_unit() * 4.0 - 2.0).collect();
    let activation_4096_bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(
            activation_4096.as_ptr().cast::<u8>(),
            std::mem::size_of_val(activation_4096.as_slice()),
        )
    };
    let activation_4096_buffer = shared_buffer_from_bytes(&device, activation_4096_bytes);

    let mut activation_lcg_14336 = Lcg(2);
    let activation_14336: Vec<f32> =
        (0..14_336).map(|_| activation_lcg_14336.next_unit() * 4.0 - 2.0).collect();
    let activation_14336_bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(
            activation_14336.as_ptr().cast::<u8>(),
            std::mem::size_of_val(activation_14336.as_slice()),
        )
    };
    let activation_14336_buffer = shared_buffer_from_bytes(&device, activation_14336_bytes);

    let uniform_4096 = uniform_u64(&device, (4096 / Q4K_BLOCK_ELEMENTS) as u64);
    let uniform_14336 = uniform_u64(&device, (14_336 / Q4K_BLOCK_ELEMENTS) as u64);

    let output = device
        .newBufferWithLength_options(WHOLE_TOKEN_HEAD_ROWS * size_of::<f32>(), MTLResourceOptions::StorageModeShared)
        .expect("device allocates the whole-token scratch output buffer");

    let total_timed_bytes: u64 = WHOLE_TOKEN_LAYER_FAMILIES
        .iter()
        .map(|family| {
            let row_bytes = ShapeCodec::Q4K.row_bytes(family.k);
            (family.rows * row_bytes * FFN_LAYERS) as u64
        })
        .sum::<u64>()
        + head_tensor_bytes as u64;

    println!(
        "=== ROW 339: whole decode token, {} dispatches (32 layers x 7 + head), {total_timed_bytes} bytes, \
         llama's own serial encoding, no hazard tracker/barrier/arena/plan cache ===",
        WHOLE_TOKEN_LAYER_FAMILIES.len() * FFN_LAYERS + 1
    );

    let q4k_shape_source_text = l3_shape_source();
    let q6k_shape_source_text = q6k_l3_shape_source();

    // ROW 354: `ROW354_EMITTED_ONLY=1` skips arms A-D (this ladder's own
    // hand-written `q4k_pair_dot`/`q6k_pair_dot` body at fixed dispatch
    // geometries) so an nsg sweep of arm E -- production's OWN emitted body,
    // whose `nsg` is a compile-time constant (`omega::sized::PACKED_ROW_NSG`,
    // `OMEGA_PACKED_ROW_NSG_WIDTH` per build) -- times only the one cell this
    // process was compiled with, keeping each of the sweep's separate
    // `cargo test` invocations to a single timed cell instead of five.
    let emitted_only = std::env::var_os("ROW354_EMITTED_ONLY").is_some();

    for arm in WHOLE_TOKEN_ARMS {
        if emitted_only {
            break;
        }
        let q4k_pipeline =
            compile_pipeline(&device, &q4k_shape_source_text, "q4k_matvec_l3_shape", arm.math);
        let q6k_pipeline =
            compile_pipeline(&device, &q6k_shape_source_text, "q6k_matvec_l3_shape", arm.math);
        let threadgroup_width = arm.simdgroups_per_tg * 32;

        let mut ops: Vec<BareDispatchOp<'_>> =
            Vec::with_capacity(WHOLE_TOKEN_LAYER_FAMILIES.len() * FFN_LAYERS + 1);
        for layer in 0..FFN_LAYERS {
            for (family, (weight_buffer, offsets)) in WHOLE_TOKEN_LAYER_FAMILIES.iter().zip(&family_buffers) {
                let total_simdgroups = family.rows / PACKED_ROWS_PER_GROUP;
                assert_eq!(
                    total_simdgroups % arm.simdgroups_per_tg,
                    0,
                    "{}: rows must divide evenly by simdgroups_per_tg={} for arm {}",
                    family.name,
                    arm.simdgroups_per_tg,
                    arm.name
                );
                let (activation, uniform) = if family.k == 4096 {
                    (&activation_4096_buffer, &uniform_4096)
                } else {
                    (&activation_14336_buffer, &uniform_14336)
                };
                ops.push(BareDispatchOp {
                    pipeline: &q4k_pipeline,
                    weight: weight_buffer,
                    weight_offset: offsets[layer],
                    activation,
                    uniform,
                    grid_threads: total_simdgroups * 32,
                    threadgroup_width,
                    weight_index: 0,
                    activation_index: 1,
                    output_index: 2,
                    uniform_index: 3,
                });
            }
        }
        let head_total_simdgroups = WHOLE_TOKEN_HEAD_ROWS / PACKED_ROWS_PER_GROUP;
        assert_eq!(
            head_total_simdgroups % arm.simdgroups_per_tg,
            0,
            "head: rows must divide evenly by simdgroups_per_tg={} for arm {}",
            arm.simdgroups_per_tg,
            arm.name
        );
        ops.push(BareDispatchOp {
            pipeline: &q6k_pipeline,
            weight: &head_buffer,
            weight_offset: 0,
            activation: &activation_4096_buffer,
            uniform: &uniform_4096,
            grid_threads: head_total_simdgroups * 32,
            threadgroup_width,
            weight_index: 0,
            activation_index: 1,
            output_index: 2,
            uniform_index: 3,
        });

        assert_eq!(ops.len(), WHOLE_TOKEN_LAYER_FAMILIES.len() * FFN_LAYERS + 1);

        let elapsed_samples = warmed_up_samples(|| time_bare_dispatch_sequence(&queue, &output, &ops));
        let ms_samples: Vec<f64> = elapsed_samples.iter().map(Duration::as_secs_f64).map(|s| s * 1e3).collect();
        let ms_stats = sample_stats(&ms_samples);
        let gbps_samples_vec = gbps_samples(&elapsed_samples, total_timed_bytes);
        let gbps_stats = sample_stats(&gbps_samples_vec);
        println!(
            "arm={} dispatches={} bytes={total_timed_bytes} median_ms={:.3} min_ms={:.3} max_ms={:.3} \
             cov_pct={:.2} ms_samples={ms_samples:?} median_gbps={:.2} mean_gbps={:.2}",
            arm.name,
            ops.len(),
            ms_stats.median,
            ms_stats.min,
            ms_stats.max,
            ms_stats.cov_pct,
            gbps_stats.median,
            gbps_stats.mean,
        );
    }

    // ROW 344: arm E is the same 225-dispatch bare sequence as arms A-D, but
    // every pipeline is PRODUCTION's own emitted kernel
    // ([`production_reduce_kernel`]) instead of this ladder's hand-written
    // `q4k_pair_dot`/`q6k_pair_dot` body -- the question ROW 339-343 never
    // answered: is the emitted body itself as fast as `pair_dot` bare, or is
    // production's ggml-port default (`metal-q4k-ggml-port`, in the default
    // `metal` feature list) the loss this whole-token gap has been carrying?
    // Fixed at `MTLMathMode::Relaxed`, matching arm C -- production has no
    // `nsg`/math knob a caller selects per dispatch the way arms A-D's
    // hand-compiled pipelines do; its own compiled shape is whatever
    // `omega::emit` renders today.
    println!(
        "=== ROW 344: whole decode token bare, PRODUCTION's own emitted kernel per shape \
         (arm E vs arm C's q4k_pair_dot/q6k_pair_dot body) ==="
    );

    let family_kernels: Vec<_> = WHOLE_TOKEN_LAYER_FAMILIES
        .iter()
        .map(|family| {
            production_reduce_kernel(&device, omega::PackedCodec::Q4K, family.rows as u32, family.k as u32)
        })
        .collect();
    let head_kernel =
        production_reduce_kernel(&device, omega::PackedCodec::Q6K, WHOLE_TOKEN_HEAD_ROWS as u32, WHOLE_TOKEN_HEAD_K as u32);

    let mut production_ops: Vec<BareDispatchOp<'_>> =
        Vec::with_capacity(WHOLE_TOKEN_LAYER_FAMILIES.len() * FFN_LAYERS + 1);
    for layer in 0..FFN_LAYERS {
        for ((family, (weight_buffer, offsets)), kernel) in
            WHOLE_TOKEN_LAYER_FAMILIES.iter().zip(&family_buffers).zip(&family_kernels)
        {
            let activation = if family.k == 4096 { &activation_4096_buffer } else { &activation_14336_buffer };
            production_ops.push(BareDispatchOp {
                pipeline: &kernel.pipeline,
                weight: weight_buffer,
                weight_offset: offsets[layer],
                activation,
                uniform: &kernel.uniform,
                grid_threads: kernel.grid_threads,
                threadgroup_width: kernel.threadgroup_width,
                weight_index: kernel.weight_index,
                activation_index: kernel.activation_index,
                output_index: kernel.output_index,
                uniform_index: kernel.uniform_index,
            });
        }
    }
    production_ops.push(BareDispatchOp {
        pipeline: &head_kernel.pipeline,
        weight: &head_buffer,
        weight_offset: 0,
        activation: &activation_4096_buffer,
        uniform: &head_kernel.uniform,
        grid_threads: head_kernel.grid_threads,
        threadgroup_width: head_kernel.threadgroup_width,
        weight_index: head_kernel.weight_index,
        activation_index: head_kernel.activation_index,
        output_index: head_kernel.output_index,
        uniform_index: head_kernel.uniform_index,
    });
    assert_eq!(production_ops.len(), WHOLE_TOKEN_LAYER_FAMILIES.len() * FFN_LAYERS + 1);

    let elapsed_samples =
        warmed_up_samples(|| time_bare_dispatch_sequence(&queue, &output, &production_ops));
    let ms_samples: Vec<f64> = elapsed_samples.iter().map(Duration::as_secs_f64).map(|s| s * 1e3).collect();
    let ms_stats = sample_stats(&ms_samples);
    let gbps_samples_vec = gbps_samples(&elapsed_samples, total_timed_bytes);
    let gbps_stats = sample_stats(&gbps_samples_vec);
    // ROW 354: the emitted body's own compiled `nsg` in the arm name, read
    // from the SAME `omega::sized::PACKED_ROW_NSG` constant
    // `production_reduce_kernel`'s emitted pipeline was built against, so
    // each of the sweep's separate `cargo test` runs self-documents which
    // cell it timed instead of relying on the caller to remember which
    // `OMEGA_PACKED_ROW_NSG_WIDTH` it built with.
    println!(
        "arm=E_production_emitted_body_nsg{}_relaxed dispatches={} bytes={total_timed_bytes} \
         median_ms={:.3} min_ms={:.3} max_ms={:.3} cov_pct={:.2} ms_samples={ms_samples:?} \
         median_gbps={:.2} mean_gbps={:.2}",
        omega::sized::PACKED_ROW_NSG,
        production_ops.len(),
        ms_stats.median,
        ms_stats.min,
        ms_stats.max,
        ms_stats.cov_pct,
        gbps_stats.median,
        gbps_stats.mean,
    );
}
