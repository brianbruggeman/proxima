//! The per-op GPU timing record every backend's op-timed execute returns.
//!
//! One contract, two producers: Metal times one command buffer per op
//! (`metal::execute_plan_op_timed`), CUDA brackets each launch with a pair of
//! events (`cuda_driver::CudaPlan::execute_named_op_timed`). The consumer side
//! (`proxima-model-interop`'s `BackendRuntime::evaluate_segment_op_timed`, the
//! N-pass reducer and the `whisper_op_timed` example) reads this one shape.
//! The record lives here, not under `metal`, so a CUDA-only host has it.

use alloc::string::String;
use alloc::vec::Vec;

use proxima_tensor::NodeId;

use crate::Codec;

/// One bound op's GPU-only execution time and the operand bytes it read.
/// `weight_name` is [`Op::name`](proxima_tensor::Op::name) off whichever operand is a named
/// block input (a model weight or the `ids`/`eps`/`rope_*`/KV-cache block
/// this program declares by name) -- `None` when every operand is itself a
/// computed node, which is the common case for a chained elementwise op.
#[derive(Debug, Clone)]
pub struct OpGpuTiming {
    pub node: NodeId,
    pub kind: &'static str,
    /// Full iteration extents of the bound operation that was timed.
    pub extents: Vec<u64>,
    /// Surviving axes for a reduction; empty for non-reductions.
    pub output_axes: Vec<u16>,
    /// This operand's TENSOR bytes -- `element_count(shape) * bytes_per_element`,
    /// where `bytes_per_element` is the operand's own dtype width for a plain
    /// buffer, or its `Codec::block_bytes`/block-elements ratio for a
    /// packed one. On CUDA, where every operand owns its buffer, it is the
    /// operand buffers' own byte length. Distinct from
    /// [`Self::bound_buffer_bytes`]: since a Metal checkpoint mapping upload binds
    /// ONE buffer spanning the whole mmap (`checkpoint_mapping_offset`), that
    /// buffer's own `length()` overstates every individual operand sharing it
    /// -- this field is what a per-tensor byte-share table needs instead.
    pub operand_bytes: u64,
    /// The device buffer's own `length()` this operand was bound against --
    /// the value `operand_bytes` used to report before it was corrected to
    /// the tensor's own byte count. Kept so a checkpoint-mapping-offset bind
    /// (one shared buffer, `bound_buffer_bytes` far larger than
    /// `operand_bytes`) stays observable rather than silently disappearing.
    pub bound_buffer_bytes: u64,
    pub gpu_ns: u64,
    pub weight_name: Option<String>,
    /// `bound.operands().len()` -- surfaced so a diagnostic caller can tell
    /// a two-operand (weight, activation) reduce, the shape
    /// `crate::msl::packed_row_block` requires, from a fused reduce whose
    /// element body absorbed a third operand (e.g. a gate/up product ahead
    /// of a down-projection), which disqualifies the row-blocked kernel via
    /// that same function's `quantized.len() != 2` check.
    pub operand_count: usize,
    /// The packed codec carried by this op's named operand, when present.
    pub packed_codec: Option<Codec>,
    /// The emitted packed-kernel body, kept separate from the broad op kind.
    /// Empty on a backend that does not classify its packed kernels (CUDA).
    pub packed_kernel_variant: &'static str,
    /// [`crate::msl::diagnose_packed_row_block`]'s own verdict on THIS op,
    /// against a REAL bound program rather than a synthetic symbolic one --
    /// `None` when the op is not a `Reduce { keep: Keep::Reduce, .. }` at
    /// all (the diagnosis does not apply), `Some("PASS")` when it took the
    /// row-blocked path, `Some(<rejection debug>)` otherwise. Always `None`
    /// on CUDA, which has no row-blocked classification.
    pub packed_row_block_rejection: Option<String>,
}
