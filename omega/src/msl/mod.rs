//! Metal Shading Language kernel emission.
//!
//! [`emit`] turns one lowered [`BoundOp`] into one [`Kernel`]: MSL source text,
//! an entry name, the buffer-index -> data [`Binding`] list a driver needs to
//! set up a dispatch, and the thread-count [`GridSpec`] for this invocation.
//!
//! # Runtime uniforms, not baked constants
//!
//! A `BoundOp` node's extents and strides are read out of a `constant
//! Uniforms&` buffer at kernel runtime — never spliced into the source text
//! as literal numbers. What *does* vary the source is the node's STRUCTURE:
//! rank (operand and output coordinate arity), operand count, which
//! [`ScalarOp`]s the body and (if present) the reduction use, and whether a
//! reduction is present at all and which [`Keep`] it is. Concrete extent and
//! stride values remain uniforms. Elementwise kernels additionally specialize
//! on the stride layout's dense/strided addressing class so dense operands can
//! skip coordinate decoding; [`elementwise_addressing_cache_token`] records
//! that class in the pipeline identity. This makes a kernel cacheable by the
//! exact source it emits rather than by node identity.
//!
//! # Execution model (v1: correctness parity with `cpu.rs`, not peak speed)
//!
//! - **Elementwise** (no reduction): one thread per output element. A
//!   thread's linear id decodes into a coordinate via the same row-major
//!   div/mod chain `cpu::unflatten` uses, each operand's read offset is
//!   `base + sum(coord[d] * stride[d])`, and the body writes directly to the
//!   dense output at its own linear id — matching `cpu::run_elementwise`.
//! - **Fused fold, `Keep::Reduce`** (reduce): one thread per OUTPUT element
//!   (matmul is one thread per `(i, j)`), with a serial loop over the
//!   reduction dims inside the kernel. `ReduceInit` seeding — including
//!   `FirstElement`'s seed-on-first-step behavior — matches
//!   `cpu::run_reduce` exactly: the accumulator is seeded from the *first*
//!   reduction step's value rather than combined into an `init` constant.
//! - **`Keep::Scan`** (scan): one thread per non-folded coordinate line,
//!   serial along the folded (innermost) dim, writing every prefix through
//!   the output strides — matching `cpu::run_scan`.
//!
//! Parity extends to the sad path: `cpu.rs` returns
//! `TensorError::GatherIndexOutOfRange` for a fetched index outside
//! `[0, extent)` rather than clamping it, and a gather kernel here agrees —
//! it clamps for memory safety (a GPU kernel cannot propagate a `Result`)
//! but also records the fault into the `Fault` buffer `crate::metal` reads
//! back after dispatch and turns into the identical error. See
//! `push_gather_fetch`'s doc for where the check is emitted.
//!
//! # dtype
//!
//! `BoundOp` carries its own element type ([`proxima_tensor::BoundOp::dtype`],
//! read straight from the [`proxima_tensor::Op`] it was built from). Every
//! buffer/scratch/accumulator declaration this module renders is spelled
//! from `type_token` rather than hardcoding `float`, so a `Float16` node
//! emits a kernel of `half` declarations while a `Float32` node emits the
//! same `float` kernel this module always has. The *op logic* — which
//! `ScalarOp` token, which reduction init, how a body's steps chain — never
//! consults dtype at all: `op_token`, `scalar_op_expr`, `init_token`,
//! `fold_init_tokens` stay total over their enums exactly as before, and
//! only the declaration spelling varies. `cpu.rs`'s own evaluator remains
//! f32-only (`cpu::reject_non_float32`) — it is the reference oracle, not
//! this crate's dtype ceiling. `omega::execute` runs its own, narrower
//! upstream gate (`Float32` or `Float16` only) before a `BoundOp` ever
//! reaches [`emit`].

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

pub use proxima_primitives::Codec;
use proxima_gguf::GgmlType;
use proxima_tensor::{
    BoundOp, BoundOpKind, ComposedBody, DType, Keep, Layout, Lookup, NodeId, NumericPolicy,
    NumericRewrite, ReduceInit, ScalarOp, StepArg, admit,
};
// `QuantizedBlock` itself is re-exported from the crate root only behind
// `std` (see `proxima_tensor::lib`'s own `#[cfg(feature = "std")]` on it),
// so `codec_from_quantized_block` -- the only user of it in this
// alloc-tier (`metal-core`) module -- stays gated the same way; an
// alloc-only build never needed this method before and does not need it now.
#[cfg(feature = "std")]
use proxima_tensor::QuantizedBlock;

use crate::error::EmitError;
use crate::identity::{
    body_token, init_token, keep_token, op_token, operand_codecs, reduce_epilogue_is_identity,
    signed_name_part,
};
#[cfg(all(
    any(feature = "metal-packed-row-nsg2", feature = "metal-q4k-ggml-port"),
    not(feature = "metal-q4k-split-k")
))]
use crate::sized::PACKED_ROW_NSG;
use crate::sized::SIMD_WIDTH;


#[macro_use]
mod kernel_types_identity;
#[macro_use]
mod emit_and_classify;
#[macro_use]
mod signature_tokens_prelude;
#[macro_use]
mod cached_attention_render;
#[macro_use]
mod elementwise_reduce_core;
#[macro_use]
mod packed_row_blocked_ggml;
#[macro_use]
mod tiled_gemm_cooperative_scan;
pub use kernel_types_identity::*;
pub use emit_and_classify::*;
pub(crate) use signature_tokens_prelude::*;
pub(crate) use cached_attention_render::*;
pub(crate) use elementwise_reduce_core::*;
use packed_row_blocked_ggml::*;
use tiled_gemm_cooperative_scan::*;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
