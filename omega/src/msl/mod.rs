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
//! skip coordinate decoding; `elementwise_addressing_cache_token` records
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

use proxima_gguf::GgmlType;
pub use proxima_primitives::Codec;
use proxima_tensor::{
    BoundOp, BoundOpKind, ComposedBody, DType, Keep, Layout, Lookup, NodeId, NumericPolicy,
    NumericRewrite, ReduceInit, ScalarOp, StepArg, admit,
};
// `QuantizedBlock` itself is re-exported from the crate root only behind
// `std` (see `proxima_tensor::lib`'s own `#[cfg(feature = "std")]` on it),
// so `codec_from_quantized_block` -- the only user of it in this
// alloc-tier (`metal-core`) module -- stays gated the same way, further
// narrowed to the driver features that actually call it
// (`kernel_types_identity.rs`'s own gate on the function): an alloc-only
// build, or a std build with none of those drivers compiled in (e.g.
// x86_64-unknown-linux-gnu with default features), never needed this
// import and it would otherwise sit unused.
#[cfg(all(
    feature = "std",
    any(
        all(feature = "metal", target_os = "macos"),
        feature = "cuda-driver",
        feature = "wgpu-backend"
    )
))]
use proxima_tensor::QuantizedBlock;

use crate::error::EmitError;
use crate::grid::checked_product;
use crate::identity::{
    MetalOnlyExtras, body_token, init_token, keep_token, op_token, operand_codecs,
    reduce_epilogue_is_identity, signed_name_part,
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
#[cfg(feature = "metal-attn-split-decode")]
mod cached_attention_decode_split;
#[cfg(feature = "metal-attn-split-rows")]
mod cached_attention_row_tiled;
mod cached_softmax_weights_render;
mod selection_render;
#[macro_use]
mod elementwise_reduce_core;
#[macro_use]
mod packed_row_blocked_ggml;
#[macro_use]
mod tiled_gemm_cooperative_scan;
mod expert_grouped_gemm;
mod wide_grid;
pub(crate) use emit_and_classify::emit_inner_with_mma_selection;
#[cfg(feature = "metal-attn-variants")]
pub(crate) use emit_and_classify::validate_attention_variant_storage;
pub use emit_and_classify::*;
pub use kernel_types_identity::*;
pub use signature_tokens_prelude::CachedAttentionForm;
pub use signature_tokens_prelude::context_chunks_for;
pub(crate) use signature_tokens_prelude::*;
#[cfg(feature = "metal-attn-variants")]
pub use signature_tokens_prelude::{
    AttentionKvReuse, AttentionKvStorage, AttentionMmaPrecision, AttentionPrefetch,
    AttentionQueryParallelism, AttentionSimdTopology, AttentionSimdgroupCount, AttentionTileHeight,
    AttentionVariant,
};
// plain (non-pub) reexports: `render_cached_attention`/
// `render_cached_attention_merge` only need to reach `msl`'s own child
// modules (`emit_and_classify`'s and `tests`'s `use super::*`), which
// already see `msl`'s private items by the descendant-module visibility
// rule -- both are declared `pub(super)` on their own definitions, so a
// `pub(crate)` reexport would elevate past what they grant. Only
// `emit_cached_attention_merge` is `pub(crate)` on its own definition
// (`omega::metal`, a *sibling* of `msl` and gated identically, calls it via
// `crate::msl::emit_cached_attention_merge` outside this subtree), so only
// it is reexported at `pub(crate)`. `render_cached_attention_merge`'s own
// definition allows `cfg(any(test, all(feature = "metal", target_os =
// "macos")))`, but its ONLY external caller (outside its own defining file,
// where `emit_cached_attention_merge` calls it directly with no import
// needed) is `tests.rs`'s `#[cfg(test)]` module -- gating this `use` any
// wider leaves it unused in a plain (non-test) `metal`+macos lib build,
// since nothing else ever names it through `msl`'s namespace.
#[cfg(feature = "metal-attn-split-decode")]
use cached_attention_decode_split::render_cached_attention_decode_split;
#[cfg(test)]
use cached_attention_render::render_cached_attention;
use cached_attention_render::{
    cached_attention_kv_codec, render_cached_attention_with_mma_selection,
};
#[cfg(feature = "metal-attn-split-rows")]
use cached_attention_row_tiled::render_cached_attention_row_tiled;
// Same "plain reexport for descendant modules" shape as
// `render_cached_attention` immediately above -- `render_cached_softmax_
// weights` is `pub(super)` on its own definition (`cached_softmax_weights_
// render.rs`), and its only caller is `emit_and_classify::emit_inner`'s
// `use super::*`.
#[cfg(test)]
pub(crate) use cached_attention_render::emit_cached_attention_merge;
#[cfg(any(test, all(feature = "metal", target_os = "macos")))]
pub(crate) use cached_attention_render::emit_cached_attention_merge_with_schedule;
#[cfg(test)]
use cached_attention_render::render_cached_attention_merge;
use cached_softmax_weights_render::render_cached_softmax_weights;
pub(crate) use elementwise_reduce_core::*;
#[cfg(any(
    feature = "metal-grouped-gemm",
    all(feature = "metal", target_os = "macos")
))]
pub(crate) use expert_grouped_gemm::ROUTE_COMPACTION_MISMATCH_FAULT;
use expert_grouped_gemm::*;
#[cfg(any(
    all(test, feature = "metal-grouped-gemm"),
    all(feature = "metal", target_os = "macos")
))]
pub(crate) use expert_grouped_gemm::{route_compaction_key, route_prepass, route_prepass_active};
use packed_row_blocked_ggml::*;
use selection_render::render_top_fraction_select;
use tiled_gemm_cooperative_scan::*;
#[cfg(any(test, all(feature = "metal", target_os = "macos")))]
pub(crate) use wide_grid::fit_flat_width;
use wide_grid::*;
// `crate::metal::prepare_uniforms_pack` (a sibling of `msl`, not a
// descendant, so `pub(super)` there would not reach it) needs this to pack
// `CachedSoftmaxWeights::Uniforms::total_elements` as `attention_rows *
// width`, matching `grid_threads`'s own total exactly.
pub(crate) use tiled_gemm_cooperative_scan::wide_cooperative_reduce_width;
// Same "sibling needs a `pub(crate)` reexport" shape as
// `wide_cooperative_reduce_width` immediately above --
// `crate::metal::prepare_uniforms_pack` needs this to size and pack the
// `CachedSoftmaxWeights` uniforms blob identically to what `render_cached_
// softmax_weights` itself emits.
pub(crate) use kernel_types_identity::softmax_runtime_rows_override;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod attn_golden_tests;
#[cfg(all(test, feature = "metal-attn-split-rows"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod attn_rows_tests;
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod attn_split_tests;
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod cooperative_fold_tests;
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod serial_fold_index32_tests;
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(crate) mod tests;
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod twin_elementwise_tests;
