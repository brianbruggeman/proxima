//! One structural + compile-option fingerprint for the kernel a `BoundOp`
//! would render — shared by every renderer ([`crate::msl`], [`crate::wgsl`],
//! [`crate::cuda`]) instead of each deriving its own.
//!
//! Before this module: `msl::kernel_cache_key` (structural fingerprint plus
//! the cooperative-reduce width, the packed row-block shape/stride tokens,
//! and — folded in separately at `metal::pipeline_for` — the math-mode
//! token), `wgsl::entry_name`, and `cuda::entry_name` each derived their own
//! notion of "which BoundOps may share a compiled kernel." Three
//! derivations of one fact drift independently: ROW 290 (`proxima-tensor/
//! docs/discipline.md`) found `msl::kernel_cache_key` missing the
//! cooperative-reduce width; main 7312713 found `wgsl::entry_name`/
//! `cuda::entry_name` missing the fused reduce epilogue; this census found a
//! THIRD live gap neither of those landings touched — `wgsl::entry_name`/
//! `cuda::entry_name` render a different body per packed codec
//! (`q4k_element` vs `q5k_element`, ..., `crate::wgsl`'s own `packed_
//! element_fn`/`crate::cuda`'s `packed_element_expr`) but never folded the
//! codec into the name at all, so two operands differing only in codec would
//! share one cached WGSL pipeline (`wgpu_driver::pipeline_for` keys on the
//! entry string alone).
//!
//! [`kernel_identity`] is the one place that now folds in the WHOLE union of
//! axes any renderer's body text can vary on: op kind, rank, operand count,
//! element body, reduce op/init/keep, the reduce's exact output-axis
//! sequence, a fused epilogue, gather presence, dtype width class, and
//! packed codec per operand — every renderer needs all of these. A renderer
//! that ALSO varies on something none of the others do ([`KernelLanguage::
//! Metal`]'s cooperative-reduce width, packed row-block shape, and math
//! mode) passes it through [`MetalOnlyExtras`] rather than this module
//! growing a language-specific branch for it — the mask is a documented
//! per-language subset of one shared output, never a second derivation.
//!
//! Mirrors [`crate::epilogue`]'s own precedent: the walk every renderer does
//! identically lives in ONE place, and what differs per language is a
//! parameter, never a second copy of the walk.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use proxima_tensor::{
    BoundOp, BoundOpKind, ComposedBody, DType, Keep, Layout, Lookup, NodeId, NumericPolicy,
    ReduceInit, ScalarOp, StepArg,
};

use crate::msl::{PackedCodec, PackedOperands};

/// Which renderer is asking [`kernel_identity`] for a fingerprint — selects
/// only the name prefix. Every other axis is computed the same way for
/// every language; see this module's doc for why a language tag, not three
/// copies, is how a per-renderer SUBSET of one shared fact is expressed
/// (Metal's own extra axes ride in [`MetalOnlyExtras`] instead).
// a build with only ONE of `metal-core`/`wgpu-backend`/`cuda` enabled
// constructs only that one variant -- each is genuinely reachable, just not
// from every feature combination this crate supports.
#[allow(
    dead_code,
    reason = "each variant is constructed by its own renderer, gated on that renderer's own feature"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KernelLanguage {
    Metal,
    Wgsl,
    Cuda,
}

impl KernelLanguage {
    const fn prefix(self) -> &'static str {
        match self {
            KernelLanguage::Metal => "omega",
            KernelLanguage::Wgsl => "omega_wgsl",
            KernelLanguage::Cuda => "omega_cuda",
        }
    }
}

/// The axes ONLY [`KernelLanguage::Metal`] varies its compiled kernel on —
/// every field is `None` (the [`Default`]) for `Wgsl`/`Cuda`, which never
/// render a cooperative-reduce width, a packed row-block body shape, or a
/// math-mode compile option. Each field's own producer stays exactly where
/// it always lived (`msl.rs`'s `tiled_gemm_threadgroup_width`/`packed_row_
/// block`/`metal::MathMode`) — this only carries their ALREADY-COMPUTED
/// values across the module boundary into the one function that renders
/// them into text, so `metal::pipeline_for`'s own `format!` fold (see its
/// doc) has nothing left to do.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MetalOnlyExtras {
    /// `tiled_gemm_threadgroup_width`'s return for this op — bakes literally
    /// into `render_reduce`'s lane-index/stride/tail-fold source text, so
    /// two ops agreeing on every other axis but picking a different width
    /// must never share a cache entry (ROW 290).
    pub cooperative_width: Option<u64>,
    /// The row-blocked/tiled-GEMM structural shape ('G'/'M'/'B'/'S') a
    /// Metal `Reduce` renders — always `Some` for Metal (defaulting to 'S'
    /// for anything not row-blocked), `None` for a non-Metal call.
    pub packed_row_block_shape: Option<char>,
    /// Whether the packed row-block's non-weight operand reads unit stride
    /// on the reduce axis — `Some` only when `packed_row_block` matched at
    /// all (`push_packed_row_blocked_body`'s stride-free specialization).
    pub packed_row_block_stride_is_one: Option<bool>,
    /// The single non-unit output axis `push_packed_row_group_bases` used
    /// for direct row addressing (ROW 350/351), when it found exactly one —
    /// `None` when the generic N-D coordinate decomposition was rendered
    /// instead (no `packed_row_block` match, a multi-row `M` block, or more
    /// than one non-unit output axis). The axis index, not just a bool: two
    /// bindings sharing rank/`output_axes` but disagreeing on WHICH axis is
    /// non-unit render different literals baked into the row-base source
    /// text and must never share a cache entry.
    pub packed_row_block_direct_axis: Option<u16>,
    /// [`numeric_policy_cache_token`] for the `Plan` compiling this op —
    /// folded in here instead of `MathMode::cache_token()` (what this field
    /// held before): `NumericPolicy` is a 4-rung ladder,
    /// `metal::numeric_policy_as_metal_math_mode` collapses two adjacent
    /// rungs onto the same `MathMode` (`BitExact` and
    /// `FusedNoReassociation` both compile `Safe`), so the `MathMode` token
    /// alone could not tell a `BitExact`-compiled kernel apart from a
    /// `FusedNoReassociation`-compiled one. The policy token is strictly
    /// finer than the math-mode token it replaces and subsumes it
    /// completely — every op below folds `numeric_policy` into its cache
    /// key ONLY through this field, never a separate math-mode token.
    pub numeric_policy_token: Option<char>,
}

/// `numeric_policy`'s single-character identity token — one letter per
/// rung on the `BitExact < FusedNoReassociation < ReassociationPermitted <
/// FastMath` ladder (`proxima_tensor::NumericPolicy`'s own doc), so two
/// `Plan`s agreeing on every structural field but compiled under different
/// policies never share a `PIPELINE_CACHE` entry ([`MetalOnlyExtras::
/// numeric_policy_token`]'s own doc explains why this replaced the coarser
/// `MathMode` token). `NumericPolicy` is `#[non_exhaustive]`, so the
/// wildcard arm folds any future rung above `FastMath` onto `'F'` — the
/// same "future rung compiles under the topmost mode" posture
/// `metal::numeric_policy_as_metal_math_mode` already takes.
#[must_use]
pub(crate) const fn numeric_policy_cache_token(policy: NumericPolicy) -> char {
    match policy {
        NumericPolicy::BitExact => 'B',
        NumericPolicy::FusedNoReassociation => 'N',
        NumericPolicy::ReassociationPermitted => 'R',
        _ => 'F',
    }
}

fn is_leaf(body: &ComposedBody) -> bool {
    body.steps.len() == 1
        && body.steps[0].args.iter().enumerate().all(
            |(index, arg)| matches!(arg, StepArg::Operand(operand) if *operand as usize == index),
        )
}

fn body_fingerprint(body: &ComposedBody) -> String {
    body.steps
        .iter()
        .map(|step| {
            let mut token = String::from(op_token(step.op));
            for arg in &step.args {
                match arg {
                    StepArg::Operand(index) => token.push_str(&format!("_o{index}")),
                    StepArg::Step(index) => token.push_str(&format!("_s{index}")),
                }
            }
            token
        })
        .collect::<Vec<_>>()
        .join("__")
}

/// A `ComposedBody`'s own name — the single leaf op's token when it is
/// nothing but its own operands in order, `fused_{fingerprint}` otherwise.
pub(crate) fn body_token(body: &ComposedBody) -> String {
    if is_leaf(body) {
        op_token(body.steps[0].op).into()
    } else {
        format!("fused_{}", body_fingerprint(body))
    }
}

pub(crate) fn op_token(op: ScalarOp) -> &'static str {
    match op {
        ScalarOp::Identity => "identity",
        ScalarOp::Add => "add",
        ScalarOp::Subtract => "subtract",
        ScalarOp::Multiply => "multiply",
        ScalarOp::Divide => "divide",
        ScalarOp::Maximum => "maximum",
        ScalarOp::Minimum => "minimum",
        ScalarOp::Negate => "negate",
        ScalarOp::Reciprocal => "reciprocal",
        ScalarOp::Exponential => "exponential",
        ScalarOp::Logarithm => "logarithm",
        ScalarOp::SquareRoot => "square_root",
        ScalarOp::Tanh => "tanh",
        ScalarOp::Erf => "erf",
        ScalarOp::Greater => "greater",
        ScalarOp::Equal => "equal",
        ScalarOp::Select => "select",
    }
}

pub(crate) fn keep_token(keep: Keep) -> &'static str {
    match keep {
        Keep::Reduce => "reduce",
        Keep::Scan => "scan",
    }
}

pub(crate) fn init_token(init: ReduceInit) -> &'static str {
    match init {
        ReduceInit::Zero => "zero",
        ReduceInit::One => "one",
        ReduceInit::NegativeInfinity => "negative_infinity",
        ReduceInit::PositiveInfinity => "positive_infinity",
        ReduceInit::FirstElement => "first_element",
    }
}

/// Whether a `Reduce`'s fused epilogue is the untouched identity default —
/// contributes nothing to [`kernel_identity`] when true, matching every
/// renderer's own "no fused epilogue anywhere names exactly what it always
/// did" posture.
pub(crate) fn reduce_epilogue_is_identity(
    body: &ComposedBody,
    operands: &[(NodeId, Layout, Option<Lookup>)],
) -> bool {
    operands.is_empty()
        && body.steps.len() == 1
        && body.steps[0].op == ScalarOp::Identity
        && body.steps[0].args == [StepArg::Operand(0)]
}

/// Per-operand packed codec, in operand order — `None` for an operand this
/// op never binds through `packed_operands` at all.
pub(crate) fn operand_codecs(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
) -> Vec<Option<PackedCodec>> {
    resolved
        .operands()
        .iter()
        .map(|(node, _, _)| packed_operands.get(node).copied())
        .collect()
}

/// `codec`'s single-character identity token — every renderer whose body
/// text branches on the codec (all three: `crate::msl`'s row-blocked/tiled
/// bodies, `crate::wgsl`'s `packed_element_fn`, `crate::cuda`'s `packed_
/// element_expr`) needs this in its cache identity, or two operands
/// differing only in codec can share one compiled kernel.
fn codec_token(codec: Option<PackedCodec>) -> char {
    match codec {
        Some(PackedCodec::Q3K) => '3',
        Some(PackedCodec::Q4K) => '4',
        Some(PackedCodec::Q5K) => '5',
        Some(PackedCodec::Q6K) => '6',
        Some(PackedCodec::Q8_0) => '8',
        Some(PackedCodec::Q4_0) => '0',
        Some(PackedCodec::Float16) => 'h',
        Some(PackedCodec::BFloat16) => 'b',
        None => 'f',
    }
}

pub(crate) fn signed_name_part(value: i64) -> String {
    if value < 0 {
        format!("n{}", value.unsigned_abs())
    } else {
        format!("p{value}")
    }
}

/// The one fingerprint every renderer's pipeline/name cache derives from —
/// see this module's doc for the union of axes folded in and why
/// [`MetalOnlyExtras`] carries the ones only [`KernelLanguage::Metal`]
/// varies on.
pub(crate) fn kernel_identity(
    language: KernelLanguage,
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
    metal: MetalOnlyExtras,
    numeric_policy: NumericPolicy,
) -> String {
    let rank = resolved.extents.len();
    let operand_count = resolved.operands().len();
    let prefix = language.prefix();
    let mut identity = match &resolved.kind {
        BoundOpKind::CachedAttention {
            query_rows,
            cached_key_rows,
            new_key_rows,
            kv_heads,
            query_groups,
            head_dim,
            scale,
            cached_lower_inclusive,
            new_upper_inclusive,
            ..
        } => {
            // `operand_count == 9` means the ninth operand carries the real
            // `new_upper_inclusive` at run time -- see `BoundOpKind::
            // CachedAttention`'s own doc -- so the identity names the
            // STRUCTURE ("dyn") rather than that filler value, which must
            // never appear to vary the key across calls whose real bound
            // differs.
            let upper_token = if operand_count == 9 {
                String::from("dyn")
            } else {
                signed_name_part(*new_upper_inclusive)
            };
            // `context_chunks` is already a deterministic function of
            // `cached_key_rows + new_key_rows` (`crate::msl::
            // context_chunks_for`), both already folded into this
            // identity above -- naming it explicitly here means a future
            // change to the sizing config's divisor/cap still shows up as
            // a distinct cache key rather than silently reusing a pipeline
            // compiled for the wrong chunk count.
            let context_chunks =
                crate::msl::context_chunks_for(*cached_key_rows + *new_key_rows, numeric_policy);
            format!(
                "{prefix}_cached_attention_q{query_rows}_c{cached_key_rows}_n{new_key_rows}_h{kv_heads}_g{query_groups}_d{head_dim}_s{:08x}_l{}_u{upper_token}_x{context_chunks}",
                scale.to_bits(),
                signed_name_part(*cached_lower_inclusive),
            )
        }
        BoundOpKind::Elementwise { .. } => {
            let body = body_token(resolved.element_body());
            format!("{prefix}_elementwise_r{rank}_n{operand_count}_{body}")
        }
        BoundOpKind::Reduce {
            reduce_op,
            init,
            keep,
            output_axes,
            epilogue_body,
            epilogue_operands,
            ..
        } => {
            let body = body_token(resolved.element_body());
            let kind = keep_token(*keep);
            let reduce_body = op_token(*reduce_op);
            let init = init_token(*init);
            // The exact ORDERED axis sequence, not merely its length: two
            // folds sharing every other axis here but keeping a DIFFERENT
            // axis set (or the same set in a different order) still emit
            // different source (`render_reduce`/`render_reduce_cooperative`
            // bake the literal axis index into the uniform-slot addressing).
            let axes = output_axes
                .iter()
                .map(u16::to_string)
                .collect::<Vec<_>>()
                .join("_");
            let epilogue = if reduce_epilogue_is_identity(epilogue_body, epilogue_operands) {
                String::new()
            } else {
                format!(
                    "_epi{}_{}",
                    epilogue_operands.len(),
                    body_token(epilogue_body)
                )
            };
            format!(
                "{prefix}_{kind}_r{rank}_ax{axes}_n{operand_count}_{body}_{reduce_body}_{init}{epilogue}"
            )
        }
        BoundOpKind::Iota => format!("{prefix}_iota_r{rank}"),
        // the literal is baked into the source, so it has to be part of the
        // identity too -- otherwise two constants of the same rank would
        // share one cached kernel and the second would run the first one's
        // value. Raw bits, not the decimal, so the name is exact and
        // identifier-safe.
        BoundOpKind::Constant { value } => {
            format!("{prefix}_constant_r{rank}_v{:08x}", value.to_bits())
        }
    };

    let gather_bits: String = resolved
        .operands()
        .iter()
        .map(|(_, _, gather)| if gather.is_some() { '1' } else { '0' })
        .collect();
    if gather_bits.contains('1') {
        identity.push_str("_g");
        identity.push_str(&gather_bits);
    }

    // `type_token`'s own "half"/"float" (or "f16"/"f32", "__half"/"float")
    // split -- every dtype every renderer accepts collapses to one of these
    // two declarations, and only `DType::Float16` ever takes the narrow one
    // (each renderer's own `type_token` match, byte-for-byte the same
    // partition).
    identity.push_str(if resolved.dtype == DType::Float16 {
        "_half"
    } else {
        "_wide"
    });

    let quantized = operand_codecs(resolved, packed_operands);
    if quantized.iter().any(Option::is_some) {
        identity.push_str("_c");
        for codec in &quantized {
            identity.push(codec_token(*codec));
        }
    }

    if let Some(shape) = metal.packed_row_block_shape {
        identity.push(shape);
    }
    if let Some(stride_is_one) = metal.packed_row_block_stride_is_one {
        identity.push(if stride_is_one { '1' } else { 'N' });
    }
    if let Some(axis) = metal.packed_row_block_direct_axis {
        identity.push_str("_da");
        identity.push_str(&axis.to_string());
    }
    if let Some(width) = metal.cooperative_width {
        identity.push_str("_w");
        identity.push_str(&width.to_string());
    }
    if let Some(token) = metal.numeric_policy_token {
        identity.push(token);
    }

    identity
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod numeric_policy_cache_token_tests {
    //! [`super::MetalOnlyExtras::numeric_policy_token`]'s cache-key fold
    //! depends on exactly one fact: the 4 [`NumericPolicy`] rungs never
    //! collide on their token, so two `Plan`s agreeing on every structural
    //! field [`super::kernel_identity`] checks still resolve to distinct
    //! `PIPELINE_CACHE` entries when they disagree on numeric policy --
    //! this replaces `metal::math_mode_cache_key_tests`, which proved the
    //! same fact for the coarser, now-removed `MathMode` token.

    use proxima_tensor::NumericPolicy;

    use super::numeric_policy_cache_token;

    #[test]
    fn all_four_rungs_produce_distinct_tokens() {
        let tokens = [
            numeric_policy_cache_token(NumericPolicy::BitExact),
            numeric_policy_cache_token(NumericPolicy::FusedNoReassociation),
            numeric_policy_cache_token(NumericPolicy::ReassociationPermitted),
            numeric_policy_cache_token(NumericPolicy::FastMath),
        ];
        for (left_index, left_token) in tokens.iter().enumerate() {
            for (right_index, right_token) in tokens.iter().enumerate() {
                if left_index != right_index {
                    assert_ne!(
                        left_token, right_token,
                        "NumericPolicy rungs {left_index} and {right_index} must never \
                         share a cache token, or two Plans compiled under different \
                         policies could share one PIPELINE_CACHE entry"
                    );
                }
            }
        }
    }
}
