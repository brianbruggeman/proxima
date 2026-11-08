use alloc::collections::BTreeMap;
use alloc::vec::Vec;

pub use proxima_numeric::codec::Codec;
use proxima_tensor::{BoundOp, NodeId, ScalarOp};

/// What buffer index `n` in [`Kernel::bindings`] is for, in dispatch order:
/// index `0..operands.len()` are inputs, then one `Indices` buffer per
/// gathered operand (in operand order), then the output, then the uniforms
/// blob (extents/strides/bases for this dispatch — see the module doc), then
/// — only when the op gathers — the fault buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Binding {
    Input(NodeId),
    /// Codec-specific payload bytes for a mixed expert source. The matching
    /// [`Self::ExpertDescriptors`] binding selects the byte span and decoder
    /// for each routed expert index.
    ExpertPayloads(NodeId),
    /// Per-expert codec and byte-span records for a mixed expert source.
    /// Kept separate from [`Self::ExpertPayloads`] so the payloads remain
    /// borrowed mapped ranges rather than one concatenated staging
    /// allocation.
    ExpertDescriptors(NodeId),
    /// The `indices` buffer a gathered operand fetches from.
    Indices(NodeId),
    Output(NodeId),
    Uniforms,
    /// Present only when `gather_count` is nonzero: a `gather_count`-long
    /// zero-initialized `atomic_uint` array. The kernel `atomic_fetch_max`s
    /// an out-of-range fetched index (plus one, so zero means "no fault")
    /// into its gathered operand's slot; the driver reads this back after
    /// dispatch and turns a nonzero slot into the same
    /// `TensorError::GatherIndexOutOfRange` `cpu::evaluate` would report —
    /// see `push_gather_fetch`'s doc for how the check is emitted.
    Fault,
    /// `CachedAttention`'s cross-threadgroup key-split scratch — present
    /// only when `cached_attention_merge_needed` admits
    /// [`NumericRewrite::ContextSplitMerge`]. The split kernel WRITES it
    /// (this binding replaces `Binding::Output` in that kernel's own
    /// `bindings` list, since the split no longer writes the op's real
    /// output directly); the merge kernel READS it. Not `NodeId`-keyed —
    /// unlike every other binding, this buffer has no program node of its
    /// own, so the identity a hazard tracker needs comes from
    /// `crate::metal`'s own scratch-buffer pointer, resolved outside
    /// `device_buffers` (see that crate's `BufferArena` scratch slot).
    Scratch,
}

/// Every packed operand a bound program has, keyed by [`NodeId`] to its
/// codec — the single source of truth [`emit`] (via the `quantized` slice it
/// derives) and the Metal driver's `correct_packed_matmul_layouts` call both
/// need, generalizing the Q4_K-only `BTreeSet<NodeId>` this crate carried
/// before Q6_K support existed. `Codec` is `proxima_numeric::codec::Codec` — the
/// source-neutral packed-layout identity (see that type's own doc); this
/// crate's per-codec derivations (`cache_token`, `block_bytes`,
/// `supports_pair_dot`, `rows_per_simdgroup`, `from_quantized_block`) live as
/// free functions in `crate::identity` since `Codec` is foreign here and
/// cannot carry inherent methods (guiding-principles §20, no blanket impls).
pub type PackedOperands = BTreeMap<NodeId, Codec>;

pub fn is_cooperative_reduce_op(op: ScalarOp) -> bool {
    matches!(
        op,
        ScalarOp::Add | ScalarOp::Multiply | ScalarOp::Maximum | ScalarOp::Minimum
    )
}

pub fn reduction_dims(resolved: &BoundOp, output_axes: &[u16]) -> Vec<u16> {
    (0..resolved.extents.len() as u16)
        .filter(|dim| !output_axes.contains(dim))
        .collect()
}

const INLINE_REDUCTION_RANK: usize = 16;

/// [`reduction_dims`] without the heap allocation, for the per-dispatch
/// shape and cache-key paths that run once per encoded op: the axes live in
/// a stack array for any rank up to [`INLINE_REDUCTION_RANK`].
pub fn with_reduction_dims<Output>(
    resolved: &BoundOp,
    output_axes: &[u16],
    visit: impl FnOnce(&[u16]) -> Output,
) -> Output {
    let rank = resolved.extents.len();
    if rank > INLINE_REDUCTION_RANK {
        return visit(&reduction_dims(resolved, output_axes));
    }
    let mut axes = [0u16; INLINE_REDUCTION_RANK];
    let mut count = 0;
    for dim in (0..rank as u16).filter(|dim| !output_axes.contains(dim)) {
        axes[count] = dim;
        count += 1;
    }
    visit(&axes[..count])
}

pub fn gather_count(resolved: &BoundOp) -> usize {
    resolved
        .operands()
        .iter()
        .filter(|(_, _, gather)| gather.is_some())
        .count()
}

/// For each operand, `Some(slot)` if it gathers: `slot` is its position
/// among only the gathered operands, matching the order [`dispatch_bindings`]
/// appends `Indices` buffers and the order the `Uniforms` gather arrays are
/// packed in.
pub fn gather_slots(resolved: &BoundOp) -> Vec<Option<usize>> {
    let mut next = 0usize;
    resolved
        .operands()
        .iter()
        .map(|(_, _, gather)| {
            gather.as_ref().map(|_| {
                let slot = next;
                next += 1;
                slot
            })
        })
        .collect()
}

/// A route selected by the token axis has zero stride in contracted
/// dimensions, so one fetched expert index can be shared by all lanes. Any
/// nonzero stride would select different experts during the reduction and is
/// therefore kept on the serial path.
pub fn gather_is_reduction_invariant(resolved: &BoundOp, output_axes: &[u16]) -> bool {
    with_reduction_dims(resolved, output_axes, |reduce_dims| {
        resolved.operands().iter().all(|(_, _, lookup)| {
            lookup.as_ref().is_none_or(|lookup| {
                reduce_dims
                    .iter()
                    .all(|&dim| lookup.index_layout.stride(dim) == 0)
            })
        })
    })
}

/// The buffer list every backend binds: reads (including fused epilogue
/// operands), one `Indices` buffer per gathered operand, the output, the
/// uniforms blob, and the fault buffer when the op gathers.
pub fn dispatch_bindings(resolved: &BoundOp) -> Vec<Binding> {
    let mut bindings: Vec<Binding> = resolved
        .all_read_sources()
        .map(|(node, _, _)| Binding::Input(*node))
        .collect();
    for (_, _, gather) in resolved.operands() {
        if let Some(gather_access) = gather {
            bindings.push(Binding::Indices(gather_access.indices));
        }
    }
    bindings.push(Binding::Output(resolved.node));
    bindings.push(Binding::Uniforms);
    if gather_count(resolved) > 0 {
        bindings.push(Binding::Fault);
    }
    bindings
}

#[cfg(test)]
mod tests {
    use proxima_numeric::{NumericPolicy, NumericRewrite, admit};

    fn policy_from_bits(bits: u8) -> NumericPolicy {
        NumericPolicy::bit_exact()
            .with_contraction(bits & 0b0000_0001 != 0)
            .with_reassociation(bits & 0b0000_0010 != 0)
            .with_nan_assumptions(bits & 0b0000_0100 != 0)
            .with_signed_zero(bits & 0b0000_1000 != 0)
            .with_approx_functions(bits & 0b0001_0000 != 0)
            .with_epilogue_sources(bits & 0b0010_0000 != 0)
            .with_activation_quantization(bits & 0b0100_0000 != 0)
            .with_bf16_input_rounding(bits & 0b1000_0000 != 0)
    }

    #[test]
    fn reassociating_rewrites_are_admitted_exactly_when_reassociation_is_granted() {
        let rewrites = [
            NumericRewrite::TreeReduce,
            NumericRewrite::ContextChunkMerge,
            NumericRewrite::ContextSplitMerge,
        ];
        for bits in 0..=u8::MAX {
            let policy = policy_from_bits(bits);
            for rewrite in rewrites {
                assert_eq!(
                    admit(policy, rewrite).is_ok(),
                    policy.reassociation,
                    "{rewrite:?} under {policy:?}"
                );
            }
        }
    }

    #[test]
    fn cuda_build_policies_agree_between_admit_and_the_reassociation_flag() {
        let policies = [
            NumericPolicy::default(),
            NumericPolicy::bit_exact(),
            NumericPolicy::bit_exact().with_activation_quantization(true),
            NumericPolicy::llama_relaxed(),
            NumericPolicy::llama_relaxed().with_activation_quantization(true),
            NumericPolicy::fast(),
        ];
        for policy in policies {
            assert_eq!(
                admit(policy, NumericRewrite::TreeReduce).is_ok(),
                policy.reassociation,
                "{policy:?}"
            );
        }
    }
}
