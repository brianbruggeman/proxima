use super::*;

/// Rewrites a packed matmul weight operand's [`Layout`] from `layout_of`'s
/// default -- row-major over the operand's own DECLARED axis order -- to the
/// layout its packed bytes actually have on disk.
///
/// `layout_of` has no way to get this right on its own: it sees only the
/// operand's declared shape, the axis order every OTHER consumer of that
/// node agrees the buffer is stored in. For a plain `f32` operand that
/// agreement is real, because the buffer was transposed at bind time to
/// match it (`proxima-model-interop::bind_matmul_weight`'s `F32` fallback,
/// `transpose_out_in_to_in_out`). A packed `Q4_K`/`Q5_K`/`Q6_K` weight is the
/// one case that cannot be transposed to match: a k-quant super-block spans
/// 256 contiguous elements of the contraction axis, so transposing it would
/// mean dequantizing first -- defeating the entire reason to keep it packed.
/// So its declared shape and its physical bytes disagree, and the `Layout`
/// must be rebuilt to describe the bytes, not the declaration.
///
/// GGUF's own on-disk convention for any 2-D weight is `[out_dim, in_dim]`
/// row-major (`out_dim` rows, each a contiguous run of `in_dim` elements) --
/// true of a packed operand regardless of how many logical axes either side
/// is split into on the consuming einsum (`wq`'s `heads`/`head_dim` split is
/// still one flat `embedding x (heads*head_dim)` buffer underneath).
/// `output_axes` on a bound reduce already names which of `extents`'s
/// iteration axes are the "out" side; the complement is "in". Within each
/// side, relative axis order is preserved from the declared shape -- only
/// which side sits inside (contiguous) and which sits outside flips.
///
/// A no-op for any operand not in `packed_operands`, and for any `BoundOp`
/// that is not a `Reduce` (a packed weight only ever reaches this crate as
/// one operand of a `Multiply`-then-`Add` fold -- see
/// `proxima-model-interop::bind_matmul_weight`'s own doc).
pub fn correct_packed_matmul_layouts(resolved: &mut [BoundOp], packed_operands: &BTreeSet<NodeId>) {
    for bound in resolved.iter_mut() {
        let extents = bound.extents.clone();
        let BoundOpKind::Reduce {
            operands,
            output_axes,
            ..
        } = &mut bound.kind
        else {
            continue;
        };
        for (node, layout, _lookup) in operands.iter_mut() {
            if packed_operands.contains(node) {
                *layout = native_packed_layout(&extents, output_axes.as_slice(), layout);
            }
        }
    }
}

/// The stride computation [`correct_packed_matmul_layouts`] applies per
/// operand: `extents`/`output_axes` come from the containing `Reduce`, and
/// `declared` is `layout_of`'s original (wrong-for-packed) `Layout`, read
/// only for its `base` and for which axes it left at stride 0 (a batch axis
/// this operand broadcasts across, e.g. sequence position -- must stay
/// broadcast rather than gain a stride from this reconstruction).
pub(super) fn native_packed_layout(
    extents: &[u64],
    output_axes: &[u16],
    declared: &Layout,
) -> Layout {
    let rank = extents.len();
    let mut strides = SmallVec::<[i64; MAX_INLINE_RANK]>::from_elem(0, rank);

    let mut in_dim = 1i64;
    for axis in 0..rank as u16 {
        if !output_axes.contains(&axis) {
            in_dim *= extents[axis as usize] as i64;
        }
    }

    // the reduction ("in") axes: innermost group, relative order preserved,
    // the LAST one contiguous -- exactly `row_major_strides` restricted to
    // this axis subset.
    let mut accumulator = 1i64;
    for axis in (0..rank as u16).rev() {
        if output_axes.contains(&axis) {
            continue;
        }
        strides[axis as usize] = accumulator;
        accumulator *= extents[axis as usize] as i64;
    }

    // the output axes: outermost group, relative order preserved, scaled by
    // the whole reduction group's flat width since it sits inside them.
    let mut accumulator = in_dim;
    for axis in output_axes.iter().rev() {
        strides[*axis as usize] = accumulator;
        accumulator *= extents[*axis as usize] as i64;
    }

    for axis in 0..rank {
        if declared.stride(axis as u16) == 0 {
            strides[axis] = 0;
        }
    }

    Layout {
        base: declared.base,
        strides,
    }
}

/// Batch driver: computes liveness once, then streams every expression
/// through a fresh [`BoundOpBuilder`], flushing whatever remains held at the end.
/// Every node `resolved` physically reads, straight off [`BoundOp::operands()`]
/// plus each gathered operand's own [`Lookup::indices`] — the same walk
/// [`crate::cpu`]'s own execution-time dead-node analysis performs, relocated
/// here so a GPU backend (which has no persistent arena to skip a slot
/// inside) can reuse it too, via [`prune_dead`] below.
pub(super) fn consumed_by_resolved_nodes(resolved: &[BoundOp]) -> BTreeSet<NodeId> {
    let mut consumed = BTreeSet::new();
    for computed in resolved {
        for (operand, _layout, lookup) in computed.all_read_sources() {
            consumed.insert(*operand);
            if let Some(lookup) = lookup {
                consumed.insert(lookup.indices);
            }
        }
        if let BoundOpKind::Reduce {
            out_scatter: Some(lookup),
            ..
        } = &computed.kind
        {
            consumed.insert(lookup.indices);
        }
    }
    consumed
}

/// Every `resolved` node neither consumed by another resolved node's own
/// operands nor named in `effective_outputs` — dead weight [`bind`]'s own
/// fusion can leave behind (`eliminate_identity_multiply` dropping a
/// [`BoundOpKind::Constant`] from a fused body once its last reader absorbed
/// it is one source; a fused-away [`BoundOpKind::Elementwise`] chain is
/// another). [`crate::cpu::StaticArena`] computes this same set today purely
/// to build its own execution-time skip list — see that type's own `dead`
/// field doc — which hides a real cost from every OTHER backend: a driver
/// with no persistent arena (every GPU backend today) has no skip list to
/// consult, so it dispatches a kernel for a node this function would already
/// tell it nobody reads.
#[must_use]
pub fn dead_resolved_nodes(resolved: &[BoundOp], effective_outputs: &[NodeId]) -> BTreeSet<NodeId> {
    let consumed = consumed_by_resolved_nodes(resolved);
    let dead: BTreeSet<NodeId> = resolved
        .iter()
        .map(|computed| computed.node)
        .filter(|node| !consumed.contains(node) && !effective_outputs.contains(node))
        .collect();
    #[cfg(feature = "instrument")]
    for node in &dead {
        debug!(
            node = node.0,
            kind = "dead_resolved_node",
            decision = "dead",
            "resolved node has zero consumers and is not a requested output"
        );
    }
    dead
}

/// Drops every [`dead_resolved_nodes`] entry from `resolved` — the one
/// GPU-facing counterpart [`crate::cpu::StaticArena`]'s own skip-at-execution
/// trick has no analogue for. A stateless driver (Metal/CUDA/wgpu today) has
/// no persistent arena to skip a slot inside between calls, so the only way
/// to avoid dispatching a dead node's kernel is to never hand it to the
/// driver's own dispatch list at all. A no-op (identity on `resolved`,
/// zero-cost when nothing is dead) unless [`dead_resolved_nodes`] finds
/// something to drop.
#[must_use]
pub fn prune_dead(resolved: Vec<BoundOp>, effective_outputs: &[NodeId]) -> Vec<BoundOp> {
    let dead = dead_resolved_nodes(&resolved, effective_outputs);
    if dead.is_empty() {
        return resolved;
    }
    resolved
        .into_iter()
        .filter(|computed| !dead.contains(&computed.node))
        .collect()
}

#[cfg(feature = "cached-attention-streaming")]
pub(super) fn elementwise_operands(
    program: &[Op],
    node: NodeId,
    body: ScalarOp,
) -> Option<&[(NodeId, IndexMap)]> {
    match program.get(node.0 as usize)? {
        Op::Elementwise {
            body: actual_body,
            operands,
            ..
        } if *actual_body == body => Some(operands),
        _ => None,
    }
}

#[cfg(feature = "cached-attention-streaming")]
pub(super) fn binary_elementwise(
    program: &[Op],
    node: NodeId,
    body: ScalarOp,
) -> Option<[NodeId; 2]> {
    let operands = elementwise_operands(program, node, body)?;
    let [(left, _), (right, _)] = operands else {
        return None;
    };
    Some([*left, *right])
}

#[cfg(feature = "cached-attention-streaming")]
pub(super) fn unary_elementwise(program: &[Op], node: NodeId, body: ScalarOp) -> Option<NodeId> {
    let operands = elementwise_operands(program, node, body)?;
    let [(source, _)] = operands else {
        return None;
    };
    Some(*source)
}

#[cfg(feature = "cached-attention-streaming")]
pub(super) fn reduced_source(
    program: &[Op],
    node: NodeId,
    body: ScalarOp,
    init: ReduceInit,
) -> Option<NodeId> {
    match program.get(node.0 as usize)? {
        Op::Reduce(reduce)
            if reduce.body == body && reduce.init == init && reduce.keep == Keep::Reduce =>
        {
            Some(reduce.operand)
        }
        _ => None,
    }
}

#[cfg(feature = "cached-attention-streaming")]
pub(super) fn constant_value(program: &[Op], node: NodeId) -> Option<f32> {
    match program.get(node.0 as usize)? {
        Op::Constant { value, .. } => Some(*value),
        _ => None,
    }
}

#[cfg(feature = "cached-attention-streaming")]
pub(super) fn decode_rotary_terms(
    program: &[Op],
    terms: [NodeId; 2],
) -> Option<(NodeId, NodeId, NodeId, NodeId)> {
    let even_product = reduced_source(program, terms[0], ScalarOp::Add, ReduceInit::Zero)?;
    let odd_product = reduced_source(program, terms[1], ScalarOp::Add, ReduceInit::Zero)?;
    let even_operands = binary_elementwise(program, even_product, ScalarOp::Multiply)?;
    let odd_operands = binary_elementwise(program, odd_product, ScalarOp::Multiply)?;
    Some((
        even_operands[0],
        odd_operands[0],
        even_operands[1],
        odd_operands[1],
    ))
}

/// One un-rotated pass-plane term (qwen35's `score_cached_pass`/
/// `score_new_pass`, `spec.rs:4910-4927,5035-5049`): a bare
/// `reduced(query_pass_grouped * key_pass)`, no even/odd split because the
/// pass plane is never rotated. Returns `(query_pass_grouped, key_pass)`.
#[cfg(feature = "cached-attention-streaming")]
pub(super) fn decode_pass_term(program: &[Op], node: NodeId) -> Option<(NodeId, NodeId)> {
    let pass_product = reduced_source(program, node, ScalarOp::Add, ReduceInit::Zero)?;
    let pass_operands = binary_elementwise(program, pass_product, ScalarOp::Multiply)?;
    Some((pass_operands[0], pass_operands[1]))
}

/// `score = Multiply(Add(rotary_sum, pass_sum), scale)` when a partial-rotary
/// pass plane is present (qwen35's chain, `spec.rs`'s own `score_cached`/
/// `score_new`), `score = Multiply(Add(even, odd), scale)` otherwise (every
/// other caller today, `rotary_dim == head_dim`). Both shapes share the outer
/// `Multiply`-by-`scale`; only the sum operand's own shape differs, so this
/// tries the nested three-term interpretation first and falls back to the
/// flat two-term one. Returns
/// `(query_even_grouped, query_odd_grouped, key_even, key_odd, pass)`, where
/// `pass` is `Some((query_pass_grouped, key_pass))` only for the nested shape.
#[cfg(feature = "cached-attention-streaming")]
pub(super) fn attention_score_sources(
    program: &[Op],
    score: NodeId,
    scale: NodeId,
) -> Option<AttentionScoreSources> {
    let scaled = binary_elementwise(program, score, ScalarOp::Multiply)?;
    if constant_value(program, scaled[1]) != constant_value(program, scale)
        || constant_value(program, scaled[1]).is_none()
    {
        return None;
    }
    let outer = binary_elementwise(program, scaled[0], ScalarOp::Add)?;
    if let Some(rotary_terms) = binary_elementwise(program, outer[0], ScalarOp::Add)
        && let Some(rotary) = decode_rotary_terms(program, rotary_terms)
        && let Some(pass) = decode_pass_term(program, outer[1])
    {
        return Some((rotary.0, rotary.1, rotary.2, rotary.3, Some(pass)));
    }
    let rotary = decode_rotary_terms(program, outer)?;
    Some((rotary.0, rotary.1, rotary.2, rotary.3, None))
}

/// `true` when `node` is exactly [`Op::Iota`] -- the raw key/query index a
/// causal or padding mask compares against.
#[cfg(feature = "cached-attention-streaming")]
pub(super) fn is_iota(program: &[Op], node: NodeId) -> bool {
    matches!(program.get(node.0 as usize), Some(Op::Iota { .. }))
}

/// The `cached_len` bound a padding predicate excludes rows at-or-past, when
/// `node` is exactly `Greater(Iota, Subtract(cached_len, one))` -- `x > n - 1`
/// excludes exactly `x >= n`, and qwen35's own builder (`spec.rs:4973-4987`,
/// `is_cached_padding`) emits precisely this shape. Anything else returns
/// `None` -- the caller declines the fusion rather than guessing at an
/// unfamiliar predicate.
#[cfg(feature = "cached-attention-streaming")]
pub(super) fn cached_len_padding_bound(program: &[Op], node: NodeId) -> Option<NodeId> {
    let operands = binary_elementwise(program, node, ScalarOp::Greater)?;
    if !is_iota(program, operands[0]) {
        return None;
    }
    let shifted = binary_elementwise(program, operands[1], ScalarOp::Subtract)?;
    (constant_value(program, shifted[1]) == Some(1.0)).then_some(shifted[0])
}

/// Walks past qwen35's own padding mask (`spec.rs:4979-4997`,
/// `is_cached_padding` selecting `-inf` for `key_index >= cached_len`) to the
/// unmasked scaled score underneath, returning `None` (decline the fusion)
/// unless `node` is exactly `Select(padding_predicate, -inf, inner)` AND the
/// predicate's own bound is the SAME `cached_len` leaf the fused op's runtime
/// `cached_key_rows` clip already reads (`cpu.rs:6944-6974`) -- that clip
/// excludes exactly the rows this mask would have scored `-inf`, which is
/// what makes dropping the mask node sound rather than a silent behavior
/// change.
#[cfg(feature = "cached-attention-streaming")]
pub(super) fn unwrap_cached_padding_select(
    program: &[Op],
    node: NodeId,
    cached_len: Option<NodeId>,
) -> Option<NodeId> {
    let operands = elementwise_operands(program, node, ScalarOp::Select)?;
    let [(predicate, _), (negative_infinity, _), (inner, _)] = operands else {
        return None;
    };
    if constant_value(program, *negative_infinity) != Some(f32::NEG_INFINITY) {
        return None;
    }
    let bound = cached_len_padding_bound(program, *predicate)?;
    (Some(bound) == cached_len).then_some(*inner)
}

#[cfg(feature = "cached-attention-streaming")]
pub(super) fn is_exact_causal_mask(program: &[Op], node: NodeId) -> bool {
    let Some(operands) = elementwise_operands(program, node, ScalarOp::Greater) else {
        return false;
    };
    let [(key, key_map), (query, query_map)] = operands else {
        return false;
    };
    if *key_map != IndexMap::Affine(map::projection(2, &[1]))
        || *query_map != IndexMap::Affine(map::projection(2, &[0]))
    {
        return false;
    }
    matches!(
        (program.get(key.0 as usize), program.get(query.0 as usize)),
        (Some(Op::Iota { .. }), Some(Op::Iota { .. }),)
    )
}

/// [`is_exact_causal_mask`]'s counterpart for
/// [`crate::spec::causal_mask_merged`]'s shape: the key side is still a bare
/// `Iota`, but the query side is `query_index + cached_len` (an
/// [`ScalarOp::Add`]) rather than a bare `Iota`, because a single-range
/// query at local position `s` sits at absolute position `cached_len + s`
/// once its own new keys are folded into the one merged range. `cached_len`
/// itself is a per-call [`crate::op::Op::Input`] (`causal_mask_merged`'s own
/// doc), never structurally checked here — only that the query side is a
/// shift of an `Iota`, which is what makes the mask exact causal rather than
/// an arbitrary comparison.
/// Returns the `cached_len` [`Op::Input`] node the mask's query side shifts
/// an `Iota` by, when `node` is exactly
/// [`crate::spec::causal_mask_merged`]'s shape — `None` for anything else.
/// The caller needs this NodeId, not just a bool: `cached_len` is a per-call
/// runtime scalar (see this function's own doc below), and the fused
/// [`BoundOpKind::CachedAttention`] this feeds must read the band bound from
/// that scalar at execution time rather than baking a value derived from
/// bound EXTENTS, which drifts from the true `cached_len` whenever the KV
/// extent is padded past the merged length (`kv-capacity-bucket`).
#[cfg(feature = "cached-attention-streaming")]
pub(super) fn exact_merged_causal_mask_cached_len(program: &[Op], node: NodeId) -> Option<NodeId> {
    let operands = elementwise_operands(program, node, ScalarOp::Greater)?;
    let [(key, key_map), (query_absolute, query_map)] = operands else {
        return None;
    };
    if *key_map != IndexMap::Affine(map::projection(2, &[1]))
        || *query_map != IndexMap::Affine(map::projection(2, &[0]))
        || !matches!(program.get(key.0 as usize), Some(Op::Iota { .. }))
    {
        return None;
    }
    let shift_operands = elementwise_operands(program, *query_absolute, ScalarOp::Add)?;
    let [(query_index, _), (cached_len, _)] = shift_operands else {
        return None;
    };
    if !matches!(program.get(query_index.0 as usize), Some(Op::Iota { .. })) {
        return None;
    }
    matches!(program.get(cached_len.0 as usize), Some(Op::Input { .. })).then_some(*cached_len)
}

/// The rank-0 [`Op::Input`] leaf named `name`, found by NAME rather than by
/// arithmetic shape — the precedent [`Op::Input`]'s own doc states
/// (`"name is identity, not decoration"`): a distributed cut edge delivers a
/// tensor over a wire keyed by name, and this is the same lookup, run
/// locally. [`cached_attention_candidates`]'s own `cached_len` operand needs
/// this rather than [`exact_merged_causal_mask_cached_len`]'s mask-arithmetic
/// walk because a two-range program's cached-range attention feeds no
/// arithmetic from `cached_len` at all — the bucket's padding is excluded by
/// a runtime BOUND on the fused op, never by a mask node in this graph, so
/// there is no expression here to walk backward from.
#[cfg(feature = "cached-attention-streaming")]
pub(super) fn find_named_input(program: &[Op], name: &str) -> Option<NodeId> {
    program.iter().enumerate().find_map(|(position, op)| {
        let is_named = matches!(op, Op::Input { .. }) && op.name() == Some(name);
        is_named.then_some(NodeId(position as u32))
    })
}

#[cfg(feature = "cached-attention-streaming")]
pub(super) fn cached_attention_candidates(
    program: &[Op],
    shapes: &Shapes,
    resolved: &[BoundOp],
    effective_outputs: &[NodeId],
    require_output_resolved: bool,
) -> Vec<(BoundOp, BTreeSet<NodeId>)> {
    let mut candidates = Vec::new();
    // resolved once: every caller supplies this leaf unconditionally
    // (`find_named_input`'s own doc), and both the padding-select walk below
    // and the ninth-operand push near the end of this loop need the SAME
    // node identity to agree it is the one true `cached_len`.
    let named_cached_len = find_named_input(program, "cached_len");
    for output_position in (0..program.len()).rev() {
        let output = NodeId(output_position as u32);
        let Some(attended_sum) = binary_elementwise(program, output, ScalarOp::Multiply) else {
            continue;
        };
        let Some(attended_parts) = binary_elementwise(program, attended_sum[0], ScalarOp::Add)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "not_online_softmax_add",
                "cached_attention decline -- weighted-value numerator is not an Add"
            );
            continue;
        };
        let Some(inverse_sum) = unary_elementwise(program, attended_sum[1], ScalarOp::Reciprocal)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "not_reciprocal_denominator",
                "cached_attention decline -- weighted-value denominator is not a Reciprocal"
            );
            continue;
        };
        let Some(sum_parts) = binary_elementwise(program, inverse_sum, ScalarOp::Add) else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "not_denominator_add",
                "cached_attention decline -- reciprocal source is not an Add"
            );
            continue;
        };
        let Some(cached_weights) =
            reduced_source(program, sum_parts[0], ScalarOp::Add, ReduceInit::Zero)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "not_cached_weight_reduce",
                "cached_attention decline -- cached-side denominator term is not a zero-init Add reduce"
            );
            continue;
        };
        let Some(new_weights) =
            reduced_source(program, sum_parts[1], ScalarOp::Add, ReduceInit::Zero)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "not_new_weight_reduce",
                "cached_attention decline -- new-side denominator term is not a zero-init Add reduce"
            );
            continue;
        };
        let Some(cached_shift) = unary_elementwise(program, cached_weights, ScalarOp::Exponential)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "not_cached_shift_exp",
                "cached_attention decline -- cached-side weight is not an Exponential"
            );
            continue;
        };
        let Some(new_shift) = unary_elementwise(program, new_weights, ScalarOp::Exponential) else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "not_new_shift_exp",
                "cached_attention decline -- new-side weight is not an Exponential"
            );
            continue;
        };
        let Some(cached_score_parts) =
            binary_elementwise(program, cached_shift, ScalarOp::Subtract)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "not_cached_score_subtract",
                "cached_attention decline -- cached-side shift source is not a Subtract"
            );
            continue;
        };
        let Some(new_score_parts) = binary_elementwise(program, new_shift, ScalarOp::Subtract)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "not_new_score_subtract",
                "cached_attention decline -- new-side shift source is not a Subtract"
            );
            continue;
        };
        if cached_score_parts[1] != new_score_parts[1] {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "score_denominator_mismatch",
                "cached_attention decline -- cached/new score denominators diverge"
            );
            continue;
        }
        let new_masked = new_score_parts[0];
        let Some(mask_parts) = elementwise_operands(program, new_masked, ScalarOp::Select) else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "mask_select_shape",
                "cached_attention decline -- new score is not a Select mask node"
            );
            continue;
        };
        let [(mask, _), (negative_infinity, _), (new_scaled, _)] = mask_parts else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "mask_select_arity",
                "cached_attention decline -- mask Select does not carry exactly 3 operands"
            );
            continue;
        };
        if !is_exact_causal_mask(program, *mask)
            || constant_value(program, *negative_infinity) != Some(f32::NEG_INFINITY)
        {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "mask_form",
                is_causal = is_exact_causal_mask(program, *mask),
                "cached_attention decline -- mask is not the exact causal form"
            );
            continue;
        }
        // qwen35's own chain masks cached-range padding with a `Select`
        // right here (`spec.rs:4979-4997`, `is_cached_padding`) before the
        // online-softmax subtract this matcher already walked past above --
        // the fused op's own runtime `cached_key_rows` clip
        // (`cpu.rs:6944-6974`) excludes exactly those rows, so dropping the
        // mask node is sound whenever its bound is the SAME `cached_len`
        // leaf the ninth operand below reads.
        let cached_scaled_source =
            unwrap_cached_padding_select(program, cached_score_parts[0], named_cached_len)
                .unwrap_or(cached_score_parts[0]);
        let Some(cached_scaled_parts) =
            binary_elementwise(program, cached_scaled_source, ScalarOp::Multiply)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "cached_scale_shape",
                "cached_attention decline -- cached padding-unwrapped score is not a scale Multiply"
            );
            continue;
        };
        let scale = cached_scaled_parts[1];
        let Some(new_scaled_parts) = binary_elementwise(program, *new_scaled, ScalarOp::Multiply)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "new_scale_shape",
                "cached_attention decline -- masked new score is not a scale Multiply"
            );
            continue;
        };
        if new_scaled_parts[1] != scale {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "scale_mismatch",
                "cached_attention decline -- cached and new score use different scale constants"
            );
            continue;
        }
        let Some((
            query_even_grouped,
            query_odd_grouped,
            cached_key_even,
            cached_key_odd,
            cached_pass,
        )) = attention_score_sources(program, cached_scaled_source, scale)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "cached_score_sources",
                "cached_attention decline -- cached score does not decompose into the qwen35 q.k score-source shape"
            );
            continue;
        };
        let Some((
            new_query_even_grouped,
            new_query_odd_grouped,
            new_key_even,
            new_key_odd,
            new_pass,
        )) = attention_score_sources(program, *new_scaled, scale)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "new_score_sources",
                "cached_attention decline -- new score does not decompose into the qwen35 q.k score-source shape"
            );
            continue;
        };
        if new_query_even_grouped != query_even_grouped
            || new_query_odd_grouped != query_odd_grouped
        {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "query_identity_mismatch",
                "cached_attention decline -- cached and new score read different query nodes"
            );
            continue;
        }
        let Some(query_even_parts) =
            binary_elementwise(program, query_even_grouped, ScalarOp::Multiply)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "query_even_group_shape",
                "cached_attention decline -- grouped query-even is not a Multiply (group broadcast) node"
            );
            continue;
        };
        let Some(query_odd_parts) =
            binary_elementwise(program, query_odd_grouped, ScalarOp::Multiply)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "query_odd_group_shape",
                "cached_attention decline -- grouped query-odd is not a Multiply (group broadcast) node"
            );
            continue;
        };
        if query_even_parts[1] != query_odd_parts[1] {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "group_broadcast_mismatch",
                "cached_attention decline -- query-even/odd use different group-broadcast operands"
            );
            continue;
        }
        let query_even = query_even_parts[0];
        let query_odd = query_odd_parts[0];
        // A pass plane must appear on BOTH the cached and new score, or not
        // at all -- qwen35's own builder always emits it on both sides
        // (`spec.rs:4910-4927,5035-5049`), so a mismatch here means this
        // program is not that shape.
        if cached_pass.is_some() != new_pass.is_some() {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "pass_presence_mismatch",
                cached_has_pass = cached_pass.is_some(),
                new_has_pass = new_pass.is_some(),
                "cached_attention decline -- pass plane present on one side of cached/new score only"
            );
            continue;
        }
        let pass = match (cached_pass, new_pass) {
            (
                Some((cached_query_pass_grouped, cached_key_pass)),
                Some((new_query_pass_grouped, new_key_pass)),
            ) => {
                if cached_query_pass_grouped != new_query_pass_grouped {
                    #[cfg(feature = "instrument")]
                    debug!(
                        node = output.0,
                        stage = "pass_query_identity_mismatch",
                        "cached_attention decline -- cached and new score read different pass-plane query nodes"
                    );
                    continue;
                }
                let Some(query_pass_parts) =
                    binary_elementwise(program, cached_query_pass_grouped, ScalarOp::Multiply)
                else {
                    #[cfg(feature = "instrument")]
                    debug!(
                        node = output.0,
                        stage = "pass_query_group_shape",
                        "cached_attention decline -- grouped pass-plane query is not a Multiply (group broadcast) node"
                    );
                    continue;
                };
                if query_pass_parts[1] != query_even_parts[1] {
                    #[cfg(feature = "instrument")]
                    debug!(
                        node = output.0,
                        stage = "pass_group_broadcast_mismatch",
                        "cached_attention decline -- pass-plane query uses a different group-broadcast operand than q_even"
                    );
                    continue;
                }
                Some((query_pass_parts[0], cached_key_pass, new_key_pass))
            }
            _ => None,
        };
        let Some(cached_value_source) =
            reduced_source(program, attended_parts[0], ScalarOp::Add, ReduceInit::Zero)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "cached_value_reduce_shape",
                "cached_attention decline -- cached attended-value term is not a zero-init Add reduce"
            );
            continue;
        };
        let Some(new_value_source) =
            reduced_source(program, attended_parts[1], ScalarOp::Add, ReduceInit::Zero)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "new_value_reduce_shape",
                "cached_attention decline -- new attended-value term is not a zero-init Add reduce"
            );
            continue;
        };
        let Some(cached_value_product) =
            binary_elementwise(program, cached_value_source, ScalarOp::Multiply)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "cached_value_product_shape",
                "cached_attention decline -- cached value-weight term is not a Multiply node"
            );
            continue;
        };
        let Some(new_value_product) =
            binary_elementwise(program, new_value_source, ScalarOp::Multiply)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "new_value_product_shape",
                "cached_attention decline -- new value-weight term is not a Multiply node"
            );
            continue;
        };
        let cached_value = cached_value_product[1];
        let new_value = new_value_product[1];
        if cached_value_product[0] != cached_weights || new_value_product[0] != new_weights {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "value_weight_identity_mismatch",
                "cached_attention decline -- value product does not multiply against this branch's own softmax weight"
            );
            continue;
        }
        let mut source_nodes = alloc::vec![
            query_even,
            query_odd,
            cached_key_even,
            cached_key_odd,
            new_key_even,
            new_key_odd,
            cached_value,
            new_value,
        ];
        if let Some((pass_query, pass_cached_key, pass_new_key)) = pass {
            source_nodes.extend([pass_query, pass_cached_key, pass_new_key]);
        }
        let mut operands = Vec::with_capacity(source_nodes.len());
        for source in &source_nodes {
            let Some((_, layout, lookup)) = resolved
                .iter()
                .flat_map(|bound| bound.operands().iter())
                .find(|(node, _, _)| node == source)
            else {
                #[cfg(feature = "instrument")]
                debug!(
                    node = output.0,
                    stage = "source_not_found",
                    source = source.0,
                    "cached_attention decline -- a score/value source node is not an operand of any resolved op"
                );
                operands.clear();
                break;
            };
            if lookup.is_some() || layout.strides.iter().any(|stride| *stride < 0) {
                #[cfg(feature = "instrument")]
                debug!(
                    node = output.0,
                    stage = "source_indirect_or_negative_stride",
                    source = source.0,
                    has_lookup = lookup.is_some(),
                    "cached_attention decline -- a score/value source is gathered indirectly or carries a negative stride"
                );
                operands.clear();
                break;
            }
            operands.push((*source, layout.clone(), None));
        }
        if operands.len() != source_nodes.len() {
            continue;
        }
        let Some(scale_value) = constant_value(program, scale) else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "scale_not_constant",
                "cached_attention decline -- the score scale operand is not a compile-time constant"
            );
            continue;
        };
        let query_shape = shapes.of(query_even_grouped);
        let cached_key_shape = shapes.of(cached_key_even);
        let new_key_shape = shapes.of(new_key_even);
        let cached_value_shape = shapes.of(cached_value);
        let new_value_shape = shapes.of(new_value);
        let Some(rotary_width) = query_shape[3].checked_mul(2) else {
            continue;
        };
        // `total_head_dim` is `rotary_width` whenever no pass plane is
        // present (every non-qwen35 caller today) -- V is never rotated, so
        // its own width is the one place the pass plane's extra columns
        // surface even when the rotary planes alone would say `rotary_width`
        // (`BoundOpKind::CachedAttention`'s own doc).
        let total_head_dim = match pass {
            Some((pass_query, _, _)) => {
                let Some(&pass_dim) = shapes.of(pass_query).last() else {
                    continue;
                };
                let Some(total) = rotary_width.checked_add(pass_dim) else {
                    continue;
                };
                total
            }
            None => rotary_width,
        };
        if query_shape.len() != 4
            || cached_key_shape.len() != 3
            || new_key_shape.len() != 3
            || cached_value_shape.len() != 3
            || new_value_shape.len() != 3
            || query_shape[1] != cached_key_shape[1]
            || query_shape[1] != new_key_shape[1]
            || cached_key_shape[1] != cached_value_shape[1]
            || new_key_shape[1] != new_value_shape[1]
            || cached_key_shape[0] != cached_value_shape[0]
            || new_key_shape[0] != new_value_shape[0]
            || cached_value_shape[2] != total_head_dim
            || new_value_shape[2] != total_head_dim
            || shapes.of(output)
                != [
                    query_shape[0],
                    query_shape[1],
                    query_shape[2],
                    total_head_dim,
                ]
        {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "shape_checks",
                ?query_shape,
                ?cached_key_shape,
                ?new_key_shape,
                ?cached_value_shape,
                ?new_value_shape,
                total_head_dim,
                output_shape = ?shapes.of(output),
                "cached_attention decline -- query/key/value/output shapes do not agree on kv_heads/head_dim"
            );
            continue;
        }
        let pair_dim = query_shape[3];
        let query_strides = [
            (query_shape[1] * query_shape[2] * pair_dim) as i64,
            (query_shape[2] * pair_dim) as i64,
            pair_dim as i64,
            1i64,
        ];
        let key_strides = [
            0i64,
            (query_shape[1] * pair_dim) as i64,
            pair_dim as i64,
            0,
            1,
        ];
        let value_strides = [
            0i64,
            (query_shape[1] * total_head_dim) as i64,
            total_head_dim as i64,
            0,
            1,
        ];
        if operands[0].1.strides.as_slice() != query_strides
            || operands[1].1.strides.as_slice() != query_strides
            || operands[2].1.strides.as_slice() != key_strides
            || operands[3].1.strides.as_slice() != key_strides
            || operands[4].1.strides.as_slice() != key_strides
            || operands[5].1.strides.as_slice() != key_strides
            || operands[6].1.strides.as_slice() != value_strides
            || operands[7].1.strides.as_slice() != value_strides
        {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "base_strides",
                ?query_strides,
                ?key_strides,
                ?value_strides,
                got = ?operands[..8].iter().map(|(_, layout, _)| layout.strides.clone()).collect::<Vec<_>>(),
                "cached_attention decline -- the base 8 operands do not carry the fused kernel's assumed GEMM strides"
            );
            continue;
        }
        if let Some((_, pass_cached_key, pass_new_key)) = pass {
            let pass_dim = total_head_dim - rotary_width;
            let pass_query_strides = [
                (query_shape[1] * query_shape[2] * pass_dim) as i64,
                (query_shape[2] * pass_dim) as i64,
                pass_dim as i64,
                1i64,
            ];
            let pass_key_strides = [
                0i64,
                (query_shape[1] * pass_dim) as i64,
                pass_dim as i64,
                0,
                1,
            ];
            let cached_key_pass_shape = shapes.of(pass_cached_key);
            let new_key_pass_shape = shapes.of(pass_new_key);
            if cached_key_pass_shape.len() != 3
                || new_key_pass_shape.len() != 3
                || cached_key_pass_shape[1] != query_shape[1]
                || new_key_pass_shape[1] != query_shape[1]
                || cached_key_pass_shape[2] != pass_dim
                || new_key_pass_shape[2] != pass_dim
                || cached_key_pass_shape[0] != cached_key_shape[0]
                || new_key_pass_shape[0] != new_key_shape[0]
                || operands[8].1.strides.as_slice() != pass_query_strides
                || operands[9].1.strides.as_slice() != pass_key_strides
                || operands[10].1.strides.as_slice() != pass_key_strides
            {
                #[cfg(feature = "instrument")]
                debug!(
                    node = output.0,
                    stage = "pass_strides",
                    ?pass_query_strides,
                    ?pass_key_strides,
                    got = ?operands[8..11].iter().map(|(_, layout, _)| layout.strides.clone()).collect::<Vec<_>>(),
                    ?cached_key_pass_shape,
                    ?new_key_pass_shape,
                    "cached_attention decline -- the pass-plane operands do not carry the fused kernel's assumed strides"
                );
                continue;
            }
        }
        // The pass triple is set aside here and re-appended AFTER the
        // optional `cached_len` push below -- `cpu.rs:6913-6917`'s own
        // `pass_start` reads the pass plane at index 8 when `cached_len` is
        // absent and index 9 when present, never at a fixed offset from the
        // base eight.
        let pass_operands = if pass.is_some() {
            Some(operands.split_off(8))
        } else {
            None
        };
        let dependencies = attention_dependencies(program, output, &source_nodes);
        let dependencies = dependencies
            .difference(&source_nodes.into_iter().collect())
            .copied()
            .collect::<BTreeSet<_>>();
        if dependencies
            .iter()
            .any(|node| effective_outputs.contains(node))
        {
            #[cfg(feature = "instrument")]
            for node in &dependencies {
                if effective_outputs.contains(node) {
                    debug!(
                        node = node.0,
                        kind = "cached_attention_absorption",
                        decision = "rejected_requested_output",
                        into = output.0,
                        "attention mask dependency not absorbed -- it is a requested output"
                    );
                }
            }
            continue;
        }
        let absorbed = removable_attention_dependencies(program, &dependencies, output);
        if absorbed.is_empty() {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "no_removable_dependencies",
                "cached_attention decline -- no intermediate ops become dead once this fusion absorbs its sources"
            );
            continue;
        }
        if require_output_resolved && !resolved.iter().any(|bound| bound.node == output) {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "output_not_resolved",
                "cached_attention decline -- the candidate output node has no resolved binding"
            );
            continue;
        }
        // Every caller of `mistral_cached_forward_program_with_experts`
        // supplies a rank-0 "cached_len" `Op::Input` unconditionally
        // (`find_named_input`'s own doc) -- when a program predates that
        // (a hand-built test fixture with no such leaf), fall back to the
        // eight-operand, unbounded shape rather than erroring: today's
        // behavior for every caller that never opted into bucketing.
        // `cached_key_shape[0] == 0` (the very first decode step, before any
        // token is cached) is skipped even when the leaf exists: an empty
        // cached range has no padding to exclude, and giving it the ninth
        // operand anyway would make its `cached_key_rows == 0` collide with
        // `single_range_dynamic`'s own discriminator (`BoundOpKind::
        // CachedAttention`'s own doc) -- the two shapes are structurally
        // indistinguishable at that value, so this is the one case that
        // must stay eight-operand regardless of bucketing.
        if cached_key_shape[0] > 0
            && let Some(cached_len_node) = named_cached_len
            && shapes.of(cached_len_node).is_empty()
        {
            operands.push((
                cached_len_node,
                Layout {
                    base: 0,
                    strides: SmallVec::new(),
                },
                None,
            ));
        }
        if let Some(pass_operands) = pass_operands {
            operands.extend(pass_operands);
        }
        let fused = BoundOp {
            node: output,
            dtype: DType::Float32,
            extents: shapes.of(output).to_vec(),
            kind: BoundOpKind::CachedAttention {
                operands,
                query_rows: query_shape[0],
                cached_key_rows: cached_key_shape[0],
                new_key_rows: new_key_shape[0],
                kv_heads: query_shape[1],
                query_groups: query_shape[2],
                head_dim: total_head_dim,
                // `rotary_width` whenever no pass plane matched (every
                // non-qwen35 caller, `total_head_dim == rotary_width`);
                // qwen35's own partial-rotary chain sets this strictly
                // below `head_dim` (`attention_score_sources`'s own doc).
                rotary_dim: rotary_width,
                scale: scale_value,
                cached_lower_inclusive: i64::MIN,
                new_upper_inclusive: 0,
            },
        };
        #[cfg(feature = "instrument")]
        debug!(
            node = output.0,
            require_output_resolved,
            kv_heads = query_shape[1],
            head_dim = total_head_dim,
            "cached_attention candidate accepted"
        );
        candidates.push((fused, absorbed));
    }
    candidates
}

/// [`cached_attention_candidates`]'s counterpart for
/// [`crate::spec::append_mistral_single_range_cached_layer`]'s output shape:
/// one merged key/value range instead of a cached/new pair, so there is no
/// online-softmax combine to unwind — `attended` is a plain single-pass
/// softmax over one masked score matrix
/// (`score_even+score_odd` -> mask -> max -> sub+exp -> sum -> reciprocal ->
/// multiply -> weight the one value range), the same eight-step chain
/// [`crate::spec::append_mistral_layer`] emits for a from-scratch (no cache)
/// forward pass. The fused [`BoundOpKind::CachedAttention`] still declares
/// two key/value ranges (its only shape today, per this module's own
/// `no new BoundOpKind` constraint): the single merged range is placed in
/// the "new" slot, which already carries the causal band restricting it to
/// non-future positions, and the "cached" slot is declared with
/// `cached_key_rows: 0` rather than duplicating the merged range into it —
/// an empty range, not a live range neutered by an unreachable band. Both
/// [`crate::physical::stream_cached_attention_split_gqa`] and the Metal
/// kernel ([`crate::msl`]'s cached-attention render) treat a zero-length
/// cached range as a first-class case: nothing iterates it, rather than
/// iterating it and skipping every row via a dead-band `continue`.
#[cfg(feature = "cached-attention-streaming")]
pub(super) fn cached_attention_single_range_candidates(
    program: &[Op],
    shapes: &Shapes,
    resolved: &[BoundOp],
    effective_outputs: &[NodeId],
) -> Vec<(BoundOp, BTreeSet<NodeId>)> {
    let mut candidates = Vec::new();
    for output_position in (0..program.len()).rev() {
        let output = NodeId(output_position as u32);
        let Some(attended_product) =
            reduced_source(program, output, ScalarOp::Add, ReduceInit::Zero)
        else {
            continue;
        };
        let Some(attended_parts) =
            binary_elementwise(program, attended_product, ScalarOp::Multiply)
        else {
            continue;
        };
        let Some(probabilities_parts) =
            binary_elementwise(program, attended_parts[0], ScalarOp::Multiply)
        else {
            continue;
        };
        let Some(weight_sum) =
            unary_elementwise(program, probabilities_parts[1], ScalarOp::Reciprocal)
        else {
            continue;
        };
        let Some(weights) = reduced_source(program, weight_sum, ScalarOp::Add, ReduceInit::Zero)
        else {
            continue;
        };
        if weights != probabilities_parts[0] {
            continue;
        }
        let Some(shifted) = unary_elementwise(program, weights, ScalarOp::Exponential) else {
            continue;
        };
        let Some(shifted_parts) = binary_elementwise(program, shifted, ScalarOp::Subtract) else {
            continue;
        };
        let Some(scores_masked_from_max) = reduced_source(
            program,
            shifted_parts[1],
            ScalarOp::Maximum,
            ReduceInit::NegativeInfinity,
        ) else {
            continue;
        };
        if scores_masked_from_max != shifted_parts[0] {
            continue;
        }
        let scores_masked = shifted_parts[0];
        let Some(mask_parts) = elementwise_operands(program, scores_masked, ScalarOp::Select)
        else {
            continue;
        };
        let [(mask, _), (negative_infinity, _), (scores_scaled, _)] = mask_parts else {
            continue;
        };
        let Some(cached_len_node) = exact_merged_causal_mask_cached_len(program, *mask) else {
            continue;
        };
        if constant_value(program, *negative_infinity) != Some(f32::NEG_INFINITY) {
            continue;
        }
        let Some(scaled_operands) = binary_elementwise(program, *scores_scaled, ScalarOp::Multiply)
        else {
            continue;
        };
        let scale = scaled_operands[1];
        // mistral's single-range chain never carries a pass plane -- this
        // matcher's own shape is full-rotary only, per its module doc.
        let Some((query_even_grouped, query_odd_grouped, key_even, key_odd, None)) =
            attention_score_sources(program, *scores_scaled, scale)
        else {
            continue;
        };
        let Some(query_even_parts) =
            binary_elementwise(program, query_even_grouped, ScalarOp::Multiply)
        else {
            continue;
        };
        let Some(query_odd_parts) =
            binary_elementwise(program, query_odd_grouped, ScalarOp::Multiply)
        else {
            continue;
        };
        if query_even_parts[1] != query_odd_parts[1] {
            continue;
        }
        let query_even = query_even_parts[0];
        let query_odd = query_odd_parts[0];
        let value = attended_parts[1];
        let source_nodes = [
            query_even, query_odd, key_even, key_odd, key_even, key_odd, value, value,
        ];
        let mut operands = Vec::with_capacity(source_nodes.len());
        for source in source_nodes {
            let Some((_, layout, lookup)) = resolved
                .iter()
                .flat_map(|bound| bound.operands().iter())
                .find(|(node, _, _)| *node == source)
            else {
                operands.clear();
                break;
            };
            if lookup.is_some() || layout.strides.iter().any(|stride| *stride < 0) {
                operands.clear();
                break;
            }
            operands.push((source, layout.clone(), None));
        }
        if operands.len() != source_nodes.len() {
            continue;
        }
        let Some(scale_value) = constant_value(program, scale) else {
            continue;
        };
        let query_shape = shapes.of(query_even_grouped);
        let key_shape = shapes.of(key_even);
        let value_shape = shapes.of(value);
        let Some(head_dim) = query_shape[3].checked_mul(2) else {
            continue;
        };
        if query_shape.len() != 4
            || key_shape.len() != 3
            || value_shape.len() != 3
            || query_shape[1] != key_shape[1]
            || key_shape[1] != value_shape[1]
            || key_shape[0] != value_shape[0]
            || value_shape[2] != head_dim
            || shapes.of(output) != [query_shape[0], query_shape[1], query_shape[2], head_dim]
            || key_shape[0] < query_shape[0]
        {
            continue;
        }
        // `key_shape[0]` (`t`, the whole merged range) minus `query_shape[0]`
        // (`s`, this call's own new positions) equals `cached_len` only when
        // `t` is exactly the merged length -- true for a plain evaluate, but
        // `kv-capacity-bucket` widens `t` to `ceil(merged_len /
        // bucket_tokens) * bucket_tokens`, so this difference silently
        // becomes `bucket - new_count`, larger than the real `cached_len` by
        // the padding. The band this feeds must therefore come from the
        // `cached_len` VALUE itself -- the same per-call `Op::Input`
        // `causal_mask_merged`'s query side already adds
        // (`exact_merged_causal_mask_cached_len` captured it above as
        // `cached_len_node`) -- carried through as this op's ninth operand
        // and read at execution time, never baked from a shape difference.
        if !shapes.of(cached_len_node).is_empty() {
            continue;
        }
        let cached_len_operand = (
            cached_len_node,
            Layout {
                base: 0,
                strides: SmallVec::new(),
            },
            None,
        );
        let pair_dim = query_shape[3];
        let query_strides = [
            (query_shape[1] * query_shape[2] * pair_dim) as i64,
            (query_shape[2] * pair_dim) as i64,
            pair_dim as i64,
            1i64,
        ];
        let key_strides = [
            0i64,
            (query_shape[1] * pair_dim) as i64,
            pair_dim as i64,
            0,
            1,
        ];
        let value_strides = [
            0i64,
            (query_shape[1] * query_shape[3] * 2) as i64,
            (query_shape[3] * 2) as i64,
            0,
            1,
        ];
        if operands[0].1.strides.as_slice() != query_strides
            || operands[1].1.strides.as_slice() != query_strides
            || operands[2].1.strides.as_slice() != key_strides
            || operands[3].1.strides.as_slice() != key_strides
            || operands[4].1.strides.as_slice() != key_strides
            || operands[5].1.strides.as_slice() != key_strides
            || operands[6].1.strides.as_slice() != value_strides
            || operands[7].1.strides.as_slice() != value_strides
        {
            continue;
        }
        let dependencies = attention_dependencies(program, output, &source_nodes);
        // `cached_len_node` sits on the same mask-chain path `source_nodes`
        // already gets excluded from -- it is about to become this op's own
        // ninth operand, so it must never be classified as absorbed
        // (removed) the way the rest of the mask arithmetic is.
        let dependencies = dependencies
            .difference(&source_nodes.into_iter().collect())
            .copied()
            .filter(|node| *node != cached_len_node)
            .collect::<BTreeSet<_>>();
        if dependencies
            .iter()
            .any(|node| effective_outputs.contains(node))
        {
            #[cfg(feature = "instrument")]
            for node in &dependencies {
                if effective_outputs.contains(node) {
                    debug!(
                        node = node.0,
                        kind = "cached_attention_absorption",
                        decision = "rejected_requested_output",
                        into = output.0,
                        "attention mask dependency not absorbed -- it is a requested output"
                    );
                }
            }
            continue;
        }
        let absorbed = removable_attention_dependencies(program, &dependencies, output);
        if absorbed.is_empty() {
            continue;
        }
        if !resolved.iter().any(|bound| bound.node == output) {
            continue;
        }
        operands.push(cached_len_operand);
        let fused = BoundOp {
            node: output,
            dtype: DType::Float32,
            extents: shapes.of(output).to_vec(),
            kind: BoundOpKind::CachedAttention {
                operands,
                query_rows: query_shape[0],
                // no separate cached range exists for a merged buffer -- see
                // this function's own doc; `cached_key_rows: 0` makes the
                // kernel's cached half a first-class empty range instead of
                // a live range neutered by an unreachable band sentinel.
                cached_key_rows: 0,
                new_key_rows: key_shape[0],
                kv_heads: query_shape[1],
                query_groups: query_shape[2],
                head_dim,
                // this matcher recognizes only the flat two-term score
                // (`attention_score_sources`'s own doc) -- full rotary,
                // never a pass plane.
                rotary_dim: head_dim,
                scale: scale_value,
                cached_lower_inclusive: i64::MIN,
                new_upper_inclusive: 0,
            },
        };
        candidates.push((fused, absorbed));
    }
    candidates
}

#[cfg(feature = "cached-attention-streaming")]
pub(super) fn attention_dependencies(
    program: &[Op],
    output: NodeId,
    sources: &[NodeId],
) -> BTreeSet<NodeId> {
    let source_set: BTreeSet<NodeId> = sources.iter().copied().collect();
    let mut visited = BTreeSet::new();
    let mut pending = vec![output];
    while let Some(node) = pending.pop() {
        if !visited.insert(node) || source_set.contains(&node) {
            continue;
        }
        match program.get(node.0 as usize) {
            Some(Op::Elementwise { operands, .. }) => {
                pending.extend(operands.iter().map(|(source, _)| *source));
            }
            Some(Op::Reduce(reduce)) => pending.push(reduce.operand),
            Some(Op::Input { .. }) | Some(Op::Iota { .. }) | Some(Op::Constant { .. }) | None => {}
        }
    }
    visited.remove(&output);
    visited
}

#[cfg(feature = "cached-attention-streaming")]
pub(super) fn has_external_attention_consumer(
    consumers: &BTreeMap<NodeId, BTreeSet<NodeId>>,
    dependencies: &BTreeSet<NodeId>,
    dependency: NodeId,
    output: NodeId,
) -> bool {
    consumers
        .get(&dependency)
        .into_iter()
        .flatten()
        .any(|consumer| !dependencies.contains(consumer) && *consumer != output)
}

#[cfg(feature = "cached-attention-streaming")]
pub(super) fn attention_consumers(
    program: &[Op],
    dependencies: &BTreeSet<NodeId>,
) -> BTreeMap<NodeId, BTreeSet<NodeId>> {
    let mut consumers = BTreeMap::new();
    for (position, operation) in program.iter().enumerate() {
        let consumer = NodeId(position as u32);
        let mut references = Vec::new();
        match operation {
            Op::Elementwise { operands, .. } => {
                references.extend(operands.iter().map(|(node, _)| *node));
            }
            Op::Reduce(reduce) => references.push(reduce.operand),
            Op::Input { .. } | Op::Iota { .. } | Op::Constant { .. } => {}
        }
        for dependency in references
            .into_iter()
            .filter(|node| dependencies.contains(node))
        {
            consumers
                .entry(dependency)
                .or_insert_with(BTreeSet::new)
                .insert(consumer);
        }
    }
    consumers
}

#[cfg(feature = "cached-attention-streaming")]
pub(super) fn removable_attention_dependencies(
    program: &[Op],
    dependencies: &BTreeSet<NodeId>,
    output: NodeId,
) -> BTreeSet<NodeId> {
    let consumers = attention_consumers(program, dependencies);
    let mut retained = dependencies
        .iter()
        .copied()
        .filter(|node| has_external_attention_consumer(&consumers, dependencies, *node, output))
        .collect::<BTreeSet<_>>();
    let mut changed = true;
    while changed {
        changed = false;
        for node in retained.clone() {
            let ancestors = match program.get(node.0 as usize) {
                Some(Op::Elementwise { operands, .. }) => {
                    operands.iter().map(|(source, _)| *source).collect()
                }
                Some(Op::Reduce(reduce)) => alloc::vec![reduce.operand],
                Some(Op::Input { .. })
                | Some(Op::Iota { .. })
                | Some(Op::Constant { .. })
                | None => Vec::new(),
            };
            for ancestor in ancestors {
                if dependencies.contains(&ancestor) && retained.insert(ancestor) {
                    changed = true;
                }
            }
        }
    }
    dependencies.difference(&retained).copied().collect()
}
