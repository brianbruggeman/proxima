use super::*;

pub(super) fn bind_cached_attention_fusion(
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
    fuse_cached_attention: bool,
    numeric_policy: NumericPolicy,
) -> Result<Vec<BoundOp>, TensorError> {
    let built = bind_plain(program, shapes, outputs, numeric_policy)?;
    #[cfg(not(feature = "cached-attention-streaming"))]
    {
        let _ = fuse_cached_attention;
        Ok(built)
    }

    #[cfg(feature = "cached-attention-streaming")]
    {
        if !fuse_cached_attention {
            return Ok(built);
        }
        // `false`: this discovery pass finds anchors `bind_plain` has already
        // folded into their single consumer (qwen35's own `attended` tap,
        // `bind.rs:993`'s `elementwise_operand_fuse`) precisely so their node
        // id can be pinned into `planning_outputs` below and survive the
        // rebuild -- requiring a resolved binding here would make discovery
        // depend on the very materialization it exists to produce.
        let mut initial_candidates =
            cached_attention_candidates(program, shapes, &built, outputs, false);
        initial_candidates.extend(cached_attention_single_range_candidates(
            program, shapes, &built, outputs,
        ));
        if initial_candidates.is_empty() {
            return Ok(built);
        }
        let mut planning_outputs = outputs.to_vec();
        if planning_outputs.is_empty() {
            let root = program
                .len()
                .checked_sub(1)
                .map(|position| NodeId(position as u32))
                .ok_or(TensorError::Empty)?;
            planning_outputs.push(root);
        }
        for (fused, _) in &initial_candidates {
            let BoundOpKind::CachedAttention { operands, .. } = &fused.kind else {
                continue;
            };
            // the anchor itself (qwen35's own `attended` tap, single-consumer
            // into the per-head gate multiply) must be pinned alongside its
            // sources -- `bind_plain`'s single-consumer elementwise fusion
            // folds an unrequested single-consumer node into its consumer
            // before this function ever sees it (`bind.rs:993`'s own
            // `elementwise_operand_fuse`), so without this the anchor never
            // gets a standalone `BoundOp` for the second pass below to find
            // (`qwen35_partial_rotary_cached_attention_fuses_and_matches_the_unfused_layer`'s
            // own comment names the same requirement for its fixture's outputs).
            if !planning_outputs.contains(&fused.node) {
                planning_outputs.push(fused.node);
            }
            for (source, _, _) in operands {
                if !planning_outputs.contains(source) {
                    planning_outputs.push(*source);
                }
            }
        }
        let rebuilt = bind_plain(program, shapes, &planning_outputs, numeric_policy)?;
        let mut candidates = cached_attention_candidates(program, shapes, &rebuilt, outputs, true);
        candidates.extend(cached_attention_single_range_candidates(
            program, shapes, &rebuilt, outputs,
        ));
        if candidates.is_empty() {
            return Ok(built);
        }
        let fused_by_node = candidates
            .iter()
            .map(|(fused, _)| (fused.node, fused))
            .collect::<BTreeMap<_, _>>();
        let absorbed = candidates
            .iter()
            .flat_map(|(_, absorbed)| absorbed.iter().copied())
            .collect::<BTreeSet<_>>();
        let mut rewritten = Vec::with_capacity(rebuilt.len());
        for bound in rebuilt {
            if let Some(fused) = fused_by_node.get(&bound.node) {
                rewritten.push((*fused).clone());
            } else if !absorbed.contains(&bound.node) {
                rewritten.push(bound);
            }
        }
        Ok(rewritten)
    }
}

/// Is `bound` a still-un-scattered `Keep::Reduce` fold — the only
/// [`BoundOpKind`] a consumer's epilogue can ever absorb (`out_scatter`'s own
/// doc: a scatter's destination is data-dependent, never a plain identity or
/// broadcast projection a consumer could read through).
#[cfg(feature = "reduce-epilogue-fusion")]
pub(super) fn is_epilogue_fusable_reduce(bound: &BoundOp) -> bool {
    matches!(
        bound.kind,
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            out_scatter: None,
            ..
        }
    )
}

/// One flag per [`NodeId`] this program can name, `true` exactly for a
/// [`is_epilogue_fusable_reduce`] node — [`find_epilogue_source`]'s own
/// lookup table, sized once per [`reduce_epilogue_fusion`] pass rather than
/// re-scanned per candidate.
#[cfg(feature = "reduce-epilogue-fusion")]
pub(super) fn reduce_epilogue_source_flags(resolved: &[BoundOp]) -> Vec<bool> {
    let node_count = resolved
        .iter()
        .map(|bound| bound.node.0 as usize + 1)
        .max()
        .unwrap_or(0);
    let mut flags = vec![false; node_count];
    for bound in resolved {
        if is_epilogue_fusable_reduce(bound) {
            flags[bound.node.0 as usize] = true;
        }
    }
    flags
}

/// How many DISTINCT resolved ops read each [`NodeId`] this program can
/// name, via any of that consumer's own [`BoundOp::all_read_sources`] —
/// [`reduce_epilogue_fusion`]'s own liveness gate (condition (b): "no OTHER
/// consumer"), computed over the ALREADY-FUSED op list so a node absorbed
/// into a `ComposedBody` upstream (never its own [`BoundOp`]) correctly
/// counts zero rather than needing a separate raw-`Op` walk.
///
/// Counts CONSUMING OPERATIONS, not operand occurrences: production SiLU
/// (`spec.rs`'s `silu` builder) reads its `gate` operand twice within the
/// SAME consumer — directly, and again inside `exp(-gate)` — and chain
/// composition preserves both as separate `operands` slots. Naively counting
/// every slot would see `gate` "referenced" twice and reject condition (b)
/// even though exactly one consumer reads it. Each `bound`'s own reads are
/// deduped to their distinct source [`NodeId`]s before folding into the
/// per-source total, so N reads of the same source by one consumer count as
/// the one reference that consumer actually is.
#[cfg(feature = "reduce-epilogue-fusion")]
pub(super) fn resolved_reference_counts(resolved: &[BoundOp]) -> BTreeMap<NodeId, u32> {
    let mut counts = BTreeMap::new();
    for bound in resolved {
        let mut sources_read = BTreeSet::new();
        for (source, _, gather) in bound.all_read_sources() {
            sources_read.insert(*source);
            if let Some(lookup) = gather {
                sources_read.insert(lookup.indices);
            }
        }
        for source in sources_read {
            *counts.entry(source).or_insert(0u32) += 1;
        }
    }
    counts
}

/// The one reduce-fold operand `consumer` can absorb into its epilogue, if
/// any: the first operand among `consumer.operands()` — no gather (a
/// gathered read is data-dependent, `apply_reduce_epilogue`'s own doc names
/// this unsupported) — whose node is [`reduce_epilogue_source_flags`]-true.
/// Structural over [`BoundOp`]/[`Layout`] only, at the RESOLVED level — this
/// is what lets the match see straight through however many raw `Op` steps
/// ordinary chain-fusion already folded into `consumer`'s own
/// [`ComposedBody`], the exact one-hop limitation a raw-`Op`-level scan hits
/// (a multi-step tail between the fold and its real, final consumer is
/// already ONE [`BoundOp`] by the time this runs, keyed at the final
/// consumer's own [`NodeId`], not at whichever raw op happened to sit
/// directly after the reduce).
///
/// Multiple `operands` slots naming the SAME reduce are not automatically a
/// conflict: production SiLU reads its reduce-derived `gate` operand once
/// directly and once more inside `exp(-gate)`, and chain composition
/// preserves both as separate slots reading the SAME [`Layout`] (identical
/// projection) — that is ONE logical read repeated, not two. Two slots
/// naming the same reduce through DIFFERENT [`Layout`]s is the real
/// conflict this declines (see below).
#[cfg(feature = "reduce-epilogue-fusion")]
pub(super) fn find_epilogue_source(consumer: &BoundOp, reduce_flags: &[bool]) -> Option<NodeId> {
    let BoundOpKind::Elementwise { operands, .. } = &consumer.kind else {
        return None;
    };
    if operands.iter().any(|(_, _, gather)| gather.is_some()) {
        return None; // no renderer/evaluator supports a gathered epilogue read.
    }
    let mut found: Option<(NodeId, &Layout)> = None;
    for (node, layout, _) in operands {
        if !reduce_flags.get(node.0 as usize).copied().unwrap_or(false) {
            continue;
        }
        match found {
            None => found = Some((*node, layout)),
            // The SAME slot (same node, same projection) read again — the
            // production-SiLU shape (`gate` used both bare and inside
            // `exp(-gate)`). One logical read; nothing more to record.
            Some((existing_node, existing_layout))
                if existing_node == *node && existing_layout == layout => {}
            // The SAME reduce read through a DIFFERENT projection — a
            // parity-selecting split like "gate = paired[..,0,..]" /
            // "up = paired[..,1,..]" both landing in one consumer.
            // `compose_reduce_epilogue`'s own "implicit fold-result slot"
            // model has room for exactly ONE such read; absorbing the fold
            // here would silently drop whichever occurrence isn't picked as
            // the sentinel while still trying to read the fold's now-gone
            // standalone buffer for the other one. Decline the whole
            // consumer rather than guess which read wins.
            Some((existing_node, _)) if existing_node == *node => return None,
            Some(_) => {}
        }
    }
    found.map(|(node, _)| node)
}

/// One (consumer, reduce) pair a single [`reduce_epilogue_fusion`] pass will
/// merge: `consumer` is a resolved [`BoundOpKind::Elementwise`] whose sole
/// reduce-fold operand (per [`find_epilogue_source`]) is `source`, `source`
/// has no OTHER reader anywhere in `resolved` and is not itself a required
/// output. Whether that operand is read broadcast (the `[s,d]`-shaped
/// "broadcast-reduce" epilogue an RMSNorm-shaped `x * inv_rms` tail needs) or
/// at plain identity (the pre-existing bias/residual epilogue shape) is
/// immaterial here — [`compose_reduce_epilogue`] widens correctly either way
/// from the two [`BoundOp`]s' own recorded extents, never from a name or
/// shape special-cased in this match.
#[cfg(feature = "reduce-epilogue-fusion")]
pub(super) fn reduce_epilogue_candidates(resolved: &[BoundOp], outputs: &[NodeId]) -> Vec<(NodeId, NodeId)> {
    let reduce_flags = reduce_epilogue_source_flags(resolved);
    let reference_counts = resolved_reference_counts(resolved);
    let mut candidates = Vec::new();
    for bound in resolved {
        let Some(source) = find_epilogue_source(bound, &reduce_flags) else {
            continue;
        };
        if reference_counts.get(&source).copied().unwrap_or(0) != 1 {
            continue; // (b): some OTHER op still reads this fold's output.
        }
        if outputs.contains(&source) {
            continue; // (b): a requested output must still materialize on its own.
        }
        candidates.push((bound.node, source));
    }
    candidates
}

/// The bind-time rewrite [`bind_with_fusion`] runs whenever
/// `reduce-epilogue-fusion` is compiled in: every
/// [`reduce_epilogue_candidates`] match becomes one merged [`BoundOp`] whose
/// `node` is the CONSUMER's id (see [`BoundOpKind::Reduce::epilogue_body`]'s
/// own doc for why), replacing both the standalone reduce and the standalone
/// consumer `resolved` already held. Runs to a FIXPOINT (bounded by
/// `resolved.len()`, so it always terminates — each round strictly shrinks
/// the op count or stops) because an RMSNorm-shaped tail needs TWO rounds:
/// round one absorbs the plain `mean/eps/sqrt/reciprocal` chain into the
/// fold itself (a PLAIN epilogue, output shape unchanged); only after that
/// does the fold's own `NodeId` carry `Keep::Reduce` for
/// [`find_epilogue_source`] to match `x * inv_rms`'s BROADCAST read in round
/// two. A backend with no epilogue renderer must reject a non-default
/// `epilogue_body`/`epilogue_operands` at bind time rather than call this at
/// all with the feature compiled in against data it cannot render — the same
/// capability contract `fuse_cached_attention: false` already gives
/// wgpu/cuda for `CachedAttention`.
#[cfg(feature = "reduce-epilogue-fusion")]
pub(super) fn reduce_epilogue_fusion(
    mut resolved: Vec<BoundOp>,
    outputs: &[NodeId],
    numeric_policy: NumericPolicy,
) -> Result<Vec<BoundOp>, TensorError> {
    for _ in 0..resolved.len() {
        let candidates = reduce_epilogue_candidates(&resolved, outputs);
        if candidates.is_empty() {
            return Ok(resolved);
        }
        let by_node: BTreeMap<NodeId, &BoundOp> =
            resolved.iter().map(|bound| (bound.node, bound)).collect();
        let mut fused_by_consumer: BTreeMap<NodeId, BoundOp> = BTreeMap::new();
        let mut absorbed: BTreeSet<NodeId> = BTreeSet::new();
        for (consumer, source) in candidates {
            if absorbed.contains(&consumer) || absorbed.contains(&source) {
                continue; // already spoken for by another pair this same round.
            }
            let Some(reduce_bound) = by_node.get(&source).copied() else {
                continue;
            };
            let Some(consumer_bound) = by_node.get(&consumer).copied() else {
                continue;
            };
            let BoundOpKind::Reduce {
                element_body,
                reduce_op,
                init,
                keep,
                operands,
                output_axes,
                out_layout,
                out_scatter: None,
                epilogue_body: inner_epilogue_body,
                epilogue_operands: inner_epilogue_operands,
                ..
            } = &reduce_bound.kind
            else {
                continue; // window-elimination or a prior pass already rewrote this reduce away.
            };
            let Some(broadcast_axes) = epilogue_broadcast_axes_for(
                output_axes,
                &reduce_bound.extents,
                &consumer_bound.extents,
            ) else {
                continue; // (a): consumer must either preserve the fold's own output shape or re-broadcast the WHOLE pre-reduction shape, nothing in between.
            };
            let BoundOpKind::Elementwise {
                operands: consumer_operands,
                ..
            } = &consumer_bound.kind
            else {
                continue;
            };
            let Some((_, source_layout, source_gather)) = consumer_operands
                .iter()
                .find(|(node, _, _)| *node == source)
            else {
                continue;
            };
            if source_gather.is_some()
                || !reads_reduce_output_identically(source_layout, out_layout, output_axes)
            {
                continue; // (a): the fold's own output must be read at genuine identity/broadcast, never gathered or permuted.
            }
            let Some((epilogue_body, epilogue_operands)) = compose_reduce_epilogue(
                output_axes,
                reduce_bound.extents.len(),
                inner_epilogue_body,
                inner_epilogue_operands,
                consumer_bound,
                source,
                numeric_policy,
            ) else {
                continue;
            };
            // Metal exposes buffer indices 0..=30. Count the signature this
            // fused op actually produces rather than capping only its
            // epilogue: fold operands, epilogue operands, one index buffer
            // per gather, output, uniforms, and the shared gather-fault
            // buffer. If it does not fit, leaving the consumer materialized
            // preserves the same algebra with two legal kernels.
            let buffer_binding_count = metal_buffer_binding_count(operands, &epilogue_operands);
            if buffer_binding_count > 31 {
                continue;
            }
            let fused = BoundOp {
                node: consumer,
                dtype: consumer_bound.dtype,
                extents: reduce_bound.extents.clone(),
                kind: BoundOpKind::Reduce {
                    element_body: element_body.clone(),
                    reduce_op: *reduce_op,
                    init: *init,
                    keep: *keep,
                    operands: operands.clone(),
                    output_axes: output_axes.clone(),
                    out_layout: out_layout.clone(),
                    out_scatter: None,
                    epilogue_body,
                    epilogue_operands,
                    epilogue_broadcast_axes: broadcast_axes,
                },
            };
            fused_by_consumer.insert(consumer, fused);
            absorbed.insert(source);
        }
        if fused_by_consumer.is_empty() {
            return Ok(resolved);
        }
        let mut rewritten = Vec::with_capacity(resolved.len());
        for bound in resolved {
            if let Some(fused) = fused_by_consumer.remove(&bound.node) {
                rewritten.push(fused);
            } else if !absorbed.contains(&bound.node) {
                rewritten.push(bound);
            }
        }
        resolved = rewritten;
    }
    Ok(resolved)
}

pub(super) fn metal_buffer_binding_count(operands: &BoundOperands, epilogue: &BoundOperands) -> usize {
    let gather_count = operands
        .iter()
        .chain(epilogue.iter())
        .filter(|(_, _, gather)| gather.is_some())
        .count();
    operands.len() + epilogue.len() + gather_count + 2 + usize::from(gather_count > 0)
}

/// Which [`BoundOpKind::Reduce::epilogue_broadcast_axes`] value `consumer`'s
/// own fusion needs, or `None` to reject the whole candidate: `Some(empty)`
/// when `consumer_extents` already equals the fold's own OUTPUT shape (the
/// pre-existing, shape-preserving PLAIN epilogue — `output_axes`'s own
/// projection of `reduce_extents`, no axis to re-broadcast over); `Some` of
/// every axis `output_axes` excludes when `consumer_extents` equals the
/// fold's FULL pre-reduction shape instead (the broadcast-reduce shape an
/// RMSNorm-style `x * inv_rms` tail needs); `None` for anything else (a
/// shape this rule was never meant to admit — e.g. a consumer that reads a
/// PARTIAL sub-broadcast of the reduced axes).
#[cfg(feature = "reduce-epilogue-fusion")]
pub(super) fn epilogue_broadcast_axes_for(
    output_axes: &[u16],
    reduce_extents: &[u64],
    consumer_extents: &[u64],
) -> Option<SmallVec<[u16; MAX_INLINE_RANK]>> {
    let projected: Vec<u64> = output_axes
        .iter()
        .map(|&axis| reduce_extents[axis as usize])
        .collect();
    if consumer_extents == projected.as_slice() {
        return Some(SmallVec::new());
    }
    if consumer_extents == reduce_extents && reduce_extents.len() > output_axes.len() {
        let broadcast_axes = (0..reduce_extents.len() as u16)
            .filter(|axis| !output_axes.contains(axis))
            .collect();
        return Some(broadcast_axes);
    }
    None
}

/// Is `consumer_layout` a genuine identity-or-broadcast read of `source`'s
/// own materialized output — same per-axis stride as `out_layout` on every
/// `output_axes` entry, and stride `0` (a true broadcast, never a permuted
/// or reversed walk) on every OTHER axis `consumer_layout` names. Rejects
/// exactly the shape `a_strided_consumer_map_does_not_fuse` proves: a
/// same-SHAPE but reversed/offset read (`coeff: -1`) has the right rank and
/// element count but the WRONG stride, so [`epilogue_broadcast_axes_for`]'s
/// shape-only check alone would wrongly admit it.
#[cfg(feature = "reduce-epilogue-fusion")]
pub(super) fn reads_reduce_output_identically(
    consumer_layout: &Layout,
    out_layout: &Layout,
    output_axes: &[u16],
) -> bool {
    if consumer_layout.base != out_layout.base {
        return false;
    }
    if consumer_layout.strides.len() == output_axes.len() {
        // A plain (shape-preserving) read: `consumer_layout` is compact,
        // rank `output_axes.len()`, LOCAL-indexed in `output_axes`'s own
        // order — compare position `index` against `out_layout`'s (full-
        // rank) stride at the GLOBAL axis `output_axes[index]` names.
        return output_axes
            .iter()
            .enumerate()
            .all(|(index, &axis)| consumer_layout.stride(index as u16) == out_layout.stride(axis));
    }
    // A broadcast-reduce read: `consumer_layout` is full rank, GLOBAL-indexed
    // the same as `out_layout` itself — every `output_axes` entry must carry
    // `out_layout`'s own real stride, every OTHER axis must be a genuine
    // broadcast (stride `0`), never a permuted or reversed walk.
    (0..consumer_layout.strides.len() as u16).all(|axis| {
        if output_axes.contains(&axis) {
            consumer_layout.stride(axis) == out_layout.stride(axis)
        } else {
            consumer_layout.stride(axis) == 0
        }
    })
}

/// Re-addresses `layout` (recorded at `output_axes.len()` rank, the reduce's
/// own OUTPUT-axis coordinate space) into `full_rank` coordinates: every
/// axis in `output_axes` keeps its own stride at its real position, every
/// OTHER axis (the reduce's own reduced axis, among others) gets stride `0`
/// — a genuine broadcast, since the fold's OWN result never varied along
/// that axis in the first place. A no-op in the common case
/// (`layout.strides.len() == full_rank` already) because a PLAIN, non-
/// broadcast prior epilogue's own operands are already recorded at the same
/// rank the fold's `extents` always carries.
#[cfg(feature = "reduce-epilogue-fusion")]
pub(super) fn broadcast_extend_operand(layout: &Layout, output_axes: &[u16], full_rank: usize) -> Layout {
    if layout.strides.len() == full_rank {
        return layout.clone();
    }
    let mut strides = SmallVec::<[i64; MAX_INLINE_RANK]>::from_elem(0, full_rank);
    for (index, &axis) in output_axes.iter().enumerate() {
        if let Some(&stride) = layout.strides.get(index) {
            strides[axis as usize] = stride;
        }
    }
    Layout {
        base: layout.base,
        strides,
    }
}

/// Grafts `consumer`'s own body onto `source`'s fold, composing through
/// whatever epilogue `source` already carries (`inner_epilogue_body`/
/// `inner_epilogue_operands` — the default identity leaf over zero operands
/// on a fold's first fusion, per [`BoundOpKind::Reduce::epilogue_body`]'s own
/// "no epilogue" convention, or a real prior epilogue on a SECOND round —
/// see [`reduce_epilogue_fusion`]'s own doc for why RMSNorm needs both).
/// `full_rank` is the FUSED op's own `extents.len()` (always `source`'s own
/// pre-reduction rank, per [`BoundOp::extents`]'s own doc); every inner
/// operand is broadcast-extended to it via [`broadcast_extend_operand`] so a
/// broadcast-reduce round (`consumer`'s own extents equal to `full_rank`,
/// e.g. RMSNorm's `[s, d]`) and a plain round (`consumer`'s own extents equal
/// to `output_axes`'s smaller projected shape) compose identically: both
/// [`Layout`]s an executor reads are already expressed in the SAME
/// coordinate space `consumer` itself walks, so [`crate::cpu::apply_body`]
/// never needs to know which round produced them.
#[cfg(feature = "reduce-epilogue-fusion")]
pub(super) fn compose_reduce_epilogue(
    output_axes: &[u16],
    full_rank: usize,
    inner_epilogue_body: &ComposedBody,
    inner_epilogue_operands: &BoundOperands,
    consumer: &BoundOp,
    source: NodeId,
    numeric_policy: NumericPolicy,
) -> Option<(ComposedBody, BoundOperands)> {
    let BoundOpKind::Elementwise {
        body: outer_body,
        operands: outer_operands,
    } = &consumer.kind
    else {
        return None;
    };
    // Every slot naming `source`, not just the first — [`find_epilogue_source`]
    // already guarantees any repeat is the SAME projection (production SiLU
    // reads `gate` once bare, once inside `exp(-gate)`, both slots naming the
    // same source), so every one of them, not only the first, must be
    // redirected to the fold's own implicit result below. Leaving a later
    // occurrence pointed at `source` would reference a producer this fusion
    // is about to remove (`source` is folded into `absorbed`, never emitted
    // as its own `BoundOp`), silently reading a buffer that no longer exists.
    let source_indices: Vec<usize> = outer_operands
        .iter()
        .enumerate()
        .filter(|(_, (node, _, _))| *node == source)
        .map(|(index, _)| index)
        .collect();
    if source_indices.is_empty() {
        return None;
    }

    let mut new_operands = BoundOperands::new();
    let mut outer_remap: Vec<u16> = vec![0; outer_operands.len()];
    for (index, operand) in outer_operands.iter().enumerate() {
        if source_indices.contains(&index) {
            continue; // replaced below by the inner fold's own implicit result.
        }
        outer_remap[index] = new_operands.len() as u16;
        new_operands.push(operand.clone());
    }
    let outer_len = new_operands.len();

    let inner_remap: Vec<u16> = (0..inner_epilogue_operands.len())
        .map(|index| (outer_len + index) as u16)
        .collect();
    for (node, layout, gather) in inner_epilogue_operands {
        new_operands.push((
            *node,
            broadcast_extend_operand(layout, output_axes, full_rank),
            gather.clone(),
        ));
    }
    let raw_fold_slot = new_operands.len() as u16;
    let inner_step_count = inner_epilogue_body.steps.len() as u16;

    // Every BodyStep this graft produces mints through `push_canonical_step`
    // (`proxima-tensor/src/bind.rs`'s own one mint point), not a bare
    // `steps.push(BodyStep { .. })` — a remap can flip an arg from
    // `StepArg::Operand` to `StepArg::Step` (the fold's own implicit result
    // taking the place of a raw operand read), which can leave a commutative
    // step's args in non-canonical order even though both `inner_epilogue_body`
    // and `outer_body` were themselves minted canonically before this graft
    // ever saw them. No constant table survives into this post-composition
    // pass (`reduce_epilogue_fusion` runs over already-`BoundOp`-resolved
    // data, not the original `Op` program), so identity elimination never
    // fires here regardless of `numeric_policy` (an empty `values` slice
    // makes `step_arg_constant` always return `None`) — harmless, since a
    // remap only changes which slot an arg names, never introduces a new
    // literal identity value, so `push_canonical_step` always appends exactly
    // one step here and the `inner_step_count`/`outer_remap` index arithmetic
    // below still lines up with the pushed order. `numeric_policy` is still
    // threaded through (rather than hard-coding a policy here) so this call
    // site tracks whatever a future constant-aware version of this graft
    // would need, instead of silently diverging from the caller's own
    // policy.
    let constants = Constants {
        ones: &[],
        values: &[],
        numeric_policy,
    };
    let mut steps: Vec<BodyStep> = Vec::new();
    let mut absorbed: Vec<NodeId> = Vec::new();
    let mut state = ComposeState {
        steps: &mut steps,
        operands: &mut new_operands,
        absorbed: &mut absorbed,
    };
    for step in &inner_epilogue_body.steps {
        let args = step
            .args
            .iter()
            .map(|arg| match arg {
                StepArg::Operand(index) => {
                    let old = *index as usize;
                    if old == inner_epilogue_operands.len() {
                        StepArg::Operand(raw_fold_slot)
                    } else {
                        StepArg::Operand(inner_remap[old])
                    }
                }
                StepArg::Step(step_index) => StepArg::Step(*step_index),
            })
            .collect();
        push_canonical_step(&mut state, step.op, args, constants);
    }
    for step in &outer_body.steps {
        let args = step
            .args
            .iter()
            .map(|arg| match arg {
                StepArg::Operand(index) => {
                    let old = *index as usize;
                    if source_indices.contains(&old) {
                        StepArg::Step(inner_step_count - 1)
                    } else {
                        StepArg::Operand(outer_remap[old])
                    }
                }
                StepArg::Step(step_index) => StepArg::Step(*step_index + inner_step_count),
            })
            .collect();
        push_canonical_step(&mut state, step.op, args, constants);
    }
    Some((ComposedBody { steps }, new_operands))
}

/// The single choke point every [`bind`]/[`bind_with_fusion`]/
/// [`bind_without_reduce_epilogue_fusion`] route eventually calls (ROW 541,
/// `docs/discipline.md`). Binds only the ops [`live::reachable`] reaches
/// from `outputs` through operands, gather indices, and reduce `out_map`
/// indices — a program built for a wider caller (a shared spec module
/// producing both a prefill and a decode graph, say) never dispatches the
/// prefill-only tail decode's own `outputs` do not reach. Node ids stay
/// exactly [`program`]'s own positions: an unreachable position is skipped
/// via [`BoundOpBuilder::skip`], never renumbered, since every backward
/// reference elsewhere in the program is a raw index into this same slice.
pub(super) fn bind_plain(
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
    numeric_policy: NumericPolicy,
) -> Result<Vec<BoundOp>, TensorError> {
    let retires = live::annotate(program, outputs);
    let reachable = live::reachable(program, outputs);
    let building = BoundOpBuilder::new(retires, numeric_policy);
    let mut built = Vec::new();
    for (position, expr) in program.iter().enumerate() {
        if reachable.contains(&NodeId(position as u32)) {
            built.extend(building.push(expr, shapes)?);
        } else {
            building.skip();
        }
    }
    built.extend(building.finish(shapes)?);
    Ok(built)
}

/// Every `program` position holding an [`Op::Input`] — the block-input node
/// order [`bind`]'s own caller binds real data against. Backend-neutral (a
/// pure scan over `&[Op]`), so any executor consuming this module's
/// [`BoundOp`]s reads the SAME node order [`crate::cpu`]'s own evaluators do
/// rather than re-deriving it — before this function was `pub`, `cpu.rs` and
/// `omega/src/metal.rs` each carried a byte-identical private copy (the
/// exact "second, parallel emitter" [`BoundOp::dtype`]'s own doc says this
/// module exists to avoid).
#[must_use]
pub fn block_node_ids(program: &[Op]) -> Vec<NodeId> {
    program
        .iter()
        .enumerate()
        .filter(|(_, expr)| matches!(expr, Op::Input { .. }))
        .map(|(position, _)| NodeId(position as u32))
        .collect()
}

/// Every node referenced as a gather's `indices` anywhere in `program` — the
/// one class of non-float32 node a float-only executor's own dtype gate must
/// exempt (an index value is an exact integer carried in a float buffer, per
/// [`crate::map::IndexMap::Computed`]'s own doc). Same reuse argument as
/// [`block_node_ids`]: this was a byte-identical private copy in both
/// `cpu.rs` and `omega/src/metal.rs`.
#[must_use]
pub fn index_node_ids(program: &[Op]) -> BTreeSet<NodeId> {
    let mut nodes = BTreeSet::new();
    for expr in program {
        match expr {
            Op::Input { .. } | Op::Iota { .. } | Op::Constant { .. } => {}
            Op::Elementwise { operands, .. } => {
                for (_, map) in operands {
                    push_indices_node(map, &mut nodes);
                }
            }
            Op::Reduce(reduce) => {
                push_indices_node(&reduce.in_map, &mut nodes);
                push_indices_node(&reduce.out_map, &mut nodes);
            }
        }
    }
    nodes
}

/// `pub(crate)`, not private: [`crate::cpu`]'s own `referenced_node_ids`
/// walks the identical `Elementwise`/`Reduce` operand-map shape as
/// [`index_node_ids`] above, over a different node set, so it shares this
/// helper rather than carrying a third copy.
pub(crate) fn push_indices_node(map: &IndexMap, nodes: &mut BTreeSet<NodeId>) {
    if let IndexMap::Computed { indices, .. } = map {
        nodes.insert(*indices);
    }
}

/// Per-node retire sets over the *emitted* (post-fusion) node sequence:
/// `result[p]` is every node whose last read is `resolved[p]`. Distinct from
/// [`live::annotate`], which computes liveness over the PROGRAM's own
/// timeline before fusion has decided which zips never materialize at all —
/// this one runs after [`bind`], over [`BoundOp::operands`] directly, so it
/// sees the fused shape an executor actually walks. Same reuse argument as
/// [`block_node_ids`]/[`index_node_ids`]: `cpu.rs`'s private `node_retirement`
/// and `omega/src/metal.rs`'s private `bound_op_retirement` were the
/// identical computation under two names.
/// The one read-source walk both [`node_retirement`] and [`node_last_reader`]
/// need, so the two never drift into "identical computation under two names"
/// (the exact defect this module's own doc says it exists to end).
pub(super) fn walk_last_reads<F: FnMut(NodeId, usize)>(resolved: &[BoundOp], mut record: F) {
    for (position, node) in resolved.iter().enumerate() {
        for (source, _, gather) in node.all_read_sources() {
            record(*source, position);
            if let Some(gather_access) = gather {
                record(gather_access.indices, position);
            }
        }
        if let BoundOpKind::Reduce {
            out_scatter: Some(lookup),
            ..
        } = &node.kind
        {
            record(lookup.indices, position);
        }
    }
}

#[must_use]
pub fn node_retirement(resolved: &[BoundOp], outputs: &[NodeId]) -> Vec<Vec<NodeId>> {
    let outputs: BTreeSet<NodeId> = outputs.iter().copied().collect();
    let mut last_use: BTreeMap<NodeId, usize> = BTreeMap::new();
    walk_last_reads(resolved, |node, position| {
        last_use.insert(node, position);
    });

    let mut retires = vec![Vec::new(); resolved.len()];
    for (node, position) in last_use {
        if !outputs.contains(&node) {
            #[cfg(feature = "instrument")]
            debug!(
                node = node.0,
                kind = "node_retirement",
                decision = "retired",
                into = resolved[position].node.0,
                "node retired -- last consumer read it at this resolved position"
            );
            retires[position].push(node);
        }
    }
    retires
}

/// Dense node -> last-reader-position table over the same emitted sequence
/// [`node_retirement`] walks, indexed by `NodeId.0`, `u32::MAX` where a node
/// is never read. An executor's per-op retirement decision is then a single
/// array read (`last_reader[node] == position`) instead of a scan over
/// remaining ops or the current op's own operand list — see
/// `omega/src/metal.rs`'s `execute_plan_inner` for the consumer this replaced
/// a per-op `iter().any()` forward scan in.
#[must_use]
pub fn node_last_reader(resolved: &[BoundOp], node_count: usize) -> Vec<u32> {
    let mut last_reader = vec![u32::MAX; node_count];
    walk_last_reads(resolved, |node, position| {
        last_reader[node.0 as usize] = position as u32;
    });
    last_reader
}

