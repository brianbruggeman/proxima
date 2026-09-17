use super::*;

pub fn bind(
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
    numeric_policy: NumericPolicy,
) -> Result<Vec<BoundOp>, TensorError> {
    bind_with_fusion(program, shapes, outputs, true, numeric_policy)
}

/// Binds the graph with cached-attention/chain fusion but leaves reduction
/// epilogues as separate operations for backends whose lowering does not yet
/// support broadcast epilogue operands. This is a capability boundary, not a
/// numerical relaxation: the returned graph is the unfused correct form.
pub fn bind_without_reduce_epilogue_fusion(
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
    fuse_cached_attention: bool,
    numeric_policy: NumericPolicy,
) -> Result<Vec<BoundOp>, TensorError> {
    admit(numeric_policy, NumericRewrite::IdentityElimination)?;
    admit(numeric_policy, NumericRewrite::ChainFusion)?;
    let built = bind_cached_attention_fusion(
        program,
        shapes,
        outputs,
        fuse_cached_attention,
        numeric_policy,
    )?;
    Ok(built)
}

/// Same as [`bind`], but `fuse_cached_attention` states whether the caller's
/// backend can render [`BoundOpKind::CachedAttention`] at all. `cpu.rs` and
/// `omega/src/metal.rs` render the fused kind, so they (via [`bind`]) pass
/// `true`; `omega`'s wgpu and cuda drivers have no renderer for it yet, so
/// they call this directly with `false` — the fused rewrite never fires for
/// them, and the plain elementwise/reduce chain `bind_plain` already
/// produces is what they emit.
///
/// `reduce-epilogue-fusion` (the `BoundOpKind::Reduce::epilogue_body`/
/// `epilogue_operands` rewrite) runs unconditionally after this, gated only
/// by the crate feature — it has no per-call capability bool of its own
/// because, unlike cached-attention, every renderer this crate ships either
/// renders the epilogue or rejects it at bind time (see this module's own
/// `reduce_epilogue_fusion`, private and feature-gated); there is no third
/// "silently ignore it" caller to protect the way `fuse_cached_attention: false`
/// protects wgpu/cuda from a fused kind they cannot render at all.
///
/// `numeric_policy` is the [`NumericPolicy`] every bit-changing rewrite this
/// function fires must clear via [`admit`] before it runs. The three
/// rewrites shipped today (identity elimination, chain fusion,
/// reduce-epilogue fusion) are classified [`NumericRewrite`]s whose
/// [`NumericRewrite::required_permissions`] is [`NumericPolicy::bit_exact()`],
/// so a call under the default policy always clears — nothing regresses. A
/// future reassociating rewrite in this crate declares its own
/// [`NumericRewrite`] variant and is admitted the same way.
pub fn bind_with_fusion(
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
    fuse_cached_attention: bool,
    numeric_policy: NumericPolicy,
) -> Result<Vec<BoundOp>, TensorError> {
    // The three rewrites this crate ships unconditionally today are
    // bit-exact by construction (identity elimination, chain fusion --
    // both inside `bind_cached_attention_fusion`'s own `bind_plain` --
    // and reduce-epilogue fusion below), so `admit` always clears at
    // `NumericPolicy::default()`; the call is the explicit, testable
    // declaration of that fact, not a behavior change (`op.rs:107`'s
    // `is_associative` has no such caller today).
    admit(numeric_policy, NumericRewrite::IdentityElimination)?;
    admit(numeric_policy, NumericRewrite::ChainFusion)?;
    #[cfg(feature = "std")]
    let fuse_cached_attention = fuse_cached_attention
        && std::env::var_os("PROXIMA_DISABLE_CACHED_ATTENTION_FUSION").is_none();
    let built = bind_cached_attention_fusion(
        program,
        shapes,
        outputs,
        fuse_cached_attention,
        numeric_policy,
    )?;
    #[cfg(feature = "instrument")]
    debug!(
        stage = "after_cached_attention_fusion",
        cached_attention_count = built
            .iter()
            .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
            .count() as u64,
        "bind_with_fusion: fused-op-kind count per stage, catches a later stage silently discarding an earlier fusion"
    );
    #[cfg(feature = "gated-delta-net-fusion")]
    #[cfg(feature = "std")]
    let gated_delta_net_fusion_disabled =
        std::env::var_os("PROXIMA_DISABLE_GATED_DELTA_NET_FUSION").is_some();
    #[cfg(feature = "gated-delta-net-fusion")]
    #[cfg(not(feature = "std"))]
    let gated_delta_net_fusion_disabled = false;
    #[cfg(feature = "gated-delta-net-fusion")]
    let built = if gated_delta_net_fusion_disabled {
        built
    } else {
        apply_gated_delta_net_fusion(
            built,
            program,
            shapes,
            outputs,
            fuse_cached_attention,
            numeric_policy,
        )?
    };
    #[cfg(feature = "moe-topk-fusion")]
    #[cfg(feature = "std")]
    let moe_topk_fusion_disabled = std::env::var_os("PROXIMA_DISABLE_MOE_TOPK_FUSION").is_some();
    #[cfg(feature = "moe-topk-fusion")]
    #[cfg(not(feature = "std"))]
    let moe_topk_fusion_disabled = false;
    #[cfg(feature = "moe-topk-fusion")]
    let built = if moe_topk_fusion_disabled {
        built
    } else {
        apply_moe_topk_fusion(built, program, shapes, outputs)?
    };
    #[cfg(feature = "reduce-epilogue-fusion")]
    {
        admit(numeric_policy, NumericRewrite::ReduceEpilogueFusion)?;
        #[cfg(feature = "std")]
        if std::env::var_os("PROXIMA_DISABLE_REDUCE_EPILOGUE_FUSION").is_some() {
            return Ok(built);
        }
        let epilogued = reduce_epilogue_fusion(built, outputs, numeric_policy)?;
        #[cfg(feature = "instrument")]
        debug!(
            stage = "after_reduce_epilogue_fusion",
            cached_attention_count = epilogued
                .iter()
                .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
                .count() as u64,
            "bind_with_fusion: fused-op-kind count per stage, catches a later stage silently discarding an earlier fusion"
        );
        Ok(epilogued)
    }
    #[cfg(not(feature = "reduce-epilogue-fusion"))]
    Ok(built)
}

/// Binary [`Op::Elementwise`] lookup, gated on this crate's own
/// `gated-delta-net-fusion` feature — a small duplicate of
/// [`binary_elementwise`] (that helper lives behind
/// `cached-attention-streaming` instead) rather than a shared function two
/// independent feature gates would both have to enable to compile.
/// A zero-based, row-major (last axis fastest) [`Layout`] over `extents` --
/// what [`Op::Input`]'s own storage always is, and what
/// [`gated_delta_net_candidates`] binds `query`/`key` to directly instead of
/// borrowing a `repeat_kv_heads` broadcast consumer's own stride-0 read (this
/// module's own doc on [`GatedDeltaNetMatch::query_was_repeated`]).
#[cfg(feature = "gated-delta-net-fusion")]
pub(super) fn natural_layout(extents: &[u64]) -> Layout {
    let mut strides = SmallVec::<[i64; MAX_INLINE_RANK]>::new();
    strides.resize(extents.len(), 0);
    let mut running = 1_i64;
    for (axis, extent) in extents.iter().enumerate().rev() {
        strides[axis] = running;
        running *= *extent as i64;
    }
    Layout { base: 0, strides }
}

/// Drops a genuine leading extent-1 axis, if `shape` has one and still has a
/// non-empty tail -- [`gated_delta_net_candidates`]'s own doc on why
/// `query`/`key` still carry the decode step's own size-1 token axis after
/// `gdn_unwrap_decode_squeeze` walks past the `repeat_kv_heads` squeeze.
#[cfg(feature = "gated-delta-net-fusion")]
pub(super) fn strip_leading_unit_axis(shape: &[u64]) -> &[u64] {
    match shape {
        [1, rest @ ..] if !rest.is_empty() => rest,
        _ => shape,
    }
}

#[cfg(feature = "gated-delta-net-fusion")]
pub(super) fn gdn_binary_elementwise(
    program: &[Op],
    node: NodeId,
    body: ScalarOp,
) -> Option<[NodeId; 2]> {
    match program.get(node.0 as usize)? {
        Op::Elementwise {
            body: actual_body,
            operands,
            ..
        } if *actual_body == body => match operands.as_slice() {
            [(left, _), (right, _)] => Some([*left, *right]),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(feature = "gated-delta-net-fusion")]
pub(super) fn gdn_unary_elementwise(
    program: &[Op],
    node: NodeId,
    body: ScalarOp,
) -> Option<NodeId> {
    match program.get(node.0 as usize)? {
        Op::Elementwise {
            body: actual_body,
            operands,
            ..
        } if *actual_body == body => match operands.as_slice() {
            [(source, _)] => Some(*source),
            _ => None,
        },
        _ => None,
    }
}

/// The same [`Op::Constant`] read [`cached_attention_candidates`]'s own
/// `scale: f32` field relies on -- `inv_sqrt_key_dim` is baked at graph-build
/// time (`1/sqrt(head_k_dim)`, a compile-time constant of the model's own
/// architecture), so [`BoundOpKind::GatedDeltaNet::inv_sqrt_key_dim`] is a
/// plain `f32`, not a bound operand.
#[cfg(feature = "gated-delta-net-fusion")]
pub(super) fn gdn_constant_value(program: &[Op], node: NodeId) -> Option<f32> {
    match program.get(node.0 as usize)? {
        Op::Constant { value, .. } => Some(*value),
        _ => None,
    }
}

#[cfg(feature = "gated-delta-net-fusion")]
pub(super) fn gdn_reduced_source(program: &[Op], node: NodeId, body: ScalarOp) -> Option<NodeId> {
    match program.get(node.0 as usize)? {
        Op::Reduce(reduce)
            if reduce.body == body
                && reduce.init == ReduceInit::Zero
                && reduce.keep == Keep::Reduce =>
        {
            Some(reduce.operand)
        }
        _ => None,
    }
}

/// `true` when `node` is [`crate::spec::repeat_kv_heads`]'s own output shape:
/// an elementwise `Multiply` against a rank-`>=1` all-ones [`Op::Constant`].
/// Reused, not restated, from that function's own doc: the donor is what
/// makes `shape::unify_iteration_space` resolve the broadcast group axis at
/// all, so its value is always exactly `1.0`.
#[cfg(feature = "gated-delta-net-fusion")]
pub(super) fn gdn_is_repeat_kv_heads_donor(program: &[Op], node: NodeId) -> bool {
    matches!(
        program.get(node.0 as usize),
        Some(Op::Constant { value, .. }) if *value == 1.0
    )
}

/// Walks past a [`crate::spec::repeat_kv_heads`] broadcast if `node` is one,
/// returning the pre-repeat source otherwise unchanged — the one place this
/// matcher intentionally disagrees with [`crate::spec::append_qwen35_delta_net_step`]'s
/// own physical operand and instead binds what llama.cpp's fused Metal kernel
/// reads directly (`gated_delta_net.metal:33-34`'s `i01 = i21 % ne01`
/// mod-broadcast), dropping the eager repeat from the hot path entirely (this
/// module's own doc on [`BoundOpKind::GatedDeltaNet`]).
#[cfg(feature = "gated-delta-net-fusion")]
pub(super) fn gdn_unwrap_repeat_kv_heads(program: &[Op], node: NodeId) -> NodeId {
    match gdn_binary_elementwise(program, node, ScalarOp::Multiply) {
        Some([left, right]) if gdn_is_repeat_kv_heads_donor(program, right) => left,
        Some([left, right]) if gdn_is_repeat_kv_heads_donor(program, left) => right,
        _ => node,
    }
}

/// The all-ones donor [`gdn_unwrap_repeat_kv_heads`] walked past, if `node`
/// was a repeat -- `repeat_kv_heads` mints a fresh `Op::Constant` per call
/// (never shared across the query/key repeat sites), so once the multiply
/// that reads it is absorbed this donor has no other consumer and would
/// otherwise linger as a dead leaf in the rewritten program.
#[cfg(feature = "gated-delta-net-fusion")]
pub(super) fn gdn_repeat_kv_heads_donor(program: &[Op], node: NodeId) -> Option<NodeId> {
    match gdn_binary_elementwise(program, node, ScalarOp::Multiply) {
        Some([_, right]) if gdn_is_repeat_kv_heads_donor(program, right) => Some(right),
        Some([left, _]) if gdn_is_repeat_kv_heads_donor(program, left) => Some(left),
        _ => None,
    }
}

/// Walks past `append_qwen35_ssm_mixer_with_taps_and_layout`'s own
/// "squeeze the size-1 decode-step `s` axis away" reduce
/// (`spec.rs:8737-8786`) if `node` is one, returning `node`'s own pre-squeeze
/// operand instead -- a plain [`ScalarOp::Add`]/[`ReduceInit::Zero`]/
/// [`Keep::Reduce`] fold whose `in_map` reads its operand through a genuine
/// identity (no permutation, no broadcast: operand axis `p` addresses
/// iteration axis `p`) and whose `out_map` is a pure projection (every axis a
/// single coeff-1 term, no offset) that keeps every iteration axis except
/// exactly one, and that one axis's own extent (read off `shapes`) is `1`.
/// Every other shape returns `node` unchanged rather than guess. This is the
/// gap [`gated_delta_net_candidates`]'s own doc names: the real program
/// threads `query`/`key`/`value`/`gate`/`beta` through this exact squeeze
/// between the algebra's own `u,g`-split construction and
/// [`append_qwen35_delta_net_step`], and unwrapping it is what lets this
/// matcher bind the program's own natural, pre-squeeze storage order instead
/// of the squeeze's own re-lettered output.
#[cfg(feature = "gated-delta-net-fusion")]
pub(super) fn gdn_unwrap_decode_squeeze(program: &[Op], shapes: &Shapes, node: NodeId) -> NodeId {
    let Some(Op::Reduce(reduce)) = program.get(node.0 as usize) else {
        return node;
    };
    if reduce.body != ScalarOp::Add
        || reduce.init != ReduceInit::Zero
        || reduce.keep != Keep::Reduce
        || reduce.in_map.is_data_dependent()
        || reduce.out_map.is_data_dependent()
    {
        return node;
    }
    let operand_extents = shapes.of(reduce.operand);
    let in_pattern = reduce.in_map.affine();
    let is_identity = in_pattern.axes.len() == operand_extents.len()
        && in_pattern.axes.iter().enumerate().all(|(axis, index)| {
            index.offset == 0
                && matches!(index.terms.as_slice(), [term] if term.coeff == 1 && term.axis as usize == axis)
        });
    if !is_identity {
        return node;
    }
    let out_pattern = reduce.out_map.affine();
    let mut kept_axes = SmallVec::<[u16; MAX_INLINE_RANK]>::new();
    for index in &out_pattern.axes {
        match index.terms.as_slice() {
            [term] if term.coeff == 1 && index.offset == 0 => kept_axes.push(term.axis),
            _ => return node,
        }
    }
    let dropped: SmallVec<[u16; MAX_INLINE_RANK]> = (0..operand_extents.len() as u16)
        .filter(|axis| !kept_axes.contains(axis))
        .collect();
    match dropped.as_slice() {
        [only] if operand_extents.get(*only as usize) == Some(&1) => reduce.operand,
        _ => node,
    }
}

/// One matched [`append_qwen35_delta_net_step`](crate::spec::append_qwen35_delta_net_step)
/// recurrence, structurally recognized by walking backward from its `out`
/// node through the exact `ScalarOp` sequence that function emits — anchored
/// on op shape, never on node names (this module's own convention;
/// [`cached_attention_candidates`] is the standing precedent). Declines
/// (returns nothing for this `output`) rather than guesses on any mismatch,
/// including a perturbed single op in the chain.
#[cfg(feature = "gated-delta-net-fusion")]
pub(super) struct GatedDeltaNetMatch {
    pub(super) query: NodeId,
    pub(super) key: NodeId,
    pub(super) value: NodeId,
    pub(super) gate: NodeId,
    pub(super) beta: NodeId,
    pub(super) state_in: NodeId,
    pub(super) inv_sqrt_key_dim: f32,
    pub(super) state_out: NodeId,
    /// `true` when [`gdn_unwrap_repeat_kv_heads`] actually walked past a
    /// broadcast for `query`/`key` (whether or not a decode-squeeze sat
    /// above it) — every REMAINING consumer of that pre-repeat source reads
    /// it through the broadcast's own stride-0 trailing axis, so
    /// [`gated_delta_net_candidates`] must derive a NATURAL layout from
    /// `query`/`key`'s own extents (dim fastest, this slice's pre-repeat
    /// storage order) instead of borrowing a consumer's `i{head}`-convention
    /// read (dim slowest) the way the other four operands safely do, and the
    /// executor must be told which convention it got
    /// ([`BoundOpKind::GatedDeltaNet`]'s own `query_key_head_stride`/
    /// `query_key_dim_stride` fields).
    pub(super) query_was_repeated: bool,
    pub(super) key_was_repeated: bool,
    /// Every node absorbed into the fused op, `out` and the six leaf sources
    /// excluded — the set [`apply_gated_delta_net_fusion`] drops from the
    /// rewritten program once the fusion actually fires.
    pub(super) absorbed: BTreeSet<NodeId>,
}

#[cfg(feature = "gated-delta-net-fusion")]
pub(super) fn match_gated_delta_net_step(
    program: &[Op],
    shapes: &Shapes,
    out: NodeId,
) -> Option<GatedDeltaNetMatch> {
    let mut absorbed = BTreeSet::new();
    let absorb = |node: NodeId, set: &mut BTreeSet<NodeId>| {
        set.insert(node);
    };

    let out_product = gdn_reduced_source(program, out, ScalarOp::Add)?;
    absorb(out_product, &mut absorbed);
    let [state_out, query_scaled] =
        gdn_binary_elementwise(program, out_product, ScalarOp::Multiply)?;
    absorb(query_scaled, &mut absorbed);

    let [state_decayed_for_update, update] =
        gdn_binary_elementwise(program, state_out, ScalarOp::Add)?;
    absorb(state_out, &mut absorbed);
    absorb(update, &mut absorbed);
    absorb(state_decayed_for_update, &mut absorbed);

    let [key_a, delta] = gdn_binary_elementwise(program, update, ScalarOp::Multiply)?;
    absorb(delta, &mut absorbed);
    let [state_in_a, decay_a] =
        gdn_binary_elementwise(program, state_decayed_for_update, ScalarOp::Multiply)?;

    let [residual, beta_bcast] = gdn_binary_elementwise(program, delta, ScalarOp::Multiply)?;
    absorb(residual, &mut absorbed);
    let [value, value_pred] = gdn_binary_elementwise(program, residual, ScalarOp::Subtract)?;
    absorb(value_pred, &mut absorbed);

    let value_pred_product = gdn_reduced_source(program, value_pred, ScalarOp::Add)?;
    absorb(value_pred_product, &mut absorbed);
    let [state_decayed, key_b] =
        gdn_binary_elementwise(program, value_pred_product, ScalarOp::Multiply)?;
    absorb(state_decayed, &mut absorbed);
    let [state_in_b, decay_b] = gdn_binary_elementwise(program, state_decayed, ScalarOp::Multiply)?;

    if state_in_a != state_in_b || decay_a != decay_b || key_a != key_b {
        return None;
    }
    let gate = gdn_unary_elementwise(program, decay_a, ScalarOp::Exponential)?;
    absorb(decay_a, &mut absorbed);

    let [query, inv_sqrt_key_dim_node] =
        gdn_binary_elementwise(program, query_scaled, ScalarOp::Multiply)?;
    let inv_sqrt_key_dim = gdn_constant_value(program, inv_sqrt_key_dim_node)?;

    // The real program threads every one of these five through its own
    // decode-squeeze reduce before `append_qwen35_delta_net_step` ever sees
    // them (`gdn_unwrap_decode_squeeze`'s own doc); query/key additionally
    // sit behind a `repeat_kv_heads` broadcast UNDER that squeeze, so the
    // squeeze must unwrap first or `gdn_unwrap_repeat_kv_heads` never finds
    // the donor multiply it looks for.
    let key_squeezed = gdn_unwrap_decode_squeeze(program, shapes, key_a);
    if key_squeezed != key_a {
        absorbed.insert(key_a);
    }
    let key = gdn_unwrap_repeat_kv_heads(program, key_squeezed);
    let key_was_repeated = key != key_squeezed;
    if key_was_repeated {
        absorbed.insert(key_squeezed);
        if let Some(donor) = gdn_repeat_kv_heads_donor(program, key_squeezed) {
            absorbed.insert(donor);
        }
    }

    let query_squeezed = gdn_unwrap_decode_squeeze(program, shapes, query);
    if query_squeezed != query {
        absorbed.insert(query);
    }
    let query = gdn_unwrap_repeat_kv_heads(program, query_squeezed);
    let query_was_repeated = query != query_squeezed;
    if query_was_repeated {
        absorbed.insert(query_squeezed);
        if let Some(donor) = gdn_repeat_kv_heads_donor(program, query_squeezed) {
            absorbed.insert(donor);
        }
    }

    Some(GatedDeltaNetMatch {
        query,
        key,
        value,
        gate,
        beta: beta_bcast,
        state_in: state_in_a,
        inv_sqrt_key_dim,
        query_was_repeated,
        key_was_repeated,
        state_out,
        absorbed,
    })
}

/// Scans `program` for [`append_qwen35_delta_net_step`](crate::spec::append_qwen35_delta_net_step)
/// candidates and, for each, resolves its six operand sources' [`Layout`]s
/// against `resolved` — the same technique [`cached_attention_candidates`]
/// uses, and for the same reason: chain fusion may already have inlined an
/// intermediate elementwise op into a reduce's own `element_body`, so the
/// true physical read is whatever `resolved`'s own `operands()` names, not
/// necessarily a node this function's own backward walk stopped at.
///
/// `state_out` is this op's own second output (ROW 547,
/// `docs/discipline.md`) -- requesting it alongside `out` no longer declines
/// the match; the fused kind supplies it directly. Still declines when any
/// OTHER absorbed node (a decode-squeeze reduce, a `repeat_kv_heads` donor,
/// ...) is itself a requested/effective output, since those genuinely
/// disappear from `resolved` once fusion fires.
#[cfg(feature = "gated-delta-net-fusion")]
pub(super) fn gated_delta_net_candidates(
    program: &[Op],
    shapes: &Shapes,
    resolved: &[BoundOp],
    effective_outputs: &[NodeId],
) -> Vec<(BoundOp, BTreeSet<NodeId>)> {
    let mut candidates = Vec::new();
    for output_position in (0..program.len()).rev() {
        let output = NodeId(output_position as u32);
        let Some(found) = match_gated_delta_net_step(program, shapes, output) else {
            continue;
        };
        if found
            .absorbed
            .iter()
            .any(|node| *node != found.state_out && effective_outputs.contains(node))
        {
            // `found.absorbed` now includes the decode-squeeze reduces
            // themselves (`gdn_unwrap_decode_squeeze`'s own doc), which a
            // caller may legitimately request as a standalone diagnostic tap
            // (`SsmMixerTaps::query`/`key`/`value`/`gate`/`beta`) independent
            // of this fusion -- absorbing one out from under such a request
            // would silently delete a node the caller is about to read, the
            // same requested-output guard [`cached_attention_candidates`]'s
            // own `dependencies` check already makes. `state_out` itself is
            // exempt (this function's own doc) -- it is now a genuine second
            // output of the fused op, never dropped.
            #[cfg(feature = "instrument")]
            debug!(
                out = output.0,
                state_out = found.state_out.0,
                "gdn candidate declined: an absorbed node other than state_out is a requested \
                 output"
            );
            continue;
        }
        let source_nodes = [
            (found.query, found.query_was_repeated),
            (found.key, found.key_was_repeated),
            (found.value, false),
            (found.gate, false),
            (found.beta, false),
            (found.state_in, false),
        ];
        let mut operands = Vec::with_capacity(source_nodes.len());
        for (source, bind_natural) in source_nodes {
            // `query`/`key` may sit behind an unwrapped `repeat_kv_heads`
            // broadcast: every remaining consumer of `source` would then read
            // it through that broadcast's own stride-0 trailing axis, not
            // `source`'s own storage -- borrowing a consumer's read here
            // would silently bind a layout the executor's own contiguity
            // check (`cpu.rs`'s `run_gated_delta_net`) then rejects at run
            // time. `source` itself is always a plain natural-order node
            // ([`Op::Input`] or an equally natural `bind_plain` output, and
            // [`apply_gated_delta_net_fusion`] forces it into the planning
            // outputs so it is actually materialized that way), so its own
            // extents fully determine a natural layout whether or not a
            // repeat was actually present.
            if bind_natural {
                operands.push((source, natural_layout(shapes.of(source)), None));
                continue;
            }
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
        // `query`/`key` bind at two DIFFERENT physical conventions depending
        // on whether a `repeat_kv_heads` broadcast was actually unwrapped
        // (`found.query_was_repeated`/`key_was_repeated`, this match's own
        // doc): the program's own pre-repeat storage, `[kv_heads, key_dim]`
        // dim FASTEST, when it was; the `i{head}` consumer convention's own
        // read, `[key_dim, heads]` dim SLOWEST, when it was not (this
        // slice's non-GQA, non-repeated shape only). Both decline this
        // candidate rather than guess if they disagree, and both normalize
        // to `[heads_axis, key_dim]` below so every match arm reads one
        // consistent order regardless of which convention actually bound.
        if found.query_was_repeated != found.key_was_repeated {
            continue;
        }
        let normalize_query_or_key = |shape: &[u64], was_repeated: bool| -> Option<[u64; 2]> {
            match (shape, was_repeated) {
                (&[heads, key_dim], true) => Some([heads, key_dim]),
                (&[key_dim, heads], false) => Some([heads, key_dim]),
                _ => None,
            }
        };
        let Some(key_shape) = normalize_query_or_key(
            strip_leading_unit_axis(shapes.of(found.key)),
            found.key_was_repeated,
        ) else {
            continue;
        };
        let Some(query_shape) = normalize_query_or_key(
            strip_leading_unit_axis(shapes.of(found.query)),
            found.query_was_repeated,
        ) else {
            continue;
        };
        let key_shape = key_shape.as_slice();
        let query_shape = query_shape.as_slice();
        let value_shape = shapes.of(found.value);
        let gate_shape = shapes.of(found.gate);
        let state_shape = shapes.of(found.state_in);
        // This slice's supported shapes: `append_qwen35_delta_net_step`'s own
        // `head` split, either the single-letter axis (`[heads, dim]`, no GQA
        // broadcast) or the real qwen35moe two-letter `head = "ug"` split --
        // `u` = kv group (query/key's own trailing axis, PRE-`repeat_kv_heads`,
        // `gdn_unwrap_repeat_kv_heads`'s own doc), `g` = query heads per group
        // (value/gate/beta/state's own extra trailing axis, since
        // `repeat_kv_heads`'s own doc proves `u`/`g` never collapse into one
        // physical axis). `query`/`key`'s own natural storage (`gdn.rs`'s own
        // struct doc) is `[kv_heads, key_dim]`, dim FASTEST -- the program's
        // own pre-repeat operand order, distinct from `value`/`gate`/`beta`/
        // `state`, whose consumer reads them un-permuted off
        // `append_qwen35_delta_net_step`'s own `j{head}`/`{head}` maps, dim
        // (where present) SLOWEST: value is `[dim, kv_heads, group]`,
        // gate/beta `[kv_heads, group]`, state `[key_dim, value_dim,
        // kv_heads, group]` -- `num_v_heads = kv_heads * group` is exactly
        // the executor's own flat value-head extent
        // (`gdn::GdnPrefillShape::heads`), so this binds the SAME struct the
        // single-axis case already does, group folded in rather than a new
        // field.
        let (kv_heads, group) = match (key_shape, value_shape, gate_shape, state_shape) {
            (
                [kv_heads, key_dim],
                [value_dim, value_kv_heads, group],
                [gate_kv_heads, gate_group],
                [state_key_dim, state_value_dim, state_kv_heads, state_group],
            ) if query_shape == key_shape
                && *kv_heads == *value_kv_heads
                && *kv_heads == *gate_kv_heads
                && *kv_heads == *state_kv_heads
                && *group == *gate_group
                && *group == *state_group
                && *state_key_dim == *key_dim
                && *state_value_dim == *value_dim
                && shapes.of(output) == value_shape =>
            {
                (*kv_heads, *group)
            }
            (
                [heads, key_dim],
                [value_dim, value_heads],
                [gate_heads],
                [state_key_dim, state_value_dim, state_heads],
            ) if query_shape == key_shape
                && *heads == *value_heads
                && *heads == *gate_heads
                && *heads == *state_heads
                && *state_key_dim == *key_dim
                && *state_value_dim == *value_dim
                && shapes.of(output) == value_shape =>
            {
                (*heads, 1)
            }
            _ => continue,
        };
        let head_k_dim = key_shape[1];
        let head_v_dim = value_shape[0];
        let num_v_heads = kv_heads * group;
        // The executor reads `query`/`key` by explicit stride rather than by
        // assuming one fixed axis order (`gdn::GdnPrefillScan`'s own doc) --
        // `key_was_repeated` and `query_was_repeated` agree by construction
        // (declined above otherwise), so one pair of strides serves both.
        let (query_key_head_stride, query_key_dim_stride) = if found.key_was_repeated {
            (head_k_dim, 1)
        } else {
            (1, kv_heads)
        };
        let fused = BoundOp {
            node: output,
            dtype: DType::Float32,
            extents: shapes.of(output).to_vec(),
            kind: BoundOpKind::GatedDeltaNet {
                operands,
                n_tokens: 1,
                kv_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                query_key_head_stride,
                query_key_dim_stride,
                inv_sqrt_key_dim: found.inv_sqrt_key_dim,
                state_out: found.state_out,
            },
        };
        candidates.push((fused, found.absorbed));
    }
    candidates
}

#[cfg(feature = "moe-topk-fusion")]
pub(super) fn moe_binary_elementwise(
    program: &[Op],
    node: NodeId,
    body: ScalarOp,
) -> Option<[NodeId; 2]> {
    match program.get(node.0 as usize)? {
        Op::Elementwise {
            body: actual_body,
            operands,
            ..
        } if *actual_body == body => match operands.as_slice() {
            [(left, _), (right, _)] => Some([*left, *right]),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(feature = "moe-topk-fusion")]
pub(super) fn moe_reduce_operand(
    program: &[Op],
    node: NodeId,
    dtype: DType,
    body: ScalarOp,
    init: ReduceInit,
) -> Option<NodeId> {
    match program.get(node.0 as usize)? {
        Op::Reduce(reduce)
            if reduce.dtype == dtype
                && reduce.body == body
                && reduce.init == init
                && reduce.keep == Keep::Reduce =>
        {
            Some(reduce.operand)
        }
        _ => None,
    }
}

/// Every [`NodeId`] in `program` that reads `node` as one of its own
/// [`Op::dependencies`] -- the forward index [`match_moe_topk`] needs because
/// `append_moe_ffn`'s own per-round chain only ever links
/// backwards from a LATER round's `selection_scores` to an EARLIER round's
/// `mask` (the exclusion `Select`), never the reverse, so finding round
/// `r + 1` from round `r`'s own `mask` is a forward lookup, not a backward
/// one the way every other fusion matcher in this module walks.
#[cfg(feature = "moe-topk-fusion")]
pub(super) fn moe_consumers(program: &[Op]) -> BTreeMap<NodeId, Vec<NodeId>> {
    let mut consumers: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();
    for (position, op) in program.iter().enumerate() {
        let node = NodeId(position as u32);
        for dependency in op.dependencies() {
            consumers.entry(dependency).or_default().push(node);
        }
    }
    consumers
}

/// One round's own three consumer-side facts: [`crate::spec::append_moe_ffn`]'s
/// own `mask`/`max_selection`/`route`/`weight` for whichever round produced
/// `selection_scores`'s own `mask`/`max_selection` pair — see
/// [`match_moe_topk`] for how successive rounds chain together.
#[cfg(feature = "moe-topk-fusion")]
pub(super) struct MoeRound {
    pub(super) mask: NodeId,
    pub(super) candidate: NodeId,
    pub(super) max_selection: NodeId,
    pub(super) shifted: NodeId,
    pub(super) route: NodeId,
    pub(super) weight: NodeId,
}

/// Matches one round of [`crate::spec::append_moe_ffn`]'s own
/// argmax-with-exclusion loop, forward from `selection_scores` (this round's
/// own live-score tensor, `scores` itself for round 0, the previous round's
/// exclusion `Select` output otherwise) using `consumers` rather than walking
/// backward from a guessed `route` node -- `selection_scores` has exactly two
/// live-score consumers in the real program (`max_selection`'s own reduce,
/// found here, and nothing else at this stage; `mask`'s own `Equal` is found
/// from `max_selection` next), so the forward search is a small, bounded scan
/// no matter how many total ops the program carries.
///
/// `max_selection_0` is `None` only for round 0's own call (this round's
/// `max_selection` becomes the caller's own anchor for every later round's
/// `weight = exp(max_selection_r - max_selection_0)` shift); every later
/// round passes `Some` of round 0's already-matched `max_selection`.
#[cfg(feature = "moe-topk-fusion")]
pub(super) fn moe_match_round(
    program: &[Op],
    selection_scores: NodeId,
    max_selection_0: Option<NodeId>,
    expert_index: NodeId,
    consumers: &BTreeMap<NodeId, Vec<NodeId>>,
) -> Option<MoeRound> {
    let max_selection = consumers
        .get(&selection_scores)?
        .iter()
        .copied()
        .find(|candidate| {
            moe_reduce_operand(
                program,
                *candidate,
                DType::Float32,
                ScalarOp::Maximum,
                ReduceInit::NegativeInfinity,
            ) == Some(selection_scores)
        })?;
    let mask = consumers
        .get(&selection_scores)?
        .iter()
        .copied()
        .find(|candidate| {
            moe_binary_elementwise(program, *candidate, ScalarOp::Equal)
                == Some([selection_scores, max_selection])
        })?;
    let candidate = consumers.get(&mask)?.iter().copied().find(|candidate| {
        moe_binary_elementwise(program, *candidate, ScalarOp::Multiply)
            == Some([mask, expert_index])
    })?;
    let route = consumers.get(&candidate)?.iter().copied().find(|route| {
        moe_reduce_operand(
            program,
            *route,
            DType::Int32,
            ScalarOp::Maximum,
            ReduceInit::Zero,
        ) == Some(candidate)
    })?;
    let anchor = max_selection_0.unwrap_or(max_selection);
    let shifted = consumers
        .get(&max_selection)?
        .iter()
        .copied()
        .find(|candidate| {
            moe_binary_elementwise(program, *candidate, ScalarOp::Subtract)
                == Some([max_selection, anchor])
        })?;
    let weight = consumers.get(&shifted)?.iter().copied().find(|candidate| {
        matches!(
            program.get(candidate.0 as usize),
            Some(Op::Elementwise { body: ScalarOp::Exponential, operands, .. })
                if operands.len() == 1 && operands[0].0 == shifted
        )
    })?;
    Some(MoeRound {
        mask,
        candidate,
        max_selection,
        shifted,
        route,
        weight,
    })
}

/// The whole fused kind's own shape: `scores`/`expert_count`/`top_k` plus
/// every round's own `route`/`weight` (in round order, [`crate::spec::MoeSite`]'s
/// own ordering) and the final `weight_total`, plus every internal node the
/// fusion consumes and must therefore drop from `resolved` once it fires.
#[cfg(feature = "moe-topk-fusion")]
pub(super) struct MoeTopKMatch {
    pub(super) scores: NodeId,
    pub(super) expert_count: u64,
    pub(super) routes: Vec<NodeId>,
    pub(super) weights: Vec<NodeId>,
    pub(super) weight_total: NodeId,
    pub(super) absorbed: BTreeSet<NodeId>,
}

/// Matches `append_moe_ffn`'s own whole routing chain, anchored
/// at `route0` (round 0's own `route` node -- this op's own eventual primary
/// `node`, mirroring [`BoundOpKind::GatedDeltaNet`]'s "first output is the
/// bound op's own node" shape). Declines (returns `None`) on ANY structural
/// deviation -- a different gating function, an `expert_bias`, a rank/shape
/// this slice does not support, `n_tokens != 1` -- rather than guess: ROW
/// 569's own design note names this as the one safe default for a
/// correctness-critical routing decision.
#[cfg(feature = "moe-topk-fusion")]
pub(super) fn match_moe_topk(
    program: &[Op],
    shapes: &Shapes,
    route0: NodeId,
    consumers: &BTreeMap<NodeId, Vec<NodeId>>,
) -> Option<MoeTopKMatch> {
    let candidate0 = moe_reduce_operand(
        program,
        route0,
        DType::Int32,
        ScalarOp::Maximum,
        ReduceInit::Zero,
    )?;
    let [mask0, expert_index] = moe_binary_elementwise(program, candidate0, ScalarOp::Multiply)?;
    if !matches!(program.get(expert_index.0 as usize), Some(Op::Iota { .. })) {
        return None;
    }
    let [scores, max_selection0] = moe_binary_elementwise(program, mask0, ScalarOp::Equal)?;
    if moe_reduce_operand(
        program,
        max_selection0,
        DType::Float32,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
    ) != Some(scores)
    {
        return None;
    }
    // Round 0's own live-score tensor is never itself the exclusion
    // `Select`'s output -- [`crate::spec::ExpertGatingFunc::Softmax`] with no
    // `expert_bias`, the ONLY shape `proxima-model-interop`'s own qwen35moe
    // builder produces (`qwen35moe/program.rs:122-135`). Any bias/Sigmoid
    // program builds a DIFFERENT node here (an `Add`/`Reciprocal` chain, not
    // this reduce's own direct source), so this also implicitly declines
    // both of those, exactly as designed.
    if matches!(
        program.get(scores.0 as usize),
        Some(Op::Elementwise {
            body: ScalarOp::Select,
            ..
        })
    ) {
        return None;
    }
    let expert_count = *shapes.of(expert_index).first()?;
    if shapes.of(expert_index) != [expert_count] {
        return None;
    }
    if shapes.of(route0) != [1] {
        // n_tokens == 1 only, the same decode-only restriction
        // `BoundOpKind::GatedDeltaNet` carries.
        return None;
    }

    let mut routes = Vec::new();
    let mut weights = Vec::new();
    let mut absorbed: BTreeSet<NodeId> = BTreeSet::new();
    let mut selection_scores = scores;
    let mut max_selection_0_node: Option<NodeId> = None;
    let mut weight_total_running: Option<NodeId> = None;
    let mut round: usize = 0;
    const ROUND_SANITY_CAP: usize = 64;
    loop {
        let matched = moe_match_round(
            program,
            selection_scores,
            max_selection_0_node,
            expert_index,
            consumers,
        )?;
        if round == 0 {
            if matched.route != route0 || matched.mask != mask0 {
                return None;
            }
            max_selection_0_node = Some(matched.max_selection);
        }
        absorbed.insert(matched.mask);
        absorbed.insert(matched.candidate);
        absorbed.insert(matched.max_selection);
        absorbed.insert(matched.shifted);
        absorbed.insert(matched.weight);
        // `route0` (round 0) becomes this op's own primary `node`, never an
        // absorbed one -- every LATER round's `route` is multi-consumer
        // (three `gathered_expert_product` gathers in
        // `append_moe_ffn`'s own
        // per-round FFN evaluation), so it materializes as its own `BoundOp`
        // in `rebuilt` regardless of this fusion; it must be dropped here so
        // the fused kind's own extra-output write supplies it instead.
        if round != 0 {
            absorbed.insert(matched.route);
        }
        routes.push(matched.route);
        weights.push(matched.weight);
        weight_total_running = Some(match weight_total_running {
            None => matched.weight,
            Some(running) => {
                let add_node =
                    consumers
                        .get(&matched.weight)?
                        .iter()
                        .copied()
                        .find(|candidate| {
                            moe_binary_elementwise(program, *candidate, ScalarOp::Add)
                                == Some([running, matched.weight])
                        })?;
                absorbed.insert(add_node);
                add_node
            }
        });
        let next_selection_scores = consumers.get(&matched.mask).and_then(|list| {
            list.iter().copied().find(|candidate| {
                matches!(
                    program.get(candidate.0 as usize),
                    Some(Op::Elementwise { body: ScalarOp::Select, operands, .. })
                        if operands.len() == 3
                            && operands[0].0 == matched.mask
                            && operands[2].0 == selection_scores
                )
            })
        });
        round += 1;
        match next_selection_scores {
            Some(next) if round < ROUND_SANITY_CAP => {
                absorbed.insert(next);
                selection_scores = next;
            }
            Some(_) => return None,
            None => break,
        }
    }
    if routes.len() < 2 {
        // A genuine single-expert "top-1" program never builds the exclusion
        // `Select` this matcher's forward walk relies on to terminate the
        // loop the same way it started -- declining rather than fusing a
        // shape this matcher cannot have actually exercised.
        return None;
    }
    let weight_total = weight_total_running?;
    Some(MoeTopKMatch {
        scores,
        expert_count,
        routes,
        weights,
        weight_total,
        absorbed,
    })
}

/// Scans `program` for [`append_moe_ffn`] round-0 candidates and,
/// for each, resolves `scores`'s own [`Layout`] against `resolved` -- the
/// same technique [`gated_delta_net_candidates`] uses for its own six
/// operand sources, scaled down to the one true operand this kind reads
/// (`routes`/`weights`/`weight_total` are outputs, not operands).
#[cfg(feature = "moe-topk-fusion")]
pub(super) fn moe_topk_candidates(
    program: &[Op],
    shapes: &Shapes,
    resolved: &[BoundOp],
    effective_outputs: &[NodeId],
) -> Vec<(BoundOp, BTreeSet<NodeId>)> {
    let consumers = moe_consumers(program);
    let mut candidates = Vec::new();
    for output_position in (0..program.len()).rev() {
        let route0 = NodeId(output_position as u32);
        let Some(found) = match_moe_topk(program, shapes, route0, &consumers) else {
            continue;
        };
        // Every extra output this kind will supply directly (`routes[1..]`,
        // every `weights` entry, `weight_total`) is fine to be a requested
        // output; any OTHER absorbed node (a mask, a max-selection reduce, an
        // exclusion `Select`, ...) genuinely disappears once fusion fires, the
        // same guard [`gated_delta_net_candidates`] makes for `state_out`.
        let legitimate_outputs: BTreeSet<NodeId> = found
            .routes
            .iter()
            .skip(1)
            .chain(found.weights.iter())
            .chain(core::iter::once(&found.weight_total))
            .copied()
            .collect();
        if found
            .absorbed
            .iter()
            .any(|node| !legitimate_outputs.contains(node) && effective_outputs.contains(node))
        {
            continue;
        }
        let Some((_, layout, lookup)) = resolved
            .iter()
            .flat_map(|bound| bound.operands().iter())
            .find(|(node, _, _)| *node == found.scores)
        else {
            continue;
        };
        if lookup.is_some() || layout.strides.iter().any(|stride| *stride < 0) {
            continue;
        }
        let top_k = found.routes.len() as u64;
        let fused = BoundOp {
            node: route0,
            dtype: DType::Int32,
            extents: shapes.of(route0).to_vec(),
            kind: BoundOpKind::MoeTopK {
                operands: vec![(found.scores, layout.clone(), None)],
                expert_count: found.expert_count,
                top_k,
                routes: found.routes.clone(),
                weights: found.weights.clone(),
                weight_total: found.weight_total,
            },
        };
        candidates.push((fused, found.absorbed));
    }
    candidates
}

/// Runs [`moe_topk_candidates`] and rewrites `built` with every
/// non-conflicting match -- the [`BoundOpKind::MoeTopK`] sibling of
/// [`apply_gated_delta_net_fusion`]'s own two-pass shape: an initial pass
/// finds candidates against `built`, widens the planning outputs to `scores`
/// (the one true operand every candidate needs materialized), rebinds, then
/// matches again against the wider `resolved` set before rewriting. Runs
/// inside [`bind_with_fusion`] only -- never [`bind_plain`], the same rule
/// [`apply_gated_delta_net_fusion`] follows.
#[cfg(feature = "moe-topk-fusion")]
pub(super) fn apply_moe_topk_fusion(
    built: Vec<BoundOp>,
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
) -> Result<Vec<BoundOp>, TensorError> {
    let initial_candidates = moe_topk_candidates(program, shapes, &built, outputs);
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
        let BoundOpKind::MoeTopK { operands, .. } = &fused.kind else {
            continue;
        };
        for (source, _, _) in operands {
            if !planning_outputs.contains(source) {
                planning_outputs.push(*source);
            }
        }
    }
    let rebuilt = bind_plain(
        program,
        shapes,
        &planning_outputs,
        NumericPolicy::bit_exact(),
    )?;
    let candidates = moe_topk_candidates(program, shapes, &rebuilt, outputs);
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

/// Runs [`gated_delta_net_candidates`] and rewrites `built` with every
/// non-conflicting match — the [`BoundOpKind::GatedDeltaNet`] sibling of
/// [`bind_cached_attention_fusion`]'s own two-pass shape: an initial pass
/// finds candidates against `built`, widens the planning outputs to every
/// source [`match_gated_delta_net_step`] needs materialized, rebinds, then
/// matches again against the wider `resolved` set before rewriting.
#[cfg(feature = "gated-delta-net-fusion")]
pub(super) fn apply_gated_delta_net_fusion(
    built: Vec<BoundOp>,
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
    fuse_cached_attention: bool,
    numeric_policy: NumericPolicy,
) -> Result<Vec<BoundOp>, TensorError> {
    let initial_candidates = gated_delta_net_candidates(program, shapes, &built, outputs);
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
        let BoundOpKind::GatedDeltaNet { operands, .. } = &fused.kind else {
            continue;
        };
        for (source, _, _) in operands {
            if !planning_outputs.contains(source) {
                planning_outputs.push(*source);
            }
        }
    }
    // `bind_plain` here used to drop every `BoundOpKind::CachedAttention`
    // `bind_cached_attention_fusion` above already spliced into `built` --
    // this rebind must carry that SAME fusion forward, or a hybrid
    // full-attention/gated-delta-net model (qwen35moe) loses all of its
    // cached-attention fusion the moment this feature is compiled in
    // (row 565: `built` measured 9-10 `CachedAttention` ops, `rebuilt` measured 0).
    let rebuilt = bind_cached_attention_fusion(
        program,
        shapes,
        &planning_outputs,
        fuse_cached_attention,
        numeric_policy,
    )?;
    let candidates = gated_delta_net_candidates(program, shapes, &rebuilt, outputs);
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
