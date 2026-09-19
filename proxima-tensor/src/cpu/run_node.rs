use super::*;

#[cfg(test)]
pub(super) fn run_node(
    resolved: &BoundOp,
    buffers: &[Option<Vec<f32>>],
) -> Result<Vec<f32>, TensorError> {
    let mut output = vec![0.0f32; node_output_len(resolved)];
    run_node_into(resolved, buffers, None, None, None, false, &mut output)?;
    Ok(output)
}

/// Evaluates one already-bound dense operation against caller-supplied f32
/// operand snapshots. This is the narrow CPU oracle used by device
/// diagnostics: it executes the exact [`BoundOp`] the device encoded without
/// rebinding the graph or retaining unrelated intermediates.
///
/// # Errors
/// Rejects packed and gathered operands explicitly because their device
/// buffers are not dense f32 tensors. Also rejects an output slice whose
/// length differs from the bound operation's resolved output length.
pub fn evaluate_bound_f32_into(
    resolved: &BoundOp,
    buffers: &[Option<&[f32]>],
    packed_operands: &[NodeId],
    output: &mut [f32],
) -> Result<(), TensorError> {
    for (operand, _, lookup) in resolved.all_read_sources() {
        if lookup.is_some() {
            return Err(TensorError::BoundF32GatherOperand {
                node: resolved.node,
                operand: *operand,
            });
        }
        if packed_operands.contains(operand) {
            return Err(TensorError::BoundF32PackedOperand {
                node: resolved.node,
                operand: *operand,
            });
        }
    }
    let expected = node_output_len(resolved);
    if output.len() != expected {
        return Err(TensorError::InputSizeMismatch {
            node: resolved.node,
            expected,
            found: output.len(),
        });
    }
    run_node_into(resolved, buffers, None, None, None, true, output)
}

/// Runs `resolved`, writing its output into a caller-provided slice instead
/// of allocating one. This is the primitive [`run_node`] (sequential) and
/// [`evaluate_parallel`] (one call per chunk, each writing a disjoint
/// sub-slice of the same parent buffer) both drive — the loop nests below
/// are written once, here.
pub(super) fn run_node_into<B: Deref<Target = [f32]> + Sync>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    quantized_weights: Option<&BTreeMap<NodeId, QuantizedBlock>>,
    expert_sources: Option<&BTreeMap<NodeId, ExpertSource<'_>>>,
    session: Option<&MatmulSession<'_>>,
    exact_activations: bool,
    output: &mut [f32],
) -> Result<(), TensorError> {
    run_node_into_with_gdn_state(
        resolved,
        buffers,
        quantized_weights,
        expert_sources,
        session,
        exact_activations,
        output,
        None,
        None,
    )
}

/// [`run_node_into`]'s own body, plus the two extra seams
/// [`BoundOpKind::GatedDeltaNet`]/[`BoundOpKind::MoeTopK`] dispatch need:
/// `gdn_state_sink` and `moe_topk_extra_sink`, each `Some` only at the call
/// sites that must persist the op's own extra outputs
/// (`run_gated_delta_net`'s/[`run_moe_topk`]'s own doc) -- `Interpreter::fold`,
/// `evaluate_quantized_with_scratch_impl`'s own loop, and
/// `run_resolved_nodes_in_arena`. Every other caller reaches
/// [`run_node_into`] above, which always passes `None` for both -- correct
/// because none of them can ever hand this a resolved `GatedDeltaNet`/
/// `MoeTopK` node (the typed executors reject both kinds before dispatch).
///
/// `clippy::too_many_arguments`: this mirrors [`run_node_into`]'s own seven,
/// plus the two seams this function adds -- splitting it further would just
/// relocate the same dispatch one level down.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_node_into_with_gdn_state<B: Deref<Target = [f32]> + Sync>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    quantized_weights: Option<&BTreeMap<NodeId, QuantizedBlock>>,
    expert_sources: Option<&BTreeMap<NodeId, ExpertSource<'_>>>,
    session: Option<&MatmulSession<'_>>,
    exact_activations: bool,
    output: &mut [f32],
    gdn_state_sink: Option<&mut Vec<f32>>,
    moe_topk_extra_sink: Option<&mut Vec<f32>>,
) -> Result<(), TensorError> {
    let gdn_debug_q_reduce = std::env::var_os("PROXIMA_DEBUG_GDN_COMPARE").is_some()
        && output.len() >= 4_096
        && matches!(
            &resolved.kind,
            BoundOpKind::Reduce { element_body, .. }
                if element_body.steps.len() == 1
                    && element_body.steps[0].op == ScalarOp::Multiply
        );
    if std::env::var_os("PROXIMA_DEBUG_GDN_COMPARE").is_some()
        && (resolved.node.0 == 41 || resolved.node.0 == 1264 || resolved.node.0 == 1265)
    {
        eprintln!(
            "gdn_q_nodes node={} extents={:?} output_len={} operands={:?} kind={:?}",
            resolved.node.0,
            resolved.extents,
            output.len(),
            resolved
                .operands()
                .iter()
                .map(|(node, _, _)| node.0)
                .collect::<Vec<_>>(),
            resolved.kind
        );
    }
    let result = match &resolved.kind {
        BoundOpKind::CachedAttention { .. } => {
            #[cfg(feature = "instrument")]
            instrument::record_op_kind(instrument::OpKind::CachedAttention);
            run_cached_attention(resolved, buffers, output)
        }
        BoundOpKind::GatedDeltaNet { .. } => {
            run_gated_delta_net(resolved, buffers, output, gdn_state_sink)
        }
        BoundOpKind::MoeTopK { .. } => run_moe_topk(resolved, buffers, output, moe_topk_extra_sink),
        BoundOpKind::Elementwise { .. } => {
            #[cfg(feature = "instrument")]
            instrument::record_op_kind(instrument::OpKind::Elementwise);
            let gather_block = quantized_gather_operand(resolved).and_then(|(source, lookup)| {
                quantized_weights
                    .and_then(|weights| weights.get(&source).map(|block| (lookup, block)))
            });
            match gather_block {
                Some((lookup, block)) => {
                    if let Some(source) = expert_sources.and_then(|sources| {
                        quantized_gather_operand(resolved)
                            .and_then(|(node, _)| sources.get(&node).copied())
                    }) {
                        run_embedding_gather_expert_source(
                            resolved, buffers, &lookup, source, output,
                        )
                    } else {
                        run_embedding_gather_quantized(resolved, buffers, &lookup, block, output)
                    }
                }
                None => run_elementwise_dispatch(resolved, buffers, session, output),
            }
        }
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            out_scatter: Some(_),
            ..
        } => {
            #[cfg(feature = "instrument")]
            instrument::record_op_kind(instrument::OpKind::Reduce);
            run_reduce_scatter(resolved, buffers, output)
        }
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            out_scatter: None,
            ..
        } => {
            #[cfg(feature = "instrument")]
            instrument::record_op_kind(instrument::OpKind::Reduce);
            match quantized_weights {
                Some(quantized_weights) => run_reduce_with_quantized_weights(
                    resolved,
                    buffers,
                    quantized_weights,
                    expert_sources,
                    session,
                    exact_activations,
                    output,
                ),
                None => run_reduce(resolved, buffers, output, None),
            }
        }
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => {
            #[cfg(feature = "instrument")]
            instrument::record_op_kind(instrument::OpKind::Scan);
            run_scan(resolved, buffers, output)
        }
        BoundOpKind::Iota => run_iota(output),
        BoundOpKind::Constant { value } => run_constant(*value, output),
    };
    result?;
    if gdn_debug_q_reduce {
        eprintln!(
            "gdn_q_reduce_output node={} extents={:?} first={:?}",
            resolved.node.0,
            resolved.extents,
            &output[..output.len().min(4)]
        );
    }
    apply_reduce_epilogue(resolved, buffers, output)
}

/// Applies [`BoundOpKind::Reduce::epilogue_body`] over
/// [`BoundOpKind::Reduce::epilogue_operands`] to each already-folded output
/// element, once per element in [`BoundOpKind::Reduce::output_axes`]
/// coordinate space — the mirror of how [`BoundOp::element_body`] combines
/// operands once per PRE-reduction step (`element_body`'s own doc in
/// `bind.rs`). Runs after every `Reduce` arm above regardless of which
/// internal fast path wrote `output`, since every one of them writes through
/// the same `out_layout` addressing this reads back. A no-op for every other
/// `BoundOpKind` and for the untouched default epilogue (`ComposedBody::leaf
/// (ScalarOp::Identity)` over zero operands — [`bind::BoundOpBuilder`]'s own
/// convention when `reduce-epilogue-fusion` never fired or is not compiled
/// in), so this never costs a scan over `output` on a plain reduce.
pub(super) fn apply_reduce_epilogue<B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    output: &mut [f32],
) -> Result<(), TensorError> {
    let BoundOpKind::Reduce {
        output_axes,
        out_layout,
        epilogue_body,
        epilogue_operands,
        epilogue_broadcast_axes,
        ..
    } = &resolved.kind
    else {
        return Ok(());
    };
    if reduce_epilogue_is_identity(epilogue_body, epilogue_operands) {
        return Ok(());
    }
    let operand_buffers: Vec<&[f32]> = epilogue_operands
        .iter()
        .map(|(node, _, lookup)| {
            if lookup.is_some() {
                return Err(TensorError::NotLowerable {
                    node: resolved.node,
                    reason: "reduce epilogue does not support a gathered operand",
                });
            }
            buffer_of(buffers, *node)
        })
        .collect::<Result<_, _>>()?;

    if epilogue_broadcast_axes.is_empty() {
        apply_plain_reduce_epilogue(
            resolved,
            output_axes,
            out_layout,
            epilogue_body,
            epilogue_operands,
            &operand_buffers,
            output,
        );
    } else {
        apply_broadcast_reduce_epilogue(
            resolved,
            output_axes,
            out_layout,
            epilogue_body,
            epilogue_operands,
            &operand_buffers,
            output,
        );
    }
    Ok(())
}

/// The pre-existing, shape-preserving epilogue: walks
/// [`BoundOpKind::Reduce::output_axes`]'s own (smaller) iteration space,
/// reading and writing the SAME `output` slot the fold itself wrote —
/// unchanged from before `epilogue_broadcast_axes` existed.
pub(super) fn apply_plain_reduce_epilogue(
    resolved: &BoundOp,
    output_axes: &[u16],
    out_layout: &bind::Layout,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, bind::Layout, Option<bind::Lookup>)],
    operand_buffers: &[&[f32]],
    output: &mut [f32],
) {
    let extents_local: Vec<u64> = output_axes
        .iter()
        .map(|&axis| resolved.extents[axis as usize])
        .collect();
    let mut local_coordinate = vec![0u64; output_axes.len()];
    let mut full_coordinate = vec![0u64; resolved.extents.len()];
    let mut operand_values = vec![0.0f32; epilogue_operands.len() + 1];
    let mut step_values = vec![0.0f32; epilogue_body.steps.len()];

    for flat in 0..odometer_len(&extents_local) {
        unflatten_into(flat, &extents_local, &mut local_coordinate);
        full_coordinate.fill(0);
        for (index, &axis) in output_axes.iter().enumerate() {
            full_coordinate[axis as usize] = local_coordinate[index];
        }
        let offset = out_layout.offset_of(&full_coordinate) as usize;
        for (slot, (_, layout, _)) in epilogue_operands.iter().enumerate() {
            operand_values[slot] =
                operand_buffers[slot][layout.offset_of(&local_coordinate) as usize];
        }
        operand_values[epilogue_operands.len()] = output[offset];
        output[offset] = apply_body(epilogue_body, &operand_values, &mut step_values);
    }
}

/// The "broadcast-reduce" epilogue (`epilogue_broadcast_axes`'s own doc): the
/// fold's own scalar result — materialized by `run_reduce`/`run_scan` into
/// exactly the first `fold_len` slots of `output`, per `out_layout`'s own
/// (smaller) addressing — is copied into `fold_scratch` FIRST, since the
/// write loop below overwrites `output` in place at a DIFFERENT (larger)
/// stride and would otherwise clobber a not-yet-read fold value out from
/// under a later coordinate sharing its physical slot. After the copy, this
/// walks the CONSUMER's own full `resolved.extents` space (RMSNorm's own
/// `[s, d]`, not the fold's smaller `[s]`), re-reading the fold's value at
/// each position via `out_layout` projected onto `output_axes` alone — a
/// genuine broadcast over every `epilogue_broadcast_axes` entry, since that
/// projection ignores them entirely.
pub(super) fn apply_broadcast_reduce_epilogue(
    resolved: &BoundOp,
    output_axes: &[u16],
    out_layout: &bind::Layout,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, bind::Layout, Option<bind::Lookup>)],
    operand_buffers: &[&[f32]],
    output: &mut [f32],
) {
    let fold_extents: Vec<u64> = output_axes
        .iter()
        .map(|&axis| resolved.extents[axis as usize])
        .collect();
    let fold_len = odometer_len(&fold_extents) as usize;
    let fold_scratch: Vec<f32> = output[..fold_len].to_vec();

    let mut full_coordinate = vec![0u64; resolved.extents.len()];
    let mut operand_values = vec![0.0f32; epilogue_operands.len() + 1];
    let mut step_values = vec![0.0f32; epilogue_body.steps.len()];

    for flat in 0..odometer_len(&resolved.extents) {
        unflatten_into(flat, &resolved.extents, &mut full_coordinate);
        // `out_layout` is full rank (`resolved.extents.len()`, per its own
        // doc) with stride `0` on every axis NOT in `output_axes` — the
        // fold's own broadcast addressing — so reading it at the FULL
        // coordinate lands on the same compact `[0, fold_len)` range
        // `run_reduce`/`run_scan` themselves wrote, regardless of this
        // position's value on a broadcast axis.
        let fold_offset = out_layout.offset_of(&full_coordinate) as usize;
        for (slot, (_, layout, _)) in epilogue_operands.iter().enumerate() {
            operand_values[slot] =
                operand_buffers[slot][layout.offset_of(&full_coordinate) as usize];
        }
        operand_values[epilogue_operands.len()] = fold_scratch[fold_offset];
        output[flat as usize] = apply_body(epilogue_body, &operand_values, &mut step_values);
    }
}

/// The untouched default an executor must treat as "no epilogue": a leaf
/// [`ScalarOp::Identity`] reading its own sole implicit slot, over zero real
/// operands — [`BoundOpKind::Reduce::epilogue_body`]'s own doc names this the
/// same convention `element_body` already uses for "nothing fused into the
/// prologue either".
pub(super) fn reduce_epilogue_is_identity(
    body: &ComposedBody,
    operands: &[(NodeId, bind::Layout, Option<bind::Lookup>)],
) -> bool {
    operands.is_empty()
        && body.steps.len() == 1
        && body.steps[0].op == ScalarOp::Identity
        && body.steps[0].args == [StepArg::Operand(0)]
}

pub(super) fn run_cached_attention<B: Deref<Target = [f32]> + Sync>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    output: &mut [f32],
) -> Result<(), TensorError> {
    let BoundOpKind::CachedAttention {
        operands,
        query_rows,
        cached_key_rows,
        new_key_rows,
        kv_heads,
        query_groups,
        head_dim,
        rotary_dim,
        scale,
        cached_lower_inclusive,
        new_upper_inclusive,
    } = &resolved.kind
    else {
        return Err(TensorError::NotLowerable {
            node: resolved.node,
            reason: "cached attention runner received another bound operation",
        });
    };
    // `rotary_dim < head_dim` (`BoundOpKind::CachedAttention`'s own doc) is
    // the discriminator for the trailing three-operand pass plane -- never
    // the operand count alone, since the optional ninth `cached_len` slot
    // already varies independently of it.
    let pass_present = rotary_dim < head_dim;
    let expected_lengths: &[usize] = if pass_present { &[11, 12] } else { &[8, 9] };
    if !expected_lengths.contains(&operands.len())
        || operands.iter().any(|(_, _, lookup)| lookup.is_some())
    {
        return Err(TensorError::NotLowerable {
            node: resolved.node,
            reason: "cached attention requires eight or nine base operands, plus three more \
                     when a partial-rotary pass plane is present",
        });
    }
    let cached_len_index = if pass_present {
        (operands.len() == 12).then_some(8)
    } else {
        (operands.len() == 9).then_some(8)
    };
    // The optional ninth operand shares its slot between two DIFFERENT
    // runtime scalars, discriminated by `cached_key_rows`
    // (`BoundOpKind::CachedAttention`'s own doc): `cached_key_rows == 0`
    // (single-range fusion) is the real `new_upper_inclusive`; `cached_key_
    // rows != 0` (two-range fusion, `cached_attention_candidates`) is the
    // CACHED range's own live row count, substituted for the compiled
    // (bucket-padded) `cached_key_rows` below. Read here, not looped over
    // with the eight Q/K/V sources below, since it is a bare rank-0 value
    // rather than a contiguous tensor tail.
    let dynamic_ninth = match cached_len_index.and_then(|index| operands.get(index)) {
        Some((node, layout, _)) => {
            if layout.base != 0 || !layout.strides.is_empty() {
                return Err(TensorError::NotLowerable {
                    node: resolved.node,
                    reason: "cached attention's cached_len operand must be a bare scalar",
                });
            }
            let cached_len_buffer = buffer_of(buffers, *node)?;
            let cached_len_value = *cached_len_buffer.first().ok_or(TensorError::NotLowerable {
                node: resolved.node,
                reason: "cached attention's cached_len operand is empty",
            })?;
            Some(cached_len_value)
        }
        None => None,
    };
    let (new_upper_inclusive, live_cached_key_rows) = match dynamic_ninth {
        Some(cached_len_value) if *cached_key_rows == 0 => {
            let dynamic_new_upper_inclusive = cached_len_value as i64;
            // The runtime `cached_len` value is a live position count, never
            // larger than the compiled buffer extent it indexes into -- a
            // value at or past `new_key_rows` means the caller handed a
            // stale or wrong-bucket length.
            if dynamic_new_upper_inclusive >= *new_key_rows as i64 {
                return Err(TensorError::NotLowerable {
                    node: resolved.node,
                    reason: "cached attention's runtime cached_len operand exceeds its buffer extent",
                });
            }
            (dynamic_new_upper_inclusive, *cached_key_rows)
        }
        Some(cached_len_value) => {
            let live = cached_len_value as u64;
            // The live cached length a `kv-capacity-bucket` caller rounded
            // UP to the compiled `cached_key_rows` -- never past it, or the
            // caller handed a length that does not fit the bucket this
            // `BoundOp` was compiled for.
            if live > *cached_key_rows {
                return Err(TensorError::NotLowerable {
                    node: resolved.node,
                    reason: "cached attention's runtime cached_len operand exceeds its buffer extent",
                });
            }
            (*new_upper_inclusive, live)
        }
        None => (*new_upper_inclusive, *cached_key_rows),
    };
    let mut sources = operands.iter().take(8).map(|(node, layout, _)| {
        if layout.base != 0 || layout.strides.last().copied() != Some(1) {
            return Err(TensorError::NotLowerable {
                node: resolved.node,
                reason: "cached attention requires zero-based contiguous source tails",
            });
        }
        buffer_of(buffers, *node)
    });
    let query_even = sources.next().ok_or(TensorError::NotLowerable {
        node: resolved.node,
        reason: "cached attention query source is missing",
    })??;
    let query_odd = sources.next().ok_or(TensorError::NotLowerable {
        node: resolved.node,
        reason: "cached attention query source is missing",
    })??;
    let cached_key_even = sources.next().ok_or(TensorError::NotLowerable {
        node: resolved.node,
        reason: "cached attention cached-key source is missing",
    })??;
    let cached_key_odd = sources.next().ok_or(TensorError::NotLowerable {
        node: resolved.node,
        reason: "cached attention cached-key source is missing",
    })??;
    let new_key_even = sources.next().ok_or(TensorError::NotLowerable {
        node: resolved.node,
        reason: "cached attention new-key source is missing",
    })??;
    let new_key_odd = sources.next().ok_or(TensorError::NotLowerable {
        node: resolved.node,
        reason: "cached attention new-key source is missing",
    })??;
    let cached_value = sources.next().ok_or(TensorError::NotLowerable {
        node: resolved.node,
        reason: "cached attention cached-value source is missing",
    })??;
    let new_value = sources.next().ok_or(TensorError::NotLowerable {
        node: resolved.node,
        reason: "cached attention new-value source is missing",
    })??;
    // `two_range_cached_bound` (`live_cached_key_rows < *cached_key_rows`)
    // takes only the LIVE prefix of each cached-range buffer: every padded
    // row a `kv-capacity-bucket` caller rounded `cached_key_rows` up past
    // lives at `[live_cached_key_rows, *cached_key_rows)`
    // (`proxima_tensor::bind::BoundOpKind::CachedAttention`'s own doc) --
    // slicing the prefix here, rather than passing the full compiled-size
    // buffer through, is what lets `AttentionExtents.cached_key_rows` below
    // carry the live count without `stream_cached_attention_split_gqa`'s own
    // exact-length shape check rejecting the mismatch.
    let pair_dim = (*rotary_dim / 2) as usize;
    let live_cached_key_rows_usize = live_cached_key_rows as usize;
    let live_pair_len = live_cached_key_rows_usize * *kv_heads as usize * pair_dim;
    let live_value_len = live_cached_key_rows_usize * *kv_heads as usize * *head_dim as usize;
    let out_of_range = TensorError::NotLowerable {
        node: resolved.node,
        reason: "cached attention's live cached_key_rows exceeds its own buffer length",
    };
    let cached_key_even = cached_key_even
        .get(..live_pair_len)
        .ok_or(out_of_range.clone())?;
    let cached_key_odd = cached_key_odd
        .get(..live_pair_len)
        .ok_or(out_of_range.clone())?;
    let cached_value = cached_value
        .get(..live_value_len)
        .ok_or(out_of_range.clone())?;
    // The trailing pass-plane triple (`pass_query`, `pass_cached_key`,
    // `pass_new_key` -- `BoundOpKind::CachedAttention`'s own doc) sits right
    // after the base eight and the optional `cached_len` scalar; `pair_dim`
    // above already used `rotary_dim`, not `head_dim`, so this is the ONLY
    // other place partial rotary changes this executor's shape.
    let pass = if pass_present {
        let pass_start = if cached_len_index.is_some() { 9 } else { 8 };
        let pass_dim = (*head_dim - *rotary_dim) as usize;
        let mut pass_sources = operands[pass_start..pass_start + 3].iter().map(
            |(node, layout, _)| -> Result<&[f32], TensorError> {
                if layout.base != 0 || layout.strides.last().copied() != Some(1) {
                    return Err(TensorError::NotLowerable {
                        node: resolved.node,
                        reason: "cached attention requires zero-based contiguous source tails",
                    });
                }
                buffer_of(buffers, *node)
            },
        );
        let pass_query = pass_sources.next().ok_or(TensorError::NotLowerable {
            node: resolved.node,
            reason: "cached attention pass-plane query source is missing",
        })??;
        let pass_cached_key = pass_sources.next().ok_or(TensorError::NotLowerable {
            node: resolved.node,
            reason: "cached attention pass-plane cached-key source is missing",
        })??;
        let pass_new_key = pass_sources.next().ok_or(TensorError::NotLowerable {
            node: resolved.node,
            reason: "cached attention pass-plane new-key source is missing",
        })??;
        let live_pass_len = live_cached_key_rows_usize * *kv_heads as usize * pass_dim;
        let pass_cached_key = pass_cached_key.get(..live_pass_len).ok_or(out_of_range)?;
        Some(crate::physical::CachedAttentionPassPlane {
            query: pass_query,
            cached_key: pass_cached_key,
            new_key: pass_new_key,
        })
    } else {
        None
    };
    let streamed = crate::physical::stream_cached_attention_split_gqa(
        [query_even, query_odd],
        [
            [cached_key_even, cached_key_odd],
            [new_key_even, new_key_odd],
        ],
        [cached_value, new_value],
        output,
        crate::physical::AttentionExtents {
            query_rows: *query_rows,
            cached_key_rows: live_cached_key_rows,
            new_key_rows: *new_key_rows,
            kv_heads: *kv_heads,
            query_groups: *query_groups,
            head_dim: *head_dim,
        },
        crate::physical::CachedAttentionRotary {
            rotary_dim: *rotary_dim,
            pass,
        },
        crate::physical::CachedAttentionScore {
            scale: *scale,
            bands: [
                crate::physical::CausalBand {
                    lower_inclusive: *cached_lower_inclusive,
                    upper_inclusive: i64::MAX,
                },
                crate::physical::CausalBand {
                    lower_inclusive: i64::MIN,
                    upper_inclusive: new_upper_inclusive,
                },
            ],
        },
    );
    if streamed {
        Ok(())
    } else {
        Err(TensorError::NotLowerable {
            node: resolved.node,
            reason: "cached attention source or output extents do not match its bound domain",
        })
    }
}

/// [`BoundOpKind::GatedDeltaNet`]'s whole computation: unpack the bound
/// shape into [`gdn::GdnPrefillShape`]/[`gdn::GdnPrefillScan`] and call
/// [`gdn::run_gdn_prefill_scan`] directly — no reimplementation
/// (`BoundOpKind::GatedDeltaNet`'s own doc, and the design's own §2/§4). This
/// slice's matched shape (single physical head axis OR the real qwen35moe
/// `kv_heads`/`group` split, `heads` innermost — [`gated_delta_net_candidates`]'s
/// own doc states the scope) happens to already match [`gdn::run_gdn_prefill_scan`]'s own
/// operand layout convention exactly, so every operand is read as a plain
/// contiguous slice with no gather/scatter reshape — REJECTED here (not
/// silently reshaped) if a caller ever binds a non-natural stride, since
/// that would mean this slice's scope assumption stopped holding.
///
/// `state` is a scratch copy of `state_in`, mutated by
/// [`run_gdn_prefill_scan`] into this step's updated recurrence state, then
/// handed back to `state_sink` (ROW 547, `docs/discipline.md`) whenever a
/// caller supplies one -- the second, state-shaped output
/// [`BoundOpKind::GatedDeltaNet::state_out`] declares. `state_sink` is
/// `None` at every call site that cannot reach a `GatedDeltaNet` node at
/// runtime (the typed executors reject the kind outright before dispatch);
/// the two real dispatch points ([`Interpreter::fold`] and
/// [`evaluate_quantized_with_scratch_impl`]'s own loop) always pass `Some`.
/// One heap allocation per call, sized `head_k_dim * head_v_dim * heads`
/// (the state's own size) — a documented, scoped exception to the
/// zero-alloc hot-path default, not an oversight.
pub(super) fn run_gated_delta_net<B: Deref<Target = [f32]> + Sync>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    output: &mut [f32],
    state_sink: Option<&mut Vec<f32>>,
) -> Result<(), TensorError> {
    let BoundOpKind::GatedDeltaNet {
        operands,
        n_tokens,
        kv_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        query_key_head_stride,
        query_key_dim_stride,
        inv_sqrt_key_dim,
        state_out: _,
    } = &resolved.kind
    else {
        return Err(TensorError::NotLowerable {
            node: resolved.node,
            reason: "gated delta net runner received another bound operation",
        });
    };
    if *n_tokens != 1 || num_v_heads % kv_heads != 0 {
        return Err(TensorError::NotLowerable {
            node: resolved.node,
            reason: "gated delta net executor only supports this slice's decode shape",
        });
    }
    let [query, key, value, gate, beta, state_in] = operands.as_slice() else {
        return Err(TensorError::NotLowerable {
            node: resolved.node,
            reason: "gated delta net requires exactly six affine, gather-free operands",
        });
    };
    for (_, layout, lookup) in [query, key, value, gate, beta, state_in] {
        if lookup.is_some() || layout.base != 0 || layout.strides.last().copied() != Some(1) {
            return Err(TensorError::NotLowerable {
                node: resolved.node,
                reason: "gated delta net requires zero-based contiguous natural-order operands",
            });
        }
    }
    let shape = GdnPrefillShape {
        positions: *n_tokens as usize,
        key_dim: *head_k_dim as usize,
        value_dim: *head_v_dim as usize,
        heads: *num_v_heads as usize,
        kv_heads: *kv_heads as usize,
    };
    let mut state = buffer_of(buffers, state_in.0)?.to_vec();
    run_gdn_prefill_scan(GdnPrefillScan {
        shape,
        query: buffer_of(buffers, query.0)?,
        key: buffer_of(buffers, key.0)?,
        query_key_head_stride: *query_key_head_stride as usize,
        query_key_dim_stride: *query_key_dim_stride as usize,
        value: buffer_of(buffers, value.0)?,
        gate: buffer_of(buffers, gate.0)?,
        beta: buffer_of(buffers, beta.0)?,
        inv_sqrt_key_dim: *inv_sqrt_key_dim,
        state: &mut state,
        output,
    })?;
    if let Some(sink) = state_sink {
        *sink = state;
    }
    Ok(())
}

/// The `moe-topk-fusion` sibling of [`run_gated_delta_net`]'s own
/// `state_sink` shape, generalized from one extra output to `2 * top_k`:
/// [`BoundOpKind::MoeTopK::routes`] (all but round 0, already this op's own
/// `output`), every [`BoundOpKind::MoeTopK::weights`] entry, then
/// `weight_total` -- exactly [`moe_topk_extra_node_order`]'s own order, so a
/// caller can `zip` this sink against that order without either side naming
/// the other's internal layout.
pub(super) fn moe_topk_extra_node_order<'routing>(
    routes: &'routing [NodeId],
    weights: &'routing [NodeId],
    weight_total: NodeId,
) -> impl Iterator<Item = NodeId> + 'routing {
    routes
        .iter()
        .skip(1)
        .copied()
        .chain(weights.iter().copied())
        .chain(core::iter::once(weight_total))
}

/// [`BoundOpKind::MoeTopK`]'s whole computation: `top_k` rounds of
/// take-the-maximum-with-exclusion over `scores`, ties broken toward the
/// HIGHER index (ROW 569, `docs/discipline.md`'s own census fixture proves
/// this is what `mask * expert_index -> reduce(Maximum)` implements, so the
/// `>=` comparison below -- which keeps advancing to a later index on an
/// exact tie -- is the bit-exact match for that construction, not an
/// arbitrary choice). `weight_r = exp(max_r - max_0)`
/// (`ExpertGatingFunc::Softmax`'s own softmax-restricted-to-top-k shape,
/// [`crate::spec::append_moe_ffn`]'s own doc), `weight_total =
/// sum(weight_0..weight_{top_k-1})` -- the caller's own final
/// `output * (1 / weight_total)` renormalization is unaffected either way,
/// since `weight_total` is this op's own third kind of output, not
/// recomputed downstream.
///
/// This slice's only supported shape is `n_tokens == 1` (decode) -- checked
/// here defensively even though [`match_moe_topk`] already declines any
/// other shape at bind time, the same belt-and-suspenders
/// [`run_gated_delta_net`] applies to its own `n_tokens`.
pub(super) fn run_moe_topk<B: Deref<Target = [f32]> + Sync>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    output: &mut [f32],
    extra_sink: Option<&mut Vec<f32>>,
) -> Result<(), TensorError> {
    let BoundOpKind::MoeTopK {
        operands,
        expert_count,
        top_k,
        ..
    } = &resolved.kind
    else {
        return Err(TensorError::NotLowerable {
            node: resolved.node,
            reason: "moe top-k runner received another bound operation",
        });
    };
    let [(scores_node, _, lookup)] = operands.as_slice() else {
        return Err(TensorError::NotLowerable {
            node: resolved.node,
            reason: "moe top-k requires exactly one operand",
        });
    };
    if lookup.is_some() {
        return Err(TensorError::NotLowerable {
            node: resolved.node,
            reason: "moe top-k requires a gather-free scores operand",
        });
    }
    let scores = buffer_of(buffers, *scores_node)?;
    let expert_count = *expert_count as usize;
    let top_k = *top_k as usize;
    if scores.len() != expert_count || output.len() != 1 {
        return Err(TensorError::NotLowerable {
            node: resolved.node,
            reason: "moe top-k executor only supports this slice's single-token decode shape",
        });
    }
    let mut live = scores.to_vec();
    let mut route0 = 0.0_f32;
    let mut extra_routes: Vec<f32> = Vec::with_capacity(top_k.saturating_sub(1));
    let mut extra_weights: Vec<f32> = Vec::with_capacity(top_k);
    let mut max_selection_0 = 0.0_f32;
    let mut weight_total = 0.0_f32;
    for round in 0..top_k {
        let mut best_index = 0_usize;
        let mut best_value = f32::NEG_INFINITY;
        for (index, value) in live.iter().enumerate() {
            if *value >= best_value {
                best_value = *value;
                best_index = index;
            }
        }
        if round == 0 {
            max_selection_0 = best_value;
            route0 = best_index as f32;
        } else {
            extra_routes.push(best_index as f32);
        }
        let weight = (best_value - max_selection_0).exp();
        extra_weights.push(weight);
        weight_total += weight;
        // ROW 569: the graph's own exclusion is `mask = Equal(selection_scores,
        // max_selection)` then `Select(mask, neg_infinity, selection_scores)`
        // -- `mask` is `true` at EVERY position tied with this round's own
        // max, not only the winning (highest) index, so an exact tie excludes
        // every tied expert in the SAME round, not just the one reported as
        // `route`. A single-index exclusion here silently diverges from the
        // unfused chain the moment two experts tie (this executor's own
        // parity test caught it: unfused round 1 = 255, a naive single-index
        // exclusion gave 12).
        for value in live.iter_mut() {
            if *value == best_value {
                *value = f32::NEG_INFINITY;
            }
        }
    }
    output[0] = route0;
    if let Some(sink) = extra_sink {
        sink.clear();
        sink.extend(extra_routes);
        sink.extend(extra_weights);
        sink.push(weight_total);
    }
    Ok(())
}

/// [`BoundOpKind::Constant`]'s whole computation: every element is the same
/// literal. Even simpler than [`run_iota`] — no operand reads, no body, and
/// not even a dependence on position.
pub(super) fn run_constant(value: f32, output: &mut [f32]) -> Result<(), TensorError> {
    output.fill(value);
    Ok(())
}

/// [`BoundOpKind::Iota`]'s whole computation: `output[i] = i`, exact in f32
/// up to `GATHER_EXTENT_EXACT_FLOAT_LIMIT` (`shape.rs`'s own doc) the same
/// way a gather index is — no operand reads, no per-step body, just the
/// position itself.
pub(super) fn run_iota(output: &mut [f32]) -> Result<(), TensorError> {
    for (index, slot) in output.iter_mut().enumerate() {
        *slot = index as f32;
    }
    Ok(())
}

/// The output length [`run_node_into`] expects from `resolved`: the full
/// iteration space for an elementwise node or a `Keep::Scan` scan (neither
/// drops any dim), or the reduced (leading dims x width) shape for a
/// `Keep::Reduce` fold.
pub(super) fn node_output_len(resolved: &BoundOp) -> usize {
    match &resolved.kind {
        BoundOpKind::CachedAttention { .. } => {
            let (query_rows, kv_heads, query_groups, head_dim) = match &resolved.kind {
                BoundOpKind::CachedAttention {
                    query_rows,
                    kv_heads,
                    query_groups,
                    head_dim,
                    ..
                } => (*query_rows, *kv_heads, *query_groups, *head_dim),
                _ => unreachable!("cached-attention output shape match is exhaustive"),
            };
            query_rows as usize * kv_heads as usize * query_groups as usize * head_dim as usize
        }
        BoundOpKind::GatedDeltaNet { .. } => element_count(&resolved.extents),
        // `output_axes` excludes the scattered axis entirely (its position
        // is data-dependent, never a pure projection — see
        // `bind::pure_projection_axes`), so the ordinary leading/width
        // product below would silently drop that axis from the length.
        // `out_scatter.extent` is the one place that axis's static width
        // survives past shape inference (`bind::build_reduce_op`'s doc).
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            output_axes,
            out_scatter: Some(target),
            ..
        } => {
            let non_scattered_product: u64 = output_axes
                .iter()
                .map(|dim| resolved.extents[*dim as usize])
                .product();
            non_scattered_product as usize * target.extent as usize
        }
        // A non-empty `epilogue_broadcast_axes` (the "broadcast-reduce"
        // epilogue shape — see `bind::BoundOpKind::Reduce::epilogue_
        // broadcast_axes`'s own doc) re-broadcasts the fold's scalar back
        // over an axis the fold itself reduced away, so the MATERIALIZED
        // output this op writes is the full `extents` product, not just the
        // fold's own smaller `output_axes` shape — `apply_reduce_epilogue`
        // walks that same full space.
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            out_scatter: None,
            epilogue_broadcast_axes,
            ..
        } if !epilogue_broadcast_axes.is_empty() => element_count(&resolved.extents),
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            output_axes,
            out_scatter: None,
            ..
        } => {
            let (leading_output_axes, last_output_dim) = output_axes_split(output_axes.as_slice());
            let leading_product: u64 = leading_output_axes
                .iter()
                .map(|dim| resolved.extents[*dim as usize])
                .product();
            let width = last_output_dim.map_or(1, |dim| resolved.extents[dim as usize] as usize);
            leading_product as usize * width
        }
        _ => element_count(&resolved.extents),
    }
}

/// The execution stage: a [`Pipe`] over a batch of ready [`BoundOp`]
/// nodes — exactly the batch one upstream [`crate::bind::BoundOpBuilder`]
/// push readies.
///
/// `In = Vec<BoundOp>`, `Out = ()`: the buffer table this stage writes into
/// is interior state, borrowed from the caller at construction rather than
/// allocated here or threaded through `In`/`Out` — `Out = ()` is literal,
/// not a value smuggled through mutation and reported via a nonempty `Out`.
/// That borrow is what lets a caller run this against its own
/// no-alloc scratch. `RefCell` is the same interior-mutability idiom
/// [`shape::ShapeTable`] and [`crate::bind::BoundOpBuilder`] already use for
/// their own per-record state, applied to the buffer table that already
/// existed here rather than to a wrapper minted to host the impl. Taking
/// the batch as `In` (rather than one `BoundOp` at a time) is what makes
/// `Second::In = First::Out` hold against `BoundOpBuilder::Out =
/// Vec<BoundOp>`, so this stage composes into the full
/// `shapes.and_then(builder).and_then(interpreter)` chain with no adapter.
pub struct Interpreter<'buffers, B: Deref<Target = [f32]> + Sync> {
    pub(super) buffers: RefCell<&'buffers mut [Option<B>]>,
}

impl<'buffers, B: Deref<Target = [f32]> + Sync + From<Vec<f32>>> Interpreter<'buffers, B> {
    /// `buffers` is caller-owned scratch, one slot per program node — the
    /// same shape `prepare` already builds locally for [`evaluate`].
    /// `Interpreter` never allocates it, resizes it, or takes ownership of
    /// it. Generic over `B` (matching `run_node_into`'s bound) so the same
    /// interpreter drives both [`evaluate`]'s `Cow`-backed table (no
    /// redundant copy of an `Op::Input` block) and a plain `Vec<f32>` table.
    #[must_use]
    pub fn new(buffers: &'buffers mut [Option<B>]) -> Self {
        Self {
            buffers: RefCell::new(buffers),
        }
    }

    /// Reads a node's computed data back out of the buffer table. Separate
    /// from `Pipe::Out` on purpose: what this stage produced for the algebra
    /// (nothing — `Out = ()`) and what a caller later wants to read out of
    /// its own state are different questions, and this crate's algebra only
    /// answers the first one through `Pipe::call`.
    #[must_use]
    pub fn get(&self, node: NodeId) -> Option<Vec<f32>> {
        self.buffers.borrow()[node.0 as usize]
            .as_deref()
            .map(<[f32]>::to_vec)
    }

    /// The actual fold: written once, against a borrowed `&[BoundOp]` rather
    /// than an owned `Vec`, so a caller driving one node at a time (like
    /// [`evaluate`]) can pass a one-element slice (`core::slice::from_ref`)
    /// with no batch allocation at all — `Pipe::call` below is the only
    /// other caller, and it just hands this its owned `Vec` by reference
    /// (`&ready`), so the streaming chain's batch contract (`In =
    /// Vec<BoundOp>`, required for `Second::In = First::Out` against
    /// [`crate::bind::BoundOpBuilder`]'s `Out`) and `evaluate`'s no-alloc
    /// per-node path both bottom out in this one written-once loop.
    fn fold(&self, ready: &[BoundOp]) -> Result<(), TensorError> {
        for resolved in ready {
            let mut output = vec![0.0f32; node_output_len(resolved)];
            // `state_out` (ROW 547, `docs/discipline.md`): `GatedDeltaNet`'s
            // own second output. `Vec::new()` costs nothing until
            // `run_gated_delta_net` actually fills it (every other kind
            // leaves it untouched), and the resolved kind decides whether the
            // sink is read below, not the allocation itself.
            let mut gdn_state = Vec::new();
            let mut moe_topk_extra = Vec::new();
            {
                let buffers = self.buffers.borrow();
                run_node_into_with_gdn_state(
                    resolved,
                    *buffers,
                    None,
                    None,
                    None,
                    false,
                    &mut output,
                    Some(&mut gdn_state),
                    Some(&mut moe_topk_extra),
                )?;
                #[cfg(feature = "instrument")]
                record_bound_op_operand_access(resolved, *buffers);
            }
            let mut buffers = self.buffers.borrow_mut();
            (*buffers)[resolved.node.0 as usize] = Some(B::from(output));
            if let BoundOpKind::GatedDeltaNet { state_out, .. } = &resolved.kind {
                (*buffers)[state_out.0 as usize] = Some(B::from(gdn_state));
            }
            if let BoundOpKind::MoeTopK {
                routes,
                weights,
                weight_total,
                ..
            } = &resolved.kind
            {
                for (extra_node, value) in moe_topk_extra_node_order(routes, weights, *weight_total)
                    .zip(moe_topk_extra.iter().copied())
                {
                    (*buffers)[extra_node.0 as usize] = Some(B::from(vec![value]));
                }
            }
        }
        Ok(())
    }
}

impl<B: Deref<Target = [f32]> + Sync + From<Vec<f32>>> Pipe for Interpreter<'_, B> {
    type In = ReadyBatch;
    type Out = ();
    type Err = TensorError;

    /// Folds a batch of ready nodes into the buffer table, in order — the
    /// same fold the buffer table already does one write at a time, just
    /// driven for every element of `ready` inside one call instead of one
    /// call per element. An empty `ready` is a no-op, not a special case.
    ///
    /// `In` stays `ReadyBatch` (owned) because that is what
    /// `BoundOpBuilder::Out` already is — changing it would break the
    /// `Second::In = First::Out` composition law the streaming chain relies
    /// on — but the owned batch is only ever borrowed from here down; see
    /// `Interpreter::fold`.
    fn call(&self, ready: ReadyBatch) -> impl Future<Output = Result<(), TensorError>> {
        async move { self.fold(&ready) }
    }
}

/// A fold's `output_axes`, split into the leading (outer) dims and the
/// innermost one (if any) — shared by [`run_reduce`] and [`node_output_len`]
/// so the two agree on shape by construction.
pub(super) fn output_axes_split(output_axes: &[u16]) -> (&[u16], Option<u16>) {
    match output_axes.split_last() {
        Some((last, leading)) => (leading, Some(*last)),
        None => (&[], None),
    }
}

/// Closed-form reads/distinct-touched accounting for one operand across a
/// bound op's own iteration space, given that operand's per-axis strides
/// (`Layout::stride` against `resolved.extents`, the same rank both share
/// throughout this module). An axis this operand broadcasts over
/// (`stride == 0`) is still visited by every position along it — the loop
/// nest re-reads the same element — so it contributes its full extent to
/// `reads` but only `1` (not `extent`) to `distinct`, since the same offset
/// resolves every time. A non-broadcast axis contributes its full extent to
/// both. `distinct` is therefore exact for an ordinary (gather-free)
/// operand, since a real tensor `Layout`'s strides never alias two distinct
/// coordinates onto the same offset outside of an explicit `stride == 0`
/// broadcast.
///
/// `O(rank)`, never `O(elements)` — every caller invokes this once per
/// bound-op evaluation (`cpu::record_bound_op_operand_access`), against the
/// UNSPLIT op's own extents, so a `BoundOp::split` chunk fan-out under
/// `evaluate_parallel` never re-derives this per chunk (that would double
/// count a broadcast operand's footprint once per chunk instead of once for
/// the whole node — see `instrument.rs`'s module comment on
/// `OperandAccess`).
#[cfg(any(feature = "instrument", test))]
pub(super) fn operand_access_footprint(extents: &[u64], strides: &[i64]) -> (u64, u64) {
    let mut reads: u64 = 1;
    let mut distinct: u64 = 1;
    for (&extent, &stride) in extents.iter().zip(strides) {
        reads *= extent;
        if stride != 0 {
            distinct *= extent;
        }
    }
    (reads, distinct)
}

/// Attributes one bound op's operand reads to their own source `NodeId`s,
/// once the op has finished running. Called from `evaluate_pooled`,
/// `evaluate_node_parallel`, and `Interpreter::fold` — the three places that
/// hold the UNSPLIT `BoundOp` right after `run_node_into`/`run_chunks_threaded`
/// returns — never from inside a per-chunk or per-element loop.
///
/// A gathered operand (`Some(lookup)`) cannot get its distinct-element count
/// from `operand_access_footprint`: which table row a gather touches is a
/// runtime index value, not a function of loop coordinate alone. Its real
/// count instead comes from the row-level witness `fill_gather_cursors`
/// already builds during execution (`instrument::commit_gather_operand_access`
/// reads it back), scaled by the table's own row width
/// (`Lookup::element_stride`) to report elements rather than rows.
#[cfg(feature = "instrument")]
pub(super) fn record_bound_op_operand_access<B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
) {
    for (source, layout, gather) in resolved.operands() {
        let strides: Vec<i64> = (0..resolved.extents.len() as u16)
            .map(|axis| layout.stride(axis))
            .collect();
        let (reads, distinct) = operand_access_footprint(&resolved.extents, &strides);
        let total_elements = buffer_of(buffers, *source).map(<[f32]>::len).unwrap_or(0) as u64;
        match gather {
            Some(lookup) => {
                let row_width = lookup.element_stride.unsigned_abs();
                instrument::commit_gather_operand_access(*source, reads, row_width, total_elements);
            }
            None => {
                instrument::record_operand_access(*source, reads, distinct, total_elements);
            }
        }
    }
}

pub(super) fn operand_buffers<'a, B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &'a [Option<B>],
) -> Result<Vec<&'a [f32]>, TensorError> {
    resolved
        .operands()
        .iter()
        .map(|(source, _, _)| buffer_of(buffers, *source))
        .collect()
}

/// A raw gather-index buffer element: whatever width the source buffer
/// stores a fetched row index at, reduced to the one signed 64-bit value
/// [`GatherCursor::fetch_and_advance`] bounds-checks and scales by
/// `element_stride`. `f32` is the f32 pipeline's own index width (every
/// buffer that pipeline carries, including `indices`, is f32 — see
/// [`crate::map::IndexMap`]'s own doc); `i64` is the typed evaluator's
/// canonical index width ([`canonical_index_buffers`]'s own doc). No other
/// width ever backs a [`GatherCursor`] directly.
pub(super) trait GatherIndexElement: Copy {
    fn as_gather_index(self) -> i64;
}

impl GatherIndexElement for f32 {
    fn as_gather_index(self) -> i64 {
        self as i64
    }
}

impl GatherIndexElement for i64 {
    fn as_gather_index(self) -> i64 {
        self
    }
}

/// Per-step gather state for one operand: an incrementally-advanced offset
/// into the `indices` buffer (mirroring how a normal operand's own running
/// offset advances by a precomputed stride each step), plus what to do with
/// a fetched value once read. `E` is the index buffer's own element width
/// ([`GatherIndexElement`]); it defaults to `f32`, the f32 pipeline's only
/// width, so every existing call site naming `GatherCursor<'a>` keeps
/// compiling unchanged. [`fill_gather_cursors_typed`] is the only source of
/// `GatherCursor<'a, i64>`.
pub(super) struct GatherCursor<'a, E = f32> {
    pub(super) buffer: &'a [E],
    pub(super) offset: i64,
    pub(super) stride: i64,
    pub(super) element_stride: i64,
    pub(super) extent: u64,
}

impl<E: GatherIndexElement> GatherCursor<'_, E> {
    /// Reads the next index, advances the cursor, and returns the offset
    /// contribution that index adds to the operand's own running offset — a
    /// real error, not a clamp or a wraparound, when the fetched index falls
    /// outside the gathered dim's extent.
    pub(super) fn fetch_and_advance(&mut self, node: NodeId) -> Result<i64, TensorError> {
        let raw = self.buffer[self.offset as usize];
        self.offset += self.stride;
        let index = raw.as_gather_index();
        if index < 0 || index as u64 >= self.extent {
            return Err(TensorError::GatherIndexOutOfRange {
                node,
                index,
                extent: self.extent,
            });
        }
        Ok(index * self.element_stride)
    }
}

/// Fills one [`GatherCursor`] per operand that gathers (`None` for the
/// rest) into a caller-owned buffer, each initialized at `coordinate` and
/// advancing by `stride_dim`'s stride per step — `stride_dim` is `None`
/// where there is no per-step dimension at all (a scalar reduction's single
/// accumulator).
///
/// Writes into `cursors` in place rather than returning a fresh `Vec`: this
/// runs once per reduction step (up to ~1e6 times for a 1024^3 GEMM), and
/// `cursors` is the caller's reused scratch buffer, sized once to operand
/// count outside the hot loop (`proxima-tensor/docs/discipline.md` ROW 2).
///
/// Under the `instrument` feature, this is also the row-level witness point
/// for [`instrument::record_gather_row`]: seeding a cursor already reads
/// this row's index value's OFFSET into the indices tensor
/// (`gather_access.index_layout.offset_of(coordinate)`) as part of normal,
/// already-paid-for addressing, so reading the raw index value itself here
/// too — once per row, never per element `fetch_and_advance` steps through —
/// piggybacks on that instead of adding a second traversal.
pub(super) fn fill_gather_cursors<'a, B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &'a [Option<B>],
    coordinate: &[u64],
    stride_dim: Option<u16>,
    cursors: &mut [Option<GatherCursor<'a>>],
) -> Result<(), TensorError> {
    for (slot, (source, _, gather)) in cursors.iter_mut().zip(resolved.operands()) {
        #[cfg(not(feature = "instrument"))]
        let _ = source;
        *slot = gather
            .as_ref()
            .map(|gather_access| {
                let buffer = buffer_of(buffers, gather_access.indices)?;
                let offset = gather_access.index_layout.offset_of(coordinate);
                #[cfg(feature = "instrument")]
                {
                    let row_index = buffer[offset as usize] as i64;
                    if row_index >= 0 {
                        instrument::record_gather_row(*source, row_index as u64);
                    }
                }
                Ok(GatherCursor {
                    buffer,
                    offset,
                    stride: stride_dim.map_or(0, |dim| gather_access.index_layout.stride(dim)),
                    element_stride: gather_access.element_stride,
                    extent: gather_access.extent,
                })
            })
            .transpose()?;
    }
    Ok(())
}

/// [`fill_gather_cursors`]'s typed counterpart: sources each cursor's raw
/// index values from `index_buffers` — the canonical `i64` table
/// [`canonical_index_buffers`] builds once at execution start — rather than
/// the operand buffer table `fill_gather_cursors` reads from. The typed
/// evaluator's index nodes carry their own integer dtype, never the
/// program's compute dtype, so they cannot live in the same `buffers: &[T]`
/// table a gathered operand's own values do.
pub(super) fn fill_gather_cursors_typed<'a>(
    resolved: &BoundOp,
    index_buffers: &'a [Option<Vec<i64>>],
    coordinate: &[u64],
    stride_dim: Option<u16>,
    cursors: &mut [Option<GatherCursor<'a, i64>>],
) -> Result<(), TensorError> {
    for (slot, (source, _, gather)) in cursors.iter_mut().zip(resolved.operands()) {
        #[cfg(not(feature = "instrument"))]
        let _ = source;
        *slot = gather
            .as_ref()
            .map(|gather_access| {
                let buffer = index_buffers[gather_access.indices.0 as usize]
                    .as_deref()
                    .ok_or(TensorError::NotLowerable {
                        node: gather_access.indices,
                        reason: "gather index buffer missing at evaluation time",
                    })?;
                let offset = gather_access.index_layout.offset_of(coordinate);
                #[cfg(feature = "instrument")]
                {
                    let row_index = buffer[offset as usize];
                    if row_index >= 0 {
                        instrument::record_gather_row(*source, row_index as u64);
                    }
                }
                Ok(GatherCursor {
                    buffer,
                    offset,
                    stride: stride_dim.map_or(0, |dim| gather_access.index_layout.stride(dim)),
                    element_stride: gather_access.element_stride,
                    extent: gather_access.extent,
                })
            })
            .transpose()?;
    }
    Ok(())
}

/// Recomputes each operand's running byte offset for a fresh coordinate,
/// writing into the caller's reused `running` buffer instead of collecting a
/// new `Vec` — the per-position counterpart of [`fill_gather_cursors`].
pub(super) fn fill_running_offsets(resolved: &BoundOp, coordinate: &[u64], running: &mut [i64]) {
    for (slot, (_, view, _)) in running.iter_mut().zip(resolved.operands()) {
        *slot = view.offset_of(coordinate);
    }
}

/// Dispatches one elementwise node across the cohort when a `session` is
/// open, the node clears [`PARALLEL_THRESHOLD`], and there is more than one
/// outer position to spread across workers. [`BoundOp::split`] only chunks
/// the outermost *axis* (`split_axis`'s own doc), which for this program's
/// shapes is the sequence dim — 6 for the forward pass this was measured
/// against, smaller than `workers` on any real box, so every elementwise
/// node fell through to the sequential fallback and the split never fired
/// (`DIAG elementwise_split_none`, measured against every node above
/// threshold). The fix chunks the same *flattened* outer-position space
/// [`run_elementwise`]'s own loop already walks instead: each outer
/// position writes an independent, contiguous `inner_len`-wide row of
/// `output` (`fill_running_offsets`/`fill_gather_cursors` reseed fresh from
/// that position's own coordinate every iteration, so no state carries
/// across positions — see their own docs), so a contiguous range of
/// positions is exactly as independent as [`matmul_rows_threaded`]'s row
/// ranges, without needing [`BoundOp::split`]'s single-axis rebase at all.
/// Reuses [`row_chunk_count`] (rows = outer positions, contraction width =
/// `inner_len`) for the same oversubscription/macs-floor policy
/// [`matmul_rows_threaded`] already tunes, rather than a second policy for
/// this axis. Falls straight through to [`run_elementwise`] whenever any
/// gate fails: no session, too few elements, or too few outer positions to
/// clear even a one-chunk-per-worker split.
///
/// Whether `resolved` is a whole-row gather over a single operand with an
/// `Identity` body — the exact shape `proxima-tensor/src/spec.rs`'s
/// `embedding_lookup` emits (`table[ids[s], d]`), and the only
/// [`BoundOpKind::Elementwise`] shape a packed, non-`Float32` operand can
/// answer without dequantizing anything but the rows actually read. A
/// second operand, or any body step beyond a bare passthrough, falls back
/// to the ordinary f32 [`run_elementwise_dispatch`] path.
pub(super) fn quantized_gather_operand(resolved: &BoundOp) -> Option<(NodeId, bind::Lookup)> {
    if !matches!(resolved.kind, BoundOpKind::Elementwise { .. }) {
        return None;
    }
    let body = resolved.element_body();
    if body.steps.len() != 1 || body.steps[0].op != ScalarOp::Identity {
        return None;
    }
    let [(source, _layout, Some(lookup))] = resolved.operands() else {
        return None;
    };
    Some((*source, lookup.clone()))
}

/// [`quantized_gather_operand`]'s executor: one embedding row per output
/// row, decoded straight out of `block`'s own packed bytes into `output`'s
/// row slice — [`dequantize_row`] never sees, and never allocates, anything
/// but the [`bind::Lookup::element_stride`]-wide row a given index selects.
/// This is what keeps `token_embd.weight` off the owned-`Vec<f32>` path
/// `proxima_model_interop::bind::bind_dense_as`'s own doc used to require:
/// the full table stays exactly as packed as every other `Q4_K`/`Q5_K`/
/// `Q6_K`/`Q8_0` weight this crate already keeps resident, and only the rows
/// a real decode step actually asks for are ever turned into f32.
pub(super) fn run_embedding_gather_quantized<B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    lookup: &bind::Lookup,
    block: &QuantizedBlock<'_>,
    output: &mut [f32],
) -> Result<(), TensorError> {
    let shape_error = || TensorError::NotLowerable {
        node: resolved.node,
        reason: "quantized embedding gather row width does not divide the output buffer evenly",
    };
    let dim = usize::try_from(lookup.element_stride).map_err(|_| shape_error())?;
    if dim == 0 || !output.len().is_multiple_of(dim) {
        return Err(shape_error());
    }
    let indices = buffer_of(buffers, lookup.indices)?;
    #[cfg(feature = "instrument")]
    debug!(
        node = resolved.node.0,
        extent = lookup.extent,
        element_stride = dim,
        output_len = output.len(),
        "quantized embedding gather addressing"
    );
    let mut coordinate = vec![0u64; resolved.extents.len().max(1)];
    for (row, out_row) in output.chunks_exact_mut(dim).enumerate() {
        coordinate[0] = row as u64;
        let index_offset = usize::try_from(lookup.index_layout.offset_of(&coordinate))
            .map_err(|_| shape_error())?;
        let raw_index = *indices.get(index_offset).ok_or_else(shape_error)?;
        let row_index = raw_index as i64;
        if row_index < 0 || row_index as u64 >= lookup.extent {
            return Err(TensorError::GatherIndexOutOfRange {
                node: resolved.node,
                index: row_index,
                extent: lookup.extent,
            });
        }
        dequantize_row(resolved.node, block, row_index as usize, dim, out_row)
            .map_err(|_| shape_error())?;
        #[cfg(feature = "instrument")]
        if row == 0 {
            debug!(
                node = resolved.node.0,
                raw_index,
                row_index,
                first = out_row.first().copied().unwrap_or(0.0),
                second = out_row.get(1).copied().unwrap_or(0.0),
                third = out_row.get(2).copied().unwrap_or(0.0),
                fourth = out_row.get(3).copied().unwrap_or(0.0),
                "quantized embedding gather first row"
            );
        }
    }
    Ok(())
}

pub(super) fn run_embedding_gather_expert_source<B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    lookup: &bind::Lookup,
    source: ExpertSource<'_>,
    output: &mut [f32],
) -> Result<(), TensorError> {
    let dim = usize::try_from(lookup.element_stride).map_err(|_| TensorError::NotLowerable {
        node: resolved.node,
        reason: "expert gather element width does not fit host usize",
    })?;
    if dim == 0 || !output.len().is_multiple_of(dim) {
        return Err(TensorError::NotLowerable {
            node: resolved.node,
            reason: "expert gather output is not a whole number of expert rows",
        });
    }
    let indices = buffer_of(buffers, lookup.indices)?;
    let coordinate_len = resolved.extents.len().max(1);
    let mut coordinate = vec![0u64; coordinate_len];
    let source_node = resolved
        .operands()
        .iter()
        .find_map(|(node, _, gather)| gather.as_ref().map(|_| *node))
        .unwrap_or(resolved.node);
    for (row, output_row) in output.chunks_exact_mut(dim).enumerate() {
        coordinate[0] = row as u64;
        let index_offset =
            usize::try_from(lookup.index_layout.offset_of(&coordinate)).map_err(|_| {
                TensorError::NotLowerable {
                    node: resolved.node,
                    reason: "expert gather index offset does not fit host usize",
                }
            })?;
        let raw_index = *indices.get(index_offset).ok_or(TensorError::NotLowerable {
            node: resolved.node,
            reason: "expert gather index buffer is shorter than its declared layout",
        })?;
        let expert_index = raw_index as usize;
        let entry = source.entries().get(expert_index).copied().ok_or(
            TensorError::GatherIndexOutOfRange {
                node: source_node,
                index: raw_index as i64,
                extent: source.entries().len() as u64,
            },
        )?;
        let expected_elements = usize::try_from(entry.out_dim)
            .ok()
            .and_then(|out| {
                usize::try_from(entry.in_dim)
                    .ok()
                    .and_then(|input| out.checked_mul(input))
            })
            .ok_or(TensorError::NotLowerable {
                node: resolved.node,
                reason: "expert entry shape overflows host usize",
            })?;
        if expected_elements != dim {
            return Err(TensorError::ExpertSourceShapeMismatch {
                node: source_node,
                expert: expert_index as u32,
                entry_out: entry.out_dim,
                entry_in: entry.in_dim,
                expected_out: dim as u32,
                expected_in: 1,
            });
        }
        dequantize_row(resolved.node, &entry.block, 0, dim, output_row).map_err(|_| {
            TensorError::NotLowerable {
                node: resolved.node,
                reason: "expert gather entry cannot decode its declared row width",
            }
        })?;
    }
    Ok(())
}

/// One row's worth of dequantized elements, decoded directly from `block`'s
/// packed bytes at `row_index`'s byte offset — no allocation, no table-wide
/// pass. `dim` must be a multiple of the codec's own block width (`QK_K`
/// for the K-quant family, `QK8_0` for `Q8_0`); every real GGUF embedding
/// width this crate has bound (Qwen3's 1024 and 2560, both multiples of
/// 256) satisfies this, and a checkpoint whose embedding width does not is
/// reported here rather than silently mis-decoded.
pub(super) fn dequantize_row(
    node: NodeId,
    block: &QuantizedBlock<'_>,
    row_index: usize,
    dim: usize,
    output: &mut [f32],
) -> Result<(), TensorError> {
    let unaligned_row = || TensorError::NotLowerable {
        node,
        reason: "quantized embedding row width does not divide the codec's own block width",
    };
    if let QuantizedBlock::Int32(_) = block {
        unreachable!("integer index blocks never enter quantized dot")
    }
    // Row-level decode only reaches codecs with a `dequantize` arm below --
    // `Q4_0`/`Q5_0`/`Float16`/`BFloat16` do have a `QuantizedBlock::block_layout`
    // entry, but no per-row dequantize path here, so they stay excluded
    // rather than silently decoding through the shared table.
    if matches!(
        block,
        QuantizedBlock::Float32(_)
            | QuantizedBlock::Packed { codec: Codec::Q4_0, bytes: _ }
            | QuantizedBlock::Packed { codec: Codec::Q5_0, bytes: _ }
            | QuantizedBlock::Packed { codec: Codec::Float16, bytes: _ }
            | QuantizedBlock::Packed { codec: Codec::BFloat16, bytes: _ }
    ) {
        return Err(unaligned_row());
    }
    let data = block.packed_bytes().ok_or_else(unaligned_row)?;
    let (block_bytes, block_elements) = block.block_layout().ok_or_else(unaligned_row)?;
    if !dim.is_multiple_of(block_elements) {
        return Err(unaligned_row());
    }
    let row_bytes = (dim / block_elements) * block_bytes;
    let start = row_index * row_bytes;
    let row_bytes_slice = data
        .get(start..start + row_bytes)
        .ok_or_else(unaligned_row)?;
    let decode = block.dequantize_fn().ok_or_else(unaligned_row)?;
    decode(row_bytes_slice, output).map_err(|_| unaligned_row())
}

/// [`evaluate_quantized_with_scratch`]'s own pre-materialization pass: if
/// `node` names a still-unbound (`buffers[node] == None`) entry in
/// `quantized_weights`, dequantizes its WHOLE buffer via [`dequantize_row`]
/// (`row_index: 0`, `dim` the node's own total element count -- a single
/// "row" spanning the entire tensor, exactly [`dequantize_row`]'s own
/// `row_bytes = (dim / block_elements) * block_bytes` arithmetic collapsed
/// to one call) and stores it. A no-op whenever `node` already has a
/// buffer or is not a quantized weight at all, which is every ordinary
/// decode call -- this only does real work for the diagnostic window this
/// defect (`docs/discipline.md`, `int8-logs`) was reported against, never
/// inside the per-element hot loop.
pub(super) fn materialize_quantized_weight_output(
    node: NodeId,
    shapes: &shape::Shapes,
    quantized_weights: &BTreeMap<NodeId, QuantizedBlock>,
    buffers: &mut [Option<Cow<'_, [f32]>>],
) -> Result<(), TensorError> {
    if buffers[node.0 as usize].is_some() {
        return Ok(());
    }
    let Some(block) = quantized_weights.get(&node) else {
        return Ok(());
    };
    #[cfg(feature = "instrument")]
    debug!(
        node = node.0,
        elements = element_count(shapes.of(node)),
        "materializing quantized weight for requested output"
    );
    let mut dequantized = vec![0.0f32; element_count(shapes.of(node))];
    dequantize_row(node, block, 0, dequantized.len(), &mut dequantized)?;
    buffers[node.0 as usize] = Some(Cow::Owned(dequantized));
    Ok(())
}
