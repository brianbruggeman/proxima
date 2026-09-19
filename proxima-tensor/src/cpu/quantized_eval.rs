use super::*;

/// [`evaluate`]'s counterpart for a program with one `Q4_K`-quantized weight
/// operand — the entry point that actually reaches [`matmul_q4k_f32`], which
/// [`evaluate`]/[`evaluate_parallel`] cannot: their `blocks: &[&[f32]]`
/// parameter is f32-only by construction, so neither has anywhere to put a
/// packed byte buffer. This function is that seam: [`Codec::Q4K`]
/// entries are held back from the f32 buffer table and instead collected
/// into a `NodeId -> &[u8]` side table that `run_reduce` consults (via
/// `quantized_operand`) for the one `Reduce` node `is_quantized_matmul_operand`
/// already proves is shaped for it — every other node in `program` still
/// runs the exact same f32 path [`evaluate`] does, unchanged.
///
/// `evaluate_typed`'s `TypedBuffer` seam was considered and rejected for
/// this: even `typed_program_plan`'s `Widened` shape only crosses dtypes
/// once, at a `Reduce` node's own accumulator boundary, but a quantized
/// matmul is mixed *within* one fused reduce body — `UInt8`-packed weight
/// times `Float32` activation into a `Float32` output — which is exactly
/// the shape `reject_non_float32`'s quantized-weight exemption carves out,
/// not a program `evaluate_typed` would ever accept. See that function's own
/// doc (`typed_program_plan`'s `TypedPlan`) for the two shapes it does
/// accept.
///
/// Every call starts and ends with an empty node-output pool and an
/// unvalidated structure cache — see [`evaluate_quantized_with_scratch`]
/// for the same contract with caller-carried state that survives across
/// calls to the same `program`.
pub fn evaluate_quantized(
    program: &[Op],
    symbols: &[u64],
    blocks: &[QuantizedBlock],
    outputs: &[NodeId],
) -> Result<Evaluated, TensorError> {
    let mut free_buffers: Vec<Vec<f32>> = Vec::new();
    let mut validated_weight_nodes: Option<BTreeSet<NodeId>> = None;
    evaluate_quantized_with_scratch(
        program,
        symbols,
        blocks,
        outputs,
        &mut free_buffers,
        &mut validated_weight_nodes,
    )
}

/// [`evaluate_quantized`] with the CPU's `Q4_K`/`Q5_K`/`Q6_K` dots run
/// through their exact dequantize-then-fold kernels
/// ([`matmul_q4k_f32`]/`matmul_q5k_f32`/`matmul_q6k_f32`) rather than the
/// `q{4,5,6}k-int8-dot` activation-quantized fast path those features
/// default on. Exists so a cross-backend parity harness (Metal's kernels
/// are exact against an f64 dequant reference) can compare against a CPU
/// reference that is ALSO exact, instead of misattributing the int8 path's
/// own ~1e-3 relative error to the other backend.
pub fn evaluate_quantized_exact(
    program: &[Op],
    symbols: &[u64],
    blocks: &[QuantizedBlock],
    outputs: &[NodeId],
) -> Result<Evaluated, TensorError> {
    let mut free_buffers: Vec<Vec<f32>> = Vec::new();
    let mut validated_weight_nodes: Option<BTreeSet<NodeId>> = None;
    evaluate_quantized_exact_with_scratch(
        program,
        symbols,
        blocks,
        outputs,
        &mut free_buffers,
        &mut validated_weight_nodes,
    )
}

/// Same contract as [`evaluate_quantized`], plus two capabilities a caller
/// cannot get from that function, both aimed at the same shape of caller: a
/// decode loop that evaluates the *same* `program` once per generated
/// token, where only `symbols` (`cached_len` growing by one) actually
/// changes between calls and every weight stays put.
///
/// `free_buffers` is [`evaluate_with_scratch`]'s own reuse pool, applied to
/// this evaluator's loop the same way this crate's internal `evaluate_pooled`
/// already applies it to [`evaluate`]/[`evaluate_parallel`] — a private
/// `take_or_allocate` helper hands a node its output storage from the pool
/// instead of a fresh `vec![0.0; n]`, and a private `retire_into` helper
/// returns a retired node's owned storage to the pool instead of dropping
/// it. `evaluate_quantized` did neither: every one of a program's nodes
/// paid a fresh heap allocation on every call, measured at 3.2-3.7 ms/step
/// of a ~68 ms cached-decode step on the real checkpoint this crate's own
/// `DIAG … loop_overhead_ms` reports.
///
/// `validated_weight_nodes` caches this module's private
/// `reject_non_float32` gate's outcome across calls. That gate's cost is
/// `O(quantized weight count * program.len())` — for every node this
/// call's `blocks` tags as a packed weight, a private
/// `is_quantized_matmul_operand` helper rescans the whole program to prove
/// it is used in a matmul shape — and neither `program` nor which nodes are
/// weight-typed changes between decode steps, so the same outcome is valid
/// on every call after the first. Measured at 1.9-2.0 ms/step on the real
/// checkpoint's 1196-node cached-forward program, roughly half of
/// `DIAG … setup_ms`. A call whose `blocks` classifies a different set of
/// nodes as weight-typed than the cached run (a genuinely different
/// program shape, not a decode step) invalidates the cache and pays the
/// full gate again — the comparison against the cached
/// `BTreeSet<NodeId>` is what decides that,
/// not a size or pointer check that a coincidental match could fool.
///
/// `evaluate_quantized` is exactly this function with `free_buffers` and
/// `validated_weight_nodes` starting, and ending, empty — the same
/// relationship [`evaluate`] has to [`evaluate_with_scratch`].
pub fn evaluate_quantized_with_scratch(
    program: &[Op],
    symbols: &[u64],
    blocks: &[QuantizedBlock],
    outputs: &[NodeId],
    free_buffers: &mut Vec<Vec<f32>>,
    validated_weight_nodes: &mut Option<BTreeSet<NodeId>>,
) -> Result<Evaluated, TensorError> {
    evaluate_quantized_with_scratch_impl(
        program,
        symbols,
        blocks,
        outputs,
        free_buffers,
        validated_weight_nodes,
        false,
        None,
    )
}

/// [`evaluate_quantized_with_scratch`]'s exact-activation counterpart --
/// see [`evaluate_quantized_exact`] for why this path exists.
pub fn evaluate_quantized_exact_with_scratch(
    program: &[Op],
    symbols: &[u64],
    blocks: &[QuantizedBlock],
    outputs: &[NodeId],
    free_buffers: &mut Vec<Vec<f32>>,
    validated_weight_nodes: &mut Option<BTreeSet<NodeId>>,
) -> Result<Evaluated, TensorError> {
    evaluate_quantized_with_scratch_impl(
        program,
        symbols,
        blocks,
        outputs,
        free_buffers,
        validated_weight_nodes,
        true,
        None,
    )
}

/// [`evaluate_quantized_with_scratch`] plus one capability neither it nor
/// [`evaluate_quantized_exact_with_scratch`] carries: a caller (the model's
/// own decode loop, via `proxima-model-interop`'s `ExpertSlab`) hands a
/// per-step `expert_sources` table -- weight `NodeId` to that node's own
/// [`ExpertSource`] snapshot -- and every gathered MoE reduce in `program`
/// resolves through it instead of through the plain contiguous-stack read.
/// `expert_sources: None` is exactly [`evaluate_quantized_with_scratch`];
/// this function exists so that caller never needs a second copy of the
/// setup/loop/finish body above it to get the extra argument through.
pub fn evaluate_quantized_with_scratch_and_experts(
    program: &[Op],
    symbols: &[u64],
    blocks: &[QuantizedBlock],
    outputs: &[NodeId],
    free_buffers: &mut Vec<Vec<f32>>,
    validated_weight_nodes: &mut Option<BTreeSet<NodeId>>,
    expert_sources: Option<&BTreeMap<NodeId, ExpertSource<'_>>>,
) -> Result<Evaluated, TensorError> {
    evaluate_quantized_with_scratch_impl(
        program,
        symbols,
        blocks,
        outputs,
        free_buffers,
        validated_weight_nodes,
        false,
        expert_sources,
    )
}

/// [`evaluate_quantized_with_scratch_and_experts`]'s exact-activation
/// counterpart, exactly as [`evaluate_quantized_exact_with_scratch`] is to
/// [`evaluate_quantized_with_scratch`].
pub fn evaluate_quantized_exact_with_scratch_and_experts(
    program: &[Op],
    symbols: &[u64],
    blocks: &[QuantizedBlock],
    outputs: &[NodeId],
    free_buffers: &mut Vec<Vec<f32>>,
    validated_weight_nodes: &mut Option<BTreeSet<NodeId>>,
    expert_sources: Option<&BTreeMap<NodeId, ExpertSource<'_>>>,
) -> Result<Evaluated, TensorError> {
    evaluate_quantized_with_scratch_impl(
        program,
        symbols,
        blocks,
        outputs,
        free_buffers,
        validated_weight_nodes,
        true,
        expert_sources,
    )
}

/// [`evaluate_quantized_with_scratch_impl`]'s own pre-materialization pass:
/// a quantized weight read anywhere other than a `Reduce` fold's own
/// primary `operands()` (that path stays on `run_reduce_with_quantized_
/// weights`, which reads `quantized_weights` directly) needs its bytes in
/// `buffers` before evaluation, or `buffer_of` finds the slot `None` and
/// raises `NotLowerable`. An `Elementwise` node's `operands()` and a
/// `Reduce`'s own `epilogue_operands` (`apply_reduce_epilogue`) are the two
/// shapes that can name such a weight today.
pub(super) fn materialize_quantized_weights_read_by_non_primary_operands(
    resolved: &[BoundOp],
    shapes: &shape::Shapes,
    quantized_weights: &BTreeMap<NodeId, QuantizedBlock>,
    expert_sources: Option<&BTreeMap<NodeId, ExpertSource<'_>>>,
    buffers: &mut [Option<Cow<'_, [f32]>>],
) -> Result<(), TensorError> {
    for computed in resolved {
        let sources: &[(NodeId, bind::Layout, Option<bind::Lookup>)] = match &computed.kind {
            BoundOpKind::Elementwise { .. } => computed.operands(),
            BoundOpKind::Reduce {
                epilogue_operands, ..
            }
            | BoundOpKind::RoundBatchedReduce {
                epilogue_operands, ..
            } => epilogue_operands,
            BoundOpKind::CachedAttention { .. }
            | BoundOpKind::GatedDeltaNet { .. }
            | BoundOpKind::MoeTopK { .. }
            | BoundOpKind::Iota
            | BoundOpKind::Constant { .. } => &[],
        };
        for (operand, ..) in sources {
            if operand.0 == 6540 {
                eprintln!(
                    "DIAG node6540 reader={} in_quantized_weights={} in_expert_sources={} buffer_some_before={}",
                    computed.node.0,
                    quantized_weights.contains_key(operand),
                    expert_sources.is_some_and(|sources| sources.contains_key(operand)),
                    buffers[operand.0 as usize].is_some(),
                );
            }
            if expert_sources.is_some_and(|sources| sources.contains_key(operand)) {
                continue;
            }
            materialize_quantized_weight_output(*operand, shapes, quantized_weights, buffers)?;
        }
    }
    Ok(())
}

// every `NodeId` the fused kernel dereferences at `fire_position` --
// `x` plus the tail's own four operand slots -- resolved from the tail
// `BoundOp`'s operand list rather than assumed contiguous with `x_node`,
// since `compose_body`'s canonicalization does not guarantee any fixed
// layout (`LayerNormClusterPlan`'s own doc on `tail_gamma_slot` et al.).
pub(super) fn layer_norm_cluster_operand_nodes(
    resolved: &[BoundOp],
    cluster: &LayerNormClusterPlan,
) -> Option<[NodeId; 5]> {
    let BoundOpKind::Elementwise {
        operands: tail_operands,
        ..
    } = &resolved[cluster.tail_index].kind
    else {
        return None;
    };
    Some([
        cluster.x_node,
        tail_operands.get(cluster.tail_reciprocal_n_slot)?.0,
        tail_operands.get(cluster.tail_epsilon_slot)?.0,
        tail_operands.get(cluster.tail_gamma_slot)?.0,
        tail_operands.get(cluster.tail_beta_slot)?.0,
    ])
}

// `expert_sources` is this slice's own addition, pushing this shared body
// one argument past clippy's default threshold; every one of its five
// public callers already threads its own six/seven positional arguments
// straight through, so a params struct here would cost more call-site
// churn than the eighth argument costs a reader.
#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_quantized_with_scratch_impl(
    program: &[Op],
    symbols: &[u64],
    blocks: &[QuantizedBlock],
    outputs: &[NodeId],
    free_buffers: &mut Vec<Vec<f32>>,
    validated_weight_nodes: &mut Option<BTreeSet<NodeId>>,
    exact_activations: bool,
    expert_sources: Option<&BTreeMap<NodeId, ExpertSource<'_>>>,
) -> Result<Evaluated, TensorError> {
    // brackets the portion of evaluate_quantized that is neither the
    // per-node loop below nor run_node_into itself -- shape::infer,
    // bind::bind, node_retirement, and the buffers table setup all run
    // here. Committed once, at the end of the call, via
    // `instrument::record_evaluate_quantized_phase` -- see that function's
    // own doc for why this replaced a per-call `DIAG` eprintln.
    #[cfg(feature = "instrument")]
    let setup_started = instrument::read_ticks();
    let shapes = shape::infer(program, symbols)?;
    let block_nodes = block_node_ids(program);
    if blocks.len() != block_nodes.len() {
        return Err(TensorError::InputCountMismatch {
            expected: block_nodes.len(),
            found: blocks.len(),
        });
    }

    let mut quantized_weights: BTreeMap<NodeId, QuantizedBlock> = BTreeMap::new();
    let mut buffers: Vec<Option<Cow<[f32]>>> = vec![None; program.len()];
    for (node, block) in block_nodes.iter().zip(blocks.iter().copied()) {
        match block {
            QuantizedBlock::Float32(data) => {
                let expected = element_count(shapes.of(*node));
                if data.len() != expected {
                    return Err(TensorError::InputSizeMismatch {
                        node: *node,
                        expected,
                        found: data.len(),
                    });
                }
                buffers[node.0 as usize] = Some(Cow::Borrowed(data));
            }
            QuantizedBlock::Int32(data) => {
                let expected = element_count(shapes.of(*node));
                if data.len() != expected {
                    return Err(TensorError::InputSizeMismatch {
                        node: *node,
                        expected,
                        found: data.len(),
                    });
                }
                buffers[node.0 as usize] =
                    Some(Cow::Owned(data.iter().map(|&value| value as f32).collect()));
            }
            QuantizedBlock::Packed { .. } => {
                quantized_weights.insert(*node, block);
            }
        }
    }

    let root = program
        .len()
        .checked_sub(1)
        .map(|last| NodeId(last as u32))
        .ok_or(TensorError::Empty)?;
    for output in outputs {
        if output.0 as usize >= program.len() {
            return Err(TensorError::UnknownOutput(*output));
        }
    }
    let effective_outputs: Vec<NodeId> = if outputs.is_empty() {
        vec![root]
    } else {
        outputs.to_vec()
    };

    let quantized_weight_nodes: BTreeSet<NodeId> = quantized_weights.keys().copied().collect();
    if validated_weight_nodes.as_ref() != Some(&quantized_weight_nodes) {
        reject_non_float32(program, &quantized_weight_nodes)?;
        // cloned, not moved: the outputs-only check right below still needs
        // its own copy of this same set — see `reject_non_float32_outputs`'s
        // own doc for why that check cannot ride this cache.
        *validated_weight_nodes = Some(quantized_weight_nodes.clone());
    }
    reject_non_float32_outputs(program, &quantized_weight_nodes, &effective_outputs)?;

    let resolved = bind::bind(
        program,
        &shapes,
        &effective_outputs,
        NumericPolicy::bit_exact(),
    )?;
    // Packed matmul lowering is valid only when the weight's contraction axes
    // are physically contiguous.  Materialize the exceptional non-contiguous
    // views before execution so `run_reduce` can honor their index maps; this
    // is the same one-time setup seam used for requested quantized outputs.
    // A caller-requested output can force a node that ordinary decode always
    // fuses away into its own standalone `BoundOp` -- `node_retirement`
    // below correctly keeps ANY node in `effective_outputs` alive, but
    // "alive" and "independently computable" are different questions.
    // `is_quantized_matmul_operand`'s own shape (an `Elementwise` multiply
    // against a `Q4_K`/`Q5_K`/`Q3_K`/`Q6_K`/`Q8_0` weight, feeding exactly
    // one `Reduce::Add`) is ordinarily consumed ONLY by
    // `run_reduce_with_quantized_weights`, which reads `quantized_weights`
    // directly and never touches `buffers` for that operand -- but once the
    // multiply itself is a requested output (or merely shares this call's
    // `effective_outputs` window with something that keeps it live), `bind`
    // stops fusing it into the reduce (the same `outputs`-liveness check
    // that protects every other requested node), and it materializes
    // standalone through the plain `run_elementwise_dispatch` path, which
    // has no quantized-operand awareness at all: `buffer_of` finds the
    // weight's slot `None` (its real bytes live in `quantized_weights`, a
    // *node* was never dequantized into `buffers`) and raises
    // `NotLowerable { reason: "operand buffer missing at evaluation time" }`
    // naming the WEIGHT, not the multiply that actually needed evaluating.
    // Dequantizing any such weight into `buffers` here -- once, at bind
    // setup, never inside the per-element hot loop -- closes the gap: it is
    // a no-op scan over `resolved` on every ordinary decode call (no
    // `Elementwise` node there ever reads a still-unbound quantized weight,
    // since the fusion above already absorbed it into its reduce), and only
    // does real work on exactly the diagnostic window this defect was
    // reported against (`docs/discipline.md`, `int8-logs`).
    //
    // A `Reduce`'s epilogue is the same gap under a different fusion: an
    // RMSNorm gamma multiply fused into `epilogue_operands` (`apply_reduce_
    // epilogue`, `run_node.rs`) reads `buffer_of` too, but `quantized_
    // operand`/`run_reduce_with_quantized_weights` only ever inspect the
    // fold's own `operands()` -- the epilogue's quantized weight is never
    // that fold's primary operand, so nothing else in this function
    // dequantizes it. Widening this scan to `epilogue_operands` (never
    // `operands()` for a `Reduce`, which stays on the direct-from-
    // `quantized_weights` path above) closes that second gap the same way.
    materialize_quantized_weights_read_by_non_primary_operands(
        &resolved,
        &shapes,
        &quantized_weights,
        expert_sources,
        &mut buffers,
    )?;
    // The weight's own `NodeId` can ALSO be named directly in
    // `effective_outputs` (a caller inspecting a raw checkpoint tensor, not
    // just the activation that multiplies it) with no live `Elementwise`
    // consumer at all left in `resolved` -- e.g. its usual multiply fused
    // cleanly into its reduce because that multiply itself was never a
    // requested output. The scan above never visits such a weight (nothing
    // in `resolved` reads it as a plain operand), so it needs this second,
    // direct pass over the request itself.
    for &output in &effective_outputs {
        if expert_sources.is_some_and(|sources| sources.contains_key(&output)) {
            continue;
        }
        materialize_quantized_weight_output(output, &shapes, &quantized_weights, &mut buffers)?;
    }
    let retires = node_retirement(&resolved, &effective_outputs);
    // ROW 181 profile-gate probe: `evaluate_named` no longer reaches this
    // loop (it routes through `evaluate_named_via_arena` ->
    // `run_resolved_nodes_in_arena` instead, see that function's own doc)
    // -- what still walks HERE is `evaluate_quantized_named`/`evaluate_quantized`/
    // `evaluate` whenever a real quantized (non-`Float32`) block is present,
    // `StaticArena` carrying no quantized-weight support today.
    #[cfg(feature = "epilogue-profile-probe")]
    let epilogue_profile_reduce_nodes = epilogue_profile_reduce_flags(&resolved, program.len());
    // `docs/discipline.md` ROW 183 Phase 2 / ROW 204, now driven through
    // `run_rewrite_worklist` (`docs/rewrite-algebra.md` §8): reduce `NodeId`
    // -> (consumer's `resolved` index, fire position), computed once per
    // call from `resolved`'s own already-bound structure (never `bind::bind`
    // itself). `epilogue_fuse_skip` is the plan's own value set, re-derived
    // rather than stored twice, so there is exactly one source of truth for
    // "which consumer position gets skipped". `epilogue_fuse_fire_at` groups
    // the SAME plan by its own `fire_position` value, so the main loop can
    // ask "does anything fire here" in O(1) per position instead of
    // scanning the whole plan every iteration.
    let (mut epilogue_fuse_plan, mut layer_norm_cluster_plan, rewrite_fires) = run_rewrite_worklist(
        &resolved,
        program.len(),
        &effective_outputs,
        &quantized_weights,
    );
    #[cfg(feature = "std")]
    if std::env::var_os("PROXIMA_DISABLE_LAYER_NORM_CLUSTER").is_some() {
        layer_norm_cluster_plan.clear();
    }
    #[cfg(feature = "std")]
    if std::env::var_os("PROXIMA_DISABLE_REWRITE_ENGINE").is_some() {
        epilogue_fuse_plan.clear();
        layer_norm_cluster_plan.clear();
    }
    record_rewrite_engine_fires(&rewrite_fires);
    let epilogue_fuse_skip: BTreeSet<NodeId> = epilogue_fuse_plan
        .values()
        .map(|(index, ..)| resolved[*index].node)
        .chain(
            layer_norm_cluster_plan
                .values()
                .flat_map(|cluster| cluster.skip),
        )
        .collect();
    let epilogue_fuse_fire_at: BTreeMap<usize, Vec<NodeId>> = {
        let mut grouped: BTreeMap<usize, Vec<NodeId>> = BTreeMap::new();
        for (reduce_node, (_, fire_position, ..)) in &epilogue_fuse_plan {
            // superseded by a cluster upgrade -- that fires via
            // `layer_norm_cluster_fire_at` instead, never both.
            if layer_norm_cluster_plan.contains_key(reduce_node) {
                continue;
            }
            grouped
                .entry(*fire_position)
                .or_default()
                .push(*reduce_node);
        }
        grouped
    };
    let layer_norm_cluster_fire_at: BTreeMap<usize, Vec<NodeId>> = {
        let mut grouped: BTreeMap<usize, Vec<NodeId>> = BTreeMap::new();
        for (reduce_node, cluster) in &layer_norm_cluster_plan {
            grouped
                .entry(cluster.fire_position)
                .or_default()
                .push(*reduce_node);
        }
        grouped
    };
    // `docs/discipline.md` ROW 204: `node_retirement`'s own schedule (built
    // above, before this plan existed) frees `x` at `E2`'s real original
    // position -- often EARLIER than a cluster's own `fire_position`, since
    // `gamma`/`beta` are commonly lowered right next to the `LayerNorm`
    // site itself, AFTER `E2` in program order (confirmed against the real
    // BGE graph: every one of the 25 sites' own `fire_position` measured
    // 2-3 positions past `E2`'s). `layer_norm_cluster_keepalive` names
    // every such node and the position it must survive to;
    // `layer_norm_cluster_retire_at` is that SAME information regrouped by
    // where the deferred retirement actually fires -- two views of one
    // fact, not two sources of truth (built from each other, one line
    // down).
    let layer_norm_cluster_keepalive: BTreeMap<NodeId, usize> = {
        let mut keepalive: BTreeMap<NodeId, usize> = BTreeMap::new();
        for cluster in layer_norm_cluster_plan.values() {
            // the fused tail at `fire_position` reads `x` AND all four tail
            // operand slots straight out of `buffers` -- keying this map on
            // `x_node` alone left `gamma`/`beta`/`reciprocal_n`/`epsilon`
            // free to retire at their own pre-fusion last use, before the
            // deferred read (ROW 204 follow-up, node 6540).
            let Some(nodes) = layer_norm_cluster_operand_nodes(&resolved, cluster) else {
                continue;
            };
            for node in nodes {
                keepalive
                    .entry(node)
                    .and_modify(|existing| *existing = (*existing).max(cluster.fire_position))
                    .or_insert(cluster.fire_position);
            }
        }
        keepalive
    };
    let layer_norm_cluster_retire_at: BTreeMap<usize, Vec<NodeId>> = {
        let mut grouped: BTreeMap<usize, Vec<NodeId>> = BTreeMap::new();
        for (&node, &keep_until) in &layer_norm_cluster_keepalive {
            grouped.entry(keep_until).or_default().push(node);
        }
        grouped
    };

    // peak live-buffer BYTE count (not slot count -- a 12-entry live set
    // could be 12 MiB or 12 GiB), fed to
    // `instrument::record_evaluate_quantized_phase` once at the end of the
    // call rather than a per-call `DIAG` eprintln. Gated (unlike
    // `peak_live_buffers` itself below, which `finish` needs
    // unconditionally): an instrument-off build pays neither the closure
    // call nor its `O(program.len())` rescan per node.
    #[cfg(feature = "instrument")]
    let live_bytes = |buffers: &[Option<Cow<[f32]>>]| -> usize {
        buffers
            .iter()
            .flatten()
            .map(|cow| cow.len() * core::mem::size_of::<f32>())
            .sum()
    };

    // `peak_live_buffers` is real, always-returned `Evaluated` state (see
    // `finish` below and `Evaluated::peak_live_buffers`'s own tests), so it
    // stays unconditional -- but tracked via `live_now`, an O(1) running
    // count incremented/decremented alongside the loop's own `Some`/`None`
    // writes, rather than `live_count(&buffers)`'s O(program.len()) full
    // rescan called once per node (the actual quadratic cost this landing
    // removes: `O(program.len())` work times `program.len()` nodes).
    let mut peak_live_buffers = live_count(&buffers);
    let mut live_now = peak_live_buffers;
    #[cfg(feature = "instrument")]
    let mut peak_live_bytes = live_bytes(&buffers);
    // everything in the loop body below OTHER than `run_node_into` -- the
    // output `Vec`'s zero-fill allocation and the live-bytes bookkeeping
    // after the call. Committed once, at the end of the call, via
    // `instrument::record_evaluate_quantized_phase`; costs nothing outside
    // an instrumented run.
    #[cfg(feature = "instrument")]
    let mut loop_overhead_ticks: u64 = 0;
    #[cfg(feature = "instrument")]
    let setup_ticks = instrument::elapsed_ticks(setup_started);
    // Entered ONCE, before the node loop -- not per matmul call -- so the
    // wake this session amortizes is paid once per forward pass, the same
    // amortization `prime/src/os/cohort.rs`'s own module doc measures
    // against `ProximaBackgroundPool`'s per-call wake. `None` whenever
    // another forward already holds the process-wide cohort
    // (`ThreadCohort::enter` returning `Err`) or the cohort itself failed
    // to build; every one of the six signatures between here and
    // `matmul_rows_threaded` falls back to the `nest_pool` dispatch path in
    // that case, unchanged.
    let session = nest_cohort().and_then(|cohort| cohort.enter().ok());
    // `docs/discipline.md` ROW 140's own cache: scoped to this ONE
    // `evaluate_quantized_with_scratch` call (one decode/prefill step),
    // never carried across steps -- a later step's `activation_node` slot
    // holds a genuinely different value, and this vec is dropped with the
    // rest of this function's locals at the end of the call, so there is no
    // staleness window to reason about. Keyed by the activation's own
    // `NodeId` rather than threaded through `MatmulSession`/`CohortSession`
    // (`prime::os::cohort`): that type is a generic thread-cohort round
    // driver with no knowledge of `NodeId`/`Q8_K`, so extending it would
    // mean teaching a reusable concurrency primitive one tensor-specific
    // cache -- the same reuse-first question this crate already answers by
    // keeping `free_buffers`/`validated_weight_nodes` as this function's own
    // scratch parameters instead of bolting them onto a foreign type.
    //
    // `NodeId` is a position in `program`'s own flat `Vec<Op>` (`op.rs`'s
    // own doc), the exact same fact `buffers` above already exploits
    // (`vec![None; program.len()]`, indexed by `node.0 as usize`) -- so this
    // cache is sized and indexed identically, an O(1) direct slot lookup
    // instead of a `BTreeMap`'s per-hit key-comparison chain over up to 96
    // hits/step (ROW 140's own measured hit count).
    #[cfg(feature = "cohort-staged-graph")]
    let mut staged_quantize_cache: Vec<Option<Arc<[u8]>>> = vec![None; program.len()];
    let mut position = 0usize;
    while position < resolved.len() {
        #[cfg(feature = "cohort-staged-graph")]
        if !exact_activations && let Some(session_ref) = session.as_ref() {
            let run_end = staged_batch_run_end(&resolved, position, &quantized_weights);
            if run_end - position >= STAGED_BATCH_MIN_LEN {
                #[cfg(feature = "instrument")]
                let batch_started = instrument::read_ticks();
                run_staged_batch(
                    &resolved[position..run_end],
                    position,
                    &mut buffers,
                    &quantized_weights,
                    expert_sources,
                    session_ref,
                    free_buffers,
                    &retires,
                    &mut live_now,
                    &mut staged_quantize_cache,
                )?;
                peak_live_buffers = peak_live_buffers.max(live_now);
                #[cfg(feature = "instrument")]
                {
                    loop_overhead_ticks += instrument::elapsed_ticks(batch_started);
                }
                position = run_end;
                continue;
            }
        }
        let computed = &resolved[position];
        // `docs/discipline.md` ROW 183 Phase 2: this position's own node is a
        // matched epilogue consumer whose value the PRODUCER's own position
        // already wrote into `buffers[computed.node]` (see the
        // `epilogue_fuse_plan` write below) -- nothing left to compute here,
        // but `retires[position]` still runs unconditionally below, exactly
        // matching ROW 167's "computed once, cheap to consult" skip shape
        // (never touches `bind::bind`, never touches `node_retirement`'s own
        // schedule).
        let epilogue_fused_away = epilogue_fuse_skip.contains(&computed.node);
        if !epilogue_fused_away {
            #[cfg(feature = "instrument")]
            let alloc_started = instrument::read_ticks();
            let mut output = take_or_allocate(free_buffers, node_output_len(computed));
            #[cfg(feature = "instrument")]
            {
                loop_overhead_ticks += instrument::elapsed_ticks(alloc_started);
            }
            #[cfg(feature = "epilogue-profile-probe")]
            let epilogue_profile_started = std::time::Instant::now();
            // `state_out` (ROW 547, `docs/discipline.md`): `GatedDeltaNet`'s
            // own second output, kept empty and unread for every other kind.
            // `moe_topk_extra` (ROW 569) is the same shape, generalized.
            let mut gdn_state = Vec::new();
            let mut moe_topk_extra = Vec::new();
            let mut round_extra: Vec<Vec<f32>> = Vec::new();
            run_node_into_with_round_sink(
                computed,
                &buffers,
                Some(&quantized_weights),
                expert_sources,
                session.as_ref(),
                exact_activations,
                &mut output,
                Some(&mut gdn_state),
                Some(&mut moe_topk_extra),
                Some(&mut round_extra),
            )?;
            #[cfg(feature = "epilogue-profile-probe")]
            epilogue_profile_record(
                computed,
                &epilogue_profile_reduce_nodes,
                epilogue_profile_started.elapsed().as_nanos() as u64,
            );
            #[cfg(feature = "instrument")]
            let bookkeeping_started = instrument::read_ticks();
            buffers[computed.node.0 as usize] = Some(Cow::Owned(output));
            if let BoundOpKind::GatedDeltaNet { state_out, .. } = &computed.kind {
                buffers[state_out.0 as usize] = Some(Cow::Owned(gdn_state));
            }
            if let BoundOpKind::MoeTopK {
                routes,
                weights,
                weight_total,
                ..
            } = &computed.kind
            {
                for (extra_node, value) in moe_topk_extra_node_order(routes, weights, *weight_total)
                    .zip(moe_topk_extra.iter().copied())
                {
                    buffers[extra_node.0 as usize] = Some(Cow::Owned(vec![value]));
                }
            }
            // round_outputs[0] is `computed.node` itself, already written
            // above -- `round_outputs[1..]` are the k-1 round-sibling nodes
            // `bind::apply_moe_round_group_fusion` dropped from the resolved
            // list entirely, each now placed here at its own buffer slot the
            // same way `MoeTopK`'s own extra outputs are, immediately above.
            if let BoundOpKind::RoundBatchedReduce { round_outputs, .. } = &computed.kind {
                for (extra_node, value) in round_outputs.iter().skip(1).zip(round_extra) {
                    buffers[extra_node.0 as usize] = Some(Cow::Owned(value));
                }
            }
            #[cfg(feature = "std")]
            if std::env::var_os("PROXIMA_CPU_TRACE_COMPUTE").is_some()
                && matches!(
                    computed.node,
                    NodeId(20)
                        | NodeId(25)
                        | NodeId(85)
                        | NodeId(88)
                        | NodeId(96)
                        | NodeId(98)
                        | NodeId(106)
                        | NodeId(109)
                )
            {
                let values = buffers[computed.node.0 as usize].as_deref().unwrap_or(&[]);
                eprintln!(
                    "cpu_trace_compute node={:?} position={} len={} checksum={} operands={:?}",
                    computed.node,
                    position,
                    values.len(),
                    values.iter().copied().sum::<f32>(),
                    computed
                        .operands()
                        .iter()
                        .map(|(node, _, _)| {
                            let buffer = buffers[node.0 as usize].as_deref();
                            (
                                *node,
                                buffer.map(|v| (v.len(), v.iter().copied().sum::<f32>())),
                            )
                        })
                        .collect::<Vec<_>>()
                );
            }
            // `computed.node` is written exactly once (this position, in
            // program order), so this is always a `None` -> `Some` transition --
            // `live_now += 1` is the O(1) replacement for rescanning `buffers`.
            live_now += 1;
            peak_live_buffers = peak_live_buffers.max(live_now);
            #[cfg(feature = "instrument")]
            {
                peak_live_bytes = peak_live_bytes.max(live_bytes(&buffers));
            }
            #[cfg(feature = "instrument")]
            {
                loop_overhead_ticks += instrument::elapsed_ticks(bookkeeping_started);
            }
        }
        // `docs/discipline.md` ROW 183 Phase 2: fire every fusion whose
        // `fire_position` is THIS position, regardless of whether this
        // position's own node was a normal compute above or itself a
        // fused-away consumer skip -- the reduce's raw values live in
        // `buffers[reduce_node]` (written either earlier, by that node's own
        // `computed.node.0` store above, or in an EARLIER loop iteration),
        // never re-read via a stale local `output`. Runs BEFORE this
        // position's own `retires` below so a broadcast operand this fusion
        // still needs cannot have been freed first (`node_retirement`'s own
        // schedule only retires a node at ITS real last consumer's original
        // position, always >= `fire_position` by construction).
        if let Some(reduce_nodes) = epilogue_fuse_fire_at.get(&position) {
            for reduce_node in reduce_nodes {
                let Some(&(consumer_index, _, kind, hoist_axis, epilogue_slots)) =
                    epilogue_fuse_plan.get(reduce_node)
                else {
                    continue;
                };
                let consumer = &resolved[consumer_index];
                let reduce_values = buffers[reduce_node.0 as usize].as_deref().unwrap_or(&[]);
                let mut fused_output = take_or_allocate(free_buffers, node_output_len(consumer));
                let fuse_started = std::time::Instant::now();
                apply_epilogue_fused_monomorphic(
                    EpilogueFuseKernel {
                        kind,
                        hoist_axis,
                        slots: epilogue_slots,
                    },
                    consumer,
                    *reduce_node,
                    reduce_values,
                    &buffers,
                    &mut fused_output,
                );
                EPILOGUE_FUSE_NANOS.fetch_add(
                    fuse_started.elapsed().as_nanos() as u64,
                    EpilogueFuseOrdering::Relaxed,
                );
                EPILOGUE_FUSE_HITS.fetch_add(1, EpilogueFuseOrdering::Relaxed);
                EPILOGUE_FUSE_ELEMENTS
                    .fetch_add(fused_output.len() as u64, EpilogueFuseOrdering::Relaxed);
                buffers[consumer.node.0 as usize] = Some(Cow::Owned(fused_output));
                live_now += 1;
                peak_live_buffers = peak_live_buffers.max(live_now);
            }
        }
        // `docs/discipline.md` ROW 204: same shape as the block above, one
        // level wider -- fires the full `R1..R2 -> tail` cluster kernel
        // instead of a single-hop epilogue, reading `x` directly rather
        // than a materialized reduce buffer.
        if let Some(reduce_nodes) = layer_norm_cluster_fire_at.get(&position) {
            for reduce_node in reduce_nodes {
                let Some(cluster) = layer_norm_cluster_plan.get(reduce_node) else {
                    continue;
                };
                let tail = &resolved[cluster.tail_index];
                let mut fused_output = take_or_allocate(free_buffers, node_output_len(tail));
                let fuse_started = std::time::Instant::now();
                apply_layer_norm_cluster_fused(
                    tail,
                    cluster.x_node,
                    cluster.row_axis,
                    LayerNormTailSlots {
                        reciprocal_n: cluster.tail_reciprocal_n_slot,
                        epsilon: cluster.tail_epsilon_slot,
                        gamma: cluster.tail_gamma_slot,
                        beta: cluster.tail_beta_slot,
                    },
                    &buffers,
                    &mut fused_output,
                );
                LAYER_NORM_CLUSTER_NANOS.fetch_add(
                    fuse_started.elapsed().as_nanos() as u64,
                    EpilogueFuseOrdering::Relaxed,
                );
                LAYER_NORM_CLUSTER_HITS.fetch_add(1, EpilogueFuseOrdering::Relaxed);
                LAYER_NORM_CLUSTER_ELEMENTS
                    .fetch_add(fused_output.len() as u64, EpilogueFuseOrdering::Relaxed);
                buffers[tail.node.0 as usize] = Some(Cow::Owned(fused_output));
                live_now += 1;
                peak_live_buffers = peak_live_buffers.max(live_now);
            }
        }
        for retired in &retires[position] {
            // NOT always a `Some` -> `None` transition, unlike
            // `evaluate_pooled`/`evaluate_parallel`'s identical loop: a
            // `Q4_K`/`Q5_K`/`Q6_K`/`Q8_0` weight node is scheduled for
            // retirement here exactly like any other operand (`node_retirement`
            // builds its schedule from the generic program graph, with no
            // knowledge of the quantized/float32 split), but it never occupied
            // a `buffers` slot in the first place -- it lives in
            // `quantized_weights` instead (see the `QuantizedBlock` match
            // above). `retire_into` reports whether the slot was actually
            // live so `live_now` only counts a real retirement.
            //
            // `docs/discipline.md` ROW 204: `node_retirement`'s own
            // schedule was built BEFORE this call's `layer_norm_cluster_plan`
            // existed, from the UNMODIFIED graph -- it still frees `x` at
            // `E2`'s real original position, `E2` being a genuine consumer
            // there regardless of whether this session's fusion later
            // replaces what runs at that position. Any node this session's
            // cluster fusion still needs alive PAST its natural retirement
            // point (`layer_norm_cluster_keepalive`) has that ONE
            // retirement event deferred to the cluster's own
            // `fire_position` instead of dropped -- see
            // `layer_norm_cluster_retire_at` below, which re-fires it there.
            if let Some(&keep_until) = layer_norm_cluster_keepalive.get(retired)
                && position < keep_until
            {
                continue;
            }
            if retire_into(&mut buffers, *retired, free_buffers) {
                // a segmented (gdn-scan producer/tail split) evaluation can retire a
                // node this diagnostic counter never saw incremented; the counter is
                // report-only (`peak_live_buffers`), so clamp rather than trap.
                live_now = live_now.saturating_sub(1);
            }
        }
        if let Some(deferred) = layer_norm_cluster_retire_at.get(&position) {
            for node in deferred {
                if retire_into(&mut buffers, *node, free_buffers) {
                    live_now = live_now.saturating_sub(1);
                }
            }
        }
        position += 1;
    }
    #[cfg(feature = "std")]
    if std::env::var_os("PROXIMA_CPU_TRACE_OUTPUTS").is_some() {
        eprintln!(
            "cpu_trace_outputs output_count={} outputs={effective_outputs:?}",
            effective_outputs.len()
        );
        let mut traced = effective_outputs.clone();
        for node in [NodeId(26), NodeId(88), NodeId(106), NodeId(109)] {
            if !traced.contains(&node) {
                traced.push(node);
            }
        }
        for output in &traced {
            let resolved_position = resolved.iter().position(|bound| bound.node == *output);
            let retirement_position = retires.iter().position(|nodes| nodes.contains(output));
            let buffer = buffers[output.0 as usize].as_deref();
            let checksum = buffer.map(|values| values.iter().copied().sum::<f32>());
            let kind = resolved_position.map(|position| format!("{:?}", resolved[position].kind));
            eprintln!(
                "cpu_trace_output node={output:?} output_count={} resolved_position={resolved_position:?} retirement_position={retirement_position:?} buffer_len={} checksum={checksum:?} kind={kind:?}",
                effective_outputs.len(),
                buffer.map_or(0, <[f32]>::len)
            );
        }
    }
    #[cfg(feature = "instrument")]
    let finish_started = instrument::read_ticks();

    let result = finish(
        &shapes,
        &effective_outputs,
        buffers,
        root,
        peak_live_buffers,
    );
    // real signal (peak live bytes, per-phase wall time) committed once per
    // call into the crate's own counter mechanism -- see
    // `instrument::record_evaluate_quantized_phase`'s own doc for why this
    // replaced a per-call `DIAG` eprintln block (17-30 lines/call, which had
    // inverted the sign of two independent measurements this session by
    // adding stderr-flush cost to the loop it was timing). Everything else
    // the removed block printed (per-node-kind ticks, elementwise phase
    // breakdown, `BodyShape` ns/element splits, reduce-path ticks, call-size
    // histogram) already lives in this module's own `REDUCE_PATH_*`,
    // `ELEMENTWISE_*`, and `MAC_OPS` counters, now readable directly via
    // `instrument::reduce_path_totals`, `instrument::elementwise_phase_totals`,
    // `instrument::elementwise_bodyshape_totals`, and
    // `instrument::elementwise_call_size_snapshot_and_reset` -- a caller
    // wanting a per-call delta resets first, the same
    // reset-then-run-then-read shape `reduce_gemm_path_totals`'s own callers
    // already use.
    #[cfg(feature = "instrument")]
    instrument::record_evaluate_quantized_phase(
        setup_ticks,
        loop_overhead_ticks,
        instrument::elapsed_ticks(finish_started),
        peak_live_bytes as u64,
    );
    Ok(result)
}

/// [`evaluate_quantized`]'s counterpart for binding by name instead of
/// position — the same resolution loop: walk `program`'s [`Op::Input`]
/// nodes in order, look each one's name up in `named`, and hand the
/// resolved positional `blocks: &[QuantizedBlock]` straight to
/// [`evaluate_quantized`]. [`evaluate_named`] no longer routes through
/// here (see that function's own doc: it reaches `evaluate_named_via_arena`
/// (private) directly, since it only ever has
/// `Float32` blocks to hand this loop) — this function remains the entry
/// point for a caller mixing in a real quantized (non-`Float32`) weight
/// block, a capability [`StaticArena`] does not carry.
pub fn evaluate_quantized_named<'block>(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'block>)],
    outputs: &[NodeId],
) -> Result<Evaluated, TensorError> {
    let mut free_buffers: Vec<Vec<f32>> = Vec::new();
    let mut validated_weight_nodes: Option<BTreeSet<NodeId>> = None;
    evaluate_quantized_named_with_scratch(
        program,
        symbols,
        named,
        outputs,
        &mut free_buffers,
        &mut validated_weight_nodes,
    )
}

/// [`evaluate_quantized_named`]'s counterpart to
/// [`evaluate_quantized_with_scratch`] — the same name-to-[`Op::Input`]
/// resolution loop, handing the resolved positional `blocks` and both
/// caller-carried pools straight through rather than duplicating either
/// Resolves a name-keyed block set into the positional order a program's
/// [`Op::Input`] nodes appear in.
///
/// Public and shared rather than private to the CPU evaluator: omega's Metal
/// driver binds blocks positionally too, and a second copy of this mapping
/// is a second thing that can drift from what the program actually declares.
/// The two backends already share [`QuantizedBlock`]; they share how a name
/// becomes a position as well.
///
/// # Errors
/// [`TensorError::UnnamedInput`] if a block input carries no name,
/// [`TensorError::UnboundInputName`] if `named` has no entry for one.
pub fn resolve_named_blocks<'block>(
    program: &[Op],
    named: &[(&str, QuantizedBlock<'block>)],
) -> Result<Vec<QuantizedBlock<'block>>, TensorError> {
    let block_nodes = block_node_ids(program);
    let mut blocks: Vec<QuantizedBlock<'block>> = Vec::with_capacity(block_nodes.len());
    for node in &block_nodes {
        let name = program[node.0 as usize]
            .name()
            .ok_or(TensorError::UnnamedInput(*node))?;
        let data = named
            .iter()
            .find(|(candidate, _)| *candidate == name)
            .map(|(_, data)| *data)
            .ok_or_else(|| TensorError::UnboundInputName(String::from(name)))?;
        blocks.push(data);
    }
    Ok(blocks)
}

pub fn resolve_named_blocks_with_experts<'block>(
    program: &[Op],
    named: &[(&str, QuantizedBlock<'block>)],
    expert_sources: Option<&BTreeMap<NodeId, ExpertSource<'block>>>,
) -> Result<Vec<QuantizedBlock<'block>>, TensorError> {
    let block_nodes = block_node_ids(program);
    let mut blocks = Vec::with_capacity(block_nodes.len());
    for node in block_nodes {
        let name = program[node.0 as usize]
            .name()
            .ok_or(TensorError::UnnamedInput(node))?;
        // A source-backed expert node must win over any stale or placeholder
        // named binding: its codec and per-expert layout are the data the
        // gathered reducer actually executes.
        if let Some(source) = expert_sources.and_then(|sources| sources.get(&node)) {
            let block = source
                .entries()
                .first()
                .map_or(QuantizedBlock::Float32(&[]), |entry| entry.block);
            blocks.push(block);
        } else if let Some(data) = named
            .iter()
            .find(|(candidate, _)| *candidate == name)
            .map(|(_, data)| *data)
        {
            blocks.push(data);
        } else {
            return Err(TensorError::UnboundInputName(String::from(name)));
        }
    }
    Ok(blocks)
}

/// evaluator's body a third time.
pub fn evaluate_quantized_named_with_scratch<'block>(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'block>)],
    outputs: &[NodeId],
    free_buffers: &mut Vec<Vec<f32>>,
    validated_weight_nodes: &mut Option<BTreeSet<NodeId>>,
) -> Result<Evaluated, TensorError> {
    // this name resolution runs before `evaluate_quantized`'s own setup
    // timer starts, so it is invisible to every counter that function
    // commits -- a linear `find` over `named` per weight tensor, O(block
    // count * named count) string compares. Committed via its own
    // counter (`instrument::record_evaluate_quantized_resolve`) rather than
    // folded into `evaluate_quantized`'s phase record, which never sees this
    // call boundary.
    #[cfg(feature = "instrument")]
    let resolve_started = instrument::read_ticks();
    let blocks = resolve_named_blocks(program, named)?;
    #[cfg(feature = "instrument")]
    instrument::record_evaluate_quantized_resolve(instrument::elapsed_ticks(resolve_started));
    evaluate_quantized_with_scratch(
        program,
        symbols,
        &blocks,
        outputs,
        free_buffers,
        validated_weight_nodes,
    )
}

/// [`evaluate_quantized_named_with_scratch`]'s exact-activation counterpart
/// -- see [`evaluate_quantized_exact`] for why this path exists. The
/// caller-facing seam a serving-config-level `exact_activations` toggle
/// (`proxima-model-interop`'s cross-backend quality harness) reaches
/// through, since a real forward step binds its weights by name, never by
/// position.
pub fn evaluate_quantized_named_exact_with_scratch<'block>(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'block>)],
    outputs: &[NodeId],
    free_buffers: &mut Vec<Vec<f32>>,
    validated_weight_nodes: &mut Option<BTreeSet<NodeId>>,
) -> Result<Evaluated, TensorError> {
    let blocks = resolve_named_blocks(program, named)?;
    evaluate_quantized_exact_with_scratch(
        program,
        symbols,
        &blocks,
        outputs,
        free_buffers,
        validated_weight_nodes,
    )
}

/// [`evaluate_quantized_named_with_scratch`] plus
/// [`evaluate_quantized_with_scratch_and_experts`]'s `expert_sources` -- the
/// entry point `proxima-model-interop`'s decode loop calls once it has a
/// `LoadedModel`-owned `ExpertSlab` to snapshot for the step.
pub fn evaluate_quantized_named_with_scratch_and_experts<'block>(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'block>)],
    outputs: &[NodeId],
    free_buffers: &mut Vec<Vec<f32>>,
    validated_weight_nodes: &mut Option<BTreeSet<NodeId>>,
    expert_sources: Option<&BTreeMap<NodeId, ExpertSource<'_>>>,
) -> Result<Evaluated, TensorError> {
    let blocks = resolve_named_blocks_with_experts(program, named, expert_sources)?;
    evaluate_quantized_with_scratch_and_experts(
        program,
        symbols,
        &blocks,
        outputs,
        free_buffers,
        validated_weight_nodes,
        expert_sources,
    )
}

/// [`evaluate_quantized_named_with_scratch_and_experts`]'s exact-activation
/// counterpart, exactly as [`evaluate_quantized_named_exact_with_scratch`]
/// is to [`evaluate_quantized_named_with_scratch`].
pub fn evaluate_quantized_named_exact_with_scratch_and_experts<'block>(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'block>)],
    outputs: &[NodeId],
    free_buffers: &mut Vec<Vec<f32>>,
    validated_weight_nodes: &mut Option<BTreeSet<NodeId>>,
    expert_sources: Option<&BTreeMap<NodeId, ExpertSource<'_>>>,
) -> Result<Evaluated, TensorError> {
    let blocks = resolve_named_blocks_with_experts(program, named, expert_sources)?;
    evaluate_quantized_exact_with_scratch_and_experts(
        program,
        symbols,
        &blocks,
        outputs,
        free_buffers,
        validated_weight_nodes,
        expert_sources,
    )
}

/// Shared body for [`evaluate`] and [`evaluate_with_scratch`] — the only
/// difference between the two public entry points is whether `free_buffers`
/// arrives pre-seeded and is read back by the caller afterward, so that
/// decision is made once, here, by each caller passing its own `Vec` (fresh
/// and discarded, or threaded through `scratch`) rather than by two
/// divergent copies of this loop.
///
/// Drives [`run_node_into`] directly, one node per iteration, rather than
/// through [`Interpreter::fold`]: `Interpreter` exposes no way to hand a
/// node its output storage from a pool instead of a fresh `vec![0.0; n]`
/// (its `Pipe::Out = ()` contract has no room for one), so reusing the same
/// execution primitive both callers already share (`run_node_into`, also
/// `evaluate_parallel`'s) is what keeps this a single behavior rather than a
/// second copy of the per-node dispatch match. `Interpreter` remains exactly
/// as it was for its own callers (the `shapes.and_then(builder)
/// .and_then(Interpreter::new(..))` `Pipe` chain a test in this module
/// exercises directly) — this function simply no longer routes through it.
pub(super) fn evaluate_pooled(
    program: &[Op],
    symbols: &[u64],
    blocks: &[&[f32]],
    outputs: &[NodeId],
    free_buffers: &mut Vec<Vec<f32>>,
) -> Result<Evaluated, TensorError> {
    #[cfg(feature = "instrument")]
    let alloc_site_guard = instrument::AllocSiteGuard::enter(instrument::AllocSite::Prepare);
    let Prepared {
        root,
        shapes,
        effective_outputs,
        mut buffers,
        resolved,
        retires,
    } = prepare(program, symbols, blocks, outputs)?;
    #[cfg(feature = "instrument")]
    drop(alloc_site_guard);

    // `live_now`: O(1) running live-buffer count, incremented/decremented
    // alongside this loop's own `Some`/`None` writes -- see
    // `evaluate_quantized`'s identical `live_now` doc for why this replaces
    // `live_count(&buffers)`'s O(program.len()) full rescan per node (the
    // quadratic cost this landing removes).
    let mut peak_live_buffers = live_count(&buffers);
    let mut live_now = peak_live_buffers;
    for (position, computed) in resolved.iter().enumerate() {
        #[cfg(feature = "instrument")]
        let alloc_site_guard =
            instrument::AllocSiteGuard::enter(instrument::AllocSite::OutputBuffer);
        let mut output = take_or_allocate(free_buffers, node_output_len(computed));
        #[cfg(feature = "instrument")]
        drop(alloc_site_guard);
        run_node_into(computed, &buffers, None, None, None, false, &mut output)?;
        #[cfg(feature = "instrument")]
        record_bound_op_operand_access(computed, &buffers);
        buffers[computed.node.0 as usize] = Some(Cow::Owned(output));
        live_now += 1;
        peak_live_buffers = peak_live_buffers.max(live_now);
        for retired in &retires[position] {
            // `blocks: &[&[f32]]` means every node this evaluator ever
            // touches is float32 and lands in `buffers` -- no quantized-weight
            // split to trip over, unlike `evaluate_quantized` -- but the
            // decrement is still gated on `retire_into`'s liveness report
            // rather than assumed, so this loop stays correct if that ever
            // changes rather than relying on an invariant nothing enforces.
            if retire_into(&mut buffers, *retired, free_buffers) {
                live_now -= 1;
            }
        }
    }

    Ok(finish(
        &shapes,
        &effective_outputs,
        buffers,
        root,
        peak_live_buffers,
    ))
}

/// Takes the buffer at `node`'s slot (leaving `None` behind, exactly as
/// before buffer reuse existed) and, if it was this evaluator's own owned
/// storage rather than a caller-borrowed [`Op::Input`] slice, stashes it in
/// `pool` for [`take_or_allocate`] to hand to a later node instead of the
/// allocator. Sound because this only ever runs after the position that
/// retired `node` has already finished reading every one of its operands
/// (`node_retirement` records a node's *last* read position, and the caller
/// only retires after `run_node_into` for that position has returned) — the
/// buffer is out of `buffers` and not yet read by anything else before it
/// lands in `pool`, so no live reference to it survives the swap.
///
/// Returns whether `node`'s slot actually held a buffer. `node_retirement`
/// schedules a retirement for every operand the program graph reads, with no
/// knowledge of `evaluate_quantized`'s split storage: a `Q4_K`/`Q5_K`/`Q6_K`/
/// `Q8_0` weight node never occupies a `buffers` slot at all (it lives in the
/// separate `quantized_weights` map instead — see the `QuantizedBlock`
/// match in `evaluate_quantized_with_scratch`), so its "retirement" here is a
/// no-op on an already-`None` slot. A caller that decremented a running live
/// count unconditionally on every retirement (as this evaluator's `live_now`
/// used to) drifted low by one per quantized weight and could underflow.
pub(super) fn retire_into(
    buffers: &mut [Option<Cow<'_, [f32]>>],
    node: NodeId,
    pool: &mut Vec<Vec<f32>>,
) -> bool {
    match buffers[node.0 as usize].take() {
        Some(Cow::Owned(buffer)) => {
            pool.push(buffer);
            true
        }
        Some(Cow::Borrowed(_)) => true,
        None => false,
    }
}

/// Hands out `required` elements of storage from `pool` when a sufficiently
/// large entry exists, or allocates fresh otherwise — the one place
/// [`evaluate_pooled`] gets a node's output buffer from. Every element
/// [`run_node_into`]'s callers write is unconditionally overwritten before
/// any node reads it back (`run_elementwise`, `run_reduce`'s NEON/tile/
/// fallback paths, and `run_scan` each write every output position once),
/// so a reused buffer's stale contents never leak into a result;
/// `Vec::resize`'s growth path only fires, and only fills the delta, when
/// `pool` had nothing big enough already, so the fill this replaces shrinks
/// to zero once `pool` reaches its working set's high-water mark.
pub(super) fn take_or_allocate(pool: &mut Vec<Vec<f32>>, required: usize) -> Vec<f32> {
    let best_fit = pool
        .iter()
        .enumerate()
        .filter(|(_, buffer)| buffer.capacity() >= required)
        .min_by_key(|(_, buffer)| buffer.capacity())
        .map(|(index, _)| index);

    match best_fit {
        Some(index) => {
            let mut buffer = pool.swap_remove(index);
            buffer.resize(required, 0.0);
            buffer
        }
        None => vec![0.0f32; required],
    }
}

/// Below this many iteration-space elements, a nest runs the plain
/// sequential path even when `workers > 1`: `std::thread::scope`'s spawn
/// and join overhead outweighs the work for a small nest.
pub(super) use crate::sized::PARALLEL_THRESHOLD;

/// chunk count is `workers * OVERSUBSCRIBE`, not `workers`: equal row
/// counts do not mean equal wall-clock (measured 2.04x spread across 8
/// equal-row chunks of a 1024^3 GEMM), and one chunk per worker leaves no
/// spare chunk for a worker that finishes early to pick up — more chunks
/// than workers lets a fast worker absorb a slow chunk's slack. Only pays
/// off under [`nest_pool`]'s dynamic claiming (see `claim_and_run`), the
/// only chunk dispatch this module has: the puller count [`run_chunks_threaded`]
/// spawns caps at `workers` regardless of `OVERSUBSCRIBE`, so raising this
/// grows the number of chunks a fixed puller count can steal from, without
/// growing the number of threads touching them.
///
/// A `4` was tried on this same mechanism: more chunks than workers gives a
/// work-stealing pool room for a fast puller to absorb a slow chunk's slack,
/// which is structural and not in dispute. The comparison that measured it —
/// 274.75 vs 270.08 mean GFLOPS at 2048^3/4 workers, n=9, 5.9 sigma — never
/// recorded the ambient load it ran under, and the 8-worker cells it was
/// meant to help never cleared their own CoV gate at any sample size tried
/// (n up to 30) under the load present when those cells were measured (5-10,
/// against a stated 2.2 plateau). That is the same evidence shape that
/// produced three false readings for `SPLIT_ALIGNMENT` on this same box:
/// strong sigma inside an unvalidated run does not rule out noise correlated
/// across that run rather than random within it. Left at `1`, the original
/// value, until a re-measurement validates its own floor first — a
/// same-code-path comparison, at a size and load where the two configurations
/// provably execute identical instructions — and only then shows an
/// oversubscription effect outside it.
pub(super) use crate::sized::OVERSUBSCRIBE;

/// Row-alignment applied to every non-final chunk boundary via
/// `BoundOp::split_aligned`. `1` is a no-op (see that method's doc): every
/// chunk boundary lands wherever `extent / chunk_count` puts it, which is
/// not necessarily a multiple of `TILE_ROWS` — so a chunk pays its own
/// row-remainder through the kernel's narrower fallback path independently
/// of every other chunk, and that per-chunk remainder count grows with
/// chunk count even though the total row count did not change. That
/// mechanism is structural and not in dispute; whether it moves
/// busy-per-MAC by a measurable amount is.
///
/// Four measurements of this same `1` -> `TILE_ROWS` change exist, all
/// against the column-panel blocking below (already landed and left on —
/// see `NEON_COLUMN_PANEL_BUDGET_BYTES`). Three, run at system load
/// 12-31, read as 3-10% busy-per-MAC improvements. None of the three
/// established a noise floor before comparing — at that load level a
/// handful of percent between two configurations is not distinguishable
/// from scheduler contention, so those figures are retained here only as
/// unverified prior readings, not as evidence.
///
/// The fourth run validated a floor first: at load 2.49-3.07, `neither` vs
/// `panel` at 512^3 and 1024^3 — sizes where `neon_column_panel_cols`
/// provably clamps the panel to one, i.e. the two configurations execute
/// identical code — agreed to within +/-3.5%, sigma up to 3.1. That is the
/// noise floor any real effect at this load has to clear. Against it,
/// alignment (`1` -> `TILE_ROWS`) on top of the panel measured
/// +2.03% / +0.31% / -0.72% at 512^3 / 1024^3 / 2048^3, 8 threads — inside
/// the floor at every size. No measurable effect anywhere in the one
/// comparison whose noise floor is known.
///
/// Set to `1` on that basis: the only measurement with a validated floor
/// found nothing outside it, and the three load-12-31 figures were never
/// shown to clear their own (unmeasured) noise, so they carry no weight
/// against it. This changes only if a future re-measurement (a) validates
/// its own floor the same way — a same-code-path comparison at a size
/// where the panel is a no-op — and (b) then shows an alignment effect
/// outside that floor.
///
/// Provenance: the three load-12-31 figures were not independently dated
/// or sample-counted in the record available to this pass — treat them as
/// unverified, not merely old. The load-2.49-3.07 measurement is this
/// session's own, 2026-08-18; its sample count for the alignment
/// comparison specifically is not broken out beyond the three-configuration
/// grid it ran alongside.
pub(super) use crate::sized::SPLIT_ALIGNMENT;

/// Same contract as [`evaluate`], including the exact same [`Evaluated`]
/// and error variants — the only difference is that each large-enough nest
/// runs its chunks across `workers` pool tasks via `run_chunks_threaded`
/// (the shared `nest_pool`), each writing a disjoint sub-slice of that
/// nest's own output buffer (see [`BoundOp::split`]). The preamble is
/// `prepare`, the same one [`evaluate`] runs — the two functions diverge
/// only in the loop below.
pub fn evaluate_parallel(
    program: &[Op],
    symbols: &[u64],
    blocks: &[&[f32]],
    outputs: &[NodeId],
    workers: NonZeroUsize,
) -> Result<Evaluated, TensorError> {
    #[cfg(feature = "instrument")]
    let evaluate_parallel_start = instrument::read_ticks();

    #[cfg(feature = "instrument")]
    let prepare_start = instrument::read_ticks();
    #[cfg(feature = "instrument")]
    let alloc_site_guard = instrument::AllocSiteGuard::enter(instrument::AllocSite::Prepare);
    let Prepared {
        root,
        shapes,
        effective_outputs,
        mut buffers,
        resolved,
        retires,
    } = prepare(program, symbols, blocks, outputs)?;
    #[cfg(feature = "instrument")]
    drop(alloc_site_guard);
    #[cfg(feature = "instrument")]
    counter!(
        instrument::SERIAL_PREPARE_TICKS,
        instrument::elapsed_ticks(prepare_start)
    );

    // `live_now`: O(1) running live-buffer count -- see `evaluate_quantized`'s
    // identical `live_now` doc for why this replaces `live_count(&buffers)`'s
    // O(program.len()) full rescan per node.
    let mut peak_live_buffers = live_count(&buffers);
    let mut live_now = peak_live_buffers;
    for (position, computed) in resolved.iter().enumerate() {
        let node_output = evaluate_node_parallel(computed, &buffers, workers)?;
        #[cfg(feature = "instrument")]
        let bookkeeping_start = instrument::read_ticks();
        buffers[computed.node.0 as usize] = Some(Cow::Owned(node_output.primary));
        if let BoundOpKind::GatedDeltaNet { state_out, .. } = &computed.kind {
            buffers[state_out.0 as usize] = Some(Cow::Owned(node_output.gdn_state));
        }
        if let BoundOpKind::MoeTopK {
            routes,
            weights,
            weight_total,
            ..
        } = &computed.kind
        {
            for (extra_node, value) in moe_topk_extra_node_order(routes, weights, *weight_total)
                .zip(node_output.moe_topk_extra.iter().copied())
            {
                buffers[extra_node.0 as usize] = Some(Cow::Owned(vec![value]));
            }
        }
        if let BoundOpKind::RoundBatchedReduce { round_outputs, .. } = &computed.kind {
            for (extra_node, value) in round_outputs.iter().skip(1).zip(node_output.round_extra) {
                buffers[extra_node.0 as usize] = Some(Cow::Owned(value));
            }
        }
        live_now += 1;
        peak_live_buffers = peak_live_buffers.max(live_now);
        for retired in &retires[position] {
            // same liveness-gated decrement as `evaluate_pooled`'s identical
            // loop -- `blocks: &[&[f32]]` means no quantized-weight split
            // exists here today, but the decrement stays conditioned on the
            // slot actually having held a buffer rather than assumed.
            if buffers[retired.0 as usize].take().is_some() {
                live_now -= 1;
            }
        }
        #[cfg(feature = "instrument")]
        counter!(
            instrument::SERIAL_BOOKKEEPING_TICKS,
            instrument::elapsed_ticks(bookkeeping_start)
        );
    }

    #[cfg(feature = "instrument")]
    let finish_start = instrument::read_ticks();
    let evaluated = finish(
        &shapes,
        &effective_outputs,
        buffers,
        root,
        peak_live_buffers,
    );
    #[cfg(feature = "instrument")]
    {
        counter!(
            instrument::SERIAL_FINISH_TICKS,
            instrument::elapsed_ticks(finish_start)
        );
        counter!(
            instrument::SERIAL_EVALUATE_PARALLEL_TICKS,
            instrument::elapsed_ticks(evaluate_parallel_start)
        );
        counter!(instrument::SERIAL_EVALUATE_PARALLEL_CALLS, 1);
    }

    Ok(evaluated)
}

/// Runs one node, threaded across `workers` when [`BoundOp::split`] finds it
/// sound and it clears [`PARALLEL_THRESHOLD`]; otherwise the plain
/// sequential path via [`run_node_into`].
/// [`evaluate_node_parallel`]'s own return, widened from a bare `Vec<f32>`
/// once a resolved node could be [`BoundOpKind::GatedDeltaNet`] or
/// [`BoundOpKind::MoeTopK`], each with extra outputs beyond `resolved.node`'s
/// own buffer -- ROW 569 found `evaluate_parallel`'s own loop had never
/// threaded either kind's extra outputs anywhere (the other three real
/// dispatch points -- `Interpreter::fold`, `evaluate_quantized_with_scratch_impl`,
/// `run_resolved_nodes_in_arena` -- already had; this one predates
/// `GatedDeltaNet`'s own `state_out` and was simply never exercised by a
/// test that ran a fused program through it, until this row's own MoE parity
/// test did). Both extra-output kinds ever fire for a chunk-split node is
/// moot: `BoundOp::split_axis` returns `None` for both, so a chunked run
/// never reaches either kind (`evaluate_node_parallel`'s own `chunks` match).
pub(super) struct ParallelNodeOutput {
    pub(super) primary: Vec<f32>,
    pub(super) gdn_state: Vec<f32>,
    pub(super) moe_topk_extra: Vec<f32>,
    pub(super) round_extra: Vec<Vec<f32>>,
}

pub(super) fn evaluate_node_parallel<B: Deref<Target = [f32]> + Sync>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    workers: NonZeroUsize,
) -> Result<ParallelNodeOutput, TensorError> {
    #[cfg(feature = "instrument")]
    let alloc_site_guard = instrument::AllocSiteGuard::enter(instrument::AllocSite::OutputBuffer);
    #[cfg(feature = "instrument")]
    let alloc_start = instrument::read_ticks();
    let mut output = vec![0.0f32; node_output_len(resolved)];
    #[cfg(feature = "instrument")]
    counter!(
        instrument::SERIAL_ALLOC_TICKS,
        instrument::elapsed_ticks(alloc_start)
    );
    #[cfg(feature = "instrument")]
    drop(alloc_site_guard);

    #[cfg(feature = "instrument")]
    let split_start = instrument::read_ticks();
    let above_threshold = element_count(&resolved.extents) >= PARALLEL_THRESHOLD;
    // oversubscribing at `workers == 1` would still spawn `OVERSUBSCRIBE - 1`
    // pool tasks (chunk count alone bounds pool concurrency — see
    // `run_chunks_threaded`'s doc), silently using more physical threads
    // than the caller asked for; only multiply once there is more than one
    // worker to spread chunks across.
    let chunk_count = if workers.get() > 1 {
        workers.get() * OVERSUBSCRIBE
    } else {
        workers.get()
    };
    let chunks = above_threshold
        .then(|| resolved.split_aligned(chunk_count, SPLIT_ALIGNMENT))
        .flatten();
    #[cfg(feature = "instrument")]
    counter!(
        instrument::SERIAL_SPLIT_TICKS,
        instrument::elapsed_ticks(split_start)
    );

    let mut gdn_state = Vec::new();
    let mut moe_topk_extra = Vec::new();
    let mut round_extra: Vec<Vec<f32>> = Vec::new();
    match chunks {
        Some(chunks) => run_chunks_threaded(&chunks, buffers, &mut output, workers)?,
        None => {
            // one node, one dispatch decision — recorded once here, not
            // re-derived from `above_threshold` after the fact, since the
            // `chunks` match is the actual arm that ran.
            #[cfg(feature = "instrument")]
            if above_threshold {
                counter!(instrument::DISPATCH_SEQUENTIAL_SPLIT_UNAVAILABLE, 1);
            } else {
                counter!(instrument::DISPATCH_SEQUENTIAL_BELOW_THRESHOLD, 1);
            }
            #[cfg(feature = "instrument")]
            let sequential_start = instrument::read_ticks();
            run_node_into_with_round_sink(
                resolved,
                buffers,
                None,
                None,
                None,
                false,
                &mut output,
                Some(&mut gdn_state),
                Some(&mut moe_topk_extra),
                Some(&mut round_extra),
            )?;
            #[cfg(feature = "instrument")]
            counter!(
                instrument::SERIAL_SEQUENTIAL_COMPUTE_TICKS,
                instrument::elapsed_ticks(sequential_start)
            );
        }
    }
    // attributed against the PARENT (unsplit) `resolved`, not any one
    // spawned chunk above — see `record_bound_op_operand_access`'s doc for
    // why: a chunk's own shrunk extents would double-count a broadcast
    // operand's footprint once per chunk instead of once for this node.
    #[cfg(feature = "instrument")]
    record_bound_op_operand_access(resolved, buffers);
    Ok(ParallelNodeOutput {
        primary: output,
        gdn_state,
        moe_topk_extra,
        round_extra,
    })
}

/// Runs each of `chunks` through the shared [`nest_pool`] (crossbeam-deque
/// work-stealing, built once and reused for every nest in the process)
/// instead of spawning a fresh OS thread per chunk. `std` implies
/// `tensor-bgpool` (`Cargo.toml`'s `std` feature doc), so this is the only
/// dispatch this module ever compiles — a fresh-`thread::scope`-per-call
/// sibling used to sit here and was removed once nothing could select it
/// anymore.
///
/// Every chunk, including the caller's own, is pulled off one shared
/// `next_index` cursor (see [`claim_and_run`]) instead of being statically
/// assigned: the calling thread and every pool task run the identical pull
/// loop, so a puller that finishes its chunk early goes straight back to
/// the cursor for the next available one rather than idling — this is what
/// lets `OVERSUBSCRIBE > 1` (`chunks.len() > workers`) actually pay off:
/// with a 1:1 static assignment (the previous shape here), a pool task
/// finishing early has nothing further to do even when a sibling chunk is
/// still running long past it. `workers.get() - 1` pool tasks are spawned —
/// the caller's requested puller count, not `chunks.len() - 1` — because
/// [`nest_pool`] is a single process-wide pool sized to `num_cpus`, shared
/// across every call regardless of its own `workers` argument: spawning one
/// task per chunk would let oversubscription silently recruit pool threads
/// past what the caller asked for on any box where `num_cpus > workers`.
/// Each spawned puller still drains the same shared cursor across every
/// chunk, so raising `OVERSUBSCRIBE` still grows the number of chunks a
/// fixed `workers` pullers can steal from, without growing puller count.
/// Completion is a real blocking handoff (`std::sync::mpsc::sync_channel`),
/// not a poll loop: the caller parks in `Receiver::recv` instead of
/// busy-spinning a `Waker::noop` future the way `proxima_primitives::block_on`
/// would.
///
/// A worker panic cannot be resumed on the joining thread the way a
/// `thread::scope`-spawned one could: `ProximaBackgroundPool`'s worker loop
/// wraps every job in `catch_unwind` and discards the payload
/// (`prime/src/os/background.rs`, `worker()`, `let _ = unwind;`),
/// converting a panic into a dropped closure with no way to recover the
/// original payload. That drop takes our own `sync_channel` sender clone
/// with it, so a panicking chunk never reports back; a chunk that never
/// reports is surfaced as `TensorError::ThreadedChunkFailed` instead.
pub(super) fn run_chunks_threaded<B: Deref<Target = [f32]> + Sync>(
    chunks: &[BoundOp],
    buffers: &[Option<B>],
    output: &mut [f32],
    workers: NonZeroUsize,
) -> Result<(), TensorError> {
    #[cfg(feature = "instrument")]
    let slice_carve_start = instrument::read_ticks();
    #[cfg(feature = "instrument")]
    let alloc_site_guard = instrument::AllocSiteGuard::enter(instrument::AllocSite::ChunkSlices);

    let mut slices = Vec::with_capacity(chunks.len());
    let mut remaining = output;
    for chunk in chunks {
        let (this_chunk, rest) = remaining.split_at_mut(node_output_len(chunk));
        slices.push(this_chunk);
        remaining = rest;
    }
    #[cfg(feature = "instrument")]
    drop(alloc_site_guard);
    #[cfg(feature = "instrument")]
    counter!(
        instrument::SERIAL_SLICE_CARVE_TICKS,
        instrument::elapsed_ticks(slice_carve_start)
    );

    if chunks.len() < 2 {
        return match (chunks.first(), slices.into_iter().next()) {
            (Some(chunk), Some(slice)) => {
                run_node_into(chunk, buffers, None, None, None, false, slice)
            }
            _ => Ok(()),
        };
    }

    let pool = nest_pool()?;

    // `buffers` is read-only for the whole call and every spawned chunk
    // needs it; cloning would copy every live intermediate tensor per
    // chunk. its address crosses the pool's 'static spawn bound the same
    // way `par_chunks_mut` (prime/src/os/par.rs:1611-1625) already does for
    // its own slice: cast to usize here, reconstruct unsafely inside the
    // closure. sound because `buffers` outlives every spawned closure — the
    // caller thread drains `result_receiver` for every chunk before this
    // function returns.
    let buffers_address = buffers.as_ptr() as usize;
    let buffers_len = buffers.len();
    // same cast, same soundness argument, for `chunks` itself: every puller
    // now needs random access to an arbitrary chunk, not just the one it
    // was statically handed.
    let chunks_address = chunks.as_ptr() as usize;
    let chunks_len = chunks.len();
    // each chunk's own disjoint output sub-slice, addressed by index so any
    // puller (caller or pool task) can claim any chunk — `Arc` because,
    // unlike `buffers`/`chunks` above, this vector is allocated fresh here
    // rather than borrowed from the caller, so it needs its own shared
    // ownership to reach every spawned closure.
    let slice_addresses: Arc<Vec<(usize, usize)>> = Arc::new(
        slices
            .iter_mut()
            .map(|slice| (slice.as_mut_ptr() as usize, slice.len()))
            .collect(),
    );

    let next_index = Arc::new(AtomicUsize::new(0));
    let (result_sender, result_receiver) = sync_channel(chunks_len);

    #[cfg(feature = "instrument")]
    let node_start = instrument::read_ticks();

    // `workers - 1` pool tasks — the puller count the caller actually asked
    // for, NOT `chunks_len - 1`. `nest_pool` is sized to `num_cpus`, shared
    // and reused process-wide, independent of any one call's `workers`
    // argument: spawning one task per CHUNK (as this used to) rather than
    // one per WORKER let `OVERSUBSCRIBE > 1` silently recruit pool threads
    // past the caller's requested count — up to `num_cpus`, on a box where
    // `num_cpus > workers` — since nothing here otherwise bounds how many
    // of the pool's own threads can be pulling `claim_and_run` at once.
    // Each spawned puller still drains the shared cursor across every one
    // of the `chunks_len` chunks, exactly like the caller does below, so
    // oversubscription still grows the number of *chunks* available to
    // steal without growing the number of *threads* touching them.
    for _ in 0..workers.get() - 1 {
        let sender = result_sender.clone();
        let next_index = Arc::clone(&next_index);
        let slice_addresses = Arc::clone(&slice_addresses);
        // the pool's own returned future only reports back through its
        // internal oneshot channel, which nothing here awaits — completion
        // is reported through `sender` instead, so the future is dropped
        // deliberately rather than driven.
        drop(pool.spawn(move || {
            claim_and_run::<B>(
                &next_index,
                chunks_address,
                chunks_len,
                buffers_address,
                buffers_len,
                &slice_addresses,
                &sender,
            );
            Ok::<(), _>(())
        }));
    }

    #[cfg(feature = "instrument")]
    let spawn_ticks = instrument::elapsed_ticks(node_start);

    // the caller pulls from the same shared cursor as every pool task
    // instead of running one reserved chunk — see this function's doc
    // comment for why. it never sits idle: finishing a chunk sends it
    // straight back to `next_index` for another.
    claim_and_run::<B>(
        &next_index,
        chunks_address,
        chunks_len,
        buffers_address,
        buffers_len,
        &slice_addresses,
        &result_sender,
    );
    drop(result_sender);

    let mut outcomes: Vec<Option<Result<(), TensorError>>> =
        (0..chunks_len).map(|_| None).collect();
    for _ in 0..chunks_len {
        match result_receiver.recv() {
            Ok((index, outcome)) => outcomes[index] = Some(outcome),
            // every sender clone is gone (each spawned closure's clone is
            // dropped whether it sends or panics), so no further chunk will
            // ever report — stop waiting instead of blocking forever on a
            // message that cannot arrive. remaining `None` slots below
            // become `ThreadedChunkFailed`.
            Err(_) => break,
        }
    }

    #[cfg(feature = "instrument")]
    {
        let total_ticks = instrument::elapsed_ticks(node_start);
        counter!(instrument::PARALLEL_NODES, 1);
        counter!(instrument::PARALLEL_NODE_TICKS, total_ticks);
        counter!(instrument::PARALLEL_SPAWN_TICKS, spawn_ticks);
        // join/teardown is whatever wall-clock the node spent that wasn't
        // already charged to spawning the pool tasks — includes the
        // caller's own claim_and_run loop, same as the thread::scope
        // sibling's join/teardown includes its own compute.
        counter!(
            instrument::PARALLEL_JOIN_TICKS,
            total_ticks.saturating_sub(spawn_ticks)
        );
    }

    for (index, outcome) in outcomes.into_iter().enumerate() {
        match outcome {
            Some(result) => result?,
            None => {
                return Err(TensorError::ThreadedChunkFailed {
                    chunk: index + 1,
                    reason: alloc::string::String::from(
                        "worker did not report a result; ProximaBackgroundPool \
                         catches and discards worker panics (see \
                         prime/src/os/background.rs worker())",
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Pulls chunk indices off `next_index` one at a time and runs each to
/// completion, reporting through `sender` — called by both the calling
/// thread and every spawned pool task in [`run_chunks_threaded`], so a
/// puller that finishes early goes straight back for the next available
/// chunk instead of stopping after whichever one it started with.
///
/// # Safety (of the `unsafe` blocks inside)
/// `chunks_address`/`buffers_address` and every `(pointer, len)` pair in
/// `slice_addresses` must stay valid, and each slice address must be unique
/// to its index, for as long as any puller can still observe `next_index`
/// below `chunks_len` — guaranteed by [`run_chunks_threaded`] draining
/// `chunks_len` results from `sender`'s channel before `chunks`, `buffers`,
/// or `output` (the parent of every `slice_addresses` entry) can drop.
/// `fetch_add` never hands out the same index twice, so no two pullers ever
/// touch the same slice.
pub(super) fn claim_and_run<B: Deref<Target = [f32]> + Sync>(
    next_index: &AtomicUsize,
    chunks_address: usize,
    chunks_len: usize,
    buffers_address: usize,
    buffers_len: usize,
    slice_addresses: &[(usize, usize)],
    sender: &SyncSender<(usize, Result<(), TensorError>)>,
) {
    loop {
        let index = next_index.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        if index >= chunks_len {
            return;
        }
        // SAFETY: see this function's doc comment.
        let chunk = unsafe { &*(chunks_address as *const BoundOp).add(index) };
        let chunk_buffers = unsafe {
            core::slice::from_raw_parts(buffers_address as *const Option<B>, buffers_len)
        };
        let (slice_address, slice_len) = slice_addresses[index];
        let chunk_output =
            unsafe { core::slice::from_raw_parts_mut(slice_address as *mut f32, slice_len) };

        #[cfg(feature = "instrument")]
        let chunk_start = instrument::read_ticks();
        #[cfg(feature = "instrument")]
        let cpu_start = instrument::thread_cpu_nanos();
        let outcome = run_node_into(chunk, chunk_buffers, None, None, None, false, chunk_output);
        #[cfg(feature = "instrument")]
        {
            let chunk_ticks = instrument::elapsed_ticks(chunk_start);
            let cpu_nanos = instrument::thread_cpu_nanos() - cpu_start;
            instrument::record_chunk_ticks(chunk_ticks);
            instrument::record_worker_busy_ticks(chunk_ticks);
            instrument::record_worker_cpu_nanos(instrument::CpuWorkload::Elementwise, cpu_nanos);
        }

        let _ = sender.send((index, outcome));
    }
}

/// The pool backing [`run_chunks_threaded`]'s and [`matmul_rows_threaded`]'s
/// chunk dispatch. Built once, on first use, and reused for every nest in
/// the process — a fresh `ProximaBackgroundPool` per node would reintroduce
/// the per-node OS-thread-spawn cost this pool exists to remove.
/// `OnceLock` only memoizes success: a failed build is not cached, so a
/// later call (after whatever exhausted OS thread resources clears up) can
/// retry instead of latching a permanent failure.
pub(super) fn nest_pool() -> Result<Arc<ProximaBackgroundPool>, TensorError> {
    if let Some(pool) = NEST_POOL.get() {
        return Ok(Arc::clone(pool));
    }
    let built = Arc::new(ProximaBackgroundPool::new().map_err(|error| {
        TensorError::ThreadedPoolUnavailable(alloc::format!("build nest thread pool: {error}"))
    })?);
    // `set` can lose a race to a concurrent first caller; either pool is
    // equally valid, so use whichever one actually landed.
    let _ = NEST_POOL.set(Arc::clone(&built));
    Ok(NEST_POOL.get().cloned().unwrap_or(built))
}

pub(super) static NEST_POOL: OnceLock<Arc<ProximaBackgroundPool>> = OnceLock::new();

/// The fixed-cohort spin barrier backing [`matmul_rows_threaded`]'s cohort
/// dispatch (see [`RowRound`]): dedicated member threads that stay parked on
/// an atomic round counter between calls instead of paying
/// `ProximaBackgroundPool`'s per-call `Mutex`+`Condvar` wake
/// (`prime/src/os/cohort.rs`'s own module doc: 2492.7 ns/round vs 19305.5
/// ns/round). Built once, on first use, sized to [`matmul_worker_count`] —
/// the same worker count [`quantized_matmul_workers`] already resolves for
/// the pool path, so a cohort round and a pool dispatch always claim the
/// same number of workers. `None` if the cohort fails to build (e.g. thread
/// spawn exhaustion); callers fall back to [`nest_pool`] in that case.
///
/// `PROXIMA_COHORT_SPIN_POLLS`, if set to a valid integer, overrides
/// [`COHORT_SPIN_POLLS`]; this exists to sweep the spin budget without a
/// rebuild. Read once, inside this same `get_or_init` that already builds
/// the cohort a single time for the process -- no separate `OnceLock` needed.
/// Default (unset) behavior is unchanged.
pub(super) fn nest_cohort() -> Option<&'static MatmulCohort> {
    static COHORT: OnceLock<Option<MatmulCohort>> = OnceLock::new();
    COHORT
        .get_or_init(|| {
            let members = NonZeroUsize::new(matmul_worker_count())?;
            let spin_polls = std::env::var("PROXIMA_COHORT_SPIN_POLLS")
                .ok()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or(COHORT_SPIN_POLLS);
            let config = MatmulCohort::builder()
                .members(members)
                .spin_polls(spin_polls)
                .build();
            ThreadCohort::from_config(config).ok()
        })
        .as_ref()
}

pub(super) fn live_count<B>(buffers: &[Option<B>]) -> usize {
    buffers.iter().filter(|entry| entry.is_some()).count()
}

// `shape::infer` (called before this) already rejects a scatter (a
// data-dependent fold output), any gather whose indices are not an integer
// dtype, and any gathered dim past 2^24 (an f32 index cannot represent a
// larger extent's values exactly), so the only restriction left to enforce
// here is that every OTHER node — every node not itself a gather's
// `indices` — is f32: every buffer slot derefs to `[f32]` regardless of
// which `B: Deref<Target = [f32]>` backs it (owned `Vec<f32>` or borrowed
// `Cow<[f32]>`), indices included (an index value is an exact integer
// carried as f32, per the module doc), so a gather's `indices` node is the
// one deliberate exception to the f32 rule rather than a second buffer kind.
//
// The real boundary this enforces is narrower than "f32-only" now reads:
// this function only gates `evaluate`/`evaluate_parallel`, the SIMD-tuned
// pipeline whose buffers, width-tiling, and dot-fold kernels are `Vec<f32>`
// throughout `cpu.rs` — regeneralizing *that* pipeline to every width is a
// materially bigger change than fits alongside adding one. A non-float32,
// non-gather-index, reduce/scan/gather-free program is not unsupported by
// this crate any more; it runs through [`evaluate_typed`] instead, which
// this function does not gate.
//
// `quantized_weights` is the second, narrower exception this now tolerates:
// a node tagged as a `Q4_K`-packed weight buffer, permitted ONLY when it is
// used exclusively as one operand of a `Reduce` whose body multiplies it
// against another operand and folds with `Add` — a matmul shape, checked by
// [`is_quantized_matmul_operand`] below, the same structural shape
// `neon_tile_plan`/`width_tile_plan` already recognize for the dense f32
// tile. `evaluate`/`evaluate_parallel` still pass an empty set (this is
// additive, not a behavior change for either), because their own `blocks:
// &[&[f32]]` parameter is itself f32-only — accepting the node here proves
// only that the *shape* of a quantized-weight matmul type-checks, not that
// this pipeline can execute one yet. [`matmul_q4k_f32`] is the dedicated,
// separately-tested execution path for that shape today; wiring a quantized
// buffer through `evaluate`'s own `blocks` array and `run_reduce`'s NEON
// tile is the remaining integration work this does not yet do.
pub(super) fn reject_non_float32(
    program: &[Op],
    quantized_weights: &BTreeSet<NodeId>,
) -> Result<(), TensorError> {
    let index_nodes = index_node_ids(program);
    let referenced_nodes = referenced_node_ids(program);
    for (position, expr) in program.iter().enumerate() {
        let node = NodeId(position as u32);
        let is_quantized_weight = quantized_weights.contains(&node)
            && (is_quantized_matmul_operand(program, node)
                || is_quantized_gather_operand(program, node));
        // an `Op::Input` `bind::BoundOpBuilder::push` never materializes into
        // a `BoundOp` (see that match arm's own `Op::Input { .. } => {}`) —
        // it is a pure buffer handle, read directly by whichever node
        // references it, never itself run through `run_node_into`. So an
        // `Input` nothing in `program` references (an ONNX initializer for a
        // shape/index tensor no lowered op still reads, e.g.) can never
        // reach this f32-only interpreter's kernels regardless of its dtype
        // — unlike every other `Op` variant, which `push`/`finish` always
        // materialize into `resolved` and the evaluator's node loop always
        // runs, dead code or not (see `BoundOpBuilder::finish`'s own doc:
        // "either a requested output or dead code, and either way it
        // materializes"). A *referenced* non-float32 `Input` still feeds a
        // node that IS unconditionally evaluated, so it stays rejected.
        let is_unreferenced_input =
            matches!(expr, Op::Input { .. }) && !referenced_nodes.contains(&node);
        if expr.dtype() != DType::Float32
            && !index_nodes.contains(&node)
            && !is_quantized_weight
            && !is_unreferenced_input
        {
            return Err(TensorError::NotLowerable {
                node,
                reason: "this pipeline's buffers and SIMD kernels are f32-only; route a \
                         non-float32 elementwise program through evaluate_typed instead",
            });
        }
    }
    Ok(())
}

/// [`reject_non_float32`]'s dead-leaf exemption is deliberately
/// output-independent — a node's own connectivity to the rest of `program`,
/// nothing about which nodes a given call happens to request — so its
/// result stays valid for [`evaluate_quantized_with_scratch`]'s
/// `validated_weight_nodes` cache across calls whose `outputs` differ, not
/// only calls whose `quantized_weights` differ. But a caller CAN request an
/// otherwise-dead non-`Float32` `Input` directly as an output (this f32-only
/// pipeline still cannot honor that: its own `blocks`/`named` parameters
/// carry no non-`Float32` view for `evaluate`/`evaluate_named` to hand back
/// out, and `evaluate_quantized`'s `QuantizedBlock` non-`Float32` variants
/// are reserved for the matmul-weight shape [`is_quantized_matmul_operand`]
/// recognizes, not a passthrough return value), so this cheap,
/// always-run-per-call, `O(outputs.len())`-plus-one-scan check closes that
/// gap without folding `outputs` into the cached structural pass above.
pub(super) fn reject_non_float32_outputs(
    program: &[Op],
    quantized_weights: &BTreeSet<NodeId>,
    outputs: &[NodeId],
) -> Result<(), TensorError> {
    let index_nodes = index_node_ids(program);
    for &node in outputs {
        let Some(expr) = program.get(node.0 as usize) else {
            continue;
        };
        let is_quantized_weight = quantized_weights.contains(&node)
            && (is_quantized_matmul_operand(program, node)
                || is_quantized_gather_operand(program, node));
        if expr.dtype() != DType::Float32 && !index_nodes.contains(&node) && !is_quantized_weight {
            return Err(TensorError::NotLowerable {
                node,
                reason: "this pipeline's buffers and SIMD kernels are f32-only; route a \
                         non-float32 elementwise program through evaluate_typed instead",
            });
        }
    }
    Ok(())
}

/// Every node referenced anywhere in `program` as an `Elementwise` operand, a
/// `Reduce`'s own operand, or either map's computed `indices` — the
/// complement of the set an `Op::Input` must fall outside of for
/// [`reject_non_float32`]'s dead-leaf exemption: a node in this set feeds
/// some other node that [`bind::BoundOpBuilder`] always materializes into a
/// [`BoundOp`](crate::bind::BoundOp), so its dtype still matters even when
/// that consumer is itself unreachable from any requested output.
pub(super) fn referenced_node_ids(program: &[Op]) -> BTreeSet<NodeId> {
    let mut nodes = BTreeSet::new();
    for expr in program {
        match expr {
            Op::Input { .. } | Op::Iota { .. } | Op::Constant { .. } => {}
            Op::Elementwise { operands, .. } => {
                for (operand, map) in operands {
                    nodes.insert(*operand);
                    push_indices_node(map, &mut nodes);
                }
            }
            Op::Reduce(fold) => {
                nodes.insert(fold.operand);
                push_indices_node(&fold.in_map, &mut nodes);
                push_indices_node(&fold.out_map, &mut nodes);
            }
        }
    }
    nodes
}

/// Whether `node` appears, anywhere in `program`, ONLY as one operand of a
/// `Multiply` [`Op::Elementwise`] — the OTHER operand `Float32` AND tracing
/// back to a real [`Op::Input`] ([`operand_traces_to_a_real_input`]'s own
/// check) — that itself feeds directly into a `Reduce` whose `body` is
/// `Add`: the exact "quantized weight x f32 activation" matmul shape
/// [`reject_non_float32`]'s quantized-weight exemption requires. A
/// quantized node used any other way (paired with a second non-float32
/// operand, a second elementwise op, a scan, a reduce with a different
/// combiner, or a partner built PURELY from index values with no real data
/// anywhere in its ancestry) does not qualify: the exemption is for the one
/// shape [`matmul_q4k_f32`] actually implements, not a blanket "trust the
/// caller's tag."
///
/// The "traces to a real `Input`" clause is this recognizer's actual axis
/// guard: reducing the packed weight's own OUTPUT axis instead of its
/// contraction axis — `per_head_channel_slice`'s former call site against a
/// packed weight (`spec.rs`'s `Qwen35DenseAttentionTaps` doc, ROW 428) — is
/// NOT distinguishable from a genuine contraction by axis position alone.
/// This crate's own shipped matmul shapes disagree on which position is
/// "the" contraction axis (`quantized_matmul_program`'s `[rows, k]` reduces
/// its LAST axis; a cached-attention-shaped reduce
/// (`a_reduce_where_activation_and_packed_weight_share_a_kept_output_axis_is_rejected`)
/// reduces its FIRST) — proven by running both through a first-axis-only and
/// a last-axis-only version of this check and watching each break a
/// DIFFERENT, already-shipped legitimate program (ROW 429's own RED data).
/// What every real activation shares, and `per_head_channel_slice`'s
/// `Op::Iota`-built one-hot mask never has, is a real [`Op::Input`] somewhere
/// in its own ancestry — a select-then-reduce whose "activation" is entirely
/// synthesized from index values is what this rejects instead.
pub(super) fn is_quantized_matmul_operand(program: &[Op], node: NodeId) -> bool {
    let mut used_as_matmul_operand = false;
    for (position, expr) in program.iter().enumerate() {
        match expr {
            Op::Elementwise { body, operands, .. } => {
                if !operands.iter().any(|(source, _)| *source == node) {
                    continue;
                }
                let other_operand_is_real_activation = operands.iter().any(|(source, _)| {
                    *source != node
                        && program[source.0 as usize].dtype() == DType::Float32
                        && operand_traces_to_a_real_input(program, *source)
                });
                if *body != ScalarOp::Multiply
                    || operands.len() != 2
                    || !other_operand_is_real_activation
                {
                    return false;
                }
                let elementwise_node = NodeId(position as u32);
                let feeds_matmul_reduce = program.iter().any(|other| {
                    matches!(other, Op::Reduce(fold) if fold.operand == elementwise_node && fold.body == ScalarOp::Add)
                });
                if !feeds_matmul_reduce {
                    return false;
                }
                used_as_matmul_operand = true;
            }
            Op::Reduce(fold) => {
                if fold.operand == node {
                    // reduced directly, not through a Multiply elementwise —
                    // not the matmul shape this exemption covers.
                    return false;
                }
            }
            Op::Input { .. } | Op::Iota { .. } | Op::Constant { .. } => {}
        }
    }
    used_as_matmul_operand
}

/// Whether `node`'s own definition, or anything upstream of it, is a real
/// [`Op::Input`] — the fact every genuine activation has (it ultimately
/// reads external data) and a purely index-derived tensor (an `Op::Iota`
/// fed through arithmetic and comparisons, never touching real data) never
/// does. `program`'s references point backwards only
/// ([`crate::op`]'s own module doc), so this is a plain DFS over strictly
/// decreasing `NodeId`s and always terminates.
pub(super) fn operand_traces_to_a_real_input(program: &[Op], node: NodeId) -> bool {
    let mut stack = alloc::vec![node];
    let mut visited: BTreeSet<NodeId> = BTreeSet::new();
    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        match &program[current.0 as usize] {
            Op::Input { .. } => return true,
            Op::Iota { .. } | Op::Constant { .. } => {}
            Op::Elementwise { operands, .. } => {
                for (source, index_map) in operands {
                    stack.push(*source);
                    if let IndexMap::Computed { indices, .. } = index_map {
                        stack.push(*indices);
                    }
                }
            }
            Op::Reduce(fold) => {
                stack.push(fold.operand);
                if let IndexMap::Computed { indices, .. } = &fold.in_map {
                    stack.push(*indices);
                }
                if let IndexMap::Computed { indices, .. } = &fold.out_map {
                    stack.push(*indices);
                }
            }
        }
    }
    false
}

/// Whether `node` appears, anywhere in `program`, ONLY as the SOLE operand of
/// an `Identity` [`Op::Elementwise`] addressed through an
/// [`crate::map::IndexMap::Computed`] pattern -- the exact "quantized
/// embedding table" gather shape [`embedding_lookup`](crate::spec::embedding_lookup)
/// builds and [`run_embedding_gather_quantized`] already executes,
/// dequantizing one row at a time straight out of the packed bytes rather
/// than materializing the whole table as f32. [`reject_non_float32`]'s
/// f32-only gate predates that execution path -- it only ever recognized
/// [`is_quantized_matmul_operand`]'s multiply-then-reduce shape, so a
/// quantized table used purely as a gather was rejected here before
/// [`run_node_into`]'s own `quantized_gather_operand` dispatch ever got a
/// chance to run it (see this module's `q8_0_embedding_gather_is_not_rejected_as_non_float32`
/// test, added against exactly that gap).
pub(super) fn is_quantized_gather_operand(program: &[Op], node: NodeId) -> bool {
    let mut used_as_gather_operand = false;
    for expr in program {
        let Op::Elementwise { body, operands, .. } = expr else {
            continue;
        };
        if !operands.iter().any(|(source, _)| *source == node) {
            continue;
        }
        let [(source, index_map)] = operands.as_slice() else {
            return false;
        };
        if *source != node || *body != ScalarOp::Identity {
            return false;
        }
        if !matches!(index_map, crate::map::IndexMap::Computed { .. }) {
            return false;
        }
        used_as_gather_operand = true;
    }
    used_as_gather_operand
}

pub(super) fn buffer_of<T, B: Deref<Target = [T]>>(
    buffers: &[Option<B>],
    node: NodeId,
) -> Result<&[T], TensorError> {
    buffers[node.0 as usize]
        .as_deref()
        .ok_or(TensorError::NotLowerable {
            node,
            reason: "operand buffer missing at evaluation time",
        })
}

pub(super) fn element_count(shape: &[u64]) -> usize {
    shape.iter().product::<u64>() as usize
}

pub(super) fn split_innermost(extents: &[u64]) -> (&[u64], usize) {
    match extents.split_last() {
        Some((last, rest)) => (rest, *last as usize),
        None => (extents, 1),
    }
}

/// Total element count of an odometer over `shape` — `0..odometer_len(shape)`
/// is the flat-index range [`unflatten_into`] walks.
pub(super) fn odometer_len(shape: &[u64]) -> u64 {
    shape.iter().product()
}

/// Writes flat index `flat`'s mixed-radix coordinate into the caller's
/// reused `coordinate` buffer instead of allocating a fresh `Vec` per call —
/// this runs once per (leading, reduction) coordinate pair in [`run_reduce`],
/// up to ~1e6 times for a 1024^3 GEMM. The allocating former version
/// (`odometer`/`unflatten`, returning `impl Iterator<Item = Vec<u64>>`)
/// accounted for roughly half of the 2.1M allocations measured after ROW 2's
/// `running`/`gather_cursors` hoist — the other half was
/// [`merge_coordinates_into`]'s former per-call `Vec` (`proxima-tensor/docs/discipline.md` ROW 2b).
pub(super) fn unflatten_into(mut flat: u64, shape: &[u64], coordinate: &mut [u64]) {
    for (dim, extent) in shape.iter().enumerate().rev() {
        coordinate[dim] = flat % extent;
        flat /= extent;
    }
}

/// Writes the union of a leading coordinate and a reduction coordinate into
/// the caller's reused `out` buffer, zeroing any dim neither side supplies
/// (there are none in practice — every dim is either leading or reduction —
/// but the zero-fill keeps the contract obvious without relying on that).
pub(super) fn merge_coordinates_into(
    leading_dims: &[u16],
    leading_coordinate: &[u64],
    reduction_dims: &[u16],
    reduction_coordinate: &[u64],
    out: &mut [u64],
) {
    out.fill(0);
    for (dim, value) in leading_dims.iter().zip(leading_coordinate) {
        out[*dim as usize] = *value;
    }
    for (dim, value) in reduction_dims.iter().zip(reduction_coordinate) {
        out[*dim as usize] = *value;
    }
}

pub(super) fn initial_value(init: ReduceInit) -> Option<f32> {
    match init {
        ReduceInit::Zero => Some(0.0),
        ReduceInit::One => Some(1.0),
        ReduceInit::NegativeInfinity => Some(f32::NEG_INFINITY),
        ReduceInit::PositiveInfinity => Some(f32::INFINITY),
        ReduceInit::FirstElement => None,
    }
}

// `evaluate`/`evaluate_parallel` both drive `run_node_into` directly (see
// `evaluate_pooled`'s doc for why), so this allocate-and-run wrapper only
// remains for tests that want a whole node's output as a `Vec` to compare
// against hand-run chunks.
