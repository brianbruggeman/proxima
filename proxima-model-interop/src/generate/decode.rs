use core::ops::ControlFlow;

use alloc::collections::VecDeque;

use super::*;

/// Measurement-only edge switch: mirrors `ServingConfig`'s three fusion
/// bools into the process env vars `proxima_tensor::bind::bind_with_fusion`
/// already reads (`PROXIMA_DISABLE_CACHED_ATTENTION_FUSION` pre-existing;
/// `PROXIMA_DISABLE_GATED_DELTA_NET_FUSION`/`PROXIMA_DISABLE_MOE_TOPK_FUSION`
/// new), so a caller can flip a compiled-in fusion off per invocation
/// through `ServingConfig` rather than only via process env directly.
#[cfg(feature = "std")]
fn apply_fusion_env_switches(serving_config: &ServingConfig) {
    set_fusion_disable_env_var(
        "PROXIMA_DISABLE_CACHED_ATTENTION_FUSION",
        !serving_config.cached_attention_fusion,
    );
    set_fusion_disable_env_var(
        "PROXIMA_DISABLE_GATED_DELTA_NET_FUSION",
        !serving_config.gated_delta_net_fusion,
    );
    set_fusion_disable_env_var(
        "PROXIMA_DISABLE_MOE_TOPK_FUSION",
        !serving_config.moe_topk_fusion,
    );
}

#[cfg(feature = "std")]
fn set_fusion_disable_env_var(name: &str, disable: bool) {
    // single-threaded CLI/bench call sites only; no concurrent env reader.
    unsafe {
        if disable {
            std::env::set_var(name, "1");
        } else {
            std::env::remove_var(name);
        }
    }
}

/// `PROXIMA_METAL_FUSE_ATTN_PARITY_STEPS` reader -- a comma list of decode
/// step indices [`run_decode_loop_placed_kv`]'s parity probe runs on, same
/// one-env-var-per-diagnostic convention as `PROXIMA_METAL_OP_PROFILE_STEP`.
#[cfg(all(
    feature = "metal",
    feature = "metal-fuse-attn-decode",
    feature = "metal-output-placement",
    target_os = "macos"
))]
fn attn_fuse_parity_target_steps() -> Vec<usize> {
    std::env::var("PROXIMA_METAL_FUSE_ATTN_PARITY_STEPS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .collect()
        })
        .unwrap_or_default()
}

/// `PROXIMA_ATTN_LAYER` reader -- which of `run_attn_fuse_parity_probe`'s own
/// `candidates` (the `CachedAttention` nodes in program order, one per
/// decoder layer) the failure report / read-source-vector dump targets.
/// Default `0` preserves round-9's layer-0-only behaviour exactly.
#[cfg(all(
    feature = "metal",
    feature = "metal-fuse-attn-decode",
    feature = "metal-output-placement",
    target_os = "macos"
))]
fn attn_fuse_parity_target_layer() -> usize {
    std::env::var("PROXIMA_ATTN_LAYER")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(0)
}

/// The live cached-row count for this step -- the ninth `CachedAttention`
/// operand (`dead_code_cached_attention.rs:1311-1319`), a runtime scalar
/// named input distinct from `symbols[1]` (the cache buffer's allocated
/// row CAPACITY, fixed per bucket). Reading `symbols[1]` here reports
/// capacity as if it were live length -- this reads the actual named
/// "cached_len" block the step supplied.
#[cfg(all(
    feature = "metal",
    feature = "metal-fuse-attn-decode",
    feature = "metal-output-placement",
    target_os = "macos"
))]
fn live_cached_len(named_blocks: &[(&str, QuantizedBlock<'_>)]) -> Option<f32> {
    named_blocks.iter().find_map(|(name, block)| {
        if *name != "cached_len" {
            return None;
        }
        match block {
            QuantizedBlock::Float32(values) => values.first().copied(),
            _ => None,
        }
    })
}

#[cfg(all(
    feature = "metal",
    feature = "metal-fuse-attn-decode",
    feature = "metal-output-placement",
    target_os = "macos"
))]
fn bits_at_slice(values: &[u32], index: usize) -> String {
    values.get(index).map_or("MISSING".to_string(), |bits| format!("0x{bits:08x}"))
}

/// First element where two equal-length f32 buffers diverge at the bit
/// level, carrying both bit patterns -- `==` on the floats themselves would
/// treat two differently-rounded NaNs as equal.
#[cfg(all(
    feature = "metal",
    feature = "metal-fuse-attn-decode",
    feature = "metal-output-placement",
    target_os = "macos"
))]
fn first_diff_f32(unfused: &[f32], fused: &[f32]) -> Option<(usize, u32, u32)> {
    unfused
        .iter()
        .zip(fused.iter())
        .enumerate()
        .find_map(|(index, (&left, &right))| {
            (left.to_bits() != right.to_bits()).then_some((index, left.to_bits(), right.to_bits()))
        })
}

/// One fused decoder layer's attended (166-role) node, plus -- under
/// Candidate B's shape only -- its softmax op's own node (154-role) and its
/// three named outputs (`cached_weight_sum`/`new_weight_sum`/`new_attended`
/// at 157/158/164). `None` for the pre-Candidate-B shape, where the
/// `CachedAttention` node IS the attended output and there is no separate
/// softmax op to report.
#[cfg(all(
    feature = "metal",
    feature = "metal-fuse-attn-decode",
    feature = "metal-output-placement",
    target_os = "macos"
))]
type AttnFuseProbeLayer = (NodeId, Option<(NodeId, NodeId, NodeId, NodeId)>);

/// Enumerates one attended (166-role) node per fused decoder layer, under
/// either recognizer shape `resolved` may contain: the pre-Candidate-B
/// shape, one [`proxima_tensor::BoundOpKind::CachedAttention`] op per layer
/// whose own node IS the attended output; or Candidate B's shape, one
/// [`proxima_tensor::BoundOpKind::CachedSoftmaxWeights`] op per layer (the
/// 154-role node, carrying named outputs `cached_weight_sum`/
/// `new_weight_sum`/`new_attended` at 157/158/164) plus a plain-looking
/// `Reduce` whose epilogue reads those outputs (the 166-role node, located
/// by scanning [`proxima_tensor::BoundOp::all_read_sources`] rather than
/// assumed at a fixed offset -- the per-layer template puts it at
/// `softmax_node + 12` and this function asserts (diagnostically) that the
/// template holds rather than trusting it blind). Returns `(attended_node,
/// None)` for the first shape, `(attended_node, Some((softmax_node,
/// cached_weight_sum, new_weight_sum, new_attended)))` for the second.
#[cfg(all(
    feature = "metal",
    feature = "metal-fuse-attn-decode",
    feature = "metal-output-placement",
    target_os = "macos"
))]
fn attn_fuse_probe_layers(
    step: usize,
    resolved: &[proxima_tensor::BoundOp],
) -> Vec<AttnFuseProbeLayer> {
    let cached_attention_nodes: Vec<NodeId> = resolved
        .iter()
        .filter(|bound| {
            matches!(
                bound.kind,
                proxima_tensor::BoundOpKind::CachedAttention { .. }
            )
        })
        .map(|bound| bound.node)
        .collect();
    if !cached_attention_nodes.is_empty() {
        return cached_attention_nodes
            .into_iter()
            .map(|node| (node, None))
            .collect();
    }

    resolved
        .iter()
        .filter_map(|bound| match bound.kind {
            proxima_tensor::BoundOpKind::CachedSoftmaxWeights {
                cached_weight_sum,
                new_weight_sum,
                new_attended,
                ..
            } => Some((bound.node, cached_weight_sum, new_weight_sum, new_attended)),
            _ => None,
        })
        .filter_map(|(softmax_node, cached_weight_sum, new_weight_sum, new_attended)| {
            let combine = resolved.iter().find(|bound| {
                bound.all_read_sources().any(|(operand, _, _)| {
                    *operand == new_attended || *operand == cached_weight_sum
                })
            })?;
            let expected = NodeId((softmax_node.0 as i32 + 12) as u32);
            if combine.node != expected {
                eprintln!(
                    "parity_layer_offset_mismatch step={step} softmax_node={} combine_node={} expected={}",
                    softmax_node.0, combine.node.0, expected.0
                );
            }
            Some((
                combine.node,
                Some((softmax_node, cached_weight_sum, new_weight_sum, new_attended)),
            ))
        })
        .collect()
}

/// Builds two throwaway plans for THIS step's `program`/`outputs` -- one
/// with [`proxima_tensor::bind_with_fusion`]'s `fuse_cached_attention` set,
/// one cleared -- and diffs every [`proxima_tensor::BoundOpKind::CachedAttention`]
/// node's output plus the logits root as raw bit patterns. Read-only: both
/// plans read the caller's resident `input_placements` but pass no
/// `output_placements`, so every root comes back through host [`Evaluated`]
/// instead of landing in a device [`PlacedBuffer`] -- neither call mutates
/// the KV cache the real step's own `evaluate_with_placements` reads next.
/// Never touches [`BackendRuntime::placed_plans`] -- both plans are built
/// and dropped locally, not inserted into the cache.
#[cfg(all(
    feature = "metal",
    feature = "metal-fuse-attn-decode",
    feature = "metal-output-placement",
    target_os = "macos"
))]
#[allow(clippy::too_many_arguments)]
fn run_attn_fuse_parity_probe(
    step: usize,
    program: &[Op],
    symbols: &[u64],
    named_blocks: &[(&str, QuantizedBlock<'_>)],
    roots: &[NodeId],
    logits_root: NodeId,
    resident_names: &alloc::collections::BTreeSet<&str>,
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
    runtime: &BackendRuntime,
) -> Result<(), InteropError> {
    let shapes = proxima_tensor::infer(program, symbols)?;
    let resolved =
        proxima_tensor::bind_with_fusion(program, &shapes, roots, true, runtime.numeric_policy)?;
    let layers = attn_fuse_probe_layers(step, &resolved);
    let candidates: Vec<NodeId> = layers.iter().map(|(attended, _)| *attended).collect();
    eprintln!("parity_candidates step={step} n={}", candidates.len());
    for (layer, (attended, softmax)) in layers.iter().enumerate() {
        match softmax {
            None => eprintln!(
                "parity_layer_ops step={step} layer={layer} shape=cached_attention attended={}",
                attended.0
            ),
            Some((softmax_node, cached_weight_sum, new_weight_sum, new_attended)) => eprintln!(
                "parity_layer_ops step={step} layer={layer} shape=candidate_b attended={} softmax={} cached_weight_sum={} new_weight_sum={} new_attended={}",
                attended.0, softmax_node.0, cached_weight_sum.0, new_weight_sum.0, new_attended.0
            ),
        }
    }

    let mut parity_outputs: Vec<NodeId> = roots.to_vec();
    for (attended, softmax) in &layers {
        parity_outputs.push(*attended);
        if let Some((softmax_node, cached_weight_sum, new_weight_sum, new_attended)) = softmax {
            parity_outputs.push(*softmax_node);
            parity_outputs.push(*cached_weight_sum);
            parity_outputs.push(*new_weight_sum);
            parity_outputs.push(*new_attended);
        }
    }
    let placed_input_nodes: Vec<NodeId> =
        input_placements.iter().map(|(node, _, _)| *node).collect();

    let build_plan = |fuse_cached_attention: bool| -> Result<omega::metal::Plan, InteropError> {
        let mut plan = plan_named_with_placed_inputs(
            program,
            symbols,
            named_blocks,
            &parity_outputs,
            runtime.numeric_policy,
            &placed_input_nodes,
            fuse_cached_attention,
        )?;
        plan.mark_resident(resident_names);
        if runtime.plan_time_constants {
            plan.mark_plan_time_constants_resident();
        }
        // ROOTCAUSE TOGGLE (temporary, reverted before this slice closes):
        // holds MTLCompileOptions.mathMode at Safe for BOTH arms so
        // compiler FMA contraction/reassociation is held constant while
        // the per-position/per-op walk below isolates the residual
        // (route steps b-e, coordinator directive). `PROXIMA_ATTN_MATH_MODE`
        // (round-4 addition, default `safe`, unset preserves the prior
        // behaviour exactly) lets a caller re-run the SAME probe under
        // Relaxed to get a real production reference instead of guessing at
        // one from a stale pre-`all_read_sources()` log.
        let probe_math_mode = match std::env::var("PROXIMA_ATTN_MATH_MODE").ok().as_deref() {
            Some("relaxed") => omega::MathMode::Relaxed,
            _ => omega::MathMode::Safe,
        };
        plan.set_math_mode(probe_math_mode)?;
        plan.set_dispatch_type(runtime.dispatch_type);
        Ok(plan)
    };
    let fused_plan = build_plan(true)?;
    let unfused_plan = build_plan(false)?;

    let fused_evaluated =
        execute_plan_named_with_placements(&fused_plan, named_blocks, input_placements, &[])?;
    let unfused_evaluated =
        execute_plan_named_with_placements(&unfused_plan, named_blocks, input_placements, &[])?;

    // V5 degenerate control: the SAME plan, evaluated twice against the
    // SAME inputs. If either arm disagrees with itself, the
    // fused-vs-unfused diff above is not evidence of a fusion defect --
    // it is evidence the Metal path is not bit-stable run-to-run at all.
    let unfused_evaluated_again =
        execute_plan_named_with_placements(&unfused_plan, named_blocks, input_placements, &[])?;
    let fused_evaluated_again =
        execute_plan_named_with_placements(&fused_plan, named_blocks, input_placements, &[])?;
    for (arm, first, second) in [
        ("unfused", &unfused_evaluated, &unfused_evaluated_again),
        ("fused", &fused_evaluated, &fused_evaluated_again),
    ] {
        let mut layers_with_diff = 0_usize;
        for node in &candidates {
            let (first_values, _) = first
                .get(*node)
                .ok_or(InteropError::MissingEvaluatedNode { node: *node })?;
            let (second_values, _) = second
                .get(*node)
                .ok_or(InteropError::MissingEvaluatedNode { node: *node })?;
            if first_diff_f32(first_values, second_values).is_some() {
                layers_with_diff += 1;
            }
        }
        let (first_logits, _) = first
            .get(logits_root)
            .ok_or(InteropError::MissingEvaluatedNode { node: logits_root })?;
        let (second_logits, _) = second
            .get(logits_root)
            .ok_or(InteropError::MissingEvaluatedNode { node: logits_root })?;
        let logits_diff = first_diff_f32(first_logits, second_logits);
        eprintln!(
            "parity_control step={step} arm={arm} layers_with_diff={layers_with_diff}/{} logits_first_diff={}",
            candidates.len(),
            match logits_diff {
                None => "none".to_string(),
                Some((element, first_bits, second_bits)) =>
                    format!("({element}, 0x{first_bits:08x}, 0x{second_bits:08x})"),
            }
        );
    }

    let leaf_extent = symbols.get(1).copied().unwrap_or_default();
    let cached_len = live_cached_len(named_blocks);
    let cached_len_display = cached_len.map_or("MISSING".to_string(), |value| value.to_string());
    for (layer, (node, softmax)) in layers.iter().enumerate() {
        let (fused_values, _) = fused_evaluated
            .get(*node)
            .ok_or(InteropError::MissingEvaluatedNode { node: *node })?;
        let (unfused_values, _) = unfused_evaluated
            .get(*node)
            .ok_or(InteropError::MissingEvaluatedNode { node: *node })?;
        match first_diff_f32(unfused_values, fused_values) {
            None => eprintln!(
                "parity step={step} layer={layer} node={} cached_len={cached_len_display} leaf_extent={leaf_extent} first_diff=none",
                node.0
            ),
            Some((element, unfused_bits, fused_bits)) => eprintln!(
                "parity step={step} layer={layer} node={} cached_len={cached_len_display} leaf_extent={leaf_extent} first_diff=({element}, 0x{unfused_bits:08x}, 0x{fused_bits:08x})",
                node.0
            ),
        }
        if let Some((softmax_node, cached_weight_sum, new_weight_sum, new_attended)) = softmax {
            for (role, role_node) in [
                ("154", softmax_node),
                ("157", cached_weight_sum),
                ("158", new_weight_sum),
                ("164", new_attended),
            ] {
                let fused_role = fused_evaluated.get(*role_node);
                let unfused_role = unfused_evaluated.get(*role_node);
                match (fused_role, unfused_role) {
                    (Some((fused_values, _)), Some((unfused_values, _))) => {
                        match first_diff_f32(unfused_values, fused_values) {
                            None => eprintln!(
                                "parity_softmax step={step} layer={layer} role={role} node={} first_diff=none",
                                role_node.0
                            ),
                            Some((element, unfused_bits, fused_bits)) => eprintln!(
                                "parity_softmax step={step} layer={layer} role={role} node={} first_diff=({element}, 0x{unfused_bits:08x}, 0x{fused_bits:08x})",
                                role_node.0
                            ),
                        }
                    }
                    _ => eprintln!(
                        "parity_softmax step={step} layer={layer} role={role} node={} MISSING",
                        role_node.0
                    ),
                }
            }
        }
    }

    let (fused_logits, _) = fused_evaluated
        .get(logits_root)
        .ok_or(InteropError::MissingEvaluatedNode { node: logits_root })?;
    let (unfused_logits, _) = unfused_evaluated
        .get(logits_root)
        .ok_or(InteropError::MissingEvaluatedNode { node: logits_root })?;
    match first_diff_f32(unfused_logits, fused_logits) {
        None => eprintln!("parity_logits step={step} first_diff=none"),
        Some((element, unfused_bits, fused_bits)) => eprintln!(
            "parity_logits step={step} first_diff=({element}, 0x{unfused_bits:08x}, 0x{fused_bits:08x})"
        ),
    }

    if step == 1
        && let Some(&(layer0_node, _)) = layers.get(attn_fuse_parity_target_layer())
    {
        let metal_unfused_full: Vec<u32> = unfused_evaluated
            .get(layer0_node)
            .map(|(values, _)| values.iter().map(|value| value.to_bits()).collect())
            .unwrap_or_default();
        let metal_fused_full: Vec<u32> = fused_evaluated
            .get(layer0_node)
            .map(|(values, _)| values.iter().map(|value| value.to_bits()).collect())
            .unwrap_or_default();
        let metal_unfused_logits_full: Vec<u32> = unfused_evaluated
            .get(logits_root)
            .map(|(values, _)| values.iter().map(|value| value.to_bits()).collect())
            .unwrap_or_default();
        let metal_fused_logits_full: Vec<u32> = fused_evaluated
            .get(logits_root)
            .map(|(values, _)| values.iter().map(|value| value.to_bits()).collect())
            .unwrap_or_default();
        write_attn_layer0_failure_report(
            layer0_node,
            program,
            symbols,
            named_blocks,
            &resolved,
            runtime.numeric_policy,
            &metal_unfused_full,
            &metal_fused_full,
            &metal_unfused_logits_full,
            &metal_fused_logits_full,
            resident_names,
            input_placements,
            runtime,
        )?;
    }

    Ok(())
}

/// Owner's slice-4 failure report for `step=1 layer=0 node=166`: the
/// element-0 chain from `q_even_grouped`/`q_odd_grouped` through the
/// `attended` node, plus the UNFUSED `BoundOp` sequence between them and a
/// CPU cross-check of the same element under both fusion states. Written to
/// `PROXIMA_ATTN_PAYLOAD_PATH` (default `attn_parity_payload.txt`, relative
/// to the process's own cwd -- run from the scratch `attn_parity/`
/// directory so the default lands there).
#[cfg(all(
    feature = "metal",
    feature = "metal-fuse-attn-decode",
    feature = "metal-output-placement",
    target_os = "macos"
))]
#[allow(clippy::too_many_arguments)]
fn write_attn_layer0_failure_report(
    attended_node: NodeId,
    program: &[Op],
    symbols: &[u64],
    named_blocks: &[(&str, QuantizedBlock<'_>)],
    fused_resolved: &[proxima_tensor::BoundOp],
    numeric_policy: proxima_tensor::NumericPolicy,
    metal_unfused_full: &[u32],
    metal_fused_full: &[u32],
    metal_unfused_logits_full: &[u32],
    metal_fused_logits_full: &[u32],
    resident_names: &alloc::collections::BTreeSet<&str>,
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
    runtime: &BackendRuntime,
) -> Result<(), InteropError> {
    let mut report = String::new();

    let fused_bound = fused_resolved
        .iter()
        .find(|bound| bound.node == attended_node)
        .ok_or(InteropError::MissingEvaluatedNode {
            node: attended_node,
        })?;
    // Candidate B's recognizer shape (`dead_code_cached_attention.rs`'s own
    // `BoundOpKind::CachedSoftmaxWeights` doc) makes `attended_node`'s own
    // `BoundOpKind` a plain `Reduce` epilogue-combine, never `CachedAttention`
    // directly -- this report's element-0 chain walk below reads fields
    // (kv_heads/query_groups/rotary_dim/cached_lower_inclusive/
    // new_upper_inclusive/scale) that exist only on the pre-Candidate-B
    // shape, so there is no way to build the SAME report for this node
    // under Candidate B. This is a diagnostic dump only
    // (`PROXIMA_ATTN_PAYLOAD_PATH`) with no bearing on decode correctness --
    // degrading to "report skipped" is correct; raising
    // `MissingEvaluatedNode` here previously aborted the whole decode call
    // over a report-writer gap, not an actually missing node (`fused_bound`
    // above already proves the node WAS found and evaluated).
    let proxima_tensor::BoundOpKind::CachedAttention {
        kv_heads,
        query_groups,
        head_dim,
        rotary_dim,
        cached_lower_inclusive,
        new_upper_inclusive,
        scale,
        cached_key_rows,
        new_key_rows,
        ..
    } = fused_bound.kind
    else {
        #[cfg(feature = "instrument")]
        debug!(
            node = attended_node.0,
            "attn_layer0_failure_report: skipping, attended node's BoundOpKind is not \
             CachedAttention (Candidate B's Reduce-epilogue shape has no equivalent report)"
        );
        return Ok(());
    };
    let operand_nodes: Vec<NodeId> = fused_bound
        .operands()
        .iter()
        .take(8)
        .map(|(node, _, _)| *node)
        .collect();
    let [q_even, q_odd, k_even_cache, k_odd_cache, new_k_even, new_k_odd, v_cache, v_new] =
        operand_nodes[..8]
            .try_into()
            .map_err(|_| InteropError::MissingEvaluatedNode {
                node: attended_node,
            })?;
    report.push_str(&format!(
        "layer0 node={} kv_heads={kv_heads} query_groups={query_groups} head_dim={head_dim} rotary_dim={rotary_dim} scale={scale} cached_lower_inclusive={cached_lower_inclusive} new_upper_inclusive={new_upper_inclusive}\n",
        attended_node.0
    ));
    report.push_str(&format!(
        "operand_nodes q_even={} q_odd={} k_even_cache={} k_odd_cache={} new_k_even={} new_k_odd={} v_cache={} v_new={}\n",
        q_even.0, q_odd.0, k_even_cache.0, k_odd_cache.0, new_k_even.0, new_k_odd.0, v_cache.0, v_new.0
    ));

    // Backward walk over the UNFUSED bind, from `attended_node` down to
    // the 8 leaf operands above -- collects every internal BoundOp on
    // layer 0's score/softmax/weighted-sum chain without assuming a node
    // numbering.
    let shapes = proxima_tensor::infer(program, symbols)?;
    let unfused_resolved =
        proxima_tensor::bind_with_fusion(program, &shapes, &[attended_node], false, numeric_policy)?;
    let by_node: alloc::collections::BTreeMap<NodeId, &proxima_tensor::BoundOp> =
        unfused_resolved.iter().map(|bound| (bound.node, bound)).collect();
    let leaves: alloc::collections::BTreeSet<NodeId> = operand_nodes.iter().copied().collect();
    let mut visited: alloc::collections::BTreeSet<NodeId> = alloc::collections::BTreeSet::new();
    let mut stack = alloc::vec![attended_node];
    let mut chain: Vec<NodeId> = Vec::new();
    while let Some(node) = stack.pop() {
        if leaves.contains(&node) || !visited.insert(node) {
            continue;
        }
        chain.push(node);
        if let Some(bound) = by_node.get(&node) {
            // `all_read_sources()`, not `operands()`: a `Reduce`'s
            // `epilogue_operands` (`types_layout_boundop.rs:529-545`) are a
            // real read this walk must follow -- `operands()` alone silently
            // drops an epilogue-only reader, which is exactly how q_odd/
            // k_odd's own even+odd combine stayed invisible to this walk
            // before this fix (the earlier `unfused_op` printout showed
            // node=134 with a single `130` (q_even) operand and no odd-dot
            // term anywhere in the chain).
            for (operand, _, _) in bound.all_read_sources() {
                stack.push(*operand);
            }
        }
    }
    chain.sort_unstable_by_key(|node| node.0);
    report.push_str(&format!(
        "unfused_chain_len={} (q_even_grouped/q_odd_grouped through node={} exclusive of the 8 leaf operands)\n",
        chain.len(),
        attended_node.0
    ));
    let packed_operands = omega::PackedOperands::new();
    for node in &chain {
        if let Some(bound) = by_node.get(node) {
            let operand_summary: Vec<String> = bound
                .operands()
                .iter()
                .map(|(operand, layout, _)| {
                    format!("{}@base={},strides={:?}", operand.0, layout.base, layout.strides)
                })
                .collect();
            // `epilogue_operands` -- a `Reduce`'s epilogue reads these but
            // `operands()` does not enumerate them (see the chain-walk fix
            // above). Printed separately so a "keep::reduce fold" node's
            // FULL read set (even-dot AND odd-dot sources both) is visible,
            // not just its primary `operands`.
            let epilogue_summary: Vec<String> = match &bound.kind {
                proxima_tensor::BoundOpKind::Reduce {
                    epilogue_operands, ..
                }
                | proxima_tensor::BoundOpKind::RoundBatchedReduce {
                    epilogue_operands, ..
                } => epilogue_operands
                    .iter()
                    .map(|(operand, layout, _)| {
                        format!("{}@base={},strides={:?}", operand.0, layout.base, layout.strides)
                    })
                    .collect(),
                _ => Vec::new(),
            };
            let dispatch = match omega::emit(bound, &packed_operands, numeric_policy) {
                Ok(kernel) => {
                    // round-4 work item B: nodes 139 (odd-dot fold) and 142
                    // (even-dot fold + epilogue add/select) are the two
                    // UNFUSED reductions feeding the t=2 residual -- their
                    // own emitted MSL, fenced the same way the fused
                    // kernel's is, so a standalone replay can compile the
                    // real per-lane reduction order instead of guessing it.
                    if node.0 == 139 || node.0 == 142 {
                        report.push_str(&format!("unfused_kernel_source_begin node={}\n", node.0));
                        report.push_str(&kernel.source);
                        report.push_str(&format!("\nunfused_kernel_source_end node={}\n", node.0));
                    }
                    format!(
                        "entry={} grid_threads={} threadgroup_width={:?} grid_depth={}",
                        kernel.entry, kernel.grid.threads, kernel.grid.threadgroup_width, kernel.grid.depth
                    )
                }
                Err(error) => format!("emit_failed={error:?}"),
            };
            report.push_str(&format!(
                "unfused_op node={} kind={} operands=[{}] epilogue_operands=[{}] dispatch=[{dispatch}]\n",
                node.0,
                bound.kind.name(),
                operand_summary.join(", "),
                epilogue_summary.join(", "),
            ));
        }
    }

    // Route step (route.d/item 4): request every chain node PLUS
    // `attended_node` as outputs of the PRODUCTION unfused Metal plan (not
    // the CPU evaluator) -- confirms requesting the intermediates does not
    // itself change `attended_node`'s own bits (a materialization-order
    // side effect would show up here), and gives the REAL Metal-computed
    // per-op values for the per-position walk below instead of a CPU
    // stand-in. Same math_mode as `run_attn_fuse_parity_probe`'s own
    // `build_plan` this round (Safe, held constant across arms).
    let placed_input_nodes: Vec<NodeId> =
        input_placements.iter().map(|(node, _, _)| *node).collect();
    let mut intermediate_outputs = chain.clone();
    intermediate_outputs.push(attended_node);
    // rootcause round 3 fixture step: the 8 leaf operands are excluded from
    // `chain` by construction (see the `unfused_chain_len` note above) but a
    // standalone fused-kernel replay needs their FULL bits, not the
    // first-8-dims preview `q_even_grouped`/`new_k_even` print elsewhere --
    // request them as outputs of the same production plan so the dump below
    // reads real Metal-computed bytes, not a CPU stand-in.
    intermediate_outputs.extend_from_slice(&operand_nodes);
    let mut unfused_intermediates_plan = plan_named_with_placed_inputs(
        program,
        symbols,
        named_blocks,
        &intermediate_outputs,
        numeric_policy,
        &placed_input_nodes,
        false,
    )?;
    unfused_intermediates_plan.mark_resident(resident_names);
    if runtime.plan_time_constants {
        unfused_intermediates_plan.mark_plan_time_constants_resident();
    }
    // round-4: same `PROXIMA_ATTN_MATH_MODE` override as `build_plan` above,
    // so this fixture's leaf/chain dump is captured under the SAME mode the
    // parity-probe arms just ran under, not silently pinned to Safe while
    // the caller asked for Relaxed.
    let intermediates_math_mode = match std::env::var("PROXIMA_ATTN_MATH_MODE").ok().as_deref() {
        Some("relaxed") => omega::MathMode::Relaxed,
        _ => omega::MathMode::Safe,
    };
    unfused_intermediates_plan.set_math_mode(intermediates_math_mode)?;
    unfused_intermediates_plan.set_dispatch_type(runtime.dispatch_type);
    let intermediates_evaluated = execute_plan_named_with_placements(
        &unfused_intermediates_plan,
        named_blocks,
        input_placements,
        &[],
    )?;
    let intermediates_attended_bits: Vec<u32> = intermediates_evaluated
        .get(attended_node)
        .map(|(values, _)| values.iter().map(|value| value.to_bits()).collect())
        .unwrap_or_default();
    let attended_bits_unchanged = intermediates_attended_bits.first() == metal_unfused_full.first()
        && intermediates_attended_bits.get(1) == metal_unfused_full.get(1);
    report.push_str(&format!(
        "unfused_intermediates_probe requested_outputs={} attended_element0={} attended_element1={} unchanged_vs_production={attended_bits_unchanged}\n",
        intermediate_outputs.len(),
        bits_at_slice(&intermediates_attended_bits, 0),
        bits_at_slice(&intermediates_attended_bits, 1),
    ));
    for node in &chain {
        if let Some((values, shape)) = intermediates_evaluated.get(*node) {
            let all_bits: Vec<u32> = values.iter().map(|value| value.to_bits()).collect();
            // query-group-0 column: every score/sum node here ends in an
            // 8-wide query_groups axis, unit stride -- index i is qg0 exactly
            // when i % 8 == 0. Covers all 6 live cached rows (t=0..5) in one
            // slice for the [1,32,1,8]-shaped score nodes (142/154), and the
            // single new-range row for the [*,1,1,8]-shaped ones (149/156).
            let qg0_column: Vec<u32> = if !all_bits.is_empty() && all_bits.len().is_multiple_of(8) {
                all_bits.iter().step_by(8).copied().collect()
            } else {
                Vec::new()
            };
            report.push_str(&format!(
                "unfused_metal_intermediate node={} shape={shape:?} len={} qg0_column={:?} first16_bits={:?}\n",
                node.0,
                all_bits.len(),
                qg0_column,
                &all_bits[..all_bits.len().min(16)]
            ));
        }
    }

    // rootcause round 3 fixture step: full raw bits for the 8 leaf operands,
    // identical Metal-produced buffers a standalone fused-kernel replay would
    // bind at the production indices -- printed complete (not truncated to
    // 16) since the dot-product walk needs every dim, not a preview.
    let leaf_labels = [
        "q_even", "q_odd", "k_even_cache", "k_odd_cache", "new_k_even", "new_k_odd", "v_cache",
        "v_new",
    ];
    for (label, node) in leaf_labels.iter().zip(operand_nodes.iter()) {
        if let Some((values, shape)) = intermediates_evaluated.get(*node) {
            let all_bits: Vec<u32> = values.iter().map(|value| value.to_bits()).collect();
            report.push_str(&format!(
                "leaf_full label={label} node={} shape={shape:?} len={} bits={:?}\n",
                node.0,
                all_bits.len(),
                all_bits,
            ));
        } else {
            report.push_str(&format!("leaf_full label={label} node={} MISSING\n", node.0));
        }
    }

    // Fused-arm entry: `context_chunks_for` is `pub` (widened this slice,
    // `omega/src/msl/signature_tokens_prelude.rs`) and `omega::emit` is the
    // SAME function `pipeline_for` calls before compiling -- both called
    // here with the production `fused_bound`/`cached_key_rows`/
    // `new_key_rows`/`query_groups`/`head_dim`/`numeric_policy`, not
    // inferred from the `cached_len <= ATTENTION_CONTEXT_KEYS_PER_CHUNK`
    // relationship.
    let live_cached_len = live_cached_len(named_blocks);
    let cache_capacity = symbols.get(1).copied().unwrap_or_default();
    let context_chunks = omega::context_chunks_for(
        cached_key_rows + new_key_rows,
        query_groups,
        head_dim,
        numeric_policy,
    );
    let fused_dispatch = match omega::emit(fused_bound, &packed_operands, numeric_policy) {
        Ok(kernel) => {
            // rootcause round 3: the production fused kernel's own MSL text,
            // fenced so a shell step can carve it into its own .metal file
            // for the standalone replay/diagnostic-store steps without this
            // harness hardcoding a scratch path.
            report.push_str("fused_kernel_source_begin\n");
            report.push_str(&kernel.source);
            report.push_str("\nfused_kernel_source_end\n");
            format!(
                "entry={} grid_threads={} threadgroup_width={:?} grid_depth={}",
                kernel.entry, kernel.grid.threads, kernel.grid.threadgroup_width, kernel.grid.depth
            )
        }
        Err(error) => format!("emit_failed={error:?}"),
    };
    report.push_str(&format!(
        "fused_arm node={} cached_lower_inclusive={cached_lower_inclusive} new_upper_inclusive={new_upper_inclusive} cached_len={} cache_capacity={cache_capacity} cached_key_rows={cached_key_rows} new_key_rows={new_key_rows} context_chunks={context_chunks} dispatch=[{fused_dispatch}]\n",
        attended_node.0,
        live_cached_len.map_or("MISSING".to_string(), |value| value.to_string()),
    ));

    // Route step (falsification toggle, not the div/reciprocal one -- that
    // one moved element1 FURTHER from the unfused value and was reverted,
    // see the report's own note): `NumericPolicy::bit_exact()` does not
    // grant `NumericRewrite::ContextChunkMerge`
    // (`proxima-tensor/src/numeric.rs:171-172,207`), so `context_chunks_for`
    // (`signature_tokens_prelude.rs:1202-1203`) unconditionally returns 1
    // for this policy -- ONE simdgroup walks every live key sequentially,
    // no cross-chunk merge at all. This BoundOp's `chunks<=1`/`context_
    // chunks>1` arms never read `block_width_for` (that only gates the
    // `single_range_dynamic` path this bind is not), so this is the ONLY
    // structural difference `bit_exact()` introduces here versus
    // `runtime.numeric_policy` (which produced `context_chunks=3` above).
    // An explicit, existing, non-env-var policy value -- not a source edit.
    let context_chunks_one_plan = plan_named_with_placed_inputs(
        program,
        symbols,
        named_blocks,
        &[attended_node],
        proxima_tensor::NumericPolicy::bit_exact(),
        &placed_input_nodes,
        true,
    );
    match context_chunks_one_plan {
        Ok(mut plan) => {
            plan.mark_resident(resident_names);
            if runtime.plan_time_constants {
                plan.mark_plan_time_constants_resident();
            }
            plan.set_math_mode(omega::MathMode::Safe)?;
            plan.set_dispatch_type(runtime.dispatch_type);
            let evaluated =
                execute_plan_named_with_placements(&plan, named_blocks, input_placements, &[]);
            match evaluated {
                Ok(evaluated) => {
                    let bits: Vec<u32> = evaluated
                        .get(attended_node)
                        .map(|(values, _)| values.iter().map(|value| value.to_bits()).collect())
                        .unwrap_or_default();
                    report.push_str(&format!(
                        "context_chunks_one_probe element0={} element1={} (compare against production fused element0={} element1={} and unfused element0={} element1={})\n",
                        bits_at_slice(&bits, 0),
                        bits_at_slice(&bits, 1),
                        bits_at_slice(metal_fused_full, 0),
                        bits_at_slice(metal_fused_full, 1),
                        bits_at_slice(metal_unfused_full, 0),
                        bits_at_slice(metal_unfused_full, 1),
                    ));
                }
                Err(error) => report.push_str(&format!("context_chunks_one_probe execute_failed={error:?}\n")),
            }
        }
        Err(error) => report.push_str(&format!("context_chunks_one_probe plan_failed={error:?}\n")),
    }

    // Element-0 payload: the SAME `named_blocks` this step's real Metal
    // evaluation reads, via the unfused plan's own outputs (every
    // intermediate the fused kernel would otherwise absorb).
    let mut payload_outputs = chain.clone();
    payload_outputs.extend(operand_nodes.iter().copied());
    payload_outputs.push(attended_node);
    payload_outputs.sort_unstable_by_key(|node| node.0);
    payload_outputs.dedup();
    let unfused_plan_evaluated = proxima_tensor::cpu::evaluate_quantized_named_with_scratch_and_experts(
        program,
        symbols,
        named_blocks,
        &payload_outputs,
        &mut Vec::new(),
        &mut None,
        None,
    )?;

    let dump_slice = |report: &mut String, label: &str, node: NodeId, max_len: usize| {
        if let Some((values, shape)) = unfused_plan_evaluated.get(node) {
            let shown = &values[..values.len().min(max_len)];
            report.push_str(&format!(
                "{label} node={} shape={shape:?} len={} dims_0_7={:?}\n",
                node.0,
                values.len(),
                &shown[..shown.len().min(8)]
            ));
            if values.len() >= 128 {
                report.push_str(&format!("{label} full_128={:?}\n", &values[..128]));
            }
        } else {
            report.push_str(&format!("{label} node={} MISSING from unfused evaluation\n", node.0));
        }
    };
    dump_slice(&mut report, "q_even_grouped", q_even, 128);
    dump_slice(&mut report, "q_odd_grouped", q_odd, 128);

    let kv_stride = usize::try_from(kv_heads).unwrap_or(1) * usize::try_from(rotary_dim / 2).unwrap_or(1);
    if let Some((k_even_values, _)) = unfused_plan_evaluated.get(k_even_cache) {
        let rows = symbols.get(1).copied().unwrap_or_default() as usize;
        let row_width = usize::try_from(rotary_dim / 2).unwrap_or(1);
        for row in [0_usize, rows.saturating_sub(1)] {
            let start = row * kv_stride;
            let end = (start + row_width).min(k_even_values.len());
            report.push_str(&format!(
                "k_even_cache row={row} kv_head=0 dims_0_7={:?}\nk_even_cache row={row} kv_head=0 full={:?}\n",
                &k_even_values[start..(start + 8).min(end)],
                &k_even_values[start..end]
            ));
        }
    }
    if let Some((k_odd_values, _)) = unfused_plan_evaluated.get(k_odd_cache) {
        let rows = symbols.get(1).copied().unwrap_or_default() as usize;
        let row_width = usize::try_from(rotary_dim / 2).unwrap_or(1);
        for row in [0_usize, rows.saturating_sub(1)] {
            let start = row * kv_stride;
            let end = (start + row_width).min(k_odd_values.len());
            report.push_str(&format!(
                "k_odd_cache row={row} kv_head=0 dims_0_7={:?}\nk_odd_cache row={row} kv_head=0 full={:?}\n",
                &k_odd_values[start..(start + 8).min(end)],
                &k_odd_values[start..end]
            ));
        }
    }
    dump_slice(&mut report, "new_k_even", new_k_even, 128);
    dump_slice(&mut report, "new_k_odd", new_k_odd, 128);

    let v_stride = usize::try_from(kv_heads).unwrap_or(1) * usize::try_from(head_dim).unwrap_or(1);
    if let Some((v_cache_values, _)) = unfused_plan_evaluated.get(v_cache) {
        let rows = symbols.get(1).copied().unwrap_or_default() as usize;
        let column_d0: Vec<f32> = (0..rows)
            .filter_map(|row| v_cache_values.get(row * v_stride).copied())
            .collect();
        report.push_str(&format!("v_cache column_d0 kv_head=0 all_rows={column_d0:?}\n"));
    }
    if let Some((v_new_values, _)) = unfused_plan_evaluated.get(v_new) {
        report.push_str(&format!(
            "v_new row0 kv_head=0 d0={:?}\n",
            v_new_values.first()
        ));
    }

    for (label, node) in &chain
        .iter()
        .map(|node| (format!("unfused_intermediate_{}", node.0), *node))
        .collect::<Vec<_>>()
    {
        dump_slice(&mut report, label, *node, 32);
    }
    dump_slice(&mut report, "attended", attended_node, 8);

    // CPU cross-evidence: same `program`/`named_blocks`, CPU evaluator,
    // both fusion states selected via `bind_with_fusion`'s own explicit
    // bool (no process env var involved).
    let cpu_fused_bits = {
        let evaluated =
            proxima_tensor::cpu::evaluate_quantized_named_with_scratch_and_experts_with_fusion(
                program,
                symbols,
                named_blocks,
                &[attended_node],
                &mut Vec::new(),
                &mut None,
                None,
                true,
            )?;
        evaluated
            .get(attended_node)
            .and_then(|(values, _)| values.first().copied())
            .map(f32::to_bits)
    };
    let cpu_unfused_bits = {
        let evaluated =
            proxima_tensor::cpu::evaluate_quantized_named_with_scratch_and_experts_with_fusion(
                program,
                symbols,
                named_blocks,
                &[attended_node],
                &mut Vec::new(),
                &mut None,
                None,
                false,
            )?;
        evaluated
            .get(attended_node)
            .and_then(|(values, _)| values.first().copied())
            .map(f32::to_bits)
    };

    report.push_str(&format!(
        "cross_evidence element0 cpu_unfused={} cpu_fused={} metal_unfused={} metal_fused={}\n",
        cpu_unfused_bits.map_or("MISSING".to_string(), |bits| format!("0x{bits:08x}")),
        cpu_fused_bits.map_or("MISSING".to_string(), |bits| format!("0x{bits:08x}")),
        bits_at_slice(metal_unfused_full, 0),
        bits_at_slice(metal_fused_full, 0),
    ));
    report.push_str(&format!(
        "cross_evidence element1 metal_unfused={} metal_fused={}\n",
        bits_at_slice(metal_unfused_full, 1),
        bits_at_slice(metal_fused_full, 1),
    ));
    // Round-2 requirement 1: the full layer-0 `attended` vector's u32 bit
    // patterns, both arms -- printed here (production, THIS invocation)
    // so the diagnostic-variant comparison later in this report can be
    // checked against these exact lines rather than re-inferred.
    report.push_str(&format!(
        "production_attended_bits_unfused len={} bits={:?}\n",
        metal_unfused_full.len(),
        metal_unfused_full
    ));
    report.push_str(&format!(
        "production_attended_bits_fused len={} bits={:?}\n",
        metal_fused_full.len(),
        metal_fused_full
    ));
    report.push_str(&format!(
        "production_logits_bits_unfused len={} first8={:?}\n",
        metal_unfused_logits_full.len(),
        &metal_unfused_logits_full[..metal_unfused_logits_full.len().min(8)]
    ));
    report.push_str(&format!(
        "production_logits_bits_fused len={} first8={:?}\n",
        metal_fused_logits_full.len(),
        &metal_fused_logits_full[..metal_fused_logits_full.len().min(8)]
    ));

    let path = std::env::var("PROXIMA_ATTN_PAYLOAD_PATH")
        .unwrap_or_else(|_| "attn_parity_payload.txt".to_string());
    std::fs::write(&path, &report)?;
    eprintln!("attn_layer0_report step=1 layer=0 node={} path={path}", attended_node.0);
    eprint!("{report}");

    write_attn_read_source_vectors(
        program,
        symbols,
        named_blocks,
        &shapes,
        &by_node,
        fused_resolved,
        resident_names,
        input_placements,
        runtime,
        attended_node,
        metal_unfused_full,
    )?;

    Ok(())
}

/// `PROXIMA_ATTN_VECTORS_DIR` optional dump: the 15 target nodes' own
/// read-source closure (every `Input(NodeId(_))` a target's emitted
/// [`omega::Kernel::bindings`] names -- same node ids the `.grid.txt` dumps
/// already show) plus the 15 targets themselves, requested as extra outputs
/// of ONE evaluation of the production UNFUSED plan. `PROXIMA_ATTN_MATH_MODE`
/// (same convention as `build_plan` above, default `safe`, unset preserves
/// the r3 fixture convention exactly) selects Safe- or Relaxed-native
/// capture -- round-9 needs both: Safe to isolate transcription bugs from
/// compile-mode contraction, Relaxed because that is what
/// `resident_nocopy_cache::dispatch` actually compiles for the serving
/// default. Also proves requesting those extra outputs does not move
/// `attended_node` by diffing this evaluation's bits against
/// `metal_unfused_full` (the SAME node's bits from
/// `run_attn_fuse_parity_probe`'s own unfused evaluation, no extra outputs
/// requested there, same math mode both places).
///
/// `target_node_offsets` are the 14 absorbed nodes' ids MINUS
/// `attended_node.0` -- `S/census/absorbed_nodes.txt`'s own per-layer
/// `ABSORBED` blocks show every gemma4 layer (0, 1, 2, 4, 10 checked
/// directly) shares this EXACT relative layout (node numbering is one fixed
/// per-layer template offset by a constant 189-node stride), so the 15-node
/// closure generalizes to any layer from its own attended node alone,
/// without re-deriving a base per layer.
#[cfg(all(
    feature = "metal",
    feature = "metal-fuse-attn-decode",
    feature = "metal-output-placement",
    target_os = "macos"
))]
const ATTN_ABSORBED_NODE_OFFSETS: [i32; 14] =
    [-32, -31, -27, -20, -24, -17, -15, -14, -12, -9, -10, -8, -4, -2];

#[cfg(all(
    feature = "metal",
    feature = "metal-fuse-attn-decode",
    feature = "metal-output-placement",
    target_os = "macos"
))]
#[allow(clippy::too_many_arguments)]
fn write_attn_read_source_vectors(
    program: &[Op],
    symbols: &[u64],
    named_blocks: &[(&str, QuantizedBlock<'_>)],
    shapes: &proxima_tensor::Shapes,
    by_node: &alloc::collections::BTreeMap<NodeId, &proxima_tensor::BoundOp>,
    fused_resolved: &[proxima_tensor::BoundOp],
    resident_names: &alloc::collections::BTreeSet<&str>,
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
    runtime: &BackendRuntime,
    attended_node: NodeId,
    metal_unfused_full: &[u32],
) -> Result<(), InteropError> {
    let Ok(dir) = std::env::var("PROXIMA_ATTN_VECTORS_DIR") else {
        return Ok(());
    };

    let target_nodes: Vec<u32> = ATTN_ABSORBED_NODE_OFFSETS
        .iter()
        .map(|offset| (attended_node.0 as i32 + offset) as u32)
        .chain(core::iter::once(attended_node.0))
        .collect();

    let mut all_nodes: alloc::collections::BTreeSet<NodeId> =
        target_nodes.iter().map(|&id| NodeId(id)).collect();
    for &id in &target_nodes {
        if let Some(bound) = by_node.get(&NodeId(id)) {
            for (operand, _, _) in bound.all_read_sources() {
                all_nodes.insert(*operand);
            }
        }
    }
    let outputs: Vec<NodeId> = all_nodes.iter().copied().collect();

    let placed_input_nodes: Vec<NodeId> =
        input_placements.iter().map(|(node, _, _)| *node).collect();
    let mut plan = plan_named_with_placed_inputs(
        program,
        symbols,
        named_blocks,
        &outputs,
        runtime.numeric_policy,
        &placed_input_nodes,
        false,
    )?;
    plan.mark_resident(resident_names);
    if runtime.plan_time_constants {
        plan.mark_plan_time_constants_resident();
    }
    // same `PROXIMA_ATTN_MATH_MODE` convention as `build_plan` above
    // (default `safe`, unset preserves the r3 fixture convention exactly)
    // -- round-9 needs a Relaxed-native capture of these same vectors to
    // gate the production (Relaxed) unfused bytes against, not only Safe.
    let vectors_math_mode = match std::env::var("PROXIMA_ATTN_MATH_MODE").ok().as_deref() {
        Some("relaxed") => omega::MathMode::Relaxed,
        _ => omega::MathMode::Safe,
    };
    plan.set_math_mode(vectors_math_mode)?;
    plan.set_dispatch_type(runtime.dispatch_type);
    let evaluated =
        execute_plan_named_with_placements(&plan, named_blocks, input_placements, &[])?;

    std::fs::create_dir_all(&dir)?;

    // `PROXIMA_ATTN_DUMP_FUSED=1` -- candidate B's own fused ops (the
    // `CachedSoftmaxWeights` node plus the 166 `Reduce` whose epilogue reads
    // it), dumped from the FUSED bind (`fused_resolved`) rather than
    // `by_node` (which is the unfused chain-walk bind and never contains
    // either op). No-op under the pre-candidate-B shape, where neither kind
    // appears in `fused_resolved` at all.
    if std::env::var("PROXIMA_ATTN_DUMP_FUSED").as_deref() == Ok("1") {
        let mut fused_manifest = String::new();
        for bound in fused_resolved.iter().filter(|bound| {
            matches!(
                bound.kind,
                proxima_tensor::BoundOpKind::CachedSoftmaxWeights { .. }
            ) || matches!(
                &bound.kind,
                proxima_tensor::BoundOpKind::Reduce { epilogue_operands, .. }
                    if !epilogue_operands.is_empty()
            )
        }) {
            let manifest_line = format!(
                "node={} kind={} extents={:?} operands={:?}",
                bound.node.0,
                bound.kind.name(),
                bound.extents,
                bound.operands()
            );
            eprintln!("attn_node_dump_fused {manifest_line}");
            fused_manifest.push_str(&manifest_line);
            fused_manifest.push('\n');
        }
        std::fs::write(format!("{dir}/fused_manifest.txt"), &fused_manifest)?;
    }

    let mut manifest = String::new();
    for &node in &outputs {
        let extents = shapes.of(node);
        let Some((values, _)) = evaluated.get(node) else {
            manifest.push_str(&format!("node={} MISSING\n", node.0));
            continue;
        };
        manifest.push_str(&format!(
            "node={} elements={} extents={extents:?} source=production_metal_unfused\n",
            node.0,
            values.len(),
        ));
        let mut bits_text = String::with_capacity(values.len() * 11);
        for value in values {
            bits_text.push_str(&format!("0x{:08x}\n", value.to_bits()));
        }
        std::fs::write(format!("{dir}/{}.bits", node.0), &bits_text)?;
    }
    std::fs::write(format!("{dir}/manifest.txt"), &manifest)?;

    let attended_from_vectors_run: Vec<u32> = evaluated
        .get(attended_node)
        .map(|(values, _)| values.iter().map(|value| value.to_bits()).collect())
        .unwrap_or_default();
    let mut unchanged_report = String::new();
    unchanged_report.push_str(&format!(
        "attended_node={} normal_probe_element0={} normal_probe_element1={} extra_outputs_element0={} extra_outputs_element1={}\n",
        attended_node.0,
        bits_at_slice(metal_unfused_full, 0),
        bits_at_slice(metal_unfused_full, 1),
        bits_at_slice(&attended_from_vectors_run, 0),
        bits_at_slice(&attended_from_vectors_run, 1),
    ));
    let element_count = metal_unfused_full.len().max(attended_from_vectors_run.len());
    let mut all_unchanged = true;
    for index in 0..element_count {
        let normal_probe = metal_unfused_full.get(index).copied();
        let extra_outputs = attended_from_vectors_run.get(index).copied();
        let unchanged = normal_probe == extra_outputs;
        all_unchanged &= unchanged;
        unchanged_report.push_str(&format!(
            "index={index} normal_probe={} extra_outputs={} unchanged={unchanged}\n",
            normal_probe.map_or("MISSING".to_string(), |bits| format!("0x{bits:08x}")),
            extra_outputs.map_or("MISSING".to_string(), |bits| format!("0x{bits:08x}")),
        ));
    }
    unchanged_report.push_str(&format!("all_unchanged={all_unchanged}\n"));
    std::fs::write(format!("{dir}/attended_unchanged.txt"), &unchanged_report)?;
    eprintln!(
        "attn_vectors_dump nodes={} all_unchanged={all_unchanged} dir={dir}",
        outputs.len()
    );

    Ok(())
}

/// `fnv1a64` of `logits`' raw bit patterns -- lets a `metal-fuse-attn-decode`
/// build and a feature-off build be compared step-for-step on the same
/// prompt without diffing the full f32 buffer by hand.
#[cfg(feature = "metal")]
fn logits_bits_hash(logits: &[f32]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    logits
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .fold(OFFSET_BASIS, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(PRIME)
        })
}

/// Resolves the `PROXIMA_PREFILL_ONE_EVALUATION`/`PROXIMA_PREFILL_SEQUENTIAL`
/// escape hatches (`ServingConfig::prefill_one_evaluation`'s own doc) into
/// the one presence-based override
/// [`LoadedModel::run_decode_loop_observed_seeded`]'s prefill batch loop
/// consumes -- read exactly once, at that driver's own entry, rather than
/// inline at each of the two read sites. Presence, not a parsed value, is
/// what both env vars mean (`generate/tests_all.rs`'s `set_var(.., "1")`-
/// around-body tests rely on this), unchanged from the two inline
/// `std::env::var_os` calls this replaces.
fn prefill_one_evaluation_requested(serving_config: &ServingConfig) -> bool {
    (serving_config.prefill_one_evaluation
        || std::env::var_os("PROXIMA_PREFILL_ONE_EVALUATION").is_some())
        && std::env::var_os("PROXIMA_PREFILL_SEQUENTIAL").is_none()
}

impl<'file> LoadedModel<'file> {
    /// [`Self::generate_with_serving_config`] against
    /// [`supported_serving_config`] -- the reachable path every existing
    /// caller and test uses, unchanged: `gpu_layers: 0` always selects the
    /// CPU backend, so this runs exactly the forward it always has, on
    /// CPU, regardless of whether this build was compiled with the
    /// `metal` feature.
    pub(super) fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<(Vec<u32>, String, bool), InteropError> {
        self.generate_with_serving_config(
            prompt,
            max_tokens,
            supported_serving_config(
                0,
                #[cfg(all(feature = "metal", target_os = "macos"))]
                omega::MathMode::default(),
            ),
        )
    }

    /// This checkpoint's own static weight names -- `mark_resident`'s own
    /// caller-supplied classification (`BackendRuntime::evaluate`'s doc),
    /// bound once at [`Self::load`] time and never mutated again. Shared by
    /// every decode-loop variant that calls `mark_resident` and by `Drop`,
    /// which hands this SAME name set to
    /// `omega::backend::release_resident_names` so a dropped model evicts
    /// exactly the device buffers it caused and nothing another model's
    /// own names might collide with.
    pub(super) fn resident_names(&self) -> BTreeSet<&str> {
        self.weights
            .owned
            .iter()
            .map(|(name, _)| name.as_str())
            .chain(self.weights.packed.iter().map(|(name, _)| name.as_str()))
            .chain(
                self.weights
                    .packed_owned
                    .iter()
                    .map(|(name, _, _)| name.as_str()),
            )
            .collect()
    }

    /// This checkpoint's own declared KV/SSM cache leaf names, derived from
    /// the compiled program's `Op::Input` set, plus each layer's own pad-row
    /// widths -- the SINGLE source of truth both
    /// [`Self::run_decode_loop_observed_seeded`] and
    /// [`Self::forward_node_values_on_backend`] read to learn which leaf
    /// names a step must feed. Before this method existed, the decode loop
    /// derived these from the program (correct) while the one-shot forward
    /// path hard-coded `kv_cache.{layer}.{k_even,k_odd,v}` (wrong for any
    /// architecture, such as a partial-rotary attention layer, whose cache
    /// leaves are named differently) -- see this crate's own
    /// `StepInputArch`-style fixtures in `tests/` for the shape a foreign
    /// architecture takes advantage of. Never trusts
    /// `self.layer_roots[layer]`'s own hand-kept discriminant over what the
    /// program actually declared (`DeclaredCacheKind`'s own doc).
    ///
    /// # Errors
    ///
    /// [`InteropError::LayerCacheKindMismatch`] when a layer's declared
    /// program leaves disagree with `self.layer_roots`' own tag for it.
    /// `self`, checked against [`Self::declared_layer_cache_names_and_widths`]
    /// and discarded if that check errors -- every production
    /// [`Self::load`]/[`Self::load_with_registry`]/[`Self::load_from_safetensors`]
    /// construction site runs through this before it ever reaches the
    /// decode loop, so a layer whose `layer_roots` say it is stateful but
    /// whose program bakes that state as constants fails here, at load,
    /// instead of silently decoding from zero state
    /// ([`InteropError::LayerCacheLeavesMissing`]'s own doc).
    pub(super) fn validated(self) -> Result<Self, InteropError> {
        self.declared_layer_cache_names_and_widths()?;
        Ok(self)
    }

    pub(super) fn declared_layer_cache_names_and_widths(
        &self,
    ) -> Result<(Vec<LayerCacheNames>, Vec<LayerPadRowWidths>), InteropError> {
        let program_input_names: BTreeSet<&str> = self
            .program
            .iter()
            .filter_map(|op| match op {
                Op::Input {
                    name: Some(name), ..
                } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        // Stays FULL LENGTH, one entry per `self.layer_roots` index --
        // [`Qwen35LayerRoots`]'s own doc: the decode loop's growth step
        // (`LoadedModel::run_decode_loop_observed_seeded`'s own
        // `active_layer_roots.iter().enumerate()` / `layer_caches[layer]`)
        // indexes `layer_caches` by the REAL architecture layer number, in
        // lockstep with `self.layer_roots` -- filtering a shared-KV layer
        // out of this vec would desync every later layer's index. A
        // shared-KV layer's own `DeclaredCacheKind::SharedFromLayer` entry
        // carries no leaf names and costs nothing at fill/grow time
        // ([`LayerCacheNames::SharedFromLayer`]'s own doc).
        let mut layer_cache_kinds: Vec<DeclaredCacheKind> =
            Vec::with_capacity(self.layer_roots.len());
        for (layer, roots) in self.layer_roots.iter().enumerate() {
            let bound = bound_cache_kind(roots);
            let declared = declared_cache_kind(&program_input_names, layer);
            match (bound, declared) {
                (DeclaredCacheKind::SharedFromLayer, None) => {
                    // by design: this layer reads a donor layer's own
                    // already-declared leaves in-graph, never its own.
                    layer_cache_kinds.push(DeclaredCacheKind::SharedFromLayer);
                }
                (_, None) => {
                    return Err(InteropError::LayerCacheLeavesMissing {
                        layer,
                        kind: bound.label(),
                        expected: bound.expected_leaf_templates(),
                    });
                }
                (_, Some(declared)) if declared != bound => {
                    return Err(InteropError::LayerCacheKindMismatch {
                        layer,
                        declared: declared.label(),
                        bound: bound.label(),
                    });
                }
                (_, Some(declared)) => layer_cache_kinds.push(declared),
            }
        }
        let cache_names: Vec<LayerCacheNames> = layer_cache_kinds
            .iter()
            .enumerate()
            .map(|(layer, kind)| match kind {
                DeclaredCacheKind::Attention => LayerCacheNames::Attention {
                    k_even: alloc::format!("kv_cache.{layer}.k_even"),
                    k_odd: alloc::format!("kv_cache.{layer}.k_odd"),
                    v: alloc::format!("kv_cache.{layer}.v"),
                },
                DeclaredCacheKind::DenseAttention => LayerCacheNames::DenseAttention {
                    k_first: alloc::format!("kv_cache.{layer}.k_first"),
                    k_second: alloc::format!("kv_cache.{layer}.k_second"),
                    k_pass: alloc::format!("kv_cache.{layer}.k_pass"),
                    v: alloc::format!("kv_cache.{layer}.v"),
                },
                DeclaredCacheKind::Ssm => LayerCacheNames::Ssm {
                    conv_history: alloc::format!("ssm_cache.{layer}.conv_history"),
                    state: alloc::format!("ssm_cache.{layer}.state"),
                },
                DeclaredCacheKind::SharedFromLayer => LayerCacheNames::SharedFromLayer,
            })
            .collect();
        let layer_row_widths: Vec<LayerPadRowWidths> = cache_names
            .iter()
            .map(|names| layer_pad_row_widths(&self.program, names))
            .collect();
        Ok((cache_names, layer_row_widths))
    }

    /// A fresh, empty [`LayerCacheState`] per layer, shaped off
    /// [`Self::declared_layer_cache_names_and_widths`]'s own `cache_names`/
    /// `layer_row_widths` pair -- the decode loop's own prefill-step state
    /// (absent a `seed`) and [`Self::forward_node_values_on_backend`]'s own
    /// always-fresh state (every one-shot forward starts from an empty
    /// cache, that method's own doc), unified so neither caller hand-picks
    /// which [`LayerCacheState`] variant a layer gets, or how big it starts,
    /// independently of what
    /// [`declared_layer_cache_names_and_widths`](Self::declared_layer_cache_names_and_widths)
    /// already decided. `layer_row_widths` (not
    /// [`crate::architecture::Architecture::step_state`]) is the `Ssm` arm's
    /// own size source -- see [`cache_leaf_total_elements`]'s own doc for
    /// why: a foreign `Architecture` that never overrides `step_state`
    /// (the trait's own `Ok(None)` default) still declares its
    /// `ssm_cache.{layer}.*` leaves as `Op::Input` ops, so the program
    /// itself, not a per-architecture hook, is what every layer's initial
    /// cache is sized from.
    pub(super) fn fresh_layer_caches(
        &self,
        cache_names: &[LayerCacheNames],
        layer_row_widths: &[LayerPadRowWidths],
    ) -> Vec<LayerCacheState> {
        cache_names
            .iter()
            .zip(layer_row_widths)
            .map(|(names, widths)| match (names, widths) {
                (LayerCacheNames::Attention { .. }, _) => {
                    LayerCacheState::Attention(LayerCache::new())
                }
                (LayerCacheNames::DenseAttention { .. }, _) => {
                    LayerCacheState::DenseAttention(Qwen35DenseAttentionCache::new())
                }
                (
                    LayerCacheNames::Ssm {
                        conv_history,
                        state,
                    },
                    LayerPadRowWidths::Ssm {
                        conv_history_len,
                        state_len,
                    },
                ) => {
                    #[cfg(feature = "instrument")]
                    debug!(
                        conv_history_name = %conv_history,
                        state_name = %state,
                        conv_history_len = *conv_history_len,
                        state_len = *state_len,
                        "ssm cache seeded at its program-declared shape"
                    );
                    #[cfg(not(feature = "instrument"))]
                    let _ = (conv_history, state);
                    LayerCacheState::Ssm(SsmLayerCache::new(*conv_history_len, *state_len))
                }
                (LayerCacheNames::SharedFromLayer, LayerPadRowWidths::SharedFromLayer) => {
                    LayerCacheState::SharedFromLayer
                }
                _ => unreachable!(
                    "cache_names/layer_row_widths built from the same layer_roots, in lockstep"
                ),
            })
            .collect()
    }

    /// Pushes one step's position/RoPE inputs, `cached_len`/`lm_head_row`
    /// scalars, [`Architecture::step_inputs`]' own per-step leaves, and
    /// every KV/SSM cache leaf ([`push_kv_named_blocks`]) into `named_blocks`
    /// -- the ONE assembly both
    /// [`Self::run_decode_loop_observed_seeded`] and
    /// [`Self::forward_node_values_on_backend`] call, so a foreign
    /// architecture's own [`Architecture::step_inputs`] override and its own
    /// cache leaf names are fed identically whether the caller is decoding
    /// token-by-token or tapping one interior node from a single forward
    /// pass. `position_inputs`/`cached_len_scalar`/`lm_head_row_scalar` are
    /// owned by the CALLER (not this method) so the `QuantizedBlock`s this
    /// method pushes can borrow them for `named_blocks`' own `'call`
    /// lifetime without a self-referential return type. Returns the
    /// [`bind_symbols`] result (this step's own `Extent::Symbolic` binding)
    /// rather than leaving the caller to re-borrow `step_input_scratch`
    /// afterward -- `named_blocks` already holds borrows into it once this
    /// method returns, so a second, independent borrow to compute symbols
    /// would conflict with the one `named_blocks` is holding.
    ///
    /// # Errors
    ///
    /// [`InteropError::UnknownStepInput`] if `Architecture::step_inputs`
    /// names a leaf this checkpoint's program never declared, plus whatever
    /// [`push_kv_named_blocks`]/[`bind_symbols`] can fail with.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn push_step_named_blocks<'call>(
        &'call self,
        position_inputs: &'call PositionInputs,
        cached_len_scalar: &'call [f32; 1],
        lm_head_row_scalar: &'call [f32; 1],
        token_history: &[u32],
        new_start: usize,
        new_count: usize,
        cache_names: &'call [LayerCacheNames],
        layer_caches: &'call [LayerCacheState],
        layer_row_widths: &[LayerPadRowWidths],
        kv_bound_extent: usize,
        kv_pad_scratch: &'call mut [KvPadScratch],
        qwen35_dense_pad_scratch: &'call mut [Qwen35DenseAttentionPadScratch],
        step_input_scratch: &'call mut Vec<StepInput>,
        named_blocks: &mut Vec<(&'call str, QuantizedBlock<'call>)>,
        single_position_step: bool,
    ) -> Result<Vec<u64>, InteropError> {
        named_blocks.push((
            "ids",
            QuantizedBlock::Int32(position_inputs.ids_i32.as_slice()),
        ));
        named_blocks.push((
            "eps",
            QuantizedBlock::Float32(position_inputs.epsilon.as_slice()),
        ));
        named_blocks.push((
            "rope_cos",
            QuantizedBlock::Float32(position_inputs.cos.as_slice()),
        ));
        named_blocks.push((
            "rope_sin",
            QuantizedBlock::Float32(position_inputs.sin.as_slice()),
        ));
        named_blocks.push((
            "cached_len",
            QuantizedBlock::Float32(cached_len_scalar.as_slice()),
        ));
        named_blocks.push((
            "lm_head_row",
            QuantizedBlock::Float32(lm_head_row_scalar.as_slice()),
        ));

        step_input_scratch.clear();
        if let Some(architecture_impl) = self.architecture_impl {
            let step_context = StepInputContext {
                all_token_ids: token_history,
                new_start,
                new_count,
            };
            architecture_impl.step_inputs(&step_context, step_input_scratch);
        }
        for step_input in step_input_scratch.iter() {
            if !self
                .program
                .iter()
                .any(|op| op.name() == Some(step_input.name))
            {
                return Err(InteropError::UnknownStepInput {
                    name: String::from(step_input.name),
                });
            }
            named_blocks.push(step_input.as_named_block());
        }
        let symbols = bind_symbols(
            new_count,
            kv_bound_extent,
            step_input_scratch,
            single_position_step,
        )?;

        push_kv_named_blocks(
            cache_names,
            layer_caches,
            layer_row_widths,
            kv_bound_extent,
            kv_pad_scratch,
            qwen35_dense_pad_scratch,
            named_blocks,
        )?;
        Ok(symbols)
    }

    /// Pages `bytes` in as expert `expert`'s new weight for layer `layer` --
    /// a thin forward onto [`crate::expert_slab::ExpertSlab::page_expert`],
    /// the primitive this method composes (P2 teaching surface: read that
    /// method's own doc for the aliasing/ownership contract this call
    /// changes). `layer` is the [`crate::bind::build_expert_slab`] SITE
    /// index (one slot per `blk.{n}.{ffn_gate,ffn_up,ffn_down}_exps.weight`
    /// tensor this checkpoint's own forward program bound, in that order),
    /// not necessarily the checkpoint's transformer layer number when more
    /// than one projection is routed per layer.
    ///
    /// # Errors
    /// [`InteropError::ExpertSwapDuringStep`] if called while a decode step
    /// is running; [`InteropError::ExpertSlabIndexOutOfRange`] if `layer` or
    /// `expert` is out of range for this checkpoint's slab.
    pub fn page_expert(
        &self,
        layer: usize,
        expert: usize,
        codec: crate::bind::Codec,
        bytes: &[u8],
        out_dim: u32,
        in_dim: u32,
    ) -> Result<u64, InteropError> {
        lock_expert_slab(&self.expert_slab)
            .page_expert(layer, expert, codec, bytes, out_dim, in_dim)
    }

    /// Pages a HOBBIT high-precision expert from one range of a live mmap,
    /// without copying its payload into a `Vec<u8>`. This composes
    /// [`crate::expert_slab::ExpertSlab::page_expert_mapped`]; that method
    /// retains the [`Arc<Mmap>`] through every per-step
    /// [`proxima_tensor::cpu::ExpertSource`] snapshot, so callers may drop
    /// their own mapping handle after this returns.
    ///
    /// `layer` is the same gathered-reduce site index as [`Self::page_expert`],
    /// while `range` selects exactly one encoded expert inside the source
    /// mapping. The slab's [`crate::expert_slab::ExpertSlab::memory`] report
    /// records this range as `mapped_bytes` and leaves `owned_bytes` unchanged.
    ///
    /// # Errors
    /// [`InteropError::ExpertMappedRangeOutOfBounds`] when `range` falls
    /// outside `mapping`, plus the same step/index errors as
    /// [`Self::page_expert`].
    pub fn page_expert_mapped(
        &self,
        layer: usize,
        expert: usize,
        codec: crate::bind::Codec,
        mapping: Arc<Mmap>,
        range: Range<usize>,
        dims: crate::expert_slab::WeightDims,
    ) -> Result<u64, InteropError> {
        lock_expert_slab(&self.expert_slab)
            .page_expert_mapped(layer, expert, codec, mapping, range, dims)
    }

    /// Attaches HOBBIT's mmap-backed low-codec expert store to this model.
    ///
    /// The mapping is parsed and indexed once, then every gate/up/down expert
    /// entry is replaced by its low-codec mapped view. The original checkpoint
    /// offsets retained by the sidecar become the high-codec promotion source.
    /// No expert payload is copied or heap-allocated by this operation.
    pub fn attach_expert_sidecar(&mut self, mapping: Arc<Mmap>) -> Result<(), InteropError> {
        mapping.advise(Advice::Random)?;
        let sidecar = crate::expert_sidecar::MappedExpertSidecar::new(mapping)?;
        self.attach_indexed_expert_sidecar(sidecar)
    }

    /// Attaches a mmap-backed sidecar and retains a checkpoint file for
    /// bounded high-codec reads during Metal expert staging.
    pub fn attach_expert_sidecar_with_checkpoint_file(
        &mut self,
        mapping: Arc<Mmap>,
        checkpoint_file: File,
    ) -> Result<(), InteropError> {
        mapping.advise(Advice::Random)?;
        let sidecar = crate::expert_sidecar::MappedExpertSidecar::new(mapping)?
            .with_checkpoint_file(checkpoint_file);
        self.attach_indexed_expert_sidecar(sidecar)
    }

    /// Attaches a sidecar that preads only the expert ranges selected per layer.
    pub fn attach_expert_sidecar_file(&mut self, file: File) -> Result<(), InteropError> {
        let sidecar = crate::expert_sidecar::MappedExpertSidecar::from_file(file)?;
        self.attach_indexed_expert_sidecar(sidecar)
    }

    /// Attaches a pread sidecar and a checkpoint file for bounded high-codec
    /// reads during Metal expert staging.
    pub fn attach_expert_sidecar_file_with_checkpoint_file(
        &mut self,
        sidecar_file: File,
        checkpoint_file: File,
    ) -> Result<(), InteropError> {
        let sidecar = crate::expert_sidecar::MappedExpertSidecar::from_file(sidecar_file)?
            .with_checkpoint_file(checkpoint_file);
        self.attach_indexed_expert_sidecar(sidecar)
    }

    pub(super) fn attach_indexed_expert_sidecar(
        &mut self,
        sidecar: crate::expert_sidecar::MappedExpertSidecar,
    ) -> Result<(), InteropError> {
        if !self
            .architecture_impl
            .is_some_and(|architecture| architecture.ffn_routing() == crate::architecture::FfnRouting::Routed)
        {
            return Err(InteropError::PreGatherExecutionUnsupported {
                architecture: self.architecture_impl.map_or_else(
                    || String::from("unknown"),
                    |value| String::from(value.name()),
                ),
                reason: String::from("expert sidecars require a qwen35moe expert graph"),
            });
        }
        sidecar.install_low_copies(
            &mut lock_expert_slab(&self.expert_slab),
            self.architecture.block_count as usize,
            self.architecture.expert_count as usize,
        )?;
        if std::env::var_os("PROXIMA_DEBUG_MEMORY_OWNERS").is_some() {
            let owned_bytes = self
                .weights
                .owned
                .iter()
                .map(|(_, values)| values.len() * core::mem::size_of::<f32>())
                .sum::<usize>();
            let packed_bytes = self
                .weights
                .packed
                .iter()
                .map(|(_, block)| block.packed_bytes().map_or(0, <[u8]>::len))
                .sum::<usize>();
            let packed_owned_bytes = self
                .weights
                .packed_owned
                .iter()
                .map(|(_, bytes, _)| bytes.len())
                .sum::<usize>();
            let slab_memory = lock_expert_slab(&self.expert_slab).memory();
            eprintln!(
                "qwen35 memory owners checkpoint_bytes={} owned_bytes={} packed_bytes={} packed_owned_bytes={} sidecar_mapped_bytes={} sidecar_owned_bytes={} sidecar_descriptors={}",
                self.checkpoint_bytes,
                owned_bytes,
                packed_bytes,
                packed_owned_bytes,
                slab_memory.mapped_bytes,
                slab_memory.owned_bytes,
                sidecar.descriptor_count(),
            );
        }
        // The whole-checkpoint Metal buffer is a convenient zero-copy fast
        // path for ordinary GGUF serving, but it makes the driver account for
        // the entire mmap even when HOBBIT substitutes every expert.  Once a
        // sidecar owns the expert bytes, drop that device-wide mapping so the
        // remaining tensors bind independently and the residency budget is
        // reflected by actual device buffers.
        #[cfg(feature = "metal")]
        {
            sidecar.discard_checkpoint_expert_pages(self.checkpoint_mapping)?;
            omega::backend::unregister_checkpoint_mapping(self.checkpoint_mapping);
        }
        self.expert_sidecar = Some(sidecar);
        Ok(())
    }

    /// Number of sidecar expert-projection records owned by this model.
    #[must_use]
    pub fn expert_sidecar_descriptor_count(&self) -> usize {
        self.expert_sidecar.as_ref().map_or(
            0,
            crate::expert_sidecar::MappedExpertSidecar::descriptor_count,
        )
    }

    /// Applies one DynaExq/HOBBIT resident-set transition between decode
    /// steps.  The caller owns the policy and the high-precision source; this
    /// method only joins that policy to this model's slab, which is the table
    /// the next decode step snapshots as [`ExpertSource`] entries.  Keeping
    /// the page callback generic preserves the zero-allocation boundary and
    /// lets a caller return bytes from an mmap or LSM segment without a
    /// trait-object allocation.
    ///
    /// The method deliberately does not apply actions while a step is active:
    /// [`ExpertSlab`] returns its typed boundary error, preventing a policy
    /// update from invalidating the borrowed sources of the current step.
    pub fn apply_expert_residency<
        const LAYERS: usize,
        const EXPERTS: usize,
        const ACTIONS: usize,
        Page,
    >(
        &self,
        policy: &mut crate::residency::ExpertResidency<LAYERS, EXPERTS>,
        actions: &crate::residency::ResidencyActions<ACTIONS>,
        page: Page,
    ) -> Result<(), InteropError>
    where
        Page: FnMut(
            crate::residency::ExpertAddress,
        ) -> Result<crate::residency::ExpertPage<'file>, InteropError>,
    {
        let mut slab = lock_expert_slab(&self.expert_slab);
        policy.apply_at_boundary(&mut slab, actions, page)
    }

    /// Applies a fixed DynaExq action batch to the attached HOBBIT sidecar.
    /// A page promotes all three projections from their original checkpoint
    /// ranges; an eviction restores all three low-codec mapped ranges.
    pub fn apply_attached_expert_residency<
        const LAYERS: usize,
        const EXPERTS: usize,
        const ACTIONS: usize,
    >(
        &self,
        policy: &mut crate::residency::ExpertResidency<LAYERS, EXPERTS>,
        actions: &crate::residency::ResidencyActions<ACTIONS>,
    ) -> Result<(), InteropError> {
        let sidecar = self.expert_sidecar.as_ref().ok_or_else(|| {
            InteropError::PreGatherExecutionUnsupported {
                architecture: String::from("qwen35moe"),
                reason: String::from("no expert sidecar is attached"),
            }
        })?;
        let mut slab = lock_expert_slab(&self.expert_slab);
        policy.apply_actions_at_boundary(&mut slab, actions, |slab, action| {
            sidecar.apply_action(slab, self.checkpoint_mapping, action)
        })
    }

    pub(super) fn reconcile_attached_qwen35moe_residency(
        &self,
        policy: &mut crate::residency::ExpertResidency<40, 256>,
    ) -> Result<(), InteropError> {
        let actions = policy.reconcile::<{ 40 * 256 * 2 }>().map_err(|error| {
            InteropError::PreGatherExecutionUnsupported {
                architecture: String::from("qwen35moe"),
                reason: error.to_string(),
            }
        })?;
        self.apply_attached_expert_residency(policy, &actions)
    }

    /// Runs the explicit router -> residency -> gather protocol for a
    /// qwen35moe runtime integration.  The current forward graph exposes the
    /// router and routed gather as one graph evaluation, so this seam accepts
    /// a caller-owned router prepass and source transition rather than
    /// pretending that the existing graph has been partitioned.  A caller
    /// that has not built that prepass gets a typed error from its router
    /// callback; the gather callback is never invoked before the boundary.
    ///
    /// The callbacks are consuming and return their storage to the caller;
    /// no trait object, boxed future, or runtime allocation is introduced by
    /// this phase boundary.  `Routes` may be a fixed-capacity route array or
    /// a `Vec` owned by the caller, and `Source` may be an expert slab view or
    /// an mmap-backed table.
    pub fn execute_qwen35moe_pre_gather<Routes, Source, Output, Router, Boundary, Gather>(
        &self,
        router: Router,
        boundary: Boundary,
        gather: Gather,
    ) -> Result<Output, InteropError>
    where
        Routes: AsRef<[crate::residency::ServeDecision]>,
        Router: FnOnce() -> Result<Routes, InteropError>,
        Boundary:
            FnOnce(crate::qwen35moe::execution::RouterResult<'_>) -> Result<Source, InteropError>,
        Gather: FnOnce(
            crate::qwen35moe::execution::GatherPhase<'_, Source>,
        ) -> Result<Output, InteropError>,
    {
        let architecture = self.architecture_impl.map_or("unknown", Architecture::name);
        if !self
            .architecture_impl
            .is_some_and(|architecture| architecture.ffn_routing() == crate::architecture::FfnRouting::Routed)
        {
            return Err(InteropError::PreGatherExecutionUnsupported {
                architecture: String::from(architecture),
                reason: String::from("the bound model is not a routed qwen35moe graph"),
            });
        }
        if self.router_roots.is_empty() {
            return Err(InteropError::PreGatherExecutionUnsupported {
                architecture: String::from(architecture),
                reason: String::from("the bound qwen35moe graph exposes no router roots"),
            });
        }
        crate::qwen35moe::execution::execute_pre_gather(router, boundary, gather)
    }

    /// Reports the active expert payloads retained by this model's slab.
    /// `owned_bytes` is the actual copied-payload footprint; `mapped_bytes`
    /// is the address-space range served directly from mmap. See
    /// [`crate::expert_slab::ExpertSlabMemory`] for why the latter is not an
    /// RSS claim.
    #[must_use]
    pub fn expert_slab_memory(&self) -> crate::expert_slab::ExpertSlabMemory {
        lock_expert_slab(&self.expert_slab).memory()
    }

    /// Removes expert `expert` of layer `layer`'s currently-bound bytes --
    /// see [`Self::page_expert`]'s own doc for what `layer` indexes, and
    /// [`crate::expert_slab::ExpertSlab::evict_expert`] for the primitive
    /// this composes.
    ///
    /// # Errors
    /// Same as [`Self::page_expert`].
    pub fn evict_expert(&self, layer: usize, expert: usize) -> Result<(), InteropError> {
        lock_expert_slab(&self.expert_slab).evict_expert(layer, expert)
    }

    /// `expert`'s current epoch for `layer`, or `None` if either index is
    /// out of range or the expert is currently evicted -- see
    /// [`crate::expert_slab::ExpertSlab::expert_epoch`].
    #[must_use]
    pub fn expert_epoch(&self, layer: usize, expert: usize) -> Option<u64> {
        lock_expert_slab(&self.expert_slab).expert_epoch(layer, expert)
    }

    /// The greedy decode loop itself: `max_tokens` steps, each one call
    /// into `BackendRuntime::evaluate` against `new_positions == 1` after
    /// the first step (`new_positions == prompt_length` on the first),
    /// growing `LayerCache` by one call's worth of positions every step
    /// instead of re-running the whole sequence from scratch -- stopping
    /// early the moment the model emits its own end-of-sequence id (see
    /// this module's doc for what that id is on the real checkpoint),
    /// never running past `max_tokens` regardless.
    ///
    /// `serving_config` is a caller-supplied override of
    /// `supported_serving_config`'s default -- the same [`ServingConfig`]
    /// [`apply_serving_config`] already gates, never a second selection
    /// mechanism. Setting `gpu_layers` to `GPU_LAYERS_ALL` (`-ngl all`) on
    /// a build compiled with this crate's `metal` feature runs this same
    /// loop against the Metal backend instead of the CPU one; every other
    /// field must already satisfy [`apply_serving_config`]'s gate the same
    /// way `supported_serving_config`'s does.
    pub fn generate_with_serving_config(
        &self,
        prompt: &str,
        max_tokens: usize,
        serving_config: ServingConfig,
    ) -> Result<(Vec<u32>, String, bool), InteropError> {
        let serving_config = {
            let mut serving_config = serving_config;
            self.apply_command_buffer_chunks_default(&mut serving_config);
            #[cfg(all(feature = "metal", target_os = "macos"))]
            self.apply_memory_fit_gate(&mut serving_config)?;
            serving_config
        };
        #[cfg(feature = "std")]
        apply_fusion_env_switches(&serving_config);
        let mut runtime = BackendRuntime::new(&serving_config);
        self.run_decode_loop(prompt, max_tokens, &serving_config, &mut runtime)
    }

    /// Runs one forward pass over `prompt`'s own tokens and returns the
    /// [`PrefixState`] it leaves behind, WITHOUT decoding anything past it
    /// -- `Self::run_decode_loop_observed_seeded` with `seed: None` and
    /// `max_tokens: 1`: step 0 of that loop always forwards the whole
    /// `next_ids` range against `cached_len == 0` before ever sampling, so
    /// asking for exactly one step is asking for exactly the prefill this
    /// primitive needs and nothing past it. The one token step 0 happens to
    /// sample (this loop's own next-token prediction) is discarded -- it is
    /// never forward-passed itself, so it is not part of the cache
    /// [`PrefixState`] reports; a caller after real generated text wants
    /// [`Self::generate_from_prefix`], not this method's own return.
    ///
    /// The returned [`PrefixState`] is independent of `self` and of this
    /// call's own `runtime` -- reusable across as many
    /// [`Self::generate_from_prefix`] calls as the caller likes, against
    /// this SAME `LoadedModel`, without re-running this forward pass.
    ///
    /// # Errors
    ///
    /// Same as [`Self::generate_with_serving_config`].
    pub fn prefill_prefix(
        &self,
        prompt: &str,
        serving_config: &ServingConfig,
    ) -> Result<PrefixState, InteropError> {
        let effective_serving_config = {
            let mut effective_serving_config = *serving_config;
            self.apply_command_buffer_chunks_default(&mut effective_serving_config);
            #[cfg(all(feature = "metal", target_os = "macos"))]
            self.apply_memory_fit_gate(&mut effective_serving_config)?;
            effective_serving_config
        };
        let mut runtime = BackendRuntime::new(&effective_serving_config);
        let (_generated_ids, _text, _stopped_by_eos, prefix_state) = self
            .run_decode_loop_observed_seeded(
                prompt,
                1,
                &effective_serving_config,
                &mut runtime,
                None,
                &mut LogitsSink::Discard,
                &mut NodeValuesSink::Discard,
                &mut |_event| ControlFlow::Continue(()),
                None,
                true,
            )?;
        Ok(prefix_state)
    }

    /// Resumes decoding from `prefix` -- `Self::run_decode_loop_observed_seeded`
    /// with `seed: Some(prefix.clone())`, `prompt` now the SUFFIX text only,
    /// so this call's own two-range forward starts from `prefix`'s cached
    /// `cached_len` rows instead of `0`, and prefills ONLY the suffix's own
    /// tokens as the new range -- the multi-row prefill
    /// [`Self::prefill_prefix`] already ran for `prefix`'s own tokens is
    /// never repeated. `prefix` is cloned, never consumed: a second call
    /// against a different suffix, or a second `LoadedModel` call entirely,
    /// sees `prefix` exactly as this call received it.
    ///
    /// `suffix` is tokenized with neither BOS nor EOS added -- it continues
    /// the sequence `prefix` already opened, so the tokenizer must see it as
    /// a continuation, not a fresh prompt. If the two texts' own token
    /// boundary does not fall on a token the tokenizer would also choose
    /// when encoding `prefix_text + suffix_text` as one string (a
    /// unigram/BPE tokenizer can merge a trailing/leading fragment across a
    /// naive substring split), the caller owns splitting the prompt on a
    /// boundary the tokenizer already treats as a hard break -- a newline is
    /// the reliable one for this crate's own vocabularies.
    ///
    /// # Errors
    ///
    /// Same as [`Self::generate_with_serving_config`].
    pub fn generate_from_prefix(
        &self,
        prefix: &PrefixState,
        suffix: &str,
        max_tokens: usize,
        serving_config: &ServingConfig,
        on_token: &mut dyn FnMut(TokenEvent<'_>) -> ControlFlow<(), ()>,
    ) -> Result<(Vec<u32>, String, bool), InteropError> {
        let effective_serving_config = {
            let mut effective_serving_config = *serving_config;
            self.apply_command_buffer_chunks_default(&mut effective_serving_config);
            // Prefix-resume reaches the same device allocator as ordinary
            // generation. Apply the identical load-time budget before the
            // resumed step, otherwise a caller can bypass the hard memory
            // ceiling simply by supplying a PrefixState.
            #[cfg(all(feature = "metal", target_os = "macos"))]
            self.apply_memory_fit_gate(&mut effective_serving_config)?;
            effective_serving_config
        };
        let mut runtime = BackendRuntime::new(&effective_serving_config);
        let seed = PrefixState {
            ids: prefix.ids.clone(),
            layer_caches: prefix.layer_caches.clone(),
            cached_len: prefix.cached_len,
        };
        let (generated_ids, text, stopped_by_eos, _final_state) = self
            .run_decode_loop_observed_seeded(
                suffix,
                max_tokens,
                &effective_serving_config,
                &mut runtime,
                None,
                &mut LogitsSink::Discard,
                &mut NodeValuesSink::Discard,
                on_token,
                Some(seed),
                true,
            )?;
        Ok((generated_ids, text, stopped_by_eos))
    }

    /// Applies this checkpoint's own resolved
    /// [`crate::architecture::Architecture::command_buffer_chunks`] as
    /// `serving_config.command_buffer_chunks`'s default -- only when the
    /// caller left that field at [`ServingConfig`]'s own type default of
    /// `1`, never overriding an explicit non-default caller value. `self
    /// .architecture_impl` is `None` for every non-registry load entry point
    /// (`Self::architecture_impl`'s own doc), which leaves this a no-op:
    /// every field this method could write is already `1`.
    pub(super) fn apply_command_buffer_chunks_default(&self, serving_config: &mut ServingConfig) {
        if serving_config.command_buffer_chunks != 1 {
            return;
        }
        if let Some(architecture) = self.architecture_impl {
            serving_config.command_buffer_chunks = architecture.command_buffer_chunks();
        }
    }

    /// The first auto-tune step (`crate::memory_fit`'s own module doc):
    /// derives this checkpoint's device-memory budget from its own shape at
    /// `serving_config.context_length`, probes the host's own device facts
    /// ([`omega::metal::system_memory_facts`]), and either leaves
    /// `serving_config.context_length` unchanged, reduces it to the
    /// largest value that fits (emitting a `context_length_reduced` warn
    /// event under `feature = "instrument"` -- callers that need the
    /// reduced value read it back off `serving_config` after this call
    /// returns, since it takes `&mut`), or refuses with
    /// [`InteropError::MemoryBudgetExceeded`] -- always
    /// before [`Self::generate_with_serving_config`]'s own next line
    /// ([`BackendRuntime::new`]) asks a device for a single buffer.
    ///
    /// A no-op when `serving_config.gpu_memory_fit` is `false` (the
    /// caller's explicit override, matching every other opt-out knob
    /// [`ServingConfig`]'s own doc already has) or when this host has no
    /// Metal device at all ([`omega::metal::system_memory_facts`] returning
    /// `Err`) -- a probe failure means this method has nothing to gate
    /// against, not that the load itself is unsafe, so it fails OPEN
    /// (proceeds unchanged) rather than refusing a load this crate cannot
    /// actually evaluate.
    ///
    /// # Errors
    ///
    /// [`InteropError::MemoryBudgetExceeded`] when even a context length of
    /// `1` cannot fit this checkpoint's own weights plus the fixed arena
    /// allowance inside the host's own reported limit.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(super) fn apply_memory_fit_gate(
        &self,
        serving_config: &mut ServingConfig,
    ) -> Result<(), InteropError> {
        if !serving_config.gpu_memory_fit {
            return Ok(());
        }
        let Ok(facts) = omega::metal::system_memory_facts() else {
            return Ok(());
        };
        let detected_limit = crate::memory_fit::HostMemoryLimit {
            limit_bytes: facts
                .recommended_max_working_set_size
                .min(facts.physical_memory_bytes),
            os_headroom_bytes: omega::sized::LOAD_TIME_FIT_OS_HEADROOM_BYTES,
        };
        let limit = serving_config
            .gpu_memory_limit_bytes
            .map_or(detected_limit, |configured| {
                crate::memory_fit::HostMemoryLimit {
                    limit_bytes: configured.min(detected_limit.available_bytes()),
                    os_headroom_bytes: 0,
                }
            });
        // Page-rounded on the dense class only: the checkpoint's whole
        // mmap is ONE no-copy `MTLBuffer`
        // (`omega::metal::checkpoint_mapping_offset`'s own doc), so the
        // real device allocation is `file_bytes.len()` rounded up to a
        // page, not the plain sum of per-tensor byte counts (which excludes
        // the GGUF header/metadata region) -- rounding the dense class
        // absorbs that difference without inventing a fourth bucket for a
        // few-KB header.
        let weights = crate::memory_fit::WeightClassBytes {
            dense_bytes: self
                .checkpoint_weight_bytes
                .dense_bytes
                .next_multiple_of(omega::metal::page_size() as u64),
            ..self.checkpoint_weight_bytes
        };
        let requested_context_length = serving_config.context_length;
        let (context_length, outcome) = crate::memory_fit::fit_context_length(
            weights,
            self.architecture.block_count,
            self.architecture.kv_heads,
            self.architecture.head_dim,
            requested_context_length,
            omega::sized::LOAD_TIME_FIT_ARENA_ALLOWANCE_BYTES,
            limit,
        )?;
        // does the caller's live memory state (e.g. another process holding
        // GPU-resident weights) push this gate to actually shrink
        // context_length, versus a static reading of the isolated numbers.
        #[cfg(feature = "instrument")]
        debug!(
            requested_context_length,
            fit_context_length = context_length,
            available_bytes = limit.available_bytes(),
            outcome = ?outcome,
            "apply_memory_fit_gate: live fit decision"
        );
        let per_class_budget = crate::memory_fit::MemoryBudget::derive(
            weights,
            self.architecture.block_count,
            self.architecture.kv_heads,
            self.architecture.head_dim,
            context_length,
            omega::sized::LOAD_TIME_FIT_ARENA_ALLOWANCE_BYTES,
        );
        crate::memory_fit::fit_per_class_budgets(
            per_class_budget,
            crate::memory_fit::PerClassBudgets {
                dense_weights_budget_bytes: serving_config.dense_weights_budget_bytes,
                expert_weights_budget_bytes: serving_config.expert_weights_budget_bytes,
                activations_budget_bytes: serving_config.activations_budget_bytes,
                kv_cache_budget_bytes: serving_config.kv_cache_budget_bytes,
            },
        )?;
        #[cfg(feature = "instrument")]
        {
            let budget = per_class_budget;
            info!(
                dense_weights_bytes = budget.dense_weights_bytes,
                expert_weights_bytes = budget.expert_weights_bytes,
                table_weights_bytes = budget.table_weights_bytes,
                kv_cache_bytes = budget.kv_cache_bytes,
                ssm_state_bytes = budget.ssm_state_bytes,
                arena_allowance_bytes = budget.arena_allowance_bytes,
                total_bytes = budget.total_bytes(),
                limit_bytes = limit.limit_bytes,
                os_headroom_bytes = limit.os_headroom_bytes,
                "memory_budget: load-time device-memory budget derived from checkpoint shape, by class"
            );
        }
        if matches!(
            outcome,
            crate::memory_fit::FitOutcome::ReducedContext { .. }
        ) {
            #[cfg(feature = "instrument")]
            if let crate::memory_fit::FitOutcome::ReducedContext { from, to } = outcome {
                proxima_telemetry::warn!(
                    from = from,
                    to = to,
                    "context_length_reduced: requested context length did not fit, reduced \
                     to the largest value that does"
                );
            }
            serving_config.context_length = context_length;
        }
        Ok(())
    }

    /// Shared by [`Self::generate_with_serving_config`] and this crate's
    /// own metal-path tests, which need to read `runtime`'s plan-cache
    /// hit/miss counters after the loop finishes -- a caller reachable
    /// only through the public method above never sees `runtime` at all.
    /// Thin delegation to [`Self::run_decode_loop_observed`] with no forced
    /// continuation and a no-op logits sink, so every existing call site
    /// keeps its exact pre-existing behavior.
    pub(crate) fn run_decode_loop(
        &self,
        prompt: &str,
        max_tokens: usize,
        serving_config: &ServingConfig,
        runtime: &mut BackendRuntime,
    ) -> Result<(Vec<u32>, String, bool), InteropError> {
        self.run_decode_loop_observed(
            prompt,
            max_tokens,
            serving_config,
            runtime,
            None,
            &mut LogitsSink::Discard,
            &mut |_event| ControlFlow::Continue(()),
        )
    }

    /// [`Self::generate_with_serving_config`], plus a `TokenEvent` for
    /// every step `decode_until_stop_or_budget` already produces --
    /// `Self::run_decode_loop_observed`'s own loop, unchanged, given a
    /// real `on_token` instead of `run_decode_loop`'s `&mut |_| Continue`.
    /// There is one decode loop in this crate; this and
    /// [`Self::generate_with_serving_config`] are the same call with
    /// different callbacks, never two implementations of the loop itself.
    ///
    /// `on_token` sees exactly what [`TokenEvent`]'s own field docs promise:
    /// one [`Phase::Prefill`] event at step `0` (prompt token count, that
    /// step's own forward-pass latency), then one [`Phase::Token`] event
    /// per generated token, `text_piece`s concatenating to this call's
    /// returned `String` on the same ids as its returned `Vec<u32>`.
    /// Returning [`ControlFlow::Break`] from any call ends decoding after
    /// that token, same as [`decode_until_stop_or_budget`]'s own doc: this
    /// call then returns `finished = false`, exactly like running out of
    /// `max_tokens`, never mistaken for the model's own eos.
    ///
    /// # Errors
    ///
    /// Same as [`Self::generate_with_serving_config`].
    pub fn generate_streaming(
        &self,
        prompt: &str,
        max_tokens: usize,
        serving_config: ServingConfig,
        on_token: &mut dyn FnMut(TokenEvent<'_>) -> ControlFlow<(), ()>,
    ) -> Result<(Vec<u32>, String, bool), InteropError> {
        let serving_config = {
            let mut serving_config = serving_config;
            self.apply_command_buffer_chunks_default(&mut serving_config);
            #[cfg(all(feature = "metal", target_os = "macos"))]
            self.apply_memory_fit_gate(&mut serving_config)?;
            serving_config
        };
        let mut runtime = BackendRuntime::new(&serving_config);
        #[cfg(feature = "metal")]
        if std::env::var_os("PROXIMA_WARMUP_BEFORE_GENERATE").is_some() {
            let mut warmup_callback = |_event: TokenEvent<'_>| ControlFlow::Continue(());
            runtime.retain_monolithic_prefill_sources = true;
            let warmup_result = self.run_decode_loop_observed(
                prompt,
                1,
                &serving_config,
                &mut runtime,
                None,
                &mut LogitsSink::Discard,
                &mut warmup_callback,
            );
            runtime.retain_monolithic_prefill_sources = false;
            if let Err(error) = warmup_result {
                clear_expert_source_cache();
                return Err(error);
            }
        }
        self.run_decode_loop_observed(
            prompt,
            max_tokens,
            &serving_config,
            &mut runtime,
            None,
            &mut LogitsSink::Discard,
            on_token,
        )
    }

    /// [`Self::run_decode_loop`]'s own body, plus the two hooks
    /// [`crate::quality::quality_report`] needs to score a variant against
    /// a reference through this SAME cached decode loop rather than a
    /// second, uncached one: `token_override` -- when `Some`, step
    /// `_step`'s emitted token is `token_override[_step]` instead of this
    /// call's own greedy sample, so a second [`LoadedModel`] can be driven
    /// through the identical token trajectory a first one already decided
    /// on (teacher forcing) -- and `logits_sink`, called every step with
    /// that step's own last-position logits (the same slice this loop
    /// already slices out of `evaluated` to sample from), so a caller can
    /// read off per-step logits without a parallel, uncached forward pass.
    /// Both are no-ops for [`Self::run_decode_loop`]'s own callers.
    /// `on_token` is [`decode_until_stop_or_budget`]'s own per-step callback,
    /// threaded straight through -- `&mut |_| ControlFlow::Continue(())` for
    /// every caller that does not need it, [`Self::generate_streaming`]'s
    /// real one for the one that does.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_decode_loop_observed(
        &self,
        prompt: &str,
        max_tokens: usize,
        serving_config: &ServingConfig,
        runtime: &mut BackendRuntime,
        token_override: Option<&[u32]>,
        logits_sink: &mut LogitsSink,
        on_token: &mut dyn FnMut(TokenEvent<'_>) -> ControlFlow<(), ()>,
    ) -> Result<(Vec<u32>, String, bool), InteropError> {
        let (generated_ids, text, stopped_by_eos, _prefix_state) = self
            .run_decode_loop_observed_seeded(
                prompt,
                max_tokens,
                serving_config,
                runtime,
                token_override,
                logits_sink,
                &mut NodeValuesSink::Discard,
                on_token,
                None,
                false,
            )?;
        Ok((generated_ids, text, stopped_by_eos))
    }

    /// [`Self::run_decode_loop_observed`]'s own body, plus a `seed`: `None`
    /// reproduces that method exactly (fresh [`LayerCacheState`] per layer,
    /// `cached_len` starting at 0, `prompt` tokenized WITH this vocab's own
    /// BOS/EOS policy); `Some(state)` resumes from a [`PrefixState`] a prior
    /// call returned instead -- `prompt` is then the SUFFIX text only,
    /// tokenized with NEITHER BOS nor EOS added (continuing the same
    /// sequence [`PrefixState::ids`] already opened), `layer_caches` starts
    /// from `state`'s own per-layer cache instead of [`LayerCache::new`],
    /// and `cached_len` starts from `state.cached_len` instead of `0`. The
    /// two-range decode loop below is BYTE-FOR-BYTE unchanged either way --
    /// this is the same primitive [`Self::generate_with_serving_config`]'s
    /// first step already runs (a multi-row forward over `next_ids` against
    /// `cached_len` rows of history), just given a nonzero `cached_len` and
    /// a non-full-prompt `next_ids` to start from -- so [`PrefixState`]
    /// itself is exactly the `(ids, layer_caches, cached_len)` triple this
    /// loop already threads through every step, exposed across calls rather
    /// than dropped at this function's own return.
    ///
    /// Always returns the FINAL [`PrefixState`] this call's own decoding
    /// left the cache in, alongside the usual generated-token triple --
    /// [`Self::run_decode_loop_observed`] discards it (nothing needs cross-
    /// call reuse there), [`Self::prefill_prefix`] is the one caller that
    /// keeps it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_decode_loop_observed_seeded(
        &self,
        prompt: &str,
        max_tokens: usize,
        serving_config: &ServingConfig,
        runtime: &mut BackendRuntime,
        token_override: Option<&[u32]>,
        logits_sink: &mut LogitsSink,
        node_values_sink: &mut NodeValuesSink,
        on_token: &mut dyn FnMut(TokenEvent<'_>) -> ControlFlow<(), ()>,
        seed: Option<PrefixState>,
        force_two_range: bool,
    ) -> Result<(Vec<u32>, String, bool, PrefixState), InteropError> {
        // Read unconditionally: the ONLY reader lives behind
        // `#[cfg(all(feature = "metal-output-placement", target_os =
        // "macos"))]` below, so a build without that cfg combination never
        // reads this parameter otherwise, and would warn on it as unused.
        let _ = force_two_range;
        let ids = if seed.is_some() {
            proxima_tokenizer::encode_with_bos_eos(prompt, &self.vocab, false, false)?
        } else {
            proxima_tokenizer::encode_with_bos_eos(
                prompt,
                &self.vocab,
                wants_bos(&self.vocab),
                self.vocab.add_eos_token().unwrap_or(false),
            )?
        };
        let seed_cached_len = seed.as_ref().map_or(0, PrefixState::len);
        let seed_ids: Vec<u32> = seed
            .as_ref()
            .map_or_else(Vec::new, |state| state.ids.clone());
        // The repetition-penalty filter's own window: prompt tokens included,
        // matching upstream (`tools/main/main.cpp:725` feeds prompt tokens
        // through the same `common_sampler_accept` generated tokens use), grown
        // by one id every decode step below. `sample_config`/`rng` are built
        // once and threaded through every step -- the same seeded
        // `fastrand::Rng` this workspace already uses for every other
        // deterministic-by-seed pipe, drawn from progressively rather than
        // reseeded per token, mirroring upstream's own one-`std::mt19937`-per-
        // sampler-chain lifetime (`proxima_tokenizer::sample`'s own doc).
        let mut token_history: Vec<u32> = {
            let mut history = seed_ids.clone();
            history.extend_from_slice(&ids);
            history
        };
        let repeat_window = serving_config.repeat_last_n.max(0) as usize;
        let sample_config = SamplingConfig {
            temperature: serving_config.temperature,
            top_k: serving_config.top_k,
            top_p: serving_config.top_p,
            min_p: serving_config.min_p,
            repeat_penalty: serving_config.repeat_penalty,
            frequency_penalty: serving_config.frequency_penalty,
            presence_penalty: serving_config.presence_penalty,
        };
        let mut rng = fastrand::Rng::with_seed(serving_config.seed);

        // Persistent device-resident KV: only reachable when this build was
        // compiled with `metal-output-placement`, this checkpoint built a
        // single-range program (`LoadedModel::single_range`'s own doc --
        // `None` for any mixture-of-experts or qwen35 checkpoint), this
        // call's own `ServingConfig` selected the Metal backend
        // (`runtime.is_metal()`), AND the caller did not ask to force the
        // two-range path (`force_two_range`). [`Self::prefill_prefix`]/
        // [`Self::generate_from_prefix`] always set `force_two_range: true`
        // -- the placed-kv path's own [`SingleRangeProgram`] never leaves a
        // host-side [`LayerCacheState`] to report as a [`PrefixState`], so
        // it is not eligible for either primitive regardless of whether
        // this checkpoint would otherwise take it. Every other caller
        // (ordinary `generate`/`generate_streaming`, `seed: None`) is
        // unaffected: CPU decode, any MoE checkpoint, and the qwen35 hybrid
        // path always fall through to the two-range `layer_roots` path
        // below, byte-for-byte unchanged.
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        if !force_two_range
            && runtime.is_metal()
            && let Some(single_range) = &self.single_range
        {
            let (generated_ids, text, stopped_by_eos) = self.run_decode_loop_placed_kv(
                single_range,
                ids.clone(),
                token_history.clone(),
                repeat_window,
                sample_config,
                rng.clone(),
                max_tokens,
                serving_config,
                runtime,
                token_override,
                logits_sink,
                on_token,
            )?;
            // No host-side `LayerCacheState` exists on this path -- the
            // KV cache never left the device (`SingleRangeProgram`'s own
            // doc) -- so there is nothing real to report here. Callers
            // that need a real [`PrefixState`] set `force_two_range: true`
            // and never reach this branch at all.
            return Ok((
                generated_ids,
                text,
                stopped_by_eos,
                PrefixState {
                    ids: Vec::new(),
                    layer_caches: Vec::new(),
                    cached_len: 0,
                },
            ));
        }

        // I11 scheduling level 2: phase scheduling, independent of the
        // admission and residency levels (`ServingConfig::phase_schedule`'s
        // own doc names this as the one site that consults it). Checked
        // here, at this driver's own entry, before any of the decode-loop
        // state below is built, rather than after -- a config this call
        // cannot serve fails before paying for cache-name derivation, SSM
        // buffer allocation, or sidecar scratch construction, not partway
        // through it.
        if !serving_config.phase_schedule.prefill_before_decode {
            return Err(InteropError::UnsupportedServingConfig(
                "phase_schedule.prefill_before_decode=false: interleaving prefill and decode \
                 steps across sequences is not implemented yet"
                    .into(),
            ));
        }
        let one_evaluation_prefill_requested = prefill_one_evaluation_requested(serving_config);

        // The program's own declared `Op::Input` leaves are the single
        // source of truth for which cache shape each layer needs fed --
        // never `self.layer_roots[layer]`'s own discriminant, which a
        // foreign `Architecture::bind` assembles by hand and can tag
        // inconsistently with the ops it actually emitted (see
        // `DeclaredCacheKind`'s own doc). The SAME derivation
        // [`Self::forward_node_values_on_backend`] calls, so a foreign
        // architecture's cache leaf names are never hard-coded twice.
        let (cache_names, layer_row_widths) = self.declared_layer_cache_names_and_widths()?;
        #[cfg(feature = "metal")]
        let qwen35_pre_gather_requested = qwen35moe_pre_gather_enabled(
            serving_config.qwen35moe_pre_gather,
            self.architecture_impl
                .is_some_and(|architecture| architecture.ffn_routing() == crate::architecture::FfnRouting::Routed),
        );
        #[cfg(feature = "metal")]
        let monolithic_high_mmap_requested = qwen35_pre_gather_requested
            && runtime.uses_gpu()
            && serving_config.qwen35moe_monolithic_high_mmap;
        #[cfg(not(feature = "metal"))]
        let monolithic_high_mmap_requested = false;
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let monolithic_all_low_requested = qwen35moe_monolithic_all_low_enabled(
            qwen35_pre_gather_requested,
            runtime.uses_gpu(),
            serving_config.qwen35moe_monolithic_all_low,
            0,
        );
        #[cfg(feature = "metal")]
        if qwen35_pre_gather_requested && !monolithic_high_mmap_requested {
            // A whole-checkpoint MTLBuffer makes mmap look cheap while still
            // charging every expert byte to Metal's working set.  Routed
            // execution supplies only selected experts through the typed
            // source table, so dense tensors must use bounded per-tensor
            // buffers instead of retaining the 24 GB checkpoint mapping.
            omega::backend::unregister_checkpoint_mapping(self.checkpoint_mapping);
        }
        let mut layer_caches: Vec<LayerCacheState> = match seed {
            Some(state) => state.layer_caches,
            None => self.fresh_layer_caches(&cache_names, &layer_row_widths),
        };
        // One [`KvPadScratch`] per layer, reused across every step of this
        // call -- only ever filled for a [`LayerCacheState::Attention`]
        // layer (the only cache shape `mistral_cached_forward_program_with_experts`
        // produces, `Qwen35LayerRoots`'s own doc), left empty and unread for
        // every `DenseAttention`/`Ssm` layer a qwen35 checkpoint carries.
        let mut kv_pad_scratch: Vec<KvPadScratch> = self
            .layer_roots
            .iter()
            .map(|_| KvPadScratch::new())
            .collect();
        // [`Qwen35DenseAttentionPadScratch`]'s own doc: the `Attention` arm's
        // padding above is not enough on its own -- a `DenseAttention` layer
        // shares the identical `Extent::Symbolic(1)` slot, so it needs the
        // same treatment or a bucketed `symbols[1]` reads past a shorter,
        // unpadded buffer on every qwen35 checkpoint.
        let mut qwen35_dense_pad_scratch: Vec<Qwen35DenseAttentionPadScratch> = self
            .layer_roots
            .iter()
            .map(|_| Qwen35DenseAttentionPadScratch::new())
            .collect();

        // ROW 531 invariant 2: recurrent state, conv history and dense-attention
        // KV roots are device-resident on Metal regardless of expert-residency
        // mode, so this gate is the architecture, never the pre-gather flag.
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        let qwen35moe_architecture = self
            .architecture_impl
            .is_some_and(|architecture| architecture.kv_cache_shape() == crate::architecture::KvCacheShape::Monolithic);
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        let dense_attention_placement_enabled = !monolithic_all_low_requested
            && qwen35_dense_attention_placement_enabled(
                qwen35moe_architecture,
                runtime.is_metal(),
                force_two_range,
                seed_cached_len,
                true,
            );
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        let dense_attention_input_nodes: Vec<Option<(NodeId, NodeId, NodeId, NodeId)>> =
            cache_names
                .iter()
                .map(|names| match names {
                    LayerCacheNames::DenseAttention {
                        k_first,
                        k_second,
                        k_pass,
                        v,
                    } if dense_attention_placement_enabled => Ok(Some((
                        find_input_node(&self.program, k_first)?,
                        find_input_node(&self.program, k_second)?,
                        find_input_node(&self.program, k_pass)?,
                        find_input_node(&self.program, v)?,
                    ))),
                    _ => Ok(None),
                })
                .collect::<Result<_, InteropError>>()?;
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        let dense_attention_positions = kv_extent(
            (seed_cached_len + ids.len() + max_tokens).min(serving_config.context_length as usize),
            serving_config.context_length as usize,
            serving_config.kv_bucket_tokens,
        );
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        let dense_attention_buffers: Vec<Option<Qwen35DenseAttentionBuffers>> = layer_row_widths
            .iter()
            .enumerate()
            .map(|(layer, widths)| -> Result<_, InteropError> {
                let LayerPadRowWidths::DenseAttention {
                    even_odd_row,
                    pass_row,
                    v_row,
                } = widths
                else {
                    return Ok(None);
                };
                if !dense_attention_placement_enabled {
                    return Ok(None);
                }
                let even_odd_byte_length = qwen35_dense_attention_placed_byte_length(
                    dense_attention_positions,
                    *even_odd_row,
                    layer,
                    "rotary key",
                )?;
                let pass_byte_length = qwen35_dense_attention_placed_byte_length(
                    dense_attention_positions,
                    *pass_row,
                    layer,
                    "pass-through key",
                )?;
                let value_byte_length = qwen35_dense_attention_placed_byte_length(
                    dense_attention_positions,
                    *v_row,
                    layer,
                    "value",
                )?;
                let buffers = Qwen35DenseAttentionBuffers {
                    k_first: allocate_placed_buffer(even_odd_byte_length)?,
                    k_second: allocate_placed_buffer(even_odd_byte_length)?,
                    k_pass: allocate_placed_buffer(pass_byte_length)?,
                    value: allocate_placed_buffer(value_byte_length)?,
                    even_odd_row_bytes: even_odd_row * core::mem::size_of::<f32>(),
                    pass_row_bytes: pass_row * core::mem::size_of::<f32>(),
                    value_row_bytes: v_row * core::mem::size_of::<f32>(),
                };
                omega::metal::zero_placed_buffer(&buffers.k_first, even_odd_byte_length);
                omega::metal::zero_placed_buffer(&buffers.k_second, even_odd_byte_length);
                omega::metal::zero_placed_buffer(&buffers.k_pass, pass_byte_length);
                omega::metal::zero_placed_buffer(&buffers.value, value_byte_length);
                Ok(Some(buffers))
            })
            .collect::<Result<_, _>>()?;

        // I11 scheduling level 3: per-layer expert residency, independent
        // of the pool-wide budget below (`ServingConfig::
        // expert_residency_schedule`'s own doc names this as the one site
        // that consults it).
        let per_layer_residency_budget = serving_config
            .expert_residency_schedule
            .per_layer_budget_bytes;
        if per_layer_residency_budget != 0 {
            return Err(InteropError::UnsupportedServingConfig(format!(
                "expert_residency_schedule.per_layer_budget_bytes={per_layer_residency_budget}: \
                 a per-layer residency pool separate from qwen35moe_residency_budget_bytes is \
                 not implemented yet"
            )));
        }

        // DynaExq observes the real routed expert ids produced by the graph.
        // The fixed matrix keeps policy state bounded and is enabled only
        // when the model owns a low-codec sidecar and the caller supplies a
        // high-precision residency budget.
        let residency_budget = serving_config.qwen35moe_residency_budget_bytes;
        let mut qwen35moe_residency = if self
            .architecture_impl
            .is_some_and(|architecture| architecture.ffn_routing() == crate::architecture::FfnRouting::Routed)
            && self.expert_sidecar.is_some()
            && residency_budget > 0
        {
            Some(crate::residency::ExpertResidency::<40, 256>::new(
                crate::residency::ResidencyConfig {
                    budget_bytes: residency_budget,
                    high_bytes_per_expert: self.expert_sidecar.as_ref().map_or(
                        0,
                        crate::expert_sidecar::MappedExpertSidecar::high_bytes_per_expert,
                    ),
                    ..crate::residency::ResidencyConfig::default()
                },
            ))
        } else {
            None
        };

        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        let ssm_placement_enabled =
            !monolithic_all_low_requested && qwen35moe_architecture && runtime.is_metal();
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        let ssm_placement_max_layer = None;
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        let ssm_state_input_nodes: Vec<Option<NodeId>> = cache_names
            .iter()
            .map(|names| match names {
                LayerCacheNames::Ssm { state, .. } => {
                    self.program
                        .iter()
                        .enumerate()
                        .find_map(|(index, op)| match op {
                            Op::Input {
                                name: Some(name), ..
                            } if name == state => Some(NodeId(index as u32)),
                            _ => None,
                        })
                }
                _ => None,
            })
            .collect();
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        let ssm_state_buffers: Vec<Option<(PlacedBuffer, PlacedBuffer)>> = layer_row_widths
            .iter()
            .map(
                |widths| -> Result<Option<(PlacedBuffer, PlacedBuffer)>, InteropError> {
                    if !ssm_placement_enabled {
                        return Ok(None);
                    }
                    match widths {
                        LayerPadRowWidths::Ssm { state_len, .. } => {
                            let byte_length = state_len * core::mem::size_of::<f32>();
                            let input = allocate_placed_buffer(byte_length)?;
                            let output = allocate_placed_buffer(byte_length)?;
                            Ok(Some((input, output)))
                        }
                        _ => Ok(None),
                    }
                },
            )
            .collect::<Result<_, _>>()?;
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        for (buffer, widths) in ssm_state_buffers.iter().zip(&layer_row_widths) {
            if let (Some((input, output)), LayerPadRowWidths::Ssm { state_len, .. }) =
                (buffer, widths)
            {
                let byte_length = state_len * core::mem::size_of::<f32>();
                omega::metal::zero_placed_buffer(input, byte_length);
                omega::metal::zero_placed_buffer(output, byte_length);
            }
        }

        // The caller's own knowledge of which named blocks are STATIC --
        // bound once in `LoadedModel::load` and never mutated again -- fixed
        // for this whole call, unlike `ids`/`eps`/`rope_cos`/`rope_sin` and
        // the KV cache's own blocks below, which change every step. This is
        // exactly the distinction `BackendRuntime::evaluate` hands to
        // `mark_resident` so the Metal driver's device-buffer cache can tell
        // "same name, same bytes" apart from "same name, new bytes" without
        // ever keying on name itself (`omega::metal::Plan::mark_resident`'s
        // own doc). Computed once, not per token: these names never change.
        let resident_names: BTreeSet<&str> = self.resident_names();

        let prompt_token_count = ids.len();
        let mut cached_len = seed_cached_len;
        let mut next_ids = ids.clone();
        let vocab_size = self.architecture.vocab as usize;
        // Reused across every step ([`Architecture::step_inputs`]'s own
        // doc) -- cleared, never reallocated from scratch, at the top of
        // each closure invocation below.
        let mut step_input_scratch: Vec<StepInput> = Vec::new();
        let gdn_prefill_names: Vec<String> = (0..self.architecture.block_count as usize)
            .map(|layer| alloc::format!("gdn_prefill.{layer}.delta_out"))
            .collect();
        let mut gdn_prefill_zero_scratch: Vec<f32> = Vec::new();
        let mut qwen35moe_pre_gather_plan: Option<Qwen35MoePreGatherPlan> = None;
        #[cfg(feature = "qwen35moe-expert-prefetch")]
        let mut qwen35moe_route_history =
            vec![Qwen35MoeRouteHistory::default(); self.architecture.block_count as usize];
        #[cfg(feature = "qwen35moe-expert-prefetch")]
        let qwen35moe_expert_prefetch_enabled =
            qwen35moe_expert_prefetch_requested(serving_config.qwen35moe_expert_prefetch);
        #[cfg(feature = "qwen35moe-expert-prefetch")]
        let mut qwen35moe_prefetch_prediction_count = 0usize;
        #[cfg(feature = "qwen35moe-expert-prefetch")]
        let mut qwen35moe_prefetch_hit_count = 0usize;
        #[cfg(feature = "qwen35moe-expert-prefetch")]
        let mut qwen35moe_prefetch_overfetch_count = 0usize;
        #[cfg(feature = "qwen35moe-expert-prefetch")]
        let mut qwen35moe_prefetch_advice_events = 0usize;
        #[cfg(feature = "qwen35moe-expert-prefetch")]
        let mut qwen35moe_prefetch_advised_bytes = 0u64;
        let mapped_window_capacity = self.expert_sidecar.as_ref().map_or([0; 3], |sidecar| {
            sidecar.mapped_window_capacity(self.architecture.expert_used_count as usize)
        });
        let mut sidecar_read_scratch =
            crate::expert_sidecar::ExpertSidecarReadScratch::with_high_cache_limit_and_window_capacity(
                usize::try_from(residency_budget).unwrap_or(0),
                mapped_window_capacity,
            )?;
        // ROW 427 named the reason a `single_position_step` architecture's
        // `new_count > 1` prefill used to split into `prompt_token_count`
        // one-position evaluations: the compiled decode program's own `s`
        // axis is `Extent::Symbolic(0)`, and
        // `proxima_tensor::spec::append_qwen35_ssm_mixer_with_taps_and_layout`'s
        // squeeze-reduce silently sums across positions for anything but a
        // literal `Extent::Static` axis. `self.qwen35moe_hparams`
        // (`Self::load`'s registry bind site) is the seam meant to fix it:
        // `qwen35moe_forward_program_at_width` builds a SECOND program with
        // `s` pinned to `Extent::Static(prompt_token_count)`, which reaches
        // that same builder's M>1 branch instead. That branch is proven
        // correct in isolation (`spec.rs`'s own oracle,
        // `qwen35_ssm_mixer_one_evaluation_matches_repeated_single_position_steps`,
        // exact agreement at a synthetic shape) but NOT YET on the real
        // `qwen3.6:35b-a3b` checkpoint at real GQA width: the real-checkpoint
        // oracle (`qwen35moe_one_evaluation_prefill_real_model`, this
        // module, below) currently measures the decoded text diverging and
        // the first GDN layer's own `block_output` already off by ~5x
        // relative to its own row norm -- a real numeric defect, not yet
        // root-caused. `PROXIMA_PREFILL_ONE_EVALUATION=1` is therefore the
        // OPT-IN escape hatch (default OFF, so every existing caller keeps
        // the proven split-loop behavior): built once here, never per step,
        // since `prompt_token_count > 1` is only ever true on the prompt's
        // own first step. `one_evaluation_prefill_requested` itself is
        // resolved once, at this call's own driver entry, above.
        // Sarathi/chunked-prefill (I9): split the prompt into
        // `serving_config.prefill_chunk_positions`-sized windows instead of
        // one whole-prompt evaluation, so peak activation memory is bounded
        // by the chunk width rather than `prompt_token_count`. `0` (the
        // field's own default) keeps the single whole-prompt chunk, byte-
        // for-byte the prior behavior. `cached_len` (below) already carries
        // KV state across these chunks the same way it carries it across
        // the one-position split loop.
        let one_evaluation_chunks: Vec<(usize, usize)> = if self.single_position_step
            && prompt_token_count > 1
            && one_evaluation_prefill_requested
        {
            let chunk_width = serving_config.prefill_chunk_positions;
            if chunk_width > 0 && chunk_width < prompt_token_count {
                let mut chunks = Vec::new();
                let mut offset = 0;
                while offset < prompt_token_count {
                    let width = chunk_width.min(prompt_token_count - offset);
                    chunks.push((offset, width));
                    offset += width;
                }
                chunks
            } else {
                alloc::vec![(0, prompt_token_count)]
            }
        } else {
            Vec::new()
        };
        let mut one_evaluation_prefill_programs: Vec<(
            usize,
            Vec<Op>,
            NodeId,
            Vec<Qwen35LayerRoots>,
        )> = Vec::new();
        if self.single_position_step
            && let Some(hparams) = self.qwen35moe_hparams.as_ref()
        {
            for &(_offset, width) in &one_evaluation_chunks {
                if one_evaluation_prefill_programs
                    .iter()
                    .any(|(built_width, ..)| *built_width == width)
                {
                    continue;
                }
                let (program, roots, layer_roots, _moe_sites, _diagnostics) =
                    crate::qwen35moe::qwen35moe_forward_program_at_width(
                        hparams,
                        Some(width as u32),
                    )
                    .map_err(|error| {
                        InteropError::PreGatherExecutionUnsupported {
                            architecture: String::from("qwen35moe"),
                            reason: alloc::format!(
                                "one-evaluation prefill program at width {width} failed to build: \
                                 {error}"
                            ),
                        }
                    })?;
                one_evaluation_prefill_programs.push((width, program, roots.logits, layer_roots));
            }
        }

        // Default-off greedy speculative decode (gemma4-only -- see
        // [`Self::speculative_verify_program`]'s own doc): read once, here,
        // outside the closure, matching [`prefill_one_evaluation_requested`]'s
        // own env-gate shape. `pending` is the queue-draining FSM's own
        // state -- popped from at the top of every closure call before any
        // forward runs, and pushed onto by the speculative verify branch
        // below whenever it accepts more than one token in a single pass.
        let speculative_enabled = std::env::var_os("PROXIMA_SPECULATIVE_DECODE").is_some();
        let speculative_k: usize = std::env::var("PROXIMA_SPECULATIVE_K")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(4);
        let mut pending: VecDeque<u32> = VecDeque::new();

        let decode_result = decode_until_stop_or_budget(
            &self.vocab,
            max_tokens,
            prompt_token_count,
            |_step| {
                #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                omega::set_capture_step(_step as u64);
                if let Some(queued) = pending.pop_front() {
                    if std::env::var_os("PROXIMA_DEBUG_SPECULATIVE").is_some() {
                        eprintln!("speculative_pending_pop step={_step}");
                    }
                    return Ok(queued);
                }
                // Speculative decode's draft half (`proxima_tokenizer::draft::
                // draft_ngram_lookup`, no second model): only attempted on a
                // genuine one-token decode step (`next_ids.len() == 1`,
                // excludes the prompt's own prefill at `_step == 0`) with a
                // real cache to draft against (`cached_len > 0`) and a
                // gemma4-only verify program bound at load time
                // (`Self::speculative_verify_program`'s own doc). Constants
                // `3`/`8` are `transformers`' own `PromptLookupCandidateGenerator`
                // defaults (`min_ngram_size`/`max_matching_ngram_size`).
                let speculative_draft: Vec<u32> = if speculative_enabled
                    && next_ids.len() == 1
                    && cached_len > 0
                    && self.speculative_verify_program.is_some()
                {
                    proxima_tokenizer::draft::draft_ngram_lookup(
                        &token_history,
                        speculative_k,
                        3,
                        8,
                    )
                } else {
                    Vec::new()
                };
                let speculative_step = !speculative_draft.is_empty();
                let speculative_ids: Vec<u32> = if speculative_step {
                    let mut ids = Vec::with_capacity(1 + speculative_draft.len());
                    ids.push(next_ids[0]);
                    ids.extend_from_slice(&speculative_draft);
                    ids
                } else {
                    Vec::new()
                };
                // ROW 130's own fix, built: every counter this step's
                // `evaluate_ms` decomposition reads is zeroed HERE, at step
                // start, and read back after `evaluate_ticks` below is computed
                // -- a single step's own cost, measured directly inside one
                // process, never inferred by differencing two independent
                // launches' cumulative-since-start counters (that differencing
                // is exact for the integer counts ROW 129 used it for, and NOT
                // for timings -- ROW 130's own postmortem on why it produced a
                // sub-bucket larger than its parent and a negative duration).
                // ROW 427 named the reason a `single_position_step`
                // architecture's `new_count > 1` prefill used to split into
                // `next_ids.len()` one-position evaluations: the compiled
                // decode program's own `s` axis is `Extent::Symbolic(0)`,
                // and `proxima_tensor::spec::append_qwen35_ssm_mixer_with_taps_and_layout`'s
                // squeeze-reduce silently sums across positions for
                // anything but a literal `Extent::Static` axis. `self.qwen35moe_hparams`
                // (`Self::load`'s registry bind site) is the seam that
                // fixes it: `qwen35moe_forward_program_at_width` builds a
                // SECOND program with `s` pinned to `Extent::Static(new_count)`,
                // which reaches that same builder's M>1 branch instead
                // (`spec.rs`'s own oracle,
                // `qwen35_ssm_mixer_one_evaluation_matches_repeated_single_position_steps`).
                // Default OFF (this module's own doc, above, on why): only
                // active when `PROXIMA_PREFILL_ONE_EVALUATION=1` was set
                // AND the alt program actually built.
                let one_evaluation_prefill = self.single_position_step
                    && next_ids.len() > 1
                    && one_evaluation_prefill_requested
                    && !one_evaluation_prefill_programs.is_empty();
                let split_prefill =
                    self.single_position_step && next_ids.len() > 1 && !one_evaluation_prefill;
                let batch_count = if one_evaluation_prefill {
                    one_evaluation_chunks.len()
                } else if split_prefill {
                    next_ids.len()
                } else {
                    1
                };
                let last_batch_index = batch_count - 1;

                let mut token_id: u32 = 0;
                for batch_index in 0..batch_count {
                    let ids_for_step: &[u32] = if speculative_step {
                        speculative_ids.as_slice()
                    } else if one_evaluation_prefill {
                        let (offset, width) = one_evaluation_chunks[batch_index];
                        &next_ids[offset..offset + width]
                    } else if split_prefill {
                        core::slice::from_ref(&next_ids[batch_index])
                    } else {
                        next_ids.as_slice()
                    };
                    // A SHARED borrow of the precomputed alt program(s)
                    // (`one_evaluation_prefill_programs`, built once before
                    // this closure, one per distinct chunk width -- never
                    // per step, since `next_ids.len() > 1` is only ever true
                    // on `_step == 0`) rather than a swap into `self`'s own
                    // fields: this closure only holds `&self`, and
                    // `self.resident_names()`'s own borrowed `BTreeSet<&str>`
                    // (computed once, above) is already live across every
                    // step, so nothing here may take `&mut self`.
                    // `speculative_step`'s own swap: identical shape to the
                    // `one_evaluation_prefill` swap above, into
                    // `self.speculative_verify_program` instead of a
                    // precomputed chunk-width program -- same weights, same
                    // per-layer cache leaves, only `logits_root` differs
                    // (every new position's own row, not just the last).
                    #[allow(clippy::expect_used)]
                    let (active_program, active_layer_roots, active_single_position_step) =
                        if speculative_step {
                            let (program, _logits_root, layer_roots) = self
                                .speculative_verify_program
                                .as_ref()
                                .expect("speculative_step only set true when this is Some");
                            (program, layer_roots, false)
                        } else if one_evaluation_prefill {
                            let (_offset, width) = one_evaluation_chunks[batch_index];
                            // every width in `one_evaluation_chunks` was built into
                            // `one_evaluation_prefill_programs` above, in the same loop.
                            let (_, program, _logits_root, layer_roots) =
                                one_evaluation_prefill_programs
                                    .iter()
                                    .find(|(built_width, ..)| *built_width == width)
                                    .expect(
                                        "chunk width program built above for every chunk width",
                                    );
                            (program, layer_roots, false)
                        } else {
                            (&self.program, &self.layer_roots, self.single_position_step)
                        };
                    #[allow(clippy::expect_used)]
                    let active_logits_root = if speculative_step {
                        self.speculative_verify_program
                            .as_ref()
                            .expect("speculative_step only set true when this is Some")
                            .1
                    } else if one_evaluation_prefill {
                        let (_offset, width) = one_evaluation_chunks[batch_index];
                        // every width in `one_evaluation_chunks` was built into
                        // `one_evaluation_prefill_programs` above, in the same loop.
                        one_evaluation_prefill_programs
                            .iter()
                            .find(|(built_width, ..)| *built_width == width)
                            .expect("chunk width program built above for every chunk width")
                            .2
                    } else {
                        self.logits_root
                    };
                    let is_last_step_batch = batch_index == last_batch_index;
                    #[cfg(feature = "instrument")]
                    proxima_tensor::instrument::reset_step();
                    #[cfg(feature = "instrument")]
                    let step_started = read_ticks();

                    let new_count = ids_for_step.len();
                    #[cfg(feature = "instrument")]
                    let apply_serving_config_started = read_ticks();
                    apply_serving_config(serving_config, cached_len + new_count)?;
                    #[cfg(feature = "instrument")]
                    let apply_serving_config_ticks = elapsed_ticks(apply_serving_config_started);

                    #[cfg(feature = "instrument")]
                    let build_position_inputs_started = read_ticks();
                    let inputs = build_position_inputs(
                        ids_for_step,
                        cached_len,
                        self.architecture.head_dim,
                        self.architecture.rope_freq_base,
                        self.architecture.rms_epsilon,
                        self.architecture_impl
                            .as_ref()
                            .and_then(|architecture| architecture.rope_freq_factors(&self.weights)),
                    );
                    #[cfg(feature = "instrument")]
                    let build_position_inputs_ticks = elapsed_ticks(build_position_inputs_started);

                    let mut named_blocks: Vec<(&str, QuantizedBlock)> = Vec::with_capacity(
                        self.weights.owned.len()
                            + self.weights.packed.len()
                            + self.weights.packed_owned.len()
                            + 3
                            + layer_caches.len() * 3,
                    );
                    #[cfg(feature = "instrument")]
                    let named_blocks_weights_started = read_ticks();
                    for (name, data) in &self.weights.owned {
                        named_blocks
                            .push((name.as_str(), QuantizedBlock::Float32(data.as_slice())));
                    }
                    for (name, block) in &self.weights.packed {
                        named_blocks.push((name.as_str(), *block));
                    }
                    for (name, bytes, kind) in &self.weights.packed_owned {
                        let block = crate::bind::as_block(*kind, bytes)
                            .ok_or(InteropError::UnsupportedCodec { codec: *kind })?;
                        named_blocks.push((name.as_str(), block));
                    }
                    // Rounds `cached_len` up to `ServingConfig::kv_bucket_tokens`
                    // (`kv_extent`'s own doc) -- `usize::MAX` in place of the
                    // placed-KV path's fixed buffer capacity: the two-range KV
                    // cache below is a growing `Vec`, not a preallocated
                    // device buffer, so there is no hard cap to clamp against.
                    let kv_bound_extent = kv_extent(
                        cached_len + new_count,
                        usize::MAX,
                        serving_config.kv_bucket_tokens,
                    );
                    // `mistral_cached_forward_program_with_experts`'s own
                    // `cached_len` `Op::Input` -- always present regardless of
                    // `ServingConfig::kv_bucket_tokens` (`proxima_tensor::bind::
                    // cached_attention_candidates`'s own doc on the runtime bound
                    // that reads it), so this scalar is fed on every step,
                    // bucketed or not.
                    let cached_len_scalar = [cached_len as f32];
                    // `mistral_cached_forward_program_with_experts_and_layer_taps`'s
                    // own `lm_head_row` `Op::Input` -- the last row of THIS
                    // step's `new_count` freshly-computed rows, host-supplied
                    // because the gather it feeds is a data-dependent index
                    // (`spec.rs`'s own doc on that leaf: an in-graph-computed
                    // index is a named `NotLowerable` gap on the typed
                    // evaluator, not a silently-guessed execution path).
                    let lm_head_row_scalar = [(new_count - 1) as f32];

                    // KV-cache HOST -> DEVICE traffic: every named block below is the
                    // FULL accumulated history (`LayerCache::append` only grows these,
                    // never truncates), so this is the full `cached_len`-sized array
                    // re-bound as a model input every single step -- not the
                    // `new_count`-sized increment. Measured directly as element
                    // counts read off the `Vec`s themselves (a size, not a timing),
                    // so it is exact and needs no instrumentation to be turned on.
                    #[cfg(feature = "instrument")]
                    let kv_cache_upload_elements: u64 = layer_caches
                        .iter()
                        .map(|cache| match cache {
                            LayerCacheState::Attention(cache) => {
                                (cache.k_even.len() + cache.k_odd.len() + cache.v.len()) as u64
                            }
                            LayerCacheState::DenseAttention(cache) => {
                                (cache.k_first.len()
                                    + cache.k_second.len()
                                    + cache.k_pass.len()
                                    + cache.v.len()) as u64
                            }
                            LayerCacheState::Ssm(cache) => {
                                (cache.conv_history.len() + cache.state.len()) as u64
                            }
                            LayerCacheState::SharedFromLayer => 0,
                        })
                        .sum();
                    #[cfg(feature = "instrument")]
                    let ssm_state_transfer_bytes: u64 = layer_caches
                        .iter()
                        .filter_map(|cache| match cache {
                            LayerCacheState::Ssm(cache) => {
                                Some((cache.state.len() * core::mem::size_of::<f32>()) as u64)
                            }
                            _ => None,
                        })
                        .sum();
                    #[cfg(feature = "instrument")]
                    let named_blocks_kv_started = read_ticks();
                    // `token_history` already carries exactly `cached_len +
                    // new_count` entries at this point (the same invariant
                    // `recent_tokens`'s repeat-penalty slice below relies on).
                    // The SAME assembly [`Self::forward_node_values_on_backend`]
                    // calls -- position/RoPE inputs, `cached_len`/`lm_head_row`,
                    // `Architecture::step_inputs`' own leaves, and every KV/SSM
                    // cache leaf -- so a foreign architecture's own leaf names
                    // are fed identically whether decoding or tapping one node.
                    let symbols = self.push_step_named_blocks(
                        &inputs,
                        &cached_len_scalar,
                        &lm_head_row_scalar,
                        &token_history,
                        cached_len,
                        new_count,
                        &cache_names,
                        &layer_caches,
                        &layer_row_widths,
                        kv_bound_extent,
                        &mut kv_pad_scratch,
                        &mut qwen35_dense_pad_scratch,
                        &mut step_input_scratch,
                        &mut named_blocks,
                        active_single_position_step,
                    )?;
                    if active_program.iter().any(|operation| {
                        operation.name().is_some_and(|name| {
                            gdn_prefill_names.iter().any(|candidate| candidate == name)
                        })
                    }) {
                        // The ordinary recurrent graph still owns a delta_out
                        // input.  It is the zero external residual when the
                        // host scan is not selected; leaving the leaf absent
                        // makes the proven single-position path fail before
                        // it can execute.
                        let shapes = proxima_tensor::shape::infer(active_program, &symbols)
                            .map_err(|error| InteropError::PreGatherExecutionUnsupported {
                                architecture: String::from("qwen35moe"),
                                reason: alloc::format!(
                                    "gdn prefill zero input shape inference failed: {error}"
                                ),
                            })?;
                        let element_count = self
                            .program
                            .iter()
                            .enumerate()
                            .find_map(|(index, operation)| {
                                matches!(operation, Op::Input { name: Some(name), .. }
                                    if name == &gdn_prefill_names[0])
                                .then(|| {
                                    shapes
                                        .of(NodeId(index as u32))
                                        .iter()
                                        .try_fold(1usize, |product, extent| {
                                            product.checked_mul(*extent as usize)
                                        })
                                })
                            })
                            .flatten()
                            .ok_or_else(|| InteropError::PreGatherExecutionUnsupported {
                                architecture: String::from("qwen35moe"),
                                reason: String::from("gdn prefill zero input shape is unavailable"),
                            })?;
                        gdn_prefill_zero_scratch.clear();
                        gdn_prefill_zero_scratch.resize(element_count, 0.0);
                        for name in &gdn_prefill_names {
                            named_blocks.push((
                                name.as_str(),
                                QuantizedBlock::Float32(&gdn_prefill_zero_scratch),
                            ));
                        }
                    }
                    #[cfg(feature = "instrument")]
                    let named_blocks_weights_ticks = elapsed_ticks(named_blocks_weights_started);
                    #[cfg(feature = "instrument")]
                    let named_blocks_kv_ticks = elapsed_ticks(named_blocks_kv_started);
                    // attribution slice (2026-09-22, OWNER_BRIEF_dominant_cost):
                    // roots vector build + program/logits-root selection between
                    // named_blocks_kv ending here and evaluate_started below --
                    // previously unattributed against wall_ms, see the followon
                    // attribution report.
                    #[cfg(feature = "instrument")]
                    let root_select_started = read_ticks();

                    let mut roots: Vec<NodeId> =
                        Vec::with_capacity(1 + active_layer_roots.len() * 3);
                    let monolithic_prefill_requested = qwen35moe_pre_gather_enabled(
                        serving_config.qwen35moe_pre_gather,
                        self.architecture_impl.is_some_and(|architecture| {
                            architecture.ffn_routing() == crate::architecture::FfnRouting::Routed
                        }),
                    ) && runtime.uses_gpu()
                        && _step == 0
                        && serving_config.qwen35moe_monolithic_all_low;
                    if step_batch_needs_logits(split_prefill, is_last_step_batch) {
                        roots.push(active_logits_root);
                    }
                    // attn_parity followon (2026-09-22, OWNER_BRIEF_gemma_head):
                    // `PROXIMA_HEAD_REPEATS=1|2|3` head-cost measurement knob.
                    // `lfm2_two_range_cached_forward_program_with_experts`
                    // (the builder gemma4's `CacheStrategy::TwoRange` production
                    // path calls) appends its `repeats - 1` duplicate head
                    // chains when the same env var is set at build time
                    // (`append_head`'s own doc), and returns their real
                    // `NodeId`s as `duplicate_head_roots`
                    // (`crate::architecture::BoundProgram::duplicate_head_roots`),
                    // threaded through `self.duplicate_head_roots` at load time --
                    // `NodeId(program.len() - offset)` is wrong for a chain (each
                    // duplicate head appends multiple ops, not one), so this reads
                    // the builder's own roots instead of reconstructing them.
                    // `prune_dead` only keeps a node reachable from a requested
                    // root, so pushing them here is what keeps the duplicate
                    // dispatches alive on the bound plan at all. Unconditional --
                    // this keeps `PROXIMA_HEAD_REPEATS` duplicate-head chains
                    // reachable from `prune_dead` on every build, not only one
                    // compiled with the unrelated `instrument` telemetry
                    // feature; `duplicate_head_roots` is `Vec::new()` (a no-op
                    // extend) on every checkpoint that never sets that env var.
                    if step_batch_needs_logits(split_prefill, is_last_step_batch) {
                        roots.extend(self.duplicate_head_roots.iter().copied());
                    }
                    roots.extend_from_slice(node_values_sink.nodes());
                    if monolithic_prefill_requested {
                        roots.extend(self.router_roots.iter().copied());
                    }
                    // `_layer` is read inside `#[cfg(all(feature =
                    // "metal-output-placement", target_os = "macos"))]`
                    // arms below -- genuinely unused under a plain `std`
                    // build, so both clippy's unused-enumerate-index lint
                    // and rustc's unused-variables lint fire on a config
                    // this loop body does not compile under.
                    #[allow(clippy::unused_enumerate_index)]
                    for (_layer, roots_for_layer) in active_layer_roots.iter().enumerate() {
                        match roots_for_layer {
                            Qwen35LayerRoots::Attention((even, odd, value)) => {
                                roots.push(*even);
                                roots.push(*odd);
                                roots.push(*value);
                            }
                            Qwen35LayerRoots::DenseAttention((first, second, pass, value)) => {
                                #[cfg(all(
                                    feature = "metal-output-placement",
                                    target_os = "macos"
                                ))]
                                let dense_attention_is_placed =
                                    dense_attention_buffers[_layer].as_ref().is_some();
                                #[cfg(not(all(
                                    feature = "metal-output-placement",
                                    target_os = "macos"
                                )))]
                                let dense_attention_is_placed = false;
                                if !dense_attention_is_placed {
                                    roots.push(*first);
                                    roots.push(*second);
                                    roots.push(*pass);
                                    roots.push(*value);
                                }
                            }
                            Qwen35LayerRoots::Ssm {
                                qkv_mixed,
                                state_out,
                            } => {
                                roots.push(*qkv_mixed);
                                #[cfg(all(
                                    feature = "metal-output-placement",
                                    target_os = "macos"
                                ))]
                                let state_is_placed = ssm_placement_enabled
                                    && ssm_placement_max_layer
                                        .is_none_or(|maximum| _layer <= maximum)
                                    && ssm_state_buffers[_layer].is_some();
                                #[cfg(not(all(
                                    feature = "metal-output-placement",
                                    target_os = "macos"
                                )))]
                                let state_is_placed = false;
                                // A placed state still has to be a requested
                                // graph output: otherwise `prune_dead` drops
                                // its producer before the output-placement
                                // binding can write the caller-owned buffer.
                                // The host readback path below remains gated
                                // by `state_is_placed`, so this requests the
                                // device write without restoring the copy.
                                let _ = state_is_placed;
                                roots.push(*state_out);
                            }
                            // gemma4 E2B's cross-layer shared-KV layer: no
                            // `K`/`V` root of its own to request -- its
                            // attention op already reads the donor layer's
                            // own already-requested nodes in-graph
                            // (`Qwen35LayerRoots::SharedFromLayer`'s own
                            // doc), so nothing is pushed here.
                            Qwen35LayerRoots::SharedFromLayer(_) => {}
                        }
                    }
                    if std::env::var_os("PROXIMA_DEBUG_GDN_BLOCK_OUTPUT").is_some()
                        && let Some(target_layer) = std::env::var("PROXIMA_DEBUG_GDN_LAYER")
                            .ok()
                            .and_then(|value| value.parse::<usize>().ok())
                        && let Some(diagnostic) = self.qwen35moe_layer_diagnostics.get(target_layer)
                    {
                        roots.push(diagnostic.block_output);
                    }
                    if std::env::var_os("PROXIMA_DEBUG_GDN_COMPARE").is_some()
                        && let Some(diagnostic) = self.qwen35moe_layer_diagnostics.first()
                        && let Some(taps) = diagnostic.ssm_taps.clone()
                    {
                        roots.extend([
                            taps.query_sequence,
                            taps.key_sequence,
                            taps.value_sequence,
                            taps.gate_sequence,
                            taps.beta_sequence,
                            taps.z_sequence,
                            taps.delta_out,
                            taps.gated_value,
                            taps.ssm_out_result,
                            diagnostic.block_input,
                            diagnostic.post_mixer_residual,
                            diagnostic.router_logits,
                        ]);
                    }
                    if std::env::var_os("PROXIMA_DEBUG_GDN_ALL_BLOCKS").is_some() {
                        roots.extend(
                            self.qwen35moe_layer_diagnostics
                                .iter()
                                .map(|diagnostic| diagnostic.block_output),
                        );
                    }
                    if std::env::var_os("PROXIMA_DEBUG_DENSE_DIGEST").is_some() {
                        for (layer, diagnostic) in
                            self.qwen35moe_layer_diagnostics.iter().enumerate()
                        {
                            if let Some(taps) = diagnostic.dense_attention_taps {
                                if let Some(operation) = active_program.get(taps.q_split.0 as usize)
                                {
                                    eprintln!(
                                        "dense_nodes layer={} normed={} q_split={} q_op={operation:?}",
                                        layer, taps.normed.0, taps.q_split.0,
                                    );
                                    if let proxima_tensor::Op::Reduce(reduce) = operation
                                        && let Some(product) =
                                            active_program.get(reduce.operand.0 as usize)
                                    {
                                        eprintln!(
                                            "dense_nodes_q_product layer={} node={} op={product:?}",
                                            layer, reduce.operand.0,
                                        );
                                        if layer == 3
                                            && std::env::var_os("PROXIMA_DEBUG_DENSE_GRAPH")
                                                .is_some()
                                            && let Some(Op::Elementwise { operands, .. }) =
                                                active_program.get(reduce.operand.0 as usize)
                                            && let Some((q_product, _)) = operands.first()
                                        {
                                            roots.push(*q_product);
                                            if let Some(Op::Reduce(qg_reduce)) =
                                                active_program.get(q_product.0 as usize)
                                            {
                                                roots.push(qg_reduce.operand);
                                                if let Some(Op::Elementwise { operands, .. }) =
                                                    active_program.get(qg_reduce.operand.0 as usize)
                                                {
                                                    for (operand, _) in operands {
                                                        roots.push(*operand);
                                                        if let Some(Op::Elementwise {
                                                            operands: weight_operands,
                                                            ..
                                                        }) =
                                                            active_program.get(operand.0 as usize)
                                                        {
                                                            roots.extend(
                                                                weight_operands
                                                                    .iter()
                                                                    .map(|(node, _)| *node),
                                                            );
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                roots.extend([
                                    diagnostic.block_input,
                                    taps.normed,
                                    taps.q_split,
                                    taps.k_normed,
                                    taps.v_new,
                                    taps.score_new,
                                    taps.attended,
                                    taps.gated_attended,
                                    taps.o_proj_out,
                                ]);
                            }
                        }
                    }
                    roots.sort_unstable_by_key(|node| node.0);
                    roots.dedup();
                    // gemma4's `KvCacheShape::Custom` excludes it from
                    // `LoadedModel::single_range` (`run_decode_loop_placed_kv`'s
                    // own doc, this function's own branch above at
                    // `self.single_range`), so its real decode step reaches
                    // THIS loop's `runtime.evaluate` below, never the
                    // placed-KV arm -- this is the parity probe's other
                    // call site, gated identically, with empty
                    // input/output placements since this arm's KV cache
                    // lives in `named_blocks`, not a device `PlacedBuffer`.
                    #[cfg(all(
                        feature = "metal",
                        feature = "metal-fuse-attn-decode",
                        feature = "metal-output-placement",
                        target_os = "macos"
                    ))]
                    if attn_fuse_parity_target_steps().contains(&_step) {
                        run_attn_fuse_parity_probe(
                            _step,
                            active_program,
                            &symbols,
                            &named_blocks,
                            &roots,
                            active_logits_root,
                            &resident_names,
                            &[],
                            runtime,
                        )?;
                    }
                    // Only requested when a routing observer is actually
                    // registered (`instrument::expert_observer`'s own doc): the
                    // CPU evaluator keeps every requested output's full lifetime
                    // alive, so a program with no observer never pays to hold
                    // these nodes live. `proxima_tensor::instrument` itself is
                    // `instrument`-feature-gated (`proxima-tensor/src/lib.rs`),
                    // so a build with `std` but not `instrument` never observes
                    // routing at all -- `observe_routing` is a compile-time
                    // `false` there, not a call into a module that does not
                    // exist.
                    #[cfg(feature = "instrument")]
                    let observe_routing = proxima_tensor::instrument::expert_observer().is_some()
                        && !self.moe_sites.0.is_empty();
                    #[cfg(not(feature = "instrument"))]
                    let observe_routing = false;
                    if observe_routing {
                        for site in &self.moe_sites.0 {
                            roots.extend(site.selected.iter().copied());
                            roots.extend(site.weights.iter().copied());
                        }
                    }

                    if let Some(name) =
                        missing_program_input(active_program, &named_blocks).filter(|name| {
                            !(qwen35moe_pre_gather_enabled(
                                serving_config.qwen35moe_pre_gather,
                                self.architecture_impl.is_some_and(|architecture| {
                                    architecture.ffn_routing()
                                        == crate::architecture::FfnRouting::Routed
                                }),
                            ) && (name.contains("_exps.weight")
                                || name.starts_with("gdn_prefill.")))
                        })
                    {
                        return Err(InteropError::MissingStepInput { name });
                    }

                    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                    let mut ssm_input_placements: Vec<(
                        NodeId,
                        &PlacedBuffer,
                        usize,
                    )> = Vec::new();
                    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                    let mut ssm_output_placements: Vec<(
                        NodeId,
                        &PlacedBuffer,
                        usize,
                    )> = Vec::new();
                    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                    for (layer, roots_for_layer) in active_layer_roots.iter().enumerate() {
                        if ssm_placement_enabled
                            && ssm_placement_max_layer.is_none_or(|maximum| layer <= maximum)
                            && let (
                                Qwen35LayerRoots::Ssm { state_out, .. },
                                Some(state_input),
                                Some((input_buffer, output_buffer)),
                            ) = (
                                roots_for_layer,
                                ssm_state_input_nodes[layer],
                                ssm_state_buffers[layer].as_ref(),
                            )
                        {
                            let (input_buffer, output_buffer) = if cached_len.is_multiple_of(2) {
                                (input_buffer, output_buffer)
                            } else {
                                (output_buffer, input_buffer)
                            };
                            ssm_input_placements.push((state_input, input_buffer, 0));
                            ssm_output_placements.push((*state_out, output_buffer, 0));
                            #[cfg(feature = "instrument")]
                            if layer == 0 {
                                debug!(
                                    step = (cached_len + batch_index) as u64,
                                    layer = layer as u64,
                                    state_in_node = state_input.0,
                                    state_out_node = state_out.0,
                                    state_in_placed_ptr =
                                        omega::placed_buffer_identity(input_buffer) as u64,
                                    state_out_placed_ptr =
                                        omega::placed_buffer_identity(output_buffer) as u64,
                                    state_in_first4 = ?read_placed_buffer_f32(input_buffer, 0, 4),
                                    "row 555: interop layer-0 ssm placement pre-dispatch"
                                );
                            }
                        }
                    }

                    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                    if std::env::var_os("PROXIMA_DEBUG_EVALUATOR_ROOTS").is_some() {
                        eprintln!(
                            "evaluator_roots step={} roots={:?} sink_nodes={:?} ssm_inputs={:?} ssm_outputs={:?}",
                            cached_len + batch_index,
                            roots,
                            node_values_sink.nodes(),
                            ssm_input_placements
                                .iter()
                                .map(|(node, _, _)| *node)
                                .collect::<Vec<_>>(),
                            ssm_output_placements
                                .iter()
                                .map(|(node, _, _)| *node)
                                .collect::<Vec<_>>(),
                        );
                    }

                    // The `StepGuard` (shadowing the lock below) closes this
                    // step on every exit -- normal return and an early `?`
                    // error alike -- so the source-snapshot boundary cannot
                    // remain open after a failed evaluation.
                    let mut expert_slab_guard = lock_expert_slab(&self.expert_slab);
                    let mut expert_slab_guard = expert_slab_guard.begin_step();

                    // the routed segment plan `qwen35moe_pre_gather_plan` builds
                    // is sliced from `self.program`'s own node ids
                    // (`self.qwen35moe_layer_diagnostics`, `self.logits_root`);
                    // `active_program` is a DIFFERENT graph during the
                    // one-evaluation prefill batch, so those node ids do not
                    // resolve against it -- ROW 591/592's `NodeId(6540)`
                    // "operand buffer missing" (a real weight input leaf
                    // reachable only in `active_program`'s own numbering).
                    let pre_gather = qwen35moe_pre_gather_enabled(
                        serving_config.qwen35moe_pre_gather,
                        self.architecture_impl.is_some_and(|architecture| {
                            architecture.ffn_routing() == crate::architecture::FfnRouting::Routed
                        }),
                    ) && !monolithic_high_mmap_requested
                        && !one_evaluation_prefill;
                    #[cfg(feature = "metal")]
                    let monolithic_all_low = qwen35moe_monolithic_all_low_enabled(
                        pre_gather,
                        runtime.uses_gpu(),
                        serving_config.qwen35moe_monolithic_all_low,
                        _step,
                    );
                    #[cfg(not(feature = "metal"))]
                    let monolithic_all_low = false;
                    if pre_gather {
                        expert_slab_guard.clear_selected_experts_for_step();
                    }

                    // This gather's own [`proxima_tensor::cpu::ExpertSource`]
                    // snapshot -- built after the residency boundary and read by
                    // `run_reduce_with_quantized_weights` under the SAME weight
                    // `NodeId` [`crate::bind::build_expert_slab`] bound it under,
                    // so a dense checkpoint's empty slab costs one `BTreeMap`
                    // miss per gathered reduce and changes nothing else.
                    let mut expert_entries_scratch: Vec<(
                        NodeId,
                        Vec<proxima_tensor::cpu::ExpertEntry<'_>>,
                    )> = Vec::new();
                    let expert_sources =
                        expert_slab_guard.sources_for_step(&mut expert_entries_scratch)?;
                    #[cfg(feature = "metal")]
                    let mut all_low_expert_scratch = Vec::new();
                    #[cfg(feature = "metal")]
                    let mut layer_window_expert_scratch = Vec::new();
                    #[cfg(feature = "metal")]
                    let all_low_expert_sources = if monolithic_all_low {
                        let sidecar = self.expert_sidecar.as_ref().ok_or_else(|| {
                            InteropError::PreGatherExecutionUnsupported {
                                architecture: String::from("qwen35moe"),
                                reason: String::from(
                                    "monolithic all-low execution requires an expert sidecar",
                                ),
                            }
                        })?;
                        if !sidecar.preserves_source_codecs() {
                            return Err(InteropError::PreGatherExecutionUnsupported {
                                architecture: String::from("qwen35moe"),
                                reason: String::from(
                                    "monolithic all-low execution requires a byte-preserving sidecar",
                                ),
                            });
                        }
                        if let Some(memory_limit) = serving_config.gpu_memory_limit_bytes {
                            let all_low_bytes = sidecar.all_low_bytes();
                            if all_low_bytes > memory_limit {
                                return Err(InteropError::PreGatherExecutionUnsupported {
                                    architecture: String::from("qwen35moe"),
                                    reason: alloc::format!(
                                        "monolithic all-low source table is {all_low_bytes} bytes, above the configured memory limit {memory_limit}"
                                    ),
                                });
                            }
                        }
                        Some(
                            expert_slab_guard
                                .all_low_sources_for_step(sidecar, &mut all_low_expert_scratch)?,
                        )
                    } else {
                        None
                    };
                    // The packed checkpoint views carry the expert input's
                    // shape and codec through plan resolution. They remain
                    // borrowed mmap ranges: Metal's expert-source executor
                    // excludes every substituted node from ordinary uploads,
                    // then binds only the routed payload and descriptor tables.
                    if std::env::var_os("PROXIMA_DEBUG_QWEN35_SEGMENTS").is_some() {
                        eprintln!(
                            "qwen35 pre_gather_enabled={pre_gather} sidecar={} gpu={}",
                            self.expert_sidecar.is_some(),
                            runtime.uses_gpu()
                        );
                    }
                    if pre_gather
                        && !monolithic_all_low
                        && qwen35moe_pre_gather_plan.as_ref().is_none_or(|plan| {
                            plan.symbols != symbols
                                || plan.gdn_backend != serving_config.gdn_prefill_backend
                                || plan.persistent_cuts != serving_config.qwen35moe_persistent_cuts
                        })
                    {
                        qwen35moe_pre_gather_plan = Some(self.qwen35moe_pre_gather_plan(
                            &symbols,
                            serving_config.gdn_prefill_backend,
                            serving_config.qwen35moe_persistent_cuts,
                        )?);
                    }
                    // Ordinary Metal full-graph execution keeps the named
                    // checkpoint stack. The explicit monolithic all-low arm
                    // is the only exception, and its byte-preserving and
                    // memory-limit guards run immediately above.
                    #[cfg(feature = "metal")]
                    let expert_source_substitutions = if monolithic_all_low {
                        all_low_expert_sources.as_ref()
                    } else if runtime.uses_gpu() {
                        None
                    } else {
                        Some(&expert_sources)
                    };
                    #[cfg(not(feature = "metal"))]
                    let expert_source_substitutions = Some(&expert_sources);

                    let current_sources = RefCell::new(CurrentExpertSources::new());
                    // Own the Arc-backed sidecar handle inside the callback so
                    // its borrow is not tied to the short `&self` call.
                    #[cfg(feature = "qwen35moe-expert-prefetch")]
                    let expert_sidecar_for_gather = self.expert_sidecar.clone();
                    let mut before_qwen35moe_gather =
                        |layer: usize,
                         position: u64,
                         routes: &[crate::residency::RoutedExpert],
                         expert_slab: &mut crate::expert_slab::ExpertSlab<'file>|
                         -> Result<(), InteropError> {
                            current_sources.borrow_mut().clear();
                            let mut selected_experts = [0_u32; 16];
                            if routes.len() > selected_experts.len() {
                                return Err(InteropError::PreGatherExecutionUnsupported {
                                    architecture: String::from("qwen35moe"),
                                    reason: String::from(
                                        "router selected more experts than the fixed staging bound",
                                    ),
                                });
                            }
                            for (index, route) in routes.iter().enumerate() {
                                selected_experts[index] = route.expert as u32;
                            }
                            if std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some() {
                                eprintln!(
                                    "qwen35 route layer={} position={} experts={:?}",
                                    layer,
                                    position,
                                    &selected_experts[..routes.len()]
                                );
                            }
                            #[cfg(feature = "qwen35moe-expert-prefetch")]
                            if qwen35moe_expert_prefetch_enabled {
                                if let Some(previous_history) = qwen35moe_route_history.get(layer) {
                                    for predicted in
                                        &previous_history.routes[..previous_history.len]
                                    {
                                        qwen35moe_prefetch_prediction_count += 1;
                                        if routes
                                            .iter()
                                            .any(|route| route.expert == predicted.expert)
                                        {
                                            qwen35moe_prefetch_hit_count += 1;
                                        } else {
                                            qwen35moe_prefetch_overfetch_count += 1;
                                        }
                                    }
                                }
                                if let (Some(policy), Some(sidecar)) = (
                                    qwen35moe_residency.as_ref(),
                                    expert_sidecar_for_gather.as_ref(),
                                ) {
                                    if let Some(next_history) =
                                        qwen35moe_route_history.get(layer.saturating_add(1))
                                    {
                                        let candidates = policy
                                            .prefetch_candidates::<16>(
                                                layer.saturating_add(1),
                                                &next_history.routes[..next_history.len],
                                                f32::NEG_INFINITY,
                                            )
                                            .map_err(|error| {
                                                InteropError::PreGatherExecutionUnsupported {
                                                    architecture: String::from("qwen35moe"),
                                                    reason: error.to_string(),
                                                }
                                            })?;
                                        let mut advised_bytes = 0_u64;
                                        for candidate in candidates.as_slice().iter().flatten() {
                                            let candidate_bytes =
                                                sidecar.advise_expert_low(candidate.address)?;
                                            advised_bytes =
                                                advised_bytes.saturating_add(candidate_bytes);
                                            qwen35moe_prefetch_advised_bytes =
                                                qwen35moe_prefetch_advised_bytes
                                                    .saturating_add(candidate_bytes);
                                        }
                                        qwen35moe_prefetch_advice_events += 1;
                                        if advised_bytes > 0
                                            && std::env::var_os("PROXIMA_DEBUG_EXPERT_PREFETCH")
                                                .is_some()
                                        {
                                            eprintln!(
                                                "qwen35 expert prefetch layer={} next_layer={} candidates={} advised_bytes={}",
                                                layer,
                                                layer.saturating_add(1),
                                                candidates.as_slice().len(),
                                                advised_bytes
                                            );
                                        }
                                    }
                                }
                            }
                            expert_slab
                                .add_selected_experts(layer, &selected_experts[..routes.len()]);
                            #[cfg(feature = "qwen35moe-expert-prefetch")]
                            if qwen35moe_expert_prefetch_enabled {
                                if let Some(history) = qwen35moe_route_history.get_mut(layer) {
                                    history.len = routes.len();
                                    history.routes[..routes.len()].copy_from_slice(routes);
                                }
                            }
                            if let Some(policy) = qwen35moe_residency.as_mut() {
                                // Capture the precision decision before the
                                // gather. Retained-set reconciliation happens
                                // once after the whole token, never between
                                // layers of this token.
                                for route in routes {
                                    let [decision] =
                                        policy.observe(position, layer, [*route]).map_err(
                                            |error| InteropError::PreGatherExecutionUnsupported {
                                                architecture: String::from("qwen35moe"),
                                                reason: error.to_string(),
                                            },
                                        )?;
                                    current_sources.borrow_mut().push(decision)?;
                                }
                            }
                            Ok(())
                        };

                    #[cfg(feature = "instrument")]
                    let root_select_ticks = elapsed_ticks(root_select_started);
                    #[cfg(feature = "instrument")]
                    let evaluate_started = read_ticks();
                    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                    #[cfg(feature = "instrument")]
                    let monolithic_profile_target = std::env::var("PROXIMA_METAL_OP_PROFILE_STEP")
                        .ok()
                        .and_then(|value| value.parse::<usize>().ok())
                        == Some(_step);
                    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                    #[cfg(not(feature = "instrument"))]
                    let monolithic_profile_target = false;
                    // `PROXIMA_METAL_DISPATCH_PROFILE_STEP`'s own reader for
                    // THIS step's two-range/non-placed-KV shape -- gemma4-E2B's
                    // real forward (`KvCacheShape::Custom` excludes it from
                    // `LoadedModel::single_range`, so it never reaches the
                    // placed-KV branch above, and it is not qwen35moe so it
                    // never reaches `monolithic_profile_target` either) falls
                    // through every arm above to plain `runtime.evaluate`
                    // below. Same default-off, one-env-var-per-step
                    // convention as `monolithic_profile_target` and the
                    // single-range path's own `PROXIMA_METAL_DISPATCH_PROFILE_STEP`
                    // reader (`Self::run_decode_loop_placed_kv`, this file
                    // around line 4134).
                    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                    #[cfg(feature = "instrument")]
                    let dispatch_profile_target =
                        std::env::var("PROXIMA_METAL_DISPATCH_PROFILE_STEP")
                            .ok()
                            .and_then(|value| value.parse::<usize>().ok())
                            == Some(_step);
                    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                    #[cfg(not(feature = "instrument"))]
                    let dispatch_profile_target = false;
                    // ROW 329: this step's encoder-split result, if
                    // `dispatch_profile_target` below actually takes it --
                    // declared here (not inside the match arm) so the
                    // post-match `metal_stage` snapshot a few lines down can
                    // read it out, same convention as
                    // `run_decode_loop_placed_kv`'s own `encoder_split_ns`.
                    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                    let mut encoder_split_ns: Option<(u64, u64)> = None;
                    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                    let evaluated = if monolithic_profile_target {
                        #[cfg(feature = "instrument")]
                        {
                            if let Some(expert_sources) = expert_source_substitutions {
                                let (evaluated, timings) = runtime
                                    .evaluate_op_timed_with_expert_sources(
                                        active_program,
                                        &symbols,
                                        &named_blocks,
                                        &roots,
                                        &resident_names,
                                        expert_sources,
                                    )?;
                                report_op_timings(_step, &timings, active_program);
                                evaluated
                            } else {
                                let (evaluated, timings, sampling_mode, _split_ns) = runtime
                                    .evaluate_dispatch_timed_with_placements(
                                        active_program,
                                        &symbols,
                                        &named_blocks,
                                        &roots,
                                        &resident_names,
                                        &ssm_input_placements,
                                        &ssm_output_placements,
                                    )?;
                                info!(
                                    step = _step as u64,
                                    sampling_mode, "dispatch_profile: qwen35moe full graph"
                                );
                                report_op_timings(_step, &timings, active_program);
                                evaluated
                            }
                        }
                        #[cfg(not(feature = "instrument"))]
                        unreachable!("the profiler is compiled out without instrumentation")
                    } else if pre_gather && !monolithic_all_low {
                        let pre_gather_plan =
                            qwen35moe_pre_gather_plan.as_ref().ok_or_else(|| {
                                InteropError::PreGatherExecutionUnsupported {
                                    architecture: String::from("qwen35moe"),
                                    reason: String::from(
                                        "the routed segment plan was not prepared",
                                    ),
                                }
                            })?;
                        self.evaluate_qwen35moe_pre_gather(
                            runtime,
                            pre_gather_plan,
                            &symbols,
                            // planning still needs the original expert input
                            // metadata; execution replaces those bindings
                            // with the selected source table before staging.
                            PreGatherContext {
                                named: &named_blocks,
                                outputs: &roots,
                                resident_names: &resident_names,
                                expert_slab: expert_slab_guard.as_slab_mut(),
                                sidecar_read_scratch: &mut sidecar_read_scratch,
                                current_sources: &current_sources,
                                position_offset: cached_len,
                                layer_window: serving_config.qwen35moe_layer_window,
                                marker: PhantomData,
                                #[cfg(feature = "metal")]
                                sidecar: self.expert_sidecar.as_ref(),
                                #[cfg(feature = "metal")]
                                all_low_expert_scratch: &mut layer_window_expert_scratch,
                                #[cfg(all(
                                    feature = "metal-output-placement",
                                    target_os = "macos"
                                ))]
                                ssm_placement: Some(&Qwen35SsmPlacement {
                                    input_nodes: &ssm_state_input_nodes,
                                    buffers: &ssm_state_buffers,
                                    maximum_layer: ssm_placement_max_layer,
                                    use_second_as_input: !cached_len.is_multiple_of(2),
                                }),
                                #[cfg(all(
                                    feature = "metal-output-placement",
                                    target_os = "macos"
                                ))]
                                dense_attention_placement: Some(&Qwen35DenseAttentionPlacement {
                                    input_nodes: &dense_attention_input_nodes,
                                    buffers: &dense_attention_buffers,
                                }),
                            },
                            &mut before_qwen35moe_gather,
                        )?
                    } else if use_metal_output_placements(
                        !ssm_input_placements.is_empty(),
                        monolithic_all_low,
                    ) {
                        runtime.evaluate_with_placements(
                            active_program,
                            &symbols,
                            &named_blocks,
                            &roots,
                            &resident_names,
                            &ssm_input_placements,
                            &ssm_output_placements,
                            expert_source_substitutions,
                        )?
                    } else if dispatch_profile_target {
                        #[cfg(feature = "instrument")]
                        {
                            // Empty placement slices: `plan_named_with_placed_inputs`
                            // with `placed_input_nodes: &[]` is byte-identical to
                            // `plan_named` (`omega/src/metal/placements_execute_named.rs:594`
                            // literally delegates to it with `&[]`), and
                            // `execute_plan_with_placements_dispatch_timed` with
                            // empty `input_placed`/`output_placed` maps takes the
                            // same `upload_block` path every node takes in
                            // `runtime.evaluate` below -- so this arm reuses the
                            // placed-KV timed executor to time the SAME unplaced
                            // shape gemma4's two-range path actually runs, never a
                            // fabricated placement.
                            let (evaluated, timings, sampling_mode, split_ns) = runtime
                                .evaluate_dispatch_timed_with_placements(
                                    active_program,
                                    &symbols,
                                    &named_blocks,
                                    &roots,
                                    &resident_names,
                                    &[],
                                    &[],
                                )?;
                            info!(
                                step = _step as u64,
                                sampling_mode, "dispatch_profile: two-range full graph"
                            );
                            report_op_timings(_step, &timings, active_program);
                            encoder_split_ns = split_ns;
                            evaluated
                        }
                        #[cfg(not(feature = "instrument"))]
                        unreachable!("the profiler is compiled out without instrumentation")
                    } else {
                        runtime.evaluate(
                            active_program,
                            &symbols,
                            &named_blocks,
                            &roots,
                            &resident_names,
                            expert_source_substitutions,
                        )?
                    };
                    #[cfg(all(
                        feature = "metal-output-placement",
                        feature = "instrument",
                        target_os = "macos"
                    ))]
                    if let Some((_, output_buffer, _)) = ssm_output_placements.first() {
                        debug!(
                            step = (cached_len + batch_index) as u64,
                            layer = 0_u64,
                            state_out_placed_ptr =
                                omega::placed_buffer_identity(output_buffer) as u64,
                            state_out_first4 = ?read_placed_buffer_f32(output_buffer, 0, 4),
                            "row 555: interop layer-0 ssm placement post-dispatch"
                        );
                    }
                    #[cfg(not(all(feature = "metal-output-placement", target_os = "macos")))]
                    let evaluated = if pre_gather && !monolithic_all_low {
                        let pre_gather_plan =
                            qwen35moe_pre_gather_plan.as_ref().ok_or_else(|| {
                                InteropError::PreGatherExecutionUnsupported {
                                    architecture: String::from("qwen35moe"),
                                    reason: String::from(
                                        "the routed segment plan was not prepared",
                                    ),
                                }
                            })?;
                        self.evaluate_qwen35moe_pre_gather(
                            runtime,
                            pre_gather_plan,
                            &symbols,
                            // planning needs the original expert descriptors;
                            // execution substitutes the selected source table.
                            PreGatherContext {
                                named: &named_blocks,
                                outputs: &roots,
                                resident_names: &resident_names,
                                expert_slab: expert_slab_guard.as_slab_mut(),
                                sidecar_read_scratch: &mut sidecar_read_scratch,
                                current_sources: &current_sources,
                                position_offset: cached_len,
                                layer_window: serving_config.qwen35moe_layer_window,
                                marker: PhantomData,
                                #[cfg(feature = "metal")]
                                sidecar: self.expert_sidecar.as_ref(),
                                #[cfg(feature = "metal")]
                                all_low_expert_scratch: &mut layer_window_expert_scratch,
                            },
                            &mut before_qwen35moe_gather,
                        )?
                    } else {
                        runtime.evaluate(
                            active_program,
                            &symbols,
                            &named_blocks,
                            &roots,
                            &resident_names,
                            expert_source_substitutions,
                        )?
                    };
                    if std::env::var_os("PROXIMA_DEBUG_GDN_ALL_DIGEST").is_some() && _step == 0 {
                        for (layer, diagnostic) in
                            self.qwen35moe_layer_diagnostics.iter().enumerate()
                        {
                            let mut digest_nodes = vec![
                                ("post_mixer", diagnostic.post_mixer_residual),
                                ("router", diagnostic.router_logits),
                            ];
                            if diagnostic.dense_attention_taps.is_some() {
                                digest_nodes.push(("mixer_output", diagnostic.mixer_output));
                                digest_nodes.push(("block_input", diagnostic.block_input));
                            }
                            for (label, node) in digest_nodes {
                                let Some((values, shape)) = evaluated.get(node) else {
                                    continue;
                                };
                                let row_count =
                                    usize::try_from(shape.first().copied().unwrap_or(1))
                                        .unwrap_or(1)
                                        .max(1);
                                let row_length = values.len() / row_count;
                                let row_start =
                                    row_length.saturating_mul(row_count.saturating_sub(1));
                                eprintln!(
                                    "gdn_all_digest mode={} layer={} label={} row={} first4={:?}",
                                    if new_count > 1 { "scan" } else { "cached" },
                                    layer,
                                    label,
                                    row_count.saturating_sub(1),
                                    values
                                        .get(row_start..row_start.saturating_add(row_length))
                                        .unwrap_or_default()
                                        .iter()
                                        .take(4)
                                        .copied()
                                        .collect::<Vec<_>>(),
                                );
                            }
                        }
                    }
                    if std::env::var_os("PROXIMA_DEBUG_GDN_BLOCK_DIGEST").is_some() && _step == 0 {
                        for (layer, diagnostic) in
                            self.qwen35moe_layer_diagnostics.iter().enumerate()
                        {
                            let Some((values, shape)) = evaluated.get(diagnostic.block_output)
                            else {
                                continue;
                            };
                            let row_count = usize::try_from(shape.first().copied().unwrap_or(1))
                                .unwrap_or(1)
                                .max(1);
                            let row_length = values.len() / row_count;
                            let row_start = row_length.saturating_mul(row_count.saturating_sub(1));
                            eprintln!(
                                "gdn_block_digest mode={} layer={} node={} row={} first4={:?}",
                                if new_count > 1 { "scan" } else { "cached" },
                                layer,
                                diagnostic.block_output.0,
                                row_count.saturating_sub(1),
                                values
                                    .get(row_start..row_start.saturating_add(row_length))
                                    .unwrap_or_default()
                                    .iter()
                                    .take(4)
                                    .copied()
                                    .collect::<Vec<_>>(),
                            );
                            if std::env::var_os("PROXIMA_DEBUG_GDN_COMPARE_ROWS").is_some()
                                && row_length > 0
                            {
                                for row_index in 0..row_count {
                                    let row_start = row_index * row_length;
                                    eprintln!(
                                        "gdn_block_row mode={} layer={} node={} row={} first4={:?}",
                                        if new_count > 1 { "scan" } else { "cached" },
                                        layer,
                                        diagnostic.block_output.0,
                                        row_index,
                                        values
                                            .get(row_start..row_start + row_length)
                                            .unwrap_or_default()
                                            .iter()
                                            .take(4)
                                            .copied()
                                            .collect::<Vec<_>>(),
                                    );
                                }
                            }
                        }
                    }
                    if std::env::var_os("PROXIMA_DEBUG_DENSE_DIGEST").is_some() && _step == 0 {
                        for (layer, diagnostic) in
                            self.qwen35moe_layer_diagnostics.iter().enumerate()
                        {
                            let Some(taps) = diagnostic.dense_attention_taps else {
                                continue;
                            };
                            if let Some(Op::Reduce(reduce)) =
                                active_program.get(taps.q_split.0 as usize)
                                && let Some(Op::Elementwise { operands, .. }) =
                                    active_program.get(reduce.operand.0 as usize)
                                && let Some((q_product, _)) = operands.first()
                                && let Some((values, shape)) = evaluated.get(*q_product)
                            {
                                eprintln!(
                                    "dense_digest mode={} layer={} label=qg_matmul shape={shape:?} first4={:?}",
                                    if new_count > 1 { "scan" } else { "cached" },
                                    layer,
                                    values.iter().take(4).copied().collect::<Vec<_>>(),
                                );
                            }
                            for (label, node) in [
                                ("block_input", diagnostic.block_input),
                                ("normed", taps.normed),
                                ("q_split", taps.q_split),
                                ("k_normed", taps.k_normed),
                                ("v_new", taps.v_new),
                                ("score_new", taps.score_new),
                                ("attended", taps.attended),
                                ("gated_attended", taps.gated_attended),
                                ("o_proj_out", taps.o_proj_out),
                            ] {
                                let Some((values, shape)) = evaluated.get(node) else {
                                    continue;
                                };
                                let row_count =
                                    usize::try_from(shape.first().copied().unwrap_or(1))
                                        .unwrap_or(1)
                                        .max(1);
                                let row_length = values.len() / row_count;
                                eprintln!(
                                    "dense_digest mode={} layer={} label={} shape={:?} first4={:?}",
                                    if new_count > 1 { "scan" } else { "cached" },
                                    layer,
                                    label,
                                    shape,
                                    values.iter().take(4).copied().collect::<Vec<_>>(),
                                );
                                if std::env::var_os("PROXIMA_DEBUG_GDN_COMPARE_ROWS").is_some()
                                    && row_length > 0
                                {
                                    for row_index in 0..row_count {
                                        let row_start = row_index * row_length;
                                        eprintln!(
                                            "dense_row mode={} layer={} label={} row={} first4={:?}",
                                            if new_count > 1 { "scan" } else { "cached" },
                                            layer,
                                            label,
                                            row_index,
                                            values
                                                .get(row_start..row_start + row_length)
                                                .unwrap_or_default()
                                                .iter()
                                                .take(4)
                                                .copied()
                                                .collect::<Vec<_>>(),
                                        );
                                    }
                                }
                            }
                            if layer == 3 {
                                for (label, node) in [
                                    ("qg_activation", NodeId(1256)),
                                    ("qg_product", NodeId(1257)),
                                    ("qg_weight_product", NodeId(1233)),
                                    ("qg_weight", NodeId(1231)),
                                    ("qg_weight_scale", NodeId(1232)),
                                ] {
                                    if let Some((values, shape)) = evaluated.get(node) {
                                        eprintln!(
                                            "dense_digest mode={} layer=3 label={} shape={shape:?} first4={:?}",
                                            if new_count > 1 { "scan" } else { "cached" },
                                            label,
                                            values.iter().take(4).copied().collect::<Vec<_>>(),
                                        );
                                    }
                                }
                            }
                        }
                    }
                    #[cfg(feature = "instrument")]
                    let evaluate_ticks = elapsed_ticks(evaluate_started);
                    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                    let metal_stage = metal_stage_totals();
                    // attribution slice (2026-09-22, OWNER_BRIEF_dominant_cost):
                    // brackets everything between evaluate() returning and the
                    // KV-append host memcpy starting -- expert-routing telemetry
                    // notify plus bookkeeping, previously folded into wall's
                    // unattributed remainder.
                    #[cfg(feature = "instrument")]
                    let post_evaluate_started = read_ticks();
                    // ROW 329's own two-range twin: the dispatch_profile_target
                    // arm above stores its `execute_plan_with_placements_dispatch_timed`
                    // split into `encoder_split_ns` instead of dropping it, so
                    // this step's `report_encoder_split` fires here exactly like
                    // `run_decode_loop_placed_kv`'s single-range arm does, using
                    // this same post-match `metal_stage` snapshot for `gpu_exec_ms`.
                    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                    if let Some(split_ns) = encoder_split_ns {
                        report_encoder_split(
                            _step,
                            split_ns,
                            ticks_to_nanos(metal_stage.gpu_exec_ticks),
                        );
                    }
                    if std::env::var_os("PROXIMA_DEBUG_PREFILL_BATCHES").is_some()
                        && _step == 0
                        && split_prefill
                    {
                        #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                        eprintln!(
                            "prefill_batch batch_index={} cached_len={} evaluate_ms={:.3} gpu_exec_calls={} gpu_exec_ms={:.3}",
                            batch_index,
                            cached_len,
                            ticks_to_nanos(evaluate_ticks) as f64 / 1e6,
                            metal_stage.gpu_exec_calls,
                            ticks_to_nanos(metal_stage.gpu_exec_ticks) as f64 / 1e6,
                        );
                        #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                        eprintln!(
                            "prefill_batch_stages batch_index={} prepare_ms={:.3} emit_ms={:.3} pipeline_lookup_ms={:.3} pipeline_misses={} pipeline_compile_ms={:.3} op_setup_ms={:.3} block_upload_ms={:.3} readback_ms={:.3}",
                            batch_index,
                            ticks_to_nanos(metal_stage.prepare_ticks) as f64 / 1e6,
                            ticks_to_nanos(metal_stage.emit_ticks) as f64 / 1e6,
                            ticks_to_nanos(metal_stage.pipeline_lookup_ticks) as f64 / 1e6,
                            metal_stage.pipeline_misses,
                            ticks_to_nanos(metal_stage.pipeline_compile_ticks) as f64 / 1e6,
                            ticks_to_nanos(metal_stage.op_setup_ticks) as f64 / 1e6,
                            ticks_to_nanos(metal_stage.block_upload_ticks) as f64 / 1e6,
                            ticks_to_nanos(metal_stage.readback_ticks) as f64 / 1e6,
                        );
                        #[cfg(not(all(feature = "instrument", feature = "metal", target_os = "macos")))]
                        eprintln!(
                            "prefill_batch batch_index={} cached_len={} evaluate_ms=unavailable",
                            batch_index, cached_len,
                        );
                    }
                    if std::env::var_os("PROXIMA_DEBUG_TOKEN_STAGES").is_some()
                        && !(_step == 0 && split_prefill)
                    {
                        #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                        eprintln!(
                            "token_stages step={} cached_len={} evaluate_ms={:.3} gpu_exec_ms={:.3} prepare_ms={:.3} pipeline_misses={} pipeline_compile_ms={:.3} op_setup_ms={:.3} block_upload_ms={:.3} readback_ms={:.3}",
                            _step,
                            cached_len,
                            ticks_to_nanos(evaluate_ticks) as f64 / 1e6,
                            ticks_to_nanos(metal_stage.gpu_exec_ticks) as f64 / 1e6,
                            ticks_to_nanos(metal_stage.prepare_ticks) as f64 / 1e6,
                            metal_stage.pipeline_misses,
                            ticks_to_nanos(metal_stage.pipeline_compile_ticks) as f64 / 1e6,
                            ticks_to_nanos(metal_stage.op_setup_ticks) as f64 / 1e6,
                            ticks_to_nanos(metal_stage.block_upload_ticks) as f64 / 1e6,
                            ticks_to_nanos(metal_stage.readback_ticks) as f64 / 1e6,
                        );
                        #[cfg(not(all(feature = "instrument", feature = "metal", target_os = "macos")))]
                        eprintln!(
                            "token_stages step={} cached_len={} evaluate_ms=unavailable",
                            _step, cached_len,
                        );
                    }

                    node_values_sink.observe(&evaluated)?;

                    if monolithic_all_low {
                        let mut router_scratch =
                            Vec::with_capacity(self.architecture.expert_used_count as usize);
                        for (layer, router_root) in self.router_roots.iter().copied().enumerate() {
                            let Some((logits, shape)) = evaluated.get(router_root) else {
                                continue;
                            };
                            visit_qwen35moe_router_selections(
                                layer,
                                cached_len,
                                RouterLogits {
                                    values: logits,
                                    shape,
                                },
                                RouterExpertCounts {
                                    expert_count: self.architecture.expert_count as usize,
                                    expert_used_count: self.architecture.expert_used_count as usize,
                                },
                                &mut router_scratch,
                                &mut |layer, position, routes| {
                                    if let Some(policy) = qwen35moe_residency.as_mut() {
                                        for route in routes {
                                            policy.observe(position, layer, [*route]).map_err(
                                                |error| {
                                                    InteropError::PreGatherExecutionUnsupported {
                                                        architecture: String::from("qwen35moe"),
                                                        reason: error.to_string(),
                                                    }
                                                },
                                            )?;
                                        }
                                    }
                                    Ok(())
                                },
                            )?;
                        }
                    }

                    if std::env::var_os("PROXIMA_DEBUG_GDN_BLOCK_OUTPUT").is_some()
                        && let Some(target_layer) = std::env::var("PROXIMA_DEBUG_GDN_LAYER")
                            .ok()
                            .and_then(|value| value.parse::<usize>().ok())
                        && let Some(diagnostic) = self.qwen35moe_layer_diagnostics.get(target_layer)
                        && std::env::var("PROXIMA_DEBUG_GDN_POSITION")
                            .ok()
                            .and_then(|value| value.parse::<usize>().ok())
                            .is_some_and(|target| cached_len + batch_index == target)
                        && let Some((values, shape)) = evaluated.get(diagnostic.block_output)
                    {
                        let row_width = shape
                            .first()
                            .and_then(|extent| usize::try_from(*extent).ok())
                            .and_then(|extent| values.len().checked_div(extent));
                        let row = row_width
                            .and_then(|width| values.get(..width))
                            .unwrap_or(values);
                        eprintln!(
                            "qwen35 monolithic block_output layer={} position={} node={:?} shape={:?} first4={:?}",
                            target_layer,
                            cached_len + batch_index,
                            diagnostic.block_output,
                            shape,
                            row.iter().take(4).copied().collect::<Vec<_>>(),
                        );
                    }

                    if std::env::var_os("PROXIMA_DEBUG_GDN_COMPARE").is_some()
                        && _step == 0
                        && let Some(diagnostic) = self.qwen35moe_layer_diagnostics.first()
                        && let Some(taps) = diagnostic.ssm_taps.clone()
                    {
                        for (label, node) in [
                            ("query_sequence", taps.query_sequence),
                            ("key_sequence", taps.key_sequence),
                            ("value_sequence", taps.value_sequence),
                            ("gate_sequence", taps.gate_sequence),
                            ("beta_sequence", taps.beta_sequence),
                            ("z_sequence", taps.z_sequence),
                            ("delta_out", taps.delta_out),
                            ("gated_value", taps.gated_value),
                            ("ssm_out_result", taps.ssm_out_result),
                            ("block_input", diagnostic.block_input),
                            ("post_mixer", diagnostic.post_mixer_residual),
                            ("router", diagnostic.router_logits),
                        ] {
                            if let Some((values, shape)) = evaluated.get(node) {
                                let first_row_len = if new_count > 1 {
                                    values.len() / new_count
                                } else {
                                    values.len()
                                };
                                let row_dump =
                                    std::env::var_os("PROXIMA_DEBUG_GDN_COMPARE_ROWS").is_some();
                                eprintln!(
                                    "gdn_compare mode={} node={} id={} shape={:?} first_row={:?}",
                                    if new_count > 1 { "scan" } else { "cached" },
                                    label,
                                    node.0,
                                    shape,
                                    &values[..values.len().min(first_row_len)],
                                );
                                if row_dump && first_row_len > 0 {
                                    for row_index in 0..new_count {
                                        let start = row_index.saturating_mul(first_row_len);
                                        let end = start.saturating_add(first_row_len);
                                        if let Some(row) = values.get(start..end) {
                                            eprintln!(
                                                "gdn_compare_row mode={} node={} label={} row={} values={row:?}",
                                                if new_count > 1 { "scan" } else { "cached" },
                                                node.0,
                                                label,
                                                row_index,
                                            );
                                        }
                                    }
                                }
                                if std::env::var_os("PROXIMA_DEBUG_GDN_ROW_DIGEST").is_some()
                                    && first_row_len > 0
                                {
                                    for row_index in 0..new_count {
                                        let start = row_index.saturating_mul(first_row_len);
                                        let end = start.saturating_add(first_row_len);
                                        if let Some(row) = values.get(start..end) {
                                            let digest =
                                                row.iter().take(4).copied().collect::<Vec<_>>();
                                            eprintln!(
                                                "gdn_compare_digest mode={} node={} label={} row={} first4={digest:?}",
                                                if new_count > 1 { "scan" } else { "cached" },
                                                node.0,
                                                label,
                                                row_index,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if std::env::var_os("PROXIMA_DEBUG_GDN_ROUTER_LAYERS").is_some() && _step == 0 {
                        for (layer, router_node) in self.router_roots.iter().copied().enumerate() {
                            let Some((values, shape)) = evaluated.get(router_node) else {
                                continue;
                            };
                            let row_width = usize::try_from(
                                shape
                                    .last()
                                    .copied()
                                    .unwrap_or(self.architecture.expert_count as u64),
                            )
                            .unwrap_or(0);
                            let rows = values.len().checked_div(row_width).unwrap_or(0);
                            let row = values.get(..row_width).unwrap_or(values);
                            let top = row
                                .iter()
                                .copied()
                                .enumerate()
                                .max_by(|left, right| left.1.total_cmp(&right.1));
                            eprintln!(
                                "gdn_router_layer mode={} layer={} node={} shape={shape:?} rows={} first_top={top:?}",
                                if new_count > 1 { "scan" } else { "cached" },
                                layer,
                                router_node.0,
                                rows,
                            );
                        }
                    }

                    // One `ExpertRouting` event per layer per new position --
                    // `proxima_tensor::instrument::ExpertObserver`'s own doc on
                    // why this is the decode loop's job, not the kernel's:
                    // `evaluated.get` reads back exactly the extra outputs
                    // `observe_routing` requested above, never the kernel's own
                    // per-position gather.
                    #[cfg(feature = "instrument")]
                    if observe_routing {
                        for site in &self.moe_sites.0 {
                            let weight_total_node =
                                site.weights.last().copied().unwrap_or(active_logits_root);
                            let Some((weight_total, _)) = evaluated.get(weight_total_node) else {
                                continue;
                            };
                            for local in 0..new_count {
                                let experts: Vec<u32> = site
                                    .selected
                                    .iter()
                                    .filter_map(|node| evaluated.get(*node))
                                    .filter_map(|(values, _)| values.get(local).copied())
                                    .map(|value| value as u32)
                                    .collect();
                                let Some(&total) = weight_total.get(local) else {
                                    continue;
                                };
                                let weights: Vec<f32> = site
                                    .weights
                                    .iter()
                                    .take(site.weights.len().saturating_sub(1))
                                    .filter_map(|node| evaluated.get(*node))
                                    .filter_map(|(values, _)| values.get(local).copied())
                                    .map(|value| value / total)
                                    .collect();
                                if experts.len() != weights.len() {
                                    continue;
                                }
                                let event = proxima_tensor::instrument::ExpertRouting {
                                    layer: site.layer,
                                    position: (cached_len + local) as u64,
                                    experts: &experts,
                                    weights: &weights,
                                };
                                proxima_tensor::instrument::notify_expert_routed(&event);
                            }
                        }
                    }

                    // KV-cache DEVICE -> HOST readback + host append: unlike the
                    // upload above, `evaluated.get(*even)` etc. is this step's own
                    // `new_count`-sized OUTPUT increment (what the forward computed
                    // for the newly-added positions), which `LayerCache::append`
                    // then extends onto the growing history -- so this side is
                    // expected to stay FLAT across tokens where the upload side
                    // grows. `layer_cache_append` ticks/bytes below are the pure
                    // host `extend_from_slice` memcpy cost, distinct from the GPU
                    // readback `metal_stage_totals` already reports.
                    #[cfg(feature = "instrument")]
                    let post_evaluate_ticks = elapsed_ticks(post_evaluate_started);
                    #[cfg(feature = "instrument")]
                    let layer_cache_append_started = read_ticks();
                    #[cfg(feature = "instrument")]
                    let mut layer_cache_append_elements: u64 = 0;
                    // History-carry bisection step 3 (diag/qwen35moe-history-carry):
                    // before-state for layers 0 (GDN) and 3 (this checkpoint's
                    // first attention layer, 3-GDN-to-1-attention interleave) --
                    // paired with the AFTER checksum below to prove whether this
                    // step's cache append actually mutated either layer's state.
                    // `.get` rather than a literal index: a foreign `Architecture`
                    // with an empty `layer_roots` (no cache leaves at all) has an
                    // empty `layer_caches` too, and this diagnostic must degrade
                    // to "nothing to report" rather than index out of bounds.
                    #[cfg(feature = "instrument")]
                    let (layer0_before_len, layer0_before_checksum) =
                        layer_caches.first().map_or((0, 0.0), layer_cache_checksum);
                    #[cfg(feature = "instrument")]
                    let (layer3_before_len, layer3_before_checksum) =
                        layer_caches.get(3).map_or((0, 0.0), layer_cache_checksum);
                    for (layer, roots_for_layer) in active_layer_roots.iter().enumerate() {
                        match (roots_for_layer, &mut layer_caches[layer]) {
                            (
                                Qwen35LayerRoots::Attention((even, odd, value)),
                                LayerCacheState::Attention(cache),
                            ) => {
                                let (even_data, _) = evaluated
                                    .get(*even)
                                    .ok_or(InteropError::MissingEvaluatedNode { node: *even })?;
                                let (odd_data, _) = evaluated
                                    .get(*odd)
                                    .ok_or(InteropError::MissingEvaluatedNode { node: *odd })?;
                                let (value_data, _) = evaluated
                                    .get(*value)
                                    .ok_or(InteropError::MissingEvaluatedNode { node: *value })?;
                                #[cfg(feature = "instrument")]
                                {
                                    layer_cache_append_elements +=
                                        (even_data.len() + odd_data.len() + value_data.len())
                                            as u64;
                                }
                                cache.append(even_data, odd_data, value_data);
                            }
                            (
                                Qwen35LayerRoots::DenseAttention((first, second, pass, value)),
                                LayerCacheState::DenseAttention(cache),
                            ) => {
                                #[cfg(all(
                                    feature = "metal-output-placement",
                                    target_os = "macos"
                                ))]
                                let dense_attention_is_placed =
                                    dense_attention_buffers[layer].as_ref().is_some();
                                #[cfg(not(all(
                                    feature = "metal-output-placement",
                                    target_os = "macos"
                                )))]
                                let dense_attention_is_placed = false;
                                if dense_attention_is_placed {
                                    continue;
                                }
                                let (first_data, _) = evaluated
                                    .get(*first)
                                    .ok_or(InteropError::MissingEvaluatedNode { node: *first })?;
                                let (second_data, _) = evaluated
                                    .get(*second)
                                    .ok_or(InteropError::MissingEvaluatedNode { node: *second })?;
                                let (pass_data, _) = evaluated
                                    .get(*pass)
                                    .ok_or(InteropError::MissingEvaluatedNode { node: *pass })?;
                                let (value_data, _) = evaluated
                                    .get(*value)
                                    .ok_or(InteropError::MissingEvaluatedNode { node: *value })?;
                                #[cfg(feature = "instrument")]
                                {
                                    layer_cache_append_elements += (first_data.len()
                                        + second_data.len()
                                        + pass_data.len()
                                        + value_data.len())
                                        as u64;
                                }
                                cache.append(first_data, second_data, pass_data, value_data);
                            }
                            (
                                Qwen35LayerRoots::Ssm {
                                    qkv_mixed,
                                    state_out,
                                },
                                LayerCacheState::Ssm(cache),
                            ) => {
                                let (qkv_mixed_data, _) = evaluated.get(*qkv_mixed).ok_or(
                                    InteropError::MissingEvaluatedNode { node: *qkv_mixed },
                                )?;
                                #[cfg(all(
                                    feature = "metal-output-placement",
                                    target_os = "macos"
                                ))]
                                let state_is_placed = ssm_output_placements
                                    .iter()
                                    .any(|(node, _, _)| node == state_out);
                                #[cfg(not(all(
                                    feature = "metal-output-placement",
                                    target_os = "macos"
                                )))]
                                let state_is_placed = false;
                                let state_out_data = if state_is_placed {
                                    None
                                } else {
                                    Some(
                                        evaluated
                                            .get(*state_out)
                                            .ok_or(InteropError::MissingEvaluatedNode {
                                                node: *state_out,
                                            })?
                                            .0,
                                    )
                                };
                                #[cfg(feature = "instrument")]
                                {
                                    layer_cache_append_elements += qkv_mixed_data.len() as u64
                                        + state_out_data.map_or(0, |data| data.len() as u64);
                                    debug!(
                                        layer = layer as u64,
                                        qkv_mixed_elements = qkv_mixed_data.len() as u64,
                                        state_elements =
                                            state_out_data.map_or(0, |data| data.len() as u64),
                                        state_bytes = state_out_data
                                            .map_or(0, |data| core::mem::size_of_val(data) as u64),
                                        state_is_placed,
                                        "ssm_state_host_transfer: recurrent output placement"
                                    );
                                }
                                // `layer_row_widths[layer]`'s own `Ssm` arm --
                                // the program's own declared
                                // `ssm_cache.{layer}.conv_history` shape, the
                                // SAME source [`fresh_layer_caches`] sized this
                                // cache's initial window from, never
                                // `Architecture::step_state` (that hook's `None`
                                // default is exactly the real-world defect this
                                // read used to reproduce on a foreign
                                // architecture).
                                let conv_history_len = match &layer_row_widths[layer] {
                                    LayerPadRowWidths::Ssm {
                                        conv_history_len, ..
                                    } => *conv_history_len,
                                    _ => unreachable!(
                                        "layer_row_widths built from the same layer_roots, in lockstep"
                                    ),
                                };
                                if let Some(state_out_data) = state_out_data {
                                    cache.advance(qkv_mixed_data, state_out_data, conv_history_len);
                                } else {
                                    cache.advance_conv_history(qkv_mixed_data, conv_history_len);
                                }
                            }
                            // gemma4 E2B's cross-layer shared-KV layer: no
                            // state of its own to append to -- its `K`/`V`
                            // were never requested as separate roots for
                            // this layer index (the earlier
                            // `roots.extend` loop's own `SharedFromLayer`
                            // no-op arm), so there is nothing here to fold
                            // in either.
                            (Qwen35LayerRoots::SharedFromLayer(_), LayerCacheState::SharedFromLayer) => {}
                            _ => unreachable!(
                                "layer_roots/layer_caches built from the same layer_roots, in lockstep"
                            ),
                        }
                    }
                    #[cfg(feature = "instrument")]
                    let layer_cache_append_ticks = elapsed_ticks(layer_cache_append_started);
                    // attribution slice (2026-09-22, OWNER_BRIEF_dominant_cost):
                    // brackets the readout-branch selection + eager debug! field
                    // evaluation (non_finite_count scans vocab_size unconditionally
                    // -- tracing macros evaluate field expressions before the
                    // level check) between append ending and logits_hash starting.
                    #[cfg(feature = "instrument")]
                    let pre_logits_started = read_ticks();
                    let cached_len_before_step = cached_len;
                    // attribution slice 2 (2026-09-22, OWNER_BRIEF_dominant_cost):
                    // splits pre_logits into the cache-checksum debug! block,
                    // the logits-fetch + shape-check debug! block, and the
                    // unconditional argmax debug! block, to find which
                    // sub-interval carries the 1.26ms the scan-disabled race
                    // left unattributed.
                    #[cfg(feature = "instrument")]
                    let checksum_started = read_ticks();
                    #[cfg(feature = "instrument")]
                    {
                        let (layer0_after_len, layer0_after_checksum) =
                            layer_caches.first().map_or((0, 0.0), layer_cache_checksum);
                        let (layer3_after_len, layer3_after_checksum) =
                            layer_caches.get(3).map_or((0, 0.0), layer_cache_checksum);
                        debug!(
                            step = _step as u64,
                            batch_index = batch_index as u64,
                            token_fed = ids_for_step[0],
                            cached_len_before = cached_len_before_step as u64,
                            layer0_len_before = layer0_before_len as u64,
                            layer0_len_after = layer0_after_len as u64,
                            layer0_checksum_before = layer0_before_checksum,
                            layer0_checksum_after = layer0_after_checksum,
                            layer3_len_before = layer3_before_len as u64,
                            layer3_len_after = layer3_after_len as u64,
                            layer3_checksum_before = layer3_before_checksum,
                            layer3_checksum_after = layer3_after_checksum,
                            "decode_loop_step_trace: layer 0/3 cache state before/after this step's append"
                        );
                    }
                    #[cfg(feature = "instrument")]
                    let checksum_ticks = elapsed_ticks(checksum_started);
                    // Cacheless (`active_layer_roots.is_empty()`) architectures have
                    // no `LayerCache` to advance -- `cached_len` stays 0 so next
                    // step's `build_position_inputs`/`apply_serving_config` above
                    // compute positions against the FULL re-fed sequence starting
                    // at 0, matching the `next_ids` re-prefill below, rather than
                    // an offset into a cache that was never populated.
                    // Speculative decode's own verify batch appended K+1
                    // positions' worth of K/V above, but only `verified.
                    // accepted + 1` of them survive (`LayerCache::truncate`'s
                    // own doc) -- the readout branch below computes that
                    // count and both advances `cached_len` and rewinds the
                    // cache to match it, so this generic advance is skipped
                    // here rather than corrected twice.
                    if !active_layer_roots.is_empty() && !speculative_step {
                        cached_len += new_count;
                    }

                    // Everything below samples a token off THIS batch's logits.
                    // For a `single_position_step` architecture's expanded
                    // prefill (`step_batches` above), every batch except the
                    // last is a known prompt token, not a sampled one -- only
                    // the cache-append above needs to run for it. Running this
                    // tail on every batch would draw from `rng` once per prompt
                    // position instead of once per generated token, diverging
                    // from the sequential single-position oracle the ROW 427
                    // tests compare against (`RE-VERIFY: rng` in that doc's own
                    // row). Skipping it here is what makes `next_ids`'s LAST
                    // batch's logits the ones `decode_until_stop_or_budget`
                    // actually samples, exactly like a `new_count == 1` decode
                    // step always has.
                    // Speculative decode's own readout, parallel to the
                    // ordinary single-row branch below (`Verified`'s own
                    // doc): `active_logits_root` here is
                    // `speculative_verify_program`'s all-positions gather
                    // (`new_count` rows of `vocab_size`, not one), so the
                    // single-row guard just below does not apply -- this
                    // branch verifies the whole drafted span in one pass
                    // and returns before reaching it.
                    if speculative_step {
                        let (logits, _shape) = evaluated.get(active_logits_root).ok_or(
                            InteropError::MissingEvaluatedNode {
                                node: active_logits_root,
                            },
                        )?;
                        if logits.len() != new_count * vocab_size {
                            return Err(InteropError::LogitsShapeMismatch {
                                expected_rows: new_count,
                                found_rows: logits.len() / vocab_size,
                                vocab: vocab_size,
                            });
                        }
                        let rows: Vec<&[f32]> = logits.chunks_exact(vocab_size).collect();
                        let verified = proxima_tokenizer::draft::verify_greedy(
                            &rows,
                            &speculative_draft,
                        )
                        .ok_or(InteropError::EmptyLogits)?;
                        let mut emitted: Vec<u32> =
                            speculative_draft[..verified.accepted].to_vec();
                        emitted.push(verified.next);
                        if std::env::var_os("PROXIMA_DEBUG_SPECULATIVE").is_some() {
                            eprintln!(
                                "speculative_verify step={_step} draft_len={} accepted={} emitted={}",
                                speculative_draft.len(),
                                verified.accepted,
                                emitted.len()
                            );
                        }

                        // The append loop above wrote `new_count` positions'
                        // worth of K/V for every layer; only
                        // `verified.accepted + 1` of them are real
                        // (`LayerCache::truncate`'s own doc -- proved
                        // against an incrementally-appended prefix in
                        // `layer_cache_truncate_tests`).
                        let keep_positions = cached_len_before_step + verified.accepted + 1;
                        for (layer, widths) in layer_row_widths.iter().enumerate() {
                            if let (
                                LayerPadRowWidths::Attention {
                                    even_odd_row,
                                    v_row,
                                },
                                LayerCacheState::Attention(cache),
                            ) = (widths, &mut layer_caches[layer])
                            {
                                cache.truncate(keep_positions, *even_odd_row, *v_row);
                            }
                        }
                        cached_len = keep_positions;
                        token_history.extend_from_slice(&emitted);
                        for &extra in &emitted[1..] {
                            pending.push_back(extra);
                        }
                        // `emitted`'s own last element is always
                        // `verified.next` (pushed onto the accepted-draft
                        // prefix immediately above), so this is the same
                        // value without an `Option` to unwrap.
                        next_ids = alloc::vec![verified.next];
                        return Ok(emitted[0]);
                    }
                    if is_last_step_batch {
                        // attribution slice (2026-09-22, OWNER_BRIEF_dominant_cost):
                        // one runtime knob, off by default and independent of
                        // PROXIMA_DEBUG_METAL_STAGES, gates both the O(vocab_size)
                        // non-finite scan below and the logits_hash FNV fold --
                        // neither feeds sample_next_token (it reads `last_position`
                        // directly).
                        #[cfg(feature = "instrument")]
                        let fetch_started = read_ticks();
                        #[cfg(any(feature = "instrument", feature = "metal"))]
                        let logits_diag_enabled = std::env::var_os("PROXIMA_LOGITS_DIAG").is_some();
                        let (logits, _shape) = evaluated.get(active_logits_root).ok_or(
                            InteropError::MissingEvaluatedNode {
                                node: active_logits_root,
                            },
                        )?;
                        // `logits_root` must be the `lm_head_row`-gathered LAST row
                        // only (`crate::architecture`'s doc on `BoundProgram::logits_root`)
                        // -- exactly one row of `vocab_size`. A foreign `Architecture`
                        // that hands back the full `[new_count, vocab]` buffer is
                        // rejected here rather than silently sampled at row 0.
                        #[cfg(feature = "instrument")]
                        let non_finite_count = if logits_diag_enabled {
                            logits.iter().filter(|value| !value.is_finite()).count() as u64
                        } else {
                            0
                        };
                        #[cfg(feature = "instrument")]
                        debug!(
                            step = _step as u64,
                            batch_index = batch_index as u64,
                            new_count = new_count as u64,
                            logits_len = logits.len() as u64,
                            vocab_size = vocab_size as u64,
                            non_finite_count,
                            first_five = ?&logits[..logits.len().min(5)],
                            "one_evaluation_prefill_batch: logits shape before sampling"
                        );
                        if logits.len() != vocab_size {
                            return Err(InteropError::LogitsShapeMismatch {
                                expected_rows: 1,
                                found_rows: logits.len() / vocab_size,
                                vocab: vocab_size,
                            });
                        }
                        let last_position = &logits[..vocab_size];
                        // attn_parity followon (2026-09-22, OWNER_BRIEF_gemma_head):
                        // per-step bytes verification for the
                        // `PROXIMA_HEAD_REPEATS` duplicate head dispatches --
                        // every duplicate must read back identical bytes to the
                        // production head, since both read the same operands,
                        // same `output.weight` range, same epilogue. Gated on
                        // `PROXIMA_HEAD_REPEATS_VERIFY` (separate from the
                        // repeats knob itself) so the memcmp cost never lands
                        // on a measurement run that only wants timing.
                        #[cfg(feature = "instrument")]
                        if std::env::var_os("PROXIMA_HEAD_REPEATS_VERIFY").is_some() {
                            eprintln!(
                                "head_repeats_verify step={_step} production_node={}",
                                active_logits_root.0,
                            );
                            for (offset, duplicate_node) in
                                self.duplicate_head_roots.iter().copied().enumerate()
                            {
                                let offset = offset + 1;
                                match evaluated.get(duplicate_node) {
                                    Some((duplicate_logits, _duplicate_shape)) => {
                                        let matches = duplicate_logits.len() == last_position.len()
                                            && duplicate_logits
                                                .iter()
                                                .zip(last_position.iter())
                                                .all(|(left, right)| left.to_bits() == right.to_bits());
                                        const SENTINEL_BITS: u32 = 0x7fc0_0000;
                                        let sentinel_survived = duplicate_logits
                                            .iter()
                                            .any(|value| value.to_bits() == SENTINEL_BITS);
                                        let max_abs_diff = duplicate_logits
                                            .iter()
                                            .zip(last_position.iter())
                                            .map(|(left, right)| (left - right).abs())
                                            .fold(0.0_f32, f32::max);
                                        let nan_count = duplicate_logits
                                            .iter()
                                            .filter(|value| value.is_nan())
                                            .count();
                                        eprintln!(
                                            "head_repeats_verify step={_step} duplicate_offset={offset} \
                                             node={} bytes_match={matches} sentinel_survived={sentinel_survived} \
                                             max_abs_diff={max_abs_diff} nan_count={nan_count} \
                                             dup_first_three={:?} prod_first_three={:?}",
                                            duplicate_node.0,
                                            &duplicate_logits[..duplicate_logits.len().min(3)],
                                            &last_position[..last_position.len().min(3)],
                                        );
                                    }
                                    None => {
                                        eprintln!(
                                            "head_repeats_verify step={_step} duplicate_offset={offset} \
                                             node={} MISSING_FROM_EVALUATED",
                                            duplicate_node.0,
                                        );
                                    }
                                }
                            }
                        }
                        #[cfg(feature = "instrument")]
                        let fetch_ticks = elapsed_ticks(fetch_started);
                        if std::env::var_os("PROXIMA_DEBUG_GDN_LOGITS").is_some() {
                            let mut ranked: Vec<usize> = (0..vocab_size).collect();
                            ranked.sort_unstable_by(|left, right| {
                                last_position[*right]
                                    .total_cmp(&last_position[*left])
                                    .then_with(|| left.cmp(right))
                            });
                            eprintln!(
                                "gdn_logits step={} batch_index={} top={:?} first={:?}",
                                _step,
                                batch_index,
                                ranked
                                    .iter()
                                    .take(8)
                                    .map(|index| (*index, last_position[*index]))
                                    .collect::<Vec<_>>(),
                                &last_position[..last_position.len().min(8)]
                            );
                        }
                        #[cfg(feature = "instrument")]
                        let argmax_started = read_ticks();
                        // intervention 2 (2026-09-22, OWNER_BRIEF_dominant_cost):
                        // the O(vocab_size) argmax scan and its `debug!` line feed
                        // no consumer other than this diagnostic -- `sample_next_token`
                        // below reads `last_position` directly. Gate both on the same
                        // `logits_diag_enabled` knob that already gates the logits_hash
                        // fold, so the default (unset) path pays neither cost.
                        #[cfg(feature = "instrument")]
                        let (scan_ticks, debug_ticks) = if logits_diag_enabled {
                            let (argmax_token, argmax_logit) = last_position
                                .iter()
                                .copied()
                                .enumerate()
                                .max_by(|left, right| left.1.total_cmp(&right.1))
                                .unwrap_or((0, f32::NEG_INFINITY));
                            let scan_ticks = elapsed_ticks(argmax_started);
                            let debug_started = read_ticks();
                            debug!(
                                step = _step as u64,
                                batch_index = batch_index as u64,
                                cached_len = cached_len as u64,
                                argmax_token = argmax_token as u64,
                                argmax_logit,
                                "decode_loop_step_trace: logits before sampling"
                            );
                            let debug_ticks = elapsed_ticks(debug_started);
                            (scan_ticks, debug_ticks)
                        } else {
                            (0_u64, 0_u64)
                        };
                        #[cfg(feature = "instrument")]
                        if std::env::var_os("PROXIMA_DEBUG_METAL_STAGES").is_some() {
                            let ms = |ticks: u64| {
                                proxima_tensor::instrument::ticks_to_nanos(ticks) as f64 / 1e6
                            };
                            eprintln!(
                                "token_breakdown_argmax_split step={} scan_ms={} debug_ms={}",
                                _step,
                                ms(scan_ticks),
                                ms(debug_ticks),
                            );
                        }
                        #[cfg(feature = "instrument")]
                        let argmax_ticks = elapsed_ticks(argmax_started);
                        #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                        let barriers_step = metal_stage.barriers_emitted;
                        #[cfg(not(all(
                            feature = "instrument",
                            feature = "metal",
                            target_os = "macos"
                        )))]
                        let barriers_step = 0_u64;
                        logits_sink.observe(last_position, barriers_step);
                        // attribution slice (2026-09-22, OWNER_BRIEF_dominant_cost):
                        // this eprintln is gated on PROXIMA_LOGITS_DIAG, not on
                        // PROXIMA_DEBUG_METAL_STAGES or "instrument" -- off by
                        // default. When set, one FNV1a64 fold over vocab_size f32
                        // elements (4 bytes/element) plus an unbuffered stderr
                        // write runs every decode step, same format as before so
                        // the existing corpus tooling keeps working.
                        #[cfg(feature = "instrument")]
                        let pre_logits_ticks = elapsed_ticks(pre_logits_started);
                        #[cfg(feature = "instrument")]
                        let logits_hash_started = read_ticks();
                        #[cfg(feature = "metal")]
                        if logits_diag_enabled {
                            eprintln!(
                                "logits_hash step={} hash=0x{:016x}",
                                _step,
                                logits_bits_hash(last_position)
                            );
                        }
                        #[cfg(feature = "instrument")]
                        let logits_hash_ticks = elapsed_ticks(logits_hash_started);
                        #[cfg(feature = "instrument")]
                        if std::env::var_os("PROXIMA_DEBUG_METAL_STAGES").is_some() {
                            let ms = |ticks: u64| {
                                proxima_tensor::instrument::ticks_to_nanos(ticks) as f64 / 1e6
                            };
                            eprintln!(
                                "token_breakdown_gaps step={} root_select_ms={} post_evaluate_ms={} pre_logits_ms={} checksum_ms={} fetch_ms={} argmax_ms={} logits_hash_ms={}",
                                _step,
                                ms(root_select_ticks),
                                ms(post_evaluate_ticks),
                                ms(pre_logits_ticks),
                                ms(checksum_ticks),
                                ms(fetch_ticks),
                                ms(argmax_ticks),
                                ms(logits_hash_ticks),
                            );
                        }

                        #[cfg(feature = "instrument")]
                        let greedy_pick_started = read_ticks();
                        token_id = match token_override.and_then(|forced| forced.get(_step)) {
                            Some(&forced_token) => forced_token,
                            None => {
                                let recent_window_start =
                                    token_history.len().saturating_sub(repeat_window);
                                let recent_tokens = &token_history[recent_window_start..];
                                sample_next_token(
                                    last_position,
                                    recent_tokens,
                                    sample_config,
                                    &mut rng,
                                )
                                .ok_or(InteropError::EmptyLogits)?
                            }
                        };
                        token_history.push(token_id);
                        #[cfg(feature = "instrument")]
                        let greedy_pick_ticks = elapsed_ticks(greedy_pick_started);
                        #[cfg(feature = "instrument")]
                        debug!(
                            step = _step as u64,
                            cached_len_after = (cached_len_before_step + new_count) as u64,
                            token_sampled = token_id,
                            "decode_loop_step_trace: sampled token fed forward as next step's next_ids"
                        );
                        // Cacheless architectures (`active_layer_roots.is_empty()`)
                        // have no KV cache carrying prior context forward, so
                        // feeding only the new token would forward a one-token
                        // sequence with no history. Re-prefill the FULL growing
                        // sequence instead -- `next_ids` here is still THIS step's
                        // own `ids_for_step` source (its last use already passed),
                        // so appending is exactly last step's sequence plus the
                        // token just sampled.
                        next_ids = if active_layer_roots.is_empty() {
                            let mut resent_sequence = next_ids.clone();
                            resent_sequence.push(token_id);
                            resent_sequence
                        } else {
                            alloc::vec![token_id]
                        };

                        #[cfg(feature = "instrument")]
                        {
                            emit_token_breakdown(&TokenBreakdown {
                                step: _step,
                                new_count,
                                cached_len_before: cached_len_before_step,
                                step_wall_ticks: elapsed_ticks(step_started),
                                apply_serving_config_ticks,
                                build_position_inputs_ticks,
                                named_blocks_weights_ticks,
                                named_blocks_kv_ticks,
                                kv_cache_upload_bytes: kv_cache_upload_elements * 4,
                                ssm_state_transfer_bytes,
                                evaluate_ticks,
                                layer_cache_append_ticks,
                                layer_cache_append_bytes: layer_cache_append_elements * 4,
                                greedy_pick_ticks,
                            });
                            // ROW 130's per-step-reset attribution: kernel / dispatch+
                            // setup / park+spin+wake, all on the CALLING thread's own
                            // wall clock (never summed across the cohort's other worker
                            // threads, which run concurrently with it, not serially
                            // inside it -- see `CohortLeaderAttribution`'s own doc).
                            // `evaluate_ns` is this step's own tick-based total, already
                            // reset per step by `reset_step`; `residual_ns` is
                            // everything `evaluate_ms` paid for that these three terms
                            // do not name -- non-matmul ops (elementwise/reduce/scan),
                            // quantize/transpose bookkeeping, and staged-batch setup
                            // outside the cohort round itself. `saturating_sub` so a
                            // negative residual is impossible to construct by
                            // arithmetic; reported as 0 with `residual_underflow=true`
                            // if the three named terms would have exceeded the parent,
                            // which is itself a sanity-gate failure worth seeing rather
                            // than silently wrapping.
                            let attribution =
                                proxima_tensor::instrument::cohort_leader_attribution();
                            let evaluate_ns = ticks_to_nanos(evaluate_ticks);
                            let named_ns = attribution.kernel_nanos
                                + attribution.dispatch_nanos
                                + attribution.park_spin_wake_nanos;
                            let residual_underflow = named_ns > evaluate_ns;
                            let residual_ns = evaluate_ns.saturating_sub(named_ns);
                            info!(
                                step = _step as u64,
                                evaluate_ms = evaluate_ns as f64 / 1e6,
                                kernel_ms = attribution.kernel_nanos as f64 / 1e6,
                                dispatch_ms = attribution.dispatch_nanos as f64 / 1e6,
                                park_spin_wake_ms = attribution.park_spin_wake_nanos as f64 / 1e6,
                                residual_ms = residual_ns as f64 / 1e6,
                                residual_underflow,
                                named_plus_residual_ms = (named_ns + residual_ns) as f64 / 1e6,
                                cached_attention_ops = proxima_tensor::instrument::path_totals()
                                    .op_kind_cached_attention,
                                "token_attribution: per-step kernel/dispatch/park-spin-wake split"
                            );
                            // ROW 140's own redundant-activation-quantize hypothesis
                            // check: `total_calls` vs `distinct_nodes` across every
                            // matmul reduce node this step evaluated. 1:1 kills the
                            // hypothesis; a ratio near the QKV/gate-up fan-out (2-3x)
                            // confirms it.
                            let (quantize_total_calls, quantize_distinct_nodes) =
                                proxima_tensor::instrument::quantize_activation_call_stats();
                            let quantize_cache_hits =
                                proxima_tensor::instrument::QUANTIZE_ACTIVATION_CACHE_HITS.get();
                            info!(
                                step = _step as u64,
                                total_calls = quantize_total_calls,
                                distinct_nodes = quantize_distinct_nodes,
                                cache_hits = quantize_cache_hits,
                                "token_quantize_calls: redundant-activation-quantize hypothesis check"
                            );
                            #[cfg(all(feature = "metal", target_os = "macos"))]
                            emit_token_breakdown_metal(
                                _step,
                                &metal_stage,
                                runtime.plans_len(),
                                runtime.plan_hits,
                                runtime.plan_misses,
                                runtime.arena_allocated_bytes(),
                                self.mapping_residency_rung,
                            );
                            // This arm's `kv_cache.{layer}.*` blocks are ordinary
                            // named blocks folded into `metal_stage`'s weight
                            // counters above (`emit_device_memory_by_class`'s own
                            // doc), never a separately sized device allocation --
                            // `None` here is honest, not a placeholder.
                            #[cfg(all(feature = "metal", target_os = "macos"))]
                            if _step == 0 {
                                emit_device_memory_by_class(
                                    _step,
                                    metal_stage.block_nocopy_bound_bytes,
                                    metal_stage.block_copied_bytes,
                                    metal_stage.block_offset_bound_bytes,
                                    None,
                                    omega::metal::current_allocated_size().unwrap_or(0),
                                    phys_footprint_bytes(),
                                );
                            }
                        }
                    }
                }

                let residency_boundary_requested = qwen35moe_residency.is_some()
                    && qwen35moe_pre_gather_enabled(
                        serving_config.qwen35moe_pre_gather,
                        self.architecture_impl.is_some_and(|architecture| {
                            architecture.ffn_routing() == crate::architecture::FfnRouting::Routed
                        }),
                    )
                    && runtime.uses_gpu()
                    && !monolithic_high_mmap_requested;
                if residency_boundary_requested && let Some(policy) = qwen35moe_residency.as_mut() {
                    // Current routes were served from fixed decisions captured
                    // before each gather. Reconcile once after the whole token,
                    // so retained residency actions cannot churn later layers
                    // of the same token.
                    self.reconcile_attached_qwen35moe_residency(policy)?;
                    if std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some() {
                        eprintln!("qwen35 residency boundary position={cached_len}");
                    }
                }

                Ok(token_id)
            },
            on_token,
        );

        #[cfg(all(feature = "metal", target_os = "macos"))]
        if should_release_monolithic_sources(
            monolithic_all_low_requested,
            runtime.retain_monolithic_prefill_sources,
        ) {
            clear_expert_source_cache();
        }

        let (generated_ids, stopped_by_eos) = decode_result?;

        #[cfg(feature = "qwen35moe-expert-prefetch")]
        if qwen35moe_expert_prefetch_enabled
            && std::env::var_os("PROXIMA_DEBUG_EXPERT_PREFETCH").is_some()
        {
            eprintln!(
                "qwen35 expert prefetch stats predictions={} hits={} overfetch={} advice_events={} advised_bytes={}",
                qwen35moe_prefetch_prediction_count,
                qwen35moe_prefetch_hit_count,
                qwen35moe_prefetch_overfetch_count,
                qwen35moe_prefetch_advice_events,
                qwen35moe_prefetch_advised_bytes
            );
        }

        let text = proxima_tokenizer::decode(&generated_ids, &self.vocab)?;
        // Exactly the tokens `cached_len` now covers: `seed_ids ++ ids`
        // (this call's own new range) plus however many of its OWN
        // generated tokens have themselves been forward-passed since --
        // every generated token except the last is (the last is only
        // just-sampled, never yet fed back through `evaluate`). Derived
        // from `cached_len` itself rather than re-counting loop iterations,
        // so it is correct on every exit path `decode_until_stop_or_budget`
        // has (`max_tokens` exhaustion, model EOS, `ControlFlow::Break(())`
        // from either phase) without special-casing any of them.
        let mut final_ids = seed_ids;
        final_ids.extend_from_slice(&ids);
        let forwarded_generated = cached_len
            .saturating_sub(final_ids.len())
            .min(generated_ids.len());
        final_ids.extend_from_slice(&generated_ids[..forwarded_generated]);
        let final_state = PrefixState {
            ids: final_ids,
            layer_caches,
            cached_len,
        };
        Ok((generated_ids, text, stopped_by_eos, final_state))
    }

    /// [`Self::run_decode_loop`]'s persistent-device-resident-KV arm:
    /// [`SingleRangeProgram`] in place of the two-range `program`, one
    /// [`PlacedBuffer`] triple per layer (`k_even`/`k_odd`/`v`) allocated
    /// ONCE, at `(prompt_len + max_tokens).min(context_length)` capacity
    /// (this call's actual reachable position count, never
    /// `context_length` itself -- see the sizing comment in the body) and
    /// held for this whole call, in place of `LayerCache`'s per-step
    /// `extend_from_slice` growth. Each step places this step's own
    /// freshly rotated key/value (`single_range.cache_roots[layer]`, the
    /// OUTPUT side) into the buffer's tail at `cached_len * row_bytes`, and
    /// reads the SAME buffer back as this step's `kv_cache.{layer}.*`
    /// input (`single_range.cache_input_nodes[layer]`) covering `[0,
    /// cached_len + new_count)` -- one program, one command buffer, the
    /// write visible to the later read through Metal's own whole-resource
    /// hazard tracking (`omega::metal::execute_plan_with_placements`'s own
    /// doc, "Within-call aliasing"). No host round trip either direction:
    /// the cache never leaves the device, and the per-layer cache roots
    /// are not requested as `Evaluated` outputs at all (only
    /// `single_range.logits_root` is).
    ///
    /// `ids`/`token_history`/`sample_config`/`rng` arrive already built by
    /// [`Self::run_decode_loop`]'s shared prefix -- this method's own body
    /// starts exactly where that function's two-range arm does.
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_decode_loop_placed_kv(
        &self,
        single_range: &SingleRangeProgram,
        ids: Vec<u32>,
        mut token_history: Vec<u32>,
        repeat_window: usize,
        sample_config: SamplingConfig,
        mut rng: fastrand::Rng,
        max_tokens: usize,
        serving_config: &ServingConfig,
        runtime: &mut BackendRuntime,
        token_override: Option<&[u32]>,
        logits_sink: &mut LogitsSink,
        on_token: &mut dyn FnMut(TokenEvent<'_>) -> ControlFlow<(), ()>,
    ) -> Result<(Vec<u32>, String, bool), InteropError> {
        let prompt_token_count = ids.len();
        let block_count = self.architecture.block_count as usize;
        let kv_heads = self.architecture.kv_heads as usize;
        let head_dim = self.architecture.head_dim as usize;
        let pairs = head_dim / 2;
        let context_length = serving_config.context_length as usize;

        // Sized from what THIS call can actually reach (`prompt_len +
        // max_tokens`), not `context_length` (default 131_072). Capping the
        // allocation at `positions_needed` (never more than
        // `context_length`) cannot admit a step this call could not already
        // reach -- `apply_serving_config` below still rejects any step whose
        // `merged_len` would exceed `context_length`.
        let reachable_positions = (ids.len() + max_tokens).min(context_length);
        let positions_needed = kv_extent(
            reachable_positions,
            context_length,
            serving_config.kv_bucket_tokens,
        );

        let row_bytes_even_odd = kv_heads * pairs * core::mem::size_of::<f32>();
        let row_bytes_v = kv_heads * head_dim * core::mem::size_of::<f32>();
        let capacity_even_odd = positions_needed * row_bytes_even_odd;
        let capacity_v = positions_needed * row_bytes_v;

        let mut k_even_buffers = Vec::with_capacity(block_count);
        let mut k_odd_buffers = Vec::with_capacity(block_count);
        let mut v_buffers = Vec::with_capacity(block_count);
        for _ in 0..block_count {
            k_even_buffers.push(allocate_placed_buffer(capacity_even_odd)?);
            k_odd_buffers.push(allocate_placed_buffer(capacity_even_odd)?);
            v_buffers.push(allocate_placed_buffer(capacity_v)?);
        }
        // `kv_extent` (unconditional, keyed off
        // `ServingConfig::kv_bucket_tokens`, see that field's own doc)
        // reads `bucket` rows per step whenever `bucket_tokens > 1`,
        // `bucket > merged_len` -- the tail `[merged_len, bucket)` was
        // never written by this call yet. A freshly allocated
        // `MTLBuffer`'s contents are undefined
        // (`omega::metal::zero_placed_buffer`'s own doc), so that tail is
        // zeroed ONCE here, at allocation, rather than paying a per-step
        // re-zero: any row a later step reads was either zeroed here or
        // overwritten by a real rotated key/value this same call already
        // wrote, since `cached_len` only grows. Unconditional (not gated
        // on the `kv-capacity-bucket` cargo feature, which only pulls in
        // `proxima-tensor`'s CPU-side correctness proof and no longer
        // controls this runtime behaviour) because
        // `ServingConfig::default`'s `kv_bucket_tokens: 32` buckets on
        // every build regardless of which cargo features are enabled.
        for layer in 0..block_count {
            omega::metal::zero_placed_buffer(&k_even_buffers[layer], capacity_even_odd);
            omega::metal::zero_placed_buffer(&k_odd_buffers[layer], capacity_even_odd);
            omega::metal::zero_placed_buffer(&v_buffers[layer], capacity_v);
        }

        let kv_cache_names: Vec<(String, String, String)> = (0..block_count)
            .map(|layer| {
                (
                    alloc::format!("kv_cache.{layer}.k_even"),
                    alloc::format!("kv_cache.{layer}.k_odd"),
                    alloc::format!("kv_cache.{layer}.v"),
                )
            })
            .collect();

        // `prepare`'s own element-count check (`omega::metal::prepare`,
        // `found != expected`) runs against EVERY named block, placed or
        // not -- these three names are always input-placed below, so their
        // data is never read, only their LENGTH, which must match this
        // step's `merged_len * elements_per_position`. `resize` only grows
        // when `merged_len` grows past the previous step's value, the same
        // amortized cost `LayerCache::append`'s `extend_from_slice` paid,
        // minus the real data this scratch never holds and the device
        // upload `execute_plan_with_placements` skips for a placed input.
        let mut cache_length_scratch_even_odd: Vec<f32> = Vec::new();
        let mut cache_length_scratch_v: Vec<f32> = Vec::new();

        let resident_names: BTreeSet<&str> = self.resident_names();

        let mut cached_len = 0usize;
        let mut next_ids = ids;
        let vocab_size = self.architecture.vocab as usize;

        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(
            &self.vocab,
            max_tokens,
            prompt_token_count,
            |_step| {
                #[cfg(feature = "instrument")]
                proxima_tensor::instrument::reset_step();
                #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                omega::set_capture_step(_step as u64);
                #[cfg(feature = "instrument")]
                let step_started = read_ticks();

                let new_count = next_ids.len();
                let merged_len = cached_len + new_count;
                #[cfg(feature = "instrument")]
                let apply_serving_config_started = read_ticks();
                apply_serving_config(serving_config, merged_len)?;
                #[cfg(feature = "instrument")]
                let apply_serving_config_ticks = elapsed_ticks(apply_serving_config_started);

                #[cfg(feature = "instrument")]
                let build_position_inputs_started = read_ticks();
                let inputs = build_position_inputs(
                    &next_ids,
                    cached_len,
                    self.architecture.head_dim,
                    self.architecture.rope_freq_base,
                    self.architecture.rms_epsilon,
                    self.architecture_impl
                        .as_ref()
                        .and_then(|architecture| architecture.rope_freq_factors(&self.weights)),
                );
                #[cfg(feature = "instrument")]
                let build_position_inputs_ticks = elapsed_ticks(build_position_inputs_started);

                let mut named_blocks: Vec<(&str, QuantizedBlock)> = Vec::with_capacity(
                    self.weights.owned.len()
                        + self.weights.packed.len()
                        + self.weights.packed_owned.len()
                        + 4
                        + block_count * 3,
                );
                named_blocks.push(("ids", QuantizedBlock::Int32(inputs.ids_i32.as_slice())));
                #[cfg(feature = "instrument")]
                let named_blocks_weights_started = read_ticks();
                for (name, data) in &self.weights.owned {
                    named_blocks.push((name.as_str(), QuantizedBlock::Float32(data.as_slice())));
                }
                for (name, block) in &self.weights.packed {
                    named_blocks.push((name.as_str(), *block));
                }
                for (name, bytes, kind) in &self.weights.packed_owned {
                    let block = crate::bind::as_block(*kind, bytes)
                        .ok_or(InteropError::UnsupportedCodec { codec: *kind })?;
                    named_blocks.push((name.as_str(), block));
                }
                named_blocks.push(("eps", QuantizedBlock::Float32(inputs.epsilon.as_slice())));
                named_blocks.push(("rope_cos", QuantizedBlock::Float32(inputs.cos.as_slice())));
                named_blocks.push(("rope_sin", QuantizedBlock::Float32(inputs.sin.as_slice())));
                let cached_len_scalar = [cached_len as f32];
                named_blocks.push(("cached_len", QuantizedBlock::Float32(&cached_len_scalar)));
                // See the sibling decode loop's own comment on
                // `lm_head_row` above -- same leaf, same host-supplied
                // reason, this step's own last new row.
                let lm_head_row_scalar = [(new_count - 1) as f32];
                named_blocks.push(("lm_head_row", QuantizedBlock::Float32(&lm_head_row_scalar)));
                #[cfg(feature = "instrument")]
                let named_blocks_weights_ticks = elapsed_ticks(named_blocks_weights_started);

                // This step's `kv_cache.{layer}.*` `named_blocks` entries are
                // placeholder scratch, not the real KV cache -- see this
                // arm's own doc on why `execute_plan_with_placements` never
                // uploads them.
                #[cfg(feature = "instrument")]
                let named_blocks_kv_started = read_ticks();
                // `serving_config.kv_bucket_tokens == 1`: `kv_bound_extent
                // == merged_len`, unchanged from the pre-bucketing shape.
                // Otherwise rounded up to that many tokens and capped at
                // `positions_needed` (this call's own per-layer buffer row
                // count) -- see `kv_extent`'s own doc. This is BOTH the
                // scratch named-block length below (the strict
                // `found == expected` validator checks it against the
                // SAME extent the KV `Op::Input` leaves bind to) and
                // `symbols[1]` a few lines down, so the two can never
                // disagree.
                let kv_bound_extent = kv_extent(
                    merged_len,
                    positions_needed,
                    serving_config.kv_bucket_tokens,
                );
                let even_odd_len = kv_bound_extent * kv_heads * pairs;
                let v_len = kv_bound_extent * kv_heads * head_dim;
                if cache_length_scratch_even_odd.len() < even_odd_len {
                    cache_length_scratch_even_odd.resize(even_odd_len, 0.0);
                }
                if cache_length_scratch_v.len() < v_len {
                    cache_length_scratch_v.resize(v_len, 0.0);
                }
                for names in &kv_cache_names {
                    named_blocks.push((
                        names.0.as_str(),
                        QuantizedBlock::Float32(&cache_length_scratch_even_odd[..even_odd_len]),
                    ));
                    named_blocks.push((
                        names.1.as_str(),
                        QuantizedBlock::Float32(&cache_length_scratch_even_odd[..even_odd_len]),
                    ));
                    named_blocks.push((
                        names.2.as_str(),
                        QuantizedBlock::Float32(&cache_length_scratch_v[..v_len]),
                    ));
                }
                #[cfg(feature = "instrument")]
                let named_blocks_kv_ticks = elapsed_ticks(named_blocks_kv_started);

                let mut input_placements: Vec<(NodeId, &PlacedBuffer, usize)> =
                    Vec::with_capacity(block_count * 3);
                let mut output_placements: Vec<(NodeId, &PlacedBuffer, usize)> =
                    Vec::with_capacity(block_count * 3);
                for layer in 0..block_count {
                    let (even_input, odd_input, value_input) =
                        single_range.cache_input_nodes[layer];
                    let (even_output, odd_output, value_output) = single_range.cache_roots[layer];
                    input_placements.push((even_input, &k_even_buffers[layer], 0));
                    input_placements.push((odd_input, &k_odd_buffers[layer], 0));
                    input_placements.push((value_input, &v_buffers[layer], 0));
                    output_placements.push((
                        even_output,
                        &k_even_buffers[layer],
                        cached_len * row_bytes_even_odd,
                    ));
                    output_placements.push((
                        odd_output,
                        &k_odd_buffers[layer],
                        cached_len * row_bytes_even_odd,
                    ));
                    output_placements.push((
                        value_output,
                        &v_buffers[layer],
                        cached_len * row_bytes_v,
                    ));
                }

                let symbols = [new_count as u64, kv_bound_extent as u64];
                // Every placed-output node must also be a `roots` entry, or
                // `prepare`'s `BoundOpBuilder::finish` (`proxima-tensor`'s
                // `bind.rs`) never force-materializes it and its write lands
                // wherever `held`'s end-of-walk flush happens to fall --
                // AFTER every layer's score already read the buffer, and
                // `omega::metal::prepare`'s own `prune_dead` pass drops it
                // from `resolved` entirely (see
                // `omega::metal::execute_plan_with_placements`'s own doc).
                // The two-range path above (`roots.push` for each
                // `cache_roots` entry) already relies on this; this arm was
                // missing it.
                let mut roots: Vec<NodeId> =
                    Vec::with_capacity(2 + single_range.cache_roots.len() * 3);
                roots.push(single_range.logits_root);
                if let Some(scratch) = single_range.duplicate_head_scratch {
                    roots.push(scratch);
                }
                for (even, odd, value) in &single_range.cache_roots {
                    roots.push(*even);
                    roots.push(*odd);
                    roots.push(*value);
                }
                #[cfg(all(
                    feature = "metal",
                    feature = "metal-fuse-attn-decode",
                    feature = "metal-output-placement",
                    target_os = "macos"
                ))]
                if attn_fuse_parity_target_steps().contains(&_step) {
                    run_attn_fuse_parity_probe(
                        _step,
                        &single_range.program,
                        &symbols,
                        &named_blocks,
                        &roots,
                        single_range.logits_root,
                        &resident_names,
                        &input_placements,
                        runtime,
                    )?;
                }
                #[cfg(feature = "instrument")]
                let evaluate_started = read_ticks();
                // ROW 329: this step's `execute_plan_with_placements_dispatch_timed`
                // encoder-split result, if the branch below actually took it
                // -- read out here (not inside the match arm) because
                // `report_encoder_split`'s own `gpu_exec_ms` needs
                // `metal_stage_totals`'s post-match snapshot, the same
                // snapshot-and-reset value `emit_token_breakdown_metal`
                // reads a few lines below, never a second counter read.
                #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                let mut encoder_split_ns: Option<(u64, u64)> = None;
                // `PROXIMA_METAL_OP_PROFILE_STEP` -- same diagnostic-only,
                // `instrument`-gated, default-off convention as
                // `run_decode_loop`'s own branch above (that one's doc has
                // the full rationale): unset in every production run, so
                // `evaluate_with_placements` is the only path a caller
                // without this env var ever takes on the default decode
                // path too. When set to this step's own index, this ONE
                // step instead runs `evaluate_op_timed_with_placements`
                // (per-op command buffers against the SAME placed-KV
                // shape) and prints the per-op GPU attribution through the
                // identical `report_op_timings` the two-range path already
                // uses.
                #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                let evaluated = match std::env::var("PROXIMA_METAL_OP_PROFILE_STEP")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                {
                    Some(target) if target == _step => {
                        let (evaluated, timings) = runtime.evaluate_op_timed_with_placements(
                            &single_range.program,
                            &symbols,
                            &named_blocks,
                            &roots,
                            &resident_names,
                            &input_placements,
                            &output_placements,
                        )?;
                        report_op_timings(_step, &timings, &single_range.program);
                        evaluated
                    }
                    // `PROXIMA_METAL_DISPATCH_PROFILE_STEP` -- this branch's
                    // own per-dispatch-in-the-batched-buffer twin: same
                    // default-off, `instrument`-gated, one-env-var-per-step
                    // convention as `PROXIMA_METAL_OP_PROFILE_STEP` above,
                    // reusing `report_op_timings` unchanged since
                    // `evaluate_dispatch_timed_with_placements` returns the
                    // same `Vec<OpGpuTiming>` shape.
                    _ if std::env::var("PROXIMA_METAL_DISPATCH_PROFILE_STEP")
                        .ok()
                        .and_then(|value| value.parse::<usize>().ok())
                        == Some(_step) =>
                    {
                        let (evaluated, timings, sampling_mode, split_ns) = runtime
                            .evaluate_dispatch_timed_with_placements(
                                &single_range.program,
                                &symbols,
                                &named_blocks,
                                &roots,
                                &resident_names,
                                &input_placements,
                                &output_placements,
                            )?;
                        info!(
                            step = _step as u64,
                            sampling_mode,
                            "dispatch_profile: per-dispatch gpu-timestamp sampling mode"
                        );
                        report_op_timings(_step, &timings, &single_range.program);
                        encoder_split_ns = split_ns;
                        evaluated
                    }
                    _ => runtime.evaluate_with_placements(
                        &single_range.program,
                        &symbols,
                        &named_blocks,
                        &roots,
                        &resident_names,
                        &input_placements,
                        &output_placements,
                        None,
                    )?,
                };
                #[cfg(not(all(feature = "instrument", feature = "metal", target_os = "macos")))]
                let evaluated = runtime.evaluate_with_placements(
                    &single_range.program,
                    &symbols,
                    &named_blocks,
                    &roots,
                    &resident_names,
                    &input_placements,
                    &output_placements,
                    None,
                )?;
                #[cfg(feature = "instrument")]
                let evaluate_ticks = elapsed_ticks(evaluate_started);
                // Snapshot-and-reset (`metal_stage_totals`'s own doc), so this
                // read must happen exactly once per step, immediately after
                // this step's own `evaluate_with_placements` call --
                // `block_offered_bytes` below is this step's REAL device
                // upload byte count: the three `kv_cache.{layer}.*` scratch
                // blocks pushed above never actually upload (they are
                // `input_placements` entries, which `execute_plan_with_placements`
                // binds directly to the caller's own device buffer instead of
                // staging through `upload_block` -- see that function's own
                // doc, "skipping the per-call `upload_block`/
                // `upload_packed_bytes` host round trip entirely"), so this
                // count legitimately falls to just the resident weights'
                // one-time cost after step 0, never a hardcoded zero.
                #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                let metal_stage = metal_stage_totals();
                // `max_command_buffers_per_token == 0` (the default) leaves this
                // step's command-buffer count unmeasured -- ROW invariant 1 is a
                // caller-opted-in ceiling, not an unconditional assertion.
                #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                if serving_config.max_command_buffers_per_token > 0
                    && metal_stage.gpu_exec_calls
                        > serving_config.max_command_buffers_per_token as u64
                {
                    return Err(InteropError::TooManyCommandBuffers {
                        step: _step,
                        committed: metal_stage.gpu_exec_calls,
                        limit: serving_config.max_command_buffers_per_token,
                    });
                }
                #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                if std::env::var_os("PROXIMA_DEBUG_TOKEN_STAGES").is_some() {
                    eprintln!(
                        "token_stages step={} cached_len={} evaluate_ms={:.3} gpu_exec_ms={:.3} prepare_ms={:.3} pipeline_misses={} pipeline_compile_ms={:.3} op_setup_ms={:.3} block_upload_ms={:.3} readback_ms={:.3}",
                        _step,
                        cached_len,
                        ticks_to_nanos(evaluate_ticks) as f64 / 1e6,
                        ticks_to_nanos(metal_stage.gpu_exec_ticks) as f64 / 1e6,
                        ticks_to_nanos(metal_stage.prepare_ticks) as f64 / 1e6,
                        metal_stage.pipeline_misses,
                        ticks_to_nanos(metal_stage.pipeline_compile_ticks) as f64 / 1e6,
                        ticks_to_nanos(metal_stage.op_setup_ticks) as f64 / 1e6,
                        ticks_to_nanos(metal_stage.block_upload_ticks) as f64 / 1e6,
                        ticks_to_nanos(metal_stage.readback_ticks) as f64 / 1e6,
                    );
                }
                #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                if let Some(split_ns) = encoder_split_ns {
                    report_encoder_split(
                        _step,
                        split_ns,
                        ticks_to_nanos(metal_stage.gpu_exec_ticks),
                    );
                }
                #[cfg(feature = "instrument")]
                let cached_len_before_step = cached_len;
                cached_len = merged_len;

                let (logits, _shape) = evaluated.get(single_range.logits_root).ok_or(
                    InteropError::MissingEvaluatedNode {
                        node: single_range.logits_root,
                    },
                )?;
                // `logits_root` must be the `lm_head_row`-gathered LAST row
                // only (`crate::architecture`'s doc on `BoundProgram::logits_root`)
                // -- exactly one row of `vocab_size`. A foreign `Architecture`
                // that hands back the full `[new_count, vocab]` buffer is
                // rejected here rather than silently sampled at row 0.
                if logits.len() != vocab_size {
                    return Err(InteropError::LogitsShapeMismatch {
                        expected_rows: 1,
                        found_rows: logits.len() / vocab_size,
                        vocab: vocab_size,
                    });
                }
                let last_position = &logits[..vocab_size];
                #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                let barriers_step = metal_stage.barriers_emitted;
                #[cfg(not(all(feature = "instrument", feature = "metal", target_os = "macos")))]
                let barriers_step = 0_u64;
                logits_sink.observe(last_position, barriers_step);
                // attribution slice (2026-09-22, OWNER_BRIEF_dominant_cost): same
                // PROXIMA_LOGITS_DIAG gate as the batch decode path's logits_hash.
                #[cfg(feature = "metal")]
                if std::env::var_os("PROXIMA_LOGITS_DIAG").is_some() {
                    eprintln!(
                        "logits_hash step={} hash=0x{:016x}",
                        _step,
                        logits_bits_hash(last_position)
                    );
                }

                #[cfg(feature = "instrument")]
                let greedy_pick_started = read_ticks();
                let token_id = match token_override.and_then(|forced| forced.get(_step)) {
                    Some(&forced_token) => forced_token,
                    None => {
                        let recent_window_start = token_history.len().saturating_sub(repeat_window);
                        let recent_tokens = &token_history[recent_window_start..];
                        sample_next_token(last_position, recent_tokens, sample_config, &mut rng)
                            .ok_or(InteropError::EmptyLogits)?
                    }
                };
                token_history.push(token_id);
                #[cfg(feature = "instrument")]
                let greedy_pick_ticks = elapsed_ticks(greedy_pick_started);
                next_ids = alloc::vec![token_id];

                #[cfg(feature = "instrument")]
                {
                    emit_token_breakdown(&TokenBreakdown {
                        step: _step,
                        new_count,
                        cached_len_before: cached_len_before_step,
                        step_wall_ticks: elapsed_ticks(step_started),
                        apply_serving_config_ticks,
                        build_position_inputs_ticks,
                        named_blocks_weights_ticks,
                        named_blocks_kv_ticks,
                        // No host-side KV cache exists on this arm (the cache
                        // lives entirely in `k_even_buffers`/`k_odd_buffers`/
                        // `v_buffers`, device-resident for the whole call), so
                        // there is no real host-upload byte count to report
                        // here. `metal_stage.block_offered_bytes` used to fill
                        // this field instead (ROW 369) -- that is the whole
                        // bound program's residency census, weights included,
                        // not a KV-cache-specific count, and it is already
                        // reported honestly under its own name by
                        // `emit_token_breakdown_metal`'s `block_offered_bytes`
                        // field below.
                        kv_cache_upload_bytes: 0,
                        ssm_state_transfer_bytes: 0,
                        evaluate_ticks,
                        // No separate host layer-cache append step on this
                        // arm -- `output_placements` above writes each
                        // layer's freshly rotated key/value straight into its
                        // `PlacedBuffer` as part of the SAME evaluate call
                        // `evaluate_ticks` already timed, so there is no
                        // second cost to attribute here.
                        layer_cache_append_ticks: 0,
                        layer_cache_append_bytes: 0,
                        greedy_pick_ticks,
                    });
                    #[cfg(all(feature = "metal", target_os = "macos"))]
                    emit_token_breakdown_metal(
                        _step,
                        &metal_stage,
                        runtime.placed_plans_len(),
                        runtime.plan_hits,
                        runtime.plan_misses,
                        runtime.arena_allocated_bytes(),
                        self.mapping_residency_rung,
                    );
                    #[cfg(all(feature = "metal", target_os = "macos"))]
                    if _step == 0 {
                        let kv_cache_device_bytes =
                            (block_count * (2 * capacity_even_odd + capacity_v)) as u64;
                        emit_device_memory_by_class(
                            _step,
                            metal_stage.block_nocopy_bound_bytes,
                            metal_stage.block_copied_bytes,
                            metal_stage.block_offset_bound_bytes,
                            Some(kv_cache_device_bytes),
                            omega::metal::current_allocated_size().unwrap_or(0),
                            phys_footprint_bytes(),
                        );
                    }
                }

                Ok(token_id)
            },
            on_token,
        )?;

        let text = proxima_tokenizer::decode(&generated_ids, &self.vocab)?;
        Ok((generated_ids, text, stopped_by_eos))
    }

    /// A one-shot forward pass over `prompt` (BOS forced, fresh KV state,
    /// same input-binding shape as `Self::run_decode_loop`'s own first
    /// step) that returns the raw values for each requested `NodeId`
    /// instead of sampling a token from the final logits.
    ///
    /// Exists as this crate's cross-oracle diagnostic surface: a decoded
    /// token is an argmax, and an argmax destroys exactly the information
    /// that tells a near-tied bf16-vs-f16 rounding gap apart from a gross
    /// defect. [`Self::forward_logits`] is the `node_ids == [logits_root]`
    /// convenience most callers want; this general form additionally lets a
    /// caller bisect a numeric divergence by depth: build
    /// [`proxima_tensor::spec::mistral_cached_forward_program_with_experts`]
    /// again at a shorter `block_count` against this same architecture and
    /// read off the last shared `NodeId` (`proxima_tensor::op::append`'s
    /// id-is-index invariant guarantees the two programs agree on every
    /// `NodeId` up to the point they diverge) -- that id is the residual
    /// stream's value right after that layer, comparable directly against
    /// an oracle's own per-layer tensor dump.
    /// `examples/smollm2_logit_oracle_diff.rs` is the worked tool.
    ///
    /// # Errors
    ///
    /// Whatever tokenizing `prompt` against this checkpoint's own
    /// [`Vocab`] or evaluating its forward program can fail with, plus
    /// [`InteropError::MissingEvaluatedNode`] if any `node_ids` entry was
    /// never computed by this checkpoint's own forward program (a caller
    /// passed a `NodeId` from a differently-shaped program).
    pub fn forward_node_values(
        &self,
        prompt: &str,
        node_ids: &[NodeId],
    ) -> Result<Vec<Vec<f32>>, InteropError> {
        self.forward_node_values_on_backend(prompt, node_ids, 0)
    }

    /// Captures `node_ids` after every stateful prompt-position evaluation
    /// through the ordinary cached decode loop. This is the causal companion
    /// to [`Self::forward_node_values_on_backend`], whose cache is always empty.
    pub fn forward_cached_node_values_on_backend(
        &self,
        prompt: &str,
        node_ids: &[NodeId],
        gpu_layers: i32,
    ) -> Result<Vec<Vec<Vec<f32>>>, InteropError> {
        let serving_config = supported_serving_config(
            gpu_layers,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            omega::MathMode::default(),
        );
        let mut runtime = BackendRuntime::new(&serving_config);
        let mut steps = Vec::new();
        let mut node_values_sink = NodeValuesSink::Collect {
            nodes: node_ids,
            steps: &mut steps,
        };
        let _ = self.run_decode_loop_observed_seeded(
            prompt,
            1,
            &serving_config,
            &mut runtime,
            None,
            &mut LogitsSink::Discard,
            &mut node_values_sink,
            &mut |_event| ControlFlow::Continue(()),
            None,
            true,
        )?;
        Ok(steps)
    }

    /// [`Self::forward_node_values`] with the backend left open --
    /// `gpu_layers` reaches `BackendRuntime::new`/`select_backend` the
    /// same way [`Self::generate_with_serving_config`]'s own
    /// `serving_config.gpu_layers` already does, so this one-shot forward
    /// can be pinned to CPU (`0`) or Metal ([`crate::serving::GPU_LAYERS_ALL`],
    /// `metal`-featured builds only) instead of always running CPU the way
    /// [`Self::forward_node_values`] does today. `crate::quality`'s
    /// reference-vs-variant harness is the reason this exists: comparing
    /// Metal's own decode path against the CPU reference needs the SAME
    /// one-shot forward run on each backend in turn, not two different
    /// programs.
    ///
    /// # Errors
    ///
    /// Whatever [`Self::forward_node_values`] can fail with, plus
    /// [`InteropError::UnsupportedServingConfig`] if `gpu_layers` requests a
    /// backend [`apply_serving_config`] does not accept (`0` and
    /// [`crate::serving::GPU_LAYERS_ALL`] on a `metal`-featured build are the
    /// only two).
    pub fn forward_node_values_on_backend(
        &self,
        prompt: &str,
        node_ids: &[NodeId],
        gpu_layers: i32,
    ) -> Result<Vec<Vec<f32>>, InteropError> {
        let serving_config = supported_serving_config(
            gpu_layers,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            omega::MathMode::default(),
        );
        let mut runtime = BackendRuntime::new(&serving_config);

        let ids = proxima_tokenizer::encode_with_bos_eos(
            prompt,
            &self.vocab,
            wants_bos(&self.vocab),
            self.vocab.add_eos_token().unwrap_or(false),
        )?;
        apply_serving_config(&serving_config, ids.len())?;
        let inputs = build_position_inputs(
            &ids,
            0,
            self.architecture.head_dim,
            self.architecture.rope_freq_base,
            self.architecture.rms_epsilon,
            self.architecture_impl
                .as_ref()
                .and_then(|architecture| architecture.rope_freq_factors(&self.weights)),
        );

        // The SAME program-derived cache-leaf-name/step_inputs assembly
        // `Self::run_decode_loop_observed_seeded` calls -- before this,
        // this method hard-coded `kv_cache.{layer}.{k_even,k_odd,v}` and
        // never ran `Architecture::step_inputs` at all, so a foreign
        // architecture with differently-named cache leaves (or a leaf only
        // `step_inputs` feeds) surfaced `InteropError::UnboundInputName`
        // the moment a caller tapped an interior node here instead of
        // decoding. `cached_len: 0`, `new_start: 0`, `new_count:
        // ids.len()` -- this is always a one-shot forward from an empty
        // cache over the WHOLE prompt (this method's own doc).
        let (cache_names, layer_row_widths) = self.declared_layer_cache_names_and_widths()?;
        let layer_caches = self.fresh_layer_caches(&cache_names, &layer_row_widths);
        let mut kv_pad_scratch: Vec<KvPadScratch> = self
            .layer_roots
            .iter()
            .map(|_| KvPadScratch::new())
            .collect();
        let mut qwen35_dense_pad_scratch: Vec<Qwen35DenseAttentionPadScratch> = self
            .layer_roots
            .iter()
            .map(|_| Qwen35DenseAttentionPadScratch::new())
            .collect();
        let mut step_input_scratch: Vec<StepInput> = Vec::new();

        let mut named_blocks: Vec<(&str, QuantizedBlock)> = Vec::with_capacity(
            self.weights.owned.len()
                + self.weights.packed.len()
                + self.weights.packed_owned.len()
                + 6
                + cache_names.len() * 4,
        );
        for (name, data) in &self.weights.owned {
            named_blocks.push((name.as_str(), QuantizedBlock::Float32(data.as_slice())));
        }
        for (name, block) in &self.weights.packed {
            named_blocks.push((name.as_str(), *block));
        }
        for (name, bytes, kind) in &self.weights.packed_owned {
            let block =
                crate::bind::as_block(*kind, bytes).ok_or(InteropError::UnsupportedCodec { codec: *kind })?;
            named_blocks.push((name.as_str(), block));
        }
        // `mistral_cached_forward_program_with_experts`'s own `cached_len`
        // `Op::Input` (ROW 404/405's runtime bound the fused Metal
        // `CachedAttention` kernel reads) is present on every program this
        // method evaluates, fresh-KV or not -- this is always a one-shot
        // forward from an empty cache (this method's own doc), so `0.0` is
        // the only correct value.
        let cached_len_scalar = [0.0f32];
        // Same `lm_head_row` leaf the decode loop feeds -- this one-shot
        // forward's own `ids` IS the whole prompt, so `ids.len() - 1` is
        // its last row, matching `Self::forward_logits_on_backend`'s own
        // "last prompt position" doc. A caller of
        // `Self::forward_node_values_on_backend` wanting the FULL
        // per-position logits (not just the last row) has no opt-in path
        // yet -- residual, not fixed here.
        let lm_head_row_scalar = [(ids.len() - 1) as f32];
        let kv_bound_extent = kv_extent(ids.len(), usize::MAX, serving_config.kv_bucket_tokens);

        let symbols = self.push_step_named_blocks(
            &inputs,
            &cached_len_scalar,
            &lm_head_row_scalar,
            &ids,
            0,
            ids.len(),
            &cache_names,
            &layer_caches,
            &layer_row_widths,
            kv_bound_extent,
            &mut kv_pad_scratch,
            &mut qwen35_dense_pad_scratch,
            &mut step_input_scratch,
            &mut named_blocks,
            self.single_position_step,
        )?;

        let resident_names: BTreeSet<&str> = self.resident_names();

        // A one-shot diagnostic forward, not a decode step -- no
        // `ExpertSlab::begin_step` `StepGuard` scopes this call, so it
        // reads whatever is currently paged (this checkpoint's own aliased
        // stack, absent a caller ever paging one) exactly as
        // `run_reduce_with_quantized_weights` always has: `None` here is
        // not "experts disabled", it is "no per-step snapshot applies to a
        // call outside the decode loop".
        // The qwen35moe diagnostic can request an interior routed node, so
        // keep it on the partition-isolated seam. Other one-shot forwards
        // retain the ordinary evaluator and its normal cache bookkeeping.
        let evaluated = if self.architecture_impl.as_ref().is_some_and(|architecture| {
            architecture.ffn_routing() == crate::architecture::FfnRouting::Routed
        }) {
            runtime.evaluate_segment(
                &self.program,
                &symbols,
                &named_blocks,
                node_ids,
                &resident_names,
                None,
            )?
        } else {
            runtime.evaluate(
                &self.program,
                &symbols,
                &named_blocks,
                node_ids,
                &resident_names,
                None,
            )?
        };

        node_ids
            .iter()
            .map(|node| {
                evaluated
                    .get(*node)
                    .map(|(data, _shape)| data.to_vec())
                    .ok_or(InteropError::MissingEvaluatedNode { node: *node })
            })
            .collect()
    }

    /// [`Self::forward_node_values`] against `[Self::logits_root]`, sliced
    /// to just the LAST prompt position -- the convenience a caller
    /// cross-checking a decoded token's own logits (not an intermediate
    /// layer) wants. See that method's own doc for why a raw logit vector,
    /// not a sampled token, is what this crate's cross-oracle diagnostics
    /// need.
    ///
    /// # Errors
    ///
    /// Whatever [`Self::forward_node_values`] can fail with.
    pub fn forward_logits(&self, prompt: &str) -> Result<Vec<f32>, InteropError> {
        self.forward_logits_on_backend(prompt, 0)
    }

    /// [`Self::forward_logits`] with the backend left open, same
    /// [`Self::forward_node_values_on_backend`]-vs-[`Self::forward_node_values`]
    /// relationship: [`Self::forward_logits`] always pins `gpu_layers: 0`
    /// (CPU), this lets a caller ask for the Metal backend's own one-shot
    /// logits instead.
    ///
    /// # Errors
    ///
    /// Whatever [`Self::forward_node_values_on_backend`] can fail with.
    pub fn forward_logits_on_backend(
        &self,
        prompt: &str,
        gpu_layers: i32,
    ) -> Result<Vec<f32>, InteropError> {
        let prompt_ids = proxima_tokenizer::encode_with_bos_eos(
            prompt,
            &self.vocab,
            wants_bos(&self.vocab),
            self.vocab.add_eos_token().unwrap_or(false),
        )?;
        if self.single_position_step && prompt_ids.len() > 1 {
            let serving_config = supported_serving_config(
                gpu_layers,
                #[cfg(all(feature = "metal", target_os = "macos"))]
                omega::MathMode::default(),
            );
            let mut runtime = BackendRuntime::new(&serving_config);
            let mut captured = Vec::new();
            let mut logits_sink = LogitsSink::Collect(&mut captured);
            let mut on_token = |_event: TokenEvent<'_>| ControlFlow::Continue(());
            self.run_decode_loop_observed(
                prompt,
                1,
                &serving_config,
                &mut runtime,
                None,
                &mut logits_sink,
                &mut on_token,
            )?;
            return captured.pop().ok_or(InteropError::EmptyLogits);
        }
        // `forward_node_values_on_backend` re-tokenizes `prompt` itself;
        // this binding survives only for the `instrument`-gated debug
        // event below (`prompt_tokens`) now that the last-row slice no
        // longer needs a token count to index with.
        #[cfg(feature = "instrument")]
        let ids = proxima_tokenizer::encode_with_bos_eos(
            prompt,
            &self.vocab,
            wants_bos(&self.vocab),
            self.vocab.add_eos_token().unwrap_or(false),
        )?;
        let mut values =
            self.forward_node_values_on_backend(prompt, &[self.logits_root], gpu_layers)?;
        let logits = values.remove(0);
        let vocab_size = self.architecture.vocab as usize;
        // `logits_root` must be the `lm_head_row`-gathered LAST row only
        // (`crate::architecture`'s doc on `BoundProgram::logits_root`) --
        // exactly one row of `vocab_size`. A foreign `Architecture` that
        // hands back the full `[new_count, vocab]` buffer is rejected here
        // rather than silently sampled at row 0.
        if logits.len() != vocab_size {
            return Err(InteropError::LogitsShapeMismatch {
                expected_rows: 1,
                found_rows: logits.len() / vocab_size,
                vocab: vocab_size,
            });
        }
        let last_position = logits[..vocab_size].to_vec();

        #[cfg(feature = "instrument")]
        {
            let mut ranked: Vec<usize> = (0..last_position.len()).collect();
            ranked.sort_by(|left, right| {
                last_position[*right]
                    .total_cmp(&last_position[*left])
                    .then_with(|| left.cmp(right))
            });
            let top1_token = ranked[0] as u64;
            let top1_logit = f64::from(last_position[ranked[0]]);
            debug!(
                prompt_tokens = ids.len() as u64,
                top1_token,
                top1_logit,
                "computed one-shot forward logits for cross-oracle comparison"
            );
        }

        Ok(last_position)
    }
}

#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
pub(super) fn use_metal_output_placements(
    has_recurrent_state: bool,
    monolithic_all_low: bool,
) -> bool {
    // `evaluate_with_placements` has carried `expert_sources` since it grew
    // `execute_plan_named_with_placements_and_expert_sources` -- excluding
    // routed-expert steps here (ROW 549) used to send every qwen35moe
    // (MoE + GDN) decode step through the unplaced `runtime.evaluate`
    // fallback, so the fused `GatedDeltaNet` state_out
    // (`omega::metal::encode_op`) never found its caller-owned buffer in
    // `device_buffers` and silently discarded the recurrent state every
    // step.
    monolithic_all_low || has_recurrent_state
}

pub(super) fn qwen35moe_pre_gather_enabled(configured: bool, routed_experts: bool) -> bool {
    configured && routed_experts
}

pub(super) fn qwen35moe_admit_low_copy(source: Codec, target: Codec) -> bool {
    source == target
}

#[cfg(any(test, feature = "metal"))]
pub(super) fn qwen35moe_monolithic_all_low_enabled(
    pre_gather: bool,
    uses_gpu: bool,
    requested: bool,
    _step: usize,
) -> bool {
    pre_gather && uses_gpu && requested
}

#[cfg(any(test, feature = "metal"))]
pub(super) fn should_release_monolithic_sources(
    monolithic_all_low: bool,
    retain_for_warmup: bool,
) -> bool {
    monolithic_all_low && !retain_for_warmup
}

pub(super) fn map_expert_sources_to_segment<'source>(
    layer: usize,
    source_program: &[Op],
    segment_program: &[Op],
    expert_sources: &BTreeMap<NodeId, ExpertSource<'source>>,
) -> Result<BTreeMap<NodeId, ExpertSource<'source>>, InteropError> {
    let mut mapped = BTreeMap::new();
    for (source_node, source) in expert_sources {
        let source_name = source_program
            .get(source_node.0 as usize)
            .map(Op::name)
            .ok_or_else(|| InteropError::PreGatherExecutionUnsupported {
                architecture: String::from("qwen35moe"),
                reason: alloc::format!(
                    "layer {layer} expert source node {} is outside the source program",
                    source_node.0
                ),
            })?;
        let mut found = false;
        for (position, operation) in segment_program.iter().enumerate() {
            if operation.name() == source_name {
                mapped.insert(NodeId(position as u32), *source);
                found = true;
            }
        }
        if !found {
            return Err(InteropError::PreGatherExecutionUnsupported {
                architecture: String::from("qwen35moe"),
                reason: alloc::format!(
                    "layer {layer} expert source node {} named {source_name:?} is absent from the gather segment",
                    source_node.0
                ),
            });
        }
    }
    Ok(mapped)
}
