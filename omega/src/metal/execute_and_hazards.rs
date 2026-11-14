use super::*;
#[cfg(feature = "metal-output-placement")]
use proxima_numeric::{FormatCodec, NumericError};

/// Resolves a program into a reusable [`Plan`]. `blocks` is read for its
/// CODECS and shapes only — the data is not captured, so the same plan runs
/// against fresh block data every call.
///
/// # Errors
/// Propagates inference, binding, dtype-gate and block-shape failures.
pub fn plan(
    program: &[Op],
    symbols: &[u64],
    blocks: &[QuantizedBlock<'_>],
    outputs: &[NodeId],
    numeric_policy: NumericPolicy,
) -> Result<Plan, MetalError> {
    plan_with_placed_inputs(
        program,
        symbols,
        blocks,
        outputs,
        numeric_policy,
        &[],
        (true, false),
    )
}

/// `options` is `(fuse_cached_attention, wavefront_schedule)`; the second is
/// the plan-time level schedule (`wavefront_schedule.rs`), off keeps program order.
pub(super) fn plan_with_placed_inputs(
    program: &[Op],
    symbols: &[u64],
    blocks: &[QuantizedBlock<'_>],
    outputs: &[NodeId],
    numeric_policy: NumericPolicy,
    placed_input_nodes: &[NodeId],
    options: (bool, bool),
) -> Result<Plan, MetalError> {
    let (fuse_cached_attention, wavefront_schedule) = options;
    #[cfg(feature = "instrument")]
    let prepare_started = read_ticks();
    let prepared = prepare(
        program,
        symbols,
        blocks,
        outputs,
        numeric_policy,
        placed_input_nodes,
        (fuse_cached_attention, wavefront_schedule),
    )?;
    #[cfg(feature = "instrument")]
    {
        counter!(PREPARE_CALLS, 1);
        counter!(PREPARE_TICKS, elapsed_ticks(prepare_started));
        emit_step_phase("prepare", prepare_started.as_raw(), read_ticks().as_raw());
    }
    Ok(plan_from_prepared(
        program,
        symbols,
        prepared,
        numeric_policy,
        crate::msl::ambient_knobs(),
    ))
}

pub(super) fn plan_from_prepared(
    program: &[Op],
    symbols: &[u64],
    prepared: Prepared,
    numeric_policy: NumericPolicy,
    emit_knobs: EmitKnobs,
) -> Plan {
    // `prepare` already built this attribution once, off the same `blocks`
    // argument, checked count- and shape-consistent against `block_nodes`
    // (ROW 327) -- cloning it here instead of re-zipping `block_nodes`
    // against `blocks` a second time is what keeps this field and
    // `prepared`'s own from being two independent computations that could
    // silently drift apart.
    let packed_operands = prepared.packed_operands.clone();
    let block_dtypes = prepared
        .block_nodes
        .iter()
        .map(|node| gpu_dtype(program, &prepared.index_nodes, &prepared.resolved, *node))
        .collect();
    Plan {
        configured_constraints: None,
        configured_input_contract: Vec::new(),
        scoped_reservation: RefCell::new(None),
        scoped_resource_rows: RefCell::new(Vec::new()),
        program: program.to_vec(),
        prepared,
        packed_operands,
        block_dtypes,
        resident_nodes: BTreeSet::new(),
        math_mode: numeric_policy_as_metal_math_mode(numeric_policy),
        numeric_policy,
        emit_knobs,
        dense_mma_input_layout: None,
        attention_mma_selection: crate::msl::AttentionMmaSelection::Legacy,
        attention_row_schedule: crate::msl::AttentionRowSchedule::legacy(),
        dispatch_type: DispatchType::default(),
        command_buffer_chunks: 0,
        command_buffer_chunks_decode_shaped: false,
        bind_row_count: plan_bind_row_count(symbols),
        #[cfg(feature = "instrument")]
        encoder_split_at: None,
        #[cfg(feature = "metal-plan-stable-buffers")]
        arena: core::cell::OnceCell::new(),
        #[cfg(feature = "metal-plan-stable-buffers")]
        uniforms: core::cell::OnceCell::new(),
        #[cfg(feature = "metal-plan-stable-buffers")]
        attention_scratch: core::cell::OnceCell::new(),
        resolved_steps: RefCell::new(None),
        #[cfg(feature = "metal-horizontal-merge")]
        merged: RefCell::new(None),
        hazard_state: RefCell::new(HazardState::new()),
        device_buffers: RefCell::new(BTreeMap::new()),
        block_identity: RefCell::new(Vec::new()),
        chunk_status_buffers: RefCell::new(Vec::new()),
        chunk_status_diagnostics: RefCell::new(Vec::new()),
    }
}

/// Same contract as [`proxima_tensor::cpu::evaluate`], and returns the same
/// [`Evaluated`] type — a CPU run and a Metal run report the identical
/// shape, so a parity test compares them directly with no adapter on either
/// side (see `Evaluated`'s own doc). `blocks` binds [`Op::Input`] inputs
/// positionally, `outputs` selects which nodes to return data for (the root
/// only, if empty).
pub fn execute(
    program: &[Op],
    symbols: &[u64],
    blocks: &[QuantizedBlock<'_>],
    outputs: &[NodeId],
    numeric_policy: NumericPolicy,
) -> Result<Evaluated, MetalError> {
    let resolved_plan = plan(program, symbols, blocks, outputs, numeric_policy)?;
    execute_plan(&resolved_plan, blocks)
}

/// Runs an already-resolved [`Plan`] against fresh block data. This is the
/// serving-loop entry point: the plan is built once, this is called per
/// token, and none of `infer`/`bind`/codec-resolution happens here.
///
/// # Errors
/// Propagates block-codec and Metal driver failures.
pub fn execute_plan(plan: &Plan, blocks: &[QuantizedBlock<'_>]) -> Result<Evaluated, MetalError> {
    execute_plan_inner(plan, blocks, &BTreeMap::new())
}

pub(super) fn execute_plan_inner(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
    expert_buffers: &BTreeMap<NodeId, ExpertSourceBuffers>,
) -> Result<Evaluated, MetalError> {
    plan.validate_configured_inputs(blocks)?;
    if plan.has_constraints() {
        if !expert_buffers.is_empty() {
            return Err(MetalError::UnsupportedConstraint {
                field: "constraints",
                reason: "expert buffers have no bounded execution contract",
            });
        }
        plan.preflight_constraints(blocks)?;
        plan.scoped_resource_rows.borrow_mut().clear();
    }
    let prepared = &plan.prepared;
    let packed_operands = &plan.packed_operands;

    // Partitioned routed programs may retain the same named expert input at
    // more than one dense position. Alias the staged descriptor by name at
    // the prepared-plan boundary so renumbering cannot leave the gather
    // reading an ordinary (unbound) input.
    let mut effective_expert_buffers = expert_buffers.clone();
    for (source_node, buffers) in expert_buffers {
        let Some(source_name) = plan.program[source_node.0 as usize].name() else {
            continue;
        };
        for (position, operation) in plan.program.iter().enumerate() {
            if operation.name() == Some(source_name) {
                effective_expert_buffers
                    .entry(NodeId(position as u32))
                    .or_insert_with(|| buffers.clone());
            }
        }
    }
    debug!(
        expert_source_nodes = ?effective_expert_buffers.keys().collect::<Vec<_>>(),
        "effective_expert_source_nodes"
    );
    for index in 0..plan.program.len().min(24) {
        debug!(op = index as u64, name = ?plan.program[index].name(), "plan_op");
    }

    let (device, queue) = device_and_queue()?;

    let mut device_buffers: BTreeMap<NodeId, DeviceBuffer> = BTreeMap::new();
    #[cfg(feature = "instrument")]
    let mut ordinary_upload_bytes = 0usize;
    #[cfg(feature = "instrument")]
    let mut ordinary_upload_blocks = 0usize;
    #[cfg(feature = "instrument")]
    let block_upload_started = read_ticks();
    for (node, block, dtype) in ordinary_block_uploads(
        &prepared.block_nodes,
        &prepared.live_block_inputs,
        blocks,
        &plan.block_dtypes,
        |node| effective_expert_buffers.contains_key(&node),
    ) {
        let resident_name = resident_name(plan, node);
        if let Some((buffer, offset)) = cross_plan_block_reuse(resident_name, &block)? {
            plan.record_scoped_buffer("input", node, &buffer);
            device_buffers.insert(node, (buffer, offset));
            continue;
        }
        #[cfg(feature = "instrument")]
        {
            ordinary_upload_bytes = ordinary_upload_bytes.saturating_add(block_byte_len(&block));
            ordinary_upload_blocks += 1;
        }
        #[cfg(feature = "instrument")]
        {
            counter!(BLOCK_UPLOAD_CALLS, 1);
            // fires unconditionally, before the upload-path match below --
            // this is what a block was OFFERED for upload, regardless of
            // which terminal path (no-copy bind, checkpoint-offset bind, or
            // a real copy) actually served it. See `BLOCK_COPIED_BYTES` /
            // `BLOCK_NOCOPY_BOUND_BYTES` / `BLOCK_OFFSET_BOUND_BYTES` for the
            // per-path split; `BLOCK_COPIED + BLOCK_NOCOPY_BOUND +
            // BLOCK_OFFSET_BOUND == BLOCK_OFFERED` on every step is the
            // partition identity that proves no path is uninstrumented.
            counter!(BLOCK_OFFERED_BYTES, block_byte_len(&block) as u64);
        }
        let buffer = match block {
            QuantizedBlock::Float32(data) => upload_block(
                &device,
                data,
                node,
                dtype,
                plan.program[node.0 as usize].name(),
                resident_name,
            )?,
            QuantizedBlock::Int32(data) => {
                upload_block_int32_as_float(&device, data, resident_name)?
            }
            // `Float16`/`BFloat16` upload their bytes UNCHANGED, same as
            // every packed codec above -- there is no host-side narrowing
            // step (unlike `upload_block`'s `Float32 -> Float16` path,
            // which narrows a caller's `&[f32]`): a `Float16` weight's on-
            // disk bytes already ARE its device buffer's bytes (native
            // `half`), and a `BFloat16` weight's bytes are widened entirely
            // on the GPU at the read (`msl::layout_render`), never on the
            // host.
            // `Q3_K` uploads its raw super-block bytes unchanged, same as
            // every other packed codec below -- `msl::Codec::Q3K`'s
            // own unpack kernel (`q3k_element`) reads them at the GPU side.
            QuantizedBlock::Packed { bytes, .. } => {
                upload_packed_bytes(&device, bytes, resident_name)?
            }
            QuantizedBlock::PackedStrided { .. } => {
                (upload_strided_runs(&device, &block, resident_name)?, 0)
            }
        };
        plan.record_scoped_buffer("input", node, &buffer.0);
        device_buffers.insert(node, buffer);
    }
    #[cfg(feature = "instrument")]
    debug!(
        blocks = ordinary_upload_blocks as u64,
        bytes = ordinary_upload_bytes as u64,
        "metal_ordinary_uploads"
    );
    #[cfg(feature = "instrument")]
    counter!(BLOCK_UPLOAD_TICKS, elapsed_ticks(block_upload_started));

    let command_buffer = new_labeled_command_buffer(&queue, "omega.single.execute_plan_inner")
        .ok_or_else(|| MetalError::CompileFailed {
            log: "command queue refused to hand out a command buffer".to_string(),
        })?;

    // ONE `MTLComputeCommandEncoder` for the whole program, opened here and
    // `endEncoding()`d once below, after every op has been encoded into it --
    // not one per op. `computeCommandEncoder()` (no dispatch-type argument)
    // defaults to `MTLDispatchTypeSerial` (Apple's own documented default,
    // confirmed against `objc2-metal-0.3.2`'s own `MTLDispatchType` doc:
    // "Command encoder dispatches are executed in dispatched order"), the
    // IDENTICAL dispatch type every one of the former per-op encoders already
    // used -- this change narrows encoder COUNT, not dispatch semantics.
    // Ordering and hazard-tracked visibility between two dispatches in one
    // serial encoder is Metal's documented behavior for tracked resources
    // (every buffer here is `storageModeShared`, never
    // `HazardTrackingModeUntracked` -- see the module doc's "Execution
    // model"), so this is the SAME correctness argument that section makes
    // for this one encoder's dispatches: they were always ordered and
    // hazard-tracked relative to each other, encoder boundaries or not.
    let encoder = EncoderGuard::new(command_buffer.computeCommandEncoder().ok_or_else(|| {
        MetalError::CompileFailed {
            log: "command buffer refused to hand out a compute encoder".to_string(),
        }
    })?);

    // `metal-buffer-pool` reclaim bookkeeping: which node ids are op OUTPUTS
    // (never block inputs) and their `(bucket, dtype)` pool key -- built once,
    // off `prepared.resolved`'s own node ids and the same `bound_output_len`/
    // `dtype` `allocate_buffer` sizes from inside `encode_op`, run through the
    // IDENTICAL `pool_bucket` function `allocate_buffer` itself uses. That
    // identity is what keeps a reclaimed buffer's stored key equal to its own
    // real Metal capacity -- see `OUTPUT_BUFFER_POOL`'s doc for the full
    // invariant and the liveness argument this feeds.
    #[cfg(feature = "metal-buffer-pool")]
    let output_meta: BTreeMap<NodeId, (usize, DType)> = prepared
        .resolved
        .iter()
        .map(|bound| {
            let byte_length = bound_output_len(bound).max(1) * device_element_bytes(bound.dtype);
            (bound.node, (pool_bucket(byte_length), bound.dtype))
        })
        .collect();
    #[cfg(feature = "metal-buffer-pool")]
    let mut reclaim_stash: Vec<(MetalBuffer, usize, DType)> = Vec::new();

    // pipelines live in this thread's `PIPELINE_CACHE`, not here: see that
    // static's own doc for why per-call was the defect.
    // (bound op, its fault buffer, gather count) for every op that gathered —
    // checked only after the single end-of-program wait below, since a fault
    // buffer is not CPU-visible until the command buffer it was written in
    // completes. See the module doc's "Gather fault reporting" section.
    let mut pending_faults: Vec<PendingFault<'_>> = Vec::new();
    let mut route_compactions = RouteCompactions::new();
    debug!(
        prepared_nodes = ?prepared.resolved.iter().map(|bound| bound.node).collect::<Vec<_>>(),
        "metal_expert_prepared_nodes"
    );
    for (position, retired) in prepared.retires.iter().enumerate() {
        if retired.contains(&NodeId(16))
            || retired.contains(&NodeId(18))
            || retired.contains(&NodeId(22))
        {
            debug!(
                position = position as u64,
                nodes = ?retired,
                "metal_expert_retirement_node16_or_18"
            );
        }
    }
    for (position, bound) in prepared.resolved.iter().enumerate() {
        if bound.all_read_sources().any(|(source, _, lookup)| {
            *source == NodeId(22)
                || lookup
                    .as_ref()
                    .is_some_and(|item| item.indices == NodeId(22))
        }) {
            debug!(
                position = position as u64,
                bound = ?bound.node,
                "metal_expert_node22_read_at_position"
            );
        }
        if matches!(bound.node, NodeId(16) | NodeId(18)) {
            debug!(
                position = position as u64,
                node = ?bound.node,
                retires = ?prepared.retires[position],
                "metal_expert_retirement"
            );
        }
    }
    for (position, bound) in prepared.resolved.iter().enumerate() {
        debug!(
            node = ?bound.node,
            kind = bound.kind.name(),
            operands = ?bound
                .operands()
                .iter()
                .map(|(node, _, lookup)| (
                    *node,
                    lookup.is_some(),
                    plan.program.get(node.0 as usize).and_then(Op::name)
                ))
                .collect::<Vec<_>>(),
            "resolved_bound"
        );
        #[cfg(feature = "instrument")]
        let expert_buffers_started = read_ticks();
        let expert_buffers = expert_buffers_for(bound, &effective_expert_buffers)?;
        #[cfg(feature = "instrument")]
        {
            counter!(EXPERT_BUFFERS_LOOKUP_CALLS, 1);
            counter!(
                EXPERT_BUFFERS_LOOKUP_TICKS,
                elapsed_ticks(expert_buffers_started)
            );
        }
        let fault = encode_op(
            &device,
            &encoder,
            &mut device_buffers,
            bound,
            packed_operands,
            None,
            None,
            None,
            plan.math_mode,
            plan.numeric_policy,
            &plan.emit_knobs,
            plan.attention_mma_selection,
            plan.attention_row_schedule,
            None,
            #[cfg(feature = "instrument")]
            1,
            None,
            None,
            expert_buffers,
            Some(&mut route_compactions),
        )?;
        if plan.has_constraints() {
            if let Some((buffer, _)) = device_buffers.get(&bound.node) {
                plan.record_scoped_buffer("output", bound.node, buffer);
            }
            let uniform_bytes = pack_uniforms(bound, plan.numeric_policy, &plan.emit_knobs)?;
            UNIFORM_BUFFERS.with(|cache| {
                if let Some((buffer, _)) = cache.borrow().get(&uniform_bytes) {
                    plan.record_scoped_buffer("uniform", bound.node, buffer);
                }
            });
        }
        if let Some((fault_buffer, gathers)) = fault {
            pending_faults.push((bound, fault_buffer, gathers));
        }
        if matches!(bound.node, NodeId(22) | NodeId(30)) {
            debug!(
                node = ?bound.node,
                has_buffer = device_buffers.contains_key(&bound.node),
                buffer_keys = ?device_buffers.keys().copied().collect::<Vec<_>>(),
                "metal_expert_post_encode"
            );
        }
        // `metal-buffer-pool` off: identical to this function before the
        // feature existed -- a retired buffer is looked up once and dropped.
        // `metal-buffer-pool` on: same lookup-and-remove, but an op-OUTPUT
        // buffer (per `output_meta`) is ALSO cloned into `reclaim_stash`
        // rather than only dropped -- the clone is not pushed into the pool
        // until after this call's `waitUntilCompleted` below, so nothing here
        // hands a still-pending buffer back out early.
        #[cfg(feature = "instrument")]
        let retire_scan_started = read_ticks();
        #[cfg(not(feature = "metal-buffer-pool"))]
        for retired in &prepared.retires[position] {
            // gather index buffers are tiny, and keeping them until the
            // command completes avoids retiring a computed index before a
            // backend binding that references it through `Binding::Indices`.
            if prepared.index_nodes.contains(retired) {
                continue;
            }
            // `prepared.last_reader[retired] == position` is the SAME
            // last-use fact `prepared.retires[position]` was already built
            // from (both trace to `node_last_reader`'s one
            // `all_read_sources()` walk, `proxima_tensor::bind`'s own
            // `walk_last_reads`) -- a member of `retires[position]` cannot
            // fail this check, so this is a proof the retirement is safe,
            // not a second independent scan. One array read replaces the
            // per-op `iter().any()` forward scan over every remaining
            // resolved op (and the current op's own operand list) that used
            // to re-derive the identical fact here on every token.
            if prepared.last_reader[retired.0 as usize] != position as u32 {
                continue;
            }
            device_buffers.remove(retired);
        }
        #[cfg(feature = "metal-buffer-pool")]
        for retired in &prepared.retires[position] {
            if prepared.index_nodes.contains(retired) {
                continue;
            }
            if let Some((buffer, _offset)) = device_buffers.remove(retired)
                && let Some(&(bucket, dtype)) = output_meta.get(retired)
            {
                reclaim_stash.push((buffer, bucket, dtype));
            }
        }
        #[cfg(feature = "instrument")]
        {
            counter!(RETIRE_SCAN_CALLS, prepared.retires[position].len() as u64);
            counter!(RETIRE_SCAN_TICKS, elapsed_ticks(retire_scan_started));
        }
    }
    encoder.finish();

    #[cfg(feature = "instrument")]
    let gpu_exec_started = read_ticks();
    // whole-program single buffer -- `prepared.resolved` is exactly what the
    // loop above encoded, position for position, so the diagnostics carry
    // the real dispatch count and op-label range instead of the untracked
    // `BufferDiagnostics::single` shape a bare `commit_and_wait` would report.
    commit_and_wait(
        &command_buffer,
        BufferDiagnostics {
            chunk_index: 0,
            chunk_count: 1,
            dispatch_count: prepared.resolved.len() as u32,
            first_op_label: resolved_op_label(&prepared.resolved, 0),
            last_op_label: resolved_op_label(
                &prepared.resolved,
                prepared.resolved.len().saturating_sub(1),
            ),
            encoder_status_requested: encoder_error_status_requested(),
        },
    )?;
    #[cfg(feature = "instrument")]
    {
        counter!(GPU_EXEC_CALLS, 1);
        counter!(GPU_EXEC_TICKS, elapsed_ticks(gpu_exec_started));
    }

    check_pending_faults(pending_faults)?;

    // Everything still in `device_buffers` at this point (never removed by
    // the retirement loop above -- e.g. the root output and any other
    // `effective_outputs` node) is now also safe to reclaim: the single
    // command buffer this whole program ran in has already completed via
    // `waitUntilCompleted` above. Cloned, not moved, so `finish` below still
    // reads the same buffers to copy their bytes out.
    #[cfg(feature = "metal-buffer-pool")]
    for (node, (buffer, _offset)) in &device_buffers {
        if let Some(&(bucket, dtype)) = output_meta.get(node) {
            reclaim_stash.push((buffer.clone(), bucket, dtype));
        }
    }

    let evaluated = finish(plan, &device_buffers, &BTreeSet::new(), None)?;

    // `OUTPUT_POOL_MAX_PER_BUCKET`: a buffer beyond the cap for its
    // `(bucket, dtype)` slot is dropped here (ordinary `Retained` drop, same
    // as the `metal-buffer-pool`-off arm always did for every buffer) rather
    // than retained -- see `OUTPUT_POOL_MAX_PER_BUCKET`'s own doc for why
    // this is a safety net, not the primary bound.
    #[cfg(feature = "metal-buffer-pool")]
    OUTPUT_BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        for (buffer, bucket, dtype) in reclaim_stash {
            let slot = pool.entry((bucket, dtype)).or_default();
            if slot.len() < OUTPUT_POOL_MAX_PER_BUCKET {
                slot.push(buffer);
            }
        }
    });

    Ok(evaluated)
}

/// Produces exactly the named-block inputs that retain their ordinary device
/// buffer binding. A substituted expert source has its own mixed payload and
/// descriptor buffers, so retaining its original stack here would allocate a
/// second device copy that no emitted mixed-expert kernel can read.
pub(super) fn ordinary_block_uploads<'plan, 'block, Substituted>(
    block_nodes: &'plan [NodeId],
    live_block_inputs: &'plan [bool],
    blocks: &'plan [QuantizedBlock<'block>],
    block_dtypes: &'plan [DType],
    is_substituted: Substituted,
) -> impl Iterator<Item = (NodeId, QuantizedBlock<'block>, DType)> + 'plan
where
    Substituted: Fn(NodeId) -> bool + 'plan,
{
    block_nodes
        .iter()
        .copied()
        .zip(live_block_inputs.iter().copied())
        .zip(blocks.iter().copied())
        .zip(block_dtypes.iter().copied())
        .filter_map(move |(((node, live), block), dtype)| {
            (live && !is_substituted(node)).then_some((node, block, dtype))
        })
}

pub(super) fn expert_buffers_for<'a>(
    bound: &BoundOp,
    expert_buffers: &'a BTreeMap<NodeId, ExpertSourceBuffers>,
) -> Result<Option<&'a ExpertSourceBuffers>, MetalError> {
    let mut matched = expert_buffers.iter().filter_map(|(node, buffers)| {
        bound
            .operands()
            .iter()
            .any(|(operand, _, _lookup)| *operand == *node)
            .then_some((*node, buffers))
    });
    let Some((first_node, first_buffers)) = matched.next() else {
        debug!(
            bound = ?bound.node,
            kind = bound.kind.name(),
            source_nodes = ?expert_buffers.keys().collect::<Vec<_>>(),
            "expert_source_not_matched"
        );
        debug!(
            bound_operands = ?bound
                .operands()
                .iter()
                .map(|(node, _, lookup)| (*node, lookup.is_some()))
                .collect::<Vec<_>>(),
            "expert_source_bound_operands"
        );
        return Ok(None);
    };
    if matched.next().is_some() {
        return Err(MetalError::ExpertSourceUnsupported {
            node: first_node,
            reason: "one gathered operation cannot bind more than one expert source table",
        });
    }
    Ok(Some(first_buffers))
}

pub(super) fn stage_expert_source_reusing(
    device: &ProtocolObject<dyn MTLDevice>,
    node: NodeId,
    source: &proxima_tensor::cpu::ExpertSource<'_>,
    previous: Option<StagedExpertSource>,
) -> Result<StagedExpertSource, MetalError> {
    #[cfg(feature = "instrument")]
    let stage_ticks_started = read_ticks();
    let stage_started = Instant::now();
    let packed_arena = source.packed_arena();
    let (owned_payload_bytes, descriptors) = match packed_arena {
        Some(arena) => (
            None,
            selected_expert_arena_descriptors(node, source, arena)?,
        ),
        None => {
            let (payload, descriptors) = selected_expert_payloads(node, source)?;
            (Some(payload), descriptors)
        }
    };
    let payload_bytes = owned_payload_bytes
        .as_deref()
        .or_else(|| packed_arena.map(|arena| arena.bytes()))
        .ok_or(MetalError::ExpertSourceUnsupported {
            node,
            reason: "expert source produced no payload bytes",
        })?;
    let descriptor_records = descriptors;
    debug!(
        node = ?node,
        entries = source.entries().len() as u64,
        selected_experts = source
            .selected_expert_ids()
            .map_or(source.entries().len(), <[_]>::len) as u64,
        selected_payload_bytes = payload_bytes.len() as u64,
        "metal_expert_source"
    );
    if let Some(selected) = source.selected_expert_ids() {
        debug!(ids = ?selected, "metal_selected_expert");
        for expert in selected {
            if let Some(descriptor) = descriptor_records.get(*expert as usize) {
                debug!(
                    expert = descriptor.expert_index,
                    codec = ?descriptor.codec,
                    epoch = descriptor.epoch,
                    offset = descriptor.byte_offset,
                    length = descriptor.byte_length,
                    "metal_selected_descriptor"
                );
            }
        }
    }
    let descriptor_bytes = pack_expert_payload_descriptors(node, &descriptor_records)?;
    let previous_payload = previous
        .as_ref()
        .map(|staged| (&staged.buffers.payloads, staged.payload_alias_address));
    let all_expert_arena = packed_arena.is_some() && source.selected_expert_ids().is_none();
    #[cfg(feature = "instrument")]
    let payload_copy_started = read_ticks();
    let (payloads, payload_offset, payload_reused) = if all_expert_arena {
        let (buffer, offset) = upload_packed_bytes(device, payload_bytes, None)?;
        (buffer, offset, false)
    } else {
        reuse_or_upload_packed_bytes(device, payload_bytes, previous_payload)?
    };
    #[cfg(feature = "instrument")]
    if payload_reused {
        counter!(EXPERT_SOURCE_REUSE_COPY_BYTES, payload_bytes.len() as u64);
        counter!(
            EXPERT_SOURCE_REUSE_COPY_TICKS,
            elapsed_ticks(payload_copy_started)
        );
    }
    #[cfg(not(feature = "instrument"))]
    let _ = payload_reused;
    let (descriptors, _, _) = reuse_or_upload_packed_bytes(
        device,
        &descriptor_bytes,
        previous
            .as_ref()
            .map(|staged| (&staged.buffers.descriptors, staged.descriptor_alias_address)),
    )?;
    #[cfg(feature = "instrument")]
    {
        counter!(EXPERT_SOURCE_STAGE_CALLS, 1);
        counter!(EXPERT_SOURCE_STAGE_BYTES, payload_bytes.len() as u64);
        counter!(
            EXPERT_SOURCE_STAGE_TICKS,
            elapsed_ticks(stage_ticks_started)
        );
    }
    debug!(
        node = ?node,
        payload_bytes = payload_bytes.len() as u64,
        descriptor_bytes = descriptor_bytes.len() as u64,
        elapsed_us = stage_started.elapsed().as_micros() as u64,
        "metal_expert_source_staged"
    );
    for descriptor in descriptor_records.iter().take(8) {
        debug!(
            expert = descriptor.expert_index,
            codec = ?descriptor.codec,
            offset = descriptor.byte_offset,
            length = descriptor.byte_length,
            out_dim = descriptor.out_dim,
            in_dim = descriptor.in_dim,
            "metal_expert_descriptor"
        );
    }
    Ok(StagedExpertSource {
        payload_alias_address: payload_alias_address(payload_bytes),
        descriptor_alias_address: payload_alias_address(&descriptor_bytes),
        _payload_bytes: owned_payload_bytes
            .and_then(|payload| host_bytes_if_aliased(payload, &payloads)),
        _descriptor_bytes: host_bytes_if_aliased(descriptor_bytes, &descriptors),
        buffers: ExpertSourceBuffers {
            node,
            payloads,
            payload_offset,
            descriptors,
            descriptor_records: descriptor_records.to_vec(),
        },
    })
}

pub(super) fn host_bytes_if_aliased(bytes: Vec<u8>, buffer: &MetalBuffer) -> Option<Vec<u8>> {
    (!bytes.is_empty() && buffer.contents().as_ptr() as *const u8 == bytes.as_ptr())
        .then_some(bytes)
}

pub(super) fn reuse_or_upload_packed_bytes(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
    previous: Option<(&MetalBuffer, usize)>,
) -> Result<(MetalBuffer, usize, bool), MetalError> {
    // A page-aligned slice into an mmap is already a valid Metal no-copy
    // source. Reusing a prior copying buffer here would turn a bounded mapped
    // range back into a host-to-device copy whenever the routed layer changes.
    // Keep this path uncached: the command buffer completes before the mmap
    // borrow ends, and the next step may select a different range.
    if is_page_aligned(bytes.as_ptr().cast(), bytes.len()) {
        let (buffer, offset) = upload_packed_bytes(device, bytes, None)?;
        return Ok((buffer, offset, false));
    }
    if let Some((buffer, previous_alias_address)) = previous
        && buffer.length() >= bytes.len()
        && buffer.contents().as_ptr() as usize != previous_alias_address
    {
        let destination = buffer.contents().as_ptr().cast::<u8>();
        // SAFETY: a copying upload owns its storage, at least `bytes.len()` long, idle once the prior command buffer completed.
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), destination, bytes.len()) };
        #[cfg(feature = "instrument")]
        counter!(EXPERT_SOURCE_BUFFER_REUSES, 1);
        return Ok((buffer.clone(), 0, true));
    }
    let (buffer, offset) = upload_packed_bytes(device, bytes, None)?;
    Ok((buffer, offset, false))
}

pub(super) fn payload_alias_address(bytes: &[u8]) -> usize {
    if bytes.is_empty() {
        0
    } else {
        bytes.as_ptr() as usize
    }
}

pub(super) fn expert_source_signature(source: &proxima_tensor::cpu::ExpertSource<'_>) -> u64 {
    let mut signature = 1469598103934665603_u64;
    let mut mix = |value: u64| {
        signature ^= value;
        signature = signature.wrapping_mul(1099511628211);
    };
    for entry in source.entries() {
        let codec = match entry.block {
            QuantizedBlock::Packed {
                codec: Codec::Q2K,
                bytes: _,
            } => 1_u64,
            QuantizedBlock::Packed {
                codec: Codec::Q3K,
                bytes: _,
            } => 2_u64,
            QuantizedBlock::Packed {
                codec: Codec::Q4K,
                bytes: _,
            } => 3_u64,
            QuantizedBlock::Packed {
                codec: Codec::Q5K,
                bytes: _,
            } => 4_u64,
            QuantizedBlock::Packed {
                codec: Codec::Q6K,
                bytes: _,
            } => 5_u64,
            _ => 0,
        };
        mix(codec);
        mix(entry.out_dim as u64);
        mix(entry.in_dim as u64);
        mix(entry.epoch);
        if let Some(bytes) = entry.block.packed_bytes() {
            mix(bytes.len() as u64);
        } else {
            mix(0);
        }
    }
    if let Some(selected) = source.selected_expert_ids() {
        mix(selected.len() as u64);
        for expert in selected {
            mix(u64::from(*expert));
        }
    } else {
        mix(u64::MAX);
    }
    signature
}

/// Refuses the transitional host-staging path when it would upload at least
/// as many expert bytes as the checkpoint block it replaces. Such a table is
/// not an offload: it only rebuilds the full expert stack in transient memory.
pub(super) fn reject_non_reducing_expert_staging(
    node: NodeId,
    original: QuantizedBlock<'_>,
    source: &proxima_tensor::cpu::ExpertSource<'_>,
) -> Result<(), MetalError> {
    // Report an unsupported codec before the size guard below.  Otherwise a
    // larger Q5_K replacement can be rejected as "not reducing" first, which
    // hides the actual typed lowering contract from the caller.
    for entry in source.entries() {
        if !matches!(
            entry.block,
            QuantizedBlock::Packed {
                codec: Codec::Q2K,
                bytes: _
            } | QuantizedBlock::Packed {
                codec: Codec::Q3K,
                bytes: _
            } | QuantizedBlock::Packed {
                codec: Codec::Q4K,
                bytes: _
            } | QuantizedBlock::Packed {
                codec: Codec::Q6K,
                bytes: _
            }
        ) {
            return Err(MetalError::ExpertSourceUnsupported {
                node,
                reason: "mixed expert lowering only has Q2_K, Q3_K, Q4_K, and Q6_K decoders",
            });
        }
    }
    let original_bytes = original
        .packed_bytes()
        .ok_or(MetalError::ExpertSourceUnsupported {
            node,
            reason: "the replaced expert block must use packed bytes",
        })?
        .len();
    let entry_bytes = |expert_index: usize| {
        source
            .entries()
            .get(expert_index)
            .ok_or(MetalError::ExpertSourceUnsupported {
                node,
                reason: "selected expert ID is outside the source table",
            })?
            .block
            .packed_bytes()
            .map(<[u8]>::len)
            .ok_or(MetalError::ExpertSourceUnsupported {
                node,
                reason: "expert entries must use packed bytes",
            })
    };
    let staged_bytes = match source.selected_expert_ids() {
        Some(selected) => selected.iter().try_fold(0usize, |total, expert| {
            total.checked_add(entry_bytes(*expert as usize)?).ok_or(
                MetalError::ExpertSourceUnsupported {
                    node,
                    reason: "expert payload byte span overflowed",
                },
            )
        })?,
        None => (0..source.entries().len()).try_fold(0usize, |total, expert_index| {
            total.checked_add(entry_bytes(expert_index)?).ok_or(
                MetalError::ExpertSourceUnsupported {
                    node,
                    reason: "expert payload byte span overflowed",
                },
            )
        })?,
    };
    if source.selected_expert_ids().is_none()
        && source.packed_arena().is_none()
        && staged_bytes >= original_bytes
    {
        return Err(MetalError::ExpertSourceUnsupported {
            node,
            reason: "transient expert staging must reduce checkpoint-resident bytes",
        });
    }
    Ok(())
}

/// Executes a plan with a borrowed per-step expert table.
///
/// Ordinary plans retain [`execute_plan`]'s exact path. A supplied table
/// stages its packed bytes and descriptor table for the one command buffer,
/// then selected gathered reductions receive the codec-tagged kernel ABI.
/// This is an execution bridge, not HOBBIT's final residency arena: the
/// staging vectors preserve correctness for discontiguous entries while the
/// residency owner grows a persistent contiguous arena.
pub fn execute_plan_with_expert_sources(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
    expert_sources: &BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'_>>,
) -> Result<Evaluated, MetalError> {
    if expert_sources.is_empty() {
        return execute_plan(plan, blocks);
    }
    #[cfg(feature = "instrument")]
    let expert_stage_started = std::time::Instant::now();
    let buffers = stage_expert_sources(plan, blocks, expert_sources)?;
    #[cfg(feature = "instrument")]
    let upload_elapsed = expert_stage_started.elapsed();
    #[cfg(feature = "instrument")]
    debug!(
        expert_source_stage_ms = upload_elapsed.as_secs_f64() * 1e3,
        source_nodes = expert_sources.len() as u64,
        staged_nodes = buffers.len() as u64,
        "expert_source_stage"
    );
    #[cfg(feature = "instrument")]
    let dispatch_started = std::time::Instant::now();
    let result = execute_plan_inner(plan, blocks, &buffers);
    // I3 lifetime trace (ROW 501 HeteGen/FlexGen): upload and dispatch are
    // recorded as adjacent, non-overlapping intervals on the current serial
    // path -- this is the baseline an overlap gate must show has changed
    // before issuing a next-layer upload during this layer's dispatch.
    #[cfg(feature = "instrument")]
    {
        let dispatch_elapsed = dispatch_started.elapsed();
        trace!(
            upload_start_us = 0u64,
            upload_end_us = upload_elapsed.as_micros() as u64,
            dispatch_start_us = upload_elapsed.as_micros() as u64,
            dispatch_end_us = (upload_elapsed + dispatch_elapsed).as_micros() as u64,
            source_nodes = expert_sources.len(),
            "expert source upload/dispatch lifetime interval"
        );
    }
    result
}

pub(super) fn stage_expert_sources(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
    expert_sources: &BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'_>>,
) -> Result<BTreeMap<NodeId, ExpertSourceBuffers>, MetalError> {
    if !expert_sources.is_empty() {
        plan.reject_bounded_extension()?;
    }
    plan.validate_configured_inputs(blocks)?;
    let plan_identity = core::ptr::from_ref(plan) as usize;
    for (node, source) in expert_sources {
        let block_position = plan
            .prepared
            .block_nodes
            .iter()
            .position(|candidate| candidate == node)
            .ok_or(MetalError::ExpertSourceUnsupported {
                node: *node,
                reason: "node is not a bound block input",
            })?;
        reject_non_reducing_expert_staging(*node, blocks[block_position], source)?;
    }
    let (device, _queue) = device_and_queue()?;
    let mut buffers = BTreeMap::new();
    for (node, source) in expert_sources {
        if !plan.prepared.block_nodes.contains(node) {
            return Err(MetalError::ExpertSourceUnsupported {
                node: *node,
                reason: "node is not a bound block input",
            });
        }
        let entries = source.entries();
        entries.first().ok_or(MetalError::ExpertSourceUnsupported {
            node: *node,
            reason: "table is empty",
        })?;
        let signature = expert_source_signature(source);
        let cache_key = (plan_identity, *node);
        EXPERT_SOURCE_CACHE.with(|cache| -> Result<(), MetalError> {
            let mut cache =
                cache
                    .try_borrow_mut()
                    .map_err(|_| MetalError::ExpertSourceUnsupported {
                        node: *node,
                        reason: "expert source cache is already borrowed",
                    })?;
            let cache_state = cache.get(&cache_key).map(|(cached_signature, _)| {
                if *cached_signature == signature {
                    ExpertSourceCacheState::Hit
                } else {
                    ExpertSourceCacheState::ReplacementMiss
                }
            });
            let needs_stage = !matches!(cache_state, Some(ExpertSourceCacheState::Hit));
            if needs_stage {
                #[cfg(feature = "instrument")]
                {
                    counter!(EXPERT_SOURCE_CACHE_MISSES, 1);
                    match cache_state {
                        None => counter!(EXPERT_SOURCE_CACHE_COLD_MISSES, 1),
                        Some(ExpertSourceCacheState::ReplacementMiss) => {
                            counter!(EXPERT_SOURCE_CACHE_REPLACEMENT_MISSES, 1)
                        }
                        Some(ExpertSourceCacheState::Hit) => unreachable!(),
                    }
                }
                let previous = cache.remove(&cache_key).map(|(_, staged)| staged);
                let staged = stage_expert_source_reusing(&device, *node, source, previous)?;
                cache.insert(cache_key, (signature, staged));
                debug!(
                    node = ?node,
                    signature,
                    payload_bytes = cache
                        .get(&cache_key)
                        .map_or(0, |(_, staged)| staged.buffers.payloads.length()),
                    "metal_expert_source_cache_miss"
                );
            } else {
                #[cfg(feature = "instrument")]
                counter!(EXPERT_SOURCE_CACHE_HITS, 1);
                debug!(
                    node = ?node,
                    signature,
                    payload_bytes = cache
                        .get(&cache_key)
                        .map_or(0, |(_, staged)| staged.buffers.payloads.length()),
                    "metal_expert_source_cache_hit"
                );
            }
            let Some((_, staged)) = cache.get(&cache_key) else {
                return Err(MetalError::ExpertSourceUnsupported {
                    node: *node,
                    reason: "expert source cache insertion did not produce a table",
                });
            };
            buffers.insert(
                *node,
                ExpertSourceBuffers {
                    node: *node,
                    payloads: staged.buffers.payloads.clone(),
                    payload_offset: staged.buffers.payload_offset,
                    descriptors: staged.buffers.descriptors.clone(),
                    descriptor_records: staged.buffers.descriptor_records.clone(),
                },
            );
            Ok(())
        })?;
    }
    Ok(buffers)
}

/// Named-block counterpart used by `omega::backend`.
pub fn execute_plan_named_with_expert_sources(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'_>)],
    expert_sources: &BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'_>>,
) -> Result<Evaluated, MetalError> {
    let resolve_started = std::time::Instant::now();
    // Keep the original checkpoint blocks for the reduction guard; the
    // execution path substitutes expert payloads only after this comparison.
    let blocks = proxima_tensor::cpu::resolve_named_blocks_with_experts(
        &plan.program,
        named,
        Some(expert_sources),
    )?;
    let resolve_elapsed_us = resolve_started.elapsed().as_micros();
    let execute_started = std::time::Instant::now();
    let result = execute_plan_with_expert_sources(plan, &blocks, expert_sources);
    debug!(
        resolve_us = resolve_elapsed_us as u64,
        execute_us = execute_started.elapsed().as_micros() as u64,
        "metal_expert_execute"
    );
    result
}

/// A device buffer a CALLER allocates, owns, and keeps alive across multiple
/// [`execute_plan_with_placements`] calls — the type this module's other
/// buffers (`MetalBuffer`, private) never had to be public for, since
/// `execute_plan`/`execute_plan_op_timed` allocate and own every buffer
/// themselves. Get one from [`allocate_placed_buffer`]; read one back with
/// [`try_read_placed_buffer_f32`].
#[cfg(feature = "metal-output-placement")]
pub type PlacedBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;

/// Allocates a `storageModeShared` buffer of `byte_len` bytes that OUTLIVES
/// any one [`execute_plan_with_placements`] call — the caller holds it,
/// passes `&buffer` into as many calls as it likes, and only it decides when
/// the buffer is dropped. Mirrors `allocate_buffer`'s own device call,
/// public and un-sized-to-an-op because a placed buffer's size is the
/// caller's own layout decision (e.g. a whole KV-cache page), not one op's
/// `bound_output_len`.
///
/// # Errors
/// Propagates a Metal device/driver failure to allocate.
#[cfg(feature = "metal-output-placement")]
pub fn allocate_placed_buffer(byte_len: usize) -> Result<PlacedBuffer, MetalError> {
    let (device, _queue) = device_and_queue()?;
    let buffer = device
        .newBufferWithLength_options(byte_len.max(1), MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| MetalError::CompileFailed {
            log: "device refused to allocate a placed buffer".to_string(),
        })?;
    #[cfg(feature = "instrument")]
    record_device_buffer("placed_buffer", "session", "", byte_len.max(1), false);
    Ok(buffer)
}

/// Wraps memory the caller owns as a placed buffer without copying it, so the
/// buffer and the range alias the same bytes. The placement of a kv buffer is
/// the caller's decision, and this is how a caller places it in memory it
/// owns. [`allocate_placed_buffer`] is the owning form; the call underneath is
/// `create_no_copy_buffer` (`newBufferWithBytesNoCopy` with no deallocator).
///
/// # Safety
/// `base` must point to `byte_len` bytes that outlive every use of the
/// returned buffer and are not freed, remapped or reused while a command
/// buffer is in flight. The buffer never owns the memory.
///
/// # Errors
/// Refuses a null or non-page-aligned pointer, a zero or non-page-aligned
/// length, and propagates a device refusal.
#[cfg(feature = "metal-output-placement")]
pub unsafe fn allocate_placed_buffer_over(
    base: *mut u8,
    byte_len: usize,
) -> Result<PlacedBuffer, MetalError> {
    if base.is_null() || byte_len == 0 || !is_page_aligned(base.cast(), byte_len) {
        return Err(MetalError::CompileFailed {
            log: "a placed buffer over caller memory needs a page-aligned, non-null pointer and a non-zero page-aligned length".to_string(),
        });
    }
    let (device, _queue) = device_and_queue()?;
    let buffer = create_no_copy_buffer(&device, base.cast(), byte_len)?;
    #[cfg(feature = "instrument")]
    record_device_buffer("placed_buffer_over", "session", "", byte_len, true);
    Ok(buffer)
}

/// Zero-fills the first `byte_len` bytes of `buffer`. A freshly allocated
/// `MTLBuffer`'s contents are undefined (`allocate_fault_buffer`'s own
/// doc), and [`allocate_placed_buffer`] never zero-fills -- most callers
/// only ever read positions they themselves already wrote. The
/// `kv-capacity-bucket` padded-tail read is the exception: a decode step
/// binds the KV `Op::Input` leaf to `bucket` rows (`bucket > merged_len`),
/// so `[merged_len, bucket)` is read even though this call never wrote it.
/// `causal_mask_merged` masks that range's attention SCORE to `-inf`
/// exactly (`ScalarOp::Select` picks the constant without reading the
/// garbage key), but the softmax weight it produces (`0.0`) still
/// multiplies the corresponding V row -- `0.0 * garbage` is `0.0` only if
/// the garbage is a normal float; an uninitialized `NaN`/`Inf` bit pattern
/// survives that multiply. `proxima-model-interop`'s
/// `run_decode_loop_placed_kv` calls this once per KV buffer at
/// allocation (not per token): every row a later step ever reads was
/// either zeroed here or overwritten by a REAL rotated key/value this same
/// call wrote first, since `cached_len` only grows.
///
/// # Errors
/// [`MetalError::PlacedBufferRange`] when `byte_len` bytes do not fit in the allocation; nothing is written.
#[cfg(feature = "metal-output-placement")]
pub fn zero_placed_buffer(buffer: &PlacedBuffer, byte_len: usize) -> Result<(), MetalError> {
    zero_placed_buffer_range(buffer, 0, byte_len)
}

/// [`zero_placed_buffer`] for `byte_len` bytes starting at `byte_offset`: a
/// caller that overwrites the head of a buffer with real rows right after
/// allocating it zeroes only the tail it will not write. No command buffer may be in flight.
///
/// # Errors
/// [`MetalError::PlacedBufferRange`] when `byte_offset + byte_len` overflows or extends past the
/// allocation; nothing is written.
#[cfg(feature = "metal-output-placement")]
pub fn zero_placed_buffer_range(
    buffer: &PlacedBuffer,
    byte_offset: usize,
    byte_len: usize,
) -> Result<(), MetalError> {
    validate_placed_buffer_range(buffer, byte_offset, byte_len)?;
    let pointer = buffer.contents();
    // SAFETY: validation proved `byte_offset + byte_len` lies inside the shared allocation; no command buffer is in flight (every placed execute waits).
    let slots = unsafe {
        core::slice::from_raw_parts_mut(pointer.as_ptr().cast::<u8>().add(byte_offset), byte_len)
    };
    slots.fill(0);
    Ok(())
}

/// Copies `source` into a caller-owned placed buffer at `byte_offset`.
/// Call this after the command buffer that last used the buffer has
/// completed; execution calls this driver performs return only after
/// `waitUntilCompleted`.
///
/// # Errors
/// Returns [`MetalError::PlacedBufferRange`] when the requested byte range
/// overflows or extends past the buffer's allocated length.
#[cfg(feature = "metal-output-placement")]
pub fn try_write_placed_buffer_f32(
    buffer: &PlacedBuffer,
    byte_offset: usize,
    source: &[f32],
) -> Result<(), MetalError> {
    let byte_length = f32_byte_length(buffer, byte_offset, source.len())?;
    validate_placed_buffer_range(buffer, byte_offset, byte_length)?;
    // SAFETY: `source` exposes `byte_length` initialized bytes, and f32 storage is valid as bytes.
    let source_bytes = unsafe { core::slice::from_raw_parts(source.as_ptr().cast::<u8>(), byte_length) };
    // SAFETY: validation proves `byte_offset` is within or one past the buffer's allocation.
    let destination = unsafe { buffer.contents().as_ptr().cast::<u8>().add(byte_offset) };
    // SAFETY: validation proves the destination span lies in the shared buffer; the source span is `source`'s f32 bytes.
    unsafe { core::ptr::copy_nonoverlapping(source_bytes.as_ptr(), destination, byte_length) };
    Ok(())
}

/// Copies a placed buffer byte range into the caller's `dest` slice.
/// Call this only after the command buffer that wrote the buffer has
/// completed; execution calls this driver performs return only after
/// `waitUntilCompleted`.
///
/// # Errors
/// Returns [`MetalError::PlacedBufferRange`] when the requested byte range
/// overflows or extends past the buffer's allocated length.
#[cfg(feature = "metal-output-placement")]
pub fn read_placed_buffer_f32_into(
    buffer: &PlacedBuffer,
    byte_offset: usize,
    dest: &mut [f32],
) -> Result<(), MetalError> {
    let byte_length = f32_byte_length(buffer, byte_offset, dest.len())?;
    validate_placed_buffer_range(buffer, byte_offset, byte_length)?;
    // SAFETY: validation proves `byte_offset` is within or one past the buffer's allocation.
    let source = unsafe { buffer.contents().as_ptr().cast::<u8>().add(byte_offset) };
    let destination = dest.as_mut_ptr().cast::<u8>();
    // SAFETY: validation proves the source span lies in the shared buffer; the destination span is `dest`'s f32 bytes.
    unsafe { core::ptr::copy_nonoverlapping(source, destination, byte_length) };
    Ok(())
}

#[cfg(feature = "metal-output-placement")]
fn f32_byte_length(
    buffer: &PlacedBuffer,
    byte_offset: usize,
    element_count: usize,
) -> Result<usize, MetalError> {
    element_count
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| MetalError::PlacedBufferRange {
            byte_offset,
            byte_length: usize::MAX,
            buffer_length: buffer.length(),
        })
}

#[cfg(feature = "metal-output-placement")]
fn validate_placed_buffer_range(
    buffer: &PlacedBuffer,
    byte_offset: usize,
    byte_length: usize,
) -> Result<(), MetalError> {
    let buffer_length = buffer.length();
    let end = byte_offset.checked_add(byte_length);
    if end.is_none_or(|end| byte_offset > buffer_length || end > buffer_length) {
        return Err(MetalError::PlacedBufferRange {
            byte_offset,
            byte_length,
            buffer_length,
        });
    }
    Ok(())
}

/// Bytes the device allocated for `buffer`: the length a caller that pools placed buffers compares against the bytes it needs.
#[cfg(feature = "metal-output-placement")]
#[must_use]
pub fn placed_buffer_len(buffer: &PlacedBuffer) -> usize {
    buffer.length()
}

/// Row 555 diagnostic: the device pointer identity behind a [`PlacedBuffer`],
/// so a caller can prove whether the buffer bound as `state_out` at step N is
/// the SAME buffer bound as `state_in` at step N+1, rather than inferring it
/// from node ids alone.
#[cfg(all(feature = "metal-output-placement", feature = "instrument"))]
#[must_use]
pub fn placed_buffer_identity(buffer: &PlacedBuffer) -> usize {
    Retained::as_ptr(buffer) as usize
}

/// Reads `element_count` `f32`s back from `buffer` starting at `byte_offset`
/// — the read-back counterpart to a placed write, for a caller that wants to
/// inspect what an [`execute_plan_with_placements`] call wrote without going
/// through [`Evaluated`]'s own output set (a placed node need not be a named
/// output at all).
///
/// The owned form of [`read_placed_buffer_f32_into`], with the same bounds check: the copy
/// goes straight into the returned `Vec`'s spare capacity, so no zero fill precedes it. Prefer
/// [`read_placed_buffer_f32_into`] when the caller already holds a slice to fill.
///
/// # Errors
/// [`MetalError::PlacedBufferRange`] when the requested byte range overflows or extends past the
/// buffer's allocated length.
#[cfg(feature = "metal-output-placement")]
pub fn try_read_placed_buffer_f32(
    buffer: &PlacedBuffer,
    byte_offset: usize,
    element_count: usize,
) -> Result<Vec<f32>, MetalError> {
    let byte_length = f32_byte_length(buffer, byte_offset, element_count)?;
    validate_placed_buffer_range(buffer, byte_offset, byte_length)?;
    let mut values: Vec<f32> = Vec::with_capacity(element_count);
    // SAFETY: validation proves `byte_offset` and the `byte_length` bytes after it lie inside the shared allocation, which is CPU-visible once the writing command buffer completed; the destination is the Vec's `element_count`-f32 capacity. The copy is bytewise, so an unaligned `byte_offset` forms no misaligned reference, and every bit pattern is a valid f32.
    unsafe {
        let source = buffer.contents().as_ptr().cast::<u8>().add(byte_offset);
        core::ptr::copy_nonoverlapping(source, values.as_mut_ptr().cast::<u8>(), byte_length);
        values.set_len(element_count);
    }
    Ok(values)
}

/// Encodes `values` into the cache format `codec` was resolved for, at `byte_offset`: the one
/// store for every cache format.
///
/// Composes [`proxima_numeric::FormatCodec`], resolved once at allocation with
/// [`proxima_numeric::FormatCodec::for_format`]: the call is one indirect jump to the instantiation
/// with the layout as a literal, where [`proxima_numeric::encode::encode_blocks`] re-matches the
/// format on every call. Scalar and block formats place alike; the byte range is the codec's whole-block
/// geometry. Same in-flight contract as [`try_write_placed_buffer_f32`]; the offset and window are checked, never trusted.
///
/// An offset at or past the end of the allocation leaves an empty window, which the codec refuses like
/// any window shorter than the packed range; the buffer is unwritten in every error case.
///
/// # Errors
/// [`NumericError::InputNotElementMultiple`] when `values.len()` is not whole blocks of the codec's
/// format, [`NumericError::OutputSizeMismatch`] when the allocation past `byte_offset` (empty if the
/// offset is past the end) is shorter than the packed range, and [`NumericError::UnsupportedEncode`]
/// for a format with no encode. No variant names an offset: the window is the quantity the codec sizes.
#[cfg(feature = "metal-output-placement")]
pub fn write_placed_buffer(
    buffer: &PlacedBuffer,
    byte_offset: usize,
    codec: &FormatCodec,
    values: &[f32],
) -> Result<(), NumericError> {
    let (start, window_len) = placed_tail(buffer, byte_offset);
    // SAFETY: `start` is inside the allocation and `window_len` bytes follow it (`placed_tail` clamps the offset to the allocation length, so the add above never leaves it); the memory is shared with no command buffer in flight (caller contract, every placed execute waits); the codec checks the packed range against `window_len`.
    let window = unsafe { core::slice::from_raw_parts_mut(start, window_len) };
    codec.encode(values, window).map(drop)
}

/// The bytes of `buffer` from `byte_offset` to its end as a base pointer and length. An offset past the
/// end is clamped to the end, so the pointer arithmetic stays within or one past the allocation and the
/// window is empty; the rest of the allocation, never more, is what a caller may form a slice over.
#[cfg(feature = "metal-output-placement")]
fn placed_tail(buffer: &PlacedBuffer, byte_offset: usize) -> (*mut u8, usize) {
    let length = buffer.length();
    let start = byte_offset.min(length);
    // SAFETY: `start <= length`, so the result is within or one past the allocation `contents()` points at.
    let base = unsafe { buffer.contents().as_ptr().cast::<u8>().add(start) };
    (base, length - start)
}

/// Decodes `element_count` elements at `byte_offset` to `f32`, the inverse of
/// [`write_placed_buffer`]: straight into the returned `Vec`'s spare capacity, so no zero fill
/// precedes the write.
///
/// # Errors
/// [`NumericError::InputNotElementMultiple`] when `element_count` is not whole blocks of the
/// codec's format, [`NumericError::InputNotBlockMultiple`] or [`NumericError::OutputSizeMismatch`]
/// when the allocation past `byte_offset` (empty if the offset is past the end) is shorter than the
/// packed range. No variant names an offset: the window is the quantity the codec sizes.
#[cfg(feature = "metal-output-placement")]
pub fn read_placed_buffer(
    buffer: &PlacedBuffer,
    byte_offset: usize,
    codec: &FormatCodec,
    element_count: usize,
) -> Result<Vec<f32>, NumericError> {
    let (start, window_len) = placed_tail(buffer, byte_offset);
    // SAFETY: same as `write_placed_buffer`: `start` is inside the allocation with `window_len` bytes after it, and no command buffer is in flight; the codec checks the packed range against `window_len`.
    let window = unsafe { core::slice::from_raw_parts(start, window_len) };
    let mut decoded: Vec<f32> = Vec::with_capacity(element_count);
    codec.decode(window, &mut decoded.spare_capacity_mut()[..element_count])?;
    // SAFETY: `decode` returned Ok, so it initialised all `element_count` slots of the spare prefix.
    unsafe { decoded.set_len(element_count) };
    Ok(decoded)
}

/// Encodes `element_count` f32s of `source` (at `source_byte_offset`) into the codec's format in
/// `destination` at `destination_byte_offset`, with no host allocation: the step-end narrowing of a
/// cache whose format is not the graph's f32 output.
///
/// # Contract
/// The buffers are distinct allocations and no command buffer is in flight. Neither offset nor the
/// source range is trusted otherwise: alignment and both ranges are checked before any slice is formed.
///
/// # Errors
/// [`NumericError::InputNotBlockMultiple`] (`format: "binary32"`, `found: source_byte_offset`) when the
/// source address is not `f32`-aligned, i.e. the offset is not a whole element,
/// [`NumericError::InputNotElementMultiple`] when `element_count` is not whole blocks of the
/// codec's format, [`NumericError::OutputSizeMismatch`] when the source past `source_byte_offset` holds
/// fewer than `element_count` `f32`s (`found` is how many it holds, `expected` is `element_count`) or
/// the destination past `destination_byte_offset` is shorter than the packed range (an offset past
/// either end is an empty window). No variant names an offset: the window is the quantity checked.
#[cfg(feature = "metal-output-placement")]
pub fn narrow_placed_buffer(
    source: &PlacedBuffer,
    source_byte_offset: usize,
    destination: &PlacedBuffer,
    destination_byte_offset: usize,
    codec: &FormatCodec,
    element_count: usize,
) -> Result<(), NumericError> {
    let (source_start, source_window) = placed_tail(source, source_byte_offset);
    if !source_start.cast::<f32>().is_aligned() {
        return Err(NumericError::InputNotBlockMultiple {
            format: "binary32",
            found: source_byte_offset,
            block_bytes: size_of::<f32>(),
        });
    }
    let source_held = source_window / size_of::<f32>();
    if element_count > source_held {
        return Err(NumericError::OutputSizeMismatch { found: source_held, expected: element_count });
    }
    let (destination_start, destination_window) = placed_tail(destination, destination_byte_offset);
    // SAFETY: idle shared buffers (every placed execute waits) and distinct allocations (caller contract); `element_count <= source_window / 4` was just checked, so the source slice lies inside the source allocation, and the destination slice is the `destination_window` bytes after a start `placed_tail` kept inside its allocation, which the codec checks against the packed range. The source pointer was just checked `f32`-aligned.
    let (rows, packed) = unsafe {
        (
            core::slice::from_raw_parts(source_start.cast::<f32>(), element_count),
            core::slice::from_raw_parts_mut(destination_start, destination_window),
        )
    };
    codec.encode(rows, packed).map(drop)
}

/// Moves `byte_len` bytes from `from_byte` to `to_byte` inside one
/// [`PlacedBuffer`], overlap allowed -- how a sliding-window KV buffer
/// compacts its live rows back to the front without leaving the device's
/// shared allocation. No command buffer may be in flight.
///
/// # Errors
/// [`MetalError::PlacedBufferRange`] when either range overflows or extends past the allocation;
/// nothing is moved.
#[cfg(feature = "metal-output-placement")]
pub fn move_placed_buffer_bytes(
    buffer: &PlacedBuffer,
    from_byte: usize,
    to_byte: usize,
    byte_len: usize,
) -> Result<(), MetalError> {
    validate_placed_buffer_range(buffer, from_byte, byte_len)?;
    validate_placed_buffer_range(buffer, to_byte, byte_len)?;
    let pointer = buffer.contents();
    // SAFETY: validation proved both ranges lie inside the allocation, and `ptr::copy` is the overlap-safe move.
    unsafe {
        let base = pointer.as_ptr().cast::<u8>();
        core::ptr::copy(base.add(from_byte), base.add(to_byte), byte_len);
    }
    Ok(())
}

/// [`DispatchType::Concurrent`]'s dataflow-hazard set, generic over the
/// identity type so this logic is testable without a real Metal device
/// (`Id = usize`/`&str` in tests, `Id = *const ProtocolObject<dyn MTLBuffer>`
/// — pointer identity from [`Retained::as_ptr`] — on the real driver path).
/// Tracks two sets since the same buffer position is bound differently on
/// each side of a hazard: `written` names every buffer some op since the
/// last barrier has WRITTEN (a RAW or WAW hazard for a later op that reads
/// or writes it); `read` names every buffer READ since the last barrier (a
/// WAR hazard for a later op that writes it — real here because
/// [`BufferArena`] reuses a whole retired slot's buffer object for a later
/// position). A fresh, never-before-seen buffer (this op's own freshly
/// [`allocate_buffer`]d output) triggers neither: nothing else has touched
/// that pointer yet. Only instantiated on [`DispatchType::Concurrent`]'s
/// path -- [`DispatchType::Serial`] never allocates one, since a serial
/// encoder already orders every dispatch for it.
#[derive(Debug)]
pub(super) struct HazardTracker<Id: Eq + core::hash::Hash + Copy> {
    pub(super) written: std::collections::HashSet<Id>,
    pub(super) read: std::collections::HashSet<Id>,
}

impl<Id: Eq + core::hash::Hash + Copy> HazardTracker<Id> {
    pub(super) fn new() -> Self {
        Self {
            written: std::collections::HashSet::new(),
            read: std::collections::HashSet::new(),
        }
    }

    /// True when encoding the next op without a barrier first would let a
    /// concurrent-dispatch-scheduled GPU thread race a still-in-flight one:
    /// RAW (an input was written since the last barrier), WAW (the output
    /// buffer was written since the last barrier), or WAR (the output
    /// buffer was read since the last barrier — arena slot reuse). Delegates
    /// to [`classify`] so the boolean and the per-cause breakdown can never
    /// drift apart.
    pub(super) fn needs_barrier(
        &self,
        inputs: &[Id],
        outputs: impl IntoIterator<Item = Id>,
    ) -> bool {
        self.classify(inputs, outputs) != HazardClass::None
    }

    /// [`needs_barrier`]'s own decomposition into which single hazard
    /// explains it, checked in the same order that `||` chain implies (a
    /// RAW input hazard first).
    pub(super) fn classify(
        &self,
        inputs: &[Id],
        outputs: impl IntoIterator<Item = Id>,
    ) -> HazardClass {
        if inputs.iter().any(|input| self.written.contains(input)) {
            return HazardClass::Raw;
        }
        let mut class = HazardClass::None;
        for output in outputs {
            if self.written.contains(&output) {
                return HazardClass::Waw;
            }
            if self.read.contains(&output) {
                class = HazardClass::War;
            }
        }
        class
    }

    /// Clears both sets — called immediately after a barrier is actually
    /// emitted, since the barrier is exactly the guarantee that every
    /// dispatch encoded before it has completed and is visible to every
    /// dispatch encoded after.
    pub(super) fn reset(&mut self) {
        self.written.clear();
        self.read.clear();
    }

    /// Records this op's own effect, called once per op regardless of
    /// whether a barrier fired for it.
    pub(super) fn record(&mut self, inputs: &[Id], outputs: impl IntoIterator<Item = Id>) {
        self.written.extend(outputs);
        self.read.extend(inputs.iter().copied());
    }
}

/// [`HazardTracker::classify`]'s result -- ROW 539's per-barrier attribution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum HazardClass {
    /// No barrier needed.
    None,
    /// An input was written since the last barrier -- a genuine dataflow
    /// edge between the writer and this reader.
    Raw,
    /// The output identity was written since the last barrier.
    Waw,
    /// The output identity was read since the last barrier -- the classic
    /// arena-slot-reuse shape: a prior op's input buffer handed straight
    /// back out as a later, unrelated op's output.
    War,
}

/// Attributes one fired barrier to its [`HazardClass`] counter and, for a
/// WAW/WAR barrier, to whether the colliding identity is a
/// [`BufferArena`]-recycled slot (a false dependency slot reuse manufactured
/// between two data-independent ops) or a persistent/output-placed buffer
/// genuinely written more than once.
#[cfg(feature = "instrument")]
pub(super) fn record_hazard_class(class: HazardClass, arena_recycled: bool) {
    match class {
        HazardClass::None => {}
        HazardClass::Raw => counter!(BARRIERS_RAW, 1),
        HazardClass::Waw | HazardClass::War => {
            if class == HazardClass::Waw {
                counter!(BARRIERS_WAW, 1);
            } else {
                counter!(BARRIERS_WAR, 1);
            }
            if arena_recycled {
                counter!(BARRIERS_WAW_WAR_ARENA_RECYCLED, 1);
            } else {
                counter!(BARRIERS_WAW_WAR_PERSISTENT, 1);
            }
        }
    }
}

/// One structured event per op-level hazard step: `reason` is `none` when no
/// barrier fired, else `true_node_dependency` when an input names the node that wrote its
/// buffer, `pointer_alias_without_node_dependency` when a RAW fired on a
/// shared buffer identity alone, `arena_slot_reuse` when a recycled slot was
/// handed to an unrelated node, `persistent_buffer_rewrite` otherwise.
/// `run_before` is the dispatching positions encoded since the previous
/// barrier. Reads: [`HazardState::raw_edges`], [`record_hazard_class`].
#[cfg(feature = "instrument")]
pub(super) fn record_hazard_op(
    position: usize,
    node: NodeId,
    class: HazardClass,
    arena_recycled: bool,
    edges: (u64, u64),
    run_before: u64,
) {
    let (true_edges, alias_edges) = edges;
    if class != HazardClass::None {
        note_barrier_pending();
    }
    let reason = match class {
        HazardClass::None => "none",
        HazardClass::Raw if true_edges > 0 => "true_node_dependency",
        HazardClass::Raw => "pointer_alias_without_node_dependency",
        HazardClass::Waw | HazardClass::War if arena_recycled => "arena_slot_reuse",
        HazardClass::Waw | HazardClass::War => "persistent_buffer_rewrite",
    };
    debug!(
        site = "op",
        position = position as u64,
        node = ?node,
        class = ?class,
        reason,
        true_edges,
        alias_edges,
        run_before,
        "hazard_op"
    );
}

/// The barriers an op fires against buffers it wrote itself (attention
/// split to merge, route prepass to place to consumer): each reads what the
/// same op just wrote, so the reason is always an internal true dependency.
#[cfg(feature = "instrument")]
pub(super) fn record_internal_barrier(site: &'static str) {
    note_barrier_pending();
    debug!(
        site,
        reason = "internal_true_dependency",
        "hazard_internal_barrier"
    );
}

#[cfg(feature = "instrument")]
thread_local! {
    static BARRIER_PENDING: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// Marks that a barrier fired since the last captured dispatch; the next
/// [`take_barrier_pending`] hands it to that dispatch's record.
#[cfg(feature = "instrument")]
fn note_barrier_pending() {
    BARRIER_PENDING.with(|pending| pending.set(true));
}

#[cfg(feature = "instrument")]
pub(super) fn take_barrier_pending() -> bool {
    BARRIER_PENDING.with(|pending| pending.replace(false))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ExpertSourceCacheState {
    Hit,
    ReplacementMiss,
}

/// [`execute_plan_with_placements`]'s per-call hazard bookkeeping, owned by
/// the [`Plan`] and reused call-to-call instead of rebuilt: [`HazardTracker`]'s
/// two `HashSet`s and [`resolve_hazard_inputs`]'s pointer list all allocate
/// on their first insert of a call. A fresh [`HazardTracker::new`] and a
/// fresh `Vec` every call paid that allocation every call, forever, even
/// though every call needs the exact same starting (empty) state -- ROW 303's
/// residual. [`HazardState::reset`] clears both without releasing their
/// capacity, so only the very first call after a plan's own construction
/// ever allocates.
pub(super) struct HazardState {
    pub(super) tracker: HazardTracker<*const ProtocolObject<dyn MTLBuffer>>,
    pub(super) inputs: Vec<*const ProtocolObject<dyn MTLBuffer>>,
    /// Every buffer the op being stepped writes, primary first; reused like
    /// `inputs` so a warm step allocates nothing.
    pub(super) outputs: Vec<*const ProtocolObject<dyn MTLBuffer>>,
    /// The node whose output last wrote each buffer identity, so a RAW barrier
    /// can tell an input that reads the writer's own node from a buffer
    /// identity two unrelated nodes happen to share.
    #[cfg(feature = "instrument")]
    pub(super) writer_nodes:
        std::collections::HashMap<*const ProtocolObject<dyn MTLBuffer>, NodeId>,
    /// Dispatching positions encoded since the last barrier this state saw.
    #[cfg(feature = "instrument")]
    pub(super) dispatches_since_barrier: u64,
}

impl HazardState {
    fn new() -> Self {
        Self {
            tracker: HazardTracker::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            #[cfg(feature = "instrument")]
            writer_nodes: std::collections::HashMap::new(),
            #[cfg(feature = "instrument")]
            dispatches_since_barrier: 0,
        }
    }

    pub(super) fn reset(&mut self) {
        self.tracker.reset();
        self.inputs.clear();
        self.outputs.clear();
        #[cfg(feature = "instrument")]
        {
            self.writer_nodes.clear();
            self.dispatches_since_barrier = 0;
        }
    }

    /// For the op about to be encoded: how many of its read operands name the
    /// node that wrote their buffer since the last barrier (a true dataflow
    /// edge) and how many only share a written buffer identity with a
    /// different node. `read_nodes` pairs with `self.inputs` in order.
    #[cfg(feature = "instrument")]
    pub(super) fn raw_edges(&self, read_nodes: &[NodeId]) -> (u64, u64) {
        read_nodes
            .iter()
            .zip(&self.inputs)
            .filter(|(_, pointer)| self.tracker.written.contains(*pointer))
            .fold((0, 0), |(true_edges, alias_edges), (node, pointer)| {
                if self.writer_nodes.get(pointer) == Some(node) {
                    (true_edges + 1, alias_edges)
                } else {
                    (true_edges, alias_edges + 1)
                }
            })
    }

    /// Records one encoded op after its hazard step: the node that now owns
    /// the output buffer identity, and the run length since the last barrier.
    #[cfg(feature = "instrument")]
    pub(super) fn note_dispatch(
        &mut self,
        output: *const ProtocolObject<dyn MTLBuffer>,
        node: NodeId,
        barrier_fired: bool,
    ) {
        if barrier_fired {
            self.dispatches_since_barrier = 0;
        }
        self.dispatches_since_barrier += 1;
        self.writer_nodes.insert(output, node);
    }
}

/// The one adapter from a dispatch's read `NodeId`s (see
/// `crate::msl::hazard_read_nodes`, the caller's own source for `nodes` below)
/// to the hazard tracker's buffer-pointer identities, resolved from
/// `device_buffers` — erroring on the first node with none instead of
/// silently dropping it from the hazard set (the previous `filter_map`
/// behavior) — a node missing from `device_buffers` means the encode this
/// identity feeds is already wrong, so hiding it from the hazard check only
/// hides a real bug behind a missing barrier.
/// Test-only surface: production hazard resolution goes through
/// [`resolve_hazard_inputs_into`] against [`Plan::hazard_state`]'s reused
/// scratch list. Kept as an owned-`Vec` wrapper here since a unit test wants
/// a value to assert on, not a buffer to manage.
#[cfg(test)]
pub(super) fn resolve_hazard_inputs(
    nodes: impl Iterator<Item = NodeId>,
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
) -> Result<Vec<*const ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let mut resolved = Vec::new();
    resolve_hazard_inputs_into(nodes, device_buffers, &mut resolved)?;
    Ok(resolved)
}

/// [`resolve_hazard_inputs`], writing into a caller-owned, reused buffer
/// instead of collecting a fresh `Vec` -- [`execute_plan_with_placements`]
/// calls this against [`Plan::hazard_state`]'s own scratch list every
/// position so a warm plan-hit step's hazard resolution allocates nothing
/// (ROW 303's residual).
pub(super) fn resolve_hazard_inputs_into(
    nodes: impl Iterator<Item = NodeId>,
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    resolved: &mut Vec<*const ProtocolObject<dyn MTLBuffer>>,
) -> Result<(), MetalError> {
    resolved.clear();
    for operand in nodes {
        let pointer = device_buffers
            .get(&operand)
            .map(|(buffer, _offset)| Retained::as_ptr(buffer))
            .ok_or(MetalError::UnresolvedHazardOperand { node: operand })?;
        resolved.push(pointer);
    }
    Ok(())
}

/// The exact per-op hazard step [`execute_plan_with_placements`]'s loop
/// runs: check, barrier-and-reset only on a hazard, then always record —
/// factored out so the tests drive THIS function instead of a hand-written
/// mirror of the loop body that could silently drift from it. `outputs` is
/// the identity of every buffer this op is about to write, already resolved
/// (placement -> arena slot -> fresh allocation) by the caller before this
/// runs: its primary output first,
/// then each extra output buffer a twin, `MoeTopK` or `CachedSoftmaxWeights`
/// dispatch binds past `bindings` (`extra_output_nodes`). Each is checked for
/// WAW/WAR and recorded written, so a later reader of any of them is a RAW.
/// Returns whether the caller must emit a barrier before encoding this op.
pub(super) fn hazard_step<Id: Eq + core::hash::Hash + Copy>(
    tracker: &mut HazardTracker<Id>,
    inputs: &[Id],
    outputs: &[Id],
) -> bool {
    let needs_barrier = tracker.needs_barrier(inputs, outputs.iter().copied());
    if needs_barrier {
        tracker.reset();
    }
    tracker.record(inputs, outputs.iter().copied());
    counter!(HAZARD_STEP_CALLS, 1);
    needs_barrier
}

/// [`execute_plan`], plus the ability to route one or more nodes' outputs
/// into a buffer the CALLER owns (`output_placements`), and to bind one or
/// more [`Op::Input`] nodes DIRECTLY to a buffer
/// the caller already owns on the device (`input_placements`), skipping the
/// per-call `upload_block`/`upload_packed_bytes` host round trip entirely.
/// Both are `&[(node, buffer, byte_offset)]` — deliberately plain tuples
/// over a named struct or a single slice with a direction field: the call
/// site destructures each triple positionally either way (`for (node,
/// buffer, offset) in placements`), and TWO plain slices already carry the
/// direction in which parameter a placement is passed to, at zero added
/// type cost. Compare both shapes, written out, for one KV-cache-shaped
/// call (`kv` bound as this program's growing history, `row` the freshly
/// computed token this call is adding to it, both backed by ONE buffer):
///
/// ```text
/// // two plain slices (chosen) — no new type, direction = which parameter
/// execute_plan_with_placements(
///     &plan, &blocks,
///     &[(kv_input_node, &kv_buffer, 0)],
///     &[(row_output_node, &kv_buffer, cached_len * row_bytes)],
/// )?;
///
/// // one slice + a direction field (rejected) — an enum earns its keep only
/// // if some caller needs to build a MIXED, order-independent placement
/// // list at runtime; no caller here does, so it is a type with nothing to
/// // do beyond what two parameter names already say for free.
/// execute_plan_with_placements(&plan, &blocks, &[
///     Placement { node: kv_input_node, buffer: &kv_buffer, offset: 0, direction: Direction::Input },
///     Placement { node: row_output_node, buffer: &kv_buffer, offset: cached_len * row_bytes, direction: Direction::Output },
/// ])?;
/// ```
///
/// Three hazards this signature exists to name explicitly, not paper over:
///
/// - **Cross-invocation liveness.** `Prepared::retires` computes per-
///   program liveness only (`bound_op_retirement`'s own doc) — it has no
///   notion of a buffer surviving into the NEXT `execute_plan_with_placements`
///   call. Every placed node (input or output) is therefore excluded from
///   this call's own retire sweep below: even if something inside THIS
///   program reads or writes it after its nominal last use, this function
///   will not drop this call's reference to it. The caller's own
///   `PlacedBuffer` handle is what actually keeps the GPU allocation alive
///   across calls; this exclusion just stops this function's bookkeeping
///   map from disagreeing with that fact.
/// - **Cross-invocation aliasing.** A placed buffer written by one call may
///   be read (as an input placement) by a LATER call. That is safe because
///   every call `commit`s and `waitUntilCompleted`s its own command buffer
///   before returning (this function, like [`execute_plan`], never overlaps
///   two command buffers) — the next call's encoder cannot begin recording
///   real GPU work against the buffer until this call's write has completed
///   and is CPU/GPU-visible: a strict happens-before, stronger than same-
///   encoder hazard tracking needs to be.
/// - **Within-call aliasing: one op writes a placed buffer, a LATER op in
///   the SAME program reads it — the KV-cache shape this exists for.** This
///   is covered by the SAME mechanism [`execute_plan`]'s own opening comment
///   already establishes for two dispatches sharing one serial encoder:
///   `MTLDispatchTypeSerial` guarantees encode order is execution order, and
///   every buffer here is device-allocated `storageModeShared` (never
///   `HazardTrackingModeUntracked`), so Metal's automatic hazard tracking
///   inserts an implicit barrier before the later dispatch. That tracking
///   operates at whole-`MTLResource` granularity, not the sub-range named by
///   `setBuffer:offset:atIndex:` — a write at one offset and a read at a
///   DIFFERENT offset of the SAME `MTLBuffer` object are still the same
///   resource to the tracker, so the barrier applies regardless of which
///   byte ranges the two dispatches actually touch. Nothing here is new
///   relative to what every existing multi-op program already relies on
///   (op N's freshly allocated output, read by op N+1) — placement changes
///   WHERE the write lands, not whether ordering holds. The proof, not just
///   the argument, is `metal_output_placement.rs`'s
///   `a_program_reads_a_placed_write_from_a_later_op_in_the_same_call` test:
///   if hazard tracking did not cover this, that test would read stale
///   (pre-write) bytes instead of the fresh write, and it does not.
///
/// On [`DispatchType::Concurrent`], the paragraph above no longer
/// applies as written: the encoder is opened with
/// `computeCommandEncoderWithDispatchType(Concurrent)` instead of the
/// dispatch-type-less `computeCommandEncoder()`, which turns OFF the
/// automatic whole-resource barrier between every pair of dispatches — two
/// independent ops (e.g. Q/K/V projected from one normed input) now execute
/// with no ordering between them at all unless something inserts one. This
/// function inserts that "something" itself, explicitly, via a private
/// `HazardTracker` walked once per op before it is encoded — see that
/// type's own doc, right above this function, for the three hazards
/// (RAW/WAW/WAR) it covers, and
/// [`MTLBarrierScope::Buffers`] is emitted only where the tracker finds one.
///
/// `PROXIMA_METAL_KIND_FILTER=<term>[,<term>]` (or `!<term>[,<term>]` for
/// the complement) -- `instrument`-gated, default-off, parsed once per
/// [`execute_plan_with_placements`] call by [`KindFilter::from_env`]. Each
/// `<term>` is either a bare substring, matched against [`classify_kind`]'s
/// own return value (the same string a caller already sees in
/// `op_profile_bucket kind=...`), or `family:<substring>`, matched against
/// [`weight_family`]'s own return value for the op's first named operand
/// (the same aggregation `proxima-model-interop`'s `report_op_timings`
/// already reuses via that function, rather than each call site keeping its
/// own copy of the layer-index-stripping rule). Unset (`None`) in every
/// production run, which is the ROW's in-buffer ablation harness's own
/// arm-selection knob -- see that row for the arm table.
///
/// `classify_kind`'s live return values, as of the `BoundOpKind::name()`
/// delegation (`refactor(omega): classify_kind names the kind through the
/// type`): `cached_attention`, `elementwise`, `iota`, `constant`,
/// `keep::scan fold` (the four `BoundOpKind::name()`-delegated arms plus
/// `Reduce { keep: Keep::Scan, .. }`), and `reduce-tiled-gemm` /
/// `reduce-packed-row-blocked` / `reduce-cooperative` /
/// `reduce-generic-scalar` / `reduce-unclassified` for `Reduce { keep:
/// Keep::Reduce, .. }`. A bare `kind:`-shaped term that matches none of
/// those is rejected eagerly by [`KindFilter::from_env`] --
/// [`MetalError::UnknownKindFilterTerm`]. A `family:` term cannot be
/// validated the same way (family names are data-dependent on the loaded
/// checkpoint, not a fixed enum), so instead [`validate_kind_filter`]
/// checks, against THIS plan's own dispatch sequence, that the filter
/// removes at least one op and not every op --
/// [`MetalError::KindFilterMatchesNothing`] otherwise. Before these two
/// checks landed (ROW 308), a stale or misspelled term silently degenerated
/// to "every op skipped" or "no op skipped" instead of failing loudly.
#[cfg(feature = "instrument")]
pub(super) enum FilterTerm {
    Kind(String),
    Family(String),
}

#[cfg(feature = "instrument")]
impl FilterTerm {
    fn parse(raw: &str) -> Result<Self, MetalError> {
        match raw.strip_prefix("family:") {
            Some(family) => Ok(Self::Family(family.to_string())),
            None => {
                if KNOWN_KIND_SUBSTRINGS
                    .iter()
                    .any(|known| known.contains(raw))
                {
                    Ok(Self::Kind(raw.to_string()))
                } else {
                    Err(MetalError::UnknownKindFilterTerm {
                        term: raw.to_string(),
                    })
                }
            }
        }
    }

    fn matches(&self, kind: &str, family: Option<&str>) -> bool {
        match self {
            Self::Kind(substring) => kind.contains(substring.as_str()),
            Self::Family(substring) => family.is_some_and(|name| name.contains(substring.as_str())),
        }
    }
}

/// [`classify_kind`]'s own live return-value vocabulary, restated here only
/// for [`FilterTerm::parse`]'s eager validation -- see [`KindFilter`]'s own
/// doc for why this list must be re-read from `classify_kind`, not
/// memorized, whenever that function gains or renames an arm.
#[cfg(feature = "instrument")]
pub(super) const KNOWN_KIND_SUBSTRINGS: &[&str] = &[
    "cached_attention",
    "elementwise",
    "iota",
    "constant",
    "keep::scan fold",
    "reduce-tiled-gemm",
    "reduce-packed-row-blocked",
    "reduce-cooperative",
    "reduce-generic-scalar",
    "reduce-unclassified",
];

#[cfg(feature = "instrument")]
pub(super) struct KindFilter {
    pub(super) raw: String,
    pub(super) terms: Vec<FilterTerm>,
    pub(super) negate: bool,
}

#[cfg(feature = "instrument")]
impl KindFilter {
    pub(super) fn from_env() -> Result<Option<Self>, MetalError> {
        let Ok(raw) = std::env::var("PROXIMA_METAL_KIND_FILTER") else {
            return Ok(None);
        };
        let (negate, body) = match raw.strip_prefix('!') {
            Some(rest) => (true, rest),
            None => (false, raw.as_str()),
        };
        let terms = body
            .split(',')
            .map(str::trim)
            .filter(|term| !term.is_empty())
            .map(FilterTerm::parse)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(Self { raw, terms, negate }))
    }

    pub(super) fn matches(&self, kind: &str, family: Option<&str>) -> bool {
        let any = self.terms.iter().any(|term| term.matches(kind, family));
        any != self.negate
    }
}

/// `blk.7.ffn_down.weight` -> `ffn_down.weight`: drops exactly one
/// `.`-delimited numeric segment (the layer index every `blk.N.*` weight
/// name carries) so a caller can sum one matmul KIND across all layers
/// instead of reporting one line per layer. The one place this transform is
/// written -- `proxima-model-interop`'s `report_op_timings` calls this
/// function rather than keeping its own copy, and `KindFilter`'s `family:`
/// term reuses it too, so a layer-count change or a naming convention
/// change only needs to land here.
#[cfg(feature = "instrument")]
#[must_use]
pub fn weight_family(name: &str) -> String {
    name.split('.')
        .filter(|segment| segment.parse::<u32>().is_err())
        .collect::<Vec<&str>>()
        .join(".")
}

/// [`weight_family`] applied to `bound`'s own first named operand -- the
/// same `find_map` [`execute_plan_op_timed`] already runs to populate
/// [`OpGpuTiming::weight_name`], reused here rather than restated so
/// [`KindFilter`]'s `family:` term and the op-timed profiler's family
/// aggregation can never read two different operands as "the" weight.
#[cfg(feature = "instrument")]
pub(super) fn bound_weight_family(bound: &BoundOp, program: &[Op]) -> Option<String> {
    bound
        .operands()
        .iter()
        .find_map(|(source, _, _)| program[source.0 as usize].name())
        .map(weight_family)
}

/// Applies `filter` to every op in `prepared.resolved` the same way the
/// main dispatch loop below will, and rejects a filter that would remove
/// zero ops or every op -- either shape means the filter's own terms never
/// isolated anything in THIS plan, the silent-degenerate failure ROW 308
/// found (see [`KindFilter`]'s own doc).
#[cfg(feature = "instrument")]
pub(super) fn validate_kind_filter(
    filter: &KindFilter,
    prepared: &Prepared,
    packed_operands: &PackedOperands,
    program: &[Op],
) -> Result<(), MetalError> {
    let total = prepared.resolved.len();
    let removed = prepared
        .resolved
        .iter()
        .filter(|bound| {
            !filter.matches(
                classify_kind(bound, packed_operands),
                bound_weight_family(bound, program).as_deref(),
            )
        })
        .count();
    if removed == 0 || removed == total {
        return Err(MetalError::KindFilterMatchesNothing {
            filter: filter.raw.clone(),
        });
    }
    Ok(())
}

/// [`KindFilter`]-excluded counterpart of [`encode_op`]'s output-buffer half:
/// called INSTEAD of [`encode_op`] for an op the filter drops, so no
/// pipeline is bound, no buffer is bound to the encoder, and no dispatch is
/// issued -- this op's output buffer keeps whatever bytes it already held (a
/// prior decode step's write, under `metal-plan-stable-buffers`'s stable
/// per-position arena slot; an uninitialized fresh allocation otherwise).
/// `device_buffers` still needs an entry for `bound.node` regardless, or
/// every downstream op that reads it as an operand fails `buffer_for`'s
/// `NotLowerable` lookup before `gpu_exec_ms` is ever read -- the ablation
/// is measuring dispatch time with this op removed, not producing a correct
/// result, but the OTHER ops in the same buffer still need to run.
#[cfg(feature = "instrument")]
pub(super) fn register_skipped_output(
    device: &ProtocolObject<dyn MTLDevice>,
    device_buffers: &mut BTreeMap<NodeId, DeviceBuffer>,
    bound: &BoundOp,
    placement: Option<(&MetalBuffer, usize)>,
) -> Result<(), MetalError> {
    let (buffer, offset) = match placement {
        Some((buffer, offset)) => (buffer.clone(), offset),
        None => (
            allocate_buffer(device, bound_output_len(bound), bound.dtype)?,
            0,
        ),
    };
    device_buffers.insert(bound.node, (buffer, offset));
    Ok(())
}

#[cfg(all(test, feature = "metal-output-placement"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod placed_format_tests {
    use super::*;
    use proxima_numeric::{FloatLayout, NumericFormat};
    use proxima_tensor::BFloat8;

    fn stored_bytes(buffer: &PlacedBuffer, len: usize) -> Vec<u8> {
        // the buffer is cpu-visible placed memory with no gpu work in flight and callers pass at most its length
        unsafe { core::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u8>(), len) }.to_vec()
    }

    fn codec(format: &NumericFormat) -> FormatCodec {
        FormatCodec::for_format(format).expect("a scalar format resolves")
    }

    #[test]
    fn bf16_placed_roundtrips_special_values() {
        let buffer = allocate_placed_buffer(16).expect("allocates BF16 fixture bytes");
        let values = [0.0f32, -0.0, 1.5, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
        let bf16 = codec(&NumericFormat::BFloat16);
        write_placed_buffer(&buffer, 0, &bf16, &values).expect("bf16 encodes");
        let actual = read_placed_buffer(&buffer, 0, &bf16, values.len()).expect("bf16 decodes");

        assert_eq!(actual[0].to_bits(), 0x0000_0000);
        assert_eq!(actual[1].to_bits(), 0x8000_0000);
        assert_eq!(actual[2], 1.5);
        assert_eq!(actual[3], f32::INFINITY);
        assert_eq!(actual[4], f32::NEG_INFINITY);
        assert!(actual[5].is_nan());
    }

    #[test]
    fn bf16_and_f16_bytes_match_the_half_crate_on_k_like_rows() {
        let values = k_like_rows(256);
        let buffer = allocate_placed_buffer(512).expect("allocates fixture bytes");
        write_placed_buffer(&buffer, 0, &codec(&NumericFormat::BFloat16), &values).expect("bf16 encodes");
        let expected: Vec<u8> = values
            .iter()
            .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect();
        assert_eq!(stored_bytes(&buffer, 512), expected);

        write_placed_buffer(&buffer, 0, &codec(&NumericFormat::Float16), &values).expect("f16 encodes");
        let expected: Vec<u8> = values
            .iter()
            .flat_map(|value| half::f16::from_f32(*value).to_bits().to_le_bytes())
            .collect();
        assert_eq!(stored_bytes(&buffer, 512), expected);
    }

    #[test]
    fn a_respelled_layout_stores_and_loads_the_same_bytes_and_bits_as_its_named_spelling() {
        let values = k_like_rows(256);
        for named in [FloatLayout::BF32, FloatLayout::F16, FloatLayout::BF16, FloatLayout::E5M2_SATURATING] {
            let respelled = FloatLayout { bias_override: Some(named.bias()), ..named };
            let by_name = allocate_placed_buffer(1024).expect("allocates named-layout bytes");
            let by_data = allocate_placed_buffer(1024).expect("allocates respelled-layout bytes");
            let named_codec = codec(&NumericFormat::Scalar(named));
            let data_codec = codec(&NumericFormat::Scalar(respelled));

            write_placed_buffer(&by_name, 0, &named_codec, &values).expect("named layout encodes");
            write_placed_buffer(&by_data, 0, &data_codec, &values).expect("respelled layout encodes");

            assert_eq!(stored_bytes(&by_data, 1024), stored_bytes(&by_name, 1024), "{named:?}");
            let expected = read_placed_buffer(&by_name, 0, &named_codec, values.len()).expect("decodes");
            let actual = read_placed_buffer(&by_data, 0, &data_codec, values.len()).expect("decodes");
            assert_eq!(
                actual.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                "{named:?}"
            );
        }
    }

    #[test]
    fn bf16_placed_preserves_adjacent_bytes_at_nonzero_offset() {
        let staging = allocate_placed_buffer(16).expect("allocates f32 staging bytes");
        let cache = allocate_placed_buffer(16).expect("allocates BF16 cache bytes");
        try_write_placed_buffer_f32(&staging, 0, &[9.0, 1.5, 2.5, -7.0]).expect("four f32s fit the 16-byte staging buffer");
        let bf16 = codec(&NumericFormat::BFloat16);
        write_placed_buffer(&cache, 0, &bf16, &[9.0]).expect("encodes");
        write_placed_buffer(&cache, 12, &bf16, &[-7.0]).expect("encodes");

        narrow_placed_buffer(&staging, 4, &cache, 8, &bf16, 2).expect("narrows");

        let read = |offset, count| read_placed_buffer(&cache, offset, &bf16, count).expect("decodes");
        assert_eq!(read(0, 1), vec![9.0]);
        assert_eq!(read(8, 2), vec![1.5, 2.5]);
        assert_eq!(read(12, 1), vec![-7.0]);
    }

    #[test]
    fn bf8_placed_matches_golden_vectors() {
        let buffer = allocate_placed_buffer(16).expect("allocates BF8 fixture bytes");
        let values = [
            0x0000_0000,
            0x8000_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x3fa0_0000,
            0x3fc0_0000,
            0x3f90_0000,
            0x3fb0_0000,
            0x3880_0000,
            0x3780_0000,
            0x4760_0000,
            0x4770_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0000,
            0xffc0_0000,
        ]
        .map(f32::from_bits);
        let expected = [
            0x00, 0x80, 0x3c, 0xbc, 0x3d, 0x3e, 0x3c, 0x3e, 0x04, 0x01, 0x7b, 0x7c, 0x7c, 0xfc,
            0x7e, 0xfe,
        ];

        let bf8 = codec(&NumericFormat::BFloat8);
        write_placed_buffer(&buffer, 0, &bf8, &values).expect("bf8 encodes");

        assert_eq!(stored_bytes(&buffer, expected.len()), expected);
        let decoded = read_placed_buffer(&buffer, 0, &bf8, expected.len()).expect("bf8 decodes");
        assert_eq!(
            decoded
                .iter()
                .map(|value| BFloat8::from_f32(*value).to_bits())
                .collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn bf8_placed_narrows_at_nonzero_offsets_without_touching_sentinels() {
        let staging = allocate_placed_buffer(16).expect("allocates f32 staging bytes");
        let cache = allocate_placed_buffer(8).expect("allocates BF8 cache bytes");
        let bf8 = codec(&NumericFormat::BFloat8);
        try_write_placed_buffer_f32(&staging, 0, &[9.0, 1.5, 2.5, -7.0]).expect("four f32s fit the 16-byte staging buffer");
        write_placed_buffer(&cache, 0, &bf8, &[9.0]).expect("encodes");
        write_placed_buffer(&cache, 7, &bf8, &[-7.0]).expect("encodes");

        narrow_placed_buffer(&staging, 4, &cache, 2, &bf8, 2).expect("narrows");

        assert_eq!(stored_bytes(&cache, 8), &[0x48, 0, 0x3e, 0x41, 0, 0, 0, 0xc7]);
        assert_eq!(read_placed_buffer(&cache, 2, &bf8, 2).expect("decodes"), [1.5, 2.5]);
    }

    #[test]
    fn bf8_placed_narrow_rounds_exact_halfway_values_to_even_like_the_incumbent() {
        let staging = allocate_placed_buffer(12).expect("allocates f32 staging bytes");
        let cache = allocate_placed_buffer(4).expect("allocates BF8 cache bytes");
        let bf8 = codec(&NumericFormat::BFloat8);
        try_write_placed_buffer_f32(&staging, 0, &[1.125, 1.375, 61440.0]).expect("three f32s fit the 12-byte staging buffer");

        narrow_placed_buffer(&staging, 0, &cache, 0, &bf8, 3).expect("narrows");

        assert_eq!(&stored_bytes(&cache, 3), &[0x3c, 0x3e, 0x7c]);
    }

    /// onnxruntime `float8.h:454-462` (`Float8E5M2::ToFloat`): every NaN code widens to the canonical
    /// quiet NaN with its sign. The earlier KV bf8 path kept payloads but was test-only
    /// (`#[cfg(test)] Bf8`), so no production behaviour changed.
    #[test]
    fn bf8_placed_decode_widens_every_nan_code_to_the_canonical_quiet_nan() {
        let buffer = allocate_placed_buffer(4).expect("allocates BF8 fixture bytes");
        let bf8 = codec(&NumericFormat::BFloat8);
        // SAFETY: the buffer holds 4 bytes and nothing else reads it here.
        unsafe { core::ptr::copy_nonoverlapping([0x7du8, 0x7e, 0x7f, 0xfd].as_ptr(), buffer.contents().as_ptr().cast::<u8>(), 4) };

        let decoded = read_placed_buffer(&buffer, 0, &bf8, 4).expect("bf8 decodes");

        let bits: Vec<u32> = decoded.iter().map(|value| value.to_bits()).collect();
        assert_eq!(bits, [0x7fc0_0000, 0x7fc0_0000, 0x7fc0_0000, 0xffc0_0000]);
    }

    #[test]
    fn a_signalling_nan_survives_a_store_and_load_in_the_graph_output_layout() {
        let buffer = allocate_placed_buffer(8).expect("allocates f32 fixture bytes");
        let f32_codec = codec(&NumericFormat::Float32);
        let signalling = [f32::from_bits(0x7f80_0001), f32::from_bits(0xff80_0001)];

        write_placed_buffer(&buffer, 0, &f32_codec, &signalling).expect("f32 encodes");
        let loaded = read_placed_buffer(&buffer, 0, &f32_codec, signalling.len()).expect("f32 decodes");

        let loaded_bits: Vec<u32> = loaded.iter().map(|value| value.to_bits()).collect();
        assert_eq!(loaded_bits, [0x7f80_0001, 0xff80_0001]);
    }

    #[test]
    fn a_q4_0_placed_round_trip_stores_the_host_encode_and_reads_the_host_decode_bit_for_bit() {
        let values = k_like_rows(96);
        let q4_0 = codec(&NumericFormat::Q4_0);
        let packed_len = q4_0.packed_len(values.len()).expect("96 values are three whole blocks");
        let mut host_packed = vec![0u8; packed_len];
        q4_0.encode(&values, &mut host_packed).expect("host encodes q4_0");
        let mut host_slots = vec![core::mem::MaybeUninit::<f32>::uninit(); values.len()];
        q4_0.decode(&host_packed, &mut host_slots).expect("host decodes q4_0");
        let host_decoded: Vec<u32> = host_slots
            .iter()
            // decode returned Ok, which initialises every slot of its output
            .map(|slot| unsafe { slot.assume_init() }.to_bits())
            .collect();
        let buffer = allocate_placed_buffer(packed_len).expect("allocates q4_0 fixture bytes");

        write_placed_buffer(&buffer, 0, &q4_0, &values).expect("q4_0 places");
        let placed = read_placed_buffer(&buffer, 0, &q4_0, values.len()).expect("q4_0 reads back");

        assert_eq!(packed_len, 54);
        assert_eq!(stored_bytes(&buffer, packed_len), host_packed);
        let placed_bits: Vec<u32> = placed.iter().map(|value| value.to_bits()).collect();
        assert_eq!(placed_bits, host_decoded);
    }

    #[test]
    fn a_placed_element_count_that_is_not_whole_blocks_is_an_error_naming_the_format() {
        let q4_0 = codec(&NumericFormat::Q4_0);
        let buffer = allocate_placed_buffer(64).expect("allocates fixture bytes");

        let refused = read_placed_buffer(&buffer, 0, &q4_0, 33);

        assert_eq!(
            refused,
            Err(NumericError::InputNotElementMultiple { format: "q4_0", found: 33, elements: 32 })
        );
    }

    #[test]
    fn narrowing_kv_like_rows_to_q4_0_stores_the_host_encode_bytes() {
        let values = k_like_rows(96);
        let q4_0 = codec(&NumericFormat::Q4_0);
        let packed_len = q4_0.packed_len(values.len()).expect("96 values are three whole blocks");
        let mut host_packed = vec![0u8; packed_len];
        q4_0.encode(&values, &mut host_packed).expect("host encodes q4_0");
        let staging = allocate_placed_buffer(values.len() * 4).expect("allocates the f32 staging rows");
        let cache = allocate_placed_buffer(packed_len).expect("allocates the q4_0 cache bytes");
        try_write_placed_buffer_f32(&staging, 0, &values).expect("the staging buffer holds every row");

        narrow_placed_buffer(&staging, 0, &cache, 0, &q4_0, values.len()).expect("q4_0 narrows");

        assert_eq!(packed_len, 54);
        assert_eq!(stored_bytes(&cache, packed_len), host_packed);
    }

    #[test]
    fn narrowing_a_placed_element_count_that_is_not_whole_blocks_is_an_error_naming_the_format() {
        let q4_0 = codec(&NumericFormat::Q4_0);
        let staging = allocate_placed_buffer(33 * 4).expect("allocates the f32 staging rows");
        let cache = allocate_placed_buffer(64).expect("allocates fixture bytes");

        let refused = narrow_placed_buffer(&staging, 0, &cache, 0, &q4_0, 33);

        assert_eq!(
            refused,
            Err(NumericError::InputNotElementMultiple { format: "q4_0", found: 33, elements: 32 })
        );
    }

    #[test]
    fn writing_a_placed_element_count_that_is_not_whole_blocks_is_an_error_naming_the_format() {
        let q4_0 = codec(&NumericFormat::Q4_0);
        let buffer = allocate_placed_buffer(64).expect("allocates fixture bytes");

        let refused = write_placed_buffer(&buffer, 0, &q4_0, &k_like_rows(33));

        assert_eq!(
            refused,
            Err(NumericError::InputNotElementMultiple { format: "q4_0", found: 33, elements: 32 })
        );
    }

    const SENTINEL: u8 = 0xa5;

    fn sentinel_buffer(len: usize) -> PlacedBuffer {
        let buffer = allocate_placed_buffer(len).expect("allocates sentinel fixture bytes");
        let length = placed_buffer_len(&buffer);
        // SAFETY: the buffer holds `length` bytes of cpu-visible placed memory with no gpu work in flight.
        unsafe { core::ptr::write_bytes(buffer.contents().as_ptr().cast::<u8>(), SENTINEL, length) };
        buffer
    }

    fn assert_all_sentinel(buffer: &PlacedBuffer) {
        let bytes = stored_bytes(buffer, placed_buffer_len(buffer));
        assert!(bytes.iter().all(|byte| *byte == SENTINEL), "a refused placed call wrote: {bytes:02x?}");
    }

    #[test]
    fn writing_at_an_offset_past_the_end_is_refused_and_leaves_the_buffer_untouched() {
        let bf16 = codec(&NumericFormat::BFloat16);
        let buffer = sentinel_buffer(16);
        let past_end = placed_buffer_len(&buffer) + 8;

        let refused = write_placed_buffer(&buffer, past_end, &bf16, &[1.5, 2.5, -3.0, 4.0]);

        assert_eq!(refused, Err(NumericError::OutputSizeMismatch { found: 0, expected: 8 }));
        assert_all_sentinel(&buffer);
    }

    #[test]
    fn writing_at_an_offset_that_leaves_a_window_shorter_than_the_packed_range_is_refused() {
        let bf16 = codec(&NumericFormat::BFloat16);
        let buffer = sentinel_buffer(16);
        let offset = placed_buffer_len(&buffer) - 4;

        let refused = write_placed_buffer(&buffer, offset, &bf16, &[1.5, 2.5, -3.0, 4.0]);

        assert_eq!(refused, Err(NumericError::OutputSizeMismatch { found: 4, expected: 8 }));
        assert_all_sentinel(&buffer);
    }

    #[test]
    fn reading_at_an_offset_past_the_end_is_refused_and_leaves_the_buffer_untouched() {
        let bf16 = codec(&NumericFormat::BFloat16);
        let buffer = sentinel_buffer(16);
        let past_end = placed_buffer_len(&buffer) + 8;

        let refused = read_placed_buffer(&buffer, past_end, &bf16, 4);

        assert_eq!(refused, Err(NumericError::OutputSizeMismatch { found: 4, expected: 0 }));
        assert_all_sentinel(&buffer);
    }

    #[test]
    fn reading_at_an_offset_that_leaves_a_window_shorter_than_the_packed_range_is_refused() {
        let bf16 = codec(&NumericFormat::BFloat16);
        let buffer = sentinel_buffer(16);
        let offset = placed_buffer_len(&buffer) - 4;

        let refused = read_placed_buffer(&buffer, offset, &bf16, 4);

        assert_eq!(refused, Err(NumericError::OutputSizeMismatch { found: 4, expected: 2 }));
        assert_all_sentinel(&buffer);
    }

    #[test]
    fn narrowing_to_a_destination_offset_past_the_end_is_refused_and_leaves_both_buffers_untouched() {
        let bf16 = codec(&NumericFormat::BFloat16);
        let staging = sentinel_buffer(16);
        let cache = sentinel_buffer(16);
        let past_end = placed_buffer_len(&cache) + 8;

        let refused = narrow_placed_buffer(&staging, 0, &cache, past_end, &bf16, 4);

        assert_eq!(refused, Err(NumericError::OutputSizeMismatch { found: 0, expected: 8 }));
        assert_all_sentinel(&cache);
        assert_all_sentinel(&staging);
    }

    #[test]
    fn narrowing_to_a_destination_offset_with_a_window_shorter_than_the_packed_range_is_refused() {
        let bf16 = codec(&NumericFormat::BFloat16);
        let staging = sentinel_buffer(16);
        let cache = sentinel_buffer(16);
        let offset = placed_buffer_len(&cache) - 4;

        let refused = narrow_placed_buffer(&staging, 0, &cache, offset, &bf16, 4);

        assert_eq!(refused, Err(NumericError::OutputSizeMismatch { found: 4, expected: 8 }));
        assert_all_sentinel(&cache);
        assert_all_sentinel(&staging);
    }

    #[test]
    fn narrowing_from_a_source_offset_past_the_end_is_refused_and_leaves_both_buffers_untouched() {
        let bf16 = codec(&NumericFormat::BFloat16);
        let staging = sentinel_buffer(16);
        let cache = sentinel_buffer(16);
        let past_end = placed_buffer_len(&staging) + 8;

        let refused = narrow_placed_buffer(&staging, past_end, &cache, 0, &bf16, 4);

        assert_eq!(refused, Err(NumericError::OutputSizeMismatch { found: 0, expected: 4 }));
        assert_all_sentinel(&cache);
        assert_all_sentinel(&staging);
    }

    #[test]
    fn narrowing_from_a_source_offset_with_fewer_rows_than_requested_is_refused() {
        let bf16 = codec(&NumericFormat::BFloat16);
        let staging = sentinel_buffer(16);
        let cache = sentinel_buffer(16);
        let offset = placed_buffer_len(&staging) - 8;

        let refused = narrow_placed_buffer(&staging, offset, &cache, 0, &bf16, 4);

        assert_eq!(refused, Err(NumericError::OutputSizeMismatch { found: 2, expected: 4 }));
        assert_all_sentinel(&cache);
        assert_all_sentinel(&staging);
    }

    #[test]
    fn narrowing_from_a_source_offset_that_is_not_a_whole_f32_is_refused_and_leaves_both_buffers_untouched() {
        let bf16 = codec(&NumericFormat::BFloat16);
        for misaligned in [1usize, 2, 3] {
            let staging = sentinel_buffer(32);
            let cache = sentinel_buffer(16);

            let refused = narrow_placed_buffer(&staging, misaligned, &cache, 0, &bf16, 4);

            assert_eq!(
                refused,
                Err(NumericError::InputNotBlockMultiple { format: "binary32", found: misaligned, block_bytes: 4 }),
                "source offset {misaligned}"
            );
            assert_all_sentinel(&cache);
            assert_all_sentinel(&staging);
        }
    }

    /// Deterministic roughly-gaussian rows with a few outlier channels, the shape a K projection has.
    fn k_like_rows(count: usize) -> Vec<f32> {
        let mut state = 0x9e37_79b9_u32;
        let mut uniform = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32
        };
        (0..count)
            .map(|index| {
                let gaussian: f32 = (0..6).map(|_| uniform()).sum::<f32>() - 3.0;
                if index % 37 == 0 { gaussian * 8.0 } else { gaussian }
            })
            .collect()
    }
}
