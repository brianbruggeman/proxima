use super::*;

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
    plan_with_placed_inputs(program, symbols, blocks, outputs, numeric_policy, &[])
}

pub(super) fn plan_with_placed_inputs(
    program: &[Op],
    symbols: &[u64],
    blocks: &[QuantizedBlock<'_>],
    outputs: &[NodeId],
    numeric_policy: NumericPolicy,
    placed_input_nodes: &[NodeId],
) -> Result<Plan, MetalError> {
    #[cfg(feature = "instrument")]
    let prepare_started = read_ticks();
    let prepared = prepare(
        program,
        symbols,
        blocks,
        outputs,
        numeric_policy,
        placed_input_nodes,
    )?;
    #[cfg(feature = "instrument")]
    {
        counter!(PREPARE_CALLS, 1);
        counter!(PREPARE_TICKS, elapsed_ticks(prepare_started));
    }
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
        .map(|node| gpu_dtype(program, &prepared.index_nodes, *node))
        .collect();
    Ok(Plan {
        program: program.to_vec(),
        prepared,
        packed_operands,
        block_dtypes,
        resident_nodes: BTreeSet::new(),
        math_mode: numeric_policy_as_metal_math_mode(numeric_policy),
        numeric_policy,
        dispatch_type: DispatchType::default(),
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
    })
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
    let debug_timing = std::env::var_os("PROXIMA_DEBUG_SEGMENT_HOST").is_some();
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
    if std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some() {
        eprintln!(
            "effective expert source nodes={:?}",
            effective_expert_buffers.keys().collect::<Vec<_>>()
        );
        for index in 0..plan.program.len().min(24) {
            eprintln!("plan op {} name={:?}", index, plan.program[index].name());
        }
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
        if let Some((buffer, offset)) = cross_plan_resident_reuse(
            resident_name,
            block_identity_key(&block).0 as *const c_void,
            block_identity_key(&block).1,
        )? {
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
            // on the GPU at the read (`msl::BF16_UNPACK_MSL`), never on the
            // host.
            // `Q3_K` uploads its raw super-block bytes unchanged, same as
            // every other packed codec below -- `msl::PackedCodec::Q3K`'s
            // own unpack kernel (`q3k_element`) reads them at the GPU side.
            QuantizedBlock::Q3K(bytes)
            | QuantizedBlock::Q4K(bytes)
            | QuantizedBlock::Q5K(bytes)
            | QuantizedBlock::Q6K(bytes)
            | QuantizedBlock::Q8_0(bytes)
            | QuantizedBlock::Q4_0(bytes)
            | QuantizedBlock::Q5_1(bytes)
            | QuantizedBlock::Q2K(bytes)
            | QuantizedBlock::Iq4Nl(bytes)
            | QuantizedBlock::Iq2Xs(bytes)
            | QuantizedBlock::Iq3Xxs(bytes)
            | QuantizedBlock::Float16(bytes)
            | QuantizedBlock::BFloat16(bytes) => {
                upload_packed_bytes(&device, bytes, resident_name)?
            }
        };
        device_buffers.insert(node, buffer);
    }
    #[cfg(feature = "instrument")]
    if std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some() {
        eprintln!(
            "metal ordinary uploads blocks={ordinary_upload_blocks} bytes={ordinary_upload_bytes}"
        );
    }
    #[cfg(feature = "instrument")]
    counter!(BLOCK_UPLOAD_TICKS, elapsed_ticks(block_upload_started));

    let command_buffer = queue
        .commandBuffer()
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
            let byte_length = bound_output_len(bound).max(1) * bound.dtype.size_bytes();
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
    if debug_timing {
        eprintln!(
            "metal expert prepared_nodes={:?}",
            prepared
                .resolved
                .iter()
                .map(|bound| bound.node)
                .collect::<Vec<_>>()
        );
        for (position, retired) in prepared.retires.iter().enumerate() {
            if retired.contains(&NodeId(16))
                || retired.contains(&NodeId(18))
                || retired.contains(&NodeId(22))
            {
                eprintln!(
                    "metal expert retirement node16_or_18 position={position} nodes={retired:?}"
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
                eprintln!(
                    "metal expert node22 read_at_position={position} bound={:?}",
                    bound.node
                );
            }
            if matches!(bound.node, NodeId(16) | NodeId(18)) {
                eprintln!(
                    "metal expert retirement position={} node={:?} retires={:?}",
                    position, bound.node, prepared.retires[position]
                );
            }
        }
    }
    for (position, bound) in prepared.resolved.iter().enumerate() {
        if std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some() {
            eprintln!(
                "resolved bound node={:?} kind={} operands={:?}",
                bound.node,
                bound.kind.name(),
                bound
                    .operands()
                    .iter()
                    .map(|(node, _, lookup)| (
                        *node,
                        lookup.is_some(),
                        plan.program.get(node.0 as usize).and_then(Op::name)
                    ))
                    .collect::<Vec<_>>()
            );
        }
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
            None,
            None,
            None,
            expert_buffers,
        )?;
        if let Some((fault_buffer, gathers)) = fault {
            pending_faults.push((bound, fault_buffer, gathers));
        }
        if debug_timing && matches!(bound.node, NodeId(22) | NodeId(30)) {
            eprintln!(
                "metal expert post_encode node={:?} has_buffer={} buffer_keys={:?}",
                bound.node,
                device_buffers.contains_key(&bound.node),
                device_buffers.keys().copied().collect::<Vec<_>>()
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
            counter!(
                RETIRE_SCAN_CALLS,
                prepared.retires[position].len() as u64
            );
            counter!(RETIRE_SCAN_TICKS, elapsed_ticks(retire_scan_started));
        }
    }
    encoder.finish();

    #[cfg(feature = "instrument")]
    let gpu_exec_started = read_ticks();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    #[cfg(feature = "instrument")]
    {
        counter!(GPU_EXEC_CALLS, 1);
        counter!(GPU_EXEC_TICKS, elapsed_ticks(gpu_exec_started));
    }

    for (bound, fault_buffer, gathers) in &pending_faults {
        check_gather_fault(bound, fault_buffer, *gathers)?;
    }

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
        if std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some() {
            eprintln!(
                "expert source not matched bound={:?} kind={} source_nodes={:?}",
                bound.node,
                bound.kind.name(),
                expert_buffers.keys().collect::<Vec<_>>()
            );
            eprintln!(
                "expert source bound_operands={:?}",
                bound
                    .operands()
                    .iter()
                    .map(|(node, _, lookup)| (*node, lookup.is_some()))
                    .collect::<Vec<_>>()
            );
        }
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
    if std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some() {
        eprintln!(
            "metal expert source node={node:?} entries={} selected_experts={} selected_payload_bytes={}",
            source.entries().len(),
            source
                .selected_expert_ids()
                .map_or(source.entries().len(), <[_]>::len),
            payload_bytes.len()
        );
        if let Some(selected) = source.selected_expert_ids() {
            eprintln!("metal selected expert ids={selected:?}");
            for expert in selected {
                if let Some(descriptor) = descriptor_records.get(*expert as usize) {
                    eprintln!(
                        "metal selected descriptor expert={} codec={:?} epoch={} offset={} length={}",
                        descriptor.expert_index,
                        descriptor.codec,
                        descriptor.epoch,
                        descriptor.byte_offset,
                        descriptor.byte_length
                    );
                }
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
    if std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some() {
        eprintln!(
            "metal expert source staged node={node:?} payload_bytes={} descriptor_bytes={} elapsed_us={}",
            payload_bytes.len(),
            descriptor_bytes.len(),
            stage_started.elapsed().as_micros()
        );
        for descriptor in descriptor_records.iter().take(8) {
            eprintln!(
                "metal expert descriptor expert={} codec={:?} offset={} length={} shape={}x{}",
                descriptor.expert_index,
                descriptor.codec,
                descriptor.byte_offset,
                descriptor.byte_length,
                descriptor.out_dim,
                descriptor.in_dim,
            );
        }
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
        // The previous buffer came from a copying upload, so its storage is
        // independent of the host vector and can be overwritten after the
        // prior command buffer has completed.
        let destination = buffer.contents().as_ptr().cast::<u8>();
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
            QuantizedBlock::Q2K(_) => 1_u64,
            QuantizedBlock::Q3K(_) => 2_u64,
            QuantizedBlock::Q4K(_) => 3_u64,
            QuantizedBlock::Q5K(_) => 4_u64,
            QuantizedBlock::Q6K(_) => 5_u64,
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
            QuantizedBlock::Q2K(_)
                | QuantizedBlock::Q3K(_)
                | QuantizedBlock::Q4K(_)
                | QuantizedBlock::Q6K(_)
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
    if std::env::var_os("PROXIMA_DEBUG_METAL_STAGES").is_some() {
        eprintln!(
            "expert_source_stage_ms={} source_nodes={} staged_nodes={}",
            upload_elapsed.as_secs_f64() * 1e3,
            expert_sources.len(),
            buffers.len(),
        );
    }
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
            let mut cache = cache.try_borrow_mut().map_err(|_| {
                MetalError::ExpertSourceUnsupported {
                    node: *node,
                    reason: "expert source cache is already borrowed",
                }
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
                if std::env::var_os("PROXIMA_DEBUG_EXPERT_SOURCE_CACHE").is_some() {
                    eprintln!(
                        "metal expert source cache miss node={node:?} signature={signature} payload_bytes={}",
                        cache
                            .get(&cache_key)
                            .map_or(0, |(_, staged)| staged.buffers.payloads.length())
                    );
                }
            } else {
                #[cfg(feature = "instrument")]
                counter!(EXPERT_SOURCE_CACHE_HITS, 1);
                if std::env::var_os("PROXIMA_DEBUG_EXPERT_SOURCE_CACHE").is_some() {
                    eprintln!(
                        "metal expert source cache hit node={node:?} signature={signature} payload_bytes={}",
                        cache
                            .get(&cache_key)
                            .map_or(0, |(_, staged)| staged.buffers.payloads.length())
                    );
                }
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
    let debug_timing = std::env::var_os("PROXIMA_DEBUG_SEGMENT_HOST").is_some();
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
    if debug_timing {
        eprintln!(
            "metal expert execute resolve_us={} execute_us={}",
            resolve_elapsed_us,
            execute_started.elapsed().as_micros(),
        );
    }
    result
}

/// A device buffer a CALLER allocates, owns, and keeps alive across multiple
/// [`execute_plan_with_placements`] calls — the type this module's other
/// buffers (`MetalBuffer`, private) never had to be public for, since
/// `execute_plan`/`execute_plan_op_timed` allocate and own every buffer
/// themselves. Get one from [`allocate_placed_buffer`]; read one back with
/// [`read_placed_buffer_f32`].
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
    device
        .newBufferWithLength_options(byte_len.max(1), MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| MetalError::CompileFailed {
            log: "device refused to allocate a placed buffer".to_string(),
        })
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
#[cfg(feature = "metal-output-placement")]
pub fn zero_placed_buffer(buffer: &PlacedBuffer, byte_len: usize) {
    let pointer = buffer.contents();
    // SAFETY: `buffer` is `storageModeShared` (`allocate_placed_buffer`'s
    // own contract) and `byte_len` is the caller's own allocated length
    // for it (mirrors `read_placed_buffer_f32`'s own SAFETY comment), so
    // this is a valid, CPU-visible, mutable byte slice for the duration of
    // this call.
    let slots = unsafe { core::slice::from_raw_parts_mut(pointer.as_ptr().cast::<u8>(), byte_len) };
    slots.fill(0);
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
/// # Panics
/// Never panics; an out-of-bounds `byte_offset`/`element_count` pair is the
/// caller's own contract to keep (documented on [`execute_plan_with_placements`]),
/// the same trust boundary `bind_buffers`' SAFETY comment already states for
/// every kernel-side binding.
#[cfg(feature = "metal-output-placement")]
#[must_use]
pub fn read_placed_buffer_f32(
    buffer: &PlacedBuffer,
    byte_offset: usize,
    element_count: usize,
) -> Vec<f32> {
    let pointer = buffer.contents();
    // SAFETY: `storageModeShared` is CPU-visible once the command buffer
    // that wrote it has completed, and `execute_plan_with_placements` always
    // `waitUntilCompleted`s before returning. `byte_offset`/`element_count`
    // staying inside `buffer`'s allocated length is the caller's contract —
    // this function has no independent way to learn that length's meaning
    // (a caller-owned buffer may hold several placed nodes at once).
    unsafe {
        let base = pointer.as_ptr().cast::<u8>().add(byte_offset).cast::<f32>();
        core::slice::from_raw_parts(base, element_count)
    }
    .to_vec()
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
    pub(super) fn needs_barrier(&self, inputs: &[Id], output: Option<Id>) -> bool {
        self.classify(inputs, output) != HazardClass::None
    }

    /// [`needs_barrier`]'s own decomposition into which single hazard
    /// explains it, checked in the same order that `||` chain implies (a
    /// RAW input hazard first).
    pub(super) fn classify(&self, inputs: &[Id], output: Option<Id>) -> HazardClass {
        if inputs.iter().any(|input| self.written.contains(input)) {
            return HazardClass::Raw;
        }
        match output {
            Some(out) if self.written.contains(&out) => HazardClass::Waw,
            Some(out) if self.read.contains(&out) => HazardClass::War,
            _ => HazardClass::None,
        }
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
    pub(super) fn record(&mut self, inputs: &[Id], output: Option<Id>) {
        if let Some(out) = output {
            self.written.insert(out);
        }
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
}

impl HazardState {
    fn new() -> Self {
        Self {
            tracker: HazardTracker::new(),
            inputs: Vec::new(),
        }
    }

    pub(super) fn reset(&mut self) {
        self.tracker.reset();
        self.inputs.clear();
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
/// mirror of the loop body that could silently drift from it. `output` is
/// the identity of the buffer this op is about to write, already resolved
/// (placement -> arena slot -> fresh allocation) by the caller before this
/// runs, since every bound op writes exactly one device buffer. Returns
/// whether the caller must emit a barrier before encoding this op.
pub(super) fn hazard_step<Id: Eq + core::hash::Hash + Copy>(
    tracker: &mut HazardTracker<Id>,
    inputs: &[Id],
    output: Id,
) -> bool {
    let needs_barrier = tracker.needs_barrier(inputs, Some(output));
    if needs_barrier {
        tracker.reset();
    }
    tracker.record(inputs, Some(output));
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

