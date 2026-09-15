use super::*;

/// `recycle` is the same pool [`proxima_tensor::cpu::evaluate_with_scratch`]
/// takes: a caller done reading a PREVIOUS call's `Evaluated` hands its
/// storage back with [`Evaluated::into_scratch`], and this call pops one
/// buffer from the pool (if any) to reuse for the root output's read-back
/// instead of allocating fresh. An empty pool (every call before this
/// landing was, implicitly) always allocates fresh, exactly as before -- see
/// `finish`'s own doc for the exact conditions a popped buffer is actually
/// reused under.
///
/// # Errors
/// Propagates block-codec and Metal driver failures, same as [`execute_plan`].
#[cfg(feature = "metal-output-placement")]
pub fn execute_plan_with_placements(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
    output_placements: &[(NodeId, &PlacedBuffer, usize)],
    recycle: &mut Vec<Vec<f32>>,
) -> Result<Evaluated, MetalError> {
    execute_plan_with_placements_inner(
        plan,
        blocks,
        input_placements,
        output_placements,
        recycle,
        &BTreeMap::new(),
    )
}

#[cfg(feature = "metal-output-placement")]
pub(super) fn execute_plan_with_placements_inner(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
    output_placements: &[(NodeId, &PlacedBuffer, usize)],
    recycle: &mut Vec<Vec<f32>>,
    expert_buffers: &BTreeMap<NodeId, ExpertSourceBuffers>,
) -> Result<Evaluated, MetalError> {
    let prepared = &plan.prepared;
    let packed_operands = &plan.packed_operands;
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
    let input_placed: BTreeMap<NodeId, (&PlacedBuffer, usize)> = input_placements
        .iter()
        .map(|(node, buffer, offset)| (*node, (*buffer, *offset)))
        .collect();
    if std::env::var_os("PROXIMA_DEBUG_PLACEMENT_KEYS").is_some() {
        eprintln!(
            "metal placement keys inputs={:?} blocks={:?}",
            input_placed.keys().collect::<Vec<_>>(),
            prepared.block_nodes,
        );
    }
    let output_placed: BTreeMap<NodeId, (&PlacedBuffer, usize)> = output_placements
        .iter()
        .map(|(node, buffer, offset)| (*node, (*buffer, *offset)))
        .collect();
    let (device, queue) = device_and_queue()?;
    // On a plan-cache HIT this is a no-op (`resolve_steps` checks staleness
    // and returns immediately): the per-position loop below indexes
    // `plan.resolved_steps` instead of every step re-deriving
    // `kernel_cache_key`/`kernel_dispatch_shape`/`pipeline_for`'s own cache
    // key. Only a genuine plan-cache MISS or a `set_math_mode` change pays
    // this once, up front, rather than once per op per step.
    resolve_steps(&device, plan)?;
    let resolved_steps = plan.resolved_steps.borrow();

    // plan-owned and never rebuilt fresh -- see `Plan::device_buffers`'s own
    // doc for why this alone (independent of the per-node reuse skip below)
    // already removes ROW 303's residual `BTreeMap::new()` allocation on a
    // warm call: every key this loop and the position loop below touch is
    // still written fresh every call exactly as before, only the map's own
    // heap nodes now outlive one call instead of being dropped with it.
    let mut device_buffers = plan.device_buffers.borrow_mut();
    for (node, buffers) in &effective_expert_buffers {
        device_buffers.insert(*node, (buffers.payloads.clone(), buffers.payload_offset));
    }
    let mut block_identity = plan.block_identity.borrow_mut();
    if block_identity.len() != prepared.block_nodes.len() {
        block_identity.resize(prepared.block_nodes.len(), None);
    }
    for (index, ((node, block), dtype)) in prepared
        .block_nodes
        .iter()
        .zip(blocks.iter())
        .zip(plan.block_dtypes.iter())
        .enumerate()
    {
        if !prepared.live_block_inputs[index] {
            block_identity[index] = None;
            continue;
        }
        if effective_expert_buffers.contains_key(node) {
            block_identity[index] = None;
            continue;
        }
        // an input-placed node skips the host round trip entirely: its
        // buffer is already on the device, owned by the caller, and this
        // call's own `block` entry for it (still required, positionally, to
        // keep `block_nodes.iter().zip(blocks.iter())` aligned) is unused.
        // The caller's own offset travels in the same `DeviceBuffer` tuple
        // every other node's buffer carries -- `buffer_for`/`bind_buffers`
        // read it generically, with no separate offset map required.
        if let Some((buffer, offset)) = input_placed.get(node) {
            device_buffers.insert(*node, ((*buffer).clone(), *offset));
            block_identity[index] = None;
            continue;
        }
        // The source table is the typed execution boundary.  Only omit the
        // packed expert stack when this call actually supplies substituted
        // expert buffers; configuration belongs to the serving plan, not an
        // ambient environment read in the Metal backend.  With no source
        // table, bind the checkpoint's ordinary expert blocks unchanged.
        if !effective_expert_buffers.is_empty()
            && plan.program[node.0 as usize]
                .name()
                .is_some_and(|name| name.contains("_exps.weight"))
        {
            // Router segments carry the full program input inventory, but
            // expert bytes are supplied only by the gather source table.
            // Do not bind the original packed stack on this placement path.
            block_identity[index] = None;
            continue;
        }
        let current_identity = block_identity_key(block);
        let resident = plan.resident_nodes.contains(node);
        #[cfg(feature = "instrument")]
        if let Some(previous_identity) = block_identity[index]
            && previous_identity != current_identity
        {
            counter!(MAPPING_REBOUND_BLOCKS, 1);
            debug!(
                block_name = %plan.program[node.0 as usize].name().unwrap_or("<unnamed>"),
                previous_pointer = previous_identity.0 as u64,
                previous_length = previous_identity.1 as u64,
                current_pointer = current_identity.0 as u64,
                current_length = current_identity.1 as u64,
                resident,
                "mapping_rebound_block: block_identity_key disagreed with the prior step's"
            );
        }
        if block_buffer_reusable(resident, block_identity[index], current_identity)
            && device_buffers.contains_key(node)
        {
            // this position's buffer is already correct in the plan-owned
            // map from a PRIOR call (never retired -- see the retirement
            // loop's own `resident_nodes` skip below) -- no upload, no
            // `device_buffers` write, this call's only cost for this node
            // is the identity comparison just made.
            continue;
        }
        block_identity[index] = Some(current_identity);
        let resident_name = resident_name(plan, *node);
        // a brand-new `Plan` (a KV-bucket-boundary reshape, say) starts with
        // an empty `device_buffers`/`block_identity` of its OWN, so the
        // fast-skip above always misses on plan 1 of a node's life even
        // though the GLOBAL, name-keyed caches
        // (`NOCOPY_BUFFERS`/`RESIDENT_BUFFERS`/the checkpoint mapping)
        // already hold this exact host range from the PRIOR plan. Consult
        // them here, before counting this node as "offered", so a reshape
        // does not re-walk every resident weight block through the upload
        // path just to have it resolve to the same cache hit it already was.
        if let Some((buffer, offset)) = cross_plan_resident_reuse(
            resident_name,
            current_identity.0 as *const c_void,
            current_identity.1,
        )? {
            device_buffers.insert(*node, (buffer, offset));
            continue;
        }
        // an input-placed node above never reaches here, so `BLOCK_OFFERED_BYTES`
        // fires only for a block that genuinely takes the host round trip
        // below -- the same "offered for upload" meaning `execute_plan`'s own
        // fire site carries, extended to this placed-buffer entry point so
        // the partition identity (`BLOCK_COPIED_BYTES + BLOCK_NOCOPY_BOUND_BYTES
        // + BLOCK_OFFSET_BOUND_BYTES == BLOCK_OFFERED_BYTES`) holds on this
        // loop too, not just `execute_plan`'s.
        #[cfg(feature = "instrument")]
        {
            counter!(BLOCK_UPLOAD_CALLS, 1);
            counter!(BLOCK_OFFERED_BYTES, block_byte_len(block) as u64);
        }
        let buffer = match block {
            QuantizedBlock::Float32(data) => upload_block(
                &device,
                data,
                *node,
                *dtype,
                plan.program[node.0 as usize].name(),
                resident_name,
            )?,
            QuantizedBlock::Int32(data) => {
                upload_block_int32_as_float(&device, data, resident_name)?
            }
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
        device_buffers.insert(*node, buffer);
    }

    #[cfg(feature = "instrument")]
    let kind_filter = KindFilter::from_env()?;
    #[cfg(feature = "instrument")]
    if let Some(filter) = &kind_filter {
        validate_kind_filter(filter, prepared, packed_operands, &plan.program)?;
    }

    let command_buffer = queue
        .commandBuffer()
        .ok_or_else(|| MetalError::CompileFailed {
            log: "command queue refused to hand out a command buffer".to_string(),
        })?;
    // `Concurrent` lets independent dispatches (e.g. Q/K/V from one normed
    // input) overlap instead of draining the pipeline between every op --
    // see [`DispatchType`]'s own doc for why that requires [`HazardTracker`]
    // below to insert the barriers `Serial` gives for free by never
    // overlapping any two dispatches in the first place.
    let dispatch_type = plan.dispatch_type;
    let encoder = EncoderGuard::new(
        command_buffer
            .computeCommandEncoderWithDispatchType(dispatch_type.as_mtl())
            .ok_or_else(|| MetalError::CompileFailed {
                log: "command buffer refused to hand out a compute encoder".to_string(),
            })?,
    );
    // plan-owned, reused across every call against this `Plan` (`HazardState`'s
    // own doc) -- `reset` clears both the tracker's sets and the input scratch
    // without releasing their capacity, so a warm call after the first never
    // pays their first-insert allocation again.
    let mut hazard_state = plan.hazard_state.borrow_mut();
    hazard_state.reset();
    // reborrowed as a plain `&mut` so `tracker`/`inputs` split into disjoint
    // field borrows below -- the borrow checker cannot see through
    // `RefMut`'s own `Deref`/`DerefMut` to know the two fields are disjoint.
    let hazard_state = &mut *hazard_state;

    // Resolved once per plan, lazily: needs only `device_buffers`' weight
    // identity (ROW 572 -- activation admits by NODE identity, output is
    // allocated at the leader's own first encode, never at plan-resolution
    // time, since `BufferArena` cannot hand out a slot shared across two
    // simultaneously-live positions; see `ensure_merged_group_resolved`).
    // A plan with no eligible group still costs one `Option::is_some()`
    // check per call after the first.
    #[cfg(feature = "metal-horizontal-merge")]
    ensure_merged_dispatches(&device, plan, &device_buffers)?;

    // diagnostic-only, `instrument`-gated, always emitted at `trace` level
    // (default-off via the runtime filter, never a bespoke env var): each
    // output-placed node's WRITE position and each of its aliased readers'
    // own position, the direct evidence a write-before-read ordering claim
    // needs (this function's own doc, "Within-call aliasing"). Silent unless
    // a caller raises `RUST_LOG` to `trace` for this target.
    let mut pending_faults: Vec<PendingFault<'_>> = Vec::new();
    for (position, bound) in prepared.resolved.iter().enumerate() {
        #[cfg(feature = "instrument")]
        {
            if output_placed.contains_key(&bound.node) {
                trace!(position, node = ?bound.node, "output-placed node write");
            }
            for (operand, _, _) in bound.operands() {
                if input_placed.contains_key(operand) {
                    trace!(position, node = ?bound.node, reads = ?operand, "placed-input node read");
                }
            }
        }
        // ROW 539: whether a `placement` below comes from the arena rather
        // than a caller-owned output-placed buffer -- the arena branch is
        // the one whose slot may be shared with an earlier, unrelated
        // position (`BufferArena::slot_is_recycled`), while an output-placed
        // buffer is a persistent identity the caller owns for the plan's
        // whole life.
        #[cfg(feature = "instrument")]
        let placement_is_arena_sourced = !output_placed.contains_key(&bound.node);
        let placement = match output_placed.get(&bound.node).copied() {
            Some(placement) => Some(placement),
            None => arena_placement(plan, position)?,
        };
        // row 555: `bound.node` above is the fused op's primary "out" node,
        // never the absorbed `state_out` node it also owns -- a caller's
        // `state_out` placement lives under a DIFFERENT key in the same
        // map, and `encode_op`'s own doc names why skipping this silently
        // discarded recurrent state on every fused `GatedDeltaNet` dispatch.
        let state_out_placement = match &bound.kind {
            BoundOpKind::GatedDeltaNet { state_out, .. } => {
                output_placed.get(state_out).copied()
            }
            _ => None,
        };
        // `ablation_skip` is `false` on every non-`instrument` build (the
        // `match` folds to the literal at compile time, so this costs
        // nothing in production) and `false` on every `instrument` build
        // where `PROXIMA_METAL_KIND_FILTER` is unset -- the only way into
        // the `register_skipped_output` arm below is an operator explicitly
        // setting that env var, which never happens outside this ablation's
        // own harness.
        #[cfg(feature = "instrument")]
        let ablation_skip = match &kind_filter {
            Some(filter) => !filter.matches(
                classify_kind(bound, packed_operands),
                bound_weight_family(bound, &plan.program).as_deref(),
            ),
            None => false,
        };
        #[cfg(not(feature = "instrument"))]
        let ablation_skip = false;

        #[cfg(feature = "metal-horizontal-merge")]
        let merged_this_position = handle_merged_position(
            &device,
            &encoder,
            plan,
            position,
            bound,
            dispatch_type,
            &mut device_buffers,
            hazard_state,
            &output_placed,
        )?;
        #[cfg(not(feature = "metal-horizontal-merge"))]
        let merged_this_position = false;

        if merged_this_position {
            // encoded (or, for a z>0 member, hazard-recorded with no
            // dispatch) entirely inside `handle_merged_position` -- nothing
            // left to do here but fall through to the retirement loop below,
            // exactly like the ordinary path's own `else` arm does.
        } else if ablation_skip {
            // `ablation_skip` is always `false` on a non-`instrument` build
            // (see its own binding above), so this arm never runs there --
            // `register_skipped_output` itself is `instrument`-gated and
            // does not exist to call outside it.
            #[cfg(feature = "instrument")]
            register_skipped_output(&device, &mut device_buffers, bound, placement)?;
        } else {
            // A skipped op above never reaches this hazard tracking either
            // -- it neither reads nor writes a buffer THIS command buffer
            // touches, so the tracker must see it as absent: no
            // `hazard_inputs`/`hazard_output` collected, no
            // `needs_barrier`/`record` call, no barrier emitted on its
            // account.
            // `Serial` never overlaps two dispatches, so no hazard this
            // tracker catches could ever race -- skip it entirely rather
            // than pay the resolve/record bookkeeping for barriers that
            // would never fire.
            let resolved_step = resolved_steps
                .as_ref()
                .and_then(|resolved| resolved.steps.get(position));
            let resolved_output: Option<DeviceBuffer> = if dispatch_type == DispatchType::Concurrent
            {
                // The hazard read set is derived from `bindings` -- the exact
                // list `bind_buffers` will bind for this op -- rather than
                // enumerated separately from `bound.all_read_sources()`: ROW
                // 323 was exactly those two enumerations drifting apart when
                // `msl::bindings` grew a source (a fused epilogue operand)
                // this hazard check had not been taught to see. Reading
                // `bindings` itself makes that class of drift impossible --
                // there is only one list, and both the encoder bind loop
                // (`bind_buffers`) and this hazard walk read the same one.
                let owned_bindings: Vec<Binding>;
                let bindings_for_hazard: &[Binding] = match resolved_step {
                    Some(step) => step.bindings.as_slice(),
                    None => {
                        let (bindings, _grid) =
                            kernel_dispatch_shape(bound, packed_operands, plan.numeric_policy)?;
                        owned_bindings = bindings;
                        owned_bindings.as_slice()
                    }
                };
                resolve_hazard_inputs_into(
                    crate::msl::hazard_read_nodes(bindings_for_hazard),
                    &device_buffers,
                    &mut hazard_state.inputs,
                )?;
                // resolved ONCE, before the hazard check, from the exact same
                // placement -> arena -> fresh-allocation chain `encode_op`
                // would otherwise pick independently below: `needs_barrier`
                // and `record` used to see two different identities for a
                // fresh allocation (the check saw `None`, since nothing is
                // known before `encode_op` allocates; the record afterward
                // saw the real pointer), so a WAW/WAR hazard against an
                // address Metal happens to reuse for that allocation could
                // never be caught. Allocating here and handing the SAME
                // buffer into `encode_op` via `placement` closes that gap.
                let resolved: DeviceBuffer = match placement {
                    Some((buffer, offset)) => (buffer.clone(), offset),
                    None => (
                        allocate_buffer(&device, bound_output_len(bound), bound.dtype)?,
                        0,
                    ),
                };
                // the write side of the same "derived from bindings" guarantee:
                // whenever `bindings_for_hazard` names an explicit
                // `Binding::Output`, it must name THIS op's own node -- if it
                // ever didn't, the buffer the hazard tracker records as
                // written and the buffer `bind_buffers` actually binds as
                // this op's output would be two different things, which is a
                // worse bug than the one this refactor closes. Redesign §4c:
                // a `CachedAttention` split kernel under `ContextSplitMerge`
                // has NO `Binding::Output` at all (`Binding::Scratch`
                // replaces it -- that kernel writes scratch, never
                // `bound.node`'s own buffer) -- `hazard_write_node` returning
                // `None` there is the honest, by-design case, not a drift:
                // `resolved` below is still `bound.node`'s own real output
                // buffer (from `placement`, independent of `bindings`), and
                // the merge dispatch this position's `encode_op` call also
                // issues is what actually writes it, in the same encoder,
                // immediately after the split.
                debug_assert!(
                    matches!(
                        crate::msl::hazard_write_node(bindings_for_hazard),
                        Some(node) if node == bound.node
                    ) || crate::msl::hazard_write_node(bindings_for_hazard).is_none()
                );
                let hazard_output = Retained::as_ptr(&resolved.0);
                // Read-only, computed BEFORE `hazard_step` mutates the
                // tracker below, so it sees the exact same state
                // `hazard_step`'s own internal check will -- `classify`
                // never mutates, so precomputing it here for the counter
                // breakdown cannot change what `hazard_step` decides.
                #[cfg(feature = "instrument")]
                let hazard_class = hazard_state
                    .tracker
                    .classify(&hazard_state.inputs, Some(hazard_output));
                if hazard_step(
                    &mut hazard_state.tracker,
                    &hazard_state.inputs,
                    hazard_output,
                ) {
                    encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                    counter!(BARRIERS_EMITTED, 1);
                    #[cfg(feature = "instrument")]
                    {
                        let arena_recycled = placement_is_arena_sourced
                            && plan
                                .arena
                                .get()
                                .is_some_and(|arena| arena.slot_is_recycled(position));
                        record_hazard_class(hazard_class, arena_recycled);
                    }
                }
                Some(resolved)
            } else {
                None
            };
            let placement = match &resolved_output {
                Some((buffer, offset)) => Some((buffer, *offset)),
                None => placement,
            };
            let uniform_buffer = plan_uniform_buffer(plan, position)?;
            // Redesign §4c: the ONE call site that resolves a scratch
            // buffer for `encode_op`'s two-dispatch `CachedAttention` form
            // -- every other `encode_op` caller passes `None` and rejects a
            // `Binding::Scratch` kernel instead (that function's own doc).
            let attention_scratch =
                attention_scratch_buffer(plan, position)?.map(|buffer| (buffer, 0usize));
            // Only `Concurrent` needs the intra-op scratch write -> read edge
            // routed through the tracker (see `encode_op`'s own doc for
            // `hazard`) -- `Serial` orders the split before the merge for
            // free, the same reason `resolved_output` above is `None` there.
            let hazard =
                (dispatch_type == DispatchType::Concurrent).then_some(&mut hazard_state.tracker);
            let bound_expert_buffers = expert_buffers_for(bound, &effective_expert_buffers)?;
            let fault = encode_op(
                &device,
                &encoder,
                &mut device_buffers,
                bound,
                packed_operands,
                placement,
                state_out_placement,
                uniform_buffer,
                plan.math_mode,
                plan.numeric_policy,
                resolved_step,
                attention_scratch,
                hazard,
                bound_expert_buffers,
            )?;
            if let Some((fault_buffer, gathers)) = fault {
                pending_faults.push((bound, fault_buffer, gathers));
            }
        }
        // explicit liveness exclusion (see this function's doc): a placed
        // node, input or output, is externally owned and always live, so it
        // is never dropped from this call's own bookkeeping map, regardless
        // of what `prepared.retires` (a per-program-only liveness sweep)
        // says. A RESIDENT block node joins that exclusion for the same
        // reason: [`Plan::mark_resident`]'s caller-owned promise means its
        // buffer is live for the plan's whole life, and leaving its entry in
        // [`Plan::device_buffers`] across calls is exactly what lets
        // `block_buffer_reusable` skip re-uploading it next call.
        for retired in &prepared.retires[position] {
            if input_placed.contains_key(retired)
                || output_placed.contains_key(retired)
                || plan.resident_nodes.contains(retired)
            {
                continue;
            }
            // Keep the identity in this command buffer's hazard sets after
            // logical retirement. BufferArena may reuse the same persistent
            // MTLBuffer for a later output before this command buffer
            // commits; clearing it here would erase the WAR/WAW edge that
            // reuse needs. HazardState resets at the next command-buffer
            // call, after every dispatch in this one has retired, so the
            // identity cannot leak across submissions.
            device_buffers.remove(retired);
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

    let placed_output_nodes: BTreeSet<NodeId> = output_placed.keys().copied().collect();
    let evaluated = finish(plan, &device_buffers, &placed_output_nodes, recycle.pop())?;

    // Expert buffers are a per-step source snapshot, not plan-resident
    // weights. `finish` has already read every requested output, so retaining
    // these entries in the plan would keep the all-expert prefill arena alive
    // while the bounded DynaExq table takes over for decode.
    drop(device_buffers);
    {
        let mut plan_buffers = plan.device_buffers.borrow_mut();
        for node in effective_expert_buffers.keys() {
            plan_buffers.remove(node);
        }
    }
    Ok(evaluated)
}

/// [`plan`] against a name-keyed block set — the shape a model binds its
/// weights in. Resolution goes through
/// [`proxima_tensor::resolve_named_blocks`], the same function the CPU
/// evaluator uses, so the two backends cannot disagree about which name is
/// which position.
///
/// # Errors
/// Propagates name-resolution and planning failures.
pub fn plan_named(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'_>)],
    outputs: &[NodeId],
    numeric_policy: NumericPolicy,
) -> Result<Plan, MetalError> {
    let blocks = resolve_named_blocks(program, named)?;
    plan_with_placed_inputs(program, symbols, &blocks, outputs, numeric_policy, &[])
}

/// [`plan`] against named blocks plus caller-owned input placements.
///
/// A placed input does not need a host payload during planning. The planner
/// still validates every ordinary named block, while the execution resolver
/// supplies an empty sentinel for each node whose bytes arrive from its
/// caller-owned [`PlacedBuffer`].
#[cfg(feature = "metal-output-placement")]
pub fn plan_named_with_placed_inputs(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'_>)],
    outputs: &[NodeId],
    numeric_policy: NumericPolicy,
    placed_input_nodes: &[NodeId],
) -> Result<Plan, MetalError> {
    let blocks = resolve_named_blocks_with_placed_nodes(program, named, |node| {
        placed_input_nodes.contains(&node)
    })?;
    plan_with_placed_inputs(
        program,
        symbols,
        &blocks,
        outputs,
        numeric_policy,
        placed_input_nodes,
    )
}

/// [`execute_plan`] against a name-keyed block set. The plan owns its
/// program, so the caller hands over only the per-call data.
///
/// # Errors
/// Propagates name-resolution and Metal driver failures.
pub fn execute_plan_named(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'_>)],
) -> Result<Evaluated, MetalError> {
    let blocks = resolve_named_blocks(&plan.program, named)?;
    execute_plan(plan, &blocks)
}

/// [`execute_plan_with_placements`] against a name-keyed block set -- the
/// name-resolving sibling of [`execute_plan_named`], mirroring how that
/// function wraps [`execute_plan`]. `blocks` (positional, for the per-node
/// upload path) is resolved once via [`resolve_named_blocks`], then handed
/// to `execute_plan_with_placements` unchanged; input-placed nodes skip that
/// resolved entry's upload just as they do in the positional call.
///
/// # Errors
/// Propagates name-resolution and Metal driver failures.
#[cfg(feature = "metal-output-placement")]
pub fn execute_plan_named_with_placements(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'_>)],
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
    output_placements: &[(NodeId, &PlacedBuffer, usize)],
) -> Result<Evaluated, MetalError> {
    let blocks = resolve_named_blocks_with_placed_inputs(plan, named, input_placements)?;
    // no scratch pool for this name-keyed convenience wrapper -- a caller
    // wanting the recycle path calls `execute_plan_with_placements` directly.
    execute_plan_with_placements(
        plan,
        &blocks,
        input_placements,
        output_placements,
        &mut Vec::new(),
    )
}

/// Executes a named plan with both caller-owned buffers and per-step expert
/// substitutions. Routed recurrent models need both capabilities in the same
/// command buffer: placement keeps recurrent state on the device while the
/// expert table selects the codec and address for the current route.
///
/// # Errors
/// Propagates name resolution, expert-source validation, and Metal failures.
#[cfg(feature = "metal-output-placement")]
pub fn execute_plan_named_with_placements_and_expert_sources(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'_>)],
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
    output_placements: &[(NodeId, &PlacedBuffer, usize)],
    expert_sources: &BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'_>>,
) -> Result<Evaluated, MetalError> {
    let blocks = resolve_named_blocks_with_placed_inputs(plan, named, input_placements)?;
    #[cfg(feature = "instrument")]
    let expert_stage_started = std::time::Instant::now();
    let expert_buffers = stage_expert_sources(plan, &blocks, expert_sources)?;
    #[cfg(feature = "instrument")]
    if std::env::var_os("PROXIMA_DEBUG_METAL_STAGES").is_some() {
        eprintln!(
            "expert_source_stage_ms={} source_nodes={} staged_nodes={}",
            expert_stage_started.elapsed().as_secs_f64() * 1e3,
            expert_sources.len(),
            expert_buffers.len(),
        );
    }
    execute_plan_with_placements_inner(
        plan,
        &blocks,
        input_placements,
        output_placements,
        &mut Vec::new(),
        &expert_buffers,
    )
}

pub(super) fn resolve_named_blocks_with_placed_inputs<'blocks>(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'blocks>)],
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
) -> Result<Vec<QuantizedBlock<'blocks>>, MetalError> {
    resolve_named_blocks_with_placed_nodes(&plan.program, named, |node| {
        input_placements
            .iter()
            .any(|(placed, _, _)| *placed == node)
    })
}

pub(super) fn resolve_named_blocks_with_placed_nodes<'blocks, IsPlaced>(
    program: &[Op],
    named: &[(&str, QuantizedBlock<'blocks>)],
    is_placed: IsPlaced,
) -> Result<Vec<QuantizedBlock<'blocks>>, MetalError>
where
    IsPlaced: Fn(NodeId) -> bool,
{
    let block_nodes = block_node_ids(program);
    let mut blocks = Vec::with_capacity(block_nodes.len());
    for node in &block_nodes {
        let name = program[node.0 as usize]
            .name()
            .ok_or(TensorError::UnnamedInput(*node))?;
        if let Some(block) = named.iter().find(|(candidate, _)| *candidate == name) {
            blocks.push(block.1);
        } else if is_placed(*node) {
            blocks.push(QuantizedBlock::Float32(&[]));
        } else {
            return Err(TensorError::UnboundInputName(String::from(name)).into());
        }
    }
    Ok(blocks)
}

/// [`execute_plan`] with the WHOLE program's own single command buffer's
/// `GPUStartTime`/`GPUEndTime` read back once, instead of
/// [`execute_plan_op_timed`]'s one-command-buffer-per-op attribution. ROW
/// 375 measured a batch of independent same-kind dispatches with the
/// per-op instrument and got a flat ~705 us reading dominated by that
/// function's own per-buffer submit/wait floor, 13-20x the in-program
/// per-dispatch cost `rmsnorm_fused_epilogue_cost.rs`'s ROW 368/372 batched
/// harness measures through this same one-encoder, one-command-buffer
/// shape `execute_plan` already uses for production. This function exists
/// so a caller batching N independent dispatches into one plan (this
/// crate's own `plan_named`/`execute_plan_named`) can read that batch's
/// real GPU occupancy without re-deriving `execute_plan`'s encode loop.
///
/// # Errors
/// Same as [`execute_plan`].
#[cfg(feature = "instrument")]
pub fn execute_plan_timed(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
) -> Result<(Evaluated, u64), MetalError> {
    let prepared = &plan.prepared;
    let packed_operands = &plan.packed_operands;

    let (device, queue) = device_and_queue()?;

    let mut device_buffers: BTreeMap<NodeId, DeviceBuffer> = BTreeMap::new();
    for (index, ((node, block), dtype)) in prepared
        .block_nodes
        .iter()
        .zip(blocks.iter())
        .zip(plan.block_dtypes.iter())
        .enumerate()
    {
        if !prepared.live_block_inputs[index] {
            continue;
        }
        let resident_name = resident_name(plan, *node);
        let buffer = match block {
            QuantizedBlock::Float32(data) => upload_block(
                &device,
                data,
                *node,
                *dtype,
                plan.program[node.0 as usize].name(),
                resident_name,
            )?,
            QuantizedBlock::Int32(data) => {
                upload_block_int32_as_float(&device, data, resident_name)?
            }
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
        device_buffers.insert(*node, buffer);
    }

    let command_buffer = queue
        .commandBuffer()
        .ok_or_else(|| MetalError::CompileFailed {
            log: "command queue refused to hand out a command buffer".to_string(),
        })?;
    let encoder = EncoderGuard::new(command_buffer.computeCommandEncoder().ok_or_else(|| {
        MetalError::CompileFailed {
            log: "command buffer refused to hand out a compute encoder".to_string(),
        }
    })?);

    let mut pending_faults: Vec<PendingFault<'_>> = Vec::new();
    for (position, bound) in prepared.resolved.iter().enumerate() {
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
            None,
        )?;
        if let Some((fault_buffer, gathers)) = fault {
            pending_faults.push((bound, fault_buffer, gathers));
        }
        for retired in &prepared.retires[position] {
            device_buffers.remove(retired);
        }
    }
    encoder.finish();

    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    let gpu_ns =
        ((command_buffer.GPUEndTime() - command_buffer.GPUStartTime()) * 1e9).max(0.0) as u64;

    for (bound, fault_buffer, gathers) in &pending_faults {
        check_gather_fault(bound, fault_buffer, *gathers)?;
    }

    let evaluated = finish(plan, &device_buffers, &BTreeSet::new(), None)?;
    Ok((evaluated, gpu_ns))
}

/// [`execute_plan_timed`] against a name-keyed block set, mirroring
/// [`execute_plan_named`]'s own name resolution.
///
/// # Errors
/// Propagates name-resolution and Metal driver failures.
#[cfg(feature = "instrument")]
pub fn execute_plan_named_timed(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'_>)],
) -> Result<(Evaluated, u64), MetalError> {
    let blocks = resolve_named_blocks(&plan.program, named)?;
    execute_plan_timed(plan, &blocks)
}

/// One [`BoundOp`]'s GPU-only execution time and the operand bytes it read,
/// as gathered by [`execute_plan_op_timed`]. `weight_name` is
/// [`Op::name`](proxima_tensor::Op::name) off whichever operand is a named
/// block input (a model weight or the `ids`/`eps`/`rope_*`/KV-cache block
/// this program declares by name) -- `None` when every operand is itself a
/// computed node, which is the common case for a chained elementwise op.
#[cfg(feature = "instrument")]
#[derive(Debug, Clone)]
pub struct OpGpuTiming {
    pub node: NodeId,
    pub kind: &'static str,
    /// Full iteration extents of the bound operation that was timed.
    pub extents: Vec<u64>,
    /// Surviving axes for a reduction; empty for non-reductions.
    pub output_axes: Vec<u16>,
    /// This operand's TENSOR bytes -- `element_count(shape) * bytes_per_element`,
    /// where `bytes_per_element` is the operand's own dtype width for a plain
    /// buffer, or its `PackedCodec::block_bytes`/block-elements ratio for a
    /// packed one. See `operand_tensor_bytes`. Distinct from
    /// [`Self::bound_buffer_bytes`]: since a checkpoint mapping upload binds
    /// ONE buffer spanning the whole mmap (`checkpoint_mapping_offset`), that
    /// buffer's own `length()` overstates every individual operand sharing it
    /// -- this field is what a per-tensor byte-share table needs instead.
    pub operand_bytes: u64,
    /// The device buffer's own `length()` this operand was bound against --
    /// the value `operand_bytes` used to report before it was corrected to
    /// the tensor's own byte count. Kept so a checkpoint-mapping-offset bind
    /// (one shared buffer, `bound_buffer_bytes` far larger than
    /// `operand_bytes`) stays observable rather than silently disappearing.
    pub bound_buffer_bytes: u64,
    pub gpu_ns: u64,
    pub weight_name: Option<String>,
    /// `bound.operands().len()` -- surfaced so a diagnostic caller can tell
    /// a two-operand (weight, activation) reduce, the shape
    /// `crate::msl::packed_row_block` requires, from a fused reduce whose
    /// element body absorbed a third operand (e.g. a gate/up product ahead
    /// of a down-projection), which disqualifies the row-blocked kernel via
    /// that same function's `quantized.len() != 2` check.
    pub operand_count: usize,
    /// The packed codec carried by this op's named operand, when present.
    pub packed_codec: Option<PackedCodec>,
    /// The emitted packed-kernel body, kept separate from the broad op kind.
    pub packed_kernel_variant: &'static str,
    /// [`crate::msl::diagnose_packed_row_block`]'s own verdict on THIS op,
    /// against a REAL bound program rather than a synthetic symbolic one --
    /// `None` when the op is not a `Reduce { keep: Keep::Reduce, .. }` at
    /// all (the diagnosis does not apply), `Some("PASS")` when it took the
    /// row-blocked path, `Some(<rejection debug>)` otherwise. Exists
    /// because a synthetic probe's rejection table and a real production
    /// run's own `classify_kind` bucket disagreed on `ffn_down`/
    /// `output.weight` -- printing this against the REAL bound op is what
    /// settles which one was wrong, rather than trusting either by
    /// inference.
    pub packed_row_block_rejection: Option<String>,
}

/// `PROXIMA_METAL_NAN_CHECK`-gated NaN probe for exactly ONE op's own output
/// buffer, called from [`execute_op_timed`] right after `waitUntilCompleted`
/// and before that op's operands are retired -- the diagnostic this crate's
/// own `PROXIMA_METAL_OP_PROFILE_STEP` path lacked to bisect the 30B
/// qwen3moe Metal decode's all-zero logits down to the exact op that first
/// produced NaN data, rather than the whole step.
///
/// Checks `is_nan()`, deliberately NOT `!is_finite()` (which would also
/// catch `-inf`): `causal_mask` (`proxima-tensor/src/spec.rs`) bakes a
/// literal `f32::NEG_INFINITY` into the program as a `constant` op, fed
/// through a `Select` into every masked attention score BEFORE softmax --
/// the SAME literal a correct CPU decode also evaluates and zeroes out, so
/// a `-inf` reaching a masked score, or even an unmasked reduce that legally
/// saturates, is not evidence of anything. This codebase never constructs an
/// intentional `f32::NAN` scalar (grepped: zero production call sites), so a
/// NaN appearing on ANY node's output has no legitimate source and is
/// unambiguous evidence of the defect this diagnostic exists to find.
///
/// Reads the buffer back through [`read_back`] (the same narrow-to-f32 path
/// [`finish`] uses for every other output), which is why this is reachable
/// only behind `instrument`: it pays a device round trip per op, same cost
/// class [`execute_op_timed`]'s own doc already accepts for GPU-time
/// attribution. Returns `true` the first time this op's own output holds a
/// NaN, letting the caller stop the step at the FIRST offending op instead
/// of running every remaining op past a value already known bad.
// a type alias, not a new type: names the tuple `check_op_output_finite`
// returns so clippy's `type_complexity` lint reads it once instead of
// flagging the signature -- first NaN/Inf index, the op's shape, its
// readback values.
#[cfg(feature = "instrument")]
pub(super) type FiniteCheckFailure = (usize, Vec<u64>, alloc::vec::Vec<f32>);

#[cfg(feature = "instrument")]
pub(super) fn check_op_output_finite(
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    prepared: &Prepared,
    program: &[Op],
    node: NodeId,
    kind: &str,
) -> Result<Option<FiniteCheckFailure>, MetalError> {
    let Some((buffer, offset)) = device_buffers.get(&node) else {
        return Ok(None);
    };
    let shape = prepared.shapes.of(node).to_vec();
    let dtype = gpu_dtype(program, &prepared.index_nodes, node);
    let data = read_back(buffer, *offset, element_count(&shape), node, dtype)?;
    let nan_count = data.iter().filter(|value| value.is_nan()).count();
    if nan_count == 0 {
        return Ok(None);
    }
    let Some(first_index) = data.iter().position(|value| value.is_nan()) else {
        return Ok(None);
    };
    let first_values: alloc::vec::Vec<f32> = data.iter().copied().take(8).collect();
    let kind_owned = kind.to_string();
    debug!(
        node = node.0,
        kind = %kind_owned,
        shape = ?shape,
        first_values = ?first_values,
        nan_count = nan_count as u64,
        total_count = data.len() as u64,
        "nan_check: first nan op output found"
    );
    Ok(Some((first_index, shape, data)))
}

/// `PROXIMA_METAL_COMPARE_CPU`-gated per-node parity probe, called from
/// [`execute_op_timed`] alongside [`check_op_output_finite`] -- reads this
/// op's own Metal output back through [`read_back`] and diffs it against
/// `cpu_reference`'s entry for the SAME [`NodeId`], computed once up front by
/// evaluating the identical program/symbols/blocks on the CPU route for
/// every non-`Op::Input` node (`proxima-model-interop/src/generate.rs`'s
/// `evaluate_op_timed`, the one call site that builds `cpu_reference` and
/// has `program`/`symbols`/`named` all in scope to do it).
///
/// Relative diff per element uses a `1e-6` floor on the denominator so a
/// near-zero CPU reference value cannot inflate a genuinely tiny absolute
/// gap into a huge ratio. Returns the max relative diff across every element
/// (`None` when this node has no CPU reference entry -- weights and other
/// `Op::Input` nodes are never in `cpu_reference` by construction). The
/// caller stops the step at the first node whose max relative diff exceeds
/// `1e-2`, printing every field this diagnostic's own report needs: node,
/// kind, shape, and the first 8 values on both sides.
#[cfg(feature = "instrument")]
pub(super) fn compare_op_output_to_cpu(
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    prepared: &Prepared,
    program: &[Op],
    node: NodeId,
    kind: &str,
    cpu_reference: &BTreeMap<NodeId, alloc::vec::Vec<f32>>,
) -> Result<Option<f32>, MetalError> {
    let Some(cpu_values) = cpu_reference.get(&node) else {
        return Ok(None);
    };
    let Some((buffer, offset)) = device_buffers.get(&node) else {
        return Ok(None);
    };
    let shape = prepared.shapes.of(node).to_vec();
    let dtype = gpu_dtype(program, &prepared.index_nodes, node);
    let metal_values = read_back(buffer, *offset, element_count(&shape), node, dtype)?;
    let max_rel_diff = metal_values
        .iter()
        .zip(cpu_values.iter())
        .map(|(metal_value, cpu_value)| (metal_value - cpu_value).abs() / cpu_value.abs().max(1e-6))
        .fold(0.0_f32, f32::max);
    if max_rel_diff > 1e-2 {
        let metal_first: alloc::vec::Vec<f32> = metal_values.iter().copied().take(8).collect();
        let cpu_first: alloc::vec::Vec<f32> = cpu_values.iter().copied().take(8).collect();
        let kind_owned = kind.to_string();
        debug!(
            node = node.0,
            kind = %kind_owned,
            shape = ?shape,
            metal_first = ?metal_first,
            cpu_first = ?cpu_first,
            max_rel_diff = max_rel_diff as f64,
            "cpu_compare: divergent op output found"
        );
    }
    Ok(Some(max_rel_diff))
}

#[cfg(feature = "instrument")]
pub(super) const CPU_BOUND_COMPARE_CAP_BYTES: usize = 64 * 1024 * 1024;

#[cfg(feature = "instrument")]
pub(super) fn validate_selected_expert_routes(
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    prepared: &Prepared,
    program: &[Op],
    bound: &BoundOp,
    expert_buffers: Option<&ExpertSourceBuffers>,
) -> Result<(), MetalError> {
    let Some(selected) = std::env::var("PROXIMA_METAL_COMPARE_BOUND_NODE")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .map(NodeId)
    else {
        return Ok(());
    };
    if selected != bound.node {
        return Ok(());
    }
    let Some(expert_buffers) = expert_buffers else {
        eprintln!("metal_expert_routes_missing_buffers node={:?}", bound.node);
        return Ok(());
    };
    eprintln!(
        "metal_expert_routes_begin node={:?} source_node={:?} extents={:?} operands={:?}",
        bound.node,
        expert_buffers.node,
        bound.extents,
        bound
            .operands()
            .iter()
            .map(|(source, layout, lookup)| (
                *source,
                layout.clone(),
                lookup.as_ref().map(|value| (
                    value.indices,
                    value.index_layout.clone(),
                    value.element_stride,
                    value.extent
                ))
            ))
            .collect::<Vec<_>>()
    );
    for (source, _, lookup) in bound.all_read_sources() {
        if *source != expert_buffers.node {
            continue;
        }
        let Some(lookup) = lookup else {
            continue;
        };
        let Some((lookup_buffer, lookup_offset)) = device_buffers.get(&lookup.indices) else {
            return Err(MetalError::UnresolvedHazardOperand {
                node: lookup.indices,
            });
        };
        let lookup_values = read_back(
            lookup_buffer,
            *lookup_offset,
            element_count(prepared.shapes.of(lookup.indices)),
            lookup.indices,
            gpu_dtype(program, &prepared.index_nodes, lookup.indices),
        )?;
        eprintln!(
            "metal_expert_lookup node={:?} values={:?}",
            lookup.indices, lookup_values
        );
        let mut seen = Vec::new();
        for value in lookup_values {
            let expert = value as u32;
            if value < 0.0 || value.fract() != 0.0 {
                return Err(MetalError::ExpertSourceUnsupported {
                    node: bound.node,
                    reason: "route index was not a non-negative integer",
                });
            }
            let Some(descriptor) = expert_buffers.descriptor_records.get(expert as usize) else {
                return Err(MetalError::ExpertSourceUnsupported {
                    node: bound.node,
                    reason: "route index exceeded the descriptor table",
                });
            };
            if descriptor.expert_index != expert
                || descriptor.byte_length == 0
                || descriptor.out_dim == 0
                || descriptor.in_dim == 0
            {
                eprintln!(
                    "metal_expert_route_invalid node={:?} expert={} descriptor={descriptor:?}",
                    bound.node, expert
                );
                return Err(MetalError::ExpertSourceUnsupported {
                    node: bound.node,
                    reason: "route selected an invalid expert descriptor",
                });
            }
            let descriptor_summary = (
                expert,
                descriptor.codec,
                descriptor.byte_offset,
                descriptor.byte_length,
            );
            if !seen.contains(&descriptor_summary) {
                seen.push(descriptor_summary);
            }
        }
        eprintln!(
            "metal_expert_routes node={:?} lookup={:?} routes={seen:?}",
            bound.node, lookup.indices
        );
    }
    Ok(())
}

/// Captures only the already-live dense operands of the selected bound op.
/// The snapshot happens before that op is encoded, so the CPU interpreter
/// receives the same payload the Metal kernel is about to read without
/// retaining or recursively evaluating any other graph node.
#[cfg(feature = "instrument")]
pub(super) fn evaluate_selected_bound_cpu(
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    prepared: &Prepared,
    packed_operands: &PackedOperands,
    program: &[Op],
    bound: &BoundOp,
) -> Result<Option<alloc::vec::Vec<f32>>, MetalError> {
    let selected = std::env::var("PROXIMA_METAL_COMPARE_BOUND_NODE")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .map(NodeId);
    if selected != Some(bound.node) {
        return Ok(None);
    }

    for (source, _, lookup) in bound.all_read_sources() {
        if lookup.is_some() {
            // Gathered operands require the full source-aware CPU evaluator;
            // this local snapshot intentionally cannot interpret descriptor
            // indexed packed bytes.
            return Ok(None);
        }
        if packed_operands.contains_key(source) {
            // A packed operand likewise has no valid dense snapshot here.
            return Ok(None);
        }
    }

    let sources: BTreeSet<NodeId> = bound
        .all_read_sources()
        .map(|(source, _, _)| *source)
        .collect();
    let output_elements = element_count(prepared.shapes.of(bound.node));
    let snapshot_elements = sources.iter().try_fold(output_elements, |total, source| {
        total.checked_add(element_count(prepared.shapes.of(*source)))
    });
    let snapshot_bytes = snapshot_elements
        .and_then(|elements| elements.checked_mul(core::mem::size_of::<f32>()))
        .unwrap_or(usize::MAX);
    if snapshot_bytes > CPU_BOUND_COMPARE_CAP_BYTES {
        return Err(MetalError::CpuBoundComparisonOverCap {
            node: bound.node,
            bytes: snapshot_bytes,
            cap_bytes: CPU_BOUND_COMPARE_CAP_BYTES,
        });
    }

    let mut owned_buffers: alloc::vec::Vec<Option<alloc::vec::Vec<f32>>> =
        alloc::vec![None; program.len()];
    for source in sources {
        let (buffer, offset) = device_buffers
            .get(&source)
            .ok_or(MetalError::UnresolvedHazardOperand { node: source })?;
        let shape = prepared.shapes.of(source);
        let dtype = gpu_dtype(program, &prepared.index_nodes, source);
        owned_buffers[source.0 as usize] = Some(read_back(
            buffer,
            *offset,
            element_count(shape),
            source,
            dtype,
        )?);
    }
    let borrowed_buffers: alloc::vec::Vec<Option<&[f32]>> = owned_buffers
        .iter()
        .map(|buffer| buffer.as_deref())
        .collect();
    let mut output = alloc::vec![0.0; output_elements];
    let packed_nodes: alloc::vec::Vec<NodeId> = packed_operands.keys().copied().collect();
    proxima_tensor::evaluate_bound_f32_into(bound, &borrowed_buffers, &packed_nodes, &mut output)?;
    Ok(Some(output))
}

// same reasoning as `FiniteCheckFailure` above: first divergent index,
// metal value, cpu value, absolute diff, relative diff.
#[cfg(feature = "instrument")]
pub(super) type BoundMismatch = (usize, f32, f32, f32, f32);

#[cfg(feature = "instrument")]
pub(super) fn compare_bound_f32(
    node: NodeId,
    metal_values: &[f32],
    cpu_values: &[f32],
) -> Result<Option<BoundMismatch>, MetalError> {
    if metal_values.len() != cpu_values.len() {
        return Err(MetalError::CpuBoundComparisonLengthMismatch {
            node,
            metal_len: metal_values.len(),
            cpu_len: cpu_values.len(),
        });
    }
    let mut max_relative = 0.0_f32;
    let mut first_mismatch = None;
    for (element, (metal_value, cpu_value)) in metal_values.iter().zip(cpu_values).enumerate() {
        let finite = metal_value.is_finite() && cpu_value.is_finite();
        let absolute = (metal_value - cpu_value).abs();
        let relative = if finite {
            absolute / cpu_value.abs().max(1e-6)
        } else {
            f32::INFINITY
        };
        max_relative = max_relative.max(relative);
        if (!finite || relative > 1e-2) && first_mismatch.is_none() {
            first_mismatch = Some((element, *metal_value, *cpu_value, relative));
        }
    }
    Ok(first_mismatch.map(|(element, metal, cpu, relative)| {
        (element, metal, cpu, relative, max_relative.max(relative))
    }))
}

#[cfg(feature = "instrument")]
pub(super) fn report_bound_operands(
    bound: &BoundOp,
    prepared: &Prepared,
    packed_operands: &PackedOperands,
    program: &[Op],
    expert_buffers: Option<&ExpertSourceBuffers>,
    device_buffers: Option<&BTreeMap<NodeId, DeviceBuffer>>,
    sample_element: Option<usize>,
) {
    let sample_coordinate = sample_element.map(|mut element| {
        let mut coordinate = alloc::vec![0_u64; prepared.shapes.of(bound.node).len()];
        for axis in (0..coordinate.len()).rev() {
            let extent = prepared.shapes.of(bound.node)[axis] as usize;
            coordinate[axis] = (element % extent) as u64;
            element /= extent;
        }
        coordinate
    });
    for (operand_index, (source, layout, lookup)) in bound.all_read_sources().enumerate() {
        eprintln!(
            "metal_bound_operand node={:?} operand={} source={source:?} source_name={:?} source_shape={:?} layout={layout:?} lookup={lookup:?} packed_codec={:?} expert_source={}",
            bound.node,
            operand_index,
            program[source.0 as usize].name(),
            prepared.shapes.of(*source),
            packed_operands.get(source),
            expert_buffers.is_some_and(|buffers| buffers.node == *source),
        );
        if let (Some(device_buffers), Some(coordinate)) =
            (device_buffers, sample_coordinate.as_deref())
            && expert_buffers.is_none()
            && let Some((buffer, offset)) = device_buffers.get(source)
            && let Ok(values) = read_back(
                buffer,
                *offset,
                element_count(prepared.shapes.of(*source)),
                *source,
                gpu_dtype(program, &prepared.index_nodes, *source),
            )
        {
            let source_offset = layout.offset_of(coordinate);
            let source_value = usize::try_from(source_offset)
                .ok()
                .and_then(|index| values.get(index).copied());
            eprintln!(
                "metal_bound_operand_sample node={:?} operand={} coordinate={coordinate:?} source_offset={source_offset} value={source_value:?}",
                bound.node, operand_index
            );
        }
    }
    if let Some(expert_buffers) = expert_buffers
        && let Some((weight_source, weight_layout, Some(lookup))) = bound.all_read_sources().next()
        && let Some((lookup_buffer, lookup_offset)) =
            device_buffers.and_then(|buffers| buffers.get(&lookup.indices))
        && let Ok(route_values) = read_back(
            lookup_buffer,
            *lookup_offset,
            element_count(prepared.shapes.of(lookup.indices)),
            lookup.indices,
            gpu_dtype(program, &prepared.index_nodes, lookup.indices),
        )
        && let Some(route) = route_values.first().copied().map(|value| value as usize)
        && let Some(descriptor) = expert_buffers.descriptor_records.get(route)
        && descriptor.codec == PackedCodec::Q2K
        && let Some((_, _, None)) = bound.all_read_sources().nth(1)
        && let Some((activation_source, _, _)) = bound.all_read_sources().nth(1)
        && let Some((activation_buffer, activation_offset)) =
            device_buffers.and_then(|buffers| buffers.get(activation_source))
        && let Ok(activation_values) = read_back(
            activation_buffer,
            *activation_offset,
            element_count(prepared.shapes.of(*activation_source)),
            *activation_source,
            gpu_dtype(program, &prepared.index_nodes, *activation_source),
        )
    {
        let payload_pointer = expert_buffers.payloads.contents().as_ptr().cast::<u8>();
        let payload_start = expert_buffers
            .payload_offset
            .saturating_add(descriptor.byte_offset);
        let payload_end = payload_start.saturating_add(descriptor.byte_length);
        if payload_end <= expert_buffers.payloads.length() {
            // `Q2_K` is the only dynamic low codec in this diagnostic arm;
            // decode the same 84-byte block geometry the emitted helper uses
            // and dot it with the already-read activation to separate a
            // payload/index fault from an upstream activation difference.
            let payload = unsafe {
                core::slice::from_raw_parts(
                    payload_pointer.add(payload_start),
                    descriptor.byte_length,
                )
            };
            let mut dot = 0.0_f32;
            for (index, activation) in activation_values.iter().enumerate() {
                if index >= usize::try_from(descriptor.in_dim).unwrap_or(0) {
                    break;
                }
                let block_start = (index / 256) * 84;
                let block = &payload[block_start..block_start + 84];
                let local = index % 256;
                let d = f16::from_le_bytes([block[80], block[81]]).to_f32();
                let dmin = f16::from_le_bytes([block[82], block[83]]).to_f32();
                let chunk = local / 128;
                let within = local % 128;
                let group = within / 32;
                let sub_block = chunk * 8 + group * 2 + usize::from(within % 32 >= 16);
                let scale_min = block[sub_block];
                let level = (block[16 + chunk * 32 + (within % 32)] >> (2 * group)) & 3;
                let weight = d * f32::from(scale_min & 0x0f) * f32::from(level)
                    - dmin * f32::from(scale_min >> 4);
                dot += weight * *activation;
            }
            eprintln!(
                "metal_expert_debug node={:?} route={} codec={:?} payload_offset={} activation_source={activation_source:?} weight_layout={weight_layout:?} weight0={} activation0={} dot={dot}",
                bound.node,
                route,
                descriptor.codec,
                payload_start,
                q2k_debug_value(payload, 0),
                activation_values.first().copied().unwrap_or_default(),
            );
        }
        let _ = weight_source;
    }
}

#[cfg(feature = "instrument")]
pub(super) fn q2k_debug_value(blocks: &[u8], index: usize) -> f32 {
    let block_start = (index / 256) * 84;
    let block = &blocks[block_start..block_start + 84];
    let local = index % 256;
    let d = f16::from_le_bytes([block[80], block[81]]).to_f32();
    let dmin = f16::from_le_bytes([block[82], block[83]]).to_f32();
    let chunk = local / 128;
    let within = local % 128;
    let group = within / 32;
    let sub_block = chunk * 8 + group * 2 + usize::from(within % 32 >= 16);
    let scale_min = block[sub_block];
    let level = (block[16 + chunk * 32 + (within % 32)] >> (2 * group)) & 3;
    d * f32::from(scale_min & 0x0f) * f32::from(level) - dmin * f32::from(scale_min >> 4)
}

