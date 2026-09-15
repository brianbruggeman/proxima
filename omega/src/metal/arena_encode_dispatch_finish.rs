use super::*;

/// CARD 6.5: whole-`MetalBuffer` device output arena, hung off the cached
/// [`Plan`] and built exactly once, in [`plan`], from
/// [`proxima_tensor::node_retirement`]'s own liveness ranges over
/// `prepared.resolved` -- never lazily, never per `execute_plan_with_placements`
/// call. `slots` holds one physical buffer per size class actually needed;
/// `position_slot` says which slot backs each plan POSITION's output.
///
/// # Whole-buffer sharing only (crit RS-3)
///
/// A slot is reused across two positions only when [`build_buffer_arena`]'s
/// single retirement-ordered pass has seen the first position's node retired
/// before assigning the slot to a later position needing the SAME byte
/// length -- never a sub-range of a larger slot. Sub-allocating would make
/// `encode_op`'s `device_buffers.insert(bound.node, (output, 0))` a lie, and
/// [`finish`]'s readback invariant ("an output node's buffer is always
/// freshly allocated ... at offset 0") would then read a co-resident node's
/// bytes instead of its own.
///
/// # Outputs are pinned by construction
///
/// `node_retirement` already excludes every `effective_outputs` node from
/// every position's retire list, so an output's slot is never handed back to
/// `free_by_size` by this struct's own build loop -- no separate output
/// check is needed here, only the `debug_assert!` in [`build_buffer_arena`]
/// that keeps that upstream invariant honest if it ever changes.
#[cfg(feature = "metal-plan-stable-buffers")]
pub(super) struct BufferArena {
    pub(super) slots: Vec<MetalBuffer>,
    pub(super) slot_bytes: Vec<usize>,
    /// Parallel to `prepared.resolved`.
    pub(super) position_slot: Vec<usize>,
    /// Live-bytes high-water mark reached while building -- MG-3's own
    /// witness against [`ARENA_TRANSIENT_CAP`].
    pub(super) peak_bytes: usize,
    /// Per-slot count of positions ever assigned that slot -- `instrument`-
    /// only, ROW 539's witness for whether a WAW/WAR barrier's colliding
    /// identity is a genuinely recycled slot (`> 1`) rather than a slot this
    /// plan only ever assigned once. Parallel to `slots`/`slot_bytes`.
    #[cfg(feature = "instrument")]
    pub(super) slot_occupancy: Vec<usize>,
}

#[cfg(feature = "metal-plan-stable-buffers")]
impl BufferArena {
    /// The `(buffer, offset)` pair [`encode_op`] binds a position's output
    /// to when the caller has not output-placed that position's node.
    /// Offset is always 0: see this struct's own "whole-buffer sharing
    /// only" doc.
    fn placement_for(&self, position: usize) -> (&MetalBuffer, usize) {
        (&self.slots[self.position_slot[position]], 0)
    }

    /// Physical slot count -- the direct witness of how much reuse
    /// [`build_buffer_arena`]'s free list actually achieved: `slots.len() <
    /// position_slot.len()` whenever two or more positions shared a slot.
    pub(super) fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// A slot's own allocated byte length -- test/diagnostic surface for
    /// asserting a growing extent forces a genuinely new slot rather than
    /// silently reusing an undersized one.
    pub(super) fn slot_byte_len(&self, slot: usize) -> usize {
        self.slot_bytes[slot]
    }

    /// True when `position`'s own slot was assigned to more than one
    /// position over this plan's whole program -- a recycled slot, ROW 539's
    /// arena-reuse witness for [`record_hazard_class`].
    #[cfg(feature = "instrument")]
    pub(super) fn slot_is_recycled(&self, position: usize) -> bool {
        self.slot_occupancy[self.position_slot[position]] > 1
    }
}

/// Builds [`BufferArena`] in one pass over `resolved`, in program order,
/// mirroring the retirement ordering [`execute_plan_with_placements`]'s own
/// dispatch loop already uses: assign THIS position's slot first (against
/// the free list as of every EARLIER position's retirements only), then
/// free whatever `retires[position]` names. `effective_outputs` is passed
/// through only for the `debug_assert!` below -- `node_retirement` itself is
/// what actually keeps an output out of `retires`.
///
/// This plan's own row count -- the largest `CachedAttention` `query_rows`
/// among `resolved`, or `1` when the plan has no attention op at all (every
/// non-attention op's own transient extents already scale with the same row
/// count, so the maximum over attention ops is the plan-wide M). Decode
/// dispatches one query row at a time (`query_rows == 1` everywhere), so this
/// returns `1` there and [`build_buffer_arena`]'s cap collapses to the
/// unscaled `ARENA_TRANSIENT_CAP` -- the constant's own decode-sized default
/// is preserved exactly. A prefill sharing ONE dispatch across `M` rows
/// (ROW 391's 1100-row interactive-chat prompt) reports `query_rows == M`,
/// so the cap scales up with it instead of being sized once for decode and
/// applied unchanged to every M.
#[cfg(feature = "metal-plan-stable-buffers")]
pub(super) fn plan_query_rows(resolved: &[BoundOp]) -> u64 {
    resolved
        .iter()
        .filter_map(|bound| match &bound.kind {
            BoundOpKind::CachedAttention { query_rows, .. } => Some(*query_rows),
            _ => None,
        })
        .max()
        .unwrap_or(1)
}

/// Prints the naive (no-reuse) transient sum against this plan's own
/// [`plan_query_rows`]-scaled cap before allocating anything, per this
/// card's memory gate.
#[cfg(feature = "metal-plan-stable-buffers")]
pub(super) fn build_buffer_arena(
    device: &ProtocolObject<dyn MTLDevice>,
    resolved: &[BoundOp],
    retires: &[Vec<NodeId>],
    effective_outputs: &[NodeId],
) -> Result<BufferArena, MetalError> {
    let outputs: BTreeSet<NodeId> = effective_outputs.iter().copied().collect();
    let naive_transient_bytes: usize = resolved
        .iter()
        .map(|bound| bound_output_len(bound).max(1) * bound.dtype.size_bytes())
        .sum();
    let uniform_bytes: usize = resolved.iter().map(pack_uniforms_byte_len).sum();
    let query_rows = plan_query_rows(resolved);
    let cap_bytes = ARENA_TRANSIENT_CAP.saturating_mul(query_rows.max(1) as usize);
    let device_limit = device.recommendedMaxWorkingSetSize();
    debug!(
        naive_transient_bytes = naive_transient_bytes as u64,
        uniform_bytes = uniform_bytes as u64,
        op_count = resolved.len() as u64,
        query_rows,
        arena_transient_cap = ARENA_TRANSIENT_CAP as u64,
        cap_bytes = cap_bytes as u64,
        device_limit,
        "buffer arena sized against the naive (no-reuse) transient sum"
    );

    let mut free_by_size: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut slots: Vec<MetalBuffer> = Vec::new();
    let mut slot_bytes: Vec<usize> = Vec::new();
    let mut position_slot: Vec<usize> = Vec::with_capacity(resolved.len());
    let mut node_slot: BTreeMap<NodeId, usize> = BTreeMap::new();
    let mut live_bytes: usize = 0;
    let mut peak_bytes: usize = 0;
    #[cfg(feature = "instrument")]
    let mut slot_occupancy: Vec<usize> = Vec::new();

    for (position, bound) in resolved.iter().enumerate() {
        let byte_length = bound_output_len(bound).max(1) * bound.dtype.size_bytes();
        let slot = match free_by_size.get_mut(&byte_length).and_then(Vec::pop) {
            Some(reused) => reused,
            None => {
                let index = slots.len();
                slots.push(allocate_buffer(
                    device,
                    bound_output_len(bound),
                    bound.dtype,
                )?);
                slot_bytes.push(byte_length);
                #[cfg(feature = "instrument")]
                slot_occupancy.push(0);
                index
            }
        };
        #[cfg(feature = "instrument")]
        {
            slot_occupancy[slot] += 1;
        }
        // every slot assignment re-occupies `byte_length` bytes, whether the
        // slot is freshly allocated or pulled back from the free list -- a
        // reused slot was subtracted out of `live_bytes` when its PREVIOUS
        // occupant retired, so skipping this on the reuse arm would double-
        // count that subtraction the next time this new occupant retires.
        live_bytes += byte_length;
        peak_bytes = peak_bytes.max(live_bytes);
        position_slot.push(slot);
        node_slot.insert(bound.node, slot);

        for retired in &retires[position] {
            debug_assert!(
                !outputs.contains(retired),
                "node_retirement must never retire an effective output"
            );
            if let Some(retired_slot) = node_slot.remove(retired) {
                live_bytes -= slot_bytes[retired_slot];
                free_by_size
                    .entry(slot_bytes[retired_slot])
                    .or_default()
                    .push(retired_slot);
            }
        }
    }

    let reuse_factor = naive_transient_bytes as f64 / peak_bytes.max(1) as f64;
    debug!(
        peak_bytes = peak_bytes as u64,
        reuse_factor, "buffer arena reached its steady-state peak"
    );
    if peak_bytes > cap_bytes {
        proxima_telemetry::error!(
            peak_bytes,
            cap_bytes,
            query_rows,
            device_limit,
            "arena peak_bytes exceeds arena_transient_cap -- MG-3 kill condition"
        );
        return Err(MetalError::ArenaOverCap {
            peak_bytes,
            cap_bytes,
            query_rows,
            device_limit,
        });
    }

    Ok(BufferArena {
        slots,
        slot_bytes,
        position_slot,
        peak_bytes,
        #[cfg(feature = "instrument")]
        slot_occupancy,
    })
}

/// CARD 6.5: one uniform buffer per plan position, allocated once in
/// [`plan`] and written IN PLACE by [`encode_op`] on every call thereafter --
/// never through the content-keyed `UNIFORM_BUFFERS` cache. That cache is a
/// dedup map shared by BYTES across every op with identical uniform bytes
/// (see `UNIFORM_BUFFERS`'s own doc); writing through it in place would
/// corrupt every other op sharing the same key. A plan-owned buffer per
/// POSITION has no such sharing hazard: each position's buffer is used by
/// exactly that position, forever, for this plan's lifetime.
#[cfg(feature = "metal-plan-stable-buffers")]
pub(super) struct PlanUniforms {
    /// Parallel to `prepared.resolved`.
    pub(super) buffers: Vec<MetalBuffer>,
}

#[cfg(feature = "metal-plan-stable-buffers")]
pub(super) fn build_plan_uniforms(
    device: &ProtocolObject<dyn MTLDevice>,
    resolved: &[BoundOp],
    numeric_policy: NumericPolicy,
) -> Result<PlanUniforms, MetalError> {
    let mut buffers = Vec::with_capacity(resolved.len());
    for bound in resolved {
        let bytes = pack_uniforms(bound, numeric_policy)?;
        let buffer = device
            .newBufferWithLength_options(bytes.len().max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| MetalError::CompileFailed {
                log: "device refused to allocate a plan uniform buffer".to_string(),
            })?;
        write_plan_uniform_bytes(&buffer, &bytes);
        #[cfg(feature = "instrument")]
        counter!(PLAN_UNIFORM_WRITES, 1);
        buffers.push(buffer);
    }
    Ok(PlanUniforms { buffers })
}

/// Initializes one plan-owned uniform buffer. The bytes are a pure function
/// of the resolved `BoundOp` and the plan's fixed numeric policy, so warm
/// execution binds this buffer unchanged instead of repacking and rewriting
/// it per dispatch.
#[cfg(feature = "metal-plan-stable-buffers")]
pub(super) fn write_plan_uniform_bytes(buffer: &ProtocolObject<dyn MTLBuffer>, bytes: &[u8]) {
    let pointer = buffer.contents();
    // SAFETY: `buffer` was allocated by `build_plan_uniforms` at exactly
    // `bytes.len().max(1)` bytes and is `storageModeShared`, so this is a
    // valid, CPU-visible, mutable byte range for the duration of this write;
    // `bytes.len()` never exceeds the buffer's own allocated length because
    // `pack_uniforms(bound)` is a pure function of `bound`'s own static
    // extents/rank/gather-count, unchanged across calls against the SAME
    // plan position.
    let destination =
        unsafe { core::slice::from_raw_parts_mut(pointer.as_ptr().cast::<u8>(), bytes.len()) };
    destination.copy_from_slice(bytes);
}

/// [`write_plan_uniform_bytes`]'s read-back counterpart -- test surface only,
/// proving a write actually landed rather than trusting the copy above.
#[cfg(all(test, feature = "metal-plan-stable-buffers"))]
pub(super) fn read_back_uniform_bytes(buffer: &ProtocolObject<dyn MTLBuffer>, byte_len: usize) -> Vec<u8> {
    let pointer = buffer.contents();
    // SAFETY: same CPU-visible, `storageModeShared` argument as
    // `write_plan_uniform_bytes` above, read rather than written.
    let source = unsafe { core::slice::from_raw_parts(pointer.as_ptr().cast::<u8>(), byte_len) };
    source.to_vec()
}

/// [`encode_op`]'s arena lookup, split out so the call site reads the same
/// three lines regardless of whether `metal-plan-stable-buffers` is
/// compiled in -- `Ok(None)` with the feature off, matching `encode_op`'s
/// pre-existing "no placement, fresh `allocate_buffer`" behavior exactly.
///
/// Builds `plan.arena` on its first call for this `Plan` rather than
/// requiring [`plan`] to have built it already -- only
/// `execute_plan_with_placements`/`execute_plan_with_placements_op_timed`
/// ever call this, so `execute_plan`/`execute_plan_op_timed` (which never
/// do) cost this device allocation zero times, not once per miss.
#[cfg(feature = "metal-plan-stable-buffers")]
pub(super) fn arena_placement(
    plan: &Plan,
    position: usize,
) -> Result<Option<(&MetalBuffer, usize)>, MetalError> {
    if plan.arena.get().is_none() {
        let (device, _queue) = device_and_queue()?;
        let arena = build_buffer_arena(
            &device,
            &plan.prepared.resolved,
            &plan.prepared.retires,
            &plan.prepared.effective_outputs,
        )?;
        // a fresh, still-empty `OnceCell` can only fail to accept this set
        // if another call already raced it in -- impossible here since
        // `plan` is `&Plan`, never shared across a concurrent write.
        let _ = plan.arena.set(arena);
    }
    Ok(plan.arena.get().map(|arena| arena.placement_for(position)))
}
#[cfg(not(feature = "metal-plan-stable-buffers"))]
pub(super) fn arena_placement(
    _plan: &Plan,
    _position: usize,
) -> Result<Option<(&MetalBuffer, usize)>, MetalError> {
    Ok(None)
}

/// [`encode_op`]'s plan-owned-uniform lookup -- see [`arena_placement`]'s
/// own doc for why this is a free function rather than an inline `#[cfg]`,
/// and for why it builds `plan.uniforms` lazily on the same schedule.
#[cfg(feature = "metal-plan-stable-buffers")]
pub(super) fn plan_uniform_buffer(plan: &Plan, position: usize) -> Result<Option<&MetalBuffer>, MetalError> {
    if plan.uniforms.get().is_none() {
        let (device, _queue) = device_and_queue()?;
        let uniforms = build_plan_uniforms(&device, &plan.prepared.resolved, plan.numeric_policy)?;
        let _ = plan.uniforms.set(uniforms);
    }
    Ok(plan
        .uniforms
        .get()
        .map(|uniforms| &uniforms.buffers[position]))
}
#[cfg(not(feature = "metal-plan-stable-buffers"))]
pub(super) fn plan_uniform_buffer(_plan: &Plan, _position: usize) -> Result<Option<&MetalBuffer>, MetalError> {
    Ok(None)
}

/// Element count (f32) [`Plan::attention_scratch`] reserves for one
/// `CachedAttention` position -- `query_rows * heads * splits *
/// (2 + head_dim)`, redesign §4c's own formula: `2` for the running
/// `(max, sum)` pair, `head_dim` for the un-normalized weighted-value
/// accumulator, one such record per split. `query_rows * heads`
/// is `resolved.extents.product() / head_dim` -- the SAME `total_elements`
/// [`crate::msl::grid_threads`]'s `CachedAttention` arm already derives.
/// `None` for every other op kind. Used both by [`Plan::attention_scratch`]'s
/// lazy builder and by [`encode_op`]'s own cold-path fallback allocation
/// (no `Plan` to own a buffer, so a fresh one is sized with this exact
/// formula every call -- the same "no placement, fresh `allocate_buffer`"
/// shape `output` itself already falls back to).
///
/// `splits` is the compiled MAXIMUM (`crate::sized::ATTENTION_SPLIT_MAX`)
/// only at `query_rows == 1` (decode: one query row per dispatch) -- there,
/// reserving the max up front is what lets a `Plan` grow its own live
/// context length token by token, all the way to `ATTENTION_SPLIT_MAX`
/// splits, without this buffer ever needing to resize mid-stream. At
/// `query_rows > 1` (prefill: every row in the batch sharing ONE dispatch)
/// that same per-row headroom multiplies `query_rows` times the FULL
/// compiled max, not the [`crate::msl::splits_for`] value this call's own
/// (fixed, already-known) `context_length` will ever actually dispatch --
/// production crash: a 900-row prefill at `head_dim=128`/`heads=32`
/// requested `900 * 32 * 32 * 130 * 4` bytes (~479 MB) PER LAYER against
/// that always-max headroom (~17 GB across a 36-layer forward), which no
/// device honors -- `allocate_buffer`'s `newBufferWithLength_options`
/// returned `None`, surfaced as `MetalError::CompileFailed` once
/// [`EncoderGuard`] stopped that `Err` from crashing the process outright.
/// Prefill's context length cannot grow after this call the way decode's
/// does, so there is nothing to protect against by over-reserving: sizing
/// against the real split count is exact, not merely smaller.
pub(super) fn cached_attention_scratch_len(bound: &BoundOp, numeric_policy: NumericPolicy) -> Option<u64> {
    let BoundOpKind::CachedAttention {
        head_dim,
        query_rows,
        cached_key_rows,
        new_key_rows,
        ..
    } = &bound.kind
    else {
        return None;
    };
    let total_elements = bound
        .extents
        .iter()
        .product::<u64>()
        .checked_div(*head_dim)?;
    let splits = if *query_rows > 1 {
        crate::msl::splits_for(cached_key_rows + new_key_rows, numeric_policy)
    } else {
        crate::sized::ATTENTION_SPLIT_MAX
    };
    Some(total_elements * splits * (2 + head_dim))
}

/// [`Plan::attention_scratch`]'s lazy builder, built alongside [`PlanUniforms`]
/// on the same schedule -- see [`plan_uniform_buffer`]'s own doc for why a
/// free function rather than an inline `#[cfg]`.
#[cfg(feature = "metal-plan-stable-buffers")]
pub(super) fn attention_scratch_buffer(
    plan: &Plan,
    position: usize,
) -> Result<Option<&MetalBuffer>, MetalError> {
    if plan.attention_scratch.get().is_none() {
        let (device, _queue) = device_and_queue()?;
        let mut buffers = Vec::with_capacity(plan.prepared.resolved.len());
        for bound in &plan.prepared.resolved {
            let buffer = match cached_attention_scratch_len(bound, plan.numeric_policy) {
                Some(elements) => {
                    Some(allocate_buffer(&device, elements as usize, DType::Float32)?)
                }
                None => None,
            };
            buffers.push(buffer);
        }
        let _ = plan.attention_scratch.set(buffers);
    }
    Ok(plan
        .attention_scratch
        .get()
        .and_then(|buffers| buffers[position].as_ref()))
}
#[cfg(not(feature = "metal-plan-stable-buffers"))]
pub(super) fn attention_scratch_buffer(
    _plan: &Plan,
    _position: usize,
) -> Result<Option<&MetalBuffer>, MetalError> {
    Ok(None)
}

/// Builds `plan.resolved_steps` on its first call, or when
/// [`Plan::set_math_mode`] moved the compiled mode since the last build --
/// every later call for the SAME mode is a no-op. `numeric_policy` cannot
/// move post-construction ([`Plan::numeric_policy`]'s own doc), so
/// `math_mode` -- the one axis that can still narrow after [`plan`] --  is
/// the only staleness key this needs. Called once per
/// [`execute_plan_with_placements`] invocation, before that function's own
/// per-position loop, so a plan-cache HIT never pays [`kernel_cache_key`] or
/// [`kernel_dispatch_shape`] again: the loop below indexes
/// `plan.resolved_steps` by position instead.
pub(super) fn resolve_steps(device: &ProtocolObject<dyn MTLDevice>, plan: &Plan) -> Result<(), MetalError> {
    let stale = plan
        .resolved_steps
        .borrow()
        .as_ref()
        .is_none_or(|resolved| resolved.math_mode != plan.math_mode);
    if !stale {
        return Ok(());
    }
    // a math-mode change recompiles every pipeline, so a merged group's own
    // compiled pipeline/base_table (keyed off the STALE step pipelines) must
    // be dropped too, or the encode loop below would dispatch a merged
    // kernel compiled under the old mode against buffers resolved for it.
    #[cfg(feature = "metal-horizontal-merge")]
    plan.merged.borrow_mut().take();
    let mut steps = Vec::with_capacity(plan.prepared.resolved.len());
    for bound in &plan.prepared.resolved {
        let mut cache_key = kernel_cache_key(bound, &plan.packed_operands, plan.numeric_policy)?;
        cache_key.push(plan.math_mode.cache_token());
        let (bindings, grid) =
            kernel_dispatch_shape(bound, &plan.packed_operands, plan.numeric_policy)?;
        let pipeline = pipeline_for(
            device,
            bound,
            &plan.packed_operands,
            &cache_key,
            plan.math_mode,
            plan.numeric_policy,
        )?;
        // Redesign §4c: a `CachedAttention` position under a policy that
        // admits `ContextSplitMerge` resolves a SECOND pipeline for the
        // merge dispatch, keyed on the split's own cache key plus `_merge`
        // so the two never collide in `PIPELINE_CACHE` even though they
        // share every other structural token.
        let merge = match crate::msl::emit_cached_attention_merge(bound, plan.numeric_policy)? {
            Some(merge_kernel) => {
                let merge_cache_key = format!("{cache_key}_merge");
                let merge_pipeline =
                    pipeline_for_kernel(device, &merge_kernel, &merge_cache_key, plan.math_mode)?;
                Some(ResolvedMergeStep {
                    pipeline: merge_pipeline,
                    bindings: merge_kernel.bindings,
                    grid: merge_kernel.grid,
                })
            }
            None => None,
        };
        steps.push(ResolvedStep {
            pipeline,
            bindings,
            grid,
            merge,
        });
    }
    #[cfg(feature = "metal-horizontal-merge")]
    let merge_candidates = {
        let identities: Vec<*const ProtocolObject<dyn MTLComputePipelineState>> = steps
            .iter()
            .map(|step| Retained::as_ptr(&step.pipeline))
            .collect();
        let writes: Vec<NodeId> = plan.prepared.resolved.iter().map(|bound| bound.node).collect();
        let reads: Vec<Vec<NodeId>> = plan
            .prepared
            .resolved
            .iter()
            .map(|bound| bound.operands().iter().map(|(node, ..)| *node).collect())
            .collect();
        let groups = group_mergeable_positions(&identities, &reads, &writes);
        debug!(
            plan_positions = steps.len(),
            merge_groups = groups.len(),
            merged_positions = groups.iter().map(Vec::len).sum::<usize>(),
            "resolve_steps computed horizontal-merge candidate groups"
        );
        groups
    };
    *plan.resolved_steps.borrow_mut() = Some(ResolvedSteps {
        math_mode: plan.math_mode,
        steps,
        #[cfg(feature = "metal-horizontal-merge")]
        merge_candidates,
    });
    Ok(())
}

/// Encodes one `BoundOp` as a compute dispatch into the CALLER's already-open
/// `encoder` — neither opened nor `endEncoding()`d here. [`execute_plan`]
/// opens exactly one `MTLComputeCommandEncoder` for the whole program and
/// ends it once, after every op has been encoded into it (see the module
/// doc's "Execution model"); [`execute_plan_op_timed`]'s diagnostic path
/// instead opens and ends one per call, one op at a time. Either way this
/// function only ever binds a pipeline, binds buffers, and dispatches —
/// exactly the same three calls regardless of how many other ops share the
/// encoder it was handed. Returns the op's fault buffer and gather count
/// when it gathers, so the caller can check it after its own wait instead
/// of here, where the buffer is not yet CPU-visible.
///
/// `placement` is `None` on every call site that predates
/// `metal-output-placement` (byte-identical to before: a fresh
/// `allocate_buffer` sized to this op's own iteration space). `Some((buffer,
/// offset))` skips that allocation and binds `bound`'s output straight into
/// the caller-owned `buffer` at `offset` instead -- that offset is then
/// carried forward in `device_buffers`' own [`DeviceBuffer`] entry for this
/// node, so a later op reading it back through
/// [`bind_buffers`]/[`buffer_for`] needs no separate offset map.
///
/// `plan_uniform` is `None` on every call site that predates
/// `metal-plan-stable-buffers` (byte-identical to before: `upload_uniforms`'s
/// allocate-or-content-cache-hit path). `Some(buffer)` skips that call
/// entirely and writes this call's fresh uniform bytes straight into the
/// plan-owned `buffer` in place instead -- see [`PlanUniforms`]'s own doc for
/// why that is sound only because the buffer is keyed by PLAN POSITION, never
/// by content.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_op(
    device: &ProtocolObject<dyn MTLDevice>,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device_buffers: &mut BTreeMap<NodeId, DeviceBuffer>,
    bound: &BoundOp,
    packed_operands: &PackedOperands,
    placement: Option<(&MetalBuffer, usize)>,
    // row 555: the fused `GatedDeltaNet` op's `state_out` is a SECOND output
    // node absorbed into this bound op's own identity (`bound.node` is the
    // primary "out" node, never `state_out` -- `gated_delta_net_candidates`'s
    // own doc), so the ordinary `placement` lookup above (keyed by
    // `bound.node`) can never resolve a caller's `state_out` placement. This
    // is that SAME lookup, keyed by `state_out` instead, resolved by every
    // placement-aware caller and `None` everywhere else -- see the
    // `BoundOpKind::GatedDeltaNet` arm below for why skipping it silently
    // discarded recurrent state every dispatch.
    state_out_placement: Option<(&MetalBuffer, usize)>,
    plan_uniform: Option<&MetalBuffer>,
    math_mode: MathMode,
    numeric_policy: NumericPolicy,
    resolved: Option<&ResolvedStep>,
    // Redesign §4c: `Some` when `crate::metal::attention_scratch_buffer`
    // already resolved a PLAN-OWNED scratch buffer for this position
    // (`execute_plan_with_placements`, the one call site with a `Plan` to
    // own it). `None` on every other call site (`execute_plan`, the
    // `*_op_timed` diagnostics) -- when `bindings` still needs one (a
    // `Binding::Scratch` slot), this function allocates a throwaway one
    // below, the same "no placement, fresh `allocate_buffer`" fallback
    // `output` itself already has.
    scratch: Option<(&MetalBuffer, usize)>,
    // `Some` only from `execute_plan_with_placements` under
    // `DispatchType::Concurrent` -- the plan's own per-call `HazardState`,
    // reborrowed. `None` everywhere else (`execute_plan`, `execute_op_timed`,
    // every `DispatchType::Serial` call), since program order alone already
    // orders a split's write before its merge's read there (see the
    // `dispatch(encoder, &pipeline, grid)` call site's own doc below).
    // `mut` so `GatedDeltaNet`'s own `state_out` write below can reborrow it
    // (`as_deref_mut`) AFTER the merge block's own reborrow, instead of the
    // merge block's tuple match moving this `Option` outright.
    mut hazard: Option<&mut HazardTracker<*const ProtocolObject<dyn MTLBuffer>>>,
    expert_buffers: Option<&ExpertSourceBuffers>,
) -> Result<Option<(MetalBuffer, usize)>, MetalError> {
    let expert_source_node = match expert_buffers {
        None => None,
        Some(expert_buffers) => bound
            .operands()
            .iter()
            .any(|(node, _, _lookup)| *node == expert_buffers.node)
            .then_some(expert_buffers.node)
            .ok_or(MetalError::ExpertSourceUnsupported {
                node: expert_buffers.node,
                reason: "expert buffers were supplied to an operation that does not gather this source",
            })
            .map(Some)?,
    };
    // `cpu::run_node_into`'s own `BoundOpKind::CachedAttention` arm
    // (`proxima-tensor/src/cpu.rs:5210-5215`) is the only place
    // `instrument::record_op_kind` was ever called -- the CPU evaluator's
    // per-node dispatch. `encode_op` is `omega::metal`'s own per-op dispatch
    // (once per bound op in a plan, on the actual backend production
    // decode runs), and it never touched that counter: a Metal build's
    // `path_totals().op_kind_cached_attention` read 0 on every step
    // regardless of whether the fused kernel ran. Recording it here, at the
    // one point every `CachedAttention` op passes through regardless of
    // cache-hit/miss or placement, makes the interop's per-step
    // `cached_attention_ops` field truthful on the backend it actually
    // reports for.
    #[cfg(feature = "instrument")]
    if matches!(bound.kind, BoundOpKind::CachedAttention { .. }) {
        record_op_kind(OpKind::CachedAttention);
    }
    // `gather_count(bound) > 0` on an `Elementwise`/`Reduce` op is exactly
    // `spec::gathered_expert_product`'s shape once bound (a `Computed`
    // gather map on one operand feeding the multiply-then-reduce chain
    // `append_moe_ffn` builds) -- `embedding_lookup`'s own gather is a bare
    // `Elementwise` with no following reduce, so this also fires there, but
    // no test asserts this counter on that path; it exists so a MoE
    // fixture's Metal run can prove the gather engaged instead of silently
    // taking a non-gathering fallback ("default-on ≠ reachable").
    #[cfg(feature = "instrument")]
    if matches!(
        bound.kind,
        BoundOpKind::Elementwise { .. } | BoundOpKind::Reduce { .. }
    ) && gather_count(bound) > 0
    {
        record_op_kind(OpKind::GatheredExpert);
    }
    // `resolved` is `Some` only from `execute_plan_with_placements`, once
    // `resolve_steps` has run: this whole block -- `kernel_cache_key`'s
    // `String`, `kernel_dispatch_shape`'s `Vec<Binding>`, and
    // `pipeline_for`'s own `format!` cache-key lookup -- is skipped on
    // every step of a plan-cache HIT, not merely made cheaper. `None` on
    // every other call site (`execute_plan`, the `*_op_timed` diagnostics),
    // byte-identical to this function's behavior before `resolved` existed.
    #[cfg(feature = "instrument")]
    let emit_started = read_ticks();
    let owned_bindings: Vec<Binding>;
    let owned_merge: Option<ResolvedMergeStep>;
    let (pipeline, bindings, grid, merge) = if let Some(step) =
        resolved.filter(|_| expert_buffers.is_none())
    {
        (
            step.pipeline.clone(),
            step.bindings.as_slice(),
            step.grid,
            step.merge.as_ref(),
        )
    } else if let Some(source_node) = expert_source_node {
        let mut cache_key = kernel_cache_key(bound, packed_operands, numeric_policy)?;
        cache_key.push(math_mode.cache_token());
        let uniform_codec = expert_buffers.and_then(|buffers| {
            let mut codecs = buffers
                .descriptor_records
                .iter()
                .filter(|descriptor| descriptor.byte_length != 0)
                .map(|descriptor| descriptor.codec);
            let first = codecs.next()?;
            (packed_operands.get(&source_node) == Some(&first)
                && codecs.all(|codec| codec == first))
            .then_some(first)
        });
        if let Some(codec) = uniform_codec {
            cache_key.push_str("_uniform_expert_");
            cache_key.push_str(codec.cache_token());
            if std::env::var_os("PROXIMA_DEBUG_EXPERT_EMIT").is_some() {
                eprintln!("expert lowering mode=uniform codec={codec:?} node={source_node:?}");
            }
        } else {
            cache_key.push_str("_mixed_expert");
            if std::env::var_os("PROXIMA_DEBUG_EXPERT_EMIT").is_some() {
                eprintln!("expert lowering mode=mixed node={source_node:?}");
            }
        }
        let (binding_identity, _) = kernel_dispatch_shape(bound, packed_operands, numeric_policy)?;
        cache_key.push_str(&format!("_{binding_identity:?}"));
        let kernel = MIXED_KERNEL_CACHE.with(|cache| cache.borrow().get(&cache_key).cloned());
        let kernel = match kernel {
            Some(kernel) => kernel,
            None => {
                let kernel = if let Some(codec) = uniform_codec {
                    crate::msl::emit_with_uniform_expert_source(
                        bound,
                        packed_operands,
                        numeric_policy,
                        source_node,
                        codec,
                    )?
                } else {
                    crate::msl::emit_with_expert_sources(
                        bound,
                        packed_operands,
                        numeric_policy,
                        source_node,
                    )?
                };
                MIXED_KERNEL_CACHE.with(|cache| {
                    cache.borrow_mut().insert(cache_key.clone(), kernel.clone());
                });
                kernel
            }
        };
        let pipeline = pipeline_for_kernel(device, &kernel, &cache_key, math_mode)?;
        owned_bindings = kernel.bindings;
        (pipeline, owned_bindings.as_slice(), kernel.grid, None)
    } else {
        // `kernel_cache_key`/`kernel_dispatch_shape` are the cheap halves of
        // `emit`'s work -- structural fingerprint, bindings, grid -- with no
        // MSL body text rendered. On a pipeline-cache HIT (the steady-decode
        // case, `plan_hits`/`gpu_exec`'s own row) `emit` itself is never
        // called; only a genuine miss inside `pipeline_for` pays for the
        // full render + compile.
        let mut cache_key = kernel_cache_key(bound, packed_operands, numeric_policy)?;
        cache_key.push(math_mode.cache_token());
        let (bindings, grid) = kernel_dispatch_shape(bound, packed_operands, numeric_policy)?;
        #[cfg(feature = "instrument")]
        {
            counter!(EMIT_CALLS, 1);
            counter!(EMIT_TICKS, elapsed_ticks(emit_started));
        }
        #[cfg(feature = "instrument")]
        let pipeline_started = read_ticks();
        let pipeline = pipeline_for(
            device,
            bound,
            packed_operands,
            &cache_key,
            math_mode,
            numeric_policy,
        )?;
        #[cfg(feature = "instrument")]
        {
            counter!(PIPELINE_LOOKUP_CALLS, 1);
            counter!(PIPELINE_LOOKUP_TICKS, elapsed_ticks(pipeline_started));
        }
        // This cold path (`resolved: None`: `execute_plan`, the
        // `*_op_timed` diagnostics) has no `Plan` to own a scratch buffer
        // or a resolved merge pipeline -- it resolves both itself, here,
        // exactly like `pipeline_for`/`kernel_dispatch_shape` just above,
        // rather than caching them plan-side. `None` (every non-
        // `CachedAttention` op, or a `CachedAttention` op under a policy
        // that withholds `ContextSplitMerge`) costs nothing extra: `emit_
        // cached_attention_merge` returns `None` before rendering anything.
        owned_merge = match crate::msl::emit_cached_attention_merge(bound, numeric_policy)? {
            Some(merge_kernel) => {
                let merge_cache_key = format!("{cache_key}_merge");
                let merge_pipeline =
                    pipeline_for_kernel(device, &merge_kernel, &merge_cache_key, math_mode)?;
                Some(ResolvedMergeStep {
                    pipeline: merge_pipeline,
                    bindings: merge_kernel.bindings,
                    grid: merge_kernel.grid,
                })
            }
            None => None,
        };
        owned_bindings = bindings;
        (
            pipeline,
            owned_bindings.as_slice(),
            grid,
            owned_merge.as_ref(),
        )
    };
    #[cfg(feature = "instrument")]
    let op_setup_started = read_ticks();
    let (output, output_offset) = match placement {
        Some((buffer, offset)) => (buffer.clone(), offset),
        None => (
            allocate_buffer(device, bound_output_len(bound), bound.dtype)?,
            0,
        ),
    };
    // `Some` only from `execute_plan_with_placements` (a `Plan`-owned,
    // call-to-call-reused buffer via `attention_scratch_buffer`). Every
    // other caller with a merge dispatch to satisfy falls back to a fresh,
    // throwaway one, sized by the SAME formula the plan-owned path uses
    // (`cached_attention_scratch_len`) -- correctness first, plan-owned
    // reuse is that call site's own optimization, not a requirement this
    // function imposes on every caller.
    let owned_scratch: Option<MetalBuffer> = if scratch.is_none() && merge.is_some() {
        Some(allocate_buffer(
            device,
            cached_attention_scratch_len(bound, numeric_policy).unwrap_or(0) as usize,
            DType::Float32,
        )?)
    } else {
        None
    };
    let scratch: Option<(&MetalBuffer, usize)> = match &owned_scratch {
        Some(buffer) => Some((buffer, 0)),
        None => scratch,
    };
    let uniforms = match plan_uniform {
        Some(buffer) => buffer.clone(),
        None => upload_uniforms(device, &pack_uniforms(bound, numeric_policy)?)?,
    };
    let gathers = gather_count(bound);
    let fault = (gathers > 0)
        .then(|| allocate_fault_buffer(device, gathers))
        .transpose()?;
    #[cfg(feature = "instrument")]
    {
        counter!(OP_SETUP_CALLS, 1);
        counter!(OP_SETUP_TICKS, elapsed_ticks(op_setup_started));
    }

    // ROW 92 named this residual (`omega/src/metal.rs:1873-1908` at that
    // row's line numbers) as four uncounted Metal API calls: `pipeline_for`'s
    // own lookup (now `PIPELINE_LOOKUP_TICKS` above), `setComputePipelineState`,
    // `bind_buffers`, `dispatch` -- the three below, wrapped together since
    // none does per-op device work heavy enough to need its own counter, and
    // splitting them finer would cost more ticks reading the clock than the
    // calls themselves take.
    #[cfg(feature = "instrument")]
    let encode_dispatch_started = read_ticks();
    encoder.setComputePipelineState(&pipeline);
    if let Err(err) = bind_buffers(
        encoder,
        bindings,
        device_buffers,
        (&output, output_offset),
        scratch,
        &uniforms,
        fault.as_ref(),
        expert_source_node,
        expert_buffers.map(|buffers| (&buffers.payloads, buffers.payload_offset)),
        expert_buffers.map(|buffers| &buffers.descriptors),
    ) {
        debug!(
            node = bound.node.0,
            kind = bound.kind.name(),
            ?bindings,
            resident = device_buffers.len() as u64,
            "encode_op failed binding this op's operands"
        );
        return Err(err);
    }
    // `render_gated_delta_net`'s own second output (ROW 547,
    // `docs/discipline.md`): `bindings` above only ever names ONE
    // `Binding::Output` (`bind_buffers`'s single `output` parameter cannot
    // carry two distinct node identities), so `state_out` binds at the next
    // free slot manually, right before dispatch, same as any other output.
    // ROW 555: a placed caller's `state_out` buffer does NOT already live in
    // `device_buffers` here -- the generic per-op placement lookup
    // (`output_placed.get(&bound.node)`, several lines above every
    // `encode_op` call site) is keyed by this op's OWN node, which is the
    // fused primary "out" node, never the absorbed `state_out` node
    // (`gated_delta_net_candidates`'s own `fused.node = output`). Without
    // `state_out_placement` threaded in separately, the caller's designated
    // ping-pong buffer was silently never written, and the interop loop's
    // own `ssm_input_placements` unconditionally re-bound `state_in` to that
    // (always-zero) buffer on the very next step -- resetting the recurrent
    // state to zero every dispatch. Measured: `state_out_placed_ptr`'s
    // post-dispatch `first4` stayed `[0.0, 0.0, 0.0, 0.0]` across steps
    // while the plan-internal `device_buffers[state_out]` buffer (a
    // DIFFERENT pointer) carried the real computed state nobody read back.
    if let BoundOpKind::GatedDeltaNet {
        state_out,
        #[cfg(feature = "instrument")]
        operands,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        ..
    } = &bound.kind
    {
        let state_elements = (*head_k_dim * *head_v_dim * *num_v_heads) as usize;
        let existing = device_buffers.get(state_out).cloned();
        #[cfg(feature = "instrument")]
        let placed = state_out_placement.is_some() || existing.is_some();
        let (state_buffer, state_offset) = match state_out_placement {
            Some((buffer, offset)) => (buffer.clone(), offset),
            None => match existing {
                Some(buffer) => buffer,
                None => (allocate_buffer(device, state_elements, bound.dtype)?, 0),
            },
        };
        // decision point: whether this node's recurrent state carried
        // forward from the caller's placement (ROW 549 -- a stale
        // `use_metal_output_placements` gate left this always missing for
        // qwen35moe's routed-expert + GDN decode step, discarding state
        // every dispatch).
        #[cfg(feature = "instrument")]
        {
            let state_in_node = operands[5].0;
            let state_in_resolved = device_buffers.get(&state_in_node);
            let state_in_pointer = state_in_resolved
                .map(|(buffer, _)| Retained::as_ptr(buffer) as usize)
                .unwrap_or(0);
            let state_in_offset = state_in_resolved.map(|(_, offset)| *offset).unwrap_or(0);
            let state_in_first4 = state_in_resolved.map(|(buffer, offset)| {
                let pointer = buffer.contents().as_ptr().cast::<u8>();
                let floats = unsafe {
                    core::slice::from_raw_parts(pointer.add(*offset).cast::<f32>(), 4)
                };
                [floats[0], floats[1], floats[2], floats[3]]
            });
            debug!(
                node = bound.node.0,
                state_out = state_out.0,
                state_in = state_in_node.0,
                placed,
                state_in_ptr = state_in_pointer as u64,
                state_in_offset = state_in_offset as u64,
                state_out_ptr = Retained::as_ptr(&state_buffer) as u64,
                state_out_offset = state_offset as u64,
                ?state_in_first4,
                "row 555: gated_delta_net state_out resolution"
            );
        }
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&state_buffer), state_offset, bindings.len());
        }
        if let Some(tracker) = hazard.as_deref_mut() {
            tracker.record(&[], Some(Retained::as_ptr(&state_buffer)));
        }
        device_buffers.insert(*state_out, (state_buffer, state_offset));
    }
    // `render_moe_topk`'s own 16 extra outputs (ROW 569, `docs/discipline.md`):
    // `bindings` above only ever names ONE `Binding::Output` (the same
    // single-output limit `GatedDeltaNet`'s own `state_out` arm names), so
    // every one of `routes[1..]`/`weights`/`weight_total` binds at the next
    // free slot manually, right before dispatch -- unlike `state_out`, none
    // of these needs a caller-supplied placement (`BoundOpKind::MoeTopK`'s
    // own doc: nothing here is cross-decode-step persistent recurrent
    // state), so this is the plain "resolve from `device_buffers`, or
    // allocate fresh" path with no placement lookup at all.
    if let BoundOpKind::MoeTopK {
        routes,
        weights,
        weight_total,
        ..
    } = &bound.kind
    {
        let extra_nodes: Vec<NodeId> = routes
            .iter()
            .skip(1)
            .chain(weights.iter())
            .chain(core::iter::once(weight_total))
            .copied()
            .collect();
        for (offset, extra_node) in extra_nodes.iter().enumerate() {
            let buffer_index = bindings.len() + offset;
            let existing = device_buffers.get(extra_node).cloned();
            let (extra_buffer, extra_offset) = match existing {
                Some(buffer) => buffer,
                None => (allocate_buffer(device, 1, bound.dtype)?, 0),
            };
            unsafe {
                encoder.setBuffer_offset_atIndex(
                    Some(&extra_buffer),
                    extra_offset,
                    buffer_index,
                );
            }
            if let Some(tracker) = hazard.as_deref_mut() {
                tracker.record(&[], Some(Retained::as_ptr(&extra_buffer)));
            }
            device_buffers.insert(*extra_node, (extra_buffer, extra_offset));
        }
    }
    dispatch(encoder, &pipeline, grid);
    // Redesign §4c: the split kernel above wrote its partial into `scratch`
    // (its own `Binding::Scratch` slot, never `output`); this second
    // dispatch reads it back and writes the position's REAL output --
    // `ggml_metal_op_flash_attn_ext`'s own one-op-two-dispatches shape,
    // `attention-kernel-design.md` §4c. Both land in the SAME encoder, in
    // program order: under `DispatchType::Serial` (this plan's default)
    // that ordering alone is Metal's own dependency guarantee, the same
    // reason this module's `HazardTracker` is never instantiated on that
    // path at all (`HazardTracker`'s own doc). Under `DispatchType::Concurrent`
    // program order is NOT a guarantee, so the split's write and the merge's
    // read of `scratch` are routed through `hazard` below -- the SAME
    // `HazardTracker` mechanism every other buffer's hazard uses, since
    // `Binding::Scratch` maps to its own buffer identity exactly like
    // `Binding::Output`/`Binding::Input` do (see that variant's own doc).
    if let Some(merge) = merge {
        if let (Some(tracker), Some((scratch_buffer, _))) = (&mut hazard, scratch) {
            let scratch_pointer = Retained::as_ptr(scratch_buffer);
            let output_pointer = Retained::as_ptr(&output);
            // the split dispatch above just wrote `scratch` -- record it,
            // then ask the tracker whether the merge's own read needs a
            // barrier first (always true: a write is never already visible
            // to a later read without one). Same "record, check, barrier"
            // shape `hazard_step` gives every other buffer, applied here to
            // the one edge that is entirely internal to this op.
            tracker.record(&[], Some(scratch_pointer));
            if tracker.needs_barrier(&[scratch_pointer], None) {
                encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                counter!(BARRIERS_EMITTED, 1);
                // a barrier is a full flush, so `reset` is correct -- but
                // the caller's OWN `hazard_step`, before this op was ever
                // encoded, already recorded this op's real output as
                // written; restore that record so a later sibling op's own
                // hazard check still sees it, exactly as if this intra-op
                // edge had never touched the tracker.
                tracker.reset();
                tracker.record(&[], Some(output_pointer));
            }
        }
        let merge_uniforms = upload_uniforms(
            device,
            &pack_cached_attention_merge_uniforms(bound, numeric_policy)?,
        )?;
        encoder.setComputePipelineState(&merge.pipeline);
        bind_buffers(
            encoder,
            &merge.bindings,
            device_buffers,
            (&output, output_offset),
            scratch,
            &merge_uniforms,
            None,
            None,
            None,
            None,
        )?;
        dispatch(encoder, &merge.pipeline, merge.grid);
    }
    #[cfg(feature = "instrument")]
    {
        counter!(ENCODE_DISPATCH_CALLS, 1);
        counter!(
            ENCODE_DISPATCH_TICKS,
            elapsed_ticks(encode_dispatch_started)
        );
    }

    device_buffers.insert(bound.node, (output, output_offset));
    Ok(fault.map(|fault_buffer| (fault_buffer, gathers)))
}

/// Reads back a dispatch's fault buffer and, if any slot recorded a fault,
/// turns it into the same `TensorError::GatherIndexOutOfRange`
/// `cpu::evaluate` reports for the identical fetched index. Slot order
/// matches `bound.operands()`' gather order — the same numbering
/// `crate::msl::gather_slots` and `push_gather_uniforms` both use — so the
/// first faulted slot's own `Lookup` supplies the extent to report.
pub(super) fn check_gather_fault(
    bound: &BoundOp,
    fault_buffer: &ProtocolObject<dyn MTLBuffer>,
    gather_count: usize,
) -> Result<(), MetalError> {
    let slots = read_fault_slots(fault_buffer, gather_count);
    let gathers: Vec<&Lookup> = bound
        .operands()
        .iter()
        .filter_map(|(_, _, gather)| gather.as_ref())
        .collect();
    for (slot, recorded) in slots.iter().enumerate() {
        if *recorded != 0 {
            if recorded & 0x8000_0000 != 0 {
                let encoded_expert = recorded & 0x7fff_ffff;
                return Err(MetalError::ExpertSourceMiss {
                    node: bound.node,
                    expert: encoded_expert.saturating_sub(1),
                });
            }
            return Err(TensorError::GatherIndexOutOfRange {
                node: bound.node,
                index: i64::from(*recorded - 1),
                extent: gathers[slot].extent,
            }
            .into());
        }
    }
    Ok(())
}

pub(super) fn read_fault_slots(buffer: &ProtocolObject<dyn MTLBuffer>, gather_count: usize) -> Vec<u32> {
    let pointer = buffer.contents();
    // SAFETY: allocated and sized to at least `gather_count` `u32`s by
    // `allocate_fault_buffer`, `storageModeShared` so CPU-visible now that
    // `waitUntilCompleted` has returned.
    unsafe { core::slice::from_raw_parts(pointer.as_ptr().cast::<u32>(), gather_count.max(1)) }
        .to_vec()
}

/// Widens a device buffer back to the host's f32 contract — see this
/// module's dtype doc for why that widening happens exactly once, here,
/// mirroring the narrowing [`upload_block`] does on the way in. `node`
/// names the output this read-back is for, used only to point an
/// [`EmitError::UnsupportedDType`] at the right place — same totality-guard
/// stance as [`upload_block`]'s `node` parameter. `byte_offset` honors a
/// placed output's own non-zero start ([`execute_plan_with_placements`]'s
/// `output_placements`): an ordinary, non-placed node's buffer is still
/// always freshly allocated at offset `0`, so passing `0` here for that case
/// is unchanged behavior, not a special case this function has to know about.
pub(super) fn read_back(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    byte_offset: usize,
    element_count: usize,
    node: NodeId,
    dtype: DType,
) -> Result<Vec<f32>, MetalError> {
    if element_count == 0 {
        return Ok(Vec::new());
    }
    match dtype {
        DType::Float16 => Ok(read_back_half(buffer, byte_offset, element_count)),
        DType::Float32
        | DType::BFloat16
        | DType::Bool
        | DType::Int8
        | DType::UInt8
        | DType::Int32
        | DType::UInt32 => Ok(read_back_as_device_f32(buffer, byte_offset, element_count)),
        DType::Int16
        | DType::UInt16
        | DType::Int64
        | DType::UInt64
        | DType::Int128
        | DType::UInt128
        | DType::Float64 => Err(EmitError::UnsupportedDType { node, dtype }.into()),
    }
}

/// Reads back an output whose device buffer holds 4-byte `float` elements
/// regardless of its logical [`DType`] — `msl::type_token` emits `"float"`
/// (never a narrower or integer MSL type) for every one of `Float32`,
/// `BFloat16`, `Bool`, `Int8`, `UInt8`, `Int32`, and `UInt32`, so the bytes
/// this reads are always IEEE-754 binary32 on the device side no matter
/// which of those logical dtypes the caller asked for. `Float16` is the one
/// dtype this backend narrows on-device (`read_back_half`); every other
/// dtype that reaches this function was upcast to `float` before dispatch
/// and is downcast back to its logical dtype by the caller after this
/// returns, not by this function.
pub(super) fn read_back_as_device_f32(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    byte_offset: usize,
    element_count: usize,
) -> Vec<f32> {
    debug_assert!(
        buffer.length() >= byte_offset + element_count * size_of::<f32>(),
        "device buffer too small for a device-f32 read-back: buffer.length()={}, byte_offset={byte_offset}, element_count={element_count}",
        buffer.length(),
    );
    let pointer = buffer.contents();
    // SAFETY: `buffer` is `storageModeShared`, so `contents()` is a
    // CPU-visible pointer to at least `byte_offset + element_count * 4`
    // initialized bytes — every output buffer this driver allocates is
    // sized to at least that many bytes past a placed node's own offset
    // (see `allocate_buffer`'s caller, `dispatch_op`, and
    // `execute_plan_with_placements`'s caller contract for a placed one)
    // before this point is reached.
    unsafe {
        let base = pointer.as_ptr().cast::<u8>().add(byte_offset).cast::<f32>();
        core::slice::from_raw_parts(base, element_count)
    }
    .to_vec()
}

/// [`read_back_as_device_f32`], writing into a caller-owned, reused `target`
/// instead of returning a fresh `Vec` -- [`finish`]'s `recycle` parameter
/// feeds this a buffer popped from [`execute_plan_with_placements`]'s own
/// pool (a PREVIOUS `Evaluated`'s storage, given back via
/// [`Evaluated::into_scratch`]) so the root output's warm-step read-back
/// allocates nothing (ROW 303's residual). `target.clear()` keeps its
/// capacity, so only a call whose output GREW past that capacity allocates.
pub(super) fn read_back_as_device_f32_into(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    byte_offset: usize,
    element_count: usize,
    target: &mut Vec<f32>,
) {
    debug_assert!(
        buffer.length() >= byte_offset + element_count * size_of::<f32>(),
        "device buffer too small for a device-f32 read-back: buffer.length()={}, byte_offset={byte_offset}, element_count={element_count}",
        buffer.length(),
    );
    let pointer = buffer.contents();
    // SAFETY: see `read_back_as_device_f32`'s own doc -- identical sizing
    // guarantee, just writing into `target` instead of collecting into a
    // fresh `Vec`.
    let slice = unsafe {
        let base = pointer.as_ptr().cast::<u8>().add(byte_offset).cast::<f32>();
        core::slice::from_raw_parts(base, element_count)
    };
    target.clear();
    target.extend_from_slice(slice);
}

pub(super) fn read_back_half(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    byte_offset: usize,
    element_count: usize,
) -> Vec<f32> {
    let pointer = buffer.contents();
    // SAFETY: `buffer` is `storageModeShared`, so `contents()` is a
    // CPU-visible pointer to at least `byte_offset + element_count * 2`
    // initialized bytes — the same sizing guarantee `read_back_as_device_f32`
    // relies on, just over the narrower element width `allocate_buffer`
    // used for a `Float16` node.
    let narrow = unsafe {
        let base = pointer.as_ptr().cast::<u8>().add(byte_offset).cast::<f16>();
        core::slice::from_raw_parts(base, element_count)
    };
    narrow.iter().map(|value| value.to_f32()).collect()
}

/// `plan` supplies every field this used to take individually (`program`,
/// `prepared.index_nodes`/`shapes`/`effective_outputs`/`root`) -- one
/// argument instead of five, under clippy's `too_many_arguments` with
/// `recycle` added by this landing.
pub(super) fn finish(
    plan: &Plan,
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    caller_owned: &BTreeSet<NodeId>,
    // A buffer popped from `execute_plan_with_placements`'s own recycle
    // pool, if one was available -- reused ONLY for `root`'s read-back (the
    // one output a caller always requests, `Evaluated::root`'s own doc) and
    // only on the `read_back_as_device_f32`-shaped dtype path; every other
    // output, and a `Float16` root, still allocates fresh exactly as before
    // this landing.
    recycle: Option<Vec<f32>>,
) -> Result<Evaluated, MetalError> {
    let program = &plan.program;
    let index_nodes = &plan.prepared.index_nodes;
    let shapes = &plan.prepared.shapes;
    let effective_outputs = &plan.prepared.effective_outputs;
    let root = plan.prepared.root;
    let mut results = Vec::with_capacity(effective_outputs.len());
    let mut placed = BTreeSet::new();
    let mut recycle = recycle;
    #[cfg(feature = "instrument")]
    let readback_started = read_ticks();
    for node in effective_outputs {
        // A caller-owned (placed) output's bytes already live in the
        // caller's own `PlacedBuffer` (`execute_plan_with_placements`'s own
        // doc) -- the KV-cache shape this exists for never reads them back
        // through `Evaluated` at all, it reads the placed buffer directly.
        // Copying them into a fresh `Vec` here would be dead work on the
        // decode critical path (a memcpy plus an allocation per node, after
        // `waitUntilCompleted`, for bytes nothing downstream consumes) --
        // `root` is the one exception: a caller always expects `.root()` to
        // resolve, so it is read back even if it happens to be placed.
        //
        // `placed` records the skip explicitly so `Evaluated::is_placed`
        // can tell a caller "this was placed, not missing" — see
        // `Evaluated::from_parts_with_placed`'s own doc.
        if *node != root && caller_owned.contains(node) {
            placed.insert(*node);
            continue;
        }
        let shape = shapes.of(*node).to_vec();
        let dtype = gpu_dtype(program, index_nodes, *node);
        let data = match device_buffers.get(node) {
            // a plain [`execute_plan`] output's buffer is freshly allocated
            // by `encode_op` at offset 0, but an [`execute_plan_with_placements`]
            // output lands in the CALLER's own buffer at whatever byte offset
            // that call chose (`output_placements`) -- so `offset` here is
            // read, never assumed to be `0`, or a placed node's read-back
            // would silently return its buffer's UNRELATED leading bytes
            // instead of the bytes this node actually wrote.
            Some((buffer, offset)) => {
                let recyclable_dtype = matches!(
                    dtype,
                    DType::Float32
                        | DType::BFloat16
                        | DType::Bool
                        | DType::Int8
                        | DType::UInt8
                        | DType::Int32
                        | DType::UInt32
                );
                if *node == root
                    && recyclable_dtype
                    && let Some(mut target) = recycle.take()
                {
                    read_back_as_device_f32_into(
                        buffer,
                        *offset,
                        element_count(&shape),
                        &mut target,
                    );
                    target
                } else {
                    read_back(buffer, *offset, element_count(&shape), *node, dtype)?
                }
            }
            None => Vec::new(),
        };
        #[cfg(feature = "instrument")]
        {
            counter!(READBACK_CALLS, 1);
            counter!(READBACK_BYTES, size_of_val(data.as_slice()) as u64);
        }
        results.push((*node, shape, data));
    }
    #[cfg(feature = "instrument")]
    counter!(READBACK_TICKS, elapsed_ticks(readback_started));
    // this backend's buffer lifetime is managed by Metal's own
    // retain/release, not counted the way `cpu::evaluate` counts its
    // `Vec<Option<Vec<f32>>>` table, so peak_live_buffers is not tracked
    // here — see `Evaluated`'s own doc for why `None` is the honest answer
    // rather than a number that would not mean the same thing.
    Ok(Evaluated::from_parts_with_placed(
        root, results, None, placed,
    ))
}

#[cfg(all(test, feature = "instrument"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(super) mod operand_tensor_bytes_tests {
    //! CARD 0.2's own tests: `operand_tensor_bytes` is the fix for the
    //! defect verified verbatim in the card's `opens` -- `operand_bytes`
    //! summing `device_buffers[source].0.length()`, which reports the
    //! shared checkpoint-mapping buffer's own size for EVERY tensor that
    //! buffer serves, rather than that tensor's own byte count. Every case
    //! here is a real GGUF codec's declared block shape, read from
    //! `crate::msl`'s own block constants rather than hand-computed, so a
    //! constant drifting there fails this test instead of silently
    //! agreeing with a stale hand-copy.

    use alloc::string::String;
    use alloc::vec;
    use core::slice;

    use objc2_metal::MTLBuffer;
    use proxima_tensor::{AlignedBuffer, DType, Extent, Op, infer};

    use super::{
        BTreeSet, NodeId, PackedCodec, PackedOperands, device_and_queue, element_count,
        operand_tensor_bytes, page_size, register_checkpoint_mapping, upload_packed_bytes,
    };
    use crate::msl::{Q4K_BLOCK_BYTES, Q5K_BLOCK_BYTES};

    /// One `Op::Input` program, so `operand_tensor_bytes` can be exercised
    /// against a real [`proxima_tensor::Shapes`] the same way [`plan`]
    /// builds one, rather than a hand-rolled shape table `Shapes`'s own
    /// module keeps private.
    fn single_input_shapes(elements: u32) -> (Vec<Op>, proxima_tensor::Shapes) {
        let program = vec![Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(elements)],
            name: Some(String::from("w")),
        }];
        let shapes = infer(&program, &[]).expect("a single Input node always infers");
        (program, shapes)
    }

    #[test]
    fn q4k_operand_reports_rows_times_k_times_144_over_256() {
        let (program, shapes) = single_input_shapes(2 * 256);
        let mut packed_operands = PackedOperands::new();
        packed_operands.insert(NodeId(0), PackedCodec::Q4K);

        let bytes = operand_tensor_bytes(
            &program,
            &BTreeSet::new(),
            &shapes,
            &packed_operands,
            NodeId(0),
            None,
        );

        assert_eq!(bytes, 2 * Q4K_BLOCK_BYTES as u64);
    }

    #[test]
    fn q5k_operand_reports_rows_times_k_times_176_over_256() {
        let (program, shapes) = single_input_shapes(3 * 256);
        let mut packed_operands = PackedOperands::new();
        packed_operands.insert(NodeId(0), PackedCodec::Q5K);

        let bytes = operand_tensor_bytes(
            &program,
            &BTreeSet::new(),
            &shapes,
            &packed_operands,
            NodeId(0),
            None,
        );

        assert_eq!(bytes, 3 * Q5K_BLOCK_BYTES as u64);
    }

    #[test]
    fn f32_operand_reports_element_count_times_4() {
        let (program, shapes) = single_input_shapes(4096);
        let packed_operands = PackedOperands::new();

        let bytes = operand_tensor_bytes(
            &program,
            &BTreeSet::new(),
            &shapes,
            &packed_operands,
            NodeId(0),
            None,
        );

        assert_eq!(bytes, 4096 * 4);
    }

    /// ROW 544: a gathered packed operand (a grouped MoE expert weight)
    /// must report bytes for the rows its OWN `lookup.indices` selects,
    /// never the full declared expert-stack shape `shapes.of(source)`
    /// carries -- the defect this test guards against overstated a
    /// qwen35moe-shaped grouped gate/up dispatch's `operand_bytes` by
    /// ~7x the whole 256-expert stack (`proxima-tensor/docs/discipline.md`
    /// ROW 543/544).
    #[test]
    fn gathered_q4k_operand_reports_selected_rows_not_the_whole_stack() {
        const EXPERT_COUNT: u32 = 256;
        const ROW_ELEMENTS: u32 = 512;
        const SELECTED: u32 = 8;

        let mut program = vec![
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(EXPERT_COUNT), Extent::Static(ROW_ELEMENTS)],
                name: Some(String::from("expert_w")),
            },
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(SELECTED)],
                name: Some(String::from("routes")),
            },
        ];
        let weight_node = NodeId(0);
        let routes_node = NodeId(1);
        let shapes = infer(&program, &[]).expect("weight+routes program infers");
        let _ = &mut program;

        let mut packed_operands = PackedOperands::new();
        packed_operands.insert(weight_node, PackedCodec::Q4K);

        let lookup = proxima_tensor::Lookup {
            indices: routes_node,
            index_layout: proxima_tensor::Layout {
                base: 0,
                strides: Default::default(),
            },
            element_stride: ROW_ELEMENTS as i64,
            extent: EXPERT_COUNT as u64,
        };

        let bytes = operand_tensor_bytes(
            &program,
            &BTreeSet::new(),
            &shapes,
            &packed_operands,
            weight_node,
            Some(&lookup),
        );

        assert_eq!(
            bytes,
            SELECTED as u64 * ROW_ELEMENTS as u64 * Q4K_BLOCK_BYTES as u64 / 256,
            "must report exactly the {SELECTED} selected rows, not the full \
             {EXPERT_COUNT}-expert stack"
        );
    }

    /// The mechanism-level counterpart to the three pure-function cases
    /// above: a tensor served by [`checkpoint_mapping_offset`] binds a
    /// buffer spanning the WHOLE registered mapping (several pages here),
    /// at a nonzero byte offset -- `bound_buffer_bytes` (the old,
    /// defective `operand_bytes`) reports that whole-mapping length for
    /// every tensor sharing it, while `operand_tensor_bytes` reports only
    /// this one tensor's own declared byte length, unaffected by which
    /// buffer or offset backs it.
    #[test]
    fn operand_bound_at_a_nonzero_mapping_offset_reports_the_tensor_length_not_the_shared_buffer() {
        let Ok((device, _queue)) = device_and_queue() else {
            // no Metal device on this host (e.g. a headless CI runner) --
            // every other Metal-gated test in this crate skips the same
            // way, so this one does too rather than failing spuriously.
            return;
        };
        let page = page_size();
        // two pages of f32 headroom -- comfortably larger than the single
        // Q4_K block this test carves a sub-slice out of, so the mapping's
        // own length is provably larger than the tensor's.
        let mapping = AlignedBuffer::new(page / 4 * 2, page).expect("page-aligned test mapping");
        // SAFETY: `mapping` owns `mapping.len()` initialized `f32`s; a byte
        // view of the exact same live range is valid for as long as
        // `mapping` is (it outlives every use of `mapping_bytes` below).
        let mapping_bytes: &[u8] =
            unsafe { slice::from_raw_parts(mapping.as_ptr().cast::<u8>(), mapping.len() * 4) };
        register_checkpoint_mapping(mapping_bytes);

        // a one-block Q4_K tensor starting at byte 144 (one block in) --
        // nonzero AND not page-aligned, so it can only reach the GPU
        // through `checkpoint_mapping_offset`, never the plain no-copy path.
        let tensor_offset = Q4K_BLOCK_BYTES;
        let tensor_bytes = &mapping_bytes[tensor_offset..tensor_offset + Q4K_BLOCK_BYTES];

        let (buffer, bound_offset) =
            upload_packed_bytes(&device, tensor_bytes, None).expect("checkpoint-mapping upload");

        assert_eq!(
            bound_offset, tensor_offset,
            "checkpoint_mapping_offset must report this tensor's own byte offset"
        );
        let bound_buffer_bytes = buffer.length();
        assert!(
            bound_buffer_bytes > tensor_bytes.len(),
            "the bound buffer spans the whole mapping ({bound_buffer_bytes} bytes), \
             not this one tensor ({} bytes) -- this IS defect 1",
            tensor_bytes.len()
        );

        let (program, shapes) = single_input_shapes(256);
        let mut packed_operands = PackedOperands::new();
        packed_operands.insert(NodeId(0), PackedCodec::Q4K);
        let operand_bytes = operand_tensor_bytes(
            &program,
            &BTreeSet::new(),
            &shapes,
            &packed_operands,
            NodeId(0),
            None,
        );
        assert_eq!(
            operand_bytes,
            tensor_bytes.len() as u64,
            "operand_tensor_bytes must report the TENSOR's own length regardless of \
             which shared buffer or offset backs it"
        );
        assert_ne!(
            operand_bytes, bound_buffer_bytes as u64,
            "the fix's whole point: operand bytes and bound buffer bytes now differ"
        );

        // leave no mapping registered for whichever test this thread runs
        // next -- `CHECKPOINT_MAPPING` is thread-local, but the default
        // std test harness reuses threads across tests in the same binary.
        super::CHECKPOINT_MAPPING.with(|cell| *cell.borrow_mut() = None);
        let _ = element_count(shapes.of(NodeId(0)));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(super) mod uniform_cache_tests {
    //! `metal::UNIFORM_BUFFERS`'s LRU bound: no eviction meant a workload
    //! whose uniform bytes vary per call grew the cache without bound (a
    //! leak by construction). These tests exercise the real
    //! `upload_uniforms` upload path against a real Metal device, so they
    //! skip (rather than fail) on a headless host with no GPU -- the same
    //! convention every other Metal-gated test in this module follows.

    use super::{
        device_and_queue, reset_uniform_cache_for_test, uniform_cache_len, upload_uniforms,
    };
    use crate::sized::UNIFORM_CACHE_ENTRIES;

    /// A distinct, non-empty uniform blob per `index` -- `upload_uniforms`
    /// requires non-empty bytes (every real `Uniforms` struct has at least
    /// two `long` fields), and content-keying means two different indices
    /// must produce byte-distinct blobs.
    fn blob(index: u64) -> Vec<u8> {
        index.to_le_bytes().to_vec()
    }

    #[test]
    fn filling_past_capacity_evicts_the_least_recently_used_entry() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_uniform_cache_for_test();
        let capacity = UNIFORM_CACHE_ENTRIES as usize;

        for index in 0..capacity as u64 {
            upload_uniforms(&device, &blob(index)).expect("upload within capacity");
        }
        assert_eq!(uniform_cache_len(), capacity);

        // touch key 0 -- a cache hit that must refresh its tick, protecting
        // it from the eviction the next insert triggers.
        let reuses_before_touch = super::UNIFORM_BUFFER_REUSES.get();
        upload_uniforms(&device, &blob(0)).expect("touch key 0");
        assert_eq!(
            super::UNIFORM_BUFFER_REUSES.get(),
            reuses_before_touch + 1,
            "touching key 0 must be a cache hit"
        );

        // one more distinct blob forces an eviction; key 1 is now the
        // least recently used (key 0 was just touched, keys 2.. were
        // inserted after key 1).
        upload_uniforms(&device, &blob(capacity as u64)).expect("upload past capacity");
        assert_eq!(
            uniform_cache_len(),
            capacity,
            "cache must stay bounded at capacity after eviction"
        );

        let reuses_before_key_zero = super::UNIFORM_BUFFER_REUSES.get();
        upload_uniforms(&device, &blob(0)).expect("key 0 must still be resident");
        assert_eq!(
            super::UNIFORM_BUFFER_REUSES.get(),
            reuses_before_key_zero + 1,
            "key 0 (touched) must have survived the eviction"
        );

        let reuses_before_key_one = super::UNIFORM_BUFFER_REUSES.get();
        upload_uniforms(&device, &blob(1)).expect("key 1 re-upload after eviction");
        assert_eq!(
            super::UNIFORM_BUFFER_REUSES.get(),
            reuses_before_key_one,
            "key 1 was the least recently used entry and must have missed the cache"
        );
    }

    #[test]
    fn a_cache_hit_refreshes_the_use_tick() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_uniform_cache_for_test();

        upload_uniforms(&device, &blob(0)).expect("insert key 0");
        upload_uniforms(&device, &blob(1)).expect("insert key 1");

        let tick_of_key = |bytes: &[u8]| -> u64 {
            super::UNIFORM_BUFFERS.with(|cache| cache.borrow().get(bytes).expect("key present").1)
        };
        let tick_before = tick_of_key(&blob(0));

        upload_uniforms(&device, &blob(0)).expect("touch key 0 again");
        let tick_after = tick_of_key(&blob(0));

        assert!(
            tick_after > tick_before,
            "a cache hit must advance key 0's use-tick ({tick_before} -> {tick_after})"
        );
    }

    #[test]
    fn uniform_buffer_reuses_still_counts_hits() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_uniform_cache_for_test();

        let reuses_before = super::UNIFORM_BUFFER_REUSES.get();
        upload_uniforms(&device, &blob(0)).expect("first upload is a miss");
        assert_eq!(
            super::UNIFORM_BUFFER_REUSES.get(),
            reuses_before,
            "a fresh blob must not count as a reuse"
        );

        upload_uniforms(&device, &blob(0)).expect("second upload of the same bytes is a hit");
        assert_eq!(
            super::UNIFORM_BUFFER_REUSES.get(),
            reuses_before + 1,
            "re-uploading identical bytes must count exactly one reuse"
        );
    }
}

/// CARD 6.5's own soundness tests: [`BufferArena`]/[`PlanUniforms`] built by
/// [`plan`] and bound (never allocated) by [`encode_op`] on every call
/// against a plan already in the plan cache.
#[cfg(all(test, feature = "metal-plan-stable-buffers"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(super) mod arena_tests {
    use alloc::vec;

    use objc2_metal::MTLDevice;
    use proxima_tensor::{
        DType, Extent, IndexMap, NodeId, NumericPolicy, Op, QuantizedBlock, ScalarOp, append, cpu,
        projection,
    };

    use super::{
        arena_placement, device_and_queue, execute_plan, execute_plan_with_placements, plan,
        plan_uniform_buffer,
    };

    /// `Input(a, extent) -> Identity -> Identity -> Input(b, extent) ->
    /// Identity` -- three dispatched (non-`Input`) elementwise ops, `node[0]`
    /// consumed only by `node[1]` so it retires right after `node[1]` runs,
    /// freeing a slot a same-size later op COULD reuse. Returns the program
    /// and the three dispatched nodes in position order.
    fn three_stage_chain(extent_a: u32, extent_b: u32) -> (Vec<Op>, [NodeId; 3]) {
        let mut program = Vec::new();
        let input_a = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(extent_a)],
                name: None,
            },
        );
        let stage_zero = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: vec![(input_a, IndexMap::Affine(projection(1, &[0])))],
                name: None,
            },
        );
        let stage_one = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: vec![(stage_zero, IndexMap::Affine(projection(1, &[0])))],
                name: None,
            },
        );
        let input_b = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(extent_b)],
                name: None,
            },
        );
        let stage_two = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: vec![(input_b, IndexMap::Affine(projection(1, &[0])))],
                name: None,
            },
        );
        (program, [stage_zero, stage_one, stage_two])
    }

    /// A ten-stage `Identity` chain, every stage the same `extent`, only the
    /// LAST stage declared reachable by any test that wants a real forward
    /// -- the shape [`live_buffer_count_bounded_in_steady_state`] and
    /// [`a_pooled_slot_is_never_rebound_before_its_last_consumers_position`]
    /// both need: whatever the binder's own elementwise-fusion pass leaves
    /// dispatched, over a long enough chain to force at least one real
    /// arena slot reuse if reuse is happening at all.
    fn ten_stage_chain(extent: u32) -> (Vec<Op>, Vec<NodeId>) {
        let mut program = Vec::new();
        let mut previous = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(extent)],
                name: None,
            },
        );
        let mut nodes = Vec::new();
        for _ in 0..10 {
            previous = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Identity,
                    operands: vec![(previous, IndexMap::Affine(projection(1, &[0])))],
                    name: None,
                },
            );
            nodes.push(previous);
        }
        (program, nodes)
    }

    /// `shared = Negate(input)` fed to TWO consumers -- fan-out the
    /// binder's own single-consumer elementwise fusion cannot collapse
    /// (inlining `shared` into either consumer alone would still leave the
    /// other needing a materialized read, so `shared` stays its own
    /// dispatched `BoundOp`). Neither consumer is declared an output, so
    /// `shared` genuinely retires once both have run -- unlike a plain
    /// linear `Identity` chain (see `ten_stage_chain`'s own doc), which
    /// fuses down to a single dispatch and never exercises retirement at
    /// all. Appends onto the CALLER's `program` so several diamonds can
    /// share one program (and one plan), each with its own `shared` node of
    /// the SAME extent -- the shape this test needs to prove reuse across
    /// independent diamonds, not just within one.
    fn append_diamond(program: &mut Vec<Op>, extent: u32) -> (NodeId, NodeId, NodeId) {
        let input = append(
            program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(extent)],
                name: None,
            },
        );
        let shared = append(
            program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Negate,
                operands: vec![(input, IndexMap::Affine(projection(1, &[0])))],
                name: None,
            },
        );
        let consumer_a = append(
            program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: vec![(shared, IndexMap::Affine(projection(1, &[0])))],
                name: None,
            },
        );
        let consumer_b = append(
            program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Negate,
                operands: vec![(shared, IndexMap::Affine(projection(1, &[0])))],
                name: None,
            },
        );
        (shared, consumer_a, consumer_b)
    }

    #[test]
    fn pooled_output_equals_the_cpu_oracle_on_a_real_forward() {
        let (program, [stage_zero, stage_one, stage_two]) = three_stage_chain(4, 4);
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [10.0f32, 20.0, 30.0, 40.0];
        let outputs = [stage_zero, stage_one, stage_two];

        let cpu_oracle = cpu::evaluate(&program, &[], &[&a, &b], &outputs)
            .expect("the CPU oracle evaluates the same chain");

        let resolved_plan = plan(
            &program,
            &[],
            &[QuantizedBlock::Float32(&a), QuantizedBlock::Float32(&b)],
            &outputs,
            NumericPolicy::default(),
        )
        .expect("plans the chain once -- its arena and plan uniforms build lazily below");
        let pooled = execute_plan_with_placements(
            &resolved_plan,
            &[QuantizedBlock::Float32(&a), QuantizedBlock::Float32(&b)],
            &[],
            &[],
            &mut Vec::new(),
        )
        .expect("runs the chain against the arena-bound, plan-owned-uniform path");

        for node in outputs {
            let (expected, _shape) = cpu_oracle.get(node).expect("oracle has this output");
            let (actual, _shape) = pooled.get(node).expect("pooled run has this output");
            assert_eq!(
                actual, expected,
                "arena-bound output for {node:?} must equal the CPU oracle's -- \
                 pooled and unpooled must agree on the real forward"
            );
        }
    }

    /// The direct witness for the fix this module carries: `execute_plan`
    /// never binds a placement (it always passes `None, None` into
    /// `encode_op`), so it must never trigger `arena_placement`/
    /// `plan_uniform_buffer`'s device allocation -- before this change,
    /// [`plan`] built the arena and plan uniforms unconditionally, so this
    /// same call sequence would have found both `OnceCell`s already full.
    #[test]
    fn execute_plan_never_builds_the_arena_or_uniforms_for_the_unplaced_path() {
        let (program, [stage_zero, stage_one, stage_two]) = three_stage_chain(4, 4);
        let a = [1.0f32; 4];
        let b = [2.0f32; 4];
        let outputs = [stage_zero, stage_one, stage_two];
        let blocks = [QuantizedBlock::Float32(&a), QuantizedBlock::Float32(&b)];
        let resolved_plan = plan(&program, &[], &blocks, &outputs, NumericPolicy::default())
            .expect("plans the three-stage chain");

        execute_plan(&resolved_plan, &blocks).expect("runs the unplaced program");

        assert!(
            resolved_plan.arena.get().is_none(),
            "execute_plan took the unplaced path and must never have built the arena"
        );
        assert!(
            resolved_plan.uniforms.get().is_none(),
            "execute_plan took the unplaced path and must never have built plan uniforms"
        );
        assert_eq!(
            resolved_plan.arena_peak_bytes(),
            None,
            "the public accessor must agree with the private field: no placed \
             execution happened, so there is no arena peak to report"
        );
    }

    #[test]
    fn a_pooled_slot_is_never_rebound_before_its_last_consumers_position() {
        // five independent diamonds, same extent -- each diamond's `shared`
        // node is a genuine intermediate (never declared an output) that
        // retires once both its consumers have run, and every diamond
        // requests the SAME byte length, so the free list has every
        // opportunity to hand an earlier diamond's freed slot to a later
        // one.
        let mut program = Vec::new();
        let mut outputs = Vec::new();
        for _ in 0..5 {
            let (_shared, consumer_a, consumer_b) = append_diamond(&mut program, 4);
            outputs.push(consumer_a);
            outputs.push(consumer_b);
        }
        let a = [1.0f32; 4];
        let blocks: Vec<QuantizedBlock<'_>> =
            core::iter::repeat_n(QuantizedBlock::Float32(a.as_slice()), 5).collect();
        let resolved_plan = plan(&program, &[], &blocks, &outputs, NumericPolicy::default())
            .expect("plans the five-diamond program");
        arena_placement(&resolved_plan, 0).expect("builds the arena on first placement lookup");

        let arena = resolved_plan
            .arena
            .get()
            .expect("arena was just built above");
        let retires = &resolved_plan.prepared.retires;
        let resolved = &resolved_plan.prepared.resolved;

        // group every position by which physical slot it was bound to --
        // any slot with 2+ positions is a real reuse event `node_retirement`
        // must have authorized.
        let mut positions_by_slot: alloc::collections::BTreeMap<usize, Vec<usize>> =
            alloc::collections::BTreeMap::new();
        for (position, &slot) in arena.position_slot.iter().enumerate() {
            positions_by_slot.entry(slot).or_default().push(position);
        }

        let mut reuse_events_checked = 0;
        for occupants in positions_by_slot.values() {
            for window in occupants.windows(2) {
                let (earlier, later) = (window[0], window[1]);
                let earlier_node = resolved[earlier].node;
                let freed_before_reassignment = retires[..later]
                    .iter()
                    .any(|freed_here| freed_here.contains(&earlier_node));
                assert!(
                    freed_before_reassignment,
                    "slot reused at position {later} before its earlier occupant \
                     (position {earlier}, node {earlier_node:?}) was retired -- a pooled \
                     buffer must never be rebound before its last consumer's position"
                );
                reuse_events_checked += 1;
            }
        }
        assert!(
            reuse_events_checked > 0,
            "degenerate gate: five same-extent diamonds must produce at least one real \
             slot reuse, or this test proves nothing"
        );
    }

    #[test]
    fn a_programs_output_buffer_is_never_in_the_free_list() {
        // stage_zero (position 0, 4 elements) retires right after stage_one
        // consumes it (position 1). stage_two (position 2) requests the
        // SAME 4-element extent -- if stage_zero's own OUTPUT were ever
        // freed, this arm would prove nothing; declaring it a program
        // OUTPUT is what pins it, per `node_retirement`'s own exclusion.
        let (program, [stage_zero, stage_one, stage_two]) = three_stage_chain(4, 4);
        let a = [1.0f32; 4];
        let b = [2.0f32; 4];
        // stage_zero declared as an output pins it -- node_retirement never
        // retires a declared output, so its slot can never reach the free
        // list stage_two's identical-sized request would otherwise pull from.
        let outputs = [stage_zero, stage_one, stage_two];
        let blocks = [QuantizedBlock::Float32(&a), QuantizedBlock::Float32(&b)];
        let resolved_plan = plan(&program, &[], &blocks, &outputs, NumericPolicy::default())
            .expect("plans the pinned-output chain");
        arena_placement(&resolved_plan, 0).expect("builds the arena on first placement lookup");
        let arena = resolved_plan
            .arena
            .get()
            .expect("arena was just built above");

        let position_of = |node: NodeId| {
            resolved_plan
                .prepared
                .resolved
                .iter()
                .position(|bound| bound.node == node)
                .expect("node is dispatched")
        };
        let stage_zero_slot = arena.position_slot[position_of(stage_zero)];
        let stage_two_slot = arena.position_slot[position_of(stage_two)];
        assert_ne!(
            stage_zero_slot, stage_two_slot,
            "a pinned program output's slot must never be handed to a later same-size op"
        );
    }

    #[test]
    fn an_extent_change_forces_a_documented_realloc() {
        // stage_zero (4 elements, 16 bytes) and stage_two (8 elements, 32
        // bytes) are both declared outputs -- independent single-op leaves,
        // immune to elementwise fusion -- so this test isolates exactly one
        // thing: two differently-sized requests in the SAME plan must never
        // be conflated by the arena's size-class bookkeeping.
        let (program, [stage_zero, _stage_one, stage_two]) = three_stage_chain(4, 8);
        let a = [1.0f32; 4];
        let b = [2.0f32; 8];
        let outputs = [stage_zero, stage_two];
        let blocks = [QuantizedBlock::Float32(&a), QuantizedBlock::Float32(&b)];
        let resolved_plan = plan(&program, &[], &blocks, &outputs, NumericPolicy::default())
            .expect("plans the size-mismatched chain");
        arena_placement(&resolved_plan, 0).expect("builds the arena on first placement lookup");
        let arena = resolved_plan
            .arena
            .get()
            .expect("arena was just built above");

        let position_of = |node: NodeId| {
            resolved_plan
                .prepared
                .resolved
                .iter()
                .position(|bound| bound.node == node)
                .expect("node is dispatched")
        };
        let stage_zero_slot = arena.position_slot[position_of(stage_zero)];
        let stage_two_slot = arena.position_slot[position_of(stage_two)];
        assert_ne!(
            stage_zero_slot, stage_two_slot,
            "a byte-length mismatch must force a genuinely new slot, never a reused one"
        );
        assert_eq!(
            arena.slot_byte_len(stage_two_slot),
            8 * size_of::<f32>(),
            "the new slot must be sized to the LARGER extent's own byte length"
        );
    }

    #[test]
    fn uniform_contents_change_per_step_while_the_buffer_identity_does_not() {
        let (device, _queue) = match device_and_queue() {
            Ok(pair) => pair,
            Err(_) => return,
        };
        let buffer = device
            .newBufferWithLength_options(8, objc2_metal::MTLResourceOptions::StorageModeShared)
            .expect("allocates an 8-byte plan-uniform-shaped buffer");
        let identity_before = objc2::rc::Retained::as_ptr(&buffer);

        super::write_plan_uniform_bytes(&buffer, &[1, 2, 3, 4, 5, 6, 7, 8]);
        let first_write = super::read_back_uniform_bytes(&buffer, 8);
        super::write_plan_uniform_bytes(&buffer, &[9, 8, 7, 6, 5, 4, 3, 2]);
        let second_write = super::read_back_uniform_bytes(&buffer, 8);
        let identity_after = objc2::rc::Retained::as_ptr(&buffer);

        assert_ne!(
            first_write, second_write,
            "successive writes into the SAME plan-owned uniform buffer must change its contents"
        );
        assert_eq!(
            identity_before, identity_after,
            "the buffer's own identity must never change across writes -- only its bytes do"
        );
    }

    #[test]
    fn live_buffer_count_bounded_in_steady_state() {
        // A ten-stage chain, every stage the SAME extent, so every stage
        // after the first two retires its predecessor and the free list can
        // fully reuse a small, bounded set of slots instead of growing one
        // slot per stage.
        let (program, nodes) = ten_stage_chain(4);
        let a = [1.0f32; 4];
        let outputs = [*nodes.last().expect("ten stages were pushed")];
        let blocks = [QuantizedBlock::Float32(&a)];
        let resolved_plan = plan(&program, &[], &blocks, &outputs, NumericPolicy::default())
            .expect("plans the ten-stage chain");
        arena_placement(&resolved_plan, 0).expect("builds the arena on first placement lookup");

        let slot_count = resolved_plan
            .arena
            .get()
            .expect("arena was just built above")
            .slot_count();
        assert!(
            slot_count <= 3,
            "ten same-size stages must reuse down to a small, bounded slot count via the \
             free list, not grow one slot per stage (got {slot_count})"
        );
    }

    #[test]
    fn two_ops_with_identical_uniform_bytes_get_distinct_plan_owned_buffers() {
        // stage_zero and stage_two are two INDEPENDENT, identically-shaped
        // Identity ops -- `pack_uniforms` packs the same bytes (rank,
        // extents, operand base/strides) for both, since a fresh `Input`'s
        // own operand base is 0 either way. The content-keyed
        // `UNIFORM_BUFFERS` cache would hand both the SAME buffer; plan-
        // owned uniforms must not.
        let (program, [stage_zero, _stage_one, stage_two]) = three_stage_chain(4, 4);
        let a = [1.0f32; 4];
        let b = [2.0f32; 4];
        let outputs = [stage_zero, stage_two];
        let blocks = [QuantizedBlock::Float32(&a), QuantizedBlock::Float32(&b)];
        let resolved_plan = plan(&program, &[], &blocks, &outputs, NumericPolicy::default())
            .expect("plans the identical-uniform chain");
        plan_uniform_buffer(&resolved_plan, 0).expect("builds the uniforms on first lookup");

        let position_of = |node: NodeId| {
            resolved_plan
                .prepared
                .resolved
                .iter()
                .position(|bound| bound.node == node)
                .expect("node is dispatched")
        };
        let stage_zero_bound = &resolved_plan.prepared.resolved[position_of(stage_zero)];
        let stage_two_bound = &resolved_plan.prepared.resolved[position_of(stage_two)];
        assert_eq!(
            super::pack_uniforms(stage_zero_bound, NumericPolicy::default())
                .expect("packs uniforms"),
            super::pack_uniforms(stage_two_bound, NumericPolicy::default())
                .expect("packs uniforms"),
            "degenerate gate: the two ops must genuinely pack identical uniform bytes"
        );

        let uniforms = resolved_plan
            .uniforms
            .get()
            .expect("uniforms were just built above");
        let stage_zero_uniform =
            objc2::rc::Retained::as_ptr(&uniforms.buffers[position_of(stage_zero)]);
        let stage_two_uniform =
            objc2::rc::Retained::as_ptr(&uniforms.buffers[position_of(stage_two)]);
        assert_ne!(
            stage_zero_uniform, stage_two_uniform,
            "two ops with identical uniform bytes must still get DISTINCT plan-owned buffers"
        );
    }
}

/// ROW 327: [`prepare`] attributes each [`QuantizedBlock`] in `blocks` to
/// the node at the same position in [`block_node_ids`]'s output -- this
/// crate's documented positional contract (`execute`'s own doc: "`blocks`
/// binds `Op::Input` inputs positionally"). A caller whose `blocks` order
/// disagrees with its own program's declaration order used to have that
/// mismatch surface as an unrelated `NotLowerable` from
/// `reject_unsupported_gpu_dtype` (whichever node lost its packed
/// classification), because the per-node shape check ran AFTER the packed
/// classification and the dtype gate. These tests pin the fix: the shape
/// check now runs first, so a mismatch reports the exact node and its
/// element-count disagreement directly.
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(super) mod block_node_attribution_tests {
    use alloc::vec;

    use proxima_tensor::{
        DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce,
        ReduceInit, ScalarOp, TensorError, append, projection,
    };

    use super::{MetalError, plan};

    /// `activation -> weight -> product -> sum`: the activation node is
    /// declared FIRST (`NodeId(0)`), the quantized weight node SECOND
    /// (`NodeId(1)`) -- the reverse of the order a caller who lists `blocks`
    /// weight-first (a natural "formula" reading order) would need. Returns
    /// the program and both nodes in DECLARATION order.
    fn activation_first_matmul_program(
        tokens: u32,
        out_dim: u32,
        in_dim: u32,
    ) -> (Vec<Op>, NodeId, NodeId) {
        let mut program = Vec::new();
        let activation = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(tokens), Extent::Static(in_dim)],
                name: None,
            },
        );
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::UInt8,
                shape: vec![Extent::Static(out_dim), Extent::Static(in_dim)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weight, IndexMap::Affine(projection(3, &[1, 2]))),
                    (activation, IndexMap::Affine(projection(3, &[0, 2]))),
                ],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        (program, activation, weight)
    }

    /// The ROW 327 repro: `blocks` lists the weight FIRST even though the
    /// program declares the activation first. The activation's declared
    /// shape (16x16=256 elements) is engineered to equal one Q6_K
    /// super-block's decode count (256), so the misattributed pair
    /// (activation node, weight's Q6_K block) passes the per-node shape
    /// check silently -- exactly the "silently attributed" half of ROW
    /// 327 -- and the mismatch only becomes visible at the SECOND pair
    /// (weight node, activation's Float32 block: declared 8x16=128 elements
    /// vs. the 256 actually handed). Before the fix that second pair was
    /// never reached with a useful diagnostic: `packed_operands_of` and
    /// `reject_unsupported_gpu_dtype` ran first and rejected the weight node
    /// with `NotLowerable` (a dtype complaint, not an ordering one).
    #[test]
    fn weight_first_blocks_against_activation_first_program_names_the_true_node() {
        let (program, _activation, weight) = activation_first_matmul_program(16, 8, 16);
        let activation_data = [0.0f32; 256]; // 16 tokens * 16 in_dim
        let packed_weight = [0u8; 210]; // one Q6_K super-block, decodes to 256 elements

        // MISORDERED: weight's block first, activation's block second --
        // `block_node_ids(&program)` is `[activation, weight]`, so this is
        // the opposite order.
        let blocks = [
            QuantizedBlock::Q6K(&packed_weight),
            QuantizedBlock::Float32(&activation_data),
        ];

        let error = match plan(&program, &[], &blocks, &[], NumericPolicy::default()) {
            Ok(_) => panic!("misordered blocks must never silently plan"),
            Err(error) => error,
        };

        match error {
            MetalError::Tensor(TensorError::InputSizeMismatch {
                node,
                expected,
                found,
            }) => {
                assert_eq!(
                    node, weight,
                    "the weight node -- 8x16=128 declared elements -- must be the node \
                     named, not a downstream node the misattribution happened to also \
                     affect"
                );
                assert_eq!(expected, 128, "weight's own declared element count");
                assert_eq!(
                    found, 256,
                    "the activation's Float32 block landed on the weight node, carrying \
                     the ACTIVATION's element count"
                );
            }
            other => panic!(
                "expected InputSizeMismatch naming node {weight:?} once the shape check \
                 runs before packed classification; got {other:?} instead"
            ),
        }
    }

    /// Same shape as above, correctly ordered blocks (activation first,
    /// matching `block_node_ids`'s `[activation, weight]` declaration
    /// order): must plan cleanly, proving the fix only rejects genuine
    /// mismatches, never a correctly-ordered call.
    #[test]
    fn declaration_ordered_blocks_plan_cleanly() {
        let (program, _activation, _weight) = activation_first_matmul_program(16, 16, 16);
        let activation_data = [0.0f32; 256]; // 16 tokens * 16 in_dim
        let packed_weight = [0u8; 210]; // one Q6_K super-block, decodes to 256 = 16 * 16

        let blocks = [
            QuantizedBlock::Float32(&activation_data),
            QuantizedBlock::Q6K(&packed_weight),
        ];

        plan(&program, &[], &blocks, &[], NumericPolicy::default())
            .expect("declaration-ordered blocks must plan");
    }
}

/// [`HazardTracker`]'s pure dataflow logic, tested with plain `&str`
/// identities so no real Metal device is required -- the real driver path
/// (`execute_plan_with_placements`) instantiates the same type with
/// `Id = *const ProtocolObject<dyn MTLBuffer>`.
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(super) mod hazard_tracker_tests {
    use std::collections::BTreeMap;

    use proxima_tensor::{
        Extent, IndexMap, Keep, NumericPolicy, Op, Reduce, ReduceInit, ScalarOp, append, bind,
        infer, projection,
    };

    use super::{
        Binding, DeviceBuffer, HazardClass, HazardTracker, MetalError, NodeId, PackedOperands,
        hazard_step, kernel_dispatch_shape, resolve_hazard_inputs,
    };
    #[cfg(feature = "instrument")]
    use super::{
        BARRIERS_RAW, BARRIERS_WAR, BARRIERS_WAW, BARRIERS_WAW_WAR_ARENA_RECYCLED,
        BARRIERS_WAW_WAR_PERSISTENT, record_hazard_class,
    };
    use crate::msl::{hazard_read_nodes, hazard_write_node};

    /// `a -> b`, `a -> c` (independent, both only read `a`), then `b, c ->
    /// d` -- the shape this whole feature exists for (Q/K/V from one normed
    /// input, then a later op that needs all three). `b` and `c` share no
    /// hazard with each other (neither reads nor writes the other), so
    /// encoding `c` right after `b` must NOT barrier; `d` reads both `b` and
    /// `c`, both written since the last barrier, so encoding `d` MUST. Driven
    /// through [`hazard_step`] -- the exact function
    /// [`execute_plan_with_placements`]'s own loop calls -- rather than a
    /// hand-rolled mirror of it.
    #[test]
    fn independent_producers_share_no_barrier_but_their_joint_consumer_does() {
        let mut hazards: HazardTracker<&str> = HazardTracker::new();

        // encode `b = f(a)`: `a` is a fresh input, never written -> no hazard.
        assert!(!hazard_step(&mut hazards, &["a"], "b"));

        // encode `c = g(a)`: `a` was only READ (by `b`'s own encode), never
        // WRITTEN, and `c` is a fresh output nothing has touched -> no hazard.
        assert!(!hazard_step(&mut hazards, &["a"], "c"));

        // encode `d = h(b, c)`: both inputs were WRITTEN since the last
        // barrier (by the two steps above) -> a barrier is required.
        assert!(
            hazard_step(&mut hazards, &["b", "c"], "d"),
            "exactly one barrier is required, immediately before encoding d"
        );
    }

    /// `BufferArena` reuses a retired slot's whole buffer for a later
    /// position (`metal-plan-stable-buffers`' own doc) -- `x` writes buffer
    /// `slot0`, `y` (independent of `x`) reads `slot0` as `p`'s arena-chosen
    /// output buffer... modeled here directly: `p` reads buffer `slot0` (a
    /// WAR-eligible read), then a LATER op `q` is placed by the arena into
    /// that SAME `slot0` identity. Writing `q` into a buffer just READ from
    /// is a WAR hazard and must barrier even though `q` shares no operand
    /// with `p`.
    #[test]
    fn arena_slot_reuse_after_a_read_emits_a_war_barrier() {
        let mut hazards: HazardTracker<&str> = HazardTracker::new();

        // encode `p`, which reads `slot0` (some earlier op's live output).
        assert!(!hazard_step(&mut hazards, &["slot0"], "p_out"));

        // encode `q`, whose output the arena has placed into `slot0` itself
        // -- the exact retired-slot-reuse shape `BufferArena::assign_slot`
        // produces. `q` has no operand overlap with `p` at all; the hazard
        // is purely WAR on the output identity.
        assert!(
            hazard_step(&mut hazards, &[], "slot0"),
            "writing into a buffer read since the last barrier must be flagged WAR"
        );
    }

    /// ROW 539's own counters: a synthetic 3-op sequence with exactly one
    /// RAW barrier (`b` reads `a`, which `a`'s own encode just wrote) and one
    /// arena-reuse WAR barrier (a later op's output is placed back into `a`'s
    /// now-read-since-barrier identity, the same shape
    /// [`arena_slot_reuse_after_a_read_emits_a_war_barrier`] drives) --
    /// exactly the scenario [`record_hazard_class`] exists to attribute.
    /// `BARRIERS_RAW`/`BARRIERS_WAR`/etc are process-wide statics; nextest's
    /// per-test process isolation is what keeps a bare `snapshot_and_reset`
    /// deterministic against any other test incrementing the same counters.
    #[cfg(feature = "instrument")]
    #[test]
    fn hazard_class_counters_attribute_one_raw_and_one_arena_reuse_war() {
        let mut hazards: HazardTracker<&str> = HazardTracker::new();

        // op0: writes `a`, no inputs -> no hazard, nothing to attribute.
        let class0 = hazards.classify(&[], Some("a"));
        assert_eq!(class0, HazardClass::None);
        assert!(!hazard_step(&mut hazards, &[], "a"));

        // op1: reads `a` (written by op0 since the last barrier) and writes
        // `b` -> RAW, a genuine dataflow edge, never arena-attributed.
        let class1 = hazards.classify(&["a"], Some("b"));
        assert_eq!(class1, HazardClass::Raw);
        assert!(hazard_step(&mut hazards, &["a"], "b"));
        record_hazard_class(class1, false);

        // op2: no inputs, output placed by the arena back into `a` -- `a`
        // was READ by op1 since the last barrier (op1's own RAW reset the
        // tracker first), so this is a WAR hazard on a recycled arena slot.
        let class2 = hazards.classify(&[], Some("a"));
        assert_eq!(class2, HazardClass::War);
        assert!(hazard_step(&mut hazards, &[], "a"));
        record_hazard_class(class2, true);

        assert_eq!(
            BARRIERS_RAW.snapshot_and_reset(),
            1,
            "op1 contributed the one RAW barrier"
        );
        assert_eq!(
            BARRIERS_WAR.snapshot_and_reset(),
            1,
            "op2 contributed the one WAR barrier"
        );
        assert_eq!(
            BARRIERS_WAW.snapshot_and_reset(),
            0,
            "no WAW hazard in this sequence"
        );
        assert_eq!(
            BARRIERS_WAW_WAR_ARENA_RECYCLED.snapshot_and_reset(),
            1,
            "op2's WAR fired on the arena-recycled `a` slot"
        );
        assert_eq!(
            BARRIERS_WAW_WAR_PERSISTENT.snapshot_and_reset(),
            0,
            "no WAW/WAR fired on a persistent identity in this sequence"
        );
    }

    /// Mirrors [`execute_plan_with_placements`]'s own output-identity
    /// resolution (placement -> arena slot -> fresh allocation) with plain
    /// `usize` identities instead of a real Metal buffer, so a 4-op program
    /// can be driven through [`hazard_step`] end to end with no device.
    fn resolve_test_output(
        node: usize,
        placement_table: &BTreeMap<usize, usize>,
        next_fresh: &mut usize,
    ) -> usize {
        if let Some(identity) = placement_table.get(&node) {
            return *identity;
        }
        let fresh = *next_fresh;
        *next_fresh += 1;
        fresh
    }

    /// The exact production sequence for a 4-op program: op0 is a plain
    /// fresh allocation with no operands, op1 does a RAW read of op0's own
    /// output, op2 is an arena WAR reuse of op0's now-retired identity, and
    /// op3 is a second fresh allocation with no overlap with anything the
    /// tracker still holds (op2's own barrier reset it). Barriers must fire
    /// immediately before op1 (RAW) and op2 (WAR), and nowhere else.
    #[test]
    fn production_sequence_barriers_before_the_raw_and_war_ops_only() {
        let mut hazards: HazardTracker<usize> = HazardTracker::new();
        let mut next_fresh = 100;
        let output0 = resolve_test_output(0, &BTreeMap::new(), &mut next_fresh);
        // op2's own output is arena-placed into op0's retired identity --
        // `arena_slot_reuse_after_a_read_emits_a_war_barrier` covers that
        // shape in isolation; here it sits inside a longer program alongside
        // a RAW hazard (op1) and a genuinely fresh allocation (op3).
        let placement_table = BTreeMap::from([(2, output0)]);

        let barrier0 = hazard_step(&mut hazards, &[], output0);

        let output1 = resolve_test_output(1, &placement_table, &mut next_fresh);
        let barrier1 = hazard_step(&mut hazards, &[output0], output1);

        let output2 = resolve_test_output(2, &placement_table, &mut next_fresh);
        assert_eq!(
            output2, output0,
            "degenerate gate: op2 must reuse op0's own identity"
        );
        let barrier2 = hazard_step(&mut hazards, &[], output2);

        let output3 = resolve_test_output(3, &placement_table, &mut next_fresh);
        let barrier3 = hazard_step(&mut hazards, &[], output3);

        assert_eq!(
            [barrier0, barrier1, barrier2, barrier3],
            [false, true, true, false],
            "barriers fire before op1 (RAW) and op2 (WAR), never before op0 or op3"
        );
    }

    /// Redesign §4c's own gap: `Binding::Scratch` is written by a
    /// `CachedAttention` split kernel and read by its merge kernel, an edge
    /// `encode_op`'s own doc names as internal to one op. Modeled here at
    /// the pure `HazardTracker` level (`encode_op`'s real fix threads the
    /// SAME `scratch`/`output` identities through this exact tracker): the
    /// split's write to `scratch`, a SIBLING op's write to an entirely
    /// unrelated buffer in between (the concurrent-dispatch case this
    /// tracker exists for -- two independent ops both queued before either
    /// finishes), then the merge's read of `scratch`. The sibling's own
    /// write must not need a barrier (it shares no identity with anything
    /// written or read so far) and must not erase `scratch`'s own
    /// written-since-last-barrier status (`record` only adds, `reset` is the
    /// only thing that clears, and the sibling never triggers one) -- so the
    /// merge's read still sees `scratch` as written and still barriers.
    #[test]
    fn a_sibling_op_between_the_split_write_and_the_merge_read_does_not_hide_the_scratch_raw_hazard()
     {
        let mut hazards: HazardTracker<&str> = HazardTracker::new();

        // the split kernel writes its partial into scratch.
        assert!(
            !hazard_step(&mut hazards, &[], "scratch"),
            "scratch is a fresh identity nothing has touched yet"
        );

        // an independent sibling op, queued in between under
        // `DispatchType::Concurrent`, touches a buffer with no relation to
        // `scratch` at all.
        assert!(
            !hazard_step(&mut hazards, &[], "sibling_out"),
            "an unrelated sibling write must not spuriously barrier"
        );

        // the merge kernel reads scratch back -- the RAW hazard the sibling
        // must not have hidden.
        assert!(
            hazard_step(&mut hazards, &["scratch"], "real_output"),
            "the merge's read of scratch must still see it as written, sibling or not"
        );
    }

    /// An operand missing from `device_buffers` is a driver bug -- the
    /// encode it feeds is already wrong -- so [`resolve_hazard_inputs`] must
    /// error, never silently drop it from the hazard set.
    #[test]
    fn a_missing_operand_buffer_is_an_error_not_a_dropped_hazard() {
        let device_buffers: BTreeMap<NodeId, DeviceBuffer> = BTreeMap::new();
        let missing = NodeId(7);

        let result = resolve_hazard_inputs(core::iter::once(missing), &device_buffers);

        match result {
            Err(MetalError::UnresolvedHazardOperand { node }) => assert_eq!(node, missing),
            other => panic!("expected UnresolvedHazardOperand, got {other:?}"),
        }
    }

    /// Logical retirement removes a node from `device_buffers`, but the
    /// arena-owned MTLBuffer remains live until this command buffer commits.
    /// Reusing that identity must therefore preserve the prior read/write
    /// history and emit the same WAR/WAW barrier as the production loop.
    #[test]
    fn logical_retirement_preserves_arena_identity_hazards() {
        let mut hazards: HazardTracker<&str> = HazardTracker::new();
        assert!(!hazard_step(&mut hazards, &["input"], "slot0"));
        // The production retirement loop removes the logical node only; it
        // intentionally does not mutate `hazards` before a same-slot reuse.
        assert!(hazard_step(&mut hazards, &[], "slot0"));
        assert!(!hazard_step(&mut hazards, &[], "slot1"));
        // A second write after the barrier is independent and starts clean.
        assert!(
            !hazards.needs_barrier(&[], Some("input")),
            "the emitted barrier must clear the retired command's old state"
        );
    }

    /// `raw -> y` (a dispatched `Identity`), then `weights -> reduced`, fused
    /// with `Add(reduced, y)` into ONE `BoundOpKind::Reduce` whose
    /// `epilogue_operands` reads `y` -- the exact decode-shaped census ROW
    /// 323 named: a fused reduce's epilogue reads a SIBLING's just-written
    /// output, not one of its own compute-step `operands()`. `extra_y_use`
    /// (a second, independent consumer of `y`) keeps `y` from being inlined
    /// away by ordinary elementwise fusion before the reduce-epilogue fold
    /// ever runs. Returns the plan's dispatched `BoundOp`s in plan order,
    /// `y`'s own node, the fused reduce's own node (the surviving `consumer`
    /// NodeId), and `PackedOperands::new()` (no packed/quantized operand
    /// here, so an empty table is exactly right -- same as [`emit`]'s own
    /// doctest).
    fn epilogue_reads_sibling_output_fixture()
    -> (Vec<super::BoundOp>, NodeId, NodeId, PackedOperands) {
        let mut program = Vec::new();
        let raw = append(
            &mut program,
            Op::Input {
                dtype: super::DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let identity = || IndexMap::Affine(projection(1, &[0]));
        let y = append(
            &mut program,
            Op::Elementwise {
                dtype: super::DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(raw, identity())],
                name: None,
            },
        );
        let weights = append(
            &mut program,
            Op::Input {
                dtype: super::DType::Float32,
                shape: alloc::vec![Extent::Static(8), Extent::Static(4)],
                name: None,
            },
        );
        let reduced = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: super::DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: weights,
                in_map: IndexMap::Affine(projection(2, &[0, 1])),
                out_map: IndexMap::Affine(projection(2, &[1])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        let consumer = append(
            &mut program,
            Op::Elementwise {
                dtype: super::DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![(reduced, identity()), (y, identity())],
                name: None,
            },
        );
        // a second, independent consumer of `y` -- without it, elementwise
        // fusion inlines `y`'s trivial `Identity` body straight into
        // `consumer`'s own composed body before `reduce-epilogue-fusion` ever
        // runs, collapsing the whole program to one op and defeating the
        // fixture's own point (a SIBLING dispatch's buffer read via the
        // epilogue). A second use forces `y` to materialize as its own
        // dispatched node, same trick `reduce_then_residual_add_program`
        // (`proxima-tensor/src/bind.rs`) uses for `x`.
        let extra_y_use = append(
            &mut program,
            Op::Elementwise {
                dtype: super::DType::Float32,
                body: ScalarOp::Negate,
                operands: alloc::vec![(y, identity())],
                name: None,
            },
        );
        let shapes = infer(&program, &[]).expect("epilogue fixture infers");
        let resolved = bind(
            &program,
            &shapes,
            &[consumer, extra_y_use],
            NumericPolicy::default(),
        )
        .expect("epilogue fixture binds");
        assert_eq!(
            resolved.len(),
            3,
            "degenerate gate: `y`, `extra_y_use`, and the fused reduce must be the only \
             dispatched ops -- `reduce-epilogue-fusion` folded `consumer` into `reduced`'s own \
             epilogue, and `extra_y_use` keeps `y` from being inlined away entirely"
        );
        let fused = resolved
            .iter()
            .find(|bound| bound.node == consumer)
            .expect("the consumer's NodeId now names the fused reduce+epilogue op");
        assert!(
            matches!(
                fused.kind,
                super::BoundOpKind::Reduce { ref epilogue_operands, .. }
                    if epilogue_operands.iter().any(|(node, _, _)| *node == y)
            ),
            "degenerate gate: the fused reduce's epilogue must read `y`, not just `weights`, got {:?}",
            fused.kind
        );
        (resolved, y, consumer, PackedOperands::new())
    }

    /// The exact case the deleted per-op forward scan's own comment worried
    /// about: a fused reduce's epilogue reads a SIBLING op's output (`y`), not
    /// one of its own `operands()` -- so `y`'s true last reader is the fused
    /// op's position, not `y`'s own dispatch position. `execute_plan_inner`'s
    /// retirement loop now trusts `prepared.last_reader[y] == position`
    /// alone; this proves that table already carries the fused read, so the
    /// O(1) lookup cannot retire `y` early.
    #[test]
    fn last_reader_table_protects_a_fused_epilogues_sibling_read() {
        let (resolved, y_node, fused_node, _packed_operands) =
            epilogue_reads_sibling_output_fixture();
        let node_count = resolved
            .iter()
            .map(|bound| bound.node.0)
            .chain(core::iter::once(y_node.0))
            .max()
            .expect("fixture has at least one node")
            + 1;
        let last_reader = super::node_last_reader(&resolved, node_count as usize);
        let fused_position = resolved
            .iter()
            .position(|bound| bound.node == fused_node)
            .expect("fused reduce is dispatched in this program");
        let y_own_position = resolved
            .iter()
            .position(|bound| bound.node == y_node)
            .expect("y is dispatched in this program");

        assert!(
            (last_reader[y_node.0 as usize] as usize) >= fused_position,
            "y's last reader must be at or after the fused epilogue's read of it, got {}",
            last_reader[y_node.0 as usize]
        );
        assert_ne!(
            last_reader[y_node.0 as usize] as usize, y_own_position,
            "y must not be retired at its own dispatch position -- the fused epilogue's read of \
             it comes later"
        );
    }

    /// `bindings()`'s `Binding::Input` entries come 1:1 from
    /// `BoundOp::all_read_sources()` (`msl::bindings`'s own doc), so deriving
    /// the hazard read set from `bindings` instead of a parallel
    /// `all_read_sources()` enumeration cannot change which barriers fire for
    /// a program that already has no missing binding -- proven here by
    /// running the tracker over the census fixture's own `bindings`, in plan
    /// order: `y` writes fresh (no prior hazard, no barrier); the fused
    /// reduce reads `y` via its epilogue, and `y` was written since the last
    /// barrier, so a barrier is required immediately before it -- exactly
    /// the RAW ROW 323 named; `extra_y_use`'s own read of `y` then sees a
    /// tracker the fused reduce's own barrier already reset, so it does not
    /// barrier again.
    #[test]
    fn hazard_walk_over_bindings_matches_the_decode_shaped_fixtures_expected_trace() {
        let (resolved, y_node, fused_node, packed_operands) =
            epilogue_reads_sibling_output_fixture();
        let mut hazards: HazardTracker<NodeId> = HazardTracker::new();

        let mut barrier_by_node = std::collections::BTreeMap::new();
        for bound in &resolved {
            let (bindings, _grid) =
                kernel_dispatch_shape(bound, &packed_operands, NumericPolicy::default())
                    .expect("fixture ops emit a shape");
            let reads: Vec<NodeId> = hazard_read_nodes(&bindings).collect();
            let write = hazard_write_node(&bindings).expect("every bound op writes one node");
            barrier_by_node.insert(bound.node, hazard_step(&mut hazards, &reads, write));
        }

        assert!(
            !barrier_by_node[&y_node],
            "y's own dispatch reads only a plain Input, never tracked as written -- no barrier"
        );
        assert!(
            barrier_by_node[&fused_node],
            "the fused reduce reads y (via its epilogue binding) after y was just written -- \
             this is the exact RAW ROW 323 named, now caught because the read set is derived \
             from the same bindings the encoder binds"
        );
    }

    /// The class fix, isolated from any real `BoundOp`/fusion machinery: a
    /// dispatch's `bindings` list gains a read a hand-written `operands()`-only
    /// enumeration would never have seen -- exactly the shape a future fusion
    /// rule (or any new `Binding` producer) could add. Deriving the hazard
    /// read set from `bindings` itself, rather than from a second, parallel
    /// operand table, means that read is barriered by CONSTRUCTION: there is
    /// no second call site left to forget to update.
    #[test]
    fn a_binding_list_gaining_a_read_is_barriered_with_no_second_call_site_to_update() {
        let mut hazards: HazardTracker<NodeId> = HazardTracker::new();
        let unrelated_input = NodeId(0);
        let producer_output = NodeId(1);

        // op0: writes `producer_output`, reading only its own unrelated input.
        let bindings0 = [
            Binding::Input(unrelated_input),
            Binding::Output(producer_output),
        ];
        let barrier0 = hazard_step(
            &mut hazards,
            &hazard_read_nodes(&bindings0).collect::<Vec<_>>(),
            hazard_write_node(&bindings0).expect("bindings0 names one output"),
        );
        assert!(
            !barrier0,
            "a fresh write with no prior hazard history never barriers"
        );

        // op1's own compute-step operand is `op1_own_operand`; its `bindings`
        // gain an EXTRA read of `producer_output` -- the slot a fused
        // epilogue (or any future `Binding` producer) would occupy, never
        // named by a bare `operands()` walk.
        let op1_own_operand = NodeId(2);
        let op1_output = NodeId(3);
        let bindings1 = [
            Binding::Input(op1_own_operand),
            Binding::Input(producer_output),
            Binding::Output(op1_output),
        ];
        let barrier1 = hazard_step(
            &mut hazards,
            &hazard_read_nodes(&bindings1).collect::<Vec<_>>(),
            hazard_write_node(&bindings1).expect("bindings1 names one output"),
        );

        assert!(
            barrier1,
            "a RAW hazard against a buffer just written must barrier even when the read arrived \
             through a binding slot no separate operand table names"
        );
    }

    /// The horizontal-packed-merge shape: 8 routed-expert gathers sharing
    /// one weight-stack NodeId and one activation NodeId, each with its own
    /// distinct output -- exactly what a merged `grid.z = 8` dispatch would
    /// hazard-record as one shared `inputs` slice against 8 outputs
    /// (`docs/discipline.md`'s design note §4). Recording all 8 writes
    /// against the SAME shared-inputs snapshot must leave every one of them
    /// visible to a later reader without a second `record` call widening
    /// [`HazardTracker::record`]'s own single-`Option<Id>` signature.
    #[test]
    fn merged_dispatch_records_all_eight_outputs_against_one_shared_input_set() {
        let mut hazards: HazardTracker<&str> = HazardTracker::new();
        let shared_inputs = ["weight_stack", "activation"];
        let outputs: Vec<String> = (0..8).map(|round| format!("round{round}_out")).collect();

        assert!(
            !hazards.needs_barrier(&shared_inputs, None),
            "the weight stack and activation are fresh reads, nothing has written them yet"
        );
        hazards.record(&shared_inputs, None);
        for output in &outputs {
            hazards.written.insert(output.as_str());
        }

        for output in &outputs {
            assert!(
                hazards.written.contains(output.as_str()),
                "every merged member's own output must be a recorded hazard write, not just the \
                 group's first member"
            );
        }

        // a later op reading round7's output sees a RAW hazard exactly as it
        // would against 8 separate dispatches -- the merge changes how many
        // GPU dispatches ran, never what a later reader observes.
        assert_eq!(
            hazards.classify(&[outputs[7].as_str()], None),
            HazardClass::Raw,
            "a later reader of the LAST merged member's output must still see a RAW hazard"
        );
    }
}

/// [`group_mergeable_positions`]'s own fixtures -- pure over `NodeId`
/// reads/writes and an opaque identity key, so these run without a device.
/// See that function's doc for the shape (`append_moe_round_output`'s 8
/// routed-expert gathers) this exists to recognize.
#[cfg(all(test, feature = "metal-horizontal-merge"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(super) mod horizontal_merge_grouping_tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::{NodeId, group_mergeable_positions};

    /// 8 independent same-identity packed reduces -- one shared weight-stack
    /// NodeId, one shared activation NodeId, a distinct per-round gather
    /// index and output -- must resolve to exactly one group of all 8.
    #[test]
    fn eight_independent_identical_identity_reduces_form_one_group_of_eight() {
        let identities = [0usize; 8];
        let weight_stack = NodeId(100);
        let activation = NodeId(101);
        let reads: Vec<Vec<NodeId>> = (0..8)
            .map(|round| vec![weight_stack, activation, NodeId(200 + round)])
            .collect();
        let writes: Vec<NodeId> = (0..8).map(|round| NodeId(300 + round)).collect();

        let groups = group_mergeable_positions(&identities, &reads, &writes);

        assert_eq!(groups.len(), 1, "all 8 independent positions form one group");
        assert_eq!(groups[0].len(), 8, "the one group must contain every position");
    }

    /// Position 5 reads position 2's output -- a genuine dataflow edge
    /// between two same-identity positions. The edge must keep position 5
    /// out of the merge group entirely (it still runs as its own individual
    /// dispatch, unchanged from today) rather than either silently merging
    /// it in (a data race: its read could observe a stale, not-yet-written
    /// slice from the SAME dispatch) or refusing to merge anyone else in
    /// the bucket.
    #[test]
    fn a_raw_edge_between_two_members_excludes_the_reader_from_the_group() {
        let identities = [0usize; 8];
        let weight_stack = NodeId(100);
        let mut reads: Vec<Vec<NodeId>> = (0..8)
            .map(|round| vec![weight_stack, NodeId(200 + round)])
            .collect();
        let writes: Vec<NodeId> = (0..8).map(|round| NodeId(300 + round)).collect();
        // position 5 also reads position 2's own output -- a RAW edge.
        reads[5].push(writes[2]);

        let groups = group_mergeable_positions(&identities, &reads, &writes);

        assert_eq!(groups.len(), 1, "the other 7 independent positions still merge into one group");
        assert_eq!(
            groups[0],
            vec![0, 1, 2, 3, 4, 6, 7],
            "position 5 -- the reader on the RAW edge -- is excluded from the merge group"
        );
        assert!(
            !groups[0].contains(&5),
            "position 5 must fall back to its own individual dispatch, never join the group"
        );
    }

    /// Two different `kernel_identity`s never merge, regardless of
    /// independence -- the predicate's first, cheapest gate.
    #[test]
    fn different_identities_never_merge_even_when_independent() {
        let identities = [0usize, 1usize];
        let reads: Vec<Vec<NodeId>> = vec![vec![NodeId(1)], vec![NodeId(2)]];
        let writes = [NodeId(10), NodeId(11)];

        let groups = group_mergeable_positions(&identities, &reads, &writes);

        assert!(
            groups.is_empty(),
            "two singleton identity buckets carry nothing to merge"
        );
    }
}

/// Production crash (2026-09-07): `-[_MTLCommandEncoder dealloc]: failed
/// assertion 'Command encoder released without endEncoding'`, caught here at
/// its true root instead of the assertion it manifested as. A pure-CPU
/// module -- [`cached_attention_scratch_len`] never touches a device -- so
/// these run on any host, without `metal-plan-stable-buffers` or a GPU.
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(super) mod attention_scratch_len_tests {
    use alloc::vec;

    use proxima_tensor::{BoundOpKind, DType, Layout, NodeId, NumericPolicy};

    use super::{BoundOp, cached_attention_scratch_len};

    /// The real openchat decode shape's own dims (`omega/tests/
    /// row_376_cached_attention_batched.rs`'s `QUERY_HEADS`/`KV_HEADS`/
    /// `HEAD_DIM`), at `query_rows` and `cached_key_rows + new_key_rows` set
    /// by the caller -- one dispatch's worth of `CachedAttention`.
    fn openchat_shaped_bound(query_rows: u64, cached_key_rows: u64, new_key_rows: u64) -> BoundOp {
        const KV_HEADS: u64 = 8;
        const QUERY_GROUPS: u64 = 4;
        const HEAD_DIM: u64 = 128;
        let operands = (0..8)
            .map(|index| {
                (
                    NodeId(index),
                    Layout {
                        base: 0,
                        strides: vec![1].into(),
                    },
                    None,
                )
            })
            .collect();
        BoundOp {
            node: NodeId(8),
            dtype: DType::Float32,
            extents: vec![query_rows, KV_HEADS, QUERY_GROUPS, HEAD_DIM],
            kind: BoundOpKind::CachedAttention {
                operands,
                query_rows,
                cached_key_rows,
                new_key_rows,
                kv_heads: KV_HEADS,
                query_groups: QUERY_GROUPS,
                head_dim: HEAD_DIM,
                rotary_dim: HEAD_DIM,
                scale: 0.5,
                cached_lower_inclusive: i64::MIN,
                new_upper_inclusive: 0,
            },
        }
    }

    /// ROW: a decode dispatch (`query_rows == 1`) still reserves the
    /// COMPILED MAXIMUM split count, unchanged from before this fix -- a
    /// live decode's own context length grows one key per token, all the
    /// way to `ATTENTION_SPLIT_MAX` splits, and this buffer must never need
    /// to resize mid-stream to keep up with it.
    #[test]
    fn decode_dispatch_still_reserves_the_compiled_split_maximum() {
        let bound = openchat_shaped_bound(1, 0, 4096);
        let elements = cached_attention_scratch_len(&bound, NumericPolicy::llama_relaxed())
            .expect("a CachedAttention bound always yields a scratch length");
        let expected_heads = 8 * 4;
        let expected = expected_heads * crate::sized::ATTENTION_SPLIT_MAX * (2 + 128);
        assert_eq!(
            elements, expected,
            "decode (query_rows == 1) must keep reserving ATTENTION_SPLIT_MAX splits"
        );
    }

    /// ROW: the production crash. Before this fix, a 900-row prefill at
    /// this exact shape requested `900 * 32 * 32 * 130 * 4` bytes (~479 MB)
    /// of scratch for ONE `CachedAttention` op -- ~17 GB across a 36-layer
    /// forward, which no device honors. This test uses 1024 rows (the
    /// owner's own re-prove shape) and asserts the byte length actually
    /// requested now, plus the exact 4x reduction the real split count
    /// (`splits_for(1024, ..) == 8`, vs. the compiled max `32`) predicts --
    /// tying the byte count to the mechanism, not just a smaller number.
    #[test]
    fn prefill_dispatch_sizes_scratch_by_the_real_split_count_not_the_compiled_max() {
        let context_length = 1024;
        let bound = openchat_shaped_bound(1024, 0, context_length);
        let policy = NumericPolicy::llama_relaxed();

        let after_elements = cached_attention_scratch_len(&bound, policy)
            .expect("a CachedAttention bound always yields a scratch length");
        let after_bytes = after_elements * 4;

        let real_splits = crate::msl::splits_for(context_length, policy);
        let total_elements = 1024 * 8 * 4;
        let expected_after = total_elements * real_splits * (2 + 128);
        let before_elements = total_elements * crate::sized::ATTENTION_SPLIT_MAX * (2 + 128);
        let before_bytes = before_elements * 4;

        std::println!(
            "cached_attention_scratch_len: context_length={context_length} \
             real_splits={real_splits} compiled_max={} \
             before_bytes={before_bytes} after_bytes={after_bytes} \
             reduction={:.1}x",
            crate::sized::ATTENTION_SPLIT_MAX,
            before_bytes as f64 / after_bytes as f64,
        );

        assert_eq!(
            after_elements, expected_after,
            "prefill (query_rows > 1) must size scratch off the REAL split count"
        );
        assert!(
            real_splits < crate::sized::ATTENTION_SPLIT_MAX,
            "this shape is only a meaningful regression check if the real split count is \
             actually smaller than the compiled max -- otherwise the fix and the pre-fix \
             formula agree by coincidence, not by the mechanism this test asserts"
        );
        assert!(
            after_bytes < before_bytes,
            "the fix must always request fewer bytes than the pre-fix always-max formula: \
             before={before_bytes} after={after_bytes}"
        );
    }
}

/// Production crash (2026-09-07): `provider error: generate: arena
/// peak_bytes=984110552 exceeds arena_transient_cap=172812125` -- a
/// ~1100-row interactive-chat prefill rejected by a cap sized once for
/// decode (`query_rows == 1`) and never scaled for a prefill's own row
/// count. Pure CPU -- [`plan_query_rows`] never touches a device -- so
/// these run on any host, without a GPU.
#[cfg(all(test, feature = "metal-plan-stable-buffers"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(super) mod plan_query_rows_tests {
    use alloc::vec;

    use proxima_tensor::{BoundOpKind, DType, Layout, NodeId};

    use super::{ARENA_TRANSIENT_CAP, BoundOp, plan_query_rows};

    /// Same real openchat decode/prefill shape `attention_scratch_len_tests`'
    /// own `openchat_shaped_bound` builds -- one `CachedAttention` dispatch
    /// at the caller's own `query_rows`.
    fn cached_attention_bound(query_rows: u64) -> BoundOp {
        const KV_HEADS: u64 = 8;
        const QUERY_GROUPS: u64 = 4;
        const HEAD_DIM: u64 = 128;
        let operands = (0..8)
            .map(|index| {
                (
                    NodeId(index),
                    Layout {
                        base: 0,
                        strides: vec![1].into(),
                    },
                    None,
                )
            })
            .collect();
        BoundOp {
            node: NodeId(8),
            dtype: DType::Float32,
            extents: vec![query_rows, KV_HEADS, QUERY_GROUPS, HEAD_DIM],
            kind: BoundOpKind::CachedAttention {
                operands,
                query_rows,
                cached_key_rows: 0,
                new_key_rows: query_rows,
                kv_heads: KV_HEADS,
                query_groups: QUERY_GROUPS,
                head_dim: HEAD_DIM,
                rotary_dim: HEAD_DIM,
                scale: 0.5,
                cached_lower_inclusive: i64::MIN,
                new_upper_inclusive: 0,
            },
        }
    }

    /// A plan with no `CachedAttention` op at all (a pure elementwise/matmul
    /// program) has no row count to read off an op, so [`plan_query_rows`]
    /// falls back to `1` -- the cap this plan sees is the unscaled
    /// `ARENA_TRANSIENT_CAP`, exactly today's behavior.
    #[test]
    fn no_attention_op_defaults_to_one_row() {
        let elementwise = BoundOp {
            node: NodeId(0),
            dtype: DType::Float32,
            extents: vec![4096],
            kind: BoundOpKind::Elementwise {
                body: proxima_tensor::ComposedBody::leaf(proxima_tensor::ScalarOp::Identity),
                operands: vec![(
                    NodeId(1),
                    Layout {
                        base: 0,
                        strides: vec![1].into(),
                    },
                    None,
                )],
            },
        };
        assert_eq!(plan_query_rows(&[elementwise]), 1);
    }

    /// `query_rows` in {1 (decode), 31 (a short prefix), 1100 (ROW's own
    /// interactive-chat reproduction)} -- [`plan_query_rows`] reads the ONE
    /// `CachedAttention` op's own field back unchanged, and the derived cap
    /// scales linearly with it (decode's `query_rows == 1` reproduces the
    /// pre-fix constant exactly).
    #[test]
    fn cap_scales_linearly_with_the_plans_own_query_rows() {
        for query_rows in [1_u64, 31, 1100] {
            let resolved = [cached_attention_bound(query_rows)];
            let observed = plan_query_rows(&resolved);
            assert_eq!(
                observed, query_rows,
                "plan_query_rows must read the plan's own CachedAttention row count back exactly"
            );

            let cap_bytes = ARENA_TRANSIENT_CAP.saturating_mul(observed.max(1) as usize);
            let expected = ARENA_TRANSIENT_CAP * query_rows as usize;
            assert_eq!(
                cap_bytes, expected,
                "cap_bytes must be exactly ARENA_TRANSIENT_CAP * query_rows at query_rows={query_rows}"
            );
        }
    }

    /// ROW: the production shape itself -- at `query_rows=1100` the naive
    /// 984 MB peak from the incident report is now well inside the
    /// per-row-scaled cap, where the old fixed `ARENA_TRANSIENT_CAP`
    /// (172_812_125 bytes) rejected it outright.
    #[test]
    fn thousand_row_prefill_shape_fits_the_scaled_cap() {
        let query_rows = 1100_u64;
        let cap_bytes = ARENA_TRANSIENT_CAP.saturating_mul(query_rows as usize);
        let production_peak_bytes = 984_110_552_usize;
        assert!(
            production_peak_bytes < cap_bytes,
            "the ROW 391 incident's own peak_bytes must fit under the scaled cap: \
             peak_bytes={production_peak_bytes} cap_bytes={cap_bytes}"
        );
        assert!(
            production_peak_bytes > ARENA_TRANSIENT_CAP,
            "this reproduction is only meaningful if the unscaled constant alone \
             would have rejected it, matching the original incident"
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(super) mod expert_payload_descriptor_tests {
    use super::{
        BoundOp, BoundOpKind, ExpertPayloadDescriptor, MetalError, PackedCodec,
        expert_payload_descriptors, live_block_inputs, ordinary_block_uploads,
        pack_expert_payload_descriptors, reject_non_reducing_expert_staging,
        selected_expert_arena_descriptors, selected_expert_payloads,
    };
    use proxima_tensor::cpu::{ExpertEntry, ExpertPayloadSpan, ExpertSource, QuantizedBlock};
    use proxima_tensor::{ComposedBody, DType, Layout, Lookup, NodeId, ScalarOp};

    #[test]
    fn live_input_mask_filters_dead_expert_blocks_and_keeps_requested_roots() {
        let resolved = [BoundOp {
            node: NodeId(43),
            dtype: DType::Float32,
            extents: vec![16],
            kind: BoundOpKind::Elementwise {
                body: ComposedBody::leaf(ScalarOp::Identity),
                operands: vec![(
                    NodeId(42),
                    Layout {
                        base: 0,
                        strides: vec![1].into(),
                    },
                    Some(Lookup {
                        indices: NodeId(41),
                        index_layout: Layout {
                            base: 0,
                            strides: vec![1].into(),
                        },
                        element_stride: 1,
                        extent: 16,
                    }),
                )],
            },
        }];

        assert_eq!(
            live_block_inputs(&[NodeId(40), NodeId(41), NodeId(42)], &resolved, &[]),
            vec![false, true, true],
            "a dead packed expert root must not enter the upload set"
        );
        assert_eq!(
            live_block_inputs(
                &[NodeId(40), NodeId(41), NodeId(42)],
                &resolved,
                &[NodeId(40)]
            ),
            vec![true, true, true],
            "an explicitly requested root remains live even without a bound consumer"
        );
    }

    #[test]
    fn mixed_hobbit_entries_keep_codec_and_byte_spans_separate() {
        let low_bytes = [0_u8; 84];
        let high_bytes = [0_u8; 144];
        let entries = [
            ExpertEntry {
                block: QuantizedBlock::Q2K(&low_bytes),
                out_dim: 256,
                in_dim: 256,
                epoch: 11,
            },
            ExpertEntry {
                block: QuantizedBlock::Q4K(&high_bytes),
                out_dim: 256,
                in_dim: 256,
                epoch: 12,
            },
        ];
        let source = ExpertSource::new(&entries);

        let descriptors = expert_payload_descriptors(NodeId(7), &source)
            .expect("packed HOBBIT entries produce descriptors");

        assert_eq!(
            descriptors,
            vec![
                ExpertPayloadDescriptor {
                    expert_index: 0,
                    codec: PackedCodec::Q2K,
                    byte_offset: 0,
                    byte_length: 84,
                    out_dim: 256,
                    in_dim: 256,
                    epoch: 11,
                },
                ExpertPayloadDescriptor {
                    expert_index: 1,
                    codec: PackedCodec::Q4K,
                    byte_offset: 84,
                    byte_length: 144,
                    out_dim: 256,
                    in_dim: 256,
                    epoch: 12,
                },
            ]
        );
        let packed = pack_expert_payload_descriptors(NodeId(7), &descriptors)
            .expect("Q2_K/Q4_K descriptors fit the Metal ABI");
        assert_eq!(&packed[0..4], &0u32.to_ne_bytes());
        assert_eq!(&packed[4..8], &1u32.to_ne_bytes());
        assert_eq!(&packed[32..36], &1u32.to_ne_bytes());
        assert_eq!(&packed[36..40], &2u32.to_ne_bytes());
    }

    #[test]
    fn selected_hobbit_entries_compact_payload_but_keep_original_ids() {
        let low_bytes = [1_u8; 84];
        let high_bytes = [2_u8; 144];
        let entries = [
            ExpertEntry {
                block: QuantizedBlock::Q2K(&low_bytes),
                out_dim: 256,
                in_dim: 256,
                epoch: 1,
            },
            ExpertEntry {
                block: QuantizedBlock::Q4K(&high_bytes),
                out_dim: 256,
                in_dim: 256,
                epoch: 2,
            },
        ];
        let selected = [1_u32];
        let source = ExpertSource::with_selected_expert_ids(&entries, &selected);
        let (payload, descriptors) = selected_expert_payloads(NodeId(9), &source)
            .expect("selected packed entries produce a compact table");
        assert_eq!(payload.len(), high_bytes.len());
        assert_eq!(descriptors[0].byte_length, 0);
        assert_eq!(descriptors[1].expert_index, 1);
        assert_eq!(descriptors[1].byte_length, high_bytes.len());
    }

    #[test]
    fn selected_hobbit_q6k_entry_keeps_codec_and_descriptor_tag() {
        let low_bytes = [1_u8; 144];
        let high_bytes = [2_u8; 210];
        let entries = [
            ExpertEntry {
                block: QuantizedBlock::Q4K(&low_bytes),
                out_dim: 256,
                in_dim: 256,
                epoch: 1,
            },
            ExpertEntry {
                block: QuantizedBlock::Q6K(&high_bytes),
                out_dim: 256,
                in_dim: 256,
                epoch: 2,
            },
        ];
        let selected = [1_u32];
        let source = ExpertSource::with_selected_expert_ids(&entries, &selected);
        let (payload, descriptors) = selected_expert_payloads(NodeId(10), &source)
            .expect("selected Q6_K expert produces a compact table");

        assert_eq!(payload, high_bytes);
        assert_eq!(descriptors[1].codec, PackedCodec::Q6K);
        assert_eq!(descriptors[1].byte_length, high_bytes.len());
        let packed = pack_expert_payload_descriptors(NodeId(10), &descriptors)
            .expect("Q6_K descriptor fits the Metal ABI");
        assert_eq!(&packed[36..40], &3u32.to_ne_bytes());
    }

    #[test]
    fn selected_hobbit_descriptor_offsets_follow_compact_payload_order() {
        let first = [1_u8; 144];
        let second = [2_u8; 210];
        let entries = [
            ExpertEntry {
                block: QuantizedBlock::Q4K(&first),
                out_dim: 256,
                in_dim: 256,
                epoch: 1,
            },
            ExpertEntry {
                block: QuantizedBlock::Q6K(&second),
                out_dim: 256,
                in_dim: 256,
                epoch: 2,
            },
        ];
        let selected = [0_u32, 1_u32];
        let source = ExpertSource::with_selected_expert_ids(&entries, &selected);
        let (payload, descriptors) = selected_expert_payloads(NodeId(11), &source)
            .expect("selected mixed entries produce a compact payload");

        assert_eq!(payload.len(), first.len() + second.len());
        assert_eq!(descriptors[0].byte_offset, 0);
        assert_eq!(descriptors[1].byte_offset, first.len());
        assert_eq!(&payload[..first.len()], &first);
        assert_eq!(&payload[first.len()..], &second);
    }

    #[test]
    fn all_expert_arena_descriptors_keep_mmap_relative_offsets() {
        let arena = [0_u8; 256];
        let entries = [
            ExpertEntry {
                block: QuantizedBlock::Q2K(&arena[7..91]),
                out_dim: 256,
                in_dim: 256,
                epoch: 3,
            },
            ExpertEntry {
                block: QuantizedBlock::Q2K(&arena[139..223]),
                out_dim: 256,
                in_dim: 256,
                epoch: 4,
            },
        ];
        let spans = [
            Some(ExpertPayloadSpan {
                offset: 7,
                length: 84,
            }),
            Some(ExpertPayloadSpan {
                offset: 139,
                length: 84,
            }),
        ];
        let source = ExpertSource::with_all_expert_arena(&entries, &arena, &spans)
            .expect("the dense packed arena validates");

        let descriptors = selected_expert_arena_descriptors(
            NodeId(13),
            &source,
            source.packed_arena().expect("the source carries its arena"),
        )
        .expect("all expert spans lower without a selected route list");

        assert_eq!(descriptors.len(), 2);
        assert_eq!(descriptors[0].expert_index, 0);
        assert_eq!(descriptors[0].byte_offset, 7);
        assert_eq!(descriptors[0].byte_length, 84);
        assert_eq!(descriptors[1].expert_index, 1);
        assert_eq!(descriptors[1].byte_offset, 139);
        assert_eq!(descriptors[1].byte_length, 84);
    }

    #[test]
    fn selected_hobbit_preserves_sparse_original_expert_indices() {
        let first = [1_u8; 144];
        let second = [2_u8; 144];
        let third = [3_u8; 144];
        let entries = [
            ExpertEntry {
                block: QuantizedBlock::Q4K(&first),
                out_dim: 256,
                in_dim: 256,
                epoch: 1,
            },
            ExpertEntry {
                block: QuantizedBlock::Q4K(&second),
                out_dim: 256,
                in_dim: 256,
                epoch: 2,
            },
            ExpertEntry {
                block: QuantizedBlock::Q4K(&third),
                out_dim: 256,
                in_dim: 256,
                epoch: 3,
            },
        ];
        let selected = [2_u32];
        let source = ExpertSource::with_selected_expert_ids(&entries, &selected);
        let (payload, descriptors) = selected_expert_payloads(NodeId(12), &source)
            .expect("a sparse route keeps the original expert index");

        assert_eq!(payload, vec![3_u8; 144]);
        assert_eq!(descriptors[0].byte_length, 0);
        assert_eq!(descriptors[1].byte_length, 0);
        assert_eq!(descriptors[2].expert_index, 2);
        assert_eq!(descriptors[2].byte_offset, 0);
        assert_eq!(descriptors[2].byte_length, 144);
    }

    #[test]
    fn mixed_hobbit_entries_accept_q3k_runtime_decoder() {
        let q3_bytes = [0_u8; 110];
        let entries = [ExpertEntry {
            block: QuantizedBlock::Q3K(&q3_bytes),
            out_dim: 256,
            in_dim: 256,
            epoch: 0,
        }];
        let source = ExpertSource::new(&entries);

        let descriptors = expert_payload_descriptors(NodeId(9), &source)
            .expect("Q3_K has a mixed-expert MSL decoder");
        assert_eq!(descriptors[0].codec, PackedCodec::Q3K);
        assert_eq!(descriptors[0].byte_length, q3_bytes.len());
        let packed = pack_expert_payload_descriptors(NodeId(9), &descriptors)
            .expect("Q3_K descriptor fits the Metal ABI");
        assert_eq!(&packed[4..8], &4u32.to_ne_bytes());
    }

    #[test]
    fn substituted_expert_stack_is_absent_from_ordinary_uploads() {
        let full_expert_stack = [0_u8; 4_096];
        let activation = [0.0_f32; 16];
        let blocks = [
            QuantizedBlock::Q4K(&full_expert_stack),
            QuantizedBlock::Float32(&activation),
        ];
        let ordinary = ordinary_block_uploads(
            &[NodeId(41), NodeId(42)],
            &[true, true],
            &blocks,
            &[DType::UInt8, DType::Float32],
            |node| node == NodeId(41),
        )
        .collect::<Vec<_>>();

        assert_eq!(
            ordinary
                .iter()
                .map(|(node, _, _)| *node)
                .collect::<Vec<_>>(),
            vec![NodeId(42)],
            "an ExpertSource replacement must remove the original expert node from ordinary upload"
        );
        assert_eq!(
            ordinary
                .iter()
                .map(|(_, block, _)| match block {
                    QuantizedBlock::Float32(values) => core::mem::size_of_val(*values),
                    QuantizedBlock::Q4K(bytes) => bytes.len(),
                    _ => 0,
                })
                .sum::<usize>(),
            core::mem::size_of_val(&activation),
            "ordinary uploads retain unrelated input bytes but none of the full expert stack"
        );
    }

    #[test]
    fn transient_staging_rejects_a_full_size_expert_stack_before_upload() {
        let original_bytes = [0_u8; 288];
        let first_expert = [0_u8; 144];
        let second_expert = [0_u8; 144];
        let entries = [
            ExpertEntry {
                block: QuantizedBlock::Q4K(&first_expert),
                out_dim: 256,
                in_dim: 256,
                epoch: 1,
            },
            ExpertEntry {
                block: QuantizedBlock::Q4K(&second_expert),
                out_dim: 256,
                in_dim: 256,
                epoch: 2,
            },
        ];
        let source = ExpertSource::new(&entries);

        let error = reject_non_reducing_expert_staging(
            NodeId(51),
            QuantizedBlock::Q4K(&original_bytes),
            &source,
        )
        .expect_err("a full-size staging table is not a low-memory substitution");

        assert!(matches!(
            error,
            MetalError::ExpertSourceUnsupported {
                node: NodeId(51),
                reason: "transient expert staging must reduce checkpoint-resident bytes",
            }
        ));
    }

    #[test]
    fn transient_staging_accepts_a_smaller_mixed_codec_table() {
        let original_bytes = [0_u8; 288];
        let low_expert = [0_u8; 84];
        let high_expert = [0_u8; 144];
        let entries = [
            ExpertEntry {
                block: QuantizedBlock::Q2K(&low_expert),
                out_dim: 256,
                in_dim: 256,
                epoch: 1,
            },
            ExpertEntry {
                block: QuantizedBlock::Q4K(&high_expert),
                out_dim: 256,
                in_dim: 256,
                epoch: 2,
            },
        ];
        let source = ExpertSource::new(&entries);

        reject_non_reducing_expert_staging(
            NodeId(52),
            QuantizedBlock::Q4K(&original_bytes),
            &source,
        )
        .expect("a smaller mixed-codec table reduces the device upload");
    }
}
