use super::*;
#[cfg(feature = "instrument")]
use objc2_metal::MTLBlitCommandEncoder;
#[cfg(feature = "instrument")]
use std::borrow::Borrow;

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
/// This plan's own row count, straight from the caller's bind-time
/// `symbols[0]` (`new_count` in `residency_caches.rs`'s own vocabulary,
/// stored once on [`Plan::bind_row_count`] at [`plan`]/[`plan_named`] time)
/// -- never inferred from any bound op's `extents`. `max(1)` covers a
/// symbols-less caller (bare `Op` unit tests) the same way decode's own
/// `new_count == 1` already does.
///
/// Reading the caller's own row count, rather than scanning `resolved` for
/// the largest leading `extents` axis, is what keeps this correct
/// regardless of a bind's op shapes: the E2B checkpoint's own bound graph on a
/// build without `metal-fuse-attn-decode` decomposes attention into plain
/// `Elementwise`/`Reduce` ops with no `CachedAttention` node to read a row
/// count off at all, while a vocab- or expert-count-leading op elsewhere in
/// the SAME graph (a router logits `Reduce`, a stacked per-expert weight
/// `Constant`) carries a leading axis with no relation to the token count --
/// scanning `extents[0]` across every op cannot tell "the token axis" apart
/// from either. `symbols[0]` has no such ambiguity: it is the exact
/// new-token count the caller resolved before calling `infer`/`bind`, for
/// every op in the program uniformly. Measured on a 1618-token rag-corpus
/// prefill (`gemma4_e2b_prefill_admits_its_true_peak_at_the_bind_time_row_count`
/// below): `new_count == 1618`, `peak_bytes == 495_315_984`, comfortably
/// under `ARENA_TRANSIENT_CAP * 1618`. Decode's own `new_count == 1`
/// collapses this to the unscaled `ARENA_TRANSIENT_CAP`, unchanged. A
/// prefill sharing ONE dispatch across `M` rows (ROW 391's 1100-row
/// interactive-chat prompt) reports `new_count == M` the same way, so the
/// cap still scales up with it.
pub(super) fn plan_bind_row_count(symbols: &[u64]) -> u64 {
    symbols.first().copied().unwrap_or(1).max(1)
}

/// Prints the naive (no-reuse) transient sum against this plan's own
/// `query_rows`-scaled cap before allocating anything, per this card's
/// memory gate. `query_rows` is the caller's own [`Plan::bind_row_count`] --
/// see [`plan_bind_row_count`]'s own doc for why that, rather than a scan
/// over `resolved`, is the cap's source of truth.
#[cfg(feature = "metal-plan-stable-buffers")]
pub(super) fn build_buffer_arena(
    device: &ProtocolObject<dyn MTLDevice>,
    resolved: &[BoundOp],
    retires: &[Vec<NodeId>],
    effective_outputs: &[NodeId],
    resident_nodes: &BTreeSet<NodeId>,
    query_rows: u64,
    numeric_policy: NumericPolicy,
) -> Result<BufferArena, MetalError> {
    let outputs: BTreeSet<NodeId> = effective_outputs.iter().copied().collect();
    let naive_transient_bytes: usize = resolved
        .iter()
        .map(|bound| bound_output_len(bound).max(1) * bound.dtype.size_bytes())
        .sum();
    let uniform_bytes: usize = resolved
        .iter()
        .map(|bound| pack_uniforms_byte_len(bound, numeric_policy))
        .sum();
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
        // A resident position's slot is written once, on the plan's cold
        // call, and never rewritten again (`resident_skip`) -- so it can
        // never share a slot with an ordinary position that keeps writing
        // every call. `resident_pinned_retires` already stops this position's
        // OWN slot from being handed FORWARD once resident; this stops the
        // opposite direction, a resident position being handed a slot an
        // ordinary, every-call position still owns, by refusing the shared
        // free list on its own allocation.
        let is_resident = resident_nodes.contains(&bound.node);
        let slot = match (!is_resident)
            .then(|| free_by_size.get_mut(&byte_length).and_then(Vec::pop))
            .flatten()
        {
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
    for (position, bound) in resolved.iter().enumerate() {
        // read unconditionally so a non-`instrument` build (where the only
        // consumer below is compiled out) does not trip `unused_variables`.
        let _ = position;
        let bytes = pack_uniforms(bound, numeric_policy)?;
        let buffer = device
            .newBufferWithLength_options(bytes.len().max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| MetalError::CompileFailed {
                log: "device refused to allocate a plan uniform buffer".to_string(),
            })?;
        #[cfg(feature = "instrument")]
        chunk_audit_record_write(Retained::as_ptr(&buffer) as usize, 0, bytes.len(), position);
        write_plan_uniform_bytes(&buffer, &bytes);
        #[cfg(feature = "instrument")]
        counter!(PLAN_UNIFORM_WRITES, 1);
        #[cfg(feature = "instrument")]
        counter!(UNIFORM_BUFFER_ALLOCATIONS, 1);
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
        let pinned_retires = resident_pinned_retires(plan);
        let arena = build_buffer_arena(
            &device,
            &plan.prepared.resolved,
            &pinned_retires,
            &plan.prepared.effective_outputs,
            &plan.resident_nodes,
            plan.bind_row_count,
            plan.numeric_policy,
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
        #[cfg(feature = "instrument")]
        let build_started = read_ticks();
        let uniforms = build_plan_uniforms(&device, &plan.prepared.resolved, plan.numeric_policy)?;
        #[cfg(feature = "instrument")]
        counter!(BUILD_PLAN_UNIFORMS_TICKS, elapsed_ticks(build_started));
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
///
/// Above one query row the two split forms (`metal-attn-split-rows`) size
/// against their own bind-time `splits`, the count their partial and merge
/// kernels stride the scratch by, which [`crate::msl::splits_for`] does not
/// reproduce.
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
    #[cfg(feature = "metal-attn-split-rows")]
    let form_splits = match crate::msl::cached_attention_form(&bound.kind, numeric_policy) {
        Some(
            crate::msl::CachedAttentionForm::TwoRangeRowTiled { splits, .. }
            | crate::msl::CachedAttentionForm::TwoRangeDecodeSplit { splits, .. },
        ) if *query_rows > 1 => Some(splits),
        _ => None,
    };
    #[cfg(not(feature = "metal-attn-split-rows"))]
    let form_splits: Option<u64> = None;
    let splits = match form_splits {
        Some(splits) => splits,
        None if *query_rows > 1 => {
            crate::msl::splits_for(cached_key_rows + new_key_rows, numeric_policy)
        }
        None => crate::sized::ATTENTION_SPLIT_MAX,
    };
    Some(total_elements * splits * (2 + head_dim))
}

/// Whether `bound`'s form shares one plan-level scratch buffer with the
/// plan's other ops of that form (`CachedAttentionForm::shares_scratch`).
#[cfg(any(test, feature = "metal-plan-stable-buffers"))]
fn scratch_is_shared(bound: &BoundOp, numeric_policy: NumericPolicy) -> bool {
    crate::msl::cached_attention_form(&bound.kind, numeric_policy)
        .is_some_and(crate::msl::CachedAttentionForm::shares_scratch)
}

/// The one scratch buffer every position that shares it reads and writes,
/// sized at the largest of their own lengths; `None` when no position
/// shares. One allocation per plan instead of one per attention op: a
/// verify step at K = 49 would otherwise reserve 135.8 MB across the 35
/// layers where this holds the 4.84 MB of the widest.
#[cfg(feature = "metal-plan-stable-buffers")]
fn shared_attention_scratch(
    device: &ProtocolObject<dyn MTLDevice>,
    plan: &Plan,
) -> Result<Option<MetalBuffer>, MetalError> {
    shared_scratch_elements(&plan.prepared.resolved, plan.numeric_policy)
        .map(|elements| allocate_buffer(device, elements as usize, DType::Float32))
        .transpose()
}

/// [`shared_attention_scratch`]'s sizing: the widest scratch length among the
/// positions that share, `None` when none does. Separate from the allocation
/// so a plan's sharing is checkable without a device.
#[cfg(any(test, feature = "metal-plan-stable-buffers"))]
fn shared_scratch_elements(resolved: &[BoundOp], numeric_policy: NumericPolicy) -> Option<u64> {
    resolved
        .iter()
        .filter(|bound| scratch_is_shared(bound, numeric_policy))
        .filter_map(|bound| cached_attention_scratch_len(bound, numeric_policy))
        .max()
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
        let shared = shared_attention_scratch(&device, plan)?;
        for bound in &plan.prepared.resolved {
            let buffer = match cached_attention_scratch_len(bound, plan.numeric_policy) {
                Some(_) if scratch_is_shared(bound, plan.numeric_policy) => shared.clone(),
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

/// One position's `(pipeline, bindings, grid, merge)`: the per-`bound` body
/// of [`resolve_steps`], extracted so [`Plan::refit_symbols`] re-resolves
/// only the positions it patched. A pure function of `bound` and the plan's
/// `(packed_operands, numeric_policy, math_mode)` -- which is why a patched
/// position's step equals the one a fresh plan would resolve for it.
pub(super) fn resolve_step(
    device: &ProtocolObject<dyn MTLDevice>,
    bound: &BoundOp,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
    math_mode: MathMode,
) -> Result<ResolvedStep, MetalError> {
    let (bindings, grid) =
        kernel_dispatch_shape(bound, packed_operands, numeric_policy)?;
    let mut cache_key =
        kernel_cache_key_for_grid(bound, packed_operands, numeric_policy, &grid)?;
    cache_key.push(math_mode.cache_token());
    #[cfg(feature = "instrument")]
    if let BoundOpKind::Reduce {
        reduce_op,
        init,
        output_axes,
        epilogue_body,
        epilogue_operands,
        epilogue_broadcast_axes,
        ..
    } = &bound.kind
    {
        crate::msl::debug_tiled_gemm_classification(
            bound,
            &crate::identity::operand_codecs(bound, packed_operands),
            *reduce_op,
            *init,
            output_axes,
            epilogue_body,
            epilogue_operands,
            epilogue_broadcast_axes,
            &cache_key,
        );
    }
    let pipeline = pipeline_for(
        device,
        bound,
        packed_operands,
        &cache_key,
        math_mode,
        numeric_policy,
    )?;
    // Redesign §4c: a `CachedAttention` position under a policy that
    // admits `ContextSplitMerge` resolves a SECOND pipeline for the
    // merge dispatch, keyed on the split's own cache key plus `_merge`
    // so the two never collide in `PIPELINE_CACHE` even though they
    // share every other structural token.
    let merge = match crate::msl::emit_cached_attention_merge(bound, numeric_policy)? {
        Some(merge_kernel) => {
            let merge_cache_key = merge_pipeline_key(&cache_key, &merge_kernel);
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
    Ok(ResolvedStep {
        pipeline,
        bindings,
        grid,
        merge,
    })
}

/// Installs `patches` (position, replacement op) into `plan` and re-derives
/// the per-position state that is a pure function of the op: its resolved
/// step (pipeline, bindings, grid) and, under `metal-plan-stable-buffers`,
/// its plan-owned uniform buffer, and drops the attention scratch when a
/// multi-row op's key range moved. Everything else the plan holds -- the
/// output arena, the retirement schedule -- is a function of operand
/// structure and output extents, which a patch never changes. All fallible work happens before the first write, so an `Err`
/// leaves `plan` exactly as it was.
pub(super) fn apply_refit(
    plan: &mut Plan,
    patches: Vec<(usize, BoundOp)>,
    next_shapes: Shapes,
) -> Result<(), MetalError> {
    let (device, _queue) = device_and_queue()?;
    let steps = refit_steps(&device, plan, &patches)?;
    #[cfg(feature = "metal-plan-stable-buffers")]
    let uniforms = refit_uniform_buffers(&device, plan, &patches)?;
    // a multi-row attention op sizes its scratch by its key range, so the
    // plan's scratch buffers are rebuilt lazily from the patched ops; a
    // single-row op reserves the compiled split ceiling whatever its range
    #[cfg(feature = "metal-plan-stable-buffers")]
    if patches.iter().any(|(_, bound)| {
        matches!(bound.kind, BoundOpKind::CachedAttention { query_rows, .. } if query_rows > 1)
    }) {
        plan.attention_scratch.take();
    }
    for (position, bound) in patches {
        plan.prepared.resolved[position] = bound;
    }
    plan.prepared.shapes = next_shapes;
    #[cfg(feature = "metal-horizontal-merge")]
    {
        let _ = steps;
        plan.resolved_steps.get_mut().take();
        plan.merged.get_mut().take();
    }
    #[cfg(not(feature = "metal-horizontal-merge"))]
    if let (Some(steps), Some(resolved)) = (steps, plan.resolved_steps.get_mut().as_mut()) {
        for (position, step) in steps {
            resolved.steps[position] = step;
        }
    }
    #[cfg(feature = "metal-plan-stable-buffers")]
    if let (Some(buffers), Some(plan_uniforms)) = (uniforms, plan.uniforms.get_mut()) {
        for (position, buffer) in buffers {
            plan_uniforms.buffers[position] = buffer;
        }
    }
    Ok(())
}

/// The steps to swap in, or `None` when the plan has not resolved its steps
/// yet (the first execution will resolve them from the patched ops) or
/// resolved them under another math mode (a rebuild is already due).
fn refit_steps(
    device: &ProtocolObject<dyn MTLDevice>,
    plan: &Plan,
    patches: &[(usize, BoundOp)],
) -> Result<Option<Vec<(usize, ResolvedStep)>>, MetalError> {
    let current = plan.resolved_steps.borrow();
    if !current
        .as_ref()
        .is_some_and(|resolved| resolved.math_mode == plan.math_mode)
    {
        return Ok(None);
    }
    patches
        .iter()
        .map(|(position, bound)| {
            resolve_step(
                device,
                bound,
                &plan.packed_operands,
                plan.numeric_policy,
                plan.math_mode,
            )
            .map(|step| (*position, step))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

#[cfg(feature = "metal-plan-stable-buffers")]
fn refit_uniform_buffers(
    device: &ProtocolObject<dyn MTLDevice>,
    plan: &Plan,
    patches: &[(usize, BoundOp)],
) -> Result<Option<Vec<(usize, MetalBuffer)>>, MetalError> {
    if plan.uniforms.get().is_none() {
        return Ok(None);
    }
    patches
        .iter()
        .map(|(position, bound)| {
            let bytes = pack_uniforms(bound, plan.numeric_policy)?;
            let buffer = device
                .newBufferWithLength_options(bytes.len().max(1), MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| MetalError::CompileFailed {
                    log: "device refused to allocate a plan uniform buffer".to_string(),
                })?;
            write_plan_uniform_bytes(&buffer, &bytes);
            Ok((*position, buffer))
        })
        .collect::<Result<Vec<_>, MetalError>>()
        .map(Some)
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
    #[cfg(feature = "instrument")]
    let resolve_started = read_ticks();
    // a math-mode change recompiles every pipeline, so a merged group's own
    // compiled pipeline/base_table (keyed off the STALE step pipelines) must
    // be dropped too, or the encode loop below would dispatch a merged
    // kernel compiled under the old mode against buffers resolved for it.
    #[cfg(feature = "metal-horizontal-merge")]
    plan.merged.borrow_mut().take();
    let mut steps = Vec::with_capacity(plan.prepared.resolved.len());
    for bound in &plan.prepared.resolved {
        steps.push(resolve_step(
            device,
            bound,
            &plan.packed_operands,
            plan.numeric_policy,
            plan.math_mode,
        )?);
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
    #[cfg(feature = "instrument")]
    counter!(RESOLVE_STEPS_TICKS, elapsed_ticks(resolve_started));
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
/// `PROXIMA_CAPTURE_NODES=<comma list>` prints one `dispatch_capture` line
/// per matched `bound.node`, right before the sole `dispatchThreads` call
/// site (`resident_nocopy_cache::dispatch`) actually submits it -- the
/// production launch record: cache key, entry name and MSL sha256 (recovered
/// from [`pipeline_buffers_upload::PIPELINE_CAPTURE`] by the compiled
/// pipeline's own pointer, so a plan-cache HIT still reports the ORIGINAL
/// creation record rather than nothing), the actual grid/threadgroup this op
/// dispatches, and every bound buffer's identity/offset/length. Reads two
/// env vars per call (`PROXIMA_CAPTURE_NODES`, parsed once per call rather
/// than cached, since this only ever fires under `instrument` for a handful
/// of explicitly named nodes) -- never on the default decode hot path.
///
/// `PROXIMA_CAPTURE_DUMP_DIR=<dir>`, on top of a `PROXIMA_CAPTURE_NODES`
/// match: stashes this dispatch's resolved buffer handles (not their bytes
/// yet -- the GPU has not run this dispatch at encode time) into
/// [`CAPTURE_DUMP_PENDING`] for [`flush_pending_capture_dumps`] to read back
/// once the command buffer carrying this dispatch has actually completed.
/// uses the renderer's own packed-row classifier so this substitution's
/// admission can never diverge from what actually dispatches. `false` on a
/// gathered weight: this diagnostic's blit substitution has no gather-aware
/// ABI.
#[cfg(feature = "instrument")]
pub(super) fn is_packed_multi_token_projection(
    bound: &BoundOp,
    packed_operands: &PackedOperands,
) -> bool {
    let quantized: Vec<Option<Codec>> = bound
        .operands()
        .iter()
        .map(|(node, _, _)| packed_operands.get(node).copied())
        .collect();
    let Some(block) = crate::msl::packed_row_block(bound, &quantized) else {
        return false;
    };
    let token_total: u64 = block
        .token_axes
        .iter()
        .map(|&axis| bound.extents[axis as usize])
        .product();
    token_total > 1
}

#[cfg(feature = "instrument")]
// eight call-site facts a diagnostic dump genuinely needs (bound op, packed
// operand codecs, the compiled pipeline, launch geometry, every binding, the
// resolved buffer map, the output slot, the uniforms buffer) -- splitting
// these into a struct only this one diagnostic function would ever construct
// is not a real type, it is this argument list with extra steps.
#[allow(clippy::too_many_arguments)]
fn capture_dispatch(
    bound: &BoundOp,
    packed_operands: &PackedOperands,
    pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    grid: GridSpec,
    bindings: &[Binding],
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    output: (&MetalBuffer, usize),
    scratch: Option<(&MetalBuffer, usize)>,
    chunk_index: usize,
    uniforms: &MetalBuffer,
) {
    let Some(wanted) = std::env::var("PROXIMA_CAPTURE_NODES").ok() else {
        return;
    };
    let output_only = wanted.trim() == "packed-multi-token";
    let matched = wanted.trim() == "all"
        || (output_only && is_packed_multi_token_projection(bound, packed_operands))
        || wanted
            .split(',')
            .filter_map(|token| token.trim().parse::<u32>().ok())
            .any(|node| node == bound.node.0);
    if !matched {
        return;
    }
    let step = CAPTURE_STEP.load(core::sync::atomic::Ordering::Relaxed);
    // `PROXIMA_CAPTURE_STEPS=<comma list>`, default (unset) every step --
    // read every call rather than cached, matching `PROXIMA_CAPTURE_NODES`'s
    // own "handful of dispatches under `instrument`" cost budget.
    if let Ok(steps_wanted) = std::env::var("PROXIMA_CAPTURE_STEPS") {
        let step_matched = steps_wanted
            .split(',')
            .filter_map(|token| token.trim().parse::<u64>().ok())
            .any(|wanted_step| wanted_step == step);
        if !step_matched {
            return;
        }
    }
    let operand_records: Vec<String> = bound
        .operands()
        .iter()
        .map(|(node, layout, _lookup)| {
            let codec = packed_operands
                .get(node)
                .map_or("unpacked".to_string(), |codec| format!("{codec:?}"));
            format!(
                "({}, {:?}, {:?}, {codec})",
                node.0, layout.strides, bound.dtype
            )
        })
        .collect();
    // `BoundOp` carries no separate output `Layout` -- a bound op's own
    // write is always contiguous into `output`/`state_out` at `encode_op`'s
    // resolved offset, so `extents` (the iteration-space shape) is the
    // output-side geometry fact this record has to report, not a stride
    // vector `BoundOp` never stores.
    debug!(
        node = bound.node.0,
        kind = bound.kind.name(),
        operands = %operand_records.join(", "),
        extents = ?bound.extents,
        dtype = ?bound.dtype,
        "bound_op"
    );
    let record = PIPELINE_CAPTURE.with(|capture| {
        capture
            .borrow()
            .get(&(Retained::as_ptr(pipeline) as usize))
            .cloned()
    });
    let (cache_key, entry, msl_sha256) = match &record {
        Some(record) => (
            record.cache_key.clone(),
            record.entry.clone(),
            record.msl_sha256.clone(),
        ),
        None => ("<unrecorded>".to_string(), "<unrecorded>".to_string(), "<unrecorded>".to_string()),
    };
    let max_threadgroup = pipeline.maxTotalThreadsPerThreadgroup();
    let threadgroup_width = match grid.threadgroup_width {
        Some(width) => (width as usize).min(max_threadgroup).max(1),
        None => (grid.threads as usize).min(max_threadgroup).max(1),
    };
    // the launch `resident_nocopy_cache::dispatch` actually issues: recorded
    // from the same `grid2d` decision and the same pipeline clamp, so a replay
    // of a flat or tile kernel is not handed a 1D grid it never ran under
    let launch2d = grid
        .grid2d
        .map(|spec| crate::msl::fit_flat_width(spec, max_threadgroup as u64));
    let (recorded_grid, recorded_threadgroup, recorded_dispatch) = match launch2d {
        Some(spec) => (
            (spec.threadgroups_x as usize, spec.threadgroups_y as usize, grid.depth as usize),
            (spec.threads_per_threadgroup_x as usize, spec.threads_per_threadgroup_y as usize, 1),
            "threadgroups",
        ),
        None => ((grid.threads as usize, 1, grid.depth as usize), (threadgroup_width, 1, 1), "threads"),
    };
    let mut buffers = Vec::new();
    let mut dump_buffers = Vec::new();
    for (index, binding) in bindings.iter().enumerate() {
        let resolved: Option<(MetalBuffer, usize)> = match binding {
            Binding::Input(node) | Binding::Indices(node) => {
                device_buffers.get(node).cloned()
            }
            Binding::Output(_) => Some((output.0.clone(), output.1)),
            _ => None,
        };
        if let Some((buffer, offset)) = resolved {
            buffers.push(format!(
                "({index}, {:?}, {offset}, {})",
                Retained::as_ptr(&buffer),
                buffer.length()
            ));
            // keeps the dump small: only the output, never the resident
            // weight/activation inputs a substitution pass never re-reads.
            if !output_only || matches!(binding, Binding::Output(_)) {
                dump_buffers.push((index, buffer, offset));
            }
        } else {
            buffers.push(format!("({index}, unresolved, 0, 0)"));
        }
    }
    if std::env::var_os("PROXIMA_CAPTURE_DUMP_DIR").is_some() {
        CAPTURE_DUMP_PENDING.with(|pending| {
            pending.borrow_mut().push(CaptureDumpEntry {
                step,
                node: bound.node.0,
                bound_op_line: format!(
                    "bound_op node={} kind={} operands=[{}] output=(extents={:?}, dtype={:?})",
                    bound.node.0,
                    bound.kind.name(),
                    operand_records.join(", "),
                    bound.extents,
                    bound.dtype,
                ),
                entry: entry.clone(),
                msl_sha256: msl_sha256.clone(),
                grid: recorded_grid,
                threadgroup: recorded_threadgroup,
                dispatch: recorded_dispatch,
                dtype: format!("{:?}", bound.dtype),
                extents: format!("{:?}", bound.extents),
                buffers: dump_buffers,
                uniforms: (uniforms.clone(), uniforms.length()),
                manifest: output_only,
            });
        });
    }
    if std::env::var_os("PROXIMA_CAPTURE_LIVE").is_some() {
        let mut live_buffers = Vec::new();
        let mut uniforms_index = None;
        let mut fault_index = None;
        let mut unreplayable = None;
        for (index, binding) in bindings.iter().enumerate() {
            let resolved = match binding {
                Binding::Input(node) | Binding::Indices(node) => device_buffers.get(node).cloned(),
                Binding::Output(_) => Some((output.0.clone(), output.1)),
                Binding::Uniforms => {
                    uniforms_index = Some(index);
                    continue;
                }
                Binding::Fault => {
                    fault_index = Some(index);
                    continue;
                }
                Binding::Scratch => scratch.map(|(buffer, offset)| (buffer.clone(), offset)),
                Binding::ExpertPayloads(_) | Binding::ExpertDescriptors(_) => None,
            };
            match resolved {
                Some((buffer, offset)) => live_buffers.push((index, buffer, offset)),
                None => unreplayable = Some(format!("binding {index} ({binding:?}) not resolvable")),
            }
        }
        let extras_reason = live_extra_buffers(bound, bindings.len(), device_buffers, &mut live_buffers);
        // SAFETY: `uniforms` is a live shared buffer of `length()` bytes.
        let uniform_bytes = unsafe {
            core::slice::from_raw_parts(uniforms.contents().as_ptr().cast::<u8>(), uniforms.length())
        }
        .to_vec();
        let live_operands = bound
            .operands()
            .iter()
            .map(|(node, _layout, _lookup)| {
                let codec = packed_operands
                    .get(node)
                    .map_or("unpacked".to_string(), |codec| format!("{codec:?}"));
                (node.0, codec)
            })
            .collect();
        CAPTURE_LIVE_PENDING.with(|pending| {
            pending.borrow_mut().push(CapturedDispatch {
                step,
                node: bound.node.0,
                kind_name: bound.kind.name(),
                entry: entry.clone(),
                msl_sha256: msl_sha256.clone(),
                operands: live_operands,
                chunk_index,
                extents: bound.extents.clone(),
                grid,
                bindings: bindings.to_vec(),
                unreplayable: unreplayable.or(extras_reason),
                uniform_bytes,
                pipeline: pipeline.clone(),
                buffers: live_buffers,
                uniforms_index,
                fault_index,
            });
        });
    }
    debug!(
        step,
        node = bound.node.0,
        key = ?cache_key,
        entry = %entry,
        msl_sha256 = %msl_sha256,
        grid_threads = grid.threads,
        grid_depth = grid.depth,
        threadgroup_width,
        grid2d = ?launch2d,
        buffers = %buffers.join(", "),
        "dispatch_capture"
    );
}

/// One [`capture_dispatch`] match under `PROXIMA_CAPTURE_DUMP_DIR`, held
/// until [`flush_pending_capture_dumps`] can read the GPU-written bytes back
/// -- at encode time (where [`capture_dispatch`] runs) this dispatch has
/// only been recorded into the command buffer, not yet executed.
#[cfg(feature = "instrument")]
struct CaptureDumpEntry {
    step: u64,
    node: u32,
    bound_op_line: String,
    entry: String,
    msl_sha256: String,
    grid: (usize, usize, usize),
    threadgroup: (usize, usize, usize),
    /// `threads` (`dispatchThreads`, `grid` is the thread count) or `threadgroups`
    /// (`dispatchThreadgroups`, `grid` is the threadgroup count): which call a
    /// replay must make for `grid` and `threadgroup` to mean what they did.
    dispatch: &'static str,
    dtype: String,
    extents: String,
    /// `(binding index, buffer, offset)`, one per resolved `Input`/`Indices`/
    /// `Output` binding -- the same set [`capture_dispatch`]'s own
    /// `dispatch_capture` line already reports as `(index, ptr, offset,
    /// length)`, just holding the live buffer handle instead of formatting
    /// it away.
    buffers: Vec<(usize, MetalBuffer, usize)>,
    uniforms: (MetalBuffer, usize),
    /// `true` under `PROXIMA_CAPTURE_NODES=packed-multi-token`: also append
    /// this node to `manifest.txt` at flush time.
    manifest: bool,
}

thread_local! {
    #[cfg(feature = "instrument")]
    static CAPTURE_DUMP_PENDING: RefCell<Vec<CaptureDumpEntry>> = const { RefCell::new(Vec::new()) };
}

/// Reads back every [`CaptureDumpEntry`] queued since the last flush and
/// writes its buffer bytes to `PROXIMA_CAPTURE_DUMP_DIR`. Call ONLY at a
/// point where every command buffer that could contain a queued dispatch has
/// already returned from `waitUntilCompleted` -- Apple Silicon's unified
/// memory makes `MTLBuffer::contents()` a plain host pointer, valid to read
/// the instant the GPU work that wrote it has retired, but not one instant
/// before (an in-flight write racing this read is exactly the hazard
/// `waitUntilCompleted` exists to close). Draining (not just reading) the
/// pending list means a later flush from the same process step never
/// re-dumps an already-written node.
///
/// `<dir>/node<id>_buf<index>_off<offset>_len<len>.bin` holds one binding's
/// raw bytes; `<dir>/node<id>_uniforms_len<len>.bin` holds the bound
/// uniforms struct, when one was bound; `<dir>/node<id>.meta` holds the
/// `bound_op` line plus extents/dtype/grid/threadgroup/entry/msl_sha256 --
/// everything [`capture_dispatch`]'s own `dispatch_capture` line already
/// prints, just also durable as a file `replay_projection.rs` can read
/// without re-parsing stderr.
#[cfg(feature = "instrument")]
pub(super) fn flush_pending_capture_dumps() {
    let Some(dir) = std::env::var_os("PROXIMA_CAPTURE_DUMP_DIR") else {
        return;
    };
    let entries = CAPTURE_DUMP_PENDING.with(|pending| core::mem::take(&mut *pending.borrow_mut()));
    if entries.is_empty() {
        return;
    }
    let dir = std::path::PathBuf::from(dir);
    if let Err(err) = std::fs::create_dir_all(&dir) {
        debug!(?err, dir = ?dir, "capture dump: failed to create dump directory");
        return;
    }
    for entry in entries {
        let node = entry.node;
        let step = entry.step;
        // A resident weight buffer is one no-copy `MTLBuffer` backing the
        // whole checkpoint mmap (gigabytes); `buffer.length() - offset` on
        // that identity is "rest of the checkpoint file", not this op's own
        // operand. Cap at 64 MiB -- comfortably above every real operand this
        // capture has seen (600x12288 f32 = 29.49 MB, the largest) -- so the
        // dumped file is bounded while still holding this op's whole real
        // slice for every non-resident buffer.
        const CAPTURE_DUMP_MAX_BYTES: usize = 64 * 1024 * 1024;
        let mut buffer_records = Vec::with_capacity(entry.buffers.len());
        let mut manifest_row: Option<(usize, String)> = None;
        for (index, buffer, offset) in &entry.buffers {
            let length = buffer
                .length()
                .saturating_sub(*offset)
                .min(CAPTURE_DUMP_MAX_BYTES);
            let pointer = buffer.contents().as_ptr().cast::<u8>();
            let bytes = unsafe { core::slice::from_raw_parts(pointer.add(*offset), length) };
            // step in the name: without it, a later step silently overwrites
            // an earlier step's dump of the same node.
            let path = dir.join(format!("node{node}_step{step}_buf{index}_off{offset}_len{length}.bin"));
            if let Err(err) = std::fs::write(&path, bytes) {
                debug!(?err, path = ?path, "capture dump: failed to write buffer bytes");
            }
            buffer_records.push(format!("({index}, off={offset}, len={length})"));
            if entry.manifest {
                use sha2::{Digest, Sha256};
                let digest = Sha256::digest(bytes);
                manifest_row = Some((length, format!("{digest:x}")));
            }
        }
        if let (true, Some((length, sha256))) = (entry.manifest, manifest_row) {
            let manifest_line =
                format!("node={node}\textents={}\toutput_len={length}\tsha256={sha256}\n", entry.extents);
            let manifest_path = dir.join("manifest.txt");
            let append_result = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&manifest_path)
                .and_then(|mut file| {
                    use std::io::Write;
                    file.write_all(manifest_line.as_bytes())
                });
            if let Err(err) = append_result {
                debug!(?err, path = ?manifest_path, "capture dump: failed to append manifest line");
            }
        }
        let (uniforms_buffer, uniforms_len) = &entry.uniforms;
        let mut uniforms_note = "none".to_string();
        if *uniforms_len > 0 {
            let pointer = uniforms_buffer.contents().as_ptr().cast::<u8>();
            let bytes = unsafe { core::slice::from_raw_parts(pointer, *uniforms_len) };
            let path = dir.join(format!("node{node}_step{step}_uniforms_len{uniforms_len}.bin"));
            if let Err(err) = std::fs::write(&path, bytes) {
                debug!(?err, path = ?path, "capture dump: failed to write uniforms bytes");
            }
            uniforms_note = format!("node{node}_step{step}_uniforms_len{uniforms_len}.bin");
        }
        let meta = format!(
            "{}\nstep={}\nentry={}\nmsl_sha256={}\ndispatch={}\ngrid={:?}\nthreadgroup={:?}\ndtype={}\nextents={}\nbuffers=[{}]\nuniforms={uniforms_note}\n",
            entry.bound_op_line,
            entry.step,
            entry.entry,
            entry.msl_sha256,
            entry.dispatch,
            entry.grid,
            entry.threadgroup,
            entry.dtype,
            entry.extents,
            buffer_records.join(", "),
        );
        let meta_path = dir.join(format!("node{node}_step{step}.meta"));
        if let Err(err) = std::fs::write(&meta_path, meta) {
            debug!(?err, path = ?meta_path, "capture dump: failed to write meta file");
        }
        // Structured event, not a hand-rolled `eprintln!` (rust.md: "never
        // hand-roll env-gated file dumps for forensics/instrumentation").
        // The metadata (node/step/entry/extents/sha256/sizes) is fully
        // captured here as typed fields; the raw buffer/uniforms BYTES stay
        // plain `std::fs::write` above, never routed through
        // `LogBody::Owned` -- `proxima-telemetry/src/pipes.rs:1859-1861`
        // (and its JSON-format twin at `:1889-1891`) render `Owned` bytes
        // via `String::from_utf8_lossy`, which is lossy for arbitrary
        // binary tensor data (a Q4_0-packed weight row is not valid UTF-8
        // in general): routing a 30 MB quantized buffer through that body
        // type and either text-format exporter would silently corrupt it,
        // not merely truncate it. The `Bytes` type `LogBody::Owned` wraps
        // has no fixed size ceiling of its own; the refusal is a fidelity
        // one, not a capacity one.
        debug!(
            node,
            step,
            dir = ?dir,
            manifest = entry.manifest,
            buffer_count = entry.buffers.len() as u64,
            "capture dump: buffers and metadata flushed"
        );
    }
}

/// One dispatch of the captured decode step, kept LIVE (pipeline, real
/// resolved buffers, uniform bytes) so a harness can re-encode it alone in
/// its own command buffer -- the isolated-replay half of the per-kernel GPU
/// time census. Recorded by [`capture_dispatch`] when `PROXIMA_CAPTURE_LIVE`
/// is set, on top of the same `PROXIMA_CAPTURE_NODES`/`PROXIMA_CAPTURE_STEPS`
/// selection the file dump uses; drained by [`take_captured_dispatches`].
#[cfg(feature = "instrument")]
pub struct CapturedDispatch {
    /// decode step this dispatch was encoded in, e.g. `23`
    pub step: u64,
    /// `NodeId` of the bound op, e.g. `4521`
    pub node: u32,
    /// `BoundOpKind::name`, e.g. `keep::reduce fold`
    pub kind_name: &'static str,
    /// emitted MSL entry name, e.g. `omega_reduce_r3_o2_n2_multiply_add_zero`
    pub entry: String,
    /// sha256 hex of the emitted MSL source, the kernel identity
    pub msl_sha256: String,
    /// `(operand node, codec)` per bound operand; codec is `unpacked` or a
    /// `Codec` debug name such as `Q4_0`
    pub operands: Vec<(u32, String)>,
    /// one-based command-buffer chunk in the captured decode step
    pub chunk_index: usize,
    /// the op's iteration-space extents, e.g. `[1, 1, 512]`
    pub extents: Vec<u64>,
    pub grid: GridSpec,
    pub bindings: Vec<Binding>,
    /// `Some(reason)` when a buffer this kernel needs was not recoverable
    /// at the dispatch site, so [`Self::time_gpu_ns`] refuses it
    pub unreplayable: Option<String>,
    /// raw bytes of the bound `Uniforms` struct at encode time
    pub uniform_bytes: Vec<u8>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    buffers: Vec<(usize, MetalBuffer, usize)>,
    uniforms_index: Option<usize>,
    fault_index: Option<usize>,
}

#[cfg(feature = "instrument")]
thread_local! {
    static CAPTURE_LIVE_PENDING: RefCell<Vec<CapturedDispatch>> = const { RefCell::new(Vec::new()) };
}

/// Drains every [`CapturedDispatch`] queued on this thread since the last
/// call. Empty unless `PROXIMA_CAPTURE_LIVE` was set while the decode step
/// ran on this same thread.
#[cfg(feature = "instrument")]
pub fn take_captured_dispatches() -> Vec<CapturedDispatch> {
    CAPTURE_LIVE_PENDING.with(|pending| core::mem::take(&mut *pending.borrow_mut()))
}

#[cfg(feature = "instrument")]
const FAULT_REPLAY_BYTES: usize = 4096;

/// A no-copy buffer backing a whole checkpoint mmap, whose host mapping a
/// harness may already have released; its contents are never read here.
#[cfg(feature = "instrument")]
const RESIDENT_WEIGHT_BUFFER_BYTES: usize = 1 << 28;

#[cfg(feature = "instrument")]
fn live_extra_buffers(
    bound: &BoundOp,
    slot_base: usize,
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    buffers: &mut Vec<(usize, MetalBuffer, usize)>,
) -> Option<String> {
    let extra_nodes: Vec<NodeId> = match &bound.kind {
        BoundOpKind::CachedSoftmaxWeights {
            cached_weight_sum,
            new_weight_sum,
            new_attended,
            ..
        } => vec![*cached_weight_sum, *new_weight_sum, *new_attended],
        BoundOpKind::GatedDeltaNet { .. } | BoundOpKind::MoeTopK { .. } => {
            return Some(format!("{} binds extra outputs outside `bindings`", bound.kind.name()));
        }
        _ => return None,
    };
    for (offset, node) in extra_nodes.iter().enumerate() {
        let Some((buffer, buffer_offset)) = device_buffers.get(node).cloned() else {
            return Some(format!("extra output node {} has no device buffer", node.0));
        };
        buffers.push((slot_base + offset, buffer, buffer_offset));
    }
    None
}

#[cfg(feature = "instrument")]
fn gpu_span_ns(command_buffer: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<f64, MetalError> {
    if command_buffer.status() == MTLCommandBufferStatus::Error {
        let reason = command_buffer
            .error()
            .map_or("no NSError".to_string(), |error| error.localizedDescription().to_string());
        return Err(MetalError::CompileFailed {
            log: format!("replay command buffer failed: {reason}"),
        });
    }
    Ok(((command_buffer.GPUEndTime() - command_buffer.GPUStartTime()) * 1e9).max(0.0))
}

#[cfg(feature = "instrument")]
fn shared_buffer_from(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
) -> Result<MetalBuffer, MetalError> {
    let buffer = device
        .newBufferWithLength_options(bytes.len().max(1), MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| MetalError::CompileFailed {
            log: "device refused to allocate a replay buffer".to_string(),
        })?;
    // SAFETY: freshly allocated shared buffer of at least `bytes.len()` bytes.
    unsafe {
        core::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            buffer.contents().as_ptr().cast::<u8>(),
            bytes.len(),
        );
    }
    Ok(buffer)
}

#[cfg(feature = "instrument")]
const REPLAY_POISON_BYTE: u8 = 0x55;

#[cfg(feature = "instrument")]
impl CapturedDispatch {
    /// The same captured buffers and uniform bytes bound to a pipeline
    /// compiled from `source`'s `entry` under the math mode `numeric_policy`
    /// maps to, launched over `threads` threads of `threadgroup_width`, so a
    /// kernel body variant is timed on the exact data the production kernel
    /// ran on. `source` must keep the captured kernel's buffer and uniform
    /// layout.
    ///
    /// # Errors
    ///
    /// [`MetalError::CompileFailed`] when the device or the MSL compiler
    /// refuses `source`.
    pub fn with_kernel_variant(
        &self,
        source: &str,
        entry: &str,
        threads: u64,
        threadgroup_width: Option<u64>,
        numeric_policy: NumericPolicy,
    ) -> Result<Self, MetalError> {
        let (device, _queue) = device_and_queue()?;
        let grid = GridSpec {
            threads,
            threadgroup_width,
            ..self.grid
        };
        let kernel = Kernel {
            source: source.to_string(),
            entry: entry.to_string(),
            bindings: self.bindings.clone(),
            grid,
        };
        let pipeline = compile_pipeline(
            &device,
            &kernel,
            numeric_policy_as_metal_math_mode(numeric_policy),
        )?;
        Ok(Self {
            step: self.step,
            node: self.node,
            kind_name: self.kind_name,
            entry: entry.to_string(),
            msl_sha256: String::new(),
            operands: self.operands.clone(),
            chunk_index: self.chunk_index,
            extents: self.extents.clone(),
            grid,
            bindings: self.bindings.clone(),
            unreplayable: self.unreplayable.clone(),
            uniform_bytes: self.uniform_bytes.clone(),
            pipeline,
            buffers: self.buffers.clone(),
            uniforms_index: self.uniforms_index,
            fault_index: self.fault_index,
        })
    }

    /// One line per bound buffer: binding index, buffer address, byte offset,
    /// the buffer's length, and its first four f32 values at that offset, so a
    /// harness can tell which bindings of a fused kernel read the same data.
    #[must_use]
    pub fn describe_buffers(&self) -> Vec<String> {
        self.buffers
            .iter()
            .map(|(index, buffer, offset)| {
                let length = buffer.length();
                let available = length.saturating_sub(*offset);
                let resident_weights = length > RESIDENT_WEIGHT_BUFFER_BYTES;
                let count = if resident_weights {
                    0
                } else {
                    (available / core::mem::size_of::<f32>()).min(4)
                };
                let base = buffer.contents().as_ptr().cast::<u8>();
                // SAFETY: shared-storage buffer idle between replays; `offset + count * 4 <= length` by `available`.
                let head: Vec<f32> = (0..count)
                    .map(|element| unsafe {
                        base.add(*offset + element * core::mem::size_of::<f32>())
                            .cast::<f32>()
                            .read_unaligned()
                    })
                    .collect();
                format!(
                    "binding={index} buffer={:p} offset={offset} length={length} head={head:?}",
                    buffer.contents().as_ptr()
                )
            })
            .collect()
    }

    /// This dispatch's buffers and uniform bytes bound to the pipeline and
    /// launch shape `template` carries (a [`Self::with_kernel_variant`] result
    /// built from a sibling in the same kernel group), so every member of a
    /// group replays through ONE pipeline object, as the live step does,
    /// instead of paying a pipeline switch per dispatch.
    #[must_use]
    pub fn with_pipeline_of(&self, template: &Self) -> Self {
        Self {
            step: self.step,
            node: self.node,
            kind_name: self.kind_name,
            entry: template.entry.clone(),
            msl_sha256: String::new(),
            operands: self.operands.clone(),
            chunk_index: self.chunk_index,
            extents: self.extents.clone(),
            grid: template.grid,
            bindings: self.bindings.clone(),
            unreplayable: self.unreplayable.clone(),
            uniform_bytes: self.uniform_bytes.clone(),
            pipeline: template.pipeline.clone(),
            buffers: self.buffers.clone(),
            uniforms_index: self.uniforms_index,
            fault_index: self.fault_index,
        }
    }

    /// Poisons the output region (`output_total`, the first `i64` of the bound
    /// uniforms, in f32 elements, bounded by the iteration-space element count:
    /// a reduce's iteration space includes its reduction axis and would
    /// overwrite neighbouring tensors of a pooled buffer), runs this dispatch once, and
    /// returns that region's bytes, so two kernels bound to the same buffers
    /// compare element for element on the data the production kernel ran on,
    /// and a kernel that fails to write an element cannot match one that did.
    ///
    /// # Errors
    ///
    /// [`MetalError::CompileFailed`] when this dispatch is unreplayable, has
    /// no output binding, or fails to run.
    pub fn replay_output(&self) -> Result<Vec<u8>, MetalError> {
        self.replay_output_elements(None)
    }

    /// [`Self::replay_output`] over `elements` f32 values when the caller knows
    /// the op writes more than `output_total` of them (a fused norm whose
    /// epilogue writes the whole iteration space), `None` for the default span.
    ///
    /// # Errors
    ///
    /// As [`Self::replay_output`].
    pub fn replay_output_elements(&self, elements: Option<u64>) -> Result<Vec<u8>, MetalError> {
        let output_index = self
            .bindings
            .iter()
            .position(|binding| matches!(binding, Binding::Output(_)))
            .ok_or_else(|| MetalError::CompileFailed {
                log: "captured dispatch has no output binding".to_string(),
            })?;
        let (_, buffer, offset) = self
            .buffers
            .iter()
            .find(|(index, _, _)| *index == output_index)
            .ok_or_else(|| MetalError::CompileFailed {
                log: "captured dispatch output buffer was not recoverable".to_string(),
            })?;
        let available = buffer.length().saturating_sub(*offset);
        let iteration_elements = self.extents.iter().product::<u64>();
        let output_elements = elements.unwrap_or_else(|| {
            self.uniform_bytes
                .first_chunk::<8>()
                .map(|bytes| i64::from_le_bytes(*bytes))
                .and_then(|total| u64::try_from(total).ok())
                .filter(|total| *total > 0 && *total <= iteration_elements)
                .unwrap_or(iteration_elements)
        });
        let span = output_elements
            .saturating_mul(core::mem::size_of::<f32>() as u64)
            .try_into()
            .map_or(available, |bytes: usize| bytes.min(available));
        let region_start = buffer.contents().as_ptr().cast::<u8>();
        // SAFETY: shared-storage buffer of `length()` bytes, idle between
        // the synchronous replays; `offset + span <= length()` by `available`.
        unsafe { core::ptr::write_bytes(region_start.add(*offset), REPLAY_POISON_BYTE, span) };
        self.time_gpu_ns(1)?;
        // SAFETY: same buffer and span, read after the replay completed.
        let region = unsafe { core::slice::from_raw_parts(region_start.add(*offset), span) };
        Ok(region.to_vec())
    }

    /// GPU time in nanoseconds (`GPUEndTime - GPUStartTime`) of ONE command
    /// buffer holding `batch` back-to-back copies of this dispatch in one
    /// serial encoder over the captured buffers. `batch == 1` is the
    /// isolated-replay number, floor included; the marginal per-dispatch
    /// cost is `(time(k) - time(1)) / (k - 1)`.
    ///
    /// # Errors
    ///
    /// [`MetalError::CompileFailed`] when the dispatch is
    /// [`Self::unreplayable`], the device refuses an allocation, or the
    /// command buffer ends in an error status.
    pub fn time_gpu_ns(&self, batch: usize) -> Result<f64, MetalError> {
        if let Some(reason) = &self.unreplayable {
            return Err(MetalError::CompileFailed { log: reason.clone() });
        }
        let (device, queue) = device_and_queue()?;
        let uniforms = shared_buffer_from(&device, &self.uniform_bytes)?;
        let fault = shared_buffer_from(&device, &[0u8; FAULT_REPLAY_BYTES])?;
        let command_buffer = queue.commandBuffer().ok_or_else(|| MetalError::CompileFailed {
            log: "queue refused a replay command buffer".to_string(),
        })?;
        let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| MetalError::CompileFailed {
            log: "command buffer refused a replay encoder".to_string(),
        })?;
        for _ in 0..batch {
            encoder.setComputePipelineState(&self.pipeline);
            for (index, buffer, offset) in &self.buffers {
                // SAFETY: the captured buffer and offset are the exact pair `encode_op` bound.
                unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), *offset, *index) };
            }
            if let Some(index) = self.uniforms_index {
                unsafe { encoder.setBuffer_offset_atIndex(Some(&uniforms), 0, index) };
            }
            if let Some(index) = self.fault_index {
                unsafe { encoder.setBuffer_offset_atIndex(Some(&fault), 0, index) };
            }
            dispatch(&encoder, &self.pipeline, self.grid);
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        gpu_span_ns(&command_buffer)
    }

    /// GPU time of `items` in the given order in one command buffer, for
    /// comparing the census replay with a live step; `items` is the whole
    /// captured step or a subset of it (one kernel family).
    ///
    /// # Errors
    ///
    /// Returns the first captured unreplayable reason, or a Metal error when
    /// buffer allocation or command execution fails.
    pub fn time_gpu_sequence_ns<Item: Borrow<Self>>(items: &[Item]) -> Result<f64, MetalError> {
        let dispatches: Vec<&Self> = items.iter().map(Borrow::borrow).collect();
        if dispatches.is_empty() {
            return Ok(0.0);
        }
        if let Some(reason) = dispatches
            .iter()
            .find_map(|dispatch| dispatch.unreplayable.as_ref())
        {
            return Err(MetalError::CompileFailed { log: reason.clone() });
        }
        let (device, queue) = device_and_queue()?;
        let uniforms: Vec<MetalBuffer> = dispatches
            .iter()
            .map(|dispatch| shared_buffer_from(&device, &dispatch.uniform_bytes))
            .collect::<Result<_, _>>()?;
        let fault = shared_buffer_from(&device, &[0u8; FAULT_REPLAY_BYTES])?;
        let command_buffer = queue.commandBuffer().ok_or_else(|| MetalError::CompileFailed {
            log: "queue refused a sequence replay command buffer".to_string(),
        })?;
        let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
            MetalError::CompileFailed {
                log: "command buffer refused a sequence replay encoder".to_string(),
            }
        })?;
        for (dispatch_record, uniforms_buffer) in dispatches.iter().zip(&uniforms) {
            encoder.setComputePipelineState(&dispatch_record.pipeline);
            for (index, buffer, offset) in &dispatch_record.buffers {
                // SAFETY: each captured buffer and offset is the pair `encode_op` bound.
                unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), *offset, *index) };
            }
            if let Some(index) = dispatch_record.uniforms_index {
                unsafe { encoder.setBuffer_offset_atIndex(Some(uniforms_buffer), 0, index) };
            }
            if let Some(index) = dispatch_record.fault_index {
                unsafe { encoder.setBuffer_offset_atIndex(Some(&fault), 0, index) };
            }
            dispatch(&encoder, &dispatch_record.pipeline, dispatch_record.grid);
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        gpu_span_ns(&command_buffer)
    }

    /// GPU spans for the captured chunks in order, with one command buffer
    /// per original chunk boundary.
    ///
    /// # Errors
    ///
    /// Returns a captured unreplayable reason or a Metal allocation or
    /// command-execution error.
    pub fn time_gpu_chunk_sequences_ns(dispatches: &[Self]) -> Result<Vec<f64>, MetalError> {
        let mut spans = Vec::new();
        let mut chunk_start = 0usize;
        while chunk_start < dispatches.len() {
            let chunk_index = dispatches[chunk_start].chunk_index;
            let mut chunk_end = chunk_start + 1;
            while chunk_end < dispatches.len()
                && dispatches[chunk_end].chunk_index == chunk_index
            {
                chunk_end += 1;
            }
            if chunk_end < dispatches.len()
                && dispatches[chunk_end].chunk_index < chunk_index
            {
                return Err(MetalError::CompileFailed {
                    log: "captured chunk indices are not ordered".to_string(),
                });
            }
            spans.push(Self::time_gpu_sequence_ns(
                &dispatches[chunk_start..chunk_end],
            )?);
            chunk_start = chunk_end;
        }
        Ok(spans)
    }
}

/// GPU time of one command buffer whose only work is a one-thread no-op
/// kernel -- the per-command-buffer floor every isolated replay pays and the
/// in-situ shared encoder does not.
#[cfg(feature = "instrument")]
pub fn time_empty_command_buffer_gpu_ns() -> Result<f64, MetalError> {
    let (device, queue) = device_and_queue()?;
    let source = "#include <metal_stdlib>\nusing namespace metal;\nkernel void m0_empty(device float* sink [[buffer(0)]], uint gid [[thread_position_in_grid]]) { }\n";
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&MTLCompileOptions::new()))
        .map_err(|error| MetalError::CompileFailed { log: error.localizedDescription().to_string() })?;
    let function = library.newFunctionWithName(&NSString::from_str("m0_empty")).ok_or_else(|| {
        MetalError::CompileFailed { log: "m0_empty entry missing".to_string() }
    })?;
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| MetalError::CompileFailed { log: error.localizedDescription().to_string() })?;
    let sink = shared_buffer_from(&device, &[0u8; 16])?;
    let command_buffer = queue.commandBuffer().ok_or_else(|| MetalError::CompileFailed {
        log: "queue refused a floor command buffer".to_string(),
    })?;
    let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| MetalError::CompileFailed {
        log: "command buffer refused a floor encoder".to_string(),
    })?;
    encoder.setComputePipelineState(&pipeline);
    unsafe { encoder.setBuffer_offset_atIndex(Some(&sink), 0, 0) };
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: 1, height: 1, depth: 1 },
    );
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    gpu_span_ns(&command_buffer)
}

/// Streams `bytes` of blit-fill through the memory system in its own
/// command buffer, untimed, so the next replay's weights come from DRAM
/// rather than the system-level cache a back-to-back replay would warm.
#[cfg(feature = "instrument")]
pub fn flush_gpu_caches(bytes: usize) -> Result<(), MetalError> {
    let (device, queue) = device_and_queue()?;
    let scratch = FLUSH_SCRATCH.with(|slot| -> Result<MetalBuffer, MetalError> {
        let mut slot = slot.borrow_mut();
        if let Some(existing) = slot.as_ref().filter(|buffer| buffer.length() >= bytes) {
            return Ok(existing.clone());
        }
        let buffer = device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| MetalError::CompileFailed {
                log: "device refused the cache-flush buffer".to_string(),
            })?;
        *slot = Some(buffer.clone());
        Ok(buffer)
    })?;
    let command_buffer = queue.commandBuffer().ok_or_else(|| MetalError::CompileFailed {
        log: "queue refused a flush command buffer".to_string(),
    })?;
    let encoder = command_buffer.blitCommandEncoder().ok_or_else(|| MetalError::CompileFailed {
        log: "command buffer refused a flush blit encoder".to_string(),
    })?;
    encoder.fillBuffer_range_value(
        &scratch,
        objc2_foundation::NSRange { location: 0, length: bytes },
        0xA5,
    );
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    gpu_span_ns(&command_buffer).map(|_| ())
}

#[cfg(feature = "instrument")]
thread_local! {
    static FLUSH_SCRATCH: RefCell<Option<MetalBuffer>> = const { RefCell::new(None) };
}

/// [`chunk_audit_record_dispatch`]'s per-dispatch feed: walks the SAME
/// `bindings` list [`bind_buffers`] just bound (this call site, right after
/// that call succeeds, mirrors [`capture_dispatch`]'s own enumeration), so
/// this dispatch's audit record can never see a different binding set than
/// what actually got encoded. `Binding::Input`/`Binding::Indices` are reads;
/// `Binding::Output` is this op's own write, resolved separately from
/// `output` (there is no `Binding::Output(node)` payload to look up -- the
/// bound op's output is always `bound.node` itself). Length for a resolved
/// input is the buffer's own remaining length from `offset` -- a
/// deliberately conservative (over-inclusive) range, since a missed real
/// hazard is worse than a false-positive one this audit's own log lets a
/// reader dismiss by inspection.
#[cfg(feature = "instrument")]
fn record_chunk_audit_dispatch(
    bound: &BoundOp,
    bindings: &[Binding],
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    output: (&MetalBuffer, usize),
) {
    let position = bound.node.0 as usize;
    let output_len = bound_output_len(bound).max(1) * bound.dtype.size_bytes();
    chunk_audit_record_dispatch(
        Retained::as_ptr(output.0) as usize,
        output.1,
        output_len,
        position,
        true,
    );
    for binding in bindings {
        let node = match binding {
            Binding::Input(node) | Binding::Indices(node) => *node,
            _ => continue,
        };
        if let Some((buffer, offset)) = device_buffers.get(&node) {
            let remaining = buffer.length().saturating_sub(*offset);
            chunk_audit_record_dispatch(
                Retained::as_ptr(buffer) as usize,
                *offset,
                remaining,
                position,
                false,
            );
        }
    }
}

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
    #[cfg(feature = "instrument")]
    capture_chunk_index: usize,
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
        let (binding_identity, grid) = kernel_dispatch_shape(bound, packed_operands, numeric_policy)?;
        let mut cache_key = kernel_cache_key_for_grid(bound, packed_operands, numeric_policy, &grid)?;
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
            cache_key.push_str(crate::msl::codec_cache_token(codec));
            if std::env::var_os("PROXIMA_DEBUG_EXPERT_EMIT").is_some() {
                eprintln!("expert lowering mode=uniform codec={codec:?} node={source_node:?}");
            }
        } else {
            cache_key.push_str("_mixed_expert");
            if std::env::var_os("PROXIMA_DEBUG_EXPERT_EMIT").is_some() {
                eprintln!("expert lowering mode=mixed node={source_node:?}");
            }
        }
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
        let (bindings, grid) = kernel_dispatch_shape(bound, packed_operands, numeric_policy)?;
        let mut cache_key = kernel_cache_key_for_grid(bound, packed_operands, numeric_policy, &grid)?;
        cache_key.push(math_mode.cache_token());
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
                let merge_cache_key = merge_pipeline_key(&cache_key, &merge_kernel);
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
    // A `RoundBatchedReduce` dispatch never uses the generic placement/fresh
    // -allocation output below: it writes `round_count` rounds into ONE
    // contiguous buffer the spliced kernel addresses via `round_gid.z`
    // (`splice_round_batched_reduce_base_table`'s own doc), so round 0's own
    // output slot must be THIS group's own buffer at offset `0`, never a
    // caller placement or a bare `bound_output_len(bound)`-sized allocation
    // sized for round 0 alone. The routes need no such handling: each round's
    // own route buffer is bound through its own trailing `Binding::Indices`
    // (`bindings`'s doc), so `bind_buffers` and the hazard walk see them.
    #[cfg(feature = "metal-moe-mul-mat-id")]
    let round_group: Option<ResolvedRoundGroup> =
        if matches!(bound.kind, BoundOpKind::RoundBatchedReduce { .. }) {
            Some(ensure_round_group_resolved(device, bound)?)
        } else {
            None
        };
    #[cfg(not(feature = "metal-moe-mul-mat-id"))]
    let round_group: Option<()> = None;
    let (output, output_offset) = match &round_group {
        #[cfg(feature = "metal-moe-mul-mat-id")]
        Some(group) => (group.output_buffer.clone(), 0),
        _ => match placement {
            Some((buffer, offset)) => (buffer.clone(), offset),
            None => (
                allocate_buffer(device, bound_output_len(bound), bound.dtype)?,
                0,
            ),
        },
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
    #[cfg(feature = "instrument")]
    if chunk_audit_enabled() {
        record_chunk_audit_dispatch(bound, bindings, device_buffers, (&output, output_offset));
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
        // the recurrent-routed family's routed-expert + GDN decode step, discarding state
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
    if matches!(&bound.kind, BoundOpKind::MoeTopK { .. }) {
        let token_count = bound.extents.iter().product::<u64>() as usize;
        let extra_outputs = bound.kind.moe_topk_extra_outputs(token_count);
        for (offset, (extra_node, element_count)) in extra_outputs.iter().enumerate() {
            let buffer_index = bindings.len() + offset;
            let existing = device_buffers.get(extra_node).cloned();
            let (extra_buffer, extra_offset) = match existing {
                Some(buffer) => buffer,
                None => (allocate_buffer(device, *element_count, bound.dtype)?, 0),
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
    // `render_cached_softmax_weights`'s own three extra outputs (buffers 5,
    // 6, 7 -- `bindings.len()` is already 5: three operand inputs, the
    // primary output, the uniforms slot): same "resolve from `device_
    // buffers`, or allocate fresh" shape as `MoeTopK`'s own arm just above,
    // but each buffer's real element count (not a placeholder `1`) --
    // `cached_weight_sum`/`new_weight_sum` are one value per attention row,
    // `new_attended` is `attention_rows * head_dim` (this op's own doc).
    if let BoundOpKind::CachedSoftmaxWeights {
        cached_weight_sum,
        new_weight_sum,
        new_attended,
        attention_rows,
        head_dim,
        ..
    } = &bound.kind
    {
        let extra_nodes_and_counts: [(NodeId, usize); 3] = [
            (*cached_weight_sum, *attention_rows as usize),
            (*new_weight_sum, *attention_rows as usize),
            (*new_attended, (*attention_rows * *head_dim) as usize),
        ];
        for (offset, (extra_node, element_count)) in extra_nodes_and_counts.iter().enumerate() {
            let buffer_index = bindings.len() + offset;
            let existing = device_buffers.get(extra_node).cloned();
            let (extra_buffer, extra_offset) = match existing {
                Some(buffer) => buffer,
                None => (allocate_buffer(device, *element_count, bound.dtype)?, 0),
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
    // `splice_round_batched_reduce_base_table`'s own `round_table [[buffer(N)]]`
    // parameter, bound OUTSIDE `bindings` at the exact slot the splice's own
    // MSL text names (`kernel.bindings.len()` at splice time, i.e.
    // `bindings.len()` here since splicing never mutates that `Vec`) -- the
    // same "one binding with no `NodeId` of its own" shape
    // `splice_horizontal_merge_base_table`'s own `base_table` already uses.
    // `round_outputs[1..]` are registered into `device_buffers` at their own
    // `(output_buffer, round * output_member_bytes)` slot, mirroring
    // `MoeTopK`'s own extra-output arm just above, so a later op reading a
    // non-leader round's own output finds the right buffer/offset.
    #[cfg(feature = "metal-moe-mul-mat-id")]
    if let (Some(group), BoundOpKind::RoundBatchedReduce { round_outputs, .. }) =
        (&round_group, &bound.kind)
    {
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&group.round_table), 0, bindings.len());
        }
        if let Some(tracker) = hazard.as_deref_mut() {
            tracker.record(&[], Some(Retained::as_ptr(&group.output_buffer)));
        }
        for (round, round_node) in round_outputs.iter().enumerate().skip(1) {
            device_buffers.insert(
                *round_node,
                (group.output_buffer.clone(), round * group.output_member_bytes),
            );
        }
    }
    #[cfg(feature = "instrument")]
    capture_dispatch(
        bound,
        packed_operands,
        &pipeline,
        grid,
        bindings,
        device_buffers,
        (&output, output_offset),
        scratch,
        capture_chunk_index,
        &uniforms,
    );
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
        #[cfg(feature = "instrument")]
        capture_dispatch(
            bound,
            packed_operands,
            &merge.pipeline,
            merge.grid,
            &merge.bindings,
            device_buffers,
            (&output, output_offset),
            scratch,
            capture_chunk_index,
            &merge_uniforms,
        );
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
    let Some((slot, recorded)) = fault_slots(fault_buffer, gather_count)
        .iter()
        .copied()
        .enumerate()
        .find(|(_, recorded)| *recorded != 0)
    else {
        return Ok(());
    };
    if recorded & 0x8000_0000 != 0 {
        let encoded_expert = recorded & 0x7fff_ffff;
        return Err(MetalError::ExpertSourceMiss {
            node: bound.node,
            expert: encoded_expert.saturating_sub(1),
        });
    }
    let lookup = bound
        .operands()
        .iter()
        .filter_map(|(_, _, gather)| gather.as_ref())
        .nth(slot)
        .ok_or_else(|| MetalError::CompileFailed {
            log: format!(
                "node {:?} recorded a gather fault in slot {slot} but has no gather operand there",
                bound.node
            ),
        })?;
    Err(TensorError::GatherIndexOutOfRange {
        node: bound.node,
        index: i64::from(recorded - 1),
        extent: lookup.extent,
    }
    .into())
}

/// Checks every dispatch's fault buffer after the step's command buffers have
/// completed, then returns each to [`allocate_fault_buffer`]'s pool.
pub(super) fn check_pending_faults(pending_faults: Vec<PendingFault<'_>>) -> Result<(), MetalError> {
    for (bound, fault_buffer, gathers) in pending_faults {
        check_gather_fault(bound, &fault_buffer, gathers)?;
        recycle_fault_buffer(fault_buffer, gathers);
    }
    Ok(())
}

pub(super) fn fault_slots(buffer: &ProtocolObject<dyn MTLBuffer>, gather_count: usize) -> &[u32] {
    let pointer = buffer.contents();
    // SAFETY: allocated and sized to at least `gather_count` `u32`s by
    // `allocate_fault_buffer`, `storageModeShared` so CPU-visible now that
    // `waitUntilCompleted` has returned; the slice borrows `buffer`, which
    // outlives every use of it.
    unsafe { core::slice::from_raw_parts(pointer.as_ptr().cast::<u32>(), gather_count.max(1)) }
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
        let dtype = gpu_dtype(program, index_nodes, &plan.prepared.resolved, *node);
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
    // attn_parity followon (2026-09-22): `PROXIMA_REPEAT_VERIFY`'s own
    // post-wait byte compare -- both the original and every copy are
    // already `effective_outputs`, so their bytes are already sitting in
    // `results` from the loop above; no second `read_back` pass needed.
    // `bytes_match`/`max_abs_diff` prove the duplicate computed the same
    // value the production dispatch did; `sentinel_survived` (every element
    // still the quiet-NaN fill pattern the pre-dispatch sentinel-fill wrote,
    // `placements_execute_named::head_debug_fill_sentinel`, reused
    // unmodified) would mean the copy's own dispatch never actually wrote
    // its buffer -- a false pass this check exists to catch.
    // marginal warm-repeat latency harness (owner brief 2026-09-22): labels
    // below are deliberately `marginal_warm_repeat_ms`-shaped, never a
    // per-kernel "cost" -- see `run_series.sh`'s own header.
    // `original` (the target node `PROXIMA_REPEAT_NODES` named) is usually
    // an INTERNAL node -- read by a later op, never itself a requested
    // output -- so it is generally absent from `results`/`effective_outputs`
    // even though its buffer is still live in `device_buffers` (every
    // resolved node's output lands in the arena regardless of whether a
    // caller asked to read it back). Each copy IS a requested output
    // (`apply_repeat_nodes` pushes it), so `results` already has its bytes;
    // the original still needs its own direct `read_back`.
    #[cfg(feature = "instrument")]
    if std::env::var_os("PROXIMA_REPEAT_VERIFY").is_some() {
        static STEP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let step = STEP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        for (original, copies) in &plan.prepared.repeat_verify_pairs {
            let Some((original_buffer, original_offset)) = device_buffers.get(original) else {
                continue;
            };
            let original_shape = shapes.of(*original).to_vec();
            let original_dtype = gpu_dtype(program, index_nodes, &plan.prepared.resolved, *original);
            let Ok(original_bytes) = read_back(
                original_buffer,
                *original_offset,
                element_count(&original_shape),
                *original,
                original_dtype,
            ) else {
                continue;
            };
            let ptr_orig = Retained::as_ptr(original_buffer) as u64;
            for copy in copies {
                let Some((_, _, copy_bytes)) = results.iter().find(|(node, _, _)| node == copy)
                else {
                    continue;
                };
                let ptr_copy = device_buffers
                    .get(copy)
                    .map_or(0u64, |(buffer, _)| Retained::as_ptr(buffer) as u64);
                let max_abs_diff = original_bytes
                    .iter()
                    .zip(copy_bytes.iter())
                    .map(|(reference, candidate)| (reference - candidate).abs())
                    .fold(0.0_f32, f32::max);
                let bytes_match = original_bytes.len() == copy_bytes.len() && max_abs_diff == 0.0;
                let sentinel_survived = !copy_bytes.is_empty() && copy_bytes.iter().all(|value| value.is_nan());
                std::eprintln!(
                    "repeat_verify step={step} node={} copy={} bytes_match={bytes_match} sentinel_survived={sentinel_survived} max_abs_diff={max_abs_diff} ptr_orig={ptr_orig} ptr_copy={ptr_copy}",
                    original.0,
                    copy.0,
                );
            }
        }
    }
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
        BTreeSet, NodeId, Codec, PackedOperands, device_and_queue, element_count,
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
        packed_operands.insert(NodeId(0), Codec::Q4K);

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
        packed_operands.insert(NodeId(0), Codec::Q5K);

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
    /// recurrent-routed-shaped grouped gate/up dispatch's `operand_bytes` by
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
        packed_operands.insert(weight_node, Codec::Q4K);

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
        packed_operands.insert(NodeId(0), Codec::Q4K);
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
            QuantizedBlock::Packed { codec: proxima_primitives::Codec::Q6K, bytes: &packed_weight },
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
            QuantizedBlock::Packed { codec: proxima_primitives::Codec::Q6K, bytes: &packed_weight },
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
    use super::record_hazard_class;
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
    /// Asserts BOTH on `classify`'s own returned [`HazardClass`] (a local
    /// value) AND on the process-global `BARRIERS_*` counters
    /// [`record_hazard_class`] feeds -- the counters are shared statics real
    /// dispatch code and other concurrently running `cargo test` threads also
    /// increment, so each is read via `.get()` BEFORE this test's own two
    /// `record_hazard_class` calls and again AFTER, asserting the DELTA is at
    /// least this test's own contribution (`>=`, never `==`) -- a concurrent
    /// writer can only add to the delta, never subtract, so `>=` is sound
    /// under a shared parallel suite. `BARRIERS_WAW`/`BARRIERS_WAW_WAR_
    /// PERSISTENT` are NOT asserted to stay at exactly this test's own zero
    /// contribution: an unrelated concurrently running test genuinely can
    /// fire a WAW or persistent-identity barrier in the same window, and
    /// there is no sound way to distinguish "this test caused zero" from "a
    /// concurrent test's own nonzero landed in this window" from outside
    /// that test's own scope -- asserting `== 0` here would be exactly the
    /// exact-value-against-shared-state defect this whole fix removes.
    #[cfg(feature = "instrument")]
    #[test]
    fn hazard_class_counters_attribute_one_raw_and_one_arena_reuse_war() {
        let mut hazards: HazardTracker<&str> = HazardTracker::new();

        // op0: writes `a`, no inputs -> no hazard, nothing to attribute.
        let class0 = hazards.classify(&[], Some("a"));
        assert_eq!(class0, HazardClass::None);
        assert!(!hazard_step(&mut hazards, &[], "a"));

        let raw_before = super::BARRIERS_RAW.get();
        let war_before = super::BARRIERS_WAR.get();
        let arena_recycled_before = super::BARRIERS_WAW_WAR_ARENA_RECYCLED.get();

        // op1: reads `a` (written by op0 since the last barrier) and writes
        // `b` -> RAW, a genuine dataflow edge, never arena-attributed.
        let class1 = hazards.classify(&["a"], Some("b"));
        assert_eq!(class1, HazardClass::Raw, "op1 contributed the one RAW barrier");
        assert!(hazard_step(&mut hazards, &["a"], "b"));
        record_hazard_class(class1, false);

        // op2: no inputs, output placed by the arena back into `a` -- `a`
        // was READ by op1 since the last barrier (op1's own RAW reset the
        // tracker first), so this is a WAR hazard on a recycled arena slot.
        let class2 = hazards.classify(&[], Some("a"));
        assert_eq!(
            class2,
            HazardClass::War,
            "op2's WAR fired on the arena-recycled `a` slot"
        );
        assert!(hazard_step(&mut hazards, &[], "a"));
        record_hazard_class(class2, true);

        assert!(
            super::BARRIERS_RAW.get() - raw_before >= 1,
            "op1 must have contributed at least one RAW barrier to the shared counter"
        );
        assert!(
            super::BARRIERS_WAR.get() - war_before >= 1,
            "op2 must have contributed at least one WAR barrier to the shared counter"
        );
        assert!(
            super::BARRIERS_WAW_WAR_ARENA_RECYCLED.get() - arena_recycled_before >= 1,
            "op2's WAR must have contributed at least one arena-recycled attribution"
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
    #[cfg(feature = "metal-attn-split-rows")]
    use super::{scratch_is_shared, shared_scratch_elements};

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

    /// One E2B checkpoint attention op as speculative verify binds it: eight query
    /// groups on one kv head, `rows` new rows over the `cached_key_rows`
    /// bucket, the sliding window (head_dim 256) or the full range (512).
    #[cfg(feature = "metal-attn-split-rows")]
    fn gemma_verify_bound(head_dim: u64, cached_key_rows: u64, rows: u64) -> BoundOp {
        const GROUPS: u64 = 8;
        let operands = (0..9)
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
            node: NodeId(9),
            dtype: DType::Float32,
            extents: vec![rows, 1, GROUPS, head_dim],
            kind: BoundOpKind::CachedAttention {
                operands,
                query_rows: rows,
                cached_key_rows,
                new_key_rows: rows,
                kv_heads: 1,
                query_groups: GROUPS,
                head_dim,
                rotary_dim: head_dim,
                scale: 1.0,
                cached_lower_inclusive: if head_dim == 256 { -511 } else { i64::MIN },
                new_upper_inclusive: 0,
            },
        }
    }

    /// The row-tiled partial and the merge stride the scratch by the form's own
    /// bind-time `splits`; `splits_for` would size it by a different divisor.
    #[cfg(feature = "metal-attn-split-rows")]
    #[test]
    fn row_tiled_scratch_is_sized_by_the_forms_own_split_count() {
        let policy = NumericPolicy::llama_relaxed();
        let bound = gemma_verify_bound(512, 1632, 49);
        let elements = cached_attention_scratch_len(&bound, policy)
            .expect("a CachedAttention bound always yields a scratch length");

        let splits = crate::msl::cached_attention_live_splits(&bound.kind, policy);
        assert_eq!(
            splits, 2,
            "25 row tiles at 2 rows each, 1681 keys: 32 threadgroups over 25 tiles -> 2 splits"
        );
        assert_eq!(elements, 49 * 8 * splits * (2 + 512));
        assert_ne!(
            splits,
            crate::msl::splits_for(1632 + 49, policy),
            "the generic rule gives another count, which would mis-stride the merge"
        );
        assert_eq!(
            elements * 4,
            1_611_904,
            "1.61 MB of scratch for the widest op"
        );
    }

    /// 35 attention positions of one verify plan (7 global, 28 sliding) share
    /// one scratch buffer sized at the widest, where one buffer per position
    /// would reserve 45.3 MB.
    #[cfg(feature = "metal-attn-split-rows")]
    #[test]
    fn the_row_tiled_positions_of_a_plan_share_one_scratch_sized_at_the_widest() {
        let policy = NumericPolicy::llama_relaxed();
        let mut resolved = Vec::new();
        for layer in 0..35 {
            let global = layer % 5 == 4;
            resolved.push(if global {
                gemma_verify_bound(512, 1632, 49)
            } else {
                gemma_verify_bound(256, 512, 49)
            });
        }
        let global_count = resolved
            .iter()
            .filter(|bound| {
                matches!(
                    &bound.kind,
                    BoundOpKind::CachedAttention { head_dim: 512, .. }
                )
            })
            .count();
        assert_eq!((global_count, resolved.len() - global_count), (7, 28));

        let sharing = resolved
            .iter()
            .filter(|bound| scratch_is_shared(bound, policy))
            .count();
        assert_eq!(sharing, 35, "every row-tiled position shares");

        let per_position: u64 = resolved
            .iter()
            .filter_map(|bound| cached_attention_scratch_len(bound, policy))
            .sum();
        let shared = shared_scratch_elements(&resolved, policy)
            .expect("35 sharing positions need one buffer");
        assert_eq!(
            shared, 402_976,
            "the widest op: 392 vectors x 2 splits x 514 floats"
        );
        assert_eq!(per_position, 11_316_256, "7 x 1.61 MB + 28 x 1.21 MB");
        assert_eq!(per_position * 4, 45_265_024);
    }

    /// The single-row decode keeps its per-position reservation at the compiled
    /// maximum split count, so nothing is shared there.
    #[cfg(feature = "metal-attn-split-rows")]
    #[test]
    fn decode_split_positions_keep_their_own_scratch() {
        let policy = NumericPolicy::llama_relaxed();
        let decode = gemma_verify_bound(256, 512, 1);
        assert!(!scratch_is_shared(&decode, policy));
        assert_eq!(
            shared_scratch_elements(&[decode.clone(), decode.clone()], policy),
            None
        );
        assert_eq!(
            cached_attention_scratch_len(&decode, policy),
            Some(8 * crate::sized::ATTENTION_SPLIT_MAX * (2 + 256))
        );
    }
}

/// Production crash (2026-09-07): `provider error: generate: arena
/// peak_bytes=984110552 exceeds arena_transient_cap=172812125` -- a
/// ~1100-row interactive-chat prefill rejected by a cap sized once for
/// decode (`query_rows == 1`) and never scaled for a prefill's own row
/// count. Pure CPU -- [`plan_bind_row_count`] never touches a device -- so
/// these run on any host, without a GPU.
#[cfg(all(test, feature = "metal-plan-stable-buffers"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(super) mod plan_query_rows_tests {
    use alloc::vec;

    use proxima_tensor::{BoundOpKind, DType, Layout, NodeId};

    use super::{ARENA_TRANSIENT_CAP, BoundOp, plan_bind_row_count};

    /// The decode step: exactly one new token this call, `symbols[0] == 1`
    /// (`residency_caches.rs`'s own `new_count` convention).
    #[test]
    fn decode_step_bind_row_count_is_one() {
        assert_eq!(plan_bind_row_count(&[1, 4096]), 1);
    }

    /// A caller with no `symbols` at all (a bare `Op` unit test, never a
    /// real `plan()`/`plan_named()` call) falls back to `1` -- decode's own
    /// row count, the same direction [`Plan::command_buffer_chunks_decode_shaped`]'s
    /// own `false` default already takes.
    #[test]
    fn symbols_less_caller_defaults_to_one_row() {
        assert_eq!(plan_bind_row_count(&[]), 1);
    }

    /// Production crash (2026-09-29): `OFF decode: Backend(Metal(
    /// ArenaOverCap { peak_bytes: 495315984, cap_bytes: 172812125,
    /// query_rows: 1, device_limit: 51539607552 }))` --
    /// `speculative_decode_parity --gpu-layers all` against the E2B checkpoint on
    /// the 1618-token `rag001` prompt (`proxima-model-interop/examples/
    /// data/speculative_corpus.jsonl`). The E2B checkpoint's attention lowers to a
    /// plain elementwise/reduce decomposition on a build without
    /// `metal-fuse-attn-decode` -- ZERO `CachedAttention` nodes in the bound
    /// graph -- so a scan over `resolved` for a `CachedAttention`-shaped row
    /// count could never see this plan's true row count and fell back to
    /// `1`, reproducing exactly the panic above. `symbols[0] == 1618` is the
    /// caller's own `new_count` for this prefill (every one of the 1618
    /// prompt tokens is a "new" token on the cold call) -- the bind-time
    /// value this fix reads instead of scanning bound-op shapes.
    #[test]
    fn gemma4_e2b_prefill_admits_its_true_peak_at_the_bind_time_row_count() {
        let query_rows = plan_bind_row_count(&[1618, 0]);
        assert_eq!(
            query_rows, 1618,
            "plan_bind_row_count must read the caller's own new_count, not a scan over bound ops"
        );

        let cap_bytes = ARENA_TRANSIENT_CAP.saturating_mul(query_rows as usize);
        let production_peak_bytes = 495_315_984_usize;
        let production_cap_bytes_before_fix = 172_812_125_usize;
        assert_eq!(
            ARENA_TRANSIENT_CAP, production_cap_bytes_before_fix,
            "this reproduction only matches the incident if ARENA_TRANSIENT_CAP is still the \
             constant the panic itself reported as cap_bytes at the pre-fix query_rows=1"
        );
        assert!(
            production_peak_bytes > production_cap_bytes_before_fix,
            "this reproduction is only meaningful if the OLD unscaled cap would have rejected \
             it, matching the original panic"
        );
        assert!(
            production_peak_bytes < cap_bytes,
            "the real incident's own peak_bytes must fit under the row-scaled cap: \
             peak_bytes={production_peak_bytes} cap_bytes={cap_bytes}"
        );
    }

    /// The fix must not weaken MG-3's own reject side: a plan whose real
    /// peak still exceeds its OWN row-scaled cap (the same `query_rows=1618`
    /// this rag001 prefill reports) must still fail `build_buffer_arena`'s
    /// `peak_bytes > cap_bytes` check. Only how `query_rows` is derived
    /// changed; the comparison itself did not.
    #[test]
    fn oversized_peak_at_the_same_row_count_is_still_rejected() {
        let query_rows = plan_bind_row_count(&[1618, 0]);
        let cap_bytes = ARENA_TRANSIENT_CAP.saturating_mul(query_rows as usize);
        let oversized_peak_bytes = cap_bytes + 1;
        assert!(
            oversized_peak_bytes > cap_bytes,
            "a plan whose peak exceeds its own row-scaled cap must still trip the reject \
             condition: peak_bytes={oversized_peak_bytes} cap_bytes={cap_bytes}"
        );
    }

    /// `new_count` in {1 (decode), 31 (a short prefix), 1100 (ROW 391's own
    /// interactive-chat reproduction)} -- [`plan_bind_row_count`] reads the
    /// caller's own `symbols[0]` back unchanged, and the derived cap scales
    /// linearly with it (decode's `new_count == 1` reproduces the pre-fix
    /// constant exactly).
    #[test]
    fn cap_scales_linearly_with_the_callers_own_new_count() {
        for new_count in [1_u64, 31, 1100] {
            let observed = plan_bind_row_count(&[new_count, 0]);
            assert_eq!(
                observed, new_count,
                "plan_bind_row_count must read the caller's own new_count back exactly"
            );

            let cap_bytes = ARENA_TRANSIENT_CAP.saturating_mul(observed.max(1) as usize);
            let expected = ARENA_TRANSIENT_CAP * new_count as usize;
            assert_eq!(
                cap_bytes, expected,
                "cap_bytes must be exactly ARENA_TRANSIENT_CAP * new_count at new_count={new_count}"
            );
        }
    }

    /// ROW 391: the production shape itself -- at `query_rows=1100` the
    /// naive 984 MB peak from the incident report is now well inside the
    /// per-row-scaled cap, where the old fixed `ARENA_TRANSIENT_CAP`
    /// (172_812_125 bytes) rejected it outright.
    #[test]
    fn thousand_row_prefill_shape_fits_the_scaled_cap() {
        let query_rows = plan_bind_row_count(&[1100, 0]);
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

    /// The invariant this fix exists to hold: a bound op whose LEADING axis
    /// is something other than the token/row axis must never move the cap.
    /// A router-logits `Reduce` over 128 experts, or a stacked per-expert
    /// weight `Constant` (recurrent-routed-style MoE layers carry both shapes;
    /// this fixture is a constructed adversarial shape, not a captured
    /// trace, chosen to isolate the hazard), carries `extents[0] ==
    /// expert_count` with no relation whatsoever to how many tokens this
    /// call is processing. The OLD `resolved`-scanning `plan_query_rows`
    /// would have read this op's `128` as the row count and inflated the
    /// cap to `ARENA_TRANSIENT_CAP * 128` on a plain DECODE step
    /// (`new_count == 1`) -- silently widening MG-3's own guard rail far
    /// past what a single-token step should ever need, and admitting a
    /// peak that step's real cap should have rejected. Reading `symbols[0]`
    /// instead never looks at `resolved` at all, so this op's shape cannot
    /// move the cap in either direction.
    #[test]
    fn expert_count_leading_op_never_inflates_the_decode_cap() {
        const EXPERT_COUNT: u64 = 128;
        let stacked_expert_weight = BoundOp {
            node: NodeId(0),
            dtype: DType::Float32,
            extents: vec![EXPERT_COUNT, 4096, 1536],
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
        // decode: exactly one new token, wholly unrelated to EXPERT_COUNT.
        let query_rows = plan_bind_row_count(&[1, 0]);
        assert_eq!(
            query_rows, 1,
            "the decode step's own row count must stay 1 regardless of any op's shape"
        );

        let cap_bytes = ARENA_TRANSIENT_CAP.saturating_mul(query_rows as usize);
        let heuristic_cap_bytes =
            ARENA_TRANSIENT_CAP.saturating_mul(stacked_expert_weight.extents[0] as usize);
        assert!(
            cap_bytes < heuristic_cap_bytes,
            "the bind-time cap must stay far below what an extents[0]-scanning heuristic would \
             have inflated it to off this expert-count-leading op: cap_bytes={cap_bytes} \
             heuristic_cap_bytes={heuristic_cap_bytes}"
        );

        // a peak the OLD heuristic's inflated cap would have wrongly admitted,
        // but the true decode-shaped cap must still reject.
        let peak_between_the_two_caps = cap_bytes + 1;
        assert!(
            peak_between_the_two_caps <= heuristic_cap_bytes,
            "this peak must sit inside the window the OLD heuristic would have wrongly admitted"
        );
        assert!(
            peak_between_the_two_caps > cap_bytes,
            "and outside the window the bind-time row count correctly rejects"
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(super) mod expert_payload_descriptor_tests {
    use super::{
        BoundOp, BoundOpKind, ExpertPayloadDescriptor, MetalError, Codec,
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
                block: QuantizedBlock::Packed { codec: Codec::Q2K, bytes: &low_bytes },
                out_dim: 256,
                in_dim: 256,
                epoch: 11,
            },
            ExpertEntry {
                block: QuantizedBlock::Packed { codec: Codec::Q4K, bytes: &high_bytes },
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
                    codec: Codec::Q2K,
                    byte_offset: 0,
                    byte_length: 84,
                    out_dim: 256,
                    in_dim: 256,
                    epoch: 11,
                },
                ExpertPayloadDescriptor {
                    expert_index: 1,
                    codec: Codec::Q4K,
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
                block: QuantizedBlock::Packed { codec: Codec::Q2K, bytes: &low_bytes },
                out_dim: 256,
                in_dim: 256,
                epoch: 1,
            },
            ExpertEntry {
                block: QuantizedBlock::Packed { codec: Codec::Q4K, bytes: &high_bytes },
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
                block: QuantizedBlock::Packed { codec: Codec::Q4K, bytes: &low_bytes },
                out_dim: 256,
                in_dim: 256,
                epoch: 1,
            },
            ExpertEntry {
                block: QuantizedBlock::Packed { codec: Codec::Q6K, bytes: &high_bytes },
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
        assert_eq!(descriptors[1].codec, Codec::Q6K);
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
                block: QuantizedBlock::Packed { codec: Codec::Q4K, bytes: &first },
                out_dim: 256,
                in_dim: 256,
                epoch: 1,
            },
            ExpertEntry {
                block: QuantizedBlock::Packed { codec: Codec::Q6K, bytes: &second },
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
                block: QuantizedBlock::Packed { codec: Codec::Q2K, bytes: &arena[7..91] },
                out_dim: 256,
                in_dim: 256,
                epoch: 3,
            },
            ExpertEntry {
                block: QuantizedBlock::Packed { codec: Codec::Q2K, bytes: &arena[139..223] },
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
                block: QuantizedBlock::Packed { codec: Codec::Q4K, bytes: &first },
                out_dim: 256,
                in_dim: 256,
                epoch: 1,
            },
            ExpertEntry {
                block: QuantizedBlock::Packed { codec: Codec::Q4K, bytes: &second },
                out_dim: 256,
                in_dim: 256,
                epoch: 2,
            },
            ExpertEntry {
                block: QuantizedBlock::Packed { codec: Codec::Q4K, bytes: &third },
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
            block: QuantizedBlock::Packed { codec: Codec::Q3K, bytes: &q3_bytes },
            out_dim: 256,
            in_dim: 256,
            epoch: 0,
        }];
        let source = ExpertSource::new(&entries);

        let descriptors = expert_payload_descriptors(NodeId(9), &source)
            .expect("Q3_K has a mixed-expert MSL decoder");
        assert_eq!(descriptors[0].codec, Codec::Q3K);
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
            QuantizedBlock::Packed { codec: Codec::Q4K, bytes: &full_expert_stack },
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
                    QuantizedBlock::Packed { codec: Codec::Q4K, bytes } => bytes.len(),
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
                block: QuantizedBlock::Packed { codec: Codec::Q4K, bytes: &first_expert },
                out_dim: 256,
                in_dim: 256,
                epoch: 1,
            },
            ExpertEntry {
                block: QuantizedBlock::Packed { codec: Codec::Q4K, bytes: &second_expert },
                out_dim: 256,
                in_dim: 256,
                epoch: 2,
            },
        ];
        let source = ExpertSource::new(&entries);

        let error = reject_non_reducing_expert_staging(
            NodeId(51),
            QuantizedBlock::Packed { codec: Codec::Q4K, bytes: &original_bytes },
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
                block: QuantizedBlock::Packed { codec: Codec::Q2K, bytes: &low_expert },
                out_dim: 256,
                in_dim: 256,
                epoch: 1,
            },
            ExpertEntry {
                block: QuantizedBlock::Packed { codec: Codec::Q4K, bytes: &high_expert },
                out_dim: 256,
                in_dim: 256,
                epoch: 2,
            },
        ];
        let source = ExpertSource::new(&entries);

        reject_non_reducing_expert_staging(
            NodeId(52),
            QuantizedBlock::Packed { codec: Codec::Q4K, bytes: &original_bytes },
            &source,
        )
        .expect("a smaller mixed-codec table reduces the device upload");
    }
}

/// `{cache_key}_merge`, plus `_wg` when the merge kernel took the flat 2D grid
/// form. The split kernel's own key already names ITS form, but the merge
/// kernel's grid (rows x heads x `SIMD_WIDTH`) is narrower, so the two forms
/// can differ and must not share a pipeline entry.
fn merge_pipeline_key(cache_key: &str, merge_kernel: &Kernel) -> String {
    let form = if merge_kernel.grid.grid2d.is_some() { "_wg" } else { "" };
    format!("{cache_key}_merge{form}")
}

#[cfg(test)]
mod merge_pipeline_key_tests {
    use alloc::string::String;
    use alloc::vec::Vec;

    use super::merge_pipeline_key;
    use crate::msl::{Grid2DForm, Grid2DSpec, GridSpec, Kernel};

    fn merge_kernel(grid2d: Option<Grid2DSpec>) -> Kernel {
        Kernel {
            source: String::new(),
            entry: String::from("omega_cached_attention_merge"),
            bindings: Vec::new(),
            grid: GridSpec {
                threads: 1,
                threadgroup_width: None,
                depth: 1,
                grid2d,
            },
        }
    }

    #[test]
    fn a_merge_kernel_that_took_the_flat_form_never_shares_a_key_with_its_linear_sibling() {
        let flat = Grid2DSpec {
            form: Grid2DForm::FlatThreadgroupIndex,
            threadgroups_x: 4,
            threadgroups_y: 2,
            threads_per_threadgroup_x: 32,
            threads_per_threadgroup_y: 1,
            threadgroup_bytes: 0,
        };

        let linear_key = merge_pipeline_key("split_key", &merge_kernel(None));
        let flat_key = merge_pipeline_key("split_key", &merge_kernel(Some(flat)));

        assert_eq!(linear_key, "split_key_merge");
        assert_eq!(flat_key, "split_key_merge_wg");
    }
}
