use super::*;

/// Everything [`execute`] needs before touching a device — the same
/// judgments [`proxima_tensor::cpu::evaluate`]'s own `prepare` makes, rebuilt
/// here over the public API since that one is private to `cpu.rs`.
pub(super) struct Prepared {
    pub(super) root: NodeId,
    pub(super) shapes: Shapes,
    pub(super) effective_outputs: Vec<NodeId>,
    pub(super) block_nodes: Vec<NodeId>,
    /// Whether each positional block input is read by the pruned bound
    /// program or is itself an explicitly requested output. Inputs retained
    /// by the source graph but unreachable from those roots are not uploaded.
    pub(super) live_block_inputs: Vec<bool>,
    /// The single, ROW-327-fixed attribution of `block_nodes` to codecs --
    /// computed once here, by [`packed_operands_of`], off blocks already
    /// checked count- and shape-consistent against `block_nodes` (see this
    /// function's own doc). [`plan`] reuses this instead of recomputing it
    /// a second time against the same `blocks` argument.
    pub(super) packed_operands: PackedOperands,
    pub(super) resolved: Vec<BoundOp>,
    pub(super) retires: Vec<Vec<NodeId>>,
    /// [`node_last_reader`]'s dense table over `resolved` -- every retirement
    /// decision below is `last_reader[node] == position`, one array read, in
    /// place of a per-op forward scan over remaining ops.
    // `metal-buffer-pool`'s retirement loop already trusts `retires[position]`
    // unconditionally (no guard to replace), so this field has no reader
    // under that feature alone -- compiled out entirely rather than kept
    // with an allow.
    #[cfg(not(feature = "metal-buffer-pool"))]
    pub(super) last_reader: Vec<u32>,
    /// Every node referenced as a gather's `indices` anywhere in the
    /// program — see [`gpu_dtype`]'s doc for why upload/read-back both
    /// need this set alongside a node's own declared dtype.
    pub(super) index_nodes: BTreeSet<NodeId>,
}

/// Raw host bytes one [`QuantizedBlock`] hands [`upload_block`]/
/// [`upload_packed_bytes`] — the split-4019 "block upload" term's byte count,
/// distinct from [`QuantizedBlock::element_count`]'s element count (a `Q4_K`
/// super-block's bytes and elements are not the same unit either).
#[cfg(feature = "instrument")]
pub(super) fn block_byte_len(block: &QuantizedBlock<'_>) -> usize {
    match block {
        QuantizedBlock::Float32(data) => size_of_val(*data),
        QuantizedBlock::Int32(data) => size_of_val(*data),
        // every other variant carries packed bytes directly --
        // `QuantizedBlock::packed_bytes` is the one place that per-variant
        // byte-slice extraction lives, instead of restating this match here.
        _ => block.packed_bytes().unwrap_or(&[]).len(),
    }
}

pub(super) fn prepare(
    program: &[Op],
    symbols: &[u64],
    blocks: &[QuantizedBlock<'_>],
    outputs: &[NodeId],
    numeric_policy: NumericPolicy,
    placed_input_nodes: &[NodeId],
) -> Result<Prepared, MetalError> {
    let shapes = infer(program, symbols)?;

    // ROW 327: `block_nodes[i]` is the ONLY node `blocks[i]` may be
    // attributed to -- this crate's positional contract, identical to
    // `proxima_tensor::cpu::evaluate`'s (see `execute`'s own doc). Every
    // classification below (`packed_operands_of`, the dtype gate, per-node
    // shape) reads off this ONE pairing; validating it here, before any of
    // them run, is what stops a caller's node/block order mismatch from
    // surfacing as an unrelated downstream rejection -- a Q6_K block bound
    // to the wrong node used to reach `reject_unsupported_gpu_dtype`
    // (`NotLowerable`, no mention of block order) instead of the precise,
    // node-carrying `InputCountMismatch`/`InputSizeMismatch` below. A
    // same-shape swap between two quantized nodes is still undetectable
    // from bytes alone -- callers with more than one packed input should
    // bind by name via [`plan_named`]/[`resolve_named_blocks`], which
    // resolves this pairing from `Op::name` instead of argument order.
    let block_nodes = block_node_ids(program);
    if blocks.len() != block_nodes.len() {
        return Err(TensorError::InputCountMismatch {
            expected: block_nodes.len(),
            found: blocks.len(),
        }
        .into());
    }
    for (node, block) in block_nodes.iter().zip(blocks.iter()) {
        if placed_input_nodes.iter().any(|placed| placed == node) {
            continue;
        }
        let expected = element_count(shapes.of(*node));
        let found = block.element_count()?;
        if found != expected {
            return Err(TensorError::InputSizeMismatch {
                node: *node,
                expected,
                found,
            }
            .into());
        }
    }

    let packed_operands = packed_operands_of(&block_nodes, blocks);
    // every packed codec's declared dtype is the "these are bytes" marker
    // `reject_unsupported_gpu_dtype`'s own doc already claims as its
    // exemption's rationale -- not just the codecs `packed_operands` above
    // has a kernel for, so any future codec added to `QuantizedBlock` before
    // it has an unpack kernel here still gets the right dtype exemption
    // rather than an unrelated "not float" rejection. Reads
    // `packed_operands`'s own keys rather than re-zipping `block_nodes`
    // against `blocks` a second time, so this set can never disagree with
    // the codec table above on which nodes are packed.
    let packed_operand_nodes: BTreeSet<NodeId> = packed_operands.keys().copied().collect();
    reject_unsupported_gpu_dtype(program, &packed_operand_nodes)?;

    let root = program
        .len()
        .checked_sub(1)
        .map(|last| NodeId(last as u32))
        .ok_or(TensorError::Empty)?;
    for output in outputs {
        if output.0 as usize >= program.len() {
            return Err(TensorError::UnknownOutput(*output).into());
        }
    }
    let effective_outputs = if outputs.is_empty() {
        alloc::vec![root]
    } else {
        outputs.to_vec()
    };

    let mut resolved = bind(program, &shapes, &effective_outputs, numeric_policy)?;
    #[cfg(feature = "instrument")]
    debug!(
        cached_attention_count = resolved
            .iter()
            .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
            .count() as u64,
        resolved_len = resolved.len() as u64,
        "prepare: proxima_tensor::bind() fused-op count reaching this driver"
    );
    #[cfg(feature = "instrument")]
    if effective_outputs.iter().any(|output| output.0 == 2)
        && let Some(bound_embedding) = resolved.iter().find(|bound| bound.node.0 == 2)
        && let Some((source, layout, gather)) = bound_embedding.operands().first()
    {
        debug!(
            node = bound_embedding.node.0,
            source = source.0,
            layout_strides = ?layout.strides,
            gather = ?gather,
            packed_codec = ?packed_operands.get(source),
            "prepared embedding gather layout"
        );
    }
    // A stateless driver has no persistent arena to skip a dead slot inside
    // between calls (unlike `proxima_tensor::cpu::StaticArena`'s own
    // execution-time skip set) -- the only way to avoid dispatching a kernel
    // nobody reads is to drop it from `resolved` before it ever reaches a
    // dispatch list. See `prune_dead`'s own doc.
    resolved = prune_dead(resolved, &effective_outputs);
    // `bind`'s `gated-delta-net-fusion` matcher (default-off in this crate's
    // own `proxima-tensor` dependency, but reachable through Cargo feature
    // unification whenever another workspace member turns it on against the
    // same `proxima-tensor`) is the only producer of
    // `BoundOpKind::GatedDeltaNet` -- `crate::msl::render_gated_delta_net`
    // now emits a real kernel for it, so this Metal path dispatches it like
    // any other op; `wgsl`/`cuda` still reject it (their own `EmitError`
    // gates), the CPU-and-Metal-ahead-of-those-two-backends gap this crate
    // already carries for other fused kinds.
    // `BoundOpBuilder::finish` (`proxima-tensor`'s `bind.rs`) flushes every
    // held elementwise op -- requested output or not -- at the very END of
    // the walk, ascending by `NodeId` among themselves, regardless of where
    // its dependencies or its own program position sit. That is correct for
    // a plain [`execute_plan`] call, where "when a value is computed" never
    // matters to a caller that only reads it back after the whole command
    // buffer completes. It is WRONG for a placed *output*
    // ([`execute_plan_with_placements`]'s own doc) this same call also
    // aliases as a placed *input* elsewhere in the SAME program (the
    // KV-cache shape that doc's "Within-call aliasing" section describes):
    // Metal's hazard tracking only orders two dispatches by ENCODE order, so
    // a deferred write encoded after a consumer that reads the SAME buffer
    // through its aliased `Op::Input` leaves that consumer reading stale
    // bytes. This promotes every `effective_outputs` node forward to right
    // after the latest position its own real operands already occupy -- the
    // position it would have held had `finish` never deferred it -- without
    // touching `BoundOpBuilder`'s own push/finish policy, which stays
    // correct for ordinary (non-aliased) outputs and is unchanged here. A
    // node this call means to output-place but never declared as an
    // `outputs` entry is invisible to this pass AND to `prune_dead` above --
    // see [`execute_plan_with_placements`]'s own doc for why every placed
    // output must be a declared output.
    promote_output_placed_nodes(&mut resolved, &effective_outputs);
    // `bind`'s own `layout_of` assumes every operand is stored row-major in
    // its DECLARED axis order -- true for every f32 buffer this driver reads
    // (bound-time-transposed to match, `bind_matmul_weight`'s own doc), but
    // never true for a packed `Q4_K`/`Q5_K`/`Q6_K` weight, whose bytes are GGUF's
    // native `[out, in]` regardless of what the declared shape says. Left
    // uncorrected, every quantized matmul reads its weight through the wrong
    // stride -- see `correct_packed_matmul_layouts`'s own doc (already
    // codec-agnostic: it takes any `packed_operands` node set).
    correct_packed_matmul_layouts(&mut resolved, &packed_operands.keys().copied().collect());
    let retires = node_retirement(&resolved, &effective_outputs);
    #[cfg(not(feature = "metal-buffer-pool"))]
    let last_reader = node_last_reader(&resolved, program.len());
    let index_nodes = index_node_ids(program);
    let live_block_inputs = live_block_inputs(&block_nodes, &resolved, &effective_outputs);
    Ok(Prepared {
        root,
        shapes,
        effective_outputs,
        block_nodes,
        live_block_inputs,
        packed_operands,
        resolved,
        retires,
        #[cfg(not(feature = "metal-buffer-pool"))]
        last_reader,
        index_nodes,
    })
}

pub(super) fn live_block_inputs(
    block_nodes: &[NodeId],
    resolved: &[BoundOp],
    effective_outputs: &[NodeId],
) -> Vec<bool> {
    let mut live_nodes = BTreeSet::new();
    for bound in resolved {
        for (source, _, lookup) in bound.all_read_sources() {
            live_nodes.insert(*source);
            if let Some(lookup) = lookup {
                live_nodes.insert(lookup.indices);
            }
        }
        if let BoundOpKind::Reduce {
            out_scatter: Some(lookup),
            ..
        } = &bound.kind
        {
            live_nodes.insert(lookup.indices);
        }
    }
    live_nodes.extend(effective_outputs.iter().copied());
    block_nodes
        .iter()
        .map(|node| live_nodes.contains(node))
        .collect()
}

// mirrors `proxima_tensor::cpu::reject_non_float32`'s exemption (a gather's
// `indices` node is the one deliberate exception, since an index value is
// an exact integer carried as f32 regardless of its own declared dtype) but
// this driver's own dtype ceiling is wider than the CPU oracle's: `Float32`
// or `Float16` may reach a device buffer, since `msl.rs` now emits a
// `half`-typed kernel for a `Float16` node instead of assuming `float`
// unconditionally (see `msl.rs`'s own dtype doc). A `BFloat16` node clears
// this gate only via the `packed_nodes` exemption below (it is never a bare
// `Float32`/`Float16` node itself) -- `packed_operands_of` puts every
// `BFloat16` block-input node into that set. Any other dtype is still
// rejected exactly as before.
pub(super) fn reject_unsupported_gpu_dtype(
    program: &[Op],
    packed_nodes: &BTreeSet<NodeId>,
) -> Result<(), TensorError> {
    let index_nodes = index_node_ids(program);
    for (position, expr) in program.iter().enumerate() {
        let node = NodeId(position as u32);
        // a gather's indices are exempt (see this function's doc); so is a
        // packed quantized weight, whose declared dtype is the marker for
        // "these are bytes" and never the element type the kernel computes
        // in — the same exemption `cpu::reject_non_float32` makes via
        // `is_quantized_matmul_operand`.
        if index_nodes.contains(&node) || packed_nodes.contains(&node) {
            continue;
        }
        if !matches!(expr.dtype(), DType::Float32 | DType::Float16) {
            return Err(TensorError::NotLowerable {
                node,
                reason: "metal execution supports float32 or float16 in v1, \
                         except for a gather's indices",
            });
        }
    }
    Ok(())
}

/// The dtype `node`'s own device buffer marshals as: `Float32` when `node`
/// is a gather's `indices` (an index value is an exact integer carried as
/// f32 regardless of its own declared dtype — see
/// [`reject_unsupported_gpu_dtype`]'s doc), otherwise `node`'s own declared
/// dtype straight off the program. `BoundOp::dtype` already carries this
/// same value for a computed node (it is built from the identical `Op`),
/// so callers that already have a `BoundOp` in hand read `bound.dtype`
/// directly instead of calling this — this exists for the two places that
/// only have a bare `NodeId`: uploading a block input and reading back a
/// requested output, either of which may name a plain `Op::Input` node
/// this driver never resolves into a `BoundOp` at all.
pub(super) fn gpu_dtype(program: &[Op], index_nodes: &BTreeSet<NodeId>, node: NodeId) -> DType {
    if index_nodes.contains(&node) {
        DType::Float32
    } else {
        program[node.0 as usize].dtype()
    }
}

pub(super) fn element_count(shape: &[u64]) -> usize {
    shape.iter().product::<u64>() as usize
}

/// `source`'s own TENSOR byte count -- `element_count(shape) *
/// bytes_per_element`, where `bytes_per_element` is exact for a plain buffer
/// ([`DType::size_bytes`]) and a `block_bytes / block_elements` ratio for a
/// packed operand ([`Codec::block_bytes`]/`block_elements`). This is
/// the value a per-operand byte-share table needs -- NOT
/// `device_buffers[source].0.length()`, which reports the shared checkpoint-
/// mapping buffer's own size for every tensor `checkpoint_mapping_offset`
/// binds into it (see [`OpGpuTiming::bound_buffer_bytes`]'s own doc).
/// `lookup: Some(_)` is a GATHERED operand (a grouped or per-route MoE
/// expert weight, addressed through `Lookup::element_stride` at runtime) --
/// `shapes.of(source)` is the operand's full declared graph shape (every
/// expert in the stack), which is NOT what the row-blocked kernel actually
/// reads: it reads exactly one `element_stride`-sized row per index in
/// `lookup.indices` (ROW 543/544, `proxima-tensor/docs/discipline.md`).
/// Reporting the full declared shape for a gathered operand overstated a
/// qwen35moe-shaped grouped gate/up dispatch's own `operand_bytes` by two
/// orders of magnitude (the whole 256-expert stack, `~151 MB`, in place of
/// the `k=8` selected rows this dispatch's `lookup.indices` shape names) --
/// found by comparing this formula's output against the emitted kernel's
/// own per-row block-read count, which reads exactly `element_stride`
/// elements per lookup, not the whole stack.
#[cfg(feature = "instrument")]
pub(super) fn operand_tensor_bytes(
    program: &[Op],
    index_nodes: &BTreeSet<NodeId>,
    shapes: &Shapes,
    packed_operands: &PackedOperands,
    source: NodeId,
    lookup: Option<&Lookup>,
) -> u64 {
    let elements = match lookup {
        Some(lookup) => {
            let rows_read = element_count(shapes.of(lookup.indices)) as u64;
            rows_read * lookup.element_stride.unsigned_abs()
        }
        None => element_count(shapes.of(source)) as u64,
    };
    match packed_operands.get(&source) {
        Some(codec) => {
            elements * crate::msl::codec_block_bytes(codec) as u64
                / crate::msl::codec_block_elements(codec) as u64
        }
        None => elements * gpu_dtype(program, index_nodes, source).size_bytes() as u64,
    }
}

/// Moves every node named in `effective_outputs` to just after the latest
/// position its own operands already occupy in `resolved` -- see this
/// function's own call site in [`prepare`] for why. A no-op for any node
/// already there (every `Reduce` root: `BoundOpBuilder::push` emits those
/// immediately, never defers them).
///
/// A node with NO operand present in `resolved` at all (every real operand
/// is an `Op::Input` leaf) is left exactly where `finish` put it, not
/// pulled to the front: this function only sees the declared tensor-graph
/// operands, never the OTHER kind of ordering constraint this whole module
/// exists for -- one node's placed OUTPUT and a different node's placed
/// INPUT aliasing the SAME `PlacedBuffer`, which is invisible to
/// `proxima-tensor`'s graph entirely (the caller supplies that aliasing at
/// execute time, not plan time). A zero-real-operand output can be the
/// aliased READER half of exactly that pair
/// (`omega/tests/metal_output_placement.rs`'s
/// `a_program_reads_a_placed_write_from_a_later_op_in_the_same_call` is
/// this shape), and moving it to position `0` would place it BEFORE the
/// write it depends on for correctness, the same bug class this function
/// exists to fix, just on the other node. Leaving it at `finish`'s own
/// position is always safe for this shape: every node `finish` flushes is
/// already in ascending `NodeId` order relative to every OTHER flushed
/// node (its own doc), so two zero-real-operand outputs still come out in
/// their aliasing-write-then-read order as long as neither is promoted.
///
/// `resolved` is already a valid topological order before this call
/// (`BoundOpBuilder`'s own invariant: a reference only ever points
/// backwards), so every candidate with a real dependency has a target
/// position provably `<=` its current one -- this only ever pulls a node
/// EARLIER, never later, so it cannot itself introduce a forward reference.
pub(super) fn promote_output_placed_nodes(resolved: &mut Vec<BoundOp>, effective_outputs: &[NodeId]) {
    let outputs: BTreeSet<NodeId> = effective_outputs.iter().copied().collect();
    if outputs.is_empty() {
        return;
    }
    let candidates: Vec<NodeId> = resolved
        .iter()
        .filter(|bound| outputs.contains(&bound.node))
        .map(|bound| bound.node)
        .collect();

    for node in candidates {
        // Recomputed fresh every iteration, not cached across the loop: an
        // earlier candidate's own move can shift this node's (and its
        // operands') position by one, and `resolved`'s exact current order
        // is the only thing this function is allowed to trust.
        let index_of: BTreeMap<NodeId, usize> = resolved
            .iter()
            .enumerate()
            .map(|(index, bound)| (bound.node, index))
            .collect();
        let Some(current_index) = index_of.get(&node).copied() else {
            continue;
        };
        // `all_read_sources`, not `operands` -- a fused `Reduce`'s
        // `epilogue_operands` are real, materialized reads this op's own
        // dispatch needs bound (`crate::msl::bindings`'s own doc says the
        // same), so a candidate this loop promotes earlier must never land
        // ahead of an epilogue operand's own producer position, or that
        // producer's buffer has not been dispatched yet when this op reads
        // it. `operands()` alone missed exactly that class of dependency:
        // the MoE top-k combine's per-round softmax weight lives ONLY in
        // the final reduce's epilogue, so this promotion moved the combine
        // ahead of its own weight's dispatch and `buffer_for` failed at
        // execution time with "operand buffer missing".
        let dependency_index = resolved[current_index]
            .all_read_sources()
            .filter_map(|(source, _, gather)| {
                let mut latest = index_of.get(source).copied();
                if let Some(lookup) = gather {
                    latest = latest.max(index_of.get(&lookup.indices).copied());
                }
                latest
            })
            .max();
        let Some(dependency) = dependency_index else {
            continue;
        };
        let target_index = dependency + 1;
        if target_index < current_index {
            let bound = resolved.remove(current_index);
            resolved.insert(target_index, bound);
        }
    }
}

/// The output length an op needs allocated: the reduced (product of
/// surviving axes) length for a `Keep::Reduce` reduce, or the full
/// iteration space otherwise (elementwise and `Keep::Scan` both write one
/// value per coordinate). Deliberately independent of [`Kernel::grid`]'s
/// thread count — a `Keep::Scan` scan dispatches one thread per *line* but
/// writes `inner_len` values per thread, so grid threads and output length
/// diverge there.
///
/// A non-empty `epilogue_broadcast_axes` (`bind::BoundOpKind::Reduce::
/// epilogue_broadcast_axes`'s own doc, `proxima_tensor::cpu`'s own
/// `node_output_len` mirrors this same widening) re-broadcasts the fold's
/// scalar back over the axes it reduced away, so the MATERIALIZED output is
/// the full `extents` product, not just `output_axes`'s smaller fold shape —
/// falls through to the same full-iteration-space arm every other kind
/// takes.
pub(super) fn bound_output_len(bound: &BoundOp) -> usize {
    match &bound.kind {
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            output_axes,
            epilogue_broadcast_axes,
            ..
        } if epilogue_broadcast_axes.is_empty() => output_axes
            .iter()
            .map(|axis| bound.extents[*axis as usize] as usize)
            .product(),
        _ => bound
            .extents
            .iter()
            .map(|extent| *extent as usize)
            .product(),
    }
}

pub(super) fn push_i64(bytes: &mut Vec<u8>, value: i64) {
    bytes.extend_from_slice(&value.to_ne_bytes());
}

/// Pushes `values` as a fixed-`width` MSL array, zero-padding any slot
/// `values` does not fill — the only case that happens is a rank-0 op,
/// where the declared array width is `max(rank, 1)` but there is no real
/// axis to supply, and that padding slot is never read by the generated
/// source (see each `render_*`'s `if rank > 0` / `.saturating_sub(1)` guards).
pub(super) fn push_i64_row(bytes: &mut Vec<u8>, values: &[i64], width: usize) {
    for slot in 0..width {
        push_i64(bytes, values.get(slot).copied().unwrap_or(0));
    }
}

/// [`push_i64_row`]'s `u64`-extents counterpart -- casts in place instead of
/// collecting `bound.extents` into a temporary `Vec<i64>` first, the
/// allocation ROW 303's residual named in [`pack_elementwise_uniforms`].
pub(super) fn push_extent_row(bytes: &mut Vec<u8>, extents: &[u64], width: usize) {
    for slot in 0..width {
        let value = extents.get(slot).map(|extent| *extent as i64).unwrap_or(0);
        push_i64(bytes, value);
    }
}

/// [`push_extent_row`]'s gathered counterpart: `axes` indexes `extents` in
/// AXIS order (a reduce's `output_axes`/reduction axes are a subset of
/// `bound.extents`, not a contiguous prefix), so this reads `extents[axis]`
/// directly per slot instead of [`pack_reduce_uniforms`] collecting the
/// gather into a temporary `Vec<i64>` first (ROW 303's residual, this
/// landing's own removal). `width` zero-pads past `axes.len()`, same
/// contract as [`push_i64_row`].
pub(super) fn push_gathered_extent_row(bytes: &mut Vec<u8>, extents: &[u64], axes: &[u16], width: usize) {
    for slot in 0..width {
        let value = axes
            .get(slot)
            .map(|axis| extents[*axis as usize] as i64)
            .unwrap_or(0);
        push_i64(bytes, value);
    }
}

/// The product [`push_gathered_extent_row`] would write, computed without
/// materializing the row -- both `output_extents.iter().product()` and
/// `reduction_extents.iter().product()` in [`pack_reduce_uniforms`] need
/// exactly this, ahead of the row itself.
pub(super) fn gathered_extent_product(extents: &[u64], axes: &[u16]) -> i64 {
    axes.iter()
        .map(|axis| extents[*axis as usize] as i64)
        .product()
}

/// Appends the four gather arrays every `Uniforms` struct declares last (via
/// `crate::msl::push_gather_uniform_fields`) when `bound` has at least one
/// gathered operand: `gather_index_base`, `gather_index_strides`,
/// `gather_element_stride`, `gather_extent` — each one array of length
/// `gather_count`. `crate::msl::gather_slots` numbers gathered operands by
/// encounter order over `bound.operands()`, so filtering in that same order
/// (below) reproduces the identical numbering without needing to re-derive
/// or look up the slot indices themselves. A no-op when `bound` has no
/// gather, matching `push_gather_uniform_fields`'s own empty-array early
/// return.
pub(super) fn push_gather_uniforms(bytes: &mut Vec<u8>, bound: &BoundOp, rank_len: usize) {
    let ordered: Vec<&Lookup> = bound
        .operands()
        .iter()
        .filter_map(|(_, _, gather)| gather.as_ref())
        .collect();
    if ordered.is_empty() {
        return;
    }

    for gather in &ordered {
        push_i64(bytes, gather.index_layout.base);
    }
    for gather in &ordered {
        push_i64_row(bytes, &gather.index_layout.strides, rank_len);
    }
    for gather in &ordered {
        push_i64(bytes, gather.element_stride);
    }
    for gather in &ordered {
        push_i64(bytes, gather.extent as i64);
    }
}

/// Bytes [`push_gather_uniforms`] appends for `gather_count` gathered
/// operands at `rank_len` — `0` operands write nothing (the early return in
/// [`push_gather_uniforms`] itself), otherwise 3 scalar `i64` fields
/// (`gather_index_base`/`gather_element_stride`/`gather_extent`) plus one
/// `rank_len`-wide `i64` row (`gather_index_strides`) PER gathered operand.
pub(super) fn gather_uniform_byte_len(gather_count: usize, rank_len: usize) -> usize {
    if gather_count == 0 {
        0
    } else {
        gather_count * (rank_len + 3) * size_of::<i64>()
    }
}

/// [`pack_uniforms`]'s own byte length, computed from `bound`'s static shape
/// (extents, operand count, gather count) rather than by actually building
/// the byte vector — every field [`pack_uniforms`]'s packers write is a
/// scalar `i64` or an `i64` row of fixed width ([`push_i64`]/[`push_i64_row`]
/// never branch on the VALUES, only on `rank_len`/`width`), so the total byte
/// count is a pure function of shape. [`build_buffer_arena`]'s own
/// `uniform_bytes` diagnostic sum used to call [`pack_uniforms`] once per op
/// just to read `.len()` off the result, allocating and filling a real byte
/// vector — for every op in the plan — purely to throw it away; this
/// mirrors [`pack_uniforms`]'s own match arms field-for-field instead.
pub(super) fn pack_uniforms_byte_len(bound: &BoundOp) -> usize {
    const WORD: usize = size_of::<i64>();
    let rank_len = bound.extents.len().max(1);
    let operand_count = bound.operands().len();
    let gather = gather_count(bound);

    match &bound.kind {
        BoundOpKind::CachedAttention {
            cached_key_rows, ..
        } => {
            // Mirrors `pack_cached_attention_uniforms`: the single-range
            // fused form (nine operands, `cached_key_rows == 0`) appends
            // `cached_key_rows`, `new_key_rows`, `context_chunks`, and
            // `splits` as runtime
            // uniform fields (`total_elements` plus these four = 5 words).
            // `two_range_cached_bound` (nine operands, `cached_key_rows !=
            // 0`) reads its runtime bound straight off `in8` in the kernel
            // body instead (`render_cached_attention`'s own doc) and keeps
            // the minimal one-word struct, so only the single-range form
            // widens the uniform blob.
            if operand_count == 9 && *cached_key_rows == 0 {
                5 * WORD
            } else {
                WORD
            }
        }
        BoundOpKind::Iota | BoundOpKind::Constant { .. } => WORD,
        BoundOpKind::Elementwise { .. } => {
            (1 + rank_len + operand_count + operand_count * rank_len) * WORD
                + gather_uniform_byte_len(gather, rank_len)
        }
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            output_axes,
            epilogue_operands,
            ..
        } => {
            let output_rank_len = output_axes.len().max(1);
            let reduce_rank_len = reduction_dims(bound, output_axes).len().max(1);
            let epilogue_len = if epilogue_operands.is_empty() {
                0
            } else {
                epilogue_operands.len() * (1 + output_rank_len)
            };
            (2 + output_rank_len
                + reduce_rank_len
                + operand_count
                + operand_count * rank_len
                + 1
                + rank_len
                + epilogue_len)
                * WORD
                + gather_uniform_byte_len(gather, rank_len)
        }
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => {
            let outer_rank_len = bound.extents.len().saturating_sub(1).max(1);
            (2 + outer_rank_len + operand_count + operand_count * rank_len + 1 + rank_len) * WORD
                + gather_uniform_byte_len(gather, rank_len)
        }
        // Mirrors `render_gated_delta_net`'s own `struct Uniforms`: four
        // `long` fields (`n_tokens`, `query_key_head_stride`,
        // `query_key_dim_stride`, `inv_sqrt_key_dim_bits`) -- `kv_heads`/
        // `num_v_heads`/`head_k_dim`/`head_v_dim` are baked `constexpr`
        // instead (that function's own doc), so they never widen this blob.
        BoundOpKind::GatedDeltaNet { .. } => 4 * WORD,
        // ROW 569: `render_moe_topk`'s own `Uniforms { long unused; }` --
        // `expert_count`/`top_k` are baked `constexpr` instead (that
        // function's own doc), so this is the same "leaf" one-`long` shape
        // `pack_leaf_uniforms` already packs for `Iota`/`Constant`.
        BoundOpKind::MoeTopK { .. } => WORD,
    }
}

pub(super) fn pack_uniforms(bound: &BoundOp, numeric_policy: NumericPolicy) -> Result<Vec<u8>, EmitError> {
    let mut bytes = Vec::new();
    pack_uniforms_into(bound, numeric_policy, &mut bytes)?;
    Ok(bytes)
}

/// [`pack_uniforms`], writing into caller-owned storage. Stable plans call
/// this once while initializing each per-position uniform buffer; the
/// unplaced path uses the owned wrapper above.
pub(super) fn pack_uniforms_into(
    bound: &BoundOp,
    numeric_policy: NumericPolicy,
    scratch: &mut Vec<u8>,
) -> Result<(), EmitError> {
    scratch.clear();
    match &bound.kind {
        BoundOpKind::CachedAttention { .. } => {
            pack_cached_attention_uniforms(bound, numeric_policy, scratch)
        }
        BoundOpKind::Elementwise { .. } => {
            pack_elementwise_uniforms(bound, scratch);
            Ok(())
        }
        BoundOpKind::Reduce {
            keep: Keep::Reduce, ..
        } => pack_reduce_uniforms(bound, scratch),
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => pack_scan_uniforms(bound, scratch),
        BoundOpKind::Iota | BoundOpKind::Constant { .. } => {
            pack_leaf_uniforms(bound, scratch);
            Ok(())
        }
        BoundOpKind::GatedDeltaNet {
            n_tokens,
            query_key_head_stride,
            query_key_dim_stride,
            inv_sqrt_key_dim,
            ..
        } => {
            pack_gated_delta_net_uniforms(
                *n_tokens,
                *query_key_head_stride,
                *query_key_dim_stride,
                *inv_sqrt_key_dim,
                scratch,
            );
            Ok(())
        }
        // `render_moe_topk`'s own `Uniforms { long unused; }`: `expert_count`/
        // `top_k` are baked `constexpr` (that function's own doc), so this
        // slot is never read by the kernel body -- packed only because
        // `bindings` gives every kind one, and `pack_leaf_uniforms`'s single
        // `long` is exactly `sizeof(Uniforms)`.
        BoundOpKind::MoeTopK { .. } => {
            pack_leaf_uniforms(bound, scratch);
            Ok(())
        }
    }
}

/// Mirrors `render_gated_delta_net`'s own `struct Uniforms` byte-for-byte,
/// in field order: `n_tokens`, `query_key_head_stride`,
/// `query_key_dim_stride`, then `inv_sqrt_key_dim` reinterpreted as raw
/// bits in a `long` slot -- every other uniform field here is already a
/// `long`, and MSL's `Uniforms` struct has no mixed-width fields to keep
/// this byte-compatible with.
pub(super) fn pack_gated_delta_net_uniforms(
    n_tokens: u64,
    query_key_head_stride: u64,
    query_key_dim_stride: u64,
    inv_sqrt_key_dim: f32,
    bytes: &mut Vec<u8>,
) {
    push_i64(bytes, n_tokens as i64);
    push_i64(bytes, query_key_head_stride as i64);
    push_i64(bytes, query_key_dim_stride as i64);
    push_i64(bytes, i64::from(inv_sqrt_key_dim.to_bits()));
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(super) mod numeric_policy_construction_tests {
    //! [`plan`]/[`plan_named`] now take `numeric_policy` as a required
    //! constructor argument and record it once, at bind time -- the fix for
    //! the two bugs this design closes (a conflated permission ladder, and
    //! a policy that never reached `bind`). These tests prove the round
    //! trip through [`numeric_policy_as_metal_math_mode`]/
    //! [`metal_math_mode_as_numeric_policy`], and that a [`Plan`] reports
    //! exactly the policy it was constructed under -- never a later
    //! default, and never silently narrowed by [`Plan::set_math_mode`].

    use alloc::vec;

    use proxima_tensor::{DType, Extent, IndexMap, NumericPolicy, Op, QuantizedBlock, append, map};

    use super::{MathMode, MetalError, metal_math_mode_as_numeric_policy, plan};
    #[cfg(feature = "instrument")]
    use super::{PIPELINE_MISSES, device_and_queue, resolve_steps};

    #[test]
    fn metal_math_mode_round_trips_through_numeric_policy() {
        for mode in [MathMode::Safe, MathMode::Relaxed, MathMode::Fast] {
            let policy = metal_math_mode_as_numeric_policy(mode);
            assert_eq!(
                super::numeric_policy_as_metal_math_mode(policy),
                mode,
                "mode {mode:?} must round-trip through its own exact permission set"
            );
        }
    }

    /// The fix for the two bugs ROW 4309 (this file) found: a
    /// contraction-only policy used to be rounded UP to `Relaxed` (which
    /// also grants `reassociation`), and a `signed_zero`-only policy up to
    /// `Fast` (which grants everything). Every one of the 32 permission
    /// subsets must instead map to the WIDEST mode whose own permission set
    /// ([`metal_math_mode_as_numeric_policy`]) is a SUBSET of what the
    /// policy actually grants ([`NumericPolicy::grants`]) -- never wider.
    #[test]
    fn numeric_policy_projection_never_grants_more_than_the_policy_allows() {
        for bits in 0u8..32 {
            let policy = NumericPolicy::bit_exact()
                .with_contraction(bits & 0b0000_0001 != 0)
                .with_reassociation(bits & 0b0000_0010 != 0)
                .with_nan_assumptions(bits & 0b0000_0100 != 0)
                .with_signed_zero(bits & 0b0000_1000 != 0)
                .with_approx_functions(bits & 0b0001_0000 != 0);
            let mode = super::numeric_policy_as_metal_math_mode(policy);
            let granted = metal_math_mode_as_numeric_policy(mode);
            assert!(
                policy.grants(granted),
                "subset {bits:05b} projected to {mode:?}, whose own permission set \
                 {granted:?} exceeds what the policy {policy:?} actually grants"
            );
        }
    }

    fn identity_program() -> (Vec<Op>, proxima_tensor::NodeId) {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(4)],
                name: None,
            },
        );
        let identity = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: proxima_tensor::ScalarOp::Identity,
                operands: vec![(source, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
        (program, identity)
    }

    #[test]
    fn plan_records_the_policy_it_bound_under_not_a_later_default() {
        let (program, identity) = identity_program();
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let blocks = [QuantizedBlock::Float32(&data)];
        let resolved_plan = plan(
            &program,
            &[],
            &blocks,
            &[identity],
            NumericPolicy::llama_relaxed(),
        )
        .expect("plans the identity program");
        assert_eq!(
            resolved_plan.numeric_policy(),
            NumericPolicy::llama_relaxed()
        );
        assert!(
            resolved_plan
                .check_numeric_policy(NumericPolicy::llama_relaxed())
                .is_ok()
        );
    }

    #[test]
    fn plan_refuses_a_mismatched_policy_instead_of_silently_narrowing() {
        let (program, identity) = identity_program();
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let blocks = [QuantizedBlock::Float32(&data)];
        let resolved_plan = plan(
            &program,
            &[],
            &blocks,
            &[identity],
            NumericPolicy::bit_exact(),
        )
        .expect("plans the identity program");
        let error = resolved_plan
            .check_numeric_policy(NumericPolicy::fast())
            .expect_err("bit-exact-bound plan does not satisfy fast");
        let MetalError::NumericPolicyMismatch { bound, requested } = error else {
            panic!("expected NumericPolicyMismatch, got {error:?}");
        };
        assert_eq!(bound, NumericPolicy::bit_exact());
        assert_eq!(requested, NumericPolicy::fast());
    }

    #[test]
    fn set_math_mode_narrows_within_the_bound_policy_but_never_widens_it() {
        let (program, identity) = identity_program();
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let blocks = [QuantizedBlock::Float32(&data)];
        let mut resolved_plan = plan(
            &program,
            &[],
            &blocks,
            &[identity],
            NumericPolicy::llama_relaxed(),
        )
        .expect("plans the identity program");
        resolved_plan
            .set_math_mode(MathMode::Safe)
            .expect("Safe needs nothing, llama_relaxed() grants it trivially");
        assert_eq!(resolved_plan.math_mode(), MathMode::Safe);
        assert_eq!(
            resolved_plan.numeric_policy(),
            NumericPolicy::llama_relaxed(),
            "narrowing math_mode must never change the bound numeric_policy"
        );
        let error = resolved_plan
            .set_math_mode(MathMode::Fast)
            .expect_err("Fast needs nan_assumptions/signed_zero/approx_functions, llama_relaxed() withholds all three");
        let MetalError::NumericPolicyMismatch { bound, requested } = error else {
            panic!("expected NumericPolicyMismatch, got {error:?}");
        };
        assert_eq!(bound, NumericPolicy::llama_relaxed());
        assert_eq!(requested, NumericPolicy::fast());
    }

    #[cfg(feature = "instrument")]
    #[test]
    fn narrowing_math_mode_forces_a_genuine_pipeline_cache_miss_not_a_stale_hit() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        let (program, identity) = identity_program();
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let blocks = [QuantizedBlock::Float32(&data)];
        let mut resolved_plan = plan(
            &program,
            &[],
            &blocks,
            &[identity],
            NumericPolicy::llama_relaxed(),
        )
        .expect("plans the identity program");
        assert_eq!(
            resolved_plan.math_mode(),
            MathMode::Relaxed,
            "llama_relaxed() constructs Relaxed, not the Fast that the old \
             widest-mode-that-covers-any-permission bug would have produced"
        );

        let _ = PIPELINE_MISSES.snapshot_and_reset();
        resolve_steps(&device, &resolved_plan).expect("first resolution compiles under Relaxed");
        let first_misses = PIPELINE_MISSES.snapshot_and_reset();
        assert_eq!(
            first_misses, 1,
            "the first resolution of a fresh plan is always a miss"
        );
        let relaxed_key = resolved_plan
            .kernel_keys()
            .expect("kernel_keys reads the just-resolved identity")[0]
            .clone();

        resolved_plan
            .set_math_mode(MathMode::Safe)
            .expect("Safe needs nothing, llama_relaxed() grants it trivially");
        resolve_steps(&device, &resolved_plan).expect("second resolution compiles under Safe");
        let second_misses = PIPELINE_MISSES.snapshot_and_reset();
        assert_eq!(
            second_misses, 1,
            "set_math_mode(Safe) must force a genuine PIPELINE_CACHE miss, never reuse the \
             Relaxed-compiled pipeline cached under the SAME numeric_policy token"
        );
        let safe_key = resolved_plan
            .kernel_keys()
            .expect("kernel_keys after narrowing")[0]
            .clone();
        assert_ne!(
            relaxed_key, safe_key,
            "Relaxed's and Safe's cache keys must differ by the math-mode token even though \
             numeric_policy (and so emit's rendered source) never changed"
        );

        let resolved_math_mode = resolved_plan
            .resolved_steps
            .borrow()
            .as_ref()
            .expect("resolve_steps populated resolved_steps")
            .math_mode;
        assert_eq!(
            resolved_math_mode,
            MathMode::Safe,
            "resolved_steps must record the mode it just compiled under, not the plan's earlier one"
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(super) mod pack_uniforms_byte_len_tests {
    //! [`super::pack_uniforms_byte_len`] mirrors [`super::pack_uniforms`]'s
    //! match arms field-for-field rather than calling it -- these tests are
    //! the parity proof: for a real elementwise op and a real `Keep::Reduce`
    //! matmul-shaped op (the two arms `build_buffer_arena`'s hot loop
    //! actually walks on every real forward), the analytically-computed
    //! length must equal the real byte vector's `.len()` exactly.

    use alloc::vec;
    use alloc::vec::Vec;

    use proxima_tensor::{
        BoundOp, BoundOpKind, DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, Reduce,
        ReduceInit, ScalarOp, append, bind, infer, map,
    };

    use super::{pack_uniforms, pack_uniforms_byte_len};

    /// The last node `program` builds -- see `msl::tests::terminal`'s own
    /// doc (ROW 541, `proxima-tensor/docs/discipline.md`): every fixture
    /// here treats it as "the answer", and `bind_plain`'s reachability pass
    /// now requires it be named explicitly rather than relying on `&[]`.
    fn terminal(program: &[Op]) -> NodeId {
        NodeId((program.len() - 1) as u32)
    }

    fn elementwise_add_op(extent_a: u32, extent_b: u32) -> BoundOp {
        let mut program = Vec::new();
        let lhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(extent_a), Extent::Static(extent_b)],
                name: None,
            },
        );
        let rhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(extent_a), Extent::Static(extent_b)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: vec![
                    (lhs, IndexMap::Affine(map::projection(2, &[0, 1]))),
                    (rhs, IndexMap::Affine(map::projection(2, &[0, 1]))),
                ],
                name: None,
            },
        );
        let shapes = infer(&program, &[]).expect("elementwise infers");
        bind(&program, &shapes, &[terminal(&program)], NumericPolicy::default())
            .expect("elementwise lowers")
            .into_iter()
            .next_back()
            .expect("one bound op emitted")
    }

    fn matmul_reduce_op(m: u32, k: u32, n: u32) -> BoundOp {
        let mut program = Vec::new();
        let lhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(m), Extent::Static(k)],
                name: None,
            },
        );
        let rhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(k), Extent::Static(n)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                    (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
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
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: Some("matmul".into()),
            }),
        );
        let shapes = infer(&program, &[]).expect("matmul infers");
        bind(&program, &shapes, &[terminal(&program)], NumericPolicy::default())
            .expect("matmul lowers")
            .into_iter()
            .next_back()
            .expect("one fused bound op emitted")
    }

    #[test]
    fn elementwise_byte_len_matches_the_real_packed_bytes() {
        let bound = elementwise_add_op(4, 3);
        assert!(
            matches!(bound.kind, BoundOpKind::Elementwise { .. }),
            "fixture must actually lower to an Elementwise BoundOp"
        );
        assert_eq!(
            pack_uniforms_byte_len(&bound),
            pack_uniforms(&bound, NumericPolicy::default())
                .expect("packs uniforms")
                .len()
        );
    }

    #[test]
    fn reduce_byte_len_matches_the_real_packed_bytes() {
        let bound = matmul_reduce_op(2, 5, 3);
        assert!(
            matches!(
                bound.kind,
                BoundOpKind::Reduce {
                    keep: Keep::Reduce,
                    ..
                }
            ),
            "fixture must actually lower to a Keep::Reduce BoundOp"
        );
        assert_eq!(
            pack_uniforms_byte_len(&bound),
            pack_uniforms(&bound, NumericPolicy::default())
                .expect("packs uniforms")
                .len()
        );
    }
}

pub(super) fn pack_cached_attention_uniforms(
    bound: &BoundOp,
    numeric_policy: NumericPolicy,
    bytes: &mut Vec<u8>,
) -> Result<(), EmitError> {
    let BoundOpKind::CachedAttention {
        head_dim,
        query_groups,
        cached_key_rows,
        new_key_rows,
        ..
    } = &bound.kind
    else {
        return Err(EmitError::RenderKindMismatch {
            node: bound.node,
            expected: "cached_attention",
            found: bound.kind.name(),
        });
    };
    // Same `cached_key_rows == 0` discriminator as `crate::msl::render_
    // cached_attention` / `grid_threads` -- `two_range_cached_bound` (nine
    // operands, `cached_key_rows != 0`) reads its live bound off `in8`
    // inside the kernel body instead of widening the dispatch or the
    // uniforms struct, so it takes the same path as the eight-operand,
    // unbucketed case below.
    let dynamic_cached_len =
        (bound.operands().len() == 9 || bound.operands().len() == 12) && *cached_key_rows == 0;
    let context_length = *cached_key_rows + *new_key_rows;
    let chunks =
        crate::msl::context_chunks_for(context_length, *query_groups, *head_dim, numeric_policy)
            as i64;
    let splits = crate::msl::splits_for(context_length, numeric_policy) as i64;
    // Redesign §5 option 2: the single-range fused path always dispatches
    // the shape-bounded compiled MAXIMUM chunk count
    // (`effective_context_chunk_cap`) -- `grid_threads`'s own
    // `CachedAttention` arm (`omega/src/msl.rs`) -- so `total_elements` here
    // must agree with that grid width, never the live `chunks` value, which
    // travels separately as its own `Uniforms` field below. Redesign §4c:
    // the dispatch is ALSO widened by the compiled `ATTENTION_SPLIT_MAX`,
    // but only when `cached_attention_merge_needed` holds for this op's own
    // context length -- `grid_threads`'s own doc explains why a split, unlike
    // a chunk, cannot be widened unconditionally without racing the real
    // dispatch's output write; this must stay in lock-step with that gate.
    let dispatch_chunks = if dynamic_cached_len {
        i64::try_from(crate::msl::effective_context_chunk_cap(*query_groups, *head_dim))
            .unwrap_or(chunks)
    } else {
        chunks
    };
    let dispatch_splits = if dynamic_cached_len
        && crate::msl::cached_attention_merge_needed(&bound.kind, numeric_policy)
    {
        i64::try_from(crate::sized::ATTENTION_SPLIT_MAX).unwrap_or(splits)
    } else {
        1
    };
    let total: i64 = bound
        .extents
        .iter()
        .map(|extent| *extent as i64)
        .product::<i64>()
        / *head_dim as i64
        * dispatch_chunks
        * dispatch_splits;
    push_i64(bytes, total);
    // Mirrors `render_cached_attention`'s `struct Uniforms` (`omega/src/msl.rs`):
    // the single-range fused form (nine operands) declares three extra
    // `long` fields so the kernel body reads the live row counts AND the
    // live chunk count off the uniform buffer instead of baking either as
    // `constexpr` at render time -- the row counts per ROW 369, the chunk
    // count per redesign §5 option 2, both keyed off the SAME `bound.kind`
    // this function already reads per dispatch, so a `kv-capacity-bucket`
    // crossing (a new `BoundOp` with new `cached_key_rows`/`new_key_rows`)
    // repacks fresh uniform values without recompiling the kernel.
    if dynamic_cached_len {
        push_i64(bytes, *cached_key_rows as i64);
        push_i64(bytes, *new_key_rows as i64);
        push_i64(bytes, chunks);
        // Redesign §4c: the cross-THREADGROUP sibling of `chunks` above, one
        // hardware level up -- see `crate::msl::splits_for`'s own doc. The
        // LIVE value (never the compiled maximum `dispatch_splits` widens the
        // grid by above), since this is what the kernel body's own
        // `slice_len`/`lo`/`hi` computation reads. `tgid`'s own decode into
        // `query_row`/`split` divides out `dispatch_splits`, NOT this field --
        // that grid-widening constant, not the live count, is what an idle
        // threadgroup beyond it was actually enumerated against. ROW 385:
        // when no merge dispatch exists, this MUST be `1`, never `splits_for`'s
        // own raw result -- the kernel body's `slice_len = ceil(live/splits)`
        // would otherwise slice off keys with no second dispatch left to
        // merge the slices back, an out-of-bounds-shaped undercount.
        let live_splits = if crate::msl::cached_attention_merge_needed(&bound.kind, numeric_policy)
        {
            splits
        } else {
            1
        };
        push_i64(bytes, live_splits);
    }
    Ok(())
}

/// Mirrors `crate::msl::render_cached_attention_merge`'s own `Uniforms`
/// struct: `total_elements` (the SAME `query_rows * heads` count
/// [`pack_cached_attention_uniforms`]'s own `total` computes before its
/// `dispatch_chunks` multiply -- the merge kernel dispatches one simdgroup
/// per output row, not per `(row, chunk)` pair) plus `splits`, the live
/// count the merge's own online-softmax combine reduces over -- the SAME
/// `crate::msl::splits_for` call [`pack_cached_attention_uniforms`] already
/// makes for the split kernel's own `Uniforms.splits`, so the two dispatches
/// can never disagree on how many partials the split kernel wrote and the
/// merge kernel reads back. Uploaded fresh every call (`upload_uniforms`,
/// not a `Plan`-owned buffer like the split kernel's own `plan_uniform`) --
/// this slice's own scope boundary, named in `encode_op`'s call site;
/// folding it into `PlanUniforms` is follow-up work, not a correctness gap.
pub(super) fn pack_cached_attention_merge_uniforms(
    bound: &BoundOp,
    numeric_policy: NumericPolicy,
) -> Result<Vec<u8>, EmitError> {
    let BoundOpKind::CachedAttention {
        head_dim,
        cached_key_rows,
        new_key_rows,
        ..
    } = &bound.kind
    else {
        return Err(EmitError::RenderKindMismatch {
            node: bound.node,
            expected: "cached_attention_merge",
            found: bound.kind.name(),
        });
    };
    let total: i64 = bound
        .extents
        .iter()
        .map(|extent| *extent as i64)
        .product::<i64>()
        / *head_dim as i64;
    let splits = crate::msl::splits_for(*cached_key_rows + *new_key_rows, numeric_policy) as i64;
    let mut bytes = Vec::with_capacity(16);
    push_i64(&mut bytes, total);
    push_i64(&mut bytes, splits);
    Ok(bytes)
}

/// Mirrors the `Uniforms` struct `crate::msl::render_iota` and
/// `crate::msl::render_constant` both declare: just `total_elements` —
/// neither leaf has operands, a per-axis extents array, or a gather, so
/// there is nothing else this struct needs to carry. `render_constant`
/// bakes its literal into the source instead of adding a field here, which
/// is what lets one packer serve both.
pub(super) fn pack_leaf_uniforms(bound: &BoundOp, bytes: &mut Vec<u8>) {
    let total: i64 = bound.extents.iter().map(|extent| *extent as i64).product();
    push_i64(bytes, total);
}

/// Mirrors the `Uniforms` struct `crate::msl::render_elementwise` declares
/// at `omega/src/msl.rs:328-335`: `total_elements`, `extents[rank_len]`,
/// `operand_base[operand_count]`, `operand_strides[operand_count][rank_len]`,
/// then — only when `bound` has a gathered operand — the four
/// `push_gather_uniform_fields` arrays [`push_gather_uniforms`] appends, in
/// that order — every field `long`, so a flat `i64` concatenation is the
/// struct's byte layout.
pub(super) fn pack_elementwise_uniforms(bound: &BoundOp, bytes: &mut Vec<u8>) {
    let rank_len = bound.extents.len().max(1);

    push_i64(
        bytes,
        bound.extents.iter().map(|extent| *extent as i64).product(),
    );
    push_extent_row(bytes, &bound.extents, rank_len);
    for (_, layout, _) in bound.operands() {
        push_i64(bytes, layout.base);
    }
    for (_, layout, _) in bound.operands() {
        push_i64_row(bytes, &layout.strides, rank_len);
    }
    push_gather_uniforms(bytes, bound, rank_len);
}

/// Mirrors the `Uniforms` struct `crate::msl::render_reduce` declares at
/// `omega/src/msl.rs:386-397`: `output_total`, `reduction_total`,
/// `output_extents[output_rank_len]`, `reduction_extents[reduce_rank_len]`,
/// `operand_base[operand_count]`,
/// `operand_strides[operand_count][rank_len]`, `out_base`,
/// `out_strides[rank_len]`, then the gather arrays (see
/// [`pack_elementwise_uniforms`]'s doc), in that order.
pub(super) fn pack_reduce_uniforms(bound: &BoundOp, bytes: &mut Vec<u8>) -> Result<(), EmitError> {
    let BoundOpKind::Reduce {
        output_axes,
        out_layout,
        epilogue_operands,
        epilogue_broadcast_axes,
        ..
    } = &bound.kind
    else {
        return Err(EmitError::RenderKindMismatch {
            node: bound.node,
            expected: "keep::reduce fold",
            found: bound.kind.name(),
        });
    };
    let rank_len = bound.extents.len().max(1);
    let output_rank_len = output_axes.len().max(1);
    let reduce_axes = reduction_dims(bound, output_axes);
    let reduce_rank_len = reduce_axes.len().max(1);
    // `crate::msl::render_reduce`'s own doc: a non-empty `epilogue_broadcast_
    // axes` widens `epilogue_operand_strides` from `output_rank_len` to the
    // full `rank_len` (`x`/`gamma` read at the whole `(s, d)` coordinate, not
    // just `s`), and needs a SEPARATE `broadcast_out_strides` row -- the
    // CONTIGUOUS row-major layout the wider materialized output buffer is
    // allocated with, never `out_layout.strides` (kept as-is above: that
    // stays the fold's own COMPACT addressing, stride `0` on every broadcast
    // axis, still needed to locate the fold's scalar).
    let is_broadcast_epilogue = !epilogue_broadcast_axes.is_empty();
    let epilogue_stride_rank_len = if is_broadcast_epilogue {
        rank_len
    } else {
        output_rank_len
    };

    // ROW 303 residual, removed by this landing: `output_extents`/
    // `reduction_extents` used to be temporary `Vec<i64>`s built by
    // gathering `bound.extents` through `output_axes`/`reduce_axes` --
    // [`push_gathered_extent_row`]/[`gathered_extent_product`] read that
    // same gather directly into `bytes` (or fold it into a product) without
    // ever materializing the intermediate row.
    push_i64(bytes, gathered_extent_product(&bound.extents, output_axes));
    push_i64(bytes, gathered_extent_product(&bound.extents, &reduce_axes));
    push_gathered_extent_row(bytes, &bound.extents, output_axes, output_rank_len);
    push_gathered_extent_row(bytes, &bound.extents, &reduce_axes, reduce_rank_len);
    for (_, layout, _) in bound.operands() {
        push_i64(bytes, layout.base);
    }
    for (_, layout, _) in bound.operands() {
        push_i64_row(bytes, &layout.strides, rank_len);
    }
    push_i64(bytes, out_layout.base);
    push_i64_row(bytes, &out_layout.strides, rank_len);
    // `crate::msl::render_reduce`'s own `Uniforms` struct declares these
    // fields ONLY when `epilogue_operands` is non-empty (byte-identical to
    // before epilogue fusion existed otherwise), so this must stay
    // conditional on the exact same test.
    if !epilogue_operands.is_empty() {
        for (_, layout, _) in epilogue_operands {
            push_i64(bytes, layout.base);
        }
        for (_, layout, _) in epilogue_operands {
            push_i64_row(bytes, &layout.strides, epilogue_stride_rank_len);
        }
    }
    if is_broadcast_epilogue {
        push_i64_row(bytes, &contiguous_strides(&bound.extents), rank_len);
    }
    push_gather_uniforms(bytes, bound, rank_len);
    Ok(())
}

/// The row-major (C-order) strides the FULL `extents` product buffer a
/// broadcast-reduce epilogue materializes is allocated with -- a genuinely
/// different address space from any [`Layout`]'s own strides (which may
/// carry a stride of `0` on a broadcast axis), computed fresh here rather
/// than read off any bound operand.
pub(super) fn contiguous_strides(extents: &[u64]) -> Vec<i64> {
    let mut strides = vec![0i64; extents.len()];
    let mut running = 1i64;
    for (axis, extent) in extents.iter().enumerate().rev() {
        strides[axis] = running;
        running *= *extent as i64;
    }
    strides
}

/// Mirrors the `Uniforms` struct `crate::msl::render_scan` declares at
/// `omega/src/msl.rs:493-503`: `outer_total`, `inner_len`,
/// `outer_extents[outer_rank_len]`, `operand_base[operand_count]`,
/// `operand_strides[operand_count][rank_len]`, `out_base`,
/// `out_strides[rank_len]`, then the gather arrays (see
/// [`pack_elementwise_uniforms`]'s doc), in that order. `crate::msl::validate`
/// already rejected a rank-0 scan before `emit` (and therefore this) ever
/// runs, so `bound.extents` is never empty here.
pub(super) fn pack_scan_uniforms(bound: &BoundOp, bytes: &mut Vec<u8>) -> Result<(), EmitError> {
    let BoundOpKind::Reduce { out_layout, .. } = &bound.kind else {
        return Err(EmitError::RenderKindMismatch {
            node: bound.node,
            expected: "keep::scan fold",
            found: bound.kind.name(),
        });
    };
    let rank = bound.extents.len();
    let rank_len = rank.max(1);
    let outer_rank = rank.saturating_sub(1);
    let outer_rank_len = outer_rank.max(1);

    let outer_extents = &bound.extents[..outer_rank];
    let inner_len = bound.extents.last().copied().unwrap_or(1) as i64;

    push_i64(
        bytes,
        outer_extents.iter().map(|extent| *extent as i64).product(),
    );
    push_i64(bytes, inner_len);
    push_extent_row(bytes, outer_extents, outer_rank_len);
    for (_, layout, _) in bound.operands() {
        push_i64(bytes, layout.base);
    }
    for (_, layout, _) in bound.operands() {
        push_i64_row(bytes, &layout.strides, rank_len);
    }
    push_i64(bytes, out_layout.base);
    push_i64_row(bytes, &out_layout.strides, rank_len);
    push_gather_uniforms(bytes, bound, rank_len);
    Ok(())
}

pub(super) fn nserror_description(error: &NSError) -> String {
    error.localizedDescription().to_string()
}

