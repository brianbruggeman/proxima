use super::*;

/// Token axes the expert-grouped kernel decomposes a flat token index into:
/// `[sequence]` for one top-k slot per dispatch, `[sequence, selected]` for the
/// stacked form that runs every selected expert of a projection at once.
#[cfg(feature = "metal-tiled-gemm")]
pub(super) const GROUPED_MAX_TOKEN_AXES: usize = 3;

/// Rows of the weight tile one expert-grouped threadgroup computes.
#[cfg(feature = "metal-grouped-gemm")]
pub(super) const GROUPED_TILE_ROWS: u64 = 64;
/// Tokens of the activation tile one expert-grouped threadgroup computes.
#[cfg(feature = "metal-grouped-gemm")]
pub(super) const GROUPED_TILE_TOKENS: u64 = 32;
/// Reduction elements one K step stages per weight row: two threads, one
/// [`DECODE_RUN_ELEMENTS`] run each. Every [`tiled_decode`] block either
/// divides or is divided by this, so a step never straddles a block.
#[cfg(feature = "metal-grouped-gemm")]
pub(super) const GROUPED_TILE_DEPTH: u64 = STAGED_K_STEP_ELEMENTS;

/// The expert-grouped counterpart of [`push_tiled_gemm_body`]: a gathered
/// expert slab `[expert, feature, reduce]` in any codec with a
/// [`tiled_decode`] description, multiplied against a dense
/// activation `[token, reduce]`, where a route index names one expert per
/// token. The tokens that name one expert are the only ones that may share a
/// weight tile, so this kernel groups them instead of tiling the token axis
/// as given.
///
/// Launch: `row_tiles * GROUPED_GEMM_ROUTE_SEGMENTS` threadgroups per z slice, one
/// z slice per expert the weight can name (`GridSpec::depth`, see
/// [`expert_group_depth`]). A threadgroup owns one `(expert, row tile, route
/// segment)`: the flat token range is cut into `GROUPED_GEMM_ROUTE_SEGMENTS`
/// contiguous segments ([`push_route_segment_bounds`]) and the threadgroup
/// walks the tokens of its expert inside its segment in rank order, in tiles
/// of [`GROUPED_TILE_TOKENS`]: it scans its segment a step at a time (every
/// thread owns `GROUPED_GEMM_SCAN_AHEAD` adjacent entries, so one barrier pair
/// covers `128 * GROUPED_GEMM_SCAN_AHEAD` entries), ranks the matching tokens
/// with a simdgroup prefix sum plus the simdgroup totals, appends them to
/// `tile_token`, and runs one full tile each time a tile's worth is pending
/// (and once more for the tail). `ggml-metal` instead compacts the route in a
/// separate pass (`kernel_mul_mm_id_map0`) and gives every token tile its own
/// threadgroup; this kernel has no second dispatch to carry that list, so each
/// threadgroup ranks its own segment of the route and the segment count is
/// what bounds the route loads per threadgroup (`token_extent / segments`
/// entries) and, with it, how finely a heavily routed expert is split. A
/// token's accumulator never depends on which tile or segment carries it:
/// the K loop, the staging and the multiply-accumulate order are per output
/// element. Every value that steers control flow (`pending_fill`, `scan_base`)
/// comes from threadgroup-shared totals, so the threadgroup leaves each branch
/// and loop together and no barrier is skipped by a subset.
///
/// An out-of-range route index is reported into the fault buffer by whichever
/// thread fetches it, and clamped exactly as the dense gather clamps it, so a
/// bad index raises the same `GatherIndexOutOfRange`.
///
/// Tile layout and thread mapping are `ggml-metal`'s `kernel_mul_mm_id`
/// (`kernels/mul_mm.metal`): both staged tiles are `half`, stored as 8x8
/// blocks of 64 contiguous elements so each `simdgroup_load` reads one
/// contiguous 128 bytes; two threads per weight row each decode one
/// [`DECODE_RUN_ELEMENTS`] run of the codec through the stager the dense tiled
/// GEMM shares ([`push_weight_cursor`], [`push_weight_decode`],
/// [`push_block_cursor_advance`]); four threads per token row convert eight
/// activations each; each
/// simdgroup accumulates a 4 by 2 grid of 8x8 blocks in float. A tile of at
/// most half [`GROUPED_TILE_TOKENS`] tokens leaves the two simdgroups that own
/// the upper token half idle. The accumulators leave through a token-major
/// threadgroup tile so the write-back walks features contiguously, one token
/// per simdgroup per step, and applies the fused epilogue per element.
#[cfg(feature = "metal-grouped-gemm")]
pub(super) fn push_expert_grouped_gemm_body(
    source: &mut String,
    resolved: &BoundOp,
    output_axes: &[u16],
    block: &TiledGemmBlock,
    element_type: &str,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
) -> Result<(), EmitError> {
    let node = resolved.node;
    let Some(expert) = block.gathered else {
        return Err(EmitError::RenderKindMismatch {
            node,
            expected: "expert-grouped gemm",
            found: "dense-weight tiled gemm",
        });
    };
    let decode = tiled_decode(block.codec).ok_or(EmitError::RenderKindMismatch {
        node,
        expected: "expert weight codec with a tiled decode description",
        found: "other packed codec",
    })?;
    let geometry = GroupedGeometry::new(block, output_axes, resolved)?;
    let compaction_buffer = grouped_route_compacted(block).then(|| bindings(resolved).len());
    push_grouped_entry(source, node, &geometry, compaction_buffer)?;
    push_grouped_threadgroup_memory(source, &geometry);
    source.push_str(&format!(
        "    device const uchar *weight_bytes = (device const uchar *)in{};\n",
        block.weight
    ));
    if compaction_buffer.is_some() {
        push_grouped_locate(source, resolved, block)?;
        source.push_str(&format!(
            "    long grouped_expert_base = grouped_expert * u.gather_element_stride[{}];\n",
            expert.slot
        ));
        source.push_str("    {\n");
        push_grouped_tile(
            source,
            block,
            &geometry,
            &decode,
            element_type,
            epilogue_body,
            epilogue_operands,
        );
        source.push_str("    }\n");
        return Ok(());
    }
    source.push_str(&format!(
        "    long grouped_expert_base = grouped_expert * u.gather_element_stride[{}];\n",
        expert.slot
    ));
    source.push_str("    uint pending_fill = 0u;\n");
    push_route_segment_bounds(source, &geometry);
    source.push_str("    for (;;) {\n");
    push_grouped_refill(source, block.weight, expert.slot, &geometry);
    source.push_str("        if (pending_fill == 0u) { break; }\n");
    source.push_str(&format!(
        "        uint tile_count = min(pending_fill, {GROUPED_TILE_TOKENS}u);\n"
    ));
    source.push_str(&format!(
        "        bool has_hi = tile_count > {}u;\n",
        GROUPED_TILE_TOKENS / 2
    ));
    source.push_str("        {\n");
    push_grouped_tile(
        source,
        block,
        &geometry,
        &decode,
        element_type,
        epilogue_body,
        epilogue_operands,
    );
    source.push_str("        }\n");
    push_grouped_consume(source, &geometry);
    source.push_str("    }\n");
    Ok(())
}

#[cfg(feature = "metal-grouped-gemm")]
const GROUPED_THREADS: u64 = (TILED_GEMM_NSG as u64) * SIMD_WIDTH;

#[cfg(feature = "metal-grouped-gemm")]
struct GroupedGeometry {
    route_segments: u64,
    scan_ahead: u64,
    rank: usize,
    output_axes: Vec<u16>,
    feature_axis: u16,
    token_axes: Vec<u16>,
    token_positions: Vec<usize>,
    route_flat: bool,
    feature_extent_expr: String,
    token_extent_expr: String,
}

#[cfg(feature = "metal-grouped-gemm")]
impl GroupedGeometry {
    fn new(
        block: &TiledGemmBlock,
        output_axes: &[u16],
        resolved: &BoundOp,
    ) -> Result<Self, EmitError> {
        let node = resolved.node;
        let (Some(_), Some(&feature_axis)) = (block.token_axes.last(), block.feature_axes.last())
        else {
            return Err(EmitError::EmptyAxisGroup {
                node,
                group: "token or feature",
            });
        };
        let token_positions = block
            .token_axes
            .iter()
            .map(|&axis| {
                output_axes
                    .iter()
                    .position(|&candidate| candidate == axis)
                    .ok_or(EmitError::AxisNotInOutputAxes { node, axis })
            })
            .collect::<Result<Vec<usize>, EmitError>>()?;
        let route_flat = block.gathered.is_some_and(|expert| expert.route_flat);
        Ok(Self {
            route_segments: crate::sized::GROUPED_GEMM_ROUTE_SEGMENTS,
            scan_ahead: crate::sized::GROUPED_GEMM_SCAN_AHEAD,
            rank: resolved.extents.len(),
            output_axes: output_axes.to_vec(),
            feature_axis,
            token_axes: block.token_axes.clone(),
            token_positions,
            route_flat,
            feature_extent_expr: group_extent(node, output_axes, &block.feature_axes)?,
            token_extent_expr: group_extent(node, output_axes, &block.token_axes)?,
        })
    }
}

#[cfg(feature = "metal-grouped-gemm")]
fn group_extent(node: NodeId, output_axes: &[u16], group: &[u16]) -> Result<String, EmitError> {
    let mut terms = Vec::with_capacity(group.len());
    for &axis in group {
        let Some(index) = output_axes.iter().position(|&candidate| candidate == axis) else {
            return Err(EmitError::AxisNotInOutputAxes { node, axis });
        };
        terms.push(format!("u.output_extents[{index}]"));
    }
    Ok(terms.join(" * "))
}

/// Declares `long {prefix}0 .. {prefix}{n-1}`, the coordinates of flat token
/// `token_expr` along each token axis, outermost first. One token axis is the
/// flat index itself. A stacked `[sequence, selected]` group peels the
/// innermost axis with a 32-bit remainder, which the kernel reaches once per
/// pending token per tile, never per route entry.
#[cfg(feature = "metal-grouped-gemm")]
fn push_token_coordinates(
    source: &mut String,
    geometry: &GroupedGeometry,
    indent: &str,
    token_expr: &str,
    prefix: &str,
) {
    source.push_str(&format!("{indent}uint {prefix}_rest = (uint)({token_expr});\n"));
    for ordinal in (1..geometry.token_positions.len()).rev() {
        let extent = format!("(uint)u.output_extents[{}]", geometry.token_positions[ordinal]);
        source.push_str(&format!(
            "{indent}long {prefix}{ordinal} = (long)({prefix}_rest % {extent});\n"
        ));
        source.push_str(&format!("{indent}{prefix}_rest /= {extent};\n"));
    }
    source.push_str(&format!("{indent}long {prefix}0 = (long){prefix}_rest;\n"));
}

/// The sum of each token coordinate declared by [`push_token_coordinates`]
/// times its stride, where `stride` names the stride of one token axis in some
/// layout (an operand, the output, the route index).
#[cfg(feature = "metal-grouped-gemm")]
fn token_offset_expr(
    geometry: &GroupedGeometry,
    prefix: &str,
    stride: impl Fn(u16) -> String,
) -> String {
    geometry
        .token_axes
        .iter()
        .enumerate()
        .map(|(ordinal, &axis)| format!("{prefix}{ordinal} * {}", stride(axis)))
        .collect::<Vec<String>>()
        .join(" + ")
}

/// Widens the scalar `uint gid` the kernel signature carries to a `uint3` so
/// the expert can ride the grid's z coordinate -- the widening
/// [`push_dense_batched_gemm_body`] performs for its batch axis -- then names
/// the threadgroup's coordinates: `gid / GROUPED_THREADS` is the flat group
/// index within one z slice, split into `(row_tile, route_segment)`.
#[cfg(feature = "metal-grouped-gemm")]
fn push_grouped_entry(
    source: &mut String,
    node: NodeId,
    geometry: &GroupedGeometry,
    compaction_buffer: Option<usize>,
) -> Result<(), EmitError> {
    let scalar_gid = "uint gid [[thread_position_in_grid]]";
    let vector_gid = match compaction_buffer {
        Some(slot) => format!(
            "device const uint *route_compaction [[buffer({slot})]],\n    uint3 grouped_gid [[thread_position_in_grid]]"
        ),
        None => "uint3 grouped_gid [[thread_position_in_grid]]".to_string(),
    };
    let gid_offset = source.find(scalar_gid).ok_or(EmitError::RenderKindMismatch {
        node,
        expected: "scalar thread_position_in_grid parameter",
        found: "missing",
    })?;
    source.replace_range(gid_offset..gid_offset + scalar_gid.len(), &vector_gid);
    source.push_str("    long gid = (long)grouped_gid.x;\n");
    if compaction_buffer.is_none() {
        source.push_str("    long grouped_expert = (long)grouped_gid.z;\n");
    }
    source.push_str(&format!(
        "    long feature_extent = {};\n",
        geometry.feature_extent_expr
    ));
    source.push_str(&format!("    long token_extent = {};\n", geometry.token_extent_expr));
    source.push_str(&format!("    long tiitg = (long)gid % {GROUPED_THREADS};\n"));
    source.push_str(&format!("    long sgitg = tiitg / {SIMD_WIDTH};\n"));
    source.push_str(&format!(
        "    long group_index = (long)gid / {GROUPED_THREADS};\n"
    ));
    if compaction_buffer.is_some() {
        source.push_str(&format!(
            "    long row_tiles = (feature_extent + {}l) / {GROUPED_TILE_ROWS}l;\n",
            GROUPED_TILE_ROWS - 1
        ));
        source.push_str("    long tile_index = group_index / row_tiles;\n");
        source.push_str("    long row_tile = group_index % row_tiles;\n");
    } else {
        source.push_str(&format!(
            "    long row_tile = group_index / {};\n",
            geometry.route_segments
        ));
        source.push_str(&format!(
            "    long route_segment = group_index % {};\n",
            geometry.route_segments
        ));
    }
    source.push_str(&format!(
        "    uint grouped_lane = (uint)(tiitg % {SIMD_WIDTH});\n"
    ));
    Ok(())
}

/// The compacted route's replacement for the scan: finds this threadgroup's
/// `(expert, tile)` in the prepass tables and stages that tile's token ids.
/// The launch covers [`grouped_tile_upper_bound`] token tiles per row tile, so
/// a threadgroup walks the per-expert tile counts until its `tile_index` falls
/// inside one expert's tiles; a `tile_index` past the realized total finds no
/// expert and leaves, before any barrier, together with the rest of its
/// threadgroup (every thread read the same words). The prepass total is
/// checked against the route length first: a buffer the prepass did not write
/// (its header word is zeroed on the host) records
/// [`ROUTE_COMPACTION_MISMATCH_FAULT`], which the host reports as a
/// compaction mismatch, and the threadgroup leaves without computing.
#[cfg(feature = "metal-grouped-gemm")]
fn push_grouped_locate(
    source: &mut String,
    resolved: &BoundOp,
    block: &TiledGemmBlock,
) -> Result<(), EmitError> {
    let Some(expert) = block.gathered else {
        return Ok(());
    };
    let slot = expert.slot;
    let experts = grouped_expert_count(resolved, block);
    let tile_tokens = GROUPED_TILE_TOKENS;
    source.push_str(&format!("    constexpr uint experts = {experts}u;\n"));
    source.push_str("    if (route_compaction[0] != (uint)token_extent) {\n");
    source.push_str("        if (tiitg == 0) {\n");
    source.push_str(&format!(
        "            atomic_fetch_max_explicit(&fault[{slot}], {ROUTE_COMPACTION_MISMATCH_FAULT}u, memory_order_relaxed);\n"
    ));
    source.push_str("        }\n");
    source.push_str("        return;\n");
    source.push_str("    }\n");
    source.push_str("    long grouped_expert = -1;\n");
    source.push_str("    long expert_tile = 0;\n");
    source.push_str("    uint expert_tokens = 0u;\n");
    source.push_str("    long tile_base = 0;\n");
    source.push_str("    for (uint scan_expert = 0u; scan_expert < experts; ++scan_expert) {\n");
    source.push_str(&format!(
        "        uint expert_count = route_compaction[{GROUPED_ROUTE_HEADER_WORDS}u + scan_expert];\n"
    ));
    source.push_str(&format!(
        "        long expert_tiles = (long)((expert_count + {}u) / {tile_tokens}u);\n",
        tile_tokens - 1
    ));
    source.push_str("        if (tile_index < tile_base + expert_tiles) {\n");
    source.push_str("            grouped_expert = (long)scan_expert;\n");
    source.push_str("            expert_tile = tile_index - tile_base;\n");
    source.push_str("            expert_tokens = expert_count;\n");
    source.push_str("            break;\n");
    source.push_str("        }\n");
    source.push_str("        tile_base += expert_tiles;\n");
    source.push_str("    }\n");
    source.push_str("    if (grouped_expert < 0) { return; }\n");
    source.push_str(&format!(
        "    uint tile_count = min({tile_tokens}u, expert_tokens - (uint)expert_tile * {tile_tokens}u);\n"
    ));
    source.push_str(&format!(
        "    bool has_hi = tile_count > {}u;\n",
        tile_tokens / 2
    ));
    source.push_str(&format!(
        "    device const uint *tile_list = route_compaction + {GROUPED_ROUTE_HEADER_WORDS}u + 2u * experts + route_compaction[{GROUPED_ROUTE_HEADER_WORDS}u + experts + (uint)grouped_expert] + (uint)expert_tile * {tile_tokens}u;\n"
    ));
    source.push_str("    if ((uint)tiitg < tile_count) { tile_token[tiitg] = (int)tile_list[tiitg]; }\n");
    source.push_str("    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    Ok(())
}

/// The slice of the route this threadgroup scans: `GROUPED_GEMM_ROUTE_SEGMENTS`
/// equal contiguous segments of the flat token range, threadgroup `route_segment`
/// owning `[segment_begin, segment_end)`. Every route entry belongs to exactly
/// one segment, so a `(expert, row tile)` loads the route once across its
/// threadgroups instead of once each; a segment past the end of the route is
/// empty and its threadgroup leaves at the first `pending_fill` test.
#[cfg(feature = "metal-grouped-gemm")]
fn push_route_segment_bounds(source: &mut String, geometry: &GroupedGeometry) {
    let segments = geometry.route_segments;
    source.push_str(&format!(
        "    long segment_length = (token_extent + {segments}l - 1l) / {segments}l;\n"
    ));
    source.push_str("    long segment_begin = route_segment * segment_length;\n");
    source.push_str("    long segment_end = min(token_extent, segment_begin + segment_length);\n");
    source.push_str("    long scan_base = segment_begin;\n");
}

/// `weight_tile` and `act_tile` are the two staged tiles of the K loop;
/// `out_tile` is the token-major accumulator restage after it and aliases
/// them (the K loop's trailing barrier fences the last tile read before any
/// thread reaches the aliased write). `tile_token` holds the pending token
/// ids: one tile plus the largest number of matches a scan step can append,
/// which is every route entry the step covers.
#[cfg(feature = "metal-grouped-gemm")]
fn push_grouped_threadgroup_memory(source: &mut String, geometry: &GroupedGeometry) {
    let weight_tile_bytes = GROUPED_TILE_ROWS * GROUPED_TILE_DEPTH * 2;
    let act_tile_bytes = GROUPED_TILE_TOKENS * GROUPED_TILE_DEPTH * 2;
    let out_tile_bytes = GROUPED_TILE_ROWS * GROUPED_TILE_TOKENS * 4;
    let shared_bytes = (weight_tile_bytes + act_tile_bytes).max(out_tile_bytes);
    source.push_str(&format!("    threadgroup uchar tg_shared[{shared_bytes}];\n"));
    source.push_str("    threadgroup half *weight_tile = (threadgroup half *)tg_shared;\n");
    source.push_str(&format!(
        "    threadgroup half *act_tile = (threadgroup half *)(tg_shared + {weight_tile_bytes});\n"
    ));
    source.push_str("    threadgroup float *out_tile = (threadgroup float *)tg_shared;\n");
    source.push_str(&format!(
        "    threadgroup int tile_token[{}];\n",
        GROUPED_TILE_TOKENS + GROUPED_THREADS * geometry.scan_ahead
    ));
    source.push_str(&format!(
        "    threadgroup uint scan_counts[{TILED_GEMM_NSG}];\n"
    ));
}

/// Scans route entries until a full tile of this expert's tokens is pending
/// or the route is exhausted. One scan step covers
/// `GROUPED_THREADS * GROUPED_GEMM_SCAN_AHEAD` consecutive route entries:
/// each thread owns `GROUPED_GEMM_SCAN_AHEAD` adjacent entries, loads all of
/// them back to back (one memory round trip serves the step), reports an
/// out-of-range index, clamps it, and counts how many name this expert. The
/// simdgroup prefix sum ranks the threads inside a simdgroup and
/// `scan_counts` carries the four simdgroup totals across the one barrier
/// pair the step pays, so ranks ascend in flat token order. The whole step
/// is appended -- `tile_token` is sized for a full tile plus one step -- so
/// no route entry is ever fetched twice.
#[cfg(feature = "metal-grouped-gemm")]
fn push_grouped_refill(
    source: &mut String,
    weight: usize,
    slot: usize,
    geometry: &GroupedGeometry,
) {
    let innermost_axis = geometry.token_axes.last().copied().unwrap_or(0);
    let entries = geometry.scan_ahead;
    source.push_str(&format!(
        "        while (pending_fill < {GROUPED_TILE_TOKENS}u && scan_base < segment_end) {{\n"
    ));
    source.push_str(&format!(
        "            long own_base = scan_base + tiitg * {entries};\n"
    ));
    source.push_str(&format!("            long routed_entry[{entries}];\n"));
    source.push_str(&format!(
        "            for (int entry = 0; entry < {entries}; ++entry) {{\n"
    ));
    source.push_str("                long entry_token = own_base + entry;\n");
    let route_offset = if geometry.route_flat {
        format!("entry_token * u.gather_index_strides[{slot}][{innermost_axis}]")
    } else {
        push_token_coordinates(
            source,
            geometry,
            "                ",
            "(entry_token < segment_end) ? entry_token : 0",
            "route_c",
        );
        token_offset_expr(geometry, "route_c", |axis| {
            format!("u.gather_index_strides[{slot}][{axis}]")
        })
    };
    source.push_str(&format!(
        "                routed_entry[entry] = (entry_token < segment_end) ? (long)gather_idx{slot}[u.gather_index_base[{slot}] + {route_offset}] : (long)-1;\n"
    ));
    source.push_str("            }\n");
    source.push_str("            uint own_count = 0u;\n");
    source.push_str(&format!(
        "            for (int entry = 0; entry < {entries}; ++entry) {{\n"
    ));
    source.push_str(&format!(
        "                long fetched{weight} = routed_entry[entry];\n"
    ));
    source.push_str("                if (own_base + entry < segment_end) {\n");
    push_gather_fault_check(source, weight, slot, "                    ");
    source.push_str(&format!(
        "                    fetched{weight} = max((long)0, min(fetched{weight}, u.gather_extent[{slot}] - 1));\n"
    ));
    source.push_str("                }\n");
    source.push_str(&format!("                routed_entry[entry] = fetched{weight};\n"));
    source.push_str(&format!(
        "                own_count += (fetched{weight} == grouped_expert) ? 1u : 0u;\n"
    ));
    source.push_str("            }\n");
    source.push_str("            uint scan_prefix = simd_prefix_exclusive_sum(own_count);\n");
    source.push_str("            uint scan_total = simd_sum(own_count);\n");
    source.push_str("            if (grouped_lane == 0u) { scan_counts[sgitg] = scan_total; }\n");
    source.push_str("            threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    source.push_str("            uint chunk_before = 0u;\n");
    source.push_str("            uint chunk_total = 0u;\n");
    source.push_str(&format!(
        "            for (int group = 0; group < {TILED_GEMM_NSG}; ++group) {{\n"
    ));
    source.push_str("                uint group_count = scan_counts[group];\n");
    source.push_str("                chunk_before += (group < (int)sgitg) ? group_count : 0u;\n");
    source.push_str("                chunk_total += group_count;\n");
    source.push_str("            }\n");
    source.push_str("            uint write_at = pending_fill + chunk_before + scan_prefix;\n");
    source.push_str(&format!(
        "            for (int entry = 0; entry < {entries}; ++entry) {{\n"
    ));
    source.push_str("                if (routed_entry[entry] == grouped_expert) {\n");
    source.push_str("                    tile_token[write_at] = (int)(own_base + entry);\n");
    source.push_str("                    write_at += 1u;\n");
    source.push_str("                }\n");
    source.push_str("            }\n");
    source.push_str("            pending_fill += chunk_total;\n");
    source.push_str(&format!(
        "            scan_base += {GROUPED_THREADS}l * {entries}l;\n"
    ));
    source.push_str("            threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    source.push_str("        }\n");
}

/// Drops the tokens the tile just consumed: whatever was pending beyond one
/// tile (fewer than one scan step's worth, so at most `GROUPED_GEMM_SCAN_AHEAD`
/// elements per thread) moves to the front. Read-barrier-write, because the
/// source and destination ranges overlap; skipped, barriers included, when
/// nothing is carried, which the threadgroup decides together from
/// `pending_fill`.
#[cfg(feature = "metal-grouped-gemm")]
fn push_grouped_consume(source: &mut String, geometry: &GroupedGeometry) {
    let block_n = GROUPED_TILE_TOKENS;
    let entries = geometry.scan_ahead;
    source.push_str(&format!(
        "        uint carried_count = (pending_fill > {block_n}u) ? (pending_fill - {block_n}u) : 0u;\n"
    ));
    source.push_str("        if (carried_count != 0u) {\n");
    source.push_str(&format!("            int carried_token[{entries}];\n"));
    source.push_str(&format!(
        "            for (int carry = 0; carry < {entries}; ++carry) {{\n"
    ));
    source.push_str(&format!(
        "                uint carry_index = (uint)tiitg + (uint)carry * {GROUPED_THREADS}u;\n"
    ));
    source.push_str(&format!(
        "                carried_token[carry] = (carry_index < carried_count) ? tile_token[carry_index + {block_n}u] : 0;\n"
    ));
    source.push_str("            }\n");
    source.push_str("            threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    source.push_str(&format!(
        "            for (int carry = 0; carry < {entries}; ++carry) {{\n"
    ));
    source.push_str(&format!(
        "                uint carry_index = (uint)tiitg + (uint)carry * {GROUPED_THREADS}u;\n"
    ));
    source.push_str(
        "                if (carry_index < carried_count) { tile_token[carry_index] = carried_token[carry]; }\n",
    );
    source.push_str("            }\n");
    source.push_str("            threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    source.push_str("        }\n");
    source.push_str("        pending_fill = carried_count;\n");
}

/// One tile for the tokens in `tile_token[0..tile_count)`: stage,
/// multiply-accumulate, K step by K step, then write back. Each K step is
/// `ggml-metal`'s order: global loads and the decode are issued into registers
/// before the barrier that waits for the previous step's multiply-accumulate,
/// so the memory latency overlaps that wait; the registers are stored to the
/// tiles after the barrier. Per-thread row and token pointers are formed once
/// per tile and advance by one K step of the weight cursor
/// ([`push_block_cursor_advance`]) and one K step of activations (32 floats)
/// per step. Measured on a `Q8_0` gathered 8-of-32-expert (1024 by 512)
/// gate projection, uniform routing, 1000 tokens, tiles dealt round robin to four threadgroups
/// (`expert_grouped_gemm_speed_probe`): 700 us with the staging inside the
/// barrier pair and the pointers rebuilt every step, 402 us in this order.
#[cfg(feature = "metal-grouped-gemm")]
fn push_grouped_tile(
    source: &mut String,
    block: &TiledGemmBlock,
    geometry: &GroupedGeometry,
    decode: &TiledDecode,
    element_type: &str,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
) {
    source.push_str("            simdgroup_float8x8 acc[8];\n");
    source.push_str(
        "            for (int i = 0; i < 8; ++i) { acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f); }\n",
    );
    push_grouped_stage_pointers(source, block, geometry, decode);
    source.push_str(&format!(
        "            for (long k0 = 0; k0 < u.reduction_total; k0 += {GROUPED_TILE_DEPTH}) {{\n"
    ));
    source.push_str("                half4 a_regs[2];\n");
    push_grouped_load(source, decode, "                ");
    source.push_str("                threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    push_grouped_store(source);
    source.push_str("                threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    push_grouped_multiply_accumulate(source);
    push_block_cursor_advance(
        source,
        "                ",
        "w_blk",
        "w_pos",
        GROUPED_TILE_DEPTH,
        decode,
    );
    source.push_str(&format!("                a_ptr += {GROUPED_TILE_DEPTH};\n"));
    source.push_str("            }\n");
    source.push_str("            threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    push_grouped_writeback(source, geometry, element_type, epilogue_body, epilogue_operands);
    source.push_str("            threadgroup_barrier(mem_flags::mem_threadgroup);\n");
}

/// What each thread stages, fixed for the whole tile. Weights: two threads
/// per tile row, `w_blk` and `w_pos` the row's weight cursor for this K step
/// (valid only while `w_valid`, a row past `feature_extent` stages zeros so its
/// accumulators stay zero; `w_half` offsets the second thread by one decode run),
/// `w_slot` its slot in the 8x8-block layout (block
/// `8 * k_block + row_block`, element `8 * (k % 8) + (row % 8)`).
/// Activations: four threads per pending token, `a_ptr` the token's row at
/// this thread's eight-element K block, `a_slot` its 16-byte row of the 8x8
/// block `4 * k_block + token_block`. `a_staged` is false for the upper token
/// half of a tile that has no token there (`has_hi`), `a_vector` selects
/// `float4` loads when the operand base and token stride keep every row
/// 16-byte aligned, else the scalar loop runs -- a runtime test on two
/// uniforms, uniform across the threadgroup.
#[cfg(feature = "metal-grouped-gemm")]
fn push_grouped_stage_pointers(
    source: &mut String,
    block: &TiledGemmBlock,
    geometry: &GroupedGeometry,
    decode: &TiledDecode,
) {
    let weight = block.weight;
    let other = block.other;
    let feature_axis = geometry.feature_axis;
    source.push_str("            long w_row = tiitg / 2;\n");
    source.push_str("            long w_half = tiitg % 2;\n");
    source.push_str(&format!(
        "            long w_feat = row_tile * {GROUPED_TILE_ROWS} + w_row;\n"
    ));
    source.push_str("            bool w_valid = w_feat < feature_extent;\n");
    source.push_str(
        "            threadgroup half *w_slot = weight_tile + 1024 * w_half + 64 * (w_row / 8) + (w_row % 8);\n",
    );
    source.push_str(&format!(
        "            long w_elem = u.operand_base[{weight}] + grouped_expert_base + (w_valid ? w_feat : 0) * u.operand_strides[{weight}][{feature_axis}] + w_half * {DECODE_RUN_ELEMENTS};\n"
    ));
    push_weight_cursor(source, "            ", decode, "w_blk", "w_pos", "w_elem");
    source.push_str("            long a_row = tiitg / 4;\n");
    source.push_str("            long a_k_block = tiitg % 4;\n");
    source.push_str(&format!(
        "            bool a_staged = a_row < {} || has_hi;\n",
        GROUPED_TILE_TOKENS / 2
    ));
    source.push_str(
        "            long a_tok = (a_row < (long)tile_count) ? (long)tile_token[a_row] : -1;\n",
    );
    source.push_str(
        "            threadgroup half4 *a_slot = (threadgroup half4 *)(act_tile + 64 * (4 * a_k_block + a_row / 8) + 8 * (a_row % 8));\n",
    );
    push_token_coordinates(source, geometry, "            ", "(a_tok < 0 ? 0 : a_tok)", "a_c");
    let activation_offset = token_offset_expr(geometry, "a_c", |axis| {
        format!("u.operand_strides[{other}][{axis}]")
    });
    source.push_str(&format!(
        "            device const float *a_ptr = in{other} + u.operand_base[{other}] + {activation_offset} + a_k_block * 8;\n"
    ));
    let alignment_terms = geometry
        .token_axes
        .iter()
        .map(|axis| format!(" | u.operand_strides[{other}][{axis}]"))
        .collect::<String>();
    source.push_str(&format!(
        "            bool a_vector = ((u.operand_base[{other}]{alignment_terms}) & 3) == 0;\n"
    ));
}

/// Issues one step's global loads and decodes into `w_regs`/`a_regs`: the row's
/// [`DECODE_RUN_ELEMENTS`] weights decode through the codec's [`tiled_decode`]
/// function (a row past `feature_extent` stages zeros), the activation row's
/// eight floats are two `float4` loads converted to `half`.
#[cfg(feature = "metal-grouped-gemm")]
fn push_grouped_load(source: &mut String, decode: &TiledDecode, indent: &str) {
    let weight_load = [
        format!("half w_regs[{DECODE_RUN_ELEMENTS}];"),
        "if (w_valid) {".to_string(),
        format!("    {};", decode.call("w_blk", "w_pos", "w_regs")),
        "} else {".to_string(),
        format!("    for (int element = 0; element < {DECODE_RUN_ELEMENTS}; ++element) {{ w_regs[element] = 0.0h; }}"),
        "}".to_string(),
    ];
    for line in weight_load {
        source.push_str(indent);
        source.push_str(&line);
        source.push('\n');
    }
    let lines = [
        "if (a_staged) {",
        "    if (a_tok < 0) {",
        "        a_regs[0] = half4(0.0h); a_regs[1] = half4(0.0h);",
        "    } else if (a_vector) {",
        "        device const float4 *a_src = (device const float4 *)a_ptr;",
        "        a_regs[0] = half4(a_src[0]); a_regs[1] = half4(a_src[1]);",
        "    } else {",
        "        for (int lane = 0; lane < 4; ++lane) {",
        "            a_regs[0][lane] = (half)a_ptr[lane]; a_regs[1][lane] = (half)a_ptr[4 + lane];",
        "        }",
        "    }",
        "}",
    ];
    for line in lines {
        source.push_str(indent);
        source.push_str(line);
        source.push('\n');
    }
}

/// Stores the registers to the tiles: the weight row's 16 elements one by one
/// into their 8x8-block slots (element `i` of the 16 sits at `512 * (i / 8) +
/// 8 * (i % 8)` from `w_slot`), the activation row as two 16-byte stores.
#[cfg(feature = "metal-grouped-gemm")]
fn push_grouped_store(source: &mut String) {
    for element in 0..DECODE_RUN_ELEMENTS {
        let slot = 512 * (element / 8) + 8 * (element % 8);
        source.push_str(&format!(
            "                w_slot[{slot}] = w_regs[{element}];\n"
        ));
    }
    source.push_str("                if (a_staged) { a_slot[0] = a_regs[0]; a_slot[1] = a_regs[1]; }\n");
}

/// The `simdgroup_matrix` step over the staged tiles, one 8-deep K block at a
/// time: simdgroup `s` holds the weight blocks of row half `s % 2` and the
/// activation blocks of token half `s / 2`, and accumulates their 4 by 2
/// outer products into eight 8x8 float accumulators, `acc[i]` for token block
/// `i / 4` and row block `i % 4`. The two simdgroups that own the upper token
/// half skip the step when the tile has no token there; their accumulators
/// stay zero.
#[cfg(feature = "metal-grouped-gemm")]
fn push_grouped_multiply_accumulate(source: &mut String) {
    source.push_str("                if ((sgitg >> 1) == 0 || has_hi) {\n");
    source.push_str(
        "                    threadgroup const half *lsma = weight_tile + 4 * 64 * (sgitg % 2);\n",
    );
    source.push_str(
        "                    threadgroup const half *lsmb = act_tile + 2 * 64 * (sgitg / 2);\n",
    );
    source.push_str(&format!(
        "                    for (short ik = 0; ik < {}; ++ik) {{\n",
        GROUPED_TILE_DEPTH / 8
    ));
    source.push_str("                        simdgroup_half8x8 ma[4];\n");
    source.push_str("                        simdgroup_half8x8 mb[2];\n");
    source.push_str(
        "                        for (short i = 0; i < 4; ++i) { simdgroup_load(ma[i], lsma + 64 * i, 8, ulong2(0), false); }\n",
    );
    source.push_str(
        "                        for (short i = 0; i < 2; ++i) { simdgroup_load(mb[i], lsmb + 64 * i, 8, ulong2(0), false); }\n",
    );
    source.push_str(
        "                        for (short i = 0; i < 8; ++i) { simdgroup_multiply_accumulate(acc[i], mb[i / 4], ma[i % 4], acc[i]); }\n",
    );
    source.push_str("                        lsma += 8 * 64;\n");
    source.push_str("                        lsmb += 4 * 64;\n");
    source.push_str("                    }\n");
    source.push_str("                }\n");
}

/// The accumulators go to `out_tile` token-major (`out_tile[token][feature]`,
/// feature contiguous), then each simdgroup walks the pending tokens four
/// apart while its 32 lanes walk the tile's features, so a store instruction
/// touches 32 consecutive features of one token. The fused epilogue runs per
/// element through [`push_reduce_epilogue_write`], reading its operands at the
/// element's real `(token, feature)` coordinate.
#[cfg(feature = "metal-grouped-gemm")]
fn push_grouped_writeback(
    source: &mut String,
    geometry: &GroupedGeometry,
    element_type: &str,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
) {
    let output_axes = &geometry.output_axes;
    let rank = geometry.rank;
    let rank_len = rank.max(1);
    let feature_axis = geometry.feature_axis;
    source.push_str("            if ((sgitg >> 1) == 0 || has_hi) {\n");
    source.push_str(&format!(
        "                threadgroup float *temp_str = out_tile + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * {GROUPED_TILE_ROWS};\n"
    ));
    source.push_str("                for (short i = 0; i < 8; ++i) {\n");
    source.push_str(&format!(
        "                    simdgroup_store(acc[i], temp_str + 8 * (i % 4) + 8 * {GROUPED_TILE_ROWS} * (i / 4), {GROUPED_TILE_ROWS}, ulong2(0), false);\n"
    ));
    source.push_str("                }\n");
    source.push_str("            }\n");
    source.push_str("            threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    source.push_str(&format!("            long coord[{rank_len}];\n"));
    source.push_str(&format!(
        "            for (int d = 0; d < {rank}; ++d) {{ coord[d] = 0; }}\n"
    ));
    source.push_str(&format!(
        "            for (long j = sgitg; j < (long)tile_count; j += {TILED_GEMM_NSG}) {{\n"
    ));
    source.push_str("                long o_tok = (long)tile_token[j];\n");
    push_token_coordinates(source, geometry, "                ", "o_tok", "o_c");
    let output_token_offset = token_offset_expr(geometry, "o_c", |axis| {
        format!("u.out_strides[{axis}]")
    });
    source.push_str(&format!(
        "                for (long o_col = (long)grouped_lane; o_col < {GROUPED_TILE_ROWS}; o_col += {SIMD_WIDTH}) {{\n"
    ));
    source.push_str(&format!(
        "                    long o_feat = row_tile * {GROUPED_TILE_ROWS} + o_col;\n"
    ));
    source.push_str("                    if (o_feat < feature_extent) {\n");
    source.push_str(&format!("                        coord[{feature_axis}] = o_feat;\n"));
    for (ordinal, axis) in geometry.token_axes.iter().enumerate() {
        source.push_str(&format!(
            "                        coord[{axis}] = o_c{ordinal};\n"
        ));
    }
    source.push_str(&format!(
        "                        long out_offset = u.out_base + o_feat * u.out_strides[{feature_axis}] + {output_token_offset};\n"
    ));
    let accumulator_expr = format!("({element_type})out_tile[j * {GROUPED_TILE_ROWS} + o_col]");
    push_reduce_epilogue_write(
        source,
        epilogue_body,
        epilogue_operands,
        output_axes.len(),
        element_type,
        "                        ",
        |dim| format!("coord[{}]", output_axes[dim]),
        &accumulator_expr,
        "out_offset",
    );
    source.push_str("                    }\n");
    source.push_str("                }\n");
    source.push_str("            }\n");
}

/// The token extent [`tiled_gemm_threadgroups`] is asked to launch for an
/// expert-grouped block, which is not the token count: the segment scan
/// launches `GROUPED_GEMM_ROUTE_SEGMENTS` threadgroups per row tile, each
/// walking its slice of the route; the compacted route launches the upper
/// bound of token tiles ([`grouped_tile_upper_bound`]) per row tile, one
/// threadgroup each, and the tiles past the realized total exit at once.
#[cfg(feature = "metal-grouped-gemm")]
pub(super) fn expert_grouped_launch_tokens(resolved: &BoundOp, block: &TiledGemmBlock) -> Option<u64> {
    if !is_expert_grouped(block) {
        return None;
    }
    let tiles = if grouped_route_compacted(block) {
        grouped_tile_upper_bound(
            grouped_token_total(resolved, block),
            grouped_expert_count(resolved, block),
        )
    } else {
        crate::sized::GROUPED_GEMM_ROUTE_SEGMENTS
    };
    Some(tiles * GROUPED_TILE_TOKENS)
}

#[cfg(not(feature = "metal-grouped-gemm"))]
pub(super) fn expert_grouped_launch_tokens(_resolved: &BoundOp, _block: &TiledGemmBlock) -> Option<u64> {
    None
}
/// Never invoked: without `metal-grouped-gemm`, [`classify_tiled_gemm`] never
/// returns an expert-grouped block, so [`push_cooperative_reduce_body`]'s
/// `is_expert_grouped` arm is unreachable -- this exists so that arm
/// type-checks in that build, the way [`push_tiled_gemm_body`]'s own stub does.
#[cfg(not(feature = "metal-grouped-gemm"))]
pub(super) fn push_expert_grouped_gemm_body(
    _source: &mut String,
    resolved: &BoundOp,
    _output_axes: &[u16],
    _block: &TiledGemmBlock,
    _element_type: &str,
    _epilogue_body: &ComposedBody,
    _epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
) -> Result<(), EmitError> {
    Err(EmitError::RenderKindMismatch {
        node: resolved.node,
        expected: "expert-grouped gemm",
        found: "build without metal-grouped-gemm",
    })
}

/// The fault word the locating GEMM records when the compaction header does
/// not match the route length. Ordinary route faults record `index + 1` and
/// expert-source misses set the top bit with `expert + 1`; the all-ones word
/// is neither, so the host names the stale prepass instead of reporting a
/// route index equal to the expert extent.
#[cfg(any(feature = "metal-grouped-gemm", all(feature = "metal", target_os = "macos")))]
pub(crate) const ROUTE_COMPACTION_MISMATCH_FAULT: u32 = u32::MAX;

/// Words of the route-compaction buffer ahead of the per-expert tables: word 0
/// is the total entry count the prepass wrote, the check the GEMM makes before
/// it trusts the tables.
#[cfg(feature = "metal-grouped-gemm")]
pub(super) const GROUPED_ROUTE_HEADER_WORDS: u64 = 1;

/// Threads of each prepass threadgroup: a simdgroup per expert,
/// `GROUPED_PREPASS_THREADS / SIMD_WIDTH` experts at a time.
#[cfg(feature = "metal-grouped-gemm")]
const GROUPED_PREPASS_THREADS: u64 = 1024;

/// Whether this expert-grouped block takes the compacted route: a prepass
/// dispatch ranks the route into per-expert token lists and the GEMM locates
/// its tile in them. The prepass reads the route as the flat stride the
/// segment scan already requires, so a route that needs the per-axis
/// decomposition stays on the segment scan.
#[cfg(feature = "metal-grouped-gemm")]
pub(super) fn grouped_route_compacted(block: &TiledGemmBlock) -> bool {
    crate::sized::GROUPED_GEMM_ROUTE_COMPACTED
        && block.gathered.is_some_and(|expert| expert.route_flat)
}

#[cfg(all(feature = "metal-tiled-gemm", not(feature = "metal-grouped-gemm")))]
pub(super) fn grouped_route_compacted(_block: &TiledGemmBlock) -> bool {
    false
}

/// Experts the weight slab can name: the extent of its gathered dimension.
#[cfg(feature = "metal-grouped-gemm")]
pub(super) fn grouped_expert_count(resolved: &BoundOp, block: &TiledGemmBlock) -> u64 {
    resolved.operands()[block.weight]
        .2
        .as_ref()
        .map_or(1, |lookup| lookup.extent.max(1))
}

/// Route entries: the flat token count the route names an expert for.
#[cfg(feature = "metal-grouped-gemm")]
pub(super) fn grouped_token_total(resolved: &BoundOp, block: &TiledGemmBlock) -> u64 {
    block
        .token_axes
        .iter()
        .map(|&axis| resolved.extents[axis as usize])
        .product()
}

/// Token tiles a launch must cover whatever the realized route is: every full
/// tile of the whole route, plus one partial tile per expert.
#[cfg(feature = "metal-grouped-gemm")]
pub(super) fn grouped_tile_upper_bound(tokens: u64, experts: u64) -> u64 {
    tokens.div_ceil(GROUPED_TILE_TOKENS) + experts
}

/// Threadgroups of each prepass dispatch: the route cut into chunks of
/// `GROUPED_GEMM_PREPASS_ENTRIES` entries, one threadgroup per chunk, at least one.
#[cfg(feature = "metal-grouped-gemm")]
pub(super) fn grouped_prepass_threadgroups(tokens: u64) -> u64 {
    tokens.div_ceil(crate::sized::GROUPED_GEMM_PREPASS_ENTRIES).max(1)
}

/// Words of the compaction buffer: header, per-expert counts, per-expert
/// exclusive entry offsets, the expert-major list of flat token ids, then the
/// count pass's partial counts (`threadgroups x experts`, row per threadgroup)
/// that the place pass reduces. The GEMM reads only the first four regions.
#[cfg(feature = "metal-grouped-gemm")]
pub(super) fn grouped_compaction_words(tokens: u64, experts: u64) -> u64 {
    GROUPED_ROUTE_HEADER_WORDS + 2 * experts + tokens + grouped_prepass_threadgroups(tokens) * experts
}

#[cfg(feature = "metal-grouped-gemm")]
fn grouped_prepass_entry(entry: &str) -> String {
    format!("{entry}_route_prepass")
}

#[cfg(feature = "metal-grouped-gemm")]
fn grouped_prepass_place_entry(entry: &str) -> String {
    format!("{entry}_route_prepass_place")
}

/// The prepass kernels, appended to the GEMM's source so they share the
/// `Uniforms` struct, and the route fetch (stride, base, clamp, fault check)
/// of the main kernel. They declare only the buffers they touch, at the slots
/// the GEMM binds them at, so the host encodes them between the GEMM's binds
/// and the GEMM's dispatch without rebinding: the gathered route, the
/// uniforms, the fault flags and the compaction buffer at `bindings.len()`.
///
/// Two dispatches of `ceil(tokens / GROUPED_GEMM_PREPASS_ENTRIES)`
/// threadgroups, threadgroup `t` owning the route entries
/// `[t * entries, (t + 1) * entries)`. The count pass: simdgroup `s` counts
/// the entries of its chunk naming expert `s`, `s + simdgroups`, ... a
/// simdgroup width at a time and writes the counts to the threadgroup's row of
/// the partial-count scratch after the list; simdgroup zero reports
/// out-of-range route indices. The place pass: every threadgroup sums the
/// scratch column of each expert over all threadgroups (the expert's count)
/// and over the threadgroups before it (what the earlier chunks place), thread
/// zero prefix-sums the counts into the offsets, and each simdgroup places the
/// matching token ids of its chunk at the expert's offset plus the earlier
/// chunks' count plus a simdgroup prefix-sum rank, so each expert's list
/// ascends in flat token order exactly as one threadgroup scanning the whole
/// route would write it. Threadgroup zero publishes the header, counts and
/// offsets. No device atomics: the only ordering across threadgroups is the
/// dispatch boundary between the two passes.
#[cfg(feature = "metal-grouped-gemm")]
pub(super) fn push_grouped_route_prepass(
    source: &mut String,
    resolved: &BoundOp,
    block: &TiledGemmBlock,
    output_axes: &[u16],
    entry: &str,
) -> Result<(), EmitError> {
    let Some(expert) = block.gathered else {
        return Ok(());
    };
    let geometry = GroupedGeometry::new(block, output_axes, resolved)?;
    let layout = bindings(resolved);
    let slot_index = |wanted: fn(&Binding) -> bool| layout.iter().position(wanted).unwrap_or(0);
    let route_buffer = layout
        .iter()
        .enumerate()
        .filter(|(_, binding)| matches!(binding, Binding::Indices(_)))
        .nth(expert.slot)
        .map_or(0, |(index, _)| index);
    let uniforms_buffer = slot_index(|binding| matches!(binding, Binding::Uniforms));
    let fault_buffer = slot_index(|binding| matches!(binding, Binding::Fault));
    let compaction_buffer = layout.len();
    let experts = grouped_expert_count(resolved, block);
    let tokens = grouped_token_total(resolved, block);
    let slot = expert.slot;
    let innermost_axis = geometry.token_axes.last().copied().unwrap_or(0);
    let weight = block.weight;
    let signature = |name: &str, extra: &str| {
        format!(
            "kernel void {name}(\n    device const float* gather_idx{slot} [[buffer({route_buffer})]],\n    constant Uniforms& u [[buffer({uniforms_buffer})]],\n    device atomic_uint* fault [[buffer({fault_buffer})]],\n    device uint* route_compaction [[buffer({compaction_buffer})]],\n    uint tiisg [[thread_index_in_simdgroup]],\n    uint sgitg [[simdgroup_index_in_threadgroup]],\n    uint sgcount [[simdgroups_per_threadgroup]],\n    uint tgid [[threadgroup_position_in_grid]],{extra}\n    uint tid [[thread_position_in_threadgroup]])\n{{\n"
        )
    };
    let chunk = crate::sized::GROUPED_GEMM_PREPASS_ENTRIES;
    let constants = format!(
        "    constexpr uint experts = {experts}u;\n    constexpr uint list_base = {GROUPED_ROUTE_HEADER_WORDS}u + 2u * experts;\n    constexpr uint scratch_base = list_base + {tokens}u;\n    long chunk_begin = (long)tgid * {chunk}l;\n    long chunk_end = min({token_extent}, chunk_begin + {chunk}l);\n",
        token_extent = geometry.token_extent_expr
    );
    let fetch = |with_fault: bool| {
        let mut fetch = String::new();
        fetch.push_str("            long entry_token = scan_base + (long)tiisg;\n");
        fetch.push_str("            bool live = entry_token < chunk_end;\n");
        fetch.push_str(&format!(
            "            long fetched{weight} = live ? (long)gather_idx{slot}[u.gather_index_base[{slot}] + entry_token * u.gather_index_strides[{slot}][{innermost_axis}]] : (long)-1;\n"
        ));
        if with_fault {
            fetch.push_str("            if (scan_expert == 0u && live) {\n");
            push_gather_fault_check(&mut fetch, weight, slot, "                ");
            fetch.push_str("            }\n");
        }
        fetch.push_str(&format!(
            "            fetched{weight} = live ? max((long)0, min(fetched{weight}, u.gather_extent[{slot}] - 1)) : (long)-1;\n"
        ));
        fetch.push_str(&format!(
            "            uint hit = (fetched{weight} == (long)scan_expert) ? 1u : 0u;\n"
        ));
        fetch
    };
    let prepass = grouped_prepass_entry(entry);
    let place = grouped_prepass_place_entry(entry);
    source.push('\n');
    source.push_str(&signature(&prepass, ""));
    source.push_str(&constants);
    push_prepass_count_body(source, &fetch(true));
    source.push_str("}\n\n");
    source.push_str(&signature(
        &place,
        "\n    uint tgcount [[threadgroups_per_grid]],",
    ));
    source.push_str(&constants);
    push_prepass_place_body(source, &fetch(false));
    source.push_str("}\n");
    Ok(())
}

/// The count pass: this threadgroup's per-expert tallies over its chunk of the
/// route, one row of the partial-count scratch.
#[cfg(feature = "metal-grouped-gemm")]
fn push_prepass_count_body(source: &mut String, fetch: &str) {
    source.push_str(
        "    for (uint scan_expert = sgitg; scan_expert < experts; scan_expert += sgcount) {\n",
    );
    source.push_str("        uint tally = 0u;\n");
    source.push_str("        for (long scan_base = chunk_begin; scan_base < chunk_end; scan_base += 32l) {\n");
    source.push_str(fetch);
    source.push_str("            tally += hit;\n");
    source.push_str("        }\n");
    source.push_str("        tally = simd_sum(tally);\n");
    source.push_str("        if (tiisg == 0u) {\n");
    source.push_str("            route_compaction[scratch_base + tgid * experts + scan_expert] = tally;\n");
    source.push_str("        }\n");
    source.push_str("    }\n");
}

/// The place pass: reduce the scratch to this threadgroup's per-expert bases,
/// then place the chunk's token ids in route order.
#[cfg(feature = "metal-grouped-gemm")]
fn push_prepass_place_body(source: &mut String, fetch: &str) {
    source.push_str("    threadgroup uint expert_count_tg[experts];\n");
    source.push_str("    threadgroup uint expert_before_tg[experts];\n");
    source.push_str("    threadgroup uint expert_base_tg[experts];\n");
    push_prepass_scratch_sums(source);
    push_prepass_offsets(source);
    source.push_str(
        "    for (uint scan_expert = sgitg; scan_expert < experts; scan_expert += sgcount) {\n",
    );
    source.push_str("        uint placed = 0u;\n");
    source.push_str("        for (long scan_base = chunk_begin; scan_base < chunk_end; scan_base += 32l) {\n");
    source.push_str(fetch);
    source.push_str("            uint rank = simd_prefix_exclusive_sum(hit);\n");
    source.push_str(
        "            if (hit != 0u) { route_compaction[list_base + expert_base_tg[scan_expert] + placed + rank] = (uint)entry_token; }\n",
    );
    source.push_str("            placed += simd_sum(hit);\n");
    source.push_str("        }\n");
    source.push_str("    }\n");
}

/// Per expert, across the scratch rows: the count over every threadgroup and
/// the count over the threadgroups before this one.
#[cfg(feature = "metal-grouped-gemm")]
fn push_prepass_scratch_sums(source: &mut String) {
    source.push_str(
        "    for (uint scan_expert = sgitg; scan_expert < experts; scan_expert += sgcount) {\n",
    );
    source.push_str("        uint total = 0u;\n");
    source.push_str("        uint before = 0u;\n");
    source.push_str("        for (uint row = tiisg; row < tgcount; row += 32u) {\n");
    source.push_str("            uint partial = route_compaction[scratch_base + row * experts + scan_expert];\n");
    source.push_str("            total += partial;\n");
    source.push_str("            before += (row < tgid) ? partial : 0u;\n");
    source.push_str("        }\n");
    source.push_str("        total = simd_sum(total);\n");
    source.push_str("        before = simd_sum(before);\n");
    source.push_str("        if (tiisg == 0u) {\n");
    source.push_str("            expert_count_tg[scan_expert] = total;\n");
    source.push_str("            expert_before_tg[scan_expert] = before;\n");
    source.push_str("        }\n");
    source.push_str("    }\n");
}

/// Between the sums and the placing: thread zero turns the counts into
/// exclusive offsets, keeps this threadgroup's base per expert (offset plus the
/// earlier chunks' count) and, in threadgroup zero, publishes the counts, the
/// offsets and the total; the barriers order that against the sums before it
/// and the placing after it.
#[cfg(feature = "metal-grouped-gemm")]
fn push_prepass_offsets(source: &mut String) {
    source.push_str("    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    source.push_str("    if (tid == 0u) {\n");
    source.push_str("        uint running = 0u;\n");
    source.push_str("        for (uint expert = 0u; expert < experts; ++expert) {\n");
    source.push_str("            expert_base_tg[expert] = running + expert_before_tg[expert];\n");
    source.push_str("            if (tgid == 0u) {\n");
    source.push_str(&format!(
        "                route_compaction[{GROUPED_ROUTE_HEADER_WORDS}u + expert] = expert_count_tg[expert];\n"
    ));
    source.push_str(&format!(
        "                route_compaction[{GROUPED_ROUTE_HEADER_WORDS}u + experts + expert] = running;\n"
    ));
    source.push_str("            }\n");
    source.push_str("            running += expert_count_tg[expert];\n");
    source.push_str("        }\n");
    source.push_str("        if (tgid == 0u) {\n");
    source.push_str("            route_compaction[0] = running;\n");
    source.push_str("        }\n");
    source.push_str("    }\n");
    source.push_str("    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
}

/// The prepass dispatches for a compacted expert-grouped op: the count pass
/// and the place pass (the GEMM's source, one entry each), the GEMM's binding
/// layout they share, [`grouped_prepass_threadgroups`] threadgroups of
/// [`GROUPED_PREPASS_THREADS`] each, and the words of the compaction buffer the
/// host binds at `bindings.len()`. `None` for every other op and for the
/// segment-scan mode. Lowering it as sibling dispatches of the same op follows
/// [`emit_cached_attention_merge`]'s precedent: the place pass reads what the
/// count pass wrote, as the merge reads the split's scratch.
#[cfg(all(feature = "metal-grouped-gemm", any(test, all(feature = "metal", target_os = "macos"))))]
pub(crate) fn route_prepass(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
) -> Result<Option<([Kernel; 2], usize)>, EmitError> {
    if !route_prepass_active(resolved, packed_operands) {
        return Ok(None);
    }
    let kernel = emit(resolved, packed_operands, numeric_policy)?;
    let quantized = operand_codecs(resolved, packed_operands);
    let Some(block) = grouped_block(resolved, &quantized) else {
        return Ok(None);
    };
    let tokens = grouped_token_total(resolved, &block);
    let words = grouped_compaction_words(tokens, grouped_expert_count(resolved, &block));
    let grid = GridSpec {
        threads: GROUPED_PREPASS_THREADS * grouped_prepass_threadgroups(tokens),
        threadgroup_width: Some(GROUPED_PREPASS_THREADS),
        depth: 1,
        grid2d: None,
    };
    let pass = |entry: String| Kernel {
        source: kernel.source.clone(),
        entry,
        bindings: kernel.bindings.clone(),
        grid,
    };
    Ok(Some((
        [
            pass(grouped_prepass_entry(&kernel.entry)),
            pass(grouped_prepass_place_entry(&kernel.entry)),
        ],
        words as usize,
    )))
}

#[cfg(all(feature = "metal-grouped-gemm", any(test, all(feature = "metal", target_os = "macos"))))]
fn grouped_block(resolved: &BoundOp, quantized: &[Option<Codec>]) -> Option<TiledGemmBlock> {
    let BoundOpKind::Reduce {
        reduce_op,
        init,
        output_axes,
        ..
    } = &resolved.kind
    else {
        return None;
    };
    tiled_gemm_block(resolved, quantized, *reduce_op, *init, output_axes)
        .filter(|block| is_expert_grouped(block) && grouped_route_compacted(block))
}

/// Cheap form of [`route_prepass`]'s decision: classification only, no source
/// rendered.
#[cfg(all(feature = "metal-grouped-gemm", any(test, all(feature = "metal", target_os = "macos"))))]
pub(crate) fn route_prepass_active(resolved: &BoundOp, packed_operands: &PackedOperands) -> bool {
    let quantized = operand_codecs(resolved, packed_operands);
    grouped_block(resolved, &quantized).is_some()
}

#[cfg(all(not(feature = "metal-grouped-gemm"), feature = "metal", target_os = "macos"))]
pub(crate) fn route_prepass(
    _resolved: &BoundOp,
    _packed_operands: &PackedOperands,
    _numeric_policy: NumericPolicy,
) -> Result<Option<([Kernel; 2], usize)>, EmitError> {
    Ok(None)
}

#[cfg(all(not(feature = "metal-grouped-gemm"), feature = "metal", target_os = "macos"))]
pub(crate) fn route_prepass_active(_resolved: &BoundOp, _packed_operands: &PackedOperands) -> bool {
    false
}

/// What one compaction is a function of: the route operand's node and the
/// layout the prepass reads it through (base offset, the stride of the
/// innermost token axis), plus the entry count and expert count that size it.
/// Two compacted grouped ops with equal keys produce byte-identical
/// compactions, so the host runs [`route_prepass`] once for both. `None` for
/// every op that has no prepass.
#[cfg(all(feature = "metal-grouped-gemm", any(test, all(feature = "metal", target_os = "macos"))))]
pub(crate) fn route_compaction_key(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
) -> Option<(NodeId, i64, i64, u64, u64)> {
    let quantized = operand_codecs(resolved, packed_operands);
    let block = grouped_block(resolved, &quantized)?;
    let lookup = resolved.operands()[block.weight].2.as_ref()?;
    let innermost_axis = *block.token_axes.last()?;
    Some((
        lookup.indices,
        lookup.index_layout.base,
        lookup.index_layout.stride(innermost_axis),
        grouped_token_total(resolved, &block),
        grouped_expert_count(resolved, &block),
    ))
}

#[cfg(not(feature = "metal-grouped-gemm"))]
pub(crate) fn route_compaction_key(
    _resolved: &BoundOp,
    _packed_operands: &PackedOperands,
) -> Option<(NodeId, i64, i64, u64, u64)> {
    None
}

/// [`render_reduce`]'s hook: appends the prepass kernel after the GEMM's own
/// when the op is a compacted expert-grouped block, nothing otherwise.
#[cfg(feature = "metal-grouped-gemm")]
pub(super) fn push_grouped_route_prepass_if_compacted(
    source: &mut String,
    resolved: &BoundOp,
    quantized: &[Option<Codec>],
    reduce_op: ScalarOp,
    init: ReduceInit,
    output_axes: &[u16],
    entry: &str,
) -> Result<(), EmitError> {
    let Some(block) = tiled_gemm_block(resolved, quantized, reduce_op, init, output_axes)
        .filter(|block| is_expert_grouped(block) && grouped_route_compacted(block))
    else {
        return Ok(());
    };
    push_grouped_route_prepass(source, resolved, &block, output_axes, entry)
}

#[cfg(not(feature = "metal-grouped-gemm"))]
pub(super) fn push_grouped_route_prepass_if_compacted(
    _source: &mut String,
    _resolved: &BoundOp,
    _quantized: &[Option<Codec>],
    _reduce_op: ScalarOp,
    _init: ReduceInit,
    _output_axes: &[u16],
    _entry: &str,
) -> Result<(), EmitError> {
    Ok(())
}
