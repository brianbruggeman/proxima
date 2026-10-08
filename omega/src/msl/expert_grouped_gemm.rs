use super::*;

/// Token axes the expert-grouped kernel decomposes a flat token index into:
/// `[sequence]` for one top-k slot per dispatch, `[sequence, selected]` for the
/// stacked form that runs every selected expert of a projection at once.
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
/// Launch: `row_tiles * GROUPED_GEMM_COL_PARTS` threadgroups per z slice, one
/// z slice per expert the weight can name (`GridSpec::depth`, see
/// [`expert_group_depth`]). A threadgroup owns one `(expert, row tile)` and
/// walks that expert's tokens in rank order, in tiles of
/// [`GROUPED_TILE_TOKENS`]: it scans the route one chunk of 128 tokens at a
/// time, ranks the matching tokens with a simdgroup prefix sum plus the
/// simdgroup totals, appends them to `tile_token`, and runs one full tile each
/// time a tile's worth is pending (and once more for the tail). With
/// `GROUPED_GEMM_COL_PARTS` above one, the tiles of an expert are dealt out
/// round robin to that many threadgroups. Every value that steers control flow
/// (`pending_fill`, `scan_base`, `tile_ordinal`) comes from threadgroup-shared
/// totals, so the threadgroup leaves each branch and loop together and no
/// barrier is skipped by a subset.
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
    push_grouped_entry(source, node, &geometry)?;
    push_grouped_threadgroup_memory(source);
    source.push_str(&format!(
        "    device const uchar *weight_bytes = (device const uchar *)in{};\n",
        block.weight
    ));
    source.push_str(&format!(
        "    long grouped_expert_base = grouped_expert * u.gather_element_stride[{}];\n",
        expert.slot
    ));
    source.push_str("    uint pending_fill = 0u;\n");
    source.push_str("    long scan_base = 0;\n");
    source.push_str("    uint tile_ordinal = 0u;\n");
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
    source.push_str(&format!(
        "        if ((tile_ordinal % {}u) == (uint)col_part) {{\n",
        geometry.col_parts
    ));
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
    source.push_str("        tile_ordinal += 1u;\n");
    push_grouped_consume(source);
    source.push_str("    }\n");
    Ok(())
}

#[cfg(feature = "metal-grouped-gemm")]
const GROUPED_THREADS: u64 = (TILED_GEMM_NSG as u64) * SIMD_WIDTH;

#[cfg(feature = "metal-grouped-gemm")]
struct GroupedGeometry {
    col_parts: u64,
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
            col_parts: crate::sized::GROUPED_GEMM_COL_PARTS,
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
/// index within one z slice, split into `(row_tile, col_part)`.
#[cfg(feature = "metal-grouped-gemm")]
fn push_grouped_entry(
    source: &mut String,
    node: NodeId,
    geometry: &GroupedGeometry,
) -> Result<(), EmitError> {
    let scalar_gid = "uint gid [[thread_position_in_grid]]";
    let vector_gid = "uint3 grouped_gid [[thread_position_in_grid]]";
    let gid_offset = source.find(scalar_gid).ok_or(EmitError::RenderKindMismatch {
        node,
        expected: "scalar thread_position_in_grid parameter",
        found: "missing",
    })?;
    source.replace_range(gid_offset..gid_offset + scalar_gid.len(), vector_gid);
    source.push_str("    long gid = (long)grouped_gid.x;\n");
    source.push_str("    long grouped_expert = (long)grouped_gid.z;\n");
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
    source.push_str(&format!(
        "    long row_tile = group_index / {};\n",
        geometry.col_parts
    ));
    source.push_str(&format!(
        "    long col_part = group_index % {};\n",
        geometry.col_parts
    ));
    source.push_str(&format!(
        "    uint grouped_lane = (uint)(tiitg % {SIMD_WIDTH});\n"
    ));
    Ok(())
}

/// `weight_tile` and `act_tile` are the two staged tiles of the K loop;
/// `out_tile` is the token-major accumulator restage after it and aliases
/// them (the K loop's trailing barrier fences the last tile read before any
/// thread reaches the aliased write). `tile_token` holds the pending token
/// ids: one tile plus the largest chunk a scan step can append.
#[cfg(feature = "metal-grouped-gemm")]
fn push_grouped_threadgroup_memory(source: &mut String) {
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
        GROUPED_TILE_TOKENS + GROUPED_THREADS
    ));
    source.push_str(&format!(
        "    threadgroup uint scan_counts[{TILED_GEMM_NSG}];\n"
    ));
}

/// Scans route chunks until a full tile of this expert's tokens is pending or
/// the route is exhausted. One thread per token: it fetches the route entry,
/// reports an out-of-range index, clamps it, and votes; the simdgroup prefix
/// sum ranks the votes inside a simdgroup and `scan_counts` carries the four
/// simdgroup totals across the barrier.
///
/// The route entries of `GROUPED_GEMM_SCAN_AHEAD` consecutive chunks are
/// loaded back to back before the first vote, so one memory round trip
/// serves them all instead of one per chunk; the chunks are then consumed in
/// order and the batch is abandoned the moment a tile fills (the unconsumed
/// entries are fetched again by the next refill).
#[cfg(feature = "metal-grouped-gemm")]
fn push_grouped_refill(
    source: &mut String,
    weight: usize,
    slot: usize,
    geometry: &GroupedGeometry,
) {
    let innermost_axis = geometry.token_axes.last().copied().unwrap_or(0);
    let ahead = geometry.scan_ahead;
    source.push_str(&format!(
        "        while (pending_fill < {GROUPED_TILE_TOKENS}u && scan_base < token_extent) {{\n"
    ));
    source.push_str(&format!("            long routed_ahead[{ahead}];\n"));
    source.push_str(&format!(
        "            for (int ahead = 0; ahead < {ahead}; ++ahead) {{\n"
    ));
    source.push_str(&format!(
        "                long ahead_token = scan_base + (long)ahead * {GROUPED_THREADS} + tiitg;\n"
    ));
    let route_offset = if geometry.route_flat {
        format!("ahead_token * u.gather_index_strides[{slot}][{innermost_axis}]")
    } else {
        push_token_coordinates(
            source,
            geometry,
            "                ",
            "(ahead_token < token_extent) ? ahead_token : 0",
            "route_c",
        );
        token_offset_expr(geometry, "route_c", |axis| {
            format!("u.gather_index_strides[{slot}][{axis}]")
        })
    };
    source.push_str(&format!(
        "                routed_ahead[ahead] = (ahead_token < token_extent) ? (long)gather_idx{slot}[u.gather_index_base[{slot}] + {route_offset}] : (long)-1;\n"
    ));
    source.push_str("            }\n");
    source.push_str(&format!(
        "            for (int ahead = 0; ahead < {ahead}; ++ahead) {{\n"
    ));
    source.push_str(&format!(
        "                if (pending_fill >= {GROUPED_TILE_TOKENS}u || scan_base >= token_extent) {{ break; }}\n"
    ));
    source.push_str("                long scan_token = scan_base + tiitg;\n");
    source.push_str(&format!(
        "                long fetched{weight} = routed_ahead[ahead];\n"
    ));
    source.push_str("                uint scan_match = 0u;\n");
    source.push_str("                if (scan_token < token_extent) {\n");
    push_gather_fault_check(source, weight, slot, "                    ");
    source.push_str(&format!(
        "                    fetched{weight} = max((long)0, min(fetched{weight}, u.gather_extent[{slot}] - 1));\n"
    ));
    source.push_str(&format!(
        "                    scan_match = (fetched{weight} == grouped_expert) ? 1u : 0u;\n"
    ));
    source.push_str("                }\n");
    source.push_str("                uint scan_prefix = simd_prefix_exclusive_sum(scan_match);\n");
    source.push_str("                uint scan_total = simd_sum(scan_match);\n");
    source.push_str("                if (grouped_lane == 0u) { scan_counts[sgitg] = scan_total; }\n");
    source.push_str("                threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    source.push_str("                uint chunk_before = 0u;\n");
    source.push_str("                uint chunk_total = 0u;\n");
    source.push_str(&format!(
        "                for (int group = 0; group < {TILED_GEMM_NSG}; ++group) {{\n"
    ));
    source.push_str("                    uint group_count = scan_counts[group];\n");
    source.push_str("                    chunk_before += (group < (int)sgitg) ? group_count : 0u;\n");
    source.push_str("                    chunk_total += group_count;\n");
    source.push_str("                }\n");
    source.push_str(
        "                if (scan_match != 0u) { tile_token[pending_fill + chunk_before + scan_prefix] = (int)scan_token; }\n",
    );
    source.push_str("                pending_fill += chunk_total;\n");
    source.push_str(&format!("                scan_base += {GROUPED_THREADS};\n"));
    source.push_str("                threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    source.push_str("            }\n");
    source.push_str("        }\n");
}

/// Drops the tokens the tile just consumed: whatever was pending beyond one
/// tile (never more than `GROUPED_THREADS - 1`, so one element per thread)
/// moves to the front. Read-barrier-write, because the source and destination
/// ranges overlap.
#[cfg(feature = "metal-grouped-gemm")]
fn push_grouped_consume(source: &mut String) {
    let block_n = GROUPED_TILE_TOKENS;
    source.push_str(&format!(
        "        uint carried_count = (pending_fill > {block_n}u) ? (pending_fill - {block_n}u) : 0u;\n"
    ));
    source.push_str("        int carried_token = 0;\n");
    source.push_str(&format!(
        "        if ((uint)tiitg < carried_count) {{ carried_token = tile_token[(uint)tiitg + {block_n}u]; }}\n"
    ));
    source.push_str("        threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    source.push_str("        if ((uint)tiitg < carried_count) { tile_token[tiitg] = carried_token; }\n");
    source.push_str("        threadgroup_barrier(mem_flags::mem_threadgroup);\n");
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
/// gate projection, uniform routing, 1000 tokens, `col_parts` 4
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
/// expert-grouped block: not the token count, because the token axis is not
/// tiled across threadgroups here -- the grid carries `GROUPED_GEMM_COL_PARTS`
/// threadgroups per row tile and each walks the tokens of its expert itself.
#[cfg(feature = "metal-grouped-gemm")]
pub(super) fn expert_grouped_launch_tokens(block: &TiledGemmBlock) -> Option<u64> {
    is_expert_grouped(block).then_some(crate::sized::GROUPED_GEMM_COL_PARTS * GROUPED_TILE_TOKENS)
}

#[cfg(not(feature = "metal-grouped-gemm"))]
pub(super) fn expert_grouped_launch_tokens(_block: &TiledGemmBlock) -> Option<u64> {
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
