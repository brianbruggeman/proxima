use super::*;

pub(super) fn render_elementwise(
    resolved: &BoundOp,
    entry: &str,
    quantized: &[Option<Codec>],
) -> Result<String, EmitError> {
    let rank = resolved.extents.len();
    let rank_len = rank.max(1);
    let operand_count = resolved.operands().len();
    let gather_count = gather_count(resolved);
    let gather_slots = gather_slots(resolved);
    let element_type = type_token(resolved.node, resolved.dtype)?;
    let coordinate_dims = elementwise_coordinate_dims(resolved, gather_count);

    let mut source = String::new();
    preamble(&mut source);

    source.push_str("struct Uniforms {\n");
    source.push_str("    long total_elements;\n");
    source.push_str(&format!("    long extents[{rank_len}];\n"));
    source.push_str(&format!("    long operand_base[{operand_count}];\n"));
    source.push_str(&format!(
        "    long operand_strides[{operand_count}][{rank_len}];\n"
    ));
    push_gather_uniform_fields(&mut source, gather_count, rank_len);
    source.push_str("};\n\n");

    kernel_signature(
        &mut source,
        quantized,
        0,
        gather_count,
        entry,
        element_type,
        false,
    );
    source.push_str("    if ((long)gid >= u.total_elements) { return; }\n");

    if !coordinate_dims.is_empty() {
        let coordinate_type = elementwise_coordinate_type(resolved);
        source.push_str(&format!("    {coordinate_type} coord[{rank_len}];\n"));
        source.push_str(&format!(
            "    {coordinate_type} remaining = ({coordinate_type})gid;\n"
        ));
        for dim in (0..resolved.extents.len()).rev() {
            if coordinate_dims.contains(&dim) {
                source.push_str(&format!(
                    "    coord[{dim}] = remaining % ({coordinate_type})u.extents[{dim}];\n"
                ));
            }
            source.push_str(&format!(
                "    remaining /= ({coordinate_type})u.extents[{dim}];\n"
            ));
        }
    }

    for (index, gather_slot) in gather_slots.iter().enumerate() {
        let dense_operand = dense_layout(&resolved.operands()[index].1, &resolved.extents);
        if dense_operand {
            source.push_str(&format!(
                "    long off{index} = u.operand_base[{index}] + (long)gid;\n"
            ));
        } else {
            source.push_str(&format!("    long off{index} = u.operand_base[{index}];\n"));
            for &dim in &coordinate_dims {
                source.push_str(&format!(
                    "    off{index} += (long)coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
                ));
            }
        }
        if let Some(slot) = gather_slot {
            push_gather_fetch(
                &mut source,
                index,
                *slot,
                rank,
                "coord",
                &format!("off{index}"),
            );
        }
    }

    source.push_str(&format!(
        "    {element_type} scratch[{}];\n",
        operand_count.max(1)
    ));
    for (index, &codec) in quantized.iter().enumerate() {
        source.push_str(&format!(
            "    scratch[{index}] = {};\n",
            operand_read(index, &format!("off{index}"), codec)
        ));
    }

    let result = push_body_steps(&mut source, resolved.element_body(), "    ", element_type);
    source.push_str(&format!("    out[gid] = {result};\n"));
    source.push_str("}\n");
    Ok(source)
}

pub(super) fn elementwise_coordinate_dims(resolved: &BoundOp, gather_count: usize) -> Vec<usize> {
    let rank = resolved.extents.len();
    let mut coordinate_dims: Vec<usize> = if gather_count > 0 {
        (0..rank).collect()
    } else {
        resolved
            .operands()
            .iter()
            .filter(|(_, layout, _)| !dense_layout(layout, &resolved.extents))
            .flat_map(|(_, layout, _)| {
                layout
                    .strides
                    .iter()
                    .enumerate()
                    .filter_map(|(dimension, stride)| (*stride != 0).then_some(dimension))
            })
            .collect()
    };
    coordinate_dims.sort_unstable();
    coordinate_dims.dedup();
    coordinate_dims
}

pub(super) fn elementwise_coordinate_type(resolved: &BoundOp) -> &'static str {
    if resolved.extents.iter().product::<u64>() <= u32::MAX as u64 {
        "uint"
    } else {
        "ulong"
    }
}

/// The exact stride-layout decisions [`render_elementwise`] bakes into MSL:
/// coordinate integer width, which axes it decodes, and which operands take
/// the dense `base + gid` path. Concrete stride values remain uniforms.
pub(super) fn elementwise_addressing_cache_token(resolved: &BoundOp) -> Option<String> {
    if !matches!(resolved.kind, BoundOpKind::Elementwise { .. }) {
        return None;
    }

    let coordinate_dims = elementwise_coordinate_dims(resolved, gather_count(resolved));
    let mut token = String::from("_ea");
    token.push(if coordinate_dims.is_empty() {
        'n'
    } else if elementwise_coordinate_type(resolved) == "uint" {
        '4'
    } else {
        '8'
    });
    token.push('c');
    for dimension in coordinate_dims {
        token.push('_');
        token.push_str(&dimension.to_string());
    }
    token.push_str("_d");
    for (_, layout, _) in resolved.operands() {
        token.push(if dense_layout(layout, &resolved.extents) {
            '1'
        } else {
            '0'
        });
    }
    Some(token)
}

pub(super) fn dense_layout(layout: &Layout, extents: &[u64]) -> bool {
    if layout.base < 0 || layout.strides.len() != extents.len() {
        return false;
    }
    let mut expected_stride = 1_i64;
    for (extent, stride) in extents.iter().zip(layout.strides.iter()).rev() {
        if *stride != expected_stride {
            return false;
        }
        expected_stride = expected_stride.saturating_mul(*extent as i64);
    }
    true
}

pub(super) fn render_reduce(
    resolved: &BoundOp,
    entry: &str,
    quantized: &[Option<Codec>],
    numeric_policy: NumericPolicy,
    expert_source_mode: bool,
) -> Result<String, EmitError> {
    let BoundOpKind::Reduce {
        reduce_op,
        init,
        output_axes,
        epilogue_body,
        epilogue_operands,
        epilogue_broadcast_axes,
        ..
    } = &resolved.kind
    else {
        return Err(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "keep::reduce fold",
            found: resolved.kind.name(),
        });
    };
    let rank = resolved.extents.len();
    let rank_len = rank.max(1);
    let operand_count = resolved.operands().len();
    let output_rank = output_axes.len();
    let output_rank_len = output_rank.max(1);
    let reduce_dims = reduction_dims(resolved, output_axes);
    let reduce_rank = reduce_dims.len();
    let reduce_rank_len = reduce_rank.max(1);
    // `push_reduce_epilogue_write`'s own doc: `epilogue_operand_strides` is
    // declared over `output_rank` (`output_axes`'s own smaller space) for the
    // pre-existing PLAIN epilogue shape. A non-empty `epilogue_broadcast_axes`
    // (`bind::BoundOpKind::Reduce::epilogue_broadcast_axes`'s own doc: the
    // RMSNorm-shaped `x * inv_rms` "broadcast-reduce" epilogue) needs the
    // WIDER full-rank space instead -- `push_cooperative_reduce_tail` is the
    // only renderer that walks it (see its own doc), so anything else
    // rejects: a mismatched axis set (this bind-time invariant is proven at
    // `bind::epilogue_broadcast_axes_for`, but a stray value here would
    // otherwise silently read/write the wrong element count), a non-
    // cooperative reduce (`push_serial_reduce_body` has no broadcast write),
    // or a tiled-GEMM/row-blocked packed match (`push_tiled_gemm_body`/
    // `push_packed_row_blocked_body` each own their write tail entirely and
    // neither has one).
    let is_broadcast_epilogue = !epilogue_broadcast_axes.is_empty();
    if is_broadcast_epilogue {
        let matches_reduce_dims = epilogue_broadcast_axes.len() == reduce_dims.len()
            && epilogue_broadcast_axes
                .iter()
                .all(|axis| reduce_dims.contains(axis));
        if !matches_reduce_dims
            || !reduce_is_cooperative_dispatch(
                resolved,
                quantized,
                numeric_policy,
                *reduce_op,
                *init,
                output_axes,
                expert_source_mode,
            )
            || tiled_gemm_block(resolved, quantized, *reduce_op, *init, output_axes).is_some()
            || packed_row_block(resolved, quantized).is_some()
        {
            return Err(EmitError::EpilogueNotSupported {
                node: resolved.node,
                reason: "the broadcast-reduce epilogue only has a Metal renderer for the plain cooperative-reduce path",
            });
        }
    }
    let gather_count = gather_count(resolved);
    let gather_slots = gather_slots(resolved);
    let element_type = type_token(resolved.node, resolved.dtype)?;
    let epilogue_operand_count = epilogue_operands.len();
    // The tiled `simdgroup_matrix` GEMM path (`push_tiled_gemm_body`) writes
    // its output through cooperative per-tile stores this module has no
    // single output-coordinate hook to splice an epilogue tail into -- every
    // other reduce renderer funnels its write through one of
    // `push_serial_reduce_body`/`push_cooperative_reduce_tail`/
    // `push_packed_row_combine_and_write`, which `push_reduce_epilogue_write`
    // now covers, so this is the one shape a fused epilogue is rejected for
    // rather than rendered, the same "no renderer, reject" contract
    // `BoundOpKind::Reduce::epilogue_body`'s own doc names.
    if !reduce_epilogue_is_identity(epilogue_body, epilogue_operands)
        && tiled_gemm_block(resolved, quantized, *reduce_op, *init, output_axes).is_some()
    {
        return Err(EmitError::EpilogueNotSupported {
            node: resolved.node,
            reason: "the tiled simdgroup_matrix GEMM kernel has no epilogue tail yet",
        });
    }

    let mut source = String::new();
    preamble(&mut source);

    source.push_str("struct Uniforms {\n");
    source.push_str("    long output_total;\n");
    source.push_str("    long reduction_total;\n");
    source.push_str(&format!("    long output_extents[{output_rank_len}];\n"));
    source.push_str(&format!("    long reduction_extents[{reduce_rank_len}];\n"));
    source.push_str(&format!("    long operand_base[{operand_count}];\n"));
    source.push_str(&format!(
        "    long operand_strides[{operand_count}][{rank_len}];\n"
    ));
    source.push_str("    long out_base;\n");
    source.push_str(&format!("    long out_strides[{rank_len}];\n"));
    // Broadcast case: `epilogue_operand_strides` widens from `output_rank_len`
    // to `rank_len` (`x`/`gamma` are read at the FULL `(s, d)` coordinate,
    // `push_cooperative_reduce_tail`'s own broadcast-write loop's doc), and
    // a fresh `broadcast_out_strides` row carries the CONTIGUOUS full-extents
    // layout the materialized output buffer is allocated with -- `out_strides`
    // above stays `out_layout`'s own compact, stride-0-on-broadcast-axes
    // addressing (still needed to locate the fold's own scalar), a genuinely
    // different address space from the widened write.
    let epilogue_stride_rank_len = if is_broadcast_epilogue {
        rank_len
    } else {
        output_rank_len
    };
    if epilogue_operand_count > 0 {
        source.push_str(&format!(
            "    long epilogue_operand_base[{epilogue_operand_count}];\n"
        ));
        source.push_str(&format!(
            "    long epilogue_operand_strides[{epilogue_operand_count}][{epilogue_stride_rank_len}];\n"
        ));
    }
    if is_broadcast_epilogue {
        source.push_str(&format!("    long broadcast_out_strides[{rank_len}];\n"));
    }
    push_gather_uniform_fields(&mut source, gather_count, rank_len);
    source.push_str("};\n\n");

    // split-K needs the actual per-dispatch threadgroup width back
    // (`kernel_signature`'s `tptg` param) ONLY on the row-blocked packed
    // path -- every other reduce kernel keeps its signature untouched, and
    // with the feature off this is always `false`, which is what makes
    // "split == 1 reproduces the current kernel exactly" hold at the source
    // level, not just numerically.
    let include_threadgroup_width =
        cfg!(feature = "metal-q4k-split-k") && packed_row_block(resolved, quantized).is_some();
    kernel_signature(
        &mut source,
        quantized,
        epilogue_operand_count,
        gather_count,
        entry,
        element_type,
        include_threadgroup_width,
    );

    if reduce_is_cooperative_dispatch(
        resolved,
        quantized,
        numeric_policy,
        *reduce_op,
        *init,
        output_axes,
        expert_source_mode,
    ) {
        push_cooperative_reduce_body(
            &mut source,
            resolved,
            *reduce_op,
            *init,
            output_axes,
            &reduce_dims,
            rank,
            quantized,
            element_type,
            epilogue_body,
            epilogue_operands,
            is_broadcast_epilogue,
            expert_source_mode,
        )?;
    } else {
        push_serial_reduce_body(
            &mut source,
            resolved,
            *reduce_op,
            *init,
            output_axes,
            &reduce_dims,
            rank,
            rank_len,
            output_rank,
            output_rank_len,
            reduce_rank,
            reduce_rank_len,
            operand_count,
            &gather_slots,
            quantized,
            element_type,
            epilogue_body,
            epilogue_operands,
        );
    }
    source.push_str("}\n");
    Ok(source)
}

/// Shared write tail for every reduce renderer -- [`push_serial_reduce_body`],
/// [`push_cooperative_reduce_tail`], and [`push_packed_row_combine_and_write`]
/// -- so a fused [`BoundOpKind::Reduce::epilogue_body`] renders identically
/// regardless of which fold produced the value being written. `coord` gives
/// the OUTPUT-axis-order coordinate expression for axis `dim` (an
/// `output_coord[dim]`-style array read, or a plain `"0"` for a rank-0
/// output where no such array exists) -- [`BoundOpKind::Reduce::
/// epilogue_operands`]'s own doc is why that space, not `full_coord`'s full
/// iteration rank, is what `epilogue_operand_strides` is declared over.
/// When the epilogue is the untouched identity default
/// ([`reduce_epilogue_is_identity`]), this emits exactly the one-line
/// `out[...] = accumulator;` every caller emitted before epilogue fusion
/// existed -- byte-for-byte, so a program with no fused epilogue anywhere
/// renders the same kernel source it always did.
#[allow(clippy::too_many_arguments)]
pub(super) fn push_reduce_epilogue_write(
    source: &mut String,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    output_rank: usize,
    element_type: &str,
    indent: &str,
    coord: impl Fn(usize) -> String,
    accumulator_expr: &str,
    out_offset_expr: &str,
) {
    if reduce_epilogue_is_identity(epilogue_body, epilogue_operands) {
        source.push_str(&format!(
            "{indent}out[{out_offset_expr}] = {accumulator_expr};\n"
        ));
        return;
    }
    let epilogue_operand_count = epilogue_operands.len();
    source.push_str(&format!(
        "{indent}{element_type} epi_scratch[{}];\n",
        epilogue_operand_count + 1
    ));
    push_epilogue_operand_reads(
        source,
        0..epilogue_operand_count,
        output_rank,
        indent,
        &coord,
    );
    source.push_str(&format!(
        "{indent}epi_scratch[{epilogue_operand_count}] = {accumulator_expr};\n"
    ));
    let epi_value = push_epilogue_body_steps(source, epilogue_body, indent, element_type);
    source.push_str(&format!("{indent}out[{out_offset_expr}] = {epi_value};\n"));
}

/// The per-element epilogue-operand read loop [`push_reduce_epilogue_write`]
/// and the hoisted broadcast write ([`push_broadcast_epilogue_write`]) both
/// need verbatim: read each epilogue operand (`gamma[d]`, `x[s,d]`, ...) at
/// the current coordinate into `epi_scratch`. Pulled out once the hoist
/// (ROW 370) needed to run it from inside a loop whose invariant steps are
/// declared outside that same loop -- keeping one copy is what stops the two
/// call sites drifting on the offset arithmetic.
pub(super) fn push_epilogue_operand_reads(
    source: &mut String,
    operand_indices: impl Iterator<Item = usize>,
    output_rank: usize,
    indent: &str,
    coord: impl Fn(usize) -> String,
) {
    for index in operand_indices {
        source.push_str(&format!(
            "{indent}long epi_off{index} = u.epilogue_operand_base[{index}];\n"
        ));
        for dim in 0..output_rank {
            source.push_str(&format!(
                "{indent}epi_off{index} += {} * u.epilogue_operand_strides[{index}][{dim}];\n",
                coord(dim)
            ));
        }
        source.push_str(&format!(
            "{indent}epi_scratch[{index}] = epi{index}[epi_off{index}];\n"
        ));
    }
}

/// True when `epilogue_operands[operand_index]`'s own [`Layout`] never
/// varies along any of `reduce_dims` -- a genuine broadcast operand
/// (`inv_dim`, `eps` in an rmsnorm chain: rank-0, every stride 0) whose
/// value is the same on every pass of [`push_broadcast_epilogue_write`]'s
/// per-lane loop, as opposed to a per-element operand (`x[s,d]`, `gamma[d]`)
/// whose stride along the loop's own axis is nonzero. A gathered operand
/// (`Lookup` present) is never treated as invariant: its effective address
/// depends on an index buffer this analysis does not follow.
pub(super) fn epilogue_operand_is_loop_invariant(
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    reduce_dims: &[u16],
    operand_index: usize,
) -> bool {
    let Some((_, layout, lookup)) = epilogue_operands.get(operand_index) else {
        return false;
    };
    lookup.is_none() && reduce_dims.iter().all(|&dim| layout.stride(dim) == 0)
}

/// [`push_body_steps`]'s counterpart for [`BoundOpKind::Reduce::
/// epilogue_body`]: identical step-emission shape over the same
/// [`scalar_op_expr`] table, reading `epi_scratch[i]`/`epi_step{k}` instead
/// of `push_body_steps`'s `scratch[i]`/`step{k}` -- the epilogue's operand
/// table is a SEPARATE array from the fold's own per-step `scratch`
/// (`push_reduce_epilogue_write`'s own doc), so the two never share a slot
/// even when a real operand index collides.
pub(super) fn push_epilogue_body_steps(
    source: &mut String,
    body: &ComposedBody,
    indent: &str,
    element_type: &str,
) -> String {
    crate::epilogue::declare_steps(
        source,
        body,
        "epi_scratch",
        "epi_step",
        scalar_op_expr,
        |source, index, expr| {
            source.push_str(&format!(
                "{indent}{element_type} epi_step{index} = {expr};\n"
            ));
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn push_serial_reduce_body(
    source: &mut String,
    resolved: &BoundOp,
    reduce_op: ScalarOp,
    init: ReduceInit,
    output_axes: &[u16],
    reduce_dims: &[u16],
    rank: usize,
    rank_len: usize,
    output_rank: usize,
    output_rank_len: usize,
    reduce_rank: usize,
    reduce_rank_len: usize,
    operand_count: usize,
    gather_slots: &[Option<usize>],
    quantized: &[Option<Codec>],
    element_type: &str,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
) {
    source.push_str("    if ((long)gid >= u.output_total) { return; }\n");

    source.push_str(&format!("    long full_coord[{rank_len}];\n"));
    for dim in 0..rank {
        source.push_str(&format!("    full_coord[{dim}] = 0;\n"));
    }

    if output_rank > 0 {
        source.push_str(&format!("    long output_coord[{output_rank_len}];\n"));
        source.push_str("    long remaining = (long)gid;\n");
        for index in (0..output_rank).rev() {
            source.push_str(&format!(
                "    output_coord[{index}] = remaining % u.output_extents[{index}]; \
                 remaining /= u.output_extents[{index}];\n"
            ));
        }
        for (index, dim) in output_axes.iter().enumerate() {
            source.push_str(&format!("    full_coord[{dim}] = output_coord[{index}];\n"));
        }
    }

    let (init_expr, seeded_init) = fold_init_tokens(init);
    source.push_str(&format!("    {element_type} accumulator = {init_expr};\n"));
    source.push_str(&format!("    bool seeded = {seeded_init};\n"));

    source.push_str("    for (long r = 0; r < u.reduction_total; r++) {\n");
    if reduce_rank > 0 {
        source.push_str(&format!(
            "        long reduction_coord[{reduce_rank_len}];\n"
        ));
        source.push_str("        long remaining_r = r;\n");
        for index in (0..reduce_rank).rev() {
            source.push_str(&format!(
                "        reduction_coord[{index}] = remaining_r % u.reduction_extents[{index}]; \
                 remaining_r /= u.reduction_extents[{index}];\n"
            ));
        }
        for (index, dim) in reduce_dims.iter().enumerate() {
            source.push_str(&format!(
                "        full_coord[{dim}] = reduction_coord[{index}];\n"
            ));
        }
    }

    for (index, gather_slot) in gather_slots.iter().enumerate() {
        source.push_str(&format!(
            "        long off{index} = u.operand_base[{index}];\n"
        ));
        for dim in 0..rank {
            source.push_str(&format!(
                "        off{index} += full_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
            ));
        }
        if let Some(slot) = gather_slot {
            push_gather_fetch(
                source,
                index,
                *slot,
                rank,
                "full_coord",
                &format!("off{index}"),
            );
        }
    }
    source.push_str(&format!(
        "        {element_type} scratch[{}];\n",
        operand_count.max(1)
    ));
    for (index, &codec) in quantized.iter().enumerate() {
        source.push_str(&format!(
            "        scratch[{index}] = {};\n",
            operand_read(index, &format!("off{index}"), codec)
        ));
    }
    let value_expr = push_body_steps(source, resolved.element_body(), "        ", element_type);
    source.push_str(&format!("        {element_type} value = {value_expr};\n"));
    let combine_expr = scalar_op_expr(reduce_op, &["accumulator", "value"]);
    source.push_str(&format!(
        "        accumulator = seeded ? {combine_expr} : value;\n"
    ));
    source.push_str("        seeded = true;\n");
    source.push_str("    }\n");

    source.push_str("    long out_offset = u.out_base;\n");
    for dim in 0..rank {
        source.push_str(&format!(
            "    out_offset += full_coord[{dim}] * u.out_strides[{dim}];\n"
        ));
    }
    push_reduce_epilogue_write(
        source,
        epilogue_body,
        epilogue_operands,
        output_rank,
        element_type,
        "    ",
        |dim| {
            if output_rank > 0 {
                format!("output_coord[{dim}]")
            } else {
                "0".to_string()
            }
        },
        "accumulator",
        "out_offset",
    );
}

/// The SIMD-group cooperative fold: `SIMD_WIDTH` lanes split one output
/// element's contraction axis, each striding through `reduction_total` by
/// `SIMD_WIDTH` so every element is visited by exactly one lane, then
/// combine via [`simd_combine_fn`]. Gathered expert products with a route
/// invariant across the reduction use the same fold; their route index is
/// fetched once per simdgroup and broadcast before the strided walk. Only lane 0 writes the result, and only
/// lane 0 seeds from the `BoundOp`'s real `ReduceInit` — every other lane
/// seeds from [`cooperative_identity_token`] so the true seed is folded into
/// the group exactly once (see that function's doc). `gid / SIMD_WIDTH` is a
/// valid output index, and `gid % SIMD_WIDTH` a valid lane-within-group
/// index, because [`GridSpec::threadgroup_width`] always pins the dispatched
/// threadgroup width to a whole multiple of `SIMD_WIDTH` — see
/// `crate::metal::dispatch` — and Metal's `dispatchThreads:
/// threadsPerThreadgroup:` (the API `dispatch` always calls) defines
/// `[[thread_position_in_grid]]` as `threadgroup_position_in_grid *
/// threadgroup_width + thread_position_in_threadgroup` even in the boundary
/// (non-full) threadgroup, so `gid` stays a flat global index unaffected by
/// how many `SIMD_WIDTH`-lane simdgroups the driver packs into one
/// threadgroup. The row-blocked packed path widens this multiple via
/// [`crate::sized::PACKED_ROW_BLOCK_SIMDGROUPS`]; every other cooperative
/// reduce stays at exactly one simdgroup per threadgroup.
/// Gather is out of scope here: [`reduce_is_cooperative`] never selects this
/// path when the op gathers, so operand offsets are read straight off
/// `operand_base`/`operand_strides` with no fetch/fault machinery.
/// See [`PackedRowBlock`]. Emits the whole body for the row-blocked packed
/// path; the caller has already emitted `output_index` (a GROUP index here)
/// and `lane`.
///
/// Whether the Q4_K arm below may defer a sub-block's scale/min to ONCE per
/// sub-block instead of once per element. The identity that makes this legal,
/// `sum_j (scale*nibble_j - min)*act_j == scale*sum(nibble_j*act_j) -
/// min*sum(act_j)`, holds only when the reduction is a plain sum of products:
/// `reduce_op == Add` (`Multiply`/`Maximum`/`Minimum` are also legal under
/// [`is_cooperative_reduce_op`] and all break the identity — a `Maximum`
/// reduce cannot be pulled outside a per-element scale at all) AND the fused
/// element body is EXACTLY `scratch[weight] * scratch[other]`, no other
/// steps (a fused body inserts arbitrary extra `ScalarOp`s between the raw
/// product and the reduce, any of which the identity does not survive).
/// Mirrors `ggml-metal.metal:5157-5175`'s `acc1`/`dall` shape
/// (`docs/discipline.md` ROW 106); the other two codecs are untouched — see
/// this function's own Q5_K/Q6_K arms for why.
pub(super) fn is_plain_product_reduce(
    resolved: &BoundOp,
    reduce_op: ScalarOp,
    weight: usize,
    other: usize,
) -> bool {
    if reduce_op != ScalarOp::Add {
        return false;
    }
    let [step] = resolved.element_body().steps.as_slice() else {
        return false;
    };
    if step.op != ScalarOp::Multiply {
        return false;
    }
    let weight = weight as u16;
    let other = other as u16;
    matches!(
        step.args.as_slice(),
        [StepArg::Operand(first), StepArg::Operand(second)]
            if (*first == weight && *second == other) || (*first == other && *second == weight)
    )
}

/// The row-blocked Q4_K header decode, one call site feature-gated between
/// `q4k_header_for` (the shift-then-branch original) and `q4k_header_for_bf`
/// (`metal-q4k-mask-fma`'s branch-free port, see [`Q4K_MASK_FMA_MSL`]).
/// Split out of [`push_packed_row_blocked_body`]'s `Codec::Q4K` arm so
/// the two `#[cfg]` bodies stay next to each other rather than interleaved
/// with the surrounding match.
#[cfg(not(feature = "metal-q4k-mask-fma"))]
pub(super) fn push_q4k_header_decode(source: &mut String) {
    source.push_str("            q4k_header hdr = q4k_header_for(blk, slot);\n");
}

#[cfg(feature = "metal-q4k-mask-fma")]
pub(super) fn push_q4k_header_decode(source: &mut String) {
    source.push_str("            q4k_header hdr = q4k_header_for_bf(blk, slot);\n");
}

/// The Q4_K SCALE-DEFERRED matvec body (`docs/discipline.md` ROW 106):
/// accumulate the raw nibble x activation product and the activation sum
/// UNSCALED across the whole sub-block, then apply `hdr.scale`/`hdr.minimum`
/// ONCE at the end instead of once per element — legal because this
/// function is only reached when `is_plain_product_reduce` has already
/// proved `reduce_op` is `Add` and the body is exactly `weight * other`, so
/// `sum_j (scale*nibble_j - min)*act_j == scale*sum(nibble_j*act_j) -
/// min*sum(act_j)`.
///
/// Two bodies behind `metal-q4k-mask-fma`, same shape as
/// [`push_q4k_header_decode`]:
///
/// Without the feature: `q4k_run8`'s shift-then-mask extraction into a
/// `levels[8]` scratch array, then a `dot`-reduce into `raw_acc`/`act_sum` —
/// mirrors `ggml-metal.metal:5157-5175`'s `acc1`/`dall` split at the ALGEBRA
/// level (defer the scale) but not at the EXTRACTION level (ggml never
/// shifts; see `q4k_run8`'s own corrected doc).
///
/// With the feature: extraction and accumulate are ONE fused loop, masked
/// without any shift, ported from ggml's ACTUAL technique
/// (`ggml-metal.metal:5157-5165`) onto this file's one-nibble-per-byte
/// layout (every element in a lane's 32-element sub-block occupies its own
/// byte, unlike ggml's two-nibbles-per-byte interleave across two
/// sub-blocks — see `q4k_run8`'s doc for why that is a different packing,
/// not a narrower ggml). A `ushort` load of one byte pair yields both
/// nibbles this lane wants at two residual scales — 1x/256x for the low
/// nibble half, 16x/4096x for the high half — inlined directly against
/// `element_type` (not through a shared MSL function typed to `float`) so
/// this stays exactly as generic over `half`/`float` as `q4k_run8`'s own
/// callers are. The residual scale is IDENTICAL for all four `c` iterations
/// a lane makes (`within < 32u` cannot change within one lane's 32-element
/// run, `q4k_run8`'s own doc establishes why), so `q4k_corr` is computed
/// ONCE, outside the loop, and folded into `hdr.scale` at the same combine
/// point the deferred-scale algebra above already uses.
#[cfg(not(feature = "metal-q4k-mask-fma"))]
pub(super) fn push_q4k_product_reduce_body(source: &mut String, sub: usize, run: usize, element_type: &str) {
    source.push_str(&format!("            {element_type} raw_acc = 0;\n"));
    source.push_str(&format!("            {element_type} act_sum = 0;\n"));
    source.push_str(&format!(
        "            for (int c = 0; c < {}; ++c) {{\n",
        sub / run
    ));
    // raw 4-bit levels (0..15) are exact in float regardless of the
    // kernel's element type; q4k_run8 takes `thread float *out`, narrowed
    // to element_type at the multiply below, same as the per-element path.
    source.push_str(&format!("                float levels[{run}];\n"));
    source.push_str(&format!(
        "                q4k_run8(blk, slot + (uint)(c * {run}), levels);\n"
    ));
    source.push_str("                raw_acc += dot(float4(levels[0], levels[1], levels[2], levels[3]), float4(acts[c * 8 + 0], acts[c * 8 + 1], acts[c * 8 + 2], acts[c * 8 + 3]));\n");
    source.push_str("                raw_acc += dot(float4(levels[4], levels[5], levels[6], levels[7]), float4(acts[c * 8 + 4], acts[c * 8 + 5], acts[c * 8 + 6], acts[c * 8 + 7]));\n");
    source.push_str("                act_sum += acts[c * 8 + 0] + acts[c * 8 + 1] + acts[c * 8 + 2] + acts[c * 8 + 3] + acts[c * 8 + 4] + acts[c * 8 + 5] + acts[c * 8 + 6] + acts[c * 8 + 7];\n");
    source.push_str("            }\n");
    source
        .push_str("            sumf[q] = sumf[q] + hdr.scale * raw_acc - hdr.minimum * act_sum;\n");
}

#[cfg(feature = "metal-q4k-mask-fma")]
pub(super) fn push_q4k_product_reduce_body(source: &mut String, sub: usize, run: usize, element_type: &str) {
    source.push_str(&format!("            {element_type} raw_acc = 0;\n"));
    source.push_str(&format!("            {element_type} act_sum = 0;\n"));
    source.push_str("            bool q4k_hi = (slot % 64u) >= 32u;\n");
    source.push_str("            ushort q4k_mask_a = q4k_hi ? 0x00F0u : 0x000Fu;\n");
    source.push_str("            ushort q4k_mask_b = q4k_hi ? 0xF000u : 0x0F00u;\n");
    source.push_str("            float q4k_corr = q4k_hi ? (1.0f / 16.0f) : 1.0f;\n");
    source.push_str(&format!(
        "            for (int c = 0; c < {}; ++c) {{\n",
        sub / run
    ));
    source.push_str(&format!(
        "                uint q4k_index = slot + (uint)(c * {run});\n"
    ));
    source.push_str("                uint q4k_group = q4k_index / 64u;\n");
    source.push_str("                uint q4k_within = q4k_index % 64u;\n");
    source.push_str("                uint q4k_byte = q4k_group * 32u + (q4k_within % 32u);\n");
    source.push_str(
        "                device const ushort *q4k_pairs = (device const ushort *)(blk + 16 + q4k_byte);\n",
    );
    source.push_str(&format!(
        "                for (int p = 0; p < {}; ++p) {{\n",
        run / 2
    ));
    source.push_str("                    ushort q4k_word = q4k_pairs[p];\n");
    source.push_str("                    float q4k_level_a = (float)(q4k_word & q4k_mask_a);\n");
    source.push_str(
        "                    float q4k_level_b = (float)(q4k_word & q4k_mask_b) * (1.0f / 256.0f);\n",
    );
    source.push_str(&format!(
        "                    {element_type} act_a = acts[c * {run} + 2 * p];\n"
    ));
    source.push_str(&format!(
        "                    {element_type} act_b = acts[c * {run} + 2 * p + 1];\n"
    ));
    source.push_str(&format!(
        "                    raw_acc += ({element_type})(q4k_level_a * (float)act_a + q4k_level_b * (float)act_b);\n"
    ));
    source.push_str("                    act_sum += act_a + act_b;\n");
    source.push_str("                }\n");
    source.push_str("            }\n");
    source.push_str(
        "            sumf[q] = sumf[q] + hdr.scale * q4k_corr * raw_acc - hdr.minimum * act_sum;\n",
    );
}

/// The row-blocked packed path's tail: combine each simdgroup's per-row
/// `sumf[q]` and write the output. `metal-q4k-split-k`-off arm -- exactly
/// [`push_packed_row_blocked_body`]'s original tail, one simdgroup per
/// row-group, lane 0 writes straight from the SIMD combine.
///
/// Reads `coord_q_cache[q]`, NOT a fresh `flat % / u.output_extents` decode:
/// the preamble above (`push_packed_row_blocked_body`'s own `weight_base`/
/// `other_base` loop) already derives that same coordinate, per `q`, to
/// address the weight/activation operands -- `flat = group_first + q` is
/// identical in both places, so re-running the same division/modulo chain a
/// second time here just to re-derive `coord_q` was paying the ladder-vs-
/// production gap's own named cost twice per thread for zero new
/// information (`docs/bench-campaigns/2026-09-03-gpu-one-risc/design-2026-09-04/kernel-body-diff.md`).
#[cfg(not(feature = "metal-q4k-split-k"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn push_packed_row_combine_and_write(
    source: &mut String,
    node: NodeId,
    reduce_op: ScalarOp,
    rows: usize,
    rank: usize,
    output_axes: &[u16],
    element_type: &str,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
) -> Result<(), EmitError> {
    let combine_fn = simd_combine_fn(node, reduce_op)?;
    source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!(
        "        {element_type} reduced = {combine_fn}(sumf[q]);\n"
    ));
    source.push_str("        long flat = group_first + q;\n");
    source.push_str("        if (lane == 0u && flat < u.output_total) {\n");
    source.push_str("            long out_offset = u.out_base;\n");
    for dim in 0..rank {
        source.push_str(&format!(
            "            out_offset += coord_q_cache[q][{dim}] * u.out_strides[{dim}];\n"
        ));
    }
    push_reduce_epilogue_write(
        source,
        epilogue_body,
        epilogue_operands,
        output_axes.len(),
        element_type,
        "            ",
        |dim| format!("coord_q_cache[q][{}]", output_axes[dim]),
        "reduced",
        "out_offset",
    );
    source.push_str("        }\n");
    source.push_str("    }\n");
    Ok(())
}

/// The row-blocked packed path's tail: combine each simdgroup's per-row
/// `sumf[q]` and write the output. `metal-q4k-split-k`-on arm -- SPLIT-K
/// COMBINE, option 1 from this landing's brief (multiple simdgroups in ONE
/// threadgroup, threadgroup memory + a barrier, one final fold), over the
/// second-choice atomic-accumulate (the output dtype can be `half`, which
/// Metal has no `atomic<half>` for) and the third-choice second-dispatch
/// partials pass (would double the kernel-launch and uniform-upload cost
/// this landing exists to avoid paying on the STARVED shapes specifically).
///
/// Each simdgroup already SIMD-folds its own interleaved slice of
/// super-blocks (`push_packed_row_blocked_body`'s `ib` loop, strided by
/// `4 * split` when split-K is active); this only combines the (at most
/// [`crate::sized::PACKED_ROW_SPLIT_K_MAX_SPLIT`]) per-simdgroup partials
/// left behind. At `split == 1` (`sgitg` always `0`) this degenerates to
/// exactly the feature-off tail: the loop over `s` never runs, so
/// `total == partial_sums[q][0] == reduced`, written by the SAME thread that
/// computed it -- same value, same order, only the intermediate trip through
/// `threadgroup` memory differs.
#[cfg(feature = "metal-q4k-split-k")]
#[allow(clippy::too_many_arguments)]
pub(super) fn push_packed_row_combine_and_write(
    source: &mut String,
    node: NodeId,
    reduce_op: ScalarOp,
    rows: usize,
    rank: usize,
    output_axes: &[u16],
    element_type: &str,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
) -> Result<(), EmitError> {
    let combine_fn = simd_combine_fn(node, reduce_op)?;
    let max_split = crate::sized::PACKED_ROW_SPLIT_K_MAX_SPLIT;
    source.push_str(&format!(
        "    threadgroup {element_type} partial_sums[{rows}][{max_split}];\n"
    ));
    source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!(
        "        {element_type} reduced = {combine_fn}(sumf[q]);\n"
    ));
    source.push_str("        if (lane == 0u) { partial_sums[q][sgitg] = reduced; }\n");
    source.push_str("    }\n");
    source.push_str("    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    source.push_str("    if (sgitg == 0u && lane == 0u) {\n");
    source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!(
        "            {element_type} total = partial_sums[q][0];\n"
    ));
    source.push_str("            for (uint s = 1u; s < split; ++s) {\n");
    let combine_expr = scalar_op_expr(reduce_op, &["total", "partial_sums[q][s]"]);
    source.push_str(&format!("                total = {combine_expr};\n"));
    source.push_str("            }\n");
    source.push_str("            long flat = group_first + q;\n");
    source.push_str("            if (flat < u.output_total) {\n");
    source.push_str("                long out_offset = u.out_base;\n");
    for dim in 0..rank {
        source.push_str(&format!(
            "                out_offset += coord_q_cache[q][{dim}] * u.out_strides[{dim}];\n"
        ));
    }
    push_reduce_epilogue_write(
        source,
        epilogue_body,
        epilogue_operands,
        output_axes.len(),
        element_type,
        "                ",
        |dim| format!("coord_q_cache[q][{}]", output_axes[dim]),
        "total",
        "out_offset",
    );
    source.push_str("            }\n");
    source.push_str("        }\n");
    source.push_str("    }\n");
    Ok(())
}

/// [`push_packed_row_blocked_body`]'s `token_total > 1` branch: `s <=
/// crate::sized::PACKED_ROW_ACTIVATION_GROUP` activation ("token") rows
/// folded against ONE streamed weight row per accumulator group --
/// [`grid_threads`]'s own tiling of `token_total` above the cap handles the
/// rest by dispatching more groups, never by truncating `s`. Reuses the
/// fully generic [`operand_read`]/[`push_body_steps`] machinery the plain
/// cooperative/serial reduce paths already use, rather than the
/// `token_total <= 1` branch's hand-tuned per-codec lane-spread -- this
/// path's gate is the s-fold itself (a weight element read from device
/// memory ONCE per `(feature row, reduce-dim element)`, copied into
/// `scratch[weight]` and reused `s` times, never re-read), so every codec
/// and reduce body `packed_row_block` admits is covered by construction.
/// The amortized per-codec header/nibble decode the `token_total <= 1`
/// branch hand-tunes (`q4k_run8`, paired-lane loads) is a follow-up, not
/// folded in here yet -- this body pays one `operand_read` per weight
/// element, same as the generic serial path, just reused across `s` instead
/// of the reduce dim alone.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub(super) fn push_packed_row_multi_row_body(
    source: &mut String,
    resolved: &BoundOp,
    reduce_op: ScalarOp,
    init: ReduceInit,
    rank: usize,
    quantized: &[Option<Codec>],
    element_type: &str,
    block: &PackedRowBlock,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    expert_source_mode: bool,
) -> Result<(), EmitError> {
    let weight = block.weight;
    let other = block.other;
    let reduce_dim = block.reduce_dim;
    let token_axes = &block.token_axes;
    let feature_axes = &block.feature_axes;
    let rank_len = rank.max(1);
    let operand_count = resolved.operands().len();
    let rows = codec_rows_per_simdgroup(block.codec);
    let cap = crate::sized::PACKED_ROW_ACTIVATION_GROUP as usize;
    let (init_expr, _) = fold_init_tokens(init);
    let identity = cooperative_identity_token(resolved.node, reduce_op)?;
    let combine_fn = simd_combine_fn(resolved.node, reduce_op)?;

    source.push_str("    long feature_total = 1;\n");
    for index in 0..feature_axes.len() {
        source.push_str(&format!(
            "    feature_total *= u.output_extents[{}];\n",
            token_axes.len() + index
        ));
    }
    source.push_str("    long token_total = 1;\n");
    for index in 0..token_axes.len() {
        source.push_str(&format!("    token_total *= u.output_extents[{index}];\n"));
    }
    source.push_str(&format!(
        "    long feature_base = (feature_total + {rows} - 1) / {rows};\n"
    ));
    source.push_str("    long token_group = output_index / feature_base;\n");
    source.push_str("    long feature_group = output_index % feature_base;\n");
    source.push_str(&format!(
        "    long feature_first = feature_group * {rows};\n"
    ));
    source.push_str(&format!("    long token_first = token_group * {cap};\n"));

    source.push_str(&format!("    {element_type} sumf[{cap}][{rows}];\n"));
    source.push_str(&format!("    for (int s = 0; s < {cap}; ++s) {{\n"));
    source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!(
        "            sumf[s][q] = (lane == 0u) ? ({init_expr}) : ({identity});\n"
    ));
    source.push_str("        }\n    }\n");

    source.push_str(&format!("    long weight_base[{rows}];\n"));
    source.push_str(&format!("    long feature_coord[{rows}][{rank_len}];\n"));
    source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str("        long flat = feature_first + q;\n");
    source.push_str(
        "        long remaining = (flat < feature_total) ? flat : (feature_total - 1);\n",
    );
    for dim in 0..rank {
        source.push_str(&format!("        feature_coord[q][{dim}] = 0;\n"));
    }
    for (index_from_end, &dim) in feature_axes.iter().enumerate().rev() {
        let full_index = token_axes.len() + index_from_end;
        source.push_str(&format!(
            "        feature_coord[q][{dim}] = remaining % u.output_extents[{full_index}]; remaining /= u.output_extents[{full_index}];\n"
        ));
    }
    source.push_str(&format!("        long wb = u.operand_base[{weight}];\n"));
    for &dim in feature_axes {
        source.push_str(&format!(
            "        wb += feature_coord[q][{dim}] * u.operand_strides[{weight}][{dim}];\n"
        ));
    }
    source.push_str("        weight_base[q] = wb;\n");
    source.push_str("    }\n");

    source.push_str(&format!("    long other_base[{cap}];\n"));
    source.push_str(&format!("    long token_coord[{cap}][{rank_len}];\n"));
    source.push_str(&format!("    for (int s = 0; s < {cap}; ++s) {{\n"));
    source.push_str("        long flat = token_first + s;\n");
    source.push_str("        long remaining = (flat < token_total) ? flat : (token_total - 1);\n");
    for dim in 0..rank {
        source.push_str(&format!("        token_coord[s][{dim}] = 0;\n"));
    }
    for (index, &dim) in token_axes.iter().enumerate().rev() {
        source.push_str(&format!(
            "        token_coord[s][{dim}] = remaining % u.output_extents[{index}]; remaining /= u.output_extents[{index}];\n"
        ));
    }
    source.push_str(&format!("        long ob = u.operand_base[{other}];\n"));
    for &dim in token_axes {
        source.push_str(&format!(
            "        ob += token_coord[s][{dim}] * u.operand_strides[{other}][{dim}];\n"
        ));
    }
    source.push_str("        other_base[s] = ob;\n");
    source.push_str("    }\n");

    // A routed expert's weight base depends on WHICH token slot `s` picked
    // it, not on the feature row `q` -- gathered once per token here,
    // alongside the token coordinate `other_base[s]` already decoded above,
    // never re-fetched per `(s, q, k)` triple below.
    let gather_slots = gather_slots(resolved);
    let weight_gathered = gather_slots[weight].is_some();
    if weight_gathered {
        source.push_str(&format!("    long weight_expert_base[{cap}];\n"));
        source.push_str(&format!("    for (int s = 0; s < {cap}; ++s) {{\n"));
        source.push_str("        long web = 0;\n");
        if let Some(slot) = gather_slots[weight] {
            push_cooperative_gather_fetch(source, weight, slot, rank, "token_coord[s]", "web");
        }
        source.push_str("        weight_expert_base[s] = web;\n");
        source.push_str("    }\n");
    }

    source.push_str(&format!(
        "    long other_stride = u.operand_strides[{other}][{reduce_dim}];\n"
    ));
    // Q4_K FAST PATH (`docs/discipline.md` ROW 389): the weight block's
    // header/nibble decode is shared across the whole `cap`-token activation
    // group via `q4k_pair_dot_mr` (this file's own multi-row generalization
    // of the M=1 decode path's `q4k_pair_dot`), instead of the per-element
    // `operand_read` below paying that decode once per token. Every other
    // codec (`Q3_K`/`Q5_K`/`Q6_K`) keeps the generic loop -- they have no
    // multi-row port yet, this landing only proves the pattern on `Q4_K`.
    // A gathered weight is excluded here -- `q4k_pair_dot_mr` hoists one
    // decode shared across every token slot, which is only sound when every
    // slot reads the SAME expert row; the generic loop below re-reads per
    // slot instead, which a routed weight requires regardless of codec.
    let fast_q4k = !expert_source_mode
        && !weight_gathered
        && block.codec == Codec::Q4K
        && element_type == "float"
        && quantized[weight] == Some(Codec::Q4K)
        && quantized[other].is_none()
        && is_plain_product_reduce(resolved, reduce_op, weight, other);
    if fast_q4k {
        push_packed_row_multi_row_q4k_body(
            source,
            weight,
            other,
            rows,
            cap,
            codec_block_bytes(block.codec),
        );
    } else {
        source.push_str("    for (long k = (long)lane; k < u.reduction_total; k += 32L) {\n");
        source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
        source.push_str(&format!(
            "            {element_type} scratch[{}];\n",
            operand_count.max(1)
        ));
        if !weight_gathered {
            // shared across every token slot -- one read per (feature row,
            // reduce element), reused `cap` times below.
            source.push_str(&format!(
                "            scratch[{weight}] = {};\n",
                operand_read(weight, "(weight_base[q] + k)", quantized[weight])
            ));
        }
        source.push_str(&format!("            for (int s = 0; s < {cap}; ++s) {{\n"));
        if weight_gathered {
            // each token slot may have routed to a different expert, so the
            // weight read moves inside the slot loop instead of being
            // hoisted above it -- still exactly one read per (slot, feature
            // row, reduce element), never per output element.
            source.push_str(&format!(
                "                scratch[{weight}] = {};\n",
                operand_read(
                    weight,
                    "(weight_base[q] + weight_expert_base[s] + k)",
                    quantized[weight]
                )
            ));
        }
        source.push_str(&format!(
            "                scratch[{other}] = {};\n",
            operand_read(
                other,
                "(other_base[s] + k * other_stride)",
                quantized[other]
            )
        ));
        let value_expr = push_body_steps(
            source,
            resolved.element_body(),
            "                ",
            element_type,
        );
        source.push_str(&format!(
            "                {element_type} value = {value_expr};\n"
        ));
        let combine_expr = scalar_op_expr(reduce_op, &["sumf[s][q]", "value"]);
        source.push_str(&format!("                sumf[s][q] = {combine_expr};\n"));
        source.push_str("            }\n");
        source.push_str("        }\n");
        source.push_str("    }\n");
    }

    source.push_str(&format!("    for (int s = 0; s < {cap}; ++s) {{\n"));
    source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!(
        "            {element_type} reduced = {combine_fn}(sumf[s][q]);\n"
    ));
    source.push_str("            if (lane == 0u) {\n");
    source.push_str("                long token_flat = token_first + s;\n");
    source.push_str("                long feature_flat = feature_first + q;\n");
    source.push_str(
        "                if (token_flat < token_total && feature_flat < feature_total) {\n",
    );
    source.push_str("                    long out_offset = u.out_base;\n");
    for &dim in feature_axes {
        source.push_str(&format!(
            "                    out_offset += feature_coord[q][{dim}] * u.out_strides[{dim}];\n"
        ));
    }
    for &dim in token_axes {
        source.push_str(&format!(
            "                    out_offset += token_coord[s][{dim}] * u.out_strides[{dim}];\n"
        ));
    }
    let output_rank = token_axes.len() + feature_axes.len();
    push_reduce_epilogue_write(
        source,
        epilogue_body,
        epilogue_operands,
        output_rank,
        element_type,
        "                    ",
        |dim| {
            if dim < token_axes.len() {
                format!("token_coord[s][{}]", token_axes[dim])
            } else {
                format!("feature_coord[q][{}]", feature_axes[dim - token_axes.len()])
            }
        },
        "reduced",
        "out_offset",
    );
    source.push_str("                }\n");
    source.push_str("            }\n");
    source.push_str("        }\n");
    source.push_str("    }\n");
    Ok(())
}

/// Renders [`push_packed_row_multi_row_body`]'s `Q4_K` fast-path reduction
/// loop: the SAME lane split (`ix`/`it`/`iq`/`ir`, `lanes_per_block = 8`) and
/// row-pointer hoist `push_q4k_ggml_port_body`/the M=1 `plain_product` arm
/// already use, generalized to a `cap`-token activation group. Each weight
/// block's header/nibble words are read ONCE per `(ib, q)` inside
/// `q4k_pair_dot_mr` and folded against every one of the `cap` activation
/// rows -- but the activation gather itself happens ONE token at a time,
/// inside `q4k_pair_dot_mr`'s own `s` loop, off `other_base`/`other_stride`/
/// `other_ib_offset` (all scalars/a `cap`-long array of `long`s, not floats)
/// rather than this function pre-gathering `yl_group[cap][16]`/
/// `yh_group[cap][16]` and handing those arrays in. See `q4k_pair_dot_mr`'s
/// own doc for why that pre-gather (256 live private floats at `cap = 8`)
/// measured SLOWER than the generic body this fast path replaced.
pub(super) fn push_packed_row_multi_row_q4k_body(
    source: &mut String,
    weight: usize,
    other: usize,
    rows: usize,
    cap: usize,
    block_bytes: usize,
) {
    source.push_str("    uint ix = (uint)lane / 8u;\n");
    source.push_str("    uint it = (uint)lane % 8u;\n");
    source.push_str("    uint iq = it / 4u;\n    uint ir = it % 4u;\n");
    source.push_str(&format!(
        "    int super_blocks = (int)u.reduction_total / {Q4K_BLOCK_ELEMENTS};\n"
    ));
    source.push_str("    int ib_first = (int)ix;\n    int ib_step = 4;\n");
    source.push_str(&format!(
        "    long blk_step = (long)ib_step * {block_bytes};\n"
    ));
    source.push_str(&format!("    device const uchar *blk_ptr[{rows}];\n"));
    source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!(
        "        blk_ptr[q] = in{weight} + ((long)((int)weight_base[q] / {Q4K_BLOCK_ELEMENTS}) + (long)ib_first) * {block_bytes};\n"
    ));
    source.push_str("    }\n");
    source.push_str(&format!(
        "    long y4_step = (long)ib_step * {Q4K_BLOCK_ELEMENTS} * other_stride;\n"
    ));
    source.push_str(&format!(
        "    long other_ib_offset = (long)ib_first * {Q4K_BLOCK_ELEMENTS} * other_stride;\n"
    ));
    source.push_str("    for (int ib = ib_first; ib < super_blocks; ib += ib_step) {\n");
    source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!("            float result[{cap}];\n"));
    source.push_str(&format!(
        "            q4k_pair_dot_mr(blk_ptr[q], iq, ir, in{other}, other_base, other_stride, other_ib_offset, {cap}u, result);\n"
    ));
    source.push_str(&format!(
        "            for (int s = 0; s < {cap}; ++s) {{ sumf[s][q] = sumf[s][q] + result[s]; }}\n"
    ));
    source.push_str("        }\n");
    source.push_str(&format!(
        "        for (int q = 0; q < {rows}; ++q) {{ blk_ptr[q] += blk_step; }}\n"
    ));
    source.push_str("        other_ib_offset += y4_step;\n");
    source.push_str("    }\n");
}

/// The sole output axis whose bound extent is `> 1`, when every OTHER
/// output axis is degenerate (extent `1`) -- the decode-matvec shape
/// `docs/discipline.md` ROW 350 measured (`output_extents = [1, rows]`).
/// `None` when zero or more than one output axis is non-unit, so the
/// generic N-D coordinate decomposition
/// [`push_packed_row_group_bases`]'s `else` arm renders stays correct for
/// every shape this does not apply to (batched decode, multi-row `M`
/// blocks routed through [`push_packed_row_multi_row_body`] instead, etc).
pub(super) fn packed_row_direct_output_axis(resolved: &BoundOp, output_axes: &[u16]) -> Option<u16> {
    let mut found = None;
    for &axis in output_axes {
        if resolved.extents[axis as usize] > 1 {
            if found.is_some() {
                return None;
            }
            found = Some(axis);
        }
    }
    found
}

/// The grouped-expert sibling of [`packed_row_direct_output_axis`]:
/// `docs/discipline.md` ROW 543/544 named the MoE gate/up reduce
/// (`output_extents = [1, selected, out]`) as the case with exactly TWO
/// non-unit output axes that still fell through to the general `%`/`/`
/// decomposition, at 98.7% of the emitted-body gap ROW 350/351 already
/// closed for the single-axis decode matvec. Returns the `(index, axis)`
/// pair for the outer "selected expert" axis and the inner "out" axis, in
/// the SAME `u.output_extents` index space `push_packed_row_group_bases`'s
/// general arm already reads -- found by walking `output_axes` from the
/// innermost entry backward (mirroring that arm's own decomposition order)
/// and skipping every unit-extent axis along the way, so axes interleaved
/// between the two non-unit ones (always extent 1, or this fast path would
/// not apply) contribute nothing to either bound. `None` whenever zero, one,
/// or three-or-more axes are non-unit, so [`packed_row_direct_output_axis`]
/// and the general decomposition both stay the correct answer for every
/// shape this one does not cover.
pub(super) fn packed_row_direct_grouped_axes(
    resolved: &BoundOp,
    output_axes: &[u16],
) -> Option<(usize, u16, usize, u16)> {
    let mut found: Vec<(usize, u16)> = Vec::new();
    for (index, &axis) in output_axes.iter().enumerate().rev() {
        if resolved.extents[axis as usize] > 1 {
            found.push((index, axis));
            if found.len() > 2 {
                return None;
            }
        }
    }
    let [(out_index, out_axis), (selected_index, selected_axis)] = found[..] else {
        return None;
    };
    Some((selected_index, selected_axis, out_index, out_axis))
}

/// Renders `weight_base[q]`/`other_base[q]`/`coord_q_cache[q]` for one
/// row-blocked group -- the ONE place every [`push_packed_row_blocked_body`]
/// body variant (the default/mask-fma arm, [`push_q4k_ggml_port_body`],
/// [`push_q6k_ggml_port_body`]) gets its row addressing from, since all
/// three run this preamble before branching on which reduce body to emit.
/// ROW 350 named the generic `%`/`/` decomposition against
/// `u.output_extents`/`u.operand_strides` as 16 division-class instructions
/// per thread that are a provable no-op whenever [`packed_row_direct_output_axis`]
/// finds exactly one non-unit output axis (the decode matvec, `s=1`) --
/// this is the SAME emit-time specialization pattern the `other_stride_is_one`
/// stride-free arm above already applies, done here for the row-base
/// addresses instead of the activation stride.
pub(super) fn push_packed_row_group_bases(
    source: &mut String,
    resolved: &BoundOp,
    output_axes: &[u16],
    rank: usize,
    rows: usize,
    weight: usize,
    other: usize,
) {
    let gather_slots = gather_slots(resolved);
    if let Some(axis) = packed_row_direct_output_axis(resolved, output_axes) {
        source.push_str(&format!(
            "    long weight_row_stride = u.operand_strides[{weight}][{axis}];\n"
        ));
        source.push_str(&format!(
            "    long other_row_stride = u.operand_strides[{other}][{axis}];\n"
        ));
        source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
        source.push_str("        long flat = group_first + q;\n");
        source.push_str(&format!(
            "        for (int d = 0; d < {rank}; ++d) {{ coord_q_cache[q][d] = 0; }}\n"
        ));
        source.push_str(&format!("        coord_q_cache[q][{axis}] = flat;\n"));
        source.push_str(&format!(
            "        weight_base[q] = u.operand_base[{weight}] + flat * weight_row_stride;\n"
        ));
        source.push_str(&format!(
            "        other_base[q] = u.operand_base[{other}] + flat * other_row_stride;\n"
        ));
        if let Some(slot) = gather_slots[weight] {
            push_cooperative_gather_fetch(
                source,
                weight,
                slot,
                rank,
                "coord_q_cache[q]",
                "weight_base[q]",
            );
        }
        if let Some(slot) = gather_slots[other] {
            push_cooperative_gather_fetch(
                source,
                other,
                slot,
                rank,
                "coord_q_cache[q]",
                "other_base[q]",
            );
        }
        source.push_str("    }\n");
        return;
    }
    if let Some((_selected_index, selected_axis, out_index, out_axis)) =
        packed_row_direct_grouped_axes(resolved, output_axes)
    {
        // a threadgroup must never straddle two experts: `group_first` is
        // always a multiple of `rows` (`group_first = output_index * rows`),
        // so `out % rows == 0` guarantees every one of this group's `rows`
        // flat indices divides to the SAME `expert_slot` -- computed once
        // here instead of once per `q` the way the general branch's
        // per-thread `%`/`/` chain would.
        if resolved.extents[out_axis as usize].is_multiple_of(rows as u64) {
            source.push_str(&format!(
                "    long weight_selected_stride = u.operand_strides[{weight}][{selected_axis}];\n"
            ));
            source.push_str(&format!(
                "    long weight_out_stride = u.operand_strides[{weight}][{out_axis}];\n"
            ));
            source.push_str(&format!(
                "    long other_selected_stride = u.operand_strides[{other}][{selected_axis}];\n"
            ));
            source.push_str(&format!(
                "    long other_out_stride = u.operand_strides[{other}][{out_axis}];\n"
            ));
            source.push_str(&format!(
                "    long group_out_extent = u.output_extents[{out_index}];\n"
            ));
            source.push_str("    long expert_slot = group_first / group_out_extent;\n");
            source.push_str("    long out_row_base = group_first % group_out_extent;\n");
            source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
            source.push_str(&format!(
                "        for (int d = 0; d < {rank}; ++d) {{ coord_q_cache[q][d] = 0; }}\n"
            ));
            source.push_str("        long out_row = out_row_base + q;\n");
            source.push_str(&format!(
                "        coord_q_cache[q][{selected_axis}] = expert_slot;\n"
            ));
            source.push_str(&format!("        coord_q_cache[q][{out_axis}] = out_row;\n"));
            source.push_str(&format!(
                "        weight_base[q] = u.operand_base[{weight}] + expert_slot * weight_selected_stride + out_row * weight_out_stride;\n"
            ));
            source.push_str(&format!(
                "        other_base[q] = u.operand_base[{other}] + expert_slot * other_selected_stride + out_row * other_out_stride;\n"
            ));
            if let Some(slot) = gather_slots[weight] {
                push_cooperative_gather_fetch(
                    source,
                    weight,
                    slot,
                    rank,
                    "coord_q_cache[q]",
                    "weight_base[q]",
                );
            }
            if let Some(slot) = gather_slots[other] {
                push_cooperative_gather_fetch(
                    source,
                    other,
                    slot,
                    rank,
                    "coord_q_cache[q]",
                    "other_base[q]",
                );
            }
            source.push_str("    }\n");
            return;
        }
    }
    source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str("        long flat = group_first + q;\n");
    source.push_str("        long remaining_q = flat;\n");
    source.push_str(&format!(
        "        for (int d = 0; d < {rank}; ++d) {{ coord_q_cache[q][d] = 0; }}\n"
    ));
    for (index, dim) in output_axes.iter().enumerate().rev() {
        source.push_str(&format!(
            "        coord_q_cache[q][{dim}] = remaining_q % u.output_extents[{index}]; remaining_q /= u.output_extents[{index}];\n"
        ));
    }
    source.push_str(&format!("        long wb = u.operand_base[{weight}];\n"));
    source.push_str(&format!("        long ob = u.operand_base[{other}];\n"));
    // Iterate the OUTPUT axes directly rather than `0..rank` minus one
    // excluded dim: `reduce_dim` is now the innermost of possibly SEVERAL
    // folded reduce dims (see `classify_packed_row_block`'s contiguous-fold
    // check), so `output_axes` -- already the exact complement of every
    // reduce dim, however many there are -- is the correct and simpler set
    // to walk here regardless of reduce rank.
    for &dim in output_axes {
        source.push_str(&format!(
            "        wb += coord_q_cache[q][{dim}] * u.operand_strides[{weight}][{dim}];\n"
        ));
        source.push_str(&format!(
            "        ob += coord_q_cache[q][{dim}] * u.operand_strides[{other}][{dim}];\n"
        ));
    }
    source.push_str("        weight_base[q] = wb;\n");
    source.push_str("        other_base[q] = ob;\n");
    if let Some(slot) = gather_slots[weight] {
        push_cooperative_gather_fetch(
            source,
            weight,
            slot,
            rank,
            "coord_q_cache[q]",
            "weight_base[q]",
        );
    }
    if let Some(slot) = gather_slots[other] {
        push_cooperative_gather_fetch(
            source,
            other,
            slot,
            rank,
            "coord_q_cache[q]",
            "other_base[q]",
        );
    }
    source.push_str("    }\n");
}

/// Every source marker a packed-row-blocked body arm below can emit, owned
/// here next to the bodies that emit them so a new arm's marker is added in
/// ONE place. [`crate::metal::classify_kind`] (the profiler's own
/// classifier, source-greps this same emitted text over in `metal.rs`)
/// consults this list instead of restating its own copy -- the restated copy
/// is exactly what went stale: it carried `q4k_pair_dot(blk`/`q4k_run8(blk`/
/// `q5k_pair_dot(blk`/`q5k_value(blk`/`q6k_value(blk`/`acc1_0` but missed
/// `q3k_pair_dot(blk` (this file, the `Q3K if plain_product` arm above),
/// `q3k_element(blk` (the `Q3K` per-element fallback), and `q6k_pair_dot(blk`
/// (the `Q6K if plain_product` arm) -- every Q3_K row-blocked dispatch and
/// every plain-product Q6_K dispatch (the shape the openchat output head
/// actually takes) undercounted into `"reduce-cooperative"`. A fourth gap
/// found by the lowering census test this same landing adds: `metal`'s own
/// feature list turns `metal-q4k-ggml-port` ON by default (`Cargo.toml`'s
/// `metal = [.., "metal-q4k-ggml-port"]`), and [`push_q6k_ggml_port_body`]
/// -- unlike its `Q4_K`/`Q5_K` siblings, which share `acc1_0` -- calls none
/// of the named helpers and has no `acc1_0` accumulator either, so it fell
/// through this same list even after the first three additions. `"sums0"`
/// closes it (unique to that function's own per-thread partial sums).
// sole caller is `crate::metal::classify_kind`, gated on `metal` (macOS-only
// driver) AND `instrument` (diagnostic-only) -- this crate's `msl` module
// itself only needs `alloc`, so a build with neither compiles this table
// with nothing left to call it.
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos", feature = "instrument")),
    allow(
        dead_code,
        reason = "sole caller is the macOS-only, instrument-gated profiler"
    )
)]
pub(crate) const PACKED_ROW_BODY_MARKERS: &[&str] = &[
    "q3k_pair_dot(blk",
    "q3k_element(blk",
    "q4k_pair_dot(blk",
    // `push_packed_row_multi_row_body`'s own Q4_K-specific staged decode
    // (landed on main while this fix was in flight, found again at rebase
    // time by this same list) -- same gap, same fix, one more marker.
    "q4k_pair_dot_mr(blk",
    "q4k_run8(blk",
    "q5k_pair_dot(blk",
    "q5k_value(blk",
    "q6k_pair_dot(blk",
    "q6k_value(blk",
    // unique to `push_q4k_ggml_port_body`/`push_q5k_ggml_port_body`'s
    // per-thread accumulator naming -- those bodies call none of the
    // helpers above by name, that is the whole point of the ggml port.
    "acc1_0",
    // unique to `push_q6k_ggml_port_body`'s own per-thread partial sums --
    // that body shares neither `acc1_0` nor any named helper call with its
    // Q4_K/Q5_K ggml-port siblings.
    "sums0",
    // `Q4_0`/`Q8_0` are flat (non-K-quant) codecs but still route through
    // `emit_and_classify`'s packed-row whitelist (`Codec::Q4_0 | Codec::Q8_0`
    // arm) -- this table went stale for them the same way it did for `Q3_K`
    // (see this const's own doc above). `push_packed_row_blocked_body`'s
    // single-row arm (`packed_row_blocked_ggml.rs`) emits
    // `q4_0_super_element(blk`/`q8_0_super_element(blk`; the classifier's
    // `op_profile_kind` count for gemma4-E2B's ~275 Q4_0 dispatches fell
    // through this gap into `"reduce-cooperative"` before these two markers.
    "q4_0_super_element(blk",
    "q8_0_super_element(blk",
    // `Codec::Q4_0 if is_plain_product_reduce`/`Codec::Q8_0 if is_plain_
    // product_reduce` (`packed_row_blocked_ggml.rs`, perf/q4_0-pair-dot):
    // the batched fast arm added alongside this landing, ported from the
    // K-quant codecs' own `q4k_pair_dot`/`q5k_pair_dot`/`q6k_pair_dot`
    // shape. Same gap this table's own doc above names for `q4_0_super_
    // element(blk`/`q8_0_super_element(blk`: a marker added for the OLD
    // per-element bodies without one for the new batched body silently
    // undercounts every plain-product `Q4_0`/`Q8_0` dispatch back into
    // `"reduce-cooperative"`.
    "q4_0_pair_dot(blk",
    "q8_0_pair_dot(blk",
    // `push_packed_row_multi_row_body`'s generic (non-`fast_q4k`) loop reads
    // through `signature_tokens_prelude::operand_read` instead of the
    // single-row body's named helpers -- for `Q4_0`/`Q8_0` that renders
    // `q4_0_element(in`/`q8_0_element(in` (no `blk` local, unlike the
    // single-row arm above), so it needs its own, distinct marker text.
    "q4_0_element(in",
    "q8_0_element(in",
];

