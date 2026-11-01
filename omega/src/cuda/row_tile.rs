use alloc::format;
use alloc::string::String;

use proxima_tensor::{BoundOp, BoundOpKind, Keep, NumericPolicy};

use super::{
    Codec, EmitError, WARP_SIZE, chunk_codec, cooperative_identity_token, fold_init_tokens, gather_count,
    flat_packed_rows, operand_read, push_body_steps,
    push_chunk_decode, push_packed_matmul_offset, push_reduce_epilogue_write, push_reduction_coord_carry,
    push_reduction_coord_decompose, reduce_is_cooperative, reduction_dims, scalar_op_expr,
    shuffle_combine_expr,
};

/// Activation rows one warp folds against a weight run it decodes once.
pub(super) const ROW_TILE: usize = 8;

/// The output-axis position a cooperative packed reduce tiles `ROW_TILE` rows
/// along, or `None` when the op keeps the one-warp-per-output-element body.
/// The axis is one every packed operand ignores (stride 0) and some unpacked
/// operand varies along, so a decoded weight run serves every row of the tile.
/// Decided from strides and operand kinds only -- never extents -- so the
/// emitted text stays extent-independent and a reused kernel stays valid.
pub(super) fn row_tile_axis(
    resolved: &BoundOp,
    quantized: &[Option<Codec>],
    numeric_policy: NumericPolicy,
) -> Option<usize> {
    let BoundOpKind::Reduce {
        keep: Keep::Reduce,
        output_axes,
        epilogue_broadcast_axes,
        ..
    } = &resolved.kind
    else {
        return None;
    };
    let eligible = epilogue_broadcast_axes.is_empty()
        && gather_count(resolved) == 0
        && quantized.iter().any(Option::is_some)
        && !reduction_dims(resolved, output_axes).is_empty()
        && flat_packed_rows(resolved, quantized)
        && reduce_is_cooperative(resolved, quantized, numeric_policy);
    if !eligible {
        return None;
    }
    let operands = resolved.operands();
    output_axes.iter().rposition(|&axis| {
        let packed_ignores = quantized
            .iter()
            .zip(operands)
            .filter(|(codec, _)| codec.is_some())
            .all(|(_, (_, layout, _))| layout.stride(axis) == 0);
        let unpacked_varies = quantized
            .iter()
            .zip(operands)
            .filter(|(codec, _)| codec.is_none())
            .any(|(_, (_, layout, _))| layout.stride(axis) != 0);
        packed_ignores && unpacked_varies && resolved.extents[axis as usize] > 1
    })
}

pub(super) fn push_row_tiled_cooperative_body(
    source: &mut String,
    resolved: &BoundOp,
    quantized: &[Option<Codec>],
    element_type: &str,
    tile_index: usize,
) -> Result<(), EmitError> {
    let BoundOpKind::Reduce {
        reduce_op,
        init,
        output_axes,
        epilogue_body,
        epilogue_operands,
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
    let output_rank = output_axes.len();
    let reduce_dims = reduction_dims(resolved, output_axes);
    let reduce_rank = reduce_dims.len();
    let operand_count = resolved.operands().len();
    let tile_axis = output_axes[tile_index] as usize;
    let (init_expr, seeded_init) = fold_init_tokens(*init);
    let identity = cooperative_identity_token(resolved.node, *reduce_op)?;

    source.push_str(&format!("    long output_index = gid / {WARP_SIZE};\n"));
    source.push_str(&format!(
        "    long row_extent = u.output_extents[{tile_index}];\n"
    ));
    source.push_str(&format!(
        "    long row_tiles = (row_extent + {ROW_TILE} - 1) / {ROW_TILE};\n"
    ));
    source.push_str("    long warp_total = u.output_total / row_extent * row_tiles;\n");
    source.push_str("    if (output_index >= warp_total) { return; }\n");
    source.push_str(&format!(
        "    unsigned int lane = (unsigned int)(gid % {WARP_SIZE});\n"
    ));
    source.push_str(&format!("    long full_coord[{rank_len}];\n"));
    for dim in 0..rank {
        source.push_str(&format!("    full_coord[{dim}] = 0;\n"));
    }
    source.push_str(&format!("    long output_coord[{output_rank}];\n"));
    source.push_str("    long remaining = output_index;\n");
    for index in (0..output_rank).rev() {
        let extent = if index == tile_index {
            "row_tiles".to_string()
        } else {
            format!("u.output_extents[{index}]")
        };
        source.push_str(&format!(
            "    output_coord[{index}] = remaining % {extent}; remaining /= {extent};\n"
        ));
    }
    for (index, dim) in output_axes.iter().enumerate() {
        source.push_str(&format!("    full_coord[{dim}] = output_coord[{index}];\n"));
    }
    source.push_str(&format!(
        "    long row_first = output_coord[{tile_index}] * {ROW_TILE};\n"
    ));
    source.push_str(&format!(
        "    long rows_here = min((long){ROW_TILE}, row_extent - row_first);\n"
    ));

    for (index, &codec) in quantized.iter().enumerate() {
        if codec.is_some() {
            continue;
        }
        source.push_str(&format!("    long row_off{index}[{ROW_TILE}];\n"));
        source.push_str("    #pragma unroll\n");
        source.push_str(&format!(
            "    for (int slot = 0; slot < {ROW_TILE}; slot++) {{\n"
        ));
        source.push_str(&format!(
            "        long row_coord = row_first + slot;\n        row_off{index}[slot] = u.operand_base[{index}];\n"
        ));
        for &dim in output_axes {
            let coord = if dim as usize == tile_axis {
                "row_coord".to_string()
            } else {
                format!("full_coord[{dim}]")
            };
            source.push_str(&format!(
                "        row_off{index}[slot] += {coord} * u.operand_strides[{index}][{dim}];\n"
            ));
        }
        source.push_str("    }\n");
    }

    source.push_str(&format!("    {element_type} accumulator[{ROW_TILE}];\n"));
    source.push_str("    bool seeded = (lane == 0u) ? ");
    source.push_str(seeded_init);
    source.push_str(" : true;\n");
    source.push_str("    #pragma unroll\n");
    source.push_str(&format!(
        "    for (int slot = 0; slot < {ROW_TILE}; slot++) {{\n"
    ));
    source.push_str(&format!(
        "        accumulator[slot] = (lane == 0u) ? {init_expr} : {identity};\n"
    ));
    source.push_str("    }\n");

    source.push_str(&format!(
        "    for (long chunk = (long)lane * 8; chunk < u.reduction_total; chunk += {WARP_SIZE} * 8) {{\n"
    ));
    push_reduction_coord_decompose(source, "chunk", reduce_rank, reduce_rank.max(1), "        ");
    source.push_str("        long r = chunk;\n");
    for (index, &codec) in quantized.iter().enumerate() {
        let Some(codec) = codec else { continue };
        let (_, layout, _) = &resolved.operands()[index];
        push_packed_matmul_offset(source, index, layout, output_axes, resolved, "        ");
        push_weight_chunk(source, resolved, index, codec)?;
    }
    source.push_str(
        "        for (long inner = 0; inner < 8 && chunk + inner < u.reduction_total; inner++) {\n",
    );
    for (index, dim) in reduce_dims.iter().enumerate() {
        source.push_str(&format!(
            "            full_coord[{dim}] = reduction_coord[{index}];\n"
        ));
    }
    for (index, &codec) in quantized.iter().enumerate() {
        if codec.is_some() {
            continue;
        }
        source.push_str(&format!("            long reduce_off{index} = 0;\n"));
        for dim in &reduce_dims {
            source.push_str(&format!(
                "            reduce_off{index} += full_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
            ));
        }
    }
    source.push_str("            #pragma unroll\n");
    source.push_str(&format!(
        "            for (int slot = 0; slot < {ROW_TILE}; slot++) {{\n"
    ));
    source.push_str("                if (slot >= rows_here) { continue; }\n");
    source.push_str(&format!(
        "                {element_type} scratch[{}];\n",
        operand_count.max(1)
    ));
    for (index, &codec) in quantized.iter().enumerate() {
        let read = match codec {
            Some(codec) => format!("{}[inner]", weight_chunk_name(codec, index)),
            None => format!("in{index}[row_off{index}[slot] + reduce_off{index}]"),
        };
        source.push_str(&format!("                scratch[{index}] = {read};\n"));
    }
    let value_expr = push_body_steps(
        source,
        resolved.element_body(),
        "                ",
        element_type,
    );
    source.push_str(&format!(
        "                {element_type} value = {value_expr};\n"
    ));
    let combine_expr = scalar_op_expr(*reduce_op, &["accumulator[slot]", "value"]);
    source.push_str(&format!(
        "                accumulator[slot] = seeded ? {combine_expr} : value;\n"
    ));
    source.push_str("            }\n");
    source.push_str("            seeded = true;\n");
    if reduce_rank > 0 {
        push_reduction_coord_carry(source, reduce_rank);
    }
    source.push_str("        }\n");
    source.push_str("    }\n");

    source.push_str("    #pragma unroll\n");
    source.push_str(&format!(
        "    for (int slot = 0; slot < {ROW_TILE}; slot++) {{\n"
    ));
    source.push_str("        if (slot >= rows_here) { continue; }\n");
    source.push_str("        #pragma unroll\n");
    source.push_str(&format!(
        "        for (int shift = {}; shift > 0; shift >>= 1) {{\n",
        WARP_SIZE / 2
    ));
    source.push_str(&format!(
        "            {element_type} shuffled = __shfl_down_sync(0xffffffffu, accumulator[slot], shift);\n"
    ));
    let shuffle_expr = shuffle_combine_expr(
        resolved.node,
        *reduce_op,
        "accumulator[slot]",
        "shuffled",
    )?;
    source.push_str(&format!("            accumulator[slot] = {shuffle_expr};\n"));
    source.push_str("        }\n");
    source.push_str("    }\n");

    source.push_str("    if (lane != 0u) { return; }\n");
    source.push_str("    #pragma unroll\n");
    source.push_str(&format!(
        "    for (int slot = 0; slot < {ROW_TILE}; slot++) {{\n"
    ));
    source.push_str("        if (slot >= rows_here) { continue; }\n");
    source.push_str(&format!(
        "        output_coord[{tile_index}] = row_first + slot;\n"
    ));
    source.push_str(&format!(
        "        full_coord[{tile_axis}] = row_first + slot;\n"
    ));
    source.push_str("        long out_offset = u.out_base;\n");
    for dim in 0..rank {
        source.push_str(&format!(
            "        out_offset += full_coord[{dim}] * u.out_strides[{dim}];\n"
        ));
    }
    push_reduce_epilogue_write(
        source,
        epilogue_body,
        epilogue_operands,
        output_rank,
        element_type,
        "        ",
        |dim| format!("output_coord[{dim}]"),
        "accumulator[slot]",
        "out_offset",
    );
    source.push_str("    }\n");
    Ok(())
}

fn weight_chunk_name(codec: Codec, index: usize) -> String {
    match codec {
        Codec::Q4_0 => format!("q4_0_chunk{index}"),
        Codec::Q8_0 => format!("q8_0_chunk{index}"),
        Codec::Q6K => format!("q6k_chunk{index}"),
        _ => format!("weight_chunk{index}"),
    }
}

fn push_weight_chunk(
    source: &mut String,
    resolved: &BoundOp,
    index: usize,
    codec: Codec,
) -> Result<(), EmitError> {
    match codec {
        Codec::Q4_0 | Codec::Q8_0 | Codec::Q6K => match chunk_codec(codec) {
            Some(spec) => push_chunk_decode(source, index, spec),
            None => return Err(EmitError::CudaUnsupportedCodec { node: resolved.node }),
        },
        _ => {
            let read = operand_read(resolved.node, index, &format!("(off{index} + slot)"), Some(codec))?;
            source.push_str(&format!("        float weight_chunk{index}[8];\n"));
            source.push_str("        #pragma unroll\n");
            source.push_str("        for (int slot = 0; slot < 8; slot++) {\n");
            source.push_str(&format!(
                "            weight_chunk{index}[slot] = (chunk + slot < u.reduction_total) ? (float)({read}) : 0.0f;\n"
            ));
            source.push_str("        }\n");
        }
    }
    Ok(())
}
