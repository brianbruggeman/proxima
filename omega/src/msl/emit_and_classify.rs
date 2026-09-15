use super::*;

pub fn emit(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
) -> Result<Kernel, EmitError> {
    emit_inner(resolved, packed_operands, numeric_policy, false)
}

pub(super) fn emit_inner(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
    expert_source_mode: bool,
) -> Result<Kernel, EmitError> {
    validate(resolved)?;
    let entry = entry_name(resolved);
    let quantized = operand_codecs(resolved, packed_operands);
    let source = match &resolved.kind {
        BoundOpKind::CachedAttention { .. } => {
            render_cached_attention(resolved, &entry, numeric_policy)
        }
        BoundOpKind::Elementwise { .. } => render_elementwise(resolved, &entry, &quantized),
        BoundOpKind::Reduce {
            keep: Keep::Reduce, ..
        } => render_reduce(
            resolved,
            &entry,
            &quantized,
            numeric_policy,
            expert_source_mode,
        ),
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => render_scan(resolved, &entry, &quantized),
        BoundOpKind::Iota => render_iota(resolved, &entry),
        BoundOpKind::Constant { value } => render_constant(resolved, &entry, *value),
        BoundOpKind::GatedDeltaNet { .. } => render_gated_delta_net(resolved, &entry),
        BoundOpKind::MoeTopK { .. } => render_moe_topk(resolved, &entry),
    }?;
    // Coupled to `render_cached_attention`'s own final-store branch by
    // construction: whenever the rendered SOURCE writes the scratch layout
    // instead of the real output (`cached_attention_merge_needed`), the
    // BINDINGS this call returns must say so too (`Binding::Scratch`
    // replacing `Binding::Output`) -- source and bindings are two views of
    // the SAME `emit` call, keyed on the SAME `numeric_policy`, so they can
    // never disagree with each other. What they CAN disagree with is a
    // caller that lacks the merge dispatch this binding shape requires --
    // `crate::metal::encode_op`'s own doc names that gap and the guard it
    // takes on its `resolved: None` (no plan-resolved merge sibling) path.
    let is_split_cached_attention = cached_attention_merge_needed(&resolved.kind, numeric_policy);
    Ok(Kernel {
        source,
        entry,
        bindings: if is_split_cached_attention {
            split_bindings_with_scratch(resolved)
        } else {
            bindings(resolved)
        },
        grid: GridSpec {
            threads: grid_threads(resolved, &quantized, numeric_policy, expert_source_mode)?,
            threadgroup_width: tiled_gemm_threadgroup_width(resolved, &quantized, numeric_policy),
            depth: 1,
        },
    })
}

/// Emits the ordinary kernel ABI plus the two buffers required by a HOBBIT
/// substitution table.  The decoder is deliberately not selected here: this
/// establishes the stable binding contract that the mixed-codec lowering will
/// consume, while ordinary [`emit`] callers remain byte-for-byte unchanged.
pub fn emit_with_expert_sources(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
    source_node: NodeId,
) -> Result<Kernel, EmitError> {
    emit_with_expert_sources_mode(resolved, packed_operands, numeric_policy, source_node, None)
}

/// Emits a substituted expert kernel while retaining the bound operand's
/// packed decoder. This is valid only when every selected descriptor uses the
/// same codec; mixed selections must use [`emit_with_expert_sources`].
pub fn emit_with_uniform_expert_source(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
    source_node: NodeId,
    codec: PackedCodec,
) -> Result<Kernel, EmitError> {
    emit_with_expert_sources_mode(
        resolved,
        packed_operands,
        numeric_policy,
        source_node,
        Some(codec),
    )
}

pub(super) fn emit_with_expert_sources_mode(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
    source_node: NodeId,
    uniform_codec: Option<PackedCodec>,
) -> Result<Kernel, EmitError> {
    // A substituted expert source may select a different codec for every
    // routed expert. Do not specialize this operand to the checkpoint's
    // original codec: the packed-row bodies bake one decoder into the kernel
    // and would interpret a low-copy Q2_K expert as Q4_K. Rendering the source
    // operand as scalar first leaves every read visible to the descriptor-aware
    // replacement below, whose codec tag selects the decoder at runtime.
    let mut source_packed_operands = packed_operands.clone();
    if let Some(codec) = uniform_codec {
        source_packed_operands.insert(source_node, codec);
    } else {
        source_packed_operands.remove(&source_node);
    }
    let mut kernel = emit_inner(
        resolved,
        &source_packed_operands,
        numeric_policy,
        uniform_codec.is_none(),
    )?;
    // Replace the gathered weight read with the descriptor-aware form. The
    // ordinary emitter remains unchanged; this opt-in path is selected only
    // when a residency table is supplied for this operand.
    if let Some((weight_index, (_, _, Some(_)))) = resolved
        .operands()
        .iter()
        .enumerate()
        .find(|(_, (node, _, _lookup))| *node == source_node)
    {
        let gather_slot = resolved
            .operands()
            .iter()
            .take(weight_index)
            .filter(|(_, _, lookup)| lookup.is_some())
            .count();
        let row_block_source = kernel.source.contains("long weight_base[");
        let offset = format!("off{weight_index}");
        for codec in [
            None,
            Some(PackedCodec::Q2K),
            Some(PackedCodec::Q4K),
            Some(PackedCodec::Q5K),
            Some(PackedCodec::Q6K),
        ] {
            for offset_name in [
                offset.as_str(),
                &format!("walk{weight_index}"),
                &format!("read_off{weight_index}"),
                &format!("running{weight_index}"),
                "(weight_base[q] + k)",
            ] {
                let expert_stride = if row_block_source {
                    String::from("0l")
                } else {
                    format!("u.gather_element_stride[{gather_slot}]")
                };
                let old = operand_read(weight_index, offset_name, codec);
                let replacement = if offset_name == format!("walk{weight_index}") {
                    format!(
                        "mixed_expert_element_from_local(expert_base{weight_index}, expert_codec{weight_index}, {offset_name})"
                    )
                } else {
                    format!(
                        "mixed_expert_element_from_offset(expert_payloads, expert_descriptors, (uint)fetched{weight_index}, {offset_name}, {}, expert_codec{weight_index})",
                        expert_stride,
                    )
                };
                let replacement = if let Some(codec) = uniform_codec {
                    format!(
                        "uniform_expert_element_from_offset_{}(expert_payloads, expert_descriptors, (uint)fetched{}, {offset_name}, {})",
                        codec.cache_token(),
                        weight_index,
                        expert_stride,
                    )
                } else {
                    replacement
                };
                kernel.source = kernel.source.replace(&old, &replacement);
            }
        }
        let expert_base =
            String::from("expert_payloads + expert_descriptors[expert_route_index[q]].byte_offset");
        // packed-row group bases fetch the route inside their per-row `q`
        // scope.  The row-blocked body consumes that route later, after the
        // scope has closed, so preserve it in a row-indexed array rather than
        // referring to the local `fetchedN` declaration out of scope.
        kernel.source = kernel
            .source
            .replace("long weight_base[", "long expert_route_index[");
        if let Some(start) = kernel.source.find("long expert_route_index[")
            && let Some(end) = kernel.source[start..].find(';')
        {
            let declaration_end = start + end + 1;
            let declaration = kernel.source[start..declaration_end].to_owned();
            let dimension = declaration
                .split_once('[')
                .and_then(|(_, rest)| rest.split_once(']'))
                .map(|(value, _)| value)
                .unwrap_or("1");
            kernel.source = kernel.source.replacen(
                &declaration,
                &format!("long weight_base[{dimension}];\n    long expert_route_index[{dimension}];\n    long expert_row_base[{dimension}];"),
                1,
            );
        }
        kernel.source = kernel
            .source
            .lines()
            .map(|line| {
                let marker = format!("blk_ptr[q] = in{weight_index} + ");
                let packed_block_marker = format!(
                    "device const uchar *blk = in{weight_index} + "
                );
                let packed_block_marker_compact = format!(
                    "device const uchar* blk = in{weight_index} + "
                );
                if line.contains(&marker)
                    || (row_block_source
                        && (line.contains(&packed_block_marker)
                            || line.contains(&packed_block_marker_compact)))
                {
                    line.replace(
                        &format!("in{weight_index} + "),
                        &format!("{expert_base} + "),
                    )
                        .replace(
                            "weight_base[q]",
                            &format!(
                                "(expert_row_base[q] - u.operand_base[{weight_index}] - expert_route_index[q] * u.gather_element_stride[{gather_slot}])"
                            ),
                        )
                } else if line.contains(&format!(
                    "fetched{weight_index} = (long)simd_broadcast_first"
                )) {
                    let missing_expert_guard = format!(
                        "        if (expert_descriptors[(uint)fetched{weight_index}].byte_length == 0u) {{\n            atomic_fetch_max_explicit(&fault[{gather_slot}], 0x80000000u | min((uint)fetched{weight_index} + 1u, 0x7fffffffu), memory_order_relaxed);\n            return;\n        }}"
                    );
                    if row_block_source {
                        format!(
                            "{line}\n{missing_expert_guard}\n        uint expert_codec{weight_index} = expert_descriptors[(uint)fetched{weight_index}].codec;\n        device const uchar *expert_base{weight_index} = expert_payloads + expert_descriptors[(uint)fetched{weight_index}].byte_offset;\n        expert_route_index[q] = fetched{weight_index};"
                        )
                    } else {
                        format!(
                            "{line}\n{missing_expert_guard}\n    uint expert_codec{weight_index} = expert_descriptors[(uint)fetched{weight_index}].codec;\n    device const uchar *expert_base{weight_index} = expert_payloads + expert_descriptors[(uint)fetched{weight_index}].byte_offset;"
                        )
                    }
                } else if !row_block_source
                    && line.trim().starts_with(&format!(
                        "fetched{weight_index} = max((long)0, min(fetched{weight_index}"
                    ))
                    && !kernel.source.contains(&format!(
                        "fetched{weight_index} = (long)simd_broadcast_first"
                    ))
                {
                    // The generic gathered body does not broadcast the route
                    // because each lane owns a distinct reduction element.
                    // It still needs the descriptor codec before the first
                    // mixed-element read below. When the broadcast marker is
                    // also present (cooperative gather fetch), the other arm
                    // above already declares this codec once.
                    format!(
                        "{line}\n    uint expert_codec{weight_index} = expert_descriptors[(uint)fetched{weight_index}].codec;"
                    )
                } else if !row_block_source
                    && (line.contains(&packed_block_marker)
                        || line.contains(&packed_block_marker_compact))
                {
                    // The ordinary packed body has no row index `q`; its
                    // fetched expert is broadcast in `fetchedN`.  Rebase the
                    // full-stack offset to that expert's compact payload
                    // before selecting the codec-specific block bytes.
                    let mut rewritten = line.replace(
                        &format!("in{weight_index} + "),
                        &format!(
                            "expert_payloads + expert_descriptors[(uint)fetched{weight_index}].byte_offset + "
                        ),
                    );
                    let full_base = format!("base{weight_index}");
                    rewritten = rewritten.replacen(
                        &full_base,
                        &format!(
                            "(base{weight_index} - fetched{weight_index} * u.gather_element_stride[{gather_slot}])"
                        ),
                        1,
                    );
                    let slot_base = String::from("slot_off");
                    rewritten.replacen(
                        &slot_base,
                        &format!(
                            "(slot_off - fetched{weight_index} * u.gather_element_stride[{gather_slot}])"
                        ),
                        1,
                    )
                } else if line.contains(&format!("int walk{weight_index} = (int)off{weight_index};")) {
                    format!(
                        "{line}\n        walk{weight_index} -= fetched{weight_index} * u.gather_element_stride[{gather_slot}];"
                    )
                } else if line.trim() == "weight_base[q] = wb;" {
                    format!("{line}\n        expert_row_base[q] = wb;")
                } else if line.trim().starts_with("weight_base[q] += fetched") {
                    format!("{line}\n        expert_row_base[q] = weight_base[q];")
                } else { line.to_owned() }
            })
            .collect::<Vec<_>>()
            .join("\n");
        if std::env::var_os("PROXIMA_DEBUG_EXPERT_EMIT").is_some() {
            eprintln!(
                "qwen35 expert lowering bound_node={:?} source_node={source_node:?} extents={:?} row_block={} multi_row={} gather={} token_total={} source_len={}",
                resolved.node,
                resolved.extents,
                row_block_source,
                kernel.source.contains("token_total"),
                kernel.source.contains("gather_idx0"),
                kernel.source.contains("token_total"),
                kernel.source.len(),
            );
            for line in kernel.source.lines().filter(|line| {
                line.contains("mixed_expert")
                    || line.contains("expert_route_index")
                    || line.contains("fetched")
                    || line.contains("walk0")
                    || line.contains("base0")
            }) {
                eprintln!("qwen35 expert msl: {line}");
            }
            if std::env::var_os("PROXIMA_DEBUG_EXPERT_SOURCE_FULL").is_some()
                && source_node == NodeId(3)
            {
                eprintln!(
                    "qwen35 expert source full begin node={source_node:?}\n{}\nqwen35 expert source full end",
                    kernel.source
                );
            }
        }
    }
    let payload_binding = Binding::ExpertPayloads(source_node);
    let descriptor_binding = Binding::ExpertDescriptors(source_node);
    let next_buffer = kernel.bindings.len();
    let signature = format!("kernel void {}(", kernel.entry);
    let signature_start = kernel
        .source
        .find(&signature)
        .ok_or(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "emitted kernel signature",
            found: "missing",
        })?;
    let body_start = kernel.source[signature_start..]
        .find(")\n{\n")
        .map(|offset| signature_start + offset)
        .ok_or(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "emitted kernel signature terminator",
            found: "missing",
        })?;
    // `MIXED_EXPERT_READ_MSL` contributes the descriptor declaration in the
    // shared preamble; do not splice a second definition into each kernel.
    let descriptor_struct = "";
    kernel.source.insert_str(signature_start, descriptor_struct);
    let adjusted_body_start = body_start + descriptor_struct.len();
    let parameters = format!(
        ",\n    device const uchar* expert_payloads [[buffer({next_buffer})]],\n    device const ExpertPayloadDescriptor* expert_descriptors [[buffer({})]]",
        next_buffer + 1
    );
    kernel.source.insert_str(adjusted_body_start, &parameters);
    kernel.bindings.push(payload_binding);
    kernel.bindings.push(descriptor_binding);
    Ok(kernel)
}

/// Splices the z-indexed base-table preamble (`docs/discipline.md`'s
/// horizontal-packed-merge design note, §5) onto an already-emitted
/// packed-row kernel: a `SliceBase` struct ahead of the signature, the
/// existing scalar `gid` parameter widened to `uint3 merge_gid` (Metal
/// rejects a signature mixing a scalar and a vector thread-position
/// attribute, so a second, separately-attributed z parameter does not
/// compile) plus a `base_table` parameter appended to it, and a `gid`
/// local restoring the scalar reads the rest of the body already makes,
/// one `SliceBase` read, and three renamed pointer locals right after the
/// body's opening brace -- the same additive-splice shape [`emit_with_expert_sources`]
/// already uses for `ExpertPayloads`/`ExpertDescriptors` above, rather than
/// threading a new parameter through [`push_packed_row_blocked_body`]'s
/// ~800-line body.
///
/// Every later reference to `in{weight_index}`/`in{other_index}`/`out` in the
/// body is RENAMED to a sliced local (`replace_whole_word`), not shadowed in
/// place: a C-family declaration's own name comes into scope at its
/// declarator, so `device const uchar* in0 = in0 + base;` would read the
/// UNINITIALIZED new `in0`, never the parameter -- this splice sidesteps that
/// trap by giving the offset locals distinct names.
///
/// Driver-side buffer binding for `base_table` (a new plan-owned `Binding`,
/// not `NodeId`-keyed like every other binding this module defines) is the
/// design note's encode-loop step, not this function's job: this proves the
/// emitted TEXT is correct in isolation, which is what lets it be unit-tested
/// without a device (see this module's own `horizontal_merge_base_table_
/// splice_tests`).
#[cfg(feature = "metal-horizontal-merge")]
pub(crate) fn splice_horizontal_merge_base_table(
    kernel: &mut Kernel,
    node: NodeId,
    weight_index: usize,
    weight_type: &str,
    other_index: usize,
    other_type: &str,
    element_type: &str,
) -> Result<(), EmitError> {
    let struct_decl =
        "struct SliceBase { ulong weight_base; ulong activation_base; ulong output_base; };\n";
    let signature = format!("kernel void {}(", kernel.entry);
    let signature_start =
        kernel
            .source
            .find(&signature)
            .ok_or(EmitError::RenderKindMismatch {
                node,
                expected: "emitted kernel signature",
                found: "missing",
            })?;
    let body_start = kernel.source[signature_start..]
        .find(")\n{\n")
        .map(|offset| signature_start + offset)
        .ok_or(EmitError::RenderKindMismatch {
            node,
            expected: "emitted kernel signature terminator",
            found: "missing",
        })?;
    kernel.source.insert_str(signature_start, struct_decl);
    let body_start_after_struct = body_start + struct_decl.len();
    // Metal rejects a kernel signature mixing a scalar and a vector
    // thread-position attribute ("expecting input declarations with either
    // all scalar types or all vector types with the same number of
    // elements") -- every packed-row kernel already declares a scalar `uint
    // gid [[thread_position_in_grid]]` (`kernel_signature`'s own doc), so
    // adding a SEPARATE `uint3 ... [[threadgroup_position_in_grid]]`
    // parameter for the z-slice index does not compile. This widens the
    // EXISTING `gid` parameter to `uint3` instead (renamed to avoid
    // colliding with the scalar local the preamble below reintroduces) --
    // one 3D dispatch already gives one threadgroup per z-slice
    // (`crate::metal::dispatch`'s own `threadgroup.depth == 1`), so
    // `merge_gid.z` and what a separate `threadgroup_position_in_grid.z`
    // would have read are the identical value.
    let scalar_gid = "    uint gid [[thread_position_in_grid]]";
    let vector_gid = "    uint3 merge_gid [[thread_position_in_grid]]";
    let gid_offset = kernel.source[signature_start..body_start_after_struct]
        .find(scalar_gid)
        .ok_or(EmitError::RenderKindMismatch {
            node,
            expected: "scalar thread_position_in_grid parameter",
            found: "missing",
        })?;
    kernel.source.replace_range(
        signature_start + gid_offset..signature_start + gid_offset + scalar_gid.len(),
        vector_gid,
    );
    let width_delta = vector_gid.len() - scalar_gid.len();
    let base_table_index = kernel.bindings.len();
    let extra_params =
        format!(",\n    device const SliceBase* base_table [[buffer({base_table_index})]]");
    let adjusted_body_start = body_start_after_struct + width_delta;
    kernel.source.insert_str(adjusted_body_start, &extra_params);
    // `)\n{\n` starts at `adjusted_body_start` post-splice: `)` + `\n` + `{`
    // is 3 bytes, so the byte right after `{` is `adjusted_body_start +
    // extra_params.len() + 3`.
    let preamble_start = adjusted_body_start + extra_params.len() + 3;
    let preamble = format!(
        "    uint gid = merge_gid.x;\n    SliceBase merge_base = base_table[merge_gid.z];\n    device const {weight_type}* sliced_weight = (device const {weight_type}*)((device const uchar*)in{weight_index} + merge_base.weight_base);\n    device const {other_type}* sliced_other = (device const {other_type}*)((device const uchar*)in{other_index} + merge_base.activation_base);\n    device {element_type}* sliced_out = (device {element_type}*)((device uchar*)out + merge_base.output_base);\n"
    );
    kernel.source.insert_str(preamble_start, &preamble);
    let body_after_preamble = preamble_start + preamble.len();
    let mut tail = replace_whole_word(
        &kernel.source[body_after_preamble..],
        &format!("in{weight_index}"),
        "sliced_weight",
    );
    tail = replace_whole_word(&tail, &format!("in{other_index}"), "sliced_other");
    tail = replace_whole_word(&tail, "out", "sliced_out");
    kernel.source.truncate(body_after_preamble);
    kernel.source.push_str(&tail);
    Ok(())
}

/// Whole-word substring replace: a plain [`str::replace`] would also match
/// `in1` inside `in10`, corrupting a sibling operand's name, so this checks
/// both neighbours of every candidate match are not identifier characters
/// before accepting it. Generated MSL source is plain ASCII (identifiers and
/// numeric literals only), so byte-wise scanning is exact here.
#[cfg(feature = "metal-horizontal-merge")]
pub(super) fn replace_whole_word(text: &str, identifier: &str, replacement: &str) -> String {
    let bytes = text.as_bytes();
    let pattern = identifier.as_bytes();
    let is_identifier_byte = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_';
    let mut output = String::with_capacity(text.len());
    let mut index = 0;
    while index < bytes.len() {
        let candidate_matches = bytes[index..].starts_with(pattern);
        let boundary_before = index == 0 || !is_identifier_byte(bytes[index - 1]);
        let after = index + pattern.len();
        let boundary_after = after >= bytes.len() || !is_identifier_byte(bytes[after]);
        if candidate_matches && boundary_before && boundary_after {
            output.push_str(replacement);
            index = after;
        } else {
            output.push(bytes[index] as char);
            index += 1;
        }
    }
    output
}

/// The row-blocked/tiled-GEMM structural shape [`kernel_cache_key`] folds
/// into [`crate::identity::MetalOnlyExtras::packed_row_block_shape`] — 'G'
/// (tiled `simdgroup_matrix` GEMM, checked FIRST: [`tiled_gemm_block`] only
/// ever returns `Some` when [`packed_row_block`] also would, since it is
/// built ON TOP of that same gate, so the two are mutually exclusive by
/// construction and this order costs nothing extra to get right), 'M'/'B'
/// (row-blocked, multiple/single activation row per streamed weight row),
/// 'S' (fully serial, every non-`Reduce` op and every `Reduce` neither path
/// claims). Kept here, not in `identity.rs`: every function it calls is
/// Metal-only private state ([`PackedRowBlock`], [`tiled_gemm_block`]).
// same gate as `kernel_cache_key`, its sole caller -- see that function's
// own comment for why `metal-core` alone (not `metal`) is the right feature.
#[cfg(any(test, feature = "metal-core"))]
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    allow(dead_code, reason = "sole caller is the macOS-only metal driver")
)]
pub(super) fn packed_row_block_shape_token(resolved: &BoundOp, quantized: &[Option<PackedCodec>]) -> char {
    let BoundOpKind::Reduce {
        reduce_op,
        init,
        output_axes,
        ..
    } = &resolved.kind
    else {
        return 'S';
    };
    if tiled_gemm_block(resolved, quantized, *reduce_op, *init, output_axes).is_some() {
        'G'
    } else if let Some(block) = packed_row_block(resolved, quantized) {
        if packed_row_block_token_total(&block, &resolved.extents) > 1 {
            'M'
        } else {
            'B'
        }
    } else {
        'S'
    }
}

/// Whether the packed row-block's non-weight operand reads unit stride on
/// the reduce axis — `push_packed_row_blocked_body`'s STRIDE-FREE
/// SPECIALIZATION (see that function's own doc) renders different source
/// text for the SAME 'B'/'M' structural shape depending on this CONCRETE
/// resolved stride, not just op structure. `None` when [`packed_row_block`]
/// has no match at all — [`packed_row_block_shape_token`]'s 'G'/'S' arms
/// never render that specialization, so a stray extra token there would
/// only cost an unnecessary cache miss, never a wrong hit; leaving it out
/// entirely is just as sound and keeps a non-matching op's identity free of
/// a token it has no reason to carry.
#[cfg(any(test, feature = "metal-core"))]
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    allow(dead_code, reason = "sole caller is the macOS-only metal driver")
)]
pub(super) fn packed_row_block_stride_is_one(
    resolved: &BoundOp,
    quantized: &[Option<PackedCodec>],
) -> Option<bool> {
    let BoundOpKind::Reduce { .. } = &resolved.kind else {
        return None;
    };
    let block = packed_row_block(resolved, quantized)?;
    Some(
        resolved.operands()[block.other]
            .1
            .stride(block.reduce_dim as u16)
            == 1,
    )
}

/// Whether [`push_packed_row_group_bases`] rendered the direct single-axis
/// row addressing for this op, and on which axis — `None` when it did not
/// (no `packed_row_block` match, or a multi-row `M` block, which always
/// takes the generic path; see [`packed_row_direct_output_axis`]'s own doc).
#[cfg(any(test, feature = "metal-core"))]
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    allow(dead_code, reason = "sole caller is the macOS-only metal driver")
)]
pub(super) fn packed_row_block_direct_axis(
    resolved: &BoundOp,
    quantized: &[Option<PackedCodec>],
) -> Option<u16> {
    let BoundOpKind::Reduce { output_axes, .. } = &resolved.kind else {
        return None;
    };
    let block = packed_row_block(resolved, quantized)?;
    if packed_row_block_token_total(&block, &resolved.extents) > 1 {
        return None;
    }
    packed_row_direct_output_axis(resolved, output_axes)
}

/// Whether [`push_packed_row_group_bases`] rendered the grouped two-axis
/// direct addressing (ROW 545) for this op, and on which `(selected, out)`
/// axis pair -- `None` when it did not, mirroring
/// [`packed_row_block_direct_axis`]'s own gate. Two bindings that agree on
/// every other cache-key axis but disagree on whether the grouped fast path
/// applied (the `out % rows_per_simdgroup` divisibility check
/// [`push_packed_row_group_bases`] itself runs) render different addressing
/// source text and must never share a pipeline-cache entry -- this is that
/// disambiguator, folded into [`crate::identity::kernel_identity`] the same
/// way the single-axis case already is.
#[cfg(any(test, feature = "metal-core"))]
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    allow(dead_code, reason = "sole caller is the macOS-only metal driver")
)]
pub(super) fn packed_row_block_grouped_axes(
    resolved: &BoundOp,
    quantized: &[Option<PackedCodec>],
) -> Option<(u16, u16)> {
    let BoundOpKind::Reduce { output_axes, .. } = &resolved.kind else {
        return None;
    };
    let block = packed_row_block(resolved, quantized)?;
    if packed_row_block_token_total(&block, &resolved.extents) > 1 {
        return None;
    }
    let (_, selected_axis, _, out_axis) = packed_row_direct_grouped_axes(resolved, output_axes)?;
    let rows = block.codec.rows_per_simdgroup() as u64;
    if resolved.extents[out_axis as usize].is_multiple_of(rows) {
        Some((selected_axis, out_axis))
    } else {
        None
    }
}

/// Cheap structural + compile-option identity for the kernel [`emit`] would
/// produce from `resolved` — built without ever rendering the MSL body
/// text, so a caller can decide whether a pipeline compile is needed before
/// paying for one. A thin Metal-specific wrapper around
/// [`crate::identity::kernel_identity`]: this crate's own [`entry_name`]
/// stays a byte-identical, narrower fingerprint (embedded literally as the
/// compiled kernel's own declared name, so its format can never change
/// without changing emitted MSL source); this key is the fuller identity a
/// caller uses to decide pipeline-cache reuse, never embedded in source
/// text, so its format is free to change as the union of axes it must
/// distinguish grows. See [`crate::identity`]'s module doc for the full
/// axis census and why one shared function derives every renderer's own
/// version of this fact instead of three.
///
/// # Errors
/// Propagates [`type_token`]'s unsupported-dtype rejection — the same gate
/// [`emit`] enforces before ever building a kernel.
// production caller is `crate::metal::encode_op` (macOS-only driver, gated
// on `metal`, which now implies `metal-core`); the `mod tests` call sites
// below are the second, so `cfg(test)` keeps a non-macOS `cargo test` build
// honest. No macOS-specific code lives in this function body, so it is
// gated on `metal-core` alone -- it builds (and is exercised) on Linux too.
// `metal-core` without `metal` (the Linux emitter-only build) has no driver
// to call it, hence the `allow`: genuinely unreachable there, not a hidden bug.
#[cfg(any(test, feature = "metal-core"))]
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    allow(dead_code, reason = "sole caller is the macOS-only metal driver")
)]
pub(crate) fn kernel_cache_key(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
) -> Result<String, EmitError> {
    // Called for its unsupported-dtype rejection alone -- `kernel_identity`
    // reads `resolved.dtype` directly for the actual half/wide classing (the
    // same partition every renderer's own `type_token` match already makes),
    // but `kernel_cache_key` has no `validate` call of its own upstream of
    // it, so this stays the one place that fails fast on a dtype `emit`
    // would also reject.
    type_token(resolved.node, resolved.dtype)?;
    let quantized = operand_codecs(resolved, packed_operands);
    let extras = crate::identity::MetalOnlyExtras {
        cooperative_width: tiled_gemm_threadgroup_width(resolved, &quantized, numeric_policy),
        packed_row_block_shape: Some(packed_row_block_shape_token(resolved, &quantized)),
        packed_row_block_stride_is_one: packed_row_block_stride_is_one(resolved, &quantized),
        packed_row_block_direct_axis: packed_row_block_direct_axis(resolved, &quantized),
        packed_row_block_grouped_axes: packed_row_block_grouped_axes(resolved, &quantized),
        elementwise_addressing: elementwise_addressing_cache_token(resolved),
        numeric_policy_token: Some(crate::identity::numeric_policy_cache_token(numeric_policy)),
        merged_z: None,
    };
    Ok(crate::identity::kernel_identity(
        crate::identity::KernelLanguage::Metal,
        resolved,
        packed_operands,
        extras,
        numeric_policy,
    ))
}

/// The dispatch-time shape of `resolved`'s kernel — buffer bindings and
/// thread count — without rendering any MSL body text. Cheap on every call
/// regardless of pipeline-cache hit or miss: [`emit`]'s `source`/`entry`
/// fields are needed only on a genuine cache miss (see
/// `crate::metal::encode_op`).
///
/// # Errors
/// Propagates [`validate`]'s structural rejection — the same gate [`emit`]
/// enforces before ever building a kernel.
// unlike `kernel_cache_key` above, this has no `mod tests` call site of its
// own, so a bare `cfg(test)` disjunct leaves it genuinely dead-code on a
// non-macOS `cargo test`/`nextest` build -- gate on the one real caller
// (`crate::metal::encode_op`, via `metal-core`, same rationale as above).
#[cfg(feature = "metal-core")]
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    allow(dead_code, reason = "sole caller is the macOS-only metal driver")
)]
pub(crate) fn kernel_dispatch_shape(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
) -> Result<(Vec<Binding>, GridSpec), EmitError> {
    validate(resolved)?;
    let quantized = operand_codecs(resolved, packed_operands);
    let is_split_cached_attention = cached_attention_merge_needed(&resolved.kind, numeric_policy);
    Ok((
        if is_split_cached_attention {
            split_bindings_with_scratch(resolved)
        } else {
            bindings(resolved)
        },
        GridSpec {
            // `kernel_dispatch_shape` has no expert-source caller (it never
            // took `expert_source_mode` before this parameter existed
            // either) -- `false` reproduces that pre-existing scope exactly.
            threads: grid_threads(resolved, &quantized, numeric_policy, false)?,
            threadgroup_width: tiled_gemm_threadgroup_width(resolved, &quantized, numeric_policy),
            depth: 1,
        },
    ))
}

// `SIMD_WIDTH` moved to `crate::sized::SIMD_WIDTH` (the build-time floor's
// only configuration surface) -- imported at the top of this file.

/// Whether `resolved` is a `Keep::Reduce` fold whose `reduce_op` is
/// associative and commutative (`Add`, `Multiply`, `Maximum`, `Minimum`) with
/// a gather whose index lookup is invariant across every reduction axis, AND
/// whose reduced-axis extent meets
/// [`crate::sized::COOPERATIVE_REDUCE_MIN_LEN`] — the set [`render_reduce`]
/// emits a SIMD-group cooperative loop for instead of the one-thread-per-
/// output serial fold. `Subtract`/`Divide` are not associative, so
/// reordering their combination across lanes is not imprecise, it is wrong —
/// they and every other `ScalarOp` stay on the serial path. A gather is
/// admitted only when its index lookup is constant over every reduced axis;
/// the fetched index is then broadcast once per simdgroup and the existing
/// lane-0 fault reporting remains sufficient.
///
/// The length gate exists because a cooperative reduce always launches
/// `SIMD_WIDTH`(32) lanes per output regardless of how many elements each
/// output folds — a 34-long attention reduce pays a full `simd_sum` combine
/// for 34 elements of real work across 32 mostly-idle lanes. `min_len == 0`
/// (the sentinel, not a real length any reduce can be shorter than) makes
/// this check vacuous, so every op that clears the op/gather gate above
/// stays cooperative — the routing every build before this key existed used,
/// and the `omega-runtime.toml` default: that "mostly-idle lanes" framing
/// turned out to predict the wrong direction on real hardware (that file's
/// own `[cooperative_reduce]` doc has the measured numbers) — these
/// reduces are memory-latency-bound, and the serial route's 32x-fewer
/// threads hides less load latency than the idle lanes cost, so routing
/// short reduces off cooperative made them slower, not faster.
pub(super) fn reduce_is_cooperative(resolved: &BoundOp) -> bool {
    match &resolved.kind {
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            reduce_op,
            output_axes,
            ..
        } => {
            is_cooperative_reduce_op(*reduce_op)
                && gather_is_reduction_invariant(resolved, output_axes)
                && meets_cooperative_min_len(reduction_len(resolved, output_axes))
        }
        _ => false,
    }
}

/// The cooperative renderer is a tree reduction across SIMD lanes.  That is
/// the [`NumericRewrite::TreeReduce`] rewrite, so a bit-exact plan must keep
/// the serial renderer even when the shape itself is eligible.  Keeping the
/// policy gate beside the structural gate prevents dispatch geometry and the
/// emitted body from selecting different fold orders.
pub(super) fn reduce_is_cooperative_for_policy(resolved: &BoundOp, policy: NumericPolicy) -> bool {
    reduce_is_cooperative(resolved) && admit(policy, NumericRewrite::TreeReduce).is_ok()
}

/// Whether `resolved` carries a broadcast-reduce epilogue (RMSNorm-shaped
/// `x * inv_rms`) -- the one write tail only [`push_cooperative_reduce_tail`]
/// renders (`render_reduce`'s own doc on `is_broadcast_epilogue`).
pub(super) fn reduce_has_broadcast_epilogue(resolved: &BoundOp) -> bool {
    matches!(
        &resolved.kind,
        BoundOpKind::Reduce { epilogue_broadcast_axes, .. } if !epilogue_broadcast_axes.is_empty()
    )
}

/// Whether this reduce takes the cooperative kernel STRUCTURE -- distinct
/// from whether its plain combine may reassociate. [`push_cooperative_reduce_body`]
/// hosts every packed-row/tiled-GEMM/gather/expert-source specialization
/// ([`tiled_gemm_block`], [`packed_row_block`]) as sub-branches that read
/// their operand once per lane and never fold across lanes in a
/// numerically-visible order -- [`NumericRewrite::TreeReduce`] permission is
/// only needed by the one sub-branch that DOES (the generic per-lane
/// `simd_sum` walk `push_cooperative_reduce_body` falls back to). Gating
/// entry to the whole function on that permission -- what a bare
/// [`reduce_is_cooperative_for_policy`] call here would do -- strands every
/// specialized decoder behind a policy check they never needed, sending a
/// bit-exact default policy (every fixture and the default runtime) to
/// [`push_serial_reduce_body`], which renders none of them: this is the
/// packed-row-blocked/tiled-GEMM/expert-source regression, not the
/// broadcast-epilogue one alone. A broadcast epilogue is folded in here for
/// the same reason -- see [`reduce_has_broadcast_epilogue`]'s own doc.
pub(super) fn reduce_is_cooperative_dispatch(
    resolved: &BoundOp,
    quantized: &[Option<PackedCodec>],
    policy: NumericPolicy,
    reduce_op: ScalarOp,
    init: ReduceInit,
    output_axes: &[u16],
    expert_source_mode: bool,
) -> bool {
    // `emit_with_expert_sources_mode`'s string substitution hunts for the
    // cooperative walk's own `walk{index}`/`off{index}`/`fetched{index}`
    // spellings (`render_reduce`'s doc on this parameter) -- the serial
    // renderer has none of those names, so expert-source hoisting needs the
    // cooperative structure exactly as unconditionally as a packed/tiled
    // match does, regardless of `quantized[weight]` (deliberately stripped
    // to `None` for the substituted operand, so neither block classifier
    // below ever fires for it).
    if expert_source_mode
        || reduce_has_broadcast_epilogue(resolved)
        || tiled_gemm_block(resolved, quantized, reduce_op, init, output_axes).is_some()
        || packed_row_block_admitted(resolved, quantized)
    {
        reduce_is_cooperative(resolved)
    } else {
        reduce_is_cooperative_for_policy(resolved, policy)
    }
}

/// [`packed_row_block`]'s own admission, plus a `debug!` naming the exact
/// [`PackedRowBlockRejection`] on decline -- this is the one call site that
/// decides whether a reduce ever reaches the packed-row-blocked kernel at
/// all (every other `packed_row_block` call site downstream only ever fires
/// on an op this admission already accepted, so logging only here names
/// each verdict once, not once per caller). ROW 538 (`docs/discipline.md`):
/// without this, a declined grouped-expert reduce silently fell back to
/// `reduce-cooperative` with no record of which of the seven
/// [`classify_packed_row_block`] conditions gave up on it.
pub(super) fn packed_row_block_admitted(resolved: &BoundOp, quantized: &[Option<PackedCodec>]) -> bool {
    // The experimental mixed-source path must use the descriptor-aware
    // element reader; row-block pointer hoisting has a separate address ABI.
    if std::env::var_os("PROXIMA_ENABLE_UNSAFE_METAL_EXPERT_SOURCES").is_some() {
        return false;
    }
    match classify_packed_row_block(resolved, quantized) {
        Ok(_) => true,
        Err(reason) => {
            debug_packed_row_block_decline(resolved.node, &reason);
            false
        }
    }
}

#[cfg(feature = "instrument")]
pub(super) fn debug_packed_row_block_decline(node: NodeId, reason: &PackedRowBlockRejection) {
    proxima_telemetry::debug!(
        node = node.0,
        reason = ?reason,
        "packed-row-block admission declined; reduce falls back to cooperative dispatch"
    );
}

#[cfg(not(feature = "instrument"))]
pub(super) fn debug_packed_row_block_decline(_node: NodeId, _reason: &PackedRowBlockRejection) {}

/// A route selected by the token axis has zero stride in contracted
/// dimensions, so one fetched expert index can be shared by all lanes. Any
/// nonzero stride would select different experts during the reduction and is
/// therefore kept on the serial path.
pub(super) fn gather_is_reduction_invariant(resolved: &BoundOp, output_axes: &[u16]) -> bool {
    let reduce_dims = reduction_dims(resolved, output_axes);
    resolved.operands().iter().all(|(_, _, lookup)| {
        lookup.as_ref().is_none_or(|lookup| {
            reduce_dims
                .iter()
                .all(|&dim| lookup.index_layout.stride(dim) == 0)
        })
    })
}

/// `length >= COOPERATIVE_REDUCE_MIN_LEN`, factored out so clippy's
/// `absurd_extreme_comparisons` lint has one site to allow rather than every
/// call site: at the `omega-runtime.toml` default (0, `u64::MIN`) the
/// comparison IS always true, and that is the intended behavior (see
/// `reduce_is_cooperative`'s own doc) -- a config-driven threshold cannot be
/// assumed non-degenerate by the linter, but `OMEGA_COOPERATIVE_REDUCE_MIN_
/// LEN` overriding it to a real value at build time makes this a genuine
/// runtime-varying comparison, not dead code.
#[allow(clippy::absurd_extreme_comparisons)]
pub(super) fn meets_cooperative_min_len(length: u64) -> bool {
    length >= crate::sized::COOPERATIVE_REDUCE_MIN_LEN
}

/// Total element count one output folds over: the product of the extents of
/// every dim [`reduction_dims`] names. Zero-rank (a scalar operand reduced
/// over nothing) has no `reduce_dims`, so `product()` over the empty
/// iterator correctly yields `1` — one element, itself.
pub(super) fn reduction_len(resolved: &BoundOp, output_axes: &[u16]) -> u64 {
    reduction_dims(resolved, output_axes)
        .iter()
        .map(|&dim| resolved.extents[dim as usize])
        .product()
}

pub(super) fn is_cooperative_reduce_op(op: ScalarOp) -> bool {
    matches!(
        op,
        ScalarOp::Add | ScalarOp::Multiply | ScalarOp::Maximum | ScalarOp::Minimum
    )
}

/// The MSL SIMD-group reduction builtin that combines one lane's private
/// accumulator across the whole 32-lane group — only called for a
/// [`is_cooperative_reduce_op`] body, so the non-cooperative arms below are
/// enumerated rather than wildcarded — adding a `ScalarOp` variant forces a
/// decision here instead of silently panicking.
pub(super) fn simd_combine_fn(node: NodeId, op: ScalarOp) -> Result<&'static str, EmitError> {
    match op {
        ScalarOp::Add => Ok("simd_sum"),
        ScalarOp::Multiply => Ok("simd_product"),
        ScalarOp::Maximum => Ok("simd_max"),
        ScalarOp::Minimum => Ok("simd_min"),
        ScalarOp::Identity
        | ScalarOp::Subtract
        | ScalarOp::Divide
        | ScalarOp::Negate
        | ScalarOp::Reciprocal
        | ScalarOp::Exponential
        | ScalarOp::Logarithm
        | ScalarOp::SquareRoot
        | ScalarOp::Tanh
        | ScalarOp::Erf
        | ScalarOp::Greater
        | ScalarOp::Equal
        | ScalarOp::Select => Err(EmitError::NonCooperativeReduceOp {
            node,
            op: op_token(op),
        }),
    }
}

/// The algebraic identity `op` folds against without changing a value: `e op
/// x == x` for every `x`. Every SIMD lane but lane 0 seeds its private
/// accumulator with this (never with the `BoundOp`'s own `ReduceInit`, which
/// may be `FirstElement` or otherwise mismatched with `op`) — folding that
/// untouched identity into the final `simd_*` combine can never perturb the
/// result, because `e op e == e` holds for any identity by definition. Lane
/// 0 alone carries the real seed, so it is folded into the group exactly
/// once, matching `cpu::run_reduce`'s single-seed semantics regardless of
/// how many idle lanes there are.
pub(super) fn cooperative_identity_token(node: NodeId, op: ScalarOp) -> Result<&'static str, EmitError> {
    match op {
        ScalarOp::Add => Ok("0.0f"),
        ScalarOp::Multiply => Ok("1.0f"),
        ScalarOp::Maximum => Ok("-INFINITY"),
        ScalarOp::Minimum => Ok("INFINITY"),
        ScalarOp::Identity
        | ScalarOp::Subtract
        | ScalarOp::Divide
        | ScalarOp::Negate
        | ScalarOp::Reciprocal
        | ScalarOp::Exponential
        | ScalarOp::Logarithm
        | ScalarOp::SquareRoot
        | ScalarOp::Tanh
        | ScalarOp::Erf
        | ScalarOp::Greater
        | ScalarOp::Equal
        | ScalarOp::Select => Err(EmitError::NonCooperativeReduceOp {
            node,
            op: op_token(op),
        }),
    }
}

/// Structural checks over a (possibly fused) [`ComposedBody`]: every step's
/// own arity matches its arg count — the same check [`validate`] always ran,
/// now per absorbed step instead of once for a single `ScalarOp`, since a
/// fused body can carry more than one.
pub(super) fn validate_body(node: NodeId, body: &ComposedBody) -> Result<(), EmitError> {
    for step in &body.steps {
        let expected = step.op.arity();
        let found = step.args.len();
        if expected != found {
            return Err(EmitError::ArityMismatch {
                node,
                expected,
                found,
            });
        }
    }
    Ok(())
}

pub(super) fn validate(resolved: &BoundOp) -> Result<(), EmitError> {
    validate_body(resolved.node, resolved.element_body())?;
    if let BoundOpKind::Reduce {
        reduce_op,
        keep,
        out_scatter,
        ..
    } = &resolved.kind
    {
        if out_scatter.is_some() {
            return Err(EmitError::ScatterNotSupported {
                node: resolved.node,
            });
        }
        if matches!(reduce_op, ScalarOp::Select) {
            return Err(EmitError::ReductionBodyIsSelect {
                node: resolved.node,
            });
        }
        if *keep == Keep::Scan && resolved.extents.is_empty() {
            return Err(EmitError::EmptyScan {
                node: resolved.node,
            });
        }
    }
    Ok(())
}

/// `pub(crate)`, not private: the Metal driver's uniforms packer
/// (`crate::metal::pack_reduce_uniforms`) needs the exact same reduce-dim set
/// this rendering uses, and duplicating the filter would risk the two
/// drifting apart.
pub(crate) fn reduction_dims(resolved: &BoundOp, output_axes: &[u16]) -> Vec<u16> {
    (0..resolved.extents.len() as u16)
        .filter(|dim| !output_axes.contains(dim))
        .collect()
}

pub(super) fn bindings(resolved: &BoundOp) -> Vec<Binding> {
    // `all_read_sources`, not `operands` -- a `BoundOpKind::Reduce` with a
    // fused epilogue reads its `epilogue_operands` too, and those need a
    // buffer bound at the exact index `kernel_signature`'s `epi{index}`
    // params claim (see that function's own doc).
    let mut bindings: Vec<Binding> = resolved
        .all_read_sources()
        .map(|(node, _, _)| Binding::Input(*node))
        .collect();
    for (_, _, gather) in resolved.operands() {
        if let Some(gather_access) = gather {
            bindings.push(Binding::Indices(gather_access.indices));
        }
    }
    bindings.push(Binding::Output(resolved.node));
    bindings.push(Binding::Uniforms);
    if gather_count(resolved) > 0 {
        bindings.push(Binding::Fault);
    }
    bindings
}

/// [`bindings`]'s split-kernel counterpart for `BoundOpKind::CachedAttention`
/// once [`cached_attention_merge_needed`] admits `ContextSplitMerge`: the
/// same read set, but the LAST slot -- the op's own output -- becomes
/// [`Binding::Scratch`] instead of [`Binding::Output`], because the split
/// kernel no longer writes `resolved.node`'s real buffer; the merge kernel
/// does. See [`render_cached_attention_merge`]'s own doc for the kernel that
/// reads this scratch buffer back out.
pub(super) fn split_bindings_with_scratch(resolved: &BoundOp) -> Vec<Binding> {
    let mut list = bindings(resolved);
    for binding in &mut list {
        if let Binding::Output(_) = binding {
            *binding = Binding::Scratch;
        }
    }
    list
}

/// [`render_cached_attention_merge`]'s own binding list: reads the scratch
/// buffer [`split_bindings_with_scratch`] wrote, writes the op's real
/// output, and needs its own (smaller) `Uniforms` blob -- no operand inputs,
/// no gather, no fault buffer, since the merge is pure scratch-to-output
/// rescale-and-copy.
#[cfg(any(test, all(feature = "metal", target_os = "macos")))]
pub(super) fn merge_bindings(resolved: &BoundOp) -> Vec<Binding> {
    alloc::vec![
        Binding::Scratch,
        Binding::Output(resolved.node),
        Binding::Uniforms
    ]
}

/// Every `NodeId` `bindings` reads from a device buffer for — the exact
/// operand set `crate::metal::bind_buffers` resolves for a
/// `Binding::Input`/`Binding::Indices` slot, in bind order. This is the one
/// adapter a hazard tracker (or anything
/// else that needs "what does this dispatch read") should walk, instead of
/// re-deriving the read set from `BoundOp::all_read_sources()` independently
/// — ROW 323's bug was exactly two call sites (`bindings()` here and the
/// hazard tracker's own operand enumeration) drifting apart when
/// `bindings()` grew a source `all_read_sources()`'s caller had not been
/// updated to match. Walking `bindings` itself makes that drift impossible:
/// there is only one list, and both the encoder bind loop and the hazard
/// walk read the same one.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn hazard_read_nodes(bindings: &[Binding]) -> impl Iterator<Item = NodeId> + '_ {
    bindings.iter().filter_map(|binding| match binding {
        Binding::Input(node) | Binding::Indices(node) => Some(*node),
        Binding::Output(_)
        | Binding::ExpertPayloads(_)
        | Binding::ExpertDescriptors(_)
        | Binding::Uniforms
        | Binding::Fault
        | Binding::Scratch => None,
    })
}

/// The single `NodeId` `bindings` writes to — every bound op writes exactly
/// one device buffer (`crate::metal::bind_buffers`'s own doc), so a well-formed
/// `bindings` list always has exactly one `Binding::Output`. `None` only if
/// `bindings` is malformed (a validation bug upstream, not a runtime case a
/// caller should expect to hit).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn hazard_write_node(bindings: &[Binding]) -> Option<NodeId> {
    bindings.iter().find_map(|binding| match binding {
        Binding::Output(node) => Some(*node),
        // `Binding::Scratch`'s write target has no `NodeId` -- see
        // `Binding::Scratch`'s own doc. A caller tracking hazards across the
        // split dispatch's scratch write needs the scratch buffer's own
        // pointer identity directly, not through this `NodeId` path.
        Binding::Input(_)
        | Binding::ExpertPayloads(_)
        | Binding::ExpertDescriptors(_)
        | Binding::Indices(_)
        | Binding::Uniforms
        | Binding::Fault
        | Binding::Scratch => None,
    })
}

/// For each operand, `Some(slot)` if it gathers — `slot` is its position
/// among only the gathered operands, 0-based, matching the order
/// [`bindings`] appends `Indices` buffers and the order the `Uniforms`
/// gather arrays are packed in. `pub(crate)` for the same reason
/// [`reduction_dims`] is: the Metal driver's uniforms packer needs the exact
/// same numbering.
pub(crate) fn gather_slots(resolved: &BoundOp) -> Vec<Option<usize>> {
    let mut next = 0usize;
    resolved
        .operands()
        .iter()
        .map(|(_, _, gather)| {
            gather.as_ref().map(|_| {
                let slot = next;
                next += 1;
                slot
            })
        })
        .collect()
}

pub(crate) fn gather_count(resolved: &BoundOp) -> usize {
    resolved
        .operands()
        .iter()
        .filter(|(_, _, gather)| gather.is_some())
        .count()
}

/// Total independent units of work `resolved` needs — see [`GridSpec`]'s doc
/// for why this, unlike [`Kernel::source`], is genuinely a function of
/// `resolved`'s concrete extents.
/// Output rows one SIMD group folds at once in the packed path — ggml's
/// `N_R0_Q4_K`. The point is the ACTIVATION: its run of
/// [`Q4K_BLOCK_ELEMENTS`]/[`SIMD_WIDTH`] values is loaded into registers
/// once and reused across all four rows, so activation traffic falls 4x and
/// the per-row work becomes one header decode plus the nibble extracts.
pub(super) const PACKED_ROWS_PER_GROUP: usize = 4;

/// Edge length of the `simdgroup_matrix` tile `push_tiled_gemm_body` uses —
/// `simdgroup_float8x8`/`simdgroup_half8x8` are fixed 8x8 by the MSL type
/// itself on every Apple GPU family that supports them, the same "hardware
/// fact, not a policy knob" class [`SIMD_WIDTH`] is in
/// (`crate::sized`'s own module doc draws this exact line): there is no
/// tuning that would make this anything but 8, so it stays a bare `const`
/// rather than threading through the sizing-config mechanism
/// [`crate::sized::TILED_GEMM_MIN_TOKENS`] uses. Only [`push_tiled_gemm_body`]
/// reads it, so it is gated the same as that function -- see its `#[cfg(not(..))]`
/// stub's own doc for why the non-feature build never needs it.
#[cfg(feature = "metal-tiled-gemm")]
pub(super) const TILE_DIM: usize = 8;

/// Number of `simdgroup`s cooperating in one [`push_tiled_gemm_body`]
/// threadgroup — ports `ggml-metal.metal:6500`'s `kernel_mul_mm` dispatch
/// (`ggml-metal.m:3102`'s `threadsPerThreadgroup:MTLSizeMake(128, 1, 1)`,
/// 128/32 = 4 `simdgroup`s). Fixed at 4 (a 2x2 grid: `sgitg & 1` selects
/// which half of [`crate::sized::TILED_GEMM_BLOCK_M`]'s rows, `sgitg >> 1`
/// selects which half of [`crate::sized::TILED_GEMM_BLOCK_N`]'s columns,
/// exactly ggml's own `mc[8]`/`THREAD_MAT_M`/`THREAD_MAT_N` split) rather
/// than threaded through the sizing-config mechanism: the 2x2 halving is
/// baked into the pointer arithmetic `push_tiled_gemm_body` emits, so a
/// value other than 4 would need a different kernel body, not just a
/// different constant — the same "hardware fact, not a policy knob" class
/// [`TILE_DIM`] and [`crate::sized::SIMD_WIDTH`] are in. `BLOCK_M`/`BLOCK_N`
/// themselves ARE the tunable axes (`crate::sized::TILED_GEMM_BLOCK_M`/
/// `TILED_GEMM_BLOCK_N`) — this only fixes how many simdgroups split them.
pub(super) const TILED_GEMM_NSG: usize = 4;

/// The one decision that both [`grid_threads`] and
/// [`push_cooperative_reduce_body`] must reach identically: whether this
/// bound op takes the row-blocked packed path. They compute different things
/// from it (dispatch geometry, kernel body), and a disagreement would not
/// fail to compile — it would silently fold the wrong rows. So it is decided
/// once, here, from the bound layout.
pub(super) struct PackedRowBlock {
    /// operand index of the packed weight
    pub(super) weight: usize,
    /// operand index of the single non-packed operand (the activation)
    pub(super) other: usize,
    pub(super) reduce_dim: usize,
    /// which codec `weight`'s bytes are packed as — decides the block byte
    /// width and which unpack function the emitted body calls.
    pub(super) codec: PackedCodec,
    /// output axes the activation owns exclusively, outermost first --
    /// empty when the op's output axes do not split cleanly into a
    /// token/feature ownership partition (every axis then counts as a
    /// feature axis; see `push_packed_row_blocked_body`'s single-row arm).
    pub(super) token_axes: Vec<u16>,
    /// output axes the weight owns exclusively, outermost first -- every
    /// output axis when `token_axes` is empty.
    pub(super) feature_axes: Vec<u16>,
}

/// The token/feature ownership split `push_packed_row_blocked_body` needs to
/// decide whether more than one activation row can be folded per streamed
/// weight row: every output axis partitions into a token group (nonzero
/// stride on `other`, zero on `weight`) and a feature group (the reverse),
/// each nesting contiguously in every layout that reads it. `None` when any
/// of those conditions fails -- the caller then treats every output axis as
/// a feature axis (`token_axes` empty), which is exactly today's row-blocked
/// behaviour for an op this split does not apply to.
pub(super) fn split_token_feature_axes(
    output_axes: &[u16],
    weight_layout: &Layout,
    other_layout: &Layout,
    out_layout: &Layout,
    extents: &[u64],
) -> Option<(Vec<u16>, Vec<u16>)> {
    let mut token_axes: Vec<u16> = Vec::new();
    let mut feature_axes: Vec<u16> = Vec::new();
    for &axis in output_axes {
        match (
            weight_layout.stride(axis) == 0,
            other_layout.stride(axis) == 0,
        ) {
            (true, false) => token_axes.push(axis),
            (false, true) => feature_axes.push(axis),
            _ => return None,
        }
    }
    if feature_axes.is_empty() {
        return None;
    }
    let reassembled: Vec<u16> = token_axes
        .iter()
        .chain(feature_axes.iter())
        .copied()
        .collect();
    if reassembled != output_axes {
        return None;
    }
    let groups_contiguous = axes_fold_contiguously(&token_axes, extents, other_layout)
        && axes_fold_contiguously(&feature_axes, extents, weight_layout)
        && axes_fold_contiguously(&token_axes, extents, out_layout)
        && axes_fold_contiguously(&feature_axes, extents, out_layout);
    if !groups_contiguous {
        return None;
    }
    Some((token_axes, feature_axes))
}

/// Product of `block.token_axes`' extents -- `1` when empty (no distinct
/// token axis, or the split did not apply), matching an ordinary product
/// over zero terms. [`push_packed_row_blocked_body`]'s own branch on
/// whether this exceeds `1` is the single decision point for which kernel
/// body shape gets emitted; [`kernel_cache_key`] and the dispatch-geometry
/// functions below all re-derive the identical value from the identical
/// block so none of them can drift from what the body actually emits.
pub(super) fn packed_row_block_token_total(block: &PackedRowBlock, extents: &[u64]) -> u64 {
    block
        .token_axes
        .iter()
        .map(|&axis| extents[axis as usize])
        .product()
}

/// Why a given [`BoundOp`] did NOT take the row-blocked packed kernel —
/// `classify_packed_row_block`'s error arm, one variant per gate in that
/// function's own condition order. `#[non_exhaustive]` so a new gate added
/// later is a compile error at every match site instead of a silently
/// unmatched `_`. Always compiled (not feature-gated itself) so
/// `classify_packed_row_block` — called from the unconditional emit path
/// — never needs a second copy of these seven conditions; only the public
/// accessor `diagnose_packed_row_block` is gated behind `instrument`, this
/// crate's diagnostic-only feature (see
/// `crate::metal::execute_plan_op_timed`'s own doc for why diagnostics
/// live behind that gate).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PackedRowBlockRejection {
    /// `reduce_is_cooperative` is false — not `Add`/`Multiply`/`Maximum`/
    /// `Minimum`, or the op gathers.
    NotCooperativeReduce,
    /// Not a `Reduce { keep: Keep::Reduce, .. }` at all.
    NotReduceKeepReduce,
    /// `quantized.len() != 2` — not a two-operand (weight, activation) op.
    OperandCountNotTwo,
    /// Neither exactly zero nor exactly one operand is packed.
    NotExactlyOnePackedOperand,
    /// Gathered packed weights need the opt-in gathered row renderer; the
    /// default build keeps the generic cooperative gather-aware path until
    /// `metal-gathered-packed-row` has been enabled and measured.
    GatheredOperand,
    /// The packed operand's codec is [`PackedCodec::Q8_0`] or
    /// [`PackedCodec::Q4_0`] — this path's lane amortization
    /// ([`Q4K_BLOCK_ELEMENTS`], 8 lanes per 32-element sub-block) is
    /// hard-coded to the K-quant family's shared 256-element super-block,
    /// which neither flat 32-element codec has an analogue for. Both
    /// always take the fully generic per-element path instead (see
    /// [`Q8_0_UNPACK_MSL`]/[`Q4_0_UNPACK_MSL`]'s own docs). Checked by
    /// WHITELISTING the three K-quant variants rather than blacklisting
    /// `Q8_0` alone -- an equality check against one non-K-quant codec
    /// silently admits any OTHER non-K-quant codec whose extent happens to
    /// be a multiple of 256 (`docs/discipline.md`'s own landmine: `Q8_0`'s
    /// addition was caught only because this was rewritten as a match, not
    /// because the single `==` check would have caught `Q4_0` too).
    NotKQuantCodec,
    /// The reduce folds ZERO axes into its output — degenerate, never
    /// observed on a real matmul (kept so the match stays exhaustive over
    /// every way `reduce_dims` (`reduction_dims`) can come back empty).
    NotExactlyOneReduceDim { reduce_dims: Vec<u16> },
    /// More than one reduce dim, but they do NOT nest contiguously for both
    /// operands (see `classify_packed_row_block`'s own doc for the
    /// contiguous-fold check this fails) — cannot be treated as one
    /// flattened reduction, so the generic per-element path runs instead.
    ReduceDimsNotContiguous { reduce_dims: Vec<u16> },
    /// The packed operand's stride at the innermost reduce dim is not 1.
    NonUnitWeightStride { stride: i64 },
    /// The flattened extent across every reduce dim is not a whole
    /// multiple of [`Q4K_BLOCK_ELEMENTS`].
    ExtentNotBlockMultiple { extent: u64 },
}

/// Whether `dims` (given OUTERMOST-first, i.e. `dims.last()` is the
/// fastest/innermost axis — the same convention [`reduction_dims`]'s own
/// callers already use) is one contiguous nested block in `layout`: each
/// outer axis's stride equals the extent of every axis nested inside it
/// times that inner axis's own stride. A single dim (or empty) trivially
/// passes (`windows(2)` yields nothing to check).
///
/// The one identity two independent folds both lean on: `classify_packed_row_block`'s
/// reduce-dim fold (below) and [`classify_tiled_gemm`]'s token/feature-axis-group
/// fold both need "a single flat index times the innermost axis's stride
/// addresses the same memory a full per-axis decomposition would" to be
/// true, and it is true exactly when this check passes — never a special
/// case for how many dims fold, or for reduce vs. output axes.
pub(super) fn axes_fold_contiguously(dims: &[u16], extents: &[u64], layout: &Layout) -> bool {
    dims.windows(2).all(|window| {
        let [outer, inner] = window else {
            return false;
        };
        let inner_extent = extents[*inner as usize] as i64;
        layout.stride(*outer) == inner_extent * layout.stride(*inner)
    })
}

/// The one decision [`packed_row_block`] and [`diagnose_packed_row_block`]
/// both need — this function is the single source of truth;
/// `packed_row_block` is `.ok()` over it so there is exactly one place the
/// seven conditions are spelled out, never two copies that could drift.
pub(super) fn classify_packed_row_block(
    resolved: &BoundOp,
    quantized: &[Option<PackedCodec>],
) -> Result<PackedRowBlock, PackedRowBlockRejection> {
    if !reduce_is_cooperative(resolved) {
        return Err(PackedRowBlockRejection::NotCooperativeReduce);
    }
    let BoundOpKind::Reduce {
        keep: Keep::Reduce,
        output_axes,
        out_layout,
        ..
    } = &resolved.kind
    else {
        return Err(PackedRowBlockRejection::NotReduceKeepReduce);
    };
    if quantized.len() != 2 {
        return Err(PackedRowBlockRejection::OperandCountNotTwo);
    }
    let packed: Vec<(usize, PackedCodec)> = quantized
        .iter()
        .enumerate()
        .filter_map(|(index, codec)| codec.map(|codec| (index, codec)))
        .collect();
    let [(weight, codec)] = packed[..] else {
        return Err(PackedRowBlockRejection::NotExactlyOnePackedOperand);
    };
    // Whitelist the K-quant family explicitly rather than blacklisting one
    // non-K-quant codec by `==` -- an equality check against `Q8_0` alone
    // would have silently admitted `Q4_0` (or any future flat-block codec)
    // the moment its extent happened to be a multiple of 256. This match
    // is exhaustive over `PackedCodec`, so a new codec added later forces a
    // decision here instead of slipping through.
    match codec {
        PackedCodec::Q3K | PackedCodec::Q4K | PackedCodec::Q5K | PackedCodec::Q6K => {}
        PackedCodec::Q2K
        | PackedCodec::Q8_0
        | PackedCodec::Q4_0
        | PackedCodec::Float16
        | PackedCodec::BFloat16 => {
            return Err(PackedRowBlockRejection::NotKQuantCodec);
        }
    }
    let other = 1 - weight;
    let reduce_dims: Vec<u16> = (0..resolved.extents.len() as u16)
        .filter(|dim| !output_axes.contains(dim))
        .collect();
    let Some(&innermost) = reduce_dims.last() else {
        return Err(PackedRowBlockRejection::NotExactlyOneReduceDim { reduce_dims });
    };
    // MULTIPLE reduce dims are only a single logical reduction if they are
    // CONTIGUOUS in memory for BOTH operands: `attn_output`'s own reduce
    // folds three axes (kv-head-group x query-group x head-dim) that are
    // exactly the row-major decomposition of one 4096-wide embedding axis
    // (`docs/discipline.md`'s "print the gate, don't infer it" table: weight
    // strides `[512, 128, 1]` against extents `[8, 4, 128]` — each outer
    // dim's stride equals the product of every dim nested inside it). The
    // row-blocked kernel body walks the flattened `reduction_total` range
    // with ONE stride per operand (`crate::metal::pack_reduce_uniforms`
    // already packs `reduction_total` as the product across every reduce
    // dim, generic in dim count), so folding is sound exactly when this
    // check passes — never a special case for three dims specifically.
    for operand in [weight, other] {
        let layout = &resolved.operands()[operand].1;
        if !axes_fold_contiguously(&reduce_dims, &resolved.extents, layout) {
            return Err(PackedRowBlockRejection::ReduceDimsNotContiguous {
                reduce_dims: reduce_dims.clone(),
            });
        }
    }
    // the packed operand must be contiguous along the INNERMOST (fastest)
    // reduce dim (its super-blocks run along `k`), and the flattened extent
    // across every folded reduce dim must be whole super-blocks.
    let stride = resolved.operands()[weight].1.stride(innermost);
    if stride != 1 {
        return Err(PackedRowBlockRejection::NonUnitWeightStride { stride });
    }
    let extent: u64 = reduce_dims
        .iter()
        .map(|&dim| resolved.extents[dim as usize])
        .product();
    if !(extent as usize).is_multiple_of(Q4K_BLOCK_ELEMENTS) {
        return Err(PackedRowBlockRejection::ExtentNotBlockMultiple { extent });
    }
    let weight_layout = &resolved.operands()[weight].1;
    let other_layout = &resolved.operands()[other].1;
    let (candidate_token_axes, candidate_feature_axes) = split_token_feature_axes(
        output_axes,
        weight_layout,
        other_layout,
        out_layout,
        &resolved.extents,
    )
    .unwrap_or_else(|| (Vec::new(), output_axes.to_vec()));
    let gathered_token_total = candidate_token_axes
        .iter()
        .map(|&axis| resolved.extents[axis as usize])
        .product::<u64>();
    let gathered_multi_row = gather_count(resolved) != 0 && gathered_token_total > 1;
    if gathered_multi_row && !cfg!(feature = "metal-gathered-packed-row") {
        return Err(PackedRowBlockRejection::GatheredOperand);
    }
    // A routed row's weight IS different per token, so it cannot be shared
    // across the multi-row activation group the way the dense path shares
    // it -- but `push_packed_row_multi_row_body` now re-gathers the
    // expert base per token slot instead of hoisting it once, so a gathered
    // op takes the SAME token/feature split a dense op would: `selected`
    // (and `sequence`) as the token axis, `d_out` as the feature axis. Each
    // token still reads its own expert's slab exactly once -- no row is
    // shared, none is re-fetched per output element.
    let (token_axes, feature_axes) = (candidate_token_axes, candidate_feature_axes);
    Ok(PackedRowBlock {
        weight,
        other,
        reduce_dim: innermost as usize,
        codec,
        token_axes,
        feature_axes,
    })
}

pub(super) fn packed_row_block(
    resolved: &BoundOp,
    quantized: &[Option<PackedCodec>],
) -> Option<PackedRowBlock> {
    // The experimental mixed-source path must use the descriptor-aware
    // element reader; row-block pointer hoisting has a separate address ABI.
    if std::env::var_os("PROXIMA_ENABLE_UNSAFE_METAL_EXPERT_SOURCES").is_some() {
        return None;
    }
    classify_packed_row_block(resolved, quantized).ok()
}

/// Public diagnostic seam for `classify_tiled_gemm`, same shape as
/// [`PackedRowBlockRejection`] for `classify_packed_row_block`: one variant
/// per `return`/`None` site in that function, in the order they are checked,
/// so a caller printing `{rejection:?}` sees exactly which condition gave up
/// on a real op instead of an inferred guess. `NotPackedRowBlock` wraps the
/// more basic gate's own rejection when that one fails first -- the tiled
/// path can never be more permissive than the row-blocked path it narrows.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TiledGemmRejection {
    /// The `metal-tiled-gemm` feature is not compiled in -- the tiled path
    /// does not exist as far as this build can observe (see
    /// `classify_tiled_gemm`'s own doc).
    FeatureDisabled,
    /// `classify_packed_row_block` itself rejected first; the tiled path
    /// can only narrow that gate's `Ok`, never rescue its `Err`.
    NotPackedRowBlock(PackedRowBlockRejection),
    /// The packed operand's codec is not [`PackedCodec::Q4K`] -- Q5_K/Q6_K
    /// have no batched-unpack helper for this path yet (see
    /// `classify_tiled_gemm`'s own comment).
    NotQ4K,
    /// `reduce_op`/`init` are not the plain `Add`-from-`Zero` shape
    /// `simdgroup_matrix` accumulation requires.
    NotAddZeroReduce,
    /// `is_plain_product_reduce` is false -- the fused body carries more
    /// than a bare `weight * activation` product.
    NotPlainProductReduce,
    /// Every output axis partitions into a token group (activation-owned)
    /// and a feature group (weight-owned) by nonzero-stride ownership; an
    /// axis neither or both operands depend on, an empty group, or the two
    /// groups interleaving in `output_axes` rather than token-group-then-
    /// feature-group (`native_packed_layout`'s own convention) is a
    /// broadcast/ordering shape this restricted path has never been
    /// measured against.
    AxisOwnershipAmbiguous,
    /// The token or feature group has more than one axis, but they do NOT
    /// nest contiguously (for the owning operand, or for the op's own
    /// output layout) — see `axes_fold_contiguously`.
    AxisGroupNotContiguous,
    /// The token group's flattened extent is below
    /// `crate::sized::TILED_GEMM_MIN_TOKENS` -- tiling overhead is not
    /// amortized at this size.
    TokenExtentBelowMinimum { token_extent: u64, min_tokens: u64 },
}

/// The additional narrowing [`push_tiled_gemm_body`]'s `simdgroup_matrix`
/// path requires on top of [`packed_row_block`]'s own row-blocked
/// eligibility -- the one decision [`grid_threads`] and
/// [`push_cooperative_reduce_body`] must reach IDENTICALLY, same discipline
/// [`PackedRowBlock`] itself follows (see its own doc): this reads a
/// CONCRETE extent (the activation/token axis, against
/// `crate::sized::TILED_GEMM_MIN_TOKENS`) on top of `packed_row_block`'s
/// own concrete-stride gate, so [`kernel_cache_key`] re-derives this too
/// rather than caching by structure alone (`docs/discipline.md` ROW 107).
pub(super) struct TiledGemmBlock {
    // only [`push_tiled_gemm_body`] reads these -- gated the same as that
    // function, so the non-feature build does not carry never-read fields.
    #[cfg(feature = "metal-tiled-gemm")]
    pub(super) weight: usize,
    #[cfg(feature = "metal-tiled-gemm")]
    pub(super) other: usize,
    #[cfg(feature = "metal-tiled-gemm")]
    pub(super) reduce_dim: usize,
    /// every output axis the ACTIVATION owns exclusively (nonzero stride on
    /// `other`, zero on `weight`), outermost first -- more than one only
    /// when [`axes_fold_contiguously`] validated them as one flattened
    /// block, the identical identity [`classify_packed_row_block`]'s own
    /// reduce-dim fold relies on. The tile loop's N side walks the
    /// flattened product of these.
    pub(super) token_axes: Vec<u16>,
    /// every output axis the WEIGHT owns exclusively (nonzero stride on
    /// `weight`, zero on `other`), outermost first -- `attn_q`/`attn_k`/
    /// `attn_v`'s own `heads`/`head_dim` split folds here the same way
    /// `attn_output`'s reduce already folds three axes. The tile loop's M
    /// side walks the flattened product of these.
    pub(super) feature_axes: Vec<u16>,
}

/// `resolved`/`quantized`/`reduce_op`/`init`/`output_axes` are exactly
/// [`push_cooperative_reduce_body`]'s own parameters -- this and
/// [`packed_row_block`] are the two gates that function consults, in order,
/// before falling back to the fully generic cooperative-reduce path.
///
/// Feature-gated: without `metal-tiled-gemm`,
/// [`crate::sized::TILED_GEMM_MIN_TOKENS`] does not exist (see that
/// constant's own doc), so this always returns
/// `Err(TiledGemmRejection::FeatureDisabled)` and every dispatch keeps taking
/// the row-blocked or generic path exactly as it does today — the tiled
/// kernel does not exist as far as the rest of this module can observe.
pub(super) fn classify_tiled_gemm(
    resolved: &BoundOp,
    quantized: &[Option<PackedCodec>],
    reduce_op: ScalarOp,
    init: ReduceInit,
    output_axes: &[u16],
) -> Result<TiledGemmBlock, TiledGemmRejection> {
    #[cfg(not(feature = "metal-tiled-gemm"))]
    {
        let _ = (resolved, quantized, reduce_op, init, output_axes);
        Err(TiledGemmRejection::FeatureDisabled)
    }
    #[cfg(feature = "metal-tiled-gemm")]
    {
        let PackedRowBlock {
            weight,
            other,
            reduce_dim,
            codec,
            ..
        } = classify_packed_row_block(resolved, quantized)
            .map_err(TiledGemmRejection::NotPackedRowBlock)?;
        // Q4_K only -- Q5_K/Q6_K have no batched-unpack helper yet
        // (`push_packed_row_blocked_body`'s own comment on their arms) and,
        // more to the point, have never been measured on this path.
        // Shipping them unmeasured on a correctness-critical GPU kernel
        // would violate the same discipline this landing's own gate
        // demands (principle 18).
        if codec != PackedCodec::Q4K {
            return Err(TiledGemmRejection::NotQ4K);
        }
        // `simdgroup_multiply_accumulate` IS a sum-of-products -- there is
        // no hardware knob for `Maximum`/`Subtract`/etc, so this only ever
        // applies to the exact shape a real matmul takes: an `Add`-reduce
        // over a plain `weight * activation` body, seeded from zero. Every
        // other combination keeps taking the row-blocked or generic path.
        if reduce_op != ScalarOp::Add || init != ReduceInit::Zero {
            return Err(TiledGemmRejection::NotAddZeroReduce);
        }
        if !is_plain_product_reduce(resolved, reduce_op, weight, other) {
            return Err(TiledGemmRejection::NotPlainProductReduce);
        }
        // A plain matmul: every output axis is EITHER token (activation-
        // owned) or feature (weight-owned) -- never both, never neither.
        // `attn_q`/`attn_k`/`attn_v` keep TWO weight-owned axes (`heads` and
        // `head_dim`, split by the einsum but one flat out-features run on
        // disk); folding them the same way `classify_packed_row_block`
        // already folds `attn_output`'s three reduce axes is what lets this
        // path reach them at all (ROW 114 -- ROW 107's "documented scope
        // limit" was this fold, not yet written).
        let weight_layout = &resolved.operands()[weight].1;
        let other_layout = &resolved.operands()[other].1;
        let mut token_axes: Vec<u16> = Vec::new();
        let mut feature_axes: Vec<u16> = Vec::new();
        for &axis in output_axes {
            match (
                weight_layout.stride(axis) == 0,
                other_layout.stride(axis) == 0,
            ) {
                (true, false) => token_axes.push(axis),
                (false, true) => feature_axes.push(axis),
                _ => return Err(TiledGemmRejection::AxisOwnershipAmbiguous),
            }
        }
        if token_axes.is_empty() || feature_axes.is_empty() {
            return Err(TiledGemmRejection::AxisOwnershipAmbiguous);
        }
        // `native_packed_layout`'s own doc: a packed weight's on-disk layout
        // is `[out_dim, in_dim]` row-major, reconstructed by walking
        // `output_axes` so the "out" (feature) side must sit LAST, after
        // every token axis -- checked here as "the two groups reassemble
        // `output_axes` in order", which also catches an interleaved shape
        // (token/feature/token) this path has never been measured against.
        let reassembled: Vec<u16> = token_axes
            .iter()
            .chain(feature_axes.iter())
            .copied()
            .collect();
        if reassembled != output_axes {
            return Err(TiledGemmRejection::AxisOwnershipAmbiguous);
        }
        // A group with more than one axis is only a single logical token/
        // feature dimension if it nests contiguously -- same identity
        // `classify_packed_row_block`'s reduce-dim fold already leans on,
        // checked for the OWNING operand (the other operand's stride is
        // uniformly zero across the group, trivially "contiguous") AND for
        // the op's own output layout, since the tile write-back below also
        // walks the flattened group with one stride.
        let BoundOpKind::Reduce { out_layout, .. } = &resolved.kind else {
            return Err(TiledGemmRejection::NotPackedRowBlock(
                PackedRowBlockRejection::NotReduceKeepReduce,
            ));
        };
        let groups_contiguous =
            axes_fold_contiguously(&token_axes, &resolved.extents, other_layout)
                && axes_fold_contiguously(&feature_axes, &resolved.extents, weight_layout)
                && axes_fold_contiguously(&token_axes, &resolved.extents, out_layout)
                && axes_fold_contiguously(&feature_axes, &resolved.extents, out_layout);
        if !groups_contiguous {
            return Err(TiledGemmRejection::AxisGroupNotContiguous);
        }
        let token_extent: u64 = token_axes
            .iter()
            .map(|&axis| resolved.extents[axis as usize])
            .product();
        if token_extent < crate::sized::TILED_GEMM_MIN_TOKENS {
            return Err(TiledGemmRejection::TokenExtentBelowMinimum {
                token_extent,
                min_tokens: crate::sized::TILED_GEMM_MIN_TOKENS,
            });
        }
        Ok(TiledGemmBlock {
            weight,
            other,
            reduce_dim,
            token_axes,
            feature_axes,
        })
    }
}

pub(super) fn tiled_gemm_block(
    resolved: &BoundOp,
    quantized: &[Option<PackedCodec>],
    reduce_op: ScalarOp,
    init: ReduceInit,
    output_axes: &[u16],
) -> Option<TiledGemmBlock> {
    if std::env::var_os("PROXIMA_ENABLE_UNSAFE_METAL_EXPERT_SOURCES").is_some() {
        return None;
    }
    classify_tiled_gemm(resolved, quantized, reduce_op, init, output_axes).ok()
}

/// Public diagnostic seam: which condition, if any, rejected `resolved` from
/// the tiled-GEMM `simdgroup_matrix` kernel. `Ok(())` means it WOULD take (or
/// does take) the tiled path. Same shape as [`diagnose_packed_row_block`],
/// one narrowing further -- see [`TiledGemmRejection`]'s own doc.
///
/// # Errors
/// Returns the specific [`TiledGemmRejection`] gate that rejected this op.
#[cfg(feature = "instrument")]
pub fn diagnose_tiled_gemm_block(
    resolved: &BoundOp,
    quantized: &[Option<PackedCodec>],
    reduce_op: ScalarOp,
    init: ReduceInit,
    output_axes: &[u16],
) -> Result<(), TiledGemmRejection> {
    classify_tiled_gemm(resolved, quantized, reduce_op, init, output_axes).map(drop)
}

/// Public diagnostic seam: which condition, if any, rejected `resolved`
/// from the row-blocked packed kernel. `Ok(())` means it WOULD take (or
/// does take) the fast path. Behind `instrument` — see
/// [`PackedRowBlockRejection`]'s own doc for why.
///
/// # Errors
/// Returns the specific [`PackedRowBlockRejection`] gate that rejected this
/// op.
#[cfg(feature = "instrument")]
pub fn diagnose_packed_row_block(
    resolved: &BoundOp,
    quantized: &[Option<PackedCodec>],
) -> Result<(), PackedRowBlockRejection> {
    classify_packed_row_block(resolved, quantized).map(drop)
}

/// Thread count [`grid_threads`]' tiled-GEMM arm dispatches -- one
/// `TILED_GEMM_NSG * SIMD_WIDTH`-thread threadgroup per
/// `crate::sized::TILED_GEMM_BLOCK_M x TILED_GEMM_BLOCK_N` output tile,
/// tiling both `feature_extent` and `token_extent`. Only ever called from
/// behind `tiled_gemm_block(..).is_some()` (`grid_threads`' own call site),
/// which is itself only `Some` behind `feature = "metal-tiled-gemm"` (see
/// [`classify_tiled_gemm`]'s doc) -- the `#[cfg(not(..))]` arm is therefore
/// as unreachable as [`push_tiled_gemm_body`]'s own stub, for the same
/// reason.
pub(super) fn tiled_gemm_threadgroups(
    node: NodeId,
    feature_extent: u64,
    token_extent: u64,
) -> Result<u64, EmitError> {
    #[cfg(not(feature = "metal-tiled-gemm"))]
    {
        let _ = (feature_extent, token_extent);
        Err(EmitError::TiledGemmFeatureDisabled { node })
    }
    #[cfg(feature = "metal-tiled-gemm")]
    {
        let _ = node;
        let row_tiles = feature_extent.div_ceil(crate::sized::TILED_GEMM_BLOCK_M);
        let col_tiles = token_extent.div_ceil(crate::sized::TILED_GEMM_BLOCK_N);
        Ok(row_tiles * col_tiles * (TILED_GEMM_NSG as u64) * SIMD_WIDTH)
    }
}

/// Split-K factor for a row-blocked packed matmul with `rows` OUTPUT rows
/// (`output_total`), dispatching `base_simdgroups` simdgroups
/// (`rows.div_ceil(codec.rows_per_simdgroup())`) -- how many simdgroups per
/// threadgroup cooperate on ONE row-group's reduction axis so the
/// dispatch's total simdgroup count reaches
/// [`crate::sized::PACKED_ROW_SPLIT_K_TARGET_SIMDGROUPS`], capped at
/// [`crate::sized::PACKED_ROW_SPLIT_K_MAX_SPLIT`].
///
/// Gated FIRST by [`crate::sized::PACKED_ROW_SPLIT_K_MAX_ROWS`] (`0` means
/// no ceiling): `target_simdgroups` alone lets integer division push a
/// shape's factor toward `1` as `rows` grows, but that fall-off is gradual,
/// not a hard cutoff -- a 4096-row op (attn_q/attn_output/ffn_down in the
/// measured decode-graph table) still clears `factor=2` on
/// `target_simdgroups` arithmetic alone even though its GB/s was already
/// close to the `ffn_*` rate at 1024 base simdgroups, which is exactly the
/// "applied to every packed matvec" loss this ceiling exists to cut off at
/// a build-time-tunable row count instead. Always `1` once
/// `base_simdgroups` already meets the target OR `rows` exceeds the
/// ceiling. Feature-off builds never see this: the caller only reaches
/// here behind `packed_row_block(..).is_some()` AND the `metal-q4k-split-k`
/// feature (see `push_cooperative_reduce_body`'s own gate).
#[cfg(feature = "metal-q4k-split-k")]
pub(super) fn packed_row_split_factor(base_simdgroups: u64, rows: u64) -> u64 {
    if base_simdgroups == 0 {
        return 1;
    }
    let max_rows = crate::sized::PACKED_ROW_SPLIT_K_MAX_ROWS;
    if max_rows != 0 && rows > max_rows {
        return 1;
    }
    (crate::sized::PACKED_ROW_SPLIT_K_TARGET_SIMDGROUPS / base_simdgroups)
        .clamp(1, crate::sized::PACKED_ROW_SPLIT_K_MAX_SPLIT)
}

/// The `metal-q4k-split-k`-off arm: split-K never engages, so the factor is
/// always `1` -- see [`packed_row_split_factor`]'s feature-on twin for the
/// real policy. Kept as a separate function (not a `cfg!()` branch inline)
/// so [`crate::sized::PACKED_ROW_SPLIT_K_TARGET_SIMDGROUPS`] is never
/// referenced from a build that never generated it.
#[cfg(not(feature = "metal-q4k-split-k"))]
pub(super) fn packed_row_split_factor(_base_simdgroups: u64, _rows: u64) -> u64 {
    1
}

/// Single source of truth for the row-blocked packed path's base simdgroup
/// count (one per [`PackedCodec::rows_per_simdgroup`] feature rows, tiled
/// again by `ceil(token_total / crate::sized::PACKED_ROW_ACTIVATION_GROUP)`
/// once more than one activation row folds per streamed weight row) and its
/// derived split-K factor -- both [`grid_threads`] and
/// [`tiled_gemm_threadgroup_width`] need the SAME pair, and
/// [`PackedRowBlock`]'s own doc already names the hazard of two independent
/// call sites silently disagreeing. `token_total == 1` (no distinct token
/// axis, or exactly one activation row) collapses the token factor to `1`,
/// so a caller passing `feature_total` for the whole output and `token_total
/// == 1` gets today's byte-identical single-row dispatch shape.
pub(super) fn packed_row_dispatch(feature_total: u64, token_total: u64, codec: PackedCodec) -> (u64, u64) {
    let base = feature_total.div_ceil(codec.rows_per_simdgroup() as u64);
    let split = packed_row_split_factor(base, feature_total);
    let token_groups = token_total.div_ceil(crate::sized::PACKED_ROW_ACTIVATION_GROUP);
    (base * token_groups, split)
}

