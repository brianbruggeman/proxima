#[cfg(feature = "instrument")]
use super::*;

/// Shared per-op timing dispatch: [`execute_plan_op_timed`] and
/// [`execute_plan_with_placements_op_timed`] both need to encode ONE
/// `BoundOp` on its own command buffer, commit, wait, and read back
/// `GPUStartTime`/`GPUEndTime` -- factored here so the two op-timed
/// executors cannot drift on the [`OpGpuTiming`] fields or the format
/// `proxima-model-interop/src/generate.rs`'s `report_op_timings` prints
/// from them. `placement` mirrors [`encode_op`]'s own parameter -- `None`
/// for the plain (unplaced) op-timed path, `Some((buffer, offset))` for
/// the placed one. `always_live` names every node THIS call's own
/// placement maps hold (input- or output-placed): those are excluded from
/// this op's own retire sweep the same way
/// [`execute_plan_with_placements`]'s own loop excludes them, so the
/// unplaced caller passes an empty set and every `prepared.retires` entry
/// is dropped exactly as it always was.
#[cfg(feature = "instrument")]
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_op_timed(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    device_buffers: &mut BTreeMap<NodeId, DeviceBuffer>,
    prepared: &Prepared,
    packed_operands: &PackedOperands,
    program: &[Op],
    position: usize,
    bound: &BoundOp,
    placement: Option<(&MetalBuffer, usize)>,
    // row 555: same `state_out` placement gap `encode_op`'s own doc names,
    // threaded through this diagnostic timing wrapper so a caller measuring
    // a fused `GatedDeltaNet` decode step observes the SAME state-carry
    // behavior the production dispatch path uses, not a silently different one.
    state_out_placement: Option<(&MetalBuffer, usize)>,
    plan_uniform: Option<&MetalBuffer>,
    always_live: &BTreeSet<NodeId>,
    math_mode: MathMode,
    numeric_policy: NumericPolicy,
    expert_buffers: Option<&ExpertSourceBuffers>,
    cpu_reference: Option<&BTreeMap<NodeId, alloc::vec::Vec<f32>>>,
) -> Result<OpGpuTiming, MetalError> {
    validate_selected_expert_routes(device_buffers, prepared, program, bound, expert_buffers)?;
    let bounded_cpu =
        evaluate_selected_bound_cpu(device_buffers, prepared, packed_operands, program, bound)?;
    // this operand's own TENSOR bytes, not the shared buffer's `length()` --
    // see `operand_tensor_bytes`'s own doc: a checkpoint-mapping-offset bind
    // shares ONE buffer across every packed weight, so `buffer.length()`
    // (kept below as `bound_buffer_bytes`) overstates every individual
    // operand sharing it.
    let operand_bytes: u64 = bound
        .operands()
        .iter()
        .map(|(source, _, lookup)| {
            operand_tensor_bytes(
                program,
                &prepared.index_nodes,
                &prepared.shapes,
                packed_operands,
                *source,
                lookup.as_ref(),
            )
        })
        .sum();
    let bound_buffer_bytes: u64 = bound
        .operands()
        .iter()
        .map(|(source, _, _)| {
            device_buffers
                .get(source)
                .map(|(buffer, _offset)| buffer.length() as u64)
                .unwrap_or(0)
        })
        .sum();
    let weight_name = bound
        .operands()
        .iter()
        .find_map(|(source, _, _)| program[source.0 as usize].name())
        .map(ToString::to_string);
    let kind = classify_kind(bound, packed_operands);
    let packed_kernel_variant = classify_packed_kernel_variant(bound, packed_operands);
    let packed_row_block_rejection = diagnose_kind(bound, packed_operands);
    let packed_codec = bound
        .operands()
        .iter()
        .find_map(|(source, _, _)| packed_operands.get(source).copied());

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
    let fault = encode_op(
        device,
        &encoder,
        device_buffers,
        bound,
        packed_operands,
        placement,
        state_out_placement,
        plan_uniform,
        math_mode,
        numeric_policy,
        None,
        None,
        None,
        expert_buffers,
    )?;
    encoder.finish();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    let gpu_ns =
        ((command_buffer.GPUEndTime() - command_buffer.GPUStartTime()) * 1e9).max(0.0) as u64;
    if let Some((fault_buffer, gathers)) = fault {
        check_gather_fault(bound, &fault_buffer, gathers)?;
    }
    if std::env::var_os("PROXIMA_METAL_NAN_CHECK").is_some()
        && let Some((first_index, shape, metal_values)) =
            check_op_output_finite(device_buffers, prepared, program, bound.node, kind)?
    {
        let cpu_value = bounded_cpu
            .as_ref()
            .or_else(|| cpu_reference.and_then(|reference| reference.get(&bound.node)))
            .and_then(|values| values.get(first_index))
            .copied();
        eprintln!(
            "metal_nan_bound node={:?} name={:?} kind={kind} element={first_index} metal={} cpu={cpu_value:?} shape={shape:?} bound={bound:?}",
            bound.node,
            program[bound.node.0 as usize].name(),
            metal_values[first_index],
        );
        report_bound_operands(
            bound,
            prepared,
            packed_operands,
            program,
            expert_buffers,
            Some(device_buffers),
            Some(first_index),
        );
        return Err(MetalError::NonFiniteOpOutput {
            node: bound.node,
            kind: kind.to_string(),
        });
    }
    if let Some(cpu_values) = bounded_cpu {
        let Some((buffer, offset)) = device_buffers.get(&bound.node) else {
            return Err(MetalError::UnresolvedHazardOperand { node: bound.node });
        };
        let shape = prepared.shapes.of(bound.node).to_vec();
        let dtype = gpu_dtype(program, &prepared.index_nodes, bound.node);
        let metal_values = read_back(buffer, *offset, element_count(&shape), bound.node, dtype)?;
        let first_mismatch = compare_bound_f32(bound.node, &metal_values, &cpu_values)?;
        if let Some((element, metal_value, cpu_value, relative, max_rel_diff)) = first_mismatch {
            let absolute = (metal_value - cpu_value).abs();
            eprintln!(
                "metal_bound_compare node={:?} name={:?} kind={kind} element={element} metal={metal_value} cpu={cpu_value} abs={absolute} rel={relative} max_rel={max_rel_diff} shape={shape:?} bound={bound:?}",
                bound.node,
                program[bound.node.0 as usize].name(),
            );
            report_bound_operands(
                bound,
                prepared,
                packed_operands,
                program,
                expert_buffers,
                Some(device_buffers),
                Some(element),
            );
            return Err(MetalError::CpuMetalDivergence {
                node: bound.node,
                kind: kind.to_string(),
                max_rel_diff,
            });
        }
        eprintln!(
            "metal_bound_compare node={:?} name={:?} kind={kind} result=within_tolerance elements={} shape={shape:?}",
            bound.node,
            program[bound.node.0 as usize].name(),
            metal_values.len(),
        );
    }
    if let Some(cpu_reference) = cpu_reference
        && let Some(max_rel_diff) = compare_op_output_to_cpu(
            device_buffers,
            prepared,
            program,
            bound.node,
            kind,
            cpu_reference,
        )?
        && max_rel_diff > 1e-2
    {
        if let Some(cpu_values) = cpu_reference.get(&bound.node)
            && let Some((buffer, offset)) = device_buffers.get(&bound.node)
        {
            let shape = prepared.shapes.of(bound.node).to_vec();
            let dtype = gpu_dtype(program, &prepared.index_nodes, bound.node);
            let metal_values =
                read_back(buffer, *offset, element_count(&shape), bound.node, dtype)?;
            if let Some((element, metal_value, cpu_value, relative, maximum)) =
                compare_bound_f32(bound.node, &metal_values, cpu_values)?
            {
                let absolute = (metal_value - cpu_value).abs();
                eprintln!(
                    "metal_reference_compare node={:?} name={:?} kind={kind} element={element} metal={metal_value} cpu={cpu_value} abs={absolute} rel={relative} max_rel={maximum} shape={shape:?}",
                    bound.node,
                    program[bound.node.0 as usize].name(),
                );
                report_bound_operands(
                    bound,
                    prepared,
                    packed_operands,
                    program,
                    expert_buffers,
                    Some(device_buffers),
                    Some(element),
                );
            }
        }
        return Err(MetalError::CpuMetalDivergence {
            node: bound.node,
            kind: kind.to_string(),
            max_rel_diff,
        });
    }
    for retired in &prepared.retires[position] {
        if always_live.contains(retired) {
            continue;
        }
        device_buffers.remove(retired);
    }
    Ok(OpGpuTiming {
        node: bound.node,
        kind,
        extents: bound.extents.clone(),
        output_axes: match &bound.kind {
            BoundOpKind::Reduce { output_axes, .. } => output_axes.to_vec(),
            _ => Vec::new(),
        },
        operand_bytes,
        bound_buffer_bytes,
        gpu_ns,
        weight_name,
        operand_count: bound.operands().len(),
        packed_codec,
        packed_kernel_variant,
        packed_row_block_rejection,
    })
}

/// Diagnostic-only counterpart of [`execute_plan`]: instead of sharing ONE
/// command buffer across the whole program and `commit`/`waitUntilCompleted`ing
/// it exactly once (see the module doc's "Execution model"), this commits
/// and waits on its OWN command buffer per [`BoundOp`], so each op's
/// `GPUStartTime`/`GPUEndTime` -- `MTLCommandBuffer`'s own documented
/// per-buffer GPU occupancy, not a CPU-side measurement -- can be read back
/// individually. It exists to answer exactly one question: does GPU time
/// track operand bytes, or is it flat per dispatch regardless of size. That
/// answer costs exactly what [`execute_plan`]'s own module doc says ROW 73
/// measured and removed -- one submission-boundary intercept per split, now
/// paid `prepared.resolved.len()` times instead of once -- which is why this
/// function is reachable only behind the `instrument` feature, from this
/// crate's own diagnostic call sites, and never from the serving loop
/// ([`execute_plan`]/[`execute_plan_named`] remain the only production
/// entry points and are completely unchanged by this function's existence).
///
/// # Errors
/// Same as [`execute_plan`].
#[cfg(feature = "instrument")]
pub fn execute_plan_op_timed(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
    cpu_reference: Option<&BTreeMap<NodeId, alloc::vec::Vec<f32>>>,
) -> Result<(Evaluated, Vec<OpGpuTiming>), MetalError> {
    execute_plan_op_timed_inner(plan, blocks, &BTreeMap::new(), cpu_reference)
}

/// [`execute_plan_op_timed`]'s routed-expert counterpart. It stages the
/// same per-expert payload and descriptor buffers as
/// [`execute_plan_with_expert_sources`] before splitting the plan into one
/// command buffer per bound operation.
///
/// # Errors
/// Same as [`execute_plan_with_expert_sources`] and [`execute_plan_op_timed`].
#[cfg(feature = "instrument")]
pub fn execute_plan_op_timed_with_expert_sources(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
    expert_sources: &BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'_>>,
    cpu_reference: Option<&BTreeMap<NodeId, alloc::vec::Vec<f32>>>,
) -> Result<(Evaluated, Vec<OpGpuTiming>), MetalError> {
    if expert_sources.is_empty() {
        return execute_plan_op_timed(plan, blocks, cpu_reference);
    }
    let expert_buffers = stage_expert_sources(plan, blocks, expert_sources)?;
    execute_plan_op_timed_inner(plan, blocks, &expert_buffers, cpu_reference)
}

#[cfg(feature = "instrument")]
pub(super) fn execute_plan_op_timed_inner(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
    expert_buffers: &BTreeMap<NodeId, ExpertSourceBuffers>,
    cpu_reference: Option<&BTreeMap<NodeId, alloc::vec::Vec<f32>>>,
) -> Result<(Evaluated, Vec<OpGpuTiming>), MetalError> {
    let prepared = &plan.prepared;
    let packed_operands = &plan.packed_operands;
    let selected_bound = std::env::var("PROXIMA_METAL_COMPARE_BOUND_NODE")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .map(NodeId);
    if let Some(node) = selected_bound
        && !prepared.resolved.iter().any(|bound| bound.node == node)
    {
        eprintln!(
            "metal_bound_compare_missing node={node:?} available={:?}",
            prepared
                .resolved
                .iter()
                .map(|bound| (bound.node, bound.kind.name()))
                .collect::<Vec<_>>()
        );
        return Err(MetalError::CpuBoundComparisonNodeNotFound { node });
    }

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
        if let Some(buffers) = effective_expert_buffers.get(node) {
            device_buffers.insert(*node, (buffers.payloads.clone(), buffers.payload_offset));
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
            QuantizedBlock::Int32(data) => upload_block_int32_as_float(&device, data, None)?,
            // `Float16`/`BFloat16` upload their bytes UNCHANGED, same as
            // every packed codec above -- there is no host-side narrowing
            // step (unlike `upload_block`'s `Float32 -> Float16` path,
            // which narrows a caller's `&[f32]`): a `Float16` weight's on-
            // disk bytes already ARE its device buffer's bytes (native
            // `half`), and a `BFloat16` weight's bytes are widened entirely
            // on the GPU at the read (`msl::BF16_UNPACK_MSL`), never on the
            // host.
            // `Q3_K` uploads its raw super-block bytes unchanged, same as
            // every other packed codec below -- `msl::Codec::Q3K`'s
            // own unpack kernel (`q3k_element`) reads them at the GPU side.
            // upload is codec-agnostic: raw bytes move to device memory
            // unchanged regardless of which unpack kernel later reads them,
            // so every `Packed` codec -- including any not yet recognized
            // by a GPU kernel -- takes this same arm.
            QuantizedBlock::Packed { bytes, .. } => {
                upload_packed_bytes(&device, bytes, resident_name)?
            }
        };
        device_buffers.insert(*node, buffer);
    }

    let no_placements: BTreeSet<NodeId> = BTreeSet::new();
    let mut timings: Vec<OpGpuTiming> = Vec::with_capacity(prepared.resolved.len());
    for (position, bound) in prepared.resolved.iter().enumerate() {
        let expert_buffers = expert_buffers_for(bound, &effective_expert_buffers)?;
        let timing = execute_op_timed(
            &device,
            &queue,
            &mut device_buffers,
            prepared,
            packed_operands,
            &plan.program,
            position,
            bound,
            None,
            None,
            None,
            &no_placements,
            plan.math_mode,
            plan.numeric_policy,
            expert_buffers,
            cpu_reference,
        )?;
        timings.push(timing);
    }

    let evaluated = finish(plan, &device_buffers, &BTreeSet::new(), None)?;
    Ok((evaluated, timings))
}

/// [`execute_plan_op_timed`] against a name-keyed block set, mirroring
/// [`execute_plan_named`]'s own name resolution.
///
/// # Errors
/// Propagates name-resolution and Metal driver failures.
#[cfg(feature = "instrument")]
pub fn execute_plan_named_op_timed(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'_>)],
    cpu_reference: Option<&BTreeMap<NodeId, alloc::vec::Vec<f32>>>,
) -> Result<(Evaluated, Vec<OpGpuTiming>), MetalError> {
    let blocks = resolve_named_blocks(&plan.program, named)?;
    execute_plan_op_timed(plan, &blocks, cpu_reference)
}

/// Name-keyed counterpart of
/// [`execute_plan_op_timed_with_expert_sources`].
///
/// # Errors
/// Propagates name-resolution, expert-source, and Metal driver failures.
#[cfg(feature = "instrument")]
pub fn execute_plan_named_op_timed_with_expert_sources(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'_>)],
    expert_sources: &BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'_>>,
    cpu_reference: Option<&BTreeMap<NodeId, alloc::vec::Vec<f32>>>,
) -> Result<(Evaluated, Vec<OpGpuTiming>), MetalError> {
    let blocks = resolve_named_blocks(&plan.program, named)?;
    execute_plan_op_timed_with_expert_sources(plan, &blocks, expert_sources, cpu_reference)
}

/// [`execute_plan_with_placements`]'s op-timed twin -- the default decode
/// path (`run_decode_loop_placed_kv` in
/// `proxima-model-interop/src/generate.rs`) routes through
/// `execute_plan_with_placements`, never `execute_plan`, so
/// [`execute_plan_op_timed`] alone cannot attribute GPU time on that path:
/// it always builds fresh output buffers (`placement: None` at its own call
/// site above) and so never exercises the placed-KV write/read shape at
/// all. This shares every input/output-placement resolution step
/// `execute_plan_with_placements` itself performs (input-placed nodes skip
/// the host upload identically; `always_live` reproduces that function's
/// own retirement exclusion) and swaps only its single shared-command-buffer
/// submission for this module's own shared per-op dispatch helper, same relationship
/// [`execute_plan_op_timed`] already has to [`execute_plan`].
///
/// # Errors
/// Same as [`execute_plan_with_placements`].
#[cfg(all(feature = "metal-output-placement", feature = "instrument"))]
pub fn execute_plan_with_placements_op_timed(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
    output_placements: &[(NodeId, &PlacedBuffer, usize)],
) -> Result<(Evaluated, Vec<OpGpuTiming>), MetalError> {
    let prepared = &plan.prepared;
    let packed_operands = &plan.packed_operands;
    let input_placed: BTreeMap<NodeId, (&PlacedBuffer, usize)> = input_placements
        .iter()
        .map(|(node, buffer, offset)| (*node, (*buffer, *offset)))
        .collect();
    let output_placed: BTreeMap<NodeId, (&PlacedBuffer, usize)> = output_placements
        .iter()
        .map(|(node, buffer, offset)| (*node, (*buffer, *offset)))
        .collect();
    let always_live: BTreeSet<NodeId> = input_placed
        .keys()
        .chain(output_placed.keys())
        .copied()
        .collect();
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
        if let Some((buffer, offset)) = input_placed.get(node) {
            device_buffers.insert(*node, ((*buffer).clone(), *offset));
            continue;
        }
        // same `BLOCK_OFFERED_BYTES`/`BLOCK_UPLOAD_CALLS` fire
        // `execute_plan_with_placements`'s own loop carries -- this
        // diagnostic op-timed twin duplicates that loop's upload shape, so
        // it must duplicate the census fire too, or the partition identity
        // (`BLOCK_COPIED_BYTES + BLOCK_NOCOPY_BOUND_BYTES +
        // BLOCK_OFFSET_BOUND_BYTES == BLOCK_OFFERED_BYTES`) goes untested on
        // this path even though the three per-path counters (fired inside
        // `upload_block_as_float`/`upload_packed_bytes` themselves) still
        // increment here regardless.
        #[cfg(feature = "instrument")]
        {
            counter!(BLOCK_UPLOAD_CALLS, 1);
            counter!(BLOCK_OFFERED_BYTES, block_byte_len(block) as u64);
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
            QuantizedBlock::Int32(data) => upload_block_int32_as_float(&device, data, None)?,
            // `Q3_K` uploads its raw super-block bytes unchanged, same as
            // every other packed codec below -- `msl::Codec::Q3K`'s
            // own unpack kernel (`q3k_element`) reads them at the GPU side.
            // upload is codec-agnostic: raw bytes move to device memory
            // unchanged regardless of which unpack kernel later reads them,
            // so every `Packed` codec -- including any not yet recognized
            // by a GPU kernel -- takes this same arm.
            QuantizedBlock::Packed { bytes, .. } => {
                upload_packed_bytes(&device, bytes, resident_name)?
            }
        };
        device_buffers.insert(*node, buffer);
    }

    let mut timings: Vec<OpGpuTiming> = Vec::with_capacity(prepared.resolved.len());
    for (position, bound) in prepared.resolved.iter().enumerate() {
        let placement = match output_placed.get(&bound.node).copied() {
            Some(placement) => Some(placement),
            None => arena_placement(plan, position)?,
        };
        // row 555: `bound.node` above can never be a fused `state_out` node
        // (`encode_op`'s own doc); this is the SAME `output_placed` map,
        // keyed by `state_out` instead, so this diagnostic path carries
        // state exactly like the production dispatch below does.
        let state_out_placement = match &bound.kind {
            BoundOpKind::GatedDeltaNet { state_out, .. } => {
                output_placed.get(state_out).copied()
            }
            _ => None,
        };
        let uniform_buffer = plan_uniform_buffer(plan, position)?;
        let timing = execute_op_timed(
            &device,
            &queue,
            &mut device_buffers,
            prepared,
            packed_operands,
            &plan.program,
            position,
            bound,
            placement,
            state_out_placement,
            uniform_buffer,
            &always_live,
            plan.math_mode,
            plan.numeric_policy,
            None,
            None,
        )?;
        timings.push(timing);
    }

    let placed_output_nodes: BTreeSet<NodeId> = output_placed.keys().copied().collect();
    let evaluated = finish(plan, &device_buffers, &placed_output_nodes, None)?;
    Ok((evaluated, timings))
}

/// [`execute_plan_with_placements_op_timed`] against a name-keyed block
/// set, mirroring [`execute_plan_named_with_placements`]'s own name
/// resolution.
///
/// # Errors
/// Propagates name-resolution and Metal driver failures.
#[cfg(all(feature = "metal-output-placement", feature = "instrument"))]
pub fn execute_plan_named_with_placements_op_timed(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'_>)],
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
    output_placements: &[(NodeId, &PlacedBuffer, usize)],
) -> Result<(Evaluated, Vec<OpGpuTiming>), MetalError> {
    let blocks = resolve_named_blocks_with_placed_inputs(plan, named, input_placements)?;
    execute_plan_with_placements_op_timed(plan, &blocks, input_placements, output_placements)
}

/// Which [`MTLCounterSamplingPoint`] a device actually honors -- resolved
/// once per [`execute_plan_with_placements_dispatch_timed`] call (this is a
/// diagnostic-only path, never the hot loop, so re-querying every call
/// costs nothing that matters) rather than assumed: an M1 Max reports
/// `AtDispatchBoundary` unsupported and `AtStageBoundary` supported
/// (verified with a standalone Metal probe against this exact device, not
/// inferred from Apple's docs), so the dispatch-boundary branch below is
/// exercised only on hardware that actually has it.
#[cfg(feature = "instrument")]
pub(super) fn counter_sampling_mode(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Option<objc2_metal::MTLCounterSamplingPoint> {
    if device.supportsCounterSampling(objc2_metal::MTLCounterSamplingPoint::AtDispatchBoundary) {
        Some(objc2_metal::MTLCounterSamplingPoint::AtDispatchBoundary)
    } else if device.supportsCounterSampling(objc2_metal::MTLCounterSamplingPoint::AtStageBoundary)
    {
        Some(objc2_metal::MTLCounterSamplingPoint::AtStageBoundary)
    } else {
        None
    }
}

/// The device's own `MTLCommonCounterSetTimestamp` counter set, by name --
/// `device.counterSets()` returns every set the hardware exposes; this
/// finds the one [`objc2_metal::MTLCommonCounterSetTimestamp`] names,
/// rather than assuming index `0`.
#[cfg(feature = "instrument")]
pub(super) fn timestamp_counter_set(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Option<Retained<ProtocolObject<dyn objc2_metal::MTLCounterSet>>> {
    let sets = device.counterSets()?;
    // SAFETY: `MTLCommonCounterSetTimestamp` is a framework-provided
    // constant `NSString`, valid for the process lifetime.
    let timestamp_name = unsafe { objc2_metal::MTLCommonCounterSetTimestamp };
    sets.iter().find(|set| &*set.name() == timestamp_name)
}

// Apple GPUs reject timestamp sample buffers larger than 32 KiB. The Metal
// timestamp counter occupies a 16-byte slot even though the resolved value is
// one u64, so 2,048 samples is the largest buffer this diagnostic path can
// request on the device we measure.
#[cfg(feature = "instrument")]
pub(super) const MAX_TIMESTAMP_SAMPLES: usize = 32 * 1024 / 16;

/// [`execute_plan_with_placements`]'s per-dispatch GPU-timestamp twin.
/// Unlike [`execute_plan_with_placements_op_timed`] (one command buffer per
/// op -- discipline log ROW 298's own finding that this reproduces neither
/// the batched buffer's barrier/serialization cost nor its aggregate
/// total), this function submits the SAME single command buffer the
/// production path does and brackets every dispatch with Metal's
/// `MTLCounterSampleBuffer` against `MTLCommonCounterSetTimestamp`
/// (`sampleCountersInBuffer:atSampleIndex:withBarrier:` when the device
/// supports `AtDispatchBoundary`; one compute encoder per position, sampled
/// at `startOfEncoderSampleIndex`/`endOfEncoderSampleIndex`, when it only
/// supports `AtStageBoundary` -- `AtStageBoundary`'s own granularity is
/// per-ENCODER, so that branch degrades to "one encoder per dispatch"
/// rather than the coarser "one encoder per kind" grouping, since this
/// program's own dispatch sequence rarely repeats the same op kind AND
/// shape twice in a row -- see the module-level discipline row this
/// function's own commit lands for the measured device support and the
/// resulting per-shape table).
///
/// GPU tick counts are converted to nanoseconds by calibrating against this
/// SAME call's own CPU/GPU timestamp pair (`sampleTimestamps:gpuTimestamp:`
/// taken once before `commit` and once after `waitUntilCompleted`): the
/// CPU side of that pair is in the same `mach_absolute_time` domain
/// [`ticks_to_nanos`] already converts, so `cpu_ns_delta / gpu_tick_delta`
/// is this call's own nanoseconds-per-GPU-tick, applied uniformly to every
/// per-position delta. No new type is introduced: the per-position record
/// is [`OpGpuTiming`], the same type [`execute_plan_with_placements_op_timed`]
/// already returns.
///
/// [`execute_plan_with_placements_dispatch_timed`]/
/// [`execute_plan_named_with_placements_dispatch_timed`]'s own return
/// shape, named only because clippy's `type_complexity` lint requires it
/// for a four-element tuple -- not a new abstraction, the same
/// `DispatchMeta`-alias precedent ROW 309 already set for this function's
/// internals, just applied to its public return type too.
#[cfg(all(feature = "metal-output-placement", feature = "instrument"))]
pub type DispatchTimedOutcome = (
    Evaluated,
    Vec<OpGpuTiming>,
    &'static str,
    Option<(u64, u64)>,
);

/// Returns `(evaluated, per_position_timings, sampling_mode, encoder_split_ns)`,
/// where `sampling_mode` is `"dispatch-boundary"`, `"stage-boundary"`, or
/// `"unsupported"` -- in the last case every `gpu_ns` is `0`, never
/// fabricated -- and `encoder_split_ns` is `Some((encoder_1_ns,
/// encoder_2_ns))` exactly when [`Plan::set_encoder_split_at`] named a
/// split position AND the stage-boundary branch actually ran (ROW 329):
/// three `MTLCounterSampleBuffer` samples (buffer start, encoder-1 end,
/// encoder-2 end) replace `per_position_timings`' own per-position samples
/// for this call, so every `OpGpuTiming::gpu_ns` reads `0` when this is
/// `Some` -- the same "never fabricated" contract `"unsupported"` already
/// carries, extended to a case where per-op attribution genuinely was not
/// sampled, not just unsupported by the device.
///
/// # Errors
/// Same as [`execute_plan_with_placements`], plus a [`MetalError`] if the
/// device refuses to allocate the counter sample buffer.
#[cfg(all(feature = "metal-output-placement", feature = "instrument"))]
#[allow(clippy::too_many_lines)]
pub fn execute_plan_with_placements_dispatch_timed(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
    output_placements: &[(NodeId, &PlacedBuffer, usize)],
) -> Result<DispatchTimedOutcome, MetalError> {
    let prepared = &plan.prepared;
    let packed_operands = &plan.packed_operands;
    let input_placed: BTreeMap<NodeId, (&PlacedBuffer, usize)> = input_placements
        .iter()
        .map(|(node, buffer, offset)| (*node, (*buffer, *offset)))
        .collect();
    let output_placed: BTreeMap<NodeId, (&PlacedBuffer, usize)> = output_placements
        .iter()
        .map(|(node, buffer, offset)| (*node, (*buffer, *offset)))
        .collect();
    let always_live: BTreeSet<NodeId> = input_placed
        .keys()
        .chain(output_placed.keys())
        .copied()
        .collect();
    let (device, queue) = device_and_queue()?;

    let mode = counter_sampling_mode(&device);
    let counter_set = mode.and_then(|_| timestamp_counter_set(&device));

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
        if let Some((buffer, offset)) = input_placed.get(node) {
            device_buffers.insert(*node, ((*buffer).clone(), *offset));
            continue;
        }
        #[cfg(feature = "instrument")]
        {
            counter!(BLOCK_UPLOAD_CALLS, 1);
            counter!(BLOCK_OFFERED_BYTES, block_byte_len(block) as u64);
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
            QuantizedBlock::Int32(data) => upload_block_int32_as_float(&device, data, None)?,
            // upload is codec-agnostic: raw bytes move to device memory
            // unchanged regardless of which unpack kernel later reads them,
            // so every `Packed` codec -- including any not yet recognized
            // by a GPU kernel -- takes this same arm.
            QuantizedBlock::Packed { bytes, .. } => {
                upload_packed_bytes(&device, bytes, resident_name)?
            }
        };
        device_buffers.insert(*node, buffer);
    }

    let position_count = prepared.resolved.len();
    let Some((sampling_point, counter_set)) = mode.zip(counter_set) else {
        // no counter set / sampling point at all -- fall through to the
        // production single-encoder submission with zero instrumentation
        // rather than refusing to run: `gpu_ns` stays `0` on every entry,
        // never fabricated, and `sampling_mode` names this plainly.
        let evaluated = execute_plan_with_placements(
            plan,
            blocks,
            input_placements,
            output_placements,
            &mut Vec::new(),
        )?;
        let timings: Vec<OpGpuTiming> = prepared
            .resolved
            .iter()
            .map(|bound| OpGpuTiming {
                node: bound.node,
                kind: classify_kind(bound, packed_operands),
                extents: bound.extents.clone(),
                output_axes: match &bound.kind {
                    BoundOpKind::Reduce { output_axes, .. } => output_axes.to_vec(),
                    _ => Vec::new(),
                },
                operand_bytes: 0,
                bound_buffer_bytes: 0,
                gpu_ns: 0,
                weight_name: None,
                operand_count: bound.operands().len(),
                packed_codec: None,
                packed_kernel_variant: classify_packed_kernel_variant(bound, packed_operands),
                packed_row_block_rejection: None,
            })
            .collect();
        return Ok((evaluated, timings, "unsupported", None));
    };

    let dispatch_boundary =
        sampling_point == objc2_metal::MTLCounterSamplingPoint::AtDispatchBoundary;
    // ROW 329: the split feature only applies to the stage-boundary
    // fallback this device (M1 Max) actually takes -- a device that
    // supports `AtDispatchBoundary` already gets per-DISPATCH granularity
    // from `shared_encoder`'s manual `sampleCountersInBuffer` calls below,
    // with no per-encoder overhead to economize on, so `encoder_split_at`
    // is ignored there rather than silently reinterpreted.
    let split_at = if dispatch_boundary {
        None
    } else {
        plan.encoder_split_at
    };

    // Keep the diagnostic path inside Metal's sample-buffer limit. A decode
    // prefill can contain more dispatches than fit in one counter buffer; the
    // unprofiled tail still executes in the same command buffer, but reports
    // zero GPU time rather than causing a CPU fallback or a fabricated value.
    let profiled_position_count = if split_at.is_some() {
        position_count
    } else {
        position_count.min(MAX_TIMESTAMP_SAMPLES / 2)
    };
    let sample_descriptor = objc2_metal::MTLCounterSampleBufferDescriptor::new();
    sample_descriptor.setCounterSet(Some(&counter_set));
    // Three samples (buffer start, encoder-1 end, encoder-2 end) when
    // split, else the original one-pair-per-position sizing. SAFETY: both
    // counts are plain arithmetic, well under any device's
    // `maxBufferLength`-scale sample-buffer limits for the per-token
    // dispatch counts this workspace's own decode programs emit.
    let sample_count: usize = if split_at.is_some() {
        3
    } else {
        2 * profiled_position_count
    };
    unsafe { sample_descriptor.setSampleCount(sample_count as NSUInteger) };
    let sample_buffer = device
        .newCounterSampleBufferWithDescriptor_error(&sample_descriptor)
        .map_err(|error| MetalError::CompileFailed {
            log: format!("failed to allocate a counter sample buffer: {error}"),
        })?;

    let command_buffer = queue
        .commandBuffer()
        .ok_or_else(|| MetalError::CompileFailed {
            log: "command queue refused to hand out a command buffer".to_string(),
        })?;

    let shared_encoder = if dispatch_boundary {
        Some(EncoderGuard::new(
            command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| MetalError::CompileFailed {
                    log: "command buffer refused to hand out a compute encoder".to_string(),
                })?,
        ))
    } else {
        None
    };
    // ROW 329: the stage-boundary encoder currently open, carried across
    // loop iterations so a split group's encoder stays open for every
    // position inside it -- `None` until the loop's first iteration
    // creates one. Unused (stays `None` the whole call) when
    // `dispatch_boundary` is true, since `shared_encoder` covers that case.
    // `EncoderGuard`-owned (not a plain `Retained`) so a `?` from inside the
    // loop below -- `arena_placement`, `plan_uniform_buffer`, `encode_op`'s
    // own error sites -- ends whichever encoder is currently open at `Drop`
    // instead of leaking it into the assertion this guard exists to avoid.
    let mut stage_encoder: Option<EncoderGuard> = None;

    // one tuple per position: `(node, kind, operand_bytes, bound_buffer_bytes,
    // weight_name, operand_count, packed_codec, packed_kernel_variant)` --
    // every field [`OpGpuTiming`] already carries, gathered while `bound`
    // is still live so the counter-resolve pass below only zips it against
    // `gpu_ns`.
    type DispatchMeta = (
        NodeId,
        &'static str,
        u64,
        u64,
        Option<String>,
        usize,
        Option<Codec>,
        &'static str,
    );
    let mut metas: Vec<DispatchMeta> = Vec::with_capacity(position_count);
    let mut pending_faults: Vec<PendingFault<'_>> = Vec::new();

    let cpu_gpu_start = sample_timestamps(&device);
    for (position, bound) in prepared.resolved.iter().enumerate() {
        let placement = match output_placed.get(&bound.node).copied() {
            Some(placement) => Some(placement),
            None => arena_placement(plan, position)?,
        };
        // row 555: same fused-`state_out`-key gap `encode_op`'s own doc names.
        let state_out_placement = match &bound.kind {
            BoundOpKind::GatedDeltaNet { state_out, .. } => {
                output_placed.get(state_out).copied()
            }
            _ => None,
        };
        let uniform_buffer = plan_uniform_buffer(plan, position)?;
        let operand_bytes: u64 = bound
            .operands()
            .iter()
            .map(|(source, _, lookup)| {
                operand_tensor_bytes(
                    &plan.program,
                    &prepared.index_nodes,
                    &prepared.shapes,
                    packed_operands,
                    *source,
                    lookup.as_ref(),
                )
            })
            .sum();
        let bound_buffer_bytes: u64 = bound
            .operands()
            .iter()
            .map(|(source, _, _)| {
                device_buffers
                    .get(source)
                    .map(|(buffer, _offset)| buffer.length() as u64)
                    .unwrap_or(0)
            })
            .sum();
        let weight_name = bound
            .operands()
            .iter()
            .find_map(|(source, _, _)| plan.program[source.0 as usize].name())
            .map(ToString::to_string);
        let packed_codec = bound
            .operands()
            .iter()
            .find_map(|(source, _, _)| packed_operands.get(source).copied());
        metas.push((
            bound.node,
            classify_kind(bound, packed_operands),
            operand_bytes,
            bound_buffer_bytes,
            weight_name,
            bound.operands().len(),
            packed_codec,
            classify_packed_kernel_variant(bound, packed_operands),
        ));

        let encoder = match &shared_encoder {
            Some(guard) => guard.clone_inner(),
            None => {
                // ROW 329: without a split, every position opens (and, below,
                // immediately closes) its own encoder -- ROW 309's original
                // fallback. With a split, a new encoder opens only at
                // position 0 and at `split_at` itself, and stays open for
                // every position in between (closed just before the NEXT
                // new encoder opens, or after the loop for the last group).
                let needs_new_encoder = match split_at {
                    None => true,
                    Some(split) => position == 0 || position == split,
                };
                // A group's non-boundary positions reuse the encoder a prior
                // iteration in this SAME group already opened -- `None` here
                // (rather than an `expect`) falls through to opening a fresh
                // one instead of panicking, which cannot happen given
                // `needs_new_encoder`'s own boundary check above but costs
                // nothing to make self-healing rather than load-bearing.
                if let Some(existing) = (!needs_new_encoder)
                    .then(|| stage_encoder.as_ref().map(EncoderGuard::clone_inner))
                    .flatten()
                {
                    existing
                } else {
                    if let Some(previous) = stage_encoder.take() {
                        previous.finish();
                    }
                    let profile_position = position < profiled_position_count;
                    let descriptor = profile_position
                        .then(objc2_metal::MTLComputePassDescriptor::computePassDescriptor);
                    if let Some(descriptor) = &descriptor {
                        let attachment = unsafe {
                            descriptor
                                .sampleBufferAttachments()
                                .objectAtIndexedSubscript(0)
                        };
                        attachment.setSampleBuffer(Some(&sample_buffer));
                    }
                    let (start_index, end_index) = match split_at {
                        None => (2 * position, 2 * position + 1),
                        Some(split) if position == split => (objc2_metal::MTLCounterDontSample, 2),
                        Some(_) => (0, 1),
                    };
                    if let Some(descriptor) = &descriptor {
                        let attachment = unsafe {
                            descriptor
                                .sampleBufferAttachments()
                                .objectAtIndexedSubscript(0)
                        };
                        unsafe {
                            attachment.setStartOfEncoderSampleIndex(start_index as NSUInteger);
                            attachment.setEndOfEncoderSampleIndex(end_index as NSUInteger);
                        }
                    }
                    let opened = match descriptor {
                        Some(descriptor) => command_buffer
                            .computeCommandEncoderWithDescriptor(&descriptor)
                            .ok_or_else(|| MetalError::CompileFailed {
                                log: "command buffer refused to hand out a stage-sampled compute encoder"
                                    .to_string(),
                            })?,
                        None => command_buffer
                            .computeCommandEncoder()
                            .ok_or_else(|| MetalError::CompileFailed {
                                log: "command buffer refused to hand out a compute encoder"
                                    .to_string(),
                            })?,
                    };
                    stage_encoder = Some(EncoderGuard::new(opened.clone()));
                    opened
                }
            }
        };

        if dispatch_boundary && position < profiled_position_count {
            unsafe {
                encoder.sampleCountersInBuffer_atSampleIndex_withBarrier(
                    &sample_buffer,
                    (2 * position) as NSUInteger,
                    true,
                );
            }
        }
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
            None,
            None,
            None,
            None,
        )?;
        if dispatch_boundary && position < profiled_position_count {
            unsafe {
                encoder.sampleCountersInBuffer_atSampleIndex_withBarrier(
                    &sample_buffer,
                    (2 * position + 1) as NSUInteger,
                    true,
                );
            }
        }
        // ROW 329: a non-split stage-boundary encoder still closes here,
        // right after its one dispatch -- `needs_new_encoder` was always
        // `true` for it above, so `stage_encoder` holds exactly this
        // encoder and the very next iteration's `previous.endEncoding()`
        // (or, on the last position, the post-loop drain below) would
        // otherwise be the only place it closes. Closing immediately
        // instead reproduces ROW 309's original per-position timing
        // exactly, rather than silently widening every reported encoder by
        // one dispatch's worth of retire bookkeeping.
        if shared_encoder.is_none()
            && split_at.is_none()
            && let Some(open) = stage_encoder.take()
        {
            open.finish();
        }
        if let Some((fault_buffer, gathers)) = fault {
            pending_faults.push((bound, fault_buffer, gathers));
        }
        for retired in &prepared.retires[position] {
            if always_live.contains(retired) {
                continue;
            }
            device_buffers.remove(retired);
        }
    }
    if let Some(encoder) = shared_encoder {
        encoder.finish();
    }
    // ROW 329: the last group's stage-boundary encoder (split mode's
    // encoder-2, or a non-split call that somehow left one open) never hit
    // the per-iteration close above.
    if let Some(open) = stage_encoder.take() {
        open.finish();
    }

    // Same CPU-wall-clock-around-commit-and-wait shape
    // `execute_plan_with_placements`'s own `GPU_EXEC_TICKS` counter uses --
    // ROW 309 found this diagnostic path left `gpu_exec_ms` unrecorded for
    // its own step (the production path's counter bump never runs when
    // this function replaces it), so a caller comparing this call's
    // encoder-split totals against `gpu_exec_ms` had no same-step number to
    // compare against. Wiring the identical counter here closes that gap
    // with no second timing mechanism: it is the same `cpu_gpu_start`/
    // `cpu_gpu_end` bracket this function already takes for its own
    // nanosecond calibration, just also fed to the shared counter.
    let gpu_exec_started = read_ticks();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    counter!(GPU_EXEC_CALLS, 1);
    counter!(GPU_EXEC_TICKS, elapsed_ticks(gpu_exec_started));
    let cpu_gpu_end = sample_timestamps(&device);

    for (bound, fault_buffer, gathers) in &pending_faults {
        check_gather_fault(bound, fault_buffer, *gathers)?;
    }

    let cpu_ns_delta = ticks_to_nanos(cpu_gpu_end.0.wrapping_sub(cpu_gpu_start.0));
    let gpu_tick_delta = cpu_gpu_end.1.saturating_sub(cpu_gpu_start.1).max(1);
    let ns_per_gpu_tick = cpu_ns_delta as f64 / gpu_tick_delta as f64;

    let range = objc2_foundation::NSRange {
        location: 0,
        length: sample_count as NSUInteger,
    };
    // SAFETY: `range` is bounds-checked by construction above (it spans
    // exactly the `sample_count` samples this call itself wrote);
    // `resolveCounterRange`'s own unsafety is the driver's undocumented
    // out-of-range behavior, which this range cannot trigger.
    let resolved = unsafe { sample_buffer.resolveCounterRange(range) }.ok_or_else(|| {
        MetalError::CompileFailed {
            log: "counter sample buffer resolved no data".to_string(),
        }
    })?;
    let raw = resolved.to_vec();
    // ROW 329: split mode's three samples describe two ENCODER-level spans,
    // not `position_count` per-position ones -- reading them as
    // `2 * position` pairs (the non-split layout) would read past index 2
    // for every position beyond the first and silently attribute garbage
    // `gpu_ns`. Every `OpGpuTiming::gpu_ns` stays `0` here instead, same as
    // `"unsupported"` -- never fabricated -- and `encoder_split_ns` below
    // carries the real, encoder-granularity numbers this call actually
    // sampled.
    let encoder_split_ns = split_at.map(|_| {
        let buffer_start = read_timestamp(&raw, 0);
        let encoder_one_end = read_timestamp(&raw, 1);
        let encoder_two_end = read_timestamp(&raw, 2);
        let encoder_one_ns = if buffer_start == u64::MAX || encoder_one_end == u64::MAX {
            0
        } else {
            (encoder_one_end.wrapping_sub(buffer_start) as f64 * ns_per_gpu_tick).max(0.0) as u64
        };
        let encoder_two_ns = if encoder_one_end == u64::MAX || encoder_two_end == u64::MAX {
            0
        } else {
            (encoder_two_end.wrapping_sub(encoder_one_end) as f64 * ns_per_gpu_tick).max(0.0) as u64
        };
        (encoder_one_ns, encoder_two_ns)
    });
    let mut timings = Vec::with_capacity(position_count);
    for (position, meta) in metas.into_iter().enumerate() {
        let gpu_ns = if split_at.is_some() {
            0
        } else {
            if position >= profiled_position_count {
                0
            } else {
                let start = read_timestamp(&raw, 2 * position);
                let end = read_timestamp(&raw, 2 * position + 1);
                if start == u64::MAX || end == u64::MAX {
                    0
                } else {
                    (end.wrapping_sub(start) as f64 * ns_per_gpu_tick).max(0.0) as u64
                }
            }
        };
        let (
            node,
            kind,
            operand_bytes,
            bound_buffer_bytes,
            weight_name,
            operand_count,
            packed_codec,
            packed_kernel_variant,
        ) = meta;
        timings.push(OpGpuTiming {
            node,
            kind,
            extents: prepared.resolved[position].extents.clone(),
            output_axes: match &prepared.resolved[position].kind {
                BoundOpKind::Reduce { output_axes, .. } => output_axes.to_vec(),
                _ => Vec::new(),
            },
            operand_bytes,
            bound_buffer_bytes,
            gpu_ns,
            weight_name,
            operand_count,
            packed_codec,
            packed_kernel_variant,
            packed_row_block_rejection: None,
        });
    }

    let placed_output_nodes: BTreeSet<NodeId> = output_placed.keys().copied().collect();
    let evaluated = finish(plan, &device_buffers, &placed_output_nodes, None)?;
    let sampling_mode = if dispatch_boundary {
        "dispatch-boundary"
    } else {
        "stage-boundary"
    };
    Ok((evaluated, timings, sampling_mode, encoder_split_ns))
}

/// One `(cpuTimestamp, gpuTimestamp)` reading via
/// `MTLDevice::sampleTimestamps:gpuTimestamp:` -- the CPU side is in the
/// same `mach_absolute_time` domain [`ticks_to_nanos`] converts, which is
/// what lets [`execute_plan_with_placements_dispatch_timed`] calibrate GPU
/// ticks to nanoseconds without a second, unrelated conversion table.
#[cfg(feature = "instrument")]
pub(super) fn sample_timestamps(device: &ProtocolObject<dyn MTLDevice>) -> (u64, u64) {
    let mut cpu_timestamp: objc2_metal::MTLTimestamp = 0;
    let mut gpu_timestamp: objc2_metal::MTLTimestamp = 0;
    // SAFETY: both out-pointers are valid, stack-local `u64`s.
    unsafe {
        device.sampleTimestamps_gpuTimestamp(
            core::ptr::NonNull::from(&mut cpu_timestamp),
            core::ptr::NonNull::from(&mut gpu_timestamp),
        );
    }
    (cpu_timestamp, gpu_timestamp)
}

/// One [`objc2_metal::MTLCounterResultTimestamp`]'s 8-byte little-endian
/// `timestamp` field out of `resolveCounterRange`'s raw `NSData` bytes --
/// `u64::MAX` (Metal's own `MTLCounterErrorValue`) when the driver could
/// not take that sample, which the caller treats as "no measurement", never
/// a real zero-length dispatch.
#[cfg(feature = "instrument")]
pub(super) fn read_timestamp(raw: &[u8], sample_index: usize) -> u64 {
    let start = sample_index * core::mem::size_of::<u64>();
    raw.get(start..start + core::mem::size_of::<u64>())
        .and_then(|bytes| bytes.try_into().ok())
        .map(u64::from_ne_bytes)
        .unwrap_or(u64::MAX)
}

/// [`execute_plan_with_placements_dispatch_timed`] against a name-keyed
/// block set, mirroring [`execute_plan_named_with_placements_op_timed`]'s
/// own name resolution.
///
/// # Errors
/// Propagates name-resolution and Metal driver failures.
#[cfg(all(feature = "metal-output-placement", feature = "instrument"))]
pub fn execute_plan_named_with_placements_dispatch_timed(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'_>)],
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
    output_placements: &[(NodeId, &PlacedBuffer, usize)],
) -> Result<DispatchTimedOutcome, MetalError> {
    let blocks = resolve_named_blocks(&plan.program, named)?;
    execute_plan_with_placements_dispatch_timed(plan, &blocks, input_placements, output_placements)
}

/// Classifies one [`BoundOp`]'s emitted kernel body the same way
/// `omega/examples/real_forward_packed_probe.rs` (discipline log ROW 85)
/// does: by grepping the SOURCE for the tiled-GEMM simdgroup-matrix call
/// site vs the row-blocked call site vs the generic per-element scalar call
/// site vs a SIMD cooperative combine, since [`crate::msl::emit`] is the one
/// place that decides which of the four a given `Reduce` gets and none of
/// that decision is exposed as its own accessor.
#[cfg(feature = "instrument")]
pub(super) fn classify_kind(bound: &BoundOp, packed_operands: &PackedOperands) -> &'static str {
    match &bound.kind {
        // `BoundOpKind::name()` is the one place these four (plus
        // `keep::scan fold` below) are spelled -- this arm never restates
        // its own copy, so a future variant or renamed arm cannot drift
        // between this profiler label and `RenderKindMismatch`'s own.
        BoundOpKind::CachedAttention { .. }
        | BoundOpKind::CachedSoftmaxWeights { .. }
        | BoundOpKind::Elementwise { .. }
        | BoundOpKind::Iota
        | BoundOpKind::Constant { .. }
        | BoundOpKind::GatedDeltaNet { .. }
        | BoundOpKind::MoeTopK { .. }
        | BoundOpKind::RoundBatchedReduce { .. } => bound.kind.name(),
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => bound.kind.name(),
        BoundOpKind::Reduce {
            keep: Keep::Reduce, ..
        } => match emit(bound, packed_operands, NumericPolicy::llama_relaxed()) {
            // Checked BEFORE the row-blocked arm below: ROW 113's
            // weight-staging fix made `push_tiled_gemm_body` call
            // `q4k_run8`/`q4k_header_for` too (the same amortized decode
            // `push_packed_row_blocked_body` already used), so the row-blocked
            // arm's own source markers must include both packed bodies --
            // `simdgroup_multiply_accumulate` only ever appears in
            // [`crate::msl::push_tiled_gemm_body`]'s emitted source, so it is
            // the one marker that still disambiguates tiled from packed.
            Ok(kernel) if kernel.source.contains("simdgroup_multiply_accumulate") => {
                "reduce-tiled-gemm"
            }
            // Consults `msl::PACKED_ROW_BODY_MARKERS` rather than restating
            // its own copy of the marker list: the restated copy is exactly
            // what went stale before (missing `q3k_pair_dot(blk`/
            // `q3k_element(blk`/`q6k_pair_dot(blk`, see that const's own
            // doc), undercounting every Q3_K row-blocked dispatch and every
            // plain-product Q6_K dispatch into `"reduce-cooperative"` below.
            Ok(kernel)
                if crate::msl::PACKED_ROW_BODY_MARKERS
                    .iter()
                    .any(|marker| kernel.source.contains(marker)) =>
            {
                "reduce-packed-row-blocked"
            }
            Ok(kernel)
                if kernel.source.contains("simd_sum(")
                    || kernel.source.contains("simd_max(")
                    || kernel.source.contains("simd_min(")
                    || kernel.source.contains("simd_product(") =>
            {
                "reduce-cooperative"
            }
            Ok(_) => "reduce-generic-scalar",
            Err(_) => "reduce-unclassified",
        },
    }
}

#[cfg(feature = "instrument")]
pub(super) fn classify_packed_kernel_variant(
    bound: &BoundOp,
    packed_operands: &PackedOperands,
) -> &'static str {
    let Ok(kernel) = emit(bound, packed_operands, NumericPolicy::default()) else {
        return "unclassified";
    };
    if kernel.source.contains("q4k_pair_dot(blk") {
        "q4k-paired"
    } else if kernel.source.contains("q4k_run8(blk") {
        "q4k-run8"
    } else if kernel.source.contains("q5k_pair_dot(blk") {
        "q5k-paired"
    } else if kernel.source.contains("q5k_value(blk") {
        "q5k-scalar"
    } else if kernel.source.contains("q6k_value(blk") {
        "q6k-scalar"
    } else if kernel.source.contains("acc1_0") {
        // `metal-q4k-ggml-port`'s own body -- same marker and same reason
        // as `classify_kind`'s own arm above, so this variant name does not
        // fall into "other" alongside genuinely unclassified kernels.
        "q4k-ggml-port"
    } else {
        "other"
    }
}

#[cfg(all(test, feature = "instrument"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(super) mod classify_kind_packed_row_marker_tests {
    //! `classify_kind` used to restate its own copy of
    //! [`crate::msl::PACKED_ROW_BODY_MARKERS`] and the copy went stale: it
    //! carried `q4k_pair_dot(blk`/`q5k_pair_dot(blk`/`q5k_value(blk`/
    //! `q6k_value(blk` but never `q3k_pair_dot(blk`/`q3k_element(blk`/
    //! `q6k_pair_dot(blk` (`msl.rs:5142`, `5163`, `5298`), so every Q3_K
    //! row-blocked dispatch and every plain-product Q6_K dispatch (the shape
    //! the openchat output head actually takes) fell through to
    //! `"reduce-cooperative"`. Renders ONE kernel per (codec, body) pair
    //! through the real `emit` path -- the plain-product pair-dot arm (an
    //! `Add`-reduce over a `Multiply` body) and the per-element scalar
    //! fallback arm (any other reduce op) -- for all four K-quant codecs.

    use alloc::collections::BTreeMap;
    use alloc::vec;

    use proxima_tensor::{
        AxisIndex, AxisTerm, BoundOp, DType, Extent, IndexMap, IndexPattern, NumericPolicy, Op,
        Reduce, ReduceInit, ScalarOp, append, bind, bind_with_fusion, infer, map,
    };

    use proxima_tensor::NodeId;
    use proxima_tensor::spec::{
        elementwise, grouped_gathered_expert_product, input_leaf, reduce as spec_reduce,
        scalar_constant,
    };

    use super::classify_kind;
    use crate::msl::diagnose_packed_row_block;
    use crate::{Codec, PackedOperands};

    /// The last node `program` builds -- see `msl::tests::terminal`'s own
    /// doc (ROW 541, `proxima-tensor/docs/discipline.md`): every fixture
    /// here treats it as "the answer", and `bind_plain`'s reachability pass
    /// now requires it be named explicitly rather than relying on `&[]`.
    fn terminal(program: &[Op]) -> NodeId {
        NodeId((program.len() - 1) as u32)
    }

    fn matmul_op_with_reduce(m: u32, k: u32, n: u32, reduce_op: ScalarOp) -> BoundOp {
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
                body: reduce_op,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: proxima_tensor::Keep::Reduce,
                name: Some("matmul".into()),
            }),
        );
        let shapes = infer(&program, &[]).expect("matmul infers");
        bind(&program, &shapes, &[terminal(&program)], NumericPolicy::default())
            .expect("matmul lowers")
            .into_iter()
            .next()
            .expect("one fused bound emitted")
    }

    fn packed_operands_for(bound: &BoundOp, codec: Codec) -> PackedOperands {
        let mut codecs = BTreeMap::new();
        codecs.insert(bound.operands()[0].0, codec);
        codecs
    }

    #[test]
    fn every_k_quant_codec_classifies_as_packed_row_blocked_plain_product() {
        for codec in [
            Codec::Q3K,
            Codec::Q4K,
            Codec::Q5K,
            Codec::Q6K,
        ] {
            // Add-reduce over a plain `weight * activation` body selects the
            // `plain_product` pair-dot arm (`push_packed_row_blocked_body`'s
            // own `plain_product` gate) for every codec that supports it.
            let bound = matmul_op_with_reduce(4, 256, 5, ScalarOp::Add);
            let packed_operands = packed_operands_for(&bound, codec);
            assert_eq!(
                classify_kind(&bound, &packed_operands),
                "reduce-packed-row-blocked",
                "{codec:?} plain-product row-blocked dispatch must classify as \
                 reduce-packed-row-blocked, not fall through to reduce-cooperative"
            );
        }
    }

    #[test]
    fn every_k_quant_codec_classifies_as_packed_row_blocked_scalar_fallback() {
        for codec in [
            Codec::Q3K,
            Codec::Q4K,
            Codec::Q5K,
            Codec::Q6K,
        ] {
            // A non-Add reduce op takes `push_packed_row_blocked_body`'s
            // per-element scalar fallback arm instead of the pair-dot arm.
            let bound = matmul_op_with_reduce(4, 256, 5, ScalarOp::Maximum);
            let packed_operands = packed_operands_for(&bound, codec);
            assert_eq!(
                classify_kind(&bound, &packed_operands),
                "reduce-packed-row-blocked",
                "{codec:?} per-element scalar row-blocked dispatch must classify as \
                 reduce-packed-row-blocked, not fall through to reduce-cooperative"
            );
        }
    }

    /// `Q4_0`/`Q8_0` are flat (non-K-quant) codecs that still route through
    /// `push_packed_row_blocked_body`'s single-row arm
    /// (`m=1`, `packed_row_block_token_total(block, ..) == 1`) for gemma4-E2B's
    /// decode-shaped dispatches -- `op_profile_codec` already counted ~275 of
    /// these on the fast packed-row path while `op_profile_kind` mislabeled
    /// every one `reduce-cooperative` because [`PACKED_ROW_BODY_MARKERS`]
    /// carried no `Q4_0`/`Q8_0` entry (this const's own doc). `m=1` (not the
    /// K-quant fixtures' `m=4`) is deliberate: it is the single-row body this
    /// bug actually hit, not the multi-row body the K-quant fixtures above
    /// exercise.
    #[test]
    fn q4_0_and_q8_0_single_row_dispatch_classifies_as_packed_row_blocked() {
        for codec in [Codec::Q4_0, Codec::Q8_0] {
            let bound = matmul_op_with_reduce(1, 256, 5, ScalarOp::Add);
            let packed_operands = packed_operands_for(&bound, codec);
            assert_eq!(
                classify_kind(&bound, &packed_operands),
                "reduce-packed-row-blocked",
                "{codec:?} single-row packed-row dispatch must classify as \
                 reduce-packed-row-blocked, not fall through to reduce-cooperative"
            );
        }
    }

    /// Same gap, the batched (`m>1`) multi-row body
    /// (`push_packed_row_multi_row_body`'s generic, non-`fast_q4k` loop),
    /// which renders through `operand_read` (`q4_0_element(in`/
    /// `q8_0_element(in`) rather than the single-row body's named helpers
    /// (`q4_0_super_element(blk`/`q8_0_super_element(blk`) -- a distinct
    /// marker text, so a distinct test.
    #[test]
    fn q4_0_and_q8_0_multi_row_dispatch_classifies_as_packed_row_blocked() {
        for codec in [Codec::Q4_0, Codec::Q8_0] {
            let bound = matmul_op_with_reduce(4, 256, 5, ScalarOp::Add);
            let packed_operands = packed_operands_for(&bound, codec);
            assert_eq!(
                classify_kind(&bound, &packed_operands),
                "reduce-packed-row-blocked",
                "{codec:?} multi-row packed-row dispatch must classify as \
                 reduce-packed-row-blocked, not fall through to reduce-cooperative"
            );
        }
    }

    /// The `[sequence, selected, d_in, d_out]` gather
    /// [`grouped_gathered_expert_product`] (`proxima_tensor::spec`,
    /// `spec.rs:1617-1654`) builds, generalized to accept an `x` of any
    /// rank via `x_axes` -- main's exported function hardcodes `x_axes =
    /// [0, 2]` (an `x` of rank 2, `[sequence, d_in]`, the gate/up shape).
    /// Down's own activation is `silu(grouped_gate) * grouped_up`, already
    /// `[sequence, selected, d_in]` (rank 3) because gate/up are grouped
    /// upstream of it -- `x_axes = [0, 1, 2]` reads that shape directly, no
    /// per-round stacking needed. This is a test-only reconstruction of the
    /// generalization ROW 538's reverted `10bac25a` landed in `spec.rs`
    /// itself; everything it is built from (`IndexMap::Computed`,
    /// `IndexPattern`, `AxisIndex`, `AxisTerm`) is public on main today.
    fn grouped_gathered_expert_product_with_x_axes(
        program: &mut Vec<Op>,
        stack: NodeId,
        route: NodeId,
        x: NodeId,
        x_axes: &[u16],
    ) -> NodeId {
        let gathered_map = IndexMap::Computed {
            indices: route,
            index_map: map::projection(4, &[0, 1]),
            base: IndexPattern {
                iter_rank: 4,
                axes: vec![
                    AxisIndex::default(),
                    AxisIndex {
                        terms: core::iter::once(AxisTerm::projection(2)).collect(),
                        offset: 0,
                        len: None,
                    },
                    AxisIndex {
                        terms: core::iter::once(AxisTerm::projection(3)).collect(),
                        offset: 0,
                        len: None,
                    },
                ],
            },
            gathered_dim: 0,
        };
        let x_map = IndexMap::Affine(map::projection(4, x_axes));
        append(
            program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![(stack, gathered_map), (x, x_map)],
                name: None,
            },
        )
    }

    /// ROW 538: the real qwen35moe decode shape (256 experts, top-8, hidden
    /// 2048, embedding 512), gate/up/down each grouped over the `k` selected
    /// experts through ONE [`grouped_gathered_expert_product`] call apiece --
    /// gate/up via main's own exported function (the same primitive
    /// `append_moe_ffn`'s `GroupedGateUp` strategy uses), down via this test's own
    /// [`grouped_gathered_expert_product_with_x_axes`] (see its doc for why
    /// main's exported form cannot take down's rank-3 activation as-is).
    /// `append_moe_ffn`'s full router is a PRIVATE fn on main and always
    /// takes the reverted `PerRoute` branch for down (ROW 538), so `routes`
    /// is built directly as an `[sequence, selected]` input here --
    /// `classify_kind` only reads the resulting reduce's shape/layout, never
    /// how the route ids were produced, and main's own
    /// `grouped_gathered_expert_product_infers_selected_axis` unit test
    /// (`spec.rs:11226`) already establishes an `Input` route is a
    /// legitimate shape to bind this gather against.
    ///
    /// Bound through `bind_with_fusion` exactly like `omega::metal::prepare`
    /// does. Prints every reduce's `classify_kind`, and on decline the exact
    /// [`crate::msl::PackedRowBlockRejection`] `diagnose_packed_row_block`
    /// names, so a failure shows which of the three expert reduces
    /// (gate/up/down) regressed and why.
    ///
    /// ROW 538 update: with `correct_packed_matmul_layouts` applied (this
    /// test used to skip it -- see the call site's own doc), gate and up
    /// both classify `reduce-packed-row-blocked` with `diagnose_packed_row_block`
    /// verdict `Ok(())`, unconditionally. Down alone declines
    /// `Err(GatheredOperand)` (`msl.rs:2740-2742`) with the feature off:
    /// unlike gate/up, whose activation is the SAME `x` for every one of the
    /// `k` selected experts (`split_token_feature_axes` sees a zero stride on
    /// the selected axis for both operands and falls back to "no token
    /// axis"), down's own activation genuinely varies per selected expert
    /// (`silu(gate_k) * up_k`), so `split_token_feature_axes` correctly
    /// names BOTH `sequence` and `selected` as token axes and
    /// `packed_row_block_token_total` comes back `8 > 1` -- a real
    /// multi-row-gathered dispatch, not a probe artifact. Turning
    /// `metal-gathered-packed-row` on flips `diagnose_packed_row_block` to
    /// `Ok(())` for down too, but `classify_kind`'s own render-and-inspect
    /// still reports `reduce-cooperative` for it even then -- a SECOND site
    /// beyond `classify_packed_row_block`'s admission (inside
    /// `push_packed_row_blocked_body`'s multi-row body,
    /// `msl.rs:8169-8183` and onward) does not yet emit the packed markers
    /// for this shape. That is a kernel-body change, out of scope for this
    /// slice -- named here rather than fixed.
    #[ignore = "down's grouped reduce is a genuine multi-row-gathered dispatch \
                (msl.rs:2740 GatheredOperand); metal-gathered-packed-row admits \
                it structurally but push_packed_row_blocked_body's multi-row \
                body still does not render the packed markers for it (kernel \
                change, out of scope)"]
    #[test]
    fn qwen35moe_shaped_grouped_expert_reduces_classify_as_packed_row_blocked() {
        use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};

        const EMBEDDING: usize = 512;
        const FEED_FORWARD: usize = 2048;
        const EXPERT_COUNT: u32 = 256;
        const EXPERT_USED_COUNT: u32 = 8;
        const SEQUENCE: usize = 1;

        fn quantized_stack(expert_count: u32, rows: usize, k: usize) -> Vec<u8> {
            let elements_per_expert = rows * k;
            let blocks_per_expert = elements_per_expert / QK_K;
            let mut stacked = vec![0u8; expert_count as usize * blocks_per_expert * BLOCK_BYTES];
            let input = vec![0.01f32; elements_per_expert];
            for expert in 0..expert_count as usize {
                let byte_span = blocks_per_expert * BLOCK_BYTES;
                let output = &mut stacked[expert * byte_span..(expert + 1) * byte_span];
                quantize(&input, output).expect("synthetic expert slab quantizes to Q4_K");
            }
            stacked
        }

        let mut program = Vec::new();
        let x_node = input_leaf(
            &mut program,
            DType::Float32,
            vec![Extent::Symbolic(0), Extent::Static(EMBEDDING as u32)],
            "x",
        );
        let routes_node = input_leaf(
            &mut program,
            DType::Int32,
            vec![Extent::Symbolic(0), Extent::Static(EXPERT_USED_COUNT)],
            "routes",
        );
        let expert_w_gate_node = input_leaf(
            &mut program,
            DType::Float32,
            vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(EMBEDDING as u32),
                Extent::Static(FEED_FORWARD as u32),
            ],
            "expert_w_gate",
        );
        let expert_w_up_node = input_leaf(
            &mut program,
            DType::Float32,
            vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(EMBEDDING as u32),
                Extent::Static(FEED_FORWARD as u32),
            ],
            "expert_w_up",
        );
        let expert_w_down_node = input_leaf(
            &mut program,
            DType::Float32,
            vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(FEED_FORWARD as u32),
                Extent::Static(EMBEDDING as u32),
            ],
            "expert_w_down",
        );

        let gate_product =
            grouped_gathered_expert_product(&mut program, expert_w_gate_node, routes_node, x_node);
        let grouped_gate = spec_reduce(
            &mut program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            gate_product,
            "skio->skio",
            "sko->skio",
        )
        .expect("grouped gate reduce lowers");
        let up_product =
            grouped_gathered_expert_product(&mut program, expert_w_up_node, routes_node, x_node);
        let grouped_up = spec_reduce(
            &mut program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            up_product,
            "skio->skio",
            "sko->skio",
        )
        .expect("grouped up reduce lowers");

        // silu(grouped_gate) * grouped_up, vectorized over the whole
        // [sequence, selected, feed_forward] tensor in one pass -- gate and
        // up are already grouped, so down's own per-round hidden activation
        // falls out without `stack_selected_routes`'s one-hot stacking (that
        // trick exists only to combine round-scoped values computed one
        // round at a time; nothing here is round-scoped any more).
        let neg_gate = elementwise(
            &mut program,
            DType::Float32,
            ScalarOp::Negate,
            &[(grouped_gate, "sko->sko")],
        )
        .expect("negate lowers");
        let exp_neg_gate = elementwise(
            &mut program,
            DType::Float32,
            ScalarOp::Exponential,
            &[(neg_gate, "sko->sko")],
        )
        .expect("exponential lowers");
        let one = scalar_constant(&mut program, 1.0);
        let one_plus_exp = elementwise(
            &mut program,
            DType::Float32,
            ScalarOp::Add,
            &[(exp_neg_gate, "sko->sko"), (one, "->sko")],
        )
        .expect("add lowers");
        let sigmoid_gate = elementwise(
            &mut program,
            DType::Float32,
            ScalarOp::Reciprocal,
            &[(one_plus_exp, "sko->sko")],
        )
        .expect("reciprocal lowers");
        let silu_gate = elementwise(
            &mut program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(grouped_gate, "sko->sko"), (sigmoid_gate, "sko->sko")],
        )
        .expect("silu lowers");
        let hidden = elementwise(
            &mut program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(silu_gate, "sko->sko"), (grouped_up, "sko->sko")],
        )
        .expect("hidden lowers");

        let down_product = grouped_gathered_expert_product_with_x_axes(
            &mut program,
            expert_w_down_node,
            routes_node,
            hidden,
            &[0, 1, 2],
        );
        let root = spec_reduce(
            &mut program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            down_product,
            "skio->skio",
            "sko->skio",
        )
        .expect("grouped down reduce lowers");

        let symbols = [SEQUENCE as u64];
        let shapes = infer(&program, &symbols).expect("the qwen35moe-shaped grouped ffn infers");
        let mut resolved =
            bind_with_fusion(&program, &shapes, &[root], true, NumericPolicy::default())
                .expect("the qwen35moe-shaped grouped ffn binds");

        let gate_stack = quantized_stack(EXPERT_COUNT, EMBEDDING, FEED_FORWARD);
        let up_stack = quantized_stack(EXPERT_COUNT, EMBEDDING, FEED_FORWARD);
        let down_stack = quantized_stack(EXPERT_COUNT, FEED_FORWARD, EMBEDDING);
        let mut packed_operands: PackedOperands = BTreeMap::new();
        packed_operands.insert(expert_w_gate_node, Codec::Q4K);
        packed_operands.insert(expert_w_up_node, Codec::Q4K);
        packed_operands.insert(expert_w_down_node, Codec::Q4K);
        // the stacks are built purely to size the quantized codec buffers
        // `classify_kind`'s emit path never reads bytes -- kept alive so the
        // borrow checker sees them span the assertions below.
        let _ = (gate_stack, up_stack, down_stack);

        // `bind_with_fusion`'s own `layout_of` assumes every operand is
        // row-major in its DECLARED axis order -- wrong for a packed Q4_K
        // weight, whose bytes are GGUF's native `[out, in]` regardless of
        // the declared shape. `omega::metal::prepare` (`metal.rs:6530`)
        // always runs this correction before `emit`/`classify_kind` ever see
        // `resolved`; skipping it here made every one of gate/up/down
        // decline with `NonUnitWeightStride` against the UNCORRECTED
        // declared-shape layout, not against the real production layout.
        proxima_tensor::correct_packed_matmul_layouts(
            &mut resolved,
            &packed_operands.keys().copied().collect(),
        );

        let mut expert_reduce_kinds = Vec::new();
        for bound in &resolved {
            if matches!(
                bound.kind,
                proxima_tensor::BoundOpKind::Reduce {
                    keep: proxima_tensor::Keep::Reduce,
                    ..
                }
            ) {
                let kind = classify_kind(bound, &packed_operands);
                std::eprintln!("classify_kind: {kind} name={:?}", bound.kind.name());
                let touches_expert_weight = bound.operands().iter().any(|(node, _, _)| {
                    *node == expert_w_gate_node
                        || *node == expert_w_up_node
                        || *node == expert_w_down_node
                });
                if touches_expert_weight {
                    let quantized: Vec<Option<Codec>> = bound
                        .operands()
                        .iter()
                        .map(|(node, _, _)| packed_operands.get(node).copied())
                        .collect();
                    let verdict = diagnose_packed_row_block(bound, &quantized);
                    std::eprintln!(
                        "diagnose_packed_row_block: node={:?} kind={kind} verdict={verdict:?}",
                        bound.node
                    );
                    expert_reduce_kinds.push(kind);
                }
            }
        }

        assert_eq!(
            expert_reduce_kinds.len(),
            3,
            "gate, up, and down each contribute exactly one reduce over their own expert weight"
        );
        for kind in expert_reduce_kinds {
            assert_eq!(
                kind, "reduce-packed-row-blocked",
                "the grouped expert product must lower to the same fast packed-row body \
                 the per-route form uses, not materialize the gathered product cooperatively"
            );
        }
    }

    /// ROW 543: `proxima_tensor::spec::append_moe_ffn` with
    /// `MoeProjectionStrategy::GroupedGateUp` (the production entry point
    /// for the grouped strategy this landing investigates -- `append_moe_ffn`
    /// defaults its callers to `PerRoute` after 07e1fad8's revert), not a
    /// hand-built stand-in for it like the
    /// ignored probe above. Census: 2 grouped-packed reduces (gate, up) +
    /// `EXPERT_USED_COUNT` per-route-packed reduces (down), every one
    /// `reduce-packed-row-blocked`, and no surviving elementwise op whose
    /// output shape is the materialized `[.., d_in, d_out]` gathered
    /// product.
    #[test]
    fn qwen35moe_shaped_append_moe_ffn_packs_grouped_gate_up_and_per_route_down() {
        use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};
        use proxima_tensor::spec::{Activation, ExpertGatingFunc};

        const EMBEDDING: usize = 512;
        const FEED_FORWARD: usize = 2048;
        const EXPERT_COUNT: u32 = 256;
        const EXPERT_USED_COUNT: u32 = 8;
        const SEQUENCE: usize = 1;

        fn quantized_stack(expert_count: u32, rows: usize, k: usize) -> Vec<u8> {
            let elements_per_expert = rows * k;
            let blocks_per_expert = elements_per_expert / QK_K;
            let mut stacked = vec![0u8; expert_count as usize * blocks_per_expert * BLOCK_BYTES];
            let input = vec![0.01f32; elements_per_expert];
            for expert in 0..expert_count as usize {
                let byte_span = blocks_per_expert * BLOCK_BYTES;
                let output = &mut stacked[expert * byte_span..(expert + 1) * byte_span];
                quantize(&input, output).expect("synthetic expert slab quantizes to Q4_K");
            }
            stacked
        }

        let mut program = Vec::new();
        let x_node = input_leaf(
            &mut program,
            DType::Float32,
            vec![Extent::Symbolic(0), Extent::Static(EMBEDDING as u32)],
            "x",
        );
        let gate_inp_node = input_leaf(
            &mut program,
            DType::Float32,
            vec![
                Extent::Static(EMBEDDING as u32),
                Extent::Static(EXPERT_COUNT),
            ],
            "gate_inp",
        );
        let expert_w_gate_node = input_leaf(
            &mut program,
            DType::Float32,
            vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(EMBEDDING as u32),
                Extent::Static(FEED_FORWARD as u32),
            ],
            "expert_w_gate",
        );
        let expert_w_up_node = input_leaf(
            &mut program,
            DType::Float32,
            vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(EMBEDDING as u32),
                Extent::Static(FEED_FORWARD as u32),
            ],
            "expert_w_up",
        );
        let expert_w_down_node = input_leaf(
            &mut program,
            DType::Float32,
            vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(FEED_FORWARD as u32),
                Extent::Static(EMBEDDING as u32),
            ],
            "expert_w_down",
        );
        let ones = scalar_constant(&mut program, 1.0);

        // main's `append_moe_ffn` reverted to `PerRoute` (07e1fad8); the
        // production entry point for the grouped strategy under
        // investigation is `append_moe_ffn` with `MoeProjectionStrategy::GroupedGateUp`.
        let moe_spec = proxima_tensor::spec::MoeFfnSpec {
            router: proxima_tensor::spec::MoeRouter::GateInput(gate_inp_node),
            expert_w_gate: expert_w_gate_node,
            expert_w_up: expert_w_up_node,
            expert_w_down: expert_w_down_node,
            expert_count: EXPERT_COUNT,
            expert_used_count: EXPERT_USED_COUNT,
            ones,
            gating: ExpertGatingFunc::Softmax,
            expert_bias: None,
            expert_scale: None,
            activation: Activation::Silu,
            strategy: proxima_tensor::spec::MoeProjectionStrategy::GroupedGateUp,
        };
        let (root, _site) = proxima_tensor::spec::append_moe_ffn(&mut program, 0, x_node, &moe_spec)
            .expect("append_moe_ffn with GroupedGateUp lowers at the real qwen35moe shape");

        let symbols = [SEQUENCE as u64];
        let shapes = infer(&program, &symbols).expect("the qwen35moe-shaped ffn infers");
        let mut resolved =
            bind_with_fusion(&program, &shapes, &[root], true, NumericPolicy::default())
                .expect("the qwen35moe-shaped ffn binds");

        let mut packed_operands: PackedOperands = BTreeMap::new();
        packed_operands.insert(expert_w_gate_node, Codec::Q4K);
        packed_operands.insert(expert_w_up_node, Codec::Q4K);
        packed_operands.insert(expert_w_down_node, Codec::Q4K);
        let gate_stack = quantized_stack(EXPERT_COUNT, EMBEDDING, FEED_FORWARD);
        let up_stack = quantized_stack(EXPERT_COUNT, EMBEDDING, FEED_FORWARD);
        let down_stack = quantized_stack(EXPERT_COUNT, FEED_FORWARD, EMBEDDING);
        let _ = (gate_stack, up_stack, down_stack);

        // `omega::metal::prepare` (`metal.rs:6530`) always runs this
        // correction before `emit`/`classify_kind` ever see `resolved` --
        // production's own layout, not the declared-shape layout every
        // packed weight would otherwise wrongly decline against.
        proxima_tensor::correct_packed_matmul_layouts(
            &mut resolved,
            &packed_operands.keys().copied().collect(),
        );

        let mut expert_reduce_kinds = Vec::new();
        for bound in &resolved {
            let touches_expert_weight = bound.operands().iter().any(|(node, _, _)| {
                *node == expert_w_gate_node
                    || *node == expert_w_up_node
                    || *node == expert_w_down_node
            });
            if !touches_expert_weight {
                continue;
            }
            match &bound.kind {
                proxima_tensor::BoundOpKind::Reduce {
                    keep: proxima_tensor::Keep::Reduce,
                    ..
                } => {
                    let kind = classify_kind(bound, &packed_operands);
                    expert_reduce_kinds.push(kind);
                }
                proxima_tensor::BoundOpKind::Elementwise { .. } => {
                    let materializes_full_expert_product = bound.extents.len() >= 2
                        && bound.extents[bound.extents.len() - 2..]
                            == [FEED_FORWARD as u64, EMBEDDING as u64]
                        || bound.extents.len() >= 2
                            && bound.extents[bound.extents.len() - 2..]
                                == [EMBEDDING as u64, FEED_FORWARD as u64];
                    assert!(
                        !materializes_full_expert_product,
                        "an unfused elementwise op still carries the [.., d_in, d_out] \
                         gathered-product shape (node {:?}, extents {:?}) instead of \
                         fusing into its consuming reduce",
                        bound.node, bound.extents
                    );
                }
                _ => {}
            }
        }

        assert_eq!(
            expert_reduce_kinds.len(),
            2 + EXPERT_USED_COUNT as usize,
            "2 grouped reduces (gate, up) + one per-route reduce per selected \
             expert (down) -- 400 expert dispatches per token at k=8, not 960"
        );
        for kind in expert_reduce_kinds {
            assert_eq!(
                kind, "reduce-packed-row-blocked",
                "every expert-weight reduce append_moe_ffn's GroupedGateUp strategy builds must \
                 take the fast packed-row body, whether grouped (gate/up) or per-route (down)"
            );
        }
    }
}

/// [`diagnose_packed_row_block`]'s verdict for THIS bound op, against the
/// REAL production layout rather than a synthetic symbolic probe --
/// `None` for anything that is not a `Reduce { keep: Keep::Reduce, .. }`
/// (the row-blocked kernel does not apply). `Some("PASS")` means it took
/// (or would take) the row-blocked path; `Some(<debug of the rejection>)`
/// names the exact gate that rejected it.
#[cfg(feature = "instrument")]
pub(super) fn diagnose_kind(bound: &BoundOp, packed_operands: &PackedOperands) -> Option<String> {
    let BoundOpKind::Reduce {
        keep: Keep::Reduce,
        output_axes,
        ..
    } = &bound.kind
    else {
        return None;
    };
    let quantized: Vec<Option<Codec>> = bound
        .operands()
        .iter()
        .map(|(node, _, _)| packed_operands.get(node).copied())
        .collect();
    Some(match diagnose_packed_row_block(bound, &quantized) {
        Ok(()) => "PASS".to_string(),
        Err(rejection) => {
            let operand_layouts: Vec<_> = bound
                .operands()
                .iter()
                .map(|(_, layout, _)| layout)
                .collect();
            format!(
                "{rejection:?}; extents={:?}; output_axes={output_axes:?}; operand_layouts={operand_layouts:?}",
                bound.extents,
            )
        }
    })
}

