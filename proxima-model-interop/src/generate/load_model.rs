use super::*;

/// How many of [`OpGpuTiming`]'s entries [`report_op_timings`] names
/// individually -- the discipline log's own "top 20 ops by GPU time" ask.
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
pub(super) const OP_PROFILE_TOP_N: usize = 20;

#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
pub(super) fn routed_segment_profile_selected(layer: usize, phase: &str) -> bool {
    std::env::var("PROXIMA_METAL_SEGMENT_OP_PROFILE")
        .ok()
        .is_some_and(|target| target == alloc::format!("{phase}:{layer}"))
}

/// ROW 329: `PROXIMA_METAL_ENCODER_SPLIT_AT`, read the same
/// unset-means-off, one-env-var-per-diagnostic-knob convention as
/// `PROXIMA_DUPLICATE_HEAD`/`PROXIMA_METAL_OP_PROFILE_STEP` above --
/// `None` (unset, or unparseable) keeps
/// `execute_plan_with_placements_dispatch_timed`'s ROW 309 default
/// (one stage-boundary encoder per position); `Some(position)` ends the
/// compute encoder immediately before that plan position and opens a
/// second one for the rest of the program, in the SAME command buffer.
#[cfg(all(
    feature = "metal-output-placement",
    feature = "instrument",
    target_os = "macos"
))]
pub(super) fn encoder_split_at_from_env() -> Option<usize> {
    std::env::var("PROXIMA_METAL_ENCODER_SPLIT_AT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

/// ROW 329's own summary line: the two ROW-309-machinery encoder GPU
/// durations `execute_plan_with_placements_dispatch_timed` reports when
/// [`encoder_split_at_from_env`] named a split, their sum, and that
/// step's own `gpu_exec_ms` (now recorded for this diagnostic path too --
/// see that function's own doc) for a same-step comparison ROW 309 could
/// not make (its own diagnostic step recorded no `gpu_exec_ms` at all).
/// Asserts nothing -- this is a print, the same "informational, not a
/// gate" contract [`report_op_timings`] already has for this crate's
/// other diagnostic-only knobs.
#[cfg(all(
    feature = "metal-output-placement",
    feature = "instrument",
    target_os = "macos"
))]
pub(super) fn report_encoder_split(step: usize, encoder_split_ns: (u64, u64), gpu_exec_ns: u64) {
    let (encoder_one_ns, encoder_two_ns) = encoder_split_ns;
    let ms = |nanos: u64| nanos as f64 / 1e6;
    info!(
        step = step as u64,
        encoder_one_gpu_ms = ms(encoder_one_ns),
        encoder_two_gpu_ms = ms(encoder_two_ns),
        sum_gpu_ms = ms(encoder_one_ns + encoder_two_ns),
        gpu_exec_ms = ms(gpu_exec_ns),
        "encoder_split: row 329 two-encoder gpu attribution inside the decode buffer"
    );
}

/// Prints the per-op GPU attribution `run_decode_loop`'s
/// `PROXIMA_METAL_OP_PROFILE_STEP` branch gathers for exactly one decode
/// step: the op count and summed GPU time (asserting the count so a
/// degenerate empty profile reads as RED, not quiet), one line per
/// `OpGpuTiming::kind` bucket, and the top [`OP_PROFILE_TOP_N`] ops by GPU
/// time with their operand bytes and bytes/ns -- exactly what settles
/// whether GPU time tracks operand bytes or is flat per dispatch.
/// shape/dtype key -> (op count, total gpu ns, total operand bytes) accumulator
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
pub(super) type CooperativeShapeCounts = alloc::collections::BTreeMap<(Vec<u64>, Vec<u16>), (u64, u64, u64)>;

#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
pub(super) fn report_op_timings(step: usize, timings: &[OpGpuTiming], program: &[Op]) {
    let op_count = timings.len();
    let total_gpu_ns: u64 = timings.iter().map(|timing| timing.gpu_ns).sum();
    let total_operand_bytes: u64 = timings.iter().map(|timing| timing.operand_bytes).sum();
    eprintln!(
        "op_profile step={step} op_count={op_count} total_gpu_ms={:.3} operand_bytes={total_operand_bytes}",
        total_gpu_ns as f64 / 1e6,
    );

    info!(
        step = step as u64,
        op_count = op_count as u64,
        total_gpu_ns,
        total_gpu_ms = total_gpu_ns as f64 / 1e6,
        total_operand_bytes,
        "op_profile: gpu op-timing summary for one decode step"
    );

    let mut by_kind: alloc::collections::BTreeMap<&'static str, (u64, u64, u64)> =
        alloc::collections::BTreeMap::new();
    for timing in timings {
        let entry = by_kind.entry(timing.kind).or_insert((0, 0, 0));
        entry.0 += 1;
        entry.1 += timing.gpu_ns;
        entry.2 += timing.operand_bytes;
    }
    for (kind, (count, ns, bytes)) in &by_kind {
        eprintln!(
            "op_profile_kind step={step} kind={kind} op_count={count} gpu_ms={:.3} operand_bytes={bytes}",
            *ns as f64 / 1e6,
        );
        info!(
            step = step as u64,
            kind = *kind,
            op_count = *count,
            gpu_ms = *ns as f64 / 1e6,
            gpu_ns_per_op = *ns as f64 / *count as f64,
            operand_bytes = *bytes,
            "op_profile_bucket: gpu op-timing bucketed by op kind"
        );
    }

    let mut cooperative_shapes: CooperativeShapeCounts = alloc::collections::BTreeMap::new();
    for timing in timings
        .iter()
        .filter(|timing| timing.kind == "reduce-cooperative" && timing.weight_name.is_none())
    {
        let entry = cooperative_shapes
            .entry((timing.extents.clone(), timing.output_axes.clone()))
            .or_insert((0, 0, 0));
        entry.0 += 1;
        entry.1 += timing.gpu_ns;
        entry.2 += timing.operand_bytes;
    }
    for ((extents, output_axes), (count, gpu_ns, bytes)) in &cooperative_shapes {
        eprintln!(
            "op_profile_cooperative_shape step={step} extents={extents:?} output_axes={output_axes:?} op_count={count} gpu_ms={:.3} operand_bytes={bytes}",
            *gpu_ns as f64 / 1e6,
        );
        info!(
            step = step as u64,
            extents = ?extents,
            output_axes = ?output_axes,
            op_count = *count,
            gpu_ms = *gpu_ns as f64 / 1e6,
            operand_bytes = *bytes,
            "op_profile_cooperative_shape: unnamed cooperative reductions grouped by iteration shape"
        );
    }

    let mut by_codec: alloc::collections::BTreeMap<String, (u64, u64)> =
        alloc::collections::BTreeMap::new();
    for timing in timings {
        let codec = timing
            .packed_codec
            .map(|value| format!("{value:?}"))
            .unwrap_or_else(|| String::from("none"));
        let entry = by_codec.entry(codec).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += timing.gpu_ns;
    }
    for (codec, (count, ns)) in &by_codec {
        eprintln!(
            "op_profile_codec step={step} codec={codec} op_count={count} gpu_ms={:.3}",
            *ns as f64 / 1e6,
        );
        info!(
            step = step as u64,
            codec = %codec,
            op_count = *count,
            gpu_ms = *ns as f64 / 1e6,
            gpu_ns_per_op = *ns as f64 / *count as f64,
            "op_profile_codec: gpu op-timing bucketed by packed codec"
        );
    }

    let mut by_variant: alloc::collections::BTreeMap<&'static str, (u64, u64)> =
        alloc::collections::BTreeMap::new();
    for timing in timings {
        let entry = by_variant
            .entry(timing.packed_kernel_variant)
            .or_insert((0, 0));
        entry.0 += 1;
        entry.1 += timing.gpu_ns;
    }
    for (variant, (count, ns)) in &by_variant {
        eprintln!(
            "op_profile_variant step={step} variant={variant} op_count={count} gpu_ms={:.3}",
            *ns as f64 / 1e6,
        );
        info!(
            step = step as u64,
            variant = *variant,
            op_count = *count,
            gpu_ms = *ns as f64 / 1e6,
            gpu_ns_per_op = *ns as f64 / *count as f64,
            "op_profile_variant: gpu op-timing bucketed by packed kernel variant"
        );
    }

    if let Ok(selected_family) = std::env::var("PROXIMA_METAL_OP_PROFILE_FAMILY") {
        for timing in timings.iter().filter(|timing| {
            timing.weight_name.as_deref().is_some_and(|name| {
                strip_layer_index(name) == selected_family
                    || name.starts_with(selected_family.as_str())
            })
        }) {
            eprintln!(
                "op_profile_selected step={step} node={} family={selected_family} gpu_ms={:.3} operand_bytes={} operand_count={}",
                timing.node.0,
                timing.gpu_ns as f64 / 1e6,
                timing.operand_bytes,
                timing.operand_count,
            );
            if let Some(op) = program.get(timing.node.0 as usize) {
                eprintln!(
                    "op_profile_selected_node step={step} node={} op={op:?}",
                    timing.node.0
                );
            }
        }
    }

    let mut ranked: Vec<&OpGpuTiming> = timings.iter().collect();
    ranked.sort_by_key(|timing| core::cmp::Reverse(timing.gpu_ns));
    for (rank, timing) in ranked.iter().take(OP_PROFILE_TOP_N).enumerate() {
        let gpu_ns_per_byte = if timing.operand_bytes == 0 {
            0.0
        } else {
            timing.gpu_ns as f64 / timing.operand_bytes as f64
        };
        info!(
            step = step as u64,
            rank = (rank + 1) as u64,
            node = timing.node.0,
            kind = timing.kind,
            weight_name = ?timing.weight_name,
            packed_codec = ?timing.packed_codec,
            operand_bytes = timing.operand_bytes,
            bound_buffer_bytes = timing.bound_buffer_bytes,
            operand_count = timing.operand_count as u64,
            gpu_ns = timing.gpu_ns,
            gpu_ns_per_byte,
            "op_profile_top: top-N ops ranked by gpu time"
        );
        if timing.weight_name.is_none() {
            eprintln!(
                "op_profile_top step={step} rank={} node={} kind={} gpu_ms={:.3} operand_bytes={} operand_count={}",
                rank + 1,
                timing.node.0,
                timing.kind,
                timing.gpu_ns as f64 / 1e6,
                timing.operand_bytes,
                timing.operand_count,
            );
            if let Some(op) = program.get(timing.node.0 as usize) {
                eprintln!(
                    "op_profile_node step={step} node={} op={op:?}",
                    timing.node.0
                );
            }
        }
    }

    for timing in timings
        .iter()
        .filter(|timing| timing.kind == "reduce-cooperative" && timing.weight_name.is_none())
    {
        eprintln!(
            "op_profile_cooperative_node step={step} node={} gpu_ms={:.3} operand_bytes={} operand_count={}",
            timing.node.0,
            timing.gpu_ns as f64 / 1e6,
            timing.operand_bytes,
            timing.operand_count,
        );
        if timing.gpu_ns >= 100_000
            && let Some(op) = program.get(timing.node.0 as usize)
        {
            eprintln!(
                "op_profile_cooperative_slow step={step} node={} gpu_ms={:.3} op={op:?}",
                timing.node.0,
                timing.gpu_ns as f64 / 1e6,
            );
        }
    }

    // one row per real weight FAMILY (`blk.N.ffn_down.weight` -> `ffn_down.weight`,
    // stripping the per-layer digit so 32 layers' worth of one matmul kind
    // aggregates into one line) -- the byte-share/time-share table the
    // discipline log's "by tensor name where you can recover it" ask ends
    // on, since a per-node top-20 line cannot show whether a whole KIND is
    // slow or just its biggest instance.
    let mut by_family: alloc::collections::BTreeMap<String, FamilyGpuStats> =
        alloc::collections::BTreeMap::new();
    for timing in timings {
        let family = match &timing.weight_name {
            Some(name) => strip_layer_index(name),
            None => String::from("(no named operand)"),
        };
        let entry = by_family.entry(family).or_default();
        entry.op_count += 1;
        entry.gpu_ns += timing.gpu_ns;
        entry.operand_bytes += timing.operand_bytes;
        entry.min_operand_count = entry.min_operand_count.min(timing.operand_count);
        entry.max_operand_count = entry.max_operand_count.max(timing.operand_count);
        match timing.packed_row_block_rejection.as_deref() {
            Some("PASS") => stats_pass(entry, timing.gpu_ns, timing.operand_bytes),
            Some(rejection) => stats_reject(entry, rejection, timing.gpu_ns, timing.operand_bytes),
            None => {}
        }
    }
    let mut family_ranked: Vec<(&String, &FamilyGpuStats)> = by_family.iter().collect();
    family_ranked.sort_by_key(|(_, stats)| core::cmp::Reverse(stats.gpu_ns));
    for (family, stats) in family_ranked {
        let gpu_ns_per_byte = if stats.operand_bytes == 0 {
            0.0
        } else {
            stats.gpu_ns as f64 / stats.operand_bytes as f64
        };
        eprintln!(
            "op_profile_family step={step} family={family} op_count={} gpu_ms={:.3} operand_bytes={} row_blocked_count={} rejected_count={} gates={:?}",
            stats.op_count,
            stats.gpu_ns as f64 / 1e6,
            stats.operand_bytes,
            stats.row_blocked_count,
            stats.rejected_count,
            stats.packed_row_block_gates,
        );
        info!(
            step = step as u64,
            family = %family,
            op_count = stats.op_count,
            gpu_ms = stats.gpu_ns as f64 / 1e6,
            operand_bytes = stats.operand_bytes,
            gpu_ns_per_byte,
            min_operand_count = stats.min_operand_count as u64,
            max_operand_count = stats.max_operand_count as u64,
            row_blocked_count = stats.row_blocked_count,
            rejected_count = stats.rejected_count,
            packed_row_block_gates = ?stats.packed_row_block_gates,
            "op_profile_family: gpu op-timing aggregated by weight family"
        );
        // A family with BOTH a row-blocked verdict AND a rejected verdict
        // (`ffn_down`/`attn_v`: 28 already-packed `Q4_K` ops PASS, 4
        // still-dequantized `Q5_K` ops reject) is exactly the case the
        // aggregate line above cannot answer on its own -- "how much of
        // this family's cost is the codec gap, not the family" -- so emit
        // the split explicitly rather than making a reader subtract two
        // numbers from a set that does not carry counts.
        if stats.row_blocked_count > 0 && stats.rejected_count > 0 {
            let passed_gpu_ns_per_byte = if stats.passed_operand_bytes == 0 {
                0.0
            } else {
                stats.passed_gpu_ns as f64 / stats.passed_operand_bytes as f64
            };
            let rejected_gpu_ns_per_byte = if stats.rejected_operand_bytes == 0 {
                0.0
            } else {
                stats.rejected_gpu_ns as f64 / stats.rejected_operand_bytes as f64
            };
            info!(
                step = step as u64,
                family = %family,
                passed_op_count = stats.row_blocked_count,
                passed_gpu_ms = stats.passed_gpu_ns as f64 / 1e6,
                passed_operand_bytes = stats.passed_operand_bytes,
                passed_gpu_ns_per_byte,
                rejected_op_count = stats.rejected_count,
                rejected_gpu_ms = stats.rejected_gpu_ns as f64 / 1e6,
                rejected_operand_bytes = stats.rejected_operand_bytes,
                rejected_gpu_ns_per_byte,
                "op_profile_family_split: codec-gap cost isolated within a mixed family"
            );
        }
    }
}

/// `TASK_VM_INFO`'s `phys_footprint` field (`mach/task_info.h`) -- macOS's
/// own accounting of this process's compressed+resident memory charge, the
/// same number Activity Monitor's "Memory" column and `footprint(1)` read.
/// Read directly via `task_info` rather than `/usr/bin/time`'s whole-process
/// peak RSS because this is sampled PER DECODE STEP, from inside the
/// process, so growth can be attributed to a step boundary instead of only
/// a start/end delta. Declares its own `task_info`/`mach_task_self` FFI
/// rather than pulling in `mach2`/`mach` (neither is otherwise in this
/// workspace's dependency graph) for a struct that is a stable, versioned,
/// public part of the mach ABI (rev1, `TASK_VM_INFO_REV1_COUNT`) -- the
/// struct here mirrors the header exactly up to (and including)
/// `phys_footprint` and stops there, matching REV1's word count so the
/// kernel fills exactly the fields declared.
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
pub(super) fn phys_footprint_bytes() -> u64 {
    #[repr(C)]
    #[derive(Default)]
    struct TaskVmInfo {
        virtual_size: u64,
        region_count: i32,
        page_size: i32,
        resident_size: u64,
        resident_size_peak: u64,
        device: u64,
        device_peak: u64,
        internal: u64,
        internal_peak: u64,
        external: u64,
        external_peak: u64,
        reusable: u64,
        reusable_peak: u64,
        purgeable_volatile_pmap: u64,
        purgeable_volatile_resident: u64,
        purgeable_volatile_virtual: u64,
        compressed: u64,
        compressed_peak: u64,
        compressed_lifetime: u64,
        phys_footprint: u64,
    }

    const TASK_VM_INFO: i32 = 22;

    unsafe extern "C" {
        fn mach_task_self() -> u32;
        fn task_info(
            target_task: u32,
            flavor: i32,
            task_info_out: *mut TaskVmInfo,
            task_info_out_count: *mut u32,
        ) -> i32;
    }

    let mut info = TaskVmInfo::default();
    let mut count = (core::mem::size_of::<TaskVmInfo>() / core::mem::size_of::<u32>()) as u32;
    let result = unsafe { task_info(mach_task_self(), TASK_VM_INFO, &mut info, &mut count) };
    if result != 0 {
        return 0;
    }
    info.phys_footprint
}

/// Every field [`emit_token_breakdown`] needs to emit one `token_breakdown`
/// event -- shared by [`LoadedModel::run_decode_loop`]'s default two-range
/// arm and [`LoadedModel::run_decode_loop_placed_kv`]'s device-resident arm
/// so the two paths emit the identical field shape from ONE `info!` call site
/// rather than each hand-rolling its own field list (the placed-KV arm used
/// to hardcode `kv_cache_upload_bytes`/`evaluate_ms`/`layer_cache_append_ms`
/// as literal zeros here -- a real regression, since the placed arm's
/// evaluate call is the same cost the two-range arm times).
#[cfg(feature = "instrument")]
pub(super) struct TokenBreakdown {
    pub(super) step: usize,
    pub(super) new_count: usize,
    pub(super) cached_len_before: usize,
    pub(super) step_wall_ticks: u64,
    pub(super) apply_serving_config_ticks: u64,
    pub(super) build_position_inputs_ticks: u64,
    pub(super) named_blocks_weights_ticks: u64,
    pub(super) named_blocks_kv_ticks: u64,
    pub(super) kv_cache_upload_bytes: u64,
    pub(super) ssm_state_transfer_bytes: u64,
    pub(super) evaluate_ticks: u64,
    pub(super) layer_cache_append_ticks: u64,
    pub(super) layer_cache_append_bytes: u64,
    pub(super) greedy_pick_ticks: u64,
}

#[cfg(feature = "instrument")]
pub(super) fn emit_token_breakdown(breakdown: &TokenBreakdown) {
    let ms = |ticks: u64| ticks_to_nanos(ticks) as f64 / 1e6;
    let TokenBreakdown {
        step,
        new_count,
        cached_len_before,
        step_wall_ticks,
        apply_serving_config_ticks,
        build_position_inputs_ticks,
        named_blocks_weights_ticks,
        named_blocks_kv_ticks,
        kv_cache_upload_bytes,
        ssm_state_transfer_bytes,
        evaluate_ticks,
        layer_cache_append_ticks,
        layer_cache_append_bytes,
        greedy_pick_ticks,
    } = *breakdown;
    info!(
        step = step as u64,
        new_count = new_count as u64,
        cached_len_before = cached_len_before as u64,
        step_wall_ms = ms(step_wall_ticks),
        apply_serving_config_ms = ms(apply_serving_config_ticks),
        build_position_inputs_ms = ms(build_position_inputs_ticks),
        named_blocks_weights_ms = ms(named_blocks_weights_ticks),
        named_blocks_kv_ms = ms(named_blocks_kv_ticks),
        kv_cache_upload_bytes,
        ssm_state_transfer_bytes,
        evaluate_ms = ms(evaluate_ticks),
        layer_cache_append_ms = ms(layer_cache_append_ticks),
        layer_cache_append_bytes,
        greedy_pick_ms = ms(greedy_pick_ticks),
        "token_breakdown: per-decode-step wall-clock attribution"
    );
    if std::env::var_os("PROXIMA_DEBUG_METAL_STAGES").is_some() {
        eprintln!(
            "token_breakdown_wall step={} wall_ms={} evaluate_ms={} position_ms={} weights_ms={} kv_ms={} append_ms={} greedy_ms={}",
            step,
            ms(step_wall_ticks),
            ms(evaluate_ticks),
            ms(build_position_inputs_ticks),
            ms(named_blocks_weights_ticks),
            ms(named_blocks_kv_ticks),
            ms(layer_cache_append_ticks),
            ms(greedy_pick_ticks),
        );
    }
}

/// [`emit_token_breakdown`]'s Metal-stage counterpart -- same sharing
/// rationale, same two call sites. `metal_stage` is this step's own
/// snapshot-and-reset delta ([`metal_stage_totals`]'s own doc), so it is
/// correct to call from either decode arm as long as it is read exactly
/// once per step, immediately after that step's `evaluate`/
/// `evaluate_with_placements` call.
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
pub(super) fn emit_token_breakdown_metal(
    step: usize,
    metal_stage: &omega::metal::MetalStageTotals,
    plan_cache_len: usize,
    plan_hits: usize,
    plan_misses: usize,
    arena_allocated_bytes: (usize, usize, usize),
) {
    let ms = |ticks: u64| ticks_to_nanos(ticks) as f64 / 1e6;
    let kind_filter = kind_filter_from_env();
    let (checkpoint_mapping_buffer_bytes, expert_mapping_buffer_bytes) =
        omega::metal::mapping_buffer_allocated_bytes();
    info!(
        step = step as u64,
        prepare_calls = metal_stage.prepare_calls,
        prepare_ms = ms(metal_stage.prepare_ticks),
        emit_calls = metal_stage.emit_calls,
        emit_ms = ms(metal_stage.emit_ticks),
        pipeline_hits = metal_stage.pipeline_hits,
        pipeline_misses = metal_stage.pipeline_misses,
        pipeline_compile_ms = ms(metal_stage.pipeline_compile_ticks),
        block_upload_calls = metal_stage.block_upload_calls,
        block_upload_ms = ms(metal_stage.block_upload_ticks),
        block_offered_bytes = metal_stage.block_offered_bytes,
        block_copied_bytes = metal_stage.block_copied_bytes,
        block_nocopy_bound_bytes = metal_stage.block_nocopy_bound_bytes,
        block_offset_bound_bytes = metal_stage.block_offset_bound_bytes,
        op_setup_calls = metal_stage.op_setup_calls,
        op_setup_ms = ms(metal_stage.op_setup_ticks),
        pipeline_lookup_calls = metal_stage.pipeline_lookup_calls,
        pipeline_lookup_ms = ms(metal_stage.pipeline_lookup_ticks),
        encode_dispatch_calls = metal_stage.encode_dispatch_calls,
        encode_dispatch_ms = ms(metal_stage.encode_dispatch_ticks),
        gpu_exec_calls = metal_stage.gpu_exec_calls,
        gpu_exec_ms = ms(metal_stage.gpu_exec_ticks),
        readback_calls = metal_stage.readback_calls,
        readback_ms = ms(metal_stage.readback_ticks),
        readback_bytes = metal_stage.readback_bytes,
        nocopy_uploads = metal_stage.nocopy_uploads,
        copying_uploads = metal_stage.copying_uploads,
        nocopy_reuses = metal_stage.nocopy_reuses,
        resident_uploads = metal_stage.resident_uploads,
        resident_reuses = metal_stage.resident_reuses,
        mapping_offset_uploads = metal_stage.mapping_offset_uploads,
        mapping_rebound_blocks = metal_stage.mapping_rebound_blocks,
        mapping_residency_rung = crate::mapping_residency::active_residency_rung_str(),
        expert_mapping_candidate_uploads = metal_stage.expert_mapping_candidate_uploads,
        expert_mapping_missed_uploads = metal_stage.expert_mapping_missed_uploads,
        nocopy_cache_len = omega::metal::nocopy_cache_len() as u64,
        resident_cache_len = omega::metal::resident_cache_len() as u64,
        resident_cache_bytes = omega::metal::resident_cache_bytes(),
        uniform_cache_len = omega::metal::uniform_cache_len() as u64,
        phys_footprint_bytes = phys_footprint_bytes(),
        device_allocated_bytes = omega::metal::current_allocated_size().unwrap_or(0),
        output_buffer_allocations = metal_stage.output_buffer_allocations,
        output_buffer_allocated_bytes = metal_stage.output_buffer_allocated_bytes,
        checkpoint_mapping_buffer_bytes,
        expert_mapping_buffer_bytes,
        plan_uniform_writes = metal_stage.plan_uniform_writes,
        barriers = metal_stage.barriers_emitted,
        barriers_raw = metal_stage.barriers_raw,
        barriers_waw = metal_stage.barriers_waw,
        barriers_war = metal_stage.barriers_war,
        barriers_waw_war_arena_recycled = metal_stage.barriers_waw_war_arena_recycled,
        barriers_waw_war_persistent = metal_stage.barriers_waw_war_persistent,
        expert_source_cache_hits = metal_stage.expert_source_cache_hits,
        expert_source_cache_misses = metal_stage.expert_source_cache_misses,
        expert_source_cache_cold_misses = metal_stage.expert_source_cache_cold_misses,
        expert_source_cache_replacement_misses = metal_stage.expert_source_cache_replacement_misses,
        expert_source_reuse_copy_bytes = metal_stage.expert_source_reuse_copy_bytes,
        expert_source_reuse_copy_ms = ms(metal_stage.expert_source_reuse_copy_ticks),
        plan_cache_len = plan_cache_len as u64,
        plan_hits = plan_hits as u64,
        plan_misses = plan_misses as u64,
        plan_arena_allocated_bytes = arena_allocated_bytes.0 as u64,
        segment_arena_allocated_bytes = arena_allocated_bytes.1 as u64,
        placed_arena_allocated_bytes = arena_allocated_bytes.2 as u64,
        ablation = !kind_filter.is_empty(),
        kind_filter,
        "token_breakdown_metal: per-decode-step metal stage attribution"
    );
    if std::env::var_os("PROXIMA_DEBUG_METAL_STAGES").is_some() {
        eprintln!(
            "token_breakdown_metal step={} prepare_ms={} emit_ms={} op_setup_ms={} encode_dispatch_calls={} encode_dispatch_ms={} readback_ms={} expert_source_cache_hits={} expert_source_cache_misses={} expert_source_cache_cold_misses={} expert_source_cache_replacement_misses={} expert_source_buffer_reuses={} expert_source_reuse_copy_bytes={} expert_source_reuse_copy_ms={} plan_handoff_reuses={} expert_source_cache_entries={} nocopy_cache_entries={} resident_cache_entries={} resident_cache_bytes={} block_upload_calls={} block_upload_ms={} block_copied_bytes={} block_nocopy_bound_bytes={} block_offset_bound_bytes={} mapping_offset_uploads={} mapping_rebound_blocks={} mapping_residency_rung={} expert_mapping_candidate_uploads={} expert_mapping_missed_uploads={} resident_uploads={} resident_reuses={} output_buffer_allocations={} output_buffer_allocated_bytes={} checkpoint_mapping_buffer_bytes={} expert_mapping_buffer_bytes={} plan_uniform_writes={} barriers={} barriers_raw={} barriers_waw={} barriers_war={} barriers_waw_war_arena_recycled={} barriers_waw_war_persistent={} plan_cache_len={} plan_hits={} plan_misses={} plan_arena_allocated_bytes={} segment_arena_allocated_bytes={} placed_arena_allocated_bytes={} gpu_exec_calls={} gpu_exec_ms={} phys_footprint_bytes={} device_allocated_bytes={}",
            step,
            ms(metal_stage.prepare_ticks),
            ms(metal_stage.emit_ticks),
            ms(metal_stage.op_setup_ticks),
            metal_stage.encode_dispatch_calls,
            ms(metal_stage.encode_dispatch_ticks),
            ms(metal_stage.readback_ticks),
            metal_stage.expert_source_cache_hits,
            metal_stage.expert_source_cache_misses,
            metal_stage.expert_source_cache_cold_misses,
            metal_stage.expert_source_cache_replacement_misses,
            metal_stage.expert_source_buffer_reuses,
            metal_stage.expert_source_reuse_copy_bytes,
            ms(metal_stage.expert_source_reuse_copy_ticks),
            metal_stage.plan_handoff_reuses,
            metal_stage.expert_source_cache_entries,
            metal_stage.nocopy_cache_entries,
            omega::metal::resident_cache_len(),
            omega::metal::resident_cache_bytes(),
            metal_stage.block_upload_calls,
            ms(metal_stage.block_upload_ticks),
            metal_stage.block_copied_bytes,
            metal_stage.block_nocopy_bound_bytes,
            metal_stage.block_offset_bound_bytes,
            metal_stage.mapping_offset_uploads,
            metal_stage.mapping_rebound_blocks,
            crate::mapping_residency::active_residency_rung_str(),
            metal_stage.expert_mapping_candidate_uploads,
            metal_stage.expert_mapping_missed_uploads,
            metal_stage.resident_uploads,
            metal_stage.resident_reuses,
            metal_stage.output_buffer_allocations,
            metal_stage.output_buffer_allocated_bytes,
            checkpoint_mapping_buffer_bytes,
            expert_mapping_buffer_bytes,
            metal_stage.plan_uniform_writes,
            metal_stage.barriers_emitted,
            metal_stage.barriers_raw,
            metal_stage.barriers_waw,
            metal_stage.barriers_war,
            metal_stage.barriers_waw_war_arena_recycled,
            metal_stage.barriers_waw_war_persistent,
            plan_cache_len,
            plan_hits,
            plan_misses,
            arena_allocated_bytes.0,
            arena_allocated_bytes.1,
            arena_allocated_bytes.2,
            metal_stage.gpu_exec_calls,
            ms(metal_stage.gpu_exec_ticks),
            phys_footprint_bytes(),
            omega::metal::current_allocated_size().unwrap_or(0),
        );
        if step == 0 {
            eprintln!(
                "nocopy_cache_lengths={:?}",
                omega::metal::nocopy_cache_lengths()
            );
            eprintln!("resident_cache_lengths_top={:?}", {
                let mut entries = omega::metal::resident_cache_lengths();
                entries.sort_unstable_by_key(|(_, bytes)| core::cmp::Reverse(*bytes));
                entries.into_iter().take(24).collect::<Vec<_>>()
            });
        }
    }
}

/// ROW 391's single load-time answer to "how much device memory does this
/// checkpoint actually use": one line, emitted once (`step == 0`) from each
/// decode arm right after that step's own [`metal_stage_totals`] snapshot,
/// naming every device-allocation class this crate can attribute directly
/// from an existing counter rather than a new one (principle 1, reuse
/// first) -- `weights_nocopy_bytes`/`weights_copied_bytes`/
/// `weights_offset_bytes` are [`omega::metal::MetalStageTotals`]'s own
/// `block_nocopy_bound_bytes`/`block_copied_bytes`/`block_offset_bound_bytes`,
/// which on step 0 hold this call's ONE-TIME resident weight upload (every
/// later step's own snapshot falls to ~0 once a name is cached -- see
/// [`emit_token_breakdown_metal`]'s own `block_upload_calls` doc).
/// `kv_cache_device_bytes` is [`None`] on the two-range (`LayerCache`) arm,
/// whose `kv_cache.{layer}.*` blocks are just more named blocks folded into
/// the SAME weight counters above, not a separately sized allocation --
/// only [`Self::run_decode_loop_placed_kv`]'s fixed-capacity
/// `k_even`/`k_odd`/`v` buffers are a distinct, independently sized class.
/// `device_allocated_bytes`/`phys_footprint_bytes` are the same
/// process-wide totals [`emit_token_breakdown_metal`] already reports, so a
/// reader can subtract the named classes from `device_allocated_bytes` and
/// see the unattributed remainder (attention scratch, the `BufferArena`'s
/// per-op output buffers, uniform buffers, fault buffers -- proxima-tensor
/// discipline.md ROW 391) directly, without this crate re-deriving each of
/// those privately-sized classes here.
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_device_memory_by_class(
    step: usize,
    weights_nocopy_bytes: u64,
    weights_copied_bytes: u64,
    weights_offset_bytes: u64,
    kv_cache_device_bytes: Option<u64>,
    device_allocated_bytes: u64,
    phys_footprint_bytes: u64,
) {
    info!(
        step = step as u64,
        weights_nocopy_bytes,
        weights_copied_bytes,
        weights_offset_bytes,
        kv_cache_device_bytes = kv_cache_device_bytes.unwrap_or(0),
        kv_cache_device_bytes_known = kv_cache_device_bytes.is_some(),
        device_allocated_bytes,
        phys_footprint_bytes,
        "device_memory_by_class: load-time device allocation broken out by class"
    );
}

/// `PROXIMA_METAL_KIND_FILTER` -- `omega::metal::execute_plan_with_placements`'s
/// own in-buffer ablation knob (that function's own `KindFilter` doc has the
/// full contract). Read here, at [`emit_token_breakdown_metal`]'s own emit
/// site, rather than threaded back from `omega::metal` through
/// `MetalStageTotals`: it is a per-call, caller-supplied env value, not a
/// device-side measurement, and every other diagnostic env knob on this
/// decode path (`PROXIMA_METAL_OP_PROFILE_STEP`) is likewise read directly at
/// its own emit site rather than plumbed through the stage-totals struct.
///
/// Cached in a `OnceLock` -- same idiom as
/// `omega::backend::Backend::from_env` and
/// `proxima_tensor::cpu::matmul_worker_count` -- because the telemetry tag
/// type only accepts `&'static str` (no per-emit allocation on the hot
/// path), and the env var cannot change once the process has started.
/// Empty (`ablation=false`, `kind_filter=""`) when unset, so a production
/// run's event is unchanged from before this ablation existed.
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
pub(super) fn kind_filter_from_env() -> &'static str {
    static KIND_FILTER: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    KIND_FILTER
        .get_or_init(|| std::env::var("PROXIMA_METAL_KIND_FILTER").unwrap_or_default())
        .as_str()
}

#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
pub(super) fn stats_pass(entry: &mut FamilyGpuStats, gpu_ns: u64, operand_bytes: u64) {
    entry.row_blocked_count += 1;
    entry.passed_gpu_ns += gpu_ns;
    entry.passed_operand_bytes += operand_bytes;
    entry.packed_row_block_gates.insert("PASS".to_string());
}

#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
pub(super) fn stats_reject(entry: &mut FamilyGpuStats, rejection: &str, gpu_ns: u64, operand_bytes: u64) {
    entry.rejected_count += 1;
    entry.rejected_gpu_ns += gpu_ns;
    entry.rejected_operand_bytes += operand_bytes;
    entry.packed_row_block_gates.insert(rejection.to_string());
}

/// One tensor family's aggregated per-op GPU cost across every layer that
/// carries it, plus the DISTINCT set of [`OpGpuTiming::packed_row_block_rejection`]
/// verdicts its ops reported -- printed as a set (rather than one value)
/// because a family's own gate can legitimately vary by shape (a family's
/// `Some("PASS")` alongside a rejection would mean SOME layers took the
/// fast path and others did not, which the aggregate ns/byte alone cannot
/// show).
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
pub(super) struct FamilyGpuStats {
    pub(super) op_count: u64,
    pub(super) gpu_ns: u64,
    pub(super) operand_bytes: u64,
    pub(super) min_operand_count: usize,
    pub(super) max_operand_count: usize,
    pub(super) row_blocked_count: u64,
    pub(super) rejected_count: u64,
    /// Sum of `gpu_ns`/`operand_bytes` over exactly this family's
    /// row-blocked (`PASS`) ops -- the already-packed slice, isolated from
    /// [`Self::rejected_gpu_ns`] so a mixed family's aggregate `gpu_ns`
    /// (which sums both) is never mistaken for either codec's own cost.
    pub(super) passed_gpu_ns: u64,
    pub(super) passed_operand_bytes: u64,
    /// Sum of `gpu_ns`/`operand_bytes` over exactly this family's rejected
    /// ops. Before `Q5_K`'s own row-blocked kernel landed, `ffn_down`/
    /// `attn_v` each carried 4 rejected ops (`NotExactlyOnePackedOperand`
    /// on a weight the loader had already dequantized back to plain
    /// `f32`) -- this field is what isolated that codec's own cost from
    /// the 28 already-fast `Q4_K` ops sharing its family (ROW 92). Kept
    /// as a general split rather than a one-off measurement: any future
    /// codec gap in a mixed family reproduces this exact shape.
    pub(super) rejected_gpu_ns: u64,
    pub(super) rejected_operand_bytes: u64,
    pub(super) packed_row_block_gates: BTreeSet<String>,
}

#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
impl Default for FamilyGpuStats {
    fn default() -> Self {
        Self {
            row_blocked_count: 0,
            rejected_count: 0,
            op_count: 0,
            gpu_ns: 0,
            operand_bytes: 0,
            min_operand_count: usize::MAX,
            max_operand_count: 0,
            passed_gpu_ns: 0,
            passed_operand_bytes: 0,
            rejected_gpu_ns: 0,
            rejected_operand_bytes: 0,
            packed_row_block_gates: BTreeSet::new(),
        }
    }
}

/// [`report_op_timings`]'s own alias for [`omega::metal::weight_family`] --
/// `omega::metal::KindFilter`'s `family:` term reuses the SAME function, so
/// there is exactly one place that strips a `blk.N.*` weight name's layer
/// index, not a copy per caller.
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
pub(super) fn strip_layer_index(name: &str) -> String {
    omega::metal::weight_family(name)
}

/// A checkpoint's weights, bound once from a caller-owned byte view, plus
/// its compiled cached forward program -- everything a generation request
/// needs that does not change between requests. Borrows `file_bytes` for
/// `'file` rather than owning it, matching the rest of this crate's
/// sans-IO discipline (this crate never opens a file itself): the caller
/// keeps its own `mmap`/`Vec<u8>` alive for as long as it holds a
/// `LoadedModel` borrowed from it.
pub struct LoadedModel<'file> {
    pub(super) weights: BoundWeights<'file>,
    pub(super) architecture: ModelArchitecture,
    /// [`Self::load`]'s resolved [`crate::architecture::Architecture`] impl,
    /// kept so [`Self::run_decode_loop_observed_seeded`] can call
    /// [`Architecture::step_inputs`] every step -- `None` only on the
    /// narrow `load_inner` fallthrough that predates the registry seam
    /// (`paired_gate_up_reduce`/`fused_qkv_reduce` on a non-qwen35
    /// checkpoint), which never resolves against an `Architecture` at all.
    pub(super) architecture_impl: Option<&'static dyn Architecture>,
    /// This checkpoint's own weight bytes, by class
    /// (`crate::bind::tensor_bytes_by_class`'s own dense/expert/table
    /// split, plus the SSM state bytes a qwen35 checkpoint's layers hold)
    /// -- kept as a plain [`crate::memory_fit::WeightClassBytes`] rather
    /// than re-borrowing `file_bytes`/`parsed` themselves, since
    /// [`Self::generate_with_serving_config`]'s own load-time memory-fit
    /// gate (`crate::memory_fit`) needs these byte counts long after
    /// `load_inner`'s local `parsed`/`file_bytes` bindings have gone out of
    /// scope. `cfg`-gated with [`Self::apply_memory_fit_gate`], its only
    /// reader.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(super) checkpoint_weight_bytes: crate::memory_fit::WeightClassBytes,
    pub(super) vocab: Vocab,
    pub(super) program: Vec<Op>,
    pub(super) logits_root: NodeId,
    /// `proxima_tensor::spec::ForwardRoots::hidden` off the dense load path
    /// (`Self::load`/`Self::load_from_safetensors`, both wrapping
    /// `mistral_cached_forward_program_with_experts`) -- `None` on the
    /// qwen35 hybrid path (`crate::qwen35::qwen35_forward_program` returns
    /// a bare `logits` root with no named hidden-state counterpart yet).
    pub(super) hidden_root: Option<NodeId>,
    /// `general.name` off the checkpoint's own metadata ([`Self::load`]/
    /// [`Self::load_with_paired_gate_up_reduce`]/[`Self::load_with_fused_qkv_reduce`]),
    /// `None` for [`Self::load_from_safetensors`] (HF's `config.json` has no
    /// equivalent key this crate reads) or a GGUF checkpoint that omits the
    /// key outright. Display-only -- see [`Self::model_name`].
    pub(super) model_name: Option<String>,
    /// `file_bytes.len()` at load time -- see [`Self::checkpoint_bytes`].
    pub(super) checkpoint_bytes: usize,
    /// One entry per forward-program layer, in layer order --
    /// [`Qwen35LayerRoots::Attention`] for every layer on the dense path
    /// (`Self::load`/`Self::load_from_safetensors` wrap
    /// `mistral_cached_forward_program_with_experts`'s own
    /// [`CachedLayerRoots`] in that variant so both checkpoint families
    /// share one cache-threading loop, [`Self::run_decode_loop`]), and a mix
    /// of [`Qwen35LayerRoots::Attention`]/[`Qwen35LayerRoots::Ssm`] on the
    /// qwen35 path (`crate::qwen35::qwen35_forward_program`'s own return).
    pub(super) layer_roots: Vec<Qwen35LayerRoots>,
    /// One post-layer residual root per dense layer, when the forward
    /// builder exposes them (`crate::architecture::BoundProgram::residual_roots`'s
    /// own doc) -- empty on the qwen35 hybrid path, which has no single
    /// per-layer residual node. See [`Self::layer_residual_roots`].
    pub(super) residual_roots: Vec<NodeId>,
    /// Graph-level producer boundaries for each qwen35moe layer.
    pub(super) qwen35moe_layer_diagnostics: Vec<crate::qwen35moe::Qwen35MoeLayerDiagnostics>,
    /// Router-logit roots aligned with routed layers.  Qwen35MoE fills this
    /// from the same graph nodes used by its gather; other architectures leave
    /// it empty.  These roots are the concrete input to a future per-layer
    /// pre-gather evaluator, not a second router computation.
    pub(super) router_roots: Vec<NodeId>,
    /// One [`proxima_tensor::spec::MoeSite`] per MoE layer this
    /// checkpoint's forward-program builder produced -- empty on a dense
    /// checkpoint. [`Self::run_decode_loop_observed`] reads this to know
    /// which extra nodes to request as step outputs when a routing observer
    /// (`proxima_tensor::instrument::ExpertObserver`, `instrument`-gated)
    /// is registered.
    pub(super) moe_sites: proxima_tensor::spec::MoeSites,
    /// `crate::architecture::BoundProgram::single_position_step` off this
    /// checkpoint's own resolved [`crate::Architecture`] (`Self::load`'s
    /// `resolved.bind(..)` for the registry path; `false` for every other
    /// load entry point below, all of which wrap
    /// `mistral_cached_forward_program_with_experts`) -- read by
    /// [`Self::run_decode_loop_observed_seeded`] to decide whether prefill
    /// batches its whole prompt into one evaluation or feeds it one
    /// position at a time.
    pub(super) single_position_step: bool,
    /// This checkpoint's own qwen35moe hparams, re-derived from `parsed`'s
    /// metadata alone (no weight bytes -- `crate::qwen35moe::hparams::from_metadata`'s
    /// own doc) at the same registry bind site that already called it once
    /// inside `crate::qwen35moe::qwen35moe_forward_program`. `None` for
    /// every other architecture. [`Self::run_decode_loop_observed_seeded`]'s
    /// own prefill batch reads this to build a SECOND, `Extent::Static`-width
    /// program via `crate::qwen35moe::qwen35moe_forward_program_at_width`
    /// on demand -- see `proxima_tensor::spec::append_qwen35_ssm_mixer_with_taps_and_layout`'s
    /// own doc on why only a literal static width ever reaches its M>1
    /// branch -- and swaps it into `program`/`logits_root`/`layer_roots`/
    /// `single_position_step` for exactly that one evaluation, restoring
    /// the ordinary `Extent::Symbolic(0)` decode program right after.
    pub(super) qwen35moe_hparams: Option<crate::qwen35moe::hparams::Architecture>,
    /// The single-range, device-resident-KV counterpart of `program`/
    /// `logits_root`/`layer_roots` above -- `None` unless this build was
    /// compiled with `metal-output-placement` AND this checkpoint took the
    /// dense, non-qwen35, non-MoE path (the single-range program is
    /// dense-only, see [`mistral_single_range_cached_forward_program`]'s
    /// own doc). CPU decode, any MoE checkpoint, and the qwen35 hybrid path
    /// always run the two-range `program`/`layer_roots` fields instead;
    /// [`Self::run_decode_loop`] picks whichever this field's presence and
    /// the runtime backend selection together allow.
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    pub(super) single_range: Option<SingleRangeProgram>,
    /// The same `file_bytes` slice [`Self::load`]/[`Self::load_inner`]
    /// registered with `omega::backend::register_checkpoint_mapping` (GGUF
    /// checkpoints only -- [`Self::load_from_safetensors`] never registers
    /// one, so this is just the raw byte view there). Kept so `Drop` can
    /// hand the identical `(pointer, length)` identity back to
    /// `omega::backend::unregister_checkpoint_mapping`, which is the only
    /// thing that lets it tell "this is still my mapping" apart from "a
    /// second model already superseded it" -- see that function's own doc.
    /// Named distinctly from [`Self::checkpoint_bytes`] (the `usize` byte
    /// COUNT the memory-fit gate reports) since this is the byte VIEW
    /// itself, not a count -- the two coexist on this struct for different
    /// readers.
    // only `Drop` (below) reads this, and `Drop` is itself `metal`-gated --
    // a `std`-only, non-`metal` build has no device buffer to release, so
    // the field is genuinely dead weight there rather than a leftover
    // `_` this cfg_attr is hiding a real bug behind.
    #[cfg_attr(
        not(feature = "metal"),
        allow(dead_code, reason = "only `Drop`, itself `metal`-gated, reads this")
    )]
    pub(super) checkpoint_mapping: &'file [u8],
    /// This checkpoint's own paged/aliased MoE expert bytes -- see
    /// [`crate::expert_slab::ExpertSlab`]'s own module doc for the
    /// ownership contract. [`crate::bind::build_expert_slab`] builds it once
    /// at bind time from `weights`/`program`'s own `_exps.weight` nodes;
    /// empty for a dense checkpoint. Behind a [`std::sync::Mutex`], not a
    /// bare [`core::cell::RefCell`]: `examples/openai_serve_gguf.rs` holds a
    /// `LoadedModel` inside an `Arc` and serves it from a multi-threaded
    /// `SendPipe` (`proxima-primitives/src/pipe/primitives.rs`'s own
    /// `Send + Sync + 'static` bound), so this type must stay `Sync` --
    /// `RefCell` is not, `Mutex` is (`cargo check --workspace --all-targets`
    /// is what caught the `RefCell` attempt failing that example's own
    /// build). `proxima_lock::Mutex` (this workspace's canonical
    /// tier-resolved mutex, principle 21) does not exist as a crate in this
    /// repo -- grepped for it before falling back to `std::sync::Mutex`
    /// here, which is legitimate per that principle's own tier-3 case: a
    /// synchronous lock guarding a step boundary no code ever holds across
    /// an `.await`. Every acquire recovers from poisoning
    /// (`unwrap_or_else(PoisonError::into_inner)`) rather than panicking --
    /// this crate's own no-panic rule -- since a poisoned lock here would
    /// mean an earlier panic mid-decode already violated that rule
    /// somewhere else; recovering is strictly better than a second panic.
    pub(super) expert_slab: std::sync::Mutex<crate::expert_slab::ExpertSlab<'file>>,
    /// HOBBIT's optional low-codec store and its mmap owner. Attachment
    /// installs the low copies once; router boundaries subsequently switch
    /// only the selected expert's three projection entries.
    pub(super) expert_sidecar: Option<crate::expert_sidecar::MappedExpertSidecar>,
}

/// Concrete qwen35moe router/gather partitions for one KV shape bucket.
///
/// GDN prefill must remain sequential by position, but the graph cuts do
/// not change while `(new_count, kv_bound_extent)` is unchanged. Keeping
/// the dense partitions beside that concrete shape prevents every prompt
/// position from repeating partitioning, topological ordering, and shape
/// inference before it can execute the same router/residency/gather phases.
pub(super) struct Qwen35MoePreGatherPlan {
    pub(super) symbols: Vec<u64>,
    pub(super) gdn_scan_enabled: bool,
    pub(super) gdn_backend: GdnPrefillBackend,
    pub(super) persistent_cuts: bool,
    pub(super) layers: Vec<Qwen35MoeLayerSegments>,
    pub(super) suffix: crate::qwen35moe::execution::MappedLayerSegment,
    pub(super) prefix_carried_nodes: BTreeSet<NodeId>,
    pub(super) global_cut_nodes: BTreeSet<NodeId>,
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    pub(super) router_cut_placements: BTreeMap<NodeId, PlacedBuffer>,
}

#[cfg(feature = "qwen35moe-expert-prefetch")]
#[derive(Clone, Copy)]
pub(super) struct Qwen35MoeRouteHistory {
    pub(super) routes: [crate::residency::RoutedExpert; 16],
    pub(super) len: usize,
}

#[cfg(feature = "qwen35moe-expert-prefetch")]
impl Default for Qwen35MoeRouteHistory {
    fn default() -> Self {
        Self {
            routes: [crate::residency::RoutedExpert {
                expert: 0,
                importance: 0.0,
            }; 16],
            len: 0,
        }
    }
}

#[cfg(feature = "qwen35moe-expert-prefetch")]
pub(super) fn qwen35moe_expert_prefetch_requested(value: bool) -> bool {
    value
}

#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
pub(super) struct Qwen35DenseAttentionBuffers {
    pub(super) k_first: PlacedBuffer,
    pub(super) k_second: PlacedBuffer,
    pub(super) k_pass: PlacedBuffer,
    pub(super) value: PlacedBuffer,
    pub(super) even_odd_row_bytes: usize,
    pub(super) pass_row_bytes: usize,
    pub(super) value_row_bytes: usize,
}

#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
pub(super) struct Qwen35DenseAttentionPlacement<'buffers> {
    pub(super) input_nodes: &'buffers [Option<(NodeId, NodeId, NodeId, NodeId)>],
    pub(super) buffers: &'buffers [Option<Qwen35DenseAttentionBuffers>],
}

// device residency is a backend property, not a pre-gather-mode property
// (ROW 531 invariant 2): the full-graph decode arm places these roots too.
#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
pub(super) fn qwen35_dense_attention_placement_enabled(
    is_qwen35moe: bool,
    is_metal: bool,
    force_two_range: bool,
    seed_cached_len: usize,
    requested: bool,
) -> bool {
    is_qwen35moe && is_metal && !force_two_range && seed_cached_len == 0 && requested
}

#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
pub(super) fn qwen35_dense_attention_placed_byte_length(
    positions: usize,
    row_elements: usize,
    layer: usize,
    leaf: &'static str,
) -> Result<usize, InteropError> {
    positions
        .checked_mul(row_elements)
        .and_then(|elements| elements.checked_mul(core::mem::size_of::<f32>()))
        .ok_or_else(|| InteropError::PreGatherExecutionUnsupported {
            architecture: String::from("qwen35moe"),
            reason: alloc::format!(
                "layer {layer} dense-attention {leaf} placed buffer size overflowed"
            ),
        })
}

#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
pub(super) fn retain_qwen35_segment_readbacks<RouterPlacement>(
    requested: &mut BTreeMap<NodeId, NodeId>,
    router_cut_placements: &BTreeMap<NodeId, RouterPlacement>,
    placed_dense_roots: Option<(NodeId, NodeId, NodeId, NodeId)>,
    program: &[Op],
) {
    requested.retain(|mapped, original| {
        if placed_dense_roots
            .is_some_and(|roots| [roots.0, roots.1, roots.2, roots.3].contains(original))
        {
            return false;
        }
        if router_cut_placements.contains_key(original) {
            // A linked gather-next-router segment can contain both the
            // boundary input and the next router's computed output. Only the
            // input side is already resident; computed outputs stay in the
            // placed-result set so the kernel writes them first.
            return !matches!(program.get(mapped.0 as usize), Some(Op::Input { .. }));
        }
        true
    });
}

pub(super) struct Qwen35MoeLayerSegments {
    pub(super) router: crate::qwen35moe::execution::MappedLayerSegment,
    pub(super) gather: crate::qwen35moe::execution::MappedLayerSegment,
    /// The graph segment from this layer's router through its gather and the
    /// next layer's router.  It is kept separate from `gather`: the former
    /// makes the layer boundary explicit for a future double-buffered
    /// working set, while the latter remains the fallback when a recurrent
    /// placement or a final suffix prevents boundary batching.
    pub(super) gather_next_router: Option<crate::qwen35moe::execution::MappedLayerSegment>,
    /// Exact two-layer window, present on the first layer of each pair when
    /// the caller requests `qwen35moe_layer_window=2`.
    pub(super) layer_window: Option<crate::qwen35moe::execution::MappedLayerSegment>,
    pub(super) gdn_scan: Option<Qwen35MoeGdnScanSegment>,
    pub(super) router_future_cuts: Vec<(NodeId, String)>,
    pub(super) next_cuts: Vec<(NodeId, String)>,
    /// Original gather cuts owned by later layers.  This is part of the
    /// linked schedule rather than an execution-time scan: retaining these
    /// values keeps a layer's router/gather boundary connected to the next
    /// consumer without rebuilding a suffix vector for every prompt row.
    pub(super) future_gather_cuts: Vec<NodeId>,
}

#[derive(Clone)]
pub(super) struct Qwen35MoeGdnScanSegment {
    pub(super) producer: crate::qwen35moe::execution::MappedLayerSegment,
    pub(super) tail: crate::qwen35moe::execution::MappedLayerSegment,
    pub(super) taps: proxima_tensor::spec::SsmMixerTaps,
    pub(super) prefill: crate::qwen35moe::Qwen35MoeGdnPrefillTaps,
    pub(super) post_mixer_residual: NodeId,
    pub(super) post_attention_norm_output: NodeId,
    pub(super) router_logits: NodeId,
    /// The ordinary (non-scan) `previous_output -> router_logits` segment for
    /// this same layer -- identical to what [`Qwen35MoeLayerSegments::router`]
    /// would hold if `gdn_scan_enabled` were false. The scan's own conv branch
    /// (`causal_conv1d` over just this call's rows) has no cross-call history
    /// input, so it is only valid for the one call that carries the model's
    /// entire causal context to date (the initial multi-position prefill).
    /// Every later single-token decode step must run through this segment
    /// instead, which reads the persisted `ssm_cache.{layer}.conv_history`/
    /// `.state` the ordinary [`SsmLayerCache`] already threads.
    pub(super) decode_router: crate::qwen35moe::execution::MappedLayerSegment,
}

/// Links one layer's carry set to the gather cuts consumed by later layers.
/// The link is plan data: execution must not rediscover the suffix while a
/// prompt row is crossing the router/residency boundary.
pub(super) fn collect_future_gather_cuts(layer: usize, gather_cuts: &[Vec<(NodeId, String)>]) -> Vec<NodeId> {
    gather_cuts[layer + 1..]
        .iter()
        .flat_map(|cuts| cuts.iter().map(|(node, _)| *node))
        .collect()
}

pub(super) fn fused_segment_experts_are_current_layer(
    segment: &crate::qwen35moe::execution::MappedLayerSegment,
    layer: usize,
) -> bool {
    segment
        .0
        .iter()
        .filter_map(Op::name)
        .filter(|name| name.contains("_exps.weight"))
        .all(|name| {
            name.strip_prefix("blk.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|value| value.parse::<usize>().ok())
                == Some(layer)
        })
}

/// Releases every device buffer this checkpoint's own load caused: the
/// no-copy/resident-copy buffers keyed under [`Self::resident_names`], and
/// (GGUF checkpoints only -- see [`Self::checkpoint_mapping`]'s own doc) the
/// whole-mapping no-copy buffer `omega::backend::register_checkpoint_mapping`
/// registered. Both releases are BY NAME/IDENTITY, never a blanket cache
/// clear, so a second `LoadedModel` loaded on this same thread keeps its
/// own weights resident regardless of drop order -- see
/// `omega::metal::release_resident_names`'s and
/// `omega::metal::unregister_checkpoint_mapping`'s own docs for the
/// mechanism. A no-op unless this build was compiled with the `metal`
/// feature: the CPU evaluator has no device buffer to release.
#[cfg(feature = "metal")]
impl Drop for LoadedModel<'_> {
    fn drop(&mut self) {
        release_resident_names(self.resident_names().iter().copied());
        unregister_checkpoint_mapping(self.checkpoint_mapping);
    }
}

/// The first `program` [`Op::Input`] leaf `named` carries no entry for --
/// the same name-resolution [`proxima_tensor::cpu::resolve_named_blocks`]
/// performs internally (and would itself error on), surfaced here as the
/// typed, architecture-facing [`InteropError::MissingStepInput`] instead
/// of the generic [`proxima_tensor::TensorError::UnboundInputName`] a
/// caller several layers down [`BackendRuntime::evaluate`] would otherwise
/// see: a program leaf beyond the decode loop's own builtin set is, by
/// construction, one [`crate::architecture::Architecture::step_inputs`]
/// either never fed or fed with the wrong name.
pub(super) fn missing_program_input(program: &[Op], named: &[(&str, QuantizedBlock<'_>)]) -> Option<String> {
    program.iter().find_map(|op| match op {
        Op::Input {
            name: Some(name), ..
        } if !named.iter().any(|(bound_name, _)| bound_name == name) => Some(name.clone()),
        _ => None,
    })
}

/// [`mistral_single_range_cached_forward_program`]'s compiled output, plus
/// the one thing that function's own signature does not return: the
/// per-layer `kv_cache.{layer}.{k_even,k_odd,v}` [`Op::Input`] node ids
/// (`cache_input_nodes`) that a placed *read* targets -- [`CachedLayerRoots`]
/// already names the OUTPUT (freshly rotated key/value) nodes a placed
/// *write* targets, but the input side has no equivalent public return, so
/// [`locate_cache_input_nodes`] resolves them by the same
/// `kv_cache.{layer}.*` name `LayerCache`'s own two-range binding already
/// uses.
#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
pub(super) struct SingleRangeProgram {
    pub(super) program: Vec<Op>,
    pub(super) logits_root: NodeId,
    pub(super) cache_roots: Vec<CachedLayerRoots>,
    pub(super) cache_input_nodes: Vec<(NodeId, NodeId, NodeId)>,
    /// ROW 326 diagnostic: `Some` only when `PROXIMA_DUPLICATE_HEAD=1` was
    /// set at load time (`build_single_range_program`'s own doc) -- a
    /// second, identical `output.weight` reduce added to this call's own
    /// requested outputs so its GPU cost is directly measurable as the
    /// A/B delta against a normal run. `None` in every production run.
    pub(super) duplicate_head_scratch: Option<NodeId>,
}

/// Scans `program` for the [`Op::Input`] node named `name` -- the input-side
/// counterpart [`SingleRangeProgram::cache_input_nodes`] needs and
/// [`CachedLayerRoots`] does not carry (see that field's own doc). O(program
/// length) per call, paid `block_count * 3` times, once at
/// [`LoadedModel::load`] time, never per decode step.
#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
pub(super) fn find_input_node(program: &[Op], name: &str) -> Result<NodeId, InteropError> {
    program
        .iter()
        .enumerate()
        .find(|(_, op)| op.name() == Some(name))
        .map(|(index, _)| NodeId(index as u32))
        .ok_or_else(|| InteropError::UnboundInputName(String::from(name)))
}

/// Builds [`SingleRangeProgram`] for a checkpoint whose
/// `architecture.expert_count == 0` -- `None` for any mixture-of-experts
/// checkpoint, since [`mistral_single_range_cached_forward_program`] is
/// dense-only (that function's own doc). Never called for a qwen35
/// checkpoint: [`LoadedModel::load`]'s qwen35 branch returns before this
/// function's own call site is reached.
///
/// # Errors
/// Whatever [`mistral_single_range_cached_forward_program`] can fail with,
/// or [`InteropError::UnboundInputName`] if a `kv_cache.{layer}.*` name this
/// function expects the program to declare is somehow absent (would mean
/// the program builder and this lookup have drifted out of sync).
#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
pub(super) fn build_single_range_program(
    architecture: &ModelArchitecture,
    qk_norm: bool,
) -> Result<Option<SingleRangeProgram>, InteropError> {
    if architecture.expert_count != 0 {
        return Ok(None);
    }
    // ROW 326/328 diagnostic: same env-knob convention as `PROXIMA_PREFAULT`/
    // `PROXIMA_MLOCK` (`proxima-model-interop/src/bind.rs`'s
    // `real_openchat_file` module) -- unset in every production run, so
    // `duplicate_head` is `DuplicateHeadPosition::None` on every path but
    // this one opt-in probe. `1` reproduces ROW 326's original after-the-
    // real-head position; `first` is ROW 328's new before-layer-0 position
    // (`DuplicateHeadPosition`'s own doc has the mechanism each arm tests).
    let duplicate_head = match std::env::var("PROXIMA_DUPLICATE_HEAD").as_deref() {
        Ok("1") => DuplicateHeadPosition::After,
        Ok("first") => DuplicateHeadPosition::Before,
        _ => DuplicateHeadPosition::None,
    };
    // ROW 373: `qk_norm` now builds correctly through this path -- the
    // builder derives its per-head norm and RoPE pairing from `qk_norm`
    // itself (`append_mistral_single_range_cached_layer`'s own doc), so a
    // qk-norm checkpoint takes the placed-KV fast path rather than falling
    // back to `Ok(None)`. `Err(TensorError::UnsupportedInBuilder)` still
    // maps to `Ok(None)` below for whatever this builder genuinely cannot
    // express (still dense-only -- MoE is turned away above).
    let (program, logits_root, cache_roots, duplicate_head_scratch) =
        match mistral_single_range_cached_forward_program(
            architecture.vocab,
            architecture.embedding,
            architecture.feed_forward,
            architecture.query_heads,
            architecture.kv_heads,
            architecture.head_dim,
            architecture.block_count,
            qk_norm,
            duplicate_head,
            true,
        ) {
            Ok(built) => built,
            Err(TensorError::UnsupportedInBuilder { .. }) => return Ok(None),
            Err(other) => return Err(InteropError::from(other)),
        };
    let block_count = architecture.block_count as usize;
    let mut cache_input_nodes = Vec::with_capacity(block_count);
    for layer in 0..block_count {
        let even = find_input_node(&program, &alloc::format!("kv_cache.{layer}.k_even"))?;
        let odd = find_input_node(&program, &alloc::format!("kv_cache.{layer}.k_odd"))?;
        let value = find_input_node(&program, &alloc::format!("kv_cache.{layer}.v"))?;
        cache_input_nodes.push((even, odd, value));
    }
    Ok(Some(SingleRangeProgram {
        program,
        logits_root,
        cache_roots,
        cache_input_nodes,
        duplicate_head_scratch,
    }))
}

/// The KV extent both cached-attention decode paths bind as this step's
/// `Extent::Symbolic(1)` (the KV `Op::Input` leaves' shape, and half the
/// backend's own plan-cache key alongside `new_count`) -- `merged_len`
/// rounded up to `bucket_tokens` (`ServingConfig::kv_bucket_tokens`) and
/// capped at `capacity` (each caller's own per-layer buffer row count,
/// never exceeded regardless of rounding; the two-range path below has no
/// fixed buffer to cap against, so it passes `usize::MAX`). `bucket_tokens
/// == 1` reduces to `merged_len` unchanged: `div_ceil(1) * 1` is the
/// identity, so the plan-cache key is untouched from its pre-bucketing
/// shape whenever a caller disables bucketing.
/// [`run_decode_loop_placed_kv`]'s own `causal_mask_merged`
/// `key_index > query_absolute` comparison masks every row in `[merged_len,
/// extent)` as invalid for every query the single-range path issues
/// (`proxima-tensor`'s `spec.rs`'s `cpu_mask_zero_ulp` test proves the
/// [`causal_mask_merged`] mechanism 0-ULP-safe for any bucket size); the
/// two-range path excludes the identical padding with a runtime BOUND on
/// the fused `BoundOpKind::CachedAttention` op instead of a mask node
/// (`proxima_tensor::bind::cached_attention_candidates`'s own doc on the
/// `cached_key_rows != 0` discriminator), so no other call site needs to
/// know which bucket size is configured either way. Plain `usize`
/// arithmetic, no platform or feature dependency of its own -- available to
/// any `std`-gated caller regardless of which backend feature is compiled
/// in.
pub(super) fn kv_extent(merged_len: usize, capacity: usize, bucket_tokens: usize) -> usize {
    merged_len
        .div_ceil(bucket_tokens)
        .saturating_mul(bucket_tokens)
        .min(capacity)
}

pub(super) const fn step_batch_needs_logits(split_prefill: bool, is_last_step_batch: bool) -> bool {
    !split_prefill || is_last_step_batch
}

/// The device-resident input and output buffer placements for one segment
/// evaluation, grouped so the method they feed keeps its argument count
/// under clippy's threshold.
#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
pub(super) struct SegmentPlacements<'placements> {
    pub(super) input_placements: &'placements [(NodeId, &'placements PlacedBuffer, usize)],
    pub(super) output_placements: &'placements [(NodeId, &'placements PlacedBuffer, usize)],
}

/// The per-call evaluation inputs for one qwen35moe pre-gather pass, grouped
/// so the method they feed keeps its argument count under clippy's
/// threshold. `'mapping` matches the borrow the sidecar and its scratch hold
/// across a layer window; `'file` matches the bound model's own file borrow.
pub(super) struct PreGatherContext<'context, 'mapping, 'file> {
    pub(super) named: &'context [(&'context str, QuantizedBlock<'context>)],
    pub(super) outputs: &'context [NodeId],
    pub(super) resident_names: &'context BTreeSet<&'context str>,
    pub(super) layer_caches: &'context [LayerCacheState],
    pub(super) expert_slab: &'context mut crate::expert_slab::ExpertSlab<'file>,
    pub(super) sidecar_read_scratch: &'context mut crate::expert_sidecar::ExpertSidecarReadScratch,
    pub(super) current_sources: &'context RefCell<CurrentExpertSources>,
    pub(super) position_offset: usize,
    pub(super) layer_window: usize,
    pub(super) gdn_backend: GdnPrefillBackend,
    // `'mapping` is only borrowed by metal-gated fields below; this marker
    // keeps the lifetime parameter used on every feature combination.
    pub(super) marker: PhantomData<&'mapping ()>,
    #[cfg(feature = "metal")]
    pub(super) sidecar: Option<&'mapping crate::expert_sidecar::MappedExpertSidecar>,
    #[cfg(feature = "metal")]
    pub(super) all_low_expert_scratch: &'context mut crate::expert_slab::AllLowExpertSourceScratch<'mapping>,
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    pub(super) ssm_placement: Option<&'context Qwen35SsmPlacement<'context>>,
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    pub(super) dense_attention_placement: Option<&'context Qwen35DenseAttentionPlacement<'context>>,
}

/// The per-call evaluation inputs for one GDN scan segment, grouped so the
/// method they feed keeps its argument count under clippy's threshold.
pub(super) struct GdnScanSegmentContext<'context, 'data> {
    pub(super) state_cache: &'context [f32],
    pub(super) gdn_backend: GdnPrefillBackend,
    pub(super) future_cuts: &'context [(NodeId, String)],
    pub(super) symbols: &'context [u64],
    pub(super) named: &'context [(&'context str, QuantizedBlock<'data>)],
    pub(super) outputs: &'context [NodeId],
    pub(super) resident_names: &'context BTreeSet<&'context str>,
    pub(super) carried: &'context mut BTreeMap<NodeId, (Vec<u64>, Vec<f32>)>,
    pub(super) results: &'context mut BTreeMap<NodeId, (Vec<u64>, Vec<f32>)>,
}

