//! Reachable text generation: bind a checkpoint's weights once, then
//! generate text repeatedly against the bound weights without re-paying
//! the load cost.
//!
//! [`LoadedModel`] is the transform pipe (`In = (String, usize), Out =
//! (Vec<u32>, String, bool)` -- `proxima_primitives::pipe::Pipe`,
//! `proxima-primitives/src/pipe/primitives.rs:91-102`'s general form,
//! since neither `In` nor `Out` is `()`). [`LoadedModel::load`] is a plain
//! constructor, not a pipe: it pays the expensive one-time cost (mmap +
//! parse + bind 226 tensors, ~4 GB / ~120 ms prefault on the real
//! openchat-3.5 checkpoint -- `crate::bind::bind_all_weights`'s own doc)
//! and hands back a value that [`Pipe::call`] is then cheap to invoke many
//! times against, one call per generation request, without rebinding.
//! That two-step shape is the direct answer to "load once, generate
//! repeatedly": a caller holds one `LoadedModel` and calls it as many
//! times as it wants, exactly the way a caller holds one bound
//! `TcpListener` and accepts many connections from it.
//!
//! `call`'s body is synchronous CPU work wrapped in `async move { .. }`
//! with no internal `.await` -- the same shape `Pipe`'s own doc's
//! `Double`/`Always`/`Discard`/`Echo` examples use. It is still the right
//! trait: the algebra's whole point is that combinators (retry, tee,
//! rate-limit, ...) compose over `Pipe` regardless of whether a given
//! impl happens to yield control anywhere inside.
//!
//! # Stopping: the model's own signal, not just the caller's budget
//!
//! `Out`'s third field is `true` exactly when decoding stopped because the
//! model emitted its own end-of-sequence token, `false` when it stopped
//! because `max_tokens` ran out first -- the two are otherwise
//! indistinguishable to a caller (`generated_ids.len() < max_tokens` is
//! not proof of an early stop if `max_tokens` itself was small). A plain
//! `bool` earns this over a new enum because this checkpoint's own
//! metadata defines exactly one stopping condition to check, confirmed by
//! reading it rather than assumed: on the real openchat-3.5-1210 fixture
//! (`~/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf`),
//! `tokenizer.ggml.eos_token_id = 32000`, which is *not* the SentencePiece
//! `</s>` (id 2) -- it is `<|end_of_turn|>`, a [`proxima_tokenizer::vocab::TokenType::Control`]
//! entry, and the same id OpenChat's own `tokenizer.chat_template` emits
//! between turns. There is no separate `tokenizer.ggml.eot_token_id` (or
//! similar) key on this fixture; the GGUF writer already folded the
//! turn-boundary marker into the one `eos_token_id` slot
//! [`proxima_tokenizer::Vocab::eos_token_id`] reads. So checking a single
//! id against [`Vocab::eos_token_id`] is this fixture's whole stopping
//! condition -- a `bool` carries it exactly; an enum would be modeling a
//! multi-token-family case this checkpoint does not have.
//!
//! The stop token itself is excluded from both the returned ids and the
//! returned text (never pushed onto `generated_ids` before the loop
//! breaks) -- symmetric exclusion, not just from decoded text, because a
//! caller who re-feeds `generated_ids` as a future prompt's tokens should
//! never see a turn-boundary marker reappear as if it were generated
//! content.

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;
use core::future::Future;

use proxima_gguf::GgmlType;
use proxima_gguf::pipe::ParsedGguf;
use proxima_primitives::pipe::Pipe;
#[cfg(not(feature = "metal"))]
use proxima_tensor::cpu::{
    evaluate_quantized_named_exact_with_scratch, evaluate_quantized_named_with_scratch,
};
use proxima_tensor::cpu::{Evaluated, QuantizedBlock};
use proxima_tensor::op::{NodeId, Op};
#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
use proxima_tensor::spec::CachedLayerRoots;
use proxima_tensor::spec::{Qwen35LayerRoots, mistral_cached_forward_program_with_experts};
use proxima_tokenizer::{SamplingConfig, Vocab, sample_next_token};

#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
use omega::backend::execute_plan_named_metal_op_timed;
#[cfg(feature = "metal")]
use omega::backend::{
    Engine, Plan, execute_plan_named, mark_resident, plan_named, plan_named_exact,
    release_resident_names, unregister_checkpoint_mapping,
};
// `set_math_mode` (unlike `mark_resident` above) takes `metal::MathMode` in
// its own signature, so unlike the ungated import above it needs the same
// `metal`+macos gate that type itself lives behind.
#[cfg(all(feature = "metal", target_os = "macos"))]
use omega::backend::set_math_mode;
// `set_dispatch_type` (unlike `mark_resident` above) takes `metal::DispatchType`
// in its own signature, so it needs the same `metal`+macos gate that type
// itself lives behind -- same reasoning as `set_math_mode` above.
#[cfg(all(feature = "metal", target_os = "macos"))]
use omega::backend::set_dispatch_type;
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
use omega::metal::OpGpuTiming;
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
use omega::metal::metal_stage_totals;
// Persistent device-resident KV: `PlacedBuffer`/`allocate_placed_buffer`/
// `execute_plan_named_with_placements` are `omega`'s own default-off
// `metal-output-placement` surface (`omega/src/metal.rs`'s own doc on
// `execute_plan_with_placements`) -- this crate's identically-named,
// identically default-off feature is a straight passthrough
// (`Cargo.toml`'s `metal-output-placement` entry), never a second gate.
// `plan_named` here is `omega::metal`'s own (the Metal-`Plan`-typed one,
// aliased to avoid colliding with `omega::backend::plan_named` above,
// which returns the backend-polymorphic `omega::backend::Plan` enum
// `PlacedBuffer` placement has no arm for) --
// `mistral_single_range_cached_forward_program` is this call's program
// builder, `proxima-tensor/src/spec.rs`'s own single-range counterpart to
// `mistral_cached_forward_program_with_experts`.
#[cfg(all(
    feature = "metal-output-placement",
    feature = "instrument",
    target_os = "macos"
))]
use omega::execute_plan_named_with_placements_dispatch_timed;
#[cfg(all(
    feature = "metal-output-placement",
    feature = "instrument",
    target_os = "macos"
))]
use omega::execute_plan_named_with_placements_op_timed;
#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
use omega::{
    PlacedBuffer, allocate_placed_buffer, execute_plan_named_with_placements,
    plan_named as plan_named_placed,
};
#[cfg(feature = "instrument")]
use proxima_telemetry::{debug, info};
#[cfg(feature = "instrument")]
use proxima_tensor::instrument::{elapsed_ticks, read_ticks, ticks_to_nanos};
#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
use proxima_tensor::TensorError;
#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
use proxima_tensor::spec::{DuplicateHeadPosition, mistral_single_range_cached_forward_program};

use crate::bind::{BoundWeights, ModelArchitecture, architecture_from_metadata, bind_all_weights};
use crate::error::InteropError;
use crate::hf_bind::bind_all_weights_from_safetensors;
#[cfg(feature = "metal")]
use crate::serving::GPU_LAYERS_ALL;
use crate::serving::ServingConfig;
use crate::serving::apply_serving_config;

/// How many of [`OpGpuTiming`]'s entries [`report_op_timings`] names
/// individually -- the discipline log's own "top 20 ops by GPU time" ask.
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
const OP_PROFILE_TOP_N: usize = 20;

/// ROW 329: `PROXIMA_METAL_ENCODER_SPLIT_AT`, read the same
/// unset-means-off, one-env-var-per-diagnostic-knob convention as
/// `PROXIMA_DUPLICATE_HEAD`/`PROXIMA_METAL_OP_PROFILE_STEP` above --
/// `None` (unset, or unparseable) keeps
/// `execute_plan_with_placements_dispatch_timed`'s ROW 309 default
/// (one stage-boundary encoder per position); `Some(position)` ends the
/// compute encoder immediately before that plan position and opens a
/// second one for the rest of the program, in the SAME command buffer.
#[cfg(all(feature = "metal-output-placement", feature = "instrument", target_os = "macos"))]
fn encoder_split_at_from_env() -> Option<usize> {
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
#[cfg(all(feature = "metal-output-placement", feature = "instrument", target_os = "macos"))]
fn report_encoder_split(step: usize, encoder_split_ns: (u64, u64), gpu_exec_ns: u64) {
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
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
fn report_op_timings(step: usize, timings: &[OpGpuTiming]) {
    let op_count = timings.len();
    let total_gpu_ns: u64 = timings.iter().map(|timing| timing.gpu_ns).sum();
    let total_operand_bytes: u64 = timings.iter().map(|timing| timing.operand_bytes).sum();

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
        info!(
            step = step as u64,
            variant = *variant,
            op_count = *count,
            gpu_ms = *ns as f64 / 1e6,
            gpu_ns_per_op = *ns as f64 / *count as f64,
            "op_profile_variant: gpu op-timing bucketed by packed kernel variant"
        );
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
fn phys_footprint_bytes() -> u64 {
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
struct TokenBreakdown {
    step: usize,
    new_count: usize,
    cached_len_before: usize,
    step_wall_ticks: u64,
    apply_serving_config_ticks: u64,
    build_position_inputs_ticks: u64,
    named_blocks_weights_ticks: u64,
    named_blocks_kv_ticks: u64,
    kv_cache_upload_bytes: u64,
    evaluate_ticks: u64,
    layer_cache_append_ticks: u64,
    layer_cache_append_bytes: u64,
    greedy_pick_ticks: u64,
}

#[cfg(feature = "instrument")]
fn emit_token_breakdown(breakdown: &TokenBreakdown) {
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
        evaluate_ms = ms(evaluate_ticks),
        layer_cache_append_ms = ms(layer_cache_append_ticks),
        layer_cache_append_bytes,
        greedy_pick_ms = ms(greedy_pick_ticks),
        "token_breakdown: per-decode-step wall-clock attribution"
    );
}

/// [`emit_token_breakdown`]'s Metal-stage counterpart -- same sharing
/// rationale, same two call sites. `metal_stage` is this step's own
/// snapshot-and-reset delta ([`metal_stage_totals`]'s own doc), so it is
/// correct to call from either decode arm as long as it is read exactly
/// once per step, immediately after that step's `evaluate`/
/// `evaluate_with_placements` call.
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
fn emit_token_breakdown_metal(
    step: usize,
    metal_stage: &omega::metal::MetalStageTotals,
    plan_cache_len: usize,
    plan_hits: usize,
    plan_misses: usize,
) {
    let ms = |ticks: u64| ticks_to_nanos(ticks) as f64 / 1e6;
    let kind_filter = kind_filter_from_env();
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
        nocopy_cache_len = omega::metal::nocopy_cache_len() as u64,
        uniform_cache_len = omega::metal::uniform_cache_len() as u64,
        phys_footprint_bytes = phys_footprint_bytes(),
        device_allocated_bytes = omega::metal::current_allocated_size().unwrap_or(0),
        output_buffer_allocations = metal_stage.output_buffer_allocations,
        plan_uniform_writes = metal_stage.plan_uniform_writes,
        barriers = metal_stage.barriers_emitted,
        plan_cache_len = plan_cache_len as u64,
        plan_hits = plan_hits as u64,
        plan_misses = plan_misses as u64,
        ablation = !kind_filter.is_empty(),
        kind_filter,
        "token_breakdown_metal: per-decode-step metal stage attribution"
    );
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
fn emit_device_memory_by_class(
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
fn kind_filter_from_env() -> &'static str {
    static KIND_FILTER: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    KIND_FILTER
        .get_or_init(|| std::env::var("PROXIMA_METAL_KIND_FILTER").unwrap_or_default())
        .as_str()
}

#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
fn stats_pass(entry: &mut FamilyGpuStats, gpu_ns: u64, operand_bytes: u64) {
    entry.row_blocked_count += 1;
    entry.passed_gpu_ns += gpu_ns;
    entry.passed_operand_bytes += operand_bytes;
    entry.packed_row_block_gates.insert("PASS".to_string());
}

#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
fn stats_reject(entry: &mut FamilyGpuStats, rejection: &str, gpu_ns: u64, operand_bytes: u64) {
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
struct FamilyGpuStats {
    op_count: u64,
    gpu_ns: u64,
    operand_bytes: u64,
    min_operand_count: usize,
    max_operand_count: usize,
    row_blocked_count: u64,
    rejected_count: u64,
    /// Sum of `gpu_ns`/`operand_bytes` over exactly this family's
    /// row-blocked (`PASS`) ops -- the already-packed slice, isolated from
    /// [`Self::rejected_gpu_ns`] so a mixed family's aggregate `gpu_ns`
    /// (which sums both) is never mistaken for either codec's own cost.
    passed_gpu_ns: u64,
    passed_operand_bytes: u64,
    /// Sum of `gpu_ns`/`operand_bytes` over exactly this family's rejected
    /// ops. Before `Q5_K`'s own row-blocked kernel landed, `ffn_down`/
    /// `attn_v` each carried 4 rejected ops (`NotExactlyOnePackedOperand`
    /// on a weight the loader had already dequantized back to plain
    /// `f32`) -- this field is what isolated that codec's own cost from
    /// the 28 already-fast `Q4_K` ops sharing its family (ROW 92). Kept
    /// as a general split rather than a one-off measurement: any future
    /// codec gap in a mixed family reproduces this exact shape.
    rejected_gpu_ns: u64,
    rejected_operand_bytes: u64,
    packed_row_block_gates: BTreeSet<String>,
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
fn strip_layer_index(name: &str) -> String {
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
    weights: BoundWeights<'file>,
    architecture: ModelArchitecture,
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
    checkpoint_weight_bytes: crate::memory_fit::WeightClassBytes,
    vocab: Vocab,
    program: Vec<Op>,
    logits_root: NodeId,
    /// `general.name` off the checkpoint's own metadata ([`Self::load`]/
    /// [`Self::load_with_paired_gate_up_reduce`]/[`Self::load_with_fused_qkv_reduce`]),
    /// `None` for [`Self::load_from_safetensors`] (HF's `config.json` has no
    /// equivalent key this crate reads) or a GGUF checkpoint that omits the
    /// key outright. Display-only -- see [`Self::model_name`].
    model_name: Option<String>,
    /// `file_bytes.len()` at load time -- see [`Self::checkpoint_bytes`].
    checkpoint_bytes: usize,
    /// One entry per forward-program layer, in layer order --
    /// [`Qwen35LayerRoots::Attention`] for every layer on the dense path
    /// (`Self::load`/`Self::load_from_safetensors` wrap
    /// `mistral_cached_forward_program_with_experts`'s own
    /// [`CachedLayerRoots`] in that variant so both checkpoint families
    /// share one cache-threading loop, [`Self::run_decode_loop`]), and a mix
    /// of [`Qwen35LayerRoots::Attention`]/[`Qwen35LayerRoots::Ssm`] on the
    /// qwen35 path (`crate::qwen35::qwen35_forward_program`'s own return).
    layer_roots: Vec<Qwen35LayerRoots>,
    /// [`Some`] only for a qwen35-architecture checkpoint -- the SSM cache
    /// shapes [`SsmLayerCache::new`] needs (`Self::run_decode_loop`'s own
    /// per-layer state-space cache), derived once at load time rather than
    /// recomputed every decode step. `None` on the dense path, which never
    /// has an [`Qwen35LayerRoots::Ssm`] entry to size.
    qwen35_ssm_shape: Option<Qwen35SsmShape>,
    /// [`Some`] only for a qwen35-architecture checkpoint --
    /// `{architecture}.attention.key_length` (`crate::qwen35::Qwen35Architecture::attn_head_dim`'s
    /// own doc on why this is not derivable from [`ModelArchitecture::head_dim`],
    /// the PARTIAL-rotary width). [`Self::run_decode_loop_observed`]'s
    /// two-range loop needs it to size a [`Qwen35DenseAttentionCache`]'s own
    /// `k_pass`/`v` row widths when padding a `DenseAttention` layer's cache
    /// out to a `kv_extent` bucket boundary, the same reason
    /// [`Self::run_decode_loop_observed`] already carries `qwen35_ssm_shape`
    /// as a separate field rather than re-deriving it from `layer_roots`.
    qwen35_attn_head_dim: Option<u32>,
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
    single_range: Option<SingleRangeProgram>,
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
    checkpoint_mapping: &'file [u8],
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
struct SingleRangeProgram {
    program: Vec<Op>,
    logits_root: NodeId,
    cache_roots: Vec<CachedLayerRoots>,
    cache_input_nodes: Vec<(NodeId, NodeId, NodeId)>,
    /// ROW 326 diagnostic: `Some` only when `PROXIMA_DUPLICATE_HEAD=1` was
    /// set at load time (`build_single_range_program`'s own doc) -- a
    /// second, identical `output.weight` reduce added to this call's own
    /// requested outputs so its GPU cost is directly measurable as the
    /// A/B delta against a normal run. `None` in every production run.
    duplicate_head_scratch: Option<NodeId>,
}

/// Scans `program` for the [`Op::Input`] node named `name` -- the input-side
/// counterpart [`SingleRangeProgram::cache_input_nodes`] needs and
/// [`CachedLayerRoots`] does not carry (see that field's own doc). O(program
/// length) per call, paid `block_count * 3` times, once at
/// [`LoadedModel::load`] time, never per decode step.
#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
fn find_input_node(program: &[Op], name: &str) -> Result<NodeId, InteropError> {
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
fn build_single_range_program(
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
fn kv_extent(merged_len: usize, capacity: usize, bucket_tokens: usize) -> usize {
    merged_len
        .div_ceil(bucket_tokens)
        .saturating_mul(bucket_tokens)
        .min(capacity)
}

/// [`SsmLayerCache`]'s own fixed sizes, all derived from
/// [`crate::qwen35::Qwen35Architecture`]'s ssm hyperparameters at load time
/// -- `qwen35.cpp:57-60`'s same derivation
/// `crate::qwen35::bind_qwen35_attn_qkv_split`'s own doc already walks
/// through for the fused `attn_qkv.weight` split.
#[derive(Debug, Clone, Copy)]
struct Qwen35SsmShape {
    /// `2 * ssm_key_dim + ssm_d_inner` -- one `qkv_mixed` row's width,
    /// matching `proxima_tensor::spec::qwen35_forward_program`'s own
    /// `ssm_cache.{layer}.conv_history` leaf shape's second axis.
    qkv_dim: usize,
    /// `ssm_d_conv - 1` -- the rolling conv-history window's fixed row
    /// count [`append_qwen35_ssm_mixer`]'s doc names (the causal conv1d
    /// kernel's own left-context width).
    conv_rows: usize,
    /// `ssm_d_state * head_v_dim * ssm_n_group * ssm_group` -- the gated
    /// DeltaNet recurrent state's flat element count, matching
    /// `qwen35_forward_program`'s own `ssm_cache.{layer}.state` leaf shape.
    state_len: usize,
}

impl<'file> LoadedModel<'file> {
    /// `true` when [`Self::load`] built a device-resident, single-range
    /// program for this checkpoint ([`Self::single_range`]'s own doc) --
    /// the ROW 373 evidence hook: a qk-norm (Qwen3) checkpoint returned
    /// `false` here before that row (rejected by
    /// `append_mistral_single_range_cached_layer`, `build_single_range_program`
    /// falling back to `Ok(None)`) and returns `true` after it, WITHOUT this
    /// crate's own decode output changing (`Self::run_decode_loop`'s own
    /// `Some(single_range)` branch is what a Metal decode step then takes).
    #[cfg(all(test, feature = "metal-output-placement", target_os = "macos"))]
    pub(crate) fn takes_placed_kv_path(&self) -> bool {
        self.single_range.is_some()
    }

    /// `general.name` off the checkpoint this call loaded, when the
    /// checkpoint declared one -- a live "what's running" indicator's own
    /// label ([`Self::model_name`]'s field doc). `None` on a safetensors
    /// checkpoint, or a GGUF checkpoint that omits the key.
    #[must_use]
    pub fn model_name(&self) -> Option<&str> {
        self.model_name.as_deref()
    }

    /// This checkpoint's own transformer block count
    /// (`{architecture}.block_count`, [`ModelArchitecture::block_count`]).
    #[must_use]
    pub fn layer_count(&self) -> u32 {
        self.architecture.block_count
    }

    /// The checkpoint file's own byte length at load time (`file_bytes.len()`
    /// passed to [`Self::load`]/[`Self::load_from_safetensors`]) -- the
    /// on-disk size a live indicator reports, not this call's resident
    /// memory footprint (weights may be memory-mapped rather than copied;
    /// see `crate::bind::bind_all_weights`'s own doc for which tensors are
    /// borrowed versus owned).
    #[must_use]
    pub fn checkpoint_bytes(&self) -> usize {
        self.checkpoint_bytes
    }

    /// Binds every weight the cached forward program needs out of
    /// `parsed`/`file_bytes` (`crate::bind::bind_all_weights`), derives
    /// [`ModelArchitecture`] from `parsed`'s own metadata
    /// ([`crate::bind::architecture_from_metadata`]), builds the vocab
    /// from the same metadata, and compiles the cached forward program
    /// once. Pays the whole load cost; every [`Pipe::call`] after reuses
    /// the result.
    ///
    /// # Errors
    ///
    /// Whatever [`crate::bind::architecture_from_metadata`],
    /// [`proxima_tokenizer::gguf::vocab_from_metadata`], or
    /// [`proxima_tensor::spec::mistral_cached_forward_program_with_experts`]
    /// can fail with.
    pub fn load(parsed: &ParsedGguf, file_bytes: &'file [u8]) -> Result<Self, InteropError> {
        Self::load_inner(parsed, file_bytes, false, false)
    }

    /// [`Self::load`] with the paired gate/up reduce
    /// (`proxima_tensor::spec::append_mistral_cached_layer`'s
    /// `paired_gate_up_reduce`) flipped on: one `Op::Reduce` per layer over
    /// `blk.{layer}.ffn_gate_up.weight` (`crate::bind::bind_matmul_weight_paired`)
    /// in place of today's two independent `ffn_gate`/`ffn_up` matvecs.
    /// `false` at [`Self::load`] reproduces this crate's forward program
    /// byte-for-byte; this constructor is the seam a caller (or a future
    /// [`crate::serving::ServingConfig`] field) flips to measure the other
    /// side. No effect on a `qwen35` checkpoint (that branch never reads
    /// this flag) or a mixture-of-experts checkpoint (routed FFN weights are
    /// untouched by this flag either way).
    ///
    /// # Errors
    ///
    /// Same as [`Self::load`].
    pub fn load_with_paired_gate_up_reduce(
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
        paired_gate_up_reduce: bool,
    ) -> Result<Self, InteropError> {
        Self::load_inner(parsed, file_bytes, paired_gate_up_reduce, false)
    }

    /// [`Self::load`] with the fused Q/K/V reduce
    /// (`proxima_tensor::spec::append_mistral_cached_layer`'s
    /// `fused_qkv_reduce`) flipped on: one `Op::Reduce` per layer over
    /// `blk.{layer}.attn_qkv.weight` (`crate::bind::bind_matmul_weight_triple`)
    /// in place of today's three independent `attn_q`/`attn_k`/`attn_v`
    /// matvecs. Unlike [`Self::load_with_paired_gate_up_reduce`], this does
    /// NOT remove dispatches -- q/k/v's three different GQA row counts mean
    /// none of them can be read back out of the shared reduce at zero
    /// extra cost (`append_mistral_cached_layer`'s `fused_qkv_reduce` doc
    /// traces the exact `shape::infer` limit this hits), so the measured
    /// effect, if any, is per-dispatch bandwidth on the one larger reduce,
    /// not fewer kernel launches. No effect on a `qwen35` checkpoint or a
    /// mixture-of-experts checkpoint, same carve-outs as the paired flag.
    ///
    /// # Errors
    ///
    /// Same as [`Self::load`].
    pub fn load_with_fused_qkv_reduce(
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
        fused_qkv_reduce: bool,
    ) -> Result<Self, InteropError> {
        Self::load_inner(parsed, file_bytes, false, fused_qkv_reduce)
    }

    fn load_inner(
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
        paired_gate_up_reduce: bool,
        fused_qkv_reduce: bool,
    ) -> Result<Self, InteropError> {
        // registers `file_bytes` -- the checkpoint's own mmap, page-aligned
        // at its base by construction -- as the single mapping every packed
        // tensor's borrowed slice can be addressed into by OFFSET instead of
        // copied into its own device buffer; see
        // `omega::metal::register_checkpoint_mapping`'s own doc. `omega` is
        // an optional dependency gated behind the `metal` feature, so this
        // registration is a no-op (compiled out) when that feature is off.
        #[cfg(feature = "metal")]
        omega::backend::register_checkpoint_mapping(file_bytes);
        // `crate::memory_fit`'s own load-time gate needs these three sums
        // long after `parsed` has gone out of scope -- computed once, here,
        // before any weight is bound, shared by both the qwen35 and dense
        // branches below.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let (dense_weight_bytes, expert_weight_bytes, table_weight_bytes) =
            crate::bind::tensor_bytes_by_class(parsed);
        // `general.architecture` read directly, before `architecture_from_metadata`
        // (which assumes the dense per-layer shape every other checkpoint this
        // crate binds has) -- qwen35's hybrid attention+state-space layers
        // (`crate::qwen35`'s own module doc) are not that shape, so this
        // checkpoint gets its own bind + forward-program seam instead of being
        // handed to the dense path, which would either fail bind on an SSM
        // layer's tensors or, worse, silently misbind them as dense attention.
        if crate::bind::metadata_str(parsed, "general.architecture")? == "qwen35" {
            let qwen_architecture = crate::qwen35::qwen35_architecture_from_metadata(parsed)?;
            let weights =
                crate::qwen35::bind_qwen35_weights(parsed, file_bytes, &qwen_architecture)?;
            let vocab = proxima_tokenizer::gguf::vocab_from_metadata(parsed)?;
            let (program, logits_root, layer_roots) =
                crate::qwen35::qwen35_forward_program(&qwen_architecture)?;
            let ssm_shape = qwen35_ssm_shape(&qwen_architecture);
            let architecture = ModelArchitecture {
                vocab: qwen_architecture.vocab,
                embedding: qwen_architecture.embedding,
                feed_forward: qwen_architecture.feed_forward,
                query_heads: qwen_architecture.query_heads,
                kv_heads: qwen_architecture.kv_heads,
                head_dim: qwen_architecture.head_dim,
                block_count: qwen_architecture.block_count,
                // Qwen3.5 never routes FFN through experts
                // (`crate::qwen35::qwen35_forward_program`'s own doc,
                // `qwen35.cpp:471`), so this checkpoint reads the same
                // `expert_count == 0` dense-FFN branch every other checkpoint
                // without a `{architecture}.expert_count` key does.
                expert_count: 0,
                expert_used_count: 0,
                rope_freq_base: qwen_architecture.rope_freq_base,
                rms_epsilon: qwen_architecture.rms_epsilon,
                tied_embeddings: false,
            };
            return Ok(Self {
                weights,
                architecture,
                #[cfg(all(feature = "metal", target_os = "macos"))]
                checkpoint_weight_bytes: crate::memory_fit::WeightClassBytes {
                    dense_bytes: dense_weight_bytes,
                    expert_bytes: expert_weight_bytes,
                    table_bytes: table_weight_bytes,
                    ssm_state_bytes: qwen35_ssm_state_bytes(
                        ssm_shape,
                        qwen_architecture.block_count,
                    ),
                },
                vocab,
                program,
                logits_root,
                layer_roots,
                model_name: crate::bind::metadata_str_opt(parsed, "general.name").map(String::from),
                checkpoint_bytes: file_bytes.len(),
                qwen35_ssm_shape: Some(ssm_shape),
                qwen35_attn_head_dim: Some(qwen_architecture.attn_head_dim),
                // The single-range program is dense-Mistral-only
                // (`SingleRangeProgram`'s own field doc); qwen35's hybrid
                // attention+state-space layers are never that shape.
                #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                single_range: None,
                checkpoint_mapping: file_bytes,
            });
        }

        let architecture = architecture_from_metadata(parsed)?;
        let vocab = proxima_tokenizer::gguf::vocab_from_metadata(parsed)?;
        // `&[]`: `Self::load`/`load_with_*` take no `ServingConfig`, so
        // there is no `weight_precision` rule set to thread here yet --
        // `crate::bind::bind_all_weights`'s own doc names this as the
        // wiring a future slice does.
        let weights = bind_all_weights(
            parsed,
            file_bytes,
            &architecture,
            paired_gate_up_reduce,
            fused_qkv_reduce,
            &[],
        )?;
        // `architecture.expert_count`/`expert_used_count` read `0` for every
        // dense checkpoint (`ModelArchitecture`'s own doc), which selects
        // exactly the dense program this crate has always built -- a
        // mixture-of-experts checkpoint (`expert_count > 0`) is the only case
        // that changes which program gets compiled here. `qk_norm` is Qwen3's
        // own per-head QK-norm (`crate::bind::checkpoint_has_qk_norm`'s own
        // doc) -- `false` reproduces the identical program this call has
        // always compiled for a checkpoint that carries no
        // `attn_q_norm.weight` tensor.
        let qk_norm = crate::bind::checkpoint_has_qk_norm(parsed);
        let (program, logits_root, cache_roots) = mistral_cached_forward_program_with_experts(
            architecture.vocab,
            architecture.embedding,
            architecture.feed_forward,
            architecture.query_heads,
            architecture.kv_heads,
            architecture.head_dim,
            architecture.block_count,
            architecture.expert_count,
            architecture.expert_used_count,
            qk_norm,
            paired_gate_up_reduce,
            fused_qkv_reduce,
        )?;
        // `mistral_single_range_cached_forward_program`'s own `w_gate`/`w_up`/
        // `wq`/`wk`/`wv` leaves (`build_single_range_program`) do not know
        // about `paired_gate_up_reduce`/`fused_qkv_reduce` yet --
        // `LoadedModel::run_decode_loop_observed`'s placed-KV fast path
        // would try to read `blk.{layer}.ffn_gate.weight`/
        // `blk.{layer}.attn_q.weight` against a `weights` set that, under
        // either flag, binds a differently-named fused tensor instead.
        // Forcing `None` here falls through to the two-range decode loop
        // below, which DOES thread both flags correctly, rather than a
        // `Metal(Tensor(UnboundInputName(..)))` panic -- the correct,
        // fusion-aware path over a crash, until the placed-KV builder
        // gains both flags (tracked, not done in this change: threading
        // `fused_qkv_reduce` through `mistral_single_range_cached_forward_program`
        // is a second builder needing the identical q/k/v leaf and
        // extract-op rewrite `append_mistral_cached_layer` just got, and is
        // out of this change's scope).
        //
        // `qk_norm` (ROW 373): `append_mistral_single_range_cached_layer`
        // (`proxima-tensor/src/spec.rs`) now takes the SAME
        // `Option<(NodeId, NodeId, NodeId)>` shape its two-range sibling
        // does and derives its RoPE pairing from `qk_norm.is_some()` the
        // same way -- a qk-norm checkpoint (Qwen3) now takes this
        // placed-KV fast path on Metal instead of falling back to the
        // two-range program (`build_single_range_program` no longer turns
        // a qk-norm request into `Ok(None)`; ROW 372's typed rejection is
        // gone from this builder). Before ROW 372, the same checkpoint
        // class silently ran attention on raw, un-normed Q/K with the wrong
        // RoPE pairing through this exact path -- no error, just a
        // structurally different (wrong) computation. ROW 372 made the
        // wrong path loud (reject); this row makes the right path fast
        // (build it correctly instead).
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        let single_range = if paired_gate_up_reduce || fused_qkv_reduce {
            None
        } else {
            build_single_range_program(&architecture, qk_norm)?
        };
        Ok(Self {
            weights,
            architecture,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            checkpoint_weight_bytes: crate::memory_fit::WeightClassBytes {
                dense_bytes: dense_weight_bytes,
                expert_bytes: expert_weight_bytes,
                table_bytes: table_weight_bytes,
                ssm_state_bytes: 0,
            },
            vocab,
            program,
            logits_root,
            layer_roots: cache_roots
                .into_iter()
                .map(Qwen35LayerRoots::Attention)
                .collect(),
            model_name: crate::bind::metadata_str_opt(parsed, "general.name").map(String::from),
            checkpoint_bytes: file_bytes.len(),
            qwen35_ssm_shape: None,
            qwen35_attn_head_dim: None,
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            single_range,
            checkpoint_mapping: file_bytes,
        })
    }

    /// [`Self::load`]'s HF/safetensors counterpart: binds every weight out
    /// of a single safetensors buffer's [`proxima_safetensors::Manifest`]
    /// (`crate::hf_bind::bind_all_weights_from_safetensors`) instead of a
    /// `ParsedGguf` tensor directory, and takes `architecture`/`vocab`
    /// already built rather than deriving them from the checkpoint itself --
    /// unlike GGUF, safetensors carries neither: `architecture` comes from
    /// `config.json` ([`crate::hf_config::architecture_from_hf_config`]),
    /// and HF's own vocabulary lives in `tokenizer.json`/`tokenizer_config.json`,
    /// files this crate has no reader for yet (out of scope for this
    /// change -- a caller builds its own [`Vocab`] however it can, the same
    /// way any [`Pipe`] caller owns its own setup-path inputs).
    ///
    /// `data_start` (`8 + header_len`) is the byte offset into `file_bytes`
    /// where tensor data begins -- `manifest`'s own `data_offsets` are
    /// relative to that point, never to the start of the file (see
    /// `crate::hf_bind::bind_all_weights_from_safetensors`'s doc); a
    /// caller who just parsed `file_bytes`'s header into `manifest` already
    /// has this value.
    ///
    /// # Errors
    ///
    /// [`InteropError::HfMoeWeightsUnsupported`] if `architecture.expert_count`
    /// is nonzero; otherwise whatever
    /// `crate::hf_bind::bind_all_weights_from_safetensors` or
    /// [`mistral_cached_forward_program_with_experts`] can fail with.
    pub fn load_from_safetensors(
        manifest: &proxima_safetensors::Manifest,
        file_bytes: &'file [u8],
        data_start: u64,
        architecture: ModelArchitecture,
        vocab: Vocab,
    ) -> Result<Self, InteropError> {
        let weights =
            bind_all_weights_from_safetensors(manifest, file_bytes, data_start, &architecture)?;
        // safetensors carries no GGUF tensor directory to probe for
        // `attn_q_norm.weight`, and no HF/safetensors checkpoint this crate
        // binds today needs QK-norm -- see [`Self::load`]'s own `qk_norm` for
        // the GGUF path that does.
        let (program, logits_root, cache_roots) = mistral_cached_forward_program_with_experts(
            architecture.vocab,
            architecture.embedding,
            architecture.feed_forward,
            architecture.query_heads,
            architecture.kv_heads,
            architecture.head_dim,
            architecture.block_count,
            architecture.expert_count,
            architecture.expert_used_count,
            false,
            false,
            false,
        )?;
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        let single_range = build_single_range_program(&architecture, false)?;
        Ok(Self {
            weights,
            architecture,
            // safetensors carries no `_exps.`-style naming convention this
            // crate has confirmed against a real checkpoint the way
            // `crate::bind::tensor_bytes_by_class` has for GGUF -- every
            // byte counts as dense here rather than guessing a split;
            // `expert_bytes`/`table_bytes` stay `0` until a real HF MoE
            // checkpoint proves what its own expert-tensor names look like.
            #[cfg(all(feature = "metal", target_os = "macos"))]
            checkpoint_weight_bytes: crate::memory_fit::WeightClassBytes {
                dense_bytes: file_bytes.len() as u64,
                expert_bytes: 0,
                table_bytes: 0,
                ssm_state_bytes: 0,
            },
            vocab,
            program,
            logits_root,
            layer_roots: cache_roots
                .into_iter()
                .map(Qwen35LayerRoots::Attention)
                .collect(),
            // safetensors carries no `general.name`-equivalent key this
            // crate reads (`Self::model_name`'s own doc).
            model_name: None,
            checkpoint_bytes: file_bytes.len(),
            qwen35_ssm_shape: None,
            qwen35_attn_head_dim: None,
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            single_range,
            checkpoint_mapping: file_bytes,
        })
    }
}

/// [`Qwen35SsmShape`]'s own derivation off a real checkpoint's ssm
/// hyperparameters -- `qwen35.cpp:57-60`'s same arithmetic
/// `crate::qwen35::Qwen35Architecture::ssm_key_dim`/`ssm_value_dim` already
/// use for the fused `attn_qkv.weight` row split, plus `head_v_dim =
/// ssm_inner_size / ssm_time_step_rank` and `ssm_group = ssm_time_step_rank
/// / ssm_group_count` (`proxima_tensor::spec::qwen35_forward_program`'s own
/// `head_v_dim`/`ssm_group` locals).
fn qwen35_ssm_shape(architecture: &crate::qwen35::Qwen35Architecture) -> Qwen35SsmShape {
    let ssm_key_dim = architecture.ssm_state_size * architecture.ssm_group_count;
    let head_v_dim = architecture.ssm_inner_size / architecture.ssm_time_step_rank;
    let ssm_group = architecture.ssm_time_step_rank / architecture.ssm_group_count;
    Qwen35SsmShape {
        qkv_dim: (2 * ssm_key_dim + architecture.ssm_inner_size) as usize,
        conv_rows: (architecture.ssm_conv_kernel.saturating_sub(1)) as usize,
        state_len: (architecture.ssm_state_size
            * head_v_dim
            * architecture.ssm_group_count
            * ssm_group) as usize,
    }
}

/// [`Qwen35SsmShape`]'s own resident bytes across every layer -- one
/// [`SsmLayerCache::new`]'s worth (`conv_rows * qkv_dim` conv-history
/// elements plus `state_len` state elements, both `f32`) times
/// `block_count` layers. `crate::memory_fit`'s own load-time gate reads
/// this as the SSM class of [`crate::memory_fit::WeightClassBytes`] -- `0`
/// for every non-qwen35 checkpoint, which never builds a
/// [`Qwen35SsmShape`] at all.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn qwen35_ssm_state_bytes(shape: Qwen35SsmShape, block_count: u32) -> u64 {
    let per_layer_elements = (shape.conv_rows * shape.qkv_dim + shape.state_len) as u64;
    per_layer_elements * core::mem::size_of::<f32>() as u64 * u64::from(block_count)
}

pub(crate) enum LogitsSink<'sink> {
    Discard,
    Collect(&'sink mut Vec<Vec<f32>>),
    /// `real_openchat_file::decode_text_is_deterministic_across_repeated_runs`'s
    /// own sink: that test is `#[cfg(feature = "metal")]`, so this variant
    /// carries the same gate (plus `test`) rather than sitting dead in a
    /// `std`-only test build.
    #[cfg(all(test, feature = "metal"))]
    SumBarriers(&'sink mut u64),
}

impl LogitsSink<'_> {
    /// `barriers_step` is this step's own `MetalStageTotals::barriers_emitted`
    /// (already read once, snapshot-and-reset, by the caller's `metal_stage`
    /// local) -- callers outside `feature = "instrument", feature = "metal",
    /// target_os = "macos"` pass `0`, matching every run where the counter
    /// itself never exists.
    fn observe(&mut self, logits: &[f32], _barriers_step: u64) {
        match self {
            Self::Discard => {}
            Self::Collect(buffer) => buffer.push(logits.to_vec()),
            #[cfg(all(test, feature = "metal"))]
            Self::SumBarriers(total) => **total += _barriers_step,
        }
    }
}

/// This call's growable per-layer key/value cache -- `F32` only:
/// [`apply_serving_config`]'s own gate rejects any other
/// `kv_cache_key_quant`/`kv_cache_value_quant` before [`LoadedModel::call`]
/// ever reaches this loop, so there is no second precision for this type
/// to carry (contrast `bind.rs`'s own `real_openchat_file::LayerCache`,
/// which still probes the rejected `Q8_0` path directly against the
/// tensor seam that gate exists to keep unreachable here).
struct LayerCache {
    k_even: Vec<f32>,
    k_odd: Vec<f32>,
    v: Vec<f32>,
}

impl LayerCache {
    fn new() -> Self {
        Self {
            k_even: Vec::new(),
            k_odd: Vec::new(),
            v: Vec::new(),
        }
    }

    fn append(&mut self, even: &[f32], odd: &[f32], value: &[f32]) {
        self.k_even.extend_from_slice(even);
        self.k_odd.extend_from_slice(odd);
        self.v.extend_from_slice(value);
    }

    fn named_blocks<'cache>(
        &'cache self,
        k_even_name: &'cache str,
        k_odd_name: &'cache str,
        v_name: &'cache str,
    ) -> [(&'cache str, QuantizedBlock<'cache>); 3] {
        [
            (k_even_name, QuantizedBlock::Float32(self.k_even.as_slice())),
            (k_odd_name, QuantizedBlock::Float32(self.k_odd.as_slice())),
            (v_name, QuantizedBlock::Float32(self.v.as_slice())),
        ]
    }
}

/// [`LayerCache`]'s bucket-padded mirror -- the two-range decode loop's own
/// fix for the plan-cache defect `Self::plans`' own doc on
/// [`BackendRuntime`] walks through: `LayerCache` grows by exactly
/// `cached_len` every step, so a plan keyed on it can never repeat, but the
/// fused `BoundOpKind::CachedAttention` op's own runtime bound
/// (`proxima_tensor::bind::cached_attention_candidates`'s own doc) makes any
/// reader that pads a copy of it out to a `kv_extent` bucket boundary
/// numerically identical to reading the exact, unpadded length. `fill`
/// copies `source`'s real content into a buffer at least `bound_extent` rows
/// long, zero-filling the remainder (never load-bearing -- the runtime bound
/// always excludes it before softmax, see that bound's own doc); `resize`
/// only grows when a step crosses into a new, larger bucket, the same
/// reuse-across-steps shape [`run_decode_loop_placed_kv`]'s own
/// single-range path's placeholder scratch.
struct KvPadScratch {
    k_even: Vec<f32>,
    k_odd: Vec<f32>,
    v: Vec<f32>,
}

impl KvPadScratch {
    fn new() -> Self {
        Self {
            k_even: Vec::new(),
            k_odd: Vec::new(),
            v: Vec::new(),
        }
    }

    fn fill(&mut self, source: &LayerCache, shape: &KvPadShape) {
        let even_odd_len = shape.even_odd_len();
        let v_len = shape.v_len();
        if self.k_even.len() < even_odd_len {
            self.k_even.resize(even_odd_len, 0.0);
        }
        if self.k_odd.len() < even_odd_len {
            self.k_odd.resize(even_odd_len, 0.0);
        }
        if self.v.len() < v_len {
            self.v.resize(v_len, 0.0);
        }
        self.k_even[..source.k_even.len()].copy_from_slice(&source.k_even);
        self.k_odd[..source.k_odd.len()].copy_from_slice(&source.k_odd);
        self.v[..source.v.len()].copy_from_slice(&source.v);
    }

    fn named_blocks<'cache>(
        &'cache self,
        k_even_name: &'cache str,
        k_odd_name: &'cache str,
        v_name: &'cache str,
        shape: &KvPadShape,
    ) -> [(&'cache str, QuantizedBlock<'cache>); 3] {
        [
            (
                k_even_name,
                QuantizedBlock::Float32(&self.k_even[..shape.even_odd_len()]),
            ),
            (
                k_odd_name,
                QuantizedBlock::Float32(&self.k_odd[..shape.even_odd_len()]),
            ),
            (
                v_name,
                QuantizedBlock::Float32(&self.v[..shape.v_len()]),
            ),
        ]
    }
}

/// [`KvPadScratch`]'s own row-width parameters, grouped into one reference
/// rather than four positional `usize`s -- every [`KvPadScratch::fill`]/
/// [`KvPadScratch::named_blocks`] call site already computes all four
/// together from `self.architecture`/`kv_bound_extent`, so one reference
/// says what was already true by convention, and keeps both methods under
/// clippy's `too_many_arguments` threshold without an `#[allow]`.
struct KvPadShape {
    bound_extent: usize,
    kv_heads: usize,
    pairs: usize,
    head_dim: usize,
}

impl KvPadShape {
    fn even_odd_len(&self) -> usize {
        self.bound_extent * self.kv_heads * self.pairs
    }

    fn v_len(&self) -> usize {
        self.bound_extent * self.kv_heads * self.head_dim
    }
}

/// [`LayerCache`]'s 4-wide counterpart for a
/// [`Qwen35LayerRoots::DenseAttention`] layer -- this checkpoint's own
/// partial-rotary gap (`proxima_tensor::spec::append_qwen35_dense_attention_layer`'s
/// own doc) needs a third K component (`k_pass`, the untouched
/// `rotary_dim..attn_head_dim` remainder) alongside the rotated
/// `k_first`/`k_second` halves [`LayerCache`]'s `k_even`/`k_odd` already
/// name for the plain single-section-RoPE checkpoints.
struct Qwen35DenseAttentionCache {
    k_first: Vec<f32>,
    k_second: Vec<f32>,
    k_pass: Vec<f32>,
    v: Vec<f32>,
}

impl Qwen35DenseAttentionCache {
    fn new() -> Self {
        Self {
            k_first: Vec::new(),
            k_second: Vec::new(),
            k_pass: Vec::new(),
            v: Vec::new(),
        }
    }

    fn append(&mut self, first: &[f32], second: &[f32], pass: &[f32], value: &[f32]) {
        self.k_first.extend_from_slice(first);
        self.k_second.extend_from_slice(second);
        self.k_pass.extend_from_slice(pass);
        self.v.extend_from_slice(value);
    }
}

/// [`KvPadShape`]'s counterpart for a [`Qwen35DenseAttentionCache`] --
/// `k_first`/`k_second` share [`KvPadShape::even_odd_len`]'s row width
/// (both are `pairs`-wide, the same rotary half [`spec::append_qwen35_dense_attention_layer`]'s
/// `k_first_cache`/`k_second_cache` leaves declare), but `k_pass`/`v` are
/// `attn_head_dim`-based, not `head_dim`-based, so they need their own
/// widths rather than reusing [`KvPadShape::v_len`].
struct Qwen35DenseAttentionPadShape {
    bound_extent: usize,
    kv_heads: usize,
    pairs: usize,
    pass_dim: usize,
    attn_head_dim: usize,
}

impl Qwen35DenseAttentionPadShape {
    fn even_odd_len(&self) -> usize {
        self.bound_extent * self.kv_heads * self.pairs
    }

    fn pass_len(&self) -> usize {
        self.bound_extent * self.kv_heads * self.pass_dim
    }

    fn v_len(&self) -> usize {
        self.bound_extent * self.kv_heads * self.attn_head_dim
    }
}

/// [`KvPadScratch`]'s counterpart for a [`Qwen35LayerRoots::DenseAttention`]
/// layer -- the same defect [`KvPadScratch`]'s own doc names
/// (`cached_len` growing 1:1 with the step index defeats
/// `proxima_tensor::bind::cached_attention_candidates`'s plan-key
/// bucketing unless the bound buffer this layer feeds the backend is padded
/// out to the SAME `kv_extent` boundary the `Attention` arm already pads
/// to) applies here identically: `qwen35_forward_program`'s dense-attention
/// layers declare `k_first_cache`/`k_second_cache`/`k_pass_cache`/`v_cache`
/// on the identical `Extent::Symbolic(1)` slot the `Attention` arm's
/// `kv_cache.{layer}.*` leaves use, so a bucketed `symbols[1]` value only
/// works when EVERY layer's bound buffer -- dense-attention included --
/// is actually that many rows long, zero-padded past the real
/// `cached_len`.
struct Qwen35DenseAttentionPadScratch {
    k_first: Vec<f32>,
    k_second: Vec<f32>,
    k_pass: Vec<f32>,
    v: Vec<f32>,
}

impl Qwen35DenseAttentionPadScratch {
    fn new() -> Self {
        Self {
            k_first: Vec::new(),
            k_second: Vec::new(),
            k_pass: Vec::new(),
            v: Vec::new(),
        }
    }

    fn fill(&mut self, source: &Qwen35DenseAttentionCache, shape: &Qwen35DenseAttentionPadShape) {
        let even_odd_len = shape.even_odd_len();
        let pass_len = shape.pass_len();
        let v_len = shape.v_len();
        if self.k_first.len() < even_odd_len {
            self.k_first.resize(even_odd_len, 0.0);
        }
        if self.k_second.len() < even_odd_len {
            self.k_second.resize(even_odd_len, 0.0);
        }
        if self.k_pass.len() < pass_len {
            self.k_pass.resize(pass_len, 0.0);
        }
        if self.v.len() < v_len {
            self.v.resize(v_len, 0.0);
        }
        self.k_first[..source.k_first.len()].copy_from_slice(&source.k_first);
        self.k_second[..source.k_second.len()].copy_from_slice(&source.k_second);
        self.k_pass[..source.k_pass.len()].copy_from_slice(&source.k_pass);
        self.v[..source.v.len()].copy_from_slice(&source.v);
    }

    fn named_blocks<'cache>(
        &'cache self,
        k_first_name: &'cache str,
        k_second_name: &'cache str,
        k_pass_name: &'cache str,
        v_name: &'cache str,
        shape: &Qwen35DenseAttentionPadShape,
    ) -> [(&'cache str, QuantizedBlock<'cache>); 4] {
        [
            (
                k_first_name,
                QuantizedBlock::Float32(&self.k_first[..shape.even_odd_len()]),
            ),
            (
                k_second_name,
                QuantizedBlock::Float32(&self.k_second[..shape.even_odd_len()]),
            ),
            (
                k_pass_name,
                QuantizedBlock::Float32(&self.k_pass[..shape.pass_len()]),
            ),
            (v_name, QuantizedBlock::Float32(&self.v[..shape.v_len()])),
        ]
    }
}

/// [`LayerCache`]'s counterpart for a [`Qwen35LayerRoots::Ssm`] layer --
/// `conv_history` is a fixed-size rolling window (the causal conv1d
/// kernel's own left context, `Qwen35SsmShape::conv_rows` rows of
/// `Qwen35SsmShape::qkv_dim` elements each, oldest row dropped as each new
/// one is appended) rather than [`LayerCache`]'s unbounded grow-forever
/// history; `state` is the gated DeltaNet recurrent state, fully replaced
/// every step (never appended to) because the mixer already folds every
/// past position into it.
struct SsmLayerCache {
    conv_history: Vec<f32>,
    state: Vec<f32>,
}

impl SsmLayerCache {
    fn new(shape: Qwen35SsmShape) -> Self {
        Self {
            conv_history: alloc::vec![0.0f32; shape.conv_rows * shape.qkv_dim],
            state: alloc::vec![0.0f32; shape.state_len],
        }
    }

    /// `qkv_mixed_new` is this step's own `new_count`-many freshly computed
    /// `qkv_mixed` rows (`shape.qkv_dim` elements each); `state_new` is the
    /// mixer's full replacement state. Keeps only `shape.conv_rows`' worth
    /// of the most recent `qkv_mixed` rows -- older rows fall out of the
    /// causal conv1d kernel's left context and are never read again.
    fn advance(&mut self, qkv_mixed_new: &[f32], state_new: &[f32], shape: Qwen35SsmShape) {
        self.conv_history.extend_from_slice(qkv_mixed_new);
        let keep = shape.conv_rows * shape.qkv_dim;
        let drop = self.conv_history.len().saturating_sub(keep);
        self.conv_history.drain(0..drop);
        self.state.clear();
        self.state.extend_from_slice(state_new);
    }

    fn named_blocks<'cache>(
        &'cache self,
        conv_history_name: &'cache str,
        state_name: &'cache str,
    ) -> [(&'cache str, QuantizedBlock<'cache>); 2] {
        [
            (
                conv_history_name,
                QuantizedBlock::Float32(self.conv_history.as_slice()),
            ),
            (state_name, QuantizedBlock::Float32(self.state.as_slice())),
        ]
    }
}

/// [`LayerCache::new`]/[`SsmLayerCache::new`] threaded per forward-program
/// layer, matching [`LoadedModel::layer_roots`]'s own per-layer discriminant
/// -- an attention layer's cache append/readback shape genuinely differs
/// from an ssm layer's, the same reason [`Qwen35LayerRoots`] itself is an
/// enum rather than a fixed-shape tuple.
enum LayerCacheState {
    Attention(LayerCache),
    DenseAttention(Qwen35DenseAttentionCache),
    Ssm(SsmLayerCache),
}

/// This call's own [`Op::Input`] names for one layer's cache, matching
/// [`LayerCacheState`]'s discriminant one-to-one -- built once per
/// [`LoadedModel::run_decode_loop`] call (never per step) since layer kind
/// and layer index never change within a call.
enum LayerCacheNames {
    Attention {
        k_even: String,
        k_odd: String,
        v: String,
    },
    DenseAttention {
        k_first: String,
        k_second: String,
        k_pass: String,
        v: String,
    },
    Ssm {
        conv_history: String,
        state: String,
    },
}

/// Every per-call input the cached forward program needs beyond the model
/// weights and the growing key/value cache: `ids_f32`/RoPE `cos`/`sin` for
/// only the `new` positions this call introduces, at their true absolute
/// angle (`start_position`, not 0 -- a generated token's position is
/// `cached_len`, never the start of the sequence), plus the
/// reduce-broadcast `eps` vector sized to match.
struct PositionInputs {
    ids_f32: Vec<f32>,
    epsilon: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

fn build_position_inputs(
    new_ids: &[u32],
    start_position: usize,
    head_dim: u32,
    rope_freq_base: f32,
    rms_epsilon: f32,
) -> PositionInputs {
    let new_count = new_ids.len();
    let pairs = head_dim as usize / 2;
    let ids_f32: Vec<f32> = new_ids.iter().map(|&id| id as f32).collect();
    let epsilon = alloc::vec![rms_epsilon; new_count];

    let mut cos = alloc::vec![0.0f32; new_count * pairs];
    let mut sin = alloc::vec![0.0f32; new_count * pairs];
    for offset in 0..new_count {
        let position = (start_position + offset) as f32;
        for pair in 0..pairs {
            let theta = position * rope_freq_base.powf(-((2 * pair) as f32) / (head_dim as f32));
            cos[offset * pairs + pair] = theta.cos();
            sin[offset * pairs + pair] = theta.sin();
        }
    }

    PositionInputs {
        ids_f32,
        epsilon,
        cos,
        sin,
    }
}

/// The fully-supported [`ServingConfig`]: every knob [`apply_serving_config`]
/// accepts today, `F32` key/value cache storage (the only precision the
/// cached-attention reduce's shared `kv_heads` axis can cross -- see
/// `bind.rs`'s own `q8_0_quantized_key_value_cache_cannot_cross_the_weight_matmul_quantized_seam`
/// for the gap this sidesteps by construction rather than by luck).
///
/// `gpu_layers` is the caller's own backend pick threaded straight through
/// to [`ServingConfig::gpu_layers`]/[`select_backend`] -- `0` for CPU,
/// [`GPU_LAYERS_ALL`] for Metal (only valid on a `metal`-featured build,
/// `apply_serving_config`'s own gate). `math_mode` is the same kind of
/// pass-through for [`ServingConfig::math_mode`] -- present only on a
/// `metal`-featured macOS build, the same gate that field carries, since
/// [`omega::MathMode`] itself does not exist otherwise. `crate::quality`'s
/// harness is the reason this is a parameter rather than always
/// `ServingConfig::default`'s `Relaxed`: without it, `quality_report` could
/// never measure any math mode other than the compiled-in default (ROW 356).
/// Every other caller here passes [`omega::MathMode::default`] to keep its
/// own behavior exactly as it was before this parameter existed.
/// `pub(crate)` rather than private: `crate::quality`'s reference/variant
/// harness needs the identical fully-supported knob set [`Self::generate`]
/// runs, with only the backend choice left open, so it reuses this function
/// instead of hand-copying its field list.
pub(crate) fn supported_serving_config(
    gpu_layers: i32,
    #[cfg(all(feature = "metal", target_os = "macos"))] math_mode: omega::MathMode,
) -> ServingConfig<'static> {
    ServingConfig {
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        gpu_layers,
        reasoning_budget: 0,
        #[cfg(all(feature = "metal", target_os = "macos"))]
        math_mode,
        ..ServingConfig::default()
    }
}

/// ROW 356's own regression: proves [`supported_serving_config`] actually
/// carries its `math_mode` argument into [`ServingConfig::math_mode`]
/// rather than the `..ServingConfig::default()` tail silently overwriting
/// it -- the exact defect the quality harness had (`generate.rs:1397` built
/// its config with `..ServingConfig::default()` and no `math_mode` field at
/// all, so every quality measurement ran `Relaxed` no matter what
/// `PROXIMA_MATH_MODE` said). Structural, no checkpoint needed: the real
/// end-to-end check is `quality::real_openchat_file::metal_vs_cpu_reports_real_drift`,
/// `#[ignore]`d because it needs a host-local model.
#[cfg(all(test, feature = "metal", target_os = "macos"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod supported_serving_config_tests {
    use super::supported_serving_config;

    #[test]
    fn threads_the_requested_math_mode_into_the_serving_config() {
        let safe_config = supported_serving_config(0, omega::MathMode::Safe);
        let fast_config = supported_serving_config(0, omega::MathMode::Fast);

        assert_eq!(safe_config.math_mode, omega::MathMode::Safe);
        assert_eq!(fast_config.math_mode, omega::MathMode::Fast);
    }
}

/// `ServingConfig::gpu_layers` (`-ngl`, `serving.rs`) is this crate's
/// existing GPU-offload knob, so backend selection reads it rather than a
/// second mechanism -- `0` (cpu-only, [`supported_serving_config`]'s own
/// default) selects [`Engine::Cpu`]; [`GPU_LAYERS_ALL`] (`-ngl all`)
/// selects [`Engine::Gpu`]. `apply_serving_config` rejects every other
/// value before a forward ever runs, so those are the only two this match
/// needs to distinguish.
#[cfg(feature = "metal")]
fn select_backend(config: &ServingConfig) -> Engine {
    if config.gpu_layers == GPU_LAYERS_ALL {
        Engine::Gpu
    } else {
        Engine::Cpu
    }
}

// `dequantize_unsupported_metal_weights`/`resolve_packed_block` (the
// per-call "convert this packed weight back to f32 because Metal has no
// unpack kernel for it yet" step) were deleted here: `Q4_K`/`Q5_K`/`Q6_K`
// -- every packed codec this checkpoint's weights actually carry -- now
// all stay packed straight to the GPU (`omega::msl::Q5K_UNPACK_MSL` is the
// row-blocked kernel `Q5_K` was still missing), so the conversion step had
// zero remaining callers. `named_blocks` below pushes `*block` directly,
// the same value this mechanism always resolved to once its lookup found
// nothing to convert.

/// Everything a decode step needs to actually run the program that is
/// backend-specific: which [`Engine`] to run it on, and the reusable state
/// each call to [`Self::evaluate`] persists across steps. Owns the plan
/// cache directly rather than through a trait object -- [`Engine`] is
/// already a closed, non-`dyn` enum (`omega::backend`'s own doc), and this
/// struct's whole job is picking one arm of it once per
/// [`LoadedModel::generate_with_serving_config`] call. This crate never links
/// `wgpu-backend`, so `Engine::Gpu` here always resolves to the Metal driver
/// through [`omega::backend::GpuDriver::for_target`].
/// [`BackendRuntime::math_mode`]/[`BackendRuntime::numeric_policy`]/
/// [`BackendRuntime::dispatch_type`] -- the three `ServingConfig` knobs
/// [`BackendRuntime::build_placed_plan`] applies together, in this order,
/// to every freshly built [`omega::metal::Plan`]. A parameter struct
/// rather than three positional arguments: every caller already copies
/// all three off `self` in one statement before the plan-cache build
/// closure, so one reference at the call site says what was already true
/// by convention, and keeps `build_placed_plan` under clippy's
/// `too_many_arguments` threshold without an `#[allow]`.
#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
#[derive(Debug, Clone, Copy, PartialEq)]
struct PlanNumerics {
    math_mode: omega::metal::MathMode,
    numeric_policy: proxima_tensor::NumericPolicy,
    dispatch_type: omega::metal::DispatchType,
}

#[cfg(feature = "metal")]
pub(crate) struct BackendRuntime {
    engine: Engine,
    /// Keyed by `(new_count, kv_bound_extent)` -- the two symbols
    /// `mistral_cached_forward_program`'s cached-attention read extent
    /// resolves against (`Extent::Symbolic(1) == kv_bound_extent`). A
    /// [`Plan`] bakes concrete shapes from those symbols
    /// (`omega::backend::plan_named`'s own doc), and RAW `cached_len` grows
    /// by `new_count` every decode step, so a plan keyed on it directly was
    /// never valid for the next step -- the exact defect ROW 392 measured
    /// (one miss and one fresh `Plan`, with its own device output buffers,
    /// per token). `kv_bound_extent` is `cached_len` rounded up to
    /// `ServingConfig::kv_bucket_tokens` (`generate::kv_extent`'s own doc),
    /// which repeats for every step inside one bucket -- the fused
    /// `BoundOpKind::CachedAttention` op's own runtime bound
    /// (`proxima_tensor::bind::cached_attention_candidates`'s own doc) is
    /// what makes a `Plan` built for that rounded shape numerically correct
    /// for every real `cached_len` the bucket covers, so ordinary autoregressive
    /// decode now hits this cache `bucket_tokens - 1` times out of every
    /// `bucket_tokens` steps instead of never.
    ///
    /// [`Self::resolve_cached_plan`] clears this on every miss instead of
    /// accumulating entries: measured on a real decode before bucketing
    /// landed (`plan_cache_len` / `plan_misses` in `token_breakdown_metal`)
    /// this map grew 1:1 with the step index and `plan_hits` never left 0,
    /// so every step but the first was retaining a `Plan` that could never
    /// be looked up again for the rest of the call -- a Rust-heap leak
    /// (`phys_footprint_bytes` climbed while `omega::metal::current_allocated_size()`
    /// stayed flat over the same steps, proving the growth was not
    /// GPU-side). Clearing on miss keeps exactly the one entry worth
    /// keeping: the bucket a caller is currently inside.
    plans: alloc::collections::BTreeMap<(usize, usize), Plan>,
    /// `ServingConfig::math_mode`, read once at construction and narrowed
    /// into every freshly-built [`Plan`] below (`set_math_mode`'s own call
    /// sites) -- a plan-cache hit reuses a `Plan` already carrying it, same
    /// as `resident_names`/`mark_resident` above.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    math_mode: omega::metal::MathMode,
    /// `ServingConfig::numeric_policy`, read once at construction and
    /// passed INTO `plan_named`/`plan_named_placed` at build time (never a
    /// post-hoc setter -- `omega::metal::Plan::numeric_policy`'s own doc:
    /// the policy is fixed for a plan's whole life). Ungated, unlike
    /// `math_mode`/`dispatch_type` above: `ServingConfig::numeric_policy`
    /// is present unconditionally (its own doc), and `omega::backend::
    /// plan_named`'s signature now takes it on every engine arm, not only
    /// the Metal one.
    numeric_policy: proxima_tensor::NumericPolicy,
    /// `ServingConfig::dispatch_type`, read once at construction and applied
    /// to every freshly-built [`Plan`] below (`set_dispatch_type`'s own call
    /// sites) -- same pattern as `math_mode` immediately above.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    dispatch_type: omega::metal::DispatchType,
    /// [`Self::evaluate_with_placements`]'s own plan cache -- same
    /// `(new_count, merged_len)` keying as `plans` above, but holding
    /// `omega::metal::Plan` directly rather than the backend-polymorphic
    /// `omega::backend::Plan` enum, since [`PlacedBuffer`] placement has no
    /// arm for CPU/wgpu and only ever runs against the Metal backend. A
    /// second map rather than a second variant on `plans`'s own `Plan`
    /// enum because the two plan types come from executing two entirely
    /// different programs (two-range vs. single-range) against the same
    /// `(new_count, merged_len)` shape space -- sharing one map would let a
    /// single-range plan satisfy a two-range lookup by coincidence of key.
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    placed_plans: alloc::collections::BTreeMap<(usize, usize), omega::metal::Plan>,
    pub(crate) plan_hits: usize,
    pub(crate) plan_misses: usize,
    /// `ServingConfig::exact_activations`, read once at construction --
    /// `Self::evaluate`'s `Engine::Cpu` arm plans through
    /// `omega::backend::plan_named_exact` instead of `plan_named` when
    /// this is `true`, so a cross-backend quality harness's CPU reference
    /// carries the same zero activation-quantization error Metal's own
    /// kernels do (see `ServingConfig::exact_activations`'s own doc). No
    /// effect on `Engine::Gpu`: `plan_named_exact` is a no-op identity on
    /// that arm.
    exact_activations: bool,
}

#[cfg(feature = "metal")]
impl BackendRuntime {
    pub(crate) fn new(config: &ServingConfig) -> Self {
        Self {
            engine: select_backend(config),
            plans: alloc::collections::BTreeMap::new(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            math_mode: config.math_mode,
            numeric_policy: config.numeric_policy,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            dispatch_type: config.dispatch_type,
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            placed_plans: alloc::collections::BTreeMap::new(),
            plan_hits: 0,
            plan_misses: 0,
            exact_activations: config.exact_activations,
        }
    }

    /// Whether this call's [`ServingConfig`] selected the Gpu engine --
    /// [`Self::engine`] is private (this struct's whole job is hiding which
    /// arm was picked), so [`Self::run_decode_loop`]'s own choice of the
    /// placed-KV decode path against [`LoadedModel::single_range`] needs
    /// this accessor rather than reading the field directly.
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    pub(crate) fn is_metal(&self) -> bool {
        matches!(self.engine, Engine::Gpu)
    }

    /// `resident_names` -- the caller's own model-weight names, fixed for
    /// the whole [`LoadedModel::generate_with_serving_config`] call -- is
    /// handed to [`mark_resident`] exactly once per distinct [`Plan`] (right
    /// after it is built, never on a cache hit, since a hit reuses the SAME
    /// `Plan` object that was already marked). See `omega::metal::Plan::mark_resident`'s
    /// own doc for why this needs a name set the tensor program itself
    /// cannot derive.
    fn evaluate(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
    ) -> Result<Evaluated, InteropError> {
        let shape = (symbols[0] as usize, symbols[1] as usize);
        let exact_activations = self.exact_activations;
        let plan = Self::resolve_cached_plan(
            &mut self.plans,
            &mut self.plan_hits,
            &mut self.plan_misses,
            shape,
            || {
                let mut plan = if exact_activations {
                    plan_named_exact(
                        self.engine,
                        None,
                        program,
                        symbols,
                        named,
                        outputs,
                        self.numeric_policy,
                    )?
                } else {
                    plan_named(
                        self.engine,
                        None,
                        program,
                        symbols,
                        named,
                        outputs,
                        self.numeric_policy,
                    )?
                };
                mark_resident(&mut plan, resident_names);
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    set_math_mode(&mut plan, self.math_mode)?;
                    set_dispatch_type(&mut plan, self.dispatch_type);
                }
                Ok(plan)
            },
        )?;
        Ok(execute_plan_named(plan, named)?)
    }

    /// [`Self::evaluate`]'s placed-KV counterpart: same `(new_count,
    /// merged_len)` plan-cache bookkeeping (`plan_hits`/`plan_misses` stay
    /// meaningful across both paths -- a caller reading them after the loop
    /// cannot tell which one ran), but plans and executes directly against
    /// `omega::metal` (`plan_named_placed`/[`execute_plan_named_with_placements`])
    /// rather than through `omega::backend`'s polymorphic entry point, and
    /// routes `input_placements`/`output_placements` into the execute call
    /// -- the whole reason this method exists next to [`Self::evaluate`]
    /// rather than adding a placement parameter there, since every other
    /// backend arm has no such parameter to accept. Clears `placed_plans`
    /// on every miss, same as `plans`/[`Self::resolve_cached_plan`] -- the
    /// identical superseded-`Plan` leak that clearing fixed there applies
    /// here unchanged: ordinary decode's `merged_len` strictly increases,
    /// so a miss means the previous entry can never be looked up again.
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    #[allow(clippy::too_many_arguments)]
    fn evaluate_with_placements(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
        input_placements: &[(NodeId, &PlacedBuffer, usize)],
        output_placements: &[(NodeId, &PlacedBuffer, usize)],
    ) -> Result<Evaluated, InteropError> {
        let shape = (symbols[0] as usize, symbols[1] as usize);
        let numerics = PlanNumerics {
            math_mode: self.math_mode,
            numeric_policy: self.numeric_policy,
            dispatch_type: self.dispatch_type,
        };
        let plan = Self::resolve_cached_plan(
            &mut self.placed_plans,
            &mut self.plan_hits,
            &mut self.plan_misses,
            shape,
            || Self::build_placed_plan(program, symbols, named, outputs, resident_names, &numerics),
        )?;
        Ok(execute_plan_named_with_placements(
            plan,
            named,
            input_placements,
            output_placements,
        )?)
    }

    /// Every [`Self::placed_plans`] build closure's shared body -- the class
    /// fix for the defect ROW 329's slice found: `evaluate_op_timed_with_placements`
    /// and `evaluate_dispatch_timed_with_placements` used to build their own
    /// `plan_named_placed` + `mark_resident` inline, never calling
    /// `set_math_mode`/`set_dispatch_type`, so a shape first resolved through
    /// either diagnostic path entered the cache carrying the DEFAULT math
    /// mode / dispatch type, and a later hit from [`Self::evaluate_with_placements`]
    /// (the production path) silently served that wrong mode. Folding all
    /// three closures through this one function makes every placed-plan
    /// build path correct by construction -- there is no longer a second
    /// closure body that can forget the call.
    ///
    /// Takes `numerics` as one reference rather than `math_mode`/
    /// `numeric_policy`/`dispatch_type` as three positional copies --
    /// [`PlanNumerics`] groups exactly the three [`ServingConfig`] knobs
    /// every caller below already reads and threads together, so the
    /// signature says that instead of leaving it to be true by convention
    /// (and drops the argument count back under clippy's threshold without
    /// an `#[allow]`).
    ///
    /// `numeric_policy` is now passed INTO `plan_named_placed` at
    /// construction, never set afterward (`omega::metal::Plan::
    /// numeric_policy`'s own doc: the policy is fixed for a plan's whole
    /// life) -- `set_math_mode` runs after construction only to NARROW the
    /// compiled `MathMode` within that already-bound policy, and is now
    /// fallible for exactly that reason.
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    fn build_placed_plan(
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
        numerics: &PlanNumerics,
    ) -> Result<omega::metal::Plan, InteropError> {
        let mut plan =
            plan_named_placed(program, symbols, named, outputs, numerics.numeric_policy)?;
        plan.mark_resident(resident_names);
        plan.set_math_mode(numerics.math_mode)?;
        plan.set_dispatch_type(numerics.dispatch_type);
        Ok(plan)
    }

    /// [`Self::evaluate_with_placements`]'s diagnostic counterpart, same
    /// relationship [`Self::evaluate_op_timed`] already has to
    /// [`Self::evaluate`]: identical plan-cache lookup against
    /// [`Self::placed_plans`], but
    /// [`omega::metal::execute_plan_named_with_placements_op_timed`] commits
    /// and waits on ITS OWN command buffer per op instead of the whole
    /// program's one, so [`OpGpuTiming`] comes back for the default decode
    /// path's placed-KV shape the same way [`Self::evaluate_op_timed`]
    /// already does for the two-range path. Reachable only behind the
    /// `instrument` feature and only from `run_decode_loop_placed_kv`'s own
    /// `PROXIMA_METAL_OP_PROFILE_STEP` branch.
    #[cfg(all(
        feature = "metal-output-placement",
        feature = "instrument",
        target_os = "macos"
    ))]
    #[allow(clippy::too_many_arguments)]
    fn evaluate_op_timed_with_placements(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
        input_placements: &[(NodeId, &PlacedBuffer, usize)],
        output_placements: &[(NodeId, &PlacedBuffer, usize)],
    ) -> Result<(Evaluated, Vec<OpGpuTiming>), InteropError> {
        let shape = (symbols[0] as usize, symbols[1] as usize);
        let numerics = PlanNumerics {
            math_mode: self.math_mode,
            numeric_policy: self.numeric_policy,
            dispatch_type: self.dispatch_type,
        };
        let plan = Self::resolve_cached_plan(
            &mut self.placed_plans,
            &mut self.plan_hits,
            &mut self.plan_misses,
            shape,
            || Self::build_placed_plan(program, symbols, named, outputs, resident_names, &numerics),
        )?;
        Ok(execute_plan_named_with_placements_op_timed(
            plan,
            named,
            input_placements,
            output_placements,
        )?)
    }

    /// [`Self::evaluate_with_placements`]'s per-dispatch GPU-timestamp
    /// counterpart -- same plan-cache lookup, but
    /// [`omega::execute_plan_named_with_placements_dispatch_timed`] submits
    /// the SAME single command buffer the production path does and reads
    /// `MTLCounterSampleBuffer` timestamps bracketing every dispatch
    /// instead of one command buffer per op
    /// ([`Self::evaluate_op_timed_with_placements`]'s own shape, which ROW
    /// 298's own finding says does not reproduce the batched buffer's
    /// cost). Reachable only behind `instrument` and only from
    /// `run_decode_loop_placed_kv`'s own `PROXIMA_METAL_DISPATCH_PROFILE_STEP`
    /// branch. That same call site also reads `PROXIMA_METAL_ENCODER_SPLIT_AT`
    /// (ROW 329, same one-env-var-per-diagnostic convention as
    /// `PROXIMA_DUPLICATE_HEAD`/`PROXIMA_METAL_OP_PROFILE_STEP` above) and
    /// applies it to the resolved plan via
    /// [`omega::metal::Plan::set_encoder_split_at`] AFTER the cache lookup
    /// below, never inside the build closure: measured directly (a
    /// `debug!` trace that showed `plan_encoder_split_at=None` reaching
    /// [`omega::metal::execute_plan_with_placements_dispatch_timed`] despite
    /// this call setting it), `ServingConfig::kv_bucket_tokens` rounds
    /// several consecutive steps' KV extents onto the SAME cache key, so
    /// the plan this call's own step reuses is often one
    /// [`Self::evaluate_with_placements`]'s closure already inserted --
    /// build-time-only would silently no-op on that hit.
    #[cfg(all(
        feature = "metal-output-placement",
        feature = "instrument",
        target_os = "macos"
    ))]
    #[allow(clippy::too_many_arguments)]
    fn evaluate_dispatch_timed_with_placements(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
        input_placements: &[(NodeId, &PlacedBuffer, usize)],
        output_placements: &[(NodeId, &PlacedBuffer, usize)],
    ) -> Result<omega::metal::DispatchTimedOutcome, InteropError> {
        let shape = (symbols[0] as usize, symbols[1] as usize);
        let numerics = PlanNumerics {
            math_mode: self.math_mode,
            numeric_policy: self.numeric_policy,
            dispatch_type: self.dispatch_type,
        };
        let plan = Self::resolve_cached_plan(
            &mut self.placed_plans,
            &mut self.plan_hits,
            &mut self.plan_misses,
            shape,
            || Self::build_placed_plan(program, symbols, named, outputs, resident_names, &numerics),
        )?;
        // Applied AFTER the cache lookup, not inside the build closure above:
        // this shape's `Plan` is just as likely to have been inserted by
        // `Self::evaluate_with_placements`'s own closure (a HIT here on a
        // step where `PROXIMA_METAL_DISPATCH_PROFILE_STEP` was unset, since
        // `ServingConfig::kv_bucket_tokens` rounds several consecutive
        // steps' KV extents to the SAME cache key) as by this function's
        // own closure -- a build-time-only `set_encoder_split_at` call
        // would silently no-op on that hit path. Cheap and always correct
        // to set unconditionally: unlike `set_math_mode`, this field never
        // invalidates `resolved_steps` (this same function's own doc).
        plan.set_encoder_split_at(encoder_split_at_from_env());
        Ok(execute_plan_named_with_placements_dispatch_timed(
            plan,
            named,
            input_placements,
            output_placements,
        )?)
    }

    /// [`Self::evaluate`]/[`Self::evaluate_op_timed`]/
    /// [`Self::evaluate_with_placements`]/
    /// [`Self::evaluate_op_timed_with_placements`]'s shared cache-lookup
    /// step, split out so the eviction policy lives in exactly one place and
    /// so every caller gets back the `Plan` it just resolved rather than a
    /// shape it must look up again -- the second lookup was the only reason
    /// `InteropError::PlanCacheEntryVanished` (now deleted) existed, for a
    /// state (a key missing immediately after this function inserted it)
    /// that cannot occur: [`alloc::collections::btree_map::Entry`] proves it
    /// at the type level instead.
    ///
    /// Generic over the cached `Plan` type because [`Self::plans`] and
    /// [`Self::placed_plans`] key different `Plan` types under the same
    /// `(usize, usize)` shape but share this exact hit/miss/evict policy.
    ///
    /// This struct's own [`Self::plans`] field comment already proved
    /// ordinary autoregressive decode's `cached_len` strictly increases, so
    /// a `Plan` keyed on it is NEVER looked up again once superseded --
    /// measured directly: `plan_cache_len`/`plan_misses` both grew 1:1 with
    /// the step index (`plan_hits` stayed 0) across a real decode, while
    /// `phys_footprint_bytes` climbed and `omega::metal::current_allocated_size()`
    /// stayed flat over the same steps -- the growth is a Rust-heap leak of
    /// superseded `Plan`s, not a Metal-driver allocation. Clearing the map on
    /// every miss keeps the one entry the doc's own rationale says is worth
    /// keeping (an immediate same-shape replay lands as a hit BEFORE the
    /// next miss would evict it) while making superseded entries collectible
    /// instead of retained for the rest of the call.
    fn resolve_cached_plan<'cache, PlanType>(
        cache: &'cache mut alloc::collections::BTreeMap<(usize, usize), PlanType>,
        plan_hits: &mut usize,
        plan_misses: &mut usize,
        shape: (usize, usize),
        build: impl FnOnce() -> Result<PlanType, InteropError>,
    ) -> Result<&'cache mut PlanType, InteropError> {
        use alloc::collections::btree_map::Entry;

        if !cache.contains_key(&shape) {
            cache.clear();
        }
        match cache.entry(shape) {
            Entry::Occupied(entry) => {
                *plan_hits += 1;
                Ok(entry.into_mut())
            }
            Entry::Vacant(entry) => {
                *plan_misses += 1;
                let plan = build()?;
                Ok(entry.insert(plan))
            }
        }
    }

    /// Live entry count in [`Self::plans`] -- the direct witness that
    /// [`Self::resolve_cached_plan`]'s clear-on-miss policy keeps this bounded at 1
    /// through ordinary autoregressive decode's strictly increasing
    /// `cached_len`, rather than growing 1:1 with the step index as it did
    /// before that policy landed. See `token_breakdown_metal`'s
    /// `plan_cache_len` field.
    #[cfg(feature = "instrument")]
    pub(crate) fn plans_len(&self) -> usize {
        self.plans.len()
    }

    /// [`Self::plans_len`]'s placed-KV counterpart -- [`Self::plans`] and
    /// [`Self::placed_plans`] are two SEPARATE maps
    /// ([`Self::evaluate_with_placements`] never touches [`Self::plans`] at
    /// all), so a caller on [`LoadedModel::run_decode_loop_placed_kv`]'s
    /// own arm reading [`Self::plans_len`] was always reading a map that
    /// stayed empty for the whole call, regardless of how many placed
    /// plans were actually cached.
    #[cfg(all(
        feature = "instrument",
        feature = "metal-output-placement",
        target_os = "macos"
    ))]
    pub(crate) fn placed_plans_len(&self) -> usize {
        self.placed_plans.len()
    }

    /// Diagnostic counterpart of [`Self::evaluate`]: same plan-cache lookup,
    /// but the Metal driver commits and waits on ONE command buffer PER
    /// `BoundOp` instead of once for the whole program, so each op's own
    /// GPU-only execution time comes back alongside the result -- see
    /// `omega::metal::execute_plan_op_timed`'s own doc for the cost this
    /// pays and why it must never replace [`Self::evaluate`] on the serving
    /// loop. Reachable only behind the `instrument` feature and only from
    /// this crate's own diagnostic call sites (`run_decode_loop`'s
    /// `PROXIMA_METAL_OP_PROFILE_STEP` branch).
    #[cfg(all(feature = "instrument", target_os = "macos"))]
    fn evaluate_op_timed(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
    ) -> Result<(Evaluated, Vec<OpGpuTiming>), InteropError> {
        let shape = (symbols[0] as usize, symbols[1] as usize);
        let plan = Self::resolve_cached_plan(
            &mut self.plans,
            &mut self.plan_hits,
            &mut self.plan_misses,
            shape,
            || {
                let mut plan = plan_named(
                    self.engine,
                    None,
                    program,
                    symbols,
                    named,
                    outputs,
                    self.numeric_policy,
                )?;
                mark_resident(&mut plan, resident_names);
                set_math_mode(&mut plan, self.math_mode)?;
                set_dispatch_type(&mut plan, self.dispatch_type);
                Ok(plan)
            },
        )?;
        Ok(execute_plan_named_metal_op_timed(plan, named)?)
    }
}

/// The CPU-direct runtime a build without the `metal` feature keeps --
/// `omega` is not even a dependency in that build (`Cargo.toml`'s `metal`
/// feature is the only thing that turns `dep:omega` on), so this calls
/// [`evaluate_quantized_named_with_scratch`] exactly as
/// [`LoadedModel::generate`] always has. `free_buffers`/`validated_weight_nodes`
/// are the same scratch this loop's local variables used to own directly --
/// moved onto this struct so [`LoadedModel::generate_with_serving_config`]'s
/// loop body reads identically whether or not `metal` is compiled in.
#[cfg(not(feature = "metal"))]
pub(crate) struct BackendRuntime {
    free_buffers: Vec<Vec<f32>>,
    validated_weight_nodes: Option<BTreeSet<NodeId>>,
    /// `ServingConfig::exact_activations`, read once at construction --
    /// see that field's own doc.
    exact_activations: bool,
}

#[cfg(not(feature = "metal"))]
impl BackendRuntime {
    pub(crate) fn new(_config: &ServingConfig) -> Self {
        Self {
            free_buffers: Vec::new(),
            validated_weight_nodes: None,
            exact_activations: _config.exact_activations,
        }
    }

    /// `resident_names` is unused on this backend: the CPU evaluator has no
    /// device buffer to cache, so there is nothing to mark resident. Carried
    /// anyway so both `BackendRuntime::evaluate` impls share one signature
    /// and the decode loop's call site never needs a `cfg` of its own.
    fn evaluate(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        _resident_names: &BTreeSet<&str>,
    ) -> Result<Evaluated, InteropError> {
        if self.exact_activations {
            return Ok(evaluate_quantized_named_exact_with_scratch(
                program,
                symbols,
                named,
                outputs,
                &mut self.free_buffers,
                &mut self.validated_weight_nodes,
            )?);
        }
        Ok(evaluate_quantized_named_with_scratch(
            program,
            symbols,
            named,
            outputs,
            &mut self.free_buffers,
            &mut self.validated_weight_nodes,
        )?)
    }
}

impl<'file> Pipe for LoadedModel<'file> {
    type In = (String, usize);
    type Out = (Vec<u32>, String, bool);
    type Err = InteropError;

    fn call(
        &self,
        input: (String, usize),
    ) -> impl Future<Output = Result<(Vec<u32>, String, bool), InteropError>> {
        async move {
            let (prompt, max_tokens) = input;
            self.generate(&prompt, max_tokens)
        }
    }
}

/// One decode step surfaced to a caller AS it happens, instead of only
/// after [`LoadedModel::generate_streaming`] returns -- the payload
/// [`decode_until_stop_or_budget`] hands to its `on_token` callback every
/// step, teaching a caller (a CLI's "loading / thinking / answering"
/// indicator) exactly what that loop already knows at that point and
/// nothing it has to re-derive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenEvent<'piece> {
    pub token_id: u32,
    /// This token's own decoded text, continuing on from whatever
    /// [`decode_until_stop_or_budget`] has already handed back for earlier
    /// tokens -- concatenating every `text_piece` across a whole decode
    /// reproduces [`proxima_tokenizer::decode`]'s own output on the same
    /// ids exactly (`decode_streamed_piece`'s own doc: incomplete
    /// multi-byte tails carry forward instead of resolving to U+FFFD mid
    /// stream).
    pub text_piece: &'piece str,
    pub phase: Phase,
    /// `0`-indexed decode step this event belongs to.
    pub step: usize,
    /// Milliseconds since this call's decode loop started (`step` `0`'s own
    /// first [`std::time::Instant::now`] reading), not this step's own
    /// duration -- a caller computes both a running tok/s and a single
    /// step's latency from two consecutive events' `elapsed_ms` without
    /// this loop tracking either itself. Plain [`std::time::Instant`]
    /// rather than `proxima_tensor::instrument`'s tick counters: those only
    /// exist behind this crate's diagnostic-only `instrument` feature
    /// (`proxima-tensor/src/lib.rs`'s own `#[cfg(feature = "instrument")]`
    /// on that module), and a live "what's running" indicator must work on
    /// every build that reaches [`LoadedModel::generate_streaming`] at all,
    /// not only one compiled for op-level profiling.
    pub elapsed_ms: u64,
}

/// [`TokenEvent::phase`]: which part of the decode loop produced this
/// event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Emitted exactly once, at decode step `0` -- the one step whose
    /// forward pass evaluates the whole encoded prompt rather than a single
    /// new token (`LoadedModel::run_decode_loop_observed`'s own doc on
    /// `new_positions == prompt_length` on the first step).
    Prefill {
        /// The encoded prompt's own token count (`ids.len()` before
        /// decoding starts), so a caller can show "127 prompt tokens"
        /// without re-encoding the prompt itself.
        prompt_tokens: usize,
    },
    /// Every decode step, prefill included -- carries the token that step
    /// produced.
    Token,
}

/// A caller's per-token decision, read back by [`decode_until_stop_or_budget`]
/// after every [`TokenEvent`]. `Stop` is the same early exit this loop
/// already gives the model's own end-of-sequence id, just requested by the
/// caller instead -- a chat template's own `<|im_end|>`, or a think-block
/// boundary a caller wants to cut before ever reaching `max_tokens`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Control {
    Continue,
    Stop,
}

/// One [`TokenEvent::text_piece`] worth of text for `token_id`, carrying
/// forward any UTF-8 tail [`proxima_tokenizer::bpe::decode_ids`] left
/// incomplete in `pending` from a previous call -- byte-level BPE has no
/// obligation to keep a multibyte character inside one token
/// ([`proxima_tokenizer::pipe::decode`]'s own doc), so a caller watching
/// tokens arrive one at a time needs the same "flag, don't drop" contract
/// that one-shot decode gives the whole sequence, just deferred: an
/// incomplete tail waits here for the token that completes it instead of
/// resolving to U+FFFD before decoding is known to be finished.
///
/// Never returns [`proxima_tokenizer::TokenizerError::InvalidUtf8`] over a
/// pending tail that turns out unresolvable -- a decode step producing a
/// token id that does not validly continue an earlier incomplete lead byte
/// is a normal outcome of autoregressive sampling, not corruption
/// (guiding-principles principle 15: an error type is not the "correct
/// treatment" for a case the caller cannot act on). Draining is
/// [`proxima_tokenizer::drain_lossy_utf8`] itself -- the same routine
/// [`proxima_tokenizer::pipe::decode`] uses for the one-shot whole-sequence
/// case -- so a genuinely invalid run resolves to one U+FFFD and draining
/// resumes on whatever bytes remain after it here exactly as it does there;
/// the only difference is this call site leaves an incomplete trailing
/// sequence in `pending` for a future token to complete, instead of
/// flushing it immediately.
fn decode_streamed_piece(
    vocab: &Vocab,
    token_id: u32,
    pending: &mut Vec<u8>,
) -> Result<String, InteropError> {
    let bytes = proxima_tokenizer::bpe::decode_ids(&[token_id], vocab)?;
    pending.extend_from_slice(&bytes);
    let mut piece = String::new();
    proxima_tokenizer::drain_lossy_utf8(pending, &mut piece);
    Ok(if vocab.is_unigram() {
        proxima_tokenizer::unigram::replace_space_markers(&piece)
    } else {
        piece
    })
}

/// The decode loop's termination policy, isolated from the forward pass
/// that produces each token: pulls up to `max_tokens` ids out of
/// `produce_next_token` (one call per step, `0`-indexed), appending each
/// to the result unless it is `vocab`'s end-of-sequence id, in which case
/// decoding stops immediately without appending that id. Returns the
/// accumulated ids plus whether the stop was the model's own signal
/// (`true`) rather than the budget running out (`false`) -- a caller
/// [`Control::Stop`] collapses into the same `false` as budget exhaustion,
/// since neither is the model's own eos.
///
/// Every step, after producing that step's token, calls `on_token` once (an
/// extra [`Phase::Prefill`] call at step `0`, ahead of that step's own
/// [`Phase::Token`] call) -- [`LoadedModel::generate_with_serving_config`]'s
/// own `&mut |_| Control::Continue` never observes a difference from this
/// function's pre-streaming behavior; [`LoadedModel::generate_streaming`]
/// is the same loop with a real callback.
///
/// Factored out so this policy -- the exact defect this module's
/// [`LoadedModel::generate`] fixed (a loop with no termination condition
/// besides the budget) -- is provable against a scripted token source,
/// without paying for a real forward pass per test.
fn decode_until_stop_or_budget(
    vocab: &Vocab,
    max_tokens: usize,
    prompt_token_count: usize,
    mut produce_next_token: impl FnMut(usize) -> Result<u32, InteropError>,
    on_token: &mut dyn FnMut(TokenEvent<'_>) -> Control,
) -> Result<(Vec<u32>, bool), InteropError> {
    let mut generated_ids = Vec::with_capacity(max_tokens);
    let mut stopped_by_eos = false;
    let mut pending_bytes: Vec<u8> = Vec::new();
    let mut unigram_leading_space_trimmed = false;
    let loop_started = std::time::Instant::now();
    for step in 0..max_tokens {
        let token_id = produce_next_token(step)?;
        let elapsed_ms = u64::try_from(loop_started.elapsed().as_millis()).unwrap_or(u64::MAX);
        let is_eos = vocab.eos_token_id() == Some(token_id);
        let mut text_piece = if is_eos {
            String::new()
        } else {
            decode_streamed_piece(vocab, token_id, &mut pending_bytes)?
        };
        // Mirrors `proxima_tokenizer::pipe::decode`'s own one-time leading-
        // space trim (SentencePiece's `escape` always prepends one), applied
        // to the FIRST non-empty piece this whole call ever emits rather
        // than every piece -- a later piece starting with the space marker
        // is a real inter-word space, not that artifact.
        if !unigram_leading_space_trimmed && vocab.is_unigram() && !text_piece.is_empty() {
            unigram_leading_space_trimmed = true;
            if text_piece.starts_with(' ') {
                text_piece.remove(0);
            }
        }
        if step == 0 {
            let control = on_token(TokenEvent {
                token_id,
                text_piece: &text_piece,
                phase: Phase::Prefill {
                    prompt_tokens: prompt_token_count,
                },
                step,
                elapsed_ms,
            });
            if control == Control::Stop {
                break;
            }
        }
        if is_eos {
            stopped_by_eos = true;
            break;
        }
        generated_ids.push(token_id);
        let control = on_token(TokenEvent {
            token_id,
            text_piece: &text_piece,
            phase: Phase::Token,
            step,
            elapsed_ms,
        });
        if control == Control::Stop {
            break;
        }
    }
    Ok((generated_ids, stopped_by_eos))
}

/// [`Vocab::add_bos_token`]'s own fallback when the checkpoint's metadata
/// carries no `tokenizer.ggml.add_bos_token` opinion at all: default to
/// requesting BOS only when the vocab actually HAS a
/// [`Vocab::bos_token_id`] to add. Every dense checkpoint this crate has
/// bound so far declares one (openchat-3.5, SmolLM2), so this reproduces
/// this crate's pre-existing unconditional `true` default for them
/// byte-for-byte; the real Qwen3.5 checkpoint declares neither the policy
/// key nor a `tokenizer.ggml.bos_token_id` key at all (confirmed via
/// `strings` on the real file -- Qwen's own tokenizer has no BOS token,
/// chat turns open on `<|im_start|>` instead), so defaulting to `true`
/// there would ask [`proxima_tokenizer::encode_with_bos_eos`] to prepend an
/// id that does not exist, surfacing
/// [`proxima_tokenizer::TokenizerError::MissingMetadataKey`] on every
/// prompt rather than the tokenizer's own real, silent policy.
fn wants_bos(vocab: &Vocab) -> bool {
    vocab
        .add_bos_token()
        .unwrap_or_else(|| vocab.bos_token_id().is_some())
}

impl<'file> LoadedModel<'file> {
    /// [`Self::generate_with_serving_config`] against
    /// [`supported_serving_config`] -- the reachable path every existing
    /// caller and test uses, unchanged: `gpu_layers: 0` always selects the
    /// CPU backend, so this runs exactly the forward it always has, on
    /// CPU, regardless of whether this build was compiled with the
    /// `metal` feature.
    fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<(Vec<u32>, String, bool), InteropError> {
        self.generate_with_serving_config(
            prompt,
            max_tokens,
            supported_serving_config(0, #[cfg(all(feature = "metal", target_os = "macos"))] omega::MathMode::default()),
        )
    }

    /// This checkpoint's own static weight names -- `mark_resident`'s own
    /// caller-supplied classification (`BackendRuntime::evaluate`'s doc),
    /// bound once at [`Self::load`] time and never mutated again. Shared by
    /// every decode-loop variant that calls `mark_resident` and by `Drop`,
    /// which hands this SAME name set to
    /// `omega::backend::release_resident_names` so a dropped model evicts
    /// exactly the device buffers it caused and nothing another model's
    /// own names might collide with.
    fn resident_names(&self) -> BTreeSet<&str> {
        self.weights
            .owned
            .iter()
            .map(|(name, _)| name.as_str())
            .chain(self.weights.packed.iter().map(|(name, _)| name.as_str()))
            .chain(
                self.weights
                    .packed_owned
                    .iter()
                    .map(|(name, _, _)| name.as_str()),
            )
            .collect()
    }

    /// The greedy decode loop itself: `max_tokens` steps, each one call
    /// into `BackendRuntime::evaluate` against `new_positions == 1` after
    /// the first step (`new_positions == prompt_length` on the first),
    /// growing `LayerCache` by one call's worth of positions every step
    /// instead of re-running the whole sequence from scratch -- stopping
    /// early the moment the model emits its own end-of-sequence id (see
    /// this module's doc for what that id is on the real checkpoint),
    /// never running past `max_tokens` regardless.
    ///
    /// `serving_config` is a caller-supplied override of
    /// `supported_serving_config`'s default -- the same [`ServingConfig`]
    /// [`apply_serving_config`] already gates, never a second selection
    /// mechanism. Setting `gpu_layers` to `GPU_LAYERS_ALL` (`-ngl all`) on
    /// a build compiled with this crate's `metal` feature runs this same
    /// loop against the Metal backend instead of the CPU one; every other
    /// field must already satisfy [`apply_serving_config`]'s gate the same
    /// way `supported_serving_config`'s does.
    pub fn generate_with_serving_config(
        &self,
        prompt: &str,
        max_tokens: usize,
        serving_config: ServingConfig,
    ) -> Result<(Vec<u32>, String, bool), InteropError> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let serving_config = {
            let mut serving_config = serving_config;
            self.apply_memory_fit_gate(&mut serving_config)?;
            serving_config
        };
        let mut runtime = BackendRuntime::new(&serving_config);
        self.run_decode_loop(prompt, max_tokens, &serving_config, &mut runtime)
    }

    /// The first auto-tune step (`crate::memory_fit`'s own module doc):
    /// derives this checkpoint's device-memory budget from its own shape at
    /// `serving_config.context_length`, probes the host's own device facts
    /// ([`omega::metal::system_memory_facts`]), and either leaves
    /// `serving_config.context_length` unchanged, reduces it to the
    /// largest value that fits (emitting a `context_length_reduced` warn
    /// event under `feature = "instrument"` -- callers that need the
    /// reduced value read it back off `serving_config` after this call
    /// returns, since it takes `&mut`), or refuses with
    /// [`InteropError::MemoryBudgetExceeded`] -- always
    /// before [`Self::generate_with_serving_config`]'s own next line
    /// ([`BackendRuntime::new`]) asks a device for a single buffer.
    ///
    /// A no-op when `serving_config.gpu_memory_fit` is `false` (the
    /// caller's explicit override, matching every other opt-out knob
    /// [`ServingConfig`]'s own doc already has) or when this host has no
    /// Metal device at all ([`omega::metal::system_memory_facts`] returning
    /// `Err`) -- a probe failure means this method has nothing to gate
    /// against, not that the load itself is unsafe, so it fails OPEN
    /// (proceeds unchanged) rather than refusing a load this crate cannot
    /// actually evaluate.
    ///
    /// # Errors
    ///
    /// [`InteropError::MemoryBudgetExceeded`] when even a context length of
    /// `1` cannot fit this checkpoint's own weights plus the fixed arena
    /// allowance inside the host's own reported limit.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn apply_memory_fit_gate(&self, serving_config: &mut ServingConfig) -> Result<(), InteropError> {
        if !serving_config.gpu_memory_fit {
            return Ok(());
        }
        let Ok(facts) = omega::metal::system_memory_facts() else {
            return Ok(());
        };
        let limit = crate::memory_fit::HostMemoryLimit {
            limit_bytes: facts
                .recommended_max_working_set_size
                .min(facts.physical_memory_bytes),
            os_headroom_bytes: omega::sized::LOAD_TIME_FIT_OS_HEADROOM_BYTES,
        };
        // Page-rounded on the dense class only: the checkpoint's whole
        // mmap is ONE no-copy `MTLBuffer`
        // (`omega::metal::checkpoint_mapping_offset`'s own doc), so the
        // real device allocation is `file_bytes.len()` rounded up to a
        // page, not the plain sum of per-tensor byte counts (which excludes
        // the GGUF header/metadata region) -- rounding the dense class
        // absorbs that difference without inventing a fourth bucket for a
        // few-KB header.
        let weights = crate::memory_fit::WeightClassBytes {
            dense_bytes: self
                .checkpoint_weight_bytes
                .dense_bytes
                .next_multiple_of(omega::metal::page_size() as u64),
            ..self.checkpoint_weight_bytes
        };
        let requested_context_length = serving_config.context_length;
        let (context_length, outcome) = crate::memory_fit::fit_context_length(
            weights,
            self.architecture.block_count,
            self.architecture.kv_heads,
            self.architecture.head_dim,
            requested_context_length,
            omega::sized::LOAD_TIME_FIT_ARENA_ALLOWANCE_BYTES,
            limit,
        )?;
        #[cfg(feature = "instrument")]
        {
            let budget = crate::memory_fit::MemoryBudget::derive(
                weights,
                self.architecture.block_count,
                self.architecture.kv_heads,
                self.architecture.head_dim,
                context_length,
                omega::sized::LOAD_TIME_FIT_ARENA_ALLOWANCE_BYTES,
            );
            info!(
                dense_weights_bytes = budget.dense_weights_bytes,
                expert_weights_bytes = budget.expert_weights_bytes,
                table_weights_bytes = budget.table_weights_bytes,
                kv_cache_bytes = budget.kv_cache_bytes,
                ssm_state_bytes = budget.ssm_state_bytes,
                arena_allowance_bytes = budget.arena_allowance_bytes,
                total_bytes = budget.total_bytes(),
                limit_bytes = limit.limit_bytes,
                os_headroom_bytes = limit.os_headroom_bytes,
                "memory_budget: load-time device-memory budget derived from checkpoint shape, by class"
            );
        }
        if matches!(outcome, crate::memory_fit::FitOutcome::ReducedContext { .. }) {
            #[cfg(feature = "instrument")]
            if let crate::memory_fit::FitOutcome::ReducedContext { from, to } = outcome {
                proxima_telemetry::warn!(
                    from = from,
                    to = to,
                    "context_length_reduced: requested context length did not fit, reduced \
                     to the largest value that does"
                );
            }
            serving_config.context_length = context_length;
        }
        Ok(())
    }

    /// Shared by [`Self::generate_with_serving_config`] and this crate's
    /// own metal-path tests, which need to read `runtime`'s plan-cache
    /// hit/miss counters after the loop finishes -- a caller reachable
    /// only through the public method above never sees `runtime` at all.
    /// Thin delegation to [`Self::run_decode_loop_observed`] with no forced
    /// continuation and a no-op logits sink, so every existing call site
    /// keeps its exact pre-existing behavior.
    pub(crate) fn run_decode_loop(
        &self,
        prompt: &str,
        max_tokens: usize,
        serving_config: &ServingConfig,
        runtime: &mut BackendRuntime,
    ) -> Result<(Vec<u32>, String, bool), InteropError> {
        self.run_decode_loop_observed(
            prompt,
            max_tokens,
            serving_config,
            runtime,
            None,
            &mut LogitsSink::Discard,
            &mut |_event| Control::Continue,
        )
    }

    /// [`Self::generate_with_serving_config`], plus a `TokenEvent` for
    /// every step [`decode_until_stop_or_budget`] already produces --
    /// [`Self::run_decode_loop_observed`]'s own loop, unchanged, given a
    /// real `on_token` instead of `run_decode_loop`'s `&mut |_| Continue`.
    /// There is one decode loop in this crate; this and
    /// [`Self::generate_with_serving_config`] are the same call with
    /// different callbacks, never two implementations of the loop itself.
    ///
    /// `on_token` sees exactly what [`TokenEvent`]'s own field docs promise:
    /// one [`Phase::Prefill`] event at step `0` (prompt token count, that
    /// step's own forward-pass latency), then one [`Phase::Token`] event
    /// per generated token, `text_piece`s concatenating to this call's
    /// returned `String` on the same ids as its returned `Vec<u32>`.
    /// Returning [`Control::Stop`] from any call ends decoding after that
    /// token, same as [`Control::Stop`]'s own doc: this call then returns
    /// `finished = false`, exactly like running out of `max_tokens`, never
    /// mistaken for the model's own eos.
    ///
    /// # Errors
    ///
    /// Same as [`Self::generate_with_serving_config`].
    pub fn generate_streaming(
        &self,
        prompt: &str,
        max_tokens: usize,
        serving_config: ServingConfig,
        on_token: &mut dyn FnMut(TokenEvent<'_>) -> Control,
    ) -> Result<(Vec<u32>, String, bool), InteropError> {
        let mut runtime = BackendRuntime::new(&serving_config);
        self.run_decode_loop_observed(
            prompt,
            max_tokens,
            &serving_config,
            &mut runtime,
            None,
            &mut LogitsSink::Discard,
            on_token,
        )
    }

    /// [`Self::run_decode_loop`]'s own body, plus the two hooks
    /// [`crate::quality::quality_report`] needs to score a variant against
    /// a reference through this SAME cached decode loop rather than a
    /// second, uncached one: `token_override` -- when `Some`, step
    /// `_step`'s emitted token is `token_override[_step]` instead of this
    /// call's own greedy sample, so a second [`LoadedModel`] can be driven
    /// through the identical token trajectory a first one already decided
    /// on (teacher forcing) -- and `logits_sink`, called every step with
    /// that step's own last-position logits (the same slice this loop
    /// already slices out of `evaluated` to sample from), so a caller can
    /// read off per-step logits without a parallel, uncached forward pass.
    /// Both are no-ops for [`Self::run_decode_loop`]'s own callers.
    /// `on_token` is [`decode_until_stop_or_budget`]'s own per-step callback,
    /// threaded straight through -- `&mut |_| Control::Continue` for every
    /// caller that does not need it, [`Self::generate_streaming`]'s real one
    /// for the one that does.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_decode_loop_observed(
        &self,
        prompt: &str,
        max_tokens: usize,
        serving_config: &ServingConfig,
        runtime: &mut BackendRuntime,
        token_override: Option<&[u32]>,
        logits_sink: &mut LogitsSink,
        on_token: &mut dyn FnMut(TokenEvent<'_>) -> Control,
    ) -> Result<(Vec<u32>, String, bool), InteropError> {
        let ids = proxima_tokenizer::encode_with_bos_eos(
            prompt,
            &self.vocab,
            wants_bos(&self.vocab),
            self.vocab.add_eos_token().unwrap_or(false),
        )?;
        // The repetition-penalty filter's own window: prompt tokens included,
        // matching upstream (`tools/main/main.cpp:725` feeds prompt tokens
        // through the same `common_sampler_accept` generated tokens use), grown
        // by one id every decode step below. `sample_config`/`rng` are built
        // once and threaded through every step -- the same seeded
        // `fastrand::Rng` this workspace already uses for every other
        // deterministic-by-seed pipe, drawn from progressively rather than
        // reseeded per token, mirroring upstream's own one-`std::mt19937`-per-
        // sampler-chain lifetime (`proxima_tokenizer::sample`'s own doc).
        let mut token_history: Vec<u32> = ids.clone();
        let repeat_window = serving_config.repeat_last_n.max(0) as usize;
        let sample_config = SamplingConfig {
            temperature: serving_config.temperature,
            top_k: serving_config.top_k,
            top_p: serving_config.top_p,
            min_p: serving_config.min_p,
            repeat_penalty: serving_config.repeat_penalty,
            frequency_penalty: serving_config.frequency_penalty,
            presence_penalty: serving_config.presence_penalty,
        };
        let mut rng = fastrand::Rng::with_seed(serving_config.seed);

        // Persistent device-resident KV: only reachable when this build was
        // compiled with `metal-output-placement`, this checkpoint built a
        // single-range program (`LoadedModel::single_range`'s own doc --
        // `None` for any mixture-of-experts or qwen35 checkpoint), AND this
        // call's own `ServingConfig` selected the Metal backend
        // (`runtime.is_metal()`). CPU decode, any MoE checkpoint, and the
        // qwen35 hybrid path always fall through to the two-range
        // `layer_roots` path below, byte-for-byte unchanged.
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        if runtime.is_metal()
            && let Some(single_range) = &self.single_range
        {
            return self.run_decode_loop_placed_kv(
                single_range,
                ids.clone(),
                token_history.clone(),
                repeat_window,
                sample_config,
                rng.clone(),
                max_tokens,
                serving_config,
                runtime,
                token_override,
                logits_sink,
                on_token,
            );
        }

        let cache_names: Vec<LayerCacheNames> = self
            .layer_roots
            .iter()
            .enumerate()
            .map(|(layer, roots)| match roots {
                Qwen35LayerRoots::Attention(_) => LayerCacheNames::Attention {
                    k_even: alloc::format!("kv_cache.{layer}.k_even"),
                    k_odd: alloc::format!("kv_cache.{layer}.k_odd"),
                    v: alloc::format!("kv_cache.{layer}.v"),
                },
                Qwen35LayerRoots::DenseAttention(_) => LayerCacheNames::DenseAttention {
                    k_first: alloc::format!("kv_cache.{layer}.k_first"),
                    k_second: alloc::format!("kv_cache.{layer}.k_second"),
                    k_pass: alloc::format!("kv_cache.{layer}.k_pass"),
                    v: alloc::format!("kv_cache.{layer}.v"),
                },
                Qwen35LayerRoots::Ssm { .. } => LayerCacheNames::Ssm {
                    conv_history: alloc::format!("ssm_cache.{layer}.conv_history"),
                    state: alloc::format!("ssm_cache.{layer}.state"),
                },
            })
            .collect();
        let mut layer_caches: Vec<LayerCacheState> = self
            .layer_roots
            .iter()
            .map(|roots| match roots {
                Qwen35LayerRoots::Attention(_) => LayerCacheState::Attention(LayerCache::new()),
                Qwen35LayerRoots::DenseAttention(_) => {
                    LayerCacheState::DenseAttention(Qwen35DenseAttentionCache::new())
                }
                Qwen35LayerRoots::Ssm { .. } => LayerCacheState::Ssm(SsmLayerCache::new(
                    self.qwen35_ssm_shape.unwrap_or(Qwen35SsmShape {
                        qkv_dim: 0,
                        conv_rows: 0,
                        state_len: 0,
                    }),
                )),
            })
            .collect();
        // One [`KvPadScratch`] per layer, reused across every step of this
        // call -- only ever filled for a [`LayerCacheState::Attention`]
        // layer (the only cache shape `mistral_cached_forward_program_with_experts`
        // produces, `Qwen35LayerRoots`'s own doc), left empty and unread for
        // every `DenseAttention`/`Ssm` layer a qwen35 checkpoint carries.
        let mut kv_pad_scratch: Vec<KvPadScratch> =
            self.layer_roots.iter().map(|_| KvPadScratch::new()).collect();
        // [`Qwen35DenseAttentionPadScratch`]'s own doc: the `Attention` arm's
        // padding above is not enough on its own -- a `DenseAttention` layer
        // shares the identical `Extent::Symbolic(1)` slot, so it needs the
        // same treatment or a bucketed `symbols[1]` reads past a shorter,
        // unpadded buffer on every qwen35 checkpoint.
        let mut qwen35_dense_pad_scratch: Vec<Qwen35DenseAttentionPadScratch> = self
            .layer_roots
            .iter()
            .map(|_| Qwen35DenseAttentionPadScratch::new())
            .collect();
        let kv_heads = self.architecture.kv_heads as usize;
        let head_dim = self.architecture.head_dim as usize;
        let pairs = head_dim / 2;
        let attn_head_dim = self.qwen35_attn_head_dim.unwrap_or(0) as usize;
        let pass_dim = attn_head_dim.saturating_sub(head_dim);

        // The caller's own knowledge of which named blocks are STATIC --
        // bound once in `LoadedModel::load` and never mutated again -- fixed
        // for this whole call, unlike `ids`/`eps`/`rope_cos`/`rope_sin` and
        // the KV cache's own blocks below, which change every step. This is
        // exactly the distinction `BackendRuntime::evaluate` hands to
        // `mark_resident` so the Metal driver's device-buffer cache can tell
        // "same name, same bytes" apart from "same name, new bytes" without
        // ever keying on name itself (`omega::metal::Plan::mark_resident`'s
        // own doc). Computed once, not per token: these names never change.
        let resident_names: BTreeSet<&str> = self.resident_names();

        let prompt_token_count = ids.len();
        let mut cached_len = 0usize;
        let mut next_ids = ids;
        let vocab_size = self.architecture.vocab as usize;

        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(
            &self.vocab,
            max_tokens,
            prompt_token_count,
            |_step| {
                // ROW 130's own fix, built: every counter this step's
                // `evaluate_ms` decomposition reads is zeroed HERE, at step
                // start, and read back after `evaluate_ticks` below is computed
                // -- a single step's own cost, measured directly inside one
                // process, never inferred by differencing two independent
                // launches' cumulative-since-start counters (that differencing
                // is exact for the integer counts ROW 129 used it for, and NOT
                // for timings -- ROW 130's own postmortem on why it produced a
                // sub-bucket larger than its parent and a negative duration).
                #[cfg(feature = "instrument")]
                proxima_tensor::instrument::reset_step();
                #[cfg(feature = "instrument")]
                let step_started = read_ticks();

                let new_count = next_ids.len();
                #[cfg(feature = "instrument")]
                let apply_serving_config_started = read_ticks();
                apply_serving_config(serving_config, cached_len + new_count)?;
                #[cfg(feature = "instrument")]
                let apply_serving_config_ticks = elapsed_ticks(apply_serving_config_started);

                #[cfg(feature = "instrument")]
                let build_position_inputs_started = read_ticks();
                let inputs = build_position_inputs(
                    &next_ids,
                    cached_len,
                    self.architecture.head_dim,
                    self.architecture.rope_freq_base,
                    self.architecture.rms_epsilon,
                );
                #[cfg(feature = "instrument")]
                let build_position_inputs_ticks = elapsed_ticks(build_position_inputs_started);

                let mut named_blocks: Vec<(&str, QuantizedBlock)> = Vec::with_capacity(
                    self.weights.owned.len()
                        + self.weights.packed.len()
                        + self.weights.packed_owned.len()
                        + 3
                        + layer_caches.len() * 3,
                );
                #[cfg(feature = "instrument")]
                let named_blocks_weights_started = read_ticks();
                named_blocks.push(("ids", QuantizedBlock::Float32(inputs.ids_f32.as_slice())));
                for (name, data) in &self.weights.owned {
                    named_blocks.push((name.as_str(), QuantizedBlock::Float32(data.as_slice())));
                }
                for (name, block) in &self.weights.packed {
                    named_blocks.push((name.as_str(), *block));
                }
                for (name, bytes, kind) in &self.weights.packed_owned {
                    named_blocks.push((name.as_str(), kind.as_block(bytes)));
                }
                named_blocks.push(("eps", QuantizedBlock::Float32(inputs.epsilon.as_slice())));
                named_blocks.push(("rope_cos", QuantizedBlock::Float32(inputs.cos.as_slice())));
                named_blocks.push(("rope_sin", QuantizedBlock::Float32(inputs.sin.as_slice())));
                // `mistral_cached_forward_program_with_experts`'s own
                // `cached_len` `Op::Input` -- always present regardless of
                // `ServingConfig::kv_bucket_tokens` (`proxima_tensor::bind::
                // cached_attention_candidates`'s own doc on the runtime bound
                // that reads it), so this scalar is fed on every step,
                // bucketed or not.
                let cached_len_scalar = [cached_len as f32];
                named_blocks.push(("cached_len", QuantizedBlock::Float32(&cached_len_scalar)));
                #[cfg(feature = "instrument")]
                let named_blocks_weights_ticks = elapsed_ticks(named_blocks_weights_started);
                // Rounds `cached_len` up to `ServingConfig::kv_bucket_tokens`
                // (`kv_extent`'s own doc) -- `usize::MAX` in place of the
                // placed-KV path's fixed buffer capacity: the two-range KV
                // cache below is a growing `Vec`, not a preallocated
                // device buffer, so there is no hard cap to clamp against.
                let kv_bound_extent = kv_extent(cached_len, usize::MAX, serving_config.kv_bucket_tokens);
                let kv_pad_shape = KvPadShape {
                    bound_extent: kv_bound_extent,
                    kv_heads,
                    pairs,
                    head_dim,
                };
                // Same `kv_bound_extent`, same [`Extent::Symbolic(1)`] slot
                // (`Qwen35DenseAttentionPadScratch`'s own doc) -- a
                // `DenseAttention` layer's row widths, not `Attention`'s.
                let qwen35_dense_pad_shape = Qwen35DenseAttentionPadShape {
                    bound_extent: kv_bound_extent,
                    kv_heads,
                    pairs,
                    pass_dim,
                    attn_head_dim,
                };

                // KV-cache HOST -> DEVICE traffic: every named block below is the
                // FULL accumulated history (`LayerCache::append` only grows these,
                // never truncates), so this is the full `cached_len`-sized array
                // re-bound as a model input every single step -- not the
                // `new_count`-sized increment. Measured directly as element
                // counts read off the `Vec`s themselves (a size, not a timing),
                // so it is exact and needs no instrumentation to be turned on.
                #[cfg(feature = "instrument")]
                let kv_cache_upload_elements: u64 = layer_caches
                    .iter()
                    .map(|cache| match cache {
                        LayerCacheState::Attention(cache) => {
                            (cache.k_even.len() + cache.k_odd.len() + cache.v.len()) as u64
                        }
                        LayerCacheState::DenseAttention(cache) => {
                            (cache.k_first.len()
                                + cache.k_second.len()
                                + cache.k_pass.len()
                                + cache.v.len()) as u64
                        }
                        LayerCacheState::Ssm(cache) => {
                            (cache.conv_history.len() + cache.state.len()) as u64
                        }
                    })
                    .sum();
                #[cfg(feature = "instrument")]
                let named_blocks_kv_started = read_ticks();
                // Two passes over the same `layer`/`cache` pairing, not one
                // interleaved pass: `KvPadScratch::named_blocks` below
                // borrows `kv_pad_scratch[layer]` immutably for as long as
                // `named_blocks` (read by `evaluate` after this loop) holds
                // it, so a later iteration's `&mut kv_pad_scratch[other_layer]`
                // would conflict even though the indices never alias --
                // the borrow checker sees one `Vec`, not per-index slots.
                // Filling every layer's scratch first, then borrowing every
                // layer's scratch second, keeps the two borrow kinds in
                // disjoint passes instead of interleaved per iteration.
                for (layer, cache) in layer_caches.iter().enumerate() {
                    match cache {
                        LayerCacheState::Attention(cache) => {
                            kv_pad_scratch[layer].fill(cache, &kv_pad_shape);
                        }
                        LayerCacheState::DenseAttention(cache) => {
                            qwen35_dense_pad_scratch[layer].fill(cache, &qwen35_dense_pad_shape);
                        }
                        LayerCacheState::Ssm(_) => {}
                    }
                }
                for (layer, names) in cache_names.iter().enumerate() {
                    match (names, &layer_caches[layer]) {
                        (LayerCacheNames::Attention { k_even, k_odd, v }, LayerCacheState::Attention(_)) => {
                            named_blocks.extend(kv_pad_scratch[layer].named_blocks(
                                k_even,
                                k_odd,
                                v,
                                &kv_pad_shape,
                            ));
                        }
                        (
                            LayerCacheNames::DenseAttention {
                                k_first,
                                k_second,
                                k_pass,
                                v,
                            },
                            LayerCacheState::DenseAttention(_),
                        ) => {
                            named_blocks.extend(qwen35_dense_pad_scratch[layer].named_blocks(
                                k_first,
                                k_second,
                                k_pass,
                                v,
                                &qwen35_dense_pad_shape,
                            ));
                        }
                        (
                            LayerCacheNames::Ssm {
                                conv_history,
                                state,
                            },
                            LayerCacheState::Ssm(cache),
                        ) => {
                            named_blocks.extend(cache.named_blocks(conv_history, state));
                        }
                        _ => unreachable!(
                            "cache_names/layer_caches built from the same layer_roots, in lockstep"
                        ),
                    }
                }
                #[cfg(feature = "instrument")]
                let named_blocks_kv_ticks = elapsed_ticks(named_blocks_kv_started);

                let symbols = [new_count as u64, kv_bound_extent as u64];
                let mut roots: Vec<NodeId> = Vec::with_capacity(1 + self.layer_roots.len() * 3);
                roots.push(self.logits_root);
                for roots_for_layer in &self.layer_roots {
                    match roots_for_layer {
                        Qwen35LayerRoots::Attention((even, odd, value)) => {
                            roots.push(*even);
                            roots.push(*odd);
                            roots.push(*value);
                        }
                        Qwen35LayerRoots::DenseAttention((first, second, pass, value)) => {
                            roots.push(*first);
                            roots.push(*second);
                            roots.push(*pass);
                            roots.push(*value);
                        }
                        Qwen35LayerRoots::Ssm {
                            qkv_mixed,
                            state_out,
                        } => {
                            roots.push(*qkv_mixed);
                            roots.push(*state_out);
                        }
                    }
                }

                #[cfg(feature = "instrument")]
                let evaluate_started = read_ticks();
                // `PROXIMA_METAL_OP_PROFILE_STEP` -- diagnostic-only, `instrument`-gated,
                // default-off: unset in every production run, so `runtime.evaluate`
                // is the only path a caller without this env var ever takes. When
                // set to this step's own index, this ONE step instead runs
                // `evaluate_op_timed` (per-op command buffers, see that method's own
                // doc for the cost) and prints the per-op GPU attribution this
                // crate's own discipline log needed to settle the `gpu_exec`
                // investigation. Every other step, and every run without the env
                // var, is byte-for-byte the pre-existing path.
                #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                let evaluated = match std::env::var("PROXIMA_METAL_OP_PROFILE_STEP")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                {
                    Some(target) if target == _step => {
                        let (evaluated, timings) = runtime.evaluate_op_timed(
                            &self.program,
                            &symbols,
                            &named_blocks,
                            &roots,
                            &resident_names,
                        )?;
                        report_op_timings(_step, &timings);
                        evaluated
                    }
                    _ => runtime.evaluate(
                        &self.program,
                        &symbols,
                        &named_blocks,
                        &roots,
                        &resident_names,
                    )?,
                };
                #[cfg(not(all(feature = "instrument", feature = "metal", target_os = "macos")))]
                let evaluated = runtime.evaluate(
                    &self.program,
                    &symbols,
                    &named_blocks,
                    &roots,
                    &resident_names,
                )?;
                #[cfg(feature = "instrument")]
                let evaluate_ticks = elapsed_ticks(evaluate_started);
                #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                let metal_stage = metal_stage_totals();

                // KV-cache DEVICE -> HOST readback + host append: unlike the
                // upload above, `evaluated.get(*even)` etc. is this step's own
                // `new_count`-sized OUTPUT increment (what the forward computed
                // for the newly-added positions), which `LayerCache::append`
                // then extends onto the growing history -- so this side is
                // expected to stay FLAT across tokens where the upload side
                // grows. `layer_cache_append` ticks/bytes below are the pure
                // host `extend_from_slice` memcpy cost, distinct from the GPU
                // readback `metal_stage_totals` already reports.
                #[cfg(feature = "instrument")]
                let layer_cache_append_started = read_ticks();
                #[cfg(feature = "instrument")]
                let mut layer_cache_append_elements: u64 = 0;
                for (layer, roots_for_layer) in self.layer_roots.iter().enumerate() {
                    match (roots_for_layer, &mut layer_caches[layer]) {
                        (
                            Qwen35LayerRoots::Attention((even, odd, value)),
                            LayerCacheState::Attention(cache),
                        ) => {
                            let (even_data, _) = evaluated
                                .get(*even)
                                .ok_or(InteropError::MissingEvaluatedNode { node: *even })?;
                            let (odd_data, _) = evaluated
                                .get(*odd)
                                .ok_or(InteropError::MissingEvaluatedNode { node: *odd })?;
                            let (value_data, _) = evaluated
                                .get(*value)
                                .ok_or(InteropError::MissingEvaluatedNode { node: *value })?;
                            #[cfg(feature = "instrument")]
                            {
                                layer_cache_append_elements +=
                                    (even_data.len() + odd_data.len() + value_data.len()) as u64;
                            }
                            cache.append(even_data, odd_data, value_data);
                        }
                        (
                            Qwen35LayerRoots::DenseAttention((first, second, pass, value)),
                            LayerCacheState::DenseAttention(cache),
                        ) => {
                            let (first_data, _) = evaluated
                                .get(*first)
                                .ok_or(InteropError::MissingEvaluatedNode { node: *first })?;
                            let (second_data, _) = evaluated
                                .get(*second)
                                .ok_or(InteropError::MissingEvaluatedNode { node: *second })?;
                            let (pass_data, _) = evaluated
                                .get(*pass)
                                .ok_or(InteropError::MissingEvaluatedNode { node: *pass })?;
                            let (value_data, _) = evaluated
                                .get(*value)
                                .ok_or(InteropError::MissingEvaluatedNode { node: *value })?;
                            #[cfg(feature = "instrument")]
                            {
                                layer_cache_append_elements += (first_data.len()
                                    + second_data.len()
                                    + pass_data.len()
                                    + value_data.len())
                                    as u64;
                            }
                            cache.append(first_data, second_data, pass_data, value_data);
                        }
                        (
                            Qwen35LayerRoots::Ssm {
                                qkv_mixed,
                                state_out,
                            },
                            LayerCacheState::Ssm(cache),
                        ) => {
                            let (qkv_mixed_data, _) = evaluated
                                .get(*qkv_mixed)
                                .ok_or(InteropError::MissingEvaluatedNode { node: *qkv_mixed })?;
                            let (state_out_data, _) = evaluated
                                .get(*state_out)
                                .ok_or(InteropError::MissingEvaluatedNode { node: *state_out })?;
                            #[cfg(feature = "instrument")]
                            {
                                layer_cache_append_elements +=
                                    (qkv_mixed_data.len() + state_out_data.len()) as u64;
                            }
                            // `Self::load`'s own invariant: an `Ssm` entry in
                            // `layer_roots` exists only when `qwen35_ssm_shape`
                            // was derived alongside it (both come from the same
                            // `crate::qwen35::Qwen35Architecture`), so this
                            // fallback shape is never actually read.
                            let shape = self.qwen35_ssm_shape.unwrap_or(Qwen35SsmShape {
                                qkv_dim: 0,
                                conv_rows: 0,
                                state_len: 0,
                            });
                            cache.advance(qkv_mixed_data, state_out_data, shape);
                        }
                        _ => unreachable!(
                            "layer_roots/layer_caches built from the same layer_roots, in lockstep"
                        ),
                    }
                }
                #[cfg(feature = "instrument")]
                let layer_cache_append_ticks = elapsed_ticks(layer_cache_append_started);
                #[cfg(feature = "instrument")]
                let cached_len_before_step = cached_len;
                cached_len += new_count;

                let (logits, _shape) =
                    evaluated
                        .get(self.logits_root)
                        .ok_or(InteropError::MissingEvaluatedNode {
                            node: self.logits_root,
                        })?;
                let last_position = &logits[(new_count - 1) * vocab_size..new_count * vocab_size];
                #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                let barriers_step = metal_stage.barriers_emitted;
                #[cfg(not(all(feature = "instrument", feature = "metal", target_os = "macos")))]
                let barriers_step = 0_u64;
                logits_sink.observe(last_position, barriers_step);

                #[cfg(feature = "instrument")]
                let greedy_pick_started = read_ticks();
                let token_id = match token_override.and_then(|forced| forced.get(_step)) {
                    Some(&forced_token) => forced_token,
                    None => {
                        let recent_window_start = token_history.len().saturating_sub(repeat_window);
                        let recent_tokens = &token_history[recent_window_start..];
                        sample_next_token(last_position, recent_tokens, sample_config, &mut rng)
                            .ok_or(InteropError::EmptyLogits)?
                    }
                };
                token_history.push(token_id);
                #[cfg(feature = "instrument")]
                let greedy_pick_ticks = elapsed_ticks(greedy_pick_started);
                next_ids = alloc::vec![token_id];

                #[cfg(feature = "instrument")]
                {
                    emit_token_breakdown(&TokenBreakdown {
                        step: _step,
                        new_count,
                        cached_len_before: cached_len_before_step,
                        step_wall_ticks: elapsed_ticks(step_started),
                        apply_serving_config_ticks,
                        build_position_inputs_ticks,
                        named_blocks_weights_ticks,
                        named_blocks_kv_ticks,
                        kv_cache_upload_bytes: kv_cache_upload_elements * 4,
                        evaluate_ticks,
                        layer_cache_append_ticks,
                        layer_cache_append_bytes: layer_cache_append_elements * 4,
                        greedy_pick_ticks,
                    });
                    // ROW 130's per-step-reset attribution: kernel / dispatch+
                    // setup / park+spin+wake, all on the CALLING thread's own
                    // wall clock (never summed across the cohort's other worker
                    // threads, which run concurrently with it, not serially
                    // inside it -- see `CohortLeaderAttribution`'s own doc).
                    // `evaluate_ns` is this step's own tick-based total, already
                    // reset per step by `reset_step`; `residual_ns` is
                    // everything `evaluate_ms` paid for that these three terms
                    // do not name -- non-matmul ops (elementwise/reduce/scan),
                    // quantize/transpose bookkeeping, and staged-batch setup
                    // outside the cohort round itself. `saturating_sub` so a
                    // negative residual is impossible to construct by
                    // arithmetic; reported as 0 with `residual_underflow=true`
                    // if the three named terms would have exceeded the parent,
                    // which is itself a sanity-gate failure worth seeing rather
                    // than silently wrapping.
                    let attribution = proxima_tensor::instrument::cohort_leader_attribution();
                    let evaluate_ns = ticks_to_nanos(evaluate_ticks);
                    let named_ns = attribution.kernel_nanos
                        + attribution.dispatch_nanos
                        + attribution.park_spin_wake_nanos;
                    let residual_underflow = named_ns > evaluate_ns;
                    let residual_ns = evaluate_ns.saturating_sub(named_ns);
                    info!(
                        step = _step as u64,
                        evaluate_ms = evaluate_ns as f64 / 1e6,
                        kernel_ms = attribution.kernel_nanos as f64 / 1e6,
                        dispatch_ms = attribution.dispatch_nanos as f64 / 1e6,
                        park_spin_wake_ms = attribution.park_spin_wake_nanos as f64 / 1e6,
                        residual_ms = residual_ns as f64 / 1e6,
                        residual_underflow,
                        named_plus_residual_ms = (named_ns + residual_ns) as f64 / 1e6,
                        cached_attention_ops =
                            proxima_tensor::instrument::path_totals().op_kind_cached_attention,
                        "token_attribution: per-step kernel/dispatch/park-spin-wake split"
                    );
                    // ROW 140's own redundant-activation-quantize hypothesis
                    // check: `total_calls` vs `distinct_nodes` across every
                    // matmul reduce node this step evaluated. 1:1 kills the
                    // hypothesis; a ratio near the QKV/gate-up fan-out (2-3x)
                    // confirms it.
                    let (quantize_total_calls, quantize_distinct_nodes) =
                        proxima_tensor::instrument::quantize_activation_call_stats();
                    let quantize_cache_hits =
                        proxima_tensor::instrument::QUANTIZE_ACTIVATION_CACHE_HITS.get();
                    info!(
                        step = _step as u64,
                        total_calls = quantize_total_calls,
                        distinct_nodes = quantize_distinct_nodes,
                        cache_hits = quantize_cache_hits,
                        "token_quantize_calls: redundant-activation-quantize hypothesis check"
                    );
                    #[cfg(all(feature = "metal", target_os = "macos"))]
                    emit_token_breakdown_metal(
                        _step,
                        &metal_stage,
                        runtime.plans_len(),
                        runtime.plan_hits,
                        runtime.plan_misses,
                    );
                    // This arm's `kv_cache.{layer}.*` blocks are ordinary
                    // named blocks folded into `metal_stage`'s weight
                    // counters above (`emit_device_memory_by_class`'s own
                    // doc), never a separately sized device allocation --
                    // `None` here is honest, not a placeholder.
                    #[cfg(all(feature = "metal", target_os = "macos"))]
                    if _step == 0 {
                        emit_device_memory_by_class(
                            _step,
                            metal_stage.block_nocopy_bound_bytes,
                            metal_stage.block_copied_bytes,
                            metal_stage.block_offset_bound_bytes,
                            None,
                            omega::metal::current_allocated_size().unwrap_or(0),
                            phys_footprint_bytes(),
                        );
                    }
                }

                Ok(token_id)
            },
            on_token,
        )?;

        let text = proxima_tokenizer::decode(&generated_ids, &self.vocab)?;
        Ok((generated_ids, text, stopped_by_eos))
    }

    /// [`Self::run_decode_loop`]'s persistent-device-resident-KV arm:
    /// [`SingleRangeProgram`] in place of the two-range `program`, one
    /// [`PlacedBuffer`] triple per layer (`k_even`/`k_odd`/`v`) allocated
    /// ONCE, at `(prompt_len + max_tokens).min(context_length)` capacity
    /// (this call's actual reachable position count, never
    /// `context_length` itself -- see the sizing comment in the body) and
    /// held for this whole call, in place of `LayerCache`'s per-step
    /// `extend_from_slice` growth. Each step places this step's own
    /// freshly rotated key/value (`single_range.cache_roots[layer]`, the
    /// OUTPUT side) into the buffer's tail at `cached_len * row_bytes`, and
    /// reads the SAME buffer back as this step's `kv_cache.{layer}.*`
    /// input (`single_range.cache_input_nodes[layer]`) covering `[0,
    /// cached_len + new_count)` -- one program, one command buffer, the
    /// write visible to the later read through Metal's own whole-resource
    /// hazard tracking (`omega::metal::execute_plan_with_placements`'s own
    /// doc, "Within-call aliasing"). No host round trip either direction:
    /// the cache never leaves the device, and the per-layer cache roots
    /// are not requested as `Evaluated` outputs at all (only
    /// `single_range.logits_root` is).
    ///
    /// `ids`/`token_history`/`sample_config`/`rng` arrive already built by
    /// [`Self::run_decode_loop`]'s shared prefix -- this method's own body
    /// starts exactly where that function's two-range arm does.
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    #[allow(clippy::too_many_arguments)]
    fn run_decode_loop_placed_kv(
        &self,
        single_range: &SingleRangeProgram,
        ids: Vec<u32>,
        mut token_history: Vec<u32>,
        repeat_window: usize,
        sample_config: SamplingConfig,
        mut rng: fastrand::Rng,
        max_tokens: usize,
        serving_config: &ServingConfig,
        runtime: &mut BackendRuntime,
        token_override: Option<&[u32]>,
        logits_sink: &mut LogitsSink,
        on_token: &mut dyn FnMut(TokenEvent<'_>) -> Control,
    ) -> Result<(Vec<u32>, String, bool), InteropError> {
        let prompt_token_count = ids.len();
        let block_count = self.architecture.block_count as usize;
        let kv_heads = self.architecture.kv_heads as usize;
        let head_dim = self.architecture.head_dim as usize;
        let pairs = head_dim / 2;
        let context_length = serving_config.context_length as usize;

        // Sized from what THIS call can actually reach (`prompt_len +
        // max_tokens`), not `context_length` (default 131_072). Capping the
        // allocation at `positions_needed` (never more than
        // `context_length`) cannot admit a step this call could not already
        // reach -- `apply_serving_config` below still rejects any step whose
        // `merged_len` would exceed `context_length`.
        let positions_needed = (ids.len() + max_tokens).min(context_length);

        let row_bytes_even_odd = kv_heads * pairs * core::mem::size_of::<f32>();
        let row_bytes_v = kv_heads * head_dim * core::mem::size_of::<f32>();
        let capacity_even_odd = positions_needed * row_bytes_even_odd;
        let capacity_v = positions_needed * row_bytes_v;

        let mut k_even_buffers = Vec::with_capacity(block_count);
        let mut k_odd_buffers = Vec::with_capacity(block_count);
        let mut v_buffers = Vec::with_capacity(block_count);
        for _ in 0..block_count {
            k_even_buffers.push(allocate_placed_buffer(capacity_even_odd)?);
            k_odd_buffers.push(allocate_placed_buffer(capacity_even_odd)?);
            v_buffers.push(allocate_placed_buffer(capacity_v)?);
        }
        // `kv_extent` (unconditional, keyed off
        // `ServingConfig::kv_bucket_tokens`, see that field's own doc)
        // reads `bucket` rows per step whenever `bucket_tokens > 1`,
        // `bucket > merged_len` -- the tail `[merged_len, bucket)` was
        // never written by this call yet. A freshly allocated
        // `MTLBuffer`'s contents are undefined
        // (`omega::metal::zero_placed_buffer`'s own doc), so that tail is
        // zeroed ONCE here, at allocation, rather than paying a per-step
        // re-zero: any row a later step reads was either zeroed here or
        // overwritten by a real rotated key/value this same call already
        // wrote, since `cached_len` only grows. Unconditional (not gated
        // on the `kv-capacity-bucket` cargo feature, which only pulls in
        // `proxima-tensor`'s CPU-side correctness proof and no longer
        // controls this runtime behaviour) because
        // `ServingConfig::default`'s `kv_bucket_tokens: 32` buckets on
        // every build regardless of which cargo features are enabled.
        for layer in 0..block_count {
            omega::metal::zero_placed_buffer(&k_even_buffers[layer], capacity_even_odd);
            omega::metal::zero_placed_buffer(&k_odd_buffers[layer], capacity_even_odd);
            omega::metal::zero_placed_buffer(&v_buffers[layer], capacity_v);
        }

        let kv_cache_names: Vec<(String, String, String)> = (0..block_count)
            .map(|layer| {
                (
                    alloc::format!("kv_cache.{layer}.k_even"),
                    alloc::format!("kv_cache.{layer}.k_odd"),
                    alloc::format!("kv_cache.{layer}.v"),
                )
            })
            .collect();

        // `prepare`'s own element-count check (`omega::metal::prepare`,
        // `found != expected`) runs against EVERY named block, placed or
        // not -- these three names are always input-placed below, so their
        // data is never read, only their LENGTH, which must match this
        // step's `merged_len * elements_per_position`. `resize` only grows
        // when `merged_len` grows past the previous step's value, the same
        // amortized cost `LayerCache::append`'s `extend_from_slice` paid,
        // minus the real data this scratch never holds and the device
        // upload `execute_plan_with_placements` skips for a placed input.
        let mut cache_length_scratch_even_odd: Vec<f32> = Vec::new();
        let mut cache_length_scratch_v: Vec<f32> = Vec::new();

        let resident_names: BTreeSet<&str> = self.resident_names();

        let mut cached_len = 0usize;
        let mut next_ids = ids;
        let vocab_size = self.architecture.vocab as usize;

        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(
            &self.vocab,
            max_tokens,
            prompt_token_count,
            |_step| {
                #[cfg(feature = "instrument")]
                proxima_tensor::instrument::reset_step();
                #[cfg(feature = "instrument")]
                let step_started = read_ticks();

                let new_count = next_ids.len();
                let merged_len = cached_len + new_count;
                #[cfg(feature = "instrument")]
                let apply_serving_config_started = read_ticks();
                apply_serving_config(serving_config, merged_len)?;
                #[cfg(feature = "instrument")]
                let apply_serving_config_ticks = elapsed_ticks(apply_serving_config_started);

                #[cfg(feature = "instrument")]
                let build_position_inputs_started = read_ticks();
                let inputs = build_position_inputs(
                    &next_ids,
                    cached_len,
                    self.architecture.head_dim,
                    self.architecture.rope_freq_base,
                    self.architecture.rms_epsilon,
                );
                #[cfg(feature = "instrument")]
                let build_position_inputs_ticks = elapsed_ticks(build_position_inputs_started);

                let mut named_blocks: Vec<(&str, QuantizedBlock)> = Vec::with_capacity(
                    self.weights.owned.len()
                        + self.weights.packed.len()
                        + self.weights.packed_owned.len()
                        + 4
                        + block_count * 3,
                );
                named_blocks.push(("ids", QuantizedBlock::Float32(inputs.ids_f32.as_slice())));
                #[cfg(feature = "instrument")]
                let named_blocks_weights_started = read_ticks();
                for (name, data) in &self.weights.owned {
                    named_blocks.push((name.as_str(), QuantizedBlock::Float32(data.as_slice())));
                }
                for (name, block) in &self.weights.packed {
                    named_blocks.push((name.as_str(), *block));
                }
                for (name, bytes, kind) in &self.weights.packed_owned {
                    named_blocks.push((name.as_str(), kind.as_block(bytes)));
                }
                named_blocks.push(("eps", QuantizedBlock::Float32(inputs.epsilon.as_slice())));
                named_blocks.push(("rope_cos", QuantizedBlock::Float32(inputs.cos.as_slice())));
                named_blocks.push(("rope_sin", QuantizedBlock::Float32(inputs.sin.as_slice())));
                let cached_len_scalar = [cached_len as f32];
                named_blocks.push(("cached_len", QuantizedBlock::Float32(&cached_len_scalar)));
                #[cfg(feature = "instrument")]
                let named_blocks_weights_ticks = elapsed_ticks(named_blocks_weights_started);

                // This step's `kv_cache.{layer}.*` `named_blocks` entries are
                // placeholder scratch, not the real KV cache -- see this
                // arm's own doc on why `execute_plan_with_placements` never
                // uploads them.
                #[cfg(feature = "instrument")]
                let named_blocks_kv_started = read_ticks();
                // `serving_config.kv_bucket_tokens == 1`: `kv_bound_extent
                // == merged_len`, unchanged from the pre-bucketing shape.
                // Otherwise rounded up to that many tokens and capped at
                // `positions_needed` (this call's own per-layer buffer row
                // count) -- see `kv_extent`'s own doc. This is BOTH the
                // scratch named-block length below (the strict
                // `found == expected` validator checks it against the
                // SAME extent the KV `Op::Input` leaves bind to) and
                // `symbols[1]` a few lines down, so the two can never
                // disagree.
                let kv_bound_extent =
                    kv_extent(merged_len, positions_needed, serving_config.kv_bucket_tokens);
                let even_odd_len = kv_bound_extent * kv_heads * pairs;
                let v_len = kv_bound_extent * kv_heads * head_dim;
                if cache_length_scratch_even_odd.len() < even_odd_len {
                    cache_length_scratch_even_odd.resize(even_odd_len, 0.0);
                }
                if cache_length_scratch_v.len() < v_len {
                    cache_length_scratch_v.resize(v_len, 0.0);
                }
                for names in &kv_cache_names {
                    named_blocks.push((
                        names.0.as_str(),
                        QuantizedBlock::Float32(&cache_length_scratch_even_odd[..even_odd_len]),
                    ));
                    named_blocks.push((
                        names.1.as_str(),
                        QuantizedBlock::Float32(&cache_length_scratch_even_odd[..even_odd_len]),
                    ));
                    named_blocks.push((
                        names.2.as_str(),
                        QuantizedBlock::Float32(&cache_length_scratch_v[..v_len]),
                    ));
                }
                #[cfg(feature = "instrument")]
                let named_blocks_kv_ticks = elapsed_ticks(named_blocks_kv_started);

                let mut input_placements: Vec<(NodeId, &PlacedBuffer, usize)> =
                    Vec::with_capacity(block_count * 3);
                let mut output_placements: Vec<(NodeId, &PlacedBuffer, usize)> =
                    Vec::with_capacity(block_count * 3);
                for layer in 0..block_count {
                    let (even_input, odd_input, value_input) =
                        single_range.cache_input_nodes[layer];
                    let (even_output, odd_output, value_output) = single_range.cache_roots[layer];
                    input_placements.push((even_input, &k_even_buffers[layer], 0));
                    input_placements.push((odd_input, &k_odd_buffers[layer], 0));
                    input_placements.push((value_input, &v_buffers[layer], 0));
                    output_placements.push((
                        even_output,
                        &k_even_buffers[layer],
                        cached_len * row_bytes_even_odd,
                    ));
                    output_placements.push((
                        odd_output,
                        &k_odd_buffers[layer],
                        cached_len * row_bytes_even_odd,
                    ));
                    output_placements.push((
                        value_output,
                        &v_buffers[layer],
                        cached_len * row_bytes_v,
                    ));
                }

                let symbols = [new_count as u64, kv_bound_extent as u64];
                // Every placed-output node must also be a `roots` entry, or
                // `prepare`'s `BoundOpBuilder::finish` (`proxima-tensor`'s
                // `bind.rs`) never force-materializes it and its write lands
                // wherever `held`'s end-of-walk flush happens to fall --
                // AFTER every layer's score already read the buffer, and
                // `omega::metal::prepare`'s own `prune_dead` pass drops it
                // from `resolved` entirely (see
                // `omega::metal::execute_plan_with_placements`'s own doc).
                // The two-range path above (`roots.push` for each
                // `cache_roots` entry) already relies on this; this arm was
                // missing it.
                let mut roots: Vec<NodeId> =
                    Vec::with_capacity(2 + single_range.cache_roots.len() * 3);
                roots.push(single_range.logits_root);
                if let Some(scratch) = single_range.duplicate_head_scratch {
                    roots.push(scratch);
                }
                for (even, odd, value) in &single_range.cache_roots {
                    roots.push(*even);
                    roots.push(*odd);
                    roots.push(*value);
                }
                #[cfg(feature = "instrument")]
                let evaluate_started = read_ticks();
                // ROW 329: this step's `execute_plan_with_placements_dispatch_timed`
                // encoder-split result, if the branch below actually took it
                // -- read out here (not inside the match arm) because
                // `report_encoder_split`'s own `gpu_exec_ms` needs
                // `metal_stage_totals`'s post-match snapshot, the same
                // snapshot-and-reset value `emit_token_breakdown_metal`
                // reads a few lines below, never a second counter read.
                #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                let mut encoder_split_ns: Option<(u64, u64)> = None;
                // `PROXIMA_METAL_OP_PROFILE_STEP` -- same diagnostic-only,
                // `instrument`-gated, default-off convention as
                // `run_decode_loop`'s own branch above (that one's doc has
                // the full rationale): unset in every production run, so
                // `evaluate_with_placements` is the only path a caller
                // without this env var ever takes on the default decode
                // path too. When set to this step's own index, this ONE
                // step instead runs `evaluate_op_timed_with_placements`
                // (per-op command buffers against the SAME placed-KV
                // shape) and prints the per-op GPU attribution through the
                // identical `report_op_timings` the two-range path already
                // uses.
                #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                let evaluated = match std::env::var("PROXIMA_METAL_OP_PROFILE_STEP")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                {
                    Some(target) if target == _step => {
                        let (evaluated, timings) = runtime.evaluate_op_timed_with_placements(
                            &single_range.program,
                            &symbols,
                            &named_blocks,
                            &roots,
                            &resident_names,
                            &input_placements,
                            &output_placements,
                        )?;
                        report_op_timings(_step, &timings);
                        evaluated
                    }
                    // `PROXIMA_METAL_DISPATCH_PROFILE_STEP` -- this branch's
                    // own per-dispatch-in-the-batched-buffer twin: same
                    // default-off, `instrument`-gated, one-env-var-per-step
                    // convention as `PROXIMA_METAL_OP_PROFILE_STEP` above,
                    // reusing `report_op_timings` unchanged since
                    // `evaluate_dispatch_timed_with_placements` returns the
                    // same `Vec<OpGpuTiming>` shape.
                    _ if std::env::var("PROXIMA_METAL_DISPATCH_PROFILE_STEP")
                        .ok()
                        .and_then(|value| value.parse::<usize>().ok())
                        == Some(_step) =>
                    {
                        let (evaluated, timings, sampling_mode, split_ns) = runtime
                            .evaluate_dispatch_timed_with_placements(
                                &single_range.program,
                                &symbols,
                                &named_blocks,
                                &roots,
                                &resident_names,
                                &input_placements,
                                &output_placements,
                            )?;
                        info!(
                            step = _step as u64,
                            sampling_mode,
                            "dispatch_profile: per-dispatch gpu-timestamp sampling mode"
                        );
                        report_op_timings(_step, &timings);
                        encoder_split_ns = split_ns;
                        evaluated
                    }
                    _ => runtime.evaluate_with_placements(
                        &single_range.program,
                        &symbols,
                        &named_blocks,
                        &roots,
                        &resident_names,
                        &input_placements,
                        &output_placements,
                    )?,
                };
                #[cfg(not(all(feature = "instrument", feature = "metal", target_os = "macos")))]
                let evaluated = runtime.evaluate_with_placements(
                    &single_range.program,
                    &symbols,
                    &named_blocks,
                    &roots,
                    &resident_names,
                    &input_placements,
                    &output_placements,
                )?;
                #[cfg(feature = "instrument")]
                let evaluate_ticks = elapsed_ticks(evaluate_started);
                // Snapshot-and-reset (`metal_stage_totals`'s own doc), so this
                // read must happen exactly once per step, immediately after
                // this step's own `evaluate_with_placements` call --
                // `block_offered_bytes` below is this step's REAL device
                // upload byte count: the three `kv_cache.{layer}.*` scratch
                // blocks pushed above never actually upload (they are
                // `input_placements` entries, which `execute_plan_with_placements`
                // binds directly to the caller's own device buffer instead of
                // staging through `upload_block` -- see that function's own
                // doc, "skipping the per-call `upload_block`/
                // `upload_packed_bytes` host round trip entirely"), so this
                // count legitimately falls to just the resident weights'
                // one-time cost after step 0, never a hardcoded zero.
                #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                let metal_stage = metal_stage_totals();
                #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                if let Some(split_ns) = encoder_split_ns {
                    report_encoder_split(
                        _step,
                        split_ns,
                        ticks_to_nanos(metal_stage.gpu_exec_ticks),
                    );
                }
                #[cfg(feature = "instrument")]
                let cached_len_before_step = cached_len;
                cached_len = merged_len;

                let (logits, _shape) = evaluated.get(single_range.logits_root).ok_or(
                    InteropError::MissingEvaluatedNode {
                        node: single_range.logits_root,
                    },
                )?;
                let last_position = &logits[(new_count - 1) * vocab_size..new_count * vocab_size];
                #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                let barriers_step = metal_stage.barriers_emitted;
                #[cfg(not(all(feature = "instrument", feature = "metal", target_os = "macos")))]
                let barriers_step = 0_u64;
                logits_sink.observe(last_position, barriers_step);

                #[cfg(feature = "instrument")]
                let greedy_pick_started = read_ticks();
                let token_id = match token_override.and_then(|forced| forced.get(_step)) {
                    Some(&forced_token) => forced_token,
                    None => {
                        let recent_window_start = token_history.len().saturating_sub(repeat_window);
                        let recent_tokens = &token_history[recent_window_start..];
                        sample_next_token(last_position, recent_tokens, sample_config, &mut rng)
                            .ok_or(InteropError::EmptyLogits)?
                    }
                };
                token_history.push(token_id);
                #[cfg(feature = "instrument")]
                let greedy_pick_ticks = elapsed_ticks(greedy_pick_started);
                next_ids = alloc::vec![token_id];

                #[cfg(feature = "instrument")]
                {
                    emit_token_breakdown(&TokenBreakdown {
                        step: _step,
                        new_count,
                        cached_len_before: cached_len_before_step,
                        step_wall_ticks: elapsed_ticks(step_started),
                        apply_serving_config_ticks,
                        build_position_inputs_ticks,
                        named_blocks_weights_ticks,
                        named_blocks_kv_ticks,
                        // No host-side KV cache exists on this arm (the cache
                        // lives entirely in `k_even_buffers`/`k_odd_buffers`/
                        // `v_buffers`, device-resident for the whole call), so
                        // there is no real host-upload byte count to report
                        // here. `metal_stage.block_offered_bytes` used to fill
                        // this field instead (ROW 369) -- that is the whole
                        // bound program's residency census, weights included,
                        // not a KV-cache-specific count, and it is already
                        // reported honestly under its own name by
                        // `emit_token_breakdown_metal`'s `block_offered_bytes`
                        // field below.
                        kv_cache_upload_bytes: 0,
                        evaluate_ticks,
                        // No separate host layer-cache append step on this
                        // arm -- `output_placements` above writes each
                        // layer's freshly rotated key/value straight into its
                        // `PlacedBuffer` as part of the SAME evaluate call
                        // `evaluate_ticks` already timed, so there is no
                        // second cost to attribute here.
                        layer_cache_append_ticks: 0,
                        layer_cache_append_bytes: 0,
                        greedy_pick_ticks,
                    });
                    #[cfg(all(feature = "metal", target_os = "macos"))]
                    emit_token_breakdown_metal(
                        _step,
                        &metal_stage,
                        runtime.placed_plans_len(),
                        runtime.plan_hits,
                        runtime.plan_misses,
                    );
                    #[cfg(all(feature = "metal", target_os = "macos"))]
                    if _step == 0 {
                        let kv_cache_device_bytes = (block_count
                            * (2 * capacity_even_odd + capacity_v))
                            as u64;
                        emit_device_memory_by_class(
                            _step,
                            metal_stage.block_nocopy_bound_bytes,
                            metal_stage.block_copied_bytes,
                            metal_stage.block_offset_bound_bytes,
                            Some(kv_cache_device_bytes),
                            omega::metal::current_allocated_size().unwrap_or(0),
                            phys_footprint_bytes(),
                        );
                    }
                }

                Ok(token_id)
            },
            on_token,
        )?;

        let text = proxima_tokenizer::decode(&generated_ids, &self.vocab)?;
        Ok((generated_ids, text, stopped_by_eos))
    }

    /// A one-shot forward pass over `prompt` (BOS forced, fresh KV state,
    /// same input-binding shape as `Self::run_decode_loop`'s own first
    /// step) that returns the raw values for each requested `NodeId`
    /// instead of sampling a token from the final logits.
    ///
    /// Exists as this crate's cross-oracle diagnostic surface: a decoded
    /// token is an argmax, and an argmax destroys exactly the information
    /// that tells a near-tied bf16-vs-f16 rounding gap apart from a gross
    /// defect. [`Self::forward_logits`] is the `node_ids == [logits_root]`
    /// convenience most callers want; this general form additionally lets a
    /// caller bisect a numeric divergence by depth: build
    /// [`proxima_tensor::spec::mistral_cached_forward_program_with_experts`]
    /// again at a shorter `block_count` against this same architecture and
    /// read off the last shared `NodeId` (`proxima_tensor::op::append`'s
    /// id-is-index invariant guarantees the two programs agree on every
    /// `NodeId` up to the point they diverge) -- that id is the residual
    /// stream's value right after that layer, comparable directly against
    /// an oracle's own per-layer tensor dump.
    /// `examples/smollm2_logit_oracle_diff.rs` is the worked tool.
    ///
    /// # Errors
    ///
    /// Whatever tokenizing `prompt` against this checkpoint's own
    /// [`Vocab`] or evaluating its forward program can fail with, plus
    /// [`InteropError::MissingEvaluatedNode`] if any `node_ids` entry was
    /// never computed by this checkpoint's own forward program (a caller
    /// passed a `NodeId` from a differently-shaped program).
    pub fn forward_node_values(
        &self,
        prompt: &str,
        node_ids: &[NodeId],
    ) -> Result<Vec<Vec<f32>>, InteropError> {
        self.forward_node_values_on_backend(prompt, node_ids, 0)
    }

    /// [`Self::forward_node_values`] with the backend left open --
    /// `gpu_layers` reaches [`BackendRuntime::new`]/[`select_backend`] the
    /// same way [`Self::generate_with_serving_config`]'s own
    /// `serving_config.gpu_layers` already does, so this one-shot forward
    /// can be pinned to CPU (`0`) or Metal ([`crate::serving::GPU_LAYERS_ALL`],
    /// `metal`-featured builds only) instead of always running CPU the way
    /// [`Self::forward_node_values`] does today. `crate::quality`'s
    /// reference-vs-variant harness is the reason this exists: comparing
    /// Metal's own decode path against the CPU reference needs the SAME
    /// one-shot forward run on each backend in turn, not two different
    /// programs.
    ///
    /// # Errors
    ///
    /// Whatever [`Self::forward_node_values`] can fail with, plus
    /// [`InteropError::UnsupportedServingConfig`] if `gpu_layers` requests a
    /// backend [`apply_serving_config`] does not accept (`0` and
    /// [`crate::serving::GPU_LAYERS_ALL`] on a `metal`-featured build are the
    /// only two).
    pub fn forward_node_values_on_backend(
        &self,
        prompt: &str,
        node_ids: &[NodeId],
        gpu_layers: i32,
    ) -> Result<Vec<Vec<f32>>, InteropError> {
        let serving_config = supported_serving_config(
            gpu_layers,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            omega::MathMode::default(),
        );
        let mut runtime = BackendRuntime::new(&serving_config);

        let ids = proxima_tokenizer::encode_with_bos_eos(
            prompt,
            &self.vocab,
            wants_bos(&self.vocab),
            self.vocab.add_eos_token().unwrap_or(false),
        )?;
        apply_serving_config(&serving_config, ids.len())?;
        let inputs = build_position_inputs(
            &ids,
            0,
            self.architecture.head_dim,
            self.architecture.rope_freq_base,
            self.architecture.rms_epsilon,
        );

        let block_count = self.architecture.block_count as usize;
        let empty_cache = LayerCache::new();
        let kv_cache_names: Vec<(String, String, String)> = (0..block_count)
            .map(|layer| {
                (
                    alloc::format!("kv_cache.{layer}.k_even"),
                    alloc::format!("kv_cache.{layer}.k_odd"),
                    alloc::format!("kv_cache.{layer}.v"),
                )
            })
            .collect();

        let mut named_blocks: Vec<(&str, QuantizedBlock)> = Vec::with_capacity(
            self.weights.owned.len()
                + self.weights.packed.len()
                + self.weights.packed_owned.len()
                + 3
                + block_count * 3,
        );
        named_blocks.push(("ids", QuantizedBlock::Float32(inputs.ids_f32.as_slice())));
        for (name, data) in &self.weights.owned {
            named_blocks.push((name.as_str(), QuantizedBlock::Float32(data.as_slice())));
        }
        for (name, block) in &self.weights.packed {
            named_blocks.push((name.as_str(), *block));
        }
        for (name, bytes, kind) in &self.weights.packed_owned {
            named_blocks.push((name.as_str(), kind.as_block(bytes)));
        }
        named_blocks.push(("eps", QuantizedBlock::Float32(inputs.epsilon.as_slice())));
        named_blocks.push(("rope_cos", QuantizedBlock::Float32(inputs.cos.as_slice())));
        named_blocks.push(("rope_sin", QuantizedBlock::Float32(inputs.sin.as_slice())));
        for (k_even_name, k_odd_name, v_name) in &kv_cache_names {
            named_blocks.extend(empty_cache.named_blocks(k_even_name, k_odd_name, v_name));
        }

        let resident_names: BTreeSet<&str> = self.resident_names();

        let symbols = [ids.len() as u64, 0u64];
        let evaluated = runtime.evaluate(
            &self.program,
            &symbols,
            &named_blocks,
            node_ids,
            &resident_names,
        )?;

        node_ids
            .iter()
            .map(|node| {
                evaluated
                    .get(*node)
                    .map(|(data, _shape)| data.to_vec())
                    .ok_or(InteropError::MissingEvaluatedNode { node: *node })
            })
            .collect()
    }

    /// [`Self::forward_node_values`] against `[Self::logits_root]`, sliced
    /// to just the LAST prompt position -- the convenience a caller
    /// cross-checking a decoded token's own logits (not an intermediate
    /// layer) wants. See that method's own doc for why a raw logit vector,
    /// not a sampled token, is what this crate's cross-oracle diagnostics
    /// need.
    ///
    /// # Errors
    ///
    /// Whatever [`Self::forward_node_values`] can fail with.
    pub fn forward_logits(&self, prompt: &str) -> Result<Vec<f32>, InteropError> {
        self.forward_logits_on_backend(prompt, 0)
    }

    /// [`Self::forward_logits`] with the backend left open, same
    /// [`Self::forward_node_values_on_backend`]-vs-[`Self::forward_node_values`]
    /// relationship: [`Self::forward_logits`] always pins `gpu_layers: 0`
    /// (CPU), this lets a caller ask for the Metal backend's own one-shot
    /// logits instead.
    ///
    /// # Errors
    ///
    /// Whatever [`Self::forward_node_values_on_backend`] can fail with.
    pub fn forward_logits_on_backend(
        &self,
        prompt: &str,
        gpu_layers: i32,
    ) -> Result<Vec<f32>, InteropError> {
        let ids = proxima_tokenizer::encode_with_bos_eos(
            prompt,
            &self.vocab,
            wants_bos(&self.vocab),
            self.vocab.add_eos_token().unwrap_or(false),
        )?;
        let mut values =
            self.forward_node_values_on_backend(prompt, &[self.logits_root], gpu_layers)?;
        let logits = values.remove(0);
        let vocab_size = self.architecture.vocab as usize;
        let new_count = ids.len();
        let last_position = logits[(new_count - 1) * vocab_size..new_count * vocab_size].to_vec();

        #[cfg(feature = "instrument")]
        {
            let mut ranked: Vec<usize> = (0..last_position.len()).collect();
            ranked.sort_by(|left, right| {
                last_position[*right]
                    .total_cmp(&last_position[*left])
                    .then_with(|| left.cmp(right))
            });
            let top1_token = ranked[0] as u64;
            let top1_logit = f64::from(last_position[ranked[0]]);
            debug!(
                prompt_tokens = ids.len() as u64,
                top1_token,
                top1_logit,
                "computed one-shot forward logits for cross-oracle comparison"
            );
        }

        Ok(last_position)
    }
}

#[cfg(all(test, feature = "std"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::string::String;
    use alloc::vec::Vec;

    use proxima_gguf::value::MetadataValue as Value;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use proxima_gguf::value::MetadataArray;
    use proxima_gguf::{GgmlType as WireType, GgufModel, TensorPayload, write_complete};
    use proxima_tokenizer::Vocab;

    use super::{Control, Phase, TokenEvent, build_position_inputs, decode_until_stop_or_budget};
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use super::{BackendRuntime, LoadedModel};
    use crate::bind::architecture_from_metadata;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use crate::serving::{GPU_LAYERS_ALL, ServingConfig};

    fn dims(values: &[u64]) -> arrayvec::ArrayVec<u64, { proxima_gguf::tensor::MAX_DIMS }> {
        values.iter().copied().collect()
    }

    /// A checkpoint that declares `llama.attention.layer_norm_rms_epsilon`
    /// (Qwen3's own value, `1e-6`, chosen because it differs from
    /// `crate::bind`'s own `RMS_EPSILON_DEFAULT` (`1e-5`) -- a test using
    /// the default would pass even if the metadata read were wired to
    /// nothing) must have that value flow all the way from
    /// [`architecture_from_metadata`] through [`build_position_inputs`]'s
    /// `epsilon` output, the exact vector `run_decode_loop` feeds every
    /// layer norm on the Metal/CPU decode path.
    #[test]
    fn checkpoint_declared_rms_epsilon_flows_into_position_inputs() {
        let embed_bytes = vec![0u8; 4 * 3 * 4]; // [embedding=4, vocab=3] f32
        let model = GgufModel {
            version: 3,
            metadata: vec![
                (
                    "general.architecture".to_string(),
                    Value::String("llama".to_string()),
                ),
                ("llama.embedding_length".to_string(), Value::U32(4)),
                ("llama.feed_forward_length".to_string(), Value::U32(8)),
                ("llama.attention.head_count".to_string(), Value::U32(2)),
                ("llama.attention.head_count_kv".to_string(), Value::U32(1)),
                ("llama.block_count".to_string(), Value::U32(1)),
                ("llama.rope.dimension_count".to_string(), Value::U32(2)),
                (
                    "llama.attention.layer_norm_rms_epsilon".to_string(),
                    Value::F32(1e-6),
                ),
            ],
            tensors: vec![TensorPayload {
                name: "token_embd.weight".to_string(),
                dims: dims(&[4, 3]),
                ggml_type: WireType::F32,
                data: &embed_bytes,
            }],
        };
        let file_bytes = write_complete(&model).expect("writes gguf with rms_epsilon metadata");
        let parsed = proxima_gguf::parse_complete(&file_bytes)
            .expect("parses gguf with rms_epsilon metadata");
        let architecture = architecture_from_metadata(&parsed)
            .expect("derive architecture from real metadata keys");

        assert_eq!(
            architecture.rms_epsilon, 1e-6,
            "architecture_from_metadata must read the checkpoint's own \
             layer_norm_rms_epsilon key, not a hard-coded default"
        );

        let inputs = build_position_inputs(
            &[7, 9],
            0,
            architecture.head_dim,
            architecture.rope_freq_base,
            architecture.rms_epsilon,
        );

        assert_eq!(
            inputs.epsilon,
            alloc::vec![1e-6, 1e-6],
            "build_position_inputs must feed the checkpoint's own epsilon into every \
             position, not RMS_EPSILON_DEFAULT"
        );
    }

    /// `int8-logs`, main `9a8b623c`: [`LoadedModel::forward_node_values_on_backend`]
    /// reported `NotLowerable { reason: "operand buffer missing at
    /// evaluation time" }` for a directly-requested `NodeId` once the
    /// requested `node_ids` window widened past roughly one layer's own
    /// node count, bisected in production between 35 and 40 requested
    /// nodes. Reproduces against the real host-local checkpoint (the exact
    /// quantized-weight, `cohort-staged-graph` CPU path production uses --
    /// a synthetic float32-only program does not engage the same matmul
    /// batching this needs): every `NodeId` in a wide window must come back
    /// with the identical value it has when requested alone.
    mod real_openchat_file {
        use core::ffi::c_void;
        use std::os::fd::AsFd;

        use proxima_tensor::op::NodeId;

        use super::super::LoadedModel;

        struct MappedGguf {
            base: *mut u8,
            len: usize,
            _file: std::fs::File,
        }

        impl MappedGguf {
            fn open(path: &std::path::Path) -> std::io::Result<Self> {
                let file = std::fs::File::open(path)?;
                let len = usize::try_from(file.metadata()?.len())
                    .expect("fixture file length fits in usize");
                // SAFETY: `len` matches the just-opened file's own length;
                // `file` is kept alive in `_file` for as long as `base` is
                // used, and the mapping is read-only/private so no writer
                // can observe or race it.
                let base = unsafe {
                    rustix::mm::mmap(
                        core::ptr::null_mut(),
                        len,
                        rustix::mm::ProtFlags::READ,
                        rustix::mm::MapFlags::PRIVATE,
                        file.as_fd(),
                        0,
                    )
                }
                .expect("mmap host-local openchat gguf fixture")
                .cast::<u8>();
                Ok(Self {
                    base,
                    len,
                    _file: file,
                })
            }

            fn as_slice(&self) -> &[u8] {
                // SAFETY: `base` points at `len` bytes mapped for `self`'s
                // whole lifetime; this borrows `self` immutably, so nothing
                // can unmap the region while the returned slice is alive.
                unsafe { core::slice::from_raw_parts(self.base, self.len) }
            }
        }

        impl Drop for MappedGguf {
            fn drop(&mut self) {
                // SAFETY: `base`/`len` are exactly what `open`'s `mmap`
                // call returned; nothing else unmaps this region.
                let _ = unsafe { rustix::mm::munmap(self.base.cast::<c_void>(), self.len) };
            }
        }

        #[test]
        #[ignore = "depends on a host-local openchat gguf checkout outside this repo"]
        fn forward_node_values_keeps_every_requested_node_live_across_a_layer_boundary() {
            let model_path = crate::test_support::openchat_gguf_path();
            crate::test_support::require_fixture(&model_path, Some("PROXIMA_OPENCHAT_GGUF"));
            let path = std::path::Path::new(&model_path);

            let mapped = MappedGguf::open(path).expect("mmap host-local openchat gguf fixture");
            let file_bytes = mapped.as_slice();
            let parsed = proxima_gguf::pipe::parse_complete(file_bytes)
                .expect("parse host-local openchat gguf fixture");
            let model = LoadedModel::load(&parsed, file_bytes)
                .expect("load real openchat checkpoint through the public path");

            let window: Vec<NodeId> = (0u32..40).map(NodeId).collect();
            let prompt = "The quick brown fox";

            let batch = model
                .forward_node_values(prompt, &window)
                .expect("every requested node across the layer boundary must evaluate");

            assert_eq!(
                batch.len(),
                window.len(),
                "one value must come back per requested node"
            );
            // A KV-cache `Op::Input` (`kv_cache.0.k_even` etc.) is legitimately
            // zero-length here -- this is a one-shot, fresh-KV-state forward
            // pass (`forward_node_values`'s own doc), so "empty" is a correct
            // answer for that node class, not evidence of eviction. NodeId(34)
            // is the exact node this defect's own bisection named: the
            // `activation * quantized-weight` multiply `is_quantized_matmul_operand`
            // ordinarily fuses into its reduce, forced standalone here only
            // because this window's own liveness protection keeps it alive --
            // its value coming back non-empty is the falsifiable claim this
            // test exists to prove.
            let quantized_matmul_multiply = NodeId(34);
            let position = window
                .iter()
                .position(|node| *node == quantized_matmul_multiply)
                .expect("the chosen window must include the node this defect was bisected to");
            assert!(
                !batch[position].is_empty(),
                "{quantized_matmul_multiply:?} came back with zero values -- evicted or never \
                 materialized despite being directly requested"
            );
        }
    }

    /// A minimal valid [`Vocab`] (every byte-level BPE vocab needs all 256
    /// base-byte tokens present or [`Vocab::new`] rejects it) plus one
    /// extra token at id `256` marked as this vocab's end-of-sequence id --
    /// enough to exercise [`decode_until_stop_or_budget`]'s stopping policy
    /// without a real checkpoint. Spells the base-byte alphabet as the
    /// SentencePiece `"<0xXX>"` fallback form directly (not through
    /// `proxima_tokenizer`'s private `byte_to_char`) since that spelling is
    /// public knowledge, not an internal detail this test needs to reach
    /// into the crate for.
    fn vocab_with_eos(eos_id: u32) -> Vocab {
        let mut tokens: Vec<String> = (0..=255u8)
            .map(|byte| alloc::format!("<0x{byte:02X}>"))
            .collect();
        tokens.push(String::from("<eos-marker>"));
        Vocab::new(tokens, &[], Some(0), Some(eos_id), None).expect("minimal vocab builds")
    }

    /// The defect this module exists to fix, proved directly: a scripted
    /// token source that would emit `999` on a 4th call never gets asked
    /// for it, because the 3rd call's token (`32000`, this vocab's eos id)
    /// stops the loop first. Also proves the eos id itself never lands in
    /// `generated_ids`.
    #[test]
    fn stops_early_when_eos_is_produced_and_excludes_it_from_ids() {
        let vocab = vocab_with_eos(32_000);
        let scripted_tokens = [10u32, 20, 32_000, 999];
        let mut calls = 0usize;

        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(
            &vocab,
            4,
            0,
            |step| {
                calls += 1;
                Ok(scripted_tokens[step])
            },
            &mut |_event| Control::Continue,
        )
        .expect("scripted token source never errors");

        assert_eq!(
            generated_ids,
            alloc::vec![10, 20],
            "eos id must not be appended to the generated ids"
        );
        assert!(
            stopped_by_eos,
            "must report that the stop was the model's own eos signal"
        );
        assert_eq!(
            calls, 3,
            "must not pull a 4th token once eos is seen on the 3rd"
        );
    }

    /// The other half of the invariant: when the model never emits eos,
    /// decoding runs the full budget and reports that distinctly from an
    /// eos stop -- `stopped_by_eos == false` is the caller's only way to
    /// tell "ran out of budget" apart from "the model finished".
    #[test]
    fn exhausts_the_budget_and_reports_it_distinctly_from_an_eos_stop() {
        let vocab = vocab_with_eos(32_000);
        let scripted_tokens = [10u32, 20, 30, 40];

        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(
            &vocab,
            scripted_tokens.len(),
            0,
            |step| Ok(scripted_tokens[step]),
            &mut |_event| Control::Continue,
        )
        .expect("scripted token source never errors");

        assert_eq!(
            generated_ids,
            alloc::vec![10, 20, 30, 40],
            "every scripted token is a real id, none is eos"
        );
        assert!(
            !stopped_by_eos,
            "budget exhaustion must not be reported as an eos stop"
        );
        assert_eq!(
            generated_ids.len(),
            scripted_tokens.len(),
            "budget exhaustion still runs every requested step"
        );
    }

    /// ROW 392's own class fix, proved directly against the real two-range
    /// path (not a hand-rolled stand-in): a synthetic one-layer
    /// mixture-of-experts checkpoint (`architecture.expert_count > 0` forces
    /// [`LoadedModel::single_range`] to `None`, `build_single_range_program`'s
    /// own doc, so `gpu_layers: GPU_LAYERS_ALL` here reaches
    /// [`BackendRuntime::evaluate`] through [`LoadedModel::run_decode_loop`],
    /// never `run_decode_loop_placed_kv`) drives the SAME 8-step greedy
    /// decode twice, once with `kv_bucket_tokens: 1` (today's pre-fix
    /// behavior: `kv_extent`'s own doc, `div_ceil(1)` is the identity) and
    /// once with `kv_bucket_tokens: 32` (`ServingConfig::default`'s own
    /// value). Two claims, both comparative rather than a single hard-coded
    /// constant, so neither depends on `omega`'s own internal per-node
    /// allocation count: bucketing must produce STRICTLY fewer plan misses
    /// and STRICTLY fewer [`omega::metal::OUTPUT_BUFFER_ALLOCATIONS`] than
    /// the unbucketed run (ROW 392's own finding: one miss, and one fresh
    /// `Plan` with its own device output buffers, per token before this
    /// fix), and the two runs must land on the IDENTICAL generated token
    /// ids -- `proxima_tensor::bind::cached_attention_candidates`'s own doc
    /// on the fused op's runtime bound is the numerics claim this equality
    /// is standing in for: a bucket's padding is invisible to softmax, so
    /// rounding `cached_len` up must never change what the model emits.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn two_range_plan_cache_buckets_cached_len_without_changing_generated_tokens() {
        fn f32_bytes(values: &[f32]) -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        }

        let vocab_size = 257u64;
        let embedding = 2u64;
        let feed_forward = 2u64;
        let expert_count = 2u64;

        let mut tokens: Vec<String> = (0..=255u8)
            .map(|byte| alloc::format!("<0x{byte:02X}>"))
            .collect();
        tokens.push(String::from("<eos-marker>"));

        let token_embd = f32_bytes(&vec![0.05f32; (vocab_size * embedding) as usize]);
        let norm_weight = f32_bytes(&vec![1.0f32; embedding as usize]);
        let square = f32_bytes(&vec![0.05f32; (embedding * embedding) as usize]);
        let gate_inp = f32_bytes(&vec![0.05f32; (embedding * expert_count) as usize]);
        let expert_stack = f32_bytes(&vec![0.05f32; (expert_count * feed_forward * embedding) as usize]);
        let output_weight = f32_bytes(&vec![0.05f32; (vocab_size * embedding) as usize]);

        let model = GgufModel {
            version: 3,
            metadata: vec![
                (
                    "general.architecture".to_string(),
                    Value::String("llama".to_string()),
                ),
                (
                    "llama.embedding_length".to_string(),
                    Value::U32(embedding as u32),
                ),
                (
                    "llama.feed_forward_length".to_string(),
                    Value::U32(feed_forward as u32),
                ),
                (
                    "llama.attention.head_count".to_string(),
                    Value::U32(1),
                ),
                (
                    "llama.attention.head_count_kv".to_string(),
                    Value::U32(1),
                ),
                ("llama.block_count".to_string(), Value::U32(1)),
                (
                    "llama.expert_count".to_string(),
                    Value::U32(expert_count as u32),
                ),
                (
                    "llama.expert_used_count".to_string(),
                    Value::U32(1),
                ),
                (
                    "tokenizer.ggml.model".to_string(),
                    Value::String("gpt2".to_string()),
                ),
                (
                    "tokenizer.ggml.tokens".to_string(),
                    Value::Array(MetadataArray::String(tokens)),
                ),
                (
                    "tokenizer.ggml.merges".to_string(),
                    Value::Array(MetadataArray::String(Vec::new())),
                ),
            ],
            tensors: vec![
                TensorPayload {
                    name: "token_embd.weight".to_string(),
                    dims: dims(&[embedding, vocab_size]),
                    ggml_type: WireType::F32,
                    data: &token_embd,
                },
                TensorPayload {
                    name: "blk.0.attn_norm.weight".to_string(),
                    dims: dims(&[embedding]),
                    ggml_type: WireType::F32,
                    data: &norm_weight,
                },
                TensorPayload {
                    name: "blk.0.ffn_norm.weight".to_string(),
                    dims: dims(&[embedding]),
                    ggml_type: WireType::F32,
                    data: &norm_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_q.weight".to_string(),
                    dims: dims(&[embedding, embedding]),
                    ggml_type: WireType::F32,
                    data: &square,
                },
                TensorPayload {
                    name: "blk.0.attn_k.weight".to_string(),
                    dims: dims(&[embedding, embedding]),
                    ggml_type: WireType::F32,
                    data: &square,
                },
                TensorPayload {
                    name: "blk.0.attn_v.weight".to_string(),
                    dims: dims(&[embedding, embedding]),
                    ggml_type: WireType::F32,
                    data: &square,
                },
                TensorPayload {
                    name: "blk.0.attn_output.weight".to_string(),
                    dims: dims(&[embedding, embedding]),
                    ggml_type: WireType::F32,
                    data: &square,
                },
                TensorPayload {
                    name: "blk.0.ffn_gate_inp.weight".to_string(),
                    dims: dims(&[embedding, expert_count]),
                    ggml_type: WireType::F32,
                    data: &gate_inp,
                },
                TensorPayload {
                    name: "blk.0.ffn_gate_exps.weight".to_string(),
                    dims: dims(&[embedding, feed_forward, expert_count]),
                    ggml_type: WireType::F32,
                    data: &expert_stack,
                },
                TensorPayload {
                    name: "blk.0.ffn_up_exps.weight".to_string(),
                    dims: dims(&[embedding, feed_forward, expert_count]),
                    ggml_type: WireType::F32,
                    data: &expert_stack,
                },
                TensorPayload {
                    name: "blk.0.ffn_down_exps.weight".to_string(),
                    dims: dims(&[feed_forward, embedding, expert_count]),
                    ggml_type: WireType::F32,
                    data: &expert_stack,
                },
                TensorPayload {
                    name: "output_norm.weight".to_string(),
                    dims: dims(&[embedding]),
                    ggml_type: WireType::F32,
                    data: &norm_weight,
                },
                TensorPayload {
                    name: "output.weight".to_string(),
                    dims: dims(&[embedding, vocab_size]),
                    ggml_type: WireType::F32,
                    data: &output_weight,
                },
            ],
        };

        let file_bytes =
            write_complete(&model).expect("writes a minimal one-layer MoE gguf fixture");
        let parsed = proxima_gguf::pipe::parse_complete(&file_bytes)
            .expect("parses the minimal one-layer MoE gguf fixture");
        let loaded = LoadedModel::load(&parsed, &file_bytes)
            .expect("loads the minimal one-layer MoE checkpoint through the public path");

        let base_config = ServingConfig {
            kv_cache_key_quant: WireType::F32,
            kv_cache_value_quant: WireType::F32,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: GPU_LAYERS_ALL,
            reasoning_budget: 0,
            ..ServingConfig::default()
        };
        let max_tokens = 8usize;

        let unbucketed_config = ServingConfig {
            kv_bucket_tokens: 1,
            ..base_config
        };
        let mut unbucketed_runtime = BackendRuntime::new(&unbucketed_config);
        let _ = omega::metal::OUTPUT_BUFFER_ALLOCATIONS.snapshot_and_reset();
        let unbucketed = loaded
            .run_decode_loop("A", max_tokens, &unbucketed_config, &mut unbucketed_runtime)
            .expect("runs the unbucketed two-range MoE decode loop on the metal backend");
        let unbucketed_allocations = omega::metal::OUTPUT_BUFFER_ALLOCATIONS.snapshot_and_reset();

        let bucketed_config = ServingConfig {
            kv_bucket_tokens: 32,
            ..base_config
        };
        let mut bucketed_runtime = BackendRuntime::new(&bucketed_config);
        let _ = omega::metal::OUTPUT_BUFFER_ALLOCATIONS.snapshot_and_reset();
        let bucketed = loaded
            .run_decode_loop("A", max_tokens, &bucketed_config, &mut bucketed_runtime)
            .expect("runs the bucketed two-range MoE decode loop on the metal backend");
        let bucketed_allocations = omega::metal::OUTPUT_BUFFER_ALLOCATIONS.snapshot_and_reset();

        assert_eq!(
            unbucketed.0.len(),
            max_tokens,
            "no eos id was declared, so both runs must exhaust the full token budget"
        );
        assert_eq!(
            unbucketed_runtime.plan_misses, max_tokens,
            "kv_bucket_tokens=1 reproduces the pre-fix shape: cached_len is strictly \
             increasing, so every step is a fresh miss"
        );
        assert_eq!(
            unbucketed_runtime.plan_hits, 0,
            "an unbucketed extent never repeats within one decode call"
        );

        assert!(
            bucketed_runtime.plan_hits > 0,
            "a bucketed extent that never hits is the null result, not a pass"
        );
        assert!(
            bucketed_runtime.plan_misses < unbucketed_runtime.plan_misses,
            "kv_bucket_tokens=32 must reduce plan_misses below the unbucketed baseline, \
             or bucketing bought nothing on this fixture"
        );
        // Non-strict: measured `0` on both arms in this sandbox (no real
        // Metal device attached, `omega`'s own Gpu-arm buffer allocation
        // never fires at all rather than firing once per miss) -- the
        // plan_hits/plan_misses assertions above are this test's load-
        // bearing, environment-independent proof of ROW 392's fix; this one
        // only guards against a REGRESSION (bucketing must never allocate
        // MORE than the unbucketed baseline) on whatever device runs it.
        assert!(
            bucketed_allocations <= unbucketed_allocations,
            "bucketed_allocations={bucketed_allocations} unbucketed_allocations={unbucketed_allocations}: \
             bucketing must never allocate MORE device output buffers than the unbucketed baseline"
        );
        assert_eq!(
            bucketed.0, unbucketed.0,
            "the fused CachedAttention op's runtime bound must make a bucket's own \
             zero-padding invisible to softmax -- rounding cached_len up must never \
             change which token is emitted"
        );
    }

    /// [`two_range_plan_cache_buckets_cached_len_without_changing_generated_tokens`]'s
    /// counterpart for a qwen35 [`Qwen35LayerRoots::DenseAttention`] layer --
    /// this crate's fake-fixture fallback for that same claim, not the full
    /// 4-layer hybrid checkpoint `feat/synthetic-qwen38-fixture`'s own
    /// `examples/synth_qwen35_gguf.rs` builds (~918 MiB, out of this slice's
    /// time budget): one synthetic, `full_attention_interval: 1` layer (so
    /// every layer is [`crate::qwen35::Qwen35LayerKind::Attention`], no
    /// state-space mixer to also fixture), just wide enough
    /// (`query_heads = kv_heads = 1`, `attention.key_length = 4`,
    /// `rope.dimension_count = 2`, so `pass_dim = 2` is exercised alongside
    /// the rotated halves) to drive `qwen35_forward_program`'s
    /// `DenseAttention` cache path through
    /// [`Qwen35DenseAttentionPadScratch`] the same way the MoE test above
    /// drives `mistral_cached_forward_program_with_experts`'s `Attention`
    /// path through [`KvPadScratch`]. Asserts `plan_hits`/`plan_misses`
    /// only, per this card's own fake-fixture allowance -- no
    /// `OUTPUT_BUFFER_ALLOCATIONS`/CPU-vs-Metal comparison, since those need
    /// the real hybrid checkpoint's own numerics to be meaningful.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn qwen35_dense_attention_two_range_plan_cache_buckets_cached_len() {
        fn f32_bytes(values: &[f32]) -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        }

        let vocab_size = 257u64;
        let embedding = 4u64;
        let feed_forward = 4u64;
        let query_heads = 1u64;
        let kv_heads = 1u64;
        let rotary_dim = 2u64;
        let attn_head_dim = 4u64;

        let mut tokens: Vec<String> = (0..=255u8)
            .map(|byte| alloc::format!("<0x{byte:02X}>"))
            .collect();
        tokens.push(String::from("<eos-marker>"));

        let token_embd = f32_bytes(&vec![0.05f32; (embedding * vocab_size) as usize]);
        let norm_weight = f32_bytes(&vec![1.0f32; embedding as usize]);
        let head_norm_weight = f32_bytes(&vec![1.0f32; attn_head_dim as usize]);
        let q_weight = f32_bytes(&vec![0.05f32; (embedding * query_heads * attn_head_dim * 2) as usize]);
        let kv_weight = f32_bytes(&vec![0.05f32; (embedding * kv_heads * attn_head_dim) as usize]);
        let output_weight = f32_bytes(&vec![0.05f32; (query_heads * attn_head_dim * embedding) as usize]);
        let ffn_gate_up = f32_bytes(&vec![0.05f32; (embedding * feed_forward) as usize]);
        let ffn_down = f32_bytes(&vec![0.05f32; (feed_forward * embedding) as usize]);
        let output_table = f32_bytes(&vec![0.05f32; (embedding * vocab_size) as usize]);

        let model = GgufModel {
            version: 3,
            metadata: vec![
                (
                    "general.architecture".to_string(),
                    Value::String("qwen35".to_string()),
                ),
                (
                    "qwen35.embedding_length".to_string(),
                    Value::U32(embedding as u32),
                ),
                (
                    "qwen35.feed_forward_length".to_string(),
                    Value::U32(feed_forward as u32),
                ),
                (
                    "qwen35.attention.head_count".to_string(),
                    Value::U32(query_heads as u32),
                ),
                (
                    "qwen35.attention.head_count_kv".to_string(),
                    Value::U32(kv_heads as u32),
                ),
                ("qwen35.block_count".to_string(), Value::U32(1)),
                (
                    "qwen35.rope.dimension_count".to_string(),
                    Value::U32(rotary_dim as u32),
                ),
                (
                    "qwen35.attention.key_length".to_string(),
                    Value::U32(attn_head_dim as u32),
                ),
                (
                    "qwen35.full_attention_interval".to_string(),
                    Value::U32(1),
                ),
                ("qwen35.ssm.conv_kernel".to_string(), Value::U32(2)),
                ("qwen35.ssm.state_size".to_string(), Value::U32(1)),
                ("qwen35.ssm.group_count".to_string(), Value::U32(1)),
                ("qwen35.ssm.time_step_rank".to_string(), Value::U32(1)),
                ("qwen35.ssm.inner_size".to_string(), Value::U32(1)),
                (
                    "tokenizer.ggml.model".to_string(),
                    Value::String("gpt2".to_string()),
                ),
                (
                    "tokenizer.ggml.tokens".to_string(),
                    Value::Array(MetadataArray::String(tokens)),
                ),
                (
                    "tokenizer.ggml.merges".to_string(),
                    Value::Array(MetadataArray::String(Vec::new())),
                ),
            ],
            tensors: vec![
                TensorPayload {
                    name: "token_embd.weight".to_string(),
                    dims: dims(&[embedding, vocab_size]),
                    ggml_type: WireType::F32,
                    data: &token_embd,
                },
                TensorPayload {
                    name: "blk.0.attn_norm.weight".to_string(),
                    dims: dims(&[embedding]),
                    ggml_type: WireType::F32,
                    data: &norm_weight,
                },
                TensorPayload {
                    name: "blk.0.post_attention_norm.weight".to_string(),
                    dims: dims(&[embedding]),
                    ggml_type: WireType::F32,
                    data: &norm_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_q.weight".to_string(),
                    dims: dims(&[embedding, query_heads * attn_head_dim * 2]),
                    ggml_type: WireType::F32,
                    data: &q_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_k.weight".to_string(),
                    dims: dims(&[embedding, kv_heads * attn_head_dim]),
                    ggml_type: WireType::F32,
                    data: &kv_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_v.weight".to_string(),
                    dims: dims(&[embedding, kv_heads * attn_head_dim]),
                    ggml_type: WireType::F32,
                    data: &kv_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_output.weight".to_string(),
                    dims: dims(&[query_heads * attn_head_dim, embedding]),
                    ggml_type: WireType::F32,
                    data: &output_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_q_norm.weight".to_string(),
                    dims: dims(&[attn_head_dim]),
                    ggml_type: WireType::F32,
                    data: &head_norm_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_k_norm.weight".to_string(),
                    dims: dims(&[attn_head_dim]),
                    ggml_type: WireType::F32,
                    data: &head_norm_weight,
                },
                TensorPayload {
                    name: "blk.0.ffn_gate.weight".to_string(),
                    dims: dims(&[embedding, feed_forward]),
                    ggml_type: WireType::F32,
                    data: &ffn_gate_up,
                },
                TensorPayload {
                    name: "blk.0.ffn_up.weight".to_string(),
                    dims: dims(&[embedding, feed_forward]),
                    ggml_type: WireType::F32,
                    data: &ffn_gate_up,
                },
                TensorPayload {
                    name: "blk.0.ffn_down.weight".to_string(),
                    dims: dims(&[feed_forward, embedding]),
                    ggml_type: WireType::F32,
                    data: &ffn_down,
                },
                TensorPayload {
                    name: "output_norm.weight".to_string(),
                    dims: dims(&[embedding]),
                    ggml_type: WireType::F32,
                    data: &norm_weight,
                },
                TensorPayload {
                    name: "output.weight".to_string(),
                    dims: dims(&[embedding, vocab_size]),
                    ggml_type: WireType::F32,
                    data: &output_table,
                },
            ],
        };

        let file_bytes =
            write_complete(&model).expect("writes a minimal one-layer qwen35 gguf fixture");
        let parsed = proxima_gguf::pipe::parse_complete(&file_bytes)
            .expect("parses the minimal one-layer qwen35 gguf fixture");
        let loaded = LoadedModel::load(&parsed, &file_bytes)
            .expect("loads the minimal one-layer qwen35 checkpoint through the public path");

        let base_config = ServingConfig {
            kv_cache_key_quant: WireType::F32,
            kv_cache_value_quant: WireType::F32,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: GPU_LAYERS_ALL,
            reasoning_budget: 0,
            ..ServingConfig::default()
        };
        let max_tokens = 8usize;

        let unbucketed_config = ServingConfig {
            kv_bucket_tokens: 1,
            ..base_config
        };
        let mut unbucketed_runtime = BackendRuntime::new(&unbucketed_config);
        let unbucketed = loaded
            .run_decode_loop("A", max_tokens, &unbucketed_config, &mut unbucketed_runtime)
            .expect("runs the unbucketed qwen35 dense-attention decode loop");

        let bucketed_config = ServingConfig {
            kv_bucket_tokens: 32,
            ..base_config
        };
        let mut bucketed_runtime = BackendRuntime::new(&bucketed_config);
        let bucketed = loaded
            .run_decode_loop("A", max_tokens, &bucketed_config, &mut bucketed_runtime)
            .expect("runs the bucketed qwen35 dense-attention decode loop");

        assert_eq!(
            unbucketed.0.len(),
            max_tokens,
            "no eos id was declared, so both runs must exhaust the full token budget"
        );
        assert_eq!(
            unbucketed_runtime.plan_misses, max_tokens,
            "kv_bucket_tokens=1 reproduces the pre-fix shape on the DenseAttention arm: \
             cached_len is strictly increasing, so every step is a fresh miss"
        );
        assert_eq!(
            unbucketed_runtime.plan_hits, 0,
            "an unbucketed extent never repeats within one decode call"
        );
        assert!(
            bucketed_runtime.plan_hits > 0,
            "a bucketed extent that never hits on the DenseAttention arm is the null \
             result, not a pass"
        );
        assert!(
            bucketed_runtime.plan_misses < unbucketed_runtime.plan_misses,
            "kv_bucket_tokens=32 must reduce plan_misses below the unbucketed baseline \
             on the DenseAttention arm, or bucketing bought nothing on this fixture"
        );
        assert_eq!(
            bucketed.0, unbucketed.0,
            "the DenseAttention arm's padded cache must make a bucket's own \
             zero-padding invisible to softmax -- rounding cached_len up must never \
             change which token is emitted"
        );
    }

    /// [`qwen35_dense_attention_two_range_plan_cache_buckets_cached_len`]'s
    /// own fixture, driven through the raw-logit payload behind that test's
    /// argmax-only token-id comparison: that comparison alone cannot rule
    /// out a real but non-argmax-flipping corruption from
    /// [`Qwen35DenseAttentionPadScratch`]'s zero-padding on this fixture's
    /// tiny, uniform (`0.05` everywhere) weights, where two distinct logit
    /// vectors can still share an argmax. Runs the SAME 8-step greedy decode
    /// twice with [`LoadedModel::run_decode_loop_observed`]'s own
    /// `LogitsSink::Collect` hook, once with `kv_bucket_tokens: 1`
    /// (`kv_extent`'s own doc: `div_ceil(1)` is the identity, so `cached_len`
    /// is never rounded up and this arm pads NOTHING) and once with
    /// `kv_bucket_tokens: 32` (`ServingConfig::default`'s own value, so the
    /// early steps round a `cached_len` as small as `1` up to `32`, the
    /// widest possible padding this fixture can exercise). Compares the
    /// FINAL step's last-row logits bit-for-bit rather than only the
    /// emitted token ids -- `proxima_tensor::bind::cached_attention_
    /// candidates`'s own doc on the fused op's runtime bound is the
    /// numerics claim this equality is standing in for on the
    /// `DenseAttention` arm, which has no such fusion and instead relies on
    /// [`append_qwen35_dense_attention_layer`]'s own `cached_len` mask.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn qwen35_dense_attention_padding_is_invisible_to_softmax() {
        fn f32_bytes(values: &[f32]) -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        }

        let vocab_size = 257u64;
        let embedding = 4u64;
        let feed_forward = 4u64;
        let query_heads = 1u64;
        let kv_heads = 1u64;
        let rotary_dim = 2u64;
        let attn_head_dim = 4u64;

        let mut tokens: Vec<String> = (0..=255u8)
            .map(|byte| alloc::format!("<0x{byte:02X}>"))
            .collect();
        tokens.push(String::from("<eos-marker>"));

        let token_embd = f32_bytes(&vec![0.05f32; (embedding * vocab_size) as usize]);
        let norm_weight = f32_bytes(&vec![1.0f32; embedding as usize]);
        let head_norm_weight = f32_bytes(&vec![1.0f32; attn_head_dim as usize]);
        let q_weight = f32_bytes(&vec![0.05f32; (embedding * query_heads * attn_head_dim * 2) as usize]);
        let kv_weight = f32_bytes(&vec![0.05f32; (embedding * kv_heads * attn_head_dim) as usize]);
        let output_weight = f32_bytes(&vec![0.05f32; (query_heads * attn_head_dim * embedding) as usize]);
        let ffn_gate_up = f32_bytes(&vec![0.05f32; (embedding * feed_forward) as usize]);
        let ffn_down = f32_bytes(&vec![0.05f32; (feed_forward * embedding) as usize]);
        let output_table = f32_bytes(&vec![0.05f32; (embedding * vocab_size) as usize]);

        let model = GgufModel {
            version: 3,
            metadata: vec![
                (
                    "general.architecture".to_string(),
                    Value::String("qwen35".to_string()),
                ),
                (
                    "qwen35.embedding_length".to_string(),
                    Value::U32(embedding as u32),
                ),
                (
                    "qwen35.feed_forward_length".to_string(),
                    Value::U32(feed_forward as u32),
                ),
                (
                    "qwen35.attention.head_count".to_string(),
                    Value::U32(query_heads as u32),
                ),
                (
                    "qwen35.attention.head_count_kv".to_string(),
                    Value::U32(kv_heads as u32),
                ),
                ("qwen35.block_count".to_string(), Value::U32(1)),
                (
                    "qwen35.rope.dimension_count".to_string(),
                    Value::U32(rotary_dim as u32),
                ),
                (
                    "qwen35.attention.key_length".to_string(),
                    Value::U32(attn_head_dim as u32),
                ),
                (
                    "qwen35.full_attention_interval".to_string(),
                    Value::U32(1),
                ),
                ("qwen35.ssm.conv_kernel".to_string(), Value::U32(2)),
                ("qwen35.ssm.state_size".to_string(), Value::U32(1)),
                ("qwen35.ssm.group_count".to_string(), Value::U32(1)),
                ("qwen35.ssm.time_step_rank".to_string(), Value::U32(1)),
                ("qwen35.ssm.inner_size".to_string(), Value::U32(1)),
                (
                    "tokenizer.ggml.model".to_string(),
                    Value::String("gpt2".to_string()),
                ),
                (
                    "tokenizer.ggml.tokens".to_string(),
                    Value::Array(MetadataArray::String(tokens)),
                ),
                (
                    "tokenizer.ggml.merges".to_string(),
                    Value::Array(MetadataArray::String(Vec::new())),
                ),
            ],
            tensors: vec![
                TensorPayload {
                    name: "token_embd.weight".to_string(),
                    dims: dims(&[embedding, vocab_size]),
                    ggml_type: WireType::F32,
                    data: &token_embd,
                },
                TensorPayload {
                    name: "blk.0.attn_norm.weight".to_string(),
                    dims: dims(&[embedding]),
                    ggml_type: WireType::F32,
                    data: &norm_weight,
                },
                TensorPayload {
                    name: "blk.0.post_attention_norm.weight".to_string(),
                    dims: dims(&[embedding]),
                    ggml_type: WireType::F32,
                    data: &norm_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_q.weight".to_string(),
                    dims: dims(&[embedding, query_heads * attn_head_dim * 2]),
                    ggml_type: WireType::F32,
                    data: &q_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_k.weight".to_string(),
                    dims: dims(&[embedding, kv_heads * attn_head_dim]),
                    ggml_type: WireType::F32,
                    data: &kv_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_v.weight".to_string(),
                    dims: dims(&[embedding, kv_heads * attn_head_dim]),
                    ggml_type: WireType::F32,
                    data: &kv_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_output.weight".to_string(),
                    dims: dims(&[query_heads * attn_head_dim, embedding]),
                    ggml_type: WireType::F32,
                    data: &output_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_q_norm.weight".to_string(),
                    dims: dims(&[attn_head_dim]),
                    ggml_type: WireType::F32,
                    data: &head_norm_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_k_norm.weight".to_string(),
                    dims: dims(&[attn_head_dim]),
                    ggml_type: WireType::F32,
                    data: &head_norm_weight,
                },
                TensorPayload {
                    name: "blk.0.ffn_gate.weight".to_string(),
                    dims: dims(&[embedding, feed_forward]),
                    ggml_type: WireType::F32,
                    data: &ffn_gate_up,
                },
                TensorPayload {
                    name: "blk.0.ffn_up.weight".to_string(),
                    dims: dims(&[embedding, feed_forward]),
                    ggml_type: WireType::F32,
                    data: &ffn_gate_up,
                },
                TensorPayload {
                    name: "blk.0.ffn_down.weight".to_string(),
                    dims: dims(&[feed_forward, embedding]),
                    ggml_type: WireType::F32,
                    data: &ffn_down,
                },
                TensorPayload {
                    name: "output_norm.weight".to_string(),
                    dims: dims(&[embedding]),
                    ggml_type: WireType::F32,
                    data: &norm_weight,
                },
                TensorPayload {
                    name: "output.weight".to_string(),
                    dims: dims(&[embedding, vocab_size]),
                    ggml_type: WireType::F32,
                    data: &output_table,
                },
            ],
        };

        let file_bytes =
            write_complete(&model).expect("writes a minimal one-layer qwen35 gguf fixture");
        let parsed = proxima_gguf::pipe::parse_complete(&file_bytes)
            .expect("parses the minimal one-layer qwen35 gguf fixture");
        let loaded = LoadedModel::load(&parsed, &file_bytes)
            .expect("loads the minimal one-layer qwen35 checkpoint through the public path");

        let base_config = ServingConfig {
            kv_cache_key_quant: WireType::F32,
            kv_cache_value_quant: WireType::F32,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: GPU_LAYERS_ALL,
            reasoning_budget: 0,
            ..ServingConfig::default()
        };
        let max_tokens = 8usize;

        let unpadded_config = ServingConfig {
            kv_bucket_tokens: 1,
            ..base_config
        };
        let mut unpadded_runtime = BackendRuntime::new(&unpadded_config);
        let mut unpadded_logits: Vec<Vec<f32>> = Vec::new();
        let unpadded = loaded
            .run_decode_loop_observed(
                "A",
                max_tokens,
                &unpadded_config,
                &mut unpadded_runtime,
                None,
                &mut super::LogitsSink::Collect(&mut unpadded_logits),
                &mut |_event| Control::Continue,
            )
            .expect("runs the unpadded (kv_bucket_tokens=1) qwen35 dense-attention decode loop");

        let padded_config = ServingConfig {
            kv_bucket_tokens: 32,
            ..base_config
        };
        let mut padded_runtime = BackendRuntime::new(&padded_config);
        let mut padded_logits: Vec<Vec<f32>> = Vec::new();
        let padded = loaded
            .run_decode_loop_observed(
                "A",
                max_tokens,
                &padded_config,
                &mut padded_runtime,
                None,
                &mut super::LogitsSink::Collect(&mut padded_logits),
                &mut |_event| Control::Continue,
            )
            .expect("runs the padded (kv_bucket_tokens=32) qwen35 dense-attention decode loop");

        assert_eq!(
            unpadded.0, padded.0,
            "the DenseAttention arm's padded cache must make a bucket's own \
             zero-padding invisible to softmax -- rounding cached_len up must never \
             change which token is emitted"
        );

        let unpadded_last = unpadded_logits
            .last()
            .expect("the unpadded decode loop must observe at least one logits row");
        let padded_last = padded_logits
            .last()
            .expect("the padded decode loop must observe at least one logits row");

        let max_abs_diff = unpadded_last
            .iter()
            .zip(padded_last.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);

        std::println!(
            "qwen35_dense_attention_padding max_abs_diff={max_abs_diff} \
             unpadded={unpadded_last:?} padded={padded_last:?}"
        );
        assert!(
            max_abs_diff == 0.0,
            "kv_bucket_tokens=32's own zero-padded cached rows must be invisible to the \
             DenseAttention arm's softmax, the same guarantee the Attention arm's fused \
             CachedAttention op already provides -- max_abs_diff={max_abs_diff} between \
             unpadded (bucket=1) and padded (bucket=32) last-row logits proves it is not"
        );
    }

    /// Degenerate control: if the eos comparison were broken (e.g. always
    /// `false`), this test's scripted eos-first source would run the full
    /// budget instead of stopping on step 1 -- confirming the two tests
    /// above are not passing by coincidence of never actually comparing
    /// against `vocab.eos_token_id()`.
    #[test]
    fn stops_on_the_very_first_token_when_it_is_eos() {
        let vocab = vocab_with_eos(32_000);
        let mut calls = 0usize;

        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(
            &vocab,
            10,
            0,
            |_step| {
                calls += 1;
                Ok(32_000)
            },
            &mut |_event| Control::Continue,
        )
        .expect("scripted token source never errors");

        assert!(
            generated_ids.is_empty(),
            "an immediate eos must produce zero generated ids"
        );
        assert!(stopped_by_eos);
        assert_eq!(
            calls, 1,
            "must stop after exactly one call, not run toward the budget of 10"
        );
    }

    /// [`TokenEvent`]'s own contract, proved end to end against a scripted
    /// source: exactly one [`Phase::Prefill`] event (carrying the prompt
    /// token count this call was given, at step `0`), then one
    /// [`Phase::Token`] event per generated token, in order -- concatenating
    /// every [`Phase::Token`] event's `text_piece` reproduces
    /// [`proxima_tokenizer::decode`]'s own output on the same ids, and
    /// every [`Phase::Token`] event's `token_id` is the matching entry of
    /// the returned `Vec<u32>`.
    #[test]
    fn streams_one_prefill_event_then_one_token_event_per_generated_token() {
        let vocab = vocab_with_eos(32_000);
        // 'H', 'i', '!' -- three tokens spelling one word this vocab's own
        // base-byte alphabet can decode without any multibyte splitting.
        let scripted_tokens = [b'H' as u32, b'i' as u32, b'!' as u32];
        let prompt_token_count = 5;

        let mut events: Vec<(Phase, u32, String, usize)> = Vec::new();
        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(
            &vocab,
            scripted_tokens.len(),
            prompt_token_count,
            |step| Ok(scripted_tokens[step]),
            &mut |event: TokenEvent<'_>| {
                events.push((
                    event.phase,
                    event.token_id,
                    String::from(event.text_piece),
                    event.step,
                ));
                Control::Continue
            },
        )
        .expect("scripted token source never errors");

        assert!(!stopped_by_eos, "the scripted source never emits eos");
        assert_eq!(generated_ids, alloc::vec![72, 105, 33]);

        let prefill_events: Vec<_> = events
            .iter()
            .filter(|(phase, ..)| matches!(phase, Phase::Prefill { .. }))
            .collect();
        assert_eq!(
            prefill_events.len(),
            1,
            "exactly one prefill event, regardless of how many tokens follow"
        );
        let (prefill_phase, _, _, prefill_step) = prefill_events[0];
        assert_eq!(
            *prefill_phase,
            Phase::Prefill {
                prompt_tokens: prompt_token_count
            },
            "prefill must carry this call's own prompt token count"
        );
        assert_eq!(*prefill_step, 0, "prefill only ever happens at step 0");

        let token_events: Vec<_> = events
            .iter()
            .filter(|(phase, ..)| matches!(phase, Phase::Token))
            .collect();
        assert_eq!(
            token_events.len(),
            scripted_tokens.len(),
            "one Token event per generated token, none skipped or doubled"
        );
        let token_ids: Vec<u32> = token_events.iter().map(|(_, id, ..)| *id).collect();
        assert_eq!(
            token_ids, generated_ids,
            "Token event ids must equal the returned ids, in order"
        );

        let streamed_text: String = token_events
            .iter()
            .map(|(_, _, piece, _)| piece.as_str())
            .collect();
        let expected_text = proxima_tokenizer::decode(&generated_ids, &vocab)
            .expect("scripted ids all resolve to real vocab bytes");
        assert_eq!(
            streamed_text, expected_text,
            "concatenated Token event text must equal a one-shot decode of the same ids"
        );
    }

    /// [`Control::Stop`]'s own contract: returning it from `on_token` ends
    /// decoding after that token, short of `max_tokens`, and is reported
    /// the same way running out of budget is -- never mistaken for the
    /// model's own eos.
    #[test]
    fn control_stop_ends_decoding_early_and_is_not_reported_as_eos() {
        let vocab = vocab_with_eos(32_000);
        let scripted_tokens = [b'H' as u32, b'i' as u32, b'!' as u32, b'?' as u32];
        let mut token_events_seen = 0usize;

        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(
            &vocab,
            scripted_tokens.len(),
            0,
            |step| Ok(scripted_tokens[step]),
            &mut |event: TokenEvent<'_>| {
                if matches!(event.phase, Phase::Token) {
                    token_events_seen += 1;
                    if token_events_seen == 3 {
                        return Control::Stop;
                    }
                }
                Control::Continue
            },
        )
        .expect("scripted token source never errors");

        assert_eq!(
            generated_ids,
            alloc::vec![72, 105, 33],
            "must stop right after the 3rd token, never pulling the 4th"
        );
        assert_eq!(token_events_seen, 3);
        assert!(
            !stopped_by_eos,
            "a caller-requested Stop must report finished=false, same as budget exhaustion"
        );
    }

    /// The regression this module exists to fix, at proxima 818f5e46: a
    /// long prompt with a small `max_tokens` budget can legitimately end
    /// mid multibyte character (a real Qwen3 checkpoint stopping mid
    /// emoji/CJK glyph is the exact production report). Byte `0xE4` is
    /// `vocab_with_eos`'s own `<0xE4>` byte-fallback token -- the first of
    /// three bytes ([`crate::generate`]'s own `hex_fallback_token` fixture
    /// doc) a real 3-byte UTF-8 codepoint like `中` starts with -- and this
    /// vocab never supplies the other two, so the budget runs out with an
    /// incomplete lead byte still pending. Neither
    /// [`decode_until_stop_or_budget`] nor [`proxima_tokenizer::decode`] on
    /// its returned ids may fail over that: the caller-visible outcome is
    /// the valid prefix plus one U+FFFD, matching `proxima_tokenizer::decode`'s
    /// own "flag, don't drop" contract for a one-shot decode of the same
    /// truncated ids.
    #[test]
    fn budget_ending_mid_multibyte_character_flags_instead_of_erroring() {
        let vocab = vocab_with_eos(32_000);
        let scripted_tokens = [b'H' as u32, b'i' as u32, 0xE4u32];

        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(
            &vocab,
            scripted_tokens.len(),
            0,
            |step| Ok(scripted_tokens[step]),
            &mut |_event| Control::Continue,
        )
        .expect("an incomplete trailing multibyte sequence must never error");

        assert_eq!(
            generated_ids,
            alloc::vec![72, 105, 0xE4],
            "the incomplete lead byte's own id is still a real generated id"
        );
        assert!(
            !stopped_by_eos,
            "the budget ran out; the model never emitted its own eos"
        );

        let text = proxima_tokenizer::decode(&generated_ids, &vocab)
            .expect("a one-shot decode of the same truncated ids must never error either");
        assert_eq!(
            text, "Hi\u{FFFD}",
            "the complete prefix stays intact and the unfinished tail becomes one U+FFFD"
        );
    }

    /// The other way an incomplete lead byte can resolve: not by the
    /// budget ending, but by the very next token NOT being a valid
    /// continuation byte (autoregressive sampling gives no guarantee that
    /// consecutive token ids retrace one contiguous encoder segmentation).
    /// [`decode_streamed_piece`]'s own doc: a stale pending tail that turns
    /// out unresolvable resolves to one U+FFFD and decoding resumes on
    /// whatever bytes follow, so decoding one bad run never fails the
    /// whole call.
    #[test]
    fn incomplete_lead_byte_followed_by_a_non_continuation_byte_flags_and_resumes() {
        let vocab = vocab_with_eos(32_000);
        // 0xE4 starts a 3-byte sequence; 'H'/'i' are plain ASCII and can
        // never be valid UTF-8 continuation bytes (those are 0x80-0xBF).
        let scripted_tokens = [0xE4u32, b'H' as u32, b'i' as u32];

        let mut events: Vec<String> = Vec::new();
        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(
            &vocab,
            scripted_tokens.len(),
            0,
            |step| Ok(scripted_tokens[step]),
            &mut |event: TokenEvent<'_>| {
                if matches!(event.phase, Phase::Token) {
                    events.push(String::from(event.text_piece));
                }
                Control::Continue
            },
        )
        .expect("a non-continuing follow-on token must never error");

        assert_eq!(generated_ids, alloc::vec![0xE4, 72, 105]);
        assert!(!stopped_by_eos);
        assert_eq!(
            events.join(""),
            "\u{FFFD}Hi",
            "the unresolvable lead byte flags as one U+FFFD, then decoding \
             resumes normally on the bytes that follow it"
        );
    }

    /// The one real check for `LoadedModel`'s `Drop` impl: loads a real
    /// checkpoint on Metal, runs a handful of decode steps (so the resident
    /// weight buffers and the checkpoint-mapping no-copy buffer are both
    /// actually populated, not just registered), drops it, and prints
    /// `MTLDevice::currentAllocatedSize` alongside the process's own
    /// `phys_footprint` before load / after generate / after drop -- the
    /// artifact this row's own INVARIANT ("dropping a `LoadedModel` releases
    /// every device allocation it caused") is checked against. `#[ignore]`d
    /// like every other host-local fixture in this crate
    /// ([`real_openchat_file`]'s own doc): this prints evidence for a human
    /// to read, it does not assert a byte-exact threshold, because the OS's
    /// own `phys_footprint` also reflects unrelated process state (allocator
    /// arenas, thread stacks) this test does not control.
    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
    mod release_on_drop_real_model {
        use core::ffi::c_void;
        use std::os::fd::AsFd;

        use super::super::LoadedModel;
        use crate::serving::GPU_LAYERS_ALL;

        struct MappedGguf {
            base: *mut u8,
            len: usize,
            _file: std::fs::File,
        }

        impl MappedGguf {
            fn open(path: &std::path::Path) -> std::io::Result<Self> {
                let file = std::fs::File::open(path)?;
                let len = usize::try_from(file.metadata()?.len())
                    .expect("fixture file length fits in usize");
                // SAFETY: `len` matches the just-opened file's own length;
                // `file` is kept alive in `_file` for as long as `base` is
                // used, and the mapping is read-only/private so no writer
                // can observe or race it.
                let base = unsafe {
                    rustix::mm::mmap(
                        core::ptr::null_mut(),
                        len,
                        rustix::mm::ProtFlags::READ,
                        rustix::mm::MapFlags::PRIVATE,
                        file.as_fd(),
                        0,
                    )
                }
                .expect("mmap host-local release-gate gguf fixture")
                .cast::<u8>();
                Ok(Self {
                    base,
                    len,
                    _file: file,
                })
            }

            fn as_slice(&self) -> &[u8] {
                // SAFETY: `base` points at `len` bytes mapped for `self`'s
                // whole lifetime; this borrows `self` immutably, so nothing
                // can unmap the region while the returned slice is alive.
                unsafe { core::slice::from_raw_parts(self.base, self.len) }
            }

            /// Explicit rather than left to `Drop` -- this test reads the
            /// device/process footprint immediately after unmapping, and
            /// that ordering (drop the model, THEN unmap, THEN measure) is
            /// the whole point: `register_checkpoint_mapping`'s own no-copy
            /// buffer aliases this mapping directly, so a real fix must
            /// release Metal's reference to it before the mapping itself
            /// goes away, not merely before the test happens to check.
            fn unmap(self) {
                // SAFETY: `base`/`len` are exactly what `open`'s `mmap`
                // call returned; nothing else unmaps this region.
                let _ = unsafe { rustix::mm::munmap(self.base.cast::<c_void>(), self.len) };
                core::mem::forget(self);
            }
        }

        impl Drop for MappedGguf {
            fn drop(&mut self) {
                // SAFETY: only reachable if `unmap` was never called --
                // `unmap` itself `mem::forget`s `self` after unmapping.
                let _ = unsafe { rustix::mm::munmap(self.base.cast::<c_void>(), self.len) };
            }
        }

        const FIXTURE_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3e4cb14174460404e7a233e531675303b2fbf7749c02f91864fe311ab6344e4f";

        #[test]
        #[ignore = "depends on a host-local ollama gguf blob outside this repo"]
        fn dropping_a_loaded_model_releases_its_device_buffers() {
            crate::test_support::require_fixture(FIXTURE_PATH, None);
            let path = std::path::Path::new(FIXTURE_PATH);

            let before_load = omega::metal::current_allocated_size();
            let before_load_footprint = super::super::phys_footprint_bytes();
            std::println!(
                "before_load current_allocated_size={before_load:?} phys_footprint_bytes={before_load_footprint}"
            );

            let mapped = MappedGguf::open(path).expect("mmap host-local release-gate fixture");
            let file_bytes = mapped.as_slice();
            let parsed = proxima_gguf::pipe::parse_complete(file_bytes)
                .expect("parse host-local release-gate fixture");
            let model = LoadedModel::load(&parsed, file_bytes)
                .expect("load real checkpoint through the public path");

            let serving_config = super::super::supported_serving_config(
                GPU_LAYERS_ALL,
                omega::MathMode::default(),
            );
            let (generated_ids, _text, _stopped_by_eos) = model
                .generate_with_serving_config("The quick brown fox", 4, serving_config)
                .expect("decode 4 tokens on Metal");
            assert_eq!(generated_ids.len(), 4, "must actually run 4 decode steps");

            let after_generate = omega::metal::current_allocated_size();
            let after_generate_footprint = super::super::phys_footprint_bytes();
            std::println!(
                "after_generate current_allocated_size={after_generate:?} phys_footprint_bytes={after_generate_footprint}"
            );

            drop(model);
            mapped.unmap();

            let after_drop = omega::metal::current_allocated_size();
            let after_drop_footprint = super::super::phys_footprint_bytes();
            std::println!(
                "after_drop current_allocated_size={after_drop:?} phys_footprint_bytes={after_drop_footprint}"
            );
        }
    }
}

/// The defect ROW 329's slice found, proved directly: every
/// [`BackendRuntime::placed_plans`] build closure now routes through
/// [`BackendRuntime::build_placed_plan`], so a shape first resolved through
/// a diagnostic path (`evaluate_op_timed_with_placements`/
/// `evaluate_dispatch_timed_with_placements`) carries the SAME math mode
/// and dispatch type a later hit from the production path
/// ([`BackendRuntime::evaluate_with_placements`]) would have applied,
/// instead of silently keeping [`omega::metal::MathMode::default`]/
/// [`omega::metal::DispatchType::default`].
///
/// Exercises [`BackendRuntime::build_placed_plan`] directly against the
/// smallest program that plans without touching a Metal device
/// ([`omega::metal::plan`]/[`omega::metal::plan_named`] only resolve
/// shapes and codecs -- no `MTLDevice` is opened until an `execute_plan*`
/// call, per that function's own doc), rather than driving a full decode
/// step through the driver.
#[cfg(all(
    test,
    feature = "metal-output-placement",
    feature = "instrument",
    target_os = "macos"
))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod placed_plan_mode_tests {
    use alloc::collections::BTreeSet;
    use alloc::string::String;
    use alloc::vec::Vec;

    use proxima_tensor::{
        DType, Extent, IndexMap, NodeId, Op, QuantizedBlock, ScalarOp, append, projection,
    };

    use super::BackendRuntime;

    /// `Input(name = "x") -> Elementwise(Identity)` -- the same minimal
    /// identity shape `omega`'s own `metal_output_placement.rs` test uses,
    /// named so [`proxima_tensor::resolve_named_blocks`] (which
    /// [`omega::plan_named`] calls) can bind it.
    fn named_identity_program() -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: Some(String::from("x")),
            },
        );
        let identity = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(source, IndexMap::Affine(projection(1, &[0])))],
                name: None,
            },
        );
        (program, identity)
    }

    #[test]
    fn build_placed_plan_applies_the_runtimes_math_mode_and_dispatch_type() {
        let (program, identity_node) = named_identity_program();
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let named: [(&str, QuantizedBlock<'_>); 1] = [("x", QuantizedBlock::Float32(&data))];
        let resident_names: BTreeSet<&str> = BTreeSet::new();

        let plan = BackendRuntime::build_placed_plan(
            &program,
            &[],
            &named,
            &[identity_node],
            &resident_names,
            &super::PlanNumerics {
                math_mode: omega::metal::MathMode::Safe,
                numeric_policy: proxima_tensor::NumericPolicy::bit_exact(),
                dispatch_type: omega::metal::DispatchType::Serial,
            },
        )
        .expect("plans the identity program under an explicit non-default mode");

        assert_eq!(
            plan.math_mode(),
            omega::metal::MathMode::Safe,
            "a freshly built placed plan must carry the caller's math mode, \
             not MathMode::default() (Relaxed)"
        );
        assert_eq!(
            plan.dispatch_type(),
            omega::metal::DispatchType::Serial,
            "a freshly built placed plan must carry the caller's dispatch type, \
             not DispatchType::default() (Concurrent)"
        );
        assert_eq!(
            plan.numeric_policy(),
            proxima_tensor::NumericPolicy::bit_exact(),
            "a freshly built placed plan must carry the caller's OWN numeric policy -- the \
             policy `plan_named_placed` was constructed under, never a later math-mode setter"
        );
    }

    /// Names the invariant `build_placed_plan`'s construction-time wiring
    /// preserves: a caller declaring [`proxima_tensor::NumericPolicy::llama_relaxed`]
    /// (this crate's own default, `ServingConfig::numeric_policy`'s doc)
    /// alongside `MathMode::Safe` (a narrower compiled mode than the bound
    /// policy grants) must still see `llama_relaxed()` on the resulting
    /// plan's `numeric_policy()` -- `set_math_mode` only narrows the
    /// COMPILED mode within the already-bound policy
    /// (`omega::metal::Plan::set_math_mode`'s own doc); it can never widen
    /// or replace the policy the plan was constructed under.
    #[test]
    fn build_placed_plan_s_numeric_policy_survives_a_narrower_math_mode() {
        let (program, identity_node) = named_identity_program();
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let named: [(&str, QuantizedBlock<'_>); 1] = [("x", QuantizedBlock::Float32(&data))];
        let resident_names: BTreeSet<&str> = BTreeSet::new();

        let plan = BackendRuntime::build_placed_plan(
            &program,
            &[],
            &named,
            &[identity_node],
            &resident_names,
            &super::PlanNumerics {
                math_mode: omega::metal::MathMode::Safe,
                numeric_policy: proxima_tensor::NumericPolicy::llama_relaxed(),
                dispatch_type: omega::metal::DispatchType::Serial,
            },
        )
        .expect("Safe never needs a permission llama_relaxed() withholds, so narrowing succeeds");

        assert_eq!(
            plan.numeric_policy(),
            proxima_tensor::NumericPolicy::llama_relaxed(),
            "the plan's bound numeric_policy must be exactly what it was constructed under, \
             regardless of the narrower compiled math_mode -- if this reads bit_exact(), the \
             construction-time wiring in build_placed_plan regressed"
        );
        assert_eq!(plan.math_mode(), omega::metal::MathMode::Safe);
    }

    /// The mismatch direction: a caller cannot narrow to a `MathMode` that
    /// needs a permission the plan's bound policy withholds.
    #[test]
    fn build_placed_plan_s_math_mode_narrowing_rejects_a_permission_the_bound_policy_withholds() {
        let (program, identity_node) = named_identity_program();
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let named: [(&str, QuantizedBlock<'_>); 1] = [("x", QuantizedBlock::Float32(&data))];
        let resident_names: BTreeSet<&str> = BTreeSet::new();

        let error = match BackendRuntime::build_placed_plan(
            &program,
            &[],
            &named,
            &[identity_node],
            &resident_names,
            &super::PlanNumerics {
                math_mode: omega::metal::MathMode::Fast,
                numeric_policy: proxima_tensor::NumericPolicy::bit_exact(),
                dispatch_type: omega::metal::DispatchType::Serial,
            },
        ) {
            Ok(_) => panic!(
                "Fast needs nan_assumptions/signed_zero/approx_functions, bit_exact grants none"
            ),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("bit_exact") || error.to_string().contains("false"),
            "error must name the bound policy: {error}"
        );
    }
}

/// [`LoadedModel::apply_memory_fit_gate`]'s own contract, exercised
/// directly against a struct-literal [`LoadedModel`] -- private-field
/// construction is legitimate here (same module tree) and cheaper than a
/// full loadable checkpoint: the gate only ever reads
/// `self.checkpoint_weight_bytes`/`self.architecture`, never `self.weights`/
/// `self.program`/`self.vocab`, so those fields are empty stand-ins.
/// Requires a real Metal device (`omega::metal::system_memory_facts`), the
/// same requirement [`crate::memory_fit`]'s own doc names for anything
/// beyond its pure formulas.
#[cfg(all(test, feature = "metal", target_os = "macos"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod memory_fit_gate_tests {
    use alloc::format;
    use alloc::string::String;
    use alloc::vec::Vec;

    use proxima_tokenizer::Vocab;

    use crate::bind::{BoundWeights, ModelArchitecture};
    use crate::serving::ServingConfig;

    use super::LoadedModel;

    /// A minimal valid byte-level BPE vocab -- every base-byte token
    /// present ([`Vocab::new`]'s own precondition), no merges, no special
    /// tokens beyond the one this gate never reads anyway (the fit gate
    /// touches `self.architecture`/`self.checkpoint_weight_bytes` only).
    fn tiny_vocab() -> Vocab {
        let tokens: Vec<String> = (0..=255u8)
            .map(|byte| format!("<0x{byte:02X}>"))
            .collect();
        Vocab::new(tokens, &[], None, None, None).expect("minimal vocab builds")
    }

    fn tiny_architecture() -> ModelArchitecture {
        ModelArchitecture {
            vocab: 1,
            embedding: 1,
            feed_forward: 1,
            query_heads: 1,
            kv_heads: 2,
            head_dim: 64,
            block_count: 2,
            expert_count: 0,
            expert_used_count: 0,
            rope_freq_base: 10_000.0,
            rms_epsilon: 1e-5,
            tied_embeddings: false,
        }
    }

    fn model_with(dense_weight_bytes: u64) -> LoadedModel<'static> {
        LoadedModel {
            weights: BoundWeights {
                resident_bytes: 0,
                owned: Vec::new(),
                packed: Vec::new(),
                packed_owned: Vec::new(),
                precision: &[],
            },
            architecture: tiny_architecture(),
            checkpoint_weight_bytes: crate::memory_fit::WeightClassBytes {
                dense_bytes: dense_weight_bytes,
                expert_bytes: 0,
                table_bytes: 0,
                ssm_state_bytes: 0,
            },
            model_name: None,
            checkpoint_bytes: dense_weight_bytes as usize,
            checkpoint_mapping: &[],
            vocab: tiny_vocab(),
            program: Vec::new(),
            logits_root: proxima_tensor::op::NodeId(0),
            layer_roots: Vec::new(),
            qwen35_ssm_shape: None,
            qwen35_attn_head_dim: None,
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            single_range: None,
        }
    }

    /// (d) the gate off: an absurd `context_length` that would otherwise
    /// need reducing (or would exceed the limit outright) passes through
    /// completely unchanged when `gpu_memory_fit` is `false` -- the
    /// caller's explicit opt-out.
    #[test]
    fn gate_off_leaves_an_otherwise_infeasible_context_length_untouched() {
        let model = model_with(1_000_000);
        let mut serving_config = ServingConfig {
            gpu_memory_fit: false,
            context_length: u32::MAX,
            ..ServingConfig::default()
        };

        model
            .apply_memory_fit_gate(&mut serving_config)
            .expect("gate must be a no-op when gpu_memory_fit is false");

        assert_eq!(
            serving_config.context_length,
            u32::MAX,
            "gate must not touch context_length when the caller opted out"
        );
    }

    /// (a) fits: this fixture's tiny weights and default 131072 context
    /// budget are well inside any real Metal device's own reported limit.
    #[test]
    fn gate_on_leaves_a_generously_fitting_context_length_unchanged() {
        let model = model_with(1_000_000);
        let mut serving_config = ServingConfig {
            gpu_memory_fit: true,
            ..ServingConfig::default()
        };
        let requested = serving_config.context_length;

        model
            .apply_memory_fit_gate(&mut serving_config)
            .expect("a real device's own limit must comfortably fit this fixture's budget");

        assert_eq!(
            serving_config.context_length, requested,
            "a generously fitting budget must not reduce context_length"
        );
    }
}
