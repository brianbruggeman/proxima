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

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec::Vec;
use core::future::Future;
use core::ops::Range;

use memmap2::{Advice, Mmap};
use proxima_gguf::GgmlType;
use proxima_gguf::pipe::ParsedGguf;
use proxima_primitives::pipe::Pipe;
#[cfg(all(
    feature = "instrument",
    feature = "metal",
    target_os = "macos",
    not(feature = "metal-output-placement")
))]
use proxima_tensor::DType;
#[cfg(not(feature = "metal"))]
use proxima_tensor::cpu::evaluate_quantized_named_with_scratch_and_experts;
use proxima_tensor::cpu::{
    Evaluated, QuantizedBlock, evaluate_quantized_named_exact_with_scratch_and_experts,
};
#[cfg(all(
    feature = "instrument",
    feature = "metal",
    target_os = "macos",
    not(feature = "metal-output-placement")
))]
use proxima_tensor::op::ScalarOp;
use proxima_tensor::op::{Extent, NodeId, Op};
#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
use proxima_tensor::spec::CachedLayerRoots;
use proxima_tensor::spec::{
    Qwen35LayerRoots, mistral_cached_forward_program_with_experts_and_layer_taps,
};
use proxima_tokenizer::{SamplingConfig, Vocab, sample_next_token};
use std::fs::File;
use std::sync::Arc;

#[cfg(all(
    feature = "instrument",
    feature = "metal",
    target_os = "macos",
    not(feature = "metal-output-placement")
))]
use omega::backend::execute_plan_named;
#[cfg(all(
    feature = "instrument",
    feature = "metal",
    target_os = "macos",
    not(feature = "metal-output-placement")
))]
use omega::backend::execute_plan_named_metal_op_timed;
#[cfg(all(
    feature = "instrument",
    feature = "metal",
    target_os = "macos"
))]
use omega::backend::execute_plan_named_metal_op_timed_with_expert_sources;
#[cfg(feature = "metal")]
use omega::backend::{
    Engine, Plan, execute_plan_named_with_expert_sources, mark_resident, plan_named,
    plan_named_exact, release_resident_names, unregister_checkpoint_mapping,
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
    execute_plan_named_with_placements_and_expert_sources, plan_named as plan_named_placed,
};
#[cfg(feature = "instrument")]
use proxima_telemetry::{debug, info};
#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
use proxima_tensor::TensorError;
#[cfg(feature = "instrument")]
use proxima_tensor::instrument::{elapsed_ticks, read_ticks, ticks_to_nanos};
#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
use proxima_tensor::spec::{DuplicateHeadPosition, mistral_single_range_cached_forward_program};

use crate::architecture::{Architecture, StepInput, StepInputContext, bind_symbols};
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

#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
fn routed_segment_profile_selected(layer: usize, phase: &str) -> bool {
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
#[cfg(all(
    feature = "metal-output-placement",
    feature = "instrument",
    target_os = "macos"
))]
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

/// `PROXIMA_METAL_COMPARE_CPU`'s own node-selection filter (`evaluate_op_timed`
/// above): `true` when `node` is a fused quantized matmul's `Multiply`
/// elementwise -- weight times activation, feeding a `Reduce::Add` -- the
/// SAME shape [`proxima_tensor::cpu`]'s `is_quantized_matmul_operand`
/// exempts from `reject_non_float32`, checked here from the elementwise
/// node's own side rather than the weight's. Neither engine's `bind::bind`
/// ever gives this node a standalone buffer unless it is itself a
/// requested output: the real Metal `plan` this diagnostic runs alongside
/// requests only `outputs` (a handful of `roots`), so `device_buffers`
/// never holds an entry for it and `compare_op_output_to_cpu` can never
/// diff it. Requesting it anyway on the CPU side only forces
/// `materialize_quantized_weight_output` to dequantize the whole weight
/// matrix to an owned `Vec<f32>` for a comparison that can never happen --
/// see `evaluate_op_timed`'s own doc for the measured cost this excludes.
#[cfg(all(
    feature = "instrument",
    feature = "metal",
    target_os = "macos",
    not(feature = "metal-output-placement")
))]
fn is_quantized_matmul_multiply(program: &[Op], node: NodeId) -> bool {
    let Op::Elementwise {
        body: ScalarOp::Multiply,
        operands,
        ..
    } = &program[node.0 as usize]
    else {
        return false;
    };
    if operands.len() != 2 {
        return false;
    }
    let has_quantized_weight_operand = operands.iter().any(|(source, _)| {
        matches!(
            &program[source.0 as usize],
            Op::Input { dtype, .. } if *dtype != DType::Float32
        )
    });
    if !has_quantized_weight_operand {
        return false;
    }
    program
        .iter()
        .any(|other| matches!(other, Op::Reduce(fold) if fold.operand == node && fold.body == ScalarOp::Add))
}

/// Prints the per-op GPU attribution `run_decode_loop`'s
/// `PROXIMA_METAL_OP_PROFILE_STEP` branch gathers for exactly one decode
/// step: the op count and summed GPU time (asserting the count so a
/// degenerate empty profile reads as RED, not quiet), one line per
/// `OpGpuTiming::kind` bucket, and the top [`OP_PROFILE_TOP_N`] ops by GPU
/// time with their operand bytes and bytes/ns -- exactly what settles
/// whether GPU time tracks operand bytes or is flat per dispatch.
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
fn report_op_timings(step: usize, timings: &[OpGpuTiming], program: &[Op]) {
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

    let mut cooperative_shapes: alloc::collections::BTreeMap<
        (Vec<u64>, Vec<u16>),
        (u64, u64, u64),
    > = alloc::collections::BTreeMap::new();
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
        if timing.gpu_ns >= 100_000 {
            if let Some(op) = program.get(timing.node.0 as usize) {
                eprintln!(
                    "op_profile_cooperative_slow step={step} node={} gpu_ms={:.3} op={op:?}",
                    timing.node.0,
                    timing.gpu_ns as f64 / 1e6,
                );
            }
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
    ssm_state_transfer_bytes: u64,
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
        expert_source_cache_hits = metal_stage.expert_source_cache_hits,
        expert_source_cache_misses = metal_stage.expert_source_cache_misses,
        plan_cache_len = plan_cache_len as u64,
        plan_hits = plan_hits as u64,
        plan_misses = plan_misses as u64,
        ablation = !kind_filter.is_empty(),
        kind_filter,
        "token_breakdown_metal: per-decode-step metal stage attribution"
    );
    if std::env::var_os("PROXIMA_DEBUG_METAL_STAGES").is_some() {
        eprintln!(
            "token_breakdown_metal step={} expert_source_cache_hits={} expert_source_cache_misses={} expert_source_buffer_reuses={} plan_handoff_reuses={} expert_source_cache_entries={} nocopy_cache_entries={} block_upload_calls={} block_copied_bytes={} block_nocopy_bound_bytes={} block_offset_bound_bytes={} resident_uploads={} resident_reuses={} gpu_exec_ms={} phys_footprint_bytes={} device_allocated_bytes={}",
            step,
            metal_stage.expert_source_cache_hits,
            metal_stage.expert_source_cache_misses,
            metal_stage.expert_source_buffer_reuses,
            metal_stage.plan_handoff_reuses,
            metal_stage.expert_source_cache_entries,
            metal_stage.nocopy_cache_entries,
            metal_stage.block_upload_calls,
            metal_stage.block_copied_bytes,
            metal_stage.block_nocopy_bound_bytes,
            metal_stage.block_offset_bound_bytes,
            metal_stage.resident_uploads,
            metal_stage.resident_reuses,
            ms(metal_stage.gpu_exec_ticks),
            phys_footprint_bytes(),
            omega::metal::current_allocated_size().unwrap_or(0),
        );
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
    /// [`Self::load`]'s resolved [`crate::architecture::Architecture`] impl,
    /// kept so [`Self::run_decode_loop_observed_seeded`] can call
    /// [`Architecture::step_inputs`] every step -- `None` only on the
    /// narrow `load_inner` fallthrough that predates the registry seam
    /// (`paired_gate_up_reduce`/`fused_qkv_reduce` on a non-qwen35
    /// checkpoint), which never resolves against an `Architecture` at all.
    architecture_impl: Option<&'static dyn Architecture>,
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
    /// `proxima_tensor::spec::ForwardRoots::hidden` off the dense load path
    /// (`Self::load`/`Self::load_from_safetensors`, both wrapping
    /// `mistral_cached_forward_program_with_experts`) -- `None` on the
    /// qwen35 hybrid path (`crate::qwen35::qwen35_forward_program` returns
    /// a bare `logits` root with no named hidden-state counterpart yet).
    hidden_root: Option<NodeId>,
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
    /// Graph-level producer boundaries for each qwen35moe layer.
    qwen35moe_layer_diagnostics: Vec<crate::qwen35moe::Qwen35MoeLayerDiagnostics>,
    /// Router-logit roots aligned with routed layers.  Qwen35MoE fills this
    /// from the same graph nodes used by its gather; other architectures leave
    /// it empty.  These roots are the concrete input to a future per-layer
    /// pre-gather evaluator, not a second router computation.
    router_roots: Vec<NodeId>,
    /// One [`proxima_tensor::spec::MoeSite`] per MoE layer this
    /// checkpoint's forward-program builder produced -- empty on a dense
    /// checkpoint. [`Self::run_decode_loop_observed`] reads this to know
    /// which extra nodes to request as step outputs when a routing observer
    /// (`proxima_tensor::instrument::ExpertObserver`, `instrument`-gated)
    /// is registered.
    moe_sites: proxima_tensor::spec::MoeSites,
    /// `crate::architecture::BoundProgram::single_position_step` off this
    /// checkpoint's own resolved [`crate::Architecture`] (`Self::load`'s
    /// `resolved.bind(..)` for the registry path; `false` for every other
    /// load entry point below, all of which wrap
    /// `mistral_cached_forward_program_with_experts`) -- read by
    /// [`Self::run_decode_loop_observed_seeded`] to decide whether prefill
    /// batches its whole prompt into one evaluation or feeds it one
    /// position at a time.
    single_position_step: bool,
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
    expert_slab: std::sync::Mutex<crate::expert_slab::ExpertSlab<'file>>,
    /// HOBBIT's optional low-codec store and its mmap owner. Attachment
    /// installs the low copies once; router boundaries subsequently switch
    /// only the selected expert's three projection entries.
    expert_sidecar: Option<crate::expert_sidecar::MappedExpertSidecar>,
}

/// Concrete qwen35moe router/gather partitions for one KV shape bucket.
///
/// GDN prefill must remain sequential by position, but the graph cuts do
/// not change while `(new_count, kv_bound_extent)` is unchanged. Keeping
/// the dense partitions beside that concrete shape prevents every prompt
/// position from repeating partitioning, topological ordering, and shape
/// inference before it can execute the same router/residency/gather phases.
struct Qwen35MoePreGatherPlan {
    symbols: Vec<u64>,
    layers: Vec<Qwen35MoeLayerSegments>,
    suffix: crate::qwen35moe::execution::MappedLayerSegment,
    prefix_carried_nodes: BTreeSet<NodeId>,
    global_cut_nodes: BTreeSet<NodeId>,
}

struct Qwen35MoeLayerSegments {
    router: crate::qwen35moe::execution::MappedLayerSegment,
    gather: crate::qwen35moe::execution::MappedLayerSegment,
    router_future_cuts: Vec<(NodeId, String)>,
    next_cuts: Vec<(NodeId, String)>,
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
fn missing_program_input(program: &[Op], named: &[(&str, QuantizedBlock<'_>)]) -> Option<String> {
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
fn kv_extent(merged_len: usize, capacity: usize, bucket_tokens: usize) -> usize {
    merged_len
        .div_ceil(bucket_tokens)
        .saturating_mul(bucket_tokens)
        .min(capacity)
}

const fn step_batch_needs_logits(split_prefill: bool, is_last_step_batch: bool) -> bool {
    !split_prefill || is_last_step_batch
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

    /// This checkpoint's pre-`lm_head` hidden-state root
    /// (`proxima_tensor::spec::ForwardRoots::hidden`), when the load path
    /// named one -- `None` on the qwen35 hybrid path
    /// (`crate::qwen35::qwen35_forward_program` carries no named
    /// hidden-state root yet). A caller composes this with
    /// [`Self::forward_node_values`] to read that tensor's row out
    /// directly; proxima names the node, it does not decide how a caller
    /// pools it (last-token, mean, or otherwise is the caller's own
    /// policy).
    #[must_use]
    pub fn hidden_root(&self) -> Option<NodeId> {
        self.hidden_root
    }

    /// Router-logit roots aligned with the routed layers of this model.
    /// These are the exact nodes already used by the production MoE graph;
    /// an empty slice means the loaded architecture has no exposed
    /// pre-gather roots.
    #[must_use]
    pub fn router_roots(&self) -> &[NodeId] {
        &self.router_roots
    }

    /// Evaluates the bound qwen35moe router roots for one prompt position.
    /// The returned vectors are the graph's actual per-layer router logits in
    /// layer order, so DynaExq can make a residency decision from execution
    /// data rather than from a duplicated host-side router. This diagnostic
    /// does not claim a memory reduction: it uses the ordinary evaluator
    /// until the production per-layer pre-gather path is enabled.
    pub fn qwen35moe_router_logits(
        &self,
        prompt: &str,
        gpu_layers: i32,
    ) -> Result<Vec<Vec<f32>>, InteropError> {
        if self.router_roots.is_empty()
            || self
                .architecture_impl
                .map_or(true, |architecture| architecture.name() != "qwen35moe")
        {
            return Err(InteropError::PreGatherExecutionUnsupported {
                architecture: String::from(
                    self.architecture_impl.map_or("unknown", Architecture::name),
                ),
                reason: String::from("the bound model does not expose qwen35moe router roots"),
            });
        }
        self.forward_node_values_on_backend(prompt, &self.router_roots, gpu_layers)
    }

    /// Builds the graph cuts that surround each qwen35moe routed layer. The
    /// cuts are plan-time data: callers evaluate one producer, apply the
    /// residency transition, then evaluate its consumer with the borrowed
    /// activation handoff. No cut is built for another architecture.
    pub fn qwen35moe_layer_boundaries(
        &self,
        symbols: &[u64],
    ) -> Result<Vec<crate::qwen35moe::execution::LayerProgramBoundary>, InteropError> {
        if self.qwen35moe_layer_diagnostics.is_empty()
            || self
                .architecture_impl
                .is_none_or(|architecture| architecture.name() != "qwen35moe")
        {
            return Err(InteropError::PreGatherExecutionUnsupported {
                architecture: String::from(
                    self.architecture_impl.map_or("unknown", Architecture::name),
                ),
                reason: String::from("the bound model has no qwen35moe layer diagnostics"),
            });
        }
        self.qwen35moe_layer_diagnostics
            .iter()
            .map(|diagnostic| {
                crate::qwen35moe::execution::split_layer_program(
                    &self.program,
                    symbols,
                    diagnostic.router_logits,
                    diagnostic.routed_output,
                )
                .map_err(InteropError::from)
            })
            .collect()
    }

    /// Builds dense router and gather programs for every routed layer in
    /// execution order. Each layer starts from the prior layer's block output,
    /// so callers can evaluate one layer, change resident expert sources, and
    /// continue without retaining a full-program suffix.
    pub fn qwen35moe_layer_segments(
        &self,
        symbols: &[u64],
    ) -> Result<
        Vec<(
            (Vec<proxima_tensor::op::Op>, Vec<(NodeId, String)>),
            (Vec<proxima_tensor::op::Op>, Vec<(NodeId, String)>),
        )>,
        InteropError,
    > {
        if self.qwen35moe_layer_diagnostics.is_empty()
            || self
                .architecture_impl
                .is_none_or(|architecture| architecture.name() != "qwen35moe")
        {
            return Err(InteropError::PreGatherExecutionUnsupported {
                architecture: String::from(
                    self.architecture_impl.map_or("unknown", Architecture::name),
                ),
                reason: String::from("the bound model has no qwen35moe layer diagnostics"),
            });
        }
        let mut segments = Vec::with_capacity(self.qwen35moe_layer_diagnostics.len());
        let mut previous_output = None;
        for diagnostic in &self.qwen35moe_layer_diagnostics {
            let pair = crate::qwen35moe::execution::split_router_and_gather_segments(
                &self.program,
                symbols,
                previous_output,
                diagnostic.router_logits,
                diagnostic.block_output,
            )
            .map_err(InteropError::from)?;
            segments.push(pair);
            previous_output = Some(diagnostic.block_output);
        }
        Ok(segments)
    }

    /// Builds one routed layer's segments on demand. The caller can drop the
    /// pair after the layer gather, keeping graph metadata bounded by one
    /// layer instead of materializing all 40 layer segments at once.
    pub fn qwen35moe_layer_segment(
        &self,
        layer: usize,
        symbols: &[u64],
        previous_layer_output: Option<NodeId>,
    ) -> Result<
        (
            (Vec<proxima_tensor::op::Op>, Vec<(NodeId, String)>),
            (Vec<proxima_tensor::op::Op>, Vec<(NodeId, String)>),
        ),
        InteropError,
    > {
        let diagnostic = self.qwen35moe_layer_diagnostics.get(layer).ok_or_else(|| {
            InteropError::PreGatherExecutionUnsupported {
                architecture: String::from("qwen35moe"),
                reason: String::from("requested routed layer is outside the bound diagnostics"),
            }
        })?;
        crate::qwen35moe::execution::split_router_and_gather_segments(
            &self.program,
            symbols,
            previous_layer_output,
            diagnostic.router_logits,
            diagnostic.block_output,
        )
        .map_err(InteropError::from)
    }

    fn qwen35moe_pre_gather_plan(
        &self,
        symbols: &[u64],
    ) -> Result<Qwen35MoePreGatherPlan, InteropError> {
        let last_layer_output = self
            .qwen35moe_layer_diagnostics
            .last()
            .map(|diagnostic| diagnostic.block_output)
            .ok_or_else(|| InteropError::PreGatherExecutionUnsupported {
                architecture: String::from("qwen35moe"),
                reason: String::from("the bound graph has no routed layer boundary"),
            })?;
        let suffix = crate::qwen35moe::execution::split_mapped_layer_segment(
            &self.program,
            symbols,
            Some(last_layer_output),
            self.logits_root,
        )
        .map_err(InteropError::from)?;

        let mut layer_pairs = Vec::with_capacity(self.qwen35moe_layer_diagnostics.len());
        let mut prefix_required_nodes = BTreeSet::new();
        let mut previous_output = None;
        for diagnostic in &self.qwen35moe_layer_diagnostics {
            let pair = crate::qwen35moe::execution::split_mapped_router_and_gather_segments(
                &self.program,
                symbols,
                previous_output,
                diagnostic.router_logits,
                diagnostic.block_output,
            )
            .map_err(InteropError::from)?;
            for (node, _) in &pair.0.1 {
                if !matches!(
                    self.program[node.0 as usize],
                    proxima_tensor::op::Op::Input { .. }
                ) {
                    prefix_required_nodes.insert(*node);
                }
            }
            layer_pairs.push(pair);
            previous_output = Some(diagnostic.block_output);
        }

        // layer zero's router is already the complete graph prefix, so a
        // second prefix segment would execute the same operations twice.
        let first_router_mapping = &layer_pairs
            .first()
            .ok_or_else(|| InteropError::PreGatherExecutionUnsupported {
                architecture: String::from("qwen35moe"),
                reason: String::from("the bound graph has no first router segment"),
            })?
            .0
            .2;
        let prefix_carried_nodes = first_router_mapping
            .keys()
            .filter(|node| prefix_required_nodes.contains(node))
            .copied()
            .collect();
        let global_cut_nodes = self
            .program
            .iter()
            .enumerate()
            .filter_map(|(index, operation)| {
                operation
                    .name()
                    .is_some_and(|name| name.starts_with("__cut_"))
                    .then_some(NodeId(index as u32))
            })
            .collect();

        let mut layers = Vec::with_capacity(layer_pairs.len());
        for layer in 0..layer_pairs.len() {
            let (router, gather) = layer_pairs[layer].clone();
            let next_router_cuts = layer_pairs
                .get(layer + 1)
                .map_or_else(Vec::new, |pair| pair.0.1.clone());
            let next_cuts = layer_pairs
                .get(layer + 1)
                .map_or_else(|| suffix.1.clone(), |pair| pair.0.1.clone());
            let mut router_future_cuts = gather.1.clone();
            router_future_cuts.extend(next_router_cuts);
            router_future_cuts.sort_by_key(|(node, _)| *node);
            router_future_cuts.dedup_by_key(|(node, _)| *node);
            layers.push(Qwen35MoeLayerSegments {
                router,
                gather,
                router_future_cuts,
                next_cuts,
            });
        }

        for (layer, segments) in layers.iter().enumerate() {
            for (phase, program) in [
                ("router", &segments.router.0),
                ("gather", &segments.gather.0),
            ] {
                proxima_tensor::shape::infer(program, symbols).map_err(|error| {
                    InteropError::PreGatherExecutionUnsupported {
                        architecture: String::from("qwen35moe"),
                        reason: alloc::format!("layer {layer} {phase} segment is invalid: {error}"),
                    }
                })?;
            }
            if std::env::var_os("PROXIMA_DEBUG_QWEN35_PLAN").is_some() && layer < 3 {
                eprintln!(
                    "qwen35 plan layer={layer} router_cuts={:?} gather_cuts={:?} router_future={:?} map={:?} op493={:?} gather_ops={:?}",
                    segments.router.1,
                    segments.gather.1,
                    segments.router_future_cuts,
                    segments
                        .gather
                        .2
                        .iter()
                        .filter(|(_, mapped)| {
                            let maximum = std::env::var("PROXIMA_DEBUG_QWEN35_PLAN_MAX_MAPPED")
                                .ok()
                                .and_then(|value| value.parse::<u32>().ok())
                                .unwrap_or(24);
                            mapped.0 <= maximum
                        })
                        .collect::<Vec<_>>(),
                    self.program.get(493),
                    segments
                        .gather
                        .0
                        .iter()
                        .enumerate()
                        .map(|(index, operation)| (index, operation.name()))
                        .collect::<Vec<_>>()
                );
            }
        }

        #[cfg(feature = "instrument")]
        debug!(
            symbol_count = symbols.len() as u64,
            layer_count = layers.len() as u64,
            source_program_ops = self.program.len() as u64,
            cached_segment_ops = (layers
                .iter()
                .map(|segments| segments.router.0.len() + segments.gather.0.len())
                .sum::<usize>()
                + suffix.0.len()) as u64,
            "qwen35moe pre-gather partitions built because the concrete shape changed"
        );

        Ok(Qwen35MoePreGatherPlan {
            symbols: symbols.to_vec(),
            layers,
            suffix,
            prefix_carried_nodes,
            global_cut_nodes,
        })
    }

    fn evaluate_qwen35moe_pre_gather<BeforeGather>(
        &self,
        runtime: &mut BackendRuntime,
        plan: &Qwen35MoePreGatherPlan,
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
        expert_slab: &mut crate::expert_slab::ExpertSlab<'file>,
        position_offset: usize,
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))] ssm_placement: Option<
            &Qwen35SsmPlacement<'_>,
        >,
        mut before_gather: BeforeGather,
    ) -> Result<Evaluated, InteropError>
    where
        BeforeGather: FnMut(
            usize,
            u64,
            &[crate::residency::RoutedExpert],
            &mut crate::expert_slab::ExpertSlab<'file>,
        ) -> Result<(), InteropError>,
    {
        let mut carried: BTreeMap<NodeId, (Vec<u64>, Vec<f32>)> = BTreeMap::new();
        let mut results: BTreeMap<NodeId, (Vec<u64>, Vec<f32>)> = BTreeMap::new();
        let mut routed_experts = Vec::with_capacity(self.architecture.expert_used_count as usize);
        let mut sidecar_read_scratch = crate::expert_sidecar::ExpertSidecarReadScratch::default();
        #[cfg(feature = "instrument")]
        let mut segment_execution_count = 0_u64;
        #[cfg(feature = "instrument")]
        let mut router_elapsed_us = 0_u64;
        #[cfg(feature = "instrument")]
        let mut gather_elapsed_us = 0_u64;
        for (index, operation) in self.program.iter().enumerate() {
            if let proxima_tensor::op::Op::Constant { value, .. } = operation {
                carried.insert(NodeId(index as u32), (Vec::new(), vec![*value]));
            }
        }
        for layer in 0..self.qwen35moe_layer_diagnostics.len() {
            let debug_layer = std::env::var("PROXIMA_DEBUG_EXPERT_GATHER_LAYER")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0);
            // A prompt segment contains several rows, each with its own
            // router result. Build one union before the gather snapshot so
            // no row reads an unselected descriptor.
            let segments = &plan.layers[layer];
            let router = &segments.router;
            let gather = &segments.gather;
            let next_cuts = &segments.next_cuts;
            let future_gather_cuts: Vec<NodeId> = plan.layers[layer + 1..]
                .iter()
                .flat_map(|future| future.gather.1.iter().map(|(node, _)| *node))
                .collect();
            let diagnostic = self.qwen35moe_layer_diagnostics[layer];
            for (is_router, program, cuts, mapping, future_cuts, segment_output) in [
                (
                    true,
                    &router.0,
                    &router.1,
                    &router.2,
                    segments.router_future_cuts.as_slice(),
                    diagnostic.router_logits,
                ),
                (
                    false,
                    &gather.0,
                    &gather.1,
                    &gather.2,
                    next_cuts.as_slice(),
                    diagnostic.block_output,
                ),
            ] {
                let mut segment_named: Vec<(&str, QuantizedBlock<'_>)> = named
                    .iter()
                    .copied()
                    .filter(|(name, _)| {
                        program
                            .iter()
                            .any(|operation| operation.name() == Some(*name))
                    })
                    .collect();
                for (node, name) in cuts {
                    if segment_named
                        .iter()
                        .any(|(candidate, _)| *candidate == name)
                    {
                        continue;
                    }
                    if name.contains("_exps.weight") {
                        // The mapped expert source supplies this input at
                        // execution time; a placeholder would make the
                        // resolver classify the node as an ordinary f32
                        // binding and bypass the source table.
                        continue;
                    }
                    let (_, values) = carried.get(node).ok_or_else(|| {
                        InteropError::PreGatherExecutionUnsupported {
                            architecture: String::from("qwen35moe"),
                            reason: alloc::format!(
                                "layer {layer} missing cut node {node:?} ({name})"
                            ),
                        }
                    })?;
                    segment_named.push((name.as_str(), QuantizedBlock::Float32(values)));
                }

                let mut requested = BTreeMap::new();
                let prefix_carried_nodes =
                    (layer == 0 && is_router).then_some(&plan.prefix_carried_nodes);
                for node in future_cuts
                    .iter()
                    .filter(|(_, name)| !named.iter().any(|(candidate, _)| *candidate == name))
                    .map(|(node, _)| node)
                    .chain(prefix_carried_nodes.into_iter().flatten())
                    .chain(plan.global_cut_nodes.iter())
                    .chain(future_gather_cuts.iter())
                    .chain(outputs)
                    .filter(|node| {
                        !matches!(
                            self.program[node.0 as usize],
                            proxima_tensor::op::Op::Input { .. }
                        )
                    })
                {
                    if let Some(mapped) = mapping.get(&node).copied() {
                        requested.insert(mapped, *node);
                    }
                }
                requested.insert(
                    mapping.get(&segment_output).copied().ok_or(
                        InteropError::MissingEvaluatedNode {
                            node: segment_output,
                        },
                    )?,
                    segment_output,
                );
                let mut requested_nodes: Vec<NodeId> = requested.keys().copied().collect();
                #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                let mut segment_input_placements = Vec::new();
                #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                let mut segment_output_placements = Vec::new();
                #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                if let Some(placement) = ssm_placement
                    && placement
                        .maximum_layer
                        .is_none_or(|maximum| layer <= maximum)
                    && let (
                        Qwen35LayerRoots::Ssm { state_out, .. },
                        Some(state_input),
                        Some((first_buffer, second_buffer)),
                    ) = (
                        &self.layer_roots[layer],
                        placement.input_nodes[layer],
                        placement.buffers[layer].as_ref(),
                    )
                {
                    let (input_buffer, output_buffer) = if placement.use_second_as_input {
                        (second_buffer, first_buffer)
                    } else {
                        (first_buffer, second_buffer)
                    };
                    if let Some(mapped_input) = mapping.get(&state_input).copied() {
                        segment_input_placements.push((mapped_input, input_buffer, 0));
                    }
                    if let Some(mapped_output) = mapping.get(state_out).copied() {
                        segment_output_placements.push((mapped_output, output_buffer, 0));
                        requested_nodes.push(mapped_output);
                    }
                }
                requested_nodes.sort_unstable_by_key(|node| node.0);
                requested_nodes.dedup();
                if std::env::var_os("PROXIMA_DEBUG_QWEN35_REQUESTS").is_some() {
                    eprintln!(
                        "qwen35 segment requests phase={} layer={} program_ops={} named_inputs={} requested_nodes={} future_cuts={} global_cuts={} outputs={}",
                        if is_router { "router" } else { "gather" },
                        layer,
                        program.len(),
                        segment_named.len(),
                        requested_nodes.len(),
                        future_cuts.len(),
                        plan.global_cut_nodes.len(),
                        outputs.len(),
                    );
                }
                if layer == debug_layer
                    && !is_router
                    && std::env::var_os("PROXIMA_DEBUG_EXPERT_GATHER_PARITY").is_some()
                {
                    if let Some(debug_node) = std::env::var("PROXIMA_DEBUG_EXPERT_GATHER_NODE")
                        .ok()
                        .and_then(|value| value.parse::<u32>().ok())
                    {
                        let debug_node = NodeId(debug_node);
                        requested_nodes.push(debug_node);
                        if let Some(proxima_tensor::op::Op::Elementwise { operands, .. }) =
                            program.get(debug_node.0 as usize)
                        {
                            requested_nodes.extend(operands.iter().map(|(node, _)| *node));
                        }
                    }
                    requested_nodes.sort_unstable_by_key(|node| node.0);
                    requested_nodes.dedup();
                }
                let evaluated = if is_router {
                    let segment_started = std::time::Instant::now();
                    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                    let result = if segment_input_placements.is_empty()
                        && segment_output_placements.is_empty()
                    {
                        runtime.evaluate_segment(
                            program,
                            symbols,
                            &segment_named,
                            &requested_nodes,
                            resident_names,
                            None,
                        )
                    } else {
                        let empty_expert_sources = BTreeMap::new();
                        runtime.evaluate_segment_with_placements_and_expert_sources(
                            program,
                            symbols,
                            &segment_named,
                            &requested_nodes,
                            resident_names,
                            &SegmentMetalBindings {
                                input_placements: &segment_input_placements,
                                output_placements: &segment_output_placements,
                                expert_sources: &empty_expert_sources,
                            },
                        )
                    };
                    #[cfg(not(all(feature = "metal-output-placement", target_os = "macos")))]
                    let result = runtime.evaluate_segment(
                        program,
                        symbols,
                        &segment_named,
                        &requested_nodes,
                        resident_names,
                        None,
                    );
                    #[cfg(feature = "instrument")]
                    {
                        router_elapsed_us += segment_started.elapsed().as_micros() as u64;
                    }
                    if std::env::var_os("PROXIMA_DEBUG_QWEN35_SEGMENTS").is_some() {
                        eprintln!(
                            "qwen35 segment phase=router layer={} elapsed_us={}",
                            layer,
                            segment_started.elapsed().as_micros()
                        );
                    }
                    result?
                } else {
                    let selected_sidecar = if let Some(sidecar) = &self.expert_sidecar
                        && sidecar.uses_file_reads()
                    {
                        sidecar.read_selected_low(
                            layer,
                            expert_slab.selected_experts(layer),
                            &mut sidecar_read_scratch,
                        )?;
                        Some(&sidecar_read_scratch)
                    } else {
                        None
                    };
                    let mut expert_entries_scratch = Vec::with_capacity(3);
                    let expert_sources = expert_slab.sources_for_layer_with_sidecar(
                        layer,
                        selected_sidecar,
                        &mut expert_entries_scratch,
                    )?;
                    let mapped_expert_sources = expert_sources
                        .iter()
                        .flat_map(|(node, source)| {
                            let source_name = self.program[node.0 as usize].name();
                            program
                                .iter()
                                .enumerate()
                                .filter_map(move |(position, operation)| {
                                    (operation.name() == source_name)
                                        .then_some((NodeId(position as u32), *source))
                                })
                        })
                        .collect::<BTreeMap<_, _>>();
                    if std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some() {
                        eprintln!(
                            "qwen35 mapped expert source nodes={:?}",
                            mapped_expert_sources.keys().collect::<Vec<_>>()
                        );
                    }
                    let segment_started = std::time::Instant::now();
                    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                    let result = if segment_input_placements.is_empty()
                        && segment_output_placements.is_empty()
                    {
                        #[cfg(feature = "instrument")]
                        if routed_segment_profile_selected(layer, "gather") {
                            let (evaluated, timings) = runtime.evaluate_segment_op_timed(
                                program,
                                symbols,
                                &segment_named,
                                &requested_nodes,
                                resident_names,
                                &mapped_expert_sources,
                            )?;
                            report_op_timings(position_offset, &timings, program);
                            Ok(evaluated)
                        } else {
                            runtime.evaluate_segment(
                                program,
                                symbols,
                                &segment_named,
                                &requested_nodes,
                                resident_names,
                                Some(&mapped_expert_sources),
                            )
                        }
                        #[cfg(not(feature = "instrument"))]
                        runtime.evaluate_segment(
                            program,
                            symbols,
                            &segment_named,
                            &requested_nodes,
                            resident_names,
                            Some(&mapped_expert_sources),
                        )
                    } else {
                        runtime.evaluate_segment_with_placements_and_expert_sources(
                            program,
                            symbols,
                            &segment_named,
                            &requested_nodes,
                            resident_names,
                            &SegmentMetalBindings {
                                input_placements: &segment_input_placements,
                                output_placements: &segment_output_placements,
                                expert_sources: &mapped_expert_sources,
                            },
                        )
                    };
                    #[cfg(all(
                        feature = "instrument",
                        feature = "metal",
                        target_os = "macos",
                        not(feature = "metal-output-placement")
                    ))]
                    let result = if routed_segment_profile_selected(layer, "gather") {
                        let (evaluated, timings) = runtime.evaluate_segment_op_timed(
                            program,
                            symbols,
                            &segment_named,
                            &requested_nodes,
                            resident_names,
                            &mapped_expert_sources,
                        )?;
                        report_op_timings(position_offset, &timings, program);
                        Ok(evaluated)
                    } else {
                        runtime.evaluate_segment(
                            program,
                            symbols,
                            &segment_named,
                            &requested_nodes,
                            resident_names,
                            Some(&mapped_expert_sources),
                        )
                    };
                    #[cfg(not(any(
                        all(feature = "metal-output-placement", target_os = "macos"),
                        all(
                            feature = "instrument",
                            feature = "metal",
                            target_os = "macos",
                            not(feature = "metal-output-placement")
                        )
                    )))]
                    let result = runtime.evaluate_segment(
                        program,
                        symbols,
                        &segment_named,
                        &requested_nodes,
                        resident_names,
                        Some(&mapped_expert_sources),
                    );
                    if layer == debug_layer
                        && std::env::var_os("PROXIMA_DEBUG_EXPERT_GATHER_PARITY").is_some()
                    {
                        let mut scratch = Vec::new();
                        let mut validated = None;
                        let expected = evaluate_quantized_named_exact_with_scratch_and_experts(
                            program,
                            symbols,
                            &segment_named,
                            &requested_nodes,
                            &mut scratch,
                            &mut validated,
                            Some(&mapped_expert_sources),
                        )?;
                        let debug_local_node = std::env::var("PROXIMA_DEBUG_EXPERT_GATHER_NODE")
                            .ok()
                            .and_then(|value| value.parse::<u32>().ok())
                            .map(NodeId);
                        let parity_node = debug_local_node
                            .and_then(|local| {
                                mapping.iter().find_map(|(original, mapped)| {
                                    (*mapped == local).then_some(*original)
                                })
                            })
                            .unwrap_or(segment_output);
                        let mapped_output = mapping
                            .get(&parity_node)
                            .copied()
                            .ok_or(InteropError::MissingEvaluatedNode { node: parity_node })?;
                        if let (Ok(actual), Some((expected_values, _))) =
                            (&result, expected.get(mapped_output))
                            && let Some((actual_values, _)) = actual.get(mapped_output)
                        {
                            let (index, maximum) = actual_values
                                .iter()
                                .zip(expected_values)
                                .enumerate()
                                .map(|(index, (actual, expected))| {
                                    (index, (actual - expected).abs())
                                })
                                .max_by(|left, right| left.1.total_cmp(&right.1))
                                .unwrap_or((0, 0.0));
                            eprintln!(
                                "qwen35 gather parity layer={layer} node={} original_op={:?} max_abs={maximum} index={index} metal={} cpu={}",
                                parity_node.0,
                                self.program
                                    .get(parity_node.0 as usize)
                                    .map(|operation| operation.name()),
                                actual_values.get(index).copied().unwrap_or_default(),
                                expected_values.get(index).copied().unwrap_or_default(),
                            );
                        }
                        if let Ok(actual) = &result {
                            if std::env::var_os("PROXIMA_DEBUG_EXPERT_GATHER_GRAPH").is_some() {
                                let graph_root =
                                    std::env::var("PROXIMA_DEBUG_EXPERT_GATHER_GRAPH_NODE")
                                        .ok()
                                        .and_then(|value| value.parse::<u32>().ok())
                                        .map(NodeId)
                                        .unwrap_or(parity_node);
                                let mut pending = vec![graph_root];
                                let mut visited = BTreeSet::new();
                                while let Some(node) = pending.pop() {
                                    if !visited.insert(node) {
                                        continue;
                                    }
                                    let Some(operation) = program.get(node.0 as usize) else {
                                        continue;
                                    };
                                    match operation {
                                        proxima_tensor::op::Op::Input { name, .. } => {
                                            eprintln!(
                                                "qwen35 gather graph input node={node:?} name={name:?}"
                                            );
                                        }
                                        proxima_tensor::op::Op::Elementwise {
                                            operands, ..
                                        } => {
                                            pending.extend(
                                                operands.iter().map(|(operand, _)| *operand),
                                            );
                                        }
                                        proxima_tensor::op::Op::Reduce(reduce) => {
                                            pending.push(reduce.operand)
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            for node in &requested_nodes {
                                let Some((actual_values, _)) = actual.get(*node) else {
                                    continue;
                                };
                                let Some((expected_values, _)) = expected.get(*node) else {
                                    continue;
                                };
                                let maximum = actual_values
                                    .iter()
                                    .zip(expected_values)
                                    .map(|(actual, expected)| (actual - expected).abs())
                                    .fold(0.0_f32, f32::max);
                                if maximum > 1.0e-3 {
                                    eprintln!(
                                        "qwen35 first gather divergence layer={layer} node={node:?} name={:?} op={:?} max_abs={maximum}",
                                        program[node.0 as usize].name(),
                                        program[node.0 as usize],
                                    );
                                    if let proxima_tensor::op::Op::Elementwise {
                                        operands, ..
                                    } = &program[node.0 as usize]
                                    {
                                        for (operand_index, (operand, _)) in
                                            operands.iter().enumerate()
                                        {
                                            if let (
                                                Some((actual_operand, _)),
                                                Some((expected_operand, _)),
                                            ) = (actual.get(*operand), expected.get(*operand))
                                            {
                                                let operand_maximum = actual_operand
                                                    .iter()
                                                    .zip(expected_operand)
                                                    .map(|(actual, expected)| {
                                                        (actual - expected).abs()
                                                    })
                                                    .fold(0.0_f32, f32::max);
                                                eprintln!(
                                                    "qwen35 gather operand node={operand:?} index={operand_index} max_abs={operand_maximum}"
                                                );
                                            }
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    #[cfg(feature = "instrument")]
                    {
                        gather_elapsed_us += segment_started.elapsed().as_micros() as u64;
                    }
                    if std::env::var_os("PROXIMA_DEBUG_QWEN35_SEGMENTS").is_some() {
                        eprintln!(
                            "qwen35 segment phase=gather layer={} elapsed_us={}",
                            layer,
                            segment_started.elapsed().as_micros()
                        );
                    }
                    result?
                };
                #[cfg(feature = "instrument")]
                {
                    segment_execution_count += 1;
                }
                if is_router {
                    let mapped_router = mapping.get(&diagnostic.router_logits).copied().ok_or(
                        InteropError::MissingEvaluatedNode {
                            node: diagnostic.router_logits,
                        },
                    )?;
                    let (router_logits, router_shape) =
                        evaluated
                            .get(mapped_router)
                            .ok_or(InteropError::MissingEvaluatedNode {
                                node: diagnostic.router_logits,
                            })?;
                    visit_qwen35moe_router_boundary(
                        layer,
                        position_offset,
                        router_logits,
                        router_shape,
                        self.architecture.expert_count as usize,
                        self.architecture.expert_used_count as usize,
                        &mut routed_experts,
                        expert_slab,
                        &mut before_gather,
                    )?;
                }
                for (mapped, original) in requested {
                    let (values, shape) = evaluated
                        .get(mapped)
                        .ok_or(InteropError::MissingEvaluatedNode { node: original })?;
                    if future_cuts.iter().any(|(node, _)| *node == original)
                        || future_gather_cuts.contains(&original)
                        || plan.prefix_carried_nodes.contains(&original)
                        || plan.global_cut_nodes.contains(&original)
                        || original == segment_output
                    {
                        carried.insert(original, (shape.to_vec(), values.to_vec()));
                    }
                    if outputs.contains(&original) {
                        results.insert(original, (shape.to_vec(), values.to_vec()));
                    }
                }
                let output_id = mapping.get(&segment_output).copied().ok_or(
                    InteropError::MissingEvaluatedNode {
                        node: segment_output,
                    },
                )?;
                if let Some((values, shape)) = evaluated.get(output_id) {
                    if !is_router && values.iter().any(|value| !value.is_finite()) {
                        let first_nonfinite = values
                            .iter()
                            .enumerate()
                            .find(|(_, value)| !value.is_finite())
                            .map(|(index, value)| (index, *value));
                        eprintln!(
                            "qwen35 nonfinite gather layer={layer} node={segment_output:?} first={first_nonfinite:?}"
                        );
                        return Err(InteropError::PreGatherExecutionUnsupported {
                            architecture: String::from("qwen35moe"),
                            reason: alloc::format!(
                                "layer {layer} expert gather produced a non-finite value"
                            ),
                        });
                    }
                    if std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some() {
                        let nan_count = values.iter().filter(|value| value.is_nan()).count();
                        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
                        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        eprintln!(
                            "qwen35 segment output layer={} phase={} node={:?} elements={} nan_count={} min={} max={} first={:?}",
                            layer,
                            if is_router { "router" } else { "gather" },
                            segment_output,
                            values.len(),
                            nan_count,
                            min,
                            max,
                            values.get(..values.len().min(4)).unwrap_or_default()
                        );
                    }
                    carried.insert(segment_output, (shape.to_vec(), values.to_vec()));
                }
            }
            #[cfg(unix)]
            if std::env::var("PROXIMA_EXPERT_SIDECAR_DISCARD_PER_LAYER")
                .ok()
                .as_deref()
                == Some("1")
                && let Some(sidecar) = &self.expert_sidecar
            {
                sidecar.discard_resident_pages()?;
            }
            // Sidecar payloads are an mmap, so leaving every low-codec page
            // resident turns a bounded device slab into an unbounded host
            // footprint over a long decode. Discard only the routes consumed
            // by this layer; `KEEP_PAGES` is an explicit diagnostic opt-out.
            #[cfg(all(feature = "metal", target_os = "macos"))]
            if std::env::var_os("PROXIMA_EXPERT_SIDECAR_KEEP_PAGES").is_none()
                && let Some(sidecar) = &self.expert_sidecar
            {
                let mut discarded = BTreeSet::new();
                for route in &routed_experts {
                    let address = crate::residency::ExpertAddress {
                        layer,
                        expert: route.expert,
                    };
                    if discarded.insert((address.layer, address.expert)) {
                        sidecar.discard_expert_low(address)?;
                    }
                }
            }
            #[cfg(all(feature = "metal", target_os = "macos"))]
            if std::env::var_os("PROXIMA_CHECKPOINT_DISCARD_PER_LAYER").is_some() {
                omega::discard_checkpoint_mmap_range(self.checkpoint_mapping).map_err(|error| {
                    InteropError::PreGatherExecutionUnsupported {
                        architecture: String::from("qwen35moe"),
                        reason: error.to_string(),
                    }
                })?;
            }
            let keep: BTreeSet<NodeId> = next_cuts.iter().map(|(node, _)| *node).collect();
            // A gather segment may carry an intermediate produced by the
            // preceding layer's gather rather than by its router. Retain
            // every such cut until its consumer layer instead of assuming
            // the next router cut list is complete.
            let future_gather_cuts = plan.layers[layer + 1..]
                .iter()
                .flat_map(|segments| segments.gather.1.iter().map(|(node, _)| *node));
            let keep: BTreeSet<NodeId> = keep.into_iter().chain(future_gather_cuts).collect();
            // Keep only the handful of graph inputs that every layer may
            // reference, plus the explicit next-segment cuts. Expert stack
            // inputs are excluded above and can never enter this carry set.
            carried.retain(|node, _| {
                plan.prefix_carried_nodes.contains(node)
                    || keep.contains(node)
                    || matches!(
                        self.program[node.0 as usize],
                        proxima_tensor::op::Op::Constant { .. }
                    )
            });
        }

        let (suffix_program, suffix_cuts, suffix_mapping) = &plan.suffix;
        let mut requested = BTreeMap::new();
        for &node in outputs {
            if let Some(mapped) = suffix_mapping.get(&node).copied() {
                requested.insert(mapped, node);
            }
        }
        let suffix_executed = !requested.is_empty();
        if suffix_executed {
            let mut suffix_named: Vec<(&str, QuantizedBlock<'_>)> = named
                .iter()
                .copied()
                .filter(|(name, _)| {
                    suffix_program
                        .iter()
                        .any(|operation| operation.name() == Some(*name))
                })
                .collect();
            for (node, name) in suffix_cuts {
                if suffix_named.iter().any(|(candidate, _)| *candidate == name) {
                    continue;
                }
                let (_, values) = carried.get(node).ok_or_else(|| {
                    InteropError::PreGatherExecutionUnsupported {
                        architecture: String::from("qwen35moe"),
                        reason: alloc::format!("suffix missing cut node {node:?} ({name})"),
                    }
                })?;
                suffix_named.push((name.as_str(), QuantizedBlock::Float32(values)));
            }
            let requested_nodes: Vec<NodeId> = requested.keys().copied().collect();
            let evaluated = runtime.evaluate_segment(
                &suffix_program,
                symbols,
                &suffix_named,
                &requested_nodes,
                resident_names,
                None,
            )?;
            #[cfg(feature = "instrument")]
            {
                segment_execution_count += 1;
            }
            for (mapped, original) in requested {
                let (values, shape) = evaluated
                    .get(mapped)
                    .ok_or(InteropError::MissingEvaluatedNode { node: original })?;
                results.insert(original, (shape.to_vec(), values.to_vec()));
            }
        }
        #[cfg(feature = "instrument")]
        {
            debug!(
                position_offset = position_offset as u64,
                layer_count = plan.layers.len() as u64,
                segment_execution_count,
                suffix_executed,
                router_elapsed_us,
                gather_elapsed_us,
                "qwen35moe pre-gather segment census recorded after requested outputs completed"
            );
            if std::env::var_os("PROXIMA_DEBUG_QWEN35_SEGMENTS").is_some() {
                eprintln!(
                    "qwen35 segment summary position={} layers={} segments={} suffix_executed={} router_elapsed_us={} gather_elapsed_us={}",
                    position_offset,
                    plan.layers.len(),
                    segment_execution_count,
                    suffix_executed,
                    router_elapsed_us,
                    gather_elapsed_us,
                );
            }
        }

        let ordered_results = outputs
            .iter()
            .map(|node| {
                results
                    .remove(node)
                    .map(|(shape, values)| (*node, shape, values))
                    .ok_or(InteropError::MissingEvaluatedNode { node: *node })
            })
            .collect::<Result<Vec<_>, _>>()?;
        #[cfg(unix)]
        if std::env::var("PROXIMA_EXPERT_SIDECAR_DISCARD")
            .ok()
            .as_deref()
            == Some("1")
            && let Some(sidecar) = &self.expert_sidecar
        {
            sidecar.discard_resident_pages()?;
        }
        Ok(Evaluated::from_parts(
            self.logits_root,
            ordered_results,
            None,
        ))
    }

    /// Graph-level producer boundaries for every qwen35moe layer, in layer
    /// order. Non-qwen35moe models return an empty slice.
    #[must_use]
    pub fn qwen35moe_layer_diagnostics(&self) -> &[crate::qwen35moe::Qwen35MoeLayerDiagnostics] {
        &self.qwen35moe_layer_diagnostics
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
        Self::load_with_registry(
            parsed,
            file_bytes,
            &crate::architecture::ArchitectureRegistry::with_builtin(),
        )
    }

    /// [`Self::load`] with the [`crate::architecture::ArchitectureRegistry`]
    /// caller-supplied rather than fixed to
    /// [`crate::architecture::ArchitectureRegistry::with_builtin`] --
    /// the seam a foreign `Architecture` (registered against its own
    /// registry, never against this crate's) loads a checkpoint through,
    /// the same way [`crate::architecture::ArchitectureRegistry::resolve`]'s
    /// own doc describes. [`Self::load`] is this call with the builtin
    /// table, unchanged.
    ///
    /// # Errors
    ///
    /// Same as [`Self::load`].
    pub fn load_with_registry(
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
        registry: &crate::architecture::ArchitectureRegistry,
    ) -> Result<Self, InteropError> {
        Self::load_inner(parsed, file_bytes, false, false, registry)
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
        Self::load_inner(
            parsed,
            file_bytes,
            paired_gate_up_reduce,
            false,
            &crate::architecture::ArchitectureRegistry::with_builtin(),
        )
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
        Self::load_inner(
            parsed,
            file_bytes,
            false,
            fused_qkv_reduce,
            &crate::architecture::ArchitectureRegistry::with_builtin(),
        )
    }

    fn load_inner(
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
        paired_gate_up_reduce: bool,
        fused_qkv_reduce: bool,
        registry: &crate::architecture::ArchitectureRegistry,
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
        // The common case resolves the checkpoint's own `general.architecture`
        // against the registered `Architecture` table and lets that impl's
        // own `bind` do everything `load_inner`'s qwen35/dense arms used to
        // assemble by hand -- see `crate::architecture`'s own module doc
        // for why this seam exists. `paired_gate_up_reduce`/`fused_qkv_reduce`
        // are per-call diagnostic knobs `Architecture::bind`'s fixed
        // signature does not carry (`crate::dense::DenseArch`'s own doc on
        // why), and (documented on both flag-carrying constructors) have
        // "no effect on a qwen35 checkpoint" -- so a qwen35 checkpoint
        // always takes this registry path regardless of either flag, and
        // only a non-qwen35 checkpoint with a flag set falls through to
        // the narrow inline path below.
        let general_architecture = crate::bind::metadata_str(parsed, "general.architecture")?;
        if std::env::var_os("PROXIMA_DEBUG_ARCH_ROUTE").is_some() {
            eprintln!(
                "architecture route value={general_architecture:?} qwen35moe={} flags=({}, {})",
                general_architecture == "qwen35moe",
                paired_gate_up_reduce,
                fused_qkv_reduce,
            );
        }
        if matches!(general_architecture, "qwen35" | "qwen35moe")
            || (!paired_gate_up_reduce && !fused_qkv_reduce)
        {
            let resolved = registry.resolve(parsed)?;
            let bound = resolved.bind(parsed, file_bytes)?;
            #[cfg(all(feature = "metal", target_os = "macos"))]
            let step_state = resolved.step_state(parsed)?;
            let vocab = proxima_tokenizer::gguf::vocab_from_metadata(parsed)?;
            // The single-range program is dense-Mistral-only
            // (`SingleRangeProgram`'s own field doc): never built for
            // qwen35's hybrid attention+state-space layers, and
            // `build_single_range_program` itself already turns away any
            // mixture-of-experts checkpoint.
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            let single_range = if resolved.name() == "qwen35" {
                None
            } else {
                let qk_norm = crate::bind::checkpoint_has_qk_norm(parsed);
                build_single_range_program(&bound.architecture, qk_norm)?
            };
            let bound = bound;
            let expert_slab =
                crate::bind::build_expert_slab(&bound.architecture, &bound.program, &bound.weights);
            return Self {
                expert_slab: std::sync::Mutex::new(expert_slab),
                expert_sidecar: None,
                weights: bound.weights,
                architecture: bound.architecture,
                architecture_impl: Some(resolved),
                #[cfg(all(feature = "metal", target_os = "macos"))]
                checkpoint_weight_bytes: crate::memory_fit::WeightClassBytes {
                    dense_bytes: dense_weight_bytes,
                    expert_bytes: expert_weight_bytes,
                    table_bytes: table_weight_bytes,
                    ssm_state_bytes: step_state.as_ref().map_or(0, |state| state.ssm_state_bytes),
                },
                vocab,
                program: bound.program,
                logits_root: bound.logits_root,
                hidden_root: bound.hidden_root,
                layer_roots: bound.layer_roots,
                qwen35moe_layer_diagnostics: bound.qwen35moe_layer_diagnostics,
                router_roots: bound.router_roots,
                moe_sites: bound.moe_sites,
                single_position_step: bound.single_position_step,
                model_name: crate::bind::metadata_str_opt(parsed, "general.name").map(String::from),
                checkpoint_bytes: file_bytes.len(),
                #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                single_range,
                checkpoint_mapping: file_bytes,
            }
            .validated();
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
        let (program, forward_roots, cache_roots, _layer_residuals, moe_sites) =
            mistral_cached_forward_program_with_experts_and_layer_taps(
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
                true,
            )?;
        let logits_root = forward_roots.logits;
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
        let expert_slab = crate::bind::build_expert_slab(&architecture, &program, &weights);
        Self {
            expert_slab: std::sync::Mutex::new(expert_slab),
            expert_sidecar: None,
            weights,
            architecture,
            architecture_impl: None,
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
            hidden_root: Some(forward_roots.hidden),
            layer_roots: cache_roots
                .into_iter()
                .map(Qwen35LayerRoots::Attention)
                .collect(),
            qwen35moe_layer_diagnostics: Vec::new(),
            router_roots: Vec::new(),
            moe_sites,
            single_position_step: false,
            model_name: crate::bind::metadata_str_opt(parsed, "general.name").map(String::from),
            checkpoint_bytes: file_bytes.len(),
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            single_range,
            checkpoint_mapping: file_bytes,
        }
        .validated()
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
        let (program, forward_roots, cache_roots, _layer_residuals, moe_sites) =
            mistral_cached_forward_program_with_experts_and_layer_taps(
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
                true,
            )?;
        let logits_root = forward_roots.logits;
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        let single_range = build_single_range_program(&architecture, false)?;
        let expert_slab = crate::bind::build_expert_slab(&architecture, &program, &weights);
        Self {
            expert_slab: std::sync::Mutex::new(expert_slab),
            expert_sidecar: None,
            weights,
            architecture,
            architecture_impl: None,
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
            hidden_root: Some(forward_roots.hidden),
            layer_roots: cache_roots
                .into_iter()
                .map(Qwen35LayerRoots::Attention)
                .collect(),
            qwen35moe_layer_diagnostics: Vec::new(),
            router_roots: Vec::new(),
            moe_sites,
            single_position_step: false,
            // safetensors carries no `general.name`-equivalent key this
            // crate reads (`Self::model_name`'s own doc).
            model_name: None,
            checkpoint_bytes: file_bytes.len(),
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            single_range,
            checkpoint_mapping: file_bytes,
        }
        .validated()
    }
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

pub enum NodeValuesSink<'sink> {
    Discard,
    Collect {
        nodes: &'sink [NodeId],
        steps: &'sink mut Vec<Vec<Vec<f32>>>,
    },
}

impl NodeValuesSink<'_> {
    pub fn nodes(&self) -> &[NodeId] {
        match self {
            Self::Discard => &[],
            Self::Collect { nodes, .. } => nodes,
        }
    }

    pub fn observe(&mut self, evaluated: &Evaluated) -> Result<(), InteropError> {
        let Self::Collect { nodes, steps } = self else {
            return Ok(());
        };
        let mut values = Vec::with_capacity(nodes.len());
        for &node in *nodes {
            let (data, _) = evaluated
                .get(node)
                .ok_or(InteropError::MissingEvaluatedNode { node })?;
            values.push(data.to_vec());
        }
        steps.push(values);
        Ok(())
    }
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
#[derive(Clone)]
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

    /// # Errors
    ///
    /// [`InteropError::CacheScratchShapeMismatch`] when `source` (this
    /// layer's real, unpadded cache) holds more elements in a leaf than
    /// `shape` sized that leaf's own scratch buffer to -- `shape`'s row
    /// widths come from the bound program's own declared cache-leaf
    /// extents ([`layer_pad_row_widths`]'s own doc), so this only fires
    /// when a foreign bind's program under-declares a leaf its own
    /// [`LayerCache::append`] then over-fills, not on any checkpoint whose
    /// program and cache stay in agreement.
    fn fill(
        &mut self,
        source: &LayerCache,
        shape: &KvPadShape,
        layer: usize,
    ) -> Result<(), InteropError> {
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
        copy_into_padded(&mut self.k_even, &source.k_even, layer, "k_even")?;
        copy_into_padded(&mut self.k_odd, &source.k_odd, layer, "k_odd")?;
        copy_into_padded(&mut self.v, &source.v, layer, "v")?;
        Ok(())
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
            (v_name, QuantizedBlock::Float32(&self.v[..shape.v_len()])),
        ]
    }
}

/// [`KvPadScratch`]'s own row-width parameters, grouped into one reference
/// rather than two positional `usize`s -- every [`KvPadScratch::fill`]/
/// [`KvPadScratch::named_blocks`] call site already computes both together
/// from `kv_bound_extent` and this layer's own [`LayerPadRowWidths`], so
/// one reference says what was already true by convention. `even_odd_row`/
/// `v_row` are each `kv_heads * width` for their own leaf, read back off
/// this layer's declared `Op::Input` shape by [`cache_leaf_row_elements`]
/// -- never derived from `ModelArchitecture` scalars a foreign bind may
/// leave zero/unset (that doc's own paragraph on why).
struct KvPadShape {
    bound_extent: usize,
    even_odd_row: usize,
    v_row: usize,
}

impl KvPadShape {
    fn even_odd_len(&self) -> usize {
        self.bound_extent * self.even_odd_row
    }

    fn v_len(&self) -> usize {
        self.bound_extent * self.v_row
    }
}

/// Copies `source` into `dest`'s own leading rows -- the shared bounds
/// check every [`KvPadScratch::fill`]/[`Qwen35DenseAttentionPadScratch::fill`]
/// leaf copy needs: `dest` was just resized to (at least) `shape`'s own
/// declared row width, so `source` (this layer's real, unpadded cache)
/// fitting inside it is the invariant the whole pad-scratch mechanism
/// depends on. Previously an unchecked `copy_from_slice`, panicking with
/// "range end index out of range" the moment a foreign bind's declared
/// shape undercounted a leaf's true width; now a named, typed error.
///
/// # Errors
///
/// [`InteropError::CacheScratchShapeMismatch`] when `source.len() >
/// dest.len()`.
fn copy_into_padded(
    dest: &mut [f32],
    source: &[f32],
    layer: usize,
    leaf: &'static str,
) -> Result<(), InteropError> {
    let expected = dest.len();
    let found = source.len();
    if found > expected {
        return Err(InteropError::CacheScratchShapeMismatch {
            layer,
            leaf,
            expected,
            found,
        });
    }
    dest[..found].copy_from_slice(source);
    Ok(())
}

/// [`LayerCache`]'s 4-wide counterpart for a
/// [`Qwen35LayerRoots::DenseAttention`] layer -- this checkpoint's own
/// partial-rotary gap (`proxima_tensor::spec::append_qwen35_dense_attention_layer`'s
/// own doc) needs a third K component (`k_pass`, the untouched
/// `rotary_dim..attn_head_dim` remainder) alongside the rotated
/// `k_first`/`k_second` halves [`LayerCache`]'s `k_even`/`k_odd` already
/// name for the plain single-section-RoPE checkpoints.
#[derive(Clone)]
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
    even_odd_row: usize,
    pass_row: usize,
    v_row: usize,
}

impl Qwen35DenseAttentionPadShape {
    fn even_odd_len(&self) -> usize {
        self.bound_extent * self.even_odd_row
    }

    fn pass_len(&self) -> usize {
        self.bound_extent * self.pass_row
    }

    fn v_len(&self) -> usize {
        self.bound_extent * self.v_row
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

    /// # Errors
    ///
    /// [`InteropError::CacheScratchShapeMismatch`] when `source` (this
    /// layer's real, unpadded cache) holds more elements in a leaf than
    /// `shape` sized that leaf's own scratch buffer to -- see
    /// [`KvPadScratch::fill`]'s own doc for why this can only fire on a
    /// bind whose program under-declares a leaf its own
    /// [`Qwen35DenseAttentionCache::append`] then over-fills. This is the
    /// exact defect measured on the real `qwen3.6:35b-a3b` checkpoint: a
    /// foreign bind's [`crate::architecture::Architecture::step_state`]
    /// left `attn_head_dim` at the trait default (`Ok(None)`), which used
    /// to size `v`'s scratch to `0` while `source.v` held real data --
    /// this fill no longer reads `attn_head_dim` at all, so that failure
    /// mode is gone; the check stays as the general safety net.
    fn fill(
        &mut self,
        source: &Qwen35DenseAttentionCache,
        shape: &Qwen35DenseAttentionPadShape,
        layer: usize,
    ) -> Result<(), InteropError> {
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
        copy_into_padded(&mut self.k_first, &source.k_first, layer, "k_first")?;
        copy_into_padded(&mut self.k_second, &source.k_second, layer, "k_second")?;
        copy_into_padded(&mut self.k_pass, &source.k_pass, layer, "k_pass")?;
        copy_into_padded(&mut self.v, &source.v, layer, "v")?;
        Ok(())
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
/// kernel's own left context, `conv_history_len` elements total, oldest row
/// dropped as each new one is appended) rather than [`LayerCache`]'s
/// unbounded grow-forever history; `state` is the gated DeltaNet recurrent
/// state, fully replaced every step (never appended to) because the mixer
/// already folds every past position into it. Both lengths come from
/// [`LayerPadRowWidths::Ssm`] -- the program's own declared
/// `ssm_cache.{layer}.conv_history`/`.state` `Op::Input` shapes
/// ([`cache_leaf_total_elements`]'s own doc on why this, not
/// [`crate::architecture::Architecture::step_state`], is authoritative).
#[derive(Clone)]
struct SsmLayerCache {
    conv_history: Vec<f32>,
    state: Vec<f32>,
}

impl SsmLayerCache {
    fn new(conv_history_len: usize, state_len: usize) -> Self {
        Self {
            conv_history: alloc::vec![0.0f32; conv_history_len],
            state: alloc::vec![0.0f32; state_len],
        }
    }

    /// `qkv_mixed_new` is this step's own `new_count`-many freshly computed
    /// `qkv_mixed` rows; `state_new` is the mixer's full replacement state.
    /// Keeps only the most recent `conv_history_len` elements -- older rows
    /// fall out of the causal conv1d kernel's left context and are never
    /// read again.
    fn advance(&mut self, qkv_mixed_new: &[f32], state_new: &[f32], conv_history_len: usize) {
        self.advance_conv_history(qkv_mixed_new, conv_history_len);
        if self.state.len() == state_new.len() {
            self.state.copy_from_slice(state_new);
        } else {
            self.state.clear();
            self.state.extend_from_slice(state_new);
        }
    }

    fn advance_conv_history(&mut self, qkv_mixed_new: &[f32], conv_history_len: usize) {
        self.conv_history.extend_from_slice(qkv_mixed_new);
        let drop = self.conv_history.len().saturating_sub(conv_history_len);
        if drop != 0 {
            let retained = self.conv_history.len() - drop;
            self.conv_history.copy_within(drop.., 0);
            self.conv_history.truncate(retained);
        }
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
#[derive(Clone)]
enum LayerCacheState {
    Attention(LayerCache),
    DenseAttention(Qwen35DenseAttentionCache),
    Ssm(SsmLayerCache),
}

/// Diagnostic-only (history-carry bisection step 3): `(element_count,
/// abs_sum)` over every `f32` this layer's cache currently holds -- a cheap
/// stand-in for a real hash that still catches the two failure shapes this
/// bisection is looking for: `element_count` frozen step-over-step means the
/// cache never grew (state leaves fed zeros or the wrong buffer);
/// `abs_sum` frozen while `element_count` grows means new rows are being
/// appended but they are all-zero.
#[cfg(feature = "instrument")]
fn layer_cache_checksum(cache: &LayerCacheState) -> (usize, f64) {
    let sum_abs =
        |values: &[f32]| -> f64 { values.iter().map(|value| f64::from(value.abs())).sum() };
    match cache {
        LayerCacheState::Attention(cache) => (
            cache.k_even.len() + cache.k_odd.len() + cache.v.len(),
            sum_abs(&cache.k_even) + sum_abs(&cache.k_odd) + sum_abs(&cache.v),
        ),
        LayerCacheState::DenseAttention(cache) => (
            cache.k_first.len() + cache.k_second.len() + cache.k_pass.len() + cache.v.len(),
            sum_abs(&cache.k_first)
                + sum_abs(&cache.k_second)
                + sum_abs(&cache.k_pass)
                + sum_abs(&cache.v),
        ),
        LayerCacheState::Ssm(cache) => (
            cache.conv_history.len() + cache.state.len(),
            sum_abs(&cache.conv_history) + sum_abs(&cache.state),
        ),
    }
}

/// A cached prefix: the token ids [`LoadedModel::prefill_prefix`] ran one
/// forward pass over, and the per-layer `LayerCacheState` that pass left
/// behind -- the SAME `(ids, layer_caches, cached_len)` triple
/// `run_decode_loop_observed_seeded` already threads through
/// its own two-range decode loop as local bindings on every call, kept
/// alive across calls instead of dropped at function return. This is not a
/// new cache shape: composing it back in
/// ([`LoadedModel::generate_from_prefix`]) is exactly step 0 of
/// [`LoadedModel::generate_with_serving_config`]'s own loop, given a
/// nonzero `cached_len` and a suffix-only `next_ids` to start from rather
/// than the whole prompt at `cached_len == 0`.
///
/// Plain host `Vec<f32>` buffers throughout (`LayerCache`'s own field
/// list) -- the two-range decode path never registers a named,
/// device-resident buffer for the KV cache the way
/// `LoadedModel::resident_names`'s STATIC weights do (only re-uploads it
/// as an ordinary named block every step, `run_decode_loop_observed_seeded`'s
/// own `named_blocks.extend(kv_pad_scratch...)` call). Releasing this state
/// is therefore exactly Rust's own default `Drop` for a `Vec` -- there is
/// no device identity to unregister the way [`LoadedModel`]'s own `Drop`
/// must, so this type carries none.
pub struct PrefixState {
    ids: Vec<u32>,
    layer_caches: Vec<LayerCacheState>,
    cached_len: usize,
}

impl PrefixState {
    /// The number of prompt/generated tokens this state's own KV cache
    /// covers -- [`LoadedModel::generate_from_prefix`]'s own resume point.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cached_len
    }

    /// `true` for a prefix that cached zero tokens -- never actually
    /// produced by [`LoadedModel::prefill_prefix`] against a non-empty
    /// prompt, but a real state a caller could still reach by prefilling
    /// an empty string.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cached_len == 0
    }
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

/// Which of the three per-layer cache shapes a layer's `Op::Input` leaves
/// actually declare, at `layer` -- [`LayerCacheNames`]/[`LayerCacheState`]
/// are now built FROM this, not from [`Qwen35LayerRoots`]'s own
/// discriminant. A foreign [`crate::architecture::Architecture::bind`] can
/// tag that enum inconsistently with the ops it actually emitted (copy a
/// [`Qwen35DenseAttentionRoots`] tuple into the wrong variant, drop the
/// `k_pass` leaf); the program's own declared leaf names cannot lie about
/// what the decode loop must feed, so they are the single source of truth
/// this type is derived from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeclaredCacheKind {
    Attention,
    DenseAttention,
    Ssm,
}

impl DeclaredCacheKind {
    fn label(self) -> &'static str {
        match self {
            Self::Attention => "kv_cache.{layer}.{k_even,k_odd,v}",
            Self::DenseAttention => "kv_cache.{layer}.{k_first,k_second,k_pass,v}",
            Self::Ssm => "ssm_cache.{layer}.{conv_history,state}",
        }
    }

    /// The `Op::Input` leaf-name templates a program must declare for a
    /// layer bound to this kind -- the exact set
    /// [`declared_layer_cache_names_and_widths`] fills in `{layer}` from,
    /// surfaced verbatim in [`InteropError::LayerCacheLeavesMissing`] so
    /// the error names precisely what the architecture's `bind` failed to
    /// emit.
    fn expected_leaf_templates(self) -> &'static [&'static str] {
        match self {
            Self::Attention => &[
                "kv_cache.{layer}.k_even",
                "kv_cache.{layer}.k_odd",
                "kv_cache.{layer}.v",
            ],
            Self::DenseAttention => &[
                "kv_cache.{layer}.k_first",
                "kv_cache.{layer}.k_second",
                "kv_cache.{layer}.k_pass",
                "kv_cache.{layer}.v",
            ],
            Self::Ssm => &["ssm_cache.{layer}.conv_history", "ssm_cache.{layer}.state"],
        }
    }
}

/// Probes `program_input_names` (every `Op::Input` name this call's own
/// program declares, collected once) for `layer`'s cache leaves -- checks
/// the widest/most specific shape (`DenseAttention`'s 4-wide `k_first`)
/// before the narrower ones so a layer that happens to declare both would
/// still resolve unambiguously. `None` when `layer` declares no recognized
/// cache leaf at all (a non-cached layer, or a name this function's own
/// three prefixes do not cover).
fn declared_cache_kind(
    program_input_names: &BTreeSet<&str>,
    layer: usize,
) -> Option<DeclaredCacheKind> {
    if program_input_names.contains(alloc::format!("kv_cache.{layer}.k_first").as_str()) {
        Some(DeclaredCacheKind::DenseAttention)
    } else if program_input_names.contains(alloc::format!("kv_cache.{layer}.k_even").as_str()) {
        Some(DeclaredCacheKind::Attention)
    } else if program_input_names
        .contains(alloc::format!("ssm_cache.{layer}.conv_history").as_str())
    {
        Some(DeclaredCacheKind::Ssm)
    } else {
        None
    }
}

/// [`Qwen35LayerRoots`]'s own discriminant, read back as a
/// [`DeclaredCacheKind`] so it can be compared against
/// [`declared_cache_kind`]'s program-derived answer for the same layer.
fn bound_cache_kind(roots: &Qwen35LayerRoots) -> DeclaredCacheKind {
    match roots {
        Qwen35LayerRoots::Attention(_) => DeclaredCacheKind::Attention,
        Qwen35LayerRoots::DenseAttention(_) => DeclaredCacheKind::DenseAttention,
        Qwen35LayerRoots::Ssm { .. } => DeclaredCacheKind::Ssm,
    }
}

/// `name`'s own declared [`Op::Input`] shape, read back out of the program
/// that emitted it, collapsed to the flat element count one cached
/// position occupies -- every `kv_cache.{layer}.*` leaf is
/// `[Extent::Symbolic(KV_BOUND), heads, width]`
/// (`proxima_tensor::spec`'s `append_qwen35_dense_attention_layer`/
/// `append_mistral_cached_layer` own `input_leaf` calls for these exact
/// names), so the row width [`KvPadShape`]/[`Qwen35DenseAttentionPadShape`]
/// need is the PRODUCT of every extent after the leading symbolic
/// bound-extent slot, not a single dimension. This is the single source of
/// truth those two shapes size their scratch buffers from -- never
/// `ModelArchitecture`/[`crate::architecture::Architecture::step_state`]
/// scalars a foreign bind may leave zero or unset -- the real defect this
/// function replaces: `LoadedModel` used to carry a single model-wide
/// `qwen35_attn_head_dim: Option<u32>`, read from a trait method whose
/// default impl is `Ok(None)`, silently sizing every `DenseAttention`
/// layer's `v`/`k_pass` scratch to zero on any foreign
/// [`crate::architecture::Architecture`] that never overrides it.
///
/// `None` when `name` is not declared at all, or when the program declared
/// it with an unexpected shape (fewer than two dimensions, or a second
/// symbolic extent) -- [`layer_pad_row_widths`] reads either case as row
/// width `0`, which the `fill`-time [`crate::error::InteropError::CacheScratchShapeMismatch`]
/// check then catches as soon as a real, nonzero-length cache actually
/// needs to copy into it.
fn cache_leaf_row_elements(program: &[Op], name: &str) -> Option<usize> {
    let shape = program.iter().find_map(|op| match op {
        Op::Input {
            name: Some(leaf_name),
            shape,
            ..
        } if leaf_name == name => Some(shape.as_slice()),
        _ => None,
    })?;
    let (_bound_extent, row_dims) = shape.split_first()?;
    row_dims
        .iter()
        .try_fold(1usize, |product, extent| match extent {
            Extent::Static(value) => Some(product * (*value as usize)),
            Extent::Symbolic(_) => None,
        })
}

/// `name`'s own declared [`Op::Input`] shape, collapsed to its flat total
/// element count -- unlike [`cache_leaf_row_elements`], every dimension
/// counts (an SSM cache leaf like `ssm_cache.{layer}.conv_history`,
/// `[Static(d_conv-1), Static(qkv_dim)]`, has no growing bound-extent slot
/// the way a KV cache leaf does: it is a fixed-size rolling window from the
/// first decode step onward, so the leading dim is real window depth, not
/// something to skip). This is the single source of truth
/// [`layer_pad_row_widths`]'s own `Ssm` arm sizes
/// [`SsmLayerCache::new`]/[`SsmLayerCache::advance`] from -- never
/// [`crate::architecture::Architecture::step_state`]'s `ssm_shape`, whose
/// default impl a foreign architecture leaves `None` (the real defect this
/// function replaces, the `Ssm` sibling of [`cache_leaf_row_elements`]'s own
/// doc on the `DenseAttention` case).
///
/// `None` when `name` is not declared at all, or when the program declared
/// it with any symbolic extent -- [`layer_pad_row_widths`] reads either case
/// as `0`, caught by the same `fill`-time bounds check every other leaf
/// shape mismatch is.
fn cache_leaf_total_elements(program: &[Op], name: &str) -> Option<usize> {
    let shape = program.iter().find_map(|op| match op {
        Op::Input {
            name: Some(leaf_name),
            shape,
            ..
        } if leaf_name == name => Some(shape.as_slice()),
        _ => None,
    })?;
    shape
        .iter()
        .try_fold(1usize, |product, extent| match extent {
            Extent::Static(value) => Some(product * (*value as usize)),
            Extent::Symbolic(_) => None,
        })
}

/// [`KvPadShape`]/[`Qwen35DenseAttentionPadShape`]'s own row widths for one
/// layer, read once (`self.program` never changes for the lifetime of a
/// decode call) rather than re-derived from architecture scalars every
/// step -- see [`cache_leaf_row_elements`]'s own doc for why this is the
/// authoritative source.
enum LayerPadRowWidths {
    Attention {
        even_odd_row: usize,
        v_row: usize,
    },
    DenseAttention {
        even_odd_row: usize,
        pass_row: usize,
        v_row: usize,
    },
    /// [`SsmLayerCache::new`]/[`SsmLayerCache::advance`]'s own initial and
    /// steady-state window sizes -- the flat element count of
    /// `ssm_cache.{layer}.conv_history`/`.state` as the program itself
    /// declared them ([`cache_leaf_total_elements`]), never
    /// [`crate::architecture::Architecture::step_state`]'s `ssm_shape`.
    Ssm {
        conv_history_len: usize,
        state_len: usize,
    },
}

/// Builds [`LayerPadRowWidths`] for one layer from its own
/// [`LayerCacheNames`] leaf names, looking each one's declared shape up in
/// `program`. A leaf [`cache_leaf_row_elements`] cannot resolve (not
/// declared, or an unexpected shape) reads as row width `0` here --
/// deliberately, not a setup-time error: `0` reproduces exactly the
/// starting state a genuinely absent leaf already left this scratch buffer
/// in before this function existed, so the SAME `fill`-time bounds check
/// ([`InteropError::CacheScratchShapeMismatch`]) catches it, at the point
/// the mismatch actually matters, instead of two divergent error paths for
/// what is the same defect.
fn layer_pad_row_widths(program: &[Op], names: &LayerCacheNames) -> LayerPadRowWidths {
    match names {
        LayerCacheNames::Attention { k_even, v, .. } => LayerPadRowWidths::Attention {
            even_odd_row: cache_leaf_row_elements(program, k_even).unwrap_or(0),
            v_row: cache_leaf_row_elements(program, v).unwrap_or(0),
        },
        LayerCacheNames::DenseAttention {
            k_first, k_pass, v, ..
        } => LayerPadRowWidths::DenseAttention {
            even_odd_row: cache_leaf_row_elements(program, k_first).unwrap_or(0),
            pass_row: cache_leaf_row_elements(program, k_pass).unwrap_or(0),
            v_row: cache_leaf_row_elements(program, v).unwrap_or(0),
        },
        LayerCacheNames::Ssm {
            conv_history,
            state,
        } => LayerPadRowWidths::Ssm {
            conv_history_len: cache_leaf_total_elements(program, conv_history).unwrap_or(0),
            state_len: cache_leaf_total_elements(program, state).unwrap_or(0),
        },
    }
}

/// One step's KV-cache `Op::Input` leaves, named and padded off whatever
/// [`LayerCacheNames`]/[`LayerCacheState`]/[`LayerPadRowWidths`] this call's
/// own layers declared -- the ONE place either
/// [`LoadedModel::run_decode_loop_observed_seeded`] (a growing cache,
/// `kv_pad_scratch` reused across steps) or
/// [`LoadedModel::forward_node_values_on_backend`] (a fresh, empty cache,
/// one shot) turns cache state into named blocks, so a foreign
/// architecture's own leaf names (`k_first`/`k_second`/`k_pass` in place of
/// `k_even`/`k_odd`) are fed identically by both callers. Two passes over
/// the same `layer`/`cache` pairing, not one interleaved pass: see this
/// function's own former call-site comment (now here) on why
/// `KvPadScratch::named_blocks`'s borrow of `kv_pad_scratch[layer]` forces
/// fill-then-emit rather than an interleaved loop.
///
/// # Errors
///
/// Whatever [`KvPadScratch::fill`]/[`Qwen35DenseAttentionPadScratch::fill`]
/// can fail with.
fn push_kv_named_blocks<'call>(
    cache_names: &'call [LayerCacheNames],
    layer_caches: &'call [LayerCacheState],
    layer_row_widths: &[LayerPadRowWidths],
    kv_bound_extent: usize,
    kv_pad_scratch: &'call mut [KvPadScratch],
    qwen35_dense_pad_scratch: &'call mut [Qwen35DenseAttentionPadScratch],
    named_blocks: &mut Vec<(&'call str, QuantizedBlock<'call>)>,
) -> Result<(), InteropError> {
    for (layer, cache) in layer_caches.iter().enumerate() {
        match (cache, &layer_row_widths[layer]) {
            (
                LayerCacheState::Attention(cache),
                LayerPadRowWidths::Attention {
                    even_odd_row,
                    v_row,
                },
            ) => {
                let shape = KvPadShape {
                    bound_extent: kv_bound_extent,
                    even_odd_row: *even_odd_row,
                    v_row: *v_row,
                };
                kv_pad_scratch[layer].fill(cache, &shape, layer)?;
            }
            (
                LayerCacheState::DenseAttention(cache),
                LayerPadRowWidths::DenseAttention {
                    even_odd_row,
                    pass_row,
                    v_row,
                },
            ) => {
                let shape = Qwen35DenseAttentionPadShape {
                    bound_extent: kv_bound_extent,
                    even_odd_row: *even_odd_row,
                    pass_row: *pass_row,
                    v_row: *v_row,
                };
                qwen35_dense_pad_scratch[layer].fill(cache, &shape, layer)?;
            }
            (LayerCacheState::Ssm(_), LayerPadRowWidths::Ssm { .. }) => {}
            _ => unreachable!(
                "layer_row_widths built from the same cache_names as layer_caches, in lockstep"
            ),
        }
    }
    for (layer, names) in cache_names.iter().enumerate() {
        match (names, &layer_caches[layer], &layer_row_widths[layer]) {
            (
                LayerCacheNames::Attention { k_even, k_odd, v },
                LayerCacheState::Attention(_),
                LayerPadRowWidths::Attention {
                    even_odd_row,
                    v_row,
                },
            ) => {
                let shape = KvPadShape {
                    bound_extent: kv_bound_extent,
                    even_odd_row: *even_odd_row,
                    v_row: *v_row,
                };
                named_blocks.extend(kv_pad_scratch[layer].named_blocks(k_even, k_odd, v, &shape));
            }
            (
                LayerCacheNames::DenseAttention {
                    k_first,
                    k_second,
                    k_pass,
                    v,
                },
                LayerCacheState::DenseAttention(_),
                LayerPadRowWidths::DenseAttention {
                    even_odd_row,
                    pass_row,
                    v_row,
                },
            ) => {
                let shape = Qwen35DenseAttentionPadShape {
                    bound_extent: kv_bound_extent,
                    even_odd_row: *even_odd_row,
                    pass_row: *pass_row,
                    v_row: *v_row,
                };
                named_blocks.extend(
                    qwen35_dense_pad_scratch[layer]
                        .named_blocks(k_first, k_second, k_pass, v, &shape),
                );
            }
            (
                LayerCacheNames::Ssm {
                    conv_history,
                    state,
                },
                LayerCacheState::Ssm(cache),
                LayerPadRowWidths::Ssm { .. },
            ) => {
                named_blocks.extend(cache.named_blocks(conv_history, state));
            }
            _ => unreachable!(
                "cache_names/layer_caches/layer_row_widths built from the same layer_roots, in lockstep"
            ),
        }
    }
    Ok(())
}

/// Every per-call input the cached forward program needs beyond the model
/// weights and the growing key/value cache: `ids_i32`/RoPE `cos`/`sin` for
/// only the `new` positions this call introduces, at their true absolute
/// angle (`start_position`, not 0 -- a generated token's position is
/// `cached_len`, never the start of the sequence), plus the
/// reduce-broadcast `eps` vector sized to match.
struct PositionInputs {
    ids_i32: Vec<i32>,
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
    _qwen35_mrope: bool,
) -> PositionInputs {
    let new_count = new_ids.len();
    let pairs = head_dim as usize / 2;
    let ids_i32: Vec<i32> = new_ids.iter().map(|&id| id as i32).collect();
    let epsilon = alloc::vec![rms_epsilon; new_count];

    let mut cos = alloc::vec![0.0f32; new_count * pairs];
    let mut sin = alloc::vec![0.0f32; new_count * pairs];
    for offset in 0..new_count {
        let position = (start_position + offset) as f32;
        for pair in 0..pairs {
            // Qwen3.6 text positions use three MRoPE sections (11, 11, 10
            // pairs). Each section restarts its local frequency index; the
            // graph still consumes one flat table, so only angle generation
            // changes here.
            let frequency_pair = pair;
            let theta =
                position * rope_freq_base.powf(-((2 * frequency_pair) as f32) / (head_dim as f32));
            cos[offset * pairs + pair] = theta.cos();
            sin[offset * pairs + pair] = theta.sin();
        }
    }

    PositionInputs {
        ids_i32,
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

#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
struct SegmentMetalBindings<'buffers, 'source> {
    input_placements: &'buffers [(NodeId, &'buffers PlacedBuffer, usize)],
    output_placements: &'buffers [(NodeId, &'buffers PlacedBuffer, usize)],
    expert_sources:
        &'buffers alloc::collections::BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'source>>,
}

#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
struct Qwen35SsmPlacement<'buffers> {
    input_nodes: &'buffers [Option<NodeId>],
    buffers: &'buffers [Option<(PlacedBuffer, PlacedBuffer)>],
    maximum_layer: Option<usize>,
    use_second_as_input: bool,
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
    /// Plans for the stable pre-gather router/gather partitions. The segment
    /// programs reuse node IDs across layers, so this cache is keyed by the
    /// partition's address and shape rather than the ordinary decode key.
    segment_plans: alloc::collections::BTreeMap<(usize, usize, usize), Plan>,
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
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    placed_segment_plans: alloc::collections::BTreeMap<(usize, usize, usize), omega::metal::Plan>,
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
            segment_plans: alloc::collections::BTreeMap::new(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            math_mode: config.math_mode,
            numeric_policy: config.numeric_policy,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            dispatch_type: config.dispatch_type,
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            placed_plans: alloc::collections::BTreeMap::new(),
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            placed_segment_plans: alloc::collections::BTreeMap::new(),
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

    #[cfg(feature = "metal")]
    fn uses_gpu(&self) -> bool {
        matches!(self.engine, Engine::Gpu)
    }

    /// `resident_names` -- the caller's own model-weight names, fixed for
    /// the whole [`LoadedModel::generate_with_serving_config`] call -- is
    /// handed to [`mark_resident`] exactly once per distinct [`Plan`] (right
    /// after it is built, never on a cache hit, since a hit reuses the SAME
    /// `Plan` object that was already marked). See `omega::metal::Plan::mark_resident`'s
    /// own doc for why this needs a name set the tensor program itself
    /// cannot derive.
    // A modified expert table crosses the same backend boundary on every
    // engine. CPU consumes it; a GPU without expert-source bindings returns
    // a typed error instead of silently reading the original full stack.
    fn evaluate(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
        expert_sources: Option<
            &alloc::collections::BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'_>>,
        >,
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
                    // The backend-polymorphic executor historically opens a
                    // serial Metal encoder. Its stable-buffer implementation
                    // must preserve that ordering: the concurrent hazard
                    // schedule is proven only for the placed single-range
                    // program, not hybrid recurrent graphs.
                    set_dispatch_type(&mut plan, omega::metal::DispatchType::Serial);
                }
                Ok(plan)
            },
        )?;
        Ok(execute_plan_named_with_expert_sources(
            plan,
            named,
            expert_sources,
        )?)
    }

    /// Evaluates one graph partition without allowing the ordinary
    /// shape-only decode-plan cache to alias a different partition having
    /// the same `(new_count, kv_bound_extent)` pair. Routed execution uses
    /// this for the router and gather programs surrounding one layer.
    pub(crate) fn evaluate_segment(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
        expert_sources: Option<
            &alloc::collections::BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'_>>,
        >,
    ) -> Result<Evaluated, InteropError> {
        let host_timing = std::env::var_os("PROXIMA_DEBUG_SEGMENT_HOST").is_some();
        let resolve_started = std::time::Instant::now();
        let program_key = program.as_ptr() as usize;
        let new_count = symbols.first().copied().unwrap_or_default() as usize;
        let kv_bound_extent = symbols.get(1).copied().unwrap_or_default() as usize;
        let exact_activations = self.exact_activations;
        let plan = Self::resolve_segment_plan(
            &mut self.segment_plans,
            &mut self.plan_hits,
            &mut self.plan_misses,
            (program_key, new_count, kv_bound_extent),
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
                    set_dispatch_type(&mut plan, omega::metal::DispatchType::Serial);
                }
                Ok(plan)
            },
        )?;
        let resolve_elapsed_us = resolve_started.elapsed().as_micros();
        let execute_started = std::time::Instant::now();
        let result = execute_plan_named_with_expert_sources(plan, named, expert_sources)
            .map_err(InteropError::from);
        if host_timing {
            eprintln!(
                "qwen35 segment host resolve_us={} execute_us={} plan_hits={} plan_misses={}",
                resolve_elapsed_us,
                execute_started.elapsed().as_micros(),
                self.plan_hits,
                self.plan_misses,
            );
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                let stage = metal_stage_totals();
                eprintln!(
                    "qwen35 segment metal prepare_ms={:.3} emit_ms={:.3} pipeline_lookup_ms={:.3} op_setup_ms={:.3} gpu_exec_ms={:.3} encode_dispatch_ms={:.3} readback_ms={:.3} block_upload_ms={:.3}",
                    ticks_to_nanos(stage.prepare_ticks) as f64 / 1_000_000.0,
                    ticks_to_nanos(stage.emit_ticks) as f64 / 1_000_000.0,
                    ticks_to_nanos(stage.pipeline_lookup_ticks) as f64 / 1_000_000.0,
                    ticks_to_nanos(stage.op_setup_ticks) as f64 / 1_000_000.0,
                    ticks_to_nanos(stage.gpu_exec_ticks) as f64 / 1_000_000.0,
                    ticks_to_nanos(stage.encode_dispatch_ticks) as f64 / 1_000_000.0,
                    ticks_to_nanos(stage.readback_ticks) as f64 / 1_000_000.0,
                    ticks_to_nanos(stage.block_upload_ticks) as f64 / 1_000_000.0,
                );
            }
        }
        result
    }

    /// Diagnostic twin of [`Self::evaluate_segment`] for a routed gather.
    /// It resolves the identical cached segment plan and preserves its
    /// per-expert source substitutions, changing only command-buffer
    /// granularity so the caller can attribute GPU time to bound ops.
    #[cfg(all(feature = "instrument", target_os = "macos"))]
    fn evaluate_segment_op_timed(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
        expert_sources: &alloc::collections::BTreeMap<
            NodeId,
            proxima_tensor::cpu::ExpertSource<'_>,
        >,
    ) -> Result<(Evaluated, Vec<OpGpuTiming>), InteropError> {
        let program_key = program.as_ptr() as usize;
        let new_count = symbols.first().copied().unwrap_or_default() as usize;
        let kv_bound_extent = symbols.get(1).copied().unwrap_or_default() as usize;
        let exact_activations = self.exact_activations;
        let plan = Self::resolve_segment_plan(
            &mut self.segment_plans,
            &mut self.plan_hits,
            &mut self.plan_misses,
            (program_key, new_count, kv_bound_extent),
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
                set_math_mode(&mut plan, self.math_mode)?;
                set_dispatch_type(&mut plan, omega::metal::DispatchType::Serial);
                Ok(plan)
            },
        )?;
        Ok(execute_plan_named_metal_op_timed_with_expert_sources(
            plan,
            named,
            expert_sources,
        )?)
    }

    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    fn evaluate_segment_with_placements_and_expert_sources(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
        bindings: &SegmentMetalBindings<'_, '_>,
    ) -> Result<Evaluated, InteropError> {
        let program_key = program.as_ptr() as usize;
        let new_count = symbols.first().copied().unwrap_or_default() as usize;
        let kv_bound_extent = symbols.get(1).copied().unwrap_or_default() as usize;
        let numerics = PlanNumerics {
            math_mode: self.math_mode,
            numeric_policy: self.numeric_policy,
            dispatch_type: omega::metal::DispatchType::Serial,
        };
        let plan = Self::resolve_segment_plan(
            &mut self.placed_segment_plans,
            &mut self.plan_hits,
            &mut self.plan_misses,
            (program_key, new_count, kv_bound_extent),
            || Self::build_placed_plan(program, symbols, named, outputs, resident_names, &numerics),
        )?;
        Ok(execute_plan_named_with_placements_and_expert_sources(
            plan,
            named,
            bindings.input_placements,
            bindings.output_placements,
            bindings.expert_sources,
        )?)
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

    fn resolve_segment_plan<'cache, PlanType>(
        cache: &'cache mut alloc::collections::BTreeMap<(usize, usize, usize), PlanType>,
        plan_hits: &mut usize,
        plan_misses: &mut usize,
        shape: (usize, usize, usize),
        build: impl FnOnce() -> Result<PlanType, InteropError>,
    ) -> Result<&'cache mut PlanType, InteropError> {
        use alloc::collections::btree_map::Entry;

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
    #[cfg(all(
        feature = "instrument",
        target_os = "macos",
        not(feature = "metal-output-placement")
    ))]
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
        // `PROXIMA_METAL_COMPARE_CPU` -- diagnostic-only, `instrument`-gated,
        // default-off: unset in every production run, so `cpu_reference`
        // stays `None` and `execute_plan_named_metal_op_timed` below runs
        // byte-for-byte the pre-existing path. When set, evaluates the SAME
        // `program`/`symbols`/`named` on the CPU route for every non-`Input`
        // node (weights are `Op::Input`, never a root here) so
        // `execute_op_timed`'s own `compare_op_output_to_cpu` hook
        // (`omega/src/metal.rs`) can diff each Metal op's output against its
        // CPU counterpart as the step runs, in program order, and stop at
        // the first node whose relative diff exceeds its own threshold.
        //
        // `is_quantized_matmul_multiply` excludes a node shape that can
        // NEVER be compared regardless of cost: a fused quantized matmul's
        // own `Multiply` elementwise (weight x activation, feeding a
        // `Reduce::Add`) is never given its own device buffer by EITHER
        // engine's `bind::bind` unless it is itself a requested output --
        // the real Metal `plan` built just above requests only `outputs`
        // (`roots`, a handful of nodes), so `device_buffers` never holds an
        // entry for one of these and `compare_op_output_to_cpu` can never
        // diff it. Requesting it here anyway forces `plan_named_cpu`'s
        // `bind::bind` to defuse it and dequantize the weight -- wasted
        // work for a comparison that can never happen -- so excluding it is
        // correct regardless of the OOM below.
        //
        // NOT YET FOUND: measured (`/usr/bin/time -l`,
        // `moe-metal-cmp2-logs/step3_run.log` /
        // `step3_run_fixed.log`) 139 GB and 138 GB peak memory footprint,
        // respectively, on the real 30B qwen3moe blob at
        // `PROXIMA_METAL_OP_PROFILE_STEP=3` with this exclusion in place --
        // i.e. excluding this node shape did NOT move the footprint outside
        // noise, so the fused-multiply dequant is NOT the OOM's dominant
        // cost. The process dies before printing a single `cpu_compare`
        // line at any node, which given `debug!` below fires BEFORE
        // `execute_plan_named` even starts suggests the dominant cost is
        // inside `execute_plan_named_cpu`/`evaluate_quantized_with_scratch_impl`
        // itself (candidate: `node_retirement`'s retire policy keeping every
        // one of this program's thousands of per-layer/per-expert
        // intermediate activation buffers alive for the whole call, not
        // only quantized weights, because `effective_outputs` names nearly
        // every node in the program). Narrowed, not proven -- the next
        // instrumentation is a live byte-counter inside
        // `evaluate_quantized_with_scratch_impl`'s per-node loop
        // (`proxima-tensor/src/cpu.rs`), not another node-shape guess.
        let cpu_reference: Option<alloc::collections::BTreeMap<NodeId, Vec<f32>>> =
            if std::env::var_os("PROXIMA_METAL_COMPARE_CPU").is_some() {
                let all_computed_nodes: Vec<NodeId> = program
                    .iter()
                    .enumerate()
                    .filter(|(index, op)| {
                        !matches!(op, Op::Input { .. })
                            && !is_quantized_matmul_multiply(program, NodeId(*index as u32))
                    })
                    .map(|(index, _)| NodeId(index as u32))
                    .collect();
                debug!(
                    program_len = program.len() as u64,
                    requested_outputs = all_computed_nodes.len() as u64,
                    "cpu_compare: about to build cpu reference plan"
                );
                let mut cpu_plan = plan_named(
                    Engine::Cpu,
                    None,
                    program,
                    symbols,
                    named,
                    &all_computed_nodes,
                    self.numeric_policy,
                )?;
                let cpu_evaluated = execute_plan_named(&mut cpu_plan, named)?;
                Some(
                    all_computed_nodes
                        .iter()
                        .filter_map(|node| {
                            cpu_evaluated
                                .get(*node)
                                .map(|(data, _)| (*node, data.to_vec()))
                        })
                        .collect(),
                )
            } else {
                None
            };
        Ok(execute_plan_named_metal_op_timed(
            plan,
            named,
            cpu_reference.as_ref(),
        )?)
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

    fn uses_gpu(&self) -> bool {
        false
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
        expert_sources: Option<
            &alloc::collections::BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'_>>,
        >,
    ) -> Result<Evaluated, InteropError> {
        if self.exact_activations {
            return Ok(evaluate_quantized_named_exact_with_scratch_and_experts(
                program,
                symbols,
                named,
                outputs,
                &mut self.free_buffers,
                &mut self.validated_weight_nodes,
                expert_sources,
            )?);
        }
        Ok(evaluate_quantized_named_with_scratch_and_experts(
            program,
            symbols,
            named,
            outputs,
            &mut self.free_buffers,
            &mut self.validated_weight_nodes,
            expert_sources,
        )?)
    }

    /// Evaluates one graph partition through the same reusable CPU scratch
    /// state as [`Self::evaluate`]. The partition owns its node numbering and
    /// therefore cannot alias the decode path's plan cache; CPU has no plan
    /// cache, so the only safe reusable state is the evaluator's scratch pool
    /// and weight-validation set. Keeping this method beside the Metal
    /// implementation gives routed callers one backend-independent seam.
    pub(crate) fn evaluate_segment(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
        expert_sources: Option<
            &alloc::collections::BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'_>>,
        >,
    ) -> Result<Evaluated, InteropError> {
        self.evaluate(
            program,
            symbols,
            named,
            outputs,
            resident_names,
            expert_sources,
        )
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
/// `decode_until_stop_or_budget` hands to its `on_token` callback every
/// step, teaching a caller (a CLI's "loading / thinking / answering"
/// indicator) exactly what that loop already knows at that point and
/// nothing it has to re-derive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenEvent<'piece> {
    pub token_id: u32,
    /// This token's own decoded text, continuing on from whatever
    /// `decode_until_stop_or_budget` has already handed back for earlier
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

/// Allocation-free decode evidence assembled from the events a caller already
/// receives. The labels preserve provenance when a record is written beside
/// measurements from another runtime or benchmark harness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeMetrics {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub elapsed_ms: u64,
    pub tokens_per_second: f64,
    pub per_token_latency_ms: f64,
    pub peak_rss_bytes: u64,
    pub cpu_percent: f64,
    pub error_count: u64,
    pub covariance: f64,
    pub timing_source: &'static str,
    pub memory_source: &'static str,
    pub resource_source: &'static str,
}

impl DecodeMetrics {
    /// Builds one record without allocating or re-reading the model output.
    /// `elapsed_ms` is the final cumulative [`TokenEvent`] timestamp, so the
    /// throughput and latency fields describe the same observed interval.
    pub fn from_events(
        events: &[TokenEvent<'_>],
        peak_rss_bytes: u64,
        cpu_percent: f64,
        error_count: u64,
        covariance: f64,
    ) -> Self {
        let mut prompt_tokens = 0;
        let mut generated_tokens = 0;
        let mut elapsed_ms = 0;
        for event in events {
            match event.phase {
                Phase::Prefill {
                    prompt_tokens: count,
                } => prompt_tokens = count,
                Phase::Token => generated_tokens += 1,
            }
            elapsed_ms = event.elapsed_ms;
        }
        let elapsed_seconds = elapsed_ms as f64 / 1000.0;
        let tokens_per_second = if elapsed_seconds > 0.0 {
            generated_tokens as f64 / elapsed_seconds
        } else {
            0.0
        };
        let per_token_latency_ms = if generated_tokens > 0 {
            elapsed_ms as f64 / generated_tokens as f64
        } else {
            0.0
        };
        Self {
            prompt_tokens,
            generated_tokens,
            elapsed_ms,
            tokens_per_second,
            per_token_latency_ms,
            peak_rss_bytes,
            cpu_percent,
            error_count,
            covariance,
            timing_source: "TokenEvent::elapsed_ms",
            memory_source: "caller_peak_rss_bytes",
            resource_source: "caller_cpu_percent_and_error_count",
        }
    }
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

/// A caller's per-token decision, read back by `decode_until_stop_or_budget`
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

/// Calls [`crate::expert_slab::ExpertSlab::end_step`] on every exit from the
/// expert-gather phase -- normal return and an early `?` error -- so the
/// source-snapshot guard cannot remain closed after a failed evaluation.
struct EndStepOnDrop<'a, 'file> {
    slab: &'a std::sync::Mutex<crate::expert_slab::ExpertSlab<'file>>,
}

impl Drop for EndStepOnDrop<'_, '_> {
    fn drop(&mut self) {
        lock_expert_slab(self.slab).end_step();
    }
}

/// Every `LoadedModel::expert_slab` acquire, in one place -- recovers from
/// poisoning instead of panicking (see that field's own doc for why) rather
/// than each call site repeating the same `unwrap_or_else`.
fn lock_expert_slab<'lock, 'file>(
    slab: &'lock std::sync::Mutex<crate::expert_slab::ExpertSlab<'file>>,
) -> std::sync::MutexGuard<'lock, crate::expert_slab::ExpertSlab<'file>> {
    slab.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn begin_expert_gather_phase<'lock, 'file>(
    slab: &'lock std::sync::Mutex<crate::expert_slab::ExpertSlab<'file>>,
) -> EndStepOnDrop<'lock, 'file> {
    lock_expert_slab(slab).begin_step();
    EndStepOnDrop { slab }
}

fn visit_qwen35moe_router_selections<BeforeGather>(
    layer: usize,
    position_offset: usize,
    logits: &[f32],
    shape: &[u64],
    expert_count: usize,
    expert_used_count: usize,
    scratch: &mut Vec<crate::residency::RoutedExpert>,
    before_gather: &mut BeforeGather,
) -> Result<(), InteropError>
where
    BeforeGather: FnMut(usize, u64, &[crate::residency::RoutedExpert]) -> Result<(), InteropError>,
{
    let [positions, shaped_experts] = shape else {
        return Err(InteropError::PreGatherExecutionUnsupported {
            architecture: String::from("qwen35moe"),
            reason: alloc::format!(
                "layer {layer} router logits have shape {shape:?}, expected [positions, experts]"
            ),
        });
    };
    let positions =
        usize::try_from(*positions).map_err(|_| InteropError::PreGatherExecutionUnsupported {
            architecture: String::from("qwen35moe"),
            reason: alloc::format!("layer {layer} router position extent does not fit usize"),
        })?;
    let shaped_experts = usize::try_from(*shaped_experts).map_err(|_| {
        InteropError::PreGatherExecutionUnsupported {
            architecture: String::from("qwen35moe"),
            reason: alloc::format!("layer {layer} router expert extent does not fit usize"),
        }
    })?;
    let expected_values = positions.checked_mul(expert_count).ok_or_else(|| {
        InteropError::PreGatherExecutionUnsupported {
            architecture: String::from("qwen35moe"),
            reason: alloc::format!("layer {layer} router shape overflows usize"),
        }
    })?;
    if shaped_experts != expert_count
        || logits.len() != expected_values
        || expert_used_count == 0
        || expert_used_count > expert_count
    {
        return Err(InteropError::PreGatherExecutionUnsupported {
            architecture: String::from("qwen35moe"),
            reason: alloc::format!(
                "layer {layer} router has shape {shape:?}, {} values, expert_count {expert_count}, and expert_used_count {expert_used_count}",
                logits.len()
            ),
        });
    }

    for (local_position, row) in logits.chunks_exact(expert_count).enumerate() {
        if std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some() {
            let nan_count = row.iter().filter(|value| value.is_nan()).count();
            let min = row.iter().copied().fold(f32::INFINITY, f32::min);
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "qwen35 router logits layer={} position={} min={} max={} nan_count={}",
                layer,
                position_offset.saturating_add(local_position),
                min,
                max,
                nan_count
            );
        }
        scratch.clear();
        for (expert, &importance) in row.iter().enumerate() {
            let candidate = crate::residency::RoutedExpert { expert, importance };
            if scratch.len() < expert_used_count {
                scratch.push(candidate);
            } else {
                let last = scratch[expert_used_count - 1];
                if importance.total_cmp(&last.importance).is_gt()
                    || (importance.total_cmp(&last.importance).is_eq() && expert < last.expert)
                {
                    scratch[expert_used_count - 1] = candidate;
                } else {
                    continue;
                }
            }
            scratch.sort_unstable_by(|left, right| {
                right
                    .importance
                    .total_cmp(&left.importance)
                    .then_with(|| left.expert.cmp(&right.expert))
            });
        }
        before_gather(
            layer,
            position_offset.saturating_add(local_position) as u64,
            scratch,
        )?;
    }
    Ok(())
}

fn visit_qwen35moe_router_boundary<'file, BeforeGather>(
    layer: usize,
    position_offset: usize,
    logits: &[f32],
    shape: &[u64],
    expert_count: usize,
    expert_used_count: usize,
    scratch: &mut Vec<crate::residency::RoutedExpert>,
    expert_slab: &mut crate::expert_slab::ExpertSlab<'file>,
    before_gather: &mut BeforeGather,
) -> Result<(), InteropError>
where
    BeforeGather: FnMut(
        usize,
        u64,
        &[crate::residency::RoutedExpert],
        &mut crate::expert_slab::ExpertSlab<'file>,
    ) -> Result<(), InteropError>,
{
    visit_qwen35moe_router_selections(
        layer,
        position_offset,
        logits,
        shape,
        expert_count,
        expert_used_count,
        scratch,
        &mut |layer, position, routes| {
            expert_slab.end_step();
            let boundary_result = before_gather(layer, position, routes, expert_slab);
            if boundary_result.is_ok() {
                expert_slab.begin_step();
            }
            boundary_result
        },
    )
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
            supported_serving_config(
                0,
                #[cfg(all(feature = "metal", target_os = "macos"))]
                omega::MathMode::default(),
            ),
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

    /// This checkpoint's own declared KV/SSM cache leaf names, derived from
    /// the compiled program's `Op::Input` set, plus each layer's own pad-row
    /// widths -- the SINGLE source of truth both
    /// [`Self::run_decode_loop_observed_seeded`] and
    /// [`Self::forward_node_values_on_backend`] read to learn which leaf
    /// names a step must feed. Before this method existed, the decode loop
    /// derived these from the program (correct) while the one-shot forward
    /// path hard-coded `kv_cache.{layer}.{k_even,k_odd,v}` (wrong for any
    /// architecture, such as a partial-rotary attention layer, whose cache
    /// leaves are named differently) -- see this crate's own
    /// `StepInputArch`-style fixtures in `tests/` for the shape a foreign
    /// architecture takes advantage of. Never trusts
    /// `self.layer_roots[layer]`'s own hand-kept discriminant over what the
    /// program actually declared (`DeclaredCacheKind`'s own doc).
    ///
    /// # Errors
    ///
    /// [`InteropError::LayerCacheKindMismatch`] when a layer's declared
    /// program leaves disagree with `self.layer_roots`' own tag for it.
    /// `self`, checked against [`Self::declared_layer_cache_names_and_widths`]
    /// and discarded if that check errors -- every production
    /// [`Self::load`]/[`Self::load_with_registry`]/[`Self::load_from_safetensors`]
    /// construction site runs through this before it ever reaches the
    /// decode loop, so a layer whose `layer_roots` say it is stateful but
    /// whose program bakes that state as constants fails here, at load,
    /// instead of silently decoding from zero state
    /// ([`InteropError::LayerCacheLeavesMissing`]'s own doc).
    fn validated(self) -> Result<Self, InteropError> {
        self.declared_layer_cache_names_and_widths()?;
        Ok(self)
    }

    fn declared_layer_cache_names_and_widths(
        &self,
    ) -> Result<(Vec<LayerCacheNames>, Vec<LayerPadRowWidths>), InteropError> {
        let program_input_names: BTreeSet<&str> = self
            .program
            .iter()
            .filter_map(|op| match op {
                Op::Input {
                    name: Some(name), ..
                } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        let mut layer_cache_kinds: Vec<DeclaredCacheKind> =
            Vec::with_capacity(self.layer_roots.len());
        for (layer, roots) in self.layer_roots.iter().enumerate() {
            let bound = bound_cache_kind(roots);
            let declared = match declared_cache_kind(&program_input_names, layer) {
                Some(declared) => declared,
                None => {
                    return Err(InteropError::LayerCacheLeavesMissing {
                        layer,
                        kind: bound.label(),
                        expected: bound.expected_leaf_templates(),
                    });
                }
            };
            if declared != bound {
                return Err(InteropError::LayerCacheKindMismatch {
                    layer,
                    declared: declared.label(),
                    bound: bound.label(),
                });
            }
            layer_cache_kinds.push(declared);
        }
        let cache_names: Vec<LayerCacheNames> = layer_cache_kinds
            .iter()
            .enumerate()
            .map(|(layer, kind)| match kind {
                DeclaredCacheKind::Attention => LayerCacheNames::Attention {
                    k_even: alloc::format!("kv_cache.{layer}.k_even"),
                    k_odd: alloc::format!("kv_cache.{layer}.k_odd"),
                    v: alloc::format!("kv_cache.{layer}.v"),
                },
                DeclaredCacheKind::DenseAttention => LayerCacheNames::DenseAttention {
                    k_first: alloc::format!("kv_cache.{layer}.k_first"),
                    k_second: alloc::format!("kv_cache.{layer}.k_second"),
                    k_pass: alloc::format!("kv_cache.{layer}.k_pass"),
                    v: alloc::format!("kv_cache.{layer}.v"),
                },
                DeclaredCacheKind::Ssm => LayerCacheNames::Ssm {
                    conv_history: alloc::format!("ssm_cache.{layer}.conv_history"),
                    state: alloc::format!("ssm_cache.{layer}.state"),
                },
            })
            .collect();
        let layer_row_widths: Vec<LayerPadRowWidths> = cache_names
            .iter()
            .map(|names| layer_pad_row_widths(&self.program, names))
            .collect();
        Ok((cache_names, layer_row_widths))
    }

    /// A fresh, empty [`LayerCacheState`] per layer, shaped off
    /// [`Self::declared_layer_cache_names_and_widths`]'s own `cache_names`/
    /// `layer_row_widths` pair -- the decode loop's own prefill-step state
    /// (absent a `seed`) and [`Self::forward_node_values_on_backend`]'s own
    /// always-fresh state (every one-shot forward starts from an empty
    /// cache, that method's own doc), unified so neither caller hand-picks
    /// which [`LayerCacheState`] variant a layer gets, or how big it starts,
    /// independently of what
    /// [`declared_layer_cache_names_and_widths`](Self::declared_layer_cache_names_and_widths)
    /// already decided. `layer_row_widths` (not
    /// [`crate::architecture::Architecture::step_state`]) is the `Ssm` arm's
    /// own size source -- see [`cache_leaf_total_elements`]'s own doc for
    /// why: a foreign `Architecture` that never overrides `step_state`
    /// (the trait's own `Ok(None)` default) still declares its
    /// `ssm_cache.{layer}.*` leaves as `Op::Input` ops, so the program
    /// itself, not a per-architecture hook, is what every layer's initial
    /// cache is sized from.
    fn fresh_layer_caches(
        &self,
        cache_names: &[LayerCacheNames],
        layer_row_widths: &[LayerPadRowWidths],
    ) -> Vec<LayerCacheState> {
        cache_names
            .iter()
            .zip(layer_row_widths)
            .map(|(names, widths)| match (names, widths) {
                (LayerCacheNames::Attention { .. }, _) => {
                    LayerCacheState::Attention(LayerCache::new())
                }
                (LayerCacheNames::DenseAttention { .. }, _) => {
                    LayerCacheState::DenseAttention(Qwen35DenseAttentionCache::new())
                }
                (
                    LayerCacheNames::Ssm {
                        conv_history,
                        state,
                    },
                    LayerPadRowWidths::Ssm {
                        conv_history_len,
                        state_len,
                    },
                ) => {
                    #[cfg(feature = "instrument")]
                    debug!(
                        conv_history_name = %conv_history,
                        state_name = %state,
                        conv_history_len = *conv_history_len,
                        state_len = *state_len,
                        "ssm cache seeded at its program-declared shape"
                    );
                    #[cfg(not(feature = "instrument"))]
                    let _ = (conv_history, state);
                    LayerCacheState::Ssm(SsmLayerCache::new(*conv_history_len, *state_len))
                }
                _ => unreachable!(
                    "cache_names/layer_row_widths built from the same layer_roots, in lockstep"
                ),
            })
            .collect()
    }

    /// Pushes one step's position/RoPE inputs, `cached_len`/`lm_head_row`
    /// scalars, [`Architecture::step_inputs`]' own per-step leaves, and
    /// every KV/SSM cache leaf ([`push_kv_named_blocks`]) into `named_blocks`
    /// -- the ONE assembly both
    /// [`Self::run_decode_loop_observed_seeded`] and
    /// [`Self::forward_node_values_on_backend`] call, so a foreign
    /// architecture's own [`Architecture::step_inputs`] override and its own
    /// cache leaf names are fed identically whether the caller is decoding
    /// token-by-token or tapping one interior node from a single forward
    /// pass. `position_inputs`/`cached_len_scalar`/`lm_head_row_scalar` are
    /// owned by the CALLER (not this method) so the `QuantizedBlock`s this
    /// method pushes can borrow them for `named_blocks`' own `'call`
    /// lifetime without a self-referential return type. Returns the
    /// [`bind_symbols`] result (this step's own `Extent::Symbolic` binding)
    /// rather than leaving the caller to re-borrow `step_input_scratch`
    /// afterward -- `named_blocks` already holds borrows into it once this
    /// method returns, so a second, independent borrow to compute symbols
    /// would conflict with the one `named_blocks` is holding.
    ///
    /// # Errors
    ///
    /// [`InteropError::UnknownStepInput`] if `Architecture::step_inputs`
    /// names a leaf this checkpoint's program never declared, plus whatever
    /// [`push_kv_named_blocks`]/[`bind_symbols`] can fail with.
    #[allow(clippy::too_many_arguments)]
    fn push_step_named_blocks<'call>(
        &'call self,
        position_inputs: &'call PositionInputs,
        cached_len_scalar: &'call [f32; 1],
        lm_head_row_scalar: &'call [f32; 1],
        token_history: &[u32],
        new_start: usize,
        new_count: usize,
        cache_names: &'call [LayerCacheNames],
        layer_caches: &'call [LayerCacheState],
        layer_row_widths: &[LayerPadRowWidths],
        kv_bound_extent: usize,
        kv_pad_scratch: &'call mut [KvPadScratch],
        qwen35_dense_pad_scratch: &'call mut [Qwen35DenseAttentionPadScratch],
        step_input_scratch: &'call mut Vec<StepInput>,
        named_blocks: &mut Vec<(&'call str, QuantizedBlock<'call>)>,
        single_position_step: bool,
    ) -> Result<Vec<u64>, InteropError> {
        named_blocks.push((
            "ids",
            QuantizedBlock::Int32(position_inputs.ids_i32.as_slice()),
        ));
        named_blocks.push((
            "eps",
            QuantizedBlock::Float32(position_inputs.epsilon.as_slice()),
        ));
        named_blocks.push((
            "rope_cos",
            QuantizedBlock::Float32(position_inputs.cos.as_slice()),
        ));
        named_blocks.push((
            "rope_sin",
            QuantizedBlock::Float32(position_inputs.sin.as_slice()),
        ));
        named_blocks.push((
            "cached_len",
            QuantizedBlock::Float32(cached_len_scalar.as_slice()),
        ));
        named_blocks.push((
            "lm_head_row",
            QuantizedBlock::Float32(lm_head_row_scalar.as_slice()),
        ));

        step_input_scratch.clear();
        if let Some(architecture_impl) = self.architecture_impl {
            let step_context = StepInputContext {
                all_token_ids: token_history,
                new_start,
                new_count,
            };
            architecture_impl.step_inputs(&step_context, step_input_scratch);
        }
        for step_input in step_input_scratch.iter() {
            if !self
                .program
                .iter()
                .any(|op| op.name() == Some(step_input.name))
            {
                return Err(InteropError::UnknownStepInput {
                    name: String::from(step_input.name),
                });
            }
            named_blocks.push(step_input.as_named_block());
        }
        let symbols = bind_symbols(
            new_count,
            kv_bound_extent,
            step_input_scratch,
            single_position_step,
        )?;

        push_kv_named_blocks(
            cache_names,
            layer_caches,
            layer_row_widths,
            kv_bound_extent,
            kv_pad_scratch,
            qwen35_dense_pad_scratch,
            named_blocks,
        )?;
        Ok(symbols)
    }

    /// Pages `bytes` in as expert `expert`'s new weight for layer `layer` --
    /// a thin forward onto [`crate::expert_slab::ExpertSlab::page_expert`],
    /// the primitive this method composes (P2 teaching surface: read that
    /// method's own doc for the aliasing/ownership contract this call
    /// changes). `layer` is the [`crate::bind::build_expert_slab`] SITE
    /// index (one slot per `blk.{n}.{ffn_gate,ffn_up,ffn_down}_exps.weight`
    /// tensor this checkpoint's own forward program bound, in that order),
    /// not necessarily the checkpoint's transformer layer number when more
    /// than one projection is routed per layer.
    ///
    /// # Errors
    /// [`InteropError::ExpertSwapDuringStep`] if called while a decode step
    /// is running; [`InteropError::ExpertSlabIndexOutOfRange`] if `layer` or
    /// `expert` is out of range for this checkpoint's slab.
    pub fn page_expert(
        &self,
        layer: usize,
        expert: usize,
        codec: crate::bind::PackedOwnedKind,
        bytes: &[u8],
        out_dim: u32,
        in_dim: u32,
    ) -> Result<u64, InteropError> {
        lock_expert_slab(&self.expert_slab)
            .page_expert(layer, expert, codec, bytes, out_dim, in_dim)
    }

    /// Pages a HOBBIT high-precision expert from one range of a live mmap,
    /// without copying its payload into a `Vec<u8>`. This composes
    /// [`crate::expert_slab::ExpertSlab::page_expert_mapped`]; that method
    /// retains the [`Arc<Mmap>`] through every per-step
    /// [`proxima_tensor::cpu::ExpertSource`] snapshot, so callers may drop
    /// their own mapping handle after this returns.
    ///
    /// `layer` is the same gathered-reduce site index as [`Self::page_expert`],
    /// while `range` selects exactly one encoded expert inside the source
    /// mapping. The slab's [`crate::expert_slab::ExpertSlab::memory`] report
    /// records this range as `mapped_bytes` and leaves `owned_bytes` unchanged.
    ///
    /// # Errors
    /// [`InteropError::ExpertMappedRangeOutOfBounds`] when `range` falls
    /// outside `mapping`, plus the same step/index errors as
    /// [`Self::page_expert`].
    pub fn page_expert_mapped(
        &self,
        layer: usize,
        expert: usize,
        codec: crate::bind::PackedOwnedKind,
        mapping: Arc<Mmap>,
        range: Range<usize>,
        out_dim: u32,
        in_dim: u32,
    ) -> Result<u64, InteropError> {
        lock_expert_slab(&self.expert_slab)
            .page_expert_mapped(layer, expert, codec, mapping, range, out_dim, in_dim)
    }

    /// Attaches HOBBIT's mmap-backed low-codec expert store to this model.
    ///
    /// The mapping is parsed and indexed once, then every gate/up/down expert
    /// entry is replaced by its low-codec mapped view. The original checkpoint
    /// offsets retained by the sidecar become the high-codec promotion source.
    /// No expert payload is copied or heap-allocated by this operation.
    pub fn attach_expert_sidecar(&mut self, mapping: Arc<Mmap>) -> Result<(), InteropError> {
        mapping.advise(Advice::Random)?;
        let sidecar = crate::expert_sidecar::MappedExpertSidecar::new(mapping)?;
        self.attach_indexed_expert_sidecar(sidecar)
    }

    /// Attaches a sidecar that preads only the expert ranges selected per layer.
    pub fn attach_expert_sidecar_file(&mut self, file: File) -> Result<(), InteropError> {
        let sidecar = crate::expert_sidecar::MappedExpertSidecar::from_file(file)?;
        self.attach_indexed_expert_sidecar(sidecar)
    }

    fn attach_indexed_expert_sidecar(
        &mut self,
        sidecar: crate::expert_sidecar::MappedExpertSidecar,
    ) -> Result<(), InteropError> {
        if self.architecture_impl.map(Architecture::name) != Some("qwen35moe") {
            return Err(InteropError::PreGatherExecutionUnsupported {
                architecture: self.architecture_impl.map_or_else(
                    || String::from("unknown"),
                    |value| String::from(value.name()),
                ),
                reason: String::from("expert sidecars require a qwen35moe expert graph"),
            });
        }
        sidecar.install_low_copies(
            &mut lock_expert_slab(&self.expert_slab),
            self.architecture.block_count as usize,
            self.architecture.expert_count as usize,
        )?;
        if std::env::var_os("PROXIMA_DEBUG_MEMORY_OWNERS").is_some() {
            let owned_bytes = self
                .weights
                .owned
                .iter()
                .map(|(_, values)| values.len() * core::mem::size_of::<f32>())
                .sum::<usize>();
            let packed_bytes = self
                .weights
                .packed
                .iter()
                .map(|(_, block)| block.packed_bytes().map_or(0, <[u8]>::len))
                .sum::<usize>();
            let packed_owned_bytes = self
                .weights
                .packed_owned
                .iter()
                .map(|(_, bytes, _)| bytes.len())
                .sum::<usize>();
            let slab_memory = lock_expert_slab(&self.expert_slab).memory();
            eprintln!(
                "qwen35 memory owners checkpoint_bytes={} owned_bytes={} packed_bytes={} packed_owned_bytes={} sidecar_mapped_bytes={} sidecar_owned_bytes={} sidecar_descriptors={}",
                self.checkpoint_bytes,
                owned_bytes,
                packed_bytes,
                packed_owned_bytes,
                slab_memory.mapped_bytes,
                slab_memory.owned_bytes,
                sidecar.descriptor_count(),
            );
        }
        // The whole-checkpoint Metal buffer is a convenient zero-copy fast
        // path for ordinary GGUF serving, but it makes the driver account for
        // the entire mmap even when HOBBIT substitutes every expert.  Once a
        // sidecar owns the expert bytes, drop that device-wide mapping so the
        // remaining tensors bind independently and the residency budget is
        // reflected by actual device buffers.
        #[cfg(feature = "metal")]
        omega::backend::unregister_checkpoint_mapping(self.checkpoint_mapping);
        self.expert_sidecar = Some(sidecar);
        Ok(())
    }

    /// Number of sidecar expert-projection records owned by this model.
    #[must_use]
    pub fn expert_sidecar_descriptor_count(&self) -> usize {
        self.expert_sidecar.as_ref().map_or(
            0,
            crate::expert_sidecar::MappedExpertSidecar::descriptor_count,
        )
    }

    /// Applies one DynaExq/HOBBIT resident-set transition between decode
    /// steps.  The caller owns the policy and the high-precision source; this
    /// method only joins that policy to this model's slab, which is the table
    /// the next decode step snapshots as [`ExpertSource`] entries.  Keeping
    /// the page callback generic preserves the zero-allocation boundary and
    /// lets a caller return bytes from an mmap or LSM segment without a
    /// trait-object allocation.
    ///
    /// The method deliberately does not apply actions while a step is active:
    /// [`ExpertSlab`] returns its typed boundary error, preventing a policy
    /// update from invalidating the borrowed sources of the current step.
    pub fn apply_expert_residency<
        const LAYERS: usize,
        const EXPERTS: usize,
        const ACTIONS: usize,
        Page,
    >(
        &self,
        policy: &mut crate::residency::ExpertResidency<LAYERS, EXPERTS>,
        actions: &crate::residency::ResidencyActions<ACTIONS>,
        page: Page,
    ) -> Result<(), InteropError>
    where
        Page: FnMut(
            crate::residency::ExpertAddress,
        ) -> Result<crate::residency::ExpertPage<'file>, InteropError>,
    {
        let mut slab = lock_expert_slab(&self.expert_slab);
        policy.apply_at_boundary(&mut slab, actions, page)
    }

    /// Applies a fixed DynaExq action batch to the attached HOBBIT sidecar.
    /// A page promotes all three projections from their original checkpoint
    /// ranges; an eviction restores all three low-codec mapped ranges.
    pub fn apply_attached_expert_residency<
        const LAYERS: usize,
        const EXPERTS: usize,
        const ACTIONS: usize,
    >(
        &self,
        policy: &mut crate::residency::ExpertResidency<LAYERS, EXPERTS>,
        actions: &crate::residency::ResidencyActions<ACTIONS>,
    ) -> Result<(), InteropError> {
        let sidecar = self.expert_sidecar.as_ref().ok_or_else(|| {
            InteropError::PreGatherExecutionUnsupported {
                architecture: String::from("qwen35moe"),
                reason: String::from("no expert sidecar is attached"),
            }
        })?;
        let mut slab = lock_expert_slab(&self.expert_slab);
        policy.apply_actions_at_boundary(&mut slab, actions, |slab, action| {
            sidecar.apply_action(slab, self.checkpoint_mapping, action)
        })
    }

    /// Runs the explicit router -> residency -> gather protocol for a
    /// qwen35moe runtime integration.  The current forward graph exposes the
    /// router and routed gather as one graph evaluation, so this seam accepts
    /// a caller-owned router prepass and source transition rather than
    /// pretending that the existing graph has been partitioned.  A caller
    /// that has not built that prepass gets a typed error from its router
    /// callback; the gather callback is never invoked before the boundary.
    ///
    /// The callbacks are consuming and return their storage to the caller;
    /// no trait object, boxed future, or runtime allocation is introduced by
    /// this phase boundary.  `Routes` may be a fixed-capacity route array or
    /// a `Vec` owned by the caller, and `Source` may be an expert slab view or
    /// an mmap-backed table.
    pub fn execute_qwen35moe_pre_gather<Routes, Source, Output, Router, Boundary, Gather>(
        &self,
        router: Router,
        boundary: Boundary,
        gather: Gather,
    ) -> Result<Output, InteropError>
    where
        Routes: AsRef<[crate::residency::ServeDecision]>,
        Router: FnOnce() -> Result<Routes, InteropError>,
        Boundary:
            FnOnce(crate::qwen35moe::execution::RouterResult<'_>) -> Result<Source, InteropError>,
        Gather: FnOnce(
            crate::qwen35moe::execution::GatherPhase<'_, Source>,
        ) -> Result<Output, InteropError>,
    {
        let architecture = self.architecture_impl.map_or("unknown", Architecture::name);
        if architecture != "qwen35moe" {
            return Err(InteropError::PreGatherExecutionUnsupported {
                architecture: String::from(architecture),
                reason: String::from("the bound model is not a routed qwen35moe graph"),
            });
        }
        if self.router_roots.is_empty() {
            return Err(InteropError::PreGatherExecutionUnsupported {
                architecture: String::from(architecture),
                reason: String::from("the bound qwen35moe graph exposes no router roots"),
            });
        }
        crate::qwen35moe::execution::execute_pre_gather(router, boundary, gather)
    }

    /// Reports the active expert payloads retained by this model's slab.
    /// `owned_bytes` is the actual copied-payload footprint; `mapped_bytes`
    /// is the address-space range served directly from mmap. See
    /// [`crate::expert_slab::ExpertSlabMemory`] for why the latter is not an
    /// RSS claim.
    #[must_use]
    pub fn expert_slab_memory(&self) -> crate::expert_slab::ExpertSlabMemory {
        lock_expert_slab(&self.expert_slab).memory()
    }

    /// Removes expert `expert` of layer `layer`'s currently-bound bytes --
    /// see [`Self::page_expert`]'s own doc for what `layer` indexes, and
    /// [`crate::expert_slab::ExpertSlab::evict_expert`] for the primitive
    /// this composes.
    ///
    /// # Errors
    /// Same as [`Self::page_expert`].
    pub fn evict_expert(&self, layer: usize, expert: usize) -> Result<(), InteropError> {
        lock_expert_slab(&self.expert_slab).evict_expert(layer, expert)
    }

    /// `expert`'s current epoch for `layer`, or `None` if either index is
    /// out of range or the expert is currently evicted -- see
    /// [`crate::expert_slab::ExpertSlab::expert_epoch`].
    #[must_use]
    pub fn expert_epoch(&self, layer: usize, expert: usize) -> Option<u64> {
        lock_expert_slab(&self.expert_slab).expert_epoch(layer, expert)
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

    /// Runs one forward pass over `prompt`'s own tokens and returns the
    /// [`PrefixState`] it leaves behind, WITHOUT decoding anything past it
    /// -- `Self::run_decode_loop_observed_seeded` with `seed: None` and
    /// `max_tokens: 1`: step 0 of that loop always forwards the whole
    /// `next_ids` range against `cached_len == 0` before ever sampling, so
    /// asking for exactly one step is asking for exactly the prefill this
    /// primitive needs and nothing past it. The one token step 0 happens to
    /// sample (this loop's own next-token prediction) is discarded -- it is
    /// never forward-passed itself, so it is not part of the cache
    /// [`PrefixState`] reports; a caller after real generated text wants
    /// [`Self::generate_from_prefix`], not this method's own return.
    ///
    /// The returned [`PrefixState`] is independent of `self` and of this
    /// call's own `runtime` -- reusable across as many
    /// [`Self::generate_from_prefix`] calls as the caller likes, against
    /// this SAME `LoadedModel`, without re-running this forward pass.
    ///
    /// # Errors
    ///
    /// Same as [`Self::generate_with_serving_config`].
    pub fn prefill_prefix(
        &self,
        prompt: &str,
        serving_config: &ServingConfig,
    ) -> Result<PrefixState, InteropError> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let effective_serving_config = {
            let mut effective_serving_config = *serving_config;
            self.apply_memory_fit_gate(&mut effective_serving_config)?;
            effective_serving_config
        };
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        let effective_serving_config = *serving_config;
        let mut runtime = BackendRuntime::new(&effective_serving_config);
        let (_generated_ids, _text, _stopped_by_eos, prefix_state) = self
            .run_decode_loop_observed_seeded(
                prompt,
                1,
                &effective_serving_config,
                &mut runtime,
                None,
                &mut LogitsSink::Discard,
                &mut NodeValuesSink::Discard,
                &mut |_event| Control::Continue,
                None,
                true,
            )?;
        Ok(prefix_state)
    }

    /// Resumes decoding from `prefix` -- `Self::run_decode_loop_observed_seeded`
    /// with `seed: Some(prefix.clone())`, `prompt` now the SUFFIX text only,
    /// so this call's own two-range forward starts from `prefix`'s cached
    /// `cached_len` rows instead of `0`, and prefills ONLY the suffix's own
    /// tokens as the new range -- the multi-row prefill
    /// [`Self::prefill_prefix`] already ran for `prefix`'s own tokens is
    /// never repeated. `prefix` is cloned, never consumed: a second call
    /// against a different suffix, or a second `LoadedModel` call entirely,
    /// sees `prefix` exactly as this call received it.
    ///
    /// `suffix` is tokenized with neither BOS nor EOS added -- it continues
    /// the sequence `prefix` already opened, so the tokenizer must see it as
    /// a continuation, not a fresh prompt. If the two texts' own token
    /// boundary does not fall on a token the tokenizer would also choose
    /// when encoding `prefix_text + suffix_text` as one string (a
    /// unigram/BPE tokenizer can merge a trailing/leading fragment across a
    /// naive substring split), the caller owns splitting the prompt on a
    /// boundary the tokenizer already treats as a hard break -- a newline is
    /// the reliable one for this crate's own vocabularies.
    ///
    /// # Errors
    ///
    /// Same as [`Self::generate_with_serving_config`].
    pub fn generate_from_prefix(
        &self,
        prefix: &PrefixState,
        suffix: &str,
        max_tokens: usize,
        serving_config: &ServingConfig,
        on_token: &mut dyn FnMut(TokenEvent<'_>) -> Control,
    ) -> Result<(Vec<u32>, String, bool), InteropError> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let effective_serving_config = {
            let mut effective_serving_config = *serving_config;
            // Prefix-resume reaches the same device allocator as ordinary
            // generation. Apply the identical load-time budget before the
            // resumed step, otherwise a caller can bypass the hard memory
            // ceiling simply by supplying a PrefixState.
            self.apply_memory_fit_gate(&mut effective_serving_config)?;
            effective_serving_config
        };
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        let effective_serving_config = *serving_config;
        let mut runtime = BackendRuntime::new(&effective_serving_config);
        let seed = PrefixState {
            ids: prefix.ids.clone(),
            layer_caches: prefix.layer_caches.clone(),
            cached_len: prefix.cached_len,
        };
        let (generated_ids, text, stopped_by_eos, _final_state) = self
            .run_decode_loop_observed_seeded(
                suffix,
                max_tokens,
                &effective_serving_config,
                &mut runtime,
                None,
                &mut LogitsSink::Discard,
                &mut NodeValuesSink::Discard,
                on_token,
                Some(seed),
                true,
            )?;
        Ok((generated_ids, text, stopped_by_eos))
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
    fn apply_memory_fit_gate(
        &self,
        serving_config: &mut ServingConfig,
    ) -> Result<(), InteropError> {
        if !serving_config.gpu_memory_fit {
            return Ok(());
        }
        let Ok(facts) = omega::metal::system_memory_facts() else {
            return Ok(());
        };
        let detected_limit = crate::memory_fit::HostMemoryLimit {
            limit_bytes: facts
                .recommended_max_working_set_size
                .min(facts.physical_memory_bytes),
            os_headroom_bytes: omega::sized::LOAD_TIME_FIT_OS_HEADROOM_BYTES,
        };
        let limit = serving_config
            .gpu_memory_limit_bytes
            .map_or(detected_limit, |configured| {
                crate::memory_fit::HostMemoryLimit {
                    limit_bytes: configured.min(detected_limit.available_bytes()),
                    os_headroom_bytes: 0,
                }
            });
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
        if matches!(
            outcome,
            crate::memory_fit::FitOutcome::ReducedContext { .. }
        ) {
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
    /// every step `decode_until_stop_or_budget` already produces --
    /// `Self::run_decode_loop_observed`'s own loop, unchanged, given a
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
        let (generated_ids, text, stopped_by_eos, _prefix_state) = self
            .run_decode_loop_observed_seeded(
                prompt,
                max_tokens,
                serving_config,
                runtime,
                token_override,
                logits_sink,
                &mut NodeValuesSink::Discard,
                on_token,
                None,
                false,
            )?;
        Ok((generated_ids, text, stopped_by_eos))
    }

    /// [`Self::run_decode_loop_observed`]'s own body, plus a `seed`: `None`
    /// reproduces that method exactly (fresh [`LayerCacheState`] per layer,
    /// `cached_len` starting at 0, `prompt` tokenized WITH this vocab's own
    /// BOS/EOS policy); `Some(state)` resumes from a [`PrefixState`] a prior
    /// call returned instead -- `prompt` is then the SUFFIX text only,
    /// tokenized with NEITHER BOS nor EOS added (continuing the same
    /// sequence [`PrefixState::ids`] already opened), `layer_caches` starts
    /// from `state`'s own per-layer cache instead of [`LayerCache::new`],
    /// and `cached_len` starts from `state.cached_len` instead of `0`. The
    /// two-range decode loop below is BYTE-FOR-BYTE unchanged either way --
    /// this is the same primitive [`Self::generate_with_serving_config`]'s
    /// first step already runs (a multi-row forward over `next_ids` against
    /// `cached_len` rows of history), just given a nonzero `cached_len` and
    /// a non-full-prompt `next_ids` to start from -- so [`PrefixState`]
    /// itself is exactly the `(ids, layer_caches, cached_len)` triple this
    /// loop already threads through every step, exposed across calls rather
    /// than dropped at this function's own return.
    ///
    /// Always returns the FINAL [`PrefixState`] this call's own decoding
    /// left the cache in, alongside the usual generated-token triple --
    /// [`Self::run_decode_loop_observed`] discards it (nothing needs cross-
    /// call reuse there), [`Self::prefill_prefix`] is the one caller that
    /// keeps it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_decode_loop_observed_seeded(
        &self,
        prompt: &str,
        max_tokens: usize,
        serving_config: &ServingConfig,
        runtime: &mut BackendRuntime,
        token_override: Option<&[u32]>,
        logits_sink: &mut LogitsSink,
        node_values_sink: &mut NodeValuesSink,
        on_token: &mut dyn FnMut(TokenEvent<'_>) -> Control,
        seed: Option<PrefixState>,
        force_two_range: bool,
    ) -> Result<(Vec<u32>, String, bool, PrefixState), InteropError> {
        // Read unconditionally: the ONLY reader lives behind
        // `#[cfg(all(feature = "metal-output-placement", target_os =
        // "macos"))]` below, so a build without that cfg combination never
        // reads this parameter otherwise, and would warn on it as unused.
        let _ = force_two_range;
        let ids = if seed.is_some() {
            proxima_tokenizer::encode_with_bos_eos(prompt, &self.vocab, false, false)?
        } else {
            proxima_tokenizer::encode_with_bos_eos(
                prompt,
                &self.vocab,
                wants_bos(&self.vocab),
                self.vocab.add_eos_token().unwrap_or(false),
            )?
        };
        let seed_cached_len = seed.as_ref().map_or(0, PrefixState::len);
        let seed_ids: Vec<u32> = seed
            .as_ref()
            .map_or_else(Vec::new, |state| state.ids.clone());
        // The repetition-penalty filter's own window: prompt tokens included,
        // matching upstream (`tools/main/main.cpp:725` feeds prompt tokens
        // through the same `common_sampler_accept` generated tokens use), grown
        // by one id every decode step below. `sample_config`/`rng` are built
        // once and threaded through every step -- the same seeded
        // `fastrand::Rng` this workspace already uses for every other
        // deterministic-by-seed pipe, drawn from progressively rather than
        // reseeded per token, mirroring upstream's own one-`std::mt19937`-per-
        // sampler-chain lifetime (`proxima_tokenizer::sample`'s own doc).
        let mut token_history: Vec<u32> = {
            let mut history = seed_ids.clone();
            history.extend_from_slice(&ids);
            history
        };
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
        // `None` for any mixture-of-experts or qwen35 checkpoint), this
        // call's own `ServingConfig` selected the Metal backend
        // (`runtime.is_metal()`), AND the caller did not ask to force the
        // two-range path (`force_two_range`). [`Self::prefill_prefix`]/
        // [`Self::generate_from_prefix`] always set `force_two_range: true`
        // -- the placed-kv path's own [`SingleRangeProgram`] never leaves a
        // host-side [`LayerCacheState`] to report as a [`PrefixState`], so
        // it is not eligible for either primitive regardless of whether
        // this checkpoint would otherwise take it. Every other caller
        // (ordinary `generate`/`generate_streaming`, `seed: None`) is
        // unaffected: CPU decode, any MoE checkpoint, and the qwen35 hybrid
        // path always fall through to the two-range `layer_roots` path
        // below, byte-for-byte unchanged.
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        if !force_two_range
            && runtime.is_metal()
            && let Some(single_range) = &self.single_range
        {
            let (generated_ids, text, stopped_by_eos) = self.run_decode_loop_placed_kv(
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
            )?;
            // No host-side `LayerCacheState` exists on this path -- the
            // KV cache never left the device (`SingleRangeProgram`'s own
            // doc) -- so there is nothing real to report here. Callers
            // that need a real [`PrefixState`] set `force_two_range: true`
            // and never reach this branch at all.
            return Ok((
                generated_ids,
                text,
                stopped_by_eos,
                PrefixState {
                    ids: Vec::new(),
                    layer_caches: Vec::new(),
                    cached_len: 0,
                },
            ));
        }

        // The program's own declared `Op::Input` leaves are the single
        // source of truth for which cache shape each layer needs fed --
        // never `self.layer_roots[layer]`'s own discriminant, which a
        // foreign `Architecture::bind` assembles by hand and can tag
        // inconsistently with the ops it actually emitted (see
        // `DeclaredCacheKind`'s own doc). The SAME derivation
        // [`Self::forward_node_values_on_backend`] calls, so a foreign
        // architecture's cache leaf names are never hard-coded twice.
        let (cache_names, layer_row_widths) = self.declared_layer_cache_names_and_widths()?;
        let mut layer_caches: Vec<LayerCacheState> = match seed {
            Some(state) => state.layer_caches,
            None => self.fresh_layer_caches(&cache_names, &layer_row_widths),
        };
        // One [`KvPadScratch`] per layer, reused across every step of this
        // call -- only ever filled for a [`LayerCacheState::Attention`]
        // layer (the only cache shape `mistral_cached_forward_program_with_experts`
        // produces, `Qwen35LayerRoots`'s own doc), left empty and unread for
        // every `DenseAttention`/`Ssm` layer a qwen35 checkpoint carries.
        let mut kv_pad_scratch: Vec<KvPadScratch> = self
            .layer_roots
            .iter()
            .map(|_| KvPadScratch::new())
            .collect();
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

        // DynaExq observes the real routed expert ids produced by the graph.
        // The fixed matrix keeps policy state bounded and is enabled only
        // when the model owns a low-codec sidecar and the caller supplies a
        // high-precision residency budget.
        let residency_budget = std::env::var("PROXIMA_QWEN35MOE_RESIDENCY_BUDGET_BYTES")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0);
        let mut qwen35moe_residency = if self
            .architecture_impl
            .is_some_and(|architecture| architecture.name() == "qwen35moe")
            && self.expert_sidecar.is_some()
            && residency_budget > 0
        {
            Some(crate::residency::ExpertResidency::<40, 256>::new(
                crate::residency::ResidencyConfig {
                    budget_bytes: residency_budget,
                    high_bytes_per_expert: self.expert_sidecar.as_ref().map_or(
                        0,
                        crate::expert_sidecar::MappedExpertSidecar::high_bytes_per_expert,
                    ),
                    ..crate::residency::ResidencyConfig::default()
                },
            ))
        } else {
            None
        };

        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        let ssm_placement_enabled = runtime.is_metal()
            && std::env::var("PROXIMA_METAL_SSM_PLACEMENT")
                .ok()
                .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"));
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        let ssm_placement_max_layer = std::env::var("PROXIMA_METAL_SSM_PLACEMENT_MAX_LAYER")
            .ok()
            .and_then(|value| value.parse::<usize>().ok());
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        let ssm_state_input_nodes: Vec<Option<NodeId>> = cache_names
            .iter()
            .map(|names| match names {
                LayerCacheNames::Ssm { state, .. } => {
                    self.program
                        .iter()
                        .enumerate()
                        .find_map(|(index, op)| match op {
                            Op::Input {
                                name: Some(name), ..
                            } if name == state => Some(NodeId(index as u32)),
                            _ => None,
                        })
                }
                _ => None,
            })
            .collect();
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        let ssm_state_buffers: Vec<Option<(PlacedBuffer, PlacedBuffer)>> = layer_row_widths
            .iter()
            .map(
                |widths| -> Result<Option<(PlacedBuffer, PlacedBuffer)>, InteropError> {
                    if !ssm_placement_enabled {
                        return Ok(None);
                    }
                    match widths {
                        LayerPadRowWidths::Ssm { state_len, .. } => {
                            let byte_length = state_len * core::mem::size_of::<f32>();
                            let input = allocate_placed_buffer(byte_length)?;
                            let output = allocate_placed_buffer(byte_length)?;
                            Ok(Some((input, output)))
                        }
                        _ => Ok(None),
                    }
                },
            )
            .collect::<Result<_, _>>()?;
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        for (buffer, widths) in ssm_state_buffers.iter().zip(&layer_row_widths) {
            if let (Some((input, output)), LayerPadRowWidths::Ssm { state_len, .. }) =
                (buffer, widths)
            {
                let byte_length = state_len * core::mem::size_of::<f32>();
                omega::metal::zero_placed_buffer(input, byte_length);
                omega::metal::zero_placed_buffer(output, byte_length);
            }
        }

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
        let mut cached_len = seed_cached_len;
        let mut next_ids = ids.clone();
        let vocab_size = self.architecture.vocab as usize;
        // Reused across every step ([`Architecture::step_inputs`]'s own
        // doc) -- cleared, never reallocated from scratch, at the top of
        // each closure invocation below.
        let mut step_input_scratch: Vec<StepInput> = Vec::new();
        // qwen35 GDN prefill is necessarily sequential by position, but its
        // routed graph partitions are invariant throughout a KV bucket.
        // Reuse them across prompt positions instead of repeating partition,
        // topological-order, and shape-inference work for every token.
        let mut qwen35moe_pre_gather_plan: Option<Qwen35MoePreGatherPlan> = None;

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
                // ROW 427: a `single_position_step` architecture (qwen35's
                // GDN mixer -- `TensorError::SingleTokenStepOnly`'s own doc)
                // refuses any `new_count != 1` bind, so a `new_count > 1`
                // prefill (the whole prompt fed as one batched `next_ids`)
                // cannot go through this step's evaluate call at all. Split
                // it into one length-1 batch per prompt position instead --
                // every batch below runs through the SAME evaluate + cache-
                // append body as an ordinary decode step, just with
                // `cached_len` advancing by one per batch rather than by
                // `next_ids.len()` in a single call. `step_batches` is a
                // single `next_ids`-sized batch (byte-identical to the
                // pre-existing single-call path) for every dense/decode
                // caller and for a qwen35 step that already carries exactly
                // one position (ordinary decode, or a one-token prompt).
                let split_prefill = self.single_position_step && next_ids.len() > 1;
                let batch_count = if split_prefill { next_ids.len() } else { 1 };
                let last_batch_index = batch_count - 1;
                let mut token_id: u32 = 0;
                for batch_index in 0..batch_count {
                    let ids_for_step: &[u32] = if split_prefill {
                        core::slice::from_ref(&next_ids[batch_index])
                    } else {
                        next_ids.as_slice()
                    };
                    let is_last_step_batch = batch_index == last_batch_index;
                    #[cfg(feature = "instrument")]
                    proxima_tensor::instrument::reset_step();
                    #[cfg(feature = "instrument")]
                    let step_started = read_ticks();

                    let new_count = ids_for_step.len();
                    #[cfg(feature = "instrument")]
                    let apply_serving_config_started = read_ticks();
                    apply_serving_config(serving_config, cached_len + new_count)?;
                    #[cfg(feature = "instrument")]
                    let apply_serving_config_ticks = elapsed_ticks(apply_serving_config_started);

                    #[cfg(feature = "instrument")]
                    let build_position_inputs_started = read_ticks();
                    let inputs = build_position_inputs(
                        ids_for_step,
                        cached_len,
                        self.architecture.head_dim,
                        self.architecture.rope_freq_base,
                        self.architecture.rms_epsilon,
                        self.architecture_impl
                            .as_ref()
                            .is_some_and(|architecture| architecture.name() == "qwen35moe"),
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
                    for (name, data) in &self.weights.owned {
                        named_blocks
                            .push((name.as_str(), QuantizedBlock::Float32(data.as_slice())));
                    }
                    for (name, block) in &self.weights.packed {
                        named_blocks.push((name.as_str(), *block));
                    }
                    for (name, bytes, kind) in &self.weights.packed_owned {
                        named_blocks.push((name.as_str(), kind.as_block(bytes)));
                    }
                    // Rounds `cached_len` up to `ServingConfig::kv_bucket_tokens`
                    // (`kv_extent`'s own doc) -- `usize::MAX` in place of the
                    // placed-KV path's fixed buffer capacity: the two-range KV
                    // cache below is a growing `Vec`, not a preallocated
                    // device buffer, so there is no hard cap to clamp against.
                    let kv_bound_extent = kv_extent(
                        cached_len + new_count,
                        usize::MAX,
                        serving_config.kv_bucket_tokens,
                    );
                    // `mistral_cached_forward_program_with_experts`'s own
                    // `cached_len` `Op::Input` -- always present regardless of
                    // `ServingConfig::kv_bucket_tokens` (`proxima_tensor::bind::
                    // cached_attention_candidates`'s own doc on the runtime bound
                    // that reads it), so this scalar is fed on every step,
                    // bucketed or not.
                    let cached_len_scalar = [cached_len as f32];
                    // `mistral_cached_forward_program_with_experts_and_layer_taps`'s
                    // own `lm_head_row` `Op::Input` -- the last row of THIS
                    // step's `new_count` freshly-computed rows, host-supplied
                    // because the gather it feeds is a data-dependent index
                    // (`spec.rs`'s own doc on that leaf: an in-graph-computed
                    // index is a named `NotLowerable` gap on the typed
                    // evaluator, not a silently-guessed execution path).
                    let lm_head_row_scalar = [(new_count - 1) as f32];

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
                    let ssm_state_transfer_bytes: u64 = layer_caches
                        .iter()
                        .filter_map(|cache| match cache {
                            LayerCacheState::Ssm(cache) => {
                                Some((cache.state.len() * core::mem::size_of::<f32>()) as u64)
                            }
                            _ => None,
                        })
                        .sum();
                    #[cfg(feature = "instrument")]
                    let named_blocks_kv_started = read_ticks();
                    // `token_history` already carries exactly `cached_len +
                    // new_count` entries at this point (the same invariant
                    // `recent_tokens`'s repeat-penalty slice below relies on).
                    // The SAME assembly [`Self::forward_node_values_on_backend`]
                    // calls -- position/RoPE inputs, `cached_len`/`lm_head_row`,
                    // `Architecture::step_inputs`' own leaves, and every KV/SSM
                    // cache leaf -- so a foreign architecture's own leaf names
                    // are fed identically whether decoding or tapping one node.
                    let symbols = self.push_step_named_blocks(
                        &inputs,
                        &cached_len_scalar,
                        &lm_head_row_scalar,
                        &token_history,
                        cached_len,
                        new_count,
                        &cache_names,
                        &layer_caches,
                        &layer_row_widths,
                        kv_bound_extent,
                        &mut kv_pad_scratch,
                        &mut qwen35_dense_pad_scratch,
                        &mut step_input_scratch,
                        &mut named_blocks,
                        self.single_position_step,
                    )?;
                    #[cfg(feature = "instrument")]
                    let named_blocks_weights_ticks = elapsed_ticks(named_blocks_weights_started);
                    #[cfg(feature = "instrument")]
                    let named_blocks_kv_ticks = elapsed_ticks(named_blocks_kv_started);

                    let mut roots: Vec<NodeId> = Vec::with_capacity(1 + self.layer_roots.len() * 3);
                    if step_batch_needs_logits(split_prefill, is_last_step_batch) {
                        roots.push(self.logits_root);
                    }
                    roots.extend_from_slice(node_values_sink.nodes());
                    for (_layer, roots_for_layer) in self.layer_roots.iter().enumerate() {
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
                                #[cfg(all(
                                    feature = "metal-output-placement",
                                    target_os = "macos"
                                ))]
                                let state_is_placed = ssm_placement_enabled
                                    && ssm_placement_max_layer
                                        .is_none_or(|maximum| _layer <= maximum)
                                    && ssm_state_buffers[_layer].is_some();
                                #[cfg(not(all(
                                    feature = "metal-output-placement",
                                    target_os = "macos"
                                )))]
                                let state_is_placed = false;
                                if !state_is_placed {
                                    roots.push(*state_out);
                                }
                            }
                        }
                    }
                    // Only requested when a routing observer is actually
                    // registered (`instrument::expert_observer`'s own doc): the
                    // CPU evaluator keeps every requested output's full lifetime
                    // alive, so a program with no observer never pays to hold
                    // these nodes live. `proxima_tensor::instrument` itself is
                    // `instrument`-feature-gated (`proxima-tensor/src/lib.rs`),
                    // so a build with `std` but not `instrument` never observes
                    // routing at all -- `observe_routing` is a compile-time
                    // `false` there, not a call into a module that does not
                    // exist.
                    #[cfg(feature = "instrument")]
                    let observe_routing = proxima_tensor::instrument::expert_observer().is_some()
                        && !self.moe_sites.0.is_empty();
                    #[cfg(not(feature = "instrument"))]
                    let observe_routing = false;
                    if observe_routing {
                        for site in &self.moe_sites.0 {
                            roots.extend(site.selected.iter().copied());
                            roots.extend(site.weights.iter().copied());
                        }
                    }

                    if let Some(name) =
                        missing_program_input(&self.program, &named_blocks).filter(|name| {
                            !(serving_config.qwen35moe_pre_gather
                                && self
                                    .architecture_impl
                                    .is_some_and(|architecture| architecture.name() == "qwen35moe")
                                && name.contains("_exps.weight"))
                        })
                    {
                        return Err(InteropError::MissingStepInput { name });
                    }

                    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                    let mut ssm_input_placements: Vec<(
                        NodeId,
                        &PlacedBuffer,
                        usize,
                    )> = Vec::new();
                    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                    let mut ssm_output_placements: Vec<(
                        NodeId,
                        &PlacedBuffer,
                        usize,
                    )> = Vec::new();
                    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                    for (layer, roots_for_layer) in self.layer_roots.iter().enumerate() {
                        if ssm_placement_enabled
                            && ssm_placement_max_layer.is_none_or(|maximum| layer <= maximum)
                            && let (
                                Qwen35LayerRoots::Ssm { state_out, .. },
                                Some(state_input),
                                Some((input_buffer, output_buffer)),
                            ) = (
                                roots_for_layer,
                                ssm_state_input_nodes[layer],
                                ssm_state_buffers[layer].as_ref(),
                            )
                        {
                            let (input_buffer, output_buffer) = if cached_len % 2 == 0 {
                                (input_buffer, output_buffer)
                            } else {
                                (output_buffer, input_buffer)
                            };
                            ssm_input_placements.push((state_input, input_buffer, 0));
                            ssm_output_placements.push((*state_out, output_buffer, 0));
                        }
                    }

                    // Keep the residency boundary mutable through graph-input
                    // preparation. Freeze it only immediately before the
                    // evaluator borrows the expert-source table.
                    let _end_step_on_drop = begin_expert_gather_phase(&self.expert_slab);
                    // Dropped before `_end_step_on_drop` (reverse declaration
                    // order), so the source borrows end before the boundary is
                    // reopened even on an early `?` return.
                    let mut expert_slab_guard = lock_expert_slab(&self.expert_slab);

                    let pre_gather = serving_config.qwen35moe_pre_gather
                        && self
                            .architecture_impl
                            .is_some_and(|architecture| architecture.name() == "qwen35moe")
                        && (self.expert_sidecar.is_some() || !runtime.uses_gpu());
                    if pre_gather {
                        expert_slab_guard.clear_selected_experts_for_step();
                    }

                    // This gather's own [`proxima_tensor::cpu::ExpertSource`]
                    // snapshot -- built after the residency boundary and read by
                    // `run_reduce_with_quantized_weights` under the SAME weight
                    // `NodeId` [`crate::bind::build_expert_slab`] bound it under,
                    // so a dense checkpoint's empty slab costs one `BTreeMap`
                    // miss per gathered reduce and changes nothing else.
                    let mut expert_entries_scratch: Vec<(
                        NodeId,
                        Vec<proxima_tensor::cpu::ExpertEntry<'_>>,
                    )> = Vec::new();
                    let expert_sources =
                        expert_slab_guard.sources_for_step(&mut expert_entries_scratch)?;

                    // The packed checkpoint views carry the expert input's
                    // shape and codec through plan resolution. They remain
                    // borrowed mmap ranges: Metal's expert-source executor
                    // excludes every substituted node from ordinary uploads,
                    // then binds only the routed payload and descriptor tables.
                    if std::env::var_os("PROXIMA_DEBUG_QWEN35_SEGMENTS").is_some() {
                        eprintln!(
                            "qwen35 pre_gather_enabled={pre_gather} sidecar={} gpu={}",
                            self.expert_sidecar.is_some(),
                            runtime.uses_gpu()
                        );
                    }
                    if pre_gather
                        && qwen35moe_pre_gather_plan
                            .as_ref()
                            .is_none_or(|plan| plan.symbols != symbols)
                    {
                        qwen35moe_pre_gather_plan = Some(self.qwen35moe_pre_gather_plan(&symbols)?);
                    }
                    // The pristine table aliases the named checkpoint stack and
                    // needs no substitution. Once a policy pages or evicts any
                    // expert, pass the table across omega's backend boundary;
                    // CPU consumes it and Metal fails closed until its packed
                    // gather has a per-expert address-table binding.
                    #[cfg(feature = "metal")]
                    let expert_source_substitutions = if runtime.uses_gpu() {
                        None
                    } else {
                        Some(&expert_sources)
                    };
                    #[cfg(not(feature = "metal"))]
                    let expert_source_substitutions = Some(&expert_sources);

                    let mut before_qwen35moe_gather =
                        |layer: usize,
                         position: u64,
                         routes: &[crate::residency::RoutedExpert],
                         expert_slab: &mut crate::expert_slab::ExpertSlab<'file>|
                         -> Result<(), InteropError> {
                            let mut selected_experts = [0_u32; 16];
                            if routes.len() > selected_experts.len() {
                                return Err(InteropError::PreGatherExecutionUnsupported {
                                    architecture: String::from("qwen35moe"),
                                    reason: String::from(
                                        "router selected more experts than the fixed staging bound",
                                    ),
                                });
                            }
                            for (index, route) in routes.iter().enumerate() {
                                selected_experts[index] = route.expert as u32;
                            }
                            if std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some() {
                                eprintln!(
                                    "qwen35 route layer={} position={} experts={:?}",
                                    layer,
                                    position,
                                    &selected_experts[..routes.len()]
                                );
                            }
                            expert_slab
                                .add_selected_experts(layer, &selected_experts[..routes.len()]);
                            if let Some(policy) = qwen35moe_residency.as_mut() {
                                // Observe the complete route set before one
                                // reconciliation. Paging once per expert
                                // made one gather pay the FSM transition cost
                                // repeatedly and could churn the same slab
                                // entries before the kernel began.
                                for route in routes {
                                    policy.observe(position, layer, [*route]).map_err(|error| {
                                        InteropError::PreGatherExecutionUnsupported {
                                            architecture: String::from("qwen35moe"),
                                            reason: error.to_string(),
                                        }
                                    })?;
                                }
                                let actions = policy.reconcile::<256>().map_err(|error| {
                                    InteropError::PreGatherExecutionUnsupported {
                                        architecture: String::from("qwen35moe"),
                                        reason: error.to_string(),
                                    }
                                })?;
                                let sidecar = self.expert_sidecar.as_ref().ok_or_else(|| {
                                    InteropError::PreGatherExecutionUnsupported {
                                        architecture: String::from("qwen35moe"),
                                        reason: String::from("no expert sidecar is attached"),
                                    }
                                })?;
                                policy.apply_actions_at_boundary(
                                    expert_slab,
                                    &actions,
                                    |slab, action| {
                                        sidecar.apply_action(slab, self.checkpoint_mapping, action)
                                    },
                                )?;
                                if std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some() {
                                    eprintln!(
                                        "qwen35 residency boundary layer={} position={} actions={:?}",
                                        layer,
                                        position,
                                        actions.as_slice()
                                    );
                                }
                            }
                            Ok(())
                        };

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
                    #[cfg(all(
                        feature = "instrument",
                        feature = "metal",
                        target_os = "macos",
                        not(feature = "metal-output-placement")
                    ))]
                    let evaluated = match std::env::var("PROXIMA_METAL_OP_PROFILE_STEP")
                        .ok()
                        .and_then(|value| value.parse::<usize>().ok())
                    {
                        Some(target) if target == _step => {
                            if let Some(layer) = expert_slab_guard.first_modified_layer() {
                                return Err(InteropError::ExpertRoutingUnsupportedByBackend {
                                    layer,
                                });
                            }
                            let (evaluated, timings) = runtime.evaluate_op_timed(
                                &self.program,
                                &symbols,
                                &named_blocks,
                                &roots,
                                &resident_names,
                            )?;
                            report_op_timings(_step, &timings, &self.program);
                            evaluated
                        }
                        _ => runtime.evaluate(
                            &self.program,
                            &symbols,
                            &named_blocks,
                            &roots,
                            &resident_names,
                            expert_source_substitutions,
                        )?,
                    };
                    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                    let evaluated = if pre_gather {
                        let pre_gather_plan =
                            qwen35moe_pre_gather_plan.as_ref().ok_or_else(|| {
                                InteropError::PreGatherExecutionUnsupported {
                                    architecture: String::from("qwen35moe"),
                                    reason: String::from(
                                        "the routed segment plan was not prepared",
                                    ),
                                }
                            })?;
                        self.evaluate_qwen35moe_pre_gather(
                            runtime,
                            pre_gather_plan,
                            &symbols,
                            // planning still needs the original expert input
                            // metadata; execution replaces those bindings
                            // with the selected source table before staging.
                            &named_blocks,
                            &roots,
                            &resident_names,
                            &mut expert_slab_guard,
                            cached_len,
                            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                            Some(&Qwen35SsmPlacement {
                                input_nodes: &ssm_state_input_nodes,
                                buffers: &ssm_state_buffers,
                                maximum_layer: ssm_placement_max_layer,
                                use_second_as_input: cached_len % 2 != 0,
                            }),
                            &mut before_qwen35moe_gather,
                        )?
                    } else if use_metal_output_placements(
                        !ssm_input_placements.is_empty(),
                        expert_source_substitutions.is_some(),
                    ) {
                        runtime.evaluate_with_placements(
                            &self.program,
                            &symbols,
                            &named_blocks,
                            &roots,
                            &resident_names,
                            &ssm_input_placements,
                            &ssm_output_placements,
                        )?
                    } else {
                        runtime.evaluate(
                            &self.program,
                            &symbols,
                            &named_blocks,
                            &roots,
                            &resident_names,
                            expert_source_substitutions,
                        )?
                    };
                    #[cfg(not(all(feature = "metal-output-placement", target_os = "macos")))]
                    let evaluated = if pre_gather {
                        let pre_gather_plan =
                            qwen35moe_pre_gather_plan.as_ref().ok_or_else(|| {
                                InteropError::PreGatherExecutionUnsupported {
                                    architecture: String::from("qwen35moe"),
                                    reason: String::from(
                                        "the routed segment plan was not prepared",
                                    ),
                                }
                            })?;
                        self.evaluate_qwen35moe_pre_gather(
                            runtime,
                            pre_gather_plan,
                            &symbols,
                            // planning needs the original expert descriptors;
                            // execution substitutes the selected source table.
                            &named_blocks,
                            &roots,
                            &resident_names,
                            &mut expert_slab_guard,
                            cached_len,
                            &mut before_qwen35moe_gather,
                        )?
                    } else {
                        runtime.evaluate(
                            &self.program,
                            &symbols,
                            &named_blocks,
                            &roots,
                            &resident_names,
                            expert_source_substitutions,
                        )?
                    };
                    #[cfg(feature = "instrument")]
                    let evaluate_ticks = elapsed_ticks(evaluate_started);
                    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                    let metal_stage = metal_stage_totals();

                    node_values_sink.observe(&evaluated)?;

                    // One `ExpertRouting` event per layer per new position --
                    // `proxima_tensor::instrument::ExpertObserver`'s own doc on
                    // why this is the decode loop's job, not the kernel's:
                    // `evaluated.get` reads back exactly the extra outputs
                    // `observe_routing` requested above, never the kernel's own
                    // per-position gather.
                    #[cfg(feature = "instrument")]
                    if observe_routing {
                        for site in &self.moe_sites.0 {
                            let weight_total_node =
                                site.weights.last().copied().unwrap_or(self.logits_root);
                            let Some((weight_total, _)) = evaluated.get(weight_total_node) else {
                                continue;
                            };
                            for local in 0..new_count {
                                let experts: Vec<u32> = site
                                    .selected
                                    .iter()
                                    .filter_map(|node| evaluated.get(*node))
                                    .filter_map(|(values, _)| values.get(local).copied())
                                    .map(|value| value as u32)
                                    .collect();
                                let Some(&total) = weight_total.get(local) else {
                                    continue;
                                };
                                let weights: Vec<f32> = site
                                    .weights
                                    .iter()
                                    .take(site.weights.len().saturating_sub(1))
                                    .filter_map(|node| evaluated.get(*node))
                                    .filter_map(|(values, _)| values.get(local).copied())
                                    .map(|value| value / total)
                                    .collect();
                                if experts.len() != weights.len() {
                                    continue;
                                }
                                let event = proxima_tensor::instrument::ExpertRouting {
                                    layer: site.layer,
                                    position: (cached_len + local) as u64,
                                    experts: &experts,
                                    weights: &weights,
                                };
                                proxima_tensor::instrument::notify_expert_routed(&event);
                            }
                        }
                    }

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
                    // History-carry bisection step 3 (diag/qwen35moe-history-carry):
                    // before-state for layers 0 (GDN) and 3 (this checkpoint's
                    // first attention layer, 3-GDN-to-1-attention interleave) --
                    // paired with the AFTER checksum below to prove whether this
                    // step's cache append actually mutated either layer's state.
                    // `.get` rather than a literal index: a foreign `Architecture`
                    // with an empty `layer_roots` (no cache leaves at all) has an
                    // empty `layer_caches` too, and this diagnostic must degrade
                    // to "nothing to report" rather than index out of bounds.
                    #[cfg(feature = "instrument")]
                    let (layer0_before_len, layer0_before_checksum) =
                        layer_caches.first().map_or((0, 0.0), layer_cache_checksum);
                    #[cfg(feature = "instrument")]
                    let (layer3_before_len, layer3_before_checksum) =
                        layer_caches.get(3).map_or((0, 0.0), layer_cache_checksum);
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
                                        (even_data.len() + odd_data.len() + value_data.len())
                                            as u64;
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
                                let (qkv_mixed_data, _) = evaluated.get(*qkv_mixed).ok_or(
                                    InteropError::MissingEvaluatedNode { node: *qkv_mixed },
                                )?;
                                #[cfg(all(
                                    feature = "metal-output-placement",
                                    target_os = "macos"
                                ))]
                                let state_is_placed = ssm_output_placements
                                    .iter()
                                    .any(|(node, _, _)| node == state_out);
                                #[cfg(not(all(
                                    feature = "metal-output-placement",
                                    target_os = "macos"
                                )))]
                                let state_is_placed = false;
                                let state_out_data = if state_is_placed {
                                    None
                                } else {
                                    Some(
                                        evaluated
                                            .get(*state_out)
                                            .ok_or(InteropError::MissingEvaluatedNode {
                                                node: *state_out,
                                            })?
                                            .0,
                                    )
                                };
                                #[cfg(feature = "instrument")]
                                {
                                    layer_cache_append_elements += qkv_mixed_data.len() as u64
                                        + state_out_data.map_or(0, |data| data.len() as u64);
                                    debug!(
                                        layer = layer as u64,
                                        qkv_mixed_elements = qkv_mixed_data.len() as u64,
                                        state_elements =
                                            state_out_data.map_or(0, |data| data.len() as u64),
                                        state_bytes = state_out_data.map_or(0, |data| {
                                            (data.len() * core::mem::size_of::<f32>()) as u64
                                        }),
                                        state_is_placed,
                                        "ssm_state_host_transfer: recurrent output placement"
                                    );
                                }
                                // `layer_row_widths[layer]`'s own `Ssm` arm --
                                // the program's own declared
                                // `ssm_cache.{layer}.conv_history` shape, the
                                // SAME source [`fresh_layer_caches`] sized this
                                // cache's initial window from, never
                                // `Architecture::step_state` (that hook's `None`
                                // default is exactly the real-world defect this
                                // read used to reproduce on a foreign
                                // architecture).
                                let conv_history_len = match &layer_row_widths[layer] {
                                    LayerPadRowWidths::Ssm {
                                        conv_history_len, ..
                                    } => *conv_history_len,
                                    _ => unreachable!(
                                        "layer_row_widths built from the same layer_roots, in lockstep"
                                    ),
                                };
                                if let Some(state_out_data) = state_out_data {
                                    cache.advance(qkv_mixed_data, state_out_data, conv_history_len);
                                } else {
                                    cache.advance_conv_history(qkv_mixed_data, conv_history_len);
                                }
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
                    #[cfg(feature = "instrument")]
                    {
                        let (layer0_after_len, layer0_after_checksum) =
                            layer_caches.first().map_or((0, 0.0), layer_cache_checksum);
                        let (layer3_after_len, layer3_after_checksum) =
                            layer_caches.get(3).map_or((0, 0.0), layer_cache_checksum);
                        debug!(
                            step = _step as u64,
                            batch_index = batch_index as u64,
                            token_fed = ids_for_step[0],
                            cached_len_before = cached_len_before_step as u64,
                            layer0_len_before = layer0_before_len as u64,
                            layer0_len_after = layer0_after_len as u64,
                            layer0_checksum_before = layer0_before_checksum,
                            layer0_checksum_after = layer0_after_checksum,
                            layer3_len_before = layer3_before_len as u64,
                            layer3_len_after = layer3_after_len as u64,
                            layer3_checksum_before = layer3_before_checksum,
                            layer3_checksum_after = layer3_after_checksum,
                            "decode_loop_step_trace: layer 0/3 cache state before/after this step's append"
                        );
                    }
                    cached_len += new_count;

                    // Everything below samples a token off THIS batch's logits.
                    // For a `single_position_step` architecture's expanded
                    // prefill (`step_batches` above), every batch except the
                    // last is a known prompt token, not a sampled one -- only
                    // the cache-append above needs to run for it. Running this
                    // tail on every batch would draw from `rng` once per prompt
                    // position instead of once per generated token, diverging
                    // from the sequential single-position oracle the ROW 427
                    // tests compare against (`RE-VERIFY: rng` in that doc's own
                    // row). Skipping it here is what makes `next_ids`'s LAST
                    // batch's logits the ones `decode_until_stop_or_budget`
                    // actually samples, exactly like a `new_count == 1` decode
                    // step always has.
                    if is_last_step_batch {
                        let (logits, _shape) = evaluated.get(self.logits_root).ok_or(
                            InteropError::MissingEvaluatedNode {
                                node: self.logits_root,
                            },
                        )?;
                        // `logits_root` must be the `lm_head_row`-gathered LAST row
                        // only (`crate::architecture`'s doc on `BoundProgram::logits_root`)
                        // -- exactly one row of `vocab_size`. A foreign `Architecture`
                        // that hands back the full `[new_count, vocab]` buffer is
                        // rejected here rather than silently sampled at row 0.
                        if logits.len() != vocab_size {
                            return Err(InteropError::LogitsShapeMismatch {
                                expected_rows: 1,
                                found_rows: logits.len() / vocab_size,
                                vocab: vocab_size,
                            });
                        }
                        let last_position = &logits[..vocab_size];
                        #[cfg(feature = "instrument")]
                        {
                            let (argmax_token, argmax_logit) = last_position
                                .iter()
                                .copied()
                                .enumerate()
                                .max_by(|left, right| left.1.total_cmp(&right.1))
                                .unwrap_or((0, f32::NEG_INFINITY));
                            debug!(
                                step = _step as u64,
                                batch_index = batch_index as u64,
                                cached_len = cached_len as u64,
                                argmax_token = argmax_token as u64,
                                argmax_logit,
                                "decode_loop_step_trace: logits before sampling"
                            );
                        }
                        #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
                        let barriers_step = metal_stage.barriers_emitted;
                        #[cfg(not(all(
                            feature = "instrument",
                            feature = "metal",
                            target_os = "macos"
                        )))]
                        let barriers_step = 0_u64;
                        logits_sink.observe(last_position, barriers_step);

                        #[cfg(feature = "instrument")]
                        let greedy_pick_started = read_ticks();
                        token_id = match token_override.and_then(|forced| forced.get(_step)) {
                            Some(&forced_token) => forced_token,
                            None => {
                                let recent_window_start =
                                    token_history.len().saturating_sub(repeat_window);
                                let recent_tokens = &token_history[recent_window_start..];
                                sample_next_token(
                                    last_position,
                                    recent_tokens,
                                    sample_config,
                                    &mut rng,
                                )
                                .ok_or(InteropError::EmptyLogits)?
                            }
                        };
                        token_history.push(token_id);
                        #[cfg(feature = "instrument")]
                        let greedy_pick_ticks = elapsed_ticks(greedy_pick_started);
                        #[cfg(feature = "instrument")]
                        debug!(
                            step = _step as u64,
                            cached_len_after = (cached_len_before_step + new_count) as u64,
                            token_sampled = token_id,
                            "decode_loop_step_trace: sampled token fed forward as next step's next_ids"
                        );
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
                                ssm_state_transfer_bytes,
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
                            let attribution =
                                proxima_tensor::instrument::cohort_leader_attribution();
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
                                cached_attention_ops = proxima_tensor::instrument::path_totals()
                                    .op_kind_cached_attention,
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
                    }
                }

                Ok(token_id)
            },
            on_token,
        )?;

        let text = proxima_tokenizer::decode(&generated_ids, &self.vocab)?;
        // Exactly the tokens `cached_len` now covers: `seed_ids ++ ids`
        // (this call's own new range) plus however many of its OWN
        // generated tokens have themselves been forward-passed since --
        // every generated token except the last is (the last is only
        // just-sampled, never yet fed back through `evaluate`). Derived
        // from `cached_len` itself rather than re-counting loop iterations,
        // so it is correct on every exit path `decode_until_stop_or_budget`
        // has (`max_tokens` exhaustion, model EOS, `Control::Stop` from
        // either phase) without special-casing any of them.
        let mut final_ids = seed_ids;
        final_ids.extend_from_slice(&ids);
        let forwarded_generated = cached_len
            .saturating_sub(final_ids.len())
            .min(generated_ids.len());
        final_ids.extend_from_slice(&generated_ids[..forwarded_generated]);
        let final_state = PrefixState {
            ids: final_ids,
            layer_caches,
            cached_len,
        };
        Ok((generated_ids, text, stopped_by_eos, final_state))
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
                    self.architecture_impl
                        .as_ref()
                        .is_some_and(|architecture| architecture.name() == "qwen35moe"),
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
                named_blocks.push(("ids", QuantizedBlock::Int32(inputs.ids_i32.as_slice())));
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
                // See the sibling decode loop's own comment on
                // `lm_head_row` above -- same leaf, same host-supplied
                // reason, this step's own last new row.
                let lm_head_row_scalar = [(new_count - 1) as f32];
                named_blocks.push(("lm_head_row", QuantizedBlock::Float32(&lm_head_row_scalar)));
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
                let kv_bound_extent = kv_extent(
                    merged_len,
                    positions_needed,
                    serving_config.kv_bucket_tokens,
                );
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
                        report_op_timings(_step, &timings, &single_range.program);
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
                        report_op_timings(_step, &timings, &single_range.program);
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
                // `logits_root` must be the `lm_head_row`-gathered LAST row
                // only (`crate::architecture`'s doc on `BoundProgram::logits_root`)
                // -- exactly one row of `vocab_size`. A foreign `Architecture`
                // that hands back the full `[new_count, vocab]` buffer is
                // rejected here rather than silently sampled at row 0.
                if logits.len() != vocab_size {
                    return Err(InteropError::LogitsShapeMismatch {
                        expected_rows: 1,
                        found_rows: logits.len() / vocab_size,
                        vocab: vocab_size,
                    });
                }
                let last_position = &logits[..vocab_size];
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
                        ssm_state_transfer_bytes: 0,
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
                        let kv_cache_device_bytes =
                            (block_count * (2 * capacity_even_odd + capacity_v)) as u64;
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

    /// Captures `node_ids` after every stateful prompt-position evaluation
    /// through the ordinary cached decode loop. This is the causal companion
    /// to [`Self::forward_node_values_on_backend`], whose cache is always empty.
    pub fn forward_cached_node_values_on_backend(
        &self,
        prompt: &str,
        node_ids: &[NodeId],
        gpu_layers: i32,
    ) -> Result<Vec<Vec<Vec<f32>>>, InteropError> {
        let serving_config = supported_serving_config(
            gpu_layers,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            omega::MathMode::default(),
        );
        let mut runtime = BackendRuntime::new(&serving_config);
        let mut steps = Vec::new();
        let mut node_values_sink = NodeValuesSink::Collect {
            nodes: node_ids,
            steps: &mut steps,
        };
        let _ = self.run_decode_loop_observed_seeded(
            prompt,
            1,
            &serving_config,
            &mut runtime,
            None,
            &mut LogitsSink::Discard,
            &mut node_values_sink,
            &mut |_event| Control::Continue,
            None,
            true,
        )?;
        Ok(steps)
    }

    /// [`Self::forward_node_values`] with the backend left open --
    /// `gpu_layers` reaches `BackendRuntime::new`/`select_backend` the
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
            self.architecture_impl
                .as_ref()
                .is_some_and(|architecture| architecture.name() == "qwen35moe"),
        );

        // The SAME program-derived cache-leaf-name/step_inputs assembly
        // `Self::run_decode_loop_observed_seeded` calls -- before this,
        // this method hard-coded `kv_cache.{layer}.{k_even,k_odd,v}` and
        // never ran `Architecture::step_inputs` at all, so a foreign
        // architecture with differently-named cache leaves (or a leaf only
        // `step_inputs` feeds) surfaced `InteropError::UnboundInputName`
        // the moment a caller tapped an interior node here instead of
        // decoding. `cached_len: 0`, `new_start: 0`, `new_count:
        // ids.len()` -- this is always a one-shot forward from an empty
        // cache over the WHOLE prompt (this method's own doc).
        let (cache_names, layer_row_widths) = self.declared_layer_cache_names_and_widths()?;
        let layer_caches = self.fresh_layer_caches(&cache_names, &layer_row_widths);
        let mut kv_pad_scratch: Vec<KvPadScratch> = self
            .layer_roots
            .iter()
            .map(|_| KvPadScratch::new())
            .collect();
        let mut qwen35_dense_pad_scratch: Vec<Qwen35DenseAttentionPadScratch> = self
            .layer_roots
            .iter()
            .map(|_| Qwen35DenseAttentionPadScratch::new())
            .collect();
        let mut step_input_scratch: Vec<StepInput> = Vec::new();

        let mut named_blocks: Vec<(&str, QuantizedBlock)> = Vec::with_capacity(
            self.weights.owned.len()
                + self.weights.packed.len()
                + self.weights.packed_owned.len()
                + 6
                + cache_names.len() * 4,
        );
        for (name, data) in &self.weights.owned {
            named_blocks.push((name.as_str(), QuantizedBlock::Float32(data.as_slice())));
        }
        for (name, block) in &self.weights.packed {
            named_blocks.push((name.as_str(), *block));
        }
        for (name, bytes, kind) in &self.weights.packed_owned {
            named_blocks.push((name.as_str(), kind.as_block(bytes)));
        }
        // `mistral_cached_forward_program_with_experts`'s own `cached_len`
        // `Op::Input` (ROW 404/405's runtime bound the fused Metal
        // `CachedAttention` kernel reads) is present on every program this
        // method evaluates, fresh-KV or not -- this is always a one-shot
        // forward from an empty cache (this method's own doc), so `0.0` is
        // the only correct value.
        let cached_len_scalar = [0.0f32];
        // Same `lm_head_row` leaf the decode loop feeds -- this one-shot
        // forward's own `ids` IS the whole prompt, so `ids.len() - 1` is
        // its last row, matching `Self::forward_logits_on_backend`'s own
        // "last prompt position" doc. A caller of
        // `Self::forward_node_values_on_backend` wanting the FULL
        // per-position logits (not just the last row) has no opt-in path
        // yet -- residual, not fixed here.
        let lm_head_row_scalar = [(ids.len() - 1) as f32];
        let kv_bound_extent = kv_extent(ids.len(), usize::MAX, serving_config.kv_bucket_tokens);

        let symbols = self.push_step_named_blocks(
            &inputs,
            &cached_len_scalar,
            &lm_head_row_scalar,
            &ids,
            0,
            ids.len(),
            &cache_names,
            &layer_caches,
            &layer_row_widths,
            kv_bound_extent,
            &mut kv_pad_scratch,
            &mut qwen35_dense_pad_scratch,
            &mut step_input_scratch,
            &mut named_blocks,
            self.single_position_step,
        )?;

        let resident_names: BTreeSet<&str> = self.resident_names();

        // A one-shot diagnostic forward, not a decode step -- no
        // `ExpertSlab::begin_step`/`end_step` pair runs around it, so this
        // reads whatever is currently paged (this checkpoint's own aliased
        // stack, absent a caller ever paging one) exactly as
        // `run_reduce_with_quantized_weights` always has: `None` here is
        // not "experts disabled", it is "no per-step snapshot applies to a
        // call outside the decode loop".
        // The qwen35moe diagnostic can request an interior routed node, so
        // keep it on the partition-isolated seam. Other one-shot forwards
        // retain the ordinary evaluator and its normal cache bookkeeping.
        let evaluated = if self
            .architecture_impl
            .as_ref()
            .is_some_and(|architecture| architecture.name() == "qwen35moe")
        {
            runtime.evaluate_segment(
                &self.program,
                &symbols,
                &named_blocks,
                node_ids,
                &resident_names,
                None,
            )?
        } else {
            runtime.evaluate(
                &self.program,
                &symbols,
                &named_blocks,
                node_ids,
                &resident_names,
                None,
            )?
        };

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
        let prompt_ids = proxima_tokenizer::encode_with_bos_eos(
            prompt,
            &self.vocab,
            wants_bos(&self.vocab),
            self.vocab.add_eos_token().unwrap_or(false),
        )?;
        if self.single_position_step && prompt_ids.len() > 1 {
            let serving_config = supported_serving_config(
                gpu_layers,
                #[cfg(all(feature = "metal", target_os = "macos"))]
                omega::MathMode::default(),
            );
            let mut runtime = BackendRuntime::new(&serving_config);
            let mut captured = Vec::new();
            let mut logits_sink = LogitsSink::Collect(&mut captured);
            let mut on_token = |_event: TokenEvent<'_>| Control::Continue;
            self.run_decode_loop_observed(
                prompt,
                1,
                &serving_config,
                &mut runtime,
                None,
                &mut logits_sink,
                &mut on_token,
            )?;
            return captured.pop().ok_or(InteropError::EmptyLogits);
        }
        // `forward_node_values_on_backend` re-tokenizes `prompt` itself;
        // this binding survives only for the `instrument`-gated debug
        // event below (`prompt_tokens`) now that the last-row slice no
        // longer needs a token count to index with.
        #[cfg(feature = "instrument")]
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
        // `logits_root` is now the `lm_head_row`-gathered LAST row only
        // (`spec.rs`'s own doc on that leaf, fed `ids.len() - 1` above) --
        // one row of `vocab_size`, already the "last prompt position"
        // this method's own doc promises.
        let last_position = logits[..vocab_size].to_vec();

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

#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
fn use_metal_output_placements(
    has_recurrent_state: bool,
    has_expert_source_substitutions: bool,
) -> bool {
    has_recurrent_state && !has_expert_source_substitutions
}

#[cfg(all(test, feature = "std"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{
        SsmLayerCache, begin_expert_gather_phase, kv_extent, lock_expert_slab,
        step_batch_needs_logits, visit_qwen35moe_router_boundary,
        visit_qwen35moe_router_selections,
    };
    use alloc::string::String;
    use alloc::vec::Vec;

    #[cfg(feature = "metal")]
    use alloc::collections::BTreeMap;

    #[cfg(all(feature = "metal", target_os = "macos"))]
    use alloc::collections::BTreeSet;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use proxima_gguf::value::MetadataArray;
    use proxima_gguf::value::MetadataValue as Value;
    use proxima_gguf::{GgmlType as WireType, GgufModel, TensorPayload, write_complete};
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use proxima_tensor::cpu::QuantizedBlock;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use proxima_tensor::{DType, Extent, NodeId, Op};
    use proxima_tokenizer::Vocab;

    #[cfg(feature = "metal")]
    #[test]
    fn segment_plan_cache_distinguishes_kv_bucket_extent() {
        let mut cache = BTreeMap::new();
        let mut hits = 0;
        let mut misses = 0;
        let first = super::BackendRuntime::resolve_segment_plan(
            &mut cache,
            &mut hits,
            &mut misses,
            (17, 1, 64),
            || Ok::<_, super::InteropError>(11_u32),
        )
        .expect("first segment shape resolves");
        assert_eq!(*first, 11);
        let second = super::BackendRuntime::resolve_segment_plan(
            &mut cache,
            &mut hits,
            &mut misses,
            (17, 1, 128),
            || Ok::<_, super::InteropError>(22_u32),
        )
        .expect("new KV bucket resolves independently");
        assert_eq!(*second, 22);
        assert_eq!(hits, 0);
        assert_eq!(misses, 2);
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    use super::{BackendRuntime, LoadedModel};
    use super::{
        Control, DecodeMetrics, Phase, TokenEvent, build_position_inputs,
        decode_until_stop_or_budget,
    };
    use crate::bind::architecture_from_metadata;

    #[test]
    fn residency_mutation_closes_only_for_the_expert_gather_phase() {
        let checkpoint_expert = [0_u8; 144];
        let routed_expert = [7_u8; 144];
        let slab = std::sync::Mutex::new(crate::expert_slab::ExpertSlab::new());
        lock_expert_slab(&slab)
            .bind_layer_stack(
                0,
                proxima_tensor::op::NodeId(1),
                crate::bind::PackedOwnedKind::Q4K,
                &checkpoint_expert,
                1,
                32,
                32,
            )
            .expect("the routed layer binds before evaluation");

        lock_expert_slab(&slab)
            .page_expert(
                0,
                0,
                crate::bind::PackedOwnedKind::Q4K,
                &routed_expert,
                32,
                32,
            )
            .expect("the current route may change residency before gather");

        let gather_phase = begin_expert_gather_phase(&slab);
        let during_gather = lock_expert_slab(&slab).page_expert(
            0,
            0,
            crate::bind::PackedOwnedKind::Q4K,
            &checkpoint_expert,
            32,
            32,
        );
        assert!(matches!(
            during_gather,
            Err(crate::InteropError::ExpertSwapDuringStep {
                layer: 0,
                expert: 0
            })
        ));

        drop(gather_phase);
        lock_expert_slab(&slab)
            .page_expert(
                0,
                0,
                crate::bind::PackedOwnedKind::Q4K,
                &checkpoint_expert,
                32,
                32,
            )
            .expect("the next router boundary reopens after gather");
    }

    #[test]
    fn router_segment_visits_current_top_k_before_the_gather_boundary() {
        let logits = [0.5_f32, 7.0, 7.0, -1.0, 9.0, 1.0, 2.0, 8.0];
        let mut route_scratch = Vec::with_capacity(2);
        let mut visited = Vec::new();

        visit_qwen35moe_router_selections(
            3,
            41,
            &logits,
            &[2, 4],
            4,
            2,
            &mut route_scratch,
            &mut |layer, position, routes| {
                visited.push((
                    layer,
                    position,
                    routes.iter().map(|route| route.expert).collect::<Vec<_>>(),
                ));
                Ok(())
            },
        )
        .expect("the two real router rows expose their top-2 routes");

        assert_eq!(
            visited,
            [(3, 41, vec![1, 2]), (3, 42, vec![0, 3])],
            "the callback receives lower-index tie breaking and every position before its gather"
        );
        assert_eq!(
            route_scratch.capacity(),
            2,
            "the fixed top-k scratch does not grow while visiting router rows"
        );
    }

    #[test]
    fn router_boundary_pages_before_refreshing_the_gather_source() {
        let checkpoint_expert = [0_u8; 144];
        let routed_expert = [7_u8; 144];
        let weight_node = proxima_tensor::op::NodeId(1);
        let mut slab = crate::expert_slab::ExpertSlab::new();
        slab.bind_layer_stack(
            0,
            weight_node,
            crate::bind::PackedOwnedKind::Q4K,
            &checkpoint_expert,
            1,
            32,
            32,
        )
        .expect("the routed layer binds before evaluation");
        slab.begin_step();
        let mut route_scratch = Vec::with_capacity(1);

        visit_qwen35moe_router_boundary(
            0,
            9,
            &[3.0],
            &[1, 1],
            1,
            1,
            &mut route_scratch,
            &mut slab,
            &mut |layer, position, routes, slab| {
                assert_eq!((layer, position), (0, 9));
                assert_eq!(routes[0].expert, 0);
                slab.page_expert(
                    layer,
                    routes[0].expert,
                    crate::bind::PackedOwnedKind::Q4K,
                    &routed_expert,
                    32,
                    32,
                )?;
                Ok(())
            },
        )
        .expect("the residency callback runs while the gather boundary is open");

        let mutation_after_boundary = slab.page_expert(
            0,
            0,
            crate::bind::PackedOwnedKind::Q4K,
            &checkpoint_expert,
            32,
            32,
        );
        assert!(matches!(
            mutation_after_boundary,
            Err(crate::InteropError::ExpertSwapDuringStep {
                layer: 0,
                expert: 0
            })
        ));

        let mut entries = Vec::new();
        let sources = slab
            .sources_for_step(&mut entries)
            .expect("the post-boundary gather source is complete");
        let source = sources
            .get(&weight_node)
            .expect("the routed layer has a refreshed source");
        let entry = source
            .entries()
            .first()
            .expect("the selected expert remains at its stable index");
        assert_eq!(entry.epoch, 1, "the gather sees the boundary page");
        assert!(matches!(
            entry.block,
            proxima_tensor::cpu::QuantizedBlock::Q4K(bytes) if bytes == routed_expert
        ));
    }

    #[test]
    fn kv_bucket_extent_includes_the_new_position() {
        let cached_len = 0usize;
        let new_count = 1usize;
        assert_eq!(
            kv_extent(cached_len + new_count, usize::MAX, 32),
            32,
            "the first position must allocate a non-empty KV bucket"
        );
        assert_eq!(
            kv_extent(cached_len, usize::MAX, 32),
            0,
            "the pre-step cache length is not a valid attention extent"
        );
    }

    #[test]
    fn intermediate_split_prefill_batch_does_not_request_logits() {
        assert!(step_batch_needs_logits(false, false));
        assert!(step_batch_needs_logits(false, true));
        assert!(!step_batch_needs_logits(true, false));
        assert!(step_batch_needs_logits(true, true));
    }

    #[test]
    fn router_selection_callback_runs_before_gather() {
        let mut scratch = Vec::new();
        let mut observed = Vec::new();
        super::visit_qwen35moe_router_selections(
            2,
            11,
            &[0.1, 0.9, 0.2, 0.8],
            &[1, 4],
            4,
            2,
            &mut scratch,
            &mut |layer, position, routes| {
                observed.push((layer, position, routes[0].expert, routes[1].expert));
                Ok(())
            },
        )
        .expect("router callback receives one deterministic top-k row");
        assert_eq!(observed, vec![(2, 11, 1, 3)]);
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_router_segment_readback_reaches_the_residency_boundary() {
        let config = super::supported_serving_config(GPU_LAYERS_ALL, omega::MathMode::default());
        let mut runtime = BackendRuntime::new(&config);
        let router = NodeId(0);
        let program = [Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1), Extent::Static(4)],
            name: Some(String::from("router_logits")),
        }];
        let logits = [0.1_f32, 0.9, 0.2, 0.8];
        let named = [("router_logits", QuantizedBlock::Float32(&logits))];
        let evaluated = runtime
            .evaluate_segment(&program, &[1, 1], &named, &[router], &BTreeSet::new(), None)
            .expect("metal returns the requested router tensor to the host boundary");
        let (readback, shape) = evaluated
            .get(router)
            .expect("the requested router tensor is present after metal execution");
        let mut scratch = Vec::new();
        let mut observed = Vec::new();

        visit_qwen35moe_router_selections(
            2,
            11,
            readback,
            shape,
            4,
            2,
            &mut scratch,
            &mut |layer, position, routes| {
                observed.push((layer, position, routes[0].expert, routes[1].expert));
                Ok(())
            },
        )
        .expect("the metal router readback drives the typed residency callback");

        assert_eq!(observed, vec![(2, 11, 1, 3)]);
    }

    #[test]
    fn decode_metrics_uses_cumulative_token_event_time() {
        let events = [
            TokenEvent {
                token_id: 1,
                text_piece: "",
                phase: Phase::Prefill { prompt_tokens: 7 },
                step: 0,
                elapsed_ms: 400,
            },
            TokenEvent {
                token_id: 1,
                text_piece: "a",
                phase: Phase::Token,
                step: 0,
                elapsed_ms: 400,
            },
            TokenEvent {
                token_id: 2,
                text_piece: "b",
                phase: Phase::Token,
                step: 1,
                elapsed_ms: 900,
            },
        ];
        let metrics = DecodeMetrics::from_events(&events, 3_500, 42.5, 1, 0.125);

        assert_eq!(metrics.prompt_tokens, 7);
        assert_eq!(metrics.generated_tokens, 2);
        assert_eq!(metrics.elapsed_ms, 900);
        assert!((metrics.tokens_per_second - (2.0 / 0.9)).abs() < f64::EPSILON);
        assert!((metrics.per_token_latency_ms - 450.0).abs() < f64::EPSILON);
        assert_eq!(metrics.peak_rss_bytes, 3_500);
        assert_eq!(metrics.cpu_percent, 42.5);
        assert_eq!(metrics.error_count, 1);
        assert_eq!(metrics.covariance, 0.125);
    }

    #[test]
    fn decode_metrics_keeps_provenance_labels_explicit() {
        let metrics = DecodeMetrics::from_events(&[], 0, 0.0, 0, 0.0);

        assert_eq!(metrics.timing_source, "TokenEvent::elapsed_ms");
        assert_eq!(metrics.memory_source, "caller_peak_rss_bytes");
        assert_eq!(
            metrics.resource_source,
            "caller_cpu_percent_and_error_count"
        );
    }

    #[test]
    fn ssm_state_replacement_preserves_fixed_capacity() {
        let mut cache = SsmLayerCache::new(4, 3);
        let state_capacity = cache.state.capacity();
        cache.advance(&[1.0, 2.0], &[3.0, 4.0, 5.0], 4);
        assert_eq!(cache.state, vec![3.0, 4.0, 5.0]);
        assert_eq!(cache.state.capacity(), state_capacity);
        cache.advance(&[6.0, 7.0], &[8.0, 9.0, 10.0], 4);
        assert_eq!(cache.state, vec![8.0, 9.0, 10.0]);
        assert_eq!(cache.state.capacity(), state_capacity);
    }

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
            false,
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
        let expert_stack = f32_bytes(&vec![
            0.05f32;
            (expert_count * feed_forward * embedding) as usize
        ]);
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
                ("llama.attention.head_count".to_string(), Value::U32(1)),
                ("llama.attention.head_count_kv".to_string(), Value::U32(1)),
                ("llama.block_count".to_string(), Value::U32(1)),
                (
                    "llama.expert_count".to_string(),
                    Value::U32(expert_count as u32),
                ),
                ("llama.expert_used_count".to_string(), Value::U32(1)),
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
        let q_weight = f32_bytes(&vec![
            0.05f32;
            (embedding * query_heads * attn_head_dim * 2) as usize
        ]);
        let kv_weight = f32_bytes(&vec![
            0.05f32;
            (embedding * kv_heads * attn_head_dim) as usize
        ]);
        let output_weight = f32_bytes(&vec![
            0.05f32;
            (query_heads * attn_head_dim * embedding) as usize
        ]);
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
                ("qwen35.full_attention_interval".to_string(), Value::U32(1)),
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
        let q_weight = f32_bytes(&vec![
            0.05f32;
            (embedding * query_heads * attn_head_dim * 2) as usize
        ]);
        let kv_weight = f32_bytes(&vec![
            0.05f32;
            (embedding * kv_heads * attn_head_dim) as usize
        ]);
        let output_weight = f32_bytes(&vec![
            0.05f32;
            (query_heads * attn_head_dim * embedding) as usize
        ]);
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
                ("qwen35.full_attention_interval".to_string(), Value::U32(1)),
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

            let serving_config =
                super::super::supported_serving_config(GPU_LAYERS_ALL, omega::MathMode::default());
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
        let tokens: Vec<String> = (0..=255u8).map(|byte| format!("<0x{byte:02X}>")).collect();
        Vocab::new(tokens, &[], None, None, None).expect("minimal vocab builds")
    }

    fn tiny_architecture() -> ModelArchitecture {
        ModelArchitecture {
            vocab: 1,
            embedding: 1,
            feed_forward: 1,
            query_heads: 1,
            kv_heads: 2,
            kv_heads_by_layer: vec![2; 2],
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
            architecture_impl: None,
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
            hidden_root: None,
            layer_roots: Vec::new(),
            qwen35moe_layer_diagnostics: Vec::new(),
            router_roots: Vec::new(),
            moe_sites: proxima_tensor::spec::MoeSites::default(),
            single_position_step: false,
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            single_range: None,
            expert_slab: std::sync::Mutex::new(crate::expert_slab::ExpertSlab::new()),
            expert_sidecar: None,
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

    #[test]
    fn configured_memory_ceiling_rejects_weights_before_device_allocation() {
        const FOUR_GIB: u64 = 4 * 1024 * 1024 * 1024;
        const FIVE_GIB: u64 = 5 * 1024 * 1024 * 1024;

        let model = model_with(FIVE_GIB);
        let mut serving_config = ServingConfig {
            gpu_memory_fit: true,
            gpu_memory_limit_bytes: Some(FOUR_GIB),
            context_length: 1,
            ..ServingConfig::default()
        };

        let error = model
            .apply_memory_fit_gate(&mut serving_config)
            .expect_err("five GiB of weights must not pass a four GiB device ceiling");

        assert!(
            matches!(
                error,
                crate::error::InteropError::MemoryBudgetExceeded {
                    dense_weights_bytes: FIVE_GIB,
                    limit_bytes: FOUR_GIB,
                    os_headroom_bytes: 0,
                    ..
                }
            ),
            "the typed error must retain the configured ceiling and offending weight class: {error:?}"
        );
    }

    /// [`PrefixState`] against the real host-local qwen3-1.7B checkpoint
    /// [`crate::test_support::qwen3_gguf_path`] resolves -- this checkpoint
    /// is qk-norm, so ordinary [`LoadedModel::generate_with_serving_config`]
    /// on Metal takes [`LoadedModel::run_decode_loop_placed_kv`]'s single-
    /// range fast path (`qwen3_split_half_rope_cpu_and_metal_greedy_decode_match`'s
    /// own doc, `proxima-model-interop/src/quality.rs`), which has no host-
    /// side [`LayerCacheState`] to report as a [`PrefixState`] at all --
    /// every test here therefore exercises the two-range SPLIT path
    /// [`LoadedModel::prefill_prefix`]/[`LoadedModel::generate_from_prefix`]
    /// force via `force_two_range: true`, not the fused single-range kernel
    /// ordinary callers get. `#[ignore]`d like every other host-local
    /// fixture in this crate.
    ///
    /// **Residual, run and measured, not hidden (ROW 406):** the two parity
    /// tests below currently FAIL on this real checkpoint --
    /// `generate_with_serving_config` over the concatenated prompt
    /// degenerates to a repeated-token greedy decode (a real, if
    /// uninteresting, model output at `temperature: 0.0`), while
    /// `generate_from_prefix`'s own resumed decode diverges to unrelated
    /// tokens from the very first generated position. The forced two-range
    /// path is exercised here in a shape that, so far as this landing's own
    /// reading of `proxima-tensor/src/spec.rs`/`omega/src/msl.rs` found, no
    /// existing caller ever produces on Metal: a multi-row query
    /// (`query_rows > 1`, the whole prefix) against a NONZERO cached range
    /// (`cached_len > 0` seeded from a prior [`PrefixState`]). Every
    /// existing two-range Metal caller's own multi-row step is prefill
    /// itself, always at `cached_len == 0`; every step with `cached_len >
    /// 0` is ordinary autoregressive decode, always `query_rows == 1`. This
    /// landing's own `prefill_prefix`/`generate_from_prefix` split is the
    /// first caller to combine the two, and the fused Metal
    /// `CachedAttention` kernel (or its uniform/identity/dispatch plumbing
    /// -- unattributed, no profiler/disassembly evidence gathered this
    /// slice) most likely does not handle that combination correctly. NOT
    /// verified against the CPU evaluator in the time this slice had --
    /// that comparison (does `run_cached_attention`'s CPU oracle, not the
    /// Metal kernel, also fail the same way?) is the next diagnostic step,
    /// left open rather than guessed at.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    mod prefix_state_real_model {
        use core::ffi::c_void;
        use std::os::fd::AsFd;

        #[cfg(all(feature = "metal", feature = "instrument"))]
        use proxima_telemetry::export::Exporter;
        #[cfg(all(feature = "metal", feature = "instrument"))]
        use proxima_telemetry::recorder::Recorder;

        use super::super::{
            BackendRuntime, Control, LoadedModel, LogitsSink, NodeValuesSink, Phase, PrefixState,
            supported_serving_config,
        };
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
                .expect("mmap host-local prefix-state gguf fixture")
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

        fn open_model(mapped: &MappedGguf) -> LoadedModel<'_> {
            let file_bytes = mapped.as_slice();
            let parsed = proxima_gguf::pipe::parse_complete(file_bytes)
                .expect("parse host-local qwen3 gguf fixture");
            LoadedModel::load(&parsed, file_bytes)
                .expect("load real qwen3 checkpoint through the public path")
        }

        /// Greedy (`temperature: 0.0`) so [`Self::generate_from_prefix`] and
        /// [`LoadedModel::generate_with_serving_config`] are directly
        /// comparable token-for-token -- any sampling randomness would make
        /// a mismatch ambiguous between "the primitive is wrong" and "the
        /// rng streams diverged".
        fn greedy_serving_config() -> super::super::ServingConfig<'static> {
            let mut config =
                supported_serving_config(GPU_LAYERS_ALL, crate::test_support::math_mode_from_env());
            config.temperature = 0.0;
            config
        }

        /// Real prose (Arthur Conan Doyle, public domain, "A Scandal in
        /// Bohemia"'s opening) rather than synthetic filler -- this is the
        /// byte-for-byte shape a real chat prompt's own shared system/
        /// history prefix takes. Ends on a newline: the tokenizer boundary
        /// this crate's vocabularies treat as a hard break, so
        /// `tokenize(PREFIX)` is a genuine prefix of `tokenize(PREFIX +
        /// SUFFIX_*)` for every `SUFFIX_*` below (each also opens on its own
        /// clause rather than continuing the prefix's last word).
        const PREFIX: &str = "To Sherlock Holmes she is always THE woman. I have seldom heard him mention her under any other name. In his eyes she eclipses and predominates the whole of her sex. It was not that he felt any emotion akin to love for Irene Adler. All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind. He was, I take it, the most perfect reasoning and observing machine that the world has seen, but as a lover he would have placed himself in a false position. He never spoke of the softer passions, save with a gibe and a sneer. They were admirable things for the observer—excellent for drawing the veil from men's motives and actions. But for the trained reasoner to admit such intrusions into his own delicate and finely adjusted temperament was to introduce a distracting factor which might throw a doubt upon all his mental results.\n";

        const SUFFIX_A: &str = "Grit in a sensitive instrument, or a crack in one of his own high-power lenses, would not be more disturbing than a strong emotion in a nature such as his.";

        const SUFFIX_B: &str = "And yet there was but one woman to him, and that woman was the late Irene Adler, of dubious and questionable memory.";

        /// Root-cause proof for ROW 406's own open residual: direct
        /// (`force_two_range: false`, [`LoadedModel::generate_with_serving_config`]'s
        /// own path -- Metal's single-range placed-KV fast path for this
        /// checkpoint) vs resumed (`force_two_range: true`, the
        /// [`PrefixState`] two-range path) compared at LOGIT precision, not
        /// post-argmax, across 5 decode steps. `PREFIX` alone tokenizes to
        /// 183 rows -- already past `ATTENTION_SPLIT_KEYS_PER_SPLIT_AT_SCALE`
        /// (128, `omega/src/msl.rs`), the split-at-scale knee `omega/tests/
        /// qwen3_gqa_qk_norm_two_range_parity.rs`'s own regression test pins.
        /// Before `fix(omega): two-range cached attention skips split merge
        /// dispatch` (cherry-picked to `main` ahead of this test), the
        /// two-range `cached_attention_merge_needed` predicate answered
        /// `true` for this op past that knee and routed it through the
        /// single-range-only `ContextSplitMerge` protocol, which
        /// reinterpreted the resumed path's already-correct, already-
        /// normalized attention output as `(max, sum, weighted[head_dim])`
        /// triples and overwrote it with garbage -- exactly the shape the
        /// prior divergent-token failure this test replaces had. With that
        /// fix on `main`, `max_diff` here is noise-floor
        /// (~1e-5, ordinary Metal-vs-Metal reduction-order float noise) at
        /// every one of the 5 steps, and both paths agree on every argmax.
        #[test]
        #[ignore = "depends on a host-local qwen3 gguf checkout outside this repo, and a real Metal device"]
        fn generate_from_prefix_matches_generate_at_logit_precision_across_five_steps() {
            let model_path = crate::test_support::qwen3_gguf_path();
            crate::test_support::require_fixture(&model_path, Some("PROXIMA_QWEN3_GGUF"));
            let mapped = MappedGguf::open(std::path::Path::new(&model_path))
                .expect("mmap host-local qwen3 gguf fixture");
            let model = open_model(&mapped);
            let serving_config = greedy_serving_config();
            let steps = 5;

            let full_prompt = alloc::format!("{PREFIX}{SUFFIX_A}");
            let mut direct_runtime = BackendRuntime::new(&serving_config);
            let mut direct_logits: Vec<Vec<f32>> = Vec::new();
            let (direct_ids, _, _, _direct_final) = model
                .run_decode_loop_observed_seeded(
                    &full_prompt,
                    steps,
                    &serving_config,
                    &mut direct_runtime,
                    None,
                    &mut LogitsSink::Collect(&mut direct_logits),
                    &mut NodeValuesSink::Discard,
                    &mut |_event| Control::Continue,
                    None,
                    false,
                )
                .expect("direct greedy generate over the concatenated prompt");

            let prefix = model
                .prefill_prefix(PREFIX, &serving_config)
                .expect("prefill the shared prefix once");
            let mut resumed_runtime = BackendRuntime::new(&serving_config);
            let seed = PrefixState {
                ids: prefix.ids.clone(),
                layer_caches: prefix.layer_caches.clone(),
                cached_len: prefix.cached_len,
            };
            let mut resumed_logits: Vec<Vec<f32>> = Vec::new();
            let (resumed_ids, _, _, _resumed_final) = model
                .run_decode_loop_observed_seeded(
                    SUFFIX_A,
                    steps,
                    &serving_config,
                    &mut resumed_runtime,
                    None,
                    &mut LogitsSink::Collect(&mut resumed_logits),
                    &mut NodeValuesSink::Discard,
                    &mut |_event| Control::Continue,
                    Some(seed),
                    true,
                )
                .expect("resume decoding from the cached prefix");

            assert_eq!(
                direct_logits.len(),
                resumed_logits.len(),
                "both paths must run the same number of decode steps"
            );
            for (step, (direct_step, resumed_step)) in
                direct_logits.iter().zip(resumed_logits.iter()).enumerate()
            {
                let max_diff = direct_step
                    .iter()
                    .zip(resumed_step.iter())
                    .map(|(expected, actual)| (expected - actual).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_diff < 1e-3,
                    "step={step}: direct and resumed logits diverge past noise floor \
                     (max_diff={max_diff}) -- the two-range split/merge dispatch \
                     regression (omega/src/msl.rs's cached_attention_merge_needed) \
                     if it comes back"
                );
            }
            assert_eq!(
                resumed_ids, direct_ids,
                "resumed and direct must sample the identical greedy tokens over 5 steps"
            );
        }

        /// (a) parity: resuming from a cached prefix must produce the
        /// IDENTICAL greedy token ids as decoding the concatenated prompt
        /// in one call.
        #[test]
        #[ignore = "depends on a host-local qwen3 gguf checkout outside this repo, and a real Metal device"]
        fn generate_from_prefix_matches_generate_over_the_full_prompt_at_temperature_zero() {
            let model_path = crate::test_support::qwen3_gguf_path();
            crate::test_support::require_fixture(&model_path, Some("PROXIMA_QWEN3_GGUF"));
            let mapped = MappedGguf::open(std::path::Path::new(&model_path))
                .expect("mmap host-local qwen3 gguf fixture");
            let model = open_model(&mapped);
            let serving_config = greedy_serving_config();
            let max_tokens = 16;

            let full_prompt = alloc::format!("{PREFIX}{SUFFIX_A}");
            let (direct_ids, direct_text, _) = model
                .generate_with_serving_config(&full_prompt, max_tokens, serving_config)
                .expect("direct greedy generate over the concatenated prompt");

            let prefix = model
                .prefill_prefix(PREFIX, &serving_config)
                .expect("prefill the shared prefix once");
            let (resumed_ids, resumed_text, _) = model
                .generate_from_prefix(
                    &prefix,
                    SUFFIX_A,
                    max_tokens,
                    &serving_config,
                    &mut |_event| Control::Continue,
                )
                .expect("resume decoding from the cached prefix");

            std::println!(
                "prefix_parity direct={direct_text:?} resumed={resumed_text:?} \
                 prefix_len={}",
                prefix.len()
            );
            assert_eq!(
                resumed_ids, direct_ids,
                "resuming from a cached prefix must produce the same greedy token ids \
                 as decoding the full prompt in one call"
            );
        }

        /// (b) reuse: two different suffixes against the SAME prefill --
        /// both parity-correct against their own direct decode, and each
        /// `generate_from_prefix` call's own `Phase::Prefill` event reports
        /// a `prompt_tokens` count bounded by the suffix alone, never the
        /// prefix's own `cached_len` -- the load-bearing proof that the
        /// second (and first) call never re-ran the prefix's own forward
        /// pass, read off the SAME per-step counter
        /// [`super::super::TokenEvent::phase`]'s own doc already promises,
        /// never wall-clock.
        #[test]
        #[ignore = "depends on a host-local qwen3 gguf checkout outside this repo, and a real Metal device"]
        fn generate_from_prefix_reuses_one_prefill_across_two_suffixes() {
            let model_path = crate::test_support::qwen3_gguf_path();
            crate::test_support::require_fixture(&model_path, Some("PROXIMA_QWEN3_GGUF"));
            let mapped = MappedGguf::open(std::path::Path::new(&model_path))
                .expect("mmap host-local qwen3 gguf fixture");
            let model = open_model(&mapped);
            let serving_config = greedy_serving_config();
            let max_tokens = 12;

            let prefix = model
                .prefill_prefix(PREFIX, &serving_config)
                .expect("prefill the shared prefix once");

            for suffix in [SUFFIX_A, SUFFIX_B] {
                let full_prompt = alloc::format!("{PREFIX}{suffix}");
                let (direct_ids, _, _) = model
                    .generate_with_serving_config(&full_prompt, max_tokens, serving_config)
                    .expect("direct greedy generate over the concatenated prompt");

                let mut prefill_rows = 0_usize;
                let (resumed_ids, _, _) = model
                    .generate_from_prefix(
                        &prefix,
                        suffix,
                        max_tokens,
                        &serving_config,
                        &mut |event| {
                            if let Phase::Prefill { prompt_tokens } = event.phase {
                                prefill_rows = prompt_tokens;
                            }
                            Control::Continue
                        },
                    )
                    .expect("resume decoding from the cached prefix");

                assert_eq!(
                    resumed_ids, direct_ids,
                    "suffix {suffix:?} must match its own direct decode"
                );
                assert!(
                    prefill_rows < prefix.len(),
                    "prefill_rows={prefill_rows} must cover only the suffix's own tokens, \
                     never the {}-token cached prefix -- a value this large would mean \
                     the prefix was re-prefilled",
                    prefix.len()
                );
            }
        }

        /// (c) drop: once every [`PrefixState`] this test built goes out of
        /// scope, the model's own resident-buffer count
        /// (`ROW 403`'s `omega::metal::current_allocated_size`, the same
        /// counter `dropping_a_loaded_model_releases_its_device_buffers`
        /// checks) is unaffected -- [`PrefixState`]'s own doc: its fields
        /// are plain host `Vec<f32>` buffers, never a named device
        /// registration, so there is nothing for a device-buffer counter to
        /// see drop at all. This test's own real assertion is therefore
        /// that the count is IDENTICAL immediately before and after the
        /// drop, not merely "close" -- proving the negative
        /// [`PrefixState`]'s doc claims (no device identity to leak)
        /// rather than assuming it.
        #[test]
        #[ignore = "depends on a host-local qwen3 gguf checkout outside this repo, and a real Metal device"]
        fn dropping_a_prefix_state_leaves_the_models_resident_buffer_count_unchanged() {
            let model_path = crate::test_support::qwen3_gguf_path();
            crate::test_support::require_fixture(&model_path, Some("PROXIMA_QWEN3_GGUF"));
            let mapped = MappedGguf::open(std::path::Path::new(&model_path))
                .expect("mmap host-local qwen3 gguf fixture");
            let model = open_model(&mapped);
            let serving_config = greedy_serving_config();

            let prefix = model
                .prefill_prefix(PREFIX, &serving_config)
                .expect("prefill the shared prefix once");
            let before_drop = omega::metal::current_allocated_size();
            drop(prefix);
            let after_drop = omega::metal::current_allocated_size();

            std::println!(
                "prefix_state_drop before_drop={before_drop:?} after_drop={after_drop:?}"
            );
            assert_eq!(
                before_drop, after_drop,
                "PrefixState owns no device buffer, so dropping it must not move the \
                 model's own resident device-allocation count at all"
            );
        }

        /// Proves the prefill mechanism with a COUNT, not a read of the
        /// source (guiding-principle 18): `run_decode_loop_observed_seeded`
        /// on architectures that accept batched positions. Qwen35's
        /// single-position GDN path intentionally takes a different branch.
        /// with `max_tokens: 1` runs the `decode_until_stop_or_budget`
        /// `for step in 0..1` loop exactly once, and that single step's own
        /// closure calls `BackendRuntime::evaluate` exactly once regardless
        /// of how many rows `next_ids` carries -- `plan_hits`/`plan_misses`
        /// (`BackendRuntime`'s own doc: every `evaluate` call is exactly one
        /// hit or one miss, never both, never neither) sum to the number of
        /// `evaluate` calls this call made. If prefill were one evaluation
        /// PER PROMPT TOKEN, this sum would equal the prompt's own token
        /// count; measured here at a prompt tokenizing to well past 200
        /// rows, it is `1` -- the whole prompt already lands in ONE
        /// `[seq_len, embedding]` program evaluation, not one per token.
        #[test]
        #[ignore = "depends on a host-local qwen3 gguf checkout outside this repo, and a real Metal device"]
        fn prefill_evaluations_per_prompt_token() {
            let model_path = crate::test_support::qwen3_gguf_path();
            crate::test_support::require_fixture(&model_path, Some("PROXIMA_QWEN3_GGUF"));
            let mapped = MappedGguf::open(std::path::Path::new(&model_path))
                .expect("mmap host-local qwen3 gguf fixture");
            let model = open_model(&mapped);
            let serving_config = greedy_serving_config();
            let prompt = alloc::format!("{PREFIX}{SUFFIX_A} {SUFFIX_B}");

            let mut runtime = BackendRuntime::new(&serving_config);
            let (_generated_ids, _text, _stopped_by_eos, prefix_state) = model
                .run_decode_loop_observed_seeded(
                    &prompt,
                    1,
                    &serving_config,
                    &mut runtime,
                    None,
                    &mut LogitsSink::Discard,
                    &mut NodeValuesSink::Discard,
                    &mut |_event| Control::Continue,
                    None,
                    true,
                )
                .expect("prefill a real multi-hundred-token prompt");

            let prompt_token_count = prefix_state.len();
            let program_evaluations = runtime.plan_hits + runtime.plan_misses;
            std::println!(
                "prefill_evaluations_per_prompt_token prompt_token_count={prompt_token_count} \
                 program_evaluations={program_evaluations}"
            );
            assert!(
                prompt_token_count > 200,
                "fixture prompt must tokenize past 200 rows to distinguish \"one \
                 evaluation total\" from \"one evaluation per token\" (got \
                 {prompt_token_count})"
            );
            assert_eq!(
                program_evaluations, 1,
                "prefill already runs the whole prompt through ONE program \
                 evaluation (new_count == prompt_token_count on step 0, \
                 generate.rs's own `run_decode_loop_observed_seeded` closure) -- \
                 a count of {prompt_token_count} here would mean prefill was \
                 actually one evaluation per prompt token"
            );
        }

        /// [`install_stdout_telemetry`]'s handle. Mirrors `bind.rs`'s own
        /// `TelemetryProbe` (this crate's test modules each carry their own
        /// copy of this fixture rather than sharing one, the same
        /// convention `MappedGguf` already follows across this file and
        /// `bind.rs`): a background pump drains the log ring every 5ms so a
        /// real decode step's several-hundred per-op events never overflow
        /// it before this test's own final drain runs, while
        /// `drained_total` survives whichever side (pump or caller) drains
        /// any given batch.
        #[cfg(all(feature = "metal", feature = "instrument"))]
        struct TelemetryProbe {
            recorder: std::sync::Arc<Recorder>,
            drained_total: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        }

        #[cfg(all(feature = "metal", feature = "instrument"))]
        impl TelemetryProbe {
            fn drain_and_total(&self) -> usize {
                let final_pass = self.recorder.drain();
                self.drained_total
                    .fetch_add(final_pass, std::sync::atomic::Ordering::Relaxed)
                    + final_pass
            }
        }

        /// `generate.rs`'s `instrument`-gated `op_profile*` events are
        /// `info!` calls -- no-ops with no recorder installed. Installs a
        /// console recorder at `debug` so a `--nocapture` run of
        /// [`prefill_step_zero_op_profile`] shows every `op_profile_top`/
        /// `op_profile_bucket` line this test's own deliverable is.
        #[cfg(all(feature = "metal", feature = "instrument"))]
        fn install_stdout_telemetry() -> TelemetryProbe {
            proxima_telemetry::emit::global::install(proxima_telemetry::emit::EnvFilter::parse(
                "debug",
            ));
            let recorder = Recorder::builder()
                .export(Exporter::std())
                .expect("console exporter installs for an instrument-gated test")
                .install()
                .expect("stdout telemetry recorder installs for an instrument-gated test");
            let drained_total = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let pump_recorder = std::sync::Arc::clone(&recorder);
            let pump_total = std::sync::Arc::clone(&drained_total);
            std::thread::Builder::new()
                .name("prefill-op-profile-telemetry-drain".to_string())
                .spawn(move || {
                    loop {
                        let drained = pump_recorder.drain();
                        pump_total.fetch_add(drained, std::sync::atomic::Ordering::Relaxed);
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                })
                .expect("spawn the test-only telemetry drain thread");
            TelemetryProbe {
                recorder,
                drained_total,
            }
        }

        /// ROW 411's own open residual: the single `runtime.evaluate` call a
        /// prefill step runs (`program_evaluations=1`, proved by
        /// [`prefill_evaluations_per_prompt_token`] above) costs
        /// 59ms/token wall-clock on an 850-token prompt -- this test walks
        /// INSIDE that one call. Setting `PROXIMA_METAL_OP_PROFILE_STEP=0`
        /// swaps prefill's own step (`_step == 0`, `generate.rs:3585-3599`)
        /// from the production batched `runtime.evaluate` to
        /// `evaluate_op_timed`, which commits ONE command buffer PER
        /// `BoundOp` and reports each op's own GPU-only time
        /// (`report_op_timings`, this file's top-of-file doc). Not a
        /// pass/fail-on-numbers test -- the printed `op_profile_top`/
        /// `op_profile_bucket`/`op_profile_codec`/`op_profile_variant`
        /// lines ARE the deliverable, exactly like `bind.rs`'s
        /// `profiles_one_real_decode_step_by_per_op_gpu_time`.
        #[cfg(all(feature = "metal", feature = "instrument"))]
        #[test]
        #[ignore = "depends on a host-local qwen3 gguf checkout outside this repo, and a real Metal device"]
        fn prefill_step_zero_op_profile() {
            let model_path = crate::test_support::qwen3_gguf_path();
            crate::test_support::require_fixture(&model_path, Some("PROXIMA_QWEN3_GGUF"));
            let mapped = MappedGguf::open(std::path::Path::new(&model_path))
                .expect("mmap host-local qwen3 gguf fixture");
            let model = open_model(&mapped);
            let serving_config = greedy_serving_config();

            // five repeats of PREFIX (183 tokens each per
            // `prefill_evaluations_per_prompt_token`) lands well past the
            // ~850-token prompt the brief measured 50.3s TTFT on.
            let prompt = PREFIX.repeat(5);

            let telemetry_recorder = install_stdout_telemetry();

            // SAFETY: this test only runs via an explicit `--ignored`
            // invocation under nextest's one-process-per-test model, the
            // same convention `bind.rs`'s own
            // `profiles_one_real_decode_step_by_per_op_gpu_time` already
            // relies on for this exact env var.
            unsafe {
                std::env::set_var("PROXIMA_METAL_OP_PROFILE_STEP", "0");
            }
            let mut runtime = BackendRuntime::new(&serving_config);
            let (_generated_ids, _text, _stopped_by_eos, prefix_state) = model
                .run_decode_loop_observed_seeded(
                    &prompt,
                    1,
                    &serving_config,
                    &mut runtime,
                    None,
                    &mut LogitsSink::Discard,
                    &mut NodeValuesSink::Discard,
                    &mut |_event| Control::Continue,
                    None,
                    true,
                )
                .expect("prefill an ~850-token prompt with step 0's op-profile branch armed");
            // SAFETY: same justification as the `set_var` above.
            unsafe {
                std::env::remove_var("PROXIMA_METAL_OP_PROFILE_STEP");
            }

            let prompt_token_count = prefix_state.len();
            let flushed = telemetry_recorder.drain_and_total();
            std::println!(
                "prefill_step_zero_op_profile prompt_token_count={prompt_token_count} \
                 telemetry_records_flushed={flushed}"
            );
            assert!(
                prompt_token_count > 700,
                "fixture prompt must tokenize past 700 rows to land in the same \
                 ~850-token regime the brief measured 50.3s TTFT on (got \
                 {prompt_token_count})"
            );
            assert!(
                flushed > 0,
                "the op-profile step emitted no op_profile telemetry -- \
                 PROXIMA_METAL_OP_PROFILE_STEP=0 must have matched prefill's own step 0"
            );
        }

        /// Times prefill-to-first-token for the same ~850-950-token prompt
        /// [`prefill_step_zero_op_profile`] above uses (`PREFIX.repeat(5)`),
        /// 3 runs against a fresh [`BackendRuntime`] each time so no run
        /// benefits from another run's warm dispatch-plan cache.
        /// `max_tokens: 1` isolates prefill's own step-0 evaluation
        /// ([`prefill_evaluations_per_prompt_token`] above: ONE
        /// `runtime.evaluate` call regardless of prompt length) from any
        /// decode-step cost, so the timed interval IS time-to-first-token,
        /// not time-to-eighth-token. A second, untimed `max_tokens: 8` pass
        /// prints the full greedy id sequence so two builds of this SAME
        /// test -- one per feature set (`metal,instrument` vs
        /// `metal,instrument,metal-tiled-gemm`) -- can be diffed
        /// byte-for-byte: per ROW 105/107/109/113's own "does not earn the
        /// production default until ... the full stack wins with it on"
        /// framing, a tiled-gemm speedup that changes the greedy decode is
        /// not a win, it is a correctness regression.
        #[test]
        #[ignore = "depends on a host-local qwen3 gguf checkout outside this repo, and a real Metal device"]
        fn prefill_ttft_850() {
            let model_path = crate::test_support::qwen3_gguf_path();
            crate::test_support::require_fixture(&model_path, Some("PROXIMA_QWEN3_GGUF"));
            let mapped = MappedGguf::open(std::path::Path::new(&model_path))
                .expect("mmap host-local qwen3 gguf fixture");
            let model = open_model(&mapped);
            let serving_config = greedy_serving_config();
            let prompt = PREFIX.repeat(5);

            let mut ttft_ms: Vec<f64> = Vec::new();
            let mut prompt_token_count = 0usize;
            for _run in 0..3 {
                let mut runtime = BackendRuntime::new(&serving_config);
                let start = std::time::Instant::now();
                let (_generated_ids, _text, _stopped_by_eos, prefix_state) = model
                    .run_decode_loop_observed_seeded(
                        &prompt,
                        1,
                        &serving_config,
                        &mut runtime,
                        None,
                        &mut LogitsSink::Discard,
                        &mut NodeValuesSink::Discard,
                        &mut |_event| Control::Continue,
                        None,
                        true,
                    )
                    .expect("prefill an ~850-token prompt for one timed TTFT run");
                let elapsed = start.elapsed();
                prompt_token_count = prefix_state.len();
                ttft_ms.push(elapsed.as_secs_f64() * 1000.0);
            }
            ttft_ms.sort_by(|left, right| left.partial_cmp(right).expect("ttft_ms never NaN"));
            let min_ms = ttft_ms[0];
            let med_ms = ttft_ms[1];
            let max_ms = ttft_ms[2];
            let tokens_per_sec_at_median = prompt_token_count as f64 / (med_ms / 1000.0);
            std::println!(
                "prefill_ttft_850 prompt_token_count={prompt_token_count} \
                 ttft_ms_min={min_ms:.1} ttft_ms_med={med_ms:.1} ttft_ms_max={max_ms:.1} \
                 tokens_per_sec_at_median={tokens_per_sec_at_median:.1}"
            );
            assert!(
                prompt_token_count > 700,
                "fixture prompt must tokenize past 700 rows to land in the same \
                 ~850-token regime this brief measured TTFT on (got \
                 {prompt_token_count})"
            );

            let mut identity_runtime = BackendRuntime::new(&serving_config);
            let (generated_ids, _text, _stopped_by_eos, _final_prefix) = model
                .run_decode_loop_observed_seeded(
                    &prompt,
                    8,
                    &serving_config,
                    &mut identity_runtime,
                    None,
                    &mut LogitsSink::Discard,
                    &mut NodeValuesSink::Discard,
                    &mut |_event| Control::Continue,
                    None,
                    true,
                )
                .expect("greedy-decode 8 tokens for cross-feature-set identity comparison");
            std::println!("prefill_ttft_850 greedy_eight_token_ids={generated_ids:?}");
        }
    }
}
