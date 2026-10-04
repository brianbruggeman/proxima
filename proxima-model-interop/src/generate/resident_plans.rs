//! Decode plans that outlive the generation that built them.
//!
//! A decode-shaped [`Plan`] (one new token) is a function of the model's
//! program, the `(new_count, kv_bound_extent)` symbols and the numeric knobs
//! it was built under -- nothing about the prompt. [`BackendRuntime`] used to
//! die with each generation, so every generation re-ran
//! `plan_named_with_placed_inputs`, rebuilt the plan's buffer arena and
//! re-dispatched its plan-time constants on its first decode step.
//!
//! A [`Plan`] holds `objc2` `Retained` Metal objects, which are not `Send`, so
//! it cannot live on [`LoadedModel`] behind a `Mutex` without making the model
//! `!Sync` (it is served from a `SendPipe` behind an `Arc`). It lives where the
//! compiled-pipeline cache already lives, in thread-local storage
//! (`omega::metal::PIPELINE_CACHE`), under the same constraint: a thread
//! reuses the plans it built itself.
//!
//! Ownership: [`LoadedModel::plan_life`] is the model's identity. An entry
//! holds only a [`Weak`] to it, so dropping the model orphans every thread's
//! entry and the next access on that thread frees it; replacing the token
//! (what `&mut self` methods that change the model do) orphans them the same
//! way. Nothing is held across a decode: [`BackendRuntime`] takes the entry at
//! construction and puts it back in `Drop`.
//!
//! Composes: [`CacheKey`](super::prompt_cache_key::CacheKey)'s rule, that a
//! [`ServingConfig`] field is either in the key or named `_` with the reason.

use alloc::collections::BTreeMap;
use alloc::sync::{Arc, Weak};
use alloc::vec::Vec;
use core::cell::RefCell;

use omega::metal::{DispatchType, MathMode, Plan};
use proxima_tensor::NumericPolicy;
use proxima_tensor::op::NodeId;

use crate::serving::{GdnPrefillBackend, ServingConfig};

/// `(new_count, kv_bound_extent, outputs, epilogue_sources)` -- the key
/// [`BackendRuntime::resolve_cached_plan`] already uses for placed plans.
pub(super) type DecodePlanKey = (usize, usize, Vec<NodeId>, bool);

/// The plans one runtime holds for the decode shape (`new_count == 1`).
pub(super) type DecodePlans = BTreeMap<DecodePlanKey, Plan>;

/// What a resident decode plan was built under. Two generations share plans
/// only when every field is equal; any difference drops the entry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct PlanIdentity {
    /// Bind-time rewrites and kernel reductions, fixed for a plan's life.
    numeric_policy: NumericPolicy,
    /// Kernel math mode applied to the compiled plan.
    math_mode: MathMode,
    /// Encoder dispatch mode applied to the plan.
    dispatch_type: DispatchType,
    /// Plan-time constants kept resident after their first dispatch.
    plan_time_constants: bool,
    /// Command-buffer split the plan was stamped with.
    command_buffer_chunks: u32,
    /// Fused cached-attention kernel instead of the unfused graph.
    cached_attention_fusion: bool,
    /// Fused gated-delta-net kernel.
    gated_delta_net_fusion: bool,
    /// Fused MoE top-k kernel.
    moe_topk_fusion: bool,
    /// Backend running the gated-delta-net prefill.
    gdn_prefill_backend: GdnPrefillBackend,
    /// Router / residency / gather as separate phases instead of one graph.
    qwen35moe_pre_gather: bool,
    /// All experts at the low codec.
    qwen35moe_monolithic_all_low: bool,
    /// Original mmap-backed expert stacks in one graph.
    qwen35moe_monolithic_high_mmap: bool,
    /// Adjacent layers executed in one sidecar window.
    qwen35moe_layer_window: usize,
    /// Size of the high-precision expert pool.
    qwen35moe_residency_budget_bytes: u64,
}

impl PlanIdentity {
    pub(super) fn of(config: &ServingConfig) -> Self {
        let ServingConfig {
            // identity of the weights: entries are owned by one LoadedModel
            model_path: _,
            // bounds admission and the memory fit only
            context_length: _,
            rope_scaling: _,
            parallel_sequences: _,
            kv_cache_key_quant: _,
            kv_cache_value_quant: _,
            flash_attention: _,
            // prefill chunk widths: prefill plans are never resident
            batch_size: _,
            ubatch_size: _,
            prefill_one_evaluation: _,
            prefill_chunk_positions: _,
            gpu_layers: _,
            gpu_memory_fit: _,
            gpu_memory_limit_bytes: _,
            dense_weights_budget_bytes: _,
            expert_weights_budget_bytes: _,
            activations_budget_bytes: _,
            kv_cache_budget_bytes: _,
            kv_offload: _,
            multimodal_projector: _,
            reasoning_budget: _,
            temperature: _,
            top_k: _,
            top_p: _,
            min_p: _,
            repeat_last_n: _,
            repeat_penalty: _,
            frequency_penalty: _,
            presence_penalty: _,
            seed: _,
            // the plan key's own second field
            kv_bucket_tokens: _,
            math_mode,
            numeric_policy,
            dispatch_type,
            // CPU reference path only; a resident plan is a Metal plan
            exact_activations: _,
            // bind-time recode applied once at load
            weight_precision: _,
            qwen35moe_pre_gather,
            // keeps cut tensors in device buffers: placement, same values
            qwen35moe_persistent_cuts: _,
            gdn_prefill_backend,
            qwen35moe_residency_budget_bytes,
            // route-history advice for prefetching: timing only
            qwen35moe_expert_prefetch: _,
            qwen35moe_monolithic_all_low,
            qwen35moe_layer_window,
            qwen35moe_monolithic_high_mmap,
            gpu_correctness_fallback: _,
            cached_attention_fusion,
            gated_delta_net_fusion,
            moe_topk_fusion,
            plan_time_constants,
            // refit vs rebuild at a bucket crossing: the same ops either way
            plan_refit: _,
            command_buffer_chunks,
            max_command_buffers_per_token: _,
            overlap_transfer_compute: _,
            admission_schedule: _,
            phase_schedule: _,
            expert_residency_schedule: _,
            // draft width reaches a plan only through its symbols, which are
            // in the plan key
            speculative: _,
            prompt_cache: _,
        } = *config;
        Self {
            numeric_policy,
            math_mode,
            dispatch_type,
            plan_time_constants,
            command_buffer_chunks,
            cached_attention_fusion,
            gated_delta_net_fusion,
            moe_topk_fusion,
            gdn_prefill_backend,
            qwen35moe_pre_gather,
            qwen35moe_monolithic_all_low,
            qwen35moe_monolithic_high_mmap,
            qwen35moe_layer_window,
            qwen35moe_residency_budget_bytes,
        }
    }
}

struct Resident {
    owner: Weak<()>,
    identity: PlanIdentity,
    plans: DecodePlans,
}

thread_local! {
    static RESIDENT: RefCell<Vec<Resident>> = const { RefCell::new(Vec::new()) };
}

fn sweep_orphans(entries: &mut Vec<Resident>) {
    entries.retain(|entry| entry.owner.strong_count() > 0);
}

/// Frees every entry on this thread whose model is gone.
pub(super) fn release_orphans() {
    RESIDENT.with(|cell| sweep_orphans(&mut cell.borrow_mut()));
}

/// Removes and returns `owner`'s resident plans when they were built under
/// `identity`; an entry built under anything else is dropped, not returned.
pub(super) fn take(owner: &Arc<()>, identity: &PlanIdentity) -> DecodePlans {
    RESIDENT.with(|cell| {
        let mut entries = cell.borrow_mut();
        sweep_orphans(&mut entries);
        let position = entries
            .iter()
            .position(|entry| Weak::ptr_eq(&entry.owner, &Arc::downgrade(owner)));
        match position.map(|index| entries.swap_remove(index)) {
            Some(entry) if entry.identity == *identity => entry.plans,
            _ => DecodePlans::new(),
        }
    })
}

/// Stores `plans` as `owner`'s resident entry on this thread. An empty map
/// stores nothing.
pub(super) fn put(owner: &Arc<()>, identity: PlanIdentity, plans: DecodePlans) {
    if plans.is_empty() {
        return;
    }
    RESIDENT.with(|cell| {
        let mut entries = cell.borrow_mut();
        sweep_orphans(&mut entries);
        entries.push(Resident {
            owner: Arc::downgrade(owner),
            identity,
            plans,
        });
    });
}

/// Plans `owner` holds resident on this thread right now.
#[cfg(test)]
pub(super) fn resident_len(owner: &Arc<()>) -> usize {
    RESIDENT.with(|cell| {
        cell.borrow()
            .iter()
            .filter(|entry| Weak::ptr_eq(&entry.owner, &Arc::downgrade(owner)))
            .map(|entry| entry.plans.len())
            .sum()
    })
}

/// Entries this thread holds, for any model.
#[cfg(test)]
pub(super) fn entry_count() -> usize {
    RESIDENT.with(|cell| cell.borrow().len())
}
