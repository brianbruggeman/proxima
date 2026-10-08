//! Plans that outlive the generation that built them.
//!
//! A [`Plan`] is a function of the model's program, the
//! `(new_count, kv_bound_extent)` symbols and the numeric knobs it was built
//! under -- nothing about the prompt. [`BackendRuntime`] used to die with each
//! generation, so every generation re-ran `plan_named_with_placed_inputs`,
//! rebuilt the plan's buffer arena and re-dispatched its plan-time constants.
//!
//! Two kinds are kept. Decode plans (`new_count == 1`) are always kept: one
//! per kv bucket, a few megabytes each. Wide plans (`new_count > 1`: a
//! prompt chunk or a speculative verify) hold an arena that scales with
//! `new_count` (402 MB at a 970-token prompt), so they are kept only up to
//! `ServingConfig::resident_prefill_plan_bytes` of output slots, least
//! recently used leaving first ([`evict_to`], [`make_room`]).
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
use proxima_gguf::GgmlType;
use proxima_tensor::NumericPolicy;
use proxima_tensor::op::NodeId;

use crate::serving::{GdnPrefillBackend, ServingConfig};

/// `(new_count, kv_bound_extent, outputs, epilogue_sources)` -- the key
/// [`BackendRuntime::resolve_cached_plan`] already uses for placed plans.
pub(super) type DecodePlanKey = (usize, usize, Vec<NodeId>, bool);

/// The plans one runtime holds for a set of shapes, keyed by [`DecodePlanKey`].
pub(super) type DecodePlans = BTreeMap<DecodePlanKey, Plan>;

/// Everything one model keeps resident on one thread: the decode plans, the
/// wide plans, and the order the wide plans were last used in (least recent
/// first). A key in `wide_order` that is not in `wide` is stale and ignored.
#[derive(Default)]
pub(super) struct ResidentPlans {
    pub(super) decode: DecodePlans,
    pub(super) wide: DecodePlans,
    pub(super) wide_order: Vec<DecodePlanKey>,
}

impl ResidentPlans {
    fn is_empty(&self) -> bool {
        self.decode.is_empty() && self.wide.is_empty()
    }
}

/// What a resident plan was built under. Two generations share plans only
/// when every field is equal; any difference drops the entry.
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
    moe_pre_gather: bool,
    /// All experts at the low codec.
    moe_monolithic_all_low: bool,
    /// Original mmap-backed expert stacks in one graph.
    moe_monolithic_high_mmap: bool,
    /// Adjacent layers executed in one sidecar window.
    moe_layer_window: usize,
    /// Size of the high-precision expert pool.
    moe_residency_budget_bytes: u64,
    /// Whether the prompt is evaluated in one call by the alternate program
    /// instead of the split loop: the two programs can share a width.
    prefill_one_evaluation: bool,
    /// Positions per chunk of that alternate program.
    prefill_chunk_positions: usize,
    /// Element type of the device-resident KV: a plan binds its cached K/V
    /// operands with this codec, so a plan built for one cannot serve the other.
    kv_cache_element: GgmlType,
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
            kv_cache_key_quant,
            // equal to the key type for every admitted config
            kv_cache_value_quant: _,
            flash_attention: _,
            // chunk widths pick which shapes occur, and the shape is in the plan key
            batch_size: _,
            ubatch_size: _,
            prefill_one_evaluation,
            prefill_chunk_positions,
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
            moe_pre_gather,
            // keeps cut tensors in device buffers: placement, same values
            moe_persistent_cuts: _,
            gdn_prefill_backend,
            moe_residency_budget_bytes,
            // route-history advice for prefetching: timing only
            moe_expert_prefetch: _,
            moe_monolithic_all_low,
            moe_layer_window,
            moe_monolithic_high_mmap,
            gpu_correctness_fallback: _,
            cached_attention_fusion,
            gated_delta_net_fusion,
            moe_topk_fusion,
            plan_time_constants,
            // refit vs rebuild at a bucket crossing: the same ops either way
            plan_refit: _,
            command_buffer_chunks,
            max_command_buffers_per_token: _,
            // device memory held for wide plans between calls: trims, never changes a plan
            resident_prefill_plan_bytes: _,
            overlap_transfer_compute: _,
            warm_model_buffers_at_load: _,
            admission_schedule: _,
            phase_schedule: _,
            expert_residency_schedule: _,
            // draft width reaches a plan only through its symbols, which are
            // in the plan key
            speculative: _,
            prompt_cache: _,
            // a skipped read has no lowering yet, so every resident plan is a dense plan
            attention: _,
            prefill: _,
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
            moe_pre_gather,
            moe_monolithic_all_low,
            moe_monolithic_high_mmap,
            moe_layer_window,
            moe_residency_budget_bytes,
            prefill_one_evaluation,
            prefill_chunk_positions,
            kv_cache_element: kv_cache_key_quant,
        }
    }
}

struct Resident {
    owner: Weak<()>,
    identity: PlanIdentity,
    plans: ResidentPlans,
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
pub(super) fn take(owner: &Arc<()>, identity: &PlanIdentity) -> ResidentPlans {
    RESIDENT.with(|cell| {
        let mut entries = cell.borrow_mut();
        sweep_orphans(&mut entries);
        let position = entries
            .iter()
            .position(|entry| Weak::ptr_eq(&entry.owner, &Arc::downgrade(owner)));
        match position.map(|index| entries.swap_remove(index)) {
            Some(entry) if entry.identity == *identity => entry.plans,
            _ => ResidentPlans::default(),
        }
    })
}

/// Stores `plans` as `owner`'s resident entry on this thread. Nothing to keep
/// stores nothing.
pub(super) fn put(owner: &Arc<()>, identity: PlanIdentity, plans: ResidentPlans) {
    if plans.is_empty() {
        return;
    }
    #[cfg(feature = "instrument")]
    proxima_telemetry::debug!(
        decode_plans = plans.decode.len() as u64,
        wide_plans = plans.wide.len() as u64,
        decode_bytes = plans.decode.values().map(plan_bytes).sum::<usize>() as u64,
        wide_bytes = plans.wide.values().map(plan_bytes).sum::<usize>() as u64,
        "resident_plans_held"
    );
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

/// Output-slot bytes `plan` holds on the device; `0` for a plan that has not
/// executed yet and so has built no arena. Peak liveness is smaller, but the
/// slots stay allocated for the plan's life, which is what residency costs.
pub(super) fn plan_bytes(plan: &Plan) -> usize {
    plan.arena_allocated_bytes().unwrap_or_default()
}

/// Marks `key` as the most recently used wide plan.
pub(super) fn touch(order: &mut Vec<DecodePlanKey>, key: &DecodePlanKey) {
    order.retain(|ordered| ordered != key);
    order.push(key.clone());
}

/// The key to evict next: a plan with no recorded use first (it was inserted
/// by a path that does not track recency), then the least recently used.
/// `keep` is never chosen.
fn least_recent<Entry>(
    plans: &BTreeMap<DecodePlanKey, Entry>,
    order: &[DecodePlanKey],
    keep: Option<&DecodePlanKey>,
) -> Option<DecodePlanKey> {
    let unordered = plans
        .keys()
        .find(|key| Some(*key) != keep && !order.contains(key));
    unordered
        .or_else(|| order.iter().find(|key| Some(*key) != keep))
        .cloned()
}

/// Removes plans, least recently used first, until what is left holds at most
/// `budget` bytes by `bytes_of`, or only `keep` is left. Returns the removed
/// plans in removal order so the caller chooses when they are dropped.
pub(super) fn evict_to<Entry>(
    plans: &mut BTreeMap<DecodePlanKey, Entry>,
    order: &mut Vec<DecodePlanKey>,
    budget: usize,
    keep: Option<&DecodePlanKey>,
    bytes_of: impl Fn(&Entry) -> usize,
) -> Vec<Entry> {
    order.retain(|key| plans.contains_key(key));
    let mut retained: usize = plans.values().map(&bytes_of).sum();
    let mut evicted = Vec::new();
    while retained > budget {
        let Some(victim) = least_recent(plans, order, keep) else {
            break;
        };
        let Some(plan) = plans.remove(&victim) else {
            break;
        };
        retained = retained.saturating_sub(bytes_of(&plan));
        order.retain(|key| key != &victim);
        evicted.push(plan);
    }
    evicted
}

/// [`evict_to`] ahead of building a plan for `new_count` rows: leaves room
/// for the plan about to be built, sized by scaling each retained plan's
/// bytes to `new_count` rows (output slots grow with the rows a plan
/// carries). An 8-row verify plan next to a 970-row prefill plan estimates
/// 3 MB and evicts nothing; a 1000-row prompt next to the same plan estimates
/// 414 MB and evicts it. Memory policy only: [`evict_to`] after the call
/// holds the budget at rest whatever the estimate was.
pub(super) fn make_room<Entry>(
    plans: &mut BTreeMap<DecodePlanKey, Entry>,
    order: &mut Vec<DecodePlanKey>,
    budget: usize,
    new_count: usize,
    bytes_of: impl Fn(&Entry) -> usize,
) -> Vec<Entry> {
    let estimate = plans
        .iter()
        .map(|(key, plan)| {
            let scaled = bytes_of(plan) as u128 * new_count as u128 / key.0.max(1) as u128;
            usize::try_from(scaled).unwrap_or(usize::MAX)
        })
        .max()
        .unwrap_or_default();
    evict_to(plans, order, budget.saturating_sub(estimate), None, bytes_of)
}

/// Plans `owner` holds resident on this thread right now, decode and wide.
#[cfg(test)]
pub(super) fn resident_len(owner: &Arc<()>) -> usize {
    RESIDENT.with(|cell| {
        cell.borrow()
            .iter()
            .filter(|entry| Weak::ptr_eq(&entry.owner, &Arc::downgrade(owner)))
            .map(|entry| entry.plans.decode.len() + entry.plans.wide.len())
            .sum()
    })
}

/// Wide plans `owner` holds resident on this thread right now.
#[cfg(test)]
pub(super) fn resident_wide_len(owner: &Arc<()>) -> usize {
    RESIDENT.with(|cell| {
        cell.borrow()
            .iter()
            .filter(|entry| Weak::ptr_eq(&entry.owner, &Arc::downgrade(owner)))
            .map(|entry| entry.plans.wide.len())
            .sum()
    })
}

/// Entries this thread holds, for any model.
#[cfg(test)]
pub(super) fn entry_count() -> usize {
    RESIDENT.with(|cell| cell.borrow().len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(new_count: usize, kv_bound: usize) -> DecodePlanKey {
        (new_count, kv_bound, Vec::new(), false)
    }

    fn bytes(entry: &usize) -> usize {
        *entry
    }

    fn plans_of(entries: &[(DecodePlanKey, usize)]) -> BTreeMap<DecodePlanKey, usize> {
        entries.iter().cloned().collect()
    }

    #[test]
    fn eviction_removes_the_least_recently_used_plan_first() {
        let mut plans = plans_of(&[(key(970, 1024), 400), (key(512, 1024), 200), (key(64, 1024), 50)]);
        let mut order = vec![key(512, 1024), key(970, 1024), key(64, 1024)];

        let evicted = evict_to(&mut plans, &mut order, 450, None, bytes);

        assert_eq!(evicted, vec![200], "the plan used longest ago leaves first");
        assert_eq!(plans.keys().cloned().collect::<Vec<_>>(), vec![key(64, 1024), key(970, 1024)]);
        assert_eq!(order, vec![key(970, 1024), key(64, 1024)], "recency drops the evicted key");
    }

    #[test]
    fn eviction_keeps_the_plan_in_use_even_when_it_alone_exceeds_the_budget() {
        let mut plans = plans_of(&[(key(970, 1024), 400), (key(64, 1024), 50)]);
        let mut order = vec![key(970, 1024), key(64, 1024)];

        let evicted = evict_to(&mut plans, &mut order, 100, Some(&key(970, 1024)), bytes);

        assert_eq!(evicted, vec![50]);
        assert!(plans.contains_key(&key(970, 1024)), "the plan being executed is never evicted");
    }

    #[test]
    fn a_zero_budget_evicts_everything_which_is_the_behaviour_before_residency() {
        let mut plans = plans_of(&[(key(970, 1024), 400), (key(64, 1024), 50)]);
        let mut order = vec![key(970, 1024), key(64, 1024)];

        let evicted = evict_to(&mut plans, &mut order, 0, None, bytes);

        assert_eq!(evicted.len(), 2);
        assert!(plans.is_empty());
        assert!(order.is_empty());
    }

    #[test]
    fn plans_that_fit_the_budget_are_all_kept() {
        let mut plans = plans_of(&[(key(970, 1024), 400), (key(64, 1024), 50)]);
        let mut order = vec![key(970, 1024), key(64, 1024)];

        let evicted = evict_to(&mut plans, &mut order, 450, None, bytes);

        assert!(evicted.is_empty());
        assert_eq!(plans.len(), 2);
    }

    #[test]
    fn a_plan_with_no_recorded_use_is_evicted_before_any_tracked_plan() {
        let mut plans = plans_of(&[(key(970, 1024), 400), (key(128, 1024), 100)]);
        let mut order = vec![key(970, 1024)];

        let evicted = evict_to(&mut plans, &mut order, 400, None, bytes);

        assert_eq!(evicted, vec![100]);
        assert!(plans.contains_key(&key(970, 1024)));
    }

    #[test]
    fn a_stale_key_in_the_order_is_dropped_and_never_chosen() {
        let mut plans = plans_of(&[(key(970, 1024), 400)]);
        let mut order = vec![key(31, 32), key(970, 1024)];

        let evicted = evict_to(&mut plans, &mut order, 0, None, bytes);

        assert_eq!(evicted, vec![400]);
        assert!(order.is_empty());
    }

    #[test]
    fn touching_a_key_makes_it_the_most_recent_without_duplicating_it() {
        let mut order = vec![key(970, 1024), key(512, 1024), key(64, 1024)];

        touch(&mut order, &key(970, 1024));
        touch(&mut order, &key(8, 1024));

        assert_eq!(order, vec![key(512, 1024), key(64, 1024), key(970, 1024), key(8, 1024)]);
    }

    #[test]
    fn a_narrow_verify_plan_leaves_the_prompt_plan_resident() {
        let mut plans = plans_of(&[(key(970, 1024), 402 << 20)]);
        let mut order = vec![key(970, 1024)];

        let evicted = make_room(&mut plans, &mut order, 512 << 20, 8, bytes);

        assert!(evicted.is_empty(), "an 8-row plan needs about 3 MiB, which fits beside 402 MiB");
        assert_eq!(plans.len(), 1);
    }

    #[test]
    fn a_longer_prompt_evicts_the_resident_prompt_plan_before_it_is_built() {
        let mut plans = plans_of(&[(key(970, 1024), 402 << 20)]);
        let mut order = vec![key(970, 1024)];

        let evicted = make_room(&mut plans, &mut order, 512 << 20, 1000, bytes);

        assert_eq!(evicted.len(), 1, "a 1000-row plan scales to about 414 MiB and cannot sit beside 402 MiB");
        assert!(plans.is_empty());
    }

    #[test]
    fn room_for_a_plan_in_an_empty_store_is_free() {
        let mut plans: BTreeMap<DecodePlanKey, usize> = BTreeMap::new();
        let mut order = Vec::new();

        let evicted = make_room(&mut plans, &mut order, 512 << 20, 1000, bytes);

        assert!(evicted.is_empty());
    }

    #[test]
    fn make_room_with_a_zero_budget_clears_the_store() {
        let mut plans = plans_of(&[(key(970, 1024), 402 << 20), (key(8, 1024), 3 << 20)]);
        let mut order = vec![key(970, 1024), key(8, 1024)];

        let evicted = make_room(&mut plans, &mut order, 0, 970, bytes);

        assert_eq!(evicted.len(), 2);
        assert!(plans.is_empty());
    }

    #[test]
    fn an_empty_resident_set_is_not_stored() {
        let owner = Arc::new(());
        let identity = PlanIdentity::of(&ServingConfig::default());

        put(&owner, identity, ResidentPlans::default());

        assert_eq!(entry_count(), 0);
    }

    #[test]
    fn identity_separates_plans_bound_to_an_f32_cache_from_those_bound_to_an_f16_one() {
        let f32_cache = ServingConfig::default();
        let f16_cache = ServingConfig {
            kv_cache_key_quant: GgmlType::F16,
            kv_cache_value_quant: GgmlType::F16,
            ..ServingConfig::default()
        };

        assert_ne!(PlanIdentity::of(&f32_cache), PlanIdentity::of(&f16_cache));
    }

    #[test]
    fn identity_distinguishes_the_two_prefill_programs() {
        let split = ServingConfig::default();
        let one_call = ServingConfig {
            prefill_one_evaluation: true,
            ..ServingConfig::default()
        };
        let chunked = ServingConfig {
            prefill_one_evaluation: true,
            prefill_chunk_positions: 256,
            ..ServingConfig::default()
        };

        assert_ne!(PlanIdentity::of(&split), PlanIdentity::of(&one_call));
        assert_ne!(PlanIdentity::of(&one_call), PlanIdentity::of(&chunked));
        assert_eq!(
            PlanIdentity::of(&split),
            PlanIdentity::of(&ServingConfig {
                resident_prefill_plan_bytes: 0,
                ..ServingConfig::default()
            }),
            "the residency budget trims plans and never changes one, so it is not part of the identity"
        );
    }
}
