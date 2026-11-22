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
//! A [`Plan`] holds `objc2` `Retained` Metal objects and `Rc` leases into the
//! transient pools of `omega::metal::MetalContext`, so it is not `Send`. The
//! plans live in [`ResidentStore`], a field of `ServingContext` beside that
//! context: the plans and the pools their leases point into move as one
//! closed graph, so any thread that serves the model resumes the plans
//! another thread built. (Compiled pipelines are `Send + Sync` and are shared
//! per device by `omega::metal::MetalDevice`.)
//!
//! Ownership: the context owns the store; dropping the model drops the
//! plans. While a request holds the context active, the store sits in this
//! thread's active slot; [`BackendRuntime`] takes the plans at construction
//! and puts them back in `Drop`.
//!
//! Composes: [`CacheKey`](super::prompt_cache_key::CacheKey)'s rule, that a
//! [`ServingConfig`] field is either in the key or named `_` with the reason.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::cell::RefCell;

use omega::PlacedBuffer;
use omega::metal::{DispatchType, MathMode, Plan};
use proxima_numeric::{NumericFormat, NumericPolicy};
use proxima_tensor::op::NodeId;
use proxima_telemetry::metric::Counter;

use super::program_cache::ProgramCache;
use crate::serving::{GdnPrefillBackend, ServingConfig};

/// The wire tags ([`crate::bind::Codec::tag`]) of the codecs a plan reads its
/// cached K and V leaves through; `None` is the f32 host or device cache. A
/// tag is `Ord`, which the 520-byte [`crate::bind::Codec`] is not.
pub(super) type KvBinding = (Option<u8>, Option<u8>);

/// Plans built from scratch by the placed path; a test reads the delta with
/// `PLAN_BUILDS.scoped`. Beside [`PLAN_REFITS`], `builds` is what a request
/// compiled and `refits` what it moved from a resident plan instead.
pub(super) static PLAN_BUILDS: Counter = Counter::new("proxima_model_interop.residency.plan_builds");

/// Placed plans moved to a new row count or KV length by `refit_symbols`.
pub(super) static PLAN_REFITS: Counter = Counter::new("proxima_model_interop.residency.plan_refits");

/// Rows at which prefill kernels change form: below it cached attention runs untiled.
const WIDE_PREFILL_ROWS: usize = 16;

fn plan_class(key: &DecodePlanKey) -> (bool, bool, bool, &[NodeId], bool, KvBinding) {
    (key.0 == 1, key.0 < WIDE_PREFILL_ROWS, key.1 == 0, key.2.as_slice(), key.3, key.4)
}

/// Moves `near`, cached under `cached`, to `symbols`: the KV extent first at the
/// cached width (the plan's attention refit), then the rows (`Plan::refit_symbols`
/// refuses a rows move that also moves the extent). `None` means the plan now
/// serves `symbols`; `Some(key)` means the refit was declined and the plan still
/// stands at `key`, which is `cached` with the extent it was moved to, so the
/// caller can put it back without a stale label.
///
/// # Errors
/// Shape inference or kernel resolution failing; the plan stays at `cached`'s symbols.
pub(super) fn refit_near_plan(
    near: &mut Plan,
    cached: &DecodePlanKey,
    symbols: &[u64],
) -> Result<Option<DecodePlanKey>, crate::error::InteropError> {
    let extent_moved = symbols.get(1).is_some_and(|extent| *extent as usize != cached.1);
    let rows_moved = symbols.first().is_some_and(|rows| *rows as usize != cached.0);
    let mut standing = cached.clone();
    if extent_moved && rows_moved {
        let mut staged = symbols.to_vec();
        staged[0] = cached.0 as u64;
        if !near.refit_symbols(&staged)? {
            return Ok(Some(standing));
        }
        standing.1 = staged[1] as usize;
    }
    Ok(if near.refit_symbols(symbols)? { None } else { Some(standing) })
}

/// Whether `cached` can be refitted to serve `wanted`: the same width class
/// (one row, under 16 rows, empty cache, output set, epilogue switch and KV
/// binding all agree; the CUDA counterpart is `cuda_resident_plans::same_width_class`)
/// and a cached plan at least as wide as the request, since a refit never
/// grows past the arena it was built with.
pub(super) fn admits_near_plan(cached: &DecodePlanKey, wanted: &DecodePlanKey) -> bool {
    cached.0 >= wanted.0 && plan_class(cached) == plan_class(wanted)
}

/// `(new_count, kv_bound_extent, outputs, epilogue_sources, kv_binding)` -- the
/// key [`BackendRuntime::resolve_cached_plan`] uses for placed plans. The
/// binding is part of the key because a plan built against f32 cached K/V and
/// one built against binary16 bytes are different programs over the same
/// shape: found by each other's key, the first would read binary16 as f32.
pub(super) type DecodePlanKey = (usize, usize, Vec<NodeId>, bool, KvBinding);

/// The key of the plan that `symbols`, `outputs`, the epilogue switch and the
/// cached K/V `codecs` select.
///
/// # Errors
/// A codec outside the wire set has no tag, so no key can name its plan.
pub(super) fn decode_plan_key(
    symbols: &[u64],
    outputs: &[NodeId],
    epilogue_sources: bool,
    codecs: (Option<crate::bind::Codec>, Option<crate::bind::Codec>),
) -> Result<DecodePlanKey, crate::error::InteropError> {
    let tag = |codec: Option<crate::bind::Codec>| {
        codec
            .map(|codec| {
                codec.tag().ok_or_else(|| {
                    crate::error::InteropError::UnsupportedServingConfig(alloc::format!(
                        "cached kv codec {codec:?} has no wire tag to key a decode plan by"
                    ))
                })
            })
            .transpose()
    };
    Ok((
        symbols[0] as usize,
        symbols[1] as usize,
        outputs.to_vec(),
        epilogue_sources,
        (tag(codecs.0)?, tag(codecs.1)?),
    ))
}

/// The plans one runtime holds for a set of shapes, keyed by [`DecodePlanKey`].
pub(super) type DecodePlans = BTreeMap<DecodePlanKey, Plan>;

/// Everything one model keeps resident on one thread: the decode plans, the
/// wide plans, the order the wide plans were last used in (least recent
/// first), and the device KV buffers a finished generation handed back
/// (`ServingConfig::resident_kv_bytes`). A key in `wide_order` that is not in
/// `wide` is stale and ignored.
#[derive(Default)]
pub(super) struct ResidentPlans {
    pub(super) decode: DecodePlans,
    pub(super) wide: DecodePlans,
    pub(super) wide_order: Vec<DecodePlanKey>,
    pub(super) kv_buffers: Vec<PlacedBuffer>,
    /// the greedy and direct kv rows programs, which a request would otherwise rebuild
    pub(super) programs: ProgramCache,
}

impl ResidentPlans {
    fn is_empty(&self) -> bool {
        self.decode.is_empty()
            && self.wide.is_empty()
            && self.kv_buffers.is_empty()
            && self.programs.is_empty()
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
    /// Plan-time level schedule applied to the bound program.
    wavefront_schedule: bool,
    /// Cached-attention source and grid selection applied to the plan.
    #[cfg(feature = "metal-attn-variants")]
    attention_variant: Option<omega::AttentionVariant>,
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
    kv_cache_element: NumericFormat,
    /// Element type of the value cache; admission lets it differ from the key's
    /// (a block key layout beside a narrow scalar value).
    kv_value_cache_element: NumericFormat,
}

impl PlanIdentity {
    pub(super) fn of(config: &ServingConfig) -> Self {
        let ServingConfig {
            // identity of the weights: entries are owned by one LoadedModel
            model_path: _,
            // bounds admission and the memory fit only
            context_length: _,
            rope_scaling: _,
            // graph-bound at load and admitted per call: one model, one identity
            recurrent: _,
            parallel_sequences: _,
            kv_cache_key_quant,
            kv_cache_value_quant,
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
            moe_high_precision_threshold: _,
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
            // placed plans refuse a configured runtime, so none is built under one
            omega_runtime: _,
            // the plan key's own second field
            kv_bucket_tokens: _,
            math_mode,
            numeric_policy,
            dispatch_type,
            wavefront_schedule,
            #[cfg(feature = "metal-attn-variants")]
            attention_variant,
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
            ignore_eos: _,
            cached_attention_fusion,
            gated_delta_net_fusion,
            moe_topk_fusion,
            plan_time_constants,
            // refit vs rebuild at a bucket crossing: the same ops either way
            plan_refit: _,
            command_buffer_chunks,
            // descriptor-level choices: the plan is built from the program they already shaped
            speculative_verify_override: _,
            kv_cache_shape_override: _,
            activation_override: _,
            max_command_buffers_per_token: _,
            // device memory held for wide plans between calls: trims, never changes a plan
            resident_prefill_plan_bytes: _,
            // device KV buffers kept bound between calls: where the rows live, never a plan
            resident_kv_bytes: _,
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
            wavefront_schedule,
            #[cfg(feature = "metal-attn-variants")]
            attention_variant,
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
            kv_value_cache_element: kv_cache_value_quant,
        }
    }
}

/// One model's resident plans and the [`PlanIdentity`] they were built under.
/// `ServingContext` owns one; activating the context swaps it into this
/// thread's active slot for a request, and the swap back on the guard's drop
/// returns it. Plans change threads only inside that context, together with
/// the transient pools their `Rc` leases point into.
#[derive(Default)]
pub(super) struct ResidentStore {
    identity: Option<PlanIdentity>,
    plans: ResidentPlans,
}

impl ResidentStore {
    /// each wide plan's learned row-template verdict.
    #[cfg(test)]
    pub(super) fn wide_template_verdicts(&self) -> Vec<Option<bool>> {
        self.plans.wide.values().map(Plan::row_template_verdict).collect()
    }

    #[cfg(test)]
    pub(super) fn wide_count(&self) -> usize {
        self.plans.wide.len()
    }

    /// Decode and wide plans held.
    #[cfg(test)]
    pub(super) fn plan_count(&self) -> usize {
        self.plans.decode.len() + self.plans.wide.len()
    }

    #[cfg(test)]
    pub(super) fn programs(&self) -> &ProgramCache {
        &self.plans.programs
    }

    /// `(new_count, outputs, bytes)` of each wide plan, then the device KV buffers
    /// as `(count, bytes)`, for tests that compare what a call left resident.
    #[cfg(test)]
    pub(super) fn footprint(&self) -> ResidentFootprint {
        ResidentFootprint {
            decode: self.plans.decode.values().map(plan_bytes).sum(),
            wide: self.plans.wide.iter().map(|(key, plan)| (key.0, key.2.len(), plan_bytes(plan))).collect(),
            kv_buffers: self.plans.kv_buffers.len(),
            kv_bytes: kv_bytes(&self.plans.kv_buffers),
        }
    }

    /// Each wide plan as `(new_count, output_count, slot_bytes, transient_buffer_id)`, for tests.
    #[cfg(test)]
    pub(super) fn wide_arenas(&self) -> Vec<(usize, usize, Vec<usize>, Option<usize>)> {
        self.plans
            .wide
            .iter()
            .map(|(key, plan)| (key.0, key.2.len(), plan.arena_slot_bytes().map(<[usize]>::to_vec).unwrap_or_default(), plan.arena_transient_buffer_id()))
            .collect()
    }

    /// Exchanges this store with the calling thread's active slot.
    pub(super) fn swap_active(&mut self) {
        ACTIVE.with(|slot| core::mem::swap(self, &mut *slot.borrow_mut()));
    }
}

/// What a store holds, for tests: decode plan bytes, each wide plan as
/// `(new_count, output_count, bytes)`, and the kept device KV buffers.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ResidentFootprint {
    pub(super) decode: usize,
    pub(super) wide: Vec<(usize, usize, usize)>,
    pub(super) kv_buffers: usize,
    pub(super) kv_bytes: usize,
}

#[cfg(test)]
impl ResidentFootprint {
    pub(super) fn wide_bytes(&self) -> usize {
        self.wide.iter().map(|(_, _, bytes)| bytes).sum()
    }
}

thread_local! {
    /// The store of the context active on this thread; empty when none is.
    static ACTIVE: RefCell<ResidentStore> = RefCell::new(ResidentStore::default());
}

/// Drops from the calling thread's active slot what a warm-up at `chunk_rows`
/// left that no request reuses: wide plans of any other `new_count` (a refit keeps
/// `new_count` and never grows, so only plans of the chunk width serve a tail) and the
/// kept device KV buffers, which are sized for the warm-up's short context.
pub(super) fn trim_active(chunk_rows: usize) {
    let (dropped_plans, dropped_buffers) = ACTIVE.with(|slot| {
        let mut store = slot.borrow_mut();
        let plans = &mut store.plans;
        let (kept, dropped): (Vec<_>, Vec<_>) = core::mem::take(&mut plans.wide)
            .into_iter()
            .partition(|(key, _)| key.0 == chunk_rows);
        plans.wide = kept.into_iter().collect();
        let wide = &plans.wide;
        plans.wide_order.retain(|key| wide.contains_key(key));
        (dropped, core::mem::take(&mut plans.kv_buffers))
    });
    drop(dropped_plans);
    drop(dropped_buffers);
}

/// Fits the row template of each wide plan in the calling thread's active slot, so a refit
/// across row counts (a short last chunk) applies it instead of binding the program again in
/// the first request. Returns how many plans cached a template; a plan that cannot refit by
/// rows counts none and builds its own verdict later, as before.
///
/// # Errors
/// Binding failing at a probe row count ([`Plan::prime_row_template`]).
pub(super) fn prime_active_row_templates() -> Result<usize, crate::error::InteropError> {
    ACTIVE.with(|slot| {
        let mut store = slot.borrow_mut();
        let mut primed = 0;
        for plan in store.plans.wide.values_mut() {
            let symbols = plan.bound_symbols().to_vec();
            primed += usize::from(plan.prime_row_template(&symbols)?);
        }
        Ok(primed)
    })
}

/// Moves every plan in the calling thread's active slot onto the pool's live transient buffer
/// ([`Plan::rejoin_transient`]), so plans built before a larger one stop holding the smaller
/// buffer the pool has replaced. Returns how many plans moved.
pub(super) fn rejoin_active_transients() -> usize {
    ACTIVE.with(|slot| {
        let mut store = slot.borrow_mut();
        let plans = &mut store.plans;
        let moved = plans.decode.values_mut().chain(plans.wide.values_mut()).map(Plan::rejoin_transient);
        moved.filter(|moved| *moved).count()
    })
}

/// Frees every plan in the calling thread's active slot.
pub(super) fn clear_active() {
    drop(ACTIVE.with(|slot| core::mem::take(&mut *slot.borrow_mut())));
}

/// Removes and returns the active store's plans when they were built under
/// `identity`; plans built under anything else are dropped, not returned.
pub(super) fn take(identity: &PlanIdentity) -> ResidentPlans {
    let store = ACTIVE.with(|slot| core::mem::take(&mut *slot.borrow_mut()));
    match store.identity {
        Some(held) if held == *identity => store.plans,
        _ => ResidentPlans::default(),
    }
}

/// Stores `plans` in the active slot under `identity`. Nothing to keep
/// stores nothing.
pub(super) fn put(identity: PlanIdentity, plans: ResidentPlans) {
    if plans.is_empty() {
        return;
    }
    #[cfg(feature = "instrument")]
    proxima_telemetry::debug!(
        decode_plans = plans.decode.len() as u64,
        wide_plans = plans.wide.len() as u64,
        decode_bytes = plans.decode.values().map(plan_bytes).sum::<usize>() as u64,
        wide_bytes = plans.wide.values().map(plan_bytes).sum::<usize>() as u64,
        kv_buffers = plans.kv_buffers.len() as u64,
        kv_bytes = kv_bytes(&plans.kv_buffers) as u64,
        "resident_plans_held"
    );
    ACTIVE.with(|slot| {
        *slot.borrow_mut() = ResidentStore {
            identity: Some(identity),
            plans,
        };
    });
}

/// Output-slot bytes `plan` holds on the device; `0` for a plan that has not
/// executed yet and so has built no arena. Peak liveness is smaller, but the
/// slots stay allocated for the plan's life, which is what residency costs.
pub(super) fn plan_bytes(plan: &Plan) -> usize {
    plan.arena_allocated_bytes().unwrap_or_default()
}

/// Device bytes the kept KV `buffers` hold.
pub(super) fn kv_bytes(buffers: &[PlacedBuffer]) -> usize {
    buffers.iter().map(omega::placed_buffer_len).sum()
}

/// Removes and returns the smallest kept buffer that holds `bytes`, or `None`
/// when no kept buffer does. A kept buffer larger than `bytes` serves a
/// shorter context; the rows past `bytes` are never bound.
pub(super) fn take_kv_buffer(
    buffers: &mut Vec<PlacedBuffer>,
    bytes: usize,
) -> Option<PlacedBuffer> {
    let index = buffers
        .iter()
        .enumerate()
        .filter(|(_, buffer)| omega::placed_buffer_len(buffer) >= bytes)
        .min_by_key(|(_, buffer)| omega::placed_buffer_len(buffer))
        .map(|(index, _)| index)?;
    Some(buffers.remove(index))
}

/// Adds `returned` to the kept buffers and frees the oldest until what is
/// kept holds at most `budget` bytes. A budget of `0` frees them all.
pub(super) fn keep_kv_buffers(
    kept: &mut Vec<PlacedBuffer>,
    returned: Vec<PlacedBuffer>,
    budget: usize,
) {
    kept.extend(returned);
    let mut held = kv_bytes(kept);
    let mut freed = 0;
    while held > budget && freed < kept.len() {
        held -= omega::placed_buffer_len(&kept[freed]);
        freed += 1;
    }
    kept.drain(..freed);
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
    evict_to(
        plans,
        order,
        budget.saturating_sub(estimate),
        None,
        bytes_of,
    )
}

/// Plans the active slot holds right now, decode and wide.
#[cfg(test)]
pub(super) fn resident_len() -> usize {
    ACTIVE.with(|slot| slot.borrow().plan_count())
}


#[cfg(test)]
mod tests {
    use super::*;

    fn key(new_count: usize, kv_bound: usize) -> DecodePlanKey {
        (new_count, kv_bound, Vec::new(), false, (None, None))
    }

    fn bytes(entry: &usize) -> usize {
        *entry
    }

    fn key_with_codecs(
        codecs: (Option<crate::bind::Codec>, Option<crate::bind::Codec>),
    ) -> DecodePlanKey {
        decode_plan_key(&[1, 512], &[NodeId(125), NodeId(128)], false, codecs)
            .expect("float16 and q4_0 are wire codecs, so both have a tag")
    }

    #[test]
    fn plan_keys_for_one_shape_differ_when_only_the_cached_kv_format_differs() {
        use crate::bind::Codec;

        let host_f32 = key_with_codecs((None, None));
        let device_f16 = key_with_codecs((Some(Codec::Float16), Some(Codec::Float16)));
        let device_q4_0_keys = key_with_codecs((Some(Codec::Q4_0), Some(Codec::Float16)));

        assert_ne!(host_f32, device_f16);
        assert_ne!(device_f16, device_q4_0_keys);
        assert_eq!(
            device_f16,
            key_with_codecs((Some(Codec::Float16), Some(Codec::Float16)))
        );

        let mut cache = BTreeMap::new();
        let (mut hits, mut misses) = (0, 0);
        let host_plan = crate::generate::BackendRuntime::resolve_cached_plan(
            &mut cache,
            &mut hits,
            &mut misses,
            host_f32,
            || Ok::<_, crate::error::InteropError>(11_u32),
        )
        .expect("the host-cache plan resolves");
        assert_eq!(*host_plan, 11);

        let device_plan = crate::generate::BackendRuntime::resolve_cached_plan(
            &mut cache,
            &mut hits,
            &mut misses,
            device_f16,
            || Ok::<_, crate::error::InteropError>(22_u32),
        )
        .expect("the device-bound plan resolves apart from the host-bound one");
        assert_eq!(*device_plan, 22);
        assert_eq!((hits, misses), (0, 2));
    }

    fn plans_of(entries: &[(DecodePlanKey, usize)]) -> BTreeMap<DecodePlanKey, usize> {
        entries.iter().cloned().collect()
    }

    #[test]
    fn eviction_removes_the_least_recently_used_plan_first() {
        let mut plans = plans_of(&[
            (key(970, 1024), 400),
            (key(512, 1024), 200),
            (key(64, 1024), 50),
        ]);
        let mut order = vec![key(512, 1024), key(970, 1024), key(64, 1024)];

        let evicted = evict_to(&mut plans, &mut order, 450, None, bytes);

        assert_eq!(evicted, vec![200], "the plan used longest ago leaves first");
        assert_eq!(
            plans.keys().cloned().collect::<Vec<_>>(),
            vec![key(64, 1024), key(970, 1024)]
        );
        assert_eq!(
            order,
            vec![key(970, 1024), key(64, 1024)],
            "recency drops the evicted key"
        );
    }

    #[test]
    fn eviction_keeps_the_plan_in_use_even_when_it_alone_exceeds_the_budget() {
        let mut plans = plans_of(&[(key(970, 1024), 400), (key(64, 1024), 50)]);
        let mut order = vec![key(970, 1024), key(64, 1024)];

        let evicted = evict_to(&mut plans, &mut order, 100, Some(&key(970, 1024)), bytes);

        assert_eq!(evicted, vec![50]);
        assert!(
            plans.contains_key(&key(970, 1024)),
            "the plan being executed is never evicted"
        );
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

        assert_eq!(
            order,
            vec![key(512, 1024), key(64, 1024), key(970, 1024), key(8, 1024)]
        );
    }

    #[test]
    fn a_narrow_verify_plan_leaves_the_prompt_plan_resident() {
        let mut plans = plans_of(&[(key(970, 1024), 402 << 20)]);
        let mut order = vec![key(970, 1024)];

        let evicted = make_room(&mut plans, &mut order, 512 << 20, 8, bytes);

        assert!(
            evicted.is_empty(),
            "an 8-row plan needs about 3 MiB, which fits beside 402 MiB"
        );
        assert_eq!(plans.len(), 1);
    }

    #[test]
    fn a_longer_prompt_evicts_the_resident_prompt_plan_before_it_is_built() {
        let mut plans = plans_of(&[(key(970, 1024), 402 << 20)]);
        let mut order = vec![key(970, 1024)];

        let evicted = make_room(&mut plans, &mut order, 512 << 20, 1000, bytes);

        assert_eq!(
            evicted.len(),
            1,
            "a 1000-row plan scales to about 414 MiB and cannot sit beside 402 MiB"
        );
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

    const PAGE: usize = 16384;

    fn placed(pages: usize) -> Result<PlacedBuffer, omega::MetalError> {
        omega::allocate_placed_buffer(pages * PAGE)
    }

    fn pages_of(buffers: &[PlacedBuffer]) -> Vec<usize> {
        buffers
            .iter()
            .map(|buffer| omega::placed_buffer_len(buffer) / PAGE)
            .collect()
    }

    #[test]
    fn a_request_binds_the_smallest_kept_buffer_that_holds_it() -> Result<(), omega::MetalError> {
        let mut kept = vec![placed(8)?, placed(2)?, placed(4)?];

        let taken = take_kv_buffer(&mut kept, 3 * PAGE);

        assert_eq!(
            taken.map(|buffer| omega::placed_buffer_len(&buffer) / PAGE),
            Some(4),
            "the 4 page buffer is the smallest that holds 3 pages"
        );
        assert_eq!(
            pages_of(&kept),
            vec![8, 2],
            "the others stay kept in their order"
        );
        Ok(())
    }

    #[test]
    fn a_request_larger_than_every_kept_buffer_binds_nothing() -> Result<(), omega::MetalError> {
        let mut kept = vec![placed(2)?, placed(4)?];

        assert!(take_kv_buffer(&mut kept, 5 * PAGE).is_none());
        assert_eq!(
            pages_of(&kept),
            vec![2, 4],
            "a miss leaves the kept buffers alone"
        );
        Ok(())
    }

    #[test]
    fn buffers_handed_back_past_the_budget_free_the_oldest_first() -> Result<(), omega::MetalError>
    {
        let mut kept = vec![placed(4)?, placed(4)?];

        keep_kv_buffers(&mut kept, vec![placed(2)?, placed(2)?], 9 * PAGE);

        assert_eq!(
            pages_of(&kept),
            vec![4, 2, 2],
            "12 pages held, the oldest 4 page buffer leaves to reach 8"
        );
        assert!(kv_bytes(&kept) <= 9 * PAGE);
        Ok(())
    }

    #[test]
    fn a_zero_budget_keeps_nothing_between_generations() -> Result<(), omega::MetalError> {
        let mut kept = Vec::new();

        keep_kv_buffers(&mut kept, vec![placed(4)?, placed(2)?], 0);

        assert!(kept.is_empty());
        Ok(())
    }

    #[test]
    fn an_empty_resident_set_is_not_stored() {
        let identity = PlanIdentity::of(&ServingConfig::default());

        put(identity, ResidentPlans::default());

        assert_eq!(resident_len(), 0);
        assert!(ACTIVE.with(|slot| slot.borrow().identity.is_none()));
    }

    #[test]
    fn identity_separates_plans_bound_to_an_f32_cache_from_those_bound_to_an_f16_one() {
        let f32_cache = ServingConfig {
            kv_cache_key_quant: NumericFormat::Float32,
            kv_cache_value_quant: NumericFormat::Float32,
            ..ServingConfig::default()
        };
        let f16_cache = ServingConfig {
            kv_cache_key_quant: NumericFormat::Float16,
            kv_cache_value_quant: NumericFormat::Float16,
            ..ServingConfig::default()
        };

        assert_ne!(PlanIdentity::of(&f32_cache), PlanIdentity::of(&f16_cache));
    }

    #[test]
    fn identity_separates_plans_whose_value_cache_format_alone_differs() {
        let f16_values = ServingConfig {
            kv_cache_key_quant: NumericFormat::Q4_0,
            kv_cache_value_quant: NumericFormat::Float16,
            ..ServingConfig::default()
        };
        let bf16_values = ServingConfig {
            kv_cache_value_quant: NumericFormat::BFloat16,
            ..f16_values
        };

        assert_ne!(PlanIdentity::of(&f16_values), PlanIdentity::of(&bf16_values));
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

#[cfg(test)]
mod near_plan_tests {
    use super::*;

    const LOGITS: u32 = 570;

    fn key(new_count: usize, outputs: &[u32]) -> DecodePlanKey {
        (new_count, 512, outputs.iter().copied().map(NodeId).collect(), false, (None, None))
    }

    #[test]
    fn near_plan_admits_a_wider_plan_in_the_same_class() {
        assert!(admits_near_plan(&key(512, &[LOGITS]), &key(487, &[LOGITS])));
    }

    #[test]
    fn near_plan_refuses_a_different_output_set() {
        assert!(!admits_near_plan(&key(512, &[]), &key(487, &[LOGITS])));
    }

    #[test]
    fn near_plan_refuses_a_different_width_class() {
        assert!(!admits_near_plan(&key(512, &[LOGITS]), &key(8, &[LOGITS])));
    }

    #[test]
    fn near_plan_refuses_a_narrower_cached_plan() {
        assert!(!admits_near_plan(&key(256, &[LOGITS]), &key(487, &[LOGITS])));
    }

    #[test]
    fn near_plan_refuses_a_different_kv_binding() {
        let mut other = key(512, &[LOGITS]);
        other.4 = (Some(1), Some(1));
        assert!(!admits_near_plan(&other, &key(487, &[LOGITS])));
    }
}
