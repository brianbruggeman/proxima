//! CUDA plans that outlive the request that built them.
//!
//! A CUDA [`Plan`] is a function of the model's program, the
//! `(new_count, kv_bound_extent)` symbols and the numeric knobs it was built
//! under -- nothing about the prompt. [`BackendRuntime`] dies with each
//! request, so every request re-ran `bind_resolved` and `emit_and_pack` for a
//! decode plan and a prefill plan. [`LoadedModel`] holds a [`CudaResidentPlans`]
//! instead: a runtime takes the entry at construction and puts it back in
//! `Drop`, once per request, never inside the step loop.
//!
//! One plan is kept per class of `(new_count == 1, kv_bound_extent == 0)`: a
//! decode plan, a one-token prefill, a wide prefill and a wide chunk over a
//! warm cache. Refresh cannot cross either boundary, so a class never shares.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use omega::backend::{Plan, carry_cuda_rows, carry_cuda_state, drain_cuda_appended_rows, drain_cuda_carried};
use proxima_gguf::GgmlType;
use proxima_primitives::sync::blocking::Mutex;
use proxima_tensor::NumericPolicy;
use proxima_tensor::op::NodeId;

use crate::serving::{GdnPrefillBackend, ServingConfig};

pub(super) type CudaPlanKey = (usize, usize, Vec<NodeId>, bool);

pub(super) type CudaPlans = BTreeMap<CudaPlanKey, Plan>;

/// What a resident plan was built under; two requests share plans only when
/// every field is equal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct CudaPlanIdentity {
    numeric_policy: NumericPolicy,
    exact_activations: bool,
    kv_cache_element: GgmlType,
    cached_attention_fusion: bool,
    gated_delta_net_fusion: bool,
    moe_topk_fusion: bool,
    gdn_prefill_backend: GdnPrefillBackend,
    moe_pre_gather: bool,
    moe_monolithic_all_low: bool,
    moe_monolithic_high_mmap: bool,
    moe_layer_window: usize,
    moe_residency_budget_bytes: u64,
    prefill_one_evaluation: bool,
    prefill_chunk_positions: usize,
}

impl CudaPlanIdentity {
    /// `None` for a configured omega runtime: its plans are built per call.
    pub(super) fn of(config: &ServingConfig) -> Option<Self> {
        config.omega_runtime.is_none().then_some(Self {
            numeric_policy: config.numeric_policy,
            exact_activations: config.exact_activations,
            kv_cache_element: config.kv_cache_key_quant,
            cached_attention_fusion: config.cached_attention_fusion,
            gated_delta_net_fusion: config.gated_delta_net_fusion,
            moe_topk_fusion: config.moe_topk_fusion,
            gdn_prefill_backend: config.gdn_prefill_backend,
            moe_pre_gather: config.moe_pre_gather,
            moe_monolithic_all_low: config.moe_monolithic_all_low,
            moe_monolithic_high_mmap: config.moe_monolithic_high_mmap,
            moe_layer_window: config.moe_layer_window,
            moe_residency_budget_bytes: config.moe_residency_budget_bytes,
            prefill_one_evaluation: config.prefill_one_evaluation,
            prefill_chunk_positions: config.prefill_chunk_positions,
        })
    }
}

struct Resident {
    identity: CudaPlanIdentity,
    plans: CudaPlans,
}

#[derive(Default)]
pub(super) struct CudaResidentPlans {
    slot: Mutex<Option<Resident>>,
}

impl CudaResidentPlans {
    pub(super) fn clear(&self) {
        self.slot.lock().take();
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.slot.lock().as_ref().map_or(0, |resident| resident.plans.len())
    }

    /// The plans built under `identity`, with no carried state selected; an
    /// entry built under anything else is dropped.
    pub(super) fn take(&self, identity: &CudaPlanIdentity) -> CudaPlans {
        let mut plans = match self.slot.lock().take() {
            Some(resident) if resident.identity == *identity => resident.plans,
            _ => CudaPlans::new(),
        };
        plans.values_mut().for_each(release_carry);
        plans
    }

    /// Keeps `plans` whose carried state drained cleanly; a plan that fails
    /// to drain is dropped rather than kept with state the next request would adopt.
    pub(super) fn put(&self, identity: CudaPlanIdentity, mut plans: CudaPlans) {
        plans.retain(|_, plan| drain_carry(plan));
        *self.slot.lock() = Some(Resident { identity, plans });
    }
}

fn release_carry(plan: &mut Plan) {
    carry_cuda_state(plan, &[]);
    carry_cuda_rows(plan, &[], 0);
}

fn drain_carry(plan: &mut Plan) -> bool {
    drain_cuda_appended_rows(plan).is_ok() && drain_cuda_carried(plan).is_ok()
}

/// Plans in different classes never evict each other and never refit into one
/// another: refresh refuses to cross the one-row boundary or the empty-cache
/// boundary (no cached attention to fuse), so a plan moved across either
/// rebinds in full.
pub(super) fn same_width_class(left: &CudaPlanKey, right: &CudaPlanKey) -> bool {
    plan_class(left) == plan_class(right)
}

fn plan_class(key: &CudaPlanKey) -> (bool, bool) {
    (key.0 == 1, key.1 == 0)
}
