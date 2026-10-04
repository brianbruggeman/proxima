//! Fixed-capacity DynaExq/HOBBIT expert residency policy.
//!
//! The policy owns only its bounded score/state matrix.  It observes routed
//! experts during a decode step, then emits page and eviction actions at the
//! next step boundary.  Applying those actions is the only operation here
//! that touches [`crate::ExpertSlab`].

use crate::bind::Codec;
use crate::{ExpertSlab, InteropError};

/// A layer/expert coordinate in a model's routed-expert matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertAddress {
    pub layer: usize,
    pub expert: usize,
}

/// One routed expert and its current router importance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoutedExpert {
    pub expert: usize,
    pub importance: f32,
}

/// The precision that can serve a routed expert in the current decode step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServePrecision {
    Low,
    High,
}

/// Per-route HOBBIT decision. A miss serves its low-precision fallback for
/// this step; promotions only take effect at the following step boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServeDecision {
    pub address: ExpertAddress,
    pub precision: ServePrecision,
}

/// An advisory expert warming request produced by a predictor such as APEX or
/// SPICE. It never changes the authoritative route or the resident set; the
/// caller may turn it into a storage operation at a later step boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrefetchCandidate {
    pub address: ExpertAddress,
    pub confidence: f32,
}

/// Fixed-capacity predictor output. The caller owns the boundary transition,
/// so no allocation or dynamic dispatch occurs while candidates are emitted.
#[derive(Debug, Clone, Copy)]
pub struct PrefetchCandidates<const CAPACITY: usize> {
    entries: [Option<PrefetchCandidate>; CAPACITY],
    len: usize,
}

impl<const CAPACITY: usize> PrefetchCandidates<CAPACITY> {
    const fn new() -> Self {
        Self {
            entries: [None; CAPACITY],
            len: 0,
        }
    }

    #[must_use]
    pub fn as_slice(&self) -> &[Option<PrefetchCandidate>] {
        &self.entries[..self.len]
    }

    fn push_unique(&mut self, candidate: PrefetchCandidate) -> Result<(), ResidencyError> {
        if self
            .entries
            .iter()
            .flatten()
            .any(|entry| entry.address == candidate.address)
        {
            return Ok(());
        }
        let Some(slot) = self.entries.get_mut(self.len) else {
            return Err(ResidencyError::PrefetchCapacityExceeded { capacity: CAPACITY });
        };
        *slot = Some(candidate);
        self.len += 1;
        Ok(())
    }
}

/// A high-precision expert source that may be borrowed by [`ExpertSlab`].
#[derive(Debug, Clone, Copy)]
pub struct ExpertPage<'bytes> {
    pub codec: Codec,
    pub bytes: &'bytes [u8],
    pub out_dim: u32,
    pub in_dim: u32,
}

/// A state-changing operation to perform only between decode steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidencyAction {
    Page(ExpertAddress),
    Evict(ExpertAddress),
}

/// Actions emitted by one budget reconciliation boundary, bounded by the
/// policy matrix (at most one evict or page per expert).
#[derive(Debug, Clone, Default)]
pub struct ResidencyActions {
    entries: Vec<ResidencyAction>,
}

impl ResidencyActions {
    #[must_use]
    pub fn as_slice(&self) -> &[ResidencyAction] {
        &self.entries
    }
}

/// Configuration for the fixed residency matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidencyConfig {
    /// Maximum bytes reserved for high-precision expert copies.
    pub budget_bytes: u64,
    /// The high-precision byte charge for every expert in this policy.
    pub high_bytes_per_expert: u64,
    /// EMA observation rate in the inclusive range `0.0..=1.0`.
    pub ema_rate: f64,
    /// A challenger must exceed a resident's hotness by this amount to evict it.
    pub hysteresis_margin: f64,
    /// A resident remains in place for at least this many routed-token positions.
    pub min_dwell_tokens: u64,
}

impl Default for ResidencyConfig {
    fn default() -> Self {
        Self {
            budget_bytes: 0,
            high_bytes_per_expert: 1,
            ema_rate: 0.1,
            hysteresis_margin: 0.0,
            min_dwell_tokens: 0,
        }
    }
}
/// A boundary operation would not fit the caller-declared fixed batch or address.
/// A boundary operation would not fit the caller-declared fixed action batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ResidencyError {
    #[error("prefetch candidate batch is full at {capacity} candidates")]
    PrefetchCapacityExceeded { capacity: usize },
    #[error("expert address ({layer}, {expert}) is outside the fixed policy matrix")]
    AddressOutOfRange { layer: usize, expert: usize },
}

#[derive(Debug, Clone, Copy)]
struct ExpertState {
    ema: f64,
    last_observed: u64,
    resident_since: u64,
    resident: bool,
    seen: bool,
}

impl ExpertState {
    const COLD: Self = Self {
        ema: 0.0,
        last_observed: 0,
        resident_since: 0,
        resident: false,
        seen: false,
    };
}

/// DynaExq's bounded hotness matrix and HOBBIT's low-on-miss serving rule.
///
/// The `layers x experts` matrix is sized once at construction from the
/// model's own metadata (GGUF `{arch}.block_count` and `{arch}.expert_count`),
/// so no model's dimensions are baked into the type. Observing routes and
/// reconciling, and applying actions never allocate: `new` reserves the state
/// matrix, the scratch target, and an action buffer of `2 * layers * experts`
/// once, and `reconcile` refills them in place.
#[derive(Debug)]
pub struct ExpertResidency {
    config: ResidencyConfig,
    layers: usize,
    experts: usize,
    states: Vec<ExpertState>,
    target: Vec<bool>,
    actions: ResidencyActions,
    last_token: u64,
}

impl ExpertResidency {
    #[must_use]
    pub fn new(config: ResidencyConfig, layers: usize, experts: usize) -> Self {
        let slots = layers.saturating_mul(experts);
        Self {
            config,
            layers,
            experts,
            states: vec![ExpertState::COLD; slots],
            target: Vec::with_capacity(slots),
            actions: ResidencyActions {
                entries: Vec::with_capacity(slots.saturating_mul(2)),
            },
            last_token: 0,
        }
    }

    fn index(&self, address: ExpertAddress) -> Option<usize> {
        (address.layer < self.layers && address.expert < self.experts)
            .then(|| address.layer * self.experts + address.expert)
    }

    /// Records one decode step's routes and returns the precision available
    /// for those routes before any next-boundary page action is applied.
    pub fn observe<const ROUTED: usize>(
        &mut self,
        token: u64,
        layer: usize,
        routed: [RoutedExpert; ROUTED],
    ) -> Result<[ServeDecision; ROUTED], ResidencyError> {
        self.last_token = self.last_token.max(token);
        let mut decisions = [ServeDecision {
            address: ExpertAddress {
                layer: 0,
                expert: 0,
            },
            precision: ServePrecision::Low,
        }; ROUTED];

        let ema_rate = self.config.ema_rate;
        for (index, route) in routed.into_iter().enumerate() {
            let state = self.state_mut(ExpertAddress {
                layer,
                expert: route.expert,
            })?;
            let elapsed = token.saturating_sub(state.last_observed);
            let decay = (1.0 - ema_rate).powi(elapsed.min(i32::MAX as u64) as i32);
            state.ema = state.ema * decay * (1.0 - ema_rate) + ema_rate;
            state.last_observed = token;
            state.seen = true;
            decisions[index] = ServeDecision {
                address: ExpertAddress {
                    layer,
                    expert: route.expert,
                },
                precision: if state.resident {
                    ServePrecision::High
                } else {
                    ServePrecision::Low
                },
            };
        }
        Ok(decisions)
    }

    /// Converts predictor output into bounded warming candidates without
    /// mutating hotness, residency, or authoritative route decisions. A
    /// confidence threshold is supplied by the caller so the predictor and
    /// the policy remain independently testable.
    pub fn prefetch_candidates<const CAPACITY: usize>(
        &self,
        layer: usize,
        predicted: &[RoutedExpert],
        minimum_confidence: f32,
    ) -> Result<PrefetchCandidates<CAPACITY>, ResidencyError> {
        let mut candidates = PrefetchCandidates::new();
        for route in predicted {
            let address = ExpertAddress {
                layer,
                expert: route.expert,
            };
            let state = self.state(address)?;
            if route.importance >= minimum_confidence && !state.resident {
                candidates.push_unique(PrefetchCandidate {
                    address,
                    confidence: route.importance,
                })?;
            }
        }
        Ok(candidates)
    }

    /// Reconciles the resident set to the EMA-ranked, byte-feasible top-N.
    /// The staged actions have no effect until [`Self::apply_at_boundary`]
    /// is called after the active decode step has ended.
    pub fn reconcile(&mut self) -> &ResidencyActions {
        let capacity = self.capacity();
        let mut target = core::mem::take(&mut self.target);
        target.clear();
        target.extend(self.states.iter().map(|state| state.resident));
        let mut resident_count = target.iter().filter(|is_targeted| **is_targeted).count();

        while resident_count > capacity {
            let Some(victim) = self.coldest_target(&target) else {
                break;
            };
            target[victim] = false;
            resident_count -= 1;
        }

        while resident_count < capacity {
            let Some(candidate) = self.hottest_not_target(&target) else {
                break;
            };
            target[candidate] = true;
            resident_count += 1;
        }

        while let Some(candidate) = self.hottest_not_target(&target) {
            let Some(victim) = self.coldest_target(&target) else {
                break;
            };
            let dwelled = self
                .last_token
                .saturating_sub(self.states[victim].resident_since)
                >= self.config.min_dwell_tokens;
            if !dwelled
                || self.hotness(candidate) <= self.hotness(victim) + self.config.hysteresis_margin
            {
                break;
            }
            target[victim] = false;
            target[candidate] = true;
        }

        let mut entries = core::mem::take(&mut self.actions.entries);
        entries.clear();
        let evictions = self
            .states
            .iter()
            .zip(&target)
            .enumerate()
            .filter(|(_, (state, is_targeted))| state.resident && !**is_targeted)
            .map(|(slot, _)| ResidencyAction::Evict(self.address(slot)));
        let pages = self
            .states
            .iter()
            .zip(&target)
            .enumerate()
            .filter(|(_, (state, is_targeted))| !state.resident && **is_targeted)
            .map(|(slot, _)| ResidencyAction::Page(self.address(slot)));
        entries.extend(evictions.chain(pages));
        self.actions.entries = entries;
        self.target = target;
        &self.actions
    }

    /// Applies the actions staged by [`Self::reconcile`] at a slab boundary.
    /// The generic page callback keeps storage ownership with the caller;
    /// pages are borrowed by the slab and no dynamic dispatch or policy
    /// allocation is involved.
    pub fn apply_at_boundary<'file, Page>(
        &mut self,
        slab: &mut ExpertSlab<'file>,
        mut page: Page,
    ) -> Result<(), InteropError>
    where
        Page: FnMut(ExpertAddress) -> Result<ExpertPage<'file>, InteropError>,
    {
        self.apply_actions_at_boundary(slab, |slab, action| match action {
            ResidencyAction::Page(address) => {
                let page = page(address)?;
                slab.page_expert_borrowed(
                    address.layer,
                    address.expert,
                    page.codec,
                    page.bytes,
                    page.out_dim,
                    page.in_dim,
                )?;
                Ok(())
            }
            ResidencyAction::Evict(address) => slab.evict_expert(address.layer, address.expert),
        })
    }

    /// Applies the staged action batch through a caller-defined storage transition.
    ///
    /// This is the projection-aware form used by a routed MoE whose one
    /// policy address controls several gathered weight tables. The callback
    /// is monomorphized and receives the slab directly, so an mmap/LSM source
    /// can update every table without a trait object or action-path allocation.
    pub fn apply_actions_at_boundary<'file, Apply>(
        &mut self,
        slab: &mut ExpertSlab<'file>,
        mut apply: Apply,
    ) -> Result<(), InteropError>
    where
        Apply: FnMut(&mut ExpertSlab<'file>, ResidencyAction) -> Result<(), InteropError>,
    {
        for index in 0..self.actions.entries.len() {
            let action = self.actions.entries[index];
            apply(slab, action)?;
            let (address, resident) = match action {
                ResidencyAction::Page(address) => (address, true),
                ResidencyAction::Evict(address) => (address, false),
            };
            let last_token = self.last_token;
            let state = self.state_mut(address).map_err(|_| {
                InteropError::ExpertSlabIndexOutOfRange {
                    layer: address.layer,
                    expert: address.expert,
                }
            })?;            state.resident = resident;
            state.resident_since = if resident { last_token } else { 0 };
        }
        self.actions.entries.clear();
        Ok(())
    }

    #[must_use]
    pub fn resident(&self, address: ExpertAddress) -> Option<bool> {
        self.index(address).map(|slot| self.states[slot].resident)    }

    fn capacity(&self) -> usize {
        let all_experts = self.states.len();
        self.config
            .budget_bytes
            .checked_div(self.config.high_bytes_per_expert)
            .map_or(all_experts, |budget_capacity| {
                budget_capacity.min(all_experts as u64) as usize
            })
    }

    fn address(&self, slot: usize) -> ExpertAddress {
        ExpertAddress {
            layer: slot / self.experts,
            expert: slot % self.experts,
        }
    }

    fn state_mut(&mut self, address: ExpertAddress) -> Result<&mut ExpertState, ResidencyError> {
        let slot = self.index(address).ok_or(ResidencyError::AddressOutOfRange {
            layer: address.layer,
            expert: address.expert,
        })?;
        Ok(&mut self.states[slot])
    }

    fn state(&self, address: ExpertAddress) -> Result<ExpertState, ResidencyError> {
        self.index(address)
            .map(|slot| self.states[slot])
            .ok_or(ResidencyError::AddressOutOfRange {
                layer: address.layer,
                expert: address.expert,
            })
    }

    fn hotness(&self, slot: usize) -> f64 {
        let state = self.states[slot];
        let elapsed = self.last_token.saturating_sub(state.last_observed);
        let decay = (1.0 - self.config.ema_rate).powi(elapsed.min(i32::MAX as u64) as i32);
        state.ema * decay
    }

    fn hottest_not_target(&self, target: &[bool]) -> Option<usize> {
        (0..self.states.len())
            .filter(|slot| !target[*slot] && self.states[*slot].seen)
            .fold(None, |best: Option<usize>, slot| {
                if best.is_none_or(|current| self.hotness(slot) > self.hotness(current)) {
                    Some(slot)
                } else {
                    best
                }
            })
    }

    fn coldest_target(&self, target: &[bool]) -> Option<usize> {
        (0..self.states.len())
            .filter(|slot| target[*slot])
            .fold(None, |coldest: Option<usize>, slot| {
                if coldest.is_none_or(|current| self.hotness(slot) < self.hotness(current)) {
                    Some(slot)
                } else {
                    coldest
                }
            })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{
        ExpertAddress, ExpertPage, ExpertResidency, ResidencyAction, ResidencyConfig, RoutedExpert,
        ServePrecision,
    };
    use crate::{Codec, ExpertSlab};
    use proxima_tensor::NodeId;

    const CONFIG: ResidencyConfig = ResidencyConfig {
        budget_bytes: 144,
        high_bytes_per_expert: 144,
        ema_rate: 0.5,
        hysteresis_margin: 0.1,
        min_dwell_tokens: 2,
    };

    fn stack() -> Vec<u8> {
        (0..4)
            .flat_map(|expert| core::iter::repeat_n(expert, 144))
            .collect()
    }

    fn apply<'a>(
        policy: &mut ExpertResidency,
        slab: &mut ExpertSlab<'a>,
        bytes: &'a [u8],
    ) {
        policy
            .apply_at_boundary(slab, |address| {
                let start = address.expert * 144;
                Ok(ExpertPage {
                    codec: Codec::Q4K,
                    bytes: &bytes[start..start + 144],
                    out_dim: 32,
                    in_dim: 32,
                })
            })
            .expect("a completed step accepts the selected actions");
    }

    #[test]
    fn stationary_trace_keeps_the_hot_expert_and_serves_high_after_boundary() {
        let bytes = stack();
        let mut slab = ExpertSlab::new();
        slab.bind_layer_stack(0, NodeId(1), Codec::Q4K, &bytes, 4, 32, 32)
            .expect("the four-expert stack binds");
        let mut policy = ExpertResidency::new(CONFIG, 1, 4);

        for token in 1..=6 {
            let served = policy
                .observe(
                    token,
                    0,
                    [RoutedExpert {
                        expert: 2,
                        importance: 1.0,
                    }],
                )
                .expect("expert 2 is inside the fixed matrix");
            policy.reconcile();
            apply(&mut policy, &mut slab, &bytes);
            if token > 1 {
                assert_eq!(served[0].precision, ServePrecision::High);
            }
        }

        assert_eq!(
            policy.resident(ExpertAddress {
                layer: 0,
                expert: 2
            }),
            Some(true)
        );
        assert_eq!(slab.expert_epoch(0, 2), Some(1));
    }

    #[test]
    fn shifting_trace_replaces_a_dwelled_resident_at_the_boundary() {
        let bytes = stack();
        let mut slab = ExpertSlab::new();
        slab.bind_layer_stack(0, NodeId(1), Codec::Q4K, &bytes, 4, 32, 32)
            .expect("the four-expert stack binds");
        let mut policy = ExpertResidency::new(CONFIG, 1, 4);

        for token in 1..=3 {
            policy
                .observe(
                    token,
                    0,
                    [RoutedExpert {
                        expert: 0,
                        importance: 1.0,
                    }],
                )
                .expect("expert 0 is inside the fixed matrix");
            policy.reconcile();
            apply(&mut policy, &mut slab, &bytes);
        }
        assert_eq!(
            policy.resident(ExpertAddress {
                layer: 0,
                expert: 0
            }),
            Some(true)
        );

        let mut last_actions = None;
        for token in 4..=9 {
            policy
                .observe(
                    token,
                    0,
                    [RoutedExpert {
                        expert: 3,
                        importance: 1.0,
                    }],
                )
                .expect("expert 3 is inside the fixed matrix");
            let staged = policy.reconcile().as_slice().to_vec();
            apply(&mut policy, &mut slab, &bytes);
            last_actions = Some(staged);        }

        assert_eq!(
            policy.resident(ExpertAddress {
                layer: 0,
                expert: 0
            }),
            Some(false)
        );
        assert_eq!(
            policy.resident(ExpertAddress {
                layer: 0,
                expert: 3
            }),
            Some(true)
        );
        assert!(
            last_actions
                .expect("the shifting trace has a final boundary")
                .iter()
                .all(|action| !matches!(
                    action,
                    ResidencyAction::Page(ExpertAddress { expert: 0, .. })
                )),
            "the former resident is not re-paged after the trace shifts"
        );
    }

    #[test]
    fn predictor_candidates_are_advisory_bounded_and_do_not_change_residency() {
        let mut policy = ExpertResidency::new(CONFIG, 1, 4);
        policy
            .observe(
                1,
                0,
                [RoutedExpert {
                    expert: 0,
                    importance: 1.0,
                }],
            )
            .expect("the observed route is inside the policy matrix");
        policy.reconcile();
        let mut slab = ExpertSlab::new();
        let bytes = stack();
        slab.bind_layer_stack(0, NodeId(1), Codec::Q4K, &bytes, 4, 32, 32)
            .expect("the predictor fixture binds its expert stack");
        apply(&mut policy, &mut slab, &bytes);

        let candidates = policy
            .prefetch_candidates::<2>(
                0,
                &[
                    RoutedExpert {
                        expert: 0,
                        importance: 1.0,
                    },
                    RoutedExpert {
                        expert: 2,
                        importance: 0.8,
                    },
                    RoutedExpert {
                        expert: 2,
                        importance: 0.7,
                    },
                    RoutedExpert {
                        expert: 3,
                        importance: 0.2,
                    },
                ],
                0.5,
            )
            .expect("predicted expert IDs are inside the policy matrix");

        assert_eq!(candidates.as_slice().len(), 1);
        assert_eq!(
            candidates.as_slice()[0]
                .expect("candidate is populated")
                .address,
            ExpertAddress {
                layer: 0,
                expert: 2,
            }
        );
        assert_eq!(
            policy.resident(ExpertAddress {
                layer: 0,
                expert: 2
            }),
            Some(false)
        );
    }

    #[test]
    fn predictor_candidates_reject_an_out_of_range_prediction() {
        let policy = ExpertResidency::new(CONFIG, 1, 4);
        let error = policy
            .prefetch_candidates::<1>(
                0,
                &[RoutedExpert {
                    expert: 4,
                    importance: 1.0,
                }],
                0.0,
            )
            .expect_err("prefetch cannot address outside the fixed matrix");
        assert_eq!(
            error,
            super::ResidencyError::AddressOutOfRange {
                layer: 0,
                expert: 4,
            }
        );
    }

    #[test]
    fn predictor_candidates_report_fixed_capacity_overflow() {
        let policy = ExpertResidency::new(CONFIG, 1, 4);
        let error = policy
            .prefetch_candidates::<1>(
                0,
                &[
                    RoutedExpert {
                        expert: 1,
                        importance: 1.0,
                    },
                    RoutedExpert {
                        expert: 2,
                        importance: 1.0,
                    },
                ],
                0.0,
            )
            .expect_err("a fixed prefetch batch must not silently drop a candidate");
        assert_eq!(
            error,
            super::ResidencyError::PrefetchCapacityExceeded { capacity: 1 }
        );
    }

    #[test]
    fn matrix_dimensions_come_from_the_constructor_at_qwen35moe_scale() {
        let mut policy = ExpertResidency::new(CONFIG, 40, 256);
        let corner = ExpertAddress {
            layer: 39,
            expert: 255,
        };

        policy
            .observe(
                1,
                39,
                [RoutedExpert {
                    expert: 255,
                    importance: 1.0,
                }],
            )
            .expect("the last layer's last expert is inside a 40 x 256 matrix");
        let staged = policy.reconcile().as_slice().to_vec();

        assert_eq!(policy.resident(corner), Some(false));
        assert_eq!(staged, [ResidencyAction::Page(corner)]);
        assert_eq!(
            policy.resident(ExpertAddress {
                layer: 40,
                expert: 0
            }),
            None
        );
    }

    #[test]
    fn observe_rejects_a_layer_past_the_constructed_matrix() {
        let mut policy = ExpertResidency::new(CONFIG, 40, 256);
        let error = policy
            .observe(
                1,
                40,
                [RoutedExpert {
                    expert: 0,
                    importance: 1.0,
                }],
            )
            .expect_err("layer 40 is outside a 40-layer matrix");
        assert_eq!(
            error,
            super::ResidencyError::AddressOutOfRange {
                layer: 40,
                expert: 0,
            }
        );
    }

    #[test]
    fn reconcile_reuses_its_buffers_across_boundaries_at_qwen35moe_scale() {
        let mut policy = ExpertResidency::new(
            ResidencyConfig {
                budget_bytes: 144 * 64,
                ..CONFIG
            },
            40,
            256,
        );
        let target_pointer = policy.target.as_ptr();
        let target_capacity = policy.target.capacity();
        let actions_pointer = policy.actions.entries.as_ptr();
        let actions_capacity = policy.actions.entries.capacity();
        assert!(actions_capacity >= 2 * 40 * 256);

        let mut staged_total = 0;
        for token in 1..=24_u64 {
            for layer in 0..40 {
                let expert = ((token as usize) * 7 + layer * 13) % 256;
                policy
                    .observe(
                        token,
                        layer,
                        [RoutedExpert {
                            expert,
                            importance: 1.0,
                        }],
                    )
                    .expect("the routed expert is inside the 40 x 256 matrix");
            }
            staged_total += policy.reconcile().as_slice().len();
            assert_eq!(policy.target.as_ptr(), target_pointer);
            assert_eq!(policy.target.capacity(), target_capacity);
            assert_eq!(policy.actions.entries.as_ptr(), actions_pointer);
            assert_eq!(policy.actions.entries.capacity(), actions_capacity);
        }
        assert!(staged_total > 0, "the trace must stage real actions");
    }
}
