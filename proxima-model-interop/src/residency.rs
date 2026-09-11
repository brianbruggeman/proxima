//! Fixed-capacity DynaExq/HOBBIT expert residency policy.
//!
//! The policy owns only its bounded score/state matrix.  It observes routed
//! experts during a decode step, then emits page and eviction actions at the
//! next step boundary.  Applying those actions is the only operation here
//! that touches [`crate::ExpertSlab`].

use crate::bind::PackedOwnedKind;
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

/// A high-precision expert source that may be borrowed by [`ExpertSlab`].
#[derive(Debug, Clone, Copy)]
pub struct ExpertPage<'bytes> {
    pub codec: PackedOwnedKind,
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

/// Fixed-capacity actions emitted by one budget reconciliation boundary.
#[derive(Debug, Clone, Copy)]
pub struct ResidencyActions<const ACTIONS: usize> {
    entries: [Option<ResidencyAction>; ACTIONS],
    len: usize,
}

impl<const ACTIONS: usize> ResidencyActions<ACTIONS> {
    const fn new() -> Self {
        Self {
            entries: [None; ACTIONS],
            len: 0,
        }
    }

    fn push(&mut self, action: ResidencyAction) -> Result<(), ResidencyError> {
        let Some(entry) = self.entries.get_mut(self.len) else {
            return Err(ResidencyError::ActionCapacityExceeded { capacity: ACTIONS });
        };
        *entry = Some(action);
        self.len += 1;
        Ok(())
    }

    #[must_use]
    pub fn as_slice(&self) -> &[Option<ResidencyAction>] {
        &self.entries[..self.len]
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

/// A boundary operation would not fit the caller-declared fixed action batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ResidencyError {
    #[error("residency action batch is full at {capacity} actions")]
    ActionCapacityExceeded { capacity: usize },
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
/// `LAYERS`, `EXPERTS`, and the output action capacity are compile-time
/// bounds. The policy never allocates while observing routes or selecting a
/// budget-feasible resident set.
#[derive(Debug, Clone)]
pub struct ExpertResidency<const LAYERS: usize, const EXPERTS: usize> {
    config: ResidencyConfig,
    states: [[ExpertState; EXPERTS]; LAYERS],
    last_token: u64,
}

impl<const LAYERS: usize, const EXPERTS: usize> ExpertResidency<LAYERS, EXPERTS> {
    #[must_use]
    pub const fn new(config: ResidencyConfig) -> Self {
        Self {
            config,
            states: [[ExpertState::COLD; EXPERTS]; LAYERS],
            last_token: 0,
        }
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

    /// Reconciles the resident set to the EMA-ranked, byte-feasible top-N.
    /// The returned actions have no effect until [`Self::apply_at_boundary`]
    /// is called after the active decode step has ended.
    pub fn reconcile<const ACTIONS: usize>(
        &self,
    ) -> Result<ResidencyActions<ACTIONS>, ResidencyError> {
        let capacity = self.capacity();
        let mut target = [[false; EXPERTS]; LAYERS];
        let mut resident_count = 0usize;

        for layer in 0..LAYERS {
            for expert in 0..EXPERTS {
                if self.states[layer][expert].resident {
                    target[layer][expert] = true;
                    resident_count += 1;
                }
            }
        }

        while resident_count > capacity {
            let Some(victim) = self.coldest_target(&target) else {
                break;
            };
            target[victim.layer][victim.expert] = false;
            resident_count -= 1;
        }

        while resident_count < capacity {
            let Some(candidate) = self.hottest_not_target(&target) else {
                break;
            };
            target[candidate.layer][candidate.expert] = true;
            resident_count += 1;
        }

        loop {
            let Some(candidate) = self.hottest_not_target(&target) else {
                break;
            };
            let Some(victim) = self.coldest_target(&target) else {
                break;
            };
            let victim_state = self.states[victim.layer][victim.expert];
            let dwelled = self.last_token.saturating_sub(victim_state.resident_since)
                >= self.config.min_dwell_tokens;
            if !dwelled
                || self.hotness(candidate) <= self.hotness(victim) + self.config.hysteresis_margin
            {
                break;
            }
            target[victim.layer][victim.expert] = false;
            target[candidate.layer][candidate.expert] = true;
        }

        let mut actions = ResidencyActions::new();
        for layer in 0..LAYERS {
            for expert in 0..EXPERTS {
                let address = ExpertAddress { layer, expert };
                if self.states[layer][expert].resident && !target[layer][expert] {
                    actions.push(ResidencyAction::Evict(address))?;
                }
            }
        }
        for layer in 0..LAYERS {
            for expert in 0..EXPERTS {
                let address = ExpertAddress { layer, expert };
                if !self.states[layer][expert].resident && target[layer][expert] {
                    actions.push(ResidencyAction::Page(address))?;
                }
            }
        }
        Ok(actions)
    }

    /// Applies actions at a slab boundary. The generic page callback keeps
    /// storage ownership with the caller; pages are borrowed by the slab and
    /// no dynamic dispatch or policy allocation is involved.
    pub fn apply_at_boundary<'file, Page, const ACTIONS: usize>(
        &mut self,
        slab: &mut ExpertSlab<'file>,
        actions: &ResidencyActions<ACTIONS>,
        mut page: Page,
    ) -> Result<(), InteropError>
    where
        Page: FnMut(ExpertAddress) -> Result<ExpertPage<'file>, InteropError>,
    {
        self.apply_actions_at_boundary(slab, actions, |slab, action| match action {
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

    /// Applies an action batch through a caller-defined storage transition.
    ///
    /// This is the projection-aware form used by a routed MoE whose one
    /// policy address controls several gathered weight tables. The callback
    /// is monomorphized and receives the slab directly, so an mmap/LSM source
    /// can update every table without a trait object or action-path allocation.
    pub fn apply_actions_at_boundary<'file, Apply, const ACTIONS: usize>(
        &mut self,
        slab: &mut ExpertSlab<'file>,
        actions: &ResidencyActions<ACTIONS>,
        mut apply: Apply,
    ) -> Result<(), InteropError>
    where
        Apply: FnMut(&mut ExpertSlab<'file>, ResidencyAction) -> Result<(), InteropError>,
    {
        for action in actions.as_slice().iter().flatten().copied() {
            apply(slab, action)?;
            let (address, resident) = match action {
                ResidencyAction::Page(address) => (address, true),
                ResidencyAction::Evict(address) => (address, false),
            };
            self.states[address.layer][address.expert].resident = resident;
            self.states[address.layer][address.expert].resident_since =
                if resident { self.last_token } else { 0 };
        }
        Ok(())
    }

    #[must_use]
    pub fn resident(&self, address: ExpertAddress) -> Option<bool> {
        self.states
            .get(address.layer)
            .and_then(|layer| layer.get(address.expert))
            .map(|state| state.resident)
    }

    fn capacity(&self) -> usize {
        if self.config.high_bytes_per_expert == 0 {
            LAYERS.saturating_mul(EXPERTS)
        } else {
            (self.config.budget_bytes / self.config.high_bytes_per_expert)
                .min(LAYERS.saturating_mul(EXPERTS) as u64) as usize
        }
    }

    fn state_mut(&mut self, address: ExpertAddress) -> Result<&mut ExpertState, ResidencyError> {
        self.states
            .get_mut(address.layer)
            .and_then(|layer| layer.get_mut(address.expert))
            .ok_or(ResidencyError::AddressOutOfRange {
                layer: address.layer,
                expert: address.expert,
            })
    }

    fn hotness(&self, address: ExpertAddress) -> f64 {
        let state = self.states[address.layer][address.expert];
        let elapsed = self.last_token.saturating_sub(state.last_observed);
        let decay = (1.0 - self.config.ema_rate).powi(elapsed.min(i32::MAX as u64) as i32);
        state.ema * decay
    }

    fn hottest_not_target(&self, target: &[[bool; EXPERTS]; LAYERS]) -> Option<ExpertAddress> {
        let mut best = None;
        for layer in 0..LAYERS {
            for expert in 0..EXPERTS {
                if target[layer][expert] {
                    continue;
                }
                if !self.states[layer][expert].seen {
                    continue;
                }
                let candidate = ExpertAddress { layer, expert };
                if best.is_none_or(|current| self.hotness(candidate) > self.hotness(current)) {
                    best = Some(candidate);
                }
            }
        }
        best
    }

    fn coldest_target(&self, target: &[[bool; EXPERTS]; LAYERS]) -> Option<ExpertAddress> {
        let mut coldest = None;
        for layer in 0..LAYERS {
            for expert in 0..EXPERTS {
                if !target[layer][expert] {
                    continue;
                }
                let candidate = ExpertAddress { layer, expert };
                if coldest.is_none_or(|current| self.hotness(candidate) < self.hotness(current)) {
                    coldest = Some(candidate);
                }
            }
        }
        coldest
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ExpertAddress, ExpertPage, ExpertResidency, ResidencyAction, ResidencyConfig, RoutedExpert,
        ServePrecision,
    };
    use crate::{ExpertSlab, PackedOwnedKind};
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
        policy: &mut ExpertResidency<1, 4>,
        slab: &mut ExpertSlab<'a>,
        actions: &super::ResidencyActions<4>,
        bytes: &'a [u8],
    ) {
        policy
            .apply_at_boundary(slab, actions, |address| {
                let start = address.expert * 144;
                Ok(ExpertPage {
                    codec: PackedOwnedKind::Q4K,
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
        slab.bind_layer_stack(0, NodeId(1), PackedOwnedKind::Q4K, &bytes, 4, 32, 32)
            .expect("the four-expert stack binds");
        let mut policy = ExpertResidency::<1, 4>::new(CONFIG);

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
            let actions = policy.reconcile::<4>().expect("one action fits");
            apply(&mut policy, &mut slab, &actions, &bytes);
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
        slab.bind_layer_stack(0, NodeId(1), PackedOwnedKind::Q4K, &bytes, 4, 32, 32)
            .expect("the four-expert stack binds");
        let mut policy = ExpertResidency::<1, 4>::new(CONFIG);

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
            let actions = policy.reconcile::<4>().expect("one action fits");
            apply(&mut policy, &mut slab, &actions, &bytes);
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
            let actions = policy.reconcile::<4>().expect("two actions fit");
            last_actions = Some(actions);
            apply(&mut policy, &mut slab, &actions, &bytes);
        }

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
                .as_slice()
                .iter()
                .flatten()
                .all(|action| !matches!(
                    action,
                    ResidencyAction::Page(ExpertAddress { expert: 0, .. })
                )),
            "the former resident is not re-paged after the trace shifts"
        );
    }
}
