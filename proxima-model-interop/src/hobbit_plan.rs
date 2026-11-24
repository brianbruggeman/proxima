//! A deterministic per-expert high/low plan for the universal HOBBIT path.
//!
//! The signal is the model's own routed-expert selection counter
//! ([`proxima_tensor::cpu::snapshot_expert_selection_top_n`], populated by one
//! calibration run). Per model layer, the experts are ranked by
//! `(count, ema, expert)`; the coldest `low_expert_ppm` parts per million are
//! assigned the low-codec sidecar copy. The plan names no family and no
//! layer: the layer index and expert count come from the checkpoint's bound
//! slab.

use alloc::string::String;
use alloc::vec::Vec;
use std::collections::BTreeMap;

use proxima_tensor::NodeId;
use serde::{Deserialize, Serialize};

/// Parts per million: a fraction of a set is `ppm / PPM`, as an integer.
pub const PPM: u32 = 1_000_000;

/// The fixed-point scale of every measured ratio the plans rank on: a
/// stored value is the measurement times this, rounded.
pub const FIXED_ONE: u32 = 1_000_000_000;

/// Converts a measured ratio to its fixed-point integer, at the point the
/// measurement enters this crate. Negative and NaN inputs store zero.
#[macro_export]
macro_rules! fixed_point {
    ($value:expr) => {{
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let stored = ($value).mul_add($crate::FIXED_ONE.into(), 0.0).round() as u64;
        stored
    }};
}

/// The number of a set's members a `ppm` share covers, rounded up.
fn ppm_count(total: u32, ppm: u32) -> usize {
    let covered = (u64::from(total) * u64::from(ppm)).div_ceil(u64::from(PPM));
    covered.min(u64::from(total)) as usize
}

/// One drained routed-expert selection entry: the gathered reduce node, the
/// expert, how often it was selected, and its selection ema in
/// [`FIXED_ONE`] fixed point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HobbitSelection {
    pub node: NodeId,
    pub expert: u32,
    pub count: u64,
    pub ema_fixed: u64,
}

/// One low expert address, in model-layer space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct HobbitExpertAddress {
    pub layer: u32,
    pub expert: u32,
}

/// A sorted low-expert set plus the deterministic ranking hash.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HobbitExpertPlan {
    pub low_expert_ppm: u32,
    pub layers: u32,
    pub experts: u32,
    /// Sorted by `(layer, expert)`.
    pub low: Vec<HobbitExpertAddress>,
    /// FNV-1a over the ordered low set and the ppm.
    pub plan_hash: u64,
}

impl HobbitExpertPlan {
    /// Builds the plan from one selection snapshot. `layer_for_node` maps a
    /// routed weight node to its model layer (the slab's own mapping);
    /// unobserved experts are coldest.
    #[must_use]
    pub fn from_selection(
        counts: &[HobbitSelection],
        layer_for_node: impl Fn(NodeId) -> Option<usize>,
        layers: u32,
        experts: u32,
        low_expert_ppm: u32,
    ) -> Self {
        let ppm = low_expert_ppm.min(PPM);
        let mut observed: BTreeMap<(u32, u32), (u64, u64)> = BTreeMap::new();
        for selection in counts {
            let Some(layer) = layer_for_node(selection.node) else {
                continue;
            };
            if (layer as u32) >= layers || selection.expert >= experts {
                continue;
            }
            observed.insert(
                (layer as u32, selection.expert),
                (selection.count, selection.ema_fixed),
            );
        }
        let per_layer = ppm_count(experts, ppm);
        let mut low = Vec::with_capacity(per_layer * layers as usize);
        for layer in 0..layers {
            let mut ranked: Vec<(u64, u64, u32)> = (0..experts)
                .map(|expert| {
                    let (count, ema) = observed
                        .get(&(layer, expert))
                        .copied()
                        .unwrap_or((0, 0));
                    (count, ema, expert)
                })
                .collect();
            ranked.sort_unstable();
            for (_, _, expert) in ranked.into_iter().take(per_layer) {
                low.push(HobbitExpertAddress { layer, expert });
            }
        }
        low.sort();
        let plan_hash = hash_plan(ppm, &low);
        Self {
            low_expert_ppm: ppm,
            layers,
            experts,
            low,
            plan_hash,
        }
    }

    /// The plan as `(layer, expert)` addresses for the sidecar install.
    pub fn addresses(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.low
            .iter()
            .map(|address| (address.layer as usize, address.expert as usize))
    }

    /// A one-line description for the driver.
    #[must_use]
    pub fn describe(&self) -> String {
        alloc::format!(
            "layers={} experts={} ppm={} low={} hash={:016x}",
            self.layers,
            self.experts,
            self.low_expert_ppm,
            self.low.len(),
            self.plan_hash
        )
    }
}

fn hash_plan(ppm: u32, low: &[HobbitExpertAddress]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut feed = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    feed(&ppm.to_le_bytes());
    for address in low {
        feed(&address.layer.to_le_bytes());
        feed(&address.expert.to_le_bytes());
    }
    hash
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const TENTH: u64 = FIXED_ONE as u64 / 10;

    fn counts(entries: &[(u32, u32, u64, u64)]) -> Vec<HobbitSelection> {
        entries
            .iter()
            .map(|(layer, expert, count, ema_tenths)| HobbitSelection {
                node: NodeId(*layer * 100),
                expert: *expert,
                count: *count,
                ema_fixed: *ema_tenths * TENTH,
            })
            .collect()
    }

    fn by_layer(node: NodeId) -> Option<usize> {
        Some((node.0 / 100) as usize)
    }

    #[test]
    fn hobbit_universal_plan_is_deterministic_and_coldest_first() {
        let snapshot = counts(&[(0, 0, 10, 10), (0, 1, 2, 2), (0, 2, 0, 0), (0, 3, 0, 0)]);
        let plan = HobbitExpertPlan::from_selection(&snapshot, by_layer, 1, 4, PPM / 2);
        assert_eq!(
            plan.low,
            vec![
                HobbitExpertAddress { layer: 0, expert: 2 },
                HobbitExpertAddress { layer: 0, expert: 3 },
            ],
            "unobserved experts are coldest, then expert id breaks the tie"
        );
        let again = HobbitExpertPlan::from_selection(&snapshot, by_layer, 1, 4, PPM / 2);
        assert_eq!(plan.plan_hash, again.plan_hash);
        assert!(plan.describe().contains("low=2"));
    }

    #[test]
    fn hobbit_universal_plan_ema_breaks_a_count_tie_before_expert_id() {
        let snapshot = counts(&[(0, 0, 5, 9), (0, 1, 5, 1), (0, 2, 5, 5), (0, 3, 5, 3)]);
        let plan = HobbitExpertPlan::from_selection(&snapshot, by_layer, 1, 4, PPM / 2);
        let experts: Vec<u32> = plan.low.iter().map(|address| address.expert).collect();
        assert_eq!(experts, vec![1, 3], "the two lowest ema are low");
    }

    #[test]
    fn hobbit_universal_plan_ppm_bounds_and_zero() {
        let snapshot = counts(&[(0, 0, 1, 10)]);
        let none = HobbitExpertPlan::from_selection(&snapshot, by_layer, 1, 4, 0);
        assert!(none.low.is_empty());
        let all = HobbitExpertPlan::from_selection(&snapshot, by_layer, 1, 4, PPM);
        assert_eq!(all.low.len(), 4);
        let over = HobbitExpertPlan::from_selection(&snapshot, by_layer, 1, 4, 2 * PPM);
        assert_eq!(over.low.len(), 4, "a share above one clamps");
    }

    /// The data files carried the shares 0.25; the integer rule selects the
    /// same counts the `ceil(total * share)` rule did, for every set size.
    #[test]
    fn hobbit_ppm_count_matches_the_shares_the_data_files_carried() {
        for total in 0..=4096u32 {
            assert_eq!(ppm_count(total, PPM / 4), total.div_ceil(4) as usize);
            assert_eq!(ppm_count(total, PPM / 2), total.div_ceil(2) as usize);
            assert_eq!(
                ppm_count(total, 3 * PPM / 4),
                (u64::from(total) * 3).div_ceil(4) as usize
            );
            assert_eq!(ppm_count(total, PPM), total as usize);
            assert_eq!(ppm_count(total, 0), 0);
        }
    }

    /// The control: the same snapshot with a mutated share changes the plan
    /// and its hash, so the hash is reading the plan and not a constant.
    #[test]
    fn control_hobbit_universal_plan_mutation_changes_the_set() {
        let snapshot = counts(&[(0, 0, 10, 10), (0, 1, 1, 1)]);
        let first = HobbitExpertPlan::from_selection(&snapshot, by_layer, 1, 4, PPM / 4);
        let second = HobbitExpertPlan::from_selection(&snapshot, by_layer, 1, 4, 3 * PPM / 4);
        assert_ne!(first.low, second.low);
        assert_ne!(first.plan_hash, second.plan_hash);
    }

    #[test]
    fn hobbit_layer_plan_ranks_by_mean_kl_then_layer() {
        let score = |layer: u32, mean_kl_fixed: u64| HobbitLayerScore {
            layer,
            mean_kl_fixed,
            top1_agreement_fixed: u64::from(FIXED_ONE),
        };
        let scores = vec![score(0, 30), score(1, 10), score(2, 10), score(3, 20)];
        let plan = HobbitLayerPlan::from_measured(scores, 4, PPM / 2);
        assert_eq!(plan.low_layers, vec![1, 2], "the lowest kl, layer id on a tie");
    }

    #[test]
    fn fixed_point_scales_and_floors_at_zero() {
        let delta = crate::hobbit::HobbitQualityDelta {
            positions: 0,
            top1_agreement: 0.5,
            mean_kl: -0.5,
        };
        assert_eq!(fixed_point!(delta.top1_agreement), u64::from(FIXED_ONE) / 2);
        assert_eq!(fixed_point!(delta.mean_kl), 0);
    }
}

/// A per-layer high/low plan for the universal grouped-GEMM path: every
/// expert of a low layer is replaced by one contiguous low-codec stack, so
/// the graph's grouped GEMM sees a uniform packed codec on every backend.
/// The per-expert plan above needs a gather-reduce lowering and is the
/// finer residual tier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HobbitLayerPlan {
    pub low_layer_ppm: u32,
    pub layers: u32,
    /// Sorted low model-layer indices.
    pub low_layers: Vec<u32>,
    /// The held-out per-layer score the plan was ranked on, in layer order.
    pub layer_scores: Vec<HobbitLayerScore>,
    pub plan_hash: u64,
}

/// One layer's measured held-out delta when only that layer's expert stacks
/// were low, both in [`FIXED_ONE`] fixed point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HobbitLayerScore {
    pub layer: u32,
    pub mean_kl_fixed: u64,
    pub top1_agreement_fixed: u64,
}

impl HobbitLayerPlan {
    /// Ranks layers by the sum of their experts' selection counts; the
    /// coldest `low_layer_ppm` parts per million become low. Unobserved
    /// layers are coldest.
    #[must_use]
    pub fn from_selection(
        counts: &[HobbitSelection],
        layer_for_node: impl Fn(NodeId) -> Option<usize>,
        layers: u32,
        low_layer_ppm: u32,
    ) -> Self {
        let ppm = low_layer_ppm.min(PPM);
        let mut per_layer = alloc::vec![0u64; layers as usize];
        for selection in counts {
            if let Some(layer) = layer_for_node(selection.node)
                && let Some(slot) = per_layer.get_mut(layer)
            {
                *slot = slot.saturating_add(selection.count);
            }
        }
        let mut ranked: alloc::vec::Vec<(u64, u32)> = (0..layers)
            .map(|layer| (per_layer[layer as usize], layer))
            .collect();
        ranked.sort_unstable();
        let take = ppm_count(layers, ppm);
        let mut low_layers: Vec<u32> = ranked
            .into_iter()
            .take(take)
            .map(|(_, layer)| layer)
            .collect();
        low_layers.sort_unstable();
        let plan_hash = hash_layers(ppm, &low_layers);
        Self {
            low_layer_ppm: ppm,
            layers,
            low_layers,
            layer_scores: Vec::new(),
            plan_hash,
        }
    }

    /// Builds the plan from measured single-layer deltas: the least
    /// sensitive layers are low, then the combined set is the caller's to
    /// gate. Deterministic tie-break by layer index.
    #[must_use]
    pub fn from_measured(
        mut scores: Vec<HobbitLayerScore>,
        layers: u32,
        low_layer_ppm: u32,
    ) -> Self {
        let ppm = low_layer_ppm.min(PPM);
        scores.sort_by_key(|score| (score.mean_kl_fixed, score.layer));
        let take = ppm_count(layers, ppm);
        let mut low_layers: Vec<u32> = scores.iter().take(take).map(|score| score.layer).collect();
        low_layers.sort_unstable();
        let plan_hash = hash_layers(ppm, &low_layers);
        Self {
            low_layer_ppm: ppm,
            layers,
            low_layers,
            layer_scores: scores,
            plan_hash,
        }
    }

    /// Rebuilds the plan from an explicit low set (used by the quality
    /// search when the combined set exceeds the bound). The hash is
    /// recomputed over the sorted set.
    #[must_use]
    pub fn from_low_layers(
        layers: u32,
        low_layer_ppm: u32,
        mut low_layers: Vec<u32>,
        layer_scores: Vec<HobbitLayerScore>,
    ) -> Self {
        low_layers.sort_unstable();
        low_layers.dedup();
        let plan_hash = hash_layers(low_layer_ppm, &low_layers);
        Self {
            low_layer_ppm,
            layers,
            low_layers,
            layer_scores,
            plan_hash,
        }
    }

    #[must_use]
    pub fn is_low(&self, layer: u32) -> bool {
        self.low_layers.binary_search(&layer).is_ok()
    }

    #[must_use]
    pub fn describe(&self) -> String {
        alloc::format!(
            "layers={} ppm={} low_layers={:?} hash={:016x}",
            self.layers,
            self.low_layer_ppm,
            self.low_layers,
            self.plan_hash
        )
    }
}

fn hash_layers(ppm: u32, low: &[u32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut feed = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    feed(&ppm.to_le_bytes());
    for layer in low {
        feed(&layer.to_le_bytes());
    }
    hash
}
