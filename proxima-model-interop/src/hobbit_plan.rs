//! A deterministic per-expert high/low plan for the universal HOBBIT path.
//!
//! The signal is the model's own routed-expert selection counter
//! ([`proxima_tensor::cpu::snapshot_expert_selection_top_n`], populated by one
//! calibration run). Per model layer, the experts are ranked by
//! `(count, ema, expert)`; the coldest `low_expert_fraction` are assigned the
//! low-codec sidecar copy. The plan names no family and no layer: the layer
//! index and expert count come from the checkpoint's bound slab.

use alloc::string::String;
use alloc::vec::Vec;
use std::collections::BTreeMap;

use proxima_tensor::NodeId;
use serde::{Deserialize, Serialize};

/// One low expert address, in model-layer space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct HobbitExpertAddress {
    pub layer: u32,
    pub expert: u32,
}

/// A sorted low-expert set plus the deterministic ranking hash.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HobbitExpertPlan {
    pub low_expert_fraction: f32,
    pub layers: u32,
    pub experts: u32,
    /// Sorted by `(layer, expert)`.
    pub low: Vec<HobbitExpertAddress>,
    /// FNV-1a over the ordered low set and the fraction bits.
    pub plan_hash: u64,
}

impl HobbitExpertPlan {
    /// Builds the plan from one selection snapshot. `layer_for_node` maps a
    /// routed weight node to its model layer (the slab's own mapping);
    /// unobserved experts are coldest.
    #[must_use]
    pub fn from_selection(
        counts: &[(NodeId, u32, u64, f64)],
        layer_for_node: impl Fn(NodeId) -> Option<usize>,
        layers: u32,
        experts: u32,
        low_expert_fraction: f32,
    ) -> Self {
        let fraction = low_expert_fraction.clamp(0.0, 1.0);
        let mut observed: BTreeMap<(u32, u32), (u64, f64)> = BTreeMap::new();
        for (node, expert, count, ema) in counts {
            let Some(layer) = layer_for_node(*node) else {
                continue;
            };
            if (layer as u32) >= layers || *expert >= experts {
                continue;
            }
            observed.insert((layer as u32, *expert), (*count, *ema));
        }
        let per_layer = ((experts as f64 * f64::from(fraction)).ceil() as usize)
            .min(experts as usize);
        let mut low = Vec::with_capacity(per_layer * layers as usize);
        for layer in 0..layers {
            let mut ranked: Vec<(u64, f64, u32)> = (0..experts)
                .map(|expert| {
                    let (count, ema) = observed
                        .get(&(layer, expert))
                        .copied()
                        .unwrap_or((0, 0.0));
                    (count, ema, expert)
                })
                .collect();
            ranked.sort_by(|left, right| {
                left.0
                    .cmp(&right.0)
                    .then_with(|| left.1.total_cmp(&right.1))
                    .then_with(|| left.2.cmp(&right.2))
            });
            for (_, _, expert) in ranked.into_iter().take(per_layer) {
                low.push(HobbitExpertAddress { layer, expert });
            }
        }
        low.sort();
        let plan_hash = hash_plan(fraction, &low);
        Self {
            low_expert_fraction: fraction,
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
            "layers={} experts={} fraction={} low={} hash={:016x}",
            self.layers,
            self.experts,
            self.low_expert_fraction,
            self.low.len(),
            self.plan_hash
        )
    }
}

fn hash_plan(fraction: f32, low: &[HobbitExpertAddress]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut feed = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    feed(&fraction.to_bits().to_le_bytes());
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

    fn counts(entries: &[(u32, u32, u64, f64)]) -> Vec<(NodeId, u32, u64, f64)> {
        entries
            .iter()
            .map(|(layer, expert, count, ema)| (NodeId(*layer * 100), *expert, *count, *ema))
            .collect()
    }

    #[test]
    fn hobbit_universal_plan_is_deterministic_and_coldest_first() {
        let snapshot = counts(&[
            (0, 0, 10, 1.0),
            (0, 1, 2, 0.2),
            (0, 2, 0, 0.0),
            (0, 3, 0, 0.0),
        ]);
        let plan = HobbitExpertPlan::from_selection(
            &snapshot,
            |node| Some((node.0 / 100) as usize),
            1,
            4,
            0.5,
        );
        assert_eq!(
            plan.low,
            vec![
                HobbitExpertAddress { layer: 0, expert: 2 },
                HobbitExpertAddress { layer: 0, expert: 3 },
            ],
            "unobserved experts are coldest, then expert id breaks the tie"
        );
        let again = HobbitExpertPlan::from_selection(
            &snapshot,
            |node| Some((node.0 / 100) as usize),
            1,
            4,
            0.5,
        );
        assert_eq!(plan.plan_hash, again.plan_hash);
        assert!(plan.describe().contains("low=2"));
    }

    #[test]
    fn hobbit_universal_plan_fraction_bounds_and_zero() {
        let snapshot = counts(&[(0, 0, 1, 1.0)]);
        let none = HobbitExpertPlan::from_selection(
            &snapshot,
            |node| Some((node.0 / 100) as usize),
            1,
            4,
            0.0,
        );
        assert!(none.low.is_empty());
        let all = HobbitExpertPlan::from_selection(
            &snapshot,
            |node| Some((node.0 / 100) as usize),
            1,
            4,
            1.0,
        );
        assert_eq!(all.low.len(), 4);
        let over = HobbitExpertPlan::from_selection(
            &snapshot,
            |node| Some((node.0 / 100) as usize),
            1,
            4,
            2.0,
        );
        assert_eq!(over.low.len(), 4, "a fraction above one clamps");
    }

    /// The control: the same snapshot with a mutated fraction changes the
    /// plan and its hash, so the hash is reading the plan and not a constant.
    #[test]
    fn control_hobbit_universal_plan_mutation_changes_the_set() {
        let snapshot = counts(&[(0, 0, 10, 1.0), (0, 1, 1, 0.1)]);
        let first = HobbitExpertPlan::from_selection(
            &snapshot,
            |node| Some((node.0 / 100) as usize),
            1,
            4,
            0.25,
        );
        let second = HobbitExpertPlan::from_selection(
            &snapshot,
            |node| Some((node.0 / 100) as usize),
            1,
            4,
            0.75,
        );
        assert_ne!(first.low, second.low);
        assert_ne!(first.plan_hash, second.plan_hash);
    }
}

/// A per-layer high/low plan for the universal grouped-GEMM path: every
/// expert of a low layer is replaced by one contiguous low-codec stack, so
/// the graph's grouped GEMM sees a uniform packed codec on every backend.
/// The per-expert plan above needs a gather-reduce lowering and is the
/// finer residual tier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HobbitLayerPlan {
    pub low_layer_fraction: f32,
    pub layers: u32,
    /// Sorted low model-layer indices.
    pub low_layers: Vec<u32>,
    /// The held-out per-layer score the plan was ranked on, in layer order.
    pub layer_scores: Vec<HobbitLayerScore>,
    pub plan_hash: u64,
}

/// One layer's measured held-out delta when only that layer's expert stacks
/// were low.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HobbitLayerScore {
    pub layer: u32,
    pub mean_kl: f64,
    pub top1_agreement: f64,
}

impl HobbitLayerPlan {
    /// Ranks layers by the sum of their experts' selection counts; the
    /// coldest `low_layer_fraction` become low. Unobserved layers are
    /// coldest.
    #[must_use]
    pub fn from_selection(
        counts: &[(NodeId, u32, u64, f64)],
        layer_for_node: impl Fn(NodeId) -> Option<usize>,
        layers: u32,
        low_layer_fraction: f32,
    ) -> Self {
        let fraction = low_layer_fraction.clamp(0.0, 1.0);
        let mut per_layer = alloc::vec![0u64; layers as usize];
        for (node, _expert, count, _ema) in counts {
            if let Some(layer) = layer_for_node(*node)
                && let Some(slot) = per_layer.get_mut(layer)
            {
                *slot = slot.saturating_add(*count);
            }
        }
        let mut ranked: alloc::vec::Vec<(u64, u32)> = (0..layers)
            .map(|layer| (per_layer[layer as usize], layer))
            .collect();
        ranked.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
        let take = ((layers as f64 * f64::from(fraction)).ceil() as usize)
            .min(layers as usize);
        let mut low_layers: Vec<u32> = ranked
            .into_iter()
            .take(take)
            .map(|(_, layer)| layer)
            .collect();
        low_layers.sort_unstable();
        let plan_hash = hash_layers(fraction, &low_layers);
        Self {
            low_layer_fraction: fraction,
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
        low_layer_fraction: f32,
    ) -> Self {
        let fraction = low_layer_fraction.clamp(0.0, 1.0);
        scores.sort_by(|left, right| {
            left.mean_kl
                .total_cmp(&right.mean_kl)
                .then_with(|| left.layer.cmp(&right.layer))
        });
        let take = ((layers as f64 * f64::from(fraction)).ceil() as usize)
            .min(layers as usize);
        let mut low_layers: Vec<u32> = scores.iter().take(take).map(|score| score.layer).collect();
        low_layers.sort_unstable();
        let plan_hash = hash_layers(fraction, &low_layers);
        Self {
            low_layer_fraction: fraction,
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
        low_layer_fraction: f32,
        mut low_layers: Vec<u32>,
        layer_scores: Vec<HobbitLayerScore>,
    ) -> Self {
        low_layers.sort_unstable();
        low_layers.dedup();
        let plan_hash = hash_layers(low_layer_fraction, &low_layers);
        Self {
            low_layer_fraction,
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
            "layers={} fraction={} low_layers={:?} hash={:016x}",
            self.layers,
            self.low_layer_fraction,
            self.low_layers,
            self.plan_hash
        )
    }
}

fn hash_layers(fraction: f32, low: &[u32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut feed = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    feed(&fraction.to_bits().to_le_bytes());
    for layer in low {
        feed(&layer.to_le_bytes());
    }
    hash
}
