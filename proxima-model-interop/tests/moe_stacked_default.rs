//! The `metal` feature set lowers every routed feed-forward block with one
//! operation per projection over the selected experts
//! ([`MoeProjectionStrategy::Stacked`]) instead of one per selected expert, and
//! brings the top-k fusion that folds the route stacks. Feature unification
//! reaches `proxima-tensor` through this crate's passthrough, so this is the
//! place that proves the serving build selects it.

#![cfg(feature = "metal")]

use proxima_tensor::spec::MoeProjectionStrategy;

#[test]
fn the_metal_feature_set_builds_routed_experts_as_stacked_projections() {
    assert_eq!(
        MoeProjectionStrategy::production(),
        MoeProjectionStrategy::Stacked
    );
}

const _: () = assert!(
    cfg!(feature = "moe-topk-fusion") && cfg!(feature = "moe-stacked-experts"),
    "metal must bring the stacked experts and the top-k fusion that folds their route stacks"
);
