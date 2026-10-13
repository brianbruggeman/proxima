//! Real-program guard for CUDA kernel reuse across a KV extent change: a
//! gqa decode program bound at two cached lengths must reuse kernels whose
//! source, entry, bindings, grid and extra outputs equal a fresh emit op for op.

#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;

use omega::{emit_cuda_reusing, emit_cuda_with_policy};
use proxima_tensor::spec::qk_norm_cached_forward_program;
use proxima_tensor::{BoundOp, NodeId, NumericPolicy, bind_with_fusion, infer, prune_dead};

const VOCAB: u32 = 64;
const EMBEDDING: u32 = 512;
const FEED_FORWARD: u32 = 128;
const QUERY_HEADS: u32 = 8;
const KV_HEADS: u32 = 1;
const HEAD_DIM: u32 = 64;
const LAYERS: u32 = 2;

fn resolved_at(cached_len: u64) -> Vec<BoundOp> {
    let (program, logits, _) = qk_norm_cached_forward_program(
        VOCAB,
        EMBEDDING,
        FEED_FORWARD,
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
        LAYERS,
    )
    .expect("the gqa decode program builds");
    let shapes = infer(&program, &[1, cached_len]).expect("the decode program infers");
    let outputs: Vec<NodeId> = vec![logits];
    let bound = bind_with_fusion(&program, &shapes, &outputs, false, NumericPolicy::default())
        .expect("the decode program binds");
    prune_dead(bound, &outputs)
}

#[test]
fn reused_kernels_equal_fresh_emit_for_every_op_of_a_real_decode_program() {
    let policy = NumericPolicy::default();
    let packed = BTreeMap::new();
    let previous = resolved_at(31);
    let resolved = resolved_at(33);
    let previous_kernels = previous
        .iter()
        .map(|bound| emit_cuda_with_policy(bound, &packed, policy).expect("previous op emits"))
        .collect::<Vec<_>>();
    let (kernels, reused) =
        emit_cuda_reusing(&previous, previous_kernels, &resolved, &packed, policy)
            .expect("reuse succeeds");
    assert!(!resolved.is_empty(), "the decode program has ops");
    assert!(
        previous.iter().zip(&resolved).any(|(old, new)| old.extents != new.extents),
        "the cached length change must move at least one op extent"
    );
    assert_eq!(
        reused,
        resolved.len(),
        "every op of the decode program reuses its kernel across the extent change"
    );
    for (bound, kernel) in resolved.iter().zip(&kernels) {
        let fresh = emit_cuda_with_policy(bound, &packed, policy).expect("fresh op emits");
        assert_eq!(kernel.source, fresh.source, "source for node {:?}", bound.node);
        assert_eq!(kernel.entry, fresh.entry, "entry for node {:?}", bound.node);
        assert_eq!(kernel.bindings, fresh.bindings, "bindings for node {:?}", bound.node);
        assert_eq!(kernel.grid, fresh.grid, "grid for node {:?}", bound.node);
        assert_eq!(kernel, &fresh, "whole kernel for node {:?}", bound.node);
    }
}
