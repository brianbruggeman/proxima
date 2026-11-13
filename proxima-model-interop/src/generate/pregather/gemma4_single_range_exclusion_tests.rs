use super::*;

/// the E2B checkpoint's own shape (`sliding_pattern::bind::declared_leaves_match_bound_leaves_tests::e2b_shaped_architecture`'s
/// own doc: 35 layers, `blk.15..=34` shared-KV) flattened into the
/// generic [`ModelHparams`] `build_single_range_program` actually
/// receives -- that type carries no sliding-window-pattern or
/// shared-KV-layer-count field at all, so this shape is
/// indistinguishable from an ordinary 35-layer dense uniform checkpoint
/// at this builder's own boundary.
fn gemma4_e2b_shaped_model_architecture() -> ModelHparams {
    ModelHparams {
        vocab: 1,
        embedding: 1536,
        feed_forward: 6144,
        expert_feed_forward: 6144,
        query_heads: 8,
        kv_heads: 1,
        kv_heads_by_layer: alloc::vec![1; 35],
        head_dim: 512,
        block_count: 35,
        expert_count: 0,
        expert_used_count: 0,
        rope_freq_base: 1_000_000.0,
        rms_epsilon: 1e-6,
        tied_embeddings: false,
        family: String::from("gemma4"),
        sliding_rope: None,
    }
}

/// The hazard the `"sliding-pattern"` exclusion exists to prevent, proven
/// directly: called on the E2B checkpoint's own shape,
/// `build_single_range_program` (`gqa_single_range_cached_forward_program`'s
/// own doc: dense-uniform-only, ONE uniform per-layer schedule, no
/// `KeySourceKind::SharedFromLayer` concept at all) still declares
/// `blk.15.attn_k.weight` -- the exact leaf name the real checkpoint's
/// own Metal run panicked on with `UnboundInputName` before this fix,
/// since the leaf binder never binds that tensor for a shared-KV
/// layer. Proves the exclusion at `Self::load` is load-bearing, not
/// dead code guarding against a case that could not occur anyway.
#[test]
#[allow(clippy::expect_used)]
fn build_single_range_program_declares_blk15_attn_k_for_gemma4_shaped_architecture() {
    let architecture = gemma4_e2b_shaped_model_architecture();
    let single_range = build_single_range_program(&architecture, false)
        .expect("gqa_single_range_cached_forward_program builds for a uniform 35-layer shape")
        .expect("expert_count == 0 does not turn this builder away");
    let declares_blk15_attn_k = single_range.program.iter().any(|operation| {
        matches!(
            operation,
            proxima_tensor::op::Op::Input { name: Some(name), .. }
                if name == "blk.15.attn_k.weight"
        )
    });
    assert!(
        declares_blk15_attn_k,
        "build_single_range_program has no gemma4 shared-KV awareness: it must still \
         declare blk.15.attn_k.weight for a uniform 35-layer shape, proving \
         Self::load's own gemma4 exclusion is load-bearing, not dead code"
    );
}

/// The actual production gate reads the architecture's measured
/// [`crate::serving_defaults::MeasuredServingDefaults::kv_cache_shape`] -- reproduced
/// here directly so flipping the sliding-pattern default back to uniform breaks
/// this test rather than silently reopening the panic this fix closed.
#[test]
#[allow(clippy::expect_used)]
fn load_single_range_exclusion_covers_gemma4() {
    assert_ne!(
        measured_serving_defaults("gemma4")
            .expect("gemma4 serving defaults embedded")
            .kv_cache_shape,
        KvCacheShape::Uniform,
        "gemma4 must stay excluded from the placed-KV single-range program"
    );
}

/// `LFM2-350M-Q4_K_M.gguf`'s own shape: no `expert_count` key, and an
/// `attention.head_count_kv` of `[0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8, 0, 8, 0, 8, 0]`,
/// which `architecture_from_metadata` reads as `kv_heads = 0` because the
/// per-layer counts differ. The zero entries are short-convolution layers
/// with no attention block for the single-range program to place.
#[test]
#[allow(clippy::expect_used)]
fn a_dense_hybrid_with_cacheless_layers_gets_no_single_range_program() {
    let architecture = ModelHparams {
        vocab: 65536,
        embedding: 1024,
        feed_forward: 4608,
        expert_feed_forward: 0,
        query_heads: 16,
        kv_heads: 0,
        kv_heads_by_layer: alloc::vec![0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8, 0, 8, 0, 8, 0],
        head_dim: 64,
        block_count: 16,
        expert_count: 0,
        expert_used_count: 0,
        rope_freq_base: 1_000_000.0,
        rms_epsilon: 1e-5,
        tied_embeddings: true,
        family: String::from("lfm2"),
        sliding_rope: None,
    };

    let single_range = build_single_range_program(&architecture, true)
        .expect("a hybrid is turned away, not built");

    assert!(single_range.is_none());
}
