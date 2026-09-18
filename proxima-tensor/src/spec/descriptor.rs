use super::*;

/// Which forward-program engine [`ModelDescriptor::cache_strategy`] selects
/// for a (future) generic `build_forward` -- [`CacheStrategy::Cacheless`] is
/// [`lfm2_forward_program_with_experts`], today's default path (full
/// reprefill every call); [`CacheStrategy::TwoRange`] is
/// [`lfm2_two_range_cached_forward_program_with_experts`], the WORKING
/// gemma4 two-range kv-cache behind the default-off `gemma4-kv-cache`
/// feature. Genuinely new: no existing type names which forward-program
/// engine a schedule targets -- that choice lives today as a
/// `#[cfg(feature = "gemma4-kv-cache")]` compile-time split
/// (`proxima-model-interop::gemma4::bind::Gemma4Arch::bind`), not as data a
/// caller can hold and branch on at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStrategy {
    Cacheless,
    TwoRange,
}

/// A whole model's build-time shape as DATA: the global hyperparameters
/// [`lfm2_forward_program_with_experts`] already takes as loose positional
/// arguments (`vocab`, `embedding`, `block_count`, `expert_count`,
/// `expert_used_count`, `leading_dense_block_count`, `embedding_scale`,
/// `logit_softcap`), one [`LayerSchedule`] per block, and which cache engine
/// to build with. `layers` reuses [`LayerSchedule`] verbatim -- it already
/// composes [`LayerKind`], [`LayerAttentionConfig`], and [`LayerFfnConfig`]
/// (`proxima-model-interop::gemma4::bind::gemma4_layer_schedule` builds
/// exactly this shape today, by hand, at bind time), so this struct does not
/// re-mint a per-layer type. Genuinely new: no existing type bundles a
/// model's full hyperparameter set with its per-layer schedule and a
/// cache-engine choice into one value a caller can build ahead of time --
/// today every caller re-derives and re-passes these as separate positional
/// arguments at each forward-program call site.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelDescriptor {
    pub vocab: u32,
    pub embedding: u32,
    pub block_count: u32,
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub leading_dense_block_count: u32,
    pub embedding_scale: Option<EmbeddingScale>,
    pub logit_softcap: Option<f32>,
    pub layers: Vec<LayerSchedule>,
    pub cache_strategy: CacheStrategy,
}

/// Real gemma4 26B-A4B header values, proven by the `#[ignore]`d
/// `builtin_registry_routes_real_gemma4_header_with_exact_tensor_directory`
/// (`proxima-model-interop/tests/real_gemma4_registry_probe.rs`), which
/// asserts each of these against the real checkpoint's own GGUF metadata.
const GEMMA4_BLOCK_COUNT: u32 = 30;
const GEMMA4_EXPERT_COUNT: u32 = 128;
const GEMMA4_EXPERT_USED_COUNT: u32 = 8;
const GEMMA4_EMBEDDING: u32 = 2816;
const GEMMA4_LOGIT_SOFTCAP: f32 = 30.0;
/// Full-attention head dim (`architecture.key_length`); sliding layers use
/// [`GEMMA4_HEAD_DIM_SWA`] instead.
const GEMMA4_HEAD_DIM_FULL: u32 = 512;
const GEMMA4_HEAD_DIM_SWA: u32 = 256;
const GEMMA4_KV_HEADS_FULL: u32 = 2;
const GEMMA4_KV_HEADS_SWA: u32 = 8;
const GEMMA4_SLIDING_WINDOW: u32 = 1024;
/// The real header marks every 6th layer (0-indexed 5, 11, 17, 23, 29) full
/// attention, every other layer sliding -- `(layer + 1).is_multiple_of(6)`
/// mirrors the real registry probe's own `expected_sliding_window_pattern`.
const GEMMA4_FULL_LAYER_PERIOD: u32 = 6;
/// `Gemma4Arch::bind`'s own two call sites into
/// `lfm2_forward_program_with_experts`/`lfm2_two_range_cached_forward_program_with_experts`
/// (`proxima-model-interop/src/gemma4/bind.rs`) both pass `0` here: every
/// gemma4 layer runs the parallel dense+MoE combination
/// ([`FfnCombination::ParallelDenseMoe`]), so there is no leading run of
/// dense-only blocks to select past.
const GEMMA4_LEADING_DENSE_BLOCK_COUNT: u32 = 0;

/// Builds the real gemma4 26B-A4B [`ModelDescriptor`] -- SWA/full dual-base
/// RoPE, unscaled attention score, value-norm, shared-KV on full layers,
/// parallel dense+MoE FFN, final-logit softcap 30. Every field this
/// function sets mirrors
/// `proxima-model-interop::gemma4::bind::gemma4_layer_schedule` node-for-node
/// (same [`LayerAttentionConfig`]/[`LayerFfnConfig`] values, same
/// sliding-vs-full split), just built here as plain data instead of at GGUF
/// bind time. `vocab` is a parameter rather than a baked-in constant because
/// it is genuinely per-checkpoint data (`hparams::Architecture::vocab` reads
/// it from the `token_embd.weight` tensor's own row count, not from a fixed
/// architecture metadata field) -- every other value here is architecture,
/// not tokenizer, and is proven fixed by the real registry probe cited on
/// each constant above.
#[must_use]
pub fn gemma4_descriptor(vocab: u32) -> ModelDescriptor {
    let ffn = LayerFfnConfig {
        post_attention_norm: true,
        combination: FfnCombination::ParallelDenseMoe(ParallelDenseMoeConfig {
            dense_post_norm: true,
            routed_post_norm: true,
            combined_post_norm: true,
            routed_pre_norm: true,
            router_scale: true,
            expert_output_scale: true,
        }),
        output_scale: true,
        routed_gating: ExpertGatingFunc::Softmax,
        routed_expert_bias: false,
        activation: Activation::GeluTanh,
    };

    let layers: Vec<LayerSchedule> = (0..GEMMA4_BLOCK_COUNT)
        .map(|layer| {
            let is_full = (layer + 1).is_multiple_of(GEMMA4_FULL_LAYER_PERIOD);
            let attention = if is_full {
                LayerAttentionConfig {
                    head_dim: GEMMA4_HEAD_DIM_FULL,
                    kv_heads: GEMMA4_KV_HEADS_FULL,
                    mask_window: None,
                    value_source_kind: ValueSourceKind::SharedWithKey,
                    rope_table: RopeTableSel {
                        cos_name: "rope_cos",
                        sin_name: "rope_sin",
                    },
                    rope_pairing: RopePairing::SplitHalf {
                        pairs: GEMMA4_HEAD_DIM_FULL / 2,
                    },
                    score_scale: AttentionScoreScale::Unscaled,
                    value_norm: true,
                }
            } else {
                LayerAttentionConfig {
                    head_dim: GEMMA4_HEAD_DIM_SWA,
                    kv_heads: GEMMA4_KV_HEADS_SWA,
                    mask_window: Some(GEMMA4_SLIDING_WINDOW),
                    value_source_kind: ValueSourceKind::ProjectedV,
                    rope_table: RopeTableSel {
                        cos_name: "rope_cos_swa",
                        sin_name: "rope_sin_swa",
                    },
                    rope_pairing: RopePairing::SplitHalf {
                        pairs: GEMMA4_HEAD_DIM_SWA / 2,
                    },
                    score_scale: AttentionScoreScale::Unscaled,
                    value_norm: true,
                }
            };
            LayerSchedule {
                kind: LayerKind::Attention,
                attention,
                ffn,
            }
        })
        .collect();

    ModelDescriptor {
        vocab,
        embedding: GEMMA4_EMBEDDING,
        block_count: GEMMA4_BLOCK_COUNT,
        expert_count: GEMMA4_EXPERT_COUNT,
        expert_used_count: GEMMA4_EXPERT_USED_COUNT,
        leading_dense_block_count: GEMMA4_LEADING_DENSE_BLOCK_COUNT,
        embedding_scale: Some(EmbeddingScale::Sqrt),
        logit_softcap: Some(GEMMA4_LOGIT_SOFTCAP),
        layers,
        cache_strategy: CacheStrategy::Cacheless,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read from `token_embd.weight`'s own row count at bind time on a real
    /// checkpoint (`hparams::Architecture::vocab`) -- any value proves this
    /// shape test, since nothing here reads `vocab` back out.
    const TEST_VOCAB: u32 = 32;

    #[test]
    fn gemma4_descriptor_matches_real_checkpoint_shape() {
        let descriptor = gemma4_descriptor(TEST_VOCAB);

        assert_eq!(descriptor.block_count, 30);
        assert_eq!(descriptor.layers.len(), 30);
        assert_eq!(descriptor.expert_count, 128);
        assert_eq!(descriptor.expert_used_count, 8);
        assert_eq!(descriptor.embedding, 2816);
        assert_eq!(descriptor.leading_dense_block_count, 0);
        assert_eq!(descriptor.logit_softcap, Some(30.0));
        assert_eq!(descriptor.embedding_scale, Some(EmbeddingScale::Sqrt));
        assert_eq!(descriptor.cache_strategy, CacheStrategy::Cacheless);

        let full_layers: Vec<u32> = descriptor
            .layers
            .iter()
            .enumerate()
            .filter(|(_, layer)| layer.attention.mask_window.is_none())
            .map(|(index, _)| index as u32)
            .collect();
        assert_eq!(full_layers, alloc::vec![5, 11, 17, 23, 29]);

        for (index, layer) in descriptor.layers.iter().enumerate() {
            assert_eq!(layer.kind, LayerKind::Attention);
            match layer.ffn.combination {
                FfnCombination::ParallelDenseMoe(_) => {}
                FfnCombination::Exclusive => {
                    panic!("layer {index} must run parallel dense+MoE FFN")
                }
            }
        }

        let sliding_kv_heads: Vec<u32> = descriptor
            .layers
            .iter()
            .filter(|layer| layer.attention.mask_window.is_some())
            .map(|layer| layer.attention.kv_heads)
            .collect();
        assert!(
            sliding_kv_heads.iter().all(|&kv_heads| kv_heads == 8),
            "every sliding layer uses 8 kv-heads: {sliding_kv_heads:?}"
        );

        let full_kv_heads: Vec<u32> = descriptor
            .layers
            .iter()
            .filter(|layer| layer.attention.mask_window.is_none())
            .map(|layer| layer.attention.kv_heads)
            .collect();
        assert!(
            full_kv_heads.iter().all(|&kv_heads| kv_heads == 2),
            "every full layer uses 2 kv-heads: {full_kv_heads:?}"
        );
    }
}
