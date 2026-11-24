use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use super::{
    Attention, Block, Decode, DenseFfn, Head, Input, Network, Norm, NormKind, RoutedFfn, Stack,
};
use crate::error::TensorError;
use crate::spec::{
    Activation, CacheMask, CacheStrategy, ExpertGatingFunc, FfnCombination, KeySourceKind,
    LayerAttentionConfig, LayerKind, LayerSchedule, ModelDescriptor, RopePairing,
    ValueSourceKind, refuse_when,
};

// the refusals name the entry point a caller used, so the arm's own name stays
const BUILDER: &str = "build_forward(CacheMask::Bounded)";

impl Network {
    /// The network the Bounded cached engine lowers: per layer one residual
    /// attention sublayer and one residual feed-forward sublayer, a token
    /// input and a norm-plus-projection head. Every tensor name is read from
    /// `descriptor.dialect` through [`Pattern::render`](crate::spec::Pattern::render),
    /// never from a block's position.
    ///
    /// The refusals are the Bounded arm's own plus one per field the arm used
    /// to ignore: a request is lowered as asked or rejected by name.
    ///
    /// # Errors
    ///
    /// [`TensorError::UnsupportedInBuilder`] naming the field this engine
    /// cannot honour, [`TensorError::LayerScheduleCountMismatch`] when the
    /// schedule and `block_count` disagree.
    pub fn from_bounded(descriptor: &ModelDescriptor) -> Result<Self, TensorError> {
        let first = refuse_unsupported(descriptor)?;
        let mut blocks = Vec::with_capacity(descriptor.layers.len() * 2);
        for (index, layer) in descriptor.layers.iter().enumerate() {
            let index = u32::try_from(index).map_err(|_| TensorError::UnsupportedInBuilder {
                builder: BUILDER,
                feature: "a layer index past u32",
            })?;
            blocks.push(attention_sublayer(descriptor, first, layer, index)?);
            blocks.push(ffn_sublayer(descriptor, index)?);
        }
        Ok(Self {
            input: Input::Tokens {
                vocab: descriptor.vocab,
                width: descriptor.embedding,
                table: global_leaf(descriptor, "embedding")?,
                scale: descriptor.embedding_scale,
                activation: descriptor.activation,
            },
            body: Stack { blocks },
            head: Head {
                norm: Norm {
                    kind: NormKind::Rms,
                    eps: None,
                    weight: Some(global_leaf(descriptor, "output_norm")?),
                    group_width: None,
                },
                output: global_leaf(descriptor, "output")?,
                logit_scale: descriptor.logit_scale,
            },
            decode: Decode {
                speculative_verify: descriptor.speculative_verify,
                command_buffer_chunks: 1,
                cache_strategy: CacheStrategy::Cached,
                cache_mask: CacheMask::Bounded,
                prefill_width: descriptor.prefill_width,
                last_row_only: descriptor.last_row_only,
                sliding_kv_ring: descriptor.sliding_kv_ring,
            },
        })
    }
}

fn refuse_unsupported(descriptor: &ModelDescriptor) -> Result<&LayerSchedule, TensorError> {
    if descriptor.layers.len() != descriptor.block_count as usize {
        return Err(TensorError::LayerScheduleCountMismatch {
            expected: descriptor.block_count,
            found: descriptor.layers.len(),
        });
    }
    let Some(first) = descriptor.layers.first() else {
        return Err(TensorError::LayerScheduleCountMismatch {
            expected: descriptor.block_count,
            found: 0,
        });
    };
    let any = |holds: fn(&LayerSchedule) -> bool| descriptor.layers.iter().any(holds);
    refuse_when(
        any(|layer| layer.attention.attention_sinks),
        BUILDER,
        "attention sinks (this engine has no per-layer sink channel)",
    )?;
    refuse_when(
        any(|layer| layer.ffn.combination == FfnCombination::RoutedWithSharedExpert),
        BUILDER,
        "a routed-with-shared-expert FFN (this engine lowers the routed branch only)",
    )?;
    refuse_when(
        any(|layer| !layer.attention.rope),
        BUILDER,
        "a layer with rope off (this engine has no per-layer rope channel)",
    )?;
    refuse_when(
        descriptor.layers.iter().any(|layer| {
            LayerAttentionConfig {
                mask_window: first.attention.mask_window,
                ..layer.attention.clone()
            } != first.attention
        }),
        BUILDER,
        "non-uniform per-layer attention config",
    )?;
    refuse_when(
        any(|layer| layer.kind != LayerKind::Attention),
        BUILDER,
        "a layer that is not attention",
    )?;
    refuse_when(
        descriptor.expert_count > 0 && moe_pairing_differs(descriptor, first),
        BUILDER,
        "a rope pairing the moe layer cannot express",
    )?;
    refuse_new_fields(descriptor)?;
    Ok(first)
}

fn moe_pairing_differs(descriptor: &ModelDescriptor, first: &LayerSchedule) -> bool {
    let derived = if descriptor.qk_norm {
        RopePairing::SplitHalf {
            pairs: first.attention.head_dim / 2,
        }
    } else {
        RopePairing::Interleaved
    };
    first.attention.rope_pairing != derived
}

fn refuse_new_fields(descriptor: &ModelDescriptor) -> Result<(), TensorError> {
    let routed = descriptor.expert_count > 0;
    refuse_when(
        routed && descriptor.qkv_biases,
        BUILDER,
        "qkv_biases with routed experts (the routed layer declares no bias leaves)",
    )?;
    refuse_when(
        routed && descriptor.fused_qkv_reduce,
        BUILDER,
        "fused_qkv_reduce with routed experts (the routed layer reads split projections)",
    )?;
    refuse_when(
        descriptor
            .layers
            .iter()
            .any(|layer| layer.ffn.activation != Activation::Silu),
        BUILDER,
        "ffn.activation other than Silu (this engine's feed-forward is SwiGLU)",
    )?;
    refuse_when(
        routed
            && descriptor
                .layers
                .iter()
                .any(|layer| layer.ffn.routed_gating != ExpertGatingFunc::Softmax),
        BUILDER,
        "ffn.routed_gating other than Softmax",
    )?;
    refuse_when(
        descriptor.layers.iter().any(|layer| {
            layer.attention.key_source_kind != KeySourceKind::ProjectedK
                || layer.attention.value_source_kind != ValueSourceKind::ProjectedV
        }),
        BUILDER,
        "key_source_kind or value_source_kind other than projected (this engine projects its own k and v)",
    )
}

fn attention_sublayer(
    descriptor: &ModelDescriptor,
    first: &LayerSchedule,
    layer: &LayerSchedule,
    index: u32,
) -> Result<Block, TensorError> {
    let mut keys: Vec<&str> = if descriptor.fused_qkv_reduce {
        alloc::vec!["attn_qkv"]
    } else {
        alloc::vec!["attn_q", "attn_k", "attn_v"]
    };
    if descriptor.qkv_biases {
        keys.extend(["attn_q_bias", "attn_k_bias", "attn_v_bias"]);
    }
    keys.push("attn_output");
    if descriptor.qk_norm {
        keys.extend(["attn_q_norm", "attn_k_norm"]);
    }
    let attention = Attention {
        query_heads: descriptor.query_heads,
        kv_heads: first.attention.kv_heads,
        head_dim: first.attention.head_dim,
        rope_pairing: first.attention.rope_pairing,
        score_scale: first.attention.score_scale,
        window: layer.attention.mask_window,
        qk_norm: descriptor.qk_norm,
        biases: descriptor.qkv_biases,
        fused_qkv: descriptor.fused_qkv_reduce,
        leaves: layer_leaves(descriptor, index, &keys)?,
    };
    Ok(Block::Residual {
        scale: descriptor.residual_scale,
        body: Stack {
            blocks: alloc::vec![
                Block::Norm(layer_norm(descriptor, index, "attn_norm")?),
                Block::Attention(attention),
            ],
        },
    })
}

fn ffn_sublayer(descriptor: &ModelDescriptor, index: u32) -> Result<Block, TensorError> {
    let ffn = if descriptor.expert_count == 0 {
        let gate_up: &[&str] = if descriptor.paired_gate_up_reduce {
            &["ffn_gate_up", "ffn_down"]
        } else {
            &["ffn_gate", "ffn_up", "ffn_down"]
        };
        Block::DenseFfn(DenseFfn {
            feed_forward: descriptor.feed_forward,
            paired_gate_up: descriptor.paired_gate_up_reduce,
            leaves: layer_leaves(descriptor, index, gate_up)?,
        })
    } else {
        Block::RoutedFfn(RoutedFfn {
            feed_forward: descriptor.feed_forward,
            expert_count: descriptor.expert_count,
            expert_used_count: descriptor.expert_used_count,
            leaves: layer_leaves(
                descriptor,
                index,
                &[
                    "ffn_gate_inp",
                    "ffn_gate_exps",
                    "ffn_up_exps",
                    "ffn_down_exps",
                ],
            )?,
        })
    };
    Ok(Block::Residual {
        scale: descriptor.residual_scale,
        body: Stack {
            blocks: alloc::vec![Block::Norm(layer_norm(descriptor, index, "ffn_norm")?), ffn],
        },
    })
}

fn layer_norm(
    descriptor: &ModelDescriptor,
    index: u32,
    key: &str,
) -> Result<Norm, TensorError> {
    Ok(Norm {
        kind: NormKind::Rms,
        eps: None,
        weight: Some(layer_leaf(descriptor, index, key)?),
        group_width: None,
    })
}

fn missing_leaf() -> TensorError {
    TensorError::UnsupportedInBuilder {
        builder: BUILDER,
        feature: "a dialect that omits a leaf this engine declares",
    }
}

fn layer_leaf(descriptor: &ModelDescriptor, index: u32, key: &str) -> Result<String, TensorError> {
    let suffix = descriptor.dialect.leaves.get(key).ok_or_else(missing_leaf)?;
    Ok(alloc::format!(
        "{}{suffix}",
        descriptor.dialect.layer.render(u64::from(index))
    ))
}

fn layer_leaves(
    descriptor: &ModelDescriptor,
    index: u32,
    keys: &[&str],
) -> Result<BTreeMap<String, String>, TensorError> {
    keys.iter()
        .map(|key| Ok(((*key).into(), layer_leaf(descriptor, index, key)?)))
        .collect()
}

fn global_leaf(descriptor: &ModelDescriptor, key: &str) -> Result<String, TensorError> {
    descriptor
        .dialect
        .globals
        .get(key)
        .cloned()
        .ok_or_else(missing_leaf)
}

#[cfg(all(test, feature = "std"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::spec::{FamilyProfile, gqa_descriptor_from_shape};

    const PROFILE: &str = "score_scale_inverse_sqrt_head_dim = true\nvalue_norm = false\nrope_layout = \"adjacent\"\n\n[ffn]\npost_attention_norm = false\ncombination = \"Exclusive\"\noutput_scale = false\nrouted_gating = \"Softmax\"\nrouted_expert_bias = false\nactivation = \"Silu\"\nexclusive_dense_post_norm = false\n";

    fn descriptor(expert_count: u32) -> ModelDescriptor {
        let profile = toml::from_str::<FamilyProfile>(PROFILE).expect("the profile parses");
        let used = if expert_count > 0 { 2 } else { 0 };
        gqa_descriptor_from_shape(
            16,
            8,
            16,
            2,
            1,
            4,
            2,
            expert_count,
            used,
            false,
            false,
            false,
            false,
            &profile,
        )
    }

    fn refusal(descriptor: &ModelDescriptor) -> &'static str {
        match Network::from_bounded(descriptor) {
            Err(TensorError::UnsupportedInBuilder { feature, .. }) => feature,
            other => panic!("expected a named refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_dense_descriptor_projects_two_residual_sublayers_per_block() {
        let network = Network::from_bounded(&descriptor(0)).expect("the dense descriptor projects");

        assert_eq!(network.body.blocks.len(), 4);
        assert!(matches!(network.body.blocks[0], Block::Residual { .. }));
        assert_eq!(network.head.output, "output.weight");
    }

    #[test]
    fn the_projection_names_leaves_from_the_dialect_not_the_position() {
        let network = Network::from_bounded(&descriptor(0)).expect("the dense descriptor projects");
        let Block::Residual { body, .. } = &network.body.blocks[2] else {
            panic!("the third block is the second layer's attention sublayer");
        };

        assert!(matches!(
            &body.blocks[0],
            Block::Norm(norm) if norm.weight.as_deref() == Some("blk.1.attn_norm.weight")
        ));
    }

    #[test]
    fn routed_experts_with_qkv_biases_are_refused_by_name() {
        let mut routed = descriptor(4);
        routed.qkv_biases = true;

        assert!(refusal(&routed).contains("qkv_biases"));
    }

    #[test]
    fn routed_experts_with_fused_qkv_are_refused_by_name() {
        let mut routed = descriptor(4);
        routed.fused_qkv_reduce = true;

        assert!(refusal(&routed).contains("fused_qkv_reduce"));
    }

    #[test]
    fn a_non_silu_activation_is_refused_by_name() {
        let mut gelu = descriptor(0);
        gelu.layers[0].ffn.activation = Activation::GeluTanh;

        assert!(refusal(&gelu).contains("ffn.activation"));
    }

    #[test]
    fn non_softmax_routed_gating_is_refused_by_name() {
        let mut sigmoid = descriptor(4);
        sigmoid.layers[0].ffn.routed_gating = ExpertGatingFunc::Sigmoid;

        assert!(refusal(&sigmoid).contains("ffn.routed_gating"));
    }

    #[test]
    fn a_shared_key_source_is_refused_by_name() {
        let mut shared = descriptor(0);
        for layer in &mut shared.layers {
            layer.attention.key_source_kind = KeySourceKind::SharedFromLayer(0);
        }

        assert!(refusal(&shared).contains("key_source_kind"));
    }
}
