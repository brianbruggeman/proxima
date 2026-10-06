//! The `sliding_pattern` header reader ([`proxima_tensor::spec::ScheduleSource::SlidingPattern`]):
//! the parsed header goes to [`proxima_tensor::spec::gemma4_descriptor_from_gguf`],
//! which builds the whole [`proxima_tensor::spec::ModelDescriptor`] (per-layer
//! [`proxima_tensor::spec::LayerAttentionConfig`]/
//! [`proxima_tensor::spec::LayerFfnConfig`] schedule included). The lowering
//! and the weight bind are the generic ones in [`crate::lowering`]: there is no
//! bespoke gemma4 forward-graph builder, no gemma4 schedule and no gemma4
//! tensor-name table in this crate.
//! Teaching pointer: read `proxima_tensor::spec::attention_forward`'s own doc
//! on `lfm2_forward_program_with_experts` before touching this file -- every
//! knob the descriptor sets is documented there, not here.


use proxima_gguf::pipe::ParsedGguf;
use proxima_tensor::spec::{ModelDescriptor, gemma4_descriptor_from_gguf};

use crate::bind::{ModelHparams, SlidingRope, find_tensor, metadata_str};
use crate::error::InteropError;
use crate::lowering::KvLayout;
use crate::profiles::family_profile;

use super::hparams::from_metadata;

/// The checkpoint's descriptor: [`gemma4_descriptor_from_gguf`] over the family
/// profile `general.architecture` names, so the values GGUF does not carry come
/// from `crate::profiles` and never from this module.
pub fn descriptor_from_gguf(
    parsed: &ParsedGguf,
    sliding_kv_ring: bool,
) -> Result<ModelDescriptor, InteropError> {
    let profile = family_profile(metadata_str(parsed, "general.architecture")?)?;
    Ok(gemma4_descriptor_from_gguf(parsed, sliding_kv_ring, &profile)?)
}

/// `PROXIMA_HEAD_REPEATS=1|2|3` (unset or unparsable reads as `1`): the
/// head-cost measurement knob, layered into [`proxima_tensor::spec::ModelDescriptor::head_repeats`]
/// here so the op-graph builders stay pure functions of the descriptor.
#[cfg(feature = "instrument")]
fn head_repeats_from_env() -> u32 {
    std::env::var("PROXIMA_HEAD_REPEATS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1)
}

/// This header's descriptor and hyperparameters, the two values
/// [`crate::lowering`] lowers and binds from. Every layer is
/// [`proxima_tensor::spec::LayerKind::Attention`] (this family has no
/// `ShortConv` layers), routed through the padded-mask cached engine: the
/// two-range one, not single-range, because the first step processes the whole
/// prompt as one `cached_len=0` call (`lfm2_single_range_cached.rs`'s own
/// module doc).
pub(crate) fn header(
    parsed: &ParsedGguf,
    layout: KvLayout,
) -> Result<(ModelDescriptor, ModelHparams), InteropError> {
    let hparams = from_metadata(parsed)?;
    let descriptor = descriptor_from_gguf(parsed, layout == KvLayout::SlidingRing)?;
    #[cfg(feature = "instrument")]
    let descriptor = ModelDescriptor {
        head_repeats: head_repeats_from_env(),
        ..descriptor
    };
    let architecture = ModelHparams {
        vocab: hparams.vocab,
        embedding: hparams.embedding,
        feed_forward: hparams.feed_forward,
        query_heads: hparams.head_count,
        kv_heads: *hparams.kv_heads_by_layer.last().unwrap_or(&0),
        kv_heads_by_layer: hparams.kv_heads_by_layer.clone(),
        head_dim: hparams.key_length,
        block_count: hparams.block_count,
        expert_count: hparams.expert_count,
        expert_used_count: hparams.expert_used_count,
        rope_freq_base: hparams.rope_freq_base,
        rms_epsilon: hparams.rms_epsilon,
        tied_embeddings: find_tensor(parsed, "output.weight").is_err(),
        family: metadata_str(parsed, "general.architecture")?.into(),
        sliding_rope: Some(SlidingRope {
            freq_base: hparams.rope_freq_base_swa,
            dimension_count: hparams.rope_dimension_count_swa,
        }),
    };
    Ok((descriptor, architecture))
}

/// Regression coverage for the bug this crate shipped once: the binder and the
/// forward program each independently gated `attn_k.weight`/
/// `attn_k_norm.weight`/`attn_v.weight` per layer, and nothing forced the
/// two gates to agree -- `gemma4_descriptor_from_gguf` used to gate
/// `ValueSourceKind::ProjectedV` (and therefore the forward program's own
/// `attn_v.weight` [`proxima_tensor::op::Op::Input`] leaf) on `is_sliding`
/// alone, the MoE convention, while E2B/E4B's own-KV FULL layers (`blk.4`,
/// `blk.9`, `blk.14` on the real `gemma4:e2b-it-qat` checkpoint) carry a
/// real `attn_v.weight` regardless of sliding vs full. This test builds the
/// ACTUAL forward program `crate::lowering::bind_checkpoint` builds (not a hand-simulated
/// stand-in) for a synthetic E2B-shaped [`Gemma4Hparams`] whose
/// `sliding_window_pattern`/`shared_kv_layers`/`block_count` are the real
/// checkpoint's own measured values (a real header dump against
/// `~/.ollama/models/blobs/sha256-3646b4c...` on 2026-09-20), then asserts
/// the forward program's own declared `Input` leaf names for these three
/// per-layer weights equal exactly the layers that own those tensors in the
/// real checkpoint (derived below from the header pattern, never from the
/// lowering under test).
#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod declared_leaves_match_bound_leaves_tests {
    use super::*;
    use proxima_tensor::spec::{ForwardProgram, build_forward};
    use crate::sliding_pattern::Gemma4Hparams;
    use arrayvec::ArrayVec;
    use proxima_tensor::spec::{CacheStrategy, KeySourceKind};
    use proxima_gguf::types::GgmlType;
    use proxima_gguf::value::{MetadataArray, MetadataValue};
    use proxima_gguf::{GgufModel, TensorPayload, parse_complete, write_complete};
    use proxima_tensor::op::Op;

    /// The real `gemma4:e2b-it-qat` checkpoint's own measured
    /// `sliding_window_pattern`/`shared_kv_layers`/`block_count` (header
    /// dump, 2026-09-20) -- every other key is a plausible dense-E2B
    /// value uninvolved in the attn_k/attn_k_norm/attn_v declare-vs-bind
    /// gate this test exercises. Written by the real GGUF encoder and
    /// parsed back by the real decoder; `token_embd.weight` is a 1-row
    /// table so vocab resolves.
    fn e2b_shaped(shared_kv_layers: u32) -> (ParsedGguf, Gemma4Hparams) {
        let mut feed_forward = alloc::vec![6144u32; 15];
        feed_forward.extend(alloc::vec![12288u32; 20]);
        let u32_array = |values: Vec<u32>| MetadataValue::Array(MetadataArray::U32(values));
        let metadata = alloc::vec![
            ("general.architecture", MetadataValue::String("gemma4".into())),
            ("gemma4.embedding_length", MetadataValue::U32(1536)),
            ("gemma4.block_count", MetadataValue::U32(35)),
            ("gemma4.attention.head_count", MetadataValue::U32(8)),
            ("gemma4.attention.head_count_kv", u32_array(alloc::vec![1; 35])),
            (
                "gemma4.attention.sliding_window_pattern",
                MetadataValue::Array(MetadataArray::Bool((0..35u32).map(|index| (index + 1) % 5 != 0).collect())),
            ),
            ("gemma4.attention.shared_kv_layers", MetadataValue::U32(shared_kv_layers)),
            ("gemma4.attention.key_length", MetadataValue::U32(512)),
            ("gemma4.attention.value_length", MetadataValue::U32(512)),
            ("gemma4.attention.key_length_swa", MetadataValue::U32(256)),
            ("gemma4.attention.value_length_swa", MetadataValue::U32(256)),
            ("gemma4.attention.sliding_window", MetadataValue::U32(512)),
            ("gemma4.attention.layer_norm_rms_epsilon", MetadataValue::F32(1e-6)),
            ("gemma4.feed_forward_length", u32_array(feed_forward)),
            ("gemma4.rope.freq_base", MetadataValue::F32(1_000_000.0)),
            ("gemma4.rope.freq_base_swa", MetadataValue::F32(10_000.0)),
            ("gemma4.rope.dimension_count", MetadataValue::U32(512)),
            ("gemma4.rope.dimension_count_swa", MetadataValue::U32(256)),
            ("gemma4.final_logit_softcapping", MetadataValue::F32(30.0)),
            ("gemma4.embedding_length_per_layer_input", MetadataValue::U32(256)),
        ];
        let table = [0u8; 1536 * 4];
        let model = GgufModel {
            version: 3,
            metadata: metadata
                .into_iter()
                .map(|(key, value)| (key.to_string(), value))
                .collect(),
            tensors: alloc::vec![TensorPayload {
                name: "token_embd.weight".to_string(),
                dims: ArrayVec::from_iter([1536u64, 1]),
                ggml_type: GgmlType::F32,
                data: &table,
            }],
        };
        let bytes = write_complete(&model).expect("the e2b-shaped model encodes");
        let parsed = parse_complete(&bytes).expect("bytes the encoder just wrote parse");
        let architecture = from_metadata(&parsed).expect("gemma4 hparams parse from the e2b-shaped header");
        (parsed, architecture)
    }

    /// The leaf names of `suffix` a checkpoint with this header stores a
    /// tensor for: own-KV layers carry K and its norm, and V where the
    /// sliding pattern or a shared-KV header says so.
    fn stored_leaf_names(
        architecture: &Gemma4Hparams,
        suffix: &str,
    ) -> alloc::collections::BTreeSet<String> {
        let first_shared_idx = architecture
            .block_count
            .saturating_sub(architecture.shared_kv_layers);
        architecture
            .sliding_window_pattern
            .iter()
            .enumerate()
            .filter_map(|(layer_index, &is_sliding)| {
                let layer = layer_index as u32;
                let is_shared_kv = layer >= first_shared_idx;
                let bound = match suffix {
                    "attn_k.weight" | "attn_k_norm.weight" => !is_shared_kv,
                    "attn_v.weight" => {
                        (is_sliding || architecture.shared_kv_layers > 0) && !is_shared_kv
                    }
                    other => unreachable!("unexpected suffix {other}"),
                };
                bound.then(|| format!("blk.{layer}.{suffix}"))
            })
            .collect()
    }

    /// The name set the ACTUAL forward program `crate::lowering::bind_checkpoint` lowers --
    /// `gemma4_descriptor_from_gguf`'s own output, built through the
    /// cacheless engine so the check is independent of which
    /// `CacheStrategy` bind picks -- declares as an `Input` leaf for `suffix`.
    fn declared_leaf_names(
        parsed: &ParsedGguf,
        suffix: &str,
    ) -> alloc::collections::BTreeSet<String> {
        let mut descriptor = descriptor_from_gguf(parsed, false)
            .expect("the e2b-shaped header carries every key the descriptor reads");
        descriptor.cache_strategy = CacheStrategy::Cacheless;
        let ForwardProgram { program, .. } = build_forward(&descriptor).expect("gemma4 e2b-shaped forward program lowers");

        program
            .iter()
            .filter_map(|op| match op {
                Op::Input {
                    name: Some(name), ..
                } => Some(name.clone()),
                _ => None,
            })
            .filter(|name| name.ends_with(suffix) && name.starts_with("blk."))
            .collect()
    }

    #[test]
    fn e2b_declared_attn_v_leaves_equal_bound_attn_v_leaves() {
        let (parsed, architecture) = e2b_shaped(20);
        let declared = declared_leaf_names(&parsed, "attn_v.weight");
        let bound = stored_leaf_names(&architecture, "attn_v.weight");
        assert_eq!(
            declared, bound,
            "forward program declares attn_v.weight leaves the binder does not bind (or vice versa)"
        );
        // The three real full own-KV layers this bug silently dropped --
        // pins the invariant to the actual header fact, not just set
        // equality (an empty-vs-empty pair would also satisfy `assert_eq`
        // above).
        for full_own_kv_layer in [4, 9, 14] {
            let name = format!("blk.{full_own_kv_layer}.attn_v.weight");
            assert!(
                declared.contains(&name),
                "expected {name} to be declared (full own-KV E2B layer has a real attn_v.weight)"
            );
        }
    }

    #[test]
    fn e2b_declared_attn_k_and_attn_k_norm_leaves_equal_bound_leaves() {
        let (parsed, architecture) = e2b_shaped(20);
        for suffix in ["attn_k.weight", "attn_k_norm.weight"] {
            let declared = declared_leaf_names(&parsed, suffix);
            let bound = stored_leaf_names(&architecture, suffix);
            assert_eq!(declared, bound, "{suffix} declare/bind set mismatch");
        }
    }

    /// The MoE path (`shared_kv_layers == 0`) must keep the exact prior
    /// gate: only sliding layers declare/bind `attn_v.weight` -- proves the
    /// E2B fix above did not widen MoE's own set.
    #[test]
    fn moe_declared_attn_v_leaves_stay_gated_on_is_sliding_only() {
        let (parsed, architecture) = e2b_shaped(0);
        let declared = declared_leaf_names(&parsed, "attn_v.weight");
        let bound = stored_leaf_names(&architecture, "attn_v.weight");
        assert_eq!(declared, bound);
        for full_layer in [4, 9, 14] {
            let name = format!("blk.{full_layer}.attn_v.weight");
            assert!(
                !declared.contains(&name),
                "MoE full layer {name} must stay SharedWithKey (no attn_v.weight)"
            );
        }
    }

    /// Hand-derived from ollama's `mlxrunner/model/gemma4/gemma4.go`
    /// `TextConfig` KV-sharing-map build (`gemma4.go:590-611`: a shared layer
    /// reuses "the last non-shared layer of the same type") for
    /// `block_count=35`, `shared_kv_layers=20` (`first_shared_idx=15`): the 16
    /// sliding shared layers all reuse own-KV layer 13, the 4 full shared
    /// layers (19, 24, 29, 34) reuse layer 14 -- exactly 20 pairs, matching
    /// the real header's `attention.shared_kv_layers=20`. Read off the
    /// production descriptor, not a private helper.
    #[test]
    fn gemma4_e2b_shared_kv_reuse_map_matches_hand_derived_table() {
        let (parsed, _) = e2b_shaped(20);
        let descriptor = descriptor_from_gguf(&parsed, false)
            .expect("the e2b-shaped header carries every key the descriptor reads");
        let shared_sources: [(usize, u32); 20] = [
            (15, 13), (16, 13), (17, 13), (18, 13), (19, 14),
            (20, 13), (21, 13), (22, 13), (23, 13), (24, 14),
            (25, 13), (26, 13), (27, 13), (28, 13), (29, 14),
            (30, 13), (31, 13), (32, 13), (33, 13), (34, 14),
        ];

        for (layer, source) in shared_sources {
            let attention = descriptor.layers[layer].attention.clone();
            assert_eq!(
                attention.key_source_kind,
                KeySourceKind::SharedFromLayer(source),
                "layer {layer} expected to read the K of layer {source}"
            );
        }
        assert!(
            descriptor.layers[..15]
                .iter()
                .all(|entry| entry.attention.key_source_kind == KeySourceKind::ProjectedK),
            "own-KV layers 0..15 project their own K"
        );
    }

    /// The header's pattern must itself imply exactly 7 full-attention
    /// layers over 35 blocks (`35 / 5`), independent of the reuse-map
    /// assertion above, so a broken pattern cannot pass it by accident.
    #[test]
    fn gemma4_e2b_header_pattern_has_seven_full_attention_layers_over_thirty_five_blocks() {
        let (parsed, _) = e2b_shaped(20);
        let descriptor = descriptor_from_gguf(&parsed, false)
            .expect("the e2b-shaped header carries every key the descriptor reads");
        let full_layers = descriptor
            .layers
            .iter()
            .filter(|entry| entry.attention.mask_window.is_none())
            .count();
        assert_eq!(full_layers, 7);
    }
}
