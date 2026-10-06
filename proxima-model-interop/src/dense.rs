//! The [`crate::architecture::Architecture`] registered as
//! [`crate::architecture::ArchitectureRegistry::with_builtin`]'s fallback --
//! every checkpoint `crate::generate::LoadedModel::load_inner`'s `else` arm
//! has ever accepted (`llama`, `mistral`, `qwen2`, `qwen3`, `mixtral`, and
//! any other `general.architecture` this crate has no dedicated hybrid
//! binder for). Composes exactly what that `else` arm always called:
//! [`crate::bind::architecture_from_metadata`],
//! [`crate::bind::checkpoint_has_qk_norm`], and (routed through
//! [`proxima_tensor::spec::build_forward`]'s `CacheStrategy::Cached`, `CacheMask::Bounded`
//! arm, via [`proxima_tensor::spec::mistral_descriptor_from_shape`] built
//! straight off this checkpoint's own parsed `architecture`, rather than a
//! direct call)
//! [`proxima_tensor::spec::mistral_cached_forward_program_with_experts_and_layer_taps`] --
//! `expert_count`/`expert_used_count` off the checkpoint's own metadata is
//! what already selects a dense vs. mixture-of-experts program inside that
//! one builder, so this one [`Architecture`] impl covers both without a
//! separate MoE arm (`crate::architecture::Architecture`'s own doc on
//! `DenseArch` being the un-registered-by-name fallback, not a name match).
//!
//! RoPE pairing is profile data (`rope_layout`), never inferred from QK-norm
//! tensors. This binder compares no family name:
//! [`crate::bind::ModelArchitecture::family`] keys
//! [`crate::profiles::family_profile`], and the profile's `rope_layout` rides
//! into the one generic [`build_forward`] call every family takes. A family with
//! no profile is an error, never a default.
//!
//! Does not carry `load_with_paired_gate_up_reduce`/`load_with_fused_qkv_reduce`'s
//! diagnostic reduce flags -- those are per-call A/B knobs
//! (`crate::generate::LoadedModel::load_with_paired_gate_up_reduce`'s own
//! doc), not part of "which architecture is this checkpoint", so
//! `load_inner` keeps its own narrow inline path for those two
//! constructors rather than widening [`Architecture::bind`]'s signature
//! for a diagnostic every other architecture would have to ignore.

use proxima_gguf::pipe::ParsedGguf;
use proxima_tensor::spec::{
    AttentionScoreScale, CacheStrategy, EmbeddingScale, ForwardProgram, LayerAttentionConfig, LayerKind,
    LayerSchedule, ModelDescriptor, build_forward, mistral_descriptor_from_shape,
};

use crate::architecture::{Architecture, BoundProgram};
use crate::bind::{
    ModelArchitecture, architecture_from_metadata, checkpoint_has_qk_norm,
    checkpoint_qkv_biases, metadata_f32_optional, metadata_u32_optional,
};
use crate::bind_leaves::bind_program_leaves;
use crate::error::InteropError;
use crate::profiles::{binding_profile, family_profile};
use crate::task::{ModelTask, classify_task};

/// The registered fallback architecture -- see [`Architecture::name`]'s own
/// doc for why "dense" is a label, not a `general.architecture` value this
/// type expects to match by name.
pub struct DenseArch;

/// The one registered [`DenseArch`] value.
pub static DENSE: DenseArch = DenseArch;

impl Architecture for DenseArch {
    fn name(&self) -> &'static str {
        "dense"
    }

    fn bind<'file>(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
    ) -> Result<BoundProgram<'file>, InteropError> {
        let architecture = architecture_from_metadata(parsed)?;
        let descriptor = descriptor_from_gguf(parsed, &architecture)?;
        bind_descriptor(parsed, file_bytes, architecture, &descriptor)
    }

    /// The same weights and cache leaves as [`Self::bind`] with every new
    /// position's logits row kept, when the family profile arms it
    /// ([`ModelDescriptor::verify`]); every family's default is off until its
    /// verify step is measured to pay for the drafts it checks.
    fn speculative_verify_program<'file>(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
    ) -> Result<Option<BoundProgram<'file>>, InteropError> {
        let architecture = architecture_from_metadata(parsed)?;
        let descriptor = descriptor_from_gguf(parsed, &architecture)?;
        descriptor
            .verify()
            .map(|verify| bind_descriptor(parsed, file_bytes, architecture, &verify))
            .transpose()
    }
}

fn bind_descriptor<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    architecture: ModelArchitecture,
    descriptor: &ModelDescriptor,
) -> Result<BoundProgram<'file>, InteropError> {
    let ForwardProgram { program, logits, layer_roots, moe_sites, layer_residuals, hidden, .. } =
        build_forward(descriptor)?;
    // `&[]`: this entry point takes no `ServingConfig`, so there is no
    // `weight_precision` rule set to thread here yet.
    let weights = bind_program_leaves(
        parsed,
        file_bytes,
        &program,
        &binding_profile(&architecture.family)?,
        &[],
    )?;
    Ok(BoundProgram {
        weights,
        architecture,
        program,
        logits_root: logits,
        hidden_root: hidden,
        residual_roots: layer_residuals,
        layer_roots,
        qwen35moe_layer_diagnostics: Vec::new(),
        router_roots: Vec::new(),
        moe_sites,
        duplicate_head_roots: Vec::new(),
        single_position_step: false,
    })
}

/// The dense checkpoint's whole pre-lowering program as one config:
/// [`mistral_descriptor_from_shape`] over the family profile
/// `general.architecture` names, the header's own scales layered on top, and
/// the logits row count the checkpoint's task needs. [`build_forward`] over
/// the result is the entire lowering; serialize the descriptor and a restored
/// copy lowers to the same program.
///
/// # Errors
///
/// The family has no profile, the header declares partial rotary, or the
/// per-layer KV head counts differ (this program has one cache shape).
pub fn descriptor_from_gguf(
    parsed: &ParsedGguf,
    architecture: &ModelArchitecture,
) -> Result<ModelDescriptor, InteropError> {
    let profile = family_profile(&architecture.family)?;
    require_full_rotary(parsed, architecture)?;
    // This builder has one KV cache shape for every attention layer. Preserve a
    // checkpoint's per-layer configuration in `ModelArchitecture`, but
    // do not silently select a representative value for this uniform
    // program.
    let kv_heads = architecture.uniform_attention_kv_heads()?;
    let descriptor = mistral_descriptor_from_shape(
        architecture.vocab,
        architecture.embedding,
        architecture.feed_forward,
        architecture.query_heads,
        kv_heads,
        architecture.head_dim,
        architecture.block_count,
        architecture.expert_count,
        architecture.expert_used_count,
        checkpoint_has_qk_norm(parsed),
        checkpoint_qkv_biases(parsed, architecture)?,
        false,
        false,
        &profile,
    );
    // Encoder-style tasks consume the full hidden sequence for pooling or a
    // task head; decoder generation samples only the final row, so the vocab
    // projection stays narrow there (`proxima-tensor/docs/discipline.md`
    // ROW 418/421 measured the cost of computing every row instead).
    let last_row_only = !matches!(classify_task(parsed).task, ModelTask::Embedding);
    let descriptor = with_conv_layers(descriptor, parsed, architecture)?;
    let descriptor = with_header_window(with_header_scales(descriptor, parsed, &architecture.family), parsed, &architecture.family);
    Ok(ModelDescriptor { last_row_only, ..descriptor })
}

/// A checkpoint whose header declares zero KV heads on some layers is a hybrid
/// of attention and short-convolution layers: the tensor directory says which
/// kind each zero-KV layer is, the dense and routed FFN widths and the leading
/// dense block count come off the header, and the program is lowered without a
/// cache because a convolution layer carries state a KV cache cannot hold
/// ([`CacheStrategy::Cacheless`]). A checkpoint with an attention layer at
/// every index is returned unchanged.
fn with_conv_layers(
    descriptor: ModelDescriptor,
    parsed: &ParsedGguf,
    architecture: &ModelArchitecture,
) -> Result<ModelDescriptor, InteropError> {
    let kinds = layer_kinds(parsed, architecture)?;
    if kinds.iter().all(|kind| *kind == LayerKind::Attention) {
        return Ok(descriptor);
    }
    let family = &architecture.family;
    let layers = descriptor
        .layers
        .iter()
        .zip(kinds)
        .map(|(layer, kind)| LayerSchedule { kind, ..layer.clone() })
        .collect();
    Ok(ModelDescriptor {
        layers,
        feed_forward: metadata_u32_optional(parsed, &format!("{family}.feed_forward_length")),
        leading_dense_block_count: metadata_u32_optional(parsed, &format!("{family}.leading_dense_block_count")),
        l_cache: metadata_u32_optional(parsed, &format!("{family}.shortconv.l_cache")),
        cache_strategy: CacheStrategy::Cacheless,
        ..descriptor
    })
}

fn layer_kinds(parsed: &ParsedGguf, architecture: &ModelArchitecture) -> Result<Vec<LayerKind>, InteropError> {
    let names: Vec<&str> = parsed.tensors.iter().map(|tensor| tensor.name.as_str()).collect();
    architecture
        .kv_heads_by_layer
        .iter()
        .enumerate()
        .map(|(layer, &kv_heads)| {
            if kv_heads != 0 {
                return Ok(LayerKind::Attention);
            }
            match LayerKind::from_tensor_names(names.iter().copied(), layer as u32)? {
                LayerKind::ShortConv => Ok(LayerKind::ShortConv),
                other => Err(InteropError::UnsupportedServingConfig(format!(
                    "layer {layer} declares zero kv heads but its tensors say {other:?}"
                ))),
            }
        })
        .collect()
}

/// `<family>.attention.sliding_window`, when the header carries it, is the
/// window of every layer; absent or zero leaves every layer on the full
/// causal mask, the same reading llama.cpp's `n_swa` gets.
fn with_header_window(descriptor: ModelDescriptor, parsed: &ParsedGguf, family: &str) -> ModelDescriptor {
    let window = metadata_u32_optional(parsed, &format!("{family}.attention.sliding_window"));
    if window == 0 {
        return descriptor;
    }
    let layers = descriptor
        .layers
        .iter()
        .map(|layer| LayerSchedule {
            attention: LayerAttentionConfig {
                mask_window: Some(window),
                ..layer.attention.clone()
            },
            ..layer.clone()
        })
        .collect();
    ModelDescriptor { layers, ..descriptor }
}

fn header_scale(parsed: &ParsedGguf, family: &str, key: &str) -> Option<f32> {
    let value = metadata_f32_optional(parsed, &format!("{family}.{key}"), 0.0);
    (value != 0.0).then_some(value)
}

fn with_header_scales(descriptor: ModelDescriptor, parsed: &ParsedGguf, family: &str) -> ModelDescriptor {
    let layers = match header_scale(parsed, family, "attention.scale") {
        Some(scale) => descriptor
            .layers
            .iter()
            .map(|layer| LayerSchedule {
                attention: LayerAttentionConfig {
                    score_scale: AttentionScoreScale::Factor(scale),
                    ..layer.attention.clone()
                },
                ..layer.clone()
            })
            .collect(),
        None => descriptor.layers.clone(),
    };
    ModelDescriptor {
        embedding_scale: header_scale(parsed, family, "embedding_scale")
            .map(EmbeddingScale::Factor)
            .or(descriptor.embedding_scale),
        logit_scale: header_scale(parsed, family, "logit_scale"),
        residual_scale: header_scale(parsed, family, "residual_scale"),
        layers,
        ..descriptor
    }
}

/// The single-range builder rotates `head_dim` channels. A header whose
/// `<arch>.rope.dimension_count` says otherwise is partial rotary, which this
/// program cannot lower; refuse it rather than rotate the wrong width.
fn require_full_rotary(parsed: &ParsedGguf, architecture: &ModelArchitecture) -> Result<(), InteropError> {
    let key = format!("{}.rope.dimension_count", architecture.family);
    match metadata_u32_optional(parsed, &key) {
        0 => Ok(()),
        rope_dimension_count if rope_dimension_count == architecture.head_dim => Ok(()),
        rope_dimension_count => Err(InteropError::PartialRotaryUnsupported {
            family: architecture.family.clone(),
            rope_dimension_count,
            head_dim: architecture.head_dim,
        }),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use arrayvec::ArrayVec;
    use proxima_gguf::types::GgmlType;
    use proxima_gguf::value::MetadataArray;
    use proxima_gguf::{GgufModel, MetadataValue, TensorPayload, parse_complete, write_complete};

    fn header_bytes(family: &str, floats: &[(&str, f32)]) -> Vec<u8> {
        let mut metadata = vec![("general.architecture".to_string(), MetadataValue::String(family.into()))];
        metadata.extend(
            floats
                .iter()
                .map(|(key, value)| (format!("{family}.{key}"), MetadataValue::F32(*value))),
        );
        let model = GgufModel {
            version: 3,
            metadata,
            tensors: Vec::new(),
        };
        write_complete(&model).expect("a header with no tensors encodes")
    }

    fn granite_input() -> ModelDescriptor {
        mistral_descriptor_from_shape(
            49155,
            1024,
            512,
            16,
            8,
            64,
            24,
            32,
            8,
            false,
            false,
            false,
            false,
            &family_profile("granitemoe").expect("profile embedded"),
        )
    }

    fn llama_input() -> ModelDescriptor {
        mistral_descriptor_from_shape(
            32000,
            4096,
            14336,
            32,
            8,
            128,
            32,
            0,
            0,
            false,
            false,
            false,
            false,
            &family_profile("llama").expect("profile embedded"),
        )
    }

    fn llama_architecture() -> ModelArchitecture {
        ModelArchitecture {
            vocab: 32000,
            embedding: 4096,
            feed_forward: 14336,
            query_heads: 32,
            kv_heads: 8,
            kv_heads_by_layer: vec![8; 32],
            head_dim: 128,
            block_count: 32,
            expert_count: 0,
            expert_used_count: 0,
            rope_freq_base: 10_000.0,
            rms_epsilon: 1e-5,
            tied_embeddings: false,
            family: "llama".to_string(),
            sliding_rope: None,
        }
    }

    #[test]
    fn an_architecture_reshaped_by_its_own_descriptor_is_unchanged() {
        let architecture = llama_architecture();

        assert_eq!(architecture.reshaped_by(&llama_input()), architecture);
    }

    #[test]
    fn a_config_that_changes_the_layer_count_and_head_width_reshapes_the_architecture() {
        let mut descriptor = llama_input();
        descriptor.block_count = 2;
        descriptor.layers.truncate(2);
        for layer in &mut descriptor.layers {
            layer.attention.head_dim = 64;
        }

        let reshaped = llama_architecture().reshaped_by(&descriptor);

        assert_eq!((reshaped.block_count, reshaped.head_dim, reshaped.kv_heads), (2, 64, 8));
    }

    #[test]
    fn a_non_uniform_schedule_keeps_the_header_head_width() {
        let mut descriptor = llama_input();
        descriptor.layers[0].attention.head_dim = 64;

        let reshaped = llama_architecture().reshaped_by(&descriptor);

        assert_eq!(reshaped.head_dim, 128);
    }

    #[test]
    fn header_scales_reach_the_descriptor_of_a_granite_shaped_header() {
        let bytes = header_bytes(
            "granitemoe",
            &[
                ("embedding_scale", 12.0),
                ("residual_scale", 0.22),
                ("logit_scale", 6.0),
                ("attention.scale", 0.015625),
            ],
        );
        let parsed = parse_complete(&bytes).expect("bytes the encoder just wrote parse");
        let input = granite_input();

        let result = with_header_scales(input.clone(), &parsed, "granitemoe");

        assert_eq!(result.embedding_scale, Some(EmbeddingScale::Factor(12.0)));
        assert_eq!(result.logit_scale, Some(6.0));
        assert_eq!(result.residual_scale, Some(0.22));
        assert_eq!(result.layers.len(), 24);
        assert!(
            result
                .layers
                .iter()
                .all(|layer| layer.attention.score_scale == AttentionScoreScale::Factor(0.015625))
        );
        let restored = ModelDescriptor {
            embedding_scale: input.embedding_scale,
            logit_scale: None,
            residual_scale: None,
            layers: input.layers.clone(),
            ..result.clone()
        };
        assert_eq!(restored, input);
    }

    #[test]
    fn a_header_without_scale_keys_leaves_the_descriptor_unchanged() {
        let bytes = header_bytes("llama", &[]);
        let parsed = parse_complete(&bytes).expect("bytes the encoder just wrote parse");
        let input = llama_input();

        assert_eq!(with_header_scales(input.clone(), &parsed, "llama"), input);
    }

    fn header_with_window(family: &str, window: u32) -> Vec<u8> {
        let model = GgufModel {
            version: 3,
            metadata: vec![
                ("general.architecture".to_string(), MetadataValue::String(family.into())),
                (format!("{family}.attention.sliding_window"), MetadataValue::U32(window)),
            ],
            tensors: Vec::new(),
        };
        write_complete(&model).expect("a header with no tensors encodes")
    }

    #[test]
    fn a_header_sliding_window_reaches_every_layer_of_the_descriptor() {
        let bytes = header_with_window("llama", 4096);
        let parsed = parse_complete(&bytes).expect("bytes the encoder just wrote parse");
        let input = llama_input();

        let result = with_header_window(input.clone(), &parsed, "llama");

        assert_eq!(result.layers.len(), 32);
        assert!(result.layers.iter().all(|layer| layer.attention.mask_window == Some(4096)));
        let restored = ModelDescriptor {
            layers: input.layers.clone(),
            ..result
        };
        assert_eq!(restored, input, "the window is the only field the header changes");
    }

    #[test]
    fn a_header_without_a_sliding_window_leaves_every_layer_unwindowed() {
        let bytes = header_bytes("llama", &[]);
        let parsed = parse_complete(&bytes).expect("bytes the encoder just wrote parse");
        let input = llama_input();

        assert_eq!(with_header_window(input.clone(), &parsed, "llama"), input);
        let zero = header_with_window("llama", 0);
        let parsed_zero = parse_complete(&zero).expect("bytes the encoder just wrote parse");
        assert_eq!(with_header_window(input.clone(), &parsed_zero, "llama"), input);
    }

    const HYBRID_EMBEDDING: u64 = 8;
    const HYBRID_VOCAB: u64 = 16;

    fn hybrid_header(kv_heads: &[u32], layer_tensors: &[&str]) -> Vec<u8> {
        let family = "lfm2moe";
        let metadata = vec![
            ("general.architecture".to_string(), MetadataValue::String(family.into())),
            (format!("{family}.embedding_length"), MetadataValue::U32(HYBRID_EMBEDDING as u32)),
            (format!("{family}.feed_forward_length"), MetadataValue::U32(32)),
            (format!("{family}.expert_feed_forward_length"), MetadataValue::U32(12)),
            (format!("{family}.attention.head_count"), MetadataValue::U32(2)),
            (
                format!("{family}.attention.head_count_kv"),
                MetadataValue::Array(MetadataArray::I32(kv_heads.iter().map(|&heads| heads as i32).collect())),
            ),
            (format!("{family}.block_count"), MetadataValue::U32(kv_heads.len() as u32)),
            (format!("{family}.expert_count"), MetadataValue::U32(4)),
            (format!("{family}.expert_used_count"), MetadataValue::U32(2)),
            (format!("{family}.leading_dense_block_count"), MetadataValue::U32(1)),
            (format!("{family}.shortconv.l_cache"), MetadataValue::U32(3)),
        ];
        let table = vec![0u8; (HYBRID_EMBEDDING * HYBRID_VOCAB * 4) as usize];
        let marker = [0u8; 4];
        let mut tensors = vec![TensorPayload {
            name: "token_embd.weight".to_string(),
            dims: ArrayVec::from_iter([HYBRID_EMBEDDING, HYBRID_VOCAB]),
            ggml_type: GgmlType::F32,
            data: &table,
        }];
        tensors.extend(layer_tensors.iter().map(|name| TensorPayload {
            name: (*name).to_string(),
            dims: ArrayVec::from_iter([1u64]),
            ggml_type: GgmlType::F32,
            data: &marker,
        }));
        write_complete(&GgufModel { version: 3, metadata, tensors }).expect("a hybrid header with marker tensors encodes")
    }

    fn hybrid_descriptor(kv_heads: &[u32], layer_tensors: &[&str]) -> Result<ModelDescriptor, InteropError> {
        let bytes = hybrid_header(kv_heads, layer_tensors);
        let parsed = parse_complete(&bytes).expect("bytes the encoder just wrote parse");
        let architecture = architecture_from_metadata(&parsed)?;
        descriptor_from_gguf(&parsed, &architecture)
    }

    #[test]
    fn zero_kv_layers_with_a_conv_tensor_become_short_conv_layers_of_a_cacheless_program() {
        let descriptor = hybrid_descriptor(&[0, 2, 0], &["blk.0.shortconv.conv.weight", "blk.1.attn_q.weight", "blk.2.shortconv.conv.weight"])
            .expect("a hybrid header describes itself");

        assert_eq!(
            descriptor.layers.iter().map(|layer| layer.kind).collect::<Vec<_>>(),
            vec![LayerKind::ShortConv, LayerKind::Attention, LayerKind::ShortConv]
        );
        assert_eq!(descriptor.cache_strategy, CacheStrategy::Cacheless);
        assert_eq!((descriptor.l_cache, descriptor.leading_dense_block_count), (3, 1));
        assert_eq!((descriptor.feed_forward, descriptor.expert_feed_forward), (32, 12));
        assert_eq!(descriptor.layers[1].attention.kv_heads, 2);
    }

    #[test]
    fn a_zero_kv_layer_whose_tensors_say_attention_is_refused() {
        let outcome = hybrid_descriptor(&[0, 2], &["blk.0.attn_q.weight", "blk.1.attn_q.weight"]);

        assert!(matches!(outcome, Err(InteropError::UnsupportedServingConfig(message)) if message.contains("layer 0")));
    }

    #[test]
    fn a_zero_kv_layer_with_no_mixer_tensor_is_refused() {
        let outcome = hybrid_descriptor(&[0, 2], &["blk.1.attn_q.weight"]);

        assert!(matches!(outcome, Err(InteropError::Tensor(proxima_tensor::TensorError::UndeterminedLayerKind { layer: 0 }))));
    }

    #[test]
    fn attention_layers_that_disagree_on_kv_heads_are_still_refused() {
        let outcome = hybrid_descriptor(&[0, 2, 4], &["blk.0.shortconv.conv.weight", "blk.1.attn_q.weight", "blk.2.attn_q.weight"]);

        assert!(matches!(outcome, Err(InteropError::HeterogeneousMetadataArray { distinct_values: 2, .. })));
    }

    #[test]
    fn a_zero_scale_means_unset_like_llama_cpp() {
        let bytes = header_bytes("llama", &[("residual_scale", 0.0), ("logit_scale", 0.0)]);
        let parsed = parse_complete(&bytes).expect("bytes the encoder just wrote parse");
        let input = llama_input();

        assert_eq!(with_header_scales(input.clone(), &parsed, "llama"), input);
    }
}
