//! the 8B-A1B short-conv checkpoint's hybrid checkpoint as data: [`ShortConvHparams`] reads this
//! architecture's own metadata shape -- a per-layer `head_count_kv` array whose
//! zero entries mark short-convolution layers ([`crate::bind::architecture_from_metadata`]
//! reads the array; the dense descriptor builder
//! ([`crate::dense::descriptor_from_gguf`]) turns the zero entries into
//! [`proxima_tensor::spec::LayerKind::ShortConv`] layers from the tensor
//! directory). [`short_conv_descriptor`] is that whole pre-lowering program as one
//! config, and [`proxima_tensor::spec::build_forward`] over it is the entire
//! lowering: `LoadedModel::load` runs this family through the same decode loop
//! every other checkpoint takes (a cacheless program, one re-prefill per
//! token), and [`short_conv_forward_values`] evaluates the same lowering once for the
//! cross-oracle diff tools.

use alloc::collections::BTreeSet;
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::value::{MetadataArray, MetadataValue};
use proxima_tensor::cpu::{QuantizedBlock, evaluate_quantized_named_with_scratch};
use proxima_tensor::op::{NodeId, Op};
use proxima_tensor::spec::{ForwardProgram, LayerSchedule, ModelDescriptor, build_forward};

use crate::bind::{
    BoundWeights, architecture_from_metadata, metadata_f32_optional, metadata_str, metadata_u32,
    metadata_u32_optional, metadata_u32_optional_or, vocab_from_token_embedding,
};
use crate::dense::descriptor_from_gguf;
use crate::bind_leaves::bind_program_leaves;
use crate::error::InteropError;
use crate::generate::build_position_inputs;
use crate::profiles::binding_profile;
use crate::rope_scaling::{RopeScaling, f32_from_u32};

/// Every hparam [`build_forward`] needs, derived from a
/// real `lfm2moe`-architecture checkpoint's own metadata --
/// [`crate::bind::ModelHparams`]'s hybrid-checkpoint counterpart, not
/// a variant of it: that struct's single `kv_heads: u32` and lack of a
/// `layer_kinds`/`leading_dense_block_count`/`l_cache` field mean it cannot
/// describe this architecture at all, not even partially.
#[derive(Debug, Clone)]
pub struct ShortConvHparams {
    pub vocab: u32,
    pub embedding: u32,
    pub feed_forward: u32,
    pub expert_feed_forward: u32,
    pub query_heads: u32,
    /// The real per-attention-layer kv head count -- every convolution
    /// layer's own `0` placeholder entry in the real checkpoint's
    /// `head_count_kv` array is skipped, not averaged in; see
    /// `metadata_u32_array_nonzero_uniform`'s own doc.
    pub kv_heads: u32,
    pub head_dim: u32,
    pub block_count: u32,
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub leading_dense_block_count: u32,
    pub l_cache: u32,
    pub rope_freq_base: f32,
    /// The checkpoint's own `{architecture}.rope.scaling.*`
    /// ([`RopeScaling::from_gguf`]); [`RopeScaling::None`] when it declares
    /// none. This program takes no `ServingConfig`, so there is no per-call
    /// override here.
    pub rope_scaling: RopeScaling,
    pub rms_epsilon: f32,
    /// One entry per block, read off the same [`ModelDescriptor`] every
    /// dense-family load lowers: the kind each block runs, its attention shape
    /// and its FFN knobs.
    pub layers: Vec<LayerSchedule>,
}

/// [`transformers/models/lfm2_moe/modeling_lfm2_moe.py`]'s own RMSNorm
/// epsilon default, used only when the real checkpoint's own
/// `{architecture}.attention.layer_norm_rms_epsilon` key is absent -- on
/// the one real checkpoint this module has been run against, the key IS
/// present (`0.00001`), so this fallback is unexercised there.
const SHORT_CONV_RMS_EPSILON_DEFAULT: f32 = 1e-5;

/// Derives [`ShortConvHparams`] from `parsed`'s own metadata --
/// [`crate::bind::architecture_from_metadata`]'s hybrid-checkpoint
/// counterpart. Reads `general.architecture` itself (`lfm2moe` on the real
/// checkpoint, not `short-conv`) rather than assuming it, the same "read the
/// wire, don't hard-code the string" shape every other key here already
/// uses.
///
/// # Errors
///
/// [`InteropError::MissingMetadataKey`] if a required key is absent;
/// [`InteropError::HeterogeneousNonzeroMetadataArray`] if
/// `{architecture}.attention.head_count_kv`'s nonzero (attention-layer)
/// entries disagree with each other; whatever
/// [`proxima_tensor::spec::LayerKind::from_tensor_names`] fails with if a
/// layer's tensor directory carries neither an attention nor a
/// short-convolution marker.
pub fn short_conv_architecture_from_metadata(
    parsed: &ParsedGguf,
) -> Result<ShortConvHparams, InteropError> {
    let architecture = metadata_str(parsed, "general.architecture")?;
    let embedding = metadata_u32(parsed, &format!("{architecture}.embedding_length"))?;
    let feed_forward = metadata_u32(parsed, &format!("{architecture}.feed_forward_length"))?;
    let expert_feed_forward = metadata_u32_optional(
        parsed,
        &format!("{architecture}.expert_feed_forward_length"),
    );
    let query_heads = metadata_u32(parsed, &format!("{architecture}.attention.head_count"))?;
    let kv_heads = metadata_u32_array_nonzero_uniform(
        parsed,
        &format!("{architecture}.attention.head_count_kv"),
    )?;
    let block_count = metadata_u32(parsed, &format!("{architecture}.block_count"))?;
    let head_dim = metadata_u32_optional_or(
        parsed,
        &format!("{architecture}.rope.dimension_count"),
        embedding / query_heads.max(1),
    );
    let vocab = vocab_from_token_embedding(parsed, embedding)?;
    let expert_count = metadata_u32_optional(parsed, &format!("{architecture}.expert_count"));
    let expert_used_count =
        metadata_u32_optional(parsed, &format!("{architecture}.expert_used_count"));
    let leading_dense_block_count =
        metadata_u32_optional(parsed, &format!("{architecture}.leading_dense_block_count"));
    let l_cache = metadata_u32(parsed, &format!("{architecture}.shortconv.l_cache"))?;
    let rope_freq_base = metadata_f32_optional(
        parsed,
        &format!("{architecture}.rope.freq_base"),
        proxima_tensor::sized::ROPE_FREQ_BASE_DEFAULT,
    );
    let rms_epsilon = metadata_f32_optional(
        parsed,
        &format!("{architecture}.attention.layer_norm_rms_epsilon"),
        SHORT_CONV_RMS_EPSILON_DEFAULT,
    );

    let layers = short_conv_descriptor(parsed)?.layers;

    Ok(ShortConvHparams {
        vocab,
        embedding,
        feed_forward,
        expert_feed_forward,
        query_heads,
        kv_heads,
        head_dim,
        block_count,
        expert_count,
        expert_used_count,
        leading_dense_block_count,
        l_cache,
        rope_freq_base,
        rope_scaling: RopeScaling::from_gguf(parsed)?,
        rms_epsilon,
        layers,
    })
}

/// [`crate::bind::metadata_u32_or_uniform_array`]'s hybrid-architecture
/// counterpart: every ZERO entry is a short-convolution layer's own
/// placeholder and is skipped rather than folded into the uniformity
/// check, but every NONZERO entry (a real attention layer's kv head count)
/// must still agree -- a checkpoint with two different real kv head
/// counts across its attention layers is not a shape this module (or
/// [`build_forward`], which takes one `kv_heads: u32`)
/// can represent, so that case is refused, not averaged or first-picked.
fn metadata_u32_array_nonzero_uniform(parsed: &ParsedGguf, key: &str) -> Result<u32, InteropError> {
    match parsed.metadata_value(key) {
        Some(MetadataValue::U32(value)) => Ok(*value),
        Some(MetadataValue::I32(value)) => {
            u32::try_from(*value).map_err(|_| InteropError::MissingMetadataKey { key: key.into() })
        }
        Some(MetadataValue::Array(MetadataArray::U32(values))) => {
            nonzero_uniform_u32_array(key, values.iter().copied())
        }
        Some(MetadataValue::Array(MetadataArray::I32(values))) => nonzero_uniform_u32_array(
            key,
            values
                .iter()
                .map(|value| u32::try_from(*value).unwrap_or(u32::MAX)),
        ),
        _ => Err(InteropError::MissingMetadataKey { key: key.into() }),
    }
}

fn nonzero_uniform_u32_array(
    key: &str,
    values: impl Iterator<Item = u32>,
) -> Result<u32, InteropError> {
    let mut distinct: BTreeSet<u32> = BTreeSet::new();
    for value in values {
        if value != 0 {
            distinct.insert(value);
        }
    }
    match distinct.len() {
        0 => Err(InteropError::MissingMetadataKey { key: key.into() }),
        1 => Ok(distinct.into_iter().next().unwrap_or(0)),
        distinct_values => Err(InteropError::HeterogeneousNonzeroMetadataArray {
            key: key.into(),
            distinct_values,
        }),
    }
}

/// One call's worth of position-dependent `Input`s
/// [`build_forward`] needs beyond the model weights --
/// [`crate::generate::PositionInputs`]'s prefill-only, always-starts-at-0
/// counterpart: this program has no key/value cache to offset positions
/// against, so every call re-derives RoPE angles for absolute positions
/// `0..ids.len()`, never a `start_position` offset.
struct ShortConvPositionInputs {
    ids_f32: Vec<f32>,
    epsilon: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

fn build_short_conv_position_inputs(
    ids: &[u32],
    head_dim: u32,
    rope_freq_base: f32,
    rms_epsilon: f32,
    rope_scaling: RopeScaling,
) -> ShortConvPositionInputs {
    let shared = build_position_inputs(
        ids,
        0,
        head_dim,
        rope_freq_base,
        rms_epsilon,
        None,
        rope_scaling,
    );
    ShortConvPositionInputs {
        ids_f32: ids.iter().map(|&id| f32_from_u32(id)).collect(),
        epsilon: shared.epsilon,
        cos: shared.cos,
        sin: shared.sin,
    }
}

/// The checkpoint's whole pre-lowering program as one config, with every new
/// row's logits kept (this module's runners read the last row of a full
/// re-prefill): the dense-family descriptor builder over the header, which
/// reads the zero-KV layers as short-convolution layers and lowers the stack
/// without a cache.
///
/// # Errors
///
/// Whatever [`architecture_from_metadata`] and
/// [`crate::dense::descriptor_from_gguf`] can fail with.
pub fn short_conv_descriptor(parsed: &ParsedGguf) -> Result<ModelDescriptor, InteropError> {
    let architecture = architecture_from_metadata(parsed)?;
    Ok(ModelDescriptor {
        last_row_only: false,
        ..descriptor_from_gguf(parsed, &architecture)?
    })
}

/// [`short_conv_descriptor`] lowered by [`build_forward`], with the weights its
/// `Input` leaves name bound from `parsed` ([`bind_program_leaves`]) --
/// lowering first, so the program decides what binds.
fn lower_and_bind<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
) -> Result<(Vec<Op>, NodeId, BoundWeights<'file>), InteropError> {
    let ForwardProgram { program, logits, .. } = build_forward(&short_conv_descriptor(parsed)?)?;
    let weights = bind_program_leaves(
        parsed,
        file_bytes,
        &program,
        &binding_profile(metadata_str(parsed, "general.architecture")?)?,
        &[],
    )?;
    Ok((program, logits, weights))
}

/// One forward pass over `ids` (no generation loop), mirroring [`crate::generate::LoadedModel::forward_logits`]/
/// [`crate::generate::LoadedModel::forward_node_values`]'s dense-checkpoint
/// shape for this hybrid checkpoint's own GGUF bind + program build. Returns
/// the LAST position's full-vocab logits, plus one raw evaluated buffer per
/// entry in `extra_node_ids`, in the same order.
///
/// `extra_node_ids` are typically derived by building
/// [`build_forward`] at two adjacent `block_count`s and
/// diffing their `Op` sequences (`smollm2_layer_oracle_diff.rs`'s own
/// technique) -- the id-is-index invariant that relies on holds here
/// identically, since a shallower build only ever appends nodes to the
/// deeper one's own prefix.
///
/// # Errors
///
/// Whatever [`lower_and_bind`]/[`build_forward`]/
/// evaluating the program can fail with, plus
/// [`InteropError::MissingEvaluatedNode`] if the evaluator's output is
/// missing the logits root or one of `extra_node_ids` -- an
/// interpreter/program-construction invariant violation, never a caller
/// mistake.
pub fn short_conv_forward_values(
    parsed: &ParsedGguf,
    file_bytes: &[u8],
    architecture: &ShortConvHparams,
    ids: &[u32],
    extra_node_ids: &[NodeId],
) -> Result<(Vec<f32>, Vec<Vec<f32>>), InteropError> {
    let (program, logits_root, weights) = lower_and_bind(parsed, file_bytes)?;

    let inputs = build_short_conv_position_inputs(
        ids,
        architecture.head_dim,
        architecture.rope_freq_base,
        architecture.rms_epsilon,
        architecture.rope_scaling,
    );
    let vocab_size = architecture.vocab as usize;

    let mut named_blocks: Vec<(&str, QuantizedBlock)> =
        Vec::with_capacity(weights.owned.len() + weights.packed.len() + 3);
    named_blocks.push(("ids", QuantizedBlock::Float32(inputs.ids_f32.as_slice())));
    for (name, data) in &weights.owned {
        named_blocks.push((name.as_str(), QuantizedBlock::Float32(data.as_slice())));
    }
    for (name, block) in &weights.packed {
        named_blocks.push((name.as_str(), *block));
    }
    named_blocks.push(("eps", QuantizedBlock::Float32(inputs.epsilon.as_slice())));
    named_blocks.push(("rope_cos", QuantizedBlock::Float32(inputs.cos.as_slice())));
    named_blocks.push(("rope_sin", QuantizedBlock::Float32(inputs.sin.as_slice())));

    let symbols = [ids.len() as u64];
    let mut free_buffers: Vec<Vec<f32>> = Vec::new();
    let mut validated_weight_nodes: Option<BTreeSet<NodeId>> = None;

    let mut outputs: Vec<NodeId> = vec![logits_root];
    outputs.extend_from_slice(extra_node_ids);

    let evaluated = evaluate_quantized_named_with_scratch(
        &program,
        &symbols,
        &named_blocks,
        &outputs,
        &mut free_buffers,
        &mut validated_weight_nodes,
    )?;

    let (logits, _shape) = evaluated
        .get(logits_root)
        .ok_or(InteropError::MissingEvaluatedNode { node: logits_root })?;
    let last_position = logits[(ids.len() - 1) * vocab_size..ids.len() * vocab_size].to_vec();

    let mut extras = Vec::with_capacity(extra_node_ids.len());
    for &node in extra_node_ids {
        let (values, _shape) = evaluated
            .get(node)
            .ok_or(InteropError::MissingEvaluatedNode { node })?;
        extras.push(values.to_vec());
    }

    Ok((last_position, extras))
}

#[cfg(all(test, feature = "std"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::error::InteropError;

    /// [`nonzero_uniform_u32_array`]'s own contract, proved directly: zero
    /// entries (convolution layers) are skipped, and every real, nonzero
    /// entry (attention layers) must agree -- the real checkpoint's own
    /// `[0, 0, 8, 0, 0, 0, 8, ...]` shape.
    #[test]
    fn nonzero_uniform_array_skips_zeros_and_requires_nonzero_agreement() {
        let uniform = nonzero_uniform_u32_array("key", [0, 0, 8, 0, 0, 0, 8].into_iter());
        assert_eq!(uniform.expect("uniform nonzero entries agree"), 8);

        let disagreeing = nonzero_uniform_u32_array("key", [0, 8, 0, 16].into_iter());
        assert!(matches!(
            disagreeing,
            Err(InteropError::HeterogeneousNonzeroMetadataArray {
                distinct_values: 2,
                ..
            })
        ));

        let all_zero = nonzero_uniform_u32_array("key", [0, 0, 0].into_iter());
        assert!(matches!(
            all_zero,
            Err(InteropError::MissingMetadataKey { .. })
        ));
    }
}
