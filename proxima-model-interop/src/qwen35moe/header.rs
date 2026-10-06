//! The `recurrent_routed_interval` header reader
//! ([`proxima_tensor::spec::ScheduleSource::RecurrentRoutedInterval`]): the
//! header's hyperparameters ([`from_metadata`]) and the descriptor
//! `program.rs` maps them to. The lowering and the weight bind are the generic
//! ones in [`crate::lowering`]; the leaf names a program declares are what its
//! weights bind from, so no tensor-name inventory exists beside it to drift
//! from it.

use proxima_gguf::pipe::ParsedGguf;
use proxima_tensor::spec::ModelDescriptor;

use crate::bind::{ModelHparams, metadata_str};
use crate::error::InteropError;

use super::hparams::from_metadata;
use super::program::descriptor_from_architecture;

/// This header's descriptor and hyperparameters, the two values
/// [`crate::lowering`] lowers and binds from.
pub(crate) fn header(parsed: &ParsedGguf) -> Result<(ModelDescriptor, ModelHparams), InteropError> {
    let hparams = from_metadata(parsed)?;
    let descriptor = descriptor_from_architecture(&hparams, None)?;
    let architecture = ModelHparams {
        vocab: hparams.vocab,
        embedding: hparams.embedding,
        feed_forward: hparams.expert_feed_forward,
        query_heads: hparams.query_heads,
        kv_heads: hparams.kv_heads_by_layer.iter().copied().max().unwrap_or(0),
        kv_heads_by_layer: hparams.kv_heads_by_layer.clone(),
        head_dim: hparams.rope_dims,
        block_count: hparams.block_count,
        expert_count: hparams.expert_count,
        expert_used_count: hparams.expert_used_count,
        rope_freq_base: hparams.rope_freq_base,
        rms_epsilon: hparams.rms_epsilon,
        tied_embeddings: false,
        family: metadata_str(parsed, "general.architecture")?.into(),
        sliding_rope: None,
    };
    Ok((descriptor, architecture))
}
