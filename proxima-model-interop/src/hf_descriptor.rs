//! A recurrent family's Hugging Face `config.json` as a [`ModelDescriptor`]:
//! the family profile ([`crate::profiles::family_profile`]) supplies what the
//! file does not carry (sandwich norms, no Q/K norm, the recurrence's loop
//! norm), the config supplies the shape and the checkpoint's own pass count
//! (`total_ut_steps`), and the caller's [`RecurrentConfig`] picks the depth and
//! guidance the graph is lowered with.
//!
//! The descriptor keeps the physical schedule (`block_count` is the checkpoint's
//! layer count) and gains a [`proxima_tensor::spec::RecurrenceConfig`]; the
//! returned [`ModelHparams`] is physical too, and
//! [`crate::generate::LoadedModel::load_from_safetensors_with_descriptor`]
//! expands it to the logical depth once the graph is built. Hugging Face keeps
//! Q/K rows in `rotate_half` order, which is split-half RoPE pairing, so unlike
//! the GGUF conversion no row is reordered on load.

use proxima_tensor::spec::{LoopCdConfig, ModelDescriptor, RopePairing, gqa_descriptor_from_shape};

use crate::bind::ModelHparams;
use crate::error::InteropError;
use crate::hf_config::{HfConfig, architecture_from_hf_config};
use crate::profiles::family_profile;
use crate::recurrent_config::RecurrentConfig;

/// Converts a parsed Ouro `config.json` and the requested policy into the
/// physical architecture and the descriptor the shared graph builder lowers.
///
/// # Errors
///
/// [`InteropError::InvalidRecurrentConfig`] for a config that is not a dense
/// full-attention Ouro with the no-early-exit, loop-norm-on settings, or a
/// policy no depth of this checkpoint could make valid;
/// [`InteropError::MissingMetadataKey`] without `total_ut_steps`; whatever the
/// family profile lookup can fail with.
pub fn ouro_descriptor_from_hf_config(
    config: &HfConfig,
    recurrent: RecurrentConfig,
) -> Result<(ModelHparams, ModelDescriptor), InteropError> {
    validate_hf_ouro(config)?;
    let architecture = architecture_from_hf_config(config)?;
    validate_dense_attention(&architecture)?;
    let profile = family_profile(&architecture.family)?;
    if profile.recurrence.is_none() {
        return Err(InteropError::UnsupportedRecurrentConfig {
            family: architecture.family,
        });
    }
    let checkpoint_depth = config
        .total_ut_steps
        .ok_or(InteropError::MissingMetadataKey {
            key: "total_ut_steps".into(),
        })?;
    let resolved = recurrent.resolve(checkpoint_depth)?;
    let dense = gqa_descriptor_from_shape(
        architecture.vocab,
        architecture.embedding,
        architecture.feed_forward,
        architecture.query_heads,
        architecture.kv_heads,
        architecture.head_dim,
        architecture.block_count,
        0,
        0,
        profile.qk_norm,
        false,
        false,
        false,
        &profile,
    );
    let mut descriptor = profile.recurring(dense, resolved.iterations.unwrap_or(checkpoint_depth));
    descriptor.loopcd = LoopCdConfig {
        mode: resolved.mode,
        strength: resolved.strength,
        reference_iteration: resolved.reference_iteration,
    };
    for layer in &mut descriptor.layers {
        layer.attention.rope_pairing = RopePairing::SplitHalf {
            pairs: architecture.head_dim / 2,
        };
    }
    Ok((architecture, descriptor))
}

fn validate_hf_ouro(config: &HfConfig) -> Result<(), InteropError> {
    let supported = config.model_type == "ouro"
        && config.hidden_act.as_deref() == Some("silu")
        && config.early_exit_threshold.unwrap_or(1.0) == 1.0
        && !config.skip_loop_final_norm.unwrap_or(false)
        && config.layer_types.len() == config.num_hidden_layers as usize
        && config
            .layer_types
            .iter()
            .all(|kind| kind == "full_attention");
    if supported {
        return Ok(());
    }
    Err(InteropError::InvalidRecurrentConfig {
        reason: "HF config is not a supported dense Ouro attention schedule",
    })
}

fn validate_dense_attention(architecture: &ModelHparams) -> Result<(), InteropError> {
    let well_formed = architecture.expert_count == 0
        && architecture.block_count > 0
        && architecture.embedding > 0
        && architecture.feed_forward > 0
        && architecture.query_heads > 0
        && architecture.kv_heads > 0
        && architecture.head_dim > 0
        && architecture.head_dim.is_multiple_of(2)
        && architecture.embedding
            == architecture
                .query_heads
                .saturating_mul(architecture.head_dim);
    if well_formed {
        return Ok(());
    }
    Err(InteropError::InvalidRecurrentConfig {
        reason: "Ouro requires dense attention with positive, even head dimensions",
    })
}
