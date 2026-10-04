//! The [`crate::architecture::Architecture`] registered as
//! [`crate::architecture::ArchitectureRegistry::with_builtin`]'s fallback --
//! every checkpoint `crate::generate::LoadedModel::load_inner`'s `else` arm
//! has ever accepted (`llama`, `mistral`, `qwen2`, `qwen3`, `mixtral`, and
//! any other `general.architecture` this crate has no dedicated hybrid
//! binder for). Composes exactly what that `else` arm always called:
//! [`crate::bind::architecture_from_metadata`],
//! [`crate::bind::checkpoint_has_qk_norm`], and (routed through
//! [`proxima_tensor::spec::build_forward`]'s `CacheStrategy::SingleRange`
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
use proxima_tensor::spec::{Qwen35LayerRoots, build_forward, mistral_descriptor_from_shape};

use crate::architecture::{Architecture, BoundProgram};
use crate::bind::{
    ModelArchitecture, architecture_from_metadata, bind_all_weights, checkpoint_has_qk_norm,
    checkpoint_qkv_biases, metadata_u32_optional,
};
use crate::error::InteropError;
use crate::profiles::family_profile;
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
        let profile = family_profile(&architecture.family)?;
        require_full_rotary(parsed, &architecture)?;
        // This builder has one KV cache shape for every layer. Preserve a
        // checkpoint's per-layer configuration in `ModelArchitecture`, but
        // do not silently select a representative value for this uniform
        // program.
        architecture.uniform_kv_heads()?;
        // `&[]`: this entry point takes no `ServingConfig`, so there is no
        // `weight_precision` rule set to thread here yet --
        // `crate::bind::bind_all_weights`'s own doc names this as the
        // wiring a future slice does, unchanged from `load_inner`'s prior
        // inline call.
        let weights = bind_all_weights(parsed, file_bytes, &architecture, false, false, &[])?;
        let qk_norm = checkpoint_has_qk_norm(parsed);
        // Encoder-style tasks consume the full hidden sequence for pooling or
        // a task head. Decoder generation only needs the final row, so keep
        // the expensive vocab projection narrow there. The task classifier
        // runs before this bind and prevents a non-generation checkpoint from
        // being mistaken for a decoder by the caller.
        let last_row_only = !matches!(classify_task(parsed).task, ModelTask::Embedding);
        // `last_row_only` -- the decode loop
        // (`crate::generate::LoadedModel`'s own decode step) only ever
        // samples the LAST row's logits, greedy or not; see
        // `mistral_cached_forward_program_with_experts_and_layer_taps`'s
        // own doc on that flag and `proxima-tensor/docs/discipline.md`
        // ROW 418/421 for the measured cost of computing every row instead.
        // Every family's values the header does not carry (RoPE pairing
        // included) come from its profile; `build_forward_matches_direct_builder_call_at_real_mistral_dims`
        // and `..._at_real_qwen2_dims` (`proxima-tensor/src/spec/tests.rs`)
        // prove the descriptor route byte-identical to the direct builder.
        let descriptor = mistral_descriptor_from_shape(
            architecture.vocab,
            architecture.embedding,
            architecture.feed_forward,
            architecture.query_heads,
            architecture.kv_heads,
            architecture.head_dim,
            architecture.block_count,
            architecture.expert_count,
            architecture.expert_used_count,
            qk_norm,
            checkpoint_qkv_biases(parsed, &architecture)?,
            false,
            false,
            &profile,
        );
        let (program, logits_root, cache_roots, moe_sites, layer_residuals, hidden_root, _head_repeats) =
            build_forward(&descriptor, last_row_only)?;
        Ok(BoundProgram {
            weights,
            architecture,
            program,
            logits_root,
            hidden_root,
            residual_roots: layer_residuals,
            layer_roots: cache_roots
                .into_iter()
                .map(Qwen35LayerRoots::Attention)
                .collect(),
            qwen35moe_layer_diagnostics: Vec::new(),
            router_roots: Vec::new(),
            moe_sites,
            duplicate_head_roots: Vec::new(),
            single_position_step: false,
        })
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
