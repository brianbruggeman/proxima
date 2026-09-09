//! The [`crate::architecture::Architecture`] registered as
//! [`crate::architecture::ArchitectureRegistry::with_builtin`]'s fallback --
//! every checkpoint `crate::generate::LoadedModel::load_inner`'s `else` arm
//! has ever accepted (`llama`, `mistral`, `qwen3`, `mixtral`, and any other
//! `general.architecture` this crate has no dedicated hybrid binder for).
//! Composes exactly what that `else` arm always called:
//! [`crate::bind::architecture_from_metadata`],
//! [`crate::bind::checkpoint_has_qk_norm`], and
//! [`proxima_tensor::spec::mistral_cached_forward_program_with_experts`] --
//! `expert_count`/`expert_used_count` off the checkpoint's own metadata is
//! what already selects a dense vs. mixture-of-experts program inside that
//! one builder, so this one [`Architecture`] impl covers both without a
//! separate MoE arm (`crate::architecture::Architecture`'s own doc on
//! `DenseArch` being the un-registered-by-name fallback, not a name match).
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
    Qwen35LayerRoots, mistral_cached_forward_program_with_experts_and_layer_taps,
};

use crate::architecture::{Architecture, BoundProgram};
use crate::bind::{architecture_from_metadata, bind_all_weights, checkpoint_has_qk_norm};
use crate::error::InteropError;

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
        // `&[]`: this entry point takes no `ServingConfig`, so there is no
        // `weight_precision` rule set to thread here yet --
        // `crate::bind::bind_all_weights`'s own doc names this as the
        // wiring a future slice does, unchanged from `load_inner`'s prior
        // inline call.
        let weights = bind_all_weights(parsed, file_bytes, &architecture, false, false, &[])?;
        let qk_norm = checkpoint_has_qk_norm(parsed);
        // `last_row_only: true` -- the decode loop
        // (`crate::generate::LoadedModel`'s own decode step) only ever
        // samples the LAST row's logits, greedy or not; see
        // `mistral_cached_forward_program_with_experts_and_layer_taps`'s
        // own doc on that flag and `proxima-tensor/docs/discipline.md`
        // ROW 418/421 for the measured cost of computing every row instead.
        let (program, roots, cache_roots, _layer_residuals, moe_sites) =
            mistral_cached_forward_program_with_experts_and_layer_taps(
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
                false,
                false,
                true,
            )?;
        Ok(BoundProgram {
            weights,
            architecture,
            program,
            logits_root: roots.logits,
            hidden_root: Some(roots.hidden),
            layer_roots: cache_roots.into_iter().map(Qwen35LayerRoots::Attention).collect(),
            moe_sites,
            single_position_step: false,
        })
    }
}
