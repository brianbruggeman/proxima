//! [`crate::architecture::Architecture`] registration for the `qwen35moe`
//! checkpoint family. The forward program lives in `program.rs`; its weights
//! bind from that program's own `Input` leaves
//! ([`crate::bind_leaves::bind_program_leaves`]), so no tensor-name inventory
//! exists beside it to drift from it.

use proxima_gguf::pipe::ParsedGguf;

use crate::architecture::{Architecture as ArchitectureTrait, BoundProgram};
use crate::bind::metadata_str;
use crate::bind_leaves::bind_program_leaves;
use crate::error::InteropError;
use crate::profiles::binding_profile;

use super::hparams::from_metadata;

/// The binding profile key and the registry name: the architecture that lowers this program
/// is what names its leaves, so a delegating foreign architecture binds the same way.
const FAMILY: &str = "qwen35moe";

/// Marker registered for `general.architecture = "qwen35moe"`.
pub struct Qwen35MoeArch;

/// The builtin `qwen35moe` registration value.
pub static QWEN35MOE: Qwen35MoeArch = Qwen35MoeArch;

impl ArchitectureTrait for Qwen35MoeArch {
    fn name(&self) -> &'static str {
        FAMILY
    }

    fn kv_cache_shape(&self) -> crate::architecture::KvCacheShape {
        crate::architecture::KvCacheShape::Monolithic
    }

    fn kv_layers(&self, parsed: &ParsedGguf) -> Result<Vec<(u32, u32, Option<u32>)>, InteropError> {
        super::hparams::kv_layers_from_metadata(parsed)
    }

    fn ffn_routing(&self) -> crate::architecture::FfnRouting {
        crate::architecture::FfnRouting::Routed
    }

    fn diagnostic_reduce_flags_apply(&self) -> bool {
        false
    }

    fn bind<'file>(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
    ) -> Result<BoundProgram<'file>, InteropError> {
        let qwen_architecture = from_metadata(parsed)?;
        let (program, roots, layer_roots, moe_sites, diagnostics) =
            super::program::qwen35moe_forward_program(&qwen_architecture)?;
        let weights = bind_program_leaves(
            parsed,
            file_bytes,
            &program,
            &binding_profile(FAMILY)?,
            &[],
        )?;
        let architecture = crate::bind::ModelArchitecture {
            vocab: qwen_architecture.vocab,
            embedding: qwen_architecture.embedding,
            feed_forward: qwen_architecture.expert_feed_forward,
            query_heads: qwen_architecture.query_heads,
            kv_heads: qwen_architecture
                .kv_heads_by_layer
                .iter()
                .copied()
                .max()
                .unwrap_or(0),
            kv_heads_by_layer: qwen_architecture.kv_heads_by_layer.clone(),
            head_dim: qwen_architecture.rope_dims,
            block_count: qwen_architecture.block_count,
            expert_count: qwen_architecture.expert_count,
            expert_used_count: qwen_architecture.expert_used_count,
            rope_freq_base: qwen_architecture.rope_freq_base,
            rms_epsilon: qwen_architecture.rms_epsilon,
            tied_embeddings: false,
            family: metadata_str(parsed, "general.architecture")?.into(),
            sliding_rope: None,
        };
        Ok(BoundProgram {
            weights,
            architecture,
            program,
            logits_root: roots.logits,
            hidden_root: Some(roots.hidden),
            residual_roots: Vec::new(),
            layer_roots,
            qwen35moe_layer_diagnostics: diagnostics.clone(),
            router_roots: diagnostics
                .iter()
                .map(|diagnostic| diagnostic.router_logits)
                .collect(),
            moe_sites,
            duplicate_head_roots: Vec::new(),
            single_position_step: true,
        })
    }
}
