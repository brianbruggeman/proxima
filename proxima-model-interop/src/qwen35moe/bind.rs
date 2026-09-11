//! Weight binding for the `qwen35moe` checkpoint family.
//!
//! This module is deliberately separate from the forward-program builder.  A
//! checkpoint can be parsed and its weights bound without pretending that a
//! dense program is valid for its mixed GDN/attention and routed-expert
//! layers.  The program seam currently returns the typed hybrid-MoE error;
//! keeping binding here makes the later program port consume the same name
//! set instead of growing a second, drifting inventory.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;

use crate::architecture::{Architecture as ArchitectureTrait, BoundProgram};
use crate::bind::{
    BoundWeights, bind_dense, bind_matmul_weight, bind_matmul_weight_transposed_f32,
    bind_moe_expert_weights, find_tensor,
};
use crate::error::InteropError;

use super::hparams::{Architecture, LayerKind, from_metadata};

/// Enumerates the checkpoint tensors in the order used by the bind pass.
///
/// The list is derived from the hparams layer kind, rather than assuming that
/// every layer has attention weights.  GDN layers have no KV heads and carry
/// the fused state-space projection; attention layers carry Q/K/V and the
/// per-head QK normalization weights.  The packed expert stacks remain whole
/// tensors so their byte ranges stay borrowable from the checkpoint mapping.
#[must_use]
pub fn qwen35moe_tensor_names(architecture: &Architecture) -> Vec<String> {
    let mut names = vec![String::from("token_embd.weight")];

    for (layer_number, layer_kind) in architecture.layer_kinds.iter().enumerate() {
        let layer = layer_number as u32;
        names.push(format!("blk.{layer}.attn_norm.weight"));
        names.push(format!("blk.{layer}.post_attention_norm.weight"));

        match layer_kind {
            LayerKind::Attention => {
                for suffix in [
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                    "attn_output.weight",
                    "attn_q_norm.weight",
                    "attn_k_norm.weight",
                ] {
                    names.push(format!("blk.{layer}.{suffix}"));
                }
            }
            LayerKind::Gdn => {
                for suffix in [
                    "attn_qkv.weight",
                    "attn_gate.weight",
                    "ssm_conv1d.weight",
                    "ssm_dt",
                    "ssm_a",
                    "ssm_beta.weight",
                    "ssm_alpha.weight",
                    "ssm_norm.weight",
                    "ssm_out.weight",
                ] {
                    names.push(format!("blk.{layer}.{suffix}"));
                }
            }
        }

        for suffix in [
            "ffn_gate_inp.weight",
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
            "ffn_gate_inp_shexp.weight",
            "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
        ] {
            names.push(format!("blk.{layer}.{suffix}"));
        }
    }

    names.push(String::from("output_norm.weight"));
    names
}

fn block_layer(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

/// Returns `(out_dim, in_dim)` for the four dense-attention matrices.
///
/// These dimensions are derived from the architecture's per-layer KV-head
/// array.  In particular, using the global query-head count for K/V would
/// bind the GQA tensors with the wrong declared shape.
fn dense_attention_dims(architecture: &Architecture, name: &str) -> Option<(usize, usize)> {
    let layer = block_layer(name)?;
    let embedding = architecture.embedding as usize;
    let head_dim = architecture.attn_head_dim as usize;
    let query_heads = architecture.query_heads as usize;
    let kv_heads = *architecture.kv_heads_by_layer.get(layer)? as usize;

    if name.ends_with("attn_q.weight") {
        Some((query_heads * head_dim * 2, embedding))
    } else if name.ends_with("attn_k.weight") || name.ends_with("attn_v.weight") {
        Some((kv_heads * head_dim, embedding))
    } else if name.ends_with("attn_output.weight") {
        Some((embedding, query_heads * head_dim))
    } else {
        None
    }
}

/// Binds every tensor named by [`qwen35moe_tensor_names`].
///
/// Expert stacks are kept as packed, borrowed blocks.  The router matrix is
/// the one exceptional axis order: GGUF stores it as `[expert, embedding]`,
/// while the eventual contraction consumes `[embedding, expert]`, so it uses
/// the existing transpose-aware matmul binder.
pub fn bind_qwen35moe_weights<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    architecture: &Architecture,
) -> Result<BoundWeights<'file>, InteropError> {
    let mut state = BoundWeights::new(&[]);

    for name in qwen35moe_tensor_names(architecture) {
        if name.ends_with("ffn_gate_inp.weight") {
            bind_matmul_weight_transposed_f32(
                parsed,
                file_bytes,
                &name,
                name.clone(),
                architecture.expert_count as usize,
                architecture.embedding as usize,
                &mut state,
            )?;
        } else if name.ends_with("ffn_gate_exps.weight") {
            bind_moe_expert_weights(
                parsed,
                file_bytes,
                block_layer(&name)
                    .ok_or_else(|| InteropError::UnknownTensor { name: name.clone() })?
                    as u32,
                "ffn_gate",
                architecture.expert_count,
                architecture.expert_feed_forward as usize,
                architecture.embedding as usize,
                &mut state,
            )?;
        } else if name.ends_with("ffn_up_exps.weight") {
            bind_moe_expert_weights(
                parsed,
                file_bytes,
                block_layer(&name)
                    .ok_or_else(|| InteropError::UnknownTensor { name: name.clone() })?
                    as u32,
                "ffn_up",
                architecture.expert_count,
                architecture.expert_feed_forward as usize,
                architecture.embedding as usize,
                &mut state,
            )?;
        } else if name.ends_with("ffn_down_exps.weight") {
            bind_moe_expert_weights(
                parsed,
                file_bytes,
                block_layer(&name)
                    .ok_or_else(|| InteropError::UnknownTensor { name: name.clone() })?
                    as u32,
                "ffn_down",
                architecture.expert_count,
                architecture.embedding as usize,
                architecture.expert_feed_forward as usize,
                &mut state,
            )?;
        } else if name.ends_with("ssm_beta.weight") || name.ends_with("ssm_alpha.weight") {
            bind_matmul_weight_transposed_f32(
                parsed,
                file_bytes,
                &name,
                name.clone(),
                architecture.ssm_time_step_rank as usize,
                architecture.embedding as usize,
                &mut state,
            )?;
        } else if name.ends_with("attn_output.weight") {
            let (out_dim, in_dim) = dense_attention_dims(architecture, &name)
                .ok_or_else(|| InteropError::UnknownTensor { name: name.clone() })?;
            bind_matmul_weight(parsed, file_bytes, name, out_dim, in_dim, &mut state)?;
        } else if let Some((out_dim, in_dim)) = dense_attention_dims(architecture, &name) {
            bind_matmul_weight(parsed, file_bytes, name, out_dim, in_dim, &mut state)?;
        } else {
            bind_dense(parsed, file_bytes, name, &mut state)?;
        }
    }

    bind_dense(
        parsed,
        file_bytes,
        String::from("output.weight"),
        &mut state,
    )?;

    for layer in 0..architecture.block_count {
        let name = format!("blk.{layer}.exp_probs_b.bias");
        if find_tensor(parsed, &name).is_ok() {
            bind_dense(parsed, file_bytes, name, &mut state)?;
        }
    }

    Ok(state)
}

/// Marker registered for `general.architecture = "qwen35moe"`.
pub struct Qwen35MoeArch;

/// The builtin `qwen35moe` registration value.
pub static QWEN35MOE: Qwen35MoeArch = Qwen35MoeArch;

impl ArchitectureTrait for Qwen35MoeArch {
    fn name(&self) -> &'static str {
        "qwen35moe"
    }

    fn bind<'file>(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
    ) -> Result<BoundProgram<'file>, InteropError> {
        let qwen_architecture = from_metadata(parsed)?;
        let weights = bind_qwen35moe_weights(parsed, file_bytes, &qwen_architecture)?;
        let (program, roots, layer_roots, moe_sites, diagnostics) =
            super::program::qwen35moe_forward_program(&qwen_architecture)?;
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
        };
        Ok(BoundProgram {
            weights,
            architecture,
            program,
            logits_root: roots.logits,
            hidden_root: Some(roots.hidden),
            layer_roots,
            qwen35moe_layer_diagnostics: diagnostics.clone(),
            router_roots: diagnostics
                .iter()
                .map(|diagnostic| diagnostic.router_logits)
                .collect(),
            moe_sites,
            single_position_step: true,
        })
    }
}
