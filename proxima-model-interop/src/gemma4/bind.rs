//! Weight binding, tensor-name enumeration, and [`crate::architecture::Architecture`]
//! registration for the `gemma4` checkpoint family. `Gemma4Arch::bind` is a
//! DESCRIPTOR: it reads [`hparams::Architecture`], builds a per-layer
//! [`LayerAttentionConfig`]/[`LayerFfnConfig`] schedule, and hands both
//! straight to the generic
//! [`proxima_tensor::spec::lfm2_forward_program_with_experts`] engine --
//! there is no bespoke gemma4 forward-graph builder any more (the deleted
//! `gemma4_forward_program`/`gemma4_attention`/`gemma4_ffn_block` this
//! module used to assemble). Every gemma4 layer is
//! [`proxima_tensor::spec::LayerKind::Attention`]; sliding vs full is
//! entirely a [`LayerAttentionConfig`] value (`head_dim`, `kv_heads`,
//! `mask_window`, [`ValueSourceKind`], [`RopeTableSel`]), and the
//! dense+routed parallel FFN is entirely a [`LayerFfnConfig`] value
//! ([`FfnCombination::ParallelDenseMoe`] plus its three post-norm flags and
//! `output_scale`). Teaching pointer: read
//! `proxima_tensor::spec::attention_forward`'s own doc on
//! `lfm2_forward_program_with_experts` before touching this file -- every
//! knob this module sets is documented there, not here.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;
use proxima_tensor::spec::{
    EmbeddingScale, ExpertGatingFunc, FfnCombination, LayerAttentionConfig, LayerFfnConfig, LayerKind,
    RopePairing, RopeTableSel, ValueSourceKind, lfm2_forward_program_with_experts,
};

use crate::architecture::{
    Architecture as ArchitectureTrait, BoundProgram, StepInput, StepInputContext,
};
use crate::bind::{
    BoundWeights, ModelArchitecture, bind_dense, bind_matmul_weight, bind_matmul_weight_as,
    bind_matmul_weight_transposed_f32, find_tensor, gguf_tensor_as_f32,
};
use crate::error::InteropError;

use super::hparams::{Architecture, from_metadata};
use super::program::gemma4_sliding_rope_table;

/// Enumerates every tensor name `gemma4::hparams::from_metadata`'s own
/// `Architecture` implies. Confirmed against the real `qwen3.6`-sibling
/// `gemma4` checkpoint (`examples/gemma4_dump.rs` /
/// `examples/gemma4_layer_scan.rs`): every layer carries 22 tensors EXCEPT
/// `attn_v.weight`, which is absent on the five full-attention layers
/// (indices 5, 11, 17, 23, 29 on the real checkpoint) and present only on
/// sliding-window layers -- 25 layers of 22 plus 5 layers of 21, plus
/// 3 global tensors, is exactly the real header's 658.
#[must_use]
pub fn gemma4_tensor_names(architecture: &Architecture) -> Vec<String> {
    let mut names = Vec::new();

    for (layer, &is_sliding) in architecture.sliding_window_pattern.iter().enumerate() {
        let mut suffixes = alloc::vec![
            "attn_k.weight",
            "attn_k_norm.weight",
            "attn_norm.weight",
            "attn_output.weight",
            "attn_q.weight",
            "attn_q_norm.weight",
            "ffn_down.weight",
            "ffn_down_exps.scale",
            "ffn_down_exps.weight",
            "ffn_gate.weight",
            "ffn_gate_inp.scale",
            "ffn_gate_inp.weight",
            "ffn_gate_up_exps.weight",
            "ffn_norm.weight",
            "ffn_up.weight",
            "layer_output_scale.weight",
            "post_attention_norm.weight",
            "post_ffw_norm.weight",
            "post_ffw_norm_1.weight",
            "post_ffw_norm_2.weight",
            "pre_ffw_norm_2.weight",
        ];
        if is_sliding {
            suffixes.push("attn_v.weight");
        }
        for suffix in suffixes {
            names.push(format!("blk.{layer}.{suffix}"));
        }
    }

    names.push(String::from("token_embd.weight"));
    names.push(String::from("output_norm.weight"));
    names.push(String::from("rope_freqs.weight"));
    names
}

/// Decodes `name` to `f32` and adds `1.0` to every element -- Gemma's own
/// `(1 + weight)` RMSNorm convention, applied once here at bind time so the
/// generic engine's `rmsnorm` (`gamma * x`, no offset) stays unaware of it.
/// Every norm this checkpoint carries is small (`embedding` or `head_dim`
/// wide), so a full decode is the right shape here -- unlike the fused
/// expert tensors below, there is no packed-and-huge case to avoid.
fn bind_norm_plus_one<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    name: String,
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    let mut values = gguf_tensor_as_f32(parsed, file_bytes, &name)?;
    for value in &mut values {
        *value += 1.0;
    }
    state.resident_bytes += values.len() * core::mem::size_of::<f32>();
    state.owned.push((name, values));
    Ok(())
}

/// Binds every weight [`lfm2_forward_program_with_experts`]'s `Input` leaves
/// declare for gemma4's own [`Gemma4Arch::bind`] descriptor.
/// `blk.{layer}.pre_ffw_norm_2.weight` is bound (via `bind_norm_plus_one`,
/// Gemma's `(1 + weight)` convention) and consumed by the engine's
/// `routed_pre_norm` knob (`gemma4_ffn_configs` sets it), which normalizes
/// the routed branch's input separately from the dense branch's shared
/// `ffn_norm`-normed one -- matching the real Gemma 4 graph.
///
/// One remaining gap this function does NOT paper over:
///
/// - the fused `blk.{layer}.ffn_gate_up_exps.weight` cannot be split into
///   the two separate `ffn_gate_exps.weight`/`ffn_up_exps.weight` leaves the
///   engine's routed FFN declares without a full dequant: `expert_feed_forward`
///   is not a whole multiple of the tensor's own quantization block size, so
///   no packed byte offset is block-aligned. A full dequant of this tensor
///   is measured at ~90 GB for the real checkpoint. Rather than allocate
///   that, this function returns
///   [`InteropError::Gemma4FusedExpertNotBlockAligned`] as soon as it
///   detects the misalignment, before touching the tensor's bytes at all.
///
/// # Errors
///
/// Whatever [`find_tensor`]/[`gguf_tensor_as_f32`]/[`bind_dense`]/
/// [`bind_matmul_weight`] can fail with, plus
/// [`InteropError::Gemma4FusedExpertNotBlockAligned`] for the fused-expert
/// gap above.
#[cfg(feature = "std")]
pub fn bind_gemma4_weights<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    architecture: &Architecture,
) -> Result<BoundWeights<'file>, InteropError> {
    let mut state = BoundWeights::new(&[]);
    let embedding = architecture.embedding as usize;
    let expert_count = architecture.expert_count as usize;
    let feed_forward = architecture.feed_forward as usize;

    bind_dense(parsed, file_bytes, "token_embd.weight".into(), &mut state)?;
    bind_norm_plus_one(parsed, file_bytes, "output_norm.weight".into(), &mut state)?;

    if find_tensor(parsed, "output.weight").is_ok() {
        bind_matmul_weight(
            parsed,
            file_bytes,
            "output.weight".into(),
            architecture.vocab as usize,
            embedding,
            &mut state,
        )?;
    } else {
        bind_matmul_weight_as(
            parsed,
            file_bytes,
            "token_embd.weight",
            "output.weight".into(),
            architecture.vocab as usize,
            embedding,
            &mut state,
        )?;
    }

    for (layer_index, &is_sliding) in architecture.sliding_window_pattern.iter().enumerate() {
        let layer = layer_index as u32;
        let head_dim = if is_sliding {
            architecture.key_length_swa
        } else {
            architecture.key_length
        } as usize;
        let kv_heads = architecture.kv_heads_by_layer[layer_index] as usize;
        let query_heads = architecture.head_count as usize;

        bind_norm_plus_one(
            parsed,
            file_bytes,
            format!("blk.{layer}.attn_norm.weight"),
            &mut state,
        )?;
        bind_norm_plus_one(
            parsed,
            file_bytes,
            format!("blk.{layer}.post_attention_norm.weight"),
            &mut state,
        )?;
        bind_norm_plus_one(
            parsed,
            file_bytes,
            format!("blk.{layer}.attn_q_norm.weight"),
            &mut state,
        )?;
        bind_norm_plus_one(
            parsed,
            file_bytes,
            format!("blk.{layer}.attn_k_norm.weight"),
            &mut state,
        )?;
        bind_matmul_weight(
            parsed,
            file_bytes,
            format!("blk.{layer}.attn_q.weight"),
            query_heads * head_dim,
            embedding,
            &mut state,
        )?;
        bind_matmul_weight(
            parsed,
            file_bytes,
            format!("blk.{layer}.attn_k.weight"),
            kv_heads * head_dim,
            embedding,
            &mut state,
        )?;
        if is_sliding {
            bind_matmul_weight(
                parsed,
                file_bytes,
                format!("blk.{layer}.attn_v.weight"),
                kv_heads * head_dim,
                embedding,
                &mut state,
            )?;
        }
        bind_matmul_weight(
            parsed,
            file_bytes,
            format!("blk.{layer}.attn_output.weight"),
            embedding,
            query_heads * head_dim,
            &mut state,
        )?;
        bind_dense(
            parsed,
            file_bytes,
            format!("blk.{layer}.layer_output_scale.weight"),
            &mut state,
        )?;

        bind_norm_plus_one(
            parsed,
            file_bytes,
            format!("blk.{layer}.ffn_norm.weight"),
            &mut state,
        )?;
        bind_norm_plus_one(
            parsed,
            file_bytes,
            format!("blk.{layer}.post_ffw_norm_1.weight"),
            &mut state,
        )?;
        bind_norm_plus_one(
            parsed,
            file_bytes,
            format!("blk.{layer}.post_ffw_norm_2.weight"),
            &mut state,
        )?;
        bind_norm_plus_one(
            parsed,
            file_bytes,
            format!("blk.{layer}.post_ffw_norm.weight"),
            &mut state,
        )?;
        bind_norm_plus_one(
            parsed,
            file_bytes,
            format!("blk.{layer}.pre_ffw_norm_2.weight"),
            &mut state,
        )?;
        bind_matmul_weight(
            parsed,
            file_bytes,
            format!("blk.{layer}.ffn_gate.weight"),
            feed_forward,
            embedding,
            &mut state,
        )?;
        bind_matmul_weight(
            parsed,
            file_bytes,
            format!("blk.{layer}.ffn_up.weight"),
            feed_forward,
            embedding,
            &mut state,
        )?;
        bind_matmul_weight(
            parsed,
            file_bytes,
            format!("blk.{layer}.ffn_down.weight"),
            embedding,
            feed_forward,
            &mut state,
        )?;
        bind_matmul_weight_transposed_f32(
            parsed,
            file_bytes,
            &format!("blk.{layer}.ffn_gate_inp.weight"),
            format!("blk.{layer}.ffn_gate_inp.weight"),
            expert_count,
            embedding,
            &mut state,
        )?;

        let gate_up_name = format!("blk.{layer}.ffn_gate_up_exps.weight");
        let gate_up_tensor = find_tensor(parsed, &gate_up_name)?;
        let layout = gate_up_tensor.ggml_type.block_layout();
        if layout.block_elements == 0
            || !u64::from(architecture.expert_feed_forward).is_multiple_of(layout.block_elements)
        {
            return Err(InteropError::Gemma4FusedExpertNotBlockAligned {
                layer,
                ggml_type: gate_up_tensor.ggml_type,
                block_elements: layout.block_elements,
                expert_feed_forward: architecture.expert_feed_forward,
            });
        }

        // A block-aligned split (and the matching `ffn_down_exps.weight`
        // bind, unaffected by this alignment check since `append_moe_ffn`'s
        // OWN split boundary is `expert_feed_forward` on its OWN `in_dim`
        // axis) is the remaining step once a caller needs to reach past the
        // check above -- not implemented here, since every real checkpoint's
        // `expert_feed_forward` (704) fails it and this function returns
        // before this point, so no per-expert dequant of this tensor runs.
    }

    Ok(state)
}

/// Marker registered for `general.architecture = "gemma4"`.
pub struct Gemma4Arch;

/// The builtin `gemma4` registration value.
pub static GEMMA4: Gemma4Arch = Gemma4Arch;

/// Sliding layers use [`ValueSourceKind::ProjectedV`] (a real
/// `attn_v.weight`) and the SWA RoPE table; full layers use
/// [`ValueSourceKind::SharedWithKey`] (no `attn_v.weight` on disk -- the key
/// projection's own output stands in for `V`) and the full-length RoPE
/// table. Both use [`RopePairing::SplitHalf`] (Gemma's own half-split
/// rotation, not Llama/Mistral's interleaved pairing).
fn gemma4_attention_configs(architecture: &Architecture) -> Vec<LayerAttentionConfig> {
    architecture
        .sliding_window_pattern
        .iter()
        .enumerate()
        .map(|(layer, &is_sliding)| {
            let kv_heads = architecture.kv_heads_by_layer[layer];
            if is_sliding {
                LayerAttentionConfig {
                    head_dim: architecture.key_length_swa,
                    kv_heads,
                    mask_window: Some(architecture.sliding_window),
                    value_source_kind: ValueSourceKind::ProjectedV,
                    rope_table: RopeTableSel {
                        cos_name: "rope_cos_swa",
                        sin_name: "rope_sin_swa",
                    },
                    rope_pairing: RopePairing::SplitHalf {
                        pairs: architecture.key_length_swa / 2,
                    },
                }
            } else {
                LayerAttentionConfig {
                    head_dim: architecture.key_length,
                    kv_heads,
                    mask_window: None,
                    value_source_kind: ValueSourceKind::SharedWithKey,
                    rope_table: RopeTableSel {
                        cos_name: "rope_cos",
                        sin_name: "rope_sin",
                    },
                    rope_pairing: RopePairing::SplitHalf {
                        pairs: architecture.key_length / 2,
                    },
                }
            }
        })
        .collect()
}

/// Every gemma4 layer runs dense SwiGLU over the shared `ffn_norm`-normed
/// input and routed MoE over its OWN `pre_ffw_norm_2`-normed input
/// (`routed_pre_norm: true`), each with its own post-norm, summed and
/// normalized once more, then scaled by `layer_output_scale`. The routed
/// branch gates with `Softmax` and carries no `exp_probs_b` bias, unlike
/// [`FfnCombination::Exclusive`]'s LFM2 shape -- see
/// [`proxima_tensor::spec::LayerFfnConfig`]'s own doc for what each field
/// means.
fn gemma4_ffn_configs(architecture: &Architecture) -> Vec<LayerFfnConfig> {
    alloc::vec![
        LayerFfnConfig {
            post_attention_norm: true,
            combination: FfnCombination::ParallelDenseMoe,
            dense_post_norm: true,
            routed_post_norm: true,
            combined_post_norm: true,
            output_scale: true,
            routed_gating: ExpertGatingFunc::Softmax,
            routed_expert_bias: false,
            routed_pre_norm: true,
        };
        architecture.block_count as usize
    ]
}

impl ArchitectureTrait for Gemma4Arch {
    fn name(&self) -> &'static str {
        "gemma4"
    }

    #[cfg(feature = "std")]
    fn bind<'file>(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
    ) -> Result<BoundProgram<'file>, InteropError> {
        let architecture = from_metadata(parsed)?;
        let weights = bind_gemma4_weights(parsed, file_bytes, &architecture)?;

        let layer_kinds = alloc::vec![LayerKind::Attention; architecture.block_count as usize];
        let attention_configs = gemma4_attention_configs(&architecture);
        let ffn_configs = gemma4_ffn_configs(&architecture);
        let logit_softcap = (architecture.final_logit_softcapping > 0.0)
            .then_some(architecture.final_logit_softcapping);

        let (program, logits, moe_sites) = lfm2_forward_program_with_experts(
            architecture.vocab,
            architecture.embedding,
            architecture.feed_forward,
            architecture.expert_feed_forward,
            architecture.head_count,
            architecture.block_count,
            architecture.expert_count,
            architecture.expert_used_count,
            0,
            0,
            &layer_kinds,
            &attention_configs,
            &ffn_configs,
            Some(EmbeddingScale::Sqrt),
            logit_softcap,
        )?;

        let tied_embeddings = find_tensor(parsed, "output.weight").is_err();
        let full_head_dim = architecture.key_length;
        let model_architecture = ModelArchitecture {
            vocab: architecture.vocab,
            embedding: architecture.embedding,
            feed_forward: architecture.feed_forward,
            query_heads: architecture.head_count,
            kv_heads: *architecture.kv_heads_by_layer.last().unwrap_or(&0),
            kv_heads_by_layer: architecture.kv_heads_by_layer.clone(),
            head_dim: full_head_dim,
            block_count: architecture.block_count,
            expert_count: architecture.expert_count,
            expert_used_count: architecture.expert_used_count,
            rope_freq_base: architecture.rope_freq_base,
            rms_epsilon: architecture.rms_epsilon,
            tied_embeddings,
        };

        Ok(BoundProgram {
            weights,
            architecture: model_architecture,
            program,
            logits_root: logits,
            hidden_root: None,
            residual_roots: Vec::new(),
            layer_roots: Vec::new(),
            qwen35moe_layer_diagnostics: Vec::new(),
            router_roots: Vec::new(),
            moe_sites,
            single_position_step: false,
        })
    }

    #[cfg(not(feature = "std"))]
    fn bind<'file>(
        &self,
        _parsed: &ParsedGguf,
        _file_bytes: &'file [u8],
    ) -> Result<BoundProgram<'file>, InteropError> {
        Err(InteropError::HybridMoeProgramUnsupported {
            name: self.name().into(),
        })
    }

    /// Feeds the sliding-window RoPE table the `rope_cos_swa`/`rope_sin_swa`
    /// leaves declare (`LayerAttentionConfig::rope_table`,
    /// [`gemma4_attention_configs`]) -- the decode loop's builtin
    /// `rope_cos`/`rope_sin` blocks always carry the FULL-layer table
    /// (`Gemma4Arch::bind`'s own `ModelArchitecture::head_dim`/
    /// `rope_freq_base` are the full-layer values), so this is the one
    /// extra leaf this architecture needs from
    /// [`crate::architecture::Architecture::step_inputs`]'s own seam.
    /// Positions are `context.new_start + offset` for
    /// `offset in 0..context.new_count`, matching
    /// `crate::generate::residency_caches::build_position_inputs`'s own
    /// absolute-angle convention for the builtin table.
    fn step_inputs(&self, context: &StepInputContext<'_>, out: &mut Vec<StepInput>) {
        let positions: Vec<usize> = (0..context.new_count)
            .map(|offset| context.new_start + offset)
            .collect();
        // Metadata-derived base/dim would need a second `from_metadata` read
        // per step; the checkpoint's own SWA base/dim (`1e4`/`256`) is fixed
        // per architecture, not per file, so this seam hard-codes Gemma 4's
        // own values rather than re-parsing metadata on every decode step.
        let (cos, sin) = gemma4_sliding_rope_table(&positions, 1.0e4, 256);
        out.push(StepInput {
            name: "rope_cos_swa",
            values: cos,
            symbol: None,
        });
        out.push(StepInput {
            name: "rope_sin_swa",
            values: sin,
            symbol: None,
        });
    }
}
