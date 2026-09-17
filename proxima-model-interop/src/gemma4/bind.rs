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
    Activation, AttentionScoreScale, EmbeddingScale, ExpertGatingFunc, FfnCombination,
    LayerAttentionConfig, LayerFfnConfig, LayerKind, LayerSchedule, ParallelDenseMoeConfig,
    RopePairing, RopeTableSel, ValueSourceKind, lfm2_forward_program_with_experts,
};

use crate::architecture::{
    Architecture as ArchitectureTrait, BoundProgram, StepInput, StepInputContext,
};
use crate::bind::{
    BoundWeights, ModelArchitecture, PackedOwnedKind, bind_dense, bind_matmul_weight,
    bind_matmul_weight_as, bind_matmul_weight_transposed_f32, bind_moe_expert_weights, find_tensor,
    gguf_tensor_as_f32,
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

/// Binds `rope_freqs.weight` (GGUF `ROPE_FREQS`) as raw, unmodified `f32`
/// values -- the per-pair frequency-scaling factor
/// [`Gemma4Arch::rope_freq_factors`] hands back to
/// `crate::generate::build_position_inputs`, which divides each full-layer
/// RoPE pair's angle by it. Unlike [`bind_norm`]'s norms this is not an
/// RMSNorm gamma shift, and it declares no `Op::Input` leaf the
/// forward program consumes -- it rides in [`BoundWeights::owned`] purely
/// as a lookup table [`Gemma4Arch::rope_freq_factors`] reads back out by
/// name, the same way every other bound weight is name-tagged there.
fn bind_rope_freqs<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    let values = gguf_tensor_as_f32(parsed, file_bytes, "rope_freqs.weight")?;
    state.resident_bytes += values.len() * core::mem::size_of::<f32>();
    state.owned.push((String::from("rope_freqs.weight"), values));
    Ok(())
}

/// The RMSNorm gamma shift this checkpoint family stores on disk, added to
/// every norm weight at bind time so the generic engine's `rmsnorm`
/// (`gamma * x`, no offset) stays unaware of the convention.
/// `modeling_gemma4.py`'s `Gemma4RMSNorm` is ones-init and applies
/// `normed * weight` directly (no `+ 1`) -- unlike gemma3's zero-init
/// `(1 + weight)` convention, whose shift is `1.0`. Gemma 4's GGUF already
/// stores the full effective gamma, so this is `0.0`: shifting by it is a
/// byte-identical no-op, keeping the convention explicit data instead of a
/// baked-in function name.
const GEMMA4_NORM_SHIFT: f32 = 0.0;

/// Decodes `name` to `f32` and adds `norm_shift` to every element -- the
/// RMSNorm gamma convention this checkpoint family uses, applied once here
/// at bind time so the generic engine's `rmsnorm` (`gamma * x`, no offset)
/// stays unaware of it. Every norm this checkpoint carries is small
/// (`embedding` or `head_dim` wide), so a full decode is the right shape
/// here -- unlike the fused expert tensors below, there is no
/// packed-and-huge case to avoid. Gemma 4 passes [`GEMMA4_NORM_SHIFT`]
/// (`0.0`, ones-init `normed * weight`); Gemma 3's zero-init
/// `(1 + weight)` convention would pass `1.0` here instead -- the shift is
/// a config value, not a hard-coded convention.
fn bind_norm<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    name: String,
    norm_shift: f32,
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    let mut values = gguf_tensor_as_f32(parsed, file_bytes, &name)?;
    if norm_shift != 0.0 {
        for value in &mut values {
            *value += norm_shift;
        }
    }
    state.resident_bytes += values.len() * core::mem::size_of::<f32>();
    state.owned.push((name, values));
    Ok(())
}

/// Binds every weight [`lfm2_forward_program_with_experts`]'s `Input` leaves
/// declare for gemma4's own [`Gemma4Arch::bind`] descriptor.
/// `blk.{layer}.pre_ffw_norm_2.weight` is bound (via [`bind_norm`] with
/// [`GEMMA4_NORM_SHIFT`]) and consumed by the engine's
/// `routed_pre_norm` knob (`gemma4_layer_schedule` sets it), which normalizes
/// the routed branch's input separately from the dense branch's shared
/// `ffn_norm`-normed one -- matching the real Gemma 4 graph.
///
/// The fused `blk.{layer}.ffn_gate_up_exps.weight` splits into the two
/// separate `ffn_gate_exps.weight`/`ffn_up_exps.weight` leaves the engine's
/// routed FFN declares WITHOUT a dequant -- see
/// [`bind_gemma4_fused_gate_up_experts`]'s own doc for the confirmed axis
/// (the real checkpoint's `ne[0]`=2816 embedding is the quantization block
/// axis; `ne[1]`=1408=2*`expert_feed_forward` is the row axis the split
/// cuts, orthogonal to blocks) and the packed-memcpy implementation.
///
/// # Errors
///
/// Whatever [`find_tensor`]/[`gguf_tensor_as_f32`]/[`bind_dense`]/
/// [`bind_matmul_weight`]/[`bind_gemma4_fused_gate_up_experts`]/
/// [`bind_moe_expert_weights`] can fail with.
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
    bind_norm(
        parsed,
        file_bytes,
        "output_norm.weight".into(),
        GEMMA4_NORM_SHIFT,
        &mut state,
    )?;
    bind_rope_freqs(parsed, file_bytes, &mut state)?;

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

        bind_norm(
            parsed,
            file_bytes,
            format!("blk.{layer}.attn_norm.weight"),
            GEMMA4_NORM_SHIFT,
            &mut state,
        )?;
        bind_norm(
            parsed,
            file_bytes,
            format!("blk.{layer}.post_attention_norm.weight"),
            GEMMA4_NORM_SHIFT,
            &mut state,
        )?;
        bind_norm(
            parsed,
            file_bytes,
            format!("blk.{layer}.attn_q_norm.weight"),
            GEMMA4_NORM_SHIFT,
            &mut state,
        )?;
        bind_norm(
            parsed,
            file_bytes,
            format!("blk.{layer}.attn_k_norm.weight"),
            GEMMA4_NORM_SHIFT,
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

        bind_norm(
            parsed,
            file_bytes,
            format!("blk.{layer}.ffn_norm.weight"),
            GEMMA4_NORM_SHIFT,
            &mut state,
        )?;
        bind_norm(
            parsed,
            file_bytes,
            format!("blk.{layer}.post_ffw_norm_1.weight"),
            GEMMA4_NORM_SHIFT,
            &mut state,
        )?;
        bind_norm(
            parsed,
            file_bytes,
            format!("blk.{layer}.post_ffw_norm_2.weight"),
            GEMMA4_NORM_SHIFT,
            &mut state,
        )?;
        bind_norm(
            parsed,
            file_bytes,
            format!("blk.{layer}.post_ffw_norm.weight"),
            GEMMA4_NORM_SHIFT,
            &mut state,
        )?;
        bind_norm(
            parsed,
            file_bytes,
            format!("blk.{layer}.pre_ffw_norm_2.weight"),
            GEMMA4_NORM_SHIFT,
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
        // `[embedding]` F32, NOT a dequant scale -- `ffn_gate_inp.weight`
        // is already F32 with its own values; this is the SEPARATE
        // architectural router-input scale. `append_routed_expert_ffn`'s
        // `router_scale` knob binds this raw (no `1 +` offset) as the
        // gamma of a `with_scale=False` RMSNorm over the router's own
        // input, then multiplies by the constant `embedding**-0.5`, before
        // the router projection (`Gemma4TextRouter.forward`). Confirmed
        // via `gemma4_dump` (`examples/gemma4_dump.rs`): `ggml_type=F32`,
        // `dims=[2816]`.
        bind_dense(
            parsed,
            file_bytes,
            format!("blk.{layer}.ffn_gate_inp.scale"),
            &mut state,
        )?;

        let expert_feed_forward = architecture.expert_feed_forward as usize;
        bind_gemma4_fused_gate_up_experts(
            parsed,
            file_bytes,
            layer,
            expert_count,
            expert_feed_forward,
            embedding,
            &mut state,
        )?;
        bind_moe_expert_weights(
            parsed,
            file_bytes,
            layer,
            "ffn_down",
            architecture.expert_count,
            embedding,
            expert_feed_forward,
            &mut state,
        )?;
        // `[expert_count]` F32, ARCHITECTURAL (not a dequant scale --
        // `ffn_down_exps.weight` is Q5_1 with its own block scales). Folded
        // into each selected expert's combination weight,
        // [`append_moe_ffn`]'s own doc on `MoeFfnSpec::expert_scale`.
        bind_dense(
            parsed,
            file_bytes,
            format!("blk.{layer}.ffn_down_exps.scale"),
            &mut state,
        )?;
    }

    Ok(state)
}

/// Splits the real checkpoint's fused `blk.{layer}.ffn_gate_up_exps.weight`
/// into the two separate `ffn_gate_exps.weight`/`ffn_up_exps.weight` leaves
/// [`lfm2_forward_program_with_experts`]'s routed FFN declares, by a packed
/// byte memcpy -- no dequantize, no new kernel.
///
/// Axis confirmed against the real checkpoint (`examples/gemma4_dump.rs`,
/// `cargo run --release --example gemma4_dump`): the fused tensor's
/// `dims = [2816, 1408, 128]` (`ne0`=embedding, `ne1`=2*`expert_feed_forward`,
/// `ne2`=expert_count), `Q3_K`, `block_elements=256`. `ne0` (2816 = 11*256)
/// is the quantization block axis; `ne1` (1408) is the row axis the
/// gate/up split cuts at row `expert_feed_forward` (704), which is
/// orthogonal to `ne0`'s blocks -- every row is a whole number of blocks
/// regardless of where the row-axis split falls, so the split never crosses
/// a block boundary. `ggml`/GGUF layout is row-major with `ne0` fastest, so
/// one expert's `ne1` rows are contiguous in the packed buffer: gate rows
/// `[0, expert_feed_forward)` and up rows
/// `[expert_feed_forward, 2*expert_feed_forward)` are each one contiguous
/// byte span per expert. Experts themselves are NOT contiguous across that
/// boundary (expert `e+1`'s gate bytes follow expert `e`'s up bytes, not
/// expert `e`'s gate bytes), so the two halves cannot be exposed as a
/// single strided borrow the way [`bind_moe_expert_weights`]'s
/// already-native-stacked fast path does -- each half is assembled into its
/// own owned packed buffer, one packed memcpy per expert per half, matching
/// the two `Q3_K`-tagged [`PackedOwnedKind::Q3K`] buffers
/// [`BoundWeights::packed_owned`] already carries for every other MoE
/// family's restack fallback ([`bind_moe_expert_weights`]).
///
/// # Errors
///
/// [`InteropError::UnknownTensor`] if the fused tensor is absent;
/// [`InteropError::UnrepresentableGgmlType`] if its `ggml_type` has no
/// [`PackedOwnedKind`] (every codec a real gemma4 checkpoint ships does);
/// whatever [`ParsedGguf::tensor_data_range`] can fail with if the tensor's
/// declared byte range does not fit `file_bytes`.
fn bind_gemma4_fused_gate_up_experts<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    layer: u32,
    expert_count: usize,
    expert_feed_forward: usize,
    embedding: usize,
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    let name = format!("blk.{layer}.ffn_gate_up_exps.weight");
    let tensor = find_tensor(parsed, &name)?;
    let layout = tensor.ggml_type.block_layout();
    let kind = PackedOwnedKind::from_ggml_type(tensor.ggml_type).ok_or_else(|| {
        InteropError::UnrepresentableGgmlType {
            tensor: name.clone(),
            ggml_type: tensor.ggml_type,
        }
    })?;

    let range = parsed.tensor_data_range(tensor, file_bytes.len() as u64)?;
    let source = &file_bytes[range.start as usize..range.end as usize];

    let bytes_per_row = (embedding as u64 / layout.block_elements) * layout.block_bytes;
    let gate_bytes = expert_feed_forward as u64 * bytes_per_row;
    let per_expert_bytes = 2 * gate_bytes;

    let mut gate_buf = Vec::with_capacity(gate_bytes as usize * expert_count);
    let mut up_buf = Vec::with_capacity(gate_bytes as usize * expert_count);

    for expert in 0..expert_count {
        let expert_start = expert as u64 * per_expert_bytes;
        let gate_start = expert_start as usize;
        let gate_end = gate_start + gate_bytes as usize;
        let up_end = gate_end + gate_bytes as usize;
        gate_buf.extend_from_slice(&source[gate_start..gate_end]);
        up_buf.extend_from_slice(&source[gate_end..up_end]);
    }

    state.resident_bytes += gate_buf.len() + up_buf.len();
    state
        .packed_owned
        .push((format!("blk.{layer}.ffn_gate_exps.weight"), gate_buf, kind));
    state
        .packed_owned
        .push((format!("blk.{layer}.ffn_up_exps.weight"), up_buf, kind));
    Ok(())
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
/// Every gemma4 layer is [`LayerKind::Attention`], runs dense SwiGLU over
/// the shared `ffn_norm`-normed input and routed MoE over its OWN
/// `pre_ffw_norm_2`-normed input (`routed_pre_norm: true`), each with its
/// own post-norm, summed and normalized once more, then scaled by
/// `layer_output_scale`. The routed branch gates with `Softmax` and carries
/// no `exp_probs_b` bias, unlike [`FfnCombination::Exclusive`]'s LFM2 shape
/// -- see [`proxima_tensor::spec::LayerFfnConfig`]'s own doc for what each
/// field means.
fn gemma4_layer_schedule(architecture: &Architecture) -> Vec<LayerSchedule> {
    let ffn = LayerFfnConfig {
        post_attention_norm: true,
        combination: FfnCombination::ParallelDenseMoe(ParallelDenseMoeConfig {
            dense_post_norm: true,
            routed_post_norm: true,
            combined_post_norm: true,
            routed_pre_norm: true,
            router_scale: true,
            expert_output_scale: true,
        }),
        output_scale: true,
        routed_gating: ExpertGatingFunc::Softmax,
        routed_expert_bias: false,
        activation: Activation::GeluTanh,
    };
    architecture
        .sliding_window_pattern
        .iter()
        .enumerate()
        .map(|(layer, &is_sliding)| {
            let kv_heads = architecture.kv_heads_by_layer[layer];
            let attention = if is_sliding {
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
                    // `Gemma4TextAttention.forward`: `self.scaling = 1.0` for
                    // every layer, sliding included -- gemma4 has no
                    // `query_pre_attn_scalar` at all (that is a gemma2/3
                    // convention this architecture does not inherit).
                    score_scale: AttentionScoreScale::Unscaled,
                    // `Gemma4TextAttention.forward` (`modeling_gemma4.py:1256-1265`):
                    // `v_norm` applies to EVERY layer's `V`, sliding and full
                    // alike -- `self.v_norm` has no per-layer-type branch.
                    value_norm: true,
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
                    // same `self.scaling = 1.0` as the sliding branch above --
                    // HF applies no per-layer-type distinction here.
                    score_scale: AttentionScoreScale::Unscaled,
                    value_norm: true,
                }
            };
            LayerSchedule {
                kind: LayerKind::Attention,
                attention,
                ffn,
            }
        })
        .collect()
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

        let schedule = gemma4_layer_schedule(&architecture);
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
            &schedule,
            Some(EmbeddingScale::Sqrt),
            logit_softcap,
            true,
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
    /// [`gemma4_layer_schedule`]) -- the decode loop's builtin
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

    /// llama.cpp's authoritative gemma4 graph (`src/models/gemma4.cpp`)
    /// applies `ggml_rope_ext` to full/global-attention layers with
    /// `n_rot=head_dim` (matching this checkpoint's `rope.dimension_count`
    /// metadata) but ALSO a per-pair `freq_factors` tensor
    /// (`rope_freqs.weight`, GGUF `ROPE_FREQS`) that ggml divides each
    /// pair's angle by -- [`bind_rope_freqs`] binds that tensor's own values
    /// verbatim into [`BoundWeights::owned`] at bind time, and this method
    /// hands the same slice straight back to
    /// [`crate::generate::build_position_inputs`], which does the dividing.
    /// The real checkpoint's `rope_freqs.weight` holds `[1.0]*64 +
    /// [1e30]*192` (confirmed shape: 256 = `head_dim/2` pair entries):
    /// dividing by `1.0` is a no-op for the first 64 pairs, and dividing by
    /// `1e30` collapses the remaining 192 pairs' `theta` far enough below
    /// one radian that `cos` rounds to exactly `1.0f32` and `sin` rounds to
    /// float noise -- the data-driven replacement for what this method used
    /// to do by returning the literal `64` and letting
    /// `build_position_inputs` truncate its loop there. `rope.
    /// dimension_count=512` is `head_dim`, i.e. `n_rot`, NOT the rotary
    /// count -- the earlier `head_dim / 2` (full rotation of every pair)
    /// read that metadata field as the rotary width, which was the
    /// original bug this checkpoint's own `rope_freqs.weight` now fixes
    /// directly, with no hard-coded pair count anywhere in this crate. The
    /// SWA table ([`Self::step_inputs`]/[`gemma4_sliding_rope_table`])
    /// carries no `freq_factors` in the real graph and is untouched --
    /// `rope_freqs.weight` is never bound into its own separate table.
    fn rope_freq_factors<'weights>(
        &self,
        weights: &'weights BoundWeights<'_>,
    ) -> Option<&'weights [f32]> {
        weights
            .owned
            .iter()
            .find(|(name, _)| name == "rope_freqs.weight")
            .map(|(_, values)| values.as_slice())
    }
}
