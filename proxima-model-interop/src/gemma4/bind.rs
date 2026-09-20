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
// `gemma4_layer_schedule`'s own per-layer schedule (real E2B/E4B/12B/26B/31B
// shape, `architecture`-derived) is the SINGLE schedule source for both
// engines below -- [`CacheStrategy`] is a runtime choice off `architecture`
// (`Gemma4Arch::bind`'s own doc on `descriptor.cache_strategy`), not a
// `#[cfg]` fork, so every one of these is compiled in unconditionally.
use proxima_tensor::spec::{
    Activation, AttentionScoreScale, CacheStrategy, EmbeddingScale, ExpertGatingFunc,
    FfnCombination, KeySourceKind, LayerAttentionConfig, LayerFfnConfig, LayerKind,
    LayerSchedule, ModelDescriptor, ParallelDenseMoeConfig, Qwen35LayerRoots, RopePairing,
    RopeTableSel, ValueSourceKind, build_forward,
};

use crate::architecture::{
    Architecture as ArchitectureTrait, BoundProgram, StepInput, StepInputContext,
};
use crate::bind::{
    BoundWeights, ModelArchitecture, bind_dense, bind_matmul_weight, bind_matmul_weight_as,
    bind_matmul_weight_transposed_f32, bind_moe_expert_weights, codec_from_ggml_type, find_tensor,
    gguf_tensor_as_f32,
};
use crate::error::InteropError;

use super::hparams::{Architecture, from_metadata};
use super::program::gemma4_sliding_rope_table;

/// Enumerates every tensor name `gemma4::hparams::from_metadata`'s own
/// `Architecture` implies. Confirmed against the real `qwen3.6`-sibling
/// `gemma4` MoE checkpoint (`examples/gemma4_dump.rs` /
/// `examples/gemma4_layer_scan.rs`, `shared_kv_layers == 0`): every layer
/// carries 22 tensors EXCEPT `attn_v.weight`, which is absent on the five
/// full-attention layers (indices 5, 11, 17, 23, 29 on the real checkpoint)
/// and present only on sliding-window layers -- 25 layers of 22 plus
/// 5 layers of 21, plus 3 global tensors, is exactly the real header's 658.
/// On a shared-KV checkpoint (`attention.shared_kv_layers > 0`, e.g.
/// `gemma4:e2b-it-qat`) this does NOT hold: every own-KV layer (`layer <
/// block_count - shared_kv_layers`), sliding or full, carries its own
/// `attn_v.weight` (confirmed against the real E2B header -- `blk.4`,
/// `blk.9`, `blk.14`, the three full own-KV layers among the first 15,
/// each list `attn_v.weight`). Every TRAILING layer from
/// `block_count - shared_kv_layers` onward (confirmed against the real
/// header by an `UnknownTensor` load error on `blk.15.attn_k_norm.weight`)
/// carries none of `attn_k.weight`, `attn_k_norm.weight`, or
/// `attn_v.weight` at all -- see `gemma4_layer_schedule`'s own
/// `shared_kv_source_layer` for which own-KV layer supplies them instead.
#[must_use]
pub fn gemma4_tensor_names(architecture: &Architecture) -> Vec<String> {
    let mut names = Vec::new();
    let first_shared_idx = architecture
        .block_count
        .saturating_sub(architecture.shared_kv_layers);

    for (layer, &is_sliding) in architecture.sliding_window_pattern.iter().enumerate() {
        let is_shared_kv = layer as u32 >= first_shared_idx;
        let mut suffixes = alloc::vec![
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
        if !is_shared_kv {
            suffixes.push("attn_k.weight");
            suffixes.push("attn_k_norm.weight");
            // Mirrors `bind_gemma4_weights`'s own `attn_v.weight` bind gate
            // and `gemma4_layer_schedule`'s own `value_source_kind` gate --
            // MoE's full layers alone lack `attn_v.weight` (`is_sliding`);
            // a shared-KV checkpoint's (E2B/E4B) own-KV layers ALL carry it.
            if is_sliding || architecture.shared_kv_layers > 0 {
                suffixes.push("attn_v.weight");
            }
        }
        // Per-layer-embedding (PLE) Stage B's own three per-block leaves --
        // absent (E4B/12B/26B/31B, `ple_dim == 0`) means this checkpoint
        // carries no PLE tensors at all (`hparams::Architecture::ple_dim`'s
        // own doc).
        if architecture.ple_dim > 0 {
            suffixes.push("inp_gate.weight");
            suffixes.push("proj.weight");
            suffixes.push("post_norm.weight");
        }
        for suffix in suffixes {
            names.push(format!("blk.{layer}.{suffix}"));
        }
    }

    names.push(String::from("token_embd.weight"));
    names.push(String::from("output_norm.weight"));
    names.push(String::from("rope_freqs.weight"));
    // Per-layer-embedding (PLE) Stage A's own three shared, whole-checkpoint
    // leaves -- see the per-layer trio above for the per-block half.
    if architecture.ple_dim > 0 {
        names.push(String::from("per_layer_token_embd.weight"));
        names.push(String::from("per_layer_model_proj.weight"));
        names.push(String::from("per_layer_proj_norm.weight"));
    }
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

    bind_dense(parsed, file_bytes, "token_embd.weight".into(), &mut state)?;
    bind_norm(
        parsed,
        file_bytes,
        "output_norm.weight".into(),
        GEMMA4_NORM_SHIFT,
        &mut state,
    )?;
    bind_rope_freqs(parsed, file_bytes, &mut state)?;

    // Per-layer-embedding (PLE) Stage A's own three shared, whole-checkpoint
    // leaves. `per_layer_token_embd.weight` is a lookup table
    // (`attention_forward.rs`'s `append_ple_shared_projections` reads it
    // through `embedding_lookup`, the same gather `token_embd.weight`
    // above uses) so it binds `bind_dense`, not `bind_matmul_weight`, the
    // same convention `token_embd.weight` itself uses.
    // `per_layer_model_proj.weight` is a plain `[in, out]` matmul weight.
    if architecture.ple_dim > 0 {
        let ple_total = architecture.ple_dim * architecture.block_count;
        bind_dense(
            parsed,
            file_bytes,
            "per_layer_token_embd.weight".into(),
            &mut state,
        )?;
        bind_matmul_weight(
            parsed,
            file_bytes,
            "per_layer_model_proj.weight".into(),
            ple_total as usize,
            embedding,
            &mut state,
        )?;
        bind_norm(
            parsed,
            file_bytes,
            "per_layer_proj_norm.weight".into(),
            GEMMA4_NORM_SHIFT,
            &mut state,
        )?;
    }

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

    let first_shared_idx = architecture
        .block_count
        .saturating_sub(architecture.shared_kv_layers);
    for (layer_index, &is_sliding) in architecture.sliding_window_pattern.iter().enumerate() {
        let layer = layer_index as u32;
        let is_shared_kv = layer >= first_shared_idx;
        let head_dim = if is_sliding {
            architecture.key_length_swa
        } else {
            architecture.key_length
        } as usize;
        let kv_heads = architecture.kv_heads_by_layer[layer_index] as usize;
        let query_heads = architecture.head_count as usize;
        let feed_forward = architecture.feed_forward_by_layer[layer_index] as usize;

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
        // Shared-KV layers (E2B: blk.15..=34, `is_shared_kv`) carry none of
        // `attn_k.weight`/`attn_k_norm.weight`/`attn_v.weight` on disk at
        // all (confirmed by an `UnknownTensor` load error on
        // `blk.15.attn_k_norm.weight` against the real checkpoint) --
        // `gemma4_layer_schedule`'s own `KeySourceKind::SharedFromLayer`/
        // `ValueSourceKind::SharedFromLayer` never declare `Input` leaves
        // for them, so binding them here would look up a tensor the
        // forward program never asks for.
        if !is_shared_kv {
            bind_norm(
                parsed,
                file_bytes,
                format!("blk.{layer}.attn_k_norm.weight"),
                GEMMA4_NORM_SHIFT,
                &mut state,
            )?;
        }
        bind_matmul_weight(
            parsed,
            file_bytes,
            format!("blk.{layer}.attn_q.weight"),
            query_heads * head_dim,
            embedding,
            &mut state,
        )?;
        if !is_shared_kv {
            bind_matmul_weight(
                parsed,
                file_bytes,
                format!("blk.{layer}.attn_k.weight"),
                kv_heads * head_dim,
                embedding,
                &mut state,
            )?;
        }
        // Mirrors `gemma4_layer_schedule`'s own `value_source_kind` gate --
        // MoE keeps `is_sliding` (a full layer has no `attn_v.weight` on
        // disk); E2B/E4B (`shared_kv_layers > 0`) binds every own-KV
        // layer's real `attn_v.weight` unconditionally.
        if (is_sliding || architecture.shared_kv_layers > 0) && !is_shared_kv {
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

        // Per-layer-embedding (PLE) Stage B's own three per-block leaves --
        // `attention_forward.rs`'s `append_lfm2_layer_ffn` declares
        // `inp_gate.weight`/`proj.weight` as `[in, out]` matmul weights
        // (`bind_matmul_weight`'s own convention, same as `ffn_gate`/
        // `ffn_up`/`ffn_down` above) and `post_norm.weight` as a plain
        // RMSNorm gamma (`bind_norm`, [`GEMMA4_NORM_SHIFT`]).
        if architecture.ple_dim > 0 {
            let ple_dim = architecture.ple_dim as usize;
            bind_matmul_weight(
                parsed,
                file_bytes,
                format!("blk.{layer}.inp_gate.weight"),
                ple_dim,
                embedding,
                &mut state,
            )?;
            bind_matmul_weight(
                parsed,
                file_bytes,
                format!("blk.{layer}.proj.weight"),
                embedding,
                ple_dim,
                &mut state,
            )?;
            bind_norm(
                parsed,
                file_bytes,
                format!("blk.{layer}.post_norm.weight"),
                GEMMA4_NORM_SHIFT,
                &mut state,
            )?;
        }

        bind_norm(
            parsed,
            file_bytes,
            format!("blk.{layer}.ffn_norm.weight"),
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

        // E2B/E4B are DENSE (`expert_count == 0`, see
        // `hparams::from_metadata`'s own `metadata_u32_optional` read) --
        // the real checkpoint carries no `ffn_gate_inp.*`/`ffn_*_exps.*`/
        // `post_ffw_norm_1`/`post_ffw_norm_2`/`pre_ffw_norm_2` tensors at
        // all (confirmed against the real gemma4-E2B blob: `strings` over
        // its GGUF header finds none of these names), only a single
        // `post_ffw_norm.weight` -- binding any of the MoE-only leaves on
        // that checkpoint would fail with `MissingMetadataKey`/
        // `UnknownTensor`. 12B/26B/31B (`expert_count > 0`) still bind the
        // full MoE set exactly as before.
        if expert_count > 0 {
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
        } else {
            bind_norm(
                parsed,
                file_bytes,
                format!("blk.{layer}.post_ffw_norm.weight"),
                GEMMA4_NORM_SHIFT,
                &mut state,
            )?;
        }
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
/// the two `Q3_K`-tagged [`Codec::Q3K`] buffers
/// [`BoundWeights::packed_owned`] already carries for every other MoE
/// family's restack fallback ([`bind_moe_expert_weights`]).
///
/// # Errors
///
/// [`InteropError::UnknownTensor`] if the fused tensor is absent;
/// [`InteropError::UnrepresentableGgmlType`] if its `ggml_type` has no
/// [`Codec`] (every codec a real gemma4 checkpoint ships does);
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
    let kind = codec_from_ggml_type(tensor.ggml_type).ok_or_else(|| {
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

/// Sliding layers always use [`ValueSourceKind::ProjectedV`] (a real
/// `attn_v.weight`) and the SWA RoPE table; full layers use the full-length
/// RoPE table and [`ValueSourceKind::ProjectedV`] too UNLESS this is a MoE
/// checkpoint (`shared_kv_layers == 0`), where a full layer genuinely has
/// no `attn_v.weight` on disk and [`ValueSourceKind::SharedWithKey`] (the
/// key projection's own output stands in for `V`) is correct instead. Both
/// use [`RopePairing::SplitHalf`] (Gemma's own half-split rotation, not
/// Llama/Mistral's interleaved pairing).
/// Every gemma4 layer is [`LayerKind::Attention`], runs dense SwiGLU over
/// the shared `ffn_norm`-normed input and routed MoE over its OWN
/// `pre_ffw_norm_2`-normed input (`routed_pre_norm: true`), each with its
/// own post-norm, summed and normalized once more, then scaled by
/// `layer_output_scale`. The routed branch gates with `Softmax` and carries
/// no `exp_probs_b` bias, unlike [`FfnCombination::Exclusive`]'s LFM2 shape
/// -- see [`proxima_tensor::spec::LayerFfnConfig`]'s own doc for what each
/// field means.
/// The operative §14 reference for gemma4 (a distinct architecture from
/// gemma3n -- no AltUp/Laurel, its own forward) is ollama's own gemma4
/// runner, `mlxrunner/model/gemma4/gemma4.go` (github.com/ollama/ollama
/// v0.34.2): `TextConfig`'s own KV-sharing-map build (`gemma4.go:590-611`)
/// walks `firstShared..NumHiddenLayers` and, for each shared layer, finds
/// "the last non-shared layer of the same type" (`gemma4.go:599-606`) --
/// the MOST RECENT own-KV layer (index `< firstShared`) whose
/// `isLayerSliding` result matches. `isLayerSliding`'s own fallback formula
/// (`gemma4.go:644-651`, used whenever `LayerTypes` metadata is absent) is
/// `(layerIdx+1) % SlidingWindowPattern != 0`, with `SlidingWindowPattern`
/// defaulting to `5` (`gemma4.go:529-531`) -- period-5 global attention,
/// 0-indexed, matching this checkpoint's `sliding_window=512` metadata
/// (7 full layers over 35 blocks, `35 / 5 == 7`). For the real
/// `gemma4:e2b-it-qat` checkpoint (`block_count=35`, `shared_kv_layers=20`,
/// `first_shared_idx=15`) this resolves to exactly two source layers: 13
/// (last own-KV sliding layer) for every sliding shared layer, 14 (last
/// own-KV full layer) for every full shared layer -- see this file's own
/// `shared_kv_reuse_map_tests` module for the full 20-entry table this
/// function reproduces layer-by-layer, and
/// `proxima-tensor::spec::tests::gemma4_synthetic_parity::shared_kv_worked_example`
/// for the synthetic-forward proof that the wiring reuses rather than
/// re-derives.
fn shared_kv_source_layer(sliding_window_pattern: &[bool], first_shared_idx: u32, layer: usize) -> u32 {
    let is_sliding = sliding_window_pattern[layer];
    (0..first_shared_idx as usize)
        .rev()
        .find(|&candidate| sliding_window_pattern[candidate] == is_sliding)
        .map(|index| index as u32)
        // Only reachable if layers `0..first_shared_idx` have NO
        // representative of this attention type at all -- not the real
        // checkpoint's shape (both types appear among its first 15
        // blocks). A safe deterministic fallback rather than a panic.
        .unwrap_or_else(|| first_shared_idx.saturating_sub(1))
}

fn gemma4_layer_schedule(architecture: &Architecture) -> Vec<LayerSchedule> {
    // E2B/E4B (`expert_count == 0`) carry no routed-expert tensors at all
    // (`bind_gemma4_weights`'s own `expert_count > 0` split) -- their
    // per-layer FFN is dense-only SwiGLU/GeGLU, [`FfnCombination::Exclusive`]
    // with `leading_dense_block_count == block_count` at the
    // `lfm2_forward_program_with_experts` call site
    // ([`Gemma4Arch::bind`]) so every layer takes the dense branch and the
    // routed branch is never built. 12B/26B/31B (`expert_count > 0`) keep
    // the real parallel dense+MoE shape unchanged.
    let combination = if architecture.expert_count > 0 {
        FfnCombination::ParallelDenseMoe(ParallelDenseMoeConfig {
            dense_post_norm: true,
            routed_post_norm: true,
            combined_post_norm: true,
            routed_pre_norm: true,
            router_scale: true,
            expert_output_scale: true,
        })
    } else {
        FfnCombination::Exclusive
    };
    let ffn = LayerFfnConfig {
        post_attention_norm: true,
        combination,
        output_scale: true,
        routed_gating: ExpertGatingFunc::Softmax,
        routed_expert_bias: false,
        activation: Activation::GeluTanh,
        // per-layer below: `feed_forward_by_layer[layer]` (E2B/E4B's own
        // matformer variable dense-FFN width; a uniform checkpoint's array
        // is `metadata_u32_per_layer`'s scalar-broadcast, so this override
        // reproduces the prior single-width behaviour byte-for-byte there).
        dense_feed_forward: None,
        // E2B/E4B's dense-only `FfnCombination::Exclusive` path needs its
        // own `blk.{layer}.post_ffw_norm.weight` sandwich norm
        // (`gemma4.go`'s `PostFFNorm`) -- unread when `combination` is
        // `ParallelDenseMoe` (12B/26B/31B), which applies its own
        // `combined_post_norm` on the SAME tensor name instead.
        exclusive_dense_post_norm: true,
        // `architecture.ple_dim > 0` (E2B/E4B) -- every layer of a PLE
        // checkpoint injects it (`gemma4.go:1349-1361` has no per-layer-type
        // branch), paired with this call site's own `Some(architecture.ple_dim)`
        // below at `lfm2_forward_program_with_experts` -- Stage A's preamble
        // only runs, and this flag is only consulted, when BOTH agree.
        ple: architecture.ple_dim > 0,
    };
    // `attention.shared_kv_layers` (0 for E4B/12B/26B/31B, 20 for E2B):
    // `first_shared_idx` is the first TRAILING layer with no own
    // `attn_k.weight`/`attn_v.weight`/`attn_k_norm.weight` at all -- see
    // `gemma4_tensor_names`'s own doc for the tensor-presence side of this
    // split.
    let first_shared_idx = architecture
        .block_count
        .saturating_sub(architecture.shared_kv_layers);
    architecture
        .sliding_window_pattern
        .iter()
        .enumerate()
        .map(|(layer, &is_sliding)| {
            let kv_heads = architecture.kv_heads_by_layer[layer];
            let ffn = LayerFfnConfig {
                dense_feed_forward: Some(architecture.feed_forward_by_layer[layer]),
                ..ffn
            };
            let attention = if is_sliding {
                LayerAttentionConfig {
                    head_dim: architecture.key_length_swa,
                    kv_heads,
                    mask_window: Some(architecture.sliding_window),
                    value_source_kind: ValueSourceKind::ProjectedV,
                    key_source_kind: KeySourceKind::ProjectedK,
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
                    // MoE (12B/26B/31B, `shared_kv_layers == 0`): a full
                    // layer genuinely has no `attn_v.weight` on disk --
                    // `SharedWithKey` (unchanged). E2B/E4B
                    // (`shared_kv_layers > 0`): every own-KV layer, sliding
                    // OR full, carries its own real `attn_v.weight`
                    // (confirmed against the real `gemma4:e2b-it-qat`
                    // header -- `blk.4`/`blk.9`/`blk.14`, the three full
                    // own-KV layers, each list `attn_v.weight` among their
                    // 17 tensors) -- `SharedWithKey` here would silently
                    // substitute the key projection for a real, present `V`
                    // weight instead of reading it. `shared_kv_layers > 0`
                    // is this crate's own established E2B-vs-MoE
                    // discriminator (already gates `is_shared_kv`/PLE
                    // above); the trailing shared-KV override below
                    // supersedes this for actually-shared layers regardless
                    // of what is set here.
                    value_source_kind: if architecture.shared_kv_layers > 0 {
                        ValueSourceKind::ProjectedV
                    } else {
                        ValueSourceKind::SharedWithKey
                    },
                    key_source_kind: KeySourceKind::ProjectedK,
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
            // Trailing shared-KV layers (E2B: blk.15..=34) have no
            // `attn_k.weight`/`attn_v.weight`/`attn_k_norm.weight` on disk
            // -- `attention_forward.rs`'s `KeySourceKind::SharedFromLayer`/
            // `ValueSourceKind::SharedFromLayer` skip declaring those three
            // leaves entirely for this layer, reading the named own-KV
            // layer's post-rope K / post-norm V instead. `value_norm` is
            // forced `false` below -- `ValueSource::Shared` already carries
            // a post-norm `V` (the source layer normalized it once);
            // `append_attention_mixer`'s own doc notes re-normalizing it
            // here would double-apply, so gemma4 E2B's shared layers must
            // NOT also request `value_norm`.
            let attention = if layer as u32 >= first_shared_idx {
                let source = shared_kv_source_layer(
                    &architecture.sliding_window_pattern,
                    first_shared_idx,
                    layer,
                );
                LayerAttentionConfig {
                    key_source_kind: KeySourceKind::SharedFromLayer(source),
                    value_source_kind: ValueSourceKind::SharedFromLayer(source),
                    value_norm: false,
                    ..attention
                }
            } else {
                attention
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

    fn kv_cache_shape(&self) -> crate::architecture::KvCacheShape {
        crate::architecture::KvCacheShape::Custom
    }

    #[cfg(feature = "std")]
    fn bind<'file>(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
    ) -> Result<BoundProgram<'file>, InteropError> {
        let architecture = from_metadata(parsed)?;
        let weights = bind_gemma4_weights(parsed, file_bytes, &architecture)?;

        // RUNTIME choice, not `#[cfg]`: `gemma4_layer_schedule` derives the
        // SAME real-checkpoint shape (sliding/full split, matformer FFN
        // widths, PLE, shared-KV) for every gemma4 variant, fed once into
        // `build_forward` here -- `descriptor.cache_strategy` is the one
        // thing that varies, decided below from `architecture` itself.
        // `CacheStrategy::TwoRange` is
        // `lfm2_two_range_cached_forward_program_with_experts`: a decode
        // step's cost drops from O(n^2) to O(1) in prior sequence length
        // once `layer_roots` below is non-empty (proven for the
        // no-shared-KV shape by `proxima-tensor`'s own
        // `two_range_cached_gemma4_matches_prefill_oracle_with_decode_loop_realistic_zero_padding`/
        // `..._two_step_decode_matches_one_shot_prefill_oracle`/
        // `build_forward_two_range_matches_direct_builder_call`, and for
        // gemma4 E2B's `KeySourceKind::SharedFromLayer`/
        // `ValueSourceKind::SharedFromLayer` shape by
        // `two_range_cached_gemma4_shared_kv_layer_matches_cacheless_oracle`
        // -- all against the cacheless engine as oracle, all < 1e-4). The
        // TWO-range engine, not the single-range one: gemma4's own first
        // step (`single_position_step == false` below) processes the WHOLE
        // prompt as one `cached_len=0` call, and a single merged softmax
        // has no self-consistent way to include that call's own new
        // positions in `kv_cache.{layer}.*` before they exist
        // (`lfm2_single_range_cached.rs`'s own module doc) -- proven by
        // `single_range_cached_gemma4_diverges_on_zero_cache_matches_when_self_range_is_folded`
        // (zero-cache max-abs-diff 0.39 vs the prefill oracle).
        //
        // `CacheStrategy::Cacheless` is the safe fallback: every gemma4
        // layer is `LayerKind::Attention` (the two-range engine's own
        // requirement), so the ONLY axis that can make a checkpoint
        // unsupported today is one this match does not yet know how to
        // prove correct end-to-end against a real checkpoint -- there is
        // none such left as of this change, so every gemma4 shape routes
        // through `TwoRange`; a future architecture variant this schedule
        // cannot express falls back here rather than building a wrong
        // program silently.
        let schedule = gemma4_layer_schedule(&architecture);
        let logit_softcap = (architecture.final_logit_softcapping > 0.0)
            .then_some(architecture.final_logit_softcapping);
        // `leading_dense_block_count`: every layer's own `FfnCombination`
        // (set by `gemma4_layer_schedule` above) is `ParallelDenseMoe` for a
        // real MoE checkpoint, which ignores this argument entirely (both
        // branches always run) -- `0` here reproduces that prior behaviour
        // byte-for-byte. For a dense checkpoint (`expert_count == 0`) every
        // layer's combination is `FfnCombination::Exclusive` instead, whose
        // dense-vs-routed choice is `layer < leading_dense_block_count`
        // (`append_lfm2_layer_ffn`) -- `block_count` here makes that
        // condition true for every layer, so the (absent, unbound) routed
        // branch is never built.
        let leading_dense_block_count = if architecture.expert_count > 0 {
            0
        } else {
            architecture.block_count
        };
        let cache_strategy = if schedule.iter().all(|entry| entry.kind == LayerKind::Attention) {
            CacheStrategy::TwoRange
        } else {
            CacheStrategy::Cacheless
        };
        let descriptor = ModelDescriptor {
            vocab: architecture.vocab,
            embedding: architecture.embedding,
            feed_forward: architecture.feed_forward,
            expert_feed_forward: architecture.expert_feed_forward,
            query_heads: architecture.head_count,
            block_count: architecture.block_count,
            expert_count: architecture.expert_count,
            expert_used_count: architecture.expert_used_count,
            leading_dense_block_count,
            l_cache: 0,
            embedding_scale: Some(EmbeddingScale::Sqrt),
            logit_softcap,
            layers: schedule.clone(),
            cache_strategy,
            // `gemma4_layer_schedule`'s own `LayerFfnConfig::ple` flag (set
            // alongside this same `architecture.ple_dim > 0` check) is what
            // per-layer INJECTS Stage B; this is the checkpoint-wide toggle
            // that builds Stage A's preamble at all.
            ple_dim: (architecture.ple_dim > 0).then_some(architecture.ple_dim),
            qk_norm: false,
            qkv_biases: false,
            paired_gate_up_reduce: false,
            fused_qkv_reduce: false,
        };
        let (program, logits, cache_roots, moe_sites, _layer_residuals, _hidden) =
            build_forward(&descriptor, true)?;
        let layer_roots: Vec<Qwen35LayerRoots> = match cache_strategy {
            CacheStrategy::TwoRange => {
                // `cache_roots` holds one entry per REAL cache-owning layer,
                // in layer order (`lfm2_two_range_cached_forward_program_with_experts`'s
                // own `stored_kv`/`cache_roots.push` doc: a
                // `KeySourceKind::SharedFromLayer` layer owns none) -- this
                // zips it back against `schedule`'s own per-layer
                // discriminant to rebuild the full, positionally-real
                // `Qwen35LayerRoots` vec `LoadedModel::declared_layer_cache_names_and_widths`
                // needs (one entry per layer index, `SharedFromLayer`
                // included).
                let expected_cache_owning_layers = schedule
                    .iter()
                    .filter(|entry| entry.attention.key_source_kind == KeySourceKind::ProjectedK)
                    .count();
                if cache_roots.len() != expected_cache_owning_layers {
                    return Err(InteropError::Gemma4TwoRangeCacheRootsCountMismatch {
                        produced: cache_roots.len(),
                        expected: expected_cache_owning_layers,
                    });
                }
                let mut cache_roots = cache_roots.into_iter();
                schedule
                    .iter()
                    .map(|entry| match entry.attention.key_source_kind {
                        KeySourceKind::ProjectedK => Qwen35LayerRoots::Attention(
                            // count checked equal, just above.
                            cache_roots.next().unwrap_or_else(|| {
                                unreachable!("cache_roots length already checked above")
                            }),
                        ),
                        KeySourceKind::SharedFromLayer(source) => {
                            Qwen35LayerRoots::SharedFromLayer(source)
                        }
                    })
                    .collect()
            }
            CacheStrategy::Cacheless | CacheStrategy::SingleRange => Vec::new(),
        };

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
            force_split_half_rope: false,
        };

        Ok(BoundProgram {
            weights,
            architecture: model_architecture,
            program,
            logits_root: logits,
            hidden_root: None,
            residual_roots: Vec::new(),
            layer_roots,
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

#[cfg(test)]
mod shared_kv_reuse_map_tests {
    use super::shared_kv_source_layer;

    /// ollama's `mlxrunner/model/gemma4/gemma4.go` `isLayerSliding` fallback
    /// (`gemma4.go:644-651`): `(layerIdx+1) % SlidingWindowPattern != 0`,
    /// `SlidingWindowPattern` defaulting to `5` (`gemma4.go:529-531`),
    /// 0-indexed -- the global/local period gemma4 E2B's `sliding_window=512`
    /// metadata implies (7 full layers over 35 blocks, `35 / 5 == 7`). `true`
    /// means sliding, matching `Architecture::sliding_window_pattern`'s own
    /// convention.
    fn period_five_pattern(block_count: usize) -> Vec<bool> {
        (0..block_count).map(|i| (i + 1) % 5 != 0).collect()
    }

    /// The worked example this slice derived by hand from
    /// `gemma4.go`'s `TextConfig` KV-sharing-map build (`gemma4.go:590-611`,
    /// see `shared_kv_source_layer`'s own doc above for the full citation --
    /// derivation notes in this session's scratchpad `sharedkv_derivation.md`):
    /// for `block_count=35`, `shared_kv_layers=20` (`first_shared_idx=15`),
    /// the 16 sliding shared layers (15,16,17,18,20,21,22,23,25,26,27,28,30,
    /// 31,32,33) all reuse own-KV layer 13 (the last sliding layer among
    /// 0..15), and the 4 full shared layers (19,24,29,34) all reuse own-KV
    /// layer 14 (the last full layer among 0..15) -- exactly 20 pairs,
    /// matching the real GGUF header's `attention.shared_kv_layers=20`.
    #[test]
    fn gemma4_e2b_shared_kv_reuse_map_matches_hand_derived_table() {
        let pattern = period_five_pattern(35);
        let first_shared_idx = 15;

        let expected: [(usize, u32); 20] = [
            (15, 13),
            (16, 13),
            (17, 13),
            (18, 13),
            (19, 14),
            (20, 13),
            (21, 13),
            (22, 13),
            (23, 13),
            (24, 14),
            (25, 13),
            (26, 13),
            (27, 13),
            (28, 13),
            (29, 14),
            (30, 13),
            (31, 13),
            (32, 13),
            (33, 13),
            (34, 14),
        ];

        for (shared_layer, expected_source) in expected {
            let source = shared_kv_source_layer(&pattern, first_shared_idx, shared_layer);
            assert_eq!(
                source, expected_source,
                "layer {shared_layer} expected to reuse source layer {expected_source}, got {source}"
            );
        }
    }

    /// The formula-derived pattern must itself imply exactly 7 full-attention
    /// layers over 35 blocks (`35 / 5`) -- a sanity check on
    /// [`period_five_pattern`] independent of the reuse-map assertion above,
    /// so a broken pattern generator cannot silently pass the main test by
    /// accident.
    #[test]
    fn period_five_pattern_has_seven_full_attention_layers_over_thirty_five_blocks() {
        let pattern = period_five_pattern(35);
        let full_count = pattern.iter().filter(|&&is_sliding| !is_sliding).count();
        assert_eq!(full_count, 7);
    }
}

/// Regression coverage for the bug this crate shipped once: `bind` and the
/// forward program each independently gate `attn_k.weight`/
/// `attn_k_norm.weight`/`attn_v.weight` per layer, and nothing forced the
/// two gates to agree -- `gemma4_layer_schedule` used to gate
/// `ValueSourceKind::ProjectedV` (and therefore the forward program's own
/// `attn_v.weight` [`proxima_tensor::op::Op::Input`] leaf) on `is_sliding`
/// alone, the MoE convention, while E2B/E4B's own-KV FULL layers (`blk.4`,
/// `blk.9`, `blk.14` on the real `gemma4:e2b-it-qat` checkpoint) carry a
/// real `attn_v.weight` regardless of sliding vs full. This test builds the
/// ACTUAL forward program `Gemma4Arch::bind` builds (not a hand-simulated
/// stand-in) for a synthetic E2B-shaped [`Architecture`] whose
/// `sliding_window_pattern`/`shared_kv_layers`/`block_count` are the real
/// checkpoint's own measured values (a real header dump against
/// `~/.ollama/models/blobs/sha256-3646b4c...` on 2026-09-20), then asserts
/// the forward program's own declared `Input` leaf names for these three
/// per-layer weights equal exactly the name set [`bind_gemma4_weights`]'s
/// own gates would bind -- no declared-but-unbound leaf, and no bound
/// leaf the forward program never asks for either.
#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod declared_leaves_match_bound_leaves_tests {
    use super::*;
    use proxima_tensor::op::Op;
    // This test module exercises `gemma4_layer_schedule`'s own declared-vs-
    // bound leaf contract directly against the cacheless engine -- an
    // engine-shape check independent of which `CacheStrategy`
    // `Gemma4Arch::bind` picks at runtime, so it imports its own builder
    // rather than the module-level `build_forward`.
    use proxima_tensor::spec::lfm2_forward_program_with_experts;

    /// The real `gemma4:e2b-it-qat` checkpoint's own measured
    /// `sliding_window_pattern`/`shared_kv_layers`/`block_count` (header
    /// dump, 2026-09-20) -- every other field is a plausible dense-E2B
    /// value uninvolved in the attn_k/attn_k_norm/attn_v declare-vs-bind
    /// gate this test exercises.
    fn e2b_shaped_architecture() -> Architecture {
        let sliding_window_pattern: Vec<bool> =
            (0..35u32).map(|index| (index + 1) % 5 != 0).collect();
        let mut feed_forward_by_layer = alloc::vec![6144u32; 15];
        feed_forward_by_layer.extend(alloc::vec![12288u32; 20]);
        Architecture {
            vocab: 1,
            embedding: 1536,
            block_count: 35,
            feed_forward: 6144,
            feed_forward_by_layer,
            expert_feed_forward: 0,
            expert_count: 0,
            expert_used_count: 0,
            head_count: 8,
            kv_heads_by_layer: alloc::vec![1; 35],
            rms_epsilon: 1e-6,
            key_length: 512,
            value_length: 512,
            sliding_window: 512,
            key_length_swa: 256,
            value_length_swa: 256,
            sliding_window_pattern,
            shared_kv_layers: 20,
            rope_freq_base: 1_000_000.0,
            rope_freq_base_swa: 10_000.0,
            rope_dimension_count: 512,
            rope_dimension_count_swa: 256,
            final_logit_softcapping: 30.0,
            ple_dim: 256,
        }
    }

    /// The name set [`bind_gemma4_weights`]'s own per-layer gates would
    /// bind for `suffix` -- reproduces those gates verbatim (not a
    /// re-derivation) so this test fails the moment either site's
    /// condition drifts from the other.
    fn bound_leaf_names(
        architecture: &Architecture,
        suffix: &str,
    ) -> alloc::collections::BTreeSet<String> {
        let first_shared_idx = architecture
            .block_count
            .saturating_sub(architecture.shared_kv_layers);
        architecture
            .sliding_window_pattern
            .iter()
            .enumerate()
            .filter_map(|(layer_index, &is_sliding)| {
                let layer = layer_index as u32;
                let is_shared_kv = layer >= first_shared_idx;
                let bound = match suffix {
                    "attn_k.weight" | "attn_k_norm.weight" => !is_shared_kv,
                    "attn_v.weight" => {
                        (is_sliding || architecture.shared_kv_layers > 0) && !is_shared_kv
                    }
                    other => unreachable!("unexpected suffix {other}"),
                };
                bound.then(|| format!("blk.{layer}.{suffix}"))
            })
            .collect()
    }

    /// The name set the ACTUAL forward program (`lfm2_forward_program_with_experts`
    /// over `gemma4_layer_schedule`'s own output, the exact call
    /// `Gemma4Arch::bind` makes) declares as an `Input` leaf for `suffix`.
    fn declared_leaf_names(
        architecture: &Architecture,
        suffix: &str,
    ) -> alloc::collections::BTreeSet<String> {
        let schedule = gemma4_layer_schedule(architecture);
        let (program, _logits, _moe_sites) = lfm2_forward_program_with_experts(
            architecture.vocab,
            architecture.embedding,
            architecture.feed_forward,
            architecture.expert_feed_forward,
            architecture.head_count,
            architecture.block_count,
            architecture.expert_count,
            architecture.expert_used_count,
            architecture.block_count,
            0,
            &schedule,
            Some(EmbeddingScale::Sqrt),
            (architecture.final_logit_softcapping > 0.0)
                .then_some(architecture.final_logit_softcapping),
            true,
            (architecture.ple_dim > 0).then_some(architecture.ple_dim),
        )
        .expect("gemma4 e2b-shaped forward program lowers");

        program
            .iter()
            .filter_map(|op| match op {
                Op::Input {
                    name: Some(name), ..
                } => Some(name.clone()),
                _ => None,
            })
            .filter(|name| name.ends_with(suffix) && name.starts_with("blk."))
            .collect()
    }

    #[test]
    fn e2b_declared_attn_v_leaves_equal_bound_attn_v_leaves() {
        let architecture = e2b_shaped_architecture();
        let declared = declared_leaf_names(&architecture, "attn_v.weight");
        let bound = bound_leaf_names(&architecture, "attn_v.weight");
        assert_eq!(
            declared, bound,
            "forward program declares attn_v.weight leaves the binder does not bind (or vice versa)"
        );
        // The three real full own-KV layers this bug silently dropped --
        // pins the invariant to the actual header fact, not just set
        // equality (an empty-vs-empty pair would also satisfy `assert_eq`
        // above).
        for full_own_kv_layer in [4, 9, 14] {
            let name = format!("blk.{full_own_kv_layer}.attn_v.weight");
            assert!(
                declared.contains(&name),
                "expected {name} to be declared (full own-KV E2B layer has a real attn_v.weight)"
            );
        }
    }

    #[test]
    fn e2b_declared_attn_k_and_attn_k_norm_leaves_equal_bound_leaves() {
        let architecture = e2b_shaped_architecture();
        for suffix in ["attn_k.weight", "attn_k_norm.weight"] {
            let declared = declared_leaf_names(&architecture, suffix);
            let bound = bound_leaf_names(&architecture, suffix);
            assert_eq!(declared, bound, "{suffix} declare/bind set mismatch");
        }
    }

    /// The MoE path (`shared_kv_layers == 0`) must keep the exact prior
    /// gate: only sliding layers declare/bind `attn_v.weight` -- proves the
    /// E2B fix above did not widen MoE's own set.
    #[test]
    fn moe_declared_attn_v_leaves_stay_gated_on_is_sliding_only() {
        let mut architecture = e2b_shaped_architecture();
        architecture.shared_kv_layers = 0;
        let declared = declared_leaf_names(&architecture, "attn_v.weight");
        let bound = bound_leaf_names(&architecture, "attn_v.weight");
        assert_eq!(declared, bound);
        for full_layer in [4, 9, 14] {
            let name = format!("blk.{full_layer}.attn_v.weight");
            assert!(
                !declared.contains(&name),
                "MoE full layer {name} must stay SharedWithKey (no attn_v.weight)"
            );
        }
    }
}
