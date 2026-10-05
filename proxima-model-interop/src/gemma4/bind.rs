//! Weight binding, tensor-name enumeration, and [`crate::architecture::Architecture`]
//! registration for the `gemma4` checkpoint family. `Gemma4Arch::bind` is a
//! DESCRIPTOR consumer: it hands the parsed header to
//! [`proxima_tensor::spec::gemma4_descriptor_from_gguf`], which builds the
//! whole [`proxima_tensor::spec::ModelDescriptor`] (per-layer
//! [`proxima_tensor::spec::LayerAttentionConfig`]/
//! [`proxima_tensor::spec::LayerFfnConfig`] schedule included), and lowers it
//! through [`proxima_tensor::spec::build_forward`] -- there is no bespoke
//! gemma4 forward-graph builder and no gemma4 schedule in this crate.
//! Teaching pointer: read `proxima_tensor::spec::attention_forward`'s own doc
//! on `lfm2_forward_program_with_experts` before touching this file -- every
//! knob the descriptor sets is documented there, not here.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;
use proxima_tensor::spec::{
    CacheStrategy, CachedLayerRoots, KeySourceKind, ModelDescriptor, Qwen35LayerRoots,
    build_forward,
    gemma4_descriptor_from_gguf,
};
use crate::architecture::{
    Architecture as ArchitectureTrait, BoundProgram, KvLayout, StepInput, StepInputContext,
};
use crate::bind::{
    BoundWeights, ModelArchitecture, SlidingRope, bind_dense, bind_matmul_weight,
    bind_matmul_weight_as, bind_matmul_weight_transposed_f32, bind_moe_expert_weights,
    codec_from_ggml_type, find_tensor, gguf_tensor_as_f32, metadata_str,
};
use crate::error::InteropError;
use crate::profiles::family_profile;

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
/// `attn_v.weight` at all -- see `gemma4_descriptor_from_gguf`'s own
/// `shared_kv_source_layer` for which own-KV layer supplies them instead.
/// The eight routed-expert leaves (`ffn_down_exps.{scale,weight}`,
/// `ffn_gate_inp.{scale,weight}`, `ffn_gate_up_exps.weight`,
/// `post_ffw_norm_1.weight`, `post_ffw_norm_2.weight`,
/// `pre_ffw_norm_2.weight`) are gated on `architecture.expert_count > 0`,
/// mirroring `bind_gemma4_weights`'s own split -- a dense checkpoint
/// (E2B/E4B) carries none of them on disk, only the single
/// `post_ffw_norm.weight` every layer lists unconditionally.
#[must_use]
pub fn gemma4_tensor_names(architecture: &Architecture) -> Vec<String> {
    let mut names = Vec::new();
    let first_shared_idx = architecture
        .block_count
        .saturating_sub(architecture.shared_kv_layers);

    let is_moe = architecture.expert_count > 0;
    for (layer, &is_sliding) in architecture.sliding_window_pattern.iter().enumerate() {
        let is_shared_kv = layer as u32 >= first_shared_idx;
        let mut suffixes = alloc::vec![
            "attn_norm.weight",
            "attn_output.weight",
            "attn_q.weight",
            "attn_q_norm.weight",
            "ffn_down.weight",
            "ffn_gate.weight",
            "ffn_norm.weight",
            "ffn_up.weight",
            "layer_output_scale.weight",
            "post_attention_norm.weight",
            "post_ffw_norm.weight",
        ];
        // `bind_gemma4_weights`'s own `expert_count > 0` split (this file's
        // own doc above it): these eight leaves exist on disk ONLY for a
        // routed-expert checkpoint (12B/26B/31B) -- a dense checkpoint
        // (E2B/E4B, `expert_count == 0`) carries none of them, only the
        // single `post_ffw_norm.weight` already listed above.
        if is_moe {
            suffixes.push("ffn_down_exps.scale");
            suffixes.push("ffn_down_exps.weight");
            suffixes.push("ffn_gate_inp.scale");
            suffixes.push("ffn_gate_inp.weight");
            suffixes.push("ffn_gate_up_exps.weight");
            suffixes.push("post_ffw_norm_1.weight");
            suffixes.push("post_ffw_norm_2.weight");
            suffixes.push("pre_ffw_norm_2.weight");
        }
        if !is_shared_kv {
            suffixes.push("attn_k.weight");
            suffixes.push("attn_k_norm.weight");
            // Mirrors `bind_gemma4_weights`'s own `attn_v.weight` bind gate
            // and `gemma4_descriptor_from_gguf`'s own `value_source_kind` gate --
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

/// Binds every weight [`proxima_tensor::spec::lfm2_forward_program_with_experts`]'s `Input` leaves
/// declare for gemma4's own [`Gemma4Arch::bind`] descriptor.
/// `blk.{layer}.pre_ffw_norm_2.weight` is bound (via `bind_norm` with
/// `GEMMA4_NORM_SHIFT`) and consumed by the engine's
/// `routed_pre_norm` knob (`gemma4_descriptor_from_gguf` sets it), which normalizes
/// the routed branch's input separately from the dense branch's shared
/// `ffn_norm`-normed one -- matching the real Gemma 4 graph.
///
/// The fused `blk.{layer}.ffn_gate_up_exps.weight` splits into the two
/// separate `ffn_gate_exps.weight`/`ffn_up_exps.weight` leaves the engine's
/// routed FFN declares WITHOUT a dequant -- see
/// `bind_gemma4_fused_gate_up_experts`'s own doc for the confirmed axis
/// (the real checkpoint's `ne[0]`=2816 embedding is the quantization block
/// axis; `ne[1]`=1408=2*`expert_feed_forward` is the row axis the split
/// cuts, orthogonal to blocks) and the packed-memcpy implementation.
///
/// # Errors
///
/// Whatever [`find_tensor`]/[`gguf_tensor_as_f32`]/[`bind_dense`]/
/// [`bind_matmul_weight`]/`bind_gemma4_fused_gate_up_experts`/
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
        // `gemma4_descriptor_from_gguf`'s own `KeySourceKind::SharedFromLayer`/
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
        // Mirrors `gemma4_descriptor_from_gguf`'s own `value_source_kind` gate --
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

/// The checkpoint's descriptor: [`gemma4_descriptor_from_gguf`] over the family
/// profile `general.architecture` names, so the values GGUF does not carry come
/// from `crate::profiles` and never from this module.
#[cfg(feature = "std")]
pub fn descriptor_from_gguf(
    parsed: &ParsedGguf,
    sliding_kv_ring: bool,
) -> Result<ModelDescriptor, InteropError> {
    let profile = family_profile(metadata_str(parsed, "general.architecture")?)?;
    Ok(gemma4_descriptor_from_gguf(parsed, sliding_kv_ring, &profile)?)
}

fn rebuild_layer_roots(
    key_sources: impl Iterator<Item = KeySourceKind> + Clone,
    cache_roots: Vec<CachedLayerRoots>,
) -> Result<Vec<Qwen35LayerRoots>, InteropError> {
    let expected = key_sources.clone().filter(|kind| *kind == KeySourceKind::ProjectedK).count();
    let produced = cache_roots.len();
    let count_mismatch = || InteropError::Gemma4TwoRangeCacheRootsCountMismatch { produced, expected };
    let mut cache_roots = cache_roots.into_iter();
    let layer_roots = key_sources
        .map(|kind| match kind {
            KeySourceKind::ProjectedK => {
                cache_roots.next().map(Qwen35LayerRoots::Attention).ok_or_else(count_mismatch)
            }
            KeySourceKind::SharedFromLayer(source) => Ok(Qwen35LayerRoots::SharedFromLayer(source)),
        })
        .collect::<Result<Vec<_>, _>>()?;
    match cache_roots.next() {
        Some(_) => Err(count_mismatch()),
        None => Ok(layer_roots),
    }
}

/// `PROXIMA_HEAD_REPEATS=1|2|3` (unset or unparsable reads as `1`): the
/// head-cost measurement knob, layered into [`proxima_tensor::spec::ModelDescriptor::head_repeats`]
/// here so the op-graph builders stay pure functions of the descriptor.
#[cfg(feature = "instrument")]
fn head_repeats_from_env() -> u32 {
    std::env::var("PROXIMA_HEAD_REPEATS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1)
}

/// [`Gemma4Arch::bind`]'s body, parameterized on `last_row_only`
/// (`lfm2_two_range_cached_forward_program_with_experts`'s own trailing
/// flag -- see its doc: `true` gathers the LM head to the last new
/// position only, `false` leaves every new position's own row in
/// `logits_root`, `[new_count, vocab]`). `Gemma4Arch::bind` above calls
/// this with `true` unchanged, so the registered decode/prefill path is
/// byte-for-byte what it was before this function existed.
/// [`bind_gemma4_all_positions_logits`] is the only other caller, with
/// `false` -- the additive SLICE 2a readout a speculative-decode verify
/// step needs (per-position logits for every candidate token from ONE
/// forward, not just the sampled last position).
#[cfg(feature = "std")]
fn bind_gemma4_with_last_row_only<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    last_row_only: bool,
    layout: KvLayout,
) -> Result<BoundProgram<'file>, InteropError> {
    let architecture = from_metadata(parsed)?;
    let weights = bind_gemma4_weights(parsed, file_bytes, &architecture)?;

    // every gemma4 shape routes through `TwoRange` (all layers are `LayerKind::Attention`);
    // the two-range engine, not single-range, because the first step processes the whole
    // prompt as one `cached_len=0` call (`lfm2_single_range_cached.rs`'s own module doc)
    let descriptor = ModelDescriptor {
        last_row_only,
        ..descriptor_from_gguf(parsed, layout == KvLayout::SlidingRing)?
    };
    #[cfg(feature = "instrument")]
    let descriptor = ModelDescriptor {
        head_repeats: head_repeats_from_env(),
        ..descriptor
    };
    let schedule = &descriptor.layers;
    let cache_strategy = descriptor.cache_strategy;
        let (program, logits, cache_roots, moe_sites, _layer_residuals, _hidden, duplicate_head_roots) =
            build_forward(&descriptor)?;
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
                rebuild_layer_roots(
                    schedule.iter().map(|entry| entry.attention.key_source_kind),
                    cache_roots,
                )?
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
            family: metadata_str(parsed, "general.architecture")?.into(),
            sliding_rope: Some(SlidingRope {
                freq_base: architecture.rope_freq_base_swa,
                dimension_count: architecture.rope_dimension_count_swa,
            }),
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
            duplicate_head_roots,
            single_position_step: false,
        })
}

/// SLICE 2a verify readout: the additive, opt-in counterpart to
/// [`Gemma4Arch::bind`] (which always gathers `logits_root` to the last new
/// position -- `bind_gemma4_with_last_row_only`'s own doc). Binds the exact
/// same weights/program shape with `last_row_only: false`, so `logits_root`
/// evaluates to every new position's own row, `[new_count, vocab]`, not
/// just the last. A speculative-decode verify step feeds this a K-token
/// candidate span (`context.new_start = cached_len`, one forward call) and
/// reads back K rows of logits instead of K separate single-position
/// decode steps -- `Gemma4Arch::bind`'s own program, decode loop, and
/// `logits_root` shape are untouched by this function existing.
#[cfg(feature = "std")]
pub fn bind_gemma4_all_positions_logits<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
) -> Result<BoundProgram<'file>, InteropError> {
    bind_gemma4_with_last_row_only(parsed, file_bytes, false, KvLayout::Full)
}

impl ArchitectureTrait for Gemma4Arch {
    fn name(&self) -> &'static str {
        "gemma4"
    }

    fn kv_cache_shape(&self) -> crate::architecture::KvCacheShape {
        crate::architecture::KvCacheShape::Custom
    }

    fn kv_layers(&self, parsed: &ParsedGguf) -> Result<Vec<(u32, u32, Option<u32>)>, InteropError> {
        super::hparams::kv_layers_from_metadata(parsed)
    }

    /// Intervention 6's measured decode configuration (RUN.md "Intervention
    /// 6" / "INTEGRATION"): whole-token latency 18.83 -> 16.60 ms (-11.9%) at
    /// K=8, bytes identical to K=1 on the six-prompt corpus, uninstrumented
    /// rollout check confirming the same order of magnitude
    /// (-1.98 ms/token). Scoped to gemma4 alone -- every other architecture
    /// keeps [`ArchitectureTrait::command_buffer_chunks`]'s own default of `1`;
    /// this measurement does not establish a universal placements-path
    /// default (the owner's own integration recommendation).
    fn command_buffer_chunks(&self) -> u32 {
        8
    }

    #[cfg(feature = "std")]
    fn bind<'file>(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
    ) -> Result<BoundProgram<'file>, InteropError> {
        bind_gemma4_with_last_row_only(parsed, file_bytes, true, KvLayout::Full)
    }

    /// [`Self::bind`] with the sliding layers' cache laid out as the ring
    /// [`KvLayout::SlidingRing`] names -- what
    /// [`crate::generate::LoadedModel::load`] binds. [`Self::bind`] itself
    /// stays the full-cache layout, so every caller that drives its own
    /// `kv_cache.*` leaves against it is unchanged.
    #[cfg(feature = "std")]
    fn bind_with_kv_layout<'file>(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
        layout: KvLayout,
    ) -> Result<BoundProgram<'file>, InteropError> {
        bind_gemma4_with_last_row_only(parsed, file_bytes, true, layout)
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

    /// [`bind_gemma4_all_positions_logits`]'s own trait-level entry point --
    /// this crate's only [`ArchitectureTrait::speculative_verify_program`]
    /// override (that method's own doc on why a capability method, never a
    /// `name() == "gemma4"` check, is what gates speculative decode's
    /// verify step).
    #[cfg(feature = "std")]
    fn speculative_verify_program<'file>(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
    ) -> Result<Option<BoundProgram<'file>>, InteropError> {
        bind_gemma4_all_positions_logits(parsed, file_bytes).map(Some)
    }

    #[cfg(feature = "std")]
    fn speculative_verify_program_with_kv_layout<'file>(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
        layout: KvLayout,
    ) -> Result<Option<BoundProgram<'file>>, InteropError> {
        bind_gemma4_with_last_row_only(parsed, file_bytes, false, layout).map(Some)
    }

    /// Feeds the sliding-window RoPE table the `rope_cos_swa`/`rope_sin_swa`
    /// leaves declare (`LayerAttentionConfig::rope_table`,
    /// `gemma4_descriptor_from_gguf`) -- the decode loop's builtin
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
        let Some(rope) = context.architecture.sliding_rope else {
            return;
        };
        let (cos, sin) =
            gemma4_sliding_rope_table(&positions, rope.freq_base, rope.dimension_count);
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
    /// pair's angle by -- `bind_rope_freqs` binds that tensor's own values
    /// verbatim into [`BoundWeights::owned`] at bind time, and this method
    /// hands the same slice straight back to
    /// `crate::generate::build_position_inputs`, which does the dividing.
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
mod rebuild_layer_roots_tests {
    use super::*;
    use proxima_tensor::op::NodeId;

    fn roots(seed: u32) -> CachedLayerRoots {
        (NodeId(seed), NodeId(seed + 1), NodeId(seed + 2))
    }

    #[test]
    fn interleaves_shared_layers_between_owning_layers_in_schedule_order() -> Result<(), InteropError> {
        let kinds = [
            KeySourceKind::ProjectedK,
            KeySourceKind::ProjectedK,
            KeySourceKind::SharedFromLayer(1),
            KeySourceKind::SharedFromLayer(1),
        ];

        let rebuilt = rebuild_layer_roots(kinds.into_iter(), alloc::vec![roots(10), roots(20)])?;

        assert!(matches!(rebuilt[0], Qwen35LayerRoots::Attention((NodeId(10), _, _))));
        assert!(matches!(rebuilt[1], Qwen35LayerRoots::Attention((NodeId(20), _, _))));
        assert!(matches!(rebuilt[2], Qwen35LayerRoots::SharedFromLayer(1)));
        assert_eq!(rebuilt.len(), 4);
        Ok(())
    }

    #[test]
    fn fewer_cache_roots_than_owning_layers_reports_both_counts() {
        let kinds = [KeySourceKind::ProjectedK, KeySourceKind::ProjectedK, KeySourceKind::SharedFromLayer(0)];

        let outcome = rebuild_layer_roots(kinds.into_iter(), alloc::vec![roots(10)]);

        assert!(matches!(
            outcome,
            Err(InteropError::Gemma4TwoRangeCacheRootsCountMismatch { produced: 1, expected: 2 })
        ));
    }

    #[test]
    fn more_cache_roots_than_owning_layers_reports_both_counts() {
        let kinds = [KeySourceKind::ProjectedK, KeySourceKind::SharedFromLayer(0)];

        let outcome = rebuild_layer_roots(kinds.into_iter(), alloc::vec![roots(10), roots(20)]);

        assert!(matches!(
            outcome,
            Err(InteropError::Gemma4TwoRangeCacheRootsCountMismatch { produced: 2, expected: 1 })
        ));
    }
}

/// Regression coverage for the bug this crate shipped once: `bind` and the
/// forward program each independently gate `attn_k.weight`/
/// `attn_k_norm.weight`/`attn_v.weight` per layer, and nothing forced the
/// two gates to agree -- `gemma4_descriptor_from_gguf` used to gate
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
    use arrayvec::ArrayVec;
    use proxima_gguf::types::GgmlType;
    use proxima_gguf::value::{MetadataArray, MetadataValue};
    use proxima_gguf::{GgufModel, TensorPayload, parse_complete, write_complete};
    use proxima_tensor::op::Op;

    /// The real `gemma4:e2b-it-qat` checkpoint's own measured
    /// `sliding_window_pattern`/`shared_kv_layers`/`block_count` (header
    /// dump, 2026-09-20) -- every other key is a plausible dense-E2B
    /// value uninvolved in the attn_k/attn_k_norm/attn_v declare-vs-bind
    /// gate this test exercises. Written by the real GGUF encoder and
    /// parsed back by the real decoder; `token_embd.weight` is a 1-row
    /// table so vocab resolves.
    fn e2b_shaped(shared_kv_layers: u32) -> (ParsedGguf, Architecture) {
        let mut feed_forward = alloc::vec![6144u32; 15];
        feed_forward.extend(alloc::vec![12288u32; 20]);
        let u32_array = |values: Vec<u32>| MetadataValue::Array(MetadataArray::U32(values));
        let metadata = alloc::vec![
            ("general.architecture", MetadataValue::String("gemma4".into())),
            ("gemma4.embedding_length", MetadataValue::U32(1536)),
            ("gemma4.block_count", MetadataValue::U32(35)),
            ("gemma4.attention.head_count", MetadataValue::U32(8)),
            ("gemma4.attention.head_count_kv", u32_array(alloc::vec![1; 35])),
            (
                "gemma4.attention.sliding_window_pattern",
                MetadataValue::Array(MetadataArray::Bool((0..35u32).map(|index| (index + 1) % 5 != 0).collect())),
            ),
            ("gemma4.attention.shared_kv_layers", MetadataValue::U32(shared_kv_layers)),
            ("gemma4.attention.key_length", MetadataValue::U32(512)),
            ("gemma4.attention.value_length", MetadataValue::U32(512)),
            ("gemma4.attention.key_length_swa", MetadataValue::U32(256)),
            ("gemma4.attention.value_length_swa", MetadataValue::U32(256)),
            ("gemma4.attention.sliding_window", MetadataValue::U32(512)),
            ("gemma4.attention.layer_norm_rms_epsilon", MetadataValue::F32(1e-6)),
            ("gemma4.feed_forward_length", u32_array(feed_forward)),
            ("gemma4.rope.freq_base", MetadataValue::F32(1_000_000.0)),
            ("gemma4.rope.freq_base_swa", MetadataValue::F32(10_000.0)),
            ("gemma4.rope.dimension_count", MetadataValue::U32(512)),
            ("gemma4.rope.dimension_count_swa", MetadataValue::U32(256)),
            ("gemma4.final_logit_softcapping", MetadataValue::F32(30.0)),
            ("gemma4.embedding_length_per_layer_input", MetadataValue::U32(256)),
        ];
        let table = [0u8; 1536 * 4];
        let model = GgufModel {
            version: 3,
            metadata: metadata
                .into_iter()
                .map(|(key, value)| (key.to_string(), value))
                .collect(),
            tensors: alloc::vec![TensorPayload {
                name: "token_embd.weight".to_string(),
                dims: ArrayVec::from_iter([1536u64, 1]),
                ggml_type: GgmlType::F32,
                data: &table,
            }],
        };
        let bytes = write_complete(&model).expect("the e2b-shaped model encodes");
        let parsed = parse_complete(&bytes).expect("bytes the encoder just wrote parse");
        let architecture = from_metadata(&parsed).expect("gemma4 hparams parse from the e2b-shaped header");
        (parsed, architecture)
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

    /// The name set the ACTUAL forward program `Gemma4Arch::bind` lowers --
    /// `gemma4_descriptor_from_gguf`'s own output, built through the
    /// cacheless engine so the check is independent of which
    /// `CacheStrategy` bind picks -- declares as an `Input` leaf for `suffix`.
    fn declared_leaf_names(
        parsed: &ParsedGguf,
        suffix: &str,
    ) -> alloc::collections::BTreeSet<String> {
        let mut descriptor = descriptor_from_gguf(parsed, false)
            .expect("the e2b-shaped header carries every key the descriptor reads");
        descriptor.cache_strategy = CacheStrategy::Cacheless;
        let (program, ..) = build_forward(&descriptor).expect("gemma4 e2b-shaped forward program lowers");

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
        let (parsed, architecture) = e2b_shaped(20);
        let declared = declared_leaf_names(&parsed, "attn_v.weight");
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
        let (parsed, architecture) = e2b_shaped(20);
        for suffix in ["attn_k.weight", "attn_k_norm.weight"] {
            let declared = declared_leaf_names(&parsed, suffix);
            let bound = bound_leaf_names(&architecture, suffix);
            assert_eq!(declared, bound, "{suffix} declare/bind set mismatch");
        }
    }

    /// The MoE path (`shared_kv_layers == 0`) must keep the exact prior
    /// gate: only sliding layers declare/bind `attn_v.weight` -- proves the
    /// E2B fix above did not widen MoE's own set.
    #[test]
    fn moe_declared_attn_v_leaves_stay_gated_on_is_sliding_only() {
        let (parsed, architecture) = e2b_shaped(0);
        let declared = declared_leaf_names(&parsed, "attn_v.weight");
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

    /// Hand-derived from ollama's `mlxrunner/model/gemma4/gemma4.go`
    /// `TextConfig` KV-sharing-map build (`gemma4.go:590-611`: a shared layer
    /// reuses "the last non-shared layer of the same type") for
    /// `block_count=35`, `shared_kv_layers=20` (`first_shared_idx=15`): the 16
    /// sliding shared layers all reuse own-KV layer 13, the 4 full shared
    /// layers (19, 24, 29, 34) reuse layer 14 -- exactly 20 pairs, matching
    /// the real header's `attention.shared_kv_layers=20`. Read off the
    /// production descriptor, not a private helper.
    #[test]
    fn gemma4_e2b_shared_kv_reuse_map_matches_hand_derived_table() {
        let (parsed, _) = e2b_shaped(20);
        let descriptor = descriptor_from_gguf(&parsed, false)
            .expect("the e2b-shaped header carries every key the descriptor reads");
        let shared_sources: [(usize, u32); 20] = [
            (15, 13), (16, 13), (17, 13), (18, 13), (19, 14),
            (20, 13), (21, 13), (22, 13), (23, 13), (24, 14),
            (25, 13), (26, 13), (27, 13), (28, 13), (29, 14),
            (30, 13), (31, 13), (32, 13), (33, 13), (34, 14),
        ];

        for (layer, source) in shared_sources {
            let attention = descriptor.layers[layer].attention.clone();
            assert_eq!(
                attention.key_source_kind,
                KeySourceKind::SharedFromLayer(source),
                "layer {layer} expected to read the K of layer {source}"
            );
        }
        assert!(
            descriptor.layers[..15]
                .iter()
                .all(|entry| entry.attention.key_source_kind == KeySourceKind::ProjectedK),
            "own-KV layers 0..15 project their own K"
        );
    }

    /// The header's pattern must itself imply exactly 7 full-attention
    /// layers over 35 blocks (`35 / 5`), independent of the reuse-map
    /// assertion above, so a broken pattern cannot pass it by accident.
    #[test]
    fn gemma4_e2b_header_pattern_has_seven_full_attention_layers_over_thirty_five_blocks() {
        let (parsed, _) = e2b_shaped(20);
        let descriptor = descriptor_from_gguf(&parsed, false)
            .expect("the e2b-shaped header carries every key the descriptor reads");
        let full_layers = descriptor
            .layers
            .iter()
            .filter(|entry| entry.attention.mask_window.is_none())
            .count();
        assert_eq!(full_layers, 7);
    }
}

/// Regression coverage for the bug this file shipped once:
/// [`gemma4_tensor_names`] unconditionally listed the eight routed-expert
/// leaves (`ffn_down_exps.*`, `ffn_gate_inp.*`, `ffn_gate_up_exps.weight`,
/// `post_ffw_norm_1.weight`, `post_ffw_norm_2.weight`,
/// `pre_ffw_norm_2.weight`) even on a dense checkpoint (`expert_count == 0`),
/// producing 280 phantom names with no tensor behind them on the real
/// `gemma4:e2b-it-qat` blob. Both fixtures below parse the real header
/// (`parse_complete`, never a hand-built buffer, per guiding-principle 9) and
/// assert [`gemma4_tensor_names`] equals the header's own tensor directory
/// exactly -- zero missing, zero extra, in both directions.
#[cfg(all(test, feature = "std"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod real_checkpoint_tensor_directory_tests {
    use std::collections::BTreeSet;
    use std::fs::File;

    use super::{from_metadata, gemma4_tensor_names};

    /// The real dense E2B checkpoint the bug report names: 35 blocks, no
    /// experts, `attention.shared_kv_layers=20`, `per_layer_token_embd`
    /// (PLE) present. No environment override exists for this path --
    /// `crate::test_support::require_fixture`'s own `None` branch, matching
    /// `real_mixtral_file`/`real_lfm2_hybrid_file`'s convention for a
    /// hardcoded fixture with no env var.
    const REAL_GEMMA4_E2B_GGUF_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

    fn assert_tensor_directory_matches(path: &str) -> (usize, usize) {
        let file = File::open(path).unwrap_or_else(|error| panic!("open {path}: {error}"));
        // SAFETY: `file` stays open (owned by this stack frame) for as long
        // as `mapping` is alive; the mapping is read-only and this test never
        // writes to the backing file, so no other process racing a write can
        // be observed as a data race on this side.
        let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("mmap the real checkpoint read-only");
        let file_bytes: &[u8] = &mapping;
        let parsed =
            proxima_gguf::parse_complete(file_bytes).expect("parses the real checkpoint's own GGUF header");
        let architecture =
            from_metadata(&parsed).expect("gemma4 hparams parse from the real checkpoint header");

        let computed_names: BTreeSet<String> = gemma4_tensor_names(&architecture).into_iter().collect();
        let real_names: BTreeSet<String> = parsed
            .tensors
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect();

        let missing: Vec<&String> = real_names.difference(&computed_names).collect();
        let extra: Vec<&String> = computed_names.difference(&real_names).collect();
        eprintln!(
            "listed = {} header = {} extra = {} missing = {}",
            computed_names.len(),
            real_names.len(),
            extra.len(),
            missing.len()
        );
        assert!(
            missing.is_empty() && extra.is_empty(),
            "gemma4_tensor_names must exactly match {path}'s own tensor directory: missing={missing:?} extra={extra:?}"
        );
        (computed_names.len(), real_names.len())
    }

    #[proxima::test]
    #[ignore = "requires a real, local gemma4 E2B (dense) GGUF blob at REAL_GEMMA4_E2B_GGUF_PATH"]
    async fn gemma4_tensor_names_matches_real_dense_e2b_header_with_no_moe_leaves() {
        crate::test_support::require_fixture(REAL_GEMMA4_E2B_GGUF_PATH, None);
        let (listed, header) = assert_tensor_directory_matches(REAL_GEMMA4_E2B_GGUF_PATH);
        assert_eq!(listed, header);
    }

    /// `PROXIMA_GEMMA4_MOE_GGUF` read the same way `crate::test_support`'s
    /// other `PROXIMA_*_GGUF` knobs are: unset falls back to this host-local
    /// `batiai/gemma4-26b` checkpoint (`gemma4.expert_count`/
    /// `gemma4.expert_used_count` present in its own GGUF metadata,
    /// confirmed via `strings` over the blob header on 2026-09-28), the real
    /// routed-expert sibling of the dense fixture above -- proves the MoE
    /// leaves this fix keeps gated on `expert_count > 0` are still produced,
    /// exactly, when a real MoE checkpoint's header says they should be.
    fn real_gemma4_moe_gguf_path() -> String {
        std::env::var("PROXIMA_GEMMA4_MOE_GGUF").unwrap_or_else(|_| {
            "/Users/brianbruggeman/.ollama/models/blobs/\
             sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129"
                .to_string()
        })
    }

    #[proxima::test]
    #[ignore = "requires a real, local gemma4 MoE GGUF blob; set PROXIMA_GEMMA4_MOE_GGUF"]
    async fn gemma4_tensor_names_matches_real_moe_header_with_expert_leaves_present() {
        let path = real_gemma4_moe_gguf_path();
        crate::test_support::require_fixture(&path, Some("PROXIMA_GEMMA4_MOE_GGUF"));
        let (listed, header) = assert_tensor_directory_matches(&path);
        assert_eq!(listed, header);
    }
}
