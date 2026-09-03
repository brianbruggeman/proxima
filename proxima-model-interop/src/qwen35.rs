//! Qwen3.8-27B's real hybrid checkpoint (`general.architecture = "qwen35"`):
//! [`Qwen35Architecture`] derives this architecture's own metadata shape --
//! `{architecture}.full_attention_interval` marks every `interval`th layer
//! (1-indexed) as dense attention, every other layer as a gated
//! state-space mixer -- the same "read the checkpoint's own per-layer
//! marker, don't assume the dense shape" move [`crate::lfm2`] makes for its
//! own hybrid checkpoint, just from a scalar interval instead of a
//! per-layer array.
//!
//! [`bind_qwen35_weights`] picks which fixed tensor set each layer binds by
//! its own [`Qwen35LayerKind`] ([`crate::bind::bind_all_weights`]'s
//! unconditional per-layer set was wrong for this checkpoint -- it demands
//! `blk.N.ffn_norm.weight`, which does not exist here; this checkpoint names
//! that tensor `post_attention_norm.weight` instead, on every layer,
//! regardless of kind). [`qwen35_forward_program`] compiles the whole
//! hybrid forward program (`proxima_tensor::spec::qwen35_forward_program`),
//! interleaving [`Qwen35LayerKind::Attention`]/[`Qwen35LayerKind::Ssm`]
//! layers per that same per-layer marker.

use alloc::format;
use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;
use proxima_tensor::op::{NodeId, Op};

use crate::bind::{
    BoundWeights, bind_dense, bind_matmul_weight, bind_matmul_weight_as, find_tensor,
    metadata_f32_optional, metadata_str, metadata_u32, metadata_u32_optional_or,
    vocab_from_token_embedding,
};
use crate::error::InteropError;

/// `name`'s own on-disk `(out_dim, in_dim)`, read from its declared GGUF
/// shape rather than derived from `{architecture}.attention.head_count` *
/// `rope.dimension_count` -- confirmed necessary against the real 27B
/// checkpoint: `rope.dimension_count` (`64`) is this architecture's
/// PARTIAL-rotary width, not the attention head's real width (`attn_q`'s
/// own on-disk shape proves the real per-head width is `512`, not `64`),
/// so deriving `attn_q`'s `out_dim` from that metadata product silently
/// disagreed with the file itself. GGUF's own `ne` convention stores a
/// dense weight as `[in_dim, out_dim]` (`crate::lfm2::bind_lfm2_shortconv_in_proj`'s
/// own doc confirms the same convention for a fused projection).
fn out_in_dims(parsed: &ParsedGguf, name: &str) -> Result<(usize, usize), InteropError> {
    let tensor = find_tensor(parsed, name)?;
    let in_dim = *tensor.dims.first().unwrap_or(&0) as usize;
    let out_dim = *tensor.dims.get(1).unwrap_or(&1) as usize;
    Ok((out_dim, in_dim))
}

/// [`bind_matmul_weight`] with `out_dim`/`in_dim` read straight from
/// `name`'s own on-disk shape ([`out_in_dims`]) instead of handed down from
/// a caller's derived hparams.
fn bind_matmul_weight_self_shaped<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    name: alloc::string::String,
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    let (out_dim, in_dim) = out_in_dims(parsed, &name)?;
    bind_matmul_weight(parsed, file_bytes, name, out_dim, in_dim, state)
}

/// [`bind_matmul_weight_self_shaped`]'s alias-under-a-different-name
/// counterpart -- [`crate::qwen35::qwen35_forward_program`]'s ssm-kind
/// branch declares its fused gated-input-projection weights under its own
/// `Op::Input` names (`ssm_in.weight`/`ssm_gate.weight`), not this
/// checkpoint's real on-disk names (`attn_qkv.weight`/`attn_gate.weight`),
/// the same source-name-vs-target-name split [`bind_matmul_weight_as`]
/// already exists for.
fn bind_matmul_weight_as_self_shaped<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    source_name: alloc::string::String,
    target_name: alloc::string::String,
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    let (out_dim, in_dim) = out_in_dims(parsed, &source_name)?;
    bind_matmul_weight_as(
        parsed,
        file_bytes,
        &source_name,
        target_name,
        out_dim,
        in_dim,
        state,
    )
}

/// One layer's real tensor shape, derived from
/// `{architecture}.full_attention_interval` rather than assumed uniform --
/// [`crate::lfm2::LayerKind`]'s counterpart for a checkpoint whose hybrid
/// marker is a scalar interval instead of a per-layer metadata array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35LayerKind {
    /// Dense self-attention: `attn_q`/`attn_k`/`attn_v`/`attn_output`, plus
    /// this checkpoint's own per-head `attn_q_norm`/`attn_k_norm`.
    Attention,
    /// A gated state-space mixer: the 7 `ssm_*` tensors plus this
    /// checkpoint's own `attn_gate`/`attn_qkv` fused input projection --
    /// confirmed present on every ssm-kind layer via `strings` on the real
    /// file, not inferred from the architecture name.
    Ssm,
}

impl Qwen35LayerKind {
    /// `layer` is dense attention iff it lands on `full_attention_interval`'s
    /// own 1-indexed boundary -- confirmed against the real checkpoint's own
    /// tensor names (`blk.3`/`blk.7`/`blk.11`/... carry `attn_q.weight`,
    /// every other layer carries `ssm_a` instead, for
    /// `full_attention_interval = 4`).
    fn from_interval(layer: u32, full_attention_interval: u32) -> Self {
        if full_attention_interval != 0 && (layer + 1).is_multiple_of(full_attention_interval) {
            Qwen35LayerKind::Attention
        } else {
            Qwen35LayerKind::Ssm
        }
    }
}

/// Every hparam this checkpoint's own metadata carries -- bind-scoped
/// today, but the `ssm_*` fields are read now so a later forward-op session
/// does not have to re-derive them: [`crate::lfm2::Lfm2Architecture`]'s own
/// precedent for holding hparams a bind-only pass does not yet consume.
#[derive(Debug, Clone)]
pub struct Qwen35Architecture {
    pub vocab: u32,
    pub embedding: u32,
    pub feed_forward: u32,
    pub query_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// The real per-head projection width (`{architecture}.attention.key_length`)
    /// -- `attention.key_length`/`value_length` on THIS checkpoint's own
    /// declared metadata, never `embedding / query_heads`: that arithmetic
    /// is not even an integer on the 27B checkpoint (`5120 / 24 = 213.33`),
    /// so `head_dim` (`rope.dimension_count`, this checkpoint's PARTIAL
    /// rotary width) cannot stand in for it either -- confirmed against
    /// both real files, where `attn_q_norm.weight`/`attn_k_norm.weight`
    /// are `attn_head_dim`-wide, not `head_dim`-wide.
    pub attn_head_dim: u32,
    pub block_count: u32,
    pub full_attention_interval: u32,
    pub rope_freq_base: f32,
    pub rms_epsilon: f32,
    pub ssm_conv_kernel: u32,
    pub ssm_state_size: u32,
    pub ssm_group_count: u32,
    pub ssm_time_step_rank: u32,
    pub ssm_inner_size: u32,
    pub layer_kinds: Vec<Qwen35LayerKind>,
}

/// llama.cpp's own RMSNorm epsilon default, used only when
/// `{architecture}.attention.layer_norm_rms_epsilon` is absent -- the same
/// fallback shape [`crate::lfm2::LFM2_RMS_EPSILON_DEFAULT`] uses.
const QWEN35_RMS_EPSILON_DEFAULT: f32 = 1e-6;

/// Derives [`Qwen35Architecture`] from `parsed`'s own metadata --
/// [`crate::lfm2::lfm2_architecture_from_metadata`]'s scalar-interval
/// counterpart.
///
/// # Errors
///
/// [`InteropError::MissingMetadataKey`] if a required key is absent.
pub fn qwen35_architecture_from_metadata(
    parsed: &ParsedGguf,
) -> Result<Qwen35Architecture, InteropError> {
    let architecture = metadata_str(parsed, "general.architecture")?;
    let embedding = metadata_u32(parsed, &format!("{architecture}.embedding_length"))?;
    let feed_forward = metadata_u32(parsed, &format!("{architecture}.feed_forward_length"))?;
    let query_heads = metadata_u32(parsed, &format!("{architecture}.attention.head_count"))?;
    let kv_heads = metadata_u32(parsed, &format!("{architecture}.attention.head_count_kv"))?;
    let block_count = metadata_u32(parsed, &format!("{architecture}.block_count"))?;
    let head_dim = metadata_u32_optional_or(
        parsed,
        &format!("{architecture}.rope.dimension_count"),
        embedding / query_heads.max(1),
    );
    let attn_head_dim = metadata_u32(parsed, &format!("{architecture}.attention.key_length"))?;
    let full_attention_interval =
        metadata_u32(parsed, &format!("{architecture}.full_attention_interval"))?;
    let vocab = vocab_from_token_embedding(parsed, embedding)?;
    let rope_freq_base = metadata_f32_optional(
        parsed,
        &format!("{architecture}.rope.freq_base"),
        proxima_tensor::sized::ROPE_FREQ_BASE_DEFAULT,
    );
    let rms_epsilon = metadata_f32_optional(
        parsed,
        &format!("{architecture}.attention.layer_norm_rms_epsilon"),
        QWEN35_RMS_EPSILON_DEFAULT,
    );
    let ssm_conv_kernel = metadata_u32(parsed, &format!("{architecture}.ssm.conv_kernel"))?;
    let ssm_state_size = metadata_u32(parsed, &format!("{architecture}.ssm.state_size"))?;
    let ssm_group_count = metadata_u32(parsed, &format!("{architecture}.ssm.group_count"))?;
    let ssm_time_step_rank =
        metadata_u32(parsed, &format!("{architecture}.ssm.time_step_rank"))?;
    let ssm_inner_size = metadata_u32(parsed, &format!("{architecture}.ssm.inner_size"))?;

    let layer_kinds = (0..block_count)
        .map(|layer| Qwen35LayerKind::from_interval(layer, full_attention_interval))
        .collect();

    Ok(Qwen35Architecture {
        vocab,
        embedding,
        feed_forward,
        query_heads,
        kv_heads,
        head_dim,
        attn_head_dim,
        block_count,
        full_attention_interval,
        rope_freq_base,
        rms_epsilon,
        ssm_conv_kernel,
        ssm_state_size,
        ssm_group_count,
        ssm_time_step_rank,
        ssm_inner_size,
        layer_kinds,
    })
}

/// Runs [`crate::bind::bind_dense`]/[`bind_matmul_weight`]
/// over every one of `architecture`'s `block_count` layers --
/// [`crate::bind::bind_all_weights`]'s per-layer-kind counterpart. Binds
/// `post_attention_norm.weight` on every layer (this checkpoint's own
/// `ffn_norm.weight` replacement, present on both layer kinds) rather than
/// the fixed dense set [`crate::bind::bind_all_weights`] demands.
///
/// The ssm-kind fused `attn_gate.weight`/`attn_qkv.weight` and all 7
/// `ssm_*` tensors bind through [`bind_dense`] rather than
/// [`bind_matmul_weight`]: no forward program consumes them yet, so this
/// pass has no derived `out_dim`/`in_dim` to hand a matmul-shaped bind, and
/// [`bind_dense`] binds any tensor's bytes without needing one.
///
/// # Errors
///
/// Whatever [`crate::bind::bind_dense`]/[`bind_matmul_weight`] can fail
/// with -- most notably
/// [`InteropError::UnknownTensor`] if a layer's own kind-specific tensor
/// set is not actually present.
pub fn bind_qwen35_weights<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    architecture: &Qwen35Architecture,
) -> Result<BoundWeights<'file>, InteropError> {
    let mut state = BoundWeights {
        resident_bytes: file_bytes.len(),
        owned: Vec::new(),
        packed: Vec::new(),
        packed_owned: Vec::new(),
    };

    bind_dense(parsed, file_bytes, "token_embd.weight".into(), &mut state)?;

    for (layer, kind) in architecture.layer_kinds.iter().enumerate() {
        let layer = layer as u32;
        bind_dense(
            parsed,
            file_bytes,
            format!("blk.{layer}.attn_norm.weight"),
            &mut state,
        )?;
        bind_dense(
            parsed,
            file_bytes,
            format!("blk.{layer}.post_attention_norm.weight"),
            &mut state,
        )?;

        match kind {
            Qwen35LayerKind::Attention => {
                bind_matmul_weight_self_shaped(
                    parsed,
                    file_bytes,
                    format!("blk.{layer}.attn_q.weight"),
                    &mut state,
                )?;
                bind_matmul_weight_self_shaped(
                    parsed,
                    file_bytes,
                    format!("blk.{layer}.attn_k.weight"),
                    &mut state,
                )?;
                bind_matmul_weight_self_shaped(
                    parsed,
                    file_bytes,
                    format!("blk.{layer}.attn_v.weight"),
                    &mut state,
                )?;
                bind_matmul_weight_self_shaped(
                    parsed,
                    file_bytes,
                    format!("blk.{layer}.attn_output.weight"),
                    &mut state,
                )?;
                bind_dense(
                    parsed,
                    file_bytes,
                    format!("blk.{layer}.attn_q_norm.weight"),
                    &mut state,
                )?;
                bind_dense(
                    parsed,
                    file_bytes,
                    format!("blk.{layer}.attn_k_norm.weight"),
                    &mut state,
                )?;
            }
            Qwen35LayerKind::Ssm => {
                // [`proxima_tensor::spec::qwen35_forward_program`]'s ssm
                // branch consumes ONE fused gated-input-projection weight
                // per name (`ssm_in.weight`/`ssm_gate.weight`), splitting
                // `q`/`k`/`v` out of the matmul's own ACTIVATION output
                // (`qkv_mixed`) inside the graph -- not the weight itself,
                // so this binds the real on-disk fused tensors
                // (`attn_qkv.weight`/`attn_gate.weight`) matmul-shaped
                // under the forward program's own names, rather than
                // pre-splitting the weight at bind time
                // ([`bind_qwen35_attn_qkv_split`]'s own row-split, built
                // for a since-abandoned pre-split forward-program design).
                bind_matmul_weight_as_self_shaped(
                    parsed,
                    file_bytes,
                    format!("blk.{layer}.attn_qkv.weight"),
                    format!("blk.{layer}.ssm_in.weight"),
                    &mut state,
                )?;
                bind_matmul_weight_as_self_shaped(
                    parsed,
                    file_bytes,
                    format!("blk.{layer}.attn_gate.weight"),
                    format!("blk.{layer}.ssm_gate.weight"),
                    &mut state,
                )?;
                // `ssm_alpha.weight`/`ssm_beta.weight` feed directly into
                // `qwen35_forward_program`'s `elementwise(Multiply,
                // [(normed, ...), (ssm_alpha/beta, ...)])` -> `reduce(Add,
                // ...)` pair (`spec.rs:4986-5017`) with no intermediate node
                // between the weight leaf and the fused reduce -- exactly
                // the direct-operand shape `quantized_operand`
                // (`cpu.rs:6007`) requires to route a `Q8_0` tensor through
                // `run_reduce_quantized` instead of dequantizing to owned
                // `f32`. `bind_matmul_weight_self_shaped` is the same
                // packed-capable bind every dense-attention projection
                // weight already uses, reused rather than duplicated.
                // `ssm_out.weight`/`ssm_conv1d.weight` stay on `bind_dense`:
                // both are consumed only after an intermediate elementwise
                // (`ssm_out_split`/`channel_slice`, `spec.rs:5036-5052,5203-5211`),
                // so the packed weight would need to survive through a
                // second op the generic evaluator does not carry a
                // quantized path for.
                bind_matmul_weight_self_shaped(
                    parsed,
                    file_bytes,
                    format!("blk.{layer}.ssm_alpha.weight"),
                    &mut state,
                )?;
                bind_matmul_weight_self_shaped(
                    parsed,
                    file_bytes,
                    format!("blk.{layer}.ssm_beta.weight"),
                    &mut state,
                )?;
                for suffix in [
                    "ssm_a",
                    "ssm_conv1d.weight",
                    "ssm_dt.bias",
                    "ssm_norm.weight",
                    "ssm_out.weight",
                ] {
                    bind_dense(
                        parsed,
                        file_bytes,
                        format!("blk.{layer}.{suffix}"),
                        &mut state,
                    )?;
                }
            }
        }

        bind_matmul_weight_self_shaped(
            parsed,
            file_bytes,
            format!("blk.{layer}.ffn_gate.weight"),
            &mut state,
        )?;
        bind_matmul_weight_self_shaped(
            parsed,
            file_bytes,
            format!("blk.{layer}.ffn_up.weight"),
            &mut state,
        )?;
        bind_matmul_weight_self_shaped(
            parsed,
            file_bytes,
            format!("blk.{layer}.ffn_down.weight"),
            &mut state,
        )?;
    }

    bind_dense(parsed, file_bytes, "output_norm.weight".into(), &mut state)?;
    // tied embeddings (confirmed on the real 2B checkpoint, `strings` shows
    // no standalone `output.weight` tensor): the same
    // `bind_matmul_weight_as` alias `crate::bind::bind_all_weights` already
    // uses for a tied-embedding dense checkpoint (`bind.rs:1122-1141`).
    if find_tensor(parsed, "output.weight").is_ok() {
        bind_matmul_weight_self_shaped(parsed, file_bytes, "output.weight".into(), &mut state)?;
    } else {
        let (vocab, embedding) = out_in_dims(parsed, "token_embd.weight")?;
        bind_matmul_weight_as(
            parsed,
            file_bytes,
            "token_embd.weight",
            "output.weight".into(),
            vocab,
            embedding,
            &mut state,
        )?;
    }
    Ok(state)
}

/// One bind attempt's report for a caller that never sees [`BoundWeights`]
/// (`pub(crate)`, `crate::generate::LoadedModel`'s own field type) --
/// [`crate::lfm2::run_lfm2_prefill`]'s bind-only counterpart, minus the
/// forward-program compile and generation loop this checkpoint has no
/// state-space kernel for yet.
///
/// # Errors
///
/// Whatever [`qwen35_architecture_from_metadata`]/[`bind_qwen35_weights`]
/// can fail with.
pub fn bind_qwen35_checkpoint(
    parsed: &ParsedGguf,
    file_bytes: &[u8],
) -> Result<(Qwen35Architecture, usize, usize, usize), InteropError> {
    let architecture = qwen35_architecture_from_metadata(parsed)?;
    let weights = bind_qwen35_weights(parsed, file_bytes, &architecture)?;
    Ok((
        architecture,
        weights.resident_bytes,
        weights.owned.len(),
        weights.packed.len(),
    ))
}

/// This checkpoint's forward-program seam -- [`crate::lfm2::run_lfm2_prefill`]'s
/// call into [`proxima_tensor::spec::lfm2_forward_program_with_experts`]
/// counterpart, minus the builder itself: every op-graph primitive that
/// builder composes (`append_attention_mixer`, `rmsnorm`, `elementwise`,
/// `reduce`, ...) is module-private to `proxima_tensor::spec`
/// (`proxima-tensor/src/spec.rs`) -- every forward program this crate runs
/// today is one call into a `pub fn ..._forward_program...` that module
/// exports whole, never a graph this crate assembles itself.
/// `proxima_tensor::spec` does not export a qwen35 one yet:
/// `append_qwen35_delta_net_step`/`append_qwen35_conv_branch` (`spec.rs`)
/// are its own state-space building blocks, still module-private, with no
/// `append_qwen35_ssm_mixer`/`qwen35_forward_program_with_experts` wrapping
/// them into something this crate can call.
///
/// The program this checkpoint needs, once that lands, is
/// [`proxima_tensor::spec::lfm2_forward_program_with_experts`]'s shape with
/// no MoE branch (the oracle asserts `ffn_gate_inp == nullptr` on every
/// layer of this checkpoint): per [`Qwen35LayerKind::Attention`] layer,
/// `append_attention_mixer`; per [`Qwen35LayerKind::Ssm`] layer, the
/// still-unwritten state-space mixer; both kinds then `attn_norm`/
/// `post_attention_norm` and a dense SwiGLU FFN
/// (`ffn_gate`/`ffn_up`/`ffn_down`), the same shape
/// `lfm2_forward_program_with_experts`'s own leading-dense-block branch
/// already builds.
///
/// # Errors
///
/// Whatever [`proxima_tensor::spec::qwen35_forward_program`] can fail with
/// (wrapped as [`InteropError::Tensor`]) -- most likely
/// [`proxima_tensor::TensorError::InvalidFullAttentionInterval`] if a
/// caller-constructed [`Qwen35Architecture`] carries `full_attention_interval
/// == 0` (the real checkpoint never does; `qwen35_architecture_from_metadata`
/// reads it straight off `{architecture}.full_attention_interval`).
pub fn qwen35_forward_program(
    architecture: &Qwen35Architecture,
) -> Result<(Vec<Op>, NodeId, Vec<proxima_tensor::spec::Qwen35LayerRoots>), InteropError> {
    let (program, logits_root, layer_roots) = proxima_tensor::spec::qwen35_forward_program(
        architecture.vocab,
        architecture.embedding,
        architecture.feed_forward,
        architecture.query_heads,
        architecture.kv_heads,
        architecture.head_dim,
        architecture.attn_head_dim,
        architecture.block_count,
        architecture.full_attention_interval,
        architecture.ssm_state_size,
        architecture.ssm_time_step_rank,
        architecture.ssm_group_count,
        architecture.ssm_inner_size,
        architecture.ssm_conv_kernel,
        architecture.rms_epsilon,
    )?;
    Ok((program, logits_root, layer_roots))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// The exact split the real checkpoint proved via `strings`:
    /// `full_attention_interval = 4` marks layers `3, 7, 11, ...` dense
    /// attention (16 of 64), every other layer a state-space mixer (48 of
    /// 64) -- asserted here as the pure arithmetic this module's bind loop
    /// relies on, independent of any real file.
    #[test]
    fn from_interval_matches_the_real_checkpoints_layer_split() {
        let kinds: Vec<Qwen35LayerKind> = (0..64)
            .map(|layer| Qwen35LayerKind::from_interval(layer, 4))
            .collect();

        let attention_count = kinds
            .iter()
            .filter(|kind| **kind == Qwen35LayerKind::Attention)
            .count();
        let ssm_count = kinds
            .iter()
            .filter(|kind| **kind == Qwen35LayerKind::Ssm)
            .count();

        assert_eq!(attention_count, 16, "one dense layer every 4th layer");
        assert_eq!(ssm_count, 48, "every other layer stays state-space");
        assert_eq!(kinds[3], Qwen35LayerKind::Attention);
        assert_eq!(kinds[7], Qwen35LayerKind::Attention);
        assert_eq!(kinds[0], Qwen35LayerKind::Ssm);
        assert_eq!(kinds[63], Qwen35LayerKind::Attention);
    }
}
