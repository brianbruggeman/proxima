//! The hybrid checkpoint whose `general.architecture` selects the `recurrent_interval` schedule source:
//! [`Qwen35Hparams`] derives this architecture's own metadata shape --
//! `{architecture}.full_attention_interval` marks every `interval`th layer
//! (1-indexed) as dense attention, every other layer as a gated
//! state-space mixer -- the same "read the checkpoint's own per-layer
//! marker, don't assume the dense shape" move [`crate::short_conv`] makes for its
//! own hybrid checkpoint, just from a scalar interval instead of a
//! per-layer array.
//!
//! [`qwen35_forward_program`] compiles the whole
//! hybrid forward program (`proxima_tensor::spec::qwen35_forward_program`),
//! interleaving [`IntervalLayerKind::Attention`]/[`IntervalLayerKind::Ssm`]
//! layers per that same per-layer marker.

use alloc::collections::BTreeSet;
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::value::{MetadataArray, MetadataValue};
use proxima_tensor::op::{NodeId, Op};
use proxima_tensor::spec::{
    CacheMask, CacheStrategy, ForwardProgram, KeySourceKind, LayerAttentionConfig, LayerKind, LayerSchedule,
    ModelDescriptor, RopeTableSel, ValueSourceKind, build_forward,
};

use crate::bind::{
    ModelHparams, metadata_f32_optional, metadata_str, metadata_u32, metadata_u32_optional_or,
    vocab_from_token_embedding,
};
use crate::bind_leaves::bind_program_leaves;
use crate::error::InteropError;
use crate::lowering::StepState;
use crate::profiles::{binding_profile, family_profile};

/// One layer's real tensor shape, derived from
/// `{architecture}.full_attention_interval` rather than assumed uniform --
/// [`proxima_tensor::spec::LayerKind`]'s counterpart for a checkpoint whose hybrid
/// marker is a scalar interval instead of a per-layer metadata array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntervalLayerKind {
    /// Dense self-attention: `attn_q`/`attn_k`/`attn_v`/`attn_output`, plus
    /// this checkpoint's own per-head `attn_q_norm`/`attn_k_norm`.
    Attention,
    /// A gated state-space mixer: the 7 `ssm_*` tensors plus this
    /// checkpoint's own `attn_gate`/`attn_qkv` fused input projection --
    /// confirmed present on every ssm-kind layer via `strings` on the real
    /// file, not inferred from the architecture name.
    Ssm,
}

impl IntervalLayerKind {
    /// `layer` is dense attention iff it lands on `full_attention_interval`'s
    /// own 1-indexed boundary -- confirmed against the real checkpoint's own
    /// tensor names (`blk.3`/`blk.7`/`blk.11`/... carry `attn_q.weight`,
    /// every other layer carries `ssm_a` instead, for
    /// `full_attention_interval = 4`).
    fn from_interval(layer: u32, full_attention_interval: u32) -> Self {
        if full_attention_interval != 0 && (layer + 1).is_multiple_of(full_attention_interval) {
            IntervalLayerKind::Attention
        } else {
            IntervalLayerKind::Ssm
        }
    }
}

/// Every hparam this checkpoint's own metadata carries -- bind-scoped
/// today, but the `ssm_*` fields are read now so a later forward-op session
/// does not have to re-derive them: [`crate::short_conv::Lfm2Hparams`]'s own
/// precedent for holding hparams a bind-only pass does not yet consume.
#[derive(Debug, Clone)]
pub struct Qwen35Hparams {
    /// `general.architecture` as the file declares it, the key the family and binding profiles resolve through.
    pub family: String,
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
    pub layer_kinds: Vec<IntervalLayerKind>,
}

/// llama.cpp's own RMSNorm epsilon default, used only when
/// `{architecture}.attention.layer_norm_rms_epsilon` is absent -- the same
/// fallback shape [`crate::short_conv::LFM2_RMS_EPSILON_DEFAULT`] uses.
const QWEN35_RMS_EPSILON_DEFAULT: f32 = 1e-6;

/// Derives [`Qwen35Hparams`] from `parsed`'s own metadata --
/// [`crate::short_conv::lfm2_architecture_from_metadata`]'s scalar-interval
/// counterpart.
///
/// # Errors
///
/// [`InteropError::MissingMetadataKey`] if a required key is absent.
pub fn qwen35_architecture_from_metadata(
    parsed: &ParsedGguf,
) -> Result<Qwen35Hparams, InteropError> {
    let architecture = metadata_str(parsed, "general.architecture")?;
    let embedding = metadata_u32(parsed, &format!("{architecture}.embedding_length"))?;
    let feed_forward = metadata_u32(parsed, &format!("{architecture}.feed_forward_length"))?;
    let query_heads = metadata_u32(parsed, &format!("{architecture}.attention.head_count"))?;
    let kv_heads =
        metadata_u32_nonzero_uniform(parsed, &format!("{architecture}.attention.head_count_kv"))?;
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
    let ssm_time_step_rank = metadata_u32(parsed, &format!("{architecture}.ssm.time_step_rank"))?;
    let ssm_inner_size = metadata_u32(parsed, &format!("{architecture}.ssm.inner_size"))?;

    let layer_kinds = (0..block_count)
        .map(|layer| IntervalLayerKind::from_interval(layer, full_attention_interval))
        .collect();

    Ok(Qwen35Hparams {
        family: architecture.to_owned(),
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

fn metadata_u32_nonzero_uniform(parsed: &ParsedGguf, key: &str) -> Result<u32, InteropError> {
    let mut values = BTreeSet::new();
    match parsed.metadata_value(key) {
        Some(MetadataValue::U32(value)) => return Ok(*value),
        Some(MetadataValue::I32(value)) => {
            let value = u32::try_from(*value)
                .map_err(|_| InteropError::MissingMetadataKey { key: key.into() })?;
            if value != 0 {
                values.insert(value);
            }
        }
        Some(MetadataValue::Array(MetadataArray::U32(array))) => {
            values.extend(array.iter().copied().filter(|value| *value != 0));
        }
        Some(MetadataValue::Array(MetadataArray::I32(array))) => {
            for value in array {
                let value = u32::try_from(*value)
                    .map_err(|_| InteropError::MissingMetadataKey { key: key.into() })?;
                if value != 0 {
                    values.insert(value);
                }
            }
        }
        _ => return Err(InteropError::MissingMetadataKey { key: key.into() }),
    }
    match values.len() {
        1 => Ok(values.into_iter().next().unwrap_or(0)),
        0 => Err(InteropError::MissingMetadataKey { key: key.into() }),
        distinct_values => Err(InteropError::HeterogeneousNonzeroMetadataArray {
            key: key.into(),
            distinct_values,
        }),
    }
}

/// One bind attempt's report for a caller that never sees [`BoundWeights`]
/// (`pub(crate)`, `crate::generate::LoadedModel`'s own field type) --
/// [`crate::short_conv::run_lfm2_prefill`]'s bind-only counterpart, minus the
/// forward-program compile and generation loop this checkpoint has no
/// state-space kernel for yet.
///
/// # Errors
///
/// Whatever `qwen35_architecture_from_metadata`, [`qwen35_forward_program`]
/// and [`bind_program_leaves`] can fail with.
pub fn bind_qwen35_checkpoint(
    parsed: &ParsedGguf,
    file_bytes: &[u8],
) -> Result<(Qwen35Hparams, usize, usize, usize), InteropError> {
    let architecture = qwen35_architecture_from_metadata(parsed)?;
    let (program, _, _) = qwen35_forward_program(&architecture)?;
    let weights = bind_program_leaves(
        parsed,
        file_bytes,
        &program,
        &binding_profile(&architecture.family)?,
        &[],
    )?;
    Ok((
        architecture,
        weights.resident_bytes,
        weights.owned.len(),
        weights.packed.len(),
    ))
}

/// This checkpoint's forward-program seam -- [`crate::short_conv::run_lfm2_prefill`]'s
/// call into [`proxima_tensor::spec::lfm2_forward_program_with_experts`]
/// counterpart, minus the builder itself: every op-graph primitive that
/// builder composes (`append_attention_mixer`, `rmsnorm`, `elementwise`,
/// `reduce`, ...) is module-private to `proxima_tensor::spec`
/// (`proxima-tensor/src/spec.rs`) -- every forward program this crate runs
/// today is one call into a `pub fn ..._forward_program...` that module
/// exports whole, never a graph this crate assembles itself.
/// `proxima_tensor::spec` does not export a recurrent-interval one yet:
/// `append_qwen35_delta_net_step`/`append_qwen35_conv_branch` (`spec.rs`)
/// are its own state-space building blocks, still module-private, with no
/// `append_qwen35_ssm_mixer`/`qwen35_forward_program_with_experts` wrapping
/// them into something this crate can call.
///
/// The program this checkpoint needs, once that lands, is
/// [`proxima_tensor::spec::lfm2_forward_program_with_experts`]'s shape with
/// no MoE branch (the oracle asserts `ffn_gate_inp == nullptr` on every
/// layer of this checkpoint): per [`IntervalLayerKind::Attention`] layer,
/// `append_attention_mixer`; per [`IntervalLayerKind::Ssm`] layer, the
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
/// caller-constructed [`Qwen35Hparams`] carries `full_attention_interval
/// == 0` (the real checkpoint never does; `qwen35_architecture_from_metadata`
/// reads it straight off `{architecture}.full_attention_interval`).
/// [`crate::generate::LoadedModel`]'s own `SsmLayerCache::new` fixed sizes,
/// all derived from [`Qwen35Hparams`]'s ssm hyperparameters at load
/// time -- llama.cpp's hybrid-model source, lines 57-60's same derivation this module's
/// `bind_qwen35_attn_qkv_split` already walks through for the fused
/// `attn_qkv.weight` split.
#[derive(Debug, Clone, Copy)]
pub struct SsmShape {
    /// `2 * ssm_key_dim + ssm_d_inner` -- one `qkv_mixed` row's width,
    /// matching `proxima_tensor::spec::qwen35_forward_program`'s own
    /// `ssm_cache.{layer}.conv_history` leaf shape's second axis.
    pub qkv_dim: usize,
    /// `ssm_d_conv - 1` -- the rolling conv-history window's fixed row
    /// count `append_qwen35_ssm_mixer`'s doc names (the causal conv1d
    /// kernel's own left-context width).
    pub conv_rows: usize,
    /// `ssm_d_state * head_v_dim * ssm_n_group * ssm_group` -- the gated
    /// DeltaNet recurrent state's flat element count, matching
    /// `qwen35_forward_program`'s own `ssm_cache.{layer}.state` leaf shape.
    pub state_len: usize,
}

/// [`SsmShape`]'s own derivation off a real checkpoint's ssm
/// hyperparameters -- llama.cpp's hybrid-model source, lines 57-60's same arithmetic
/// [`Qwen35Hparams`]'s own `ssm_key_dim`/`ssm_value_dim` derivation
/// already uses for the fused `attn_qkv.weight` row split, plus
/// `head_v_dim = ssm_inner_size / ssm_time_step_rank` and `ssm_group =
/// ssm_time_step_rank / ssm_group_count`
/// (`proxima_tensor::spec::qwen35_forward_program`'s own `head_v_dim`/
/// `ssm_group` locals).
#[must_use]
pub fn qwen35_ssm_shape(architecture: &Qwen35Hparams) -> SsmShape {
    let ssm_key_dim = architecture.ssm_state_size * architecture.ssm_group_count;
    let head_v_dim = architecture.ssm_inner_size / architecture.ssm_time_step_rank;
    let ssm_group = architecture.ssm_time_step_rank / architecture.ssm_group_count;
    SsmShape {
        qkv_dim: (2 * ssm_key_dim + architecture.ssm_inner_size) as usize,
        conv_rows: (architecture.ssm_conv_kernel.saturating_sub(1)) as usize,
        state_len: (architecture.ssm_state_size
            * head_v_dim
            * architecture.ssm_group_count
            * ssm_group) as usize,
    }
}

/// [`SsmShape`]'s own resident bytes across every layer -- one
/// `SsmLayerCache::new`'s worth (`conv_rows * qkv_dim` conv-history
/// elements plus `state_len` state elements, both `f32`) times
/// `block_count` layers. `crate::memory_fit`'s own load-time gate reads
/// this as the SSM class of `crate::memory_fit::WeightClassBytes` -- `0`
/// for every non-recurrent-interval checkpoint, which never builds a [`SsmShape`]
/// at all. Plain arithmetic, no platform dependency -- unlike the field it
/// used to feed directly, this function itself is not `metal`-gated, so
/// [`crate::lowering::step_state`] can call it on every build.
#[must_use]
pub fn qwen35_ssm_state_bytes(shape: SsmShape, block_count: u32) -> u64 {
    let per_layer_elements = (shape.conv_rows * shape.qkv_dim + shape.state_len) as u64;
    per_layer_elements * core::mem::size_of::<f32>() as u64 * u64::from(block_count)
}

/// The checkpoint's whole pre-lowering program as one config: the header's
/// layer kinds, attention and recurrence shapes over the family profile's FFN,
/// score scale and rope pairing. [`proxima_tensor::spec::build_forward`] over
/// the result is the entire lowering; serialize the descriptor, edit the layer
/// schedule, and a restored copy lowers the edited program with no Rust.
///
/// # Errors
///
/// The family has no embedded profile.
pub fn descriptor_from_architecture(architecture: &Qwen35Hparams) -> Result<ModelDescriptor, InteropError> {
    let profile = family_profile(&architecture.family)?;
    let attention = LayerAttentionConfig {
        head_dim: architecture.attn_head_dim,
        kv_heads: architecture.kv_heads,
        mask_window: None,
        value_source_kind: ValueSourceKind::ProjectedV,
        key_source_kind: KeySourceKind::ProjectedK,
        rope_table: RopeTableSel {
            cos_name: "rope_cos".into(),
            sin_name: "rope_sin".into(),
        },
        rope_pairing: profile.rope_pairing(architecture.head_dim),
        score_scale: profile.score_scale(architecture.attn_head_dim),
        value_norm: profile.value_norm,
    };
    let ffn = profile.layer_ffn(0);
    let layers = architecture
        .layer_kinds
        .iter()
        .map(|kind| match kind {
            IntervalLayerKind::Attention => LayerSchedule {
                kind: LayerKind::Attention,
                attention: attention.clone(),
                ffn,
            },
            IntervalLayerKind::Ssm => LayerSchedule {
                kind: LayerKind::Gdn,
                attention: attention.clone(),
                ffn,
            },
        })
        .collect();
    Ok(ModelDescriptor {
        vocab: architecture.vocab,
        embedding: architecture.embedding,
        feed_forward: architecture.feed_forward,
        expert_feed_forward: architecture.feed_forward,
        query_heads: architecture.query_heads,
        block_count: architecture.block_count,
        expert_count: 0,
        expert_used_count: 0,
        leading_dense_block_count: architecture.block_count,
        l_cache: 0,
        embedding_scale: profile.embedding_scale,
        logit_softcap: None,
        logit_scale: None,
        residual_scale: None,
        layers,
        cache_strategy: CacheStrategy::Cached,
        cache_mask: CacheMask::Bounded,
        ple_dim: None,
        sliding_kv_ring: false,
        qk_norm: false,
        qkv_biases: false,
        paired_gate_up_reduce: false,
        fused_qkv_reduce: false,
        head_repeats: 1,
        last_row_only: true,
        speculative_verify: profile.speculative_verify,
        ssm_conv_kernel: architecture.ssm_conv_kernel,
        ssm_state_size: architecture.ssm_state_size,
        ssm_group_count: architecture.ssm_group_count,
        ssm_time_step_rank: architecture.ssm_time_step_rank,
        ssm_inner_size: architecture.ssm_inner_size,
        ssm_epsilon: architecture.rms_epsilon,
        v_head_reordered: false,
        expert_shared_feed_forward: 0,
        prefill_width: None,
        gated_attention: true,
    })
}

/// This checkpoint's forward program: [`descriptor_from_architecture`] lowered
/// by [`proxima_tensor::spec::build_forward`].
///
/// # Errors
///
/// The family has no profile, or the descriptor does not lower.
pub fn qwen35_forward_program(
    architecture: &Qwen35Hparams,
) -> Result<(Vec<Op>, NodeId, Vec<proxima_tensor::spec::LayerCacheRoots>), InteropError> {
    let ForwardProgram {
        program,
        logits,
        layer_roots,
        ..
    } = build_forward(&descriptor_from_architecture(architecture)?)?;
    Ok((program, logits, layer_roots))
}

/// The `recurrent_interval` header reader
/// ([`proxima_tensor::spec::ScheduleSource::RecurrentInterval`]): this header's
/// descriptor and hyperparameters, the two values [`crate::lowering`] lowers
/// and binds from.
///
/// # Errors
///
/// Whatever [`qwen35_architecture_from_metadata`] and
/// [`descriptor_from_architecture`] can fail with.
pub(crate) fn header(parsed: &ParsedGguf) -> Result<(ModelDescriptor, ModelHparams), InteropError> {
    let hparams = qwen35_architecture_from_metadata(parsed)?;
    let descriptor = descriptor_from_architecture(&hparams)?;
    let architecture = ModelHparams {
        vocab: hparams.vocab,
        embedding: hparams.embedding,
        feed_forward: hparams.feed_forward,
        query_heads: hparams.query_heads,
        kv_heads: hparams.kv_heads,
        kv_heads_by_layer: vec![hparams.kv_heads; hparams.block_count as usize],
        head_dim: hparams.head_dim,
        block_count: hparams.block_count,
        expert_count: 0,
        expert_used_count: 0,
        rope_freq_base: hparams.rope_freq_base,
        rms_epsilon: hparams.rms_epsilon,
        tied_embeddings: false,
        family: hparams.family.clone(),
        sliding_rope: None,
    };
    Ok((descriptor, architecture))
}

/// The per-decode-step recurrent scratch sizing, re-derived straight off the
/// header: [`qwen35_ssm_shape`] and the resident bytes across every layer.
///
/// # Errors
///
/// Whatever [`qwen35_architecture_from_metadata`] can fail with.
pub(crate) fn step_state(parsed: &ParsedGguf) -> Result<StepState, InteropError> {
    let hparams = qwen35_architecture_from_metadata(parsed)?;
    let shape = qwen35_ssm_shape(&hparams);
    Ok(StepState {
        ssm_shape: shape,
        attn_head_dim: hparams.attn_head_dim,
        ssm_state_bytes: qwen35_ssm_state_bytes(shape, hparams.block_count),
    })
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
        let kinds: Vec<IntervalLayerKind> = (0..64)
            .map(|layer| IntervalLayerKind::from_interval(layer, 4))
            .collect();

        let attention_count = kinds
            .iter()
            .filter(|kind| **kind == IntervalLayerKind::Attention)
            .count();
        let ssm_count = kinds
            .iter()
            .filter(|kind| **kind == IntervalLayerKind::Ssm)
            .count();

        assert_eq!(attention_count, 16, "one dense layer every 4th layer");
        assert_eq!(ssm_count, 48, "every other layer stays state-space");
        assert_eq!(kinds[3], IntervalLayerKind::Attention);
        assert_eq!(kinds[7], IntervalLayerKind::Attention);
        assert_eq!(kinds[0], IntervalLayerKind::Ssm);
        assert_eq!(kinds[63], IntervalLayerKind::Attention);
    }
}
