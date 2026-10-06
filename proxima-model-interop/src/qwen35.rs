//! Qwen3.8-27B's real hybrid checkpoint (`general.architecture = "qwen35"`):
//! [`Qwen35Architecture`] derives this architecture's own metadata shape --
//! `{architecture}.full_attention_interval` marks every `interval`th layer
//! (1-indexed) as dense attention, every other layer as a gated
//! state-space mixer -- the same "read the checkpoint's own per-layer
//! marker, don't assume the dense shape" move [`crate::lfm2`] makes for its
//! own hybrid checkpoint, just from a scalar interval instead of a
//! per-layer array.
//!
//! [`qwen35_forward_program`] compiles the whole
//! hybrid forward program (`proxima_tensor::spec::qwen35_forward_program`),
//! interleaving [`Qwen35LayerKind::Attention`]/[`Qwen35LayerKind::Ssm`]
//! layers per that same per-layer marker.

use alloc::collections::BTreeSet;
use alloc::format;
use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::value::{MetadataArray, MetadataValue};
use proxima_tensor::op::{NodeId, Op};

use crate::bind::{
    metadata_f32_optional, metadata_str, metadata_u32, metadata_u32_optional_or,
    vocab_from_token_embedding,
};
use crate::bind_leaves::bind_program_leaves;
use crate::error::InteropError;
use crate::profiles::binding_profile;

/// One layer's real tensor shape, derived from
/// `{architecture}.full_attention_interval` rather than assumed uniform --
/// [`proxima_tensor::spec::LayerKind`]'s counterpart for a checkpoint whose hybrid
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
/// [`crate::lfm2::run_lfm2_prefill`]'s bind-only counterpart, minus the
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
) -> Result<(Qwen35Architecture, usize, usize, usize), InteropError> {
    let architecture = qwen35_architecture_from_metadata(parsed)?;
    let (program, _, _) = qwen35_forward_program(&architecture)?;
    let weights = bind_program_leaves(
        parsed,
        file_bytes,
        &program,
        &binding_profile(FAMILY)?,
        &[],
    )?;
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
/// [`crate::generate::LoadedModel`]'s own `SsmLayerCache::new` fixed sizes,
/// all derived from [`Qwen35Architecture`]'s ssm hyperparameters at load
/// time -- `qwen35.cpp:57-60`'s same derivation this module's
/// `bind_qwen35_attn_qkv_split` already walks through for the fused
/// `attn_qkv.weight` split.
#[derive(Debug, Clone, Copy)]
pub struct Qwen35SsmShape {
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

/// [`Qwen35SsmShape`]'s own derivation off a real checkpoint's ssm
/// hyperparameters -- `qwen35.cpp:57-60`'s same arithmetic
/// [`Qwen35Architecture`]'s own `ssm_key_dim`/`ssm_value_dim` derivation
/// already uses for the fused `attn_qkv.weight` row split, plus
/// `head_v_dim = ssm_inner_size / ssm_time_step_rank` and `ssm_group =
/// ssm_time_step_rank / ssm_group_count`
/// (`proxima_tensor::spec::qwen35_forward_program`'s own `head_v_dim`/
/// `ssm_group` locals).
#[must_use]
pub fn qwen35_ssm_shape(architecture: &Qwen35Architecture) -> Qwen35SsmShape {
    let ssm_key_dim = architecture.ssm_state_size * architecture.ssm_group_count;
    let head_v_dim = architecture.ssm_inner_size / architecture.ssm_time_step_rank;
    let ssm_group = architecture.ssm_time_step_rank / architecture.ssm_group_count;
    Qwen35SsmShape {
        qkv_dim: (2 * ssm_key_dim + architecture.ssm_inner_size) as usize,
        conv_rows: (architecture.ssm_conv_kernel.saturating_sub(1)) as usize,
        state_len: (architecture.ssm_state_size
            * head_v_dim
            * architecture.ssm_group_count
            * ssm_group) as usize,
    }
}

/// [`Qwen35SsmShape`]'s own resident bytes across every layer -- one
/// `SsmLayerCache::new`'s worth (`conv_rows * qkv_dim` conv-history
/// elements plus `state_len` state elements, both `f32`) times
/// `block_count` layers. `crate::memory_fit`'s own load-time gate reads
/// this as the SSM class of `crate::memory_fit::WeightClassBytes` -- `0`
/// for every non-qwen35 checkpoint, which never builds a [`Qwen35SsmShape`]
/// at all. Plain arithmetic, no platform dependency -- unlike the field it
/// used to feed directly, this function itself is not `metal`-gated, so
/// [`crate::architecture::Architecture::step_state`] can call it on every
/// build.
#[must_use]
pub fn qwen35_ssm_state_bytes(shape: Qwen35SsmShape, block_count: u32) -> u64 {
    let per_layer_elements = (shape.conv_rows * shape.qkv_dim + shape.state_len) as u64;
    per_layer_elements * core::mem::size_of::<f32>() as u64 * u64::from(block_count)
}

pub fn qwen35_forward_program(
    architecture: &Qwen35Architecture,
) -> Result<(Vec<Op>, NodeId, Vec<proxima_tensor::spec::Qwen35LayerRoots>), InteropError> {
    let (program, logits_root, layer_roots) =
        proxima_tensor::spec::qwen35_forward_program_with_last_row(
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
            true,
        )?;
    Ok((program, logits_root, layer_roots))
}

/// The binding profile key and the registry name: the architecture that lowers this program
/// is what names its leaves, so a delegating foreign architecture binds the same way.
const FAMILY: &str = "qwen35";

/// The [`crate::architecture::Architecture`] registered under the name
/// `"qwen35"` -- [`crate::architecture::ArchitectureRegistry::with_builtin`]'s
/// hybrid-checkpoint arm, and the worked example that trait's own doc
/// points a foreign architecture at. Named distinctly from
/// [`Qwen35Architecture`] (the per-checkpoint metadata this arm derives
/// inside [`crate::architecture::Architecture::bind`]) because the two are different kinds of
/// value: `Qwen35Arch` is a stateless, `'static` marker one `bind` call
/// derives fresh metadata against every load; `Qwen35Architecture` is that
/// derived, per-checkpoint data.
pub struct Qwen35Arch;

/// The one registered [`Qwen35Arch`] value -- see
/// [`crate::architecture::ArchitectureRegistry::with_builtin`]'s own doc
/// for why a `'static` marker, not an owned value, is what a registry
/// entry is.
pub static QWEN35: Qwen35Arch = Qwen35Arch;

impl crate::architecture::Architecture for Qwen35Arch {
    fn name(&self) -> &'static str {
        FAMILY
    }

    fn kv_cache_shape(&self) -> crate::architecture::KvCacheShape {
        crate::architecture::KvCacheShape::Custom
    }

    fn diagnostic_reduce_flags_apply(&self) -> bool {
        false
    }

    fn bind<'file>(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
    ) -> Result<crate::architecture::BoundProgram<'file>, InteropError> {
        let qwen_architecture = qwen35_architecture_from_metadata(parsed)?;
        let (program, logits_root, layer_roots) = qwen35_forward_program(&qwen_architecture)?;
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
            feed_forward: qwen_architecture.feed_forward,
            query_heads: qwen_architecture.query_heads,
            kv_heads: qwen_architecture.kv_heads,
            kv_heads_by_layer: vec![
                qwen_architecture.kv_heads;
                qwen_architecture.block_count as usize
            ],
            head_dim: qwen_architecture.head_dim,
            block_count: qwen_architecture.block_count,
            // Qwen3.5 never routes FFN through experts
            // (`qwen35_forward_program`'s own doc, `qwen35.cpp:471`), so
            // this checkpoint reads the same `expert_count == 0` dense-FFN
            // branch every other checkpoint without a
            // `{architecture}.expert_count` key does.
            expert_count: 0,
            expert_used_count: 0,
            rope_freq_base: qwen_architecture.rope_freq_base,
            rms_epsilon: qwen_architecture.rms_epsilon,
            tied_embeddings: false,
            family: metadata_str(parsed, "general.architecture")?.into(),
            sliding_rope: None,
        };
        Ok(crate::architecture::BoundProgram {
            weights,
            architecture,
            program,
            logits_root,
            hidden_root: None,
            residual_roots: Vec::new(),
            layer_roots,
            qwen35moe_layer_diagnostics: Vec::new(),
            router_roots: Vec::new(),
            moe_sites: proxima_tensor::spec::MoeSites::default(),
            duplicate_head_roots: Vec::new(),
            // gated-DeltaNet's `s`-axis reduce sums positions instead of
            // stepping through them (`BoundProgram::single_position_step`'s
            // own doc). ROW 427: `run_decode_loop_observed_seeded` now
            // splits its prefill into one `new_count == 1` evaluation per
            // prompt position when this is set, through the same per-step
            // evaluate + cache-append path a decode step already uses (see
            // that method's own `step_batches` doc), so setting `true` here
            // no longer turns a multi-token qwen35 prompt into a hard
            // `bind_symbols` error -- it is the reason that split exists.
            single_position_step: true,
        })
    }

    fn step_state(
        &self,
        parsed: &ParsedGguf,
    ) -> Result<Option<crate::architecture::StepState>, InteropError> {
        // Re-derives `Qwen35Architecture` from `parsed`'s own metadata --
        // the same pure, file-free read `Architecture::bind` just did --
        // rather than threading it through `BoundProgram` for this one
        // hook's sake (see the trait method's own doc).
        let qwen_architecture = qwen35_architecture_from_metadata(parsed)?;
        let shape = qwen35_ssm_shape(&qwen_architecture);
        Ok(Some(crate::architecture::StepState {
            ssm_shape: shape,
            attn_head_dim: qwen_architecture.attn_head_dim,
            ssm_state_bytes: qwen35_ssm_state_bytes(shape, qwen_architecture.block_count),
        }))
    }
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
