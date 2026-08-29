//! LFM2.5-8B-A1B's real hybrid checkpoint: [`Lfm2Architecture`] derives
//! this architecture's own metadata shape -- a per-layer `head_count_kv`
//! array whose zero entries mark short-convolution layers
//! ([`crate::bind::architecture_from_metadata`] correctly refuses this
//! shape, see that function's own doc for why a genuinely hybrid
//! architecture cannot use the generic dense-checkpoint reader) -- binds
//! every weight [`proxima_tensor::spec::lfm2_forward_program_with_experts`]
//! needs ([`bind_lfm2_weights`], including [`bind_lfm2_shortconv_in_proj`]'s
//! binder-side split of the real checkpoint's one fused `in_proj` tensor),
//! and runs the resulting program end to end ([`run_lfm2_prefill`]):
//! tokenize, one whole-sequence prefill pass per generated token (this
//! architecture's own forward program is prefill-only --
//! [`lfm2_forward_program_with_experts`]'s own doc), greedy-pick, repeat.
//!
//! Three real-checkpoint gaps a prior session named rather than closed are
//! closed here: `attn_q_norm.weight`/`attn_k_norm.weight` (a per-head
//! RMSNorm on Q/K, applied BEFORE RoPE, on the 6 attention layers --
//! [`proxima_tensor::spec::append_attention_mixer`]'s own doc cites
//! `Lfm2MoeAttention`'s exact placement), `blk.{layer}.exp_probs_b.bias`
//! (a learned per-expert bias that gates top-k *selection* only, never the
//! selected experts' combination weight -- bound here, consumed by
//! [`proxima_tensor::spec::append_moe_ffn`]'s own `expert_bias` parameter),
//! and `{architecture}.expert_gating_func == 2`
//! (`LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`, `llama-hparams.h:14`) --
//! [`bind_lfm2_weights`] binds all three, and [`run_lfm2_prefill`] builds
//! the program with [`proxima_tensor::spec::ExpertGatingFunc::Sigmoid`]
//! rather than the softmax-style top-k reweighting a dense-gated checkpoint
//! (Mixtral) still gets.

use alloc::collections::BTreeSet;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::types::GgmlType;
use proxima_gguf::value::{MetadataArray, MetadataValue};
use proxima_tensor::cpu::{QuantizedBlock, evaluate_quantized_named_with_scratch};
use proxima_tensor::op::NodeId;
use proxima_tensor::spec::{LayerKind, lfm2_forward_program_with_experts};
use proxima_tokenizer::Vocab;

use crate::bind::{
    BoundWeights, aligned_f32_view, bind_dense_as, bind_matmul_weight, bind_matmul_weight_as, bind_moe_expert_weights,
    find_tensor, metadata_f32_optional, metadata_str, metadata_u32, metadata_u32_optional, metadata_u32_optional_or,
    reinterpret_f32, vocab_from_token_embedding,
};
use crate::error::InteropError;

/// Every hparam [`lfm2_forward_program_with_experts`] needs, derived from a
/// real `lfm2moe`-architecture checkpoint's own metadata --
/// [`crate::bind::ModelArchitecture`]'s hybrid-checkpoint counterpart, not
/// a variant of it: that struct's single `kv_heads: u32` and lack of a
/// `layer_kinds`/`leading_dense_block_count`/`l_cache` field mean it cannot
/// describe this architecture at all, not even partially.
#[derive(Debug, Clone)]
pub struct Lfm2Architecture {
    pub vocab: u32,
    pub embedding: u32,
    pub feed_forward: u32,
    pub expert_feed_forward: u32,
    pub query_heads: u32,
    /// The real per-attention-layer kv head count -- every convolution
    /// layer's own `0` placeholder entry in the real checkpoint's
    /// `head_count_kv` array is skipped, not averaged in; see
    /// `metadata_u32_array_nonzero_uniform`'s own doc.
    pub kv_heads: u32,
    pub head_dim: u32,
    pub block_count: u32,
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub leading_dense_block_count: u32,
    pub l_cache: u32,
    pub rope_freq_base: f32,
    pub rms_epsilon: f32,
    pub layer_kinds: Vec<LayerKind>,
}

/// [`transformers/models/lfm2_moe/modeling_lfm2_moe.py`]'s own RMSNorm
/// epsilon default, used only when the real checkpoint's own
/// `{architecture}.attention.layer_norm_rms_epsilon` key is absent -- on
/// the one real checkpoint this module has been run against, the key IS
/// present (`0.00001`), so this fallback is unexercised there.
const LFM2_RMS_EPSILON_DEFAULT: f32 = 1e-5;

/// Derives [`Lfm2Architecture`] from `parsed`'s own metadata --
/// [`crate::bind::architecture_from_metadata`]'s hybrid-checkpoint
/// counterpart. Reads `general.architecture` itself (`lfm2moe` on the real
/// checkpoint, not `lfm2`) rather than assuming it, the same "read the
/// wire, don't hard-code the string" shape every other key here already
/// uses.
///
/// # Errors
///
/// [`InteropError::MissingMetadataKey`] if a required key is absent;
/// [`InteropError::HeterogeneousNonzeroMetadataArray`] if
/// `{architecture}.attention.head_count_kv`'s nonzero (attention-layer)
/// entries disagree with each other; whatever
/// [`proxima_tensor::spec::LayerKind::from_tensor_names`] fails with if a
/// layer's tensor directory carries neither an attention nor a
/// short-convolution marker.
pub fn lfm2_architecture_from_metadata(parsed: &ParsedGguf) -> Result<Lfm2Architecture, InteropError> {
    let architecture = metadata_str(parsed, "general.architecture")?;
    let embedding = metadata_u32(parsed, &format!("{architecture}.embedding_length"))?;
    let feed_forward = metadata_u32(parsed, &format!("{architecture}.feed_forward_length"))?;
    let expert_feed_forward = metadata_u32_optional(parsed, &format!("{architecture}.expert_feed_forward_length"));
    let query_heads = metadata_u32(parsed, &format!("{architecture}.attention.head_count"))?;
    let kv_heads = metadata_u32_array_nonzero_uniform(parsed, &format!("{architecture}.attention.head_count_kv"))?;
    let block_count = metadata_u32(parsed, &format!("{architecture}.block_count"))?;
    let head_dim = metadata_u32_optional_or(parsed, &format!("{architecture}.rope.dimension_count"), embedding / query_heads.max(1));
    let vocab = vocab_from_token_embedding(parsed, embedding)?;
    let expert_count = metadata_u32_optional(parsed, &format!("{architecture}.expert_count"));
    let expert_used_count = metadata_u32_optional(parsed, &format!("{architecture}.expert_used_count"));
    let leading_dense_block_count = metadata_u32_optional(parsed, &format!("{architecture}.leading_dense_block_count"));
    let l_cache = metadata_u32(parsed, &format!("{architecture}.shortconv.l_cache"))?;
    let rope_freq_base = metadata_f32_optional(
        parsed,
        &format!("{architecture}.rope.freq_base"),
        proxima_tensor::sized::ROPE_FREQ_BASE_DEFAULT,
    );
    let rms_epsilon = metadata_f32_optional(
        parsed,
        &format!("{architecture}.attention.layer_norm_rms_epsilon"),
        LFM2_RMS_EPSILON_DEFAULT,
    );

    let names: Vec<&str> = parsed.tensors.iter().map(|tensor| tensor.name.as_str()).collect();
    let mut layer_kinds = Vec::with_capacity(block_count as usize);
    for layer in 0..block_count {
        layer_kinds.push(LayerKind::from_tensor_names(names.iter().copied(), layer)?);
    }

    Ok(Lfm2Architecture {
        vocab,
        embedding,
        feed_forward,
        expert_feed_forward,
        query_heads,
        kv_heads,
        head_dim,
        block_count,
        expert_count,
        expert_used_count,
        leading_dense_block_count,
        l_cache,
        rope_freq_base,
        rms_epsilon,
        layer_kinds,
    })
}

/// [`crate::bind::metadata_u32_or_uniform_array`]'s hybrid-architecture
/// counterpart: every ZERO entry is a short-convolution layer's own
/// placeholder and is skipped rather than folded into the uniformity
/// check, but every NONZERO entry (a real attention layer's kv head count)
/// must still agree -- a checkpoint with two different real kv head
/// counts across its attention layers is not a shape this module (or
/// [`lfm2_forward_program_with_experts`], which takes one `kv_heads: u32`)
/// can represent, so that case is refused, not averaged or first-picked.
fn metadata_u32_array_nonzero_uniform(parsed: &ParsedGguf, key: &str) -> Result<u32, InteropError> {
    match parsed.metadata_value(key) {
        Some(MetadataValue::U32(value)) => Ok(*value),
        Some(MetadataValue::I32(value)) => {
            u32::try_from(*value).map_err(|_| InteropError::MissingMetadataKey { key: key.into() })
        }
        Some(MetadataValue::Array(MetadataArray::U32(values))) => nonzero_uniform_u32_array(key, values.iter().copied()),
        Some(MetadataValue::Array(MetadataArray::I32(values))) => {
            nonzero_uniform_u32_array(key, values.iter().map(|value| u32::try_from(*value).unwrap_or(u32::MAX)))
        }
        _ => Err(InteropError::MissingMetadataKey { key: key.into() }),
    }
}

fn nonzero_uniform_u32_array(key: &str, values: impl Iterator<Item = u32>) -> Result<u32, InteropError> {
    let mut distinct: BTreeSet<u32> = BTreeSet::new();
    for value in values {
        if value != 0 {
            distinct.insert(value);
        }
    }
    match distinct.len() {
        0 => Err(InteropError::MissingMetadataKey { key: key.into() }),
        1 => Ok(distinct.into_iter().next().unwrap_or(0)),
        distinct_values => Err(InteropError::HeterogeneousNonzeroMetadataArray { key: key.into(), distinct_values }),
    }
}

/// Row-splits a real checkpoint's fused `blk.{layer}.shortconv.in_proj.weight`
/// (GGUF's on-disk `[out_dim = 3 * embedding, in_dim = embedding]`
/// row-major layout: `3 * embedding` rows of `embedding` contiguous
/// elements each) into the three same-width `b`/`c`/`x` projections
/// [`lfm2_forward_program_with_experts`]'s own `LayerKind::ShortConv`
/// branch declares as separate `Input`s (`proxima-tensor/src/spec.rs`,
/// `append_lfm2_conv_mixer`'s own doc) -- that doc already proves the split
/// cannot happen inside the tensor program's own `Affine` grammar
/// (`shape::unify_iteration_space` resolves a pure single-term axis's
/// extent from the sliced operand's own FULL buffer width regardless of
/// offset), so it happens here, in the binder, instead.
///
/// The split is a ROW split, never a column slice: HuggingFace's own
/// reference (`Lfm2MoeShortConv.slow_forward`,
/// `transformers/models/lfm2_moe/modeling_lfm2_moe.py:443-444`) computes
/// `BCx = in_proj(x).transpose(-1, -2)` then `B, C, x = BCx.chunk(3,
/// dim=-2)` -- chunking the LINEAR LAYER'S OUTPUT axis, which is exactly
/// GGUF's `out_dim` axis, the *row* axis of the on-disk `[out_dim, in_dim]`
/// buffer. `B` is rows `0..embedding`, `C` is rows `embedding..2*embedding`,
/// `x` (ungated) is rows `2*embedding..3*embedding` -- `Bx = B * x` feeds
/// [`proxima_tensor::spec`]'s causal convolution, `C` gates the convolved
/// result (`modeling_lfm2_moe.py:446,462`), matching
/// `append_lfm2_conv_mixer`'s own `b_proj`/`c_proj`/`x_proj` argument order.
///
/// Block-quantized types (`Q4_K`/`Q5_K`/`Q6_K`) never need a mid-block
/// slice for this: `ggml` requires a row's own element count
/// (`in_dim = embedding`) be a whole multiple of the codec's
/// `block_elements` to quantize that row at all (confirmed on the real
/// checkpoint: `Q4_K`'s `block_elements = 256`, `embedding = 2048 = 8 *
/// 256`), and a K-quant superblock never spans two rows regardless of
/// `in_dim`'s own divisibility -- so a row-COUNT split is always a
/// byte-offset split at an exact multiple of one block's own
/// `block_bytes`, verified by this function's own arithmetic
/// (`rows_per_chunk * bytes_per_row`), never assumed. For the real
/// checkpoint's `Q4_K` `in_proj` (`embedding = 2048`): `bytes_per_row =
/// (2048 / 256) * 144 = 1152`; the `B`/`C` boundary lands at byte
/// `2048 * 1152 = 2359296`, the `C`/`x` boundary at `4096 * 1152 =
/// 4718592`, and the whole tensor spans `6144 * 1152 = 7077888` bytes --
/// every one an exact multiple of `144` (`1152 / 144 = 8` blocks per row).
///
/// # Errors
///
/// [`InteropError::UnknownTensor`] if the fused tensor is absent;
/// [`InteropError::ShortConvInProjShapeMismatch`] if its element count is
/// not exactly `3 * embedding * embedding`;
/// [`InteropError::ShortConvInProjNotBlockAligned`] if `embedding` is not a
/// whole multiple of the tensor's own codec `block_elements` (never
/// observed on the real checkpoint, but not assumed away either);
/// [`InteropError::UnrepresentableGgmlType`] for any `GgmlType` besides
/// `F32`/`Q4_K`/`Q5_K`/`Q6_K`.
pub(crate) fn bind_lfm2_shortconv_in_proj<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    layer: u32,
    embedding: u32,
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    let name = format!("blk.{layer}.shortconv.in_proj.weight");
    let tensor = find_tensor(parsed, &name)?;
    let elements = tensor.element_count();
    let expected = 3u64 * u64::from(embedding) * u64::from(embedding);
    if elements != expected {
        return Err(InteropError::ShortConvInProjShapeMismatch { layer, elements, embedding, expected });
    }

    let layout = tensor.ggml_type.block_layout();
    let elements_per_row = u64::from(embedding);
    if layout.block_elements == 0 || !elements_per_row.is_multiple_of(layout.block_elements) {
        return Err(InteropError::ShortConvInProjNotBlockAligned { layer, ggml_type: tensor.ggml_type, embedding });
    }

    let range = parsed.tensor_data_range(tensor, file_bytes.len() as u64)?;
    let bytes = &file_bytes[range.start as usize..range.end as usize];
    let bytes_per_row = (elements_per_row / layout.block_elements) * layout.block_bytes;
    let chunk_bytes = (u64::from(embedding) * bytes_per_row) as usize;

    let (b_bytes, rest) = bytes.split_at(chunk_bytes);
    let (c_bytes, x_bytes) = rest.split_at(chunk_bytes);

    for (suffix, chunk) in [("b", b_bytes), ("c", c_bytes), ("x", x_bytes)] {
        let chunk_name = format!("{name}.{suffix}");
        match tensor.ggml_type {
            GgmlType::F32 => match aligned_f32_view(chunk) {
                Some(view) => state.packed.push((chunk_name, QuantizedBlock::Float32(view))),
                None => {
                    let owned = reinterpret_f32(chunk);
                    state.resident_bytes += owned.len() * core::mem::size_of::<f32>();
                    state.owned.push((chunk_name, owned));
                }
            },
            GgmlType::Q4_K => state.packed.push((chunk_name, QuantizedBlock::Q4K(chunk))),
            GgmlType::Q5_K => state.packed.push((chunk_name, QuantizedBlock::Q5K(chunk))),
            GgmlType::Q6_K => state.packed.push((chunk_name, QuantizedBlock::Q6K(chunk))),
            other => return Err(InteropError::UnrepresentableGgmlType { tensor: name, ggml_type: other }),
        }
    }
    Ok(())
}

/// Runs [`crate::bind::bind_dense`]/[`bind_matmul_weight`]/
/// [`bind_lfm2_shortconv_in_proj`] over every one of `architecture`'s
/// `block_count` layers -- [`crate::bind::bind_all_weights`]'s
/// hybrid-checkpoint counterpart. Two real-checkpoint naming quirks this
/// function papers over at bind time rather than in
/// [`lfm2_forward_program_with_experts`] itself (that program's own `Input`
/// names, `output_norm.weight`/`output.weight`, match every OTHER
/// checkpoint this crate has bound): this checkpoint ties its output
/// projection to `token_embd.weight` (no separate `output.weight` tensor
/// exists on disk) and names its final norm `token_embd_norm.weight` (no
/// `output_norm.weight` tensor exists either) -- confirmed via `strings` on
/// the real file, neither name present. [`bind_dense_as`]/
/// [`bind_matmul_weight_as`] bind the real on-disk tensor under the
/// program's expected alias, the same "the binder papers over a naming
/// difference, the program never learns about it" shape
/// [`bind_lfm2_shortconv_in_proj`] itself uses.
///
/// # Errors
///
/// Whatever [`crate::bind::bind_dense`]/[`bind_matmul_weight`]/
/// [`bind_lfm2_shortconv_in_proj`]/[`bind_moe_expert_weights`] can fail
/// with.
pub(crate) fn bind_lfm2_weights<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    architecture: &Lfm2Architecture,
) -> Result<BoundWeights<'file>, InteropError> {
    let mut state = BoundWeights { resident_bytes: file_bytes.len(), owned: Vec::new(), packed: Vec::new(), packed_owned: Vec::new() };

    let embedding = architecture.embedding as usize;
    let feed_forward = architecture.feed_forward as usize;
    let expert_feed_forward = architecture.expert_feed_forward as usize;
    let vocab = architecture.vocab as usize;
    let kv_dim = architecture.kv_heads as usize * architecture.head_dim as usize;

    bind_dense_as(parsed, file_bytes, "token_embd.weight", "token_embd.weight".into(), &mut state)?;

    for (layer, kind) in architecture.layer_kinds.iter().enumerate() {
        let layer = layer as u32;
        bind_dense_as(parsed, file_bytes, &format!("blk.{layer}.attn_norm.weight"), format!("blk.{layer}.attn_norm.weight"), &mut state)?;
        bind_dense_as(parsed, file_bytes, &format!("blk.{layer}.ffn_norm.weight"), format!("blk.{layer}.ffn_norm.weight"), &mut state)?;

        match kind {
            LayerKind::Attention => {
                bind_matmul_weight(parsed, file_bytes, format!("blk.{layer}.attn_q.weight"), embedding, embedding, &mut state)?;
                bind_matmul_weight(parsed, file_bytes, format!("blk.{layer}.attn_k.weight"), kv_dim, embedding, &mut state)?;
                bind_matmul_weight(parsed, file_bytes, format!("blk.{layer}.attn_v.weight"), kv_dim, embedding, &mut state)?;
                bind_matmul_weight(parsed, file_bytes, format!("blk.{layer}.attn_output.weight"), embedding, embedding, &mut state)?;
                bind_dense_as(
                    parsed,
                    file_bytes,
                    &format!("blk.{layer}.attn_q_norm.weight"),
                    format!("blk.{layer}.attn_q_norm.weight"),
                    &mut state,
                )?;
                bind_dense_as(
                    parsed,
                    file_bytes,
                    &format!("blk.{layer}.attn_k_norm.weight"),
                    format!("blk.{layer}.attn_k_norm.weight"),
                    &mut state,
                )?;
            }
            LayerKind::ShortConv => {
                bind_lfm2_shortconv_in_proj(parsed, file_bytes, layer, architecture.embedding, &mut state)?;
                bind_dense_as(
                    parsed,
                    file_bytes,
                    &format!("blk.{layer}.shortconv.conv.weight"),
                    format!("blk.{layer}.shortconv.conv.weight"),
                    &mut state,
                )?;
                bind_matmul_weight(parsed, file_bytes, format!("blk.{layer}.shortconv.out_proj.weight"), embedding, embedding, &mut state)?;
            }
        }

        if layer < architecture.leading_dense_block_count {
            bind_matmul_weight(parsed, file_bytes, format!("blk.{layer}.ffn_gate.weight"), feed_forward, embedding, &mut state)?;
            bind_matmul_weight(parsed, file_bytes, format!("blk.{layer}.ffn_up.weight"), feed_forward, embedding, &mut state)?;
            bind_matmul_weight(parsed, file_bytes, format!("blk.{layer}.ffn_down.weight"), embedding, feed_forward, &mut state)?;
        } else {
            let expert_count = architecture.expert_count;
            bind_matmul_weight(parsed, file_bytes, format!("blk.{layer}.ffn_gate_inp.weight"), expert_count as usize, embedding, &mut state)?;
            for (projection, out_dim, in_dim) in [
                ("ffn_gate", expert_feed_forward, embedding),
                ("ffn_up", expert_feed_forward, embedding),
                ("ffn_down", embedding, expert_feed_forward),
            ] {
                bind_moe_expert_weights(parsed, file_bytes, layer, projection, expert_count, out_dim, in_dim, &mut state)?;
            }
            bind_dense_as(
                parsed,
                file_bytes,
                &format!("blk.{layer}.exp_probs_b.bias"),
                format!("blk.{layer}.exp_probs_b.bias"),
                &mut state,
            )?;
        }
    }

    bind_dense_as(parsed, file_bytes, "token_embd_norm.weight", "output_norm.weight".into(), &mut state)?;
    bind_matmul_weight_as(parsed, file_bytes, "token_embd.weight", "output.weight".into(), vocab, embedding, &mut state)?;
    Ok(state)
}

/// One call's worth of position-dependent `Input`s
/// [`lfm2_forward_program_with_experts`] needs beyond the model weights --
/// [`crate::generate::PositionInputs`]'s prefill-only, always-starts-at-0
/// counterpart: this program has no key/value cache to offset positions
/// against, so every call re-derives RoPE angles for absolute positions
/// `0..ids.len()`, never a `start_position` offset.
struct Lfm2PositionInputs {
    ids_f32: Vec<f32>,
    epsilon: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

fn build_lfm2_position_inputs(ids: &[u32], head_dim: u32, rope_freq_base: f32, rms_epsilon: f32) -> Lfm2PositionInputs {
    let pairs = head_dim as usize / 2;
    let ids_f32: Vec<f32> = ids.iter().map(|&id| id as f32).collect();
    let epsilon = vec![rms_epsilon; ids.len()];

    let mut cos = vec![0.0f32; ids.len() * pairs];
    let mut sin = vec![0.0f32; ids.len() * pairs];
    for (position, _) in ids.iter().enumerate() {
        for pair in 0..pairs {
            let theta = position as f32 * rope_freq_base.powf(-((2 * pair) as f32) / (head_dim as f32));
            cos[position * pairs + pair] = theta.cos();
            sin[position * pairs + pair] = theta.sin();
        }
    }

    Lfm2PositionInputs { ids_f32, epsilon, cos, sin }
}

/// Binds `architecture`'s weights, builds
/// [`lfm2_forward_program_with_experts`] once, then greedily generates up
/// to `max_new_tokens` tokens -- one full re-prefill of the growing
/// sequence per step, since this program's own scope is prefill-only (no
/// key/value or convolution-state cache to carry a `new_count == 1` step
/// against; see [`lfm2_forward_program_with_experts`]'s own doc). Stops
/// early on the vocab's own `eos_token_id`, matching every other decode
/// loop in this crate.
///
/// # Errors
///
/// Whatever `bind_lfm2_weights`, [`lfm2_forward_program_with_experts`],
/// tokenizing `prompt`, or evaluating the program can fail with.
#[allow(clippy::too_many_arguments)]
pub fn run_lfm2_prefill(
    parsed: &ParsedGguf,
    file_bytes: &[u8],
    architecture: &Lfm2Architecture,
    vocab: &Vocab,
    prompt: &str,
    max_new_tokens: usize,
) -> Result<(Vec<u32>, String), InteropError> {
    let weights = bind_lfm2_weights(parsed, file_bytes, architecture)?;
    let (program, logits_root) = lfm2_forward_program_with_experts(
        architecture.vocab,
        architecture.embedding,
        architecture.feed_forward,
        architecture.expert_feed_forward,
        architecture.query_heads,
        architecture.kv_heads,
        architecture.head_dim,
        architecture.block_count,
        architecture.expert_count,
        architecture.expert_used_count,
        architecture.leading_dense_block_count,
        architecture.l_cache,
        &architecture.layer_kinds,
    )?;

    let mut ids = proxima_tokenizer::encode_with_bos_eos(prompt, vocab, vocab.add_bos_token().unwrap_or(true), vocab.add_eos_token().unwrap_or(false))?;
    let vocab_size = architecture.vocab as usize;
    let eos_id = vocab.eos_token_id();
    let mut free_buffers: Vec<Vec<f32>> = Vec::new();
    let mut validated_weight_nodes: Option<BTreeSet<proxima_tensor::op::NodeId>> = None;

    for _ in 0..max_new_tokens {
        let inputs = build_lfm2_position_inputs(&ids, architecture.head_dim, architecture.rope_freq_base, architecture.rms_epsilon);

        let mut named_blocks: Vec<(&str, QuantizedBlock)> = Vec::with_capacity(weights.owned.len() + weights.packed.len() + 3);
        named_blocks.push(("ids", QuantizedBlock::Float32(inputs.ids_f32.as_slice())));
        for (name, data) in &weights.owned {
            named_blocks.push((name.as_str(), QuantizedBlock::Float32(data.as_slice())));
        }
        for (name, block) in &weights.packed {
            named_blocks.push((name.as_str(), *block));
        }
        named_blocks.push(("eps", QuantizedBlock::Float32(inputs.epsilon.as_slice())));
        named_blocks.push(("rope_cos", QuantizedBlock::Float32(inputs.cos.as_slice())));
        named_blocks.push(("rope_sin", QuantizedBlock::Float32(inputs.sin.as_slice())));

        let symbols = [ids.len() as u64];
        let evaluated = evaluate_quantized_named_with_scratch(
            &program,
            &symbols,
            &named_blocks,
            &[logits_root],
            &mut free_buffers,
            &mut validated_weight_nodes,
        )?;
        let (logits, _shape) = evaluated.get(logits_root).ok_or(InteropError::MissingEvaluatedNode { node: logits_root })?;
        let last_position = &logits[(ids.len() - 1) * vocab_size..ids.len() * vocab_size];

        let mut next_token = 0u32;
        let mut best = f32::NEG_INFINITY;
        for (token, logit) in last_position.iter().enumerate() {
            if *logit > best {
                best = *logit;
                next_token = token as u32;
            }
        }

        ids.push(next_token);
        if eos_id == Some(next_token) {
            break;
        }
    }

    let text = proxima_tokenizer::decode(&ids, vocab)?;
    Ok((ids, text))
}

/// One forward pass over `ids` (no generation loop) -- [`run_lfm2_prefill`]'s
/// cross-oracle counterpart, mirroring [`crate::generate::LoadedModel::forward_logits`]/
/// [`crate::generate::LoadedModel::forward_node_values`]'s dense-checkpoint
/// shape for this hybrid checkpoint's own GGUF bind + program build. Returns
/// the LAST position's full-vocab logits, plus one raw evaluated buffer per
/// entry in `extra_node_ids`, in the same order.
///
/// `extra_node_ids` are typically derived by building
/// [`lfm2_forward_program_with_experts`] at two adjacent `block_count`s and
/// diffing their `Op` sequences (`smollm2_layer_oracle_diff.rs`'s own
/// technique) -- the id-is-index invariant that relies on holds here
/// identically, since a shallower build only ever appends nodes to the
/// deeper one's own prefix.
///
/// # Errors
///
/// Whatever `bind_lfm2_weights`/[`lfm2_forward_program_with_experts`]/
/// evaluating the program can fail with, plus
/// [`InteropError::MissingEvaluatedNode`] if the evaluator's output is
/// missing the logits root or one of `extra_node_ids` -- an
/// interpreter/program-construction invariant violation, never a caller
/// mistake.
pub fn lfm2_forward_values(
    parsed: &ParsedGguf,
    file_bytes: &[u8],
    architecture: &Lfm2Architecture,
    ids: &[u32],
    extra_node_ids: &[NodeId],
) -> Result<(Vec<f32>, Vec<Vec<f32>>), InteropError> {
    let weights = bind_lfm2_weights(parsed, file_bytes, architecture)?;
    let (program, logits_root) = lfm2_forward_program_with_experts(
        architecture.vocab,
        architecture.embedding,
        architecture.feed_forward,
        architecture.expert_feed_forward,
        architecture.query_heads,
        architecture.kv_heads,
        architecture.head_dim,
        architecture.block_count,
        architecture.expert_count,
        architecture.expert_used_count,
        architecture.leading_dense_block_count,
        architecture.l_cache,
        &architecture.layer_kinds,
    )?;

    let inputs = build_lfm2_position_inputs(ids, architecture.head_dim, architecture.rope_freq_base, architecture.rms_epsilon);
    let vocab_size = architecture.vocab as usize;

    let mut named_blocks: Vec<(&str, QuantizedBlock)> = Vec::with_capacity(weights.owned.len() + weights.packed.len() + 3);
    named_blocks.push(("ids", QuantizedBlock::Float32(inputs.ids_f32.as_slice())));
    for (name, data) in &weights.owned {
        named_blocks.push((name.as_str(), QuantizedBlock::Float32(data.as_slice())));
    }
    for (name, block) in &weights.packed {
        named_blocks.push((name.as_str(), *block));
    }
    named_blocks.push(("eps", QuantizedBlock::Float32(inputs.epsilon.as_slice())));
    named_blocks.push(("rope_cos", QuantizedBlock::Float32(inputs.cos.as_slice())));
    named_blocks.push(("rope_sin", QuantizedBlock::Float32(inputs.sin.as_slice())));

    let symbols = [ids.len() as u64];
    let mut free_buffers: Vec<Vec<f32>> = Vec::new();
    let mut validated_weight_nodes: Option<BTreeSet<NodeId>> = None;

    let mut outputs: Vec<NodeId> = vec![logits_root];
    outputs.extend_from_slice(extra_node_ids);

    let evaluated = evaluate_quantized_named_with_scratch(
        &program,
        &symbols,
        &named_blocks,
        &outputs,
        &mut free_buffers,
        &mut validated_weight_nodes,
    )?;

    let (logits, _shape) = evaluated.get(logits_root).ok_or(InteropError::MissingEvaluatedNode { node: logits_root })?;
    let last_position = logits[(ids.len() - 1) * vocab_size..ids.len() * vocab_size].to_vec();

    let mut extras = Vec::with_capacity(extra_node_ids.len());
    for &node in extra_node_ids {
        let (values, _shape) = evaluated.get(node).ok_or(InteropError::MissingEvaluatedNode { node })?;
        extras.push(values.to_vec());
    }

    Ok((last_position, extras))
}

#[cfg(all(test, feature = "std"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use proxima_gguf::quant::q4_k;
    use proxima_gguf::{GgmlType as WireType, GgufModel, TensorPayload, write_complete};

    use super::*;
    use crate::error::InteropError;

    fn dims(values: &[u64]) -> arrayvec::ArrayVec<u64, { proxima_gguf::tensor::MAX_DIMS }> {
        values.iter().copied().collect()
    }

    /// One real Q4_K-quantized fused `in_proj` tensor, `embedding = 256`
    /// (exactly one `Q4_K` super-block per row, `QK_K = 256`): row `r`'s
    /// 256 elements are all `r as f32`, so a correct `B`/`C`/`x` row split
    /// reconstructs row ranges `0..256`, `256..512`, `512..768`
    /// respectively -- an off-by-one-chunk split would instead reconstruct
    /// an adjacent, distinguishable range.
    fn quantized_fused_in_proj(embedding: u32) -> (ParsedGguf, Vec<u8>) {
        let rows = 3 * embedding as usize;
        let mut flat = vec![0.0f32; rows * embedding as usize];
        for row in 0..rows {
            let value = row as f32;
            flat[row * embedding as usize..(row + 1) * embedding as usize].fill(value);
        }
        let block_count = (rows * embedding as usize) / q4_k::QK_K;
        let mut quantized = vec![0u8; block_count * q4_k::BLOCK_BYTES];
        q4_k::quantize(&flat, &mut quantized).expect("quantize a real constant-per-row matrix");

        let model = GgufModel {
            version: 3,
            metadata: alloc::vec![("general.architecture".to_string(), proxima_gguf::value::MetadataValue::String("lfm2moe".to_string()))],
            tensors: alloc::vec![TensorPayload {
                name: "blk.0.shortconv.in_proj.weight".to_string(),
                dims: dims(&[u64::from(embedding), 3 * u64::from(embedding)]),
                ggml_type: WireType::Q4_K,
                data: quantized.as_slice(),
            }],
        };
        let file_bytes = write_complete(&model).expect("writes a real gguf file");
        let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parses it back");
        (parsed, file_bytes)
    }

    fn dequantize_chunk(chunk: &QuantizedBlock, elements: usize) -> Vec<f32> {
        match chunk {
            QuantizedBlock::Q4K(bytes) => {
                let mut output = vec![0.0f32; elements];
                q4_k::dequantize(bytes, &mut output).expect("dequantize a split chunk");
                output
            }
            other => panic!("expected a Q4K chunk, got {other:?}"),
        }
    }

    /// The real split, proved against real quantized bytes: each of
    /// `b`/`c`/`x` dequantizes to its own contiguous, distinguishable row
    /// range of the source matrix (`0..256`, `256..512`, `512..768`) --
    /// never a mid-range mix, which an off-by-one row count or a
    /// mid-block byte offset would produce instead.
    #[test]
    fn splits_a_real_q4_k_in_proj_into_its_three_row_ranges() {
        let embedding = 256u32;
        let (parsed, file_bytes) = quantized_fused_in_proj(embedding);
        let mut state = BoundWeights { resident_bytes: 0, owned: Vec::new(), packed: Vec::new(), packed_owned: Vec::new() };

        bind_lfm2_shortconv_in_proj(&parsed, &file_bytes, 0, embedding, &mut state).expect("split a real q4_k in_proj");

        assert_eq!(state.packed.len(), 3, "b/c/x each bind packed for a Q4_K source");
        let elements = embedding as usize * embedding as usize;

        let (b_name, b_block) = &state.packed[0];
        let (c_name, c_block) = &state.packed[1];
        let (x_name, x_block) = &state.packed[2];
        assert_eq!(b_name, "blk.0.shortconv.in_proj.weight.b");
        assert_eq!(c_name, "blk.0.shortconv.in_proj.weight.c");
        assert_eq!(x_name, "blk.0.shortconv.in_proj.weight.x");

        let b_values = dequantize_chunk(b_block, elements);
        let c_values = dequantize_chunk(c_block, elements);
        let x_values = dequantize_chunk(x_block, elements);

        for row in 0..embedding as usize {
            let b_row = &b_values[row * embedding as usize..(row + 1) * embedding as usize];
            let c_row = &c_values[row * embedding as usize..(row + 1) * embedding as usize];
            let x_row = &x_values[row * embedding as usize..(row + 1) * embedding as usize];
            for &value in b_row {
                assert!((value - row as f32).abs() < 0.5, "b row {row} reconstructed {value}, want ~{row}");
            }
            for &value in c_row {
                let expected = (embedding as usize + row) as f32;
                assert!((value - expected).abs() < 0.5, "c row {row} reconstructed {value}, want ~{expected}");
            }
            for &value in x_row {
                let expected = (2 * embedding as usize + row) as f32;
                assert!((value - expected).abs() < 0.5, "x row {row} reconstructed {value}, want ~{expected}");
            }
        }
    }

    /// The block-boundary arithmetic this split relies on, stated as
    /// assertions rather than prose: `Q4_K`'s `256`-element super-block
    /// never spans two rows of a `2048`-wide real checkpoint row (`8`
    /// whole blocks per row), so every chunk boundary this function
    /// computes lands on an exact multiple of `block_bytes`.
    #[test]
    fn real_checkpoint_row_width_is_a_whole_number_of_q4_k_blocks() {
        let embedding = 2048u64;
        assert_eq!(embedding % q4_k::QK_K as u64, 0, "2048 must be a whole multiple of Q4_K's 256-element block");
        let blocks_per_row = embedding / q4_k::QK_K as u64;
        assert_eq!(blocks_per_row, 8);
        let bytes_per_row = blocks_per_row * q4_k::BLOCK_BYTES as u64;
        assert_eq!(bytes_per_row, 1152, "8 blocks * 144 bytes/block");
        assert_eq!(embedding * bytes_per_row, 2_359_296, "the b/c boundary, an exact multiple of 144");
        assert_eq!(2 * embedding * bytes_per_row, 4_718_592, "the c/x boundary, an exact multiple of 144");
        assert_eq!(3 * embedding * bytes_per_row, 7_077_888, "the whole tensor, an exact multiple of 144");
    }

    /// The defect this shape-check exists to catch: a fused tensor whose
    /// element count disagrees with `3 * embedding * embedding` (a
    /// malformed file, or a caller passing the wrong `embedding`) must
    /// surface as a typed error, never a silent wrong split or a slice
    /// panic.
    #[test]
    fn shape_mismatch_is_a_typed_error_not_a_panic() {
        let embedding = 256u32;
        let (parsed, file_bytes) = quantized_fused_in_proj(embedding);
        let mut state = BoundWeights { resident_bytes: 0, owned: Vec::new(), packed: Vec::new(), packed_owned: Vec::new() };

        let outcome = bind_lfm2_shortconv_in_proj(&parsed, &file_bytes, 0, embedding + 1, &mut state);
        assert!(
            matches!(outcome, Err(InteropError::ShortConvInProjShapeMismatch { .. })),
            "wrong embedding must be a named shape-mismatch error, got {outcome:?}"
        );
    }

    /// [`nonzero_uniform_u32_array`]'s own contract, proved directly: zero
    /// entries (convolution layers) are skipped, and every real, nonzero
    /// entry (attention layers) must agree -- the real checkpoint's own
    /// `[0, 0, 8, 0, 0, 0, 8, ...]` shape.
    #[test]
    fn nonzero_uniform_array_skips_zeros_and_requires_nonzero_agreement() {
        let uniform = nonzero_uniform_u32_array("key", [0, 0, 8, 0, 0, 0, 8].into_iter());
        assert_eq!(uniform.expect("uniform nonzero entries agree"), 8);

        let disagreeing = nonzero_uniform_u32_array("key", [0, 8, 0, 16].into_iter());
        assert!(matches!(disagreeing, Err(InteropError::HeterogeneousNonzeroMetadataArray { distinct_values: 2, .. })));

        let all_zero = nonzero_uniform_u32_array("key", [0, 0, 0].into_iter());
        assert!(matches!(all_zero, Err(InteropError::MissingMetadataKey { .. })));
    }
}
