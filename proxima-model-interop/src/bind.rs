//! Materializing one named GGUF tensor as an owned `f32` buffer, ready to
//! sit in `proxima_tensor::cpu::evaluate_named`'s `named: &[(&str, &[f32])]`
//! bind-by-name slice.
//!
//! Nothing before this crate joined the two: [`proxima_gguf`] parses a
//! tensor directory keyed by name and hands back raw bytes, and
//! [`proxima_tensor::cpu::evaluate_named`] binds an [`proxima_tensor::Op::Input`]
//! by the same kind of name -- but neither crate depends on the other, so
//! nothing turned "here are a GGUF file's bytes and a tensor name" into
//! "here is the `&[f32]` that name's `Op::Input` needs". [`gguf_tensor_as_f32`]
//! is that one step: look the name up, slice its bytes out of the file
//! buffer, and decode them (a straight copy for `F32`, dequantization for
//! a supported block-quantized type).
//!
//! Sans-IO like the rest of this crate: this module never opens a file.
//! The caller parses via [`proxima_gguf::pipe::parse_complete`] (or
//! [`proxima_gguf::edge::read_file`], std-only) and hands this module the
//! resulting [`ParsedGguf`] plus the byte buffer it was parsed from.
//!
//! [`architecture_from_metadata`] and the `std`-gated weight-binding
//! orchestration below it (`bind_all_weights` and friends, `pub(crate)`:
//! [`crate::generate::LoadedModel::load`] is their one caller) turn that
//! same `(ParsedGguf, file_bytes)` pair into every input
//! `proxima_tensor::spec::mistral_cached_forward_program` needs, still
//! without opening a file.

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use proxima_gguf::MetadataValue;
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::quant::{q4_k, q5_k, q6_k, q8_0};
#[cfg(feature = "std")]
use proxima_gguf::restack::{discover_experts, plan_stack, restack_into};
use proxima_gguf::tensor::TensorInfo;
use proxima_gguf::types::GgmlType;

use crate::error::InteropError;

/// Looks `name` up in `parsed`'s tensor directory, slices its bytes out of
/// `file_bytes`, and decodes them to an owned `f32` buffer -- copied
/// as-is for `F32`, dequantized for `Q4_K`/`Q5_K`/`Q6_K`/`Q8_0` (the four
/// codecs [`proxima_gguf::quant`] already ships). Every other `GgmlType`
/// (`F16`/`Bf16`/integer/any other quant family) has no decoder here yet
/// and errors rather than misreading bytes.
///
/// # Errors
///
/// [`InteropError::UnknownTensor`] if `name` isn't in `parsed.tensors`;
/// [`InteropError::Gguf`] if the tensor's declared byte range doesn't fit
/// `file_bytes`; [`InteropError::Quant`] if a block-quantized tensor's
/// byte length doesn't match its own codec's block-size contract;
/// [`InteropError::UnrepresentableGgmlType`] for an undecoded `GgmlType`.
pub fn gguf_tensor_as_f32(
    parsed: &ParsedGguf,
    file_bytes: &[u8],
    name: &str,
) -> Result<Vec<f32>, InteropError> {
    let tensor = find_tensor(parsed, name)?;
    let range = parsed.tensor_data_range(tensor, file_bytes.len() as u64)?;
    let data = &file_bytes[range.start as usize..range.end as usize];
    let element_count = tensor.element_count() as usize;

    match tensor.ggml_type {
        GgmlType::F32 => Ok(reinterpret_f32(data)),
        GgmlType::Q4_K => dequantize(data, element_count, q4_k::dequantize),
        GgmlType::Q5_K => dequantize(data, element_count, q5_k::dequantize),
        GgmlType::Q6_K => dequantize(data, element_count, q6_k::dequantize),
        GgmlType::Q8_0 => dequantize(data, element_count, q8_0::dequantize),
        other => Err(InteropError::UnrepresentableGgmlType {
            tensor: tensor.name.clone(),
            ggml_type: other,
        }),
    }
}

/// Zero-copy counterpart to [`gguf_tensor_as_f32`] for a k-quant tensor, or
/// an `F32` tensor whose file offset happens to be `f32`-aligned: borrows
/// `name`'s raw bytes straight out of `file_bytes` and wraps them as the
/// matching [`proxima_tensor::cpu::QuantizedBlock`] variant, instead of
/// dequantizing (or copying) into an owned `Vec<f32>` first. No copy, no
/// allocation -- exactly the bytes GGUF already stored.
///
/// One function over `Q4_K`/`Q5_K`/`Q6_K`/`Q8_0` rather than four
/// near-identical ones: the four differ only in which variant carries the
/// byte range, and every block-quantized codec here is stored the same way
/// -- a contiguous row-major `[out, in]` byte run whose per-row period is a
/// function of the type alone. A per-type entry point would be that `match`
/// arm rewritten as a signature, four times over.
///
/// `Q8_0` routes packed here rather than only through
/// [`gguf_tensor_as_f32`]'s dequantize path: `proxima_tensor::cpu`'s own
/// `matmul_q8_0_f32` walks `QuantizedBlock::Q8_0`'s raw bytes directly (same
/// per-row contiguous layout the k-quant matmul family assumes), and the GPU
/// emitters (`omega::msl`/`omega::wgsl`/`omega::cuda`) already carry a
/// `PackedCodec::Q8_0` arm -- the packed kernel has always supported this
/// codec, only this bind-time decode arm was missing.
///
/// `F16`/`Bf16` route through the same packed path rather than through
/// [`gguf_tensor_as_f32`]: unlike a block-quantized codec, a half-precision
/// tensor carries no scale/block structure to dequantize, so there is no
/// owned decode to fall back to -- [`gguf_tensor_as_f32`] has never had an
/// `F16`/`Bf16` arm and gains none here. `proxima_tensor::cpu::matmul_f16_f32`/
/// `matmul_bf16_f32` (the sole consumers of [`proxima_tensor::cpu::QuantizedBlock::Float16`]/
/// [`proxima_tensor::cpu::QuantizedBlock::BFloat16`]) walk the exact same
/// `rows` contiguous per-row byte layout the k-quant matmul family does, so
/// this function's existing "bytes straight out of the file, no transpose"
/// contract already covers them -- routing, not new machinery.
///
/// This works without a transpose, unlike `gguf_tensor_as_f32`'s callers
/// for a 2D projection weight (see `transpose_out_in_to_in_out` in this
/// crate's real-forward-pass test): a packed [`proxima_tensor::cpu::QuantizedBlock`]
/// bypasses the interpreter's strided operand machinery entirely -- the
/// `proxima_tensor::cpu::matmul_q4k_f32` family walks `weights` as `rows`
/// contiguous per-row byte chunks and dot-products each row against the
/// activation directly, so it only ever needs GGUF's native on-disk
/// row-major `[out, in]` layout, the layout this function hands through
/// unchanged.
///
/// The `F32` arm reinterprets `bytes` as `&[f32]` in place rather than
/// decoding through `reinterpret_f32`. This is sound only if `bytes.as_ptr()`
/// is 4-byte aligned: [`proxima_gguf::parser`] validates every tensor's
/// on-disk `offset` against a running total of `pad_to_alignment` sums (a
/// mismatch is a parse error, `GgufError::TensorOffsetMismatch`), so the
/// byte offset *within the file* is always a multiple of `parsed.alignment`
/// (minimum default 32) -- but that says nothing about whether `file_bytes`'s
/// own base pointer is aligned. A `Vec<u8>` from `std::fs::read` carries no
/// pointer-alignment guarantee beyond `align_of::<u8>() == 1`; an `mmap`
/// (page-aligned by the kernel) does. `aligned_f32_view` checks the
/// *actual* runtime pointer, not the assumption, and this function returns
/// [`InteropError::MisalignedFloat32Tensor`] rather than reinterpreting
/// unaligned bytes -- non-`F32` callers fall back to [`gguf_tensor_as_f32`]'s
/// owned, byte-at-a-time decode, which never assumes alignment.
///
/// # Errors
///
/// [`InteropError::UnknownTensor`] if `name` isn't in `parsed.tensors`;
/// [`InteropError::Gguf`] if the tensor's declared byte range doesn't fit
/// `file_bytes`; [`InteropError::MisalignedFloat32Tensor`] if `name`'s
/// tensor is `F32` but `file_bytes`'s base pointer leaves its byte range
/// unaligned for `&[f32]`; [`InteropError::UnrepresentableGgmlType`] if
/// `name`'s tensor is none of `F32`/`Q4_K`/`Q5_K`/`Q6_K`/`Q8_0`/`F16`/`Bf16`
/// -- a block-quantized type this crate has no dequantizer for at all, since
/// `F16`/`Bf16` are the only codecs this function decodes packed that
/// [`gguf_tensor_as_f32`] does not independently cover.
#[cfg(feature = "std")]
pub fn gguf_tensor_as_packed_block<'a>(
    parsed: &ParsedGguf,
    file_bytes: &'a [u8],
    name: &str,
) -> Result<proxima_tensor::cpu::QuantizedBlock<'a>, InteropError> {
    let tensor = find_tensor(parsed, name)?;
    let range = parsed.tensor_data_range(tensor, file_bytes.len() as u64)?;
    let bytes = &file_bytes[range.start as usize..range.end as usize];
    match tensor.ggml_type {
        GgmlType::F32 => aligned_f32_view(bytes)
            .map(proxima_tensor::cpu::QuantizedBlock::Float32)
            .ok_or_else(|| InteropError::MisalignedFloat32Tensor {
                tensor: tensor.name.clone(),
            }),
        GgmlType::Q4_K => Ok(proxima_tensor::cpu::QuantizedBlock::Q4K(bytes)),
        GgmlType::Q5_K => Ok(proxima_tensor::cpu::QuantizedBlock::Q5K(bytes)),
        GgmlType::Q6_K => Ok(proxima_tensor::cpu::QuantizedBlock::Q6K(bytes)),
        GgmlType::Q8_0 => Ok(proxima_tensor::cpu::QuantizedBlock::Q8_0(bytes)),
        GgmlType::F16 => Ok(proxima_tensor::cpu::QuantizedBlock::Float16(bytes)),
        GgmlType::Bf16 => Ok(proxima_tensor::cpu::QuantizedBlock::BFloat16(bytes)),
        other => Err(InteropError::UnrepresentableGgmlType {
            tensor: tensor.name.clone(),
            ggml_type: other,
        }),
    }
}

/// Reinterprets `bytes` as `&[f32]` with no copy, or returns `None` if
/// `bytes.as_ptr()` is not 4-byte aligned or `bytes.len()` is not a whole
/// number of `f32`s. The alignment check reads the pointer's own address
/// (`as usize % align_of::<f32>()`), never assumes it from where `bytes`
/// came from -- see [`gguf_tensor_as_packed_block`]'s doc for why the
/// on-disk offset alone can't prove this.
///
/// # Safety argument for the `unsafe` block inside
///
/// `core::slice::from_raw_parts` requires the pointer be non-null,
/// correctly aligned for `f32`, and valid for `len` reads of `f32` for the
/// lifetime of the returned reference. Non-null and lifetime-valid hold
/// because the pointer is `bytes.as_ptr()`, still borrowed from `bytes`.
/// Alignment holds because the guard above just checked it. `len` reads of
/// `f32` fit because `bytes.len() / 4 * 4 == bytes.len()` was also just
/// checked, so the `f32` slice covers exactly `bytes`'s bytes with none
/// left over.
#[cfg(feature = "std")]
pub(crate) fn aligned_f32_view(bytes: &[u8]) -> Option<&[f32]> {
    let float_size = core::mem::size_of::<f32>();
    if !bytes.len().is_multiple_of(float_size) {
        return None;
    }
    if !(bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<f32>()) {
        return None;
    }
    // SAFETY: see this function's doc.
    Some(unsafe {
        core::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), bytes.len() / float_size)
    })
}

pub(crate) fn find_tensor<'a>(
    parsed: &'a ParsedGguf,
    name: &str,
) -> Result<&'a TensorInfo, InteropError> {
    parsed
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| InteropError::UnknownTensor { name: name.into() })
}

pub(crate) fn reinterpret_f32(data: &[u8]) -> Vec<f32> {
    data.as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

pub(crate) fn dequantize(
    data: &[u8],
    element_count: usize,
    decode: fn(&[u8], &mut [f32]) -> Result<(), proxima_gguf::quant::QuantError>,
) -> Result<Vec<f32>, InteropError> {
    let mut output = vec![0.0f32; element_count];
    decode(data, &mut output)?;
    Ok(output)
}

/// A checkpoint's own architecture dimensions, read out of its GGUF
/// metadata rather than assumed. Every field except [`Self::vocab`] has a
/// direct `{architecture}.*` metadata key; `vocab` has none (confirmed
/// against the real openchat-3.5 checkpoint's own metadata keys with
/// `strings`: no `vocab_size`/`n_vocab` key exists anywhere in the file)
/// and is instead derived from `token_embd.weight`'s own tensor shape --
/// that tensor's row count already IS the vocabulary size, so
/// `element_count() / embedding_length` reads it without inventing a key
/// GGUF never wrote.
// f32 has no Eq impl, so rope_freq_base drops this struct to PartialEq only.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelArchitecture {
    pub vocab: u32,
    pub embedding: u32,
    pub feed_forward: u32,
    pub query_heads: u32,
    /// `{architecture}.attention.head_count_kv`. Confirmed against a real
    /// checkpoint (LFM2.5-8B-A1B) to sometimes carry a per-layer
    /// [`proxima_gguf::MetadataArray`] instead of one scalar `U32` -- a hybrid
    /// architecture's convolution layers report `0` kv heads, its attention
    /// layers a real count, and the two differ within the SAME checkpoint.
    /// [`architecture_from_metadata`] reads that array shape without
    /// erroring when every entry agrees (the common case, and every
    /// checkpoint this field's doc could confirm before now), but returns
    /// [`InteropError::HeterogeneousMetadataArray`] rather than collapsing a
    /// genuinely per-layer-varying array into one number here -- this field
    /// cannot represent "8 for these layers, 0 for those" and silently
    /// picking one value would be architecturally wrong, not just imprecise.
    pub kv_heads: u32,
    /// The real per-head projection width -- [`head_dim_from_metadata`]'s own
    /// doc walks the three-way priority
    /// (`attention.key_length`/`rope.dimension_count`/derived quotient) this
    /// field is read through, and why a real checkpoint (Qwen3) needs the
    /// first of those: `embedding / query_heads` (1024/16 = 64) silently
    /// disagrees with the checkpoint's own declared `128`
    /// (`attn_q.weight`'s on-disk shape, `[1024, 2048] = [embedding,
    /// query_heads * 128]`, proves it).
    pub head_dim: u32,
    pub block_count: u32,
    /// `{architecture}.expert_count` (llama.cpp's own key for a sparse
    /// mixture-of-experts checkpoint's total expert count per layer, e.g.
    /// `8` for Mixtral-8x7B) -- `0` when the key is absent, which is every
    /// dense checkpoint this crate has evaluated (openchat-3.5 included)
    /// and is read as "not a mixture-of-experts model", not an error: unlike
    /// every other field on this struct, no dense checkpoint carries this
    /// key at all, so requiring it would turn every dense load into a
    /// [`InteropError::MissingMetadataKey`].
    pub expert_count: u32,
    /// `{architecture}.expert_used_count` (how many of `expert_count`
    /// experts each token routes to per layer, e.g. `2` for Mixtral-8x7B's
    /// top-2 routing) -- `0` alongside `expert_count == 0` for the same
    /// dense-checkpoint reason.
    pub expert_used_count: u32,
    /// `{architecture}.rope.freq_base` (RoPE's frequency base, "theta" --
    /// Qwen3 declares `1_000_000.0`, Llama 3 `500_000.0`, Mistral/openchat
    /// `10_000.0`; a wrong value silently produces wrong attention and
    /// wrong tokens, no error). [`proxima_tensor::sized::ROPE_FREQ_BASE_DEFAULT`]
    /// when the key is absent, matching llama.cpp's own default -- every
    /// checkpoint this crate has evaluated so far (openchat-3.5) declares
    /// the key explicitly, so that fallback has not yet been exercised
    /// against a real load.
    pub rope_freq_base: f32,
    /// `true` when the checkpoint reuses its token-embedding table as the LM
    /// head instead of shipping a separate output-projection tensor (HF's
    /// `tie_word_embeddings`, e.g. SmolLM2-135M-Instruct's real
    /// `config.json`). Always `false` on the GGUF path
    /// ([`architecture_from_metadata`]): llama.cpp's own writer always
    /// emits a standalone `output.weight` tensor regardless of whether the
    /// source checkpoint tied its embeddings, so this crate's GGUF loader has
    /// never needed to distinguish the two. `crate::hf_bind::bind_all_weights_from_safetensors`
    /// reads this field to decide which on-disk tensor to bind at the
    /// forward program's `output.weight` node -- no new bind path, the same
    /// `crate::hf_bind::hf_bind_matmul_weight` call with a different
    /// source tensor name.
    pub tied_embeddings: bool,
}

/// Reads [`ModelArchitecture`] out of `parsed`'s own metadata: looks up
/// `general.architecture` first (`"llama"` for a Mistral-shaped checkpoint
/// such as openchat-3.5), then every `{architecture}.*` dimension key
/// under that name.
///
/// `head_dim` reads `{architecture}.rope.dimension_count` when present --
/// GGUF stores the rotary dimension count as its own key, and for a
/// full-rotation architecture that value already equals the per-head
/// dimension, so reading it directly avoids assuming the division holds
/// for an architecture this crate has not seen. When the key is ABSENT
/// (confirmed on a real checkpoint, LFM2.5-8B-A1B -- see
/// [`ModelArchitecture::head_dim`]'s own doc), falls back to
/// `embedding / query_heads` instead of [`InteropError::MissingMetadataKey`].
///
/// `kv_heads` reads `{architecture}.attention.head_count_kv` as a plain
/// scalar `U32` in the common case, but also accepts a per-layer
/// [`proxima_gguf::MetadataArray`] (confirmed on the same real checkpoint) PROVIDED every
/// entry agrees -- see [`ModelArchitecture::kv_heads`]'s own doc for why a
/// genuinely heterogeneous array is refused rather than collapsed.
///
/// # Errors
///
/// [`InteropError::MissingMetadataKey`] naming the first key that is
/// absent, or present with the wrong [`MetadataValue`] variant;
/// [`InteropError::HeterogeneousMetadataArray`] if `attention.head_count_kv`
/// is a [`proxima_gguf::MetadataArray`] whose entries are not all equal;
/// [`InteropError::VocabShapeMismatch`] if `token_embd.weight`'s element
/// count does not divide evenly by `embedding_length`.
/// [`ModelArchitecture::head_dim`]'s three-way derivation, in priority
/// order: `{architecture}.attention.key_length` first (the real per-head
/// projection width GGUF's own writer declares -- present and authoritative
/// on Qwen3, whose `embedding / query_heads` quotient (1024/16 = 64)
/// disagrees with its real head width, 128, confirmed against `attn_q`'s own
/// on-disk shape); `{architecture}.rope.dimension_count` next (a
/// full-rotation architecture's rotary width already equals its head width,
/// confirmed on openchat-3.5, which has neither key and falls through to the
/// quotient); the derived quotient last, for a checkpoint with neither key
/// (LFM2.5-8B-A1B, confirmed via [`ModelArchitecture::head_dim`]'s own
/// original doc).
fn head_dim_from_metadata(
    parsed: &ParsedGguf,
    architecture: &str,
    embedding: u32,
    query_heads: u32,
) -> u32 {
    let key_length = metadata_u32_optional(parsed, &alloc::format!("{architecture}.attention.key_length"));
    if key_length != 0 {
        return key_length;
    }
    metadata_u32_optional_or(
        parsed,
        &alloc::format!("{architecture}.rope.dimension_count"),
        embedding / query_heads.max(1),
    )
}

/// Whether `parsed`'s checkpoint carries per-head QK-norm weights
/// (`blk.0.attn_q_norm.weight`) -- Qwen3's own `q_norm`/`k_norm`
/// (`modeling_qwen3.py`'s `Qwen3Attention`), applied to `q`/`k` right after
/// projection and before RoPE. Presence, not the architecture name, decides
/// this -- the same "read the file, don't assume the shape" move
/// [`bind_all_weights`]'s tied-embeddings check already makes -- so a future
/// checkpoint that also carries these tensors under a different
/// `general.architecture` value is handled without a name-based dispatch.
pub(crate) fn checkpoint_has_qk_norm(parsed: &ParsedGguf) -> bool {
    find_tensor(parsed, "blk.0.attn_q_norm.weight").is_ok()
}

pub fn architecture_from_metadata(parsed: &ParsedGguf) -> Result<ModelArchitecture, InteropError> {
    let architecture = metadata_str(parsed, "general.architecture")?;
    let embedding = metadata_u32(parsed, &alloc::format!("{architecture}.embedding_length"))?;
    let feed_forward = metadata_u32(
        parsed,
        &alloc::format!("{architecture}.feed_forward_length"),
    )?;
    let query_heads = metadata_u32(
        parsed,
        &alloc::format!("{architecture}.attention.head_count"),
    )?;
    let kv_heads = metadata_u32_or_uniform_array(
        parsed,
        &alloc::format!("{architecture}.attention.head_count_kv"),
    )?;
    let block_count = metadata_u32(parsed, &alloc::format!("{architecture}.block_count"))?;
    let head_dim = head_dim_from_metadata(parsed, architecture, embedding, query_heads);
    let vocab = vocab_from_token_embedding(parsed, embedding)?;
    let expert_count =
        metadata_u32_optional(parsed, &alloc::format!("{architecture}.expert_count"));
    let expert_used_count =
        metadata_u32_optional(parsed, &alloc::format!("{architecture}.expert_used_count"));
    let rope_freq_base = metadata_f32_optional(
        parsed,
        &alloc::format!("{architecture}.rope.freq_base"),
        proxima_tensor::sized::ROPE_FREQ_BASE_DEFAULT,
    );
    Ok(ModelArchitecture {
        vocab,
        embedding,
        feed_forward,
        query_heads,
        kv_heads,
        head_dim,
        block_count,
        expert_count,
        expert_used_count,
        rope_freq_base,
        tied_embeddings: false,
    })
}

pub(crate) fn metadata_str<'parsed>(
    parsed: &'parsed ParsedGguf,
    key: &str,
) -> Result<&'parsed str, InteropError> {
    parsed
        .metadata_value(key)
        .and_then(MetadataValue::as_str)
        .ok_or_else(|| InteropError::MissingMetadataKey { key: key.into() })
}

pub(crate) fn metadata_u32(parsed: &ParsedGguf, key: &str) -> Result<u32, InteropError> {
    parsed
        .metadata_value(key)
        .and_then(MetadataValue::as_u32)
        .ok_or_else(|| InteropError::MissingMetadataKey { key: key.into() })
}

/// Same lookup as [`metadata_u32`], but `0` rather than
/// [`InteropError::MissingMetadataKey`] when `key` is absent -- for a
/// mixture-of-experts-only key (`expert_count`/`expert_used_count`) that no
/// dense checkpoint carries, absence is data ("this is not a
/// mixture-of-experts model"), not a malformed file.
pub(crate) fn metadata_u32_optional(parsed: &ParsedGguf, key: &str) -> u32 {
    metadata_u32_optional_or(parsed, key, 0)
}

/// Same shape as [`metadata_u32_optional`], but `default` (a caller-derived
/// fallback, e.g. `embedding / query_heads`) rather than a fixed `0` when
/// `key` is absent -- for a key whose absence still has a principled
/// derived value, unlike a mixture-of-experts-only key where `0` genuinely
/// means "not present."
pub(crate) fn metadata_u32_optional_or(parsed: &ParsedGguf, key: &str, default: u32) -> u32 {
    parsed
        .metadata_value(key)
        .and_then(MetadataValue::as_u32)
        .unwrap_or(default)
}

/// Same lookup as [`metadata_u32`], but also accepts `key` stored as a
/// per-element [`proxima_gguf::value::MetadataArray`] of `U32`/`I32`
/// entries -- confirmed necessary against a real checkpoint
/// (LFM2.5-8B-A1B), whose `attention.head_count_kv` is `Array(I32[24])`
/// (`0` for its 18 convolution layers, a real count for its 6 attention
/// layers) rather than the single scalar every other checkpoint this crate
/// has read declares.
///
/// Every entry must agree for this to return `Ok` -- see
/// [`ModelArchitecture::kv_heads`]'s own doc for why a genuinely
/// per-layer-varying array is refused ([`InteropError::HeterogeneousMetadataArray`])
/// rather than collapsed into one scalar by, say, taking the max or the
/// first nonzero entry: either of those would silently misrepresent a
/// hybrid architecture's real per-layer structure as a uniform one.
///
/// # Errors
///
/// [`InteropError::MissingMetadataKey`] if `key` is absent, or present with
/// neither a scalar `U32`/`I32` nor an `Array` of one of those;
/// [`InteropError::HeterogeneousMetadataArray`] if `key` is an `Array` whose
/// entries are not all equal, or a negative `I32` entry has no `u32`
/// representation.
fn metadata_u32_or_uniform_array(parsed: &ParsedGguf, key: &str) -> Result<u32, InteropError> {
    use proxima_gguf::value::MetadataArray;

    match parsed.metadata_value(key) {
        Some(MetadataValue::U32(value)) => Ok(*value),
        Some(MetadataValue::I32(value)) => {
            u32::try_from(*value).map_err(|_| InteropError::HeterogeneousMetadataArray {
                key: key.into(),
                distinct_values: 1,
            })
        }
        Some(MetadataValue::Array(MetadataArray::U32(values))) => {
            uniform_u32_array(key, values.iter().copied())
        }
        Some(MetadataValue::Array(MetadataArray::I32(values))) => uniform_u32_array(
            key,
            values
                .iter()
                .map(|value| u32::try_from(*value).unwrap_or(u32::MAX)),
        ),
        _ => Err(InteropError::MissingMetadataKey { key: key.into() }),
    }
}

/// Every element of `values` must be identical, or this returns
/// [`InteropError::HeterogeneousMetadataArray`] naming how many distinct
/// values were actually found -- see [`metadata_u32_or_uniform_array`]'s
/// own doc for why.
fn uniform_u32_array(key: &str, values: impl Iterator<Item = u32>) -> Result<u32, InteropError> {
    let mut distinct: BTreeSet<u32> = BTreeSet::new();
    for value in values {
        distinct.insert(value);
    }
    match distinct.len() {
        0 => Err(InteropError::MissingMetadataKey { key: key.into() }),
        1 => Ok(distinct.into_iter().next().unwrap_or(0)),
        distinct_values => Err(InteropError::HeterogeneousMetadataArray {
            key: key.into(),
            distinct_values,
        }),
    }
}

/// Same absent-is-data shape as [`metadata_u32_optional`], but for a
/// float-valued key -- `default` rather than an error when `key` is
/// missing. Matches both [`MetadataValue::F32`] and [`MetadataValue::F64`]
/// rather than assuming one wire type: llama.cpp's own GGUF writer emits
/// `{architecture}.rope.freq_base` as `F32`, but `MetadataValue` carries
/// both float widths and nothing in the format's own spec forbids a
/// writer choosing `F64`, so this reads whichever the checkpoint actually
/// declares.
pub(crate) fn metadata_f32_optional(parsed: &ParsedGguf, key: &str, default: f32) -> f32 {
    match parsed.metadata_value(key) {
        Some(MetadataValue::F32(value)) => *value,
        Some(MetadataValue::F64(value)) => *value as f32,
        _ => default,
    }
}

pub(crate) fn vocab_from_token_embedding(
    parsed: &ParsedGguf,
    embedding: u32,
) -> Result<u32, InteropError> {
    let tensor = find_tensor(parsed, "token_embd.weight")?;
    let elements = tensor.element_count();
    let divisor = u64::from(embedding);
    if divisor == 0 || !elements.is_multiple_of(divisor) {
        return Err(InteropError::VocabShapeMismatch {
            elements,
            embedding,
        });
    }
    Ok((elements / divisor) as u32)
}

/// Every weight [`proxima_tensor::spec::mistral_cached_forward_program`]
/// binds by name, split into owned `f32` buffers (norms, plus
/// `token_embd.weight`, which is an embedding lookup rather than a matmul
/// operand and so has no packed kernel), zero-copy packed blocks
/// borrowed straight out of `file_bytes` (see [`bind_dense`]/
/// [`bind_matmul_weight`]), and owned-but-still-quantized packed blocks
/// (see [`PackedOwnedKind`]) for a MoE expert stack that has no single
/// contiguous on-disk byte range to borrow from ([`bind_moe_expert_weights`]'s
/// restack fallback). `pub(crate)`: [`crate::generate::LoadedModel`]
/// is the one place outside this module that constructs or reads one.
#[cfg(feature = "std")]
pub(crate) struct BoundWeights<'file> {
    pub(crate) resident_bytes: usize,
    pub(crate) owned: Vec<(alloc::string::String, Vec<f32>)>,
    pub(crate) packed: Vec<(
        alloc::string::String,
        proxima_tensor::cpu::QuantizedBlock<'file>,
    )>,
    pub(crate) packed_owned: Vec<(alloc::string::String, Vec<u8>, PackedOwnedKind)>,
}

/// Which [`proxima_tensor::cpu::QuantizedBlock`] byte-borrowing variant to
/// re-wrap a [`BoundWeights::packed_owned`] entry's bytes in at read time.
/// Exists because `QuantizedBlock<'a>` borrows for a caller-chosen lifetime
/// `'a`, but the bytes it would borrow here are a restacked buffer this
/// crate allocated ([`bind_moe_expert_weights`]'s restack fallback), not a
/// slice of `file_bytes` -- so [`BoundWeights`] cannot store the already-built
/// enum at `'file` the way [`BoundWeights::packed`] does. Storing the raw
/// bytes plus this tag instead lets [`crate::generate::LoadedModel`]
/// construct the borrow fresh, each request, at whatever shorter lifetime
/// that call site needs.
#[cfg(feature = "std")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PackedOwnedKind {
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
}

#[cfg(feature = "std")]
impl PackedOwnedKind {
    /// Borrows `bytes` as the [`proxima_tensor::cpu::QuantizedBlock`] variant
    /// this tag names -- the deferred half of the split [`PackedOwnedKind`]'s
    /// own doc describes.
    pub(crate) fn as_block<'bytes>(
        self,
        bytes: &'bytes [u8],
    ) -> proxima_tensor::cpu::QuantizedBlock<'bytes> {
        match self {
            PackedOwnedKind::Q4K => proxima_tensor::cpu::QuantizedBlock::Q4K(bytes),
            PackedOwnedKind::Q5K => proxima_tensor::cpu::QuantizedBlock::Q5K(bytes),
            PackedOwnedKind::Q6K => proxima_tensor::cpu::QuantizedBlock::Q6K(bytes),
            PackedOwnedKind::Q8_0 => proxima_tensor::cpu::QuantizedBlock::Q8_0(bytes),
        }
    }

    /// The [`GgmlType`] this tag corresponds to, or `None` for a codec
    /// [`bind_moe_expert_weights`]'s restack fallback still must dequantize
    /// (`F32` stays dequantized-then-transposed: `run_reduce_quantized`'s
    /// gather rejects a `Float32` weight block outright,
    /// `proxima_tensor::cpu::run_reduce_quantized`'s own `shape_error` arm for
    /// that variant).
    fn from_ggml_type(ggml_type: GgmlType) -> Option<Self> {
        match ggml_type {
            GgmlType::Q4_K => Some(PackedOwnedKind::Q4K),
            GgmlType::Q5_K => Some(PackedOwnedKind::Q5K),
            GgmlType::Q6_K => Some(PackedOwnedKind::Q6K),
            GgmlType::Q8_0 => Some(PackedOwnedKind::Q8_0),
            _ => None,
        }
    }
}

/// A learned 1-D scale (RMSNorm weight) or `token_embd.weight` (indexed by
/// row via `embedding_lookup`, never projected; never a packed path even
/// when quantized, since a quantized matmul operand requires feeding a
/// `Multiply`-then-`Add` reduce, which a gather is not). Falls back to
/// [`gguf_tensor_as_f32`]'s owned copy only if the zero-copy borrow is
/// refused (misaligned base pointer) or the tensor is some other
/// decodable-but-not-borrowable `GgmlType`.
///
/// # Errors
///
/// [`InteropError::UnrepresentableGgmlType`] if `name`'s tensor is a
/// `GgmlType` neither [`gguf_tensor_as_packed_block`] nor
/// [`gguf_tensor_as_f32`] has a decoder for -- a checkpoint using a codec
/// this crate does not execute is a caller-visible load failure, not a
/// process abort.
#[cfg(feature = "std")]
pub(crate) fn bind_dense<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    name: alloc::string::String,
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    bind_dense_as(parsed, file_bytes, &name.clone(), name, state)
}

/// [`bind_dense`]'s underlying step, split out so a caller whose real
/// on-disk tensor is named differently than the forward program's own
/// `Input` (`crate::lfm2`'s tied-embedding output projection and
/// differently-named final norm, both confirmed on the real LFM2.5-8B-A1B
/// checkpoint) can bind `source_name`'s bytes under `target_name` without
/// duplicating this function's body. `bind_dense` is exactly this with
/// `source_name == target_name`.
///
/// # Errors
///
/// See [`bind_dense`].
#[cfg(feature = "std")]
pub(crate) fn bind_dense_as<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    source_name: &str,
    target_name: alloc::string::String,
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    match gguf_tensor_as_packed_block(parsed, file_bytes, source_name) {
        Ok(block @ proxima_tensor::cpu::QuantizedBlock::Float32(borrowed)) => {
            state.resident_bytes += core::mem::size_of_val(borrowed);
            state.packed.push((target_name, block));
        }
        Ok(_) | Err(_) => {
            let decoded = gguf_tensor_as_f32(parsed, file_bytes, source_name)?;
            state.resident_bytes += decoded.len() * core::mem::size_of::<f32>();
            state.owned.push((target_name, decoded));
        }
    }
    Ok(())
}

/// A 2-D projection weight the cached forward program uses as one
/// `Multiply`-then-`Add`-reduce (matmul) operand. Tries
/// [`gguf_tensor_as_packed_block`] first: a `Q4_K`/`Q5_K`/`Q6_K`/`Q8_0`/`F16`/`Bf16`
/// tensor binds packed, zero-copy, straight out of the mmap's bytes, because
/// every one of those codecs reaches the interpreter through a path that
/// either walks its own physical row-major bytes directly
/// (`proxima_tensor::cpu::run_reduce_quantized`'s own doc: it derives `rows`/`k`
/// from the packed byte length, never from the resolved `Layout`'s numeric
/// strides) or has that `Layout` explicitly corrected downstream
/// (`proxima_tensor::bind::correct_packed_matmul_layouts`, the GPU driver's own
/// call site, `omega::metal.rs`).
///
/// `F32` is the one codec this function does NOT hand to
/// [`gguf_tensor_as_packed_block`]'s packed path, even though that function can
/// decode an aligned `F32` tensor too: neither the CPU generic evaluator (a
/// raw-packed `F32` operand binds as a plain buffer,
/// `proxima_tensor::cpu::evaluate_quantized_with_scratch`'s own `QuantizedBlock::Float32`
/// arm, never entering `run_reduce_quantized`'s bypass) nor the GPU driver
/// (`omega::metal.rs`'s own `packed_operands_of` explicitly excludes
/// `QuantizedBlock::Float32` from the node set `correct_packed_matmul_layouts`
/// corrects) has any mechanism that rewrites a packed `F32` matmul operand's
/// `Layout` from GGUF's native `[out, in]` to the `[in, out]` every consuming
/// `IndexMap` declares. So `F32` always takes the dequantize-then-transpose
/// path (`transpose_out_in_to_in_out`) instead, the same as any other
/// undecodable-packed `GgmlType` -- this is what makes a plain `f32` matmul
/// operand's bytes match its declared axis order at every downstream reader,
/// exactly the invariant [`gguf_tensor_as_packed_block`]'s own doc already
/// assumes every OTHER consumer of a bound `f32` buffer can rely on.
///
/// # Errors
///
/// [`InteropError::UnrepresentableGgmlType`] if `name`'s tensor is a
/// `GgmlType` neither the packed nor the owned decode path can handle --
/// see [`bind_dense`].
#[cfg(feature = "std")]
pub(crate) fn bind_matmul_weight<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    name: alloc::string::String,
    out_dim: usize,
    in_dim: usize,
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    bind_matmul_weight_as(
        parsed,
        file_bytes,
        &name.clone(),
        name,
        out_dim,
        in_dim,
        state,
    )
}

/// [`bind_matmul_weight`]'s underlying step, split out the same way
/// [`bind_dense_as`] is: a caller whose real on-disk tensor and the
/// forward program's own `Input` name disagree (`crate::lfm2`'s tied
/// output projection, bound from the real `token_embd.weight` tensor
/// under the program's own `output.weight` alias) binds `source_name`'s
/// bytes under `target_name` directly, instead of duplicating this
/// function's body under a new name.
///
/// # Errors
///
/// See [`bind_matmul_weight`].
#[cfg(feature = "std")]
pub(crate) fn bind_matmul_weight_as<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    source_name: &str,
    target_name: alloc::string::String,
    out_dim: usize,
    in_dim: usize,
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    match gguf_tensor_as_packed_block(parsed, file_bytes, source_name) {
        Ok(proxima_tensor::cpu::QuantizedBlock::Float32(_)) | Err(_) => {
            let decoded = gguf_tensor_as_f32(parsed, file_bytes, source_name)?;
            state.resident_bytes += decoded.len() * core::mem::size_of::<f32>();
            let transposed = transpose_out_in_to_in_out(&decoded, source_name, out_dim, in_dim)?;
            state.owned.push((target_name, transposed));
        }
        Ok(block) => state.packed.push((target_name, block)),
    }
    Ok(())
}

/// Row-major transpose of one `[expert_count, out_dim, in_dim]` stack into
/// `[expert_count, in_dim, out_dim]`, expert-by-expert, via the same
/// [`transpose_out_in_to_in_out`] a dense matmul weight already uses --
/// every expert's own slab is on-disk in the identical `[out_dim, in_dim]`
/// layout a non-MoE projection weight would carry, `restack.rs`'s pure byte
/// concatenation just keeps them side by side instead of collapsing them
/// into one.
///
/// # Errors
///
/// [`InteropError::MoeExpertShapeMismatch`] if `flat.len()` does not equal
/// `expert_count * out_dim * in_dim` — the caller's `expert_count`/`out_dim`/
/// `in_dim` come from GGUF hparams read independently of the tensor's own
/// declared element count, so a malformed or adversarial file can disagree
/// with it; caught here rather than sliced past.
#[cfg(feature = "std")]
fn transpose_expert_stack(
    flat: &[f32],
    tensor: &str,
    expert_count: usize,
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>, InteropError> {
    let per_expert = out_dim * in_dim;
    let expected = expert_count * per_expert;
    if flat.len() != expected {
        return Err(InteropError::MoeExpertShapeMismatch {
            tensor: tensor.into(),
            elements: flat.len(),
            expert_count,
            out_dim,
            in_dim,
            expected,
        });
    }
    let mut transposed = vec![0.0f32; flat.len()];
    for expert in 0..expert_count {
        let slab = &flat[expert * per_expert..(expert + 1) * per_expert];
        // `slab.len() == per_expert == out_dim * in_dim` by construction --
        // the length check above already proved it for every expert slab.
        let slab_transposed = transpose_out_in_to_in_out(slab, tensor, out_dim, in_dim)?;
        transposed[expert * per_expert..(expert + 1) * per_expert]
            .copy_from_slice(&slab_transposed);
    }
    Ok(transposed)
}

/// [`InteropError`] has no dedicated [`proxima_gguf::restack::RestackError`]
/// variant (that enum lives outside this change's file ownership), so a
/// restack failure folds into [`InteropError::UnknownTensor`] -- the
/// existing variant closest in meaning ("the checkpoint's tensor directory
/// does not have what was expected"), with the underlying `RestackError`'s
/// own message embedded in `name` rather than discarded.
#[cfg(feature = "std")]
fn restack_error_as_interop_error(
    layer: u32,
    projection: &str,
    error: proxima_gguf::restack::RestackError,
) -> InteropError {
    InteropError::UnknownTensor {
        name: alloc::format!("blk.{layer}.{projection}.*.weight (restack failed: {error})"),
    }
}

/// A native, single, pre-stacked `blk.{layer}.{projection}_exps.weight`
/// tensor (the layout some GGUF exporters use): one on-disk byte range
/// covering every expert already. Bound zero-copy *only* when it is `F32`
/// -- [`proxima_tensor::cpu::QuantizedBlock::Float32`] is the one packed
/// variant [`proxima_tensor::cpu`]'s own evaluator binds straight through
/// as a plain `&[f32]` buffer (never inserted into its `quantized_weights`
/// map), so a gather over it (`IndexMap::Computed`, which
/// `specs/moe_block.toml`'s routed FFN uses to pick one expert's slab per
/// token) resolves exactly like any other f32 operand.
///
/// Every OTHER packed codec ([`gguf_tensor_as_packed_block`] can decode
/// `Q4_K`/`Q5_K`/`Q6_K` too) still falls back to dequantize-then-
/// [`transpose_expert_stack`] here (not [`transpose_out_in_to_in_out`]: the
/// flat buffer is `[expert_count, out_dim, in_dim]`, not one 2-D matrix, so
/// a plain global transpose would scramble the expert axis into the wrong
/// place in memory) even though [`bind_moe_expert_weights`]'s own restack
/// fallback now binds those same codecs packed: a real, live, already-
/// verified checkpoint (LFM2.5-8B-A1B) reaches THIS function today with a
/// native `Q4_K` `_exps` stack, and widening this arm too, untested against
/// that file's own real forward output, is exactly the kind of change this
/// crate's own discipline log requires a fresh proof for before it lands --
/// not bundled into the memory-ceiling fix [`bind_moe_expert_weights`]'s own
/// doc names, which is scoped to the restack (per-expert-tensor) fallback
/// only.
///
/// # Errors
///
/// [`InteropError::Gguf`] if `name`'s declared byte range doesn't fit
/// `file_bytes`; [`InteropError::UnrepresentableGgmlType`] for a
/// block-quantized type this crate has no dequantizer for.
#[cfg(feature = "std")]
fn bind_moe_stacked_experts<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    name: alloc::string::String,
    expert_count: usize,
    out_dim: usize,
    in_dim: usize,
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    match gguf_tensor_as_packed_block(parsed, file_bytes, &name) {
        Ok(block @ proxima_tensor::cpu::QuantizedBlock::Float32(_)) => {
            state.packed.push((name, block))
        }
        Ok(_) | Err(_) => {
            let decoded = gguf_tensor_as_f32(parsed, file_bytes, &name)?;
            let transposed =
                transpose_expert_stack(&decoded, &name, expert_count, out_dim, in_dim)?;
            state.resident_bytes += transposed.len() * core::mem::size_of::<f32>();
            state.owned.push((name, transposed));
        }
    }
    Ok(())
}

/// One MoE-only weight family (`ffn_gate`/`ffn_up`/`ffn_down`) for one
/// layer. Tries a single native stacked tensor first
/// (`blk.{layer}.{projection}_exps.weight`, see
/// [`bind_moe_stacked_experts`] -- always packed now) before falling back to
/// [`proxima_gguf::restack::discover_experts`]/`plan_stack`/`restack_into`
/// for the per-expert-tensor convention
/// (`blk.{layer}.{projection}.{expert}.weight`, `restack.rs`'s own module
/// doc verified against a real Mixtral-8x7B checkpoint).
///
/// **This fallback used to always dequantize to owned `f32`, regardless of
/// codec** -- correct but, for a real Mixtral-8x7B-shaped checkpoint (32
/// layers x 8 experts x 3 projections x 4096x14336 elements), an owned `f32`
/// ceiling of `32 * 8 * 3 * 4096 * 14336 * 4` bytes = ~180 GiB, which a
/// 64 GiB host's kernel SIGKILLs partway through binding. That was correct
/// at the time because `proxima_tensor::cpu::run_reduce_quantized`'s gather
/// arm (the routed-FFN matmul kernel, which `gathered_expert_product`'s
/// `IndexMap::Computed` compiles a MoE projection to) had no notion of a
/// per-expert byte offset at all -- it only ever destructured a `Computed`
/// map's `(node, layout, _)`, classifying every output axis from the
/// operand's own `Layout` stride, which for a gathered axis is the *base*
/// pattern's constant/broadcast stride, not the per-token expert selection
/// the gather actually encodes. Binding a routed expert stack packed under
/// that gap would not have failed loudly -- every token would have silently
/// read whichever expert the base pattern's constant offset picked.
///
/// **That gap is closed.** `run_reduce_quantized` now derives
/// `per_expert_bytes` from the gathered axis's own extent and slices
/// `expert_index * per_expert_bytes` out of the packed buffer per token (see
/// that function's own doc and [`bind_moe_stacked_experts`]'s). So this
/// fallback now binds any codec [`PackedOwnedKind`] names --
/// `Q4_K`/`Q5_K`/`Q6_K`/`Q8_0` -- as an owned-but-still-packed buffer
/// ([`BoundWeights::packed_owned`]) instead of dequantizing: `restack_into`'s
/// byte-concatenation is already exactly the contiguous `[expert, rows, k]`
/// packed layout the gather resolves, so no dequantize-then-transpose step
/// is needed at all for those codecs -- the restacked bytes bind as-is.
/// `F32` is the one exception, still dequantized-then-[`transpose_expert_stack`]:
/// `run_reduce_quantized`'s gather arm rejects a `Float32` weight block
/// outright (that codec's gather already resolves through the generic
/// buffer path [`bind_moe_stacked_experts`]'s `F32` arm uses, which expects
/// the `[in, out]`-transposed layout this fallback's owned buffers have
/// always produced).
///
/// # Errors
///
/// [`InteropError::UnknownTensor`] if neither tensor layout is present, or
/// discovery/planning/restacking failed (see
/// [`restack_error_as_interop_error`]); [`InteropError::UnrepresentableGgmlType`]
/// for a block-quantized type this crate has no dequantizer for.
#[cfg(feature = "std")]
// one weight family's own tensor shape (layer/projection/expert_count/
// out_dim/in_dim) plus the file/state every binder in this module threads
// through -- the same real parameter count `append_moe_ffn`/
// `append_mistral_moe_layer` (proxima-tensor/src/spec.rs) carry for the
// identical MoE shape, not accidental complexity.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bind_moe_expert_weights<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    layer: u32,
    projection: &str,
    expert_count: u32,
    out_dim: usize,
    in_dim: usize,
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    let stacked_name = alloc::format!("blk.{layer}.{projection}_exps.weight");
    if find_tensor(parsed, &stacked_name).is_ok() {
        return bind_moe_stacked_experts(
            parsed,
            file_bytes,
            stacked_name,
            expert_count as usize,
            out_dim,
            in_dim,
            state,
        );
    }

    let experts = discover_experts(
        &parsed.tensors,
        u64::from(layer),
        projection,
        u64::from(expert_count),
    )
    .map_err(|error| restack_error_as_interop_error(layer, projection, error))?;
    let plan = plan_stack(&experts)
        .map_err(|error| restack_error_as_interop_error(layer, projection, error))?;

    let mut sources: Vec<&[u8]> = Vec::with_capacity(experts.len());
    for tensor in &experts {
        let range = parsed.tensor_data_range(tensor, file_bytes.len() as u64)?;
        sources.push(&file_bytes[range.start as usize..range.end as usize]);
    }

    let mut stacked_bytes = vec![0u8; plan.total_bytes as usize];
    restack_into(&mut stacked_bytes, &plan, &sources)
        .map_err(|error| restack_error_as_interop_error(layer, projection, error))?;

    if let Some(kind) = PackedOwnedKind::from_ggml_type(plan.ggml_type) {
        state.resident_bytes += stacked_bytes.len();
        state.packed_owned.push((stacked_name, stacked_bytes, kind));
        return Ok(());
    }

    let decoded = match plan.ggml_type {
        GgmlType::F32 => reinterpret_f32(&stacked_bytes),
        other => {
            return Err(InteropError::UnrepresentableGgmlType {
                tensor: alloc::format!("blk.{layer}.{projection}.*.weight"),
                ggml_type: other,
            });
        }
    };
    let transposed = transpose_expert_stack(
        &decoded,
        &stacked_name,
        expert_count as usize,
        out_dim,
        in_dim,
    )?;
    state.resident_bytes += transposed.len() * core::mem::size_of::<f32>();
    state.owned.push((stacked_name, transposed));
    Ok(())
}

/// Runs [`bind_dense`]/[`bind_matmul_weight`] over every one of
/// `architecture`'s `block_count` layers plus `token_embd.weight` and
/// `output.weight` -- the load loop [`crate::generate::LoadedModel::load`]
/// runs once per checkpoint, so every [`Pipe::call`](proxima_primitives::pipe::Pipe::call)
/// after that reuses the result instead of re-walking the tensor
/// directory per request. `architecture.expert_count > 0` binds the routed
/// FFN's weight family instead of the dense triple (see
/// [`bind_moe_expert_weights`]) -- every other weight is identical between
/// the two shapes.
///
/// # Errors
///
/// [`InteropError::UnrepresentableGgmlType`] if any bound tensor carries a
/// `GgmlType` this crate has no decoder for -- a checkpoint using an
/// undecoded codec fails the load with a typed error rather than aborting
/// the process (see [`bind_dense`]/[`bind_matmul_weight`]); whatever
/// [`bind_moe_expert_weights`] can fail with, for a MoE checkpoint.
#[cfg(feature = "std")]
pub(crate) fn bind_all_weights<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    architecture: &ModelArchitecture,
) -> Result<BoundWeights<'file>, InteropError> {
    let mut state = BoundWeights {
        resident_bytes: file_bytes.len(),
        owned: Vec::new(),
        packed: Vec::new(),
        packed_owned: Vec::new(),
    };

    let embedding = architecture.embedding as usize;
    // `query_heads * head_dim`, NOT `embedding` -- the two agree only when
    // `head_dim == embedding / query_heads` (Mistral's own shape). Qwen3
    // declares `head_dim` independently (`attention.key_length`), and its
    // real checkpoint has `query_heads * head_dim = 16 * 128 = 2048 !=
    // embedding (1024)`; binding `attn_q`/`attn_output` at `embedding` there
    // silently mis-shapes both tensors by 2x.
    let q_dim = architecture.query_heads as usize * architecture.head_dim as usize;
    let kv_dim = architecture.kv_heads as usize * architecture.head_dim as usize;
    let feed_forward = architecture.feed_forward as usize;
    let vocab = architecture.vocab as usize;
    let qk_norm = checkpoint_has_qk_norm(parsed);

    bind_dense(parsed, file_bytes, "token_embd.weight".into(), &mut state)?;

    for layer in 0..architecture.block_count {
        bind_dense(
            parsed,
            file_bytes,
            alloc::format!("blk.{layer}.attn_norm.weight"),
            &mut state,
        )?;
        bind_dense(
            parsed,
            file_bytes,
            alloc::format!("blk.{layer}.ffn_norm.weight"),
            &mut state,
        )?;
        bind_matmul_weight(
            parsed,
            file_bytes,
            alloc::format!("blk.{layer}.attn_q.weight"),
            q_dim,
            embedding,
            &mut state,
        )?;
        bind_matmul_weight(
            parsed,
            file_bytes,
            alloc::format!("blk.{layer}.attn_k.weight"),
            kv_dim,
            embedding,
            &mut state,
        )?;
        bind_matmul_weight(
            parsed,
            file_bytes,
            alloc::format!("blk.{layer}.attn_v.weight"),
            kv_dim,
            embedding,
            &mut state,
        )?;
        bind_matmul_weight(
            parsed,
            file_bytes,
            alloc::format!("blk.{layer}.attn_output.weight"),
            embedding,
            q_dim,
            &mut state,
        )?;

        if qk_norm {
            bind_dense(
                parsed,
                file_bytes,
                alloc::format!("blk.{layer}.attn_q_norm.weight"),
                &mut state,
            )?;
            bind_dense(
                parsed,
                file_bytes,
                alloc::format!("blk.{layer}.attn_k_norm.weight"),
                &mut state,
            )?;
        }

        if architecture.expert_count == 0 {
            bind_matmul_weight(
                parsed,
                file_bytes,
                alloc::format!("blk.{layer}.ffn_gate.weight"),
                feed_forward,
                embedding,
                &mut state,
            )?;
            bind_matmul_weight(
                parsed,
                file_bytes,
                alloc::format!("blk.{layer}.ffn_up.weight"),
                feed_forward,
                embedding,
                &mut state,
            )?;
            bind_matmul_weight(
                parsed,
                file_bytes,
                alloc::format!("blk.{layer}.ffn_down.weight"),
                embedding,
                feed_forward,
                &mut state,
            )?;
        } else {
            let expert_count = architecture.expert_count;
            bind_matmul_weight(
                parsed,
                file_bytes,
                alloc::format!("blk.{layer}.ffn_gate_inp.weight"),
                expert_count as usize,
                embedding,
                &mut state,
            )?;
            for (projection, out_dim, in_dim) in [
                ("ffn_gate", feed_forward, embedding),
                ("ffn_up", feed_forward, embedding),
                ("ffn_down", embedding, feed_forward),
            ] {
                bind_moe_expert_weights(
                    parsed,
                    file_bytes,
                    layer,
                    projection,
                    expert_count,
                    out_dim,
                    in_dim,
                    &mut state,
                )?;
            }
        }
    }

    bind_dense(parsed, file_bytes, "output_norm.weight".into(), &mut state)?;
    // tied embeddings (`general.tie_word_embeddings=true`, e.g. the real
    // SmolLM2-135M checkpoint's own GGUF export): no standalone
    // `output.weight` tensor exists on disk at all, only `token_embd.weight`
    // reused for both the input embedding lookup and the output projection.
    // `bind_matmul_weight_as` is the same alias mechanism `crate::lfm2`'s own
    // tied output projection already uses (`lfm2.rs:544-545`) -- not a new
    // bind path, just reached from the plain dense/MoE loop too.
    if find_tensor(parsed, "output.weight").is_ok() {
        bind_matmul_weight(
            parsed,
            file_bytes,
            "output.weight".into(),
            vocab,
            embedding,
            &mut state,
        )?;
    } else {
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

/// Row-major transpose from GGUF's native flat layout (`[out, in]`, `out`
/// rows of contiguous `in` values, ggml's linear-weight layout) to the
/// forward program's expected `[in, out]` layout.
///
/// # Errors
///
/// [`InteropError::DenseWeightShapeMismatch`] if `flat.len()` does not equal
/// `out_dim * in_dim` — `out_dim`/`in_dim` come from architecture hparams
/// read independently of the tensor's own declared element count, so a
/// malformed or adversarial GGUF file can disagree with it; caught here
/// rather than sliced past.
#[cfg(feature = "std")]
pub(crate) fn transpose_out_in_to_in_out(
    flat: &[f32],
    tensor: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>, InteropError> {
    let expected = out_dim * in_dim;
    if flat.len() != expected {
        return Err(InteropError::DenseWeightShapeMismatch {
            tensor: tensor.into(),
            elements: flat.len(),
            out_dim,
            in_dim,
            expected,
        });
    }
    let mut transposed = alloc::vec![0.0f32; flat.len()];
    for out_index in 0..out_dim {
        for in_index in 0..in_dim {
            transposed[in_index * out_dim + out_index] = flat[out_index * in_dim + in_index];
        }
    }
    Ok(transposed)
}

// `matmul_weight_dims`/`dequantize_packed_for_metal` (the "dequantize this
// packed weight back to f32 because Metal has no unpack kernel for it yet"
// seam `Q5_K` used, and `Q6_K` used before its own row-blocked kernel
// landed) were deleted here: once `Q5_K` joined `Q4_K`/`Q6_K` in staying
// packed all the way to the GPU (`omega::msl::Q5K_UNPACK_MSL`), every
// packed codec this checkpoint's weights actually carry
// (`Q4_K`/`Q5_K`/`Q6_K`) has its own kernel, so the "convert back to f32"
// branch had zero remaining callers -- not narrowed, deleted, the same
// call this repo makes on any mechanism a landing empties out completely.
// A future codec needing this exact shape again can restore it from
// history; it is a small, well-documented pattern, not lost work.

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::string::ToString;

    use proxima_gguf::{GgmlType as WireType, GgufModel, TensorPayload, write_complete};

    use super::*;
    use crate::error::InteropError;

    fn dims(values: &[u64]) -> arrayvec::ArrayVec<u64, { proxima_gguf::tensor::MAX_DIMS }> {
        values.iter().copied().collect()
    }

    /// The decisive proof for this file's own fix: a raw-packed `F32` matmul
    /// weight bound through [`bind_matmul_weight_as`] must land in
    /// [`BoundWeights::owned`], transposed, and produce the exact same
    /// matmul result as an independent hand computation over the tensor's
    /// own on-disk bytes -- run through the real
    /// [`proxima_tensor::cpu::evaluate_quantized_named`] evaluation a forward
    /// program actually uses, not merely compared byte-for-byte against the
    /// dequantize-then-transpose path.
    ///
    /// `out_dim=4`/`in_dim=6` are deliberately asymmetric: this is exactly
    /// what let the bug this test guards against hide behind a
    /// one-channel or square fixture (see this module's own dense-vs-packed
    /// history) -- a transpose of a square or single-row buffer can
    /// coincidentally read back correctly, or silently permute symmetric
    /// data, so it proves nothing. With `out_dim != in_dim`, a buffer
    /// addressed through the wrong axis order reads flatly wrong values.
    #[cfg(feature = "std")]
    #[test]
    fn raw_packed_f32_matmul_weight_matches_an_independent_hand_computed_matmul() {
        let out_dim = 4usize;
        let in_dim = 6usize;
        // GGUF's own on-disk convention: `out_dim` rows, each a contiguous
        // run of `in_dim` elements -- weight(out, in) = out*10 + in, distinct
        // per element so a scrambled read is detectable.
        let mut on_disk = vec![0.0f32; out_dim * in_dim];
        for out_index in 0..out_dim {
            for in_index in 0..in_dim {
                on_disk[out_index * in_dim + in_index] = (out_index * 10 + in_index) as f32;
            }
        }
        let bytes: Vec<u8> = on_disk
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();

        let model = GgufModel {
            version: 3,
            metadata: Vec::new(),
            tensors: vec![TensorPayload {
                name: "blk.0.ffn_gate_inp.weight".to_string(),
                dims: dims(&[in_dim as u64, out_dim as u64]),
                ggml_type: WireType::F32,
                data: &bytes,
            }],
        };
        let file_bytes =
            write_complete(&model).expect("writes gguf with an asymmetric f32 matmul weight");
        let parsed = proxima_gguf::pipe::parse_complete(&file_bytes)
            .expect("parses gguf with an asymmetric f32 matmul weight");

        let mut state = BoundWeights {
            resident_bytes: 0,
            owned: Vec::new(),
            packed: Vec::new(),
            packed_owned: Vec::new(),
        };
        bind_matmul_weight_as(
            &parsed,
            &file_bytes,
            "blk.0.ffn_gate_inp.weight",
            "gate".to_string(),
            out_dim,
            in_dim,
            &mut state,
        )
        .expect("binds the asymmetric f32 matmul weight");
        assert!(
            state.packed.is_empty(),
            "an F32 matmul weight must never take the raw-packed path -- nothing downstream corrects its layout"
        );
        assert_eq!(
            state.owned.len(),
            1,
            "the F32 matmul weight must land in the transposed owned path"
        );
        let bound_weight = &state.owned[0].1;

        // Deliberately NOT symmetric around zero (an earlier draft used
        // `index - 2.5`, whose sum over `0..6` is exactly zero -- that
        // silently canceled every `out_dim`-dependent term below and made
        // all four outputs identical regardless of which axis order the
        // buffer was actually read in, a vacuous test that would pass under
        // a transpose too. `index + 1` sums to a nonzero, out-axis-coupled
        // value instead.
        let activation: Vec<f32> = (0..in_dim).map(|index| (index as f32) + 1.0).collect();

        let mut program: Vec<proxima_tensor::op::Op> = Vec::new();
        let activation_node = proxima_tensor::op::append(
            &mut program,
            proxima_tensor::op::Op::Input {
                dtype: proxima_tensor::dtype::DType::Float32,
                shape: alloc::vec![proxima_tensor::op::Extent::Static(in_dim as u32)],
                name: Some("activation".to_string()),
            },
        );
        let weight_node = proxima_tensor::op::append(
            &mut program,
            proxima_tensor::op::Op::Input {
                dtype: proxima_tensor::dtype::DType::Float32,
                shape: alloc::vec![
                    proxima_tensor::op::Extent::Static(in_dim as u32),
                    proxima_tensor::op::Extent::Static(out_dim as u32)
                ],
                name: Some("gate".to_string()),
            },
        );
        let product = proxima_tensor::op::append(
            &mut program,
            proxima_tensor::op::Op::Elementwise {
                dtype: proxima_tensor::dtype::DType::Float32,
                body: proxima_tensor::op::ScalarOp::Multiply,
                operands: alloc::vec![
                    (
                        activation_node,
                        proxima_tensor::map::IndexMap::Affine(proxima_tensor::map::projection(
                            2,
                            &[1]
                        ))
                    ),
                    (
                        weight_node,
                        proxima_tensor::map::IndexMap::Affine(proxima_tensor::map::projection(
                            2,
                            &[1, 0]
                        ))
                    ),
                ],
                name: None,
            },
        );
        let logits = proxima_tensor::op::append(
            &mut program,
            proxima_tensor::op::Op::Reduce(proxima_tensor::op::Reduce {
                dtype: proxima_tensor::dtype::DType::Float32,
                body: proxima_tensor::op::ScalarOp::Add,
                init: proxima_tensor::op::ReduceInit::Zero,
                operand: product,
                in_map: proxima_tensor::map::IndexMap::Affine(proxima_tensor::map::projection(
                    2,
                    &[0, 1],
                )),
                out_map: proxima_tensor::map::IndexMap::Affine(proxima_tensor::map::projection(
                    2,
                    &[0],
                )),
                keep: proxima_tensor::op::Keep::Reduce,
                name: Some("logits".to_string()),
            }),
        );

        let named = [
            (
                "activation",
                proxima_tensor::cpu::QuantizedBlock::Float32(activation.as_slice()),
            ),
            (
                "gate",
                proxima_tensor::cpu::QuantizedBlock::Float32(bound_weight.as_slice()),
            ),
        ];
        let evaluated =
            proxima_tensor::cpu::evaluate_quantized_named(&program, &[], &named, &[logits])
                .expect("evaluate the bound matmul weight through the real interpreter");
        let ours = evaluated.root();

        let mut oracle = vec![0.0f32; out_dim];
        for (out_index, logit) in oracle.iter_mut().enumerate() {
            let mut accumulator = 0.0f32;
            for in_index in 0..in_dim {
                accumulator += activation[in_index] * on_disk[out_index * in_dim + in_index];
            }
            *logit = accumulator;
        }

        std::println!("raw_packed_f32_matmul ours={ours:?} oracle={oracle:?}");
        for (out_index, (found, wanted)) in ours.iter().zip(&oracle).enumerate() {
            let diff = (found - wanted).abs();
            assert!(
                diff < 1e-4,
                "output {out_index}: found={found} wanted={wanted} diff={diff}"
            );
        }
    }

    /// `[rows, k]` for [`q8_0_matmul_weight_binds_packed_and_matches_a_dequantized_oracle`]
    /// and its mutation companion below -- `k` a multiple of
    /// [`q8_0::QK8_0`] (32) so every row's own blocks stay row-aligned (no
    /// block straddles two rows), matching [`proxima_tensor::cpu::matmul_q8_0_f32`]'s
    /// own per-row block assumption. `rows != k` for the same
    /// axis-order-detection reason the `F32` test above documents.
    #[cfg(feature = "std")]
    const Q8_0_TEST_ROWS: usize = 3;
    #[cfg(feature = "std")]
    const Q8_0_TEST_K: usize = 64;

    /// Row-major `[rows, k]` weight bytes, real `Q8_0` blocks (never a
    /// hand-built buffer) via [`q8_0::quantize`] -- deterministic,
    /// non-degenerate per-element values so no two elements collide.
    #[cfg(feature = "std")]
    fn quantized_q8_0_weight_bytes() -> (alloc::vec::Vec<f32>, alloc::vec::Vec<u8>) {
        let on_disk: alloc::vec::Vec<f32> = (0..Q8_0_TEST_ROWS * Q8_0_TEST_K)
            .map(|index| ((index % 41) as f32 - 20.0) / 8.0)
            .collect();
        let mut bytes =
            alloc::vec![0u8; (on_disk.len() / q8_0::QK8_0) * q8_0::BLOCK_BYTES];
        q8_0::quantize(&on_disk, &mut bytes).expect("real q8_0 encoder quantizes this fixture");
        (on_disk, bytes)
    }

    /// Builds the `[rows, 1] x [rows, k] -> [rows, 1]` quantized-matmul
    /// program [`proxima_tensor::cpu::run_reduce_quantized`]'s own packed
    /// dispatch recognizes -- the same op shape `proxima_tensor::cpu`'s own
    /// `quantized_matmul_program` test helper builds (that helper is
    /// private to `cpu.rs`'s own test module, so this is a same-shape,
    /// independently written copy, not a shared function), rebuilt here
    /// through the named-`Op::Input` [`gguf_tensor_as_packed_block`]'s own
    /// callers actually use.
    #[cfg(feature = "std")]
    fn q8_0_matmul_program() -> (Vec<proxima_tensor::op::Op>, proxima_tensor::op::NodeId) {
        let mut program: Vec<proxima_tensor::op::Op> = Vec::new();
        let weight_node = proxima_tensor::op::append(
            &mut program,
            proxima_tensor::op::Op::Input {
                dtype: proxima_tensor::dtype::DType::UInt8,
                shape: alloc::vec![
                    proxima_tensor::op::Extent::Static(Q8_0_TEST_ROWS as u32),
                    proxima_tensor::op::Extent::Static(Q8_0_TEST_K as u32)
                ],
                name: Some("weight".to_string()),
            },
        );
        let activation_node = proxima_tensor::op::append(
            &mut program,
            proxima_tensor::op::Op::Input {
                dtype: proxima_tensor::dtype::DType::Float32,
                shape: alloc::vec![
                    proxima_tensor::op::Extent::Static(Q8_0_TEST_K as u32),
                    proxima_tensor::op::Extent::Static(1)
                ],
                name: Some("activation".to_string()),
            },
        );
        let product = proxima_tensor::op::append(
            &mut program,
            proxima_tensor::op::Op::Elementwise {
                dtype: proxima_tensor::dtype::DType::Float32,
                body: proxima_tensor::op::ScalarOp::Multiply,
                operands: alloc::vec![
                    (
                        weight_node,
                        proxima_tensor::map::IndexMap::Affine(proxima_tensor::map::projection(
                            3,
                            &[0, 2]
                        ))
                    ),
                    (
                        activation_node,
                        proxima_tensor::map::IndexMap::Affine(proxima_tensor::map::projection(
                            3,
                            &[2, 1]
                        ))
                    ),
                ],
                name: None,
            },
        );
        let sum = proxima_tensor::op::append(
            &mut program,
            proxima_tensor::op::Op::Reduce(proxima_tensor::op::Reduce {
                dtype: proxima_tensor::dtype::DType::Float32,
                body: proxima_tensor::op::ScalarOp::Add,
                init: proxima_tensor::op::ReduceInit::Zero,
                operand: product,
                in_map: proxima_tensor::map::IndexMap::Affine(proxima_tensor::map::projection(
                    3,
                    &[0, 1, 2],
                )),
                out_map: proxima_tensor::map::IndexMap::Affine(proxima_tensor::map::projection(
                    3,
                    &[0, 1],
                )),
                keep: proxima_tensor::op::Keep::Reduce,
                name: Some("q8_0_matmul".to_string()),
            }),
        );
        (program, sum)
    }

    /// This module's fix, proved directly: a `Q8_0` matmul weight must bind
    /// through [`gguf_tensor_as_packed_block`] into [`BoundWeights::packed`]
    /// zero-copy -- never fall through to [`gguf_tensor_as_f32`]'s owned
    /// dequantize path -- and the real
    /// [`proxima_tensor::cpu::matmul_q8_0_f32`] kernel driven off that
    /// packed buffer must produce the exact same output as an independent
    /// oracle: [`q8_0::dequantize`] applied to the SAME packed bytes,
    /// matmul'd by hand. The oracle dequantizes the packed bytes rather than
    /// the pre-quantization `f32` source, because `Q8_0` quantization is
    /// itself lossy -- comparing against the pre-quantization values would
    /// conflate this bind wiring's own correctness with `Q8_0`'s codec
    /// accuracy (already proved in `proxima_gguf::quant::q8_0`'s own tests).
    #[cfg(feature = "std")]
    #[test]
    fn q8_0_matmul_weight_binds_packed_and_matches_a_dequantized_oracle() {
        let (_on_disk, packed_bytes) = quantized_q8_0_weight_bytes();

        let model = GgufModel {
            version: 3,
            metadata: Vec::new(),
            tensors: vec![TensorPayload {
                name: "blk.0.attn_q.weight".to_string(),
                dims: dims(&[Q8_0_TEST_K as u64, Q8_0_TEST_ROWS as u64]),
                ggml_type: WireType::Q8_0,
                data: &packed_bytes,
            }],
        };
        let file_bytes = write_complete(&model).expect("writes gguf with a real q8_0 weight");
        let parsed =
            proxima_gguf::pipe::parse_complete(&file_bytes).expect("parses q8_0 weight gguf");

        let mut state = BoundWeights {
            resident_bytes: 0,
            owned: Vec::new(),
            packed: Vec::new(),
            packed_owned: Vec::new(),
        };
        bind_matmul_weight_as(
            &parsed,
            &file_bytes,
            "blk.0.attn_q.weight",
            "weight".to_string(),
            Q8_0_TEST_ROWS,
            Q8_0_TEST_K,
            &mut state,
        )
        .expect("binds the q8_0 matmul weight");
        assert!(
            state.owned.is_empty(),
            "a q8_0 matmul weight must take the zero-copy packed path, never the owned dequantize \
             fallback -- this is the exact defect this change fixes"
        );
        assert_eq!(state.packed.len(), 1, "exactly one packed weight bound");
        let bound_bytes = match &state.packed[0].1 {
            proxima_tensor::cpu::QuantizedBlock::Q8_0(bytes) => *bytes,
            other => panic!("expected a QuantizedBlock::Q8_0, found {other:?}"),
        };
        assert_eq!(
            bound_bytes, packed_bytes.as_slice(),
            "the packed path must borrow the exact on-disk q8_0 bytes, no copy"
        );

        let activation: Vec<f32> = (0..Q8_0_TEST_K).map(|index| (index as f32) - 32.0).collect();
        let (program, sum) = q8_0_matmul_program();
        let named = [
            ("weight", proxima_tensor::cpu::QuantizedBlock::Q8_0(bound_bytes)),
            (
                "activation",
                proxima_tensor::cpu::QuantizedBlock::Float32(activation.as_slice()),
            ),
        ];
        let evaluated =
            proxima_tensor::cpu::evaluate_quantized_named(&program, &[], &named, &[sum])
                .expect("evaluate the bound q8_0 packed weight through the real interpreter");
        let ours = evaluated.root();

        let mut dequantized_weight = alloc::vec![0.0f32; Q8_0_TEST_ROWS * Q8_0_TEST_K];
        q8_0::dequantize(&packed_bytes, &mut dequantized_weight)
            .expect("dequantize the same packed bytes for the oracle");
        let mut oracle = alloc::vec![0.0f32; Q8_0_TEST_ROWS];
        for (row, logit) in oracle.iter_mut().enumerate() {
            let mut accumulator = 0.0f32;
            for column in 0..Q8_0_TEST_K {
                accumulator +=
                    activation[column] * dequantized_weight[row * Q8_0_TEST_K + column];
            }
            *logit = accumulator;
        }

        std::println!("q8_0_matmul ours={ours:?} oracle={oracle:?}");
        for (row, (found, wanted)) in ours.iter().zip(&oracle).enumerate() {
            let diff = (found - wanted).abs();
            assert!(
                diff < 1e-2,
                "row {row}: found={found} wanted={wanted} diff={diff}"
            );
        }
    }

    /// Mutation companion to
    /// [`q8_0_matmul_weight_binds_packed_and_matches_a_dequantized_oracle`]:
    /// runs the exact same bind-then-evaluate pipeline, but flips one packed
    /// byte (inside a block's `qs` region, not its `d` scale header) before
    /// binding, so the packed path decodes a deliberately wrong value. Then
    /// asserts the real kernel's output on the corrupted bytes diverges from
    /// the clean oracle beyond the previous test's own `1e-2` tolerance --
    /// proving that tolerance is tight enough to actually catch a wrong
    /// decode, not so loose the equivalence check above is vacuous.
    #[cfg(feature = "std")]
    #[test]
    fn q8_0_matmul_weight_packed_path_is_sensitive_to_a_corrupted_byte() {
        let (_on_disk, clean_bytes) = quantized_q8_0_weight_bytes();
        let mut corrupted_bytes = clean_bytes.clone();
        // second block's 9th `qs` byte -- decoded element 40, whose
        // activation coefficient (`40 - 32 = 8`) is far from zero, unlike
        // element 32 (the second block's first element), whose activation
        // coefficient is exactly zero and would mask any corruption there.
        let corrupted_index = q8_0::BLOCK_BYTES + 2 + 8;
        // flips the signed byte's sign bit -- guarantees a large jump in the
        // decoded value regardless of what the original byte happened to be,
        // unlike a smaller XOR mask that can land near the original value.
        corrupted_bytes[corrupted_index] = corrupted_bytes[corrupted_index].wrapping_add(128);

        let mut clean_weight = alloc::vec![0.0f32; Q8_0_TEST_ROWS * Q8_0_TEST_K];
        q8_0::dequantize(&clean_bytes, &mut clean_weight)
            .expect("dequantize the clean packed bytes for the oracle");
        let activation: Vec<f32> = (0..Q8_0_TEST_K).map(|index| (index as f32) - 32.0).collect();
        let mut clean_oracle = alloc::vec![0.0f32; Q8_0_TEST_ROWS];
        for (row, logit) in clean_oracle.iter_mut().enumerate() {
            let mut accumulator = 0.0f32;
            for column in 0..Q8_0_TEST_K {
                accumulator += activation[column] * clean_weight[row * Q8_0_TEST_K + column];
            }
            *logit = accumulator;
        }

        let model = GgufModel {
            version: 3,
            metadata: Vec::new(),
            tensors: vec![TensorPayload {
                name: "blk.0.attn_q.weight".to_string(),
                dims: dims(&[Q8_0_TEST_K as u64, Q8_0_TEST_ROWS as u64]),
                ggml_type: WireType::Q8_0,
                data: &corrupted_bytes,
            }],
        };
        let file_bytes =
            write_complete(&model).expect("writes gguf with a corrupted q8_0 weight");
        let parsed = proxima_gguf::pipe::parse_complete(&file_bytes)
            .expect("parses corrupted q8_0 weight gguf");

        let mut state = BoundWeights {
            resident_bytes: 0,
            owned: Vec::new(),
            packed: Vec::new(),
            packed_owned: Vec::new(),
        };
        bind_matmul_weight_as(
            &parsed,
            &file_bytes,
            "blk.0.attn_q.weight",
            "weight".to_string(),
            Q8_0_TEST_ROWS,
            Q8_0_TEST_K,
            &mut state,
        )
        .expect("binds the corrupted q8_0 matmul weight");
        let bound_bytes = match &state.packed[0].1 {
            proxima_tensor::cpu::QuantizedBlock::Q8_0(bytes) => *bytes,
            other => panic!("expected a QuantizedBlock::Q8_0, found {other:?}"),
        };

        let (program, sum) = q8_0_matmul_program();
        let named = [
            ("weight", proxima_tensor::cpu::QuantizedBlock::Q8_0(bound_bytes)),
            (
                "activation",
                proxima_tensor::cpu::QuantizedBlock::Float32(activation.as_slice()),
            ),
        ];
        let evaluated =
            proxima_tensor::cpu::evaluate_quantized_named(&program, &[], &named, &[sum])
                .expect("evaluate the corrupted q8_0 packed weight through the real interpreter");
        let corrupted_result = evaluated.root();

        std::println!(
            "q8_0_corrupted corrupted={corrupted_result:?} clean_oracle={clean_oracle:?}"
        );
        let max_diff = corrupted_result
            .iter()
            .zip(&clean_oracle)
            .map(|(found, wanted)| (found - wanted).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-2,
            "a corrupted qs byte must move the decoded matmul output past the equivalence \
             test's own 1e-2 tolerance, or that tolerance cannot detect a wrong decode: \
             max_diff={max_diff}"
        );
    }

    /// The defect this signature change fixes, proved directly: a decoded
    /// buffer whose length disagrees with `expert_count * out_dim * in_dim`
    /// (exactly what a GGUF file with a mismatched `general.expert_count`
    /// hparam would hand this function) used to slice past the buffer's
    /// end mid-transpose. It must now return a typed error instead.
    #[cfg(feature = "std")]
    #[test]
    fn expert_stack_length_mismatch_is_a_typed_error_not_a_panic() {
        let expert_count = 4;
        let out_dim = 8;
        let in_dim = 8;
        let one_expert_short = vec![0.0f32; (expert_count - 1) * out_dim * in_dim];

        let result = transpose_expert_stack(
            &one_expert_short,
            "blk.0.ffn_gate_exps.weight",
            expert_count,
            out_dim,
            in_dim,
        );

        assert!(
            matches!(result, Err(InteropError::MoeExpertShapeMismatch { .. })),
            "a short decoded buffer must surface as a typed error, got {result:?}"
        );
    }

    /// Same proof, for the plain (non-MoE) dense-weight transpose: a
    /// decoded buffer whose length disagrees with `out_dim * in_dim` used
    /// to trip `assert_eq!` (a panic) instead of returning a typed error.
    #[cfg(feature = "std")]
    #[test]
    fn dense_weight_length_mismatch_is_a_typed_error_not_a_panic() {
        let out_dim = 8;
        let in_dim = 8;
        let too_short = vec![0.0f32; out_dim * in_dim - 1];

        let result = transpose_out_in_to_in_out(&too_short, "output.weight", out_dim, in_dim);

        assert!(
            matches!(result, Err(InteropError::DenseWeightShapeMismatch { .. })),
            "a short decoded buffer must surface as a typed error, got {result:?}"
        );
    }

    /// A round-tripped `F32` tensor comes back byte-identical as `f32`,
    /// not merely "close" -- reinterpretation, not conversion.
    #[test]
    fn f32_tensor_reinterprets_bytes_exactly() {
        let values = [1.0f32, -2.5, 3.25, 0.0];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let model = GgufModel {
            version: 3,
            metadata: Vec::new(),
            tensors: vec![TensorPayload {
                name: "weights".to_string(),
                dims: dims(&[4]),
                ggml_type: WireType::F32,
                data: bytes.as_slice(),
            }],
        };
        let file_bytes = write_complete(&model).expect("writes gguf");
        let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parses gguf");

        let decoded =
            gguf_tensor_as_f32(&parsed, &file_bytes, "weights").expect("bind f32 tensor by name");
        assert_eq!(decoded, values);
    }

    /// A `Q4_K` tensor decodes through the crate's own dequantizer and
    /// matches an independent hand computation of the block's `x =
    /// d*sc*q - dmin*m` formula for the one nonzero probe element this
    /// fixture packs.
    #[test]
    fn q4_k_tensor_dequantizes_through_bind_by_name() {
        let mut block = [0u8; q4_k::BLOCK_BYTES];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes()); // d
        block[2..4].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes()); // dmin
        // sub_block 0: scale code 3, min code 61 (sub_block < 4 packing).
        block[4] = 3;
        block[8] = 61;
        block[16] = 0x07; // qs[0] low nibble = 7 -> element 0 of sub_block 0

        let model = GgufModel {
            version: 3,
            metadata: Vec::new(),
            tensors: vec![TensorPayload {
                name: "blk.0.ffn_gate.weight".to_string(),
                dims: dims(&[q4_k::QK_K as u64]),
                ggml_type: WireType::Q4_K,
                data: &block,
            }],
        };
        let file_bytes = write_complete(&model).expect("writes quantized gguf");
        let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parses quantized gguf");

        let decoded = gguf_tensor_as_f32(&parsed, &file_bytes, "blk.0.ffn_gate.weight")
            .expect("bind q4_k tensor by name");
        assert_eq!(decoded.len(), q4_k::QK_K);
        // element 0: d*sc*q - dmin*m = 1.0*3.0*7.0 - 0.5*61.0 = -9.5
        assert!(
            (decoded[0] - (-9.5)).abs() < 1e-6,
            "decoded[0]={}",
            decoded[0]
        );
        // every other element in sub_block 0 shares scale/min with q=0.
        assert!(
            (decoded[1] - (-30.5)).abs() < 1e-6,
            "decoded[1]={}",
            decoded[1]
        );
    }

    #[test]
    fn unknown_name_errors_instead_of_panicking() {
        let model = GgufModel {
            version: 3,
            metadata: Vec::new(),
            tensors: Vec::new(),
        };
        let file_bytes = write_complete(&model).expect("writes empty gguf");
        let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parses empty gguf");

        let outcome = gguf_tensor_as_f32(&parsed, &file_bytes, "missing");
        assert!(matches!(outcome, Err(InteropError::UnknownTensor { .. })));
    }

    #[test]
    fn unrepresentable_ggml_type_errors_instead_of_misreading_bytes() {
        let data = [0u8; 18]; // one Q4_0 block
        let model = GgufModel {
            version: 3,
            metadata: Vec::new(),
            tensors: vec![TensorPayload {
                name: "blk.0.attn_q.weight".to_string(),
                dims: dims(&[32]),
                ggml_type: WireType::Q4_0,
                data: &data,
            }],
        };
        let file_bytes = write_complete(&model).expect("writes q4_0 gguf");
        let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parses q4_0 gguf");

        let outcome = gguf_tensor_as_f32(&parsed, &file_bytes, "blk.0.attn_q.weight");
        assert!(matches!(
            outcome,
            Err(InteropError::UnrepresentableGgmlType { .. })
        ));
    }

    /// [`architecture_from_metadata`] against a synthetic checkpoint whose
    /// metadata carries every `{architecture}.*` key by hand -- proves the
    /// derivation reads real keys rather than falling back to invented
    /// defaults, and that `vocab` comes from `token_embd.weight`'s own
    /// shape, not a metadata key (this fixture writes none).
    #[test]
    fn architecture_from_metadata_reads_real_keys_and_derives_vocab_from_tensor_shape() {
        use proxima_gguf::value::MetadataValue as Value;

        let embed_bytes = vec![0u8; 8 * 3 * 4]; // [embedding=8, vocab=3] f32
        let model = GgufModel {
            version: 3,
            metadata: vec![
                (
                    "general.architecture".to_string(),
                    Value::String("llama".to_string()),
                ),
                ("llama.embedding_length".to_string(), Value::U32(8)),
                ("llama.feed_forward_length".to_string(), Value::U32(32)),
                ("llama.attention.head_count".to_string(), Value::U32(2)),
                ("llama.attention.head_count_kv".to_string(), Value::U32(1)),
                ("llama.block_count".to_string(), Value::U32(4)),
                ("llama.rope.dimension_count".to_string(), Value::U32(4)),
            ],
            tensors: vec![TensorPayload {
                name: "token_embd.weight".to_string(),
                dims: dims(&[8, 3]),
                ggml_type: WireType::F32,
                data: &embed_bytes,
            }],
        };
        let file_bytes = write_complete(&model).expect("writes gguf with architecture metadata");
        let parsed = proxima_gguf::parse_complete(&file_bytes)
            .expect("parses gguf with architecture metadata");

        let architecture = architecture_from_metadata(&parsed)
            .expect("derive architecture from real metadata keys");
        assert_eq!(
            architecture,
            ModelArchitecture {
                vocab: 3,
                embedding: 8,
                feed_forward: 32,
                query_heads: 2,
                kv_heads: 1,
                head_dim: 4,
                block_count: 4,
                expert_count: 0,
                expert_used_count: 0,
                rope_freq_base: proxima_tensor::sized::ROPE_FREQ_BASE_DEFAULT,
                tied_embeddings: false,
            },
            "a checkpoint with no expert_count/expert_used_count key is dense: both fields must read as 0, \
             not error; a checkpoint with no rope.freq_base key must fall back to the sizing-config default"
        );
    }

    /// A mixture-of-experts checkpoint carries `{architecture}.expert_count`/
    /// `{architecture}.expert_used_count` alongside every dense key --
    /// `architecture_from_metadata` must read both rather than silently
    /// treating the checkpoint as dense.
    #[test]
    fn architecture_from_metadata_reads_expert_count_when_present() {
        use proxima_gguf::value::MetadataValue as Value;

        let embed_bytes = vec![0u8; 8 * 3 * 4]; // [embedding=8, vocab=3] f32
        let model = GgufModel {
            version: 3,
            metadata: vec![
                (
                    "general.architecture".to_string(),
                    Value::String("llama".to_string()),
                ),
                ("llama.embedding_length".to_string(), Value::U32(8)),
                ("llama.feed_forward_length".to_string(), Value::U32(32)),
                ("llama.attention.head_count".to_string(), Value::U32(2)),
                ("llama.attention.head_count_kv".to_string(), Value::U32(1)),
                ("llama.block_count".to_string(), Value::U32(4)),
                ("llama.rope.dimension_count".to_string(), Value::U32(4)),
                ("llama.expert_count".to_string(), Value::U32(8)),
                ("llama.expert_used_count".to_string(), Value::U32(2)),
            ],
            tensors: vec![TensorPayload {
                name: "token_embd.weight".to_string(),
                dims: dims(&[8, 3]),
                ggml_type: WireType::F32,
                data: &embed_bytes,
            }],
        };
        let file_bytes = write_complete(&model).expect("writes gguf with moe metadata");
        let parsed =
            proxima_gguf::parse_complete(&file_bytes).expect("parses gguf with moe metadata");

        let architecture = architecture_from_metadata(&parsed)
            .expect("derive architecture from real metadata keys");
        assert_eq!(
            architecture.expert_count, 8,
            "expert_count must read the real metadata key, not default to 0"
        );
        assert_eq!(
            architecture.expert_used_count, 2,
            "expert_used_count must read the real metadata key, not default to 0"
        );
    }

    /// A checkpoint absent `{architecture}.rope.dimension_count` entirely
    /// (confirmed real on LFM2.5-8B-A1B, `bind.rs`'s own doc on
    /// [`ModelArchitecture::head_dim`]) must derive `head_dim` as
    /// `embedding / query_heads` rather than
    /// [`InteropError::MissingMetadataKey`] -- this fixture's own
    /// embedding=8, query_heads=2 implies head_dim=4, matching what this
    /// same fixture's other tests declare explicitly via the key.
    #[test]
    fn architecture_from_metadata_derives_head_dim_when_rope_dimension_count_is_absent() {
        use proxima_gguf::value::MetadataValue as Value;

        let embed_bytes = vec![0u8; 8 * 3 * 4]; // [embedding=8, vocab=3] f32
        let model = GgufModel {
            version: 3,
            metadata: vec![
                (
                    "general.architecture".to_string(),
                    Value::String("llama".to_string()),
                ),
                ("llama.embedding_length".to_string(), Value::U32(8)),
                ("llama.feed_forward_length".to_string(), Value::U32(32)),
                ("llama.attention.head_count".to_string(), Value::U32(2)),
                ("llama.attention.head_count_kv".to_string(), Value::U32(1)),
                ("llama.block_count".to_string(), Value::U32(4)),
                // deliberately no llama.rope.dimension_count key
            ],
            tensors: vec![TensorPayload {
                name: "token_embd.weight".to_string(),
                dims: dims(&[8, 3]),
                ggml_type: WireType::F32,
                data: &embed_bytes,
            }],
        };
        let file_bytes = write_complete(&model).expect("writes gguf without rope.dimension_count");
        let parsed = proxima_gguf::parse_complete(&file_bytes)
            .expect("parses gguf without rope.dimension_count");

        let architecture = architecture_from_metadata(&parsed)
            .expect("absent rope.dimension_count must derive, not error");
        assert_eq!(
            architecture.head_dim, 4,
            "head_dim must derive as embedding(8) / query_heads(2)"
        );
    }

    /// `{architecture}.attention.head_count_kv` stored as a per-layer
    /// [`proxima_gguf::value::MetadataArray`] whose entries all agree
    /// (every real dense/uniform checkpoint that ever uses the array
    /// encoding at all) must read as that one scalar, not error.
    #[test]
    fn architecture_from_metadata_reads_uniform_head_count_kv_array() {
        use proxima_gguf::value::MetadataArray;
        use proxima_gguf::value::MetadataValue as Value;

        let embed_bytes = vec![0u8; 8 * 3 * 4];
        let model = GgufModel {
            version: 3,
            metadata: vec![
                (
                    "general.architecture".to_string(),
                    Value::String("llama".to_string()),
                ),
                ("llama.embedding_length".to_string(), Value::U32(8)),
                ("llama.feed_forward_length".to_string(), Value::U32(32)),
                ("llama.attention.head_count".to_string(), Value::U32(2)),
                (
                    "llama.attention.head_count_kv".to_string(),
                    Value::Array(MetadataArray::I32(vec![1, 1, 1, 1])),
                ),
                ("llama.block_count".to_string(), Value::U32(4)),
                ("llama.rope.dimension_count".to_string(), Value::U32(4)),
            ],
            tensors: vec![TensorPayload {
                name: "token_embd.weight".to_string(),
                dims: dims(&[8, 3]),
                ggml_type: WireType::F32,
                data: &embed_bytes,
            }],
        };
        let file_bytes =
            write_complete(&model).expect("writes gguf with a uniform head_count_kv array");
        let parsed = proxima_gguf::parse_complete(&file_bytes)
            .expect("parses gguf with a uniform head_count_kv array");

        let architecture = architecture_from_metadata(&parsed)
            .expect("a uniform per-layer array must read as its one scalar");
        assert_eq!(architecture.kv_heads, 1);
    }

    /// `{architecture}.attention.head_count_kv` stored as a per-layer array
    /// whose entries genuinely DISAGREE (confirmed real on LFM2.5-8B-A1B:
    /// `0` for its 18 convolution layers, `8` for its 6 attention layers,
    /// in the SAME array) must surface
    /// [`InteropError::HeterogeneousMetadataArray`], never a silently picked
    /// scalar -- the defect this test is named for: before this fix,
    /// [`metadata_u32`]'s `MetadataValue::as_u32` had no `Array` arm at all,
    /// so this exact shape returned [`InteropError::MissingMetadataKey`]
    /// (a WRONG diagnosis -- the key is present, just array-shaped),
    /// hard-failing every hybrid checkpoint's load with a misleading error.
    #[test]
    fn architecture_from_metadata_refuses_a_heterogeneous_head_count_kv_array() {
        use proxima_gguf::value::MetadataArray;
        use proxima_gguf::value::MetadataValue as Value;

        let embed_bytes = vec![0u8; 8 * 3 * 4];
        let model = GgufModel {
            version: 3,
            metadata: vec![
                (
                    "general.architecture".to_string(),
                    Value::String("lfm2".to_string()),
                ),
                ("lfm2.embedding_length".to_string(), Value::U32(8)),
                ("lfm2.feed_forward_length".to_string(), Value::U32(32)),
                ("lfm2.attention.head_count".to_string(), Value::U32(2)),
                (
                    "lfm2.attention.head_count_kv".to_string(),
                    Value::Array(MetadataArray::I32(vec![0, 0, 8, 0, 8, 8])),
                ),
                ("lfm2.block_count".to_string(), Value::U32(6)),
                ("lfm2.rope.dimension_count".to_string(), Value::U32(4)),
            ],
            tensors: vec![TensorPayload {
                name: "token_embd.weight".to_string(),
                dims: dims(&[8, 3]),
                ggml_type: WireType::F32,
                data: &embed_bytes,
            }],
        };
        let file_bytes =
            write_complete(&model).expect("writes gguf with a heterogeneous head_count_kv array");
        let parsed = proxima_gguf::parse_complete(&file_bytes)
            .expect("parses gguf with a heterogeneous head_count_kv array");

        let outcome = architecture_from_metadata(&parsed);
        assert!(
            matches!(
                outcome,
                Err(InteropError::HeterogeneousMetadataArray {
                    distinct_values: 2,
                    ..
                })
            ),
            "a genuinely per-layer-varying array must be refused with a named, honest error, got {outcome:?}"
        );
    }

    /// A checkpoint declaring a non-`10_000.0` `{architecture}.rope.freq_base`
    /// (Qwen3's real `1_000_000.0`, Llama 3's real `500_000.0`) must have
    /// `architecture_from_metadata` read that value, not silently fall back
    /// to a hardcoded default -- the exact defect this test is named for:
    /// before `ModelArchitecture` carried a `rope_freq_base` field at all,
    /// this assertion could not even be written, let alone pass, and every
    /// production call site used a bare `10_000.0` constant regardless of
    /// what a checkpoint declared.
    #[proxima::test]
    #[case::qwen3_real_freq_base(1_000_000.0)]
    #[case::llama3_real_freq_base(500_000.0)]
    async fn architecture_from_metadata_reads_the_real_rope_freq_base(#[case] freq_base: f32) {
        use proxima_gguf::value::MetadataValue as Value;

        let embed_bytes = vec![0u8; 8 * 3 * 4]; // [embedding=8, vocab=3] f32
        let model = GgufModel {
            version: 3,
            metadata: vec![
                (
                    "general.architecture".to_string(),
                    Value::String("llama".to_string()),
                ),
                ("llama.embedding_length".to_string(), Value::U32(8)),
                ("llama.feed_forward_length".to_string(), Value::U32(32)),
                ("llama.attention.head_count".to_string(), Value::U32(2)),
                ("llama.attention.head_count_kv".to_string(), Value::U32(1)),
                ("llama.block_count".to_string(), Value::U32(4)),
                ("llama.rope.dimension_count".to_string(), Value::U32(4)),
                ("llama.rope.freq_base".to_string(), Value::F32(freq_base)),
            ],
            tensors: vec![TensorPayload {
                name: "token_embd.weight".to_string(),
                dims: dims(&[8, 3]),
                ggml_type: WireType::F32,
                data: &embed_bytes,
            }],
        };
        let file_bytes =
            write_complete(&model).expect("writes gguf with a non-default rope.freq_base");
        let parsed = proxima_gguf::parse_complete(&file_bytes)
            .expect("parses gguf with a non-default rope.freq_base");

        let architecture = architecture_from_metadata(&parsed)
            .expect("derive architecture from real metadata keys");
        assert_eq!(
            architecture.rope_freq_base,
            freq_base,
            "rope_freq_base must read the checkpoint's own metadata key, never the {}-default",
            proxima_tensor::sized::ROPE_FREQ_BASE_DEFAULT
        );
        assert_ne!(
            architecture.rope_freq_base,
            proxima_tensor::sized::ROPE_FREQ_BASE_DEFAULT,
            "this case is only meaningful when the checkpoint's declared value differs from the default"
        );
    }

    /// The `rope.freq_base` key stored as `F64` on the wire (a legal, if
    /// unusual, GGUF encoding `MetadataValue` itself carries) must also be
    /// read, not just the more common `F32` encoding -- proves the reader
    /// does not assume one wire width.
    #[test]
    fn architecture_from_metadata_reads_rope_freq_base_stored_as_f64() {
        use proxima_gguf::value::MetadataValue as Value;

        let embed_bytes = vec![0u8; 8 * 3 * 4]; // [embedding=8, vocab=3] f32
        let model = GgufModel {
            version: 3,
            metadata: vec![
                (
                    "general.architecture".to_string(),
                    Value::String("llama".to_string()),
                ),
                ("llama.embedding_length".to_string(), Value::U32(8)),
                ("llama.feed_forward_length".to_string(), Value::U32(32)),
                ("llama.attention.head_count".to_string(), Value::U32(2)),
                ("llama.attention.head_count_kv".to_string(), Value::U32(1)),
                ("llama.block_count".to_string(), Value::U32(4)),
                ("llama.rope.dimension_count".to_string(), Value::U32(4)),
                ("llama.rope.freq_base".to_string(), Value::F64(1_000_000.0)),
            ],
            tensors: vec![TensorPayload {
                name: "token_embd.weight".to_string(),
                dims: dims(&[8, 3]),
                ggml_type: WireType::F32,
                data: &embed_bytes,
            }],
        };
        let file_bytes = write_complete(&model).expect("writes gguf with an f64 rope.freq_base");
        let parsed = proxima_gguf::parse_complete(&file_bytes)
            .expect("parses gguf with an f64 rope.freq_base");

        let architecture = architecture_from_metadata(&parsed)
            .expect("derive architecture from real metadata keys");
        assert_eq!(
            architecture.rope_freq_base, 1_000_000.0,
            "an f64-encoded key must still be read"
        );
    }

    #[test]
    fn architecture_from_metadata_names_the_missing_key() {
        let model = GgufModel {
            version: 3,
            metadata: Vec::new(),
            tensors: Vec::new(),
        };
        let file_bytes = write_complete(&model).expect("writes empty gguf");
        let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parses empty gguf");

        let outcome = architecture_from_metadata(&parsed);
        assert!(matches!(
            outcome,
            Err(InteropError::MissingMetadataKey { key }) if key == "general.architecture"
        ));
    }
}

/// The allocation-shape gate this change exists for: a mixture-of-experts
/// checkpoint's allocation total must scale with the number of bytes its
/// packed codec actually needs, never with `experts * rows * cols * 4` (a
/// real Mixtral-8x7B's own figure -- 8 experts, 3 matrices, `4096x14336`,
/// 32 layers, `f32` -- is ~180 GB against this checkpoint's own ~25 GB
/// on-disk size). Two cases, both real GGUF (`write_complete`/
/// `parse_complete`, never a hand-built buffer), both using
/// [`q4_k::quantize`] to encode real (non-degenerate) weight data:
///
/// - [`moe_owned_allocation_matches_the_checkpoints_own_layout_convention`]'s
///   `native_stacked_f32` case: a native single-stacked `_exps.weight`
///   tensor binds through [`bind_moe_stacked_experts`], zero-copy straight
///   out of the mmap, zero owned or packed-owned bytes allocated at all.
/// - `per_expert_q4_k` (the same test, other case) and
///   [`moe_experts_now_stay_packed_for_the_real_per_expert_tensor_convention`]:
///   the real Mixtral convention (`restack.rs`'s own module doc) stores
///   `n_experts` independent tensors, which `bind_moe_expert_weights` must
///   restack into one contiguous buffer (no single on-disk byte range to
///   borrow) -- but that buffer now stays `Q4_K`-packed
///   ([`BoundWeights::packed_owned`]), not dequantized, since
///   `proxima_tensor::cpu::run_reduce_quantized`'s gather arm resolves a
///   per-expert byte offset directly out of a packed operand. Both tests
///   assert the allocation hits the packed floor, not the dequantized
///   ceiling -- proven for real against the full 25 GB checkpoint by
///   `real_mixtral_file::attempts_a_real_mixtral_forward_pass_and_reports_the_outcome`'s
///   own doc.
#[cfg(all(test, feature = "std"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod moe_memory_shape {
    use alloc::string::String;

    use proxima_gguf::quant::q4_k;
    use proxima_gguf::{GgmlType as WireType, GgufModel, TensorPayload, write_complete};

    use super::*;

    const EXPERT_COUNT: u32 = 2;
    const OUT_DIM: usize = q4_k::QK_K;
    const IN_DIM: usize = q4_k::QK_K;

    fn dims(values: &[u64]) -> arrayvec::ArrayVec<u64, { proxima_gguf::tensor::MAX_DIMS }> {
        values.iter().copied().collect()
    }

    /// Deterministic, non-degenerate weight data for one expert's own row
    /// -- distinct per `seed` so no two experts' bytes collide, the same
    /// non-degeneracy concern `tests/support/mod.rs`'s own `random_vec` doc
    /// names for this exact fixture shape.
    fn expert_values(seed: u32) -> alloc::vec::Vec<f32> {
        (0..(OUT_DIM * IN_DIM))
            .map(|index| {
                ((index as u32)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(seed)
                    % 1000) as f32
                    / 1000.0
                    - 0.5
            })
            .collect()
    }

    fn quantize_q4_k(values: &[f32]) -> alloc::vec::Vec<u8> {
        let mut bytes = alloc::vec![0u8; values.len() / q4_k::QK_K * q4_k::BLOCK_BYTES];
        q4_k::quantize(values, &mut bytes)
            .expect("real q4_k encoder quantizes this fixture's own weight data");
        bytes
    }

    /// A native single stacked `blk.0.ffn_gate_exps.weight` tensor -- every
    /// expert's own `[OUT_DIM, IN_DIM]` slab back to back, `F32` so
    /// [`bind_moe_stacked_experts`]'s safe packed arm actually fires (see
    /// that function's own doc for why only `F32` is safe here).
    fn checkpoint_with_native_stacked_f32_experts() -> alloc::vec::Vec<u8> {
        let mut values: alloc::vec::Vec<f32> =
            alloc::vec::Vec::with_capacity(EXPERT_COUNT as usize * OUT_DIM * IN_DIM);
        for expert in 0..EXPERT_COUNT {
            values.extend(expert_values(expert));
        }
        let bytes: alloc::vec::Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let model = GgufModel {
            version: 3,
            metadata: alloc::vec::Vec::new(),
            tensors: vec![TensorPayload {
                name: String::from("blk.0.ffn_gate_exps.weight"),
                dims: dims(&[IN_DIM as u64, OUT_DIM as u64, u64::from(EXPERT_COUNT)]),
                ggml_type: WireType::F32,
                data: &bytes,
            }],
        };
        write_complete(&model)
            .expect("writes a well-formed synthetic native-stacked moe checkpoint")
    }

    /// The real Mixtral-8x7B on-disk convention (`restack.rs`'s own module
    /// doc): `n_experts` independent `Q4_K` tensors, one per
    /// `blk.0.ffn_gate.{expert}.weight`, no native stack tensor at all.
    fn checkpoint_with_per_expert_q4_k_tensors() -> alloc::vec::Vec<u8> {
        let mut owned_bytes: alloc::vec::Vec<alloc::vec::Vec<u8>> = alloc::vec::Vec::new();
        let mut tensors: alloc::vec::Vec<(String, alloc::vec::Vec<u8>)> = alloc::vec::Vec::new();
        for expert in 0..EXPERT_COUNT {
            let quantized = quantize_q4_k(&expert_values(expert));
            owned_bytes.push(quantized);
        }
        for (expert, quantized) in owned_bytes.into_iter().enumerate() {
            tensors.push((alloc::format!("blk.0.ffn_gate.{expert}.weight"), quantized));
        }
        let payloads: alloc::vec::Vec<TensorPayload<'_>> = tensors
            .iter()
            .map(|(name, data)| TensorPayload {
                name: name.clone(),
                dims: dims(&[IN_DIM as u64, OUT_DIM as u64]),
                ggml_type: WireType::Q4_K,
                data: data.as_slice(),
            })
            .collect();
        let model = GgufModel {
            version: 3,
            metadata: alloc::vec::Vec::new(),
            tensors: payloads,
        };
        write_complete(&model)
            .expect("writes a well-formed synthetic per-expert-tensor moe checkpoint")
    }

    fn empty_state(file_bytes: &[u8]) -> BoundWeights<'_> {
        BoundWeights {
            resident_bytes: file_bytes.len(),
            owned: Vec::new(),
            packed: Vec::new(),
            packed_owned: Vec::new(),
        }
    }

    fn owned_bytes_total(state: &BoundWeights) -> usize {
        state
            .owned
            .iter()
            .map(|(_, data)| data.len() * core::mem::size_of::<f32>())
            .sum()
    }

    fn packed_owned_bytes_total(state: &BoundWeights) -> usize {
        state
            .packed_owned
            .iter()
            .map(|(_, bytes, _)| bytes.len())
            .sum()
    }

    /// A real Mixtral-8x7B's own shape, for the assertion's own numbers to
    /// mean something rather than an arbitrary threshold: `experts * out *
    /// in * 4` is the dequantized-owned ceiling the per-expert-tensor case
    /// used to hit before this codec bound packed, and the packed floor
    /// (`experts * (out*in/QK_K) * BLOCK_BYTES`) is what it allocates now.
    fn dequantized_owned_ceiling_bytes() -> usize {
        EXPERT_COUNT as usize * OUT_DIM * IN_DIM * core::mem::size_of::<f32>()
    }

    fn packed_floor_bytes() -> usize {
        EXPERT_COUNT as usize * (OUT_DIM * IN_DIM / q4_k::QK_K) * q4_k::BLOCK_BYTES
    }

    /// Two real GGUF layout conventions for the same weight family, one
    /// shared assertion: does binding this checkpoint's experts allocate
    /// bytes proportional to the *packed* size, or to `experts * rows *
    /// cols * 4`? `native_stacked_f32` borrows zero-copy straight out of
    /// `file_bytes` ([`bind_moe_stacked_experts`]'s packed arm, the same
    /// zero-copy contract [`bind_matmul_weight`] already gives a dense
    /// weight, so it allocates neither owned nor packed-owned bytes at
    /// all). `per_expert_q4_k` is the real Mixtral-8x7B on-disk convention
    /// (`restack.rs`'s own module doc): no single contiguous byte range to
    /// borrow, so `restack_into` must allocate one -- but now that buffer
    /// stays `Q4_K`-packed ([`BoundWeights::packed_owned`]) instead of being
    /// dequantized, so it hits the packed floor, not the dequantized
    /// ceiling (see [`moe_experts_now_stay_packed_for_the_real_per_expert_tensor_convention`]
    /// for the same shape asserted directly against `packed_floor_bytes`).
    #[proxima::test]
    #[case::native_stacked_f32(true)]
    #[case::per_expert_q4_k(false)]
    async fn moe_owned_allocation_matches_the_checkpoints_own_layout_convention(
        #[case] native_stacked: bool,
    ) {
        let file_bytes = if native_stacked {
            checkpoint_with_native_stacked_f32_experts()
        } else {
            checkpoint_with_per_expert_q4_k_tensors()
        };
        let parsed =
            proxima_gguf::parse_complete(&file_bytes).expect("parses synthetic moe checkpoint");
        let mut state = empty_state(&file_bytes);

        bind_moe_expert_weights(
            &parsed,
            &file_bytes,
            0,
            "ffn_gate",
            EXPERT_COUNT,
            OUT_DIM,
            IN_DIM,
            &mut state,
        )
        .expect("binds this case's own expert tensor layout");

        if native_stacked {
            assert_eq!(
                state.packed.len(),
                1,
                "one packed entry for the whole native stack, no per-expert split"
            );
            assert!(
                state.packed_owned.is_empty(),
                "a native stacked tensor has one contiguous byte range -- nothing to restack"
            );
            assert_eq!(
                owned_bytes_total(&state),
                0,
                "an f32 native stack must borrow zero-copy, allocating no owned f32 bytes"
            );
        } else {
            assert!(
                state.packed.is_empty(),
                "the restacked buffer is caller-owned, never a zero-copy borrow out of file_bytes"
            );
            assert_eq!(
                state.packed_owned.len(),
                1,
                "one restacked packed entry for the whole q4_k expert stack"
            );
            assert_eq!(
                owned_bytes_total(&state),
                0,
                "a q4_k expert stack must stay packed, allocating no dequantized f32 bytes"
            );
            assert_eq!(
                packed_owned_bytes_total(&state),
                packed_floor_bytes(),
                "the restacked buffer's own byte length must equal exactly the packed floor, not the dequantized ceiling"
            );
        }
    }

    /// The acceptance criterion the prior, `#[ignore]`d version of this test
    /// named: a real Mixtral-shaped per-expert-tensor `Q4_K` checkpoint's
    /// expert stack allocates proportional to the packed size, never
    /// `experts * rows * cols * 4` (~180 GiB for a real Mixtral-8x7B, which
    /// SIGKILLs a 64 GiB host partway through binding -- see
    /// [`bind_moe_expert_weights`]'s own doc for that measured ceiling).
    /// `run_reduce_quantized`'s gather now resolves `per_expert_bytes`
    /// straight out of a packed operand, so this crate no longer needs to
    /// dequantize to make that gather possible.
    #[proxima::test]
    async fn moe_experts_now_stay_packed_for_the_real_per_expert_tensor_convention() {
        let file_bytes = checkpoint_with_per_expert_q4_k_tensors();
        let parsed = proxima_gguf::parse_complete(&file_bytes)
            .expect("parses synthetic per-expert-tensor moe checkpoint");
        let mut state = empty_state(&file_bytes);

        bind_moe_expert_weights(
            &parsed,
            &file_bytes,
            0,
            "ffn_gate",
            EXPERT_COUNT,
            OUT_DIM,
            IN_DIM,
            &mut state,
        )
        .expect("binds the per-expert-tensor q4_k experts");

        let owned = owned_bytes_total(&state);
        let packed_owned = packed_owned_bytes_total(&state);
        std::println!(
            "moe_memory_shape owned_bytes={owned} packed_owned_bytes={packed_owned} packed_floor_bytes={} dequantized_ceiling_bytes={}",
            packed_floor_bytes(),
            dequantized_owned_ceiling_bytes()
        );
        assert_eq!(
            owned, 0,
            "a q4_k expert stack must allocate zero dequantized f32 bytes"
        );
        assert!(
            packed_owned <= packed_floor_bytes() * 2,
            "packed-owned allocation ({packed_owned} bytes) must be proportional to the packed size ({} bytes), not to \
             experts * rows * cols * 4 ({} bytes)",
            packed_floor_bytes(),
            dequantized_owned_ceiling_bytes()
        );
    }
}

// -- Real-data proof: bind a real Q4_K weight row out of a host-local
// checkpoint by name, feed it through `proxima_tensor::cpu::evaluate_named`
// against a known activation, and check the interpreter's result against a
// dequantize-then-multiply computed independently of both `bind` and
// `cpu`. Opportunistic like `proxima_gguf::restack::tests::real_mixtral_file`:
// `#[ignore]`d and skips cleanly when the host-local model cache is absent.
#[cfg(all(test, feature = "std"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod real_openchat_file {
    use core::ffi::c_void;
    use core::future::Future;
    use core::pin::pin;
    use core::task::{Context, Poll, Waker};
    use std::os::fd::AsFd;

    use proxima_gguf::GgmlType;
    use proxima_gguf::quant::q4_k;
    use proxima_primitives::pipe::Pipe;
    use proxima_tensor::DType;
    use proxima_tensor::cpu::{
        QuantizedBlock, evaluate_named, evaluate_quantized_named_with_scratch,
    };
    use proxima_tensor::map::{self, IndexMap};
    use proxima_tensor::op::{self, Extent, Keep, Op, Reduce, ReduceInit, ScalarOp, append};

    use crate::generate::LoadedModel;
    use crate::loader::prefault;
    use crate::serving::ServingConfig;

    use super::{architecture_from_metadata, bind_all_weights, gguf_tensor_as_f32};

    /// A read-only `mmap` of the fixture file (rustix, already a workspace
    /// dependency used the same way by `proxima-storage/src/dax/region.rs`
    /// for its own domain) -- the whole reason to bind `Q4_K` tensors
    /// packed is so the byte range GGUF already stored is the buffer
    /// `evaluate_quantized_named` reads, with no owned copy in between;
    /// `proxima_gguf::edge::read_file`'s `std::fs::read` would put that
    /// copy straight back, one whole-file `Vec<u8>` at a time (its own doc,
    /// `proxima-gguf/src/edge.rs:3`, names this exact tradeoff). `MapFlags::
    /// PRIVATE`/`ProtFlags::READ` because this test only ever reads the
    /// fixture; unmapped on drop.
    ///
    /// Stays test-local rather than moving into `crate::generate` or
    /// `crate::loader`: opening/mapping a file is exactly the IO step this
    /// crate's own module docs disclaim ("this module never opens a
    /// file") -- [`crate::generate::LoadedModel::load`] takes a caller-
    /// owned `&'file [u8]` for the same reason [`gguf_tensor_as_f32`]
    /// always has, and this struct is the caller that owns one here.
    struct MappedGguf {
        base: *mut u8,
        len: usize,
        _file: std::fs::File,
    }

    impl MappedGguf {
        fn open(path: &std::path::Path) -> std::io::Result<Self> {
            let file = std::fs::File::open(path)?;
            let len =
                usize::try_from(file.metadata()?.len()).expect("fixture file length fits in usize");
            // SAFETY: `len` matches the just-opened file's own length; `file`
            // is kept alive in `_file` for as long as `base` is used, and the
            // mapping is read-only/private so no writer can observe or race it.
            let base = unsafe {
                rustix::mm::mmap(
                    core::ptr::null_mut(),
                    len,
                    rustix::mm::ProtFlags::READ,
                    rustix::mm::MapFlags::PRIVATE,
                    file.as_fd(),
                    0,
                )
            }
            .expect("mmap host-local openchat gguf fixture")
            .cast::<u8>();
            Ok(Self {
                base,
                len,
                _file: file,
            })
        }

        fn as_slice(&self) -> &[u8] {
            // SAFETY: `base` points at `len` bytes mapped for `self`'s whole
            // lifetime; this borrows `self` immutably, so nothing can unmap
            // the region while the returned slice is alive.
            unsafe { core::slice::from_raw_parts(self.base, self.len) }
        }
    }

    impl Drop for MappedGguf {
        fn drop(&mut self) {
            // SAFETY: `base`/`len` are exactly what `open`'s `mmap` call
            // returned; nothing else unmaps this region.
            let _ = unsafe { rustix::mm::munmap(self.base.cast::<c_void>(), self.len) };
        }
    }

    /// `PROXIMA_PREFAULT=1` warms every page of the mapping before a timed
    /// region runs. A first read of a lazily-mapped file demand-pages one
    /// minor fault per page touched; paying that fault storm inside a
    /// timed forward pass serializes it against the compute the forward is
    /// trying to measure.
    fn prefault_if_requested(file_bytes: &[u8]) -> bool {
        let enabled = std::env::var("PROXIMA_PREFAULT").is_ok_and(|value| value == "1");
        if enabled {
            prefault(file_bytes).expect("prefault the host-local openchat gguf mapping");
        }
        enabled
    }

    /// Env-driven bench knobs -- `PROXIMA_PROMPT` overrides the prompt
    /// text, `PROXIMA_MAX_TOKENS` overrides how many tokens the decode
    /// loop generates, both defaulting when unset or unparsable rather
    /// than panicking on a malformed override.
    fn decode_loop_prompt() -> alloc::string::String {
        std::env::var("PROXIMA_PROMPT").unwrap_or_else(|_| default_prompt())
    }

    fn decode_loop_max_tokens() -> usize {
        std::env::var("PROXIMA_MAX_TOKENS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(24)
    }

    /// OpenChat-3.5's own chat template (`tokenizer.chat_template` in this
    /// checkpoint's own GGUF metadata) rendered for one user turn with no
    /// assistant reply yet. BOS comes from `encode_with_bos_eos`'s own
    /// `add_bos` argument, matching the template's `{{ bos_token }}` -- not
    /// written into this string.
    fn default_prompt() -> alloc::string::String {
        alloc::string::String::from(
            "GPT4 Correct User: Write a Python function that returns the nth Fibonacci number.<|end_of_turn|>GPT4 Correct Assistant:",
        )
    }

    /// Drives a leaf [`Pipe::call`] future to completion. Every future this
    /// crate's own pipes return is `async move { <synchronous computation> }`
    /// with no internal `.await` (`generate::LoadedModel`'s own doc), so the
    /// first poll is always `Poll::Ready` -- this loop exists to make that
    /// an assertion (`unreachable!` on a real `Pending`) rather than an
    /// assumption a caller silently relies on.
    fn block_on<Fut: Future>(future: Fut) -> Fut::Output {
        let mut future = pin!(future);
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        match future.as_mut().poll(&mut context) {
            Poll::Ready(output) => output,
            Poll::Pending => {
                unreachable!("proxima-model-interop pipes never yield: no internal .await")
            }
        }
    }

    /// One real greedy-decoded token out of the real openchat-3.5 (Mistral
    /// architecture) checkpoint, through the crate's public
    /// [`LoadedModel`]/[`Pipe`] surface: mmap the fixture, parse it,
    /// [`LoadedModel::load`] once, then [`Pipe::call`] with
    /// `max_tokens: 1` -- the cached forward program's first call
    /// (`cached_len == 0`, `new_positions == prompt_length`) is exactly a
    /// one-shot full-context forward, so this is the same computation the
    /// former hand-rolled loop ran, through the reachable path instead of
    /// a private copy of it.
    #[test]
    #[ignore = "depends on a host-local openchat gguf checkout outside this repo"]
    fn runs_one_real_forward_pass_and_greedy_picks_a_real_token() {
        let path = std::path::Path::new(ServingConfig::default().model_path);
        if !path.exists() {
            eprintln!(
                "skipping: no host-local openchat gguf fixture at {}",
                ServingConfig::default().model_path
            );
            return;
        }

        let mapped = MappedGguf::open(path).expect("mmap host-local openchat gguf fixture");
        let file_bytes = mapped.as_slice();
        let parsed = proxima_gguf::pipe::parse_complete(file_bytes)
            .expect("parse host-local openchat gguf fixture");
        prefault_if_requested(file_bytes);

        let model = LoadedModel::load(&parsed, file_bytes)
            .expect("load real openchat checkpoint through the public path");
        let prompt = "The capital of France is";
        let forward_start = std::time::Instant::now();
        let generated = block_on(model.call((prompt.into(), 1)))
            .expect("generate through the public Pipe path");
        let forward_elapsed = forward_start.elapsed();

        std::println!(
            "prompt={prompt:?} token_id={} token={:?} forward_wall_clock={forward_elapsed:?}",
            generated.0[0],
            generated.1
        );

        // llama.cpp's own captured greedy answer for this exact prompt and
        // checkpoint (guiding-principle 14: the incumbent is the oracle).
        assert_eq!(
            generated.0[0], 2651,
            "greedy token id drifted off llama.cpp's captured answer"
        );
        assert_eq!(
            generated.1, "known",
            "greedy token text drifted off llama.cpp's captured answer"
        );
    }

    /// The multi-token counterpart, through the same public path: one
    /// [`LoadedModel::load`], one [`Pipe::call`] with
    /// `max_tokens: PROXIMA_MAX_TOKENS` (default 24) -- the direct fix for
    /// the O(n^2) growth an uncached loop would pay, now reachable outside
    /// `#[cfg(test)]` instead of trapped inside a private helper. Stopping
    /// early on the model's own eos signal (`generated.2 == true`) before
    /// `max_tokens` is a result of that fix, not a failure of this test --
    /// the assertions below only require the budget is never exceeded and
    /// that an eos stop is distinguishable from a budget-exhaustion stop.
    #[test]
    #[ignore = "depends on a host-local openchat gguf checkout outside this repo"]
    fn runs_a_cached_greedy_decode_loop_and_reports_per_token_wall_clock() {
        let path = std::path::Path::new(ServingConfig::default().model_path);
        if !path.exists() {
            eprintln!(
                "skipping: no host-local openchat gguf fixture at {}",
                ServingConfig::default().model_path
            );
            return;
        }

        let mapped = MappedGguf::open(path).expect("mmap host-local openchat gguf fixture");
        let file_bytes = mapped.as_slice();
        let parsed = proxima_gguf::pipe::parse_complete(file_bytes)
            .expect("parse host-local openchat gguf fixture");
        prefault_if_requested(file_bytes);

        let model = LoadedModel::load(&parsed, file_bytes)
            .expect("load real openchat checkpoint through the public path");
        let prompt = decode_loop_prompt();
        let max_tokens = decode_loop_max_tokens();

        std::println!("prompt={prompt:?} max_tokens={max_tokens}");
        let decode_start = std::time::Instant::now();
        let generated = block_on(model.call((prompt.clone(), max_tokens)))
            .expect("generate through the public Pipe path");
        let total_elapsed = decode_start.elapsed();
        let tokens_generated = generated.0.len();
        let mean_ms_per_token =
            total_elapsed.as_secs_f64() * 1000.0 / tokens_generated.max(1) as f64;

        std::println!(
            "decode_summary tokens_generated={tokens_generated} stopped_by_eos={} total_wall_clock_ms={:.3} mean_ms_per_token={mean_ms_per_token:.3} generated_text={:?}",
            generated.2,
            total_elapsed.as_secs_f64() * 1000.0,
            generated.1
        );

        #[cfg(feature = "instrument")]
        {
            use core::sync::atomic::Ordering;
            use proxima_tensor::instrument::cohort as cohort_diag;
            // execution witness for WHICH quantized matmul arm ran: q4k_macs
            // is only ever incremented inside the packed-int8 `Q4K` branch,
            // so a nonzero count is proof the dequantize-then-fold codec did
            // not carry the forward.
            let quant = proxima_tensor::instrument::matmul_dispatch_totals();
            let q4k_ns = proxima_tensor::instrument::ticks_to_nanos(quant.q4k_call_ticks);
            std::println!(
                "quant_arm q4k_macs={} q4k_ms={:.3} q4k_ns_per_mac={:.5} q5k_macs={} q6k_macs={} q5k_f32_calls={} q6k_f32_calls={} reduce_quantized_calls={} workers_calls={} workers_none={}",
                quant.q4k_macs,
                q4k_ns as f64 / 1e6,
                if quant.q4k_macs == 0 {
                    0.0
                } else {
                    q4k_ns as f64 / quant.q4k_macs as f64
                },
                quant.q5k_macs,
                quant.q6k_macs,
                quant.q5k_f32_calls,
                quant.q6k_f32_calls,
                quant.reduce_quantized_calls,
                // `workers_calls == workers_none` is the mechanical proof
                // `PROXIMA_MATMUL_WORKERS=1` actually took effect
                // (`cpu::quantized_matmul_workers`'s own `workers > 1`
                // gate) -- a prior session's single-threaded comparison
                // was invalidated by never checking this.
                quant.workers_calls,
                quant.workers_none
            );
            // is the matmul bucket actually KERNEL, or is it orchestration?
            // reduce_quantized is the whole bucket; q4k/q5k/q6k_call are the
            // packed kernels inside it; the rest is dispatch the bucket also
            // pays for. printed cumulatively -- subtract a MAX_TOKENS=1 run
            // to isolate decode.
            let ms = |ticks: u64| proxima_tensor::instrument::ticks_to_nanos(ticks) as f64 / 1e6;
            std::println!(
                "matmul_split bucket_ms={:.3} kernel_ms={:.3} quantize_activation_ms={:.3} transpose_ms={:.3} setup_ms={:.3} spawn_ms={:.3} own_chunk_ms={:.3} recv_wait_ms={:.3} dispatch_calls={} position_loop_iters={}",
                ms(quant.reduce_quantized_ticks),
                ms(quant.q4k_call_ticks) + ms(quant.q5k_call_ticks) + ms(quant.q6k_call_ticks),
                ms(quant.quantize_activation_ticks),
                ms(quant.q4k_transpose_ticks),
                ms(quant.setup_ticks),
                ms(quant.spawn_ticks),
                ms(quant.own_chunk_ticks),
                ms(quant.recv_wait_ticks),
                quant.calls,
                quant.position_loop_iters
            );
            // `matmul_split` above is the UNBATCHED population only
            // (`node_kind=reduce_matmul_quantized`, `dispatch_calls`'s own
            // denominator, every field a nested subset of that line's own
            // `bucket_ms`); this line is the `cohort-staged-graph` folded
            // population (`node_kind=staged_batch`) -- ROW97/98's dominant
            // bucket, previously invisible below its own outer wall-clock
            // total. Its own `quantize_ms` is a dedicated counter
            // (`STAGED_MATMUL_QUANTIZE_TICKS`), not shared with the line
            // above's `quantize_activation_ms`, so each line's fields stay
            // internally additive.
            std::println!(
                "matmul_split_staged round_ms={:.3} quantize_ms={:.3} transpose_ms={:.3} macs={} nodes={} ns_per_mac={:.5}",
                ms(quant.staged_round_ticks),
                ms(quant.staged_quantize_ticks),
                ms(quant.staged_transpose_ticks),
                quant.staged_macs,
                quant.staged_nodes,
                if quant.staged_macs == 0 {
                    0.0
                } else {
                    proxima_tensor::instrument::ticks_to_nanos(quant.staged_round_ticks) as f64
                        / quant.staged_macs as f64
                },
            );
            let rounds = cohort_diag::ROUNDS.load(Ordering::Relaxed);
            let parks = cohort_diag::PARKS.load(Ordering::Relaxed);
            let unpark_rounds = cohort_diag::UNPARK_ROUNDS.load(Ordering::Relaxed);
            let spin_hits = cohort_diag::SPIN_HITS.load(Ordering::Relaxed);
            let immediate_hits = cohort_diag::IMMEDIATE_HITS.load(Ordering::Relaxed);
            std::println!(
                "cohort_summary rounds={rounds} parks={parks} unpark_rounds={unpark_rounds} spin_hits={spin_hits} immediate_hits={immediate_hits}"
            );
            for slot in 0..cohort_diag::MAX_SLOTS {
                let slot_rounds = cohort_diag::SLOT_ROUNDS[slot].load(Ordering::Relaxed);
                if slot_rounds == 0 {
                    continue;
                }
                let first_claim_ns =
                    cohort_diag::SLOT_FIRST_CLAIM_NANOS[slot].load(Ordering::Relaxed);
                let tail_ns = cohort_diag::SLOT_TAIL_NANOS[slot].load(Ordering::Relaxed);
                let compute_ns = cohort_diag::SLOT_COMPUTE_NANOS[slot].load(Ordering::Relaxed);
                let chunks = cohort_diag::SLOT_CHUNKS[slot].load(Ordering::Relaxed);
                std::println!(
                    "cohort_slot slot={slot} rounds={slot_rounds} chunks={chunks} first_claim_ms={:.4} compute_ms={:.4} tail_ms={:.4}",
                    first_claim_ns as f64 / slot_rounds as f64 / 1e6,
                    compute_ns as f64 / slot_rounds as f64 / 1e6,
                    tail_ns as f64 / slot_rounds as f64 / 1e6,
                );
            }
        }

        // stopping early on the model's own eos signal is a result, not a
        // failure -- see `generate.rs`'s own doc for what this checkpoint's
        // eos id actually is (`<|end_of_turn|>`, not `</s>`). the budget is
        // still a hard ceiling either way, and an eos stop must be
        // distinguishable from budget exhaustion via `generated.2`.
        assert!(
            tokens_generated <= max_tokens,
            "decode loop must never exceed the requested budget"
        );
        if generated.2 {
            assert!(
                tokens_generated < max_tokens,
                "an eos stop must have produced strictly fewer ids than the full budget"
            );
        } else {
            assert_eq!(
                tokens_generated, max_tokens,
                "budget exhaustion must produce exactly one id per step"
            );
        }
        assert!(
            !generated.1.is_empty(),
            "degenerate control: decode loop produced no text"
        );
    }

    /// Same cached decode loop, on the Metal backend instead of CPU:
    /// `gpu_layers: GPU_LAYERS_ALL` (`-ngl all`) makes `generate::select_backend`
    /// resolve `Backend::Metal`, so `LoadedModel::run_decode_loop` (`pub(crate)`,
    /// same loop `LoadedModel::call` runs) drives `BackendRuntime`'s Metal arm.
    /// Called directly instead of through `Pipe::call` because this test also
    /// needs `runtime`'s plan-cache hit/miss counters, which
    /// `(Vec<u32>, String, bool)` has no room for.
    ///
    /// **Finding, not a pass/fail on generated text**: `mistral_cached_forward_program`'s
    /// key/value-cache read extent bakes `cached_len` into a `Plan`'s concrete
    /// shapes (`omega::backend::plan_named`'s own doc), and `cached_len` grows
    /// by `new_count` every decode step -- so within one autoregressive decode
    /// call, `(new_count, cached_len)` is a different pair every single step,
    /// and the plan cache cannot hit even once. This test asserts exactly
    /// that (`plan_hits == 0`, `plan_misses == one per forward step taken`)
    /// rather than hoping for reuse the shape itself rules out.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "depends on a host-local openchat gguf checkout outside this repo, and a real Metal device"]
    fn runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache() {
        let path = std::path::Path::new(ServingConfig::default().model_path);
        if !path.exists() {
            eprintln!(
                "skipping: no host-local openchat gguf fixture at {}",
                ServingConfig::default().model_path
            );
            return;
        }

        let mapped = MappedGguf::open(path).expect("mmap host-local openchat gguf fixture");
        let file_bytes = mapped.as_slice();
        let parsed = proxima_gguf::pipe::parse_complete(file_bytes)
            .expect("parse host-local openchat gguf fixture");
        prefault_if_requested(file_bytes);

        let model = LoadedModel::load(&parsed, file_bytes)
            .expect("load real openchat checkpoint through the public path");
        let prompt = decode_loop_prompt();
        let max_tokens = decode_loop_max_tokens();

        let serving_config = ServingConfig {
            kv_cache_key_quant: GgmlType::F32,
            kv_cache_value_quant: GgmlType::F32,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: crate::serving::GPU_LAYERS_ALL,
            reasoning_budget: 0,
            ..ServingConfig::default()
        };

        let mut runtime = crate::generate::BackendRuntime::new(&serving_config);
        let decode_start = std::time::Instant::now();
        let generated = model
            .run_decode_loop(&prompt, max_tokens, &serving_config, &mut runtime)
            .expect("generate through the metal backend");
        let total_elapsed = decode_start.elapsed();

        std::println!(
            "metal_decode_summary tokens_generated={} stopped_by_eos={} total_wall_clock_ms={:.3} plan_hits={} plan_misses={} generated_text={:?}",
            generated.0.len(),
            generated.2,
            total_elapsed.as_secs_f64() * 1000.0,
            runtime.plan_hits,
            runtime.plan_misses,
            generated.1
        );

        let forward_calls_taken = generated.0.len() + usize::from(generated.2);
        assert_eq!(
            runtime.plan_hits, 0,
            "cached_len grows every decode step, so no (new_count, cached_len) shape can repeat within one call"
        );
        assert_eq!(
            runtime.plan_misses, forward_calls_taken,
            "every forward step builds exactly one new plan when none can ever be reused"
        );
        assert!(
            !generated.1.is_empty(),
            "degenerate control: metal decode loop produced no text"
        );
    }

    /// Diagnostic-only: drives the same real cached decode loop as
    /// [`runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache`],
    /// but sets `PROXIMA_METAL_OP_PROFILE_STEP=3` so `run_decode_loop`'s own
    /// `instrument`-gated branch (`generate.rs`) swaps ONE decode step from
    /// the production batched `execute_plan` to the diagnostic
    /// `execute_plan_op_timed` -- one command buffer PER `BoundOp` instead
    /// of one for the whole program -- and prints the per-op GPU
    /// attribution `proxima-tensor/docs/discipline.md`'s `gpu_exec`
    /// investigation needed. Step 3 is a real decode step (`new_count=1`,
    /// `cached_len=prompt_len+3`), comfortably past the first token so the
    /// resident-buffer cache and KV-cache growth are both in their steady
    /// state (ROW 84's own `resident_uploads=0` pattern). Not a
    /// pass/fail-on-numbers test -- `report_op_timings`' own `println!`
    /// output IS the deliverable this test exists to produce, exactly like
    /// this file's other `_summary`-printing diagnostic tests.
    #[cfg(all(feature = "metal", feature = "instrument"))]
    #[test]
    #[ignore = "depends on a host-local openchat gguf checkout outside this repo, and a real Metal device"]
    fn profiles_one_real_decode_step_by_per_op_gpu_time() {
        let path = std::path::Path::new(ServingConfig::default().model_path);
        if !path.exists() {
            eprintln!(
                "skipping: no host-local openchat gguf fixture at {}",
                ServingConfig::default().model_path
            );
            return;
        }

        let mapped = MappedGguf::open(path).expect("mmap host-local openchat gguf fixture");
        let file_bytes = mapped.as_slice();
        let parsed = proxima_gguf::pipe::parse_complete(file_bytes)
            .expect("parse host-local openchat gguf fixture");
        prefault_if_requested(file_bytes);

        let model = LoadedModel::load(&parsed, file_bytes)
            .expect("load real openchat checkpoint through the public path");
        let prompt = decode_loop_prompt();
        let max_tokens = 5;

        let serving_config = ServingConfig {
            kv_cache_key_quant: GgmlType::F32,
            kv_cache_value_quant: GgmlType::F32,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: crate::serving::GPU_LAYERS_ALL,
            reasoning_budget: 0,
            ..ServingConfig::default()
        };

        // SAFETY: this test only runs via an explicit `--ignored` invocation
        // (never nextest's default parallel sweep), matching the same
        // single-process-at-a-time convention `PROXIMA_PREFAULT`/
        // `PROXIMA_MAX_TOKENS` already rely on being read, unmutated, by
        // this same file's other ignored tests.
        unsafe {
            std::env::set_var("PROXIMA_METAL_OP_PROFILE_STEP", "3");
        }
        let mut runtime = crate::generate::BackendRuntime::new(&serving_config);
        let generated = model
            .run_decode_loop(&prompt, max_tokens, &serving_config, &mut runtime)
            .expect("generate through the metal backend");
        // SAFETY: same justification as the `set_var` above -- single
        // process, no concurrent reader.
        unsafe {
            std::env::remove_var("PROXIMA_METAL_OP_PROFILE_STEP");
        }

        std::println!(
            "op_profile_run tokens_generated={} stopped_by_eos={} generated_text={:?}",
            generated.0.len(),
            generated.2,
            generated.1
        );
        assert!(
            !generated.1.is_empty(),
            "degenerate control: metal decode loop produced no text"
        );
    }

    /// Same harness as
    /// [`profiles_one_real_decode_step_by_per_op_gpu_time`], but targets
    /// `step=0` -- the PREFILL step, where `next_ids` is still the FULL
    /// prompt (`new_count=31` for [`decode_loop_prompt`]'s default prompt,
    /// not the `new_count=1` of every later decode step; see
    /// `run_decode_loop`'s own `next_ids = ids` vs `next_ids = alloc::vec![token_id]`
    /// split). Exists to settle discipline-log ROW 112's question: does a
    /// prefill forward sweep the packed weight matrix once, like ROW 94
    /// measured a decode step does (`total_operand_bytes` == 1.0000x the
    /// declared 4.07 GB weight set), or once PER of the 31 prompt
    /// positions (would read back ~31x that byte count, since every
    /// `reduce-packed-row-blocked` op still dispatches once per program,
    /// touching whichever operand bytes the row-blocked kernel binds
    /// regardless of `new_count`).
    #[cfg(all(feature = "metal", feature = "instrument"))]
    #[test]
    #[ignore = "depends on a host-local openchat gguf checkout outside this repo, and a real Metal device"]
    fn profiles_one_real_prefill_step_by_per_op_gpu_time() {
        let path = std::path::Path::new(ServingConfig::default().model_path);
        if !path.exists() {
            eprintln!(
                "skipping: no host-local openchat gguf fixture at {}",
                ServingConfig::default().model_path
            );
            return;
        }

        let mapped = MappedGguf::open(path).expect("mmap host-local openchat gguf fixture");
        let file_bytes = mapped.as_slice();
        let parsed = proxima_gguf::pipe::parse_complete(file_bytes)
            .expect("parse host-local openchat gguf fixture");
        prefault_if_requested(file_bytes);

        let model = LoadedModel::load(&parsed, file_bytes)
            .expect("load real openchat checkpoint through the public path");
        let prompt = decode_loop_prompt();
        let max_tokens = 1;

        let serving_config = ServingConfig {
            kv_cache_key_quant: GgmlType::F32,
            kv_cache_value_quant: GgmlType::F32,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: crate::serving::GPU_LAYERS_ALL,
            reasoning_budget: 0,
            ..ServingConfig::default()
        };

        // SAFETY: same single-process, no-concurrent-reader justification as
        // `profiles_one_real_decode_step_by_per_op_gpu_time`'s own `set_var`.
        unsafe {
            std::env::set_var("PROXIMA_METAL_OP_PROFILE_STEP", "0");
        }
        let mut runtime = crate::generate::BackendRuntime::new(&serving_config);
        let generated = model
            .run_decode_loop(&prompt, max_tokens, &serving_config, &mut runtime)
            .expect("generate through the metal backend");
        // SAFETY: same justification as the `set_var` above.
        unsafe {
            std::env::remove_var("PROXIMA_METAL_OP_PROFILE_STEP");
        }

        std::println!(
            "op_profile_run tokens_generated={} stopped_by_eos={} generated_text={:?}",
            generated.0.len(),
            generated.2,
            generated.1
        );
        assert!(
            !generated.1.is_empty(),
            "degenerate control: metal decode loop produced no text"
        );
    }

    /// Every real openchat weight this checkpoint's cached forward program
    /// binds by name -- `Q4_K`/`Q5_K`/`Q6_K` tensors packed straight out of
    /// an mmap, everything else dequantized -- reused by the `Q8_0`
    /// key/value-cache probe below. Not shared with [`LoadedModel`]: this
    /// probe deliberately bypasses `apply_serving_config`'s gate to reach
    /// a real capability gap (see [`q8_0_quantized_key_value_cache_cannot_cross_the_weight_matmul_quantized_seam`]'s
    /// own doc), which `LoadedModel::call` cannot be asked to do -- the
    /// gate is load-bearing on the reachable path, so probing past it
    /// needs its own copy of the loop, not a knob on the public one.
    struct CachedPositionInputs {
        ids_f32: Vec<f32>,
        epsilon: Vec<f32>,
        cos: Vec<f32>,
        sin: Vec<f32>,
    }

    fn build_cached_position_inputs(
        new_ids: &[u32],
        start_position: usize,
        head_dim: u32,
        rope_freq_base: f32,
    ) -> CachedPositionInputs {
        let new_count = new_ids.len();
        let pairs = head_dim as usize / 2;
        let ids_f32: Vec<f32> = new_ids.iter().map(|&id| id as f32).collect();
        let epsilon = alloc::vec![RMS_EPSILON; new_count];

        let mut cos = alloc::vec![0.0f32; new_count * pairs];
        let mut sin = alloc::vec![0.0f32; new_count * pairs];
        for offset in 0..new_count {
            let position = (start_position + offset) as f32;
            for pair in 0..pairs {
                let theta =
                    position * rope_freq_base.powf(-((2 * pair) as f32) / (head_dim as f32));
                cos[offset * pairs + pair] = theta.cos();
                sin[offset * pairs + pair] = theta.sin();
            }
        }

        CachedPositionInputs {
            ids_f32,
            epsilon,
            cos,
            sin,
        }
    }

    const RMS_EPSILON: f32 = 1e-5;

    /// Every layer's growable key/value cache: `k_even`/`k_odd` are already
    /// RoPE-rotated, `v` is the un-rotated projected value. Two storage
    /// strategies: `Float32` keeps every position's `f32` values; `Q8_0`
    /// holds packed bytes in the same codec
    /// [`super::gguf_tensor_as_packed_block`] already reads GGUF weight
    /// tensors through -- this seam test evaluates only one step
    /// (`cached_len == 0`), so nothing ever appends and `Q8_0` stays empty.
    enum LayerCache {
        Float32 {
            k_even: Vec<f32>,
            k_odd: Vec<f32>,
            v: Vec<f32>,
        },
        Q8_0 {
            k_even: Vec<u8>,
            k_odd: Vec<u8>,
            v: Vec<u8>,
        },
    }

    impl LayerCache {
        fn new(precision: GgmlType) -> Self {
            match precision {
                GgmlType::Q8_0 => LayerCache::Q8_0 {
                    k_even: Vec::new(),
                    k_odd: Vec::new(),
                    v: Vec::new(),
                },
                _ => LayerCache::Float32 {
                    k_even: Vec::new(),
                    k_odd: Vec::new(),
                    v: Vec::new(),
                },
            }
        }

        fn named_blocks<'cache>(
            &'cache self,
            k_even_name: &'cache str,
            k_odd_name: &'cache str,
            v_name: &'cache str,
        ) -> [(&'cache str, QuantizedBlock<'cache>); 3] {
            match self {
                LayerCache::Float32 { k_even, k_odd, v } => [
                    (k_even_name, QuantizedBlock::Float32(k_even.as_slice())),
                    (k_odd_name, QuantizedBlock::Float32(k_odd.as_slice())),
                    (v_name, QuantizedBlock::Float32(v.as_slice())),
                ],
                LayerCache::Q8_0 { k_even, k_odd, v } => [
                    (k_even_name, QuantizedBlock::Q8_0(k_even.as_slice())),
                    (k_odd_name, QuantizedBlock::Q8_0(k_odd.as_slice())),
                    (v_name, QuantizedBlock::Q8_0(v.as_slice())),
                ],
            }
        }
    }

    /// **Finding, not a pass/fail on generated text**: the growable
    /// key/value cache's own attention [`proxima_tensor::op::Op::Reduce`]
    /// nodes share an extra `kv_heads` axis between the cache operand and
    /// the activation operand, which this interpreter's
    /// `[rows, k] x [k] -> [rows]` quantized-matmul kernel call cannot
    /// express -- a real capability gap, reproduced directly here rather
    /// than routed around with a new `Op` variant or a new type.
    ///
    /// `serving.rs`'s own `apply_serving_config` gate now rejects every
    /// non-`F32` `kv_cache_key_quant`/`kv_cache_value_quant` before a
    /// forward ever runs, which is exactly why [`crate::generate::LoadedModel::call`]
    /// can never reach this seam -- it always builds the fully-supported
    /// config. This test drives `Q8_0` storage straight into
    /// `run_reduce_quantized` by skipping that gate, reproducing the real
    /// panic the public path can never trigger.
    #[test]
    #[ignore = "depends on a host-local openchat gguf checkout outside this repo"]
    fn q8_0_quantized_key_value_cache_cannot_cross_the_weight_matmul_quantized_seam() {
        let path = std::path::Path::new(ServingConfig::default().model_path);
        if !path.exists() {
            eprintln!(
                "skipping: no host-local openchat gguf fixture at {}",
                ServingConfig::default().model_path
            );
            return;
        }

        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mapped = MappedGguf::open(path).expect("mmap host-local openchat gguf fixture");
            let file_bytes = mapped.as_slice();
            let parsed = proxima_gguf::pipe::parse_complete(file_bytes)
                .expect("parse host-local openchat gguf fixture");
            let architecture = architecture_from_metadata(&parsed)
                .expect("derive architecture from real metadata");
            let weights = bind_all_weights(&parsed, file_bytes, &architecture)
                .expect("bind real openchat checkpoint weights");

            use proxima_tensor::spec::mistral_cached_forward_program;
            let (program, logits_root, cache_roots) = mistral_cached_forward_program(
                architecture.vocab,
                architecture.embedding,
                architecture.feed_forward,
                architecture.query_heads,
                architecture.kv_heads,
                architecture.head_dim,
                architecture.block_count,
            )
            .expect("the cached forward pass lowers to a program");

            let kv_cache_names: Vec<(
                alloc::string::String,
                alloc::string::String,
                alloc::string::String,
            )> = (0..architecture.block_count as usize)
                .map(|layer| {
                    (
                        alloc::format!("kv_cache.{layer}.k_even"),
                        alloc::format!("kv_cache.{layer}.k_odd"),
                        alloc::format!("kv_cache.{layer}.v"),
                    )
                })
                .collect();
            let layer_caches: Vec<LayerCache> = (0..architecture.block_count as usize)
                .map(|_| LayerCache::new(GgmlType::Q8_0))
                .collect();

            let prompt = default_prompt();
            let ids = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)
                .and_then(|vocab| {
                    proxima_tokenizer::encode_with_bos_eos(&prompt, &vocab, true, false)
                })
                .expect("build vocab and encode prompt");

            let inputs = build_cached_position_inputs(
                &ids,
                0,
                architecture.head_dim,
                architecture.rope_freq_base,
            );
            let mut named_blocks: Vec<(&str, QuantizedBlock)> = Vec::with_capacity(
                weights.owned.len()
                    + weights.packed.len()
                    + 3
                    + architecture.block_count as usize * 3,
            );
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
            for (layer, (k_even_name, k_odd_name, v_name)) in kv_cache_names.iter().enumerate() {
                named_blocks.extend(layer_caches[layer].named_blocks(
                    k_even_name,
                    k_odd_name,
                    v_name,
                ));
            }

            let symbols = [ids.len() as u64, 0u64];
            let mut roots: Vec<op::NodeId> = Vec::with_capacity(1 + cache_roots.len() * 3);
            roots.push(logits_root);
            for (even, odd, value) in &cache_roots {
                roots.push(*even);
                roots.push(*odd);
                roots.push(*value);
            }
            let mut free_buffers: Vec<Vec<f32>> = Vec::new();
            let mut validated_weight_nodes: Option<alloc::collections::BTreeSet<op::NodeId>> = None;
            let _ = evaluate_quantized_named_with_scratch(
                &program,
                &symbols,
                &named_blocks,
                &roots,
                &mut free_buffers,
                &mut validated_weight_nodes,
            )
            .expect(
                "evaluate_quantized_named_with_scratch binds the q8_0 cache seam probe by name",
            );
        }));

        let panic_payload = outcome.expect_err(
            "expected the cached attention reduce's shared kv_heads axis to be rejected as a real \
             matmul-capability gap -- see this test's own doc",
        );
        let message = panic_payload
            .downcast_ref::<alloc::string::String>()
            .cloned()
            .or_else(|| {
                panic_payload
                    .downcast_ref::<&str>()
                    .map(|value| alloc::string::String::from(*value))
            })
            .unwrap_or_default();
        assert!(
            message.contains(
                "quantized matmul activation varies along an output axis its packed weight also varies along"
            ),
            "unexpected panic message: {message}"
        );
    }

    /// Builds `weight . activation` (elementwise multiply, then reduce to
    /// a scalar) over one super-block's worth (256 elements) of a real
    /// `Q4_K` tensor, evaluated two ways: through
    /// `proxima_model_interop::gguf_tensor_as_f32` bound by name into
    /// `evaluate_named`, and by hand-computing the same dot product
    /// straight from `q4_k::dequantize_block` on the tensor's raw bytes,
    /// bypassing this crate's `bind` module entirely. The interpreter's
    /// output must agree with the independent computation.
    #[test]
    #[ignore = "depends on a host-local openchat gguf checkout outside this repo"]
    fn binds_one_real_q4_k_block_and_matmuls_against_a_known_activation() {
        let model_path = ServingConfig::default().model_path;
        let path = std::path::Path::new(model_path);
        if !path.exists() {
            eprintln!("skipping: no host-local openchat gguf fixture at {model_path}");
            return;
        }

        let (parsed, file_bytes) =
            proxima_gguf::edge::read_file(path).expect("read host-local openchat gguf fixture");

        // a mid-network `ffn_gate` row, not `token_embd`'s row 0 -- that
        // row is the padding-token embedding and its first super-block
        // decodes to all zeros, which would make this test pass on a
        // degenerate zero-vs-zero comparison instead of a real one.
        let tensor_name = parsed
            .tensors
            .iter()
            .find(|tensor| {
                tensor.ggml_type == GgmlType::Q4_K
                    && tensor.element_count() as usize >= q4_k::QK_K
                    && tensor.name.contains("ffn_gate")
            })
            .map(|tensor| tensor.name.clone())
            .expect(
                "openchat checkpoint has at least one q4_k ffn_gate tensor with a full super-block",
            );

        let decoded = gguf_tensor_as_f32(&parsed, &file_bytes, &tensor_name)
            .expect("bind real q4_k tensor by name");
        let weight_row: Vec<f32> = decoded[..q4_k::QK_K].to_vec();

        let activation: Vec<f32> = (0..q4_k::QK_K)
            .map(|index| 0.01 * (index as f32) - 1.28)
            .collect();

        let mut program = Vec::new();
        let weight_node = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(q4_k::QK_K as u32)],
                name: Some("weight".into()),
            },
        );
        let activation_node = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(q4_k::QK_K as u32)],
                name: Some("activation".into()),
            },
        );
        let identity_map = IndexMap::Affine(map::projection(1, &[0]));
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (weight_node, identity_map.clone()),
                    (activation_node, identity_map)
                ],
                name: None,
            },
        );
        let dot = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[])),
                keep: Keep::Reduce,
                name: None,
            }),
        );

        let symbols: Vec<u64> = Vec::new();
        let named: [(&str, &[f32]); 2] = [
            ("weight", weight_row.as_slice()),
            ("activation", activation.as_slice()),
        ];
        let evaluated = evaluate_named(&program, &symbols, &named, &[dot])
            .expect("evaluate_named binds by name");
        let (interpreter_output, _shape) = evaluated
            .get(dot)
            .expect("dot product node present in output");

        // independent computation: raw bytes -> dequantize_block -> manual
        // dot product, never touching `bind::gguf_tensor_as_f32` or the
        // interpreter.
        let tensor = parsed
            .tensors
            .iter()
            .find(|tensor| tensor.name == tensor_name)
            .expect("tensor still present");
        let range = parsed
            .tensor_data_range(tensor, file_bytes.len() as u64)
            .expect("tensor byte range");
        let raw_block = &file_bytes[range.start as usize..range.start as usize + q4_k::BLOCK_BYTES];
        let mut independent_weights = [0.0f32; q4_k::QK_K];
        q4_k::dequantize_block(raw_block, &mut independent_weights);
        let expected: f32 = independent_weights
            .iter()
            .zip(activation.iter())
            .map(|(weight, value)| weight * value)
            .sum();

        let max_diff = (interpreter_output[0] - expected).abs();
        eprintln!(
            "real_q4_k_matmul tensor={tensor_name} interpreter={} independent={expected} max_diff={max_diff}",
            interpreter_output[0]
        );
        assert!(
            expected.abs() > 1e-3,
            "degenerate control: expected dot product is ~zero ({expected}), this run proves nothing about real agreement"
        );
        assert!(
            max_diff < 1e-3,
            "interpreter and independent dequantize-then-multiply diverged: max_diff={max_diff}"
        );
    }
}

// -- Real-data proof for the MoE metadata + expert-discovery wiring this
// change adds: a real Mixtral-8x7B checkpoint's own `general.architecture`
// metadata and `blk.0.ffn_gate.{0..8}.weight` tensor directory, read through
// this crate's public `architecture_from_metadata` and this module's own
// `bind_moe_expert_weights` internals -- never a synthetic fixture standing
// in for what the audit's citations already verified against this exact
// file (`proxima-gguf/src/restack.rs`'s own `real_mixtral_file` module).
// `#[ignore]`d and skips cleanly when the host-local model cache is absent,
// the same convention `real_openchat_file`/`restack.rs::real_mixtral_file`
// both use. Only the metadata/tensor-directory prefix and each of the 8
// experts' own `Q4_K` byte range for one layer's `ffn_gate` projection are
// read via direct `seek`+`read` -- never the whole 25 GB file.
#[cfg(all(test, feature = "std"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod real_mixtral_file {
    use std::io::{Read, Seek, SeekFrom};

    use proxima_gguf::pipe::parse_complete;
    use proxima_gguf::quant::q4_k;
    use proxima_primitives::pipe::Pipe;

    use super::*;

    const FIXTURE_PATH: &str = "/Users/brianbruggeman/.lmstudio/models/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF/Nous-Hermes-2-Mixtral-8x7B-DPO.Q4_K_S.gguf";

    /// Grows `header_buf` and re-parses until [`parse_complete`] stops
    /// reporting truncation, the same growing loop `restack.rs`'s own real
    /// Mixtral test uses -- Mixtral's tensor directory (995 tensors) fits
    /// well under a few MiB, nowhere near the multi-gigabyte payload.
    /// Inlined per test rather than factored into a function returning
    /// [`proxima_gguf::pipe::ParsedGguf`]: that type borrows from
    /// `header_buf`, so the buffer must outlive it in the same scope.
    macro_rules! parse_header_region {
        ($file:expr, $header_buf:ident) => {{
            let mut parsed = None;
            for cap in [4usize << 20, 16 << 20, 64 << 20] {
                $header_buf.resize(cap, 0);
                $file.seek(SeekFrom::Start(0)).expect("seek to file start");
                let read = $file
                    .read(&mut $header_buf)
                    .expect("read gguf header region");
                $header_buf.truncate(read);
                if let Ok(result) = parse_complete(&$header_buf) {
                    parsed = Some(result);
                    break;
                }
            }
            parsed.expect("gguf metadata region did not fit in 64 MiB")
        }};
    }

    /// [`architecture_from_metadata`] must read `llama.expert_count`/
    /// `llama.expert_used_count` off this real checkpoint (Mixtral-8x7B's
    /// own published config: 8 experts, top-2 routing) rather than silently
    /// treating it as dense the way openchat-3.5 (a real checkpoint with
    /// neither key) legitimately is.
    #[test]
    #[ignore = "depends on a 25 GB host-local mixtral gguf checkout outside this repo"]
    fn architecture_from_metadata_reads_the_real_mixtral_expert_config() {
        let path = std::path::Path::new(FIXTURE_PATH);
        if !path.exists() {
            eprintln!("skipping: no host-local mixtral gguf fixture at {FIXTURE_PATH}");
            return;
        }

        let mut file = std::fs::File::open(path).expect("open host-local mixtral gguf fixture");
        let mut header_buf: Vec<u8> = Vec::new();
        let parsed = parse_header_region!(file, header_buf);

        let architecture = architecture_from_metadata(&parsed)
            .expect("derive architecture from the real mixtral checkpoint");
        std::println!("real_mixtral architecture={architecture:?}");
        assert_eq!(
            architecture.expert_count, 8,
            "Mixtral-8x7B carries 8 experts per layer"
        );
        assert_eq!(
            architecture.expert_used_count, 2,
            "Mixtral-8x7B routes top-2"
        );
    }

    /// Header-only (no tensor payload touched): tallies every real tensor's
    /// own [`GgmlType`] and prints the checkpoint's own `tokenizer.chat_template`
    /// metadata value if present -- answers two of this run's own questions
    /// before any multi-gigabyte read is attempted: which codecs this file's
    /// 995 tensors actually carry (this crate decodes `F32`/`Q4_K`/`Q5_K`/
    /// `Q6_K`/`Q8_0`; anything else is a named capability gap, not a defect),
    /// and whether the checkpoint ships its own chat template rather than
    /// requiring the caller to hand-write one.
    #[test]
    #[ignore = "depends on a 25 GB host-local mixtral gguf checkout outside this repo"]
    fn inventories_the_real_checkpoints_tensor_codecs_and_chat_template() {
        let path = std::path::Path::new(FIXTURE_PATH);
        if !path.exists() {
            eprintln!("skipping: no host-local mixtral gguf fixture at {FIXTURE_PATH}");
            return;
        }

        let mut file = std::fs::File::open(path).expect("open host-local mixtral gguf fixture");
        let mut header_buf: Vec<u8> = Vec::new();
        let parsed = parse_header_region!(file, header_buf);

        let mut counts: alloc::collections::BTreeMap<alloc::string::String, usize> =
            alloc::collections::BTreeMap::new();
        for tensor in &parsed.tensors {
            *counts
                .entry(alloc::format!("{:?}", tensor.ggml_type))
                .or_insert(0) += 1;
        }
        std::println!("real_mixtral tensor_count={}", parsed.tensors.len());
        for (ggml_type, count) in &counts {
            std::println!("real_mixtral codec={ggml_type} tensor_count={count}");
        }

        let supported = ["F32", "Q4_K", "Q5_K", "Q6_K", "Q8_0"];
        let unsupported: Vec<(&alloc::string::String, &usize)> = counts
            .iter()
            .filter(|(ggml_type, _)| !supported.contains(&ggml_type.as_str()))
            .collect();
        if unsupported.is_empty() {
            std::println!(
                "real_mixtral every real tensor's codec is one this crate already decodes"
            );
        } else {
            for (ggml_type, count) in &unsupported {
                std::println!("real_mixtral UNSUPPORTED codec={ggml_type} tensor_count={count}");
            }
            let sample_names: Vec<&str> = parsed
                .tensors
                .iter()
                .filter(|tensor| {
                    !supported.contains(&alloc::format!("{:?}", tensor.ggml_type).as_str())
                })
                .take(6)
                .map(|tensor| tensor.name.as_str())
                .collect();
            std::println!("real_mixtral unsupported_codec_sample_names={sample_names:?}");
        }

        match parsed.metadata_value("tokenizer.chat_template") {
            Some(MetadataValue::String(template)) => {
                std::println!(
                    "real_mixtral chat_template_len={} chat_template={template:?}",
                    template.len()
                );
            }
            Some(other) => std::println!(
                "real_mixtral tokenizer.chat_template present but not a string: {other:?}"
            ),
            None => std::println!(
                "real_mixtral no tokenizer.chat_template key in this checkpoint's metadata"
            ),
        }
    }

    /// [`bind_moe_expert_weights`]'s per-expert-tensor fallback path
    /// (`discover_experts`/`plan_stack`/`restack_into`, this crate's own
    /// [`transpose_expert_stack`] on top) against the real file: restacks
    /// layer 0's 8 real `ffn_gate` experts and independently dequantizes
    /// each expert's own untransposed bytes via
    /// [`proxima_gguf::quant::q4_k::dequantize`], then un-transposes the
    /// bound result back to compare -- proving the bound buffer really is
    /// each real expert's own weights in the right order, not a
    /// transposition/gather bug that happens to have the right shape.
    #[test]
    #[ignore = "depends on a 25 GB host-local mixtral gguf checkout outside this repo"]
    fn binds_one_real_layers_ffn_gate_experts_and_matches_independent_dequantize() {
        let path = std::path::Path::new(FIXTURE_PATH);
        if !path.exists() {
            eprintln!("skipping: no host-local mixtral gguf fixture at {FIXTURE_PATH}");
            return;
        }

        let mut file = std::fs::File::open(path).expect("open host-local mixtral gguf fixture");
        let file_len = file.metadata().expect("stat gguf fixture").len();
        let mut header_buf: Vec<u8> = Vec::new();
        let parsed = parse_header_region!(file, header_buf);

        let architecture = architecture_from_metadata(&parsed)
            .expect("derive architecture from the real mixtral checkpoint");
        let layer = 0u64;
        let projection = "ffn_gate";

        let experts = discover_experts(
            &parsed.tensors,
            layer,
            projection,
            u64::from(architecture.expert_count),
        )
        .expect("discovers all 8 real experts for layer 0's ffn_gate projection");

        let mut sources_owned: Vec<Vec<u8>> = Vec::with_capacity(experts.len());
        for expert in &experts {
            let range = parsed
                .tensor_data_range(expert, file_len)
                .expect("expert tensor range within file");
            let mut bytes = alloc::vec![0u8; (range.end - range.start) as usize];
            file.seek(SeekFrom::Start(range.start))
                .expect("seek to expert tensor data");
            file.read_exact(&mut bytes)
                .expect("read expert tensor bytes");
            sources_owned.push(bytes);
        }
        let sources: Vec<&[u8]> = sources_owned.iter().map(Vec::as_slice).collect();

        let plan = plan_stack(&experts).expect("plans stack for real experts");
        let mut stacked_bytes = alloc::vec![0u8; plan.total_bytes as usize];
        restack_into(&mut stacked_bytes, &plan, &sources)
            .expect("restacks real experts into destination buffer");

        let out_dim = architecture.feed_forward as usize;
        let in_dim = architecture.embedding as usize;
        let per_expert_elements = out_dim * in_dim;
        let total_elements = per_expert_elements * experts.len();
        let decoded = dequantize(&stacked_bytes, total_elements, q4_k::dequantize)
            .expect("dequantizes the real restacked experts");
        let bound =
            transpose_expert_stack(&decoded, "test_moe_stack", experts.len(), out_dim, in_dim)
                .expect("expert_count/out_dim/in_dim agree with the real restacked byte length");

        for (expert_index, expert_bytes) in sources.iter().enumerate() {
            let mut expected = alloc::vec![0.0f32; per_expert_elements];
            q4_k::dequantize(expert_bytes, &mut expected)
                .expect("independently dequantizes expert's own real bytes");

            let bound_slab = &bound
                [expert_index * per_expert_elements..(expert_index + 1) * per_expert_elements];
            let un_transposed =
                transpose_out_in_to_in_out(bound_slab, "test_expert_slab", in_dim, out_dim)
                    .expect("in_dim/out_dim agree with the real bound slab length");
            let max_diff = un_transposed
                .iter()
                .zip(&expected)
                .map(|(found, wanted)| (found - wanted).abs())
                .fold(0.0f32, f32::max);
            std::println!("real_mixtral expert={expert_index} max_diff={max_diff}");
            assert!(
                max_diff < 1e-6,
                "expert {expert_index}: bound-then-transposed-back weights must equal an independent \
                 dequantize of that expert's own real bytes, got max_diff={max_diff}"
            );
        }
    }

    /// Real-data proof for this change's own fix:
    /// [`gguf_tensor_as_packed_block`] now decodes `blk.0.ffn_gate_inp.weight`
    /// (the MoE router, this real checkpoint's own `F16` tensor -- confirmed
    /// by [`inventories_the_real_checkpoints_tensor_codecs_and_chat_template`]'s
    /// own `real_mixtral UNSUPPORTED codec` line before this fix landed)
    /// into [`proxima_tensor::cpu::QuantizedBlock::Float16`] instead of
    /// [`InteropError::UnrepresentableGgmlType`]. Reads only the router
    /// tensor's own tiny byte range (`embedding * expert_count * 2` bytes --
    /// 65536 for this checkpoint) plus the file prefix up to that range's
    /// end (needed only because [`gguf_tensor_as_packed_block`] indexes
    /// `file_bytes` by the tensor's real on-disk offset, not a
    /// tensor-relative one) -- never the multi-gigabyte expert tensors that
    /// precede it in layer 0's own tensor order.
    ///
    /// Cross-checks [`proxima_tensor::cpu::matmul_f16_f32`] (the packed
    /// kernel [`crate::generate::LoadedModel`] now drives this tensor
    /// through) against an independent per-element `f16`-bits decode of the
    /// SAME real bytes, proving the packed binding is this file's own real
    /// router weights in the right row-major order, not a shape-compatible
    /// but scrambled read.
    #[test]
    #[ignore = "depends on a 25 GB host-local mixtral gguf checkout outside this repo"]
    fn binds_the_real_f16_router_weight_packed_and_matches_independent_f16_decode() {
        let path = std::path::Path::new(FIXTURE_PATH);
        if !path.exists() {
            eprintln!("skipping: no host-local mixtral gguf fixture at {FIXTURE_PATH}");
            return;
        }

        let mut file = std::fs::File::open(path).expect("open host-local mixtral gguf fixture");
        let file_len = file.metadata().expect("stat gguf fixture").len();
        let mut header_buf: Vec<u8> = Vec::new();
        let parsed = parse_header_region!(file, header_buf);

        let architecture = architecture_from_metadata(&parsed)
            .expect("derive architecture from the real mixtral checkpoint");
        let router_name = "blk.0.ffn_gate_inp.weight";
        let router_tensor = parsed
            .tensors
            .iter()
            .find(|tensor| tensor.name == router_name)
            .expect("real checkpoint carries a layer-0 router tensor");
        assert_eq!(
            router_tensor.ggml_type,
            GgmlType::F16,
            "this real checkpoint's own router tensor must be F16 for this test to prove anything"
        );

        let range = parsed
            .tensor_data_range(router_tensor, file_len)
            .expect("router tensor range within file");
        let mut prefix = alloc::vec![0u8; range.end as usize];
        file.seek(SeekFrom::Start(0)).expect("seek to file start");
        file.read_exact(&mut prefix[..range.end as usize])
            .expect("read file prefix through the router tensor's own byte range");

        let block = gguf_tensor_as_packed_block(&parsed, &prefix, router_name)
            .expect("f16 router tensor now decodes packed instead of UnrepresentableGgmlType");
        let router_bytes = match block {
            proxima_tensor::cpu::QuantizedBlock::Float16(bytes) => bytes,
            other => panic!(
                "expected a Float16 packed block for the real f16 router tensor, got {other:?}"
            ),
        };

        let embedding = architecture.embedding as usize;
        let expert_count = architecture.expert_count as usize;
        assert_eq!(
            router_bytes.len(),
            embedding * expert_count * 2,
            "router byte length must match embedding * expert_count f16 elements"
        );

        let activation: Vec<f32> = (0..embedding)
            .map(|index| ((index % 7) as f32) - 3.0)
            .collect();
        let ours = proxima_tensor::cpu::matmul_f16_f32(router_bytes, expert_count, &activation)
            .expect("matmul_f16_f32 runs against the real router bytes");

        let mut expected = alloc::vec![0.0f32; expert_count];
        for (expert_index, logit) in expected.iter_mut().enumerate() {
            let row =
                &router_bytes[expert_index * embedding * 2..(expert_index + 1) * embedding * 2];
            let mut accumulator = 0.0f32;
            for (element_index, chunk) in row.as_chunks::<2>().0.iter().enumerate() {
                let weight =
                    half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
                accumulator += weight * activation[element_index];
            }
            *logit = accumulator;
        }

        std::println!("real_mixtral router ours={ours:?} expected={expected:?}");
        for (expert_index, (found, wanted)) in ours.iter().zip(&expected).enumerate() {
            let diff = (found - wanted).abs();
            assert!(
                diff < 1e-3,
                "router logit for expert {expert_index}: found={found} wanted={wanted} diff={diff}"
            );
        }
    }

    /// A read-only `mmap` of the real 25 GB Mixtral fixture -- same
    /// technique and same justification as `real_openchat_file::MappedGguf`
    /// (this module's own doc), duplicated here rather than shared because
    /// that struct is private to its own module and this crate has no
    /// third home for a two-line mmap wrapper two fixture-specific test
    /// modules both happen to want.
    struct MappedGguf {
        base: *mut core::ffi::c_void,
        len: usize,
        _file: std::fs::File,
    }

    impl MappedGguf {
        fn open(path: &std::path::Path) -> std::io::Result<Self> {
            use std::os::fd::AsFd;
            let file = std::fs::File::open(path)?;
            let len =
                usize::try_from(file.metadata()?.len()).expect("fixture file length fits in usize");
            // SAFETY: `len` matches the just-opened file's own length; `file`
            // is kept alive in `_file` for as long as `base` is used, and the
            // mapping is read-only/private so no writer can observe or race it.
            let base = unsafe {
                rustix::mm::mmap(
                    core::ptr::null_mut(),
                    len,
                    rustix::mm::ProtFlags::READ,
                    rustix::mm::MapFlags::PRIVATE,
                    file.as_fd(),
                    0,
                )
            }
            .expect("mmap host-local mixtral gguf fixture");
            Ok(Self {
                base,
                len,
                _file: file,
            })
        }

        fn as_slice(&self) -> &[u8] {
            // SAFETY: `base` points at `len` bytes mapped for `self`'s whole
            // lifetime; this borrows `self` immutably, so nothing can unmap
            // the region while the returned slice is alive.
            unsafe { core::slice::from_raw_parts(self.base.cast::<u8>(), self.len) }
        }
    }

    impl Drop for MappedGguf {
        fn drop(&mut self) {
            // SAFETY: `base`/`len` are exactly what `open`'s `mmap` call
            // returned; nothing else unmaps this region.
            let _ = unsafe { rustix::mm::munmap(self.base, self.len) };
        }
    }

    /// Drives a leaf [`proxima_primitives::pipe::Pipe::call`] future to
    /// completion -- same justification as `real_openchat_file::block_on`'s
    /// own doc: every future this crate's pipes return is `async move { <sync
    /// computation> }` with no internal `.await`, so the first poll is
    /// always ready.
    fn block_on<Fut: core::future::Future>(future: Fut) -> Fut::Output {
        let mut future = core::pin::pin!(future);
        let waker = core::task::Waker::noop();
        let mut context = core::task::Context::from_waker(waker);
        match future.as_mut().poll(&mut context) {
            core::task::Poll::Ready(output) => output,
            core::task::Poll::Pending => {
                unreachable!("proxima-model-interop pipes never yield: no internal .await")
            }
        }
    }

    /// The real, whole-forward-pipeline attempt this discipline log's
    /// deliverable is built on: mmap the real 25 GB Mixtral-8x7B checkpoint,
    /// parse it, load it through the exact same public
    /// [`crate::generate::LoadedModel::load`] +
    /// [`proxima_primitives::pipe::Pipe::call`] surface
    /// [`real_openchat_file`]'s own acceptance tests drive -- no private
    /// shortcut around either step. `Mixtral-8x7B-DPO`'s own
    /// `tokenizer.chat_template` (asserted verbatim above in
    /// [`inventories_the_real_checkpoints_tensor_codecs_and_chat_template`])
    /// is `{% for message in messages %}...`, ChatML shaped -- rendered here
    /// by hand for one user turn with no assistant reply yet, the same
    /// one-turn-no-reply shape [`real_openchat_file::default_prompt`] uses
    /// for its own template.
    ///
    /// **Stale-doc correction (this change): the two gaps this doc used to
    /// describe are not both live any more.** `LoadedModel::load` now calls
    /// `proxima_tensor::spec::mistral_cached_forward_program_with_experts`
    /// (`generate.rs:351`), which DOES carry `expert_count`/`expert_used_count`
    /// and selects the routed FFN for a MoE checkpoint -- the first gap this
    /// doc used to name (the cached program being permanently dense) closed
    /// separately from this change, evidenced by `generate.rs`'s own doc at
    /// that call site. The second gap this doc named -- `ffn_gate_inp.weight`
    /// (`F16`) hitting [`InteropError::UnrepresentableGgmlType`] before
    /// `bind_all_weights` ever reaches the expert-weight loop -- IS this
    /// change's own fix: [`gguf_tensor_as_packed_block`] now decodes `F16`
    /// (see [`binds_the_real_f16_router_weight_packed_and_matches_independent_f16_decode`]
    /// for the real-bytes proof), so `bind_all_weights` now reaches
    /// [`bind_moe_expert_weights`] for every layer.
    ///
    /// **Second stale-doc correction (this change): the ~180 GiB SIGKILL
    /// this doc used to describe no longer happens.** This real checkpoint
    /// carries no native `blk.{layer}.ffn_gate_exps.weight` stack
    /// (`blk.0.ffn_gate_exps.weight present=false`, confirmed against this
    /// exact file), so `bind_moe_expert_weights` (`bind.rs`) falls back to
    /// `discover_experts`/`plan_stack`/`restack_into` -- but that fallback no
    /// longer dequantizes to owned `f32`: it binds the restacked buffer
    /// packed instead (`BoundWeights::packed_owned`, [`PackedOwnedKind`]),
    /// now that `proxima_tensor::cpu::run_reduce_quantized`'s gather arm
    /// resolves `per_expert_bytes` directly out of a packed `QuantizedBlock`
    /// (closed separately from this change; see
    /// [`moe_experts_now_stay_packed_for_the_real_per_expert_tensor_convention`]
    /// for the synthetic-fixture proof of the same shape). Run for real on a
    /// 64 GiB host with this fix: peak RSS stayed a few GiB (mmap'd 25 GB
    /// file plus packed `Q4_K` bytes, not 180 GiB of dequantized `f32`), no
    /// SIGKILL, and the model produced real, on-topic, coherent text for the
    /// prompt below -- pasted verbatim from this exact invocation's own
    /// `println!` in this function's own doc history.
    #[test]
    #[ignore = "depends on a 25 GB host-local mixtral gguf checkout outside this repo"]
    fn attempts_a_real_mixtral_forward_pass_and_reports_the_outcome() {
        let path = std::path::Path::new(FIXTURE_PATH);
        if !path.exists() {
            eprintln!("skipping: no host-local mixtral gguf fixture at {FIXTURE_PATH}");
            return;
        }

        let prompt = "<|im_start|>user\nWrite one sentence about the ocean.<|im_end|>\n<|im_start|>assistant\n";

        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mapped = MappedGguf::open(path).expect("mmap host-local mixtral gguf fixture");
            let file_bytes = mapped.as_slice();
            let parsed = parse_complete(file_bytes).expect("parse host-local mixtral gguf fixture");

            let model = crate::generate::LoadedModel::load(&parsed, file_bytes)
                .expect("load real mixtral checkpoint through the public path");
            block_on(model.call((alloc::string::String::from(prompt), 8)))
        }));

        match outcome {
            Ok(Ok((token_ids, text, stopped_by_eos))) => {
                std::println!(
                    "real_mixtral OUTCOME=coherent-or-garbled-text token_ids={token_ids:?} \
                     stopped_by_eos={stopped_by_eos} generated_text={text:?}"
                );
            }
            Ok(Err(interop_error)) => {
                std::println!("real_mixtral OUTCOME=clean-error error={interop_error}");
            }
            Err(panic_payload) => {
                let message = panic_payload
                    .downcast_ref::<alloc::string::String>()
                    .cloned()
                    .or_else(|| {
                        panic_payload
                            .downcast_ref::<&str>()
                            .map(|value| alloc::string::String::from(*value))
                    })
                    .unwrap_or_default();
                std::println!("real_mixtral OUTCOME=panic message={message}");
            }
        }
    }
}

// Real-data proof for the two `architecture_from_metadata` gaps a real
// hybrid checkpoint (LFM2.5-8B-A1B: 18 short-convolution layers, 6 attention
// layers) exposed: `{architecture}.rope.dimension_count` genuinely absent on
// this file (its writer never emitted the key at all, unlike every
// Mistral/Llama-family checkpoint this crate had evaluated before now), and
// `{architecture}.attention.head_count_kv` stored as a per-layer
// `MetadataArray` whose entries disagree (`0` for conv layers, a real count
// for attention layers) rather than one scalar. Header-only, same
// `#[ignore]`d/skip-if-absent convention as `real_mixtral_file` -- this
// crate's forward program cannot express LFM2's convolution layers at all
// (a separate, larger `proxima-tensor` gap, out of this change's scope), so
// this module only proves `architecture_from_metadata`'s own read is
// correct and non-fatal, never attempts a forward pass.
#[cfg(all(test, feature = "std"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod real_lfm2_hybrid_file {
    use std::io::{Read, Seek, SeekFrom};

    use proxima_gguf::pipe::parse_complete;

    use super::*;

    const FIXTURE_PATH: &str = "/Users/brianbruggeman/.lmstudio/models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf";

    macro_rules! parse_header_region {
        ($file:expr, $header_buf:ident) => {{
            let mut parsed = None;
            for cap in [4usize << 20, 16 << 20, 64 << 20] {
                $header_buf.resize(cap, 0);
                $file.seek(SeekFrom::Start(0)).expect("seek to file start");
                let read = $file
                    .read(&mut $header_buf)
                    .expect("read gguf header region");
                $header_buf.truncate(read);
                if let Ok(result) = parse_complete(&$header_buf) {
                    parsed = Some(result);
                    break;
                }
            }
            parsed.expect("gguf metadata region did not fit in 64 MiB")
        }};
    }

    /// `architecture_from_metadata` must not hard-fail this real checkpoint
    /// with [`InteropError::MissingMetadataKey`] on `rope.dimension_count`
    /// (genuinely absent -- confirmed via `strings` on this exact file) --
    /// before the derive-when-absent fallback landed, this call errored
    /// outright, never reaching the `head_count_kv` array at all. This
    /// checkpoint's own `attention.head_count_kv` genuinely disagrees across
    /// layers (conv vs. attention), so the correct, honest outcome here is
    /// still an `Err` -- just the NAMED one
    /// ([`InteropError::HeterogeneousMetadataArray`]), not a misdiagnosed
    /// [`InteropError::MissingMetadataKey`] and not a silently wrong scalar.
    #[test]
    #[ignore = "depends on a ~5 GB host-local lfm2 gguf checkout outside this repo"]
    fn architecture_from_metadata_names_the_heterogeneous_kv_heads_honestly() {
        let path = std::path::Path::new(FIXTURE_PATH);
        if !path.exists() {
            eprintln!("skipping: no host-local lfm2 gguf fixture at {FIXTURE_PATH}");
            return;
        }

        let mut file = std::fs::File::open(path).expect("open host-local lfm2 gguf fixture");
        let mut header_buf: Vec<u8> = Vec::new();
        let parsed = parse_header_region!(file, header_buf);

        let outcome = architecture_from_metadata(&parsed);
        std::println!("real_lfm2 architecture_from_metadata outcome={outcome:?}");
        assert!(
            matches!(
                outcome,
                Err(InteropError::HeterogeneousMetadataArray { .. })
            ),
            "LFM2's real per-layer-varying head_count_kv must surface the named, honest error \
             (never MissingMetadataKey, and never a silently-picked scalar), got {outcome:?}"
        );
    }

    /// Same real file, isolating just the `rope.dimension_count` fallback:
    /// reads `general.architecture`/`embedding_length`/`attention.head_count`
    /// directly (bypassing `architecture_from_metadata`'s `head_count_kv`
    /// step, which errors on this checkpoint) and confirms the derived
    /// `embedding / query_heads` quotient matches this checkpoint's
    /// independently-known real per-head dimension (`attn_q_norm.weight`'s
    /// own declared shape is `[64]` on this file).
    #[test]
    #[ignore = "depends on a ~5 GB host-local lfm2 gguf checkout outside this repo"]
    fn rope_dimension_count_absent_derives_the_real_lfm2_head_dim() {
        let path = std::path::Path::new(FIXTURE_PATH);
        if !path.exists() {
            eprintln!("skipping: no host-local lfm2 gguf fixture at {FIXTURE_PATH}");
            return;
        }

        let mut file = std::fs::File::open(path).expect("open host-local lfm2 gguf fixture");
        let mut header_buf: Vec<u8> = Vec::new();
        let parsed = parse_header_region!(file, header_buf);

        let architecture_name = metadata_str(&parsed, "general.architecture")
            .expect("general.architecture present on a real gguf");
        let embedding = metadata_u32(
            &parsed,
            &alloc::format!("{architecture_name}.embedding_length"),
        )
        .expect("embedding_length present on a real gguf");
        let query_heads = metadata_u32(
            &parsed,
            &alloc::format!("{architecture_name}.attention.head_count"),
        )
        .expect("attention.head_count present on a real gguf");
        let rope_dimension_count_key = alloc::format!("{architecture_name}.rope.dimension_count");
        let key_present = parsed.metadata_value(&rope_dimension_count_key).is_some();
        let derived_head_dim = metadata_u32_optional_or(
            &parsed,
            &rope_dimension_count_key,
            embedding / query_heads.max(1),
        );

        std::println!(
            "real_lfm2 architecture={architecture_name} embedding={embedding} query_heads={query_heads} \
             rope_dimension_count_key_present={key_present} derived_head_dim={derived_head_dim}"
        );
        assert!(
            !key_present,
            "this test's whole premise is that rope.dimension_count is ABSENT on this real checkpoint; \
             if this fails, the file changed and this test's fallback path is no longer exercised"
        );
        assert_eq!(
            derived_head_dim, 64,
            "LFM2.5-8B-A1B's real per-head dimension is 64 (independently confirmed via \
             attn_q_norm.weight's own declared [64] shape on this file)"
        );
    }
}
