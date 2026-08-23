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
pub fn gguf_tensor_as_f32(parsed: &ParsedGguf, file_bytes: &[u8], name: &str) -> Result<Vec<f32>, InteropError> {
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
/// One function over `Q4_K`/`Q5_K`/`Q6_K` rather than three near-identical
/// ones: the three differ only in which variant carries the byte range, and
/// every k-quant super-block is stored the same way -- a contiguous
/// row-major `[out, in]` byte run whose per-row period is a function of the
/// type alone. A per-type entry point would be that `match` arm rewritten as
/// a signature, three times over.
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
/// decoding through [`reinterpret_f32`]. This is sound only if `bytes.as_ptr()`
/// is 4-byte aligned: [`proxima_gguf::parser`] validates every tensor's
/// on-disk `offset` against a running total of `pad_to_alignment` sums (a
/// mismatch is a parse error, `GgufError::TensorOffsetMismatch`), so the
/// byte offset *within the file* is always a multiple of `parsed.alignment`
/// (minimum default 32) -- but that says nothing about whether `file_bytes`'s
/// own base pointer is aligned. A `Vec<u8>` from `std::fs::read` carries no
/// pointer-alignment guarantee beyond `align_of::<u8>() == 1`; an `mmap`
/// (page-aligned by the kernel) does. [`aligned_f32_view`] checks the
/// *actual* runtime pointer, not the assumption, and this function returns
/// [`InteropError::MisalignedFloat32Tensor`] rather than reinterpreting
/// unaligned bytes -- callers fall back to [`gguf_tensor_as_f32`]'s owned,
/// byte-at-a-time decode, which never assumes alignment.
///
/// # Errors
///
/// [`InteropError::UnknownTensor`] if `name` isn't in `parsed.tensors`;
/// [`InteropError::Gguf`] if the tensor's declared byte range doesn't fit
/// `file_bytes`; [`InteropError::MisalignedFloat32Tensor`] if `name`'s
/// tensor is `F32` but `file_bytes`'s base pointer leaves its byte range
/// unaligned for `&[f32]`; [`InteropError::UnrepresentableGgmlType`] if
/// `name`'s tensor is none of `F32`/`Q4_K`/`Q5_K`/`Q6_K` -- callers route
/// anything else through [`gguf_tensor_as_f32`] instead, which is what this
/// crate has a decoder for.
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
fn aligned_f32_view(bytes: &[u8]) -> Option<&[f32]> {
    let float_size = core::mem::size_of::<f32>();
    if !bytes.len().is_multiple_of(float_size) {
        return None;
    }
    if !(bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<f32>()) {
        return None;
    }
    // SAFETY: see this function's doc.
    Some(unsafe { core::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), bytes.len() / float_size) })
}

fn find_tensor<'a>(parsed: &'a ParsedGguf, name: &str) -> Result<&'a TensorInfo, InteropError> {
    parsed
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| InteropError::UnknownTensor { name: name.into() })
}

fn reinterpret_f32(data: &[u8]) -> Vec<f32> {
    data.as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

fn dequantize(
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelArchitecture {
    pub vocab: u32,
    pub embedding: u32,
    pub feed_forward: u32,
    pub query_heads: u32,
    pub kv_heads: u32,
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
}

/// Reads [`ModelArchitecture`] out of `parsed`'s own metadata: looks up
/// `general.architecture` first (`"llama"` for a Mistral-shaped checkpoint
/// such as openchat-3.5), then every `{architecture}.*` dimension key
/// under that name. `head_dim` reads `{architecture}.rope.dimension_count`
/// directly rather than deriving `embedding / query_heads` -- GGUF stores
/// the rotary dimension count as its own key, and for a full-rotation
/// architecture (every Mistral/Llama-family checkpoint this crate has
/// evaluated) that value already equals the per-head dimension, so reading
/// it directly avoids assuming the division holds for an architecture this
/// crate has not seen.
///
/// # Errors
///
/// [`InteropError::MissingMetadataKey`] naming the first key that is
/// absent, or present with the wrong [`MetadataValue`] variant;
/// [`InteropError::VocabShapeMismatch`] if `token_embd.weight`'s element
/// count does not divide evenly by `embedding_length`.
pub fn architecture_from_metadata(parsed: &ParsedGguf) -> Result<ModelArchitecture, InteropError> {
    let architecture = metadata_str(parsed, "general.architecture")?;
    let embedding = metadata_u32(parsed, &alloc::format!("{architecture}.embedding_length"))?;
    let feed_forward = metadata_u32(parsed, &alloc::format!("{architecture}.feed_forward_length"))?;
    let query_heads = metadata_u32(parsed, &alloc::format!("{architecture}.attention.head_count"))?;
    let kv_heads = metadata_u32(parsed, &alloc::format!("{architecture}.attention.head_count_kv"))?;
    let block_count = metadata_u32(parsed, &alloc::format!("{architecture}.block_count"))?;
    let head_dim = metadata_u32(parsed, &alloc::format!("{architecture}.rope.dimension_count"))?;
    let vocab = vocab_from_token_embedding(parsed, embedding)?;
    let expert_count = metadata_u32_optional(parsed, &alloc::format!("{architecture}.expert_count"));
    let expert_used_count = metadata_u32_optional(parsed, &alloc::format!("{architecture}.expert_used_count"));
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
    })
}

fn metadata_str<'parsed>(parsed: &'parsed ParsedGguf, key: &str) -> Result<&'parsed str, InteropError> {
    parsed
        .metadata_value(key)
        .and_then(MetadataValue::as_str)
        .ok_or_else(|| InteropError::MissingMetadataKey { key: key.into() })
}

fn metadata_u32(parsed: &ParsedGguf, key: &str) -> Result<u32, InteropError> {
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
fn metadata_u32_optional(parsed: &ParsedGguf, key: &str) -> u32 {
    parsed.metadata_value(key).and_then(MetadataValue::as_u32).unwrap_or(0)
}

fn vocab_from_token_embedding(parsed: &ParsedGguf, embedding: u32) -> Result<u32, InteropError> {
    let tensor = find_tensor(parsed, "token_embd.weight")?;
    let elements = tensor.element_count();
    let divisor = u64::from(embedding);
    if divisor == 0 || !elements.is_multiple_of(divisor) {
        return Err(InteropError::VocabShapeMismatch { elements, embedding });
    }
    Ok((elements / divisor) as u32)
}

/// Every weight [`proxima_tensor::spec::mistral_cached_forward_program`]
/// binds by name, split into owned `f32` buffers (norms, plus
/// `token_embd.weight`, which is an embedding lookup rather than a matmul
/// operand and so has no packed kernel) and zero-copy packed blocks
/// borrowed straight out of `file_bytes` -- see [`bind_dense`]/
/// [`bind_matmul_weight`]. `pub(crate)`: [`crate::generate::LoadedModel`]
/// is the one place outside this module that constructs or reads one.
#[cfg(feature = "std")]
pub(crate) struct BoundWeights<'file> {
    pub(crate) resident_bytes: usize,
    pub(crate) owned: Vec<(alloc::string::String, Vec<f32>)>,
    pub(crate) packed: Vec<(alloc::string::String, proxima_tensor::cpu::QuantizedBlock<'file>)>,
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
    match gguf_tensor_as_packed_block(parsed, file_bytes, &name) {
        Ok(block @ proxima_tensor::cpu::QuantizedBlock::Float32(borrowed)) => {
            state.resident_bytes += core::mem::size_of_val(borrowed);
            state.packed.push((name, block));
        }
        Ok(_) | Err(_) => {
            let decoded = gguf_tensor_as_f32(parsed, file_bytes, &name)?;
            state.resident_bytes += decoded.len() * core::mem::size_of::<f32>();
            state.owned.push((name, decoded));
        }
    }
    Ok(())
}

/// A 2-D projection weight the cached forward program uses as one
/// `Multiply`-then-`Add`-reduce (matmul) operand. Tries
/// [`gguf_tensor_as_packed_block`] first: a `Q4_K`/`Q5_K`/`Q6_K` tensor
/// binds packed, zero-copy, straight out of the mmap's bytes; only `F32`
/// falls back to dequantize-then-transpose (`transpose_out_in_to_in_out`),
/// since GGUF's on-disk row-major `[out, in]` layout needs an explicit
/// transpose to become the `[in, out]` layout the forward program's
/// access patterns expect, and a packed weight's own matmul kernel walks
/// the native `[out, in]` layout directly instead.
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
    match gguf_tensor_as_packed_block(parsed, file_bytes, &name) {
        Ok(block) => state.packed.push((name, block)),
        Err(_) => {
            let decoded = gguf_tensor_as_f32(parsed, file_bytes, &name)?;
            state.resident_bytes += decoded.len() * core::mem::size_of::<f32>();
            state.owned.push((name.clone(), transpose_out_in_to_in_out(&decoded, &name, out_dim, in_dim)?));
        }
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
        transposed[expert * per_expert..(expert + 1) * per_expert].copy_from_slice(&slab_transposed);
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
fn restack_error_as_interop_error(layer: u32, projection: &str, error: proxima_gguf::restack::RestackError) -> InteropError {
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
/// `Q4_K`/`Q5_K`/`Q6_K` too) is deliberately NOT bound packed here even
/// though it could be: [`bind_moe_expert_weights`]'s own doc names the
/// reason -- the packed matmul kernel that would consume it has no notion
/// of a gather at all, so a packed binding would silently feed the
/// interpreter a buffer whose per-token expert selection is dropped, not
/// rejected. Falls back to dequantize-then-[`transpose_expert_stack`] (not
/// [`transpose_out_in_to_in_out`]: the flat buffer is `[expert_count,
/// out_dim, in_dim]`, not one 2-D matrix, so a plain global transpose
/// would scramble the expert axis into the wrong place in memory).
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
        Ok(block @ proxima_tensor::cpu::QuantizedBlock::Float32(_)) => state.packed.push((name, block)),
        Ok(_) | Err(_) => {
            let decoded = gguf_tensor_as_f32(parsed, file_bytes, &name)?;
            let transposed = transpose_expert_stack(&decoded, &name, expert_count, out_dim, in_dim)?;
            state.resident_bytes += transposed.len() * core::mem::size_of::<f32>();
            state.owned.push((name, transposed));
        }
    }
    Ok(())
}

/// One MoE-only weight family (`ffn_gate`/`ffn_up`/`ffn_down`) for one
/// layer. Tries a single native stacked tensor first
/// (`blk.{layer}.{projection}_exps.weight`, see
/// [`bind_moe_stacked_experts`] -- zero-copy when `F32`, dequantized
/// otherwise) before falling back to
/// [`proxima_gguf::restack::discover_experts`]/`plan_stack`/`restack_into`
/// for the per-expert-tensor convention
/// (`blk.{layer}.{projection}.{expert}.weight`, `restack.rs`'s own module
/// doc verified against a real Mixtral-8x7B checkpoint) -- **always
/// dequantized** in that fallback, never packed, for the same reason
/// [`bind_moe_stacked_experts`]'s non-`F32` arm is: this function exists
/// specifically so a routed expert's weight slab can be gathered by
/// `IndexMap::Computed` (`proxima_tensor::spec::gathered_expert_product`,
/// `spec.rs:847`), and that gather is resolved *only* over a plain `&[f32]`
/// buffer (`proxima_tensor::cpu::operand_buffers`/`GatherCursor`,
/// `cpu.rs:2177`/`cpu.rs:1841-2654` -- the sole runtime consumer of a
/// `Computed` map). The packed matmul path
/// (`proxima_tensor::cpu::run_reduce_quantized`, `cpu.rs:3348-3660`, the
/// sole consumer of a non-`Float32` `QuantizedBlock`) never reads the
/// `Lookup` a `Computed` map produces at bind time
/// (`proxima_tensor::bind::build_operand`, `bind.rs:761-788`, returns
/// `(node, layout, Some(lookup))` for a `Computed` map, but
/// `run_reduce_quantized` only ever destructures `(_, layout, _)`) -- it
/// classifies every output axis purely from the operand's own `Layout`
/// stride (`cpu.rs:3436-3448`), which for a gathered axis is the *base*
/// pattern's stride (a constant/broadcast axis, `AxisIndex::default()` in
/// `gathered_expert_product`), not the per-token expert selection the
/// gather actually encodes. Binding a routed expert stack packed would
/// therefore not fail loudly -- it would silently run every token against
/// whichever expert the base pattern's constant offset picks, regardless
/// of that token's own `route`. This is a real `proxima-tensor` interpreter
/// gap (no packed-operand gather support), not something this crate's own
/// binder can safely route around: keeping the owned dequantized path here
/// is the correct, if memory-expensive, choice until that gap closes.
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
fn bind_moe_expert_weights<'file>(
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
        return bind_moe_stacked_experts(parsed, file_bytes, stacked_name, expert_count as usize, out_dim, in_dim, state);
    }

    let experts = discover_experts(&parsed.tensors, u64::from(layer), projection, u64::from(expert_count))
        .map_err(|error| restack_error_as_interop_error(layer, projection, error))?;
    let plan = plan_stack(&experts).map_err(|error| restack_error_as_interop_error(layer, projection, error))?;

    let mut sources: Vec<&[u8]> = Vec::with_capacity(experts.len());
    for tensor in &experts {
        let range = parsed.tensor_data_range(tensor, file_bytes.len() as u64)?;
        sources.push(&file_bytes[range.start as usize..range.end as usize]);
    }

    let mut stacked_bytes = vec![0u8; plan.total_bytes as usize];
    restack_into(&mut stacked_bytes, &plan, &sources).map_err(|error| restack_error_as_interop_error(layer, projection, error))?;

    let total_elements = out_dim * in_dim * expert_count as usize;
    let decoded = match plan.ggml_type {
        GgmlType::F32 => reinterpret_f32(&stacked_bytes),
        GgmlType::Q4_K => dequantize(&stacked_bytes, total_elements, q4_k::dequantize)?,
        GgmlType::Q5_K => dequantize(&stacked_bytes, total_elements, q5_k::dequantize)?,
        GgmlType::Q6_K => dequantize(&stacked_bytes, total_elements, q6_k::dequantize)?,
        GgmlType::Q8_0 => dequantize(&stacked_bytes, total_elements, q8_0::dequantize)?,
        other => {
            return Err(InteropError::UnrepresentableGgmlType {
                tensor: alloc::format!("blk.{layer}.{projection}.*.weight"),
                ggml_type: other,
            });
        }
    };
    let transposed = transpose_expert_stack(&decoded, &stacked_name, expert_count as usize, out_dim, in_dim)?;
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
    };

    let embedding = architecture.embedding as usize;
    let kv_dim = architecture.kv_heads as usize * architecture.head_dim as usize;
    let feed_forward = architecture.feed_forward as usize;
    let vocab = architecture.vocab as usize;

    bind_dense(parsed, file_bytes, "token_embd.weight".into(), &mut state)?;

    for layer in 0..architecture.block_count {
        bind_dense(parsed, file_bytes, alloc::format!("blk.{layer}.attn_norm.weight"), &mut state)?;
        bind_dense(parsed, file_bytes, alloc::format!("blk.{layer}.ffn_norm.weight"), &mut state)?;
        bind_matmul_weight(parsed, file_bytes, alloc::format!("blk.{layer}.attn_q.weight"), embedding, embedding, &mut state)?;
        bind_matmul_weight(parsed, file_bytes, alloc::format!("blk.{layer}.attn_k.weight"), kv_dim, embedding, &mut state)?;
        bind_matmul_weight(parsed, file_bytes, alloc::format!("blk.{layer}.attn_v.weight"), kv_dim, embedding, &mut state)?;
        bind_matmul_weight(parsed, file_bytes, alloc::format!("blk.{layer}.attn_output.weight"), embedding, embedding, &mut state)?;

        if architecture.expert_count == 0 {
            bind_matmul_weight(parsed, file_bytes, alloc::format!("blk.{layer}.ffn_gate.weight"), feed_forward, embedding, &mut state)?;
            bind_matmul_weight(parsed, file_bytes, alloc::format!("blk.{layer}.ffn_up.weight"), feed_forward, embedding, &mut state)?;
            bind_matmul_weight(parsed, file_bytes, alloc::format!("blk.{layer}.ffn_down.weight"), embedding, feed_forward, &mut state)?;
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
                bind_moe_expert_weights(parsed, file_bytes, layer, projection, expert_count, out_dim, in_dim, &mut state)?;
            }
        }
    }

    bind_dense(parsed, file_bytes, "output_norm.weight".into(), &mut state)?;
    bind_matmul_weight(parsed, file_bytes, "output.weight".into(), vocab, embedding, &mut state)?;
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

        let result = transpose_expert_stack(&one_expert_short, "blk.0.ffn_gate_exps.weight", expert_count, out_dim, in_dim);

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
        let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
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

        let decoded = gguf_tensor_as_f32(&parsed, &file_bytes, "weights").expect("bind f32 tensor by name");
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

        let decoded =
            gguf_tensor_as_f32(&parsed, &file_bytes, "blk.0.ffn_gate.weight").expect("bind q4_k tensor by name");
        assert_eq!(decoded.len(), q4_k::QK_K);
        // element 0: d*sc*q - dmin*m = 1.0*3.0*7.0 - 0.5*61.0 = -9.5
        assert!((decoded[0] - (-9.5)).abs() < 1e-6, "decoded[0]={}", decoded[0]);
        // every other element in sub_block 0 shares scale/min with q=0.
        assert!((decoded[1] - (-30.5)).abs() < 1e-6, "decoded[1]={}", decoded[1]);
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
        assert!(matches!(outcome, Err(InteropError::UnrepresentableGgmlType { .. })));
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
                ("general.architecture".to_string(), Value::String("llama".to_string())),
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
        let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parses gguf with architecture metadata");

        let architecture = architecture_from_metadata(&parsed).expect("derive architecture from real metadata keys");
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
            },
            "a checkpoint with no expert_count/expert_used_count key is dense: both fields must read as 0, \
             not error"
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
                ("general.architecture".to_string(), Value::String("llama".to_string())),
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
        let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parses gguf with moe metadata");

        let architecture = architecture_from_metadata(&parsed).expect("derive architecture from real metadata keys");
        assert_eq!(architecture.expert_count, 8, "expert_count must read the real metadata key, not default to 0");
        assert_eq!(
            architecture.expert_used_count, 2,
            "expert_used_count must read the real metadata key, not default to 0"
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
/// checkpoint's owned-allocation total must scale with the number of bytes
/// its packed codec actually needs, never with `experts * rows * cols * 4`
/// (a real Mixtral-8x7B's own figure -- 8 experts, 3 matrices, `4096x14336`,
/// 32 layers, `f32` -- is ~180 GB against this checkpoint's own ~25 GB
/// on-disk size). Two cases, both real GGUF (`write_complete`/
/// `parse_complete`, never a hand-built buffer), both using
/// [`q4_k::quantize`] to encode real (non-degenerate) weight data:
///
/// - [`moe_owned_allocation_matches_the_checkpoints_own_layout_convention`]'s
///   `native_stacked_f32` case is GREEN today: a native single-stacked
///   `_exps.weight` tensor binds through [`bind_moe_stacked_experts`],
///   which is safe to keep packed only for `F32` (this crate's own doc on
///   that function names why a quantized codec is not). This fixture is
///   `F32`, so the assertion is "zero new owned bytes, borrowed straight
///   out of the mmap" -- and it is real proof, not a tautology: reverting
///   to the pre-change `gguf_tensor_as_f32`-always path makes it fail,
///   because that path allocates and copies regardless of codec.
/// - [`moe_experts_do_not_yet_stay_packed_for_the_real_per_expert_tensor_convention`]
///   is `#[ignore]`d and RED by design: the real Mixtral convention
///   (`restack.rs`'s own module doc) stores `n_experts` independent
///   tensors, which `bind_moe_expert_weights` must dequantize into one
///   flat `f32` buffer for [`proxima_tensor`]'s `IndexMap::Computed` gather
///   to read (see [`bind_moe_expert_weights`]'s own doc for the exact
///   `proxima-tensor` incompatibility this hits). Nothing in this crate,
///   `bind.rs` included, can safely close this without a
///   `proxima-tensor` interpreter capability this change does not add --
///   this test exists so that capability landing is the thing that turns
///   it green, not a silent memory regression nobody is watching.
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
            .map(|index| ((index as u32).wrapping_mul(2_654_435_761).wrapping_add(seed) % 1000) as f32 / 1000.0 - 0.5)
            .collect()
    }

    fn quantize_q4_k(values: &[f32]) -> alloc::vec::Vec<u8> {
        let mut bytes = alloc::vec![0u8; values.len() / q4_k::QK_K * q4_k::BLOCK_BYTES];
        q4_k::quantize(values, &mut bytes).expect("real q4_k encoder quantizes this fixture's own weight data");
        bytes
    }

    /// A native single stacked `blk.0.ffn_gate_exps.weight` tensor -- every
    /// expert's own `[OUT_DIM, IN_DIM]` slab back to back, `F32` so
    /// [`bind_moe_stacked_experts`]'s safe packed arm actually fires (see
    /// that function's own doc for why only `F32` is safe here).
    fn checkpoint_with_native_stacked_f32_experts() -> alloc::vec::Vec<u8> {
        let mut values: alloc::vec::Vec<f32> = alloc::vec::Vec::with_capacity(EXPERT_COUNT as usize * OUT_DIM * IN_DIM);
        for expert in 0..EXPERT_COUNT {
            values.extend(expert_values(expert));
        }
        let bytes: alloc::vec::Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
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
        write_complete(&model).expect("writes a well-formed synthetic native-stacked moe checkpoint")
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
        write_complete(&model).expect("writes a well-formed synthetic per-expert-tensor moe checkpoint")
    }

    fn empty_state(file_bytes: &[u8]) -> BoundWeights<'_> {
        BoundWeights {
            resident_bytes: file_bytes.len(),
            owned: Vec::new(),
            packed: Vec::new(),
        }
    }

    fn owned_bytes_total(state: &BoundWeights) -> usize {
        state.owned.iter().map(|(_, data)| data.len() * core::mem::size_of::<f32>()).sum()
    }

    /// A real Mixtral-8x7B's own shape, for the assertion's own numbers to
    /// mean something rather than an arbitrary threshold: `experts * out *
    /// in * 4` is the dequantized-owned ceiling this test's per-expert-tensor
    /// case (deliberately) still hits, and the packed floor
    /// (`experts * (out*in/QK_K) * BLOCK_BYTES`) it never reaches without a
    /// `proxima-tensor` capability this crate does not add.
    fn dequantized_owned_ceiling_bytes() -> usize {
        EXPERT_COUNT as usize * OUT_DIM * IN_DIM * core::mem::size_of::<f32>()
    }

    fn packed_floor_bytes() -> usize {
        EXPERT_COUNT as usize * (OUT_DIM * IN_DIM / q4_k::QK_K) * q4_k::BLOCK_BYTES
    }

    /// Two real GGUF layout conventions for the same weight family, one
    /// shared assertion: does binding this checkpoint's experts allocate
    /// owned `f32` bytes proportional to the *packed* size, or to
    /// `experts * rows * cols * 4`? `native_stacked_f32` is GREEN today
    /// (zero owned bytes -- [`bind_moe_stacked_experts`]'s packed arm
    /// borrows straight out of `file_bytes`, the same zero-copy contract
    /// [`bind_matmul_weight`] already gives a dense weight); reverting to
    /// the pre-change `gguf_tensor_as_f32`-always path makes this case fail,
    /// because that path allocates and copies regardless of codec.
    /// `per_expert_q4_k` is the real Mixtral-8x7B on-disk convention
    /// (`restack.rs`'s own module doc) and hits exactly the dequantized
    /// ceiling -- documented here as a passing, concrete measurement of the
    /// gap's own size (see [`moe_experts_do_not_yet_stay_packed_for_the_real_per_expert_tensor_convention`]
    /// for the same gap asserted the other way, `#[ignore]`d).
    #[proxima::test]
    #[case::native_stacked_f32(true)]
    #[case::per_expert_q4_k(false)]
    async fn moe_owned_allocation_matches_the_checkpoints_own_layout_convention(#[case] native_stacked: bool) {
        let file_bytes = if native_stacked {
            checkpoint_with_native_stacked_f32_experts()
        } else {
            checkpoint_with_per_expert_q4_k_tensors()
        };
        let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parses synthetic moe checkpoint");
        let mut state = empty_state(&file_bytes);

        bind_moe_expert_weights(&parsed, &file_bytes, 0, "ffn_gate", EXPERT_COUNT, OUT_DIM, IN_DIM, &mut state)
            .expect("binds this case's own expert tensor layout");

        if native_stacked {
            assert_eq!(state.packed.len(), 1, "one packed entry for the whole native stack, no per-expert split");
            assert_eq!(owned_bytes_total(&state), 0, "an f32 native stack must borrow zero-copy, allocating no owned f32 bytes");
        } else {
            assert!(state.packed.is_empty(), "the per-expert-tensor path never binds packed -- see bind_moe_expert_weights's own doc");
            assert_eq!(
                owned_bytes_total(&state),
                dequantized_owned_ceiling_bytes(),
                "today's owned allocation is exactly experts * rows * cols * 4, the shape this change cannot yet avoid"
            );
        }
    }

    /// RED by design (`#[ignore]`d, matching this crate's own convention for
    /// a real, named, not-yet-closed capability gap -- `capability_matrix.rs`'s
    /// own module doc: "a cell that cannot pass yet is still written and
    /// `#[ignore]`d with the exact missing piece named"): the real Mixtral
    /// per-expert-tensor convention still dequantizes every expert to owned
    /// `f32`, so the owned total hits the dequantized ceiling, not the
    /// packed floor. Un-ignoring this and seeing it pass is the acceptance
    /// criterion for whatever lands the `proxima-tensor` gather-over-packed
    /// -blocks capability [`bind_moe_expert_weights`]'s own doc names.
    #[proxima::test]
    #[ignore = "blocked on a proxima-tensor interpreter capability (gather over a packed QuantizedBlock) this change does not add -- see bind_moe_expert_weights's own doc"]
    async fn moe_experts_do_not_yet_stay_packed_for_the_real_per_expert_tensor_convention() {
        let file_bytes = checkpoint_with_per_expert_q4_k_tensors();
        let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parses synthetic per-expert-tensor moe checkpoint");
        let mut state = empty_state(&file_bytes);

        bind_moe_expert_weights(&parsed, &file_bytes, 0, "ffn_gate", EXPERT_COUNT, OUT_DIM, IN_DIM, &mut state)
            .expect("binds the per-expert-tensor q4_k experts");

        let owned = owned_bytes_total(&state);
        std::println!(
            "moe_memory_shape owned_bytes={owned} packed_floor_bytes={} dequantized_ceiling_bytes={}",
            packed_floor_bytes(),
            dequantized_owned_ceiling_bytes()
        );
        assert!(
            owned <= packed_floor_bytes() * 2,
            "owned allocation ({owned} bytes) must be proportional to the packed size ({} bytes), not to \
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
    use proxima_tensor::cpu::{QuantizedBlock, evaluate_named, evaluate_quantized_named_with_scratch};
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
            let len = usize::try_from(file.metadata()?.len()).expect("fixture file length fits in usize");
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
            Ok(Self { base, len, _file: file })
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
            Poll::Pending => unreachable!("proxima-model-interop pipes never yield: no internal .await"),
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
        let parsed = proxima_gguf::pipe::parse_complete(file_bytes).expect("parse host-local openchat gguf fixture");
        prefault_if_requested(file_bytes);

        let model = LoadedModel::load(&parsed, file_bytes).expect("load real openchat checkpoint through the public path");
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
        assert_eq!(generated.0[0], 2651, "greedy token id drifted off llama.cpp's captured answer");
        assert_eq!(generated.1, "known", "greedy token text drifted off llama.cpp's captured answer");
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
        let parsed = proxima_gguf::pipe::parse_complete(file_bytes).expect("parse host-local openchat gguf fixture");
        prefault_if_requested(file_bytes);

        let model = LoadedModel::load(&parsed, file_bytes).expect("load real openchat checkpoint through the public path");
        let prompt = decode_loop_prompt();
        let max_tokens = decode_loop_max_tokens();

        std::println!("prompt={prompt:?} max_tokens={max_tokens}");
        let decode_start = std::time::Instant::now();
        let generated = block_on(model.call((prompt.clone(), max_tokens)))
            .expect("generate through the public Pipe path");
        let total_elapsed = decode_start.elapsed();
        let tokens_generated = generated.0.len();
        let mean_ms_per_token = total_elapsed.as_secs_f64() * 1000.0 / tokens_generated.max(1) as f64;

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
                "quant_arm q4k_macs={} q4k_ms={:.3} q4k_ns_per_mac={:.5} q5k_macs={} q6k_macs={} q5k_f32_calls={} q6k_f32_calls={} reduce_quantized_calls={}",
                quant.q4k_macs,
                q4k_ns as f64 / 1e6,
                if quant.q4k_macs == 0 { 0.0 } else { q4k_ns as f64 / quant.q4k_macs as f64 },
                quant.q5k_macs,
                quant.q6k_macs,
                quant.q5k_f32_calls,
                quant.q6k_f32_calls,
                quant.reduce_quantized_calls
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
                    proxima_tensor::instrument::ticks_to_nanos(quant.staged_round_ticks) as f64 / quant.staged_macs as f64
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
                let first_claim_ns = cohort_diag::SLOT_FIRST_CLAIM_NANOS[slot].load(Ordering::Relaxed);
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
        assert!(tokens_generated <= max_tokens, "decode loop must never exceed the requested budget");
        if generated.2 {
            assert!(
                tokens_generated < max_tokens,
                "an eos stop must have produced strictly fewer ids than the full budget"
            );
        } else {
            assert_eq!(tokens_generated, max_tokens, "budget exhaustion must produce exactly one id per step");
        }
        assert!(!generated.1.is_empty(), "degenerate control: decode loop produced no text");
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
        let parsed = proxima_gguf::pipe::parse_complete(file_bytes).expect("parse host-local openchat gguf fixture");
        prefault_if_requested(file_bytes);

        let model = LoadedModel::load(&parsed, file_bytes).expect("load real openchat checkpoint through the public path");
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
        assert!(!generated.1.is_empty(), "degenerate control: metal decode loop produced no text");
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
        let parsed = proxima_gguf::pipe::parse_complete(file_bytes).expect("parse host-local openchat gguf fixture");
        prefault_if_requested(file_bytes);

        let model = LoadedModel::load(&parsed, file_bytes).expect("load real openchat checkpoint through the public path");
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
        assert!(!generated.1.is_empty(), "degenerate control: metal decode loop produced no text");
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
        let parsed = proxima_gguf::pipe::parse_complete(file_bytes).expect("parse host-local openchat gguf fixture");
        prefault_if_requested(file_bytes);

        let model = LoadedModel::load(&parsed, file_bytes).expect("load real openchat checkpoint through the public path");
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
        assert!(!generated.1.is_empty(), "degenerate control: metal decode loop produced no text");
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

    fn build_cached_position_inputs(new_ids: &[u32], start_position: usize, head_dim: u32) -> CachedPositionInputs {
        let new_count = new_ids.len();
        let pairs = head_dim as usize / 2;
        let ids_f32: Vec<f32> = new_ids.iter().map(|&id| id as f32).collect();
        let epsilon = alloc::vec![RMS_EPSILON; new_count];

        let mut cos = alloc::vec![0.0f32; new_count * pairs];
        let mut sin = alloc::vec![0.0f32; new_count * pairs];
        for offset in 0..new_count {
            let position = (start_position + offset) as f32;
            for pair in 0..pairs {
                let theta = position * ROPE_FREQ_BASE.powf(-((2 * pair) as f32) / (head_dim as f32));
                cos[offset * pairs + pair] = theta.cos();
                sin[offset * pairs + pair] = theta.sin();
            }
        }

        CachedPositionInputs { ids_f32, epsilon, cos, sin }
    }

    const ROPE_FREQ_BASE: f32 = 10_000.0;
    const RMS_EPSILON: f32 = 1e-5;

    /// Every layer's growable key/value cache: `k_even`/`k_odd` are already
    /// RoPE-rotated, `v` is the un-rotated projected value. Two storage
    /// strategies: `Float32` keeps every position's `f32` values; `Q8_0`
    /// holds packed bytes in the same codec
    /// [`super::gguf_tensor_as_packed_block`] already reads GGUF weight
    /// tensors through -- this seam test evaluates only one step
    /// (`cached_len == 0`), so nothing ever appends and `Q8_0` stays empty.
    enum LayerCache {
        Float32 { k_even: Vec<f32>, k_odd: Vec<f32>, v: Vec<f32> },
        Q8_0 { k_even: Vec<u8>, k_odd: Vec<u8>, v: Vec<u8> },
    }

    impl LayerCache {
        fn new(precision: GgmlType) -> Self {
            match precision {
                GgmlType::Q8_0 => LayerCache::Q8_0 { k_even: Vec::new(), k_odd: Vec::new(), v: Vec::new() },
                _ => LayerCache::Float32 { k_even: Vec::new(), k_odd: Vec::new(), v: Vec::new() },
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
            let parsed = proxima_gguf::pipe::parse_complete(file_bytes).expect("parse host-local openchat gguf fixture");
            let architecture = architecture_from_metadata(&parsed).expect("derive architecture from real metadata");
            let weights = bind_all_weights(&parsed, file_bytes, &architecture).expect("bind real openchat checkpoint weights");

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

            let kv_cache_names: Vec<(alloc::string::String, alloc::string::String, alloc::string::String)> = (0..architecture
                .block_count as usize)
                .map(|layer| {
                    (
                        alloc::format!("kv_cache.{layer}.k_even"),
                        alloc::format!("kv_cache.{layer}.k_odd"),
                        alloc::format!("kv_cache.{layer}.v"),
                    )
                })
                .collect();
            let layer_caches: Vec<LayerCache> =
                (0..architecture.block_count as usize).map(|_| LayerCache::new(GgmlType::Q8_0)).collect();

            let prompt = default_prompt();
            let ids = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)
                .and_then(|vocab| proxima_tokenizer::encode_with_bos_eos(&prompt, &vocab, true, false))
                .expect("build vocab and encode prompt");

            let inputs = build_cached_position_inputs(&ids, 0, architecture.head_dim);
            let mut named_blocks: Vec<(&str, QuantizedBlock)> =
                Vec::with_capacity(weights.owned.len() + weights.packed.len() + 3 + architecture.block_count as usize * 3);
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
                named_blocks.extend(layer_caches[layer].named_blocks(k_even_name, k_odd_name, v_name));
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
            .expect("evaluate_quantized_named_with_scratch binds the q8_0 cache seam probe by name");
        }));

        let panic_payload = outcome.expect_err(
            "expected the cached attention reduce's shared kv_heads axis to be rejected as a real \
             matmul-capability gap -- see this test's own doc",
        );
        let message = panic_payload
            .downcast_ref::<alloc::string::String>()
            .cloned()
            .or_else(|| panic_payload.downcast_ref::<&str>().map(|value| alloc::string::String::from(*value)))
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

        let (parsed, file_bytes) = proxima_gguf::edge::read_file(path).expect("read host-local openchat gguf fixture");

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
            .expect("openchat checkpoint has at least one q4_k ffn_gate tensor with a full super-block");

        let decoded = gguf_tensor_as_f32(&parsed, &file_bytes, &tensor_name).expect("bind real q4_k tensor by name");
        let weight_row: Vec<f32> = decoded[..q4_k::QK_K].to_vec();

        let activation: Vec<f32> = (0..q4_k::QK_K).map(|index| 0.01 * (index as f32) - 1.28).collect();

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
                operands: alloc::vec![(weight_node, identity_map.clone()), (activation_node, identity_map)],
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
        let named: [(&str, &[f32]); 2] = [("weight", weight_row.as_slice()), ("activation", activation.as_slice())];
        let evaluated = evaluate_named(&program, &symbols, &named, &[dot]).expect("evaluate_named binds by name");
        let (interpreter_output, _shape) = evaluated.get(dot).expect("dot product node present in output");

        // independent computation: raw bytes -> dequantize_block -> manual
        // dot product, never touching `bind::gguf_tensor_as_f32` or the
        // interpreter.
        let tensor = parsed.tensors.iter().find(|tensor| tensor.name == tensor_name).expect("tensor still present");
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
        assert!(max_diff < 1e-3, "interpreter and independent dequantize-then-multiply diverged: max_diff={max_diff}");
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

    const FIXTURE_PATH: &str =
        "/Users/brianbruggeman/.lmstudio/models/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF/Nous-Hermes-2-Mixtral-8x7B-DPO.Q4_K_S.gguf";

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
                let read = $file.read(&mut $header_buf).expect("read gguf header region");
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

        let architecture = architecture_from_metadata(&parsed).expect("derive architecture from the real mixtral checkpoint");
        std::println!("real_mixtral architecture={architecture:?}");
        assert_eq!(architecture.expert_count, 8, "Mixtral-8x7B carries 8 experts per layer");
        assert_eq!(architecture.expert_used_count, 2, "Mixtral-8x7B routes top-2");
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

        let mut counts: alloc::collections::BTreeMap<alloc::string::String, usize> = alloc::collections::BTreeMap::new();
        for tensor in &parsed.tensors {
            *counts.entry(alloc::format!("{:?}", tensor.ggml_type)).or_insert(0) += 1;
        }
        std::println!("real_mixtral tensor_count={}", parsed.tensors.len());
        for (ggml_type, count) in &counts {
            std::println!("real_mixtral codec={ggml_type} tensor_count={count}");
        }

        let supported = ["F32", "Q4_K", "Q5_K", "Q6_K", "Q8_0"];
        let unsupported: Vec<(&alloc::string::String, &usize)> =
            counts.iter().filter(|(ggml_type, _)| !supported.contains(&ggml_type.as_str())).collect();
        if unsupported.is_empty() {
            std::println!("real_mixtral every real tensor's codec is one this crate already decodes");
        } else {
            for (ggml_type, count) in &unsupported {
                std::println!("real_mixtral UNSUPPORTED codec={ggml_type} tensor_count={count}");
            }
            let sample_names: Vec<&str> = parsed
                .tensors
                .iter()
                .filter(|tensor| !supported.contains(&alloc::format!("{:?}", tensor.ggml_type).as_str()))
                .take(6)
                .map(|tensor| tensor.name.as_str())
                .collect();
            std::println!("real_mixtral unsupported_codec_sample_names={sample_names:?}");
        }

        match parsed.metadata_value("tokenizer.chat_template") {
            Some(MetadataValue::String(template)) => {
                std::println!("real_mixtral chat_template_len={} chat_template={template:?}", template.len());
            }
            Some(other) => std::println!("real_mixtral tokenizer.chat_template present but not a string: {other:?}"),
            None => std::println!("real_mixtral no tokenizer.chat_template key in this checkpoint's metadata"),
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

        let architecture = architecture_from_metadata(&parsed).expect("derive architecture from the real mixtral checkpoint");
        let layer = 0u64;
        let projection = "ffn_gate";

        let experts = discover_experts(&parsed.tensors, layer, projection, u64::from(architecture.expert_count))
            .expect("discovers all 8 real experts for layer 0's ffn_gate projection");

        let mut sources_owned: Vec<Vec<u8>> = Vec::with_capacity(experts.len());
        for expert in &experts {
            let range = parsed.tensor_data_range(expert, file_len).expect("expert tensor range within file");
            let mut bytes = alloc::vec![0u8; (range.end - range.start) as usize];
            file.seek(SeekFrom::Start(range.start)).expect("seek to expert tensor data");
            file.read_exact(&mut bytes).expect("read expert tensor bytes");
            sources_owned.push(bytes);
        }
        let sources: Vec<&[u8]> = sources_owned.iter().map(Vec::as_slice).collect();

        let plan = plan_stack(&experts).expect("plans stack for real experts");
        let mut stacked_bytes = alloc::vec![0u8; plan.total_bytes as usize];
        restack_into(&mut stacked_bytes, &plan, &sources).expect("restacks real experts into destination buffer");

        let out_dim = architecture.feed_forward as usize;
        let in_dim = architecture.embedding as usize;
        let per_expert_elements = out_dim * in_dim;
        let total_elements = per_expert_elements * experts.len();
        let decoded = dequantize(&stacked_bytes, total_elements, q4_k::dequantize).expect("dequantizes the real restacked experts");
        let bound = transpose_expert_stack(&decoded, "test_moe_stack", experts.len(), out_dim, in_dim)
            .expect("expert_count/out_dim/in_dim agree with the real restacked byte length");

        for (expert_index, expert_bytes) in sources.iter().enumerate() {
            let mut expected = alloc::vec![0.0f32; per_expert_elements];
            q4_k::dequantize(expert_bytes, &mut expected).expect("independently dequantizes expert's own real bytes");

            let bound_slab = &bound[expert_index * per_expert_elements..(expert_index + 1) * per_expert_elements];
            let un_transposed = transpose_out_in_to_in_out(bound_slab, "test_expert_slab", in_dim, out_dim)
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
            let len = usize::try_from(file.metadata()?.len()).expect("fixture file length fits in usize");
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
            Ok(Self { base, len, _file: file })
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
            core::task::Poll::Pending => unreachable!("proxima-model-interop pipes never yield: no internal .await"),
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
    /// **This is not expected to reach a decode step, for two independent
    /// reasons, and this test observes whichever one the current code hits
    /// first.** `LoadedModel::load` always builds
    /// `proxima_tensor::spec::mistral_cached_forward_program`
    /// (`generate.rs:344`), which has no `expert_count`/`expert_used_count`
    /// parameter at all and unconditionally binds a dense
    /// `blk.{layer}.ffn_gate.weight`/`ffn_up.weight`/`ffn_down.weight` triple
    /// per layer (`spec.rs:1610-1627`) -- the routed alternative
    /// (`append_mistral_moe_layer`/`append_moe_ffn`, `spec.rs:983`/`893`) is
    /// only ever reachable through the separate, uncached
    /// `mistral_forward_program` (`spec.rs:1115`), which nothing in this
    /// crate calls. That mismatch would surface as
    /// `TensorError::UnboundInputName("blk.0.ffn_gate.weight")` out of the
    /// forward `call` itself, once weight loading gets that far.
    ///
    /// It does not get that far: this real checkpoint stores every layer's
    /// `ffn_gate_inp.weight` (the MoE router) as `F16` (confirmed above --
    /// all 32 `F16` tensors in this file are exactly the 32 layers' own
    /// `ffn_gate_inp.weight`), and neither `gguf_tensor_as_packed_block` nor
    /// `gguf_tensor_as_f32` decodes `F16`, so `bind_matmul_weight` (`bind.rs`)
    /// returns [`InteropError::UnrepresentableGgmlType`] for
    /// `blk.0.ffn_gate_inp.weight` on the very first layer, and
    /// `LoadedModel::load` propagates that cleanly as `Err` (`generate.rs`'s
    /// own `bind_all_weights(..)?`) -- before `bind_all_weights` ever reaches
    /// the expert-weight loop this module's other tests already prove
    /// correct in isolation. Still wrapped in `catch_unwind`: this observes
    /// whatever the current code does, and a stale assumption about which of
    /// the two gaps fires first should not abort the test binary.
    #[test]
    #[ignore = "depends on a 25 GB host-local mixtral gguf checkout outside this repo"]
    fn attempts_a_real_mixtral_forward_pass_and_reports_the_outcome() {
        let path = std::path::Path::new(FIXTURE_PATH);
        if !path.exists() {
            eprintln!("skipping: no host-local mixtral gguf fixture at {FIXTURE_PATH}");
            return;
        }

        let prompt =
            "<|im_start|>user\nWrite one sentence about the ocean.<|im_end|>\n<|im_start|>assistant\n";

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
                    .or_else(|| panic_payload.downcast_ref::<&str>().map(|value| alloc::string::String::from(*value)))
                    .unwrap_or_default();
                std::println!("real_mixtral OUTCOME=panic message={message}");
            }
        }
    }
}
