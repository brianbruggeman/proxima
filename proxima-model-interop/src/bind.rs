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
    data.chunks_exact(4)
        .map(|chunk| {
            let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
            f32::from_le_bytes(bytes)
        })
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
    Ok(ModelArchitecture {
        vocab,
        embedding,
        feed_forward,
        query_heads,
        kv_heads,
        head_dim,
        block_count,
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
#[cfg(feature = "std")]
pub(crate) fn bind_dense<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    name: alloc::string::String,
    state: &mut BoundWeights<'file>,
) {
    match gguf_tensor_as_packed_block(parsed, file_bytes, &name) {
        Ok(block @ proxima_tensor::cpu::QuantizedBlock::Float32(borrowed)) => {
            state.resident_bytes += core::mem::size_of_val(borrowed);
            state.packed.push((name, block));
        }
        Ok(_) | Err(_) => {
            let decoded = gguf_tensor_as_f32(parsed, file_bytes, &name)
                .unwrap_or_else(|error| panic!("bind real tensor {name} by name: {error}"));
            state.resident_bytes += decoded.len() * core::mem::size_of::<f32>();
            state.owned.push((name, decoded));
        }
    }
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
#[cfg(feature = "std")]
pub(crate) fn bind_matmul_weight<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    name: alloc::string::String,
    out_dim: usize,
    in_dim: usize,
    state: &mut BoundWeights<'file>,
) {
    match gguf_tensor_as_packed_block(parsed, file_bytes, &name) {
        Ok(block) => state.packed.push((name, block)),
        Err(_) => {
            let decoded = gguf_tensor_as_f32(parsed, file_bytes, &name)
                .unwrap_or_else(|error| panic!("bind real tensor {name} by name: {error}"));
            state.resident_bytes += decoded.len() * core::mem::size_of::<f32>();
            state.owned.push((name, transpose_out_in_to_in_out(&decoded, out_dim, in_dim)));
        }
    }
}

/// Runs [`bind_dense`]/[`bind_matmul_weight`] over every one of
/// `architecture`'s `block_count` layers plus `token_embd.weight` and
/// `output.weight` -- the load loop [`crate::generate::LoadedModel::load`]
/// runs once per checkpoint, so every [`Pipe::call`](proxima_primitives::pipe::Pipe::call)
/// after that reuses the result instead of re-walking the tensor
/// directory per request.
#[cfg(feature = "std")]
pub(crate) fn bind_all_weights<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    architecture: &ModelArchitecture,
) -> BoundWeights<'file> {
    let mut state = BoundWeights {
        resident_bytes: file_bytes.len(),
        owned: Vec::new(),
        packed: Vec::new(),
    };

    let embedding = architecture.embedding as usize;
    let kv_dim = architecture.kv_heads as usize * architecture.head_dim as usize;
    let feed_forward = architecture.feed_forward as usize;
    let vocab = architecture.vocab as usize;

    bind_dense(parsed, file_bytes, "token_embd.weight".into(), &mut state);

    for layer in 0..architecture.block_count {
        bind_dense(parsed, file_bytes, alloc::format!("blk.{layer}.attn_norm.weight"), &mut state);
        bind_dense(parsed, file_bytes, alloc::format!("blk.{layer}.ffn_norm.weight"), &mut state);
        bind_matmul_weight(parsed, file_bytes, alloc::format!("blk.{layer}.attn_q.weight"), embedding, embedding, &mut state);
        bind_matmul_weight(parsed, file_bytes, alloc::format!("blk.{layer}.attn_k.weight"), kv_dim, embedding, &mut state);
        bind_matmul_weight(parsed, file_bytes, alloc::format!("blk.{layer}.attn_v.weight"), kv_dim, embedding, &mut state);
        bind_matmul_weight(parsed, file_bytes, alloc::format!("blk.{layer}.attn_output.weight"), embedding, embedding, &mut state);
        bind_matmul_weight(parsed, file_bytes, alloc::format!("blk.{layer}.ffn_gate.weight"), feed_forward, embedding, &mut state);
        bind_matmul_weight(parsed, file_bytes, alloc::format!("blk.{layer}.ffn_up.weight"), feed_forward, embedding, &mut state);
        bind_matmul_weight(parsed, file_bytes, alloc::format!("blk.{layer}.ffn_down.weight"), embedding, feed_forward, &mut state);
    }

    bind_dense(parsed, file_bytes, "output_norm.weight".into(), &mut state);
    bind_matmul_weight(parsed, file_bytes, "output.weight".into(), vocab, embedding, &mut state);
    state
}

/// Row-major transpose from GGUF's native flat layout (`[out, in]`, `out`
/// rows of contiguous `in` values, ggml's linear-weight layout) to the
/// forward program's expected `[in, out]` layout.
#[cfg(feature = "std")]
pub(crate) fn transpose_out_in_to_in_out(flat: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    assert_eq!(flat.len(), out_dim * in_dim, "flat buffer length must match out_dim * in_dim");
    let mut transposed = alloc::vec![0.0f32; flat.len()];
    for out_index in 0..out_dim {
        for in_index in 0..in_dim {
            transposed[in_index * out_dim + out_index] = flat[out_index * in_dim + in_index];
        }
    }
    transposed
}

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
            }
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
            let weights = bind_all_weights(&parsed, file_bytes, &architecture);

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
