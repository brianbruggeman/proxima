//! The bidirectional GGUF <-> safetensors transform, over the one thing
//! both formats actually agree on: a named tensor is `(name, dtype, shape,
//! bytes)`. Everything either format carries beyond that is where the two
//! directions stop being symmetric — see each function's doc for exactly
//! what survives and what doesn't.
//!
//! Sans-IO, like both crates underneath it: this module never opens a
//! file. `gguf_to_safetensors` takes a [`ParsedGguf`] plus the byte buffer
//! it was parsed from (the same pair [`proxima_gguf::edge::read_file`]
//! hands back); `safetensors_to_gguf` takes a [`SafetensorsModel`] a caller
//! already built (e.g. via `proxima_safetensors::parse_complete` plus
//! slicing tensor bytes out of its own buffer).

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use arrayvec::ArrayVec;
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::sized::MAX_SUPPORTED_VERSION;
use proxima_gguf::tensor::MAX_DIMS;
use proxima_gguf::value::MetadataValue;
use proxima_gguf::{GgufModel, TensorPayload as GgufTensorPayload};
use proxima_safetensors::{SafetensorsModel, TensorPayload as SafetensorsTensorPayload};

use crate::dtype::{dtype_to_ggml, ggml_to_dtype};
use crate::error::InteropError;

/// GGUF -> safetensors. Every tensor with a mapped `GgmlType` (see
/// [`crate::dtype::ggml_to_dtype`]) carries its name, dtype, shape, and
/// exact bytes across losslessly; a tensor with a block-quantized type
/// makes the whole call fail with [`InteropError::UnrepresentableGgmlType`]
/// rather than silently dropping or mis-typing it.
///
/// GGUF's typed KV metadata (arbitrary architecture/hyperparameter/
/// tokenizer entries, some of them numeric or array-valued) has no home in
/// safetensors' flat `__metadata__: {string: string}` map. Rather than
/// drop it, every entry is carried into `__metadata__` as text: a
/// `MetadataValue::String` passes through verbatim; every other variant
/// (numbers, bools, arrays) is `Debug`-formatted. The VALUE survives and
/// stays inspectable; the ORIGINAL TYPE does not — `safetensors_to_gguf`
/// reading that string back gets a `MetadataValue::String`, not the
/// original `U32`/`Bool`/`Array`. This is the one place this transform is
/// documented-lossy rather than exact; see `safetensors_to_gguf`'s doc for
/// why the reverse direction doesn't have the same problem.
///
/// # Errors
///
/// [`InteropError::UnrepresentableGgmlType`] for a block-quantized tensor;
/// [`InteropError::Gguf`] if a tensor's declared byte range doesn't fit in
/// `file_bytes` (a malformed `ParsedGguf`/`file_bytes` pairing).
pub fn gguf_to_safetensors<'a>(
    parsed: &ParsedGguf,
    file_bytes: &'a [u8],
) -> Result<SafetensorsModel<'a>, InteropError> {
    let mut tensors = Vec::with_capacity(parsed.tensors.len());
    for tensor in &parsed.tensors {
        let dtype = ggml_to_dtype(tensor.ggml_type).ok_or_else(|| {
            InteropError::UnrepresentableGgmlType {
                tensor: tensor.name.clone(),
                ggml_type: tensor.ggml_type,
            }
        })?;
        let range = parsed.tensor_data_range(tensor, file_bytes.len() as u64)?;
        let data = &file_bytes[range.start as usize..range.end as usize];
        tensors.push(SafetensorsTensorPayload {
            name: tensor.name.clone(),
            dtype,
            shape: tensor.dims.iter().copied().collect(),
            data,
        });
    }

    let mut metadata = BTreeMap::new();
    for (key, value) in &parsed.metadata {
        metadata.insert(key.clone(), stringify_metadata_value(value));
    }

    Ok(SafetensorsModel { tensors, metadata })
}

/// safetensors -> GGUF. Every tensor with a mapped `DType` (see
/// [`crate::dtype::dtype_to_ggml`]) carries its name, dtype, shape, and
/// exact bytes across losslessly.
///
/// Metadata is the one direction this transform is NOT lossy in: every
/// `__metadata__` entry is already a flat string, and a GGUF
/// `MetadataValue::String` is exactly that — no type information is
/// discarded, because safetensors never had any beyond "it's a string" to
/// begin with. The output carries `general.architecture`-shaped hand-offs
/// exactly as written; it just can't invent typed fields safetensors never
/// had. `version` is fixed at [`MAX_SUPPORTED_VERSION`] since safetensors
/// has no version concept of its own to carry over.
///
/// # Errors
///
/// [`InteropError::UnrepresentableDType`] for a dtype ggml has no wire type
/// for (`Bool`, any unsigned integer, `Int128`/`UInt128`);
/// [`InteropError::TooManyDimensions`] if a tensor's shape has more than
/// [`MAX_DIMS`] dimensions.
pub fn safetensors_to_gguf<'a>(
    model: &SafetensorsModel<'a>,
) -> Result<GgufModel<'a>, InteropError> {
    let mut tensors = Vec::with_capacity(model.tensors.len());
    for tensor in &model.tensors {
        let ggml_type =
            dtype_to_ggml(tensor.dtype).ok_or_else(|| InteropError::UnrepresentableDType {
                tensor: tensor.name.clone(),
                dtype: tensor.dtype,
            })?;

        let mut dims: ArrayVec<u64, MAX_DIMS> = ArrayVec::new();
        for dim in &tensor.shape {
            dims.try_push(*dim)
                .map_err(|_| InteropError::TooManyDimensions {
                    tensor: tensor.name.clone(),
                    found: tensor.shape.len(),
                    max: MAX_DIMS,
                })?;
        }

        tensors.push(GgufTensorPayload {
            name: tensor.name.clone(),
            dims,
            ggml_type,
            data: tensor.data,
        });
    }

    let metadata = model
        .metadata
        .iter()
        .map(|(key, value)| (key.clone(), MetadataValue::String(value.clone())))
        .collect();

    Ok(GgufModel {
        version: MAX_SUPPORTED_VERSION,
        metadata,
        tensors,
    })
}

fn stringify_metadata_value(value: &MetadataValue) -> String {
    match value {
        MetadataValue::String(text) => text.clone(),
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
