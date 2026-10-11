//! Offline checkpoint conversion: [`convert_codec`] re-encodes the tensors a
//! [`CodecPolicySettings`] names into the policy's target codec and writes the whole
//! checkpoint back out through [`proxima_gguf::write_complete`], the writer
//! [`crate::safetensors_to_gguf`] feeds.
//!
//! Teaching pointer: the per-tensor work is two existing primitives composed, not a new codec.
//! [`crate::gguf_tensor_as_f32`] dequantizes whatever the source stores (F32, F16, Q8_0, Q4_0,
//! and every other decoder behind `proxima_gguf::quant::dispatch`) and
//! [`proxima_gguf::quant::scaled_float::quantize_block`] packs 32 weights at a time under
//! [`FloatLayout::E3M2`]. Compose them directly when you need one tensor; call this when you
//! need a whole checkpoint with the policy applied and every other tensor carried byte for byte.

use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::quant::QuantError;
use proxima_gguf::quant::scaled_float::{BLOCK_BYTES, BLOCK_ELEMENTS, quantize_block};
use proxima_gguf::tensor::TensorInfo;
use proxima_gguf::{GgmlType, GgufModel, TensorPayload, write_complete};
use proxima_primitives::Codec;
use proxima_primitives::float_layout::FloatLayout;

use crate::bind::gguf_tensor_as_f32;
use crate::codec_policy_settings::CodecPolicySettings;
use crate::error::InteropError;

const WEIGHT_SUFFIX: &str = ".weight";

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;

/// Whether `policy` converts the tensor named `tensor_name`: a `.weight` tensor with one of
/// the policy's classes as a whole dot-separated segment (`blk.3.ffn_up_exps.weight` matches
/// `ffn_up_exps`; `attn_q_norm` does not match `attn_q`, biases never match).
#[must_use]
pub fn policy_selects(policy: &CodecPolicySettings, tensor_name: &str) -> bool {
    tensor_name.strip_suffix(WEIGHT_SUFFIX).is_some_and(|stem| {
        stem.split('.')
            .any(|segment| policy.classes.iter().any(|class| class == segment))
    })
}

/// Converts the tensors `policy` selects and returns the written checkpoint bytes together
/// with how many tensors were converted. The caller derives the rest: kept is
/// `parsed.tensors.len() - converted`, source size is `file_bytes.len()`, output size is the
/// returned length. Metadata, tensor order and every unselected tensor's bytes are carried
/// unchanged.
///
/// # Errors
///
/// [`InteropError::ConversionTargetUnsupported`] if the policy targets a codec with no
/// block encoder here; [`InteropError::RowNotBlockMultiple`] if a selected tensor's rows are
/// not whole blocks; [`InteropError::Quant`] if a source cannot be decoded or a weight is not
/// finite; [`InteropError::Gguf`] if a tensor range or the write fails.
pub fn convert_codec(
    parsed: &ParsedGguf,
    file_bytes: &[u8],
    policy: &CodecPolicySettings,
) -> Result<(Vec<u8>, usize), InteropError> {
    let layout = target_layout(policy)?;
    let converted = parsed
        .tensors
        .iter()
        .map(|tensor| convert_selected(parsed, file_bytes, tensor, policy, layout))
        .collect::<Result<Vec<_>, _>>()?;
    let tensors = parsed
        .tensors
        .iter()
        .zip(&converted)
        .map(|(tensor, blocks)| payload(parsed, file_bytes, tensor, blocks.as_deref()))
        .collect::<Result<Vec<_>, _>>()?;
    let model = GgufModel {
        version: parsed.version,
        metadata: parsed.metadata.clone(),
        tensors,
    };
    Ok((write_complete(&model)?, converted.iter().flatten().count()))
}

fn target_layout(policy: &CodecPolicySettings) -> Result<Option<FloatLayout>, InteropError> {
    match policy.weights.codec() {
        None => Ok(None),
        Some(Codec::Fp6E3M2) => Ok(Some(FloatLayout::E3M2)),
        Some(codec) => Err(InteropError::ConversionTargetUnsupported { codec }),
    }
}

fn convert_selected(
    parsed: &ParsedGguf,
    file_bytes: &[u8],
    tensor: &TensorInfo,
    policy: &CodecPolicySettings,
    layout: Option<FloatLayout>,
) -> Result<Option<Vec<u8>>, InteropError> {
    let Some(layout) = layout.filter(|_| policy_selects(policy, &tensor.name)) else {
        return Ok(None);
    };
    let row_elements = tensor.dims.first().copied().unwrap_or(0);
    if row_elements % BLOCK_ELEMENTS as u64 != 0 || row_elements == 0 {
        return Err(InteropError::RowNotBlockMultiple {
            tensor: tensor.name.clone(),
            row_elements,
            block_elements: BLOCK_ELEMENTS,
        });
    }
    let weights = gguf_tensor_as_f32(parsed, file_bytes, &tensor.name)?;
    Ok(Some(encode_blocks(&weights, layout)?))
}

fn encode_blocks(weights: &[f32], layout: FloatLayout) -> Result<Vec<u8>, QuantError> {
    let mut blocks = vec![0u8; weights.len() / BLOCK_ELEMENTS * BLOCK_BYTES];
    weights
        .as_chunks::<BLOCK_ELEMENTS>()
        .0
        .iter()
        .zip(blocks.as_chunks_mut::<BLOCK_BYTES>().0)
        .try_for_each(|(source, block)| quantize_block(source, layout, block))?;
    Ok(blocks)
}

fn payload<'a>(
    parsed: &ParsedGguf,
    file_bytes: &'a [u8],
    tensor: &TensorInfo,
    blocks: Option<&'a [u8]>,
) -> Result<TensorPayload<'a>, InteropError> {
    let (ggml_type, data) = match blocks {
        Some(blocks) => (GgmlType::Fp6E3M2, blocks),
        None => {
            let range = parsed.tensor_data_range(tensor, file_bytes.len() as u64)?;
            (
                tensor.ggml_type,
                &file_bytes[range.start as usize..range.end as usize],
            )
        }
    };
    Ok(TensorPayload {
        name: tensor.name.clone(),
        dims: tensor.dims.clone(),
        ggml_type,
        data,
    })
}
