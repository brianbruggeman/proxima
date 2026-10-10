//! borrowed tensor views for the pinned lfm2 hybrid layer contract.

use alloc::format;
use alloc::string::String;

use proxima_safetensors::{Manifest, TensorEntry};
use proxima_tensor::DType;
use proxima_tensor::spec::LayerKind;

use crate::error::InteropError;
use crate::hf_config::{HfConfig, architecture_from_hf_config, lfm_layer_kinds_from_manifest};

#[derive(Debug, Clone, Copy)]
pub struct LfmTensorView<'manifest, 'file> {
    /// Exact checkpoint name, such as `model.layers.2.self_attn.q_proj.weight`.
    pub name: &'manifest str,
    /// Scalar format recorded in the safetensors header, such as `BFloat16`.
    pub dtype: DType,
    /// Checkpoint row-major dimensions, such as `[2048, 2048]` for Q projection.
    pub shape: &'manifest [u64],
    /// Raw tensor bytes borrowed from the caller's safetensors file buffer.
    pub bytes: &'file [u8],
}

#[derive(Debug, Clone, Copy)]
pub struct LfmCommonLayerWeights<'manifest, 'file> {
    /// Learned scale applied before the layer's mixer.
    pub operator_norm: LfmTensorView<'manifest, 'file>,
    /// Learned scale applied before the layer's feed-forward network.
    pub ffn_norm: LfmTensorView<'manifest, 'file>,
    /// First SwiGLU projection, named `feed_forward.w1.weight` in the file.
    pub ffn_w1: LfmTensorView<'manifest, 'file>,
    /// Output SwiGLU projection, named `feed_forward.w2.weight` in the file.
    pub ffn_w2: LfmTensorView<'manifest, 'file>,
    /// Second SwiGLU projection, named `feed_forward.w3.weight` in the file.
    pub ffn_w3: LfmTensorView<'manifest, 'file>,
}

#[derive(Debug, Clone, Copy)]
pub enum LfmMixerWeights<'manifest, 'file> {
    /// Depthwise causal convolution and its joined input/output projections.
    ShortConv {
        conv_weight: LfmTensorView<'manifest, 'file>,
        in_proj: LfmTensorView<'manifest, 'file>,
        out_proj: LfmTensorView<'manifest, 'file>,
    },
    /// Grouped-query attention projections and per-head Q/K normalization.
    Attention {
        q_proj: LfmTensorView<'manifest, 'file>,
        k_proj: LfmTensorView<'manifest, 'file>,
        v_proj: LfmTensorView<'manifest, 'file>,
        out_proj: LfmTensorView<'manifest, 'file>,
        q_layernorm: LfmTensorView<'manifest, 'file>,
        k_layernorm: LfmTensorView<'manifest, 'file>,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct LfmLayerWeights<'manifest, 'file> {
    /// Zero-based position in the config's immutable `layer_types` schedule.
    pub layer_index: u32,
    /// Norm and feed-forward tensors shared by both LFM2 mixer kinds.
    pub common: LfmCommonLayerWeights<'manifest, 'file>,
    /// Mixer tensors selected by the pinned layer schedule.
    pub mixer: LfmMixerWeights<'manifest, 'file>,
}

/// binds one lfm2 layer's exact checkpoint tensors as borrowed byte views.
///
/// The manifest establishes names and shapes; the config establishes the mixer
/// schedule and effective SwiGLU width. No tensor bytes are copied.
pub fn bind_lfm_layer<'manifest, 'file>(
    config: &HfConfig,
    manifest: &'manifest Manifest,
    file_bytes: &'file [u8],
    data_start: u64,
    layer_index: u32,
) -> Result<LfmLayerWeights<'manifest, 'file>, InteropError> {
    let layer_kinds = lfm_layer_kinds_from_manifest(config, manifest)?;
    let layer_position = usize::try_from(layer_index).map_err(|_| {
        malformed(format!(
            "layer index {layer_index} exceeds platform index width"
        ))
    })?;
    let layer_kind = layer_kinds.get(layer_position).copied().ok_or_else(|| {
        malformed(format!(
            "layer index {layer_index} is outside the LFM2 schedule"
        ))
    })?;
    let architecture = architecture_from_hf_config(config)?;
    let hidden_width = u64::from(config.hidden_size);
    let feed_forward_width = u64::from(architecture.feed_forward);
    let prefix = format!("model.layers.{layer_index}.");

    let common = LfmCommonLayerWeights {
        operator_norm: tensor_view(
            manifest,
            file_bytes,
            data_start,
            format!("{prefix}operator_norm.weight"),
            &[hidden_width],
        )?,
        ffn_norm: tensor_view(
            manifest,
            file_bytes,
            data_start,
            format!("{prefix}ffn_norm.weight"),
            &[hidden_width],
        )?,
        ffn_w1: tensor_view(
            manifest,
            file_bytes,
            data_start,
            format!("{prefix}feed_forward.w1.weight"),
            &[feed_forward_width, hidden_width],
        )?,
        ffn_w2: tensor_view(
            manifest,
            file_bytes,
            data_start,
            format!("{prefix}feed_forward.w2.weight"),
            &[hidden_width, feed_forward_width],
        )?,
        ffn_w3: tensor_view(
            manifest,
            file_bytes,
            data_start,
            format!("{prefix}feed_forward.w3.weight"),
            &[feed_forward_width, hidden_width],
        )?,
    };

    let mixer = match layer_kind {
        LayerKind::ShortConv => LfmMixerWeights::ShortConv {
            conv_weight: tensor_view(
                manifest,
                file_bytes,
                data_start,
                format!("{prefix}conv.conv.weight"),
                &[hidden_width, 1, u64::from(config.conv_l_cache.unwrap_or(0))],
            )?,
            in_proj: tensor_view(
                manifest,
                file_bytes,
                data_start,
                format!("{prefix}conv.in_proj.weight"),
                &[hidden_width * 3, hidden_width],
            )?,
            out_proj: tensor_view(
                manifest,
                file_bytes,
                data_start,
                format!("{prefix}conv.out_proj.weight"),
                &[hidden_width, hidden_width],
            )?,
        },
        LayerKind::Attention => {
            let head_dim = config
                .head_dim
                .unwrap_or_else(|| config.hidden_size / config.num_attention_heads.max(1));
            let query_width = u64::from(config.num_attention_heads) * u64::from(head_dim);
            let key_value_width = u64::from(
                config
                    .num_key_value_heads
                    .unwrap_or(config.num_attention_heads),
            ) * u64::from(head_dim);
            LfmMixerWeights::Attention {
                q_proj: tensor_view(
                    manifest,
                    file_bytes,
                    data_start,
                    format!("{prefix}self_attn.q_proj.weight"),
                    &[query_width, hidden_width],
                )?,
                k_proj: tensor_view(
                    manifest,
                    file_bytes,
                    data_start,
                    format!("{prefix}self_attn.k_proj.weight"),
                    &[key_value_width, hidden_width],
                )?,
                v_proj: tensor_view(
                    manifest,
                    file_bytes,
                    data_start,
                    format!("{prefix}self_attn.v_proj.weight"),
                    &[key_value_width, hidden_width],
                )?,
                out_proj: tensor_view(
                    manifest,
                    file_bytes,
                    data_start,
                    format!("{prefix}self_attn.out_proj.weight"),
                    &[hidden_width, hidden_width],
                )?,
                q_layernorm: tensor_view(
                    manifest,
                    file_bytes,
                    data_start,
                    format!("{prefix}self_attn.q_layernorm.weight"),
                    &[u64::from(head_dim)],
                )?,
                k_layernorm: tensor_view(
                    manifest,
                    file_bytes,
                    data_start,
                    format!("{prefix}self_attn.k_layernorm.weight"),
                    &[u64::from(head_dim)],
                )?,
            }
        }
        LayerKind::Gdn => return Err(malformed("LFM2 does not define GDN layers".into())),
    };

    Ok(LfmLayerWeights {
        layer_index,
        common,
        mixer,
    })
}

fn tensor_view<'manifest, 'file>(
    manifest: &'manifest Manifest,
    file_bytes: &'file [u8],
    data_start: u64,
    name: String,
    expected_shape: &[u64],
) -> Result<LfmTensorView<'manifest, 'file>, InteropError> {
    let entry = manifest
        .tensor(&name)
        .ok_or_else(|| malformed(format!("required LFM2 tensor {name} is missing")))?;
    if entry.shape != expected_shape {
        return Err(malformed(format!(
            "LFM2 tensor {} has shape {:?}, expected {:?}",
            entry.name, entry.shape, expected_shape
        )));
    }
    validate_tensor_bytes(entry, file_bytes, data_start)
}

fn validate_tensor_bytes<'manifest, 'file>(
    entry: &'manifest TensorEntry,
    file_bytes: &'file [u8],
    data_start: u64,
) -> Result<LfmTensorView<'manifest, 'file>, InteropError> {
    if !matches!(
        entry.dtype,
        DType::Float32 | DType::Float16 | DType::BFloat16
    ) {
        return Err(malformed(format!(
            "LFM2 tensor {} has unsupported scalar dtype {:?}",
            entry.name, entry.dtype
        )));
    }
    let element_count = entry
        .shape
        .iter()
        .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension));
    let element_byte_len = u64::try_from(entry.dtype.size_bytes()).map_err(|_| {
        malformed(format!(
            "LFM2 tensor {} element width exceeds u64",
            entry.name
        ))
    })?;
    let expected_byte_len = element_count
        .and_then(|count| count.checked_mul(element_byte_len))
        .ok_or_else(|| malformed(format!("LFM2 tensor {} byte length overflows", entry.name)))?;
    let declared_byte_len = entry
        .data_offsets
        .1
        .checked_sub(entry.data_offsets.0)
        .ok_or_else(|| {
            malformed(format!(
                "LFM2 tensor {} has reversed data offsets",
                entry.name
            ))
        })?;
    if declared_byte_len != expected_byte_len {
        return Err(malformed(format!(
            "LFM2 tensor {} declares {} bytes for shape {:?} and dtype {:?}, expected {expected_byte_len}",
            entry.name, declared_byte_len, entry.shape, entry.dtype
        )));
    }
    let start = data_start
        .checked_add(entry.data_offsets.0)
        .and_then(|offset| usize::try_from(offset).ok());
    let end = data_start
        .checked_add(entry.data_offsets.1)
        .and_then(|offset| usize::try_from(offset).ok());
    let Some((start, end)) = start.zip(end) else {
        return Err(malformed(format!(
            "LFM2 tensor {} data range overflows",
            entry.name
        )));
    };
    let bytes = file_bytes.get(start..end).ok_or_else(|| {
        malformed(format!(
            "LFM2 tensor {} data range {start}..{end} exceeds {} bytes",
            entry.name,
            file_bytes.len()
        ))
    })?;
    Ok(LfmTensorView {
        name: entry.name.as_str(),
        dtype: entry.dtype,
        shape: entry.shape.as_slice(),
        bytes,
    })
}

fn malformed(reason: String) -> InteropError {
    InteropError::MalformedHfConfig { reason }
}

#[cfg(test)]
mod tests;
