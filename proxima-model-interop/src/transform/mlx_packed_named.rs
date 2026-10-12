//! The affine-quantized export of a GGUF as Hugging Face named tensors: the
//! stored leaves of [`super::export_plain_named`] with every `.weight` matrix
//! replaced by the `weight`, `scales` and `biases` triple of
//! [`super::export_affine`], the rest cast to the export float dtype.
//!
//! One tensor is decoded, packed and dropped at a time, so peak memory is the
//! packed output plus the largest single tensor as `f32`, not the whole model
//! as `f32`. A matrix whose last axis is not a multiple of the group size
//! stays float, as in mlx-lm. Quantization is uniform: one [`AffineParams`]
//! for every matrix.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use proxima_gguf::TensorInfo;
use proxima_gguf::pipe::ParsedGguf;
use proxima_tensor::DType;

use super::mlx_affine::{AffineExport, AffineParams, export_affine};
use super::mlx_conventions::QuantizedConventions;
use super::mlx_names::{FusedName, Layout, MlxNames};
use super::mlx_plain::{PlainTensor, export_plain_leaf};
use super::mlx_plain_named::{interleave, leaf_names};
use crate::error::InteropError;
use crate::profiles::BindingProfile;

type Directory<'parsed> = BTreeMap<&'parsed str, &'parsed TensorInfo>;

/// Every exportable tensor of `parsed` quantized with `params`, float
/// remainders and the `scales` and `biases` as `dtype`, named for `layout`
/// under `hf_prefix`.
///
/// # Errors
///
/// [`InteropError::MlxUnmappedLeaves`] when a leaf has no stored tensor and no
/// layout row; [`InteropError::AffineDTypeUnsupported`] for a non-float
/// `dtype`; whatever [`super::export_plain`] and [`super::export_affine`] can
/// fail with.
#[allow(clippy::too_many_arguments)]
// the eight inputs are the checkpoint, the naming and the family's quantized conventions;
// grouping them would mint a type whose only job is to be passed here
pub fn export_affine_named(
    parsed: &ParsedGguf,
    file_bytes: &[u8],
    params: AffineParams,
    dtype: DType,
    binding: &BindingProfile,
    layout: Layout,
    hf_prefix: &str,
    conventions: QuantizedConventions,
) -> Result<Vec<PlainTensor>, InteropError> {
    let names = conventions.apply(leaf_names(parsed, binding, layout, hf_prefix)?);
    let directory: Directory = parsed
        .tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect();
    let load =
        |tensor: &TensorInfo, kind: DType| export_plain_leaf(parsed, file_bytes, tensor, kind);
    let mut groups = pack_singles(&names, &directory, &load, (params, conventions), dtype)?;
    groups.extend(pack_fused(&names, &directory, &load, params, dtype)?);
    Ok(groups.into_iter().flatten().collect())
}

fn pack_singles(
    names: &MlxNames,
    directory: &Directory,
    load: &impl Fn(&TensorInfo, DType) -> Result<PlainTensor, InteropError>,
    (base, conventions): (AffineParams, QuantizedConventions),
    dtype: DType,
) -> Result<Vec<Vec<PlainTensor>>, InteropError> {
    names
        .mapped
        .iter()
        .filter_map(|(leaf, hf)| directory.get(leaf.as_str()).map(|info| (*info, hf)))
        .map(|(info, hf)| {
            let quantizable = hf.ends_with(".weight") && info.dims.len() >= 2;
            let params = conventions.params_for(hf, base)?;
            let read = |kind: DType| load(info, kind).map(|tensor| renamed(tensor, hf));
            quantize_or_cast(read, quantizable, params, dtype)
        })
        .collect()
}

fn pack_fused(
    names: &MlxNames,
    directory: &Directory,
    load: &impl Fn(&TensorInfo, DType) -> Result<PlainTensor, InteropError>,
    params: AffineParams,
    dtype: DType,
) -> Result<Vec<Vec<PlainTensor>>, InteropError> {
    fused_groups(names, directory)
        .into_iter()
        .map(|(hf, parts)| {
            let read = |kind: DType| {
                let loaded: Result<Vec<PlainTensor>, InteropError> =
                    parts.iter().map(|info| load(info, kind)).collect();
                loaded.map(|loaded| interleave(hf, &loaded))
            };
            quantize_or_cast(read, true, params, dtype)
        })
        .collect()
}

fn renamed(mut tensor: PlainTensor, name: &str) -> PlainTensor {
    tensor.name = name.into();
    tensor
}

fn fused_groups<'names, 'parsed>(
    names: &'names MlxNames,
    directory: &Directory<'parsed>,
) -> Vec<(&'names str, Vec<&'parsed TensorInfo>)> {
    let targets: alloc::collections::BTreeSet<&str> =
        names.fused.iter().map(|item| item.hf.as_str()).collect();
    targets
        .into_iter()
        .filter_map(|target| {
            let mut group: Vec<&FusedName> = names
                .fused
                .iter()
                .filter(|item| item.hf == target)
                .collect();
            group.sort_by_key(|item| item.part);
            let parts: Option<Vec<&TensorInfo>> = group
                .iter()
                .map(|item| directory.get(item.leaf.as_str()).copied())
                .collect();
            parts.map(|parts| (target, parts))
        })
        .collect()
}

fn quantize_or_cast(
    read: impl Fn(DType) -> Result<PlainTensor, InteropError>,
    quantizable: bool,
    params: AffineParams,
    dtype: DType,
) -> Result<Vec<PlainTensor>, InteropError> {
    if !quantizable {
        return Ok(vec![read(dtype)?]);
    }
    match pack_tensor(&read(DType::Float32)?, params, dtype)? {
        Some(triple) => Ok(triple.into()),
        None => Ok(vec![read(dtype)?]),
    }
}

fn pack_tensor(
    tensor: &PlainTensor,
    params: AffineParams,
    dtype: DType,
) -> Result<Option<[PlainTensor; 3]>, InteropError> {
    let columns = tensor
        .shape
        .last()
        .map_or(0, |dim| usize::try_from(*dim).unwrap_or(0));
    let values: Vec<f32> = tensor
        .bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();
    let rows = values.len().checked_div(columns).unwrap_or(0);
    let AffineExport::Packed(packed) = export_affine(&values, [rows, columns], params, dtype)?
    else {
        return Ok(None);
    };
    let Some(base) = tensor.name.strip_suffix(".weight") else {
        return Ok(None);
    };
    let lead = &tensor.shape[..tensor.shape.len() - 1];
    let shaped = |last: usize| [lead, &[last as u64]].concat();
    let words = packed.weight.iter().flat_map(|word| word.to_le_bytes());
    Ok(Some([
        PlainTensor {
            name: format!("{base}.weight"),
            shape: shaped(packed.weight_shape[1]),
            dtype: DType::UInt32,
            bytes: words.collect(),
        },
        PlainTensor {
            name: format!("{base}.scales"),
            shape: shaped(packed.scales_shape[1]),
            dtype: packed.dtype,
            bytes: packed.scales,
        },
        PlainTensor {
            name: format!("{base}.biases"),
            shape: shaped(packed.scales_shape[1]),
            dtype: packed.dtype,
            bytes: packed.biases,
        },
    ]))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use arrayvec::ArrayVec;
    use proxima_gguf::{GgmlType, GgufModel, TensorPayload, parse_complete, write_complete};

    use super::*;
    use crate::profiles::binding_profile;
    use crate::transform::STANDARD_LAYOUT;

    const MATRIX: &str = "blk.0.ffn_down.weight";
    const NORM: &str = "blk.0.attn_norm.weight";

    fn q8_0_matrix_bytes() -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&0x3800_u16.to_le_bytes());
        data.extend((0..32_i8).map(|index| (index - 16) as u8));
        data.extend_from_slice(&0x3400_u16.to_le_bytes());
        data.extend((0..32).map(|index| if index == 0 { 0x80 } else { 0x7F }));
        data
    }

    fn q8_0_matrix_values() -> Vec<f32> {
        let mut values: Vec<f32> = (0..32).map(|index| (index as f32 - 16.0) * 0.5).collect();
        values.push(-32.0);
        values.extend([31.75; 31]);
        values
    }

    fn norm_bytes() -> Vec<u8> {
        (0..32)
            .flat_map(|index| (1.0_f32 + index as f32 * 0.03125).to_le_bytes())
            .collect()
    }

    fn gguf_of(tensors: &[(&str, &[u64], GgmlType, &[u8])]) -> Vec<u8> {
        let payloads = tensors
            .iter()
            .map(|(name, dims, ggml_type, data)| TensorPayload {
                name: (*name).to_string(),
                dims: dims.iter().copied().collect::<ArrayVec<u64, 4>>(),
                ggml_type: *ggml_type,
                data,
            })
            .collect();
        write_complete(&GgufModel {
            version: 3,
            metadata: Vec::new(),
            tensors: payloads,
        })
        .expect("writes the gguf")
    }

    fn export_with(
        group_size: usize,
        dtype: DType,
        bytes: &[u8],
    ) -> Result<Vec<PlainTensor>, InteropError> {
        let parsed = parse_complete(bytes).expect("parses the gguf just written");
        let binding = binding_profile("granitemoe").expect("granitemoe binding profile");
        let params = AffineParams::new(4, group_size).expect("grid member");
        export_affine_named(
            &parsed,
            bytes,
            params,
            dtype,
            &binding,
            STANDARD_LAYOUT,
            "model.",
            QuantizedConventions::NONE,
        )
    }

    fn matrix_and_norm() -> Vec<u8> {
        gguf_of(&[
            (MATRIX, &[32, 2], GgmlType::Q8_0, &q8_0_matrix_bytes()),
            (NORM, &[32], GgmlType::F32, &norm_bytes()),
        ])
    }

    fn shapes_by_name(tensors: &[PlainTensor]) -> Vec<(&str, DType, &[u64])> {
        tensors
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor.dtype, tensor.shape.as_slice()))
            .collect()
    }

    #[test]
    fn affine_named_matrix_becomes_a_weight_scales_biases_triple_and_the_norm_stays_bf16() {
        let tensors = export_with(32, DType::BFloat16, &matrix_and_norm()).unwrap();

        assert_eq!(
            shapes_by_name(&tensors),
            vec![
                (
                    "model.layers.0.mlp.down_proj.weight",
                    DType::UInt32,
                    &[2, 4][..]
                ),
                (
                    "model.layers.0.mlp.down_proj.scales",
                    DType::BFloat16,
                    &[2, 1][..]
                ),
                (
                    "model.layers.0.mlp.down_proj.biases",
                    DType::BFloat16,
                    &[2, 1][..]
                ),
                (
                    "model.layers.0.input_layernorm.weight",
                    DType::BFloat16,
                    &[32][..]
                ),
            ]
        );
    }

    #[test]
    fn affine_named_triple_bytes_equal_export_affine_of_the_hand_derived_q8_0_values() {
        let tensors = export_with(32, DType::BFloat16, &matrix_and_norm()).unwrap();

        let params = AffineParams::new(4, 32).unwrap();
        let direct =
            export_affine(&q8_0_matrix_values(), [2, 32], params, DType::BFloat16).unwrap();
        let AffineExport::Packed(packed) = direct else {
            panic!("32 columns at group 32 quantize");
        };
        let words: Vec<u8> = packed
            .weight
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        assert_eq!(tensors[0].bytes, words);
        assert_eq!(tensors[1].bytes, packed.scales);
        assert_eq!(tensors[2].bytes, packed.biases);
    }

    #[test]
    fn affine_named_degenerate_control_altered_source_value_changes_the_packed_bytes() {
        let mut altered = q8_0_matrix_bytes();
        altered[2] ^= 0x7F;
        let changed = gguf_of(&[(MATRIX, &[32, 2], GgmlType::Q8_0, &altered)]);

        let original = export_with(32, DType::BFloat16, &matrix_and_norm()).unwrap();
        let tensors = export_with(32, DType::BFloat16, &changed).unwrap();

        assert_ne!(tensors[0].bytes, original[0].bytes);
    }

    #[test]
    fn affine_named_matrix_narrower_than_the_group_stays_a_bf16_matrix() {
        let tensors = export_with(64, DType::BFloat16, &matrix_and_norm()).unwrap();

        assert_eq!(
            shapes_by_name(&tensors)[0],
            (
                "model.layers.0.mlp.down_proj.weight",
                DType::BFloat16,
                &[2, 32][..]
            )
        );
        assert_eq!(tensors.len(), 2);
    }

    #[test]
    fn affine_named_refuses_an_integer_scale_dtype() {
        let refusal = export_with(32, DType::UInt32, &matrix_and_norm()).unwrap_err();

        assert!(matches!(
            refusal,
            InteropError::AffineDTypeUnsupported {
                dtype: DType::UInt32
            }
        ));
    }
}
