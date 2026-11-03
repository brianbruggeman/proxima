//! The affine-quantized export of a GGUF as Hugging Face named tensors, whole:
//! [`export_affine_named`] is [`super::mlx_weight_plan::WeightPlan`] with a
//! quantization, converted eagerly. A caller that writes the weights to disk
//! drives the plan block by block instead.

use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;
use proxima_tensor::DType;

use super::mlx_affine::AffineParams;
use super::mlx_conventions::QuantizedConventions;
use super::mlx_names::Layout;
use super::mlx_plain::PlainTensor;
use super::mlx_weight_plan::{WeightPlan, available_workers};
use crate::error::InteropError;
use crate::profiles::BindingProfile;

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
    let quantization = Some((params, conventions));
    WeightPlan::new(
        parsed,
        file_bytes,
        dtype,
        binding,
        layout,
        hf_prefix,
        quantization,
    )?
    .convert_all(available_workers())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use arrayvec::ArrayVec;
    use proxima_gguf::{GgmlType, GgufModel, TensorPayload, parse_complete, write_complete};

    use super::*;
    use crate::profiles::binding_profile;
    use crate::transform::STANDARD_LAYOUT;
    use crate::transform::mlx_affine::{AffineExport, export_affine};
    use crate::transform::mlx_plain::export_plain_tensor;

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
        let tensors = export_with(32, DType::BFLOAT16, &matrix_and_norm()).unwrap();

        assert_eq!(
            shapes_by_name(&tensors),
            vec![
                (
                    "model.layers.0.mlp.down_proj.weight",
                    DType::UINT32,
                    &[2, 4][..]
                ),
                (
                    "model.layers.0.mlp.down_proj.scales",
                    DType::BFLOAT16,
                    &[2, 1][..]
                ),
                (
                    "model.layers.0.mlp.down_proj.biases",
                    DType::BFLOAT16,
                    &[2, 1][..]
                ),
                (
                    "model.layers.0.input_layernorm.weight",
                    DType::BFLOAT16,
                    &[32][..]
                ),
            ]
        );
    }

    #[test]
    fn affine_named_triple_bytes_equal_export_affine_of_the_hand_derived_q8_0_values() {
        let tensors = export_with(32, DType::BFLOAT16, &matrix_and_norm()).unwrap();

        let params = AffineParams::new(4, 32).unwrap();
        let direct =
            export_affine(&q8_0_matrix_values(), [2, 32], params, DType::BFLOAT16).unwrap();
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

        let original = export_with(32, DType::BFLOAT16, &matrix_and_norm()).unwrap();
        let tensors = export_with(32, DType::BFLOAT16, &changed).unwrap();

        assert_ne!(tensors[0].bytes, original[0].bytes);
    }

    #[test]
    fn affine_named_matrix_narrower_than_the_group_stays_a_bf16_matrix() {
        let tensors = export_with(64, DType::BFLOAT16, &matrix_and_norm()).unwrap();

        assert_eq!(
            shapes_by_name(&tensors)[0],
            (
                "model.layers.0.mlp.down_proj.weight",
                DType::BFLOAT16,
                &[2, 32][..]
            )
        );
        assert_eq!(tensors.len(), 2);
    }

    #[test]
    fn affine_named_refuses_an_integer_scale_dtype() {
        let refusal = export_with(32, DType::UINT32, &matrix_and_norm()).unwrap_err();

        assert!(matches!(
            refusal,
            InteropError::AffineDTypeUnsupported {
                dtype: DType::UINT32
            }
        ));
    }

    #[test]
    fn parallel_quantize_is_byte_equal_to_the_sequential_oracle_at_every_worker_count() {
        let bytes = gguf_of(&[
            (MATRIX, &[32, 2], GgmlType::Q8_0, &q8_0_matrix_bytes()),
            (
                "blk.1.ffn_down.weight",
                &[32, 2],
                GgmlType::Q8_0,
                &q8_0_matrix_bytes(),
            ),
            (NORM, &[32], GgmlType::F32, &norm_bytes()),
            (
                "blk.1.attn_norm.weight",
                &[32],
                GgmlType::F32,
                &norm_bytes(),
            ),
            (
                "blk.0.ffn_gate_exps.weight",
                &[32, 2],
                GgmlType::Q8_0,
                &q8_0_matrix_bytes(),
            ),
            (
                "blk.0.ffn_up_exps.weight",
                &[32, 2],
                GgmlType::Q8_0,
                &q8_0_matrix_bytes(),
            ),
        ]);
        let parsed = parse_complete(&bytes).expect("parses the gguf just written");
        let binding = binding_profile("granitemoe").expect("granitemoe binding profile");
        let params = AffineParams::new(4, 32).expect("grid member");
        let plan = WeightPlan::new(
            &parsed,
            &bytes,
            DType::BFLOAT16,
            &binding,
            STANDARD_LAYOUT,
            "model.",
            Some((params, QuantizedConventions::NONE)),
        )
        .expect("maps every leaf of the fixture");
        let quantize = |workers| plan.convert_all(workers).expect("quantizes the fixture");

        let sequential = quantize(1);

        assert_eq!(
            sequential.len(),
            11,
            "two matrices and one fused pair as triples, two norms"
        );
        assert!(
            shapes_by_name(&sequential)
                .iter()
                .any(|(name, _, _)| *name == "model.layers.0.block_sparse_moe.input_linear.weight"),
            "the fused gate and up pair packs through pack_fused"
        );
        for workers in [2, 3, 8] {
            assert_eq!(quantize(workers), sequential, "workers={workers}");
        }
    }

    #[test]
    fn plain_one_row_blocks_equal_the_whole_tensor_export_of_each_leaf() {
        let bytes = matrix_and_norm();
        let parsed = parse_complete(&bytes).expect("parses the gguf just written");
        let binding = binding_profile("granitemoe").expect("granitemoe binding profile");
        let mut plan = WeightPlan::new(
            &parsed,
            &bytes,
            DType::BFLOAT16,
            &binding,
            STANDARD_LAYOUT,
            "model.",
            None,
        )
        .expect("maps every leaf of the fixture");
        plan.block_elements = 1;

        let blocked = plan.convert_all(2).expect("one-row blocks");

        let whole = |leaf: &str| {
            let info = parsed
                .tensors
                .iter()
                .find(|tensor| tensor.name == leaf)
                .unwrap();
            let range = parsed.tensor_data_range(info, bytes.len() as u64).unwrap();
            export_plain_tensor(
                info,
                &bytes[range.start as usize..range.end as usize],
                DType::BFLOAT16,
            )
            .unwrap()
            .bytes
        };
        assert_eq!(blocked.len(), 2);
        assert_eq!(blocked[0].bytes, whole(MATRIX));
        assert_eq!(blocked[1].bytes, whole(NORM));
    }

    #[test]
    fn one_row_blocks_are_byte_equal_to_whole_tensor_blocks_and_headers_describe_every_tensor() {
        let bytes = gguf_of(&[
            (MATRIX, &[32, 2], GgmlType::Q8_0, &q8_0_matrix_bytes()),
            (NORM, &[32], GgmlType::F32, &norm_bytes()),
            (
                "blk.0.ffn_gate_exps.weight",
                &[32, 2],
                GgmlType::Q8_0,
                &q8_0_matrix_bytes(),
            ),
            (
                "blk.0.ffn_up_exps.weight",
                &[32, 2],
                GgmlType::Q8_0,
                &q8_0_matrix_bytes(),
            ),
        ]);
        let parsed = parse_complete(&bytes).expect("parses the gguf just written");
        let binding = binding_profile("granitemoe").expect("granitemoe binding profile");
        let params = AffineParams::new(4, 32).expect("grid member");
        let mut plan = WeightPlan::new(
            &parsed,
            &bytes,
            DType::BFLOAT16,
            &binding,
            STANDARD_LAYOUT,
            "model.",
            Some((params, QuantizedConventions::NONE)),
        )
        .expect("maps every leaf of the fixture");
        let whole = plan.convert_all(1).expect("whole-tensor blocks");

        plan.block_elements = 1;
        let blocks = plan.blocks();
        let blocked = plan.convert_all(3).expect("one-row blocks");

        assert_eq!(
            blocks.len(),
            5,
            "two rows of the matrix, one of the norm, two experts of the fused pair"
        );
        assert_eq!(blocked, whole);
        let headers = plan.headers().expect("headers");
        assert_eq!(
            headers
                .iter()
                .map(|header| (header.name.as_str(), header.dtype, header.shape.as_slice()))
                .collect::<Vec<_>>(),
            shapes_by_name(&whole)
        );
        assert!(
            headers
                .iter()
                .zip(&whole)
                .all(|(header, tensor)| header.byte_len() == tensor.bytes.len() as u64)
        );
    }
}
