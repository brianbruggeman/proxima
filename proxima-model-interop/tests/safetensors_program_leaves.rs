#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(feature = "std")]

use std::collections::BTreeMap;
use std::vec;
use std::vec::Vec;

use proxima_model_interop::profiles::{BindingProfile, TensorAlias};
use proxima_gguf::types::GgmlType;
use proxima_model_interop::{
    BoundWeights, Codec, InteropError, LeasedWeights, ModelWeights, NamePattern,
    WeightPrecisionRule, retarget_recoded_leaves,
};
use proxima_safetensors::{Manifest, SafetensorsModel, TensorPayload, write_complete};
use proxima_numeric::DType;
use proxima_tensor::cpu::QuantizedBlock;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::op::{Extent, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp, append};

/// The unruled bind every test here exercises: no `weight_precision` rules,
/// so each leaf keeps its stored dtype.
fn bind_safetensors_program_leaves<'file>(
    manifest: &Manifest,
    file_bytes: &'file [u8],
    data_start: u64,
    program: &[Op],
    binding: &BindingProfile,
    runtime_inputs: &[String],
) -> Result<BoundWeights<'file>, InteropError> {
    proxima_model_interop::bind_safetensors_program_leaves(
        manifest,
        file_bytes,
        data_start,
        program,
        binding,
        runtime_inputs,
        &[],
    )
}

fn fixture_tensor_bytes<'file>(
    file_bytes: &'file [u8],
    data_start: u64,
    entry: &proxima_safetensors::TensorEntry,
) -> &'file [u8] {
    let start = usize::try_from(
        data_start
            .checked_add(entry.data_offsets.0)
            .expect("fixture tensor start does not overflow"),
    )
    .expect("fixture tensor start fits usize");
    let end = usize::try_from(
        data_start
            .checked_add(entry.data_offsets.1)
            .expect("fixture tensor end does not overflow"),
    )
    .expect("fixture tensor end fits usize");
    file_bytes
        .get(start..end)
        .expect("fixture tensor range lies within bytes")
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<u8>>()
}

fn bound_f32_values<'weights>(
    weights: &'weights BoundWeights<'_>,
    name: &str,
) -> Option<&'weights [f32]> {
    weights
        .owned()
        .iter()
        .find_map(|(bound_name, values)| (bound_name == name).then_some(values.as_slice()))
        .or_else(|| {
            weights.packed().iter().find_map(|(bound_name, block)| {
                if bound_name == name {
                    if let QuantizedBlock::Float32(values) = block {
                        Some(*values)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
        })
}

fn program_for(weight: &str, norm: &str, alias: &str, token: &str) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let activations = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(1), Extent::Static(3)],
            name: Some(token.into()),
        },
    );
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(3), Extent::Static(2)],
            name: Some(weight.into()),
        },
    );
    append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(2)],
            name: Some(norm.into()),
        },
    );
    append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(2)],
            name: Some(alias.into()),
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (activations, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (weight, IndexMap::Affine(map::projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    let output = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, output)
}

fn safetensors_fixture() -> Vec<u8> {
    for padding_length in 0..4 {
        let tensors = [
            (
                "matrix.f32",
                DType::FLOAT32,
                vec![2, 3],
                f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ),
            (
                "alignment.pad",
                DType::UINT8,
                vec![padding_length],
                vec![0; padding_length as usize],
            ),
            ("norm.f32", DType::FLOAT32, vec![2], f32_bytes(&[1.0, 2.0])),
            (
                "matrix.f16",
                DType::FLOAT16,
                vec![2, 3],
                [0x3c00u16, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect(),
            ),
            (
                "norm.f16",
                DType::FLOAT16,
                vec![2],
                [0x3c00u16, 0x4000]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect(),
            ),
            (
                "matrix.bf16",
                DType::BFLOAT16,
                vec![2, 3],
                [0x3f80u16, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect(),
            ),
            (
                "norm.bf16",
                DType::BFLOAT16,
                vec![2],
                [0x3f80u16, 0x4000]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect(),
            ),
            (
                "extra.weight",
                DType::FLOAT32,
                vec![2],
                f32_bytes(&[7.0, 8.0]),
            ),
        ];
        let payloads = tensors
            .iter()
            .map(|(name, dtype, shape, bytes)| TensorPayload {
                name: (*name).into(),
                dtype: *dtype,
                shape: shape.clone(),
                data: bytes,
            })
            .collect();
        let bytes = write_complete(&SafetensorsModel {
            tensors: payloads,
            metadata: BTreeMap::new(),
        })
        .expect("writes mixed dtype safetensors fixture");
        let manifest = proxima_safetensors::parse_complete(&bytes)
            .expect("parses mixed dtype safetensors fixture");
        let data_start = header_data_start(&bytes) as usize;
        let entry = manifest.tensor("norm.f32").expect("fixture has f32 norm");
        let address = bytes.as_ptr().addr() + data_start + entry.data_offsets.0 as usize;
        if address.is_multiple_of(core::mem::align_of::<f32>()) {
            return bytes;
        }
    }
    panic!("one of four byte paddings must align the f32 tensor");
}

/// `8 + header_len`, read off a real written safetensors buffer's own
/// 8-byte little-endian length prefix -- the `data_start` every
/// `bind_all_weights_from_safetensors` call in this module needs, since
/// `Manifest`'s own `data_offsets` are relative to this point (see
/// `bind_all_weights_from_safetensors`'s own doc), not to byte 0.
fn header_data_start(file_bytes: &[u8]) -> u64 {
    let mut length_prefix = [0u8; 8];
    length_prefix.copy_from_slice(&file_bytes[..8]);
    8 + u64::from_le_bytes(length_prefix)
}

/// The smallest real dense architecture this crate's weight names can
/// bind: 1 layer, embedding=4, feed_forward=8, 2 query heads, 1 kv head
/// (GQA), head_dim=2, vocab=3 -- every dimension distinct so a
/// transposed or mis-shaped bind would produce a length mismatch, not
/// silently pass.
/// Synthetic scalar matmul evaluation proves bound storage is consumable.
/// Proves generic safetensors bindings feed the existing CPU evaluator.
#[test]
fn architecture_matrix_safetensors_program_leaves() {
    let file_bytes = safetensors_fixture();
    let manifest = proxima_safetensors::parse_complete(&file_bytes)
        .expect("parses mixed dtype safetensors fixture");
    let data_start = header_data_start(&file_bytes);
    let aliases = vec![
        TensorAlias::Rename {
            leaf: "alias.f32".into(),
            from: "norm.f32".into(),
        },
        TensorAlias::Rename {
            leaf: "alias.f16".into(),
            from: "norm.f16".into(),
        },
        TensorAlias::Rename {
            leaf: "alias.bf16".into(),
            from: "norm.bf16".into(),
        },
    ];
    let mut bound_leaf_count = 0;
    let mut runtime_input_count = 0;
    let mut packed_matmul_count = 0;
    let mut native_decode_count = 0;
    let mut f32_transpose_count = 0;
    let mut alias_match_count = 0;
    let mut evaluated_matmul_count = 0;
    let mut extra_weight_count = 0;
    let mut f32_borrow_count = 0;
    let mut part_alias_rejected = 0;
    let mut join_alias_rejected = 0;

    for (suffix, matrix_dtype) in [
        ("f32", DType::FLOAT32),
        ("f16", DType::FLOAT16),
        ("bf16", DType::BFLOAT16),
    ] {
        let matrix_name = format!("matrix.{suffix}");
        let norm_name = format!("norm.{suffix}");
        let alias_name = format!("alias.{suffix}");
        let token_name = format!("token.{suffix}");
        let (program, output) = program_for(&matrix_name, &norm_name, &alias_name, &token_name);
        let binding = BindingProfile {
            aliases: aliases.clone(),
            decode_f32: Vec::new(),
            extra: if suffix == "f32" {
                vec!["extra.weight".into()]
            } else {
                Vec::new()
            },
            optional: Vec::new(),
            unread_prefixes: Vec::new(),
        };
        let weights = bind_safetensors_program_leaves(
            &manifest,
            &file_bytes,
            data_start,
            &program,
            &binding,
            std::slice::from_ref(&token_name),
        )
        .expect("program leaves bind from manifest/profile data");
        let program_weight_names = [&matrix_name, &norm_name, &alias_name];
        bound_leaf_count += program_weight_names
            .iter()
            .filter(|weight_name| {
                weights
                    .owned()
                    .iter()
                    .any(|(name, _)| name == **weight_name)
                    || weights
                        .packed()
                        .iter()
                        .any(|(name, _)| name == **weight_name)
            })
            .count();
        runtime_input_count += usize::from(
            !weights.owned().iter().any(|(name, _)| name == &token_name)
                && !weights.packed().iter().any(|(name, _)| name == &token_name),
        );
        let norm_values =
            bound_f32_values(&weights, &norm_name).expect("native norm is bound as f32 values");
        let alias_values =
            bound_f32_values(&weights, &alias_name).expect("renamed norm is bound as f32 values");
        assert_eq!(
            alias_values, norm_values,
            "renamed leaf uses its configured source bytes"
        );
        alias_match_count += 1;
        if matrix_dtype != DType::FLOAT32 {
            let matrix_block = weights
                .packed()
                .iter()
                .find(|(name, _)| name == &matrix_name)
                .expect("f16 and bf16 matmuls remain packed");
            let QuantizedBlock::Packed { bytes, codec } = matrix_block.1 else {
                panic!("low-precision matrix should be a borrowed packed block");
            };
            let expected_codec = match matrix_dtype {
                DType::FLOAT16 => Codec::Float16,
                DType::BFLOAT16 => Codec::BFloat16,
                _ => panic!("fixture low-precision matrix has a supported dtype"),
            };
            assert_eq!(codec, expected_codec);
            let entry = manifest
                .tensor(&matrix_name)
                .expect("fixture has matrix tensor");
            let original_bytes = fixture_tensor_bytes(&file_bytes, data_start, entry);
            assert_eq!(bytes, original_bytes);
            assert!(core::ptr::eq(bytes.as_ptr(), original_bytes.as_ptr()));
            packed_matmul_count += 1;
            native_decode_count += usize::from(norm_values == [1.0, 2.0]);
            native_decode_count += usize::from(alias_values == [1.0, 2.0]);
        } else {
            let transposed = weights
                .owned()
                .iter()
                .find(|(name, _)| name == &matrix_name)
                .expect("f32 matrix is decoded and transposed");
            assert_eq!(transposed.1, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
            f32_transpose_count += 1;
            assert_eq!(
                bound_f32_values(&weights, "extra.weight"),
                Some(&[7.0, 8.0][..]),
                "configured extra weight is bound with its exact values"
            );
            extra_weight_count += 1;
            f32_borrow_count += usize::from(weights.packed().iter().any(|(name, block)| {
                name == &norm_name && matches!(block, QuantizedBlock::Float32(_))
            }));
            let norm_block = weights
                .packed()
                .iter()
                .find(|(name, _)| name == &norm_name)
                .expect("aligned f32 norm borrows from the safetensors buffer");
            let QuantizedBlock::Float32(bound_norm_bytes) = norm_block.1 else {
                panic!("native f32 norm must borrow its payload");
            };
            let entry = manifest
                .tensor(&norm_name)
                .expect("fixture has native f32 norm");
            let source_bytes = fixture_tensor_bytes(&file_bytes, data_start, entry);
            assert!(core::ptr::eq(
                bound_norm_bytes.as_ptr().cast::<u8>(),
                source_bytes.as_ptr()
            ));
        }
        assert!(!weights.owned().iter().any(|(name, _)| name == &token_name));
        assert!(!weights.packed().iter().any(|(name, _)| name == &token_name));

        let token_values = [1.0f32, 2.0, 3.0];
        let mut named = vec![(token_name.as_str(), QuantizedBlock::Float32(&token_values))];
        for (name, values) in weights.owned() {
            named.push((name.as_str(), QuantizedBlock::Float32(values)));
        }
        for (name, block) in weights.packed() {
            named.push((name.as_str(), *block));
        }
        let evaluated =
            proxima_tensor::cpu::evaluate_quantized_named(&program, &[], &named, &[output])
                .expect("evaluates each dtype's bound matrix");
        let (values, axes) = evaluated.get(output).expect("matmul result is retained");
        assert_eq!(axes, [1, 2]);
        assert_eq!(values, [14.0, 32.0]);
        evaluated_matmul_count += 1;
    }

    let (f16_program, _) = program_for("matrix.f16", "norm.f16", "alias.f16", "token.f16");
    let forced_decode = bind_safetensors_program_leaves(
        &manifest,
        &file_bytes,
        data_start,
        &f16_program,
        &BindingProfile {
            aliases: aliases.clone(),
            decode_f32: vec!["matrix.f16".into()],
            ..BindingProfile::default()
        },
        &["token.f16".into()],
    )
    .expect("profile can force a packed f16 matmul to owned f32");
    assert_eq!(
        forced_decode
            .owned()
            .iter()
            .find(|(name, _)| name == "matrix.f16")
            .expect("forced decode produces an owned f32 matrix")
            .1,
        [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
    assert!(
        forced_decode
            .packed()
            .iter()
            .all(|(name, _)| name != "matrix.f16")
    );
    let decode_rule_count = 1;

    assert_eq!(bound_leaf_count, 9);
    assert_eq!(runtime_input_count, 3);
    assert_eq!(packed_matmul_count, 2);
    assert_eq!(native_decode_count, 4);
    assert_eq!(f32_transpose_count, 1);
    assert_eq!(alias_match_count, 3);
    assert_eq!(evaluated_matmul_count, 3);
    assert_eq!(extra_weight_count, 1);
    assert_eq!(decode_rule_count, 1);
    assert_eq!(f32_borrow_count, 1);

    let f32_entry = manifest.tensor("norm.f32").expect("fixture has f32 norm");
    let mut misaligned_manifest = manifest.clone();
    for entry in &mut misaligned_manifest.tensors {
        if entry.data_offsets.0 >= f32_entry.data_offsets.0 {
            entry.data_offsets = (entry.data_offsets.0 + 1, entry.data_offsets.1 + 1);
        }
    }
    let absolute_start = data_start as usize + f32_entry.data_offsets.0 as usize;
    let mut misaligned_bytes = file_bytes.clone();
    misaligned_bytes.insert(absolute_start, 0);
    let misaligned_program = vec![Op::Input {
        dtype: DType::FLOAT32,
        shape: vec![Extent::Static(2)],
        name: Some("norm.f32".into()),
    }];
    let misaligned = bind_safetensors_program_leaves(
        &misaligned_manifest,
        &misaligned_bytes,
        data_start,
        &misaligned_program,
        &BindingProfile::default(),
        &[],
    )
    .expect("misaligned f32 decodes bytewise");
    assert_eq!(misaligned.owned()[0].1, [1.0, 2.0]);
    let misaligned_f32_decode_count = 1;

    let mut wrong_axes = manifest.clone();
    wrong_axes
        .tensors
        .iter_mut()
        .find(|entry| entry.name == "matrix.f32")
        .expect("fixture has f32 matrix")
        .shape = vec![1, 6];
    let (f32_program, _) = program_for("matrix.f32", "norm.f32", "alias.f32", "token.f32");
    assert!(matches!(
        bind_safetensors_program_leaves(
            &wrong_axes,
            &file_bytes,
            data_start,
            &f32_program,
            &BindingProfile {
                aliases: aliases.clone(),
                ..BindingProfile::default()
            },
            &["token.f32".into()],
        ),
        Err(InteropError::SafetensorsAxesMismatch { .. })
    ));
    let wrong_axes_rejected = 1;

    let mut wrong_byte_length = manifest.clone();
    wrong_byte_length
        .tensors
        .iter_mut()
        .find(|entry| entry.name == "matrix.f32")
        .expect("fixture has f32 matrix")
        .data_offsets
        .1 -= 2;
    assert!(matches!(
        bind_safetensors_program_leaves(
            &wrong_byte_length,
            &file_bytes,
            data_start,
            &f32_program,
            &BindingProfile {
                aliases: aliases.clone(),
                ..BindingProfile::default()
            },
            &["token.f32".into()],
        ),
        Err(InteropError::SafetensorsByteLengthMismatch { .. })
    ));
    let byte_length_rejected = 1;

    let mut missing = manifest.clone();
    missing.tensors.retain(|entry| entry.name != "matrix.f32");
    assert!(matches!(
        bind_safetensors_program_leaves(
            &missing,
            &file_bytes,
            data_start,
            &f32_program,
            &BindingProfile {
                aliases: aliases.clone(),
                ..BindingProfile::default()
            },
            &["token.f32".into()],
        ),
        Err(InteropError::UnknownTensor { .. })
    ));
    let missing_rejected = 1;
    assert!(matches!(
        bind_safetensors_program_leaves(
            &manifest,
            &file_bytes,
            u64::MAX,
            &f32_program,
            &BindingProfile {
                aliases: aliases.clone(),
                ..BindingProfile::default()
            },
            &["token.f32".into()],
        ),
        Err(InteropError::SafetensorsDataRangeInvalid { .. })
    ));
    let offset_overflow_rejected = 1;

    let mut overlapping_ranges = manifest.clone();
    let norm_range = overlapping_ranges
        .tensor("norm.f32")
        .expect("fixture has f32 norm")
        .data_offsets;
    let matrix_entry = overlapping_ranges
        .tensors
        .iter_mut()
        .find(|entry| entry.name == "matrix.f32")
        .expect("fixture has f32 matrix");
    matrix_entry.data_offsets = (norm_range.0, norm_range.0 + 24);
    assert!(matches!(
        bind_safetensors_program_leaves(
            &overlapping_ranges,
            &file_bytes,
            data_start,
            &f32_program,
            &BindingProfile {
                aliases: aliases.clone(),
                ..BindingProfile::default()
            },
            &["token.f32".into()],
        ),
        Err(InteropError::SafetensorsDataRangesOverlap { .. })
    ));
    let overlapping_ranges_rejected = 1;

    let extra_entry = manifest
        .tensor("extra.weight")
        .expect("fixture has extra tensor");
    let truncated_end = data_start as usize + extra_entry.data_offsets.1 as usize - 1;
    assert!(matches!(
        bind_safetensors_program_leaves(
            &manifest,
            &file_bytes[..truncated_end],
            data_start,
            &f32_program,
            &BindingProfile {
                extra: vec!["extra.weight".into()],
                ..BindingProfile::default()
            },
            &["token.f32".into()],
        ),
        Err(InteropError::SafetensorsDataRangeInvalid { .. })
    ));
    let short_payload_rejected = 1;

    let alias_program = vec![Op::Input {
        dtype: DType::FLOAT32,
        shape: vec![Extent::Static(2)],
        name: Some("part.weight".into()),
    }];
    let symbolic_program = vec![Op::Input {
        dtype: DType::FLOAT32,
        shape: vec![Extent::Symbolic(1)],
        name: Some("norm.f32".into()),
    }];
    assert!(matches!(
        bind_safetensors_program_leaves(
            &manifest,
            &file_bytes,
            data_start,
            &symbolic_program,
            &BindingProfile::default(),
            &[],
        ),
        Err(InteropError::SafetensorsBindingUnsupported { .. })
    ));
    let symbolic_weight_rejected = 1;

    for (alias_index, alias) in [
        TensorAlias::Part {
            leaf: "part.weight".into(),
            from: "norm.f32".into(),
            part: 0,
            of: 1,
        },
        TensorAlias::Join {
            leaf: "part.weight".into(),
            from: vec!["norm.f32".into()],
        },
    ]
    .into_iter()
    .enumerate()
    {
        assert!(matches!(
            bind_safetensors_program_leaves(
                &manifest,
                &file_bytes,
                data_start,
                &alias_program,
                &BindingProfile {
                    aliases: vec![alias],
                    ..BindingProfile::default()
                },
                &[],
            ),
            Err(InteropError::SafetensorsBindingUnsupported { .. })
        ));
        if alias_index == 0 {
            part_alias_rejected = 1;
        } else {
            join_alias_rejected = 1;
        }
    }

    let mut gathered_program = vec![Op::Input {
        dtype: DType::INT32,
        shape: vec![Extent::Static(1)],
        name: Some("expert.index".into()),
    }];
    let gather_index = NodeId(0);
    let gathered_leaf = append(
        &mut gathered_program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(2), Extent::Static(3), Extent::Static(2)],
            name: Some("gather.weight".into()),
        },
    );
    let gather_activations = append(
        &mut gathered_program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(1), Extent::Static(3)],
            name: Some("activation".into()),
        },
    );
    append(
        &mut gathered_program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (
                    gather_activations,
                    IndexMap::Affine(map::projection(3, &[0, 2])),
                ),
                (
                    gathered_leaf,
                    IndexMap::Computed {
                        indices: gather_index,
                        index_map: map::projection(3, &[0]),
                        base: map::projection(3, &[2, 1]),
                        gathered_dim: 0,
                    },
                ),
            ],
            name: None,
        },
    );
    let mut gathered_manifest = manifest.clone();
    gathered_manifest
        .tensors
        .push(proxima_safetensors::TensorEntry {
            name: "gather.weight".into(),
            dtype: DType::FLOAT32,
            shape: vec![2, 3, 2],
            data_offsets: (0, 48),
        });
    assert!(matches!(
        bind_safetensors_program_leaves(
            &gathered_manifest,
            &file_bytes,
            data_start,
            &gathered_program,
            &BindingProfile::default(),
            &["expert.index".into(), "activation".into()],
        ),
        Err(InteropError::SafetensorsBindingUnsupported { .. })
    ));
    let gathered_role_rejected = 1;

    println!(
        "dtypes=3 bound_leaves={bound_leaf_count} runtime_inputs_skipped={runtime_input_count} packed_matmuls={packed_matmul_count} native_decodes={native_decode_count} f32_transposes={f32_transpose_count} alias_matches={alias_match_count} evaluated_matmuls={evaluated_matmul_count} decode_rules={decode_rule_count} extra_weights={extra_weight_count} overlapping_ranges_rejected={overlapping_ranges_rejected} native_f32_borrows={f32_borrow_count} misaligned_f32_decodes={misaligned_f32_decode_count} missing_weights_rejected={missing_rejected} wrong_axes_rejected={wrong_axes_rejected} dtype_byte_length_rejected={byte_length_rejected} short_payload_rejected={short_payload_rejected} offset_overflow_rejected={offset_overflow_rejected} part_alias_rejected={part_alias_rejected} join_alias_rejected={join_alias_rejected} gathered_role_rejected={gathered_role_rejected} symbolic_weight_rejected={symbolic_weight_rejected}"
    );
}

/// The ggml type the fixture's own 2-byte matrix is stored as, read off the
/// manifest so the rule target is the stored format of a real tensor, not a
/// spelled one.
fn fixture_two_byte_target(manifest: &Manifest) -> GgmlType {
    let stored = narrow_fixture_tensor(manifest, &[2, 3]).dtype;
    GgmlType::from_numeric_format(&stored.numeric_format())
        .expect("the two-byte fixture dtype has a ggml wire type")
}

fn input_names(program: &[Op]) -> Vec<&str> {
    program
        .iter()
        .filter_map(|op| match op {
            Op::Input {
                name: Some(name), ..
            } => Some(name.as_str()),
            _ => None,
        })
        .collect()
}

fn alias_binding(from: &str) -> BindingProfile {
    BindingProfile {
        aliases: vec![TensorAlias::Rename {
            leaf: "alias".into(),
            from: from.into(),
        }],
        ..BindingProfile::default()
    }
}

#[test]
fn a_wide_weight_leaf_recodes_to_half_width_under_a_suffix_rule() {
    let file_bytes = safetensors_fixture();
    let manifest = proxima_safetensors::parse_complete(&file_bytes).expect("parses fixture");
    let data_start = header_data_start(&file_bytes);
    let matrix = wide_fixture_tensor(&manifest, &[2, 3]);
    let norm = wide_fixture_tensor(&manifest, &[2]);
    let (mut program, _) = program_for(matrix, norm, "alias", "tokens");
    let rules = [WeightPrecisionRule {
        pattern: NamePattern::Suffix(matrix),
        target: fixture_two_byte_target(&manifest),
    }];

    let weights = proxima_model_interop::bind_safetensors_program_leaves(
        &manifest,
        &file_bytes,
        data_start,
        &program,
        &alias_binding(norm),
        &["tokens".into()],
        &rules,
    )
    .expect("a suffix rule over a stored weight leaf binds");

    assert_eq!(weights.packed_owned().len(), 1);
    let (name, bytes, _) = &weights.packed_owned()[0];
    assert!(name.starts_with(matrix) && name.contains('@'), "{name}");
    assert_eq!(bytes.len(), 2 * 2 * 3);
    let oracle = narrow_fixture_tensor(&manifest, &[2, 3]);
    assert_eq!(
        bytes.as_slice(),
        fixture_tensor_bytes(&file_bytes, data_start, oracle),
        "stored [out, in] order, packed, not transposed"
    );
    assert!(weights.owned().iter().all(|(owned, _)| !owned.starts_with(matrix)));
    let recoded = name.clone();
    retarget_recoded_leaves(&mut program, &weights);
    let leaves = input_names(&program);
    assert!(leaves.contains(&recoded.as_str()));
    assert!(!leaves.contains(&matrix));
}

/// The fixture's tensor of `shape` stored at two bytes per element: the oracle
/// a half-width recode of the wide tensor must equal byte for byte.
fn narrow_fixture_tensor<'manifest>(
    manifest: &'manifest Manifest,
    shape: &[u64],
) -> &'manifest proxima_safetensors::TensorEntry {
    manifest
        .tensors
        .iter()
        .find(|entry| entry.shape == shape && entry.dtype.size_bytes() == 2)
        .expect("the fixture stores a narrow tensor of that shape")
}

/// The fixture's tensor of `shape` stored at four bytes per element, so a test
/// names it from the manifest rather than spelling the stored format.
fn wide_fixture_tensor<'manifest>(manifest: &'manifest Manifest, shape: &[u64]) -> &'manifest str {
    manifest
        .tensors
        .iter()
        .find(|entry| entry.shape == shape && entry.dtype.size_bytes() == 4)
        .map(|entry| entry.name.as_str())
        .expect("the fixture stores a wide tensor of that shape")
}

/// `program_for` with the weight leaf declared in the stored `[out, in]` order
/// and read `oi`: a contracted leaf the binder gives the native role.
fn stored_order_program(weight: &str, token: &str) -> Vec<Op> {
    let (mut program, _) = program_for(weight, "scale", "bias", token);
    for op in &mut program {
        match op {
            Op::Input { shape, name: Some(name), .. } if name.as_str() == weight => {
                *shape = vec![Extent::Static(2), Extent::Static(3)];
            }
            Op::Elementwise { operands, .. } => {
                operands[1].1 = IndexMap::Affine(map::projection(3, &[1, 2]));
            }
            _ => {}
        }
    }
    program
}

#[test]
fn a_stored_order_contracted_leaf_recodes_under_a_suffix_rule() {
    let file_bytes = safetensors_fixture();
    let manifest = proxima_safetensors::parse_complete(&file_bytes).expect("parses fixture");
    let data_start = header_data_start(&file_bytes);
    let matrix = wide_fixture_tensor(&manifest, &[2, 3]);
    let program = stored_order_program(matrix, "tokens");
    let rules = [WeightPrecisionRule {
        pattern: NamePattern::Suffix(matrix),
        target: fixture_two_byte_target(&manifest),
    }];
    let binding = BindingProfile {
        aliases: ["scale", "bias"]
            .map(|leaf| TensorAlias::Rename {
                leaf: leaf.into(),
                from: wide_fixture_tensor(&manifest, &[2]).into(),
            })
            .into(),
        ..BindingProfile::default()
    };

    let weights = proxima_model_interop::bind_safetensors_program_leaves(
        &manifest,
        &file_bytes,
        data_start,
        &program,
        &binding,
        &["tokens".into()],
        &rules,
    )
    .expect("a suffix rule over a stored-order leaf binds");

    assert_eq!(weights.packed_owned().len(), 1);
    let (name, bytes, _) = &weights.packed_owned()[0];
    assert!(name.starts_with(matrix) && name.contains('@'), "{name}");
    let oracle = narrow_fixture_tensor(&manifest, &[2, 3]);
    assert_eq!(
        bytes.as_slice(),
        fixture_tensor_bytes(&file_bytes, data_start, oracle),
        "stored order, packed, not transposed"
    );
}

/// Binds the stored-order fixture program under a suffix rule through `model`,
/// so a persisted file and its source are read by the same program and rules.
fn bind_stored_order<'file, 'store>(
    model: &'store ModelWeights,
    (manifest, file_bytes, data_start): (&Manifest, &'file [u8], u64),
    (matrix, norm, target): (&str, &str, GgmlType),
) -> LeasedWeights<'file, 'store> {
    let rules = [WeightPrecisionRule {
        pattern: NamePattern::Suffix(matrix),
        target,
    }];
    let binding = BindingProfile {
        aliases: ["scale", "bias"]
            .map(|leaf| TensorAlias::Rename {
                leaf: leaf.into(),
                from: norm.into(),
            })
            .into(),
        ..BindingProfile::default()
    };
    model
        .bind(
            manifest,
            file_bytes,
            data_start,
            &stored_order_program(matrix, "tokens"),
            &binding,
            &["tokens".into()],
            &rules,
        )
        .expect("the stored-order program binds under the suffix rule")
}

#[test]
fn a_persisted_checkpoint_rebinds_with_no_recode_and_the_same_narrow_bytes() {
    let file_bytes = safetensors_fixture();
    let manifest = proxima_safetensors::parse_complete(&file_bytes).expect("parses fixture");
    let data_start = header_data_start(&file_bytes);
    let matrix = wide_fixture_tensor(&manifest, &[2, 3]);
    let identity = (
        matrix,
        wide_fixture_tensor(&manifest, &[2]),
        fixture_two_byte_target(&manifest),
    );
    let model = ModelWeights::new();
    let first_bytes = {
        let first = bind_stored_order(&model, (&manifest, &file_bytes, data_start), identity);
        let [(name, bytes, _)] = first.packed_owned() else {
            panic!("the rule matches exactly one leaf, so one buffer is recoded");
        };
        assert!(name.starts_with(matrix) && name.contains('@'), "{name}");
        bytes.clone()
    };

    let mut persisted = Vec::new();
    let substituted = model
        .persist(&manifest, &file_bytes, data_start, &mut persisted)
        .expect("a held recode persists");

    assert_eq!(substituted, 1, "one rule-matched leaf was recoded");
    let reopened = proxima_safetensors::parse_complete(&persisted).expect("parses persisted file");
    let entry = reopened.tensor(matrix).expect("the stored name is kept");
    assert_eq!(entry.shape, [2, 3], "the stored shape is kept");
    assert_eq!(entry.dtype, narrow_fixture_tensor(&manifest, &[2, 3]).dtype);
    assert_eq!(reopened.tensors.len(), manifest.tensors.len());
    let carried = manifest.tensor("extra.weight").expect("fixture has extra");
    assert_eq!(
        fixture_tensor_bytes(
            &persisted,
            header_data_start(&persisted),
            reopened.tensor("extra.weight").expect("kept")
        ),
        fixture_tensor_bytes(&file_bytes, data_start, carried),
        "an unrecoded tensor is carried byte for byte"
    );

    let second_model = ModelWeights::new();
    let second_start = header_data_start(&persisted);
    let second = bind_stored_order(
        &second_model,
        (&reopened, &persisted, second_start),
        identity,
    );
    assert!(second.packed_owned().is_empty(), "no leaf is recoded again");
    assert!(second.owned().iter().all(|(name, _)| !name.contains('@')));
    let (_, QuantizedBlock::Packed { bytes, .. }) = second
        .packed()
        .iter()
        .find(|(name, _)| name == matrix)
        .expect("the rule-matched leaf binds under its stored name")
    else {
        panic!("the rule-matched leaf binds as a packed block");
    };
    let mapping = persisted.as_ptr_range();
    assert!(
        mapping.contains(&bytes.as_ptr()),
        "the block borrows the second mapping instead of owning a copy"
    );
    assert_eq!(
        *bytes,
        first_bytes.as_slice(),
        "bit for bit the first bind's bytes"
    );
}

#[test]
fn a_model_reports_a_recode_to_publish_only_after_a_rule_recoded_a_leaf() {
    let file_bytes = safetensors_fixture();
    let manifest = proxima_safetensors::parse_complete(&file_bytes).expect("parses fixture");
    let data_start = header_data_start(&file_bytes);
    let identity = (
        wide_fixture_tensor(&manifest, &[2, 3]),
        wide_fixture_tensor(&manifest, &[2]),
        fixture_two_byte_target(&manifest),
    );
    let model = ModelWeights::new();
    assert!(
        !model.has_recoded(&manifest),
        "a model that bound nothing has nothing worth a store entry"
    );

    drop(bind_stored_order(&model, (&manifest, &file_bytes, data_start), identity));

    assert!(
        model.has_recoded(&manifest),
        "a model whose rule recoded a leaf has a store entry worth writing"
    );
}

#[test]
fn persisting_a_model_that_recoded_nothing_substitutes_nothing() {
    let file_bytes = safetensors_fixture();
    let manifest = proxima_safetensors::parse_complete(&file_bytes).expect("parses fixture");
    let mut persisted = Vec::new();

    let substituted = ModelWeights::new()
        .persist(
            &manifest,
            &file_bytes,
            header_data_start(&file_bytes),
            &mut persisted,
        )
        .expect("an empty model persists a plain copy");

    assert_eq!(substituted, 0);
    let reopened = proxima_safetensors::parse_complete(&persisted).expect("parses persisted file");
    assert_eq!(reopened.tensors.len(), manifest.tensors.len());
}

#[test]
fn persist_aligned_starts_every_large_tensor_on_the_alignment_and_keeps_every_byte() {
    const ALIGNMENT: usize = 16;
    let file_bytes = safetensors_fixture();
    let manifest = proxima_safetensors::parse_complete(&file_bytes).expect("parses fixture");
    let source_start = header_data_start(&file_bytes) as usize;
    let mut persisted = Vec::new();

    ModelWeights::new()
        .persist_aligned(
            &manifest,
            &file_bytes,
            header_data_start(&file_bytes),
            ALIGNMENT,
            &mut persisted,
        )
        .expect("an empty model persists an aligned copy");

    let stored = proxima_safetensors::parse_complete(&persisted).expect("parses persisted file");
    let stored_start = header_data_start(&persisted) as usize;
    assert_eq!(stored_start % ALIGNMENT, 0, "the data region is aligned");
    let originals: Vec<_> = stored
        .tensors
        .iter()
        .filter(|entry| !proxima_safetensors::is_filler_name(&entry.name))
        .collect();
    assert_eq!(originals.len(), manifest.tensors.len(), "every tensor survives");
    for entry in originals {
        let source = manifest.tensor(&entry.name).expect("the name is kept");
        let from = |bytes: &[u8], start: usize, range: (u64, u64)| {
            bytes[start + range.0 as usize..start + range.1 as usize].to_vec()
        };
        assert_eq!(
            from(&persisted, stored_start, entry.data_offsets),
            from(&file_bytes, source_start, source.data_offsets),
            "tensor {} bytes",
            entry.name
        );
        let length = entry.data_offsets.1 - entry.data_offsets.0;
        if length >= ALIGNMENT as u64 {
            assert_eq!(
                (stored_start + entry.data_offsets.0 as usize) % ALIGNMENT,
                0,
                "tensor {} offset",
                entry.name
            );
        }
    }
}

#[test]
fn a_norm_scale_named_weight_is_never_recoded() {
    let file_bytes = safetensors_fixture();
    let manifest = proxima_safetensors::parse_complete(&file_bytes).expect("parses fixture");
    let data_start = header_data_start(&file_bytes);
    let matrix = wide_fixture_tensor(&manifest, &[2, 3]);
    let scale = wide_fixture_tensor(&manifest, &[2]);
    let (program, _) = program_for(matrix, "layer.norm.weight", "bias", "tokens");
    let binding = BindingProfile {
        aliases: ["layer.norm.weight", "bias"]
            .map(|leaf| TensorAlias::Rename {
                leaf: leaf.into(),
                from: scale.into(),
            })
            .into(),
        ..BindingProfile::default()
    };
    let rules = [WeightPrecisionRule {
        pattern: NamePattern::Suffix(".weight"),
        target: fixture_two_byte_target(&manifest),
    }];

    let weights = proxima_model_interop::bind_safetensors_program_leaves(
        &manifest,
        &file_bytes,
        data_start,
        &program,
        &binding,
        &["tokens".into()],
        &rules,
    )
    .expect("a scale leaf binds under a rule that names its suffix");

    assert_eq!(weights.recoded_weights().len(), 0);
    assert!(weights.names().contains("layer.norm.weight"));
}

#[test]
fn precision_rule_leaves_norm_and_bias_leaves_untouched() {
    let file_bytes = safetensors_fixture();
    let manifest = proxima_safetensors::parse_complete(&file_bytes).expect("parses fixture");
    let data_start = header_data_start(&file_bytes);
    let matrix = wide_fixture_tensor(&manifest, &[2, 3]);
    let norm = wide_fixture_tensor(&manifest, &[2]);
    let (program, _) = program_for(matrix, norm, "alias", "tokens");
    let scales_only: Vec<Op> = program
        .into_iter()
        .filter(|op| {
            matches!(op, Op::Input { name: Some(name), .. } if name == norm || name == "alias")
        })
        .collect();
    let rules = [WeightPrecisionRule {
        pattern: NamePattern::Prefix(""),
        target: fixture_two_byte_target(&manifest),
    }];

    let weights = proxima_model_interop::bind_safetensors_program_leaves(
        &manifest,
        &file_bytes,
        data_start,
        &scales_only,
        &alias_binding(norm),
        &[],
        &rules,
    )
    .expect("scale leaves bind under a catch-all rule");

    assert_eq!(weights.packed_owned().len(), 0);
    let mut bound_names: Vec<&str> = weights
        .owned()
        .iter()
        .map(|(name, _)| name.as_str())
        .chain(weights.packed().iter().map(|(name, _)| name.as_str()))
        .collect();
    bound_names.sort_unstable();
    let mut expected = ["alias", norm];
    expected.sort_unstable();
    assert_eq!(bound_names, expected);
}

#[test]
fn an_empty_rule_set_binds_bit_for_bit_as_before() {
    let file_bytes = safetensors_fixture();
    let manifest = proxima_safetensors::parse_complete(&file_bytes).expect("parses fixture");
    let data_start = header_data_start(&file_bytes);
    let matrix = wide_fixture_tensor(&manifest, &[2, 3]);
    let norm = wide_fixture_tensor(&manifest, &[2]);
    let (program, _) = program_for(matrix, norm, "alias", "tokens");

    let weights = proxima_model_interop::bind_safetensors_program_leaves(
        &manifest,
        &file_bytes,
        data_start,
        &program,
        &alias_binding(norm),
        &["tokens".into()],
        &[],
    )
    .expect("an unruled bind is the pre-rule bind");

    let entry = manifest.tensor(matrix).expect("fixture matrix");
    let stored = fixture_tensor_bytes(&file_bytes, data_start, entry);
    let width = stored.len() / 6;
    let transposed_from_file: Vec<u8> = [0usize, 3, 1, 4, 2, 5]
        .into_iter()
        .flat_map(|index| stored[index * width..(index + 1) * width].iter().copied())
        .collect();
    let (_, bound) = weights
        .owned()
        .iter()
        .find(|(name, _)| name == matrix)
        .expect("unruled matmul leaf stays an owned transposed copy");
    let bound_bytes: Vec<u8> = bound.iter().flat_map(|value| value.to_le_bytes()).collect();
    assert_eq!(bound_bytes, transposed_from_file);
    assert_eq!(weights.packed_owned().len(), 0);
}
