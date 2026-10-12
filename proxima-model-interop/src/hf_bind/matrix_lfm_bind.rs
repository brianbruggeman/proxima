use super::*;

use proxima_primitives::pipe::sans_io::Outcome;
use proxima_safetensors::SafetensorsParser;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::op::{Extent, Keep, Reduce, ReduceInit, ScalarOp, append};
use proxima_tensor::spec::LayerKind;

use crate::hf_config::{architecture_from_hf_config, lfm_layer_kinds_from_manifest};
use crate::profiles::{TensorAlias, binding_profile};

type LeafSpec = (&'static str, &'static str, &'static [u64]);

const COMMON: [LeafSpec; 5] = [
    ("attn_norm.weight", "operator_norm.weight", &[2048]),
    ("ffn_norm.weight", "ffn_norm.weight", &[2048]),
    ("ffn_gate.weight", "feed_forward.w1.weight", &[8192, 2048]),
    ("ffn_down.weight", "feed_forward.w2.weight", &[2048, 8192]),
    ("ffn_up.weight", "feed_forward.w3.weight", &[8192, 2048]),
];

const SHORTCONV: [LeafSpec; 3] = [
    ("shortconv.conv.weight", "conv.conv.weight", &[2048, 1, 3]),
    (
        "shortconv.in_proj.weight",
        "conv.in_proj.weight",
        &[6144, 2048],
    ),
    (
        "shortconv.out_proj.weight",
        "conv.out_proj.weight",
        &[2048, 2048],
    ),
];

const ATTENTION: [LeafSpec; 6] = [
    ("attn_q.weight", "self_attn.q_proj.weight", &[2048, 2048]),
    ("attn_k.weight", "self_attn.k_proj.weight", &[512, 2048]),
    ("attn_v.weight", "self_attn.v_proj.weight", &[512, 2048]),
    (
        "attn_output.weight",
        "self_attn.out_proj.weight",
        &[2048, 2048],
    ),
    ("attn_q_norm.weight", "self_attn.q_layernorm.weight", &[64]),
    ("attn_k_norm.weight", "self_attn.k_layernorm.weight", &[64]),
];

fn pinned_config_and_manifest() -> (crate::HfConfig, Manifest) {
    let config_bytes = include_bytes!(
        "../../../proxima-tensor/specs/small-model-architecture-matrix/fixtures/semantic-pair-configs/lfm_text.json"
    );
    let header_bytes = include_bytes!(
        "../../../proxima-tensor/specs/small-model-architecture-matrix/fixtures/lfm_text.safetensors.header"
    );
    let config = crate::parse_hf_config(config_bytes).expect("pinned config parses");
    let mut parser = SafetensorsParser::new();
    parser.feed(header_bytes);
    let manifest = match parser.poll().expect("pinned header parses") {
        Outcome::Event(manifest) => manifest.clone(),
        Outcome::NeedMore => panic!("complete pinned header must produce a manifest"),
    };
    (config, manifest)
}

fn compact_selected_payload(manifest: &mut Manifest) -> Vec<u8> {
    manifest.tensors.retain(|entry| {
        entry.name.starts_with("model.layers.0.") || entry.name.starts_with("model.layers.2.")
    });
    let mut file_bytes = Vec::new();
    for entry in &mut manifest.tensors {
        let start = u64::try_from(file_bytes.len()).expect("fixture offset fits u64");
        let byte_len = usize::try_from(entry.byte_len()).expect("fixture tensor fits usize");
        file_bytes.resize(file_bytes.len() + byte_len, 0xA5);
        let end = u64::try_from(file_bytes.len()).expect("fixture end fits u64");
        entry.data_offsets = (start, end);
    }
    file_bytes
}

fn expected_leaves() -> Vec<(String, String, &'static [u64])> {
    let mut leaves = Vec::new();
    for (layer, mixer) in [(0, &SHORTCONV[..]), (2, &ATTENTION[..])] {
        for &(leaf_suffix, source_suffix, shape) in COMMON.iter().chain(mixer.iter()) {
            leaves.push((
                format!("blk.{layer}.{leaf_suffix}"),
                format!("model.layers.{layer}.{source_suffix}"),
                shape,
            ));
        }
    }
    leaves
}

#[test]
fn architecture_matrix_lfm_global_leaves() {
    let (_, pinned_manifest) = pinned_config_and_manifest();
    let pinned_sources = [
        ("model.embed_tokens.weight", &[65536, 2048][..]),
        ("model.embedding_norm.weight", &[2048][..]),
    ];
    for (name, shape) in pinned_sources {
        let entry = pinned_manifest
            .tensor(name)
            .expect("pinned LFM header contains each graph-global source");
        assert_eq!(entry.dtype, DType::BFloat16);
        assert_eq!(entry.shape, shape);
    }

    let table_values = [1.0f32, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
    let table_bytes = table_values
        .into_iter()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect::<Vec<_>>();
    let norm_bytes = [1.0f32, 1.0]
        .into_iter()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect::<Vec<_>>();
    let mut file_bytes = table_bytes;
    file_bytes.extend_from_slice(&norm_bytes);
    let manifest = Manifest {
        tensors: vec![
            proxima_safetensors::TensorEntry {
                name: "model.embed_tokens.weight".into(),
                dtype: DType::BFloat16,
                shape: vec![4, 2],
                data_offsets: (0, 16),
            },
            proxima_safetensors::TensorEntry {
                name: "model.embedding_norm.weight".into(),
                dtype: DType::BFloat16,
                shape: vec![2],
                data_offsets: (16, 20),
            },
        ],
        ..Manifest::default()
    };

    let mut program = Vec::new();
    let token_ids = append(
        &mut program,
        Op::Input {
            dtype: DType::Int32,
            shape: vec![Extent::Static(1)],
            name: Some("ids".into()),
        },
    );
    let embedding = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4), Extent::Static(2)],
            name: Some("token_embd.weight".into()),
        },
    );
    let hidden = proxima_tensor::spec::embedding_lookup(&mut program, embedding, token_ids);
    let output_norm = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(2)],
            name: Some("output_norm.weight".into()),
        },
    );
    let inverse_width = append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: Vec::new(),
            value: 0.5,
        },
    );
    let epsilon = append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: Vec::new(),
            value: 0.0,
        },
    );
    let normed =
        proxima_tensor::spec::rmsnorm(&mut program, hidden, output_norm, inverse_width, epsilon)
            .expect("global output norm composes from existing operations");
    let output_weight = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(2), Extent::Static(4)],
            name: Some("output.weight".into()),
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (normed, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (output_weight, IndexMap::Affine(map::projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    let logits = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    let binding = binding_profile("lfm2").expect("LFM binding profile parses");
    let embedding_source = super::safetensors_source(
        &manifest,
        &binding,
        "token_embd.weight",
        &[Extent::Static(4), Extent::Static(2)],
    )
    .expect("embedding leaf resolves to the pinned source name");
    let output_source = super::safetensors_source(
        &manifest,
        &binding,
        "output.weight",
        &[Extent::Static(2), Extent::Static(4)],
    )
    .expect("tied head resolves to the same pinned source name");
    assert_eq!(embedding_source.name, "model.embed_tokens.weight");
    assert_eq!(output_source.name, "model.embed_tokens.weight");
    assert!(core::ptr::eq(embedding_source, output_source));
    let weights = bind_safetensors_program_leaves(
        &manifest,
        &file_bytes,
        0,
        &program,
        &binding,
        &["ids".into()],
    )
    .expect("global graph leaves resolve through generic profile binding");

    assert_eq!(weights.owned().len(), 2);
    assert!(weights.owned().iter().any(|(name, values)| {
        name == "token_embd.weight" && values == &[1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    }));
    assert!(
        weights
            .owned()
            .iter()
            .any(|(name, values)| { name == "output_norm.weight" && values == &[1.0, 1.0] })
    );
    assert_eq!(weights.packed().len(), 1);
    let (output_source, output_block) = &weights.packed()[0];
    assert_eq!(output_source, "output.weight");
    let QuantizedBlock::Packed { bytes, codec } = output_block else {
        panic!("tied BF16 output matrix remains packed");
    };
    assert_eq!(*codec, Codec::BFloat16);
    assert_eq!(*bytes, &file_bytes[..16]);
    assert!(core::ptr::eq(bytes.as_ptr(), file_bytes.as_ptr()));

    let token_values = [0i32];
    let mut named = vec![("ids", QuantizedBlock::Int32(&token_values))];
    named.extend(
        weights
            .owned()
            .iter()
            .map(|(name, values)| (name.as_str(), QuantizedBlock::Float32(values))),
    );
    named.extend(
        weights
            .packed()
            .iter()
            .map(|(name, block)| (name.as_str(), *block)),
    );
    let evaluated = proxima_tensor::cpu::evaluate_quantized_named(&program, &[], &named, &[logits])
        .expect("generic evaluator consumes bound embedding, norm, and tied head");
    let (values, axes) = evaluated.get(logits).expect("logits root is retained");
    assert_eq!(axes, [1, 4]);
    assert_eq!(values, [2.0, 1.0, 2.0, 3.0]);

    let missing_norm = Manifest {
        tensors: manifest
            .tensors
            .iter()
            .filter(|entry| entry.name != "model.embedding_norm.weight")
            .cloned()
            .collect(),
        ..Manifest::default()
    };
    assert!(matches!(
        bind_safetensors_program_leaves(
            &missing_norm,
            &file_bytes,
            0,
            &program,
            &binding,
            &["ids".into()],
        ),
        Err(InteropError::UnknownTensor { .. })
    ));
    println!(
        "pinned_sources=2 bound_leaves=3 tied_source_uses=2 evaluated_logits=4 missing_source_rejected=1"
    );
}

fn append_matrix_leaf(
    program: &mut Vec<Op>,
    leaf: &str,
    source_axes: &[u64],
    runtime_inputs: &mut Vec<String>,
) {
    let output_width = u32::try_from(source_axes[0]).expect("output axis fits u32");
    let input_width = u32::try_from(source_axes[1]).expect("input axis fits u32");
    let runtime_name = format!("runtime.{leaf}");
    let activation = append(
        program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1), Extent::Static(input_width)],
            name: Some(runtime_name.clone()),
        },
    );
    runtime_inputs.push(runtime_name);
    let weight = append(
        program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(input_width), Extent::Static(output_width)],
            name: Some(leaf.into()),
        },
    );
    let product = append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (activation, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (weight, IndexMap::Affine(map::projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    append(
        program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
}

#[test]
fn architecture_matrix_lfm_layer_bind() {
    let (config, mut manifest) = pinned_config_and_manifest();
    let architecture = architecture_from_hf_config(&config)
        .expect("pinned config derives its effective architecture");
    assert_eq!(architecture.feed_forward, 8192);
    assert!(architecture.tied_embeddings);
    let kinds = lfm_layer_kinds_from_manifest(&config, &manifest)
        .expect("pinned header matches all scheduled layer kinds");
    assert_eq!(kinds.len(), 16);
    assert_eq!(
        kinds
            .iter()
            .filter(|kind| **kind == LayerKind::ShortConv)
            .count(),
        10
    );
    assert_eq!(
        kinds
            .iter()
            .filter(|kind| **kind == LayerKind::Attention)
            .count(),
        6
    );

    let binding = binding_profile("lfm2").expect("pinned family binding profile parses");
    let profile_rows = binding
        .aliases
        .iter()
        .filter(|alias| {
            if let TensorAlias::Rename { leaf, from } = alias
                && leaf.starts_with("blk.")
            {
                assert!(
                    manifest.tensor(from).is_some(),
                    "pinned source {from} is missing"
                );
                true
            } else {
                false
            }
        })
        .count();
    assert_eq!(profile_rows, 146);

    let file_bytes = compact_selected_payload(&mut manifest);
    let expected = expected_leaves();
    assert_eq!(expected.len(), 19);
    let mut program = Vec::new();
    let mut runtime_inputs = Vec::new();
    let mut matrix_roles = 0;
    for (leaf, _, shape) in &expected {
        if shape.len() == 2 && !leaf.contains(".shortconv.") {
            append_matrix_leaf(&mut program, leaf, shape, &mut runtime_inputs);
            matrix_roles += 1;
        } else {
            append(
                &mut program,
                Op::Input {
                    dtype: DType::BFloat16,
                    shape: shape
                        .iter()
                        .map(|axis| {
                            Extent::Static(u32::try_from(*axis).expect("fixture axis fits u32"))
                        })
                        .collect(),
                    name: Some(leaf.clone()),
                },
            );
        }
    }
    assert_eq!(matrix_roles, 10);
    let views = bind_safetensors_program_views(
        &manifest,
        &file_bytes,
        0,
        &program,
        &binding,
        &runtime_inputs,
    )
    .expect("program leaves resolve through the profile");
    assert_eq!(views.len(), expected.len());

    for ((leaf, source, shape), view) in expected.iter().zip(&views) {
        let entry = manifest
            .tensor(source)
            .expect("source exists in selected header");
        let start = usize::try_from(entry.data_offsets.0).expect("fixture start fits usize");
        let end = usize::try_from(entry.data_offsets.1).expect("fixture end fits usize");
        assert_eq!(&view.leaf, leaf);
        assert_eq!(view.name, source);
        assert_eq!(view.shape, *shape);
        assert_eq!(view.dtype, DType::BFloat16);
        assert_eq!(view.bytes.len(), end - start);
        assert_eq!(view.bytes.as_ptr(), file_bytes[start..end].as_ptr());
        if shape.len() == 2 && !leaf.contains(".shortconv.") {
            assert_eq!(view.role, Role::InOut);
            assert_eq!(view.program_axes.as_slice(), &[shape[1], shape[0]]);
        } else {
            assert_eq!(view.role, Role::Native);
            assert_eq!(view.program_axes.as_slice(), *shape);
        }
    }

    let mut wrong_conv_shape = manifest.clone();
    wrong_conv_shape
        .tensors
        .iter_mut()
        .find(|entry| entry.name == "model.layers.0.conv.conv.weight")
        .expect("convolution source exists")
        .shape[2] = 2;
    assert!(matches!(
        bind_safetensors_program_views(
            &wrong_conv_shape,
            &file_bytes,
            0,
            &program,
            &binding,
            &runtime_inputs,
        ),
        Err(InteropError::SafetensorsAxesMismatch { .. })
    ));

    let mut wrong_attention_shape = manifest.clone();
    wrong_attention_shape
        .tensors
        .iter_mut()
        .find(|entry| entry.name == "model.layers.2.self_attn.q_proj.weight")
        .expect("query source exists")
        .shape[0] = 2047;
    assert!(matches!(
        bind_safetensors_program_views(
            &wrong_attention_shape,
            &file_bytes,
            0,
            &program,
            &binding,
            &runtime_inputs,
        ),
        Err(InteropError::SafetensorsAxesMismatch { .. })
    ));

    println!(
        "layers=16 shortconv=10 attention=6 profile_matches=19 bad_shapes_rejected=2 model_binder_symbols=0"
    );
}
