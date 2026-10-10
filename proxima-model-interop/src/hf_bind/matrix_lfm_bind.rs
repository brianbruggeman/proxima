use super::*;

use proxima_primitives::pipe::sans_io::Outcome;
use proxima_safetensors::SafetensorsParser;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::op::{Keep, Reduce, ReduceInit, ScalarOp, append};
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
