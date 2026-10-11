use std::path::Path;

use arrayvec::ArrayVec;
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::quant::scaled_float::{BLOCK_BYTES, BLOCK_ELEMENTS, dequantize_block};
use proxima_gguf::quant::{f16, q4_0, q8_0};
use proxima_gguf::tensor::MAX_DIMS;
use proxima_gguf::{GgmlType, GgufModel, TensorPayload, parse_complete, write_complete};
use proxima_primitives::float_layout::FloatLayout;

use super::{convert_codec, policy_selects};
use crate::codec_policy_settings::{CodecPolicySettings, CodecPolicyWeights};
use crate::gguf_tensor_as_f32;
use crate::test_support::require_fixture;

const GRANITE_ENV: &str = "PROXIMA_ARCH_GRANITE_MOE_GGUF";
const GRANITE_DEFAULT: &str = "/Users/brianbruggeman/.ollama/models/blobs/\
     sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80";
const SAVING_FLOOR_BYTES: usize = 250_000_000;
const MAX_NORMAL_RATIO: f32 = 28.0;
const SUBNORMAL_HALF_SPACING: f32 = 0.03125;
const ROW_ELEMENTS: u64 = 1024;
const ROWS: u64 = 512;
const SELECTED_NAMES: [&str; 7] = [
    "blk.0.ffn_gate_exps.weight",
    "blk.0.ffn_up_exps.weight",
    "blk.0.ffn_down_exps.weight",
    "blk.0.attn_q.weight",
    "blk.0.attn_k.weight",
    "blk.0.attn_v.weight",
    "blk.0.attn_output.weight",
];
const EXCLUDED_NAMES: [&str; 6] = [
    "token_embd.weight",
    "output.weight",
    "output_norm.weight",
    "blk.0.attn_norm.weight",
    "blk.0.ffn_norm.weight",
    "blk.0.ffn_gate_inp.weight",
];

fn fp6_policy() -> CodecPolicySettings {
    CodecPolicySettings::builder()
        .weights(CodecPolicyWeights::Fp6E3M2)
        .build()
}

fn granite_path() -> String {
    let path = std::env::var(GRANITE_ENV).unwrap_or_else(|_| GRANITE_DEFAULT.into());
    require_fixture(&path, Some(GRANITE_ENV));
    path
}

fn map_granite() -> memmap2::Mmap {
    let file = std::fs::File::open(Path::new(&granite_path())).expect("open the granite blob");
    // SAFETY: the blob is a read-only host fixture no test writes or truncates.
    unsafe { memmap2::Mmap::map(&file) }.expect("map the granite blob")
}

fn granite_q8_0_weights() -> Vec<f32> {
    let mapping = map_granite();
    let parsed = parse_complete(&mapping).expect("granite parses as gguf");
    let tensor = parsed
        .tensors
        .iter()
        .find(|tensor| tensor.ggml_type == GgmlType::Q8_0 && tensor.dims.as_slice() == [1024, 512])
        .expect("granite holds a 1024x512 q8_0 tensor");
    gguf_tensor_as_f32(&parsed, &mapping, &tensor.name).expect("dequantize the granite tensor")
}

fn dims(values: &[u64]) -> ArrayVec<u64, MAX_DIMS> {
    values.iter().copied().collect()
}

fn encode_source(ggml_type: GgmlType, weights: &[f32]) -> Vec<u8> {
    match ggml_type {
        GgmlType::F32 => weights
            .iter()
            .flat_map(|weight| weight.to_le_bytes())
            .collect(),
        GgmlType::F16 => {
            let mut bytes = vec![0u8; weights.len() * 2];
            f16::quantize(weights, &mut bytes).expect("finite granite weights fit f16");
            bytes
        }
        GgmlType::Q8_0 => {
            let mut bytes = vec![0u8; weights.len() / 32 * 34];
            q8_0::quantize(weights, &mut bytes).expect("finite granite weights fit q8_0");
            bytes
        }
        GgmlType::Q4_0 => {
            let mut bytes = vec![0u8; weights.len() / 32 * 18];
            q4_0::quantize(weights, &mut bytes).expect("finite granite weights fit q4_0");
            bytes
        }
        other => panic!("no source encoder for {other:?}"),
    }
}

fn single_tensor_checkpoint(ggml_type: GgmlType, weights: &[f32]) -> Vec<u8> {
    let data = encode_source(ggml_type, weights);
    let model = GgufModel {
        version: 3,
        metadata: Vec::new(),
        tensors: vec![TensorPayload {
            name: "blk.0.attn_q.weight".into(),
            dims: dims(&[ROW_ELEMENTS, ROWS]),
            ggml_type,
            data: &data,
        }],
    };
    write_complete(&model).expect("write the source checkpoint")
}

fn half_spacing(ratio: f32) -> f32 {
    if ratio < 0.25 {
        SUBNORMAL_HALF_SPACING
    } else {
        2.0f32.powi(ratio.log2().floor() as i32 - 3)
    }
}

fn tensor_bytes<'a>(parsed: &ParsedGguf, file: &'a [u8], name: &str) -> &'a [u8] {
    let tensor = parsed
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .expect("tensor present in the checkpoint");
    let range = parsed
        .tensor_data_range(tensor, file.len() as u64)
        .expect("tensor range fits the file");
    &file[range.start as usize..range.end as usize]
}

fn assert_converted_within_block_bound(source_type: GgmlType) {
    let source_file = single_tensor_checkpoint(source_type, &granite_q8_0_weights());
    let source = parse_complete(&source_file).expect("source parses");
    let source_values =
        gguf_tensor_as_f32(&source, &source_file, "blk.0.attn_q.weight").expect("decode source");

    let (output_file, converted) =
        convert_codec(&source, &source_file, &fp6_policy()).expect("convert the checkpoint");

    let output = parse_complete(&output_file).expect("converted checkpoint parses");
    assert_eq!(converted, 1);
    assert_eq!(output.tensors[0].ggml_type, GgmlType::Fp6E3M2);
    let blocks = tensor_bytes(&output, &output_file, "blk.0.attn_q.weight");
    let checked = blocks
        .as_chunks::<BLOCK_BYTES>()
        .0
        .iter()
        .zip(source_values.as_chunks::<BLOCK_ELEMENTS>().0)
        .map(|(block, values)| assert_block_within_bound(block, values))
        .sum::<usize>();
    assert_eq!(
        checked,
        source_values.len(),
        "{source_type:?}: every element is bounded or exactly saturated"
    );
}

fn assert_element_within_bound(value: f32, restored: f32, scale: f32) {
    let ratio = if scale > 0.0 {
        value.abs() / scale
    } else {
        0.0
    };
    let bound = if ratio <= MAX_NORMAL_RATIO {
        scale * half_spacing(ratio) + 4.0 * f32::EPSILON * value.abs()
    } else {
        scale * (ratio - MAX_NORMAL_RATIO) + 4.0 * f32::EPSILON * value.abs()
    };
    assert!(
        (value - restored).abs() <= bound,
        "value {value} restored as {restored} exceeds bound {bound} at scale {scale}"
    );
}

fn assert_block_within_bound(block: &[u8; BLOCK_BYTES], values: &[f32; BLOCK_ELEMENTS]) -> usize {
    let mut decoded = [0.0f32; BLOCK_ELEMENTS];
    dequantize_block(block, FloatLayout::E3M2, &mut decoded).expect("six-bit layout");
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    values
        .iter()
        .zip(&decoded)
        .inspect(|(value, restored)| assert_element_within_bound(**value, **restored, scale))
        .count()
}

#[test]
fn convert_codec_selection_expert_and_attention_projections_only() {
    let policy = fp6_policy();
    let names: Vec<&str> = SELECTED_NAMES
        .iter()
        .chain(&EXCLUDED_NAMES)
        .copied()
        .collect();

    let selected: Vec<&str> = names
        .iter()
        .copied()
        .filter(|name| policy_selects(&policy, name))
        .collect();

    assert_eq!(selected, SELECTED_NAMES);
}

#[test]
fn convert_codec_source_f32_within_block_bound() {
    assert_converted_within_block_bound(GgmlType::F32);
}

#[test]
fn convert_codec_source_f16_within_block_bound() {
    assert_converted_within_block_bound(GgmlType::F16);
}

#[test]
fn convert_codec_source_q8_0_within_block_bound() {
    assert_converted_within_block_bound(GgmlType::Q8_0);
}

#[test]
fn convert_codec_source_q4_0_within_block_bound() {
    assert_converted_within_block_bound(GgmlType::Q4_0);
}

#[test]
fn convert_codec_checkpoint_reloads_with_every_tensor_present() {
    let weights = granite_q8_0_weights();
    let slice = &weights[..(ROW_ELEMENTS * 4) as usize];
    let data = encode_source(GgmlType::Q8_0, slice);
    let tensors = SELECTED_NAMES
        .iter()
        .chain(&EXCLUDED_NAMES)
        .map(|name| TensorPayload {
            name: (*name).into(),
            dims: dims(&[ROW_ELEMENTS, 4]),
            ggml_type: GgmlType::Q8_0,
            data: &data,
        })
        .collect();
    let source_file = write_complete(&GgufModel {
        version: 3,
        metadata: Vec::new(),
        tensors,
    })
    .expect("write the source checkpoint");
    let source = parse_complete(&source_file).expect("source parses");

    let (output_file, converted) =
        convert_codec(&source, &source_file, &fp6_policy()).expect("convert the checkpoint");

    let output = parse_complete(&output_file).expect("converted checkpoint parses");
    assert_eq!(output.tensors.len(), source.tensors.len());
    assert_eq!(converted, SELECTED_NAMES.len());
    for tensor in &output.tensors {
        let selected = SELECTED_NAMES.contains(&tensor.name.as_str());
        let expected = if selected {
            GgmlType::Fp6E3M2
        } else {
            GgmlType::Q8_0
        };
        assert_eq!(tensor.ggml_type, expected, "{}", tensor.name);
        if !selected {
            assert_eq!(tensor_bytes(&output, &output_file, &tensor.name), data);
        }
    }
}

#[test]
fn convert_codec_checkpoint_moe_q8_0_saves_250mb() {
    let mapping = map_granite();
    let parsed = parse_complete(&mapping).expect("granite parses as gguf");
    let policy = fp6_policy();
    let expected = parsed
        .tensors
        .iter()
        .filter(|tensor| {
            tensor.name.ends_with(".weight")
                && policy
                    .classes
                    .iter()
                    .any(|class| tensor.name.contains(&format!(".{class}.")))
        })
        .count();

    let (output_file, converted) =
        convert_codec(&parsed, &mapping, &policy).expect("convert the granite checkpoint");

    let saved = mapping.len() - output_file.len();
    std::println!(
        "convert_codec granite: converted {converted} of {expected} matching, bytes {} -> {} saved {saved}",
        mapping.len(),
        output_file.len()
    );
    assert!(expected > 0, "the policy matched no tensor in the header");
    assert_eq!(converted, expected);
    assert!(saved >= SAVING_FLOOR_BYTES, "saved only {saved} bytes");
}
