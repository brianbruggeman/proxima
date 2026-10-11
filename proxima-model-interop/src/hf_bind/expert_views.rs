#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;

use proxima_primitives::pipe::sans_io::Outcome;
use proxima_safetensors::{SafetensorsModel, SafetensorsParser, TensorPayload, write_complete};
use proxima_tensor::NodeId;

use super::*;
use crate::expert_slab::ExpertSlab;
use crate::profiles::binding_profile;

const EXPERTS: usize = 2;
const INPUT: usize = 3;
const OUTPUT: usize = 2;
const FUSED: &str = "model.layers.0.block_sparse_moe.input_linear.weight";
const DOWN: &str = "model.layers.0.block_sparse_moe.output_linear.weight";

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bf16_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16))
        .collect()
}

fn range(first: u32, count: u32) -> Vec<f32> {
    (first..first + count).map(|value| value as f32).collect()
}

struct Checkpoint {
    file: Vec<u8>,
    manifest: Manifest,
    data_start: u64,
}

fn checkpoint(dtype: DType, encode: fn(&[f32]) -> Vec<u8>) -> Checkpoint {
    let fused = encode(&range(1, 24));
    let down = encode(&range(101, 18));
    let tensors = vec![
        TensorPayload {
            name: FUSED.into(),
            dtype,
            shape: vec![EXPERTS as u64, 2 * OUTPUT as u64, INPUT as u64],
            data: &fused,
        },
        TensorPayload {
            name: DOWN.into(),
            dtype,
            shape: vec![EXPERTS as u64, INPUT as u64, INPUT as u64],
            data: &down,
        },
    ];
    let file = write_complete(&SafetensorsModel {
        tensors,
        metadata: BTreeMap::new(),
    })
    .expect("writes a real safetensors buffer");
    let header_length = u64::from_le_bytes(file[..8].try_into().expect("8-byte prefix"));
    let data_start = 8 + header_length;
    let mut parser = SafetensorsParser::new();
    parser.feed(&file[..data_start as usize]);
    let Outcome::Event(manifest) = parser.poll().expect("header parses") else {
        panic!("a complete header must produce a manifest");
    };
    Checkpoint {
        manifest: manifest.clone(),
        file,
        data_start,
    }
}

fn bind<'file>(checkpoint: &'file Checkpoint, leaf: &str, rows: usize) -> BoundWeights<'file> {
    let binding = binding_profile("granitemoe").expect("granitemoe parses");
    let shape = [
        Extent::Static(EXPERTS as u32),
        Extent::Static(INPUT as u32),
        Extent::Static(rows as u32),
    ];
    let mut weights = BoundWeights::new(&[]);
    bind_safetensors_expert_stack(
        &checkpoint.manifest,
        &checkpoint.file,
        checkpoint.data_start,
        &binding,
        leaf,
        &shape,
        &mut weights,
    )
    .expect("the stack binds");
    weights
}

fn part_ranges<'file>(weights: &BoundWeights<'file>, leaf: &str) -> Vec<&'file [u8]> {
    let (_, ranges, codec) = weights
        .expert_ranges
        .iter()
        .find(|(name, _, _)| name == leaf)
        .expect("fused part binds as per-expert file ranges");
    assert_eq!(*codec, Codec::BFloat16);
    ranges.clone()
}

fn part_values(weights: &BoundWeights<'_>, leaf: &str) -> Vec<f32> {
    part_ranges(weights, leaf)
        .into_iter()
        .flat_map(bf16_values)
        .collect()
}

fn expert_rows(values: &[f32], expert: usize) -> &[f32] {
    &values[expert * OUTPUT * INPUT..(expert + 1) * OUTPUT * INPUT]
}

#[test]
fn hf_expert_stack_gate_and_up_are_row_parts_of_each_expert_in_out_in_order() {
    let checkpoint = checkpoint(DType::BFloat16, bf16_bytes);

    let gate = bind(&checkpoint, "blk.0.ffn_gate_exps.weight", OUTPUT);
    let up = bind(&checkpoint, "blk.0.ffn_up_exps.weight", OUTPUT);
    let gate_values = part_values(&gate, "blk.0.ffn_gate_exps.weight");
    let up_values = part_values(&up, "blk.0.ffn_up_exps.weight");

    assert_eq!(expert_rows(&gate_values, 0), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(
        expert_rows(&up_values, 0),
        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    );
    assert_eq!(
        expert_rows(&gate_values, 1),
        [13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
    );
    assert_eq!(
        expert_rows(&up_values, 1),
        [19.0, 20.0, 21.0, 22.0, 23.0, 24.0]
    );
    assert!(gate.owned.is_empty() && up.owned.is_empty(), "no f32 copy");
    assert!(
        gate.packed_owned.is_empty() && up.packed_owned.is_empty(),
        "no byte copy"
    );
}

#[test]
fn hf_expert_stack_is_not_the_transposed_in_out_layout() {
    let checkpoint = checkpoint(DType::BFloat16, bf16_bytes);

    let gate = bind(&checkpoint, "blk.0.ffn_gate_exps.weight", OUTPUT);
    let gate_values = part_values(&gate, "blk.0.ffn_gate_exps.weight");

    let transposed_control = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    assert_ne!(expert_rows(&gate_values, 0), transposed_control);
}

#[test]
fn hf_expert_stack_down_borrows_the_mapped_bytes_without_a_copy() {
    let checkpoint = checkpoint(DType::BFloat16, bf16_bytes);

    let down = bind(&checkpoint, "blk.0.ffn_down_exps.weight", INPUT);

    let (_, block) = down.packed.first().expect("a whole tensor binds packed");
    let QuantizedBlock::Packed { codec, bytes } = block else {
        panic!("a bf16 stack binds as a packed block");
    };
    let mapped = checkpoint.file.as_ptr_range();
    assert_eq!(*codec, Codec::BFloat16);
    assert!(
        mapped.contains(&bytes.as_ptr()),
        "bytes alias the file buffer"
    );
    assert_eq!(bf16_values(bytes), range(101, 18));
    assert!(down.owned.is_empty() && down.packed_owned.is_empty());
}

#[test]
fn hf_expert_stack_f32_checkpoint_keeps_the_in_out_transposed_decode() {
    let checkpoint = checkpoint(DType::Float32, f32_bytes);

    let gate = bind(&checkpoint, "blk.0.ffn_gate_exps.weight", OUTPUT);

    let (_, values) = gate.owned.first().expect("f32 binds owned");
    assert_eq!(
        &values[..INPUT * OUTPUT],
        [1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        "the gather kernel rejects a Float32 block, so f32 stays [in, out]"
    );
}

#[test]
fn hf_expert_stack_fused_parts_borrow_the_mapped_bytes_without_a_copy() {
    let checkpoint = checkpoint(DType::BFloat16, bf16_bytes);

    let up = bind(&checkpoint, "blk.0.ffn_up_exps.weight", OUTPUT);

    let ranges = part_ranges(&up, "blk.0.ffn_up_exps.weight");
    let mapped = checkpoint.file.as_ptr_range();
    assert_eq!(ranges.len(), EXPERTS);
    assert!(
        ranges
            .iter()
            .all(|range| mapped.contains(&range.as_ptr()) && range.len() == OUTPUT * INPUT * 2),
        "each expert's up rows alias the file buffer"
    );
}

#[test]
fn hf_expert_stack_fused_parts_fill_the_expert_slab_in_place() {
    let checkpoint = checkpoint(DType::BFloat16, bf16_bytes);
    let gate = bind(&checkpoint, "blk.0.ffn_gate_exps.weight", OUTPUT);
    let ranges = part_ranges(&gate, "blk.0.ffn_gate_exps.weight");
    let mut slab = ExpertSlab::new();

    slab.bind_layer_experts(
        0,
        NodeId(3),
        Codec::BFloat16,
        &ranges,
        OUTPUT as u32,
        INPUT as u32,
    )
    .expect("per-expert ranges bind");

    let mut scratch = Vec::new();
    let sources = slab
        .sources_for_step(&mut scratch)
        .expect("a complete layer snapshots");
    let entries = sources[&NodeId(3)].entries();
    let mapped = checkpoint.file.as_ptr_range();
    assert_eq!(entries.len(), EXPERTS);
    assert!(entries.iter().all(|entry| {
        let bytes = entry.block.packed_bytes().expect("a packed entry");
        mapped.contains(&bytes.as_ptr())
    }));
    assert_eq!(slab.memory().owned_bytes, 0);
}

#[test]
fn hf_expert_stack_empty_range_list_is_refused_by_the_slab() {
    let mut slab = ExpertSlab::new();

    let refused = slab.bind_layer_experts(0, NodeId(3), Codec::BFloat16, &[], 2, 3);

    assert!(matches!(
        refused,
        Err(InteropError::ExpertSlabIndexOutOfRange {
            layer: 0,
            expert: 0
        })
    ));
}
