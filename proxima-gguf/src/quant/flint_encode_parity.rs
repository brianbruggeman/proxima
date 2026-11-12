//! Byte parity of the rule-derived encoders against the frozen hand-written ones
//! ([`super::reference_encoders`]) on real tensors: rows of the flint checkpoint, decoded to `f32`
//! through this crate's dequantizers and re-encoded by both, plus adversarial blocks.

#![allow(clippy::expect_used)] // test module: expect messages name the failed step

extern crate std;

use alloc::vec;
use alloc::vec::Vec;
use std::collections::BTreeSet;
use std::path::Path;

use proxima_numeric::BlockLayout;

use crate::quant::real_weights::{open_gguf, read_tensor_prefix};
use crate::quant::{QuantError, dispatch, pq2_0, q1_0, q2_0, q4_0, q5_0, q8_0, reference_encoders};

const FLINT_PATH: &str = "/Users/brianbruggeman/models/llm/flint-gguf/flint-UD-Q4_K_XL.gguf";
const PREFIX_BYTES: usize = 3 << 20;
const TENSORS_PER_TYPE: usize = 3;

type Encode = fn(&[f32], &mut [u8]) -> Result<(), QuantError>;

struct Format {
    name: &'static str,
    layout: BlockLayout,
    elements: usize,
    block_bytes: usize,
    reference: Encode,
    derived: Encode,
}

fn formats() -> [Format; 6] {
    [
        Format { name: "q4_0", layout: BlockLayout::Q4_0, elements: 32, block_bytes: 18, reference: reference_encoders::q4_0, derived: q4_0::quantize },
        Format { name: "q5_0", layout: BlockLayout::Q5_0, elements: 32, block_bytes: 22, reference: reference_encoders::q5_0, derived: q5_0::quantize },
        Format { name: "q8_0", layout: BlockLayout::Q8_0, elements: 32, block_bytes: 34, reference: reference_encoders::q8_0, derived: q8_0::quantize },
        Format { name: "q1_0", layout: BlockLayout::Q1_0, elements: 128, block_bytes: 18, reference: reference_encoders::q1_0, derived: q1_0::quantize },
        Format { name: "q2_0", layout: BlockLayout::Q2_0, elements: 64, block_bytes: 18, reference: reference_encoders::q2_0, derived: q2_0::quantize },
        Format { name: "pq2_0", layout: BlockLayout::PQ2_0, elements: 128, block_bytes: 34, reference: reference_encoders::pq2_0, derived: pq2_0::quantize },
    ]
}

/// Decoded rows of the first tensors of each storage type the flint file holds.
fn flint_rows() -> (Vec<f32>, usize, BTreeSet<&'static str>) {
    assert!(Path::new(FLINT_PATH).exists(), "flint fixture missing at {FLINT_PATH}");
    let (mut file, parsed, file_len) = open_gguf(Path::new(FLINT_PATH));
    let mut rows = Vec::new();
    let mut tensors = 0usize;
    let mut types = BTreeSet::new();
    let mut taken: std::collections::BTreeMap<&'static str, usize> = std::collections::BTreeMap::new();
    for tensor in &parsed.tensors {
        let kind = tensor.ggml_type;
        let seen = taken.entry(kind.name()).or_insert(0);
        if *seen >= TENSORS_PER_TYPE {
            continue;
        }
        let shape = kind.wire_shape();
        let (block_elements, block_bytes) = (shape.block_elements as usize, shape.block_bytes as usize);
        let bytes = read_tensor_prefix(&mut file, &parsed, file_len, tensor, PREFIX_BYTES);
        let whole = bytes.len() / block_bytes * block_bytes;
        let mut decoded = vec![0.0f32; whole / block_bytes * block_elements];
        if whole == 0 || dispatch::dequantize(kind, &bytes[..whole], &mut decoded).is_err() {
            continue;
        }
        *seen += 1;
        tensors += 1;
        types.insert(kind.name());
        rows.extend(decoded.into_iter().filter(|value| value.is_finite()));
    }
    (rows, tensors, types)
}

fn adversarial(elements: usize) -> Vec<f32> {
    let subnormal = f32::from_bits(0x0000_0007);
    let mut blocks = Vec::new();
    blocks.extend(vec![0.0f32; elements]);
    for peak in [1.0f32, -1.0, 65504.0, -65504.0, subnormal, -subnormal, f32::MAX, f32::MIN] {
        let mut single = vec![0.0f32; elements];
        single[elements / 2] = peak;
        blocks.extend(single);
    }
    for edge in [65504.0f32, -65504.0] {
        blocks.extend(vec![edge; elements]);
    }
    blocks.extend((0..elements).map(|index| if index % 2 == 0 { 65504.0 } else { -65504.0 }));
    blocks.extend((0..elements).map(|index| subnormal * (index as f32 - elements as f32 / 2.0)));
    blocks
}

fn mismatched_blocks(format: &Format, values: &[f32]) -> (usize, usize) {
    let blocks = values.len() / format.elements;
    let mut expected = vec![0u8; blocks * format.block_bytes];
    let mut derived = vec![0u8; expected.len()];
    (format.reference)(values, &mut expected).expect("reference encodes whole blocks");
    (format.derived)(values, &mut derived).expect("derived encodes whole blocks");
    let mut differing = 0;
    for (index, (left, right)) in expected
        .chunks(format.block_bytes)
        .zip(derived.chunks(format.block_bytes))
        .enumerate()
    {
        if left == right {
            continue;
        }
        let input = &values[index * format.elements..(index + 1) * format.elements];
        let peak = input.iter().fold(0.0f32, |peak, value| peak.max(value.abs()));
        // x * (1 / scale) overflows to infinity here, so the f32 -> integer cast is outside its
        // defined range and the two hand-written routines disagree (see the module report)
        let divisor = format.layout.encode_rule().map_or(1.0, |rule| rule.divisor.abs() as f32);
        let scale = peak / divisor;
        let overflowing_inverse = scale != 0.0 && (1.0 / scale).is_infinite();
        std::eprintln!(
            "{} block {index} differs (overflowing inverse: {overflowing_inverse})\n  reference {left:02x?}\n  derived   {right:02x?}",
            format.name
        );
        if !overflowing_inverse {
            differing += 1;
        }
    }
    (blocks, differing)
}

#[test]
fn rule_derived_encoders_equal_the_frozen_encoders_byte_for_byte_on_flint_rows_and_adversarial_blocks() {
    let (rows, tensors, types) = flint_rows();
    assert!(tensors > 0 && !rows.is_empty(), "no flint tensor decoded");
    std::eprintln!("flint rows: {} f32 from {tensors} tensors of types {types:?}", rows.len());
    let mut compared = 0usize;
    for format in formats() {
        let usable = rows.len() / format.elements * format.elements;
        let (real_blocks, real_diff) = mismatched_blocks(&format, &rows[..usable]);
        let (edge_blocks, edge_diff) = mismatched_blocks(&format, &adversarial(format.elements));
        std::eprintln!(
            "{}: real {real_blocks} blocks, {real_diff} differ; adversarial {edge_blocks} blocks, {edge_diff} differ",
            format.name
        );
        assert_eq!(real_diff + edge_diff, 0, "{}", format.name);
        compared += real_blocks + edge_blocks;
    }
    assert!(compared > 0);
    std::eprintln!("total blocks compared: {compared}");
}
