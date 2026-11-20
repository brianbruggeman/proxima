//! A BF16 weight is bound as an OFFSET view into the registered checkpoint mapping only on an
//! 8-byte boundary; the matvec matches the host decode at offsets 0 and 8, and a misaligned offset
//! (1, 3, 3611) takes the copy path and matches too. Bound as a view, the odd offsets 1 and 3 came
//! back NaN (PROXNUM-569). Those offsets are where a safetensors file
//! whose JSON header is not padded (`8 + header_len` not a multiple of 8) places its first tensor:
//! e2b's `per_layer_model_proj.weight` sat at mapping offset 3611 and every decode after token 0
//! was 0 (PROXNUM-565). The weight has that tensor's shape, 8960 rows of 1536 `bf16`, filled from
//! the crate's LCG; the same bytes are laid at each offset in one page-aligned mapping.
#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::slice;

use proxima_numeric::codec::Codec;
use proxima_numeric::{DType, NumericPolicy};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    AlignedBuffer, Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, append, map,
};

const ROWS: usize = 8960;
const REDUCTION: usize = 1536;
const BF16_BYTES: usize = 2;
const WEIGHT_SEED: u64 = 565;
const ACTIVATION_SEED: u64 = 2026;
const RELATIVE_TOLERANCE: f64 = 1.0e-6;
const MAPPING_SLACK_PAGES: usize = 2;

fn bf16_weight() -> Vec<u8> {
    let mut lcg = Lcg(WEIGHT_SEED);
    (0..ROWS * REDUCTION)
        .flat_map(|_| ((((lcg.next_unit() - 0.5) * 0.2).to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn activation() -> Vec<f32> {
    let mut lcg = Lcg(ACTIVATION_SEED);
    (0..REDUCTION).map(|_| lcg.next_unit() * 4.0 - 2.0).collect()
}

fn program() -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(&mut program, Op::Input { dtype: DType::UINT8, shape: vec![Extent::Static(ROWS as u32), Extent::Static(REDUCTION as u32)], name: None });
    let activation = append(&mut program, Op::Input { dtype: DType::FLOAT32, shape: vec![Extent::Static(1), Extent::Static(REDUCTION as u32)], name: None });
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![(weight, IndexMap::Affine(map::projection(3, &[1, 2]))), (activation, IndexMap::Affine(map::projection(3, &[0, 2])))],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("logits".into()),
        }),
    );
    (program, sum)
}

fn host_reference(weight: &[u8], activation: &[f32]) -> Vec<(f64, f64)> {
    weight
        .as_chunks::<{ REDUCTION * BF16_BYTES }>()
        .0
        .iter()
        .map(|row| {
            row.as_chunks::<BF16_BYTES>()
                .0
                .iter()
                .zip(activation)
                .map(|(pair, act)| f64::from(f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16)) * f64::from(*act))
                .fold((0.0f64, 0.0f64), |(sum, magnitude), term| (sum + term, magnitude + term.abs()))
        })
        .collect()
}

/// Lays `weight` at `offset` inside a page-aligned mapping, registers the mapping so the weight is
/// bound as an offset view, runs the matvec and returns how many logits leave tolerance (a NaN leaves it).
fn logits_out_of_tolerance_at(offset: usize, weight: &[u8], activation: &[f32]) -> (usize, Vec<f32>) {
    let page = omega::page_size();
    let mapping_bytes = (weight.len() / page + MAPPING_SLACK_PAGES) * page;
    let mut mapping = AlignedBuffer::new(mapping_bytes / 4, page).expect("page-aligned mapping");
    let mut image = vec![0u8; mapping.len() * 4];
    image[offset..offset + weight.len()].copy_from_slice(weight);
    for (word, chunk) in mapping.iter_mut().zip(image.as_chunks::<4>().0) {
        *word = f32::from_bits(u32::from_le_bytes(*chunk));
    }
    // SAFETY: `mapping` owns `mapping.len()` initialised f32s and outlives every use of this view.
    let mapped: &[u8] = unsafe { slice::from_raw_parts(mapping.as_ptr().cast::<u8>(), mapping.len() * 4) };
    omega::backend::register_checkpoint_mapping(mapped);

    let (program, output) = program();
    let bound = &mapped[offset..offset + weight.len()];
    let blocks = [QuantizedBlock::Packed { codec: Codec::BFloat16, bytes: bound }, QuantizedBlock::Float32(activation)];
    let outputs = [output];
    let plan = omega::plan(&program, &[], &blocks, &outputs, NumericPolicy::default()).expect("metal plans the matvec");
    let got = omega::execute_plan(&plan, &blocks).expect("metal runs the matvec").root().to_vec();
    omega::backend::unregister_checkpoint_mapping(mapped);

    let want = host_reference(weight, activation);
    let wrong = got
        .iter()
        .zip(&want)
        .filter(|(value, (reduced, magnitude))| !value.is_finite() || (f64::from(**value) - reduced).abs() > RELATIVE_TOLERANCE * magnitude + 1.0e-6)
        .count();
    (wrong, got)
}

#[test]
fn a_bf16_weight_at_an_eight_byte_aligned_mapping_offset_matches_the_host_decode() {
    let (weight, activation) = (bf16_weight(), activation());
    for offset in [0, 8] {
        let (wrong, got) = logits_out_of_tolerance_at(offset, &weight, &activation);
        assert_eq!(got.len(), ROWS, "one f32 per row at offset {offset}");
        assert_eq!(wrong, 0, "offset {offset}: {wrong} of {ROWS} logits differ from the host decode; first logits {:?}", &got[..4]);
    }
}

#[test]
fn a_bf16_weight_at_an_odd_mapping_offset_is_copied_and_matches_the_host_decode() {
    let (weight, activation) = (bf16_weight(), activation());
    for offset in [1, 3, 3611] {
        let (wrong, got) = logits_out_of_tolerance_at(offset, &weight, &activation);
        assert_eq!(got.len(), ROWS, "one f32 per row at offset {offset}");
        assert_eq!(wrong, 0, "offset {offset}: {wrong} of {ROWS} logits differ from the host decode (PROXNUM-569: a misaligned offset must take the copy path); first logits {:?}", &got[..4]);
    }
}
