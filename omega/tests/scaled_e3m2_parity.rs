//! Metal parity of the scaled e3m2 matvec against an f64 dot-product loop.
//!
//! The weights are real packed e3m2 blocks: gaussian-shaped rows quantized by
//! `proxima_gguf::quant::scaled_float::quantize_block`, the encoder a converted
//! checkpoint goes through. The reference never touches the kernels or the
//! codec's decoder: it unpacks the bytes bit by bit from the layout in the spec
//! (four 6-bit fields per little-endian 3-byte word, behind an f16 scale),
//! decodes each field from the OCP E3M2 formula, and accumulates in f64.
//!
//! The bound is the worst-case rounding bound of an f32 dot product over `depth`
//! terms, `gamma_depth * sum(|term|)` with `gamma_n = n u / (1 - n u)`,
//! `u = 2^-24`. It holds for any association of the sum, so it covers the
//! contraction and reassociation that `NumericPolicy::llama_relaxed()` grants
//! the kernel. `NumericPolicy` itself is a permission set and carries no
//! tolerance.
//!
//! Three kernels serve the cases, told apart by the shape token in the plan's
//! kernel key, which every case asserts: the single-activation row-blocked
//! matvec (`B`), the multi-activation row-blocked fold (`M`, 128 tokens is
//! below the tiled minimum of `sized::TILED_GEMM_MIN_TOKENS`) and the tiled
//! `simdgroup_matrix` GEMM (`G`, 192 tokens). The row-blocked kernels decode a
//! block exactly in f32 (an 11-bit scale times a 3-bit significand). The tiled
//! kernel stages the decoded weight tile as `half` (`scaled_e3m2_decode` casts
//! to `half`), so its reference rounds each weight to f16 first; the only
//! rounding left in every case is then the dot product.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use half::f16;
use proxima_gguf::quant::scaled_float::quantize_block;
use proxima_primitives::float_layout::FloatLayout;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, append, projection,
};
use rayon::prelude::*;

const BLOCK_ELEMENTS: usize = 32;
const BLOCK_BYTES: usize = omega::Codec::FP6_E3M2_BLOCK_BYTES;
const SCALE_BYTES: usize = 2;
const GROUP_ELEMENTS: usize = 4;
const GROUP_BYTES: usize = 3;
const FIELD_BITS: usize = 6;
const FIELD_MASK: u32 = 0x3f;
const WEIGHT_STD: f32 = 0.02;
const WEIGHT_SEED: u64 = 0x5eed_0e3a;
const ACTIVATION_SEED: u64 = 0xac71_0e3a;

#[derive(Clone, Copy)]
enum Path {
    RowBlockedSingle,
    RowBlockedMulti,
    TiledGemm,
}

impl Path {
    fn weight_rounds_to_half(self) -> bool {
        matches!(self, Self::TiledGemm)
    }

    fn key_marker(self) -> &'static str {
        match self {
            Self::RowBlockedSingle => "cFfB1",
            Self::RowBlockedMulti => "cFfM1",
            Self::TiledGemm => "cFfG1",
        }
    }
}

#[derive(Clone, Copy)]
struct Case {
    rows: usize,
    depth: usize,
    tokens: usize,
    path: Path,
}

struct Reference {
    sum: f64,
    magnitude: f64,
}

struct MetalRun {
    values: Vec<f32>,
    kernel_keys: Vec<String>,
}

struct Outcome {
    worst_ratio: f64,
    worst_row: usize,
    bound: f64,
    compared: usize,
    kernel_keys: Vec<String>,
}

fn gaussian_shaped(lcg: &mut Lcg) -> f32 {
    (0..4).map(|_| lcg.next_unit()).sum::<f32>() * 0.5 * WEIGHT_STD
}

fn weight_rows(case: Case) -> Vec<f32> {
    let mut lcg = Lcg(WEIGHT_SEED ^ (case.rows * case.depth) as u64);
    (0..case.rows * case.depth)
        .map(|_| gaussian_shaped(&mut lcg))
        .collect()
}

fn activation_rows(case: Case) -> Vec<f32> {
    let mut lcg = Lcg(ACTIVATION_SEED ^ (case.tokens * case.depth) as u64);
    (0..case.tokens * case.depth)
        .map(|_| lcg.next_unit())
        .collect()
}

fn pack_weights(values: &[f32]) -> Vec<u8> {
    let mut packed = vec![0u8; values.len() / BLOCK_ELEMENTS * BLOCK_BYTES];
    values
        .as_chunks::<BLOCK_ELEMENTS>()
        .0
        .iter()
        .zip(packed.as_chunks_mut::<BLOCK_BYTES>().0)
        .for_each(|(block_values, block_bytes)| {
            quantize_block(block_values, FloatLayout::E3M2, block_bytes)
                .expect("finite gaussian-shaped weights quantize");
        });
    packed
}

fn field_value(field: u32) -> f64 {
    let sign = if field >> 5 == 1 { -1.0 } else { 1.0 };
    let exponent = i32::try_from((field >> 2) & 0b111).expect("three bits fit i32");
    let mantissa = f64::from(field & 0b11) / 4.0;
    match exponent {
        0 => sign * mantissa * 2f64.powi(-2),
        _ => sign * (1.0 + mantissa) * 2f64.powi(exponent - 3),
    }
}

fn block_fields(block: &[u8; BLOCK_BYTES]) -> impl Iterator<Item = u32> + '_ {
    block[SCALE_BYTES..]
        .as_chunks::<GROUP_BYTES>()
        .0
        .iter()
        .flat_map(|group| {
            let word = u32::from_le_bytes([group[0], group[1], group[2], 0]);
            (0..GROUP_ELEMENTS).map(move |slot| (word >> (FIELD_BITS * slot)) & FIELD_MASK)
        })
}

fn staged_weight(weight: f64, rounds_to_half: bool) -> f64 {
    if rounds_to_half {
        f64::from(f16::from_f64(weight))
    } else {
        weight
    }
}

fn decode_weights(packed: &[u8], rounds_to_half: bool) -> Vec<f64> {
    packed
        .as_chunks::<BLOCK_BYTES>()
        .0
        .iter()
        .flat_map(|block| {
            let scale = f64::from(f16::from_le_bytes([block[0], block[1]]).to_f32());
            block_fields(block)
                .map(move |field| staged_weight(scale * field_value(field), rounds_to_half))
        })
        .collect()
}

fn codes_present(packed: &[u8]) -> usize {
    let mut seen = [false; 64];
    packed
        .as_chunks::<BLOCK_BYTES>()
        .0
        .iter()
        .flat_map(block_fields)
        .for_each(|field| seen[field as usize] = true);
    seen.iter().filter(|present| **present).count()
}

fn references(case: Case, weights: &[f64], activation: &[f32]) -> Vec<Reference> {
    (0..case.tokens * case.rows)
        .into_par_iter()
        .map(|index| {
            let (token, row) = (index / case.rows, index % case.rows);
            let weight_row = &weights[row * case.depth..(row + 1) * case.depth];
            let activation_row = &activation[token * case.depth..(token + 1) * case.depth];
            weight_row.iter().zip(activation_row).fold(
                Reference {
                    sum: 0.0,
                    magnitude: 0.0,
                },
                |reference, (&weight, &value)| {
                    let term = weight * f64::from(value);
                    Reference {
                        sum: reference.sum + term,
                        magnitude: reference.magnitude + term.abs(),
                    }
                },
            )
        })
        .collect()
}

fn matmul_program(case: Case) -> (Vec<Op>, NodeId) {
    let (tokens, depth, rows) = (case.tokens as u32, case.depth as u32, case.rows as u32);
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::UInt8,
            shape: vec![Extent::Static(rows), Extent::Static(depth)],
            name: None,
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(tokens), Extent::Static(depth)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(3, &[1, 2]))),
                (activation, IndexMap::Affine(projection(3, &[0, 2]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, sum)
}

fn metal_run(case: Case, packed: &[u8], activation: &[f32]) -> MetalRun {
    let (program, sum) = matmul_program(case);
    let blocks = [
        QuantizedBlock::Packed {
            codec: omega::Codec::Fp6E3M2,
            bytes: packed,
        },
        QuantizedBlock::Float32(activation),
    ];
    let plan = omega::plan(
        &program,
        &[],
        &blocks,
        &[sum],
        NumericPolicy::llama_relaxed(),
    )
    .expect("metal plans an e3m2 matvec");
    let values = omega::execute_plan(&plan, &blocks)
        .expect("metal runs the e3m2 matvec on a real device")
        .root()
        .to_vec();
    let kernel_keys = plan
        .kernel_keys()
        .expect("the plan reports its kernel keys");
    MetalRun {
        values,
        kernel_keys,
    }
}

fn dot_rounding_bound(depth: usize) -> f64 {
    let accumulated = depth as f64 * 2f64.powi(-24);
    accumulated / (1.0 - accumulated)
}

fn compare(case: Case, metal: MetalRun, reference: &[Reference]) -> Outcome {
    assert_eq!(
        metal.values.len(),
        case.tokens * case.rows,
        "metal output length"
    );
    assert_eq!(
        reference.len(),
        metal.values.len(),
        "reference output length"
    );
    let (worst_index, worst_ratio) = metal
        .values
        .iter()
        .zip(reference)
        .map(|(&value, expected)| (f64::from(value) - expected.sum).abs() / expected.magnitude)
        .enumerate()
        .fold((0, 0.0), |worst, (index, ratio)| {
            if ratio > worst.1 {
                (index, ratio)
            } else {
                worst
            }
        });
    Outcome {
        worst_ratio,
        worst_row: worst_index % case.rows,
        bound: dot_rounding_bound(case.depth),
        compared: metal.values.len(),
        kernel_keys: metal.kernel_keys,
    }
}

fn parity_outcome(case: Case, perturb: impl FnOnce(&mut [u8])) -> Outcome {
    let packed = pack_weights(&weight_rows(case));
    let activation = activation_rows(case);
    assert_eq!(
        codes_present(&packed),
        64,
        "degenerate gate: the weights must exercise all 64 e3m2 codes"
    );
    let expected = references(
        case,
        &decode_weights(&packed, case.path.weight_rounds_to_half()),
        &activation,
    );
    let mut under_test = packed.clone();
    perturb(&mut under_test);
    let metal = metal_run(case, &under_test, &activation);
    let outcome = compare(case, metal, &expected);
    assert!(
        outcome
            .kernel_keys
            .iter()
            .any(|key| key.contains(case.path.key_marker())),
        "degenerate gate: kernel keys {:?} do not carry the {} path marker",
        outcome.kernel_keys,
        case.path.key_marker()
    );
    outcome
}

fn report(case: Case, outcome: &Outcome) {
    eprintln!(
        "scaled_e3m2_parity rows={} depth={} tokens={} compared={} worst_ratio={:e} worst_row={} bound={:e} kernel_keys={:?}",
        case.rows,
        case.depth,
        case.tokens,
        outcome.compared,
        outcome.worst_ratio,
        outcome.worst_row,
        outcome.bound,
        outcome.kernel_keys
    );
}

fn assert_within_bound(case: Case) {
    let outcome = parity_outcome(case, |_| {});
    report(case, &outcome);

    assert_eq!(
        outcome.compared,
        case.tokens * case.rows,
        "degenerate gate: every output element must be compared"
    );
    assert!(
        outcome.worst_ratio <= outcome.bound,
        "{}x{} at {} tokens: worst |metal - f64| / sum|term| = {:e} at row {} exceeds the f32 \
         dot rounding bound {:e}",
        case.rows,
        case.depth,
        case.tokens,
        outcome.worst_ratio,
        outcome.worst_row,
        outcome.bound
    );
}

#[test]
fn scaled_e3m2_parity_1024x512_1_token() {
    assert_within_bound(Case {
        rows: 1024,
        depth: 512,
        tokens: 1,
        path: Path::RowBlockedSingle,
    });
}

#[test]
fn scaled_e3m2_parity_1024x512_128_tokens() {
    assert_within_bound(Case {
        rows: 1024,
        depth: 512,
        tokens: 128,
        path: Path::RowBlockedMulti,
    });
}

#[test]
fn scaled_e3m2_parity_1536x6144_1_token() {
    assert_within_bound(Case {
        rows: 1536,
        depth: 6144,
        tokens: 1,
        path: Path::RowBlockedSingle,
    });
}

#[test]
fn scaled_e3m2_parity_1536x6144_128_tokens() {
    assert_within_bound(Case {
        rows: 1536,
        depth: 6144,
        tokens: 128,
        path: Path::RowBlockedMulti,
    });
}

#[test]
fn scaled_e3m2_parity_1024x512_192_tokens_tiled_gemm() {
    assert_within_bound(Case {
        rows: 1024,
        depth: 512,
        tokens: 192,
        path: Path::TiledGemm,
    });
}

#[test]
fn scaled_e3m2_parity_1536x6144_192_tokens_tiled_gemm() {
    assert_within_bound(Case {
        rows: 1536,
        depth: 6144,
        tokens: 192,
        path: Path::TiledGemm,
    });
}

#[test]
fn scaled_e3m2_parity_control_perturbed_block_fails() {
    let case = Case {
        rows: 1024,
        depth: 512,
        tokens: 1,
        path: Path::RowBlockedSingle,
    };
    let perturbed_row = 0;

    let outcome = parity_outcome(case, |packed| {
        packed[SCALE_BYTES..BLOCK_BYTES]
            .iter_mut()
            .for_each(|byte| *byte ^= 0xff);
    });
    report(case, &outcome);

    assert!(
        outcome.worst_ratio > outcome.bound,
        "perturbed block ratio {:e} stayed within the bound {:e}",
        outcome.worst_ratio,
        outcome.bound
    );
    assert_eq!(
        outcome.worst_row, perturbed_row,
        "the failure must land on the row that was perturbed"
    );
}
