//! The IQ4_XS decode matvec is rendered from `BlockLayout::IQ4_XS` and its explicit level table
//! (`msl/sub_block_assembled_matvec.rs`, `level_matvec_msl`). Its output on real flint expert tensors, at
//! the single-token routed shape decode runs (one token, six selected experts), must equal an
//! independent f64 reduction over the gguf dequantizer's (`proxima_gguf::quant::iq4_xs`) values.
//! A control compares the same output against the neighbouring row's reduction and must fail.
#![cfg(all(feature = "metal", target_os = "macos"))]
// workspace lints deny unwrap/expect (Cargo.toml:93-94) and an integration test binary is not cfg(test), so a failed fixture read must be an allowed panic.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use proxima_gguf::parser::{GgufEvent, GgufParser};
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::types::GgmlType;
use proxima_numeric::codec::Codec;
use proxima_numeric::{DType, NumericPolicy};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp, append, map};

const FLINT_PATH: &str = "/Users/brianbruggeman/models/llm/flint-gguf/flint-UD-Q4_K_XL.gguf";
const ACTIVATION_SEED: u64 = 2026;
const QK_K: usize = 256;
const SELECTED: usize = 6;
const EXPERTS: usize = 16;
const ROWS: usize = 96;
const RELATIVE_TOLERANCE: f64 = 1.0e-6;

/// What the reduce folds. Only `PlainProduct` at f32 has a rendered row-blocked body; the others must decline it.
#[derive(Clone, Copy, PartialEq)]
enum Variant {
    PlainProduct,
    /// the same product reduced at half precision
    Float16,
    /// the product negated before the reduce, so the fused element body is not a plain product
    NegatedProduct,
}

impl Variant {
    fn dtype(self) -> DType {
        if self == Self::Float16 { DType::FLOAT16 } else { DType::FLOAT32 }
    }

    fn sign(self) -> f64 {
        if self == Self::NegatedProduct { -1.0 } else { 1.0 }
    }

    /// half carries ten mantissa bits through the whole fold, f32 rounds only at the end.
    fn relative_tolerance(self) -> f64 {
        if self == Self::Float16 { 2.0e-3 } else { RELATIVE_TOLERANCE }
    }
}

#[derive(Clone, Copy)]
struct Case {
    tensor: &'static str,
    per_selected_activation: bool,
    tokens: usize,
    variant: Variant,
}

const GATE: Case = Case { tensor: "blk.19.ffn_gate_exps.weight", per_selected_activation: false, tokens: 1, variant: Variant::PlainProduct };
const DOWN: Case = Case { tensor: "blk.19.ffn_down_exps.weight", per_selected_activation: true, tokens: 1, variant: Variant::PlainProduct };
const GATE_BATCH: Case = Case { tensor: "blk.19.ffn_gate_exps.weight", per_selected_activation: false, tokens: 5, variant: Variant::PlainProduct };
const DOWN_BATCH: Case = Case { tensor: "blk.19.ffn_down_exps.weight", per_selected_activation: true, tokens: 5, variant: Variant::PlainProduct };
const GATE_HALF: Case = Case { variant: Variant::Float16, ..GATE };
const GATE_NEGATED: Case = Case { variant: Variant::NegatedProduct, ..GATE };

/// Each token names `SELECTED` distinct consecutive experts starting at a token-dependent offset, the way the
/// flint level-codec parity fixture routes.
fn expert_of(token: usize, slot: usize) -> usize {
    token * 5 % 7 + slot
}

fn header(file: &mut File, file_len: u64) -> ParsedGguf {
    let mut prefix_len = 1usize << 22;
    loop {
        let mut buffer = vec![0u8; prefix_len];
        file.seek(SeekFrom::Start(0)).unwrap();
        let read = file.read(&mut buffer).unwrap();
        buffer.truncate(read);
        if let Ok((parser, events)) = GgufParser::new().push(&buffer) {
            let (mut version, mut metadata, mut tensors, mut completion) = (None, Vec::new(), Vec::new(), None);
            for event in events {
                match event {
                    GgufEvent::Header { version: found, .. } => version = Some(found),
                    GgufEvent::Metadata { key, value } => metadata.push((key, value)),
                    GgufEvent::Tensor(tensor) => tensors.push(tensor),
                    GgufEvent::Complete { data_offset, alignment } => completion = Some((data_offset, alignment)),
                }
            }
            if let (Some(version), Some((data_offset, alignment))) = (version, completion) {
                parser.finish().unwrap();
                return ParsedGguf { version, tensor_count: tensors.len() as u64, kv_count: metadata.len() as u64, metadata, tensors, data_offset, alignment };
            }
        }
        assert!((prefix_len as u64) < file_len, "no gguf header in the first {prefix_len} bytes");
        prefix_len *= 2;
    }
}

struct Slab {
    bytes: Vec<u8>,
    reduction: usize,
    row_bytes: usize,
}

/// The first `ROWS` rows of each of the first `EXPERTS` experts of `case.tensor`, concatenated in expert order.
fn read_slab(case: &Case) -> Slab {
    let path = proxima_gguf::checkpoint_resolver::require_checkpoint("PROXIMA_LEVEL_CODEC_PARITY_GGUF", FLINT_PATH, 0);
    let mut file = File::open(&path).expect("the checkpoint opens");
    let file_len = file.metadata().unwrap().len();
    let parsed = header(&mut file, file_len);
    let tensor = parsed.tensors.iter().find(|tensor| tensor.name == case.tensor).expect("the tensor is in the directory");
    assert_eq!(tensor.ggml_type, GgmlType::Iq4Xs, "{} is an IQ4_XS tensor", case.tensor);
    let reduction = tensor.dims[0] as usize;
    let rows_per_expert = tensor.dims[1] as usize;
    let row_bytes = reduction / QK_K * proxima_gguf::quant::iq4_xs::BLOCK_BYTES;
    let range = parsed.tensor_data_range(tensor, file_len).unwrap();
    let mut bytes = vec![0u8; EXPERTS * ROWS * row_bytes];
    for (expert, target) in bytes.chunks_exact_mut(ROWS * row_bytes).enumerate() {
        file.seek(SeekFrom::Start(range.start + (expert * rows_per_expert * row_bytes) as u64)).unwrap();
        file.read_exact(target).unwrap();
    }
    Slab { bytes, reduction, row_bytes }
}

fn activation(reduction: usize, case: &Case) -> Vec<f32> {
    let mut lcg = Lcg(ACTIVATION_SEED);
    let rows = if case.per_selected_activation { case.tokens * SELECTED } else { case.tokens };
    (0..rows * reduction).map(|_| lcg.next_unit() * 4.0 - 2.0).collect()
}

fn input(program: &mut Vec<Op>, dtype: DType, shape: &[usize]) -> NodeId {
    append(program, Op::Input { dtype, shape: shape.iter().map(|&extent| Extent::Static(extent as u32)).collect(), name: None })
}

fn expert_gather_map(route: NodeId) -> IndexMap {
    let axis = |dim: u16| map::AxisIndex { terms: core::iter::once(map::AxisTerm::projection(dim)).collect(), offset: 0, len: None };
    IndexMap::Computed {
        indices: route,
        index_map: map::projection(4, &[0, 1]),
        base: map::IndexPattern { iter_rank: 4, axes: vec![map::AxisIndex::default(), axis(2), axis(3)] },
        gathered_dim: 0,
    }
}

fn program(reduction: usize, case: &Case) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let tokens = case.tokens;
    let weight = input(&mut program, DType::UINT8, &[EXPERTS, ROWS, reduction]);
    let route = input(&mut program, DType::FLOAT32, &[tokens, SELECTED]);
    let (activation, activation_map) = if case.per_selected_activation {
        (input(&mut program, case.variant.dtype(), &[tokens, SELECTED, reduction]), map::projection(4, &[0, 1, 3]))
    } else {
        (input(&mut program, case.variant.dtype(), &[tokens, reduction]), map::projection(4, &[0, 3]))
    };
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: case.variant.dtype(),
            body: ScalarOp::Multiply,
            operands: vec![(weight, expert_gather_map(route)), (activation, IndexMap::Affine(activation_map))],
            name: None,
        },
    );
    let reduced_operand = if case.variant == Variant::NegatedProduct {
        append(
            &mut program,
            Op::Elementwise {
                dtype: case.variant.dtype(),
                body: ScalarOp::Negate,
                operands: vec![(product, IndexMap::Affine(map::projection(4, &[0, 1, 2, 3])))],
                name: None,
            },
        )
    } else {
        product
    };
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: case.variant.dtype(),
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: reduced_operand,
            in_map: IndexMap::Affine(map::projection(4, &[0, 1, 2, 3])),
            out_map: IndexMap::Affine(map::projection(4, &[0, 1, 2])),
            keep: Keep::Reduce,
            name: Some("stacked_expert_product".into()),
        }),
    );
    (program, sum)
}

struct Run {
    output: Vec<f32>,
    kernel_keys: Vec<String>,
}

fn run(case: &Case, slab: &Slab, activation: &[f32]) -> Run {
    let (program, output) = program(slab.reduction, case);
    let route: Vec<f32> = (0..case.tokens * SELECTED).map(|pair| expert_of(pair / SELECTED, pair % SELECTED) as f32).collect();
    let blocks = [QuantizedBlock::Packed { codec: Codec::Iq4Xs, bytes: &slab.bytes }, QuantizedBlock::Float32(&route), QuantizedBlock::Float32(activation)];
    let outputs = [output];
    let plan = omega::plan(&program, &[], &blocks, &outputs, NumericPolicy::default()).expect("metal plans the expert product");
    let kernel_keys = plan.kernel_keys().expect("the plan exposes its kernel keys");
    let evaluated = omega::execute_plan(&plan, &blocks).expect("metal runs the expert product");
    Run { output: evaluated.root().to_vec(), kernel_keys }
}

/// `(reduction of the host-decoded row against the activation, sum of the absolute terms)` per token, selected slot and row.
fn reference(case: &Case, slab: &Slab, activation: &[f32]) -> Vec<(f64, f64)> {
    let mut values = vec![0.0f32; slab.reduction];
    let mut reduced = Vec::with_capacity(case.tokens * SELECTED * ROWS);
    for token in 0..case.tokens {
        for slot in 0..SELECTED {
            let expert = expert_of(token, slot);
            let activation_row = if case.per_selected_activation { token * SELECTED + slot } else { token };
            let row_activation = &activation[activation_row * slab.reduction..(activation_row + 1) * slab.reduction];
            for row in 0..ROWS {
                let start = (expert * ROWS + row) * slab.row_bytes;
                proxima_gguf::quant::iq4_xs::dequantize(&slab.bytes[start..start + slab.row_bytes], &mut values).expect("the gguf dequantizer reads the row");
                let terms = values.iter().zip(row_activation).map(|(weight, act)| f64::from(*weight) * f64::from(*act));
                let (sum, magnitude) = terms.fold((0.0f64, 0.0f64), |(sum, magnitude), term| (sum + term, magnitude + term.abs()));
                reduced.push((sum * case.variant.sign(), magnitude));
            }
        }
    }
    reduced
}

fn out_of_tolerance(got: &[f32], want: &[(f64, f64)], shift: usize, relative_tolerance: f64) -> usize {
    got.iter()
        .enumerate()
        .filter(|(at, value)| {
            let (reduced, magnitude) = want[(at + shift) % want.len()];
            (f64::from(**value) - reduced).abs() > relative_tolerance * magnitude + 1.0e-6
        })
        .count()
}

fn assert_row_blocked_matches_host_decode(case: &Case) {
    let slab = read_slab(case);
    let activation = activation(slab.reduction, case);
    let actual = run(case, &slab, &activation);
    let want = reference(case, &slab, &activation);
    let row_blocked = actual.kernel_keys.iter().any(|key| key.contains("_cpfB"));
    assert_eq!(
        row_blocked,
        case.variant == Variant::PlainProduct,
        "{}: row-blocked kernel selection must follow whether the body is rendered: {:?}",
        case.tensor,
        actual.kernel_keys
    );
    assert_eq!(actual.output.len(), want.len(), "{}: one f32 per selected expert and row", case.tensor);
    assert!(actual.output.iter().all(|value| value.is_finite()), "{}: a non-finite output would pass the tolerance comparison vacuously", case.tensor);
    let tolerance = case.variant.relative_tolerance();
    let wrong = out_of_tolerance(&actual.output, &want, 0, tolerance);
    assert_eq!(wrong, 0, "{}: {wrong} of {} outputs differ from the host-decoded f64 reduction", case.tensor, want.len());
    let neighbour = out_of_tolerance(&actual.output, &want, 1, tolerance);
    assert!(neighbour > want.len() * 9 / 10, "{}: control: only {neighbour} outputs left tolerance against the neighbouring row, the check does not discriminate", case.tensor);
}

#[test]
fn iq4xs_gate_row_blocked_matvec_matches_the_host_decode() {
    assert_row_blocked_matches_host_decode(&GATE);
}

#[test]
fn iq4xs_down_row_blocked_matvec_matches_the_host_decode() {
    assert_row_blocked_matches_host_decode(&DOWN);
}

#[test]
fn iq4xs_gate_row_blocked_matvec_matches_the_host_decode_across_tokens() {
    assert_row_blocked_matches_host_decode(&GATE_BATCH);
}

#[test]
fn iq4xs_down_row_blocked_matvec_matches_the_host_decode_across_tokens() {
    assert_row_blocked_matches_host_decode(&DOWN_BATCH);
}

#[test]
fn iq4xs_half_precision_reduce_declines_the_row_blocked_kernel_and_matches_the_host_decode() {
    assert_row_blocked_matches_host_decode(&GATE_HALF);
}

#[test]
fn iq4xs_negated_product_reduce_declines_the_row_blocked_kernel_and_matches_the_host_decode() {
    assert_row_blocked_matches_host_decode(&GATE_NEGATED);
}

fn worst_distance(output: &[f32], want: &[(f64, f64)]) -> f64 {
    output.iter().zip(want).map(|(got, (reduced, magnitude))| (f64::from(*got) - reduced).abs() / magnitude.max(1.0e-9)).fold(0.0f64, f64::max)
}

/// The cooperative per-element kernel (`PROXIMA_PACKED_ROW_DISABLE` keeps the codec off the row-blocked matvec) on the
/// same inputs and the same f64 reference: the rendered matvec must be no further from the reference than it.
#[test]
fn iq4xs_row_blocked_matvec_is_no_further_from_the_host_decode_than_the_cooperative_kernel() {
    let slab = read_slab(&GATE_BATCH);
    let activation = activation(slab.reduction, &GATE_BATCH);
    let want = reference(&GATE_BATCH, &slab, &activation);
    let cooperative = temp_env::with_var("PROXIMA_PACKED_ROW_DISABLE", Some("iq4xs"), || run(&GATE_BATCH, &slab, &activation));
    assert!(cooperative.kernel_keys.iter().all(|key| !key.contains("_cpfB")), "the switch must keep the codec off the row-blocked kernel: {:?}", cooperative.kernel_keys);
    let rendered = temp_env::with_var("PROXIMA_PACKED_ROW_DISABLE", None::<&str>, || run(&GATE_BATCH, &slab, &activation));
    assert!(rendered.kernel_keys.iter().any(|key| key.contains("_cpfB")), "the row-blocked kernel was never selected: {:?}", rendered.kernel_keys);
    let (rendered_worst, cooperative_worst) = (worst_distance(&rendered.output, &want), worst_distance(&cooperative.output, &want));
    assert!(rendered_worst <= cooperative_worst, "rendered {rendered_worst:e} is further from the f64 reduction than cooperative {cooperative_worst:e}");
}

const BIT_FIXTURE_NAME: &str = "row_blocked_iq4xs_blk19_gate_exps_t1.u32le";
/// the row-blocked output bits of `GATE` (576 f32, little-endian u32) captured from the kernel before level-table staging.
const BIT_FIXTURE: &[u8] = include_bytes!("fixtures/row_blocked_iq4xs_blk19_gate_exps_t1.u32le");

fn bit_mismatches(output: &[f32], fixture: &[u8]) -> usize {
    let (words, _) = fixture.as_chunks::<4>();
    output.iter().zip(words).filter(|(value, word)| value.to_bits() != u32::from_le_bytes(**word)).count()
}

/// Tolerance tests cannot see a one-bit change; this pins the exact f32 bits of the unmodified kernel.
#[test]
fn iq4xs_gate_row_blocked_matvec_is_bit_identical_across_runs_and_to_the_vendored_fixture() {
    let slab = read_slab(&GATE);
    let activation = activation(slab.reduction, &GATE);
    let first = run(&GATE, &slab, &activation);
    let second = run(&GATE, &slab, &activation);
    assert!(first.kernel_keys.iter().any(|key| key.contains("_cpfB")), "the row-blocked kernel was never selected: {:?}", first.kernel_keys);
    let total = GATE.tokens * SELECTED * ROWS;
    assert_eq!(first.output.len(), total, "one f32 per selected expert and row");
    assert_eq!(second.output.len(), total, "one f32 per selected expert and row");
    let run_to_run = first.output.iter().zip(&second.output).filter(|(left, right)| left.to_bits() != right.to_bits()).count();
    println!("run_to_run mismatches={run_to_run} of {total}");
    assert_eq!(run_to_run, 0, "a kernel that is not bit-stable run to run cannot carry a bit-identity gate");

    if BIT_FIXTURE.is_empty() {
        let directory = tempfile::Builder::new().prefix("iq4xs_row_blocked_bits").tempdir().expect("scratch tempdir for the captured bits").keep();
        let path = directory.join(BIT_FIXTURE_NAME);
        let bytes: Vec<u8> = first.output.iter().flat_map(|value| value.to_bits().to_le_bytes()).collect();
        std::fs::write(&path, &bytes).expect("the captured bits are written to the scratch directory");
        panic!("no vendored fixture: wrote {} bytes ({total} f32) to {}; copy it to omega/tests/fixtures/{BIT_FIXTURE_NAME}", bytes.len(), path.display());
    }
    assert_eq!(BIT_FIXTURE.len(), total * 4, "the fixture holds one little-endian u32 per output element");
    let mismatches = bit_mismatches(&first.output, BIT_FIXTURE);
    println!("fixture mismatches={mismatches} of {total}");
    assert_eq!(mismatches, 0, "{mismatches} of {total} output bits differ from the pre-staging fixture");

    let mut flipped = BIT_FIXTURE.to_vec();
    flipped[0] ^= 1;
    assert_eq!(bit_mismatches(&first.output, &flipped), 1, "control: one flipped fixture bit must be counted exactly once");
}
