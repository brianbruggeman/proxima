//! IQ4_XS and IQ3_S expert projections on real flint tensors, through the expert-grouped tiled GEMM
//! (decoded from `BlockLayout::IQ4_XS` / `IQ3_S` and their `LevelTable`s by
//! `layout_render::level_block_element_msl`) against the cooperative per-element kernel those codecs
//! ran on before they had a `tiled_decode` description.
//!
//! The fixtures hold the raw f32 bits the cooperative kernel produced at proxima main 3a220af2
//! (hand `IQ4_XS_UNPACK_MSL` / `IQ3_S_UNPACK_MSL`); `fixtures/level_codec_gemm_PROVENANCE.txt` records how
//! they were captured. Setting `PROXNUM77_CAPTURE_DIR` writes the bits instead of comparing (the capture
//! run). Three claims, each its own test:
//! - the cooperative kernel, now rendered from the description (`PROXIMA_TILED_GEMM_DISABLE` keeps the
//!   codec off the tiled path), reproduces the hand-written kernel's bits exactly;
//! - the tiled kernel is selected for the codec, and agrees with the cooperative bits to the half
//!   precision the staged weight tile holds (bit equality is counted and printed, not asserted: the tile is
//!   `half` and the multiply is a matrix op, so the f32 fold order differs by construction);
//! - a control: the tiled result compared against another tensor's cooperative bits must fail.
#![cfg(all(feature = "metal", target_os = "macos"))]
// workspace lints deny unwrap/expect (Cargo.toml:93-94) and an integration test binary is not cfg(test), so a failed fixture read must be an allowed panic.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

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
const TOKENS: usize = 84;
const SELECTED: usize = 6;
const EXPERTS: usize = 16;
const ROWS: usize = 96;
const HALF_RELATIVE_TOLERANCE: f32 = 2.0e-3;

#[derive(Clone, Copy)]
struct Case {
    fixture: &'static str,
    tensor: &'static str,
    ggml_type: GgmlType,
    codec: Codec,
    disable_token: &'static str,
    per_selected_activation: bool,
}

const CASES: [Case; 4] = [
    Case {
        fixture: "level_gemm_iq4xs_blk19_ffn_gate_exps.f32le",
        tensor: "blk.19.ffn_gate_exps.weight",
        ggml_type: GgmlType::Iq4Xs,
        codec: Codec::Iq4Xs,
        disable_token: "iq4xs",
        per_selected_activation: false,
    },
    Case {
        fixture: "level_gemm_iq4xs_blk19_ffn_up_exps.f32le",
        tensor: "blk.19.ffn_up_exps.weight",
        ggml_type: GgmlType::Iq4Xs,
        codec: Codec::Iq4Xs,
        disable_token: "iq4xs",
        per_selected_activation: false,
    },
    Case {
        fixture: "level_gemm_iq4xs_blk19_ffn_down_exps.f32le",
        tensor: "blk.19.ffn_down_exps.weight",
        ggml_type: GgmlType::Iq4Xs,
        codec: Codec::Iq4Xs,
        disable_token: "iq4xs",
        per_selected_activation: true,
    },
    Case {
        fixture: "level_gemm_iq3s_blk2_ffn_gate_exps.f32le",
        tensor: "blk.2.ffn_gate_exps.weight",
        ggml_type: GgmlType::Iq3S,
        codec: Codec::Iq3S,
        disable_token: "iq3s",
        per_selected_activation: false,
    },
];

fn block_bytes(codec: Codec) -> usize {
    match codec {
        Codec::Iq4Xs => proxima_gguf::quant::iq4_xs::BLOCK_BYTES,
        Codec::Iq3S => proxima_gguf::quant::iq3_s::BLOCK_BYTES,
        other => panic!("{other:?} is not a codec this test covers"),
    }
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
}

/// The first `ROWS` rows of each of the first `EXPERTS` experts of `case.tensor`, concatenated in expert order.
fn read_slab(case: &Case) -> Slab {
    let path = proxima_gguf::checkpoint_resolver::require_checkpoint("PROXIMA_LEVEL_CODEC_PARITY_GGUF", FLINT_PATH, 0);
    let mut file = File::open(&path).expect("the checkpoint opens");
    let file_len = file.metadata().unwrap().len();
    let parsed = header(&mut file, file_len);
    let tensor = parsed.tensors.iter().find(|tensor| tensor.name == case.tensor).expect("the tensor is in the directory");
    assert_eq!(tensor.ggml_type, case.ggml_type, "{} has the expected quantization", case.tensor);
    let reduction = tensor.dims[0] as usize;
    let rows_per_expert = tensor.dims[1] as usize;
    let row_bytes = reduction / QK_K * block_bytes(case.codec);
    let range = parsed.tensor_data_range(tensor, file_len).unwrap();
    let mut bytes = vec![0u8; EXPERTS * ROWS * row_bytes];
    for (expert, target) in bytes.chunks_exact_mut(ROWS * row_bytes).enumerate() {
        file.seek(SeekFrom::Start(range.start + (expert * rows_per_expert * row_bytes) as u64)).unwrap();
        file.read_exact(target).unwrap();
    }
    Slab { bytes, reduction }
}

/// Each token names `SELECTED` distinct consecutive experts starting at a token-dependent offset, so the load is skewed the way a router's is.
fn route() -> Vec<f32> {
    (0..TOKENS * SELECTED).map(|pair| ((pair / SELECTED) * 5 % 7 + pair % SELECTED) as f32).collect()
}

/// One hidden-state row per token; a down projection gets its own row per selected expert, a gate or up projection shares the token's row.
fn activation(reduction: usize, per_selected: bool) -> Vec<f32> {
    let mut lcg = Lcg(ACTIVATION_SEED);
    let rows = if per_selected { TOKENS * SELECTED } else { TOKENS };
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

fn program(reduction: usize, per_selected: bool) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = input(&mut program, DType::UINT8, &[EXPERTS, ROWS, reduction]);
    let route = input(&mut program, DType::FLOAT32, &[TOKENS, SELECTED]);
    let (activation, activation_map) = if per_selected {
        (input(&mut program, DType::FLOAT32, &[TOKENS, SELECTED, reduction]), map::projection(4, &[0, 1, 3]))
    } else {
        (input(&mut program, DType::FLOAT32, &[TOKENS, reduction]), map::projection(4, &[0, 3]))
    };
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![(weight, expert_gather_map(route)), (activation, IndexMap::Affine(activation_map))],
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

fn run(case: &Case, slab: &Slab) -> Run {
    run_with(case, slab, &activation(slab.reduction, case.per_selected_activation))
}

fn run_with(case: &Case, slab: &Slab, activation: &[f32]) -> Run {
    let (program, output) = program(slab.reduction, case.per_selected_activation);
    let route = route();
    let blocks = [QuantizedBlock::Packed { codec: case.codec, bytes: &slab.bytes }, QuantizedBlock::Float32(&route), QuantizedBlock::Float32(activation)];
    let outputs = [output];
    let plan = omega::plan(&program, &[], &blocks, &outputs, NumericPolicy::default()).expect("metal plans the expert product");
    let kernel_keys = plan.kernel_keys().expect("the plan exposes its kernel keys");
    let evaluated = omega::execute_plan(&plan, &blocks).expect("metal runs the expert product");
    Run { output: evaluated.root().to_vec(), kernel_keys }
}

fn cooperative(case: &Case, slab: &Slab) -> Run {
    temp_env::with_var("PROXIMA_TILED_GEMM_DISABLE", Some(case.disable_token), || run(case, slab))
}

fn tiled(case: &Case, slab: &Slab) -> Run {
    temp_env::with_var("PROXIMA_TILED_GEMM_DISABLE", None::<&str>, || run(case, slab))
}

fn grouped_tiled_kernel_ran(keys: &[String]) -> bool {
    let bodies = ["F_w128", "F1_w128", "FN_w128"];
    keys.iter().any(|key| key.contains("_g10") && bodies.iter().any(|body| key.contains(body)))
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn fixture_bits(case: &Case) -> Vec<u32> {
    let bytes = std::fs::read(fixture_path(case.fixture)).expect("the fixture is committed");
    bytes.as_chunks::<4>().0.iter().map(|word| u32::from_le_bytes(*word)).collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn worst_relative_row_error(want: &[u32], got: &[f32]) -> f32 {
    want.as_chunks::<ROWS>()
        .0
        .iter()
        .zip(got.as_chunks::<ROWS>().0.iter())
        .map(|(want_row, got_row)| {
            let want_row: Vec<f32> = want_row.iter().map(|bits| f32::from_bits(*bits)).collect();
            let norm = want_row.iter().map(|value| value * value).sum::<f32>().sqrt();
            let worst = want_row.iter().zip(got_row).map(|(want, got)| (want - got).abs()).fold(0.0f32, f32::max);
            worst / norm.max(1.0e-6)
        })
        .fold(0.0f32, f32::max)
}

fn capture_requested(case: &Case, slab: &Slab) -> bool {
    let Some(directory) = std::env::var_os("PROXNUM77_CAPTURE_DIR") else {
        return false;
    };
    let output = cooperative(case, slab).output;
    let bytes: Vec<u8> = bits(&output).iter().flat_map(|word| word.to_le_bytes()).collect();
    std::fs::write(std::path::Path::new(&directory).join(case.fixture), bytes).expect("the capture writes");
    true
}

fn assert_rendered_cooperative_equals_hand(case: &Case) {
    let slab = read_slab(case);
    if capture_requested(case, &slab) {
        return;
    }
    let expected = fixture_bits(case);
    let actual = cooperative(case, &slab);
    assert!(!grouped_tiled_kernel_ran(&actual.kernel_keys),"{}: the cooperative control must not run the tiled kernel: {:?}", case.tensor, actual.kernel_keys);
    assert_eq!(expected.len(), TOKENS * SELECTED * ROWS, "the fixture holds one f32 per token, selected expert and row");
    let mismatches = bits(&actual.output).iter().zip(&expected).filter(|(got, want)| got != want).count();
    assert_eq!(mismatches, 0, "{}: {mismatches} of {} cooperative f32 outputs differ from the hand-written kernel's bits", case.tensor, expected.len());
}

fn assert_tiled_agrees_with_cooperative(case: &Case) {
    let slab = read_slab(case);
    if std::env::var_os("PROXNUM77_CAPTURE_DIR").is_some() {
        return;
    }
    let expected = fixture_bits(case);
    let actual = tiled(case, &slab);
    assert!(grouped_tiled_kernel_ran(&actual.kernel_keys), "{}: the expert-grouped tiled kernel was never selected: {:?}", case.tensor, actual.kernel_keys);
    let identical = bits(&actual.output).iter().zip(&expected).filter(|(got, want)| got == want).count();
    let worst = worst_relative_row_error(&expected, &actual.output);
    eprintln!("{}: elements={} bit_identical={identical} worst_relative_row_error={worst:e}", case.tensor, expected.len());
    assert!(worst <= HALF_RELATIVE_TOLERANCE, "{}: tiled result drifted from the cooperative bits by {worst:e} of a row's norm", case.tensor);
}

#[test]
fn iq4xs_gate_rendered_cooperative_kernel_equals_the_hand_kernel_bit_for_bit() {
    assert_rendered_cooperative_equals_hand(&CASES[0]);
}

#[test]
fn iq4xs_up_rendered_cooperative_kernel_equals_the_hand_kernel_bit_for_bit() {
    assert_rendered_cooperative_equals_hand(&CASES[1]);
}

#[test]
fn iq4xs_down_rendered_cooperative_kernel_equals_the_hand_kernel_bit_for_bit() {
    assert_rendered_cooperative_equals_hand(&CASES[2]);
}

#[test]
fn iq3s_gate_rendered_cooperative_kernel_equals_the_hand_kernel_bit_for_bit() {
    assert_rendered_cooperative_equals_hand(&CASES[3]);
}

#[test]
fn iq4xs_gate_tiled_gemm_agrees_with_the_cooperative_bits() {
    assert_tiled_agrees_with_cooperative(&CASES[0]);
}

#[test]
fn iq4xs_up_tiled_gemm_agrees_with_the_cooperative_bits() {
    assert_tiled_agrees_with_cooperative(&CASES[1]);
}

#[test]
fn iq4xs_down_tiled_gemm_agrees_with_the_cooperative_bits() {
    assert_tiled_agrees_with_cooperative(&CASES[2]);
}

#[test]
fn iq3s_gate_tiled_gemm_agrees_with_the_cooperative_bits() {
    assert_tiled_agrees_with_cooperative(&CASES[3]);
}

const FLINT_GATED_ACTIVATION_ROWS: &str = include_str!("fixtures/flint_blk46_expert_gated_activation_rows.txt");
const HALF_MAX: f32 = 65504.0;

/// The two 512-wide rows flint's block-46 gated expert product handed its down projection for prompt token 2 (selected slots 0 and 1).
fn flint_gated_activation_rows() -> Vec<Vec<f32>> {
    FLINT_GATED_ACTIVATION_ROWS
        .lines()
        .map(|line| {
            line.trim()
                .trim_start_matches('[')
                .trim_end_matches(']')
                .split(',')
                .map(|value| value.trim().parse::<f32>().expect("a captured f32"))
                .collect()
        })
        .collect()
}

#[test]
fn iq4_xs_down_projection_activation_past_the_half_range_stays_finite_and_matches_the_cooperative_kernel() {
    if std::env::var_os("PROXNUM77_CAPTURE_DIR").is_some() {
        return;
    }
    let case = &CASES[2];
    let slab = read_slab(case);
    let captured = flint_gated_activation_rows();
    assert!(captured.iter().all(|row| row.len() == slab.reduction && row.iter().any(|value| value.abs() > HALF_MAX)), "degenerate: the fixture rows must overflow half");
    let activation: Vec<f32> = (0..TOKENS * SELECTED)
        .flat_map(|index| {
            let scale = if index % 3 == 0 { 1.0 } else { 1.0 / 64.0 };
            captured[index % 2].iter().map(move |value| value * scale)
        })
        .collect();
    let tiled_run = temp_env::with_var("PROXIMA_TILED_GEMM_DISABLE", None::<&str>, || run_with(case, &slab, &activation));
    let cooperative_run = temp_env::with_var("PROXIMA_TILED_GEMM_DISABLE", Some(case.disable_token), || run_with(case, &slab, &activation));
    assert!(grouped_tiled_kernel_ran(&tiled_run.kernel_keys), "the tiled kernel must run: {:?}", tiled_run.kernel_keys);
    let non_finite = tiled_run.output.iter().filter(|value| !value.is_finite()).count();
    assert_eq!(non_finite, 0, "the tiled kernel produced non-finite outputs from finite activations");
    let worst = worst_relative_row_error(&bits(&cooperative_run.output), &tiled_run.output);
    assert!(worst <= HALF_RELATIVE_TOLERANCE, "tiled drifted from the cooperative kernel by {worst:e} of a row's norm");
}

#[test]
fn control_one_flipped_scale_byte_changes_only_its_row_of_the_tiled_result() {
    if std::env::var_os("PROXNUM77_CAPTURE_DIR").is_some() {
        return;
    }
    let case = &CASES[0];
    let slab = read_slab(case);
    let clean = tiled(case, &slab);
    let mut flipped = Slab { bytes: slab.bytes.clone(), reduction: slab.reduction };
    // byte 1 is the high byte of the first block's f16 super-block scale, row 0 of expert 0
    flipped.bytes[1] ^= 0x20;
    let changed = tiled(case, &flipped);
    let differing_rows: Vec<usize> = (0..clean.output.len())
        .filter(|&at| clean.output[at].to_bits() != changed.output[at].to_bits())
        .map(|at| at % ROWS)
        .collect();
    assert!(!differing_rows.is_empty(), "flipping a scale byte must change the result: the control saw no effect");
    assert!(differing_rows.iter().all(|&row| row == 0), "only the flipped row may change, saw rows {differing_rows:?}");
}
