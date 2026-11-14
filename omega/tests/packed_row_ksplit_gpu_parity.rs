//! The K-split packed matvec (`PROXIMA_PACKED_K_SPLIT`) on the Metal GPU, over real flint tensors at
//! the single-token shape decode runs. Only the association of the per-block sums changes with the
//! split, so each split-k output must stay inside the f64 host-reduction tolerance of the decoded rows,
//! and must stay inside that same tolerance of the split-1 output of the same op. Bit equality across
//! splits is never asserted; the max absolute and relative difference is printed per case.
//! Its own binary so the env mutation cannot race any other test's plan.
#![cfg(all(
    feature = "metal",
    target_os = "macos",
    not(feature = "metal-q4k-mask-fma")
))]
// workspace lints deny unwrap/expect (Cargo.toml:93-94) and an integration test binary is not cfg(test), so a failed fixture read must be an allowed panic.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use proxima_gguf::parser::{GgufEvent, GgufParser};
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::TensorInfo;
use proxima_gguf::types::GgmlType;
use proxima_numeric::codec::Codec;
use proxima_numeric::{DType, NumericPolicy};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp, append, map};

const FLINT_PATH: &str = "/Users/brianbruggeman/models/llm/flint-gguf/flint-UD-Q4_K_XL.gguf";
const K_SPLIT_ENV: &str = "PROXIMA_PACKED_K_SPLIT";
const ACTIVATION_SEED: u64 = 2026;
const RELATIVE_TOLERANCE: f64 = 1.0e-6;
const ABSOLUTE_TOLERANCE: f64 = 1.0e-6;
const SIMD_WIDTH: usize = 32;
const Q8_LAYER: usize = 10;

#[derive(Clone, Copy, Debug)]
enum Target {
    Q8AttnQ,
    Q8AttnOutput,
    Bf16Router,
}

struct Weight {
    name: String,
    codec: Codec,
    bytes: Vec<u8>,
    reduction: usize,
    rows: usize,
    unit_elements: usize,
}

struct Run {
    keys: Vec<String>,
    values: Vec<f32>,
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

fn is_matvec(tensor: &TensorInfo, ggml_type: GgmlType, reduction: u64, rows: u64) -> bool {
    tensor.ggml_type == ggml_type && tensor.dims.len() == 2 && tensor.dims[0] == reduction && tensor.dims[1] == rows
}

/// Pinned by name; the llama-gguf tensor listing sizes `blk.10.attn_q.weight` and `blk.10.attn_output.weight`
/// at 8912896 bytes, which is 4096 x 64 and 2048 x 128 blocks of 34 bytes (Q8_0), and `blk.2.ffn_gate_inp.weight` at 1048576 = 256 x 2048 x 2.
fn locate(parsed: &ParsedGguf, target: Target) -> &TensorInfo {
    let (name, ggml_type, reduction, rows) = match target {
        Target::Q8AttnQ => (format!("blk.{Q8_LAYER}.attn_q.weight"), GgmlType::Q8_0, 2048, 4096),
        Target::Q8AttnOutput => (format!("blk.{Q8_LAYER}.attn_output.weight"), GgmlType::Q8_0, 4096, 2048),
        Target::Bf16Router => ("blk.2.ffn_gate_inp.weight".to_string(), GgmlType::Bf16, 2048, 256),
    };
    let tensor = parsed.tensors.iter().find(|tensor| tensor.name == name).unwrap_or_else(|| panic!("{name} is not in the flint directory"));
    assert!(is_matvec(tensor, ggml_type, reduction, rows), "{name} is {:?} {:?}, expected {ggml_type:?} {reduction} -> {rows}", tensor.ggml_type, tensor.dims);
    tensor
}

fn read_weight(target: Target) -> Weight {
    let path = proxima_gguf::checkpoint_resolver::require_checkpoint("PROXIMA_LEVEL_CODEC_PARITY_GGUF", FLINT_PATH, 0);
    let mut file = File::open(&path).expect("the checkpoint opens");
    let file_len = file.metadata().unwrap().len();
    let parsed = header(&mut file, file_len);
    let tensor = locate(&parsed, target);
    let (codec, unit_elements, unit_bytes) = match tensor.ggml_type {
        GgmlType::Q8_0 => (Codec::Q8_0, proxima_gguf::quant::q8_0::QK8_0, proxima_gguf::quant::q8_0::BLOCK_BYTES),
        GgmlType::Bf16 => (Codec::BFloat16, 1, proxima_gguf::quant::bf16::BLOCK_BYTES),
        other => panic!("{} is {other:?}, not a codec this parity file covers", tensor.name),
    };
    let reduction = tensor.dims[0] as usize;
    let rows = tensor.dims[1] as usize;
    let range = parsed.tensor_data_range(tensor, file_len).unwrap();
    let mut bytes = vec![0u8; rows * (reduction / unit_elements) * unit_bytes];
    file.seek(SeekFrom::Start(range.start)).unwrap();
    file.read_exact(&mut bytes).unwrap();
    Weight { name: tensor.name.clone(), codec, bytes, reduction, rows, unit_elements: if unit_elements == 1 { 8 } else { unit_elements } }
}

fn activation(reduction: usize) -> Vec<f32> {
    let mut lcg = Lcg(ACTIVATION_SEED);
    (0..reduction).map(|_| lcg.next_unit() * 4.0 - 2.0).collect()
}

fn program(weight: &Weight) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weights = append(&mut program, Op::Input { dtype: DType::UINT8, shape: vec![Extent::Static(weight.rows as u32), Extent::Static(weight.reduction as u32)], name: None });
    let activation = append(&mut program, Op::Input { dtype: DType::FLOAT32, shape: vec![Extent::Static(1), Extent::Static(weight.reduction as u32)], name: None });
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![(weights, IndexMap::Affine(map::projection(3, &[1, 2]))), (activation, IndexMap::Affine(map::projection(3, &[0, 2])))],
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
            name: Some("ksplit_matvec".into()),
        }),
    );
    (program, sum)
}

fn decode_row(weight: &Weight, row: usize, values: &mut [f32]) {
    let row_bytes = weight.bytes.len() / weight.rows;
    let bytes = &weight.bytes[row * row_bytes..(row + 1) * row_bytes];
    match weight.codec {
        Codec::Q8_0 => proxima_gguf::quant::q8_0::dequantize(bytes, values).expect("the gguf dequantizer reads the q8_0 row"),
        Codec::BFloat16 => proxima_gguf::quant::bf16::dequantize(bytes, values).expect("the gguf dequantizer reads the bf16 row"),
        other => panic!("no host decoder wired for {other:?}"),
    }
}

/// `(reduction of the host-decoded row against the activation, sum of the absolute terms)` per row.
fn reference(weight: &Weight, activation: &[f32]) -> Vec<(f64, f64)> {
    let mut values = vec![0.0f32; weight.reduction];
    (0..weight.rows)
        .map(|row| {
            decode_row(weight, row, &mut values);
            values.iter().zip(activation).map(|(value, act)| f64::from(*value) * f64::from(*act)).fold((0.0f64, 0.0f64), |(sum, magnitude), term| (sum + term, magnitude + term.abs()))
        })
        .collect()
}

fn run_under_split(weight: &Weight, activation: &[f32], split: &str) -> Run {
    temp_env::with_vars([(K_SPLIT_ENV, Some(split))], || {
        let (program, output) = program(weight);
        let blocks = [QuantizedBlock::Packed { codec: weight.codec, bytes: &weight.bytes }, QuantizedBlock::Float32(activation)];
        let outputs = [output];
        let plan = omega::plan(&program, &[], &blocks, &outputs, NumericPolicy::default()).expect("metal plans the matvec");
        let keys = plan.kernel_keys().expect("the plan exposes its kernel keys");
        let evaluated = omega::execute_plan(&plan, &blocks).expect("metal runs the matvec");
        Run { keys, values: evaluated.root().to_vec() }
    })
}

fn out_of_tolerance(got: &[f32], want: &[(f64, f64)], shift: usize) -> usize {
    got.iter()
        .enumerate()
        .filter(|(at, value)| {
            let (reduced, magnitude) = want[(at + shift) % want.len()];
            (f64::from(**value) - reduced).abs() > RELATIVE_TOLERANCE * magnitude + ABSOLUTE_TOLERANCE
        })
        .count()
}

/// split the plan can feed: one simdgroup per `SIMD_WIDTH` units of the row, rounded down to a power of two, capped by the request.
fn expected_split(weight: &Weight, requested: usize) -> usize {
    let units = weight.reduction / weight.unit_elements;
    let feedable = units.div_ceil(SIMD_WIDTH).max(1);
    let capped = feedable.min(requested);
    1usize << capped.ilog2()
}

fn assert_ksplit_parity(target: Target, split: &str) {
    let weight = read_weight(target);
    let requested: usize = split.parse().expect("the split is an integer");
    let expected = expected_split(&weight, requested);
    let activation = activation(weight.reduction);
    let want = reference(&weight, &activation);

    let baseline = run_under_split(&weight, &activation, "1");
    let split_run = run_under_split(&weight, &activation, split);

    assert!(baseline.keys.iter().all(|key| !key.contains("_ks")), "the split-1 baseline carries a split token: {:?}", baseline.keys);
    assert!(expected > 1, "{} cannot feed a split of {requested}; the case would not exercise a split", weight.name);
    let token = format!("_ks{expected}");
    assert!(split_run.keys.iter().any(|key| key.contains(&token)), "{}: no `{token}` kernel was selected for split {split}: {:?}", weight.name, split_run.keys);
    assert_eq!(split_run.values.len(), want.len(), "one f32 per row");
    assert!(split_run.values.iter().chain(&baseline.values).all(|value| value.is_finite()), "a non-finite output would pass the tolerance comparison vacuously");

    let first_failure = split_run.values.iter().zip(&want).enumerate().find(|(_, (got, (reduced, magnitude)))| (f64::from(**got) - reduced).abs() > RELATIVE_TOLERANCE * magnitude + ABSOLUTE_TOLERANCE);
    let (ref_abs, ref_rel) = split_run.values.iter().zip(&want).fold((0.0f64, 0.0f64), |(max_abs, max_rel), (got, (reduced, _))| {
        let diff = (f64::from(*got) - reduced).abs();
        (max_abs.max(diff), max_rel.max(diff / reduced.abs().max(f64::MIN_POSITIVE)))
    });
    eprintln!("ksplit_parity tensor={} split={split} max_abs_vs_f64={ref_abs:e} max_rel_vs_f64={ref_rel:e} first_failure={first_failure:?}", weight.name);
    let wrong = out_of_tolerance(&split_run.values, &want, 0);
    assert_eq!(wrong, 0, "{}: {wrong} of {} split-{expected} outputs differ from the host-decoded f64 reduction; first failure (row, (got, (f64 reduction, magnitude))): {first_failure:?}", weight.name, want.len());
    let neighbour = out_of_tolerance(&split_run.values, &want, 1);
    assert!(neighbour > want.len() * 9 / 10, "control: only {neighbour} outputs left tolerance against the neighbouring row, the check does not discriminate");

    let (max_abs, max_rel, outside) = split_run.values.iter().zip(&baseline.values).zip(&want).fold((0.0f64, 0.0f64, 0usize), |(max_abs, max_rel, outside), ((got, base), (_, magnitude))| {
        let diff = (f64::from(*got) - f64::from(*base)).abs();
        let relative = diff / f64::from(*base).abs().max(f64::MIN_POSITIVE);
        (max_abs.max(diff), max_rel.max(relative), outside + usize::from(diff > RELATIVE_TOLERANCE * magnitude + ABSOLUTE_TOLERANCE))
    });
    eprintln!("ksplit_parity tensor={} codec={:?} rows={} k={} requested={requested} effective={expected} keys={:?} max_abs_vs_split1={max_abs:e} max_rel_vs_split1={max_rel:e}", weight.name, weight.codec, weight.rows, weight.reduction, split_run.keys);
    assert_eq!(outside, 0, "{}: {outside} split-{expected} outputs differ from the split-1 output by more than the f64-reference tolerance (max abs {max_abs:e})", weight.name);
}

fn assert_whole_block_lane_never_splits(target: Target, split: &str) {
    let weight = read_weight(target);
    let activation = activation(weight.reduction);

    let baseline = run_under_split(&weight, &activation, "1");
    let requested = run_under_split(&weight, &activation, split);

    assert!(requested.keys.iter().all(|key| !key.contains("_ks")), "{}: a whole-block lane took a split under {split}: {:?}", weight.name, requested.keys);
    assert_eq!(requested.keys, baseline.keys, "{}: the kernel identity moved under split {split}", weight.name);
    assert_eq!(requested.values.len(), weight.rows, "one f32 per row");
    assert!(requested.values.iter().all(|value| value.is_finite()), "a non-finite output would pass the bit comparison vacuously");
    let differing = requested.values.iter().zip(&baseline.values).filter(|(got, base)| got.to_bits() != base.to_bits()).count();
    assert_eq!(differing, 0, "{}: {differing} of {} outputs differ bit for bit from split 1", weight.name, weight.rows);
}

#[proxima::test]
#[case::q8_0_attn_q_split_2("q8_attn_q", "2")]
#[case::q8_0_attn_output_split_2("q8_attn_output", "2")]
#[case::q8_0_attn_output_split_4("q8_attn_output", "4")]
async fn a_whole_block_lane_ignores_the_split_and_matches_split_1_bit_for_bit(#[case] tensor: &str, #[case] split: &str) {
    let target = match tensor {
        "q8_attn_q" => Target::Q8AttnQ,
        "q8_attn_output" => Target::Q8AttnOutput,
        other => panic!("unknown target {other}"),
    };
    assert_whole_block_lane_never_splits(target, split);
}

#[proxima::test]
#[case::bf16_router_split_2("2")]
#[case::bf16_router_split_4("4")]
async fn split_k_matvec_matches_the_host_decode_and_the_unsplit_output(#[case] split: &str) {
    assert_ksplit_parity(Target::Bf16Router, split);
}
