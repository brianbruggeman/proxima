//! The Q8_0 lane-run matvec (`PROXIMA_PACKED_LANE_RUN_BYTES=8`) on the Metal GPU, over real flint
//! tensors at the single-token shape decode runs, under the policy flint serves with
//! (`NumericPolicy::llama_relaxed()`, proxima-model-interop/src/serving.rs:1940). The run only moves
//! where each lane reads and when the scale applies, so the association of the sums changes and the
//! output must stay inside the f64 host-reduction tolerance; bit equality with the switch off is
//! never asserted, the max absolute and relative difference is printed per tensor.
//! Its own binary so the env mutation cannot race any other test's plan.
#![cfg(all(
    feature = "metal",
    target_os = "macos",
    not(feature = "metal-q4k-mask-fma")
))]
// workspace lints deny unwrap/expect and an integration test binary is not cfg(test), so a failed fixture read must be an allowed panic.
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
const LANE_RUN_ENV: &str = "PROXIMA_PACKED_LANE_RUN_BYTES";
const ACTIVATION_SEED: u64 = 2026;
const RELATIVE_TOLERANCE: f64 = 1.0e-6;
const ABSOLUTE_TOLERANCE: f64 = 1.0e-6;
const Q8_LAYER: usize = 10;

#[derive(Clone, Copy, Debug)]
enum Target {
    AttnQ,
    AttnOutput,
    AttnK,
}

struct Weight {
    name: String,
    codec: Codec,
    bytes: Vec<u8>,
    reduction: usize,
    rows: usize,
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

fn locate(parsed: &ParsedGguf, target: Target) -> &TensorInfo {
    let (name, reduction, rows) = match target {
        Target::AttnQ => (format!("blk.{Q8_LAYER}.attn_q.weight"), 2048, 4096),
        Target::AttnOutput => (format!("blk.{Q8_LAYER}.attn_output.weight"), 4096, 2048),
        Target::AttnK => (format!("blk.{Q8_LAYER}.attn_k.weight"), 2048, 512),
    };
    let tensor = parsed.tensors.iter().find(|tensor| tensor.name == name).unwrap_or_else(|| panic!("{name} is not in the flint directory"));
    assert!(is_matvec(tensor, GgmlType::Q8_0, reduction, rows), "{name} is {:?} {:?}, expected Q8_0 {reduction} -> {rows}", tensor.ggml_type, tensor.dims);
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
        other => panic!("{} is {other:?}, not a codec this parity file covers", tensor.name),
    };
    let reduction = tensor.dims[0] as usize;
    let rows = tensor.dims[1] as usize;
    let range = parsed.tensor_data_range(tensor, file_len).unwrap();
    let mut bytes = vec![0u8; rows * (reduction / unit_elements) * unit_bytes];
    file.seek(SeekFrom::Start(range.start)).unwrap();
    file.read_exact(&mut bytes).unwrap();
    Weight { name: tensor.name.clone(), codec, bytes, reduction, rows }
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
            name: Some("lane_run_matvec".into()),
        }),
    );
    (program, sum)
}

fn decode_row(weight: &Weight, row: usize, values: &mut [f32]) {
    let row_bytes = weight.bytes.len() / weight.rows;
    let bytes = &weight.bytes[row * row_bytes..(row + 1) * row_bytes];
    match weight.codec {
        Codec::Q8_0 => proxima_gguf::quant::q8_0::dequantize(bytes, values).expect("the gguf dequantizer reads the q8_0 row"),
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

fn run_under_lane_run(weight: &Weight, activation: &[f32], lane_run: Option<&str>) -> Run {
    temp_env::with_vars([(K_SPLIT_ENV, Some("1")), (LANE_RUN_ENV, lane_run)], || {
        let (program, output) = program(weight);
        let blocks = [QuantizedBlock::Packed { codec: weight.codec, bytes: &weight.bytes }, QuantizedBlock::Float32(activation)];
        let outputs = [output];
        let plan = omega::plan(&program, &[], &blocks, &outputs, NumericPolicy::llama_relaxed()).expect("metal plans the matvec");
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

fn assert_lane_run_parity(target: Target) {
    let weight = read_weight(target);
    let activation = activation(weight.reduction);
    let want = reference(&weight, &activation);

    let run = run_under_lane_run(&weight, &activation, Some("8"));
    let off = run_under_lane_run(&weight, &activation, Some("0"));
    let unset = run_under_lane_run(&weight, &activation, None);

    assert!(run.keys.iter().any(|key| key.contains("_lr8")), "{}: no `_lr8` kernel was selected, the run fell back to the old body: {:?}", weight.name, run.keys);
    assert!(off.keys.iter().all(|key| !key.contains("_lr")), "{}: the switch-off plan carries a lane-run token: {:?}", weight.name, off.keys);
    assert_eq!(off.keys, unset.keys, "{}: switch `=0` and switch unset plan different kernels", weight.name);
    assert_eq!(run.values.len(), want.len(), "one f32 per row");
    let non_finite = run.values.iter().chain(&off.values).filter(|value| !value.is_finite()).count();
    assert_eq!(non_finite, 0, "{}: non-finite outputs would pass the tolerance comparison vacuously", weight.name);

    let wrong = out_of_tolerance(&run.values, &want, 0);
    let neighbour = out_of_tolerance(&run.values, &want, 1);
    let (max_abs, max_rel) = run.values.iter().zip(&off.values).fold((0.0f64, 0.0f64), |(max_abs, max_rel), (got, base)| {
        let diff = (f64::from(*got) - f64::from(*base)).abs();
        (max_abs.max(diff), max_rel.max(diff / f64::from(*base).abs().max(f64::MIN_POSITIVE)))
    });
    let bit_differing_off_vs_unset = off.values.iter().zip(&unset.values).filter(|(left, right)| left.to_bits() != right.to_bits()).count();
    eprintln!(
        "lane_run_parity tensor={} rows={} k={} keys={:?} outside_tolerance={wrong} non_finite={non_finite} neighbour_outside={neighbour} max_abs_vs_switch_off={max_abs:e} max_rel_vs_switch_off={max_rel:e} off_vs_unset_bit_diffs={bit_differing_off_vs_unset}",
        weight.name, weight.rows, weight.reduction, run.keys
    );
    assert_eq!(wrong, 0, "{}: {wrong} of {} lane-run outputs differ from the host-decoded f64 reduction", weight.name, want.len());
    assert!(neighbour > want.len() * 9 / 10, "control: only {neighbour} outputs left tolerance against the neighbouring row, the check does not discriminate");
    assert_eq!(bit_differing_off_vs_unset, 0, "{}: switch `=0` differs bit for bit from the switch unset", weight.name);
}

#[proxima::test]
#[case::q8_0_attn_q_2048_to_4096("attn_q")]
#[case::q8_0_attn_output_4096_to_2048("attn_output")]
#[case::q8_0_attn_k_2048_to_512("attn_k")]
async fn lane_run_8_matches_the_host_decode_and_the_switch_off_plan_stays_unchanged(#[case] tensor: &str) {
    let target = match tensor {
        "attn_q" => Target::AttnQ,
        "attn_output" => Target::AttnOutput,
        "attn_k" => Target::AttnK,
        other => panic!("unknown target {other}"),
    };
    assert_lane_run_parity(target);
}
