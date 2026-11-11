//! Binds a checkpoint's prefill program at several row counts and reports, field by field,
//! what changes between them and whether each change is affine in the row count.
//!
//! Usage: `plan_rows_diff <model.gguf> [kv_capacity]`

use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::process::ExitCode;

use memmap2::Mmap;
use omega::emit_cuda_with_policy;
use proxima_gguf::{GgmlType, parse_complete};
use proxima_model_interop::{Codec, KvLayout, bind_checkpoint_with_kv_layout};
use proxima_tensor::spec::LayerCacheRoots;
use proxima_tensor::{BoundOp, NodeId, Op, bind_with_fusion, correct_packed_matmul_layouts, infer, prune_dead};
use proxima_numeric::{NumericPolicy};

const ROW_COUNTS: [u64; 5] = [2, 17, 277, 511, 512];

#[derive(Debug, thiserror::Error)]
enum DiffError {
    #[error("usage: plan_rows_diff <model.gguf> [kv_capacity]")]
    Usage,
    #[error("{0}")]
    Failed(String),
}

fn failed(error: impl std::fmt::Display) -> DiffError {
    DiffError::Failed(error.to_string())
}

fn codec_of(ggml_type: GgmlType) -> Option<Codec> {
    ggml_type
        .numeric_format()
        .filter(|format| !matches!(format, Codec::Scalar(_)))
}

fn packed_operands(program: &[Op], tensors: &BTreeMap<String, GgmlType>) -> BTreeMap<NodeId, Codec> {
    program
        .iter()
        .enumerate()
        .filter_map(|(index, op)| {
            let name = op.name().filter(|_| matches!(op, Op::Input { .. }))?;
            let codec = codec_of(*tensors.get(name)?)?;
            Some((NodeId(index as u32), codec))
        })
        .collect()
}

struct OpDigest {
    kind: String,
    skeleton: String,
    numbers: Vec<i128>,
    entry: String,
    source_hash: u64,
    threads: u64,
}

fn skeleton_and_numbers(text: &str) -> (String, Vec<i128>) {
    let mut skeleton = String::new();
    let mut numbers = Vec::new();
    let characters: Vec<char> = text.chars().collect();
    let mut index = 0;
    while index < characters.len() {
        let negative = characters[index] == '-'
            && characters.get(index + 1).is_some_and(char::is_ascii_digit);
        if characters[index].is_ascii_digit() || negative {
            let start = index;
            index += 1;
            while index < characters.len() && characters[index].is_ascii_digit() {
                index += 1;
            }
            let literal: String = characters[start..index].iter().collect();
            numbers.push(literal.parse::<i128>().unwrap_or(i128::MIN));
            skeleton.push('#');
        } else {
            skeleton.push(characters[index]);
            index += 1;
        }
    }
    (skeleton, numbers)
}

fn text_hash(text: &str) -> u64 {
    text.bytes()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3))
}

fn digest(
    resolved: &BoundOp,
    packed: &BTreeMap<NodeId, Codec>,
    policy: NumericPolicy,
) -> Result<OpDigest, DiffError> {
    let (skeleton, numbers) = skeleton_and_numbers(&format!("{resolved:?}"));
    let kernel = emit_cuda_with_policy(resolved, packed, policy).map_err(failed)?;
    Ok(OpDigest {
        kind: resolved.kind.name().to_owned(),
        skeleton,
        numbers,
        entry: kernel.entry.clone(),
        source_hash: text_hash(&kernel.source),
        threads: kernel.grid.threads,
    })
}

fn print_gemm_operands(
    program: &[Op],
    packed: &BTreeMap<NodeId, Codec>,
    policy: NumericPolicy,
    resolved: &[BoundOp],
) -> Result<(), DiffError> {
    for op in resolved {
        let kernel = emit_cuda_with_policy(op, packed, policy).map_err(failed)?;
        if !kernel.entry.contains("mma16x8x8") {
            continue;
        }
        let operands: Vec<String> = op
            .operands()
            .iter()
            .map(|(node, layout, _)| {
                let shape = format!("{:?}", program[node.0 as usize]);
                let head: String = shape.chars().take(40).collect();
                format!("node{} codec={:?} layout={layout:?} {head}", node.0, packed.get(node))
            })
            .collect();
        println!("gemm entry={} dtype={:?} operands={operands:#?}", kernel.entry, op.dtype);
    }
    Ok(())
}

fn bind_at(
    program: &[Op],
    symbols: &[u64],
    outputs: &[NodeId],
    packed: &BTreeMap<NodeId, Codec>,
    policy: NumericPolicy,
) -> Result<Vec<OpDigest>, DiffError> {
    let shapes = infer(program, symbols).map_err(failed)?;
    let bound = bind_with_fusion(program, &shapes, outputs, policy.reassociation, policy).map_err(failed)?;
    let mut resolved = prune_dead(bound, outputs);
    correct_packed_matmul_layouts(&mut resolved, &packed.keys().copied().collect());
    if env::var_os("PLAN_ROWS_DIFF_GEMM").is_some() && symbols[0] == 512 {
        print_gemm_operands(program, packed, policy, &resolved)?;
    }
    resolved.iter().map(|op| digest(op, packed, policy)).collect()
}

fn affine(values: &[(u64, i128)]) -> bool {
    let (first_rows, first_value) = values[0];
    let (last_rows, last_value) = values[values.len() - 1];
    values.iter().all(|&(rows, value)| {
        (value - first_value) * i128::from(last_rows - first_rows)
            == (last_value - first_value) * i128::from(rows - first_rows)
    })
}

#[derive(Default)]
struct Tally {
    ops: usize,
    skeleton_differs: usize,
    count_differs: usize,
    entry_differs: usize,
    source_differs: usize,
    numeric_fields: usize,
    varying_fields: usize,
    non_affine_fields: usize,
    threads_non_affine: usize,
    details: Vec<String>,
}

fn compare(label: &str, per_rows: &[(u64, Vec<OpDigest>)]) -> Tally {
    let mut tally = Tally::default();
    let reference = &per_rows[0].1;
    if per_rows.iter().any(|(_, ops)| ops.len() != reference.len()) {
        let counts: Vec<_> = per_rows.iter().map(|(rows, ops)| (*rows, ops.len())).collect();
        tally.details.push(format!("{label}: op counts differ {counts:?}"));
        return tally;
    }
    tally.ops = reference.len();
    for position in 0..reference.len() {
        compare_position(label, position, per_rows, &mut tally);
    }
    tally
}

fn compare_position(label: &str, position: usize, per_rows: &[(u64, Vec<OpDigest>)], tally: &mut Tally) {
    let base = &per_rows[0].1[position];
    let at = |rows_index: usize| &per_rows[rows_index].1[position];
    let differs = |field: &dyn Fn(&OpDigest) -> bool| (1..per_rows.len()).any(|index| field(at(index)));
    if differs(&|op| op.kind != base.kind || op.skeleton != base.skeleton) {
        tally.skeleton_differs += 1;
        tally.details.push(format!("{label} op {position} {}: skeleton differs", base.kind));
        return;
    }
    tally.count_differs += usize::from(differs(&|op| op.numbers.len() != base.numbers.len()));
    if differs(&|op| op.entry != base.entry) {
        tally.entry_differs += 1;
        let entries: Vec<(u64, &str)> = per_rows
            .iter()
            .map(|(rows, ops)| (*rows, ops[position].entry.as_str()))
            .collect();
        tally.details.push(format!("{label} op {position} {}: kernel entry {entries:?}", base.kind));
    }
    tally.source_differs += usize::from(differs(&|op| op.source_hash != base.source_hash));
    let threads: Vec<(u64, i128)> = per_rows
        .iter()
        .map(|(rows, ops)| (*rows, i128::from(ops[position].threads)))
        .collect();
    if !affine(&threads) {
        tally.threads_non_affine += 1;
        tally.details.push(format!("{label} op {position} {}: grid threads {threads:?}", base.kind));
    }
    for field in 0..base.numbers.len() {
        let values: Vec<(u64, i128)> = per_rows
            .iter()
            .map(|(rows, ops)| (*rows, ops[position].numbers[field]))
            .collect();
        tally.numeric_fields += 1;
        if values.iter().any(|&(_, value)| value != values[0].1) {
            tally.varying_fields += 1;
            if !affine(&values) {
                tally.non_affine_fields += 1;
                tally.details.push(format!("{label} op {position} {} field {field}: {values:?}", base.kind));
            }
        }
    }
}

fn report(label: &str, tally: &Tally) {
    println!(
        "{label}: ops={} skeleton_differs={} numeric_count_differs={} entry_differs={} source_differs={} \
         numeric_fields={} varying_with_rows={} varying_not_affine={} grid_threads_not_affine={}",
        tally.ops,
        tally.skeleton_differs,
        tally.count_differs,
        tally.entry_differs,
        tally.source_differs,
        tally.numeric_fields,
        tally.varying_fields,
        tally.non_affine_fields,
        tally.threads_non_affine,
    );
    tally.details.iter().filter(|line| line.contains("kernel entry") || line.contains("skeleton")).take(6).for_each(|line| println!("  {line}"));
}

fn run() -> Result<(), DiffError> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or(DiffError::Usage)?;
    let capacity: u64 = args.next().map_or(Ok(1376), |text| text.parse().map_err(failed))?;
    let file = File::open(&path).map_err(failed)?;
    let mapping = unsafe { Mmap::map(&file) }.map_err(failed)?;
    let parsed = parse_complete(&mapping).map_err(failed)?;
    let tensors: BTreeMap<String, GgmlType> =
        parsed.tensors.iter().map(|info| (info.name.clone(), info.ggml_type)).collect();
    let bound = bind_checkpoint_with_kv_layout(&parsed, &mapping, KvLayout::SlidingRing).map_err(failed)?;
    let program = &bound.program;
    let packed = packed_operands(program, &tensors);
    let policy = NumericPolicy::llama_relaxed()
        .with_epilogue_sources(true)
        .with_activation_quantization(true);
    let cache_roots: Vec<NodeId> = bound
        .layer_roots
        .iter()
        .flat_map(|roots| match roots {
            LayerCacheRoots::Attention((even, odd, value)) => vec![*even, *odd, *value],
            _ => Vec::new(),
        })
        .collect();
    let mut with_logits = cache_roots.clone();
    with_logits.push(bound.logits_root);
    println!("model={path} program_ops={} packed_inputs={} kv_capacity={capacity}", program.len(), packed.len());
    for (label, outputs) in [("no_logits", &cache_roots), ("with_logits", &with_logits)] {
        let mut per_rows = Vec::new();
        for rows in ROW_COUNTS {
            let digests = bind_at(program, &[rows, capacity, capacity], outputs, &packed, policy)?;
            per_rows.push((rows, digests));
        }
        report(label, &compare(label, &per_rows));
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("plan_rows_diff: {error}");
            ExitCode::FAILURE
        }
    }
}
