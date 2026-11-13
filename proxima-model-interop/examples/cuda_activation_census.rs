//! Census of the matmul readers omega's CUDA lowering quantizes the activation for, per phase,
//! over a real checkpoint: which weight codecs and shapes it derives a `BlockQuantize` node for,
//! and which it declines. Binds only (`omega::cuda_driver::probe_bound_graph`); no kernel is
//! emitted and no device is touched.
//!
//! The decode (1 new token) and prefill (N new tokens) programs, symbols, outputs and weight
//! blocks come from `LoadedModel::with_cuda_bind_inputs`, the same assembly the CUDA decode loop
//! feeds its plan (`generate/decode.rs`), so no second model builder exists here.
//!
//! Usage: `cuda_activation_census <model.gguf> <prompt_tokens>` (prompt_tokens >= 2)

use std::collections::BTreeMap;
use std::env;
use std::fmt::Display;
use std::fs::File;
use std::process::ExitCode;

use memmap2::Mmap;
use omega::cuda_driver::probe_bound_graph;
use proxima_gguf::parse_complete;
use proxima_model_interop::LoadedModel;
use proxima_numeric::{DType, NumericFormat, NumericPolicy};
use proxima_tensor::{
    BoundOp, BoundOpKind, Keep, NodeId, Op, QuantizedBlock, ScalarOp, Shapes, activation_is_flat, collapsed_extent, infer,
    is_derived,
};

const GROUP_ELEMENTS: u64 = 32;

#[derive(Debug, thiserror::Error)]
enum CensusError {
    #[error("usage: cuda_activation_census <model.gguf> <prompt_tokens, at least 2>")]
    Usage,
    #[error("{0}")]
    Failed(String),
}

fn failed(error: impl Display) -> CensusError {
    CensusError::Failed(error.to_string())
}

#[derive(Default)]
struct Tally {
    readers: usize,
    derived: usize,
    declined: BTreeMap<(String, bool, bool), usize>,
}

struct Reader {
    weight: NodeId,
    activation_slot: usize,
}

fn reader_of(bound: &BoundOp, program: &[Op], packed: &BTreeMap<NodeId, NumericFormat>, sources: &BTreeMap<NodeId, NodeId>) -> Option<Reader> {
    let BoundOpKind::Reduce { keep: Keep::Reduce, reduce_op: ScalarOp::Add, operands, .. } = &bound.kind else {
        return None;
    };
    let is_weight = |node: &NodeId| packed.get(node).is_some_and(|codec| !codec.is_quantized_activation());
    let mut weights = operands.iter().enumerate().filter(|(_, (node, _, _))| is_weight(node));
    let (weight_slot, (weight, _, _)) = weights.next()?;
    if weights.next().is_some() {
        return None;
    }
    let reads_f32 = |node: &NodeId| {
        sources.contains_key(node)
            || (!packed.contains_key(node)
                && program.get(node.0 as usize).is_some_and(|op| matches!(op.dtype(), DType::FLOAT32)))
    };
    let activation_slot = (0..operands.len()).find(|slot| *slot != weight_slot && reads_f32(&operands[*slot].0))?;
    Some(Reader { weight: *weight, activation_slot })
}

fn record(phase: &str, bound: &BoundOp, reader: &Reader, shapes: &Shapes, packed: &BTreeMap<NodeId, NumericFormat>, sources: &BTreeMap<NodeId, NodeId>, tally: &mut Tally) {
    let (read_node, layout, _) = &bound.operands()[reader.activation_slot];
    let activation = sources.get(read_node).copied().unwrap_or(*read_node);
    let weight_codec = packed.get(&reader.weight);
    let extent = collapsed_extent(shapes.of(activation));
    let flat = extent.is_some_and(|[tokens, depth]| activation_is_flat(bound, layout, tokens, depth));
    let derived_held = sources.contains_key(read_node).then(|| packed.get(read_node));
    let extent_text = extent.map_or_else(|| String::from("none"), |[tokens, depth]| format!("[{tokens}, {depth}]"));
    let derived_text = match derived_held {
        Some(held) => format!("yes node{} held={held:?}", read_node.0),
        None => String::from("no"),
    };
    println!(
        "{phase} reader=node{} weight={weight_codec:?} activation=node{} extent={extent_text} flat={flat} derived={derived_text}",
        bound.node.0, activation.0,
    );
    tally.readers += 1;
    if derived_held.is_some() {
        tally.derived += 1;
        return;
    }
    let aligned = extent.is_some_and(|[_, depth]| depth % GROUP_ELEMENTS == 0);
    *tally.declined.entry((format!("{weight_codec:?}"), aligned, flat)).or_default() += 1;
}

fn print_summary(phase: &str, tally: &Tally) {
    println!("{phase} summary: readers={} derived={} declined={}", tally.readers, tally.derived, tally.readers - tally.derived);
    for ((codec, aligned, flat), count) in &tally.declined {
        println!("{phase} declined: weight={codec} depth_mod_32_zero={aligned} flat={flat} count={count}");
    }
}

fn census(
    phase: &str,
    program: &[Op],
    symbols: &[u64],
    outputs: &[NodeId],
    named: &[(&str, QuantizedBlock<'_>)],
    policy: NumericPolicy,
) -> Result<(), CensusError> {
    let (bound, packed) = probe_bound_graph(program, symbols, outputs, named, policy, None).map_err(failed)?;
    let shapes = infer(program, symbols).map_err(failed)?;
    let sources: BTreeMap<NodeId, NodeId> = bound
        .iter()
        .filter(|op| is_derived(op, program.len()))
        .filter_map(|op| Some((op.node, op.operands().first()?.0)))
        .collect();
    println!("{phase} program_ops={} bound_ops={} symbols={symbols:?}", program.len(), bound.len());
    let mut tally = Tally::default();
    for op in &bound {
        if let Some(reader) = reader_of(op, program, &packed, &sources) {
            record(phase, op, &reader, &shapes, &packed, &sources, &mut tally);
        }
    }
    print_summary(phase, &tally);
    Ok(())
}

fn run() -> Result<(), CensusError> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or(CensusError::Usage)?;
    let prompt_tokens: usize = args.next().ok_or(CensusError::Usage)?.parse().map_err(|_| CensusError::Usage)?;
    if prompt_tokens < 2 {
        return Err(CensusError::Usage);
    }
    let file = File::open(&path).map_err(failed)?;
    // SAFETY: `file` outlives the read-only mapping, which LoadedModel borrows for this process only.
    let bytes = unsafe { Mmap::map(&file) }.map_err(failed)?;
    let parsed = parse_complete(&bytes).map_err(failed)?;
    let model = LoadedModel::load(&parsed, &bytes).map_err(failed)?;
    let policy = NumericPolicy::llama_relaxed().with_activation_quantization(true);
    let kv_capacity = prompt_tokens + 1;
    for (phase, new_count) in [("decode", 1), ("prefill", prompt_tokens)] {
        model
            .with_cuda_bind_inputs(new_count, kv_capacity, |program, symbols, outputs, named| {
                census(phase, program, symbols, outputs, named, policy)
            })
            .map_err(failed)??;
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("cuda_activation_census: {error}");
            ExitCode::FAILURE
        }
    }
}
