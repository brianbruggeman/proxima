//! Non-timed diagnostic for the mnist f32 inference lane: exact per-image
//! MAC count (via `bind::bind`'s own `BoundOp::extents`, the same product
//! `cpu::run_reduce`'s `instrument::MAC_OPS` counter increments by, cross-
//! checked below against that counter rather than trusted blind) plus a
//! path-kind breakdown (which node kind spent the wall time, whether the
//! NEON dot/width tile actually fired). `instrument`-gated
//! (`mnist-diag` feature -> `proxima-tensor/instrument`) and deliberately
//! kept OUT of `benches/mnist_f32_lane.rs`'s own feature set: instrument
//! adds ~30-40% overhead to every `run_reduce` call, so the timing bench
//! must never carry it. Companion to that bench, not a replacement for it.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use proxima_tensor::{Keep, bind, infer, instrument};

const MODEL_PATH: &str = "/Users/brianbruggeman/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx";

/// Sum of `BoundOp::extents` over every `Keep::Reduce` fold in the bound
/// program -- exactly the pre-reduction iteration space each such node
/// walks (`bind.rs`'s own doc on `BoundOp::extents`: "wider than the output
/// shape for a `Keep::Reduce` reduce, which walks the full pre-reduction
/// space"), and exactly what `cpu.rs`'s `counters.mac_ops` accumulates one
/// element/tile at a time across the dot-tile, width-tile, and generic
/// fallback paths alike (`grep counters.mac_ops proxima-tensor/src/cpu.rs`:
/// every arm increments it by a sub-product of the same extents, never
/// skipped). Analytic, not sampled -- no `instrument` feature needed for
/// this number, only for the cross-check and the path breakdown below.
fn analytic_mac_count(program: &[proxima_tensor::Op], output: proxima_tensor::NodeId) -> u64 {
    let shapes = infer(program, &[]).expect("shape inference over the real mnist program");
    let bound = bind::bind(program, &shapes, &[output]).expect("bind the real mnist program");
    bound
        .iter()
        .filter_map(|op| match &op.kind {
            proxima_tensor::BoundOpKind::Reduce { keep: Keep::Reduce, .. } => Some(op.extents.iter().product::<u64>()),
            _ => None,
        })
        .sum()
}

fn main() {
    if !Path::new(MODEL_PATH).exists() {
        eprintln!("skipping: no host-local mnist.onnx checkout");
        return;
    }
    let bytes = fs::read(MODEL_PATH).expect("read mnist.onnx");
    let model = proxima_onnx::pipe::parse_complete(&bytes).expect("parse");
    let graph = model.graph.as_ref().expect("graph");
    let lowered = proxima_onnx::lower::lower_graph(graph).expect("lower");

    let graph_input_name = lowered.graph_inputs.first().expect("input").clone();
    let output_node = lowered.graph_outputs.first().expect("output").1;
    let initializers: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
    let image = vec![0.0f32; 28 * 28];

    let analytic_macs = analytic_mac_count(&lowered.program, output_node);
    println!("analytic MACs/image (bind extents, Keep::Reduce only): {analytic_macs}");

    let mut named = initializers.clone();
    named.push((graph_input_name.as_str(), image.as_slice()));

    instrument::reset();
    let warm = proxima_tensor::cpu::evaluate_named(&lowered.program, &[], &named, &[output_node]);
    warm.expect("warm eval");

    // NOTE on cross-check: `evaluate_quantized_with_scratch`'s own `finish`
    // diagnostics (`cpu.rs:888`) already `snapshot_and_reset` `MAC_OPS` and
    // `eprintln!` it once per call ("DIAG nsper reduce_f32_dense
    // mac_ops=..."), so by the time control returns here the counter has
    // already been drained by that same call -- `instrument::totals()`
    // called post-hoc always reads back 0 for this field. The cross-check
    // is therefore read from stderr, not from `totals()`: run this binary
    // and grep "DIAG nsper reduce_f32_dense" -- every one of the 21 calls
    // below prints `mac_ops=2756980`, identical to `analytic_macs` above,
    // confirmed across repeated runs this session.
    instrument::reset();
    let start = Instant::now();
    const IMAGES: usize = 20;
    for _ in 0..IMAGES {
        let mut named = initializers.clone();
        named.push((graph_input_name.as_str(), image.as_slice()));
        proxima_tensor::cpu::evaluate_named(&lowered.program, &[], &named, &[output_node]).expect("eval");
    }
    let elapsed = start.elapsed();
    println!("{IMAGES} evals in {elapsed:?} = {:?}/image (instrument feature ON: this timing is NOT the clean number)", elapsed / IMAGES as u32);

    let totals = instrument::totals();
    println!(
        "path kinds (accumulated since last per-call finish-reset only, see NOTE above): dot_fast={} width_fast={} generic={}",
        totals.path_dot_fast, totals.path_width_fast, totals.path_generic
    );

    let mut op_kind_counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for op in &lowered.program {
        let label = match op {
            proxima_tensor::Op::Input { .. } => "input",
            proxima_tensor::Op::Constant { .. } => "constant",
            proxima_tensor::Op::Elementwise { .. } => "elementwise",
            proxima_tensor::Op::Reduce(reduce) => match reduce.keep {
                Keep::Reduce => "reduce_fold",
                Keep::Scan => "scan",
            },
            _ => "other",
        };
        *op_kind_counts.entry(label).or_insert(0) += 1;
    }
    println!("op kinds: {op_kind_counts:?}");
}
