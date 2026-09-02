//! `docs/discipline.md` ROW 181's Phase 1 profile gate: per-node-class wall
//! time inside `cpu::evaluate_quantized_with_scratch`'s own resolved-node
//! loop (`cpu.rs`) -- the loop `cpu::evaluate_named`/`evaluate_quantized_named`
//! actually walks, and so the loop `benches/mnist_f32_lane.rs`'s sealed
//! `evaluate_named` call exercises. Attributed by
//! `proxima_tensor::cpu::epilogue_profile_totals()` (`epilogue-profile-probe`
//! feature) into three buckets -- (a) every `Keep::Reduce` fold, tile-routed
//! and generic combined; (b) an `Elementwise` node whose sole non-broadcast
//! operand is a reduce output (a post-reduce epilogue -- bias-add,
//! batchnorm-scale-shift, clip-after-matmul); (c) everything else.
//! `proxima-tensor/instrument` rides along ONLY for the corroborating
//! `path_*` tile-vs-generic counters -- this binary is a non-timed
//! diagnostic, never the sealed number (`benches/mnist_f32_lane.rs` owns
//! that, and deliberately excludes both `instrument` and this probe from
//! its own feature set).
//!
//! `epilogue-profile-diag` feature, presence-guarded on the same host-local
//! mnist.onnx + MNIST idx checkout `mnist_f32_lane.rs`/`mnist_diag.rs` use,
//! clean skip when either is absent.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs;
use std::path::{Path, PathBuf};

use proxima_tensor::{cpu, instrument};

const MODEL_PATH: &str =
    "/Users/brianbruggeman/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx";
const DATASET_DIR: &str = "/Users/brianbruggeman/.cache/burn-dataset/mnist";
const PROFILE_IMAGES: usize = 200;

fn test_images_path() -> PathBuf {
    Path::new(DATASET_DIR).join("test/t10k-images-idx3-ubyte")
}

fn idx_header(bytes: &[u8]) -> (usize, Vec<usize>) {
    let dimension_count = bytes[3] as usize;
    let item_count = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let mut extents = Vec::with_capacity(dimension_count - 1);
    for axis in 1..dimension_count {
        let offset = 4 + axis * 4;
        extents.push(u32::from_be_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize);
    }
    (item_count, extents)
}

fn load_normalized_images(path: &Path, limit: usize) -> Vec<Vec<f32>> {
    let bytes = fs::read(path).expect("read idx3 image file");
    let (item_count, extents) = idx_header(&bytes);
    let pixel_count = extents.iter().product::<usize>();
    let take = item_count.min(limit);
    let header_length = 4 + extents.len() * 4 + 4;
    (0..take)
        .map(|image_index| {
            let start = header_length + image_index * pixel_count;
            bytes[start..start + pixel_count]
                .iter()
                .map(|&pixel| ((pixel as f32 / 255.0) - 0.1307) / 0.3081)
                .collect()
        })
        .collect()
}

fn main() {
    if !Path::new(MODEL_PATH).exists() {
        eprintln!("epilogue_profile: skipping, no host-local mnist.onnx checkout at {MODEL_PATH}");
        return;
    }
    if !test_images_path().exists() {
        eprintln!(
            "epilogue_profile: skipping, no host-local MNIST idx dataset under {DATASET_DIR}"
        );
        return;
    }

    let bytes = fs::read(MODEL_PATH).expect("read the real mnist.onnx checkpoint");
    let model =
        proxima_onnx::pipe::parse_complete(&bytes).expect("parse the real mnist.onnx checkpoint");
    let graph = model.graph.as_ref().expect("real mnist model has a graph");
    let lowered =
        proxima_onnx::lower::lower_graph(graph).expect("lower the real mnist.onnx graph to Op");

    let graph_input_name = lowered
        .graph_inputs
        .first()
        .expect("real mnist model declares at least one input")
        .clone();
    let output_node = lowered
        .graph_outputs
        .first()
        .expect("real mnist model declares at least one output")
        .1;
    let initializers: Vec<(&str, &[f32])> = lowered
        .initializers
        .iter()
        .map(|(name, data)| (name.as_str(), data.as_slice()))
        .collect();

    let images = load_normalized_images(&test_images_path(), PROFILE_IMAGES);
    assert!(
        !images.is_empty(),
        "expected at least one real mnist test image"
    );

    // warm-up: outside both instrument and the probe's own reset window, so
    // first-call effects (allocator warm-up, page faults) never pollute the
    // attributed breakdown.
    let mut named = initializers.clone();
    named.push((graph_input_name.as_str(), images[0].as_slice()));
    let _ =
        cpu::evaluate_named(&lowered.program, &[], &named, &[output_node]).expect("warm-up eval");

    instrument::reset();
    cpu::epilogue_profile_reset();

    for image in &images {
        let mut named = initializers.clone();
        named.push((graph_input_name.as_str(), image.as_slice()));
        let evaluated = cpu::evaluate_named(&lowered.program, &[], &named, &[output_node])
            .expect("evaluate real mnist image");
        std::hint::black_box(&evaluated);
    }

    let (reduce_nanos, reduce_calls, epilogue_nanos, epilogue_calls, other_nanos, other_calls) =
        cpu::epilogue_profile_totals();
    let total_nanos = reduce_nanos + epilogue_nanos + other_nanos;
    let total_calls = reduce_calls + epilogue_calls + other_calls;
    let percent = |nanos: u64| -> f64 {
        if total_nanos == 0 {
            0.0
        } else {
            nanos as f64 / total_nanos as f64 * 100.0
        }
    };

    println!(
        "epilogue_profile: {} real mnist images, evaluate_quantized_with_scratch node-class breakdown",
        images.len()
    );
    println!(
        "  (a) reduce-fold      : {:>10} calls, {:>12} ns total, {:6.2}% of step time, {:.1} ns/call",
        reduce_calls,
        reduce_nanos,
        percent(reduce_nanos),
        reduce_nanos as f64 / reduce_calls.max(1) as f64
    );
    println!(
        "  (b) post-reduce epi  : {:>10} calls, {:>12} ns total, {:6.2}% of step time, {:.1} ns/call",
        epilogue_calls,
        epilogue_nanos,
        percent(epilogue_nanos),
        epilogue_nanos as f64 / epilogue_calls.max(1) as f64
    );
    println!(
        "  (c) everything else  : {:>10} calls, {:>12} ns total, {:6.2}% of step time, {:.1} ns/call",
        other_calls,
        other_nanos,
        percent(other_nanos),
        other_nanos as f64 / other_calls.max(1) as f64
    );
    println!(
        "  total                 : {total_calls:>10} calls, {total_nanos:>12} ns total over {} images ({:.3} ms/image)",
        images.len(),
        total_nanos as f64 / images.len() as f64 / 1e6
    );

    let path_totals = instrument::totals();
    println!(
        "  corroborating path_* counters (reduce tile-vs-generic split, instrument feature): dot_fast={} width_fast={} conv_tile={} generic={}",
        path_totals.path_dot_fast,
        path_totals.path_width_fast,
        path_totals.path_conv_tile,
        path_totals.path_generic
    );

    const GATE_THRESHOLD_PERCENT: f64 = 10.0;
    let epilogue_percent = percent(epilogue_nanos);
    if epilogue_percent < GATE_THRESHOLD_PERCENT {
        println!(
            "GATE: STOP after phase 1 -- class (b) post-reduce-epilogue mass is {epilogue_percent:.2}% of step time, below the {GATE_THRESHOLD_PERCENT}% bar. Lever not worth building."
        );
    } else {
        println!(
            "GATE: PROCEED to phase 2 -- class (b) post-reduce-epilogue mass is {epilogue_percent:.2}% of step time, at/above the {GATE_THRESHOLD_PERCENT}% bar."
        );
    }
}
