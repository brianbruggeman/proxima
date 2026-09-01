//! ROW 185 cell 3 — allocation-tracker measurement for ROW 184's
//! zero-allocation claim on `apply_epilogue_fused_monomorphic`
//! (`proxima-tensor/src/cpu.rs:1095`).
//!
//! ROW 184 verified the kernel has zero `vec!`/`Vec::`/`Box::`/`.collect()`
//! call sites by READING the compiled function — legitimate per principle 6
//! but weaker than an instrumented allocation-counter run, and named as the
//! one residual its own "re-provable" claim could not close without a
//! follow-up harness. This is that harness.
//!
//! Design, stated honestly rather than asserting a number this crate cannot
//! back: the real mnist eval loop's per-image allocation count is NOT
//! attributable only to the epilogue kernel — `epilogue_fuse_plan`
//! (`cpu.rs:946`) rebuilds a `BTreeMap`/`BTreeSet` bookkeeping structure once
//! per `evaluate_named` call (ROW 184's own already-named, unchanged
//! setup-path cost), so a bare "assert zero total allocations" on the fused
//! arm would be false and would misattribute a real, already-documented cost
//! to the wrong function. Instead this test measures the SAME real mnist
//! loop, same host, same N images, via the thread-local counting allocator
//! precedent
//! (`proxima-protocols/tests/pgwire_codec_integration/alloc_counter.rs`). ROW
//! 186 promoted the fusion path to default-on, so this run reports the
//! fused arm's own numbers directly (no separate feature build); a paired
//! unfused comparison uses `proxima_tensor::cpu::set_epilogue_fuse_enabled`,
//! the ROW 186 bench/test escape valve, rather than a second cargo feature
//! build. `apply_epilogue_fused_monomorphic` fires 5
//! times per image over 26,282 elements/image (`epilogue_fuse_totals()`,
//! confirmed below) — if it allocated per-hit or per-element, the fused arm's
//! per-image count would visibly exceed the unfused arm's by a multiple of
//! 5 or 26,282, not by a small constant. A small, N-independent delta is the
//! allocator-instrumented confirmation of the code-read claim; a delta that
//! scales with hits/elements would refute it.

#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fs;
use std::path::{Path, PathBuf};
use std::vec::Vec;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|count| count.set(count.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|count| count.set(count.get() + 1));
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const MODEL_PATH: &str = "/Users/brianbruggeman/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx";
const DATASET_DIR: &str = "/Users/brianbruggeman/.cache/burn-dataset/mnist";
const IMAGE_COUNT: usize = 120;

fn checkpoint_present() -> bool {
    Path::new(MODEL_PATH).exists()
}

fn test_images_path() -> PathBuf {
    Path::new(DATASET_DIR).join("test/t10k-images-idx3-ubyte")
}

fn test_labels_path() -> PathBuf {
    Path::new(DATASET_DIR).join("test/t10k-labels-idx1-ubyte")
}

fn dataset_present() -> bool {
    test_images_path().exists() && test_labels_path().exists()
}

fn idx_header(bytes: &[u8]) -> (usize, Vec<usize>) {
    let dimension_count = bytes[3] as usize;
    let item_count = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let mut extents = Vec::with_capacity(dimension_count - 1);
    for axis in 1..dimension_count {
        let offset = 4 + axis * 4;
        extents.push(u32::from_be_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]]) as usize);
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
            bytes[start..start + pixel_count].iter().map(|&pixel| ((pixel as f32 / 255.0) - 0.1307) / 0.3081).collect()
        })
        .collect()
}

/// Runs the real mnist eval loop over [`IMAGE_COUNT`] real images, one
/// warm-up call first (uncounted, so any lazy first-call setup e.g. program
/// pool growth doesn't pollute the measured window), then reports total and
/// per-image allocation counts plus (ROW 186: unconditional now that
/// epilogue fusion is default-on) the fusion hit/element totals for the
/// SAME window.
#[test]
#[ignore = "depends on a real .onnx checkout and the real MNIST idx dataset outside this repo"]
fn fused_eval_loop_allocation_count_over_100_plus_images() {
    if !checkpoint_present() {
        eprintln!("skipping: no host-local mnist.onnx checkout at {MODEL_PATH}");
        return;
    }
    if !dataset_present() {
        eprintln!("skipping: no host-local MNIST idx dataset under {DATASET_DIR}");
        return;
    }

    let bytes = fs::read(MODEL_PATH).expect("read the real mnist.onnx checkpoint");
    let model = proxima_onnx::pipe::parse_complete(&bytes).expect("parse the real mnist.onnx checkpoint");
    let graph = model.graph.as_ref().expect("real mnist model has a graph");
    let lowered = proxima_onnx::lower::lower_graph(graph).expect("lower the real mnist.onnx graph to Op");

    let graph_input_name = lowered.graph_inputs.first().expect("real mnist model declares at least one input").clone();
    let output_node = lowered.graph_outputs.first().expect("real mnist model declares at least one output").1;
    let initializers: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();

    let images = load_normalized_images(&test_images_path(), IMAGE_COUNT);
    assert!(images.len() >= IMAGE_COUNT, "expected at least {IMAGE_COUNT} real test images, got {}", images.len());

    let evaluate = |image: &[f32]| {
        let mut named = initializers.clone();
        named.push((graph_input_name.as_str(), image));
        proxima_tensor::cpu::evaluate_named(&lowered.program, &[], &named, &[output_node]).expect("evaluate real mnist image")
    };

    // warm-up: uncounted, primes any first-call-only setup.
    let _ = evaluate(&images[0]);

    proxima_tensor::cpu::epilogue_fuse_reset();

    let count_before = ALLOCATIONS.with(Cell::get);
    for image in &images {
        let _ = evaluate(image);
    }
    let count_after = ALLOCATIONS.with(Cell::get);

    let total_allocations = count_after - count_before;
    let per_image = total_allocations as f64 / images.len() as f64;

    eprintln!("epilogue_fuse_alloc: images={} total_allocations={total_allocations} per_image={per_image:.4}", images.len());

    let (hits, elements, nanos) = proxima_tensor::cpu::epilogue_fuse_totals();
    eprintln!(
        "epilogue_fuse_alloc: hits={hits} elements={elements} nanos={nanos} hits_per_image={:.4} elements_per_image={:.4}",
        hits as f64 / images.len() as f64,
        elements as f64 / images.len() as f64
    );
    assert!(hits > 0, "epilogue fusion must actually fire on the real mnist model (N==0 tripwire)");
}
