//! Model-instance restore speed, at the level this crate's primitives
//! genuinely reach: **region-level, host-process** — a named, shareable host
//! memory region ([`proxima_vm::named_memory::GuestMemoryRegion`],
//! `named_memory.rs`'s own M4 module) holding a real `mnist.onnx` model's
//! weight buffers, restored by a fresh consumer as a second mapping
//! ([`GuestMemoryRegion::map_shared_view`]) of the same backing object —
//! never a guest-VM-level restore. `snapshot::LayeredBase`/`WarmVm` (M7,
//! `src/snapshot.rs`) build the guest-VM-level layered-restore design this
//! probe's PREPARE/RESTORE split otherwise mirrors, but every path through
//! `WarmVm::new_layered(_over)` calls `proxima_vm_layered_vcpu_create`
//! (`src/snapshot.rs`'s own `LayeredHandle::construct`), which needs a real
//! `hv_vcpu` tied to this process's one `hv_vm` — the
//! `com.apple.security.hypervisor` entitlement `tests/boot.rs` gates its own
//! `SignedGuest` probes behind. Running a full ONNX forward pass INSIDE that
//! guest would additionally need a guest-side aarch64 binary that can decode
//! IEEE-754 float ops without libm — out of this probe's budget. This probe
//! instead exercises `GuestMemoryRegion` directly, the same M4 primitive
//! `LayeredBase` itself is built from (`snapshot.rs`'s own `LayeredBase`
//! struct wraps exactly one `GuestMemoryRegion`) — restore is a second
//! `mach_make_memory_entry_64`-backed mapping of the SAME weight bytes, never
//! a copy, but the inference forward pass runs on the HOST CPU, not inside
//! any guest instruction stream.
//!
//! Three phases:
//!
//! 1. PREPARE (untimed): parse+lower the real `mnist.onnx` checkpoint
//!    (`proxima_onnx::pipe::parse_complete` -> `proxima_onnx::lower::lower_graph`,
//!    the same chain `proxima-onnx/tests/real_mnist_accuracy.rs` uses),
//!    concatenate every initializer's `f32` weights as little-endian bytes
//!    into one flat blob with a `(name, byte_offset, f32_count)` manifest,
//!    and write that blob into a fresh [`GuestMemoryRegion`] — the "base
//!    layer."
//! 2. RESTORE (timed, `>=20` cycles x 5 runs): a fresh
//!    [`GuestMemoryRegion::map_shared_view`] call (`map_nanos`, the claim),
//!    decode the mapped bytes back into `&[f32]` slices per the manifest
//!    (`decode_nanos`), and run one real forward pass via
//!    `proxima_tensor::cpu::evaluate_named` (`forward_nanos`) against one
//!    real, normalized `t10k` test image.
//! 3. Correctness gate: the restored-weights logits must equal, byte-exact
//!    within 1e-6, a directly-loaded run's logits over the same image and
//!    the same in-memory initializers, never touching the mapped region.
//!
//! Context row: the naive incumbent cold start — read `mnist.onnx` from
//! disk, parse, lower, and run the first forward pass, timed the same way,
//! same iteration/run shape.
//!
//! Presence-guarded: exits cleanly (not an error) when the host-local
//! `mnist.onnx` checkout or MNIST idx dataset is absent, the same convention
//! `real_mnist_accuracy.rs::checkpoint_present`/`dataset_present` use.

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use proxima_tensor::{NodeId, Op};
use proxima_vm::named_memory::GuestMemoryRegion;

const MODEL_PATH: &str =
    "/Users/brianbruggeman/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx";
const DATASET_DIR: &str = "/Users/brianbruggeman/.cache/burn-dataset/mnist";
const INPUT_PIXELS: usize = 28 * 28;
const OUTPUT_CLASSES: usize = 10;
const ITERATIONS_PER_RUN: usize = 20;
const RUN_COUNT: usize = 5;
const CORRECTNESS_TOLERANCE: f32 = 1e-6;

fn test_image_path() -> PathBuf {
    Path::new(DATASET_DIR).join("test/t10k-images-idx3-ubyte")
}

/// idx3 header parse, mirroring `real_mnist_accuracy.rs::idx_header` — this
/// probe needs exactly one normalized test image, not the full split.
fn load_one_normalized_image(path: &Path) -> Result<[f32; INPUT_PIXELS], Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let dimension_count = bytes[3] as usize;
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
    let pixel_count: usize = extents.iter().product();
    if pixel_count != INPUT_PIXELS {
        return Err(format!(
            "expected {INPUT_PIXELS} pixels per image, idx3 header declares {pixel_count}"
        )
        .into());
    }
    let header_length = 4 + extents.len() * 4 + 4;
    let mut image = [0.0_f32; INPUT_PIXELS];
    for (slot, &pixel) in image
        .iter_mut()
        .zip(&bytes[header_length..header_length + pixel_count])
    {
        *slot = ((pixel as f32) / 255.0 - 0.1307) / 0.3081;
    }
    Ok(image)
}

/// Parsed+lowered model state — the same shape
/// `examples/support/wire_to_weights_pipeline.rs::ModelState` carries,
/// reproduced locally rather than `#[path]`-included: that module also pulls
/// in `proxima::pipe`/`proxima::request` for its HTTP composition, which
/// this VM-level probe has no use for.
struct ModelState {
    program: Vec<Op>,
    initializers: Vec<(String, Vec<f32>)>,
    graph_input_name: String,
    output_node: NodeId,
}

fn load_model(path: &Path) -> Result<ModelState, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let model = proxima_onnx::pipe::parse_complete(&bytes)
        .map_err(|error| format!("parse mnist.onnx: {error}"))?;
    let graph = model.graph.as_ref().ok_or("mnist.onnx has no graph")?;
    let lowered = proxima_onnx::lower::lower_graph(graph)
        .map_err(|error| format!("lower mnist.onnx: {error}"))?;
    let graph_input_name = lowered
        .graph_inputs
        .first()
        .ok_or("mnist.onnx declares no graph input")?
        .clone();
    let output_node = lowered
        .graph_outputs
        .first()
        .ok_or("mnist.onnx declares no graph output")?
        .1;
    Ok(ModelState {
        program: lowered.program,
        initializers: lowered.initializers,
        graph_input_name,
        output_node,
    })
}

fn forward(
    model: &ModelState,
    initializers: &[(&str, &[f32])],
    image: &[f32; INPUT_PIXELS],
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut named: Vec<(&str, &[f32])> = initializers.to_vec();
    named.push((model.graph_input_name.as_str(), image.as_slice()));
    let evaluated =
        proxima_tensor::cpu::evaluate_named(&model.program, &[], &named, &[model.output_node])
            .map_err(|error| format!("mnist forward failed: {error}"))?;
    let (logits, shape) = evaluated
        .get(model.output_node)
        .ok_or("mnist forward produced no output")?;
    if shape != [1_u64, OUTPUT_CLASSES as u64] {
        return Err(format!("expected a 1x{OUTPUT_CLASSES} logit row, got shape {shape:?}").into());
    }
    Ok(logits.to_vec())
}

/// One entry in the weight manifest: `name` at byte offset `offset` in the
/// mapped region, `f32_count` little-endian `f32`s long.
struct WeightEntry {
    name: String,
    offset: usize,
    f32_count: usize,
}

/// Concatenates every initializer's `f32` weights as little-endian bytes
/// into one flat blob, recording each one's `(name, offset, f32_count)` —
/// the "base layer" PREPARE writes into a [`GuestMemoryRegion`] exactly
/// once, never touched again per this design's own "never written after
/// creation" discipline (`snapshot.rs`'s `LayeredBase` doc).
fn build_weight_blob(initializers: &[(String, Vec<f32>)]) -> (Vec<u8>, Vec<WeightEntry>) {
    let total_f32_count: usize = initializers.iter().map(|(_, data)| data.len()).sum();
    let mut blob = Vec::with_capacity(total_f32_count * 4);
    let mut manifest = Vec::with_capacity(initializers.len());
    for (name, data) in initializers {
        let offset = blob.len();
        for value in data {
            blob.extend_from_slice(&value.to_le_bytes());
        }
        manifest.push(WeightEntry {
            name: name.clone(),
            offset,
            f32_count: data.len(),
        });
    }
    (blob, manifest)
}

/// Decodes `&[f32]` slices back out of `mapped` per `manifest` — a plain
/// little-endian decode (`chunks_exact(4)`/`f32::from_le_bytes`), the same
/// convention `examples/support/wire_to_weights_pipeline.rs::ParseImage`
/// already establishes for this repo's byte-buffer request bodies. Timed
/// separately from the mapping call itself (`decode_nanos`), so the
/// restore/map number the pre-registered claim gates on is never inflated
/// by this reconstruction cost.
fn decode_named<'manifest>(
    mapped: &[u8],
    manifest: &'manifest [WeightEntry],
) -> Vec<(&'manifest str, Vec<f32>)> {
    manifest
        .iter()
        .map(|entry| {
            let byte_range = &mapped[entry.offset..entry.offset + entry.f32_count * 4];
            let (chunks, _remainder) = byte_range.as_chunks::<4>();
            let values: Vec<f32> = chunks
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect();
            (entry.name.as_str(), values)
        })
        .collect()
}

fn percentile_50(mut samples: Vec<u128>) -> u128 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn coefficient_of_variation(samples: &[u128]) -> f64 {
    let mean = samples.iter().sum::<u128>() as f64 / samples.len() as f64;
    if mean == 0.0 {
        return 0.0;
    }
    let variance = samples
        .iter()
        .map(|&value| (value as f64 - mean).powi(2))
        .sum::<f64>()
        / samples.len() as f64;
    variance.sqrt() / mean
}

fn report_phase(label: &str, run_p50s: &[u128]) {
    let overall_p50 = percentile_50(run_p50s.to_vec());
    let coefficient_variation = coefficient_of_variation(run_p50s);
    println!("phase_p50_of_run_p50s_nanos:{label}:{overall_p50}");
    println!("phase_cov_across_runs:{label}:{coefficient_variation:.4}");
    for (run_index, &value) in run_p50s.iter().enumerate() {
        println!("phase_run_p50_nanos:{label}:{run_index}:{value}");
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    if !Path::new(MODEL_PATH).exists() {
        eprintln!(
            "restore_to_inference_probe: skipping, no host-local mnist.onnx checkout at {MODEL_PATH}"
        );
        return Ok(());
    }
    if !test_image_path().exists() {
        eprintln!(
            "restore_to_inference_probe: skipping, no host-local MNIST idx dataset under {DATASET_DIR}"
        );
        return Ok(());
    }

    let model = load_model(Path::new(MODEL_PATH))?;
    let image = load_one_normalized_image(&test_image_path())?;

    // PREPARE -- untimed. Real weight buffers, concatenated into one flat
    // blob, written into a fresh named region exactly once.
    let (weight_blob, manifest) = build_weight_blob(&model.initializers);
    println!("weight_blob_bytes:{}", weight_blob.len());
    println!("initializer_count:{}", manifest.len());

    // `GuestMemoryRegion::create` rounds up to the host's page granularity
    // (`mach_vm_allocate`/`mmap`), so the mapped region is legitimately
    // larger than the blob it holds -- write into the blob-sized prefix,
    // never the whole (padded) mapping.
    let mut base_region = GuestMemoryRegion::create(weight_blob.len())?;
    base_region.primary_slice_mut()[..weight_blob.len()].copy_from_slice(&weight_blob);

    // Direct-load reference logits -- the model's own owned initializers,
    // never touching the mapped region at all. The correctness oracle every
    // restored run below is checked against.
    let direct_initializers: Vec<(&str, &[f32])> = model
        .initializers
        .iter()
        .map(|(name, data)| (name.as_str(), data.as_slice()))
        .collect();
    let direct_logits = forward(&model, &direct_initializers, &image)?;

    let mut map_run_p50s = Vec::with_capacity(RUN_COUNT);
    let mut decode_run_p50s = Vec::with_capacity(RUN_COUNT);
    let mut forward_run_p50s = Vec::with_capacity(RUN_COUNT);
    let mut total_run_p50s = Vec::with_capacity(RUN_COUNT);
    let mut first_restored_logits: Option<Vec<f32>> = None;

    for run_index in 0..RUN_COUNT {
        let mut map_samples = Vec::with_capacity(ITERATIONS_PER_RUN);
        let mut decode_samples = Vec::with_capacity(ITERATIONS_PER_RUN);
        let mut forward_samples = Vec::with_capacity(ITERATIONS_PER_RUN);
        let mut total_samples = Vec::with_capacity(ITERATIONS_PER_RUN);

        for iteration in 0..ITERATIONS_PER_RUN {
            let total_start = Instant::now();

            let map_start = Instant::now();
            let view = base_region.map_shared_view()?;
            let map_nanos = map_start.elapsed().as_nanos();

            let decode_start = Instant::now();
            let named = decode_named(view.as_slice(), &manifest);
            let named_refs: Vec<(&str, &[f32])> = named
                .iter()
                .map(|(name, data)| (*name, data.as_slice()))
                .collect();
            let decode_nanos = decode_start.elapsed().as_nanos();

            let forward_start = Instant::now();
            let logits = forward(&model, &named_refs, &image)?;
            let forward_nanos = forward_start.elapsed().as_nanos();

            let total_nanos = total_start.elapsed().as_nanos();

            println!("iteration_map_nanos:{run_index}:{iteration}:{map_nanos}");
            println!("iteration_decode_nanos:{run_index}:{iteration}:{decode_nanos}");
            println!("iteration_forward_nanos:{run_index}:{iteration}:{forward_nanos}");
            println!("iteration_total_nanos:{run_index}:{iteration}:{total_nanos}");

            map_samples.push(map_nanos);
            decode_samples.push(decode_nanos);
            forward_samples.push(forward_nanos);
            total_samples.push(total_nanos);

            if first_restored_logits.is_none() {
                first_restored_logits = Some(logits);
            }
        }

        map_run_p50s.push(percentile_50(map_samples));
        decode_run_p50s.push(percentile_50(decode_samples));
        forward_run_p50s.push(percentile_50(forward_samples));
        total_run_p50s.push(percentile_50(total_samples));
    }

    report_phase("map", &map_run_p50s);
    report_phase("decode", &decode_run_p50s);
    report_phase("forward", &forward_run_p50s);
    report_phase("restore_to_first_logit_total", &total_run_p50s);

    // Correctness gate -- byte-faithful mapping proof: the restored path's
    // logits must equal the direct-load path's logits within tolerance,
    // over the whole 10-class row, not a sample.
    let restored_logits = first_restored_logits.ok_or("no restored logits captured")?;
    let mut max_absolute_difference = 0.0_f32;
    let mut byte_faithful = restored_logits.len() == direct_logits.len();
    for (restored, direct) in restored_logits.iter().zip(direct_logits.iter()) {
        let difference = (restored - direct).abs();
        max_absolute_difference = max_absolute_difference.max(difference);
        if difference > CORRECTNESS_TOLERANCE {
            byte_faithful = false;
        }
    }
    println!("correctness_max_absolute_difference:{max_absolute_difference}");
    println!("correctness_byte_faithful:{byte_faithful}");
    println!("correctness_restored_argmax:{}", argmax(&restored_logits));
    println!("correctness_direct_argmax:{}", argmax(&direct_logits));

    // Context row -- the naive incumbent cold start this replaces: read the
    // .onnx from disk, parse, lower, first forward, timed the same way.
    let mut naive_run_p50s = Vec::with_capacity(RUN_COUNT);
    for run_index in 0..RUN_COUNT {
        let mut naive_samples = Vec::with_capacity(ITERATIONS_PER_RUN);
        for iteration in 0..ITERATIONS_PER_RUN {
            let naive_start = Instant::now();
            let naive_model = load_model(Path::new(MODEL_PATH))?;
            let naive_initializers: Vec<(&str, &[f32])> = naive_model
                .initializers
                .iter()
                .map(|(name, data)| (name.as_str(), data.as_slice()))
                .collect();
            let _logits = forward(&naive_model, &naive_initializers, &image)?;
            let naive_nanos = naive_start.elapsed().as_nanos();
            println!("iteration_naive_cold_start_nanos:{run_index}:{iteration}:{naive_nanos}");
            naive_samples.push(naive_nanos);
        }
        naive_run_p50s.push(percentile_50(naive_samples));
    }
    report_phase("naive_cold_start", &naive_run_p50s);

    let restore_total_p50 = percentile_50(total_run_p50s.clone());
    let naive_total_p50 = percentile_50(naive_run_p50s.clone());
    let ratio = naive_total_p50 as f64 / restore_total_p50.max(1) as f64;
    println!("naive_over_restore_ratio:{ratio:.2}");

    Ok(())
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .unwrap_or(0)
}
