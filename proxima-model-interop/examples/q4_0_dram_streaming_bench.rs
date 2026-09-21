//! Pre-registered measurement: proxima's Q4_0 matvec real DRAM streaming
//! rate, with the per-command-buffer floor AMORTIZED across ~1000+
//! dispatches inside ONE command buffer -- the prior mistake this bench
//! exists to correct was one op per buffer (floor-dominated), the exact
//! shape `gemma4_kernel_newcount_sweep.rs`'s own `bench_packed_row_q4_0_matvec`
//! group uses (`omega::execute` -> one command buffer per call).
//!
//! This bench drives the SAME production dispatch: `Op::Elementwise{Multiply}`
//! feeding `Op::Reduce{Add}` over a `QuantizedBlock::Packed{codec: Codec::Q4_0}`
//! weight input -- the exact op shape `bind`'s packed-row-blocked classifier
//! lowers to Metal's packed-row Q4_0 kernel (`omega/src/msl/packed_row_blocked_ggml.rs`),
//! the SAME shape `omega/tests/q4_0_real_checkpoint_parity.rs` proves
//! numerically correct and `gemma4_kernel_newcount_sweep.rs`'s own
//! `packed_row_q4_0_program` already exercises. No hand-rolled kernel.
//!
//! # Real tensor enumeration -- a documented deviation from the task's
//! assumed 7-tensors/layer
//!
//! The task brief assumed `attn_q/k/v/o.weight` + `ffn_gate/up/down.weight`
//! per layer (7/layer x 35 = 245). Ground truth from this checkpoint
//! (probed directly, not assumed -- principle 6) is DIFFERENT:
//! `attn_o` does not exist under that name (it is `attn_output.weight`),
//! and gemma4-E2B's cross-layer shared-KV attention
//! (`caba9263d feat(gemma4): implement E2B cross-layer shared-KV attention`)
//! means only SOME layers own `attn_k`/`attn_v` -- others reuse an earlier
//! layer's KV through `proj.weight`/`inp_gate.weight`, which exist on
//! EVERY layer instead. The real per-token Q4_0 tensor set enumerated
//! directly from the checkpoint is 275 tensors (not 245), summed below.
//! This bench uses the REAL set, in forward (layer, then tensor-name)
//! order -- the mechanically correct reading of "every layer's real
//! per-token Q4_0 weight tensors", not the assumed-uniform guess.
//!
//! # Three arms, one command-buffer structure each
//!
//! Each arm builds ONE `Vec<Op>` program covering [`REPEATS_PER_PASS`]
//! forward-order passes over the real tensor list (so N = REPEATS_PER_PASS
//! x tensor_count dispatches, ~1000+), plans it ONCE via
//! [`omega::plan_named`] (one `computeCommandEncoder`, `endEncoding`d once
//! -- that pair's own doc), then times [`omega::execute_plan_named`]
//! (commit once, `waitUntilCompleted` once) across [`MEASURE_RUNS`]
//! INTERLEAVED repeats per arm, reading `omega::metal::metal_stage_totals()`
//! -- the `gpu_exec_ticks` clean aggregate (`GPUEnd - GPUStart` of that ONE
//! command buffer) -- immediately after each call.
//!
//! 1. **matvec**: real Q4_0 weight bytes, real k per tensor.
//! 2. **floor control**: same tensors, same rows, K TRUNCATED to one
//!    Q4_0 block (32 elements, 18 bytes) per row -- near-zero bytes moved,
//!    isolates the per-dispatch launch/drain floor F.
//! 3. **copy ceiling control**: a trivial `Op::Elementwise{Add}` kernel
//!    (`x + 0.0`, forces a real read+write, not an aliasable `Identity`)
//!    over F32 buffers sized to match each tensor's own Q4_0 byte count --
//!    the box's own measured DRAM streaming ceiling, not a quoted spec number.
//!
//! # Dispatch-count assertion (principle: N==0 or wrong N is RED)
//!
//! Every arm's dispatch count is asserted TWICE: once against this bench's
//! own construction count (`REPEATS_PER_PASS * tensor_count`), and once
//! against the REAL bound-op count from `proxima_tensor::bind` (CPU-side,
//! no GPU dispatch) filtered to the op kind each arm's roots lower to
//! (`Reduce` for matvec/floor, `Elementwise` for copy) -- proving the graph
//! compiler did not silently fuse or dedup dispatches sharing input nodes.
//!
//! # Run
//!
//! ```sh
//! ollama stop gemma4:e2b-it-qat 2>/dev/null || true
//! CARGO_TARGET_DIR=<scratch>/target-q4dram cargo run -p proxima-model-interop \
//!     --release --features "metal instrument" --example q4_0_dram_streaming_bench
//! ```

#![cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::fs::File;
use std::sync::OnceLock;
use std::time::Instant;

use memmap2::{Mmap, MmapOptions};
use omega::metal::metal_stage_totals;
use omega::{execute_plan_named, plan_named};
use proxima_gguf::parse_complete;
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::types::GgmlType;
use proxima_primitives::Codec;
use proxima_tensor::instrument::ticks_to_nanos;
use proxima_tensor::{
    BoundOpKind, DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce,
    ReduceInit, ScalarOp, append, bind, infer, map,
};

const MODEL_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/\
sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

/// Full forward-order passes encoded into ONE command buffer -- amortizes
/// the per-command-buffer floor below 1% (task brief's own target).
const REPEATS_PER_PASS: usize = 5;
/// Interleaved measurement repeats per arm.
const MEASURE_RUNS: usize = 5;
const Q4_0_BLOCK_ELEMENTS: u32 = 32;
const Q4_0_BLOCK_BYTES: usize = 18;

fn model_bytes() -> &'static [u8] {
    static BYTES: OnceLock<Mmap> = OnceLock::new();
    &BYTES.get_or_init(|| {
        let file = File::open(MODEL_PATH).expect("open gemma4-E2B blob");
        // SAFETY: mapping lives in a process-lifetime `OnceLock`, same
        // convention every real-checkpoint bench in this crate uses.
        unsafe { MmapOptions::new().map(&file) }.expect("mmap gemma4-E2B blob")
    })[..]
}

fn parsed_gguf() -> &'static ParsedGguf {
    static PARSED: OnceLock<ParsedGguf> = OnceLock::new();
    PARSED.get_or_init(|| parse_complete(model_bytes()).expect("parse gemma4-E2B header"))
}

struct RealTensor {
    name: String,
    bytes: &'static [u8],
    rows: u32,
    k: u32,
}

/// Every real `blk.N.*` `Q4_0` tensor in this checkpoint, forward order
/// (layer ascending, tensor name ascending within a layer for a
/// deterministic tie-break). Ground truth from the parsed directory, not
/// an assumed 7-tensors/layer convention -- see module doc.
fn forward_order_q4_0_tensors() -> Vec<RealTensor> {
    let parsed = parsed_gguf();
    let bytes = model_bytes();
    let mut by_layer: BTreeMap<u32, Vec<String>> = BTreeMap::new();
    for tensor in &parsed.tensors {
        if tensor.ggml_type != GgmlType::Q4_0 {
            continue;
        }
        let Some(rest) = tensor.name.strip_prefix("blk.") else {
            continue;
        };
        let Some((layer_str, _suffix)) = rest.split_once('.') else {
            continue;
        };
        let Ok(layer) = layer_str.parse::<u32>() else {
            continue;
        };
        by_layer.entry(layer).or_default().push(tensor.name.clone());
    }

    let mut out = Vec::new();
    for (_layer, mut names) in by_layer {
        names.sort();
        for name in names {
            let tensor = parsed
                .tensors
                .iter()
                .find(|candidate| candidate.name == name)
                .expect("name came from this same directory");
            let range = parsed
                .tensor_data_range(tensor, bytes.len() as u64)
                .expect("tensor byte range within file bounds");
            out.push(RealTensor {
                name,
                bytes: &bytes[range.start as usize..range.end as usize],
                k: tensor.dims[0] as u32,
                rows: tensor.dims[1] as u32,
            });
        }
    }
    out
}

/// Truncates real Q4_0 packed bytes to ONE block (32 elements, 18 bytes)
/// per row -- keeps every row's own valid block-0 encoding (scale +
/// nibbles), drops every other block. Row layout is `blocks_per_row`
/// contiguous 18-byte blocks per row, row-major
/// (`q4_0_real_checkpoint_parity.rs:201-207`'s own `row_bytes` convention).
fn truncate_to_one_block_per_row(bytes: &[u8], rows: u32, k: u32) -> Vec<u8> {
    let blocks_per_row = (k / Q4_0_BLOCK_ELEMENTS) as usize;
    let row_bytes = blocks_per_row * Q4_0_BLOCK_BYTES;
    let mut out = Vec::with_capacity(rows as usize * Q4_0_BLOCK_BYTES);
    for row in 0..rows as usize {
        let start = row * row_bytes;
        out.extend_from_slice(&bytes[start..start + Q4_0_BLOCK_BYTES]);
    }
    out
}

fn deterministic_f32(seed: u64, count: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as u32 as f32) / (u32::MAX as f32)
        })
        .collect()
}

fn append_packed_row_matvec(program: &mut Vec<Op>, weight: NodeId, activation: NodeId, label: String) -> NodeId {
    let product = append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (activation, IndexMap::Affine(map::projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    append(
        program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some(label),
        }),
    )
}

/// `x + zero` -- forces a genuine elementwise read+write dispatch (never
/// aliasable the way a bare `ScalarOp::Identity` pass-through might be).
fn append_copy_dispatch(program: &mut Vec<Op>, data: NodeId, zero: NodeId, label: String) -> NodeId {
    append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: vec![
                (data, IndexMap::Affine(map::projection(1, &[0]))),
                (zero, IndexMap::Affine(map::projection(1, &[]))),
            ],
            name: Some(label),
        },
    )
}

struct ArmBuild<'a> {
    program: Vec<Op>,
    roots: Vec<NodeId>,
    named: Vec<(String, QuantizedBlock<'a>)>,
    total_bytes_per_pass: u64,
}

/// Builds the matvec or floor-control arm. `truncated_storage` and
/// `activation_storage` are caller-owned, filled here in a first pass
/// (no borrows yet), then borrowed from in a second pass -- avoids a
/// self-referential struct while keeping every buffer alive exactly as
/// long as `main`'s own scope, where every `execute_plan_named` call
/// also lives.
fn build_matvec_arm<'a>(
    tensors: &'a [RealTensor],
    truncate: bool,
    truncated_storage: &'a mut Vec<Vec<u8>>,
    activation_storage: &'a mut Vec<Vec<f32>>,
) -> ArmBuild<'a> {
    if truncate {
        for tensor in tensors {
            truncated_storage.push(truncate_to_one_block_per_row(tensor.bytes, tensor.rows, tensor.k));
        }
    }
    for (index, tensor) in tensors.iter().enumerate() {
        let k_used = if truncate { Q4_0_BLOCK_ELEMENTS } else { tensor.k };
        activation_storage.push(deterministic_f32(index as u64 + 1, k_used as usize));
    }

    let mut program = Vec::new();
    let mut named: Vec<(String, QuantizedBlock<'a>)> = Vec::new();
    let mut weight_nodes = Vec::with_capacity(tensors.len());
    let mut activation_nodes = Vec::with_capacity(tensors.len());
    let mut total_bytes_per_pass: u64 = 0;

    for (index, tensor) in tensors.iter().enumerate() {
        let k_used = if truncate { Q4_0_BLOCK_ELEMENTS } else { tensor.k };
        let weight_bytes: &'a [u8] = if truncate {
            &truncated_storage[index]
        } else {
            tensor.bytes
        };
        total_bytes_per_pass += weight_bytes.len() as u64;

        let weight_name = format!("w{index}");
        let weight_node = append(
            &mut program,
            Op::Input {
                dtype: DType::UInt8,
                shape: vec![Extent::Static(tensor.rows), Extent::Static(k_used)],
                name: Some(weight_name.clone()),
            },
        );
        named.push((
            weight_name,
            QuantizedBlock::Packed {
                codec: Codec::Q4_0,
                bytes: weight_bytes,
            },
        ));
        weight_nodes.push(weight_node);

        let activation_name = format!("a{index}");
        let activation_node = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(k_used), Extent::Static(1)],
                name: Some(activation_name.clone()),
            },
        );
        named.push((activation_name, QuantizedBlock::Float32(&activation_storage[index])));
        activation_nodes.push(activation_node);
    }

    let mut roots = Vec::with_capacity(tensors.len() * REPEATS_PER_PASS);
    for repeat in 0..REPEATS_PER_PASS {
        for index in 0..tensors.len() {
            let label = format!("{}_{}_r{repeat}", tensors[index].name, if truncate { "floor" } else { "matvec" });
            let root = append_packed_row_matvec(&mut program, weight_nodes[index], activation_nodes[index], label);
            roots.push(root);
        }
    }

    ArmBuild {
        program,
        roots,
        named,
        total_bytes_per_pass,
    }
}

fn build_copy_arm<'a>(tensors: &'a [RealTensor], data_storage: &'a mut Vec<Vec<f32>>) -> ArmBuild<'a> {
    for (index, tensor) in tensors.iter().enumerate() {
        let byte_count = tensor.bytes.len();
        let elems = byte_count / 4;
        data_storage.push(deterministic_f32(index as u64 + 5000, elems.max(1)));
    }

    let mut program = Vec::new();
    let mut named: Vec<(String, QuantizedBlock<'a>)> = Vec::new();
    let zero_data: &'a [f32] = {
        // leaked once, process-lifetime -- a single shared scalar constant
        // read (broadcast) by every one of the arm's dispatches, negligible
        // next to the multi-GB data buffers this arm actually streams.
        static ZERO: OnceLock<Vec<f32>> = OnceLock::new();
        &ZERO.get_or_init(|| vec![0.0f32])[..]
    };
    let zero_node = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: Vec::new(),
            name: Some("zero".to_string()),
        },
    );
    named.push(("zero".to_string(), QuantizedBlock::Float32(zero_data)));

    let mut data_nodes = Vec::with_capacity(tensors.len());
    let mut total_bytes_per_pass: u64 = 0;
    for (index, tensor) in tensors.iter().enumerate() {
        let elems = data_storage[index].len() as u32;
        total_bytes_per_pass += (elems as u64) * 4;
        let data_name = format!("c{index}");
        let data_node = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(elems)],
                name: Some(data_name.clone()),
            },
        );
        named.push((data_name, QuantizedBlock::Float32(&data_storage[index])));
        data_nodes.push(data_node);
        let _ = tensor;
    }

    let mut roots = Vec::with_capacity(tensors.len() * REPEATS_PER_PASS);
    for repeat in 0..REPEATS_PER_PASS {
        for index in 0..tensors.len() {
            let label = format!("{}_copy_r{repeat}", tensors[index].name);
            let root = append_copy_dispatch(&mut program, data_nodes[index], zero_node, label);
            roots.push(root);
        }
    }

    ArmBuild {
        program,
        roots,
        named,
        total_bytes_per_pass,
    }
}

struct Stats {
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
    cov_pct: f64,
}

fn stats(samples_ms: &[f64]) -> Stats {
    let len = samples_ms.len();
    let mean = samples_ms.iter().sum::<f64>() / len as f64;
    let variance = samples_ms.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / len as f64;
    let cov_pct = if mean.abs() > f64::MIN_POSITIVE {
        100.0 * variance.sqrt() / mean
    } else {
        0.0
    };
    Stats {
        mean_ms: mean,
        min_ms: samples_ms.iter().cloned().fold(f64::INFINITY, f64::min),
        max_ms: samples_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        cov_pct,
    }
}

/// CPU-side, no GPU dispatch: proves the plan this arm is about to run
/// really does encode `expected` dispatches of `kind`, not fewer (silent
/// fusion/dedup across shared input nodes) or more.
fn assert_dispatch_count(program: &[Op], roots: &[NodeId], expected: usize, arm: &str, reduce_kind: bool) {
    let shapes = infer(program, &[]).unwrap_or_else(|error| panic!("{arm}: program infers: {error:?}"));
    let resolved =
        bind(program, &shapes, roots, NumericPolicy::llama_relaxed()).unwrap_or_else(|error| panic!("{arm}: program binds: {error:?}"));
    let actual = resolved
        .iter()
        .filter(|bound| {
            if reduce_kind {
                matches!(bound.kind, BoundOpKind::Reduce { .. })
            } else {
                matches!(bound.kind, BoundOpKind::Elementwise { .. })
            }
        })
        .count();
    assert_eq!(
        actual, expected,
        "{arm}: dispatch count RED -- expected {expected}, bind() resolved {actual} \
         (own construction also expected roots.len()={})",
        roots.len()
    );
    assert_eq!(
        roots.len(),
        expected,
        "{arm}: own construction count drifted from expected"
    );
    println!("q4_0_dram_streaming_bench arm={arm} dispatch_count_asserted N={actual}");
}

struct PreparedArm<'a> {
    arm: &'static str,
    plan: omega::Plan,
    named_refs: Vec<(&'a str, QuantizedBlock<'a>)>,
    total_bytes_per_pass: u64,
    dispatch_count: usize,
    wall_ms: Vec<f64>,
    gpu_ms: Vec<f64>,
}

fn prepare_arm<'a>(build: &'a ArmBuild<'a>, arm: &'static str) -> PreparedArm<'a> {
    let named_refs: Vec<(&'a str, QuantizedBlock<'a>)> =
        build.named.iter().map(|(name, block)| (name.as_str(), *block)).collect();
    let plan = plan_named(
        &build.program,
        &[],
        &named_refs,
        &build.roots,
        NumericPolicy::llama_relaxed(),
    )
    .unwrap_or_else(|error| panic!("{arm}: metal plans the batched program: {error:?}"));
    // untimed warm-up -- absorbs one-time pipeline compile cost, never
    // mixed into the timed samples below.
    execute_plan_named(&plan, &named_refs).expect("warm-up run");
    let _ = metal_stage_totals();
    PreparedArm {
        arm,
        plan,
        named_refs,
        total_bytes_per_pass: build.total_bytes_per_pass,
        dispatch_count: build.roots.len(),
        wall_ms: Vec::with_capacity(MEASURE_RUNS),
        gpu_ms: Vec::with_capacity(MEASURE_RUNS),
    }
}

/// One measured commit+wait for this arm's already-planned command buffer.
/// Printed immediately, per-run, so every cell in the summary table is
/// grep-able from this run's own stdout (principle 19: results traced to
/// records, not only a summary line).
fn step_arm(prepared: &mut PreparedArm<'_>, run: usize) {
    let started = Instant::now();
    execute_plan_named(&prepared.plan, &prepared.named_refs)
        .unwrap_or_else(|error| panic!("{}: timed run {run}: {error:?}", prepared.arm));
    let wall = started.elapsed().as_secs_f64() * 1e3;
    let stage = metal_stage_totals();
    let gpu_exec_ns = ticks_to_nanos(stage.gpu_exec_ticks);
    let gpu = gpu_exec_ns as f64 / 1e6;
    prepared.wall_ms.push(wall);
    prepared.gpu_ms.push(gpu);
    let total_bytes = prepared.total_bytes_per_pass * REPEATS_PER_PASS as u64;
    let gbps = total_bytes as f64 / (gpu_exec_ns as f64 / 1e9) / 1e9;
    println!(
        "q4_0_dram_streaming_bench arm={} run={run} wall_ms={wall:.4} gpu_ms={gpu:.4} \
         total_bytes={total_bytes} gbps={gbps:.3}",
        prepared.arm
    );
}

fn summarize_arm(prepared: &PreparedArm<'_>) {
    let wall_stats = stats(&prepared.wall_ms);
    let gpu_stats = stats(&prepared.gpu_ms);
    let total_bytes = prepared.total_bytes_per_pass * REPEATS_PER_PASS as u64;
    let gbps_from_mean_gpu = total_bytes as f64 / (gpu_stats.mean_ms / 1e3) / 1e9;
    let us_per_dispatch = gpu_stats.mean_ms * 1000.0 / prepared.dispatch_count as f64;
    println!(
        "q4_0_dram_streaming_bench SUMMARY arm={} N={} total_bytes={total_bytes} \
         wall_ms_mean={:.4} wall_ms_min={:.4} wall_ms_max={:.4} wall_cov_pct={:.2} \
         gpu_ms_mean={:.4} gpu_ms_min={:.4} gpu_ms_max={:.4} gpu_cov_pct={:.2} \
         gbps={gbps_from_mean_gpu:.3} us_per_dispatch={us_per_dispatch:.4}",
        prepared.arm,
        prepared.dispatch_count,
        wall_stats.mean_ms,
        wall_stats.min_ms,
        wall_stats.max_ms,
        wall_stats.cov_pct,
        gpu_stats.mean_ms,
        gpu_stats.min_ms,
        gpu_stats.max_ms,
        gpu_stats.cov_pct,
    );
}

fn main() {
    println!("q4_0_dram_streaming_bench host_loadout recorded via `uptime` in the caller's run log, not in-process");

    let tensors = forward_order_q4_0_tensors();
    let total_real_bytes: u64 = tensors.iter().map(|tensor| tensor.bytes.len() as u64).sum();
    println!(
        "q4_0_dram_streaming_bench tensor_count={} total_q4_0_bytes_one_pass={total_real_bytes} \
         ({:.3} GiB)",
        tensors.len(),
        total_real_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    let expected_matvec_dispatches = tensors.len() * REPEATS_PER_PASS;

    // ---- matvec arm ----
    let mut matvec_truncated: Vec<Vec<u8>> = Vec::new();
    let mut matvec_activation: Vec<Vec<f32>> = Vec::new();
    let matvec_build = build_matvec_arm(&tensors, false, &mut matvec_truncated, &mut matvec_activation);
    assert_dispatch_count(
        &matvec_build.program,
        &matvec_build.roots,
        expected_matvec_dispatches,
        "matvec",
        true,
    );

    // ---- floor control arm ----
    let mut floor_truncated: Vec<Vec<u8>> = Vec::new();
    let mut floor_activation: Vec<Vec<f32>> = Vec::new();
    let floor_build = build_matvec_arm(&tensors, true, &mut floor_truncated, &mut floor_activation);
    assert_dispatch_count(
        &floor_build.program,
        &floor_build.roots,
        expected_matvec_dispatches,
        "floor",
        true,
    );

    // ---- copy ceiling arm ----
    let mut copy_data: Vec<Vec<f32>> = Vec::new();
    let copy_build = build_copy_arm(&tensors, &mut copy_data);
    assert_dispatch_count(
        &copy_build.program,
        &copy_build.roots,
        expected_matvec_dispatches,
        "copy",
        false,
    );

    println!(
        "q4_0_dram_streaming_bench PLAN_SUMMARY matvec_bytes_per_pass={} floor_bytes_per_pass={} copy_bytes_per_pass={}",
        matvec_build.total_bytes_per_pass, floor_build.total_bytes_per_pass, copy_build.total_bytes_per_pass
    );

    // All three plans built and warmed up BEFORE any timed sample is
    // taken, so the interleave below is a genuine per-run rotation
    // (matvec run i, floor run i, copy run i) across three simultaneously
    // live Metal plans -- not three sequential arm-complete blocks.
    let mut matvec_prepared = prepare_arm(&matvec_build, "matvec");
    let mut floor_prepared = prepare_arm(&floor_build, "floor");
    let mut copy_prepared = prepare_arm(&copy_build, "copy");

    for run in 0..MEASURE_RUNS {
        step_arm(&mut matvec_prepared, run);
        step_arm(&mut floor_prepared, run);
        step_arm(&mut copy_prepared, run);
    }

    summarize_arm(&matvec_prepared);
    summarize_arm(&floor_prepared);
    summarize_arm(&copy_prepared);

    let matvec_gbps = {
        let stats = stats(&matvec_prepared.gpu_ms);
        (matvec_prepared.total_bytes_per_pass * REPEATS_PER_PASS as u64) as f64 / (stats.mean_ms / 1e3) / 1e9
    };
    let floor_us_per_dispatch = {
        let stats = stats(&floor_prepared.gpu_ms);
        stats.mean_ms * 1000.0 / floor_prepared.dispatch_count as f64
    };
    let copy_gbps_read_only = {
        let stats = stats(&copy_prepared.gpu_ms);
        (copy_prepared.total_bytes_per_pass * REPEATS_PER_PASS as u64) as f64 / (stats.mean_ms / 1e3) / 1e9
    };
    let copy_gbps_read_write = copy_gbps_read_only * 2.0;
    let branch = if matvec_gbps <= 110.0 {
        "kernel IS the gap (<=110 GB/s) -- port llama's 2-lanes-per-block/4-rows-per-simdgroup/activation-in-registers structure"
    } else if matvec_gbps >= 150.0 {
        "bound is wrong (>=150 GB/s) -- excess is in non-matvec dispatches or per-token buffer work"
    } else {
        "between 110-150 GB/s -- port anyway, largest single term"
    };
    println!(
        "q4_0_dram_streaming_bench DECISION_RULE matvec_gbps={matvec_gbps:.3} \
         floor_us_per_dispatch={floor_us_per_dispatch:.4} \
         copy_ceiling_gbps_read_only={copy_gbps_read_only:.3} \
         copy_ceiling_gbps_read_plus_write={copy_gbps_read_write:.3} branch={branch:?}"
    );
}
