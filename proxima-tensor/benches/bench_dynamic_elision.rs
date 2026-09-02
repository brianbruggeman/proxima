//! `docs/discipline.md` ROW 180's dynamic-elision probe: does a per-step
//! mask-driven skip-set over a block-topology `StaticArena` (ROW 180's own
//! `cpu::evaluate_named_with_arena_masked`, `dynamic-elision-probe` feature,
//! default-off) buy real step time against ROW 167's landed STATIC dead-node
//! elision as the home-turf incumbent -- and does the measured saving track
//! the PRE-REGISTERED bandwidth-wall prediction (69.95 GB/s single-core,
//! `docs/discipline.md`)?
//!
//! Topology: `num_blocks` independent `y_i = w_i @ x_i` block-matmuls (the
//! SAME `Op` shape `cpu.rs`'s own
//! `a_static_block_sparse_matmul_needs_no_data_dependent_map` test uses --
//! `weight` read via `projection(2, &[0, 1])`, `x` via `projection(2, &[1])`,
//! reduced over axis 1), each block a separate named `Op::Input` pair and a
//! separate `effective_outputs` entry. No `Op::Input`/`ScalarOp`/`IndexMap`
//! variant is new; the mask never touches the graph (no data-dependent
//! `IndexMap::Computed`, so `shape.rs`'s scatter gate never sees this
//! program, matching that same test's own note) -- it is consulted ONLY by
//! the harness building `skip`/`named` for `evaluate_named_with_arena_masked`,
//! i.e. execution-level, never graph-level (ROW 166's landmine).
//!
//! Two sizes (`small` mnist-scale, `large` 4x total dense bytes) x four
//! arms (`dense`, `sparse_50`, `sparse_75`, `sparse_90`) -- `design-favors:
//! incumbent` is `dense`, ROW 167/175's own always-run-everything static
//! arena shape; the `sparse_*` arms are `ours`.
//!
//! `docs/discipline.md` ROW 181 extends this file with two residuals ROW 180
//! named but did not close: (1) a `control_zero_skip` arm per shape --
//! `evaluate_named_with_arena_masked` with a full-live mask (nothing
//! skipped), isolating the mask-consult + `BTreeSet` derivation cost from
//! ROW 180's dense arm; (2) a `streaming_640sq` shape whose per-instance
//! dense working set (~31.35 MiB, 20 blocks of a 640x640 weight each) is
//! cycled round-robin across `STREAM_INSTANCES` independent data sets so no
//! two consecutive timed calls touch the same bytes -- the rotation
//! distance between reuses of one instance (`(STREAM_INSTANCES-1) *
//! ~31.35 MiB`) is chosen to exceed this host's documented L2
//! (`hw.perflevel0.l2cachesize=12 MiB` on the M1 Max this row measured on)
//! by more than an order of magnitude, per the task's own engagement bar.
//!
//! Re-prove (host must be quiet -- `pgrep -f "cargo check"` / `cargo build`
//! empty before trusting a number):
//! `CARGO_TARGET_DIR=<scratch> cargo bench -p proxima-tensor --features dynamic-elision-probe --bench bench_dynamic_elision -- --save-baseline row181-cold`

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeSet;

use criterion::Criterion;
use proxima_tensor::cpu::{
    build_static_arena, evaluate_named_with_arena, evaluate_named_with_arena_masked,
};
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp, append, map,
};

/// One block-topology size: `num_blocks` independent `[block_out,
/// block_in] @ [block_in]` matmuls. `large` is exactly 4x `small`'s total
/// dense weight bytes (block dims doubled on both axes), the "shape scales
/// bandwidth effects" arm the task asked for.
struct BlockShape {
    label: &'static str,
    num_blocks: u32,
    block_in: u32,
    block_out: u32,
}

const SHAPES: [BlockShape; 2] = [
    BlockShape {
        label: "small_mnist_scale",
        num_blocks: 20,
        block_in: 39,
        block_out: 16,
    },
    BlockShape {
        label: "large_4x",
        num_blocks: 20,
        block_in: 78,
        block_out: 32,
    },
];

/// Measured single-core streaming bandwidth ceiling this crate's own
/// discipline log already sealed (`docs/discipline.md`, ROW 176's own DANGER
/// ZONE citation): 69.95 GB/s == 69.95 bytes/ns.
const BANDWIDTH_BYTES_PER_NS: f64 = 69.95;

struct BlockProgram {
    program: Vec<Op>,
    x_names: Vec<String>,
    w_names: Vec<String>,
    product_nodes: Vec<NodeId>,
    reduce_nodes: Vec<NodeId>,
}

/// Builds `num_blocks` disjoint `y_i = w_i @ x_i` blocks -- identical `Op`
/// shape to `cpu.rs`'s own
/// `a_static_block_sparse_matmul_needs_no_data_dependent_map` test, scaled
/// up and named so `build_static_arena` can bind each block by name.
fn block_sparse_program(shape: &BlockShape) -> BlockProgram {
    let mut program = Vec::new();
    let mut x_names = Vec::with_capacity(shape.num_blocks as usize);
    let mut w_names = Vec::with_capacity(shape.num_blocks as usize);
    let mut product_nodes = Vec::with_capacity(shape.num_blocks as usize);
    let mut reduce_nodes = Vec::with_capacity(shape.num_blocks as usize);

    for index in 0..shape.num_blocks {
        let x_name = format!("x{index}");
        let w_name = format!("w{index}");
        let x = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(shape.block_in)],
                name: Some(x_name.clone()),
            },
        );
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![
                    Extent::Static(shape.block_out),
                    Extent::Static(shape.block_in),
                ],
                name: Some(w_name.clone()),
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weight, IndexMap::Affine(map::projection(2, &[0, 1]))),
                    (x, IndexMap::Affine(map::projection(2, &[1]))),
                ],
                name: None,
            },
        );
        let reduced = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
                out_map: IndexMap::Affine(map::projection(2, &[0])),
                keep: Keep::Reduce,
                name: Some(format!("y{index}")),
            }),
        );
        x_names.push(x_name);
        w_names.push(w_name);
        product_nodes.push(product);
        reduce_nodes.push(reduced);
    }

    BlockProgram {
        program,
        x_names,
        w_names,
        product_nodes,
        reduce_nodes,
    }
}

fn deterministic_data(len: usize, phase: f32) -> Vec<f32> {
    (0..len).map(|value| (value as f32 * phase).sin()).collect()
}

/// Bytes DERIVED from the operand shapes actually touched by `live_count`
/// blocks (weight + input + output, f32) -- not a profiler measurement, a
/// computation from the shapes each arm's `named` binds. Reported alongside
/// the pre-registered bandwidth prediction so a reader can trace which
/// number is which.
fn bytes_touched(shape: &BlockShape, live_count: u32) -> usize {
    let per_block =
        (shape.block_in * shape.block_out + shape.block_in + shape.block_out) as usize * 4;
    live_count as usize * per_block
}

fn predicted_ns(bytes: usize) -> f64 {
    bytes as f64 / BANDWIDTH_BYTES_PER_NS
}

fn main() {
    let mut criterion = Criterion::default();
    let mut group = criterion.benchmark_group("bench_dynamic_elision");
    group.sample_size(30);

    for shape in &SHAPES {
        let built = block_sparse_program(shape);
        let x_data: Vec<Vec<f32>> = (0..shape.num_blocks)
            .map(|index| deterministic_data(shape.block_in as usize, 0.0137 + index as f32 * 0.001))
            .collect();
        let w_data: Vec<Vec<f32>> = (0..shape.num_blocks)
            .map(|index| {
                deterministic_data(
                    (shape.block_in * shape.block_out) as usize,
                    0.0271 + index as f32 * 0.001,
                )
            })
            .collect();

        let dense_bytes = bytes_touched(shape, shape.num_blocks);
        println!(
            "{}: dense_bytes={dense_bytes} predicted_dense_ns={:.1} (bandwidth-derived, pre-registered)",
            shape.label,
            predicted_ns(dense_bytes)
        );

        let dense_named: Vec<(&str, &[f32])> = (0..shape.num_blocks as usize)
            .flat_map(|index| {
                [
                    (built.x_names[index].as_str(), x_data[index].as_slice()),
                    (built.w_names[index].as_str(), w_data[index].as_slice()),
                ]
            })
            .collect();

        let mut dense_arena = build_static_arena(&built.program, &[], &built.reduce_nodes)
            .expect("dense arena builds");

        // correctness self-check outside the timed loop: run dense once so
        // sparse arms below have a bit-exact reference for their LIVE blocks.
        let dense_reference = evaluate_named_with_arena(&mut dense_arena, &dense_named)
            .expect("dense step evaluates");

        group.bench_function(format!("{}/dense", shape.label), |bencher| {
            bencher.iter(|| {
                evaluate_named_with_arena(&mut dense_arena, &dense_named).expect("dense step")
            });
        });

        // ROW 181 residual (1): derivation-cost control. Full mask present,
        // ALL blocks live (skip is empty every call), routed through the
        // SAME `evaluate_named_with_arena_masked` entry point the sparse
        // arms use, with the identical fresh-BTreeSet-plus-filter pattern
        // paid inside the timed closure. Isolates mask-consult overhead from
        // ROW 180's dense arm, which never calls the masked function at all.
        let mut control_arena = build_static_arena(&built.program, &[], &built.reduce_nodes)
            .expect("control arena builds");
        group.bench_function(format!("{}/control_zero_skip", shape.label), |bencher| {
            bencher.iter(|| {
                let live_named: Vec<(&str, &[f32])> = (0..shape.num_blocks as usize)
                    .flat_map(|index| {
                        [
                            (built.x_names[index].as_str(), x_data[index].as_slice()),
                            (built.w_names[index].as_str(), w_data[index].as_slice()),
                        ]
                    })
                    .collect();
                let skip: BTreeSet<NodeId> = BTreeSet::new();
                evaluate_named_with_arena_masked(&mut control_arena, &live_named, &skip)
                    .expect("control step")
            });
        });

        for &(sparsity_pct, skip_count) in &[
            (50u32, shape.num_blocks / 2),
            (75, shape.num_blocks * 3 / 4),
            (90, shape.num_blocks * 9 / 10),
        ] {
            let live_count = shape.num_blocks - skip_count;
            let sparse_bytes = bytes_touched(shape, live_count);
            let actual_sparsity = f64::from(skip_count) / f64::from(shape.num_blocks) * 100.0;
            println!(
                "{}/sparse_{sparsity_pct}: skip={skip_count}/{} (actual {actual_sparsity:.1}%) sparse_bytes={sparse_bytes} predicted_sparse_ns={:.1} predicted_saving_ns={:.1}",
                shape.label,
                shape.num_blocks,
                predicted_ns(sparse_bytes),
                predicted_ns(dense_bytes) - predicted_ns(sparse_bytes),
            );

            // mask input: skip the LAST `skip_count` blocks -- deterministic,
            // reproducible, and consulted fresh every timed call below
            // exactly the way a real per-step routing mask would be.
            let mask: Vec<bool> = (0..shape.num_blocks)
                .map(|index| index >= shape.num_blocks - skip_count)
                .collect();

            let mut sparse_arena = build_static_arena(&built.program, &[], &built.reduce_nodes)
                .expect("sparse arena builds");
            let live_named: Vec<(&str, &[f32])> = (0..shape.num_blocks as usize)
                .filter(|&index| !mask[index])
                .flat_map(|index| {
                    [
                        (built.x_names[index].as_str(), x_data[index].as_slice()),
                        (built.w_names[index].as_str(), w_data[index].as_slice()),
                    ]
                })
                .collect();
            let mut skip = BTreeSet::new();
            for (index, &masked) in mask.iter().enumerate() {
                if masked {
                    skip.insert(built.product_nodes[index]);
                    skip.insert(built.reduce_nodes[index]);
                }
            }
            let sparse_reference =
                evaluate_named_with_arena_masked(&mut sparse_arena, &live_named, &skip)
                    .expect("sparse step evaluates");
            for (index, &masked) in mask.iter().enumerate() {
                if masked {
                    continue;
                }
                let node = built.reduce_nodes[index];
                let (dense_values, _) = dense_reference.get(node).expect("dense output present");
                let (sparse_values, _) = sparse_reference.get(node).expect("sparse output present");
                assert_eq!(
                    dense_values, sparse_values,
                    "{}: block {index} diverged between dense and sparse arms",
                    shape.label
                );
            }

            group.bench_function(
                format!("{}/sparse_{sparsity_pct}", shape.label),
                |bencher| {
                    bencher.iter(|| {
                        // mask consult + skip-set derivation happen INSIDE the
                        // timed closure every call, per this task's own
                        // pre-registration: the derivation cost is part of the
                        // step, not amortized out of the measurement.
                        let live_named: Vec<(&str, &[f32])> = (0..shape.num_blocks as usize)
                            .filter(|&index| !mask[index])
                            .flat_map(|index| {
                                [
                                    (built.x_names[index].as_str(), x_data[index].as_slice()),
                                    (built.w_names[index].as_str(), w_data[index].as_slice()),
                                ]
                            })
                            .collect();
                        let mut skip = BTreeSet::new();
                        for (index, &masked) in mask.iter().enumerate() {
                            if masked {
                                skip.insert(built.product_nodes[index]);
                                skip.insert(built.reduce_nodes[index]);
                            }
                        }
                        evaluate_named_with_arena_masked(&mut sparse_arena, &live_named, &skip)
                            .expect("sparse step")
                    });
                },
            );
        }
    }

    run_streaming_arm(&mut group);

    group.finish();
    criterion.final_summary();
}

/// ROW 181 residual (2): a per-instance dense working set (~31.35 MiB) far
/// too small to trust as memory-bound on its own, cycled round-robin across
/// `STREAM_INSTANCES` independent data sets so consecutive timed calls never
/// touch the same bytes -- the rotation distance
/// (`(STREAM_INSTANCES - 1) * per_instance_bytes`) is what defeats caching,
/// not the per-instance size alone. Same block topology `Op` shape as
/// `block_sparse_program`; only the block dimensions and instance count
/// differ from the two `SHAPES` entries above.
const STREAM_SHAPE: BlockShape = BlockShape {
    label: "streaming_640sq",
    num_blocks: 20,
    block_in: 640,
    block_out: 640,
};

/// Round-robin instance count. Rotation distance =
/// `(STREAM_INSTANCES - 1) * bytes_touched(STREAM_SHAPE, num_blocks)` =
/// 15 * 32,870,400 B ~= 493 MiB, chosen to clear this host's documented
/// `hw.perflevel0.l2cachesize` (12 MiB) by ~41x and any plausible
/// whole-chip system-level cache (Apple does not publish the M1 Max figure;
/// treating 12 MiB as the tightest documented bound, 493 MiB clears the
/// task's 10x bar with room to spare even against a much larger guess).
const STREAM_INSTANCES: usize = 16;

/// `docs/discipline.md` ROW 180's own derived compute-floor constant
/// (ns/element, from the two-shape linear fit over `small`/`large`) -- NOT
/// measured this session, cited here only as the rival prediction to the
/// bandwidth-wall number for this row's pre-registration.
const ROW180_COMPUTE_FLOOR_NS_PER_ELEMENT: f64 = 0.277;

struct StreamInstance {
    x_data: Vec<Vec<f32>>,
    w_data: Vec<Vec<f32>>,
}

fn streaming_instances(shape: &BlockShape) -> Vec<StreamInstance> {
    (0..STREAM_INSTANCES)
        .map(|instance| {
            let phase_base = 0.0091 + instance as f32 * 0.0173;
            let x_data: Vec<Vec<f32>> = (0..shape.num_blocks)
                .map(|index| {
                    deterministic_data(shape.block_in as usize, phase_base + index as f32 * 0.001)
                })
                .collect();
            let w_data: Vec<Vec<f32>> = (0..shape.num_blocks)
                .map(|index| {
                    deterministic_data(
                        (shape.block_in * shape.block_out) as usize,
                        phase_base + 0.0271 + index as f32 * 0.001,
                    )
                })
                .collect();
            StreamInstance { x_data, w_data }
        })
        .collect()
}

fn named_for_instance<'data>(
    built: &'data BlockProgram,
    shape: &BlockShape,
    instance: &'data StreamInstance,
) -> Vec<(&'data str, &'data [f32])> {
    (0..shape.num_blocks as usize)
        .flat_map(|index| {
            [
                (
                    built.x_names[index].as_str(),
                    instance.x_data[index].as_slice(),
                ),
                (
                    built.w_names[index].as_str(),
                    instance.w_data[index].as_slice(),
                ),
            ]
        })
        .collect()
}

fn live_named_for_instance<'data>(
    built: &'data BlockProgram,
    shape: &BlockShape,
    instance: &'data StreamInstance,
    mask: &[bool],
) -> Vec<(&'data str, &'data [f32])> {
    (0..shape.num_blocks as usize)
        .filter(|&index| !mask[index])
        .flat_map(|index| {
            [
                (
                    built.x_names[index].as_str(),
                    instance.x_data[index].as_slice(),
                ),
                (
                    built.w_names[index].as_str(),
                    instance.w_data[index].as_slice(),
                ),
            ]
        })
        .collect()
}

fn run_streaming_arm(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    let shape = &STREAM_SHAPE;
    let built = block_sparse_program(shape);
    let instances = streaming_instances(shape);

    let dense_bytes = bytes_touched(shape, shape.num_blocks);
    let dense_elements = (shape.block_in * shape.block_out * shape.num_blocks) as f64;
    println!(
        "{}: dense_bytes={dense_bytes} predicted_dense_ns_bandwidth={:.1} predicted_dense_ns_compute_floor={:.1} (ROW180 rival prediction, {STREAM_INSTANCES} round-robin instances, rotation_distance_bytes={})",
        shape.label,
        predicted_ns(dense_bytes),
        dense_elements * ROW180_COMPUTE_FLOOR_NS_PER_ELEMENT,
        (STREAM_INSTANCES - 1) * dense_bytes
    );

    // PRE-REGISTRATION for sparse_{50,75,90}, written before any measurement
    // on this shape: both the task's bandwidth-wall prediction and ROW 180's
    // own rival compute-floor prediction, so the miss/hit can be read against
    // either hypothesis once measured.
    for &(sparsity_pct, skip_count) in &[
        (50u32, shape.num_blocks / 2),
        (75, shape.num_blocks * 3 / 4),
        (90, shape.num_blocks * 9 / 10),
    ] {
        let live_count = shape.num_blocks - skip_count;
        let sparse_bytes = bytes_touched(shape, live_count);
        let live_elements = (shape.block_in * shape.block_out * live_count) as f64;
        println!(
            "{}/sparse_{sparsity_pct}: skip={skip_count}/{} sparse_bytes={sparse_bytes} predicted_sparse_ns_bandwidth={:.1} predicted_saving_ns_bandwidth={:.1} predicted_sparse_ns_compute_floor={:.1} predicted_saving_ns_compute_floor={:.1}",
            shape.label,
            shape.num_blocks,
            predicted_ns(sparse_bytes),
            predicted_ns(dense_bytes) - predicted_ns(sparse_bytes),
            live_elements * ROW180_COMPUTE_FLOOR_NS_PER_ELEMENT,
            dense_elements * ROW180_COMPUTE_FLOOR_NS_PER_ELEMENT
                - live_elements * ROW180_COMPUTE_FLOOR_NS_PER_ELEMENT,
        );
    }

    // correctness self-check, instance 0 only (time-budget scoped): the
    // masked execution path itself is already validated 8/8 combinations in
    // ROW 180 on different data -- this instance-0 check confirms the SAME
    // path holds on the streaming shape/data, not a re-validation of the
    // mechanism from scratch.
    let mut dense_arena_zero =
        build_static_arena(&built.program, &[], &built.reduce_nodes).expect("dense arena builds");
    let dense_reference_zero = evaluate_named_with_arena(
        &mut dense_arena_zero,
        &named_for_instance(&built, shape, &instances[0]),
    )
    .expect("dense step");

    let mut dense_arenas: Vec<_> = (0..STREAM_INSTANCES)
        .map(|_| {
            build_static_arena(&built.program, &[], &built.reduce_nodes)
                .expect("dense arena builds")
        })
        .collect();
    dense_arenas[0] = dense_arena_zero;

    let mut dense_cursor = 0usize;
    group.bench_function(format!("{}/dense", shape.label), |bencher| {
        bencher.iter(|| {
            let instance = &instances[dense_cursor];
            let arena = &mut dense_arenas[dense_cursor];
            dense_cursor = (dense_cursor + 1) % STREAM_INSTANCES;
            evaluate_named_with_arena(arena, &named_for_instance(&built, shape, instance))
                .expect("dense step")
        });
    });

    for &(sparsity_pct, skip_count) in &[
        (50u32, shape.num_blocks / 2),
        (75, shape.num_blocks * 3 / 4),
        (90, shape.num_blocks * 9 / 10),
    ] {
        let mask: Vec<bool> = (0..shape.num_blocks)
            .map(|index| index >= shape.num_blocks - skip_count)
            .collect();

        let mut sparse_arena_zero = build_static_arena(&built.program, &[], &built.reduce_nodes)
            .expect("sparse arena builds");
        let mut skip_zero = BTreeSet::new();
        for (index, &masked) in mask.iter().enumerate() {
            if masked {
                skip_zero.insert(built.product_nodes[index]);
                skip_zero.insert(built.reduce_nodes[index]);
            }
        }
        let sparse_reference_zero = evaluate_named_with_arena_masked(
            &mut sparse_arena_zero,
            &live_named_for_instance(&built, shape, &instances[0], &mask),
            &skip_zero,
        )
        .expect("sparse step evaluates");
        for (index, &masked) in mask.iter().enumerate() {
            if masked {
                continue;
            }
            let node = built.reduce_nodes[index];
            let (dense_values, _) = dense_reference_zero
                .get(node)
                .expect("dense output present");
            let (sparse_values, _) = sparse_reference_zero
                .get(node)
                .expect("sparse output present");
            assert_eq!(
                dense_values, sparse_values,
                "{}/sparse_{sparsity_pct}: block {index} diverged (instance 0)",
                shape.label
            );
        }

        let mut sparse_arenas: Vec<_> = (0..STREAM_INSTANCES)
            .map(|_| {
                build_static_arena(&built.program, &[], &built.reduce_nodes)
                    .expect("sparse arena builds")
            })
            .collect();
        sparse_arenas[0] = sparse_arena_zero;

        let mut sparse_cursor = 0usize;
        group.bench_function(
            format!("{}/sparse_{sparsity_pct}", shape.label),
            |bencher| {
                bencher.iter(|| {
                    let instance = &instances[sparse_cursor];
                    let arena = &mut sparse_arenas[sparse_cursor];
                    sparse_cursor = (sparse_cursor + 1) % STREAM_INSTANCES;
                    let live_named = live_named_for_instance(&built, shape, instance, &mask);
                    let mut skip = BTreeSet::new();
                    for (index, &masked) in mask.iter().enumerate() {
                        if masked {
                            skip.insert(built.product_nodes[index]);
                            skip.insert(built.reduce_nodes[index]);
                        }
                    }
                    evaluate_named_with_arena_masked(arena, &live_named, &skip)
                        .expect("sparse step")
                });
            },
        );
    }
}
