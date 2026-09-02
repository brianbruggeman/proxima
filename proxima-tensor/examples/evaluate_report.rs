#![allow(clippy::expect_used)]
//! The task's single coherent report: one run of a 1024^3 GEMM at a given
//! thread count, printing allocations (count/bytes/top sites), which code
//! path each node actually took, the parallel dispatch breakdown, and every
//! stage timing this crate's `instrument` feature exposes — with the
//! residual between their sum and wall-clock named explicitly rather than
//! absorbed. `matmul_program_rhs_transposed`/`Lcg`/`random_vec` copied
//! verbatim from `examples/parallel_breakdown.rs`. Not part of the crate's
//! public surface; throwaway for this measurement task.

use std::alloc::{GlobalAlloc, Layout, System};
use std::env;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use proxima_tensor::instrument;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    Extent, IndexMap, NodeId, Op, ReduceInit, ScalarOp, append, evaluate_parallel, map,
};

/// Wraps the system allocator to count every allocation the process makes
/// (unconditionally — a call-site counter inside the crate would only ever
/// report sites the crate author remembered to instrument) and, under the
/// `instrument` feature, attributes each one's byte size to whichever
/// `AllocSiteGuard` region was active when it fired. One atomic increment
/// per real heap allocation, never per element.
struct CountingAllocator;

static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        instrument::record_alloc(instrument::current_alloc_site(), layout.size() as u64);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn random_vec(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..n).map(|_| lcg.next_unit() * scale).collect()
}

fn matmul_program_rhs_transposed(m: u32, k: u32, n: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs = append(
        &mut program,
        Op::Input {
            dtype: proxima_tensor::DType::Float32,
            shape: vec![Extent::Static(m), Extent::Static(k)],
            name: None,
        },
    );
    let rhs = append(
        &mut program,
        Op::Input {
            dtype: proxima_tensor::DType::Float32,
            shape: vec![Extent::Static(n), Extent::Static(k)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: proxima_tensor::DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (rhs, IndexMap::Affine(map::projection(3, &[1, 2]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(proxima_tensor::Reduce {
            dtype: proxima_tensor::DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: proxima_tensor::Keep::Reduce,
            name: Some("matmul_rhs_transposed".into()),
        }),
    );
    (program, sum)
}

fn report_for_threads(program: &[Op], lhs: &[f32], rhs_t: &[f32], threads: usize) {
    let workers = NonZeroUsize::new(threads).expect("threads must be nonzero");

    instrument::reset();
    instrument::reset_parallel();
    instrument::reset_serial();
    instrument::reset_path();
    instrument::reset_alloc_sites();
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);

    // The output buffer's own allocation is timed as one call
    // (`SERIAL_ALLOC_NANOS`, cpu.rs's `vec![0.0f32; node_output_len(..)]`):
    // std's `vec![0.0; n]` for a zero-pattern `f32` compiles to a single
    // `alloc_zeroed` call (`SpecFromElem`'s `IsZero` path in
    // `alloc::vec::spec_from_elem`), not an allocate-then-memset pair, so
    // there is no separate "zero-fill" step inside this call to time apart
    // from the allocation. `examples/alloc_zero_fill_experiment.rs` isolates
    // the best available split (with_capacity-only vs the production
    // vec![0.0; n] expression vs an explicit manual fill loop) as a
    // standalone microbenchmark instead — quoted below, not re-decomposed
    // here, since decomposing it in situ would mean running different code
    // than production runs.
    let wall_start = Instant::now();
    let evaluated =
        evaluate_parallel(program, &[], &[lhs, rhs_t], &[], workers).expect("gemm evaluates");
    let wall_ns = wall_start.elapsed().as_nanos() as u64;
    let checksum = evaluated.root()[0];

    let alloc_count = ALLOC_COUNT.load(Ordering::Relaxed);
    let alloc_bytes = ALLOC_BYTES.load(Ordering::Relaxed);
    let alloc_sites = instrument::alloc_totals();
    let parallel = instrument::parallel_totals();
    let serial = instrument::serial_totals();
    let path = instrument::path_totals();

    // every `_ticks` field below is a raw `proxima_clock::Ticks` delta
    // (`instrument::read_ticks`'s doc); converted to nanoseconds once here,
    // not per element, matching `bind.rs`'s DIAG block.
    let chunk_min_nanos = instrument::ticks_to_nanos(parallel.chunk_ticks_min);
    let chunk_max_nanos = instrument::ticks_to_nanos(parallel.chunk_ticks_max);
    let chunk_sum_nanos = instrument::ticks_to_nanos(parallel.chunk_ticks_sum);
    let spawn_nanos = instrument::ticks_to_nanos(parallel.spawn_ticks);
    let join_nanos = instrument::ticks_to_nanos(parallel.join_ticks);
    let node_nanos = instrument::ticks_to_nanos(parallel.node_ticks);
    let prepare_nanos = instrument::ticks_to_nanos(serial.prepare_ticks);
    let alloc_nanos = instrument::ticks_to_nanos(serial.alloc_ticks);
    let split_nanos = instrument::ticks_to_nanos(serial.split_ticks);
    let slice_carve_nanos = instrument::ticks_to_nanos(serial.slice_carve_ticks);
    let finish_nanos = instrument::ticks_to_nanos(serial.finish_ticks);
    let bookkeeping_nanos = instrument::ticks_to_nanos(serial.bookkeeping_ticks);
    let sequential_compute_nanos = instrument::ticks_to_nanos(serial.sequential_compute_ticks);

    let imbalance = if chunk_min_nanos == 0 {
        0.0
    } else {
        chunk_max_nanos as f64 / chunk_min_nanos as f64
    };
    let spawn_percent = if node_nanos == 0 {
        0.0
    } else {
        spawn_nanos as f64 / node_nanos as f64 * 100.0
    };

    let stage_sum_ns = prepare_nanos
        + alloc_nanos
        + split_nanos
        + slice_carve_nanos
        + finish_nanos
        + bookkeeping_nanos
        + sequential_compute_nanos
        + node_nanos;
    let residual_ns = wall_ns as i64 - stage_sum_ns as i64;

    println!("==== threads={threads} ====");
    println!("checksum={checksum:.5} wall_ns={wall_ns}");
    println!(
        "allocations: count={alloc_count} bytes={alloc_bytes} \
         top_sites=[output_buffer count={} bytes={}, prepare count={} bytes={}, \
         chunk_slices count={} bytes={}, other count={} bytes={}]",
        alloc_sites.output_buffer_count,
        alloc_sites.output_buffer_bytes,
        alloc_sites.prepare_count,
        alloc_sites.prepare_bytes,
        alloc_sites.chunk_slices_count,
        alloc_sites.chunk_slices_bytes,
        alloc_sites.other_count,
        alloc_sites.other_bytes,
    );
    println!(
        "paths: op_kind[elementwise={} reduce={} scan={}] \
         dispatch[parallel={} sequential_below_threshold={} sequential_split_unavailable={}]",
        path.op_kind_elementwise,
        path.op_kind_reduce,
        path.op_kind_scan,
        path.dispatch_parallel,
        path.dispatch_sequential_below_threshold,
        path.dispatch_sequential_split_unavailable,
    );
    #[cfg(target_arch = "aarch64")]
    {
        let (neon_gate_passes, neon_invocations, neon_fallback) =
            proxima_tensor::cpu::neon_tile_counters();
        let (width_gate_passes, width_invocations, width_fallback) =
            proxima_tensor::cpu::width_tile_counters();
        println!(
            "neon_dot_tile: gate_passes={neon_gate_passes} main_invocations={neon_invocations} \
             row_remainder_invocations={} row_remainder_elements={} fallback_elements={neon_fallback} \
             column_tail_present={}",
            proxima_tensor::cpu::neon_tile_row_remainder_invocations(),
            proxima_tensor::cpu::neon_tile_row_remainder_elements(),
            instrument::NEON_TILE_COLUMN_TAIL_PRESENT.get(),
        );
        println!(
            "elementwise_width_tile: gate_passes={width_gate_passes} invocations={width_invocations} \
             fallback_elements={width_fallback} column_tail_present={}",
            instrument::WIDTH_TILE_COLUMN_TAIL_PRESENT.get(),
        );
    }
    println!(
        "parallel: nodes={} chunk_count={} chunk_min_ns={} chunk_max_ns={} chunk_mean_ns={:.1} \
         imbalance={imbalance:.3}x spawn_ns={} spawn_percent={spawn_percent:.2}% join_ns={} node_ns={}",
        parallel.parallel_nodes,
        parallel.chunk_count,
        chunk_min_nanos,
        chunk_max_nanos,
        if parallel.chunk_count == 0 {
            0.0
        } else {
            chunk_sum_nanos as f64 / parallel.chunk_count as f64
        },
        spawn_nanos,
        join_nanos,
        node_nanos,
    );
    println!(
        "stages_ns: prepare={} alloc={} split={} slice_carve={} finish={} bookkeeping={} \
         sequential_compute={} parallel_node={} sum={stage_sum_ns} wall={wall_ns} residual={residual_ns}",
        prepare_nanos,
        alloc_nanos,
        split_nanos,
        slice_carve_nanos,
        finish_nanos,
        bookkeeping_nanos,
        sequential_compute_nanos,
        node_nanos,
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let thread_counts: Vec<usize> = if args.len() > 1 {
        args[1..]
            .iter()
            .map(|value| {
                value
                    .parse()
                    .expect("thread count must be a positive integer")
            })
            .collect()
    } else {
        vec![1, 2, 4, 8]
    };

    let (m, k, n) = (1024u32, 1024u32, 1024u32);
    let (program, _sum) = matmul_program_rhs_transposed(m, k, n);
    let lhs = random_vec(1, (m * k) as usize, 1.0);
    let rhs_t = random_vec(2, (n * k) as usize, 1.0);

    // one untimed warm-up so allocator free-lists/page tables are already
    // warm before the first measured run.
    let workers = NonZeroUsize::new(1).expect("nonzero");
    let _ = evaluate_parallel(&program, &[], &[&lhs, &rhs_t], &[], workers)
        .expect("warmup gemm evaluates");

    for threads in thread_counts {
        report_for_threads(&program, &lhs, &rhs_t, threads);
    }
}
