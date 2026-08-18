//! Real execution-witness counters for [`crate::cpu`]'s bound-op kernels,
//! gated entirely behind the `instrument` feature.
//!
//! Every field here is incremented from a plain local accumulator inside
//! the kernel loop and committed to the process-wide [`proxima_telemetry`]
//! counters exactly once, at the end of the bound-op call — never as an
//! atomic increment inside a loop that can run ~1e9 times, or the
//! instrument would perturb the thing it measures.

use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};
use std::thread::ThreadId;

use proxima_telemetry::counter;
use proxima_telemetry::metric::Counter;

pub static BOUND_OPS: Counter = Counter::new("proxima_tensor.bound_ops");
pub static MAC_OPS: Counter = Counter::new("proxima_tensor.mac_ops");
pub static OPERAND_LOADS: Counter = Counter::new("proxima_tensor.operand_loads");
pub static DISTINCT_OPERAND_ELEMENTS: Counter = Counter::new("proxima_tensor.distinct_operand_elements");
pub static OUTPUT_WRITES: Counter = Counter::new("proxima_tensor.output_writes");
pub static PATH_DOT_FAST: Counter = Counter::new("proxima_tensor.path.dot_fast");
pub static PATH_WIDTH_FAST: Counter = Counter::new("proxima_tensor.path.width_fast");
pub static PATH_GENERIC: Counter = Counter::new("proxima_tensor.path.generic");
pub static LEADING_ITERS: Counter = Counter::new("proxima_tensor.leading_iters");
pub static KERNEL_CALLS: Counter = Counter::new("proxima_tensor.kernel_calls");

// per-parallel-node wall-clock breakdown for `cpu::run_chunks_threaded` /
// `cpu::evaluate_node_parallel`: where does thread::scope time actually go.
pub static PARALLEL_NODES: Counter = Counter::new("proxima_tensor.parallel_nodes");
pub static PARALLEL_NODE_NANOS: Counter = Counter::new("proxima_tensor.parallel_node_nanos");
pub static PARALLEL_SPAWN_NANOS: Counter = Counter::new("proxima_tensor.parallel_spawn_nanos");
pub static PARALLEL_JOIN_NANOS: Counter = Counter::new("proxima_tensor.parallel_join_nanos");
pub static PARALLEL_CHUNK_COUNT: Counter = Counter::new("proxima_tensor.parallel_chunk_count");
pub static PARALLEL_CHUNK_NANOS_SUM: Counter =
    Counter::new("proxima_tensor.parallel_chunk_nanos_sum");
// Counter has no min/max form, so the extremes live in their own atomics,
// updated with fetch_min/fetch_max — the same lock-free discipline the
// counters use, just without a running sum.
pub static PARALLEL_CHUNK_NANOS_MIN: AtomicU64 = AtomicU64::new(u64::MAX);
pub static PARALLEL_CHUNK_NANOS_MAX: AtomicU64 = AtomicU64::new(0);

/// Records one chunk's compute duration into the sum/count/min/max quartet.
/// Called once per chunk after `run_node_into` returns — never inside the
/// kernel loop itself.
pub fn record_chunk_nanos(nanos: u64) {
    counter!(PARALLEL_CHUNK_NANOS_SUM, nanos);
    counter!(PARALLEL_CHUNK_COUNT, 1);
    PARALLEL_CHUNK_NANOS_MIN.fetch_min(nanos, Ordering::Relaxed);
    PARALLEL_CHUNK_NANOS_MAX.fetch_max(nanos, Ordering::Relaxed);
}

/// One process run's worth of parallel-dispatch timing, read back by the
/// `sweep_gemm` harness after evaluation — a snapshot, not a reset, so the
/// harness can also print an end-of-run summary without disturbing counters
/// a caller still wants to read again.
#[derive(Debug, Clone, Copy, Default)]
pub struct ParallelTotals {
    pub parallel_nodes: u64,
    pub node_nanos: u64,
    pub spawn_nanos: u64,
    pub join_nanos: u64,
    pub chunk_count: u64,
    pub chunk_nanos_sum: u64,
    pub chunk_nanos_min: u64,
    pub chunk_nanos_max: u64,
}

#[must_use]
pub fn parallel_totals() -> ParallelTotals {
    let chunk_count = PARALLEL_CHUNK_COUNT.get();
    let observed_min = PARALLEL_CHUNK_NANOS_MIN.load(Ordering::Relaxed);
    ParallelTotals {
        parallel_nodes: PARALLEL_NODES.get(),
        node_nanos: PARALLEL_NODE_NANOS.get(),
        spawn_nanos: PARALLEL_SPAWN_NANOS.get(),
        join_nanos: PARALLEL_JOIN_NANOS.get(),
        chunk_count,
        chunk_nanos_sum: PARALLEL_CHUNK_NANOS_SUM.get(),
        // no chunk was ever recorded: report 0, not the u64::MAX sentinel.
        chunk_nanos_min: if chunk_count == 0 { 0 } else { observed_min },
        chunk_nanos_max: PARALLEL_CHUNK_NANOS_MAX.load(Ordering::Relaxed),
    }
}

/// Resets the parallel-dispatch counters to their initial state — mirrors
/// [`reset`] but kept separate so a caller can reset one family without
/// disturbing the kernel counters.
pub fn reset_parallel() {
    let _ = PARALLEL_NODES.snapshot_and_reset();
    let _ = PARALLEL_NODE_NANOS.snapshot_and_reset();
    let _ = PARALLEL_SPAWN_NANOS.snapshot_and_reset();
    let _ = PARALLEL_JOIN_NANOS.snapshot_and_reset();
    let _ = PARALLEL_CHUNK_COUNT.snapshot_and_reset();
    let _ = PARALLEL_CHUNK_NANOS_SUM.snapshot_and_reset();
    PARALLEL_CHUNK_NANOS_MIN.store(u64::MAX, Ordering::Relaxed);
    PARALLEL_CHUNK_NANOS_MAX.store(0, Ordering::Relaxed);
}

// chunk duration (above) scatters by construction as chunk count grows past
// worker count under oversubscription, so it cannot tell a balanced pool
// apart from an unbalanced one. what actually decides whether the parallel
// region is bottlenecked on one straggler is each PULLER's total busy time —
// summed across every chunk that puller claimed — which is why this is
// keyed by the calling thread, not by chunk index.
static WORKER_BUSY_NANOS: Mutex<Vec<(ThreadId, u64)>> = Mutex::new(Vec::new());

/// Adds `nanos` to the current thread's running total. Called from the same
/// per-chunk timing site as [`record_chunk_nanos`] — this is a second,
/// orthogonal aggregation of the identical measurement, grouped by puller
/// instead of by chunk.
pub fn record_worker_busy_nanos(nanos: u64) {
    let thread_id = std::thread::current().id();
    let mut totals = WORKER_BUSY_NANOS.lock().unwrap_or_else(PoisonError::into_inner);
    match totals.iter_mut().find(|(existing, _)| *existing == thread_id) {
        Some((_, total)) => *total += nanos,
        None => totals.push((thread_id, nanos)),
    }
}

/// Every worker's accumulated busy time from the most recent parallel
/// region(s) since the last [`reset_worker_busy`] — one entry per distinct
/// thread that claimed at least one chunk. Order is not meaningful.
#[must_use]
pub fn worker_busy_snapshot() -> Vec<u64> {
    let totals = WORKER_BUSY_NANOS.lock().unwrap_or_else(PoisonError::into_inner);
    totals.iter().map(|(_, nanos)| *nanos).collect()
}

pub fn reset_worker_busy() {
    let mut totals = WORKER_BUSY_NANOS.lock().unwrap_or_else(PoisonError::into_inner);
    totals.clear();
}

// the busy total above is `Instant`-derived, so a worker the OS descheduled
// keeps accruing "busy" nanos while off-core. on a box carrying any ambient
// load that turns the 1->8 scaling read into a measurement of the host: a
// register-only fma control (zero memory traffic, so no scaling effect is
// even possible) measured +41.2% wall growth 1->8 against +6.8% cpu growth,
// n=9, 2026-08-18. every 1->8 figure taken before this existed used the wall
// form and is not separable from that. the cpu clock below is the same
// measurement against a clock that stops when the thread does; carry BOTH,
// because their ratio is the only in-band report of how much the host
// interfered with the run.
static WORKER_CPU_NANOS: Mutex<Vec<(ThreadId, u64)>> = Mutex::new(Vec::new());

#[repr(C)]
struct Timespec {
    seconds: i64,
    nanos: i64,
}

unsafe extern "C" {
    fn clock_gettime(clock_id: i32, out: *mut Timespec) -> i32;
}

#[cfg(target_os = "macos")]
const CLOCK_THREAD_CPUTIME_ID: i32 = 16;
#[cfg(not(target_os = "macos"))]
const CLOCK_THREAD_CPUTIME_ID: i32 = 3;

/// This thread's consumed CPU time. Unlike an [`Instant`](std::time::Instant)
/// delta, this does not advance while the thread is off-core.
#[must_use]
pub fn thread_cpu_nanos() -> u64 {
    let mut now = Timespec { seconds: 0, nanos: 0 };
    if unsafe { clock_gettime(CLOCK_THREAD_CPUTIME_ID, &mut now) } != 0 {
        return 0;
    }
    (now.seconds as u64) * 1_000_000_000 + (now.nanos as u64)
}

/// Adds `nanos` of consumed CPU time to the current thread's running total,
/// the deschedule-immune peer of [`record_worker_busy_nanos`].
pub fn record_worker_cpu_nanos(nanos: u64) {
    let thread_id = std::thread::current().id();
    let mut totals = WORKER_CPU_NANOS.lock().unwrap_or_else(PoisonError::into_inner);
    match totals.iter_mut().find(|(existing, _)| *existing == thread_id) {
        Some((_, total)) => *total += nanos,
        None => totals.push((thread_id, nanos)),
    }
}

#[must_use]
pub fn worker_cpu_snapshot() -> Vec<u64> {
    let totals = WORKER_CPU_NANOS.lock().unwrap_or_else(PoisonError::into_inner);
    totals.iter().map(|(_, nanos)| *nanos).collect()
}

pub fn reset_worker_cpu() {
    let mut totals = WORKER_CPU_NANOS.lock().unwrap_or_else(PoisonError::into_inner);
    totals.clear();
}

// `evaluate_parallel`'s own wall-clock, decomposed into every named part
// that is NOT inside `run_chunks_threaded`'s `thread::scope` (which
// `PARALLEL_NODE_NANOS` above already measures, now scoped to start right
// before `thread::scope`, after slice-carving — see `cpu::run_chunks_threaded`).
// Each part is timed once per `evaluate_parallel` call (or once per resolved
// node, for the per-node parts) and committed after the timed region, never
// as a per-element accumulation.
pub static SERIAL_PREPARE_NANOS: Counter = Counter::new("proxima_tensor.serial_prepare_nanos");
pub static SERIAL_ALLOC_NANOS: Counter = Counter::new("proxima_tensor.serial_alloc_nanos");
pub static SERIAL_SPLIT_NANOS: Counter = Counter::new("proxima_tensor.serial_split_nanos");
pub static SERIAL_SLICE_CARVE_NANOS: Counter = Counter::new("proxima_tensor.serial_slice_carve_nanos");
pub static SERIAL_FINISH_NANOS: Counter = Counter::new("proxima_tensor.serial_finish_nanos");
pub static SERIAL_BOOKKEEPING_NANOS: Counter = Counter::new("proxima_tensor.serial_bookkeeping_nanos");
// only nonzero on the `workers == 1` (or below-threshold) arm, where
// `evaluate_node_parallel` never reaches `run_chunks_threaded` at all.
pub static SERIAL_SEQUENTIAL_COMPUTE_NANOS: Counter =
    Counter::new("proxima_tensor.serial_sequential_compute_nanos");
pub static SERIAL_EVALUATE_PARALLEL_NANOS: Counter =
    Counter::new("proxima_tensor.serial_evaluate_parallel_nanos");
pub static SERIAL_EVALUATE_PARALLEL_CALLS: Counter =
    Counter::new("proxima_tensor.serial_evaluate_parallel_calls");

/// One process run's worth of `evaluate_parallel`'s serial-remainder
/// breakdown, read back the same way [`parallel_totals`] is.
#[derive(Debug, Clone, Copy, Default)]
pub struct SerialTotals {
    pub prepare_nanos: u64,
    pub alloc_nanos: u64,
    pub split_nanos: u64,
    pub slice_carve_nanos: u64,
    pub finish_nanos: u64,
    pub bookkeeping_nanos: u64,
    pub sequential_compute_nanos: u64,
    pub evaluate_parallel_nanos: u64,
    pub evaluate_parallel_calls: u64,
}

#[must_use]
pub fn serial_totals() -> SerialTotals {
    SerialTotals {
        prepare_nanos: SERIAL_PREPARE_NANOS.get(),
        alloc_nanos: SERIAL_ALLOC_NANOS.get(),
        split_nanos: SERIAL_SPLIT_NANOS.get(),
        slice_carve_nanos: SERIAL_SLICE_CARVE_NANOS.get(),
        finish_nanos: SERIAL_FINISH_NANOS.get(),
        bookkeeping_nanos: SERIAL_BOOKKEEPING_NANOS.get(),
        sequential_compute_nanos: SERIAL_SEQUENTIAL_COMPUTE_NANOS.get(),
        evaluate_parallel_nanos: SERIAL_EVALUATE_PARALLEL_NANOS.get(),
        evaluate_parallel_calls: SERIAL_EVALUATE_PARALLEL_CALLS.get(),
    }
}

/// Resets the serial-breakdown counters to their initial state — mirrors
/// [`reset_parallel`] but kept separate so a caller can reset one family
/// without disturbing the others.
pub fn reset_serial() {
    let _ = SERIAL_PREPARE_NANOS.snapshot_and_reset();
    let _ = SERIAL_ALLOC_NANOS.snapshot_and_reset();
    let _ = SERIAL_SPLIT_NANOS.snapshot_and_reset();
    let _ = SERIAL_SLICE_CARVE_NANOS.snapshot_and_reset();
    let _ = SERIAL_FINISH_NANOS.snapshot_and_reset();
    let _ = SERIAL_BOOKKEEPING_NANOS.snapshot_and_reset();
    let _ = SERIAL_SEQUENTIAL_COMPUTE_NANOS.snapshot_and_reset();
    let _ = SERIAL_EVALUATE_PARALLEL_NANOS.snapshot_and_reset();
    let _ = SERIAL_EVALUATE_PARALLEL_CALLS.snapshot_and_reset();
}

// which `run_node_into` arm ran — once per node in the sequential path,
// once per chunk in the parallel path (`cpu::run_chunks_threaded` calls
// `run_node_into` once per spawned chunk, never per element).
pub static OP_KIND_ELEMENTWISE: Counter = Counter::new("proxima_tensor.op_kind.elementwise");
pub static OP_KIND_REDUCE: Counter = Counter::new("proxima_tensor.op_kind.reduce");
pub static OP_KIND_SCAN: Counter = Counter::new("proxima_tensor.op_kind.scan");

/// Which `run_node_into` arm this bound op's `BoundOpKind`/`Keep` resolved
/// to — set once per call from the match already driving dispatch, never
/// re-derived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpKind {
    Elementwise,
    Reduce,
    Scan,
}

pub fn record_op_kind(kind: OpKind) {
    match kind {
        OpKind::Elementwise => counter!(OP_KIND_ELEMENTWISE, 1),
        OpKind::Reduce => counter!(OP_KIND_REDUCE, 1),
        OpKind::Scan => counter!(OP_KIND_SCAN, 1),
    }
}

// why `evaluate_node_parallel` took the sequential arm for a given node —
// below `PARALLEL_THRESHOLD` (`cpu.rs`'s element-count gate) versus
// `BoundOp::split` itself returning `None` (`bind.rs::split`, which always
// returns `None` for `parts < 2` — a `workers == 1` run is always
// "split unavailable", never "below threshold", even on a huge node).
// `PARALLEL_NODES` (already above) is the third arm: dispatched parallel.
// All three are set once per node, from the same `chunks` match already
// driving dispatch in `evaluate_node_parallel`.
pub static DISPATCH_SEQUENTIAL_BELOW_THRESHOLD: Counter =
    Counter::new("proxima_tensor.dispatch.sequential_below_threshold");
pub static DISPATCH_SEQUENTIAL_SPLIT_UNAVAILABLE: Counter =
    Counter::new("proxima_tensor.dispatch.sequential_split_unavailable");

/// Snapshot of the op-kind and dispatch-reason counters, read back the same
/// way [`parallel_totals`]/[`serial_totals`] are.
#[derive(Debug, Clone, Copy, Default)]
pub struct PathTotals {
    pub op_kind_elementwise: u64,
    pub op_kind_reduce: u64,
    pub op_kind_scan: u64,
    pub dispatch_parallel: u64,
    pub dispatch_sequential_below_threshold: u64,
    pub dispatch_sequential_split_unavailable: u64,
}

#[must_use]
pub fn path_totals() -> PathTotals {
    PathTotals {
        op_kind_elementwise: OP_KIND_ELEMENTWISE.get(),
        op_kind_reduce: OP_KIND_REDUCE.get(),
        op_kind_scan: OP_KIND_SCAN.get(),
        dispatch_parallel: PARALLEL_NODES.get(),
        dispatch_sequential_below_threshold: DISPATCH_SEQUENTIAL_BELOW_THRESHOLD.get(),
        dispatch_sequential_split_unavailable: DISPATCH_SEQUENTIAL_SPLIT_UNAVAILABLE.get(),
    }
}

/// Resets the op-kind and dispatch-reason counters — mirrors
/// [`reset_parallel`]/[`reset_serial`].
pub fn reset_path() {
    let _ = OP_KIND_ELEMENTWISE.snapshot_and_reset();
    let _ = OP_KIND_REDUCE.snapshot_and_reset();
    let _ = OP_KIND_SCAN.snapshot_and_reset();
    let _ = DISPATCH_SEQUENTIAL_BELOW_THRESHOLD.snapshot_and_reset();
    let _ = DISPATCH_SEQUENTIAL_SPLIT_UNAVAILABLE.snapshot_and_reset();
}

// whether a tile-path node had leftover columns past the last full tile
// (`tiled_width_cols < width`) — computed once from the node's own shape,
// not re-evaluated per loop iteration, and committed once per node
// alongside the rest of that node's tile counters.
pub static NEON_TILE_COLUMN_TAIL_PRESENT: Counter =
    Counter::new("proxima_tensor.neon_tile.column_tail_present");
pub static WIDTH_TILE_COLUMN_TAIL_PRESENT: Counter =
    Counter::new("proxima_tensor.width_tile.column_tail_present");

// per-allocation call-site attribution. A caller's wrapped `GlobalAlloc`
// (only a binary may install one, never this library) reads
// `current_alloc_site()` inside its own `alloc`/`dealloc` and calls
// `record_alloc` once per real heap allocation — never per element, since
// the number of allocations on the evaluate path is bounded by the number
// of nodes, not by tensor size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AllocSite {
    #[default]
    Other,
    Prepare,
    OutputBuffer,
    ChunkSlices,
}

std::thread_local! {
    static CURRENT_ALLOC_SITE: core::cell::Cell<AllocSite> = const { core::cell::Cell::new(AllocSite::Other) };
}

#[must_use]
pub fn current_alloc_site() -> AllocSite {
    CURRENT_ALLOC_SITE.with(core::cell::Cell::get)
}

/// RAII guard: labels every allocation made while it is alive as `site`,
/// restoring whatever label was active before on drop — nests correctly if
/// a labeled region ever calls into another one.
pub struct AllocSiteGuard {
    previous: AllocSite,
}

impl AllocSiteGuard {
    #[must_use]
    pub fn enter(site: AllocSite) -> Self {
        let previous = CURRENT_ALLOC_SITE.with(|cell| cell.replace(site));
        Self { previous }
    }
}

impl Drop for AllocSiteGuard {
    fn drop(&mut self) {
        CURRENT_ALLOC_SITE.with(|cell| cell.set(self.previous));
    }
}

pub static ALLOC_SITE_OTHER_COUNT: AtomicU64 = AtomicU64::new(0);
pub static ALLOC_SITE_OTHER_BYTES: AtomicU64 = AtomicU64::new(0);
pub static ALLOC_SITE_PREPARE_COUNT: AtomicU64 = AtomicU64::new(0);
pub static ALLOC_SITE_PREPARE_BYTES: AtomicU64 = AtomicU64::new(0);
pub static ALLOC_SITE_OUTPUT_BUFFER_COUNT: AtomicU64 = AtomicU64::new(0);
pub static ALLOC_SITE_OUTPUT_BUFFER_BYTES: AtomicU64 = AtomicU64::new(0);
pub static ALLOC_SITE_CHUNK_SLICES_COUNT: AtomicU64 = AtomicU64::new(0);
pub static ALLOC_SITE_CHUNK_SLICES_BYTES: AtomicU64 = AtomicU64::new(0);

/// Called once per real heap allocation by the process's wrapped
/// `GlobalAlloc` — never per element.
pub fn record_alloc(site: AllocSite, bytes: u64) {
    let (count, byte_total) = match site {
        AllocSite::Other => (&ALLOC_SITE_OTHER_COUNT, &ALLOC_SITE_OTHER_BYTES),
        AllocSite::Prepare => (&ALLOC_SITE_PREPARE_COUNT, &ALLOC_SITE_PREPARE_BYTES),
        AllocSite::OutputBuffer => (&ALLOC_SITE_OUTPUT_BUFFER_COUNT, &ALLOC_SITE_OUTPUT_BUFFER_BYTES),
        AllocSite::ChunkSlices => (&ALLOC_SITE_CHUNK_SLICES_COUNT, &ALLOC_SITE_CHUNK_SLICES_BYTES),
    };
    count.fetch_add(1, Ordering::Relaxed);
    byte_total.fetch_add(bytes, Ordering::Relaxed);
}

/// Snapshot of every allocation site's (count, bytes) pair, for a caller
/// that wants to print "top sites by bytes" without pulling in a full
/// telemetry exporter.
#[derive(Debug, Clone, Copy, Default)]
pub struct AllocTotals {
    pub other_count: u64,
    pub other_bytes: u64,
    pub prepare_count: u64,
    pub prepare_bytes: u64,
    pub output_buffer_count: u64,
    pub output_buffer_bytes: u64,
    pub chunk_slices_count: u64,
    pub chunk_slices_bytes: u64,
}

#[must_use]
pub fn alloc_totals() -> AllocTotals {
    AllocTotals {
        other_count: ALLOC_SITE_OTHER_COUNT.load(Ordering::Relaxed),
        other_bytes: ALLOC_SITE_OTHER_BYTES.load(Ordering::Relaxed),
        prepare_count: ALLOC_SITE_PREPARE_COUNT.load(Ordering::Relaxed),
        prepare_bytes: ALLOC_SITE_PREPARE_BYTES.load(Ordering::Relaxed),
        output_buffer_count: ALLOC_SITE_OUTPUT_BUFFER_COUNT.load(Ordering::Relaxed),
        output_buffer_bytes: ALLOC_SITE_OUTPUT_BUFFER_BYTES.load(Ordering::Relaxed),
        chunk_slices_count: ALLOC_SITE_CHUNK_SLICES_COUNT.load(Ordering::Relaxed),
        chunk_slices_bytes: ALLOC_SITE_CHUNK_SLICES_BYTES.load(Ordering::Relaxed),
    }
}

/// Resets every allocation-site counter to zero.
pub fn reset_alloc_sites() {
    ALLOC_SITE_OTHER_COUNT.store(0, Ordering::Relaxed);
    ALLOC_SITE_OTHER_BYTES.store(0, Ordering::Relaxed);
    ALLOC_SITE_PREPARE_COUNT.store(0, Ordering::Relaxed);
    ALLOC_SITE_PREPARE_BYTES.store(0, Ordering::Relaxed);
    ALLOC_SITE_OUTPUT_BUFFER_COUNT.store(0, Ordering::Relaxed);
    ALLOC_SITE_OUTPUT_BUFFER_BYTES.store(0, Ordering::Relaxed);
    ALLOC_SITE_CHUNK_SLICES_COUNT.store(0, Ordering::Relaxed);
    ALLOC_SITE_CHUNK_SLICES_BYTES.store(0, Ordering::Relaxed);
}

/// Which straight-line branch a bound op actually took — set once per call,
/// never re-derived from tensor shape after the fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Path {
    DotFast,
    WidthFast,
    Generic,
}

/// One bound-op call's local tally. All fields are plain `u64`s bumped by
/// real per-branch loop bounds already in scope at the call site (`width`,
/// `reduction_total`, `raw.len()`, per-operand contiguity) — never a
/// quantity re-derived from the op's declared extents after the fact.
#[derive(Debug, Default)]
pub struct KernelCounters {
    pub mac_ops: u64,
    pub operand_loads: u64,
    pub output_writes: u64,
    pub leading_iters: u64,
    pub kernel_calls: u64,
}

impl KernelCounters {
    pub fn commit(&self, path: Path, distinct_operand_elements: u64) {
        counter!(BOUND_OPS, 1);
        counter!(MAC_OPS, self.mac_ops);
        counter!(OPERAND_LOADS, self.operand_loads);
        counter!(DISTINCT_OPERAND_ELEMENTS, distinct_operand_elements);
        counter!(OUTPUT_WRITES, self.output_writes);
        counter!(LEADING_ITERS, self.leading_iters);
        counter!(KERNEL_CALLS, self.kernel_calls);
        match path {
            Path::DotFast => counter!(PATH_DOT_FAST, 1),
            Path::WidthFast => counter!(PATH_WIDTH_FAST, 1),
            Path::Generic => counter!(PATH_GENERIC, 1),
        }
    }
}

/// Snapshot of every counter's running total, for a caller (the
/// `profile_hot` example) that wants to print them without pulling in the
/// full recorder/exporter machinery.
#[derive(Debug, Clone, Copy, Default)]
pub struct Totals {
    pub bound_ops: u64,
    pub mac_ops: u64,
    pub operand_loads: u64,
    pub distinct_operand_elements: u64,
    pub output_writes: u64,
    pub path_dot_fast: u64,
    pub path_width_fast: u64,
    pub path_generic: u64,
    pub leading_iters: u64,
    pub kernel_calls: u64,
}

#[must_use]
pub fn totals() -> Totals {
    Totals {
        bound_ops: BOUND_OPS.get(),
        mac_ops: MAC_OPS.get(),
        operand_loads: OPERAND_LOADS.get(),
        distinct_operand_elements: DISTINCT_OPERAND_ELEMENTS.get(),
        output_writes: OUTPUT_WRITES.get(),
        path_dot_fast: PATH_DOT_FAST.get(),
        path_width_fast: PATH_WIDTH_FAST.get(),
        path_generic: PATH_GENERIC.get(),
        leading_iters: LEADING_ITERS.get(),
        kernel_calls: KERNEL_CALLS.get(),
    }
}

/// Resets every counter to zero — lets a caller isolate one program's totals
/// from whatever ran earlier in the same process.
pub fn reset() {
    let _ = BOUND_OPS.snapshot_and_reset();
    let _ = MAC_OPS.snapshot_and_reset();
    let _ = OPERAND_LOADS.snapshot_and_reset();
    let _ = DISTINCT_OPERAND_ELEMENTS.snapshot_and_reset();
    let _ = OUTPUT_WRITES.snapshot_and_reset();
    let _ = PATH_DOT_FAST.snapshot_and_reset();
    let _ = PATH_WIDTH_FAST.snapshot_and_reset();
    let _ = PATH_GENERIC.snapshot_and_reset();
    let _ = LEADING_ITERS.snapshot_and_reset();
    let _ = KERNEL_CALLS.snapshot_and_reset();
}
