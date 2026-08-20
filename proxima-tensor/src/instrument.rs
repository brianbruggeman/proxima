//! Real execution-witness counters for [`crate::cpu`]'s bound-op kernels,
//! gated entirely behind the `instrument` feature.
//!
//! Every field here is incremented from a plain local accumulator inside
//! the kernel loop and committed to the process-wide [`proxima_telemetry`]
//! counters exactly once, at the end of the bound-op call — never as an
//! atomic increment inside a loop that can run ~1e9 times, or the
//! instrument would perturb the thing it measures.

use core::future::Future;
use core::sync::atomic::{AtomicU64, Ordering};
use std::collections::{BTreeMap, HashSet};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::thread::ThreadId;

use proxima_clock::ticks::Ticks;
use proxima_primitives::pipe::Pipe;
use proxima_telemetry::counter;
use proxima_telemetry::metric::Counter;

use crate::op::NodeId;

// No hardware tick source ships in `proxima-clock` itself (by design — see
// that crate's module doc: "implement `Pipe` for your type, zero edits to
// this crate") and none of the workspace's existing monotonic readers fit:
// `prime::core::timer::Clock`'s production impl and
// `proxima_core::time::drivers::std_thread::StdThreadDriver` both still
// round-trip through `std::time::Instant`/`Duration` (the conversion this
// module exists to avoid), and `prime::core::timer::Clock` is ms-resolution
// besides. `raw_tick` below is the hardware read `TensorTickSource` (the
// `Pipe` source form the doc names) wraps; hot-path call sites in `cpu.rs`
// call [`read_ticks`] directly, a plain function, not the async `Pipe::call`
// path, to keep the hot loop allocation- and `Future`-free.
//
// Darwin: `mach_absolute_time` is the raw hardware counter with NO
// multiply/divide at read time — the conversion to nanoseconds
// (`mach_timebase_info`'s numer/denom) happens once, lazily, in
// [`ticks_to_nanos`], never per read. This is the actual saving over
// `std::time::Instant::now()`, whose Apple implementation performs that
// conversion on every read.
#[cfg(target_os = "macos")]
// `libc::mach_absolute_time` is deprecated in favor of the `mach2` crate;
// this file already carries `libc` as its one direct-syscall dependency
// (`thread_cpu_nanos`/`ru_minflt` below), so staying on it here avoids
// adding a second FFI crate for one function pair.
#[allow(deprecated)]
fn raw_tick() -> u64 {
    // SAFETY: `mach_absolute_time` takes no arguments and only reads a
    // hardware register; always safe to call.
    unsafe { libc::mach_absolute_time() }
}

// Non-Darwin: `clock_gettime(CLOCK_MONOTONIC)` is already ticksecond-native
// (the kernel/vDSO does its own scaling once, not duplicated per caller), so
// there is no separate timebase conversion to defer — [`ticks_to_nanos`] is
// the identity function on this path.
#[cfg(not(target_os = "macos"))]
fn raw_tick() -> u64 {
    let mut now = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: `now` is a valid out-pointer; `CLOCK_MONOTONIC` is supported
    // on every target this crate builds for.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now) };
    (now.tv_sec as u64) * 1_000_000_000 + (now.tv_nsec as u64)
}

/// One hardware-clock reading, hot-path shape: a plain function returning
/// [`Ticks`], not a method on a bespoke stopwatch type — every one of
/// `cpu.rs`'s ~25,000-per-forward call sites wants exactly this and nothing
/// more. Composes with [`elapsed_ticks`] the same way any two [`Ticks`]
/// readings do (`Ticks::wrapping_sub`).
#[must_use]
pub fn read_ticks() -> Ticks {
    Ticks::from_raw(raw_tick())
}

/// `read_ticks() - started`, in raw tick units — never converted to
/// nanoseconds here. Store the result straight into a counter; convert with
/// [`ticks_to_nanos`] once, at the print/export edge, not per call.
#[must_use]
pub fn elapsed_ticks(started: Ticks) -> u64 {
    read_ticks().wrapping_sub(started)
}

#[cfg(target_os = "macos")]
// same `libc`-over-`mach2` rationale as `raw_tick` above.
#[allow(deprecated)]
fn timebase() -> (u64, u64) {
    static TIMEBASE: OnceLock<(u64, u64)> = OnceLock::new();
    *TIMEBASE.get_or_init(|| {
        let mut info = libc::mach_timebase_info { numer: 0, denom: 0 };
        // SAFETY: `info` is a valid out-pointer.
        unsafe { libc::mach_timebase_info(&mut info) };
        (u64::from(info.numer), u64::from(info.denom).max(1))
    })
}

/// The one-time-per-export conversion [`Ticks`]'s own doc names as the only
/// place this multiply/divide belongs. Identity on platforms whose raw tick
/// unit is already nanoseconds (everywhere [`raw_tick`] is not
/// `mach_absolute_time`).
#[must_use]
pub fn ticks_to_nanos(ticks: u64) -> u64 {
    #[cfg(target_os = "macos")]
    {
        let (numer, denom) = timebase();
        u64::try_from(u128::from(ticks) * u128::from(numer) / u128::from(denom)).unwrap_or(u64::MAX)
    }
    #[cfg(not(target_os = "macos"))]
    {
        ticks
    }
}

/// The hardware tick source, expressed as the source-shaped [`Pipe`]
/// `proxima-clock`'s own module doc names (`In = ()`, `Out = Ticks`,
/// `Err = Infallible`) — see `proxima_clock::ticks` for why a tick source is
/// a `Pipe` and not a bespoke clock trait. This is the composable handle for
/// callers building a pipe chain (e.g. `.and_then` into
/// `proxima_clock::anchor::ToUnixNanos`); `cpu.rs`'s hot loop calls
/// [`read_ticks`] directly instead, the same raw read this type's `call`
/// wraps, so a hot-path measurement never pays for a `Future`.
#[derive(Debug, Clone, Copy, Default)]
pub struct TensorTickSource;

impl Pipe for TensorTickSource {
    type In = ();
    type Out = Ticks;
    type Err = core::convert::Infallible;

    fn call(&self, (): ()) -> impl Future<Output = Result<Ticks, core::convert::Infallible>> {
        core::future::ready(Ok(read_ticks()))
    }
}

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
pub static PARALLEL_NODE_TICKS: Counter = Counter::new("proxima_tensor.parallel_node_ticks");
pub static PARALLEL_SPAWN_TICKS: Counter = Counter::new("proxima_tensor.parallel_spawn_ticks");
pub static PARALLEL_JOIN_TICKS: Counter = Counter::new("proxima_tensor.parallel_join_ticks");
pub static PARALLEL_CHUNK_COUNT: Counter = Counter::new("proxima_tensor.parallel_chunk_count");
pub static PARALLEL_CHUNK_TICKS_SUM: Counter =
    Counter::new("proxima_tensor.parallel_chunk_ticks_sum");
// Counter has no min/max form, so the extremes live in their own atomics,
// updated with fetch_min/fetch_max — the same lock-free discipline the
// counters use, just without a running sum.
pub static PARALLEL_CHUNK_TICKS_MIN: AtomicU64 = AtomicU64::new(u64::MAX);
pub static PARALLEL_CHUNK_TICKS_MAX: AtomicU64 = AtomicU64::new(0);

/// Records one chunk's compute duration into the sum/count/min/max quartet.
/// Called once per chunk after `run_node_into` returns — never inside the
/// kernel loop itself.
pub fn record_chunk_ticks(ticks: u64) {
    counter!(PARALLEL_CHUNK_TICKS_SUM, ticks);
    counter!(PARALLEL_CHUNK_COUNT, 1);
    PARALLEL_CHUNK_TICKS_MIN.fetch_min(ticks, Ordering::Relaxed);
    PARALLEL_CHUNK_TICKS_MAX.fetch_max(ticks, Ordering::Relaxed);
}

/// One process run's worth of parallel-dispatch timing, read back by the
/// `sweep_gemm` harness after evaluation — a snapshot, not a reset, so the
/// harness can also print an end-of-run summary without disturbing counters
/// a caller still wants to read again.
#[derive(Debug, Clone, Copy, Default)]
pub struct ParallelTotals {
    pub parallel_nodes: u64,
    pub node_ticks: u64,
    pub spawn_ticks: u64,
    pub join_ticks: u64,
    pub chunk_count: u64,
    pub chunk_ticks_sum: u64,
    pub chunk_ticks_min: u64,
    pub chunk_ticks_max: u64,
}

#[must_use]
pub fn parallel_totals() -> ParallelTotals {
    let chunk_count = PARALLEL_CHUNK_COUNT.get();
    let observed_min = PARALLEL_CHUNK_TICKS_MIN.load(Ordering::Relaxed);
    ParallelTotals {
        parallel_nodes: PARALLEL_NODES.get(),
        node_ticks: PARALLEL_NODE_TICKS.get(),
        spawn_ticks: PARALLEL_SPAWN_TICKS.get(),
        join_ticks: PARALLEL_JOIN_TICKS.get(),
        chunk_count,
        chunk_ticks_sum: PARALLEL_CHUNK_TICKS_SUM.get(),
        // no chunk was ever recorded: report 0, not the u64::MAX sentinel.
        chunk_ticks_min: if chunk_count == 0 { 0 } else { observed_min },
        chunk_ticks_max: PARALLEL_CHUNK_TICKS_MAX.load(Ordering::Relaxed),
    }
}

/// Resets the parallel-dispatch counters to their initial state — mirrors
/// [`reset`] but kept separate so a caller can reset one family without
/// disturbing the kernel counters.
pub fn reset_parallel() {
    let _ = PARALLEL_NODES.snapshot_and_reset();
    let _ = PARALLEL_NODE_TICKS.snapshot_and_reset();
    let _ = PARALLEL_SPAWN_TICKS.snapshot_and_reset();
    let _ = PARALLEL_JOIN_TICKS.snapshot_and_reset();
    let _ = PARALLEL_CHUNK_COUNT.snapshot_and_reset();
    let _ = PARALLEL_CHUNK_TICKS_SUM.snapshot_and_reset();
    PARALLEL_CHUNK_TICKS_MIN.store(u64::MAX, Ordering::Relaxed);
    PARALLEL_CHUNK_TICKS_MAX.store(0, Ordering::Relaxed);
}

// chunk duration (above) scatters by construction as chunk count grows past
// worker count under oversubscription, so it cannot tell a balanced pool
// apart from an unbalanced one. what actually decides whether the parallel
// region is bottlenecked on one straggler is each PULLER's total busy time —
// summed across every chunk that puller claimed — which is why this is
// keyed by the calling thread, not by chunk index.
static WORKER_BUSY_TICKS: Mutex<Vec<(ThreadId, u64)>> = Mutex::new(Vec::new());

/// Adds `ticks` to the current thread's running total. Called from the same
/// per-chunk timing site as [`record_chunk_ticks`] — this is a second,
/// orthogonal aggregation of the identical measurement, grouped by puller
/// instead of by chunk.
pub fn record_worker_busy_ticks(ticks: u64) {
    let thread_id = std::thread::current().id();
    let mut totals = WORKER_BUSY_TICKS.lock().unwrap_or_else(PoisonError::into_inner);
    match totals.iter_mut().find(|(existing, _)| *existing == thread_id) {
        Some((_, total)) => *total += ticks,
        None => totals.push((thread_id, ticks)),
    }
}

/// Every worker's accumulated busy time from the most recent parallel
/// region(s) since the last [`reset_worker_busy`] — one entry per distinct
/// thread that claimed at least one chunk. Order is not meaningful.
#[must_use]
pub fn worker_busy_snapshot() -> Vec<u64> {
    let totals = WORKER_BUSY_TICKS.lock().unwrap_or_else(PoisonError::into_inner);
    totals.iter().map(|(_, ticks)| *ticks).collect()
}

pub fn reset_worker_busy() {
    let mut totals = WORKER_BUSY_TICKS.lock().unwrap_or_else(PoisonError::into_inner);
    totals.clear();
}

// the busy total above is `Instant`-derived, so a worker the OS descheduled
// keeps accruing "busy" ticks while off-core. on a box carrying any ambient
// load that turns the 1->8 scaling read into a measurement of the host: a
// register-only fma control (zero memory traffic, so no scaling effect is
// even possible) measured +41.2% wall growth 1->8 against +6.8% cpu growth,
// n=9, 2026-08-18. every 1->8 figure taken before this existed used the wall
// form and is not separable from that. the cpu clock below is the same
// measurement against a clock that stops when the thread does; carry BOTH,
// because their ratio is the only in-band report of how much the host
// interfered with the run.
static WORKER_CPU_NANOS: Mutex<Vec<(ThreadId, u64)>> = Mutex::new(Vec::new());

/// This thread's consumed CPU time. Unlike an [`Instant`](std::time::Instant)
/// delta, this does not advance while the thread is off-core.
#[must_use]
pub fn thread_cpu_nanos() -> u64 {
    let mut now = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut now) } != 0 {
        return 0;
    }
    (now.tv_sec as u64) * 1_000_000_000 + (now.tv_nsec as u64)
}

/// Adds `nanos` of consumed CPU time to the current thread's running total,
/// the deschedule-immune peer of [`record_worker_busy_ticks`].
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
    totals.iter().map(|(_, ticks)| *ticks).collect()
}

pub fn reset_worker_cpu() {
    let mut totals = WORKER_CPU_NANOS.lock().unwrap_or_else(PoisonError::into_inner);
    totals.clear();
}

/// This process's cumulative minor-fault count (`getrusage`'s `ru_minflt`)
/// at the moment of the call — monotonic for the process lifetime, so a
/// caller diffs two readings to get the fault count over an interval.
/// Used to test whether a measured wall-time change is first-touch page-in
/// (mmap demand paging) rather than compute: a forward pass that walks a
/// weight mapping for the first time pays one minor fault per page touched.
#[must_use]
pub fn ru_minflt() -> u64 {
    let mut usage: libc::rusage = unsafe { core::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return 0;
    }
    usage.ru_minflt as u64
}

// `matmul_rows_threaded`'s own dispatch-overhead breakdown (`cpu.rs`) --
// distinct from `PARALLEL_SPAWN_TICKS`/`PARALLEL_JOIN_TICKS` above, which
// only ever fire from `run_chunks_threaded`'s `BoundOp`-chunk path. The
// quantized-matmul forward never reaches that path (`evaluate_quantized`'s
// loop calls `run_node_into` directly, no `evaluate_node_parallel`), so
// without these, the row-chunk pool dispatch inside a matmul node had no
// spawn-vs-wait-vs-compute breakdown at all. Committed once per
// `matmul_rows_threaded` call, never per chunk.
// `quantized_matmul_workers`'s own call/decision counters -- the single
// choke point every `Q4_K`/`Q5_K`/`Q6_K` row-batch call passes through
// regardless of int8-dot vs f32-dot kernel. `MATMUL_WORKERS_CALLS` is the
// true total row-batch count; `MATMUL_WORKERS_NONE` is how many of those
// took the sequential (no thread pool) fallback -- their difference is
// `MATMUL_DISPATCH_CALLS`'s expected value if every threaded decision
// reaches `matmul_rows_threaded`.
pub static MATMUL_WORKERS_CALLS: Counter = Counter::new("proxima_tensor.matmul.workers_calls");
pub static MATMUL_WORKERS_NONE: Counter = Counter::new("proxima_tensor.matmul.workers_none");

pub static MATMUL_DISPATCH_CALLS: Counter = Counter::new("proxima_tensor.matmul.dispatch_calls");
// everything `matmul_rows_threaded` (`cpu.rs`) does BEFORE its own
// spawn/own_chunk/recv_wait timer chain starts: the `output` Vec alloc,
// the `chunk_ranges` Vec build (one `split_at_mut` per chunk), the
// `nest_pool()` OnceLock fetch, and the `Arc`/`sync_channel` allocations --
// none of which any existing MATMUL_* counter captured, so a caller could
// not tell "the dispatch chain is slow" apart from "the untimed setup
// before it is slow" (`reduce_quantized_ms` minus spawn+own_chunk+recv_wait
// is exactly this).
pub static MATMUL_SETUP_TICKS: Counter = Counter::new("proxima_tensor.matmul.setup_ticks");
// `std::thread::available_parallelism()` (`cpu.rs::quantized_matmul_workers`)
// is called once per row-batch call (once per position, per matmul node --
// 1296 times this forward), not cached anywhere. Unlike a libc `sysconf`
// result an OS is free to memoize internally, Rust's std does not cache
// this across calls on every platform, so a caller could not previously
// tell "the per-call syscall/query cost is negligible" apart from "it is
// the missing time" -- this is the direct witness.
pub static MATMUL_AVAILABLE_PARALLELISM_TICKS: Counter =
    Counter::new("proxima_tensor.matmul.available_parallelism_ticks");
pub static MATMUL_SPAWN_TICKS: Counter = Counter::new("proxima_tensor.matmul.spawn_ticks");
pub static MATMUL_OWN_CHUNK_TICKS: Counter = Counter::new("proxima_tensor.matmul.own_chunk_ticks");
pub static MATMUL_RECV_WAIT_TICKS: Counter = Counter::new("proxima_tensor.matmul.recv_wait_ticks");
// the activation-quantize preamble every `matmul_q4k_q8k_f32` call pays
// once, BEFORE `quantized_matmul_workers`/`matmul_rows_threaded` even run
// (`quantize_row_q8k` in `cpu.rs`) -- not part of the row-chunk dispatch at
// all, so it needed a separate timer once the spawn/own-chunk/recv-wait
// trio above did not account for a node's full wall time on its own.
pub static MATMUL_QUANTIZE_ACTIVATION_TICKS: Counter =
    Counter::new("proxima_tensor.matmul.quantize_activation_ticks");
// whole-function timer around `run_reduce_quantized` (once per matmul
// NODE, same granularity as the per-node-kind table), to localize a gap
// between a node's total wall time and the sum of
// spawn/own-chunk/recv-wait/quantize-activation: if this matches the node
// total, the gap is inside the position loop (a codec path none of the
// above four time); if it matches the four-way sum instead, the gap is
// OUTSIDE `run_reduce_quantized` entirely.
pub static MATMUL_REDUCE_QUANTIZED_TICKS: Counter =
    Counter::new("proxima_tensor.matmul.reduce_quantized_ticks");
// `matmul_q5k_f32`/`matmul_q6k_f32` (`cpu.rs`) never call
// `quantized_matmul_workers` at all -- they are a plain sequential
// `chunks_exact().map().collect()` over every weight row, unconditionally,
// regardless of size. Neither codec's `-int8-dot` feature is enabled by
// this workspace's default features, so every `Q5_K`/`Q6_K` weight tensor
// in a real checkpoint runs its matmul fully single-threaded, invisible to
// every `MATMUL_*` counter above (none of which this call path ever
// reaches). These two counters are the only witness of that time.
pub static MATMUL_Q5K_F32_CALLS: Counter = Counter::new("proxima_tensor.matmul.q5k_f32_calls");
pub static MATMUL_Q5K_F32_TICKS: Counter = Counter::new("proxima_tensor.matmul.q5k_f32_ticks");
pub static MATMUL_Q6K_F32_CALLS: Counter = Counter::new("proxima_tensor.matmul.q6k_f32_calls");
pub static MATMUL_Q6K_F32_TICKS: Counter = Counter::new("proxima_tensor.matmul.q6k_f32_ticks");

// per-call mac/ticks witness for `run_reduce_quantized`'s position loop
// (`cpu.rs`), split by codec -- `rows * contraction_width` is an element
// count already at hand at the call site (`run_reduce_quantized`'s own
// `rows`/`k` locals), not re-derived from tensor shape after the fact.
// Answers "measured ns/mac in situ" per codec, directly comparable against
// the isolated single-threaded kernel bench (0.0334 ns/mac) and ggml's
// (0.0255 ns/mac) -- if one codec's in-situ ns/mac is far worse than the
// other two, this is the only witness that names which.
pub static MATMUL_Q4K_MACS: Counter = Counter::new("proxima_tensor.matmul.q4k_macs");
pub static MATMUL_Q4K_CALL_TICKS: Counter = Counter::new("proxima_tensor.matmul.q4k_call_ticks");
pub static MATMUL_Q5K_MACS: Counter = Counter::new("proxima_tensor.matmul.q5k_macs");
pub static MATMUL_Q5K_CALL_TICKS: Counter = Counter::new("proxima_tensor.matmul.q5k_call_ticks");
pub static MATMUL_Q6K_MACS: Counter = Counter::new("proxima_tensor.matmul.q6k_macs");
pub static MATMUL_Q6K_CALL_TICKS: Counter = Counter::new("proxima_tensor.matmul.q6k_call_ticks");
// how many times `run_reduce_quantized`'s position loop ran a single
// weight's matmul, and how many distinct nodes it ran across -- the
// direct witness for whether a node's `leading_total` positions are
// dispatched as one wide row-batch or `leading_total` separate ones
// (each paying its own `matmul_rows_threaded` spawn/recv-wait).
pub static MATMUL_POSITION_LOOP_ITERS: Counter =
    Counter::new("proxima_tensor.matmul.position_loop_iters");
pub static MATMUL_REDUCE_QUANTIZED_CALLS: Counter =
    Counter::new("proxima_tensor.matmul.reduce_quantized_calls");

// diagnostic-only, keyed by (rows, k) -- every aggregate MATMUL_Q4K_MACS/
// MATMUL_Q4K_CALL_TICKS above sums across all 7 matmul shapes a real
// forward pass runs per layer, so it cannot tell "attn_q's threading win
// held" apart from "attn_k lost it and attn_q's win hid the loss in the
// average." `run_reduce_quantized` (`cpu.rs`) already has `rows`/`k` at the
// exact site the aggregate counters fire from; this bucket is the same
// measurement (per-call ticks including the activation-quantize preamble,
// wrapping the whole `matmul_q4k_q8k_f32` call) split by shape instead of
// summed away.
type ShapeKey = (u64, u64);
type ShapeTotals = (u64, u64, u64);
static Q4K_SHAPE_TICKS: Mutex<BTreeMap<ShapeKey, ShapeTotals>> = Mutex::new(BTreeMap::new());

/// Adds one `(rows, k)`-shaped call's macs and elapsed ticks to that
/// shape's running `(calls, macs, ticks)` triple.
pub fn record_q4k_shape_call(rows: u64, k: u64, macs: u64, ticks: u64) {
    let mut buckets = Q4K_SHAPE_TICKS.lock().unwrap_or_else(PoisonError::into_inner);
    let entry = buckets.entry((rows, k)).or_insert((0, 0, 0));
    entry.0 += 1;
    entry.1 += macs;
    entry.2 += ticks;
}

/// Every distinct `(rows, k)` shape recorded since the last
/// [`reset_q4k_shape_buckets`], as `(rows, k, calls, macs, ticks)` — sorted
/// by key (`BTreeMap` iteration order), not by any measured field.
#[must_use]
pub fn q4k_shape_snapshot() -> Vec<(u64, u64, u64, u64, u64)> {
    let buckets = Q4K_SHAPE_TICKS.lock().unwrap_or_else(PoisonError::into_inner);
    buckets
        .iter()
        .map(|(&(rows, k), &(calls, macs, ticks))| (rows, k, calls, macs, ticks))
        .collect()
}

pub fn reset_q4k_shape_buckets() {
    let mut buckets = Q4K_SHAPE_TICKS.lock().unwrap_or_else(PoisonError::into_inner);
    buckets.clear();
}

/// One process run's worth of [`matmul_rows_threaded`](crate::cpu)'s own
/// dispatch-overhead breakdown, read back the same way [`parallel_totals`]
/// is.
#[derive(Debug, Clone, Copy, Default)]
pub struct MatmulDispatchTotals {
    pub workers_calls: u64,
    pub workers_none: u64,
    pub calls: u64,
    pub setup_ticks: u64,
    pub available_parallelism_ticks: u64,
    pub spawn_ticks: u64,
    pub own_chunk_ticks: u64,
    pub recv_wait_ticks: u64,
    pub quantize_activation_ticks: u64,
    pub reduce_quantized_ticks: u64,
    pub q5k_f32_calls: u64,
    pub q5k_f32_ticks: u64,
    pub q6k_f32_calls: u64,
    pub q6k_f32_ticks: u64,
    pub q4k_macs: u64,
    pub q4k_call_ticks: u64,
    pub q5k_macs: u64,
    pub q5k_call_ticks: u64,
    pub q6k_macs: u64,
    pub q6k_call_ticks: u64,
    pub position_loop_iters: u64,
    pub reduce_quantized_calls: u64,
    pub q4k_transpose_ticks: u64,
}

#[must_use]
pub fn matmul_dispatch_totals() -> MatmulDispatchTotals {
    MatmulDispatchTotals {
        workers_calls: MATMUL_WORKERS_CALLS.get(),
        workers_none: MATMUL_WORKERS_NONE.get(),
        calls: MATMUL_DISPATCH_CALLS.get(),
        setup_ticks: MATMUL_SETUP_TICKS.get(),
        available_parallelism_ticks: MATMUL_AVAILABLE_PARALLELISM_TICKS.get(),
        spawn_ticks: MATMUL_SPAWN_TICKS.get(),
        own_chunk_ticks: MATMUL_OWN_CHUNK_TICKS.get(),
        recv_wait_ticks: MATMUL_RECV_WAIT_TICKS.get(),
        quantize_activation_ticks: MATMUL_QUANTIZE_ACTIVATION_TICKS.get(),
        reduce_quantized_ticks: MATMUL_REDUCE_QUANTIZED_TICKS.get(),
        q5k_f32_calls: MATMUL_Q5K_F32_CALLS.get(),
        q5k_f32_ticks: MATMUL_Q5K_F32_TICKS.get(),
        q6k_f32_calls: MATMUL_Q6K_F32_CALLS.get(),
        q6k_f32_ticks: MATMUL_Q6K_F32_TICKS.get(),
        q4k_macs: MATMUL_Q4K_MACS.get(),
        q4k_call_ticks: MATMUL_Q4K_CALL_TICKS.get(),
        q5k_macs: MATMUL_Q5K_MACS.get(),
        q5k_call_ticks: MATMUL_Q5K_CALL_TICKS.get(),
        q6k_macs: MATMUL_Q6K_MACS.get(),
        q6k_call_ticks: MATMUL_Q6K_CALL_TICKS.get(),
        position_loop_iters: MATMUL_POSITION_LOOP_ITERS.get(),
        reduce_quantized_calls: MATMUL_REDUCE_QUANTIZED_CALLS.get(),
        q4k_transpose_ticks: MATMUL_Q4K_TRANSPOSE_TICKS.get(),
    }
}

/// Resets the matmul dispatch-overhead counters — mirrors [`reset_parallel`].
pub fn reset_matmul_dispatch() {
    let _ = MATMUL_WORKERS_CALLS.snapshot_and_reset();
    let _ = MATMUL_WORKERS_NONE.snapshot_and_reset();
    let _ = MATMUL_DISPATCH_CALLS.snapshot_and_reset();
    let _ = MATMUL_SETUP_TICKS.snapshot_and_reset();
    let _ = MATMUL_AVAILABLE_PARALLELISM_TICKS.snapshot_and_reset();
    let _ = MATMUL_SPAWN_TICKS.snapshot_and_reset();
    let _ = MATMUL_OWN_CHUNK_TICKS.snapshot_and_reset();
    let _ = MATMUL_RECV_WAIT_TICKS.snapshot_and_reset();
    let _ = MATMUL_QUANTIZE_ACTIVATION_TICKS.snapshot_and_reset();
    let _ = MATMUL_REDUCE_QUANTIZED_TICKS.snapshot_and_reset();
    let _ = MATMUL_Q4K_TRANSPOSE_TICKS.snapshot_and_reset();
    let _ = MATMUL_Q5K_F32_CALLS.snapshot_and_reset();
    let _ = MATMUL_Q5K_F32_TICKS.snapshot_and_reset();
    let _ = MATMUL_Q6K_F32_CALLS.snapshot_and_reset();
    let _ = MATMUL_Q6K_F32_TICKS.snapshot_and_reset();
    let _ = MATMUL_Q4K_MACS.snapshot_and_reset();
    let _ = MATMUL_Q4K_CALL_TICKS.snapshot_and_reset();
    let _ = MATMUL_Q5K_MACS.snapshot_and_reset();
    let _ = MATMUL_Q5K_CALL_TICKS.snapshot_and_reset();
    let _ = MATMUL_Q6K_MACS.snapshot_and_reset();
    let _ = MATMUL_Q6K_CALL_TICKS.snapshot_and_reset();
    let _ = MATMUL_POSITION_LOOP_ITERS.snapshot_and_reset();
    let _ = MATMUL_REDUCE_QUANTIZED_CALLS.snapshot_and_reset();
}

// `evaluate_parallel`'s own wall-clock, decomposed into every named part
// that is NOT inside `run_chunks_threaded`'s `thread::scope` (which
// `PARALLEL_NODE_TICKS` above already measures, now scoped to start right
// before `thread::scope`, after slice-carving — see `cpu::run_chunks_threaded`).
// Each part is timed once per `evaluate_parallel` call (or once per resolved
// node, for the per-node parts) and committed after the timed region, never
// as a per-element accumulation.
pub static SERIAL_PREPARE_TICKS: Counter = Counter::new("proxima_tensor.serial_prepare_ticks");
pub static SERIAL_ALLOC_TICKS: Counter = Counter::new("proxima_tensor.serial_alloc_ticks");
pub static SERIAL_SPLIT_TICKS: Counter = Counter::new("proxima_tensor.serial_split_ticks");
pub static SERIAL_SLICE_CARVE_TICKS: Counter = Counter::new("proxima_tensor.serial_slice_carve_ticks");
pub static SERIAL_FINISH_TICKS: Counter = Counter::new("proxima_tensor.serial_finish_ticks");
pub static SERIAL_BOOKKEEPING_TICKS: Counter = Counter::new("proxima_tensor.serial_bookkeeping_ticks");
// only nonzero on the `workers == 1` (or below-threshold) arm, where
// `evaluate_node_parallel` never reaches `run_chunks_threaded` at all.
pub static SERIAL_SEQUENTIAL_COMPUTE_TICKS: Counter =
    Counter::new("proxima_tensor.serial_sequential_compute_ticks");
pub static SERIAL_EVALUATE_PARALLEL_TICKS: Counter =
    Counter::new("proxima_tensor.serial_evaluate_parallel_ticks");
pub static SERIAL_EVALUATE_PARALLEL_CALLS: Counter =
    Counter::new("proxima_tensor.serial_evaluate_parallel_calls");

/// One process run's worth of `evaluate_parallel`'s serial-remainder
/// breakdown, read back the same way [`parallel_totals`] is.
#[derive(Debug, Clone, Copy, Default)]
pub struct SerialTotals {
    pub prepare_ticks: u64,
    pub alloc_ticks: u64,
    pub split_ticks: u64,
    pub slice_carve_ticks: u64,
    pub finish_ticks: u64,
    pub bookkeeping_ticks: u64,
    pub sequential_compute_ticks: u64,
    pub evaluate_parallel_ticks: u64,
    pub evaluate_parallel_calls: u64,
}

#[must_use]
pub fn serial_totals() -> SerialTotals {
    SerialTotals {
        prepare_ticks: SERIAL_PREPARE_TICKS.get(),
        alloc_ticks: SERIAL_ALLOC_TICKS.get(),
        split_ticks: SERIAL_SPLIT_TICKS.get(),
        slice_carve_ticks: SERIAL_SLICE_CARVE_TICKS.get(),
        finish_ticks: SERIAL_FINISH_TICKS.get(),
        bookkeeping_ticks: SERIAL_BOOKKEEPING_TICKS.get(),
        sequential_compute_ticks: SERIAL_SEQUENTIAL_COMPUTE_TICKS.get(),
        evaluate_parallel_ticks: SERIAL_EVALUATE_PARALLEL_TICKS.get(),
        evaluate_parallel_calls: SERIAL_EVALUATE_PARALLEL_CALLS.get(),
    }
}

/// Resets the serial-breakdown counters to their initial state — mirrors
/// [`reset_parallel`] but kept separate so a caller can reset one family
/// without disturbing the others.
pub fn reset_serial() {
    let _ = SERIAL_PREPARE_TICKS.snapshot_and_reset();
    let _ = SERIAL_ALLOC_TICKS.snapshot_and_reset();
    let _ = SERIAL_SPLIT_TICKS.snapshot_and_reset();
    let _ = SERIAL_SLICE_CARVE_TICKS.snapshot_and_reset();
    let _ = SERIAL_FINISH_TICKS.snapshot_and_reset();
    let _ = SERIAL_BOOKKEEPING_TICKS.snapshot_and_reset();
    let _ = SERIAL_SEQUENTIAL_COMPUTE_TICKS.snapshot_and_reset();
    let _ = SERIAL_EVALUATE_PARALLEL_TICKS.snapshot_and_reset();
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

// per-operand-node witness: which weights actually got touched, and how
// much. Keyed by the operand's own `NodeId` (the tensor being read, e.g. one
// weight matrix's `Op::Input` node) rather than by the bound op reading it,
// because the question this answers — "how cold is this weight" — is a
// property of the tensor, not of any one kernel invocation that touched it.
//
// `reads`/`bytes` sum across every bound-op call that reads this node within
// one process run, because that genuinely accumulates: reading the same
// weight on every decode step spends bandwidth every time, so the total is
// the honest answer to "how many bytes moved for this node". `distinct`/
// `total_elements` instead take the running max/set-union, because a
// weight's own footprint does not grow just because more calls read it —
// summing per-call closed-form distinct counts across repeated calls to the
// SAME node (or across a `BoundOp::split` chunk fan-out of the SAME logical
// node) would double-count the overlap. Both `record_operand_access` and
// `commit_gather_operand_access` are called exactly once per (bound-op,
// operand) pair, from `cpu::record_bound_op_operand_access`, which is itself
// called once per node evaluation from `evaluate_pooled`/
// `evaluate_node_parallel`/`Interpreter::fold` — never per element, and
// always against the UNSPLIT op's own extents/strides, so a parallel
// chunk fan-out is invisible to this accounting (see that function's doc).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OperandAccess {
    pub reads: u64,
    pub bytes: u64,
    pub distinct_elements: u64,
    pub total_elements: u64,
}

static OPERAND_ACCESS: Mutex<BTreeMap<NodeId, OperandAccess>> = Mutex::new(BTreeMap::new());

/// Records one operand's participation in one bound-op call. `reads` and
/// `distinct_elements` are closed-form counts computed once from the
/// caller's own extents/strides (`cpu::operand_access_footprint`) — never
/// derived by walking elements. See the module-level comment above this
/// struct for why `reads`/`bytes` sum across calls while `distinct_elements`/
/// `total_elements` take the max.
pub fn record_operand_access(node: NodeId, reads: u64, distinct_elements: u64, total_elements: u64) {
    let bytes = reads * size_of::<f32>() as u64;
    let mut table = OPERAND_ACCESS.lock().unwrap_or_else(PoisonError::into_inner);
    let entry = table.entry(node).or_default();
    entry.reads += reads;
    entry.bytes += bytes;
    entry.distinct_elements = entry.distinct_elements.max(distinct_elements);
    entry.total_elements = entry.total_elements.max(total_elements);
}

/// Same accounting as [`record_operand_access`], for a gathered (embedding
/// -lookup-shaped) operand: `distinct_elements` cannot be derived from
/// strides alone (which row a gather touches is a runtime index value, not
/// an affine function of the loop coordinate), so it is read back from the
/// real row-index witness [`record_gather_row`] built up during execution,
/// then scaled by `row_width` (the table's own row stride) to report
/// elements rather than rows — matching what [`OperandAccess::distinct_elements`]
/// means for every other operand.
pub fn commit_gather_operand_access(node: NodeId, reads: u64, row_width: u64, total_elements: u64) {
    let distinct_elements = gather_distinct_rows(node) * row_width;
    record_operand_access(node, reads, distinct_elements, total_elements);
}

/// One operand node's snapshot row, node-attributed — the shape
/// [`operand_access_totals`] returns since, unlike every other snapshot in
/// this file, there is one entry per node rather than one process-wide
/// total.
#[derive(Debug, Clone, Copy)]
pub struct OperandAccessRow {
    pub node: NodeId,
    pub access: OperandAccess,
}

/// Every operand node witnessed so far, in `NodeId` order. Mirrors
/// [`worker_busy_snapshot`]'s shape (a plain snapshot, not a reset) so a
/// caller can print an end-of-run table without disturbing counters it
/// still wants to read again.
#[must_use]
pub fn operand_access_totals() -> Vec<OperandAccessRow> {
    let table = OPERAND_ACCESS.lock().unwrap_or_else(PoisonError::into_inner);
    table
        .iter()
        .map(|(&node, &access)| OperandAccessRow { node, access })
        .collect()
}

/// This node's own accumulated access, or `None` if it was never handed to
/// [`record_operand_access`]/[`commit_gather_operand_access`] at all —
/// distinct from "recorded zero reads", which is a real `Some` row with
/// every field `0`. A caller that wants to tell "never read" apart from
/// "not instrumented" needs exactly this distinction; folding both into a
/// bare `0` would erase it.
#[must_use]
pub fn operand_access_of(node: NodeId) -> Option<OperandAccess> {
    let table = OPERAND_ACCESS.lock().unwrap_or_else(PoisonError::into_inner);
    table.get(&node).copied()
}

pub fn reset_operand_access() {
    let mut table = OPERAND_ACCESS.lock().unwrap_or_else(PoisonError::into_inner);
    table.clear();
    let mut rows = GATHER_ROWS_TOUCHED.lock().unwrap_or_else(PoisonError::into_inner);
    rows.clear();
}

// distinct-row witness for gathered operands, keyed by the table's own
// `NodeId` — a real per-run set, not a closed form, because which rows a
// gather touches depends on the fetched index values, which are only known
// at runtime. Populated by `cpu::fill_gather_cursors`, which already reads
// each gathered operand's index once per row (elementwise/scan) or once per
// leading x reduction step (reduce's generic fallback, which never reaches
// the ~1e9-iteration fast/tile paths in the first place — see that
// function's own call sites) to seed its cursor; this instrument piggybacks
// on that same, already-paid-for read rather than adding a second one.
static GATHER_ROWS_TOUCHED: Mutex<BTreeMap<NodeId, HashSet<u64>>> = Mutex::new(BTreeMap::new());

/// Marks `row_index` as touched for the gathered table `node`. Called once
/// per row-level index fetch, from `cpu::fill_gather_cursors` — see the
/// static above for why that call frequency is cheap.
pub fn record_gather_row(node: NodeId, row_index: u64) {
    let mut rows = GATHER_ROWS_TOUCHED.lock().unwrap_or_else(PoisonError::into_inner);
    rows.entry(node).or_default().insert(row_index);
}

/// How many distinct rows of `node`'s table have been touched so far in
/// this process run. `0` for a node that was never gathered from at all,
/// same as an empty set would report — gather rows have no "never
/// instrumented" case to distinguish, unlike [`operand_access_of`], because
/// a gathered operand always reaches [`commit_gather_operand_access`] once
/// its bound op finishes.
#[must_use]
pub fn gather_distinct_rows(node: NodeId) -> u64 {
    let rows = GATHER_ROWS_TOUCHED.lock().unwrap_or_else(PoisonError::into_inner);
    rows.get(&node).map_or(0, HashSet::len) as u64
}

/// Total wall ticks the leader spends transposing `matmul_q4k_q8k_f32_impl`'s
/// row-major `[row][position]` result back into the position-major layout
/// `run_reduce_quantized`'s callers consume (`cpu.rs`'s copy loop after the
/// wide fold). Leader-serial: it is not inside any cohort round, so no
/// dispatch counter sees it.
pub static MATMUL_Q4K_TRANSPOSE_TICKS: Counter =
    Counter::new("proxima_tensor.matmul.q4k_transpose_ticks");

/// The cohort's own round-level forensics (`prime::os::cohort::diag`),
/// re-exported so a consumer of this crate's `instrument` feature reads
/// park/spin/unpark tallies and per-slot claim latencies without taking its
/// own `prime` dependency.
#[cfg(feature = "tensor-cohort")]
pub use prime::os::cohort::diag as cohort;
