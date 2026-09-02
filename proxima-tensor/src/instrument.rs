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
#[cfg(target_os = "macos")]
use std::sync::OnceLock;
use std::sync::{Mutex, PoisonError};
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
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
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
/// unit is already nanoseconds (everywhere `raw_tick` is not
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
pub static DISTINCT_OPERAND_ELEMENTS: Counter =
    Counter::new("proxima_tensor.distinct_operand_elements");
pub static OUTPUT_WRITES: Counter = Counter::new("proxima_tensor.output_writes");
pub static PATH_DOT_FAST: Counter = Counter::new("proxima_tensor.path.dot_fast");
pub static PATH_WIDTH_FAST: Counter = Counter::new("proxima_tensor.path.width_fast");
pub static PATH_GENERIC: Counter = Counter::new("proxima_tensor.path.generic");
/// `docs/discipline.md` ROW 149: the blocked (outer ci x contiguous inner
/// ky,kx) 2D GEMM tile for `Conv`'s own disjoint-leading-axis reduce shape —
/// distinct from [`PATH_DOT_FAST`], whose `neon_tile_plan` gate requires a
/// single shared leading axis `Conv` never has.
pub static PATH_CONV_TILE: Counter = Counter::new("proxima_tensor.path.conv_tile");
pub static LEADING_ITERS: Counter = Counter::new("proxima_tensor.leading_iters");
pub static KERNEL_CALLS: Counter = Counter::new("proxima_tensor.kernel_calls");

// per-path-kind wall-clock split for `run_reduce` alone (residual-profile
// task, 2026-08-30): `PATH_DOT_FAST`/`PATH_WIDTH_FAST`/`PATH_CONV_TILE`/
// `PATH_GENERIC` above are pure invocation counts shared with
// `run_elementwise_range`'s OWN, unrelated `Path::WidthFast`/`Path::Generic`
// usage (its affine-fast-path-vs-gather-loop split, a different question).
// These four are `run_reduce`-only, timed from a single `read_ticks()` call
// placed once `path` is decided (before any of `run_reduce`'s three early
// returns), committed at whichever of those returns actually fires — never
// re-derived from `PATH_*`'s counts, and never mixed with elementwise's own
// timing (`ELEMENTWISE_LOOP_TICKS*`). Answers `docs/discipline.md`'s own
// standing question — ROW 149's residual attribution to `width_fast`/
// `dot_fast` was "by elimination", never measured directly.
pub static REDUCE_PATH_DOT_FAST_TICKS: Counter =
    Counter::new("proxima_tensor.reduce_path.dot_fast_ticks");
pub static REDUCE_PATH_WIDTH_FAST_TICKS: Counter =
    Counter::new("proxima_tensor.reduce_path.width_fast_ticks");
pub static REDUCE_PATH_CONV_TILE_TICKS: Counter =
    Counter::new("proxima_tensor.reduce_path.conv_tile_ticks");
pub static REDUCE_PATH_GENERIC_TICKS: Counter =
    Counter::new("proxima_tensor.reduce_path.generic_ticks");

/// Records one `run_reduce` call's elapsed ticks against the path it
/// actually took — called once per call, from whichever of the three early
/// returns (or the final tail) fires, never per element.
pub fn record_reduce_path_ticks(path: Path, ticks: u64) {
    match path {
        Path::DotFast => counter!(REDUCE_PATH_DOT_FAST_TICKS, ticks),
        Path::WidthFast => counter!(REDUCE_PATH_WIDTH_FAST_TICKS, ticks),
        Path::ConvTile => counter!(REDUCE_PATH_CONV_TILE_TICKS, ticks),
        Path::Generic => counter!(REDUCE_PATH_GENERIC_TICKS, ticks),
    }
}

/// Snapshot of the four reduce-path (ALL reduces, not just gemm-shaped —
/// see [`reduce_gemm_path_totals`] for the gemm-restricted split) ticks
/// totals — `(dot_fast_ticks, width_fast_ticks, conv_tile_ticks,
/// generic_ticks)`. Same `.get()`-not-reset shape [`path_totals`] uses;
/// pair with [`reset_reduce_path`] for a per-run delta the way
/// [`reduce_gemm_path_totals`]'s own callers do.
#[must_use]
pub fn reduce_path_totals() -> (u64, u64, u64, u64) {
    (
        REDUCE_PATH_DOT_FAST_TICKS.get(),
        REDUCE_PATH_WIDTH_FAST_TICKS.get(),
        REDUCE_PATH_CONV_TILE_TICKS.get(),
        REDUCE_PATH_GENERIC_TICKS.get(),
    )
}

/// Resets the four reduce-path ticks counters to zero.
pub fn reset_reduce_path() {
    let _ = REDUCE_PATH_DOT_FAST_TICKS.snapshot_and_reset();
    let _ = REDUCE_PATH_WIDTH_FAST_TICKS.snapshot_and_reset();
    let _ = REDUCE_PATH_CONV_TILE_TICKS.snapshot_and_reset();
    let _ = REDUCE_PATH_GENERIC_TICKS.snapshot_and_reset();
}

// route-census task (2026-09-01): `REDUCE_PATH_*_TICKS` above sums BOTH
// populations `cpu::reduce_is_gemm_shaped` distinguishes -- the 96
// GEMM-shaped `MatMul` folds AND the 74 small single-operand reduces
// (LayerNorm mean/variance, softmax max/sum), which structurally can
// still land in `Path::WidthFast`/`Path::DotFast` (a `Unary` body can
// pass `body_shape_is_affine_fast_path` too) even though neither
// `width_tile_plan` nor `neon_tile_plan` ever accepts a non-`Binary`
// body -- so the all-reduce split alone cannot answer "of the 96
// MatMuls, how many actually took each route". These four pairs are the
// SAME four `run_reduce` return points as `record_reduce_path_ticks`,
// gated additionally on `cpu::reduce_is_gemm_shaped(resolved)`, so they
// are a pure ADDITIVE detail (never a replacement) the same way
// `EPILOGUE_PROFILE_REDUCE_GEMM_*` sits beside `EPILOGUE_PROFILE_REDUCE_*`
// in `cpu.rs`.
pub static REDUCE_GEMM_PATH_DOT_FAST_CALLS: Counter =
    Counter::new("proxima_tensor.reduce_gemm_path.dot_fast_calls");
pub static REDUCE_GEMM_PATH_DOT_FAST_TICKS: Counter =
    Counter::new("proxima_tensor.reduce_gemm_path.dot_fast_ticks");
pub static REDUCE_GEMM_PATH_WIDTH_FAST_CALLS: Counter =
    Counter::new("proxima_tensor.reduce_gemm_path.width_fast_calls");
pub static REDUCE_GEMM_PATH_WIDTH_FAST_TICKS: Counter =
    Counter::new("proxima_tensor.reduce_gemm_path.width_fast_ticks");
pub static REDUCE_GEMM_PATH_CONV_TILE_CALLS: Counter =
    Counter::new("proxima_tensor.reduce_gemm_path.conv_tile_calls");
pub static REDUCE_GEMM_PATH_CONV_TILE_TICKS: Counter =
    Counter::new("proxima_tensor.reduce_gemm_path.conv_tile_ticks");
pub static REDUCE_GEMM_PATH_GENERIC_CALLS: Counter =
    Counter::new("proxima_tensor.reduce_gemm_path.generic_calls");
pub static REDUCE_GEMM_PATH_GENERIC_TICKS: Counter =
    Counter::new("proxima_tensor.reduce_gemm_path.generic_ticks");

/// Records one `run_reduce` call's elapsed ticks against the path it took,
/// restricted to gemm-shaped (two-distinct-operand) reduce folds -- called
/// once per call, alongside [`record_reduce_path_ticks`], only when the
/// caller has already established `cpu::reduce_is_gemm_shaped(resolved)`.
pub fn record_reduce_gemm_path_ticks(path: Path, ticks: u64) {
    match path {
        Path::DotFast => {
            counter!(REDUCE_GEMM_PATH_DOT_FAST_CALLS, 1);
            counter!(REDUCE_GEMM_PATH_DOT_FAST_TICKS, ticks);
        }
        Path::WidthFast => {
            counter!(REDUCE_GEMM_PATH_WIDTH_FAST_CALLS, 1);
            counter!(REDUCE_GEMM_PATH_WIDTH_FAST_TICKS, ticks);
        }
        Path::ConvTile => {
            counter!(REDUCE_GEMM_PATH_CONV_TILE_CALLS, 1);
            counter!(REDUCE_GEMM_PATH_CONV_TILE_TICKS, ticks);
        }
        Path::Generic => {
            counter!(REDUCE_GEMM_PATH_GENERIC_CALLS, 1);
            counter!(REDUCE_GEMM_PATH_GENERIC_TICKS, ticks);
        }
    }
}

/// Snapshot of the eight gemm-restricted route-census counters:
/// `(dot_fast_calls, dot_fast_ticks, width_fast_calls, width_fast_ticks,
/// conv_tile_calls, conv_tile_ticks, generic_calls, generic_ticks)`.
#[must_use]
pub fn reduce_gemm_path_totals() -> (u64, u64, u64, u64, u64, u64, u64, u64) {
    (
        REDUCE_GEMM_PATH_DOT_FAST_CALLS.get(),
        REDUCE_GEMM_PATH_DOT_FAST_TICKS.get(),
        REDUCE_GEMM_PATH_WIDTH_FAST_CALLS.get(),
        REDUCE_GEMM_PATH_WIDTH_FAST_TICKS.get(),
        REDUCE_GEMM_PATH_CONV_TILE_CALLS.get(),
        REDUCE_GEMM_PATH_CONV_TILE_TICKS.get(),
        REDUCE_GEMM_PATH_GENERIC_CALLS.get(),
        REDUCE_GEMM_PATH_GENERIC_TICKS.get(),
    )
}

/// Resets the eight gemm-restricted route-census counters to zero.
pub fn reset_reduce_gemm_path() {
    let _ = REDUCE_GEMM_PATH_DOT_FAST_CALLS.snapshot_and_reset();
    let _ = REDUCE_GEMM_PATH_DOT_FAST_TICKS.snapshot_and_reset();
    let _ = REDUCE_GEMM_PATH_WIDTH_FAST_CALLS.snapshot_and_reset();
    let _ = REDUCE_GEMM_PATH_WIDTH_FAST_TICKS.snapshot_and_reset();
    let _ = REDUCE_GEMM_PATH_CONV_TILE_CALLS.snapshot_and_reset();
    let _ = REDUCE_GEMM_PATH_CONV_TILE_TICKS.snapshot_and_reset();
    let _ = REDUCE_GEMM_PATH_GENERIC_CALLS.snapshot_and_reset();
    let _ = REDUCE_GEMM_PATH_GENERIC_TICKS.snapshot_and_reset();
}

// composition-split task (2026-09-01): closes ROW 213's named residual
// (`bge_route_census.rs`'s own "H2/H3 note") -- `REDUCE_GEMM_PATH_WIDTH_FAST_TICKS`
// above times the WHOLE `run_reduce` call for a width-tile-routed node, never
// isolating ns strictly inside `gemm_width_tile_neon` from the address
// computation / column tail / row-remainder dispatch / output store that
// surrounds it inside `run_width_tile_neon`, or from the plan-resolution /
// gate-check overhead in `run_reduce` outside `run_width_tile_neon` entirely.
// `WIDTH_TILE_KERNEL_TICKS` sums ticks read at the CALL boundary around every
// `gemm_width_tile_neon` invocation (main tile loop and the row-remainder
// macro, both), never inside the kernel's own k-loop -- a read pair around a
// ~1-2us kernel call is cheap relative to the call; a read pair inside the
// per-k inner loop would dominate it (see this module's own overhead
// measurement, `record_width_tile_split_ticks`'s doc). `WIDTH_TILE_FN_TICKS`
// times `run_width_tile_neon` entry-to-exit, so `fn_ticks - kernel_ticks` is
// the surround, and `REDUCE_GEMM_PATH_WIDTH_FAST_TICKS - fn_ticks` is
// `run_reduce`'s own overhead outside `run_width_tile_neon` entirely.
pub static WIDTH_TILE_KERNEL_TICKS: Counter =
    Counter::new("proxima_tensor.width_tile.kernel_ticks");
pub static WIDTH_TILE_FN_TICKS: Counter = Counter::new("proxima_tensor.width_tile.fn_ticks");
pub static WIDTH_TILE_FN_CALLS: Counter = Counter::new("proxima_tensor.width_tile.fn_calls");
/// MACs computed strictly by `gemm_width_tile_neon` invocations (main tile
/// plus row-remainder tiles), `ROWS * tile_cols * reduction_total` per call,
/// summed the same way `run_reduce`'s own `counters.mac_ops` tally does --
/// but tracked separately here, inside `run_width_tile_neon` where
/// `plan.reduction_total` is read directly, so this total never mixes in the
/// column-tail scalar fallback's own MACs the way `run_reduce`'s aggregate
/// `MAC_OPS` counter does (that counter adds `fallback_delta *
/// reduction_total` too, since the fallback cell is still a real MAC, just
/// not a kernel one).
pub static WIDTH_TILE_KERNEL_MACS: Counter = Counter::new("proxima_tensor.width_tile.kernel_macs");

/// Records one `run_width_tile_neon` call's split: `kernel_ticks` is the sum
/// of every `gemm_width_tile_neon` call-boundary pair taken inside it (main
/// tile plus row-remainder tiles), `kernel_macs` is the MACs those same
/// calls computed, `fn_ticks` is the whole function's own entry-to-exit
/// elapsed ticks. Called once per `run_width_tile_neon` call, from its own
/// tail, mirroring [`record_reduce_gemm_path_ticks`]'s once-per-call commit
/// shape.
pub fn record_width_tile_split_ticks(kernel_ticks: u64, kernel_macs: u64, fn_ticks: u64) {
    counter!(WIDTH_TILE_KERNEL_TICKS, kernel_ticks);
    counter!(WIDTH_TILE_KERNEL_MACS, kernel_macs);
    counter!(WIDTH_TILE_FN_TICKS, fn_ticks);
    counter!(WIDTH_TILE_FN_CALLS, 1);
}

/// Snapshot of the kernel/macs/fn/calls quadruple: `(kernel_ticks,
/// kernel_macs, fn_ticks, fn_calls)`. `fn_ticks - kernel_ticks` is ticks
/// spent in the rest of `run_width_tile_neon`; pair with
/// [`reduce_gemm_path_totals`]'s own `width_fast_ticks` (whole `run_reduce`)
/// to get the third bucket, ticks in `run_reduce` outside
/// `run_width_tile_neon` entirely.
#[must_use]
pub fn width_tile_split_totals() -> (u64, u64, u64, u64) {
    (
        WIDTH_TILE_KERNEL_TICKS.get(),
        WIDTH_TILE_KERNEL_MACS.get(),
        WIDTH_TILE_FN_TICKS.get(),
        WIDTH_TILE_FN_CALLS.get(),
    )
}

/// Resets the kernel/macs/fn/calls quadruple to zero.
pub fn reset_width_tile_split() {
    let _ = WIDTH_TILE_KERNEL_TICKS.snapshot_and_reset();
    let _ = WIDTH_TILE_KERNEL_MACS.snapshot_and_reset();
    let _ = WIDTH_TILE_FN_TICKS.snapshot_and_reset();
    let _ = WIDTH_TILE_FN_CALLS.snapshot_and_reset();
}

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
    let mut totals = WORKER_BUSY_TICKS
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    match totals
        .iter_mut()
        .find(|(existing, _)| *existing == thread_id)
    {
        Some((_, total)) => *total += ticks,
        None => totals.push((thread_id, ticks)),
    }
}

/// Every worker's accumulated busy time from the most recent parallel
/// region(s) since the last [`reset_worker_busy`] — one entry per distinct
/// thread that claimed at least one chunk. Order is not meaningful.
#[must_use]
pub fn worker_busy_snapshot() -> Vec<u64> {
    let totals = WORKER_BUSY_TICKS
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    totals.iter().map(|(_, ticks)| *ticks).collect()
}

pub fn reset_worker_busy() {
    let mut totals = WORKER_BUSY_TICKS
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
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
//
// the pool is shared by two structurally different workloads --
// `matmul_rows_threaded`'s row-chunk path (`cpu.rs::run_row_chunk`) and
// `claim_and_run`'s elementwise/node-chunk path -- and a bare `u64` cannot
// say which one produced a given nanosecond. `CpuWorkload` is the
// discriminant that keeps them separable at the point they are recorded,
// rather than mixed in one pool and un-mixed downstream (a downstream
// consumer has no way to recover the split once summed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuWorkload {
    /// `matmul_rows_threaded`'s row-chunk dispatch (`cpu.rs::run_row_chunk`).
    MatmulRow,
    /// `claim_and_run`'s elementwise/reduce/scan node-chunk dispatch.
    Elementwise,
}

static WORKER_CPU_NANOS: Mutex<Vec<(ThreadId, CpuWorkload, u64)>> = Mutex::new(Vec::new());

/// This thread's consumed CPU time. Unlike an [`Instant`](std::time::Instant)
/// delta, this does not advance while the thread is off-core.
#[must_use]
pub fn thread_cpu_nanos() -> u64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut now) } != 0 {
        return 0;
    }
    (now.tv_sec as u64) * 1_000_000_000 + (now.tv_nsec as u64)
}

/// Adds `nanos` of consumed CPU time to the current thread's running total
/// for `workload`, the deschedule-immune peer of [`record_worker_busy_ticks`].
pub fn record_worker_cpu_nanos(workload: CpuWorkload, nanos: u64) {
    let thread_id = std::thread::current().id();
    let mut totals = WORKER_CPU_NANOS
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    match totals
        .iter_mut()
        .find(|(existing_thread, existing_workload, _)| {
            *existing_thread == thread_id && *existing_workload == workload
        }) {
        Some((_, _, total)) => *total += nanos,
        None => totals.push((thread_id, workload, nanos)),
    }
}

/// Every worker's accumulated CPU time across BOTH workloads -- the sum a
/// caller wants when it does not need the matmul/elementwise split, kept
/// available alongside [`worker_cpu_snapshot_for`] rather than forcing every
/// existing all-workload consumer to add the two split snapshots back
/// together itself.
#[must_use]
pub fn worker_cpu_snapshot() -> Vec<u64> {
    let totals = WORKER_CPU_NANOS
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    totals.iter().map(|(_, _, nanos)| *nanos).collect()
}

/// Every worker's accumulated CPU time for `workload` alone -- the split a
/// caller needs to divide by a workload-specific denominator (e.g. matmul
/// macs) without elementwise CPU time inflating the numerator.
#[must_use]
pub fn worker_cpu_snapshot_for(workload: CpuWorkload) -> Vec<u64> {
    let totals = WORKER_CPU_NANOS
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    totals
        .iter()
        .filter(|(_, existing_workload, _)| *existing_workload == workload)
        .map(|(_, _, nanos)| *nanos)
        .collect()
}

pub fn reset_worker_cpu() {
    let mut totals = WORKER_CPU_NANOS
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
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
    let mut buckets = Q4K_SHAPE_TICKS
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
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
    let buckets = Q4K_SHAPE_TICKS
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    buckets
        .iter()
        .map(|(&(rows, k), &(calls, macs, ticks))| (rows, k, calls, macs, ticks))
        .collect()
}

pub fn reset_q4k_shape_buckets() {
    let mut buckets = Q4K_SHAPE_TICKS
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    buckets.clear();
}

// width-gate-decline task (2026-09-01): `width_tile_plan` (`cpu.rs`) has
// eight `return None` points, and the `Path::WidthFast` label
// (`record_reduce_gemm_path_ticks`) commits identically whether a node's
// `None` sent it through the untiled per-element scalar loop at
// `run_reduce`'s tail or whether it never got that far at all -- the label
// alone cannot name which of the 96 BGE `MatMul` folds declined, or why.
// Keyed by `(NodeId, reason)` the same shape `Q4K_SHAPE_TICKS` above uses
// for `(rows, k)`: one node structurally hits the same condition on every
// call (the gate is a function of the node's fixed shape/layout, not of
// per-call data), so `calls` is a witness the decline is not a one-off, and
// the shape/stride fields are the first-observed values, not an average.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WidthDeclineReason {
    /// `!FUSED_MULTIPLY_ADD || reduce_op != ScalarOp::Add`.
    NoFusedMultiplyAdd,
    /// The fused body is not `BodyShape::Binary(ScalarOp::Multiply, ..)`.
    NotMultiplyAddBody,
    /// `init` is `ReduceInit::FirstElement` (unseeded).
    FirstElementInit,
    /// `leading_output_axes.len() != 1 || reduction_dims.len() != 1`.
    AxesShape,
    /// `width < WIDTH_TILE_VECS * 4`.
    NarrowWidth,
    /// `last_output_dim` is `None`.
    NoOutputDim,
    /// Either operand carries a gather (`IndexMap` with a computed index).
    Gathered,
    /// Neither operand pairs `(width-stride 0, width-stride 1)` — the tile
    /// needs exactly one operand row-broadcast and the other column-major
    /// over the width dim; any other stride pairing declines here.
    StrideLayout,
}

/// `calls`, then the first-observed `(m, k, n, stride_a, stride_b)` at the
/// point of decline — `-1` for any field not yet resolvable when that
/// particular condition fires (e.g. `AxesShape` fires before `m`/`k` can be
/// read off `leading_output_axes[0]`/`reduction_dims[0]`, since those are
/// exactly the indices that condition rejects).
pub type WidthDeclineTotals = (u64, i64, i64, i64, i64, i64);

/// One [`width_tile_decline_snapshot`] row: `(node, reason, calls, m, k, n,
/// stride_a, stride_b)` — factored out purely to clear clippy's
/// `type_complexity` lint on the `Vec` return type, not a new domain concept.
pub type WidthDeclineRow = (u32, WidthDeclineReason, u64, i64, i64, i64, i64, i64);

static WIDTH_TILE_DECLINE: Mutex<BTreeMap<(u32, WidthDeclineReason), WidthDeclineTotals>> =
    Mutex::new(BTreeMap::new());

/// Records one `width_tile_plan` decline for `node`, first-observed shape
/// `(m, k, n)` and operand strides `(stride_a, stride_b)` (`-1` where not
/// resolvable at that decline point, see [`WidthDeclineTotals`]).
pub fn record_width_tile_decline(
    node: NodeId,
    reason: WidthDeclineReason,
    m: i64,
    k: i64,
    n: i64,
    stride_a: i64,
    stride_b: i64,
) {
    let mut buckets = WIDTH_TILE_DECLINE
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let entry = buckets
        .entry((node.0, reason))
        .or_insert((0, m, k, n, stride_a, stride_b));
    entry.0 += 1;
}

/// Every distinct `(NodeId, reason)` decline recorded since the last
/// [`reset_width_tile_decline`], as `(node, reason, calls, m, k, n,
/// stride_a, stride_b)` — sorted by key (`BTreeMap` iteration order).
#[must_use]
pub fn width_tile_decline_snapshot() -> Vec<WidthDeclineRow> {
    let buckets = WIDTH_TILE_DECLINE
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    buckets
        .iter()
        .map(|(&(node, reason), &(calls, m, k, n, stride_a, stride_b))| {
            (node, reason, calls, m, k, n, stride_a, stride_b)
        })
        .collect()
}

pub fn reset_width_tile_decline() {
    let mut buckets = WIDTH_TILE_DECLINE
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
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
    pub staged_round_ticks: u64,
    pub staged_transpose_ticks: u64,
    pub staged_macs: u64,
    pub staged_nodes: u64,
    pub staged_quantize_ticks: u64,
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
        staged_round_ticks: STAGED_MATMUL_ROUND_TICKS.get(),
        staged_transpose_ticks: STAGED_MATMUL_TRANSPOSE_TICKS.get(),
        staged_macs: STAGED_MATMUL_MACS.get(),
        staged_nodes: STAGED_MATMUL_NODES.get(),
        staged_quantize_ticks: STAGED_MATMUL_QUANTIZE_TICKS.get(),
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
    let _ = STAGED_MATMUL_ROUND_TICKS.snapshot_and_reset();
    let _ = STAGED_MATMUL_TRANSPOSE_TICKS.snapshot_and_reset();
    let _ = STAGED_MATMUL_MACS.snapshot_and_reset();
    let _ = STAGED_MATMUL_NODES.snapshot_and_reset();
    let _ = STAGED_MATMUL_QUANTIZE_TICKS.snapshot_and_reset();
    let _ = MATMUL_CHUNKS_CREATED.snapshot_and_reset();
    let _ = MATMUL_COHORT_DISPATCH_CALLS.snapshot_and_reset();
    let _ = MATMUL_POOL_CLAIM_ATTEMPTS.snapshot_and_reset();
    let _ = MATMUL_CHUNK_RUNS.snapshot_and_reset();
    MATMUL_CHUNKS_PER_DISPATCH_MIN.store(u64::MAX, Ordering::Relaxed);
    MATMUL_CHUNKS_PER_DISPATCH_MAX.store(0, Ordering::Relaxed);
    for bucket in &MATMUL_CHUNKS_PER_DISPATCH_HISTOGRAM {
        bucket.store(0, Ordering::Relaxed);
    }
}

// Answers this task's own question -- "total chunks created", "total
// Receiver::recv waits", "total chunk claims off the atomic cursor", and the
// chunks-per-dispatch distribution -- none of which the counters above
// capture on their own: `MATMUL_DISPATCH_CALLS`/`MATMUL_RECV_WAIT_TICKS` only
// ever fire from `matmul_rows_threaded`'s pool-path branch (`session ==
// None`), never its `CohortSession` branch (`session == Some`, the one a
// forward pass actually enters via `nest_cohort().enter()` in `cpu.rs`), so a
// caller reading only those two could not tell "the cohort path never blocks
// on a channel" apart from "nothing dispatched at all". `MATMUL_CHUNKS_CREATED`
// fires from BOTH branches (recorded once `chunk_ranges_len` is known, before
// the branch), and `MATMUL_COHORT_DISPATCH_CALLS` is the cohort branch's own
// per-dispatch counter, the direct peer of `MATMUL_DISPATCH_CALLS` for the
// pool branch.
pub static MATMUL_CHUNKS_CREATED: Counter = Counter::new("proxima_tensor.matmul.chunks_created");
pub static MATMUL_COHORT_DISPATCH_CALLS: Counter =
    Counter::new("proxima_tensor.matmul.cohort_dispatch_calls");
// every `next_index.fetch_add` inside `claim_and_run_rows` (`cpu.rs`), pool
// path only -- includes the one exhausted claim each puller (the calling
// thread and every spawned worker) makes when it observes `index >=
// chunk_ranges.len()` and returns, so this is always
// `MATMUL_CHUNKS_CREATED`'s pool-path share plus one exhausted claim per
// puller, never equal to it.
pub static MATMUL_POOL_CLAIM_ATTEMPTS: Counter =
    Counter::new("proxima_tensor.matmul.pool_claim_attempts");
// `run_row_chunk` (`cpu.rs`) is the one place both the pool path
// (`claim_and_run_rows`) and the cohort path (`RowRound::run_chunk`) land
// after a claim succeeds -- so this is the path-agnostic "a chunk actually
// ran" count, always equal to `MATMUL_CHUNKS_CREATED` by construction (every
// created chunk is claimed and run exactly once); recording both is the
// cross-check that construction invariant actually holds in situ.
pub static MATMUL_CHUNK_RUNS: Counter = Counter::new("proxima_tensor.matmul.chunk_runs");

const MATMUL_CHUNKS_HISTOGRAM_BUCKETS: usize = 128;
pub static MATMUL_CHUNKS_PER_DISPATCH_MIN: AtomicU64 = AtomicU64::new(u64::MAX);
pub static MATMUL_CHUNKS_PER_DISPATCH_MAX: AtomicU64 = AtomicU64::new(0);
// one bucket per chunk-count value (clamped at the top bucket past
// `MATMUL_CHUNKS_HISTOGRAM_BUCKETS - 1`, comfortably above the largest
// legal chunk count `row_chunk_count` can produce -- `workers *
// ROW_OVERSUBSCRIBE` with `ROW_OVERSUBSCRIBE = 4` -- on any host this crate
// targets) -- lets [`chunks_per_dispatch_median`] recover an exact median
// without storing every individual dispatch's chunk count in an unbounded
// `Vec`.
pub static MATMUL_CHUNKS_PER_DISPATCH_HISTOGRAM: [AtomicU64; MATMUL_CHUNKS_HISTOGRAM_BUCKETS] =
    [const { AtomicU64::new(0) }; MATMUL_CHUNKS_HISTOGRAM_BUCKETS];

/// Records one `matmul_rows_threaded` call's `chunk_ranges_len` into the
/// sum/min/max/histogram quartet -- called once per dispatch, from BOTH the
/// cohort and pool branches, before either branch's own timer chain starts.
pub fn record_chunks_created(chunk_count: usize) {
    let chunk_count_u64 = chunk_count as u64;
    counter!(MATMUL_CHUNKS_CREATED, chunk_count_u64);
    MATMUL_CHUNKS_PER_DISPATCH_MIN.fetch_min(chunk_count_u64, Ordering::Relaxed);
    MATMUL_CHUNKS_PER_DISPATCH_MAX.fetch_max(chunk_count_u64, Ordering::Relaxed);
    let bucket = chunk_count.min(MATMUL_CHUNKS_HISTOGRAM_BUCKETS - 1);
    MATMUL_CHUNKS_PER_DISPATCH_HISTOGRAM[bucket].fetch_add(1, Ordering::Relaxed);
}

/// The exact median of every `chunk_count` recorded via
/// [`record_chunks_created`] since the last reset, given the caller's own
/// total dispatch count (`MATMUL_DISPATCH_CALLS.get() +
/// MATMUL_COHORT_DISPATCH_CALLS.get()`) as the histogram's total mass --
/// walks the fixed-size histogram instead of sorting a stored `Vec`, the same
/// bounded-memory trade [`record_chunks_created`]'s own doc explains. Returns
/// 0 when `dispatch_count` is 0 (nothing recorded).
#[must_use]
pub fn chunks_per_dispatch_median(dispatch_count: u64) -> u64 {
    if dispatch_count == 0 {
        return 0;
    }
    let target = dispatch_count / 2;
    let mut cumulative = 0u64;
    for (bucket, count) in MATMUL_CHUNKS_PER_DISPATCH_HISTOGRAM.iter().enumerate() {
        cumulative += count.load(Ordering::Relaxed);
        if cumulative > target {
            return bucket as u64;
        }
    }
    (MATMUL_CHUNKS_HISTOGRAM_BUCKETS - 1) as u64
}

/// One process run's worth of the chunk-count-and-claims witness above, read
/// back the same way [`matmul_dispatch_totals`] is.
#[derive(Debug, Clone, Copy, Default)]
pub struct MatmulChunkTotals {
    pub chunks_created: u64,
    pub chunks_per_dispatch_min: u64,
    pub chunks_per_dispatch_max: u64,
    pub chunks_per_dispatch_median: u64,
    pub cohort_dispatch_calls: u64,
    pub pool_claim_attempts: u64,
    pub chunk_runs: u64,
}

#[must_use]
pub fn matmul_chunk_totals() -> MatmulChunkTotals {
    let chunks_created = MATMUL_CHUNKS_CREATED.get();
    let dispatch_count = MATMUL_DISPATCH_CALLS.get() + MATMUL_COHORT_DISPATCH_CALLS.get();
    let observed_min = MATMUL_CHUNKS_PER_DISPATCH_MIN.load(Ordering::Relaxed);
    MatmulChunkTotals {
        chunks_created,
        chunks_per_dispatch_min: if dispatch_count == 0 { 0 } else { observed_min },
        chunks_per_dispatch_max: MATMUL_CHUNKS_PER_DISPATCH_MAX.load(Ordering::Relaxed),
        chunks_per_dispatch_median: chunks_per_dispatch_median(dispatch_count),
        cohort_dispatch_calls: MATMUL_COHORT_DISPATCH_CALLS.get(),
        pool_claim_attempts: MATMUL_POOL_CLAIM_ATTEMPTS.get(),
        chunk_runs: MATMUL_CHUNK_RUNS.get(),
    }
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
pub static SERIAL_SLICE_CARVE_TICKS: Counter =
    Counter::new("proxima_tensor.serial_slice_carve_ticks");
pub static SERIAL_FINISH_TICKS: Counter = Counter::new("proxima_tensor.serial_finish_ticks");
pub static SERIAL_BOOKKEEPING_TICKS: Counter =
    Counter::new("proxima_tensor.serial_bookkeeping_ticks");
// only nonzero on the `workers == 1` (or below-threshold) arm, where
// `evaluate_node_parallel` never reaches `run_chunks_threaded` at all.
pub static SERIAL_SEQUENTIAL_COMPUTE_TICKS: Counter =
    Counter::new("proxima_tensor.serial_sequential_compute_ticks");
pub static SERIAL_EVALUATE_PARALLEL_TICKS: Counter =
    Counter::new("proxima_tensor.serial_evaluate_parallel_ticks");
pub static SERIAL_EVALUATE_PARALLEL_CALLS: Counter =
    Counter::new("proxima_tensor.serial_evaluate_parallel_calls");

// `run_elementwise_range`'s own fixed-per-call breakdown, split at the same
// three seams the decode-speed investigation measured against: everything
// before `step_values` is carved (operand span resolution, stride/gather
// scratch), the `step_values` allocation itself (sized for the `Generic`
// fused-body table even when the node's body is `Unary`/`Binary` and never
// reads it), and the position loop that follows. Committed once per node
// call, never per element or per position.
pub static ELEMENTWISE_SETUP_TICKS: Counter =
    Counter::new("proxima_tensor.elementwise_setup_ticks");
pub static ELEMENTWISE_STEP_VALUES_TICKS: Counter =
    Counter::new("proxima_tensor.elementwise_step_values_ticks");
pub static ELEMENTWISE_LOOP_TICKS: Counter = Counter::new("proxima_tensor.elementwise_loop_ticks");
pub static ELEMENTWISE_RANGE_CALLS: Counter =
    Counter::new("proxima_tensor.elementwise_range_calls");
// ROW 178 row-flattening mechanism check: how many `run_elementwise_range`
// calls collapsed their whole outer-row odometer into one
// `elementwise_width_fast` call, and the total row count those calls
// covered -- `hits / ELEMENTWISE_RANGE_CALLS` and `rows / hits` (mean rows
// collapsed per hit) are the two numbers that confirm or refute engagement.
pub static ELEMENTWISE_FLAT_RANGE_HITS: Counter =
    Counter::new("proxima_tensor.elementwise_flat_range_hits");
pub static ELEMENTWISE_FLAT_RANGE_ROWS: Counter =
    Counter::new("proxima_tensor.elementwise_flat_range_rows");
// `run_elementwise_dispatch`'s own cohort-round count -- how many of a
// forward pass's elementwise nodes actually open a `CohortSession::run`
// round (as opposed to falling straight through to the sequential
// `run_elementwise` because `outer_len < 2`, `workers <= 1`, or the node is
// below `PARALLEL_THRESHOLD`).
pub static ELEMENTWISE_COHORT_ROUNDS: Counter =
    Counter::new("proxima_tensor.elementwise_cohort_rounds");

/// Snapshot of `run_elementwise_range`'s own fixed-per-call breakdown --
/// `(calls, setup_ticks, step_values_ticks, loop_ticks, cohort_rounds)`.
/// `.get()`-not-reset, same shape [`path_totals`] uses; pair with
/// [`reset_elementwise_phase`] for a per-run delta.
#[must_use]
pub fn elementwise_phase_totals() -> (u64, u64, u64, u64, u64) {
    (
        ELEMENTWISE_RANGE_CALLS.get(),
        ELEMENTWISE_SETUP_TICKS.get(),
        ELEMENTWISE_STEP_VALUES_TICKS.get(),
        ELEMENTWISE_LOOP_TICKS.get(),
        ELEMENTWISE_COHORT_ROUNDS.get(),
    )
}

/// Resets `run_elementwise_range`'s own phase-breakdown counters to zero.
pub fn reset_elementwise_phase() {
    let _ = ELEMENTWISE_RANGE_CALLS.snapshot_and_reset();
    let _ = ELEMENTWISE_SETUP_TICKS.snapshot_and_reset();
    let _ = ELEMENTWISE_STEP_VALUES_TICKS.snapshot_and_reset();
    let _ = ELEMENTWISE_LOOP_TICKS.snapshot_and_reset();
    let _ = ELEMENTWISE_COHORT_ROUNDS.snapshot_and_reset();
}

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
        AllocSite::OutputBuffer => (
            &ALLOC_SITE_OUTPUT_BUFFER_COUNT,
            &ALLOC_SITE_OUTPUT_BUFFER_BYTES,
        ),
        AllocSite::ChunkSlices => (
            &ALLOC_SITE_CHUNK_SLICES_COUNT,
            &ALLOC_SITE_CHUNK_SLICES_BYTES,
        ),
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
    ConvTile,
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
            Path::ConvTile => counter!(PATH_CONV_TILE, 1),
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
    pub path_conv_tile: u64,
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
        path_conv_tile: PATH_CONV_TILE.get(),
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
    let _ = PATH_CONV_TILE.snapshot_and_reset();
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
pub fn record_operand_access(
    node: NodeId,
    reads: u64,
    distinct_elements: u64,
    total_elements: u64,
) {
    let bytes = reads * size_of::<f32>() as u64;
    let mut table = OPERAND_ACCESS
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
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
    let table = OPERAND_ACCESS
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
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
    let table = OPERAND_ACCESS
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    table.get(&node).copied()
}

pub fn reset_operand_access() {
    let mut table = OPERAND_ACCESS
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    table.clear();
    let mut rows = GATHER_ROWS_TOUCHED
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
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
    let mut rows = GATHER_ROWS_TOUCHED
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
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
    let rows = GATHER_ROWS_TOUCHED
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    rows.get(&node).map_or(0, HashSet::len) as u64
}

/// Total wall ticks the leader spends transposing `matmul_q4k_q8k_f32_impl`'s
/// row-major `[row][position]` result back into the position-major layout
/// `run_reduce_quantized`'s callers consume (`cpu.rs`'s copy loop after the
/// wide fold). Leader-serial: it is not inside any cohort round, so no
/// dispatch counter sees it.
pub static MATMUL_Q4K_TRANSPOSE_TICKS: Counter =
    Counter::new("proxima_tensor.matmul.q4k_transpose_ticks");

// `cpu::run_staged_batch`'s own coverage of the matmul dispatch story
// above: `run_reduce_quantized`'s counters (this whole section) only ever
// see the unbatched matmul population (`node_kind=reduce_matmul_quantized`,
// ROW98's own 65/225 per step) -- `MatmulStagePlan`'s dot calls run inside
// a `StagedRound` chunk closure and never call `run_reduce_quantized` at
// all, so before this counter existed the 160/225 folded matmul nodes
// (ROW97, the DOMINANT node_kind bucket) had zero sub-attribution: only
// `node_kind=staged_batch`'s own outer wall time, no split of quantize vs
// kernel vs transpose inside it. `build_matmul_stage_plan`'s own quantize
// call reuses `MATMUL_QUANTIZE_ACTIVATION_TICKS` directly (same semantic,
// same call shape, just a second call site) so that counter's own meaning
// stays "activation-quantize time across every matmul node this process
// ran," not "...across only the unbatched ones." Round and transpose get
// their own counters below rather than folding into the unbatched-only
// `MATMUL_OWN_CHUNK_TICKS`/`MATMUL_Q4K_TRANSPOSE_TICKS` (Q4K-only by name)
// because a staged round always contains only quantized-matmul stages
// (ROW97's landed `is_staged_batch_eligible` restriction) but may mix
// Q4K/Q5K/Q6K within one run, so a codec-specific name would misrepresent
// what is inside it.
pub static STAGED_MATMUL_ROUND_TICKS: Counter =
    Counter::new("proxima_tensor.matmul.staged_round_ticks");
pub static STAGED_MATMUL_TRANSPOSE_TICKS: Counter =
    Counter::new("proxima_tensor.matmul.staged_transpose_ticks");
pub static STAGED_MATMUL_MACS: Counter = Counter::new("proxima_tensor.matmul.staged_macs");
pub static STAGED_MATMUL_NODES: Counter = Counter::new("proxima_tensor.matmul.staged_nodes");
// deliberately its OWN counter, not a second call site into
// `MATMUL_QUANTIZE_ACTIVATION_TICKS` -- an earlier version of this
// instrumentation shared that counter across both call sites, and it broke
// `matmul_split`'s own arithmetic: `bucket_ms` (`MATMUL_REDUCE_QUANTIZED_TICKS`)
// is scoped to the unbatched population only, so a quantize counter mixing
// in the staged population's calls no longer nests inside it, and
// `quantize_ms + kernel_ms + transpose_ms` stopped summing anywhere near
// `bucket_ms`. Keeping the two call sites' ticks in separate counters keeps
// each of `matmul_split`'s own fields internally consistent (nested subsets
// of that same line's own `bucket_ms`) and makes `matmul_split_staged`'s
// quantize time an honest, separately-attributed figure instead of a silent
// reinterpretation of what `quantize_activation_ms` used to mean.
pub static STAGED_MATMUL_QUANTIZE_TICKS: Counter =
    Counter::new("proxima_tensor.matmul.staged_quantize_ticks");

// discipline.md ROW 140's own hypothesis check: does the SAME activation
// node (e.g. the post-attention-norm vector feeding `attn_q`/`attn_k`/
// `attn_v`, or the post-FFN-norm vector feeding `ffn_gate`/`ffn_up`) get
// re-quantized to Q8_K once per CONSUMING matmul node, rather than once
// per DISTINCT activation? Keyed by `activation_node` (never `resolved.node`,
// the matmul node itself -- the whole point is to see several different
// matmul nodes collapse onto the same key), incremented once per call to
// `cpu::build_matmul_stage_plan`/`cpu::run_reduce_quantized`, the two call
// sites that each independently quantize their own `activation` operand
// before this counter existed. `total_calls` (`.iter().sum()`) vs
// `distinct_nodes` (count of nonzero entries) is the ratio the hypothesis
// lives or dies on: 1:1 kills it, >1:1 by roughly the fan-out (3 for QKV, 2
// for gate/up) confirms it.
//
// ROW 141: this was a `Mutex<BTreeMap<NodeId, u64>>` -- a per-call
// `O(log n)` key comparison chain PLUS a mutex acquisition, on the exact
// path ROW 140 measured a `-1.104 ms` residual delta from. `NodeId` is a
// position in the program's own flat `Vec<Op>` (`op.rs`'s own doc, and the
// same fact `cpu.rs`'s `staged_quantize_cache` now exploits directly), so a
// `Vec<u64>` indexed by `node.0` is the same O(1)-slot shape, grown once to
// cover the largest node id this process has seen (never shrunk on
// [`reset_step`] -- only zeroed -- so steady-state decode pays no further
// resize after the first step touches every node this checkpoint's graph
// has). The `Mutex` itself stays: both call sites run on the leader thread
// only, serially, strictly before any cohort round opens (`cpu.rs`'s own
// doc on why `Some(session)` is safe there), so there is never real
// contention -- but a bare `static mut`/`UnsafeCell` would need an unsafe
// single-writer invariant this instrument-only diagnostic has no
// reason to take on.
static QUANTIZE_ACTIVATION_CALLS_BY_NODE: Mutex<Vec<u64>> = Mutex::new(Vec::new());

/// Records one activation-quantize call against `activation_node` — called
/// from `cpu::build_matmul_stage_plan` (the staged-batch path) and
/// `cpu::run_reduce_quantized` (the unbatched path), both BEFORE this
/// counter existed had no way to tell "quantized once, reused three times"
/// apart from "quantized three times, once per consumer".
pub fn record_quantize_activation_call(activation_node: NodeId) {
    let mut calls = QUANTIZE_ACTIVATION_CALLS_BY_NODE
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let index = activation_node.0 as usize;
    if index >= calls.len() {
        calls.resize(index + 1, 0);
    }
    calls[index] += 1;
}

/// `(total_calls, distinct_activation_nodes)` across every
/// [`record_quantize_activation_call`] since the last [`reset_step`] —
/// `total_calls / distinct_activation_nodes` is the redundancy ratio ROW
/// 140's hypothesis names: `1.0` kills it, `> 1.0` confirms real re-quantize
/// fan-out and states its own size.
#[must_use]
pub fn quantize_activation_call_stats() -> (u64, u64) {
    let calls = QUANTIZE_ACTIVATION_CALLS_BY_NODE
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let total_calls: u64 = calls.iter().sum();
    let distinct_nodes = calls.iter().filter(|&&count| count > 0).count() as u64;
    (total_calls, distinct_nodes)
}

fn reset_quantize_activation_calls() {
    let mut calls = QUANTIZE_ACTIVATION_CALLS_BY_NODE
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    calls.iter_mut().for_each(|count| *count = 0);
    let _ = QUANTIZE_ACTIVATION_CACHE_HITS.snapshot_and_reset();
}

/// How many times `cpu::build_matmul_stage_plan`'s own per-step
/// `staged_quantize_cache` (ROW 140's fix) served an already-quantized
/// `Arc<[u8]>` instead of paying `quantize_row_q8k_dispatch` again for the
/// same activation node. Zero on a build that never lands the fix; nonzero
/// is the direct witness that the cache is doing real work, not just
/// compiling.
pub static QUANTIZE_ACTIVATION_CACHE_HITS: Counter =
    Counter::new("proxima_tensor.matmul.quantize_activation_cache_hits");

/// Records one cache hit — called from `cpu::build_matmul_stage_plan` when
/// `activation_node` is already present in this step's own
/// `staged_quantize_cache`.
pub fn record_quantize_activation_cache_hit() {
    counter!(QUANTIZE_ACTIVATION_CACHE_HITS, 1);
}

/// The cohort's own round-level forensics (`prime::os::cohort::diag`),
/// re-exported so a consumer of this crate's `instrument` feature reads
/// park/spin/unpark tallies and per-slot claim latencies without taking its
/// own `prime` dependency.
#[cfg(feature = "tensor-cohort")]
pub use prime::os::cohort::diag as cohort;

/// Per-step-reset attribution of the CALLING (leader) thread's own wall
/// clock inside every `CohortSession::run`/`run_with_completion` call this
/// step made — the instrumentation `docs/discipline.md` ROW 130 named and
/// did not build. Every field is nanoseconds on the calling thread's own
/// timeline, never a sum across the cohort's other worker threads: those
/// run concurrently with the leader, not serially inside its wall clock, so
/// summing their ticks in would not sum to the step's own wall time.
///
/// - `kernel_nanos`: time strictly inside `CohortRound::run_chunk` for
///   chunks the leader itself claimed (`prime::os::cohort::diag::SLOT_KERNEL_NANOS[0]`).
/// - `dispatch_nanos`: round setup (control-block reset, `round` publish,
///   issuing unparks) plus the leader's own claim-loop overhead (the cursor
///   `fetch_add`, the completion check, the `catch_unwind` wrapper) around
///   chunks that were not kernel time — everything the calling thread pays
///   to DISPATCH work, whether to itself or to other members.
/// - `park_spin_wake_nanos`: the calling thread's own wait for chunk
///   completion after its own claim loop ran dry — the tail spin-wait
///   (`prime::os::cohort::diag::LEADER_SPIN_NANOS`) plus the time spent
///   issuing unpark wakeups (`UNPARK_NANOS`, itself a wait-adjacent action:
///   it exists only because something is parked waiting to be told to stop).
#[cfg(feature = "tensor-cohort")]
#[derive(Debug, Clone, Copy, Default)]
pub struct CohortLeaderAttribution {
    pub kernel_nanos: u64,
    pub dispatch_nanos: u64,
    pub park_spin_wake_nanos: u64,
}

/// Reads the leader-thread attribution accumulated since the last
/// [`reset_cohort_attribution`] (or process start, before any reset).
#[cfg(feature = "tensor-cohort")]
#[must_use]
pub fn cohort_leader_attribution() -> CohortLeaderAttribution {
    let leader_kernel = cohort::SLOT_KERNEL_NANOS[0].load(Ordering::Relaxed);
    let leader_compute = cohort::SLOT_COMPUTE_NANOS[0].load(Ordering::Relaxed);
    let leader_claim_overhead = leader_compute.saturating_sub(leader_kernel);
    let setup = cohort::LEADER_SETUP_NANOS.load(Ordering::Relaxed);
    let spin = cohort::LEADER_SPIN_NANOS.load(Ordering::Relaxed);
    let unpark = cohort::UNPARK_NANOS.load(Ordering::Relaxed);
    CohortLeaderAttribution {
        kernel_nanos: leader_kernel,
        dispatch_nanos: setup + leader_claim_overhead,
        park_spin_wake_nanos: spin + unpark,
    }
}

/// Zeroes every cohort diag counter — the leader-attribution fields above
/// plus every field [`cohort`] already carried (rounds/parks/spin
/// hits/per-slot claim latencies). One call so a caller resetting for a new
/// decode step does not have to enumerate the cohort's own internals.
#[cfg(feature = "tensor-cohort")]
pub fn reset_cohort_attribution() {
    cohort::reset();
}

/// Resets every counter this module's `evaluate_ms` decomposition depends
/// on, in one call — matmul dispatch, serial bookkeeping, op-kind dispatch,
/// and (when `tensor-cohort` is enabled) the cohort's own leader
/// attribution. Intended to be called once at the START of a decode step,
/// paired with a read of the same families at the step's end, so a single
/// step's cost is measured directly rather than inferred by differencing
/// two process launches (`docs/discipline.md` ROW 130's own postmortem).
pub fn reset_step() {
    reset_matmul_dispatch();
    reset_serial();
    reset_path();
    reset_parallel();
    reset_quantize_activation_calls();
    #[cfg(feature = "tensor-cohort")]
    reset_cohort_attribution();
}

// achieved-ns/element investigation (nsper task, 2026-08-21): `body_shape`
// (`cpu.rs`'s `BodyShape` enum) is a DIFFERENT axis from `Path`
// (`WidthFast`/`Generic`, the affine-operand fast-path gate) -- a `Generic`
// body can still take the affine fast path, and a `Unary`/`Binary` body can
// still fall to the gather loop. `ELEMENTWISE_LOOP_TICKS`/`OUTPUT_WRITES`
// mix every shape together, so neither can answer "is `Generic` slower per
// element than the monomorphic `Unary`/`Binary` kernel this crate's own
// 0.38ns/element figure (`cpu.rs:2159`) was measured against". These four
// split `run_elementwise_range`'s own loop ticks and elements-written count
// by `shape` alone, committed once per call from the `shape` already in
// scope -- never per element.
pub static ELEMENTWISE_LOOP_TICKS_MONOMORPHIC: Counter =
    Counter::new("proxima_tensor.elementwise_loop_ticks_monomorphic");
pub static ELEMENTWISE_ELEMENTS_MONOMORPHIC: Counter =
    Counter::new("proxima_tensor.elementwise_elements_monomorphic");
pub static ELEMENTWISE_LOOP_TICKS_GENERIC: Counter =
    Counter::new("proxima_tensor.elementwise_loop_ticks_generic");
pub static ELEMENTWISE_ELEMENTS_GENERIC: Counter =
    Counter::new("proxima_tensor.elementwise_elements_generic");

// fast_path-vs-slow-path split within `Generic` (A-vs-B task, 2026-08-21):
// `Generic`'s own 14.9x-slower-than-monomorphic figure mixes two different
// code paths -- `generic_body_is_affine_fast_path` gating whether a call
// takes `elementwise_width_generic` (per-step monomorphic dispatch, no
// per-element `apply_body` interpreter) or falls to the per-element
// `apply_body`/`apply_scalar_op` gather loop. These four split the same
// loop ticks and element count `ELEMENTWISE_LOOP_TICKS_GENERIC`/
// `ELEMENTWISE_ELEMENTS_GENERIC` already carry, by the `fast_path` bool
// already computed once per call at `cpu.rs`'s `run_elementwise_range` --
// never re-derived, never sampled per element.
pub static ELEMENTWISE_LOOP_TICKS_GENERIC_FAST: Counter =
    Counter::new("proxima_tensor.elementwise_loop_ticks_generic_fast");
pub static ELEMENTWISE_ELEMENTS_GENERIC_FAST: Counter =
    Counter::new("proxima_tensor.elementwise_elements_generic_fast");
pub static ELEMENTWISE_LOOP_TICKS_GENERIC_SLOW: Counter =
    Counter::new("proxima_tensor.elementwise_loop_ticks_generic_slow");
pub static ELEMENTWISE_ELEMENTS_GENERIC_SLOW: Counter =
    Counter::new("proxima_tensor.elementwise_elements_generic_slow");

// fast_path-vs-slow-path split within `Unary`/`Binary` (`Monomorphic`)
// (residual-profile task, 2026-08-30): `ELEMENTWISE_LOOP_TICKS_MONOMORPHIC`
// above mixes two different code paths exactly the way the pre-existing
// `Generic` split (above) already separates for the fused-body case — a
// `Binary` body (e.g. Conv's own `window_materialize` multiply, whose image
// operand reads through a strided/dilated `window_axis` pattern) can still
// fail `body_shape_is_affine_fast_path` and fall to the per-element
// gather loop (`elementwise_width_fast`'s `false` arm in
// `run_elementwise_range`) despite being classified `Monomorphic` by
// `BodyShape`. Same split, same commit site, same `fast_path` bool already
// in scope — never re-derived, never sampled per element.
pub static ELEMENTWISE_LOOP_TICKS_MONOMORPHIC_FAST: Counter =
    Counter::new("proxima_tensor.elementwise_loop_ticks_monomorphic_fast");
pub static ELEMENTWISE_ELEMENTS_MONOMORPHIC_FAST: Counter =
    Counter::new("proxima_tensor.elementwise_elements_monomorphic_fast");
pub static ELEMENTWISE_LOOP_TICKS_MONOMORPHIC_SLOW: Counter =
    Counter::new("proxima_tensor.elementwise_loop_ticks_monomorphic_slow");
pub static ELEMENTWISE_ELEMENTS_MONOMORPHIC_SLOW: Counter =
    Counter::new("proxima_tensor.elementwise_elements_monomorphic_slow");

// window-materialize-shaped copy split within `Monomorphic`/`fast_path`
// (rung 2, `docs/discipline.md` ROW 154): a narrower slice of
// `ELEMENTWISE_LOOP_TICKS_MONOMORPHIC_FAST` above — this call's body was a
// bare identity copy AND `run_elementwise_range`'s own block sweep
// (`block_dim`/`block_extent`, ROW 150) was engaged, so every block-aligned
// row took `window_copy_block`'s row-segment copy instead of
// `elementwise_width_fast`'s per-row dispatch. Same commit site, same
// per-call constant (`window_copy_operand`) already in scope — never
// re-derived, never sampled per element.
pub static ELEMENTWISE_LOOP_TICKS_WINDOW_COPY: Counter =
    Counter::new("proxima_tensor.elementwise_loop_ticks_window_copy");
pub static ELEMENTWISE_ELEMENTS_WINDOW_COPY: Counter =
    Counter::new("proxima_tensor.elementwise_elements_window_copy");

/// Snapshot of `BodyShape`/fast-vs-slow-path split ns/element figures'
/// underlying ticks+elements pairs, `.get()`-not-reset: `(monomorphic,
/// generic, monomorphic_fast, monomorphic_slow, generic_fast, generic_slow,
/// window_copy)`, each a `(ticks, elements)` pair. Pair with
/// [`reset_elementwise_bodyshape`] for a per-run delta.
#[must_use]
pub fn elementwise_bodyshape_totals() -> ElementwiseBodyShapeTotals {
    ElementwiseBodyShapeTotals {
        monomorphic: (
            ELEMENTWISE_LOOP_TICKS_MONOMORPHIC.get(),
            ELEMENTWISE_ELEMENTS_MONOMORPHIC.get(),
        ),
        generic: (
            ELEMENTWISE_LOOP_TICKS_GENERIC.get(),
            ELEMENTWISE_ELEMENTS_GENERIC.get(),
        ),
        monomorphic_fast: (
            ELEMENTWISE_LOOP_TICKS_MONOMORPHIC_FAST.get(),
            ELEMENTWISE_ELEMENTS_MONOMORPHIC_FAST.get(),
        ),
        monomorphic_slow: (
            ELEMENTWISE_LOOP_TICKS_MONOMORPHIC_SLOW.get(),
            ELEMENTWISE_ELEMENTS_MONOMORPHIC_SLOW.get(),
        ),
        generic_fast: (
            ELEMENTWISE_LOOP_TICKS_GENERIC_FAST.get(),
            ELEMENTWISE_ELEMENTS_GENERIC_FAST.get(),
        ),
        generic_slow: (
            ELEMENTWISE_LOOP_TICKS_GENERIC_SLOW.get(),
            ELEMENTWISE_ELEMENTS_GENERIC_SLOW.get(),
        ),
        window_copy: (
            ELEMENTWISE_LOOP_TICKS_WINDOW_COPY.get(),
            ELEMENTWISE_ELEMENTS_WINDOW_COPY.get(),
        ),
    }
}

/// [`elementwise_bodyshape_totals`]'s own return shape — named fields over a
/// seven-tuple-of-pairs so a caller's field access reads as English rather
/// than a positional index.
#[derive(Debug, Clone, Copy, Default)]
pub struct ElementwiseBodyShapeTotals {
    pub monomorphic: (u64, u64),
    pub generic: (u64, u64),
    pub monomorphic_fast: (u64, u64),
    pub monomorphic_slow: (u64, u64),
    pub generic_fast: (u64, u64),
    pub generic_slow: (u64, u64),
    pub window_copy: (u64, u64),
}

/// Resets every `BodyShape`/fast-vs-slow-path ticks+elements counter to zero.
pub fn reset_elementwise_bodyshape() {
    let _ = ELEMENTWISE_LOOP_TICKS_MONOMORPHIC.snapshot_and_reset();
    let _ = ELEMENTWISE_ELEMENTS_MONOMORPHIC.snapshot_and_reset();
    let _ = ELEMENTWISE_LOOP_TICKS_GENERIC.snapshot_and_reset();
    let _ = ELEMENTWISE_ELEMENTS_GENERIC.snapshot_and_reset();
    let _ = ELEMENTWISE_LOOP_TICKS_MONOMORPHIC_FAST.snapshot_and_reset();
    let _ = ELEMENTWISE_ELEMENTS_MONOMORPHIC_FAST.snapshot_and_reset();
    let _ = ELEMENTWISE_LOOP_TICKS_MONOMORPHIC_SLOW.snapshot_and_reset();
    let _ = ELEMENTWISE_ELEMENTS_MONOMORPHIC_SLOW.snapshot_and_reset();
    let _ = ELEMENTWISE_LOOP_TICKS_GENERIC_FAST.snapshot_and_reset();
    let _ = ELEMENTWISE_ELEMENTS_GENERIC_FAST.snapshot_and_reset();
    let _ = ELEMENTWISE_LOOP_TICKS_GENERIC_SLOW.snapshot_and_reset();
    let _ = ELEMENTWISE_ELEMENTS_GENERIC_SLOW.snapshot_and_reset();
    let _ = ELEMENTWISE_LOOP_TICKS_WINDOW_COPY.snapshot_and_reset();
    let _ = ELEMENTWISE_ELEMENTS_WINDOW_COPY.snapshot_and_reset();
}

// the Adam-chain dedicated kernel split within `Generic` (`docs/discipline.md`
// ROW 179): `cpu.rs`'s `BodyShape::FusedAdamUpdate` is a structurally
// detected sub-case of what used to be classified `Generic` — the same
// 8-`BodyStep` bias-corrected update chain `optimizer::adam_step` fuses for
// every Adam-updated parameter, now walked by a dedicated register-resident
// kernel (`elementwise_width_fused_adam_update`) instead of
// `elementwise_width_generic`'s step-outer tile loop when
// `fused_adam_update_is_affine_fast_path` holds. `HITS` counts calls (one
// per `run_elementwise_range` invocation that matched AND took the fast
// path — a cohort-parallel node contributes one hit per worker chunk, same
// granularity `ELEMENTWISE_FLAT_RANGE_HITS` already uses), not nodes, so a
// caller reporting "N nodes matched" divides by the per-step call count
// separately. Same commit site, same per-call constants already in scope —
// never re-derived, never sampled per element.
pub static ELEMENTWISE_LOOP_TICKS_FUSED_ADAM: Counter =
    Counter::new("proxima_tensor.elementwise_loop_ticks_fused_adam");
pub static ELEMENTWISE_ELEMENTS_FUSED_ADAM: Counter =
    Counter::new("proxima_tensor.elementwise_elements_fused_adam");
pub static ELEMENTWISE_FUSED_ADAM_HITS: Counter =
    Counter::new("proxima_tensor.elementwise_fused_adam_hits");

// call-size distribution: how many `run_elementwise_range` calls processed
// how many elements, this process run. A `Counter` can only sum, so the
// histogram itself needs a map -- kept as a plain `size -> call_count` table
// rather than per-call log lines (547 calls/decode step would flood stderr).
// Committed once per call, read back and cleared by
// `elementwise_call_size_snapshot_and_reset` the same `snapshot_and_reset`
// shape every other per-step counter here uses.
static ELEMENTWISE_CALL_SIZES: Mutex<BTreeMap<u64, u64>> = Mutex::new(BTreeMap::new());

/// Records one `run_elementwise_range` call's total elements written
/// (`counters.output_writes`, already the exact per-call count the caller
/// computed for its own commit -- never re-derived from extents here).
pub fn record_elementwise_call_size(elements: u64) {
    let mut table = ELEMENTWISE_CALL_SIZES
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    *table.entry(elements).or_insert(0) += 1;
}

/// Reads back every distinct call size seen since the last reset, paired
/// with how many calls landed at that size, and clears the table -- one
/// decode step's distribution per call, matching every other
/// `snapshot_and_reset` in this module.
#[must_use]
pub fn elementwise_call_size_snapshot_and_reset() -> Vec<(u64, u64)> {
    let mut table = ELEMENTWISE_CALL_SIZES
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let sizes: Vec<(u64, u64)> = table.iter().map(|(size, count)| (*size, *count)).collect();
    table.clear();
    sizes
}

// `evaluate_quantized_with_scratch`/`evaluate_quantized_named_with_scratch`'s
// own per-call phase breakdown -- previously a `DIAG` eprintln fired on
// every call (17-30 lines/call under `instrument`, since removed: it
// inverted the sign of two independent measurements this session by adding
// stderr-flush cost to the very loop it was timing). Same shape as
// `reduce_gemm_path_totals` above: plain `Counter`s, read back via
// `evaluate_quantized_phase_totals`, reset via `reset_evaluate_quantized_phase`.
pub static EVALUATE_QUANTIZED_CALLS: Counter =
    Counter::new("proxima_tensor.evaluate_quantized_calls");
pub static EVALUATE_QUANTIZED_RESOLVE_TICKS: Counter =
    Counter::new("proxima_tensor.evaluate_quantized_resolve_ticks");
pub static EVALUATE_QUANTIZED_SETUP_TICKS: Counter =
    Counter::new("proxima_tensor.evaluate_quantized_setup_ticks");
pub static EVALUATE_QUANTIZED_LOOP_OVERHEAD_TICKS: Counter =
    Counter::new("proxima_tensor.evaluate_quantized_loop_overhead_ticks");
pub static EVALUATE_QUANTIZED_FINISH_TICKS: Counter =
    Counter::new("proxima_tensor.evaluate_quantized_finish_ticks");
// buffer occupancy has no running sum that means anything (a peak, not a
// total), so it uses the same fetch_max discipline as
// `PARALLEL_CHUNK_TICKS_MAX` above rather than a `Counter`.
pub static EVALUATE_QUANTIZED_PEAK_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);

/// Records one `evaluate_quantized_with_scratch` call's phase ticks and peak
/// live-buffer byte count. Called once per call, at the same three points the
/// removed `DIAG` prints read from (`setup_ms`, `loop_overhead_ms`,
/// `finish_ms`, `peak_live_bytes`). `evaluate_quantized_named_with_scratch`'s
/// own name-resolution step is a separate call boundary entirely (it runs
/// BEFORE this function is ever entered), so its ticks commit through
/// [`record_evaluate_quantized_resolve`] instead of here.
pub fn record_evaluate_quantized_phase(
    setup_ticks: u64,
    loop_overhead_ticks: u64,
    finish_ticks: u64,
    peak_live_bytes: u64,
) {
    counter!(EVALUATE_QUANTIZED_CALLS, 1);
    counter!(EVALUATE_QUANTIZED_SETUP_TICKS, setup_ticks);
    counter!(EVALUATE_QUANTIZED_LOOP_OVERHEAD_TICKS, loop_overhead_ticks);
    counter!(EVALUATE_QUANTIZED_FINISH_TICKS, finish_ticks);
    EVALUATE_QUANTIZED_PEAK_LIVE_BYTES.fetch_max(peak_live_bytes, Ordering::Relaxed);
}

/// Records one `evaluate_quantized_named_with_scratch` call's name-to-position
/// resolution ticks — a linear `find` over `named` per weight tensor, O(block
/// count * named count) string compares, invisible to every counter
/// [`record_evaluate_quantized_phase`] commits (that timer starts only after
/// this resolution step already returned).
pub fn record_evaluate_quantized_resolve(ticks: u64) {
    counter!(EVALUATE_QUANTIZED_RESOLVE_TICKS, ticks);
}

/// One process run's worth of `evaluate_quantized`'s phase breakdown --
/// `(calls, resolve_ticks, setup_ticks, loop_overhead_ticks, finish_ticks,
/// peak_live_bytes)`.
#[must_use]
pub fn evaluate_quantized_phase_totals() -> (u64, u64, u64, u64, u64, u64) {
    (
        EVALUATE_QUANTIZED_CALLS.get(),
        EVALUATE_QUANTIZED_RESOLVE_TICKS.get(),
        EVALUATE_QUANTIZED_SETUP_TICKS.get(),
        EVALUATE_QUANTIZED_LOOP_OVERHEAD_TICKS.get(),
        EVALUATE_QUANTIZED_FINISH_TICKS.get(),
        EVALUATE_QUANTIZED_PEAK_LIVE_BYTES.load(Ordering::Relaxed),
    )
}

/// Resets the `evaluate_quantized` phase-breakdown counters to zero.
pub fn reset_evaluate_quantized_phase() {
    let _ = EVALUATE_QUANTIZED_CALLS.snapshot_and_reset();
    let _ = EVALUATE_QUANTIZED_RESOLVE_TICKS.snapshot_and_reset();
    let _ = EVALUATE_QUANTIZED_SETUP_TICKS.snapshot_and_reset();
    let _ = EVALUATE_QUANTIZED_LOOP_OVERHEAD_TICKS.snapshot_and_reset();
    let _ = EVALUATE_QUANTIZED_FINISH_TICKS.snapshot_and_reset();
    EVALUATE_QUANTIZED_PEAK_LIVE_BYTES.store(0, Ordering::Relaxed);
}
