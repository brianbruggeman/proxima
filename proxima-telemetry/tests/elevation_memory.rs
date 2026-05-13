//! Memory profile of `ElevationSink` under load — promotes the discipline-log
//! claim "spikier but bounded memory" from REASONED to MEASURED
//! (`docs/error-elevation/discipline.md`, "Resumption state").
//!
//! Two things are measured here, split by what kind of proof each needs:
//!
//! - The occupancy BOUND (`buffers.len() <= max_traces`, each ring
//!   `<= per_trace_ring`) needs access to `ElevationSink`'s private state, so
//!   that assertion lives next to the existing `elevation_sink_tests` module in
//!   `src/pipes.rs` (`occupancy_is_hard_bounded_by_max_traces_and_per_trace_ring`)
//!   — the natural place for whitebox access, same pattern as
//!   `count_cap_bounds_concurrent_traces` already there.
//! - The ALLOCATION counts below are blackbox (call the pipe, count
//!   allocations via a counting global allocator — the same pattern as
//!   `proxima-protocols/tests/pgwire_codec_integration/alloc_counter.rs`), so
//!   they belong in their own integration-test binary: a `#[global_allocator]`
//!   is process-wide, and `cargo nextest` gives each test its own process, so
//!   per-test deltas stay clean.
//!
//! The elevated downstream in every test here is `NullPipe` (not a capturing
//! sink) — the burst measurement isolates `ElevationSink`'s OWN snapshot/sort/
//! request-build allocations from whatever a real exporter would add on top.

#![cfg(feature = "elevation")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::executor::block_on;
use smallvec::SmallVec;

use proxima_primitives::pipe::SendPipe;
use proxima_telemetry::id::{SpanId, TraceFlags, TraceId};
use proxima_telemetry::level::Level;
use proxima_telemetry::log::{LogBody, LogRecord};
use proxima_telemetry::pipes::{ElevationSink, NullPipe, into_telemetry_handle, log_batch_request};
use proxima_telemetry::sized;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocs() -> usize {
    ALLOC_COUNT.load(Ordering::Relaxed)
}

fn trace_id(byte: u8) -> TraceId {
    TraceId::from_bytes([byte; 16])
}

// mirrors `pipes::elevation_sink_tests::verbose_log` — a record as it looks
// emitted inside a verbose-sampled trace.
fn verbose_log(trace: TraceId, level: Level, ts_ns: u64) -> LogRecord {
    LogRecord {
        ts_ns,
        observed_ts_ns: ts_ns,
        level,
        body: LogBody::Empty,
        attrs: SmallVec::new(),
        trace_id: Some(trace),
        span_id: Some(SpanId::from_bytes([1; 8])),
        trace_flags: TraceFlags::SAMPLED.with_verbose_buffered(),
        module_path: "test",
        file_line: (0, 0),
    }
}

fn sink(max_traces: usize, per_trace_ring: usize) -> ElevationSink {
    ElevationSink::new(
        into_telemetry_handle(NullPipe::new()),
        Level::ERROR,
        per_trace_ring,
        max_traces,
        0,
        true,
    )
}

fn push_batch(target: &ElevationSink, trace: TraceId, level: Level, ts_start: u64, count: usize) {
    let records: Vec<LogRecord> = (0..count as u64)
        .map(|offset| verbose_log(trace, level, ts_start + offset))
        .collect();
    block_on(SendPipe::call(target, log_batch_request(records))).expect("buffer ok");
}

/// `size_of::<LogRecord>()` and the worst-case steady-state byte ceiling for
/// the BUILD-TIME default caps: `max_traces * per_trace_ring * size_of::<LogRecord>()`.
/// This is the formula from the discipline-log claim, made concrete. The
/// assertion is a generous sanity bound (a record ballooning past 512 bytes
/// would be a real regression worth noticing); the printed numbers are the
/// actual measured ceiling.
#[test]
fn record_size_and_default_cap_byte_ceiling() {
    let record_size = core::mem::size_of::<LogRecord>();
    let max_traces = sized::ELEVATION_MAX_TRACES;
    let per_trace_ring = sized::ELEVATION_PER_TRACE_RING;
    let worst_case_records = max_traces * per_trace_ring;
    let worst_case_bytes = worst_case_records * record_size;

    println!(
        "size_of::<LogRecord>() = {record_size} bytes; default caps max_traces={max_traces} \
         per_trace_ring={per_trace_ring} -> worst-case steady-state occupancy = \
         {worst_case_records} records = {worst_case_bytes} bytes \
         ({:.2} MiB), EXCLUDING heap-owned payload inside each record \
         (Bytes/String bodies, Tag attrs that spill SmallVec's inline [Tag;4])",
        worst_case_bytes as f64 / (1024.0 * 1024.0)
    );

    assert!(
        record_size <= 512,
        "LogRecord grew past the 512-byte sanity bound ({record_size} bytes) — \
         re-check the byte ceiling above, it just moved"
    );
}

/// Steady-state marginal allocation cost of buffering one verbose record into
/// an ALREADY-WARM trace (the ring, its Arc, and the DashMap entry all
/// pre-exist). Measured by differencing a 1-record batch against a
/// 1000-record batch into the same warm trace — the O(1) call/request-build
/// overhead cancels out, leaving the true per-record marginal.
#[test]
fn warm_trace_record_push_marginal_allocation_rate() {
    let target = sink(16, 8192);
    let trace = trace_id(1);

    // warm the buffer: this first push pays the one-time setup cost (a
    // separate test measures that cost directly).
    push_batch(&target, trace, Level::INFO, 0, 1);

    let before_small = allocs();
    push_batch(&target, trace, Level::INFO, 1_000, 1);
    let delta_small = allocs() - before_small;

    let before_big = allocs();
    push_batch(&target, trace, Level::INFO, 2_000, 1_000);
    let delta_big = allocs() - before_big;

    let marginal = (delta_big - delta_small) as f64 / 999.0;

    println!(
        "warm-trace push: 1-record batch = {delta_small} allocs (batch Vec + O(1) request \
         overhead); 1000-record batch = {delta_big} allocs; marginal per additional record \
         = {marginal:.4} allocs/record"
    );

    assert!(
        delta_big >= delta_small,
        "a bigger batch must never allocate less than a smaller one"
    );
    assert!(
        marginal <= 1.0,
        "warm-path per-record marginal allocation rate regressed to {marginal:.4} \
         allocs/record — the per-trace ring is pre-sized at construction \
         (`ArrayQueue::new`) and `LogRecord::clone` should not allocate for an \
         empty-attrs record, so this should stay near zero"
    );
}

/// One-time setup cost of the FIRST record buffered for a brand-new
/// `trace_id`: `Arc<TraceBuffer>` + `LogRing::new` (an `ArrayQueue` backing
/// array sized to `per_trace_ring`) + the `DashMap` entry insertion. This is
/// the "cold" half of "spikier" — a burst of NEW traces pays this once each,
/// not on every record.
#[test]
fn first_push_to_new_trace_pays_setup_allocation() {
    let target = sink(16, 8192);
    let trace = trace_id(2);

    // pay this process's one-time first-use costs (the async executor's
    // thread-local waker, any lazy-initialized statics in transitive deps) on
    // a THROWAWAY trace before opening the measurement window below, so the
    // delta isolates the new-trace setup cost, not "first block_on ever in
    // this process."
    push_batch(&target, trace_id(200), Level::INFO, 0, 1);

    let before = allocs();
    push_batch(&target, trace, Level::INFO, 0, 1);
    let delta_cold = allocs() - before;

    let before_warm = allocs();
    push_batch(&target, trace, Level::INFO, 1, 1);
    let delta_warm = allocs() - before_warm;

    println!(
        "first push to a new trace_id = {delta_cold} allocs (Arc<TraceBuffer> + \
         LogRing/ArrayQueue backing array + DashMap entry insert + batch Vec); \
         second push to the now-warm trace = {delta_warm} allocs"
    );

    assert!(
        delta_cold > delta_warm,
        "the cold (new-trace) path must allocate strictly more than the warm path \
         (cold={delta_cold}, warm={delta_warm}) — otherwise the ring/Arc/map-entry \
         setup isn't actually a one-time cost"
    );
    assert!(
        delta_cold <= 8,
        "new-trace setup allocation count regressed to {delta_cold} — expected a small \
         constant (Arc + ArrayQueue backing array + DashMap entry + batch Vec), not \
         something that scales with per_trace_ring"
    );
}

/// The trigger replay-drain burst: a warmed trace holding `ring_occupancy`
/// records receives one ERROR-level record, which triggers
/// `ElevationState::drain_trace` — `LogRing::snapshot` (drains the ring into a
/// `Vec::with_capacity`, then `.to_vec()`s the slice — two allocations by
/// inspection of `log_buffer/ring.rs`), `sort_by_key` (a scratch-probe against
/// plain `Vec<LogRecord>::sort_by_key` confirmed std's stable sort allocates a
/// merge scratch buffer only once the slice exceeds ~20 elements — 0 allocs at
/// len 8/16/20, 1 alloc at len 21/32/64 — so this contributes 0 allocs at
/// ring_occupancy=8 and 1 at ring_occupancy=64), then `log_batch_request`
/// builds the replay envelope and `call_dyn` dispatches it through the erased
/// `PipeHandle` (a plausible source of the remaining constant delta). Run at
/// two ring occupancies to show the allocation EVENT COUNT is near
/// occupancy-invariant (the burst is "few, but large" allocations, not "one
/// alloc per buffered record") — the BYTE volume still scales with occupancy,
/// which is exactly what "spikier" describes.
#[test]
fn trigger_replay_drain_burst_allocation_count() {
    for ring_occupancy in [8usize, 64] {
        let target = sink(16, ring_occupancy);
        let trace = trace_id(3);
        push_batch(&target, trace, Level::INFO, 0, ring_occupancy);

        // baseline: a same-shape non-triggering single-record push into a warm
        // trace, so the burst delta below isolates the DRAIN, not the ordinary
        // per-record admit path already measured above.
        let baseline_before = allocs();
        push_batch(&target, trace, Level::INFO, 10_000, 1);
        let baseline = allocs() - baseline_before;

        let burst_before = allocs();
        push_batch(&target, trace, Level::ERROR, 20_000, 1);
        let burst_total = allocs() - burst_before;
        let burst_only = burst_total.saturating_sub(baseline);

        println!(
            "ring_occupancy={ring_occupancy}: non-triggering push baseline = {baseline} \
             allocs; triggering push total = {burst_total} allocs; \
             burst-attributable (total - baseline) = {burst_only} allocs"
        );

        assert!(
            burst_total >= baseline,
            "a triggering push (which does everything the baseline push does, plus the \
             drain) must never allocate less than the baseline push"
        );
        assert!(
            burst_only <= 10,
            "burst-attributable allocation count regressed to {burst_only} for \
             ring_occupancy={ring_occupancy} — expected a small constant (snapshot's two \
             Vecs, an occasional sort scratch buffer, dispatch), not something that scales \
             with ring occupancy"
        );
    }
}
