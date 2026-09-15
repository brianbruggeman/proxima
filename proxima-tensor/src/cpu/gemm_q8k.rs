use super::*;

/// Below this many total multiply-accumulates (`rows * activation.len()`),
/// a quantized matmul's row loop runs sequentially even when more than one
/// hardware thread exists: `std::thread::scope`'s spawn/join overhead would
/// outweigh the work. Reuses [`PARALLEL_THRESHOLD`], the same element-count
/// floor [`evaluate_node_parallel`] already gates its own per-node chunk
/// dispatch on, rather than a second magic number for the same policy.
/// `None` also covers `rows < workers`, where a per-row split would leave
/// some worker with nothing to do.
///
/// This is a different axis than [`BoundOp::split`]/[`evaluate_node_parallel`]:
/// that machinery chunks a reduce node along its outermost *surviving*
/// output axis (`output_axes.first()` — `bind.rs`'s own doc), which for a
/// batch-1 decode step is the batch axis, extent 1 — nothing to split.
/// `matmul_q4k_f32`/`matmul_q4k_q8k_f32`'s weight-row loop has no such
/// dependency on `BoundOp` at all (`rows`/`k` arrive as plain integers, not
/// a bound node), so it can chunk the one axis that is actually wide at
/// batch-1: weight rows.
///
/// Queries Apple's performance-core count (`hw.perflevel0.logicalcpu`) via
/// `sysctlbyname`, so matmul dispatch spawns workers only across P-cores and
/// skips the E-cores that add per-call dispatch cost without contributing
/// matmul throughput (measured: 8 P-cores beats 10 logical cores on every
/// shape, `docs/discipline.md`). On a homogeneous machine every core reports
/// as perflevel0, so this returns the same count `available_parallelism`
/// would — the fallback in [`matmul_worker_count`] is for when the sysctl is
/// absent or answers something nonsensical, not a second code path for
/// homogeneous boxes.
#[cfg(target_vendor = "apple")]
pub(super) fn performance_core_count() -> Option<usize> {
    let name = c"hw.perflevel0.logicalcpu";
    let mut value: i32 = 0;
    let mut size = core::mem::size_of::<i32>();
    // FFI: sysctlbyname has no safe wrapper in libc; the output pointer and
    // size are stack-local and sized to match the i32 the sysctl documents.
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&raw mut value).cast(),
            &raw mut size,
            core::ptr::null_mut(),
            0,
        )
    };
    if status != 0 || value <= 0 {
        return None;
    }
    Some(value as usize)
}

/// Linux's analogue of Apple's `hw.perflevel0.logicalcpu`: on a hybrid
/// Intel part (Alder Lake and later) the kernel exposes the P-core set at
/// `/sys/devices/cpu_core/cpus` (`E`-cores at the sibling `cpu_atom/cpus`,
/// which this function has no need to read since it only wants the
/// performance set). The path is absent on a non-hybrid CPU -- that IS the
/// answer there, not a missing one: every core is already a performance
/// core, so `None` here is this crate's existing "nothing extra to learn"
/// contract, and [`matmul_worker_count`]'s own `available_parallelism()`
/// fallback already returns the right count for that case.
///
/// Deliberately does NOT re-derive process affinity
/// (`sched_getaffinity`/cgroup quota) here: `std::thread::available_parallelism`'s
/// own documentation ("Host environments such as VMs or container
/// orchestrators may want to restrict the amount of parallelism...") states
/// that it already honors those limits on Linux, and
/// [`matmul_worker_count`]'s `.filter(|&count| count >= 1 && count <=
/// available)` bound is what keeps a global P-core count from ever
/// exceeding what the process may actually use -- duplicating the affinity
/// read here would be a second source of truth for the same fact.
#[cfg(target_os = "linux")]
pub(super) fn performance_core_count() -> Option<usize> {
    let text = std::fs::read_to_string("/sys/devices/cpu_core/cpus").ok()?;
    parse_cpu_list_count(&text)
}

/// Parses a Linux sysfs CPU list (`/sys/devices/cpu_core/cpus`'s own
/// format, e.g. `"0-7,16-23"` or a bare `"4"`) into the count of CPU ids it
/// names -- [`performance_core_count`]'s only consumer, split out so the
/// parsing itself is testable on every host (this crate's dev boxes are
/// aarch64-darwin; the file this reads only exists on a real Linux kernel).
#[cfg(any(test, target_os = "linux"))]
pub(super) fn parse_cpu_list_count(text: &str) -> Option<usize> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let mut count = 0usize;
    for range in text.split(',') {
        let mut bounds = range.splitn(2, '-');
        let start: usize = bounds.next()?.trim().parse().ok()?;
        let end: usize = match bounds.next() {
            Some(end) => end.trim().parse().ok()?,
            None => start,
        };
        count += end.checked_sub(start)?.checked_add(1)?;
    }
    if count == 0 { None } else { Some(count) }
}

#[cfg(not(any(target_vendor = "apple", target_os = "linux")))]
pub(super) fn performance_core_count() -> Option<usize> {
    None
}

/// Worker count for the row split, resolved once and cached for the process
/// lifetime. `std::thread::available_parallelism` is a `sysctl` on macOS —
/// measured at 3.53 us/call, 4.768 ms across the 1350 calls one real forward
/// pass makes through [`quantized_matmul_workers`] — so calling it per
/// matmul is pure waste on a value that never changes at runtime.
///
/// Prefers [`performance_core_count`] over `available_parallelism` on Apple
/// targets: `available_parallelism` returns P+E, but only the P cores run
/// this workload at full speed (measured, see [`performance_core_count`]'s
/// doc), so counting E-cores in the worker pool adds coordination overhead
/// without adding throughput. Falls back to `available_parallelism()` when
/// the sysctl is unavailable or answers something nonsensical (`<= 0` or
/// larger than `available_parallelism()` itself).
///
/// `PROXIMA_MATMUL_WORKERS`, if set to a valid non-zero integer, overrides
/// both of the above; this exists to sweep worker counts without a rebuild.
/// The env var is read once via `OnceLock`, never per call — a per-call
/// `std::env::var` allocates a `String` on every one of those 1350 calls and
/// would contaminate the very cost this cache exists to remove. Default
/// (unset) behavior is unchanged otherwise.
pub(super) fn matmul_worker_count() -> usize {
    static WORKER_COUNT: OnceLock<usize> = OnceLock::new();
    *WORKER_COUNT.get_or_init(|| {
        std::env::var("PROXIMA_MATMUL_WORKERS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&count| count > 0)
            .unwrap_or_else(|| {
                let available = thread::available_parallelism()
                    .map(NonZeroUsize::get)
                    .unwrap_or(1);
                performance_core_count()
                    .filter(|&count| count >= 1 && count <= available)
                    .unwrap_or(available)
            })
    })
}

pub(super) fn quantized_matmul_workers(rows: usize, contraction_width: usize) -> Option<usize> {
    // proxima-debugger diagnostic (this is the ONE choke point every
    // quantized-matmul row-batch call passes through, `Q4_K`/`Q5_K`/`Q6_K`,
    // int8-dot or f32-dot alike): counts total calls and how many actually
    // return `None` (sequential fallback, no thread pool at all) versus
    // `Some` (threaded via `matmul_rows_threaded`) -- settles whether the
    // 1296-call `PARALLEL_NODES`/`MATMUL_DISPATCH_CALLS` figure already
    // covers every row-batch this forward pass runs, or whether a
    // sequential remainder is hiding node wall time `matmul_rows_threaded`
    // never sees.
    #[cfg(feature = "instrument")]
    counter!(instrument::MATMUL_WORKERS_CALLS, 1);
    let total_macs = rows.checked_mul(contraction_width)?;
    if total_macs < PARALLEL_THRESHOLD {
        #[cfg(feature = "instrument")]
        counter!(instrument::MATMUL_WORKERS_NONE, 1);
        return None;
    }
    #[cfg(feature = "instrument")]
    let diag_available_parallelism_started = instrument::read_ticks();
    let workers = matmul_worker_count();
    #[cfg(feature = "instrument")]
    counter!(
        instrument::MATMUL_AVAILABLE_PARALLELISM_TICKS,
        instrument::elapsed_ticks(diag_available_parallelism_started)
    );
    let decision = (workers > 1 && rows >= workers).then_some(workers);
    #[cfg(feature = "instrument")]
    if decision.is_none() {
        counter!(instrument::MATMUL_WORKERS_NONE, 1);
    }
    decision
}

pub(super) use crate::sized::{MIN_MACS_PER_CHUNK, ROW_OVERSUBSCRIBE};

/// Runs `rows` independent per-row computations (`dot_row`) through the
/// shared [`nest_pool`], each writing its own contiguous sub-range of the
/// returned buffer — the row-loop counterpart of [`run_chunks_threaded`]'s
/// pool sibling (`BoundOp`-chunk parallelism, one level up the call stack):
/// no [`BoundOp`] exists at this call site to split, only a row count and a
/// per-row closure, so this dispatches directly over row indices instead of
/// `BoundOp` chunks. Every chunk's slice is carved via `split_at_mut` before
/// any chunk is spawned, so no two pullers ever touch the same output
/// element — same soundness argument as `run_chunks_threaded`'s own slice
/// carve.
///
/// Chunk assignment is dynamic, the same shared-cursor mechanism
/// `run_chunks_threaded`/`claim_and_run` use, applied to row ranges instead
/// of `BoundOp`s ([`claim_and_run_rows`]): `rows` is split into up to
/// `workers * ROW_OVERSUBSCRIBE` ranges (more chunks than pullers), and both
/// the `workers - 1` spawned pool tasks and the calling thread pull the next
/// unclaimed chunk off a shared [`AtomicUsize`] cursor instead of each
/// owning one fixed range. A prior 1:1 static split left the calling thread
/// idling in `Receiver::recv` for whichever spawned chunk ran longest even
/// though equal row counts do not mean equal wall-clock (measured 2.04x
/// spread across 8 equal-row chunks of a 1024^3 GEMM, see [`OVERSUBSCRIBE`]'s
/// doc) — a fast puller now claims another chunk instead of idling.
///
/// `contraction_width` (the per-row `k`, i.e. `activation.len()` at every
/// call site) caps that split: `rows * contraction_width` total multiply-add
/// work is floored against [`MIN_MACS_PER_CHUNK`] before the
/// `workers * ROW_OVERSUBSCRIBE` oversubscription is applied, so a call
/// carrying little total work (e.g. `attn_k`/`attn_v`'s narrow projection)
/// gets fewer, larger chunks instead of the same fixed 40-way split a wide
/// call like `ffn_up`/`ffn_gate` earns — see [`MIN_MACS_PER_CHUNK`]'s own
/// doc for the measurement that picked the floor.
///
/// # Safety (of the `unsafe` blocks inside)
/// `dot_row`'s address crosses the pool's `'static` spawn bound the same way
/// `buffers_address`/`chunks_address` do in `run_chunks_threaded`: cast to
/// `usize` here, reconstructed unsafely inside each pool closure. Sound
/// because this function blocks in `Receiver::recv` for every spawned chunk
/// before returning, so `dot_row` (borrowed from the caller for the whole
/// call) outlives every reconstructed reference. Each chunk's output slice
/// is likewise unique by construction (`split_at_mut` above, carved before
/// any puller starts claiming), and `AtomicUsize::fetch_add` never hands the
/// same chunk index to two pullers, so no two closures ever alias the same
/// output range.
/// The chunk count [`matmul_rows_threaded`] splits `rows` into: capped at
/// `workers * ROW_OVERSUBSCRIBE`, but never more than
/// `rows * contraction_width` total macs supports at
/// [`MIN_MACS_PER_CHUNK`] macs per chunk. A call carrying little total work
/// (narrow `contraction_width`, few `rows`) gets fewer, coarser chunks
/// instead of the fixed oversubscription split every shape used to pay --
/// see [`MIN_MACS_PER_CHUNK`]'s own doc for the per-shape measurement that
/// motivated this.
pub(super) fn row_chunk_count(rows: usize, workers: usize, contraction_width: usize) -> usize {
    let oversubscribed = workers.saturating_mul(ROW_OVERSUBSCRIBE);
    let total_macs = rows.saturating_mul(contraction_width);
    let work_chunks = (total_macs / MIN_MACS_PER_CHUNK).max(1);
    oversubscribed.min(work_chunks).clamp(1, rows.max(1))
}

/// `PROXIMA_COHORT_QUORUM=1` routes the matmul row-cohort round through
/// `CohortSession::run_with_completion(round, Some(&Quorum(chunk_total)))`
/// instead of the zero-overhead `CohortSession::run` default -- exercises
/// the `FanInCompletion` dial `cohort.rs` added (landed, never called from
/// this crate) without changing what gets computed: `Quorum(chunk_total)`
/// is satisfied only once every chunk has retired, the same point cursor
/// exhaustion already stops dispatch at, so the two paths compute the same
/// output. What differs is cost: one extra `completion_ptr` load plus two
/// more atomic loads and a vtable call per claimed chunk, paid by every
/// cohort member on every chunk. Read once and cached, so toggling the env
/// var mid-process has no effect after the first call.
pub(super) fn cohort_quorum_completion_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("PROXIMA_COHORT_QUORUM").is_ok_and(|value| value == "1"))
}

/// [`matmul_rows_threaded`]'s cohort dispatch shape: one round over the same
/// `(row_start, pointer, len)` chunk ranges the pool path carves via
/// `split_at_mut`, run through [`CohortSession::run`] instead of
/// `nest_pool`'s spawn/channel dance. No `'static` erasure is needed here —
/// unlike the pool path, [`CohortSession::run`] blocks the calling thread
/// until every member reports done, so `dot_row` and `chunk_ranges` can stay
/// ordinary borrows for the round's whole lifetime, the same argument
/// `std::thread::scope` relies on.
///
/// `dot_row` can fail (`Row: Fn(usize, &mut [f32]) -> Result<(), TensorError>`,
/// the second argument one row's `width`-wide output slot — `width` is 1 for
/// every caller except [`matmul_q4k_q8k_f32_wide_impl`], which folds every
/// sequence position into that slot so the row's weight bytes are read once
/// and reused across all of them). [`CohortRound::run_chunk`] returns that
/// `Result` directly — [`CohortSession::run`]'s own `RoundReport::first_error`
/// carries the first `Err` any member observes back to the caller, replacing
/// the hand-rolled `OnceLock` this dispatch used to publish through itself.
pub(super) struct RowRound<'round, Row> {
    pub(super) dot_row: &'round Row,
    pub(super) width: usize,
    pub(super) chunk_ranges: &'round [(usize, usize, usize)],
}

impl<Row> CohortRound<TensorError> for RowRound<'_, Row>
where
    Row: Fn(usize, &mut [f32]) -> Result<(), TensorError> + Sync,
{
    fn chunks(&self) -> usize {
        self.chunk_ranges.len()
    }

    fn run_chunk(&self, chunk: ChunkIndex) -> Result<(), TensorError> {
        let (chunk_start, slice_address, slice_len) = self.chunk_ranges[chunk.0];
        // SAFETY: unique to this chunk by construction (`split_at_mut` in
        // `matmul_rows_threaded` before the round starts); the parent
        // `output` outlives every reconstructed slice because
        // `CohortSession::run` does not return until every member has
        // reported done, i.e. until this closure has returned.
        let chunk_output =
            unsafe { core::slice::from_raw_parts_mut(slice_address as *mut f32, slice_len) };
        run_row_chunk(self.dot_row, self.width, chunk_start, chunk_output)
    }
}

pub(super) fn matmul_rows_threaded<Row>(
    rows: usize,
    width: usize,
    workers: usize,
    session: Option<&MatmulSession<'_>>,
    contraction_width: usize,
    dot_row: Row,
) -> Result<Vec<f32>, TensorError>
where
    Row: Fn(usize, &mut [f32]) -> Result<(), TensorError> + Sync,
{
    // proxima-debugger diagnostic: everything this function does before its
    // own spawn/own_chunk/recv_wait timer chain starts -- the `output`
    // alloc, the `chunk_ranges` build, `nest_pool()`, and the `Arc`/
    // `sync_channel` allocations. Named `MATMUL_SETUP_TICKS` so a caller can
    // tell "the dispatch chain is slow" apart from "this untimed setup,
    // paid once per call, is slow" -- see that counter's doc.
    #[cfg(feature = "instrument")]
    let diag_setup_started = instrument::read_ticks();
    let mut output = vec![0.0f32; rows * width];
    let chunk_count = row_chunk_count(rows, workers, contraction_width.saturating_mul(width));
    let chunk_len = rows.div_ceil(chunk_count);

    let mut chunk_ranges = Vec::with_capacity(chunk_count);
    let mut remaining = output.as_mut_slice();
    let mut row_start = 0usize;
    while !remaining.is_empty() {
        let take_rows = chunk_len.min(remaining.len() / width);
        let (slice, rest) = remaining.split_at_mut(take_rows * width);
        remaining = rest;
        chunk_ranges.push((row_start, slice.as_mut_ptr() as usize, slice.len()));
        row_start += take_rows;
    }
    let chunk_ranges_len = chunk_ranges.len();
    #[cfg(feature = "instrument")]
    instrument::record_chunks_created(chunk_ranges_len);

    if let Some(session) = session {
        #[cfg(feature = "instrument")]
        counter!(
            instrument::MATMUL_SETUP_TICKS,
            instrument::elapsed_ticks(diag_setup_started)
        );
        #[cfg(feature = "instrument")]
        counter!(instrument::PARALLEL_NODES, 1);
        #[cfg(feature = "instrument")]
        counter!(instrument::MATMUL_COHORT_DISPATCH_CALLS, 1);
        let round = RowRound {
            dot_row: &dot_row,
            width,
            chunk_ranges: &chunk_ranges,
        };
        // `CohortSession::run` fuses the leader's own claim loop and its
        // wait for the dedicated members into one call (`cohort.rs`'s
        // `run_round(control)` followed by the `done` spin) -- unlike the
        // pool path, it does not expose a separate claim-only timer, so
        // this whole call is charged to `MATMUL_OWN_CHUNK_TICKS` rather
        // than split against `MATMUL_RECV_WAIT_TICKS` (which stays 0 on
        // this path). Nonzero here is the direct witness that the leader
        // is claiming chunks, not spinning idle -- see `cohort.rs`'s
        // `CohortSession::run` doc for the +14.8 ms it cost while it was.
        #[cfg(feature = "instrument")]
        let diag_own_chunk_started = instrument::read_ticks();
        let report = if cohort_quorum_completion_enabled() {
            session.run_with_completion(&round, Some(&Quorum(chunk_ranges_len)))
        } else {
            session.run(&round)
        };
        #[cfg(feature = "instrument")]
        counter!(
            instrument::MATMUL_OWN_CHUNK_TICKS,
            instrument::elapsed_ticks(diag_own_chunk_started)
        );
        if let Some(error) = report.first_error {
            return Err(error);
        }
        if report.abandoned > 0 {
            return Err(TensorError::ThreadedChunkFailed {
                chunk: report.first_abandoned.map_or(0, |chunk| chunk.0 + 1),
                reason: alloc::string::String::from(
                    "cohort member panicked while running this row chunk",
                ),
            });
        }
        return Ok(output);
    }

    let pool = nest_pool()?;
    // SAFETY-relevant: see this function's doc comment for why casting
    // `dot_row`'s address across the pool's `'static` bound is sound here.
    let dot_row_address = &dot_row as *const Row as usize;
    let next_index = Arc::new(AtomicUsize::new(0));
    let chunk_ranges: Arc<Vec<(usize, usize, usize)>> = Arc::new(chunk_ranges);
    let spawned_count = workers
        .saturating_sub(1)
        .min(chunk_ranges_len.saturating_sub(1));
    let (result_sender, result_receiver) = sync_channel(chunk_ranges_len);
    #[cfg(feature = "instrument")]
    counter!(
        instrument::MATMUL_SETUP_TICKS,
        instrument::elapsed_ticks(diag_setup_started)
    );

    #[cfg(feature = "instrument")]
    counter!(instrument::PARALLEL_NODES, 1);

    // proxima-debugger diagnostic (this call's own dispatch-overhead
    // breakdown, `instrument.rs::MATMUL_*_TICKS`): times the spawn loop,
    // the caller's own claiming loop, and the `Receiver::recv` wait
    // separately so a caller can tell whether this dispatch is bottlenecked
    // on spawn (granularity too fine), recv (a straggler the cursor could
    // not route around fast enough), or neither (own-chunk + per-chunk
    // compute already accounted by `record_chunk_ticks` dominates and
    // dispatch is not the ceiling).
    #[cfg(feature = "instrument")]
    let diag_spawn_started = instrument::read_ticks();

    for _ in 0..spawned_count {
        let sender = result_sender.clone();
        let next_index = Arc::clone(&next_index);
        let chunk_ranges = Arc::clone(&chunk_ranges);
        drop(pool.spawn(move || {
            claim_and_run_rows::<Row>(&next_index, dot_row_address, width, &chunk_ranges, &sender);
            Ok::<(), _>(())
        }));
    }

    #[cfg(feature = "instrument")]
    let diag_spawn_ticks = instrument::elapsed_ticks(diag_spawn_started);

    #[cfg(feature = "instrument")]
    let diag_own_chunk_started = instrument::read_ticks();
    // the caller pulls from the same shared cursor as every pool task
    // instead of running one reserved chunk: it never sits idle, since
    // finishing a chunk sends it straight back to `next_index` for another.
    claim_and_run_rows::<Row>(
        &next_index,
        dot_row_address,
        width,
        &chunk_ranges,
        &result_sender,
    );
    drop(result_sender);
    #[cfg(feature = "instrument")]
    let diag_own_chunk_ticks = instrument::elapsed_ticks(diag_own_chunk_started);

    #[cfg(feature = "instrument")]
    let diag_recv_started = instrument::read_ticks();
    let mut outcomes: Vec<Option<Result<(), TensorError>>> =
        (0..chunk_ranges_len).map(|_| None).collect();
    for _ in 0..chunk_ranges_len {
        match result_receiver.recv() {
            Ok((index, outcome)) => outcomes[index] = Some(outcome),
            // every sender clone is gone (each spawned closure's clone is
            // dropped whether it sends or panics), so no further chunk will
            // ever report — stop waiting instead of blocking forever.
            Err(_) => break,
        }
    }
    #[cfg(feature = "instrument")]
    {
        let diag_recv_ticks = instrument::elapsed_ticks(diag_recv_started);
        counter!(instrument::MATMUL_DISPATCH_CALLS, 1);
        counter!(instrument::MATMUL_SPAWN_TICKS, diag_spawn_ticks);
        counter!(instrument::MATMUL_OWN_CHUNK_TICKS, diag_own_chunk_ticks);
        counter!(instrument::MATMUL_RECV_WAIT_TICKS, diag_recv_ticks);
    }
    for (index, outcome) in outcomes.into_iter().enumerate() {
        match outcome {
            Some(result) => result?,
            None => {
                return Err(TensorError::ThreadedChunkFailed {
                    chunk: index + 1,
                    reason: alloc::string::String::from(
                        "worker did not report a result; ProximaBackgroundPool \
                         catches and discards worker panics (see \
                         prime/src/os/background.rs worker())",
                    ),
                });
            }
        }
    }
    Ok(output)
}

/// Pulls row-chunk indices off `next_index` one at a time and runs each to
/// completion through [`run_row_chunk`], reporting through `sender` — the
/// row-loop counterpart of [`claim_and_run`]'s shared-cursor claim loop,
/// called by both the calling thread and every spawned pool task in
/// [`matmul_rows_threaded`] so a puller that finishes early goes straight
/// back for the next available chunk instead of idling.
///
/// # Safety (of the `unsafe` blocks inside)
/// `dot_row_address` and every `(row_start, pointer, len)` triple in
/// `chunk_ranges` must stay valid, and each slice must be unique to its
/// index, for as long as any puller can still observe `next_index` below
/// `chunk_ranges.len()` — guaranteed by [`matmul_rows_threaded`] draining
/// `chunk_ranges.len()` results from `sender`'s channel before `dot_row` or
/// `output` (the parent of every `chunk_ranges` entry) can drop.
/// `fetch_add` never hands out the same index twice, so no two pullers ever
/// touch the same slice.
pub(super) fn claim_and_run_rows<Row>(
    next_index: &AtomicUsize,
    dot_row_address: usize,
    width: usize,
    chunk_ranges: &[(usize, usize, usize)],
    sender: &SyncSender<(usize, Result<(), TensorError>)>,
) where
    Row: Fn(usize, &mut [f32]) -> Result<(), TensorError>,
{
    loop {
        let index = next_index.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        #[cfg(feature = "instrument")]
        counter!(instrument::MATMUL_POOL_CLAIM_ATTEMPTS, 1);
        if index >= chunk_ranges.len() {
            return;
        }
        // SAFETY: see this function's doc comment.
        let dot_row = unsafe { &*(dot_row_address as *const Row) };
        let (chunk_start, slice_address, slice_len) = chunk_ranges[index];
        // SAFETY: unique to this chunk by construction (`split_at_mut` in
        // `matmul_rows_threaded`); the parent `output` outlives every
        // reconstructed slice per this function's doc comment.
        let chunk_output =
            unsafe { core::slice::from_raw_parts_mut(slice_address as *mut f32, slice_len) };
        let outcome = run_row_chunk(dot_row, width, chunk_start, chunk_output);
        let _ = sender.send((index, outcome));
    }
}

/// Runs one contiguous row range of a [`matmul_rows_threaded`] dispatch,
/// writing each row's `width`-wide result into its matching `chunk_output`
/// slot (`width` is 1 for every caller except the folded Q4_K wide path).
pub(super) fn run_row_chunk<Row>(
    dot_row: &Row,
    width: usize,
    chunk_start: usize,
    chunk_output: &mut [f32],
) -> Result<(), TensorError>
where
    Row: Fn(usize, &mut [f32]) -> Result<(), TensorError>,
{
    #[cfg(feature = "instrument")]
    let chunk_started = instrument::read_ticks();
    // proxima-debugger diagnostic: `Instant`-elapsed wall time (below) keeps
    // accruing while this worker is off-core, so on a box carrying ambient
    // load it cannot tell "the kernel is slower in situ" apart from "this
    // thread got descheduled" (`instrument.rs`'s own doc on
    // `WORKER_CPU_NANOS` already established this for the 1->8 scaling
    // read). `thread_cpu_nanos` is the deschedule-immune peer, reused here
    // via `record_worker_cpu_nanos` -- this row-chunk path shares the same
    // `WORKER_CPU_NANOS` pool as `claim_and_run`'s elementwise/node-chunk
    // path (they DO mix within one forward pass), which is why the call
    // below tags itself `CpuWorkload::MatmulRow` rather than leaving the
    // two workloads to be summed together downstream.
    #[cfg(feature = "instrument")]
    let chunk_cpu_started = instrument::thread_cpu_nanos();
    for (offset, slot) in chunk_output.chunks_exact_mut(width).enumerate() {
        dot_row(chunk_start + offset, slot)?;
    }
    #[cfg(feature = "instrument")]
    {
        instrument::record_chunk_ticks(instrument::elapsed_ticks(chunk_started));
        instrument::record_worker_cpu_nanos(
            instrument::CpuWorkload::MatmulRow,
            instrument::thread_cpu_nanos() - chunk_cpu_started,
        );
        counter!(instrument::MATMUL_CHUNK_RUNS, 1);
    }
    Ok(())
}

// ---------------------------------------------------------------------
// `q4k-int8-dot`: int8 dot directly on packed `Q4_K` nibbles against a
// `Q8_K`-quantized activation, skipping `dot_q4k_f32`'s per-superblock
// `[f32; 256]` dequantize entirely. Still its own compile-time feature so
// the codec path stays reachable, but now ON by default alongside its
// `q5k`/`q6k` siblings -- the e2e bench this gate waited for exists: on the
// real openchat-3.5 forward, adding q5k+q6k moved `reduce_matmul_quantized`
// 513.70 -> 497.80 ms with the greedy token bit-identical over 8 runs.
// Note what that measures: 15.91 ms of the 134.84 ms those 9 tensors cost,
// an ~11.8% cut on them, NOT the near-elimination the dequantize framing
// suggests. See `proxima-tensor/docs/discipline.md` for the landing rows.
// ---------------------------------------------------------------------

/// Byte offsets into one packed `Q4_K` super-block ([`Q4K_BLOCK_BYTES`]
/// bytes), mirroring `proxima_gguf::quant::q4_k`'s private layout
/// constants -- duplicated here (not re-exported from that module) because
/// [`dot_q4k_q8k`] reads the raw bytes directly rather than calling
/// `dequantize_block`, which is the entire point: no `[f32; 256]`
/// intermediate.
#[cfg(feature = "q4k-int8-dot")]
pub(super) const Q4K_D_OFFSET: usize = 0;
#[cfg(feature = "q4k-int8-dot")]
pub(super) const Q4K_DMIN_OFFSET: usize = 2;
#[cfg(feature = "q4k-int8-dot")]
pub(super) const Q4K_SCALES_OFFSET: usize = 4;
#[cfg(feature = "q4k-int8-dot")]
pub(super) const Q4K_SCALE_BYTES: usize = 12;
#[cfg(feature = "q4k-int8-dot")]
pub(super) const Q4K_QS_OFFSET: usize = Q4K_SCALES_OFFSET + Q4K_SCALE_BYTES;
/// Sub-blocks of 32 elements per `Q4_K` super-block (`QK_K/32` = 8) --
/// [`Q4K_BLOCK_ELEMENTS`] is `pub(crate)`-visible above; this is the same
/// number under the name the int8 dot's loop structure uses it by. `Q5_K`
/// shares this exact sub-block shape (`q5_k.rs`'s own module doc: "the same
/// super-block/sub-block shape... as `q4_k`"), so [`dot_q5k_q8k_block_scalar`]
/// reuses this constant rather than defining an identical `Q5K_SUB_BLOCKS`.
#[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot"))]
pub(super) const Q4K_SUB_BLOCKS: usize = Q4K_BLOCK_ELEMENTS / 32;

/// Bytes per `Q8_K` super-block: `f32` scale (4 bytes), plus `QK_K` `i8`
/// quants (256 bytes), plus `QK_K/16` `i16` per-16-element partial sums
/// (16 times 2 bytes = 32 bytes) -- 292 total. Mirrors ggml's own
/// `block_q8_K` byte-for-byte (`ggml-common.h:333`,
/// `static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K +
/// QK_K/16*sizeof(int16_t), ...)`) -- deliberately: [`dot_q4k_q8k`]
/// takes `activation_q8k` as raw bytes in that exact layout rather than a
/// new struct type. Per guiding-principles §1: a byte buffer already in
/// ggml's own wire shape needs no host type any more than `dot_q4k_f32`'s
/// `weight_row: &[u8]` does -- a `(d, qs, bsums)` tuple or three parallel
/// slices would ALSO work, but would require [`quantize_row_q8k`] and
/// [`dot_q4k_q8k`] to agree on three independent buffer lengths instead of
/// one, for no capability a caller gains.
// `Q8_K` is the one activation format every K-quant weight codec (`Q4_K`,
// `Q5_K`, `Q6_K`) dots against -- shared, not duplicated per format, so
// these constants and `quantize_row_q8k` below build under ANY of the
// three weight codecs' int8-dot features, not `q4k-int8-dot` alone.
#[cfg(any(
    feature = "q4k-int8-dot",
    feature = "q5k-int8-dot",
    feature = "q6k-int8-dot"
))]
pub(super) const Q8K_BLOCK_BYTES: usize = 4 + Q4K_BLOCK_ELEMENTS + (Q4K_BLOCK_ELEMENTS / 16) * 2;
#[cfg(any(
    feature = "q4k-int8-dot",
    feature = "q5k-int8-dot",
    feature = "q6k-int8-dot"
))]
pub(super) const Q8K_D_OFFSET: usize = 0;
#[cfg(any(
    feature = "q4k-int8-dot",
    feature = "q5k-int8-dot",
    feature = "q6k-int8-dot"
))]
pub(super) const Q8K_QS_OFFSET: usize = 4;
#[cfg(any(
    feature = "q4k-int8-dot",
    feature = "q5k-int8-dot",
    feature = "q6k-int8-dot"
))]
pub(super) const Q8K_BSUMS_OFFSET: usize = Q8K_QS_OFFSET + Q4K_BLOCK_ELEMENTS;
#[cfg(any(
    feature = "q4k-int8-dot",
    feature = "q5k-int8-dot",
    feature = "q6k-int8-dot"
))]
pub(super) const Q8K_BSUMS_COUNT: usize = Q4K_BLOCK_ELEMENTS / 16;

#[cfg(any(
    feature = "q4k-int8-dot",
    feature = "q5k-int8-dot",
    feature = "q6k-int8-dot"
))]
pub(super) fn f16_le_at(bytes: &[u8], offset: usize) -> f32 {
    let mut raw = [0u8; 2];
    raw.copy_from_slice(&bytes[offset..offset + 2]);
    half::f16::from_le_bytes(raw).to_f32()
}

/// Quantizes an activation vector into packed `Q8_K` bytes (`Q8K_BLOCK_BYTES`
/// per 256-element super-block) -- the one pass [`dot_q4k_q8k`]'s int8
/// mechanism needs, hoisted OUT of the per-row loop the same way this
/// module's docs already measured a conversion pipe at (52 vs 52
/// instructions, `docs/discipline.md`): paying this per row instead of
/// once per `matmul_q4k_q8k_f32` call would cost `rows`x -- 4096x at this
/// crate's real weight-matrix shapes. Ports `quantize_row_q8_K_ref`
/// (`ggml-quants.c:2471-2505`) bit-for-bit: per super-block, finds the
/// largest-magnitude element, scales by `-127/max`, rounds every element to
/// `i8` via [`proxima_gguf::quant::q4_k::nearest_int`] (the same
/// ties-to-even bit trick `Q4_K`'s own reference quantizer uses -- ggml
/// calls one `nearest_int` for every k-quant codec, not a `Q8_K`-specific
/// one), then folds each 16-element run into one `i16` partial sum
/// (`bsums`) [`dot_q4k_q8k`]'s mins correction consumes without
/// re-scanning `qs`.
///
/// `activation.len()` must be a whole multiple of `Q4K_BLOCK_ELEMENTS`
/// (256); `output.len()` must exactly equal the block count times
/// `Q8K_BLOCK_BYTES`. No allocation: `output` is caller-provided.
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if either length requirement
/// above is not met.
#[cfg(any(
    feature = "q4k-int8-dot",
    feature = "q5k-int8-dot",
    feature = "q6k-int8-dot"
))]
pub(super) use crate::sized::MIN_QUANTIZE_BLOCKS_FOR_DISPATCH;

/// [`quantize_row_q8k`] dispatched across the cohort when a `session` is
/// open and the call's super-block count clears
/// [`MIN_QUANTIZE_BLOCKS_FOR_DISPATCH`] -- the same shape
/// [`run_elementwise_dispatch`] has to [`run_elementwise`]: every `Q8_K`
/// super-block quantizes independently (this function's own doc), so a
/// contiguous range of blocks is exactly as independent as
/// [`ElementwiseRowRound`]'s outer-position ranges. Falls straight through
/// to [`quantize_row_q8k`] whenever any gate fails: no session, too few
/// blocks, or fewer than one worker.
#[cfg(any(
    feature = "q4k-int8-dot",
    feature = "q5k-int8-dot",
    feature = "q6k-int8-dot"
))]
pub(super) fn quantize_row_q8k_dispatch(
    activation: &[f32],
    output: &mut [u8],
    session: Option<&MatmulSession<'_>>,
) -> Result<(), TensorError> {
    let Some(session) = session else {
        return quantize_row_q8k(activation, output);
    };
    if !activation.len().is_multiple_of(Q4K_BLOCK_ELEMENTS) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length is not a whole multiple of the q8_k super-block size",
        });
    }
    let block_count = activation.len() / Q4K_BLOCK_ELEMENTS;
    if output.len() != block_count * Q8K_BLOCK_BYTES {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "q8_k output length does not match the activation block count",
        });
    }
    if block_count < MIN_QUANTIZE_BLOCKS_FOR_DISPATCH {
        return quantize_row_q8k(activation, output);
    }
    let workers = matmul_worker_count();
    if workers <= 1 {
        return quantize_row_q8k(activation, output);
    }
    let chunk_count = (workers * OVERSUBSCRIBE).min(block_count);
    let block_chunk_len = block_count.div_ceil(chunk_count);
    let mut chunk_ranges = Vec::with_capacity(chunk_count);
    let mut remaining_in = activation;
    let mut remaining_out = &mut *output;
    while !remaining_out.is_empty() {
        let take_blocks = block_chunk_len.min(remaining_out.len() / Q8K_BLOCK_BYTES);
        let (in_slice, in_rest) = remaining_in.split_at(take_blocks * Q4K_BLOCK_ELEMENTS);
        let (out_slice, out_rest) = remaining_out.split_at_mut(take_blocks * Q8K_BLOCK_BYTES);
        remaining_in = in_rest;
        remaining_out = out_rest;
        chunk_ranges.push((
            in_slice.as_ptr() as usize,
            in_slice.len(),
            out_slice.as_mut_ptr() as usize,
            out_slice.len(),
        ));
    }
    if chunk_ranges.len() < 2 {
        return quantize_row_q8k(activation, output);
    }
    let round = QuantizeRound {
        chunk_ranges: &chunk_ranges,
    };
    let report = session.run(&round);
    if report.abandoned > 0 {
        return Err(TensorError::ThreadedChunkFailed {
            chunk: report.first_abandoned.map_or(0, |chunk| chunk.0 + 1),
            reason: alloc::string::String::from(
                "cohort member panicked while running this quantize chunk",
            ),
        });
    }
    Ok(())
}

/// [`quantize_row_q8k_dispatch`]'s cohort dispatch shape: one round over
/// `(in_ptr, in_len, out_ptr, out_len)` block ranges, run through
/// [`CohortSession::run`]. No error path -- every range's shape was already
/// validated whole, by construction, before the round opens, so
/// [`quantize_q8k_block`] cannot fail the way a matmul row's dot product can.
#[cfg(any(
    feature = "q4k-int8-dot",
    feature = "q5k-int8-dot",
    feature = "q6k-int8-dot"
))]
pub(super) struct QuantizeRound<'round> {
    pub(super) chunk_ranges: &'round [(usize, usize, usize, usize)],
}

#[cfg(any(
    feature = "q4k-int8-dot",
    feature = "q5k-int8-dot",
    feature = "q6k-int8-dot"
))]
impl CohortRound<TensorError> for QuantizeRound<'_> {
    fn chunks(&self) -> usize {
        self.chunk_ranges.len()
    }

    fn run_chunk(&self, chunk: ChunkIndex) -> Result<(), TensorError> {
        let (in_ptr, in_len, out_ptr, out_len) = self.chunk_ranges[chunk.0];
        // SAFETY: unique to this chunk by construction (`split_at`/
        // `split_at_mut` in `quantize_row_q8k_dispatch` before the round
        // starts); the parent `activation`/`output` outlive every
        // reconstructed slice because `CohortSession::run` does not return
        // until every member has reported done.
        let in_slice = unsafe { core::slice::from_raw_parts(in_ptr as *const f32, in_len) };
        // SAFETY: same argument as `in_slice` above, mutable side.
        let out_slice = unsafe { core::slice::from_raw_parts_mut(out_ptr as *mut u8, out_len) };
        for (block, out_block) in in_slice
            .as_chunks::<Q4K_BLOCK_ELEMENTS>()
            .0
            .iter()
            .zip(out_slice.as_chunks_mut::<Q8K_BLOCK_BYTES>().0)
        {
            quantize_q8k_block(block, out_block);
        }
        Ok(())
    }
}

#[cfg(any(
    feature = "q4k-int8-dot",
    feature = "q5k-int8-dot",
    feature = "q6k-int8-dot"
))]
pub fn quantize_row_q8k(activation: &[f32], output: &mut [u8]) -> Result<(), TensorError> {
    if !activation.len().is_multiple_of(Q4K_BLOCK_ELEMENTS) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length is not a whole multiple of the q8_k super-block size",
        });
    }
    let block_count = activation.len() / Q4K_BLOCK_ELEMENTS;
    if output.len() != block_count * Q8K_BLOCK_BYTES {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "q8_k output length does not match the activation block count",
        });
    }
    for (chunk, out_block) in activation
        .as_chunks::<Q4K_BLOCK_ELEMENTS>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<Q8K_BLOCK_BYTES>().0)
    {
        quantize_q8k_block(chunk, out_block);
    }
    Ok(())
}

#[cfg(any(
    feature = "q4k-int8-dot",
    feature = "q5k-int8-dot",
    feature = "q6k-int8-dot"
))]
pub(super) fn quantize_q8k_block(chunk: &[f32], out_block: &mut [u8]) {
    let mut amax = 0.0f32;
    let mut max = 0.0f32;
    for &value in chunk {
        let absolute = value.abs();
        if absolute > amax {
            amax = absolute;
            max = value;
        }
    }
    if amax == 0.0 {
        out_block.fill(0);
        return;
    }

    let iscale = -127.0f32 / max;
    let mut levels = [0i8; Q4K_BLOCK_ELEMENTS];
    for (level, &value) in levels.iter_mut().zip(chunk.iter()) {
        *level = proxima_gguf::quant::q4_k::nearest_int(iscale * value).min(127) as i8;
    }

    let scale = 1.0 / iscale;
    out_block[Q8K_D_OFFSET..Q8K_D_OFFSET + 4].copy_from_slice(&scale.to_le_bytes());

    let qs = &mut out_block[Q8K_QS_OFFSET..Q8K_QS_OFFSET + Q4K_BLOCK_ELEMENTS];
    for (slot, &level) in qs.iter_mut().zip(levels.iter()) {
        *slot = level.cast_unsigned();
    }

    let bsums_region = &mut out_block[Q8K_BSUMS_OFFSET..Q8K_BSUMS_OFFSET + Q8K_BSUMS_COUNT * 2];
    for (sixteen, bytes) in levels
        .as_chunks::<16>()
        .0
        .iter()
        .zip(bsums_region.as_chunks_mut::<2>().0)
    {
        let sum: i16 = sixteen.iter().map(|&level| i16::from(level)).sum();
        bytes.copy_from_slice(&sum.to_le_bytes());
    }
}

/// [`quantize_q8k_block`]'s inverse: one packed `Q8_K` super-block back to
/// its `f32` levels (`scale * level`), the same `d`/`qs` fields
/// [`dot_q4k_q8k_block_scalar`] already reads (`bsums` is a fused-path-only
/// correction term, unused by a plain dequantize-then-fold). Exists for
/// [`QuantDot::Unfused`]'s own `In`/`Out` shape to match [`QuantDot::Fused`]
/// exactly: both take the SAME packed `Q8_K` activation bytes, so an
/// unfused reference needs a way back to `f32` for those bytes, not just
/// for the weight row (`proxima_gguf`'s codec `dequantize` already covers
/// the weight side).
///
/// # Panics
/// If `block.len() != Q8K_BLOCK_BYTES` or `output.len() != Q4K_BLOCK_ELEMENTS`.
#[cfg(any(
    feature = "q4k-int8-dot",
    feature = "q5k-int8-dot",
    feature = "q6k-int8-dot"
))]
pub(super) fn dequantize_q8k_block(block: &[u8], output: &mut [f32]) {
    let mut d_bytes = [0u8; 4];
    d_bytes.copy_from_slice(&block[Q8K_D_OFFSET..Q8K_D_OFFSET + 4]);
    let scale = f32::from_le_bytes(d_bytes);
    let qs = &block[Q8K_QS_OFFSET..Q8K_QS_OFFSET + Q4K_BLOCK_ELEMENTS];
    for (level, out) in qs.iter().zip(output.iter_mut()) {
        *out = scale * f32::from(level.cast_signed());
    }
}

/// A `Q4_K`/`Q5_K`/`Q6_K` weight row's dot product against a packed `Q8_K`
/// activation row, as a [`Pipe`]: `In` = the activation row's packed bytes
/// ([`quantize_row_q8k`]'s own output shape), `Out` = the row's dot
/// product, `Err` = the same [`TensorError`] the underlying kernel already
/// raises on a malformed shape. The weight row travels with the pipe value
/// itself as a [`QuantizedBlock`] -- the codec (`Q4K`/`Q5K`/`Q6K`) is the
/// variant `QuantizedBlock` already carries, not a second marker type
/// minted to say the same thing again.
///
/// [`Self::Fused`] calls straight into the codec's own int8 kernel
/// ([`dot_q4k_q8k`]/[`dot_q5k_q8k`]/[`dot_q6k_q8k`]) -- packed nibbles in,
/// one integer accumulate, no `f32` intermediate. [`Self::Unfused`]
/// dequantizes both operands to `f32` first (the weight row via
/// `proxima_gguf`'s own codec `dequantize`, the activation row via this
/// module's private `dequantize_q8k_block`) and folds with a plain `f32`
/// multiply-add -- the incumbent shape this crate's own parity tests
/// already hold as ground truth (see
/// `matmul_q4k_f32_matches_dequantize_then_f32_matmul`).
///
/// Selecting between the two at a call site with no branch on the caller's
/// part is the reason this is one enum with one [`Pipe`] impl rather than
/// two free functions: `QuantDot::Fused(block).call(q8k_row)` and
/// `QuantDot::Unfused(block).call(q8k_row)` are the same shape, so a caller
/// choosing fused-vs-unfused per matmul row (a build-time feature gate or a
/// measured per-target decision) holds either behind one type, matched once
/// inside `call` rather than at every call site.
#[cfg(any(
    feature = "q4k-int8-dot",
    feature = "q5k-int8-dot",
    feature = "q6k-int8-dot"
))]
pub enum QuantDot<'a> {
    Fused(QuantizedBlock<'a>),
    Unfused(QuantizedBlock<'a>),
}

#[cfg(any(
    feature = "q4k-int8-dot",
    feature = "q5k-int8-dot",
    feature = "q6k-int8-dot"
))]
impl<'a> Pipe for QuantDot<'a> {
    type In = &'a [u8];
    type Out = f32;
    type Err = TensorError;

    fn call(&self, activation_q8k: &'a [u8]) -> impl Future<Output = Result<f32, TensorError>> {
        let result = match self {
            Self::Fused(block) => fused_quant_dot(*block, activation_q8k),
            Self::Unfused(block) => unfused_quant_dot(*block, activation_q8k),
        };
        async move { result }
    }
}

/// [`QuantDot::Fused`]'s own body: select the codec's int8 kernel by
/// matching [`QuantizedBlock`]'s variant, the same table
/// [`dot_fn_for`] already builds for the `cohort-staged-graph` batching
/// path -- this is the non-batched, single-row counterpart. A codec whose
/// int8-dot feature is not compiled in (or a non-K-quant variant like
/// `Q8_0`/`Float16`) is an honest [`TensorError::NotLowerable`], never a
/// silent fallback to a different codec's kernel.
#[cfg(any(
    feature = "q4k-int8-dot",
    feature = "q5k-int8-dot",
    feature = "q6k-int8-dot"
))]
pub(super) fn fused_quant_dot(block: QuantizedBlock<'_>, activation_q8k: &[u8]) -> Result<f32, TensorError> {
    match block {
        #[cfg(feature = "q4k-int8-dot")]
        QuantizedBlock::Q4K(bytes) => dot_q4k_q8k(bytes, activation_q8k),
        #[cfg(feature = "q5k-int8-dot")]
        QuantizedBlock::Q5K(bytes) => dot_q5k_q8k(bytes, activation_q8k),
        #[cfg(feature = "q6k-int8-dot")]
        QuantizedBlock::Q6K(bytes) => dot_q6k_q8k(bytes, activation_q8k),
        _ => Err(TensorError::NotLowerable {
            node: NodeId(0),
            reason: "QuantDot::Fused only supports a K-quant codec whose int8-dot feature is enabled",
        }),
    }
}

/// [`QuantDot::Unfused`]'s own body: dequantize both operands to `f32`
/// (weight row via `proxima_gguf`'s codec `dequantize`, activation row via
/// [`dequantize_q8k_block`]) and fold with a plain multiply-add. Every
/// length check mirrors [`dot_q4k_q8k`]'s own -- this path takes the
/// identical `In` shape, so it must reject the identical malformed shapes.
#[cfg(any(
    feature = "q4k-int8-dot",
    feature = "q5k-int8-dot",
    feature = "q6k-int8-dot"
))]
pub(super) fn unfused_quant_dot(block: QuantizedBlock<'_>, activation_q8k: &[u8]) -> Result<f32, TensorError> {
    let (weight_bytes, block_bytes, qk_k): (&[u8], usize, usize) = match block {
        QuantizedBlock::Q4K(bytes) => (
            bytes,
            proxima_gguf::quant::q4_k::BLOCK_BYTES,
            proxima_gguf::quant::q4_k::QK_K,
        ),
        QuantizedBlock::Q5K(bytes) => (
            bytes,
            proxima_gguf::quant::q5_k::BLOCK_BYTES,
            proxima_gguf::quant::q5_k::QK_K,
        ),
        QuantizedBlock::Q6K(bytes) => (
            bytes,
            proxima_gguf::quant::q6_k::BLOCK_BYTES,
            proxima_gguf::quant::q6_k::QK_K,
        ),
        _ => {
            return Err(TensorError::NotLowerable {
                node: NodeId(0),
                reason: "QuantDot::Unfused only supports a K-quant codec (Q4_K/Q5_K/Q6_K)",
            });
        }
    };
    if !weight_bytes.len().is_multiple_of(block_bytes) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of its codec's block size",
        });
    }
    let block_count = weight_bytes.len() / block_bytes;
    if activation_q8k.len() != block_count * Q8K_BLOCK_BYTES {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "q8_k activation length does not match the weight row's block count",
        });
    }

    let elements = block_count * qk_k;
    let mut weight_f32 = vec![0.0f32; elements];
    let dequantize_result = match block {
        QuantizedBlock::Q4K(bytes) => proxima_gguf::quant::q4_k::dequantize(bytes, &mut weight_f32),
        QuantizedBlock::Q5K(bytes) => proxima_gguf::quant::q5_k::dequantize(bytes, &mut weight_f32),
        QuantizedBlock::Q6K(bytes) => proxima_gguf::quant::q6_k::dequantize(bytes, &mut weight_f32),
        _ => unreachable!("codec already matched above"),
    };
    dequantize_result.map_err(|_| TensorError::QuantizedShapeMismatch {
        reason: "weight row failed to dequantize despite passing its own shape check",
    })?;

    let mut activation_f32 = vec![0.0f32; elements];
    for (block_bytes, block_f32) in activation_q8k
        .as_chunks::<Q8K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation_f32.as_chunks_mut::<Q4K_BLOCK_ELEMENTS>().0)
    {
        dequantize_q8k_block(block_bytes, block_f32);
    }

    Ok(weight_f32
        .iter()
        .zip(&activation_f32)
        .map(|(weight, value)| weight * value)
        .sum())
}

/// One `Q4_K`-weight-row x `Q8_K`-activation int8 dot product --
/// `dot_q4k_f32`'s packed-arithmetic sibling: same `weight_row` shape
/// (raw `Q4_K` bytes, a whole number of `Q4K_BLOCK_BYTES` super-blocks),
/// but `activation_q8k` is [`quantize_row_q8k`]'s packed `Q8_K` bytes
/// instead of a plain `f32` slice, and the fold is an integer dot on the
/// packed 4-bit nibbles rather than an `f32` multiply-add over a
/// dequantized scratch buffer. `dot_q4k_f32` is left untouched as the
/// correct codec path for non-matmul consumers (module-level comment
/// above) -- this is an additional arm, not a replacement.
///
/// Caches `std::is_x86_feature_detected!("avx2")`, probed once per process
/// life -- the same [`OnceLock`] shape [`matmul_worker_count`] already uses
/// for `PROXIMA_MATMUL_WORKERS`/`performance_core_count`, so
/// [`dot_q4k_q8k`]'s per-block hot loop never repeats the CPUID probe the
/// detection macro performs. Only compiled when the build itself did NOT
/// already declare AVX2 present at compile time (`q4k_avx2` off): a
/// `-C target-feature=+avx2` / `-C target-cpu=native` build already knows
/// the answer statically and calls [`dot_q4k_q8k_block_avx2`] unconditionally
/// (`dot_q4k_q8k`'s `q4k_avx2` arm), skipping this check entirely -- this
/// function exists for the DEFAULT `cargo build --release` on an ordinary
/// x86_64 host, which has no reason to know at compile time whether the
/// CPU it will run on has AVX2 (essentially universal since 2013, but not
/// guaranteed by the bare `x86_64` target triple).
#[cfg(all(target_arch = "x86_64", feature = "q4k-int8-dot", not(q4k_avx2)))]
pub(super) fn avx2_runtime_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| std::is_x86_feature_detected!("avx2"))
}

/// Dispatches to `dot_q4k_q8k_block_neon_dotprod` when built with the
/// `q4k_dotprod` cfg (`build.rs`: every aarch64 target this workspace
/// builds for), to `dot_q4k_q8k_block_avx2` when built with the
/// `q4k_avx2` cfg (`build.rs`: an x86 target whose `CARGO_CFG_TARGET_FEATURE`
/// lists `avx2` -- unlike aarch64's `FEAT_DotProd`, AVX2 is NOT in the x86-64
/// baseline ISA, so this one is opt-in via `-C target-feature=+avx2` /
/// `-C target-cpu`, not implied by the target triple alone), to the SAME
/// `dot_q4k_q8k_block_avx2` chosen at RUNTIME via
/// `avx2_runtime_available` on a plain x86_64 build that did not opt in
/// at compile time (so a default `cargo build --release` on a modern
/// x86_64 host still gets the fast kernel instead of silently falling back
/// to scalar), and to the portable `dot_q4k_q8k_block_scalar` everywhere
/// else (or when the runtime probe reports AVX2 absent). All three/four
/// compute the identical mechanism -- read 4.5 bits/weight off `weight_row`
/// and do the multiply-accumulate against `Q8_K` `i8` activations directly,
/// no `f32` intermediate at all -- the NEON arm is an acceleration of that
/// mechanism (`vdotq_s32`'s 16-lane int8 dot via inline `sdot`,
/// `core::arch::aarch64::vdotq_s32` itself being unstable on this toolchain
/// -- `stdarch_neon_dotprod`), and the AVX2 arm is a second, independent
/// acceleration of it (`_mm256_maddubs_epi16` + `_mm256_madd_epi16`'s
/// 32-lane unsigned-times-signed int8 dot), not a different one.
///
/// AVX-512 VNNI (`_mm512_dpbusd_epi32`) and AVX-VNNI
/// (`_mm256_dpbusd_epi32`) would each do this same dot in one instruction
/// instead of AVX2's maddubs+madd pair -- not implemented here: this crate
/// cannot execute either on its aarch64-darwin dev boxes, so shipping an
/// unverified single-instruction accumulation path (whose saturation
/// semantics need re-checking against `dot_q4k_q8k_block_scalar` byte for
/// byte, not assumed) is deferred rather than guessed at. AVX2 is the
/// floor that matters (present on essentially every x86_64 CPU since
/// 2013); the follow-up is a `dpbusd`-based
/// `dot_q4k_q8k_block_avxvnni`/`_avx512vnni` pair selected ahead of the
/// AVX2 arm in `avx2_runtime_available`'s priority order, verified on
/// real VNNI hardware before it lands.
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of `Q4K_BLOCK_BYTES`, or `activation_q8k.len()` does
/// not equal the row's block count times `Q8K_BLOCK_BYTES`.
#[cfg(feature = "q4k-int8-dot")]
pub fn dot_q4k_q8k(weight_row: &[u8], activation_q8k: &[u8]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q4K_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q4_k block size",
        });
    }
    let block_count = weight_row.len() / Q4K_BLOCK_BYTES;
    if activation_q8k.len() != block_count * Q8K_BLOCK_BYTES {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "q8_k activation length does not match the weight row's block count",
        });
    }

    #[cfg(all(target_arch = "x86_64", not(q4k_avx2)))]
    let use_avx2_runtime = avx2_runtime_available();

    let mut acc = 0.0f32;
    for (weight_block, q8k_block) in weight_row
        .as_chunks::<Q4K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation_q8k.as_chunks::<Q8K_BLOCK_BYTES>().0)
    {
        #[cfg(q4k_dotprod)]
        // SAFETY: `q4k_dotprod` is emitted by build.rs only for aarch64
        // targets, all of which carry FEAT_DotProd (build.rs's own doc).
        let block_sum = unsafe { dot_q4k_q8k_block_neon_dotprod(weight_block, q8k_block) };
        #[cfg(all(q4k_avx2, not(q4k_dotprod)))]
        // SAFETY: `q4k_avx2` is emitted by build.rs only when
        // `CARGO_CFG_TARGET_FEATURE` actually lists `avx2` (build.rs's own
        // doc) -- the caller opted the build itself into AVX2, so the
        // instructions this block issues are guaranteed present.
        let block_sum = unsafe { dot_q4k_q8k_block_avx2(weight_block, q8k_block) };
        #[cfg(all(target_arch = "x86_64", not(q4k_dotprod), not(q4k_avx2)))]
        let block_sum = if use_avx2_runtime {
            // SAFETY: `use_avx2_runtime` is true only when
            // `avx2_runtime_available` confirmed
            // `std::is_x86_feature_detected!("avx2")` before this loop
            // started.
            unsafe { dot_q4k_q8k_block_avx2(weight_block, q8k_block) }
        } else {
            dot_q4k_q8k_block_scalar(weight_block, q8k_block)
        };
        #[cfg(not(any(q4k_dotprod, q4k_avx2, target_arch = "x86_64")))]
        let block_sum = dot_q4k_q8k_block_scalar(weight_block, q8k_block);
        acc += block_sum;
    }
    Ok(acc)
}

/// [`dot_q4k_q8k`] with the dispatch forced to
/// `dot_q4k_q8k_block_scalar` regardless of `q4k_dotprod` -- the "what
/// does portable packing alone buy" measurement the discipline log's
/// packed-kernel row reports standalone, next to the `vdotq_s32`-
/// accelerated number `dot_q4k_q8k` itself produces on an aarch64 build.
/// Also what non-aarch64 targets (`cargo check --target
/// x86_64-unknown-linux-gnu`) actually call, via `dot_q4k_q8k`'s own
/// `not(q4k_dotprod)` arm -- this function exists so that arm's code path
/// stays reachable, and separately benchable, from an aarch64 host too.
///
/// # Errors
/// Same as [`dot_q4k_q8k`].
#[cfg(feature = "q4k-int8-dot")]
pub fn dot_q4k_q8k_portable(weight_row: &[u8], activation_q8k: &[u8]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q4K_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q4_k block size",
        });
    }
    let block_count = weight_row.len() / Q4K_BLOCK_BYTES;
    if activation_q8k.len() != block_count * Q8K_BLOCK_BYTES {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "q8_k activation length does not match the weight row's block count",
        });
    }

    let mut acc = 0.0f32;
    for (weight_block, q8k_block) in weight_row
        .as_chunks::<Q4K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation_q8k.as_chunks::<Q8K_BLOCK_BYTES>().0)
    {
        acc += dot_q4k_q8k_block_scalar(weight_block, q8k_block);
    }
    Ok(acc)
}

/// The portable packed-nibble x `Q8_K` int8 dot -- no dequantize pass, no
/// `f32` intermediate, no architecture intrinsics. This is the mechanism
/// itself (reading 4.5 bits/weight off `weight_block` instead of the 32
/// bits/weight a decoded `f32` row costs, 7.11x less traffic than
/// `dot_q4k_f32`'s scratch buffer); [`dot_q4k_q8k_block_neon_dotprod`]
/// accelerates this same computation with `vdotq_s32`, it does not replace
/// it -- every non-aarch64 target this crate builds for (the
/// `cargo check --target x86_64-unknown-linux-gnu` gate cell) runs this
/// function, not a stand-in nobody exercises.
///
/// Ports the scalar body of `ggml_vec_dot_q4_K_q8_K`
/// (`ggml-cpu/quants.c:515`, the `#else` arm every architecture file's own
/// vectorized version specializes): per sub-block of 32, unpack its 6-bit
/// `(scale, min)` pair via [`proxima_gguf::quant::q4_k::get_scale_min_k4`],
/// dot the sub-block's 32 nibbles (0..15, an unsigned weight code -- never
/// sign-extended) against the matching 32 `Q8_K` `i8` activations, scale
/// by the sub-block's 6-bit scale code; separately, the mins correction
/// sums each sub-block's `Q8_K` `bsums` pair times its 6-bit min code.
/// `d`/`dmin` (the super-block's own `f16` scale pair) and the `Q8_K`
/// block's `f32` scale multiply in only at the very end -- two `f32`
/// operations total per 256-element super-block, exactly matching this
/// component's module doc.
#[cfg(feature = "q4k-int8-dot")]
pub(super) fn dot_q4k_q8k_block_scalar(weight_block: &[u8], q8k_block: &[u8]) -> f32 {
    let d_weight = f16_le_at(weight_block, Q4K_D_OFFSET);
    let dmin_weight = f16_le_at(weight_block, Q4K_DMIN_OFFSET);
    let mut scales = [0u8; Q4K_SCALE_BYTES];
    scales.copy_from_slice(&weight_block[Q4K_SCALES_OFFSET..Q4K_SCALES_OFFSET + Q4K_SCALE_BYTES]);
    let qs = &weight_block[Q4K_QS_OFFSET..Q4K_QS_OFFSET + Q4K_BLOCK_ELEMENTS / 2];

    let mut d_bytes = [0u8; 4];
    d_bytes.copy_from_slice(&q8k_block[Q8K_D_OFFSET..Q8K_D_OFFSET + 4]);
    let activation_scale = f32::from_le_bytes(d_bytes);
    let activation_qs = &q8k_block[Q8K_QS_OFFSET..Q8K_QS_OFFSET + Q4K_BLOCK_ELEMENTS];
    let bsums = &q8k_block[Q8K_BSUMS_OFFSET..Q8K_BSUMS_OFFSET + Q8K_BSUMS_COUNT * 2];

    let mut sumi = 0i32;
    let mut mins_correction = 0i32;
    for sub_block in 0..Q4K_SUB_BLOCKS {
        let (scale_code, min_code) =
            proxima_gguf::quant::q4_k::get_scale_min_k4(sub_block, &scales);

        let bsum_lo = i16::from_le_bytes([bsums[sub_block * 4], bsums[sub_block * 4 + 1]]);
        let bsum_hi = i16::from_le_bytes([bsums[sub_block * 4 + 2], bsums[sub_block * 4 + 3]]);
        mins_correction += i32::from(bsum_lo + bsum_hi) * i32::from(min_code);

        let byte_base = (sub_block / 2) * 32;
        let is_high_nibble = sub_block % 2 == 1;
        let activation_base = sub_block * 32;
        let mut partial = 0i32;
        for offset in 0..32 {
            let byte = qs[byte_base + offset];
            let nibble = i32::from(if is_high_nibble {
                byte >> 4
            } else {
                byte & 0x0F
            });
            let activation_value = i32::from(activation_qs[activation_base + offset].cast_signed());
            partial += nibble * activation_value;
        }
        sumi += partial * i32::from(scale_code);
    }

    let d = activation_scale * d_weight;
    let dmin = activation_scale * dmin_weight;
    d.mul_add(sumi as f32, -(dmin * mins_correction as f32))
}

/// Issues the ARM `FEAT_DotProd` `sdot` instruction directly via inline
/// asm rather than `core::arch::aarch64::vdotq_s32` -- that safe intrinsic
/// is gated behind the unstable `stdarch_neon_dotprod` feature on this
/// toolchain (probed against `rustc 1.97.1`; ggml's own C `ggml_vdotq_s32`
/// wrapper is the exact analogue this mirrors, `ggml-cpu-impl.h:312-321`).
/// `acc + sum over 4 lanes of (a[4i..4i+4] . b[4i..4i+4])` per output lane,
/// four independent lanes -- the standard armv8.2 `SDOT (vector)` encoding.
///
/// # Safety
/// Caller guarantees `FEAT_DotProd` is available -- this crate's `build.rs`
/// only ever calls this function under the `q4k_dotprod` cfg, which it
/// emits solely for aarch64 targets (see that cfg's doc). Shared by every
/// K-quant codec's `_block_neon_dotprod` kernel (`Q4_K`/`Q5_K`/`Q6_K`), not
/// duplicated per format -- the instruction itself has no codec-specific
/// behavior.
#[cfg(all(
    target_arch = "aarch64",
    any(
        feature = "q4k-int8-dot",
        feature = "q5k-int8-dot",
        feature = "q6k-int8-dot"
    )
))]
#[target_feature(enable = "dotprod")]
#[inline]
pub(super) unsafe fn sdot_s32(
    acc: core::arch::aarch64::int32x4_t,
    a: core::arch::aarch64::int8x16_t,
    b: core::arch::aarch64::int8x16_t,
) -> core::arch::aarch64::int32x4_t {
    // SAFETY: caller-guaranteed FEAT_DotProd (this fn's own doc); operands
    // are NEON vector registers, `options(pure, nomem, nostack)` matches
    // that no memory is touched and the instruction has no side effects.
    unsafe {
        let result: core::arch::aarch64::int32x4_t;
        core::arch::asm!(
            "sdot {result:v}.4s, {a:v}.16b, {b:v}.16b",
            result = inlateout(vreg) acc => result,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack),
        );
        result
    }
}

/// [`dot_q4k_q8k_block_scalar`]'s mechanism, `vdotq_s32`-accelerated:
/// identical per-sub-block structure (unpack scale/min, dot 32 nibbles
/// against 32 `Q8_K` activations, scale, accumulate; mins correction
/// identical), but the 32-nibble dot is two 16-lane `sdot_s32` calls
/// (`ggml_vec_dot_q4_K_q8_K`'s `__ARM_NEON` arm, `arch/arm/quants.c:2408-
/// 2427`) instead of a 32-iteration scalar loop -- low/high nibbles split
/// in-register via `vandq_u8`/`vshrq_n_u8`, never written to memory. The
/// scale/min codes are unpacked ONCE per super-block by the same bit-trick
/// ggml's NEON arm uses (`arch/arm/quants.c:2367-2381`) rather than calling
/// [`proxima_gguf::quant::q4_k::get_scale_min_k4`] per sub-block -- same
/// identity as that function (see its own doc), just the vectorized route
/// instead of the scalar one. The mins correction (`sum(bsums[i] *
/// min_code[i])`) is reduced with `vpaddq_s16`/`vmull_s16`/`vaddvq_s32`
/// mirroring `arch/arm/quants.c:2380-2387`, in place of the auto-vectorized
/// scalar loop this replaced.
///
/// # Safety
/// Caller guarantees `FEAT_DotProd`; `weight_block.len() ==
/// Q4K_BLOCK_BYTES` and `q8k_block.len() == Q8K_BLOCK_BYTES` (both
/// [`dot_q4k_q8k`]'s own `chunks_exact` calls already guarantee before
/// calling this).
#[cfg(all(q4k_dotprod, feature = "q4k-int8-dot"))]
pub(super) unsafe fn dot_q4k_q8k_block_neon_dotprod(weight_block: &[u8], q8k_block: &[u8]) -> f32 {
    let d_weight = f16_le_at(weight_block, Q4K_D_OFFSET);
    let dmin_weight = f16_le_at(weight_block, Q4K_DMIN_OFFSET);
    let mut scales = [0u8; Q4K_SCALE_BYTES];
    scales.copy_from_slice(&weight_block[Q4K_SCALES_OFFSET..Q4K_SCALES_OFFSET + Q4K_SCALE_BYTES]);

    let mut d_bytes = [0u8; 4];
    d_bytes.copy_from_slice(&q8k_block[Q8K_D_OFFSET..Q8K_D_OFFSET + 4]);
    let activation_scale = f32::from_le_bytes(d_bytes);
    let bsums = &q8k_block[Q8K_BSUMS_OFFSET..Q8K_BSUMS_OFFSET + Q8K_BSUMS_COUNT * 2];

    // SAFETY: caller-guaranteed FEAT_DotProd (this fn's own doc);
    // `mins_correction_neon`'s own preconditions (`scales`/`bsums` lengths)
    // are met by the fixed-size array and the slice sized above.
    let (scale_lo, scale_hi, mins_correction) = unsafe { mins_correction_neon(&scales, bsums) };

    // SAFETY: caller-guaranteed FEAT_DotProd; `q4_ptr`/`q8_ptr` each walk
    // exactly `Q4K_BLOCK_ELEMENTS / 2` / `Q4K_BLOCK_ELEMENTS` bytes across
    // the 4 unrolled sub-block pairs below, both within the slices' checked
    // bounds.
    unsafe {
        let m4b = vdupq_n_u8(0x0f);
        let mzero = vdupq_n_s32(0);
        let q4_base = weight_block[Q4K_QS_OFFSET..].as_ptr();
        let q8_base = q8k_block[Q8K_QS_OFFSET..].as_ptr().cast::<i8>();

        let mut sumi1: i32 = 0;
        let mut sumi2: i32 = 0;
        // Hand-unrolled (not `for j in 0..4`): each `2 * j` / `2 * j + 1`
        // scale index below is a literal so `scale_byte` compiles to a
        // single `ubfx` on a register-resident word, matching ggml's
        // `arch/arm/quants.c:2408-2427` `utmp`-in-registers shape, instead
        // of round-tripping a `[u8; 8]` through the stack the way indexing
        // a runtime `j` into an array would.
        macro_rules! sub_block_pair {
            ($q4_offset:expr, $q8_offset:expr, $scale_word:expr) => {{
                let q4bits = vld1q_u8_x2(q4_base.add($q4_offset));
                let lo0 = vreinterpretq_s8_u8(vandq_u8(q4bits.0, m4b));
                let lo1 = vreinterpretq_s8_u8(vandq_u8(q4bits.1, m4b));
                let q8_lo = vld1q_s8_x2(q8_base.add($q8_offset));
                let partial_lo = sdot_s32(sdot_s32(mzero, lo0, q8_lo.0), lo1, q8_lo.1);
                sumi1 += vaddvq_s32(partial_lo) * scale_byte($scale_word, 0);

                let hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.0, 4));
                let hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.1, 4));
                let q8_hi = vld1q_s8_x2(q8_base.add($q8_offset + 32));
                let partial_hi = sdot_s32(sdot_s32(mzero, hi0, q8_hi.0), hi1, q8_hi.1);
                sumi2 += vaddvq_s32(partial_hi) * scale_byte($scale_word, 1);
            }};
        }
        sub_block_pair!(0, 0, scale_lo);
        sub_block_pair!(32, 64, scale_lo >> 16);
        sub_block_pair!(64, 128, scale_hi);
        sub_block_pair!(96, 192, scale_hi >> 16);

        let d = activation_scale * d_weight;
        let dmin = activation_scale * dmin_weight;
        d.mul_add((sumi1 + sumi2) as f32, -(dmin * mins_correction as f32))
    }
}

/// [`dot_q4k_q8k_block_neon_dotprod`]'s scale-unpack and mins-correction
/// step, factored out so the test below can exercise it in isolation
/// against [`proxima_gguf::quant::q4_k::get_scale_min_k4`]'s scalar route
/// to the identical quantity. Unpacks all 8 sub-blocks' 6-bit scale/min
/// codes ONCE via the same bit-trick ggml's NEON arm uses
/// (`arch/arm/quants.c:2367-2381`), then reduces `sum(bsums[i] *
/// min_code[i])` with `vpaddq_s16`/`vmull_s16`/`vaddvq_s32`
/// (`arch/arm/quants.c:2380-2387`) in place of a scalar loop.
///
/// Returns `(scale_lo, scale_hi, mins_correction)`: `scale_lo`/`scale_hi`
/// each pack four sub-blocks' masked scale bytes little-endian (byte `k` of
/// `scale_lo` is sub-block `k`'s scale, byte `k` of `scale_hi` is sub-block
/// `k + 4`'s), matching `get_scale_min_k4(sub_block, scales).0` byte for
/// byte once unpacked via [`scale_byte`]; `mins_correction` is the widened
/// `i32` reduction, matching `sum(get_scale_min_k4(i, scales).1 as i32 *
/// bsum_pair_sum(i) as i32)` for `i in 0..Q4K_SUB_BLOCKS`. Returned as two
/// plain `u32` words rather than a `[u8; 8]` so callers extract each byte
/// with a register-resident `ubfx`-style shift ([`scale_byte`]) instead of
/// indexing an array the compiler may otherwise round-trip through the
/// stack (bounds-check codegen on a non-constant index).
///
/// `Q5_K` shares this exact 12-byte scale/min layout and the same
/// [`Q4K_SUB_BLOCKS`]/`bsums` shape (`arch/arm/quants.c:2611-2622` mirrors
/// `arch/arm/quants.c:2367-2381` byte for byte), so
/// [`dot_q5k_q8k_block_neon_dotprod`] calls this same function directly
/// rather than duplicating it.
///
/// # Safety
/// Caller guarantees `FEAT_DotProd`; `bsums.len() == Q8K_BSUMS_COUNT * 2`
/// (16 `i16`s) so the two 8-lane `vld1q_s16` loads stay in bounds.
#[cfg(all(q4k_dotprod, any(feature = "q4k-int8-dot", feature = "q5k-int8-dot")))]
pub(super) unsafe fn mins_correction_neon(scales: &[u8; Q4K_SCALE_BYTES], bsums: &[u8]) -> (u32, u32, i32) {
    // Same masks as `get_scale_min_k4`'s scalar bit-trick, applied once to
    // the whole 12-byte field instead of once per sub-block per call.
    const KMASK1: u32 = 0x3f3f_3f3f;
    const KMASK2: u32 = 0x0f0f_0f0f;
    const KMASK3: u32 = 0x0303_0303;
    let word_0 = u32::from_le_bytes([scales[0], scales[1], scales[2], scales[3]]);
    let word_1 = u32::from_le_bytes([scales[4], scales[5], scales[6], scales[7]]);
    let word_2 = u32::from_le_bytes([scales[8], scales[9], scales[10], scales[11]]);
    let mins_lo = word_1 & KMASK1;
    let mins_hi = ((word_2 >> 4) & KMASK2) | (((word_1 >> 6) & KMASK3) << 4);
    let scale_hi = (word_2 & KMASK2) | (((word_0 >> 6) & KMASK3) << 4);
    let scale_lo = word_0 & KMASK1;

    // SAFETY: caller-guaranteed FEAT_DotProd; caller-guaranteed
    // `bsums.len() == 32` bytes (16 `i16`s), so the two 8-lane `vld1q_s16`
    // loads below stay in bounds.
    let mins_correction = unsafe {
        let mins_words = [mins_lo, mins_hi];
        let mins8 = vld1_u32(mins_words.as_ptr());
        let mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        let bsums_ptr = bsums.as_ptr().cast::<i16>();
        let q8sums = vpaddq_s16(vld1q_s16(bsums_ptr), vld1q_s16(bsums_ptr.add(8)));
        let mins_product = vaddq_s32(
            vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
            vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)),
        );
        vaddvq_s32(mins_product)
    };

    (scale_lo, scale_hi, mins_correction)
}

/// Extracts byte `index` (`0..=3`) of a [`mins_correction_neon`]-returned
/// `scale_lo`/`scale_hi` word as `i32`, ready to multiply against a
/// `vaddvq_s32` dot-partial. `index` is a literal at
/// [`dot_q4k_q8k_block_neon_dotprod`]'s call sites, so this compiles to one
/// `ubfx` on a register the value already lives in -- ggml's
/// `arch/arm/quants.c` equivalent keeps `utmp` in registers the same way
/// and extracts with the same instruction.
#[cfg(all(q4k_dotprod, any(feature = "q4k-int8-dot", feature = "q5k-int8-dot")))]
#[inline(always)]
pub(super) fn scale_byte(word: u32, index: u32) -> i32 {
    ((word >> (index * 8)) & 0xff) as i32
}

/// Horizontal sum of an `__m256i` holding eight packed `i32` lanes down to
/// one scalar -- the standard AVX2 idiom (extract the high 128 bits, add to
/// the low 128, fold 64-then-32), used by [`dot_q4k_q8k_block_avx2`] to
/// collapse [`_mm256_madd_epi16`]'s eight-lane pairwise-sum result into the
/// same single `i32` partial-dot value [`dot_q4k_q8k_block_scalar`]'s
/// 32-iteration scalar loop accumulates directly.
///
/// # Safety
/// Caller guarantees AVX2 is available -- every intrinsic this function
/// calls (`_mm256_extracti128_si256`/`_mm256_castsi256_si128` need AVX;
/// `_mm_add_epi32`/`_mm_unpackhi_epi64`/`_mm_shuffle_epi32`/
/// `_mm_cvtsi128_si32` are SSE2, x86-64 baseline) is a "safe" intrinsic
/// function under this toolchain's target-feature rules once the enclosing
/// function's `#[target_feature(enable = "avx2")]` statically guarantees
/// the feature, which is why the body below needs no inner `unsafe {}` --
/// this function itself stays `unsafe fn` only so its signature doesn't
/// imply it is callable outside an AVX2-guaranteed build. Compiled on every
/// x86_64 target (`target_arch` gate, not `q4k_avx2`): `dot_q4k_q8k`'s
/// runtime-dispatch arm calls this on a plain `cargo build` x86_64 host too,
/// gated by `std::is_x86_feature_detected!("avx2")` at the call site rather
/// than by a build-time flag -- see `avx2_runtime_available`'s own doc.
#[cfg(all(target_arch = "x86_64", feature = "q4k-int8-dot"))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn hsum_epi32_avx2(v: __m256i) -> i32 {
    let high = _mm256_extracti128_si256(v, 1);
    let low = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(low, high);
    let high64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi32(sum128, high64);
    let high32 = _mm_shuffle_epi32(sum64, 0b01);
    let sum32 = _mm_add_epi32(sum64, high32);
    _mm_cvtsi128_si32(sum32)
}

/// [`dot_q4k_q8k_block_scalar`]'s mechanism, AVX2-accelerated: identical
/// per-sub-block structure (unpack scale/min via `get_scale_min_k4`, dot 32
/// nibbles against 32 `Q8_K` activations, scale, accumulate; mins correction
/// identical), but the 32-nibble dot is one `_mm256_maddubs_epi16` (32-lane
/// unsigned-nibble x signed-`i8` multiply, pairwise-summed to 16 `i16`
/// lanes) followed by `_mm256_madd_epi16` against an all-ones vector
/// (pairwise-summed to 8 `i32` lanes) and [`hsum_epi32_avx2`], instead of a
/// 32-iteration scalar loop -- low/high nibbles split via
/// `_mm256_and_si256`/`_mm256_srli_epi16` exactly as
/// `ggml_vec_dot_q4_K_q8_K`'s `__AVX2__` arm does
/// (`ggml-cpu/arch/x86/quants.c`), but WITHOUT that function's
/// `_mm256_shuffle_epi8`-based scale broadcast: this kernel multiplies each
/// 32-lane partial dot by its scalar `i32` scale code AFTER the horizontal
/// sum, the same order [`dot_q4k_q8k_block_scalar`] uses, rather than
/// folding the scale into the SIMD `madd` itself -- integer multiplication
/// distributes over integer addition exactly, so this is the identical
/// mechanism at the identical resulting value, just without minting a
/// scale-shuffle table this component doesn't otherwise need.
///
/// # Safety
/// Caller guarantees AVX2 is available; `weight_block.len() ==
/// Q4K_BLOCK_BYTES` and `q8k_block.len() == Q8K_BLOCK_BYTES` (both
/// [`dot_q4k_q8k`]'s own `chunks_exact` calls already guarantee before
/// calling this). Compiled on every x86_64 target -- see
/// [`hsum_epi32_avx2`]'s own doc for why this is `target_arch`-gated rather
/// than `q4k_avx2`-gated.
#[cfg(all(target_arch = "x86_64", feature = "q4k-int8-dot"))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn dot_q4k_q8k_block_avx2(weight_block: &[u8], q8k_block: &[u8]) -> f32 {
    let d_weight = f16_le_at(weight_block, Q4K_D_OFFSET);
    let dmin_weight = f16_le_at(weight_block, Q4K_DMIN_OFFSET);
    let mut scales = [0u8; Q4K_SCALE_BYTES];
    scales.copy_from_slice(&weight_block[Q4K_SCALES_OFFSET..Q4K_SCALES_OFFSET + Q4K_SCALE_BYTES]);

    let mut d_bytes = [0u8; 4];
    d_bytes.copy_from_slice(&q8k_block[Q8K_D_OFFSET..Q8K_D_OFFSET + 4]);
    let activation_scale = f32::from_le_bytes(d_bytes);
    let bsums = &q8k_block[Q8K_BSUMS_OFFSET..Q8K_BSUMS_OFFSET + Q8K_BSUMS_COUNT * 2];

    let mut mins_correction = 0i32;
    for sub_block in 0..Q4K_SUB_BLOCKS {
        let (_, min_code) = proxima_gguf::quant::q4_k::get_scale_min_k4(sub_block, &scales);
        let bsum_lo = i16::from_le_bytes([bsums[sub_block * 4], bsums[sub_block * 4 + 1]]);
        let bsum_hi = i16::from_le_bytes([bsums[sub_block * 4 + 2], bsums[sub_block * 4 + 3]]);
        mins_correction += i32::from(bsum_lo + bsum_hi) * i32::from(min_code);
    }

    // SAFETY: caller-guaranteed AVX2; `q4_base`/`q8_base` each walk exactly
    // `Q4K_BLOCK_ELEMENTS / 2` / `Q4K_BLOCK_ELEMENTS` bytes across the
    // `Q4K_SUB_BLOCKS / 2` loop iterations below, both within the slices'
    // checked bounds (`_mm256_loadu_si256` needs no alignment).
    unsafe {
        let m4 = _mm256_set1_epi8(0x0f);
        let ones = _mm256_set1_epi16(1);
        let q4_base = weight_block[Q4K_QS_OFFSET..].as_ptr();
        let q8_base = q8k_block[Q8K_QS_OFFSET..].as_ptr().cast::<i8>();

        let mut sumi = 0i32;
        for j in 0..Q4K_SUB_BLOCKS / 2 {
            let q4bits = _mm256_loadu_si256(q4_base.add(j * 32).cast());
            let q4_lo = _mm256_and_si256(q4bits, m4);
            let q4_hi = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            let q8_lo = _mm256_loadu_si256(q8_base.add(j * 64).cast());
            let dot_lo = _mm256_madd_epi16(_mm256_maddubs_epi16(q4_lo, q8_lo), ones);
            let scale_lo = proxima_gguf::quant::q4_k::get_scale_min_k4(2 * j, &scales).0;
            sumi += hsum_epi32_avx2(dot_lo) * i32::from(scale_lo);

            let q8_hi = _mm256_loadu_si256(q8_base.add(j * 64 + 32).cast());
            let dot_hi = _mm256_madd_epi16(_mm256_maddubs_epi16(q4_hi, q8_hi), ones);
            let scale_hi = proxima_gguf::quant::q4_k::get_scale_min_k4(2 * j + 1, &scales).0;
            sumi += hsum_epi32_avx2(dot_hi) * i32::from(scale_hi);
        }

        let d = activation_scale * d_weight;
        let dmin = activation_scale * dmin_weight;
        d.mul_add(sumi as f32, -(dmin * mins_correction as f32))
    }
}

/// A full `Q4_K`-quantized weight matrix (`rows` x `k`) times one `f32`
/// activation vector -- [`matmul_q4k_f32`]'s packed-arithmetic sibling.
/// Quantizes `activation` to `Q8_K` exactly once ([`quantize_row_q8k`],
/// this function's own doc note on why: hoisted out of the row loop), then
/// calls [`dot_q4k_q8k`] per row against that one shared quantized buffer.
///
/// # Errors
/// Propagates [`quantize_row_q8k`]'s and [`dot_q4k_q8k`]'s
/// [`TensorError::QuantizedShapeMismatch`], or reports the same error if
/// `weights.len()` is not a whole multiple of `rows`.
#[cfg(feature = "q4k-int8-dot")]
pub fn matmul_q4k_q8k_f32(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
) -> Result<Vec<f32>, TensorError> {
    matmul_q4k_q8k_f32_impl(weights, rows, activation, 1, None)
}

/// Benchable entry point for the multi-position (`leading_total > 1`) wide
/// fold `run_reduce_quantized` already calls at real `leading_total` scale
/// for prefill (`cpu.rs:7093`) -- same `matmul_q4k_q8k_f32_impl`,
/// `session = None`, no new logic. Exists only so a bench/example outside
/// this module can measure the exact kernel production prefill dispatches
/// to, instead of approximating it with a `leading_total`-times loop over
/// [`matmul_q4k_q8k_f32`].
///
/// # Errors
/// Same as [`matmul_q4k_q8k_f32`], plus `leading_total == 0` or
/// `activation.len()` not a whole multiple of `leading_total`.
#[cfg(feature = "q4k-int8-dot")]
pub fn matmul_q4k_q8k_f32_wide(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
    leading_total: usize,
) -> Result<Vec<f32>, TensorError> {
    matmul_q4k_q8k_f32_impl(weights, rows, activation, leading_total, None)
}

/// [`matmul_q4k_q8k_f32`]'s body, plus `leading_total` (the sequence-position
/// count [`run_reduce_quantized`] already derives as `activation.len() / k`
/// at `cpu.rs:2179`) and the [`CohortSession`] a caller already inside a
/// forward pass's session can supply so `matmul_rows_threaded` dispatches
/// through the cohort instead of `nest_pool`. `matmul_q4k_q8k_f32` itself
/// stays the stable 3-argument public entry point (`leading_total = 1`,
/// `session = None`, unchanged call sites in every bench/test);
/// [`run_reduce_quantized`] is the only caller that passes `leading_total >
/// 1` or a session.
///
/// `activation` stays one contiguous position-major `&[f32]` of
/// `leading_total * k` elements — the same buffer [`run_reduce_quantized`]
/// already holds, not a `&[&[f32]]` or a generic batch type. Quantizing it
/// once here (a single [`quantize_row_q8k`] call over the whole buffer,
/// since every `Q8_K` super-block is 256 elements and `k` is always a whole
/// multiple of that, no super-block ever straddles a position boundary)
/// means each weight row's bytes are read once and its dot reused across
/// every position, instead of the weight stream being re-read once per
/// position the way a `leading_total`-times loop over the narrow
/// 3-argument entry point would.
#[cfg(feature = "q4k-int8-dot")]
pub(super) fn matmul_q4k_q8k_f32_impl(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
    leading_total: usize,
    session: Option<&MatmulSession<'_>>,
) -> Result<Vec<f32>, TensorError> {
    if rows == 0 {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "matmul_q4k_q8k_f32 called with zero rows",
        });
    }
    if !weights.len().is_multiple_of(rows) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight byte length is not a whole multiple of the row count",
        });
    }
    if leading_total == 0 || !activation.len().is_multiple_of(leading_total) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length is not a whole multiple of the position count",
        });
    }
    let k = activation.len() / leading_total;
    if !k.is_multiple_of(Q4K_BLOCK_ELEMENTS) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length is not a whole multiple of the q8_k super-block size",
        });
    }
    let block_count = activation.len() / Q4K_BLOCK_ELEMENTS;
    let q8k_row_bytes = (k / Q4K_BLOCK_ELEMENTS) * Q8K_BLOCK_BYTES;
    let mut activation_q8k = vec![0u8; block_count * Q8K_BLOCK_BYTES];
    // proxima-debugger diagnostic: this preamble runs BEFORE
    // `quantized_matmul_workers`/`matmul_rows_threaded`, so none of the
    // spawn/own-chunk/recv-wait timers in `matmul_rows_threaded` see it --
    // timed separately to settle whether it is the source of the gap
    // between a matmul node's total wall time and its threaded-dispatch
    // time.
    #[cfg(feature = "instrument")]
    let diag_quantize_started = instrument::read_ticks();
    quantize_row_q8k_dispatch(activation, &mut activation_q8k, session)?;
    #[cfg(feature = "instrument")]
    counter!(
        instrument::MATMUL_QUANTIZE_ACTIVATION_TICKS,
        instrument::elapsed_ticks(diag_quantize_started)
    );

    let row_bytes = weights.len() / rows;
    match quantized_matmul_workers(rows, activation.len()) {
        Some(workers) => {
            matmul_rows_threaded(rows, leading_total, workers, session, k, |row, slot| {
                let start = row * row_bytes;
                let weight_row = &weights[start..start + row_bytes];
                for (position, output_slot) in slot.iter_mut().enumerate() {
                    let q8k_start = position * q8k_row_bytes;
                    *output_slot = dot_q4k_q8k(
                        weight_row,
                        &activation_q8k[q8k_start..q8k_start + q8k_row_bytes],
                    )?;
                }
                Ok(())
            })
        }
        None => weights.chunks_exact(row_bytes).try_fold(
            Vec::with_capacity(rows * leading_total),
            |mut output, weight_row| {
                for position in 0..leading_total {
                    let q8k_start = position * q8k_row_bytes;
                    output.push(dot_q4k_q8k(
                        weight_row,
                        &activation_q8k[q8k_start..q8k_start + q8k_row_bytes],
                    )?);
                }
                Ok::<Vec<f32>, TensorError>(output)
            },
        ),
    }
}

/// [`matmul_q4k_q8k_f32`] with every row routed through
/// [`dot_q4k_q8k_portable`] instead of [`dot_q4k_q8k`] -- the matrix-level
/// counterpart of that function's own doc: the standalone "portable
/// packing alone" measurement, callable (and benchable) on any host
/// regardless of which accelerated arm that host's build would otherwise
/// pick.
///
/// # Errors
/// Same as [`matmul_q4k_q8k_f32`].
#[cfg(feature = "q4k-int8-dot")]
pub fn matmul_q4k_q8k_portable_f32(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
) -> Result<Vec<f32>, TensorError> {
    if rows == 0 {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "matmul_q4k_q8k_portable_f32 called with zero rows",
        });
    }
    if !weights.len().is_multiple_of(rows) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight byte length is not a whole multiple of the row count",
        });
    }
    if !activation.len().is_multiple_of(Q4K_BLOCK_ELEMENTS) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length is not a whole multiple of the q8_k super-block size",
        });
    }
    let block_count = activation.len() / Q4K_BLOCK_ELEMENTS;
    let mut activation_q8k = vec![0u8; block_count * Q8K_BLOCK_BYTES];
    quantize_row_q8k(activation, &mut activation_q8k)?;

    let row_bytes = weights.len() / rows;
    weights
        .chunks_exact(row_bytes)
        .map(|weight_row| dot_q4k_q8k_portable(weight_row, &activation_q8k))
        .collect()
}

// ---------------------------------------------------------------------
// `q5k-int8-dot` (default-off): `q4k-int8-dot`'s mechanism applied to
// `Q5_K` -- packed int8 dot directly against `Q8_K`, no `[f32; 256]`
// dequantize pass. `Q5_K` shares `Q4_K`'s exact super-block/sub-block
// shape (8 sub-blocks of 32, the same bit-interleaved 6-bit scale/min
// packing -- `get_scale_min_k4` above is reused unchanged) plus one
// extra `qh` high-bit plane; see `proxima-tensor/docs/discipline.md` for
// the row this landed under.
// ---------------------------------------------------------------------

/// Byte offsets into one packed `Q5_K` super-block ([`Q5K_BLOCK_BYTES`]
/// bytes), mirroring `proxima_gguf::quant::q5_k`'s private layout
/// constants -- duplicated here for the same reason [`Q4K_D_OFFSET`] and
/// siblings are: [`dot_q5k_q8k`] reads the raw bytes directly rather than
/// calling `dequantize_block`.
#[cfg(feature = "q5k-int8-dot")]
pub(super) const Q5K_D_OFFSET: usize = 0;
#[cfg(feature = "q5k-int8-dot")]
pub(super) const Q5K_DMIN_OFFSET: usize = 2;
#[cfg(feature = "q5k-int8-dot")]
pub(super) const Q5K_SCALES_OFFSET: usize = 4;
/// `qh` sits between `scales` and `qs` in `Q5_K`'s on-disk layout
/// (`proxima_gguf::quant::q5_k`'s own module doc, ported from
/// `ggml-common.h:302-313`) -- unlike `Q4_K`, which has no high-bit plane
/// at all.
#[cfg(feature = "q5k-int8-dot")]
pub(super) const Q5K_QH_OFFSET: usize = Q5K_SCALES_OFFSET + Q4K_SCALE_BYTES;
#[cfg(feature = "q5k-int8-dot")]
pub(super) const Q5K_QH_BYTES: usize = Q4K_BLOCK_ELEMENTS / 8;
#[cfg(feature = "q5k-int8-dot")]
pub(super) const Q5K_QS_OFFSET: usize = Q5K_QH_OFFSET + Q5K_QH_BYTES;

/// One `Q5_K`-weight-row x `Q8_K`-activation int8 dot product --
/// [`dot_q4k_q8k`]'s sibling for the 5-bit codec. Dispatches to
/// `dot_q5k_q8k_block_neon_dotprod` under `q4k_dotprod` (the same
/// arch-wide cfg [`dot_q4k_q8k`] keys off -- `FEAT_DotProd` availability is
/// a property of the target, not the weight codec) and to the portable
/// `dot_q5k_q8k_block_scalar` everywhere else. No AVX2 arm yet -- the
/// task ordering this landed under ran portable-then-aarch64 first; an
/// AVX2 arm is future work, not a correctness gap (the portable arm is
/// what an x86-64 build without `+avx2` runs regardless).
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of `Q5K_BLOCK_BYTES`, or `activation_q8k.len()` does
/// not equal the row's block count times `Q8K_BLOCK_BYTES`.
#[cfg(feature = "q5k-int8-dot")]
pub fn dot_q5k_q8k(weight_row: &[u8], activation_q8k: &[u8]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q5K_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q5_k block size",
        });
    }
    let block_count = weight_row.len() / Q5K_BLOCK_BYTES;
    if activation_q8k.len() != block_count * Q8K_BLOCK_BYTES {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "q8_k activation length does not match the weight row's block count",
        });
    }

    let mut acc = 0.0f32;
    for (weight_block, q8k_block) in weight_row
        .as_chunks::<Q5K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation_q8k.as_chunks::<Q8K_BLOCK_BYTES>().0)
    {
        #[cfg(q4k_dotprod)]
        // SAFETY: `q4k_dotprod` is emitted by build.rs only for aarch64
        // targets, all of which carry FEAT_DotProd.
        let block_sum = unsafe { dot_q5k_q8k_block_neon_dotprod(weight_block, q8k_block) };
        #[cfg(not(q4k_dotprod))]
        let block_sum = dot_q5k_q8k_block_scalar(weight_block, q8k_block);
        acc += block_sum;
    }
    Ok(acc)
}

/// [`dot_q5k_q8k`] with the dispatch forced to
/// `dot_q5k_q8k_block_scalar` regardless of `q4k_dotprod` -- the
/// standalone "portable packing alone" measurement, same role
/// [`dot_q4k_q8k_portable`] plays for `Q4_K`.
///
/// # Errors
/// Same as [`dot_q5k_q8k`].
#[cfg(feature = "q5k-int8-dot")]
pub fn dot_q5k_q8k_portable(weight_row: &[u8], activation_q8k: &[u8]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q5K_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q5_k block size",
        });
    }
    let block_count = weight_row.len() / Q5K_BLOCK_BYTES;
    if activation_q8k.len() != block_count * Q8K_BLOCK_BYTES {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "q8_k activation length does not match the weight row's block count",
        });
    }

    let mut acc = 0.0f32;
    for (weight_block, q8k_block) in weight_row
        .as_chunks::<Q5K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation_q8k.as_chunks::<Q8K_BLOCK_BYTES>().0)
    {
        acc += dot_q5k_q8k_block_scalar(weight_block, q8k_block);
    }
    Ok(acc)
}

/// The portable packed-nibble x `Q8_K` int8 dot for `Q5_K` -- no
/// dequantize pass, no `f32` intermediate. Identical structure to
/// [`dot_q4k_q8k_block_scalar`] (unpack each sub-block's 6-bit
/// scale/min via [`proxima_gguf::quant::q4_k::get_scale_min_k4`], dot 32
/// nibbles against 32 `Q8_K` activations, scale, accumulate; mins
/// correction identical -- `Q5_K` shares `Q4_K`'s exact bit-interleaved
/// scale/min packing) plus one addition: each nibble is OR'd with its
/// `qh` high bit before the multiply. `qh_mask = 1u8 << sub_block` is not
/// an approximation -- it is the exact bit `proxima_gguf::quant::q5_k`'s
/// own `dequantize_block` reads for this `sub_block` (derived from that
/// function's `mask_lo`/`mask_hi` cycling, which starts at
/// `1u8`/`2u8` and shifts left by 2 every 64-element chunk: the two
/// sub-blocks sharing a chunk land on consecutive bits, and consecutive
/// chunks land on the next two bits up, i.e. bit index == `sub_block`
/// exactly, for every one of the 8 sub-blocks).
#[cfg(feature = "q5k-int8-dot")]
pub(super) fn dot_q5k_q8k_block_scalar(weight_block: &[u8], q8k_block: &[u8]) -> f32 {
    let d_weight = f16_le_at(weight_block, Q5K_D_OFFSET);
    let dmin_weight = f16_le_at(weight_block, Q5K_DMIN_OFFSET);
    let mut scales = [0u8; Q4K_SCALE_BYTES];
    scales.copy_from_slice(&weight_block[Q5K_SCALES_OFFSET..Q5K_SCALES_OFFSET + Q4K_SCALE_BYTES]);
    let qh = &weight_block[Q5K_QH_OFFSET..Q5K_QH_OFFSET + Q5K_QH_BYTES];
    let qs = &weight_block[Q5K_QS_OFFSET..Q5K_QS_OFFSET + Q4K_BLOCK_ELEMENTS / 2];

    let mut d_bytes = [0u8; 4];
    d_bytes.copy_from_slice(&q8k_block[Q8K_D_OFFSET..Q8K_D_OFFSET + 4]);
    let activation_scale = f32::from_le_bytes(d_bytes);
    let activation_qs = &q8k_block[Q8K_QS_OFFSET..Q8K_QS_OFFSET + Q4K_BLOCK_ELEMENTS];
    let bsums = &q8k_block[Q8K_BSUMS_OFFSET..Q8K_BSUMS_OFFSET + Q8K_BSUMS_COUNT * 2];

    let mut sumi = 0i32;
    let mut mins_correction = 0i32;
    for sub_block in 0..Q4K_SUB_BLOCKS {
        let (scale_code, min_code) =
            proxima_gguf::quant::q4_k::get_scale_min_k4(sub_block, &scales);

        let bsum_lo = i16::from_le_bytes([bsums[sub_block * 4], bsums[sub_block * 4 + 1]]);
        let bsum_hi = i16::from_le_bytes([bsums[sub_block * 4 + 2], bsums[sub_block * 4 + 3]]);
        mins_correction += i32::from(bsum_lo + bsum_hi) * i32::from(min_code);

        let byte_base = (sub_block / 2) * 32;
        let is_high_nibble = sub_block % 2 == 1;
        let activation_base = sub_block * 32;
        let qh_mask = 1u8 << sub_block;
        let mut partial = 0i32;
        for offset in 0..32 {
            let byte = qs[byte_base + offset];
            let nibble = i32::from(if is_high_nibble {
                byte >> 4
            } else {
                byte & 0x0F
            });
            let high_bit = i32::from(qh[offset] & qh_mask != 0) * 16;
            let level = nibble + high_bit;
            let activation_value = i32::from(activation_qs[activation_base + offset].cast_signed());
            partial += level * activation_value;
        }
        sumi += partial * i32::from(scale_code);
    }

    let d = activation_scale * d_weight;
    let dmin = activation_scale * dmin_weight;
    d.mul_add(sumi as f32, -(dmin * mins_correction as f32))
}

/// [`dot_q5k_q8k_block_scalar`]'s mechanism, `vdotq_s32`-accelerated.
/// Ports `ggml_vec_dot_q5_K_q8_K`'s `__ARM_NEON` arm
/// (`arch/arm/quants.c:2512-2579`) directly: per 64-element chunk (`j` in
/// `0..4`), extracts the current chunk's two high-bit planes from the
/// (persistently right-shifted) `qh` register pair via
/// `vandq_u8`/`vshlq_n_u8` with `mone`/`mtwo` masks, ORs each into its
/// nibble half, then two [`sdot_s32`] pairs per chunk (low nibble pair,
/// high nibble pair) instead of [`dot_q5k_q8k_block_scalar`]'s
/// 32-iteration scalar loop per sub-block. Scale/min unpack routes through
/// [`mins_correction_neon`] -- the same once-per-super-block NEON bit-trick
/// [`dot_q4k_q8k_block_neon_dotprod`] uses, since `Q5_K`'s 12-byte
/// scale/min field is byte-identical in layout -- in place of the 16 scalar
/// `get_scale_min_k4` calls (8 for the mins correction, 8 more inside this
/// loop for `scale_lo`/`scale_hi`) that path used to make per block.
///
/// # Safety
/// Caller guarantees `FEAT_DotProd`; `weight_block.len() ==
/// Q5K_BLOCK_BYTES` and `q8k_block.len() == Q8K_BLOCK_BYTES` (both
/// [`dot_q5k_q8k`]'s own `chunks_exact` calls already guarantee before
/// calling this).
#[cfg(all(q4k_dotprod, feature = "q5k-int8-dot"))]
pub(super) unsafe fn dot_q5k_q8k_block_neon_dotprod(weight_block: &[u8], q8k_block: &[u8]) -> f32 {
    let d_weight = f16_le_at(weight_block, Q5K_D_OFFSET);
    let dmin_weight = f16_le_at(weight_block, Q5K_DMIN_OFFSET);
    let mut scales = [0u8; Q4K_SCALE_BYTES];
    scales.copy_from_slice(&weight_block[Q5K_SCALES_OFFSET..Q5K_SCALES_OFFSET + Q4K_SCALE_BYTES]);

    let mut d_bytes = [0u8; 4];
    d_bytes.copy_from_slice(&q8k_block[Q8K_D_OFFSET..Q8K_D_OFFSET + 4]);
    let activation_scale = f32::from_le_bytes(d_bytes);
    let bsums = &q8k_block[Q8K_BSUMS_OFFSET..Q8K_BSUMS_OFFSET + Q8K_BSUMS_COUNT * 2];

    // SAFETY: caller-guaranteed FEAT_DotProd (this fn's own doc);
    // `mins_correction_neon`'s own preconditions (`scales`/`bsums` lengths)
    // are met by the fixed-size array and the slice sized above.
    let (scale_lo, scale_hi, mins_correction) = unsafe { mins_correction_neon(&scales, bsums) };

    // SAFETY: caller-guaranteed FEAT_DotProd; `q5_base` walks exactly
    // `Q4K_BLOCK_ELEMENTS / 2` bytes, `qh_base` is read once (32 bytes,
    // never advanced) and `q8_base` walks exactly `Q4K_BLOCK_ELEMENTS`
    // bytes, across the `Q4K_SUB_BLOCKS / 2` loop iterations below -- all
    // within the slices' checked bounds.
    unsafe {
        let m4b = vdupq_n_u8(0x0f);
        let mone = vdupq_n_u8(1);
        let mtwo = vdupq_n_u8(2);
        let mzero = vdupq_n_s32(0);
        let q5_base = weight_block[Q5K_QS_OFFSET..].as_ptr();
        let qh_base = weight_block[Q5K_QH_OFFSET..].as_ptr();
        let q8_base = q8k_block[Q8K_QS_OFFSET..].as_ptr().cast::<i8>();

        let mut qhbits0 = vld1q_u8(qh_base);
        let mut qhbits1 = vld1q_u8(qh_base.add(16));

        let mut sumi: i32 = 0;
        for j in 0..Q4K_SUB_BLOCKS / 2 {
            let q5bits0 = vld1q_u8(q5_base.add(j * 32));
            let q5bits1 = vld1q_u8(q5_base.add(j * 32 + 16));

            let q5h0 = vshlq_n_u8(vandq_u8(mone, qhbits0), 4);
            let q5h1 = vshlq_n_u8(vandq_u8(mone, qhbits1), 4);
            let q5h2 = vshlq_n_u8(vandq_u8(mtwo, qhbits0), 3);
            let q5h3 = vshlq_n_u8(vandq_u8(mtwo, qhbits1), 3);
            qhbits0 = vshrq_n_u8(qhbits0, 2);
            qhbits1 = vshrq_n_u8(qhbits1, 2);

            let q5bytes0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits0, m4b), q5h0));
            let q5bytes1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits1, m4b), q5h1));
            let q5bytes2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits0, 4), q5h2));
            let q5bytes3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits1, 4), q5h3));

            let q8b0 = vld1q_s8(q8_base.add(j * 64));
            let q8b1 = vld1q_s8(q8_base.add(j * 64 + 16));
            let q8b2 = vld1q_s8(q8_base.add(j * 64 + 32));
            let q8b3 = vld1q_s8(q8_base.add(j * 64 + 48));

            let scale_word = if j < 2 { scale_lo } else { scale_hi };
            let scale_shift = (j % 2) as u32 * 2;

            let partial_lo = sdot_s32(sdot_s32(mzero, q5bytes0, q8b0), q5bytes1, q8b1);
            sumi += vaddvq_s32(partial_lo) * scale_byte(scale_word, scale_shift);

            let partial_hi = sdot_s32(sdot_s32(mzero, q5bytes2, q8b2), q5bytes3, q8b3);
            sumi += vaddvq_s32(partial_hi) * scale_byte(scale_word, scale_shift + 1);
        }

        let d = activation_scale * d_weight;
        let dmin = activation_scale * dmin_weight;
        d.mul_add(sumi as f32, -(dmin * mins_correction as f32))
    }
}

/// A full `Q5_K`-quantized weight matrix (`rows` x `k`) times one `f32`
/// activation vector — [`matmul_q5k_f32`]'s packed-arithmetic sibling,
/// same structure as [`matmul_q4k_q8k_f32`].
///
/// # Errors
/// Propagates [`quantize_row_q8k`]'s and [`dot_q5k_q8k`]'s
/// [`TensorError::QuantizedShapeMismatch`], or reports the same error if
/// `weights.len()` is not a whole multiple of `rows`.
#[cfg(feature = "q5k-int8-dot")]
pub fn matmul_q5k_q8k_f32(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
) -> Result<Vec<f32>, TensorError> {
    matmul_q5k_q8k_f32_impl(weights, rows, activation, 1, None)
}

/// [`matmul_q5k_q8k_f32`]'s body, plus `leading_total` (the sequence-position
/// count [`run_reduce_quantized`] derives as `activation.len() / k`) and the
/// [`CohortSession`] a caller already inside a forward pass's session can
/// supply — identical shape to [`matmul_q4k_q8k_f32_impl`]: `activation` is
/// one contiguous position-major `&[f32]` of `leading_total * k` elements,
/// quantized to `Q8_K` once, so each weight row's bytes are read once and its
/// dot reused across every position instead of the weight stream being
/// re-read once per position.
#[cfg(feature = "q5k-int8-dot")]
pub(super) fn matmul_q5k_q8k_f32_impl(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
    leading_total: usize,
    session: Option<&MatmulSession<'_>>,
) -> Result<Vec<f32>, TensorError> {
    if rows == 0 {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "matmul_q5k_q8k_f32 called with zero rows",
        });
    }
    if !weights.len().is_multiple_of(rows) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight byte length is not a whole multiple of the row count",
        });
    }
    if leading_total == 0 || !activation.len().is_multiple_of(leading_total) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length is not a whole multiple of the position count",
        });
    }
    let k = activation.len() / leading_total;
    if !k.is_multiple_of(Q4K_BLOCK_ELEMENTS) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length is not a whole multiple of the q8_k super-block size",
        });
    }
    let block_count = activation.len() / Q4K_BLOCK_ELEMENTS;
    let q8k_row_bytes = (k / Q4K_BLOCK_ELEMENTS) * Q8K_BLOCK_BYTES;
    let mut activation_q8k = vec![0u8; block_count * Q8K_BLOCK_BYTES];
    quantize_row_q8k_dispatch(activation, &mut activation_q8k, session)?;

    let row_bytes = weights.len() / rows;
    match quantized_matmul_workers(rows, activation.len()) {
        Some(workers) => {
            matmul_rows_threaded(rows, leading_total, workers, session, k, |row, slot| {
                let start = row * row_bytes;
                let weight_row = &weights[start..start + row_bytes];
                for (position, output_slot) in slot.iter_mut().enumerate() {
                    let q8k_start = position * q8k_row_bytes;
                    *output_slot = dot_q5k_q8k(
                        weight_row,
                        &activation_q8k[q8k_start..q8k_start + q8k_row_bytes],
                    )?;
                }
                Ok(())
            })
        }
        None => weights.chunks_exact(row_bytes).try_fold(
            Vec::with_capacity(rows * leading_total),
            |mut output, weight_row| {
                for position in 0..leading_total {
                    let q8k_start = position * q8k_row_bytes;
                    output.push(dot_q5k_q8k(
                        weight_row,
                        &activation_q8k[q8k_start..q8k_start + q8k_row_bytes],
                    )?);
                }
                Ok::<Vec<f32>, TensorError>(output)
            },
        ),
    }
}

/// [`matmul_q5k_q8k_f32`] with every row routed through
/// [`dot_q5k_q8k_portable`] instead of [`dot_q5k_q8k`] -- the matrix-level
/// "portable packing alone" measurement, callable regardless of which
/// accelerated arm the host build would otherwise pick.
///
/// # Errors
/// Same as [`matmul_q5k_q8k_f32`].
#[cfg(feature = "q5k-int8-dot")]
pub fn matmul_q5k_q8k_portable_f32(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
) -> Result<Vec<f32>, TensorError> {
    if rows == 0 {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "matmul_q5k_q8k_portable_f32 called with zero rows",
        });
    }
    if !weights.len().is_multiple_of(rows) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight byte length is not a whole multiple of the row count",
        });
    }
    if !activation.len().is_multiple_of(Q4K_BLOCK_ELEMENTS) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length is not a whole multiple of the q8_k super-block size",
        });
    }
    let block_count = activation.len() / Q4K_BLOCK_ELEMENTS;
    let mut activation_q8k = vec![0u8; block_count * Q8K_BLOCK_BYTES];
    quantize_row_q8k(activation, &mut activation_q8k)?;

    let row_bytes = weights.len() / rows;
    weights
        .chunks_exact(row_bytes)
        .map(|weight_row| dot_q5k_q8k_portable(weight_row, &activation_q8k))
        .collect()
}

// ---------------------------------------------------------------------
// `q6k-int8-dot` (default-off): `q4k-int8-dot`'s mechanism applied to
// `Q6_K` -- packed int8 dot directly against `Q8_K`. `Q6_K` has a
// DIFFERENT super-block shape from `Q4_K`/`Q5_K`: 16 sub-blocks of 16
// (not 8 of 32), one signed 8-bit scale per sub-block, no `dmin` term at
// all (`x = d*sc*(q-32)`, `proxima_gguf::quant::q6_k`'s own module doc).
// See `proxima-tensor/docs/discipline.md` for the row this landed under.
// ---------------------------------------------------------------------

/// Byte offsets into one packed `Q6_K` super-block ([`Q6K_BLOCK_BYTES`]
/// bytes), mirroring `proxima_gguf::quant::q6_k`'s private layout
/// constants -- duplicated here for the same reason [`Q4K_D_OFFSET`] and
/// siblings are. Note the field order: `d` TRAILS the block here (unlike
/// `Q4_K`/`Q5_K`, where it leads) -- `proxima_gguf::quant::q6_k`'s own
/// module doc flags this explicitly as the one layout trap this codec has
/// that the others don't.
#[cfg(feature = "q6k-int8-dot")]
pub(super) const Q6K_QL_OFFSET: usize = 0;
#[cfg(feature = "q6k-int8-dot")]
pub(super) const Q6K_QL_BYTES: usize = Q4K_BLOCK_ELEMENTS / 2;
#[cfg(feature = "q6k-int8-dot")]
pub(super) const Q6K_QH_OFFSET: usize = Q6K_QL_OFFSET + Q6K_QL_BYTES;
#[cfg(feature = "q6k-int8-dot")]
pub(super) const Q6K_QH_BYTES: usize = Q4K_BLOCK_ELEMENTS / 4;
#[cfg(feature = "q6k-int8-dot")]
pub(super) const Q6K_SCALES_OFFSET: usize = Q6K_QH_OFFSET + Q6K_QH_BYTES;
#[cfg(feature = "q6k-int8-dot")]
pub(super) const Q6K_D_OFFSET: usize = Q6K_SCALES_OFFSET + proxima_gguf::quant::q6_k::SUB_BLOCKS;

/// One `Q6_K`-weight-row x `Q8_K`-activation int8 dot product --
/// [`dot_q4k_q8k`]'s sibling for the 6-bit codec. Dispatches to
/// `dot_q6k_q8k_block_neon_dotprod` under `q4k_dotprod` and to the
/// portable `dot_q6k_q8k_block_scalar` everywhere else -- same dispatch
/// shape as [`dot_q5k_q8k`], no AVX2 arm yet (same rationale: portable
/// arm first, aarch64 second, per this landing's task ordering).
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of `Q6K_BLOCK_BYTES`, or `activation_q8k.len()` does
/// not equal the row's block count times `Q8K_BLOCK_BYTES`.
#[cfg(feature = "q6k-int8-dot")]
pub fn dot_q6k_q8k(weight_row: &[u8], activation_q8k: &[u8]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q6K_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q6_k block size",
        });
    }
    let block_count = weight_row.len() / Q6K_BLOCK_BYTES;
    if activation_q8k.len() != block_count * Q8K_BLOCK_BYTES {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "q8_k activation length does not match the weight row's block count",
        });
    }

    let mut acc = 0.0f32;
    for (weight_block, q8k_block) in weight_row
        .as_chunks::<Q6K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation_q8k.as_chunks::<Q8K_BLOCK_BYTES>().0)
    {
        #[cfg(q4k_dotprod)]
        // SAFETY: `q4k_dotprod` is emitted by build.rs only for aarch64
        // targets, all of which carry FEAT_DotProd.
        let block_sum = unsafe { dot_q6k_q8k_block_neon_dotprod(weight_block, q8k_block) };
        #[cfg(not(q4k_dotprod))]
        let block_sum = dot_q6k_q8k_block_scalar(weight_block, q8k_block);
        acc += block_sum;
    }
    Ok(acc)
}

/// [`dot_q6k_q8k`] with the dispatch forced to
/// `dot_q6k_q8k_block_scalar` regardless of `q4k_dotprod`.
///
/// # Errors
/// Same as [`dot_q6k_q8k`].
#[cfg(feature = "q6k-int8-dot")]
pub fn dot_q6k_q8k_portable(weight_row: &[u8], activation_q8k: &[u8]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q6K_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q6_k block size",
        });
    }
    let block_count = weight_row.len() / Q6K_BLOCK_BYTES;
    if activation_q8k.len() != block_count * Q8K_BLOCK_BYTES {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "q8_k activation length does not match the weight row's block count",
        });
    }

    let mut acc = 0.0f32;
    for (weight_block, q8k_block) in weight_row
        .as_chunks::<Q6K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation_q8k.as_chunks::<Q8K_BLOCK_BYTES>().0)
    {
        acc += dot_q6k_q8k_block_scalar(weight_block, q8k_block);
    }
    Ok(acc)
}

/// The portable packed-nibble x `Q8_K` int8 dot for `Q6_K` -- no
/// dequantize pass, no `f32` intermediate. Unlike [`dot_q4k_q8k_block_scalar`]/
/// [`dot_q5k_q8k_block_scalar`], this does not reuse those codecs'
/// `get_scale_min_k4` unpack (`Q6_K` has no bit-interleaved scale/min
/// pair at all, just 16 plain signed-`i8` scales and no `dmin`) or their
/// `byte_base = (sub_block/2)*32` nibble addressing (`Q6_K`'s sub-blocks
/// are 16 wide, not 32, and its `ql`/`qh` byte layout is genuinely
/// different -- see [`proxima_gguf::quant::q6_k::unpack_levels`], which
/// this function's addressing (`half`/`local_sub`/`lane`/`subhalf`) is
/// derived from and stays consistent with: `half = sub_block / 8`,
/// `local_sub = sub_block % 8`, `lane = local_sub / 2`,
/// `subhalf = local_sub % 2`; for output offset `e` within the sub-block,
/// `l = subhalf*16 + e` is `unpack_levels`'s own index parameter). Each
/// unpacked 6-bit level is biased by -32 before the multiply
/// (`q6_k.rs`'s own `x = d*sc*(q-32)` doc), matching what
/// `proxima_gguf::quant::q6_k::dequantize_block` computes exactly, just
/// against a `Q8_K` `i8` activation instead of an `f32` one.
#[cfg(feature = "q6k-int8-dot")]
pub(super) fn dot_q6k_q8k_block_scalar(weight_block: &[u8], q8k_block: &[u8]) -> f32 {
    let d_weight = f16_le_at(weight_block, Q6K_D_OFFSET);
    let ql = &weight_block[Q6K_QL_OFFSET..Q6K_QL_OFFSET + Q6K_QL_BYTES];
    let qh = &weight_block[Q6K_QH_OFFSET..Q6K_QH_OFFSET + Q6K_QH_BYTES];
    let scales =
        &weight_block[Q6K_SCALES_OFFSET..Q6K_SCALES_OFFSET + proxima_gguf::quant::q6_k::SUB_BLOCKS];

    let mut d_bytes = [0u8; 4];
    d_bytes.copy_from_slice(&q8k_block[Q8K_D_OFFSET..Q8K_D_OFFSET + 4]);
    let activation_scale = f32::from_le_bytes(d_bytes);
    let activation_qs = &q8k_block[Q8K_QS_OFFSET..Q8K_QS_OFFSET + Q4K_BLOCK_ELEMENTS];

    let sub_block_elements = proxima_gguf::quant::q6_k::SUB_BLOCK_ELEMENTS;
    let mut sumi = 0i32;
    for (sub_block, &scale_byte) in scales.iter().enumerate() {
        let half = sub_block / 8;
        let local_sub = sub_block % 8;
        let lane = local_sub / 2;
        let subhalf = local_sub % 2;
        let scale = i32::from(scale_byte.cast_signed());
        let ql_half = &ql[half * 64..half * 64 + 64];
        let qh_half = &qh[half * 32..half * 32 + 32];
        let activation_base = sub_block * sub_block_elements;

        let mut partial = 0i32;
        for offset in 0..sub_block_elements {
            let l = subhalf * sub_block_elements + offset;
            let ql_byte = if lane == 0 || lane == 2 {
                ql_half[l]
            } else {
                ql_half[l + 32]
            };
            let nibble = if lane < 2 {
                ql_byte & 0x0F
            } else {
                ql_byte >> 4
            };
            let high = (qh_half[l] >> (2 * lane)) & 0x03;
            let level = i32::from(nibble) | (i32::from(high) << 4);
            let quant = level - 32;
            let activation_value = i32::from(activation_qs[activation_base + offset].cast_signed());
            partial += quant * activation_value;
        }
        sumi += partial * scale;
    }

    let d = activation_scale * d_weight;
    d * sumi as f32
}

/// [`dot_q6k_q8k_block_scalar`]'s mechanism, `vdotq_s32`-accelerated.
/// Ports `ggml_vec_dot_q6_K_q8_K`'s plain `__ARM_NEON` arm
/// (`arch/arm/quants.c:3001-3090`, the non-`__ARM_FEATURE_MATMUL_INT8`,
/// non-SVE arm) with one deliberate simplification: ggml's version keeps
/// levels unbiased (`0..63`) through the dot and corrects for the -32
/// bias afterward via `bsums`/`isum_mins` (an optimization to avoid a
/// per-lane subtract); this port applies the -32 bias directly in-register
/// via [`vsubq_s8`] right after assembling each `q6bytes` lane, then dots
/// against `Q8_K` `i8` activations with no separate correction term
/// needed -- the SAME value, a simpler derivation, one extra vector op per
/// lane (8 total) traded for not needing `y[i].bsums` decoded at all here.
///
/// # Safety
/// Caller guarantees `FEAT_DotProd`; `weight_block.len() ==
/// Q6K_BLOCK_BYTES` and `q8k_block.len() == Q8K_BLOCK_BYTES` (both
/// [`dot_q6k_q8k`]'s own `chunks_exact` calls already guarantee before
/// calling this).
#[cfg(all(q4k_dotprod, feature = "q6k-int8-dot"))]
pub(super) unsafe fn dot_q6k_q8k_block_neon_dotprod(weight_block: &[u8], q8k_block: &[u8]) -> f32 {
    let d_weight = f16_le_at(weight_block, Q6K_D_OFFSET);
    let mut scales = [0i8; 16];
    for (slot, byte) in scales.iter_mut().zip(
        weight_block[Q6K_SCALES_OFFSET..Q6K_SCALES_OFFSET + proxima_gguf::quant::q6_k::SUB_BLOCKS]
            .iter(),
    ) {
        *slot = byte.cast_signed();
    }

    let mut d_bytes = [0u8; 4];
    d_bytes.copy_from_slice(&q8k_block[Q8K_D_OFFSET..Q8K_D_OFFSET + 4]);
    let activation_scale = f32::from_le_bytes(d_bytes);

    // SAFETY: caller-guaranteed FEAT_DotProd; `ql_base`/`qh_base` each walk
    // exactly `Q6K_QL_BYTES` / `Q6K_QH_BYTES` bytes and `q8_base` walks
    // exactly `Q4K_BLOCK_ELEMENTS` bytes across the two `half` iterations
    // below, all within the slices' checked bounds.
    unsafe {
        let m4b = vdupq_n_u8(0x0f);
        let high_bits_mask = vdupq_n_u8(0x03);
        let m32s = vdupq_n_s8(32);
        let mzero = vdupq_n_s32(0);
        let ql_base = weight_block[Q6K_QL_OFFSET..].as_ptr();
        let qh_base = weight_block[Q6K_QH_OFFSET..].as_ptr();
        let q8_base = q8k_block[Q8K_QS_OFFSET..].as_ptr().cast::<i8>();

        // FOUR accumulators, not one. `Q6_K` is 16 sub-blocks of 16 (vs
        // `Q4_K`/`Q5_K`'s 8 of 32), so a single `sumi` chains 16 dependent
        // `madd`s per super-block against `Q4_K`'s 3 -- measured 0.0429 ns/mac
        // here vs 0.0245 there, a 1.75x gap on only 1.33x the instructions
        // (190 vs 143), which is the signature of dependency depth, not
        // volume. Integer addition is associative, so splitting the chain is
        // bit-identical rather than merely close. The same defect cost 3.2x in
        // `dot_q4k_f32` in an earlier round, where the fix measured 5.68x --
        // width and depth turned out not to be independent factors.
        let mut sumi0: i32 = 0;
        let mut sumi1: i32 = 0;
        let mut sumi2: i32 = 0;
        let mut sumi3: i32 = 0;
        for half in 0..2usize {
            let qhbits0 = vld1q_u8(qh_base.add(half * 32));
            let qhbits1 = vld1q_u8(qh_base.add(half * 32 + 16));
            let ql0 = vld1q_u8(ql_base.add(half * 64));
            let ql1 = vld1q_u8(ql_base.add(half * 64 + 16));
            let ql2 = vld1q_u8(ql_base.add(half * 64 + 32));
            let ql3 = vld1q_u8(ql_base.add(half * 64 + 48));
            let q8_half_base = q8_base.add(half * 128);
            let scale_half = &scales[half * 8..half * 8 + 8];

            let low0 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql0, m4b),
                    vshlq_n_u8(vandq_u8(qhbits0, high_bits_mask), 4),
                )),
                m32s,
            );
            let low1 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql1, m4b),
                    vshlq_n_u8(vandq_u8(qhbits1, high_bits_mask), 4),
                )),
                m32s,
            );
            let low2 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql2, m4b),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qhbits0, 2), high_bits_mask), 4),
                )),
                m32s,
            );
            let low3 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql3, m4b),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qhbits1, 2), high_bits_mask), 4),
                )),
                m32s,
            );

            let q8_lo0 = vld1q_s8(q8_half_base);
            let q8_lo1 = vld1q_s8(q8_half_base.add(16));
            let q8_lo2 = vld1q_s8(q8_half_base.add(32));
            let q8_lo3 = vld1q_s8(q8_half_base.add(48));
            sumi0 += vaddvq_s32(sdot_s32(mzero, low0, q8_lo0)) * i32::from(scale_half[0]);
            sumi1 += vaddvq_s32(sdot_s32(mzero, low1, q8_lo1)) * i32::from(scale_half[1]);
            sumi2 += vaddvq_s32(sdot_s32(mzero, low2, q8_lo2)) * i32::from(scale_half[2]);
            sumi3 += vaddvq_s32(sdot_s32(mzero, low3, q8_lo3)) * i32::from(scale_half[3]);

            let high0 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql0, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qhbits0, 4), high_bits_mask), 4),
                )),
                m32s,
            );
            let high1 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql1, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qhbits1, 4), high_bits_mask), 4),
                )),
                m32s,
            );
            let high2 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql2, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qhbits0, 6), high_bits_mask), 4),
                )),
                m32s,
            );
            let high3 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql3, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qhbits1, 6), high_bits_mask), 4),
                )),
                m32s,
            );

            let q8_hi0 = vld1q_s8(q8_half_base.add(64));
            let q8_hi1 = vld1q_s8(q8_half_base.add(80));
            let q8_hi2 = vld1q_s8(q8_half_base.add(96));
            let q8_hi3 = vld1q_s8(q8_half_base.add(112));
            sumi0 += vaddvq_s32(sdot_s32(mzero, high0, q8_hi0)) * i32::from(scale_half[4]);
            sumi1 += vaddvq_s32(sdot_s32(mzero, high1, q8_hi1)) * i32::from(scale_half[5]);
            sumi2 += vaddvq_s32(sdot_s32(mzero, high2, q8_hi2)) * i32::from(scale_half[6]);
            sumi3 += vaddvq_s32(sdot_s32(mzero, high3, q8_hi3)) * i32::from(scale_half[7]);
        }

        activation_scale * d_weight * (sumi0 + sumi1 + sumi2 + sumi3) as f32
    }
}

/// A full `Q6_K`-quantized weight matrix (`rows` x `k`) times one `f32`
/// activation vector — [`matmul_q6k_f32`]'s packed-arithmetic sibling.
///
/// # Errors
/// Propagates [`quantize_row_q8k`]'s and [`dot_q6k_q8k`]'s
/// [`TensorError::QuantizedShapeMismatch`], or reports the same error if
/// `weights.len()` is not a whole multiple of `rows`.
#[cfg(feature = "q6k-int8-dot")]
pub fn matmul_q6k_q8k_f32(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
) -> Result<Vec<f32>, TensorError> {
    matmul_q6k_q8k_f32_impl(weights, rows, activation, 1, None)
}

/// [`matmul_q6k_q8k_f32`]'s body, plus `leading_total` (the sequence-position
/// count [`run_reduce_quantized`] derives as `activation.len() / k`) and the
/// [`CohortSession`] a caller already inside a forward pass's session can
/// supply — identical shape to [`matmul_q4k_q8k_f32_impl`]: `activation` is
/// one contiguous position-major `&[f32]` of `leading_total * k` elements,
/// quantized to `Q8_K` once, so each weight row's bytes are read once and its
/// dot reused across every position instead of the weight stream being
/// re-read once per position.
#[cfg(feature = "q6k-int8-dot")]
pub(super) fn matmul_q6k_q8k_f32_impl(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
    leading_total: usize,
    session: Option<&MatmulSession<'_>>,
) -> Result<Vec<f32>, TensorError> {
    if rows == 0 {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "matmul_q6k_q8k_f32 called with zero rows",
        });
    }
    if !weights.len().is_multiple_of(rows) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight byte length is not a whole multiple of the row count",
        });
    }
    if leading_total == 0 || !activation.len().is_multiple_of(leading_total) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length is not a whole multiple of the position count",
        });
    }
    let k = activation.len() / leading_total;
    if !k.is_multiple_of(Q4K_BLOCK_ELEMENTS) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length is not a whole multiple of the q8_k super-block size",
        });
    }
    let block_count = activation.len() / Q4K_BLOCK_ELEMENTS;
    let q8k_row_bytes = (k / Q4K_BLOCK_ELEMENTS) * Q8K_BLOCK_BYTES;
    let mut activation_q8k = vec![0u8; block_count * Q8K_BLOCK_BYTES];
    quantize_row_q8k_dispatch(activation, &mut activation_q8k, session)?;

    let row_bytes = weights.len() / rows;
    match quantized_matmul_workers(rows, activation.len()) {
        Some(workers) => {
            matmul_rows_threaded(rows, leading_total, workers, session, k, |row, slot| {
                let start = row * row_bytes;
                let weight_row = &weights[start..start + row_bytes];
                for (position, output_slot) in slot.iter_mut().enumerate() {
                    let q8k_start = position * q8k_row_bytes;
                    *output_slot = dot_q6k_q8k(
                        weight_row,
                        &activation_q8k[q8k_start..q8k_start + q8k_row_bytes],
                    )?;
                }
                Ok(())
            })
        }
        None => weights.chunks_exact(row_bytes).try_fold(
            Vec::with_capacity(rows * leading_total),
            |mut output, weight_row| {
                for position in 0..leading_total {
                    let q8k_start = position * q8k_row_bytes;
                    output.push(dot_q6k_q8k(
                        weight_row,
                        &activation_q8k[q8k_start..q8k_start + q8k_row_bytes],
                    )?);
                }
                Ok::<Vec<f32>, TensorError>(output)
            },
        ),
    }
}

/// [`matmul_q6k_q8k_f32`] with every row routed through
/// [`dot_q6k_q8k_portable`] instead of [`dot_q6k_q8k`].
///
/// # Errors
/// Same as [`matmul_q6k_q8k_f32`].
#[cfg(feature = "q6k-int8-dot")]
pub fn matmul_q6k_q8k_portable_f32(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
) -> Result<Vec<f32>, TensorError> {
    if rows == 0 {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "matmul_q6k_q8k_portable_f32 called with zero rows",
        });
    }
    if !weights.len().is_multiple_of(rows) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight byte length is not a whole multiple of the row count",
        });
    }
    if !activation.len().is_multiple_of(Q4K_BLOCK_ELEMENTS) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length is not a whole multiple of the q8_k super-block size",
        });
    }
    let block_count = activation.len() / Q4K_BLOCK_ELEMENTS;
    let mut activation_q8k = vec![0u8; block_count * Q8K_BLOCK_BYTES];
    quantize_row_q8k(activation, &mut activation_q8k)?;

    let row_bytes = weights.len() / rows;
    weights
        .chunks_exact(row_bytes)
        .map(|weight_row| dot_q6k_q8k_portable(weight_row, &activation_q8k))
        .collect()
}

/// Ported from ggml tinyBLAS's `gemm_bloc`: `ROWS` x [`TILE_COLS`]
/// output accumulators declared as `float32x4_t`, a native NEON vector
/// register type, not an `[f32; 4]` array indexed by a loop variable
/// (`proxima-tensor/docs/discipline.md` — attempt 2 spilled 737 `str q`
/// instructions doing exactly that). `av` holds one `float32x4_t` per tile
/// row, loaded once per `k`-step and reused across all [`TILE_COLS`]
/// columns; `bv` is loaded once per column per step and fused against every
/// row's `av`, giving 0.42 loads per multiply-accumulate the way tinyBLAS's
/// own microkernel does, versus this crate's un-tiled 2.0.
///
/// Generic over the row count (`ROWS`) rather than fixed at [`TILE_ROWS`] so
/// the row-remainder pass can call the identical kernel body monomorphised
/// at whichever width `1..=5` the leftover row count needs, instead of a
/// hand-duplicated copy per width.
#[cfg(target_arch = "aarch64")]
pub(super) unsafe fn gemm_tile_neon<const ROWS: usize>(
    a: KStridedTile,
    b: KStridedTile,
    k: usize,
    out: &mut [[f32; TILE_COLS]; ROWS],
) {
    // `vdupq_n_f32` requires the `neon` target feature, unconditionally
    // present in the aarch64 base ISA this module is gated on.
    let mut acc = [[unsafe { vdupq_n_f32(0.0) }; TILE_COLS]; ROWS];
    let steps = k / 4;
    for step in 0..steps {
        let l = step * 4;
        let mut av = [unsafe { vdupq_n_f32(0.0) }; ROWS];
        for (row, lane) in av.iter_mut().enumerate() {
            // caller guarantees `a.base + row * a.k_stride + l + 4 <= a.data.len()`
            // via the reduction-dim contiguity and row-count checks in
            // `neon_tile_plan` and its `run_reduce` call site.
            let offset = (a.base + row as i64 * a.k_stride + l as i64) as usize;
            *lane = unsafe { vld1q_f32(a.data.as_ptr().add(offset)) };
        }
        let mut bv = [unsafe { vdupq_n_f32(0.0) }; TILE_COLS];
        for (column, lane) in bv.iter_mut().enumerate() {
            // caller guarantees `b.base + column * b.k_stride + l + 4 <= b.data.len()`
            // by the same contiguity and column-count checks.
            let offset = (b.base + column as i64 * b.k_stride + l as i64) as usize;
            *lane = unsafe { vld1q_f32(b.data.as_ptr().add(offset)) };
        }
        for (row, acc_row) in acc.iter_mut().enumerate() {
            let row_vector = av[row];
            for (acc_lane, &bv_lane) in acc_row.iter_mut().zip(bv.iter()) {
                // both operands are `float32x4_t`; NEON `fmla.4s` has no
                // aliasing hazard between distinct accumulator lanes.
                *acc_lane = unsafe { vfmaq_f32(*acc_lane, row_vector, bv_lane) };
            }
        }
    }
    for (row, (acc_row, out_row)) in acc.iter().zip(out.iter_mut()).enumerate() {
        for (column, (&acc_lane, out_value)) in acc_row.iter().zip(out_row.iter_mut()).enumerate() {
            // horizontal combine of one lane-group; sound for any float
            // values, no aliasing or bounds precondition beyond `acc` being
            // fully initialized above.
            let mut total = *out_value + unsafe { vaddvq_f32(acc_lane) };
            for l in steps * 4..k {
                let offset_a = (a.base + row as i64 * a.k_stride + l as i64) as usize;
                let offset_b = (b.base + column as i64 * b.k_stride + l as i64) as usize;
                total = a.data[offset_a].mul_add(b.data[offset_b], total);
            }
            *out_value = total;
        }
    }
}

