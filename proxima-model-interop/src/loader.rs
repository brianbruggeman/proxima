//! Explicit page-warming for a byte view a caller already owns.
//!
//! Nothing here opens or maps a file: [`prefault`] takes the `&[u8]` a
//! caller's own `mmap` (or any other backing buffer) already produced and
//! touches it, in parallel, through the same
//! [`prime::os::background::ProximaBackgroundPool`]
//! [`proxima_tensor::cpu`]'s `matmul_rows_threaded` dispatches row work
//! through (`proxima-tensor/src/cpu.rs:4371`) — this module mirrors that
//! function's shared-cursor claim loop, applied to page ranges instead of
//! matmul rows. A first read of a lazily-mapped file demand-pages one minor
//! fault per page touched; paying that fault storm inside a timed forward
//! pass serializes it against the compute the forward is trying to measure.
//! Touching every page explicitly, in parallel, before the timed section
//! starts amortizes the fault cost across every worker the forward pass will
//! use anyway, instead of paying it fault-by-fault mid-compute on whichever
//! thread happens to read that byte range first.
//!
//! Explicit, not automatic: nothing in [`crate::bind`] or [`crate::transform`]
//! calls this. A caller serving many small models, or one that will only
//! read a slice of a large mapping, should skip it — warming pages the
//! caller never reads back is wasted work, and only the caller knows which
//! case it is in.

use core::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{SyncSender, sync_channel};
use std::sync::{Arc, OnceLock};

use prime::os::background::ProximaBackgroundPool;

use crate::error::InteropError;

/// Stride between page touches, in bytes. Smaller than every real page size
/// this runs on (4 KiB on x86_64/most arm64, 16 KiB on Apple Silicon), so
/// every page is touched regardless of platform — matching the touch-loop
/// shape the serial prefault experiment measured (`prefault_experiment.patch`,
/// preserved in the perf log for this change) rather than querying the
/// platform's real page size.
pub const PREFAULT_STRIDE_BYTES: usize = 4096;

/// Oversubscription factor for chunk count vs. worker count. Page-touch cost
/// is close to uniform per byte (unlike `matmul_rows_threaded`'s rows, whose
/// cost varies with a row's quant type), so `workers` chunks alone already
/// balance well and measured better: 3-run comparison over a 4.14 GB
/// real-checkpoint mmap (`bind::real_openchat_file`'s host-local fixture)
/// gave oversubscribe=1 a lower mean AND a tighter spread than
/// oversubscribe=2 (prefault wall 121.84 ms, CoV 0.10% vs. 122.97 ms,
/// CoV 1.62%; forward wall 360.39 ms, CoV 0.38% vs. 364.50 ms, CoV 0.45%) —
/// the extra chunking bought nothing and added scheduling variance instead.
pub const PREFAULT_OVERSUBSCRIBE: usize = 1;

/// Touches one byte per [`PREFAULT_STRIDE_BYTES`] of `bytes`, dispatched
/// across `prefault_pool`'s workers, and blocks until every touch has
/// completed — so nothing observes a partially-warmed mapping once this
/// returns `Ok`. A no-op for an empty slice.
///
/// # Errors
///
/// [`InteropError::PrefaultPoolUnavailable`] if the shared background pool
/// fails to build (see `prefault_pool`), or if a worker never reports a
/// completed chunk (`ProximaBackgroundPool` catches and discards worker
/// panics rather than propagating them).
pub fn prefault(bytes: &[u8]) -> Result<(), InteropError> {
    if bytes.is_empty() {
        return Ok(());
    }
    let pool = prefault_pool()?;
    let workers = pool.workers().max(1);
    let page_count = bytes.len().div_ceil(PREFAULT_STRIDE_BYTES);
    let chunk_count = workers
        .saturating_mul(PREFAULT_OVERSUBSCRIBE)
        .clamp(1, page_count.max(1));
    let chunk_pages = page_count.div_ceil(chunk_count);
    let chunk_stride = chunk_pages
        .saturating_mul(PREFAULT_STRIDE_BYTES)
        .max(PREFAULT_STRIDE_BYTES);

    let mut chunk_ranges = Vec::with_capacity(chunk_count);
    let mut offset = 0usize;
    while offset < bytes.len() {
        let end = (offset + chunk_stride).min(bytes.len());
        chunk_ranges.push((offset, end));
        offset = end;
    }
    let chunk_ranges_len = chunk_ranges.len();
    let chunk_ranges = Arc::new(chunk_ranges);

    // SAFETY-relevant: cast to `usize` here, reconstructed unsafely inside
    // each pool closure, the same way `matmul_rows_threaded` crosses the
    // pool's `'static` spawn bound with `dot_row_address`
    // (proxima-tensor/src/cpu.rs:4402). Sound because this function blocks
    // in `Receiver::recv` for every spawned chunk before returning, so
    // `bytes` (borrowed from the caller for the whole call) outlives every
    // reconstructed reference, and every chunk range in `chunk_ranges` is
    // disjoint and read-only, so no two closures ever race a write.
    let base_address = bytes.as_ptr() as usize;
    let next_index = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = sync_channel(chunk_ranges_len);
    let spawned_count = workers.saturating_sub(1).min(chunk_ranges_len.saturating_sub(1));

    for _ in 0..spawned_count {
        let sender = sender.clone();
        let next_index = Arc::clone(&next_index);
        let chunk_ranges = Arc::clone(&chunk_ranges);
        drop(pool.spawn(move || {
            claim_and_touch(&next_index, base_address, &chunk_ranges, &sender);
            Ok::<(), proxima_core::ProximaError>(())
        }));
    }

    // the calling thread pulls from the same shared cursor as every pool
    // task instead of running one reserved chunk, so it never idles once
    // its own chunk finishes early — same rationale as
    // `matmul_rows_threaded`'s own-chunk claim (proxima-tensor/src/cpu.rs:4445).
    claim_and_touch(&next_index, base_address, &chunk_ranges, &sender);
    drop(sender);

    let mut reported = 0usize;
    while reported < chunk_ranges_len {
        match receiver.recv() {
            Ok(()) => reported += 1,
            // every sender clone is gone (each spawned closure's clone is
            // dropped whether it sends or panics), so no further chunk will
            // ever report — stop waiting instead of blocking forever.
            Err(_) => break,
        }
    }
    if reported < chunk_ranges_len {
        return Err(InteropError::PrefaultPoolUnavailable(alloc::format!(
            "prefault: {} of {chunk_ranges_len} page-touch chunks never reported; \
             ProximaBackgroundPool catches and discards worker panics",
            chunk_ranges_len - reported,
        )));
    }
    Ok(())
}

/// Pulls chunk indices off `next_index` one at a time and touches each
/// range's pages — the page-touch counterpart of `matmul_rows_threaded`'s
/// `claim_and_run_rows` (proxima-tensor/src/cpu.rs:4505), called by both the
/// calling thread and every spawned pool task in [`prefault`] so a puller
/// that finishes early goes straight back for the next available chunk
/// instead of idling.
///
/// # Safety (of the `unsafe` block inside)
/// `base_address` must stay valid, and every `(start, end)` pair in
/// `chunk_ranges` must lie within the mapping it points into, for as long as
/// any puller can still observe `next_index` below `chunk_ranges.len()` —
/// guaranteed by [`prefault`] draining `chunk_ranges.len()` results from
/// `sender`'s channel before its own `bytes` borrow can end.
fn claim_and_touch(
    next_index: &AtomicUsize,
    base_address: usize,
    chunk_ranges: &[(usize, usize)],
    sender: &SyncSender<()>,
) {
    loop {
        let index = next_index.fetch_add(1, Ordering::Relaxed);
        if index >= chunk_ranges.len() {
            return;
        }
        let (start, end) = chunk_ranges[index];
        // SAFETY: see this function's doc comment.
        let chunk = unsafe { core::slice::from_raw_parts((base_address + start) as *const u8, end - start) };
        let mut checksum: u8 = 0;
        for touch_offset in (0..chunk.len()).step_by(PREFAULT_STRIDE_BYTES) {
            checksum = core::hint::black_box(checksum.wrapping_add(chunk[touch_offset]));
        }
        core::hint::black_box(checksum);
        let _ = sender.send(());
    }
}

/// The pool backing [`prefault`]'s chunk dispatch. Built once, on first use,
/// and reused for every call in the process — a fresh `ProximaBackgroundPool`
/// per call would reintroduce the per-call OS-thread-spawn cost this pool
/// exists to remove. `OnceLock` only memoizes success: a failed build is not
/// cached, so a later call (after whatever exhausted OS thread resources
/// clears up) can retry instead of latching a permanent failure. Deliberately
/// this module's own pool, not [`proxima_tensor::cpu`]'s private `nest_pool`
/// (proxima-tensor/src/cpu.rs:1209): that pool is internal to the tensor
/// crate's own module, not part of its public surface, so this crate builds
/// its own instance of the same primitive rather than reaching across a
/// crate boundary into another crate's private state.
fn prefault_pool() -> Result<Arc<ProximaBackgroundPool>, InteropError> {
    if let Some(pool) = PREFAULT_POOL.get() {
        return Ok(Arc::clone(pool));
    }
    let built = Arc::new(ProximaBackgroundPool::new().map_err(|error| {
        InteropError::PrefaultPoolUnavailable(alloc::format!("build prefault thread pool: {error}"))
    })?);
    // `set` can lose a race to a concurrent first caller; either pool is
    // equally valid, so use whichever one actually landed.
    let _ = PREFAULT_POOL.set(Arc::clone(&built));
    Ok(PREFAULT_POOL.get().cloned().unwrap_or(built))
}

static PREFAULT_POOL: OnceLock<Arc<ProximaBackgroundPool>> = OnceLock::new();

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn empty_slice_is_a_no_op() {
        let outcome = prefault(&[]);
        assert!(outcome.is_ok(), "empty slice must not touch the pool at all");
    }

    /// Every byte at a `PREFAULT_STRIDE_BYTES` boundary is read, including
    /// the last chunk's tail — a real GGUF file's length is never an exact
    /// multiple of the stride, so the last chunk's `chunk_stride` clamp must
    /// still cover the remainder instead of dropping it.
    #[test]
    fn covers_a_buffer_that_spans_many_chunks_and_a_partial_tail() {
        let bytes = alloc::vec![7u8; PREFAULT_STRIDE_BYTES * 37 + 129];
        let outcome = prefault(&bytes);
        assert!(outcome.is_ok(), "prefault over a multi-chunk buffer must succeed: {outcome:?}");
    }

    #[test]
    fn single_byte_buffer_succeeds() {
        let bytes = [42u8];
        let outcome = prefault(&bytes);
        assert!(outcome.is_ok(), "single-byte buffer must succeed: {outcome:?}");
    }
}
