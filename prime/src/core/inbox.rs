//! lock-free bounded MPSC inbox built on per-producer SPSC lanes.
//!
//! design: instead of one shared ring with `tail`-CAS contention (Vyukov MPSC),
//! we pre-allocate `num_lanes` independent SPSC rings. each `Producer` owns
//! one lane for its lifetime; `Producer::try_send` is a single uncontended
//! `Release` store. the (single) `Consumer` round-robins across lanes,
//! draining whichever has data. result: zero atomic contention between
//! producers, regardless of count.
//!
//! `Producer: Clone` allocates the next available lane via a monotonic
//! counter; cloning more than `num_lanes` times panics. for proxima the
//! caller knows `num_lanes` at startup (= num_cores).
//!
//! compile-time flag selects layer:
//! - `runtime-prime-inbox-alloc`: heap-backed Box<[Lane<T>]>, runtime
//!   `num_lanes` and `lane_capacity`
//! - `runtime-prime-inbox-const`: stack-backed (TODO)
//!
//! no_std + alloc only — uses `core::*` and `alloc::*` exclusively.
//! AtomicWaker reuses `futures::task::AtomicWaker` (no_std-compat).

#[cfg(all(
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-inbox-const",
))]
compile_error!(
    "runtime-prime-inbox-alloc and runtime-prime-inbox-const are mutually \
     exclusive — pick one"
);

extern crate alloc;

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
#[cfg(feature = "std")]
use core::cell::Cell;
use core::cell::UnsafeCell;
use core::fmt;
use core::future::Future;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::pin::Pin;
use core::sync::atomic::{self, AtomicBool, AtomicUsize, Ordering};
use core::task::{Context, Poll};

use atomic_waker::AtomicWaker;
#[cfg(feature = "std")]
use crossbeam_queue::SegQueue;
use crossbeam_utils::CachePadded;

#[derive(Debug)]
pub enum SendError<T> {
    Full(T),
    Disconnected(T),
    /// No SPSC lane could be assigned to the calling thread because
    /// the inbox's `num_lanes` capacity is exhausted. Returned only
    /// from `try_send_mpsc`. Permanent for new caller threads against
    /// this inbox — caller must size `num_lanes` large enough at
    /// `channel()` construction to cover every distinct (cloned
    /// `Producer` + mpsc-using thread) for the inbox's lifetime.
    NoLanes(T),
    /// Inbox has been quiesced via `Producer::close()` (typically
    /// `CoreShardHandle::quiesce()`). New pushes are rejected; the
    /// consumer continues draining in-flight items. Distinct from
    /// `Disconnected` (all producers dropped — terminal) and `Full`
    /// (transient back-pressure).
    Closed(T),
}

impl<T> SendError<T> {
    pub fn into_inner(self) -> T {
        match self {
            Self::Full(value)
            | Self::Disconnected(value)
            | Self::NoLanes(value)
            | Self::Closed(value) => value,
        }
    }
}

impl<T> fmt::Display for SendError<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full(_) => formatter.write_str("inbox lane full"),
            Self::Disconnected(_) => formatter.write_str("inbox consumer dropped"),
            Self::NoLanes(_) => formatter.write_str("inbox num_lanes exhausted"),
            Self::Closed(_) => formatter.write_str("inbox quiesced (closed)"),
        }
    }
}

/// quiesce-marker bit in `Lane::tail`. set via `fetch_or` from any
/// thread (typically by `Inner::close()`); preserved by the producer's
/// CAS-publish. position counter occupies the bottom 63 bits of `tail`
/// on 64-bit; the marker reserves the top bit. position wraps within
/// `POSITION_MASK` (avoid bleeding into the closed bit), enforced by
/// CAS publishing a position-masked next value.
const CLOSED_BIT: usize = 1 << (usize::BITS as usize - 1);
const POSITION_MASK: usize = !CLOSED_BIT;

/// internal lane-level error so the slow path in `try_send` can
/// distinguish "lane full" from "closed" without leaking the
/// inner-level `SendError` into `Lane`.
enum LaneSendErr<T> {
    Full(T),
    Closed(T),
}

#[derive(Debug, PartialEq, Eq)]
pub enum TryRecvError {
    Empty,
    Disconnected,
}

impl fmt::Display for TryRecvError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("inbox empty"),
            Self::Disconnected => formatter.write_str("inbox all producers dropped"),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct RecvError;

impl fmt::Display for RecvError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("inbox closed")
    }
}

/// one SPSC lane. owned-by-producer for writes, consumer round-robins reads.
/// head/tail are each wrapped in `CachePadded` so they live on separate cache
/// lines regardless of platform — kills producer↔consumer false sharing.
/// hand-rolled 64-byte padding was wrong on aarch64-apple (128-byte lines).
///
/// `cached_head` is producer-private (only the single producer reads/writes)
/// and `cached_tail` is consumer-private. they let the hot path skip the
/// `Acquire` load on `head` / `tail` when the cached value still proves there
/// is room / data — stale caches only cause false-full / false-empty which
/// then refreshes, never data races.
struct Lane<T> {
    head: CachePadded<AtomicUsize>,
    tail: CachePadded<AtomicUsize>,
    cached_head: CachePadded<UnsafeCell<usize>>,
    cached_tail: CachePadded<UnsafeCell<usize>>,
    capacity: usize,
    mask: usize,
    slots: Box<[UnsafeCell<MaybeUninit<T>>]>,
}

// SAFETY: producer (single) writes to slots[tail & mask] then publishes via
// tail.store(Release). consumer (single) reads slots[head & mask] after
// observing tail > head via Acquire load. no two threads ever touch the
// same slot concurrently. cached_head is single-producer only;
// cached_tail is single-consumer only — by the SPSC contract.
unsafe impl<T: Send> Sync for Lane<T> {}
unsafe impl<T: Send> Send for Lane<T> {}

impl<T> Lane<T> {
    fn new(capacity: usize) -> Self {
        assert!(
            capacity > 0 && capacity.is_power_of_two(),
            "lane capacity must be a non-zero power of two; got {capacity}",
        );
        let mut slots: Vec<UnsafeCell<MaybeUninit<T>>> = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(UnsafeCell::new(MaybeUninit::uninit()));
        }
        Self {
            head: CachePadded::new(AtomicUsize::new(0)),
            tail: CachePadded::new(AtomicUsize::new(0)),
            cached_head: CachePadded::new(UnsafeCell::new(0)),
            cached_tail: CachePadded::new(UnsafeCell::new(0)),
            capacity,
            mask: capacity - 1,
            slots: slots.into_boxed_slice(),
        }
    }

    /// SPSC producer side. caller guarantees single-writer to this lane.
    ///
    /// **Closed encoding**: the top bit of `tail` (`CLOSED_BIT`) is a
    /// quiesce marker. The fast-path capacity check
    /// `raw_tail.wrapping_sub(cached_head) >= capacity` naturally
    /// rejects pushes when the bit is set (the diff becomes a huge
    /// number), routing producers through the slow path which then
    /// distinguishes Closed from Full. The producer publishes via
    /// **CAS** instead of a blind store so a concurrently-set
    /// `CLOSED_BIT` (from `Inner::close()` on another thread) is
    /// preserved. The position-counter increment is masked into the
    /// bottom 63 bits so a wrap cannot bleed into the closed bit.
    #[inline(always)]
    fn try_send(&self, value: T) -> Result<(), LaneSendErr<T>> {
        let raw_tail = self.tail.load(Ordering::Relaxed);
        // SAFETY: single producer reads/writes cached_head.
        let cached_head = unsafe { *self.cached_head.get() };
        if raw_tail.wrapping_sub(cached_head) >= self.capacity {
            // false-full: refresh from atomic and re-check. only path
            // that pays the Acquire load — keeps the common case
            // branch-light. also: this is where we detect Closed.
            let head = self.head.load(Ordering::Acquire);
            // SAFETY: single producer
            unsafe { *self.cached_head.get() = head };
            if raw_tail.wrapping_sub(head) >= self.capacity {
                if raw_tail & CLOSED_BIT != 0 {
                    return Err(LaneSendErr::Closed(value));
                }
                return Err(LaneSendErr::Full(value));
            }
        }
        // raw_tail's CLOSED_BIT is provably clear here: if it were set,
        // wrapping_sub would have routed us through the slow path
        // above, where we either returned Closed or refreshed
        // cached_head to a value that left us still "full" → Closed.
        // The only way to fall through is CLOSED_BIT == 0.
        let position = raw_tail & POSITION_MASK;
        // SAFETY: single producer per lane; we own slot[position & mask]
        // until publish. consumer cannot reach it because the CAS below
        // hasn't run.
        unsafe {
            (*self.slots[position & self.mask].get()).write(value);
        }
        // Publish via CAS. Preserves any CLOSED_BIT set concurrently by
        // `Inner::close()`; position increments in the bottom 63 bits
        // only (mask after add) so the position wrap never bleeds into
        // the closed bit. SPSC: only the quiesce thread can cause CAS
        // to retry, and only by setting the closed bit (a one-shot).
        let mut current = raw_tail;
        loop {
            let next_position = (current & POSITION_MASK).wrapping_add(1) & POSITION_MASK;
            let next = next_position | (current & CLOSED_BIT);
            match self.tail.compare_exchange_weak(
                current,
                next,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(()),
                Err(actual) => current = actual,
            }
        }
    }

    /// SPSC consumer side. returns None on empty.
    ///
    /// Reads of `tail` mask out `CLOSED_BIT` before comparing with
    /// `head` — the closed marker is metadata for the producer's
    /// rejection path; the consumer continues draining published
    /// payloads regardless of the bit.
    #[inline(always)]
    fn try_recv(&self) -> Option<T> {
        let head = self.head.load(Ordering::Relaxed);
        // SAFETY: single consumer reads/writes cached_tail.
        let cached_tail = unsafe { *self.cached_tail.get() };
        if head == cached_tail {
            // false-empty: refresh from atomic. mask out CLOSED_BIT
            // so we don't confuse the marker for a published position.
            let raw_tail = self.tail.load(Ordering::Acquire);
            let tail = raw_tail & POSITION_MASK;
            // SAFETY: single consumer
            unsafe { *self.cached_tail.get() = tail };
            if head == tail {
                return None;
            }
        }
        // SAFETY: cached_tail > head (after refresh if needed) means producer
        // published slot[head & mask]. single consumer per inbox, so no other
        // reader.
        let value = unsafe { (*self.slots[head & self.mask].get()).assume_init_read() };
        self.head.store(head.wrapping_add(1), Ordering::Release);
        Some(value)
    }

    /// quiesce this lane: set `CLOSED_BIT` on `tail` via fetch_or so
    /// concurrent producer CAS-publishes preserve it. idempotent.
    fn close(&self) {
        self.tail.fetch_or(CLOSED_BIT, Ordering::Release);
    }

    fn drop_remaining(&mut self) {
        let head = *self.head.get_mut();
        // mask out CLOSED_BIT: it's not part of the position counter.
        let tail = *self.tail.get_mut() & POSITION_MASK;
        let mut position = head;
        while position != tail {
            let slot = &mut self.slots[position & self.mask];
            // SAFETY: positions in [head, tail) hold initialized payloads.
            unsafe {
                slot.get_mut().assume_init_drop();
            }
            position = position.wrapping_add(1);
        }
    }
}

impl<T> Drop for Lane<T> {
    fn drop(&mut self) {
        self.drop_remaining();
    }
}

struct Inner<T> {
    lanes: Box<[Lane<T>]>,
    used_lanes: AtomicUsize,
    /// recycled lanes returned by `LaneCacheEntry::drop` on producer-
    /// thread exit. `try_allocate_lane_for_thread` consults this before
    /// bumping `used_lanes`, so a long-running process with thread
    /// churn (test harnesses, request-per-thread servers) doesn't
    /// monotonically exhaust the lane count.
    /// Only populated under `std` — the lane-cache entry that drives
    /// recycling is itself std-gated. C2 (lane-ticket) provides the
    /// no_std-clean alternative.
    #[cfg(feature = "std")]
    free_lanes: SegQueue<usize>,
    producer_count: AtomicUsize,
    consumer_alive: AtomicBool,
    waker: AtomicWaker,
    /// set by the consumer after registering its waker; cleared by the first
    /// producer to observe the consumer is waiting. cuts `AtomicWaker::wake`
    /// out of the hot path entirely when the consumer is actively draining
    /// (the common case in real workloads — only matters for futures that
    /// actually park).
    consumer_parked: AtomicBool,
    drain_cursor: AtomicUsize,
}

impl<T> Inner<T> {
    fn new(num_lanes: usize, lane_capacity: usize) -> Arc<Self> {
        assert!(num_lanes > 0, "inbox num_lanes must be > 0");
        let mut lanes: Vec<Lane<T>> = Vec::with_capacity(num_lanes);
        for _ in 0..num_lanes {
            lanes.push(Lane::new(lane_capacity));
        }
        Arc::new(Self {
            lanes: lanes.into_boxed_slice(),
            used_lanes: AtomicUsize::new(1),
            #[cfg(feature = "std")]
            free_lanes: SegQueue::new(),
            producer_count: AtomicUsize::new(1),
            consumer_alive: AtomicBool::new(true),
            waker: AtomicWaker::new(),
            consumer_parked: AtomicBool::new(false),
            drain_cursor: AtomicUsize::new(0),
        })
    }

    #[inline(always)]
    fn try_send_on(&self, lane_index: usize, value: T) -> Result<(), SendError<T>> {
        if !self.consumer_alive.load(Ordering::Acquire) {
            return Err(SendError::Disconnected(value));
        }
        match self.lanes[lane_index].try_send(value) {
            Ok(()) => {
                // Dekker-pattern fence — matches the fence in
                // `Recv::poll` after it stores `consumer_parked=true`.
                // Producer's tail.store(Release) and consumer's
                // consumer_parked.store(true, Release) are on different
                // atomics; without a SeqCst total-order participation
                // on both sides, producer's `consumer_parked.load` and
                // consumer's `try_recv` can both observe stale values
                // — consumer parks, producer skips wake, deadlock.
                // The Relaxed load below previously commented that the
                // consumer's post-register re-check covers a missed
                // wake. That re-check relies on this fence pair: with
                // SeqCst fences on both sides, whichever sequenced
                // first, the other observes the preceding store.
                //
                // Currently latent: worker_main calls `try_recv`
                // directly, not via `Recv::poll`. Fence preserved here
                // so the future-facing `recv()` path is sound when
                // exposed.
                atomic::fence(Ordering::SeqCst);
                if self.consumer_parked.load(Ordering::Acquire)
                    && self
                        .consumer_parked
                        .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                {
                    self.waker.wake();
                }
                Ok(())
            }
            Err(LaneSendErr::Full(value)) => Err(SendError::Full(value)),
            Err(LaneSendErr::Closed(value)) => Err(SendError::Closed(value)),
        }
    }

    /// quiesce the inbox: every lane's `CLOSED_BIT` is set so new
    /// pushes via any `Producer` return `SendError::Closed(value)`.
    /// idempotent — no panic if called twice. consumer side is
    /// unaffected; drain continues until empty (then `try_recv`
    /// returns `Empty`, eventually `Disconnected` once all producers
    /// drop). Pair with caller-side wakeup-fire so a parked consumer
    /// observes the close promptly.
    fn close(&self) {
        for lane in &self.lanes {
            lane.close();
        }
    }

    fn try_recv(&self) -> Result<T, TryRecvError> {
        let num_lanes = self.lanes.len();
        let start = self.drain_cursor.load(Ordering::Relaxed);
        // sticky-lane fast path: try the most recently active lane first.
        // for the single-producer pattern that dominates (spawn_burst,
        // bench harnesses) this is one atomic load instead of `num_lanes`.
        // for multi-producer fan-in the lane will be empty about as often
        // as not, so we fall through to the scan — same cost as before.
        if let Some(value) = self.lanes[start].try_recv() {
            return Ok(value);
        }
        // scan the rest, starting just past the sticky cursor for fairness.
        // when a different lane wins, we update the cursor to bias future
        // drains toward it — sustained multi-producer traffic naturally
        // round-robins because each push lands on a different lane and
        // each successful pop re-aims the cursor.
        for offset in 1..num_lanes {
            let lane_index = (start + offset) % num_lanes;
            if let Some(value) = self.lanes[lane_index].try_recv() {
                self.drain_cursor.store(lane_index, Ordering::Relaxed);
                return Ok(value);
            }
        }
        if self.producer_count.load(Ordering::Acquire) == 0 {
            Err(TryRecvError::Disconnected)
        } else {
            Err(TryRecvError::Empty)
        }
    }
}

/// per-thread cache mapping the Inner pointer (as raw `usize`) to a
/// lane guard. on thread exit the cache drops, each guard's drop
/// returns its lane to the inbox's free-lane pool so the next thread
/// to send into that inbox can reuse it. without recycling, processes
/// with thread churn (test harnesses, request-per-thread servers)
/// monotonically exhaust `num_lanes`.
///
/// Only compiled under `std` — the TLS backing requires it.
#[cfg(feature = "std")]
struct LaneCacheEntry {
    inner_ptr: usize,
    lane: usize,
    /// invoked once on drop: returns `lane` to `inner_ptr`'s free-lane
    /// queue. boxed because each entry's closure captures a different
    /// `Arc<Inner<T>>` (the inbox payload type varies). `Option` lets
    /// `Drop` `take()` and consume the `FnOnce` exactly once.
    release: Option<Box<dyn FnOnce() + Send + 'static>>,
}

#[cfg(feature = "std")]
impl Drop for LaneCacheEntry {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            release();
        }
    }
}

/// thread-local 4-way associative cache of (inner_ptr, lane). single-
/// slot was the original design but missed every other call on
/// alternating-inbox patterns (e.g., `spawn_burst` dispatching round-
/// robin across cores). 4 slots cover the common `num_cores ≤ 4`
/// dispatch patterns at zero overhead: the scan is 4 branchless
/// ptr compares against a fixed-size array. an entry of (0, 0) is
/// empty; replacement is round-robin via `HOT_LANES_NEXT`.
#[cfg(feature = "std")]
const HOT_LANES_SIZE: usize = 4;

// std-only TLS; deferred-debt for no_std cliff. C1 (thread-identity-trait)
// in woolly-watching-cupcake routes this through ThreadIdentity with a
// std-backed default and a no_std single-thread stub.
// DC5 transitional gate: the TLS block + try_send_mpsc are unavailable
// under alloc-only. C2 (lane-ticket) replaces this with a no_std-clean API.
#[cfg(feature = "std")]
std::thread_local! {
    static LANE_CACHE: core::cell::RefCell<Vec<LaneCacheEntry>> = const {
        core::cell::RefCell::new(Vec::new())
    };
    static HOT_LANES: core::cell::Cell<[(usize, usize); HOT_LANES_SIZE]> = const {
        core::cell::Cell::new([(0, 0); HOT_LANES_SIZE])
    };
    /// round-robin replacement cursor for HOT_LANES on cold-path
    /// insert. avoids LRU-tracking overhead — small N, the eviction
    /// policy barely matters.
    static HOT_LANES_NEXT: core::cell::Cell<usize> = const {
        core::cell::Cell::new(0)
    };
}

#[cfg(feature = "std")]
impl<T> Inner<T> {
    /// allocate a lane for the calling thread. consults `free_lanes`
    /// first (a previously-allocated lane returned by an exited
    /// thread); falls back to bumping `used_lanes`. returns `None`
    /// when no recycled lane is available AND `used_lanes >= num_lanes`
    /// — the caller surfaces this via `SendError::NoLanes`. never
    /// panics.
    fn try_allocate_lane_for_thread(&self) -> Option<usize> {
        if let Some(recycled) = self.free_lanes.pop() {
            return Some(recycled);
        }
        let index = self.used_lanes.fetch_add(1, Ordering::AcqRel);
        if index < self.lanes.len() {
            Some(index)
        } else {
            // race-safe: leave used_lanes saturated; another exhausted
            // caller's fetch_add returns a higher value, still > len.
            None
        }
    }

    /// return a lane to the free pool. called by `LaneCacheEntry::drop`
    /// on producer-thread exit.
    fn release_lane(&self, lane: usize) {
        self.free_lanes.push(lane);
    }
}

/// producer handle. cheap to send across threads; `Clone` allocates the next
/// available lane (panics if the inbox's `num_lanes` is exhausted — caller
/// must size `num_lanes >= max_concurrent_producers` at construction).
pub struct Producer<T> {
    inner: Arc<Inner<T>>,
    lane_index: usize,
}

impl<T> Producer<T> {
    /// non-blocking SPSC send to this producer's dedicated lane. caller
    /// is responsible for ensuring only one thread sends on this Producer
    /// instance (the SPSC contract). For shared `Arc<Producer>` patterns
    /// where multiple threads must send through one handle, use
    /// [`try_send_mpsc`] instead.
    pub fn try_send(&self, value: T) -> Result<(), SendError<T>> {
        self.inner.try_send_on(self.lane_index, value)
    }

    /// quiesce the inbox via the producer handle. after this call, every
    /// `try_send*` (this producer or any other) returns
    /// `SendError::Closed(value)`. idempotent; consumer continues
    /// draining in-flight items normally. caller is responsible for
    /// firing any external wake (e.g. reactor user-event) so a parked
    /// consumer observes the close promptly. distinct from dropping the
    /// producer — which signals "no more producers exist" via
    /// `producer_count == 0` and yields `Disconnected` once all
    /// producers are gone. `close()` is the graceful "stop accepting,
    /// but drain in-flight" mode HTTP quiesce wants.
    pub fn close(&self) {
        self.inner.close();
    }

    /// non-blocking MPSC-safe send for `Arc<Producer>` shared across
    /// threads. Each thread is lazily assigned its own SPSC lane on
    /// first call from that thread; subsequent calls from the same
    /// thread hit a thread-local Vec cache (linear scan over typically
    /// ≤ `num_cores` entries).
    ///
    /// Use this when a single `Producer` instance is shared across
    /// producer threads — for example, when a `CoreShardHandle` lives
    /// behind `Arc<dyn Runtime>` and every spawning thread calls
    /// `spawn_on_core(...)` against the same shard. Each thread gets
    /// its own SPSC lane, preserving the per-lane SPSC contract.
    ///
    /// For the single-thread case (or the per-thread-clone pattern used
    /// in `bench_inbox`), prefer [`try_send`]: it's a single uncontended
    /// store with no thread-local lookup.
    ///
    /// Returns `Err(SendError::NoLanes(value))` if the inbox's
    /// `num_lanes` capacity is exhausted AND no recycled lane is
    /// available. Never panics.
    ///
    /// Lane assignment is reused across threads: when a producer thread
    /// exits, its `LaneCacheEntry` drops and returns the lane to the
    /// inbox's free-lane pool. The next new thread to send picks up the
    /// recycled lane. Long-running services with stable thread pools
    /// allocate once at startup; test harnesses or request-per-thread
    /// patterns recycle naturally.
    ///
    /// Not available under `alloc`-only (no `std` feature). C2
    /// (lane-ticket) provides the no_std-clean replacement.
    #[cfg(feature = "std")]
    pub fn try_send_mpsc(&self, value: T) -> Result<(), SendError<T>>
    where
        T: Send + 'static,
    {
        let inner_ptr = Arc::as_ptr(&self.inner) as usize;
        // hot path: scan the 4-slot associative cache. ~4 ptr compares
        // + branch; no RefCell borrow, no Vec scan. covers `num_cores
        // ≤ 4` dispatch patterns at zero overhead vs single-slot.
        let hot = HOT_LANES.with(Cell::get);
        let mut hit: Option<usize> = None;
        for entry in &hot {
            if entry.0 == inner_ptr && inner_ptr != 0 {
                hit = Some(entry.1);
                break;
            }
        }
        let lane = if let Some(lane) = hit {
            Some(lane)
        } else {
            // cold path: scan the per-thread Vec cache. on hit,
            // promote the entry to one of the hot slots. on miss,
            // allocate a fresh lane and add to both caches.
            self.try_send_mpsc_cold(inner_ptr)
        };
        match lane {
            Some(lane) => self.inner.try_send_on(lane, value),
            None => Err(SendError::NoLanes(value)),
        }
    }

    #[cfg(feature = "std")]
    #[cold]
    #[inline(never)]
    fn try_send_mpsc_cold(&self, inner_ptr: usize) -> Option<usize>
    where
        T: Send + 'static,
    {
        let lane = LANE_CACHE.with(|cell| -> Option<usize> {
            let mut cache = cell.borrow_mut();
            for entry in cache.iter() {
                if entry.inner_ptr == inner_ptr {
                    return Some(entry.lane);
                }
            }
            let lane = self.inner.try_allocate_lane_for_thread()?;
            let inner_for_release = self.inner.clone();
            cache.push(LaneCacheEntry {
                inner_ptr,
                lane,
                release: Some(Box::new(move || {
                    inner_for_release.release_lane(lane);
                })),
            });
            Some(lane)
        })?;
        // round-robin replace one of the hot slots.
        HOT_LANES.with(|cell| {
            let mut slots = cell.get();
            let next = HOT_LANES_NEXT.with(Cell::get);
            slots[next] = (inner_ptr, lane);
            cell.set(slots);
            HOT_LANES_NEXT.with(|cursor| {
                cursor.set((next + 1) % HOT_LANES_SIZE);
            });
        });
        Some(lane)
    }
}

impl<T> Clone for Producer<T> {
    fn clone(&self) -> Self {
        let index = self.inner.used_lanes.fetch_add(1, Ordering::AcqRel);
        assert!(
            index < self.inner.lanes.len(),
            "inbox exhausted: requested clone #{index} but num_lanes = {}",
            self.inner.lanes.len(),
        );
        self.inner.producer_count.fetch_add(1, Ordering::AcqRel);
        Self {
            inner: self.inner.clone(),
            lane_index: index,
        }
    }
}

impl<T> Drop for Producer<T> {
    fn drop(&mut self) {
        if self.inner.producer_count.fetch_sub(1, Ordering::AcqRel) == 1 {
            // last producer gone — always wake; consumer needs to see
            // disconnect regardless of whether it was parked.
            self.inner.consumer_parked.store(false, Ordering::Release);
            self.inner.waker.wake();
        }
    }
}

/// consumer handle. `!Sync` by intent — only one thread polls it at a time.
/// `Send` is fine: the consumer may be moved *to* a worker thread once at
/// construction, then it stays there. `PhantomData<core::cell::Cell<()>>`
/// gives us `Send + !Sync` (Cell is Send when T: Send, never Sync).
pub struct Consumer<T> {
    inner: Arc<Inner<T>>,
    _not_sync: PhantomData<core::cell::Cell<()>>,
}

impl<T> Consumer<T> {
    /// non-blocking drain across lanes. round-robins to keep producers fair.
    pub fn try_recv(&self) -> Result<T, TryRecvError> {
        self.inner.try_recv()
    }

    pub fn recv(&self) -> Recv<'_, T> {
        Recv { consumer: self }
    }
}

impl<T> Drop for Consumer<T> {
    fn drop(&mut self) {
        self.inner.consumer_alive.store(false, Ordering::Release);
    }
}

/// state-machine future returned by `Consumer::recv`. polls on caller's stack;
/// no `Box::pin`.
pub struct Recv<'consumer, T> {
    consumer: &'consumer Consumer<T>,
}

impl<T> Future for Recv<'_, T> {
    type Output = Result<T, RecvError>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        match self.consumer.try_recv() {
            Ok(value) => return Poll::Ready(Ok(value)),
            Err(TryRecvError::Disconnected) => return Poll::Ready(Err(RecvError)),
            Err(TryRecvError::Empty) => {}
        }
        self.consumer.inner.waker.register(context.waker());
        self.consumer
            .inner
            .consumer_parked
            .store(true, Ordering::Release);
        // Dekker-pattern fence — pairs with the one in `try_send_on`
        // after the lane push. Without it, consumer_parked.store and
        // tail.load are on different atomics and provide no
        // cross-variable visibility; both sides can observe stale
        // values and miss the wake.
        atomic::fence(Ordering::SeqCst);
        // re-check after registering + flagging parked, so producers that
        // sent between our first try_recv and our flag-set still wake us.
        match self.consumer.try_recv() {
            Ok(value) => {
                self.consumer
                    .inner
                    .consumer_parked
                    .store(false, Ordering::Release);
                Poll::Ready(Ok(value))
            }
            Err(TryRecvError::Disconnected) => {
                self.consumer
                    .inner
                    .consumer_parked
                    .store(false, Ordering::Release);
                Poll::Ready(Err(RecvError))
            }
            Err(TryRecvError::Empty) => Poll::Pending,
        }
    }
}

/// construct an inbox with `num_lanes` SPSC rings each of `lane_capacity`.
/// returns the initial `Producer` (occupying lane 0) and the single `Consumer`.
/// additional producers come from `Producer::clone` (up to `num_lanes - 1`).
/// both `num_lanes` and `lane_capacity` must be non-zero; capacity must be
/// a power of two.
#[must_use]
pub fn channel<T>(num_lanes: usize, lane_capacity: usize) -> (Producer<T>, Consumer<T>) {
    let inner = Inner::new(num_lanes, lane_capacity);
    let producer = Producer {
        inner: inner.clone(),
        lane_index: 0,
    };
    let consumer = Consumer {
        inner,
        _not_sync: PhantomData,
    };
    (producer, consumer)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::sync::Arc as StdArc;
    use core::sync::atomic::{AtomicUsize as StdAtomicUsize, Ordering as StdOrdering};
    #[cfg(feature = "std")]
    use std::thread;

    #[test]
    fn spsc_roundtrip_preserves_value() {
        let (producer, consumer) = channel::<u64>(1, 4);
        producer.try_send(42).expect("send");
        assert_eq!(consumer.try_recv().expect("recv"), 42);
    }

    #[test]
    fn try_recv_on_empty_returns_empty() {
        let (_producer, consumer) = channel::<u64>(1, 4);
        assert_eq!(consumer.try_recv(), Err(TryRecvError::Empty));
    }

    #[test]
    fn try_send_on_full_returns_payload_unharmed() {
        let (producer, _consumer) = channel::<u64>(1, 4);
        for index in 0..4 {
            producer.try_send(index).expect("fill");
        }
        match producer.try_send(99) {
            Err(SendError::Full(payload)) => assert_eq!(payload, 99),
            other => panic!("expected Full(99), got {other:?}"),
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn mpsc_fanin_preserves_count() {
        const PRODUCERS: usize = 4;
        const PER_PRODUCER: usize = 10_000;
        let (producer, consumer) = channel::<u64>(PRODUCERS, 1024);
        let mut threads = Vec::with_capacity(PRODUCERS);
        let mut handles = Vec::with_capacity(PRODUCERS);
        handles.push(producer.clone());
        handles.push(producer.clone());
        handles.push(producer.clone());
        handles.push(producer);
        for handle in handles {
            threads.push(thread::spawn(move || {
                for index in 0..PER_PRODUCER {
                    loop {
                        match handle.try_send(index as u64) {
                            Ok(()) => break,
                            Err(SendError::Full(_)) => thread::yield_now(),
                            Err(SendError::Disconnected(_)) => panic!("disconnected"),
                            Err(SendError::NoLanes(_)) => {
                                panic!("inbox under-sized — should not hit NoLanes in this test")
                            }
                            Err(SendError::Closed(_)) => panic!("unexpected Closed in this test"),
                        }
                    }
                }
            }));
        }
        let total = PRODUCERS * PER_PRODUCER;
        let mut received: usize = 0;
        while received < total {
            match consumer.try_recv() {
                Ok(_) => received += 1,
                Err(TryRecvError::Empty) => thread::yield_now(),
                Err(TryRecvError::Disconnected) => break,
            }
        }
        for thread in threads {
            thread.join().expect("producer join");
        }
        assert_eq!(received, total);
    }

    #[test]
    fn recv_future_wakes_on_send() {
        use alloc::task::Wake;
        use core::task::{Context, Waker};

        struct CountingWaker(StdAtomicUsize);
        impl Wake for CountingWaker {
            fn wake(self: StdArc<Self>) {
                self.0.fetch_add(1, StdOrdering::AcqRel);
            }
        }

        let (producer, consumer) = channel::<u64>(1, 4);
        let counter = StdArc::new(CountingWaker(StdAtomicUsize::new(0)));
        let waker: Waker = counter.clone().into();
        let mut context = Context::from_waker(&waker);

        {
            let mut future = consumer.recv();
            let pinned = unsafe { Pin::new_unchecked(&mut future) };
            assert!(pinned.poll(&mut context).is_pending());
        }
        assert_eq!(counter.0.load(StdOrdering::Acquire), 0);

        producer.try_send(7).expect("send");
        assert!(counter.0.load(StdOrdering::Acquire) >= 1);

        let mut future = consumer.recv();
        let pinned = unsafe { Pin::new_unchecked(&mut future) };
        match pinned.poll(&mut context) {
            Poll::Ready(Ok(value)) => assert_eq!(value, 7),
            other => panic!("expected Ready(Ok(7)), got {other:?}"),
        }
    }

    #[test]
    fn last_producer_drop_yields_recv_error() {
        let (producer, consumer) = channel::<u64>(2, 4);
        let producer_clone = producer.clone();
        drop(producer);
        drop(producer_clone);
        assert_eq!(consumer.try_recv(), Err(TryRecvError::Disconnected));
    }

    #[test]
    fn send_after_consumer_drop_returns_disconnected() {
        let (producer, consumer) = channel::<u64>(1, 4);
        drop(consumer);
        match producer.try_send(1) {
            Err(SendError::Disconnected(value)) => assert_eq!(value, 1),
            other => panic!("expected Disconnected(1), got {other:?}"),
        }
    }

    #[test]
    fn payload_drop_runs_for_values_left_in_ring() {
        let drops = StdArc::new(StdAtomicUsize::new(0));
        #[derive(Debug)]
        struct Counted(StdArc<StdAtomicUsize>);
        impl Drop for Counted {
            fn drop(&mut self) {
                self.0.fetch_add(1, StdOrdering::AcqRel);
            }
        }
        let (producer, consumer) = channel::<Counted>(1, 4);
        producer
            .try_send(Counted(drops.clone()))
            .expect("send first");
        producer
            .try_send(Counted(drops.clone()))
            .expect("send second");
        drop(producer);
        drop(consumer);
        assert_eq!(drops.load(StdOrdering::Acquire), 2);
    }

    #[cfg(feature = "std")]
    #[test]
    fn cloning_past_num_lanes_panics() {
        let (producer, _consumer) = channel::<u64>(2, 4);
        let _clone1 = producer.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _clone2 = producer.clone();
        }));
        assert!(result.is_err(), "expected panic on lane exhaustion");
    }

    /// Happy-path Bug B: N producer threads share ONE `Producer` via
    /// `Arc<Producer>` and each calls `try_send_mpsc`. Each thread gets
    /// its own SPSC lane (lazily allocated on first call), so concurrent
    /// sends never race on the same lane's head/tail. Verifies all values
    /// arrive once (no loss, no duplication).
    #[cfg(feature = "std")]
    #[test]
    fn try_send_mpsc_routes_each_thread_to_its_own_lane() {
        const THREADS: usize = 4;
        const PER_THREAD: usize = 5_000;
        // num_lanes must accommodate the initial Producer's lane 0 plus
        // one lane per mpsc thread.
        let (producer, consumer) = channel::<u64>(THREADS + 1, 1024);
        let producer = StdArc::new(producer);
        let mut handles = Vec::with_capacity(THREADS);
        for thread_index in 0..THREADS {
            let producer = producer.clone();
            handles.push(std::thread::spawn(move || {
                for index in 0..PER_THREAD {
                    let value = (thread_index as u64) * (PER_THREAD as u64) + index as u64;
                    loop {
                        match producer.try_send_mpsc(value) {
                            Ok(()) => break,
                            Err(SendError::Full(_)) => std::thread::yield_now(),
                            Err(SendError::Disconnected(_)) => panic!("disconnected"),
                            Err(SendError::NoLanes(_)) => {
                                panic!("inbox under-sized — should not hit NoLanes in this test")
                            }
                            Err(SendError::Closed(_)) => panic!("unexpected Closed in this test"),
                        }
                    }
                }
            }));
        }
        let total = THREADS * PER_THREAD;
        let mut received: Vec<u64> = Vec::with_capacity(total);
        while received.len() < total {
            match consumer.try_recv() {
                Ok(value) => received.push(value),
                Err(TryRecvError::Empty) => std::thread::yield_now(),
                Err(TryRecvError::Disconnected) => panic!("consumer disconnected"),
            }
        }
        for handle in handles {
            handle.join().expect("producer thread join");
        }
        assert_eq!(received.len(), total);
        received.sort_unstable();
        let mut expected: Vec<u64> = Vec::with_capacity(total);
        for thread_index in 0..THREADS {
            for index in 0..PER_THREAD {
                expected.push((thread_index as u64) * (PER_THREAD as u64) + index as u64);
            }
        }
        expected.sort_unstable();
        assert_eq!(received, expected, "no loss + no duplication");
    }

    /// Edge: the same thread calling `try_send_mpsc` multiple times must
    /// keep using the same lane (thread-local cache hit), not allocate a
    /// fresh one each call. Verified by sending more than `num_lanes`
    /// items from one thread — if the cache miss were on every call,
    /// `used_lanes` would exhaust.
    #[cfg(feature = "std")]
    #[test]
    fn try_send_mpsc_caches_lane_per_thread() {
        let (producer, consumer) = channel::<u64>(2, 64);
        for index in 0..16 {
            producer.try_send_mpsc(index).expect("send");
        }
        let mut total = 0_u64;
        loop {
            match consumer.try_recv() {
                Ok(value) => total += value,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
        let expected: u64 = (0..16).sum();
        assert_eq!(total, expected);
    }

    /// Sad: when CONCURRENT threads exceed `num_lanes` (lanes pinned by
    /// live threads, none yet exited and recycled), the next thread's
    /// first `try_send_mpsc` returns `Err(SendError::NoLanes(value))` —
    /// it MUST NOT panic. Threads are held alive via a Barrier so the
    /// lane-recycling Drop hook can't return their lanes mid-test.
    #[cfg(feature = "std")]
    #[test]
    fn try_send_mpsc_returns_no_lanes_when_concurrent_threads_exceed_num_lanes() {
        let (producer, _consumer) = channel::<u64>(3, 4);
        let producer = StdArc::new(producer);
        // gate: producer threads A and B park here until C has tried
        // its send, so their lane assignments are NOT yet recycled.
        let gate = StdArc::new(std::sync::Barrier::new(3));

        let producer_a = producer.clone();
        let gate_a = gate.clone();
        let handle_a = std::thread::spawn(move || {
            producer_a.try_send_mpsc(1).expect("send a");
            gate_a.wait();
        });
        let producer_b = producer.clone();
        let gate_b = gate.clone();
        let handle_b = std::thread::spawn(move || {
            producer_b.try_send_mpsc(2).expect("send b");
            gate_b.wait();
        });
        // small spin until A and B have both registered their lanes.
        // when free_lanes is empty AND used_lanes >= num_lanes, the
        // third thread's first call must return NoLanes.
        while producer.inner.used_lanes.load(Ordering::Acquire) < producer.inner.lanes.len() {
            std::thread::yield_now();
        }
        let producer_c = producer.clone();
        let outcome = std::thread::spawn(move || producer_c.try_send_mpsc(99))
            .join()
            .expect("c join");
        // release A and B.
        gate.wait();
        handle_a.join().expect("a join");
        handle_b.join().expect("b join");
        match outcome {
            Err(SendError::NoLanes(value)) => assert_eq!(value, 99),
            other => panic!("expected Err(NoLanes(99)) — got {other:?}"),
        }
    }

    /// Concurrency: drives an Arc-shared producer at high fan-in with
    /// short bursts and verifies the consumer drains everything once.
    /// This is the regression test for the original Bug B race
    /// (concurrent `try_send` on a single SPSC lane caused data loss).
    #[cfg(feature = "std")]
    #[test]
    fn try_send_mpsc_high_fanin_no_data_loss() {
        const THREADS: usize = 8;
        const PER_THREAD: usize = 2_000;
        // num_lanes = THREADS + 1: lane 0 reserved for the initial
        // Producer's `try_send`; lanes 1..=THREADS for each mpsc thread.
        let (producer, consumer) = channel::<u64>(THREADS + 1, 256);
        let producer = StdArc::new(producer);
        let mut handles = Vec::with_capacity(THREADS);
        for thread_index in 0..THREADS {
            let producer = producer.clone();
            handles.push(std::thread::spawn(move || {
                for index in 0..PER_THREAD {
                    let value = (thread_index as u64) << 32 | index as u64;
                    loop {
                        match producer.try_send_mpsc(value) {
                            Ok(()) => break,
                            Err(SendError::Full(_)) => std::thread::yield_now(),
                            Err(SendError::Disconnected(_)) => panic!("disconnected"),
                            Err(SendError::NoLanes(_)) => {
                                panic!("inbox under-sized — should not hit NoLanes in this test")
                            }
                            Err(SendError::Closed(_)) => panic!("unexpected Closed in this test"),
                        }
                    }
                }
            }));
        }
        let total = THREADS * PER_THREAD;
        let mut received: usize = 0;
        let mut counts_per_thread = [0_usize; THREADS];
        while received < total {
            match consumer.try_recv() {
                Ok(value) => {
                    let thread_index = (value >> 32) as usize;
                    assert!(thread_index < THREADS);
                    counts_per_thread[thread_index] += 1;
                    received += 1;
                }
                Err(TryRecvError::Empty) => std::thread::yield_now(),
                Err(TryRecvError::Disconnected) => panic!("consumer disconnected"),
            }
        }
        for handle in handles {
            handle.join().expect("join");
        }
        for (index, count) in counts_per_thread.iter().enumerate() {
            assert_eq!(*count, PER_THREAD, "thread {index} lost values: {count}");
        }
    }

    /// Bug B lane recycling: producer threads come and go, but the
    /// inbox's `num_lanes` capacity must NOT monotonically deplete.
    /// Spawn many short-lived producer threads sequentially against an
    /// inbox with very few lanes. Each thread allocates a lane via
    /// `try_send_mpsc`, sends a value, and exits — its `LaneCacheEntry`
    /// drops, returning the lane. The next thread picks up the same
    /// lane via `free_lanes`. Total threads ≫ `num_lanes` succeed
    /// without `SendError::NoLanes`.
    #[cfg(feature = "std")]
    #[test]
    fn try_send_mpsc_recycles_lanes_when_threads_exit() {
        const NUM_LANES: usize = 2; // lane 0 reserved for initial Producer; 1 mpsc slot
        const ROUNDS: usize = 64;
        let (producer, consumer) = channel::<u64>(NUM_LANES, 4);
        let producer = StdArc::new(producer);
        let mut received = 0_usize;
        for round in 0..ROUNDS {
            let producer_for_thread = producer.clone();
            std::thread::spawn(move || {
                producer_for_thread
                    .try_send_mpsc(round as u64)
                    .expect("send must succeed after recycling");
            })
            .join()
            .expect("join");
            // drain so the recycled lane is empty for the next thread.
            // without this, sent values pile up in the same lane round
            // after round and the test exercises lane Full instead of
            // lane recycling.
            while consumer.try_recv().is_ok() {
                received += 1;
            }
        }
        // sanity: `used_lanes` MUST NOT have grown unbounded. one
        // initial allocation + at most one fetch_add miss after first
        // recycle; rest come from free_lanes.
        let used = producer.inner.used_lanes.load(Ordering::Acquire);
        assert_eq!(received, ROUNDS, "every send delivered exactly once");
        assert!(
            used <= NUM_LANES + 1,
            "used_lanes monotonically grew to {used} despite recycling — \
             cache entries failed to release lanes on thread exit",
        );
    }

    /// SPSC fast path: a single thread using `try_send` (the SPSC-only
    /// method) must still work. The Bug B fix ADDS `try_send_mpsc`
    /// without changing `try_send` — bench_inbox's single-Producer SPSC
    /// win against flume is preserved by this.
    #[test]
    fn try_send_still_works_for_single_producer() {
        let (producer, consumer) = channel::<u64>(1, 4);
        for value in 0..4 {
            producer.try_send(value).expect("send");
        }
        for expected in 0..4 {
            assert_eq!(consumer.try_recv().expect("recv"), expected);
        }
    }

    // ---- closed / quiesce semantics ----

    /// after `Producer::close`, subsequent `try_send` returns `Closed`.
    /// The fast-path capacity check naturally routes closed pushes
    /// through the slow path; no branch was added to the hot path.
    #[test]
    fn closed_inbox_rejects_new_pushes_via_try_send() {
        let (producer, _consumer) = channel::<u64>(1, 16);
        producer.try_send(1).expect("pre-close send");
        producer.close();
        match producer.try_send(2) {
            Err(SendError::Closed(value)) => assert_eq!(value, 2),
            other => panic!("expected Closed(2), got {other:?}"),
        }
    }

    /// closing does NOT discard in-flight items. The consumer drains
    /// everything pushed before close() returns, regardless of how many.
    #[test]
    fn closed_inbox_preserves_in_flight_items() {
        let (producer, consumer) = channel::<u64>(1, 16);
        for value in 0..10 {
            producer.try_send(value).expect("pre-close send");
        }
        producer.close();
        // post-close push fails
        match producer.try_send(99) {
            Err(SendError::Closed(value)) => assert_eq!(value, 99),
            other => panic!("expected Closed(99), got {other:?}"),
        }
        // consumer drains all 10 in order.
        for expected in 0..10_u64 {
            assert_eq!(consumer.try_recv().expect("drain"), expected);
        }
        // after drain, queue is Empty (not Disconnected — producer still alive).
        assert_eq!(consumer.try_recv(), Err(TryRecvError::Empty));
    }

    /// quiesce is idempotent: calling close() repeatedly is a no-op
    /// (`fetch_or` of CLOSED_BIT is idempotent, no internal state
    /// corruption). Crucial for graceful-shutdown handshakes where a
    /// caller may or may not have already quiesced.
    #[test]
    fn closed_idempotent() {
        let (producer, _consumer) = channel::<u64>(1, 4);
        producer.close();
        producer.close();
        producer.close();
        match producer.try_send(1) {
            Err(SendError::Closed(_)) => {}
            other => panic!("expected Closed, got {other:?}"),
        }
    }

    /// stress test: many producer threads, single closer (main).
    /// guarantees: count(successful sends) + count(Closed errors) ==
    /// total attempts; consumer drains exactly count(successful sends).
    /// no payload corruption or lost items.
    ///
    /// Consumer is `!Send` (SPSC contract) so it stays on the main
    /// thread. A separate closer thread fires after >=100 successful
    /// pushes so the race window is non-trivial (thread-spawn latency
    /// varies across platforms — count-based handshake is reliable).
    #[cfg(feature = "std")]
    #[test]
    fn closed_races_with_concurrent_pushes() {
        use std::thread;
        const PRODUCERS: usize = 4;
        const PUSHES_PER_PRODUCER: usize = 5_000;
        const TOTAL: usize = PRODUCERS * PUSHES_PER_PRODUCER;

        // PRODUCERS + 1 lanes: lane 0 reserved for original Producer;
        // each producer thread via try_send_mpsc allocates a new lane.
        let (producer, consumer) = channel::<u64>(PRODUCERS + 1, 256);
        let producer = StdArc::new(producer);
        let success_count = StdArc::new(AtomicUsize::new(0));
        let closed_count = StdArc::new(AtomicUsize::new(0));
        let other_err_count = StdArc::new(AtomicUsize::new(0));
        let producers_done = StdArc::new(AtomicUsize::new(0));

        let producer_for_closer = producer.clone();
        let success_for_closer = success_count.clone();
        let closer_handle = thread::spawn(move || {
            while success_for_closer.load(Ordering::Acquire) < 100 {
                core::hint::spin_loop();
            }
            producer_for_closer.close();
        });

        let mut handles = Vec::new();
        for thread_id in 0..PRODUCERS {
            let producer = producer.clone();
            let success_count = success_count.clone();
            let closed_count = closed_count.clone();
            let other_err_count = other_err_count.clone();
            let producers_done = producers_done.clone();
            handles.push(thread::spawn(move || {
                for index in 0..PUSHES_PER_PRODUCER {
                    let value = (thread_id * PUSHES_PER_PRODUCER + index) as u64;
                    loop {
                        match producer.try_send_mpsc(value) {
                            Ok(()) => {
                                success_count.fetch_add(1, Ordering::Relaxed);
                                break;
                            }
                            Err(SendError::Full(_)) => {
                                std::thread::yield_now();
                            }
                            Err(SendError::Closed(_)) => {
                                closed_count.fetch_add(1, Ordering::Relaxed);
                                break;
                            }
                            Err(_) => {
                                other_err_count.fetch_add(1, Ordering::Relaxed);
                                break;
                            }
                        }
                    }
                }
                producers_done.fetch_add(1, Ordering::Release);
            }));
        }

        // drain on main thread; the consumer is !Send so it stays here.
        let mut drained: usize = 0;
        loop {
            while consumer.try_recv().is_ok() {
                drained += 1;
            }
            if producers_done.load(Ordering::Acquire) == PRODUCERS {
                while consumer.try_recv().is_ok() {
                    drained += 1;
                }
                break;
            }
            std::thread::yield_now();
        }

        closer_handle.join().expect("closer");
        for handle in handles {
            handle.join().expect("producer thread");
        }

        let success = success_count.load(Ordering::Relaxed);
        let closed = closed_count.load(Ordering::Relaxed);
        let other = other_err_count.load(Ordering::Relaxed);

        assert_eq!(other, 0, "unexpected SendError variant fired {other}x");
        assert_eq!(
            success + closed,
            TOTAL,
            "success={success} + closed={closed} != total={TOTAL}",
        );
        assert_eq!(
            drained, success,
            "drained={drained} != success={success}; item leak or corruption",
        );
        assert!(success > 0, "no pushes succeeded — close was instant?");
    }

    /// Verify that the closed-bit encoding doesn't corrupt the
    /// position counter. After close(), position keeps tracking
    /// pushes (which now fail) but the bit is preserved across the
    /// producer's CAS publish. Consumer reads of tail mask out the
    /// bit; pre-close items drain in order.
    #[test]
    fn closed_bit_does_not_corrupt_position_counter() {
        let (producer, consumer) = channel::<u64>(1, 16);
        for value in 0..5_u64 {
            producer.try_send(value).expect("send");
        }
        producer.close();
        for expected in 0..5_u64 {
            assert_eq!(consumer.try_recv().expect("drain"), expected);
        }
        assert_eq!(consumer.try_recv(), Err(TryRecvError::Empty));
        match producer.try_send(99) {
            Err(SendError::Closed(value)) => assert_eq!(value, 99),
            other => panic!("expected Closed(99), got {other:?}"),
        }
    }
}

// ---- inbox-dynamic: [floor, ceiling] lane pool, lazy-alloc, no task loss ----
//
// `runtime-prime-inbox-dynamic` replaces the fixed-array incumbent with a
// configurable lane pool. Floor lanes are allocated eagerly at channel();
// additional lanes grow lazily on demand up to ceiling; at ceiling the caller
// gets a transient Busy error (never silent drop, never panic). Producer::clone
// is decoupled from lane allocation — clone bumps producer_count + shares the
// Arc<Inner>; the lane is claimed on first send via the cold path. Per-send
// hot path: a cached *const Lane<T> set at claim time → single deref, no walk.
//
// C1 implements ReleasePolicy::Never (hold high-water; recycle via free_lanes).
// Always (C2) adds crossbeam-epoch reclamation of above-floor idle lanes.
#[cfg(feature = "runtime-prime-inbox-dynamic")]
pub mod inbox_dynamic {
    extern crate alloc;

    use alloc::boxed::Box;
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use core::cell::{Cell, UnsafeCell};
    use core::fmt;
    use core::future::Future;
    use core::marker::PhantomData;
    use core::mem::MaybeUninit;
    use core::pin::Pin;
    use core::str::FromStr;
    use core::sync::atomic::{self, AtomicBool, AtomicPtr, AtomicUsize, Ordering};
    use core::task::{Context, Poll};

    use atomic_waker::AtomicWaker;
    use crossbeam_queue::SegQueue;
    use crossbeam_utils::CachePadded;

    use super::{CLOSED_BIT, POSITION_MASK};

    // lanes per chunk. 64 keeps the chunk-pointer array tiny and chunk
    // allocations rare even at large producer counts.
    const CHUNK_SIZE: usize = 64;

    // hard upper limit on chunks in the registry. 1024 chunks × 64 lanes =
    // 65536 lanes maximum. unbounded (ceiling=0) is bounded here in practice.
    const MAX_CHUNKS: usize = 1024;

    // ---- error types ----

    enum ClaimError {
        AtCeiling,
    }

    /// reclamation policy for above-floor idle lane rings.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub enum ReleasePolicy {
        /// never release — hold at high-water; recycle the ring in place via
        /// free_lanes. zero reclamation cost. opt-in for bare-metal /
        /// `floor==ceiling` provision-once setups where the producer set is known
        /// and bounded. NOT the default: in a long-running process a transient
        /// producer spike would be retained for the whole process life.
        Never,
        /// reclaim: the consumer frees an above-floor lane's ring once its
        /// producer has dropped and the lane is drained, so resident memory
        /// tracks live producers — a spike is released once it settles. the
        /// DEFAULT. cost is proportional to producer CHURN, not steady state:
        /// it ties Never on the single-producer hot path (measured 84.2 vs 84.4
        /// Melem/s, host-b) because reclaim only runs on a fast-path miss
        /// over a non-empty abandoned queue. only continuous churn (producers
        /// dropping every iteration) pays — ~19% on the fan-in p64 bench — which
        /// a long-running daemon does not do.
        #[default]
        Always,
    }

    /// parse error for config fields.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ParseError(alloc::string::String);

    impl fmt::Display for ParseError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str(&self.0)
        }
    }

    impl core::error::Error for ParseError {}

    impl FromStr for ReleasePolicy {
        type Err = ParseError;

        fn from_str(input: &str) -> Result<Self, Self::Err> {
            match input.trim().to_ascii_lowercase().as_str() {
                "never" | "hold" | "none" => Ok(Self::Never),
                "always" => Ok(Self::Always),
                other => Err(ParseError(alloc::format!(
                    "release_policy: '{other}' is not 'never' or 'always'"
                ))),
            }
        }
    }

    // ---- config + builder (gate point 12: both entry points, equal state) ----

    /// configuration for the dynamic inbox lane pool.
    ///
    /// create via `InboxDynamicConfig::default()` (num_physical_cores floor,
    /// 1024 ceiling, Never release, 1024 lane_capacity) or via the fluent
    /// `InboxDynamicConfig::builder()`. Both produce equal state for the same
    /// knob values — verified by the gate point-12 fixture in `tests`.
    ///
    /// `[inbox]` section in `prime-runtime.toml` maps to this struct.
    /// `lanes_per_core` × `N` + `lanes_headroom` determines `floor`;
    /// `lanes_ceiling` maps to `ceiling`.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct InboxDynamicConfig {
        /// lanes allocated eagerly at channel(). default = num physical cores.
        pub floor: usize,
        /// maximum lanes; 0 = unbounded (capped by MAX_CHUNKS * CHUNK_SIZE).
        pub ceiling: usize,
        /// reclamation policy (C1: only Never is active).
        pub release: ReleasePolicy,
        /// ring capacity per lane; must be a non-zero power of two.
        pub lane_capacity: usize,
    }

    impl Default for InboxDynamicConfig {
        fn default() -> Self {
            let cores = num_cpus::get_physical().max(1);
            Self {
                floor: cores,
                ceiling: 1024,
                release: ReleasePolicy::Always,
                lane_capacity: 1024,
            }
        }
    }

    impl InboxDynamicConfig {
        #[must_use]
        pub fn builder() -> Builder {
            Builder::default()
        }

        fn effective_ceiling(&self) -> usize {
            if self.ceiling == 0 {
                MAX_CHUNKS * CHUNK_SIZE
            } else {
                self.ceiling
            }
        }
    }

    /// fluent builder for `InboxDynamicConfig`. gate point 12 requires this
    /// and the struct literal to produce equal state for identical knobs.
    #[derive(Debug, Clone)]
    pub struct Builder {
        floor: usize,
        ceiling: usize,
        release: ReleasePolicy,
        lane_capacity: usize,
    }

    impl Default for Builder {
        fn default() -> Self {
            let defaults = InboxDynamicConfig::default();
            Self {
                floor: defaults.floor,
                ceiling: defaults.ceiling,
                release: defaults.release,
                lane_capacity: defaults.lane_capacity,
            }
        }
    }

    impl Builder {
        #[must_use]
        pub fn floor(mut self, lanes: usize) -> Self {
            self.floor = lanes;
            self
        }

        #[must_use]
        pub fn ceiling(mut self, lanes: usize) -> Self {
            self.ceiling = lanes;
            self
        }

        #[must_use]
        pub fn release(mut self, policy: ReleasePolicy) -> Self {
            self.release = policy;
            self
        }

        #[must_use]
        pub fn lane_capacity(mut self, capacity: usize) -> Self {
            self.lane_capacity = capacity;
            self
        }

        #[must_use]
        pub fn build(self) -> InboxDynamicConfig {
            InboxDynamicConfig {
                floor: self.floor,
                ceiling: self.ceiling,
                release: self.release,
                lane_capacity: self.lane_capacity,
            }
        }
    }

    // ---- send / recv errors ----

    #[derive(Debug)]
    pub enum SendError<T> {
        /// lane ring is full — transient; retry or back-pressure.
        Full(T),
        /// consumer dropped — terminal.
        Disconnected(T),
        /// inbox quiesced — no new pushes accepted.
        Closed(T),
        /// at lane ceiling AND no recycled lanes available — transient; retry.
        /// the item is NEVER dropped; the caller decides backpressure policy.
        Busy(T),
    }

    impl<T> SendError<T> {
        pub fn into_inner(self) -> T {
            match self {
                Self::Full(value)
                | Self::Disconnected(value)
                | Self::Closed(value)
                | Self::Busy(value) => value,
            }
        }
    }

    impl<T> fmt::Display for SendError<T> {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Full(_) => formatter.write_str("inbox lane full"),
                Self::Disconnected(_) => formatter.write_str("inbox consumer dropped"),
                Self::Closed(_) => formatter.write_str("inbox quiesced (closed)"),
                Self::Busy(_) => formatter.write_str("inbox at lane ceiling — retry"),
            }
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    pub enum TryRecvError {
        Empty,
        Disconnected,
    }

    impl fmt::Display for TryRecvError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Empty => formatter.write_str("inbox empty"),
                Self::Disconnected => formatter.write_str("inbox all producers dropped"),
            }
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    pub struct RecvError;

    impl fmt::Display for RecvError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("inbox closed")
        }
    }

    // ---- Lane (hot path identical to incumbent inbox-alloc) ----

    enum LaneSendErr<T> {
        Full(T),
        Closed(T),
    }

    /// per-producer SPSC ring. identical algorithm to the incumbent `Lane<T>`
    /// in inbox-alloc. cached_head (producer-private) and cached_tail
    /// (consumer-private) avoid the Acquire load on the common path.
    struct Lane<T> {
        head: CachePadded<AtomicUsize>,
        tail: CachePadded<AtomicUsize>,
        cached_head: CachePadded<UnsafeCell<usize>>,
        cached_tail: CachePadded<UnsafeCell<usize>>,
        capacity: usize,
        mask: usize,
        slots: Box<[UnsafeCell<MaybeUninit<T>>]>,
    }

    // SAFETY: single producer writes; single consumer reads — guaranteed by the
    // SPSC contract enforced via lane ownership. no two threads touch one slot
    // concurrently.
    unsafe impl<T: Send> Sync for Lane<T> {}
    unsafe impl<T: Send> Send for Lane<T> {}

    impl<T> Lane<T> {
        fn new(capacity: usize) -> Self {
            assert!(
                capacity > 0 && capacity.is_power_of_two(),
                "lane capacity must be a non-zero power of two; got {capacity}",
            );
            let mut slots: Vec<UnsafeCell<MaybeUninit<T>>> = Vec::with_capacity(capacity);
            for _ in 0..capacity {
                slots.push(UnsafeCell::new(MaybeUninit::uninit()));
            }
            Self {
                head: CachePadded::new(AtomicUsize::new(0)),
                tail: CachePadded::new(AtomicUsize::new(0)),
                cached_head: CachePadded::new(UnsafeCell::new(0)),
                cached_tail: CachePadded::new(UnsafeCell::new(0)),
                capacity,
                mask: capacity - 1,
                slots: slots.into_boxed_slice(),
            }
        }

        #[inline(always)]
        fn try_send(&self, value: T) -> Result<(), LaneSendErr<T>> {
            let raw_tail = self.tail.load(Ordering::Relaxed);
            // SAFETY: single producer reads/writes cached_head.
            let cached_head = unsafe { *self.cached_head.get() };
            if raw_tail.wrapping_sub(cached_head) >= self.capacity {
                let head = self.head.load(Ordering::Acquire);
                // SAFETY: single producer
                unsafe { *self.cached_head.get() = head };
                if raw_tail.wrapping_sub(head) >= self.capacity {
                    if raw_tail & CLOSED_BIT != 0 {
                        return Err(LaneSendErr::Closed(value));
                    }
                    return Err(LaneSendErr::Full(value));
                }
            }
            let position = raw_tail & POSITION_MASK;
            // SAFETY: single producer owns slot[position & mask] until publish.
            unsafe {
                (*self.slots[position & self.mask].get()).write(value);
            }
            let mut current = raw_tail;
            loop {
                let next_position = (current & POSITION_MASK).wrapping_add(1) & POSITION_MASK;
                let next = next_position | (current & CLOSED_BIT);
                match self.tail.compare_exchange_weak(
                    current,
                    next,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return Ok(()),
                    Err(actual) => current = actual,
                }
            }
        }

        #[inline(always)]
        fn try_recv(&self) -> Option<T> {
            let head = self.head.load(Ordering::Relaxed);
            // SAFETY: single consumer reads/writes cached_tail.
            let cached_tail = unsafe { *self.cached_tail.get() };
            if head == cached_tail {
                let raw_tail = self.tail.load(Ordering::Acquire);
                let tail = raw_tail & POSITION_MASK;
                // SAFETY: single consumer
                unsafe { *self.cached_tail.get() = tail };
                if head == tail {
                    return None;
                }
            }
            // SAFETY: cached_tail > head → producer published this slot.
            let value = unsafe { (*self.slots[head & self.mask].get()).assume_init_read() };
            self.head.store(head.wrapping_add(1), Ordering::Release);
            Some(value)
        }

        fn close(&self) {
            self.tail.fetch_or(CLOSED_BIT, Ordering::Release);
        }

        /// true when the consumer has drained every published slot. used by
        /// ReleasePolicy::Always reclamation: only an empty, producer-abandoned
        /// lane is freed (so no in-flight task is ever lost).
        fn is_empty(&self) -> bool {
            let head = self.head.load(Ordering::Acquire);
            let tail = self.tail.load(Ordering::Acquire) & POSITION_MASK;
            head == tail
        }

        fn drop_remaining(&mut self) {
            let head = *self.head.get_mut();
            let tail = *self.tail.get_mut() & POSITION_MASK;
            let mut position = head;
            while position != tail {
                let slot = &mut self.slots[position & self.mask];
                // SAFETY: positions in [head, tail) hold initialized payloads.
                unsafe { slot.get_mut().assume_init_drop() };
                position = position.wrapping_add(1);
            }
        }
    }

    impl<T> Drop for Lane<T> {
        fn drop(&mut self) {
            self.drop_remaining();
        }
    }

    // ---- chunked lane registry ----
    //
    // A fixed-size array of AtomicPtr<LaneChunk<T>>. Each chunk holds
    // CHUNK_SIZE lane slots (AtomicPtr<Lane<T>>). Chunks allocate on demand;
    // once written, a chunk pointer is never changed or freed until the Registry
    // drops — so *const Lane<T> pointers cached by Producers are always valid.
    //
    // lane_index → chunk_index = lane_index / CHUNK_SIZE
    //            → slot_index  = lane_index % CHUNK_SIZE
    //
    // null slot within a chunk = lane not yet allocated (beyond floor, not yet
    // claimed). null chunk pointer = chunk not yet needed.

    struct LaneChunk<T> {
        slots: [AtomicPtr<Lane<T>>; CHUNK_SIZE],
    }

    impl<T> LaneChunk<T> {
        fn new() -> Box<Self> {
            // SAFETY: AtomicPtr<Lane<T>> is valid all-zeros (null ptr); zeroed
            // memory is a correct representation for this type.
            unsafe {
                let layout = alloc::alloc::Layout::new::<Self>();
                let raw = alloc::alloc::alloc_zeroed(layout) as *mut Self;
                assert!(!raw.is_null(), "LaneChunk allocation failed");
                Box::from_raw(raw)
            }
        }
    }

    impl<T> Drop for LaneChunk<T> {
        fn drop(&mut self) {
            for slot in &self.slots {
                let raw = slot.load(Ordering::Relaxed);
                if !raw.is_null() {
                    // SAFETY: allocated via Box::into_raw in Registry::publish_lane.
                    drop(unsafe { Box::from_raw(raw) });
                }
            }
        }
    }

    struct Registry<T> {
        chunks: [AtomicPtr<LaneChunk<T>>; MAX_CHUNKS],
    }

    impl<T> Registry<T> {
        fn new() -> Box<Self> {
            // SAFETY: AtomicPtr<LaneChunk<T>> is valid all-zeros.
            unsafe {
                let layout = alloc::alloc::Layout::new::<Self>();
                let raw = alloc::alloc::alloc_zeroed(layout) as *mut Self;
                assert!(!raw.is_null(), "Registry allocation failed");
                Box::from_raw(raw)
            }
        }

        fn ensure_chunk(&self, lane_index: usize) -> *mut LaneChunk<T> {
            let chunk_index = lane_index / CHUNK_SIZE;
            debug_assert!(chunk_index < MAX_CHUNKS);
            let chunk_slot = &self.chunks[chunk_index];
            let existing = chunk_slot.load(Ordering::Acquire);
            if !existing.is_null() {
                return existing;
            }
            let new_chunk = Box::into_raw(LaneChunk::<T>::new());
            match chunk_slot.compare_exchange(
                core::ptr::null_mut(),
                new_chunk,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => new_chunk,
                Err(winner) => {
                    // another thread won; free our allocation and use theirs.
                    drop(unsafe { Box::from_raw(new_chunk) });
                    winner
                }
            }
        }

        /// write a lane into the registry with Release ordering so the
        /// consumer's Acquire in lane_ptr_at observes the initialized Lane.
        fn publish_lane(&self, lane_index: usize, lane: Box<Lane<T>>) {
            let chunk = self.ensure_chunk(lane_index);
            let slot_index = lane_index % CHUNK_SIZE;
            let raw = Box::into_raw(lane);
            // SAFETY: chunk is non-null (ensure_chunk guarantees it).
            unsafe { (*chunk).slots[slot_index].store(raw, Ordering::Release) };
        }

        /// load a lane pointer. null = not yet allocated. Acquire pairs with
        /// the Release in publish_lane.
        fn lane_ptr_at(&self, lane_index: usize) -> *const Lane<T> {
            let chunk_index = lane_index / CHUNK_SIZE;
            if chunk_index >= MAX_CHUNKS {
                return core::ptr::null();
            }
            let chunk_raw = self.chunks[chunk_index].load(Ordering::Acquire);
            if chunk_raw.is_null() {
                return core::ptr::null();
            }
            // SAFETY: chunk_raw is non-null and stable.
            unsafe { (*chunk_raw).slots[lane_index % CHUNK_SIZE].load(Ordering::Acquire) }
        }

        /// atomically null a lane slot and return the previous pointer, so the
        /// (single) consumer can free the ring under ReleasePolicy::Always. the
        /// chunk pointer itself is never freed here, so the address space stays
        /// stable; a later claim of this index re-allocates a fresh ring.
        fn take_lane(&self, lane_index: usize) -> *mut Lane<T> {
            let chunk_index = lane_index / CHUNK_SIZE;
            if chunk_index >= MAX_CHUNKS {
                return core::ptr::null_mut();
            }
            let chunk_raw = self.chunks[chunk_index].load(Ordering::Acquire);
            if chunk_raw.is_null() {
                return core::ptr::null_mut();
            }
            // SAFETY: chunk_raw non-null and stable.
            unsafe {
                (*chunk_raw).slots[lane_index % CHUNK_SIZE]
                    .swap(core::ptr::null_mut(), Ordering::AcqRel)
            }
        }
    }

    impl<T> Drop for Registry<T> {
        fn drop(&mut self) {
            for slot in &self.chunks {
                let raw = slot.load(Ordering::Relaxed);
                if !raw.is_null() {
                    // SAFETY: allocated via Box::into_raw in ensure_chunk.
                    drop(unsafe { Box::from_raw(raw) });
                }
            }
        }
    }

    // ---- Inner ----

    struct Inner<T> {
        registry: Box<Registry<T>>,
        /// high-water mark of allocated lanes. Consumer scans 0..used_lanes.
        used_lanes: AtomicUsize,
        ceiling: usize,
        /// eager floor: lanes < floor are never reclaimed (always retained),
        /// regardless of release policy.
        floor: usize,
        lane_capacity: usize,
        release: ReleasePolicy,
        free_lanes: SegQueue<usize>,
        /// ReleasePolicy::Always: above-floor lanes whose producer has dropped.
        /// the consumer frees each once it is drained (empty), then recycles the
        /// index. floor lanes + Never go straight to `free_lanes` (ring kept).
        abandoned: SegQueue<usize>,
        producer_count: AtomicUsize,
        consumer_alive: AtomicBool,
        waker: AtomicWaker,
        consumer_parked: AtomicBool,
        drain_cursor: AtomicUsize,
        /// bumped whenever a NEW lane ring is allocated (grow). the single
        /// consumer caches resolved lane pointers and rebuilds only when this
        /// changes — turning the per-recv chunk->lane registry chase into a
        /// direct array index. recycled lanes reuse a stable ring pointer and
        /// do not bump. (a future release policy that frees rings bumps here too
        /// so the cache invalidates.)
        registry_gen: AtomicUsize,
    }

    impl<T> Inner<T> {
        fn new(config: &InboxDynamicConfig) -> Arc<Self> {
            let ceiling = config.effective_ceiling();
            // floor must be ≥ 1 (lane 0 always allocated for initial producer)
            // and ≤ ceiling.
            let floor = config.floor.min(ceiling).max(1);
            let registry = Registry::<T>::new();
            // allocate all floor lane rings eagerly. the initial producer
            // holds lane 0; indices 1..floor are pre-warmed so the first
            // (floor-1) claim_lane calls find an existing ring and skip the
            // lazy-alloc path.
            for index in 0..floor {
                registry.publish_lane(index, Box::new(Lane::new(config.lane_capacity)));
            }
            // used_lanes starts at 1: lane 0 is claimed by the initial producer.
            // indices 1..floor have pre-allocated rings; claim_lane will find them
            // via lane_ptr_at() and will not allocate. the consumer scans
            // 0..used_lanes (high-water), so it only drains lanes that have been
            // explicitly claimed.
            Arc::new(Self {
                registry,
                used_lanes: AtomicUsize::new(1),
                ceiling,
                floor,
                lane_capacity: config.lane_capacity,
                release: config.release,
                free_lanes: SegQueue::new(),
                abandoned: SegQueue::new(),
                producer_count: AtomicUsize::new(1),
                consumer_alive: AtomicBool::new(true),
                waker: AtomicWaker::new(),
                consumer_parked: AtomicBool::new(false),
                drain_cursor: AtomicUsize::new(0),
                registry_gen: AtomicUsize::new(0),
            })
        }

        /// claim a lane index for a new producer. cold path.
        /// order: recycled → grow (lazy allocate ring beyond floor) → Busy.
        fn claim_lane(&self) -> Result<usize, ClaimError> {
            if let Some(recycled) = self.free_lanes.pop() {
                // ReleasePolicy::Always frees a reclaimed lane's ring (reclaim_
                // abandoned nulls the slot) before recycling its index, so a
                // recycled lane may be null here. re-publish a fresh ring before
                // handing it back — else the producer derefs a null lane. (Never/
                // floor recycling keeps the ring, so the slot stays non-null and
                // we reuse it untouched.)
                if self.registry.lane_ptr_at(recycled).is_null() {
                    self.registry
                        .publish_lane(recycled, Box::new(Lane::new(self.lane_capacity)));
                    self.registry_gen.fetch_add(1, Ordering::Release);
                }
                return Ok(recycled);
            }
            let index = self.used_lanes.fetch_add(1, Ordering::AcqRel);
            if index < self.ceiling {
                let lane_ptr = self.registry.lane_ptr_at(index);
                if lane_ptr.is_null() {
                    // lazily allocate the ring for this index (beyond floor).
                    self.registry
                        .publish_lane(index, Box::new(Lane::new(self.lane_capacity)));
                    // new ring => invalidate the consumer's cached pointer set.
                    self.registry_gen.fetch_add(1, Ordering::Release);
                }
                Ok(index)
            } else {
                // used_lanes is now saturated above ceiling; we cannot undo
                // the fetch_add without a CAS loop that costs more than it
                // saves. Saturated values above ceiling are harmless: the slot
                // is null so the consumer skips it.
                Err(ClaimError::AtCeiling)
            }
        }

        fn release_lane(&self, lane_index: usize) {
            // Always: hand above-floor lanes to the consumer for reclamation
            // (it frees the ring once drained). floor lanes, and the whole Never
            // policy, recycle the ring in place via free_lanes.
            if matches!(self.release, ReleasePolicy::Always) && lane_index >= self.floor {
                self.abandoned.push(lane_index);
                // nudge the consumer so it reclaims promptly even if idle.
                self.waker.wake();
            } else {
                self.free_lanes.push(lane_index);
            }
        }

        /// ReleasePolicy::Always: free the rings of drained, producer-abandoned,
        /// above-floor lanes so resident memory tracks live producers. called by
        /// the single consumer off the hot path (only on a fast-path miss). a
        /// not-yet-drained lane is requeued and reclaimed on a later pass — so no
        /// in-flight task is ever dropped.
        fn reclaim_abandoned(&self) {
            let mut requeue: Option<usize> = None;
            while let Some(lane_index) = self.abandoned.pop() {
                let lane_ptr = self.registry.lane_ptr_at(lane_index);
                if lane_ptr.is_null() {
                    continue;
                }
                // SAFETY: sole consumer; the producer that owned this lane has
                // dropped, so no concurrent writer remains.
                if unsafe { (*lane_ptr).is_empty() } {
                    let old = self.registry.take_lane(lane_index);
                    if !old.is_null() {
                        // SAFETY: produced by Box::into_raw in publish_lane; the
                        // slot is now null so no other path can reach it.
                        drop(unsafe { Box::from_raw(old) });
                    }
                    self.free_lanes.push(lane_index);
                    // ring freed/slot nulled => invalidate the consumer cache.
                    self.registry_gen.fetch_add(1, Ordering::Release);
                } else {
                    // still draining; retry on a later pass (bounded per call).
                    requeue = Some(lane_index);
                    break;
                }
            }
            if let Some(lane_index) = requeue {
                self.abandoned.push(lane_index);
            }
        }

        #[inline(always)]
        fn try_send_on(&self, lane_ptr: *const Lane<T>, value: T) -> Result<(), SendError<T>> {
            if !self.consumer_alive.load(Ordering::Acquire) {
                return Err(SendError::Disconnected(value));
            }
            // SAFETY: lane_ptr is a stable address published by publish_lane
            // (Release) and loaded with Acquire by the producer at claim time.
            // The Arc<Inner<T>> keeps the Registry alive. The Registry keeps
            // the LaneChunk alive. The LaneChunk keeps the Lane alive.
            // No mutable alias exists: only the owning Producer ever calls
            // try_send on this lane, and only the single Consumer calls try_recv.
            let result = unsafe { (*lane_ptr).try_send(value) };
            match result {
                Ok(()) => {
                    // Dekker-pattern fence — pairs with the fence in Recv::poll.
                    atomic::fence(Ordering::SeqCst);
                    if self.consumer_parked.load(Ordering::Acquire)
                        && self
                            .consumer_parked
                            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
                            .is_ok()
                    {
                        self.waker.wake();
                    }
                    Ok(())
                }
                Err(LaneSendErr::Full(value)) => Err(SendError::Full(value)),
                Err(LaneSendErr::Closed(value)) => Err(SendError::Closed(value)),
            }
        }

        fn close_all(&self) {
            let high_water = self.used_lanes.load(Ordering::Acquire).min(self.ceiling);
            for index in 0..high_water {
                let lane_ptr = self.registry.lane_ptr_at(index);
                if !lane_ptr.is_null() {
                    // SAFETY: stable pointer, see try_send_on.
                    unsafe { (*lane_ptr).close() };
                }
            }
        }
    }

    // ---- TLS lane cache for try_send_mpsc ----

    const HOT_SIZE: usize = 4;

    // (inner_ptr as usize, lane_index, lane_raw as usize) — type-erased so
    // the TLS statics don't monomorphize over T.
    std::thread_local! {
        static HOT_MPSC: Cell<[(usize, usize, usize); HOT_SIZE]> =
            const { Cell::new([(0, 0, 0); HOT_SIZE]) };
        static HOT_MPSC_NEXT: Cell<usize> = const { Cell::new(0) };
    }

    // ---- Producer ----

    pub struct Producer<T> {
        inner: Arc<Inner<T>>,
        /// null = lane not yet claimed (lazy clone path).
        lane_raw: Cell<*const Lane<T>>,
        lane_index: Cell<usize>,
    }

    // SAFETY: Producer: Send — moved to a producer thread once; the Cell<*const
    // Lane<T>> is thread-local by the SPSC contract (only one thread calls
    // try_send at a time). Raw pointer is stable and live via Arc<Inner<T>>.
    //
    // Producer: Sync — needed so Arc<Producer<T>>: Send (bench + MPSC patterns).
    // try_send_mpsc(&self) is safe to call from multiple threads because each
    // thread gets its own lane via the TLS cache; the Cell fields are only
    // accessed by the producer thread that owns this Producer (try_send contract).
    // Callers MUST NOT call try_send from concurrent threads through Arc — the
    // SPSC contract forbids that. try_send_mpsc is the concurrent-safe path.
    unsafe impl<T: Send> Send for Producer<T> {}
    unsafe impl<T: Send> Sync for Producer<T> {}

    impl<T: Send + 'static> Producer<T> {
        /// non-blocking SPSC send. if the lane hasn't been claimed yet (clone
        /// path with null lane_raw), claims it now (cold path, once per producer).
        pub fn try_send(&self, value: T) -> Result<(), SendError<T>> {
            let raw = self.lane_raw.get();
            if !raw.is_null() {
                return self.inner.try_send_on(raw, value);
            }
            self.claim_and_send_spsc(value)
        }

        #[cold]
        #[inline(never)]
        fn claim_and_send_spsc(&self, value: T) -> Result<(), SendError<T>> {
            match self.inner.claim_lane() {
                Ok(index) => {
                    let raw = self.inner.registry.lane_ptr_at(index);
                    debug_assert!(!raw.is_null());
                    self.lane_raw.set(raw);
                    self.lane_index.set(index);
                    self.inner.try_send_on(raw, value)
                }
                Err(ClaimError::AtCeiling) => Err(SendError::Busy(value)),
            }
        }

        pub fn close(&self) {
            self.inner.close_all();
        }

        /// MPSC-safe send. each calling thread gets its own lane via a TLS
        /// 4-slot associative cache (hot path) + slow path on miss.
        pub fn try_send_mpsc(&self, value: T) -> Result<(), SendError<T>> {
            let inner_ptr = Arc::as_ptr(&self.inner) as usize;
            let hot = HOT_MPSC.with(Cell::get);
            for entry in &hot {
                if entry.0 == inner_ptr && inner_ptr != 0 {
                    let raw = entry.2 as *const Lane<T>;
                    return self.inner.try_send_on(raw, value);
                }
            }
            self.try_send_mpsc_cold(inner_ptr, value)
        }

        #[cold]
        #[inline(never)]
        fn try_send_mpsc_cold(&self, inner_ptr: usize, value: T) -> Result<(), SendError<T>> {
            // Check if this thread already has a lane in the hot cache under
            // a different slot (eviction race). If found, promote it.
            // Otherwise allocate a new lane and register a thread-exit hook.
            let lane_index = match self.inner.claim_lane() {
                Ok(index) => index,
                Err(ClaimError::AtCeiling) => return Err(SendError::Busy(value)),
            };
            let raw = self.inner.registry.lane_ptr_at(lane_index);
            debug_assert!(!raw.is_null());

            // register the release hook for thread exit.
            let inner_arc = self.inner.clone();
            let release_fn: Box<dyn FnOnce() + Send + 'static> =
                Box::new(move || inner_arc.release_lane(lane_index));
            RELEASE_HOOKS.with(|cell| cell.borrow_mut().push(release_fn));

            // promote into the hot cache.
            HOT_MPSC.with(|cell| {
                let mut slots = cell.get();
                let next = HOT_MPSC_NEXT.with(Cell::get);
                slots[next] = (inner_ptr, lane_index, raw as usize);
                cell.set(slots);
                HOT_MPSC_NEXT.with(|cursor| {
                    cursor.set((next + 1) % HOT_SIZE);
                });
            });

            self.inner.try_send_on(raw, value)
        }
    }

    // thread-exit lane-release hooks. the vec's Drop impl runs on thread exit
    // and calls every registered closure, returning lanes to free_lanes pools.
    struct ReleaseHooks(Vec<Box<dyn FnOnce() + Send + 'static>>);

    impl ReleaseHooks {
        fn push(&mut self, hook: Box<dyn FnOnce() + Send + 'static>) {
            self.0.push(hook);
        }
    }

    impl Drop for ReleaseHooks {
        fn drop(&mut self) {
            for hook in self.0.drain(..) {
                hook();
            }
        }
    }

    std::thread_local! {
        static RELEASE_HOOKS: std::cell::RefCell<ReleaseHooks> =
            std::cell::RefCell::new(ReleaseHooks(Vec::new()));
    }

    impl<T: Send + 'static> Clone for Producer<T> {
        fn clone(&self) -> Self {
            // clone bumps producer_count and shares Inner. lane allocation
            // is deferred to first send — so clone NEVER exhausts/panics.
            self.inner.producer_count.fetch_add(1, Ordering::AcqRel);
            Self {
                inner: self.inner.clone(),
                lane_raw: Cell::new(core::ptr::null()),
                lane_index: Cell::new(0),
            }
        }
    }

    impl<T> Drop for Producer<T> {
        fn drop(&mut self) {
            // only return the lane if it was actually claimed.
            if !self.lane_raw.get().is_null() {
                self.inner.release_lane(self.lane_index.get());
            }
            if self.inner.producer_count.fetch_sub(1, Ordering::AcqRel) == 1 {
                self.inner.consumer_parked.store(false, Ordering::Release);
                self.inner.waker.wake();
            }
        }
    }

    // ---- Consumer ----

    /// single-consumer-private cache of the resolved STICKY lane pointer. the
    /// single-producer 80% case drains the same lane every call, so caching one
    /// pointer (keyed on registry generation + lane index) collapses the per-recv
    /// chunk->lane registry chase to a cached deref — no Vec, no bounds check,
    /// no grow/reserve. multi-producer scan falls back to the registry.
    struct ConsumerCache<T> {
        generation: usize,
        index: usize,
        ptr: *const Lane<T>,
    }

    pub struct Consumer<T> {
        inner: Arc<Inner<T>>,
        cache: UnsafeCell<ConsumerCache<T>>,
        _not_sync: PhantomData<core::cell::Cell<()>>,
    }

    // SAFETY: the cached raw pointers reference Lane<T> rings owned by the shared
    // Arc<Inner> (alive for the consumer's lifetime; stable under release=Never).
    // Consumer is !Sync (UnsafeCell + PhantomData<Cell>), so only one thread ever
    // touches the cache; Send just transfers that sole ownership once at startup.
    unsafe impl<T: Send> Send for Consumer<T> {}

    impl<T> Consumer<T> {
        pub fn try_recv(&self) -> Result<T, TryRecvError> {
            let inner = &*self.inner;
            let generation = inner.registry_gen.load(Ordering::Acquire);
            let cursor = inner.drain_cursor.load(Ordering::Relaxed);
            // SAFETY: Consumer is !Sync — the single drainer is the sole accessor.
            let cache = unsafe { &mut *self.cache.get() };
            // sticky fast path: same generation + same lane => cached pointer,
            // no registry chase. (the single-producer 80% case.)
            let start_ptr = if cache.generation == generation && cache.index == cursor {
                cache.ptr
            } else {
                let resolved = inner.registry.lane_ptr_at(cursor);
                cache.generation = generation;
                cache.index = cursor;
                cache.ptr = resolved;
                resolved
            };
            if !start_ptr.is_null() {
                // SAFETY: pointer published by publish_lane, observed via the
                // registry_gen Acquire; stable for the inbox lifetime.
                if let Some(value) = unsafe { (*start_ptr).try_recv() } {
                    return Ok(value);
                }
            }
            // slow path: scan the other lanes (multi-producer / sticky drained).
            // eager-release runs here, off the single-producer hot path: a
            // fast-path hit returns above without ever touching reclamation.
            if matches!(inner.release, ReleasePolicy::Always) {
                inner.reclaim_abandoned();
            }
            let high_water = inner.used_lanes.load(Ordering::Acquire).min(inner.ceiling);
            if high_water == 0 {
                return Err(TryRecvError::Empty);
            }
            let start = if cursor < high_water { cursor } else { 0 };
            for offset in 1..high_water {
                let raw = start + offset;
                let index = if raw < high_water {
                    raw
                } else {
                    raw - high_water
                };
                let lane_ptr = inner.registry.lane_ptr_at(index);
                if lane_ptr.is_null() {
                    continue;
                }
                // SAFETY: as above.
                if let Some(value) = unsafe { (*lane_ptr).try_recv() } {
                    inner.drain_cursor.store(index, Ordering::Relaxed);
                    return Ok(value);
                }
            }
            if inner.producer_count.load(Ordering::Acquire) == 0 {
                Err(TryRecvError::Disconnected)
            } else {
                Err(TryRecvError::Empty)
            }
        }

        pub fn recv(&self) -> Recv<'_, T> {
            Recv { consumer: self }
        }

        /// diagnostics only: (used_lanes high-water, abandoned-pending count).
        #[doc(hidden)]
        #[must_use]
        pub fn debug_lane_stats(&self) -> (usize, usize) {
            (
                self.inner.used_lanes.load(Ordering::Acquire),
                self.inner.abandoned.len(),
            )
        }
    }

    impl<T> Drop for Consumer<T> {
        fn drop(&mut self) {
            self.inner.consumer_alive.store(false, Ordering::Release);
        }
    }

    /// async recv future. lives on caller's stack — no Box::pin.
    pub struct Recv<'consumer, T> {
        consumer: &'consumer Consumer<T>,
    }

    impl<T> Future for Recv<'_, T> {
        type Output = Result<T, RecvError>;

        fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
            match self.consumer.try_recv() {
                Ok(value) => return Poll::Ready(Ok(value)),
                Err(TryRecvError::Disconnected) => return Poll::Ready(Err(RecvError)),
                Err(TryRecvError::Empty) => {}
            }
            self.consumer.inner.waker.register(context.waker());
            self.consumer
                .inner
                .consumer_parked
                .store(true, Ordering::Release);
            // Dekker-pattern fence — pairs with the SeqCst fence in try_send_on.
            atomic::fence(Ordering::SeqCst);
            match self.consumer.try_recv() {
                Ok(value) => {
                    self.consumer
                        .inner
                        .consumer_parked
                        .store(false, Ordering::Release);
                    Poll::Ready(Ok(value))
                }
                Err(TryRecvError::Disconnected) => {
                    self.consumer
                        .inner
                        .consumer_parked
                        .store(false, Ordering::Release);
                    Poll::Ready(Err(RecvError))
                }
                Err(TryRecvError::Empty) => Poll::Pending,
            }
        }
    }

    // ---- channel constructor ----

    /// create a dynamic lane-pool inbox. `floor` lanes are allocated eagerly;
    /// additional lanes grow on first claim up to `ceiling`. The initial
    /// `Producer` holds lane 0 (always within the floor). Further producers
    /// come from `Producer::clone` — which never panics or blocks.
    #[must_use]
    pub fn channel<T: Send + 'static>(config: &InboxDynamicConfig) -> (Producer<T>, Consumer<T>) {
        let inner = Inner::new(config);
        let lane_raw = inner.registry.lane_ptr_at(0);
        debug_assert!(!lane_raw.is_null(), "floor must include lane 0");
        let producer = Producer {
            inner: inner.clone(),
            lane_raw: Cell::new(lane_raw),
            lane_index: Cell::new(0),
        };
        let consumer = Consumer {
            inner,
            // generation usize::MAX forces a resolve on the first try_recv.
            cache: UnsafeCell::new(ConsumerCache {
                generation: usize::MAX,
                index: 0,
                ptr: core::ptr::null(),
            }),
            _not_sync: PhantomData,
        };
        (producer, consumer)
    }

    // ---- tests (gate: ≥6, happy/sad/edge/DROP/CONCURRENCY/AT-CEILING/recycle/static-mode) ----

    #[cfg(test)]
    #[allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )]
    mod tests {
        use alloc::sync::Arc;
        use core::sync::atomic::{AtomicUsize as CoreAtomicUsize, Ordering as CoreOrdering};
        use std::thread;

        use super::*;

        fn cfg(floor: usize, ceiling: usize) -> InboxDynamicConfig {
            InboxDynamicConfig {
                floor,
                ceiling,
                release: ReleasePolicy::Never,
                lane_capacity: 64,
            }
        }

        // gate point 12: config struct literal and builder produce equal state.
        #[test]
        fn config_and_builder_produce_equal_state() {
            let via_struct = InboxDynamicConfig {
                floor: 4,
                ceiling: 128,
                release: ReleasePolicy::Never,
                lane_capacity: 256,
            };
            let via_builder = InboxDynamicConfig::builder()
                .floor(4)
                .ceiling(128)
                .release(ReleasePolicy::Never)
                .lane_capacity(256)
                .build();
            assert_eq!(via_struct, via_builder);
        }

        // happy: basic roundtrip
        #[test]
        fn spsc_roundtrip_preserves_value() {
            let (producer, consumer) = channel::<u64>(&cfg(2, 16));
            producer.try_send(42).expect("send");
            assert_eq!(consumer.try_recv().expect("recv"), 42);
        }

        // happy: empty recv
        #[test]
        fn try_recv_on_empty_returns_empty() {
            let (_producer, consumer) = channel::<u64>(&cfg(2, 16));
            assert_eq!(consumer.try_recv(), Err(TryRecvError::Empty));
        }

        // sad: full ring returns payload unharmed
        #[test]
        fn try_send_on_full_returns_payload_unharmed() {
            let config = InboxDynamicConfig {
                floor: 1,
                ceiling: 8,
                release: ReleasePolicy::Never,
                lane_capacity: 4,
            };
            let (producer, _consumer) = channel::<u64>(&config);
            for index in 0..4 {
                producer.try_send(index).expect("fill");
            }
            match producer.try_send(99) {
                Err(SendError::Full(payload)) => assert_eq!(payload, 99),
                other => panic!("expected Full(99), got {other:?}"),
            }
        }

        // edge: floor==ceiling = static mode, zero grow after init
        #[test]
        fn floor_eq_ceiling_static_mode_zero_dynamic_alloc() {
            let config = cfg(4, 4);
            let (producer, consumer) = channel::<u64>(&config);
            let used_before = producer.inner.used_lanes.load(CoreOrdering::Acquire);
            let c1 = producer.clone();
            let c2 = producer.clone();
            let c3 = producer.clone();
            producer.try_send(1).expect("p");
            c1.try_send(2).expect("c1");
            c2.try_send(3).expect("c2");
            c3.try_send(4).expect("c3");
            let mut received: Vec<u64> =
                (0..4).map(|_| consumer.try_recv().expect("recv")).collect();
            received.sort_unstable();
            assert_eq!(received, [1, 2, 3, 4]);
            // used_lanes must not have grown beyond floor (static mode).
            let used_after = producer.inner.used_lanes.load(CoreOrdering::Acquire);
            assert!(
                used_after <= used_before + 3,
                "used_lanes={used_after} grew unexpectedly for clones within floor"
            );
        }

        // sad: at-ceiling returns Busy, not drop, not panic
        #[test]
        fn at_ceiling_returns_busy_not_drop_or_panic() {
            let (producer, _consumer) = channel::<u64>(&cfg(1, 1));
            // lane 0 already held by initial producer.
            let clone = producer.clone();
            match clone.try_send(99) {
                Err(SendError::Busy(value)) => assert_eq!(value, 99),
                other => panic!("expected Busy(99), got {other:?}"),
            }
        }

        // edge: clone never panics even far above ceiling
        #[test]
        fn clone_never_panics_even_above_ceiling() {
            let (producer, _consumer) = channel::<u64>(&cfg(1, 1));
            let c1 = producer.clone();
            let c2 = producer.clone();
            let c3 = producer.clone();
            for clone in [c1, c2, c3] {
                match clone.try_send(0) {
                    Err(SendError::Busy(_)) => {}
                    other => panic!("expected Busy, got {other:?}"),
                }
            }
        }

        // CONCURRENCY: producers > floor proves grow + no panic + no loss
        #[test]
        fn grow_beyond_floor_no_loss() {
            const THREADS: usize = 8;
            const PER_THREAD: usize = 1000;
            let config = InboxDynamicConfig {
                floor: 2,
                ceiling: 32,
                release: ReleasePolicy::Never,
                lane_capacity: 256,
            };
            let (producer, consumer) = channel::<u64>(&config);
            let producer = Arc::new(producer);
            let mut handles = Vec::with_capacity(THREADS);
            for thread_id in 0..THREADS {
                let prod = producer.clone();
                handles.push(thread::spawn(move || {
                    for index in 0..PER_THREAD {
                        let value = (thread_id * PER_THREAD + index) as u64;
                        loop {
                            match prod.try_send_mpsc(value) {
                                Ok(()) => break,
                                Err(SendError::Full(_)) | Err(SendError::Busy(_)) => {
                                    thread::yield_now();
                                }
                                Err(other) => panic!("unexpected: {other}"),
                            }
                        }
                    }
                }));
            }
            let total = THREADS * PER_THREAD;
            let mut received = 0usize;
            while received < total {
                match consumer.try_recv() {
                    Ok(_) => received += 1,
                    Err(TryRecvError::Empty) => thread::yield_now(),
                    Err(TryRecvError::Disconnected) => break,
                }
            }
            for handle in handles {
                handle.join().expect("join");
            }
            assert_eq!(received, total, "no item lost across grow");
        }

        // REGRESSION: under ReleasePolicy::Always the consumer frees a drained
        // above-floor lane's ring and recycles its index; a later claim of that
        // index must re-publish a fresh ring, not hand back the freed (null) lane
        // — else the producer derefs null (debug_assert / segfault). churning
        // concurrent producer bursts forces reclaim-then-reclaim of recycled
        // indices, which panicked before the fix.
        #[test]
        fn recycled_lane_after_always_reclaim_is_republished() {
            const THREADS: usize = 8;
            const PER_THREAD: usize = 400;
            const ROUNDS: usize = 6;
            let config = InboxDynamicConfig {
                floor: 1,
                ceiling: 16,
                release: ReleasePolicy::Always,
                lane_capacity: 64,
            };
            let (producer, consumer) = channel::<u64>(&config);
            let producer = Arc::new(producer);
            let mut received = 0usize;
            for round in 0..ROUNDS {
                let mut handles = Vec::with_capacity(THREADS);
                for thread_id in 0..THREADS {
                    let prod = producer.clone();
                    handles.push(thread::spawn(move || {
                        for index in 0..PER_THREAD {
                            let value = (round * THREADS * PER_THREAD
                                + thread_id * PER_THREAD
                                + index) as u64;
                            loop {
                                match prod.try_send_mpsc(value) {
                                    Ok(()) => break,
                                    Err(SendError::Full(_)) | Err(SendError::Busy(_)) => {
                                        thread::yield_now()
                                    }
                                    Err(other) => panic!("unexpected: {other}"),
                                }
                            }
                        }
                    }));
                }
                // drain while the round's producers run AND after they exit, so the
                // consumer reclaims abandoned above-floor lanes and recycles them
                // into the next round's claims.
                let target = (round + 1) * THREADS * PER_THREAD;
                while received < target {
                    match consumer.try_recv() {
                        Ok(_) => received += 1,
                        Err(TryRecvError::Empty) => thread::yield_now(),
                        Err(TryRecvError::Disconnected) => break,
                    }
                }
                for handle in handles {
                    handle.join().expect("join");
                }
            }
            assert_eq!(
                received,
                THREADS * PER_THREAD * ROUNDS,
                "no item lost across reclaim/recycle churn"
            );
        }

        // DROP: payload drop runs for in-flight values when both sides drop
        #[test]
        fn payload_drop_runs_for_values_left_in_ring() {
            let drops = Arc::new(CoreAtomicUsize::new(0));
            #[derive(Debug)]
            struct Counted(Arc<CoreAtomicUsize>);
            impl Drop for Counted {
                fn drop(&mut self) {
                    self.0.fetch_add(1, CoreOrdering::AcqRel);
                }
            }
            let config = InboxDynamicConfig {
                floor: 1,
                ceiling: 4,
                release: ReleasePolicy::Never,
                lane_capacity: 16,
            };
            let (producer, consumer) = channel::<Counted>(&config);
            producer.try_send(Counted(drops.clone())).expect("send 1");
            producer.try_send(Counted(drops.clone())).expect("send 2");
            drop(producer);
            drop(consumer);
            assert_eq!(
                drops.load(CoreOrdering::Acquire),
                2,
                "drop must run for both items"
            );
        }

        // recycle-after-thread-exit: lane reused, no monotonic growth
        #[test]
        fn lane_recycled_after_thread_exits() {
            let config = cfg(1, 3);
            let (producer, consumer) = channel::<u64>(&config);
            let prod = Arc::new(producer);
            for _ in 0..8 {
                let prod_clone = prod.clone();
                thread::spawn(move || {
                    prod_clone.try_send_mpsc(1).expect("send");
                })
                .join()
                .expect("join");
                while consumer.try_recv().is_ok() {}
            }
            // used_lanes must not have grown monotonically past ceiling.
            let used = prod.inner.used_lanes.load(CoreOrdering::Acquire);
            assert!(
                used <= 4,
                "used_lanes={used} grew monotonically; recycle not working"
            );
        }

        // ReleasePolicy::Always frees a drained, producer-abandoned, above-floor
        // lane ring (memory tracks live producers) without losing the in-flight
        // value.
        #[test]
        fn always_frees_drained_abandoned_lane() {
            let config = InboxDynamicConfig {
                floor: 1,
                ceiling: 16,
                release: ReleasePolicy::Always,
                lane_capacity: 4,
            };
            let (producer, consumer) = channel::<u64>(&config);
            let prod = Arc::new(producer);
            // a second producer on another thread claims an above-floor lane
            // (index 1), sends, then exits -> lane abandoned.
            let prod_clone = prod.clone();
            thread::spawn(move || {
                prod_clone.try_send_mpsc(99).expect("send");
            })
            .join()
            .expect("join");
            assert!(
                !prod.inner.registry.lane_ptr_at(1).is_null(),
                "above-floor lane should be allocated before reclaim"
            );
            // drain the value (no loss), then a follow-up recv hits the slow path
            // and reclaims the now-empty abandoned lane.
            assert_eq!(consumer.try_recv().expect("recv"), 99);
            let _ = consumer.try_recv();
            assert!(
                prod.inner.registry.lane_ptr_at(1).is_null(),
                "Always must free the drained, abandoned above-floor lane ring"
            );
        }

        // release policy parses all variants and rejects garbage
        #[test]
        fn release_policy_parses_all_variants() {
            assert_eq!(
                ReleasePolicy::from_str("none").expect("none"),
                ReleasePolicy::Never
            );
            assert_eq!(
                ReleasePolicy::from_str("hold").expect("hold"),
                ReleasePolicy::Never
            );
            assert_eq!(
                ReleasePolicy::from_str("never").expect("never"),
                ReleasePolicy::Never
            );
            assert_eq!(
                ReleasePolicy::from_str("always").expect("always"),
                ReleasePolicy::Always
            );
            assert!(ReleasePolicy::from_str("garbage").is_err());
        }

        // last producer drop disconnects consumer
        #[test]
        fn last_producer_drop_yields_disconnected() {
            let (producer, consumer) = channel::<u64>(&cfg(2, 16));
            let clone = producer.clone();
            drop(producer);
            drop(clone);
            assert_eq!(consumer.try_recv(), Err(TryRecvError::Disconnected));
        }
    }
}

// ---- inbox-const: stack-backed SPSC ring ----
//
// `runtime-prime-inbox-const` provides a fully no_std, no-alloc SPSC channel
// backed by const-generic stack storage. The algorithm mirrors `Lane<T>` from
// inbox-alloc: same CLOSED_BIT encoding, same CAS-publish, same Dekker-fence
// pair. The only structural difference is storage: `[UnsafeCell<MaybeUninit<T>>; CAP]`
// on the stack instead of `Box<[...]>` on the heap.
//
// Public surface:
//   `Inbox::<T, CAP>::new() -> Inbox<T, CAP>`  — create on stack
//   `Inbox::<T, CAP>::split(&mut self) -> (Producer<'_, T, CAP>, Consumer<'_, T, CAP>)`
//   `Producer::try_send(T) -> Result<(), SendError<T>>`
//   `Producer::close()`
//   `Consumer::try_recv() -> Result<T, TryRecvError>`
//   `Consumer::recv() -> Recv<'_, T, CAP>`  — async, no Box::pin
//
// CAP must be a non-zero power of two (asserted at runtime in new()).
// No Box, Arc, alloc::* anywhere in this path.
#[cfg(feature = "runtime-prime-inbox-const")]
pub mod inbox_const {
    use core::cell::UnsafeCell;
    use core::future::Future;
    use core::marker::PhantomData;
    use core::mem::MaybeUninit;
    use core::pin::Pin;
    use core::sync::atomic::{self, AtomicBool, AtomicUsize, Ordering};
    use core::task::{Context, Poll};

    use atomic_waker::AtomicWaker;
    use crossbeam_utils::CachePadded;

    use super::{CLOSED_BIT, POSITION_MASK, RecvError, SendError, TryRecvError};

    /// stack-backed SPSC inbox. place on the stack (or in a static) and
    /// call `split()` to obtain the `Producer` + `Consumer` halves.
    /// CAP must be a non-zero power of two.
    pub struct Inbox<T, const CAP: usize> {
        head: CachePadded<AtomicUsize>,
        tail: CachePadded<AtomicUsize>,
        cached_head: CachePadded<UnsafeCell<usize>>,
        cached_tail: CachePadded<UnsafeCell<usize>>,
        slots: [UnsafeCell<MaybeUninit<T>>; CAP],
        producer_alive: AtomicBool,
        consumer_alive: AtomicBool,
        waker: AtomicWaker,
        consumer_parked: AtomicBool,
    }

    // SAFETY: the SPSC contract is upheld by Producer/Consumer lifetimes:
    // only one Producer (single writer) and one Consumer (single reader)
    // are created via split(); they borrow &Inbox for their lifetime.
    // No two threads touch the same slot concurrently.
    unsafe impl<T: Send, const CAP: usize> Sync for Inbox<T, CAP> {}
    unsafe impl<T: Send, const CAP: usize> Send for Inbox<T, CAP> {}

    impl<T, const CAP: usize> Inbox<T, CAP> {
        /// create an empty inbox on the stack. asserts that CAP is a
        /// non-zero power of two (same invariant as Lane::new in inbox-alloc).
        pub const fn new() -> Self {
            assert!(
                CAP > 0 && CAP.is_power_of_two(),
                "inbox_const CAP must be a non-zero power of two",
            );
            // SAFETY: MaybeUninit<T> is validly uninit; UnsafeCell wrapping it
            // is sound; the const array init is the canonical pattern for this.
            let slots = unsafe {
                let mut array: [UnsafeCell<MaybeUninit<T>>; CAP] =
                    MaybeUninit::uninit().assume_init();
                let mut index = 0;
                while index < CAP {
                    array[index] = UnsafeCell::new(MaybeUninit::uninit());
                    index += 1;
                }
                array
            };
            Self {
                head: CachePadded::new(AtomicUsize::new(0)),
                tail: CachePadded::new(AtomicUsize::new(0)),
                cached_head: CachePadded::new(UnsafeCell::new(0)),
                cached_tail: CachePadded::new(UnsafeCell::new(0)),
                slots,
                producer_alive: AtomicBool::new(true),
                consumer_alive: AtomicBool::new(true),
                waker: AtomicWaker::new(),
                consumer_parked: AtomicBool::new(false),
            }
        }

        /// split the inbox into producer + consumer halves. each half
        /// holds a pinned reference back to this Inbox. may be called
        /// only once; subsequent calls would create a second Producer
        /// which violates the SPSC contract — prevent by calling split
        /// immediately after construction and not exposing the Inbox
        /// directly.
        pub fn split(&mut self) -> (Producer<'_, T, CAP>, Consumer<'_, T, CAP>) {
            let inbox = &*self;
            let producer = Producer {
                inbox,
                _not_send: PhantomData,
            };
            let consumer = Consumer {
                inbox,
                _not_sync: PhantomData,
            };
            (producer, consumer)
        }

        #[inline(always)]
        fn try_send_inner(&self, value: T) -> Result<(), SendError<T>> {
            if !self.consumer_alive.load(Ordering::Acquire) {
                return Err(SendError::Disconnected(value));
            }
            let raw_tail = self.tail.load(Ordering::Relaxed);
            // SAFETY: single producer reads/writes cached_head.
            let cached_head = unsafe { *self.cached_head.get() };
            if raw_tail.wrapping_sub(cached_head) >= CAP {
                let head = self.head.load(Ordering::Acquire);
                // SAFETY: single producer
                unsafe { *self.cached_head.get() = head };
                if raw_tail.wrapping_sub(head) >= CAP {
                    if raw_tail & CLOSED_BIT != 0 {
                        return Err(SendError::Closed(value));
                    }
                    return Err(SendError::Full(value));
                }
            }
            let position = raw_tail & POSITION_MASK;
            // SAFETY: single producer; slot not yet visible to consumer
            // (CAS publish hasn't run). index masked into [0, CAP).
            unsafe {
                (*self.slots[position & (CAP - 1)].get()).write(value);
            }
            // CAS-publish: preserves any concurrently-set CLOSED_BIT.
            let mut current = raw_tail;
            loop {
                let next_position = (current & POSITION_MASK).wrapping_add(1) & POSITION_MASK;
                let next = next_position | (current & CLOSED_BIT);
                match self.tail.compare_exchange_weak(
                    current,
                    next,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(actual) => current = actual,
                }
            }
            // Dekker-pattern fence — pairs with the fence in Recv::poll.
            atomic::fence(Ordering::SeqCst);
            if self.consumer_parked.load(Ordering::Acquire)
                && self
                    .consumer_parked
                    .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                self.waker.wake();
            }
            Ok(())
        }

        #[inline(always)]
        fn try_recv_inner(&self) -> Result<T, TryRecvError> {
            let head = self.head.load(Ordering::Relaxed);
            // SAFETY: single consumer reads/writes cached_tail.
            let cached_tail = unsafe { *self.cached_tail.get() };
            if head == cached_tail {
                let raw_tail = self.tail.load(Ordering::Acquire);
                let tail = raw_tail & POSITION_MASK;
                // SAFETY: single consumer
                unsafe { *self.cached_tail.get() = tail };
                if head == tail {
                    if !self.producer_alive.load(Ordering::Acquire) {
                        return Err(TryRecvError::Disconnected);
                    }
                    return Err(TryRecvError::Empty);
                }
            }
            // SAFETY: cached_tail > head → producer published slot.
            // single consumer; no other reader.
            let value = unsafe { (*self.slots[head & (CAP - 1)].get()).assume_init_read() };
            self.head.store(head.wrapping_add(1), Ordering::Release);
            Ok(value)
        }

        fn close_inner(&self) {
            self.tail.fetch_or(CLOSED_BIT, Ordering::Release);
        }

        fn drop_remaining(&mut self) {
            let head = *self.head.get_mut();
            let tail = *self.tail.get_mut() & POSITION_MASK;
            let mut position = head;
            while position != tail {
                let slot = &mut self.slots[position & (CAP - 1)];
                // SAFETY: positions in [head, tail) hold initialized payloads.
                unsafe {
                    slot.get_mut().assume_init_drop();
                }
                position = position.wrapping_add(1);
            }
        }
    }

    impl<T, const CAP: usize> Drop for Inbox<T, CAP> {
        fn drop(&mut self) {
            self.drop_remaining();
        }
    }

    impl<T, const CAP: usize> Default for Inbox<T, CAP> {
        fn default() -> Self {
            Self::new()
        }
    }

    /// producer half of the SPSC inbox. `!Sync` — only one thread sends.
    /// `Send` — may be moved to a worker thread once.
    /// the `'inbox` lifetime guarantees the Inbox outlives all handles.
    pub struct Producer<'inbox, T, const CAP: usize> {
        inbox: &'inbox Inbox<T, CAP>,
        // explicitly not Sync: only one thread may hold Producer at a time.
        _not_send: PhantomData<*mut ()>,
    }

    // SAFETY: Producer is effectively `Send` (single writer). The *mut ()
    // PhantomData makes it !Sync (not Sync), but Send is safe because we
    // never share Producer across threads — the caller moves it into the
    // producing thread once.
    unsafe impl<T: Send, const CAP: usize> Send for Producer<'_, T, CAP> {}

    impl<T, const CAP: usize> Producer<'_, T, CAP> {
        /// non-blocking SPSC send. returns `Err(SendError::Full(value))` when
        /// the ring is full; `Err(SendError::Closed(value))` after `close()`;
        /// `Err(SendError::Disconnected(value))` when consumer is dropped.
        pub fn try_send(&self, value: T) -> Result<(), SendError<T>> {
            self.inbox.try_send_inner(value)
        }

        /// quiesce: set CLOSED_BIT on tail so future sends return Closed.
        /// consumer continues draining in-flight items. idempotent.
        pub fn close(&self) {
            self.inbox.close_inner();
        }
    }

    impl<T, const CAP: usize> Drop for Producer<'_, T, CAP> {
        fn drop(&mut self) {
            self.inbox.producer_alive.store(false, Ordering::Release);
            // always wake: consumer needs to observe Disconnected.
            self.inbox.consumer_parked.store(false, Ordering::Release);
            self.inbox.waker.wake();
        }
    }

    /// consumer half. `!Sync` — only one thread polls at a time.
    /// `Send` — may be moved to a worker thread once at construction.
    pub struct Consumer<'inbox, T, const CAP: usize> {
        inbox: &'inbox Inbox<T, CAP>,
        _not_sync: PhantomData<core::cell::Cell<()>>,
    }

    // SAFETY: Consumer references &Inbox<T, CAP> which is Sync when T: Send.
    // The !Sync from Cell<()> is intentional; Send is safe because we move
    // the consumer to exactly one thread.
    unsafe impl<T: Send, const CAP: usize> Send for Consumer<'_, T, CAP> {}

    impl<T, const CAP: usize> Consumer<'_, T, CAP> {
        /// non-blocking drain. returns `Empty` or `Disconnected`.
        pub fn try_recv(&self) -> Result<T, TryRecvError> {
            self.inbox.try_recv_inner()
        }

        /// async recv future. no Box::pin; lives on caller's stack.
        pub fn recv(&self) -> Recv<'_, T, CAP> {
            Recv { consumer: self }
        }
    }

    impl<T, const CAP: usize> Drop for Consumer<'_, T, CAP> {
        fn drop(&mut self) {
            self.inbox.consumer_alive.store(false, Ordering::Release);
        }
    }

    /// state-machine future for `Consumer::recv`. polls on the caller's stack;
    /// no `Box::pin`. identical Dekker-fence pair as inbox-alloc's `Recv`.
    pub struct Recv<'consumer, T, const CAP: usize> {
        consumer: &'consumer Consumer<'consumer, T, CAP>,
    }

    impl<T, const CAP: usize> Future for Recv<'_, T, CAP> {
        type Output = Result<T, RecvError>;

        fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
            match self.consumer.try_recv() {
                Ok(value) => return Poll::Ready(Ok(value)),
                Err(TryRecvError::Disconnected) => return Poll::Ready(Err(RecvError)),
                Err(TryRecvError::Empty) => {}
            }
            self.consumer.inbox.waker.register(context.waker());
            self.consumer
                .inbox
                .consumer_parked
                .store(true, Ordering::Release);
            // Dekker-pattern fence — pairs with the SeqCst fence in try_send_inner.
            atomic::fence(Ordering::SeqCst);
            match self.consumer.try_recv() {
                Ok(value) => {
                    self.consumer
                        .inbox
                        .consumer_parked
                        .store(false, Ordering::Release);
                    Poll::Ready(Ok(value))
                }
                Err(TryRecvError::Disconnected) => {
                    self.consumer
                        .inbox
                        .consumer_parked
                        .store(false, Ordering::Release);
                    Poll::Ready(Err(RecvError))
                }
                Err(TryRecvError::Empty) => Poll::Pending,
            }
        }
    }

    #[cfg(test)]
    #[allow(clippy::unwrap_used, clippy::expect_used)]
    mod tests {
        use super::*;
        use core::mem;

        // helper: build an Inbox on the stack, split, return it and halves
        // as a tuple so tests don't have to repeat this pattern.
        // the macro avoids lifetime issues: macros inline the binding.
        macro_rules! make_inbox {
            ($t:ty, $cap:expr, $inbox:ident, $prod:ident, $cons:ident) => {
                let mut $inbox: Inbox<$t, $cap> = Inbox::new();
                let ($prod, $cons) = $inbox.split();
            };
        }

        #[test]
        fn spsc_const_roundtrip_preserves_value() {
            make_inbox!(u64, 4, inbox, producer, consumer);
            producer.try_send(42).expect("send");
            assert_eq!(consumer.try_recv().expect("recv"), 42);
        }

        #[test]
        fn try_recv_on_empty_returns_empty() {
            make_inbox!(u64, 4, inbox, _producer, consumer);
            assert_eq!(consumer.try_recv(), Err(TryRecvError::Empty));
        }

        #[test]
        fn try_send_on_full_returns_payload_unharmed() {
            make_inbox!(u64, 4, inbox, producer, _consumer);
            for index in 0..4 {
                producer.try_send(index).expect("fill");
            }
            match producer.try_send(99) {
                Err(SendError::Full(payload)) => assert_eq!(payload, 99),
                other => panic!("expected Full(99), got {other:?}"),
            }
        }

        #[test]
        fn producer_dropped_returns_disconnected_to_consumer() {
            make_inbox!(u64, 4, inbox, producer, consumer);
            drop(producer);
            assert_eq!(consumer.try_recv(), Err(TryRecvError::Disconnected));
        }

        #[test]
        fn closed_via_quiesce_rejects_new_sends_but_drains_pending() {
            make_inbox!(u64, 16, inbox, producer, consumer);
            for value in 0..5_u64 {
                producer.try_send(value).expect("pre-close send");
            }
            producer.close();
            // new send must be rejected
            match producer.try_send(99) {
                Err(SendError::Closed(value)) => assert_eq!(value, 99),
                other => panic!("expected Closed(99), got {other:?}"),
            }
            // pre-close items drain in order
            for expected in 0..5_u64 {
                assert_eq!(consumer.try_recv().expect("drain"), expected);
            }
            // after drain, Empty (producer still alive via close, not drop)
            assert_eq!(consumer.try_recv(), Err(TryRecvError::Empty));
        }

        #[test]
        fn inbox_lives_on_stack_no_heap_allocations() {
            // CAP=16, T=u64 → size = 16*8 (slots) + 4*CachePadded + 3*AtomicBool + AtomicWaker
            // All stack. This test proves the type is Sized and constructs without panic.
            let size = mem::size_of::<Inbox<u64, 16>>();
            assert!(
                size > 0,
                "Inbox<u64,16> must have non-zero size; got {size}",
            );
            // construct and use — if this compiles and runs, the type lives on the stack.
            make_inbox!(u64, 16, inbox, producer, consumer);
            producer.try_send(1).expect("send");
            assert_eq!(consumer.try_recv().expect("recv"), 1);
        }

        #[test]
        fn closed_bit_does_not_corrupt_position_counter_const() {
            make_inbox!(u64, 16, inbox, producer, consumer);
            for value in 0..5_u64 {
                producer.try_send(value).expect("send");
            }
            producer.close();
            for expected in 0..5_u64 {
                assert_eq!(consumer.try_recv().expect("drain"), expected);
            }
            assert_eq!(consumer.try_recv(), Err(TryRecvError::Empty));
            match producer.try_send(99) {
                Err(SendError::Closed(value)) => assert_eq!(value, 99),
                other => panic!("expected Closed(99), got {other:?}"),
            }
        }

        // the recv future itself never allocates (that's what this proves);
        // building a `Waker` to poll it with does, via `alloc::task::Wake`,
        // and `inbox_const` is deliberately alloc-free in its own path (see
        // the module doc comment), so `alloc` isn't linked without this
        // feature — test-harness-only need, not a production dependency.
        #[cfg(feature = "alloc")]
        #[test]
        fn wake_via_recv_future_no_alloc() {
            use alloc::sync::Arc;
            use alloc::task::Wake;
            use core::sync::atomic::{AtomicUsize as CoreAtomicUsize, Ordering as CoreOrdering};
            use core::task::Waker;

            struct CountingWaker(CoreAtomicUsize);
            impl Wake for CountingWaker {
                fn wake(self: Arc<Self>) {
                    self.0.fetch_add(1, CoreOrdering::AcqRel);
                }
            }

            make_inbox!(u64, 4, inbox, producer, consumer);
            let counter = Arc::new(CountingWaker(CoreAtomicUsize::new(0)));
            let waker: Waker = counter.clone().into();
            let mut context = Context::from_waker(&waker);

            {
                let mut future = consumer.recv();
                let pinned = unsafe { Pin::new_unchecked(&mut future) };
                assert!(pinned.poll(&mut context).is_pending());
            }
            assert_eq!(counter.0.load(CoreOrdering::Acquire), 0);

            producer.try_send(7).expect("send");
            assert!(counter.0.load(CoreOrdering::Acquire) >= 1);

            let mut future = consumer.recv();
            let pinned = unsafe { Pin::new_unchecked(&mut future) };
            match pinned.poll(&mut context) {
                Poll::Ready(Ok(value)) => assert_eq!(value, 7),
                other => panic!("expected Ready(Ok(7)), got {other:?}"),
            }
        }
    }
}
