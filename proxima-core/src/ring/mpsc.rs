#[cfg(feature = "alloc")]
use alloc::boxed::Box;

#[cfg(feature = "alloc")]
use super::CapacityError;
use core::mem::MaybeUninit;

// the ring's atomics + UnsafeCell are cfg-switched to loom under `--features
// loom` so the loom model checker (tests/loom_ring.rs) explores every
// interleaving of the Vyukov push/dequeue protocol on STABLE rust — replacing
// the nightly ThreadSanitizer job. loom's UnsafeCell uses `.with`/`.with_mut`
// closures instead of `.get()`, so the three raw cell accesses go through the
// `*_cell` helpers below.
#[cfg(not(feature = "loom"))]
use core::cell::UnsafeCell;
#[cfg(not(feature = "loom"))]
use core::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "loom")]
use loom::cell::UnsafeCell;
#[cfg(feature = "loom")]
use loom::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "loom")]
unsafe fn write_cell<T>(cell: &UnsafeCell<MaybeUninit<T>>, record: T) {
    cell.with_mut(|ptr| unsafe { (*ptr).write(record) });
}
#[cfg(not(feature = "loom"))]
unsafe fn write_cell<T>(cell: &UnsafeCell<MaybeUninit<T>>, record: T) {
    unsafe { (*cell.get()).write(record) };
}

#[cfg(feature = "loom")]
unsafe fn read_cell<T>(cell: &UnsafeCell<MaybeUninit<T>>) -> T {
    cell.with(|ptr| unsafe { (*ptr).assume_init_read() })
}
#[cfg(not(feature = "loom"))]
unsafe fn read_cell<T>(cell: &UnsafeCell<MaybeUninit<T>>) -> T {
    unsafe { (*cell.get()).assume_init_read() }
}

#[cfg(all(feature = "loom", feature = "alloc"))]
unsafe fn drop_cell<T>(cell: &UnsafeCell<MaybeUninit<T>>) {
    cell.with_mut(|ptr| unsafe { (*ptr).assume_init_drop() });
}
#[cfg(all(not(feature = "loom"), feature = "alloc"))]
unsafe fn drop_cell<T>(cell: &UnsafeCell<MaybeUninit<T>>) {
    unsafe { (*cell.get()).assume_init_drop() };
}

// One cell per slot: a sequence stamp (the handoff protocol) plus the payload.
// `sequence` encodes the slot's lap state — a producer may claim it only when
// `sequence == its claim position`; a consumer frees it by stamping
// `position + capacity` (the next lap's claim position). This is Dmitry
// Vyukov's bounded-MPMC cell, run in its full multi-producer / multi-consumer
// form: both ends linearise their position counter with a CAS.
struct Cell<T> {
    sequence: AtomicUsize,
    data: UnsafeCell<MaybeUninit<T>>,
}

// The lock-free MPMC algorithm, shared by the alloc `Ring` (Box buffer) and the
// no-alloc `StaticRing` (inline array). Operates over a `&[Cell<T>]`; capacity
// and mask derive from the slice length (a power of two >= 2). Single source of
// truth for the Vyukov protocol — both ring types delegate, so there is one
// implementation to audit and one for loom to model-check.
#[inline]
fn cells_push<T>(cells: &[Cell<T>], enqueue_pos: &AtomicUsize, record: T) -> Result<(), T> {
    let mask = cells.len() - 1;
    let mut pos = enqueue_pos.load(Ordering::Relaxed);
    loop {
        let cell = &cells[pos & mask];
        let sequence = cell.sequence.load(Ordering::Acquire);
        let diff = sequence.wrapping_sub(pos) as isize;
        if diff == 0 {
            match enqueue_pos.compare_exchange_weak(
                pos,
                pos.wrapping_add(1),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // SAFETY: CAS winner exclusively owns this slot for this lap;
                    // the consumer freed it (sequence == pos), so no aliasing.
                    unsafe {
                        write_cell(&cell.data, record);
                    }
                    cell.sequence.store(pos.wrapping_add(1), Ordering::Release);
                    return Ok(());
                }
                Err(actual) => pos = actual,
            }
        } else if diff < 0 {
            return Err(record);
        } else {
            pos = enqueue_pos.load(Ordering::Relaxed);
        }
    }
}

#[inline]
fn cells_dequeue<T>(cells: &[Cell<T>], dequeue_pos: &AtomicUsize) -> Option<T> {
    let mask = cells.len() - 1;
    let cap = cells.len();
    let mut pos = dequeue_pos.load(Ordering::Relaxed);
    loop {
        let cell = &cells[pos & mask];
        let sequence = cell.sequence.load(Ordering::Acquire);
        let diff = sequence.wrapping_sub(pos.wrapping_add(1)) as isize;
        if diff == 0 {
            match dequeue_pos.compare_exchange_weak(
                pos,
                pos.wrapping_add(1),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // SAFETY: CAS winner owns this slot; the Acquire observed the
                    // producer's Release publish, so the payload read is valid and
                    // no other consumer touches the cell.
                    let record = unsafe { read_cell(&cell.data) };
                    cell.sequence
                        .store(pos.wrapping_add(cap), Ordering::Release);
                    return Some(record);
                }
                Err(actual) => pos = actual,
            }
        } else if diff < 0 {
            return None;
        } else {
            pos = dequeue_pos.load(Ordering::Relaxed);
        }
    }
}

/// Fixed-capacity bounded ring with **lock-free multi-producer, multi-consumer**
/// access.
///
/// Any number of threads may call [`Ring::push`] concurrently; ownership of
/// each slot is linearised by a CAS on `enqueue_pos`, so two producers never
/// touch the same cell. Likewise any number of threads may [`Ring::dequeue`]
/// (or drain via [`Drainer::drain_into`]) concurrently; each pop is linearised
/// by a CAS on `dequeue_pos`, so two consumers never read the same cell. This
/// lets a background drainer and an overflowing producer (elastic
/// producer-assist) drain the same ring at once. Capacity is rounded up to a
/// power of two (minimum 2) at construction; no resize occurs. Full pushes
/// return the record via `Err` without blocking; empty dequeues return `None`.
#[cfg(feature = "alloc")]
pub struct Ring<T> {
    buf: Box<[Cell<T>]>,
    // monotonically increasing claim/read counters; slot index is `pos & mask`.
    enqueue_pos: AtomicUsize,
    dequeue_pos: AtomicUsize,
    mask: usize,
    cap: usize,
}

// SAFETY: Ring<T> is Send when T is Send — cells are accessed only through the
// Vyukov sequence protocol, which establishes the happens-before edges below.
#[cfg(feature = "alloc")]
unsafe impl<T: Send> Send for Ring<T> {}

// SAFETY: &Ring<T> is Sync when T is Send. Producers synchronise slot ownership
// via the `enqueue_pos` CAS (exactly one winner per position) and publish the
// payload via the cell's `sequence` Release; a consumer's Acquire load of the
// same `sequence` observes the completed write. Consumers synchronise slot
// ownership via the `dequeue_pos` CAS (exactly one winner per position), so two
// consumers never read the same cell. Freeing is the mirror: the consumer's
// Release store is observed by the next producer's Acquire.
#[cfg(feature = "alloc")]
unsafe impl<T: Send> Sync for Ring<T> {}

#[cfg(feature = "alloc")]
impl<T> Ring<T> {
    /// Allocate a ring. `cap` is rounded up to a power of two, minimum 2 (the
    /// Vyukov protocol is degenerate at capacity 1: `pos` and `pos + 1` alias
    /// the lone cell). Returns [`Error::InvalidInput`] only for `cap == 0`.
    pub fn new(cap: usize) -> Result<Self, CapacityError> {
        if cap == 0 {
            return Err(CapacityError);
        }
        Ok(Self::build(cap))
    }

    /// Allocate a ring without the `Result` — `cap` is clamped to at least 1,
    /// then rounded up to a power of two (minimum 2) exactly as [`new`](Self::new).
    /// Use when the capacity comes from a validated config or a `max(1)` clamp, so
    /// `new`'s `cap == 0` arm is unreachable and threading a `Result` through every
    /// caller is noise.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self::build(cap.max(1))
    }

    // shared allocation core; `cap >= 1` is the caller's precondition (`new`
    // guards 0, `with_capacity` clamps) so this cannot fail.
    fn build(cap: usize) -> Self {
        let cap = cap.next_power_of_two().max(2);
        let buf = (0..cap)
            .map(|index| Cell {
                sequence: AtomicUsize::new(index),
                data: UnsafeCell::new(MaybeUninit::uninit()),
            })
            .collect::<Box<[_]>>();

        Self {
            buf,
            enqueue_pos: AtomicUsize::new(0),
            dequeue_pos: AtomicUsize::new(0),
            mask: cap - 1,
            cap,
        }
    }

    /// Push one record. Safe to call from any number of threads concurrently.
    ///
    /// On success the record is consumed. If the ring is at capacity the record
    /// is handed back via `Err(record)` (no blocking) so the caller can retry it
    /// under a lossless overflow policy instead of losing it.
    pub fn push(&self, record: T) -> Result<(), T> {
        cells_push(&self.buf, &self.enqueue_pos, record)
    }

    /// Pop one record in FIFO order. Safe to call from any number of threads
    /// concurrently (multi-consumer).
    ///
    /// Returns `None` if the ring is empty (no blocking). The CAS on
    /// `dequeue_pos` is the linearisation point: exactly one consumer wins each
    /// position, so two consumers never read the same cell. This is the mirror
    /// of [`push`](Self::push) — see the cell `sequence` protocol there.
    pub fn dequeue(&self) -> Option<T> {
        cells_dequeue(&self.buf, &self.dequeue_pos)
    }

    /// Number of items currently in the ring (snapshot, not linearizable).
    pub fn len(&self) -> usize {
        let enqueue = self.enqueue_pos.load(Ordering::Acquire);
        let dequeue = self.dequeue_pos.load(Ordering::Acquire);
        enqueue.wrapping_sub(dequeue)
    }

    /// True if the ring contains no items (snapshot).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The ring's actual capacity (the power-of-two `new` rounded up to).
    pub fn cap(&self) -> usize {
        self.cap
    }
}

#[cfg(feature = "alloc")]
impl<T> Drop for Ring<T> {
    fn drop(&mut self) {
        // exclusive &mut self, so a relaxed load is a plain read (no contention);
        // get_mut is unavailable on loom atomics, load works on both.
        let dequeue = self.dequeue_pos.load(Ordering::Relaxed);
        let enqueue = self.enqueue_pos.load(Ordering::Relaxed);

        let mut pos = dequeue;
        while pos != enqueue {
            let slot = pos & self.mask;
            let published = self.buf[slot].sequence.load(Ordering::Relaxed) == pos.wrapping_add(1);
            if published {
                // SAFETY: exclusive &mut self, so no producer is mid-write; a
                // sequence of pos+1 means this slot holds a published, unread item.
                unsafe {
                    drop_cell(&self.buf[slot].data);
                }
            }
            pos = pos.wrapping_add(1);
        }
    }
}

// Slots ahead to prefetch the drain output buffer. The store stream into the
// caller's buffer is latency-bound (read-for-ownership) once it exceeds L1; at
// ~6 ns/op and ~100 ns miss latency the cover distance is ~16 (latency / op).
// Tuned: K=16 takes drain_into from 79 -> 6.3 ns/elem at 1M (bench_ring_decompose).
#[cfg(feature = "alloc")]
const PREFETCH_DISTANCE: usize = 16;

/// Batch-drain handle for a [`Ring<T>`]. The ring is multi-consumer, so any
/// number of `Drainer`s (or bare [`Ring::dequeue`] callers) may drain the same
/// ring concurrently while producers push — each pop is CAS-linearised on
/// `dequeue_pos`.
#[cfg(feature = "alloc")]
pub struct Drainer<'ring, T> {
    ring: &'ring Ring<T>,
}

#[cfg(feature = "alloc")]
impl<'ring, T> Drainer<'ring, T> {
    /// Wrap a ring reference in a drainer.
    pub fn new(ring: &'ring Ring<T>) -> Self {
        Self { ring }
    }

    /// Move up to `out.len()` items out of the ring into `out` in FIFO order.
    ///
    /// Returns the number of items written. Each item is claimed via the
    /// multi-consumer [`Ring::dequeue`] CAS, so concurrent drainers split the
    /// stream without ever reading the same cell; the loop stops at the first
    /// empty pop (another consumer may have raced us to the tail).
    pub fn drain_into(&mut self, out: &mut [T]) -> usize {
        let mut count = 0;
        while count < out.len() {
            match self.ring.dequeue() {
                Some(record) => {
                    // the store stream into the caller's buffer is latency-bound
                    // (RFO) past L1; prefetch the slot we will write K ahead so it
                    // does not stall, keeping the drain flat at large batch sizes.
                    let ahead = count + PREFETCH_DISTANCE;
                    if ahead < out.len() {
                        // SAFETY: ahead < out.len(), so the pointer is in bounds;
                        // it is only passed to a prefetch hint, never dereferenced.
                        let slot = unsafe { out.as_ptr().add(ahead) };
                        crate::arch::prefetch_for_write(slot.cast());
                    }
                    out[count] = record;
                    count += 1;
                }
                None => break,
            }
        }
        count
    }
}

/// No-alloc, const-capacity bounded ring with lock-free MPMC access — the
/// bare-metal tier of [`Ring`]. The `[Cell<T>; N]` buffer lives **inline** (no
/// heap), so it compiles and runs where `alloc` is unavailable
/// (DPDK/SPDK/embedded). Same Vyukov algorithm as [`Ring`] — both delegate to the
/// shared `cells_*` fns, so loom's model-check of that algorithm covers this too.
///
/// `N` MUST be a power of two >= 2 (asserted in [`StaticRing::new`]); the
/// protocol needs it for the `pos & (N - 1)` slot index.
///
/// # Why [`new`](StaticRing::new) is not `const` — yet
///
/// The whole point of the no-alloc tier is to back a `static` on bare metal — no
/// heap, and no lazy `OnceCell`/runtime init either. That wants a `const fn new`
/// so a caller can write `static RING: StaticRing<T, N> = StaticRing::new();`.
/// It is NOT const-constructible today, and the blocker is a real stable-Rust
/// limitation, not a design choice — worth spelling out so nobody re-derives it:
///
/// The Vyukov protocol seeds **each cell's `sequence` to its own index** (cell
/// `i` starts at `i`, so the empty-slot test `seq == pos` holds slot-by-slot).
/// Stable `const fn` cannot build a `[Cell<T>; N]` with per-index values:
/// - `core::array::from_fn(|i| …)` — the obvious tool — is **not `const`**.
/// - `[const { EXPR }; N]` requires `EXPR` independent of the index (every cell
///   identical), so it cannot set `sequence = i`.
/// - a `const` `while` loop *can* fill a `[MaybeUninit<Cell<T>>; N]`, but turning
///   that into `[Cell<T>; N]` needs `MaybeUninit::array_assume_init`
///   (**unstable**), and `transmute::<[MaybeUninit<Cell<T>>; N], [Cell<T>; N]>`
///   is rejected — **`E0512`**, transmute on a const-generic-`N` (dependently
///   sized) array. `transmute_unchecked` would work but is also unstable.
///
/// TODO(const-new): make `new` a `const fn` as soon as ANY of these stabilises —
/// a `const core::array::from_fn`, a `const MaybeUninit::array_assume_init`, or a
/// `transmute`/`transmute_unchecked` that accepts equal-size `[_; N]` arrays.
/// Nothing else blocks it (the pow2 assert is already `const`-friendly), and it
/// is the last missing piece for a genuinely heap-free `static` sink stack
/// ([`StaticBoundedQueue`](crate::ring::StaticBoundedQueue) and the pipe-forms
/// `StaticSinkFront` inherit the same limitation transitively through here).
pub struct StaticRing<T, const N: usize> {
    buf: [Cell<T>; N],
    enqueue_pos: AtomicUsize,
    dequeue_pos: AtomicUsize,
}

// SAFETY: identical protocol to `Ring`; only WHERE the cells live changes (inline
// vs heap), not HOW they synchronise. See `Ring`'s Send/Sync safety notes.
unsafe impl<T: Send, const N: usize> Send for StaticRing<T, N> {}
unsafe impl<T: Send, const N: usize> Sync for StaticRing<T, N> {}

impl<T, const N: usize> StaticRing<T, N> {
    /// Construct an empty ring. Panics if `N` is not a power of two >= 2.
    #[must_use]
    pub fn new() -> Self {
        assert!(
            N >= 2 && N.is_power_of_two(),
            "StaticRing capacity N must be a power of two >= 2"
        );
        Self {
            buf: core::array::from_fn(|index| Cell {
                sequence: AtomicUsize::new(index),
                data: UnsafeCell::new(MaybeUninit::uninit()),
            }),
            enqueue_pos: AtomicUsize::new(0),
            dequeue_pos: AtomicUsize::new(0),
        }
    }

    /// Push one record; `Err(record)` if full (no blocking). MPMC-safe.
    pub fn push(&self, record: T) -> Result<(), T> {
        cells_push(&self.buf, &self.enqueue_pos, record)
    }

    /// Pop one record in FIFO order; `None` if empty. MPMC-safe.
    pub fn dequeue(&self) -> Option<T> {
        cells_dequeue(&self.buf, &self.dequeue_pos)
    }

    /// Items currently buffered (snapshot, not linearizable).
    #[must_use]
    pub fn len(&self) -> usize {
        self.enqueue_pos
            .load(Ordering::Acquire)
            .wrapping_sub(self.dequeue_pos.load(Ordering::Acquire))
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub const fn capacity(&self) -> usize {
        N
    }
}

impl<T, const N: usize> Default for StaticRing<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Drop for StaticRing<T, N> {
    fn drop(&mut self) {
        // exclusive access in Drop; dequeue drains + drops each buffered record.
        // Empty slots hold uninit data that dequeue never reads (protocol-gated).
        while self.dequeue().is_some() {}
    }
}

// Ring/StaticRing's atomics + UnsafeCell are cfg-swapped to loom under
// `--features loom` (see the module doc comment above) — those only work
// inside an actual loom::model(...) closure, which these plain #[test]
// functions don't provide.
#[cfg(all(test, not(feature = "loom")))]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
mod tests {
    extern crate std;

    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::StaticRing;

    // Ring/Drainer are alloc-gated (see their #[cfg(feature = "alloc")]
    // definitions above); nested so a no-alloc build still exercises
    // StaticRing's tests below instead of losing this whole file's coverage.
    #[cfg(feature = "alloc")]
    mod ring_tests {
        use super::*;
        use std::sync::Arc;
        use std::thread;

        use super::super::{Drainer, Ring};
        use alloc::collections::VecDeque;
        use alloc::vec;
        use alloc::vec::Vec;
        use rstest::rstest;

        #[rstest]
        #[case::push_all_drain_all(8)]
        #[case::small_cap(2)]
        #[case::large_cap(64)]
        fn happy_push_n_drain_n_in_order(#[case] cap: usize) {
            let ring = Ring::new(cap).unwrap();
            for item in 0..cap {
                ring.push(item as u32).unwrap();
            }

            let mut out = vec![0u32; cap];
            let mut drainer = Drainer::new(&ring);
            let received = drainer.drain_into(&mut out);

            assert_eq!(received, cap);
            for (idx, value) in out.iter().enumerate() {
                assert_eq!(*value, idx as u32);
            }
        }

        #[rstest]
        #[case::small(2)]
        #[case::many(16)]
        fn sad_push_into_full_ring_returns_full(#[case] cap: usize) {
            let ring = Ring::<u64>::new(cap).unwrap();
            for item in 0..ring.cap() {
                ring.push(item as u64).unwrap();
            }

            // a full ring hands the record back so a lossless caller can retry it.
            assert_eq!(ring.push(99), Err(99));
        }

        // Worked example (paper proof): cap=4, fill to capacity, 5th push is Full
        // (never overwrites slot 0's unconsumed item), drain recovers FIFO, then the
        // freed slot accepts a new lap. This is the locked test for the Vyukov
        // full-detection + reclaim derivation.
        #[test]
        fn worked_example_fifo_full_at_cap_then_reclaim() {
            let ring = Ring::new(4).unwrap();
            assert_eq!(ring.cap(), 4);

            for value in 0..4u32 {
                ring.push(value).unwrap();
            }
            assert_eq!(ring.push(99), Err(99));

            let mut out = [0u32; 4];
            let mut drainer = Drainer::new(&ring);
            assert_eq!(drainer.drain_into(&mut out), 4);
            assert_eq!(out, [0, 1, 2, 3]);

            // slot 0 (and the rest) freed for the next lap.
            ring.push(7).unwrap();
            let mut out2 = [0u32; 1];
            assert_eq!(drainer.drain_into(&mut out2), 1);
            assert_eq!(out2[0], 7);
        }

        #[test]
        fn edge_min_cap_push_full_drain_push_again() {
            let ring = Ring::new(2).unwrap();

            ring.push(42u32).unwrap();
            ring.push(43u32).unwrap();
            assert_eq!(ring.push(99u32), Err(99));

            let mut out = [0u32; 2];
            let mut drainer = Drainer::new(&ring);
            assert_eq!(drainer.drain_into(&mut out), 2);
            assert_eq!(out, [42, 43]);

            ring.push(7u32).unwrap();
            let mut out2 = [0u32; 1];
            assert_eq!(drainer.drain_into(&mut out2), 1);
            assert_eq!(out2[0], 7);
        }

        #[test]
        fn cap_rounds_up_to_power_of_two_min_two() {
            assert_eq!(Ring::<u8>::new(1).unwrap().cap(), 2);
            assert_eq!(Ring::<u8>::new(3).unwrap().cap(), 4);
            assert_eq!(Ring::<u8>::new(1000).unwrap().cap(), 1024);
            assert_eq!(Ring::<u8>::new(4096).unwrap().cap(), 4096);
        }

        #[test]
        fn with_capacity_is_infallible_and_clamps_zero_to_min() {
            // same pow2 rounding as `new`, but no Result — and 0 clamps to the min
            // rather than erroring, for callers whose capacity is already nonzero.
            assert_eq!(Ring::<u8>::with_capacity(0).cap(), 2);
            assert_eq!(Ring::<u8>::with_capacity(1).cap(), 2);
            assert_eq!(Ring::<u8>::with_capacity(3).cap(), 4);
            let ring = Ring::with_capacity(1000);
            assert_eq!(ring.cap(), 1024);
            ring.push(7u32).unwrap();
            assert_eq!(ring.dequeue(), Some(7));
        }

        #[test]
        fn drop_items_inside_ring_are_dropped_properly() {
            let counter = Arc::new(AtomicUsize::new(0));

            struct Counted(Arc<AtomicUsize>);
            impl Drop for Counted {
                fn drop(&mut self) {
                    self.0.fetch_add(1, Ordering::Relaxed);
                }
            }

            {
                let ring = Ring::new(4).unwrap();
                // Counted isn't Debug, so assert on is_ok() rather than unwrap().
                assert!(ring.push(Counted(Arc::clone(&counter))).is_ok());
                assert!(ring.push(Counted(Arc::clone(&counter))).is_ok());
                // ring goes out of scope here with 2 items still inside
            }

            assert_eq!(counter.load(Ordering::Relaxed), 2);
        }

        #[test]
        fn concurrency_single_producer_order_preserved() {
            const ITEMS: u32 = 10_000;
            let ring = Arc::new(Ring::<u32>::new(512).unwrap());

            let producer_ring = Arc::clone(&ring);
            let producer = thread::spawn(move || {
                let mut pushed = 0u32;
                while pushed < ITEMS {
                    if producer_ring.push(pushed).is_ok() {
                        pushed += 1;
                    } else {
                        thread::yield_now();
                    }
                }
            });

            let consumer_ring = Arc::clone(&ring);
            let consumer = thread::spawn(move || {
                let mut received = Vec::with_capacity(ITEMS as usize);
                let mut buf = vec![0u32; 64];
                let mut drainer = Drainer::new(&*consumer_ring);
                while received.len() < ITEMS as usize {
                    let count = drainer.drain_into(&mut buf);
                    received.extend_from_slice(&buf[..count]);
                    if count == 0 {
                        thread::yield_now();
                    }
                }
                received
            });

            producer.join().expect("producer panicked");
            let received = consumer.join().expect("consumer panicked");

            assert_eq!(received.len(), ITEMS as usize);
            for (idx, value) in received.iter().enumerate() {
                assert_eq!(*value, idx as u32);
            }
        }

        // The multi-producer safety proof. P producers (>> ring slots, to force many
        // producers onto each cell) each push a disjoint id range; every record
        // carries a redundant checksum so a torn write is detectable. One consumer
        // drains concurrently. Invariants asserted: no torn record (checksum holds),
        // and the drained id multiset is exactly the pushed set (no loss, no dupe) —
        // which is precisely what the old non-atomic SPSC push violated when two
        // threads shared a ring.
        #[test]
        fn mpsc_many_producers_no_loss_no_tear() {
            #[derive(Clone, Copy)]
            struct Tagged {
                id: u64,
                check: u64,
            }
            const MAGIC: u64 = 0x9E37_79B9_7F4A_7C15;
            fn make(id: u64) -> Tagged {
                Tagged {
                    id,
                    check: id ^ MAGIC,
                }
            }

            const PRODUCERS: u64 = 8;
            const PER_PRODUCER: u64 = 4_000;
            const TOTAL: usize = (PRODUCERS * PER_PRODUCER) as usize;

            for _round in 0..5 {
                // small ring => every cell is shared by several producers.
                let ring = Arc::new(Ring::<Tagged>::new(16).unwrap());

                let producers: Vec<_> = (0..PRODUCERS)
                    .map(|producer_id| {
                        let ring = Arc::clone(&ring);
                        thread::spawn(move || {
                            let base = producer_id * PER_PRODUCER;
                            let mut sent = 0u64;
                            while sent < PER_PRODUCER {
                                if ring.push(make(base + sent)).is_ok() {
                                    sent += 1;
                                } else {
                                    thread::yield_now();
                                }
                            }
                        })
                    })
                    .collect();

                let consumer_ring = Arc::clone(&ring);
                let consumer = thread::spawn(move || {
                    let mut seen: Vec<bool> = vec![false; TOTAL];
                    let mut received = 0usize;
                    let mut buf = vec![make(0); 32];
                    let mut drainer = Drainer::new(&*consumer_ring);
                    while received < TOTAL {
                        let count = drainer.drain_into(&mut buf);
                        if count == 0 {
                            thread::yield_now();
                            continue;
                        }
                        for tagged in buf.iter().take(count) {
                            assert_eq!(tagged.check, tagged.id ^ MAGIC, "torn record");
                            let id = tagged.id as usize;
                            assert!(!seen[id], "duplicate id {id}");
                            seen[id] = true;
                            received += 1;
                        }
                    }
                    seen
                });

                for producer in producers {
                    producer.join().expect("producer panicked");
                }
                let seen = consumer.join().expect("consumer panicked");
                assert!(seen.into_iter().all(|hit| hit), "missing ids");
            }
        }

        // Worked example (paper proof) for the MPMC dequeue: cap=2, two slots full,
        // two consumers race. Between them they recover {A, B} once each in FIFO
        // order, a third dequeue is empty, and the freed slots accept the next lap.
        // This is the locked test for the Vyukov MPMC dequeue CAS derivation
        // (docs/tracing/discipline.md, "Lossless backpressure").
        #[test]
        fn worked_example_mpmc_two_consumers_fifo_then_reclaim() {
            let ring = Ring::new(2).unwrap();
            ring.push(b'A').unwrap();
            ring.push(b'B').unwrap();
            assert_eq!(
                ring.push(b'C'),
                Err(b'C'),
                "full ring hands the record back"
            );

            // two consumers between them pop both items, FIFO, once each.
            let first = ring.dequeue().expect("first pop");
            let second = ring.dequeue().expect("second pop");
            assert_eq!([first, second], [b'A', b'B'], "FIFO order across consumers");
            assert_eq!(ring.dequeue(), None, "empty after both consumed");

            // the freed slots accept the next lap.
            ring.push(b'D').unwrap();
            assert_eq!(ring.dequeue(), Some(b'D'));
        }

        // The multi-CONSUMER safety proof — the dequeue mirror of
        // mpsc_many_producers_no_loss_no_tear. P producers push a disjoint id range;
        // C consumers drain concurrently. Every record carries a redundant checksum
        // (torn-write detector); each consumer marks ids in a shared seen-set under a
        // no-duplicate assertion. Invariants: no torn record, no id consumed twice
        // (the dequeue CAS gives each cell to exactly one consumer), and the union of
        // all consumers' drains == the pushed set (no loss). This is exactly what a
        // non-atomic single-consumer drain run multi-consumer would violate.
        #[test]
        fn mpmc_many_producers_many_consumers_no_loss_no_tear() {
            #[derive(Clone, Copy)]
            struct Tagged {
                id: u64,
                check: u64,
            }
            const MAGIC: u64 = 0x9E37_79B9_7F4A_7C15;
            fn make(id: u64) -> Tagged {
                Tagged {
                    id,
                    check: id ^ MAGIC,
                }
            }

            const PRODUCERS: u64 = 8;
            const CONSUMERS: usize = 4;
            const PER_PRODUCER: u64 = 4_000;
            const TOTAL: usize = (PRODUCERS * PER_PRODUCER) as usize;

            for _round in 0..5 {
                // small ring => every cell is shared by several producers AND consumers.
                let ring = Arc::new(Ring::<Tagged>::new(16).unwrap());
                // shared seen-set: one slot per id, asserted set-once by whichever
                // consumer pops it. A torn dequeue CAS would let two consumers claim
                // the same cell and trip the duplicate assertion.
                let seen: Arc<Vec<AtomicUsize>> =
                    Arc::new((0..TOTAL).map(|_| AtomicUsize::new(0)).collect());
                let received = Arc::new(AtomicUsize::new(0));

                let producers: Vec<_> = (0..PRODUCERS)
                    .map(|producer_id| {
                        let ring = Arc::clone(&ring);
                        thread::spawn(move || {
                            let base = producer_id * PER_PRODUCER;
                            let mut sent = 0u64;
                            while sent < PER_PRODUCER {
                                if ring.push(make(base + sent)).is_ok() {
                                    sent += 1;
                                } else {
                                    thread::yield_now();
                                }
                            }
                        })
                    })
                    .collect();

                let consumers: Vec<_> = (0..CONSUMERS)
                    .map(|_| {
                        let ring = Arc::clone(&ring);
                        let seen = Arc::clone(&seen);
                        let received = Arc::clone(&received);
                        thread::spawn(move || {
                            while received.load(Ordering::Relaxed) < TOTAL {
                                match ring.dequeue() {
                                    Some(tagged) => {
                                        assert_eq!(tagged.check, tagged.id ^ MAGIC, "torn record");
                                        let prior = seen[tagged.id as usize]
                                            .fetch_add(1, Ordering::Relaxed);
                                        assert_eq!(prior, 0, "duplicate id {}", tagged.id);
                                        received.fetch_add(1, Ordering::Relaxed);
                                    }
                                    None => thread::yield_now(),
                                }
                            }
                        })
                    })
                    .collect();

                for producer in producers {
                    producer.join().expect("producer panicked");
                }
                for consumer in consumers {
                    consumer.join().expect("consumer panicked");
                }
                assert!(
                    seen.iter().all(|hit| hit.load(Ordering::Relaxed) == 1),
                    "every id consumed exactly once"
                );
            }
        }

        #[rstest]
        #[case::tiny(4, 100)]
        #[case::medium(32, 1_000)]
        #[case::large(256, 10_000)]
        fn property_random_push_drain_interleaving_preserves_fifo(
            #[case] cap: usize,
            #[case] total_ops: usize,
        ) {
            let ring = Ring::new(cap).unwrap();
            let mut drainer = Drainer::new(&ring);
            let mut expected_queue: VecDeque<u32> = VecDeque::new();
            let mut next_push_value = 0u32;
            let mut drain_buf = vec![0u32; cap];

            for op_idx in 0..total_ops {
                let do_push = op_idx % 3 != 2;

                if do_push && expected_queue.len() < cap {
                    if ring.push(next_push_value).is_ok() {
                        expected_queue.push_back(next_push_value);
                        next_push_value = next_push_value.wrapping_add(1);
                    }
                } else {
                    let count = drainer.drain_into(&mut drain_buf);
                    for drained in drain_buf.iter().take(count) {
                        let expected = expected_queue.pop_front().expect("queue underflow");
                        assert_eq!(*drained, expected);
                    }
                }
            }
        }

        #[test]
        fn invalid_cap_zero_returns_error() {
            let result = Ring::<u32>::new(0);
            assert!(matches!(result, Err(crate::ring::CapacityError)));
        }
    }

    #[test]
    fn static_ring_fifo_roundtrip() {
        let ring = StaticRing::<u32, 4>::new();
        assert_eq!(ring.capacity(), 4);
        assert!(ring.is_empty());
        for value in [10, 20, 30] {
            ring.push(value).unwrap();
        }
        assert_eq!(ring.len(), 3);
        assert_eq!(ring.dequeue(), Some(10));
        assert_eq!(ring.dequeue(), Some(20));
        assert_eq!(ring.dequeue(), Some(30));
        assert_eq!(ring.dequeue(), None);
    }

    #[test]
    fn static_ring_full_hands_record_back_without_blocking() {
        let ring = StaticRing::<u32, 4>::new();
        for value in [1, 2, 3, 4] {
            ring.push(value).unwrap();
        }
        // at capacity: the record comes back via Err, never dropped, never blocks.
        assert_eq!(ring.push(5), Err(5));
        assert_eq!(ring.len(), 4);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn static_ring_non_power_of_two_panics() {
        let _ring = StaticRing::<u32, 6>::new();
    }

    #[test]
    fn static_ring_drop_drains_buffered_records() {
        extern crate std;
        use std::sync::Arc;

        struct Counted(Arc<AtomicUsize>);
        impl Drop for Counted {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        {
            let ring = StaticRing::<Counted, 8>::new();
            for _ in 0..3 {
                assert!(ring.push(Counted(Arc::clone(&drops))).is_ok());
            }
        } // ring dropped with 3 buffered — its Drop must drain them
        assert_eq!(drops.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn static_ring_concurrent_mpmc_no_loss_no_dup() {
        extern crate std;
        use std::sync::Arc;
        use std::thread;

        const PRODUCERS: usize = 4;
        const PER: u32 = 10_000;
        let total = PRODUCERS * PER as usize;

        let ring = Arc::new(StaticRing::<u32, 1024>::new());
        let seen = Arc::new(AtomicUsize::new(0));
        let sum = Arc::new(AtomicUsize::new(0));
        let mut handles = std::vec::Vec::new();

        for producer in 0..PRODUCERS {
            let ring = Arc::clone(&ring);
            handles.push(thread::spawn(move || {
                for index in 0..PER {
                    let value = producer as u32 * PER + index;
                    while ring.push(value).is_err() {
                        std::hint::spin_loop();
                    }
                }
            }));
        }
        for _ in 0..2 {
            let ring = Arc::clone(&ring);
            let seen = Arc::clone(&seen);
            let sum = Arc::clone(&sum);
            handles.push(thread::spawn(move || {
                loop {
                    if let Some(value) = ring.dequeue() {
                        seen.fetch_add(1, Ordering::Relaxed);
                        sum.fetch_add(value as usize, Ordering::Relaxed);
                    } else if seen.load(Ordering::Relaxed) >= total {
                        break;
                    } else {
                        std::hint::spin_loop();
                    }
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(seen.load(Ordering::Relaxed), total, "no item lost");
        // every distinct value 0..total appeared exactly once → sum is unique-sum.
        let expected: usize = (0..total).sum();
        assert_eq!(
            sum.load(Ordering::Relaxed),
            expected,
            "no duplicate or torn read"
        );
    }
}
