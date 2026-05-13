//! `BucketTable<V>` — a lock-free, fixed-capacity, open-addressing concurrent
//! map keyed by owned byte strings, returning shared `Arc<V>` handles.
//!
//! It exists because the workspace has no `no_std` concurrent map and no
//! `no_std` lock: `dashmap` (rate_limit's old backing) is std-only and pulls a
//! sharded-lock design. `BucketTable` composes two primitives only:
//! [`rustc_hash::FxHasher`] (the same fast non-cryptographic hash a downstream consumer uses, for
//! the home-slot index and the per-slot fast-reject) and [`arc_swap::ArcSwap`]
//! (safe lock-free atomic-`Arc` publication: reads are a lock-free `load`, and
//! reclamation is safe because a slot's whole state is published behind one
//! atomically-swapped `Arc` that is only ever REPLACED, never mutated in place,
//! so a reader's `load` returns a consistent whole snapshot — old `Arc` or new
//! `Arc`, never a mix). Capacity is fixed at construction from `max_keys` with
//! load-factor headroom, so a miss-probe always reaches an `Empty` slot and
//! terminates. An insert builds its `Full` state and installs it with a single
//! CAS against the exact `Arc` it observed; the loser of a same-key race drops
//! its value and re-probes onto the winner's entry, so racers converge on ONE
//! published bucket. `make` runs at most once per call (lazily, reused across
//! CAS retries) and never on a read hit; under a same-key insert race it may
//! run once per racer with all but the winner's value dropped (rate-limit
//! buckets are cheap to build and idempotent, so this is preferred over a
//! reservation state that would cost an extra allocation on every insert and
//! could strand a slot if `make` panicked).
//!
//! Why arc-swap and not the prior seqlock: an earlier hand-rolled `UnsafeCell` +
//! generation seqlock (commit 5fe261c1) was loom-PROVEN to have a data race —
//! the optimistic read of a non-`Copy` `Arc` payload physically races a
//! reclaiming writer (undefined behaviour). arc-swap is itself loom-tested
//! upstream and gives the same lock-free reads with no raw-pointer escape
//! hatch in this module. Safe Rust cannot data-race, so there is no loom model
//! to carry.
//!
//! Tier: **std only**. arc-swap is std for our purposes (its `no_std` support is
//! a nightly experimental feature we do not use). A `no_std + alloc` tier would
//! need a `no_std` safe atomic-`Arc` (none in-tree) — deferred, and not
//! reachable through `proxima-pipe` today anyway (rate_limit, the sole consumer,
//! is itself std-only).

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::hash::Hasher;
use core::sync::atomic::{AtomicUsize, Ordering};

use arc_swap::ArcSwap;
use rustc_hash::FxHasher;

const LOAD_FACTOR_NUM: usize = 4;
const LOAD_FACTOR_DEN: usize = 3;

#[inline]
fn fxhash(key: &[u8]) -> u64 {
    let mut hasher = FxHasher::default();
    hasher.write(key);
    hasher.finish()
}

/// The whole published state of one slot. `Full` is immutable once published
/// and is only ever REPLACED behind the slot's `ArcSwap`, never mutated in
/// place — so any reader's `load` sees a consistent whole snapshot.
enum SlotState<V> {
    /// Never used; a probe terminating here proves the key is absent.
    Empty,
    /// A removed entry; a probe continues past it but records the first one as
    /// reusable space.
    Tomb,
    /// A live entry. Immutable once published.
    Full {
        hash: u64,
        key: Box<[u8]>,
        bucket: Arc<V>,
    },
}

/// One open-addressing slot: always holds an `Arc<SlotState<V>>`, swapped
/// atomically on every transition.
struct Slot<V>(ArcSwap<SlotState<V>>);

/// Lock-free fixed-capacity concurrent map: byte-key -> `Arc<V>`.
pub struct BucketTable<V> {
    slots: Box<[Slot<V>]>,
    /// shared canonical `Tomb` singleton, so a CAS that installs/claims against
    /// a tomb has a stable pointer to guess. (`Empty` needs no field — every
    /// slot starts at one shared `Empty` and an installing reader CASes against
    /// the exact `Arc` its own `load` returned.)
    tomb: Arc<SlotState<V>>,
    /// advisory live-entry count (Relaxed).
    count: AtomicUsize,
    mask: u64,
    max_keys: usize,
}

impl<V> BucketTable<V> {
    /// Build a table sized for at most `max_keys` live entries. CAP is the next
    /// power of two at or above `ceil(max_keys * 4 / 3)`, then bumped to
    /// guarantee `CAP > max_keys` so a probe that misses always hits an `Empty`
    /// slot.
    ///
    /// This is the only constructor: a single-parameter type does not earn a
    /// separate `new` + builder surface (the principle-4 fluent-surface
    /// exception for one mandatory parameter), so `with_max_keys` is both the
    /// terse and the fluent form.
    #[must_use]
    pub fn with_max_keys(max_keys: usize) -> Self {
        let needed = max_keys
            .saturating_mul(LOAD_FACTOR_NUM)
            .div_ceil(LOAD_FACTOR_DEN);
        let mut capacity = needed.max(1).next_power_of_two().max(2);
        if capacity <= max_keys {
            capacity = (max_keys + 1).next_power_of_two();
        }
        assert!(
            capacity > max_keys,
            "bucket table capacity must exceed max_keys for probe termination"
        );
        let empty: Arc<SlotState<V>> = Arc::new(SlotState::Empty);
        let tomb: Arc<SlotState<V>> = Arc::new(SlotState::Tomb);
        let mut slots = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(Slot(ArcSwap::new(empty.clone())));
        }
        Self {
            slots: slots.into_boxed_slice(),
            tomb,
            count: AtomicUsize::new(0),
            mask: (capacity as u64) - 1,
            max_keys,
        }
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.slots.len()
    }

    #[inline]
    fn home(&self, hash: u64) -> usize {
        (hash & self.mask) as usize
    }

    #[inline]
    fn next(&self, index: usize) -> usize {
        ((index as u64 + 1) & self.mask) as usize
    }

    /// Advisory live-entry count (Relaxed).
    #[must_use]
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the shared handle for `key`, inserting one made by `make` if
    /// absent. Concurrent callers racing the same key converge on one published
    /// handle via a single install CAS (the loser re-probes onto the winner's
    /// entry). `make` runs at most once per call — lazily, only on a confirmed
    /// miss, and the built `Full` is reused across CAS retries — and never on a
    /// read hit. If `make` panics nothing has been installed, so the table is
    /// left unchanged (no stranded slot).
    pub fn get_or_insert(&self, key: &[u8], make: impl FnOnce() -> V) -> Arc<V> {
        let hash = fxhash(key);
        // fast path: a read-only probe. a hit returns without building anything.
        if let Some(bucket) = self.get(key, hash) {
            return bucket;
        }
        // confirmed miss: build the entry exactly once (make is consumed here, so
        // a panic in make leaves the table untouched — nothing was installed).
        let bucket = Arc::new(make());
        let full = Arc::new(SlotState::Full {
            hash,
            key: key.into(),
            bucket: bucket.clone(),
        });
        'restart: loop {
            let mut first_tomb: Option<usize> = None;
            let mut index = self.home(hash);
            for _ in 0..self.capacity() {
                let guard = self.slots[index].0.load();
                match &**guard {
                    SlotState::Empty => {
                        // key absent. reuse a recorded tomb if any, else this empty.
                        let (target, current) = match first_tomb {
                            Some(tomb_index) => (tomb_index, self.tomb.clone()),
                            None => (index, arc_swap::Guard::into_inner(guard)),
                        };
                        if self.install(target, &current, &full) {
                            return bucket;
                        }
                        continue 'restart;
                    }
                    SlotState::Tomb => {
                        if first_tomb.is_none() {
                            first_tomb = Some(index);
                        }
                        index = self.next(index);
                    }
                    SlotState::Full {
                        hash: stored_hash,
                        key: stored_key,
                        bucket: found,
                    } => {
                        // a racer published our key while we were building; use theirs.
                        if *stored_hash == hash && stored_key.as_ref() == key {
                            return found.clone();
                        }
                        index = self.next(index);
                    }
                }
            }
            // full wrap without an Empty: every slot is Full/Tomb. the key is
            // absent (we scanned all). reuse the first tomb if one was seen, else
            // the table is truly full of distinct keys — CAP > max_keys plus
            // eviction keeps that off the live path, so evict the lru and retry.
            if let Some(tomb_index) = first_tomb {
                let tomb = self.tomb.clone();
                if self.install(tomb_index, &tomb, &full) {
                    return bucket;
                }
                continue 'restart;
            }
            debug_assert!(
                self.count.load(Ordering::Relaxed) >= self.max_keys,
                "wrapped with no empty/tomb yet count below max_keys"
            );
            self.evict_one_lru(|_| 0);
        }
    }

    /// Read-only lookup: the shared handle for `key`, or `None` if absent. Stops
    /// at the first `Empty` (proof of absence), probing past `Tomb` and
    /// non-matching `Full`. This is the hot path — a lock-free `load` per slot
    /// and an `Arc` clone on a hit, no allocation.
    #[must_use]
    pub fn get(&self, key: &[u8], hash: u64) -> Option<Arc<V>> {
        let mut index = self.home(hash);
        for _ in 0..self.capacity() {
            let guard = self.slots[index].0.load();
            match &**guard {
                SlotState::Empty => return None,
                SlotState::Full {
                    hash: stored_hash,
                    key: stored_key,
                    bucket,
                } if *stored_hash == hash && stored_key.as_ref() == key => {
                    return Some(bucket.clone());
                }
                _ => index = self.next(index),
            }
        }
        None
    }

    /// Install `full` at `target` with one CAS against `current` (the exact `Arc`
    /// the prober observed). Returns true on a win (and bumps `count`); false on
    /// a lost CAS so the caller re-probes onto the winner's entry. `full` is
    /// cloned per attempt (refcount only), so a retry never rebuilds the value.
    fn install(
        &self,
        target: usize,
        current: &Arc<SlotState<V>>,
        full: &Arc<SlotState<V>>,
    ) -> bool {
        let prev = self.slots[target].0.compare_and_swap(current, full.clone());
        let won = Arc::ptr_eq(&arc_swap::Guard::into_inner(prev), current);
        if won {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
        won
    }

    /// Tombstone the entry at `index` if it still holds `expected_full` (the
    /// exact `Arc` the caller observed). Skips if it changed underneath — a
    /// racing reader either saw `expected_full` (consistent) or sees `Tomb`,
    /// never a torn mix. Decrements `count` on a successful swap.
    fn tombstone(&self, index: usize, expected_full: &Arc<SlotState<V>>) {
        let prev = self.slots[index]
            .0
            .compare_and_swap(expected_full, self.tomb.clone());
        if Arc::ptr_eq(&arc_swap::Guard::into_inner(prev), expected_full) {
            self.count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Evict the `Full` slot whose `metric` is the strict minimum. Ties resolve
    /// to the LOWEST slot index (deterministic — a divergence from dashmap's
    /// nondeterministic iteration-order tie-break, documented intentionally).
    /// Best-effort: if the victim changed underneath the CAS it is skipped
    /// (matches dashmap's non-atomic evict).
    pub fn evict_one_lru(&self, metric: impl Fn(&V) -> u64) {
        let mut victim: Option<(usize, Arc<SlotState<V>>)> = None;
        let mut victim_metric: u64 = 0;
        for index in 0..self.capacity() {
            let guard = self.slots[index].0.load();
            if let SlotState::Full { bucket, .. } = &**guard {
                let measured = metric(bucket);
                if victim.is_none() || measured < victim_metric {
                    victim = Some((index, arc_swap::Guard::into_inner(guard)));
                    victim_metric = measured;
                }
            }
        }
        if let Some((index, full)) = victim {
            self.tombstone(index, &full);
        }
    }

    /// Tombstone every `Full` slot whose `metric` is strictly less than
    /// `now - ttl`. A value exactly equal to the cutoff survives.
    pub fn sweep_idle(&self, now: u64, ttl: u64, metric: impl Fn(&V) -> u64) {
        let cutoff = now.saturating_sub(ttl);
        for index in 0..self.capacity() {
            let guard = self.slots[index].0.load();
            let idle = match &**guard {
                SlotState::Full { bucket, .. } => metric(bucket) < cutoff,
                _ => false,
            };
            if idle {
                self.tombstone(index, &arc_swap::Guard::into_inner(guard));
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use core::sync::atomic::AtomicU64 as StdAtomicU64;
    use core::sync::atomic::Ordering::Relaxed;
    use rstest::rstest;
    use std::sync::Barrier;

    // a value type whose eviction metric is a settable atomic, so tests pin the
    // lru/sweep ordering deterministically without any clock. `last_access`
    // mirrors the real bucket shape (AtomicBucket::last_access_micros).
    struct Timed {
        last_access: StdAtomicU64,
    }

    impl Timed {
        fn at(value: u64) -> Self {
            Self {
                last_access: StdAtomicU64::new(value),
            }
        }
        fn set(&self, value: u64) {
            self.last_access.store(value, Relaxed);
        }
        fn metric(value: &Timed) -> u64 {
            value.last_access.load(Relaxed)
        }
    }

    // test-only readback of a slot's stored key + whether it is Full, so threaded
    // tests can assert the returned bucket's key matches the request.
    impl<V> BucketTable<V> {
        fn full_key_at(&self, index: usize) -> Option<Vec<u8>> {
            match &**self.slots[index].0.load() {
                SlotState::Full { key, .. } => Some(key.to_vec()),
                _ => None,
            }
        }
    }

    #[rstest]
    #[case::one(1)]
    #[case::two(2)]
    #[case::three(3)]
    #[case::seven(7)]
    #[case::hundred(100)]
    #[case::hundred_k(100_000)]
    fn cap_exceeds_max_keys(#[case] max_keys: usize) {
        let table: BucketTable<u8> = BucketTable::with_max_keys(max_keys);
        assert!(
            table.capacity() > max_keys,
            "cap {} must exceed max_keys {max_keys}",
            table.capacity()
        );
        assert!(table.capacity().is_power_of_two());
    }

    #[test]
    fn phase_walk_evict_reuses_tombstone_and_sweeps() {
        // 5-phase lifecycle: insert a,b; get a is the same Arc; touch a newer
        // than b then evict lru (=b); insert c reuses b's tomb; sweep strict-<
        // drops the stale entry and spares one exactly at the cutoff.
        let table: BucketTable<Timed> = BucketTable::with_max_keys(8);

        let arc_a = table.get_or_insert(b"a", || Timed::at(10));
        let arc_b = table.get_or_insert(b"b", || Timed::at(20));
        assert_eq!(table.len(), 2);

        let arc_a_again = table.get_or_insert(b"a", || Timed::at(999));
        assert!(
            Arc::ptr_eq(&arc_a, &arc_a_again),
            "same key returns the same Arc"
        );
        assert_eq!(Arc::strong_count(&arc_a), 3, "no second make ran for a");

        // a is now the newest, b the oldest -> evict b.
        arc_a.set(30);
        table.evict_one_lru(Timed::metric);
        assert_eq!(table.len(), 1);
        let arc_b_after = table.get_or_insert(b"b", || Timed::at(40));
        assert!(
            !Arc::ptr_eq(&arc_b, &arc_b_after),
            "b was evicted, fresh Arc on reinsert"
        );

        let arc_c = table.get_or_insert(b"c", || Timed::at(15));
        assert_eq!(table.len(), 3, "c reuses a tombstone, three live");

        // sweep at now=50, ttl=20 -> cutoff=30. a(30) survives (==cutoff, strict
        // boundary value), c(15) swept, b(40) kept.
        arc_a.set(30);
        arc_c.set(15);
        arc_b_after.set(40);
        table.sweep_idle(50, 20, Timed::metric);
        assert_eq!(table.len(), 2, "only c (strictly < cutoff) swept");
        assert!(Arc::ptr_eq(
            &table.get_or_insert(b"a", || Timed::at(0)),
            &arc_a
        ));
        assert!(Arc::ptr_eq(
            &table.get_or_insert(b"b", || Timed::at(0)),
            &arc_b_after
        ));
    }

    #[test]
    fn sweep_strict_less_than_cutoff_spares_equal() {
        let table: BucketTable<Timed> = BucketTable::with_max_keys(8);
        let equal = table.get_or_insert(b"equal", || Timed::at(100));
        let below = table.get_or_insert(b"below", || Timed::at(99));
        // now=200 ttl=100 -> cutoff=100. equal==cutoff survives, below<cutoff swept.
        table.sweep_idle(200, 100, Timed::metric);
        assert_eq!(table.len(), 1);
        assert!(Arc::ptr_eq(
            &table.get_or_insert(b"equal", || Timed::at(0)),
            &equal
        ));
        drop(below);
    }

    #[test]
    fn lru_tie_breaks_to_lowest_slot_index() {
        // three keys with EQUAL metric; the lowest occupied slot index is evicted.
        let table: BucketTable<Timed> = BucketTable::with_max_keys(8);
        let keys: [&[u8]; 3] = [b"k0", b"k1", b"k2"];
        let mut arcs = Vec::new();
        for key in keys {
            arcs.push(table.get_or_insert(key, || Timed::at(7)));
        }
        // find which key sits at the lowest slot index.
        let mut occupied: Vec<(usize, &[u8])> = Vec::new();
        for slot_index in 0..table.capacity() {
            if let Some(key_bytes) = table.full_key_at(slot_index)
                && let Some(found) = keys
                    .iter()
                    .copied()
                    .find(|candidate| *candidate == key_bytes.as_slice())
            {
                occupied.push((slot_index, found));
            }
        }
        occupied.sort_by_key(|(slot_index, _)| *slot_index);
        let lowest_key = occupied[0].1;

        table.evict_one_lru(Timed::metric);
        assert_eq!(table.len(), 2);
        // the lowest-index key is now absent (reinsert makes a fresh Arc).
        let reinserted = table.get_or_insert(lowest_key, || Timed::at(7));
        let original = arcs
            .iter()
            .zip(keys)
            .find(|(_, key)| *key == lowest_key)
            .map(|(arc, _)| arc)
            .unwrap();
        assert!(
            !Arc::ptr_eq(&reinserted, original),
            "lowest-index slot was the victim"
        );
    }

    // a hasher harness is impractical to inject through FxHasher; instead we
    // exercise the probe/collision path with real keys forced to collide by a
    // CAP=2 table (mask=1: every key homes to slot 0 or 1, so a second distinct
    // key probes into the other).
    #[test]
    fn collision_probe_and_concurrent_same_key_share_one_bucket() {
        let table: Arc<BucketTable<Timed>> = Arc::new(BucketTable::with_max_keys(1));
        assert_eq!(table.capacity(), 2);

        // concurrent same-key insert: both get the SAME published Arc (the
        // single install CAS admits one winner). make runs at most once per
        // racer (the loser's value is dropped), so 2 threads -> 1 or 2 builds.
        let make_calls = Arc::new(StdAtomicU64::new(0));
        let barrier = Arc::new(Barrier::new(2));
        let mut handles = Vec::new();
        for _ in 0..2 {
            let table = table.clone();
            let make_calls = make_calls.clone();
            let barrier = barrier.clone();
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                table.get_or_insert(b"shared", || {
                    make_calls.fetch_add(1, Relaxed);
                    Timed::at(1)
                })
            }));
        }
        let first = handles.pop().unwrap().join().unwrap();
        let second = handles.pop().unwrap().join().unwrap();
        assert!(
            Arc::ptr_eq(&first, &second),
            "racing same key shares one bucket"
        );
        let builds = make_calls.load(Relaxed);
        assert!(
            (1..=2).contains(&builds),
            "make ran at most once per racer, got {builds}"
        );
        assert_eq!(table.len(), 1, "exactly one entry published");

        // a second distinct key lands at the probed slot (CAP=2, one free).
        let other = table.get_or_insert(b"other", || Timed::at(2));
        assert_eq!(table.len(), 2);

        // get-across-tombstone: tombstone "shared", "other" still findable.
        for slot_index in 0..table.capacity() {
            if table.full_key_at(slot_index).as_deref() == Some(b"shared".as_slice()) {
                let full = arc_swap::Guard::into_inner(table.slots[slot_index].0.load());
                table.tombstone(slot_index, &full);
            }
        }
        assert_eq!(table.len(), 1);
        let other_again = table.get_or_insert(b"other", || Timed::at(99));
        assert!(
            Arc::ptr_eq(&other, &other_again),
            "found across the tombstone"
        );

        // reclaim the tombstone with a fresh key/Arc.
        let reclaimed = table.get_or_insert(b"shared", || Timed::at(3));
        assert!(
            !Arc::ptr_eq(&reclaimed, &first),
            "reclaimed slot, fresh Arc"
        );
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn concurrent_same_key_reclaim_no_duplicate() {
        // many threads racing the same key into a table where that key's slot may
        // be tombstoned-and-reclaimed must end with exactly one slot holding K
        // (len does not double-count). a second get returns the same Arc. the
        // arc-swap CAS is the single point that admits one winner per publish.
        let table: Arc<BucketTable<Timed>> = Arc::new(BucketTable::with_max_keys(64));
        let barrier = Arc::new(Barrier::new(8));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let table = table.clone();
            let barrier = barrier.clone();
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..50 {
                    let _ = table.get_or_insert(b"K", || Timed::at(1));
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
        // exactly one slot holds K.
        let k_slots = (0..table.capacity())
            .filter(|index| table.full_key_at(*index).as_deref() == Some(b"K".as_slice()))
            .count();
        assert_eq!(
            k_slots, 1,
            "K occupies exactly one slot, no duplicate insert"
        );
        assert_eq!(table.len(), 1, "count does not double-count K");
        let first = table.get_or_insert(b"K", || Timed::at(9));
        let second = table.get_or_insert(b"K", || Timed::at(9));
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn threaded_stress_returns_correct_key_and_stays_bounded() {
        // 8 threads churn a bounded keyspace with periodic eviction; every
        // returned bucket must carry the requested key and len stays within the
        // capacity headroom. deterministic (no sleeps).
        let max_keys = 64;
        let table: Arc<BucketTable<KeyedValue>> = Arc::new(BucketTable::with_max_keys(max_keys));
        let capacity = table.capacity();
        let barrier = Arc::new(Barrier::new(8));
        let mut handles = Vec::new();
        for thread_id in 0..8u64 {
            let table = table.clone();
            let barrier = barrier.clone();
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                for op in 0..10_000u64 {
                    let key = format!("k{}", (thread_id * 7 + op) % 128);
                    let bytes = key.as_bytes();
                    let bucket = table.get_or_insert(bytes, || KeyedValue::new(bytes));
                    assert_eq!(
                        bucket.key.as_slice(),
                        bytes,
                        "returned bucket carries the requested key"
                    );
                    bucket.last_access.store(op, Relaxed);
                    if op % 256 == 0 {
                        table.evict_one_lru(|value| value.last_access.load(Relaxed));
                    }
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
        assert!(
            table.len() <= capacity,
            "len {} must stay within capacity {capacity}",
            table.len()
        );
    }

    // a value carrying its own key bytes so the stress test can verify the
    // returned bucket matches the requested key.
    struct KeyedValue {
        key: Vec<u8>,
        last_access: StdAtomicU64,
    }

    impl KeyedValue {
        fn new(key: &[u8]) -> Self {
            Self {
                key: key.to_vec(),
                last_access: StdAtomicU64::new(0),
            }
        }
    }
}
