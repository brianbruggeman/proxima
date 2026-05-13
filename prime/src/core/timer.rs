//! per-core hashed timer wheel. single-thread-owned (`!Send`, `!Sync`).
//!
//! design: 4-level hierarchical hashed wheel (Varghese & Lauck 1987).
//! `LEVELS = 4` wheels, each with `BOTTOM_SLOTS` (256) slots. Level `k`
//! has granularity `BOTTOM_SLOTS^k` ticks, covering the deadline range
//! `[base_k, base_k + BOTTOM_SLOTS^(k+1))`. With 256 slots × 4 levels
//! we cover `256^4 = 4.29 G` ticks above `current_tick` — at ms
//! resolution, ~50 days. Deadlines beyond that are clamped into the
//! top level (lazy cascade still drains them once `current_tick`
//! catches up).
//!
//! Per-level slot is a `u32` head into an intrusive linked list
//! chained through `Entry::next`. Insert is O(1): pick level via
//! `(64 - offset.leading_zeros() - 1) >> 8` (with a fast-path for
//! `offset < BOTTOM_SLOTS`), compute slot via shift+mask, prepend.
//! Zero allocations on the hot path. No `BTreeMap`.
//!
//! cancellation is O(1) via slab generation; cancelled entries are
//! skipped on fire (lazy removal).
//!
//! advance/cascade: when `current_tick` crosses a multiple of L1's
//! granularity (256), the L1 slot whose range now overlaps L0 is
//! drained and each entry re-placed (it may still fall in L1 if
//! `deadline` is far enough out, but typically falls into L0). Same
//! pattern for L2 → L1, L3 → L2.
//!
//! `Tick = u64`. resolution chosen by the `Clock` impl — std impl
//! uses milliseconds; bench / test impls use plain integer counters.
//!
//! no_std + alloc only — `core::*` and `alloc::*` exclusively.

extern crate alloc;

use alloc::vec::Vec;
use core::task::Waker;

use super::sized;

pub type Tick = u64;

/// monotonic tick source. `now()` must be non-decreasing.
pub trait Clock {
    fn now(&self) -> Tick;
}

/// opaque handle returned by `register`; pass to `cancel` to remove a timer
/// before it fires. cheap to copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimerKey {
    slab_index: u32,
    generation: u32,
}

struct Entry {
    deadline: Tick,
    waker: Option<Waker>,
    generation: u32,
    /// intrusive linked-list pointer. carries one of two meanings:
    ///
    ///   - in a wheel slot (any level): next entry in that slot
    ///   - in the free list: next free slab index (`u32::MAX` = none)
    next: u32,
}

const NONE: u32 = u32::MAX;

/// number of hierarchical wheel levels. L0 is the finest; L(LEVELS-1)
/// the coarsest. With `BOTTOM_SLOTS = 256 = 2^8` and `LEVELS = 8` we
/// cover `256^8 = 2^64` ticks — the FULL `u64` deadline range. No
/// clamping, no overflow bucket: every representable deadline maps
/// exactly to one level + slot. Memory cost: 8 levels × 256 × 4 B =
/// 8 KiB per wheel. Tokio uses 6 levels of 64 slots (covering
/// `64^6 ≈ 6.87×10^10` ticks, ~2 years at ms); we go wider because
/// `Tick` is `u64` and we don't want any edge case where a long
/// deadline silently mis-fires.
const LEVELS: usize = 8;

/// pick the wheel level for an offset (`deadline - current_tick`).
///
/// Fast path: `offset < BOTTOM_SLOTS` → level 0.
///
/// General case: `level = floor(log_BOTTOM_SLOTS(offset))`, computed
/// branchlessly as `(bits - 1) / SHIFT_PER_LEVEL` where `bits = 64 -
/// offset.leading_zeros()` and `SHIFT_PER_LEVEL = log2(BOTTOM_SLOTS)`.
/// With `LEVELS = 8` and `BOTTOM_SLOTS = 256`, level is always in
/// `0..=7` and no clamp is needed (the entire u64 range is covered).
#[inline]
fn pick_level(offset: u64, shift_per_level: u32, bottom_slots: u64) -> usize {
    if offset < bottom_slots {
        return 0;
    }
    let bits = 64 - offset.leading_zeros() as usize;
    // bits-1 because we want the index of the top bit, not the count.
    // With LEVELS=8 and shift=8: max level = 63/8 = 7 = LEVELS-1. ✓
    let level = (bits - 1) / shift_per_level as usize;
    debug_assert!(
        level < LEVELS,
        "pick_level overflow — increase LEVELS or BOTTOM_SLOTS"
    );
    level.min(LEVELS - 1)
}

pub struct TimerWheel<C: Clock> {
    clock: C,
    current_tick: Tick,
    slab: Vec<Entry>,
    free_head: u32,
    /// `levels[k]` is the k-th wheel, indexed `0..BOTTOM_SLOTS`. Each
    /// slot holds the head of an intrusive linked list through
    /// `Entry::next`. `u32::MAX` = empty.
    levels: [Vec<u32>; LEVELS],
    /// number of live timers — registered minus fired/cancelled-and-
    /// reclaimed. Allows fast `is_empty()` / `next_deadline` short-
    /// circuit.
    live: usize,
    generation_counter: u32,
    /// `log2(BOTTOM_SLOTS)`. Cached so `pick_level` doesn't recompute.
    shift_per_level: u32,
    /// `BOTTOM_SLOTS` as a `Tick` (u64). Cached likewise.
    bottom_slots_t: Tick,
    /// `BOTTOM_SLOTS - 1` mask for slot indexing.
    slot_mask: Tick,
}

impl<C: Clock> TimerWheel<C> {
    #[must_use]
    pub fn new(clock: C) -> Self {
        debug_assert!(
            sized::TIMER_BOTTOM_SLOTS.is_power_of_two() && sized::TIMER_BOTTOM_SLOTS > 0,
            "TIMER_BOTTOM_SLOTS must be non-zero power of two"
        );
        let current_tick = clock.now();
        let bottom_slots = sized::TIMER_BOTTOM_SLOTS;
        let shift_per_level = bottom_slots.trailing_zeros();
        let make_level = || -> Vec<u32> { (0..bottom_slots).map(|_| NONE).collect() };
        // build LEVELS Vec<u32>s, each pre-filled with NONE.
        // can't use [make_level(); LEVELS] because Vec isn't Copy;
        // can't use const-init because LEVELS varies. Use array::from_fn.
        let levels: [Vec<u32>; LEVELS] = core::array::from_fn(|_| make_level());
        Self {
            clock,
            current_tick,
            // pre-reserve enough for one wheel revolution worth of
            // entries. avoids 6-8 Vec resizes on bench_timer's 10k
            // workload (each resize is a memcpy of all prior entries).
            slab: Vec::with_capacity(bottom_slots),
            free_head: NONE,
            levels,
            live: 0,
            generation_counter: 0,
            shift_per_level,
            bottom_slots_t: bottom_slots as Tick,
            slot_mask: (bottom_slots as Tick) - 1,
        }
    }

    pub fn now(&self) -> Tick {
        self.clock.now()
    }

    /// register a waker to fire at `deadline`. if `deadline <= current_tick`,
    /// the next `advance` (or the `advance` already in flight) fires it.
    pub fn register(&mut self, deadline: Tick, waker: Waker) -> TimerKey {
        self.generation_counter = self.generation_counter.wrapping_add(1);
        let generation = self.generation_counter;
        let slab_index = self.alloc_slot(Entry {
            deadline,
            waker: Some(waker),
            generation,
            next: NONE,
        });
        self.place(slab_index, deadline);
        self.live = self.live.saturating_add(1);
        TimerKey {
            slab_index,
            generation,
        }
    }

    /// cancel a registered timer. returns true if found-and-cancelled, false
    /// if already fired or never registered (stale generation).
    pub fn cancel(&mut self, key: TimerKey) -> bool {
        let index = key.slab_index as usize;
        let Some(entry) = self.slab.get_mut(index) else {
            return false;
        };
        if entry.generation != key.generation || entry.waker.is_none() {
            return false;
        }
        // lazy cancel: drop the waker; the entry is reclaimed on its scheduled
        // fire. avoids the O(slot-len) unlink walk.
        entry.waker = None;
        true
    }

    /// advance to `now`, firing all timers with `deadline <= now`. returns
    /// the number of timers actually woken (cancelled entries don't count).
    pub fn advance(&mut self, now: Tick) -> usize {
        if now <= self.current_tick {
            return 0;
        }
        let mut fired = 0usize;
        let bottom_slots = self.bottom_slots_t;
        let mask = self.slot_mask;
        let span = now.saturating_sub(self.current_tick);

        // Phase 1: drain L0 slots from (current_tick, now].
        if span >= bottom_slots {
            for slot_index in 0..sized::TIMER_BOTTOM_SLOTS {
                fired += self.drain_l0_slot(slot_index);
            }
        } else {
            let mut tick = self.current_tick.wrapping_add(1);
            while tick <= now {
                let slot_index = (tick & mask) as usize;
                fired += self.drain_l0_slot(slot_index);
                tick = tick.wrapping_add(1);
            }
        }

        let old_tick = self.current_tick;
        self.current_tick = now;

        // Phase 2: cascade higher levels down. For each level k >= 1,
        // any slot whose range now overlaps (or precedes) `now` is
        // drained and re-placed. Specifically: level k's slot s covers
        // ticks where `(t >> (k * shift_per_level)) & mask == s`, so
        // when current_tick crosses a multiple of `BOTTOM_SLOTS^k`, the
        // slot for that multiple flushes.
        for level in 1..LEVELS {
            let granularity_shift = level as u32 * self.shift_per_level;
            let granularity: Tick = 1u64 << granularity_shift;
            // ticks crossed at this level's granularity.
            let old_div = old_tick >> granularity_shift;
            let new_div = now >> granularity_shift;
            if old_div == new_div {
                continue;
            }
            let crossings = new_div - old_div;
            // If we crossed more than BOTTOM_SLOTS at this level, every
            // slot needs flushing; do a single full pass.
            if crossings >= bottom_slots {
                for slot in 0..sized::TIMER_BOTTOM_SLOTS {
                    fired += self.cascade_slot(level, slot, now);
                }
            } else {
                let mut div = old_div.wrapping_add(1);
                let _ = granularity; // not directly used; div math is enough
                while div <= new_div {
                    let slot = (div & mask) as usize;
                    fired += self.cascade_slot(level, slot, now);
                    div = div.wrapping_add(1);
                }
            }
        }

        fired
    }

    /// the next deadline that will fire, or None if no timers are registered.
    /// useful for the executor: park reactor for at most `next_deadline - now`.
    #[must_use]
    pub fn next_deadline(&self) -> Option<Tick> {
        if self.live == 0 {
            return None;
        }
        // Scan each level from L0 outward. For each level, scan slots
        // ahead of current position and walk the intrusive list to
        // find the minimum live (waker.is_some()) deadline.
        let mut best: Option<Tick> = None;
        for level in 0..LEVELS {
            let granularity_shift = level as u32 * self.shift_per_level;
            let level_start = self.current_tick >> granularity_shift;
            for offset in 0..sized::TIMER_BOTTOM_SLOTS {
                let div = level_start.wrapping_add(offset as Tick + 1);
                let slot = (div & self.slot_mask) as usize;
                let mut cursor = self.levels[level][slot];
                while cursor != NONE {
                    let entry = &self.slab[cursor as usize];
                    if entry.waker.is_some() && entry.deadline > self.current_tick {
                        best = Some(match best {
                            Some(current) => current.min(entry.deadline),
                            None => entry.deadline,
                        });
                    }
                    cursor = entry.next;
                }
                if best.is_some() {
                    return best;
                }
            }
        }
        best
    }

    fn alloc_slot(&mut self, entry: Entry) -> u32 {
        if self.free_head != NONE {
            let index = self.free_head;
            self.free_head = self.slab[index as usize].next;
            self.slab[index as usize] = entry;
            index
        } else {
            let raw = self.slab.len();
            assert!(raw < u32::MAX as usize, "timer slab capacity > u32::MAX");
            self.slab.push(entry);
            raw as u32
        }
    }

    fn release_slot(&mut self, index: u32) {
        self.slab[index as usize].waker = None;
        self.slab[index as usize].next = self.free_head;
        self.free_head = index;
        self.live = self.live.saturating_sub(1);
    }

    /// place `slab_index` (whose deadline is `deadline`) into the right
    /// wheel level + slot. O(1), branchless except for the past-deadline
    /// check.
    fn place(&mut self, slab_index: u32, deadline: Tick) {
        if deadline <= self.current_tick {
            // past deadline — fire on next advance. park in L0's next slot.
            let next_tick = self.current_tick.wrapping_add(1);
            let slot = (next_tick & self.slot_mask) as usize;
            self.push_l(0, slot, slab_index);
            return;
        }
        let offset = deadline - self.current_tick;
        let level = pick_level(offset, self.shift_per_level, self.bottom_slots_t);
        let granularity_shift = level as u32 * self.shift_per_level;
        let slot = ((deadline >> granularity_shift) & self.slot_mask) as usize;
        self.push_l(level, slot, slab_index);
    }

    #[inline]
    fn push_l(&mut self, level: usize, slot: usize, slab_index: u32) {
        let head = self.levels[level][slot];
        self.slab[slab_index as usize].next = head;
        self.levels[level][slot] = slab_index;
    }

    fn drain_l0_slot(&mut self, slot_index: usize) -> usize {
        let mut cursor = self.levels[0][slot_index];
        self.levels[0][slot_index] = NONE;
        let mut fired = 0;
        while cursor != NONE {
            let next = self.slab[cursor as usize].next;
            if let Some(waker) = self.slab[cursor as usize].waker.take() {
                waker.wake();
                fired += 1;
            }
            self.release_slot(cursor);
            cursor = next;
        }
        fired
    }

    /// cascade one slot from `level` (>= 1) into lower levels. Each
    /// entry is re-placed via `place()`, which picks the right level
    /// for its remaining offset (typically L0 if it's now imminent;
    /// possibly the same level if `crossings < BOTTOM_SLOTS`).
    /// Entries whose deadline <= `now` fire immediately.
    fn cascade_slot(&mut self, level: usize, slot: usize, now: Tick) -> usize {
        let mut cursor = self.levels[level][slot];
        self.levels[level][slot] = NONE;
        let mut fired = 0;
        while cursor != NONE {
            let next = self.slab[cursor as usize].next;
            let entry_deadline = self.slab[cursor as usize].deadline;
            if entry_deadline <= now {
                if let Some(waker) = self.slab[cursor as usize].waker.take() {
                    waker.wake();
                    fired += 1;
                }
                self.release_slot(cursor);
            } else {
                // re-place into the appropriate level for the now-smaller
                // offset.
                self.place(cursor, entry_deadline);
            }
            cursor = next;
        }
        fired
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::sync::Arc as StdArc;
    use alloc::task::Wake;
    use core::sync::atomic::{AtomicU64, Ordering};

    struct TestClock(StdArc<AtomicU64>);
    impl Clock for TestClock {
        fn now(&self) -> Tick {
            self.0.load(Ordering::Acquire)
        }
    }

    struct FireCounter(StdArc<AtomicU64>);
    impl Wake for FireCounter {
        fn wake(self: StdArc<Self>) {
            self.0.fetch_add(1, Ordering::AcqRel);
        }
    }

    fn fresh() -> (TimerWheel<TestClock>, StdArc<AtomicU64>, StdArc<AtomicU64>) {
        let clock_state = StdArc::new(AtomicU64::new(0));
        let fire_count = StdArc::new(AtomicU64::new(0));
        let wheel = TimerWheel::new(TestClock(clock_state.clone()));
        (wheel, clock_state, fire_count)
    }

    fn waker_for(fire_count: &StdArc<AtomicU64>) -> core::task::Waker {
        StdArc::new(FireCounter(fire_count.clone())).into()
    }

    #[test]
    fn register_then_advance_past_deadline_fires_waker() {
        let (mut wheel, _clock, fire_count) = fresh();
        let key = wheel.register(10, waker_for(&fire_count));
        assert_eq!(wheel.advance(5), 0, "before deadline");
        assert_eq!(fire_count.load(Ordering::Acquire), 0);
        assert_eq!(wheel.advance(10), 1, "exactly at deadline fires");
        assert_eq!(fire_count.load(Ordering::Acquire), 1);
        // second advance must not re-fire (entry was released).
        assert_eq!(wheel.advance(20), 0, "no second fire");
        assert!(!wheel.cancel(key), "cancel after fire returns false");
    }

    #[test]
    fn cancel_before_fire_prevents_wake() {
        let (mut wheel, _clock, fire_count) = fresh();
        let key = wheel.register(20, waker_for(&fire_count));
        assert!(wheel.cancel(key));
        assert_eq!(wheel.advance(50), 0);
        assert_eq!(fire_count.load(Ordering::Acquire), 0);
    }

    #[test]
    fn deadline_in_past_fires_on_next_advance() {
        let clock = StdArc::new(AtomicU64::new(100));
        let fire_count = StdArc::new(AtomicU64::new(0));
        let mut wheel = TimerWheel::new(TestClock(clock.clone()));
        wheel.register(50, waker_for(&fire_count)); // already past current_tick=100
        wheel.advance(101);
        assert_eq!(fire_count.load(Ordering::Acquire), 1);
    }

    #[test]
    fn far_future_entry_cascades_into_wheel_then_fires() {
        let (mut wheel, _clock, fire_count) = fresh();
        // BOTTOM_SLOTS=256, so register 1000 ticks ahead → L1.
        wheel.register(1000, waker_for(&fire_count));
        // advance close to it: cascade should pull it from L1 into L0.
        wheel.advance(800);
        assert_eq!(fire_count.load(Ordering::Acquire), 0, "not fired yet");
        wheel.advance(1000);
        assert_eq!(fire_count.load(Ordering::Acquire), 1, "fired at deadline");
    }

    #[test]
    fn next_deadline_returns_minimum() {
        let (mut wheel, _clock, fire_count) = fresh();
        wheel.register(50, waker_for(&fire_count));
        wheel.register(30, waker_for(&fire_count));
        wheel.register(70, waker_for(&fire_count));
        assert_eq!(wheel.next_deadline(), Some(30));
    }

    #[test]
    fn many_timers_same_slot_all_fire() {
        let (mut wheel, _clock, fire_count) = fresh();
        // 100 timers, all at deadline=5 (same bottom-wheel slot).
        for _ in 0..100 {
            wheel.register(5, waker_for(&fire_count));
        }
        let fired = wheel.advance(10);
        assert_eq!(fired, 100);
        assert_eq!(fire_count.load(Ordering::Acquire), 100);
    }

    #[test]
    fn cancel_after_fire_returns_false() {
        let (mut wheel, _clock, fire_count) = fresh();
        let key = wheel.register(5, waker_for(&fire_count));
        wheel.advance(10);
        assert!(!wheel.cancel(key), "cancel after fire returns false");
    }

    #[test]
    fn huge_advance_past_full_revolution_fires_everything() {
        let (mut wheel, _clock, fire_count) = fresh();
        // spread 5 timers across the wheel.
        for offset in [10, 50, 100, 200, 250] {
            wheel.register(offset, waker_for(&fire_count));
        }
        // jump beyond a full revolution.
        let fired = wheel.advance(10_000);
        assert_eq!(fired, 5);
        assert_eq!(fire_count.load(Ordering::Acquire), 5);
    }

    #[test]
    fn slab_reuse_after_fire_works() {
        let (mut wheel, _clock, fire_count) = fresh();
        for round in 0..10 {
            let base = round * 100;
            wheel.register(base + 50, waker_for(&fire_count));
            wheel.advance(base + 100);
        }
        assert_eq!(fire_count.load(Ordering::Acquire), 10);
        // slab should be reused, not grown to 10 entries.
        assert!(wheel.slab.len() <= 2);
    }

    #[test]
    fn pick_level_thresholds() {
        // BOTTOM_SLOTS = 256, shift = 8, LEVELS = 8.
        // Level k covers offsets [256^k, 256^(k+1)).
        // L0 = 0..256, L1 = 256..65536, L2 = 65536..16M, L3 = 16M..4.3G,
        // L4 = 4.3G..1.1T, L5 = 1.1T..281T, L6 = 281T..72e15, L7 = 72e15..u64::MAX.
        assert_eq!(pick_level(0, 8, 256), 0);
        assert_eq!(pick_level(1, 8, 256), 0);
        assert_eq!(pick_level(255, 8, 256), 0);
        assert_eq!(pick_level(256, 8, 256), 1);
        assert_eq!(pick_level(65535, 8, 256), 1);
        assert_eq!(pick_level(65536, 8, 256), 2);
        assert_eq!(pick_level(16777216, 8, 256), 3);
        assert_eq!(pick_level(4_294_967_296, 8, 256), 4);
        assert_eq!(pick_level(1_099_511_627_776, 8, 256), 5);
        assert_eq!(pick_level(281_474_976_710_656, 8, 256), 6);
        assert_eq!(pick_level(72_057_594_037_927_936, 8, 256), 7);
        // u64::MAX is in L7 (the top level), no clamping required.
        assert_eq!(pick_level(u64::MAX, 8, 256), 7);
    }

    #[test]
    fn very_far_future_eventually_fires_after_multi_level_cascade() {
        // Deadline 200_000 lives in L2 initially (BOTTOM_SLOTS^2 =
        // 65536, so 200_000 ∈ [65_536, 16_777_216) = L2).
        let (mut wheel, _clock, fire_count) = fresh();
        wheel.register(200_000, waker_for(&fire_count));
        // Advance step-by-step in chunks; cascading L2→L1→L0 must work.
        wheel.advance(100_000);
        assert_eq!(fire_count.load(Ordering::Acquire), 0, "not yet");
        wheel.advance(199_999);
        assert_eq!(fire_count.load(Ordering::Acquire), 0, "still not yet");
        wheel.advance(200_000);
        assert_eq!(fire_count.load(Ordering::Acquire), 1, "fired");
    }

    #[test]
    fn mixed_levels_register_and_drain() {
        let (mut wheel, _clock, fire_count) = fresh();
        // One timer in each of L0, L1, L2, L3.
        wheel.register(100, waker_for(&fire_count)); // L0
        wheel.register(1_000, waker_for(&fire_count)); // L1
        wheel.register(100_000, waker_for(&fire_count)); // L2
        wheel.register(20_000_000, waker_for(&fire_count)); // L3
        // Big advance fires all.
        let fired = wheel.advance(50_000_000);
        assert_eq!(fired, 4);
        assert_eq!(fire_count.load(Ordering::Acquire), 4);
    }

    #[test]
    fn next_deadline_with_only_far_level_works() {
        let (mut wheel, _clock, fire_count) = fresh();
        wheel.register(500_000, waker_for(&fire_count)); // L2
        assert_eq!(wheel.next_deadline(), Some(500_000));
    }

    #[test]
    fn very_high_register_count_100k_no_corruption() {
        // 100k timers spread across many slots and levels. Confirms
        // slab growth + intrusive lists handle high N without loss.
        const N: u64 = 100_000;
        let (mut wheel, _clock, fire_count) = fresh();
        for index in 0..N {
            // multiply by a prime to scatter across all 8 levels
            // (deadline range 1..~1.3e6).
            let deadline = index.wrapping_mul(13).wrapping_add(1);
            wheel.register(deadline, waker_for(&fire_count));
        }
        // huge advance past all deadlines.
        let horizon = N.wrapping_mul(13).wrapping_add(2);
        let fired = wheel.advance(horizon);
        assert_eq!(fired as u64, N, "every timer must fire exactly once");
        assert_eq!(fire_count.load(Ordering::Acquire), N);
    }

    #[test]
    fn deadline_in_level_7_far_future_eventually_fires() {
        // Deadline at 2^56 ticks: in L7 (top level). Verify the wheel
        // doesn't lose it and it fires when advance crosses past.
        let (mut wheel, _clock, fire_count) = fresh();
        let deadline = 1u64 << 56;
        wheel.register(deadline, waker_for(&fire_count));
        // Advance close — must cascade down through L6, L5, etc.
        wheel.advance(deadline - 1);
        assert_eq!(fire_count.load(Ordering::Acquire), 0, "still pending");
        wheel.advance(deadline);
        assert_eq!(fire_count.load(Ordering::Acquire), 1, "fired");
    }

    #[test]
    fn u64_max_deadline_fits_without_clamping() {
        // Just confirms a u64::MAX deadline registers cleanly.
        // We can't actually advance to u64::MAX in a test, but
        // register + cancel should work without panic.
        let (mut wheel, _clock, fire_count) = fresh();
        let key = wheel.register(u64::MAX, waker_for(&fire_count));
        assert!(wheel.cancel(key), "cancel before fire");
        assert_eq!(fire_count.load(Ordering::Acquire), 0);
    }
}
