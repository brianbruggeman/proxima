use core::sync::atomic::{AtomicU32, Ordering, fence};

/// `COUNT` `u64`s, updated and read as one atomic-seeming unit. Internal
/// building block for [`crate::coarse::TickCell`] (`COUNT = 1`, one tick
/// count) and [`crate::anchor::AnchorCell`] (`COUNT = 2`, a `(ticks,
/// unix_nanos)` pair that must never be observed half-updated).
///
/// Each `u64` is held as a `[hi, lo]` pair of 32-bit atomics because
/// `AtomicU64` does not exist on every target this crate must build for:
/// `thumbv7m-none-eabi` (Cortex-M3, ARMv7-M) reports `target_has_atomic =
/// "8,16,32,ptr"` — no native 64-bit compare-and-swap, so
/// `core::sync::atomic::AtomicU64` is simply absent from `core` for that
/// target (verified: `use core::sync::atomic::AtomicU64;` fails to resolve
/// under `--target thumbv7m-none-eabi`). `portable-atomic`'s fallback for
/// such targets requires either the `critical-section` feature (every
/// atomic op in the whole build, not just this cell's, pays a
/// critical-section cost once any crate in the dependency graph enables
/// it) or the `unsafe-assume-single-core` feature/cfg (a global,
/// unsafe, deployment-wide assumption a leaf library must not make on a
/// downstream integrator's behalf). A seqlock over native 32-bit atomics
/// avoids both: no new dependency, no unsafe assumption, and reads are
/// genuinely lock-free (never block, never take a critical section) —
/// only a rare concurrent writer costs a reader a retry.
///
/// # Contract
///
/// **Single writer.** [`SeqU64s::store`] is not safe to call concurrently
/// from more than one caller — the odd/even sequence bump is not itself
/// compare-and-swapped. This matches the shape both cells need: one
/// hardware-clock owner (or one core) writes; any number of readers, on
/// any core, read concurrently and lock-free.
pub(crate) struct SeqU64s<const COUNT: usize> {
    sequence: AtomicU32,
    halves: [[AtomicU32; 2]; COUNT],
}

impl<const COUNT: usize> SeqU64s<COUNT> {
    pub(crate) fn new(values: [u64; COUNT]) -> Self {
        Self {
            sequence: AtomicU32::new(0),
            halves: core::array::from_fn(|index| {
                let [high, low] = split(values[index]);
                [AtomicU32::new(high), AtomicU32::new(low)]
            }),
        }
    }

    /// Replace the payload. See the struct doc's single-writer contract.
    pub(crate) fn store(&self, values: [u64; COUNT]) {
        let sequence = self.sequence.load(Ordering::Relaxed);
        // odd sequence == "a write is in flight" — readers spin past it.
        self.sequence
            .store(sequence.wrapping_add(1), Ordering::Relaxed);
        // a `Release` STORE would order the accesses BEFORE it, which is the
        // wrong direction here: what must not move is the half stores BELOW
        // it. only a release FENCE pins them under the odd bump.
        fence(Ordering::Release);
        for (half, value) in self.halves.iter().zip(values) {
            let [high, low] = split(value);
            half[0].store(high, Ordering::Relaxed);
            half[1].store(low, Ordering::Relaxed);
        }
        self.sequence
            .store(sequence.wrapping_add(2), Ordering::Release);
    }

    /// Read the payload as one atomic-seeming unit. Lock-free: never
    /// blocks, retries only if a writer's update overlapped this read.
    pub(crate) fn load(&self) -> [u64; COUNT] {
        loop {
            let before = self.sequence.load(Ordering::Acquire);
            if before & 1 != 0 {
                core::hint::spin_loop();
                continue;
            }
            let values = core::array::from_fn(|index| {
                let half = &self.halves[index];
                join(
                    half[0].load(Ordering::Relaxed),
                    half[1].load(Ordering::Relaxed),
                )
            });
            // mirror of the writer's fence: an `Acquire` LOAD below the half
            // reads would not stop them sinking past it, so the validating
            // read would compare a sequence taken BEFORE the data it guards.
            fence(Ordering::Acquire);
            if before == self.sequence.load(Ordering::Relaxed) {
                return values;
            }
            core::hint::spin_loop();
        }
    }
}

/// Split a `u64` into `[high, low]` `u32` halves.
// intentional truncation: `low` deliberately keeps only the low 32 bits —
// the high bits already live in `high` via the shifted cast beside it.
#[allow(clippy::cast_possible_truncation)]
const fn split(value: u64) -> [u32; 2] {
    [(value >> 32) as u32, value as u32]
}

/// Join `high`/`low` `u32` halves back into a `u64`.
const fn join(high: u32, low: u32) -> u64 {
    (high as u64) << 32 | low as u64
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::SeqU64s;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    #[test]
    fn round_trips_a_value_spanning_both_halves() {
        let cell = SeqU64s::new([0]);

        cell.store([0xFFFF_FFFF_0000_0001]);

        assert_eq!(cell.load(), [0xFFFF_FFFF_0000_0001]);
    }

    #[test]
    fn round_trips_each_value_of_a_multi_value_payload_independently() {
        let cell = SeqU64s::new([0, 0]);

        cell.store([24_000_000, 1_753_500_000_000_000_000]);

        assert_eq!(cell.load(), [24_000_000, 1_753_500_000_000_000_000]);
    }

    #[test]
    fn never_observes_a_torn_update() {
        // one core-frequency reader race, matching the anchor cell's real
        // shape: a writer re-anchors (ticks, unix_nanos) together while a
        // reader loop reads the pair concurrently; every observed pair must
        // be one the writer actually stored, never a hi/lo or first/second
        // mismatch.
        let cell = Arc::new(SeqU64s::new([0, 0]));
        let stop = Arc::new(AtomicBool::new(false));

        let writer_cell = Arc::clone(&cell);
        let writer_stop = Arc::clone(&stop);
        let writer = thread::spawn(move || {
            for generation in 1..200_000u64 {
                if writer_stop.load(Ordering::Relaxed) {
                    break;
                }
                // both halves derived from the same generation counter, so a
                // reader can check `second == first * 3` to detect tearing.
                writer_cell.store([generation, generation.wrapping_mul(3)]);
            }
        });

        let reader_cell = Arc::clone(&cell);
        for _ in 0..500_000 {
            let [first, second] = reader_cell.load();
            assert_eq!(
                second,
                first.wrapping_mul(3),
                "reader observed a torn (first, second) pair: {first}, {second}"
            );
        }

        stop.store(true, Ordering::Relaxed);
        writer.join().expect("writer thread panicked");
    }
}
