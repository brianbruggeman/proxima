use core::sync::atomic::{AtomicU32, Ordering, fence};

/// A seqlock over `N` 32-bit words, updated and read as one atomic-seeming
/// unit. Internal building block for [`crate::coarse::TickCell`] (`N = 2`,
/// one `u64` tick count) and [`crate::anchor::AnchorCell`] (`N = 4`, a
/// `(ticks, unix_nanos)` pair that must never be observed half-updated).
///
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
/// **Single writer.** [`SeqWords::store`] is not safe to call concurrently
/// from more than one caller — the odd/even sequence bump is not itself
/// compare-and-swapped. This matches the shape both cells need: one
/// hardware-clock owner (or one core) writes; any number of readers, on
/// any core, read concurrently and lock-free.
struct SeqWords<const N: usize> {
    sequence: AtomicU32,
    words: [AtomicU32; N],
}

impl<const N: usize> SeqWords<N> {
    fn new(words: [u32; N]) -> Self {
        Self {
            sequence: AtomicU32::new(0),
            words: core::array::from_fn(|index| AtomicU32::new(words[index])),
        }
    }

    /// Replace the payload. See the struct doc's single-writer contract.
    fn store(&self, new_words: [u32; N]) {
        let sequence = self.sequence.load(Ordering::Relaxed);
        // odd sequence == "a write is in flight" — readers spin past it.
        self.sequence
            .store(sequence.wrapping_add(1), Ordering::Relaxed);
        // a `Release` STORE would order the accesses BEFORE it, which is the
        // wrong direction here: what must not move is the word stores BELOW
        // it. only a release FENCE pins them under the odd bump.
        fence(Ordering::Release);
        for (word, new_word) in self.words.iter().zip(new_words) {
            word.store(new_word, Ordering::Relaxed);
        }
        self.sequence
            .store(sequence.wrapping_add(2), Ordering::Release);
    }

    /// Read the payload as one atomic-seeming unit. Lock-free: never
    /// blocks, retries only if a writer's update overlapped this read.
    fn load(&self) -> [u32; N] {
        loop {
            let before = self.sequence.load(Ordering::Acquire);
            if before & 1 != 0 {
                core::hint::spin_loop();
                continue;
            }
            let mut words = [0u32; N];
            for (slot, word) in words.iter_mut().zip(&self.words) {
                *slot = word.load(Ordering::Relaxed);
            }
            // mirror of the writer's fence: an `Acquire` LOAD below the word
            // reads would not stop them sinking past it, so the validating
            // read would compare a sequence taken BEFORE the data it guards.
            fence(Ordering::Acquire);
            let after = self.sequence.load(Ordering::Relaxed);
            if before == after {
                return words;
            }
            core::hint::spin_loop();
        }
    }
}

/// Split a `u64` into big-endian-independent `[hi, lo]` `u32` halves for a
/// two-word [`SeqWords`] payload.
// intentional truncation: `lo` deliberately keeps only the low 32 bits —
// the high bits already live in `hi` via the shifted cast beside it.
#[allow(clippy::cast_possible_truncation)]
const fn split_u64(value: u64) -> [u32; 2] {
    let hi = (value >> 32) as u32;
    let lo = value as u32;
    [hi, lo]
}

/// Join `[hi, lo]` `u32` halves back into a `u64`.
const fn join_u64(words: [u32; 2]) -> u64 {
    (words[0] as u64) << 32 | words[1] as u64
}

/// Single `u64`, seqlock-protected across two `u32` words.
pub(crate) struct SeqU64 {
    inner: SeqWords<2>,
}

impl SeqU64 {
    pub(crate) fn new(value: u64) -> Self {
        Self {
            inner: SeqWords::new(split_u64(value)),
        }
    }

    pub(crate) fn store(&self, value: u64) {
        self.inner.store(split_u64(value));
    }

    pub(crate) fn load(&self) -> u64 {
        join_u64(self.inner.load())
    }
}

/// A `(u64, u64)` pair, seqlock-protected across four `u32` words so the
/// two halves are always observed together — never one fresh, one stale.
pub(crate) struct SeqU64Pair {
    inner: SeqWords<4>,
}

impl SeqU64Pair {
    pub(crate) fn new(first: u64, second: u64) -> Self {
        let [first_hi, first_lo] = split_u64(first);
        let [second_hi, second_lo] = split_u64(second);
        Self {
            inner: SeqWords::new([first_hi, first_lo, second_hi, second_lo]),
        }
    }

    pub(crate) fn store(&self, first: u64, second: u64) {
        let [first_hi, first_lo] = split_u64(first);
        let [second_hi, second_lo] = split_u64(second);
        self.inner.store([first_hi, first_lo, second_hi, second_lo]);
    }

    pub(crate) fn load(&self) -> (u64, u64) {
        let [first_hi, first_lo, second_hi, second_lo] = self.inner.load();
        (
            join_u64([first_hi, first_lo]),
            join_u64([second_hi, second_lo]),
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{SeqU64, SeqU64Pair};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    #[test]
    fn seq_u64_round_trips_values_spanning_both_halves() {
        let cell = SeqU64::new(0);

        cell.store(0xFFFF_FFFF_0000_0001);

        assert_eq!(cell.load(), 0xFFFF_FFFF_0000_0001);
    }

    #[test]
    fn seq_u64_pair_never_observes_a_torn_update() {
        // one core-frequency reader race, matching the anchor cell's real
        // shape: a writer re-anchors (ticks, unix_nanos) together while a
        // reader loop reads the pair concurrently; every observed pair must
        // be one the writer actually stored, never a hi/lo or first/second
        // mismatch.
        let cell = Arc::new(SeqU64Pair::new(0, 0));
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
                writer_cell.store(generation, generation.wrapping_mul(3));
            }
        });

        let reader_cell = Arc::clone(&cell);
        for _ in 0..500_000 {
            let (first, second) = reader_cell.load();
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
