//! Shared deterministic-random generator and sweep bookkeeping for the
//! `fuzz_*` bins in this crate.
//!
//! Fallback path: no `cargo-fuzz` / nightly toolchain on this box (see
//! `fuzz/README.md`). Same shape as `examples/fuzz/main.rs` -- a fixed-seed
//! xorshift64* generator, not real randomness, so every run reproduces the
//! same byte-for-byte inputs and a finding is a finding you can re-hit.
//!
//! The contract every `fuzz_*` bin asserts, per target: fed arbitrary bytes,
//! the parser under test never panics and returns `Result`/`Option` on every
//! input. A real panic aborts the process before `main` returns, so the
//! sweep completing at all -- with every planned draw accounted for -- is
//! itself the no-panic proof (same reasoning as the `examples/fuzz`
//! precedent).

/// Fixed-seed xorshift64-star generator. Deterministic across runs.
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self { state }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn next_below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next_u64() as usize) % bound
    }

    #[must_use]
    pub fn next_bytes(&mut self, length: usize) -> Vec<u8> {
        (0..length).map(|_| (self.next_u64() & 0xff) as u8).collect()
    }
}

/// Outcome of one sweep: every draw is accounted for as either a successful
/// parse or a rejected one -- never a panic, which would have aborted the
/// process before this struct could be built.
#[derive(Debug, Clone, Copy)]
pub struct SweepReport {
    pub target: &'static str,
    pub seed: u64,
    pub iterations: usize,
    pub accepted: usize,
    pub rejected: usize,
}

impl SweepReport {
    pub fn print_line(&self) {
        println!(
            "fuzz: target={} seed={:#018x} iterations={} accepted={} rejected={} panics=0",
            self.target, self.seed, self.iterations, self.accepted, self.rejected
        );
    }
}

/// Runs `iterations` draws of length `0..=max_len` from a seeded generator
/// through `feed`, which returns `true` when the parser accepted the input.
/// `feed` panicking is the bug under test -- this function does not catch
/// it, it lets it abort, because that is the signal the contract failed.
pub fn run_no_panic_sweep(
    target: &'static str,
    seed: u64,
    iterations: usize,
    max_len: usize,
    mut feed: impl FnMut(&[u8]) -> bool,
) -> SweepReport {
    let mut generator = Xorshift64::new(seed);
    let mut accepted = 0usize;
    let mut rejected = 0usize;

    for _ in 0..iterations {
        let length = generator.next_below(max_len + 1);
        let input = generator.next_bytes(length);
        if feed(&input) {
            accepted += 1;
        } else {
            rejected += 1;
        }
    }

    assert_eq!(
        accepted + rejected,
        iterations,
        "every planned draw must have run -- a panic partway through aborts before this assert"
    );

    SweepReport {
        target,
        seed,
        iterations,
        accepted,
        rejected,
    }
}
