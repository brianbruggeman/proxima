//! Target 4: `proxima-gguf::parse_complete` -- the GGUF header + metadata
//! key/value table + tensor-info table decoder
//! (`proxima-gguf/src/pipe.rs:106`), the persistence boundary a model
//! loader hands an on-disk (or attacker-controlled) byte buffer to.
//! Arbitrary/truncated bytes must error, never panic.
//!
//! Run: `cargo run --bin fuzz_gguf`

use proxima_fuzz_harness::{run_no_panic_sweep, SweepReport, Xorshift64};
use proxima_gguf::parse_complete;

const SEED: u64 = 0x6666_0000_1111_2222;
const ITERATIONS: usize = 150_000;
const MAX_LEN: usize = 512;

// GGUF magic (b"GGUF") + version 3 -- puts random-tail draws past the
// cheap magic-byte rejection and into the metadata/tensor-table decode
// this target exists to stress.
const MAGIC_PREFIX: &[u8] = b"GGUF\x03\x00\x00\x00";

fn main() {
    println!("fuzz_gguf: proxima-gguf::parse_complete no-panic sweep");
    let reports = run(ITERATIONS);
    for report in &reports {
        report.print_line();
    }
}

fn run(iterations: usize) -> [SweepReport; 2] {
    [
        run_no_panic_sweep("gguf::parse_complete::random", SEED, iterations, MAX_LEN, |bytes| {
            parse_complete(bytes).is_ok()
        }),
        magic_prefixed_sweep(iterations),
    ]
}

fn magic_prefixed_sweep(iterations: usize) -> SweepReport {
    let mut generator = Xorshift64::new(SEED.wrapping_add(1));
    let mut accepted = 0usize;
    let mut rejected = 0usize;

    for _ in 0..iterations {
        let tail_length = generator.next_below(MAX_LEN + 1);
        let mut input = MAGIC_PREFIX.to_vec();
        input.extend(generator.next_bytes(tail_length));
        if parse_complete(&input).is_ok() {
            accepted += 1;
        } else {
            rejected += 1;
        }
    }

    assert_eq!(accepted + rejected, iterations);
    SweepReport {
        target: "gguf::parse_complete::magic_prefixed",
        seed: SEED.wrapping_add(1),
        iterations,
        accepted,
        rejected,
    }
}

#[cfg(test)]
mod tests {
    use super::run;

    #[test]
    fn no_panic_smoke() {
        for report in run(3_000) {
            assert_eq!(report.iterations, 3_000);
        }
    }
}
