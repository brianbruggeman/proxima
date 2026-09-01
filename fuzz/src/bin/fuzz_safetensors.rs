//! Target 6: `proxima-safetensors::parse_complete` -- the safetensors
//! header (JSON metadata length prefix + JSON tensor-descriptor map) load
//! boundary (`proxima-safetensors/src/pipe.rs:23`). Arbitrary bytes,
//! including a well-formed 8-byte little-endian length prefix pointing at
//! garbage JSON, must error, never panic.
//!
//! Run: `cargo run --bin fuzz_safetensors`

use proxima_fuzz_harness::{run_no_panic_sweep, SweepReport, Xorshift64};
use proxima_safetensors::parse_complete;

const SEED: u64 = 0x5AFE_0000_1111_2222;
const ITERATIONS: usize = 150_000;
const MAX_LEN: usize = 512;

fn main() {
    println!("fuzz_safetensors: proxima-safetensors::parse_complete no-panic sweep");
    let reports = run(ITERATIONS);
    for report in &reports {
        report.print_line();
    }
}

fn run(iterations: usize) -> [SweepReport; 2] {
    [
        run_no_panic_sweep(
            "safetensors::parse_complete::random",
            SEED,
            iterations,
            MAX_LEN,
            |bytes| parse_complete(bytes).is_ok(),
        ),
        length_prefixed_sweep(iterations),
    ]
}

// an honest 8-byte LE header-length prefix (the wire format's first field)
// followed by random bytes -- puts draws past the cheap length-field
// rejection and into the JSON header decode this target exists to stress.
fn length_prefixed_sweep(iterations: usize) -> SweepReport {
    let mut length_generator = Xorshift64::new(SEED.wrapping_add(1));
    let mut tail_generator = Xorshift64::new(SEED.wrapping_add(2));
    let mut accepted = 0usize;
    let mut rejected = 0usize;

    for _ in 0..iterations {
        let declared_header_len = length_generator.next_below(MAX_LEN + 1) as u64;
        let tail_length = tail_generator.next_below(MAX_LEN + 1);
        let mut input = declared_header_len.to_le_bytes().to_vec();
        input.extend(tail_generator.next_bytes(tail_length));
        if parse_complete(&input).is_ok() {
            accepted += 1;
        } else {
            rejected += 1;
        }
    }

    assert_eq!(accepted + rejected, iterations);
    SweepReport {
        target: "safetensors::parse_complete::length_prefixed",
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
