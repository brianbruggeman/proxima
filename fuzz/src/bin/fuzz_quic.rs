//! Target 3: `proxima-protocols::quic::packet::header` -- `parse_long`
//! (long-header packets, including the version=0 Version Negotiation path)
//! and `parse_short` (short-header/1-RTT packets), both
//! `proxima-protocols/src/quic/packet/header.rs`. Arbitrary bytes must
//! never panic on either path.
//!
//! Run: `cargo run --bin fuzz_quic`

use proxima_fuzz_harness::{run_no_panic_sweep, SweepReport, Xorshift64};
use proxima_protocols::quic::packet::header::{parse_long, parse_short, MAX_CID_LEN};

const SEED: u64 = 0x9c1c_0000_1111_2222;
const ITERATIONS: usize = 150_000;
const MAX_LEN: usize = 128;

fn main() {
    println!("fuzz_quic: proxima-protocols::quic::packet::header::{{parse_long,parse_short}} no-panic sweep");
    let reports = run(ITERATIONS);
    for report in &reports {
        report.print_line();
    }
}

fn run(iterations: usize) -> [SweepReport; 3] {
    [
        run_no_panic_sweep("quic::parse_long", SEED, iterations, MAX_LEN, |bytes| {
            parse_long(bytes).is_ok()
        }),
        short_header_sweep(iterations),
        version_negotiation_sweep(iterations),
    ]
}

// dcid_len is connection state the wire format does not encode -- exercise
// the full 0..=MAX_CID_LEN range against random input bytes.
fn short_header_sweep(iterations: usize) -> SweepReport {
    let mut input_generator = Xorshift64::new(SEED.wrapping_add(1));
    let mut dcid_generator = Xorshift64::new(SEED.wrapping_add(2));
    let mut accepted = 0usize;
    let mut rejected = 0usize;

    for _ in 0..iterations {
        let length = input_generator.next_below(MAX_LEN + 1);
        let input = input_generator.next_bytes(length);
        let dcid_len = dcid_generator.next_below(MAX_CID_LEN + 2);
        if parse_short(&input, dcid_len).is_ok() {
            accepted += 1;
        } else {
            rejected += 1;
        }
    }

    assert_eq!(accepted + rejected, iterations);
    SweepReport {
        target: "quic::parse_short",
        seed: SEED.wrapping_add(1),
        iterations,
        accepted,
        rejected,
    }
}

// force the version=0 branch (Version Negotiation) by fixing the first 5
// bytes (long-header form bit + all-zero version) and randomizing the rest.
fn version_negotiation_sweep(iterations: usize) -> SweepReport {
    let mut generator = Xorshift64::new(SEED.wrapping_add(3));
    let mut accepted = 0usize;
    let mut rejected = 0usize;

    for _ in 0..iterations {
        let tail_length = generator.next_below(MAX_LEN + 1);
        let mut input = vec![0x80u8, 0x00, 0x00, 0x00, 0x00];
        input.extend(generator.next_bytes(tail_length));
        if parse_long(&input).is_ok() {
            accepted += 1;
        } else {
            rejected += 1;
        }
    }

    assert_eq!(accepted + rejected, iterations);
    SweepReport {
        target: "quic::parse_long::version_negotiation",
        seed: SEED.wrapping_add(3),
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
