//! Target 2b: `proxima-protocols::http2_codec::frame` -- `FrameHeader::parse`
//! (9-byte frame header) and `parse_payload` (typed payload decode), both
//! `proxima-protocols/src/http2_codec/frame.rs`. Feeds arbitrary bytes to
//! the header parse, then -- when a header decodes -- feeds arbitrary
//! payload bytes of exactly the declared length (and, separately,
//! truncated-short) to `parse_payload`. Neither call may panic.
//!
//! Run: `cargo run --bin fuzz_h2`

use bytes::Bytes;
use proxima_fuzz_harness::{run_no_panic_sweep, SweepReport, Xorshift64};
use proxima_protocols::http2_codec::frame::{parse_payload, FrameHeader};

const SEED: u64 = 0x2020_0000_1111_2222;
const ITERATIONS: usize = 150_000;
const HEADER_LEN: usize = 9;
const MAX_PAYLOAD_LEN: usize = 256;

fn main() {
    println!("fuzz_h2: proxima-protocols::http2_codec::frame::{{FrameHeader::parse,parse_payload}} no-panic sweep");
    let (header_report, payload_report) = run(ITERATIONS);
    header_report.print_line();
    payload_report.print_line();
}

fn run(iterations: usize) -> (SweepReport, SweepReport) {
    (
        run_no_panic_sweep("h2::frame_header", SEED, iterations, HEADER_LEN + 8, |bytes| {
            FrameHeader::parse(bytes).is_some()
        }),
        payload_sweep(iterations),
    )
}

// header bytes and payload bytes are drawn independently, then
// parse_payload is fed the header it was declared against plus a
// possibly-mismatched-length payload -- the shape a hostile peer sends.
fn payload_sweep(iterations: usize) -> SweepReport {
    let mut header_generator = Xorshift64::new(SEED.wrapping_add(1));
    let mut payload_generator = Xorshift64::new(SEED.wrapping_add(2));
    let mut accepted = 0usize;
    let mut rejected = 0usize;

    for _ in 0..iterations {
        let header_length = header_generator.next_below(HEADER_LEN + 8 + 1);
        let header_bytes = header_generator.next_bytes(header_length);
        let Some(header) = FrameHeader::parse(&header_bytes) else {
            rejected += 1;
            continue;
        };
        let payload_length = payload_generator.next_below(MAX_PAYLOAD_LEN + 1);
        let payload = Bytes::from(payload_generator.next_bytes(payload_length));
        match parse_payload(&header, &payload) {
            Ok(_) => accepted += 1,
            Err(_) => rejected += 1,
        }
    }

    assert_eq!(accepted + rejected, iterations);
    SweepReport {
        target: "h2::frame_payload",
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
        let (header_report, payload_report) = run(3_000);
        assert_eq!(header_report.iterations, 3_000);
        assert_eq!(payload_report.iterations, 3_000);
    }
}
