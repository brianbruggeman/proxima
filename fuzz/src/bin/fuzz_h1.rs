//! Target 2a: `proxima-protocols::http1_codec` request-head parser --
//! `parse_head(buffer: &[u8]) -> Result<Status<'_>, ParseError>`
//! (`proxima-protocols/src/http1_codec/h1.rs:229`). Also sweeps truncated
//! prefixes of a well-formed head (the shape a partial-read socket buffer
//! hands the parser mid-stream) plus the response-head parser,
//! `parse_response_head` (`h1_client.rs:108`).
//!
//! Run: `cargo run --bin fuzz_h1`

use proxima_fuzz_harness::{run_no_panic_sweep, SweepReport};
use proxima_protocols::http1_codec::h1::parse_head;
use proxima_protocols::http1_codec::h1_client::parse_response_head;

const SEED: u64 = 0x1481_0000_1111_2222;
const ITERATIONS: usize = 150_000;
const MAX_LEN: usize = 512;

const WELL_FORMED_REQUEST: &[u8] =
    b"GET /foo/bar?baz=1 HTTP/1.1\r\nHost: example.com\r\nContent-Length: 4\r\n\r\nabcd";
const WELL_FORMED_RESPONSE: &[u8] =
    b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: keep-alive\r\n\r\nabcd";

fn main() {
    println!("fuzz_h1: proxima-protocols::http1_codec::{{parse_head,parse_response_head}} no-panic sweep");
    let reports = run(ITERATIONS);
    for report in &reports {
        report.print_line();
    }
}

fn run(iterations: usize) -> [SweepReport; 4] {
    [
        run_no_panic_sweep("h1::request_head::random", SEED, iterations, MAX_LEN, |bytes| {
            parse_head(bytes).is_ok()
        }),
        truncation_sweep(
            "h1::request_head::truncated",
            SEED.wrapping_add(1),
            WELL_FORMED_REQUEST,
            |bytes| parse_head(bytes).is_ok(),
        ),
        run_no_panic_sweep(
            "h1::response_head::random",
            SEED.wrapping_add(2),
            iterations,
            MAX_LEN,
            |bytes| parse_response_head(bytes).is_ok(),
        ),
        truncation_sweep(
            "h1::response_head::truncated",
            SEED.wrapping_add(3),
            WELL_FORMED_RESPONSE,
            |bytes| parse_response_head(bytes).is_ok(),
        ),
    ]
}

// every prefix length of a known-good head, 0..=full length -- the shape a
// partial socket read hands the parser mid-stream.
fn truncation_sweep(
    target: &'static str,
    seed: u64,
    well_formed: &[u8],
    mut feed: impl FnMut(&[u8]) -> bool,
) -> SweepReport {
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    for length in 0..=well_formed.len() {
        if feed(&well_formed[..length]) {
            accepted += 1;
        } else {
            rejected += 1;
        }
    }
    let iterations = well_formed.len() + 1;
    assert_eq!(accepted + rejected, iterations);
    SweepReport {
        target,
        seed,
        iterations,
        accepted,
        rejected,
    }
}

#[cfg(test)]
mod tests {
    use super::{run, WELL_FORMED_REQUEST};
    use proxima_protocols::http1_codec::h1::parse_head;

    #[test]
    fn no_panic_smoke() {
        let reports = run(3_000);
        assert_eq!(reports[0].iterations, 3_000);
        assert_eq!(reports[2].iterations, 3_000);
    }

    #[test]
    fn well_formed_head_parses() {
        assert!(parse_head(WELL_FORMED_REQUEST).is_ok());
    }
}
