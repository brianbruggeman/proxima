//! Target 5: `proxima-onnx::parse_complete` (protobuf `ModelProto` decode,
//! `proxima-onnx/src/pipe.rs:23`) chained into `lower_graph`
//! (`proxima-onnx/src/lower.rs:195`) -- the full lowering entry from raw,
//! attacker-controlled protobuf bytes to the tensor-op program. Arbitrary
//! and truncated bytes must error at either stage, never panic.
//!
//! Run: `cargo run --bin fuzz_onnx`

use proxima_fuzz_harness::{run_no_panic_sweep, SweepReport, Xorshift64};
use proxima_onnx::{lower_graph, parse_complete};

const SEED: u64 = 0x0057_0000_1111_2222;
const ITERATIONS: usize = 150_000;
const MAX_LEN: usize = 512;

fn main() {
    println!("fuzz_onnx: proxima-onnx::{{parse_complete,lower_graph}} no-panic sweep");
    let reports = run(ITERATIONS);
    for report in &reports {
        report.print_line();
    }
}

fn run(iterations: usize) -> [SweepReport; 2] {
    [
        run_no_panic_sweep("onnx::parse_complete::random", SEED, iterations, MAX_LEN, |bytes| {
            parse_complete(bytes).is_ok()
        }),
        parse_then_lower_sweep(iterations),
    ]
}

// when parse_complete accepts a random draw and it carries a graph, chain
// straight into lower_graph -- both stages of the lowering entry the
// production loader drives, on the same bytes.
fn parse_then_lower_sweep(iterations: usize) -> SweepReport {
    let mut generator = Xorshift64::new(SEED.wrapping_add(1));
    let mut accepted = 0usize;
    let mut rejected = 0usize;

    for _ in 0..iterations {
        let length = generator.next_below(MAX_LEN + 1);
        let input = generator.next_bytes(length);
        let outcome = match parse_complete(&input) {
            Ok(model) => match model.graph {
                Some(graph) => lower_graph(&graph).is_ok(),
                None => false,
            },
            Err(_) => false,
        };
        if outcome {
            accepted += 1;
        } else {
            rejected += 1;
        }
    }

    assert_eq!(accepted + rejected, iterations);
    SweepReport {
        target: "onnx::parse_complete::then_lower_graph",
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
