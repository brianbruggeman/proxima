//! Target 5: `proxima-onnx::parse_complete` (protobuf `ModelProto` decode,
//! `proxima-onnx/src/pipe.rs:23`) chained into `lower_graph`
//! (`proxima-onnx/src/lower.rs:195`) -- the full lowering entry from raw,
//! attacker-controlled protobuf bytes to the tensor-op program. Arbitrary
//! and truncated bytes must error at either stage, never panic.
//!
//! Uniform random bytes essentially never decode to a model with a graph
//! (the random sweep's `accepted` is the decodes; the chained sweep drew 0
//! lowerable graphs in 150000), so the chained sweeps mutate a serialized
//! seed model instead: every draw starts one to four byte edits away from a
//! model that parses, lowers and evaluates.
//!
//! Run: `cargo run --bin fuzz_onnx`

#[path = "fuzz_onnx/mutation.rs"]
mod mutation;
#[path = "fuzz_onnx/seed.rs"]
mod seed;

use proxima_fuzz_harness::{SweepReport, Xorshift64, run_no_panic_sweep};
use proxima_onnx::{lower_graph, parse_complete};

use mutation::mutate;
use seed::seed_model_bytes;

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
        run_no_panic_sweep(
            "onnx::parse_complete::random",
            SEED,
            iterations,
            MAX_LEN,
            |bytes| parse_complete(bytes).is_ok(),
        ),
        parse_then_lower_sweep(iterations),
    ]
}

fn parse_then_lower_sweep(iterations: usize) -> SweepReport {
    mutation_sweep(
        "onnx::parse_complete::then_lower_graph",
        SEED.wrapping_add(1),
        iterations,
        lowered_from_parsed_graph,
    )
}

fn lowered_from_parsed_graph(bytes: &[u8]) -> bool {
    parse_complete(bytes)
        .ok()
        .and_then(|model| model.graph)
        .is_some_and(|graph| lower_graph(&graph).is_ok())
}

fn mutation_sweep(
    target: &'static str,
    seed: u64,
    iterations: usize,
    mut feed: impl FnMut(&[u8]) -> bool,
) -> SweepReport {
    let base = seed_model_bytes();
    let mut generator = Xorshift64::new(seed);
    let accepted = (0..iterations)
        .filter(|_| feed(&mutate(&base, &mut generator)))
        .count();
    SweepReport {
        target,
        seed,
        iterations,
        accepted,
        rejected: iterations - accepted,
    }
}

#[cfg(test)]
mod tests {
    use proxima_fuzz_harness::run_no_panic_sweep;
    use proxima_onnx::lower_graph;

    use super::{
        MAX_LEN, SEED, lowered_from_parsed_graph, parse_complete, parse_then_lower_sweep, run,
        seed_model_bytes,
    };

    #[test]
    fn no_panic_smoke() {
        for report in run(3_000) {
            assert_eq!(report.iterations, 3_000);
        }
    }

    #[test]
    fn seed_model_parses_and_lowers_to_a_program() {
        let bytes = seed_model_bytes();

        let model = parse_complete(&bytes).expect("seed model bytes parse");
        let graph = model.graph.expect("seed model carries a graph");

        assert_eq!(graph.node.len(), 4);
        assert!(lower_graph(&graph).is_ok());
    }

    #[test]
    fn mutated_seed_draws_reach_lower_graph_and_some_still_lower() {
        let report = parse_then_lower_sweep(3_000);

        assert!(report.accepted > 0, "no mutated draw lowered");
        assert!(report.rejected > 0, "no mutated draw was rejected");
    }

    #[test]
    fn uniform_random_draws_never_lower_which_is_why_the_seed_exists() {
        let report = run_no_panic_sweep(
            "onnx::control::random_then_lower",
            SEED.wrapping_add(1),
            3_000,
            MAX_LEN,
            lowered_from_parsed_graph,
        );

        assert_eq!(report.accepted, 0);
    }
}
