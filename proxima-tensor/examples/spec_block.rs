#![allow(clippy::expect_used)]
//! Runs a tensor program that exists only as a TOML file.
//!
//! The claim under test is "a new architecture is a config file, not a PR".
//! Parsing proves nothing on its own — a spec that parses and then fails to
//! infer, bind, or evaluate is still a PR waiting to happen. So this walks
//! the whole path (`toml` -> `ProgramSpec` -> `Vec<Op>` -> `infer` -> bind ->
//! `evaluate_parallel`) and asserts the output is finite and non-vacuous,
//! printing the element count it actually checked.

use std::env;
use std::fs;
use std::num::NonZeroUsize;

use proxima_tensor::spec::ProgramSpec;
use proxima_tensor::{NodeId, Op, evaluate_parallel, infer};

struct Lcg(u64);

impl Lcg {
    fn next_unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let bits = (self.0 >> 33) as u32;
        (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

fn main() {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "proxima-tensor/specs/attention_block.toml".to_string());
    let sequence: usize = env::args()
        .nth(2)
        .map_or(4, |raw| raw.parse().expect("sequence must be an integer"));

    let text = fs::read_to_string(&path).expect("spec file is readable");
    let spec: ProgramSpec = toml::from_str(&text).expect("spec parses");
    let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");
    let symbols = [sequence as u64];
    let _shapes = infer(&program, &symbols).expect("program infers");

    let model = 8usize;
    let inputs = [
        random_vec(1, sequence * model),
        vec![1.0 / model as f32; sequence],
        random_vec(3, model * model),
        random_vec(4, model * model),
        random_vec(5, model * model),
    ];
    let borrowed: Vec<&[f32]> = inputs.iter().map(Vec::as_slice).collect();

    let workers = NonZeroUsize::new(1).expect("one worker is nonzero");
    // a finite, non-empty result only proves the pipeline ran. the softmax
    // rows summing to one proves it computed attention, so the probability
    // node is requested as an extra output purely to be checked.
    let probabilities = spec
        .node
        .iter()
        .position(|node| node.id() == "probabilities")
        .expect("spec defines a probabilities node");
    let probabilities = NodeId(probabilities as u32);

    // naming any output at all narrows the retained set, so the root has to
    // be asked for explicitly once anything else is
    let root = NodeId(program.len() as u32 - 1);
    let evaluated =
        evaluate_parallel(&program, &symbols, &borrowed, &[root, probabilities], workers)
            .expect("program evaluates");

    let output = evaluated.root();
    assert!(!output.is_empty(), "a zero-element output is a vacuous pass");
    let finite = output.iter().filter(|value| value.is_finite()).count();
    assert_eq!(finite, output.len(), "every output element must be finite");

    let (rows, _) = evaluated.get(probabilities).expect("probabilities were requested");
    assert_eq!(rows.len(), sequence * sequence, "softmax is (sequence x sequence)");
    for (index, row) in rows.chunks_exact(sequence).enumerate() {
        let total: f32 = row.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "softmax row {index} sums to {total}, not 1.0 — this is not attention"
        );
    }

    println!(
        "spec={path} nodes={} sequence={sequence} elements={} finite={finite} checksum={:.6}",
        program.len(),
        output.len(),
        output.iter().sum::<f32>(),
    );
}
