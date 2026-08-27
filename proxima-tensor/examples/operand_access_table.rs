#![allow(clippy::expect_used)]
//! Prints a per-node operand-access table for a real spec: reads, bytes,
//! and distinct-vs-total element counts for every operand node the graph
//! actually reads from, attributed by `NodeId` via `instrument`'s
//! `operand_access_totals` API.
//!
//! The point is not the numbers this particular toy spec produces — every
//! weight in `transformer_block.toml` is dense and broadcasts only over the
//! sequence axis, so nothing here is "cold" in the sense a real decode loop
//! would show. The point is that the table is inspectable: a human can look
//! at `distinct/total` per row and see which operands are read in full
//! every call versus which ones only ever touch a slice — the same shape a
//! real weight-quantization decision would read off a real model's decode
//! trace.

use std::env;
use std::fs;

use proxima_tensor::instrument;
use proxima_tensor::spec::ProgramSpec;
use proxima_tensor::{NodeId, Op, evaluate, infer};

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
        .unwrap_or_else(|| "proxima-tensor/specs/transformer_block.toml".to_string());
    let sequence: usize = env::args()
        .nth(2)
        .map_or(4, |raw| raw.parse().expect("sequence must be an integer"));

    let text = fs::read_to_string(&path).expect("spec file is readable");
    let spec: ProgramSpec = toml::from_str(&text).expect("spec parses");
    let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");
    let symbols = [sequence as u64];
    let _shapes = infer(&program, &symbols).expect("program infers");

    let names: Vec<String> = spec.node.iter().map(|node| node.id().to_string()).collect();

    let model = 8usize;
    let inputs = [
        random_vec(1, sequence * model),
        vec![1.0 / model as f32; sequence],
        vec![1.0; sequence],
        random_vec(4, model * model),
        random_vec(5, model * model),
        random_vec(6, model * model),
        random_vec(7, model * model),
        random_vec(8, model * 16),
        random_vec(9, model * 16),
        random_vec(10, 16 * model),
    ];
    let borrowed: Vec<&[f32]> = inputs.iter().map(Vec::as_slice).collect();

    let root = NodeId(program.len() as u32 - 1);

    instrument::reset_operand_access();
    let evaluated = evaluate(&program, &symbols, &borrowed, &[root]).expect("program evaluates");
    let output = evaluated.root();
    let finite = output.iter().filter(|value| value.is_finite()).count();

    println!("spec={path} sequence={sequence} nodes={} output_elements={}", program.len(), output.len());
    assert_eq!(finite, output.len(), "every output element must be finite");

    let mut rows = instrument::operand_access_totals();
    rows.sort_by_key(|row| row.node);

    println!(
        "{:>4} {:<18} {:>10} {:>12} {:>14} {:>12} {:>8}",
        "node", "name", "reads", "bytes", "distinct_elem", "total_elem", "touched%"
    );
    for row in &rows {
        let name = names.get(row.node.0 as usize).map_or("?", String::as_str);
        let touched_percent = if row.access.total_elements == 0 {
            0.0
        } else {
            row.access.distinct_elements as f64 / row.access.total_elements as f64 * 100.0
        };
        println!(
            "{:>4} {:<18} {:>10} {:>12} {:>14} {:>12} {:>7.1}%",
            row.node.0,
            name,
            row.access.reads,
            row.access.bytes,
            row.access.distinct_elements,
            row.access.total_elements,
            touched_percent,
        );
    }
}
