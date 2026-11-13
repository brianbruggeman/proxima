//! A dumped node holds the value the node computed, even when a later node
//! reuses its arena slot before the command buffer completes.

#![cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use omega::{execute_plan_with_placements, plan, set_capture_step};
use proxima_numeric::{DType, NumericPolicy};
use proxima_tensor::{Extent, IndexMap, NodeId, Op, QuantizedBlock, ScalarOp, append, projection};

const LEFT: [f32; 8] = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0];
const RIGHT: [f32; 8] = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0];

fn input(program: &mut Vec<Op>, name: &str) -> NodeId {
    append(
        program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(LEFT.len() as u32)],
            name: Some(name.to_string()),
        },
    )
}

fn add(program: &mut Vec<Op>, lhs: NodeId, rhs: NodeId) -> NodeId {
    append(
        program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            operands: vec![
                (lhs, IndexMap::Affine(projection(1, &[0]))),
                (rhs, IndexMap::Affine(projection(1, &[0]))),
            ],
            name: None,
        },
    )
}

/// each node is read by the next two, so none fuses into its consumer and
/// the arena retires early nodes' slots to later ones.
fn fibonacci_chain() -> (Vec<Op>, Vec<NodeId>) {
    let mut program = Vec::new();
    let left = input(&mut program, "left");
    let right = input(&mut program, "right");
    let mut nodes = vec![add(&mut program, left, right)];
    let mut previous = right;
    for _ in 0..5 {
        let latest = *nodes.last().expect("chain has a head");
        nodes.push(add(&mut program, latest, previous));
        previous = latest;
    }
    (program, nodes)
}

fn output_binding_file(dir: &Path, node: u32) -> PathBuf {
    let prefix = format!("node{node}_step0_buf");
    let mut dumps: Vec<(usize, PathBuf)> = std::fs::read_dir(dir)
        .expect("dump dir is readable")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter_map(|path| {
            let name = path.file_name()?.to_str()?.to_string();
            let rest = name.strip_prefix(&prefix)?;
            let (index, _) = rest.split_once('_')?;
            Some((index.parse().ok()?, path))
        })
        .collect();
    dumps.sort();
    dumps.pop().expect("the captured node dumped a buffer").1
}

#[test]
fn captured_node_keeps_its_value_after_a_later_node_reuses_its_slot() {
    let dump_dir = tempfile::tempdir().expect("creates dump dir");
    let (program, nodes) = fibonacci_chain();
    let blocks = [
        ("left", QuantizedBlock::Float32(&LEFT)),
        ("right", QuantizedBlock::Float32(&RIGHT)),
    ];
    let positional = [blocks[0].1, blocks[1].1];
    let last = *nodes.last().expect("chain has a tail");
    let execution = plan(&program, &[], &positional, &[last], NumericPolicy::default())
        .expect("plans the add chain");
    let captured_node = nodes[0].0.to_string();

    temp_env::with_vars(
        [
            ("PROXIMA_CAPTURE_NODES", Some(captured_node.as_str())),
            ("PROXIMA_CAPTURE_DUMP_DIR", dump_dir.path().to_str()),
            ("PROXIMA_CAPTURE_STEPS", None),
            ("PROXIMA_CAPTURE_LIVE", None),
        ],
        || {
            set_capture_step(0);
            execute_plan_with_placements(&execution, &positional, &[], &[], &mut Vec::new())
                .expect("executes the add chain");
        },
    );

    let bytes = std::fs::read(output_binding_file(dump_dir.path(), nodes[0].0))
        .expect("dump file is readable");
    let captured: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .take(LEFT.len())
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    let expected: Vec<f32> = LEFT.iter().zip(&RIGHT).map(|(left, right)| left + right).collect();
    let differing = captured
        .iter()
        .zip(&expected)
        .filter(|(got, want)| got.to_bits() != want.to_bits())
        .count()
        + expected.len().saturating_sub(captured.len());
    assert_eq!(differing, 0, "captured {captured:?} expected {expected:?}");
}
