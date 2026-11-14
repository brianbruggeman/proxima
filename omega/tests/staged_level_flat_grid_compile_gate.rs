//! Compile gate for the staged level-table matvec in the flat 2D grid form. The flat form rewrites
//! the kernel's grid attributes (`wide_grid::widen_thread_index`), and the stage's own thread
//! attributes must follow: a scalar `thread_position_in_threadgroup` beside the vector threadgroup
//! position is rejected by the Metal compiler, not by the emitter's own text tests.
//!
//! A missing `xcrun`/`metal` fails the test, never skips it.

#![cfg(all(feature = "metal", target_os = "macos"))]
// test fixtures: expect() and unwrap() carry the failure message
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::process::Command;

use omega::msl::Grid2DForm;
use omega::{Codec, Kernel};
use proxima_numeric::{DType, NumericPolicy};
use proxima_tensor::{BoundOp, Extent, IndexMap, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp, append, bind, infer, map};

const ROWS_DIVIDING_THE_PINNED_WIDTH: u32 = 4_000_000_000;
const ROWS_LEAVING_A_PARTIAL_THREADGROUP: u32 = 4_000_000_004;

fn input(program: &mut Vec<Op>, shape: &[u32]) -> NodeId {
    append(
        program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: shape.iter().map(|extent| Extent::Static(*extent)).collect(),
            name: None,
        },
    )
}

fn level_matvec(rows: u32, codec: Codec) -> (BoundOp, BTreeMap<NodeId, Codec>) {
    let mut program = Vec::new();
    let weights = input(&mut program, &[rows, 256]);
    let activation = input(&mut program, &[1, 256]);
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weights, IndexMap::Affine(map::projection(3, &[1, 2]))),
                (activation, IndexMap::Affine(map::projection(3, &[0, 2]))),
            ],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    let shapes = infer(&program, &[]).expect("program infers");
    let terminal = NodeId((program.len() - 1) as u32);
    let bound = bind(&program, &shapes, &[terminal], NumericPolicy::default())
        .expect("program lowers")
        .into_iter()
        .next_back()
        .expect("one bound op");
    let weight = bound.operands()[0].0;
    (bound, BTreeMap::from([(weight, codec)]))
}

fn staged_flat_kernels() -> Vec<(String, Kernel)> {
    let cases = [
        ("iq3_s", Codec::Iq3S, "iq3_s_element_stage"),
        ("iq4_xs", Codec::Iq4Xs, "iq4_xs_element_stage"),
    ];
    let rows = [ROWS_DIVIDING_THE_PINNED_WIDTH, ROWS_LEAVING_A_PARTIAL_THREADGROUP];
    let mut kernels = Vec::new();
    for (name, codec, stage) in cases {
        for row_count in rows {
            let label = format!("{name} {row_count} rows");
            let (bound, packed) = level_matvec(row_count, codec);
            let kernel = omega::emit(&bound, &packed, NumericPolicy::default())
                .unwrap_or_else(|error| panic!("{label}: emit failed: {error}"));
            assert_eq!(
                kernel.grid.grid2d.map(|spec| spec.form),
                Some(Grid2DForm::FlatThreadgroupIndex),
                "{label}: the fixture must take the flat form"
            );
            assert!(kernel.source.contains(stage), "{label}: the fixture must stage its table:\n{}", kernel.source);
            assert!(
                kernel.source.contains("    uint stage_tid = wide_lane;\n"),
                "{label}: the flat form must rebuild stage_tid from the linear thread index:\n{}",
                kernel.source
            );
            kernels.push((label, kernel));
        }
    }
    kernels
}

#[test]
fn staged_level_matvecs_in_the_flat_grid_form_compile_with_the_metal_toolchain() {
    let kernels = temp_env::with_var("PROXIMA_TILED_GEMM_GRID2D", Some("0"), staged_flat_kernels);
    assert_eq!(kernels.len(), 4, "two level codecs at two row counts");

    let mut compiled = 0usize;
    for (label, kernel) in &kernels {
        let directory = tempfile::tempdir().expect("tempdir creation must not fail in ci");
        let metal_path = directory.path().join("kernel.metal");
        let air_path = directory.path().join("kernel.air");
        std::fs::write(&metal_path, &kernel.source).expect("write metal source to a temp file");

        let output = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-c"])
            .arg(&metal_path)
            .arg("-o")
            .arg(&air_path)
            .output()
            .unwrap_or_else(|error| panic!("metal toolchain unavailable ({error}) -- this is a red gate, not a skip"));

        assert!(
            output.status.success(),
            "{label}: metal compile failed for entry `{}`:\n--- source ---\n{}\n--- stderr ---\n{}",
            kernel.entry,
            kernel.source,
            String::from_utf8_lossy(&output.stderr)
        );
        compiled += 1;
    }
    assert_eq!(compiled, 4, "compiled {compiled} staged flat-form kernels, expected exactly 4");
}
