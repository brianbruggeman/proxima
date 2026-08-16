//! The compile gate: every emitted kernel must actually assemble under the
//! real Metal toolchain, not just "look like" MSL. An integration test (not
//! a `#[cfg(test)]` unit test) so it always links std/tempfile regardless of
//! which feature set `-p omega`'s own unit tests are built with.
//!
//! If `xcrun`/`metal` is unavailable, this test FAILS with a clear message —
//! a missing toolchain is a red gate here, never a silent skip.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::process::Command;

use proxima_tensor::{
    DType, Expr, Extent, Fold, FoldInit, IndexMap, Keep, ScalarOp, append, infer, lower, map,
};

fn elementwise_tanh_kernel() -> omega::Kernel {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Expr::Block {
            dtype: DType::Float32,
            shape: vec![Extent::Static(64)],
            name: None,
        },
    );
    append(
        &mut program,
        Expr::Zip {
            dtype: DType::Float32,
            body: ScalarOp::Tanh,
            operands: vec![(source, IndexMap::Affine(map::projection(1, &[0])))],
            name: None,
        },
    );
    let shapes = infer(&program, &[]).expect("elementwise infers");
    let nests = lower(&program, &shapes, &[]).expect("elementwise lowers");
    omega::emit(&nests[0]).expect("elementwise emits")
}

fn fused_matmul_kernel() -> omega::Kernel {
    let mut program = Vec::new();
    let lhs = append(
        &mut program,
        Expr::Block {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4), Extent::Static(3)],
            name: None,
        },
    );
    let rhs = append(
        &mut program,
        Expr::Block {
            dtype: DType::Float32,
            shape: vec![Extent::Static(3), Extent::Static(5)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Expr::Zip {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    append(
        &mut program,
        Expr::Fold(Fold {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: FoldInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Last,
            name: Some("matmul".into()),
        }),
    );
    let shapes = infer(&program, &[]).expect("matmul infers");
    let nests = lower(&program, &shapes, &[]).expect("matmul lowers");
    omega::emit(&nests[0]).expect("matmul emits")
}

fn cumsum_kernel() -> omega::Kernel {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Expr::Block {
            dtype: DType::Float32,
            shape: vec![Extent::Static(16)],
            name: None,
        },
    );
    append(
        &mut program,
        Expr::Fold(Fold {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: FoldInit::Zero,
            operand: source,
            in_map: IndexMap::Affine(map::projection(1, &[0])),
            out_map: IndexMap::Affine(map::projection(1, &[0])),
            keep: Keep::All,
            name: None,
        }),
    );
    let shapes = infer(&program, &[]).expect("cumsum infers");
    let nests = lower(&program, &shapes, &[]).expect("cumsum lowers");
    omega::emit(&nests[0]).expect("cumsum emits")
}

#[test]
fn emitted_source_compiles_with_the_metal_toolchain() {
    let kernels = [
        elementwise_tanh_kernel(),
        fused_matmul_kernel(),
        cumsum_kernel(),
    ];

    let mut compiled = 0usize;
    for kernel in &kernels {
        let dir = tempfile::tempdir().expect("tempdir creation must not fail in ci");
        let metal_path = dir.path().join("kernel.metal");
        let air_path = dir.path().join("kernel.air");
        std::fs::write(&metal_path, &kernel.source).expect("write metal source to a temp file");

        let output = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-c"])
            .arg(&metal_path)
            .arg("-o")
            .arg(&air_path)
            .output();

        let output = match output {
            Ok(output) => output,
            Err(error) => panic!(
                "metal toolchain unavailable ({error}) — this is a red gate, not a skip; \
                 install/finish downloading the Metal toolchain and re-run"
            ),
        };

        assert!(
            output.status.success(),
            "metal compile failed for entry `{}`:\n--- source ---\n{}\n--- stderr ---\n{}",
            kernel.entry,
            kernel.source,
            String::from_utf8_lossy(&output.stderr)
        );
        compiled += 1;
    }

    assert!(
        compiled >= 2,
        "compiled {compiled} kernels — need at least 2 to prove this is not a vacuous pass"
    );
}
