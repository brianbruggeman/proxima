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
    AxisTerm, DType, Extent, IndexMap, Keep, Op, Reduce, ReduceInit, ScalarOp, append, bind, infer,
    map,
};

fn elementwise_tanh_kernel() -> omega::Kernel {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(64)],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Tanh,
            operands: vec![(source, IndexMap::Affine(map::projection(1, &[0])))],
            name: None,
        },
    );
    let shapes = infer(&program, &[]).expect("elementwise infers");
    let nests = bind(&program, &shapes, &[]).expect("elementwise lowers");
    omega::emit(&nests[0], &std::collections::BTreeMap::new()).expect("elementwise emits")
}

/// `Erf` has no `metal_stdlib` counterpart (verified against the real
/// toolchain — see `msl.rs`'s `PROXIMA_ERF_FN` doc), so this is the one
/// kernel in this gate whose body is not a single `metal_stdlib` call: it
/// exercises the hand-rolled `proxima_erf` helper `preamble` always emits.
fn elementwise_erf_kernel() -> omega::Kernel {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(64)],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Erf,
            operands: vec![(source, IndexMap::Affine(map::projection(1, &[0])))],
            name: None,
        },
    );
    let shapes = infer(&program, &[]).expect("erf infers");
    let nests = bind(&program, &shapes, &[]).expect("erf lowers");
    omega::emit(&nests[0], &std::collections::BTreeMap::new()).expect("erf emits")
}

fn fused_matmul_kernel() -> omega::Kernel {
    let mut program = Vec::new();
    let lhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4), Extent::Static(3)],
            name: None,
        },
    );
    let rhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(3), Extent::Static(5)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
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
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("matmul".into()),
        }),
    );
    let shapes = infer(&program, &[]).expect("matmul infers");
    let nests = bind(&program, &shapes, &[]).expect("matmul lowers");
    omega::emit(&nests[0], &std::collections::BTreeMap::new()).expect("matmul emits")
}

/// `simdgroup_matrix`-tiled Q4_K GEMM (ROW 107) -- 16 tokens clears
/// `TILED_GEMM_MIN_TOKENS` (8), 4 weight rows is deliberately not a multiple
/// of `TILE_DIM` (8) so the boundary-tile mask is present in the emitted
/// source this gate hands to the real Metal compiler.
#[cfg(feature = "metal-tiled-gemm")]
fn tiled_gemm_q4k_kernel() -> omega::Kernel {
    let mut program = Vec::new();
    let lhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4), Extent::Static(256)],
            name: None,
        },
    );
    let rhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(256), Extent::Static(16)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
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
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            // token axis first, feature axis last -- see
            // `omega/src/msl.rs`'s `tiled_gemm_op` test fixture doc for why
            // this order is load-bearing for a packed weight's
            // `native_packed_layout` stride reconstruction.
            out_map: IndexMap::Affine(map::projection(3, &[1, 0])),
            keep: Keep::Reduce,
            name: Some("tiled_gemm".into()),
        }),
    );
    let shapes = infer(&program, &[]).expect("tiled gemm infers");
    let nests = bind(&program, &shapes, &[]).expect("tiled gemm lowers");
    let weight_node = nests[0].operands()[0].0;
    let mut q4k = std::collections::BTreeMap::new();
    q4k.insert(weight_node, omega::PackedCodec::Q4K);
    let kernel = omega::emit(&nests[0], &q4k).expect("tiled gemm emits");
    assert!(
        kernel.source.contains("simdgroup_multiply_accumulate"),
        "fixture must actually take the tiled GEMM path for this gate to mean anything:\n{}",
        kernel.source
    );
    kernel
}

fn cumsum_kernel() -> omega::Kernel {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(16)],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: source,
            in_map: IndexMap::Affine(map::projection(1, &[0])),
            out_map: IndexMap::Affine(map::projection(1, &[0])),
            keep: Keep::Scan,
            name: None,
        }),
    );
    let shapes = infer(&program, &[]).expect("cumsum infers");
    let nests = bind(&program, &shapes, &[]).expect("cumsum lowers");
    omega::emit(&nests[0], &std::collections::BTreeMap::new()).expect("cumsum emits")
}

/// `table[ids[s], d]`: a standalone elementwise gather.
fn embedding_lookup_kernel() -> omega::Kernel {
    let mut program = Vec::new();
    let table = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1000), Extent::Static(8)],
            name: None,
        },
    );
    let ids = append(
        &mut program,
        Op::Input {
            dtype: DType::Int32,
            shape: vec![Extent::Static(4)],
            name: None,
        },
    );
    let gathered_map = IndexMap::Computed {
        indices: ids,
        index_map: map::projection(2, &[0]),
        base: map::IndexPattern {
            iter_rank: 2,
            axes: vec![
                map::AxisIndex::default(),
                map::AxisIndex {
                    terms: vec![AxisTerm::projection(1)].into(),
                    offset: 0,
                },
            ],
        },
        gathered_dim: 0,
    };
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: vec![(table, gathered_map)],
            name: None,
        },
    );
    let shapes = infer(&program, &[]).expect("embedding lookup infers");
    let nests = bind(&program, &shapes, &[]).expect("embedding lookup lowers");
    omega::emit(&nests[0], &std::collections::BTreeMap::new()).expect("embedding lookup emits")
}

/// `sum_k table[ids[i], k] * weight[k, j]`: a gather fused into a reduction.
fn embedding_matmul_kernel() -> omega::Kernel {
    let mut program = Vec::new();
    let table = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1000), Extent::Static(6)],
            name: None,
        },
    );
    let ids = append(
        &mut program,
        Op::Input {
            dtype: DType::Int32,
            shape: vec![Extent::Static(4)],
            name: None,
        },
    );
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(6), Extent::Static(3)],
            name: None,
        },
    );
    let gather_map = IndexMap::Computed {
        indices: ids,
        index_map: map::projection(3, &[0]),
        base: map::IndexPattern {
            iter_rank: 3,
            axes: vec![
                map::AxisIndex::default(),
                map::AxisIndex {
                    terms: vec![AxisTerm::projection(2)].into(),
                    offset: 0,
                },
            ],
        },
        gathered_dim: 0,
    };
    let weight_map = IndexMap::Affine(map::projection(3, &[2, 1]));
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(table, gather_map), (weight, weight_map)],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("embedding_matmul".into()),
        }),
    );
    let shapes = infer(&program, &[]).expect("embedding matmul infers");
    let nests = bind(&program, &shapes, &[]).expect("embedding matmul lowers");
    omega::emit(&nests[0], &std::collections::BTreeMap::new()).expect("embedding matmul emits")
}

/// `Op::Iota` standing alone: a leaf with no operands, so this is the
/// smallest program that exercises `render_iota` — no elementwise consumer
/// is needed to prove the emitted kernel assembles under the real
/// toolchain.
fn iota_kernel() -> omega::Kernel {
    let mut program = Vec::new();
    append(
        &mut program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(16),
        },
    );
    let shapes = infer(&program, &[]).expect("iota infers");
    let nests = bind(&program, &shapes, &[]).expect("iota lowers");
    omega::emit(&nests[0], &std::collections::BTreeMap::new()).expect("iota emits")
}

#[test]
fn emitted_source_compiles_with_the_metal_toolchain() {
    #[allow(unused_mut)]
    let mut kernels = vec![
        elementwise_tanh_kernel(),
        elementwise_erf_kernel(),
        fused_matmul_kernel(),
        cumsum_kernel(),
        embedding_lookup_kernel(),
        embedding_matmul_kernel(),
        iota_kernel(),
    ];
    #[cfg(feature = "metal-tiled-gemm")]
    kernels.push(tiled_gemm_q4k_kernel());

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

    let expected = if cfg!(feature = "metal-tiled-gemm") { 8 } else { 7 };
    assert_eq!(
        compiled, expected,
        "compiled {compiled} kernels, expected exactly {expected} (including a gather, an iota, \
         the hand-rolled erf helper, and — with `metal-tiled-gemm` — the tiled GEMM kernel) — a \
         mismatch means a fixture silently stopped compiling into the gate"
    );
}

// a guard test used to live here asserting `cfg!(feature = "metal")`, because
// the parity suite is `#![cfg(feature = "metal")]` and compiled to zero cases
// without it — and zero cases reports the same "N passed, 0 skipped, exit 0"
// as full coverage. Moving the objc2 deps to a macOS target section let
// `metal` become a default feature instead, so the suite is simply always
// built here and the guard has nothing left to guard.
