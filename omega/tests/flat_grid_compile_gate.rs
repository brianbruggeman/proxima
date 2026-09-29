//! Compile gate for the flat 2D grid form: every kernel form whose grid can
//! pass `u32::MAX` threads is emitted at a shape that does, checked to have
//! taken the flat form, and handed to the real Metal compiler. The flat form
//! rewrites each kernel's grid attributes, so a form that only parses in the
//! emitter's own tests (a scalar `thread_position_in_threadgroup` beside the
//! vector threadgroup position, a `uint3` batch index) fails here, not on a
//! user's first long prompt.
//!
//! A missing `xcrun`/`metal` fails the test, never skips it.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::process::Command;

use omega::msl::Grid2DForm;
use omega::{Codec, Kernel};
use proxima_tensor::{
    BoundOp, BoundOpKind, DType, Extent, IndexMap, Keep, Layout, NodeId, NumericPolicy, Op,
    Reduce, ReduceInit, ScalarOp, append, bind, infer, map,
};

fn terminal(program: &[Op]) -> NodeId {
    NodeId((program.len() - 1) as u32)
}

fn input(program: &mut Vec<Op>, dtype: DType, shape: &[u32]) -> NodeId {
    append(
        program,
        Op::Input {
            dtype,
            shape: shape.iter().map(|extent| Extent::Static(*extent)).collect(),
            name: None,
        },
    )
}

fn reduce(
    program: &mut Vec<Op>,
    body: ScalarOp,
    operand: NodeId,
    rank: u16,
    out_axes: &[u16],
    keep: Keep,
) {
    let all_axes: Vec<u16> = (0..rank).collect();
    append(
        program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body,
            init: ReduceInit::Zero,
            operand,
            in_map: IndexMap::Affine(map::projection(rank, &all_axes)),
            out_map: IndexMap::Affine(map::projection(rank, out_axes)),
            keep,
            name: None,
        }),
    );
}

fn bound_of(program: &[Op]) -> BoundOp {
    let shapes = infer(program, &[]).expect("program infers");
    bind(program, &shapes, &[terminal(program)], NumericPolicy::default())
        .expect("program lowers")
        .into_iter()
        .next_back()
        .expect("one bound op")
}

fn emit_flat(label: &str, bound: &BoundOp, packed: &BTreeMap<NodeId, Codec>, policy: NumericPolicy) -> Kernel {
    let kernel = omega::emit(bound, packed, policy)
        .unwrap_or_else(|error| panic!("{label}: emit failed: {error}"));
    assert!(
        kernel.grid.threads > u64::from(u32::MAX),
        "{label}: the fixture must pass u32::MAX threads, got {}",
        kernel.grid.threads
    );
    assert_eq!(
        kernel.grid.grid2d.map(|spec| spec.form),
        Some(Grid2DForm::FlatThreadgroupIndex),
        "{label}: the fixture must take the flat form"
    );
    kernel
}

fn position_only(kind: BoundOpKind) -> BoundOp {
    BoundOp {
        node: NodeId(0),
        dtype: DType::Float32,
        extents: vec![70_000, 70_000],
        kind,
    }
}

fn layouts(count: u32) -> Vec<(NodeId, Layout, Option<proxima_tensor::Lookup>)> {
    (0..count)
        .map(|index| {
            (
                NodeId(index),
                Layout {
                    base: 0,
                    strides: vec![1].into(),
                },
                None,
            )
        })
        .collect()
}

fn elementwise_tanh_2d() -> BoundOp {
    let mut program = Vec::new();
    let source = input(&mut program, DType::Float32, &[70_000, 70_000]);
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Tanh,
            operands: vec![(source, IndexMap::Affine(map::projection(2, &[0, 1])))],
            name: None,
        },
    );
    bound_of(&program)
}

fn per_layer_projection() -> (BoundOp, BTreeMap<NodeId, Codec>) {
    let mut program = Vec::new();
    let weights = input(&mut program, DType::Float16, &[8960, 1536]);
    let activations = input(&mut program, DType::Float32, &[1873, 1536]);
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weights, IndexMap::Affine(map::projection(3, &[1, 2]))),
                (activations, IndexMap::Affine(map::projection(3, &[0, 2]))),
            ],
            name: None,
        },
    );
    reduce(&mut program, ScalarOp::Add, product, 3, &[0, 1], Keep::Reduce);
    let bound = bound_of(&program);
    let weight = bound.operands()[0].0;
    (bound, BTreeMap::from([(weight, Codec::Float16)]))
}

fn serial_maximum_over_a_pair() -> BoundOp {
    let mut program = Vec::new();
    let source = input(&mut program, DType::Float32, &[70_000, 70_000, 2]);
    reduce(&mut program, ScalarOp::Maximum, source, 3, &[0, 1], Keep::Reduce);
    bound_of(&program)
}

fn scan_over_lines() -> BoundOp {
    let mut program = Vec::new();
    let source = input(&mut program, DType::Float32, &[70_000, 70_000, 4]);
    reduce(&mut program, ScalarOp::Add, source, 3, &[0, 1, 2], Keep::Scan);
    bound_of(&program)
}

fn tiled_q4k_gemm() -> (BoundOp, BTreeMap<NodeId, Codec>) {
    let mut program = Vec::new();
    let lhs = input(&mut program, DType::Float32, &[300_000, 256]);
    let rhs = input(&mut program, DType::Float32, &[256, 4_000_000]);
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
    reduce(&mut program, ScalarOp::Add, product, 3, &[1, 0], Keep::Reduce);
    let bound = bound_of(&program);
    let weight = bound.operands()[0].0;
    (bound, BTreeMap::from([(weight, Codec::Q4K)]))
}

fn dense_batched_scores() -> BoundOp {
    let mut program = Vec::new();
    let weight = input(&mut program, DType::Float32, &[2_000_000, 8, 128]);
    let other = input(&mut program, DType::Float32, &[2_000_000, 8, 128]);
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(map::projection(4, &[3, 2, 1]))),
                (other, IndexMap::Affine(map::projection(4, &[0, 2, 1]))),
            ],
            name: None,
        },
    );
    reduce(&mut program, ScalarOp::Add, product, 4, &[0, 2, 3], Keep::Reduce);
    bound_of(&program)
}

fn gated_delta_net() -> BoundOp {
    BoundOp {
        node: NodeId(6),
        dtype: DType::Float32,
        extents: vec![1, 1 << 25, 128],
        kind: BoundOpKind::GatedDeltaNet {
            operands: layouts(6),
            n_tokens: 1,
            kv_heads: 1,
            num_v_heads: 1 << 25,
            head_k_dim: 4,
            head_v_dim: 128,
            query_key_head_stride: 4,
            query_key_dim_stride: 1,
            inv_sqrt_key_dim: 0.5,
            state_out: NodeId(7),
        },
    }
}

fn cached_softmax_weights() -> BoundOp {
    let attention_rows = 200_000_000u64;
    let head_dim = 64u64;
    let strides = [
        vec![attention_rows as i64, 1],
        vec![1],
        vec![head_dim as i64, 1],
    ];
    let operands = strides
        .into_iter()
        .enumerate()
        .map(|(index, stride)| {
            (
                NodeId(index as u32),
                Layout {
                    base: 0,
                    strides: stride.into(),
                },
                None,
            )
        })
        .collect();
    BoundOp {
        node: NodeId(3),
        dtype: DType::Float32,
        extents: vec![1024, attention_rows],
        kind: BoundOpKind::CachedSoftmaxWeights {
            operands,
            cached_weight_sum: NodeId(4),
            new_weight_sum: NodeId(5),
            new_attended: NodeId(6),
            cached_key_rows: 1024,
            new_key_rows: 1,
            query_rows: attention_rows,
            attention_rows,
            head_dim,
        },
    }
}

fn cached_attention() -> BoundOp {
    BoundOp {
        node: NodeId(9),
        dtype: DType::Float32,
        extents: vec![300_000_000, 1, 1, 4],
        kind: BoundOpKind::CachedAttention {
            operands: layouts(9),
            query_rows: 1,
            cached_key_rows: 0,
            new_key_rows: 8,
            kv_heads: 1,
            query_groups: 1,
            head_dim: 4,
            rotary_dim: 4,
            scale: 0.5,
            cached_lower_inclusive: i64::MIN,
            new_upper_inclusive: 0,
        },
    }
}

fn flat_kernels() -> Vec<(&'static str, Kernel)> {
    let none = BTreeMap::new();
    let default_policy = NumericPolicy::default();
    let (projection, projection_codecs) = per_layer_projection();
    let (tiled, tiled_codecs) = tiled_q4k_gemm();
    vec![
        ("iota", emit_flat("iota", &position_only(BoundOpKind::Iota), &none, default_policy)),
        (
            "constant",
            emit_flat("constant", &position_only(BoundOpKind::Constant { value: 1.5 }), &none, default_policy),
        ),
        ("elementwise", emit_flat("elementwise", &elementwise_tanh_2d(), &none, default_policy)),
        (
            "cooperative reduce",
            emit_flat("cooperative reduce", &projection, &projection_codecs, NumericPolicy::llama_relaxed()),
        ),
        (
            "serial reduce",
            emit_flat("serial reduce", &serial_maximum_over_a_pair(), &none, default_policy),
        ),
        ("scan", emit_flat("scan", &scan_over_lines(), &none, default_policy)),
        ("tiled gemm", emit_flat("tiled gemm", &tiled, &tiled_codecs, default_policy)),
        (
            "dense batched gemm",
            emit_flat("dense batched gemm", &dense_batched_scores(), &none, default_policy),
        ),
        ("gated delta net", emit_flat("gated delta net", &gated_delta_net(), &none, default_policy)),
        (
            "cached softmax weights",
            emit_flat("cached softmax weights", &cached_softmax_weights(), &none, default_policy),
        ),
        ("cached attention", emit_flat("cached attention", &cached_attention(), &none, default_policy)),
    ]
}

#[test]
fn every_flat_form_kernel_compiles_with_the_metal_toolchain() {
    let kernels = temp_env::with_var("PROXIMA_TILED_GEMM_GRID2D", None::<&str>, flat_kernels);
    assert_eq!(kernels.len(), 11, "one fixture per kernel form that can take the flat path");

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
            .unwrap_or_else(|error| {
                panic!(
                    "metal toolchain unavailable ({error}) -- this is a red gate, not a skip"
                )
            });

        assert!(
            output.status.success(),
            "{label}: metal compile failed for entry `{}`:\n--- source ---\n{}\n--- stderr ---\n{}",
            kernel.entry,
            kernel.source,
            String::from_utf8_lossy(&output.stderr)
        );
        compiled += 1;
    }
    assert_eq!(compiled, 11, "compiled {compiled} flat-form kernels, expected exactly 11");
}
