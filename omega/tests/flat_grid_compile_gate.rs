//! Compile gate for the flat 2D grid form: every kernel form whose grid can
//! pass `u32::MAX` threads is emitted at a shape that does, checked to have
//! taken the flat form, and handed to the real Metal compiler. The flat form
//! rewrites each kernel's grid attributes, so a form that only parses in the
//! emitter's own tests (a scalar `thread_position_in_threadgroup` beside the
//! vector threadgroup position, a `uint3` batch index) fails here, not on a
//! user's first long prompt.
//!
//! A missing `xcrun`/`metal` fails the test, never skips it.

// test fixtures: expect() and unwrap() carry the failure message
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

fn packed_row_q4k_matvec(rows: u32) -> (BoundOp, BTreeMap<NodeId, Codec>) {
    let mut program = Vec::new();
    let weights = input(&mut program, DType::Float32, &[rows, 256]);
    let activation = input(&mut program, DType::Float32, &[1, 256]);
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weights, IndexMap::Affine(map::projection(3, &[1, 2]))),
                (activation, IndexMap::Affine(map::projection(3, &[0, 2]))),
            ],
            name: None,
        },
    );
    reduce(&mut program, ScalarOp::Add, product, 3, &[0, 1], Keep::Reduce);
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
            query_rows: 1,
            attention_rows,
            head_dim,
        },
    }
}

fn cached_attention(query_groups: u64, new_key_rows: u64) -> BoundOp {
    BoundOp {
        node: NodeId(9),
        dtype: DType::Float32,
        extents: vec![300_000_000, 1, 1, 4],
        kind: BoundOpKind::CachedAttention {
            operands: layouts(9),
            query_rows: 1,
            cached_key_rows: 0,
            new_key_rows,
            kv_heads: 1,
            query_groups,
            head_dim: 4,
            rotary_dim: 4,
            scale: 0.5,
            cached_lower_inclusive: i64::MIN,
            new_upper_inclusive: 0,
        },
    }
}

#[cfg(feature = "metal-moe-mul-mat-id")]
fn round_batched_gathered_matmul(round_count: u32) -> BoundOp {
    let mut program = Vec::new();
    let weight = input(&mut program, DType::Float32, &[8, 16, 256]);
    let route = input(&mut program, DType::Int32, &[4]);
    let activation = input(&mut program, DType::Float32, &[4, 256]);
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (
                    weight,
                    IndexMap::Computed {
                        indices: route,
                        index_map: map::projection(3, &[0]),
                        base: map::IndexPattern {
                            iter_rank: 3,
                            axes: vec![
                                map::AxisIndex::default(),
                                map::AxisIndex {
                                    terms: core::iter::once(map::AxisTerm::projection(1)).collect(),
                                    offset: 0,
                                    len: None,
                                },
                                map::AxisIndex {
                                    terms: core::iter::once(map::AxisTerm::projection(2)).collect(),
                                    offset: 0,
                                    len: None,
                                },
                            ],
                        },
                        gathered_dim: 0,
                    },
                ),
                (activation, IndexMap::Affine(map::projection(3, &[0, 2]))),
            ],
            name: None,
        },
    );
    reduce(&mut program, ScalarOp::Add, product, 3, &[0, 1], Keep::Reduce);
    let gathered = bound_of(&program);
    let BoundOp { node, dtype, extents, kind } = gathered;
    let BoundOpKind::Reduce {
        element_body,
        reduce_op,
        init,
        keep,
        operands,
        output_axes,
        out_layout,
        out_scatter,
        epilogue_body,
        epilogue_operands,
        epilogue_broadcast_axes,
    } = kind
    else {
        unreachable!("a gathered matmul binds to a reduce fold")
    };
    let route_node = operands
        .iter()
        .find_map(|(_, _, lookup)| lookup.as_ref().map(|lookup| lookup.indices))
        .expect("the gathered operand names its route");
    let mut wide_extents = extents;
    wide_extents[0] = 20_000_000;
    wide_extents[1] = 1_000;
    BoundOp {
        node,
        dtype,
        extents: wide_extents,
        kind: BoundOpKind::RoundBatchedReduce {
            element_body,
            reduce_op,
            init,
            keep,
            operands,
            output_axes,
            out_layout,
            out_scatter,
            epilogue_body,
            epilogue_operands,
            epilogue_broadcast_axes,
            round_count,
            round_routes: vec![route_node; round_count as usize],
            round_outputs: vec![node; round_count as usize],
        },
    }
}

fn flat_kernels() -> Vec<(&'static str, Kernel)> {
    let none = BTreeMap::new();
    let default_policy = NumericPolicy::default();
    let (projection, projection_codecs) = per_layer_projection();
    let (tiled, tiled_codecs) = tiled_q4k_gemm();
    let (packed_even, packed_even_codecs) = packed_row_q4k_matvec(4_000_000_000);
    let (packed_odd, packed_odd_codecs) = packed_row_q4k_matvec(4_000_000_004);
    let kernels = vec![
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
        (
            "packed row-blocked, groups divide the pinned width",
            emit_flat("packed row even", &packed_even, &packed_even_codecs, default_policy),
        ),
        (
            "packed row-blocked, odd group count falls back to one simdgroup",
            emit_flat("packed row odd", &packed_odd, &packed_odd_codecs, default_policy),
        ),
        (
            "cached attention, per-query-head grid",
            emit_flat("cached attention", &cached_attention(1, 8), &none, default_policy),
        ),
        (
            "cached attention, four query groups sharing a threadgroup",
            emit_flat("cached attention groups", &cached_attention(4, 1_000_000), &none, default_policy),
        ),
    ];
    #[cfg(feature = "metal-moe-mul-mat-id")]
    let kernels = {
        let mut kernels = kernels;
        kernels.push((
            "round-batched gathered reduce",
            emit_flat(
                "round-batched reduce",
                &round_batched_gathered_matmul(3),
                &none,
                default_policy,
            ),
        ));
        kernels
    };
    kernels
}

fn expected_kernel_count() -> usize {
    14 + usize::from(cfg!(feature = "metal-moe-mul-mat-id"))
}

#[test]
fn every_flat_form_kernel_compiles_with_the_metal_toolchain() {
    let kernels = temp_env::with_var("PROXIMA_TILED_GEMM_GRID2D", None::<&str>, flat_kernels);
    let expected = expected_kernel_count();
    assert_eq!(
        kernels.len(),
        expected,
        "one fixture per kernel form that can take the flat path"
    );

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
    assert_eq!(
        compiled, expected,
        "compiled {compiled} flat-form kernels, expected exactly {expected}"
    );
}

fn launch_width(label: &str, kernel: &Kernel) -> u64 {
    kernel
        .grid
        .grid2d
        .map(|spec| spec.threads_per_threadgroup_x)
        .unwrap_or_else(|| panic!("{label}: the flat form carries its launch"))
}

#[test]
fn the_packed_row_fixtures_take_the_launch_widths_their_labels_name() {
    let default_policy = NumericPolicy::default();
    let (even, even_codecs) = packed_row_q4k_matvec(4_000_000_000);
    let (odd, odd_codecs) = packed_row_q4k_matvec(4_000_000_004);

    let (even_kernel, odd_kernel) = temp_env::with_var("PROXIMA_TILED_GEMM_GRID2D", None::<&str>, || {
        (
            emit_flat("packed row even", &even, &even_codecs, default_policy),
            emit_flat("packed row odd", &odd, &odd_codecs, default_policy),
        )
    });

    let pinned_even = even_kernel
        .grid
        .threadgroup_width
        .expect("a packed row-blocked matvec pins a threadgroup width");
    let pinned_odd = odd_kernel
        .grid
        .threadgroup_width
        .expect("a packed row-blocked matvec pins a threadgroup width");
    // split-k owns the row-blocked geometry when it is compiled in: a fixture
    // this large is past its row ceiling, so the width is one simdgroup and
    // the divisible case is the one-simdgroup case
    #[cfg(not(feature = "metal-q4k-split-k"))]
    assert!(
        pinned_even > omega::sized::SIMD_WIDTH,
        "the even fixture only tests the divisible case when the pinned width exceeds one simdgroup, \
         got {pinned_even}"
    );
    #[cfg(feature = "metal-q4k-split-k")]
    assert_eq!(
        pinned_even,
        omega::sized::SIMD_WIDTH,
        "past split-k's row ceiling the pinned width is one simdgroup"
    );
    assert_eq!(even_kernel.grid.threads % pinned_even, 0, "even grid divides the pinned width");
    assert_eq!(launch_width("packed row even", &even_kernel), pinned_even);
    #[cfg(not(feature = "metal-q4k-split-k"))]
    {
        assert_ne!(odd_kernel.grid.threads % pinned_odd, 0, "odd grid must not divide the pinned width");
        assert_eq!(launch_width("packed row odd", &odd_kernel), omega::sized::SIMD_WIDTH);
    }
    #[cfg(feature = "metal-q4k-split-k")]
    assert_eq!(
        (pinned_odd, odd_kernel.grid.threads % omega::sized::SIMD_WIDTH),
        (omega::sized::SIMD_WIDTH, 0),
        "a one-simdgroup row-blocked grid is always a whole number of simdgroups, so the odd \
         fixture has no indivisible case under split-k"
    );
}
