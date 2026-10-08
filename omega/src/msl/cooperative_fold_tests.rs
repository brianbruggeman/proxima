use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use proxima_tensor::{
    DType, Extent, IndexMap, Keep, Op, Reduce, ReduceInit, ScalarOp, append, bind, infer, map,
};

use super::*;

fn sum_of_squares_op(width: u32) -> BoundOp {
    let mut program = Vec::new();
    let row = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1), Extent::Static(width)],
            name: None,
        },
    );
    let squared = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (row, IndexMap::Affine(map::projection(2, &[0, 1]))),
                (row, IndexMap::Affine(map::projection(2, &[0, 1]))),
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
            operand: squared,
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[0])),
            keep: Keep::Reduce,
            name: Some("sum_of_squares".into()),
        }),
    );
    let root = NodeId((program.len() - 1) as u32);
    let shapes = infer(&program, &[]).expect("sum of squares infers");
    bind(&program, &shapes, &[root], NumericPolicy::default())
        .expect("sum of squares lowers")
        .into_iter()
        .next()
        .expect("one fused bound emitted")
}

#[test]
fn cooperative_fold_issues_every_slot_load_before_the_first_fold() {
    let source = emit(
        &sum_of_squares_op(1536),
        &BTreeMap::new(),
        NumericPolicy::llama_relaxed(),
    )
    .expect("hidden-width sum of squares emits")
    .source;
    let unroll = crate::sized::COOPERATIVE_REDUCE_UNROLL;
    let load_loop = source
        .find(&format!("float batch[{unroll}]["))
        .expect("loads land in a batch array");
    let fold = source
        .find("accumulator = seeded ? (accumulator + value) : value;")
        .expect("the fold still runs per slot");
    assert!(
        load_loop < fold,
        "every slot is loaded before slot 0 is folded"
    );
    assert!(source.contains("batch[slot][0] = in_range ? in0[walk0 + slot * advance0]"));
    assert!(source.contains(&format!("walk0 += {unroll} * advance0;")));
}

#[cfg(feature = "metal-wide-cooperative-reduce")]
#[test]
fn cooperative_fold_keeps_slot_order_so_the_sum_is_the_serial_sum() {
    let source = emit(
        &sum_of_squares_op(1536),
        &BTreeMap::new(),
        NumericPolicy::llama_relaxed(),
    )
    .expect("hidden-width sum of squares emits")
    .source;
    let fold_loop = source
        .find("if ((r + slot * 256) < total_r) {")
        .expect("the fold loop walks slots under the same range guard as the loads");
    assert!(
        source[fold_loop..].contains("scratch[0] = batch[slot][0];"),
        "each slot's operands come from its own batch row"
    );
    assert!(
        !source.contains("batch[slot + 1]"),
        "no slot is folded out of order"
    );
}

#[test]
fn bit_exact_policy_keeps_the_serial_fold_and_never_batches() {
    let source = emit(
        &sum_of_squares_op(1536),
        &BTreeMap::new(),
        NumericPolicy::bit_exact(),
    )
    .expect("hidden-width sum of squares emits")
    .source;
    assert!(
        !source.contains("batch["),
        "no cooperative fold means no batched loads"
    );
    assert!(source.contains("for (long r = 0; r < u.reduction_total; r++)"));
}

#[cfg(all(feature = "reduce-epilogue-fusion", feature = "metal-wide-cooperative-reduce"))]
mod normalization_rows {
    use super::*;

    fn rmsnorm_over_row(width: u32) -> BoundOp {
        let mut program = Vec::new();
        let full = || IndexMap::Affine(map::projection(2, &[0, 1]));
        let keep_row = || IndexMap::Affine(map::projection(1, &[0]));
        let scalar_row = || IndexMap::Affine(map::projection(1, &[]));
        let input = |program: &mut Vec<Op>, shape: Vec<Extent>, name: &str| {
            append(
                program,
                Op::Input {
                    dtype: DType::Float32,
                    shape,
                    name: Some(name.into()),
                },
            )
        };
        let row = input(&mut program, vec![Extent::Static(1), Extent::Static(width)], "x");
        let gamma = input(&mut program, vec![Extent::Static(width)], "gamma");
        let inverse_width = input(&mut program, Vec::new(), "inv_dim");
        let epsilon = input(&mut program, Vec::new(), "eps");
        let elementwise = |program: &mut Vec<Op>, body: ScalarOp, operands: Vec<(NodeId, IndexMap)>| {
            append(
                program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body,
                    operands,
                    name: None,
                },
            )
        };
        let squared = elementwise(&mut program, ScalarOp::Multiply, vec![(row, full()), (row, full())]);
        let sum_squares = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: squared,
                in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
                out_map: IndexMap::Affine(map::projection(2, &[0])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        let mean = elementwise(
            &mut program,
            ScalarOp::Multiply,
            vec![(sum_squares, keep_row()), (inverse_width, scalar_row())],
        );
        let shifted = elementwise(&mut program, ScalarOp::Add, vec![(mean, keep_row()), (epsilon, scalar_row())]);
        let root_mean_square = elementwise(&mut program, ScalarOp::SquareRoot, vec![(shifted, keep_row())]);
        let inverse = elementwise(&mut program, ScalarOp::Reciprocal, vec![(root_mean_square, keep_row())]);
        let broadcast_row = IndexMap::Affine(map::projection(2, &[0]));
        let normed = elementwise(&mut program, ScalarOp::Multiply, vec![(row, full()), (inverse, broadcast_row)]);
        let broadcast_gamma = IndexMap::Affine(map::projection(2, &[1]));
        let scaled = elementwise(&mut program, ScalarOp::Multiply, vec![(normed, full()), (gamma, broadcast_gamma)]);
        let shapes = infer(&program, &[]).expect("rmsnorm infers");
        bind(&program, &shapes, &[scaled], NumericPolicy::default())
            .expect("rmsnorm lowers")
            .into_iter()
            .find(reduce_has_broadcast_epilogue)
            .expect("the tail folds into the reduce as a broadcast epilogue")
    }

    fn emitted(width: u32) -> String {
        emit(&rmsnorm_over_row(width), &BTreeMap::new(), NumericPolicy::default())
            .expect("normalization row emits")
            .source
    }

    fn dispatch_width(width: u32) -> u64 {
        tiled_gemm_threadgroup_width(&rmsnorm_over_row(width), &[], NumericPolicy::default())
            .expect("a cooperative reduce reports its threadgroup width")
    }

    const CAPTURED_HIDDEN_WIDTH: u64 = 1536;
    const CAPTURED_NARROW_ROW: u64 = 512;

    #[test]
    fn hidden_width_norm_class_takes_its_own_width_key() {
        let expected = crate::sized::HIDDEN_NORM_REDUCE_WIDTH;
        assert_eq!(dispatch_width(1536), expected);
        assert!(emitted(1536).contains(&format!("lane = gid % {expected}u;")));
    }

    #[test]
    fn selector_returns_512_for_the_captured_one_row_shapes() {
        assert_eq!(broadcast_width_for(CAPTURED_HIDDEN_WIDTH, 1, 256, 512), 512);
        assert_eq!(broadcast_width_for(CAPTURED_NARROW_ROW, 1, 256, 512), 512);
    }

    #[test]
    fn selector_keeps_256_for_a_wide_reduce_and_for_many_rows() {
        assert_eq!(broadcast_width_for(4096, 8, 256, 512), 256);
        assert_eq!(broadcast_width_for(CAPTURED_HIDDEN_WIDTH, 35, 256, 512), 256);
        assert_eq!(broadcast_width_for(1024, 1, 256, 256), 256);
    }

    #[test]
    fn selector_with_the_key_at_256_reproduces_the_one_lane_per_four_widths() {
        for reduction_total in [128, 256, 512, 1024, 1536, 4096] {
            for rows in [1, 8] {
                assert_eq!(
                    broadcast_width_for(reduction_total, rows, 256, 256),
                    quarter_width(reduction_total, 256),
                    "reduction {reduction_total}, rows {rows}"
                );
            }
        }
    }

    #[test]
    fn selector_leaves_a_one_row_reduce_shorter_than_the_key_on_the_quarter_rule() {
        assert_eq!(broadcast_width_for(100, 1, 256, 512), 32);
        assert_eq!(broadcast_width_for(256, 1, 256, 512), 64);
        assert_eq!(broadcast_width_for(4096, 1, 256, 512), 512);
    }

    #[test]
    fn hidden_norm_default_in_the_sizing_toml_is_512() {
        let toml = include_str!("../../omega-runtime.toml");
        assert!(toml.lines().any(|line| line == "hidden_norm_width = 512"));
    }

    #[test]
    fn hidden_width_kernel_renders_the_512_lane_body() {
        if crate::sized::HIDDEN_NORM_REDUCE_WIDTH != 512 {
            return;
        }
        let source = emitted(1536);
        assert!(source.contains("lane = gid % 512u;"), "{source}");
        assert!(source.contains("partials[16]"), "{source}");
        assert!(source.contains("(lane % 32u) < 16u"), "{source}");
    }

    #[test]
    fn row_wider_than_the_cap_is_held_to_the_cap() {
        let widest = crate::sized::BROADCAST_REDUCE_MAX_WIDTH.max(crate::sized::HIDDEN_NORM_REDUCE_WIDTH);
        assert_eq!(dispatch_width(4096), widest);
    }

    #[test]
    fn row_of_one_simdgroup_keeps_the_single_simd_fold() {
        let source = emitted(128);
        assert_eq!(dispatch_width(128), 32);
        assert!(!source.contains("partials["), "one simdgroup has no partials to combine:\n{source}");
        assert!(!source.contains("threadgroup_barrier"));
    }

    #[test]
    fn broadcast_fold_combines_partials_with_a_second_simd_pass_behind_one_barrier() {
        let source = emitted(1536);
        let simdgroups = dispatch_width(1536) / 32;
        if crate::sized::COOPERATIVE_REDUCE_BROADCAST_SIMD_FOLD == 0 {
            assert!(source.contains("partials[0] = reduced;"));
            assert_eq!(source.matches("threadgroup_barrier(").count(), 2);
            return;
        }
        assert_eq!(source.matches("threadgroup_barrier(").count(), 1, "{source}");
        assert!(
            source.contains(&format!(
                "float reduced = simd_sum((lane % 32u) < {simdgroups}u ? partials[lane % 32u] : (float)0.0f);"
            )),
            "{source}"
        );
        assert!(!source.contains("partials[0] = reduced;"));
    }

    #[test]
    fn second_simd_pass_reads_partials_only_after_the_barrier_that_publishes_them() {
        if crate::sized::COOPERATIVE_REDUCE_BROADCAST_SIMD_FOLD == 0 {
            return;
        }
        let source = emitted(1536);
        let publish = source.find("partials[lane / 32u] = partial;").expect("lane 0 of each simdgroup publishes");
        let barrier = source.find("threadgroup_barrier(").expect("a barrier orders the publish");
        let combine = source.find("partials[lane % 32u]").expect("every lane reads one partial");
        assert!(publish < barrier && barrier < combine, "{source}");
    }
}
