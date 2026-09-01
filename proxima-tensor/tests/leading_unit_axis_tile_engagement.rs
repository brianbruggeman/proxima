//! `docs/discipline.md` ROW 196/197: `neon_tile_plan` and `width_tile_plan`
//! both gated on `leading_output_axes.len() == 1`, so a matmul whose graph
//! keeps a size-1 batch axis as its own leading output axis (BGE's real
//! `MatMul([1,seq,384], [384,N])` shape, never flattened into the token axis
//! by `lower_matmul`) never fired either AArch64 SIMD tile — the scalar
//! fallback ran instead, at ~1/5 the throughput.
//!
//! This test proves the class fix (`resolve_reduce_axis_shape` eliding
//! extent-1 leading axes before either gate sees them, `cpu.rs`) two ways:
//! (1) bit-identical output — same bytes, not just within tolerance — between
//! the batched (`[1,M,K]`) and unbatched (`[M,K]`) programs over identical
//! data, since both should now walk the exact same tile kernel in the exact
//! same order; (2) the width tile's own invocation counter proves the tile
//! ACTUALLY fired for the batched call, not merely that the fallback path
//! happened to produce the right answer (the same silent-fallthrough trap
//! `neon_tile_full_output.rs::check_width_size` already guards against).

use proxima_tensor::{Extent, IndexMap, NodeId, Op, ReduceInit, ScalarOp, append, evaluate, map};

/// `neon_tile_full_output.rs::matmul_program`, unbatched: `[M,K] x [K,N]`,
/// plain (non-transposed) RHS — the layout `width_tile_plan` targets.
fn matmul_program_unbatched(m: u32, k: u32, n: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs = append(&mut program, Op::Input { dtype: proxima_tensor::DType::Float32, shape: vec![Extent::Static(m), Extent::Static(k)], name: None });
    let rhs = append(&mut program, Op::Input { dtype: proxima_tensor::DType::Float32, shape: vec![Extent::Static(k), Extent::Static(n)], name: None });
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: proxima_tensor::DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(lhs, IndexMap::Affine(map::projection(3, &[0, 2]))), (rhs, IndexMap::Affine(map::projection(3, &[2, 1])))],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(proxima_tensor::Reduce {
            dtype: proxima_tensor::DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: proxima_tensor::Keep::Reduce,
            name: Some("matmul_unbatched".into()),
        }),
    );
    (program, sum)
}

/// The BGE-shaped twin: `[1,M,K] x [K,N] -> [1,M,N]`, a size-1 batch axis
/// kept as its own leading output axis (rank 4: batch, m, n, k — indices
/// 0,1,2,3) rather than flattened into `m` — exactly what `lower_matmul`
/// (`proxima-onnx/src/lower.rs:743-826`) produces for BGE's real graph.
fn matmul_program_batched(m: u32, k: u32, n: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs = append(&mut program, Op::Input { dtype: proxima_tensor::DType::Float32, shape: vec![Extent::Static(1), Extent::Static(m), Extent::Static(k)], name: None });
    let rhs = append(&mut program, Op::Input { dtype: proxima_tensor::DType::Float32, shape: vec![Extent::Static(k), Extent::Static(n)], name: None });
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: proxima_tensor::DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(lhs, IndexMap::Affine(map::projection(4, &[0, 1, 3]))), (rhs, IndexMap::Affine(map::projection(4, &[3, 2])))],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(proxima_tensor::Reduce {
            dtype: proxima_tensor::DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(4, &[0, 1, 2, 3])),
            out_map: IndexMap::Affine(map::projection(4, &[0, 1, 2])),
            keep: proxima_tensor::Keep::Reduce,
            name: Some("matmul_batched".into()),
        }),
    );
    (program, sum)
}

/// `M=8, K=64, N=64` — both `K` and `N` are multiples of
/// `WIDTH_TILE_VECS*4=16` (`cpu.rs`'s own doc), so the width tile covers the
/// whole shape with no column tail, isolating the leading-axis gate as the
/// only variable between the batched and unbatched arms.
#[test]
fn leading_unit_axis_matches_unbatched_bit_identical() {
    let (m, k, n) = (8u32, 64u32, 64u32);
    let lhs: Vec<f32> = (0..(m * k) as usize).map(|index| (index as f32 * 0.0137).sin()).collect();
    let rhs: Vec<f32> = (0..(k * n) as usize).map(|index| (index as f32 * 0.0271).cos()).collect();

    let (unbatched_program, _) = matmul_program_unbatched(m, k, n);
    let unbatched = match evaluate(&unbatched_program, &[], &[&lhs, &rhs], &[]) {
        Ok(evaluated) => evaluated,
        Err(error) => panic!("unbatched gemm evaluates: {error}"),
    };

    let (batched_program, _) = matmul_program_batched(m, k, n);
    let batched = match evaluate(&batched_program, &[], &[&lhs, &rhs], &[]) {
        Ok(evaluated) => evaluated,
        Err(error) => panic!("batched (size-1 leading axis) gemm evaluates: {error}"),
    };

    let unbatched_root = unbatched.root();
    let batched_root = batched.root();
    assert_eq!(batched_root.len(), unbatched_root.len(), "batched/unbatched output length mismatch");
    assert_eq!(
        batched_root.to_vec(),
        unbatched_root.to_vec(),
        "batched (size-1 leading axis) output diverged from the unbatched output bit-for-bit -- \
         eliding the unit axis before the tile gate must produce the identical fold order, not \
         merely a numerically-close answer"
    );
}

/// Engagement proof: the width tile's own invocation counter must show a
/// nonzero delta for the BATCHED call specifically -- without this, the bit
/// identity above could pass purely because BOTH arms fell through to the
/// same scalar generic loop, which proves nothing about the gate fix.
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
#[test]
fn leading_unit_axis_fires_the_width_tile() {
    let (m, k, n) = (8u32, 64u32, 64u32);
    let lhs: Vec<f32> = (0..(m * k) as usize).map(|index| (index as f32 * 0.0137).sin()).collect();
    let rhs: Vec<f32> = (0..(k * n) as usize).map(|index| (index as f32 * 0.0271).cos()).collect();

    let (gate_before, invocations_before, _) = proxima_tensor::cpu::width_tile_counters();

    let (batched_program, _) = matmul_program_batched(m, k, n);
    let batched = match evaluate(&batched_program, &[], &[&lhs, &rhs], &[]) {
        Ok(evaluated) => evaluated,
        Err(error) => panic!("batched (size-1 leading axis) gemm evaluates: {error}"),
    };
    std::hint::black_box(batched.root());

    let (gate_after, invocations_after, _) = proxima_tensor::cpu::width_tile_counters();
    let gate_delta = gate_after - gate_before;
    let invocations_delta = invocations_after - invocations_before;
    println!("leading_unit_axis_fires_the_width_tile: gate_passes={gate_delta} invocations={invocations_delta}");

    assert!(
        gate_delta > 0,
        "width_tile_plan declined the size-1-leading-axis shape -- the leading-axis gate did not \
         see through the batch axis, so the fix at resolve_reduce_axis_shape did not engage"
    );
    assert!(
        invocations_delta > 0,
        "width_tile_plan's gate passed but the tile kernel was never invoked -- a silent fallthrough \
         to the generic scalar path would still leave gate_delta > 0 while doing none of the SIMD work"
    );
}
