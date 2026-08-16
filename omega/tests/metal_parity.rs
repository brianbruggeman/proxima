//! GPU-vs-CPU parity gate for the Metal execution driver.
//!
//! Every test here requires a real Metal device — none of them skip. If
//! `MTLCreateSystemDefaultDevice` returns `None`, `omega::execute` returns
//! `MetalError::NoDevice`, and every `.expect(...)`/`.expect_err(...)` below
//! turns that into a loud test failure rather than a silently green (or
//! silently skipped) run.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use omega::MetalError;
use proxima_tensor::{
    AffineMap, AffineTerm, DType, DimExpr, Expr, Extent, Fold, FoldInit, IndexMap, Keep, NodeId,
    ScalarOp, TensorError, affine, append, evaluate, projection,
};

/// Asserts `cpu` and `metal` agree within `1e-6`, refusing a vacuous
/// (zero-element) comparison, and prints how many elements were compared and
/// the max abs diff observed — the number a report can cite even on a
/// passing run.
fn assert_parity(case: &str, cpu: &[f32], metal: &[f32]) -> f32 {
    assert!(
        !cpu.is_empty(),
        "{case}: cpu produced zero elements, a 0-length comparison proves nothing"
    );
    assert_eq!(
        cpu.len(),
        metal.len(),
        "{case}: element count mismatch (cpu={}, metal={})",
        cpu.len(),
        metal.len()
    );
    let max_abs_diff = cpu
        .iter()
        .zip(metal.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs_diff <= 1e-6,
        "{case}: max abs diff {max_abs_diff} exceeds 1e-6 tolerance across {} elements",
        cpu.len()
    );
    println!(
        "{case}: {} elements compared, max abs diff = {max_abs_diff:e}",
        cpu.len()
    );
    max_abs_diff
}

fn tanh_chain_program(extent: u32, depth: usize) -> Vec<Expr> {
    let mut program = Vec::new();
    let mut current = append(
        &mut program,
        Expr::Block {
            dtype: DType::Float32,
            shape: vec![Extent::Static(extent)],
            name: None,
        },
    );
    for _ in 0..depth {
        current = append(
            &mut program,
            Expr::Zip {
                dtype: DType::Float32,
                body: ScalarOp::Tanh,
                operands: vec![(current, IndexMap::Affine(projection(1, &[0])))],
                name: None,
            },
        );
    }
    let _ = current;
    program
}

fn matmul_program(m: u32, k: u32, n: u32, symbolic: bool) -> (Vec<Expr>, NodeId) {
    let mut program = Vec::new();
    let lhs_shape = if symbolic {
        vec![Extent::Symbolic(0), Extent::Static(k)]
    } else {
        vec![Extent::Static(m), Extent::Static(k)]
    };
    let lhs = append(
        &mut program,
        Expr::Block {
            dtype: DType::Float32,
            shape: lhs_shape,
            name: None,
        },
    );
    let rhs = append(
        &mut program,
        Expr::Block {
            dtype: DType::Float32,
            shape: vec![Extent::Static(k), Extent::Static(n)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Expr::Zip {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (lhs, IndexMap::Affine(projection(3, &[0, 2]))),
                (rhs, IndexMap::Affine(projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Expr::Fold(Fold {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: FoldInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(projection(3, &[0, 1])),
            keep: Keep::Last,
            name: Some("matmul".into()),
        }),
    );
    (program, sum)
}

fn softmax_program(n: u32, d: u32) -> Vec<Expr> {
    let mut program = Vec::new();
    let input = append(
        &mut program,
        Expr::Block {
            dtype: DType::Float32,
            shape: vec![Extent::Static(n), Extent::Static(d)],
            name: None,
        },
    );
    let row_map = IndexMap::Affine(projection(2, &[0, 1]));
    let broadcast_map = IndexMap::Affine(projection(2, &[0]));

    let max = append(
        &mut program,
        Expr::Fold(Fold {
            dtype: DType::Float32,
            body: ScalarOp::Maximum,
            init: FoldInit::NegativeInfinity,
            operand: input,
            in_map: row_map.clone(),
            out_map: broadcast_map.clone(),
            keep: Keep::Last,
            name: None,
        }),
    );
    let shifted = append(
        &mut program,
        Expr::Zip {
            dtype: DType::Float32,
            body: ScalarOp::Subtract,
            operands: vec![(input, row_map.clone()), (max, broadcast_map.clone())],
            name: None,
        },
    );
    let exponentiated = append(
        &mut program,
        Expr::Zip {
            dtype: DType::Float32,
            body: ScalarOp::Exponential,
            operands: vec![(shifted, row_map.clone())],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Expr::Fold(Fold {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: FoldInit::Zero,
            operand: exponentiated,
            in_map: row_map.clone(),
            out_map: broadcast_map.clone(),
            keep: Keep::Last,
            name: None,
        }),
    );
    append(
        &mut program,
        Expr::Zip {
            dtype: DType::Float32,
            body: ScalarOp::Divide,
            operands: vec![(exponentiated, row_map), (sum, broadcast_map)],
            name: None,
        },
    );
    program
}

fn cumsum_program(extent: u32) -> Vec<Expr> {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Expr::Block {
            dtype: DType::Float32,
            shape: vec![Extent::Static(extent)],
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
            in_map: IndexMap::Affine(projection(1, &[0])),
            out_map: IndexMap::Affine(projection(1, &[0])),
            keep: Keep::All,
            name: None,
        }),
    );
    program
}

/// A per-position ("locally connected") kernel: `kernel[h, r]` pins both
/// iteration dims via pure projection, while `signal[h + r]` is the
/// two-term windowed access under test.
fn conv_window_program(taps: u32, width: u32, signal_len: u32) -> Vec<Expr> {
    let mut program = Vec::new();
    let kernel = append(
        &mut program,
        Expr::Block {
            dtype: DType::Float32,
            shape: vec![Extent::Static(taps), Extent::Static(width)],
            name: None,
        },
    );
    let signal = append(
        &mut program,
        Expr::Block {
            dtype: DType::Float32,
            shape: vec![Extent::Static(signal_len)],
            name: None,
        },
    );
    let window = IndexMap::Affine(affine(
        2,
        &[(&[AffineTerm::scaled(0, 1), AffineTerm::scaled(1, 1)], 0)],
    ));
    let product = append(
        &mut program,
        Expr::Zip {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (kernel, IndexMap::Affine(projection(2, &[0, 1]))),
                (signal, window),
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
            in_map: IndexMap::Affine(projection(2, &[0, 1])),
            out_map: IndexMap::Affine(projection(2, &[0])),
            keep: Keep::Last,
            name: None,
        }),
    );
    program
}

/// `table[ids[s], d]` over iteration space `(s, d)`: the same worked example
/// `map.rs`'s docs use.
fn embedding_lookup_program(vocab: u32, dim: u32, seq: u32) -> Vec<Expr> {
    let mut program = Vec::new();
    let table = append(
        &mut program,
        Expr::Block {
            dtype: DType::Float32,
            shape: vec![Extent::Static(vocab), Extent::Static(dim)],
            name: None,
        },
    );
    let ids = append(
        &mut program,
        Expr::Block {
            dtype: DType::Int32,
            shape: vec![Extent::Static(seq)],
            name: None,
        },
    );
    let gathered_map = IndexMap::Computed {
        indices: ids,
        index_map: projection(2, &[0]),
        base: AffineMap {
            iter_rank: 2,
            dims: vec![
                DimExpr::default(),
                DimExpr {
                    terms: vec![AffineTerm::projection(1)],
                    offset: 0,
                },
            ],
        },
        gathered_dim: 0,
    };
    append(
        &mut program,
        Expr::Zip {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: vec![(table, gathered_map)],
            name: None,
        },
    );
    program
}

/// `sum_k table[ids[i], k] * weight[k, j]` — an embedding lookup fused
/// straight into a contraction, mirroring [`matmul_program`] with `lhs`
/// replaced by a gather.
fn embedding_matmul_program(vocab: u32, embed_dim: u32, seq: u32, out_dim: u32) -> Vec<Expr> {
    let mut program = Vec::new();
    let table = append(
        &mut program,
        Expr::Block {
            dtype: DType::Float32,
            shape: vec![Extent::Static(vocab), Extent::Static(embed_dim)],
            name: None,
        },
    );
    let ids = append(
        &mut program,
        Expr::Block {
            dtype: DType::Int32,
            shape: vec![Extent::Static(seq)],
            name: None,
        },
    );
    let weight = append(
        &mut program,
        Expr::Block {
            dtype: DType::Float32,
            shape: vec![Extent::Static(embed_dim), Extent::Static(out_dim)],
            name: None,
        },
    );

    let gather_map = IndexMap::Computed {
        indices: ids,
        index_map: projection(3, &[0]),
        base: AffineMap {
            iter_rank: 3,
            dims: vec![
                DimExpr::default(),
                DimExpr {
                    terms: vec![AffineTerm::projection(2)],
                    offset: 0,
                },
            ],
        },
        gathered_dim: 0,
    };
    let weight_map = IndexMap::Affine(projection(3, &[2, 1]));

    let product = append(
        &mut program,
        Expr::Zip {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(table, gather_map), (weight, weight_map)],
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
            in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(projection(3, &[0, 1])),
            keep: Keep::Last,
            name: Some("embedding_matmul".into()),
        }),
    );
    program
}

#[test]
fn matmul_parity_is_exact_for_integer_valued_inputs() {
    let (m, k, n) = (4usize, 3usize, 5usize);
    let (program, _sum) = matmul_program(m as u32, k as u32, n as u32, false);
    let lhs: Vec<f32> = (0..m * k).map(|value| value as f32).collect();
    let rhs: Vec<f32> = (0..k * n).map(|value| value as f32).collect();

    let cpu = evaluate(&program, &[], &[&lhs, &rhs], &[]).expect("cpu matmul evaluates");
    let metal = omega::execute(&program, &[], &[&lhs, &rhs], &[])
        .expect("metal matmul executes on a real device");

    let max_abs_diff = assert_parity("matmul", cpu.root(), metal.root());
    assert_eq!(
        max_abs_diff, 0.0,
        "integer-valued matmul inputs must round-trip exactly through f32 multiply-add \
         (max abs diff was {max_abs_diff})"
    );
}

#[test]
fn tanh_chain_parity_matches_within_epsilon() {
    let program = tanh_chain_program(4, 8);
    let input = [0.1, 0.2, 0.3, 0.4f32];

    let cpu = evaluate(&program, &[], &[&input], &[]).expect("cpu tanh chain evaluates");
    let metal = omega::execute(&program, &[], &[&input], &[])
        .expect("metal tanh chain executes on a real device");

    assert_parity("tanh_chain", cpu.root(), metal.root());
}

#[test]
fn softmax_parity_matches_within_epsilon() {
    let program = softmax_program(2, 4);
    let input = [1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0f32];

    let cpu = evaluate(&program, &[], &[&input], &[]).expect("cpu softmax evaluates");
    let metal = omega::execute(&program, &[], &[&input], &[])
        .expect("metal softmax executes on a real device");

    assert_parity("softmax", cpu.root(), metal.root());
}

#[test]
fn cumsum_parity_matches_exactly() {
    let program = cumsum_program(6);
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];

    let cpu = evaluate(&program, &[], &[&data], &[]).expect("cpu cumsum evaluates");
    let metal = omega::execute(&program, &[], &[&data], &[])
        .expect("metal cumsum executes on a real device");

    assert_parity("cumsum", cpu.root(), metal.root());
}

#[test]
fn conv_window_parity_matches_within_epsilon() {
    let program = conv_window_program(6, 3, 8);
    let kernel_data: Vec<f32> = (0..18).map(|value| value as f32).collect();
    let signal_data: Vec<f32> = (0..8).map(|value| value as f32).collect();

    let cpu =
        evaluate(&program, &[], &[&kernel_data, &signal_data], &[]).expect("cpu conv evaluates");
    let metal = omega::execute(&program, &[], &[&kernel_data, &signal_data], &[])
        .expect("metal conv executes on a real device");

    assert_parity("conv_window", cpu.root(), metal.root());
}

#[test]
fn embedding_lookup_parity_is_exact_for_integer_valued_inputs() {
    let (vocab, dim, seq) = (50_000usize, 8usize, 4usize);
    let program = embedding_lookup_program(vocab as u32, dim as u32, seq as u32);
    let table_data: Vec<f32> = (0..vocab * dim).map(|value| (value % 97) as f32).collect();
    let ids_data = [3.0f32, 49_999.0, 12_345.0, 0.0];

    let cpu = evaluate(&program, &[], &[&table_data, &ids_data], &[])
        .expect("cpu embedding lookup evaluates");
    let metal = omega::execute(&program, &[], &[&table_data, &ids_data], &[])
        .expect("metal embedding lookup executes on a real device");

    let max_abs_diff = assert_parity("embedding_lookup", cpu.root(), metal.root());
    assert_eq!(
        max_abs_diff, 0.0,
        "gathering integer-valued table rows must round-trip exactly (max abs diff was \
         {max_abs_diff})"
    );
}

#[test]
fn embedding_matmul_parity_matches_within_epsilon() {
    let (vocab, embed_dim, seq, out_dim) = (100usize, 6usize, 4usize, 3usize);
    let program =
        embedding_matmul_program(vocab as u32, embed_dim as u32, seq as u32, out_dim as u32);
    let table_data: Vec<f32> = (0..vocab * embed_dim)
        .map(|value| (value % 13) as f32)
        .collect();
    let ids_data = [3.0f32, 99.0, 50.0, 0.0];
    let weight_data: Vec<f32> = (0..embed_dim * out_dim)
        .map(|value| (value % 5) as f32)
        .collect();

    let cpu = evaluate(&program, &[], &[&table_data, &ids_data, &weight_data], &[])
        .expect("cpu embedding matmul evaluates");
    let metal = omega::execute(&program, &[], &[&table_data, &ids_data, &weight_data], &[])
        .expect("metal embedding matmul executes on a real device");

    assert_parity("embedding_matmul", cpu.root(), metal.root());
}

#[test]
fn out_of_range_gather_index_produces_the_same_error_on_cpu_and_metal() {
    let (vocab, dim, seq) = (4usize, 2usize, 3usize);
    let program = embedding_lookup_program(vocab as u32, dim as u32, seq as u32);
    let table_data: Vec<f32> = (0..vocab * dim).map(|value| value as f32).collect();
    // every fetched index is the same out-of-range value: which thread's
    // atomic_fetch_max "wins" on the metal side then cannot change what gets
    // reported, so the two backends' field values are directly comparable
    // rather than merely both-erroring.
    let ids_data = [vocab as f32, vocab as f32, vocab as f32];

    let cpu_error = evaluate(&program, &[], &[&table_data, &ids_data], &[])
        .expect_err("cpu rejects the out-of-range gather");
    let metal_error = omega::execute(&program, &[], &[&table_data, &ids_data], &[])
        .expect_err("metal rejects the out-of-range gather too, not clamping it away");

    assert!(
        matches!(cpu_error, TensorError::GatherIndexOutOfRange { .. }),
        "{cpu_error}"
    );
    match metal_error {
        MetalError::Tensor(tensor_error) => assert_eq!(
            tensor_error, cpu_error,
            "cpu and metal must report the identical TensorError fields"
        ),
        other => panic!("expected MetalError::Tensor, got {other:?}"),
    }
}

#[test]
fn multi_output_parity_covers_both_an_intermediate_and_the_root() {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Expr::Block {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: None,
        },
    );
    let mut current = source;
    let mut nodes = vec![source];
    for _ in 0..4 {
        current = append(
            &mut program,
            Expr::Zip {
                dtype: DType::Float32,
                body: ScalarOp::Tanh,
                operands: vec![(current, IndexMap::Affine(projection(1, &[0])))],
                name: None,
            },
        );
        nodes.push(current);
    }
    let midpoint = nodes[2];
    let root = current;
    let input = [0.1, 0.2, 0.3, 0.4f32];

    let cpu =
        evaluate(&program, &[], &[&input], &[midpoint, root]).expect("cpu multi-output evaluates");
    let metal = omega::execute(&program, &[], &[&input], &[midpoint, root])
        .expect("metal multi-output executes on a real device");

    let (cpu_mid, _) = cpu.get(midpoint).expect("cpu midpoint present");
    let (metal_mid, _) = metal.get(midpoint).expect("metal midpoint present");
    assert_parity("multi_output_midpoint", cpu_mid, metal_mid);

    let (cpu_root, _) = cpu.get(root).expect("cpu root present");
    let (metal_root, _) = metal.get(root).expect("metal root present");
    assert_parity("multi_output_root", cpu_root, metal_root);
}

#[test]
fn symbolic_extent_parity_holds_across_two_different_bindings() {
    let (program, _sum) = matmul_program(0, 3, 5, true);

    for m in [4usize, 8usize] {
        let lhs: Vec<f32> = (0..m * 3).map(|value| (value % 7) as f32).collect();
        let rhs: Vec<f32> = (0..3 * 5).map(|value| (value % 5) as f32).collect();

        let cpu = evaluate(&program, &[m as u64], &[&lhs, &rhs], &[])
            .expect("cpu symbolic matmul evaluates");
        let metal = omega::execute(&program, &[m as u64], &[&lhs, &rhs], &[])
            .expect("metal symbolic matmul executes on a real device, uniforms not baked");

        assert_parity(&format!("symbolic_matmul_m{m}"), cpu.root(), metal.root());
    }
}

#[test]
fn block_count_mismatch_produces_the_same_tensor_error_as_cpu() {
    let mut program = Vec::new();
    append(
        &mut program,
        Expr::Block {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: None,
        },
    );

    let cpu_error = evaluate(&program, &[], &[], &[]).expect_err("cpu rejects missing block");
    let metal_error =
        omega::execute(&program, &[], &[], &[]).expect_err("metal rejects missing block too");

    match metal_error {
        MetalError::Tensor(tensor_error) => assert_eq!(tensor_error, cpu_error),
        other => panic!("expected MetalError::Tensor, got {other:?}"),
    }
}

#[test]
fn block_size_mismatch_produces_the_same_tensor_error_as_cpu() {
    let mut program = Vec::new();
    append(
        &mut program,
        Expr::Block {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: None,
        },
    );
    let too_short = [1.0, 2.0f32];

    let cpu_error =
        evaluate(&program, &[], &[&too_short], &[]).expect_err("cpu rejects wrong block size");
    let metal_error = omega::execute(&program, &[], &[&too_short], &[])
        .expect_err("metal rejects wrong block size too");

    match metal_error {
        MetalError::Tensor(tensor_error) => assert_eq!(tensor_error, cpu_error),
        other => panic!("expected MetalError::Tensor, got {other:?}"),
    }
}
