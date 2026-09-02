//! GPU-vs-CPU parity gate for the Metal execution driver.
//!
//! Every test here requires a real Metal device — none of them skip. If
//! `MTLCreateSystemDefaultDevice` returns `None`, `omega::execute` returns
//! `MetalError::NoDevice`, and every `.expect(...)`/`.expect_err(...)` below
//! turns that into a loud test failure rather than a silently green (or
//! silently skipped) run.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use conflaguration::Validate;
use omega::MetalError;
use proxima_tensor::spec::ProgramSpec;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    AxisIndex, AxisTerm, BoundOpKind, DType, Extent, IndexMap, IndexPattern, Keep, NodeId, Op,
    QuantizedBlock, Reduce, ReduceInit, ScalarOp, TensorError, affine, append, bind, evaluate,
    infer, projection,
};

/// Asserts `cpu` and `metal` agree within `1e-6`, refusing a vacuous
/// (zero-element) comparison, and prints how many elements were compared and
/// the max abs diff observed — the number a report can cite even on a
/// passing run.
fn assert_parity(case: &str, cpu: &[f32], metal: &[f32]) -> f32 {
    assert_parity_within(case, cpu, metal, 1e-6)
}

/// Same as [`assert_parity`], but with a caller-chosen tolerance instead of
/// the default `1e-6` — for cases where a longer chain of GPU-vs-CPU
/// reduction (matmul into softmax into matmul, on non-trivial varied inputs)
/// legitimately accumulates more float error than a single fused op does.
fn assert_parity_within(case: &str, cpu: &[f32], metal: &[f32], epsilon: f32) -> f32 {
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
        max_abs_diff <= epsilon,
        "{case}: max abs diff {max_abs_diff} exceeds {epsilon} tolerance across {} elements",
        cpu.len()
    );
    println!(
        "{case}: {} elements compared, max abs diff = {max_abs_diff:e}",
        cpu.len()
    );
    max_abs_diff
}

fn tanh_chain_program(extent: u32, depth: usize) -> Vec<Op> {
    let mut program = Vec::new();
    let mut current = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(extent)],
            name: None,
        },
    );
    for _ in 0..depth {
        current = append(
            &mut program,
            Op::Elementwise {
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

fn reciprocal_program(extent: u32) -> Vec<Op> {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(extent)],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Reciprocal,
            operands: vec![(source, IndexMap::Affine(projection(1, &[0])))],
            name: None,
        },
    );
    program
}

fn square_root_program(extent: u32) -> Vec<Op> {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(extent)],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::SquareRoot,
            operands: vec![(source, IndexMap::Affine(projection(1, &[0])))],
            name: None,
        },
    );
    program
}

/// `multiply -> square_root -> reciprocal` — RMSNorm's own `inv_rms` shape
/// (`attention_block.toml`'s `mean_square`/`rms`/`inv_rms` nodes), asserted
/// to fuse into one `BoundOp` the same way
/// `a_multi_operand_elementwise_fusion_chain_matches_cpu_on_a_real_device`
/// does for multiply/add, so a failure here can only be `square_root` or
/// `reciprocal` disagreeing inside a fused kernel, not some other op.
fn multiply_sqrt_reciprocal_chain_program() -> Vec<Op> {
    let mut program = Vec::new();
    let identity = || IndexMap::Affine(projection(1, &[0]));
    let a = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: None,
        },
    );
    let scale = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: None,
        },
    );
    let squared = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(a, identity()), (scale, identity())],
            name: None,
        },
    );
    let rooted = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::SquareRoot,
            operands: vec![(squared, identity())],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Reciprocal,
            operands: vec![(rooted, identity())],
            name: None,
        },
    );
    program
}

fn matmul_program(m: u32, k: u32, n: u32, symbolic: bool) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs_shape = if symbolic {
        vec![Extent::Symbolic(0), Extent::Static(k)]
    } else {
        vec![Extent::Static(m), Extent::Static(k)]
    };
    let lhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: lhs_shape,
            name: None,
        },
    );
    let rhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(k), Extent::Static(n)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
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
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("matmul".into()),
        }),
    );
    (program, sum)
}

fn softmax_program(n: u32, d: u32) -> Vec<Op> {
    let mut program = Vec::new();
    let input = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(n), Extent::Static(d)],
            name: None,
        },
    );
    let row_map = IndexMap::Affine(projection(2, &[0, 1]));
    let broadcast_map = IndexMap::Affine(projection(2, &[0]));

    let max = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Maximum,
            init: ReduceInit::NegativeInfinity,
            operand: input,
            in_map: row_map.clone(),
            out_map: broadcast_map.clone(),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    let shifted = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Subtract,
            operands: vec![(input, row_map.clone()), (max, broadcast_map.clone())],
            name: None,
        },
    );
    let exponentiated = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Exponential,
            operands: vec![(shifted, row_map.clone())],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: exponentiated,
            in_map: row_map.clone(),
            out_map: broadcast_map.clone(),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Divide,
            operands: vec![(exponentiated, row_map), (sum, broadcast_map)],
            name: None,
        },
    );
    program
}

fn cumsum_program(extent: u32) -> Vec<Op> {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(extent)],
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
            in_map: IndexMap::Affine(projection(1, &[0])),
            out_map: IndexMap::Affine(projection(1, &[0])),
            keep: Keep::Scan,
            name: None,
        }),
    );
    program
}

/// A per-position ("locally connected") kernel: `kernel[h, r]` pins both
/// iteration dims via pure projection, while `signal[h + r]` is the
/// two-term windowed access under test.
fn conv_window_program(taps: u32, width: u32, signal_len: u32) -> Vec<Op> {
    let mut program = Vec::new();
    let kernel = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(taps), Extent::Static(width)],
            name: None,
        },
    );
    let signal = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(signal_len)],
            name: None,
        },
    );
    let window = IndexMap::Affine(affine(
        2,
        &[(&[AxisTerm::scaled(0, 1), AxisTerm::scaled(1, 1)], 0)],
    ));
    let product = append(
        &mut program,
        Op::Elementwise {
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
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(2, &[0, 1])),
            out_map: IndexMap::Affine(projection(2, &[0])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    program
}

/// `table[ids[s], d]` over iteration space `(s, d)`: the same worked example
/// `map.rs`'s docs use.
fn embedding_lookup_program(vocab: u32, dim: u32, seq: u32) -> Vec<Op> {
    let mut program = Vec::new();
    let table = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(vocab), Extent::Static(dim)],
            name: None,
        },
    );
    let ids = append(
        &mut program,
        Op::Input {
            dtype: DType::Int32,
            shape: vec![Extent::Static(seq)],
            name: None,
        },
    );
    let gathered_map = IndexMap::Computed {
        indices: ids,
        index_map: projection(2, &[0]),
        base: IndexPattern {
            iter_rank: 2,
            axes: vec![
                AxisIndex::default(),
                AxisIndex {
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
    program
}

/// `sum_k table[ids[i], k] * weight[k, j]` — an embedding lookup fused
/// straight into a contraction, mirroring [`matmul_program`] with `lhs`
/// replaced by a gather.
fn embedding_matmul_program(vocab: u32, embed_dim: u32, seq: u32, out_dim: u32) -> Vec<Op> {
    let mut program = Vec::new();
    let table = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(vocab), Extent::Static(embed_dim)],
            name: None,
        },
    );
    let ids = append(
        &mut program,
        Op::Input {
            dtype: DType::Int32,
            shape: vec![Extent::Static(seq)],
            name: None,
        },
    );
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(embed_dim), Extent::Static(out_dim)],
            name: None,
        },
    );

    let gather_map = IndexMap::Computed {
        indices: ids,
        index_map: projection(3, &[0]),
        base: IndexPattern {
            iter_rank: 3,
            axes: vec![
                AxisIndex::default(),
                AxisIndex {
                    terms: vec![AxisTerm::projection(2)].into(),
                    offset: 0,
                },
            ],
        },
        gathered_dim: 0,
    };
    let weight_map = IndexMap::Affine(projection(3, &[2, 1]));

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
            in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(projection(3, &[0, 1])),
            keep: Keep::Reduce,
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
    let metal = omega::execute(
        &program,
        &[],
        &[QuantizedBlock::Float32(&lhs), QuantizedBlock::Float32(&rhs)],
        &[],
    )
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
    let metal = omega::execute(&program, &[], &[QuantizedBlock::Float32(&input)], &[])
        .expect("metal tanh chain executes on a real device");

    assert_parity("tanh_chain", cpu.root(), metal.root());
}

#[test]
fn reciprocal_parity_matches_within_epsilon() {
    let program = reciprocal_program(4);
    let input = [1.0, 2.0, 0.5, -4.0f32];

    let cpu = evaluate(&program, &[], &[&input], &[]).expect("cpu reciprocal evaluates");
    let metal = omega::execute(&program, &[], &[QuantizedBlock::Float32(&input)], &[])
        .expect("metal reciprocal executes on a real device");

    assert_parity("reciprocal", cpu.root(), metal.root());
}

#[test]
fn square_root_parity_matches_within_epsilon() {
    let program = square_root_program(4);
    let input = [1.0, 4.0, 9.0, 0.25f32];

    let cpu = evaluate(&program, &[], &[&input], &[]).expect("cpu square_root evaluates");
    let metal = omega::execute(&program, &[], &[QuantizedBlock::Float32(&input)], &[])
        .expect("metal square_root executes on a real device");

    assert_parity("square_root", cpu.root(), metal.root());
}

#[test]
fn multiply_sqrt_reciprocal_chain_matches_cpu_on_a_real_device() {
    let program = multiply_sqrt_reciprocal_chain_program();

    let shapes = infer(&program, &[]).expect("multiply/sqrt/reciprocal chain infers");
    let resolved = bind(&program, &shapes, &[]).expect("multiply/sqrt/reciprocal chain resolves");
    assert_eq!(
        resolved.len(),
        1,
        "multiply, square_root, and reciprocal must fuse into one BoundOp before this ever \
         reaches the Metal driver"
    );

    let a_data = [1.0, 2.0, 3.0, 4.0f32];
    let scale_data = [4.0, 2.0, 1.0, 0.5f32];

    let cpu = evaluate(&program, &[], &[&a_data, &scale_data], &[])
        .expect("cpu multiply/sqrt/reciprocal chain evaluates");
    let metal = omega::execute(
        &program,
        &[],
        &[
            QuantizedBlock::Float32(&a_data),
            QuantizedBlock::Float32(&scale_data),
        ],
        &[],
    )
    .expect("metal multiply/sqrt/reciprocal chain executes on a real device");

    assert_parity("multiply_sqrt_reciprocal_chain", cpu.root(), metal.root());
}

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// `next_unit` scaled into `[0.98, 1.02)` — a repeated `Multiply` reduce over
/// raw `[-1, 1)` values collapses toward `0.0` within a few dozen terms
/// (magnitude < 1 shrinks every step), which would make a broken lane
/// combination and a correct one equally indistinguishable at `1e-6`. Values
/// near `1.0` keep the running product away from both underflow and
/// overflow across a contraction long enough to span multiple SIMD lanes.
fn random_vec_near_one(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| 1.0 + lcg.next_unit() * 0.02).collect()
}

/// One `Reduce` over the last axis of a `(rows, cols)` input, output
/// `(rows,)` — the minimal program shape to drive `body`/`init` straight at
/// `crate::msl::render_reduce` without an elementwise fusion in the way.
fn axis_reduce_program(rows: u32, cols: u32, body: ScalarOp, init: ReduceInit) -> Vec<Op> {
    let mut program = Vec::new();
    let input = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(rows), Extent::Static(cols)],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body,
            init,
            operand: input,
            in_map: IndexMap::Affine(projection(2, &[0, 1])),
            out_map: IndexMap::Affine(projection(2, &[0])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    program
}

/// Inputs are LCG-derived rather than uniform constants: uniform q/k/v rows
/// make every attention score identical, so the softmax collapses to a
/// uniform `1/SEQUENCE` and the Metal-vs-CPU comparison stays symmetric
/// under an index-map transpose, a reduction-order bug, or a wrong
/// broadcast — all invisible when every row is the same value. Varied
/// inputs make those bugs show up as a real numeric divergence.
#[test]
fn attention_block_spec_parity_matches_within_epsilon() {
    const SEQUENCE: usize = 4;
    const MODEL: usize = 8;

    let text = include_str!("../../proxima-tensor/specs/attention_block.toml");
    let spec: ProgramSpec = toml::from_str(text).expect("spec parses");
    spec.validate().expect("spec is structurally sound");
    let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");

    let symbols = [SEQUENCE as u64];
    infer(&program, &symbols).expect("the block infers");

    let activations = random_vec(1, SEQUENCE * MODEL);
    let inverse_dim = vec![1.0 / MODEL as f32; SEQUENCE];
    let wq = random_vec(2, MODEL * MODEL);
    let wk = random_vec(3, MODEL * MODEL);
    let wv = random_vec(4, MODEL * MODEL);
    let blocks: [&[f32]; 5] = [&activations, &inverse_dim, &wq, &wk, &wv];

    let cpu = evaluate(&program, &symbols, &blocks, &[]).expect("cpu attention block evaluates");
    let gpu_blocks: [QuantizedBlock<'_>; 5] = blocks.map(QuantizedBlock::Float32);
    let metal = omega::execute(&program, &symbols, &gpu_blocks, &[])
        .expect("metal attention block executes on a real device");

    // With uniform inputs every row collapsed and the diff floored at 0e0.
    // Varied inputs push a real number through rmsnorm -> 3 matmuls -> softmax
    // -> matmul, so GPU-vs-CPU reduction order legitimately accumulates more
    // float error than the 1e-6 default; observed max abs diff was ~4.3e-6.
    assert_parity_within("attention_block", cpu.root(), metal.root(), 1e-5);
}

/// `b = a * scale; c = b + bias; d = c * c` — a multi-operand
/// elementwise-into-elementwise fusion chain (as opposed to
/// `tanh_chain_program`'s single-operand unary chain), asserted to bind to
/// one `BoundOp` before it is even handed to the Metal driver, so a failure
/// here can only be the fused MSL kernel disagreeing with the CPU
/// interpreter, not some other op in the program.
#[test]
fn a_multi_operand_elementwise_fusion_chain_matches_cpu_on_a_real_device() {
    let mut program = Vec::new();
    let identity = || IndexMap::Affine(projection(1, &[0]));
    let a = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: None,
        },
    );
    let scale = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: None,
        },
    );
    let bias = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: None,
        },
    );
    let b = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(a, identity()), (scale, identity())],
            name: None,
        },
    );
    let c = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: vec![(b, identity()), (bias, identity())],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(c, identity()), (c, identity())],
            name: None,
        },
    );

    let shapes = infer(&program, &[]).expect("elementwise chain infers");
    let resolved = bind(&program, &shapes, &[]).expect("elementwise chain resolves");
    assert_eq!(
        resolved.len(),
        1,
        "b and c must fuse into d's own BoundOp before this ever reaches the Metal driver"
    );
    assert!(matches!(resolved[0].kind, BoundOpKind::Elementwise { .. }));

    let a_data = [1.0, 2.0, 3.0, 4.0f32];
    let scale_data = [2.0, 0.5, -1.0, 3.0f32];
    let bias_data = [1.0, 1.0, 1.0, 1.0f32];

    let cpu = evaluate(&program, &[], &[&a_data, &scale_data, &bias_data], &[])
        .expect("cpu elementwise chain evaluates");
    let metal = omega::execute(
        &program,
        &[],
        &[
            QuantizedBlock::Float32(&a_data),
            QuantizedBlock::Float32(&scale_data),
            QuantizedBlock::Float32(&bias_data),
        ],
        &[],
    )
    .expect("metal elementwise chain executes on a real device");

    assert_parity("elementwise_fusion_chain", cpu.root(), metal.root());
}

#[test]
fn softmax_parity_matches_within_epsilon() {
    let program = softmax_program(2, 4);
    let input = [1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0f32];

    let cpu = evaluate(&program, &[], &[&input], &[]).expect("cpu softmax evaluates");
    let metal = omega::execute(&program, &[], &[QuantizedBlock::Float32(&input)], &[])
        .expect("metal softmax executes on a real device");

    assert_parity("softmax", cpu.root(), metal.root());
}

#[test]
fn cumsum_parity_matches_exactly() {
    let program = cumsum_program(6);
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];

    let cpu = evaluate(&program, &[], &[&data], &[]).expect("cpu cumsum evaluates");
    let metal = omega::execute(&program, &[], &[QuantizedBlock::Float32(&data)], &[])
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
    let metal = omega::execute(
        &program,
        &[],
        &[
            QuantizedBlock::Float32(&kernel_data),
            QuantizedBlock::Float32(&signal_data),
        ],
        &[],
    )
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
    let metal = omega::execute(
        &program,
        &[],
        &[
            QuantizedBlock::Float32(&table_data),
            QuantizedBlock::Float32(&ids_data),
        ],
        &[],
    )
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
    let metal = omega::execute(
        &program,
        &[],
        &[
            QuantizedBlock::Float32(&table_data),
            QuantizedBlock::Float32(&ids_data),
            QuantizedBlock::Float32(&weight_data),
        ],
        &[],
    )
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
    let metal_error = omega::execute(
        &program,
        &[],
        &[
            QuantizedBlock::Float32(&table_data),
            QuantizedBlock::Float32(&ids_data),
        ],
        &[],
    )
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
        Op::Input {
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
            Op::Elementwise {
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
    let metal = omega::execute(
        &program,
        &[],
        &[QuantizedBlock::Float32(&input)],
        &[midpoint, root],
    )
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
        let metal = omega::execute(
            &program,
            &[m as u64],
            &[QuantizedBlock::Float32(&lhs), QuantizedBlock::Float32(&rhs)],
            &[],
        )
        .expect("metal symbolic matmul executes on a real device, uniforms not baked");

        assert_parity(&format!("symbolic_matmul_m{m}"), cpu.root(), metal.root());
    }
}

#[test]
fn block_count_mismatch_produces_the_same_tensor_error_as_cpu() {
    let mut program = Vec::new();
    append(
        &mut program,
        Op::Input {
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
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: None,
        },
    );
    let too_short = [1.0, 2.0f32];

    let cpu_error =
        evaluate(&program, &[], &[&too_short], &[]).expect_err("cpu rejects wrong block size");
    let metal_error = omega::execute(&program, &[], &[QuantizedBlock::Float32(&too_short)], &[])
        .expect_err("metal rejects wrong block size too");

    match metal_error {
        MetalError::Tensor(tensor_error) => assert_eq!(tensor_error, cpu_error),
        other => panic!("expected MetalError::Tensor, got {other:?}"),
    }
}

/// `k = 97` is prime and well past `SIMD_WIDTH` (32): every one of the 32
/// lanes a cooperative reduce dispatches gets a real, uneven share of the
/// contraction (97 = 3*32 + 1), and the tail lane (lane 0 alone gets a 4th
/// element) is exactly the boundary a naive `reduction_total / SIMD_WIDTH`
/// assumption would get wrong. Varied LCG inputs (not the constant/sequential
/// data `matmul_parity_is_exact_for_integer_valued_inputs` uses) so a broken
/// lane combination cannot hide behind every lane holding the same value.
#[test]
fn matmul_parity_holds_over_a_contraction_spanning_multiple_simd_lanes() {
    let (m, k, n) = (5usize, 97usize, 6usize);
    let (program, _sum) = matmul_program(m as u32, k as u32, n as u32, false);
    let lhs = random_vec(0x5eed_5eed_5eed_5eed, m * k);
    let rhs = random_vec(0x1234_5678_9abc_def0, k * n);

    let cpu = evaluate(&program, &[], &[&lhs, &rhs], &[]).expect("cpu matmul evaluates");
    let metal = omega::execute(
        &program,
        &[],
        &[QuantizedBlock::Float32(&lhs), QuantizedBlock::Float32(&rhs)],
        &[],
    )
    .expect("metal matmul executes on a real device");

    // observed worst-case diff on this host: 7.629395e-6 (k=97 lanes
    // reassociated) — a real, expected float-reassociation cost, not a
    // shared-epsilon change. Widened to 1e-5 for this call site only.
    assert_parity_within("matmul_wide_k", cpu.root(), metal.root(), 1e-5);
}

/// `Maximum`/`Minimum` and `Multiply` reduces, isolated from any fused
/// elementwise body, over the same `cols = 97` multi-lane contraction —
/// `matmul_parity_holds_over_a_contraction_spanning_multiple_simd_lanes`
/// only exercises `Add` (via `simd_sum`); this covers the other three
/// cooperative bodies (`simd_max`, `simd_min`, `simd_product`) directly.
#[test]
fn axis_reduce_parity_holds_for_every_cooperative_reduce_body() {
    let (rows, cols) = (4u32, 97u32);
    let data = random_vec(0x9e37_79b9_7f4a_7c15, (rows * cols) as usize);
    let data_near_one = random_vec_near_one(0xbf58_476d_1ce4_e5b9, (rows * cols) as usize);

    let cases = [
        (
            "axis_max",
            ScalarOp::Maximum,
            ReduceInit::NegativeInfinity,
            &data,
        ),
        (
            "axis_min",
            ScalarOp::Minimum,
            ReduceInit::PositiveInfinity,
            &data,
        ),
        (
            "axis_multiply",
            ScalarOp::Multiply,
            ReduceInit::One,
            &data_near_one,
        ),
    ];

    for (case, body, init, input) in cases {
        let program = axis_reduce_program(rows, cols, body, init);
        let cpu = evaluate(&program, &[], &[input], &[]).unwrap_or_else(|error| {
            panic!("{case}: cpu axis reduce evaluates: {error}");
        });
        let metal = omega::execute(&program, &[], &[QuantizedBlock::Float32(input)], &[])
            .unwrap_or_else(|error| {
                panic!("{case}: metal axis reduce executes on a real device: {error}");
            });
        assert_parity(case, cpu.root(), metal.root());
    }
}

/// Same shape [`matmul_program`] builds, parameterized over `dtype` instead
/// of hardcoding `Float32` — the f16 parity gate below builds the identical
/// structure twice, once per dtype, so a divergence in the comparison can
/// only be the dtype itself, never an accidental shape mismatch between two
/// independently-hand-written programs.
fn matmul_program_with_dtype(m: u32, k: u32, n: u32, dtype: DType) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs = append(
        &mut program,
        Op::Input {
            dtype,
            shape: vec![Extent::Static(m), Extent::Static(k)],
            name: None,
        },
    );
    let rhs = append(
        &mut program,
        Op::Input {
            dtype,
            shape: vec![Extent::Static(k), Extent::Static(n)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype,
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
        Op::Reduce(Reduce {
            dtype,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("matmul_f16".into()),
        }),
    );
    (program, sum)
}

/// The GPU-vs-CPU parity gate for real f16 execution: an f16-tagged program
/// dispatches through `omega::execute` (`msl.rs` emits `half` buffers and
/// `metal.rs` marshals f32 host data down to `half::f16` at upload and back
/// up at read-back — see `metal.rs`'s dtype doc), compared against the
/// identically-shaped f32-tagged program run through `cpu::evaluate`, the
/// f32 reference oracle. `m`, `k`, `n` are pairwise coprime and none is a
/// power of two, so a row/column swap or a transposed stride would not
/// accidentally produce the right element count at the right shape.
///
/// f16 carries 10 explicit mantissa bits (11 with the implicit leading
/// one) versus f32's 23, so relative error is the only fair comparison
/// across magnitudes -- with one exception, below. `RELATIVE_EPSILON` is
/// set from a measured run, not a round number: with `Lcg` seeds 11/12 over
/// `m=17, k=23, n=13` (221 output elements, each a 23-term dot product), the
/// worst observed relative error at a non-degenerate reference magnitude
/// was `1.6379411e-3`, consistent with f16's ~2^-11 (4.9e-4) per-rounding
/// unit compounding over a 23-deep accumulation chain. `RELATIVE_EPSILON`
/// is set to 5e-3, roughly 3x that measured worst case, to absorb
/// run-to-run rounding-mode variance (a different `Lcg` seed shifts which
/// dot products land near a rounding boundary) without the gate flaking on
/// a re-run.
///
/// A pure-relative bound is unsound at a zero crossing: `test_support::
/// Lcg::next_unit`'s corrected [-1,1) range (see that function's doc) means
/// some of these 23-term dot products now sum near zero by cancellation.
/// One measured case: reference=-3.4488782e-3, f16 observed=-5.126953e-3,
/// an absolute diff of 1.678075e-3 -- ordinary f16 rounding noise -- but a
/// relative error of 0.4865, because the denominator is tiny, not because
/// the compute is wrong. The bound below is `atol + rtol * |reference|`,
/// the standard combined tolerance for exactly this shape of problem.
/// `ABSOLUTE_EPSILON` is set from the same measured run's largest absolute
/// diff across all 221 elements, `3.3743382e-3`, with ~20% headroom.
#[test]
fn matmul_parity_is_within_f16_epsilon_of_the_f32_cpu_oracle() {
    const RELATIVE_EPSILON: f32 = 5e-3;
    const ABSOLUTE_EPSILON: f32 = 4e-3;

    let (m, k, n) = (17usize, 23usize, 13usize);
    let (f32_program, _) = matmul_program_with_dtype(m as u32, k as u32, n as u32, DType::Float32);
    let (f16_program, _) = matmul_program_with_dtype(m as u32, k as u32, n as u32, DType::Float16);

    let lhs = random_vec(11, m * k);
    let rhs = random_vec(12, k * n);

    let cpu =
        evaluate(&f32_program, &[], &[&lhs, &rhs], &[]).expect("f32 cpu oracle matmul evaluates");
    let metal = omega::execute(
        &f16_program,
        &[],
        &[QuantizedBlock::Float32(&lhs), QuantizedBlock::Float32(&rhs)],
        &[],
    )
    .expect("f16 metal matmul executes on a real device");

    assert!(
        !cpu.root().is_empty(),
        "cpu oracle produced zero elements, a 0-length comparison proves nothing"
    );
    assert_eq!(
        cpu.root().len(),
        m * n,
        "unexpected element count from the cpu oracle"
    );
    assert_eq!(
        cpu.root().len(),
        metal.root().len(),
        "element count mismatch (cpu={}, metal={})",
        cpu.root().len(),
        metal.root().len()
    );

    let mut worst_relative = 0.0f32;
    let mut worst_absolute = 0.0f32;
    for (reference, observed) in cpu.root().iter().zip(metal.root().iter()) {
        let denominator = reference.abs().max(1e-6);
        let absolute = (reference - observed).abs();
        let relative = absolute / denominator;
        let bound = ABSOLUTE_EPSILON + RELATIVE_EPSILON * reference.abs();
        assert!(
            absolute <= bound,
            "f16 matmul disagrees beyond the combined tolerance: reference={reference} \
             observed={observed} abs_diff={absolute} bound={bound} \
             (atol={ABSOLUTE_EPSILON}, rtol={RELATIVE_EPSILON})"
        );
        worst_relative = worst_relative.max(relative);
        worst_absolute = worst_absolute.max(absolute);
    }
    println!(
        "matmul_f16: {} elements compared, worst relative error = {worst_relative:e}, \
         worst absolute diff = {worst_absolute:e} (atol = {ABSOLUTE_EPSILON:e}, \
         rtol = {RELATIVE_EPSILON:e})",
        cpu.root().len()
    );
}

#[test]
fn page_aligned_input_takes_the_no_copy_metal_upload_path() {
    let page_elements = (omega::page_size() / std::mem::size_of::<f32>()) as u32;

    let mut program = Vec::new();
    let input = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(page_elements)],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Tanh,
            operands: vec![(input, IndexMap::Affine(projection(1, &[0])))],
            name: None,
        },
    );

    let mut aligned =
        proxima_tensor::AlignedBuffer::new(page_elements as usize, omega::page_size())
            .expect("small aligned request never fails");
    for (index, value) in aligned.iter_mut().enumerate() {
        *value = (index % 7) as f32 * 0.1;
    }
    assert_eq!(
        aligned.len(),
        page_elements as usize,
        "an already-page-sized request must round to itself, no padding"
    );

    let nocopy_before = omega::metal::NOCOPY_BUFFER_UPLOADS.get();
    let copy_before = omega::metal::COPYING_BUFFER_UPLOADS.get();

    let cpu =
        evaluate(&program, &[], &[&aligned[..]], &[]).expect("cpu evaluates the page-sized chain");
    let metal = omega::execute(&program, &[], &[QuantizedBlock::Float32(&aligned[..])], &[])
        .expect("metal executes the page-aligned block on a real device");

    let nocopy_after = omega::metal::NOCOPY_BUFFER_UPLOADS.get();
    let copy_after = omega::metal::COPYING_BUFFER_UPLOADS.get();

    assert_parity("page_aligned_input", cpu.root(), metal.root());

    println!(
        "page_aligned_input_takes_the_no_copy_metal_upload_path: nocopy {nocopy_before}->{nocopy_after}, copy {copy_before}->{copy_after}"
    );
    assert_eq!(
        nocopy_after - nocopy_before,
        1,
        "a page-aligned {page_elements}-element block must take the zero-copy upload path exactly once"
    );
    assert_eq!(
        copy_after, copy_before,
        "the page-aligned block must not also count as a copying upload"
    );
}

/// Reports the no-copy-vs-copy upload split across every real
/// [`omega::execute`] call this whole test binary makes — not a
/// per-program assertion (see the other tests for that), just the number
/// this task's report is required to cite. Sorts alphabetically last in
/// this file (`u` after `t`), so a single-process, single-threaded run
/// (`cargo test -p omega --features metal --test metal_parity --
/// --test-threads=1`) executes it after every other test in the binary has
/// already incremented these same process-global counters — printed here,
/// not asserted to an exact value, since which other tests ran (and how
/// many blocks each uploaded) is this file's business, not this test's.
#[ignore = "meaningful only under a single-process run (cargo test -- --test-threads=1 --ignored); nextest isolates every test into its own process, so this would read 0/0 there"]
#[test]
fn upload_path_totals_report_after_the_full_parity_suite() {
    let nocopy = omega::metal::NOCOPY_BUFFER_UPLOADS.get();
    let copy = omega::metal::COPYING_BUFFER_UPLOADS.get();
    println!("metal_parity upload path totals: nocopy={nocopy} copy={copy}");
    assert!(
        nocopy + copy > 0,
        "no real upload_block call was observed by this binary at all"
    );
}

/// The claim the shader landed without: that Metal, handed PACKED `Q4_K`
/// bytes, computes the same matmul the CPU computes from those same bytes
/// dequantized to `f32` first.
///
/// The oracle is deliberately the DEQUANTIZED-then-`f32` CPU path, not
/// `evaluate_quantized`. The CPU's quantized path routes through
/// `matmul_q4k_q8k_f32`, which quantizes the ACTIVATION to `Q8_K` as a
/// second lossy step; the shader does an exact unpack against an untouched
/// `f32` activation. Comparing against `evaluate_quantized` would fold that
/// unrelated activation-quantization error into this gate and force a
/// tolerance loose enough to hide a real unpack bug.
///
/// The weight buffer is never materialized as `f32` on the GPU side: the
/// bytes go up as bytes and `q4k_element` unpacks at the read. That is the
/// whole reason this path is worth having — a 7B `Q4_K_S` sweep is 3.784 GB
/// against 14.5 GB as `f16`, and decode is bandwidth-bound.
#[test]
fn metal_matmul_on_packed_q4k_weights_matches_the_dequantized_f32_cpu_path() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, dequantize, quantize};

    let rows: u32 = 5;
    let blocks_per_row = 3usize;
    let k = QK_K as u32 * blocks_per_row as u32;

    let activation: Vec<f32> = random_vec(13, k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();
    let weight_f32: Vec<f32> = random_vec(17, rows as usize * k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();

    let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
    for (row_f32, row_blocks) in weight_f32
        .chunks_exact(k as usize)
        .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
    {
        quantize(row_f32, row_blocks).expect("row length is a whole multiple of QK_K");
    }

    let mut dequantized: Vec<f32> = vec![0.0; rows as usize * k as usize];
    for (row_blocks, row_f32) in weight_blocks
        .chunks_exact(blocks_per_row * BLOCK_BYTES)
        .zip(dequantized.chunks_exact_mut(k as usize))
    {
        dequantize(row_blocks, row_f32).expect("a whole number of q4_k super-blocks");
    }

    let (packed_program, packed_sum) = q4k_matmul_program(rows, k, DType::UInt8);
    let metal = omega::execute(
        &packed_program,
        &[],
        &[
            QuantizedBlock::Q4K(&weight_blocks),
            QuantizedBlock::Float32(&activation),
        ],
        &[packed_sum],
    )
    .expect("metal executes a packed q4_k matmul on a real device");

    let (f32_program, f32_sum) = q4k_matmul_program(rows, k, DType::Float32);
    let cpu = evaluate(&f32_program, &[], &[&dequantized, &activation], &[f32_sum])
        .expect("dequantized f32 cpu matmul evaluates");

    let actual = metal.root();
    let expected = cpu.root();
    assert_eq!(
        actual.len(),
        rows as usize,
        "degenerate gate: no outputs compared"
    );
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(got.is_finite(), "metal produced a non-finite value: {got}");
        max_diff = max_diff.max((got - want).abs());
    }
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude;
    eprintln!(
        "packed-q4k metal vs dequantized-f32 cpu: rows={rows} k={k} \
         max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}"
    );
    // both sides fold the SAME dequantized values; the only spread is f32
    // summation order across 768 terms, so this is a float-noise bound, not
    // a quantization-error bound. A wrong unpack moves it by orders of
    // magnitude, not by ulps.
    assert!(
        relative < 1e-5,
        "packed unpack disagrees with the dequantized reference: relative={relative} max_diff={max_diff}"
    );
}

/// Same claim as [`metal_matmul_on_packed_q4k_weights_matches_the_dequantized_f32_cpu_path`],
/// but with a MANY-token activation ([`q4k_tiled_gemm_program`]'s `[k,
/// tokens]` shape) instead of that test's single-column `[k, 1]` one, so
/// that this is the arm that actually clears `TILED_GEMM_MIN_TOKENS` and
/// takes the `simdgroup_matrix`-tiled path (ROW 107) rather than the
/// row-blocked one. `rows=12`/`tokens=20` are both deliberately NOT whole
/// multiples of `TILE_DIM` (8), so a wrong boundary-tile mask on EITHER
/// axis would show up as a real numeric disagreement here, not just a
/// missing/extra write.
#[cfg(feature = "metal-tiled-gemm")]
#[test]
fn metal_matmul_on_packed_q4k_weights_matches_the_dequantized_f32_cpu_path_at_tile_scale() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, dequantize, quantize};

    let rows: u32 = 12;
    let blocks_per_row = 3usize;
    let k = QK_K as u32 * blocks_per_row as u32;
    let tokens: u32 = 20;

    let activation: Vec<f32> = random_vec(23, k as usize * tokens as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();
    let weight_f32: Vec<f32> = random_vec(29, rows as usize * k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();

    let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
    for (row_f32, row_blocks) in weight_f32
        .chunks_exact(k as usize)
        .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
    {
        quantize(row_f32, row_blocks).expect("row length is a whole multiple of QK_K");
    }

    let mut dequantized: Vec<f32> = vec![0.0; rows as usize * k as usize];
    for (row_blocks, row_f32) in weight_blocks
        .chunks_exact(blocks_per_row * BLOCK_BYTES)
        .zip(dequantized.chunks_exact_mut(k as usize))
    {
        dequantize(row_blocks, row_f32).expect("a whole number of q4_k super-blocks");
    }

    let (packed_program, packed_sum) = q4k_tiled_gemm_program(rows, k, tokens, DType::UInt8);
    let metal = omega::execute(
        &packed_program,
        &[],
        &[
            QuantizedBlock::Q4K(&weight_blocks),
            QuantizedBlock::Float32(&activation),
        ],
        &[packed_sum],
    )
    .expect("metal executes a tiled packed q4_k gemm on a real device");

    let (f32_program, f32_sum) = q4k_tiled_gemm_program(rows, k, tokens, DType::Float32);
    let cpu = evaluate(&f32_program, &[], &[&dequantized, &activation], &[f32_sum])
        .expect("dequantized f32 cpu gemm evaluates");

    let actual = metal.root();
    let expected = cpu.root();
    let element_count = rows as usize * tokens as usize;
    assert_eq!(
        actual.len(),
        element_count,
        "degenerate gate: no outputs compared"
    );
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(got.is_finite(), "metal produced a non-finite value: {got}");
        max_diff = max_diff.max((got - want).abs());
    }
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude;
    eprintln!(
        "tiled-gemm packed-q4k metal vs dequantized-f32 cpu: rows={rows} k={k} tokens={tokens} \
         max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}"
    );
    // Unlike the row-blocked path's own 1e-5 bound (both sides stay f32
    // throughout), the tiled path casts the dequantized weight to `half`
    // for `simdgroup_half8x8` -- a real, necessary precision cost of using
    // the hardware matrix unit (ggml's own `kernel_mul_mm_q4_K_f32`
    // instantiation makes the identical choice, `T = half` at
    // `ggml-metal.metal:6927`). `5e-3` is this suite's own established
    // ceiling for any f16-involved path
    // (`matmul_parity_is_within_f16_epsilon_of_the_f32_cpu_oracle`), not a
    // number invented for this row. Measured at landing: relative ~3.3e-5,
    // ~150x inside this bound.
    assert!(
        relative < 5e-3,
        "tiled GEMM packed unpack disagrees with the dequantized reference: relative={relative} max_diff={max_diff}"
    );
}

/// `[rows, k] x [k, tokens] -> [rows, tokens]` -- [`q4k_matmul_program`]'s
/// same shape generalized to a many-token activation, the shape
/// [`classify_tiled_gemm`](omega::msl) gates the tiled `simdgroup_matrix`
/// path on.
#[cfg(feature = "metal-tiled-gemm")]
fn q4k_tiled_gemm_program(
    rows: u32,
    k: u32,
    tokens: u32,
    weight_dtype: DType,
) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: weight_dtype,
            shape: vec![Extent::Static(rows), Extent::Static(k)],
            name: None,
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(k), Extent::Static(tokens)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(3, &[0, 2]))),
                (activation, IndexMap::Affine(projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
            // token axis FIRST, feature axis LAST -- the convention every
            // real matmul in `proxima-tensor/src/spec.rs` follows and
            // `classify_tiled_gemm`'s own doc requires for
            // `native_packed_layout`'s packed-stride reconstruction to
            // describe real bytes (see that function's doc, and
            // `omega/src/msl.rs`'s `tiled_gemm_op` test fixture doc).
            out_map: IndexMap::Affine(projection(3, &[1, 0])),
            keep: Keep::Reduce,
            name: Some("q4k_tiled_gemm".into()),
        }),
    );
    (program, sum)
}

/// Same claim as [`metal_matmul_on_packed_q4k_weights_matches_the_dequantized_f32_cpu_path`],
/// one codec over: the weight buffer is never materialized as `f32` on the
/// GPU side, `output.weight` — the ONE `Q6_K` tensor
/// `openchat-3.5-1210.Q4_K_S.gguf` carries and 60% of this checkpoint's GPU
/// time (`proxima-tensor/docs/discipline.md`'s own measurement) — is the
/// real shape this proves packed reads for.
#[test]
fn metal_matmul_on_packed_q6k_weights_matches_the_dequantized_f32_cpu_path() {
    use proxima_gguf::quant::q6_k::{BLOCK_BYTES, QK_K, dequantize, quantize};

    let rows: u32 = 5;
    let blocks_per_row = 3usize;
    let k = QK_K as u32 * blocks_per_row as u32;

    let activation: Vec<f32> = random_vec(13, k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();
    let weight_f32: Vec<f32> = random_vec(17, rows as usize * k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();

    let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
    for (row_f32, row_blocks) in weight_f32
        .chunks_exact(k as usize)
        .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
    {
        quantize(row_f32, row_blocks).expect("row length is a whole multiple of QK_K");
    }

    let mut dequantized: Vec<f32> = vec![0.0; rows as usize * k as usize];
    for (row_blocks, row_f32) in weight_blocks
        .chunks_exact(blocks_per_row * BLOCK_BYTES)
        .zip(dequantized.chunks_exact_mut(k as usize))
    {
        dequantize(row_blocks, row_f32).expect("a whole number of q6_k super-blocks");
    }

    let (packed_program, packed_sum) = q4k_matmul_program(rows, k, DType::UInt8);
    let metal = omega::execute(
        &packed_program,
        &[],
        &[
            QuantizedBlock::Q6K(&weight_blocks),
            QuantizedBlock::Float32(&activation),
        ],
        &[packed_sum],
    )
    .expect("metal executes a packed q6_k matmul on a real device");

    let (f32_program, f32_sum) = q4k_matmul_program(rows, k, DType::Float32);
    let cpu = evaluate(&f32_program, &[], &[&dequantized, &activation], &[f32_sum])
        .expect("dequantized f32 cpu matmul evaluates");

    let actual = metal.root();
    let expected = cpu.root();
    assert_eq!(
        actual.len(),
        rows as usize,
        "degenerate gate: no outputs compared"
    );
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(got.is_finite(), "metal produced a non-finite value: {got}");
        max_diff = max_diff.max((got - want).abs());
    }
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude;
    eprintln!(
        "packed-q6k metal vs dequantized-f32 cpu: rows={rows} k={k} \
         max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}"
    );
    assert!(
        relative < 1e-5,
        "packed unpack disagrees with the dequantized reference: relative={relative} max_diff={max_diff}"
    );
}

/// Same claim as [`metal_matmul_on_packed_q4k_weights_matches_the_dequantized_f32_cpu_path`],
/// a third codec over: the weight buffer is never materialized as `f32` on
/// the GPU side. `blk.{n}.attn_v.weight`/`blk.{n}.ffn_down.weight` (4 layers
/// each) are the only `Q5_K` tensors `openchat-3.5-1210.Q4_K_S.gguf`
/// carries, and ROW 92's isolated per-op profiling measured those 8 ops at
/// over half of one decode step's GPU time before this landing, once told
/// apart from the 56 already-fast `Q4_K` ops sharing their two families.
#[test]
fn metal_matmul_on_packed_q5k_weights_matches_the_dequantized_f32_cpu_path() {
    use proxima_gguf::quant::q5_k::{BLOCK_BYTES, QK_K, dequantize, quantize};

    let rows: u32 = 5;
    let blocks_per_row = 3usize;
    let k = QK_K as u32 * blocks_per_row as u32;

    let activation: Vec<f32> = random_vec(13, k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();
    let weight_f32: Vec<f32> = random_vec(17, rows as usize * k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();

    let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
    for (row_f32, row_blocks) in weight_f32
        .chunks_exact(k as usize)
        .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
    {
        quantize(row_f32, row_blocks).expect("row length is a whole multiple of QK_K");
    }

    let mut dequantized: Vec<f32> = vec![0.0; rows as usize * k as usize];
    for (row_blocks, row_f32) in weight_blocks
        .chunks_exact(blocks_per_row * BLOCK_BYTES)
        .zip(dequantized.chunks_exact_mut(k as usize))
    {
        dequantize(row_blocks, row_f32).expect("a whole number of q5_k super-blocks");
    }

    let (packed_program, packed_sum) = q4k_matmul_program(rows, k, DType::UInt8);
    let metal = omega::execute(
        &packed_program,
        &[],
        &[
            QuantizedBlock::Q5K(&weight_blocks),
            QuantizedBlock::Float32(&activation),
        ],
        &[packed_sum],
    )
    .expect("metal executes a packed q5_k matmul on a real device");

    let (f32_program, f32_sum) = q4k_matmul_program(rows, k, DType::Float32);
    let cpu = evaluate(&f32_program, &[], &[&dequantized, &activation], &[f32_sum])
        .expect("dequantized f32 cpu matmul evaluates");

    let actual = metal.root();
    let expected = cpu.root();
    assert_eq!(
        actual.len(),
        rows as usize,
        "degenerate gate: no outputs compared"
    );
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(got.is_finite(), "metal produced a non-finite value: {got}");
        max_diff = max_diff.max((got - want).abs());
    }
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude;
    eprintln!(
        "packed-q5k metal vs dequantized-f32 cpu: rows={rows} k={k} \
         max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}"
    );
    assert!(
        relative < 1e-5,
        "packed unpack disagrees with the dequantized reference: relative={relative} max_diff={max_diff}"
    );
}

/// `[rows, k] x [k, 1] -> [rows, 1]`, with the weight operand's declared
/// dtype as a parameter: `UInt8` marks "this operand arrives as packed
/// bytes" (the same marker `cpu::quantized_matmul_program` uses), `Float32`
/// builds the identical program for the dequantized oracle.
fn q4k_matmul_program(rows: u32, k: u32, weight_dtype: DType) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: weight_dtype,
            shape: vec![Extent::Static(rows), Extent::Static(k)],
            name: None,
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(k), Extent::Static(1)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(3, &[0, 2]))),
                (activation, IndexMap::Affine(projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("q4k_matmul".into()),
        }),
    );
    (program, sum)
}

/// `[rows, k] x [k, 1] -> [rows, 1]`, the same shape [`q4k_matmul_program`]
/// builds, but with the compute dtype (activation, the elementwise product,
/// and the reduce) as its own axis independent of the weight operand's
/// declared dtype -- `weight_dtype` stays the packed-bytes marker
/// (`DType::UInt8`) for every packed codec regardless of `compute_dtype`,
/// the same way `reject_unsupported_gpu_dtype` treats it, and is
/// `DType::Float32` only for the plain (unpacked) `float32` cell.
fn codec_matmul_program(
    rows: u32,
    k: u32,
    weight_dtype: DType,
    compute_dtype: DType,
) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: weight_dtype,
            shape: vec![Extent::Static(rows), Extent::Static(k)],
            name: None,
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: compute_dtype,
            shape: vec![Extent::Static(k), Extent::Static(1)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: compute_dtype,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(3, &[0, 2]))),
                (activation, IndexMap::Affine(projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: compute_dtype,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("codec_matmul".into()),
        }),
    );
    (program, sum)
}

/// The stratified matrix the coordinator asked for: [`QuantizedBlock`]'s
/// codec axis crossed with the compute dtype axis, one parameterized body
/// instead of one test per cell. `float32`/`q4_k`/`q5_k`/`q6_k`/`q8_0`/`q4_0`
/// all have a real Metal unpack kernel today -- `q8_0`/`q4_0`'s are the
/// fully generic per-element path (`omega::msl::Q8_0_UNPACK_MSL`/
/// `Q4_0_UNPACK_MSL`), not the row-blocked fast path the K-quants take,
/// since their 32-element flat blocks have no analogue to the K-quants'
/// shared 256-element super-block.
///
/// The oracle is always the plain-`f32` CPU path, weight dequantized first
/// when the cell's codec is packed -- the same reasoning
/// `metal_matmul_on_packed_q4k_weights_matches_the_dequantized_f32_cpu_path`
/// gives for isolating the unpack/dtype error from `Q8_K` activation
/// quantization noise. Tolerance is RELATIVE to the reference's own
/// magnitude and keyed to `compute_dtype`: `Float32` compounds only
/// summation-order float noise (tight bound), `Float16` additionally rounds
/// every intermediate through 10 mantissa bits, the same
/// `matmul_parity_is_within_f16_epsilon_of_the_f32_cpu_oracle` measured at
/// `~1.6e-3` worst case over a 23-term dot product — this test's dot
/// product is deeper (768 terms), so its `Float16` epsilon is looser still.
#[proxima::test(runtime = "tokio")]
#[case::float32_at_float32("float32", DType::Float32, 1e-5)]
#[case::float32_at_float16("float32", DType::Float16, 1e-2)]
#[case::q4k_at_float32("q4_k", DType::Float32, 1e-5)]
#[case::q4k_at_float16("q4_k", DType::Float16, 1e-2)]
#[case::q5k_at_float32("q5_k", DType::Float32, 1e-5)]
#[case::q5k_at_float16("q5_k", DType::Float16, 1e-2)]
#[case::q6k_at_float32("q6_k", DType::Float32, 1e-5)]
#[case::q6k_at_float16("q6_k", DType::Float16, 1e-2)]
#[case::q8_0_at_float32("q8_0", DType::Float32, 1e-5)]
#[case::q8_0_at_float16("q8_0", DType::Float16, 1e-2)]
#[case::q4_0_at_float32("q4_0", DType::Float32, 1e-5)]
#[case::q4_0_at_float16("q4_0", DType::Float16, 1e-2)]
async fn metal_matmul_parity_across_codec_and_dtype(
    #[case] codec: &str,
    #[case] compute_dtype: DType,
    #[case] epsilon: f32,
) {
    use proxima_gguf::quant::q4_0;
    use proxima_gguf::quant::q4_k;
    use proxima_gguf::quant::q5_k;
    use proxima_gguf::quant::q6_k;
    use proxima_gguf::quant::q8_0;

    let rows: u32 = 5;
    let blocks_per_row = 3usize;
    // `Q4_K`/`Q5_K`/`Q6_K` share one super-block element count (`QK_K ==
    // 256`, every codec's own module doc) so this shape's `k` is valid for
    // any of them.
    let k = q4_k::QK_K as u32 * blocks_per_row as u32;

    let weight_f32: Vec<f32> = random_vec(17, rows as usize * k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();
    let activation: Vec<f32> = random_vec(13, k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();

    let packed_weight_blocks: Option<Vec<u8>> = match codec {
        "float32" => None,
        "q4_k" => {
            let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * q4_k::BLOCK_BYTES];
            for (row_f32, row_blocks) in weight_f32
                .chunks_exact(k as usize)
                .zip(weight_blocks.chunks_exact_mut(blocks_per_row * q4_k::BLOCK_BYTES))
            {
                q4_k::quantize(row_f32, row_blocks)
                    .expect("row length is a whole multiple of QK_K");
            }
            Some(weight_blocks)
        }
        "q5_k" => {
            let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * q5_k::BLOCK_BYTES];
            for (row_f32, row_blocks) in weight_f32
                .chunks_exact(k as usize)
                .zip(weight_blocks.chunks_exact_mut(blocks_per_row * q5_k::BLOCK_BYTES))
            {
                q5_k::quantize(row_f32, row_blocks)
                    .expect("row length is a whole multiple of QK_K");
            }
            Some(weight_blocks)
        }
        "q6_k" => {
            let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * q6_k::BLOCK_BYTES];
            for (row_f32, row_blocks) in weight_f32
                .chunks_exact(k as usize)
                .zip(weight_blocks.chunks_exact_mut(blocks_per_row * q6_k::BLOCK_BYTES))
            {
                q6_k::quantize(row_f32, row_blocks)
                    .expect("row length is a whole multiple of QK_K");
            }
            Some(weight_blocks)
        }
        "q8_0" => {
            // `Q8_0`'s block is 32 elements, not the K-quants' 256 --
            // `blocks_per_row` above counts K-quant super-blocks, so this
            // codec's own per-row block count is `k / QK8_0` instead.
            let q8_blocks_per_row = k as usize / q8_0::QK8_0;
            let mut weight_blocks =
                vec![0u8; rows as usize * q8_blocks_per_row * q8_0::BLOCK_BYTES];
            for (row_f32, row_blocks) in weight_f32
                .chunks_exact(k as usize)
                .zip(weight_blocks.chunks_exact_mut(q8_blocks_per_row * q8_0::BLOCK_BYTES))
            {
                q8_0::quantize(row_f32, row_blocks)
                    .expect("row length is a whole multiple of QK8_0");
            }
            Some(weight_blocks)
        }
        "q4_0" => {
            // `Q4_0`'s block is 32 elements too -- same axis reasoning as
            // `q8_0` above, its own per-row block count, not the K-quant
            // `blocks_per_row`.
            let q4_0_blocks_per_row = k as usize / q4_0::QK4_0;
            let mut weight_blocks =
                vec![0u8; rows as usize * q4_0_blocks_per_row * q4_0::BLOCK_BYTES];
            for (row_f32, row_blocks) in weight_f32
                .chunks_exact(k as usize)
                .zip(weight_blocks.chunks_exact_mut(q4_0_blocks_per_row * q4_0::BLOCK_BYTES))
            {
                q4_0::quantize(row_f32, row_blocks)
                    .expect("row length is a whole multiple of QK4_0");
            }
            Some(weight_blocks)
        }
        other => panic!("unhandled codec case in this matrix: {other}"),
    };

    let weight_dtype = if packed_weight_blocks.is_some() {
        DType::UInt8
    } else {
        DType::Float32
    };
    let (program, sum) = codec_matmul_program(rows, k, weight_dtype, compute_dtype);
    let weight_block = match (&packed_weight_blocks, codec) {
        (Some(bytes), "q4_k") => QuantizedBlock::Q4K(bytes),
        (Some(bytes), "q5_k") => QuantizedBlock::Q5K(bytes),
        (Some(bytes), "q6_k") => QuantizedBlock::Q6K(bytes),
        (Some(bytes), "q8_0") => QuantizedBlock::Q8_0(bytes),
        (Some(bytes), "q4_0") => QuantizedBlock::Q4_0(bytes),
        (Some(_), other) => panic!("unhandled codec case in this matrix: {other}"),
        (None, _) => QuantizedBlock::Float32(&weight_f32),
    };
    let metal = omega::execute(
        &program,
        &[],
        &[weight_block, QuantizedBlock::Float32(&activation)],
        &[sum],
    )
    .unwrap_or_else(|error| {
        panic!("{codec}@{compute_dtype:?}: metal executes on a real device: {error}")
    });

    let cpu_weight: Vec<f32> = match (&packed_weight_blocks, codec) {
        (None, _) => weight_f32.clone(),
        (Some(bytes), "q4_k") => {
            let mut dequantized = vec![0.0f32; rows as usize * k as usize];
            for (row_blocks, row_f32) in bytes
                .chunks_exact(blocks_per_row * q4_k::BLOCK_BYTES)
                .zip(dequantized.chunks_exact_mut(k as usize))
            {
                q4_k::dequantize(row_blocks, row_f32).expect("a whole number of q4_k super-blocks");
            }
            dequantized
        }
        (Some(bytes), "q5_k") => {
            let mut dequantized = vec![0.0f32; rows as usize * k as usize];
            for (row_blocks, row_f32) in bytes
                .chunks_exact(blocks_per_row * q5_k::BLOCK_BYTES)
                .zip(dequantized.chunks_exact_mut(k as usize))
            {
                q5_k::dequantize(row_blocks, row_f32).expect("a whole number of q5_k super-blocks");
            }
            dequantized
        }
        (Some(bytes), "q6_k") => {
            let mut dequantized = vec![0.0f32; rows as usize * k as usize];
            for (row_blocks, row_f32) in bytes
                .chunks_exact(blocks_per_row * q6_k::BLOCK_BYTES)
                .zip(dequantized.chunks_exact_mut(k as usize))
            {
                q6_k::dequantize(row_blocks, row_f32).expect("a whole number of q6_k super-blocks");
            }
            dequantized
        }
        (Some(bytes), "q8_0") => {
            let q8_blocks_per_row = k as usize / q8_0::QK8_0;
            let mut dequantized = vec![0.0f32; rows as usize * k as usize];
            for (row_blocks, row_f32) in bytes
                .chunks_exact(q8_blocks_per_row * q8_0::BLOCK_BYTES)
                .zip(dequantized.chunks_exact_mut(k as usize))
            {
                q8_0::dequantize(row_blocks, row_f32).expect("a whole number of q8_0 blocks");
            }
            dequantized
        }
        (Some(bytes), "q4_0") => {
            let q4_0_blocks_per_row = k as usize / q4_0::QK4_0;
            let mut dequantized = vec![0.0f32; rows as usize * k as usize];
            for (row_blocks, row_f32) in bytes
                .chunks_exact(q4_0_blocks_per_row * q4_0::BLOCK_BYTES)
                .zip(dequantized.chunks_exact_mut(k as usize))
            {
                q4_0::dequantize(row_blocks, row_f32).expect("a whole number of q4_0 blocks");
            }
            dequantized
        }
        (Some(_), other) => panic!("unhandled codec case in this matrix: {other}"),
    };
    let (f32_program, f32_sum) = codec_matmul_program(rows, k, DType::Float32, DType::Float32);
    let cpu = evaluate(&f32_program, &[], &[&cpu_weight, &activation], &[f32_sum])
        .expect("f32 cpu oracle evaluates");

    let actual = metal.root();
    let expected = cpu.root();
    assert_eq!(
        actual.len(),
        rows as usize,
        "degenerate gate: no outputs compared"
    );
    assert_eq!(actual.len(), expected.len());

    let max_diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(&got, &want)| (got - want).abs())
        .fold(0.0f32, f32::max);
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude;
    eprintln!(
        "{codec}@{compute_dtype:?}: relative={relative} epsilon={epsilon} (max_diff={max_diff} max_magnitude={max_magnitude})"
    );
    assert!(
        relative <= epsilon,
        "{codec}@{compute_dtype:?}: relative diff {relative} exceeds {epsilon} -- max_diff={max_diff} \
         max_magnitude={max_magnitude}"
    );
}

// `metal_rejects_a_codec_with_no_unpack_kernel_and_names_it` (single case:
// `q8_0`) retired here: `Q8_0` now has a real Metal unpack kernel
// (`omega::msl::Q8_0_UNPACK_MSL`), covered instead by
// `metal_matmul_parity_across_codec_and_dtype`'s `q8_0_at_float32`/
// `q8_0_at_float16` cases above. No `QuantizedBlock` variant remains
// unsupported on Metal, so this contract has no live case to assert against
// -- a future codec added to `QuantizedBlock` ahead of its own MSL unpack
// kernel should get this test (and a single `#[case]`) back.
