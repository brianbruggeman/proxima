//! GPU-vs-CPU parity gate for the portable `wgpu`/WGSL backend
//! (`omega::wgpu_driver`, reached through `omega::backend` exactly the way
//! `backend_parity.rs` reaches Metal), run through the SAME
//! `plan_named`/`execute_plan_named` wrapper.
//!
//! # Why this is not `backend_parity.rs`'s own `real_forward_fixture`
//!
//! That fixture (`support::real_forward_fixture`) is a full cached-attention
//! transformer forward: it needs `Op::Elementwise`'s gather form
//! (`embedding_lookup`'s `IndexMap::Computed`) to bind the token embedding
//! table, RoPE, and a causal mask. `omega::wgsl`'s v1 scope is explicitly
//! elementwise + `Keep::Reduce` + `Keep::Scan` with **no gather** (see that
//! module's own doc) — running the real fixture through the wgpu backend
//! would fail on the embedding lookup before ever reaching a matmul.
//!
//! This test instead builds a standalone two-layer MLP
//! (`matmul -> erf -> matmul -> tanh -> cumsum`) that stays entirely inside
//! v1's op set while still exercising every kernel shape that set covers:
//! `Keep::Reduce` (both matmuls, `Add`-reduce over a `Multiply` body — the
//! same shape a real matmul takes), an elementwise `Erf` (the ported
//! polynomial) and `Tanh`, and `Keep::Scan` (a per-row cumulative sum).

#![cfg(all(feature = "cpu", feature = "wgpu-backend"))]
// every expect below runs against data this test just built or a real
// device call; a failure there IS the test failing, not a case to recover.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use omega::backend::{Backend, execute_plan_named, plan_named};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp,
    append, map,
};

const BATCH: u32 = 4;
const IN_FEATURES: u32 = 8;
const HIDDEN: u32 = 16;
const OUT_FEATURES: u32 = 8;

/// `m`/`k`/`n` name the matmul shape for callers' readability; the
/// projections below encode it structurally, so the function itself never
/// reads them.
fn append_matmul(program: &mut Vec<Op>, lhs: NodeId, rhs: NodeId, _m: u32, _k: u32, _n: u32) -> NodeId {
    let product = append(
        program,
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
        program,
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
    )
}

/// `matmul(x, w1) -> erf -> matmul(_, w2) -> tanh -> cumsum(last axis)`,
/// entirely within `omega::wgsl`'s v1 op set — see the module doc.
fn two_layer_mlp_fixture() -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let x = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(BATCH), Extent::Static(IN_FEATURES)],
            name: Some("x".into()),
        },
    );
    let w1 = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(IN_FEATURES), Extent::Static(HIDDEN)],
            name: Some("w1".into()),
        },
    );
    let hidden = append_matmul(&mut program, x, w1, BATCH, IN_FEATURES, HIDDEN);
    let hidden_act = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Erf,
            operands: vec![(hidden, IndexMap::Affine(map::projection(2, &[0, 1])))],
            name: None,
        },
    );
    let w2 = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(HIDDEN), Extent::Static(OUT_FEATURES)],
            name: Some("w2".into()),
        },
    );
    let output = append_matmul(&mut program, hidden_act, w2, BATCH, HIDDEN, OUT_FEATURES);
    let output_act = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Tanh,
            operands: vec![(output, IndexMap::Affine(map::projection(2, &[0, 1])))],
            name: None,
        },
    );
    let cumsum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: output_act,
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            keep: Keep::Scan,
            name: None,
        }),
    );
    (program, cumsum)
}

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

#[test]
fn the_two_layer_mlp_runs_on_wgpu_at_cpu_parity() {
    let (program, _root) = two_layer_mlp_fixture();

    let x = random_vec(1, (BATCH * IN_FEATURES) as usize);
    let w1 = random_vec(2, (IN_FEATURES * HIDDEN) as usize);
    let w2 = random_vec(3, (HIDDEN * OUT_FEATURES) as usize);
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("x", QuantizedBlock::Float32(&x)),
        ("w1", QuantizedBlock::Float32(&w1)),
        ("w2", QuantizedBlock::Float32(&w2)),
    ];

    let mut cpu_plan = plan_named(Backend::Cpu, &program, &[], &named, &[])
        .expect("omega::backend plans the mlp on cpu");
    let cpu = execute_plan_named(&mut cpu_plan, &named).expect("omega::backend runs the mlp on cpu");

    let mut wgpu_plan = plan_named(Backend::Wgpu, &program, &[], &named, &[])
        .expect("omega::backend plans the mlp on wgpu");
    let wgpu = execute_plan_named(&mut wgpu_plan, &named).expect("omega::backend runs the mlp on a real device");

    let expected = cpu.root();
    let actual = wgpu.root();
    assert_eq!(
        actual.len(),
        (BATCH * OUT_FEATURES) as usize,
        "degenerate gate: the cumsum output must be one row per batch element"
    );
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(got.is_finite(), "wgpu, via the wrapper, produced a non-finite value: {got}");
        max_diff = max_diff.max((got - want).abs());
    }
    let max_magnitude = expected.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude.max(f32::MIN_POSITIVE);
    eprintln!("wgpu mlp parity: max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}");
    assert!(
        relative < 1e-4,
        "omega::backend's cpu and wgpu arms disagree on the mlp: relative={relative} max_diff={max_diff}"
    );
}
