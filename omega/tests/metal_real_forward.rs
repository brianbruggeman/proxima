//! The gate that ROWS 69-78 never had: the REAL forward graph, end to end,
//! CPU against device.
//!
//! Every other GPU number in this workspace comes from a synthetic matvec.
//! This binds `mistral_cached_forward_program` — the same builder
//! `proxima-model-interop` uses for a real token — and runs it through both
//! evaluators on identical named blocks.
//!
//! The architecture is scaled down (2 layers, 64-wide) so it runs in a test,
//! but the OP SET and the graph shape are the production ones: embedding
//! gather, RMSNorm, RoPE, grouped-query attention with a KV cache, SwiGLU,
//! and the output projection. That is the coverage that matters here; the
//! full-size numbers are what the probe next to this measures.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[cfg(feature = "metal-buffer-pool")]
use proxima_tensor::Reduce;
use proxima_tensor::cpu::evaluate_quantized_named_with_scratch;
use proxima_tensor::{BoundOpKind, NodeId, NumericPolicy, bind, infer};
#[cfg(feature = "metal-buffer-pool")]
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, Op, QuantizedBlock, ReduceInit, ScalarOp, append, projection,
};

mod support;
use support::{as_named_blocks, real_forward_fixture, real_forward_fixture_with_cached_len};

#[test]
fn metal_runs_the_real_forward_graph_and_agrees_with_the_cpu() {
    const VOCAB: usize = 64;

    let (program, symbols, roots, owned) = real_forward_fixture();
    let named = as_named_blocks(&owned);

    let mut free_buffers: Vec<Vec<f32>> = Vec::new();
    let mut validated = None;
    let cpu = evaluate_quantized_named_with_scratch(
        &program,
        &symbols,
        &named,
        &roots,
        &mut free_buffers,
        &mut validated,
    )
    .expect("cpu runs the real forward");

    let plan = omega::plan_named(&program, &symbols, &named, &roots, NumericPolicy::default())
        .expect("metal plans the real forward");
    let metal = omega::execute_plan_named(&plan, &named)
        .expect("metal runs the real forward on a real device");

    let expected = cpu.root();
    let actual = metal.root();
    assert_eq!(
        actual.len(),
        VOCAB,
        "degenerate gate: logits must be one row of the vocabulary"
    );
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(got.is_finite(), "metal produced a non-finite logit: {got}");
        max_diff = max_diff.max((got - want).abs());
    }
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude.max(f32::MIN_POSITIVE);
    eprintln!(
        "real forward: max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}"
    );
    assert!(
        relative < 1e-4,
        "metal disagrees with the cpu on the real forward: relative={relative} max_diff={max_diff}"
    );
}

/// The gate `metal_runs_the_real_forward_graph_and_agrees_with_the_cpu`
/// cannot be: that test's `cached_len` is always zero (`symbols = [1, 0]`),
/// so every fold over the online-softmax combine's cached-block `t` axis
/// degenerates to its `ReduceInit` identity and never actually reduces
/// anything. This test sets `cached_len = 5` and requests EVERY node in the
/// program as an output on both backends (an output request bypasses
/// `bind`'s elementwise-into-reduce fusion entirely — see
/// `bind.rs`'s `requesting_the_intermediate_elementwise_op_as_an_output_prevents_fusion`
/// -- so every intermediate materializes and can be diffed node-by-node),
/// reporting the FIRST node id whose Metal output disagrees with the CPU's
/// past a real floating-point tolerance.
#[test]
fn metal_agrees_with_cpu_on_a_nonempty_kv_cache() {
    const CACHED_LEN: u64 = 5;

    let (program, symbols, _roots, owned) = real_forward_fixture_with_cached_len(CACHED_LEN);
    let named = as_named_blocks(&owned);
    let all_nodes: Vec<NodeId> = (0..program.len() as u32).map(NodeId).collect();

    let mut free_buffers: Vec<Vec<f32>> = Vec::new();
    let mut validated = None;
    let cpu = evaluate_quantized_named_with_scratch(
        &program,
        &symbols,
        &named,
        &all_nodes,
        &mut free_buffers,
        &mut validated,
    )
    .expect("cpu runs the real forward with a non-empty cache");

    let plan = omega::plan_named(
        &program,
        &symbols,
        &named,
        &all_nodes,
        NumericPolicy::default(),
    )
    .expect("metal plans the real forward with a non-empty cache");
    let metal = omega::execute_plan_named(&plan, &named)
        .expect("metal runs the real forward with a non-empty cache on a real device");

    let mut first_divergence: Option<(NodeId, f32, f32, f32)> = None;
    for &node in &all_nodes {
        let Some((cpu_data, _cpu_shape)) = cpu.get(node) else {
            continue;
        };
        let Some((metal_data, _metal_shape)) = metal.get(node) else {
            continue;
        };
        assert_eq!(
            cpu_data.len(),
            metal_data.len(),
            "node {node:?} shape disagreement: cpu={} metal={}",
            cpu_data.len(),
            metal_data.len()
        );
        let max_magnitude = cpu_data
            .iter()
            .map(|value| value.abs())
            .fold(0.0f32, f32::max);
        for (&got, &want) in metal_data.iter().zip(cpu_data.iter()) {
            let diff = (got - want).abs();
            let relative = diff / max_magnitude.max(f32::MIN_POSITIVE);
            if relative > 1e-4 && first_divergence.is_none() {
                first_divergence = Some((node, got, want, relative));
            }
        }
    }

    if let Some((node, got, want, relative)) = first_divergence {
        let op = &program[node.0 as usize];
        panic!(
            "metal first diverges from cpu at node {node:?} op={op:?}: metal={got} cpu={want} relative={relative}"
        );
    }
}

#[test]
fn fused_cached_attention_root_agrees_between_cpu_and_metal() {
    let (program, symbols, roots, owned) = real_forward_fixture_with_cached_len(5);
    let output_roots = [roots[0]];
    let named = as_named_blocks(&owned);
    let shapes = infer(&program, &symbols).expect("cached fixture infers");
    let resolved = bind(&program, &shapes, &output_roots, NumericPolicy::default())
        .expect("cached fixture binds");
    assert!(
        resolved
            .iter()
            .any(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
    );

    let mut free_buffers = Vec::new();
    let mut validated = None;
    let cpu = evaluate_quantized_named_with_scratch(
        &program,
        &symbols,
        &named,
        &output_roots,
        &mut free_buffers,
        &mut validated,
    )
    .expect("cpu runs the fused cached root");
    let plan = omega::plan_named(
        &program,
        &symbols,
        &named,
        &output_roots,
        NumericPolicy::default(),
    )
    .expect("metal plans the fused cached root");
    let metal = omega::execute_plan_named(&plan, &named).expect("metal runs the fused cached root");
    let max_diff = cpu
        .root()
        .iter()
        .zip(metal.root())
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0f32, f32::max);
    eprintln!("fused cached root: max_diff={max_diff}");
    assert!(max_diff < 1e-4, "fused cached root max_diff={max_diff}");
}

/// Runs the SAME resolved [`omega::Plan`] against the SAME blocks twice in a
/// row and asserts the second call's output is bit-identical to the first's,
/// node by node, over EVERY node in the program (the same
/// `requesting-every-node-bypasses-fusion` shape
/// `metal_agrees_with_cpu_on_a_nonempty_kv_cache` uses, so intermediates
/// materialize as real device buffers rather than fusing away).
///
/// This is the gate a stale-buffer bug would fail and a correct one cannot:
/// a serving loop calls [`omega::execute_plan_named`] on one [`omega::Plan`]
/// repeatedly, so an op-output buffer pool ([`metal-buffer-pool`] feature)
/// that hands back a buffer still carrying a PRIOR call's leftover contents
/// -- instead of one Metal's own compute dispatch fully overwrites this
/// call -- would only ever surface here, on the second run, never on a
/// single-run parity test. Compiled and run unconditionally (not gated to
/// `metal-buffer-pool`) because determinism across repeated calls is a
/// property of `execute_plan` itself, independent of that feature.
#[test]
fn running_the_same_plan_twice_reproduces_the_first_run_exactly() {
    const CACHED_LEN: u64 = 5;

    let (program, symbols, _roots, owned) = real_forward_fixture_with_cached_len(CACHED_LEN);
    let named = as_named_blocks(&owned);
    let all_nodes: Vec<NodeId> = (0..program.len() as u32).map(NodeId).collect();

    let plan = omega::plan_named(
        &program,
        &symbols,
        &named,
        &all_nodes,
        NumericPolicy::default(),
    )
    .expect("metal plans the real forward with a non-empty cache");

    let first = omega::execute_plan_named(&plan, &named).expect("first metal run succeeds");
    let second = omega::execute_plan_named(&plan, &named).expect("second metal run succeeds");

    for &node in &all_nodes {
        let Some((first_data, first_shape)) = first.get(node) else {
            continue;
        };
        let Some((second_data, second_shape)) = second.get(node) else {
            continue;
        };
        assert_eq!(
            first_shape, second_shape,
            "node {node:?} shape changed between two runs of the same plan"
        );
        assert_eq!(
            first_data, second_data,
            "node {node:?} disagrees between two runs of the identical plan: a pooled buffer \
             served stale contents instead of this call's own dispatch output"
        );
    }
}

/// A single symbolic `(m, k) x (k, n) -> (m, n)` matmul program, `m`
/// SYMBOLIC and bound fresh per call via `symbols` -- the same shape
/// `metal_parity.rs`'s own `symbolic_extent_parity_holds_across_two_different_bindings`
/// test uses to prove Metal handles a runtime-bound extent, reused here to
/// drive [`omega::execute`] (which re-`plan`s every call, exactly like a
/// serving loop re-binding a growing KV-cache extent every token) through
/// MANY distinct output sizes in one thread.
#[cfg(feature = "metal-buffer-pool")]
fn growing_matmul_program(k: u32, n: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Symbolic(0), Extent::Static(k)],
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
            name: Some("growing_matmul".into()),
        }),
    );
    (program, sum)
}

/// Drives [`omega::execute`] across 48 DISTINCT `m` values (mimicking a
/// cached-attention extent that grows by one token every call) and asserts
/// [`omega::metal::output_buffer_pool_len`] -- the pool's total retained
/// buffer count across every `(bucket, dtype)` slot -- stays FAR below 48
/// afterward, rather than growing one entry per distinct size the way the
/// exact-size-keyed pool this feature shipped with first did.
///
/// `k=3, n=5` fixed, `m` symbolic: the reduce output (`sum`, shape `(m, n)`)
/// and its own operand (`product`, shape `(m, k, n)`) both grow their byte
/// length linearly with `m`, so this exercises the pool exactly the way a
/// growing KV extent does -- a handful of ops whose OUTPUT size scales with
/// a runtime-bound dimension, replanned/reexecuted every call.
#[test]
#[cfg(feature = "metal-buffer-pool")]
fn running_a_growing_extent_through_the_pool_keeps_retained_buffers_bounded() {
    const K: u32 = 3;
    const N: u32 = 5;
    const DISTINCT_SIZES: u32 = 48;

    let (program, sum) = growing_matmul_program(K, N);

    for m in 1..=DISTINCT_SIZES {
        let lhs: Vec<f32> = (0..m * K).map(|value| (value % 7) as f32).collect();
        let rhs: Vec<f32> = (0..K * N).map(|value| (value % 5) as f32).collect();
        let symbols = [u64::from(m)];

        let result = omega::execute(
            &program,
            &symbols,
            &[QuantizedBlock::Float32(&lhs), QuantizedBlock::Float32(&rhs)],
            &[sum],
        )
        .unwrap_or_else(|error| panic!("metal executes the growing matmul at m={m}: {error}"));
        assert_eq!(
            result.root().len(),
            (m * N) as usize,
            "m={m}: sum output element count disagrees with the shape it was planned for"
        );
    }

    let retained = omega::metal::output_buffer_pool_len();
    println!("output_buffer_pool_len after {DISTINCT_SIZES} distinct output sizes: {retained}");
    assert!(
        retained < 24,
        "pool retained {retained} buffers across {DISTINCT_SIZES} distinct output sizes -- \
         bucketing should collapse most of those {DISTINCT_SIZES} exact sizes into a handful \
         of shared power-of-two buckets, not grow roughly one-for-one with distinct sizes"
    );
    assert!(
        retained > 0,
        "pool retained 0 buffers after {DISTINCT_SIZES} real executions -- reclaim is not \
         running at all, which would make this test vacuous"
    );
}
