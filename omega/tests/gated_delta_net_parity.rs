//! Metal-vs-CPU parity for the fused `BoundOpKind::GatedDeltaNet` kernel
//! (`omega::msl::render_gated_delta_net`) -- both backends run the exact
//! same bound program (`gated-delta-net-fusion`'s matcher output), so this
//! is the same "one oracle, two backends" shape
//! `gdn_sequence_projection_parity.rs` already uses for the sequence tail.
//!
//! `proxima_tensor::cpu::run_gated_delta_net` calls
//! `proxima_tensor::gdn::run_gdn_prefill_scan` directly (`bind.rs`'s own
//! doc on `BoundOpKind::GatedDeltaNet`), the same scalar per-token loop
//! `render_gated_delta_net`'s own doc says this kernel is ported from -- so
//! CPU and Metal compute the exact same arithmetic in the exact same order,
//! and a real divergence here means the Metal kernel's addressing, not its
//! numerics, is wrong.
//!
//! Also proves ROW 547's own second output: `state_out` matches CPU exactly
//! like `out` does, and `state_in`'s own bytes come back byte-identical to
//! what was uploaded, proving the kernel never wrote into it (the bug this
//! row fixed -- `render_gated_delta_net` used to write the updated state
//! back into `state_in`'s own buffer in place).

#![cfg(all(
    feature = "metal",
    feature = "gated-delta-net-fusion",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::spec::{append_qwen35_delta_net_step, elementwise};
use proxima_tensor::{BoundOpKind, DType, Extent, NodeId, NumericPolicy, Op, ScalarOp, append};

/// Deterministic, non-degenerate fill -- real-shaped small values, never
/// all-zero/all-one filler (guiding-principle 9).
struct Lcg(u64);

impl Lcg {
    fn next_unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (((self.0 >> 33) as u32) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn leaf(program: &mut Vec<Op>, shape: &[usize]) -> NodeId {
    append(
        program,
        Op::Input {
            dtype: DType::Float32,
            shape: shape
                .iter()
                .map(|extent| Extent::Static(*extent as u32))
                .collect(),
            name: None,
        },
    )
}

/// The pre-`repeat_kv_heads` broadcast pattern the matcher walks past
/// (`gated_delta_net_candidates`'s own doc, `proxima-tensor/src/bind.rs`) --
/// rebuilt here with the public [`elementwise`] builder since the crate's
/// own `broadcast_kv_heads` test helper is private to `bind.rs`'s
/// `#[cfg(test)]` module.
fn broadcast_kv_heads(program: &mut Vec<Op>, x: NodeId, kv_heads: u32, group: u32) -> NodeId {
    let donor = append(
        program,
        Op::Constant {
            dtype: DType::Float32,
            shape: vec![Extent::Static(kv_heads), Extent::Static(group)],
            value: 1.0,
        },
    );
    elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(x, "ui->iug"), (donor, "ug->iug")],
    )
    .expect("kv-heads broadcast lowers")
}

struct Synthetic {
    program: Vec<Op>,
    out: NodeId,
    state_out: NodeId,
    state_in: NodeId,
    inputs: Vec<(NodeId, Vec<f32>)>,
}

/// One `append_qwen35_delta_net_step` recurrence at the real qwen35moe GQA
/// split when called with `(16, 2, 128, 128)` (`ssm.group_count 16`,
/// `head_v_dim = 4096 / 32 = 128`) -- the same shape
/// `bind.rs`'s own `synthetic_gated_delta_net_gqa_program` builds, ported
/// here onto the crate's public spec builders.
fn synthetic_gated_delta_net_gqa_program(
    kv_heads: usize,
    group: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> Synthetic {
    let mut program = Vec::new();
    let num_v_heads = kv_heads * group;
    let query_pre = leaf(&mut program, &[kv_heads, head_k_dim]);
    let key_pre = leaf(&mut program, &[kv_heads, head_k_dim]);
    let value = leaf(&mut program, &[head_v_dim, kv_heads, group]);
    let gate = leaf(&mut program, &[kv_heads, group]);
    let beta = leaf(&mut program, &[kv_heads, group]);
    let state_in = leaf(&mut program, &[head_k_dim, head_v_dim, kv_heads, group]);
    let inv_sqrt_key_dim = append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: Vec::new(),
            value: 1.0 / (head_k_dim as f32).sqrt(),
        },
    );
    let query = broadcast_kv_heads(&mut program, query_pre, kv_heads as u32, group as u32);
    let key = broadcast_kv_heads(&mut program, key_pre, kv_heads as u32, group as u32);

    let (out, state_out) = append_qwen35_delta_net_step(
        &mut program,
        query,
        key,
        value,
        gate,
        beta,
        state_in,
        inv_sqrt_key_dim,
        "ug",
    )
    .expect("synthetic GQA gated-delta-net step builds");

    let mut lcg = Lcg(7);
    let mut fill = |count: usize| -> Vec<f32> { (0..count).map(|_| lcg.next_unit()).collect() };
    let inputs = vec![
        (query_pre, fill(head_k_dim * kv_heads)),
        (key_pre, fill(head_k_dim * kv_heads)),
        (value, fill(head_v_dim * num_v_heads)),
        (gate, fill(num_v_heads)),
        (beta, fill(num_v_heads)),
        (state_in, fill(head_k_dim * head_v_dim * num_v_heads)),
    ];
    Synthetic {
        program,
        out,
        state_out,
        state_in,
        inputs,
    }
}

fn assert_fused_bind_matches(synthetic: &Synthetic, shape_name: &str, max_relative_error: f32) {
    let shapes = proxima_tensor::infer(&synthetic.program, &[])
        .unwrap_or_else(|error| panic!("{shape_name} program infers: {error}"));
    let resolved = proxima_tensor::bind(
        &synthetic.program,
        &shapes,
        &[synthetic.out],
        NumericPolicy::bit_exact(),
    )
    .unwrap_or_else(|error| panic!("{shape_name} program binds: {error}"));
    assert!(
        resolved
            .iter()
            .any(|bound| matches!(bound.kind, BoundOpKind::GatedDeltaNet { .. })),
        "{shape_name}: matcher must fire so this test actually exercises \
         render_gated_delta_net, got kinds {:?}",
        resolved
            .iter()
            .map(|bound| bound.kind.name())
            .collect::<Vec<_>>()
    );

    let blocks: Vec<proxima_tensor::QuantizedBlock<'_>> = synthetic
        .inputs
        .iter()
        .map(|(_, values)| proxima_tensor::QuantizedBlock::Float32(values.as_slice()))
        .collect();

    let requested_outputs = [synthetic.out, synthetic.state_out, synthetic.state_in];
    let cpu = proxima_tensor::cpu::evaluate_quantized_exact(
        &synthetic.program,
        &[],
        &blocks,
        &requested_outputs,
    )
    .unwrap_or_else(|error| panic!("{shape_name}: cpu evaluates: {error}"));
    let metal = omega::execute(
        &synthetic.program,
        &[],
        &blocks,
        &requested_outputs,
        NumericPolicy::bit_exact(),
    )
    .unwrap_or_else(|error| panic!("{shape_name}: metal evaluates: {error}"));

    assert_matches(&cpu, &metal, synthetic.out, shape_name, "out", max_relative_error);
    assert_matches(
        &cpu,
        &metal,
        synthetic.state_out,
        shape_name,
        "state_out",
        max_relative_error,
    );

    let original_state_in = &synthetic
        .inputs
        .iter()
        .find(|(node, _)| *node == synthetic.state_in)
        .unwrap_or_else(|| panic!("{shape_name}: state_in is one of the fed inputs"))
        .1;
    let metal_state_in = metal
        .get(synthetic.state_in)
        .unwrap_or_else(|| panic!("{shape_name}: metal retains state_in as a requested output"))
        .0;
    assert_eq!(
        metal_state_in,
        original_state_in.as_slice(),
        "{shape_name}: state_in's own buffer must read back byte-identical to what was \
         uploaded -- the Metal kernel must never write into it"
    );
}

/// One node's Metal/CPU relative-error comparison, shared by `out` and
/// `state_out` -- both are real op outputs after ROW 547, so both get the
/// exact same check.
fn assert_matches(
    cpu: &proxima_tensor::cpu::Evaluated,
    metal: &proxima_tensor::cpu::Evaluated,
    node: NodeId,
    shape_name: &str,
    label: &str,
    max_relative_error: f32,
) {
    let expected = cpu
        .get(node)
        .unwrap_or_else(|| panic!("{shape_name}: cpu retains {label}"))
        .0;
    let actual = metal
        .get(node)
        .unwrap_or_else(|| panic!("{shape_name}: metal retains {label}"))
        .0;
    assert!(
        !actual.is_empty(),
        "{shape_name}: {label} comparison must not be vacuous"
    );
    assert_eq!(actual.len(), expected.len());

    let max_relative = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            let denominator = expected.abs().max(1.0e-6);
            (actual - expected).abs() / denominator
        })
        .fold(0.0_f32, f32::max);
    eprintln!("gated_delta_net {shape_name} {label} max_relative_error={max_relative}");
    assert!(
        max_relative <= max_relative_error,
        "{shape_name}: gated delta net Metal/CPU {label} parity diverged by {max_relative} \
         (bound {max_relative_error})"
    );
}

/// Small, hand-checkable GQA shape (`kv_heads=2`, `group=3`,
/// `head_k_dim=head_v_dim=2`) -- narrow enough that CPU's and Metal's
/// summation orders agree exactly (no floating-point reassociation gap),
/// so this asserts the tighter bound.
#[test]
fn metal_matches_cpu_at_small_gqa_shape() {
    let synthetic = synthetic_gated_delta_net_gqa_program(2, 3, 2, 2);
    assert_fused_bind_matches(&synthetic, "small_gqa", 1.0e-6);
}

/// The real qwen35moe GQA split (`ssm.group_count 16`, `ssm.state_size
/// 128`, `head_v_dim = 4096 / 32 = 128`) -- `head_k_dim = 128` sums 128
/// terms per reduce, wide enough that Metal's and CPU's sequential
/// accumulation orders can legally land on different (still IEEE-754
/// legal) f32 roundings, so this asserts the wider, still tight, bound.
#[test]
fn metal_matches_cpu_at_real_qwen35moe_gqa_shape() {
    let synthetic = synthetic_gated_delta_net_gqa_program(16, 2, 128, 128);
    assert_fused_bind_matches(&synthetic, "real_qwen35moe", 1.0e-5);
}
