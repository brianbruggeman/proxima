//! Metal-vs-CPU parity for the multi-position (M=4) qwen35 GDN mixer at real
//! checkpoint dims (`kv_heads=16, group=2, key_dim=2048, value_dim=4096,
//! l_cache=4`) -- the same graph and seeded inputs
//! `proxima_tensor::spec::tests::qwen35_ssm_mixer_one_evaluation_matches_repeated_single_position_steps_at_real_dims`
//! builds via `append_qwen35_ssm_mixer_with_taps`, evaluated three ways: CPU
//! reference, Metal with the production output set, and Metal with every
//! `SsmMixerTaps` node also requested. A CPU/Metal disagreement that vanishes
//! once the tap nodes are also requested as outputs is the signature of the
//! planner retiring or absorbing a node that still has readers.

#![cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::spec::{
    GdnOutputGate, SsmMixerTaps, append_qwen35_ssm_mixer_with_taps, input_leaf, scalar_constant,
};
use proxima_tensor::test_support::{ParityRun, compare_rows_relative_to_norm};
use proxima_tensor::{DType, Extent, NodeId, NumericPolicy, Op, QuantizedBlock};

const KEY_DIM: u32 = 2048;
const VALUE_DIM: u32 = 4096;
const KV_HEADS: u32 = 16;
const GROUP: u32 = 2;
const L_CACHE: u32 = 4;
const EMBEDDING: u32 = 6;

fn deterministic_wave(index: usize, modulus: usize, scale: f32) -> f32 {
    ((index % modulus) as f32 + 1.0) * scale
}

struct RealDimsFixture {
    program: Vec<Op>,
    mixer_out: NodeId,
    taps: SsmMixerTaps,
    named: Vec<(&'static str, Vec<f32>)>,
}

/// Reproduces the real-dims oracle's `build_program`/seeded-fill exactly
/// (`proxima-tensor/src/spec.rs`, `qwen35_ssm_mixer_one_evaluation_matches_repeated_single_position_steps_at_real_dims`),
/// static `s = positions` rather than that test's symbolic single-step form.
fn build_fixture(positions: u32) -> RealDimsFixture {
    let qkv_dim = 2 * KEY_DIM + VALUE_DIM;
    let num_v_heads = KV_HEADS * GROUP;
    let head_k_dim = KEY_DIM / KV_HEADS;
    let head_v_dim = VALUE_DIM / num_v_heads;

    let x_data: Vec<f32> = (0..(positions * EMBEDDING) as usize)
        .map(|index| deterministic_wave(index, 23, 0.01))
        .collect();
    let attn_norm_weight_data: Vec<f32> = (0..EMBEDDING as usize)
        .map(|index| 1.0 + deterministic_wave(index, 7, 0.02))
        .collect();
    let wqkv_data: Vec<f32> = (0..(EMBEDDING * qkv_dim) as usize)
        .map(|index| deterministic_wave(index, 29, 0.002))
        .collect();
    let wqkv_gate_data: Vec<f32> = (0..(EMBEDDING * VALUE_DIM) as usize)
        .map(|index| deterministic_wave(index, 31, 0.003))
        .collect();
    let conv_weight_data: Vec<f32> = (0..(qkv_dim * L_CACHE) as usize)
        .map(|index| deterministic_wave(index, 17, 0.01))
        .collect();
    let ssm_beta_data: Vec<f32> = (0..(EMBEDDING * num_v_heads) as usize)
        .map(|index| deterministic_wave(index, 11, 0.02) - 0.1)
        .collect();
    let ssm_alpha_data: Vec<f32> = (0..(EMBEDDING * num_v_heads) as usize)
        .map(|index| deterministic_wave(index, 13, 0.02) - 0.1)
        .collect();
    let ssm_dt_bias_data: Vec<f32> = (0..num_v_heads as usize)
        .map(|index| deterministic_wave(index, 5, 0.01))
        .collect();
    let ssm_a_data: Vec<f32> = (0..num_v_heads as usize)
        .map(|index| -deterministic_wave(index, 5, 0.05))
        .collect();
    let ssm_norm_weight_data: Vec<f32> = (0..head_v_dim as usize)
        .map(|index| 1.0 + deterministic_wave(index, 3, 0.01))
        .collect();
    let ssm_out_data: Vec<f32> = (0..(VALUE_DIM * EMBEDDING) as usize)
        .map(|index| deterministic_wave(index, 19, 0.004))
        .collect();
    let head_eps_data = vec![1e-6_f32; (KV_HEADS * GROUP) as usize];
    let initial_state = vec![0.0_f32; (head_k_dim * head_v_dim * KV_HEADS * GROUP) as usize];
    let initial_history = vec![0.0_f32; ((L_CACHE - 1) * qkv_dim) as usize];
    let eps_data = vec![1e-6_f32; positions as usize];

    let mut program = Vec::new();
    let x = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(positions), Extent::Static(EMBEDDING)],
        "x",
    );
    let inv_dim = scalar_constant(&mut program, 1.0 / EMBEDDING as f32);
    let eps = input_leaf(&mut program, DType::Float32, vec![Extent::Static(positions)], "eps");
    let head_eps = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(KV_HEADS), Extent::Static(GROUP)],
        "head_eps",
    );
    let one = scalar_constant(&mut program, 1.0);
    let inv_sqrt_key_dim = scalar_constant(&mut program, 1.0 / (head_k_dim as f32).sqrt());
    let inv_head_v_dim = scalar_constant(&mut program, 1.0 / head_v_dim as f32);
    let attn_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(EMBEDDING)],
        "attn_norm_weight",
    );
    let wqkv = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(EMBEDDING), Extent::Static(qkv_dim)],
        "wqkv",
    );
    let wqkv_gate = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(EMBEDDING), Extent::Static(VALUE_DIM)],
        "wqkv_gate",
    );
    let conv_weight = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(qkv_dim), Extent::Static(L_CACHE)],
        "conv_weight",
    );
    let conv_history_in = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(L_CACHE - 1), Extent::Static(qkv_dim)],
        "conv_history_in",
    );
    let ssm_beta = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(EMBEDDING), Extent::Static(num_v_heads)],
        "ssm_beta",
    );
    let ssm_alpha = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(EMBEDDING), Extent::Static(num_v_heads)],
        "ssm_alpha",
    );
    let ssm_dt_bias = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(num_v_heads)],
        "ssm_dt_bias",
    );
    let ssm_a = input_leaf(&mut program, DType::Float32, vec![Extent::Static(num_v_heads)], "ssm_a");
    let ssm_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(head_v_dim)],
        "ssm_norm_weight",
    );
    let ssm_out = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(VALUE_DIM), Extent::Static(EMBEDDING)],
        "ssm_out",
    );
    let state_in = input_leaf(
        &mut program,
        DType::Float32,
        vec![
            Extent::Static(head_k_dim),
            Extent::Static(head_v_dim),
            Extent::Static(KV_HEADS),
            Extent::Static(GROUP),
        ],
        "state_in",
    );

    let (mixer_out, taps) = append_qwen35_ssm_mixer_with_taps(
        &mut program,
        x,
        inv_dim,
        eps,
        head_eps,
        one,
        inv_sqrt_key_dim,
        inv_head_v_dim,
        Some(attn_norm_weight),
        wqkv,
        wqkv_gate,
        conv_weight,
        conv_history_in,
        ssm_beta,
        ssm_alpha,
        ssm_dt_bias,
        ssm_a,
        ssm_norm_weight,
        ssm_out,
        state_in,
        KEY_DIM,
        VALUE_DIM,
        KV_HEADS,
        GROUP,
        L_CACHE,
        GdnOutputGate::Silu,
        Some(positions),
    )
    .expect("the qwen35 ssm mixer lowers at real dims");

    let named = vec![
        ("x", x_data),
        ("eps", eps_data),
        ("head_eps", head_eps_data),
        ("attn_norm_weight", attn_norm_weight_data),
        ("wqkv", wqkv_data),
        ("wqkv_gate", wqkv_gate_data),
        ("conv_weight", conv_weight_data),
        ("conv_history_in", initial_history),
        ("ssm_beta", ssm_beta_data),
        ("ssm_alpha", ssm_alpha_data),
        ("ssm_dt_bias", ssm_dt_bias_data),
        ("ssm_a", ssm_a_data),
        ("ssm_norm_weight", ssm_norm_weight_data),
        ("ssm_out", ssm_out_data),
        ("state_in", initial_state),
    ];

    RealDimsFixture { program, mixer_out, taps, named }
}

/// Production output set: only what a serving loop actually reads back --
/// `mixer_out` (the `[s, d]` projected residual) and the carried recurrent
/// state.
fn production_outputs(fixture: &RealDimsFixture) -> Vec<NodeId> {
    vec![fixture.mixer_out, fixture.taps.state_out]
}

/// Every intermediate `SsmMixerTaps` node in addition to the production set
/// -- requesting these as outputs forces the planner to keep every reader
/// alive rather than retiring/absorbing a node once its last "production"
/// consumer is satisfied.
fn wide_outputs(fixture: &RealDimsFixture) -> Vec<NodeId> {
    let taps = &fixture.taps;
    vec![
        fixture.mixer_out,
        taps.state_out,
        taps.qkv_mixed,
        taps.query_sequence,
        taps.key_sequence,
        taps.value_sequence,
        taps.beta_sequence,
        taps.gate_sequence,
        taps.z_sequence,
        taps.query,
        taps.key,
        taps.value,
        taps.gate,
        taps.beta,
        taps.z_head,
        taps.delta_out,
        taps.z,
        taps.gated_rmsnorm_out,
        taps.gated_value,
        taps.ssm_out_result,
    ]
}

fn assert_production_parity(positions: u32, include_wide: bool) {
    let fixture = build_fixture(positions);
    let named: Vec<(&str, &[f32])> = fixture.named.iter().map(|(name, data)| (*name, data.as_slice())).collect();
    let quantized_named: Vec<(&str, QuantizedBlock<'_>)> = fixture
        .named
        .iter()
        .map(|(name, data)| (*name, QuantizedBlock::Float32(data.as_slice())))
        .collect();

    let production = production_outputs(&fixture);
    let symbols = [u64::from(positions)];

    let cpu_production = proxima_tensor::cpu::evaluate_named(&fixture.program, &symbols, &named, &production)
        .expect("cpu reference evaluates the production output set");
    let production_plan =
        omega::plan_named(&fixture.program, &symbols, &quantized_named, &production, NumericPolicy::default())
            .expect("metal plan builds for the production output set");
    let metal_production = omega::execute_plan_named(&production_plan, &quantized_named)
        .expect("metal evaluates the production output set");

    let wide_result = include_wide.then(|| {
        let wide = wide_outputs(&fixture);
        let wide_plan =
            omega::plan_named(&fixture.program, &symbols, &quantized_named, &wide, NumericPolicy::default())
                .expect("metal plan builds for the wide output set");
        omega::execute_plan_named(&wide_plan, &quantized_named).expect("metal evaluates the wide output set")
    });

    let mut failures = Vec::new();
    for (label, node) in [("mixer_out", fixture.mixer_out), ("state_out", fixture.taps.state_out)] {
        let cpu_values = cpu_production.get(node).expect("cpu reference produced this node").0;
        let metal_values = metal_production.get(node).expect("metal production run produced this node").0;
        let wide_values =
            wide_result.as_ref().map(|wide| wide.get(node).expect("metal wide run produced this node").0);

        let mut candidates = vec![ParityRun { label: "metal_production", values: metal_values }];
        if let Some(wide_values) = wide_values {
            candidates.push(ParityRun { label: "metal_wide", values: wide_values });
        }

        let case_label = format!("qwen35_mixer_multi_position m={positions} tap={label}");
        let case_failures = compare_rows_relative_to_norm(
            &case_label,
            positions as usize,
            1e-4,
            &ParityRun { label: "cpu", values: cpu_values },
            &candidates,
        );
        failures.extend(case_failures.into_iter().filter(|failure| failure.contains("metal_production")));
    }

    assert!(
        failures.is_empty(),
        "metal disagrees with cpu on the production output set for the multi-position (M={positions}) qwen35 mixer at real dims:\n{}",
        failures.join("\n")
    );
}

#[proxima::test]
#[case::m4(4)]
#[case::m8(8)]
#[case::m13(13)]
#[case::m16(16)]
async fn metal_multi_position_qwen35_mixer_matches_cpu_at_real_dims_on_production_outputs(#[case] positions: u32) {
    assert_production_parity(positions, false);
}
