//! CPU-vs-CUDA parity of the qwen35 gated-delta-net mixer at the real head geometry
//! (`kv_heads` 16, `group` 3, `head_k_dim` = `head_v_dim` = 128), checked tap by tap so a
//! disagreement names the first op that diverges instead of only the mixer output.

#![cfg(all(feature = "cpu", feature = "cuda-driver"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use omega::Codec;
use omega::backend::{Engine, GpuDriver, execute_plan_named, plan_named};
use proxima_tensor::spec::{
    GdnOutputGate, SsmMixerTaps, append_delta_net_mixer_with_taps_and_layout, input_leaf,
    count_constant, scalar_constant,
};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{Extent, NodeId, QuantizedBlock, infer};
use proxima_numeric::{DType, NumericPolicy};

const KV_HEADS: u32 = 16;
const GROUP: u32 = 3;
const HEAD_DIM: u32 = 128;
const MODEL_DIM: u32 = 64;
const CONV_KERNEL: u32 = 4;
const TOLERANCE: f32 = 1e-3;

struct Fixture {
    program: Vec<proxima_tensor::Op>,
    leaves: Vec<(String, NodeId)>,
    mixer_out: NodeId,
    taps: SsmMixerTaps,
    symbols: [u64; 1],
}

fn leaf(program: &mut Vec<proxima_tensor::Op>, name: &str, shape: Vec<Extent>) -> NodeId {
    input_leaf(program, DType::FLOAT32, shape, name)
}

fn fixture(positions: u32) -> Fixture {
    let (x_extent, prefill_width) = match positions {
        1 => (Extent::Symbolic(0), None),
        width => (Extent::Static(width), Some(width)),
    };
    let num_v_heads = KV_HEADS * GROUP;
    let key_dim = KV_HEADS * HEAD_DIM;
    let value_dim = num_v_heads * HEAD_DIM;
    let qkv_dim = 2 * key_dim + value_dim;
    let mut program = Vec::new();
    let x = leaf(&mut program, "x", vec![x_extent, Extent::Static(MODEL_DIM)]);
    let inv_dim = count_constant(&mut program, MODEL_DIM);
    let eps = leaf(&mut program, "eps", vec![x_extent]);
    let head_eps = leaf(
        &mut program,
        "head_eps",
        vec![Extent::Static(KV_HEADS), Extent::Static(GROUP)],
    );
    let one = scalar_constant(&mut program, 1.0);
    let inv_sqrt_key_dim = scalar_constant(&mut program, 1.0 / (HEAD_DIM as f32).sqrt());
    let inv_head_v_dim = count_constant(&mut program, HEAD_DIM);
    let attn_norm_weight = leaf(&mut program, "attn_norm_weight", vec![Extent::Static(MODEL_DIM)]);
    let wqkv = leaf(&mut program, "wqkv", vec![Extent::Static(MODEL_DIM), Extent::Static(qkv_dim)]);
    let wqkv_gate = leaf(
        &mut program,
        "wqkv_gate",
        vec![Extent::Static(MODEL_DIM), Extent::Static(value_dim)],
    );
    let conv_weight = leaf(
        &mut program,
        "conv_weight",
        vec![Extent::Static(qkv_dim), Extent::Static(CONV_KERNEL)],
    );
    let conv_history_in = leaf(
        &mut program,
        "conv_history_in",
        vec![Extent::Static(CONV_KERNEL - 1), Extent::Static(qkv_dim)],
    );
    let ssm_beta = leaf(&mut program, "ssm_beta", vec![Extent::Static(MODEL_DIM), Extent::Static(num_v_heads)]);
    let ssm_alpha = leaf(&mut program, "ssm_alpha", vec![Extent::Static(MODEL_DIM), Extent::Static(num_v_heads)]);
    let ssm_dt_bias = leaf(&mut program, "ssm_dt_bias", vec![Extent::Static(num_v_heads)]);
    let ssm_a = leaf(&mut program, "ssm_a", vec![Extent::Static(num_v_heads)]);
    let ssm_norm_weight = leaf(&mut program, "ssm_norm_weight", vec![Extent::Static(HEAD_DIM)]);
    let ssm_out = leaf(&mut program, "ssm_out", vec![Extent::Static(value_dim), Extent::Static(MODEL_DIM)]);
    let state_in = leaf(
        &mut program,
        "state_in",
        vec![
            Extent::Static(HEAD_DIM),
            Extent::Static(HEAD_DIM),
            Extent::Static(KV_HEADS),
            Extent::Static(GROUP),
        ],
    );
    let (mixer_out, taps) = append_delta_net_mixer_with_taps_and_layout(
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
        key_dim,
        value_dim,
        KV_HEADS,
        GROUP,
        CONV_KERNEL,
        GdnOutputGate::Silu,
        false,
        false,
        None,
        prefill_width,
    )
    .expect("the qwen35 gdn mixer lowers");
    let leaves = [
        ("x", x),
        ("eps", eps),
        ("head_eps", head_eps),
        ("attn_norm_weight", attn_norm_weight),
        ("wqkv", wqkv),
        ("wqkv_gate", wqkv_gate),
        ("conv_weight", conv_weight),
        ("conv_history_in", conv_history_in),
        ("ssm_beta", ssm_beta),
        ("ssm_alpha", ssm_alpha),
        ("ssm_dt_bias", ssm_dt_bias),
        ("ssm_a", ssm_a),
        ("ssm_norm_weight", ssm_norm_weight),
        ("ssm_out", ssm_out),
        ("state_in", state_in),
    ]
    .into_iter()
    .map(|(name, node)| (name.to_owned(), node))
    .collect();
    Fixture {
        program,
        leaves,
        mixer_out,
        taps,
        symbols: [u64::from(positions)],
    }
}

fn leaf_data(fixture: &Fixture) -> Vec<(String, Vec<f32>)> {
    let shapes = infer(&fixture.program, &fixture.symbols)
        .expect("the mixer program infers at the width it was built for");
    let leaves = &fixture.leaves;
    let mut lcg = Lcg(11);
    leaves
        .iter()
        .map(|(name, node)| {
            let count: usize = shapes.of(*node).iter().map(|extent| *extent as usize).product();
            let data = match name.as_str() {
                "eps" | "head_eps" => vec![1e-6; count],
                "ssm_a" => (0..count).map(|_| -lcg.next_unit().abs() - 0.1).collect(),
                _ => (0..count).map(|_| lcg.next_unit()).collect(),
            };
            (name.clone(), data)
        })
        .collect()
}

fn tap_roots(fixture: &Fixture) -> Vec<(&'static str, NodeId)> {
    let taps = &fixture.taps;
    vec![
        ("qkv_mixed", taps.qkv_mixed),
        ("query_sequence", taps.query_sequence),
        ("key_sequence", taps.key_sequence),
        ("value_sequence", taps.value_sequence),
        ("gate_sequence", taps.gate_sequence),
        ("beta_sequence", taps.beta_sequence),
        ("z_sequence", taps.z_sequence),
        ("query", taps.query),
        ("key", taps.key),
        ("value", taps.value),
        ("gate", taps.gate),
        ("beta", taps.beta),
        ("z_head", taps.z_head),
        ("state_out", taps.state_out),
        ("delta_out", taps.delta_out),
        ("z", taps.z),
        ("gated_rmsnorm_out", taps.gated_rmsnorm_out),
        ("gated_value", taps.gated_value),
        ("ssm_out_result", taps.ssm_out_result),
        ("mixer_out", fixture.mixer_out),
    ]
}

fn evaluate(
    engine: Engine,
    driver: Option<GpuDriver>,
    fixture: &Fixture,
    named: &[(&str, QuantizedBlock<'_>)],
    roots: &[NodeId],
    policy: NumericPolicy,
) -> proxima_tensor::cpu::Evaluated {
    let mut plan = plan_named(engine, driver, &fixture.program, &fixture.symbols, named, roots, policy)
        .expect("the mixer plans");
    execute_plan_named(&mut plan, named).expect("the mixer executes")
}

fn relative_error(expected: &[f32], actual: &[f32]) -> f32 {
    let max_magnitude = expected.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
    let max_diff = expected
        .iter()
        .zip(actual)
        .map(|(want, got)| if got.is_finite() { (want - got).abs() } else { f32::INFINITY })
        .fold(0.0f32, f32::max);
    max_diff / max_magnitude.max(f32::MIN_POSITIVE)
}

fn assert_taps_match_cpu(positions: u32) {
    let fixture = fixture(positions);
    let owned = leaf_data(&fixture);
    let named: Vec<(&str, QuantizedBlock<'_>)> = owned
        .iter()
        .map(|(name, data)| (name.as_str(), QuantizedBlock::Float32(data)))
        .collect();
    let tapped = tap_roots(&fixture);
    let roots: Vec<NodeId> = tapped.iter().map(|(_, node)| *node).collect();

    let cpu = evaluate(Engine::Cpu, None, &fixture, &named, &roots, NumericPolicy::default());
    let cuda = evaluate(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &fixture,
        &named,
        &roots,
        NumericPolicy::bit_exact().with_reassociation(true),
    );

    let mut offenders = Vec::new();
    for (name, node) in &tapped {
        let (expected, _) = cpu.get(*node).expect("cpu returns every requested tap");
        let (actual, _) = cuda.get(*node).expect("cuda returns every requested tap");
        let relative = relative_error(expected, actual);
        eprintln!("tap {name}: elements={} relative={relative}", expected.len());
        if relative.is_nan() || relative >= TOLERANCE {
            offenders.push(format!("{name}={relative}"));
        }
    }
    assert!(offenders.is_empty(), "taps where cuda disagrees with cpu: {offenders:?}");
}

fn assert_fused_roots_match_cpu(positions: u32) {
    let fixture = fixture(positions);
    let owned = leaf_data(&fixture);
    let named: Vec<(&str, QuantizedBlock<'_>)> = owned
        .iter()
        .map(|(name, data)| (name.as_str(), QuantizedBlock::Float32(data)))
        .collect();
    let roots = [fixture.mixer_out, fixture.taps.state_out];

    let cpu = evaluate(Engine::Cpu, None, &fixture, &named, &roots, NumericPolicy::default());
    let cuda = evaluate(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &fixture,
        &named,
        &roots,
        NumericPolicy::bit_exact().with_reassociation(true),
    );

    for (name, node) in [("mixer_out", roots[0]), ("state_out", roots[1])] {
        let (expected, _) = cpu.get(node).expect("cpu returns the root");
        let (actual, _) = cuda.get(node).expect("cuda returns the root");
        let relative = relative_error(expected, actual);
        eprintln!("fused {name}: relative={relative}");
        assert!(relative < TOLERANCE, "{name}: cuda vs cpu relative={relative}");
    }
}

#[test]
fn cuda_gdn_mixer_taps_match_cpu_at_one_position() {
    assert_taps_match_cpu(1);
}

#[test]
fn cuda_gdn_mixer_taps_match_cpu_across_a_five_token_prefill() {
    assert_taps_match_cpu(5);
}

#[test]
fn cuda_gdn_mixer_fused_roots_match_cpu_at_one_position() {
    assert_fused_roots_match_cpu(1);
}

#[test]
fn cuda_gdn_mixer_fused_roots_match_cpu_across_a_five_token_prefill() {
    assert_fused_roots_match_cpu(5);
}

struct PackedOutProjection {
    packed: Vec<u8>,
    transposed_decoded: Vec<f32>,
}

fn packed_out_projection(value_dim: usize, embedding: usize) -> PackedOutProjection {
    use proxima_gguf::quant::q1_0::{BLOCK_BYTES, QK_Q1_0, dequantize, quantize};
    let mut lcg = Lcg(23);
    let rows: Vec<f32> = (0..embedding * value_dim).map(|_| lcg.next_unit()).collect();
    let row_bytes = value_dim / QK_Q1_0 * BLOCK_BYTES;
    let mut packed = vec![0u8; embedding * row_bytes];
    let mut decoded = vec![0.0f32; embedding * value_dim];
    for ((row, bytes), row_decoded) in rows
        .chunks_exact(value_dim)
        .zip(packed.chunks_exact_mut(row_bytes))
        .zip(decoded.chunks_exact_mut(value_dim))
    {
        quantize(row, bytes).expect("a whole number of blocks");
        dequantize(bytes, row_decoded).expect("a whole number of blocks");
    }
    let mut transposed_decoded = vec![0.0f32; embedding * value_dim];
    for output in 0..embedding {
        for input in 0..value_dim {
            transposed_decoded[input * embedding + output] = decoded[output * value_dim + input];
        }
    }
    PackedOutProjection {
        packed,
        transposed_decoded,
    }
}

fn mixer_out_with_ssm_out(
    engine: Engine,
    driver: Option<GpuDriver>,
    fixture: &Fixture,
    owned: &[(String, Vec<f32>)],
    ssm_out: QuantizedBlock<'_>,
    policy: NumericPolicy,
) -> Vec<f32> {
    let mut named: Vec<(&str, QuantizedBlock<'_>)> = owned
        .iter()
        .filter(|(name, _)| name != "ssm_out")
        .map(|(name, data)| (name.as_str(), QuantizedBlock::Float32(data)))
        .collect();
    named.push(("ssm_out", ssm_out));
    let roots = [fixture.mixer_out];
    evaluate(engine, driver, fixture, &named, &roots, policy)
        .get(fixture.mixer_out)
        .expect("the mixer output is returned")
        .0
        .to_vec()
}

#[test]
fn packed_q1_0_ssm_out_matches_its_float32_decode_on_cpu_and_cuda() {
    let fixture = fixture(1);
    let owned = leaf_data(&fixture);
    let value_dim = (KV_HEADS * GROUP * HEAD_DIM) as usize;
    let projection = packed_out_projection(value_dim, MODEL_DIM as usize);

    let oracle = mixer_out_with_ssm_out(
        Engine::Cpu,
        None,
        &fixture,
        &owned,
        QuantizedBlock::Float32(&projection.transposed_decoded),
        NumericPolicy::default(),
    );
    let packed_block = || QuantizedBlock::Packed {
        codec: Codec::Q1_0,
        bytes: &projection.packed,
    };
    let cpu_packed = mixer_out_with_ssm_out(
        Engine::Cpu,
        None,
        &fixture,
        &owned,
        packed_block(),
        NumericPolicy::default(),
    );
    let cuda_packed = mixer_out_with_ssm_out(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &fixture,
        &owned,
        packed_block(),
        NumericPolicy::bit_exact().with_reassociation(true),
    );

    let cpu_error = relative_error(&oracle, &cpu_packed);
    let cuda_error = relative_error(&oracle, &cuda_packed);
    assert!(
        cpu_error < TOLERANCE && cuda_error < TOLERANCE,
        "packed ssm_out vs its float32 decode: cpu relative={cpu_error}, cuda relative={cuda_error}"
    );
}
