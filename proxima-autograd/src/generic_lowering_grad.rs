//! Gradient checks for the model-agnostic diffusion lowerings in
//! `proxima_tensor::spec` (`layer_norm_no_affine`, `modulate`,
//! `non_causal_attention`, `multi_axis_rope_tables` -> `fused_rope_pair`,
//! `tap_layers`). Each test differentiates the real builder output with
//! [`crate::adjoint::differentiate`] and compares every requested input's
//! gradient with central finite differences of the forward program.
//!
//! The tests live here, not in `proxima-tensor`, because `proxima-autograd`
//! depends on `proxima-tensor`; the reverse edge would be a cycle.

use alloc::string::String;
use alloc::vec::Vec;

use proxima_numeric::DType;
use proxima_tensor::op::{self, Extent, NodeId, Op, ReduceInit, ScalarOp};
use proxima_tensor::spec::{
    RopePairing, cross_kv, elementwise_like, fused_rope_pair, layer_norm, layer_norm_no_affine,
    modulate, multi_axis_rope_tables, non_causal_attention, reduce_like, scalar_constant_like,
    tap_layers,
};

// central difference of an f32 loss of magnitude L at step h errs by about
// eps_f32 * L / h (roundoff, eps_f32 = 1.2e-7) plus h^2/6 * |f'''| (truncation).
// with h = 1e-2, L <= ~10 and |f'''| <= ~60 on these unit-scale inputs that is
// ~1e-4 + ~1e-3, so 3e-3 absolute leaves a little headroom without hiding a
// wrong adjoint, whose error is of the order of the gradient itself (~0.1..1).
const STEP: f32 = 1e-2;
const ABS_TOLERANCE: f32 = 3e-3;

struct Lcg(u64);

impl Lcg {
    fn values(seed: u64, count: usize) -> alloc::vec::Vec<f32> {
        let mut state = Lcg(seed);
        (0..count).map(|_| state.next()).collect()
    }

    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn input(program: &mut Vec<Op>, name: &str, shape: &[u32]) -> NodeId {
    op::append(
        program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: shape.iter().map(|&extent| Extent::Static(extent)).collect(),
            name: Some(name.into()),
        },
    )
}

/// `sum(out * w)` with a fixed non-uniform weight leaf `w`, so no output
/// symmetry (a normalized row sums to zero) collapses the gradient to zero.
fn weighted_loss(program: &mut Vec<Op>, out: NodeId, axes: &str, shape: &[u32]) -> NodeId {
    weighted_loss_named(program, out, axes, "w", shape)
}

fn weighted_loss_named(
    program: &mut Vec<Op>,
    out: NodeId,
    axes: &str,
    name: &str,
    shape: &[u32],
) -> NodeId {
    let weights = input(program, name, shape);
    let pattern = alloc::format!("{axes}->{axes}");
    let weighted = elementwise_like(
        program,
        out,
        ScalarOp::Multiply,
        &[(out, pattern.as_str()), (weights, pattern.as_str())],
    )
    .expect("weighting the output builds");
    reduce_like(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        weighted,
        pattern.as_str(),
        alloc::format!("->{axes}").as_str(),
    )
    .expect("loss reduce builds")
}

fn borrowed(owned: &[(String, Vec<f32>)]) -> Vec<(&str, &[f32])> {
    owned
        .iter()
        .map(|(n, v)| (n.as_str(), v.as_slice()))
        .collect()
}

struct Report {
    name: String,
    max_abs: f32,
    max_rel: f32,
}

fn check(
    lowering: &str,
    program: &[Op],
    loss: NodeId,
    bound: &[(&str, Vec<f32>)],
    wrt: &[&str],
) -> Vec<Report> {
    let differentiated =
        crate::adjoint::differentiate(program, loss).expect("scalar loss differentiates");
    let named = |values: &[(&str, Vec<f32>)]| -> Vec<(String, Vec<f32>)> {
        values
            .iter()
            .map(|(n, v)| ((*n).into(), v.clone()))
            .collect()
    };
    let owned = named(bound);
    let loss_at = |values: &[(String, Vec<f32>)]| {
        proxima_tensor::cpu::evaluate_named(program, &[], &borrowed(values), &[loss])
            .expect("forward program evaluates")
            .get(loss)
            .expect("loss requested")
            .0[0]
    };
    let mut reports = Vec::new();
    for &target in wrt {
        let grad = differentiated
            .gradient_of_named(target)
            .unwrap_or_else(|| panic!("{target} feeds the loss and has a gradient"));
        let evaluated = proxima_tensor::cpu::evaluate_named(
            &differentiated.program,
            &[],
            &borrowed(&owned),
            &[grad],
        )
        .expect("adjoint program evaluates");
        let analytic = evaluated.get(grad).expect("gradient requested").0.to_vec();
        let slot = owned.iter().position(|(n, _)| n == target).expect("bound");
        assert_eq!(
            analytic.len(),
            owned[slot].1.len(),
            "{target} gradient size"
        );
        let (mut max_abs, mut max_rel) = (0f32, 0f32);
        let mut probe = owned.clone();
        for index in 0..analytic.len() {
            let original = probe[slot].1[index];
            probe[slot].1[index] = original + STEP;
            let plus = loss_at(&probe);
            probe[slot].1[index] = original - STEP;
            let minus = loss_at(&probe);
            probe[slot].1[index] = original;
            let numeric = (plus - minus) / (2.0 * STEP);
            let abs = (analytic[index] - numeric).abs();
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(abs / (numeric.abs().max(1.0)));
        }
        reports.push(Report {
            name: target.into(),
            max_abs,
            max_rel,
        });
    }
    for report in &reports {
        std::println!(
            "{lowering} wrt={} max_abs={:e} max_rel={:e}",
            report.name,
            report.max_abs,
            report.max_rel
        );
        assert!(
            report.max_abs < ABS_TOLERANCE,
            "{lowering}: {} gradient differs from central differences by {} (tolerance {ABS_TOLERANCE})",
            report.name,
            report.max_abs
        );
    }
    reports
}

#[proxima::test]
async fn layer_norm_no_affine_gradient_matches_finite_differences() {
    let (s, d) = (3u32, 4u32);
    let mut program = Vec::new();
    let x = input(&mut program, "x", &[s, d]);
    let out = layer_norm_no_affine(&mut program, x, d, 1e-5).expect("layer norm builds");
    let loss = weighted_loss(&mut program, out, "sd", &[s, d]);
    let bound = [("x", Lcg::values(1, 12)), ("w", Lcg::values(2, 12))];
    let reports = check("layer_norm_no_affine", &program, loss, &bound, &["x"]);
    assert_eq!(reports.len(), 1);
}

#[proxima::test]
async fn modulate_gradient_matches_finite_differences_for_input_shift_and_scale() {
    let (s, d) = (3u32, 4u32);
    let mut program = Vec::new();
    let normed = input(&mut program, "normed", &[s, d]);
    let shift = input(&mut program, "shift", &[d]);
    let scale = input(&mut program, "scale", &[d]);
    let out = modulate(&mut program, normed, shift, scale).expect("modulate builds");
    let loss = weighted_loss(&mut program, out, "sd", &[s, d]);
    let bound = [
        ("normed", Lcg::values(3, 12)),
        ("shift", Lcg::values(4, 4)),
        ("scale", Lcg::values(5, 4)),
        ("w", Lcg::values(6, 12)),
    ];
    let reports = check(
        "modulate",
        &program,
        loss,
        &bound,
        &["normed", "shift", "scale"],
    );
    assert_eq!(reports.len(), 3);
}

#[proxima::test]
async fn non_causal_attention_gradient_matches_finite_differences_for_q_k_and_v() {
    let (s, t, u, i, d) = (3u32, 2u32, 2u32, 4u32, 3u32);
    let mut program = Vec::new();
    let q = input(&mut program, "q", &[s, u, i]);
    let k = input(&mut program, "k", &[t, u, i]);
    let v = input(&mut program, "v", &[t, u, d]);
    let scale = scalar_constant_like(&mut program, 0.5, q).expect("scale builds");
    let out = non_causal_attention(&mut program, q, k, v, scale).expect("attention builds");
    let loss = weighted_loss(&mut program, out, "sugd", &[s, u, 1, d]);
    let bound = [
        ("q", Lcg::values(7, 24)),
        ("k", Lcg::values(8, 16)),
        ("v", Lcg::values(9, 12)),
        ("w", Lcg::values(10, 18)),
    ];
    let reports = check(
        "non_causal_attention",
        &program,
        loss,
        &bound,
        &["q", "k", "v"],
    );
    assert_eq!(reports.len(), 3);
}

#[proxima::test]
async fn fused_rope_pair_gradient_matches_finite_differences_with_multi_axis_tables() {
    let (s, h) = (3u32, 2u32);
    let sections = [4u32, 2];
    let width: u32 = sections.iter().sum();
    let pairs = width / 2;
    let positions = [0.0f32, 1.0, 1.0, 0.0, 2.0, 3.0];
    let (cos_bytes, sin_bytes) =
        multi_axis_rope_tables(&positions, &sections, &[10000.0, 100.0], DType::FLOAT32)
            .expect("tables build");
    let decode = |bytes: &[u8]| -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect()
    };
    let mut program = Vec::new();
    let x = input(&mut program, "x", &[s, h, width]);
    let cos = input(&mut program, "cos", &[s, pairs]);
    let sin = input(&mut program, "sin", &[s, pairs]);
    let (first, second) = fused_rope_pair(&mut program, x, 'h', cos, sin, RopePairing::Interleaved)
        .expect("rope builds");
    let weighted_first = weighted_loss_named(&mut program, first, "shi", "w1", &[s, h, pairs]);
    let weighted_second = weighted_loss_named(&mut program, second, "shi", "w2", &[s, h, pairs]);
    let loss = elementwise_like(
        &mut program,
        weighted_first,
        ScalarOp::Add,
        &[(weighted_first, "->"), (weighted_second, "->")],
    )
    .expect("loss sums");
    let bound = [
        ("x", Lcg::values(11, (s * h * width) as usize)),
        ("cos", decode(&cos_bytes)),
        ("sin", decode(&sin_bytes)),
        ("w1", Lcg::values(12, (s * h * pairs) as usize)),
        ("w2", Lcg::values(13, (s * h * pairs) as usize)),
    ];
    let reports = check("fused_rope_pair", &program, loss, &bound, &["x"]);
    assert_eq!(reports.len(), 1);
}

#[proxima::test]
async fn tap_layers_gradient_matches_finite_differences_and_routes_only_to_tapped_layers() {
    let (s, d) = (3u32, 4u32);
    let mut program = Vec::new();
    let layers: Vec<NodeId> = (0..3)
        .map(|layer| input(&mut program, &alloc::format!("layer{layer}"), &[s, d]))
        .collect();
    let out = tap_layers(&mut program, &layers, &[2, 0]).expect("taps build");
    let loss = weighted_loss(&mut program, out, "skd", &[s, 2, d]);
    let bound = [
        ("layer0", Lcg::values(14, 12)),
        ("layer1", Lcg::values(15, 12)),
        ("layer2", Lcg::values(16, 12)),
        ("w", Lcg::values(17, 24)),
    ];
    let reports = check("tap_layers", &program, loss, &bound, &["layer0", "layer2"]);
    assert_eq!(reports.len(), 2);
    let differentiated = crate::adjoint::differentiate(&program, loss).expect("differentiates");
    assert!(
        differentiated.gradient_of_named("layer1").is_none(),
        "an untapped layer receives no gradient"
    );
}

#[proxima::test]
async fn grad_layer_norm_matches_finite_differences_for_x_weight_and_bias() {
    let (s, d) = (3u32, 4u32);
    let mut program = Vec::new();
    let x = input(&mut program, "x", &[s, d]);
    let weight = input(&mut program, "weight", &[d]);
    let bias = input(&mut program, "bias", &[d]);
    let out = layer_norm(&mut program, x, weight, bias, d, 1e-5).expect("layer norm builds");
    let loss = weighted_loss(&mut program, out, "sd", &[s, d]);
    let bound = [
        ("x", Lcg::values(11, 12)),
        ("weight", Lcg::values(12, 4)),
        ("bias", Lcg::values(13, 4)),
        ("w", Lcg::values(14, 12)),
    ];
    let reports = check(
        "layer_norm",
        &program,
        loss,
        &bound,
        &["x", "weight", "bias"],
    );
    assert_eq!(reports.len(), 3);
}

// cross-attention, non-Whisper shape: s = 3 queries over t = 5 memory positions.
const CROSS_S: u32 = 3;
const CROSS_T: u32 = 5;
const CROSS_U: u32 = 2;
const CROSS_I: u32 = 4;
const CROSS_D: u32 = 3;
const CROSS_M: u32 = 6;
const CROSS_SCALE: f32 = 0.5;

fn cross_values() -> Vec<(&'static str, Vec<f32>)> {
    let (u, i, d, m) = (CROSS_U, CROSS_I, CROSS_D, CROSS_M);
    vec![
        ("q", Lcg::values(21, (CROSS_S * u * i) as usize)),
        ("memory", Lcg::values(22, (CROSS_T * m) as usize)),
        ("k_weight", Lcg::values(23, (m * u * i) as usize)),
        ("k_bias", Lcg::values(24, (u * i) as usize)),
        ("v_weight", Lcg::values(25, (m * u * d) as usize)),
    ]
}

/// graph that projects K and V from the memory (`cross_kv`; v has no bias) and
/// attends; returns the program, the k and v nodes and the attention output.
fn cross_recomputed() -> (Vec<Op>, NodeId, NodeId, NodeId) {
    let (s, t, u, i, d, m) = (CROSS_S, CROSS_T, CROSS_U, CROSS_I, CROSS_D, CROSS_M);
    let mut program = Vec::new();
    let q = input(&mut program, "q", &[s, u, i]);
    let memory = input(&mut program, "memory", &[t, m]);
    let k_weight = input(&mut program, "k_weight", &[m, u, i]);
    let k_bias = input(&mut program, "k_bias", &[u, i]);
    let v_weight = input(&mut program, "v_weight", &[m, u, d]);
    let (k, v) = cross_kv(&mut program, memory, k_weight, Some(k_bias), v_weight, None)
        .expect("cross kv builds");
    let scale = scalar_constant_like(&mut program, CROSS_SCALE, q).expect("scale builds");
    let out = non_causal_attention(&mut program, q, k, v, scale).expect("attention builds");
    (program, k, v, out)
}

/// graph that binds cached `cross_k`/`cross_v` as inputs and runs the same attention.
fn cross_cached() -> (Vec<Op>, NodeId) {
    let (s, t, u, i, d) = (CROSS_S, CROSS_T, CROSS_U, CROSS_I, CROSS_D);
    let mut program = Vec::new();
    let q = input(&mut program, "q", &[s, u, i]);
    let k = input(&mut program, "cross_k", &[t, u, i]);
    let v = input(&mut program, "cross_v", &[t, u, d]);
    let scale = scalar_constant_like(&mut program, CROSS_SCALE, q).expect("scale builds");
    let out = non_causal_attention(&mut program, q, k, v, scale).expect("attention builds");
    (program, out)
}

/// `sum(out * out)` as a scalar loss node.
fn sum_of_squares(program: &mut Vec<Op>, out: NodeId) -> NodeId {
    let axes = "sugd";
    let squared = elementwise_like(
        program,
        out,
        ScalarOp::Multiply,
        &[(out, "sugd->sugd"), (out, "sugd->sugd")],
    )
    .expect("square builds");
    reduce_like(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        squared,
        "sugd->sugd",
        alloc::format!("->{axes}").as_str(),
    )
    .expect("loss reduce builds")
}

fn evaluate_nodes(program: &[Op], bound: &[(&str, Vec<f32>)], nodes: &[NodeId]) -> Vec<Vec<f32>> {
    let owned: Vec<(String, Vec<f32>)> = bound
        .iter()
        .map(|(n, v)| ((*n).into(), v.clone()))
        .collect();
    let evaluated = proxima_tensor::cpu::evaluate_named(program, &[], &borrowed(&owned), nodes)
        .expect("program evaluates");
    nodes
        .iter()
        .map(|&node| evaluated.get(node).expect("node requested").0.to_vec())
        .collect()
}

fn named<'a>(bound: &'a [(&'static str, Vec<f32>)], name: &str) -> &'a [f32] {
    &bound.iter().find(|(n, _)| *n == name).expect("bound").1
}

/// plain-loop reference: `softmax_t(scale * q.k) . v`, k and v projected from the memory.
fn cross_reference(bound: &[(&'static str, Vec<f32>)]) -> Vec<f32> {
    let (s, t, u, i, d, m) = (
        CROSS_S as usize,
        CROSS_T as usize,
        CROSS_U as usize,
        CROSS_I as usize,
        CROSS_D as usize,
        CROSS_M as usize,
    );
    let (q, memory) = (named(bound, "q"), named(bound, "memory"));
    let (wk, bk, wv) = (
        named(bound, "k_weight"),
        named(bound, "k_bias"),
        named(bound, "v_weight"),
    );
    let mut k = vec![0f32; t * u * i];
    let mut v = vec![0f32; t * u * d];
    for pos in 0..t {
        for head in 0..u {
            for col in 0..i {
                let dot: f32 = (0..m)
                    .map(|c| memory[pos * m + c] * wk[(c * u + head) * i + col])
                    .sum();
                k[(pos * u + head) * i + col] = dot + bk[head * i + col];
            }
            for col in 0..d {
                let dot: f32 = (0..m)
                    .map(|c| memory[pos * m + c] * wv[(c * u + head) * d + col])
                    .sum();
                v[(pos * u + head) * d + col] = dot;
            }
        }
    }
    let mut out = vec![0f32; s * u * d];
    for row in 0..s {
        for head in 0..u {
            let scores: Vec<f32> = (0..t)
                .map(|pos| {
                    CROSS_SCALE
                        * (0..i)
                            .map(|c| q[(row * u + head) * i + c] * k[(pos * u + head) * i + c])
                            .sum::<f32>()
                })
                .collect();
            let peak = scores.iter().cloned().fold(f32::MIN, f32::max);
            let weights: Vec<f32> = scores.iter().map(|x| (x - peak).exp()).collect();
            let total: f32 = weights.iter().sum();
            for col in 0..d {
                out[(row * u + head) * d + col] = (0..t)
                    .map(|pos| weights[pos] / total * v[(pos * u + head) * d + col])
                    .sum();
            }
        }
    }
    out
}

/// spec metric (R10): `max|g_ad - g_fd| / max|g_fd|` at `h = 5e-3`, returns (elements, error).
fn spec_gradient_error(
    program: &[Op],
    loss: NodeId,
    bound: &[(&'static str, Vec<f32>)],
    target: &str,
) -> (usize, f32) {
    const SPEC_STEP: f32 = 5e-3;
    let differentiated = crate::adjoint::differentiate(program, loss).expect("differentiates");
    let grad = differentiated
        .gradient_of_named(target)
        .unwrap_or_else(|| panic!("{target} feeds the loss and has a gradient"));
    let analytic = evaluate_nodes(&differentiated.program, bound, &[grad]).remove(0);
    let slot = bound.iter().position(|(n, _)| *n == target).expect("bound");
    assert_eq!(
        analytic.len(),
        bound[slot].1.len(),
        "{target} gradient size"
    );
    let mut probe = bound.to_vec();
    let (mut worst, mut largest) = (0f32, 0f32);
    for index in 0..analytic.len() {
        let original = bound[slot].1[index];
        probe[slot].1[index] = original + SPEC_STEP;
        let plus = evaluate_nodes(program, &probe, &[loss])[0][0];
        probe[slot].1[index] = original - SPEC_STEP;
        let minus = evaluate_nodes(program, &probe, &[loss])[0][0];
        probe[slot].1[index] = original;
        let numeric = (plus - minus) / (2.0 * SPEC_STEP);
        worst = worst.max((analytic[index] - numeric).abs());
        largest = largest.max(numeric.abs());
    }
    assert!(
        largest > 0.0,
        "{target}: central-difference gradient is identically zero"
    );
    (analytic.len(), worst / largest)
}

#[proxima::test]
async fn grad_cross_attention_forward_matches_a_naive_reference() {
    let bound = cross_values();
    let (program, _, _, out) = cross_recomputed();
    let got = evaluate_nodes(&program, &bound, &[out]).remove(0);
    let want = cross_reference(&bound);
    assert_eq!(got.len(), want.len());
    assert!(!want.is_empty(), "compared zero elements");
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    std::println!("cross_attention forward n={} max_abs={worst:e}", got.len());
    assert!(
        worst < 1e-5,
        "forward differs from the reference by {worst}"
    );
}

#[proxima::test]
async fn grad_cross_attention_gradient_to_query_and_memory_matches_central_differences() {
    let bound = cross_values();
    let (mut program, _, _, out) = cross_recomputed();
    let loss = sum_of_squares(&mut program, out);
    for target in ["q", "memory"] {
        let (count, error) = spec_gradient_error(&program, loss, &bound, target);
        std::println!("cross_attention wrt={target} n={count} max|g_ad-g_fd|/max|g_fd|={error:e}");
        assert!(count > 0, "{target}: compared zero elements");
        assert!(error <= 5e-4, "{target}: error {error} exceeds 5e-4");
    }
}

#[proxima::test]
async fn grad_cross_attention_cached_and_recomputed_kv_give_equal_outputs_and_query_gradients() {
    let bound = cross_values();
    let (mut recomputed, k, v, out) = cross_recomputed();
    let kv = evaluate_nodes(&recomputed, &bound, &[k, v]);
    let loss = sum_of_squares(&mut recomputed, out);
    let (mut cached, cached_out) = cross_cached();
    let cached_loss = sum_of_squares(&mut cached, cached_out);
    let cached_bound = vec![
        ("q", named(&bound, "q").to_vec()),
        ("cross_k", kv[0].clone()),
        ("cross_v", kv[1].clone()),
    ];
    let from_graph = evaluate_nodes(&recomputed, &bound, &[out]).remove(0);
    let from_cache = evaluate_nodes(&cached, &cached_bound, &[cached_out]).remove(0);
    let output_gap = from_graph
        .iter()
        .zip(&from_cache)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let query_grad = |program: &[Op], loss: NodeId, values: &[(&str, Vec<f32>)]| {
        let differentiated = crate::adjoint::differentiate(program, loss).expect("differentiates");
        let grad = differentiated
            .gradient_of_named("q")
            .expect("q has a gradient");
        evaluate_nodes(&differentiated.program, values, &[grad]).remove(0)
    };
    let grad_graph = query_grad(&recomputed, loss, &bound);
    let grad_cache = query_grad(&cached, cached_loss, &cached_bound);
    let grad_gap = grad_graph
        .iter()
        .zip(&grad_cache)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    std::println!(
        "cross_attention cached-vs-recomputed outputs n={} max_abs={output_gap:e}; q grad n={} max_abs={grad_gap:e}",
        from_graph.len(),
        grad_graph.len()
    );
    assert!(
        !from_graph.is_empty() && !grad_graph.is_empty(),
        "compared zero elements"
    );
    assert!(output_gap <= 1e-6, "outputs differ by {output_gap}");
    assert!(grad_gap <= 1e-6, "query gradients differ by {grad_gap}");
}
