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
    RopePairing, elementwise_like, fused_rope_pair, layer_norm, layer_norm_no_affine, modulate,
    multi_axis_rope_tables, non_causal_attention, reduce_like, scalar_constant_like, tap_layers,
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
