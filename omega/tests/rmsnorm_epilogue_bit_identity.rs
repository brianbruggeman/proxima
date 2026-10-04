//! Does folding gemma4's RMSNorm apply into its sum-of-squares reduce change
//! the bits the GPU produces? Runs the same norm chain (`x * inv_rms * gamma`,
//! plus a residual add on the hidden-width post-norms) twice per policy and
//! shape: once with the sum-of-squares requested as an extra output root,
//! which keeps the reduce and the apply as two dispatches, and once with only
//! the real output requested, which lets the reduce-epilogue pass fold the
//! apply into the reduce. The shapes are the ones gemma4-E2B decode runs:
//! `[1, 1536]` hidden norms, `[1, 8, 256]` per-head Q norm, `[1, 1, 256]` K
//! norm, and `[1, 8, 512]` global-layer Q norm.

#![cfg(all(
    feature = "metal",
    feature = "reduce-epilogue-fusion",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    BoundOp, BoundOpKind, DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, Reduce,
    ReduceInit, ScalarOp, append, bind, infer, projection,
};

mod support;
use support::as_named_blocks;

const TEN_ULP_RELATIVE: f32 = 1.2e-6;

#[derive(Clone, Copy)]
struct Shape {
    label: &'static str,
    heads: Option<u32>,
    dim: u32,
    residual: bool,
}

impl Shape {
    fn extents(self) -> Vec<Extent> {
        let mut extents = vec![Extent::Static(1)];
        extents.extend(self.heads.map(Extent::Static));
        extents.push(Extent::Static(self.dim));
        extents
    }

    fn rank(self) -> u16 {
        2 + u16::from(self.heads.is_some())
    }

    fn elements(self) -> usize {
        (self.heads.unwrap_or(1) * self.dim) as usize
    }

    fn full(self) -> IndexMap {
        let axes: Vec<u16> = (0..self.rank()).collect();
        IndexMap::Affine(projection(self.rank(), &axes))
    }

    fn per_row(self) -> IndexMap {
        let axes: Vec<u16> = (0..self.rank() - 1).collect();
        IndexMap::Affine(projection(self.rank() - 1, &axes))
    }

    fn scalar(self) -> IndexMap {
        IndexMap::Affine(projection(self.rank() - 1, &[]))
    }

    fn row_over_dim(self) -> IndexMap {
        let axes: Vec<u16> = (0..self.rank() - 1).collect();
        IndexMap::Affine(projection(self.rank(), &axes))
    }

    fn dim_over_rows(self) -> IndexMap {
        IndexMap::Affine(projection(self.rank(), &[self.rank() - 1]))
    }
}

const SHAPES: [Shape; 5] = [
    Shape {
        label: "hidden_residual",
        heads: None,
        dim: 1536,
        residual: true,
    },
    Shape {
        label: "hidden_plain",
        heads: None,
        dim: 1536,
        residual: false,
    },
    Shape {
        label: "q_heads",
        heads: Some(8),
        dim: 256,
        residual: false,
    },
    Shape {
        label: "k_head",
        heads: Some(1),
        dim: 256,
        residual: false,
    },
    Shape {
        label: "global_q_heads",
        heads: Some(8),
        dim: 512,
        residual: false,
    },
];

struct Chain {
    output: NodeId,
    sum_squares: NodeId,
}

fn input(program: &mut Vec<Op>, name: &str, shape: Vec<Extent>) -> NodeId {
    append(
        program,
        Op::Input {
            dtype: DType::Float32,
            shape,
            name: Some(name.into()),
        },
    )
}

fn elementwise(program: &mut Vec<Op>, body: ScalarOp, operands: Vec<(NodeId, IndexMap)>) -> NodeId {
    append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body,
            operands,
            name: None,
        },
    )
}

fn append_chain(program: &mut Vec<Op>, shape: Shape) -> Chain {
    let x = input(program, "x", shape.extents());
    let gamma = input(program, "gamma", vec![Extent::Static(shape.dim)]);
    let residual = input(program, "residual", shape.extents());
    let inv_dim = input(program, "inv_dim", Vec::new());
    let eps = input(program, "eps", Vec::new());
    let all_axes: Vec<u16> = (0..shape.rank()).collect();
    let row_axes: Vec<u16> = (0..shape.rank() - 1).collect();

    let squared = elementwise(
        program,
        ScalarOp::Multiply,
        vec![(x, shape.full()), (x, shape.full())],
    );
    let sum_squares = append(
        program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: squared,
            in_map: IndexMap::Affine(projection(shape.rank(), &all_axes)),
            out_map: IndexMap::Affine(projection(shape.rank(), &row_axes)),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    let mean = elementwise(
        program,
        ScalarOp::Multiply,
        vec![(sum_squares, shape.per_row()), (inv_dim, shape.scalar())],
    );
    let mean_eps = elementwise(
        program,
        ScalarOp::Add,
        vec![(mean, shape.per_row()), (eps, shape.scalar())],
    );
    let root = elementwise(
        program,
        ScalarOp::SquareRoot,
        vec![(mean_eps, shape.per_row())],
    );
    let inverse = elementwise(program, ScalarOp::Reciprocal, vec![(root, shape.per_row())]);
    let normed = elementwise(
        program,
        ScalarOp::Multiply,
        vec![(x, shape.full()), (inverse, shape.row_over_dim())],
    );
    let scaled = elementwise(
        program,
        ScalarOp::Multiply,
        vec![(normed, shape.full()), (gamma, shape.dim_over_rows())],
    );
    let output = match shape.residual {
        true => elementwise(
            program,
            ScalarOp::Add,
            vec![(scaled, shape.full()), (residual, shape.full())],
        ),
        false => scaled,
    };
    Chain {
        output,
        sum_squares,
    }
}

fn named_inputs(shape: Shape) -> Vec<(String, Vec<f32>)> {
    let mut lcg = Lcg(0x5eed_0000 + u64::from(shape.dim));
    let count = shape.elements();
    vec![
        (
            "x".to_string(),
            (0..count).map(|_| lcg.next_unit()).collect(),
        ),
        (
            "gamma".to_string(),
            (0..shape.dim)
                .map(|_| 1.0 + 0.1 * lcg.next_unit())
                .collect(),
        ),
        (
            "residual".to_string(),
            (0..count).map(|_| lcg.next_unit()).collect(),
        ),
        ("inv_dim".to_string(), vec![1.0 / shape.dim as f32]),
        ("eps".to_string(), vec![1e-6]),
    ]
}

fn epilogued(resolved: &[BoundOp]) -> usize {
    resolved
        .iter()
        .filter(|bound| matches!(&bound.kind, BoundOpKind::Reduce { epilogue_operands, .. } if !epilogue_operands.is_empty()))
        .count()
}

fn run_arm(policy: NumericPolicy, shape: Shape, fused: bool) -> (Vec<f32>, usize) {
    let mut program = Vec::new();
    let chain = append_chain(&mut program, shape);
    let roots = match fused {
        true => vec![chain.output],
        false => vec![chain.output, chain.sum_squares],
    };
    let shapes = infer(&program, &[]).expect("the rmsnorm chain infers");
    let resolved = bind(&program, &shapes, &roots, policy).expect("the rmsnorm chain binds");
    let owned = named_inputs(shape);
    let named = as_named_blocks(&owned);
    let plan =
        omega::plan_named(&program, &[], &named, &roots, policy).expect("metal plans the chain");
    let evaluated = omega::execute_plan_named(&plan, &named).expect("metal runs the chain");
    (evaluated.root().to_vec(), epilogued(&resolved))
}

struct Drift {
    bit_diff_count: usize,
    relative_to_peak: f32,
}

fn measure(policy_label: &str, policy: NumericPolicy, shape: Shape) -> Drift {
    let (unfused, unfused_epilogues) = run_arm(policy, shape, false);
    let (fused, fused_epilogues) = run_arm(policy, shape, true);
    let bit_diff_count = unfused
        .iter()
        .zip(&fused)
        .filter(|(lhs, rhs)| lhs.to_bits() != rhs.to_bits())
        .count();
    let max_abs = unfused
        .iter()
        .zip(&fused)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f32, f32::max);
    let peak = unfused
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    eprintln!(
        "rmsnorm bit identity policy={policy_label} shape={} unfused_epilogues={unfused_epilogues} \
         fused_epilogues={fused_epilogues} bit_diff={bit_diff_count}/{} max_abs={max_abs:e} peak={peak:e}",
        shape.label,
        shape.elements()
    );
    assert_eq!(
        unfused_epilogues, 0,
        "the extra sum-of-squares root must keep the pair unfused"
    );
    assert_eq!(fused_epilogues, 1, "the apply must fold into the reduce");
    Drift {
        bit_diff_count,
        relative_to_peak: max_abs / peak.max(f32::MIN_POSITIVE),
    }
}

#[test]
fn hidden_and_single_head_norms_keep_every_bit_under_the_bit_exact_policy() {
    let single_reduce_shapes = SHAPES
        .iter()
        .filter(|shape| shape.heads.is_none_or(|heads| heads == 1));

    for shape in single_reduce_shapes {
        let drift = measure("bit_exact", NumericPolicy::bit_exact(), *shape);

        assert_eq!(
            drift.bit_diff_count, 0,
            "shape={}: fusing changed bits",
            shape.label
        );
    }
}

#[test]
fn every_gemma4_norm_shape_stays_within_ten_ulp_of_the_unfused_pair() {
    for (label, policy) in [
        ("bit_exact", NumericPolicy::bit_exact()),
        ("llama_relaxed", NumericPolicy::llama_relaxed()),
    ] {
        for shape in SHAPES {
            let drift = measure(label, policy, shape);

            assert!(
                drift.relative_to_peak < TEN_ULP_RELATIVE,
                "policy={label} shape={}: relative drift {:e}",
                shape.label,
                drift.relative_to_peak
            );
        }
    }
}
