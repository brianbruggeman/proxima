#![cfg(all(
    feature = "metal",
    target_os = "macos",
    not(feature = "metal-q4k-mask-fma")
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use omega::Codec;
use proxima_numeric::{DType, NumericPolicy};
use proxima_tensor::{Extent, IndexMap, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp, append, projection};

const TOKENS: u32 = 1;
const IN_DIM: u32 = 512;
const OUT_ROWS: u32 = 24;

fn matvec_program(negated: bool) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(&mut program, Op::Input { dtype: DType::UINT8, shape: vec![Extent::Static(OUT_ROWS), Extent::Static(IN_DIM)], name: None });
    let activation = append(&mut program, Op::Input { dtype: DType::FLOAT32, shape: vec![Extent::Static(TOKENS), Extent::Static(IN_DIM)], name: None });
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(3, &[1, 2]))),
                (activation, IndexMap::Affine(projection(3, &[0, 2]))),
            ],
            name: None,
        },
    );
    let reduced = if negated {
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
                body: ScalarOp::Negate,
                operands: vec![(product, IndexMap::Affine(projection(3, &[0, 1, 2])))],
                name: None,
            },
        )
    } else {
        product
    };
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: reduced,
            in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, sum)
}

fn emit_msl(codec: Codec, negated: bool) -> omega::Kernel {
    let (program, sum) = matvec_program(negated);
    let shapes = proxima_tensor::infer(&program, &[]).expect("the synthetic program infers");
    let packed: omega::PackedOperands = [(NodeId(0), codec)].into_iter().collect();
    let mut bound = proxima_tensor::bind(&program, &shapes, &[sum], NumericPolicy::default()).expect("binds");
    proxima_tensor::correct_packed_matmul_layouts(&mut bound, &[NodeId(0)].into_iter().collect());
    let resolved = bound.iter().find(|op| op.node == sum).expect("the reduce node is bound");
    omega::emit(resolved, &packed, NumericPolicy::default()).expect("emits")
}

fn fixture(file_name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/packed_row_codecs").join(file_name);
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("fixture {} is readable: {error}", path.display()))
}

fn first_difference(actual: &str, expected: &str) -> Option<String> {
    let mut actual_lines = actual.lines();
    let mut expected_lines = expected.lines();
    let mut line_number = 0usize;
    loop {
        line_number += 1;
        match (actual_lines.next(), expected_lines.next()) {
            (None, None) => return (actual != expected).then(|| "line endings or trailing bytes differ".to_string()),
            (got, want) if got != want => {
                return Some(format!("line {line_number}: emitted {got:?} but fixture has {want:?}"));
            }
            _ => {}
        }
    }
}

/// `push_packed_row_blocked_body` reduces each output row with four `reduced_row{n}` simd sums; the
/// cooperative per-element kernel has none.
const ROW_BLOCKED_MARKER: &str = "reduced_row0";

/// A K-split row-blocked body (the scalar-run codecs at the default `ksplit_max`) reduces its rows
/// through a threadgroup `partial_sums` table instead of the `reduced_row{n}` simd sums.
const K_SPLIT_MARKER: &str = "partial_sums";

#[proxima::test]
#[case::q3k_plain("q3k_plain", Codec::Q3K, false, true)]
#[case::q3k_negated("q3k_negated", Codec::Q3K, true, true)]
#[case::q4k_plain("q4k_plain", Codec::Q4K, false, true)]
#[case::q4k_negated("q4k_negated", Codec::Q4K, true, true)]
#[case::q5k_plain("q5k_plain", Codec::Q5K, false, true)]
#[case::q5k_negated("q5k_negated", Codec::Q5K, true, true)]
#[case::q6k_plain("q6k_plain", Codec::Q6K, false, true)]
#[case::q6k_negated("q6k_negated", Codec::Q6K, true, true)]
#[case::q8_0_plain("q8_0_plain", Codec::Q8_0, false, true)]
#[case::q8_0_negated("q8_0_negated", Codec::Q8_0, true, true)]
#[case::q4_0_plain("q4_0_plain", Codec::Q4_0, false, true)]
#[case::q4_0_negated("q4_0_negated", Codec::Q4_0, true, true)]
#[case::fp6e3m2_plain("fp6e3m2_plain", Codec::Fp6E3M2, false, true)]
#[case::fp6e3m2_negated("fp6e3m2_negated", Codec::Fp6E3M2, true, true)]
#[case::iq4xs_plain("iq4xs_plain", Codec::Iq4Xs, false, true)]
#[case::iq4xs_negated_declines_row_blocked("iq4xs_negated", Codec::Iq4Xs, true, false)]
#[case::iq3s_plain("iq3s_plain", Codec::Iq3S, false, true)]
#[case::iq3s_negated_declines_row_blocked("iq3s_negated", Codec::Iq3S, true, false)]
#[case::f16_plain_takes_the_scalar_run_body("f16_plain", Codec::Float16, false, true)]
#[case::bf16_plain_takes_the_scalar_run_body("bf16_plain", Codec::BFloat16, false, true)]async fn single_token_matvec_emits_byte_identical_msl_to_the_pinned_fixture(
    #[case] name: &str,
    #[case] codec: Codec,
    #[case] negated: bool,
    #[case] row_blocked: bool,
) {
    let kernel = emit_msl(codec, negated);
    let expected = fixture(&format!("{name}.msl"));

    if let Some(difference) = first_difference(&kernel.source, &expected) {
        panic!("{name}: MSL drifted from tests/fixtures/packed_row_codecs/{name}.msl, first difference at {difference}");
    }
    assert_eq!(
        kernel.source.contains(ROW_BLOCKED_MARKER) || kernel.source.contains(K_SPLIT_MARKER),
        row_blocked,
        "{name}: kernel kind (row-blocked = {row_blocked}) changed; `{ROW_BLOCKED_MARKER}` or `{K_SPLIT_MARKER}` marks the row-blocked body"
    );
}

#[test]
fn first_difference_names_the_line_of_the_first_mismatch() {
    let difference = first_difference("alpha\nbeta\ngamma\n", "alpha\nbeta\ndelta\n").expect("the third line differs");
    assert!(difference.starts_with("line 3:"), "{difference}");
    assert_eq!(first_difference("alpha\nbeta\n", "alpha\nbeta\n"), None);
}
