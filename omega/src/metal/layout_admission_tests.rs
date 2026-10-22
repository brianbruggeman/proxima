use std::collections::{BTreeMap, BTreeSet};

use proxima_primitives::float_layout::{FloatLayout, SignEncoding, Specials};
use proxima_tensor::{
    AxisTerm, DType, Extent, IndexMap, NodeId, NumericPolicy, Op, ScalarOp, TensorError, append,
    bind, infer, map,
};

use super::prepare_uniforms_pack::{device_element_bytes, reject_unsupported_gpu_dtype};
use crate::msl::emit;

const E4M3_FROM_FIELDS: FloatLayout = FloatLayout {
    exponent_bits: 4,
    mantissa_bits: 3,
    specials: Specials::None,
    bias_override: None,
    sign: SignEncoding::Magnitude,
};

fn negate_program(dtype: DType) -> Vec<Op> {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype,
            shape: vec![Extent::Static(4), Extent::Static(8)],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype,
            body: ScalarOp::Negate,
            operands: vec![(source, IndexMap::Affine(map::projection(2, &[0, 1])))],
            name: None,
        },
    );
    program
}

fn gathered_program(table_dtype: DType, indices_dtype: DType) -> Vec<Op> {
    let mut program = Vec::new();
    let table = append(
        &mut program,
        Op::Input {
            dtype: table_dtype,
            shape: vec![Extent::Static(8)],
            name: None,
        },
    );
    let indices = append(
        &mut program,
        Op::Input {
            dtype: indices_dtype,
            shape: vec![Extent::Static(4)],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype: table_dtype,
            body: ScalarOp::Negate,
            operands: vec![(
                table,
                IndexMap::Computed {
                    indices,
                    index_map: map::projection(1, &[0]),
                    base: map::IndexPattern {
                        iter_rank: 1,
                        axes: vec![map::AxisIndex {
                            terms: core::iter::once(AxisTerm::projection(0)).collect(),
                            offset: 0,
                            len: None,
                        }],
                    },
                    gathered_dim: 0,
                },
            )],
            name: None,
        },
    );
    program
}

fn emitted_source(program: &[Op]) -> String {
    let shapes = infer(program, &[]).expect("the negate program infers");
    let terminal = NodeId((program.len() - 1) as u32);
    let bound = bind(program, &shapes, &[terminal], NumericPolicy::default())
        .expect("the negate program lowers")
        .into_iter()
        .next()
        .expect("one bound op");
    emit(&bound, &BTreeMap::new(), NumericPolicy::default())
        .expect("the layout operand emits")
        .source
}

fn expected_token(layout: &FloatLayout) -> &'static str {
    if *layout == FloatLayout::F16 {
        "half"
    } else {
        "float"
    }
}

#[proxima::test]
#[case::binary32(FloatLayout::BF32)]
#[case::binary16(FloatLayout::F16)]
#[case::brain16(FloatLayout::BF16)]
#[case::e5m2_saturating(FloatLayout::E5M2_SATURATING)]
#[case::e2m1(FloatLayout::E2M1)]
#[case::e3m2(FloatLayout::E3M2)]
#[case::e2m3(FloatLayout::E2M3)]
#[case::e4m3_from_fields(E4M3_FROM_FIELDS)]
async fn float_layout_admit_planner_accepts_layout_operand_and_output(#[case] layout: FloatLayout) {
    let program = negate_program(DType::Layout(layout));

    reject_unsupported_gpu_dtype(&program, &BTreeSet::new())
        .expect("a float layout is admitted as operand and output");

    let source = emitted_source(&program);
    let token = expected_token(&layout);
    assert!(
        source.contains(&format!("device const {token}")),
        "the operand buffer is declared {token}:\n{source}"
    );
    assert!(
        source.contains(&format!("device {token}")),
        "the output buffer is declared {token}:\n{source}"
    );
}

#[test]
fn float_layout_admit_rejects_float64() {
    let program = negate_program(DType::FLOAT64);

    let rejection = reject_unsupported_gpu_dtype(&program, &BTreeSet::new());

    assert!(
        matches!(rejection, Err(TensorError::NotLowerable { .. })),
        "{rejection:?}"
    );
}

#[test]
fn float_layout_admit_rejects_int128_operand() {
    let program = negate_program(DType::INT128);

    let rejection = reject_unsupported_gpu_dtype(&program, &BTreeSet::new());

    assert!(
        matches!(rejection, Err(TensorError::NotLowerable { .. })),
        "{rejection:?}"
    );
}

#[test]
fn float_layout_admit_keeps_gather_index_exemption() {
    let program = gathered_program(DType::Layout(E4M3_FROM_FIELDS), DType::INT32);

    reject_unsupported_gpu_dtype(&program, &BTreeSet::new())
        .expect("a gather's Int32 indices stay exempt");

    let not_an_index = negate_program(DType::INT32);
    assert!(
        reject_unsupported_gpu_dtype(&not_an_index, &BTreeSet::new()).is_err(),
        "Int32 as a non-index operand stays rejected"
    );
}

#[test]
fn float_layout_admit_buffer_bytes_derive_from_layout_fields() {
    let binary16_by_fields = FloatLayout {
        exponent_bits: 5,
        mantissa_bits: 10,
        specials: Specials::InfinityOnOverflow,
        bias_override: Some(15),
        sign: SignEncoding::Magnitude,
    };
    let cases = [
        (DType::Layout(FloatLayout::BF32), 4),
        (DType::Layout(FloatLayout::F16), 2),
        (DType::Layout(binary16_by_fields), 2),
        (DType::Layout(FloatLayout::BF16), 4),
        (DType::Layout(FloatLayout::E5M2_SATURATING), 4),
        (DType::Layout(FloatLayout::E2M1), 4),
        (DType::Layout(E4M3_FROM_FIELDS), 4),
    ];

    let observed = cases.map(|(dtype, _)| device_element_bytes(dtype));

    assert_eq!(observed, cases.map(|(_, bytes)| bytes));
}
