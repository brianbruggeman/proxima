use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp, append, bind, infer,
    map,
};

use super::*;

const DECODER_START: &str = "static inline float scaled_e3m2_field";

pub fn matmul_op(m: u32, k: u32, n: u32) -> BoundOp {
    let mut program = Vec::new();
    let lhs = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(m), Extent::Static(k)],
            name: None,
        },
    );
    let rhs = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(k), Extent::Static(n)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("matmul".into()),
        }),
    );
    let shapes = infer(&program, &[]).expect("matmul infers");
    bind(
        &program,
        &shapes,
        &[NodeId((program.len() - 1) as u32)],
        NumericPolicy::default(),
    )
    .expect("matmul lowers")
    .into_iter()
    .next()
    .expect("one fused bound emitted")
}

pub fn e3m2_kernel_source(bound: &BoundOp) -> String {
    let packed = BTreeMap::from([(bound.operands()[0].0, Codec::Fp6E3M2)]);
    emit(bound, &packed, NumericPolicy::default())
        .expect("an e3m2 kernel emits")
        .source
}

pub fn full_matvec_source(rows: u32, reduction: u32) -> String {
    e3m2_kernel_source(&matmul_op(rows, reduction, 1))
}

pub fn matvec_source(rows: u32, reduction: u32) -> String {
    let source = full_matvec_source(rows, reduction);
    let start = source
        .find(DECODER_START)
        .expect("the e3m2 decoders are spliced in");
    source[start..].to_string()
}
