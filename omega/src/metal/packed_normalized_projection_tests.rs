use proxima_gguf::quant::q8_0;
use proxima_numeric::{DType, NumericPolicy};
use proxima_tensor::cpu::QuantizedBlock;
use proxima_tensor::{
    BoundOpKind, Extent, IndexMap, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp, append,
    projection,
};

use super::prepare_uniforms_pack::prepare;
use crate::Codec;

const KV_HEADS: u32 = 4;
const QUERY_GROUPS: u32 = 8;
const HEAD_DIM: u32 = 128;
// one output row keeps the normalized activation (4096 elements) at least as large as the
// reduce's iteration space, so the binder's broadcast quarantine leaves it fused unless the
// packed weight is named
const OUTPUT_ROWS: u32 = 1;

/// flint's attention output projection: a Q8_0 weight contracted over
/// `[kv_heads, query_groups, head_dim]` against the attention result, which the
/// graph builds as `values * reciprocal(softmax denominator)` (iteration axes
/// `(token, unit, group, dim, row)`).
fn normalized_attention_output_program() -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::UINT8,
            shape: vec![
                Extent::Static(OUTPUT_ROWS),
                Extent::Static(KV_HEADS),
                Extent::Static(QUERY_GROUPS),
                Extent::Static(HEAD_DIM),
            ],
            name: None,
        },
    );
    let values = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![
                Extent::Static(1),
                Extent::Static(KV_HEADS),
                Extent::Static(QUERY_GROUPS),
                Extent::Static(HEAD_DIM),
            ],
            name: None,
        },
    );
    let denominator = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![
                Extent::Static(1),
                Extent::Static(KV_HEADS),
                Extent::Static(QUERY_GROUPS),
            ],
            name: None,
        },
    );
    let inverse = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Reciprocal,
            operands: vec![(denominator, IndexMap::Affine(projection(3, &[0, 1, 2])))],
            name: None,
        },
    );
    let normalized = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (values, IndexMap::Affine(projection(4, &[0, 1, 2, 3]))),
                (inverse, IndexMap::Affine(projection(4, &[0, 1, 2]))),
            ],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(5, &[4, 1, 2, 3]))),
                (normalized, IndexMap::Affine(projection(5, &[0, 1, 2, 3]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(5, &[0, 1, 2, 3, 4])),
            out_map: IndexMap::Affine(projection(5, &[0, 4])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, sum)
}

fn smooth_signal(count: usize, phase: f32) -> Vec<f32> {
    (0..count)
        .map(|index| (index as f32 * 0.37 + phase).sin())
        .collect()
}

#[test]
fn a_normalized_attention_output_reduce_binds_as_a_bare_packed_product() {
    let in_dim = (KV_HEADS * QUERY_GROUPS * HEAD_DIM) as usize;
    let row_bytes = in_dim / q8_0::QK8_0 * q8_0::BLOCK_BYTES;
    let mut packed = vec![0u8; OUTPUT_ROWS as usize * row_bytes];
    for (row, chunk) in packed.chunks_exact_mut(row_bytes).enumerate() {
        q8_0::quantize(&smooth_signal(in_dim, row as f32), chunk).expect("q8_0 quantizes a row");
    }
    let values = smooth_signal(in_dim, 0.5);
    let denominator: Vec<f32> = (0..KV_HEADS * QUERY_GROUPS)
        .map(|index| 1.0 + index as f32)
        .collect();
    let blocks = [
        QuantizedBlock::Packed {
            codec: Codec::Q8_0,
            bytes: &packed,
        },
        QuantizedBlock::Float32(&values),
        QuantizedBlock::Float32(&denominator),
    ];
    let (program, sum) = normalized_attention_output_program();

    let prepared = prepare(
        &program,
        &[],
        &blocks,
        &[sum],
        NumericPolicy::default(),
        &[],
        (true, false),
    )
    .expect("the normalized attention output prepares");

    let bound = prepared
        .resolved
        .iter()
        .find(|bound| bound.node == sum)
        .expect("the reduce survives binding");
    let BoundOpKind::Reduce { .. } = &bound.kind else {
        panic!("node {sum:?} binds as a reduce");
    };
    assert_eq!(
        bound.operands().len(),
        2,
        "the softmax normalization must materialize ahead of the packed weight reduce, \
         not fuse into it as a third operand"
    );
    let quantized: Vec<Option<Codec>> = bound
        .operands()
        .iter()
        .map(|(node, _, _)| prepared.packed_operands.get(node).copied())
        .collect();
    assert!(
        crate::msl::packed_row_block(bound, &quantized).is_some(),
        "a bare Q8_0 weight times one activation over three folded reduce axes takes the row-blocked kernel"
    );
}
