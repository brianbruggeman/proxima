//! A uniform expert source whose packed-row body is switched off must still
//! read its expert weights through the descriptor payload.
//!
//! `PROXIMA_PACKED_ROW_DISABLE` keeps a codec off the row-blocked matvec, so
//! the uniform lowering renders the codec's scalar `<codec>_element(in0 + ...)`
//! read instead of the packed-row body. That read has to be rewritten to the
//! descriptor-aware fetch, or it indexes the stripped plan operand and returns
//! the wrong weights. The environment is process-global, so this file holds one
//! test and gets its own process.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::expect_used)]

use std::collections::BTreeMap;

use proxima_gguf::quant::q3_k;
use proxima_numeric::codec::Codec;
use proxima_numeric::{DType, NumericPolicy};
use proxima_tensor::cpu::{
    ExpertEntry, ExpertSource, QuantizedBlock,
    evaluate_quantized_named_exact_with_scratch_and_experts,
};
use proxima_tensor::map::{self, AxisIndex, AxisTerm};
use proxima_tensor::{Extent, IndexMap, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp, append};

const EXPERTS: usize = 3;
const ROWS: usize = 512;
const WIDTH: usize = 8 * q3_k::QK_K;
const SEQUENCE: usize = 2;

fn gathered_expert_program() -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::UINT8,
            shape: vec![
                Extent::Static(EXPERTS as u32),
                Extent::Static(ROWS as u32),
                Extent::Static(WIDTH as u32),
            ],
            name: Some("weight".into()),
        },
    );
    let route = append(
        &mut program,
        Op::Input {
            dtype: DType::INT32,
            shape: vec![Extent::Static(SEQUENCE as u32)],
            name: Some("route".into()),
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(SEQUENCE as u32), Extent::Static(WIDTH as u32)],
            name: Some("activation".into()),
        },
    );
    let gather = IndexMap::Computed {
        indices: route,
        index_map: map::projection(3, &[0]),
        base: map::IndexPattern {
            iter_rank: 3,
            axes: vec![
                AxisIndex::default(),
                AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(1)).collect(),
                    offset: 0,
                    len: None,
                },
                AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(2)).collect(),
                    offset: 0,
                    len: None,
                },
            ],
        },
        gathered_dim: 0,
    };
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, gather),
                (activation, IndexMap::Affine(map::projection(3, &[0, 2]))),
            ],
            name: Some("uniform_product".into()),
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("uniform_reduce".into()),
        }),
    );
    (program, sum)
}

fn expert_weights(seed: usize) -> Vec<f32> {
    (0..ROWS * WIDTH)
        .map(|index| ((index * (seed * 11 + 3) % 127) as f32 - 63.0) * 0.015)
        .collect()
}

fn pack_q3k(values: &[f32]) -> Vec<u8> {
    let mut packed = vec![0_u8; ROWS * (WIDTH / q3_k::QK_K) * q3_k::BLOCK_BYTES];
    q3_k::quantize(values, &mut packed).expect("the Q3_K fixture has whole super-block rows");
    packed
}

#[test]
fn uniform_q3k_expert_source_without_packed_rows_matches_cpu() {
    let (program, output) = gathered_expert_program();
    let packed: Vec<Vec<u8>> = (1..=EXPERTS)
        .map(|seed| pack_q3k(&expert_weights(seed)))
        .collect();
    let placeholder = packed.concat();
    let routes = [2.0_f32, 0.0_f32];
    let activations: Vec<f32> = (0..SEQUENCE * WIDTH)
        .map(|index| ((index * 17 % 97) as f32 - 48.0) * 0.02)
        .collect();
    let named = [
        (
            "weight",
            QuantizedBlock::Packed {
                codec: Codec::Q3K,
                bytes: &placeholder,
            },
        ),
        ("route", QuantizedBlock::Float32(&routes)),
        ("activation", QuantizedBlock::Float32(&activations)),
    ];
    let plan = temp_env::with_var("PROXIMA_PACKED_ROW_DISABLE", Some("q3k"), || {
        omega::plan_named(&program, &[], &named, &[output], NumericPolicy::default())
    });
    let plan = plan.expect("the Q3_K placeholder supplies the plan's static shape");
    let entries: Vec<ExpertEntry<'_>> = packed
        .iter()
        .enumerate()
        .map(|(expert, bytes)| ExpertEntry {
            block: QuantizedBlock::Packed {
                codec: Codec::Q3K,
                bytes,
            },
            out_dim: ROWS as u32,
            in_dim: WIDTH as u32,
            epoch: expert as u64,
        })
        .collect();
    let selected = [0_u32, 2_u32];
    let source = ExpertSource::with_selected_expert_ids(&entries, &selected);
    let sources = BTreeMap::from([(NodeId(0), source)]);

    let mut cpu_scratch = proxima_tensor::cpu::BufferPools::default();
    let mut validated_weight_nodes = None;
    let expected = evaluate_quantized_named_exact_with_scratch_and_experts(
        &program,
        &[],
        &named,
        &[output],
        &mut cpu_scratch,
        &mut validated_weight_nodes,
        Some(&sources),
    )
    .expect("the CPU evaluator supplies the uniform Q3_K reference");
    let evaluated = omega::metal::execute_plan_named_with_expert_sources(&plan, &named, &sources)
        .expect("the uniform Q3_K source table executes on Metal");
    assert_eq!(evaluated.root().len(), SEQUENCE * ROWS);
    for (index, (actual, reference)) in evaluated.root().iter().zip(expected.root()).enumerate() {
        assert!(
            (actual - reference).abs() <= 1.0e-4,
            "uniform Q3_K output {index} differs: metal={actual} cpu={reference}"
        );
    }
}
