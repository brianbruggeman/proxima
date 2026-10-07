//! A warm decode step allocates no device buffers: the router's extra
//! outputs bind plan-owned storage, and every gathered dispatch's fault
//! buffer is recycled once the step's checks have read it.
//!
//! Counted through `OUTPUT_BUFFER_ALLOCATIONS` (`metal_stage_totals`), which
//! `allocate_buffer` and `allocate_fault_buffer` both increment on a fresh
//! `newBuffer`. A granite decode step reported 360 of them (the fused top-k's
//! extra outputs) before any fix, and a further 576, one fault buffer per
//! gathered expert dispatch, once fault buffers were counted too.

#![cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use omega::metal::metal_stage_totals;

mod lookup {
    use omega::MetalError;
    use proxima_tensor::{
        AxisIndex, AxisTerm, DType, Extent, IndexMap, IndexPattern, NodeId, NumericPolicy, Op,
        QuantizedBlock, ScalarOp, TensorError, append, projection,
    };

    use super::metal_stage_totals;

    fn embedding_lookup_program(vocab: u32, dim: u32, seq: u32) -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let table = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(vocab), Extent::Static(dim)],
                name: Some(String::from("table")),
            },
        );
        let ids = append(
            &mut program,
            Op::Input {
                dtype: DType::Int32,
                shape: vec![Extent::Static(seq)],
                name: Some(String::from("ids")),
            },
        );
        let gathered_map = IndexMap::Computed {
            indices: ids,
            index_map: projection(2, &[0]),
            base: IndexPattern {
                iter_rank: 2,
                axes: vec![
                    AxisIndex::default(),
                    AxisIndex {
                        terms: vec![AxisTerm::projection(1)].into(),
                        offset: 0,
                        len: None,
                    },
                ],
            },
            gathered_dim: 0,
        };
        let lookup = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: vec![(table, gathered_map)],
                name: None,
            },
        );
        (program, lookup)
    }

    fn lookup_blocks<'data>(
        table: &'data [f32],
        ids: &'data [f32],
    ) -> Vec<(&'static str, QuantizedBlock<'data>)> {
        vec![
            ("table", QuantizedBlock::Float32(table)),
            ("ids", QuantizedBlock::Float32(ids)),
        ]
    }

    #[test]
    fn a_recycled_fault_buffer_still_reports_an_out_of_range_gather_and_is_not_poisoned_by_it() {
        let (vocab, dim, seq) = (4usize, 2usize, 3usize);
        let (program, lookup) = embedding_lookup_program(vocab as u32, dim as u32, seq as u32);
        let table: Vec<f32> = (0..vocab * dim).map(|value| value as f32).collect();
        let in_range = [1.0_f32, 3.0, 0.0];
        let out_of_range = [vocab as f32; 3];
        let plan = omega::plan_named(
            &program,
            &[],
            &lookup_blocks(&table, &in_range),
            &[lookup],
            NumericPolicy::default(),
        )
        .expect("plans the embedding lookup");
        let run = |ids: &[f32]| {
            omega::execute_plan_named_with_placements(&plan, &lookup_blocks(&table, ids), &[], &[])
        };

        let _ = metal_stage_totals();
        run(&in_range).expect("first valid step runs and returns its fault buffer to the pool");
        let cold_totals = metal_stage_totals();
        let warm = run(&in_range).expect("second valid step runs");
        let warm_totals = metal_stage_totals();
        assert!(
            cold_totals.output_buffer_allocations > 0,
            "the cold step must allocate its fault buffer, or the warm count proves nothing"
        );
        assert_eq!(
            warm_totals.output_buffer_allocations, 0,
            "the warm gathered step allocated a fault buffer instead of recycling one"
        );
        let expected_rows: Vec<f32> = [1usize, 3, 0]
            .iter()
            .flat_map(|row| table[row * dim..(row + 1) * dim].to_vec())
            .collect();
        assert_eq!(
            warm.get(lookup).expect("warm lookup output present").0,
            expected_rows.as_slice()
        );

        let fault = run(&out_of_range)
            .expect_err("a recycled fault buffer let an out-of-range gather index through");
        assert!(
            matches!(
                &fault,
                MetalError::Tensor(TensorError::GatherIndexOutOfRange { extent, .. })
                    if *extent == vocab as u64
            ),
            "{fault:?}"
        );

        run(&in_range).expect("a step after a faulted one starts from a clean fault buffer");
    }
}
