//! A warm decode step allocates no device buffers: the router's extra
//! outputs bind plan-owned storage, and every gathered dispatch's fault
//! buffer is recycled once the step's checks have read it.
//!
//! Counted through `OUTPUT_BUFFER_ALLOCATIONS` (`metal_stage_totals`), which
//! `allocate_buffer` and `allocate_fault_buffer` both increment on a fresh
//! `newBuffer`. A routed decode step reported 360 of them (the fused top-k's
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

#[cfg(feature = "moe-topk-fusion")]
mod routed_moe {
    use super::metal_stage_totals;
    use proxima_tensor::spec::{
        Activation, ExpertGatingFunc, MoeFfnSpec, MoeProjectionStrategy, MoeRouter,
        append_moe_ffn, input_leaf, scalar_constant,
    };
    use proxima_tensor::{DType, Extent, NodeId, NumericPolicy, Op, QuantizedBlock};

    const EXPERT_COUNT: u32 = 32;
    const EXPERTS_USED: u32 = 8;
    const EMBEDDING: u32 = 8;
    const FEED_FORWARD: u32 = 8;

    struct RoutedStep {
        program: Vec<Op>,
        output: NodeId,
        x: Vec<f32>,
        logits: Vec<f32>,
        expert_weights: Vec<f32>,
    }

    fn routed_decode_step() -> RoutedStep {
        let mut program = Vec::new();
        let mut leaf = |shape: Vec<u32>, name: &str| {
            input_leaf(
                &mut program,
                DType::Float32,
                shape.into_iter().map(Extent::Static).collect(),
                name,
            )
        };
        let x = leaf(vec![1, EMBEDDING], "x");
        let logits = leaf(vec![1, EXPERT_COUNT], "logits");
        let expert_w_gate = leaf(vec![EXPERT_COUNT, EMBEDDING, FEED_FORWARD], "expert_w_gate");
        let expert_w_up = leaf(vec![EXPERT_COUNT, EMBEDDING, FEED_FORWARD], "expert_w_up");
        let expert_w_down = leaf(vec![EXPERT_COUNT, FEED_FORWARD, EMBEDDING], "expert_w_down");
        let ones = scalar_constant(&mut program, 1.0);
        let moe_spec = MoeFfnSpec {
            router: MoeRouter::Logits(logits),
            expert_w_gate,
            expert_w_up,
            expert_w_down,
            expert_count: EXPERT_COUNT,
            expert_used_count: EXPERTS_USED,
            ones,
            gating: ExpertGatingFunc::Softmax,
            expert_bias: None,
            expert_scale: None,
            activation: Activation::Silu,
            strategy: MoeProjectionStrategy::PerRoute,
        };
        let (output, _site) = append_moe_ffn(&mut program, 0, x, &moe_spec)
            .expect("routed moe ffn lowers at 32 experts, top 8");
        let expert_cells = (EXPERT_COUNT * EMBEDDING * FEED_FORWARD) as usize;
        RoutedStep {
            program,
            output,
            x: (0..EMBEDDING).map(|index| 0.1 + index as f32 * 0.05).collect(),
            logits: (0..EXPERT_COUNT)
                .map(|index| ((index * 7 + 3) % EXPERT_COUNT) as f32 * 0.17)
                .collect(),
            expert_weights: (0..expert_cells)
                .map(|index| ((index % 13) as f32 - 6.0) * 0.03)
                .collect(),
        }
    }

    fn blocks(step: &RoutedStep) -> [QuantizedBlock<'_>; 5] {
        [
            QuantizedBlock::Float32(&step.x),
            QuantizedBlock::Float32(&step.logits),
            QuantizedBlock::Float32(&step.expert_weights),
            QuantizedBlock::Float32(&step.expert_weights),
            QuantizedBlock::Float32(&step.expert_weights),
        ]
    }

    #[test]
    fn a_warm_routed_moe_step_allocates_no_device_buffers() {
        let step = routed_decode_step();
        let blocks = blocks(&step);
        let plan = omega::plan(
            &step.program,
            &[],
            &blocks,
            &[step.output],
            NumericPolicy::default(),
        )
        .expect("plans the full routed moe step");

        let _ = metal_stage_totals();
        let cold = omega::execute_plan_with_placements(&plan, &blocks, &[], &[], &mut Vec::new())
            .expect("cold step builds the arena and runs");
        let cold_totals = metal_stage_totals();
        let warm = omega::execute_plan_with_placements(&plan, &blocks, &[], &[], &mut Vec::new())
            .expect("warm step runs against the plan-owned buffers");
        let warm_totals = metal_stage_totals();

        assert!(
            cold_totals.output_buffer_allocations > 0 && cold_totals.physical_dispatch_calls > 0,
            "the cold step must build device buffers and dispatch: allocations={} dispatches={}",
            cold_totals.output_buffer_allocations,
            cold_totals.physical_dispatch_calls
        );
        assert_eq!(
            warm_totals.output_buffer_allocations, 0,
            "a warm step over {} dispatches allocated {} device buffers ({} bytes)",
            warm_totals.physical_dispatch_calls,
            warm_totals.output_buffer_allocations,
            warm_totals.output_buffer_allocated_bytes
        );
        let cold_values = cold.get(step.output).expect("cold output present").0;
        let warm_values = warm.get(step.output).expect("warm output present").0;
        assert_eq!(
            cold_values, warm_values,
            "recycling buffers across steps must not change the step's result"
        );
    }
}
