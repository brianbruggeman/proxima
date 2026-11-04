use super::*;

/// The identity of one weight block that shapes a prepared step: its node and the
/// packed layout it carries. Float and integer inputs carry no layout decision.
type WeightKind = (NodeId, Codec, usize, usize);

struct PlanState {
    program_key: u64,
    symbols: Vec<u64>,
    outputs: Vec<NodeId>,
    weight_kinds: Vec<WeightKind>,
    options: StepOptions,
    prepared: StepPrepared,
}

/// A CPU step that is bound once per structural class and executed many times.
///
/// [`evaluate_quantized_named_with_scratch_and_experts`] re-runs `infer`, `bind`,
/// the retirement schedule and the fusion schedules on every call. This type keeps
/// that work between calls: a call with the same program key, outputs and weight
/// kinds reuses it outright when the symbols are unchanged, and patches only the
/// bound ops a changed symbol reaches ([`bind::refresh_bound_ops`]) when they move.
/// A call whose class differs, or whose patch is refused, binds afresh.
pub struct CpuStepPlan {
    numeric_policy: NumericPolicy,
    state: Option<PlanState>,
    binds: u64,
    refits: u64,
    refusals: u64,
}

impl Default for CpuStepPlan {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuStepPlan {
    #[must_use]
    pub fn new() -> Self {
        Self::with_numeric_policy(NumericPolicy::bit_exact())
    }

    /// A plan whose steps bind under `numeric_policy`, so a caller that grants the f32 online
    /// attention gets the arithmetic a gpu binds.
    #[must_use]
    pub fn with_numeric_policy(numeric_policy: NumericPolicy) -> Self {
        Self { numeric_policy, state: None, binds: 0, refits: 0, refusals: 0 }
    }

    /// Full binds performed so far: the first call of a class, and every refused patch.
    #[must_use]
    pub fn binds(&self) -> u64 {
        self.binds
    }

    /// Calls served by patching the retained bound ops at new symbols.
    #[must_use]
    pub fn refits(&self) -> u64 {
        self.refits
    }

    /// Patches [`bind::refresh_bound_ops`] refused, each followed by a full bind.
    #[must_use]
    pub fn refusals(&self) -> u64 {
        self.refusals
    }

    /// Runs one step. `program` is `(ops, key)`; the key names the program so a
    /// plan never serves a different one. `shape` is `(symbols, outputs)`.
    ///
    /// # Errors
    /// Whatever [`evaluate_quantized_named_with_scratch_and_experts`] rejects.
    pub fn execute(
        &mut self,
        program: (&[Op], u64),
        shape: (&[u64], &[NodeId]),
        named: &[(&str, QuantizedBlock<'_>)],
        scratch: &mut BufferPools,
        expert_sources: Option<&BTreeMap<NodeId, ExpertSource<'_>>>,
        exact_activations: bool,
    ) -> Result<Evaluated, TensorError> {
        let (ops, program_key) = program;
        let (symbols, outputs) = shape;
        let blocks = resolve_named_blocks_with_experts(ops, named, expert_sources)?;
        self.settle(ops, program_key, symbols, outputs, &blocks)?;
        let state = self.state.as_ref().ok_or(TensorError::Empty)?;
        let mut inputs = bind_step_blocks(ops, &state.prepared.shapes, &blocks)?;
        materialize_step_inputs(ops, &state.prepared, &mut inputs, &mut scratch.0)?;
        run_prepared_step(
            &state.prepared,
            ops,
            inputs,
            scratch,
            expert_sources,
            (exact_activations, None),
            #[cfg(feature = "instrument")]
            0,
        )
    }

    fn settle(
        &mut self,
        ops: &[Op],
        program_key: u64,
        symbols: &[u64],
        outputs: &[NodeId],
        blocks: &[QuantizedBlock<'_>],
    ) -> Result<(), TensorError> {
        let weight_kinds = weight_kinds_of(ops, blocks)?;
        let options = StepOptions {
            fuse_cached_attention: true,
            allow_fusion: true,
            numeric_policy: self.numeric_policy,
        };
        let same_class = |state: &PlanState| {
            state.program_key == program_key
                && state.outputs == outputs
                && state.weight_kinds == weight_kinds
        };
        match self.state.take() {
            Some(state) if same_class(&state) && state.symbols == symbols => {
                self.state = Some(state);
            }
            Some(state) if same_class(&state) => {
                self.state = Some(self.refit_or_bind(state, ops, symbols, blocks)?);
            }
            _ => {
                let prepared = bind_class(ops, symbols, outputs, blocks, options)?;
                self.binds += 1;
                self.state = Some(PlanState {
                    program_key,
                    symbols: symbols.to_vec(),
                    outputs: outputs.to_vec(),
                    weight_kinds,
                    options,
                    prepared,
                });
            }
        }
        Ok(())
    }

    fn refit_or_bind(
        &mut self,
        state: PlanState,
        ops: &[Op],
        symbols: &[u64],
        blocks: &[QuantizedBlock<'_>],
    ) -> Result<PlanState, TensorError> {
        let quantized_weights = class_weights(ops, blocks)?;
        let next_shapes = shape::infer(ops, symbols)?;
        match refit_unadjusted(&state, ops, &next_shapes) {
            Ok(unadjusted) => {
                self.refits += 1;
                let PlanState {
                    program_key,
                    outputs,
                    weight_kinds,
                    options,
                    prepared,
                    ..
                } = state;
                let request = (
                    next_shapes,
                    prepared.root,
                    prepared.effective_outputs,
                    prepared.planning_outputs,
                );
                Ok(PlanState {
                    program_key,
                    symbols: symbols.to_vec(),
                    outputs,
                    weight_kinds,
                    options,
                    prepared: assemble_step(ops, &quantized_weights, request, unadjusted),
                })
            }
            Err(_) => {
                self.refusals += 1;
                self.binds += 1;
                let prepared = prepare_step(
                    ops,
                    next_shapes,
                    &quantized_weights,
                    &state.outputs,
                    state.options,
                )?;
                Ok(PlanState {
                    symbols: symbols.to_vec(),
                    prepared,
                    ..state
                })
            }
        }
    }
}

fn refit_unadjusted(
    state: &PlanState,
    ops: &[Op],
    next_shapes: &shape::Shapes,
) -> Result<Vec<BoundOp>, bind::RefreshRefusal> {
    let previous = &state.prepared;
    let changed: BTreeSet<NodeId> = (0..ops.len() as u32)
        .map(NodeId)
        .filter(|node| previous.shapes.of(*node) != next_shapes.of(*node))
        .collect();
    let masks: Vec<u64> = previous
        .unadjusted
        .iter()
        .map(|bound| u64::from(bind::touches_changed(bound, ops, &changed)))
        .collect();
    bind::refresh_bound_ops(
        &previous.unadjusted,
        ops,
        &previous.planning_outputs,
        next_shapes,
        &masks,
        1,
        state.options.numeric_policy,
    )
}

fn weight_kinds_of(
    ops: &[Op],
    blocks: &[QuantizedBlock<'_>],
) -> Result<Vec<WeightKind>, TensorError> {
    let nodes = block_node_ids(ops);
    if nodes.len() != blocks.len() {
        return Err(TensorError::InputCountMismatch {
            expected: nodes.len(),
            found: blocks.len(),
        });
    }
    Ok(nodes
        .iter()
        .zip(blocks)
        .filter_map(|(node, block)| match *block {
            QuantizedBlock::Packed { codec, .. } => Some((*node, codec, 0, 0)),
            QuantizedBlock::PackedStrided {
                codec,
                expert_bytes,
                stride_bytes,
                ..
            } => Some((*node, codec, expert_bytes, stride_bytes)),
            QuantizedBlock::Float32(_) | QuantizedBlock::Int32(_) => None,
        })
        .collect())
}

fn class_weights<'call>(
    ops: &[Op],
    blocks: &[QuantizedBlock<'call>],
) -> Result<BTreeMap<NodeId, QuantizedBlock<'call>>, TensorError> {
    let nodes = block_node_ids(ops);
    Ok(nodes
        .iter()
        .zip(blocks.iter().copied())
        .filter(|(_, block)| {
            matches!(
                block,
                QuantizedBlock::Packed { .. } | QuantizedBlock::PackedStrided { .. }
            )
        })
        .map(|(node, block)| (*node, block))
        .collect())
}

fn bind_class(
    ops: &[Op],
    symbols: &[u64],
    outputs: &[NodeId],
    blocks: &[QuantizedBlock<'_>],
    options: StepOptions,
) -> Result<StepPrepared, TensorError> {
    let shapes = shape::infer(ops, symbols)?;
    let quantized_weights = class_weights(ops, blocks)?;
    let weight_nodes: BTreeSet<NodeId> = quantized_weights.keys().copied().collect();
    reject_non_float32(ops, &weight_nodes)?;
    prepare_step(ops, shapes, &quantized_weights, outputs, options)
}

/// Whether `program` bound at `symbols` reads every K/V cache input `is_kv_input`
/// accepts only through a fused cached attention, which bounds its own live rows
/// at run time: the condition under which a caller may bind the cached range
/// padded to a capacity bucket so the plan survives every step of a call.
/// Binds without executing; the answer is the one a [`CpuStepPlan`] would give.
///
/// # Errors
/// Shape inference or binding failures at `symbols`.
#[cfg(feature = "cached-attention-streaming")]
pub fn probe_kv_padding(
    program: &[Op],
    symbols: &[u64],
    outputs: &[NodeId],
    is_kv_input: impl Fn(&str) -> bool,
) -> Result<bool, TensorError> {
    let shapes = shape::infer(program, symbols)?;
    let (_, effective_outputs) = effective_step_outputs(program, outputs)?;
    Ok(bind::kv_read_only_by_fused_attention_in_program(
        program,
        &shapes,
        &effective_outputs,
        NumericPolicy::bit_exact(),
        is_kv_input,
    ))
}

/// Without cached-attention streaming nothing is fused, so no cache range can bind padded.
#[cfg(not(feature = "cached-attention-streaming"))]
#[expect(clippy::missing_errors_doc, reason = "the streaming-enabled build documents the errors")]
pub fn probe_kv_padding(
    _program: &[Op],
    _symbols: &[u64],
    _outputs: &[NodeId],
    _is_kv_input: impl Fn(&str) -> bool,
) -> Result<bool, TensorError> {
    Ok(false)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::map::{self, IndexMap};
    use crate::op::{Extent, Keep, ReduceInit, append};

    const INNER: usize = 8;
    const COLUMNS: usize = 4;

    fn matmul() -> (Vec<Op>, NodeId, NodeId) {
        let mut program = Vec::new();
        let lhs = append(
            &mut program,
            Op::Input {
                dtype: DType::FLOAT32,
                shape: vec![Extent::Symbolic(0), Extent::Static(INNER as u32)],
                name: Some("lhs".into()),
            },
        );
        let rhs = append(
            &mut program,
            Op::Input {
                dtype: DType::FLOAT32,
                shape: vec![Extent::Static(INNER as u32), Extent::Static(COLUMNS as u32)],
                name: Some("rhs".into()),
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
        let sum = append(
            &mut program,
            Op::Reduce(crate::op::Reduce {
                dtype: DType::FLOAT32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        (program, product, sum)
    }

    fn activations(rows: usize) -> Vec<f32> {
        (0..rows * INNER)
            .map(|index| (index as f32 * 0.37).sin())
            .collect()
    }

    fn weights() -> Vec<f32> {
        (0..INNER * COLUMNS)
            .map(|index| (index as f32 * 0.11).cos())
            .collect()
    }

    fn one_shot(program: &[Op], rows: usize, outputs: &[NodeId]) -> Evaluated {
        let (lhs, rhs) = (activations(rows), weights());
        let named = [
            ("lhs", QuantizedBlock::Float32(&lhs)),
            ("rhs", QuantizedBlock::Float32(&rhs)),
        ];
        evaluate_quantized_named_with_scratch_and_experts(
            program,
            &[rows as u64],
            &named,
            outputs,
            &mut BufferPools::default(),
            &mut None,
            None,
        )
        .expect("the one-shot evaluation runs")
    }

    fn planned(
        plan: &mut CpuStepPlan,
        key: u64,
        program: &[Op],
        rows: usize,
        outputs: &[NodeId],
    ) -> Result<Evaluated, TensorError> {
        let (lhs, rhs) = (activations(rows), weights());
        let named = [
            ("lhs", QuantizedBlock::Float32(&lhs)),
            ("rhs", QuantizedBlock::Float32(&rhs)),
        ];
        plan.execute(
            (program, key),
            (&[rows as u64], outputs),
            &named,
            &mut BufferPools::default(),
            None,
            false,
        )
    }

    fn assert_bits_equal(left: &Evaluated, right: &Evaluated, node: NodeId) {
        let (left_values, left_shape) = left.get(node).expect("planned output present");
        let (right_values, right_shape) = right.get(node).expect("one-shot output present");
        assert_eq!(left_shape, right_shape, "output shape of {node:?}");
        let left_bits: Vec<u32> = left_values.iter().map(|value| value.to_bits()).collect();
        let right_bits: Vec<u32> = right_values.iter().map(|value| value.to_bits()).collect();
        assert_eq!(left_bits, right_bits, "output bits of {node:?}");
    }

    #[test]
    fn repeating_the_symbols_reuses_the_bound_step_without_a_second_bind() {
        let (program, _, sum) = matmul();
        let mut plan = CpuStepPlan::new();

        let first = planned(&mut plan, 7, &program, 3, &[sum]).expect("first call");
        let second = planned(&mut plan, 7, &program, 3, &[sum]).expect("second call");

        assert_eq!((plan.binds(), plan.refits()), (1, 0));
        assert_bits_equal(&first, &one_shot(&program, 3, &[sum]), sum);
        assert_bits_equal(&second, &one_shot(&program, 3, &[sum]), sum);
    }

    #[test]
    fn moving_a_symbol_patches_the_bound_step_and_matches_a_fresh_bind_bit_for_bit() {
        let (program, _, sum) = matmul();
        let mut plan = CpuStepPlan::new();

        planned(&mut plan, 7, &program, 3, &[sum]).expect("bind at three rows");
        let patched = planned(&mut plan, 7, &program, 5, &[sum]).expect("patch to five rows");

        assert_eq!((plan.binds(), plan.refits(), plan.refusals()), (1, 1, 0));
        assert_bits_equal(&patched, &one_shot(&program, 5, &[sum]), sum);
    }

    #[test]
    fn a_different_output_set_is_a_different_class_and_binds_again() {
        let (program, product, sum) = matmul();
        let mut plan = CpuStepPlan::new();

        planned(&mut plan, 7, &program, 3, &[sum]).expect("sum class");
        let other = planned(&mut plan, 7, &program, 3, &[product]).expect("product class");

        assert_eq!((plan.binds(), plan.refits()), (2, 0));
        assert_bits_equal(&other, &one_shot(&program, 3, &[product]), product);
    }

    #[test]
    fn a_different_program_key_binds_again_even_at_identical_symbols() {
        let (program, _, sum) = matmul();
        let mut plan = CpuStepPlan::new();

        planned(&mut plan, 7, &program, 3, &[sum]).expect("first program");
        planned(&mut plan, 8, &program, 3, &[sum]).expect("second program key");

        assert_eq!(plan.binds(), 2);
    }

    #[test]
    fn an_unbound_input_name_is_refused_and_leaves_the_plan_unbound() {
        let (program, _, sum) = matmul();
        let mut plan = CpuStepPlan::new();
        let lhs = activations(3);
        let named = [("lhs", QuantizedBlock::Float32(&lhs))];

        let outcome = plan.execute(
            (&program, 7),
            (&[3], &[sum]),
            &named,
            &mut BufferPools::default(),
            None,
            false,
        );

        assert!(
            matches!(outcome, Err(TensorError::UnboundInputName(_))),
            "{outcome:?}"
        );
        assert_eq!(plan.binds(), 0);
    }
}
