use std::collections::BTreeMap;

use omega::{MetalError, PackedOperands, emit, plan};
use proxima_tensor::cpu::QuantizedBlock;
use proxima_tensor::op::{Extent, NodeId, Op};
use proxima_tensor::{
    BoundOp, DType, NumericPolicy, Shapes, TensorError, bind, bind_with_fusion, block_node_ids,
    infer,
};

use super::QatError;
use super::graph::{StepGraph, op_name};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Accepted,
    Rejected(String),
}

pub struct NodeLine {
    pub node: NodeId,
    pub op: String,
    pub side: &'static str,
    pub verdict: Verdict,
}

pub struct PlanOutcome {
    pub rejected: BTreeMap<NodeId, String>,
    pub failure: Option<String>,
}

fn node_ids(count: usize) -> Vec<NodeId> {
    (0..count)
        .filter_map(|index| u32::try_from(index).ok().map(NodeId))
        .collect()
}

fn emit_verdict(bound: &BoundOp) -> Verdict {
    match emit(bound, &PackedOperands::new(), NumericPolicy::default()) {
        Ok(_) => Verdict::Accepted,
        Err(error) => Verdict::Rejected(error.to_string()),
    }
}

fn side_of(graph: &StepGraph, node: NodeId) -> &'static str {
    if (node.0 as usize) < graph.forward_nodes {
        "forward"
    } else {
        "adjoint"
    }
}

pub fn classify_nodes(graph: &StepGraph) -> Result<(Vec<NodeLine>, PlanOutcome), QatError> {
    let shapes = infer(&graph.program, &[])?;
    let every_node = node_ids(graph.program.len());
    let policy = NumericPolicy::default();
    let bound = bind_with_fusion(&graph.program, &shapes, &every_node, false, policy)?;
    let emitted: BTreeMap<NodeId, Verdict> =
        bound.iter().map(|op| (op.node, emit_verdict(op))).collect();
    let outcome = plan_outcome(&graph.program, &graph.outputs)?;
    let lines = graph
        .program
        .iter()
        .zip(&every_node)
        .map(|(op, node)| NodeLine {
            node: *node,
            op: op_name(op),
            side: side_of(graph, *node),
            verdict: node_verdict(op, *node, &emitted, &outcome),
        })
        .collect();
    Ok((lines, outcome))
}

fn node_verdict(
    op: &Op,
    node: NodeId,
    emitted: &BTreeMap<NodeId, Verdict>,
    outcome: &PlanOutcome,
) -> Verdict {
    match (op, outcome.rejected.get(&node), emitted.get(&node)) {
        (_, Some(reason), _) => Verdict::Rejected(reason.clone()),
        (Op::Input { .. }, None, _) => Verdict::Accepted,
        (_, None, Some(verdict)) => verdict.clone(),
        (_, None, None) => Verdict::Rejected("no bound op for this node".to_owned()),
    }
}

fn element_count(shapes: &Shapes, node: NodeId) -> usize {
    usize::try_from(shapes.of(node).iter().product::<u64>()).unwrap_or(0)
}

fn zero_blocks<'zeros>(
    program: &[Op],
    shapes: &Shapes,
    zeros: &'zeros [f32],
) -> Vec<QuantizedBlock<'zeros>> {
    block_node_ids(program)
        .iter()
        .map(|node| {
            QuantizedBlock::Float32(
                zeros
                    .get(..element_count(shapes, *node))
                    .unwrap_or_default(),
            )
        })
        .collect()
}

fn float_leaf(op: &Op) -> Op {
    match op {
        Op::Input { shape, name, .. } => Op::Input {
            dtype: DType::FLOAT32,
            shape: shape.clone(),
            name: name.clone(),
        },
        other => other.clone(),
    }
}

pub fn plan_outcome(program: &[Op], outputs: &[NodeId]) -> Result<PlanOutcome, QatError> {
    let shapes = infer(program, &[])?;
    let widest = block_node_ids(program)
        .iter()
        .map(|node| element_count(&shapes, *node))
        .max()
        .unwrap_or(0);
    let zeros = vec![0.0f32; widest];
    let mut patched = program.to_vec();
    let mut rejected = BTreeMap::new();
    loop {
        let blocks = zero_blocks(&patched, &shapes, &zeros);
        let planned = plan(&patched, &[], &blocks, outputs, NumericPolicy::default());
        match planned {
            Ok(_) => {
                return Ok(PlanOutcome {
                    rejected,
                    failure: None,
                });
            }
            Err(MetalError::Tensor(TensorError::NotLowerable { node, reason })) => {
                rejected.insert(node, reason.to_owned());
                patched[node.0 as usize] = float_leaf(&patched[node.0 as usize]);
            }
            Err(error) => {
                let failure = Some(error.to_string());
                return Ok(PlanOutcome { rejected, failure });
            }
        }
    }
}

pub fn fused_verdicts(graph: &StepGraph) -> Result<Vec<(NodeId, &'static str, Verdict)>, QatError> {
    let shapes = infer(&graph.program, &[])?;
    let bound = bind(
        &graph.program,
        &shapes,
        &graph.outputs,
        NumericPolicy::default(),
    )?;
    Ok(bound
        .iter()
        .map(|op| (op.node, op.kind.name(), emit_verdict(op)))
        .collect())
}

pub fn control_int32_input() -> Result<(usize, usize, usize), QatError> {
    let program = vec![Op::Input {
        dtype: DType::INT32,
        shape: vec![Extent::Static(1)],
        name: Some("control_input".to_owned()),
    }];
    let outcome = plan_outcome(&program, &[])?;
    let rejected = outcome.rejected.len();
    Ok((program.len(), program.len() - rejected, rejected))
}
