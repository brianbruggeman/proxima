use std::path::Path;

use proxima_autograd::adjoint::{Differentiated, differentiate_wanted};
use proxima_autograd::error::AutogradError;
use proxima_autograd::loss::softmax_cross_entropy;
use proxima_model_interop::bind_symbols;
use proxima_tensor::op::{Extent, NodeId, Op, Reduce, append};
use proxima_tensor::spec::{ForwardProgram, ModelDescriptor, build_forward};
use proxima_tensor::{AxisIndex, DType, IndexMap, IndexPattern, Shapes, infer};

use super::QatError;

pub struct Refusal {
    pub node: NodeId,
    pub op: String,
    pub operand_map: String,
    pub variant: String,
    pub reason: String,
}

pub struct StepGraph {
    pub program: Vec<Op>,
    pub forward_nodes: usize,
    pub refusals: Vec<Refusal>,
    pub outputs: Vec<NodeId>,
}

pub fn load_descriptor(path: &Path, layers: u32) -> Result<ModelDescriptor, QatError> {
    let mut descriptor: ModelDescriptor =
        conflaguration::from_file(path).map_err(|error| QatError::Config(error.to_string()))?;
    descriptor.layers.truncate(layers as usize);
    descriptor.block_count = layers;
    descriptor.leading_dense_block_count = layers.min(descriptor.leading_dense_block_count);
    descriptor.last_row_only = false;
    Ok(descriptor)
}

pub fn step_graph(descriptor: &ModelDescriptor, sequence: u32) -> Result<StepGraph, QatError> {
    let symbols = bind_symbols(sequence as usize, sequence as usize, &[], false)?;
    let ForwardProgram {
        program, logits, ..
    } = build_forward(descriptor)?;
    let mut forward = fix_symbols(&program, &symbols)?;
    check_logits_shape(&forward, logits, sequence, descriptor.vocab)?;
    let loss = append_loss(&mut forward, logits, sequence, descriptor.vocab);
    let wanted = weight_nodes(&forward);
    let (differentiated, refusals) = differentiate_cutting(forward.clone(), loss, &wanted)?;
    let forward_nodes = forward.len();
    let outputs = gradient_outputs(&differentiated, &wanted, loss);
    forward.extend(differentiated.program.into_iter().skip(forward_nodes));
    Ok(StepGraph {
        program: forward,
        forward_nodes,
        refusals,
        outputs,
    })
}

fn weight_nodes(program: &[Op]) -> Vec<NodeId> {
    program
        .iter()
        .enumerate()
        .filter(
            |(_, op)| matches!(op, Op::Input { name: Some(name), .. } if name.ends_with(".weight")),
        )
        .filter_map(|(position, _)| u32::try_from(position).ok().map(NodeId))
        .collect()
}

fn gradient_outputs(
    differentiated: &Differentiated,
    wanted: &[NodeId],
    loss: NodeId,
) -> Vec<NodeId> {
    let gradients = wanted
        .iter()
        .filter_map(|node| differentiated.gradient_of(*node));
    std::iter::once(loss).chain(gradients).collect()
}

fn differentiate_cutting(
    mut program: Vec<Op>,
    loss: NodeId,
    wanted: &[NodeId],
) -> Result<(Differentiated, Vec<Refusal>), QatError> {
    let mut refusals = Vec::new();
    loop {
        let error = match differentiate_wanted(&program, loss, wanted) {
            Ok(differentiated) => return Ok((differentiated, refusals)),
            Err(error) => error,
        };
        let node = refused_node(&error).ok_or(QatError::Autograd(error.clone()))?;
        refusals.push(refusal(&program, node, &error));
        program[node.0 as usize] = cut_leaf(&program, node)?;
    }
}

fn refused_node(error: &AutogradError) -> Option<NodeId> {
    match error {
        AutogradError::ScanAdjointUnsupported { node }
        | AutogradError::UnsupportedReduceBody { node, .. }
        | AutogradError::UnsupportedElementwiseBody { node, .. }
        | AutogradError::NonProjectionOperandMap { node, .. }
        | AutogradError::NonProjectionIndexMap { node, .. }
        | AutogradError::ReduceOverGatherUnsupported { node, .. }
        | AutogradError::ScatterOutputUnsupported { node, .. } => Some(*node),
        _ => None,
    }
}

fn refusal(program: &[Op], node: NodeId, error: &AutogradError) -> Refusal {
    let debug = format!("{error:?}");
    let variant = debug.split([' ', '{', '(']).next().unwrap_or_default();
    Refusal {
        node,
        op: op_name(&program[node.0 as usize]),
        operand_map: operand_map_text(&program[node.0 as usize], error),
        variant: variant.to_owned(),
        reason: error.to_string(),
    }
}

fn cut_leaf(program: &[Op], node: NodeId) -> Result<Op, QatError> {
    let shapes: Shapes = infer(program, &[])?;
    let shape = shapes
        .of(node)
        .iter()
        .map(|dim| u32::try_from(*dim).map(Extent::Static))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| QatError::ExtentOverflow { value: u64::MAX })?;
    Ok(Op::Input {
        dtype: program[node.0 as usize].dtype(),
        shape,
        name: Some(format!("cut.{}", node.0)),
    })
}

pub fn op_name(op: &Op) -> String {
    match op {
        Op::Input { .. } => "Input".to_owned(),
        Op::Constant { .. } => "Constant".to_owned(),
        Op::Iota { .. } => "Iota".to_owned(),
        Op::Elementwise { body, .. } => format!("Elementwise({body:?})"),
        Op::Reduce(reduce) => format!("Reduce({:?},{:?})", reduce.body, reduce.keep),
    }
}

fn check_logits_shape(
    program: &[Op],
    logits: NodeId,
    rows: u32,
    vocab: u32,
) -> Result<(), QatError> {
    let found = infer(program, &[])?.of(logits).to_vec();
    match found.as_slice() {
        [row_count, width] if *row_count == u64::from(rows) && *width == u64::from(vocab) => Ok(()),
        _ => Err(QatError::LogitsShape {
            found,
            rows: u64::from(rows),
            vocab,
        }),
    }
}

fn append_loss(program: &mut Vec<Op>, logits: NodeId, rows: u32, vocab: u32) -> NodeId {
    let one_hot = append(
        program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(rows), Extent::Static(vocab)],
            name: Some("targets_one_hot".into()),
        },
    );
    softmax_cross_entropy(program, DType::Float32, logits, one_hot, 2, 1)
}

fn fix_extent(extent: &Extent, symbols: &[u64]) -> Result<Extent, QatError> {
    let Extent::Symbolic(symbol) = extent else {
        return Ok(*extent);
    };
    let value = *symbols
        .get(usize::from(*symbol))
        .ok_or(QatError::UnboundSymbol {
            symbol: *symbol,
            bound: symbols.len(),
        })?;
    let fixed = u32::try_from(value).map_err(|_| QatError::ExtentOverflow { value })?;
    Ok(Extent::Static(fixed))
}

fn fix_extents(extents: &[Extent], symbols: &[u64]) -> Result<Vec<Extent>, QatError> {
    extents
        .iter()
        .map(|extent| fix_extent(extent, symbols))
        .collect()
}

fn fix_axis(axis: &AxisIndex, symbols: &[u64]) -> Result<AxisIndex, QatError> {
    let len = axis
        .len
        .as_ref()
        .map(|extent| fix_extent(extent, symbols))
        .transpose()?;
    Ok(AxisIndex {
        len,
        ..axis.clone()
    })
}

fn fix_pattern(pattern: &IndexPattern, symbols: &[u64]) -> Result<IndexPattern, QatError> {
    let axes = pattern
        .axes
        .iter()
        .map(|axis| fix_axis(axis, symbols))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(IndexPattern {
        iter_rank: pattern.iter_rank,
        axes,
    })
}

fn fix_map(map: &IndexMap, symbols: &[u64]) -> Result<IndexMap, QatError> {
    match map {
        IndexMap::Affine(pattern) => Ok(IndexMap::Affine(fix_pattern(pattern, symbols)?)),
        IndexMap::Computed {
            indices,
            index_map,
            base,
            gathered_dim,
        } => Ok(IndexMap::Computed {
            indices: *indices,
            index_map: fix_pattern(index_map, symbols)?,
            base: fix_pattern(base, symbols)?,
            gathered_dim: *gathered_dim,
        }),
    }
}

fn fix_operands(
    operands: &[(NodeId, IndexMap)],
    symbols: &[u64],
) -> Result<Vec<(NodeId, IndexMap)>, QatError> {
    operands
        .iter()
        .map(|(node, map)| Ok((*node, fix_map(map, symbols)?)))
        .collect()
}

fn fix_op(op: &Op, symbols: &[u64]) -> Result<Op, QatError> {
    Ok(match op {
        Op::Input { dtype, shape, name } => Op::Input {
            dtype: *dtype,
            shape: fix_extents(shape, symbols)?,
            name: name.clone(),
        },
        Op::Constant {
            dtype,
            shape,
            value,
        } => Op::Constant {
            dtype: *dtype,
            shape: fix_extents(shape, symbols)?,
            value: *value,
        },
        Op::Iota { dtype, extent } => Op::Iota {
            dtype: *dtype,
            extent: fix_extent(extent, symbols)?,
        },
        Op::Elementwise {
            dtype,
            body,
            operands,
            name,
        } => Op::Elementwise {
            dtype: *dtype,
            body: *body,
            operands: fix_operands(operands, symbols)?,
            name: name.clone(),
        },
        Op::Reduce(reduce) => Op::Reduce(Reduce {
            in_map: fix_map(&reduce.in_map, symbols)?,
            out_map: fix_map(&reduce.out_map, symbols)?,
            ..reduce.clone()
        }),
    })
}

pub fn fix_symbols(program: &[Op], symbols: &[u64]) -> Result<Vec<Op>, QatError> {
    program.iter().map(|op| fix_op(op, symbols)).collect()
}

fn refused_operand(error: &AutogradError) -> Option<NodeId> {
    match error {
        AutogradError::NonProjectionOperandMap { operand, .. }
        | AutogradError::NonProjectionIndexMap { operand, .. }
        | AutogradError::ReduceOverGatherUnsupported { operand, .. } => Some(*operand),
        _ => None,
    }
}

fn axis_text(axis: &AxisIndex) -> String {
    let terms: Vec<String> = axis
        .terms
        .iter()
        .map(|term| format!("{}*i{}", term.coeff, term.axis))
        .collect();
    format!("{}+{}", terms.join("+"), axis.offset)
}

fn operand_map_text(op: &Op, error: &AutogradError) -> String {
    let Op::Elementwise { operands, .. } = op else {
        return String::new();
    };
    let Some(target) = refused_operand(error) else {
        return String::new();
    };
    operands
        .iter()
        .filter(|(node, _)| *node == target)
        .map(|(_, map)| match map {
            IndexMap::Affine(pattern) => {
                let axes: Vec<String> = pattern.axes.iter().map(axis_text).collect();
                format!("[{}]", axes.join(", "))
            }
            IndexMap::Computed { .. } => "computed".to_owned(),
        })
        .collect::<Vec<_>>()
        .join(" | ")
}
