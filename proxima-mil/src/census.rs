//! The MIL census: every node of a lowered omega program is classified
//! exactly once as mapped to MIL operators or refused with a typed reason.
//!
//! The tables here are the ones in `proxima-tensor/specs/coreml-ane-lane/SPEC.md`
//! ("MIL mapping"): [`scalar_form`] for the 17 [`ScalarOp`]s, [`reduce_operator`]
//! and [`scan_operator`] for [`Reduce`] bodies, and the private `view` module for the operand
//! `proxima_tensor::IndexMap` forms. They read only the algebra (`proxima-tensor`), never a
//! model format, so the census serves every door. Teaching pointer: the census
//! is the Validate stage's MIL check; `MilWriter` (Emit) consumes the same
//! classification, so what the census maps the writer can emit.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt;

use proxima_tensor::{Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp, Shapes, TensorError, infer};

mod view;

pub(crate) use view::is_contraction;
use view::operand_views;

/// How one [`ScalarOp`] reaches MIL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MilForm {
    /// One MIL operator of this name.
    Operator(&'static str),
    /// A fixed composition of MIL operators, for a body MIL has no operator for.
    Composition(&'static [&'static str]),
}

impl MilForm {
    /// The MIL operator names the form emits, in order.
    #[must_use]
    pub fn operators(&self) -> &[&'static str] {
        match self {
            Self::Operator(name) => core::slice::from_ref(name),
            Self::Composition(names) => names,
        }
    }
}

/// Every [`ScalarOp`], in declaration order; [`scalar_form`] is exhaustive over them.
pub const SCALAR_OPS: [ScalarOp; 17] = [
    ScalarOp::Identity,
    ScalarOp::Add,
    ScalarOp::Subtract,
    ScalarOp::Multiply,
    ScalarOp::Divide,
    ScalarOp::Maximum,
    ScalarOp::Minimum,
    ScalarOp::Negate,
    ScalarOp::Reciprocal,
    ScalarOp::Exponential,
    ScalarOp::Logarithm,
    ScalarOp::SquareRoot,
    ScalarOp::Tanh,
    ScalarOp::Erf,
    ScalarOp::Greater,
    ScalarOp::Equal,
    ScalarOp::Select,
];

/// The MIL form of a scalar body. `Negate` is the one composition: the compiler
/// answers `Unknown operator 'neg'` (`LANEFIX/mil_ops_probe.txt`), so it is `mul`
/// by a constant -1.
#[must_use]
pub const fn scalar_form(body: ScalarOp) -> MilForm {
    match body {
        ScalarOp::Identity => MilForm::Operator("identity"),
        ScalarOp::Add => MilForm::Operator("add"),
        ScalarOp::Subtract => MilForm::Operator("sub"),
        ScalarOp::Multiply => MilForm::Operator("mul"),
        ScalarOp::Divide => MilForm::Operator("real_div"),
        ScalarOp::Maximum => MilForm::Operator("maximum"),
        ScalarOp::Minimum => MilForm::Operator("minimum"),
        ScalarOp::Negate => MilForm::Composition(&["mul"]),
        ScalarOp::Reciprocal => MilForm::Operator("inverse"),
        ScalarOp::Exponential => MilForm::Operator("exp"),
        ScalarOp::Logarithm => MilForm::Operator("log"),
        ScalarOp::SquareRoot => MilForm::Operator("sqrt"),
        ScalarOp::Tanh => MilForm::Operator("tanh"),
        ScalarOp::Erf => MilForm::Operator("erf"),
        ScalarOp::Greater => MilForm::Operator("greater"),
        ScalarOp::Equal => MilForm::Operator("equal"),
        ScalarOp::Select => MilForm::Operator("select"),
    }
}

/// The MIL reduce operator for a body and seed, or why there is none.
///
/// # Errors
/// [`Refusal::ReduceSeed`] when the pair is not one the SPEC table names.
pub const fn reduce_operator(body: ScalarOp, init: ReduceInit) -> Result<&'static str, Refusal> {
    match (body, init) {
        (ScalarOp::Add, ReduceInit::Zero) => Ok("reduce_sum"),
        (ScalarOp::Multiply, ReduceInit::One) => Ok("reduce_prod"),
        (ScalarOp::Maximum, ReduceInit::NegativeInfinity) => Ok("reduce_max"),
        (ScalarOp::Minimum, ReduceInit::PositiveInfinity) => Ok("reduce_min"),
        (ScalarOp::Maximum, ReduceInit::FirstElement) => Ok("reduce_argmax"),
        _ => Err(Refusal::ReduceSeed { body, init }),
    }
}

/// The MIL scan operator for a [`Keep::Scan`] body, or why there is none.
///
/// # Errors
/// [`Refusal::ScanBody`] naming the absent operator, or
/// [`Refusal::NonAssociativeScan`] for a body no scan operator could express.
pub const fn scan_operator(body: ScalarOp, init: ReduceInit) -> Result<&'static str, Refusal> {
    match (body, init) {
        (ScalarOp::Add, ReduceInit::Zero) => Ok("cumsum"),
        (ScalarOp::Add, _) => Err(Refusal::ReduceSeed { body, init }),
        (ScalarOp::Maximum, _) => Err(Refusal::ScanBody {
            body,
            missing: "cummax",
        }),
        (ScalarOp::Minimum, _) => Err(Refusal::ScanBody {
            body,
            missing: "cummin",
        }),
        (ScalarOp::Multiply, _) => Err(Refusal::ScanBody {
            body,
            missing: "cumprod",
        }),
        _ => Err(Refusal::NonAssociativeScan { body }),
    }
}

/// Why a node has no MIL form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum Refusal {
    /// MIL has no scan operator for this associative body.
    #[error("scan body {body:?} has no mil operator, `{missing}` is unknown to the compiler")]
    ScanBody {
        body: ScalarOp,
        missing: &'static str,
    },
    /// The body is not associative, so no prefix scan operator can express it.
    #[error("scan body {body:?} is not associative and has no mil operator")]
    NonAssociativeScan { body: ScalarOp },
    /// The body and seed pair is not one the mapping table names.
    #[error("reduce body {body:?} seeded {init:?} has no mil reduce operator")]
    ReduceSeed { body: ScalarOp, init: ReduceInit },
    /// A data-dependent output is a scatter; `scatter_along_axis` exists in MIL but is not placeable on the Neural Engine.
    #[error(
        "data-dependent output is a scatter, mil `scatter_along_axis` is not placeable on the neural engine"
    )]
    Scatter,
    /// The operand reads a window no view operator expresses (overlapping, dilated or strided merges).
    #[error("operand {operand} reads a window no mil view operator expresses")]
    WindowedRead { operand: usize },
    /// The operand reads one iteration axis through two operand axes.
    #[error("operand {operand} reads one iteration axis through two operand axes")]
    DiagonalRead { operand: usize },
}

/// A node with no MIL form, naming the node and the missing form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("node {node} has no mil form: {reason}")]
pub struct NoMilForm {
    /// The refused node's position in the program.
    pub node: NodeId,
    /// The missing form.
    pub reason: Refusal,
}

/// The result of classifying a program: `mapped + refused.len() == nodes`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MilCensus {
    /// Nodes in the program, e.g. 1551 for bge-small at batch 1 and its shortest pinned sequence.
    pub nodes: usize,
    /// Nodes with a MIL form.
    pub mapped: usize,
    /// Nodes without one, each naming its reason.
    pub refused: Vec<NoMilForm>,
    /// MIL operators the mapped nodes imply, counted by name; function inputs and
    /// constants are immediates and count none.
    pub operators: BTreeMap<&'static str, usize>,
}

impl MilCensus {
    fn record(&mut self, node: NodeId, outcome: Result<Vec<&'static str>, Refusal>) {
        self.nodes += 1;
        match outcome {
            Ok(operators) => {
                self.mapped += 1;
                operators
                    .into_iter()
                    .for_each(|name| *self.operators.entry(name).or_default() += 1);
            }
            Err(reason) => self.refused.push(NoMilForm { node, reason }),
        }
    }
}

impl fmt::Display for MilCensus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "mil_census nodes={} mapped={} refused={}",
            self.nodes,
            self.mapped,
            self.refused.len()
        )
    }
}

/// Classifies every node of `program`, resolving symbolic extents with `symbols`.
///
/// # Errors
/// The [`TensorError`] shape inference raises for a malformed program.
pub fn census(program: &[Op], symbols: &[u64]) -> Result<MilCensus, TensorError> {
    let shapes = infer(program, symbols)?;
    let mut tally = MilCensus::default();
    for (index, op) in (0_u32..).zip(program) {
        tally.record(NodeId(index), classify(program, &shapes, NodeId(index), op));
    }
    Ok(tally)
}

fn classify(
    program: &[Op],
    shapes: &Shapes,
    node: NodeId,
    op: &Op,
) -> Result<Vec<&'static str>, Refusal> {
    match op {
        Op::Input { .. } | Op::Constant { .. } => Ok(Vec::new()),
        Op::Iota { .. } => Ok(alloc::vec!["range_1d"]),
        Op::Elementwise { body, operands, .. } => {
            let extents = shapes.of(node);
            let mut operators = scalar_form(*body).operators().to_vec();
            for (position, (_, map)) in operands.iter().enumerate() {
                operators.extend(operand_views(map, extents, position)?);
            }
            Ok(operators)
        }
        Op::Reduce(reduce) => reduce_operators(program, shapes, reduce),
    }
}

fn reduce_operators(
    program: &[Op],
    shapes: &Shapes,
    reduce: &Reduce,
) -> Result<Vec<&'static str>, Refusal> {
    if reduce.out_map.is_data_dependent() {
        return Err(Refusal::Scatter);
    }
    if reduce.keep == Keep::Scan {
        return Ok(alloc::vec![scan_operator(reduce.body, reduce.init)?]);
    }
    match contraction_views(program, shapes, reduce) {
        Some(views) => Ok(views?),
        None => plain_reduce(shapes, reduce),
    }
}

fn plain_reduce(shapes: &Shapes, reduce: &Reduce) -> Result<Vec<&'static str>, Refusal> {
    let extents = view::reduce_extents(&reduce.in_map, shapes, reduce.operand)?;
    let mut operators = alloc::vec![reduce_operator(reduce.body, reduce.init)?];
    operators.extend(operand_views(&reduce.in_map, &extents, 0)?);
    Ok(operators)
}

fn contraction_views(
    program: &[Op],
    shapes: &Shapes,
    reduce: &Reduce,
) -> Option<Result<Vec<&'static str>, Refusal>> {
    let Op::Elementwise {
        body: ScalarOp::Multiply,
        operands,
        ..
    } = program.get(reduce.operand.0 as usize)?
    else {
        return None;
    };
    let [(_, left_map), (_, right_map)] = operands.as_slice() else {
        return None;
    };
    (reduce.body == ScalarOp::Add
        && reduce.init == ReduceInit::Zero
        && view::is_contraction(reduce, left_map, right_map))
    .then(|| {
        let extents = shapes.of(reduce.operand);
        let mut operators = alloc::vec!["matmul"];
        operators.extend(operand_views(left_map, extents, 0)?);
        operators.extend(operand_views(right_map, extents, 1)?);
        Ok(operators)
    })
}
