use std::collections::{BTreeMap, BTreeSet};

use proxima_onnx::census::{Census, census};
use proxima_onnx::lower::{Lowered, lower_graph_pinned};
use proxima_onnx::messages::{DimensionValue, GraphProto, ModelProto, TypeValue};
use proxima_onnx::pipe::parse_complete;
use proxima_tensor::{Evaluated, NodeId, QuantizedBlock};

use super::error::{DoorError, StageKind, ValidationError};
use super::{DoorBatch, OnnxDoor};
use crate::backend::{Plan, execute_plan_named, plan_named};

/// One state of the door, holding only what is legal in it. `advance`
/// consumes a state and returns the next, so a stale state cannot be reused.
///
/// `Executed` leads back to `Planned` for the next batch under the same pins;
/// every other state has one successor.
pub enum DoorStage<'bytes> {
    Parsed {
        model: ModelProto<'bytes>,
    },
    Validated {
        graph: GraphProto<'bytes>,
        census: Census<'bytes>,
    },
    Lowered {
        lowered: Lowered,
    },
    Planned {
        lowered: Lowered,
        plan: Plan,
    },
    Executed {
        lowered: Lowered,
        plan: Plan,
        evaluated: Evaluated,
    },
}

impl<'bytes> DoorStage<'bytes> {
    /// Parses `bytes` into the first state.
    ///
    /// # Errors
    /// [`DoorError::Parse`] when the wire bytes are not a `ModelProto`.
    pub fn start(bytes: &'bytes [u8]) -> Result<Self, DoorError> {
        let model = parse_complete(bytes).map_err(DoorError::Parse)?;
        Ok(Self::Parsed { model })
    }

    /// The stage that produced this state.
    #[must_use]
    pub const fn kind(&self) -> StageKind {
        match self {
            Self::Parsed { .. } => StageKind::Parse,
            Self::Validated { .. } => StageKind::Validate,
            Self::Lowered { .. } => StageKind::Lower,
            Self::Planned { .. } => StageKind::Plan,
            Self::Executed { .. } => StageKind::Execute,
        }
    }

    /// Runs the stage after this state against `batch`.
    ///
    /// # Errors
    /// The next stage's [`DoorError`] variant.
    pub fn advance(self, door: &OnnxDoor, batch: &DoorBatch) -> Result<Self, DoorError> {
        match self {
            Self::Parsed { model } => validate(model, batch),
            Self::Validated { graph, .. } => lower(&graph, batch),
            Self::Lowered { lowered } => plan(lowered, door, batch),
            Self::Planned { lowered, plan } => execute(lowered, plan, batch),
            Self::Executed { lowered, plan, .. } => Ok(Self::Planned { lowered, plan }),
        }
    }
}

fn validate<'bytes>(
    model: ModelProto<'bytes>,
    batch: &DoorBatch,
) -> Result<DoorStage<'bytes>, DoorError> {
    let graph = model
        .graph
        .ok_or(DoorError::Validate(ValidationError::NoGraph))?;
    let census = census(&graph);
    let unpinned = unpinned_dimensions(&graph, &batch.pins);
    if census.unlowerable.is_empty() && unpinned.is_empty() {
        return Ok(DoorStage::Validated { graph, census });
    }
    let unlowerable = census
        .unlowerable
        .iter()
        .map(|&(op_type, count)| (op_type.to_owned(), count))
        .collect();
    Err(DoorError::Validate(ValidationError::Rejected {
        unlowerable,
        unpinned,
    }))
}

fn unpinned_dimensions(graph: &GraphProto<'_>, pins: &BTreeMap<String, u64>) -> Vec<String> {
    graph
        .input
        .iter()
        .filter_map(|input| match input.r#type.as_ref()?.value.as_ref()? {
            TypeValue::Tensor(tensor) => tensor.shape.as_ref(),
            _ => None,
        })
        .flat_map(|shape| shape.dim.iter())
        .filter_map(|dimension| match &dimension.value {
            Some(DimensionValue::Param(name)) if !pins.contains_key(*name) => Some(*name),
            _ => None,
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(str::to_owned)
        .collect()
}

fn lower<'bytes>(
    graph: &GraphProto<'bytes>,
    batch: &DoorBatch,
) -> Result<DoorStage<'bytes>, DoorError> {
    let pins: BTreeMap<&str, u64> = batch
        .pins
        .iter()
        .map(|(name, extent)| (name.as_str(), *extent))
        .collect();
    let lowered = lower_graph_pinned(graph, &pins).map_err(DoorError::Lower)?;
    Ok(DoorStage::Lowered { lowered })
}

fn plan<'bytes>(
    lowered: Lowered,
    door: &OnnxDoor,
    batch: &DoorBatch,
) -> Result<DoorStage<'bytes>, DoorError> {
    let roots = output_roots(&lowered);
    let named = named_blocks(&lowered, batch);
    let plan = plan_named(
        door.engine,
        door.gpu_driver,
        &lowered.program,
        &[],
        &named,
        &roots,
        door.numeric_policy,
    )
    .map_err(DoorError::Plan)?;
    Ok(DoorStage::Planned { lowered, plan })
}

fn execute<'bytes>(
    lowered: Lowered,
    mut plan: Plan,
    batch: &DoorBatch,
) -> Result<DoorStage<'bytes>, DoorError> {
    let named = named_blocks(&lowered, batch);
    let evaluated = execute_plan_named(&mut plan, &named).map_err(DoorError::Execute)?;
    drop(named);
    Ok(DoorStage::Executed {
        lowered,
        plan,
        evaluated,
    })
}

fn output_roots(lowered: &Lowered) -> Vec<NodeId> {
    lowered
        .graph_outputs
        .iter()
        .map(|(_, node)| *node)
        .collect()
}

fn named_blocks<'a>(
    lowered: &'a Lowered,
    batch: &'a DoorBatch,
) -> Vec<(&'a str, QuantizedBlock<'a>)> {
    let weights = lowered.initializers.iter();
    let inputs = batch.inputs.iter();
    weights
        .chain(inputs)
        .map(|(name, data)| (name.as_str(), QuantizedBlock::Float32(data.as_slice())))
        .collect()
}
