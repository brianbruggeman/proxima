use proxima_tensor::{Evaluated, Lowered, NodeId, QuantizedBlock};

use super::error::{DoorError, StageKind};
use super::reader::Reader;
use super::writer::{NoSink, Writer};
use super::{Door, DoorBatch};
use crate::backend::{Plan, execute_plan_named, plan_named};

/// One state of the door, holding only what is legal in it. `advance`
/// consumes a state and returns the next, so a stale state cannot be reused.
///
/// Reading runs `Parsed`, `Validated`, `Lowered`; writing runs `Lowered`,
/// `Emitted`, `Packaged`, `Verified`; running runs `Lowered`, `Planned`,
/// `Executed`. `Executed` leads back to `Planned` for the next batch under the
/// same pins and `Verified` leads back to `Lowered` for the next package; every
/// other state has one successor. A door whose writer is [`NoSink`] never
/// enters the writing states.
pub enum DoorStage<'bytes, R: Reader, W: Writer = NoSink> {
    Parsed {
        parsed: R::Parsed<'bytes>,
    },
    Validated {
        validated: R::Validated<'bytes>,
    },
    Lowered {
        lowered: Lowered,
    },
    Emitted {
        lowered: Lowered,
        emitted: W::Emitted,
    },
    Packaged {
        lowered: Lowered,
        package: W::Package,
    },
    Verified {
        lowered: Lowered,
        package: W::Package,
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

impl<'bytes, R: Reader, W: Writer> DoorStage<'bytes, R, W> {
    /// Parses `bytes` with `reader` into the first reading state.
    ///
    /// # Errors
    /// [`DoorError::Parse`] when the bytes are not the reader's format.
    pub fn begin(reader: &R, bytes: &'bytes [u8]) -> Result<Self, DoorError<R, W>> {
        let parsed = reader.parse(bytes).map_err(DoorError::Parse)?;
        Ok(Self::Parsed { parsed })
    }

    /// Emits an already lowered program with `writer`, the first writing
    /// state; a program that did not come from a reader enters here.
    ///
    /// # Errors
    /// [`DoorError::Emit`] when the writer cannot express the program.
    pub fn emit(writer: &W, lowered: Lowered) -> Result<Self, DoorError<R, W>> {
        let emitted = writer.emit(&lowered).map_err(DoorError::Emit)?;
        Ok(Self::Emitted { lowered, emitted })
    }

    /// The stage that produced this state.
    #[must_use]
    pub const fn kind(&self) -> StageKind {
        match self {
            Self::Parsed { .. } => StageKind::Parse,
            Self::Validated { .. } => StageKind::Validate,
            Self::Lowered { .. } => StageKind::Lower,
            Self::Emitted { .. } => StageKind::Emit,
            Self::Packaged { .. } => StageKind::Package,
            Self::Verified { .. } => StageKind::Verify,
            Self::Planned { .. } => StageKind::Plan,
            Self::Executed { .. } => StageKind::Execute,
        }
    }

    /// Runs the stage after this state against `batch`.
    ///
    /// # Errors
    /// The next stage's [`DoorError`] variant.
    pub fn advance(self, door: &Door<R, W>, batch: &DoorBatch) -> Result<Self, DoorError<R, W>> {
        match self {
            Self::Parsed { parsed } => validate(parsed, door, batch),
            Self::Validated { validated } => lower(&validated, door, batch),
            Self::Lowered { lowered } => plan(lowered, door, batch),
            Self::Emitted { lowered, emitted } => package(lowered, emitted, door),
            Self::Packaged { lowered, package } => verify(lowered, package, door),
            Self::Verified { lowered, .. } => Ok(Self::Lowered { lowered }),
            Self::Planned { lowered, plan } => execute(lowered, plan, batch),
            Self::Executed { lowered, plan, .. } => Ok(Self::Planned { lowered, plan }),
        }
    }
}

fn validate<'bytes, R: Reader, W: Writer>(
    parsed: R::Parsed<'bytes>,
    door: &Door<R, W>,
    batch: &DoorBatch,
) -> Result<DoorStage<'bytes, R, W>, DoorError<R, W>> {
    let validated = door
        .reader
        .validate(parsed, &batch.pins)
        .map_err(DoorError::Validate)?;
    Ok(DoorStage::Validated { validated })
}

fn lower<'bytes, R: Reader, W: Writer>(
    validated: &R::Validated<'bytes>,
    door: &Door<R, W>,
    batch: &DoorBatch,
) -> Result<DoorStage<'bytes, R, W>, DoorError<R, W>> {
    let lowered = door.reader.lower(validated, &batch.pins);
    door.cache.borrow_mut().record_lowering();
    Ok(DoorStage::Lowered {
        lowered: lowered.map_err(DoorError::Lower)?,
    })
}

fn package<'bytes, R: Reader, W: Writer>(
    lowered: Lowered,
    emitted: W::Emitted,
    door: &Door<R, W>,
) -> Result<DoorStage<'bytes, R, W>, DoorError<R, W>> {
    let package = door
        .writer
        .package(&lowered, emitted)
        .map_err(DoorError::Package)?;
    Ok(DoorStage::Packaged { lowered, package })
}

fn verify<'bytes, R: Reader, W: Writer>(
    lowered: Lowered,
    package: W::Package,
    door: &Door<R, W>,
) -> Result<DoorStage<'bytes, R, W>, DoorError<R, W>> {
    door.writer.verify(&package).map_err(DoorError::Verify)?;
    Ok(DoorStage::Verified { lowered, package })
}

fn plan<'bytes, R: Reader, W: Writer>(
    lowered: Lowered,
    door: &Door<R, W>,
    batch: &DoorBatch,
) -> Result<DoorStage<'bytes, R, W>, DoorError<R, W>> {
    let roots = output_roots(&lowered);
    let named = named_blocks(&lowered, batch);
    let settings = &door.settings;
    let plan = plan_named(
        settings.engine,
        settings.gpu_driver,
        &lowered.program,
        &[],
        &named,
        &roots,
        settings.numeric_policy,
    )
    .map_err(DoorError::Plan)?;
    door.cache.borrow_mut().record_plan();
    Ok(DoorStage::Planned { lowered, plan })
}

fn execute<'bytes, R: Reader, W: Writer>(
    lowered: Lowered,
    mut plan: Plan,
    batch: &DoorBatch,
) -> Result<DoorStage<'bytes, R, W>, DoorError<R, W>> {
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
