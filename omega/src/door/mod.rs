//! The door: a [`Pipe`] that takes the bytes of a model and one batch of
//! named input tensors and returns the program's output tensors, generic over
//! the format the bytes are in.
//!
//! The door is one enum state machine, [`DoorStage`], over eight stages. A
//! format supplies exactly two sans-IO halves and owns nothing else:
//!
//! - a [`Reader`], bytes to [`Lowered`]: parse, validate, lower;
//! - a [`Writer`], [`Lowered`] to a package: emit, package, verify.
//!
//! The door adds the stages every format shares: plan
//! ([`crate::backend::plan_named`]) and execute
//! ([`crate::backend::execute_plan_named`]). Each state holds only the data
//! legal in it, and a failure is a [`DoorError`] variant for the stage that
//! raised it, with the format's own error as its `#[source]`.
//!
//! # What a caller can do with the door
//!
//! Calling a format's reader, `plan_named` and `execute_plan_named` by hand
//! works. The door adds what hand-wrapping does not give:
//!
//! - one error type, [`DoorError`], with a variant per stage, where the
//!   hand-wrapped caller holds a different error type per call;
//! - plan reuse: the lowering and the plan of each pin set are kept, up to
//!   `max_pin_sets` of them, where a hand-wrapped caller re-lowers and
//!   re-plans on every call and owns its own cache; [`Door::stats`] counts the
//!   work done;
//! - composition: `Door` is a [`Pipe`], so `tokenizer.and_then(door)`
//!   ([`proxima_primitives::pipe::PipeExt::and_then`]) chains it behind any
//!   pipe whose output is a [`DoorBatch`];
//! - a new format is two trait impls, not a copy of the stage enum, the
//!   error, the cache and the pipe.
//!
//! The door names no format and no model and applies no pooling or
//! normalisation; what a model family needs on top of the program outputs is
//! the caller's.
//!
//! # Tier
//!
//! std, behind the `door` feature: it plans through [`crate::backend`]. The
//! future is not `Send` because [`Plan`] is `!Sync`, which is why this is a
//! [`Pipe`] and not a `SendPipe`.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::time::Instant;

use proxima_primitives::pipe::Pipe;
use proxima_tensor::{Evaluated, Lowered, NodeId};

use crate::backend::Plan;

mod cache;
mod error;
mod reader;
mod settings;
mod stage;
mod writer;

pub use cache::DoorStats;
use cache::PlanCache;
pub use error::{DoorError, StageKind};
pub use reader::{Pins, Reader, ReaderWork};
pub use settings::{DoorConfigError, DoorSettings};
pub use stage::DoorStage;
pub use writer::{NoSink, Writer};

/// One call's input: the extents that bind the program's symbolic dimensions,
/// and a tensor per program input.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DoorBatch {
    /// Symbolic dimension name to extent, e.g. `{"batch_size": 32,
    /// "sequence_length": 48}`.
    pub pins: BTreeMap<String, u64>,
    /// Program input name to its row-major values. Integer inputs are carried
    /// as `f32` holding the exact integer, e.g. token id `101.0`.
    pub inputs: Vec<(String, Vec<f32>)>,
}

/// One call's result: the evaluated tensors and which node is which program
/// output.
#[derive(Debug)]
pub struct DoorOutput {
    pub evaluated: Evaluated,
    /// Program output name and the node that computes it, e.g.
    /// `("last_hidden_state", NodeId(1243))`.
    pub outputs: Vec<(String, NodeId)>,
}

impl DoorOutput {
    /// The values and shape of the program output called `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<(&[f32], &[u64])> {
        let (_, node) = self.outputs.iter().find(|(output, _)| output == name)?;
        self.evaluated.get(*node)
    }
}

/// A format's reader and writer, the model bytes, and the engine to run the
/// program on; see the module doc.
///
/// Built with [`Door::new`], driven with [`Pipe::call`] to run a batch and
/// with [`Door::write`] to package a lowered program.
#[derive(Debug)]
pub struct Door<R: Reader, W: Writer = NoSink> {
    reader: R,
    writer: W,
    bytes: Vec<u8>,
    settings: DoorSettings,
    cache: RefCell<PlanCache>,
}

impl<R: Reader, W: Writer> Door<R, W> {
    /// Holds `bytes` for `reader` to read. A door that only writes passes
    /// empty bytes.
    pub fn new(reader: R, writer: W, bytes: impl Into<Vec<u8>>, settings: DoorSettings) -> Self {
        Self {
            reader,
            writer,
            bytes: bytes.into(),
            settings,
            cache: RefCell::new(PlanCache::new(settings.max_pin_sets)),
        }
    }

    /// The reader this door was built with.
    #[must_use]
    pub const fn reader(&self) -> &R {
        &self.reader
    }

    /// The writer this door was built with.
    #[must_use]
    pub const fn writer(&self) -> &W {
        &self.writer
    }

    /// What the door has parsed, decoded, lowered, planned and evicted so far.
    #[must_use]
    pub fn stats(&self) -> DoorStats {
        let work = self.reader.work();
        DoorStats {
            decodes: work.decodes,
            decoded_bytes: work.decoded_bytes,
            ..self.cache.borrow().stats()
        }
    }

    /// Emits, packages and verifies `lowered` with the writer.
    ///
    /// # Errors
    /// The failing writing stage's [`DoorError`] variant.
    pub fn write(&self, lowered: Lowered) -> Result<W::Package, DoorError<R, W>> {
        let batch = DoorBatch::default();
        let mut stage = DoorStage::emit(&self.writer, lowered)?;
        loop {
            stage = match stage {
                DoorStage::Verified { package, .. } => return Ok(package),
                other => other.advance(self, &batch)?,
            };
        }
    }

    fn run(&self, batch: &DoorBatch) -> Result<DoorOutput, DoorError<R, W>> {
        let mut stage = self.first_stage(batch)?;
        loop {
            stage = match stage {
                DoorStage::Executed {
                    lowered,
                    plan,
                    evaluated,
                } => return Ok(self.retain(batch, lowered, plan, evaluated)),
                other => other.advance(self, batch)?,
            };
        }
    }

    fn first_stage(&self, batch: &DoorBatch) -> Result<DoorStage<'_, R, W>, DoorError<R, W>> {
        let cached = self.cache.borrow_mut().take(&batch.pins);
        if let Some((lowered, plan)) = cached {
            return Ok(DoorStage::Planned { lowered, plan });
        }
        let started = Instant::now();
        let parsed = DoorStage::begin(&self.reader, &self.bytes)?;
        self.cache.borrow_mut().record_parse(started.elapsed());
        Ok(parsed)
    }

    fn retain(
        &self,
        batch: &DoorBatch,
        lowered: Lowered,
        plan: Plan,
        evaluated: Evaluated,
    ) -> DoorOutput {
        let outputs = lowered.graph_outputs.clone();
        self.cache
            .borrow_mut()
            .store(batch.pins.clone(), lowered, plan);
        DoorOutput { evaluated, outputs }
    }
}

impl<R: Reader, W: Writer> Pipe for Door<R, W>
where
    R: Debug + 'static,
    W: Debug + 'static,
{
    type In = DoorBatch;
    type Out = DoorOutput;
    type Err = DoorError<R, W>;

    fn call(&self, input: DoorBatch) -> impl Future<Output = Result<DoorOutput, DoorError<R, W>>> {
        async move { self.run(&input) }
    }
}
