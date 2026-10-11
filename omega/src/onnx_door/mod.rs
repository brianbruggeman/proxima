//! The ONNX door: a [`Pipe`] that takes the bytes of an ONNX model and one
//! batch of named input tensors and returns the graph's output tensors.
//!
//! The door is an enum state machine, [`DoorStage`], over five stages: parse
//! ([`proxima_onnx::pipe::parse_complete`]), validate (the
//! [`proxima_onnx::census`] of op types, and the symbolic dimensions the batch
//! pins), lower ([`proxima_onnx::lower::lower_graph_pinned`]), plan
//! ([`crate::backend::plan_named`]) and execute
//! ([`crate::backend::execute_plan_named`]). Each state holds only the data
//! legal in it, and a failure is a [`DoorError`] variant for the stage that
//! raised it.
//!
//! # What a caller can do with the door
//!
//! Calling `parse_complete`, `lower_graph_pinned`, `plan_named` and
//! `execute_plan_named` by hand works, and `examples/wire_to_weights.rs` does.
//! The door adds three things that hand-wrapping does not give:
//!
//! - one error type, [`DoorError`], with a variant per stage, where the
//!   hand-wrapped caller holds a different error type per call;
//! - validation before lowering: every unlowerable op type and every unpinned
//!   symbolic dimension is reported in one [`ValidationError`], where
//!   `lower_graph_pinned` stops at the first node it cannot lower;
//! - composition: `OnnxDoor` is a [`Pipe`], so `tokenizer.and_then(door)`
//!   ([`proxima_primitives::pipe::PipeExt::and_then`]) chains it behind any pipe whose output is a
//!   [`DoorBatch`].
//!
//! The door names no model and applies no pooling or normalisation; what a
//! model family needs on top of the graph outputs is the caller's.
//!
//! # Tier
//!
//! std, behind the `onnx-door` feature: it plans through [`crate::backend`]
//! and configures through `conflaguration`. The future is not `Send` because
//! [`Plan`](crate::backend::Plan) is `!Sync`, which is why this is a
//! [`Pipe`] and not a `SendPipe`.

use std::collections::BTreeMap;

use proxima_primitives::pipe::Pipe;
use proxima_tensor::{Evaluated, NodeId, NumericPolicy};

use crate::backend::{Engine, GpuDriver};

mod config;
mod error;
mod stage;

pub use config::{DoorConfig, DoorConfigBuilder, DoorConfigError, door_config_builder};
pub use error::{DoorError, StageKind, ValidationError};
pub use stage::DoorStage;

/// One call's input: the extents that bind the graph's symbolic dimensions,
/// and a tensor per graph input.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DoorBatch {
    /// Symbolic dimension name to extent, e.g. `{"batch_size": 32,
    /// "sequence_length": 48}`.
    pub pins: BTreeMap<String, u64>,
    /// Graph input name to its row-major values. Integer inputs are carried
    /// as `f32` holding the exact integer, e.g. token id `101.0`.
    pub inputs: Vec<(String, Vec<f32>)>,
}

/// One call's result: the evaluated tensors and which node is which graph
/// output.
#[derive(Debug)]
pub struct DoorOutput {
    pub evaluated: Evaluated,
    /// Graph output name and the node that computes it, e.g.
    /// `("last_hidden_state", NodeId(1243))`.
    pub outputs: Vec<(String, NodeId)>,
}

impl DoorOutput {
    /// The values and shape of the graph output called `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<(&[f32], &[u64])> {
        let (_, node) = self.outputs.iter().find(|(output, _)| output == name)?;
        self.evaluated.get(*node)
    }
}

/// An ONNX model and the engine to run it on; see the module doc.
///
/// Built from model bytes with [`OnnxDoor::from_bytes`], driven with
/// [`Pipe::call`].
#[derive(Debug)]
pub struct OnnxDoor {
    bytes: Vec<u8>,
    engine: Engine,
    gpu_driver: Option<GpuDriver>,
    numeric_policy: NumericPolicy,
}

impl OnnxDoor {
    /// Holds `bytes` and resolves `config` once, so a misspelled engine fails
    /// here and not on the first call.
    ///
    /// # Errors
    /// [`DoorConfigError`] when a name in `config` is not a compiled-in value.
    pub fn from_bytes(
        bytes: impl Into<Vec<u8>>,
        config: &DoorConfig,
    ) -> Result<Self, DoorConfigError> {
        Ok(Self {
            bytes: bytes.into(),
            engine: config.resolve_engine()?,
            gpu_driver: config.resolve_gpu_driver()?,
            numeric_policy: config.resolve_numeric_policy()?,
        })
    }

    fn run(&self, batch: &DoorBatch) -> Result<DoorOutput, DoorError> {
        let mut stage = DoorStage::start(&self.bytes)?;
        loop {
            stage = match stage {
                DoorStage::Executed {
                    lowered, evaluated, ..
                } => {
                    return Ok(DoorOutput {
                        evaluated,
                        outputs: lowered.graph_outputs,
                    });
                }
                other => other.advance(self, batch)?,
            };
        }
    }
}

impl Pipe for OnnxDoor {
    type In = DoorBatch;
    type Out = DoorOutput;
    type Err = DoorError;

    fn call(&self, input: DoorBatch) -> impl Future<Output = Result<DoorOutput, DoorError>> {
        async move { self.run(&input) }
    }
}
