//! The ONNX door: [`Door`] instantiated with [`OnnxReader`] and no writer. It
//! takes the bytes of an ONNX model and one batch of named input tensors and
//! returns the graph's output tensors.
//!
//! This module names the ONNX instance and nothing else. The stage machine,
//! the error, the plan cache and the pipe shape are [`crate::door`]'s, shared
//! by every format; ONNX supplies [`OnnxReader`], whose parse
//! ([`proxima_onnx::pipe::parse_complete`]), validate (the
//! [`proxima_onnx::census`] of op types, and the symbolic dimensions the batch
//! pins) and lower ([`proxima_onnx::lower::lower_graph_pinned`]) steps the
//! door drives, then plans and executes through [`crate::backend`].
//!
//! # What a caller can do with the door
//!
//! Calling `parse_complete`, `lower_graph_pinned`, `plan_named` and
//! `execute_plan_named` by hand works, and `examples/wire_to_weights.rs` does.
//! The door adds one error type with a variant per stage, validation that
//! reports every unlowerable op and unpinned dimension in one
//! [`ValidationError`], plan reuse across pin sets counted by
//! [`OnnxDoor::stats`], and composition: `OnnxDoor` is a [`Pipe`](proxima_primitives::pipe::Pipe), so
//! `tokenizer.and_then(door)`
//! ([`proxima_primitives::pipe::PipeExt::and_then`]) chains it behind any pipe
//! whose output is a [`DoorBatch`].
//!
//! # Tier
//!
//! std, behind the `onnx-door` feature, which implies `door`; configuration is
//! through `conflaguration` under the `OMEGA_ONNX_DOOR` prefix.

use crate::door::{Door, DoorError as GenericDoorError, DoorStage as GenericDoorStage};

mod config;
mod reader;
mod validation;

pub use crate::door::{
    DoorBatch, DoorConfigError, DoorOutput, DoorSettings, DoorStats, NoSink, StageKind,
};
pub use config::{DoorConfig, DoorConfigBuilder, door_config_builder};
pub use reader::{OnnxReader, ValidatedGraph};
pub use validation::ValidationError;

/// The one generic [`Door`] reading ONNX and writing nothing.
pub type OnnxDoor = Door<OnnxReader, NoSink>;

/// [`GenericDoorError`] for the ONNX instance; the variants are the same.
pub type DoorError = GenericDoorError<OnnxReader, NoSink>;

/// [`GenericDoorStage`] for the ONNX instance; the states are the same.
pub type DoorStage<'bytes> = GenericDoorStage<'bytes, OnnxReader, NoSink>;

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
        let settings = config.settings()?;
        Ok(Self::new(OnnxReader::new(), NoSink, bytes, settings))
    }
}

impl<'bytes> DoorStage<'bytes> {
    /// Parses `bytes` into the first state of the ONNX instance.
    ///
    /// # Errors
    /// [`DoorError::Parse`] when the wire bytes are not a `ModelProto`.
    pub fn start(bytes: &'bytes [u8]) -> Result<Self, DoorError> {
        GenericDoorStage::begin(&OnnxReader::new(), bytes)
    }
}
