use proxima_mil::writer::{MilError, MilPackage};
use proxima_tensor::{Lowered, Shapes};

use super::compile::{CompileError, Compiler};
use crate::door::{MilSink, Writer};

/// Why a writing stage of the Core ML door failed: the writer's own refusal
/// at emit and package, the compiler's at verify.
#[derive(Debug, thiserror::Error)]
pub enum LaneError {
    #[error("{0}")]
    Mil(#[from] MilError),

    #[error("{0}")]
    Compile(#[from] CompileError),
}

/// The Core ML door's [`Writer`]: [`MilSink`] for emit and package, and
/// [`Compiler`] for verify.
///
/// Verify is where `xcrun coremlcompiler` judges the package, so a program
/// the compiler refuses surfaces as the door's `Verify` error carrying
/// [`CompileError::UnknownOperator`] or [`CompileError::MissingParam`], while
/// a program MIL has no form for surfaces as `Emit` carrying
/// [`MilError::NoMilForm`]. [`MilSink`] alone checks only that the package
/// files are present; compose this one when the compiler is the validator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreMlSink {
    compiler: Compiler,
}

impl CoreMlSink {
    #[must_use]
    pub const fn new(compiler: Compiler) -> Self {
        Self { compiler }
    }
}

impl Writer for CoreMlSink {
    type Emitted = (Shapes, Vec<Vec<u8>>);
    type Package = MilPackage;
    type Error = LaneError;

    fn emit(&self, lowered: &Lowered) -> Result<Self::Emitted, LaneError> {
        Ok(MilSink.emit(lowered)?)
    }

    fn package(&self, lowered: &Lowered, emitted: Self::Emitted) -> Result<MilPackage, LaneError> {
        Ok(MilSink.package(lowered, emitted)?)
    }

    fn verify(&self, package: &MilPackage) -> Result<(), LaneError> {
        MilSink.verify(package)?;
        self.compiler.compile(package)?;
        Ok(())
    }
}
