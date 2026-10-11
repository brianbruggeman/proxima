use std::collections::BTreeMap;
use std::error::Error;

use proxima_tensor::Lowered;

/// Symbolic dimension name to extent, the pin set a call binds.
pub type Pins = BTreeMap<String, u64>;

/// Work a reader did that the door cannot see, e.g. `decodes: 1` after the
/// weights of a model were decoded once for every pin set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReaderWork {
    /// Times initializers were decoded to f32.
    pub decodes: u64,
    /// f32 bytes those decodes produced, e.g. `133_000_000`.
    pub decoded_bytes: u64,
}

/// The reading half of a format: bytes in, a [`Lowered`] program out, as
/// three sans-IO steps the door's [`DoorStage`](super::DoorStage) drives one
/// at a time.
///
/// A reader is the pipe `bytes + pins -> Lowered` with its parse, validate and
/// lower stages exposed, so a failure names the stage that raised it and
/// carries the format's own error as the `#[source]`. Each step consumes the
/// previous step's value, so a stale value cannot be reused. Compose a reader
/// into a door with [`Door::new`](super::Door::new); the door owns the stage
/// order, the error type, the plan cache and the pipe shape, so a format
/// supplies only these three steps.
pub trait Reader {
    /// What the parse step holds, borrowing from the input bytes.
    type Parsed<'bytes>;
    /// What the validate step holds, borrowing from the input bytes.
    type Validated<'bytes>;
    /// Why parse failed.
    type ParseError: Error + 'static;
    /// Why validate refused the parsed value.
    type ValidateError: Error + 'static;
    /// Why lowering failed.
    type LowerError: Error + 'static;

    /// Reads `bytes` into the format's own parsed value.
    ///
    /// # Errors
    /// [`Reader::ParseError`] when the bytes are not the format.
    fn parse<'bytes>(&self, bytes: &'bytes [u8]) -> Result<Self::Parsed<'bytes>, Self::ParseError>;

    /// Checks the parsed value against `pins` before lowering is paid for.
    ///
    /// # Errors
    /// [`Reader::ValidateError`] naming every reason lowering cannot succeed.
    fn validate<'bytes>(
        &self,
        parsed: Self::Parsed<'bytes>,
        pins: &Pins,
    ) -> Result<Self::Validated<'bytes>, Self::ValidateError>;

    /// Lowers the validated value at `pins` into the format-neutral program.
    ///
    /// # Errors
    /// [`Reader::LowerError`] for the first node that cannot lower.
    fn lower(
        &self,
        validated: &Self::Validated<'_>,
        pins: &Pins,
    ) -> Result<Lowered, Self::LowerError>;

    /// Work done inside the reader, merged into [`DoorStats`](super::DoorStats).
    fn work(&self) -> ReaderWork {
        ReaderWork::default()
    }
}
