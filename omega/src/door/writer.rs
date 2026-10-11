use std::convert::Infallible;
use std::error::Error;

use proxima_tensor::Lowered;

/// The writing half of a format: a [`Lowered`] program in, the format's
/// package out, as three sans-IO steps the door's
/// [`DoorStage`](super::DoorStage) drives one at a time: emit, package, verify.
///
/// Each step's failure surfaces as the door's `Emit`, `Package` or `Verify`
/// error with [`Writer::Error`] as the `#[source]`. A format that only reads
/// uses [`NoSink`], which the door never drives.
pub trait Writer {
    /// What emit produces, e.g. the encoded graph before it is bundled.
    type Emitted;
    /// What package produces, the finished artifact a caller keeps.
    type Package;
    /// Why any of the three steps failed.
    type Error: Error + 'static;

    /// Encodes the program in the format's own wire form.
    ///
    /// # Errors
    /// [`Writer::Error`] for a node the format cannot express.
    fn emit(&self, lowered: &Lowered) -> Result<Self::Emitted, Self::Error>;

    /// Bundles the emitted value with whatever else the artifact carries.
    ///
    /// # Errors
    /// [`Writer::Error`] when the bundle cannot be formed.
    fn package(
        &self,
        lowered: &Lowered,
        emitted: Self::Emitted,
    ) -> Result<Self::Package, Self::Error>;

    /// Checks the finished package before it is returned.
    ///
    /// # Errors
    /// [`Writer::Error`] when the package is rejected.
    fn verify(&self, package: &Self::Package) -> Result<(), Self::Error>;
}

/// The writer of a door that only reads. Its steps cannot fail and the door
/// never calls them on the read path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoSink;

impl Writer for NoSink {
    type Emitted = ();
    type Package = ();
    type Error = Infallible;

    fn emit(&self, _lowered: &Lowered) -> Result<(), Infallible> {
        Ok(())
    }

    fn package(&self, _lowered: &Lowered, _emitted: ()) -> Result<(), Infallible> {
        Ok(())
    }

    fn verify(&self, _package: &()) -> Result<(), Infallible> {
        Ok(())
    }
}
