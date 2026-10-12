use proxima_mil::census::census;
use proxima_mil::weights::SourceDType;
use proxima_mil::writer::{MilError, MilPackage, MilRequest, MilWriter, WeightView};
use proxima_tensor::{Lowered, Shapes, infer};

use super::Writer;

/// The Core ML half of the door that writes: a [`Lowered`] program in, an
/// `.mlpackage` held as three in-memory files out.
///
/// It is the [`Writer`] impl over [`proxima_mil::writer::MilWriter`], the
/// sans-IO `Pipe` that does the encoding. Emit refuses a program the
/// [`proxima_mil::census`] has no MIL form for and resolves the shapes and the
/// little-endian weight bytes; package hands those to `MilWriter`; verify
/// checks the three files the compiler reads are present. `xcrun
/// coremlcompiler` is the edge the verify stage calls once the `coreml-lane`
/// feature supplies it, as `plan_named` is the edge to Metal.
///
/// Compose it into a door with [`Door::new`](super::Door::new); drive the
/// writing stages with [`Door::write`](super::Door::write). Calling
/// `MilWriter` by hand works and gives the same bytes; the sink adds the
/// stage-tagged [`DoorError`](super::DoorError) and shares the door's plan
/// cache and reader with every other instance.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MilSink;

impl Writer for MilSink {
    type Emitted = (Shapes, Vec<Vec<u8>>);
    type Package = MilPackage;
    type Error = MilError;

    fn emit(&self, lowered: &Lowered) -> Result<Self::Emitted, MilError> {
        output_name(lowered)?;
        let offenders = census(&lowered.program, &[])?.refused;
        if !offenders.is_empty() {
            return Err(MilError::NoMilForm { offenders });
        }
        let shapes = infer(&lowered.program, &[])?;
        Ok((shapes, little_endian_weights(lowered)))
    }

    fn package(
        &self,
        lowered: &Lowered,
        (shapes, encoded): Self::Emitted,
    ) -> Result<MilPackage, MilError> {
        let weights = weight_views(lowered, &encoded);
        let request = MilRequest {
            program: &lowered.program,
            shapes: &shapes,
            weights: &weights,
            output: output_name(lowered)?,
        };
        MilWriter::new().write(&request)
    }

    fn verify(&self, package: &MilPackage) -> Result<(), MilError> {
        match package.files().iter().find(|(_, bytes)| bytes.is_empty()) {
            Some((path, _)) => Err(MilError::EmptyPackageFile { path }),
            None => Ok(()),
        }
    }
}

fn output_name(lowered: &Lowered) -> Result<&str, MilError> {
    let (name, node) = lowered.graph_outputs.first().ok_or(MilError::NoOutput)?;
    match node.0 as usize + 1 == lowered.program.len() {
        true => Ok(name),
        false => Err(MilError::OutputNotFinal { node: *node }),
    }
}

fn little_endian_weights(lowered: &Lowered) -> Vec<Vec<u8>> {
    lowered
        .initializers
        .iter()
        .map(|(_, values)| {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        })
        .collect()
}

fn weight_views<'a>(lowered: &'a Lowered, encoded: &'a [Vec<u8>]) -> Vec<WeightView<'a>> {
    lowered
        .initializers
        .iter()
        .zip(encoded)
        .map(|((name, _), bytes)| WeightView {
            name,
            dtype: SourceDType::Float32,
            bytes,
        })
        .collect()
}
