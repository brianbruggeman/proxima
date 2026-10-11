use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::Instant;

use proxima_onnx::census::{Census, census};
use proxima_onnx::lower::{decode_initializers, lower_graph_pinned_decoded};
use proxima_onnx::messages::{GraphProto, ModelProto};
use proxima_onnx::pipe::parse_complete;
use proxima_onnx::{LowerError, OnnxError};
use proxima_tensor::Lowered;

use super::validation::ValidationError;
use crate::door::{Pins, Reader, ReaderWork};

/// What the validate stage keeps: the graph and the op census it passed.
#[derive(Debug)]
pub struct ValidatedGraph<'bytes> {
    pub graph: GraphProto<'bytes>,
    pub census: Census<'bytes>,
}

#[derive(Debug, Default)]
struct DecodeCache {
    initializers: Option<BTreeMap<String, Vec<f32>>>,
    work: ReaderWork,
}

/// The ONNX half of the door that reads: `ModelProto` bytes to
/// [`Lowered`] through [`proxima_onnx`].
///
/// Parse is [`parse_complete`], validate is the [`census`] of op types plus the
/// symbolic dimensions the pins leave open, lower is
/// [`lower_graph_pinned_decoded`]. The initializers are decoded once and kept
/// for every pin set, which is the one piece of state the reader owns.
#[derive(Debug, Default)]
pub struct OnnxReader {
    decoded: RefCell<DecodeCache>,
}

impl OnnxReader {
    /// A reader that has decoded nothing yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn decode(&self, graph: &GraphProto<'_>) -> Result<BTreeMap<String, Vec<f32>>, LowerError> {
        let started = Instant::now();
        let decoded = decode_initializers(graph)?;
        let floats: usize = decoded.values().map(Vec::len).sum();
        let mut cache = self.decoded.borrow_mut();
        cache.work.decodes += 1;
        cache.work.decoded_bytes += (floats * size_of::<f32>()) as u64;
        proxima_telemetry::debug!(
            decoded_bytes = cache.work.decoded_bytes,
            decode_ms = started.elapsed().as_secs_f64() * 1e3,
            "onnx reader decoded the initializers it shares across pin sets"
        );
        Ok(decoded)
    }
}

impl Reader for OnnxReader {
    type Parsed<'bytes> = ModelProto<'bytes>;
    type Validated<'bytes> = ValidatedGraph<'bytes>;
    type ParseError = OnnxError;
    type ValidateError = ValidationError;
    type LowerError = LowerError;

    fn parse<'bytes>(&self, bytes: &'bytes [u8]) -> Result<ModelProto<'bytes>, OnnxError> {
        parse_complete(bytes)
    }

    fn validate<'bytes>(
        &self,
        parsed: Self::Parsed<'bytes>,
        pins: &Pins,
    ) -> Result<Self::Validated<'bytes>, ValidationError> {
        let graph = parsed.graph.ok_or(ValidationError::NoGraph)?;
        let census = census(&graph);
        match ValidationError::rejection(&graph, &census, pins) {
            Some(rejection) => Err(rejection),
            None => Ok(ValidatedGraph { graph, census }),
        }
    }

    fn lower(&self, validated: &ValidatedGraph<'_>, pins: &Pins) -> Result<Lowered, LowerError> {
        let named: BTreeMap<&str, u64> = pins
            .iter()
            .map(|(name, extent)| (name.as_str(), *extent))
            .collect();
        let kept = self.decoded.borrow_mut().initializers.take();
        let decoded = match kept {
            Some(decoded) => decoded,
            None => self.decode(&validated.graph)?,
        };
        let lowered = lower_graph_pinned_decoded(&validated.graph, &named, &decoded);
        self.decoded.borrow_mut().initializers = Some(decoded);
        lowered
    }

    fn work(&self) -> ReaderWork {
        self.decoded.borrow().work
    }
}
