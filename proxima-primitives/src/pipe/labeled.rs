use alloc::sync::Arc;
use bytes::Bytes;

use crate::pipe::request::{Request, Response};
use crate::pipe::telemetry_surface::{Labels, NoopTelemetry, TelemetryHandle};

/// Telemetry sink + base labels for a payload (HTTP: the request context).
///
/// Implement this on any input type to make `Retry` and `RateLimit` emit
/// telemetry. HTTP implements it on `Request` and `Response`.
pub trait Labeled {
    fn telemetry(&self) -> TelemetryHandle;
    fn labels(&self) -> Labels;
}

impl Labeled for Request<Bytes> {
    fn telemetry(&self) -> TelemetryHandle {
        self.context.telemetry.clone()
    }

    fn labels(&self) -> Labels {
        self.context.metric_labels(&[])
    }
}

impl Labeled for Response<Bytes> {
    fn telemetry(&self) -> TelemetryHandle {
        Arc::new(NoopTelemetry)
    }

    fn labels(&self) -> Labels {
        Labels::empty()
    }
}
