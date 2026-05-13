use std::pin::Pin;

use futures::Stream;

use crate::event::RecordingEvent;
use proxima_core::ProximaError;

pub type RecordingEventStream<'lifetime> =
    Pin<Box<dyn Stream<Item = Result<RecordingEvent, ProximaError>> + Send + 'lifetime>>;

pub trait RecordingSource: Send + Sync {
    fn events<'lifetime>(&'lifetime self) -> RecordingEventStream<'lifetime>;
}

pub type DynRecordingSource = std::sync::Arc<dyn RecordingSource>;
