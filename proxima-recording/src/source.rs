use std::pin::Pin;

use futures::Stream;

use crate::event::RecordingEvent;
use proxima_core::ProximaError;

/// The event stream a [`RecordingSource`] yields. Boxed because the set of
/// sources is open — a consumer registers its own format through
/// [`crate::factory::RecordingSourceRegistry`] — and `RecordingSource` is
/// therefore consumed as `dyn`, where RPITIT is not object-safe.
pub type RecordingEventStream<'lifetime> =
    Pin<Box<dyn Stream<Item = Result<RecordingEvent, ProximaError>> + Send + 'lifetime>>;

pub trait RecordingSource: Send + Sync {
    fn events<'lifetime>(&'lifetime self) -> RecordingEventStream<'lifetime>;
}

pub type DynRecordingSource = std::sync::Arc<dyn RecordingSource>;
