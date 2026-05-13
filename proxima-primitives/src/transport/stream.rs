use std::pin::Pin;

use futures::Stream;

use proxima_core::ProximaError;

/// Boxed, send-able generic stream of fallible items.
/// The uniform stream seam for all transport primitives in this crate.
pub type GenericStream<T> = Pin<Box<dyn Stream<Item = Result<T, ProximaError>> + Send>>;
