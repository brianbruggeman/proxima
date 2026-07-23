//! Typed pipe handles for the Kafka broker-facade pipeline.
//!
//! The business handler pipe carries [`RequestBody`] straight to
//! [`ResponseBody`] — no `Request`/`Response` envelope cell (payload-no-cell:
//! a pipe is `P -> Q`, and `RequestBody`/`ResponseBody` are already
//! self-describing — `correlation_id` lives outside them entirely, threaded
//! by `crate::framed_app::dispatch` directly, never through the envelope).
//! `KafkaPipeHandle` is an instantiation of the generic erased form
//! `proxima_primitives::pipe::alloc_tier::PipeHandle<In, Out>`. Mirrors
//! `proxima_redis::pipes` / `proxima_pgwire::pipes` 1:1.

use proxima_primitives::pipe::alloc_tier;

use crate::wire::{RequestBody, ResponseBody};

/// Runtime-erased handle for Kafka broker-facade handler pipes.
pub type KafkaPipeHandle = alloc_tier::PipeHandle<RequestBody, ResponseBody>;

/// Wrap any Kafka-compatible pipe in a [`KafkaPipeHandle`] — the bridge
/// between a business handler you write (`impl SendPipe<In = RequestBody,
/// Out = ResponseBody>`) and every seam that wants the type-erased
/// [`KafkaPipeHandle`] ([`crate::KafkaAnyProtocol::new`],
/// `proxima::ListenerProtocolExt::kafka`).
///
/// ```
/// use proxima_kafka::{RequestBody, ResponseBody, into_kafka_handle};
/// use proxima_core::ProximaError;
/// use proxima_primitives::pipe::SendPipe;
///
/// struct Broker;
/// impl SendPipe for Broker {
///     type In = RequestBody;
///     type Out = ResponseBody;
///     type Err = ProximaError;
///     async fn call(&self, _request: RequestBody) -> Result<ResponseBody, ProximaError> {
///         unreachable!("illustrative — no request is dispatched in this doctest")
///     }
/// }
///
/// let handle = into_kafka_handle(Broker);
/// # let _ = handle;
/// ```
pub use alloc_tier::into_handle as into_kafka_handle;
