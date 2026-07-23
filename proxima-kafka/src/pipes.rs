//! Typed pipe handles for the Kafka broker-facade pipeline.
//!
//! `KafkaPipeRequest = Request<RequestBody>` / `KafkaPipeReply =
//! Response<ResponseBody>`: the business handler pipe is fully typed — no
//! downcast, no type erasure. `KafkaPipeHandle` is an instantiation of the
//! generic erased form `proxima_primitives::pipe::alloc_tier::PipeHandle<In,
//! Out>`. Mirrors `proxima_redis::pipes` / `proxima_pgwire::pipes` 1:1.

use proxima_primitives::pipe::alloc_tier;

use crate::wire::{RequestBody, ResponseBody};

/// Typed request carrying a decoded [`RequestBody`] as payload — every
/// Produce/Fetch/Metadata this facade recognizes reaches the handler
/// through this one shape (`ApiVersions` is answered protocol-level by the
/// connection driver itself and never reaches a handler — see
/// `crate::connection`'s doc).
pub type KafkaPipeRequest = proxima_primitives::pipe::request::Request<RequestBody>;

/// Typed response carrying an encoded-shape [`ResponseBody`] as payload.
pub type KafkaPipeReply = proxima_primitives::pipe::request::Response<ResponseBody>;

/// Runtime-erased handle for Kafka broker-facade handler pipes.
pub type KafkaPipeHandle = alloc_tier::PipeHandle<KafkaPipeRequest, KafkaPipeReply>;

/// Wrap any Kafka-compatible pipe in a [`KafkaPipeHandle`] — the bridge
/// between a business handler you write (`impl SendPipe<In =
/// KafkaPipeRequest, Out = KafkaPipeReply>`) and every seam that wants the
/// type-erased [`KafkaPipeHandle`] ([`crate::KafkaAnyProtocol::new`],
/// `proxima::ListenerProtocolExt::kafka`).
///
/// ```
/// use proxima_kafka::{KafkaPipeRequest, KafkaPipeReply, into_kafka_handle};
/// use proxima_core::ProximaError;
/// use proxima_primitives::pipe::SendPipe;
///
/// struct Broker;
/// impl SendPipe for Broker {
///     type In = KafkaPipeRequest;
///     type Out = KafkaPipeReply;
///     type Err = ProximaError;
///     async fn call(&self, _request: KafkaPipeRequest) -> Result<KafkaPipeReply, ProximaError> {
///         unreachable!("illustrative — no request is dispatched in this doctest")
///     }
/// }
///
/// let handle = into_kafka_handle(Broker);
/// # let _ = handle;
/// ```
pub use alloc_tier::into_handle as into_kafka_handle;
