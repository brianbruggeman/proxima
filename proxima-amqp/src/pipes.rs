//! Typed pipe handles for the AMQP business-handler seam.
//!
//! Everything protocol-level (connection/channel lifecycle, exchange/queue
//! declare, `basic.consume`/`cancel`/`qos`) is dispatched inside
//! [`crate::connection`] directly against [`crate::broker::AmqpBroker`] —
//! there is no per-command business verb the way redis has GET/SET/etc.
//! `basic.publish` is AMQP's one "here is a message, do something with it"
//! moment, so it is the one dispatch point a business handler gets: every
//! reassembled publish (see [`crate::fsm::Advanced::Publish`]) reaches
//! [`AmqpPipeHandle`] BEFORE the broker routes it, so a handler can
//! observe, transform, persist, or reject a message (an `Err` drops it
//! without routing) — mirroring redis's "everything not intercepted at the
//! protocol level reaches the handler" shape, generalized to AMQP's own
//! one business verb.

use proxima_primitives::pipe::alloc_tier;

/// One `basic.publish`'s business payload.
#[derive(Debug, Clone)]
pub struct AmqpMessage {
    pub exchange: Vec<u8>,
    pub routing_key: Vec<u8>,
    pub properties: Vec<u8>,
    pub body: Vec<u8>,
    pub mandatory: bool,
    pub immediate: bool,
}

/// Typed request carrying an [`AmqpMessage`] as payload.
pub type AmqpPipeRequest = proxima_primitives::pipe::request::Request<AmqpMessage>;

/// Typed response. AMQP 0-9-1 has no synchronous per-publish reply outside
/// publisher-confirms (not implemented — see the crate-level gap notes), so
/// the handler's reply carries no payload; only `Ok`/`Err` matters (route
/// vs. drop).
pub type AmqpPipeReply = proxima_primitives::pipe::request::Response<()>;

/// Runtime-erased handle for AMQP publish-handler pipes.
pub type AmqpPipeHandle = alloc_tier::PipeHandle<AmqpPipeRequest, AmqpPipeReply>;

/// Wrap any AMQP-compatible pipe in an [`AmqpPipeHandle`] — the bridge
/// between a business handler you write (`impl SendPipe<In =
/// AmqpPipeRequest, Out = AmqpPipeReply>`) and every seam that wants the
/// type-erased [`AmqpPipeHandle`] ([`crate::AmqpAnyProtocol::new`],
/// `proxima::ListenerProtocolExt::amqp`). `Err` drops the publish without
/// routing it — the one route-vs-drop decision a handler gets (see the
/// module doc).
///
/// ```
/// use proxima_amqp::{AmqpPipeRequest, AmqpPipeReply, into_amqp_handle};
/// use proxima_core::ProximaError;
/// use proxima_primitives::pipe::{SendPipe, request::Response};
///
/// struct ObservePublish;
/// impl SendPipe for ObservePublish {
///     type In = AmqpPipeRequest;
///     type Out = AmqpPipeReply;
///     type Err = ProximaError;
///     async fn call(&self, request: AmqpPipeRequest) -> Result<AmqpPipeReply, ProximaError> {
///         let _routing_key = &request.payload.routing_key; // observe, transform, persist, ...
///         Ok(Response::typed(200, ()))
///     }
/// }
///
/// let handle = into_amqp_handle(ObservePublish);
/// # let _ = handle;
/// ```
pub use alloc_tier::into_handle as into_amqp_handle;
