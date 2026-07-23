//! Typed pipe handles for the MQTT pipeline.
//!
//! `MqttPipeRequest = Request<MqttRequest>` / `MqttPipeReply =
//! Response<MqttReply>`: the business handler pipe is fully typed — no
//! downcast, no type erasure. `MqttPipeHandle` is an instantiation of the
//! generic erased form `proxima_primitives::pipe::alloc_tier::PipeHandle<In,
//! Out>`. Mirrors `proxima_redis::pipes` 1:1.

use proxima_primitives::pipe::alloc_tier;

use proxima_protocols::mqtt::{MqttReply, MqttRequest};

/// Typed request carrying an [`MqttRequest`] as payload — one of
/// `CONNECT`/`PUBLISH`/`SUBSCRIBE`/`UNSUBSCRIBE`/`PING`/`DISCONNECT`.
pub type MqttPipeRequest = proxima_primitives::pipe::request::Request<MqttRequest>;

/// Typed response carrying an [`MqttReply`] as payload.
pub type MqttPipeReply = proxima_primitives::pipe::request::Response<MqttReply>;

/// Runtime-erased handle for MQTT connect/publish-handler pipes.
pub type MqttPipeHandle = alloc_tier::PipeHandle<MqttPipeRequest, MqttPipeReply>;

/// Wrap any MQTT-compatible pipe in an [`MqttPipeHandle`] — the bridge
/// between a business handler you write (`impl SendPipe<In =
/// MqttPipeRequest, Out = MqttPipeReply>`) and every seam that wants the
/// type-erased [`MqttPipeHandle`] ([`crate::MqttAnyProtocol::new`],
/// `proxima::ListenerProtocolExt::mqtt`).
///
/// ```
/// use proxima_mqtt::{MqttPipeRequest, MqttPipeReply, MqttReply, MqttRequest, into_mqtt_handle};
/// use proxima_core::ProximaError;
/// use proxima_primitives::pipe::{SendPipe, request::Response};
///
/// struct AcceptAll;
/// impl SendPipe for AcceptAll {
///     type In = MqttPipeRequest;
///     type Out = MqttPipeReply;
///     type Err = ProximaError;
///     async fn call(&self, request: MqttPipeRequest) -> Result<MqttPipeReply, ProximaError> {
///         let reply = match request.payload {
///             MqttRequest::Connect { .. } => MqttReply::ConnAck { session_present: false, return_code: 0 },
///             MqttRequest::Publish { .. } => MqttReply::Published,
///             MqttRequest::Subscribe { filters } => {
///                 MqttReply::SubAck { packet_id: 1, granted: vec![0; filters.len()] }
///             }
///             MqttRequest::Unsubscribe { .. } => MqttReply::UnsubAck { packet_id: 1 },
///             MqttRequest::Ping => MqttReply::Pong,
///             MqttRequest::Disconnect => MqttReply::Disconnected,
///         };
///         Ok(Response::typed(200, reply))
///     }
/// }
///
/// let handle = into_mqtt_handle(AcceptAll);
/// # let _ = handle;
/// ```
pub use alloc_tier::into_handle as into_mqtt_handle;
