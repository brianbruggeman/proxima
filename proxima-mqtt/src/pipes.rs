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

/// Wrap any MQTT-compatible pipe in an [`MqttPipeHandle`].
pub use alloc_tier::into_handle as into_mqtt_handle;
