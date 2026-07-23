//! proxima's own MQTT v3.1.1 broker facade.
//!
//! The sans-IO MQTT codec ([`Packet`], [`ParseError`], [`parse_packet`],
//! [`encode`], [`Connection`]) and the MQTT-over-`Pipe` contract
//! ([`pipe_contract`]) live in [`proxima_protocols::mqtt`] — see its docs
//! for the wire layer. This crate is the std facade built on top: the
//! async [`client::MqttClientUpstream`] Pipe driving the sans-IO
//! [`client::ClientSession`] over a pluggable transport (prime, tokio,
//! TLS-wrapped) — the same split `proxima-redis` uses between
//! `proxima_protocols::redis` and its own client.
//!
//! The `listen` feature (below) adds the server side: [`connection`]'s
//! sans-IO-over-any-`futures::io`-stream driver, [`pipe::MqttConnectionPipe`]
//! (the connection layer as a real `Pipe`), and
//! [`any_protocol::MqttAnyProtocol`] — the `AnyProtocol` candidate that
//! mounts mqtt into the open universal listener
//! (`Listener::builder().accept("mqtt")`). There is no standalone
//! `MqttListenProtocol` bind+accept loop: mqtt's pub/sub listen-side
//! surface mirrors redis's own `RedisAnyProtocol`-only shape exactly —
//! [`broker::MqttBroker`] reuses
//! [`proxima_primitives::pipe::KeyedFanOut`] for PUBLISH/SUBSCRIBE fan-out
//! the same way `proxima_redis::broker::RedisBroker` does for
//! PUBLISH/SUBSCRIBE/PSUBSCRIBE.
//!
//! ## Scope
//!
//! **v3.1.1 only** ([`Packet::Connect`]'s `protocol_level` is read as `4`;
//! there is no v5 property/reason-code parsing). **Every delivery is
//! downgraded to QoS 0 and the retain flag is cleared** — [`broker::MqttBroker`]
//! fans a `PUBLISH` out to every subscriber as QoS 0, regardless of the
//! QoS the publisher used or the QoS a subscriber requested in its
//! `SUBSCRIBE` (`SUBACK`'s `granted` is always `[0, 0, ...]`); there is no
//! QoS 1/2 redelivery bookkeeping and **no retained-message storage** — a
//! late subscriber never receives a publish that happened before it
//! subscribed. This is a routing fabric (like redis PUBLISH/SUBSCRIBE),
//! not a durable broker.

#[cfg(feature = "client")]
pub mod client;

#[cfg(feature = "listen")]
pub mod any_protocol;
#[cfg(feature = "listen")]
pub mod broker;
#[cfg(feature = "listen")]
pub mod config;
#[cfg(feature = "listen")]
pub mod connection;
#[cfg(feature = "listen")]
pub mod error;
#[cfg(feature = "listen")]
pub mod pipe;
#[cfg(feature = "listen")]
pub mod pipes;
#[cfg(feature = "listen")]
pub mod topic_filter;
#[cfg(feature = "listen")]
pub mod wait_sources;

pub use proxima_protocols::mqtt::{
    Connection, ParseError, Packet, PacketType, parse_packet, pipe_contract,
};
pub use proxima_protocols::mqtt::pipe_contract::{MqttReply, MqttRequest, verb};

#[cfg(feature = "client")]
pub use client::{ClientError, ClientSession, MqttClientConfig, MqttClientUpstream, MqttConfigError, Step};

// the server-side surface an MQTT connect-auth handler builds against —
// re-exported so an engine author imports everything from proxima-mqtt
// and never reaches past it into proxima-primitives/proxima-protocols
// internals (teaching surface, workspace principle 2), mirroring
// proxima-redis's own top-level re-export shape.
#[cfg(feature = "listen")]
pub use any_protocol::MqttAnyProtocol;
#[cfg(feature = "listen")]
pub use broker::{MqttBroker, PushSink};
#[cfg(feature = "listen")]
pub use config::MqttServerConfig;
#[cfg(feature = "listen")]
pub use connection::serve_connection;
#[cfg(feature = "listen")]
pub use error::MqttServeError;
#[cfg(feature = "listen")]
pub use pipe::MqttConnectionPipe;
#[cfg(feature = "listen")]
pub use pipes::{MqttPipeHandle, MqttPipeReply, MqttPipeRequest, into_mqtt_handle};
#[cfg(feature = "listen")]
pub use topic_filter::TopicFilterSet;
