//! proxima's own AMQP 0-9-1 broker facade — client and listener, built on
//! the sans-IO frame envelope in [`proxima_protocols::amqp`].
//!
//! That upstream codec stops at the frame envelope
//! (type/channel/length/payload/0xCE — [`proxima_protocols::amqp::Frame`],
//! [`proxima_protocols::amqp::parse_frame`]): it does not decode a single
//! method argument. This crate lifts it into a full broker:
//!
//! - [`wire`] — AMQP 0-9-1 method-argument primitives this crate adds:
//!   scalars, `shortstr`/`longstr`, bit-packed flags, and the
//!   [`wire::FieldValue`]/[`wire::FieldTable`] grammar.
//! - [`method`] — typed [`method::Method`] decode/encode for the
//!   connection/channel/exchange/queue/basic subset a broker needs.
//! - [`frame`] — the write-side frame envelope [`proxima_protocols::amqp`]
//!   doesn't supply (it only decodes).
//! - [`fsm::Connection`] — the sans-IO server-side protocol state machine:
//!   protocol-header handshake, per-channel content (method + header +
//!   body) reassembly, DoS-capped frame/message sizes. No socket
//!   (workspace principle 11).
//! - [`topic::TopicSet`] — AMQP topic-exchange (`#`/`*`) binding-key
//!   matching, the routing-key sibling of
//!   `proxima_redis::glob::GlobSet`'s PSUBSCRIBE matching.
//!
//! The `client` feature adds the std client: [`client::AmqpClientUpstream`]
//! (an async `Pipe`, `basic.publish`/`basic.consume`) and
//! [`client::AmqpClient`] (a blocking driver), both driving the sans-IO
//! [`client::ClientSession`] over a pluggable transport — the same split
//! `proxima_redis::client` uses.
//!
//! The `listen` feature adds the server side: [`connection::serve_connection`]
//! (the per-connection I/O driver over the [`fsm::Connection`]),
//! [`broker::AmqpBroker`] (exchange -> queue routing on
//! [`proxima_primitives::pipe::KeyedFanOut`] — the same broadcast registry
//! `proxima_redis::broker::RedisBroker`'s PUBLISH/SUBSCRIBE fabric uses),
//! and [`any_protocol::AmqpAnyProtocol`] — the `AnyProtocol` candidate that
//! mounts AMQP into the open universal listener
//! (`Listener::builder().accept("amqp")`), mirroring
//! `proxima_redis::any_protocol::RedisAnyProtocol`'s shape exactly (no
//! standalone `AmqpListenProtocol` bind+accept loop).
//!
//! ## Scope gaps (read before relying on this as a drop-in RabbitMQ)
//!
//! `proxima_protocols::amqp` decodes only the frame *envelope* — every
//! method argument, every wire primitive (`shortstr`/`longstr`/field-table/
//! bit-packing), and the connection/channel FSM are built fresh in this
//! crate; there was no method-level codec to lift beyond the envelope.
//! Deliberately out of implemented scope on top of that:
//!
//! - **No message persistence.** [`broker::AmqpBroker`] is a live fan-out
//!   (like redis PUBLISH/SUBSCRIBE), not a store-and-forward queue — a
//!   `basic.publish` to a queue with zero consumers is dropped, not
//!   buffered. `queue.declare-ok`'s `message-count` is always `0`.
//! - **No publisher confirms / consumer acks.** `basic.qos`/`basic.ack`/
//!   `basic.nack` are accepted and no-op'd; every delivery behaves as if
//!   `no_ack` were set. No redelivery, no unacked-message tracking.
//! - **No `mandatory`/`immediate` enforcement.** Both flags decode
//!   correctly but an unroutable mandatory publish is silently dropped
//!   rather than triggering `basic.return`.
//! - **No SASL negotiation.** `connection.start-ok`'s `mechanism`/
//!   `response` are decoded but not authenticated — any client is
//!   admitted (this crate is a wire/routing facade, not an auth boundary;
//!   compose it behind `proxima-auth` for real credential checks, the same
//!   posture redis's facade takes).
//! - **`headers`-kind exchanges are not implemented** — only
//!   `direct`/`fanout`/`topic` ([`broker::ExchangeKind`]); declaring a
//!   `headers` exchange fails with `channel.close` (`COMMAND_INVALID`).
//! - **`channel.flow`, `basic.get`, `queue.{purge,unbind,delete}`,
//!   `exchange.delete`, `tx.*`, `confirm.*`** are not decoded —
//!   [`method::decode`] reports [`method::MethodError::Unsupported`] for
//!   any of these, which the connection driver renders as
//!   `connection.close`.

pub mod frame;
pub mod fsm;
pub mod method;
pub mod topic;
pub mod wire;

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

pub use fsm::{Advanced, Connection, Limits, PROTOCOL_HEADER};
pub use method::{Method, MethodError};
pub use wire::{FieldTable, FieldValue, WireError};

#[cfg(feature = "client")]
pub use client::{
    AmqpClient, AmqpClientConfig, AmqpClientUpstream, AmqpConfigError, ClientDelivery, ClientError,
    ClientSession, Step,
};

// the server-side surface a broker handler builds against — re-exported so
// an engine author imports everything from proxima-amqp and never reaches
// past it into proxima-primitives/proxima-protocols internals (teaching
// surface, workspace principle 2), mirroring proxima-redis's own top-level
// re-export shape.
#[cfg(feature = "listen")]
pub use any_protocol::AmqpAnyProtocol;
#[cfg(feature = "listen")]
pub use broker::{AmqpBroker, ConsumerSink, Delivery, ExchangeKind};
#[cfg(feature = "listen")]
pub use config::AmqpServerConfig;
#[cfg(feature = "listen")]
pub use connection::serve_connection;
#[cfg(feature = "listen")]
pub use error::AmqpServeError;
#[cfg(feature = "listen")]
pub use pipe::AmqpConnectionPipe;
#[cfg(feature = "listen")]
pub use pipes::{AmqpMessage, AmqpPipeHandle, AmqpPipeReply, AmqpPipeRequest, into_amqp_handle};
