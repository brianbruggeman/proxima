//! Protocol axis for [`ListenerBuilder`] — TYPE-SPECIFIC (no blanket impl
//! over every `SpecBuilder`, unlike the retired
//! `proxima_config::sugar::ProtocolSugar`). `.http()`/`.https()`/`.grpc()`
//! are the url-less listener twins of the client's own axis (a listener
//! dispatches to a `.handle(pipe)` already on hand, it doesn't dial out).
//! `.kafka()`/`.mqtt()`/`.amqp()`/`.memcached()`/`.redis()` take the crate's
//! typed handle and delegate to the existing
//! [`ListenerBuilder::protocol`](crate::listener::handle::ListenerBuilder::protocol)
//! seam — the SAME mechanism a third-party protocol uses (see the
//! `TestThriftExt`-style test in `tests/e2e`). `.pgwire()` (bespoke — see
//! its own doc), `.dns()` (dual-transport — see its own doc), and
//! `.websocket()` (h1 upgrade wiring, not a peer `AnyProtocol` — see its own
//! doc) are the three axes that do NOT delegate straight to `.protocol()`.
//!
//! The impl block lives in `handle.rs`, not here — `.pgwire()`/`.dns()`/
//! `.websocket()` accumulate onto private `ListenerBuilder` fields
//! (`pgwire_query`/`dns_handler`/`websocket_handler`), so the whole trait
//! impl (Rust requires one coherent `impl Trait for Type` block) stays
//! where those fields are defined.

/// `.http()`/`.https()`/`.grpc()` + every protocol terminal axis. Bring into
/// scope with `use proxima::ListenerProtocolExt;` (or `proxima::prelude::*`).
pub trait ListenerProtocolExt: Sized {
    /// Bind address carrier (the `http` spec key) — see
    /// [`bind_from_spec`](crate::listener::handle::bind_from_spec). Not a
    /// dial url on this side; a listener dispatches to `.handle(pipe)`.
    #[must_use]
    fn http(self, bind: impl Into<String>) -> Self;

    /// Same as [`Self::http`] — TLS falls out of `.tls(config)`, not the
    /// scheme, on the listener side.
    #[must_use]
    fn https(self, bind: impl Into<String>) -> Self;

    /// Select gRPC as the listen protocol (rides h2) — url-less, since a
    /// listener dispatches to `.handle(pipe)` rather than dialing out.
    #[must_use]
    fn grpc(self) -> Self;

    /// Select Kafka as the listen protocol, delegating to `.protocol(impl
    /// AnyProtocol)`.
    #[cfg(all(
        feature = "kafka-listener",
        any(feature = "http1", feature = "http1-native")
    ))]
    #[must_use]
    fn kafka(self, handler: proxima_kafka::KafkaPipeHandle) -> Self;

    /// Select MQTT as the listen protocol, delegating to `.protocol(impl
    /// AnyProtocol)`.
    #[cfg(all(
        feature = "mqtt-listener",
        any(feature = "http1", feature = "http1-native")
    ))]
    #[must_use]
    fn mqtt(self, handler: proxima_mqtt::MqttPipeHandle) -> Self;

    /// Select AMQP as the listen protocol, delegating to `.protocol(impl
    /// AnyProtocol)`.
    #[cfg(all(
        feature = "amqp-listener",
        any(feature = "http1", feature = "http1-native")
    ))]
    #[must_use]
    fn amqp(self, handler: proxima_amqp::AmqpPipeHandle) -> Self;

    /// Select memcached as the listen protocol, delegating to `.protocol(impl
    /// AnyProtocol)`.
    #[cfg(all(
        feature = "memcached-listener",
        any(feature = "http1", feature = "http1-native")
    ))]
    #[must_use]
    fn memcached(self, handler: proxima_memcached::MemcachedPipeHandle) -> Self;

    /// Select Redis/Valkey as the listen protocol, delegating to
    /// `.protocol(impl AnyProtocol)` — migrated off the old bespoke
    /// `redis_handler`/`redis_axis` fields (Section F of the builder-sugar
    /// design; its doc already noted no TLS conflict, unlike pgwire).
    #[cfg(all(
        feature = "redis-listener",
        any(feature = "http1", feature = "http1-native")
    ))]
    #[must_use]
    fn redis(self, handler: proxima_redis::RedisPipeHandle) -> Self;

    /// Select PostgreSQL wire protocol as the listen protocol — KEEPS its
    /// bespoke path (a fresh single-candidate `AnyListenProtocol` carrying
    /// the typed query engine, registered directly in `.serve()`), never
    /// `.protocol()`: the real reason is the TLS double-wrap guard
    /// (`.pgwire(query)` + `.tls(config)` hard-error together, since pgwire
    /// manages its own in-band `SSLRequest` upgrade) plus needing a FRESH
    /// registration per `query` handle every call, which `App::new()`'s
    /// static pre-registration can't provide. See
    /// [`ListenerBuilder::pgwire`](crate::listener::handle::ListenerBuilder::pgwire)'s
    /// own doc for the full reasoning.
    #[cfg(feature = "pgwire")]
    #[must_use]
    fn pgwire(self, query: proxima_pgwire::PgPipeHandle) -> Self;

    /// The one dual-transport protocol: branches on `.tcp()`/`.udp()` at
    /// `.serve()` time rather than delegating straight to `.protocol()`.
    /// `.tcp()` (default) resolves a single-candidate DNS-over-TCP
    /// `AnyListenProtocol`; `.udp()` resolves a
    /// `DatagramProtocolListenProtocol` wrapping `DnsDatagramProtocol`,
    /// self-registered the way the native h3 listener is; `.quic()` is a
    /// config error (DNS-over-QUIC/DoQ unimplemented). See
    /// [`ListenerBuilder::dns`](crate::listener::handle::ListenerBuilder::dns)'s
    /// own doc.
    #[cfg(feature = "dns-listener")]
    #[must_use]
    fn dns(self, handler: proxima_dns::DnsPipeHandle) -> Self;

    /// Wire a WebSocket (RFC 6455) handler into h1's existing
    /// `UpgradeHandler` seam — NOT a peer `AnyProtocol` candidate. Implies
    /// `.tcp()`. See
    /// [`ListenerBuilder::websocket`](crate::listener::handle::ListenerBuilder::websocket)'s
    /// own doc.
    #[cfg(all(
        feature = "websocket-upgrade",
        any(feature = "http1", feature = "http1-native")
    ))]
    #[must_use]
    fn websocket(self, handler: crate::listener::websocket::WebSocketHandler) -> Self;
}
