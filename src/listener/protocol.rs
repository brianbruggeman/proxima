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
    ///
    /// ```
    /// use proxima::{Listener, ListenerBuilderEntry, ListenerProtocolExt, Request, Response, ProximaError};
    /// use proxima::pipe::into_handle;
    /// use proxima::SendPipe;
    /// use bytes::Bytes;
    /// use std::net::{Ipv4Addr, SocketAddr};
    ///
    /// struct Hello;
    /// impl SendPipe for Hello {
    ///     type In = Request<Bytes>;
    ///     type Out = Response<Bytes>;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    ///         Ok(Response::ok("hello, proxima"))
    ///     }
    /// }
    ///
    /// # #[proxima::main]
    /// # async fn main() -> Result<(), ProximaError> {
    /// let bind = SocketAddr::from((Ipv4Addr::LOCALHOST, 0));
    /// // `Listener::builder()` + `.http(bind.to_string())` is what
    /// // `Listener::http(bind)` (the one-liner entry point) does for you.
    /// let server = Listener::builder()
    ///     .bind(bind)
    ///     .http(bind.to_string())
    ///     .handle(into_handle(Hello))
    ///     .serve()
    ///     .await?;
    /// server.stop();
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    fn http(self, bind: impl Into<String>) -> Self;

    /// Same as [`Self::http`] — TLS falls out of `.tls(config)`, not the
    /// scheme, on the listener side (a client's `.https(url)` instead reads
    /// the scheme out of its dial url; a listener has no url to read a
    /// scheme from, so `.tls(TlsConfig)` is the only on/off switch). See
    /// [`crate::listener::handle::ListenerBuilder::tls`] for cert material.
    ///
    /// ```
    /// use proxima::{Listener, ListenerBuilderEntry, ListenerProtocolExt, Request, Response, ProximaError};
    /// use proxima::pipe::into_handle;
    /// use proxima::SendPipe;
    /// use bytes::Bytes;
    /// use std::net::{Ipv4Addr, SocketAddr};
    ///
    /// struct Hello;
    /// impl SendPipe for Hello {
    ///     type In = Request<Bytes>;
    ///     type Out = Response<Bytes>;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    ///         Ok(Response::ok("hello, proxima"))
    ///     }
    /// }
    ///
    /// # #[proxima::main]
    /// # async fn main() -> Result<(), ProximaError> {
    /// let bind = SocketAddr::from((Ipv4Addr::LOCALHOST, 0));
    /// // `.https(..)` writes the exact same spec key `.http(..)` does — plain
    /// // TCP here, since no `.tls(config)` was attached. Pair it with
    /// // `.tls(TlsConfig)` to actually terminate TLS.
    /// let server = Listener::builder()
    ///     .bind(bind)
    ///     .https(bind.to_string())
    ///     .handle(into_handle(Hello))
    ///     .serve()
    ///     .await?;
    /// server.stop();
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    fn https(self, bind: impl Into<String>) -> Self;

    /// Select gRPC as the listen protocol (rides h2) — url-less, since a
    /// listener dispatches to `.handle(pipe)` rather than dialing out.
    ///
    /// gRPC rides h2, never QUIC — combining `.grpc()` with
    /// [`crate::ListenerTransportExt::quic`] is rejected at `.serve()` with a
    /// named [`crate::ProximaError::Config`] instead of silently picking one
    /// axis over the other:
    ///
    /// ```
    /// use proxima::{Listener, ListenerBuilderEntry, ListenerProtocolExt, ListenerTransportExt, Request, Response, ProximaError};
    /// use proxima::pipe::into_handle;
    /// use proxima::SendPipe;
    /// use bytes::Bytes;
    /// use std::net::{Ipv4Addr, SocketAddr};
    ///
    /// struct Hello;
    /// impl SendPipe for Hello {
    ///     type In = Request<Bytes>;
    ///     type Out = Response<Bytes>;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    ///         Ok(Response::ok("hello"))
    ///     }
    /// }
    ///
    /// # #[proxima::main]
    /// # async fn main() -> Result<(), ProximaError> {
    /// let bind = SocketAddr::from((Ipv4Addr::LOCALHOST, 0));
    /// let outcome = Listener::builder()
    ///     .bind(bind)
    ///     .grpc()
    ///     .quic()
    ///     .handle(into_handle(Hello))
    ///     .serve()
    ///     .await;
    /// assert!(matches!(outcome, Err(ProximaError::Config(_))));
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Dropping `.quic()` (the default is `.tcp()`) serves it for real:
    ///
    /// ```
    /// use proxima::{Listener, ListenerBuilderEntry, ListenerProtocolExt, Request, Response, ProximaError};
    /// use proxima::pipe::into_handle;
    /// use proxima::SendPipe;
    /// use bytes::Bytes;
    /// use std::net::{Ipv4Addr, SocketAddr};
    ///
    /// struct Hello;
    /// impl SendPipe for Hello {
    ///     type In = Request<Bytes>;
    ///     type Out = Response<Bytes>;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    ///         Ok(Response::ok("hello"))
    ///     }
    /// }
    ///
    /// # #[proxima::main]
    /// # async fn main() -> Result<(), ProximaError> {
    /// let bind = SocketAddr::from((Ipv4Addr::LOCALHOST, 0));
    /// let server = Listener::builder()
    ///     .bind(bind)
    ///     .grpc()
    ///     .handle(into_handle(Hello))
    ///     .serve()
    ///     .await?;
    /// server.stop();
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    fn grpc(self) -> Self;

    /// Select Kafka as the listen protocol, delegating to `.protocol(impl
    /// AnyProtocol)`.
    ///
    /// Every protocol axis below `.grpc()` is TCP-only (each one's
    /// `AnyProtocol::drive` takes `Box<dyn StreamConnection>`, a byte
    /// stream — there is no "kafka over QUIC" wire this facade speaks).
    /// Pairing one with `.quic()` is rejected at `.serve()` with a named
    /// [`crate::ProximaError::Config`], not a silent fallback to TCP:
    ///
    /// ```
    /// use proxima::{Listener, ListenerBuilderEntry, ListenerProtocolExt, ListenerTransportExt, Request, Response, ProximaError};
    /// use proxima::pipe::into_handle;
    /// use proxima::SendPipe;
    /// use bytes::Bytes;
    /// use proxima_kafka::{RequestBody, ResponseBody, into_kafka_handle};
    ///
    /// // No client ever dials in this doctest, so neither handler body
    /// // runs — each only has to satisfy its typed contract.
    /// struct Unimplemented;
    /// impl SendPipe for Unimplemented {
    ///     type In = RequestBody;
    ///     type Out = ResponseBody;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: RequestBody) -> Result<ResponseBody, ProximaError> {
    ///         unreachable!("no client connects in this doctest")
    ///     }
    /// }
    ///
    /// struct Dispatch; // `.handle(pipe)` is still required before `.serve()`.
    /// impl SendPipe for Dispatch {
    ///     type In = Request<Bytes>;
    ///     type Out = Response<Bytes>;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    ///         unreachable!("rejected before any accept loop starts")
    ///     }
    /// }
    ///
    /// # #[proxima::main]
    /// # async fn main() -> Result<(), ProximaError> {
    /// let bind = std::net::SocketAddr::from(([127, 0, 0, 1], 0));
    /// let outcome = Listener::builder()
    ///     .bind(bind)
    ///     .kafka(into_kafka_handle(Unimplemented))
    ///     .quic()
    ///     .handle(into_handle(Dispatch))
    ///     .serve()
    ///     .await;
    /// assert!(matches!(outcome, Err(ProximaError::Config(message)) if message.contains("TCP-only")));
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Dropping `.quic()` (or spelling `.tcp()`, the default) actually binds:
    ///
    /// ```
    /// use proxima::{Listener, ListenerBuilderEntry, ListenerProtocolExt, Request, Response, ProximaError};
    /// use proxima::pipe::into_handle;
    /// use proxima::SendPipe;
    /// use bytes::Bytes;
    /// use proxima_kafka::{RequestBody, ResponseBody, into_kafka_handle};
    ///
    /// struct Unimplemented;
    /// impl SendPipe for Unimplemented {
    ///     type In = RequestBody;
    ///     type Out = ResponseBody;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: RequestBody) -> Result<ResponseBody, ProximaError> {
    ///         unreachable!("no client connects in this doctest")
    ///     }
    /// }
    ///
    /// struct Dispatch; // `.handle(pipe)` is still required even though
    ///                  // `.kafka(handler)` carries its own engine — see the
    ///                  // module doc: `.kafka()` delegates to `.protocol()`.
    /// impl SendPipe for Dispatch {
    ///     type In = Request<Bytes>;
    ///     type Out = Response<Bytes>;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    ///         Ok(Response::new(404))
    ///     }
    /// }
    ///
    /// # #[proxima::main]
    /// # async fn main() -> Result<(), ProximaError> {
    /// let bind = std::net::SocketAddr::from(([127, 0, 0, 1], 0));
    /// let server = Listener::builder()
    ///     .bind(bind)
    ///     .kafka(into_kafka_handle(Unimplemented))
    ///     .handle(into_handle(Dispatch))
    ///     .serve()
    ///     .await?;
    /// server.stop();
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(all(
        feature = "kafka-listener",
        any(feature = "http1", feature = "http1-native")
    ))]
    #[must_use]
    fn kafka(self, handler: proxima_kafka::KafkaPipeHandle) -> Self;

    /// Select MQTT as the listen protocol, delegating to `.protocol(impl
    /// AnyProtocol)`.
    ///
    /// ```
    /// use proxima::{Listener, ListenerBuilderEntry, ListenerProtocolExt, Request, Response, ProximaError};
    /// use proxima::pipe::into_handle;
    /// use proxima::SendPipe;
    /// use bytes::Bytes;
    /// use proxima_mqtt::{MqttPipeRequest, MqttPipeReply, into_mqtt_handle};
    ///
    /// struct Unimplemented;
    /// impl SendPipe for Unimplemented {
    ///     type In = MqttPipeRequest;
    ///     type Out = MqttPipeReply;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: MqttPipeRequest) -> Result<MqttPipeReply, ProximaError> {
    ///         unreachable!("no client connects in this doctest")
    ///     }
    /// }
    ///
    /// struct Dispatch;
    /// impl SendPipe for Dispatch {
    ///     type In = Request<Bytes>;
    ///     type Out = Response<Bytes>;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    ///         Ok(Response::new(404))
    ///     }
    /// }
    ///
    /// # #[proxima::main]
    /// # async fn main() -> Result<(), ProximaError> {
    /// let bind = std::net::SocketAddr::from(([127, 0, 0, 1], 0));
    /// let server = Listener::builder()
    ///     .bind(bind)
    ///     .mqtt(into_mqtt_handle(Unimplemented))
    ///     .handle(into_handle(Dispatch))
    ///     .serve()
    ///     .await?;
    /// server.stop();
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(all(
        feature = "mqtt-listener",
        any(feature = "http1", feature = "http1-native")
    ))]
    #[must_use]
    fn mqtt(self, handler: proxima_mqtt::MqttPipeHandle) -> Self;

    /// Select AMQP as the listen protocol, delegating to `.protocol(impl
    /// AnyProtocol)`.
    ///
    /// ```
    /// use proxima::{Listener, ListenerBuilderEntry, ListenerProtocolExt, Request, Response, ProximaError};
    /// use proxima::pipe::into_handle;
    /// use proxima::SendPipe;
    /// use bytes::Bytes;
    /// use proxima_amqp::{AmqpPipeRequest, AmqpPipeReply, into_amqp_handle};
    ///
    /// struct Unimplemented;
    /// impl SendPipe for Unimplemented {
    ///     type In = AmqpPipeRequest;
    ///     type Out = AmqpPipeReply;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: AmqpPipeRequest) -> Result<AmqpPipeReply, ProximaError> {
    ///         unreachable!("no client connects in this doctest")
    ///     }
    /// }
    ///
    /// struct Dispatch;
    /// impl SendPipe for Dispatch {
    ///     type In = Request<Bytes>;
    ///     type Out = Response<Bytes>;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    ///         Ok(Response::new(404))
    ///     }
    /// }
    ///
    /// # #[proxima::main]
    /// # async fn main() -> Result<(), ProximaError> {
    /// let bind = std::net::SocketAddr::from(([127, 0, 0, 1], 0));
    /// let server = Listener::builder()
    ///     .bind(bind)
    ///     .amqp(into_amqp_handle(Unimplemented))
    ///     .handle(into_handle(Dispatch))
    ///     .serve()
    ///     .await?;
    /// server.stop();
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(all(
        feature = "amqp-listener",
        any(feature = "http1", feature = "http1-native")
    ))]
    #[must_use]
    fn amqp(self, handler: proxima_amqp::AmqpPipeHandle) -> Self;

    /// Select memcached as the listen protocol, delegating to `.protocol(impl
    /// AnyProtocol)`.
    ///
    /// ```
    /// use proxima::{Listener, ListenerBuilderEntry, ListenerProtocolExt, Request, Response, ProximaError};
    /// use proxima::pipe::into_handle;
    /// use proxima::SendPipe;
    /// use bytes::Bytes;
    /// use proxima_memcached::{MemcachedRequest, Reply, into_memcached_handle};
    ///
    /// struct Unimplemented;
    /// impl SendPipe for Unimplemented {
    ///     type In = MemcachedRequest;
    ///     type Out = Reply;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: Self::In) -> Result<Self::Out, ProximaError> {
    ///         unreachable!("no client connects in this doctest")
    ///     }
    /// }
    ///
    /// struct Dispatch;
    /// impl SendPipe for Dispatch {
    ///     type In = Request<Bytes>;
    ///     type Out = Response<Bytes>;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    ///         Ok(Response::new(404))
    ///     }
    /// }
    ///
    /// # #[proxima::main]
    /// # async fn main() -> Result<(), ProximaError> {
    /// let bind = std::net::SocketAddr::from(([127, 0, 0, 1], 0));
    /// let server = Listener::builder()
    ///     .bind(bind)
    ///     .memcached(into_memcached_handle(Unimplemented))
    ///     .handle(into_handle(Dispatch))
    ///     .serve()
    ///     .await?;
    /// server.stop();
    /// # Ok(())
    /// # }
    /// ```
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
    ///
    /// ```
    /// use proxima::{Listener, ListenerBuilderEntry, ListenerProtocolExt, Request, Response, ProximaError};
    /// use proxima::pipe::into_handle;
    /// use proxima::SendPipe;
    /// use bytes::Bytes;
    /// use proxima_redis::{RedisRequest, into_redis_handle};
    /// use proxima_redis::RespValue;
    ///
    /// struct Unimplemented;
    /// impl SendPipe for Unimplemented {
    ///     type In = RedisRequest;
    ///     type Out = RespValue;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: RedisRequest) -> Result<RespValue, ProximaError> {
    ///         unreachable!("no client connects in this doctest")
    ///     }
    /// }
    ///
    /// struct Dispatch;
    /// impl SendPipe for Dispatch {
    ///     type In = Request<Bytes>;
    ///     type Out = Response<Bytes>;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    ///         Ok(Response::new(404))
    ///     }
    /// }
    ///
    /// # #[proxima::main]
    /// # async fn main() -> Result<(), ProximaError> {
    /// let bind = std::net::SocketAddr::from(([127, 0, 0, 1], 0));
    /// let server = Listener::builder()
    ///     .bind(bind)
    ///     .redis(into_redis_handle(Unimplemented))
    ///     .handle(into_handle(Dispatch))
    ///     .serve()
    ///     .await?;
    /// server.stop();
    /// # Ok(())
    /// # }
    /// ```
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
    ///
    /// ```
    /// use proxima::{Listener, ListenerBuilderEntry, ListenerProtocolExt, Request, Response, ProximaError};
    /// use proxima::pipe::into_handle;
    /// use proxima::SendPipe;
    /// use bytes::Bytes;
    /// use proxima_pgwire::{PgReply, QueryRequest, into_pg_handle};
    ///
    /// struct Unimplemented;
    /// impl SendPipe for Unimplemented {
    ///     type In = QueryRequest;
    ///     type Out = PgReply;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: QueryRequest) -> Result<PgReply, ProximaError> {
    ///         unreachable!("no client connects in this doctest")
    ///     }
    /// }
    ///
    /// struct Dispatch; // still required — pgwire carries its own engine,
    ///                  // but `.handle(pipe)` is the one input every
    ///                  // `ListenerBuilder` needs before `.serve()`.
    /// impl SendPipe for Dispatch {
    ///     type In = Request<Bytes>;
    ///     type Out = Response<Bytes>;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    ///         Ok(Response::new(404))
    ///     }
    /// }
    ///
    /// # #[proxima::main]
    /// # async fn main() -> Result<(), ProximaError> {
    /// let bind = std::net::SocketAddr::from(([127, 0, 0, 1], 0));
    /// let server = Listener::builder()
    ///     .bind(bind)
    ///     .pgwire(into_pg_handle(Unimplemented))
    ///     .handle(into_handle(Dispatch))
    ///     .serve()
    ///     .await?;
    /// server.stop();
    /// # Ok(())
    /// # }
    /// ```
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
    ///
    /// `.dns()` alone (or paired with `.tcp()`) is DNS-over-TCP (RFC 1035
    /// §4.2.2, a 2-byte length prefix over a byte stream):
    ///
    /// ```
    /// use proxima::{Listener, ListenerBuilderEntry, ListenerProtocolExt, Request, Response, ProximaError};
    /// use proxima::pipe::into_handle;
    /// use proxima::SendPipe;
    /// use bytes::Bytes;
    /// use proxima_dns::{DnsPipeRequest, DnsPipeReply, into_dns_handle};
    ///
    /// struct Unimplemented;
    /// impl SendPipe for Unimplemented {
    ///     type In = DnsPipeRequest;
    ///     type Out = DnsPipeReply;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: DnsPipeRequest) -> Result<DnsPipeReply, ProximaError> {
    ///         unreachable!("no client connects in this doctest")
    ///     }
    /// }
    ///
    /// struct Dispatch;
    /// impl SendPipe for Dispatch {
    ///     type In = Request<Bytes>;
    ///     type Out = Response<Bytes>;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    ///         Ok(Response::new(404))
    ///     }
    /// }
    ///
    /// # #[proxima::main]
    /// # async fn main() -> Result<(), ProximaError> {
    /// let bind = std::net::SocketAddr::from(([127, 0, 0, 1], 0));
    /// let server = Listener::builder()
    ///     .bind(bind)
    ///     .dns(into_dns_handle(Unimplemented))
    ///     .handle(into_handle(Dispatch))
    ///     .serve()
    ///     .await?;
    /// server.stop();
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// `.dns().udp()` is the classic UDP resolver wire instead — same
    /// handler, a completely different `ListenProtocol` underneath:
    ///
    /// ```
    /// use proxima::{Listener, ListenerBuilderEntry, ListenerProtocolExt, ListenerTransportExt, Request, Response, ProximaError};
    /// use proxima::pipe::into_handle;
    /// use proxima::SendPipe;
    /// use bytes::Bytes;
    /// use proxima_dns::{DnsPipeRequest, DnsPipeReply, into_dns_handle};
    ///
    /// struct Unimplemented;
    /// impl SendPipe for Unimplemented {
    ///     type In = DnsPipeRequest;
    ///     type Out = DnsPipeReply;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: DnsPipeRequest) -> Result<DnsPipeReply, ProximaError> {
    ///         unreachable!("no client connects in this doctest")
    ///     }
    /// }
    ///
    /// struct Dispatch;
    /// impl SendPipe for Dispatch {
    ///     type In = Request<Bytes>;
    ///     type Out = Response<Bytes>;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    ///         Ok(Response::new(404))
    ///     }
    /// }
    ///
    /// # #[proxima::main]
    /// # async fn main() -> Result<(), ProximaError> {
    /// let bind = std::net::SocketAddr::from(([127, 0, 0, 1], 0));
    /// let server = Listener::builder()
    ///     .bind(bind)
    ///     .dns(into_dns_handle(Unimplemented))
    ///     .udp()
    ///     .handle(into_handle(Dispatch))
    ///     .serve()
    ///     .await?;
    /// server.stop();
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// `.dns().quic()` (DNS-over-QUIC / DoQ) is unimplemented — a named
    /// config error rather than silently falling back to TCP:
    ///
    /// ```
    /// use proxima::{Listener, ListenerBuilderEntry, ListenerProtocolExt, ListenerTransportExt, Request, Response, ProximaError};
    /// use proxima::pipe::into_handle;
    /// use proxima::SendPipe;
    /// use bytes::Bytes;
    /// use proxima_dns::{DnsPipeRequest, DnsPipeReply, into_dns_handle};
    ///
    /// struct Unimplemented;
    /// impl SendPipe for Unimplemented {
    ///     type In = DnsPipeRequest;
    ///     type Out = DnsPipeReply;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: DnsPipeRequest) -> Result<DnsPipeReply, ProximaError> {
    ///         unreachable!("no client connects in this doctest")
    ///     }
    /// }
    ///
    /// struct Dispatch;
    /// impl SendPipe for Dispatch {
    ///     type In = Request<Bytes>;
    ///     type Out = Response<Bytes>;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    ///         unreachable!("rejected before any accept loop starts")
    ///     }
    /// }
    ///
    /// # #[proxima::main]
    /// # async fn main() -> Result<(), ProximaError> {
    /// let bind = std::net::SocketAddr::from(([127, 0, 0, 1], 0));
    /// let outcome = Listener::builder()
    ///     .bind(bind)
    ///     .dns(into_dns_handle(Unimplemented))
    ///     .quic()
    ///     .handle(into_handle(Dispatch))
    ///     .serve()
    ///     .await;
    /// assert!(matches!(outcome, Err(ProximaError::Config(message)) if message.contains("DoQ")));
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "dns-listener")]
    #[must_use]
    fn dns(self, handler: proxima_dns::DnsPipeHandle) -> Self;

    /// Wire a WebSocket (RFC 6455) handler into h1's existing
    /// `UpgradeHandler` seam — NOT a peer `AnyProtocol` candidate. Implies
    /// `.tcp()`. See
    /// [`ListenerBuilder::websocket`](crate::listener::handle::ListenerBuilder::websocket)'s
    /// own doc.
    ///
    /// ```
    /// use proxima::{Listener, ListenerBuilderEntry, ListenerProtocolExt, Request, Response, ProximaError};
    /// use proxima::pipe::into_handle;
    /// use proxima::SendPipe;
    /// use proxima::upgrade::{HijackedSocket, UpgradeFuture};
    /// use bytes::Bytes;
    /// use std::sync::Arc;
    ///
    /// struct Dispatch;
    /// impl SendPipe for Dispatch {
    ///     type In = Request<Bytes>;
    ///     type Out = Response<Bytes>;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    ///         Ok(Response::new(404))
    ///     }
    /// }
    ///
    /// // Called once per accepted WebSocket handshake with the raw
    /// // post-101 socket — frame parsing is the caller's own business
    /// // (`proxima_protocols::websocket_frame`, behind `websocket-frame`).
    /// fn on_upgrade(_socket: HijackedSocket) -> UpgradeFuture {
    ///     Box::pin(async { Ok(()) })
    /// }
    ///
    /// # #[proxima::main]
    /// # async fn main() -> Result<(), ProximaError> {
    /// let bind = std::net::SocketAddr::from(([127, 0, 0, 1], 0));
    /// let server = Listener::builder()
    ///     .bind(bind)
    ///     .http(bind.to_string())
    ///     .websocket(Arc::new(on_upgrade))
    ///     .handle(into_handle(Dispatch))
    ///     .serve()
    ///     .await?;
    /// server.stop();
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(all(
        feature = "websocket-upgrade",
        any(feature = "http1", feature = "http1-native")
    ))]
    #[must_use]
    fn websocket(self, handler: crate::listener::websocket::WebSocketHandler) -> Self;
}
