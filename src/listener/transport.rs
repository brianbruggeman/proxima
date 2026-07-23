//! Transport axis for [`ListenerBuilder`] — TYPE-SPECIFIC (no blanket impl
//! over every `SpecBuilder`, unlike the retired
//! `proxima_config::sugar::TransportSugar`). Picks the wire
//! [`resolve_listen_protocol`](crate::listener::handle::resolve_listen_protocol)
//! reads. There is no listener-side `.proxy()` — a listener has no upstream
//! to route through (see `reject_dead_axes`, which still hard-errors if a
//! caller reaches `.proxy()` through some other door).

use crate::listener::handle::ListenerBuilder;
use proxima_config::sugar::SpecBuilder;

/// `.tcp()` / `.udp()` / `.quic()` — the wire pick. Bring into scope with
/// `use proxima::ListenerTransportExt;` (or `proxima::prelude::*`).
///
/// These three compose with [`crate::ListenerProtocolExt`]'s protocol axis
/// (`.http()`/`.grpc()`/…) rather than standing alone — a listener always
/// picks BOTH "what wire" (this trait) and "what protocol on that wire"
/// (the other one). `.tcp()` is the default and rarely needs spelling out;
/// `.quic()` matters because it changes which listener actually gets
/// built — see [`crate::ListenerProtocolExt::http`]'s doc for the
/// `.http().quic()` => HTTP/3 composition this axis feeds.
pub trait ListenerTransportExt: Sized {
    /// The h1+h2 ALPN combiner (the default) — plaintext TCP.
    ///
    /// ```
    /// use proxima::{Listener, ListenerBuilderEntry, ListenerTransportExt, Request, Response, ProximaError};
    /// use proxima::pipe::into_handle;
    /// use proxima::SendPipe;
    /// use bytes::Bytes;
    /// use std::net::{Ipv4Addr, SocketAddr};
    ///
    /// struct Echo;
    /// impl SendPipe for Echo {
    ///     type In = Request<Bytes>;
    ///     type Out = Response<Bytes>;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    ///         Ok(Response::ok("ok"))
    ///     }
    /// }
    ///
    /// # #[proxima::main]
    /// # async fn main() -> Result<(), ProximaError> {
    /// let bind = SocketAddr::from((Ipv4Addr::LOCALHOST, 0));
    /// // `.tcp()` is implicit — this is the same listener as leaving it off.
    /// let server = Listener::http(bind).tcp().handle(into_handle(Echo)).serve().await?;
    /// server.stop();
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    fn tcp(self) -> Self;

    /// UDP — only meaningful paired with a dual-transport protocol axis
    /// (currently `.dns()`; see [`crate::ListenerProtocolExt::dns`]'s own
    /// doc for the worked `.dns().tcp()` vs `.dns().udp()` comparison and
    /// the branching this feeds). Every other protocol axis (`.http()`,
    /// `.grpc()`, `.kafka()`, …) is TCP-only; pairing `.udp()` with one of
    /// those is rejected at `.serve()` time with a named
    /// [`crate::ProximaError::Config`] — see
    /// [`crate::ListenerProtocolExt::kafka`]'s doc for that error text.
    #[must_use]
    fn udp(self) -> Self;

    /// HTTP/3 over QUIC — resolves to the native h3 `DatagramProtocol`
    /// listener (`resolve_listen_protocol`'s `transport == "quic"` branch).
    /// Only meaningful paired with `.http()`/`.https()` (h3 IS http-over-quic,
    /// there is no "grpc over quic" or "kafka over quic" wire) — see
    /// [`crate::ListenerProtocolExt::http`]'s doc for the worked
    /// `.http().quic()` composition and [`crate::ListenerProtocolExt::grpc`]'s
    /// doc for why `.grpc().quic()` is a config error instead.
    ///
    /// ```
    /// use proxima::{Listener, ListenerBuilderEntry, ListenerProtocolExt, ListenerTransportExt, Request, Response, ProximaError};
    /// use proxima::pipe::into_handle;
    /// use proxima::SendPipe;
    /// use bytes::Bytes;
    /// use std::net::{Ipv4Addr, SocketAddr};
    ///
    /// struct Echo;
    /// impl SendPipe for Echo {
    ///     type In = Request<Bytes>;
    ///     type Out = Response<Bytes>;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    ///         Ok(Response::ok("ok"))
    ///     }
    /// }
    ///
    /// # #[proxima::main]
    /// # async fn main() -> Result<(), ProximaError> {
    /// // pick a free UDP port up front (h3's datagram bind wants a fixed
    /// // address, not the OS-assigned-on-bind convenience TCP gets here).
    /// let probe = std::net::UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    /// let bind = probe.local_addr().unwrap();
    /// drop(probe);
    ///
    /// // `.http()` + `.quic()` together IS HTTP/3 — no separate "h3" method.
    /// // `dev_self_signed`/`dev_sans` ask the native h3 listener to mint a
    /// // throwaway dev certificate instead of requiring real cert material —
    /// // production wires real certs through `.tls(TlsConfig)` instead.
    /// let server = Listener::builder()
    ///     .bind(bind)
    ///     .http(bind.to_string())
    ///     .quic()
    ///     .spec("dev_self_signed", serde_json::json!(true))
    ///     .spec("dev_sans", serde_json::json!(["localhost"]))
    ///     .handle(into_handle(Echo))
    ///     .serve()
    ///     .await?;
    /// server.stop();
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    fn quic(self) -> Self;
}

impl ListenerTransportExt for ListenerBuilder {
    fn tcp(self) -> Self {
        self.set("transport", "tcp")
    }

    fn udp(self) -> Self {
        self.set("transport", "udp")
    }

    fn quic(self) -> Self {
        self.set("transport", "quic")
    }
}
