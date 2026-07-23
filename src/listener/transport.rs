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
pub trait ListenerTransportExt: Sized {
    /// The h1+h2 ALPN combiner (the default) — plaintext TCP.
    #[must_use]
    fn tcp(self) -> Self;

    /// UDP — only meaningful paired with a dual-transport protocol axis
    /// (currently `.dns()`; see its own doc for the branching this feeds).
    #[must_use]
    fn udp(self) -> Self;

    /// HTTP/3 over QUIC — resolves to the native h3 `DatagramProtocol`
    /// listener (`resolve_listen_protocol`'s `transport == "quic"` branch).
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
