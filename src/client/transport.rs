//! Transport axis for [`ClientBuilder`] — TYPE-SPECIFIC (no blanket impl over
//! every `SpecBuilder`, unlike the retired `proxima_config::sugar::TransportSugar`).
//! Picks the wire under the app protocol (`.http`/`.grpc`/…) and the egress
//! route. Lowers to the `transport` / `proxy` spec keys [`crate::load`]'s
//! factory dispatch reads.

use crate::client::handle::ClientBuilder;
use proxima_config::sugar::SpecBuilder;

/// `.tcp()` / `.udp()` / `.quic()` (transport pick) + `.proxy(url)` (egress
/// routing, client-only — a listener has no upstream to route through, so
/// this has no listener-side twin). Bring into scope with
/// `use proxima::ClientTransportExt;` (or `proxima::prelude::*`).
pub trait ClientTransportExt: Sized {
    /// Force plaintext TCP (no TLS, no QUIC).
    ///
    /// ```
    /// use proxima::{Client, ClientProtocolExt, ClientTransportExt};
    ///
    /// let client = Client::builder().http("http://localhost:8080").tcp().build()?;
    /// # Ok::<(), proxima::ProximaError>(())
    /// ```
    #[must_use]
    fn tcp(self) -> Self;

    /// Force UDP (currently meaningful only paired with a protocol whose
    /// wire is UDP-shaped — e.g. `.dns()`; the `http`/`grpc` factories dial
    /// TCP regardless).
    ///
    /// ```
    /// use proxima::{Client, ClientTransportExt};
    /// # use proxima::ClientProtocolExt;
    ///
    /// // `.dns()` needs the `dns-client` feature (default build has it off);
    /// // gated here so this example still compiles without it.
    /// # #[cfg(all(feature = "dns-client", any(target_os = "linux", target_os = "macos")))]
    /// let client = Client::builder().dns("dns://1.1.1.1:53").udp().build()?;
    /// # Ok::<(), proxima::ProximaError>(())
    /// ```
    #[must_use]
    fn udp(self) -> Self;

    /// Force HTTP/3 over QUIC — dispatches through the native h3 upstream
    /// (`h3-native`) instead of the h1/h2 prime client. See
    /// [`crate::load::canonical_h3`] for the field-forwarding contract.
    ///
    /// ```
    /// use proxima::{Client, ClientProtocolExt, ClientTransportExt};
    ///
    /// // `.http()` + `.quic()` together is the client-side twin of the
    /// // listener's `.http().quic()` => HTTP/3 composition.
    /// let client = Client::builder().http("https://localhost:8443").quic().build()?;
    /// # Ok::<(), proxima::ProximaError>(())
    /// ```
    #[must_use]
    fn quic(self) -> Self;

    /// Route egress through an HTTP proxy (the `proxy` key) — a CONNECT
    /// tunnel dialed before the upstream.
    ///
    /// ```
    /// use proxima::{Client, ClientProtocolExt, ClientTransportExt};
    ///
    /// let client = Client::builder()
    ///     .http("http://localhost:8080")
    ///     .proxy("http://proxy.example.internal:3128")
    ///     .build()?;
    /// # Ok::<(), proxima::ProximaError>(())
    /// ```
    #[must_use]
    fn proxy(self, url: impl Into<String>) -> Self;
}

impl ClientTransportExt for ClientBuilder {
    fn tcp(self) -> Self {
        self.set("transport", "tcp")
    }

    fn udp(self) -> Self {
        self.set("transport", "udp")
    }

    fn quic(self) -> Self {
        self.set("transport", "quic")
    }

    fn proxy(self, url: impl Into<String>) -> Self {
        self.set("proxy", url.into())
    }
}

// Spec-shape assertions live in `handle.rs`'s test module, alongside the
// rest of the builder-axis parity tests — they need `Client::builder()..
// .build()` then a look at the private `Inner::spec`, which is only
// visible inside `handle`'s own module.
