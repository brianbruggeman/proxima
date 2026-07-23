//! Security axis for [`ClientBuilder`] — TYPE-SPECIFIC (no blanket impl).
//! The client-side twin of the listener's inherent `.tls(TlsConfig)`
//! (`src/listener/handle.rs:466`), which stays a bare inherent method with
//! no trait minted for it (it needs real cert material a client never
//! carries). A client's `.tls()` is a zero-arg ASSERTION over a dial url it
//! already carries via `.http(url)` — the wire must be TLS, not a silent
//! `auto`-negotiated fallback. See [`crate::load::canonical_http`]'s
//! `transport` forwarding and
//! `proxima_http::http1::prime_upstream::build_prime_upstream`'s
//! `transport == "tls"` + `http://` scheme rejection for where this is
//! actually enforced (the bug fix in Section B of the design).

use crate::client::handle::ClientBuilder;
use proxima_config::sugar::SpecBuilder;

/// `.tls()` — assert the wire must be TLS. Bring into scope with
/// `use proxima::ClientSecurityExt;` (or `proxima::prelude::*`).
pub trait ClientSecurityExt: Sized {
    /// TLS over TCP (h1/h2 by ALPN) — writes the `transport` spec key to
    /// `"tls"`. Combining this with an `http://` (non-`https`) dial url is a
    /// config error at build time (`build_prime_upstream`), never a silent
    /// plaintext downgrade.
    #[must_use]
    fn tls(self) -> Self;
}

impl ClientSecurityExt for ClientBuilder {
    fn tls(self) -> Self {
        self.set("transport", "tls")
    }
}
