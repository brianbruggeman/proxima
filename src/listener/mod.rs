//! The serve-side peer of [`Client`](crate::Client) — same [`SpecBuilder`]
//! coin, opposite face. `Client::builder()` accumulates a spec and resolves
//! it via `load(Spec)` into a dialing `PipeHandle`; `Listener::builder()`
//! accumulates a spec and resolves it via the [`ListenRegistry`](crate::ListenRegistry)
//! (through [`App`](crate::App)) into a serving [`Server`](crate::server::Server).
//!
//! `Listener` itself is [`proxima_listen::handle::Listener`] — the
//! pre-existing, real bind/protocol/spec/dispatch/shutdown carrier
//! (`proxima-listen/src/handle.rs`, produced by `ListenerSpec::attach(dispatch)`,
//! run via `Listener::run_with_runtime`). This module does NOT define a
//! second `Listener` type. `Listener::builder()` / `Listener::http(bind)`
//! reach [`ListenerBuilder`] through the [`ListenerBuilderEntry`] trait
//! defined in [`handle`] — a foreign-type extension, not a peer type, since
//! Rust's orphan rule forbids this crate from adding an inherent method to a
//! type it doesn't own. Import the trait alongside `Listener` to unlock the
//! static methods: `use proxima::{Listener, ListenerBuilderEntry};`.
//!
//! Both builders impl the SAME [`SpecBuilder`](crate::SpecBuilder) seam, but
//! each gets its OWN type-specific axis extension traits — no blanket impl
//! over every `SpecBuilder` (the retired `proxima_config::sugar::{ProtocolSugar,
//! TransportSugar}` were exactly that, and are gone). Transport
//! ([`ListenerTransportExt`] / `ClientTransportExt`) and protocol
//! ([`ListenerProtocolExt`] / `ClientProtocolExt`) are separate traits per
//! type; security (`.tls(TlsConfig)`) stays a bare inherent method on
//! `ListenerBuilder` — real cert material, no trait minted for it. A few
//! axes are honestly asymmetric between the two builders — a listener needs
//! cert material (or a typed query engine, for pgwire) a client never
//! carries, and has no url to dial:
//!
//! | axis | client (`ClientBuilder`) | listener (`ListenerBuilder`) |
//! | --- | --- | --- |
//! | `.tcp()` / `.udp()` / `.quic()` | `ClientTransportExt` | [`ListenerTransportExt`] — `.quic()` resolves to the native h3 `DatagramProtocol` listener |
//! | `.tls()` | `ClientSecurityExt`, zero-arg — the dial url comes from a separately-chained `.http(url)` | inherent `.tls(TlsConfig)` — real cert material required; composes as a decorator over whatever `resolve_listen_protocol` resolves |
//! | `.proxy(url)` | `ClientTransportExt` | no listener meaning — `.serve()` hard-errors if present |
//! | `.http(url)` / `.https(url)` | real (dials the url) | real — carries the BIND address (`bind.to_string()`), read by `bind_from_spec` when `.bind(addr)` wasn't called directly |
//! | `.grpc(url)` / `.grpc()` | url-carrying | url-less — listener dispatches to `.handle(pipe)`, not a url; resolves to `"h2"` (gRPC rides h2) |
//! | `.kafka()`/`.mqtt()`/`.amqp()`/`.memcached()`/`.redis()` | DSN, delegates to `.protocol()` | typed handle, delegates to `.protocol()` |
//! | `.pgwire()` | DSN, delegates to `.protocol()` | typed query engine — KEEPS its bespoke fresh-registration path (TLS double-wrap guard) |
//! | `.dns()` | DSN, delegates to `.protocol()` | the one dual-transport axis — branches on `.tcp()`/`.udp()` at `.serve()` time |
//! | (no client twin) | — | `.websocket(handler)` — wires into h1's Upgrade seam, not a peer `AnyProtocol` |
//! | (no client twin) | — | inherent `.h2()` — the other name for the same shared `"h2"` protocol |
//!
//! `use proxima::{ListenerTransportExt, ListenerProtocolExt};` (or
//! `proxima::prelude::*`) brings the listener's axes into scope. There is no
//! separate listener DSL — one `SpecBuilder` seam, one resolver mirroring
//! `load.rs`, honestly asymmetric axes only where a listener's inputs
//! genuinely differ from a client's.
//!
//! ```ignore
//! use proxima::prelude::*;
//! use proxima::into_handle;
//!
//! let server = Listener::builder()
//!     .bind("127.0.0.1:8080".parse()?)
//!     .tcp()
//!     .handle(into_handle(my_pipe))
//!     .serve()
//!     .await?;
//! server.run_until_signal().await;
//!
//! // mirrors `Client::http(url)`:
//! let server = Listener::http("127.0.0.1:8080".parse()?)
//!     .handle(into_handle(my_pipe))
//!     .serve()
//!     .await?;
//! ```

pub mod handle;
pub mod protocol;
pub mod transport;
#[cfg(all(
    feature = "websocket-upgrade",
    any(feature = "http1", feature = "http1-native")
))]
pub mod websocket;

pub use handle::{ListenerBuilder, ListenerBuilderEntry};
pub use protocol::ListenerProtocolExt;
pub use transport::ListenerTransportExt;
#[cfg(all(
    feature = "websocket-upgrade",
    any(feature = "http1", feature = "http1-native")
))]
pub use websocket::WebSocketHandler;
