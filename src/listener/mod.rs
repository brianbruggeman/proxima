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
//! Both builders impl the SAME [`SpecBuilder`](crate::SpecBuilder) seam and
//! thereby get the SAME [`ProtocolSugar`](crate::ProtocolSugar) /
//! [`TransportSugar`](crate::TransportSugar) axes — no listener-specific
//! per-wire methods (`.h1()`/`.h3_native()` would fork the sugar instead of
//! mirroring it; the wire is picked by the shared `.tcp()`/`.tls()`/`.h3()`/
//! `.grpc()` axes, resolved to a concrete `ListenProtocol` by
//! `resolve_listen_protocol` — the listen-side mirror of `load.rs`'s
//! client-side factory dispatch). A few axes are honestly asymmetric and
//! shadow or extend the blanket method with an inherent one carrying more
//! than a client ever needs — a listener needs cert material (or a typed
//! query engine, for pgwire) a client never carries, and has no url to dial:
//!
//! | axis | client (`ClientBuilder`) | listener (`ListenerBuilder`) |
//! | --- | --- | --- |
//! | `.tcp()` / `.auto()` | real (`TransportSugar`) | real — resolves to the h1+h2 ALPN combiner (`"http"`) |
//! | `.tls()` | real, zero-arg (`TransportSugar`) — the dial url comes from a separately-chained `.http(url)` | shadowed: inherent `.tls(TlsConfig)` — real cert material required; resolves to the SAME `"http"` combiner (TLS is spec data, not a different protocol) |
//! | `.h3()` | real (`TransportSugar`) | real — resolves to `"h3-native"`, self-registered onto the fresh `App` (not in `App::new()`'s default set) |
//! | `.proxy(url)` | real | no listener meaning — `.serve()` hard-errors if present |
//! | `.http(url)` / `.https(url)` | real (dials the url) | real — carries the BIND address (`bind.to_string()`), read by `bind_from_spec` when `.bind(addr)` wasn't called directly |
//! | `.grpc(url)` / `.grpc()` | real, url-carrying | shadowed: inherent url-less `.grpc()` — listener dispatches to `.handle(pipe)`, not a url; resolves to `"h2"` (gRPC rides h2), self-registered like `.h3()` |
//! | (no client twin) | — | inherent `.h2()` — the other name for the same shared `"h2"` protocol |
//! | (no client twin) | — | inherent `.pgwire(query)` (feature `pgwire`) — carries a typed SQL query engine, self-registered fresh on every `.serve()` |
//!
//! `use proxima::TransportSugar` still brings `.tcp()`/`.auto()`/`.h3()` into
//! scope on `ListenerBuilder`. There is no separate listener DSL — one
//! `SpecBuilder` seam, one resolver mirroring `load.rs`, honestly asymmetric
//! axes only where a listener's inputs genuinely differ from a client's.
//!
//! ```ignore
//! use proxima::{Listener, ListenerBuilderEntry, TransportSugar, into_handle};
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

pub use handle::{ListenerBuilder, ListenerBuilderEntry};
