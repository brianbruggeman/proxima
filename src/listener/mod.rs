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
//! second `Listener` type. `Listener::builder()` reaches [`ListenerBuilder`]
//! through the [`ListenerBuilderEntry`] trait
//! defined in [`handle`] — a foreign-type extension, not a peer type, since
//! Rust's orphan rule forbids this crate from adding an inherent method to a
//! type it doesn't own. Import the trait alongside `Listener` to unlock the
//! static method: `use proxima::{Listener, ListenerBuilderEntry};`.
//!
//! Both builders impl the one [`SpecBuilder`](crate::SpecBuilder) seam, but
//! they are NOT symmetric in which axes actually do something — a listener
//! needs cert material a client never carries, and has no listener-side
//! `.h3()`/`.proxy()` wiring at all today:
//!
//! | axis | client (`ClientBuilder`) | listener (`ListenerBuilder`) |
//! | --- | --- | --- |
//! | `.tcp()` | real (`TransportSugar`) | real (`TransportSugar`) |
//! | `.tls()` | real, url-carrying (`TransportSugar`) | shadowed: inherent `.tls(TlsConfig)` — real cert material required |
//! | `.h3()` / `.proxy(url)` | real / real | no listener wiring — `.serve()` hard-errors if requested |
//! | `.http(url)` / `.https(url)` | real (dials the url) | inert marker if called — `.http()` needs no call, it's the implicit default |
//! | `.grpc(url)` / `.grpc()` | real, url-carrying | shadowed: inherent url-less `.grpc()` — listener dispatches to `.handle(pipe)`, not a url |
//!
//! `use proxima::TransportSugar` still brings `.tcp()`/`.auto()` into scope on
//! `ListenerBuilder`. There is no separate listener DSL — one `SpecBuilder`
//! seam, honestly asymmetric axes.
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
//! ```

pub mod handle;

pub use handle::{ListenerBuilder, ListenerBuilderEntry};
