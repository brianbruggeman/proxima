//! Wave D Phase 3 primitives merge: `proxima-pipe` + `proxima-stream` +
//! `proxima-sync` + `proxima-transport` folded into one crate.
//!
//! Folding transport in here (rather than into stream first) dissolves the
//! former pipe<->stream<->transport dependency cycle (stream depended on pipe
//! for `BindAddr`/`PeerInfo`; pipe depended on transport for `Replay`/
//! `tap_complete` in `pipe::diff`/`pipe::retry`) into intra-crate module
//! references — `crate::pipe`, `crate::stream`, `crate::sync`,
//! `crate::transport` all resolve within this one crate, no Cargo dependency
//! edges between them.
//!
//! - [`pipe`] — the async `Pipe`/`SendPipe` form family, `Handler`
//!   (Request->Response), `SourcePipe` (Signal->()), `Body`, `HeaderList`,
//!   the telemetry/capture trait surfaces, `UpgradeHandler`, endpoint
//!   metadata. Plugin authors' primary surface.
//! - [`stream`] — sans-IO byte-stream traits (`StreamConnection`/
//!   `Listener`/`Upstream` over `futures::io`), shared by every wire-protocol
//!   stack (h1, h2, h3, quic, tls, websocket, framing-json).
//! - [`sync`] — runtime-agnostic concurrency primitives shaped like
//!   `tokio::sync` (`Mutex`, `Notify`, `mpsc`, `oneshot`, `watch`,
//!   `broadcast`, `blocking`, `task`, `shutdown`).
//! - [`transport`] — `Replay` (record-and-replay), the `tap_complete` taps,
//!   and the `GenericStream` seam, below [`pipe`].

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

// the test harness always links std; a no_std crate must name it explicitly
// so `sync::blocking::futex`'s tests (compiled only when `std` is off)
// resolve `std::`.
#[cfg(all(test, not(feature = "std")))]
extern crate std;

pub mod driver;
pub mod pipe;
pub mod stream;
pub mod sync;
pub mod transport;

pub use driver::block_on;
