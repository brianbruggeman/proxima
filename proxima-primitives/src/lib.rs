//! The pipe algebra and the runtime-agnostic primitives beneath it.
//!
//! One trait, [`pipe::Pipe`], carries typed `In -> Out` with an associated
//! `Err`; picking `In` or `Out` as `()` gives the four forms (transform,
//! source, sink, observe). Everything else here composes over it or sits
//! under it.
//!
//! - [`pipe`] -- [`pipe::Pipe`] and its additive `Send` form
//!   [`pipe::SendPipe`], the [`PipeExt`] combinator sugar
//!   (`.and_then`/`.filter`/`.fanout`/`.fanin`), `Request`/`Response`
//!   envelopes over `Bytes`, `HeaderList`, the streaming bodies
//!   (`ResponseStream`/`RequestStream` over `ChunkStream`) and the in-process
//!   `Carry` payload, `Handler` (`Request<Bytes> -> Response<Bytes>`),
//!   `SourcePipe`, and the resilience, rate-limit, retry, routing and
//!   telemetry/capture surfaces. Plugin authors' primary surface.
//! - [`sync`] -- runtime-agnostic concurrency primitives shaped like
//!   `tokio::sync` (`Mutex`, `Notify`, `mpsc`, `oneshot`, `watch`,
//!   `broadcast`, plus `blocking`, `task`, `shutdown`), so no pipe needs
//!   tokio to coordinate.
//! - [`stream`] -- byte-stream and datagram traits (`StreamConnection`,
//!   `StreamListener`, `StreamUpstream`, `DatagramSocket`) built on
//!   `futures::io`, shared by every wire-protocol stack above them.
//! - [`transport`] -- `Replay` (record-and-replay of a stream with fan-out
//!   sinks), the `tap_complete` taps and the `GenericStream` seam; sits below
//!   [`pipe`].
//! - [`driver`] -- [`block_on`], the no-runtime way to drive one future to
//!   completion.
//! - [`codec`], [`block_layout`], [`float_layout`] -- data descriptions of
//!   packed numeric storage: [`Codec`] names a packed byte layout,
//!   `BlockLayout` describes a block of stored numbers, `FloatLayout` one
//!   stored float.
//!
//! The crate is `no_std` unless the default `std` feature is on. `pipe`
//! splits further: the pipe forms compile with no allocator, most of the
//! envelope and combinator surface needs `alloc`, and registries and
//! lifecycle helpers need `std`. Each item's own doc names its tier.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

// the test harness always links std; a no_std crate must name it explicitly
// so `sync::blocking::futex`'s tests (compiled only when `std` is off)
// resolve `std::`.
#[cfg(all(test, not(feature = "std")))]
extern crate std;

pub mod block_layout;
pub mod codec;
pub mod driver;
pub mod float_layout;
pub mod pipe;
pub mod stream;
pub mod sync;
pub mod transport;

pub use codec::Codec;
pub use driver::block_on;
// Fluent combinator sugar (`.and_then`/`.filter`/`.fanout`/`.fanin`) over the
// pipe algebra — no prelude in this crate, so callers reach it via an
// explicit `use proxima_primitives::PipeExt`.
pub use pipe::PipeExt;
