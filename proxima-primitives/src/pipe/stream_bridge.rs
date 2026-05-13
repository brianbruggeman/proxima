//! `AsStream`/`AsSink` — the two orphan-rule-forced newtypes bridging
//! proxima's async source/sink vocabulary ([`UnpinPipe`], [`DrainSink`])
//! onto `futures::Stream`/`futures::Sink`, plus the std-tier `.into_reader()`/
//! `.into_writer()` lift onto `tokio_util::io::{StreamReader, SinkWriter,
//! CopyToBytes}` (never a hand-rolled `PipeReader`/`PipeWriter` — that would
//! reinvent what `tokio-util` already ships, violating P1/RISC).
//!
//! ## Scoping: `UnpinPipe`/`DrainSink` only, never arbitrary `Pipe::call`
//!
//! These bridges attach to the exhaustible source/sink vocabulary
//! ([`UnpinPipe::call`] with `Err = Exhausted`, the sink-shaped
//! [`DrainSink::accept`]), not to arbitrary `Pipe<In, Out>::call`. A
//! request/response `Pipe` link is not a stream anyway: per
//! `docs/pipe-to-metal/edges.md`'s reshape ruling, "the stream character lives
//! in the driver loop, not inside `Pipe`." There is no `into_stream` on
//! arbitrary `Pipe` here, by design — only on the exhaustible `UnpinPipe`
//! source shape and `DrainSink`.
//!
//! ## Why the newtype, not a blanket impl
//!
//! `futures::Stream`/`futures::Sink` are foreign traits; `UnpinPipe`/
//! `DrainSink` are ours, but a bare `impl<S: UnpinPipe<..>> Stream for S` is
//! rejected by the orphan rule (E0210 — no local type in the impl, `S` is an
//! unconstrained parameter). `AsStream<S>`/`AsSink<S>` are the minimal local
//! newtype that satisfies the rule; both are pure forwarding wrappers, adding
//! no buffering, no state, no behavior of their own.
//!
//! ## Busy-poll caveat (shared with every T0 source/sink in this module)
//!
//! `UnpinPipe`/`DrainSink` implementors at the T0 floor (`FanIn`, `RingSink`)
//! register no [`core::task::Waker`] when they have nothing ready — the
//! `docs/pipe-to-metal/edges.md` ruling names this explicitly: "true poll-mode
//! ... can't be wake-driven, only busy-polled." `AsStream`/`AsSink` forward
//! that characteristic unchanged: a `Poll::Pending` here is not a broken
//! contract, it is the inherited caller-re-drives model of the wrapped
//! source/sink.

use core::future::Future;
use core::ops::ControlFlow;
use core::pin::Pin;
use core::task::{Context, Poll};

use bytes::Bytes;

use crate::pipe::drain_sink::DrainSink;
use crate::pipe::fan_in::Exhausted;
use crate::pipe::primitives::UnpinPipe;

/// Forwards an [`UnpinPipe`] source's `call` straight through as
/// `futures::Stream` — the orphan-forced newtype (see module doc). `Ok(item)`
/// becomes `Some(item)`; [`Exhausted`] becomes `None`. No `Unpin` bound is
/// needed on `S`: `Pin<&mut Self>` derefs to `&Self` regardless of `S`'s own
/// `Unpin`-ness, and `UnpinPipe::call` only ever needs `&self`.
pub struct AsStream<S>(S);

impl<S> AsStream<S> {
    /// Recover the wrapped source.
    #[must_use]
    pub fn into_inner(self) -> S {
        self.0
    }
}

impl<S: UnpinPipe<In = (), Err = Exhausted>> futures::Stream for AsStream<S> {
    type Item = S::Out;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut call = self.0.call(());
        match Pin::new(&mut call).poll(cx) {
            Poll::Ready(Ok(item)) => Poll::Ready(Some(item)),
            Poll::Ready(Err(Exhausted)) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// `.into_stream()`/`.as_stream()` — lifts any exhaustible [`UnpinPipe`]
/// source onto `futures::Stream` by composing [`AsStream`] (no new behavior,
/// just the orphan-forced wrapper).
pub trait PollSourceExt: UnpinPipe<In = (), Err = Exhausted> + Sized {
    /// Owns `self`, wrapping it as a `futures::Stream`.
    fn into_stream(self) -> AsStream<Self> {
        AsStream(self)
    }

    /// Borrows `self` MUTABLY as a `futures::Stream`, so the source can be
    /// reused once the stream handle is dropped — the same shape as
    /// `Iterator::by_ref`. `&mut Self` composes via the [`UnpinPipe`]
    /// blanket impl for `&mut S` below, with no wrapper reinvention.
    fn as_stream(&mut self) -> AsStream<&mut Self> {
        AsStream(self)
    }
}

impl<S: UnpinPipe<In = (), Err = Exhausted>> PollSourceExt for S {}

// forwarding blanket impl: `&mut S` re-calls the same underlying source, so
// `as_stream`'s `AsStream<&mut Self>` is a genuine `UnpinPipe`, not a stub.
impl<S: UnpinPipe<In = (), Err = Exhausted> + ?Sized> UnpinPipe for &mut S {
    type In = ();
    type Out = S::Out;
    type Err = Exhausted;

    fn call(&self, (): ()) -> impl Future<Output = Result<S::Out, Exhausted>> + Unpin {
        (**self).call(())
    }
}

/// The one real error [`AsSink::start_send`] can report: [`DrainSink::accept`]
/// only signals backpressure via `ControlFlow::Break` (no richer cause), and
/// `poll_ready` is expected to prevent this in the well-behaved case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AsSinkError {
    /// The sink had no room for the item.
    #[error("sink rejected the item: no capacity")]
    Rejected,
}

// `std::io::Error` has no generic `From<E>` blanket (only the `E: Into<Box<dyn
// Error+Send+Sync>>`-bounded `Error::other` associated fn), so `SinkWriter`'s
// `E: Into<io::Error>` bound needs this explicit bridge — `io-bridge`-gated
// since it exists only to satisfy that std-tier composition.
#[cfg(feature = "io-bridge")]
impl From<AsSinkError> for std::io::Error {
    fn from(error: AsSinkError) -> Self {
        std::io::Error::other(error)
    }
}

/// Forwards [`DrainSink::accept`] straight through as `futures::Sink<Bytes>` —
/// the sink-side orphan-forced newtype (see module doc), box-free, no extra
/// buffering.
pub struct AsSink<S>(S);

impl<S> AsSink<S> {
    /// Recover the wrapped sink.
    #[must_use]
    pub fn into_inner(self) -> S {
        self.0
    }
}

impl<S: DrainSink<Item = [u8]> + Unpin> futures::Sink<Bytes> for AsSink<S> {
    type Error = AsSinkError;

    fn poll_ready(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        // busy-poll model (see module doc): a full sink reports `Pending`
        // without arming a wake; the caller re-drives.
        if self.get_mut().0.has_capacity() {
            Poll::Ready(Ok(()))
        } else {
            Poll::Pending
        }
    }

    fn start_send(self: Pin<&mut Self>, item: Bytes) -> Result<(), Self::Error> {
        match self.get_mut().0.accept(&item) {
            ControlFlow::Continue(()) => Ok(()),
            ControlFlow::Break(()) => Err(AsSinkError::Rejected),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        // a `DrainSink` write already lands in its backing storage (the ring
        // slot); there is no separate flush stage to drive.
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }
}

/// `.into_sink()`/`.as_sink()` — lifts any `[u8]`-shaped [`DrainSink`] onto
/// `futures::Sink<Bytes>` by composing [`AsSink`].
pub trait DrainSinkExt: DrainSink<Item = [u8]> + Sized {
    /// Owns `self`, wrapping it as a `futures::Sink<Bytes>`.
    fn into_sink(self) -> AsSink<Self> {
        AsSink(self)
    }

    /// Borrows `self` MUTABLY as a `futures::Sink<Bytes>` (mirrors
    /// [`PollSourceExt::as_stream`] — `accept` is `&mut self`, so `&self`
    /// cannot forward it).
    fn as_sink(&mut self) -> AsSink<&mut Self> {
        AsSink(self)
    }
}

impl<S: DrainSink<Item = [u8]>> DrainSinkExt for S {}

impl<S: DrainSink + ?Sized> DrainSink for &mut S {
    type Item = S::Item;

    fn accept(&mut self, item: &Self::Item) -> ControlFlow<()> {
        (**self).accept(item)
    }

    fn has_capacity(&self) -> bool {
        (**self).has_capacity()
    }
}

/// std-tier `.into_reader()`/`.into_writer()`: lifts [`AsStream`]/[`AsSink`]
/// onto `futures::io::{AsyncRead, AsyncWrite}` — the workspace's canonical
/// std-tier byte-stream trait (see `proxima_core::io`'s module doc for the
/// full tier rule: `prime::os::net::TcpStream` implements it, and every real
/// std transport in this workspace binds it). The composition itself reuses
/// `tokio-util`'s own `StreamReader`/`SinkWriter`/`CopyToBytes` (P1 — no
/// hand-rolled `PipeReader`/`PipeWriter`), then crosses from tokio's native
/// `AsyncRead`/`AsyncWrite` to the canonical `futures::io` ones via
/// `tokio_util::compat` — the SAME bridge idiom
/// `proxima-http/src/listener/mod.rs`'s `socket.compat()` already uses at the
/// tokio-accept boundary, not a new adapter.
#[cfg(feature = "io-bridge")]
mod reader_writer {
    use bytes::Bytes;
    use tokio_util::compat::{Compat, TokioAsyncReadCompatExt as _, TokioAsyncWriteCompatExt as _};
    use tokio_util::io::{CopyToBytes, SinkWriter, StreamReader};

    use super::{AsSink, AsStream, DrainSinkExt as _, Exhausted, PollSourceExt as _};
    use crate::pipe::drain_sink::DrainSink;
    use crate::pipe::primitives::UnpinPipe;

    /// `.into_reader()`/`.as_reader()` — wraps `tokio_util::io::StreamReader`
    /// over [`AsStream`], then `.compat()`s it onto `futures::io::AsyncRead`
    /// (the canonical std trait). `StreamReader` needs a fallible byte-chunk
    /// stream (`Item = Result<Bytes, Error>`), so only an [`UnpinPipe`] source
    /// shaped that way gets a reader; `Error` must convert into
    /// `std::io::Error` because `StreamReader`'s own `AsyncRead` impl is
    /// std::io::Error-shaped.
    pub trait IntoReader<Error>:
        UnpinPipe<In = (), Out = Result<Bytes, Error>, Err = Exhausted> + Unpin + Sized
    where
        Error: Into<std::io::Error>,
    {
        /// Owns `self`; composes `StreamReader::new` over
        /// [`Self::into_stream`](PollSourceExt::into_stream), then `.compat()`
        /// onto `futures::io::AsyncRead`.
        fn into_reader(self) -> Compat<StreamReader<AsStream<Self>, Bytes>> {
            StreamReader::new(self.into_stream()).compat()
        }

        /// Borrows `self` mutably; same composition over
        /// [`Self::as_stream`](PollSourceExt::as_stream).
        fn as_reader(&mut self) -> Compat<StreamReader<AsStream<&mut Self>, Bytes>> {
            StreamReader::new(self.as_stream()).compat()
        }
    }

    impl<S, Error> IntoReader<Error> for S
    where
        S: UnpinPipe<In = (), Out = Result<Bytes, Error>, Err = Exhausted> + Unpin,
        Error: Into<std::io::Error>,
    {
    }

    /// `.into_writer()`/`.as_writer()` — wraps `tokio_util::io::SinkWriter`
    /// over `CopyToBytes` over [`AsSink`], then `.compat_write()`s it onto
    /// `futures::io::AsyncWrite` (the canonical std trait): `SinkWriter`
    /// writes a `Sink<&[u8]>`; `CopyToBytes` is `tokio-util`'s own
    /// `Sink<Bytes> -> Sink<&[u8]>` adapter (its docs name this exact
    /// composition), so no new adapter is authored here, only wired.
    pub trait IntoWriter: DrainSink<Item = [u8]> + Unpin + Sized {
        /// Owns `self`; composes `SinkWriter::new(CopyToBytes::new(..))` over
        /// [`Self::into_sink`](DrainSinkExt::into_sink), then `.compat_write()`
        /// onto `futures::io::AsyncWrite`.
        fn into_writer(self) -> Compat<SinkWriter<CopyToBytes<AsSink<Self>>>> {
            SinkWriter::new(CopyToBytes::new(self.into_sink())).compat_write()
        }

        /// Borrows `self` mutably; same composition over
        /// [`Self::as_sink`](DrainSinkExt::as_sink).
        fn as_writer(&mut self) -> Compat<SinkWriter<CopyToBytes<AsSink<&mut Self>>>> {
            SinkWriter::new(CopyToBytes::new(self.as_sink())).compat_write()
        }
    }

    impl<S: DrainSink<Item = [u8]> + Unpin> IntoWriter for S {}
}

#[cfg(feature = "io-bridge")]
pub use reader_writer::{IntoReader, IntoWriter};

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use core::cell::Cell;
    use core::task::Waker;

    use futures::{Sink, Stream};

    use super::*;
    use crate::pipe::drain_sink::RingSink;

    // a captured-shaped HTTP/1 request, split across three poll_next calls —
    // real bytes, not `b"AAAA"`, matching P9.
    const REQUEST_LINE: &[u8] = b"GET /orders?id=42 HTTP/1.1\r\n";
    const HOST_HEADER: &[u8] = b"Host: api.example.internal\r\n";
    const TAIL: &[u8] = b"Accept: application/json\r\n\r\n";

    struct ThreeChunkSource {
        chunks: [&'static [u8]; 3],
        next: Cell<usize>,
    }

    impl UnpinPipe for ThreeChunkSource {
        type In = ();
        type Out = Result<Bytes, std::io::Error>;
        type Err = Exhausted;

        fn call(&self, (): ()) -> impl Future<Output = Result<Self::Out, Exhausted>> + Unpin {
            let index = self.next.get();
            if index >= self.chunks.len() {
                return core::future::ready(Err(Exhausted));
            }
            let chunk = self.chunks[index];
            self.next.set(index + 1);
            core::future::ready(Ok(Ok(Bytes::from_static(chunk))))
        }
    }

    fn noop_context() -> Context<'static> {
        Context::from_waker(Waker::noop())
    }

    #[test]
    fn as_stream_forwards_poll_next_in_order() {
        let source = ThreeChunkSource {
            chunks: [REQUEST_LINE, HOST_HEADER, TAIL],
            next: Cell::new(0),
        };
        let mut stream = source.into_stream();
        let mut cx = noop_context();

        let first = Pin::new(&mut stream).poll_next(&mut cx);
        let Poll::Ready(Some(Ok(chunk))) = first else {
            panic!("expected the first chunk, got {first:?}");
        };
        assert_eq!(&chunk[..], REQUEST_LINE);

        let second = Pin::new(&mut stream).poll_next(&mut cx);
        let Poll::Ready(Some(Ok(chunk))) = second else {
            panic!("expected the second chunk, got {second:?}");
        };
        assert_eq!(&chunk[..], HOST_HEADER);
    }

    #[test]
    fn as_stream_drains_to_none_when_source_is_exhausted() {
        let source = ThreeChunkSource {
            chunks: [REQUEST_LINE, HOST_HEADER, TAIL],
            next: Cell::new(3),
        };
        let mut stream = source.into_stream();
        let mut cx = noop_context();
        let Poll::Ready(None) = Pin::new(&mut stream).poll_next(&mut cx) else {
            panic!("expected the source to report exhausted");
        };
    }

    #[test]
    fn as_sink_round_trips_real_bytes_through_a_ring() {
        let mut sink: AsSink<RingSink<4, 64>> = RingSink::new().into_sink();
        let mut cx = noop_context();

        assert!(matches!(
            Pin::new(&mut sink).poll_ready(&mut cx),
            Poll::Ready(Ok(()))
        ));
        Pin::new(&mut sink)
            .start_send(Bytes::from_static(REQUEST_LINE))
            .expect("ring has room for one frame");
        Pin::new(&mut sink)
            .start_send(Bytes::from_static(HOST_HEADER))
            .expect("ring has room for a second frame");

        let mut ring = sink.into_inner();
        assert_eq!(ring.pop(), Some(REQUEST_LINE));
        assert_eq!(ring.pop(), Some(HOST_HEADER));
    }

    #[test]
    fn as_sink_reports_rejected_when_frame_too_large_for_the_ring() {
        let mut sink: AsSink<RingSink<1, 4>> = RingSink::new().into_sink();
        let error = Pin::new(&mut sink)
            .start_send(Bytes::from_static(b"toolong"))
            .expect_err("frame exceeds the 4-byte slot capacity");
        assert_eq!(error, AsSinkError::Rejected);
    }

    #[test]
    fn as_stream_as_stream_borrow_leaves_the_source_usable_after() {
        let mut source = ThreeChunkSource {
            chunks: [REQUEST_LINE, HOST_HEADER, TAIL],
            next: Cell::new(0),
        };
        let mut cx = noop_context();
        {
            let mut borrowed = source.as_stream();
            let first = Pin::new(&mut borrowed).poll_next(&mut cx);
            assert!(first.is_ready());
        }
        // the borrow was dropped; `source` itself still owns its cursor state
        // and can be polled directly (or wrapped again) afterward.
        assert_eq!(source.next.get(), 1);
    }
}

#[cfg(all(test, feature = "io-bridge"))]
mod io_bridge_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use bytes::Bytes;
    use core::cell::Cell;
    use core::future::Future;
    use futures::io::{AsyncReadExt, AsyncWriteExt};

    use super::{Exhausted, IntoReader as _, IntoWriter as _};
    use crate::pipe::drain_sink::RingSink;
    use crate::pipe::primitives::UnpinPipe;

    // the same captured-shaped HTTP/1 request as the `AsStream`/`AsSink`
    // tests above, split across three chunks — real bytes, P9.
    const REQUEST_LINE: &[u8] = b"GET /orders?id=42 HTTP/1.1\r\n";
    const HOST_HEADER: &[u8] = b"Host: api.example.internal\r\n";
    const TAIL: &[u8] = b"Accept: application/json\r\n\r\n";

    struct ChunkedRequestSource {
        chunks: [&'static [u8]; 3],
        next: Cell<usize>,
    }

    impl UnpinPipe for ChunkedRequestSource {
        type In = ();
        type Out = Result<Bytes, std::io::Error>;
        type Err = Exhausted;

        fn call(&self, (): ()) -> impl Future<Output = Result<Self::Out, Exhausted>> + Unpin {
            let index = self.next.get();
            if index >= self.chunks.len() {
                return core::future::ready(Err(Exhausted));
            }
            let chunk = self.chunks[index];
            self.next.set(index + 1);
            core::future::ready(Ok(Ok(Bytes::from_static(chunk))))
        }
    }

    #[test]
    fn into_reader_round_trips_a_captured_http_request() {
        let source = ChunkedRequestSource {
            chunks: [REQUEST_LINE, HOST_HEADER, TAIL],
            next: Cell::new(0),
        };
        let mut reader = source.into_reader();

        let mut collected = Vec::new();
        futures::executor::block_on(async {
            reader
                .read_to_end(&mut collected)
                .await
                .expect("an in-memory UnpinPipe stream never errors")
        });

        let mut expected = Vec::new();
        expected.extend_from_slice(REQUEST_LINE);
        expected.extend_from_slice(HOST_HEADER);
        expected.extend_from_slice(TAIL);
        assert_eq!(collected, expected, "reader output equals the fed bytes");
    }

    #[test]
    fn into_writer_round_trips_bytes_into_a_ring_sink() {
        let mut writer = RingSink::<4, 64>::new().into_writer();

        futures::executor::block_on(async {
            writer
                .write_all(REQUEST_LINE)
                .await
                .expect("ring has room for the request line");
            writer
                .write_all(HOST_HEADER)
                .await
                .expect("ring has room for the host header");
            writer
                .close()
                .await
                .expect("close is a no-op success for a RingSink");
        });

        let mut ring = writer
            .into_inner()
            .into_inner()
            .into_inner()
            .into_inner();
        assert_eq!(ring.pop(), Some(REQUEST_LINE), "first frame written");
        assert_eq!(ring.pop(), Some(HOST_HEADER), "second frame written");
    }
}
