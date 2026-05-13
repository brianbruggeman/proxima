//! Poll-based async IO traits over an associated error type — proxima's
//! no_std + no_alloc async IO **floor**, and ONLY the floor.
//!
//! ## Canonical trait per tier (the rule this module exists to teach)
//!
//! - **std tier: `futures::io::{AsyncRead, AsyncWrite}` is canonical.**
//!   Industry standard, already what `prime::os::net::TcpStream` implements
//!   (`prime/src/os/net.rs:64,594,638`) and what every real std transport in
//!   this workspace binds (`proxima_primitives::stream::StreamConnection`,
//!   `serve_h1_connection`). A std-only type — anything that already needs
//!   `std::sync::Mutex`, a std socket, or similar — implements `futures::io`
//!   and STOPS there; it must NOT also implement this module's traits. Two
//!   `AsyncRead`s on one real socket is not redundancy-as-safety-margin, it
//!   is the exact ambiguity that costs a reader an afternoon (see
//!   `docs/pipe-to-metal/edges.md`'s 2026-07-16 concentration entry).
//! - **no_std / no-alloc floor: this module's `AsyncRead`/`AsyncWrite` is
//!   canonical, and earns its existence for exactly one reason** —
//!   `futures::io`'s traits are defined inside `futures-io`'s own
//!   `#[cfg(feature = "std")]`-gated module (verified against the vendored
//!   crate: `futures-io-0.3.x/src/lib.rs`'s `mod if_std`), so with
//!   `default-features = false` they do not exist at all, not even as an
//!   importable symbol. `tokio::io` is std-only too. There is no poll-based
//!   async IO trait in the std/futures ecosystem that compiles at the
//!   bare-metal floor, so a no_std/no-alloc primitive (e.g.
//!   `proxima_primitives::pipe::{RingSource, RingSink}`, the workspace's
//!   T0 *DK-shaped ring types) that wants a standard streaming-poll surface
//!   has no other option. These traits carry their own [`AsyncRead::Error`]
//!   instead of `std::io::Error` (guiding-principle 3), so they are just
//!   `core::pin` + `core::task` + an associated type — nothing else — and
//!   therefore compile at every tier of the `{no_std, std} × {no_alloc,
//!   alloc}` matrix, bare metal included.
//! - **The compat bridges below (`Prepend`/`FromFutures`/`FromTokio`/
//!   `IntoFutures`/`IntoTokio`) are the ONLY other legitimate reason to
//!   touch this module from std code**: they let a std listener (already on
//!   `futures::io`/`tokio::io`) round-trip bytes through a floor-shaped
//!   adapter (e.g. re-presenting sniffed preface bytes) without the floor
//!   adapter itself needing to know which std ecosystem it is embedded in.
//!
//! Poll-based, not `async fn` in trait: the caller drives the state machine
//! explicitly (principle 11), which is what lets a self-referential reader be
//! pinned and stay heap-free.

use core::pin::Pin;
use core::task::{Context, Poll};

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Read bytes from an async source, driven by explicit `poll`.
///
/// `Error` is the reader's own error — never `std::io::Error` — so an impl at
/// the no_alloc floor can use a `Copy` enum and stay allocation-free.
///
/// Canonical ONLY at the no_std/no-alloc floor (see the module doc's tier
/// rule): a type that already requires std should implement
/// `futures::io::AsyncRead` instead and never this trait too.
pub trait AsyncRead {
    /// The reader's error. A bare-metal impl typically uses a small `Copy`
    /// enum; a std adapter can set this to `std::io::Error`.
    type Error;

    /// Attempt to read into `buf`, returning the number of bytes read. `Ok(0)`
    /// signals end-of-stream. `Poll::Pending` registers `cx`'s waker.
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<Result<usize, Self::Error>>;
}

/// Write bytes to an async sink, driven by explicit `poll`.
///
/// Canonical ONLY at the no_std/no-alloc floor (see [`AsyncRead`]'s doc and
/// this module's tier rule): a std-only type implements `futures::io::AsyncWrite`
/// instead, never both.
pub trait AsyncWrite {
    /// The writer's error — never `std::io::Error` (see [`AsyncRead::Error`]).
    type Error;

    /// Attempt to write `buf`, returning the number of bytes accepted.
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, Self::Error>>;

    /// Flush buffered bytes toward their destination.
    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>>;

    /// Close the sink; no further writes follow a successful close.
    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>>;
}

/// Replay a prepended byte buffer on the first reads, then delegate to the
/// wrapped reader — the canonical "re-present sniffed / over-read bytes"
/// adapter (h1-vs-h2 preface detection, a PROXY header's leftover application
/// bytes). Alloc-tier (it owns a `Vec` buffer) but carries the INNER reader's
/// `Error`, so it composes [`AsyncRead`] at no_std+alloc — bare metal included,
/// no `std::io::Error`, no runtime binding. This is what the std-locked
/// `futures::io`-based prepend adapters lift onto.
#[cfg(feature = "alloc")]
pub struct Prepend<S> {
    leftover: Vec<u8>,
    cursor: usize,
    inner: S,
}

#[cfg(feature = "alloc")]
impl<S> Prepend<S> {
    /// Wrap `inner`, replaying `leftover` before any of `inner`'s own bytes.
    #[must_use]
    pub fn new(leftover: Vec<u8>, inner: S) -> Self {
        Self {
            leftover,
            cursor: 0,
            inner,
        }
    }

    /// Recover the wrapped reader (its prepended bytes may be partly drained).
    pub fn into_inner(self) -> S {
        self.inner
    }
}

#[cfg(feature = "alloc")]
impl<S: AsyncRead + Unpin> AsyncRead for Prepend<S> {
    type Error = S::Error;

    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<Result<usize, Self::Error>> {
        let this = self.get_mut();
        if this.cursor < this.leftover.len() {
            let remaining = &this.leftover[this.cursor..];
            let count = remaining.len().min(buf.len());
            buf[..count].copy_from_slice(&remaining[..count]);
            this.cursor += count;
            if this.cursor == this.leftover.len() {
                // drop the buffer once drained — a plain passthrough from here.
                this.leftover = Vec::new();
            }
            return Poll::Ready(Ok(count));
        }
        Pin::new(&mut this.inner).poll_read(cx, buf)
    }
}

#[cfg(feature = "alloc")]
impl<S: AsyncWrite + Unpin> AsyncWrite for Prepend<S> {
    type Error = S::Error;

    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, Self::Error>> {
        Pin::new(&mut self.get_mut().inner).poll_write(cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Pin::new(&mut self.get_mut().inner).poll_flush(cx)
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Pin::new(&mut self.get_mut().inner).poll_close(cx)
    }
}

/// Bridge a `futures::io` reader/writer INTO the [`AsyncRead`]/[`AsyncWrite`]
/// seam (`Error = std::io::Error`). Lets a std futures socket feed an
/// `io` adapter (e.g. [`Prepend`]) at the std tier. Std-only: `futures-io`
/// is defined over `std::io::Error`.
#[cfg(feature = "io-async-compat")]
pub struct FromFutures<S>(pub S);

#[cfg(feature = "io-async-compat")]
impl<S: futures_io::AsyncRead + Unpin> AsyncRead for FromFutures<S> {
    type Error = std::io::Error;

    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<Result<usize, Self::Error>> {
        Pin::new(&mut self.get_mut().0).poll_read(cx, buf)
    }
}

#[cfg(feature = "io-async-compat")]
impl<S: futures_io::AsyncWrite + Unpin> AsyncWrite for FromFutures<S> {
    type Error = std::io::Error;

    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, Self::Error>> {
        Pin::new(&mut self.get_mut().0).poll_write(cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Pin::new(&mut self.get_mut().0).poll_flush(cx)
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Pin::new(&mut self.get_mut().0).poll_close(cx)
    }
}

/// Bridge an [`AsyncRead`]/[`AsyncWrite`] BACK to `futures::io`, so an
/// `io` adapter can be handed to a std driver that still speaks
/// `futures::io`. Requires the seam's `Error` to be convertible into
/// `std::io::Error` (the `FromFutures` round-trip is exact).
#[cfg(feature = "io-async-compat")]
pub struct IntoFutures<S>(pub S);

#[cfg(feature = "io-async-compat")]
impl<S> futures_io::AsyncRead for IntoFutures<S>
where
    S: AsyncRead + Unpin,
    S::Error: Into<std::io::Error>,
{
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<Result<usize, std::io::Error>> {
        match Pin::new(&mut self.get_mut().0).poll_read(cx, buf) {
            Poll::Ready(Ok(count)) => Poll::Ready(Ok(count)),
            Poll::Ready(Err(error)) => Poll::Ready(Err(error.into())),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(feature = "io-async-compat")]
impl<S> futures_io::AsyncWrite for IntoFutures<S>
where
    S: AsyncWrite + Unpin,
    S::Error: Into<std::io::Error>,
{
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, std::io::Error>> {
        match Pin::new(&mut self.get_mut().0).poll_write(cx, buf) {
            Poll::Ready(Ok(count)) => Poll::Ready(Ok(count)),
            Poll::Ready(Err(error)) => Poll::Ready(Err(error.into())),
            Poll::Pending => Poll::Pending,
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), std::io::Error>> {
        match Pin::new(&mut self.get_mut().0).poll_flush(cx) {
            Poll::Ready(Ok(())) => Poll::Ready(Ok(())),
            Poll::Ready(Err(error)) => Poll::Ready(Err(error.into())),
            Poll::Pending => Poll::Pending,
        }
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), std::io::Error>> {
        match Pin::new(&mut self.get_mut().0).poll_close(cx) {
            Poll::Ready(Ok(())) => Poll::Ready(Ok(())),
            Poll::Ready(Err(error)) => Poll::Ready(Err(error.into())),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Bridge a `tokio::io` reader/writer INTO the [`AsyncRead`]/[`AsyncWrite`]
/// seam (`Error = std::io::Error`). Lets a std tokio socket feed an
/// `io` adapter (e.g. [`Prepend`]) at the std tier — the tokio sibling
/// of [`FromFutures`]. Std-only: `tokio::io` is defined over `std::io::Error`.
#[cfg(feature = "io-async-compat-tokio")]
pub struct FromTokio<S>(pub S);

#[cfg(feature = "io-async-compat-tokio")]
impl<S: tokio::io::AsyncRead + Unpin> AsyncRead for FromTokio<S> {
    type Error = std::io::Error;

    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<Result<usize, Self::Error>> {
        let mut read_buf = tokio::io::ReadBuf::new(buf);
        match Pin::new(&mut self.get_mut().0).poll_read(cx, &mut read_buf) {
            Poll::Ready(Ok(())) => Poll::Ready(Ok(read_buf.filled().len())),
            Poll::Ready(Err(error)) => Poll::Ready(Err(error)),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(feature = "io-async-compat-tokio")]
impl<S: tokio::io::AsyncWrite + Unpin> AsyncWrite for FromTokio<S> {
    type Error = std::io::Error;

    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, Self::Error>> {
        Pin::new(&mut self.get_mut().0).poll_write(cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Pin::new(&mut self.get_mut().0).poll_flush(cx)
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Pin::new(&mut self.get_mut().0).poll_shutdown(cx)
    }
}

/// Bridge an [`AsyncRead`]/[`AsyncWrite`] BACK to `tokio::io`, so an
/// `io` adapter can be handed to a std driver that still speaks
/// `tokio::io` (e.g. a `tokio-rustls` acceptor, `.compat()` into
/// `futures::io`). Requires the seam's `Error` to be convertible into
/// `std::io::Error` (the `FromTokio` round-trip is exact) — the tokio
/// sibling of [`IntoFutures`].
#[cfg(feature = "io-async-compat-tokio")]
pub struct IntoTokio<S>(pub S);

#[cfg(feature = "io-async-compat-tokio")]
impl<S> tokio::io::AsyncRead for IntoTokio<S>
where
    S: AsyncRead + Unpin,
    S::Error: Into<std::io::Error>,
{
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let dst = buf.initialize_unfilled();
        match Pin::new(&mut self.get_mut().0).poll_read(cx, dst) {
            Poll::Ready(Ok(count)) => {
                buf.advance(count);
                Poll::Ready(Ok(()))
            }
            Poll::Ready(Err(error)) => Poll::Ready(Err(error.into())),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(feature = "io-async-compat-tokio")]
impl<S> tokio::io::AsyncWrite for IntoTokio<S>
where
    S: AsyncWrite + Unpin,
    S::Error: Into<std::io::Error>,
{
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        match Pin::new(&mut self.get_mut().0).poll_write(cx, buf) {
            Poll::Ready(Ok(count)) => Poll::Ready(Ok(count)),
            Poll::Ready(Err(error)) => Poll::Ready(Err(error.into())),
            Poll::Pending => Poll::Pending,
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match Pin::new(&mut self.get_mut().0).poll_flush(cx) {
            Poll::Ready(Ok(())) => Poll::Ready(Ok(())),
            Poll::Ready(Err(error)) => Poll::Ready(Err(error.into())),
            Poll::Pending => Poll::Pending,
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match Pin::new(&mut self.get_mut().0).poll_close(cx) {
            Poll::Ready(Ok(())) => Poll::Ready(Ok(())),
            Poll::Ready(Err(error)) => Poll::Ready(Err(error.into())),
            Poll::Pending => Poll::Pending,
        }
    }
}

// mocks below are slice-cursor + Copy-enum shaped on purpose: they double as
// proof that the no_alloc floor (no Vec, no Box, no String) is enough to
// implement both traits.
#[cfg(test)]
mod tests {
    use core::task::Waker;

    #[cfg(feature = "io-async-compat-tokio")]
    use tokio::io::{AsyncRead as _, AsyncWrite as _};

    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum MockError {
        Broken,
    }

    struct SliceReader<'a> {
        data: &'a [u8],
        pos: usize,
    }

    impl<'a> SliceReader<'a> {
        fn new(data: &'a [u8]) -> Self {
            Self { data, pos: 0 }
        }
    }

    impl AsyncRead for SliceReader<'_> {
        type Error = MockError;

        fn poll_read(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<Result<usize, Self::Error>> {
            let this = self.get_mut();
            let remaining = &this.data[this.pos..];
            let count = remaining.len().min(buf.len());
            buf[..count].copy_from_slice(&remaining[..count]);
            this.pos += count;
            Poll::Ready(Ok(count))
        }
    }

    struct BrokenReader;

    impl AsyncRead for BrokenReader {
        type Error = MockError;

        fn poll_read(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            _buf: &mut [u8],
        ) -> Poll<Result<usize, Self::Error>> {
            Poll::Ready(Err(MockError::Broken))
        }
    }

    // pends `pending_remaining` times before delegating to the inner reader —
    // exercises the waker-registration contract without an executor.
    struct PendingOnceReader<'a> {
        inner: SliceReader<'a>,
        pending_remaining: u8,
    }

    impl AsyncRead for PendingOnceReader<'_> {
        type Error = MockError;

        fn poll_read(
            self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<Result<usize, Self::Error>> {
            let this = self.get_mut();
            if this.pending_remaining > 0 {
                this.pending_remaining -= 1;
                cx.waker().wake_by_ref();
                return Poll::Pending;
            }
            Pin::new(&mut this.inner).poll_read(cx, buf)
        }
    }

    struct SliceWriter<'a> {
        buf: &'a mut [u8],
        pos: usize,
    }

    impl AsyncWrite for SliceWriter<'_> {
        type Error = MockError;

        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            data: &[u8],
        ) -> Poll<Result<usize, Self::Error>> {
            let this = self.get_mut();
            let remaining = this.buf.len() - this.pos;
            let count = remaining.min(data.len());
            this.buf[this.pos..this.pos + count].copy_from_slice(&data[..count]);
            this.pos += count;
            Poll::Ready(Ok(count))
        }

        fn poll_flush(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn poll_close(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }
    }

    #[test]
    fn read_happy_path_fills_buffer_in_one_poll() {
        let data = b"proxima";
        let mut reader = SliceReader::new(data);
        let mut context = Context::from_waker(Waker::noop());
        let mut buf = [0u8; 7];

        let outcome = Pin::new(&mut reader).poll_read(&mut context, &mut buf);

        assert_eq!(
            outcome,
            Poll::Ready(Ok(7)),
            "a buffer sized to fit must drain the source in one poll"
        );
        assert_eq!(&buf, data, "bytes read must match source data");
    }

    #[test]
    fn read_returns_zero_at_eof() {
        let data = b"ab";
        let mut reader = SliceReader::new(data);
        let mut context = Context::from_waker(Waker::noop());
        let mut buf = [0u8; 2];

        let _ = Pin::new(&mut reader).poll_read(&mut context, &mut buf);
        let eof_outcome = Pin::new(&mut reader).poll_read(&mut context, &mut buf);

        assert_eq!(
            eof_outcome,
            Poll::Ready(Ok(0)),
            "a drained reader must signal eof, not error or pend"
        );
    }

    #[test]
    fn read_reassembles_across_undersized_buffers() {
        let data = b"proxima-tier-lift";
        let mut reader = SliceReader::new(data);
        let mut context = Context::from_waker(Waker::noop());
        let mut assembled = [0u8; 32];
        let mut written = 0usize;

        loop {
            let mut chunk = [0u8; 3];
            match Pin::new(&mut reader).poll_read(&mut context, &mut chunk) {
                Poll::Ready(Ok(0)) => break,
                Poll::Ready(Ok(count)) => {
                    assembled[written..written + count].copy_from_slice(&chunk[..count]);
                    written += count;
                }
                Poll::Ready(Err(error)) => panic!("unexpected read error: {error:?}"),
                Poll::Pending => panic!("SliceReader never pends"),
            }
        }

        assert_eq!(
            written,
            data.len(),
            "partial reads must sum to the full source length"
        );
        assert_eq!(
            &assembled[..written],
            data,
            "reassembled bytes must match source data byte-for-byte"
        );
    }

    #[test]
    fn read_error_propagates_through_poll() {
        let mut reader = BrokenReader;
        let mut context = Context::from_waker(Waker::noop());
        let mut buf = [0u8; 4];

        let outcome = Pin::new(&mut reader).poll_read(&mut context, &mut buf);

        assert_eq!(
            outcome,
            Poll::Ready(Err(MockError::Broken)),
            "broken reader must surface its error, not silently succeed"
        );
    }

    // the E2E capability proof: an adapter composes AsyncRead at the alloc tier,
    // carrying the inner reader's Error (not std::io::Error), so it reaches
    // no_std+alloc / bare metal — exactly what the std futures::io adapters lift onto.
    #[cfg(feature = "alloc")]
    #[test]
    fn prepend_replays_leftover_then_delegates_to_inner() {
        use alloc::vec::Vec;

        let mut prepended = Prepend::new(b"PRI ".to_vec(), SliceReader::new(b"proxima"));
        let mut context = Context::from_waker(Waker::noop());
        let mut assembled: Vec<u8> = Vec::new();

        loop {
            let mut chunk = [0u8; 4];
            match Pin::new(&mut prepended).poll_read(&mut context, &mut chunk) {
                Poll::Ready(Ok(0)) => break,
                Poll::Ready(Ok(count)) => assembled.extend_from_slice(&chunk[..count]),
                other => panic!("prepend read must not error or pend here: {other:?}"),
            }
        }

        assert_eq!(
            &assembled[..],
            b"PRI proxima",
            "the prepended leftover must precede the inner reader's bytes, intact"
        );
    }

    // the C2+C3 std composition: a std futures::io reader, bridged INTO the seam
    // via FromFutures, then wrapped by the io Prepend adapter — the exact
    // shape a std listener uses to feed its sniffed bytes back onto the seam.
    #[cfg(feature = "io-async-compat")]
    #[test]
    fn from_futures_bridge_feeds_the_prepend_adapter() {
        use alloc::vec::Vec;

        let socket = futures::io::Cursor::new(b"proxima".to_vec());
        let mut adapter = Prepend::new(b">>".to_vec(), FromFutures(socket));
        let mut context = Context::from_waker(Waker::noop());
        let mut assembled: Vec<u8> = Vec::new();

        loop {
            let mut chunk = [0u8; 4];
            match Pin::new(&mut adapter).poll_read(&mut context, &mut chunk) {
                Poll::Ready(Ok(0)) => break,
                Poll::Ready(Ok(count)) => assembled.extend_from_slice(&chunk[..count]),
                other => panic!("bridged read must not error or pend here: {other:?}"),
            }
        }

        assert_eq!(
            &assembled[..],
            b">>proxima",
            "prepend leftover then the bridged futures reader's bytes, in order"
        );
    }

    // the tokio sibling of TokioSliceReader-shaped mocks below: manually
    // implements tokio::io::AsyncRead over a byte slice, no executor needed.
    #[cfg(feature = "io-async-compat-tokio")]
    struct TokioSliceReader<'a> {
        data: &'a [u8],
        pos: usize,
    }

    #[cfg(feature = "io-async-compat-tokio")]
    impl<'a> TokioSliceReader<'a> {
        fn new(data: &'a [u8]) -> Self {
            Self { data, pos: 0 }
        }
    }

    #[cfg(feature = "io-async-compat-tokio")]
    impl tokio::io::AsyncRead for TokioSliceReader<'_> {
        fn poll_read(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &mut tokio::io::ReadBuf<'_>,
        ) -> Poll<std::io::Result<()>> {
            let this = self.get_mut();
            let remaining = &this.data[this.pos..];
            let count = remaining.len().min(buf.remaining());
            buf.put_slice(&remaining[..count]);
            this.pos += count;
            Poll::Ready(Ok(()))
        }
    }

    // the tokio-analogue of from_futures_bridge_feeds_the_prepend_adapter: a
    // tokio::io reader, bridged INTO the seam via FromTokio, then wrapped by
    // the io Prepend adapter — the shape the tokio proxy-protocol
    // listener path uses to feed its leftover header bytes.
    #[cfg(feature = "io-async-compat-tokio")]
    #[test]
    fn from_tokio_bridge_feeds_the_prepend_adapter() {
        use alloc::vec::Vec;

        let socket = TokioSliceReader::new(b"proxima");
        let mut adapter = Prepend::new(b">>".to_vec(), FromTokio(socket));
        let mut context = Context::from_waker(Waker::noop());
        let mut assembled: Vec<u8> = Vec::new();

        loop {
            let mut chunk = [0u8; 4];
            match Pin::new(&mut adapter).poll_read(&mut context, &mut chunk) {
                Poll::Ready(Ok(0)) => break,
                Poll::Ready(Ok(count)) => assembled.extend_from_slice(&chunk[..count]),
                other => panic!("bridged read must not error or pend here: {other:?}"),
            }
        }

        assert_eq!(
            &assembled[..],
            b">>proxima",
            "prepend leftover then the bridged tokio reader's bytes, in order"
        );
    }

    // round trip: an io reader bridged FROM tokio then bridged BACK
    // INTO tokio must reproduce the source bytes exactly — proves the two
    // bridges are inverses at the byte level, not just type-compatible.
    #[cfg(feature = "io-async-compat-tokio")]
    #[test]
    fn into_tokio_round_trips_through_from_tokio() {
        let data = b"proxima-tier-lift";
        let socket = TokioSliceReader::new(data);
        let mut bridged = IntoTokio(FromTokio(socket));
        let mut context = Context::from_waker(Waker::noop());
        let mut backing = [0u8; 32];
        let mut read_buf = tokio::io::ReadBuf::new(&mut backing);

        let outcome = Pin::new(&mut bridged).poll_read(&mut context, &mut read_buf);

        assert!(
            matches!(outcome, Poll::Ready(Ok(()))),
            "round-trip read must not error or pend: {outcome:?}"
        );
        assert_eq!(
            read_buf.filled(),
            data,
            "IntoTokio(FromTokio(_)) must preserve bytes byte-for-byte"
        );
    }

    // the write-side round trip: io AsyncWrite bridged back into
    // tokio::io::AsyncWrite over a Vec<u8> sink (tokio implements AsyncWrite
    // for Vec<u8> under io-util) — write, flush, shutdown all delegate.
    #[cfg(feature = "io-async-compat-tokio")]
    #[test]
    fn into_tokio_write_flush_shutdown_delegate_to_from_tokio() {
        let sink: Vec<u8> = Vec::new();
        let mut bridged = IntoTokio(FromTokio(sink));
        let mut context = Context::from_waker(Waker::noop());

        let write_outcome = Pin::new(&mut bridged).poll_write(&mut context, b"proxima!");
        assert!(
            matches!(write_outcome, Poll::Ready(Ok(8))),
            "writer must accept the full payload: {write_outcome:?}"
        );

        let flush_outcome = Pin::new(&mut bridged).poll_flush(&mut context);
        assert!(
            matches!(flush_outcome, Poll::Ready(Ok(()))),
            "flush must complete synchronously for a Vec sink: {flush_outcome:?}"
        );

        let shutdown_outcome = Pin::new(&mut bridged).poll_shutdown(&mut context);
        assert!(
            matches!(shutdown_outcome, Poll::Ready(Ok(()))),
            "shutdown must complete synchronously for a Vec sink: {shutdown_outcome:?}"
        );
        assert_eq!(
            &bridged.0.0[..],
            b"proxima!",
            "bytes actually written must match the source payload"
        );
    }

    #[test]
    fn write_flush_close_happy_path() {
        let mut backing = [0u8; 8];
        let mut context = Context::from_waker(Waker::noop());

        {
            let mut writer = SliceWriter {
                buf: &mut backing,
                pos: 0,
            };

            let write_outcome = Pin::new(&mut writer).poll_write(&mut context, b"proxima!");
            assert_eq!(
                write_outcome,
                Poll::Ready(Ok(8)),
                "writer must accept the full payload"
            );

            let flush_outcome = Pin::new(&mut writer).poll_flush(&mut context);
            assert_eq!(
                flush_outcome,
                Poll::Ready(Ok(())),
                "flush must complete synchronously for an in-memory sink"
            );

            let close_outcome = Pin::new(&mut writer).poll_close(&mut context);
            assert_eq!(
                close_outcome,
                Poll::Ready(Ok(())),
                "close must complete synchronously for an in-memory sink"
            );
        }

        assert_eq!(
            &backing, b"proxima!",
            "bytes actually written must match the source payload"
        );
    }

    #[test]
    fn pending_registers_waker_then_makes_progress() {
        let data = b"waking";
        let mut reader = PendingOnceReader {
            inner: SliceReader::new(data),
            pending_remaining: 1,
        };
        let mut context = Context::from_waker(Waker::noop());
        let mut buf = [0u8; 6];

        let first_poll = Pin::new(&mut reader).poll_read(&mut context, &mut buf);
        assert_eq!(
            first_poll,
            Poll::Pending,
            "reader not yet ready must return pending and register the waker"
        );

        let second_poll = Pin::new(&mut reader).poll_read(&mut context, &mut buf);
        assert_eq!(
            second_poll,
            Poll::Ready(Ok(6)),
            "reader must make progress on the next poll once ready"
        );
        assert_eq!(
            &buf, data,
            "bytes read after pending must match source data"
        );
    }
}
