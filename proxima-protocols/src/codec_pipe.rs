//! [`FrameCodecPipe<C>`] — the GENERIC adapter proving ANY
//! [`proxima_codec::FrameCodec`] composes directly as a
//! `proxima_primitives::pipe::Pipe` with no codec rewrite,
//! not just [`crate::http1_codec::codec_trait::H1RequestCodec`].
//!
//! Generalized from the H1-only spike
//! (`http1_codec::frame_codec_pipe::FrameCodecPipe`, see git history):
//! that version hardcoded `inner: H1RequestCodec` and re-owned
//! `RequestHead` fields by hand. The two things that varied per codec
//! were (1) how a borrowed `Frame<'a>` re-owns itself past the
//! borrow (each codec's `Frame` shape is different — a `RequestHead`
//! has method/path/headers, a gRPC `Frame` has a compression flag +
//! payload, ...) and (2) how a codec's `Error` signals "not enough
//! bytes yet" vs a hard parse failure (each codec names this
//! differently — `H1RequestCodec`'s `FrameError::Partial`, gRPC's
//! `ParseError::Short`/`PartialPayload`, ...). [`OwnFrame`] and
//! [`Incomplete`] are exactly those two per-codec seams; this adapter
//! is generic over both, so `FrameCodecPipe<C>` needs writing ONCE
//! (RISC, guiding-principle 1) and each codec supplies a small,
//! codec-specific `OwnFrame`/`Incomplete` impl — the same shape as
//! `AndThen`'s own `Second::Err: From<First::Err>` composition seam.
//!
//! Mirrors `http3_codec::qpack::part_source`'s precedent: a
//! `proxima-protocols` module optionally depending on
//! `proxima-primitives` to prove an existing sans-IO engine already
//! satisfies a `pipe` primitive, rather than inventing a new trait
//! hierarchy.
//!
//! `FrameCodecPipe::call` returns ONE frame per invocation
//! (`Option<(C::Owned, usize)>`, never a batching `Vec`) — a caller
//! with more than one frame buffered advances by `consumed` and calls
//! again, mirroring [`proxima_codec::FrameCodec::parse_frame`]'s own
//! one-frame-per-call contract.

use core::future::Future;

use bytes::Bytes;
use proxima_codec::{Addressed, Datagram, FrameCodec};
use proxima_primitives::pipe::{Pipe, SendPipe};

/// Bridges a borrowed `C::Frame<'a>` into an owned value backed by the
/// SAME `Bytes` allocation the input window came from — the borrow
/// crossing a `Pipe::call` must pay (`Pipe::Out` cannot borrow from
/// `Pipe::In` once `call` returns it by value; see
/// `proxima_primitives::pipe::Pipe`'s RPITIT contract).
/// Every real impl re-slices the source `Bytes` (`Bytes::slice_ref` or
/// equivalent) rather than copying — an `Arc` refcount bump over
/// already-refcounted storage, not a fresh byte copy.
pub trait OwnFrame: FrameCodec {
    /// The owned counterpart of `Self::Frame<'_>`.
    type Owned;

    /// Re-own one parsed frame given the `Bytes` window it was parsed from.
    fn own_frame(source: &Bytes, frame: &Self::Frame<'_>) -> Self::Owned;
}

/// Whether a [`FrameCodec::Error`] means "the buffer does not hold a
/// complete frame yet — read more and retry" (mapped to `Ok(None)` by
/// [`FrameCodecPipe`]) as opposed to a hard parse failure (mapped to
/// `Err`). Every sans-IO frame codec already models this distinction
/// as a value in its own error enum (a `Partial`/`Incomplete`/`Short`
/// variant); this trait just projects it to a `bool` so the generic
/// adapter can act on it without knowing the concrete enum.
pub trait Incomplete {
    fn is_incomplete(&self) -> bool;
}

/// `Pipe<In = Bytes, Out = Option<(C::Owned, usize)>, Err = C::Error>`
/// over ANY [`FrameCodec`] `C` that supplies [`OwnFrame`] +
/// [`Incomplete`]. Zero-sized when `C` is; clone freely.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameCodecPipe<C> {
    inner: C,
}

impl<C> FrameCodecPipe<C> {
    #[must_use]
    pub const fn new(inner: C) -> Self {
        Self { inner }
    }
}

impl<C> Pipe for FrameCodecPipe<C>
where
    C: OwnFrame,
    C::Error: Incomplete,
{
    type In = Bytes;
    type Out = Option<(C::Owned, usize)>;
    type Err = C::Error;

    fn call(&self, input: Bytes) -> impl Future<Output = Result<Self::Out, Self::Err>> {
        async move {
            match self.inner.parse_frame(&input) {
                Ok((frame, consumed)) => Ok(Some((C::own_frame(&input, &frame), consumed))),
                Err(error) if error.is_incomplete() => Ok(None),
                Err(error) => Err(error),
            }
        }
    }
}

/// Cross-core form of [`FrameCodecPipe`] — needed to compose in an `AndThen`
/// alongside a cross-core combinator (`Retry`, ...), which is `SendPipe`-only
/// (see `proxima-net`'s `pipe_connection` module). `C::Owned: Send` is the
/// only extra bound: `FrameCodec` itself already requires `Send + Sync +
/// 'static` (proto crates are cross-core by construction), so the codec and
/// its error are already covered.
impl<C> SendPipe for FrameCodecPipe<C>
where
    C: OwnFrame + Send + Sync + 'static,
    C::Error: Incomplete,
    C::Owned: Send,
{
    type In = Bytes;
    type Out = Option<(C::Owned, usize)>;
    type Err = C::Error;

    fn call(&self, input: Bytes) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            match self.inner.parse_frame(&input) {
                Ok((frame, consumed)) => Ok(Some((C::own_frame(&input, &frame), consumed))),
                Err(error) if error.is_incomplete() => Ok(None),
                Err(error) => Err(error),
            }
        }
    }
}

/// Lifts `App: Pipe<In = Frame, Out = AppOut>` into
/// `Pipe<In = Option<(Frame, usize)>, Out = Option<(AppOut, usize)>>` —
/// runs `App` only when the upstream [`FrameCodecPipe`] produced a
/// complete frame; a partial frame (`None`) passes straight through
/// with no call, and the byte-accounting (`usize`) rides along
/// unchanged. This is the seam that lets an `AndThen<FrameCodecPipe<C>,
/// OnFrame<App>>` stay ONE plain `Pipe` end to end, so a driver only
/// ever has to understand one contract ("`Option<(Out, usize)>` = wait
/// for more bytes or here is a result") no matter what `App` is —
/// including an `App` that is itself wrapped in a combinator (`Retry`,
/// `Filter`, ...): see `proxima-net`'s `pipe_connection` module for the
/// end-to-end proof.
#[derive(Debug, Clone, Copy, Default)]
pub struct OnFrame<App> {
    app: App,
}

impl<App> OnFrame<App> {
    #[must_use]
    pub const fn new(app: App) -> Self {
        Self { app }
    }
}

impl<App> Pipe for OnFrame<App>
where
    App: Pipe,
{
    type In = Option<(App::In, usize)>;
    type Out = Option<(App::Out, usize)>;
    type Err = App::Err;

    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>> {
        async move {
            match input {
                None => Ok(None),
                Some((frame, consumed)) => {
                    let out = self.app.call(frame).await?;
                    Ok(Some((out, consumed)))
                }
            }
        }
    }
}

/// Cross-core form of [`OnFrame`] — the seam that lets `App` itself be a
/// `SendPipe`-only combinator (`Retry<Inner>` only implements `SendPipe` for
/// an arbitrary `Inner`, never the base `Pipe`; see
/// `proxima_primitives::pipe::retry`). `App::In: Send` mirrors `AndThen`'s own
/// `First::In: Send` bound: an async fn's parameter must itself be `Send`
/// for the returned future to be `Send`, regardless of where it is awaited.
impl<App> SendPipe for OnFrame<App>
where
    App: SendPipe,
    App::In: Send,
{
    type In = Option<(App::In, usize)>;
    type Out = Option<(App::Out, usize)>;
    type Err = App::Err;

    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            match input {
                None => Ok(None),
                Some((frame, consumed)) => {
                    let out = self.app.call(frame).await?;
                    Ok(Some((out, consumed)))
                }
            }
        }
    }
}

/// Bridges a borrowed `C::Message<'_>` into an owned value backed by the
/// SAME `Bytes` allocation the input packet came from — the [`OwnFrame`]
/// seam, restated for [`Datagram`]: the borrow crossing a `Pipe::call`
/// boundary must pay (`Pipe::Out` cannot borrow from `Pipe::In` once
/// `call` returns it by value).
pub trait OwnMessage: Datagram {
    /// The owned counterpart of `Self::Message<'_>`.
    type Owned;

    /// Re-own one decoded message given the `Bytes` packet it was
    /// decoded from.
    fn own_message(source: &Bytes, message: &Self::Message<'_>) -> Self::Owned;
}

/// `Pipe<In = Addressed<Bytes>, Out = Addressed<C::Owned>, Err = C::Error>`
/// over ANY [`Datagram`] `C` that supplies [`OwnMessage`] — the
/// [`FrameCodecPipe`] shape, restated for datagrams. The one thing this
/// adapter deliberately does NOT do that [`FrameCodecPipe`] does: collapse
/// an [`Incomplete`] error into `Ok(None)`. A [`Datagram::decode`] failure
/// has no "read more and retry" meaning — every `Err` here is a hard,
/// per-packet failure, by construction (there is no `Incomplete` impl
/// bound on `C::Error` to even check). Zero-sized when `C` is; clone
/// freely.
#[derive(Debug, Clone, Copy, Default)]
pub struct DatagramPipe<C> {
    inner: C,
}

impl<C> DatagramPipe<C> {
    #[must_use]
    pub const fn new(inner: C) -> Self {
        Self { inner }
    }
}

impl<C> Pipe for DatagramPipe<C>
where
    C: OwnMessage,
{
    type In = Addressed<Bytes>;
    type Out = Addressed<C::Owned>;
    type Err = C::Error;

    fn call(&self, input: Addressed<Bytes>) -> impl Future<Output = Result<Self::Out, Self::Err>> {
        async move {
            let decoded = self.inner.decode(input.peer, &input.message)?;
            Ok(Addressed {
                peer: decoded.peer,
                message: C::own_message(&input.message, &decoded.message),
            })
        }
    }
}

/// Cross-core form of [`DatagramPipe`] — mirrors [`FrameCodecPipe`]'s own
/// `SendPipe` impl for the same reason (composing alongside a
/// `SendPipe`-only combinator). `C::Owned: Send` is the only extra bound:
/// `Datagram` itself already requires `Send + Sync + 'static`, and its
/// `Error` associated type already requires `Send + Sync + 'static`.
impl<C> SendPipe for DatagramPipe<C>
where
    C: OwnMessage + Send + Sync + 'static,
    C::Owned: Send,
{
    type In = Addressed<Bytes>;
    type Out = Addressed<C::Owned>;
    type Err = C::Error;

    fn call(
        &self,
        input: Addressed<Bytes>,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let decoded = self.inner.decode(input.peer, &input.message)?;
            Ok(Addressed {
                peer: decoded.peer,
                message: C::own_message(&input.message, &decoded.message),
            })
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use core::convert::Infallible;

    /// Dependency-free executor for the always-ready probe futures (mirrors
    /// `primitives.rs`'s own `block_on` test helper).
    fn block_on<Fut: Future>(future: Fut) -> Fut::Output {
        let mut pinned = core::pin::pin!(future);
        let mut context = core::task::Context::from_waker(core::task::Waker::noop());
        loop {
            if let core::task::Poll::Ready(output) = pinned.as_mut().poll(&mut context) {
                return output;
            }
        }
    }

    struct Double;

    impl Pipe for Double {
        type In = u32;
        type Out = u32;
        type Err = Infallible;

        fn call(&self, input: u32) -> impl Future<Output = Result<u32, Infallible>> {
            async move { Ok(input * 2) }
        }
    }

    #[test]
    fn on_frame_passes_none_through_without_calling_app() {
        let pipe = OnFrame::new(Double);
        let outcome = block_on(Pipe::call(&pipe, None)).expect("infallible");
        assert!(outcome.is_none());
    }

    #[test]
    fn on_frame_applies_app_and_keeps_consumed() {
        let pipe = OnFrame::new(Double);
        let outcome = block_on(Pipe::call(&pipe, Some((21u32, 7usize)))).expect("infallible");
        assert_eq!(outcome, Some((42, 7)));
    }

    impl SendPipe for Double {
        type In = u32;
        type Out = u32;
        type Err = Infallible;

        fn call(&self, input: u32) -> impl Future<Output = Result<u32, Infallible>> + Send {
            async move { Ok(input * 2) }
        }
    }

    #[test]
    fn on_frame_is_a_send_pipe_too() {
        let pipe = OnFrame::new(Double);
        let outcome = block_on(SendPipe::call(&pipe, Some((5u32, 1usize)))).expect("infallible");
        assert_eq!(outcome, Some((10, 1)));
    }

    use core::net::SocketAddr;

    /// Trivial [`Datagram`] used only by this module's own tests — a
    /// nonempty-buffer POD message, borrowed zero-copy.
    struct EchoDatagram;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct EmptyPacket;

    impl core::fmt::Display for EmptyPacket {
        fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(formatter, "empty datagram")
        }
    }

    impl core::error::Error for EmptyPacket {}

    impl Datagram for EchoDatagram {
        type Message<'a> = &'a [u8];
        type Error = EmptyPacket;

        fn decode<'a>(
            &self,
            peer: SocketAddr,
            bytes: &'a [u8],
        ) -> Result<Addressed<&'a [u8]>, EmptyPacket> {
            if bytes.is_empty() {
                return Err(EmptyPacket);
            }
            Ok(Addressed {
                peer,
                message: bytes,
            })
        }

        fn encode(
            &self,
            addressed: &Addressed<&[u8]>,
            dest: &mut alloc::vec::Vec<u8>,
        ) -> Result<(), EmptyPacket> {
            dest.extend_from_slice(addressed.message);
            Ok(())
        }
    }

    impl OwnMessage for EchoDatagram {
        type Owned = Bytes;

        fn own_message(source: &Bytes, message: &&[u8]) -> Bytes {
            source.slice_ref(message)
        }
    }

    fn loopback_peer() -> SocketAddr {
        SocketAddr::from((core::net::Ipv4Addr::LOCALHOST, 11211))
    }

    #[test]
    fn datagram_pipe_decodes_and_reowns_across_the_call_boundary() {
        let pipe = DatagramPipe::new(EchoDatagram);
        let peer = loopback_peer();
        let input = Addressed {
            peer,
            message: Bytes::from_static(b"hello"),
        };

        let outcome = block_on(Pipe::call(&pipe, input)).expect("decode should succeed");
        assert_eq!(outcome.peer, peer);
        assert_eq!(outcome.message, Bytes::from_static(b"hello"));
    }

    #[test]
    fn datagram_pipe_propagates_hard_error_with_no_incomplete_collapsing() {
        // FrameCodecPipe collapses an Incomplete error into Ok(None); a
        // Datagram has no such signal — Out is Addressed<C::Owned>, never
        // Option<_>, so every decode failure surfaces as a plain Err.
        let pipe = DatagramPipe::new(EchoDatagram);
        let input = Addressed {
            peer: loopback_peer(),
            message: Bytes::new(),
        };

        let outcome = block_on(Pipe::call(&pipe, input));
        assert_eq!(outcome, Err(EmptyPacket));
    }
}
