//! `DrainSink` — the zero-copy push **sink**, the dual of [`crate::pipe::drain_source::DrainSource`].
//!
//! Where a `DrainSource` *lends* borrowed items out (`drain_ready` pushes
//! `&Item` into a visitor), a `DrainSink` *accepts* borrowed items in
//! (`accept(&Item)`) and writes them into its own storage — a ring slot, an
//! mmap region, an NVMe queue entry — with **no owned copy**. The borrow is
//! scoped to `accept`; the item never escapes.
//!
//! This is the missing leg of the Source/Sink symmetry: the owned push sink is
//! already `SendPipe<In=Item, Out=()>` (`FanOut` is its 1→N fan), so no owned
//! sink trait is minted here. The *zero-copy* push sink cannot be expressed by
//! `SendPipe` (its `In` is owned/sized), so it is a thin trait — exactly as the
//! source side needed `DrainSource` distinct from `FanIn`'s `UnpinPipe` sources.
//!
//! TIER: T0 — no_std + no-alloc. The `*DK` push shape: a frame is a `&[u8]`
//! view written straight into a stack ring (`RingSink`), no heap, no copy
//! beyond the unavoidable slot write. Owned `emit([u8; N])` would force a copy
//! *into* the item before the call; `accept(&[u8])` writes the slot directly.

use core::ops::ControlFlow;

/// A sink that accepts borrowed items in place. The zero-copy dual of
/// [`crate::pipe::drain_source::DrainSource`].
///
/// `accept` returns [`ControlFlow`]: `Continue` = took it, keep pushing;
/// `Break` = full / backpressured, the caller should pause or shed. `?Sized`
/// `Item` mirrors `DrainSource::Item` so `[u8]` slices are first-class.
pub trait DrainSink {
    /// The item accepted, borrowed. `?Sized` so it can be `[u8]`.
    type Item: ?Sized;

    /// Accept one borrowed item. `Break` signals backpressure (no room).
    fn accept(&mut self, item: &Self::Item) -> ControlFlow<()>;

    /// Whether the sink can take at least one more item right now — lets a
    /// caller avoid a `Break` round-trip. Default `true`.
    #[must_use]
    fn has_capacity(&self) -> bool {
        true
    }
}

/// A fixed-capacity ring sink over a stack arena — the dual of
/// [`crate::pipe::drain_source::RingSource`]. `RingSource` lends `&[u8]` out of its
/// arena; `RingSink` writes `&[u8]` into one. Together: a zero-copy, no-heap
/// relay.
pub struct RingSink<const SLOTS: usize, const SLOT: usize> {
    arena: [[u8; SLOT]; SLOTS],
    lengths: [usize; SLOTS],
    read: usize,
    write: usize,
    count: usize,
}

impl<const SLOTS: usize, const SLOT: usize> Default for RingSink<SLOTS, SLOT> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const SLOTS: usize, const SLOT: usize> RingSink<SLOTS, SLOT> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            arena: [[0u8; SLOT]; SLOTS],
            lengths: [0usize; SLOTS],
            read: 0,
            write: 0,
            count: 0,
        }
    }

    #[must_use]
    pub fn is_full(&self) -> bool {
        self.count == SLOTS
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.count
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Pop the oldest frame (the read side — a relay worker drains here, e.g.
    /// to feed a `RingSource` or an async exporter). Borrows into the arena.
    #[must_use]
    pub fn pop(&mut self) -> Option<&[u8]> {
        if self.count == 0 {
            return None;
        }
        let slot = self.read % SLOTS;
        self.read = self.read.wrapping_add(1);
        self.count -= 1;
        Some(&self.arena[slot][..self.lengths[slot]])
    }
}

impl<const SLOTS: usize, const SLOT: usize> DrainSink for RingSink<SLOTS, SLOT> {
    type Item = [u8];

    fn accept(&mut self, frame: &[u8]) -> ControlFlow<()> {
        if self.count == SLOTS || frame.len() > SLOT {
            return ControlFlow::Break(());
        }
        let slot = self.write % SLOTS;
        self.arena[slot][..frame.len()].copy_from_slice(frame);
        self.lengths[slot] = frame.len();
        self.write = self.write.wrapping_add(1);
        self.count += 1;
        ControlFlow::Continue(())
    }

    fn has_capacity(&self) -> bool {
        !self.is_full()
    }
}

/// The one real error [`RingSink`]'s `AsyncWrite` impl can report: a write
/// that can never fit (`frame.len() > SLOT`) is permanent, not transient — a
/// full-but-fits ring reports backpressure via `Pending` instead (see the
/// impl's doc for the poll-mode caveat this shares with every T0 sink here).
#[cfg(feature = "io-async")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RingSinkWriteError {
    /// `len` exceeds the ring's fixed `slot` capacity; no retry will help.
    #[error("write of {len} bytes exceeds the ring's {slot}-byte slot capacity")]
    FrameTooLarge {
        /// The rejected write's length.
        len: usize,
        /// The ring's fixed per-slot capacity (`SLOT`).
        slot: usize,
    },
}

// direct impl, not a wrapper: `RingSink` already IS the byte-stream state
// (arena + write cursor), so it composes `proxima_core::io::AsyncWrite`
// straight, mirroring `RingSource`'s `AsyncRead` (same LOCKED
// part-source-sink-design ruling; see that impl's doc in `drain_source.rs`).
// This is the FLOOR form (canonical at no_std/no-alloc, where `futures::io`
// cannot compile at all — see `proxima_core::io`'s module doc); the std-tier
// sibling below forwards to it.
#[cfg(feature = "io-async")]
impl<const SLOTS: usize, const SLOT: usize> proxima_core::io::AsyncWrite for RingSink<SLOTS, SLOT> {
    type Error = RingSinkWriteError;

    /// Poll-mode caveat (shared with `RingSource::poll_read`): a full-but-fits
    /// ring returns `Pending` WITHOUT arming a wake — busy-poll, matching
    /// every other T0 sink here ([`DrainFanOut`]).
    fn poll_write(
        self: core::pin::Pin<&mut Self>,
        _cx: &mut core::task::Context<'_>,
        buf: &[u8],
    ) -> core::task::Poll<Result<usize, Self::Error>> {
        if buf.len() > SLOT {
            return core::task::Poll::Ready(Err(RingSinkWriteError::FrameTooLarge {
                len: buf.len(),
                slot: SLOT,
            }));
        }
        match self.get_mut().accept(buf) {
            ControlFlow::Continue(()) => core::task::Poll::Ready(Ok(buf.len())),
            ControlFlow::Break(()) => core::task::Poll::Pending,
        }
    }

    fn poll_flush(
        self: core::pin::Pin<&mut Self>,
        _cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Result<(), Self::Error>> {
        // a write already lands in its ring slot; there is no separate flush
        // stage to drive.
        core::task::Poll::Ready(Ok(()))
    }

    fn poll_close(
        self: core::pin::Pin<&mut Self>,
        _cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Result<(), Self::Error>> {
        core::task::Poll::Ready(Ok(()))
    }
}

// std-tier ecosystem bridge: `futures::io::AsyncWrite` is the workspace's
// canonical std-tier trait (see `proxima_core::io`'s module doc for the
// full tier rule), so a std caller holding a `RingSink` reaches for THIS
// impl. No new logic: forwards to the already-tested floor `poll_write`/
// `poll_flush`/`poll_close` above, converting `RingSinkWriteError` into
// `std::io::Error` via `Error::other` (the one real error this sink can
// report — a write that can never fit).
#[cfg(all(feature = "io-async", feature = "std"))]
impl<const SLOTS: usize, const SLOT: usize> futures::io::AsyncWrite for RingSink<SLOTS, SLOT> {
    fn poll_write(
        self: core::pin::Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
        buf: &[u8],
    ) -> core::task::Poll<std::io::Result<usize>> {
        match <Self as proxima_core::io::AsyncWrite>::poll_write(self, cx, buf) {
            core::task::Poll::Ready(Ok(count)) => core::task::Poll::Ready(Ok(count)),
            core::task::Poll::Ready(Err(error)) => {
                core::task::Poll::Ready(Err(std::io::Error::other(error)))
            }
            core::task::Poll::Pending => core::task::Poll::Pending,
        }
    }

    fn poll_flush(
        self: core::pin::Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<std::io::Result<()>> {
        match <Self as proxima_core::io::AsyncWrite>::poll_flush(self, cx) {
            core::task::Poll::Ready(Ok(())) => core::task::Poll::Ready(Ok(())),
            core::task::Poll::Ready(Err(error)) => {
                core::task::Poll::Ready(Err(std::io::Error::other(error)))
            }
            core::task::Poll::Pending => core::task::Poll::Pending,
        }
    }

    fn poll_close(
        self: core::pin::Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<std::io::Result<()>> {
        match <Self as proxima_core::io::AsyncWrite>::poll_close(self, cx) {
            core::task::Poll::Ready(Ok(())) => core::task::Poll::Ready(Ok(())),
            core::task::Poll::Ready(Err(error)) => {
                core::task::Poll::Ready(Err(std::io::Error::other(error)))
            }
            core::task::Poll::Pending => core::task::Poll::Pending,
        }
    }
}

/// 1→N zero-copy fan-out over [`DrainSink`]s — the dual of
/// [`crate::pipe::drain_source::DrainFanIn`]. `DrainFanIn` routes N sources → 1
/// visitor; `DrainFanOut` routes 1 borrowed item → N sinks. Fixed-arity, T0.
pub struct DrainFanOut<K, const N: usize> {
    sinks: [K; N],
}

impl<K, const N: usize> DrainFanOut<K, N> {
    #[must_use]
    pub const fn new(sinks: [K; N]) -> Self {
        Self { sinks }
    }

    /// The sinks (for the relay worker to drain each).
    #[must_use]
    pub fn sinks_mut(&mut self) -> &mut [K; N] {
        &mut self.sinks
    }
}

impl<K: DrainSink, const N: usize> DrainFanOut<K, N> {
    /// Push `item` into every sink, all-or-nothing: stop at the first `Break`
    /// (a full sink) and report it. The earlier sinks have already taken it.
    pub fn push_all(&mut self, item: &K::Item) -> ControlFlow<()> {
        for sink in &mut self.sinks {
            sink.accept(item)?;
        }
        ControlFlow::Continue(())
    }

    /// Best-effort: push into every sink that has room, skip the full ones.
    pub fn push_best_effort(&mut self, item: &K::Item) {
        for sink in &mut self.sinks {
            if sink.has_capacity() {
                let _ = sink.accept(item);
            }
        }
    }
}

// marker propagation — RingSink / DrainFanOut carry the markers their parts do
// (mirrors the AndThen propagation in primitives and DropSafe on the source side).
mod marker_propagation {
    use super::{DrainFanOut, DrainSink, RingSink};
    use proxima_core::markers::{AllocFree, DropSafe, NoStd};

    impl<const SLOTS: usize, const SLOT: usize> NoStd for RingSink<SLOTS, SLOT> {}
    impl<const SLOTS: usize, const SLOT: usize> AllocFree for RingSink<SLOTS, SLOT> {}
    impl<const SLOTS: usize, const SLOT: usize> DropSafe for RingSink<SLOTS, SLOT> {}

    impl<K: DrainSink + NoStd, const N: usize> NoStd for DrainFanOut<K, N> {}
    impl<K: DrainSink + AllocFree, const N: usize> AllocFree for DrainFanOut<K, N> {}
    impl<K: DrainSink + DropSafe, const N: usize> DropSafe for DrainFanOut<K, N> {}
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn ring_sink_accepts_then_pops_in_order() {
        let mut sink: RingSink<4, 8> = RingSink::new();
        assert_eq!(sink.accept(b"hello"), ControlFlow::Continue(()));
        assert_eq!(sink.accept(b"world"), ControlFlow::Continue(()));
        assert_eq!(sink.len(), 2);
        assert_eq!(sink.pop(), Some(&b"hello"[..]));
        assert_eq!(sink.pop(), Some(&b"world"[..]));
        assert_eq!(sink.pop(), None);
    }

    #[test]
    fn ring_sink_breaks_when_full_no_panic() {
        let mut sink: RingSink<1, 4> = RingSink::new();
        assert_eq!(sink.accept(b"abcd"), ControlFlow::Continue(()));
        assert!(!sink.has_capacity());
        assert_eq!(
            sink.accept(b"efgh"),
            ControlFlow::Break(()),
            "full sink => Break"
        );
    }

    #[test]
    fn ring_sink_breaks_on_oversize_frame() {
        let mut sink: RingSink<4, 4> = RingSink::new();
        assert_eq!(
            sink.accept(b"toolong!!"),
            ControlFlow::Break(()),
            "frame > SLOT => Break"
        );
        assert!(sink.is_empty());
    }

    #[test]
    fn drain_fan_out_routes_item_to_all_sinks() {
        let mut fan: DrainFanOut<RingSink<4, 8>, 3> =
            DrainFanOut::new([RingSink::new(), RingSink::new(), RingSink::new()]);
        assert_eq!(fan.push_all(b"ping"), ControlFlow::Continue(()));
        for sink in fan.sinks_mut() {
            assert_eq!(sink.pop(), Some(&b"ping"[..]));
        }
    }

    #[test]
    fn drain_fan_out_best_effort_skips_full_sinks() {
        // sink 0 cap 1 (will fill), sink 1 cap 4 (room) — heterogeneous via same type, cap 1.
        let mut fan: DrainFanOut<RingSink<1, 8>, 2> =
            DrainFanOut::new([RingSink::new(), RingSink::new()]);
        fan.push_best_effort(b"a"); // both take it, both now full
        fan.push_best_effort(b"b"); // both full -> both skipped
        for sink in fan.sinks_mut() {
            assert_eq!(sink.pop(), Some(&b"a"[..]));
            assert_eq!(sink.pop(), None, "second push was skipped (full)");
        }
    }

    #[cfg(feature = "io-async")]
    #[test]
    fn ring_sink_async_write_accepts_frames_then_reports_oversize_error() {
        use core::pin::Pin;
        use core::task::{Context, Poll, Waker};

        use proxima_core::io::AsyncWrite as _;

        const REQUEST_LINE: &[u8] = b"GET /orders?id=42 HTTP/1.1\r\n";

        let mut sink: RingSink<2, 32> = RingSink::new();
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);

        let written = match Pin::new(&mut sink).poll_write(&mut cx, REQUEST_LINE) {
            Poll::Ready(Ok(count)) => count,
            other => panic!("expected an immediate accept, got {other:?}"),
        };
        assert_eq!(written, REQUEST_LINE.len());
        assert_eq!(sink.pop(), Some(REQUEST_LINE), "the write reached the ring");

        let oversize = [0u8; 64];
        match Pin::new(&mut sink).poll_write(&mut cx, &oversize) {
            Poll::Ready(Err(RingSinkWriteError::FrameTooLarge { len, slot })) => {
                assert_eq!(len, oversize.len());
                assert_eq!(slot, 32);
            }
            other => panic!("expected FrameTooLarge, got {other:?}"),
        }
    }
}
