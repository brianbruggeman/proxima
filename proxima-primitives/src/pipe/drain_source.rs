//! `DrainSource` / `DrainFanIn` — zero-copy N→1 merge via the visitor-push model.
//!
//! The zero-copy counterpart to the owned [`crate::pipe::fan_in::FanIn`]: the merged
//! item *borrows* into the producing source's ring slot (`&[u8]`) instead of
//! being copied out. The borrow is scoped to a visitor closure call, so it
//! never escapes the merge loop.
//!
//! ## Why push, not a GAT lending pull
//!
//! The obvious zero-copy design is a GAT lending source
//! (`type Item<'a>; poll_next<'a>(&'a mut self) -> Poll<Option<Item<'a>>>`).
//! It was tried and **empirically falsified**: the array-FSM
//! `match self.sources[i].poll_next(cx) { Ready(Some(item)) => return … }`
//! fails to compile (`E0499` — the returned `Item<'a>` keeps `self.sources[i]`
//! borrowed for `'a`, conflicting with the next loop iteration re-polling it).
//! That is the classic lending-iterator-in-a-loop borrow that needs Polonius,
//! which is not stable. The visitor-push model sidesteps it entirely: the
//! `&Item` borrow lives only for the `visitor(item)` call inside
//! `drain_ready`, never crossing the loop boundary, so it compiles on stable
//! and stays zero-copy.
//!
//! ## Execution model
//!
//! This is the **poll-mode / spin-drain** surface (no `Waker`, no `Context`) —
//! the \*DK shape: a source drains the items ready *now* and returns; the caller
//! re-drives. It is distinct from the waker-registered async
//! [`crate::pipe::fan_in::FanIn`] (an `UnpinPipe` merge); the two coexist
//! (sync-zero-copy vs async-owned), bridged by a batching adapter where an
//! async consumer needs owned items.
//!
//! TIER: T0 — no_std + no-alloc. `&mut dyn FnMut` is a stack fat-pointer (no
//! heap); the consumer-facing `drain_each` takes `impl FnMut` (monomorphised),
//! so the only vtable hop is once per source-batch, not per item.

use core::ops::ControlFlow;

/// The outcome of one [`DrainSource::drain_ready`] pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainState {
    /// Items may be available on the next call — the source is not exhausted,
    /// it just ran out of ready items this pass (or the visitor stopped it).
    More,
    /// The source is permanently exhausted; it will not be polled again.
    Drained,
}

/// A poll-mode source that pushes borrowed items into a visitor.
///
/// The source applies `visitor` to each ready item while holding the borrow
/// internally — so the item can be a `?Sized` slice view (`[u8]`) into a ring
/// slot with zero copy. The visitor returns [`ControlFlow`]: `Continue` keeps
/// draining, `Break` stops (backpressure), in which case `drain_ready` returns
/// [`DrainState::More`] (the source is not exhausted, the consumer stopped it).
pub trait DrainSource {
    /// The item lent to the visitor. `?Sized` so it can be a `[u8]` slot view.
    type Item: ?Sized;

    /// Drain all currently-ready items, calling `visitor` on each (borrowed)
    /// item in order. No alloc, no syscall, no waker on the hot path.
    fn drain_ready(
        &mut self,
        visitor: &mut dyn FnMut(&Self::Item) -> ControlFlow<()>,
    ) -> DrainState;
}

/// Fixed-arity N→1 zero-copy merge over `[S; N]`. Round-robin fair; ends when
/// every source has drained. Shares the FSM shape of [`crate::pipe::fan_in::FanIn`]
/// (cursor + live-set + remaining) but drives sources push-style so items
/// borrow source slots without copying. No_std + no-alloc.
pub struct DrainFanIn<S, const N: usize> {
    sources: [S; N],
    live: [bool; N],
    remaining: usize,
    cursor: usize,
}

impl<S, const N: usize> DrainFanIn<S, N> {
    /// Merge `sources`. All start live.
    #[must_use]
    pub fn new(sources: [S; N]) -> Self {
        Self {
            sources,
            live: [true; N],
            remaining: N,
            cursor: 0,
        }
    }

    /// Sources not yet permanently drained.
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.remaining
    }

    /// Visit the live sources once each, round-robin from the cursor, draining
    /// each fully (the burst pattern: a source emits its ready items per visit,
    /// then the next source). Items are read in-place — zero copy. Returns when
    /// the visitor breaks (backpressure → [`DrainState::More`], resuming at the
    /// same source next call) or every source has drained.
    ///
    /// Cross-call fairness: the cursor advances by one each call so a perpetually
    /// busy source can't permanently win the first slot. `impl FnMut`
    /// monomorphises the consumer; the per-source `drain_ready` takes the
    /// `&mut dyn FnMut`, so the only vtable hop is once per source, not per item.
    pub fn drain_each<F>(&mut self, mut visitor: F) -> DrainState
    where
        S: DrainSource,
        F: FnMut(&S::Item) -> ControlFlow<()>,
    {
        if self.remaining == 0 {
            return DrainState::Drained;
        }
        // capture a STABLE start: mutating self.cursor mid-loop would corrupt
        // the index calc and skip sources. visitor + broke are non-self
        // captures, so they don't conflict with the &mut self.sources[index].
        let start = self.cursor;
        let mut any_more = false;
        for step in 0..N {
            let index = (start + step) % N;
            if !self.live[index] {
                continue;
            }
            let mut broke = false;
            let state = self.sources[index].drain_ready(&mut |item| {
                let flow = visitor(item);
                if flow.is_break() {
                    broke = true;
                }
                flow
            });
            match state {
                DrainState::Drained => {
                    self.live[index] = false;
                    self.remaining -= 1;
                    if self.remaining == 0 {
                        return DrainState::Drained;
                    }
                }
                DrainState::More => any_more = true,
            }
            if broke {
                self.cursor = index; // resume this source next call
                return DrainState::More;
            }
        }
        self.cursor = (start + 1) % N; // advance once for cross-call fairness
        if any_more {
            DrainState::More
        } else {
            DrainState::Drained
        }
    }
}

// Nesting: a merge is itself a source, so `DrainFanIn<DrainFanIn<R, 4>, 3>`
// composes with no extra machinery.
impl<S: DrainSource, const N: usize> DrainSource for DrainFanIn<S, N> {
    type Item = S::Item;

    fn drain_ready(
        &mut self,
        visitor: &mut dyn FnMut(&Self::Item) -> ControlFlow<()>,
    ) -> DrainState {
        self.drain_each(|item| visitor(item))
    }
}

/// A fixed-capacity ring of byte frames over a stack arena — the zero-copy \*DK
/// source (DPDK mbuf ring / NVMe CQ view / per-core telemetry ring). Items lent
/// to the visitor borrow directly into `arena`; no heap, no copy on the read.
pub struct RingSource<const SLOTS: usize, const SLOT: usize> {
    arena: [[u8; SLOT]; SLOTS],
    lengths: [usize; SLOTS],
    read: usize,
    write: usize,
    count: usize,
    closed: bool,
    // AsyncRead's byte-stream cursor into the CURRENT oldest frame — how far
    // a `poll_read` call has already delivered when the caller's buffer is
    // smaller than the frame. `DrainSource::drain_ready` never touches this
    // (it always hands back a whole frame per visitor call); the two
    // consumption models are not meant to interleave on the same instance —
    // pick one per `RingSource` (see the `io-async` impl below). Only exists
    // under `io-async` so the T0 floor build carries no dead field.
    #[cfg(feature = "io-async")]
    frame_offset: usize,
}

impl<const SLOTS: usize, const SLOT: usize> Default for RingSource<SLOTS, SLOT> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const SLOTS: usize, const SLOT: usize> RingSource<SLOTS, SLOT> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            arena: [[0u8; SLOT]; SLOTS],
            lengths: [0usize; SLOTS],
            read: 0,
            write: 0,
            count: 0,
            closed: false,
            #[cfg(feature = "io-async")]
            frame_offset: 0,
        }
    }

    /// Copy one frame into the next slot. `false` if full or oversized (the
    /// producer's concern — the ring never allocates).
    pub fn push(&mut self, frame: &[u8]) -> bool {
        if self.count == SLOTS || frame.len() > SLOT {
            return false;
        }
        let slot = self.write % SLOTS;
        self.arena[slot][..frame.len()].copy_from_slice(frame);
        self.lengths[slot] = frame.len();
        self.write = self.write.wrapping_add(1);
        self.count += 1;
        true
    }

    /// No more frames will arrive — the next fully-drained pass returns
    /// [`DrainState::Drained`].
    pub fn close(&mut self) {
        self.closed = true;
    }
}

impl<const SLOTS: usize, const SLOT: usize> DrainSource for RingSource<SLOTS, SLOT> {
    type Item = [u8];

    fn drain_ready(&mut self, visitor: &mut dyn FnMut(&[u8]) -> ControlFlow<()>) -> DrainState {
        while self.count > 0 {
            let slot = self.read % SLOTS;
            // zero-copy: borrow straight into the arena; valid for the visitor call.
            let view: &[u8] = &self.arena[slot][..self.lengths[slot]];
            self.read = self.read.wrapping_add(1);
            self.count -= 1;
            if visitor(view).is_break() {
                return DrainState::More;
            }
        }
        if self.closed {
            DrainState::Drained
        } else {
            DrainState::More
        }
    }
}

// direct impl, not a wrapper: `RingSource` already IS the byte-stream state
// (arena + read cursor), so it composes `proxima_core::io::AsyncRead`
// straight, per the LOCKED part-source-sink-design ruling (no blanket
// unifying `Pipe`/streaming io — `docs/proxima-pipe/part-source-sink-design.md`).
// This is the FLOOR form (canonical at no_std/no-alloc, where
// `futures::io` cannot compile at all — see `proxima_core::io`'s module
// doc); the std-tier sibling immediately below forwards to it.
#[cfg(feature = "io-async")]
impl<const SLOTS: usize, const SLOT: usize> proxima_core::io::AsyncRead for RingSource<SLOTS, SLOT> {
    // a read never truly fails here — the worst case is a short read (normal
    // `AsyncRead` behaviour), so there is no error to carry.
    type Error = core::convert::Infallible;

    /// Poll-mode caveat (`docs/pipe-to-metal/edges.md`'s reshape ruling): this
    /// ring has no waker registration, so an empty-but-open ring returns
    /// `Pending` WITHOUT arming a wake — busy-poll, not wake-driven, matching
    /// every other T0 source in this crate ([`FanIn`], [`DrainFanIn`]).
    fn poll_read(
        self: core::pin::Pin<&mut Self>,
        _cx: &mut core::task::Context<'_>,
        buf: &mut [u8],
    ) -> core::task::Poll<Result<usize, Self::Error>> {
        let this = self.get_mut();
        if this.count == 0 {
            return if this.closed {
                core::task::Poll::Ready(Ok(0))
            } else {
                core::task::Poll::Pending
            };
        }
        let slot = this.read % SLOTS;
        let frame = &this.arena[slot][this.frame_offset..this.lengths[slot]];
        let take = frame.len().min(buf.len());
        buf[..take].copy_from_slice(&frame[..take]);
        this.frame_offset += take;
        if this.frame_offset == this.lengths[slot] {
            this.frame_offset = 0;
            this.read = this.read.wrapping_add(1);
            this.count -= 1;
        }
        core::task::Poll::Ready(Ok(take))
    }
}

// std-tier ecosystem bridge: `futures::io::AsyncRead` is the workspace's
// canonical std-tier trait (`prime::os::net::TcpStream` implements it, and
// every real std transport binds it — see `proxima_core::io`'s own module
// doc for the full tier rule), so a std caller holding a `RingSource`
// reaches for THIS impl, not the floor one above. No new logic: it forwards
// straight into the already-tested floor `poll_read` (the `Infallible`
// error can never fire, so the conversion is an exhaustive match, not a
// guess) — one ring, one read loop, two trait doors for two tiers.
#[cfg(all(feature = "io-async", feature = "std"))]
impl<const SLOTS: usize, const SLOT: usize> futures::io::AsyncRead for RingSource<SLOTS, SLOT> {
    fn poll_read(
        self: core::pin::Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
        buf: &mut [u8],
    ) -> core::task::Poll<std::io::Result<usize>> {
        match <Self as proxima_core::io::AsyncRead>::poll_read(self, cx, buf) {
            core::task::Poll::Ready(Ok(count)) => core::task::Poll::Ready(Ok(count)),
            core::task::Poll::Ready(Err(never)) => match never {},
            core::task::Poll::Pending => core::task::Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    // OTLP-shaped frames: tag 0x0a (field 1, length-delimited) + len + body.
    fn ring<const SLOTS: usize, const SLOT: usize>(frames: &[&[u8]]) -> RingSource<SLOTS, SLOT> {
        let mut source = RingSource::new();
        for frame in frames {
            assert!(source.push(frame), "test ring overflow");
        }
        source.close();
        source
    }

    #[test]
    fn merges_two_rings_round_robin_zero_copy() {
        let core0 = ring::<4, 8>(&[&[0x0a, 0x01, 0x41], &[0x0a, 0x01, 0x42]]);
        let core1 = ring::<4, 8>(&[&[0x0a, 0x01, 0x43]]);
        let mut fan = DrainFanIn::new([core0, core1]);

        // collect the payload byte of each frame (in-place read, no owned copy
        // beyond the single byte we pull for the assertion).
        let mut got = [0u8; 8];
        let mut count = 0;
        let state = fan.drain_each(|frame: &[u8]| {
            got[count] = frame[2];
            count += 1;
            ControlFlow::Continue(())
        });

        assert_eq!(state, DrainState::Drained);
        // source-at-a-time burst drain from cursor 0: core0 fully (A,B) then core1 (C).
        assert_eq!(&got[..count], b"ABC");
    }

    #[test]
    fn visitor_break_is_backpressure_not_drain() {
        let core0 = ring::<4, 8>(&[&[1], &[2]]);
        let core1 = ring::<4, 8>(&[&[3]]);
        let mut fan = DrainFanIn::new([core0, core1]);
        let mut seen = 0;
        let state = fan.drain_each(|_frame: &[u8]| {
            seen += 1;
            ControlFlow::Break(())
        });
        assert_eq!(state, DrainState::More, "break = source not exhausted");
        assert_eq!(seen, 1, "stopped after the first item");
        assert_eq!(fan.live_count(), 2, "both sources still live");
    }

    #[test]
    fn open_empty_ring_is_more_closed_empty_is_drained() {
        let mut open: RingSource<2, 4> = RingSource::new();
        assert_eq!(
            open.drain_ready(&mut |_| ControlFlow::Continue(())),
            DrainState::More,
            "open empty ring may get more"
        );
        open.close();
        assert_eq!(
            open.drain_ready(&mut |_| ControlFlow::Continue(())),
            DrainState::Drained,
            "closed empty ring is drained"
        );
    }

    #[test]
    fn nested_merge_composes_without_new_types() {
        let level_a = DrainFanIn::new([ring::<2, 4>(&[&[1]]), ring::<2, 4>(&[&[2]])]);
        let level_b = DrainFanIn::new([ring::<2, 4>(&[&[3]]), ring::<2, 4>(&[&[4]])]);
        let mut root = DrainFanIn::new([level_a, level_b]);
        let mut count = 0;
        root.drain_each(|_| {
            count += 1;
            ControlFlow::Continue(())
        });
        assert_eq!(count, 4, "all four leaf frames reach the root visitor");
    }

    // real captured-shaped HTTP/1 request bytes, split across frames smaller
    // than the read buffer AND a read buffer smaller than a frame — exercises
    // both the whole-frame and the partial-frame (`frame_offset`) paths.
    #[cfg(feature = "io-async")]
    #[test]
    fn ring_source_async_read_reassembles_a_captured_http_request() {
        use core::pin::Pin;
        use core::task::{Context, Poll, Waker};

        use proxima_core::io::AsyncRead as _;

        const REQUEST_LINE: &[u8] = b"GET /orders?id=42 HTTP/1.1\r\n";
        const HOST_HEADER: &[u8] = b"Host: api.example.internal\r\n";

        let mut source: RingSource<4, 64> = RingSource::new();
        assert!(source.push(REQUEST_LINE));
        assert!(source.push(HOST_HEADER));
        source.close();

        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        let mut collected = [0u8; 128];
        let mut total = 0;
        // a 5-byte buffer forces the request line to be delivered across
        // several `poll_read` calls, proving `frame_offset` resumes correctly.
        loop {
            let mut chunk = [0u8; 5];
            match Pin::new(&mut source).poll_read(&mut cx, &mut chunk) {
                Poll::Ready(Ok(0)) => break,
                Poll::Ready(Ok(count)) => {
                    collected[total..total + count].copy_from_slice(&chunk[..count]);
                    total += count;
                }
                Poll::Ready(Err(never)) => match never {},
                Poll::Pending => panic!("closed ring with ready frames must not report Pending"),
            }
        }

        let mut expected = [0u8; 128];
        let mut expected_len = 0;
        expected[..REQUEST_LINE.len()].copy_from_slice(REQUEST_LINE);
        expected_len += REQUEST_LINE.len();
        expected[expected_len..expected_len + HOST_HEADER.len()].copy_from_slice(HOST_HEADER);
        expected_len += HOST_HEADER.len();

        assert_eq!(&collected[..total], &expected[..expected_len]);
    }
}
