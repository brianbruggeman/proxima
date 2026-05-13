//! Canonical tiered datagram-I/O buffers.
//!
//! ONE buffer pair every datagram listener composes, replacing the
//! hand-rolled `recv_storage`/`recv_meta` drain-arena and `send_arena`/
//! `send_spans` staging in `proxima-h3/src/native/listen.rs`. Two-tier by
//! design (principle 3): on the `alloc` tier the backings are [`Vec`] and
//! grow dynamically with load (a server never overflows a fixed cap); on the
//! bare `no_std + no-alloc` tier they are fixed-cap arrays sized by
//! `build.rs` from `proxima-core.toml` (principle 12). Same API on both
//! tiers — the [`DcidTable`](../../proxima_protocols/quic/endpoint) hashbrown/
//! heapless split, generalized.
//!
//! The I/O *drive* (fill from / drain to a socket) lives in
//! `proxima-stream`'s `DatagramSocketBatchExt`, not here — these are pure
//! buffers so they stay `no_std` and reusable in any I/O loop (DPDK/AF_XDP
//! drive the slab directly via [`RecvSlab::unfilled_slots_mut`] +
//! [`RecvSlab::commit_filled`]).
//!
//! Addresses use [`core::net::SocketAddr`] (no_std since Rust 1.77) — the
//! same type the socket layer hands back, so there is no conversion at the
//! boundary and no bespoke peer-address newtype (principle 1).

#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::fmt;
use core::net::{IpAddr, Ipv4Addr, SocketAddr};

use crate::sized;

const UNSPECIFIED_PEER: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0);

/// Zero-copy view of one received datagram, borrowed from a [`RecvSlab`].
///
/// The borrow keeps the slab immutable while the view is alive — the borrow
/// checker mechanically forbids `clear`/grow while any view is held (the
/// fill -> drain -> iterate -> clear ordering invariant on [`RecvSlab`]).
#[derive(Copy, Clone, Debug)]
pub struct RecvView<'slab> {
    pub bytes: &'slab [u8],
    pub peer: SocketAddr,
}

/// Index entry for one staged outbound datagram in a [`SendBatch`].
#[derive(Copy, Clone, Debug)]
pub struct SendSpan {
    /// Byte offset of the payload in the send arena.
    pub offset: u32,
    /// Payload length. `u16` suffices — UDP payload caps at 65507.
    pub len: u16,
    /// Destination address.
    pub peer: SocketAddr,
}

impl SendSpan {
    #[cfg(not(feature = "alloc"))]
    const fn placeholder() -> Self {
        Self {
            offset: 0,
            len: 0,
            peer: UNSPECIFIED_PEER,
        }
    }
}

/// Fallible batch-buffer errors — there is no `panic!`/`assert!` on any
/// hot path (the reason [`SendBatch`] does NOT wrap
/// [`ByteArena`](crate::arena::ByteArena), whose `append` asserts).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BatchError {
    /// No-alloc recv slab at capacity; the burst exceeded `INITIAL_CAP` slots.
    SlabFull { capacity: usize },
    /// Send arena would exceed the `u32` offset space (resets every tick).
    ArenaOverflow,
    /// No-alloc send arena out of bytes.
    ArenaFull { capacity: usize },
    /// No-alloc span table full.
    SpansFull { capacity: usize },
}

impl fmt::Display for BatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SlabFull { capacity } => write!(formatter, "recv slab full ({capacity} slots)"),
            Self::ArenaOverflow => formatter.write_str("send arena u32 offset space exhausted"),
            Self::ArenaFull { capacity } => write!(formatter, "send arena full ({capacity} bytes)"),
            Self::SpansFull { capacity } => write!(formatter, "send spans full ({capacity})"),
        }
    }
}

impl core::error::Error for BatchError {}

/// Fixed-slot receive slab — the iovec-addressable storage a batched
/// `recvmmsg` fills. `alloc` tier grows to drain a burst to empty; no-alloc
/// tier is hard-capped at `INITIAL_CAP` (overflow stays in the kernel buffer
/// and drains next iteration — bounded, not silently lost).
///
/// Not [`BufferPool`](crate::buffer::BufferPool): that hands out one `BytesMut` per
/// call (per-connection reads); `recvmmsg` needs N same-size slots
/// addressable as `&mut [&mut [u8]]` in ONE syscall, which a per-call pool
/// can't form without a collecting allocation. `RecvSlab` IS that slab.
///
/// Ordering invariant (enforced by `&self` vs `&mut self`): `clear` ->
/// fill/drain -> iterate [`filled_datagrams`](Self::filled_datagrams) ->
/// drop views -> `clear`. A `RecvView` cannot outlive a grow or clear.
pub struct RecvSlab<const SLOT_BYTES: usize, const INITIAL_CAP: usize> {
    #[cfg(feature = "alloc")]
    slots: Vec<[u8; SLOT_BYTES]>,
    #[cfg(not(feature = "alloc"))]
    slots: [[u8; SLOT_BYTES]; INITIAL_CAP],
    // Meta is stored in the batched-receive's NATIVE `(len, peer)` form so the
    // syscall writes straight into persistent slab storage via `unfilled_mut` —
    // no staging array, no second copy, no `u16` conversion on the hot path
    // (the incumbent hand-rolled loop wrote its meta in place; this matches it).
    #[cfg(feature = "alloc")]
    meta: Vec<(usize, SocketAddr)>,
    #[cfg(not(feature = "alloc"))]
    meta: [(usize, SocketAddr); INITIAL_CAP],
    filled: usize,
}

impl<const SLOT_BYTES: usize, const INITIAL_CAP: usize> RecvSlab<SLOT_BYTES, INITIAL_CAP> {
    /// New slab pre-grown to `INITIAL_CAP` slots.
    #[must_use]
    pub fn new() -> Self {
        #[cfg(feature = "alloc")]
        {
            Self {
                slots: alloc::vec![[0u8; SLOT_BYTES]; INITIAL_CAP],
                meta: alloc::vec![(0usize, UNSPECIFIED_PEER); INITIAL_CAP],
                filled: 0,
            }
        }
        #[cfg(not(feature = "alloc"))]
        {
            Self {
                slots: [[0u8; SLOT_BYTES]; INITIAL_CAP],
                meta: [(0usize, UNSPECIFIED_PEER); INITIAL_CAP],
                filled: 0,
            }
        }
    }

    /// Filled slot count since the last [`clear`](Self::clear).
    #[must_use]
    pub fn filled(&self) -> usize {
        self.filled
    }

    /// Current slot capacity (alloc: live allocation; no-alloc: `INITIAL_CAP`).
    #[must_use]
    pub fn capacity(&self) -> usize {
        #[cfg(feature = "alloc")]
        {
            self.slots.len()
        }
        #[cfg(not(feature = "alloc"))]
        {
            INITIAL_CAP
        }
    }

    /// Ensure at least `needed` slots exist. Alloc tier grows; no-alloc tier
    /// returns [`BatchError::SlabFull`] (never silently truncates).
    ///
    /// # Errors
    /// [`BatchError::SlabFull`] on the no-alloc tier when `needed > INITIAL_CAP`.
    pub fn ensure_capacity(&mut self, needed: usize) -> Result<(), BatchError> {
        #[cfg(feature = "alloc")]
        {
            if self.slots.len() < needed {
                self.slots.resize(needed, [0u8; SLOT_BYTES]);
                self.meta.resize(needed, (0usize, UNSPECIFIED_PEER));
            }
            Ok(())
        }
        #[cfg(not(feature = "alloc"))]
        {
            if needed > INITIAL_CAP {
                Err(BatchError::SlabFull {
                    capacity: INITIAL_CAP,
                })
            } else {
                Ok(())
            }
        }
    }

    /// Reset the filled cursor; retains the backing allocation for reuse.
    pub fn clear(&mut self) {
        self.filled = 0;
    }

    /// Zero-copy iterator over the filled datagrams. A batched receive writes
    /// at most `SLOT_BYTES` into each slot, so the stored length is already
    /// in-bounds — no per-view clamp on the read path.
    pub fn filled_datagrams(&self) -> impl Iterator<Item = RecvView<'_>> + '_ {
        (0..self.filled).map(move |index| {
            let (len, peer) = self.meta[index];
            RecvView {
                bytes: &self.slots[index][..len],
                peer,
            }
        })
    }

    /// Mutable borrow of the unfilled slot + meta suffix at the fill cursor —
    /// the iovec targets AND `(len, peer)` out-descriptors a batched
    /// `recvmmsg` fills in ONE syscall. The receive writes both straight into
    /// persistent storage; [`commit`](Self::commit) then just advances the
    /// cursor — no staging array, no second copy.
    pub fn unfilled_mut(&mut self) -> (&mut [[u8; SLOT_BYTES]], &mut [(usize, SocketAddr)]) {
        (
            &mut self.slots[self.filled..],
            &mut self.meta[self.filled..],
        )
    }

    /// Mutable borrow of the unfilled slot suffix only — for a DPDK/AF_XDP ring
    /// that fills bytes itself and supplies its meta via
    /// [`commit_filled`](Self::commit_filled).
    pub fn unfilled_slots_mut(&mut self) -> &mut [[u8; SLOT_BYTES]] {
        &mut self.slots[self.filled..]
    }

    /// Advance the fill cursor by `count` after a batched receive wrote into the
    /// region returned by [`unfilled_mut`](Self::unfilled_mut). O(1).
    pub fn commit(&mut self, count: usize) {
        self.filled += count;
    }

    /// Commit `count` slots whose `(len, peer)` descriptors a ring source
    /// produced separately (the DPDK/AF_XDP seam that uses
    /// [`unfilled_slots_mut`](Self::unfilled_slots_mut) for bytes). The drive
    /// layer uses the zero-copy [`unfilled_mut`](Self::unfilled_mut) +
    /// [`commit`](Self::commit) path instead.
    pub fn commit_filled(&mut self, count: usize, meta: &[(usize, SocketAddr)]) {
        let start = self.filled;
        self.meta[start..start + count].copy_from_slice(&meta[..count]);
        self.filled += count;
    }
}

impl<const SLOT_BYTES: usize, const INITIAL_CAP: usize> Default
    for RecvSlab<SLOT_BYTES, INITIAL_CAP>
{
    fn default() -> Self {
        Self::new()
    }
}

/// Send-staging arena + per-datagram span index for batched `sendmmsg`.
///
/// Owns its OWN byte backing (it deliberately does NOT wrap
/// [`ByteArena`](crate::arena::ByteArena), whose `append` asserts on `u32`
/// overflow — a production panic). [`try_append`](Self::try_append) is
/// fully fallible, so a full arena/span table logs-and-drops at the caller
/// instead of aborting. Alloc tier grows; no-alloc tier is fixed.
pub struct SendBatch<const SEND_BYTES: usize, const SPAN_CAP: usize> {
    #[cfg(feature = "alloc")]
    arena: Vec<u8>,
    #[cfg(not(feature = "alloc"))]
    arena: [u8; SEND_BYTES],
    #[cfg(feature = "alloc")]
    spans: Vec<SendSpan>,
    #[cfg(not(feature = "alloc"))]
    spans: [SendSpan; SPAN_CAP],
    cursor: usize,
    span_count: usize,
}

impl<const SEND_BYTES: usize, const SPAN_CAP: usize> SendBatch<SEND_BYTES, SPAN_CAP> {
    /// New staging batch pre-grown to `SEND_BYTES`.
    #[must_use]
    pub fn new() -> Self {
        #[cfg(feature = "alloc")]
        {
            Self {
                // capacity-reserved but empty: the alloc tier APPENDS (cursor is
                // always the arena end), matching the hand-rolled staging Vec's
                // `extend_from_slice` speed — no pre-zeroed indexed write.
                arena: Vec::with_capacity(SEND_BYTES),
                spans: Vec::new(),
                cursor: 0,
                span_count: 0,
            }
        }
        #[cfg(not(feature = "alloc"))]
        {
            Self {
                arena: [0u8; SEND_BYTES],
                spans: [SendSpan::placeholder(); SPAN_CAP],
                cursor: 0,
                span_count: 0,
            }
        }
    }

    /// Stage one outbound datagram (one `memcpy` into the stable backing —
    /// the cost of a contiguous `sendmmsg`-ready layout).
    ///
    /// # Errors
    /// [`BatchError`] when the `u32` offset space, the no-alloc byte cap, or
    /// the no-alloc span cap is exhausted. The caller logs and drops.
    pub fn try_append(&mut self, bytes: &[u8], peer: SocketAddr) -> Result<(), BatchError> {
        // single fallible guard: offsets live in `u32` (SendSpan.offset), so the
        // arena cap IS u32::MAX. checked_add stays 32-bit-safe; the filter folds
        // the offset-space limit into the same expression (no separate try_from).
        let end = self
            .cursor
            .checked_add(bytes.len())
            .filter(|&end| end <= u32::MAX as usize)
            .ok_or(BatchError::ArenaOverflow)?;
        #[cfg(not(feature = "alloc"))]
        {
            if end > SEND_BYTES {
                return Err(BatchError::ArenaFull {
                    capacity: SEND_BYTES,
                });
            }
            if self.span_count >= SPAN_CAP {
                return Err(BatchError::SpansFull { capacity: SPAN_CAP });
            }
        }
        // alloc tier appends (cursor == arena.len()); no-alloc writes in place.
        #[cfg(feature = "alloc")]
        self.arena.extend_from_slice(bytes);
        #[cfg(not(feature = "alloc"))]
        self.arena[self.cursor..end].copy_from_slice(bytes);
        let span = SendSpan {
            offset: self.cursor as u32,
            len: bytes.len() as u16,
            peer,
        };
        #[cfg(feature = "alloc")]
        self.spans.push(span);
        #[cfg(not(feature = "alloc"))]
        {
            self.spans[self.span_count] = span;
        }
        self.cursor = end;
        self.span_count += 1;
        Ok(())
    }

    /// Reset for the next iteration; retains the backing allocation.
    pub fn reset(&mut self) {
        self.cursor = 0;
        self.span_count = 0;
        #[cfg(feature = "alloc")]
        {
            self.arena.clear();
            self.spans.clear();
        }
    }

    /// Whether any datagrams are staged.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.span_count == 0
    }

    /// Count of staged datagrams.
    #[must_use]
    pub fn len(&self) -> usize {
        self.span_count
    }

    /// The staged spans (the drive layer builds iovecs from these).
    #[must_use]
    pub fn spans(&self) -> &[SendSpan] {
        &self.spans[..self.span_count]
    }

    /// Borrow one span's payload bytes from the arena.
    #[must_use]
    pub fn slice_for(&self, span: &SendSpan) -> &[u8] {
        let start = span.offset as usize;
        &self.arena[start..start + span.len as usize]
    }
}

impl<const SEND_BYTES: usize, const SPAN_CAP: usize> Default for SendBatch<SEND_BYTES, SPAN_CAP> {
    fn default() -> Self {
        Self::new()
    }
}

/// The canonical per-socket datagram buffer pair: a [`RecvSlab`] + a
/// [`SendBatch`]. One per worker socket (a QUIC endpoint multiplexes every
/// connection over it), so at default sizing 8 workers hold ~1 MiB total —
/// far under the 500 MB / 10k-connection budget.
pub struct DatagramBatch<
    const SLOT_BYTES: usize,
    const INITIAL_CAP: usize,
    const SEND_BYTES: usize,
    const SPAN_CAP: usize,
> {
    pub recv: RecvSlab<SLOT_BYTES, INITIAL_CAP>,
    pub send: SendBatch<SEND_BYTES, SPAN_CAP>,
}

impl<
    const SLOT_BYTES: usize,
    const INITIAL_CAP: usize,
    const SEND_BYTES: usize,
    const SPAN_CAP: usize,
> DatagramBatch<SLOT_BYTES, INITIAL_CAP, SEND_BYTES, SPAN_CAP>
{
    #[must_use]
    pub fn new() -> Self {
        Self {
            recv: RecvSlab::new(),
            send: SendBatch::new(),
        }
    }
}

impl<
    const SLOT_BYTES: usize,
    const INITIAL_CAP: usize,
    const SEND_BYTES: usize,
    const SPAN_CAP: usize,
> Default for DatagramBatch<SLOT_BYTES, INITIAL_CAP, SEND_BYTES, SPAN_CAP>
{
    fn default() -> Self {
        Self::new()
    }
}

/// Build-time-sized default, parameters from `proxima-core.toml [batch]`.
pub type DefaultDatagramBatch = DatagramBatch<
    { sized::BATCH_RECV_SLOT_BYTES },
    { sized::BATCH_RECV_INITIAL_CAP },
    { sized::BATCH_SEND_ARENA_INITIAL_BYTES },
    { sized::BATCH_SEND_SPAN_CAP },
>;

#[cfg(all(test, feature = "alloc"))]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // a real QUIC v1 Initial: long-header first byte 0xc0 + version 0x00000001.
    const QUIC_INITIAL: &[u8] = &[0xc0, 0x00, 0x00, 0x00, 0x01, 0x08, 0xde, 0xad];
    const PEER_A: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 4433);
    const PEER_B: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 4434);

    #[test]
    fn recv_slab_commit_then_filled_datagrams_yields_zero_copy_views_in_order() {
        let mut slab = RecvSlab::<2048, 4>::new();
        slab.unfilled_slots_mut()[0][..QUIC_INITIAL.len()].copy_from_slice(QUIC_INITIAL);
        slab.unfilled_slots_mut()[1][..3].copy_from_slice(b"ack");
        slab.commit_filled(2, &[(QUIC_INITIAL.len(), PEER_A), (3, PEER_B)]);
        let views: Vec<_> = slab.filled_datagrams().collect();
        assert_eq!(views.len(), 2);
        assert_eq!(views[0].bytes, QUIC_INITIAL);
        assert_eq!(views[0].peer, PEER_A);
        assert_eq!(views[1].bytes, b"ack");
        assert_eq!(views[1].peer, PEER_B);
    }

    #[test]
    fn recv_slab_grows_past_initial_cap_to_drain_a_burst() {
        let mut slab = RecvSlab::<64, 2>::new();
        slab.ensure_capacity(6).unwrap();
        assert!(slab.capacity() >= 6, "alloc tier grows past INITIAL_CAP");
        slab.clear();
        assert_eq!(slab.filled(), 0);
    }

    #[test]
    fn send_batch_try_append_and_spans_round_trip_through_the_arena() {
        let mut batch = SendBatch::<4096, 256>::new();
        batch.try_append(QUIC_INITIAL, PEER_A).unwrap();
        batch.try_append(b"second", PEER_B).unwrap();
        assert_eq!(batch.len(), 2);
        let spans: Vec<_> = batch.spans().to_vec();
        assert_eq!(batch.slice_for(&spans[0]), QUIC_INITIAL);
        assert_eq!(spans[0].peer, PEER_A);
        assert_eq!(batch.slice_for(&spans[1]), b"second");
        assert_eq!(spans[1].peer, PEER_B);
    }

    #[test]
    fn send_batch_reset_retains_allocation_and_clears_spans() {
        let mut batch = SendBatch::<4096, 256>::new();
        batch.try_append(QUIC_INITIAL, PEER_A).unwrap();
        batch.reset();
        assert!(batch.is_empty());
        batch.try_append(b"after-reset", PEER_B).unwrap();
        let spans: Vec<_> = batch.spans().to_vec();
        assert_eq!(batch.slice_for(&spans[0]), b"after-reset");
    }

    #[test]
    fn default_datagram_batch_constructs_from_sized_consts() {
        let batch = DefaultDatagramBatch::new();
        assert_eq!(batch.recv.capacity(), sized::BATCH_RECV_INITIAL_CAP);
        assert!(batch.send.is_empty());
    }
}
