//! completion-based io_uring reactor for prime workers.
//!
//! unlike the readiness reactor (epoll/kqueue), io_uring delivers *completion
//! events* — each submitted operation (accept, read, write, shutdown) has a
//! CQE that carries the result. this reactor manages the IoUring ring, a slab
//! of pending operations, and per-operation waker storage.
//!
//! ## driving the ring
//!
//! the opportunistic-drain approach: each `poll_accept`/`poll_read`/
//! `poll_write` call invokes `PrimeUringReactor::drain_cqes()` with
//! `ring.submit()` before consulting the result slab. this is non-blocking
//! and catches completions that arrived since the previous poll. correct
//! (smoke tests pass) but suboptimal under load — a park-hook wired into
//! CoreShard is the perf follow-on.
//!
//! ## thread locality
//!
//! `PrimeUringReactor` is `!Send` (enforced by `PhantomData<*mut ()>`). the
//! thread-local `CURRENT_URING` is `UnsafeCell<Option<PrimeUringReactor>>`,
//! initialised lazily on first use. `TcpListener`/`TcpStream` call
//! `with_current_uring` on every poll — no cached pointer needed because
//! the TLS access is already lock-free.

#![cfg(all(
    target_os = "linux",
    feature = "io-uring",
    feature = "runtime-prime-reactor"
))]

use std::io;
use std::marker::PhantomData;
use std::task::Waker;

use io_uring::{IoUring, opcode};

/// per-direction read/write buffer size for `TcpStream`; owned here (rather
/// than in `tcp_stream`) because a cancelled op's buffer is parked in
/// `OpSlot` until the kernel confirms it is done writing into it.
///
/// build-time tunable — traces to `[io_uring].buf_size` in
/// `prime-runtime.toml` (principle 12), overridable via
/// `PRIME_IO_URING_BUF_SIZE`.
pub(super) const BUF_SIZE: usize = crate::core::sized::IO_URING_BUF_SIZE;

/// memory a submitted SQE still points at. dropping this before the matching
/// CQE arrives is exactly the drop-path use-after-free this reactor guards
/// against, so cancellation moves ownership here instead of back to the
/// caller.
pub(super) enum ParkedResource {
    /// `TcpStream::read_buf` / `write_buf`.
    StreamBuffer(Box<[u8; BUF_SIZE]>),
    /// `TcpListener`'s per-accept kernel-written scratch.
    AcceptStorage {
        addr_storage: Box<libc::sockaddr_storage>,
        addr_len: Box<libc::socklen_t>,
    },
}

/// state of a single slab slot.
enum SlotState {
    /// slot is live; operation in flight.
    InFlight { waker: Option<Waker> },
    /// operation completed; result ready to consume.
    Done { result: i32 },
    /// cancellation requested while in flight; `parked` stays alive until
    /// the real CQE (cancel-won or op-completed, either way) is drained.
    Cancelling { parked: ParkedResource },
    /// slot tracks only an `IORING_OP_ASYNC_CANCEL` SQE's own completion —
    /// no waker, no parked resource, reclaimed silently on arrival.
    CancelSqe,
    /// slot is free.
    Free,
}

struct OpSlot {
    state: SlotState,
    /// dead-slot generation counter (bumped on free) so stale user-data ids
    /// coming back from the ring are discarded.
    generation: u32,
}

/// debug-narrate a parked resource's reclaim — the entire point of parking
/// is that this only ever runs once the kernel's own CQE confirms it is
/// done touching the memory, so the log line records exactly that moment.
fn log_parked_reclaim(parked: &ParkedResource, cqe_result: i32) {
    match parked {
        ParkedResource::StreamBuffer(buffer) => {
            tracing::debug!(
                bytes = buffer.len(),
                cqe_result,
                "kernel confirmed completion, freeing parked stream buffer"
            );
        }
        ParkedResource::AcceptStorage {
            addr_storage,
            addr_len,
        } => {
            tracing::debug!(
                storage_bytes = std::mem::size_of_val(addr_storage.as_ref()),
                addr_len = **addr_len,
                cqe_result,
                "kernel confirmed completion, freeing parked accept storage"
            );
        }
    }
}

/// 64-bit user_data encoding: high 32 bits = generation, low 32 bits = index.
#[inline]
fn pack(index: u32, generation: u32) -> u64 {
    ((generation as u64) << 32) | (index as u64)
}

#[inline]
fn unpack(raw: u64) -> (u32, u32) {
    (raw as u32, (raw >> 32) as u32)
}

/// bump generation, skipping zero (dead sentinel).
#[inline]
fn bump(current: u32) -> u32 {
    match current.wrapping_add(1) {
        0 => 1,
        next => next,
    }
}

/// per-worker io_uring reactor. owns the IoUring ring and a slab of pending
/// operations. `!Send` — lives entirely on the worker thread.
pub struct PrimeUringReactor {
    ring: IoUring,
    slots: Vec<OpSlot>,
    free_head: Option<u32>,
    next_generation: u32,
    _not_send: PhantomData<*mut ()>,
}

impl PrimeUringReactor {
    /// construct with a 64-entry SQ/CQ ring. 64 entries is enough for the
    /// correctness floor; larger rings amortise submit_and_wait latency under
    /// load (perf follow-on).
    pub fn new() -> io::Result<Self> {
        let ring = IoUring::new(64)?;
        Ok(Self {
            ring,
            slots: Vec::new(),
            free_head: None,
            next_generation: 1,
            _not_send: PhantomData,
        })
    }

    /// the raw fd of the io_uring ring itself. register this with epoll so
    /// that when completions are available (CQ non-empty), epoll wakes up.
    pub fn ring_fd(&self) -> std::os::fd::RawFd {
        use std::os::fd::AsRawFd;
        self.ring.as_raw_fd()
    }

    /// allocate a slab slot in the given initial state, return the packed
    /// user_data id. shared by `register_op` (real ops) and
    /// `register_cancel_slot` (tracking-only cancel SQEs).
    fn allocate_slot(&mut self, state: SlotState) -> u64 {
        let generation = self.next_generation;
        self.next_generation = bump(self.next_generation);
        let index = match self.free_head.take() {
            Some(free) => {
                let slot = &mut self.slots[free as usize];
                slot.state = state;
                slot.generation = generation;
                free
            }
            None => {
                let raw = self.slots.len() as u32;
                self.slots.push(OpSlot { state, generation });
                raw
            }
        };
        pack(index, generation)
    }

    /// allocate a slab slot, return the packed user_data id.
    pub fn register_op(&mut self) -> u64 {
        self.allocate_slot(SlotState::InFlight { waker: None })
    }

    /// allocate a slab slot purely to track an `IORING_OP_ASYNC_CANCEL`
    /// SQE's own completion — no waker, no parked resource.
    fn register_cancel_slot(&mut self) -> u64 {
        self.allocate_slot(SlotState::CancelSqe)
    }

    /// store (or update) the waker for a live slot. no-op if the id is stale.
    pub fn set_waker(&mut self, user_data: u64, waker: Waker) {
        let (index, generation) = unpack(user_data);
        let Some(slot) = self.slots.get_mut(index as usize) else {
            return;
        };
        if slot.generation != generation {
            return;
        }
        if let SlotState::InFlight { waker: slot_waker } = &mut slot.state {
            *slot_waker = Some(waker);
        }
    }

    /// check whether a slot has a completed result. returns `Some(result)`
    /// and transitions the slot back to `Free` when it does. returns `None`
    /// if the operation is still in flight or if the id is stale.
    pub fn take_result(&mut self, user_data: u64) -> Option<i32> {
        let (index, generation) = unpack(user_data);
        let slot = self.slots.get_mut(index as usize)?;
        if slot.generation != generation {
            return None;
        }
        match &slot.state {
            SlotState::Done { result } => {
                let result = *result;
                slot.state = SlotState::Free;
                self.free_head = Some(index);
                Some(result)
            }
            _ => None,
        }
    }

    /// drain up to all pending CQEs without blocking. wakes any tasks whose
    /// operations completed. called opportunistically from every poll method
    /// before consulting result slots.
    pub fn drain_cqes(&mut self) -> io::Result<usize> {
        self.ring.submit()?;
        let mut count = 0;
        // SAFETY: no other code holds a mutable reference to ring or slots
        // during this call — we are the sole accessor on the worker thread.
        let completion = unsafe { self.ring.completion_shared() };
        for cqe in completion {
            let user_data = cqe.user_data();
            let result = cqe.result();
            let (index, generation) = unpack(user_data);
            let Some(slot) = self.slots.get_mut(index as usize) else {
                continue;
            };
            if slot.generation != generation {
                continue;
            }
            match &mut slot.state {
                SlotState::InFlight { waker } => {
                    let maybe_waker = waker.take();
                    slot.state = SlotState::Done { result };
                    if let Some(waker) = maybe_waker {
                        waker.wake();
                    }
                    count += 1;
                }
                SlotState::Cancelling { parked } => {
                    // real completion for the cancelled op — whether the
                    // cancel won the race (-ECANCELED) or lost it (a real
                    // byte count), the kernel is done touching `parked`'s
                    // memory, so dropping it here (via the state overwrite)
                    // is the earliest safe moment.
                    log_parked_reclaim(parked, result);
                    slot.state = SlotState::Free;
                    self.free_head = Some(index);
                    count += 1;
                }
                SlotState::CancelSqe => {
                    // the ASYNC_CANCEL SQE's own completion; nothing to
                    // wake, nothing parked — just reclaim the slot.
                    slot.state = SlotState::Free;
                    self.free_head = Some(index);
                    count += 1;
                }
                SlotState::Done { .. } | SlotState::Free => {}
            }
        }
        Ok(count)
    }

    /// mutable access to the IoUring ring for SQE submission.
    pub fn ring_mut(&mut self) -> &mut IoUring {
        &mut self.ring
    }

    /// cancel an in-flight op, submitting a real `IORING_OP_ASYNC_CANCEL` and
    /// parking `parked` until the op's own CQE (cancelled or raced-to-
    /// completion, either way) is drained. `parked` must be the memory the
    /// original SQE pointed at — freeing it earlier is the drop-path
    /// use-after-free this reactor exists to prevent.
    pub(super) fn cancel_op(&mut self, user_data: u64, parked: ParkedResource) {
        let (index, generation) = unpack(user_data);
        let Some(slot) = self.slots.get(index as usize) else {
            return;
        };
        if slot.generation != generation {
            return;
        }

        if matches!(slot.state, SlotState::Done { .. }) {
            // kernel already produced the result; nothing left to race.
            if let Some(slot) = self.slots.get_mut(index as usize) {
                slot.state = SlotState::Free;
            }
            self.free_head = Some(index);
            return;
        }
        if !matches!(slot.state, SlotState::InFlight { .. }) {
            // already free / cancelling / a cancel-tracking slot — the
            // caller's own bookkeeping guarantees this shouldn't happen for
            // a live op, but there is nothing to park defensively either way.
            return;
        }

        let cancel_user_data = self.register_cancel_slot();
        let cancel_sqe = opcode::AsyncCancel::new(user_data)
            .build()
            .user_data(cancel_user_data);
        // SAFETY: sole accessor of ring/slots on this worker thread, same
        // contract as drain_cqes / the SQE pushes in tcp_stream/tcp_listener.
        unsafe {
            let _ = self.ring.submission().push(&cancel_sqe);
        }
        let _ = self.ring.submit();

        if let Some(slot) = self.slots.get_mut(index as usize) {
            slot.state = SlotState::Cancelling { parked };
        }
    }
}

thread_local! {
    /// lazily-initialised per-worker io_uring reactor. set to `Some` on first
    /// use by `TcpListener` or `TcpStream`; stays `None` on threads that never
    /// touch the uring backend (epoll path, tokio threads, etc.).
    ///
    /// `UnsafeCell` is used (vs `RefCell`) for the same reason CURRENT_REACTOR
    /// uses it: eliminates the borrow-tracking branch on every WouldBlock poll.
    /// the reactor is single-thread-owned by construction.
    pub static CURRENT_URING: std::cell::UnsafeCell<Option<PrimeUringReactor>> =
        const { std::cell::UnsafeCell::new(None) };
}

/// return `&mut PrimeUringReactor` for the calling worker. lazily initialises
/// the reactor on first call. on initialisation, the ring fd is registered
/// with the epoll reactor (via `CURRENT_REACTOR`) so that when the io_uring
/// completion queue becomes non-empty, epoll wakes up and the worker drains
/// CQEs immediately rather than sleeping until a timeout.
///
/// # Safety (caller contract)
/// must be called on a single thread that exclusively owns this TLS slot.
/// no other borrow into `CURRENT_URING` may be live when this is called.
pub fn with_current_uring<F, T>(func: F) -> io::Result<T>
where
    F: FnOnce(&mut PrimeUringReactor) -> io::Result<T>,
{
    CURRENT_URING.with(|cell| {
        let opt = unsafe { &mut *cell.get() };
        if opt.is_none() {
            let reactor = PrimeUringReactor::new()?;
            // register the ring fd with the epoll reactor so that CQE arrivals
            // wake the parked worker without waiting for a timer timeout.
            register_ring_fd_with_epoll(reactor.ring_fd())?;
            *opt = Some(reactor);
        }
        match opt {
            Some(reactor) => func(reactor),
            None => Err(io::Error::other(
                "io_uring reactor slot empty immediately after initialisation",
            )),
        }
    })
}

/// register the io_uring ring fd with the thread-local epoll reactor for
/// read interest. best-effort: if CURRENT_REACTOR is null (off-worker thread
/// or reactor not initialised), the registration is skipped. the worker will
/// still drain CQEs via `drain_cqes_if_initialized` after `turn()` returns
/// on timeout.
// TODO(floor-io): reuse prime::os::readiness::Readiness here, needs real-io_uring validation
fn register_ring_fd_with_epoll(ring_fd: std::os::fd::RawFd) -> io::Result<()> {
    use super::super::core_shard::CURRENT_REACTOR;
    use super::super::reactor::Interest;

    let reactor_ptr = CURRENT_REACTOR.with(std::cell::Cell::get);
    if reactor_ptr.is_null() {
        return Ok(());
    }
    // SAFETY: pointer is set by core_shard::worker_main; valid for the
    // worker's lifetime. we're on the worker thread (same thread that set it).
    let reactor = unsafe { &mut *reactor_ptr };
    // ignore error if fd is already registered (idempotent on re-init).
    let _ = reactor.register(ring_fd, Interest::Read);
    Ok(())
}

/// drain pending CQEs only if the uring reactor has already been initialised
/// on this thread. no-op (returns Ok(0)) when uninitialised — does NOT
/// initialise the uring. called from the CoreShard worker loop so it runs on
/// every idle iteration regardless of whether the worker is using uring I/O.
pub fn drain_cqes_if_initialized() -> io::Result<usize> {
    CURRENT_URING.with(|cell| {
        let opt = unsafe { &mut *cell.get() };
        match opt {
            Some(reactor) => reactor.drain_cqes(),
            None => Ok(0),
        }
    })
}

#[cfg(test)]
#[cfg(all(
    target_os = "linux",
    feature = "io-uring",
    feature = "runtime-prime-reactor"
))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use io_uring::{opcode, types};
    use std::net::{TcpListener, TcpStream};
    use std::os::fd::AsRawFd;

    /// real, connected loopback pair — the server half is what the reactor
    /// submits a Recv against; both ends must stay alive for the fd to stay
    /// valid for the duration of the test.
    fn loopback_pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback listener");
        let addr = listener.local_addr().expect("read bound addr");
        let client = TcpStream::connect(addr).expect("connect loopback client");
        let (server, _peer) = listener.accept().expect("accept loopback server");
        (server, client)
    }

    /// blocks on real kernel completions (`submit_and_wait`) and drains until
    /// the tracked slot leaves `Cancelling`, or panics after too many rounds.
    /// no sleeps: every wait is a real syscall bound to actual CQE arrivals.
    fn wait_until_reclaimed(reactor: &mut PrimeUringReactor, index: u32) {
        for _ in 0..16 {
            let is_cancelling = matches!(reactor.slots[index as usize].state, SlotState::Cancelling { .. });
            if !is_cancelling {
                return;
            }
            reactor
                .ring_mut()
                .submit_and_wait(1)
                .expect("wait for a real cqe");
            reactor.drain_cqes().expect("drain after wait");
        }
        panic!("slot never left Cancelling after 16 real completion rounds");
    }

    /// reproduces (and, once fixed, disproves) the drop-path use-after-free
    /// hazard: cancelling an in-flight op must not hand the caller's buffer
    /// back for deallocation before the kernel has actually confirmed (via a
    /// real CQE) that it is done writing into it.
    #[test]
    fn cancel_op_parks_buffer_until_kernel_confirms_completion() {
        let mut reactor = PrimeUringReactor::new().expect("construct reactor");
        let (server, client) = loopback_pair();

        let mut buffer = Box::new([0u8; BUF_SIZE]);
        let user_data = reactor.register_op();
        let sqe = opcode::Recv::new(types::Fd(server.as_raw_fd()), buffer.as_mut_ptr(), BUF_SIZE as u32)
            .build()
            .user_data(user_data);
        unsafe {
            reactor
                .ring_mut()
                .submission()
                .push(&sqe)
                .expect("push recv sqe");
        }
        reactor.ring_mut().submit().expect("submit recv sqe");

        let drained_before_cancel = reactor.drain_cqes().expect("drain before cancel");
        assert_eq!(
            drained_before_cancel, 0,
            "peer never wrote — recv must still be in flight before cancel"
        );

        // this is exactly what `TcpStream::drop` does: hand the live buffer
        // to the reactor rather than dropping it itself.
        reactor.cancel_op(user_data, ParkedResource::StreamBuffer(buffer));

        let (index, _generation) = unpack(user_data);
        let slot_freed_immediately = matches!(reactor.slots[index as usize].state, SlotState::Free);
        assert!(
            !slot_freed_immediately,
            "cancel_op freed the slab slot before the kernel confirmed the \
             recv was cancelled or completed; a buffer bound to this op \
             (read_buf in TcpStream::drop) could be deallocated right now \
             while the kernel still holds a raw pointer into it — this is \
             the drop-path use-after-free"
        );

        let reused_while_still_live = reactor.register_op();
        let (reused_index, _) = unpack(reused_while_still_live);
        assert_ne!(
            reused_index, index,
            "a fresh op must not reuse the slot of a still-parked cancellation"
        );

        // now let the kernel actually finish — either the cancel wins
        // (-ECANCELED) or real bytes land, either way this is what proves
        // the kernel is done touching the parked buffer's memory.
        let mut peer = client;
        use std::io::Write;
        peer.write_all(b"parked-buffer-completion-proof")
            .expect("write payload to force real completion");

        wait_until_reclaimed(&mut reactor, index);
        assert!(
            matches!(reactor.slots[index as usize].state, SlotState::Free),
            "slot must be Free once the real CQE is drained, not stuck or leaked"
        );

        let reused_after_completion = reactor.register_op();
        let (final_index, _) = unpack(reused_after_completion);
        assert_eq!(
            final_index, index,
            "the reclaimed slot must be available for reuse once the kernel confirmed completion"
        );

        drop(server);
    }
}
