//! TCP listener + stream built on `proxima::runtime::prime::os::reactor`.
//! implements `futures::io::AsyncRead + AsyncWrite` so it composes with
//! runtime-neutral protocols (proxima's native `serve_h2_connection`,
//! `serve_h1_connection`, etc.) without needing a tokio I/O context.
//!
//! design:
//!   - Sockets are constructed via `socket2` and set non-blocking.
//!   - On first poll, the stream/listener lazily registers with the worker
//!     thread's Reactor (via the `CURRENT_REACTOR` thread-local published by
//!     CoreShard) AND caches the reactor's raw pointer in its own slot.
//!     subsequent polls deref the cached pointer directly — no thread-local
//!     read, no RefCell borrow check, no closure indirection.
//!   - The cached waker is compared with `will_wake` to elide the Arc clone
//!     when the same task re-polls the same source (the common case under
//!     burst load).
//!   - `poll_read` / `poll_write` / `poll_accept` follow the standard
//!     non-blocking-fd dance: try the syscall; on `WouldBlock` register the
//!     current waker with the Reactor and return `Pending`.
//!   - On drop, the source is deregistered.
//!
//! ## Send / thread affinity contract
//!
//! `TcpListener` and `TcpStream` are `Send` (auto-derived — every field is
//! Send) and intentionally `!Sync` (`PhantomData<Cell<()>>` enforces this).
//! Send is allowed so the types compose with executor APIs that demand
//! `Send` bounds (e.g. `serve_h2_connection`'s `S: AsyncRead + AsyncWrite
//! + Send + 'static`).
//!
//! Polling these types is **only sound on the proxima worker thread that
//! registered them** (the one whose `CURRENT_REACTOR` is non-null and
//! points at the Reactor that issued the cached `SourceKey`). The cached
//! reactor pointer is invalid on any other thread. Polling off-worker is
//! undefined behavior — but it cannot happen by accident through the
//! provided runtime APIs because tasks spawned on a CoreShard never
//! migrate to another thread (the runtime is per-core, not work-stealing).
//!
//! There is no auto-derived `!Send` marker that captures this rule, so we
//! rely on the runtime topology to enforce it. Manual `std::thread::spawn`
//! with a `TcpStream` would violate the contract.

// matches os.rs's `pub mod core_shard;` gate exactly — this file imports
// core_shard::CURRENT_REACTOR, and that module doesn't exist without the
// full executor + reactor + inbox-alloc conjunction, not `runtime-prime-reactor`
// alone.
#![cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-inbox-alloc",
    any(target_os = "macos", target_os = "linux"),
))]

use std::io;
use std::marker::PhantomData;
#[cfg(target_os = "linux")]
use std::mem;
use std::net::SocketAddr;
use std::os::fd::AsRawFd;
use std::pin::Pin;
use std::ptr;
use std::task::{Context, Poll};

use std::task::Waker;

use futures::io::{AsyncRead, AsyncWrite};
use socket2::{Domain, Protocol, SockAddr, Socket, Type};

use super::core_shard::{self, CURRENT_REACTOR};
use super::reactor::{Interest, Reactor, SourceKey};

/// 1024 is the standard default accept-queue depth (matches `SOMAXCONN` on
/// most kernels and the historical libc default) — deep enough to absorb
/// connection bursts without dropping SYNs under normal load.
const DEFAULT_LISTEN_BACKLOG: i32 = 1024;

/// non-blocking TCP listener bound to the proxima reactor on the worker
/// thread that constructs it. accept() returns a futures-io TcpStream.
pub struct TcpListener {
    socket: Socket,
    source: Option<SourceKey>,
    /// cached raw pointer to the worker's `Reactor`. set on first
    /// registration; subsequent polls deref directly without going through
    /// the thread-local. `*mut` is `!Send` so the type's auto-Send
    /// disappears — see the `unsafe impl Send` block and module docs.
    reactor_ptr: *mut Reactor,
    /// retained for `!Sync` (which IS enforced by `Cell<()>` here).
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

// SAFETY: every owned field is conceptually Send — Socket is Send, Option
// <SourceKey> is plain data, Option<Waker> is Send. The cached
// `*mut Reactor` is the only !Send field, and we restore Send so the type
// composes with executor APIs that require it.
//
// The CONTRACT (see module docs): the cached pointer is only valid on the
// worker thread that produced it. Polling on a different thread will
// dereference a pointer that belongs to another worker's Reactor — UB.
// The proxima runtime never migrates tasks cross-thread (per-core, not
// work-stealing), so under normal use this cannot happen. Users who hand
// these types to `std::thread::spawn` violate the contract.
unsafe impl Send for TcpListener {}

impl TcpListener {
    /// bind a non-blocking listening socket. must be called on a proxima
    /// worker thread (CoreShard worker_main has set CURRENT_REACTOR).
    pub fn bind(addr: SocketAddr) -> io::Result<Self> {
        Self::bind_with_backlog(addr, DEFAULT_LISTEN_BACKLOG)
    }

    /// bind a non-blocking listening socket with an explicit accept-queue
    /// depth. must be called on a proxima worker thread.
    pub fn bind_with_backlog(addr: SocketAddr, backlog: i32) -> io::Result<Self> {
        let domain = match addr {
            SocketAddr::V4(_) => Domain::IPV4,
            SocketAddr::V6(_) => Domain::IPV6,
        };
        let socket = Socket::new(domain, Type::STREAM, Some(Protocol::TCP))?;
        socket.set_nonblocking(true)?;
        socket.set_reuse_address(true)?;
        // SO_REUSEPORT: the per-core serve model binds the SAME (possibly
        // concrete) port once per CoreShard worker; without it, every core
        // after the first hits AddrInUse on a fixed port. SO_REUSEADDR alone is
        // not enough for same-port multi-bind on macOS/BSD. (proxima-listen
        // documents the same REUSEADDR + REUSEPORT requirement.)
        #[cfg(unix)]
        socket.set_reuse_port(true)?;
        let sock_addr = SockAddr::from(addr);
        socket.bind(&sock_addr)?;
        socket.listen(backlog)?;
        Ok(Self {
            socket,
            source: None,
            reactor_ptr: ptr::null_mut(),
            _not_sync: PhantomData,
        })
    }

    /// the local socket address the listener bound to (post-`bind`).
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        let sock_addr = self.socket.local_addr()?;
        sock_addr
            .as_socket()
            .ok_or_else(|| io::Error::other("local_addr was not an IP socket"))
    }

    /// accept the next pending connection. on first call (or after
    /// `Pending`) registers the listening fd's read waker with the reactor.
    pub fn poll_accept(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<io::Result<(TcpStream, SocketAddr)>> {
        let this = self.get_mut();
        match this.socket.accept() {
            Ok((socket, sock_addr)) => {
                socket.set_nonblocking(true)?;
                let _ = socket.set_nodelay(true);
                let peer = sock_addr
                    .as_socket()
                    .ok_or_else(|| io::Error::other("peer addr was not an IP socket"))?;
                Poll::Ready(Ok((TcpStream::from_socket(socket), peer)))
            }
            Err(err) if err.kind() == io::ErrorKind::WouldBlock => {
                if let Err(register_err) = this.register_read_waker(context) {
                    return Poll::Ready(Err(register_err));
                }
                core_shard::note_reactor_pending();
                Poll::Pending
            }
            Err(err) => Poll::Ready(Err(err)),
        }
    }

    /// async accept future for one connection.
    pub fn accept(&mut self) -> Accept<'_> {
        Accept { listener: self }
    }

    fn register_read_waker(&mut self, context: &Context<'_>) -> io::Result<()> {
        let reactor = ensure_reactor_ptr(&mut self.reactor_ptr)?;
        if self.source.is_none() {
            let key = reactor.register(self.socket.as_raw_fd(), Interest::Read)?;
            self.source = Some(key);
        }
        let Some(key) = self.source else {
            return Err(io::Error::other(
                "TcpListener: missing source key after register",
            ));
        };
        // `register_read_waker_ref` clones the waker only when the slot's
        // stored waker does not already `will_wake` the same task — that
        // check is correct against the live slot state (the reactor's
        // `turn` takes the waker out when it fires, so a stale local
        // cache would deadlock; the slot itself is authoritative).
        if !reactor.register_read_waker_ref(key, context.waker()) {
            return Err(io::Error::other("TcpListener: reactor source went stale"));
        }
        Ok(())
    }
}

impl Drop for TcpListener {
    fn drop(&mut self) {
        deregister_on_drop(&mut self.source, self.reactor_ptr);
    }
}

/// future returned by `TcpListener::accept`. polls `poll_accept`.
pub struct Accept<'listener> {
    listener: &'listener mut TcpListener,
}

impl std::future::Future for Accept<'_> {
    type Output = io::Result<(TcpStream, SocketAddr)>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        Pin::new(&mut *this.listener).poll_accept(context)
    }
}

/// state machine for the `Connect` future.
///
/// `Init` — not yet started; transitions to `Pending` on first poll.
/// `Pending` — connect syscall returned EINPROGRESS; waiting for write-readiness.
/// `Done` — terminal; reachable only on logic error.
enum ConnectState {
    Init,
    Pending {
        socket: Socket,
        source: SourceKey,
        reactor_ptr: *mut Reactor,
    },
    Done,
}

// SAFETY: Socket is Send; *mut Reactor follows the same contract as TcpStream.
unsafe impl Send for ConnectState {}

/// future returned by `TcpStream::connect`. polls until the TCP handshake
/// completes or fails, then yields a `TcpStream`.
pub struct Connect {
    addr: SocketAddr,
    state: ConnectState,
}

// SAFETY: Connect contains ConnectState which is Send per above.
unsafe impl Send for Connect {}

impl std::future::Future for Connect {
    type Output = io::Result<TcpStream>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match &mut this.state {
            ConnectState::Init => {
                let domain = match this.addr {
                    SocketAddr::V4(_) => Domain::IPV4,
                    SocketAddr::V6(_) => Domain::IPV6,
                };
                let socket = match Socket::new(domain, Type::STREAM, Some(Protocol::TCP)) {
                    Ok(sock) => sock,
                    Err(err) => return Poll::Ready(Err(err)),
                };
                if let Err(err) = socket.set_nonblocking(true) {
                    return Poll::Ready(Err(err));
                }
                let _ = socket.set_nodelay(true);
                let sock_addr = SockAddr::from(this.addr);
                match socket.connect(&sock_addr) {
                    Ok(()) => {
                        // immediate connect (loopback / already established)
                        return Poll::Ready(Ok(TcpStream::from_socket(socket)));
                    }
                    Err(err) if is_connect_in_progress(&err) => {}
                    Err(err) => return Poll::Ready(Err(err)),
                }
                let mut reactor_ptr: *mut Reactor = ptr::null_mut();
                let reactor = match ensure_reactor_ptr(&mut reactor_ptr) {
                    Ok(reactor) => reactor,
                    Err(err) => return Poll::Ready(Err(err)),
                };
                let source = match reactor.register(socket.as_raw_fd(), Interest::Write) {
                    Ok(key) => key,
                    Err(err) => return Poll::Ready(Err(err)),
                };
                if !reactor.register_write_waker_ref(source, context.waker()) {
                    return Poll::Ready(Err(io::Error::other(
                        "Connect: reactor source went stale immediately after register",
                    )));
                }
                this.state = ConnectState::Pending {
                    socket,
                    source,
                    reactor_ptr,
                };
                core_shard::note_reactor_pending();
                Poll::Pending
            }

            ConnectState::Pending { socket, .. } => {
                // re-probe completion via a second connect() call — unambiguous
                // under a spurious poll: EALREADY means still in progress,
                // EISCONN means established, any other error is the failure.
                // take_error() alone is ambiguous because Ok(None) is returned
                // both for "not yet connected" and "connected" before writable.
                let sock_addr = SockAddr::from(this.addr);
                let probe = socket.connect(&sock_addr);

                // classify into an owned outcome so the socket borrow is dropped
                // before the mem::replace that follows.
                enum Outcome {
                    Connected,
                    InProgress,
                    Failed(io::Error),
                }
                let outcome = match probe {
                    Ok(()) => Outcome::Connected,
                    Err(ref err) if is_already_connected(err) => Outcome::Connected,
                    Err(ref err) if is_connect_in_progress(err) || is_connect_resuming(err) => {
                        Outcome::InProgress
                    }
                    Err(err) => {
                        // cross-check with SO_ERROR for the precise failure errno.
                        let precise = socket.take_error().ok().flatten().unwrap_or(err);
                        Outcome::Failed(precise)
                    }
                };

                match outcome {
                    Outcome::Connected => {
                        let ConnectState::Pending {
                            socket: owned_socket,
                            source: owned_source,
                            reactor_ptr: owned_ptr,
                        } = std::mem::replace(&mut this.state, ConnectState::Done)
                        else {
                            unreachable!()
                        };
                        // deregister the write-only registration;
                        // TcpStream::ensure_registered will re-register
                        // for the needed interest on first I/O poll.
                        if !owned_ptr.is_null() {
                            // SAFETY: same invariant as TcpStream::ensure_registered.
                            let reactor = unsafe { &mut *owned_ptr };
                            let _ = reactor.deregister(owned_source);
                        }
                        Poll::Ready(Ok(TcpStream::from_socket(owned_socket)))
                    }

                    Outcome::InProgress => {
                        // spurious poll or not-yet-resolved: re-arm the waker
                        // so the next real writable edge wakes this task.
                        let ConnectState::Pending {
                            source,
                            reactor_ptr,
                            ..
                        } = &this.state
                        else {
                            unreachable!()
                        };
                        let source = *source;
                        let reactor_ptr = *reactor_ptr;
                        if !reactor_ptr.is_null() {
                            // SAFETY: same invariant as ensure_reactor_ptr.
                            let reactor = unsafe { &mut *reactor_ptr };
                            if !reactor.register_write_waker_ref(source, context.waker()) {
                                return Poll::Ready(Err(io::Error::other(
                                    "Connect: reactor source went stale",
                                )));
                            }
                        }
                        core_shard::note_reactor_pending();
                        Poll::Pending
                    }

                    Outcome::Failed(err) => {
                        this.state = ConnectState::Done;
                        Poll::Ready(Err(err))
                    }
                }
            }

            ConnectState::Done => {
                Poll::Ready(Err(io::Error::other("Connect polled after completion")))
            }
        }
    }
}

impl Drop for Connect {
    fn drop(&mut self) {
        if let ConnectState::Pending {
            source,
            reactor_ptr,
            ..
        } = &mut self.state
        {
            deregister_on_drop(&mut Some(*source), *reactor_ptr);
        }
    }
}

/// returns true when a `connect(2)` error means the handshake is in
/// progress (normal for non-blocking sockets).
///
/// POSIX: non-blocking `connect` returns EINPROGRESS.
/// macOS also returns EWOULDBLOCK (same numeric value 36) as an alias.
/// Linux: only EINPROGRESS (115). We match both via kind OR raw code.
#[inline]
fn is_connect_in_progress(err: &io::Error) -> bool {
    if err.kind() == io::ErrorKind::WouldBlock {
        return true;
    }
    // EINPROGRESS is not mapped to a std ErrorKind variant; check raw.
    #[cfg(target_os = "linux")]
    const EINPROGRESS: i32 = 115;
    #[cfg(target_os = "macos")]
    const EINPROGRESS: i32 = 36;
    err.raw_os_error() == Some(EINPROGRESS)
}

/// returns true when a second `connect(2)` on an already-connected socket
/// returns EISCONN — unambiguous signal that the handshake completed.
#[inline]
fn is_already_connected(err: &io::Error) -> bool {
    #[cfg(target_os = "linux")]
    const EISCONN: i32 = 106;
    #[cfg(target_os = "macos")]
    const EISCONN: i32 = 56;
    err.raw_os_error() == Some(EISCONN)
}

/// returns true when a second `connect(2)` on a still-connecting socket
/// returns EALREADY — the handshake is still in flight (spurious poll).
#[inline]
fn is_connect_resuming(err: &io::Error) -> bool {
    #[cfg(target_os = "linux")]
    const EALREADY: i32 = 114;
    #[cfg(target_os = "macos")]
    const EALREADY: i32 = 37;
    err.raw_os_error() == Some(EALREADY)
}

/// non-blocking TCP stream bound to the proxima reactor. `futures::io`
/// compatible — composes directly with `serve_h2_connection` and friends.
pub struct TcpStream {
    socket: Socket,
    source: Option<SourceKey>,
    /// cached raw pointer to the worker's `Reactor`. see [TcpListener]'s
    /// field docs and module-level "Send / thread affinity contract".
    reactor_ptr: *mut Reactor,
    /// last waker registered for read-readiness. used to skip the reactor
    /// call when the same task re-polls with the same waker (the common
    /// case for steady-state stream reads under one connection task).
    ///
    /// SAFETY of this cache: `Reactor::turn` uses `wake_by_ref` to fire
    /// the slot's stored waker, so the waker REMAINS in the slot across
    /// wake events. A local equality check via `Waker::will_wake` is
    /// therefore sufficient to know the reactor slot is already armed
    /// with the right waker — no need to re-call
    /// `reactor.register_read_waker_ref`.
    ///
    /// HISTORY: an earlier version cached this and the reactor's `turn`
    /// used `take()` to consume the waker on fire — the slot was left
    /// waker-less, so the local cache went stale and reads deadlocked.
    /// The `wake_by_ref` change closed that race; the test
    /// `many_read_block_cycles_do_not_starve` is the regression guard.
    last_read_waker: Option<Waker>,
    /// same as `last_read_waker` but for write-readiness.
    last_write_waker: Option<Waker>,
    /// Reactor read epoch observed when `read(2)` last returned
    /// `WouldBlock`. If the epoch has not changed on a later poll, no new
    /// read event has arrived and we can return `Pending` without another
    /// EAGAIN syscall.
    last_read_blocked_epoch: Option<u32>,
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

// SAFETY: see `TcpListener` above — same reasoning. `*mut Reactor` is
// the only !Send field; we restore Send to support executor APIs that
// require it, with the documented contract that polling must remain on
// the worker thread that registered the stream.
unsafe impl Send for TcpStream {}

impl TcpStream {
    fn from_socket(socket: Socket) -> Self {
        Self {
            socket,
            source: None,
            reactor_ptr: ptr::null_mut(),
            last_read_waker: None,
            last_write_waker: None,
            last_read_blocked_epoch: None,
            _not_sync: PhantomData,
        }
    }

    /// asynchronously connect to `addr`. must be called on a proxima
    /// worker thread (CURRENT_REACTOR must be non-null).
    pub fn connect(addr: SocketAddr) -> Connect {
        Connect {
            addr,
            state: ConnectState::Init,
        }
    }

    /// Attempt a non-blocking read without registering a reactor waker.
    ///
    /// Returns [`io::ErrorKind::WouldBlock`] when the stream is not currently
    /// readable. This is the same syscall path as [`AsyncRead::poll_read`],
    /// minus the reactor wait/registration work.
    #[inline]
    pub fn try_read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.read_into(buf)
    }

    /// lazy register helper. on first call, registers with the reactor
    /// for the requested interest and caches the reactor pointer + source
    /// key. subsequent calls reregister only when the requested interest
    /// broadens the current source. read polls never narrow an already
    /// read/write-registered source because a write waker may be live.
    fn ensure_registered(&mut self, interest: Interest) -> io::Result<(&mut Reactor, SourceKey)> {
        let reactor = ensure_reactor_ptr(&mut self.reactor_ptr)?;
        if let Some(key) = self.source {
            if interest == Interest::ReadWrite {
                reactor.reregister(key, Interest::ReadWrite)?;
            }
            return Ok((reactor, key));
        }
        let key = reactor.register(self.socket.as_raw_fd(), interest)?;
        self.source = Some(key);
        Ok((reactor, key))
    }

    fn register_read_waker(&mut self, context: &Context<'_>) -> io::Result<u32> {
        // local-waker fast path: if we already registered this exact waker
        // with the reactor, skip the call entirely. `reactor.turn` uses
        // `wake_by_ref` (not `take`), so the slot's stored waker persists
        // across wake events — our local cache stays valid.
        let cached_waker_matches = self
            .last_read_waker
            .as_ref()
            .is_some_and(|cached| cached.will_wake(context.waker()));
        let epoch = {
            let (reactor, key) = self.ensure_registered(Interest::Read)?;
            if !cached_waker_matches && !reactor.register_read_waker_ref(key, context.waker()) {
                return Err(io::Error::other("TcpStream: reactor source went stale"));
            }
            reactor
                .read_ready_epoch(key)
                .ok_or_else(|| io::Error::other("TcpStream: reactor source went stale"))?
        };
        if !cached_waker_matches {
            // promote into the local cache for the next poll.
            self.last_read_waker = Some(context.waker().clone());
        }
        Ok(epoch)
    }

    fn register_write_waker(&mut self, context: &Context<'_>) -> io::Result<()> {
        if let Some(cached) = &self.last_write_waker
            && cached.will_wake(context.waker())
        {
            return Ok(());
        }
        let (reactor, key) = self.ensure_registered(Interest::ReadWrite)?;
        if !reactor.register_write_waker_ref(key, context.waker()) {
            return Err(io::Error::other("TcpStream: reactor source went stale"));
        }
        self.last_write_waker = Some(context.waker().clone());
        Ok(())
    }

    #[inline]
    fn read_into(&self, buf: &mut [u8]) -> io::Result<usize> {
        // SAFETY: this TCP socket is non-blocking and `read(2)` writes at most
        // `len` initialized bytes into the caller-owned buffer, returning the
        // initialized byte count.
        let len = buf.len().min(isize::MAX as usize);
        let n = unsafe { libc::read(self.socket.as_raw_fd(), buf.as_mut_ptr().cast(), len) };
        if n >= 0 {
            Ok(n as usize)
        } else {
            Err(io::Error::last_os_error())
        }
    }
}

impl Drop for TcpStream {
    fn drop(&mut self) {
        deregister_on_drop(&mut self.source, self.reactor_ptr);
    }
}

impl AsyncRead for TcpStream {
    #[inline]
    fn poll_read(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        if let Some(blocked_epoch) = this.last_read_blocked_epoch {
            match this.register_read_waker(context) {
                Ok(current_epoch) if current_epoch == blocked_epoch => {
                    core_shard::note_reactor_pending();
                    return Poll::Pending;
                }
                Ok(_) => {}
                Err(register_err) => return Poll::Ready(Err(register_err)),
            }
        }
        match this.read_into(buf) {
            Ok(n) => {
                #[cfg(feature = "runtime-prime-reactor-trace")]
                crate::trace::record_read_ready();
                this.last_read_blocked_epoch = None;
                Poll::Ready(Ok(n))
            }
            Err(err) if err.kind() == io::ErrorKind::WouldBlock => {
                let epoch = match this.register_read_waker(context) {
                    Ok(epoch) => epoch,
                    Err(register_err) => return Poll::Ready(Err(register_err)),
                };
                this.last_read_blocked_epoch = Some(epoch);
                core_shard::note_reactor_pending();
                #[cfg(feature = "runtime-prime-reactor-trace")]
                crate::trace::record_read_pending();
                Poll::Pending
            }
            Err(err) => {
                this.last_read_blocked_epoch = None;
                Poll::Ready(Err(err))
            }
        }
    }
}

impl AsyncWrite for TcpStream {
    #[inline]
    fn poll_write(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        match this.socket.send(buf) {
            Ok(n) => Poll::Ready(Ok(n)),
            Err(err) if err.kind() == io::ErrorKind::WouldBlock => {
                if let Err(register_err) = this.register_write_waker(context) {
                    return Poll::Ready(Err(register_err));
                }
                core_shard::note_reactor_pending();
                Poll::Pending
            }
            Err(err) => Poll::Ready(Err(err)),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<io::Result<()>> {
        // TCP has no userspace flush. kernel sends buffered bytes on its own.
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let _ = this.socket.shutdown(std::net::Shutdown::Both);
        Poll::Ready(Ok(()))
    }
}

/// returns a `&mut Reactor` for the calling worker. on first call, reads
/// `CURRENT_REACTOR` and caches it into `slot`; subsequent calls deref
/// `slot` directly with no thread-local read.
///
/// returns `Err` if called off a proxima worker (`CURRENT_REACTOR` null)
/// AND no pointer has been cached yet.
fn ensure_reactor_ptr(slot: &mut *mut Reactor) -> io::Result<&mut Reactor> {
    if slot.is_null() {
        let raw = CURRENT_REACTOR.with(std::cell::Cell::get);
        if raw.is_null() {
            return Err(io::Error::other(
                "proxima TcpListener/TcpStream used off worker thread \
                 (CURRENT_REACTOR is null — construct via spawn_factory_on_core)",
            ));
        }
        *slot = raw;
    }
    // SAFETY: CURRENT_REACTOR is set by CoreShard::worker_main to the raw
    // pointer of its own `UnsafeCell<Reactor>` (alive for the worker
    // thread's lifetime, cleared on worker exit by CurrentGuards::drop).
    // The pointer remains valid until the worker thread exits. By the
    // module-level Send contract, callers must only poll on the worker
    // thread that produced the cached pointer; the runtime's per-core
    // (no work-stealing) topology enforces this.
    //
    // No aliasing: the worker loop holds no outstanding borrow while
    // tasks are being polled (it only borrows the reactor when calling
    // `turn` between executor ticks); the Reactor's own methods don't
    // re-enter via wakers (wakers push to ready queues, not the reactor).
    Ok(unsafe { &mut **slot })
}

/// drop-time deregistration. uses the cached reactor pointer if non-null;
/// otherwise no-op (we never registered or the worker exited first).
fn deregister_on_drop(source: &mut Option<SourceKey>, reactor_ptr: *mut Reactor) {
    let Some(key) = source.take() else {
        return;
    };
    if reactor_ptr.is_null() {
        return;
    }
    // SAFETY: same invariants as `ensure_reactor_ptr`'s deref — the
    // pointer is to thread-owned data on the worker that registered the
    // source. Drop runs on whatever thread holds the type; under the
    // documented contract that thread is the worker. If the contract was
    // violated (move-then-drop on another thread), this is UB — but we
    // can't detect it without a thread-id check on every poll, which is
    // exactly the overhead this design eliminates.
    let reactor = unsafe { &mut *reactor_ptr };
    let _ = reactor.deregister(key);
}

/// Maximum datagrams moved per `recvmmsg`/`sendmmsg` syscall by
/// [`UdpSocket::poll_recv_batch`] / [`UdpSocket::poll_send_batch`]. Bounds
/// the on-stack header/iovec/addr scratch so batching never heap-allocates.
pub const DATAGRAM_BATCH: usize = 32;

/// Requested `SO_RCVBUF`/`SO_SNDBUF` for a [`UdpSocket`] (8 MiB). Sized to
/// absorb a QUIC connection-handshake burst on one shared socket; the kernel
/// clamps to `net.core.{rmem,wmem}_max`, so a server raises those sysctls too.
const UDP_SOCKET_BUFFER_BYTES: usize = 8 * 1024 * 1024;

/// Non-blocking UDP socket bound to the proxima reactor on the worker
/// thread that constructs it. Same Send/!Sync contract as
/// [`TcpListener`] / [`TcpStream`] — polling must happen on the worker
/// thread that registered the source.
///
/// QUIC's I/O facade (`proxima-quic`) composes this primitive to drive
/// `proxima_protocols::quic::Connection` from any executor (`prime` in
/// production; tokio via the tokio-compat feature on consumers).
pub struct UdpSocket {
    socket: Socket,
    source: Option<SourceKey>,
    reactor_ptr: *mut Reactor,
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

// SAFETY: same contract as TcpListener / TcpStream — see module docs.
unsafe impl Send for UdpSocket {}

impl UdpSocket {
    /// Bind a non-blocking UDP socket. Must be called on a proxima
    /// worker thread (CURRENT_REACTOR is non-null).
    ///
    /// # Errors
    ///
    /// Bubbles up any [`std::io::Error`] from `socket(2)` / `bind(2)`.
    pub fn bind(addr: SocketAddr) -> io::Result<Self> {
        let domain = match addr {
            SocketAddr::V4(_) => Domain::IPV4,
            SocketAddr::V6(_) => Domain::IPV6,
        };
        let socket = Socket::new(domain, Type::DGRAM, Some(Protocol::UDP))?;
        socket.set_nonblocking(true)?;
        socket.set_reuse_address(true)?;
        // SO_REUSEPORT: the per-core serve model binds this SAME port once per
        // core (one datagram socket per CoreShard worker). Without it the N
        // sockets share only SO_REUSEADDR and never form a load-balancing
        // reuseport group — the kernel funnels EVERY datagram to whichever
        // socket won the bind race (a single, run-random core), so a multi-core
        // UDP/QUIC server never scales past one core. Mirrors the TcpListener
        // fix; must be set before bind().
        #[cfg(unix)]
        socket.set_reuse_port(true)?;
        // QUIC multiplexes many connections over ONE datagram socket. A
        // connection burst (each peer's handshake CRYPTO arriving at once)
        // can momentarily exceed the kernel's default datagram buffer faster
        // than a single serve task drains it; a dropped Initial costs that
        // peer a full PTO backoff (~30s of 1+2+4+8+16s) to recover — a
        // per-connection stall that wrecks tail latency under load. Request a
        // generous buffer. The kernel clamps to `net.core.{rmem,wmem}_max`, so
        // a server deployment also raises those sysctls; best-effort here
        // (a clamp/failure is non-fatal, the socket still works).
        let _ = socket.set_recv_buffer_size(UDP_SOCKET_BUFFER_BYTES);
        let _ = socket.set_send_buffer_size(UDP_SOCKET_BUFFER_BYTES);
        let sock_addr = SockAddr::from(addr);
        socket.bind(&sock_addr)?;
        Ok(Self {
            socket,
            source: None,
            reactor_ptr: ptr::null_mut(),
            _not_sync: PhantomData,
        })
    }

    /// Local bind address post-`bind` (resolves ephemeral port).
    ///
    /// # Errors
    ///
    /// Bubbles up any [`std::io::Error`] from `getsockname(2)`.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.socket
            .local_addr()?
            .as_socket()
            .ok_or_else(|| io::Error::other("UdpSocket local_addr not IP"))
    }

    /// Non-blocking `recvfrom`. Returns
    /// `Poll::Pending` when no datagram is available + registers the
    /// caller's waker with the reactor for read-readiness.
    ///
    /// # Errors
    ///
    /// On UB-class errors only (reactor cache went stale; otherwise
    /// `WouldBlock` is converted to `Pending`).
    pub fn poll_recv_from(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<(usize, SocketAddr)>> {
        let this = self.get_mut();
        // socket2's recv_from takes &mut [MaybeUninit<u8>]; convert via
        // the raw fd path so callers can use plain &mut [u8].
        // SAFETY: writing `&mut [u8]` as `&mut [MaybeUninit<u8>]` is sound for
        // a non-reading source — `recv_from` only writes bytes. The
        // MaybeUninit transmute is required by socket2's API surface.
        let buf_ptr = buf.as_mut_ptr().cast::<core::mem::MaybeUninit<u8>>();
        let buf_slice = unsafe { core::slice::from_raw_parts_mut(buf_ptr, buf.len()) };
        match this.socket.recv_from(buf_slice) {
            Ok((len, sock_addr)) => {
                let peer = sock_addr
                    .as_socket()
                    .ok_or_else(|| io::Error::other("recv_from peer not IP"))?;
                Poll::Ready(Ok((len, peer)))
            }
            Err(err) if err.kind() == io::ErrorKind::WouldBlock => {
                if let Err(register_err) = this.register_read_waker(context) {
                    return Poll::Ready(Err(register_err));
                }
                core_shard::note_reactor_pending();
                Poll::Pending
            }
            Err(err) => Poll::Ready(Err(err)),
        }
    }

    /// Non-blocking `sendto`. Returns
    /// `Poll::Pending` when the kernel send buffer is full + registers
    /// the caller's waker with the reactor for write-readiness.
    ///
    /// # Errors
    ///
    /// On UB-class errors only.
    pub fn poll_send_to(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buf: &[u8],
        peer: SocketAddr,
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let sock_addr = SockAddr::from(peer);
        match this.socket.send_to(buf, &sock_addr) {
            Ok(written) => Poll::Ready(Ok(written)),
            Err(err) if err.kind() == io::ErrorKind::WouldBlock => {
                if let Err(register_err) = this.register_write_waker(context) {
                    return Poll::Ready(Err(register_err));
                }
                core_shard::note_reactor_pending();
                Poll::Pending
            }
            Err(err) => Poll::Ready(Err(err)),
        }
    }

    /// Synchronous send for the cold path (e.g. CONNECTION_CLOSE in
    /// teardown when no executor is driving). Fails with
    /// [`std::io::ErrorKind::WouldBlock`] when the kernel buffer is full.
    ///
    /// # Errors
    ///
    /// Bubbles up `sendto(2)` errors verbatim.
    pub fn send_to_blocking(&self, buf: &[u8], peer: SocketAddr) -> io::Result<usize> {
        self.socket.send_to(buf, &SockAddr::from(peer))
    }

    /// Batched non-blocking receive via `recvmmsg(2)` — pulls up to
    /// `min(bufs.len(), out_meta.len())` datagrams (capped at
    /// [`DATAGRAM_BATCH`]) in ONE syscall, writing each datagram's `(len,
    /// peer)` into `out_meta`, and returns the count received. `Poll::Pending`
    /// (reactor read-waker registered) when the socket is empty.
    ///
    /// The datagram analog of the reactor's batched readiness: one syscall
    /// amortizes the kernel-entry cost across a whole burst — the difference
    /// between QUIC throughput that scales and the per-packet `recvfrom` floor.
    ///
    /// # Errors
    ///
    /// Bubbles `recvmmsg(2)` errors other than `WouldBlock`.
    #[cfg(target_os = "linux")]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn poll_recv_batch(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        bufs: &mut [&mut [u8]],
        out_meta: &mut [(usize, SocketAddr)],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let want = bufs.len().min(out_meta.len()).min(DATAGRAM_BATCH);
        if want == 0 {
            return Poll::Ready(Ok(0));
        }
        // SAFETY: mmsghdr/iovec/sockaddr_storage are plain C PODs for which
        // all-zero is valid; every field read by the syscall is set below.
        let mut headers: [libc::mmsghdr; DATAGRAM_BATCH] = unsafe { mem::zeroed() };
        let mut iovecs: [libc::iovec; DATAGRAM_BATCH] = unsafe { mem::zeroed() };
        let mut addrs: [libc::sockaddr_storage; DATAGRAM_BATCH] = unsafe { mem::zeroed() };
        for index in 0..want {
            iovecs[index].iov_base = bufs[index].as_mut_ptr().cast();
            iovecs[index].iov_len = bufs[index].len();
            let header = &mut headers[index].msg_hdr;
            header.msg_name = ptr::addr_of_mut!(addrs[index]).cast();
            header.msg_namelen = mem::size_of::<libc::sockaddr_storage>() as u32;
            header.msg_iov = ptr::addr_of_mut!(iovecs[index]);
            header.msg_iovlen = 1;
        }
        // SAFETY: fd is the bound socket; headers/iovecs/addrs are valid for
        // `want` entries; null timeout returns immediately (non-blocking).
        let received = unsafe {
            libc::recvmmsg(
                this.socket.as_raw_fd(),
                headers.as_mut_ptr(),
                want as u32,
                0,
                ptr::null_mut(),
            )
        };
        if received < 0 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::WouldBlock {
                if let Err(register_err) = this.register_read_waker(context) {
                    return Poll::Ready(Err(register_err));
                }
                core_shard::note_reactor_pending();
                return Poll::Pending;
            }
            return Poll::Ready(Err(err));
        }
        let count = received as usize;
        for index in 0..count {
            // SAFETY: the kernel wrote a valid sockaddr of msg_namelen bytes.
            let sock_addr =
                unsafe { SockAddr::new(addrs[index], headers[index].msg_hdr.msg_namelen) };
            let peer = sock_addr
                .as_socket()
                .ok_or_else(|| io::Error::other("recvmmsg peer not IP"))?;
            out_meta[index] = (headers[index].msg_len as usize, peer);
        }
        Poll::Ready(Ok(count))
    }

    /// Non-Linux fallback: loop single `recv_from` until the socket drains or
    /// fills the batch — same return contract as the `recvmmsg` path.
    #[cfg(not(target_os = "linux"))]
    pub fn poll_recv_batch(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
        bufs: &mut [&mut [u8]],
        out_meta: &mut [(usize, SocketAddr)],
    ) -> Poll<io::Result<usize>> {
        let want = bufs.len().min(out_meta.len());
        let mut count = 0;
        while count < want {
            match self.as_mut().poll_recv_from(context, bufs[count]) {
                Poll::Ready(Ok((len, peer))) => {
                    out_meta[count] = (len, peer);
                    count += 1;
                }
                Poll::Ready(Err(_)) if count > 0 => return Poll::Ready(Ok(count)),
                Poll::Ready(Err(err)) => return Poll::Ready(Err(err)),
                Poll::Pending if count > 0 => return Poll::Ready(Ok(count)),
                Poll::Pending => return Poll::Pending,
            }
        }
        Poll::Ready(Ok(count))
    }

    /// Batched non-blocking send via `sendmmsg(2)` — ships all `packets`
    /// (each `(bytes, peer)`) in as few syscalls as possible (chunks of
    /// [`DATAGRAM_BATCH`]) and returns the count accepted. On a full send
    /// buffer with nothing yet sent, `Poll::Pending` (reactor write-waker
    /// registered); on partial progress returns the count so the caller retries
    /// the remainder.
    ///
    /// Zero-copy: each iovec borrows the caller's serialized packet bytes.
    ///
    /// # Errors
    ///
    /// Bubbles `sendmmsg(2)` errors other than `WouldBlock`.
    #[cfg(target_os = "linux")]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn poll_send_batch(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        packets: &[(&[u8], SocketAddr)],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let total = packets.len();
        let mut sent = 0;
        while sent < total {
            let chunk = (total - sent).min(DATAGRAM_BATCH);
            // SAFETY: see poll_recv_batch — plain C PODs, fields set below.
            let mut headers: [libc::mmsghdr; DATAGRAM_BATCH] = unsafe { mem::zeroed() };
            let mut iovecs: [libc::iovec; DATAGRAM_BATCH] = unsafe { mem::zeroed() };
            let mut addrs: [libc::sockaddr_storage; DATAGRAM_BATCH] = unsafe { mem::zeroed() };
            for index in 0..chunk {
                let (bytes, peer) = packets[sent + index];
                let sock_addr = SockAddr::from(peer);
                let addr_len = sock_addr.len();
                // SAFETY: copy the sockaddr bytes into stable stack storage;
                // sock_addr drops after, but addrs[index] owns the bytes for
                // the syscall's lifetime.
                unsafe {
                    ptr::copy_nonoverlapping(
                        sock_addr.as_ptr().cast::<u8>(),
                        ptr::addr_of_mut!(addrs[index]).cast::<u8>(),
                        addr_len as usize,
                    );
                }
                iovecs[index].iov_base = bytes.as_ptr().cast_mut().cast();
                iovecs[index].iov_len = bytes.len();
                let header = &mut headers[index].msg_hdr;
                header.msg_name = ptr::addr_of_mut!(addrs[index]).cast();
                header.msg_namelen = addr_len;
                header.msg_iov = ptr::addr_of_mut!(iovecs[index]);
                header.msg_iovlen = 1;
            }
            // SAFETY: fd is the bound socket; headers valid for `chunk` entries.
            let pushed = unsafe {
                libc::sendmmsg(
                    this.socket.as_raw_fd(),
                    headers.as_mut_ptr(),
                    chunk as u32,
                    0,
                )
            };
            if pushed < 0 {
                let err = io::Error::last_os_error();
                if err.kind() == io::ErrorKind::WouldBlock {
                    if sent > 0 {
                        return Poll::Ready(Ok(sent));
                    }
                    if let Err(register_err) = this.register_write_waker(context) {
                        return Poll::Ready(Err(register_err));
                    }
                    core_shard::note_reactor_pending();
                    return Poll::Pending;
                }
                return Poll::Ready(Err(err));
            }
            let accepted = pushed as usize;
            sent += accepted;
            // kernel took fewer than offered (send-buffer pressure): stop,
            // the caller retries the remainder on the next writable wake.
            if accepted < chunk {
                break;
            }
        }
        Poll::Ready(Ok(sent))
    }

    /// Non-Linux fallback: loop single `send_to` — same return contract as the
    /// `sendmmsg` path, one syscall per datagram.
    #[cfg(not(target_os = "linux"))]
    pub fn poll_send_batch(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
        packets: &[(&[u8], SocketAddr)],
    ) -> Poll<io::Result<usize>> {
        let mut sent = 0;
        while sent < packets.len() {
            let (bytes, peer) = packets[sent];
            match self.as_mut().poll_send_to(context, bytes, peer) {
                Poll::Ready(Ok(_)) => sent += 1,
                Poll::Ready(Err(_)) if sent > 0 => return Poll::Ready(Ok(sent)),
                Poll::Ready(Err(err)) => return Poll::Ready(Err(err)),
                Poll::Pending if sent > 0 => return Poll::Ready(Ok(sent)),
                Poll::Pending => return Poll::Pending,
            }
        }
        Poll::Ready(Ok(sent))
    }

    fn register_read_waker(&mut self, context: &Context<'_>) -> io::Result<()> {
        let reactor = ensure_reactor_ptr(&mut self.reactor_ptr)?;
        if self.source.is_none() {
            let key = reactor.register(self.socket.as_raw_fd(), Interest::ReadWrite)?;
            self.source = Some(key);
        }
        let Some(key) = self.source else {
            return Err(io::Error::other("UdpSocket: missing source key"));
        };
        if !reactor.register_read_waker_ref(key, context.waker()) {
            return Err(io::Error::other("UdpSocket: reactor source went stale"));
        }
        Ok(())
    }

    fn register_write_waker(&mut self, context: &Context<'_>) -> io::Result<()> {
        let reactor = ensure_reactor_ptr(&mut self.reactor_ptr)?;
        if self.source.is_none() {
            let key = reactor.register(self.socket.as_raw_fd(), Interest::ReadWrite)?;
            self.source = Some(key);
        }
        let Some(key) = self.source else {
            return Err(io::Error::other("UdpSocket: missing source key"));
        };
        if !reactor.register_write_waker_ref(key, context.waker()) {
            return Err(io::Error::other("UdpSocket: reactor source went stale"));
        }
        Ok(())
    }
}

impl Drop for UdpSocket {
    fn drop(&mut self) {
        deregister_on_drop(&mut self.source, self.reactor_ptr);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::os::core_shard;
    use proxima_runtime::CoreId;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    #[test]
    fn listener_accept_and_stream_echo() {
        // launch a proxima core, run a small echo test on it.
        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let done = Arc::new(AtomicBool::new(false));
        let done_for_factory = done.clone();
        let addr_chan = std::sync::Arc::new(std::sync::Mutex::new(None::<SocketAddr>));
        let addr_for_factory = addr_chan.clone();

        handle
            .dispatch_factory(Box::new(move || {
                let done = done_for_factory.clone();
                let addr_handle = addr_for_factory.clone();
                Box::pin(async move {
                    use futures::io::{AsyncReadExt, AsyncWriteExt};
                    let mut listener =
                        TcpListener::bind("127.0.0.1:0".parse().unwrap()).expect("bind");
                    let bound = listener.local_addr().expect("local_addr");
                    *addr_handle.lock().unwrap() = Some(bound);
                    let (mut stream, _peer) = listener.accept().await.expect("accept");
                    let mut buf = [0u8; 4];
                    stream.read_exact(&mut buf).await.expect("read");
                    stream.write_all(&buf).await.expect("write");
                    done.store(true, Ordering::Release);
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch_factory");

        // wait for the listener to bind.
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        let bound = loop {
            if let Some(addr) = *addr_chan.lock().unwrap() {
                break addr;
            }
            assert!(std::time::Instant::now() < deadline, "listener never bound");
            std::thread::sleep(Duration::from_millis(5));
        };

        // client: open a std TcpStream, write 4 bytes, read 4 bytes.
        let mut client = std::net::TcpStream::connect(bound).expect("connect");
        use std::io::{Read, Write};
        client.write_all(b"ping").expect("client write");
        let mut buf = [0u8; 4];
        client.read_exact(&mut buf).expect("client read");
        assert_eq!(&buf, b"ping");

        // wait for the future to mark done.
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        while !done.load(Ordering::Acquire) {
            assert!(
                std::time::Instant::now() < deadline,
                "future never finished"
            );
            std::thread::sleep(Duration::from_millis(5));
        }

        handle.shutdown_and_join().expect("shutdown");
    }

    /// regression: an earlier version cached the last-registered waker in
    /// the TcpStream itself and short-circuited `register_*_waker` when the
    /// new poll's waker matched. The reactor's `turn`, however, *takes* the
    /// slot's waker on fire (the slot is left waker-less), so subsequent
    /// pending polls saw a stale local cache, skipped re-registering, and
    /// the slot was never re-armed → deadlock on the second iteration.
    /// this test does N read/wait/read cycles on one socket; a stalled
    /// elision would hang past the deadline.
    #[test]
    fn many_read_block_cycles_do_not_starve() {
        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let done = Arc::new(AtomicBool::new(false));
        let done_for_factory = done.clone();
        let addr_chan = std::sync::Arc::new(std::sync::Mutex::new(None::<SocketAddr>));
        let addr_for_factory = addr_chan.clone();

        // server: accept, read 8 messages of 4 bytes each, with the client
        // intentionally pausing between sends so every read goes through a
        // pending → wake → ready cycle.
        handle
            .dispatch_factory(Box::new(move || {
                let done = done_for_factory.clone();
                let addr_handle = addr_for_factory.clone();
                Box::pin(async move {
                    use futures::io::AsyncReadExt;
                    let mut listener =
                        TcpListener::bind("127.0.0.1:0".parse().unwrap()).expect("bind");
                    let bound = listener.local_addr().expect("local_addr");
                    *addr_handle.lock().unwrap() = Some(bound);
                    let (mut stream, _peer) = listener.accept().await.expect("accept");
                    for _ in 0..8 {
                        let mut buf = [0u8; 4];
                        stream.read_exact(&mut buf).await.expect("read");
                    }
                    done.store(true, Ordering::Release);
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch_factory");

        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        let bound = loop {
            if let Some(addr) = *addr_chan.lock().unwrap() {
                break addr;
            }
            assert!(std::time::Instant::now() < deadline, "listener never bound");
            std::thread::sleep(Duration::from_millis(5));
        };

        // client: write 4 bytes, sleep 20ms, repeat 8 times. The sleep
        // forces the server's read into Pending → reactor fires → ready.
        use std::io::Write;
        let mut client = std::net::TcpStream::connect(bound).expect("connect");
        for _ in 0..8 {
            client.write_all(b"ping").expect("client write");
            client.flush().expect("client flush");
            std::thread::sleep(Duration::from_millis(20));
        }

        // generous deadline; failure mode is hang, so any reasonable bound works.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !done.load(Ordering::Acquire) {
            assert!(
                std::time::Instant::now() < deadline,
                "server never observed all 8 reads — waker elision regression"
            );
            std::thread::sleep(Duration::from_millis(5));
        }

        handle.shutdown_and_join().expect("shutdown");
    }

    /// C28 UdpSocket worked example: bind a non-blocking UDP socket on
    /// a proxima worker, send one datagram from a std-blocking
    /// `std::net::UdpSocket` to it, and verify the server receives the
    /// bytes + peer addr.
    #[test]
    fn udp_socket_recv_from_round_trips_one_datagram() {
        use core::future::poll_fn;

        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let done = Arc::new(AtomicBool::new(false));
        let done_for_factory = done.clone();
        let addr_chan = std::sync::Arc::new(std::sync::Mutex::new(None::<SocketAddr>));
        let addr_for_factory = addr_chan.clone();

        handle
            .dispatch_factory(Box::new(move || {
                let done = done_for_factory.clone();
                let addr_handle = addr_for_factory.clone();
                Box::pin(async move {
                    let mut socket = UdpSocket::bind("127.0.0.1:0".parse().unwrap()).expect("bind");
                    let bound = socket.local_addr().expect("local_addr");
                    *addr_handle.lock().unwrap() = Some(bound);
                    let mut buf = [0u8; 16];
                    let (len, _peer) =
                        poll_fn(|cx| Pin::new(&mut socket).poll_recv_from(cx, &mut buf))
                            .await
                            .expect("recv");
                    assert_eq!(&buf[..len], b"udp-hello");
                    done.store(true, Ordering::Release);
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch_factory");

        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        let bound = loop {
            if let Some(addr) = *addr_chan.lock().unwrap() {
                break addr;
            }
            assert!(std::time::Instant::now() < deadline, "UDP never bound");
            std::thread::sleep(Duration::from_millis(5));
        };

        // Give the server a chance to call recv_from + park.
        std::thread::sleep(Duration::from_millis(20));
        let client = std::net::UdpSocket::bind("127.0.0.1:0").expect("client bind");
        client.send_to(b"udp-hello", bound).expect("client send");

        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        while !done.load(Ordering::Acquire) {
            assert!(
                std::time::Instant::now() < deadline,
                "udp recv never completed"
            );
            std::thread::sleep(Duration::from_millis(5));
        }

        handle.shutdown_and_join().expect("shutdown");
    }

    /// SO_REUSEPORT regression: the per-core serve model binds ONE port once per
    /// core to form a load-balancing datagram group. Without SO_REUSEPORT (only
    /// SO_REUSEADDR) the second UDP bind on the same explicit port fails
    /// EADDRINUSE on Linux, the group never forms, and a multi-core UDP/QUIC
    /// server pins one core. Two binds on the same port succeeding proves the
    /// flag is set.
    #[test]
    fn udp_socket_reuse_port_allows_two_binds_on_same_port() {
        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let second_ok = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));
        let ok_for_factory = second_ok.clone();
        let done_for_factory = done.clone();

        handle
            .dispatch_factory(Box::new(move || {
                let ok = ok_for_factory.clone();
                let done = done_for_factory.clone();
                Box::pin(async move {
                    let first =
                        UdpSocket::bind("127.0.0.1:0".parse().unwrap()).expect("first bind");
                    let port = first.local_addr().expect("local_addr").port();
                    let same: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
                    let second = UdpSocket::bind(same);
                    ok.store(second.is_ok(), Ordering::Release);
                    // hold both so neither drops before the flag is read.
                    let _keep = (first, second);
                    done.store(true, Ordering::Release);
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch_factory");

        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        while !done.load(Ordering::Acquire) {
            assert!(
                std::time::Instant::now() < deadline,
                "reuseport bind never ran"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            second_ok.load(Ordering::Acquire),
            "second UdpSocket bind on the same port failed — SO_REUSEPORT not set"
        );
        handle.shutdown_and_join().expect("shutdown");
    }

    /// Batched recv (`recvmmsg` on linux, looped `recv_from` elsewhere) drains
    /// a whole burst, and batched send (`sendmmsg`/looped `send_to`) ships the
    /// echoes back — proving the one-syscall fast path round-trips real
    /// datagrams to the right peers.
    #[test]
    fn udp_socket_batch_echoes_a_burst() {
        use core::future::poll_fn;

        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let done = Arc::new(AtomicBool::new(false));
        let done_for_factory = done.clone();
        let addr_chan = std::sync::Arc::new(std::sync::Mutex::new(None::<SocketAddr>));
        let addr_for_factory = addr_chan.clone();

        handle
            .dispatch_factory(Box::new(move || {
                let done = done_for_factory.clone();
                let addr_handle = addr_for_factory.clone();
                Box::pin(async move {
                    let mut socket = UdpSocket::bind("127.0.0.1:0".parse().unwrap()).expect("bind");
                    *addr_handle.lock().unwrap() = Some(socket.local_addr().expect("local_addr"));
                    let mut storage: Vec<[u8; 2048]> = vec![[0u8; 2048]; 8];
                    let mut meta = vec![(0usize, SocketAddr::from(([0u8, 0, 0, 0], 0))); 8];
                    let mut total = 0;
                    while total < 3 {
                        let mut refs: Vec<&mut [u8]> =
                            storage.iter_mut().map(|slot| slot.as_mut_slice()).collect();
                        let count = poll_fn(|cx| {
                            Pin::new(&mut socket).poll_recv_batch(cx, &mut refs, &mut meta)
                        })
                        .await
                        .expect("recv_batch");
                        drop(refs);
                        // echo each received datagram straight back to its sender.
                        let packets: Vec<(&[u8], SocketAddr)> = (total..total + count)
                            .map(|index| {
                                let (len, peer) = meta[index - total];
                                (&storage[index - total][..len], peer)
                            })
                            .collect();
                        let sent =
                            poll_fn(|cx| Pin::new(&mut socket).poll_send_batch(cx, &packets))
                                .await
                                .expect("send_batch");
                        assert_eq!(sent, count, "send_batch shipped every echo");
                        total += count;
                    }
                    done.store(true, Ordering::Release);
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch_factory");

        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        let bound = loop {
            if let Some(addr) = *addr_chan.lock().unwrap() {
                break addr;
            }
            assert!(std::time::Instant::now() < deadline, "UDP never bound");
            std::thread::sleep(Duration::from_millis(5));
        };

        std::thread::sleep(Duration::from_millis(20));
        let client = std::net::UdpSocket::bind("127.0.0.1:0").expect("client bind");
        client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .expect("read timeout");
        let payloads: [&[u8]; 3] = [b"alpha", b"bravo", b"charlie"];
        for payload in payloads {
            client.send_to(payload, bound).expect("client send");
        }

        let mut echoed: Vec<Vec<u8>> = Vec::new();
        for _ in 0..3 {
            let mut buf = [0u8; 32];
            let (len, _peer) = client.recv_from(&mut buf).expect("client recv echo");
            echoed.push(buf[..len].to_vec());
        }

        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        while !done.load(Ordering::Acquire) {
            assert!(
                std::time::Instant::now() < deadline,
                "batch echo never completed"
            );
            std::thread::sleep(Duration::from_millis(5));
        }

        for payload in payloads {
            assert!(
                echoed.iter().any(|datagram| datagram.as_slice() == payload),
                "echo missing for {payload:?}"
            );
        }

        handle.shutdown_and_join().expect("shutdown");
    }

    /// regression guard for the spurious-poll bug in the `Connect` future.
    ///
    /// technique: a `std::net::TcpListener` is bound and its accept queue is
    /// saturated by 30 blocking connections held open from a background
    /// thread.  with the queue full, new non-blocking `connect(2)` calls
    /// return `EINPROGRESS` (the kernel drops incoming SYNs when the SYN
    /// queue overflows, so no SYN-ACK is sent and the socket stays in
    /// `SYN_SENT`).  the `Connect` future's first poll therefore transitions
    /// `Init → Pending` and returns `Pending`.  the second poll is the
    /// "spurious" poll under test: before any writable edge fires, polling
    /// again must return `Pending` — not a half-open `TcpStream`.
    ///
    /// the inner poll is exercised via `std::future::poll_fn` wrapping:
    /// `poll_fn(|cx| Poll::Ready(connect.as_mut().poll(cx)))` immediately
    /// resolves and hands back the raw `Poll` value from the inner future.
    #[test]
    fn connect_pending_survives_spurious_poll() {
        use std::future::poll_fn;
        use std::task::Poll;

        // a connect to a routable but non-responsive address (RFC 5737
        // TEST-NET-1) holds EINPROGRESS deterministically: the SYN follows
        // the default route and is never answered, so both back-to-back polls
        // observe the in-progress state long before any ICMP reply. backlog
        // saturation does NOT work cross-platform — linux completes the
        // loopback handshake into a full accept queue instead of dropping the
        // SYN, so the connection lands between the two polls.
        let blackhole: std::net::SocketAddr = "192.0.2.1:80".parse().expect("addr");

        let result_chan: Arc<Mutex<Option<(bool, bool)>>> = Arc::new(std::sync::Mutex::new(None));
        let result_for_factory = result_chan.clone();
        let done = Arc::new(AtomicBool::new(false));
        let done_clone = done.clone();

        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");

        handle
            .dispatch_factory(Box::new(move || {
                let done = done_clone.clone();
                let result_handle = result_for_factory.clone();
                Box::pin(async move {
                    let mut connect = Box::pin(TcpStream::connect(blackhole));

                    // first poll: Init → Pending (EINPROGRESS, SYN unanswered)
                    let first: Poll<io::Result<TcpStream>> =
                        poll_fn(|cx| Poll::Ready(connect.as_mut().poll(cx))).await;
                    let first_is_pending = matches!(first, Poll::Pending);

                    // spurious poll: Pending arm re-probes via connect() syscall;
                    // must classify EALREADY/EINPROGRESS → InProgress → Pending.
                    let second: Poll<io::Result<TcpStream>> =
                        poll_fn(|cx| Poll::Ready(connect.as_mut().poll(cx))).await;
                    let second_is_pending = matches!(second, Poll::Pending);

                    *result_handle.lock().unwrap() = Some((first_is_pending, second_is_pending));
                    done.store(true, Ordering::Release);
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch_factory");

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !done.load(Ordering::Acquire) {
            assert!(
                std::time::Instant::now() < deadline,
                "spurious-poll test future never completed"
            );
            std::thread::sleep(Duration::from_millis(5));
        }

        handle.shutdown_and_join().expect("shutdown");

        let (first_is_pending, second_is_pending) =
            result_chan.lock().unwrap().expect("result not set");

        assert!(
            first_is_pending,
            "first poll returned Ready — connect did not enter EINPROGRESS \
             (no default route to the TEST-NET blackhole?)"
        );
        assert!(
            second_is_pending,
            "spurious poll returned Ready — spurious-poll bug regression: \
             Connect yielded a half-open TcpStream before the writable edge"
        );
    }
}
