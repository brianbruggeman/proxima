//! Connection upgrade / hijack.
//!
//! After the H1 listener writes a `200 Connection established` (for
//! CONNECT) or `101 Switching Protocols` (for Upgrade) response
//! head, the H1 framing is done with this socket — subsequent bytes
//! belong to the new protocol. The listener cedes the raw socket to
//! a Pipe-provided `UpgradeHandler` which drives whatever comes
//! next: a TCP tunnel, a WebSocket loop, h2c, etc.
//!
//! The listener itself stays protocol-agnostic. It only knows: "this
//! Response carries an upgrade handler — write the head, then call
//! the handler with the unsplit socket and any bytes I happened to
//! buffer past the request head."
//!
//! For CONNECT, the request body is empty by definition (no
//! Content-Length, no Transfer-Encoding), so `leftover` is normally
//! empty too. An aggressive client that pipelined tunnel data ahead
//! of the server's 200 will have those bytes surfaced via
//! `leftover` — the handler must drain that before reading the
//! socket.

#[cfg(feature = "std")]
use core::future::Future;
#[cfg(feature = "std")]
use core::pin::Pin;

#[cfg(feature = "std")]
use alloc::boxed::Box;
#[cfg(feature = "std")]
use bytes::Bytes;
#[cfg(feature = "std")]
use futures::io::{AsyncRead, AsyncWrite};
#[cfg(feature = "std")]
use proxima_core::ProximaError;

/// Trait alias for the socket types the listener can hand off, keyed on
/// `futures::io::{AsyncRead, AsyncWrite} + Send + Unpin`. Runtime-agnostic
/// at the trait surface — tokio-shaped streams cross the boundary via
/// `tokio_util::compat::TokioAsyncReadCompatExt` inside the listener glue.
#[cfg(feature = "std")]
pub trait HijackStream: AsyncRead + AsyncWrite + Send + Unpin {}
#[cfg(feature = "std")]
impl<T: AsyncRead + AsyncWrite + Send + Unpin + ?Sized> HijackStream for T {}

/// Socket handed to an upgrade handler. Owns the underlying stream
/// (the listener has dropped its references) and surfaces any bytes
/// the listener buffered past the request head.
#[cfg(feature = "std")]
pub struct HijackedSocket {
    pub stream: Box<dyn HijackStream>,
    pub leftover: Bytes,
}

#[cfg(feature = "std")]
impl HijackedSocket {
    #[must_use]
    pub fn new(stream: Box<dyn HijackStream>, leftover: Bytes) -> Self {
        Self { stream, leftover }
    }
}

#[cfg(feature = "std")]
pub type UpgradeFuture = Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send>>;

/// Pipe-provided hook that runs the post-upgrade protocol. The
/// listener invokes it exactly once after writing the upgrade
/// response head. The handler owns the socket for its lifetime;
/// dropping the future closes the socket.
#[cfg(feature = "std")]
pub struct UpgradeHandler {
    inner: Box<dyn FnOnce(HijackedSocket) -> UpgradeFuture + Send>,
}

#[cfg(feature = "std")]
impl UpgradeHandler {
    #[must_use]
    pub fn new<F, Fut>(handler: F) -> Self
    where
        F: FnOnce(HijackedSocket) -> Fut + Send + 'static,
        Fut: Future<Output = Result<(), ProximaError>> + Send + 'static,
    {
        Self {
            inner: Box::new(move |socket| Box::pin(handler(socket))),
        }
    }

    pub fn invoke(self, socket: HijackedSocket) -> UpgradeFuture {
        (self.inner)(socket)
    }
}

#[cfg(feature = "std")]
impl core::fmt::Debug for UpgradeHandler {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("UpgradeHandler")
            .finish_non_exhaustive()
    }
}

/// Erased `() -> UpgradeHandler` accept hook: the connection-layer sibling
/// of [`crate::pipe::handler::PipeHandle`]. An unconditional-upgrade
/// protocol pipe (redis/pgwire/mqtt/amqp) upgrades every accepted socket
/// with zero dependence on the inbound bytes, so its `SendPipe::In` is
/// `()` rather than a synthetic `Request<Bytes>`.
#[cfg(feature = "std")]
pub type AcceptHandle = crate::pipe::alloc_tier::PipeHandle<(), UpgradeHandler>;

// Local (?Send) upgrade variant for the io_uring listener path.
//
// io_uring's `tokio_uring::net::TcpStream` is `!Send` (held in `Rc`),
// so the post-upgrade handler must be `!Send` too. The Send-bound
// `UpgradeHandler` above can't carry an `Rc<TcpStream>`-backed
// stream — hence the parallel `Local*` types here.
//
// Pipe authors install a `LocalUpgradeHandler` via the
// thread-local registry (see `local_slots` below) using the ticket
// the io_uring listener placed in `request.context.local_upgrade_ticket`.
// The listener picks the handler back up after `Pipe::call` returns
// and hijacks the socket. The Send-bounded `Response.upgrade` field
// is irrelevant on the io_uring path and is ignored when a local
// handler is installed.

#[cfg(feature = "std")]
pub trait LocalHijackStream: AsyncRead + AsyncWrite + Unpin {}
#[cfg(feature = "std")]
impl<T: AsyncRead + AsyncWrite + Unpin + ?Sized> LocalHijackStream for T {}

#[cfg(feature = "std")]
pub struct LocalHijackedSocket {
    pub stream: Box<dyn LocalHijackStream>,
    pub leftover: Bytes,
}

#[cfg(feature = "std")]
impl LocalHijackedSocket {
    #[must_use]
    pub fn new(stream: Box<dyn LocalHijackStream>, leftover: Bytes) -> Self {
        Self { stream, leftover }
    }
}

#[cfg(feature = "std")]
pub type LocalUpgradeFuture = Pin<Box<dyn Future<Output = Result<(), ProximaError>>>>;

#[cfg(feature = "std")]
pub struct LocalUpgradeHandler {
    inner: Box<dyn FnOnce(LocalHijackedSocket) -> LocalUpgradeFuture>,
}

#[cfg(feature = "std")]
impl LocalUpgradeHandler {
    #[must_use]
    pub fn new<F, Fut>(handler: F) -> Self
    where
        F: FnOnce(LocalHijackedSocket) -> Fut + 'static,
        Fut: Future<Output = Result<(), ProximaError>> + 'static,
    {
        Self {
            inner: Box::new(move |socket| Box::pin(handler(socket))),
        }
    }

    pub fn invoke(self, socket: LocalHijackedSocket) -> LocalUpgradeFuture {
        (self.inner)(socket)
    }
}

#[cfg(feature = "std")]
impl core::fmt::Debug for LocalUpgradeHandler {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("LocalUpgradeHandler")
            .finish_non_exhaustive()
    }
}

/// Thread-local installer registry for io_uring upgrade handlers.
///
/// The io_uring listener mints a ticket per request and places it in
/// `request.context.local_upgrade_ticket` before dispatching. Pipe
/// authors that want to upgrade call [`install`] with that ticket and
/// a `LocalUpgradeHandler`. The listener calls [`take`] after dispatch
/// completes to retrieve the handler and hijack the socket.
///
/// The registry is `thread_local!` because the io_uring listener
/// pins each accepted connection to one OS thread (via `spawn_local`
/// on `tokio_uring`'s current-thread runtime). The Pipe future is
/// polled on the same thread for its lifetime, so install + take see
/// the same map.
///
/// Gotcha: don't call [`install`] from a `tokio::spawn`'d task nested
/// inside `Pipe::call` — that task may run on a different OS
/// thread and miss the listener's `take`. Install directly from the
/// `Pipe::call` future.
#[cfg(feature = "std")]
pub mod local_slots {
    use super::LocalUpgradeHandler;
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    thread_local! {
        static SLOTS: RefCell<BTreeMap<u64, LocalUpgradeHandler>> = const { RefCell::new(BTreeMap::new()) };
    }

    static TICKET_GEN: AtomicU64 = AtomicU64::new(1);

    #[must_use]
    pub fn next_ticket() -> u64 {
        TICKET_GEN.fetch_add(1, Ordering::Relaxed)
    }

    pub fn install(ticket: u64, handler: LocalUpgradeHandler) {
        SLOTS.with(|cell| {
            cell.borrow_mut().insert(ticket, handler);
        });
    }

    pub fn take(ticket: u64) -> Option<LocalUpgradeHandler> {
        SLOTS.with(|cell| cell.borrow_mut().remove(&ticket))
    }

    pub fn discard(ticket: u64) {
        let _ = take(ticket);
    }
}
