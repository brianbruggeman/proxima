//! `RedisConnSource<R>` — the caller-supplied `UnpinPipe<In = (), Out =
//! WireEvent, Err = Exhausted> + DropSafe` enum
//! [`proxima_listen::wait_for_wire_event`] merges via `FanIn`, replacing
//! `connection.rs`'s own `select_biased!` (shutdown / pub-sub push /
//! socket read) with the shared outer-wait driver.
//!
//! Every variant only WAITS — it never dispatches the business handler
//! (that stays sequential in `connection.rs`'s own inner loop; see
//! `proxima_listen::serve_multiplexed`'s module doc for why that can't be
//! merged into this same race). Reading and peeking a channel are both
//! [`DropSafe`]: a read future dropped mid-poll loses nothing (the kernel
//! still holds the unread bytes for the next call), and a channel `.next()`
//! future dropped mid-`Pending` has registered no side effect either — the
//! receiver itself remembers readiness, not the transient poll future (see
//! `proxima_core::markers::DropSafe`'s own doc + `fan_in.rs:274/288/302`'s
//! enforced bound on every merged source).
//!
//! # Why `proxima_primitives::sync::blocking::Mutex`, not `RefCell`
//!
//! `FanIn::call` takes `&self` (`fan_in.rs`'s own contract), so a source
//! that needs `&mut` access to its read half / channel / oneshot needs
//! interior mutability. `RefCell` is the obvious first reach, but it is
//! never `Sync` — and `RedisConnectionPipe: SendPipe`'s upgrade closure
//! (`pipe.rs`, `UpgradeHandler::new`) requires the WHOLE connection future
//! to be `Send`, a pre-existing constraint from the cross-core
//! `SendPipe`/`AnyProtocol` contract, unrelated to and predating this
//! retrofit. `&FanIn<RedisConnSource<..>, ..>` crossing an `.await` demands
//! `RedisConnSource: Sync`, which `RefCell` can never provide. `Mutex` is
//! the minimal fix: never contended (one task drives the whole `FanIn`
//! sequentially, and `FanIn` itself never polls two sources at once) and
//! never held across an `.await` (acquired and released within one
//! synchronous `poll()`), so this is a correctness-only cost (a few atomic
//! ops per poll), not a real lock-contention concern — but it IS a real
//! deviation from the RefCell-shaped design the design spike assumed. The
//! workspace-canonical sync `Mutex` (`proxima_primitives::sync::blocking`,
//! parking_lot-backed on std) is used rather than `std::sync::Mutex` — same
//! usage, no poisoning to recover from.

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use bytes::Bytes;
use futures::channel::mpsc::UnboundedReceiver;
use futures::channel::oneshot;
use futures::io::AsyncRead;
use futures::stream::Stream;

use proxima_core::markers::DropSafe;
use proxima_listen::WireEvent;
use proxima_primitives::pipe::{Exhausted, UnpinPipe};
use proxima_primitives::sync::blocking::{Mutex, MutexGuard};

/// `parking_lot::Mutex` never poisons, so this is a plain passthrough — kept
/// as a named helper so every call site reads `lock(mutex)` uniformly rather
/// than `mutex.lock()` in some places and a poison-recovery dance in others.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock()
}

/// One of the three sources `connection.rs`'s outer wait races: the socket
/// read half, this connection's pub/sub push channel, or the listener's
/// graceful-shutdown signal. Generic over `R` (the read half's concrete
/// type) so it carries no dependency on the listener's own socket type.
pub enum RedisConnSource<R> {
    Read {
        read_half: Mutex<R>,
        scratch: Mutex<std::vec::Vec<u8>>,
    },
    Push {
        push_rx: Mutex<UnboundedReceiver<Bytes>>,
    },
    Shutdown {
        receiver: Mutex<Option<oneshot::Receiver<()>>>,
    },
}

impl<R> RedisConnSource<R> {
    #[must_use]
    pub fn read(read_half: R, read_buffer_bytes: usize) -> Self {
        Self::Read {
            read_half: Mutex::new(read_half),
            scratch: Mutex::new(std::vec![0_u8; read_buffer_bytes.max(1)]),
        }
    }

    #[must_use]
    pub fn push(push_rx: UnboundedReceiver<Bytes>) -> Self {
        Self::Push {
            push_rx: Mutex::new(push_rx),
        }
    }

    #[must_use]
    pub fn shutdown(receiver: oneshot::Receiver<()>) -> Self {
        Self::Shutdown {
            receiver: Mutex::new(Some(receiver)),
        }
    }
}

// justified: see this module's doc — every variant only observes readiness
// (a socket read, a channel peek, a oneshot poll), none of them mid-sends a
// partial reply, so dropping an in-flight `call` future mid-poll leaves no
// torn state.
impl<R> DropSafe for RedisConnSource<R> {}

/// The future behind [`RedisConnSource::call`] — one hand-written `poll`
/// spanning all three variants (an `async move` block is never provably
/// `Unpin`, which `UnpinPipe::call`'s contract requires; see
/// `proxima_primitives::pipe::primitives`'s own module doc). `Unpin`
/// unconditionally: every field is a plain reference into the source's own
/// `Mutex`es, never a pinned value.
enum RedisConnCall<'source, R> {
    Read {
        read_half: &'source Mutex<R>,
        scratch: &'source Mutex<std::vec::Vec<u8>>,
    },
    Push {
        push_rx: &'source Mutex<UnboundedReceiver<Bytes>>,
    },
    Shutdown {
        receiver: &'source Mutex<Option<oneshot::Receiver<()>>>,
    },
}

impl<R: AsyncRead + Unpin> Future for RedisConnCall<'_, R> {
    type Output = Result<WireEvent, Exhausted>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.get_mut() {
            RedisConnCall::Read { read_half, scratch } => {
                let mut read_half = lock(read_half);
                let mut scratch = lock(scratch);
                match Pin::new(&mut *read_half).poll_read(cx, &mut scratch) {
                    Poll::Ready(Ok(0)) => Poll::Ready(Ok(WireEvent::Stop)),
                    Poll::Ready(Ok(count)) => Poll::Ready(Ok(WireEvent::Read(
                        Bytes::copy_from_slice(&scratch[..count]),
                    ))),
                    Poll::Ready(Err(error)) => Poll::Ready(Ok(WireEvent::Failed(error))),
                    Poll::Pending => Poll::Pending,
                }
            }
            RedisConnCall::Push { push_rx } => {
                let mut push_rx = lock(push_rx);
                match Pin::new(&mut *push_rx).poll_next(cx) {
                    Poll::Ready(Some(bytes)) => Poll::Ready(Ok(WireEvent::Push(bytes))),
                    // the sender half lives on this same connection's
                    // `PushSink`, held for the connection's whole lifetime —
                    // `None` cannot happen while the loop runs (mirrors the
                    // house style already documented on the old
                    // `select_biased!` arm this replaces); treated as
                    // "will never produce again" rather than a hard error.
                    Poll::Ready(None) => Poll::Ready(Err(Exhausted)),
                    Poll::Pending => Poll::Pending,
                }
            }
            RedisConnCall::Shutdown { receiver } => {
                let mut receiver = lock(receiver);
                match receiver.as_mut() {
                    None => Poll::Ready(Err(Exhausted)),
                    Some(inner) => match Pin::new(inner).poll(cx) {
                        Poll::Ready(_fired_or_dropped) => {
                            *receiver = None;
                            Poll::Ready(Ok(WireEvent::Stop))
                        }
                        Poll::Pending => Poll::Pending,
                    },
                }
            }
        }
    }
}

impl<R: AsyncRead + Unpin> UnpinPipe for RedisConnSource<R> {
    type In = ();
    type Out = WireEvent;
    type Err = Exhausted;

    fn call(&self, (): ()) -> impl Future<Output = Result<WireEvent, Exhausted>> + Unpin {
        match self {
            RedisConnSource::Read { read_half, scratch } => RedisConnCall::Read {
                read_half,
                scratch,
            },
            RedisConnSource::Push { push_rx } => RedisConnCall::Push { push_rx },
            RedisConnSource::Shutdown { receiver } => RedisConnCall::Shutdown { receiver },
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::channel::mpsc;
    use std::io;
    use std::task::Waker;

    fn block_on<Fut: Future>(future: Fut) -> Fut::Output {
        let mut pinned = core::pin::pin!(future);
        let mut context = Context::from_waker(Waker::noop());
        loop {
            if let Poll::Ready(output) = pinned.as_mut().poll(&mut context) {
                return output;
            }
        }
    }

    // an in-memory reader yielding one fixed chunk, then EOF forever after.
    struct OnceReader {
        chunk: std::vec::Vec<u8>,
        served: bool,
    }

    impl AsyncRead for OnceReader {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<io::Result<usize>> {
            if self.served {
                return Poll::Ready(Ok(0));
            }
            let count = self.chunk.len().min(buf.len());
            buf[..count].copy_from_slice(&self.chunk[..count]);
            self.served = true;
            Poll::Ready(Ok(count))
        }
    }

    #[test]
    fn read_source_yields_bytes_then_stop_on_eof() {
        let source = RedisConnSource::read(
            OnceReader {
                chunk: b"PING\r\n".to_vec(),
                served: false,
            },
            64,
        );
        let first = block_on(UnpinPipe::call(&source, ())).expect("first poll");
        match first {
            WireEvent::Read(bytes) => assert_eq!(&bytes[..], b"PING\r\n"),
            other => panic!("expected Read, got {other:?}"),
        }
        let second = block_on(UnpinPipe::call(&source, ())).expect("second poll");
        assert!(matches!(second, WireEvent::Stop), "EOF must be Stop, not Exhausted (only ONE source dying must not stall the whole wait)");
    }

    struct FailingReader;

    impl AsyncRead for FailingReader {
        fn poll_read(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            _buf: &mut [u8],
        ) -> Poll<io::Result<usize>> {
            Poll::Ready(Err(io::Error::other("boom")))
        }
    }

    #[test]
    fn read_source_surfaces_a_real_io_error_as_failed_not_stop() {
        let source = RedisConnSource::read(FailingReader, 64);
        let outcome = block_on(UnpinPipe::call(&source, ())).expect("poll");
        assert!(
            matches!(outcome, WireEvent::Failed(_)),
            "a hard io error must not be silently downgraded to Stop"
        );
    }

    #[test]
    fn push_source_yields_a_pushed_item() {
        let (tx, rx) = mpsc::unbounded::<Bytes>();
        tx.unbounded_send(Bytes::from_static(b"+OK\r\n")).expect("send");
        let source = RedisConnSource::<OnceReader>::push(rx);
        let outcome = block_on(UnpinPipe::call(&source, ())).expect("poll");
        match outcome {
            WireEvent::Push(bytes) => assert_eq!(&bytes[..], b"+OK\r\n"),
            other => panic!("expected Push, got {other:?}"),
        }
    }

    #[test]
    fn shutdown_source_fires_stop_once_signaled() {
        let (tx, rx) = oneshot::channel::<()>();
        let source = RedisConnSource::<OnceReader>::shutdown(rx);
        tx.send(()).expect("send shutdown");
        let outcome = block_on(UnpinPipe::call(&source, ())).expect("poll");
        assert!(matches!(outcome, WireEvent::Stop));
    }
}
