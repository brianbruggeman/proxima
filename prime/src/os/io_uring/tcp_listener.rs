//! io_uring-backed TCP listener for the prime runtime.
//!
//! uses the `Accept` opcode to accept connections without polling. the
//! accepted fd is returned in the CQE result field.
//!
//! ## Send / thread affinity
//!
//! same contract as `net::TcpListener`: `Send` is restored via `unsafe impl`
//! so the type composes with executor APIs requiring `Send`. polling must
//! remain on the worker thread that submitted the SQE; the prime runtime
//! enforces this via its per-core (no-work-stealing) topology.

#![cfg(all(
    target_os = "linux",
    feature = "io-uring",
    feature = "runtime-prime-reactor"
))]

use std::io;
use std::marker::PhantomData;
use std::net::SocketAddr;
use std::os::fd::AsRawFd;
use std::pin::Pin;
use std::task::{Context, Poll};

use io_uring::{opcode, types};
use socket2::{Domain, Protocol, SockAddr, Socket, Type};

use super::reactor::{ParkedResource, with_current_uring};
use super::tcp_stream::TcpStream;

/// state machine for a single in-flight accept.
enum AcceptState {
    /// no accept submitted yet.
    Idle,
    /// SQE submitted; `user_data` identifies the slab slot. the Box fields
    /// are stable heap addresses that the kernel writes into.
    InFlight {
        user_data: u64,
        addr_storage: Box<libc::sockaddr_storage>,
        addr_len: Box<libc::socklen_t>,
    },
}

/// io_uring TCP listener. bind + accept using the `Accept` opcode.
///
/// `Send` is deliberately restored — see module doc for the contract.
pub struct TcpListener {
    socket: Socket,
    accept_state: AcceptState,
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

// SAFETY: see module doc. `*mut` fields from the socket2 crate are the only
// !Send pieces. the proxima per-core runtime never migrates tasks.
unsafe impl Send for TcpListener {}

impl TcpListener {
    /// bind a non-blocking TCP listening socket.
    pub fn bind(addr: SocketAddr) -> io::Result<Self> {
        let domain = match addr {
            SocketAddr::V4(_) => Domain::IPV4,
            SocketAddr::V6(_) => Domain::IPV6,
        };
        let socket = Socket::new(domain, Type::STREAM, Some(Protocol::TCP))?;
        socket.set_nonblocking(true)?;
        socket.set_reuse_address(true)?;
        let sock_addr = SockAddr::from(addr);
        socket.bind(&sock_addr)?;
        socket.listen(1024)?;
        Ok(Self {
            socket,
            accept_state: AcceptState::Idle,
            _not_sync: PhantomData,
        })
    }

    /// the socket address this listener bound to.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.socket
            .local_addr()?
            .as_socket()
            .ok_or_else(|| io::Error::other("local_addr: not an IP socket"))
    }

    /// accept one connection. returns a future that resolves when a client
    /// connects. poll-safe to call repeatedly from the same task.
    pub fn accept(&mut self) -> Accept<'_> {
        Accept { listener: self }
    }

    pub(super) fn poll_accept(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<io::Result<(TcpStream, SocketAddr)>> {
        let this = self.get_mut();

        // drain completed CQEs (opportunistic, non-blocking).
        if let Err(err) = with_current_uring(|reactor| reactor.drain_cqes()) {
            return Poll::Ready(Err(err));
        }

        match &this.accept_state {
            AcceptState::InFlight { user_data, .. } => {
                let user_data = *user_data;
                let result = with_current_uring(|reactor| Ok(reactor.take_result(user_data)));
                match result {
                    Err(err) => return Poll::Ready(Err(err)),
                    Ok(Some(res)) => {
                        let AcceptState::InFlight {
                            addr_storage,
                            addr_len,
                            ..
                        } = std::mem::replace(&mut this.accept_state, AcceptState::Idle)
                        else {
                            unreachable!()
                        };
                        return if res < 0 {
                            Poll::Ready(Err(io::Error::from_raw_os_error(-res)))
                        } else {
                            let accepted_fd = res;
                            let stream = build_stream(accepted_fd)?;
                            let peer = parse_peer_addr(&addr_storage, *addr_len)?;
                            Poll::Ready(Ok((stream, peer)))
                        };
                    }
                    Ok(None) => {
                        // still in flight — update waker and return Pending.
                        let _ = with_current_uring(|reactor| {
                            reactor.set_waker(user_data, context.waker().clone());
                            Ok(())
                        });
                        return Poll::Pending;
                    }
                }
            }
            AcceptState::Idle => {}
        }

        // submit a new Accept SQE.
        let submit_result = with_current_uring(|reactor| {
            let mut addr_storage: Box<libc::sockaddr_storage> =
                Box::new(unsafe { std::mem::zeroed() });
            let mut addr_len: Box<libc::socklen_t> =
                Box::new(std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t);

            let user_data = reactor.register_op();

            let sqe = opcode::Accept::new(
                types::Fd(this.socket.as_raw_fd()),
                addr_storage.as_mut() as *mut libc::sockaddr_storage as *mut _,
                addr_len.as_mut() as *mut libc::socklen_t,
            )
            .build()
            .user_data(user_data);

            unsafe {
                reactor
                    .ring_mut()
                    .submission()
                    .push(&sqe)
                    .map_err(|_| io::Error::other("io_uring submission queue full"))?;
            }

            reactor.set_waker(user_data, context.waker().clone());

            Ok((user_data, addr_storage, addr_len))
        });

        match submit_result {
            Err(err) => Poll::Ready(Err(err)),
            Ok((user_data, addr_storage, addr_len)) => {
                this.accept_state = AcceptState::InFlight {
                    user_data,
                    addr_storage,
                    addr_len,
                };
                Poll::Pending
            }
        }
    }
}

impl Drop for TcpListener {
    fn drop(&mut self) {
        // take ownership rather than borrowing: the kernel may still write
        // into addr_storage/addr_len, so they must be handed to the reactor
        // instead of being freed by this struct's own field teardown.
        let previous = std::mem::replace(&mut self.accept_state, AcceptState::Idle);
        if let AcceptState::InFlight {
            user_data,
            addr_storage,
            addr_len,
        } = previous
        {
            let _ = with_current_uring(|reactor| {
                reactor.cancel_op(
                    user_data,
                    ParkedResource::AcceptStorage {
                        addr_storage,
                        addr_len,
                    },
                );
                Ok(())
            });
        }
    }
}

/// future returned by `TcpListener::accept`.
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

/// wrap an accepted raw fd in a non-blocking socket2 Socket, then TcpStream.
fn build_stream(fd: i32) -> io::Result<TcpStream> {
    use std::os::fd::FromRawFd;
    let socket = unsafe { Socket::from_raw_fd(fd) };
    socket.set_nonblocking(true)?;
    let _ = socket.set_nodelay(true);
    Ok(TcpStream::from_socket(socket))
}

/// parse the peer address out of the filled-in sockaddr_storage.
fn parse_peer_addr(
    storage: &libc::sockaddr_storage,
    len: libc::socklen_t,
) -> io::Result<SocketAddr> {
    // SAFETY: the kernel wrote a valid sockaddr into storage with length len.
    let sock_addr = unsafe { SockAddr::new(*storage, len) };
    sock_addr
        .as_socket()
        .ok_or_else(|| io::Error::other("accepted peer addr is not an IP socket"))
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
    use crate::os::core_shard;
    use proxima_runtime::CoreId;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    #[test]
    fn bind_and_local_addr() {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let listener = TcpListener::bind(addr).expect("bind");
        let local = listener.local_addr().expect("local_addr");
        assert_eq!(local.ip(), addr.ip());
        assert!(local.port() > 0);
    }

    #[test]
    fn accept_connect_echo() {
        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let done = Arc::new(AtomicBool::new(false));
        let done_for_factory = done.clone();
        let addr_chan = Arc::new(std::sync::Mutex::new(None::<SocketAddr>));
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

        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        let bound = loop {
            if let Some(addr) = *addr_chan.lock().unwrap() {
                break addr;
            }
            assert!(std::time::Instant::now() < deadline, "listener never bound");
            std::thread::sleep(Duration::from_millis(5));
        };

        let mut client = std::net::TcpStream::connect(bound).expect("connect");
        use std::io::{Read, Write};
        client.write_all(b"ping").expect("client write");
        let mut buf = [0u8; 4];
        client.read_exact(&mut buf).expect("client read");
        assert_eq!(&buf, b"ping");

        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while !done.load(Ordering::Acquire) {
            assert!(
                std::time::Instant::now() < deadline,
                "future never finished"
            );
            std::thread::sleep(Duration::from_millis(5));
        }

        handle.shutdown_and_join().expect("shutdown");
    }
}
