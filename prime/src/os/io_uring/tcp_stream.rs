//! io_uring-backed TCP stream for the prime runtime.
//!
//! ## buffer ownership
//!
//! `futures::io::AsyncRead::poll_read` receives a caller-provided `&mut [u8]`
//! that is only valid for the synchronous duration of the call. io_uring
//! operations borrow memory asynchronously (the kernel writes into the buffer
//! after the call returns). these two ownership models are incompatible, so
//! each TcpStream owns a fixed-size heap buffer per direction:
//!
//! - `read_buf`: the kernel reads into this. on completion, bytes are copied
//!   into the caller's buffer slice. 16 KiB covers typical h1/h2 frames.
//! - `write_buf`: caller bytes are copied into this, then the kernel drains
//!   it. on completion, bytes are considered sent.
//!
//! this is one extra copy per I/O per direction — the correctness floor cost.
//! perf follow-on: fixed buffers (IORING_OP_REGISTER_BUFFERS) eliminate the
//! copy and the page-pin overhead.

#![cfg(all(
    target_os = "linux",
    feature = "io-uring",
    feature = "runtime-prime-reactor"
))]

use std::io;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::io::{AsyncRead, AsyncWrite};
use io_uring::{opcode, types};
use socket2::Socket;
use std::os::fd::AsRawFd;

use super::reactor::{BUF_SIZE, ParkedResource, with_current_uring};

/// per-direction operation state for TcpStream.
enum OpState {
    /// no operation submitted.
    Idle,
    /// SQE submitted; waiting for CQE. u64 is the packed reactor user_data.
    InFlight(u64),
}

/// io_uring-backed TCP stream. implements `futures::io::AsyncRead + AsyncWrite`.
///
/// `Send` is deliberately restored — see `TcpListener` module doc for the
/// contract.
pub struct TcpStream {
    socket: Socket,
    read_buf: Box<[u8; BUF_SIZE]>,
    write_buf: Box<[u8; BUF_SIZE]>,
    read_state: OpState,
    write_state: OpState,
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

// SAFETY: socket2::Socket is the only !Send-via-raw-fd field; same contract
// as net::TcpStream. prime per-core runtime never migrates tasks.
unsafe impl Send for TcpStream {}

impl TcpStream {
    pub(super) fn from_socket(socket: Socket) -> Self {
        Self {
            socket,
            read_buf: Box::new([0u8; BUF_SIZE]),
            write_buf: Box::new([0u8; BUF_SIZE]),
            read_state: OpState::Idle,
            write_state: OpState::Idle,
            _not_sync: PhantomData,
        }
    }
}

/// cancel `state` if it is in flight, handing `buffer` to the reactor so the
/// kernel's still-pending write into it cannot race a `Drop`-triggered free.
/// the caller's buffer field is left holding a fresh, empty replacement —
/// the struct is being torn down regardless, so the replacement is dropped
/// immediately after this returns.
fn cancel_in_flight(state: &OpState, buffer: &mut Box<[u8; BUF_SIZE]>) {
    let OpState::InFlight(user_data) = state else {
        return;
    };
    let user_data = *user_data;
    let parked = std::mem::replace(buffer, Box::new([0u8; BUF_SIZE]));
    let _ = with_current_uring(|reactor| {
        reactor.cancel_op(user_data, ParkedResource::StreamBuffer(parked));
        Ok(())
    });
}

impl Drop for TcpStream {
    fn drop(&mut self) {
        cancel_in_flight(&self.read_state, &mut self.read_buf);
        cancel_in_flight(&self.write_state, &mut self.write_buf);
    }
}

impl AsyncRead for TcpStream {
    fn poll_read(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();

        if let Err(err) = with_current_uring(|reactor| reactor.drain_cqes()) {
            return Poll::Ready(Err(err));
        }

        if let OpState::InFlight(user_data) = &this.read_state {
            let user_data = *user_data;
            let maybe_result = with_current_uring(|reactor| Ok(reactor.take_result(user_data)));
            match maybe_result {
                Err(err) => return Poll::Ready(Err(err)),
                Ok(Some(result)) => {
                    this.read_state = OpState::Idle;
                    if result < 0 {
                        return Poll::Ready(Err(io::Error::from_raw_os_error(-result)));
                    }
                    let bytes = result as usize;
                    let copy_len = bytes.min(buf.len());
                    buf[..copy_len].copy_from_slice(&this.read_buf[..copy_len]);
                    return Poll::Ready(Ok(copy_len));
                }
                Ok(None) => {
                    let _ = with_current_uring(|reactor| {
                        reactor.set_waker(user_data, context.waker().clone());
                        Ok(())
                    });
                    return Poll::Pending;
                }
            }
        }

        // Idle — submit a new Recv SQE.
        let read_buf_ptr = this.read_buf.as_mut_ptr();
        let read_len = BUF_SIZE.min(buf.len().max(1));
        let fd = this.socket.as_raw_fd();

        let submit_result = with_current_uring(|reactor| {
            let user_data = reactor.register_op();
            let sqe = opcode::Recv::new(types::Fd(fd), read_buf_ptr, read_len as u32)
                .build()
                .user_data(user_data);
            unsafe {
                reactor
                    .ring_mut()
                    .submission()
                    .push(&sqe)
                    .map_err(|_| io::Error::other("io_uring SQ full on read"))?;
            }
            reactor.set_waker(user_data, context.waker().clone());
            Ok(user_data)
        });

        match submit_result {
            Err(err) => Poll::Ready(Err(err)),
            Ok(user_data) => {
                this.read_state = OpState::InFlight(user_data);
                Poll::Pending
            }
        }
    }
}

impl AsyncWrite for TcpStream {
    fn poll_write(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();

        if let Err(err) = with_current_uring(|reactor| reactor.drain_cqes()) {
            return Poll::Ready(Err(err));
        }

        if let OpState::InFlight(user_data) = &this.write_state {
            let user_data = *user_data;
            let maybe_result = with_current_uring(|reactor| Ok(reactor.take_result(user_data)));
            match maybe_result {
                Err(err) => return Poll::Ready(Err(err)),
                Ok(Some(result)) => {
                    this.write_state = OpState::Idle;
                    if result < 0 {
                        return Poll::Ready(Err(io::Error::from_raw_os_error(-result)));
                    }
                    return Poll::Ready(Ok(result as usize));
                }
                Ok(None) => {
                    let _ = with_current_uring(|reactor| {
                        reactor.set_waker(user_data, context.waker().clone());
                        Ok(())
                    });
                    return Poll::Pending;
                }
            }
        }

        // Idle — copy caller bytes into owned write buffer and submit Send SQE.
        let copy_len = buf.len().min(BUF_SIZE);
        this.write_buf[..copy_len].copy_from_slice(&buf[..copy_len]);
        let write_buf_ptr = this.write_buf.as_ptr();
        let fd = this.socket.as_raw_fd();

        let submit_result = with_current_uring(|reactor| {
            let user_data = reactor.register_op();
            let sqe = opcode::Send::new(types::Fd(fd), write_buf_ptr, copy_len as u32)
                .build()
                .user_data(user_data);
            unsafe {
                reactor
                    .ring_mut()
                    .submission()
                    .push(&sqe)
                    .map_err(|_| io::Error::other("io_uring SQ full on write"))?;
            }
            reactor.set_waker(user_data, context.waker().clone());
            Ok(user_data)
        });

        match submit_result {
            Err(err) => Poll::Ready(Err(err)),
            Ok(user_data) => {
                this.write_state = OpState::InFlight(user_data);
                Poll::Pending
            }
        }
    }

    fn poll_flush(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let _ = this.socket.shutdown(std::net::Shutdown::Both);
        Poll::Ready(Ok(()))
    }
}

#[cfg(test)]
#[cfg(all(
    target_os = "linux",
    feature = "io-uring",
    feature = "runtime-prime-reactor"
))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::super::tcp_listener::TcpListener;
    use super::*;
    use crate::os::core_shard;
    use proxima_runtime::CoreId;
    use std::net::SocketAddr;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    fn launch_echo(
        handle: &core_shard::CoreShardHandle,
        addr_chan: Arc<std::sync::Mutex<Option<SocketAddr>>>,
        done: Arc<AtomicBool>,
        msg_count: usize,
    ) {
        handle
            .dispatch_factory(Box::new(move || {
                Box::pin(async move {
                    use futures::io::{AsyncReadExt, AsyncWriteExt};
                    let mut listener =
                        TcpListener::bind("127.0.0.1:0".parse().unwrap()).expect("bind");
                    let bound = listener.local_addr().expect("local_addr");
                    *addr_chan.lock().unwrap() = Some(bound);
                    let (mut stream, _peer) = listener.accept().await.expect("accept");
                    for _ in 0..msg_count {
                        let mut buf = [0u8; 4];
                        stream.read_exact(&mut buf).await.expect("read");
                        stream.write_all(&buf).await.expect("write");
                    }
                    done.store(true, Ordering::Release);
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch");
    }

    fn wait_addr(
        addr_chan: &Arc<std::sync::Mutex<Option<SocketAddr>>>,
        deadline: std::time::Instant,
    ) -> SocketAddr {
        loop {
            if let Some(addr) = *addr_chan.lock().unwrap() {
                return addr;
            }
            assert!(std::time::Instant::now() < deadline, "listener never bound");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn read_write_echo() {
        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let done = Arc::new(AtomicBool::new(false));
        let addr_chan = Arc::new(std::sync::Mutex::new(None::<SocketAddr>));

        launch_echo(&handle, addr_chan.clone(), done.clone(), 1);

        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        let bound = wait_addr(&addr_chan, deadline);

        let mut client = std::net::TcpStream::connect(bound).expect("connect");
        use std::io::{Read, Write};
        client.write_all(b"pong").expect("write");
        let mut buf = [0u8; 4];
        client.read_exact(&mut buf).expect("read");
        assert_eq!(&buf, b"pong");

        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while !done.load(Ordering::Acquire) {
            assert!(std::time::Instant::now() < deadline, "echo never finished");
            std::thread::sleep(Duration::from_millis(5));
        }
        handle.shutdown_and_join().expect("shutdown");
    }

    #[test]
    fn shutdown_closes_peer() {
        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let done = Arc::new(AtomicBool::new(false));
        let addr_chan = Arc::new(std::sync::Mutex::new(None::<SocketAddr>));
        let done_clone = done.clone();
        let addr_clone = addr_chan.clone();

        handle
            .dispatch_factory(Box::new(move || {
                Box::pin(async move {
                    use futures::io::AsyncWriteExt;
                    let mut listener =
                        TcpListener::bind("127.0.0.1:0".parse().unwrap()).expect("bind");
                    let bound = listener.local_addr().expect("local_addr");
                    *addr_clone.lock().unwrap() = Some(bound);
                    let (mut stream, _peer) = listener.accept().await.expect("accept");
                    stream.write_all(b"bye!").await.expect("write");
                    stream.close().await.expect("close");
                    done_clone.store(true, Ordering::Release);
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch");

        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        let bound = wait_addr(&addr_chan, deadline);

        let mut client = std::net::TcpStream::connect(bound).expect("connect");
        use std::io::Read;
        let mut received = Vec::new();
        client.read_to_end(&mut received).expect("read to end");
        assert_eq!(&received, b"bye!");

        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while !done.load(Ordering::Acquire) {
            assert!(
                std::time::Instant::now() < deadline,
                "server never finished"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        handle.shutdown_and_join().expect("shutdown");
    }
}
