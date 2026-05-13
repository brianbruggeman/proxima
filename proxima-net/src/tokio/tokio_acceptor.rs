//! Tokio backing for the runtime-agnostic `AcceptorFactory`/`TcpAcceptor`
//! surface. Builds the listen socket through `socket2` so bind options
//! (backlog, SO_REUSEPORT, TCP Fast Open) are honored, then hands the
//! pre-bound std listener to tokio via `from_std`. `bind` must be called
//! from within a future running on a tokio worker so the reactor is live.

use std::io;
use std::net::SocketAddr;
use std::task::{Context, Poll};

use socket2::{Domain, Protocol, Socket, Type};
use tokio::net::TcpListener as TokioTcpListenerInner;

use proxima_primitives::stream::{AcceptorFactory, StreamConnection, TcpAcceptor, TcpBindOptions};

use super::tokio_stream_listener::TokioTcpConnection;

/// Builds tokio-backed acceptors. Unit struct — holds no state, shared via
/// `Arc<dyn AcceptorFactory>` in the serve path.
pub struct TokioAcceptorFactory;

impl AcceptorFactory for TokioAcceptorFactory {
    fn bind(&self, addr: SocketAddr, options: TcpBindOptions) -> io::Result<Box<dyn TcpAcceptor>> {
        let domain = if addr.is_ipv4() {
            Domain::IPV4
        } else {
            Domain::IPV6
        };
        let socket = Socket::new(domain, Type::STREAM, Some(Protocol::TCP))?;
        socket.set_nonblocking(true)?;
        socket.set_reuse_address(true)?;
        #[cfg(unix)]
        if options.reuseport {
            socket.set_reuse_port(true)?;
        }
        socket.bind(&addr.into())?;
        if let Some(queue) = options.tcp_fastopen {
            apply_tcp_fastopen(&socket, queue)?;
        }
        socket.listen(options.backlog as i32)?;

        let std_listener: std::net::TcpListener = socket.into();
        std_listener.set_nonblocking(true)?;
        let listener = TokioTcpListenerInner::from_std(std_listener)?;

        Ok(Box::new(TokioAcceptor { listener }))
    }
}

/// A tokio `TcpListener` exposed through the worker-pinned `TcpAcceptor`
/// surface. Each accepted stream is reshaped into the futures-io
/// `TokioTcpConnection` so consumers stay runtime-agnostic.
pub struct TokioAcceptor {
    listener: TokioTcpListenerInner,
}

impl TcpAcceptor for TokioAcceptor {
    fn poll_accept(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<Box<dyn StreamConnection>>> {
        match self.listener.poll_accept(cx) {
            Poll::Ready(Ok((stream, _peer))) => {
                let _ = stream.set_nodelay(true);
                let conn = TokioTcpConnection::from_tokio(stream);
                Poll::Ready(Ok(Box::new(conn) as Box<dyn StreamConnection>))
            }
            Poll::Ready(Err(err)) => Poll::Ready(Err(err)),
            Poll::Pending => Poll::Pending,
        }
    }

    fn local_addr(&self) -> io::Result<SocketAddr> {
        self.listener.local_addr()
    }
}

#[cfg(target_os = "linux")]
fn apply_tcp_fastopen(socket: &Socket, queue: u32) -> io::Result<()> {
    use std::os::unix::io::AsRawFd;
    // linux IPPROTO_TCP/TCP_FASTOPEN are stable since 3.7; inlining the
    // setsockopt avoids pulling libc just for one optname.
    const IPPROTO_TCP: i32 = 6;
    const TCP_FASTOPEN: i32 = 23;
    let value = queue as i32;
    unsafe extern "C" {
        fn setsockopt(
            sockfd: i32,
            level: i32,
            optname: i32,
            optval: *const core::ffi::c_void,
            optlen: u32,
        ) -> i32;
    }
    let ret = unsafe {
        setsockopt(
            socket.as_raw_fd(),
            IPPROTO_TCP,
            TCP_FASTOPEN,
            std::ptr::from_ref(&value).cast::<core::ffi::c_void>(),
            std::mem::size_of::<i32>() as u32,
        )
    };
    if ret != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn apply_tcp_fastopen(_socket: &Socket, _queue: u32) -> io::Result<()> {
    // non-linux platforms use different optname/semantics, and this crate
    // carries no logging dep; accept the request as a no-op so operators
    // can target linux and dev elsewhere without conditional config.
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::io::{AsyncReadExt, AsyncWriteExt};
    use std::future::poll_fn;
    use tokio::io::{AsyncReadExt as TokioAsyncReadExt, AsyncWriteExt as TokioAsyncWriteExt};

    #[proxima::test]
    async fn acceptor_round_trips_four_bytes() {
        let factory = TokioAcceptorFactory;
        let mut acceptor = factory
            .bind(
                "127.0.0.1:0".parse().expect("parse bind addr"),
                TcpBindOptions::default(),
            )
            .expect("bind acceptor");
        let local = acceptor.local_addr().expect("local addr");

        let mut client = tokio::net::TcpStream::connect(local)
            .await
            .expect("client connect");

        let mut server_conn = poll_fn(|cx| acceptor.poll_accept(cx))
            .await
            .expect("server accept");

        client.write_all(b"ping").await.expect("client write");
        client.flush().await.expect("client flush");

        let mut received = [0_u8; 4];
        server_conn
            .read_exact(&mut received)
            .await
            .expect("server read");
        assert_eq!(&received, b"ping");

        server_conn.write_all(b"pong").await.expect("server write");
        server_conn.flush().await.expect("server flush");

        let mut response = [0_u8; 4];
        client.read_exact(&mut response).await.expect("client read");
        assert_eq!(&response, b"pong");
    }
}
