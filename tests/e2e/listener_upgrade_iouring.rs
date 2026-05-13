//! End-to-end Upgrade coverage for the io_uring listener path.
//! A Pipe installs a `LocalUpgradeHandler` via the thread-local
//! ticket registry; the listener writes a `101 Switching Protocols`
//! response head and hands the !Send `Rc<TcpStream>`-backed socket
//! over. The handler echoes back whatever the client sends.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(all(target_os = "linux", feature = "io-uring"))]

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures::AsyncReadExt as FuturesAsyncReadExt;
use futures::AsyncWriteExt as FuturesAsyncWriteExt;
use futures::channel::oneshot;
use proxima::listeners::HttpListenerSpec;
use proxima::{
    LocalHijackedSocket, LocalUpgradeHandler, PipeHandle, ProximaError, Request, Response,
    into_handle,
};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Pipe that installs a local upgrade handler echoing client bytes
/// back over the hijacked socket, then returns a 101 response.
struct UpgradeEcho;

impl proxima::SendPipe for UpgradeEcho {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let ticket = request.context.local_upgrade_ticket;
        async move {
            let ticket = ticket.ok_or_else(|| {
                ProximaError::Config(
                    "no local_upgrade_ticket on context — only valid on io_uring listener path"
                        .into(),
                )
            })?;
            let handler = LocalUpgradeHandler::new(|socket: LocalHijackedSocket| async move {
                let mut stream = socket.stream;
                // any pipelined bytes go first (CONNECT-style client may
                // have sent tunnel data ahead of our 101).
                if !socket.leftover.is_empty() {
                    FuturesAsyncWriteExt::write_all(&mut stream, &socket.leftover)
                        .await
                        .map_err(ProximaError::Io)?;
                }
                let mut buf = [0_u8; 4096];
                loop {
                    let read = FuturesAsyncReadExt::read(&mut stream, &mut buf)
                        .await
                        .map_err(ProximaError::Io)?;
                    if read == 0 {
                        break;
                    }
                    FuturesAsyncWriteExt::write_all(&mut stream, &buf[..read])
                        .await
                        .map_err(ProximaError::Io)?;
                }
                Ok(())
            });
            proxima::upgrade::local_slots::install(ticket, handler);
            Ok(Response::new(101)
                .with_header("connection", "upgrade")
                .with_header("upgrade", "x-proxima-echo")
                .with_header("content-length", "0")
                .with_body(bytes::Bytes::new()))
        }
    }
}

#[test]
fn iouring_buffered_upgrade_hijacks_and_echoes() {
    tokio_uring::start(async move {
        let port = 28467_u16;
        let bind: SocketAddr = format!("127.0.0.1:{port}").parse().expect("addr");
        let dispatch: PipeHandle = into_handle(UpgradeEcho);
        let spec = Arc::new(HttpListenerSpec {
            max_body_bytes: None,
        });
        let raw_spec = json!({});
        let (shutdown_tx, shutdown_rx) = oneshot::channel();

        let raw_spec_clone = raw_spec.clone();
        let telemetry: proxima::TelemetryHandle = Arc::new(proxima::NoopTelemetry);
        tokio::task::spawn_local(async move {
            if let Err(error) = proxima::listeners::http_uring::serve_uring(
                bind,
                dispatch,
                spec,
                &raw_spec_clone,
                telemetry,
                shutdown_rx,
            )
            .await
            {
                eprintln!("serve_uring error: {error:?}");
            }
        });
        tokio::time::sleep(Duration::from_millis(150)).await;

        let mut stream = tokio::net::TcpStream::connect(bind).await.expect("connect");
        stream.set_nodelay(true).expect("nodelay");
        stream
            .write_all(
                b"GET /tunnel HTTP/1.1\r\nHost: test\r\nConnection: upgrade\r\nUpgrade: x-proxima-echo\r\nContent-Length: 0\r\n\r\n",
            )
            .await
            .expect("write request");
        stream.flush().await.expect("flush");

        // read the 101 head + CRLF CRLF
        let mut head = Vec::new();
        let mut buf = [0_u8; 256];
        loop {
            let read = stream.read(&mut buf).await.expect("read head");
            if read == 0 {
                panic!("peer closed before 101");
            }
            head.extend_from_slice(&buf[..read]);
            if head.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }
        let head_text = String::from_utf8_lossy(&head);
        assert!(
            head_text.starts_with("HTTP/1.1 101"),
            "expected 101 Switching Protocols, got: {head_text}"
        );

        // anything past the \r\n\r\n is post-upgrade bytes (none here)
        let separator = head
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .expect("response separator");
        let post_head = &head[separator + 4..];
        assert!(
            post_head.is_empty(),
            "unexpected bytes past head: {post_head:?}"
        );

        // hijacked socket — server should echo whatever we send
        stream
            .write_all(b"hello upgrade")
            .await
            .expect("write payload");
        stream.flush().await.expect("flush payload");

        let mut echoed = Vec::new();
        let mut expected = b"hello upgrade".len();
        while echoed.len() < expected {
            let read = stream.read(&mut buf).await.expect("read echo");
            if read == 0 {
                panic!("peer closed mid-echo at {} of {expected}", echoed.len());
            }
            echoed.extend_from_slice(&buf[..read]);
        }
        assert_eq!(&echoed[..expected], b"hello upgrade");

        // second roundtrip — proves the socket is genuinely owned by
        // the upgrade handler, not a one-shot.
        stream.write_all(b"more bytes").await.expect("write 2");
        stream.flush().await.expect("flush 2");
        expected = b"more bytes".len();
        let mut second = Vec::new();
        while second.len() < expected {
            let read = stream.read(&mut buf).await.expect("read echo 2");
            if read == 0 {
                panic!("peer closed mid second echo");
            }
            second.extend_from_slice(&buf[..read]);
        }
        assert_eq!(&second[..expected], b"more bytes");

        // close from the client side — handler should observe EOF +
        // exit cleanly.
        drop(stream);
        let _ = shutdown_tx.send(());
    });
}
