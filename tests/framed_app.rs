//! Proves `App` serves `FramedListenProtocol` end-to-end inside a plain
//! `#[proxima::test]` — the pattern a downstream consumer's framed serve copies. Registers
//! the protocol via the builder, mounts an echo pipe, runs on an
//! ephemeral port, and round-trips length-prefixed frames.

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(feature = "tcp")]

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use proxima::listeners::FramedListenProtocol;
use proxima::pipe::{into_handle};
use proxima::request::{Request, Response};
use proxima::{App, HeaderList, MountTarget, ProximaError, RunConfig, Spec, TokioPerCoreRuntime};
use proxima_primitives::pipe::SendPipe;

struct UppercasePipe;

impl SendPipe for UppercasePipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            let (_, bytes) = request.body_bytes().await?;
            let upper: Vec<u8> = bytes.iter().map(u8::to_ascii_uppercase).collect();
            Ok(Response {
                status: 200,
                metadata: HeaderList::new(),
                payload: Bytes::from(upper),
                stream: None,
                upgrade: None,
            })
        }
    }
}


fn find_free_port() -> SocketAddr {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
    listener.local_addr().expect("local addr")
}

async fn send_frame(stream: &mut tokio::net::TcpStream, payload: &[u8]) {
    let len = u32::try_from(payload.len()).unwrap();
    stream.write_all(&len.to_be_bytes()).await.unwrap();
    stream.write_all(payload).await.unwrap();
    stream.flush().await.unwrap();
}

async fn recv_frame(stream: &mut tokio::net::TcpStream) -> Vec<u8> {
    let mut len_buf = [0_u8; 4];
    stream.read_exact(&mut len_buf).await.unwrap();
    let len = u32::from_be_bytes(len_buf) as usize;
    let mut payload = vec![0_u8; len];
    stream.read_exact(&mut payload).await.unwrap();
    payload
}

#[proxima::test]
async fn app_serves_framed_listener_round_trip() {
    let addr = find_free_port();

    let mut app = App::builder()
        .with_listen_protocol(Arc::new(FramedListenProtocol::new("framed")))
        .expect("register framed protocol")
        .build()
        .expect("build app")
        .with_runtime(Arc::new(
            TokioPerCoreRuntime::new(2).expect("per-core runtime"),
        ));

    app.pipe("echo", Spec::Handle(into_handle(UppercasePipe)))
        .await
        .expect("register echo pipe");
    app.mount("/{*path}", MountTarget::Named("echo".into()))
        .expect("mount echo");

    let _shutdown = app
        .run_until_signal(RunConfig {
            bind: addr,
            protocol: "framed".into(),
            spec: json!({ "reject_zero_len": true, "max_frame_bytes": 16 * 1024 * 1024 }),
        })
        .await
        .expect("serve framed");

    // guard the whole client interaction so a wiring hang fails fast.
    let outcome = tokio::time::timeout(Duration::from_secs(5), async move {
        let mut stream = tokio::net::TcpStream::connect(addr)
            .await
            .expect("listener must already be accepting when run_until_signal returns");
        send_frame(&mut stream, b"hello framed app").await;
        assert_eq!(recv_frame(&mut stream).await, b"HELLO FRAMED APP");
        send_frame(&mut stream, b"second").await;
        assert_eq!(recv_frame(&mut stream).await, b"SECOND");
    })
    .await;

    outcome.expect("framed app round-trip timed out (App serve wiring hang)");
}
