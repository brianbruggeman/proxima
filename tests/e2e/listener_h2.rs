//! End-to-end test for the native HTTP/2 listener (no h2 crate on
//! the server side). Client uses the `h2` crate to drive prior-
//! knowledge HTTP/2 over plain TCP (no TLS — both sides agree on
//! h2 out-of-band).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(feature = "http2")]

use std::future::Future;

use bytes::Bytes;
use proxima::ResponseStream;
use proxima::error::ProximaError;
use proxima::h2::serve_h2_connection;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::request::{Request, Response};
use proxima_primitives::pipe::SendPipe;
use tokio::net::{TcpListener, TcpStream};
use tokio_util::compat::TokioAsyncReadCompatExt;

struct ConstantOk;

impl SendPipe for ConstantOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { Ok(Response::ok(Bytes::from_static(b"ok"))) }
    }
}


/// Pipe that returns 100 large response headers — forces the
/// encoded HPACK block to exceed `peer_max_frame_size` (16,384) so
/// HEADERS must split into HEADERS + N CONTINUATION frames.
struct HugeHeadersPipe;

impl SendPipe for HugeHeadersPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            let mut response = Response::ok(Bytes::from_static(b"ok"));
            // Each header is ~250 bytes encoded (literal name + literal
            // value, no static-table hit). 100 of them ≈ 25 KiB > 16 KiB
            // max frame, forcing at least one CONTINUATION.
            for index in 0..100 {
                let name = format!("x-bench-header-{index:03}");
                let value = "X".repeat(200);
                let _ = response.metadata.insert(name, value);
            }
            Ok(response)
        }
    }
}


/// Pipe that returns a streaming response body — 8 separate
/// chunks of "chunk-N\n". Exercises the chunk-pull pump in the
/// native h2 server: handler returns before the body is consumed,
/// the pump pulls each chunk and emits as DATA frames.
struct StreamingChunksPipe;

impl SendPipe for StreamingChunksPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            let chunks: Vec<Result<Bytes, ProximaError>> = (0..8)
                .map(|index| Ok(Bytes::from(format!("chunk-{index}\n"))))
                .collect();
            let stream = futures::stream::iter(chunks);
            Ok(Response::streamed(ResponseStream::new(stream)))
        }
    }
}


/// Pipe that streams 256 1-KiB chunks (256 KiB total), each
/// filled with a unique byte so order corruption is visible. Total
/// vastly exceeds the 65,535-byte initial send window — many chunks
/// must queue while the send window is exhausted, then drain in
/// order on `WindowGranted`. Earlier rev had a single-slot
/// `pending_sends` HashMap that clobbered chunk N with chunk N+1;
/// this regression test would have failed against that rev.
struct ManyChunkStreamingPipe;

const MANY_CHUNK_COUNT: usize = 256;
const MANY_CHUNK_SIZE: usize = 1024;

impl SendPipe for ManyChunkStreamingPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            let chunks: Vec<Result<Bytes, ProximaError>> = (0..MANY_CHUNK_COUNT)
                .map(|index| {
                    let byte = (index % 256) as u8;
                    Ok(Bytes::from(vec![byte; MANY_CHUNK_SIZE]))
                })
                .collect();
            let stream = futures::stream::iter(chunks);
            Ok(Response::streamed(ResponseStream::new(stream)))
        }
    }
}


struct EchoBodyPipe;

impl SendPipe for EchoBodyPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            let (_request, bytes) = request.body_bytes().await?;
            Ok(Response::ok(bytes))
        }
    }
}


async fn spawn_native_server(dispatch: PipeHandle) -> std::net::SocketAddr {
    spawn_native_server_with_admission(dispatch, proxima_listen::admission::ConnAdmission::unbounded())
        .await
}

/// Sibling of [`spawn_native_server`] that takes a caller-supplied
/// [`proxima_listen::admission::ConnAdmission`] instead of an unbounded
/// one, so a test can exercise a real `max_in_flight_requests` cap.
async fn spawn_native_server_with_admission(
    dispatch: PipeHandle,
    admission: proxima_listen::admission::ConnAdmission,
) -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        let (socket, peer) = listener.accept().await.expect("accept tcp");
        let _ = serve_h2_connection(
            socket.compat(),
            dispatch,
            admission,
            Some(proxima::stream::PeerInfo::Tcp(peer)),
        )
        .await;
    });
    addr
}

/// The handler behind the admission-shed tests below: signals
/// `entered_tx` as soon as it is dispatched (proving `ConnAdmission`
/// genuinely admitted it — the ONLY way to observe this signal is past
/// `request_admit()` returning `Admit`), then blocks on `release_rx`
/// until the test says to proceed. Holds the listener's one capacity
/// slot open long enough for a concurrent second stream to be shed
/// against a real held slot, never a race against an empty cap.
struct SlowGatePipe {
    entered_tx: std::sync::Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    release_rx: std::sync::Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
}

impl SendPipe for SlowGatePipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let entered_tx = self.entered_tx.lock().expect("lock").take();
        let release_rx = self.release_rx.lock().expect("lock").take();
        async move {
            if let Some(entered_tx) = entered_tx {
                let _ = entered_tx.send(());
            }
            if let Some(release_rx) = release_rx {
                let _ = release_rx.await;
            }
            Ok(Response::new(200).with_body(Bytes::from_static(b"slow-ok")))
        }
    }
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_h2_listener_round_trip_constant_response_body() {
    let dispatch: PipeHandle = into_handle(ConstantOk);
    let addr = spawn_native_server(dispatch).await;
    let tcp = TcpStream::connect(addr).await.expect("connect");
    let (mut h2_client, h2_conn) = h2::client::handshake(tcp).await.expect("handshake");
    let conn_task = tokio::spawn(async move {
        let _ = h2_conn.await;
    });

    let request = http::Request::builder()
        .method("GET")
        .uri("http://localhost/ping")
        .body(())
        .expect("request");
    let (response_future, _) = h2_client.send_request(request, true).expect("send_request");
    let response = response_future.await.expect("response");
    assert_eq!(response.status(), 200);
    let mut body = response.into_body();
    let mut collected = Vec::new();
    while let Some(chunk) = body.data().await {
        let chunk = chunk.expect("chunk");
        let len = chunk.len();
        body.flow_control()
            .release_capacity(len)
            .expect("flow control");
        collected.extend_from_slice(&chunk);
    }
    assert_eq!(&collected[..], b"ok");

    drop(h2_client);
    drop(conn_task);
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_h2_listener_round_trip_initial_window_plus_one_echo() {
    // 65535 = default initial flow-control window. one byte over forces
    // the client to pause until we issue a WINDOW_UPDATE. exposes the
    // auto-replenishment requirement in the native server.
    let dispatch: PipeHandle = into_handle(EchoBodyPipe);
    let addr = spawn_native_server(dispatch).await;
    let tcp = TcpStream::connect(addr).await.expect("connect");
    let (mut h2_client, h2_conn) = h2::client::handshake(tcp).await.expect("handshake");
    let conn_task = tokio::spawn(async move {
        let _ = h2_conn.await;
    });

    let payload = Bytes::from(vec![b'q'; 65_536]);
    let request = http::Request::builder()
        .method("POST")
        .uri("http://localhost/echo")
        .body(())
        .expect("request");
    let (response_future, mut send_stream) = h2_client
        .send_request(request, false)
        .expect("send_request");
    send_stream
        .send_data(payload.clone(), true)
        .expect("send body");
    let response = response_future.await.expect("response");
    assert_eq!(response.status(), 200);
    let mut body = response.into_body();
    let mut collected = Vec::new();
    while let Some(chunk) = body.data().await {
        let chunk = chunk.expect("chunk");
        let len = chunk.len();
        body.flow_control()
            .release_capacity(len)
            .expect("flow control");
        collected.extend_from_slice(&chunk);
    }
    assert_eq!(collected.len(), 65_536);
    assert_eq!(&collected[..], &payload[..]);

    drop(h2_client);
    drop(conn_task);
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_h2_listener_huge_headers_forces_continuation() {
    let dispatch: PipeHandle = into_handle(HugeHeadersPipe);
    let addr = spawn_native_server(dispatch).await;
    let tcp = TcpStream::connect(addr).await.expect("connect");
    let (mut h2_client, h2_conn) = h2::client::handshake(tcp).await.expect("handshake");
    let conn_task = tokio::spawn(async move {
        let _ = h2_conn.await;
    });

    let request = http::Request::builder()
        .method("GET")
        .uri("http://localhost/huge")
        .body(())
        .expect("request");
    let (response_future, _) = h2_client.send_request(request, true).expect("send_request");
    let response = response_future.await.expect("response");
    assert_eq!(response.status(), 200);
    // 100 large response headers should round-trip end-to-end —
    // the server's CONTINUATION-emit path is what makes that work.
    let response_headers = response.headers();
    let mut count = 0;
    for (name, _value) in response_headers.iter() {
        if name.as_str().starts_with("x-bench-header-") {
            count += 1;
        }
    }
    assert_eq!(count, 100, "all 100 large headers must round-trip");

    let mut body = response.into_body();
    while let Some(chunk) = body.data().await {
        let chunk = chunk.expect("chunk");
        let len = chunk.len();
        body.flow_control()
            .release_capacity(len)
            .expect("flow control");
    }

    drop(h2_client);
    drop(conn_task);
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_h2_listener_streaming_response_body() {
    let dispatch: PipeHandle = into_handle(StreamingChunksPipe);
    let addr = spawn_native_server(dispatch).await;
    let tcp = TcpStream::connect(addr).await.expect("connect");
    let (mut h2_client, h2_conn) = h2::client::handshake(tcp).await.expect("handshake");
    let conn_task = tokio::spawn(async move {
        let _ = h2_conn.await;
    });

    let request = http::Request::builder()
        .method("GET")
        .uri("http://localhost/stream")
        .body(())
        .expect("request");
    let (response_future, _) = h2_client.send_request(request, true).expect("send_request");
    let response = response_future.await.expect("response");
    assert_eq!(response.status(), 200);
    let mut body = response.into_body();
    let mut collected = Vec::new();
    while let Some(chunk) = body.data().await {
        let chunk = chunk.expect("chunk");
        let len = chunk.len();
        body.flow_control()
            .release_capacity(len)
            .expect("flow control");
        collected.extend_from_slice(&chunk);
    }
    // Concatenated payload from 8 chunks: "chunk-0\nchunk-1\n...chunk-7\n".
    let expected: String = (0..8).map(|index| format!("chunk-{index}\n")).collect();
    assert_eq!(collected, expected.as_bytes());

    drop(h2_client);
    drop(conn_task);
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_h2_listener_load_1000_small_get() {
    // Sustained-traffic regression: hammer the connection with many
    // sequential requests to catch leaks / window-accounting drift /
    // resource exhaustion that single-request tests miss.
    let dispatch: PipeHandle = into_handle(ConstantOk);
    let addr = spawn_native_server(dispatch).await;
    let tcp = TcpStream::connect(addr).await.expect("connect");
    let (mut h2_client, h2_conn) = h2::client::handshake(tcp).await.expect("handshake");
    let conn_task = tokio::spawn(async move {
        let _ = h2_conn.await;
    });

    for index in 0..1000 {
        let request = http::Request::builder()
            .method("GET")
            .uri("http://localhost/")
            .body(())
            .expect("request");
        let (response_future, _) = h2_client.send_request(request, true).expect("send_request");
        let response = response_future.await.expect("response");
        assert_eq!(response.status(), 200, "iteration {index}");
        let mut body = response.into_body();
        while let Some(chunk) = body.data().await {
            let chunk = chunk.expect("chunk");
            let len = chunk.len();
            body.flow_control()
                .release_capacity(len)
                .expect("flow control");
            std::hint::black_box(chunk);
        }
    }

    drop(h2_client);
    drop(conn_task);
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_h2_listener_load_500_32kib_echo() {
    // POST echo with 32 KiB body, repeated. exercises auto-WINDOW_UPDATE
    // under sustained flow-control pressure.
    let dispatch: PipeHandle = into_handle(EchoBodyPipe);
    let addr = spawn_native_server(dispatch).await;
    let tcp = TcpStream::connect(addr).await.expect("connect");
    let (mut h2_client, h2_conn) = h2::client::handshake(tcp).await.expect("handshake");
    let conn_task = tokio::spawn(async move {
        let _ = h2_conn.await;
    });

    let payload = Bytes::from(vec![b'p'; 32 * 1024]);
    // 70k+ proven locally; cap at 1k for CI runtime sanity.
    for index in 0..1_000 {
        let request = http::Request::builder()
            .method("POST")
            .uri("http://localhost/echo")
            .body(())
            .expect("request");
        let (response_future, mut send_stream) = h2_client
            .send_request(request, false)
            .expect("send_request");
        send_stream
            .send_data(payload.clone(), true)
            .expect("send body");
        let response = response_future
            .await
            .unwrap_or_else(|err| panic!("iteration {index} response error: {err:?}"));
        assert_eq!(response.status(), 200, "iteration {index}");
        let mut body = response.into_body();
        let mut collected = 0;
        while let Some(chunk) = body.data().await {
            let chunk = chunk.expect("chunk");
            let len = chunk.len();
            body.flow_control()
                .release_capacity(len)
                .expect("flow control");
            collected += len;
        }
        assert_eq!(collected, 32 * 1024, "iteration {index}");
    }

    drop(h2_client);
    drop(conn_task);
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_h2_listener_round_trip_32kib_echo() {
    let dispatch: PipeHandle = into_handle(EchoBodyPipe);
    let addr = spawn_native_server(dispatch).await;
    let tcp = TcpStream::connect(addr).await.expect("connect");
    let (mut h2_client, h2_conn) = h2::client::handshake(tcp).await.expect("handshake");
    let conn_task = tokio::spawn(async move {
        let _ = h2_conn.await;
    });

    let payload = Bytes::from(vec![b'p'; 32 * 1024]);
    let request = http::Request::builder()
        .method("POST")
        .uri("http://localhost/echo")
        .body(())
        .expect("request");
    let (response_future, mut send_stream) = h2_client
        .send_request(request, false)
        .expect("send_request");
    send_stream
        .send_data(payload.clone(), true)
        .expect("send body");
    let response = response_future.await.expect("response");
    assert_eq!(response.status(), 200);
    let mut body = response.into_body();
    let mut collected = Vec::new();
    while let Some(chunk) = body.data().await {
        let chunk = chunk.expect("chunk");
        let len = chunk.len();
        body.flow_control()
            .release_capacity(len)
            .expect("flow control");
        collected.extend_from_slice(&chunk);
    }
    assert_eq!(collected.len(), 32 * 1024);
    assert_eq!(&collected[..], &payload[..]);

    drop(h2_client);
    drop(conn_task);
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_h2_listener_streaming_response_queue_preserves_order() {
    // 256 unique-byte chunks of 1 KiB each = 256 KiB stream, far
    // beyond the 65,535-byte initial send window. Forces many
    // chunks to queue in `pending_sends` while the window is
    // exhausted; they must drain in order on WindowGranted. Earlier
    // single-slot map clobbered chunk N with chunk N+1, both losing
    // data and corrupting order — this test catches both.
    let dispatch: PipeHandle = into_handle(ManyChunkStreamingPipe);
    let addr = spawn_native_server(dispatch).await;
    let tcp = TcpStream::connect(addr).await.expect("connect");
    let (mut h2_client, h2_conn) = h2::client::handshake(tcp).await.expect("handshake");
    let conn_task = tokio::spawn(async move {
        let _ = h2_conn.await;
    });

    let request = http::Request::builder()
        .method("GET")
        .uri("http://localhost/stream")
        .body(())
        .expect("request");
    let (response_future, _) = h2_client.send_request(request, true).expect("send_request");
    let response = response_future.await.expect("response");
    assert_eq!(response.status(), 200);
    let mut body = response.into_body();
    let mut collected = Vec::with_capacity(MANY_CHUNK_COUNT * MANY_CHUNK_SIZE);
    while let Some(chunk) = body.data().await {
        let chunk = chunk.expect("chunk");
        let len = chunk.len();
        body.flow_control()
            .release_capacity(len)
            .expect("flow control");
        collected.extend_from_slice(&chunk);
    }
    let expected_total = MANY_CHUNK_COUNT * MANY_CHUNK_SIZE;
    assert_eq!(collected.len(), expected_total, "total bytes must match");
    for chunk_index in 0..MANY_CHUNK_COUNT {
        let byte = (chunk_index % 256) as u8;
        let start = chunk_index * MANY_CHUNK_SIZE;
        let end = start + MANY_CHUNK_SIZE;
        let slice = &collected[start..end];
        assert!(
            slice.iter().all(|value| *value == byte),
            "chunk {chunk_index} out of order or corrupted",
        );
    }

    drop(h2_client);
    drop(conn_task);
}

/// Mirrors what `predicate.and_then(inner)` produces at the pipe edge:
/// `/reject` returns the exact `ProximaError::Forbidden` a filter's
/// `RejectMode::Drop` emits, `/internal` returns a genuinely internal
/// failure, everything else is admitted. One pipe drives every
/// filter-over-h2 regression scenario below.
struct FilterRoutedPipe;

impl SendPipe for FilterRoutedPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            match request.path.as_ref() {
                b"/reject" => Err(ProximaError::Forbidden("blocked by filter".into())),
                b"/internal" => Err(ProximaError::Upstream("boom".into())),
                _ => Ok(Response::ok(Bytes::from_static(b"ok"))),
            }
        }
    }
}


async fn collect_body(response: h2::client::ResponseFuture) -> (http::StatusCode, Vec<u8>) {
    let response = response.await.expect("response");
    let status = response.status();
    let mut body = response.into_body();
    let mut collected = Vec::new();
    while let Some(chunk) = body.data().await {
        let chunk = chunk.expect("chunk");
        let len = chunk.len();
        body.flow_control()
            .release_capacity(len)
            .expect("flow control");
        collected.extend_from_slice(&chunk);
    }
    (status, collected)
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_h2_listener_filter_rejection_renders_403_and_connection_survives() {
    // Regression for the h2 filter-rejection bug (proxima-http/src/http2/server.rs):
    // a filter's Forbidden Err used to become RST_STREAM, silently
    // dropping the 403. It must render as a real response, and a
    // later request on the SAME connection must still succeed —
    // proving the rejection didn't poison the multiplexed socket.
    let dispatch: PipeHandle = into_handle(FilterRoutedPipe);
    let addr = spawn_native_server(dispatch).await;
    let tcp = TcpStream::connect(addr).await.expect("connect");
    let (mut h2_client, h2_conn) = h2::client::handshake(tcp).await.expect("handshake");
    let conn_task = tokio::spawn(async move {
        let _ = h2_conn.await;
    });

    let rejected_request = http::Request::builder()
        .method("GET")
        .uri("http://localhost/reject")
        .body(())
        .expect("request");
    let (response_future, _) = h2_client
        .send_request(rejected_request, true)
        .expect("send_request");
    let (status, body) = collect_body(response_future).await;
    assert_eq!(status, 403, "filter rejection renders as 403");
    assert_eq!(&body[..], b"blocked by filter");

    let ok_request = http::Request::builder()
        .method("GET")
        .uri("http://localhost/ok")
        .body(())
        .expect("request");
    let (response_future, _) = h2_client
        .send_request(ok_request, true)
        .expect("send_request");
    let (status, body) = collect_body(response_future).await;
    assert_eq!(
        status, 200,
        "connection must survive a prior filter rejection"
    );
    assert_eq!(&body[..], b"ok");

    drop(h2_client);
    drop(conn_task);
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_h2_listener_internal_error_still_resets_stream_only() {
    // Unchanged-behaviour regression: a genuinely internal (non-filter)
    // handler error must still RST_STREAM rather than render a body —
    // and, exactly as before this fix, must not take the connection
    // down with it.
    let dispatch: PipeHandle = into_handle(FilterRoutedPipe);
    let addr = spawn_native_server(dispatch).await;
    let tcp = TcpStream::connect(addr).await.expect("connect");
    let (mut h2_client, h2_conn) = h2::client::handshake(tcp).await.expect("handshake");
    let conn_task = tokio::spawn(async move {
        let _ = h2_conn.await;
    });

    let internal_request = http::Request::builder()
        .method("GET")
        .uri("http://localhost/internal")
        .body(())
        .expect("request");
    let (response_future, _) = h2_client
        .send_request(internal_request, true)
        .expect("send_request");
    let error = response_future
        .await
        .expect_err("internal error must still reset the stream, not render");
    assert!(error.is_reset(), "expected RST_STREAM, got {error:?}");
    assert_eq!(error.reason(), Some(h2::Reason::INTERNAL_ERROR));

    let ok_request = http::Request::builder()
        .method("GET")
        .uri("http://localhost/ok")
        .body(())
        .expect("request");
    let (response_future, _) = h2_client
        .send_request(ok_request, true)
        .expect("send_request");
    let (status, body) = collect_body(response_future).await;
    assert_eq!(
        status, 200,
        "connection must survive an internal error on another stream"
    );
    assert_eq!(&body[..], b"ok");

    drop(h2_client);
    drop(conn_task);
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_h2_listener_round_trip_echoes_request_body() {
    let dispatch: PipeHandle = into_handle(EchoBodyPipe);
    let addr = spawn_native_server(dispatch).await;
    let tcp = TcpStream::connect(addr).await.expect("connect");
    let (mut h2_client, h2_conn) = h2::client::handshake(tcp).await.expect("handshake");
    let conn_task = tokio::spawn(async move {
        let _ = h2_conn.await;
    });

    let request = http::Request::builder()
        .method("POST")
        .uri("http://localhost/echo")
        .body(())
        .expect("request");
    let (response_future, mut send_stream) = h2_client
        .send_request(request, false)
        .expect("send_request");
    send_stream
        .send_data(Bytes::from_static(b"hello native h2"), true)
        .expect("send body");
    let response = response_future.await.expect("response");
    assert_eq!(response.status(), 200);
    let mut body = response.into_body();
    let mut collected = Vec::new();
    while let Some(chunk) = body.data().await {
        let chunk = chunk.expect("chunk");
        let len = chunk.len();
        body.flow_control()
            .release_capacity(len)
            .expect("flow control");
        collected.extend_from_slice(&chunk);
    }
    assert_eq!(&collected[..], b"hello native h2");

    drop(h2_client);
    drop(conn_task);
}

/// Pre-existing (already-correct) behavior, pinned as a real regression
/// test instead of living only in `examples/any_listener_production.rs`'s
/// §4 narrative: a BODYLESS shed request gets the in-band 503 — never a
/// stream reset. No body-stream receiver is ever opened for this request
/// (`end_stream` arrives with the HEADERS frame), so this case was never
/// exposed to the defect [`native_h2_listener_body_carrying_shed_request_receives_in_band_503_not_reset`]
/// below proves fixed.
#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_h2_listener_bodyless_shed_request_receives_in_band_503() {
    let admission = proxima_listen::admission::ConnAdmission::new(1);
    let (entered_tx, entered_rx) = tokio::sync::oneshot::channel::<()>();
    let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
    let dispatch: PipeHandle = into_handle(SlowGatePipe {
        entered_tx: std::sync::Mutex::new(Some(entered_tx)),
        release_rx: std::sync::Mutex::new(Some(release_rx)),
    });
    let addr = spawn_native_server_with_admission(dispatch, admission).await;
    let tcp = TcpStream::connect(addr).await.expect("connect");
    let (mut h2_client, h2_conn) = h2::client::handshake(tcp).await.expect("handshake");
    let conn_task = tokio::spawn(async move {
        let _ = h2_conn.await;
    });

    let slow_request = http::Request::builder()
        .method("GET")
        .uri("http://localhost/slow")
        .body(())
        .expect("request");
    let (slow_response_future, _) = h2_client
        .send_request(slow_request, true)
        .expect("send_request");

    entered_rx.await.expect("slow handler signals entered");

    let shed_request = http::Request::builder()
        .method("GET")
        .uri("http://localhost/shed-bodyless")
        .body(())
        .expect("request");
    let (shed_response_future, _) = h2_client
        .send_request(shed_request, true)
        .expect("send_request");
    let (shed_status, shed_body) = collect_body(shed_response_future).await;
    assert_eq!(
        shed_status, 503,
        "the shed bodyless request must get the in-band 503"
    );
    assert_eq!(&shed_body[..], b"service unavailable");

    release_tx.send(()).expect("release send");
    let (slow_status, slow_body) = collect_body(slow_response_future).await;
    assert_eq!(slow_status, 200);
    assert_eq!(&slow_body[..], b"slow-ok");

    drop(h2_client);
    drop(conn_task);
}

/// The h2 defect this proves fixed: `ConnAdmission::request_admit`
/// shedding a BODY-CARRYING request used to never dispatch the built
/// `Request`, so its body-stream receiver (bundled into that `Request`)
/// dropped with it. The client's subsequent DATA frame then landed on a
/// closed channel, and the `BodyData` handler's `Some(Err(_))` arm reset
/// the stream (`RST_STREAM(INTERNAL_ERROR)`) instead of delivering the
/// 503 already queued. This sends a REAL body (not the bodyless GET the
/// pre-existing demo used), so the fix — draining the body in the
/// background instead of dropping it — is actually exercised.
#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_h2_listener_body_carrying_shed_request_receives_in_band_503_not_reset() {
    let admission = proxima_listen::admission::ConnAdmission::new(1);
    let (entered_tx, entered_rx) = tokio::sync::oneshot::channel::<()>();
    let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
    let dispatch: PipeHandle = into_handle(SlowGatePipe {
        entered_tx: std::sync::Mutex::new(Some(entered_tx)),
        release_rx: std::sync::Mutex::new(Some(release_rx)),
    });
    let addr = spawn_native_server_with_admission(dispatch, admission).await;
    let tcp = TcpStream::connect(addr).await.expect("connect");
    let (mut h2_client, h2_conn) = h2::client::handshake(tcp).await.expect("handshake");
    let conn_task = tokio::spawn(async move {
        let _ = h2_conn.await;
    });

    // Stream A: admitted, holds the listener's ONLY capacity slot open
    // until the test releases it below.
    let slow_request = http::Request::builder()
        .method("GET")
        .uri("http://localhost/slow")
        .body(())
        .expect("request");
    let (slow_response_future, _) = h2_client
        .send_request(slow_request, true)
        .expect("send_request");

    entered_rx.await.expect("slow handler signals entered");

    // Stream B: concurrent with A, carries a REAL body — must be shed
    // (capacity is exhausted) and must receive the in-band 503, never a
    // stream reset.
    let shed_request = http::Request::builder()
        .method("POST")
        .uri("http://localhost/shed-with-body")
        .body(())
        .expect("request");
    let (shed_response_future, mut shed_send_stream) = h2_client
        .send_request(shed_request, false)
        .expect("send_request");
    shed_send_stream
        .send_data(Bytes::from_static(b"a real request body, not bodyless"), true)
        .expect("send body");

    let shed_response = shed_response_future.await.expect(
        "body-carrying shed request must receive an in-band response, not a stream reset",
    );
    assert_eq!(
        shed_response.status(),
        503,
        "body-carrying shed request must get the in-band 503"
    );
    let mut shed_body = shed_response.into_body();
    let mut collected = Vec::new();
    while let Some(chunk) = shed_body.data().await {
        let chunk = chunk.expect("chunk — must not be a stream reset");
        let len = chunk.len();
        shed_body
            .flow_control()
            .release_capacity(len)
            .expect("flow control");
        collected.extend_from_slice(&chunk);
    }
    assert_eq!(&collected[..], b"service unavailable");

    release_tx.send(()).expect("release send");
    let (slow_status, slow_body) = collect_body(slow_response_future).await;
    assert_eq!(slow_status, 200);
    assert_eq!(&slow_body[..], b"slow-ok");

    drop(h2_client);
    drop(conn_task);
}
