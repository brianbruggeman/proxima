#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(feature = "http1")]

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use bytes::Bytes;
use http_body_util::Full;
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn as pipe_fn;
use hyper_util::rt::TokioIo;
use proxima::App;
use proxima::{Labels, PipeHandle};
use proxima::{MountTarget, RunConfig};
use serde_json::json;
use tokio::net::TcpListener;

const ECHO_HEADER: &str = "x-fake-upstream-call-count";

#[proxima::test]
async fn cached_http_pipe_hits_origin_once_then_cache() {
    let upstream_calls = Arc::new(AtomicUsize::new(0));
    let upstream_addr = start_fake_origin(upstream_calls.clone()).await;

    let mut app = App::new().expect("app should construct");
    let upstream_url = format!("http://{upstream_addr}");
    let composed = json!({
        "name": "cached-origin",
        "upstreams": [
            {"kv": "cache", "ttl": "1h", "max_entries": 100, "name": "cache"},
            {"http": upstream_url, "name": "origin"},
        ],
        "select": {"algorithm": "fallthrough", "miss_on": ["no_data"]},
        "write_back": [[1, 0]],
    });
    let handle = app
        .pipe("cached", composed)
        .await
        .expect("pipe should register");
    app.mount("/{*path}", MountTarget::Handle(handle.clone()))
        .expect("mount should succeed");

    let listener_addr = pick_free_addr().await;
    let shutdown = app
        .run_until_signal(RunConfig::http(listener_addr))
        .await
        .expect("run should start");

    let response_one = client_get(listener_addr, "/users/42").await;
    assert_eq!(
        response_one.status, 200,
        "first call returns origin response"
    );
    assert_eq!(
        response_one.body, b"hello from origin",
        "body matches origin"
    );
    assert_eq!(
        upstream_calls.load(Ordering::Relaxed),
        1,
        "first call hits origin"
    );

    let response_two = client_get(listener_addr, "/users/42").await;
    assert_eq!(response_two.status, 200, "second call still 200");
    assert_eq!(
        response_two.body, b"hello from origin",
        "second call matches body"
    );
    assert_eq!(
        upstream_calls.load(Ordering::Relaxed),
        1,
        "second call must NOT hit origin (cache hit)"
    );

    let response_other = client_get(listener_addr, "/users/99").await;
    assert_eq!(response_other.status, 200);
    assert_eq!(
        upstream_calls.load(Ordering::Relaxed),
        2,
        "different cache key must miss and hit origin"
    );

    shutdown.stop();
}

#[proxima::test]
async fn empty_app_returns_404_for_unmatched_path() {
    let app = App::new().expect("app should construct");
    let listener_addr = pick_free_addr().await;
    let shutdown = app
        .run_until_signal(RunConfig::http(listener_addr))
        .await
        .expect("run should start");

    let response = client_get(listener_addr, "/anything").await;
    assert_eq!(response.status, 404);

    shutdown.stop();
}

#[proxima::test]
async fn library_user_can_query_p50_p90_p99_from_app_metrics() {
    let upstream_calls = Arc::new(AtomicUsize::new(0));
    let upstream_addr = start_fake_origin(upstream_calls.clone()).await;

    let mut app = App::new().expect("app");
    let composed = json!({
        "name": "echo-cached",
        "upstreams": [
            {"kv": "cache", "ttl": "1h", "max_entries": 100},
            {"http": format!("http://{upstream_addr}"), "name": "origin"},
        ],
        "select": {"algorithm": "fallthrough", "miss_on": ["no_data"]},
        "write_back": [[1, 0]],
    });
    let svc = app.pipe("echo-cached", composed).await.expect("pipe");
    app.mount("/{*path}", MountTarget::Handle(svc.clone()))
        .expect("mount");

    let listener_addr = pick_free_addr().await;
    let shutdown = app
        .run_until_signal(RunConfig::http(listener_addr))
        .await
        .expect("run");

    for index in 0..50 {
        let _ = client_get(listener_addr, &format!("/items/{index}")).await;
    }
    for index in 0..50 {
        let _ = client_get(listener_addr, &format!("/items/{index}")).await;
    }

    let metrics = app
        .metrics()
        .expect("default LoadContext registers Metrics");

    // mounted via MountTarget::Handle (not Named), so the mount-site label
    // degrades to "anonymous" (TARGET 3 — pipe_label derives from the mount
    // site, not the pipe's own construction-time name; §7.4, intended).
    let upstream_labels = Labels::from_pairs(&[
        ("pipe", "anonymous"),
        ("upstream", "origin"),
        ("status_class", "2xx"),
    ]);
    let summary = metrics
        .histogram_summary("proxima.upstream.latency_ms", &upstream_labels)
        .expect("upstream latency histogram should have samples");
    assert!(
        summary.count >= 50,
        "should have >= 50 origin samples, got {}",
        summary.count
    );
    assert!(summary.p50 >= 0.0);
    assert!(summary.p90 >= summary.p50);
    assert!(summary.p99 >= summary.p90);

    let cache_labels = Labels::from_pairs(&[("cache_name", "kv:cache"), ("pipe", "anonymous")]);
    let hits = metrics
        .counter("proxima.cache.hits_total", &cache_labels)
        .expect("cache hits counter should exist");
    let misses = metrics
        .counter("proxima.cache.misses_total", &cache_labels)
        .expect("cache misses counter should exist");
    assert!(hits >= 40, "second pass should mostly hit; got {hits} hits");
    assert!(misses >= 40, "first pass should miss; got {misses} misses");

    let upstream_calls_value = metrics
        .counter("proxima.upstream.calls_total", &upstream_labels)
        .expect("upstream calls counter should exist");
    assert!(upstream_calls_value >= 50);

    shutdown.stop();
}

#[proxima::test]
async fn http_upstream_streams_multi_chunk_response_without_buffering() {
    let upstream_addr = start_chunked_origin().await;
    let mut app = App::new().expect("app");
    app.pipe(
        "passthrough",
        json!({"http": format!("http://{upstream_addr}"), "name": "passthrough"}),
    )
    .await
    .expect("pipe");
    app.mount("/{*path}", MountTarget::Named("passthrough".into()))
        .expect("mount");

    let listener_addr = pick_free_addr().await;
    let shutdown = app
        .run_until_signal(RunConfig::http(listener_addr))
        .await
        .expect("run");

    let response = client_get(listener_addr, "/anything").await;
    assert_eq!(response.status, 200);
    let body_text = String::from_utf8_lossy(&response.body);
    let position1 = body_text.find("chunk1").expect("chunk1 in body");
    let position2 = body_text.find("chunk2").expect("chunk2 in body");
    let position3 = body_text.find("chunk3").expect("chunk3 in body");
    assert!(
        position1 < position2 && position2 < position3,
        "chunks must appear in order"
    );
    shutdown.stop();
}

async fn start_chunked_origin() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _peer)) = listener.accept().await else {
                continue;
            };
            tokio::spawn(async move {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let mut buffer = [0u8; 1024];
                let _ = socket.read(&mut buffer).await;
                let response = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n6\r\nchunk1\r\n6\r\nchunk2\r\n6\r\nchunk3\r\n0\r\n\r\n";
                let _ = socket.write_all(response).await;
                let _ = socket.shutdown().await;
            });
        }
    });
    addr
}

#[proxima::test]
async fn listener_drains_in_flight_request_before_shutdown() {
    use proxima::{ProximaError, Request, Response, into_handle};
    use proxima_primitives::pipe::SendPipe;
    use std::future::Future;
    use std::sync::atomic::{AtomicBool, Ordering};

    let request_finished = Arc::new(AtomicBool::new(false));
    let finish_flag = request_finished.clone();

    struct SlowPipe {
        finish_flag: Arc<AtomicBool>,
    }
    impl SendPipe for SlowPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
            let flag = self.finish_flag.clone();
            async move {
                tokio::time::sleep(Duration::from_millis(150)).await;
                flag.store(true, Ordering::Relaxed);
                Ok(Response::ok(bytes::Bytes::from_static(b"slow done")))
            }
        }
    }


    let mut app = App::new().expect("app");
    app.pipe(
        "slow",
        json!({"http": "http://placeholder", "name": "slow"}),
    )
    .await
    .expect("seed pipe");
    let slow_handle = into_handle(SlowPipe { finish_flag });
    app.mount("/", MountTarget::Handle(slow_handle))
        .expect("mount");

    let listener_addr = pick_free_addr().await;
    let shutdown = app
        .run_until_signal(RunConfig {
            bind: listener_addr,
            protocol: "http".into(),
            spec: json!({"drain_timeout_ms": 5000}),
        })
        .await
        .expect("run");

    let request_task = tokio::spawn(async move { client_get(listener_addr, "/").await });
    tokio::time::sleep(Duration::from_millis(30)).await;
    shutdown.stop();

    let response = request_task.await.expect("join");
    assert_eq!(response.status, 200);
    assert!(
        request_finished.load(Ordering::Relaxed),
        "in-flight request should complete during drain"
    );
}

#[proxima::test]
async fn fallthroughs_metric_emitted_on_cache_miss_origin_hit() {
    use proxima::Labels;

    let upstream_calls = Arc::new(AtomicUsize::new(0));
    let upstream_addr = start_fake_origin(upstream_calls.clone()).await;

    let mut app = App::new().expect("app");
    let composed = json!({
        "name": "cached-origin",
        "upstreams": [
            {"kv": "cache", "max_entries": 100, "name": "cache"},
            {"http": format!("http://{upstream_addr}"), "name": "origin"},
        ],
        "select": {"algorithm": "fallthrough", "miss_on": ["no_data"]},
        "write_back": [["origin", "cache"]],
    });
    app.pipe("cached-origin", composed).await.expect("pipe");
    app.mount("/{*path}", "cached-origin").expect("mount");

    let listener_addr = pick_free_addr().await;
    let shutdown = app
        .run_until_signal(RunConfig::http(listener_addr))
        .await
        .expect("run");

    let _ = client_get(listener_addr, "/items/42").await;

    let metrics = app.metrics().expect("metrics");
    let labels = Labels::from_pairs(&[("pipe", "cached-origin")]);
    let fallthroughs = metrics
        .counter(
            "proxima.selection.fallthroughs_total",
            &Labels::from_pairs(&[("pipe", "cached-origin"), ("reason", "NoData")]),
        )
        .or_else(|| metrics.counter("proxima.selection.fallthroughs_total", &labels))
        .expect("fallthrough counter present");
    assert!(
        fallthroughs >= 1,
        "fallthrough counter should fire on cache miss"
    );

    let request_count = metrics
        .counter("proxima.requests_total", &labels)
        .expect("requests_total counter");
    assert!(request_count >= 1);

    shutdown.stop();
}

#[proxima::test]
async fn http_listener_emits_request_latency_histogram() {
    use proxima::Labels;

    let upstream_calls = Arc::new(AtomicUsize::new(0));
    let upstream_addr = start_fake_origin(upstream_calls.clone()).await;

    let mut app = App::new().expect("app");
    app.pipe(
        "svc",
        json!({"http": format!("http://{upstream_addr}"), "name": "svc"}),
    )
    .await
    .expect("pipe");
    app.mount("/{*path}", "svc").expect("mount");

    let listener_addr = pick_free_addr().await;
    let shutdown = app
        .run_until_signal(RunConfig::http(listener_addr))
        .await
        .expect("run");

    for index in 0..10 {
        let _ = client_get(listener_addr, &format!("/x/{index}")).await;
    }

    let metrics = app.metrics().expect("metrics");
    let summary = metrics
        .histogram_summary(
            "proxima.request.latency_ms",
            &Labels::from_pairs(&[("pipe", "svc")]),
        )
        .expect("request latency histogram should be populated");
    assert!(summary.count >= 10);

    shutdown.stop();
}

#[proxima::test]
async fn traceparent_header_is_propagated_to_upstream() {
    let received_headers: Arc<std::sync::Mutex<Vec<(String, String)>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let collector = received_headers.clone();
    let upstream_addr = start_header_capture_origin(collector).await;

    let mut app = App::new().expect("app");
    app.pipe(
        "svc",
        json!({"http": format!("http://{upstream_addr}"), "name": "svc"}),
    )
    .await
    .expect("pipe");
    app.mount("/{*path}", "svc").expect("mount");

    let listener_addr = pick_free_addr().await;
    let shutdown = app
        .run_until_signal(RunConfig::http(listener_addr))
        .await
        .expect("run");

    let trace_header = "00-0af7651916cd43dd8448eb211c80319c-b9c7c989f97918e1-01";
    let response =
        client_get_with_headers(listener_addr, "/probe", &[("traceparent", trace_header)]).await;
    assert_eq!(response.status, 200);

    let captured = received_headers.lock().expect("lock").clone();
    let traceparent = captured
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("traceparent"));
    assert!(
        traceparent.is_some(),
        "upstream should have received traceparent"
    );
    assert_eq!(traceparent.unwrap().1, trace_header);

    shutdown.stop();
}

async fn start_header_capture_origin(
    collector: Arc<std::sync::Mutex<Vec<(String, String)>>>,
) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        loop {
            let Ok((socket, _peer)) = listener.accept().await else {
                continue;
            };
            let collector = collector.clone();
            tokio::spawn(async move {
                let io = TokioIo::new(socket);
                let collector = collector.clone();
                let handler = pipe_fn(move |req: hyper::Request<Incoming>| {
                    let collector = collector.clone();
                    async move {
                        let mut entries = collector.lock().expect("lock");
                        for (name, value) in req.headers().iter() {
                            if let Ok(text) = value.to_str() {
                                entries.push((name.as_str().to_string(), text.to_string()));
                            }
                        }
                        Ok::<_, std::convert::Infallible>(
                            hyper::Response::builder()
                                .status(200)
                                .body(http_body_util::Full::new(Bytes::from_static(b"ok")))
                                .unwrap_or_else(|_| {
                                    hyper::Response::new(http_body_util::Full::new(
                                        Bytes::from_static(b"err"),
                                    ))
                                }),
                        )
                    }
                });
                let _ = hyper::server::conn::http1::Builder::new()
                    .serve_connection(io, handler)
                    .await;
            });
        }
    });
    addr
}

async fn client_get_with_headers(
    addr: SocketAddr,
    path: &str,
    headers: &[(&str, &str)],
) -> ClientResponse {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    let mut request = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n");
    for (name, value) in headers {
        request.push_str(&format!("{name}: {value}\r\n"));
    }
    request.push_str("\r\n");
    stream.write_all(request.as_bytes()).await.expect("write");
    let mut bytes = Vec::new();
    stream.read_to_end(&mut bytes).await.expect("read");
    let header_end = bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("header terminator");
    let header_text = std::str::from_utf8(&bytes[..header_end]).expect("utf8");
    let status_line = header_text.split("\r\n").next().expect("status line");
    let status = status_line
        .split(' ')
        .nth(1)
        .expect("status code")
        .parse::<u16>()
        .expect("parse");
    let body = bytes[header_end + 4..].to_vec();
    ClientResponse { status, body }
}

#[proxima::test]
async fn app_update_pipe_swaps_in_new_handle_for_existing_mount() {
    use proxima::{ProximaError, Request, Response, into_handle};
    use proxima_primitives::pipe::SendPipe;
    use std::future::Future;

    struct StaticBody(&'static str);
    impl SendPipe for StaticBody {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
            let body = self.0;
            async move { Ok(Response::ok(bytes::Bytes::from_static(body.as_bytes()))) }
        }
    }


    let mut app = App::new().expect("app");
    app.pipe("svc", Spec::Handle(into_handle(StaticBody("first"))))
        .await
        .expect("seed");
    app.mount("/x", "svc").expect("mount");

    let listener_addr = pick_free_addr().await;
    let shutdown = app
        .run_until_signal(RunConfig::http(listener_addr))
        .await
        .expect("run");

    let response_one = client_get(listener_addr, "/x").await;
    assert!(
        String::from_utf8_lossy(&response_one.body).contains("first"),
        "first response should carry 'first': {:?}",
        response_one.body
    );

    app.update_pipe("svc", Spec::Handle(into_handle(StaticBody("second"))))
        .await
        .expect("update");

    let response_two = client_get(listener_addr, "/x").await;
    assert!(
        String::from_utf8_lossy(&response_two.body).contains("second"),
        "second response should carry 'second': {:?}",
        response_two.body
    );
    assert!(
        !String::from_utf8_lossy(&response_two.body).contains("first"),
        "should NOT carry 'first' after update"
    );

    shutdown.stop();
}

use proxima::Spec;

#[proxima::test]
async fn http_listener_rejects_oversize_body() {
    let mut app = App::new().expect("app");
    use proxima::{ProximaError, Request, Response, into_handle};
    use proxima_primitives::pipe::SendPipe;
    use std::future::Future;
    struct Echo;
    impl SendPipe for Echo {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
            async move {
                let (_, bytes) = request.body_bytes().await?;
                Ok(Response::ok(bytes))
            }
        }
    }

    app.pipe("svc", Spec::Handle(into_handle(Echo)))
        .await
        .expect("seed");
    app.mount("/", "svc").expect("mount");

    let listener_addr = pick_free_addr().await;
    let shutdown = app
        .run_until_signal(RunConfig {
            bind: listener_addr,
            protocol: "http".into(),
            spec: json!({"max_body_bytes": 16}),
        })
        .await
        .expect("run");

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let payload = "x".repeat(64);
    let request = format!(
        "POST / HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        payload.len(),
        payload
    );
    let mut stream = tokio::net::TcpStream::connect(listener_addr)
        .await
        .expect("connect");
    stream.write_all(request.as_bytes()).await.expect("write");
    let mut bytes = Vec::new();
    stream.read_to_end(&mut bytes).await.expect("read");
    let header_end = bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("header terminator");
    let header_text = std::str::from_utf8(&bytes[..header_end]).expect("utf8");
    let status = header_text
        .split("\r\n")
        .next()
        .and_then(|line| line.split(' ').nth(1))
        .and_then(|code| code.parse::<u16>().ok())
        .expect("status");
    // 413 Payload Too Large is the RFC 9110 §15.5.14 status for this
    // condition. The new native listener returns it directly; the
    // hyper-based listener used to fold this into a generic 400 via
    // ProximaError::Body, which was less precise.
    assert!(
        status == 413,
        "oversize body should produce 413 (got {status}): {header_text}"
    );

    shutdown.stop();
}

// Buffered uring path can't peek the socket for EOF without consuming
// bytes (tokio_uring 0.5 has no MSG_PEEK), so cancel-on-disconnect
// only fires on the streaming path. Tracked as a Stage 7 serve-loop
// restructure (continuous-read model).
#[cfg(not(all(target_os = "linux", feature = "io-uring")))]
#[proxima::test]
async fn http_listener_cancels_dispatch_when_client_disconnects_mid_request() {
    use proxima::{ProximaError, Request, Response, into_handle};
    use proxima_primitives::pipe::SendPipe;
    use std::future::Future;
    use std::sync::atomic::{AtomicBool, Ordering};

    // Pipe that sleeps long enough that the test client will
    // disconnect before it can finish. It must observe its
    // cancel Signal firing to set the `cancelled` flag.
    struct SlowAndCancelable {
        observed_cancel: Arc<AtomicBool>,
    }
    impl SendPipe for SlowAndCancelable {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
            let cancel = request.context.cancel.clone();
            let observed = self.observed_cancel.clone();
            async move {
                tokio::select! {
                    _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => {
                        Ok(Response::ok(
                            bytes::Bytes::from_static(b"never"),
                        ))
                    }
                    _ = cancel.fired() => {
                        observed.store(true, Ordering::SeqCst);
                        Err(ProximaError::Body("client disconnected".into()))
                    }
                }
            }
        }
    }


    let observed_cancel = Arc::new(AtomicBool::new(false));
    let mut app = App::new().expect("app");
    app.pipe(
        "slow",
        Spec::Handle(into_handle(SlowAndCancelable {
            observed_cancel: observed_cancel.clone(),
        })),
    )
    .await
    .expect("seed");
    app.mount("/", "slow").expect("mount");

    let listener_addr = pick_free_addr().await;
    let shutdown = app
        .run_until_signal(RunConfig::http(listener_addr))
        .await
        .expect("run");

    // Connect, send a request head, then drop the connection
    // before the response can arrive.
    use tokio::io::AsyncWriteExt;
    let mut stream = tokio::net::TcpStream::connect(listener_addr)
        .await
        .expect("connect");
    stream
        .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await
        .expect("write");
    stream.flush().await.expect("flush");
    // Brief moment for the server to receive + start dispatch.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    drop(stream); // client disconnects

    // Give the server a moment to detect EOF + fire cancel.
    for _ in 0..50 {
        if observed_cancel.load(Ordering::SeqCst) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }

    assert!(
        observed_cancel.load(Ordering::SeqCst),
        "Pipe must observe its cancel Signal firing on client disconnect"
    );

    shutdown.stop();
}

#[proxima::test]
async fn quiesce_returns_503_during_window() {
    use proxima::{ProximaError, Request, Response, into_handle};
    use proxima_primitives::pipe::SendPipe;
    use std::future::Future;
    struct Always200;
    impl SendPipe for Always200 {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
            async move { Ok(Response::ok(bytes::Bytes::from_static(b"hi"))) }
        }
    }

    let mut app = App::new().expect("app");
    app.pipe("svc", Spec::Handle(into_handle(Always200)))
        .await
        .expect("seed");
    app.mount("/", "svc").expect("mount");

    let listener_addr = pick_free_addr().await;
    let shutdown = app
        .run_until_signal(RunConfig {
            bind: listener_addr,
            protocol: "http".into(),
            spec: json!({
                "quiesce_duration_ms": 200,
                "quiesce_status": 503,
                "quiesce_retry_after": "2",
                "drain_timeout_ms": 1000,
            }),
        })
        .await
        .expect("run");

    shutdown.stop();
    tokio::time::sleep(Duration::from_millis(30)).await;

    let response = client_get(listener_addr, "/anything").await;
    assert_eq!(response.status, 503, "quiesce window should return 503");
}

#[proxima::test]
async fn cache_spec_without_caps_errors() {
    use proxima::{LoadContext, load};
    let context = LoadContext::with_default_registry().expect("ctx");
    let outcome = load(json!({"kv": "cache"}), &context).await;
    assert!(
        matches!(outcome, Err(proxima::ProximaError::Config(_))),
        "kv:memory without caps must error",
    );
}

#[proxima::test]
async fn template_expansion_in_injected_headers() {
    let captured: Arc<std::sync::Mutex<Vec<(String, String)>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let collector = captured.clone();
    let upstream_addr = start_header_capture_origin(collector).await;

    let mut app = App::new().expect("app");
    let _ = app
        .pipe(
            "svc",
            json!({
                "http": format!("http://{upstream_addr}"),
                "name": "svc",
                "headers": { "request": { "x-trace": "{{request.trace_id}}" } },
            }),
        )
        .await
        .expect("pipe");
    app.mount("/{*path}", "svc").expect("mount");

    let listener_addr = pick_free_addr().await;
    let shutdown = app
        .run_until_signal(RunConfig::http(listener_addr))
        .await
        .expect("run");

    let _ = client_get_with_headers(
        listener_addr,
        "/probe",
        &[(
            "traceparent",
            "00-deadbeefdeadbeefdeadbeefdeadbeef-aaaabbbbccccdddd-01",
        )],
    )
    .await;

    let entries = captured.lock().expect("lock").clone();
    let trace_header = entries
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("x-trace"))
        .expect("x-trace header on upstream");
    assert!(
        trace_header.1.starts_with("00-"),
        "template should expand trace_id: {:?}",
        trace_header.1
    );

    shutdown.stop();
}

#[proxima::test]
async fn requests_total_emitted_on_404_with_unrouted_label() {
    use proxima::Labels;

    let app = App::new().expect("app");
    let listener_addr = pick_free_addr().await;
    let shutdown = app
        .run_until_signal(RunConfig::http(listener_addr))
        .await
        .expect("run");

    let response = client_get(listener_addr, "/missing").await;
    assert_eq!(response.status, 404);

    let metrics = app.metrics().expect("metrics");
    let count = metrics
        .counter(
            "proxima.requests_total",
            &Labels::from_pairs(&[("pipe", "__unrouted__"), ("status_class", "4xx")]),
        )
        .expect("404 should emit requests_total with __unrouted__ pipe");
    assert!(count >= 1);

    shutdown.stop();
}

#[proxima::test]
async fn connections_accepted_total_fires_per_accept() {
    use proxima::Labels;
    let app = App::new().expect("app");
    let listener_addr = pick_free_addr().await;
    let shutdown = app
        .run_until_signal(RunConfig::http(listener_addr))
        .await
        .expect("run");

    let _ = client_get(listener_addr, "/").await;
    let _ = client_get(listener_addr, "/").await;
    let _ = client_get(listener_addr, "/").await;

    let metrics = app.metrics().expect("metrics");
    let count = metrics
        .counter(
            "proxima.connections_accepted_total",
            &Labels::from_pairs(&[("listener", "http")]),
        )
        .expect("connections_accepted_total should be set");
    assert!(count >= 3, "expected at least 3 accepts, got {count}");

    shutdown.stop();
}

#[proxima::test]
async fn shutdown_signal_stops_listener() {
    let app = App::new().expect("app should construct");
    let listener_addr = pick_free_addr().await;
    let shutdown = app
        .run_until_signal(RunConfig::http(listener_addr))
        .await
        .expect("run should start");

    shutdown.stop();
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        TcpListener::bind(listener_addr).await.is_ok(),
        "port should be free after shutdown"
    );
}

#[proxima::test]
async fn library_compose_cache_then_origin_via_handle() {
    let upstream_calls = Arc::new(AtomicUsize::new(0));
    let upstream_addr = start_fake_origin(upstream_calls.clone()).await;
    let mut app = App::new().expect("app");

    let cache: PipeHandle = app
        .pipe(
            "cache",
            json!({"kv": "cache", "ttl": "10m", "max_entries": 100}),
        )
        .await
        .expect("cache");
    let origin: PipeHandle = app
        .pipe("origin", json!({"http": format!("http://{upstream_addr}")}))
        .await
        .expect("origin");

    let composed_spec = json!({
        "name": "compose",
        "upstreams": [
            {"kv": "cache", "max_entries": 100, "name": "inline-cache"},
            {"http": format!("http://{upstream_addr}"), "name": "inline-origin"},
        ],
        "select": {"algorithm": "fallthrough", "miss_on": ["no_data"]},
        "write_back": [[1, 0]],
    });
    let composed = app.pipe("composed", composed_spec).await.expect("composed");
    app.mount("/x", MountTarget::Handle(composed.clone()))
        .expect("mount");

    let _ = cache;
    let _ = origin;

    let listener_addr = pick_free_addr().await;
    let shutdown = app
        .run_until_signal(RunConfig::http(listener_addr))
        .await
        .expect("run");

    let _ = client_get(listener_addr, "/x").await;
    let _ = client_get(listener_addr, "/x").await;
    assert_eq!(
        upstream_calls.load(Ordering::Relaxed),
        1,
        "second call must hit cache"
    );
    shutdown.stop();
}

async fn start_fake_origin(counter: Arc<AtomicUsize>) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind origin");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        loop {
            let Ok((socket, _peer)) = listener.accept().await else {
                continue;
            };
            let counter = counter.clone();
            tokio::spawn(async move {
                let io = TokioIo::new(socket);
                let counter = counter.clone();
                let handler = pipe_fn(move |_req: hyper::Request<Incoming>| {
                    let counter = counter.clone();
                    async move {
                        let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                        let response = hyper::Response::builder()
                            .status(200)
                            .header(ECHO_HEADER, count.to_string())
                            .body(Full::new(Bytes::from_static(b"hello from origin")))
                            .unwrap_or_else(|_| {
                                hyper::Response::new(Full::new(Bytes::from_static(b"err")))
                            });
                        Ok::<_, Infallible>(response)
                    }
                });
                let _ = http1::Builder::new().serve_connection(io, handler).await;
            });
        }
    });
    addr
}

async fn pick_free_addr() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    drop(listener);
    addr
}

struct ClientResponse {
    status: u16,
    body: Vec<u8>,
}

async fn client_get(addr: SocketAddr, path: &str) -> ClientResponse {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    let request = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request");
    let mut bytes = Vec::new();
    stream.read_to_end(&mut bytes).await.expect("read");
    let header_end = bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("response should contain header terminator");
    let header_text = std::str::from_utf8(&bytes[..header_end]).expect("header utf8");
    let mut header_lines = header_text.split("\r\n");
    let status_line = header_lines.next().expect("status line");
    let status = status_line
        .split(' ')
        .nth(1)
        .expect("status code in line")
        .parse::<u16>()
        .expect("status parses");
    let body = bytes[header_end + 4..].to_vec();
    ClientResponse { status, body }
}
