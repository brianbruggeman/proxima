#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

// Tier-2 network bench: real localhost sockets, no in-memory shortcuts.
//
// - http_listener_request: proxima HttpListenProtocol bound on 127.0.0.1:0,
//   hyper client drives single requests, measures rps + latency.
// - tcp_stream_listener_msg: proxima StreamListenerProtocol bound on 127.0.0.1:0,
//   raw TcpStream client sends a frame + reads response, measures msg/sec.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use http_body_util::{BodyExt, Empty};
use hyper::{Request as HyperRequest, body::Incoming};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use proxima::{App, MountTarget, RunConfig};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::runtime::Runtime;

fn build_runtime() -> Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("tokio runtime")
}

struct HttpFixture {
    addr: SocketAddr,
    shutdown: Option<proxima::Shutdown>,
}

impl Drop for HttpFixture {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            shutdown.stop();
        }
    }
}

fn boot_http_synth(runtime: &Runtime) -> HttpFixture {
    runtime.block_on(async {
        let mut app = App::new().expect("app");
        let handle = app
            .pipe("echo", json!({"synth": {"status": 200, "body": "ok"}}))
            .await
            .expect("register pipe");
        app.mount("/", MountTarget::Handle(handle)).expect("mount");
        let bind: SocketAddr = "127.0.0.1:0".parse().expect("bind addr");
        // bind ourselves so we can return the OS-chosen port to the bench.
        let listener = std::net::TcpListener::bind(bind).expect("std bind");
        listener.set_nonblocking(true).expect("nonblocking");
        let addr = listener.local_addr().expect("local addr");
        drop(listener);
        let shutdown = app
            .run_until_signal(RunConfig::http(addr))
            .await
            .expect("run");
        // give the listener a moment to actually bind on the spawned task.
        tokio::time::sleep(Duration::from_millis(50)).await;
        HttpFixture {
            addr,
            shutdown: Some(shutdown),
        }
    })
}

fn http_listener_request(criterion: &mut Criterion) {
    let runtime = build_runtime();
    let fixture = boot_http_synth(&runtime);
    let client: Client<_, Empty<Bytes>> = Client::builder(TokioExecutor::new()).build_http();
    let client = Arc::new(client);
    let url = format!("http://{}/", fixture.addr);
    let mut group = criterion.benchmark_group("network_http_listener");
    group.throughput(Throughput::Elements(1));
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("synth_200_keepalive", |bencher| {
        let client = client.clone();
        let url = url.clone();
        bencher.to_async(&runtime).iter(move || {
            let client = client.clone();
            let url = url.clone();
            async move {
                let req = HyperRequest::builder()
                    .uri(&url)
                    .body(Empty::<Bytes>::new())
                    .expect("build req");
                let resp: hyper::Response<Incoming> = client.request(req).await.expect("request");
                let _bytes = resp
                    .into_body()
                    .collect()
                    .await
                    .expect("collect body")
                    .to_bytes();
            }
        });
    });
    group.finish();
}

struct TcpFixture {
    addr: SocketAddr,
    shutdown: Option<proxima::Shutdown>,
}

impl Drop for TcpFixture {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            shutdown.stop();
        }
    }
}

fn boot_tcp_stream_synth(runtime: &Runtime) -> TcpFixture {
    runtime.block_on(async {
        let mut app = App::new().expect("app");
        let handle = app
            .pipe("echo", json!({"synth": {"status": 200, "body": "ok"}}))
            .await
            .expect("register pipe");
        app.mount("/", MountTarget::Handle(handle)).expect("mount");
        let bind: SocketAddr = "127.0.0.1:0".parse().expect("bind addr");
        let probe = std::net::TcpListener::bind(bind).expect("std bind");
        probe.set_nonblocking(true).expect("nonblocking");
        let addr = probe.local_addr().expect("local addr");
        drop(probe);
        let run = RunConfig {
            bind: addr,
            protocol: "stream".into(),
            spec: serde_json::Value::Null,
        };
        let shutdown = app.run_until_signal(run).await.expect("run");
        tokio::time::sleep(Duration::from_millis(50)).await;
        TcpFixture {
            addr,
            shutdown: Some(shutdown),
        }
    })
}

fn tcp_stream_listener_msg(criterion: &mut Criterion) {
    let runtime = build_runtime();
    let fixture = boot_tcp_stream_synth(&runtime);
    let mut group = criterion.benchmark_group("network_tcp_listener");
    group.throughput(Throughput::Elements(1));
    group.measurement_time(Duration::from_secs(5));
    // Each iter opens a fresh connection (no keepalive on raw stream proto),
    // writes a payload, drains response. The cost is dominated by connect +
    // accept on localhost, which is what gateway-shaped traffic actually
    // looks like for short-lived streams.
    group.bench_function("connect_write_read_short", |bencher| {
        let addr = fixture.addr;
        bencher.to_async(&runtime).iter(move || async move {
            let mut stream = TcpStream::connect(addr).await.expect("connect");
            stream.write_all(b"ping").await.expect("write");
            stream.flush().await.expect("flush");
            let mut buf = [0_u8; 64];
            let _ = stream.read(&mut buf).await;
            drop(stream);
        });
    });
    group.finish();
}

criterion_group!(benches, http_listener_request, tcp_stream_listener_msg,);
criterion_main!(benches);
