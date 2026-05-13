//! Integration test for the full nginx-style multi-listener example.
//!
//! Drives `examples/config/nginx-multi/proxima.toml` via `App::load_full`,
//! starting two HTTP listeners on ephemeral ports. Uses hyper directly
//! against each bound port. Verifies:
//!   - per-listener routers (admin port doesn't leak to API port)
//!   - host filter (api mount only matches Host: localhost / api.example.com)
//!   - method filter (admin port rejects POST)
//!   - round-robin LB across three synth backends

#![allow(clippy::expect_used, clippy::unwrap_used)]
#![cfg(feature = "http1")]

use std::path::PathBuf;

use http_body_util::{BodyExt, Empty};
use hyper::Request as HyperRequest;
use hyper::body::Bytes;
use hyper_util::rt::TokioIo;
use proxima::App;
use tempfile::tempdir;

async fn http_get(
    addr: std::net::SocketAddr,
    path: &str,
    host_header: Option<&str>,
) -> (u16, String) {
    let stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    let io = TokioIo::new(stream);
    let (mut sender, conn) = hyper::client::conn::http1::handshake::<_, Empty<Bytes>>(io)
        .await
        .expect("handshake");
    tokio::spawn(async move {
        let _ = conn.await;
    });

    let host = host_header.unwrap_or(&format!("{addr}")).to_string();
    let req = HyperRequest::builder()
        .method("GET")
        .uri(path)
        .header("host", host)
        .body(Empty::<Bytes>::new())
        .expect("build req");
    let response = sender.send_request(req).await.expect("send");
    let status = response.status().as_u16();
    let body = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    (status, String::from_utf8_lossy(&body).to_string())
}

async fn http_post(addr: std::net::SocketAddr, path: &str) -> u16 {
    let stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    let io = TokioIo::new(stream);
    let (mut sender, conn) = hyper::client::conn::http1::handshake::<_, Empty<Bytes>>(io)
        .await
        .expect("handshake");
    tokio::spawn(async move {
        let _ = conn.await;
    });
    let req = HyperRequest::builder()
        .method("POST")
        .uri(path)
        .header("host", format!("{addr}"))
        .body(Empty::<Bytes>::new())
        .expect("build req");
    let response = sender.send_request(req).await.expect("send");
    response.status().as_u16()
}

#[proxima::test]
async fn nginx_multi_listener_config_routes_correctly() {
    let dir = tempdir().expect("tempdir");
    let static_dir = dir.path().join("static");
    std::fs::create_dir_all(&static_dir).expect("mkdir");
    std::fs::write(static_dir.join("index.html"), b"<h1>STATIC HOME</h1>").expect("write index");
    std::fs::write(static_dir.join("alpha.txt"), b"alpha-here\n").expect("write alpha");

    let config_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/config/nginx-multi/proxima.toml");
    let toml_text = std::fs::read_to_string(&config_path).expect("read config");
    // substitute /var/www/static and the bind ports for ephemeral ports
    // pick two free ports by binding briefly via std then closing.
    fn pick_port() -> u16 {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("pick port");
        listener.local_addr().expect("addr").port()
    }
    let api_port = pick_port();
    let admin_port = pick_port();
    let toml_text = toml_text
        .replace("/var/www/static", static_dir.to_str().expect("utf8"))
        .replace("0.0.0.0:8080", &format!("127.0.0.1:{api_port}"))
        .replace("0.0.0.0:8081", &format!("127.0.0.1:{admin_port}"));

    let mut app = App::new().expect("app");
    let handles = app
        .load_full(proxima::load::Spec::Toml(toml_text))
        .await
        .expect("load_full");
    assert_eq!(handles.len(), 2, "two [[listen]] blocks");

    let api_addr = handles[0].bind_addr().expect("api bind");
    let admin_addr = handles[1].bind_addr().expect("admin bind");

    // wait for both listeners to be reachable
    for addr in [api_addr, admin_addr] {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            if tokio::net::TcpStream::connect(addr).await.is_ok() {
                break;
            }
            if std::time::Instant::now() >= deadline {
                panic!("listener at {addr} not reachable");
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    }

    // 1. static file via api port (no host header → host filter on /api rejects, falls to /{*path} static)
    let (status, body) = http_get(api_addr, "/index.html", None).await;
    assert_eq!(status, 200);
    assert!(body.contains("STATIC HOME"), "got {body:?}");

    let (status, body) = http_get(api_addr, "/alpha.txt", None).await;
    assert_eq!(status, 200);
    assert!(body.contains("alpha-here"), "got {body:?}");

    // 2. /api/v1 on api port without right host → host filter rejects api,
    //    falls through to /{*path} static, /api/v1 isn't a file → fs returns 404.
    let (status, _) = http_get(api_addr, "/api/v1", None).await;
    assert_eq!(
        status, 404,
        "no-host /api/v1 falls through to static and 404s"
    );

    // 3. /api/v1 with Host: localhost hits the LB pool, round-robins.
    let mut bodies = Vec::new();
    for _ in 0..3 {
        let (status, body) = http_get(api_addr, "/api/v1", Some("localhost")).await;
        assert_eq!(status, 200);
        bodies.push(body);
    }
    let mut sorted = bodies.clone();
    sorted.sort();
    assert_eq!(
        sorted,
        vec!["from-be1\n", "from-be2\n", "from-be3\n"],
        "expected round-robin across 3 backends"
    );

    // 4. admin port — separate router
    let (status, body) = http_get(admin_addr, "/admin", None).await;
    assert_eq!(status, 200);
    assert!(body.contains("admin endpoint"), "got {body:?}");

    // 5. POST on admin port rejected by methods=["GET"]
    let status = http_post(admin_addr, "/admin").await;
    assert_eq!(status, 404, "POST should not match GET-only mount");

    // 6. admin response NOT visible on api port (no leakage between routers)
    let (_, body) = http_get(api_addr, "/admin", None).await;
    assert!(
        !body.contains("admin endpoint"),
        "admin pipe should not leak to api router; got {body:?}"
    );

    for handle in handles {
        handle.stop().await;
    }
}
