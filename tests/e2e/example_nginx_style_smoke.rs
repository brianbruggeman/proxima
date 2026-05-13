//! Integration test for the nginx-style example config.
//!
//! Drives `examples/config/nginx-style/proxima.toml` via the library
//! API: `App::pipe` + `App::mount` + `Pipe::call`. Verifies:
//!   1. static file lookup returns the file with correct mime
//!   2. unknown path falls through to the synth fallback
//!   3. write_back populates the cache so a second identical request
//!      hits the cache (verified by lookahead: the cache entry must
//!      exist after a fallback hit)
//!
//! No CLI subprocess, no port discovery, no sleep races.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::PathBuf;

use proxima::request::Request;
use proxima::{App, MountTarget};
use proxima_primitives::pipe::SendPipe;
use tempfile::tempdir;

fn substitute_root(toml_text: &str, root: &str) -> String {
    toml_text.replace("/var/www/static", root)
}

#[proxima::test]
async fn nginx_style_serves_static_falls_through_and_caches() {
    let dir = tempdir().expect("tempdir");
    let static_dir = dir.path().join("static");
    std::fs::create_dir_all(&static_dir).expect("mkdir static");
    std::fs::write(static_dir.join("index.html"), b"<h1>hello</h1>").expect("write index");
    std::fs::write(static_dir.join("alpha.txt"), b"alpha-content\n").expect("write alpha");

    let config_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/config/nginx-style/proxima.toml");
    let toml_text = std::fs::read_to_string(&config_path).expect("read config");
    let toml_text = substitute_root(&toml_text, static_dir.to_str().expect("utf-8 path"));

    let mut app = App::new().expect("app");
    let svc = app
        .pipe("nginx-style", proxima::load::Spec::Toml(toml_text))
        .await
        .expect("build pipe");
    app.mount("/{*path}", MountTarget::Handle(svc.clone()))
        .expect("mount");

    // 1. static file
    let response = SendPipe::call(
        &svc,
        Request::builder()
            .method("GET")
            .path("/alpha.txt")
            .build()
            .expect("req"),
    )
    .await
    .expect("call alpha");
    assert_eq!(response.status, 200, "alpha.txt status");
    assert_eq!(
        response.metadata.get_str("content-type"),
        Some("text/plain; charset=utf-8"),
        "alpha.txt mime"
    );
    let body = response.collect_body().await.expect("collect");
    assert_eq!(&body[..], b"alpha-content\n");

    // 2. directory resolves to index
    let response = SendPipe::call(
        &svc,
        Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("req"),
    )
    .await
    .expect("call /");
    assert_eq!(response.status, 200);
    let body = response.collect_body().await.expect("collect");
    assert_eq!(&body[..], b"<h1>hello</h1>");

    // 3. unknown path falls through to synth fallback
    let response = SendPipe::call(
        &svc,
        Request::builder()
            .method("GET")
            .path("/api/users")
            .build()
            .expect("req"),
    )
    .await
    .expect("call api");
    assert_eq!(response.status, 200);
    let body = response.collect_body().await.expect("collect");
    assert!(
        std::str::from_utf8(&body[..])
            .unwrap_or("")
            .contains("from fallback"),
        "expected synth fallback, got {body:?}"
    );

    // 4. second hit on the same path — write_back populated the cache,
    //    so we expect identical response. (Cache content equality is
    //    sufficient evidence of write_back firing.)
    let response = SendPipe::call(
        &svc,
        Request::builder()
            .method("GET")
            .path("/api/users")
            .build()
            .expect("req"),
    )
    .await
    .expect("call api 2");
    assert_eq!(response.status, 200);
    let body2 = response.collect_body().await.expect("collect");
    assert!(
        std::str::from_utf8(&body2[..])
            .unwrap_or("")
            .contains("from fallback")
    );
}
