//! End-to-end smoke for `#[proxima::test]`. Covers: plain async on the adaptive
//! default + explicit prime + explicit tokio; rstest `#[case]`/`#[case::desc]`/
//! `#[values]` parameterization; `-> Result` return parity; and cassette
//! record->replay. Failure-reporting semantics are proven by the unit tests in
//! `proxima::test_support`.
// the harness is opt-in: this file builds only with the test-support feature, so
// the default `cargo test` is unaffected. The proxima-test lane runs it with
// `--features test-prime`.
#![cfg(feature = "test-support")]
// test code: expect() with a message is the project convention for tests.
#![allow(clippy::expect_used)]

#[cfg(any(
    feature = "runtime-tokio",
    all(
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    )
))]
use bytes::Bytes;

#[proxima::test]
async fn default_runtime_plain_async_passes() {
    assert_eq!(2 + 2, 4);
}

#[proxima::test(runtime = "prime")]
async fn explicit_prime_passes() {
    assert_eq!(6 * 7, 42);
}

#[proxima::test(runtime = "tokio")]
async fn explicit_tokio_passes() {
    assert_eq!(40 + 2, 42);
}

// rstest #[case] parity with terse case::desc naming -> sized::small / sized::large.
#[proxima::test]
#[case::small(1)]
#[case::large(1_000)]
async fn sized(#[case] value: u32) {
    assert!(value >= 1);
}

// explicit prime + parameterized.
#[proxima::test(runtime = "prime")]
#[case(2)]
#[case(4)]
async fn prime_even_cases(#[case] value: u32) {
    assert_eq!(value % 2, 0);
}

// rstest #[values] matrix.
#[proxima::test]
async fn values_matrix(#[values(1_u32, 2, 3)] value: u32) {
    assert!((1..=3).contains(&value));
}

// `-> Result` return parity with #[proxima::test]/#[rstest].
#[proxima::test]
async fn returns_ok_result() -> Result<(), String> {
    if 2 + 2 == 4 {
        Ok(())
    } else {
        Err("arithmetic broke".to_string())
    }
}

// Cassette record -> replay through the macro: `cx` is the injected TestCtx
// handle (stripped by type), the body drives a Pipe, and the committed
// `tests/cassettes/weather.jsonl` makes subsequent runs replay deterministically
// with no live dependency (SmokeUpstream is never called on replay).
//
// cfg-gated with `weather_cassette_round_trips` below, its only user — see
// that test for why it needs a real runtime backend.
#[cfg(any(
    feature = "runtime-tokio",
    all(
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    )
))]
struct SmokeUpstream;

#[cfg(any(
    feature = "runtime-tokio",
    all(
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    )
))]
impl proxima_primitives::pipe::SendPipe for SmokeUpstream {
    type In = proxima::Request<Bytes>;
    type Out = proxima::Response<Bytes>;
    type Err = proxima::ProximaError;

    fn call(
        &self,
        _request: proxima::Request<Bytes>,
    ) -> impl std::future::Future<Output = Result<proxima::Response<Bytes>, proxima::ProximaError>> + Send
    {
        async { Ok(proxima::Response::new(200).with_body("{\"temp\":12}")) }
    }
}


// --- #[proxima::fixture]: native rstest-style fixtures, no rstest dep ---

#[proxima::fixture]
fn answer() -> u32 {
    42
}

// async fixture taking another fixture as a dependency (resolved by name).
#[proxima::fixture]
async fn doubled(answer: u32) -> u32 {
    answer * 2
}

// #[default(expr)] supplies a dependency's value in default().
#[proxima::fixture]
fn greeting(#[default("world")] name: &'static str) -> String {
    format!("hi {name}")
}

// #[once]: computed once per process, shared as &'static T.
#[proxima::fixture(once)]
fn shared_answer() -> u32 {
    42
}

#[proxima::test]
async fn uses_plain_fixture(answer: u32) {
    assert_eq!(answer, 42);
}

#[proxima::test]
async fn uses_once_fixture(shared_answer: &'static u32) {
    assert_eq!(*shared_answer, 42);
}

#[proxima::test(runtime = "prime")]
async fn uses_async_dep_fixture(doubled: u32) {
    assert_eq!(doubled, 84);
}

#[proxima::test]
async fn uses_default_dep(greeting: String) {
    assert_eq!(greeting, "hi world");
}

// #[with(..)] overrides the dependency at the call site (partial_1).
#[proxima::test]
async fn uses_with_override(#[with("mars")] greeting: String) {
    assert_eq!(greeting, "hi mars");
}

// #[from(..)] resolves a differently-named param from a fixture.
#[proxima::test]
async fn uses_from_alias(#[from(answer)] aliased: u32) {
    assert_eq!(aliased, 42);
}

// proxima's middleware/pipe primitives composed as a fixture stack: a
// source/sink synth wrapped in retry middleware. rate-limit, record, and replay
// compose identically (each wraps an inner Pipe), so a fixture can hand the body
// any stack — the substrate primitives ARE the fixture vocabulary.
#[proxima::fixture]
fn upstream_stack() -> proxima::PipeHandle {
    let synth = proxima::SynthUpstream::new("synth", 200, "ok");
    let with_retries = proxima::Retry::new(proxima::into_handle(synth)).with_max_attempts(3);
    proxima::into_handle(with_retries)
}

#[proxima::test]
async fn pipe_stack_fixture(upstream_stack: proxima::PipeHandle) {
    let request = proxima::Request::builder()
        .method("GET")
        .path("/")
        .build()
        .expect("request");
    let response = proxima_primitives::pipe::SendPipe::call(&upstream_stack, request)
        .await
        .expect("call");
    assert_eq!(response.status, 200);
    let body = response.collect_body().await.expect("body");
    assert_eq!(&body[..], b"ok");
}

// config fixture: proxima's first-class config substrate as a test input —
// parsed format-agnostically, then realized as a live Pipe. rstest can't.
#[proxima::fixture]
async fn cache_cfg() -> proxima::test_support::ConfigFixture {
    proxima::test_support::ConfigFixture::from_raw(
        r#"{"synth":{"status":200,"body":"cfg-ok"}}"#,
        Some("json"),
    )
    .await
    .expect("config fixture")
}

#[proxima::test]
async fn config_fixture_builds_pipe(cache_cfg: proxima::test_support::ConfigFixture) {
    assert_eq!(cache_cfg.value()["synth"]["status"].as_u64(), Some(200));
    let pipe = cache_cfg.into_pipe().await.expect("pipe");
    let request = proxima::Request::builder()
        .method("GET")
        .path("/")
        .build()
        .expect("request");
    let response = proxima_primitives::pipe::SendPipe::call(&pipe, request)
        .await
        .expect("call");
    assert_eq!(response.status, 200);
    let body = response.collect_body().await.expect("body");
    assert_eq!(&body[..], b"cfg-ok");
}

// config-matrix: parametrize a test across config profiles — a values dimension
// whose literals are config sources, each built into a ConfigFixture per case.
#[proxima::test]
async fn config_matrix(
    #[config_values(
        r#"{"synth":{"status":200,"body":"profile-a"}}"#,
        r#"{"synth":{"status":200,"body":"profile-b"}}"#
    )]
    cfg: proxima::test_support::ConfigFixture,
) {
    let pipe = cfg.into_pipe().await.expect("pipe");
    let request = proxima::Request::builder()
        .method("GET")
        .path("/")
        .build()
        .expect("request");
    let response = proxima_primitives::pipe::SendPipe::call(&pipe, request)
        .await
        .expect("call");
    assert_eq!(response.status, 200);
}

// teardown: the body registers async cleanup via cx.defer; it runs after the
// body on the same runtime (pass-or-panic guarantee proven by the test_support
// unit test). End-to-end here that the macro + TestCtx + defer compose.
#[proxima::test]
async fn uses_defer(cx: proxima::test_support::TestCtx) {
    cx.defer(async move {
        // async cleanup would go here (close a handle, drop a temp resource).
    });
    assert_eq!(1 + 1, 2);
}

// hot-swap fixture: a live control plane, reconfigured mid-test. The router
// reflects the new impl atomically (DaemonControlPlane::apply → App::update_pipe).
// rstest fundamentally cannot express this (no runtime/router/snapshot).
#[proxima::fixture]
async fn live_plane() -> proxima::test_support::LivePlane {
    let plane = proxima::test_support::LivePlane::with_pipe(
        "leaf",
        r#"{"synth":{"status":200,"body":"v1"}}"#,
    )
    .expect("plane");
    plane.start("leaf").await.expect("start");
    plane.mount("/leaf", "leaf").await.expect("mount");
    plane
}

#[proxima::test]
async fn hot_swap_fixture(live_plane: proxima::test_support::LivePlane) {
    let before = live_plane.call("/leaf").await.expect("call v1").payload;
    assert_eq!(&before[..], b"v1");

    live_plane
        .hot_swap("leaf", r#"{"synth":{"status":200,"body":"v2"}}"#)
        .await
        .expect("hot swap");

    let after = live_plane.call("/leaf").await.expect("call v2").payload;
    assert_eq!(&after[..], b"v2", "router reflects the hot-swapped spec");
}

// config-overlay: a base config deep-merged with each #[overlay_case] patch.
// Proves the base is preserved where the patch is silent (status stays 200).
#[proxima::test]
#[overlay_case(r#"{"synth":{"body":"patched-a"}}"#)]
#[overlay_case(r#"{"synth":{"body":"patched-b"}}"#)]
async fn config_overlay(
    #[overlay(r#"{"synth":{"status":200,"body":"base"}}"#)]
    cfg: proxima::test_support::ConfigFixture,
) {
    let body = cfg.value()["synth"]["body"].as_str().expect("body");
    assert!(
        body == "patched-a" || body == "patched-b",
        "patch applied: {body}"
    );
    assert_eq!(
        cfg.value()["synth"]["status"].as_u64(),
        Some(200),
        "base value preserved through the overlay merge"
    );
}

// cassette sugar: `#[cassette(inner)]` hands the body a record/replay PipeHandle
// directly (no manual TestCtx::cassette_pipe). The committed `sugar.jsonl` makes
// subsequent runs replay with no live dependency.
//
// needs a real runtime backend: the cassette machinery constructs one via
// `offline_runtime()`, which errors when neither the full prime bundle nor
// runtime-tokio is linked (see src/app.rs) — `test-support` alone isn't enough.
#[cfg(any(
    feature = "runtime-tokio",
    all(
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    )
))]
#[proxima::test(cassette = "sugar")]
async fn cassette_sugar(
    #[cassette(proxima::SynthUpstream::new("synth", 200, "sugar-ok"))] pipe: proxima::PipeHandle,
) {
    let request = proxima::Request::builder()
        .method("GET")
        .path("/v1/sugar")
        .build()
        .expect("request");
    let response = proxima_primitives::pipe::SendPipe::call(&pipe, request)
        .await
        .expect("call");
    assert_eq!(response.status, 200);
    let body = response.collect_body().await.expect("body");
    assert_eq!(&body[..], b"sugar-ok");
}

// needs a real runtime backend: see `cassette_sugar` above.
#[cfg(any(
    feature = "runtime-tokio",
    all(
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    )
))]
#[proxima::test(cassette = "weather")]
async fn weather_cassette_round_trips(cx: proxima::test_support::TestCtx) {
    let pipe = proxima::test_support::cassette_pipe(&cx, SmokeUpstream)
        .await
        .expect("cassette pipe");
    let request = proxima::Request::builder()
        .method("GET")
        .path("/v1/forecast")
        .query_param("city", "oslo")
        .build()
        .expect("request");
    let response = proxima_primitives::pipe::SendPipe::call(&pipe, request)
        .await
        .expect("cassette call");
    assert_eq!(response.status, 200);
    let body = response.collect_body().await.expect("body");
    assert_eq!(&body[..], b"{\"temp\":12}");
}
