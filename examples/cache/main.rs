#![allow(clippy::unwrap_used, clippy::expect_used)]

//! `cache` — fallthrough + write-back: a cache in front of an origin.
//!
//! There is no single `Cache` primitive in proxima. What the `cached`
//! scenario (`scenarios/cached/scenario.toml`) calls "a cache in front of an
//! origin" is three real primitives wired together, the same way
//! `src/load.rs::build_composed` wires any multi-upstream pipe:
//!
//! - two upstreams behind one `Selection` — a `kv:cache` lookup
//!   ([`KvUpstream`] over [`KvCache`]) and a `synth` origin
//! - [`Fallthrough`] selection, `miss_on = [no_data]` — try the cache first;
//!   a cache miss (`ProximaError::NoData`) falls through to the origin
//! - [`WriteBack`] wrapping the whole dispatch — after ANY response, tap the
//!   body and `put` it into the cache backend, so a miss this request
//!   becomes a hit next request
//!
//! `filter` taught the gate (a decision pipe's `Err` short-circuits the inner
//! pipe via `AndThen`). `transform` taught the plain `In -> Out` pipe. This
//! example composes both ideas at the upstream-selection layer instead of
//! the middleware layer.
//!
//! Run: `cargo run --example cache`

use core::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use bytes::Bytes;

use proxima::selection::Selection;
use proxima::upstreams::KvUpstream;
use proxima::{
    Fallthrough, KvCache, KvCaps, KvHandle, ProximaError, Request, Response, SendPipe,
    SynthUpstream, UpstreamRef, WriteBack, into_handle,
};

#[tokio::main]
async fn main() {
    println!("cache: fallthrough + write-back (cache in front of an origin)\n");

    let origin_calls = Arc::new(AtomicUsize::new(0));

    let cache_backend =
        KvCache::new("cache", None, KvCaps::entries(1024)).expect("kv cache backend");
    let cache_upstream = into_handle(KvUpstream::new(cache_backend.clone()));

    let origin_upstream = into_handle(CountingOrigin {
        inner: Arc::new(SynthUpstream::new(
            "origin",
            200,
            r#"{"id":"chatcmpl-fake","object":"chat.completion","choices":[]}"#,
        )),
        calls: origin_calls.clone(),
    });

    let upstreams = Arc::new(vec![
        UpstreamRef::new(cache_upstream, "cache", 1),
        UpstreamRef::new(origin_upstream, "origin", 1),
    ]);
    let dispatch = CachedOriginDispatch {
        upstreams,
        selection: Arc::new(Fallthrough::miss_on_no_data()),
    };

    let write_back_target: Arc<dyn KvHandle> = cache_backend.clone();
    let cached_origin = WriteBack::single(into_handle(dispatch), write_back_target);

    let request = || {
        Request::builder()
            .method("GET")
            .path("/v1/chat/completions")
            .build()
            .expect("request builder")
    };

    println!("-- request 1: cache empty, falls through to origin, write-back populates cache --");
    let first = SendPipe::call(&cached_origin, request())
        .await
        .expect("first call");
    let first_status = first.status;
    let first_hit_header = first
        .metadata
        .get_str("x-proxima-cache")
        .map(str::to_string);
    let first_body = first.collect_body().await.expect("collect first body");
    assert_eq!(
        origin_calls.load(Ordering::Relaxed),
        1,
        "the miss must fall through to the origin"
    );
    assert_eq!(
        first_hit_header, None,
        "the first response comes straight from the origin, not the cache"
    );
    assert_eq!(
        cache_backend.entries(),
        1,
        "write-back must populate the cache after the origin answers"
    );
    println!(
        "  status={first_status} cache-header={first_hit_header:?} body={}",
        String::from_utf8_lossy(&first_body)
    );
    println!(
        "  origin calls so far: {}\n",
        origin_calls.load(Ordering::Relaxed)
    );

    println!("-- requests 2..6: cache hits, origin never called again --");
    for attempt in 2..=6 {
        let response = SendPipe::call(&cached_origin, request())
            .await
            .expect("cached call");
        let status = response.status;
        let hit_header = response
            .metadata
            .get_str("x-proxima-cache")
            .map(str::to_string);
        let body = response.collect_body().await.expect("collect body");
        assert_eq!(status, 200);
        assert_eq!(
            hit_header.as_deref(),
            Some("HIT"),
            "request {attempt} must be served from the cache, not the origin"
        );
        assert_eq!(
            &body[..],
            &first_body[..],
            "the cache must serve the exact body the origin produced on the miss"
        );
        println!("  request {attempt}: status={status} cache-header={hit_header:?}");
    }

    let total_origin_calls = origin_calls.load(Ordering::Relaxed);
    assert_eq!(
        total_origin_calls, 1,
        "origin must be called exactly once across all 6 requests: fallthrough hit the cache for the rest"
    );
    assert_eq!(
        cache_backend.entries(),
        1,
        "the cache still holds exactly the one write-back entry"
    );

    println!(
        "\norigin called {total_origin_calls} time(s) across 6 requests; \
         the remaining 5 were served straight from the cache"
    );
}

// ── the dispatch: Fallthrough selection over [cache, origin] ───────────────
//
// This is the same shape as `DispatchPipe` in `src/load.rs` (private to that
// module): a thin `Pipe` whose whole job is `Selection::dispatch` over a
// fixed upstream list. There is no exported "cache in front of an origin"
// combinator — this wiring, plus `WriteBack` around it, IS the composition.

#[derive(Clone)]
struct CachedOriginDispatch {
    upstreams: Arc<Vec<UpstreamRef>>,
    selection: Arc<Fallthrough>,
}

impl SendPipe for CachedOriginDispatch {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let upstreams = self.upstreams.clone();
        let selection = self.selection.clone();
        async move {
            let outcome = Selection::dispatch(selection.as_ref(), &upstreams, request).await?;
            Ok(outcome.response)
        }
    }
}


// ── the origin: a synth upstream with a call counter ────────────────────────
//
// SynthUpstream itself doesn't count calls; this wrapper is the same
// instrumentation trick `fallback`'s `Cache` and `filter`'s `Ledger` use — a
// plain delegate that bumps an `AtomicUsize` so the example can assert on
// real call counts, not just eyeball the printed lines.

#[derive(Clone)]
struct CountingOrigin {
    inner: Arc<SynthUpstream>,
    calls: Arc<AtomicUsize>,
}

impl SendPipe for CountingOrigin {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let inner = self.inner.clone();
        async move { SendPipe::call(inner.as_ref(), request).await }
    }
}

