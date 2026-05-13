#![allow(clippy::unwrap_used, clippy::expect_used)]

//! `fallback` — try an alternate pipe on failure.
//!
//! `retry` re-runs the SAME pipe against a fresh attempt. `Fallback` is its
//! sibling turned sideways: instead of trying `primary` again, it tries a
//! DIFFERENT pipe, `secondary`. `Fallback::call` (in
//! `proxima_primitives::pipe::resilience::fallback`) is exactly this:
//!
//! ```text
//! match primary.call(input.clone()).await {
//!     Ok(out) => Ok(out),           // secondary never runs
//!     Err(_)  => secondary.call(input).await,
//! }
//! ```
//!
//! `primary` sees a clone of the input (`P::In: Clone` is required so the
//! original can be replayed) and, on any error, `secondary` gets the exact
//! same input. On success, `secondary` is skipped entirely — not called with
//! a cheap no-op, just never invoked.
//!
//! Two scenarios below share one `secondary` shape (a cache with an atomic
//! hit counter) and only swap `primary`'s health:
//!
//! 1. a healthy primary — the live answer wins, the cache counter stays 0.
//! 2. a failing primary — the cache serves a degraded-but-present answer,
//!    and the counter proves it was actually called.
//!
//! Run: `cargo run --example fallback`

use core::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use proxima_primitives::block_on;
use proxima_primitives::pipe::{Fallback, Pipe};

fn main() {
    println!("fallback: try an alternate pipe on failure\n");

    println!("-- primary healthy: live answer wins, cache untouched --");
    let cache_hits = Arc::new(AtomicU32::new(0));
    let healthy = Fallback {
        primary: LiveService { healthy: true },
        secondary: Cache {
            hits: Arc::clone(&cache_hits),
        },
    };
    let response = block_on(Pipe::call(&healthy, Query { id: 7 })).expect("healthy fallback");
    assert_eq!(
        response,
        Response {
            source: Source::Live,
            value: 70
        },
        "primary's own answer"
    );
    assert_eq!(
        cache_hits.load(Ordering::SeqCst),
        0,
        "secondary was never called"
    );
    println!("  query {{ id: 7 }} -> {response:?}");
    println!(
        "  cache hits: {} (secondary untouched)\n",
        cache_hits.load(Ordering::SeqCst)
    );

    println!("-- primary down: cache serves a degraded answer --");
    let cache_hits = Arc::new(AtomicU32::new(0));
    let degraded = Fallback {
        primary: LiveService { healthy: false },
        secondary: Cache {
            hits: Arc::clone(&cache_hits),
        },
    };
    let response = block_on(Pipe::call(&degraded, Query { id: 7 })).expect("degraded fallback");
    assert_eq!(
        response,
        Response {
            source: Source::Cache,
            value: 7
        },
        "cache's own answer"
    );
    assert_eq!(
        cache_hits.load(Ordering::SeqCst),
        1,
        "secondary served exactly once"
    );
    println!("  query {{ id: 7 }} -> {response:?}");
    println!(
        "  cache hits: {} (secondary served)\n",
        cache_hits.load(Ordering::SeqCst)
    );

    println!(
        "same Fallback wiring both times; only LiveService's health changed \
         which pipe answered the query"
    );
}

// ── the two pipes Fallback composes ─────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct Query {
    id: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    Live,
    Cache,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Response {
    source: Source,
    value: u32,
}

/// The primary: a live upstream that either answers or is down. `healthy`
/// stands in for whatever a real primary would fail on — a timeout, a
/// connection refusal, a 5xx.
struct LiveService {
    healthy: bool,
}

impl Pipe for LiveService {
    type In = Query;
    type Out = Response;
    type Err = &'static str;

    fn call(&self, query: Query) -> impl Future<Output = Result<Response, &'static str>> {
        let healthy = self.healthy;
        async move {
            if healthy {
                Ok(Response {
                    source: Source::Live,
                    value: query.id * 10,
                })
            } else {
                Err("upstream unavailable")
            }
        }
    }
}

/// The secondary: a cache that always answers, never fails. `Err` matches
/// `LiveService::Err` because `Fallback` requires both pipes to share one
/// error type — `Cache` just never constructs it. `hits` counts calls so the
/// example can prove it was skipped when the primary succeeded.
struct Cache {
    hits: Arc<AtomicU32>,
}

impl Pipe for Cache {
    type In = Query;
    type Out = Response;
    type Err = &'static str;

    fn call(&self, query: Query) -> impl Future<Output = Result<Response, &'static str>> {
        self.hits.fetch_add(1, Ordering::SeqCst);
        async move {
            Ok(Response {
                source: Source::Cache,
                value: query.id,
            })
        }
    }
}
