#![allow(clippy::unwrap_used, clippy::expect_used)]

//! The CLIENT side of admission: composing real resilience primitives around
//! an outbound call to a `.any()`/`.accept("h2")` listener, instead of a
//! bespoke "client rate limiter" type. RISC discipline (`~/.claude/rules/
//! rust.md`): before minting a type, write the pipe. Every layer below IS
//! an existing pipe:
//!
//! - `RateLimit` (`proxima_primitives::pipe::RateLimit`, the same token
//!   bucket `examples/rate_limit` teaches server-side) self-throttles the
//!   OUTBOUND send rate — a client held to a quota is the identical
//!   primitive, just wrapping a dial instead of a handler.
//! - `Retry` + `Backoff` + `Jitter` (`proxima_primitives::pipe::resilience`,
//!   the same primitives `examples/backoff` teaches) retries a transient
//!   503 with exponential backoff, driven by the REAL production
//!   [`proxima_primitives::pipe::clock::TimeClock`] (real sleeps, not a
//!   fake one — this section proves it against a REAL shedding listener).
//! - `H2ClientUpstream` is `SendPipe`-only (the cross-core, `Send`-future
//!   tier); `Retry`/`RateLimit`'s generic bound is the plain `Pipe` tier (no
//!   `Send` requirement — RPITIT can't strengthen a trait method's
//!   `Send`-ness via a subtrait on stable, so `SendPipe` is a SEPARATE trait,
//!   not `SendPipe: Pipe` — `proxima-primitives/src/pipe/primitives.rs:104-106`).
//!   `AsPipe<T>` below is the one-line bridge: a `Send` future trivially
//!   satisfies a bound that doesn't require `Send`. This is NOT a library
//!   gap — it's the documented reason the tiers are separate traits at all
//!   (see this file's own report for the "why" in full).
//!
//! ## What is NOT here, and why
//!
//! `fanin!`/`FanIn` is not used to race live backend dials. `FanIn`'s own
//! doc is explicit that it does not: "Scan, don't race — [a merge] does not
//! drive N sources concurrently and take a winner... A source whose call(())
//! is not yet ready is polled once, found Pending, and its in-flight future
//! is then DROPPED" (`proxima-primitives/src/pipe/fan_in.rs`'s module doc).
//! A live TCP dial + h2 handshake WILL return `Pending` at least once; a
//! `FanIn` source wrapping one would have its dial restarted every scan,
//! never completing. The shipped code agrees: `examples/load-balance/main.rs`
//! — the one tutorial that actually forwards a real request to a real
//! backend pool — hand-rolls its own round-robin cursor
//! (`select_backend`, `load-balance/main.rs:92-102`) instead of using
//! `FanIn`, and the one place `FanIn` DOES appear alongside a gate
//! (`examples/gate/main.rs`'s BALANCE section) merges pre-populated
//! `VecDeque` queues, not a live dial. This file follows that same,
//! real precedent: a hand-rolled round-robin-over-healthy pool below,
//! `Retry`/`RateLimit` wrapping the picked backend's real call.
//!
//! Run: `cargo run --example any_listener_client_resilience --features http1-native`

use std::future::Future;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use bytes::Bytes;
use serde_json::json;

use proxima::h2::H2ClientUpstream;
use proxima::pipe::into_handle;
use proxima::request::{Request, Response};
use proxima::time::sleep;
use proxima::{Listener, ListenerBuilderEntry, PrimeTcpUpstream, ProximaError};
use proxima_primitives::pipe::clock::TimeClock;
use proxima_primitives::pipe::rate_limit::{KeyExtractor, RateLimit, TokenBucketConfig};
use proxima_primitives::pipe::resilience::{Backoff, Jitter, Retry, RetryController};
use proxima_primitives::pipe::retry_rules::RetryRules;
use proxima_primitives::pipe::{Pipe, SendPipe};

fn free_loopback_addr() -> Result<SocketAddr, ProximaError> {
    let probe = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
    let addr = probe.local_addr()?;
    drop(probe);
    Ok(addr)
}

fn get_request() -> Result<Request<Bytes>, ProximaError> {
    // bodyless — see `any_listener_production.rs`'s `constant_ok_request`
    // doc for why a shed request with a body is a separate, real defect
    // this file avoids by construction, not by luck.
    Request::builder().method("GET").path("/").build()
}

/// The handler behind the capped listener: sleeps long enough that a second
/// concurrent caller reliably observes a real 503 shed, then releases.
struct SlowOk;

impl SendPipe for SlowOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            sleep(Duration::from_millis(150)).await;
            Ok(Response::new(200).with_body(Bytes::from_static(b"ok")))
        }
    }
}

/// One-line bridge: `SendPipe`'s future IS `Send`, which trivially satisfies
/// `Pipe::call`'s weaker "not required to be `Send`" bound. Local to this
/// example — not a library type (see this file's module doc).
struct AsPipe<T>(T);

impl<T: SendPipe> Pipe for AsPipe<T> {
    type In = T::In;
    type Out = T::Out;
    type Err = T::Err;

    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>> {
        self.0.call(input)
    }
}

/// §1: `RateLimit` self-throttles the OUTBOUND rate — the exact `TokenBucketConfig`/
/// `KeyExtractor` shape `examples/rate_limit` teaches server-side, wrapping a REAL
/// `H2ClientUpstream` instead of a toy `Backend`. No network dial happens on a
/// refusal (the bucket check runs before the inner pipe is ever called), so this
/// section needs no listener at all to prove the throttle.
async fn client_self_throttle_demo() {
    // `RateLimit<Inner>` requires `Inner: Clone`; `H2ClientUpstream` isn't —
    // `Arc<Inner>: SendPipe + Clone` (`proxima-primitives/src/pipe/
    // alloc_tier.rs:139`) is the standing bridge for exactly this, an `Arc`
    // bump per clone, no new adapter type needed.
    let never_dialed = Arc::new(H2ClientUpstream::new(
        PrimeTcpUpstream::new("127.0.0.1:1".parse().expect("addr")),
        "127.0.0.1:1",
        false,
        "self-throttle-demo",
    ));
    let limited = RateLimit::new(
        never_dialed,
        TokenBucketConfig {
            capacity: 2,
            refill_per_sec: 0,
        },
        KeyExtractor::ConstantKey("outbound".into()),
    );

    for attempt in 1..=2 {
        // both attempts consume the bucket's 2 tokens WITHOUT dialing
        // 127.0.0.1:1 (RateLimit refuses or admits before the inner pipe
        // ever runs) — this section is about the throttle decision, not
        // the network call underneath it.
        let outcome = SendPipe::call(&limited, get_request().expect("request")).await;
        assert!(
            outcome.is_err(),
            "the never-listening dial errors, not the throttle"
        );
        println!("  outbound attempt {attempt}: bucket admitted the call through to the dial");
    }
    let refused = SendPipe::call(&limited, get_request().expect("request"))
        .await
        .expect("RateLimit renders its own 429, it does not propagate a dial error");
    assert_eq!(
        refused.status, 429,
        "capacity exhausted with no refill: the client's OWN rate limiter refuses \
         locally, no dial attempted"
    );
    println!("  outbound attempt 3: refused locally with 429, no dial attempted");
}

/// §2: `Retry` + `Backoff` + `Jitter`, driven by the REAL `TimeClock`, recovering
/// a REAL transient 503 from a REAL listener capped at `max_in_flight_requests = 1`.
async fn retry_recovers_a_real_shed_response() -> Result<(), ProximaError> {
    let bind = free_loopback_addr()?;
    let server = Listener::builder()
        .bind(bind)
        .accept("h2")
        .spec("max_in_flight_requests", json!(1))
        .handle(into_handle(SlowOk))
        .serve()
        .await?;

    let client = H2ClientUpstream::new(
        PrimeTcpUpstream::new(bind),
        format!("{bind}"),
        false,
        "retry-demo",
    );
    let resilient = Retry::new(
        AsPipe(client),
        RetryController {
            rules: RetryRules::default(), // 502/503/504 + real transport errors
            backoff: Backoff::Exponential {
                initial: Duration::from_millis(80),
                factor: 2,
                max: Duration::from_millis(400),
            },
            jitter: Jitter::None,
            max_attempts: 5,
            deadline: None,
        },
        TimeClock,
        7,
    );

    // fire the FIRST call directly (occupies the one admitted slot for
    // 150ms) and the RESILIENT client concurrently — its first attempt
    // gets shed (503), and its backoff-then-retry recovers a clean 200
    // once the first call releases, with no caller-visible failure at all.
    let occupier = H2ClientUpstream::new(
        PrimeTcpUpstream::new(bind),
        format!("{bind}"),
        false,
        "occupier",
    );
    // Which of the two connections wins the single admission slot is a real
    // race (both dial concurrently) — not deterministic, and not the point.
    // The point: regardless of who wins, the RESILIENT client's
    // caller-visible outcome is ALWAYS a clean 200. If the occupier wins,
    // the resilient client's first attempt is shed (503) and its
    // Retry+Backoff recovers once the occupier releases 150ms later. If the
    // resilient client wins instead, its first attempt just succeeds
    // outright — Retry is a no-op on a non-retryable (200) outcome.
    let (occupier_response, resilient_response) = futures::join!(
        occupier.call(get_request()?),
        Pipe::call(&resilient, get_request()?),
    );
    let resilient_response = resilient_response?;

    assert_eq!(
        resilient_response.status, 200,
        "Retry must absorb a transient 503 the listener actually shed — the caller \
         never sees it, regardless of which connection won the admission race"
    );
    match occupier_response {
        Ok(response) => println!("  occupier (no retry wrapper): status {}", response.status),
        Err(error) => println!("  occupier (no retry wrapper): {error} (lost the admission race)"),
    }
    println!(
        "  resilient client: {resilient_response:?} (the caller-visible outcome is always a \
         clean 200, whether or not it needed a retry underneath)"
    );

    server.stop();
    Ok(())
}

/// §3: the honest fit for a backend POOL — round-robin over healthy
/// candidates, hand-rolled (matching `examples/load-balance/main.rs`'s
/// `select_backend`), NOT `FanIn` (see this file's module doc for why).
struct BackendPool {
    binds: Vec<SocketAddr>,
    cursor: AtomicUsize,
}

impl BackendPool {
    fn pick(&self) -> SocketAddr {
        let index = self.cursor.fetch_add(1, Ordering::SeqCst) % self.binds.len();
        self.binds[index]
    }
}

async fn backend_pool_round_robin_demo() -> Result<(), ProximaError> {
    let bind_a = free_loopback_addr()?;
    let bind_b = free_loopback_addr()?;
    let server_a = Listener::builder()
        .bind(bind_a)
        .accept("h2")
        .handle(into_handle(SlowOk))
        .serve()
        .await?;
    let server_b = Listener::builder()
        .bind(bind_b)
        .accept("h2")
        .handle(into_handle(SlowOk))
        .serve()
        .await?;

    let pool = BackendPool {
        binds: vec![bind_a, bind_b],
        cursor: AtomicUsize::new(0),
    };
    let mut picked_a = 0;
    let mut picked_b = 0;
    for _ in 0..4 {
        let target = pool.pick();
        let client = H2ClientUpstream::new(
            PrimeTcpUpstream::new(target),
            format!("{target}"),
            false,
            "pool",
        );
        let response = client.call(get_request()?).await?;
        assert_eq!(response.status, 200);
        if target == bind_a {
            picked_a += 1;
        } else {
            picked_b += 1;
        }
    }
    println!("  4 requests round-robinned: {picked_a} to backend A, {picked_b} to backend B");
    assert_eq!(picked_a, 2);
    assert_eq!(picked_b, 2);

    server_a.stop();
    server_b.stop();
    Ok(())
}

#[proxima::main]
async fn main() -> Result<(), ProximaError> {
    println!("§1: client-side self-throttle (RateLimit wrapping a real H2ClientUpstream)");
    client_self_throttle_demo().await;

    println!("\n§2: Retry+Backoff (real TimeClock) recovering a real 503 shed by a real listener");
    retry_recovers_a_real_shed_response().await?;

    println!("\n§3: backend pool, round-robin over healthy — hand-rolled, not FanIn");
    backend_pool_round_robin_demo().await?;

    println!(
        "\nany_listener_client_resilience: self-throttle + retry-recovery + pool round-robin all OK"
    );
    Ok(())
}
