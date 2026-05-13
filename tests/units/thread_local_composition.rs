//! Stage 10 closure test: prove the per-thread substrate composes
//! middlewares around an inner `Pipe` impl that holds `!Send` state
//! (`Rc`/`RefCell`). The composed chain runs end-to-end on a `LocalSet`
//! and exercises the fork pattern across the wrapper ecosystem.

// drives the composed chain on a `tokio::task::LocalSet` explicitly — needs
// `tokio`.
#![cfg(feature = "tokio")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::future::Future;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use proxima::causality::{Causal, CausalIndex};
use proxima::error::ProximaError;
use proxima::middlewares::auth::Auth;
use proxima::middlewares::isolate::Isolate;
use proxima::middlewares::rate_limit::{KeyExtractor, RateLimit, TokenBucketConfig};
use proxima::middlewares::retry::{Retry, RetryPredicate};
use proxima::middlewares::transform::{RequestOp, ResponseOp, Transform};
use proxima::pipe::{ThreadLocalPipeHandle, into_thread_local_handle};
use proxima::request::{Request, Response};
use proxima::RoutingPipe;
use proxima_primitives::pipe::Pipe;

// Counts requests in `Rc<RefCell<u64>>` — deliberately `!Send` so it
// can only be held by a per-thread Pipe. Proves the fork actually
// lets per-core authors keep `!Send` state in the impl.
struct LocalCounter {
    label: &'static str,
    hits: Rc<RefCell<u64>>,
}

impl Pipe for LocalCounter {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let hits = self.hits.clone();
        let label = self.label;
        async move {
            *hits.borrow_mut() += 1;
            Ok(Response::ok(Bytes::from_static(label.as_bytes())))
        }
    }
}


fn request_with_auth(token: &str, path: &str) -> Request<Bytes> {
    Request::builder()
        .method("GET")
        .path(path)
        .header("authorization", format!("Bearer {token}"))
        .build()
        .expect("builder")
}

#[proxima::test(runtime = "tokio")]
async fn auth_isolate_rate_limit_retry_transform_around_local_pipe() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async move {
            let hits = Rc::new(RefCell::new(0u64));
            let leaf: ThreadLocalPipeHandle = into_thread_local_handle(LocalCounter {
                label: "counted",
                hits: hits.clone(),
            });

            // Inside out: leaf <- transform <- retry <- rate_limit <- isolate <- auth.
            let transformed: ThreadLocalPipeHandle = into_thread_local_handle(
                Transform::<ThreadLocalPipeHandle>::new(leaf)
                    .with_request_op(RequestOp::SetHeader {
                        name: "x-trace".into(),
                        value: "tlc".into(),
                    })
                    .with_response_op(ResponseOp::SetHeader {
                        name: "x-served".into(),
                        value: "tlc".into(),
                    }),
            );

            let retried: ThreadLocalPipeHandle = into_thread_local_handle(
                Retry::<ThreadLocalPipeHandle>::new(transformed)
                    .with_max_attempts(2)
                    .with_base_delay(Duration::ZERO)
                    .with_max_delay(Duration::ZERO)
                    .with_predicate(RetryPredicate::OnAnyError),
            );

            let limited: ThreadLocalPipeHandle =
                into_thread_local_handle(RateLimit::<ThreadLocalPipeHandle>::new(
                    retried,
                    TokenBucketConfig {
                        capacity: 10,
                        refill_per_sec: 10,
                    },
                    KeyExtractor::PathAndMethod,
                ));

            let isolated: ThreadLocalPipeHandle = into_thread_local_handle(
                Isolate::<ThreadLocalPipeHandle>::new(limited)
                    .with_timeout(Duration::from_millis(500))
                    .with_panic_barrier(true),
            );

            let mut allow: BTreeSet<String> = BTreeSet::new();
            allow.insert("secret".into());
            let auth = Auth::<ThreadLocalPipeHandle> {
                inner: isolated,
                header: "authorization".into(),
                allow,
                realm: Arc::from(b"proxima".as_slice()),
                on_unauthorized_status: 401,
                strip_prefix: Some("Bearer ".into()),
            };
            let chain: ThreadLocalPipeHandle = into_thread_local_handle(auth);

            // admitted path: token matches, inner increments counter, response
            // carries the transform-injected header.
            let response = Pipe::call(&chain, request_with_auth("secret", "/x"))
                .await
                .expect("admitted call");
            assert_eq!(response.status, 200);
            let served = response
                .metadata
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(b"x-served"))
                .expect("x-served header");
            assert_eq!(served.1.as_ref(), b"tlc");
            let body = response.collect_body().await.expect("collect");
            assert_eq!(&body[..], b"counted");
            assert_eq!(*hits.borrow(), 1);

            // rejected path: bad token short-circuits at the auth wrapper, leaf
            // not invoked, counter unchanged.
            let response = Pipe::call(&chain, request_with_auth("nope", "/x"))
                .await
                .expect("rejected call");
            assert_eq!(response.status, 401);
            assert_eq!(*hits.borrow(), 1);
        })
        .await;
}

#[proxima::test(runtime = "tokio")]
async fn routing_pipe_dispatches_to_local_inners() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async move {
            let users_hits = Rc::new(RefCell::new(0u64));
            let posts_hits = Rc::new(RefCell::new(0u64));
            let users: ThreadLocalPipeHandle = into_thread_local_handle(LocalCounter {
                label: "users",
                hits: users_hits.clone(),
            });
            let posts: ThreadLocalPipeHandle = into_thread_local_handle(LocalCounter {
                label: "posts",
                hits: posts_hits.clone(),
            });

            let router: RoutingPipe<ThreadLocalPipeHandle> =
                RoutingPipe::<ThreadLocalPipeHandle>::new("api")
                    .route("/users/{id}", users)
                    .route("/posts/{id}", posts);

            let users_req = Request::builder()
                .method("GET")
                .path("/users/42")
                .build()
                .expect("builder");
            let response = Pipe::call(&router, users_req).await.expect("call");
            let body = response.collect_body().await.expect("collect");
            assert_eq!(&body[..], b"users");

            let posts_req = Request::builder()
                .method("GET")
                .path("/posts/9")
                .build()
                .expect("builder");
            let response = Pipe::call(&router, posts_req).await.expect("call");
            let body = response.collect_body().await.expect("collect");
            assert_eq!(&body[..], b"posts");

            assert_eq!(*users_hits.borrow(), 1);
            assert_eq!(*posts_hits.borrow(), 1);
        })
        .await;
}

#[proxima::test(runtime = "tokio")]
async fn causal_wrapper_records_around_local_inner() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async move {
            let hits = Rc::new(RefCell::new(0u64));
            let leaf: ThreadLocalPipeHandle = into_thread_local_handle(LocalCounter {
                label: "leaf",
                hits: hits.clone(),
            });
            let index = CausalIndex::new();
            let recorder = Causal::<ThreadLocalPipeHandle>::new(leaf, "leaf-node", index.clone());
            let recorder_handle: ThreadLocalPipeHandle = into_thread_local_handle(recorder);

            let request = Request::builder()
                .method("GET")
                .path("/edge")
                .body("payload")
                .build()
                .expect("builder");
            let response = Pipe::call(&recorder_handle, request)
                .await
                .expect("call");
            assert_eq!(response.status, 200);
            let body = response.collect_body().await.expect("collect");
            assert_eq!(&body[..], b"leaf");
            assert_eq!(*hits.borrow(), 1);

            // exactly one edge recorded for this leaf node.
            let edges = index.edges();
            let leaf_edges: Vec<_> = edges
                .iter()
                .filter(|edge| edge.node_id == "leaf-node")
                .collect();
            assert_eq!(leaf_edges.len(), 1, "one causal edge per call");
        })
        .await;
}
