//! [`Handler`] is the served-HTTP face of the [`SendPipe`] root form: a
//! `Request<Bytes> -> Response<Bytes>` pipe with `Err` pinned to
//! [`ProximaError`]. Every upstream, every middleware, every composition
//! primitive that dispatches HTTP traffic implements it. Listeners dispatch
//! into a [`PipeHandle`]; pipelines record from one; the daemon hot-swaps
//! them.
//!
//! # Handler is the served boundary
//!
//! ```text
//!        request                                response
//!           │                                      ▲
//!           ▼                                      │
//!        ┌──────────────────────────────────────────┐
//!        │                Handler                   │
//!        │  async fn call(Request) -> Result<...>   │
//!        └──────────────────────────────────────────┘
//! ```
//!
//! `Handler` is a blanket impl over
//! `SendPipe<In = Request<Bytes>, Out = Response<Bytes>, Err = ProximaError>` —
//! nothing to implement, any qualifying `SendPipe` already satisfies it.
//! [`PipeHandle`] is the runtime-dispatch alias for the erased form so
//! heterogeneous handlers can sit side by side. [`ThreadLocalHandler`] /
//! [`ThreadLocalPipeHandle`] are the `!Send` sibling for `Rc` / `RefCell`
//! per-core state.
//!
//! # Substrate primitives
//!
//! These are also `Handler`s — they wrap any inner handler and don't care
//! what it does. [`Diff`](crate::pipe::Diff) runs two handlers in parallel and
//! reports divergence; [`Isolate`](crate::pipe::Isolate) catches panics and
//! enforces a time budget; [`SwappablePipe`](crate::pipe::SwappablePipe) hot-swaps
//! the inner handle at runtime without tearing in-flight calls; the daemon
//! control plane's `apply` verb rides on it.
//!
//! # Recording wraps any Pipe
//!
//! Recording is not a separate tracing subsystem — it composes from the
//! same primitives. `Tee`-shaped wrapping (record a copy of the
//! request/response while passing through) lives downstream in
//! `proxima-recording`, built over this crate's [`Handler`] boundary the
//! same way every other composition primitive is.
//!
//! # Serving a Handler
//!
//! A `Handler` doesn't serve traffic by itself — a
//! [`ListenProtocol`](crate::pipe::ListenProtocol) does. The fluent builder
//! [`ServeBuilder`](crate::pipe::ServeBuilder) wires a handler into a listener:
//!
//! ```ignore
//! use proxima::{HttpListener, ListenProtocolFluent};
//! HttpListener::http("0.0.0.0:8080".parse()?)
//!     .fluent()
//!     .dispatch(handle)
//!     .await?;
//! ```

// every item here touches Box/Rc/Arc unconditionally via alloc_tier;
// there is no alloc-free subset.
#![cfg(feature = "alloc")]

use bytes::Bytes;
use proxima_core::ProximaError;

use crate::pipe::primitives::Pipe;
use crate::pipe::request::{Request, Response};
use crate::pipe::{SendPipe, alloc_tier};

/// The served-HTTP face of [`SendPipe`]: `Request<Bytes> -> Response<Bytes>`,
/// `Err` pinned to [`ProximaError`]. Blanket-implemented for every qualifying
/// `SendPipe` — nothing to implement directly.
pub trait Handler: SendPipe<In = Request<Bytes>, Out = Response<Bytes>, Err = ProximaError> {}

impl<Implementor> Handler for Implementor where
    Implementor: SendPipe<In = Request<Bytes>, Out = Response<Bytes>, Err = ProximaError>
{
}

/// Erased, shareable served handler — the runtime-dispatch alias `Arc<dyn
/// SendDynPipe<Request<Bytes>, Response<Bytes>>>` so heterogeneous handlers
/// sit side by side in routing tables and swap cells.
pub type PipeHandle = alloc_tier::PipeHandle<Request<Bytes>, Response<Bytes>>;

/// Erase a [`Handler`] into a shareable [`PipeHandle`].
pub fn into_handle<Implementor>(pipe: Implementor) -> PipeHandle
where
    Implementor: SendPipe<In = Request<Bytes>, Out = Response<Bytes>, Err = ProximaError> + 'static,
{
    alloc_tier::into_handle(pipe)
}

/// Per-thread sibling of [`Handler`] for runtimes that pin work to a single
/// core (DPDK, per-core executors) and may hold `Rc` / `RefCell` state.
/// Blanket-implemented over the no-Send root [`Pipe`].
pub trait ThreadLocalHandler: Pipe<In = Request<Bytes>, Out = Response<Bytes>, Err = ProximaError> {}

impl<Implementor> ThreadLocalHandler for Implementor where
    Implementor: Pipe<In = Request<Bytes>, Out = Response<Bytes>, Err = ProximaError>
{
}

/// Erased, shareable per-thread handler — the `Rc`-backed sibling of
/// [`PipeHandle`].
pub type ThreadLocalPipeHandle = alloc_tier::LocalPipeHandle<Request<Bytes>, Response<Bytes>>;

/// Erase a [`ThreadLocalHandler`] into a shareable [`ThreadLocalPipeHandle`].
pub fn into_thread_local_handle<Implementor>(pipe: Implementor) -> ThreadLocalPipeHandle
where
    Implementor: Pipe<In = Request<Bytes>, Out = Response<Bytes>, Err = ProximaError> + 'static,
{
    alloc_tier::into_local_handle(pipe)
}

// `#[proxima::test]` and inline `tokio::spawn` pull in the `proxima` /
// `tokio` dev-dependencies, which the loom build keeps out of the graph
// (see `[target.'cfg(not(loom))'.dev-dependencies]` in Cargo.toml); these
// tests are unrelated to the Notify/watch loom protocol.
#[cfg(all(test, not(loom)))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use core::future::Future;

    struct EchoPipe {
        label: alloc::string::String,
    }

    impl SendPipe for EchoPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let label = self.label.clone();
            async move {
                let (_, bytes) = request.body_bytes().await?;
                Ok(Response::ok(bytes).with_header("x-pipe", label))
            }
        }
    }

    impl Pipe for EchoPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
            SendPipe::call(self, request)
        }
    }

    #[proxima::test]
    async fn dyn_handle_dispatches_to_inner() {
        let inner: PipeHandle = into_handle(EchoPipe {
            label: "echo".into(),
        });
        let request = Request::builder()
            .method("POST")
            .path("/")
            .body("ping")
            .build()
            .expect("builder");
        let response = SendPipe::call(&inner, request).await.expect("call");
        let body = response.payload;
        assert_eq!(&body[..], b"ping");
    }

    // proves the Handler::call future is Send: tokio::spawn requires Send
    // and the test would fail to compile otherwise. guards against an
    // accidental ?Send weakening of the trait surface.
    #[proxima::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pipe_call_future_is_send_for_tokio_spawn() {
        let handle: PipeHandle = into_handle(EchoPipe {
            label: "spawnable".into(),
        });
        let request = Request::builder()
            .method("POST")
            .path("/")
            .body("ok")
            .build()
            .expect("builder");
        let join = tokio::spawn(async move { SendPipe::call(&handle, request).await });
        let response = join.await.expect("join").expect("call");
        let body = response.payload;
        assert_eq!(&body[..], b"ok");
    }

    use alloc::rc::Rc as TestRc;
    use core::cell::RefCell;

    struct CounterPipe {
        // Rc<RefCell<_>> deliberately disqualifies this from Handler: not Send.
        count: TestRc<RefCell<usize>>,
    }

    impl Pipe for CounterPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
            let count = self.count.clone();
            async move {
                *count.borrow_mut() += 1;
                Ok(Response::ok("counted"))
            }
        }
    }

    // `Rc<RefCell<_>>` state is deliberately `!Send`, so this cannot ride
    // the `#[proxima::test]` harness (its `run` bounds the body `Send` under
    // the prime driver) — a plain synchronous executor sidesteps that bound
    // without pulling tokio in.
    #[test]
    fn thread_local_pipe_runs_with_rc_state() {
        futures::executor::block_on(async {
            let count: TestRc<RefCell<usize>> = TestRc::new(RefCell::new(0));
            let pipe = CounterPipe {
                count: count.clone(),
            };
            let handle: ThreadLocalPipeHandle = into_thread_local_handle(pipe);
            for _ in 0..3 {
                let request = Request::builder()
                    .method("GET")
                    .path("/")
                    .build()
                    .expect("builder");
                let _ = Pipe::call(&handle, request).await.expect("call");
            }
            assert_eq!(*count.borrow(), 3);
        });
    }

    #[proxima::test]
    async fn blanket_lifts_a_send_pipe_into_thread_local() {
        // EchoPipe impls SendPipe with the served In/Out/Err shape; the
        // blanket auto-derives Handler. Dispatching a Handler-erased handle
        // must call into the same impl.
        let pipe = EchoPipe {
            label: "blanket".into(),
        };
        let request = Request::builder()
            .method("POST")
            .path("/")
            .body("via-blanket")
            .build()
            .expect("builder");
        let response = SendPipe::call(&pipe, request).await.expect("call");
        let body = response.payload;
        assert_eq!(&body[..], b"via-blanket");
    }
}
