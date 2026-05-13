//! `Offload` — a typed Pipe combinator that runs its inner Pipe's `call`
//! on the runtime's background pool instead of the calling executor
//! thread.
//!
//! `SpreadToPeers` promises isolation for a synchronously-blocking
//! handler (one that never yields — `std::thread::sleep`, a blocking
//! FFI call): a per-core executor has no neighbor thread to steal work
//! from, so a blocking `Pipe::call` there wedges every connection
//! sharing that core. A background-pool thread has no such neighbor.
//!
//! Wrap the served [`PipeHandle`] in `Offload` ONCE per listener
//! (`HttpListenProtocol::serve`, before the accept loop starts) — never
//! per request. The per-request hot path underneath then stays a plain
//! `SendPipe::call` await: no `Box::pin`, no per-request clone.

use std::any::Any;
use std::future::Future;
use std::sync::Arc;

use bytes::Bytes;

use proxima_core::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::handler::PipeHandle;
use proxima_primitives::pipe::request::{Request, Response};
use proxima_runtime::Runtime;

/// Wraps a [`PipeHandle`] so every call runs on the runtime's
/// background pool rather than the caller's executor thread. See the
/// module docs for the isolation contract this satisfies.
pub struct Offload {
    inner: PipeHandle,
    runtime: Arc<dyn Runtime>,
}

impl Offload {
    #[must_use]
    pub fn new(inner: PipeHandle, runtime: Arc<dyn Runtime>) -> Self {
        Self { inner, runtime }
    }
}

impl SendPipe for Offload {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        // `PipeHandle` is `Arc<dyn DynPipe>` — cloning is an atomic
        // refcount bump, not a deep copy, so owning one per background
        // job (instead of borrowing) costs nothing worth measuring and
        // satisfies `spawn_background_blocking`'s `'static` bound.
        let inner = self.inner.clone();
        let runtime = self.runtime.clone();
        async move {
            let work: Box<dyn FnOnce() -> Result<Box<dyn Any + Send>, ProximaError> + Send> =
                Box::new(move || {
                    futures::executor::block_on(SendPipe::call(&inner, request))
                        .map(|response| Box::new(response) as Box<dyn Any + Send>)
                });
            let boxed = runtime.spawn_background_blocking(work).await?;
            boxed
                .downcast::<Response<Bytes>>()
                .map(|response| *response)
                .map_err(|_| ProximaError::Body("offload: downcast mismatch".into()))
        }
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::pin::Pin;

    use proxima_runtime::{BackgroundHandle, CoreId, SpawnError};

    use super::*;
    use proxima_primitives::pipe::handler::into_handle;

    /// Fake `Runtime` whose ONLY meaningful method is
    /// `spawn_background_blocking` — it hands `work` to a freshly spawned
    /// `std::thread`, so awaiting the returned handle genuinely crosses
    /// threads (not an inline stand-in). The other methods are never
    /// exercised by `Offload::call` and panic if they ever are, so a
    /// future change routing through them fails loudly instead of
    /// silently passing.
    struct ThreadSpawningRuntime;

    impl Runtime for ThreadSpawningRuntime {
        fn spawn_on_current_core(&self, _future: Pin<Box<dyn Future<Output = ()> + 'static>>) {
            unreachable!("offload never calls spawn_on_current_core")
        }

        fn spawn_on_core(
            &self,
            _core_id: CoreId,
            _future: Pin<Box<dyn Future<Output = ()> + Send + 'static>>,
        ) -> Result<(), SpawnError> {
            unreachable!("offload never calls spawn_on_core")
        }

        fn spawn_factory_on_core(
            &self,
            _core_id: CoreId,
            _factory: Box<
                dyn FnOnce() -> Pin<Box<dyn Future<Output = ()> + 'static>> + Send + 'static,
            >,
        ) -> Result<(), SpawnError> {
            unreachable!("offload never calls spawn_factory_on_core")
        }

        fn spawn_background_blocking(
            &self,
            work: Box<dyn FnOnce() -> Result<Box<dyn Any + Send>, ProximaError> + Send>,
        ) -> BackgroundHandle<Box<dyn Any + Send>> {
            let (result_tx, result_rx) = futures::channel::oneshot::channel();
            std::thread::spawn(move || {
                let _ = result_tx.send(work());
            });
            Box::pin(async move {
                result_rx
                    .await
                    .unwrap_or_else(|_| Err(ProximaError::Body("background thread dropped".into())))
            })
        }

        fn timer_at(
            &self,
            _deadline: std::time::Instant,
        ) -> Pin<Box<dyn Future<Output = ()> + 'static>> {
            unreachable!("offload never calls timer_at")
        }

        fn num_cores(&self) -> usize {
            1
        }

        fn current_core(&self) -> CoreId {
            CoreId(0)
        }
    }

    struct EchoThreadId;

    impl SendPipe for EchoThreadId {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            async move {
                let thread_id = format!("{:?}", std::thread::current().id());
                Ok(Response::ok(thread_id))
            }
        }
    }


    struct FixedBody;

    impl SendPipe for FixedBody {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            async move { Ok(Response::ok("payload")) }
        }
    }


    fn request() -> Request<Bytes> {
        Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("builder")
    }

    #[proxima::test]
    async fn offload_runs_inner_pipe_off_the_calling_thread() {
        let runtime: Arc<dyn Runtime> = Arc::new(ThreadSpawningRuntime);
        let inner: PipeHandle = into_handle(EchoThreadId);
        let offload = Offload::new(inner, runtime);

        let calling_thread = format!("{:?}", std::thread::current().id());
        let response = SendPipe::call(&offload, request())
            .await
            .expect("offload call");
        let inner_thread = String::from_utf8(response.payload.to_vec()).expect("utf8 body");

        assert_ne!(
            calling_thread, inner_thread,
            "inner pipe must run on the background pool, not the caller's thread"
        );
    }

    #[proxima::test]
    async fn offload_propagates_inner_pipe_response_body() {
        let runtime: Arc<dyn Runtime> = Arc::new(ThreadSpawningRuntime);
        let inner: PipeHandle = into_handle(FixedBody);
        let offload = Offload::new(inner, runtime);

        let response = SendPipe::call(&offload, request())
            .await
            .expect("offload call");
        assert_eq!(&response.payload[..], b"payload");
    }
}
