use bytes::Bytes;
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use arc_swap::ArcSwap;

use crate::pipe::ProximaError;
use crate::pipe::SendPipe;
use crate::pipe::handler::{PipeHandle, into_handle};
use crate::pipe::request::{Request, Response};

// SwappablePipe is Send-only by design. Stage 10 fork stops here:
// the swap discipline (ArcSwap<PipeHandle>) is fundamentally a
// cross-thread atomic mutation. A per-core `!Send` variant would need a
// different mechanism (per-core Cell<Rc<...>> + cross-core broadcast
// over Runtime::spawn_on_core), which is a Stage 11 / DPDK-era addition.
// The plan classifies swap as a bootstrap-style cross-thread primitive
// and explicitly leaves Send required here.

/// addressable, atomically-swappable node in a Pipe chain. wraps an
/// `ArcSwap<PipeHandle>` so a swap replaces the delegate handle
/// without tearing in-flight calls — a call that already grabbed the
/// old handle finishes against the old impl; subsequent calls dispatch
/// to the new impl.
///
/// minimum-viable splice: no checkpoint negotiation, no quiescence.
/// Pipes with cross-call state that genuinely need a quiet point
/// before they can be replaced should drain their own state under the
/// swap (e.g., by holding a lock that the new impl picks up).
pub struct SwappablePipe {
    delegate: ArcSwap<PipeHandle>,
}

impl SwappablePipe {
    #[must_use]
    pub fn new(initial: PipeHandle) -> Self {
        Self {
            delegate: ArcSwap::from_pointee(initial),
        }
    }

    pub fn swap(&self, new: PipeHandle) {
        self.delegate.store(Arc::new(new));
    }

    #[must_use]
    pub fn current(&self) -> PipeHandle {
        PipeHandle::clone(&self.delegate.load_full())
    }
}

impl SendPipe for SwappablePipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        // grab the current delegate *before* awaiting so the swap discipline
        // holds: an in-flight call always finishes on the handle it observed
        // at entry, regardless of later swaps.
        let handle = PipeHandle::clone(&self.delegate.load_full());
        async move { SendPipe::call(&handle, request).await }
    }
}

/// per-chain node registry. lookup-by-id swaps; iteration listed; thread-safe.
/// lock-free via `ArcSwap<HashMap<...>>`: reads are atomic loads, writes do
/// copy-on-write. `register` is rare (chain construction); `swap` lookups
/// happen at the cross-core swap broadcast frequency.
pub struct SwapRegistry {
    nodes: ArcSwap<HashMap<String, Arc<SwappablePipe>>>,
}

impl Default for SwapRegistry {
    fn default() -> Self {
        Self {
            nodes: ArcSwap::from_pointee(HashMap::new()),
        }
    }
}

impl SwapRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// install a fresh swappable node. returns its handle for chain wiring.
    /// re-registration with the same id replaces the prior node — useful for
    /// scenarios that rebuild the chain top-to-bottom.
    pub fn register(&self, node_id: impl Into<String>, initial: PipeHandle) -> PipeHandle {
        let node_id = node_id.into();
        let node = Arc::new(SwappablePipe::new(initial));
        let handle = node.clone();
        // CAS-loop: read current map, clone, insert, swap.
        loop {
            let current = self.nodes.load_full();
            let mut next: HashMap<String, Arc<SwappablePipe>> = (*current).clone();
            next.insert(node_id.clone(), node.clone());
            let prev = self.nodes.compare_and_swap(&current, Arc::new(next));
            if Arc::ptr_eq(&prev, &current) {
                break;
            }
        }
        into_handle(SwappableHandle { inner: handle })
    }

    /// atomically swap the delegate behind `node_id`. returns Ok if the node
    /// exists, Err with a structured config error if not — silently dropping
    /// a swap that targets a missing node would surface as confusing latent
    /// behavior on the next request.
    pub fn swap(&self, node_id: &str, new: PipeHandle) -> Result<(), ProximaError> {
        let snapshot = self.nodes.load_full();
        let Some(node) = snapshot.get(node_id) else {
            return Err(ProximaError::Config(format!(
                "swap: no node registered for id \"{node_id}\""
            )));
        };
        node.swap(new);
        Ok(())
    }

    #[must_use]
    pub fn node_ids(&self) -> Vec<String> {
        self.nodes.load_full().keys().cloned().collect()
    }
}

/// thin Pipe wrapper that owns the Arc<SwappablePipe> so the registry's
/// node and the chain's handle share the same swap target.
struct SwappableHandle {
    inner: Arc<SwappablePipe>,
}

impl SendPipe for SwappableHandle {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        SendPipe::call(self.inner.as_ref(), request)
    }
}

// `#[proxima::test]` and inline `tokio::task::{LocalSet, spawn_local, yield_now}`
// pull in the `proxima` / `tokio` dev-dependencies, which the loom build keeps
// out of the graph (see `[target.'cfg(not(loom))'.dev-dependencies]` in
// Cargo.toml); these tests are unrelated to the Notify/watch loom protocol.
#[cfg(all(test, not(loom)))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use bytes::Bytes;

    struct Constant {
        body: &'static [u8],
    }

    impl SendPipe for Constant {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let body = self.body;
            async move { Ok(Response::ok(Bytes::from_static(body))) }
        }
    }

    fn fresh_request() -> Request<Bytes> {
        Request::builder()
            .method("GET")
            .path("/x")
            .body("")
            .build()
            .expect("builder")
    }

    #[proxima::test]
    async fn swap_replaces_delegate_for_subsequent_calls() {
        let registry = SwapRegistry::new();
        let handle = registry.register("downstream-proxy", into_handle(Constant { body: b"v1" }));
        let response = SendPipe::call(&handle, fresh_request()).await.expect("v1");
        let body = response.collect_body().await.expect("body");
        assert_eq!(&body[..], b"v1");

        registry
            .swap("downstream-proxy", into_handle(Constant { body: b"v2" }))
            .expect("swap");
        let response = SendPipe::call(&handle, fresh_request()).await.expect("v2");
        let body = response.collect_body().await.expect("body");
        assert_eq!(&body[..], b"v2");
    }

    #[proxima::test(runtime = "tokio")]
    async fn in_flight_call_finishes_on_pre_swap_delegate() {
        // a Pipe that holds the call open until a notifier fires — lets
        // the test interleave the swap between call entry and call completion.
        struct Held {
            label: &'static [u8],
            notify: Arc<crate::sync::Notify>,
        }

        impl SendPipe for Held {
            type In = Request<Bytes>;
            type Out = Response<Bytes>;
            type Err = ProximaError;

            fn call(
                &self,
                _request: Request<Bytes>,
            ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
                let notify = self.notify.clone();
                let label = self.label;
                async move {
                    notify.notified().await;
                    Ok(Response::ok(Bytes::from_static(label)))
                }
            }
        }

        // Handler::call returns ?Send after Stage 1, so the spawned per-call
        // futures must run on a LocalSet rather than tokio::spawn.
        tokio::task::LocalSet::new()
            .run_until(async move {
                let notify_old = Arc::new(crate::sync::Notify::new());
                let notify_new = Arc::new(crate::sync::Notify::new());
                let registry = SwapRegistry::new();
                let handle = registry.register(
                    "downstream-proxy",
                    into_handle(Held {
                        label: b"old",
                        notify: notify_old.clone(),
                    }),
                );

                // 1. spawn first call — it grabs the OLD delegate and parks on notify_old.
                let handle_for_old_call = handle.clone();
                let in_flight_old = tokio::task::spawn_local(async move {
                    SendPipe::call(&handle_for_old_call, fresh_request()).await
                });
                // yield enough times for the spawned task to enter call() and observe
                // the old delegate before we swap.
                for _ in 0..16 {
                    tokio::task::yield_now().await;
                }

                // 2. swap to NEW delegate (uses notify_new).
                registry
                    .swap(
                        "downstream-proxy",
                        into_handle(Held {
                            label: b"new",
                            notify: notify_new.clone(),
                        }),
                    )
                    .expect("swap");

                // 3. release the in-flight OLD call — it must finish on the old impl.
                notify_old.notify_waiters();
                let response_old = in_flight_old.await.expect("join").expect("call");
                let body_old = response_old.collect_body().await.expect("body");
                assert_eq!(
                    &body_old[..],
                    b"old",
                    "in-flight call must finish on pre-swap delegate"
                );

                // 4. fresh call after swap hits NEW. spawn it, then notify_new.
                let handle_for_new_call = handle.clone();
                let in_flight_new = tokio::task::spawn_local(async move {
                    SendPipe::call(&handle_for_new_call, fresh_request()).await
                });
                for _ in 0..16 {
                    tokio::task::yield_now().await;
                }
                notify_new.notify_waiters();
                let response_new = in_flight_new.await.expect("join").expect("call");
                let body_new = response_new.collect_body().await.expect("body");
                assert_eq!(&body_new[..], b"new", "post-swap call hits new delegate");
            })
            .await;
    }

    #[proxima::test]
    async fn swap_unknown_node_returns_config_error() {
        let registry = SwapRegistry::new();
        let outcome = registry.swap("missing", into_handle(Constant { body: b"x" }));
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn node_ids_lists_registered_nodes() {
        let registry = SwapRegistry::new();
        let _ = registry.register("alpha", into_handle(Constant { body: b"" }));
        let _ = registry.register("beta", into_handle(Constant { body: b"" }));
        let mut ids = registry.node_ids();
        ids.sort();
        assert_eq!(ids, vec!["alpha".to_string(), "beta".to_string()]);
    }
}
