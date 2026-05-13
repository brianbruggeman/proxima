//! The listener registry, `ServeContext`/`ListenProtocol` serve surface,
//! and the fluent `ServeBuilder`. std tier: the reactor adapter that binds
//! sockets and drives the [`crate::admission`] core's admit/route
//! decisions onto a runtime.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::future::Future;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;

use arc_swap::ArcSwap;
use bytes::Bytes;
use futures::channel::oneshot;
use proxima_telemetry::warn;
use serde_json::Value;

use crate::{DispatchPolicy, Route};
use proxima_core::ProximaError;
use proxima_primitives::pipe::handler::{PipeHandle, ThreadLocalPipeHandle};
use proxima_primitives::pipe::telemetry_surface::TelemetryHandle;
use proxima_runtime::{CoreId, Runtime};
use proxima_primitives::stream::PeerInfo;

#[derive(Clone, Copy, Debug, Default)]
pub enum HandlerDispatch {
    #[default]
    Inline,
    SpreadToPeers {
        num_cores: usize,
    },
}

pub struct ServeContext {
    pub telemetry: TelemetryHandle,
    pub runtime: Option<Arc<dyn Runtime>>,
    pub acceptor_factory: Option<Arc<dyn proxima_primitives::stream::AcceptorFactory>>,
    /// UDP sibling of `acceptor_factory`: the runtime-agnostic datagram socket
    /// source a QUIC/h3 listener binds through (so it names neither prime nor
    /// tokio). `None` when no runtime is installed (the tokio-default path).
    pub datagram_factory: Option<Arc<dyn proxima_primitives::stream::DatagramFactory>>,
    pub handler_dispatch: HandlerDispatch,
    /// fired once this lane's listening socket has completed its real
    /// `bind`/`listen` syscalls. `Listener::run_with_runtime` blocks on one
    /// signal per lane before handing back a `ListenerHandle` — closing the
    /// startup race where `bind_addr()` reports a resolved address before
    /// any lane has actually started accepting on it.
    pub ready_signal: Option<std::sync::mpsc::Sender<()>>,
}

impl ServeContext {
    #[must_use]
    pub fn new(telemetry: TelemetryHandle) -> Self {
        Self {
            telemetry,
            runtime: None,
            acceptor_factory: None,
            datagram_factory: None,
            handler_dispatch: HandlerDispatch::Inline,
            ready_signal: None,
        }
    }

    #[must_use]
    pub fn with_runtime(mut self, runtime: Arc<dyn Runtime>) -> Self {
        self.runtime = Some(runtime);
        self
    }

    #[must_use]
    pub fn with_acceptor_factory(
        mut self,
        factory: Arc<dyn proxima_primitives::stream::AcceptorFactory>,
    ) -> Self {
        self.acceptor_factory = Some(factory);
        self
    }

    #[must_use]
    pub fn with_datagram_factory(
        mut self,
        factory: Arc<dyn proxima_primitives::stream::DatagramFactory>,
    ) -> Self {
        self.datagram_factory = Some(factory);
        self
    }

    #[must_use]
    pub fn with_handler_dispatch(mut self, dispatch: HandlerDispatch) -> Self {
        self.handler_dispatch = dispatch;
        self
    }

    #[must_use]
    pub fn with_ready_signal(mut self, ready_signal: std::sync::mpsc::Sender<()>) -> Self {
        self.ready_signal = Some(ready_signal);
        self
    }

    #[must_use]
    pub fn clone_telemetry(&self) -> TelemetryHandle {
        self.telemetry.clone()
    }
}

impl HandlerDispatch {
    /// Bridge to the admission core's [`DispatchPolicy`] — the sole
    /// implementation of the reserve-core-0 round-robin decision. `num_cores`
    /// saturates at `u16::MAX` (no deployment plausibly spreads across more
    /// peer cores than that).
    #[must_use]
    pub fn as_policy(&self) -> DispatchPolicy {
        match *self {
            HandlerDispatch::Inline => DispatchPolicy::Inline,
            HandlerDispatch::SpreadToPeers { num_cores } => DispatchPolicy::SpreadToPeers {
                num_cores: u16::try_from(num_cores).unwrap_or(u16::MAX),
            },
        }
    }
}

/// Spawn an admitted connection's future onto the core [`Route`] names.
/// `route` is a decision from the [`crate::admission`] module (either
/// [`crate::ListenerCore::admit`] for a listener already tracking admission,
/// or [`DispatchPolicy::route`] for one that only wants the routing half) —
/// this function never re-derives it.
pub fn dispatch_handler(
    runtime: Option<&Arc<dyn Runtime>>,
    route: Route,
    conn_future: Pin<Box<dyn Future<Output = ()> + Send + 'static>>,
) {
    match (runtime, route) {
        (Some(rt), Route::Inline) => {
            rt.spawn_on_current_core(conn_future);
        }
        // No runtime injected — only reachable when a caller drives a
        // listener without going through `App`/`serve-prime`. Under
        // `tokio`, this must land on the caller's existing `LocalSet`
        // (byte-identical to the pre-migration behaviour). Without it,
        // an OS thread + `block_on` gives the same "runs independently"
        // contract with no runtime dependency.
        #[cfg(feature = "tokio")]
        (None, _) => {
            tokio::task::spawn_local(conn_future);
        }
        #[cfg(not(feature = "tokio"))]
        (None, _) => {
            std::thread::spawn(move || futures::executor::block_on(conn_future));
        }
        (Some(rt), Route::Peer(index)) => {
            let target = CoreId(usize::from(index));
            if let Err(err) = rt.spawn_on_core(target, conn_future) {
                warn!(core = target.0, error = %err, "shed connection: spawn_on_core failed");
            }
        }
    }
}

/// The peer address to admit against, from an accepted connection's
/// [`PeerInfo`]. TCP carries a real address; UDS and the no_std/other
/// fallback have none, so they collapse onto loopback — a single admission
/// bucket for local-trusted transports, not a real per-peer boundary.
#[must_use]
pub fn peer_ip(peer: Option<&PeerInfo>) -> IpAddr {
    match peer {
        Some(PeerInfo::Tcp(addr)) => addr.ip(),
        _ => IpAddr::V4(Ipv4Addr::LOCALHOST),
    }
}

pub trait ListenProtocol: Send + Sync + 'static {
    fn name(&self) -> &str;

    /// Returns a `?Send` future. The trait stays `Send + Sync` so the
    /// registry can hold protocol instances across threads; the future the
    /// protocol returns is per-core (it awaits `?Send` `Pipe::call`s).
    /// Cross-thread bootstrap is via `Runtime::spawn_factory_on_core`: a
    /// `Send` factory closure crosses the worker channel, and the protocol's
    /// `serve` future is constructed locally on the destination core.
    fn serve(
        &self,
        bind: SocketAddr,
        dispatch: PipeHandle,
        spec: &Value,
        context: ServeContext,
        shutdown: oneshot::Receiver<()>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + '_>>;
}

/// Per-thread sibling of [`ListenProtocol`]. The serve loop dispatches
/// into a [`ThreadLocalPipeHandle`] and returns a `?Send` future,
/// suitable for per-core executors that pin acceptance + dispatch to a
/// single thread.
/// Fluent builder for [`ListenProtocol::serve`]. Each setter is
/// optional; awaiting the builder fills in defaults and dispatches
/// to the protocol's positional `serve` implementation.
///
/// Defaults: bind = `127.0.0.1:0` (ephemeral loopback), dispatch =
/// `NullDispatch` (returns 503 on every request, so missed wiring
/// is obvious instead of silent), spec = `Value::Null`, context =
/// `ServeContext::new(NoopTelemetry)`, shutdown = a never-fires
/// receiver.
///
/// ```ignore
/// use proxima::{HttpListenProtocol, ListenProtocolFluent};
///
/// HttpListenProtocol::new()
///     .fluent()
///     .bind("127.0.0.1:0".parse().unwrap())
///     // `.dispatch(handle)` — pass a `PipeHandle`; omitted here so
///     // missed wiring surfaces as a 503 rather than a panic.
///     .await?;
/// ```
pub struct ServeBuilder<'protocol, P: ListenProtocol + ?Sized> {
    protocol: &'protocol P,
    bind: Option<SocketAddr>,
    dispatch: Option<PipeHandle>,
    spec: Option<Value>,
    context: Option<ServeContext>,
    shutdown: Option<oneshot::Receiver<()>>,
}

impl<'protocol, P: ListenProtocol + ?Sized> ServeBuilder<'protocol, P> {
    fn new(protocol: &'protocol P) -> Self {
        Self {
            protocol,
            bind: None,
            dispatch: None,
            spec: None,
            context: None,
            shutdown: None,
        }
    }

    #[must_use]
    pub fn bind(mut self, addr: SocketAddr) -> Self {
        self.bind = Some(addr);
        self
    }

    #[must_use]
    pub fn dispatch(mut self, dispatch: PipeHandle) -> Self {
        self.dispatch = Some(dispatch);
        self
    }

    #[must_use]
    pub fn spec(mut self, spec: Value) -> Self {
        self.spec = Some(spec);
        self
    }

    #[must_use]
    pub fn context(mut self, context: ServeContext) -> Self {
        self.context = Some(context);
        self
    }

    #[must_use]
    pub fn shutdown(mut self, shutdown: oneshot::Receiver<()>) -> Self {
        self.shutdown = Some(shutdown);
        self
    }
}

impl<'protocol, P: ListenProtocol + ?Sized> std::future::IntoFuture for ServeBuilder<'protocol, P> {
    type Output = Result<(), ProximaError>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'protocol>>;

    fn into_future(self) -> Self::IntoFuture {
        let Self {
            protocol,
            bind,
            dispatch,
            spec,
            context,
            shutdown,
        } = self;
        let bind = bind.unwrap_or_else(|| SocketAddr::from(([127, 0, 0, 1], 0)));
        let dispatch = dispatch.unwrap_or_else(null_dispatch);
        let spec = spec.unwrap_or(Value::Null);
        let context = context.unwrap_or_else(|| {
            ServeContext::new(Arc::new(proxima_primitives::pipe::telemetry_surface::NoopTelemetry))
        });
        let shutdown = shutdown.unwrap_or_else(|| oneshot::channel().1);
        Box::pin(async move {
            protocol
                .serve(bind, dispatch, &spec, context, shutdown)
                .await
        })
    }
}

/// Extension trait: `protocol.fluent()` returns a [`ServeBuilder`].
/// Blanket-impl'd for every [`ListenProtocol`], so any concrete
/// listener — `H1ListenProtocol`, `H2ListenProtocol`,
/// `H3ListenProtocol`, `HttpListenProtocol`, future ones — gets
/// the fluent surface for free.
pub trait ListenProtocolFluent: ListenProtocol {
    fn fluent(&self) -> ServeBuilder<'_, Self> {
        ServeBuilder::new(self)
    }
}

impl<P: ListenProtocol + ?Sized> ListenProtocolFluent for P {}

fn null_dispatch() -> PipeHandle {
    use proxima_primitives::pipe::SendPipe;
    use proxima_primitives::pipe::handler::into_handle;
    use proxima_primitives::pipe::request::{Request, Response};
    struct NullPipe;
    impl SendPipe for NullPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            async move {
                let mut response = Response::new(503);
                response.payload = bytes::Bytes::from_static(b"no pipe mounted on this listener");
                Ok(response)
            }
        }
    }
    into_handle(NullPipe)
}

pub trait ThreadLocalListenProtocol: 'static {
    fn name(&self) -> &str;

    fn serve(
        &self,
        bind: SocketAddr,
        dispatch: ThreadLocalPipeHandle,
        spec: &Value,
        context: ServeContext,
        shutdown: oneshot::Receiver<()>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + '_>>;
}

#[derive(Default)]
pub struct ThreadLocalListenRegistry {
    protocols: RefCell<BTreeMap<String, Rc<dyn ThreadLocalListenProtocol>>>,
}

impl ThreadLocalListenRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self {
            protocols: RefCell::new(BTreeMap::new()),
        }
    }

    pub fn register(
        &self,
        protocol: Rc<dyn ThreadLocalListenProtocol>,
    ) -> Result<(), ProximaError> {
        let name = protocol.name().to_string();
        let mut guard = self.protocols.borrow_mut();
        if guard.contains_key(&name) {
            return Err(ProximaError::Registry(format!(
                "thread-local listen protocol '{name}' already registered"
            )));
        }
        guard.insert(name, protocol);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Result<Rc<dyn ThreadLocalListenProtocol>, ProximaError> {
        let guard = self.protocols.borrow();
        guard.get(name).cloned().ok_or_else(|| {
            ProximaError::Registry(format!("no thread-local listen protocol named '{name}'"))
        })
    }

    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.protocols.borrow().keys().cloned().collect()
    }
}

pub struct ListenRegistry {
    protocols: ArcSwap<BTreeMap<String, Arc<dyn ListenProtocol>>>,
}

impl Default for ListenRegistry {
    fn default() -> Self {
        Self {
            protocols: ArcSwap::from_pointee(BTreeMap::new()),
        }
    }
}

impl ListenRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, protocol: Arc<dyn ListenProtocol>) -> Result<(), ProximaError> {
        let name = protocol.name().to_string();
        loop {
            let current = self.protocols.load_full();
            if current.contains_key(&name) {
                return Err(ProximaError::Registry(format!(
                    "listen protocol '{name}' already registered"
                )));
            }
            let mut next: BTreeMap<String, Arc<dyn ListenProtocol>> = (*current).clone();
            next.insert(name.clone(), protocol.clone());
            let prev = self.protocols.compare_and_swap(&current, Arc::new(next));
            if Arc::ptr_eq(&prev, &current) {
                return Ok(());
            }
        }
    }

    pub fn get(&self, name: &str) -> Result<Arc<dyn ListenProtocol>, ProximaError> {
        self.protocols
            .load_full()
            .get(name)
            .cloned()
            .ok_or_else(|| ProximaError::Registry(format!("no listen protocol named '{name}'")))
    }

    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.protocols.load_full().keys().cloned().collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    struct StubProto {
        registered_name: String,
    }

    impl ListenProtocol for StubProto {
        fn name(&self) -> &str {
            &self.registered_name
        }

        fn serve(
            &self,
            _bind: SocketAddr,
            _dispatch: PipeHandle,
            _spec: &Value,
            _context: ServeContext,
            _shutdown: oneshot::Receiver<()>,
        ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + '_>> {
            Box::pin(async move { Ok(()) })
        }
    }

    #[proxima::test]
    async fn fluent_serve_builder_dispatches_with_defaults() {
        let proto = StubProto {
            registered_name: "fluent-default".into(),
        };
        // No setters called — every field defaulted by the builder.
        let outcome: Result<(), ProximaError> = proto.fluent().await;
        assert!(outcome.is_ok());
    }

    #[proxima::test]
    async fn fluent_serve_builder_forwards_overrides() {
        let proto = StubProto {
            registered_name: "fluent-overrides".into(),
        };
        let (_shutdown_tx, shutdown_rx) = oneshot::channel();
        let outcome: Result<(), ProximaError> = proto
            .fluent()
            .bind("127.0.0.1:0".parse().expect("addr"))
            .spec(serde_json::json!({"max_body_bytes": 1024}))
            .shutdown(shutdown_rx)
            .await;
        assert!(outcome.is_ok());
    }

    #[proxima::test]
    async fn register_and_lookup_round_trip() {
        let registry = ListenRegistry::new();
        registry
            .register(Arc::new(StubProto {
                registered_name: "test".into(),
            }))
            .expect("register");
        let proto = registry.get("test").expect("get");
        assert_eq!(proto.name(), "test");
    }

    #[proxima::test]
    async fn duplicate_register_returns_registry_error() {
        let registry = ListenRegistry::new();
        registry
            .register(Arc::new(StubProto {
                registered_name: "dup".into(),
            }))
            .expect("first register");
        let outcome = registry.register(Arc::new(StubProto {
            registered_name: "dup".into(),
        }));
        assert!(matches!(outcome, Err(ProximaError::Registry(_))));
    }

    struct LocalStubProto {
        registered_name: String,
    }

    impl ThreadLocalListenProtocol for LocalStubProto {
        fn name(&self) -> &str {
            &self.registered_name
        }

        fn serve(
            &self,
            _bind: SocketAddr,
            _dispatch: ThreadLocalPipeHandle,
            _spec: &Value,
            _context: ServeContext,
            _shutdown: oneshot::Receiver<()>,
        ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + '_>> {
            Box::pin(async move { Ok(()) })
        }
    }

    #[proxima::test]
    async fn thread_local_registry_register_and_lookup_round_trip() {
        let registry = ThreadLocalListenRegistry::new();
        registry
            .register(Rc::new(LocalStubProto {
                registered_name: "local".into(),
            }))
            .expect("register");
        let proto = registry.get("local").expect("get");
        assert_eq!(proto.name(), "local");
    }

    #[proxima::test]
    async fn thread_local_registry_duplicate_register_errors() {
        let registry = ThreadLocalListenRegistry::new();
        registry
            .register(Rc::new(LocalStubProto {
                registered_name: "dup".into(),
            }))
            .expect("first register");
        let outcome = registry.register(Rc::new(LocalStubProto {
            registered_name: "dup".into(),
        }));
        assert!(matches!(outcome, Err(ProximaError::Registry(_))));
    }

    #[test]
    fn handler_dispatch_as_policy_reserves_core_zero_and_round_robins() {
        let policy = HandlerDispatch::SpreadToPeers { num_cores: 4 }.as_policy();
        let mut cursor = 0usize;
        let routes: [Route; 6] = core::array::from_fn(|_| policy.route(&mut cursor));
        assert_eq!(
            routes,
            [
                Route::Peer(1),
                Route::Peer(2),
                Route::Peer(3),
                Route::Peer(1),
                Route::Peer(2),
                Route::Peer(3),
            ],
            "expected reserve-core-0 round-robin from SpreadToPeers {{num_cores: 4}}"
        );

        let single_core_policy = HandlerDispatch::SpreadToPeers { num_cores: 1 }.as_policy();
        let mut single_cursor = 0usize;
        assert_eq!(
            single_core_policy.route(&mut single_cursor),
            Route::Peer(0),
            "single core must always return core 0"
        );
    }
}
