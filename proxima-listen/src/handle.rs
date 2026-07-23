use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use conflaguration::Settings;
use futures::FutureExt;
use futures::channel::oneshot;
use proxima_telemetry::warn;
use serde_json::Value;
#[cfg(feature = "tokio")]
use tokio::task::JoinHandle;

use crate::{ListenProtocol, ListenRegistry, ListenTuningConfig, ServeContext};
use proxima_core::ProximaError;
use proxima_primitives::pipe::handler::PipeHandle;
use proxima_primitives::pipe::telemetry_surface::TelemetryHandle;

/// upper bound on how long a caller blocks for a listener to report ready
/// before giving up and returning an error — generous relative to a
/// bind/listen syscall so only a genuinely stuck lane (not scheduler
/// jitter) trips it. Shared by `run_with_runtime` (multi-lane) and
/// `App::run_until_signal` (single core-0 lane).
pub const LISTENER_READY_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone)]
pub enum ShutdownPolicy {
    Immediate,
    Drain {
        timeout: Duration,
    },
    Quiesce {
        duration: Duration,
        then: Box<ShutdownPolicy>,
    },
}

impl ShutdownPolicy {
    /// Drain with the process-default timeout (historically a fixed 30s;
    /// now sourced from [`ListenTuningConfig::drain_timeout_ms`], which
    /// defaults from the `sized` floor and is env/file/builder overridable).
    #[must_use]
    pub fn drain_30s() -> Self {
        Self::Drain {
            timeout: Duration::from_millis(ListenTuningConfig::default().drain_timeout_ms),
        }
    }
}

#[derive(Clone)]
pub struct ListenerSpec {
    pub bind: SocketAddr,
    pub protocol_name: String,
    pub shutdown: ShutdownPolicy,
    pub spec: Value,
    /// A protocol resolved at construction time instead of by name through
    /// the [`ListenRegistry`] at serve time. Populated by [`Self::protocol`]
    /// (the escape hatch below), and by the umbrella `proxima` crate's
    /// `ListenerBuilder::serve` when its `ListenerProtocolExt`/
    /// `ListenerTransportExt` spec resolves to a protocol outside `App::new()`'s
    /// default set (`h2`/`h3-native`) — see `resolve_listen_protocol` in
    /// `src/listener/handle.rs` — so a fluent caller's choice can never
    /// desync from `protocol_name`. `None` for the registry-lookup path
    /// (`ListenerSpec::http`, or any spec built from config/deserialization)
    /// — [`Listener::run_with_runtime`] falls back to
    /// `registry.get(&self.protocol_name)` exactly as before.
    ///
    /// This is also how TLS composes: there is deliberately no `tls` field
    /// on this struct (a typed `Option<TlsConfig>` slot would make TLS a
    /// property of every protocol variant — a protocol × tls matrix). TLS
    /// termination is instead [`TlsListenProtocol`], a `ListenProtocol`
    /// DECORATOR that wraps whatever concrete protocol is carried here —
    /// on/off is the presence of that wrapper, composed the same way any
    /// other concrete protocol reaches this field: through [`Self::protocol`].
    pub protocol: Option<Arc<dyn ListenProtocol>>,
}

impl ListenerSpec {
    #[must_use]
    pub fn http(bind: SocketAddr) -> Self {
        Self {
            bind,
            protocol_name: "http".into(),
            shutdown: ShutdownPolicy::drain_30s(),
            spec: Value::Null,
            protocol: None,
        }
    }

    /// Escape hatch for any concrete [`ListenProtocol`] — h1/h2/h3-native
    /// (whose types live in `proxima-http`, a crate that already depends on
    /// this one, so this crate cannot host named per-protocol constructors
    /// for them without a cyclic dependency), or a caller's own out-of-crate
    /// impl. The mirror of the client's `.protocol(impl ClientProtocol)`
    /// escape hatch. Sets `protocol_name` from
    /// [`ListenProtocol::name`] so the wire/serialized string always
    /// matches the carried `Arc`, and carries the `Arc` itself so
    /// [`Listener::run_with_runtime`] resolves it directly instead of a
    /// registry lookup by name.
    #[must_use]
    pub fn protocol(bind: SocketAddr, protocol: Arc<dyn ListenProtocol>) -> Self {
        Self {
            bind,
            protocol_name: protocol.name().to_string(),
            shutdown: ShutdownPolicy::drain_30s(),
            spec: Value::Null,
            protocol: Some(protocol),
        }
    }

    #[must_use]
    pub fn with_shutdown(mut self, policy: ShutdownPolicy) -> Self {
        self.shutdown = policy;
        self
    }

    #[must_use]
    pub fn with_spec(mut self, spec: Value) -> Self {
        self.spec = spec;
        self
    }

    pub fn attach(self, dispatch: PipeHandle) -> Listener {
        Listener {
            bind: self.bind,
            protocol_name: self.protocol_name,
            shutdown: self.shutdown,
            spec: self.spec,
            protocol: self.protocol,
            dispatch,
        }
    }
}

pub struct Listener {
    pub bind: SocketAddr,
    pub protocol_name: String,
    pub shutdown: ShutdownPolicy,
    pub spec: Value,
    pub protocol: Option<Arc<dyn ListenProtocol>>,
    pub dispatch: PipeHandle,
}

impl Listener {
    /// Bind one listener per core via SO_REUSEPORT and dispatch each
    /// listener future to its own per-core worker. The kernel
    /// hash-distributes incoming connections across the N listeners
    /// (Linux's SO_REUSEPORT load balancer); each connection is
    /// processed inline on the core that accepted it, never
    /// migrating.
    ///
    /// If `bind.port() == 0` (ephemeral), the resolver binds once
    /// temporarily to capture an OS-assigned port, then drops that
    /// socket and dispatches the real per-core listeners at the
    /// resolved port. The brief temporary bind is the only
    /// kernel-side coordination needed.
    pub fn run_with_runtime(
        self,
        registry: &ListenRegistry,
        telemetry: TelemetryHandle,
        runtime: Option<Arc<dyn proxima_runtime::Runtime>>,
        acceptor_factory: Option<Arc<dyn proxima_primitives::stream::AcceptorFactory>>,
        datagram_factory: Option<Arc<dyn proxima_primitives::stream::DatagramFactory>>,
    ) -> Result<ListenerHandle, ProximaError> {
        let runtime = runtime.ok_or_else(|| {
            ProximaError::Config(
                "Listener::run_with_runtime requires a Runtime: enable the \
                 `runtime-tokio` feature and pass an installed runtime. \
                 ListenProtocol::serve returns ?Send and cannot be \
                 tokio::spawn'd onto the work-stealing executor"
                    .into(),
            )
        })?;
        // A carried `Arc` (from `.protocol(bind, ..)`, a composed
        // `TlsListenProtocol` wrapper, or the umbrella crate's fluent
        // sugar) resolves directly — the fluent path never round-trips
        // through a typo-able registry lookup. Absent (registry-driven /
        // deserialized specs), fall back to the by-name lookup exactly as
        // before. TLS is NOT a field read here — a `TlsListenProtocol`
        // (if `self.protocol` carries one) injects its own marker into the
        // spec it hands its wrapped inner protocol, inside its own `serve`.
        let protocol = match self.protocol {
            Some(protocol) => protocol,
            None => registry.get(&self.protocol_name)?,
        };
        let dispatch = self.dispatch;
        let mut spec = self.spec.clone();
        let policy = self.shutdown.clone();
        attach_shutdown_to_spec(&mut spec, &policy);
        let resolved_addr = resolve_listen_port(self.bind)?;
        attach_reuseport_flag(&mut spec);

        let tuning = ListenTuningConfig::from_env()
            .map_err(|err| ProximaError::Config(format!("listen tuning config: {err}")))?;
        let is_http = self.protocol_name == "http";
        let use_spread = is_http
            && (cfg!(any(
                target_os = "macos",
                target_os = "freebsd",
                target_os = "openbsd",
                target_os = "netbsd"
            )) || tuning.http_handler_spread);

        let num_cores = runtime.num_cores().max(1);
        let num_lanes = if use_spread { 1 } else { num_cores };

        let shutdown_notify = Arc::new(proxima_primitives::sync::Notify::new());
        let (shutdown_tx, shutdown_rx) = oneshot::channel();

        let runtime_for_handle = runtime.clone();
        let bridge_runtime = runtime.clone();
        let notify_for_signaler = shutdown_notify.clone();
        let bridge_future: Pin<Box<dyn Future<Output = ()> + Send + 'static>> =
            Box::pin(async move {
                let _ = shutdown_rx.await;
                notify_for_signaler.notify_waiters();
            });
        if let Err(err) = bridge_runtime.spawn_on_core(proxima_runtime::CoreId(0), bridge_future) {
            return Err(ProximaError::Config(format!(
                "listener shutdown bridge spawn failed: {err}"
            )));
        }
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<()>();
        for core_index in 0..num_lanes {
            let protocol_for_lane = protocol.clone();
            let dispatch_for_lane = dispatch.clone();
            let spec_for_lane = spec.clone();
            let telemetry_for_lane = telemetry.clone();
            let runtime_for_factory = runtime.clone();
            let notify_for_lane = shutdown_notify.clone();
            let acceptor_factory_for_lane = acceptor_factory.clone();
            let datagram_factory_for_lane = datagram_factory.clone();
            let ready_tx_for_lane = ready_tx.clone();
            let handler_dispatch_for_lane = if use_spread {
                crate::HandlerDispatch::SpreadToPeers { num_cores }
            } else {
                crate::HandlerDispatch::Inline
            };
            if let Err(err) = runtime.spawn_factory_on_core(
                proxima_runtime::CoreId(core_index),
                Box::new(move || {
                    let mut context = ServeContext::new(telemetry_for_lane)
                        .with_runtime(runtime_for_factory)
                        .with_handler_dispatch(handler_dispatch_for_lane)
                        .with_ready_signal(ready_tx_for_lane);
                    if let Some(factory) = acceptor_factory_for_lane {
                        context = context.with_acceptor_factory(factory);
                    }
                    if let Some(factory) = datagram_factory_for_lane {
                        context = context.with_datagram_factory(factory);
                    }
                    let (lane_shutdown_tx, lane_shutdown_rx) = oneshot::channel();
                    Box::pin(async move {
                        let serve_future = serve(
                            protocol_for_lane,
                            resolved_addr,
                            dispatch_for_lane,
                            spec_for_lane,
                            context,
                            lane_shutdown_rx,
                        )
                        .fuse();
                        futures::pin_mut!(serve_future);
                        let notify_future = notify_for_lane.notified().fuse();
                        futures::pin_mut!(notify_future);
                        futures::select_biased! {
                            _ = notify_future => {
                                let _ = lane_shutdown_tx.send(());
                                if let Err(error) = serve_future.await {
                                    warn!(
                                        ?error,
                                        core = core_index,
                                        "listener lane exited with error",
                                    );
                                }
                            }
                            outcome = serve_future => {
                                if let Err(error) = outcome {
                                    warn!(
                                        ?error,
                                        core = core_index,
                                        "listener lane exited with error",
                                    );
                                }
                            }
                        }
                    })
                }),
            ) {
                return Err(proxima_core::ProximaError::Config(format!(
                    "listener lane spawn failed on core {core_index}: {err}"
                )));
            }
        }
        // `resolved_addr` is valid the moment a probe socket bound it (see
        // `resolve_listen_port`), but nothing is actually LISTENING until a
        // lane's `serve` future gets its first poll and calls the acceptor's
        // real `bind`/`listen` — that happens whenever the target core's
        // inbox drains, not synchronously with the `spawn_factory_on_core`
        // call above. Block here for one ack per lane so a caller holding
        // `ListenerHandle::bind_addr()` never observes a resolved-but-not-yet-
        // listening address (proven: a burst of clients dialing immediately
        // after this returned got `ECONNREFUSED` before this wait existed).
        drop(ready_tx);
        for _ in 0..num_lanes {
            if ready_rx.recv_timeout(LISTENER_READY_TIMEOUT).is_err() {
                return Err(ProximaError::Config(format!(
                    "listener did not become ready on all {num_lanes} lane(s) within {LISTENER_READY_TIMEOUT:?}"
                )));
            }
        }
        Ok(ListenerHandle {
            bind_addr: Some(resolved_addr),
            shutdown: Some(shutdown_tx),
            #[cfg(feature = "tokio")]
            join: None,
            _runtime: Some(runtime_for_handle),
        })
    }
}

/// Marker key on the listener spec that tells the protocol
/// (currently HTTP) to construct its listening socket with
/// SO_REUSEPORT so multiple cores can bind to the same port.
pub const REUSEPORT_SPEC_KEY: &str = "__proxima_reuseport";

fn attach_reuseport_flag(spec: &mut Value) {
    if !matches!(spec, Value::Object(_)) {
        *spec = Value::Object(serde_json::Map::new());
    }
    if let Value::Object(table) = spec {
        table.insert(REUSEPORT_SPEC_KEY.to_string(), Value::Bool(true));
    }
}

/// Resolve a possibly-zero port to an OS-assigned one. If the input
/// already has a non-zero port we trust it. Otherwise we briefly bind
/// a SO_REUSEPORT'd socket to grab a port and immediately drop it —
/// the actual per-core listeners then rebind to that port.
fn resolve_listen_port(addr: SocketAddr) -> Result<SocketAddr, ProximaError> {
    if addr.port() != 0 {
        return Ok(addr);
    }
    let socket = build_reuseport_socket(&addr).map_err(|err| {
        ProximaError::Io(std::io::Error::other(format!(
            "resolve port: build socket: {err}"
        )))
    })?;
    socket.bind(&addr.into()).map_err(|err| {
        ProximaError::Io(std::io::Error::other(format!("resolve port: bind: {err}")))
    })?;
    let resolved = socket
        .local_addr()
        .map_err(|err| {
            ProximaError::Io(std::io::Error::other(format!(
                "resolve port: local_addr: {err}"
            )))
        })?
        .as_socket()
        .ok_or_else(|| ProximaError::Io(std::io::Error::other("resolve port: not IP")))?;
    drop(socket);
    Ok(resolved)
}

/// Build a TCP socket with SO_REUSEADDR + SO_REUSEPORT set BEFORE
/// bind. The bind itself happens at the caller. Used both for the
/// port resolver and (via `bind_reuseport_listener`) for each
/// per-core accept lane.
pub fn build_reuseport_socket(addr: &SocketAddr) -> std::io::Result<socket2::Socket> {
    use socket2::{Domain, Protocol, Socket, Type};
    let domain = if addr.is_ipv4() {
        Domain::IPV4
    } else {
        Domain::IPV6
    };
    let socket = Socket::new(domain, Type::STREAM, Some(Protocol::TCP))?;
    socket.set_reuse_address(true)?;
    #[cfg(unix)]
    socket.set_reuse_port(true)?;
    socket.set_nonblocking(true)?;
    Ok(socket)
}

/// Bind a new SO_REUSEPORT'd `tokio::net::TcpListener` to `addr`.
/// Each per-core accept lane calls this from inside its own factory.
/// The kernel routes incoming SYNs across all sockets in the same
/// `(addr, port)` SO_REUSEPORT group.
///
/// Only reachable from the tokio-backed legacy accept loops
/// (`ListenProtocol` impls fall back to their own `AcceptorFactory` +
/// `poll_accept` path — see `serve_via_factory` siblings — when no
/// tokio runtime is present).
#[cfg(feature = "tokio")]
pub fn bind_reuseport_listener(addr: SocketAddr) -> std::io::Result<tokio::net::TcpListener> {
    bind_reuseport_listener_with_options(addr, None)
}

/// Same as [`bind_reuseport_listener`] but lets the caller enable
/// TCP Fast Open (RFC 7413) on the listening socket. `tcp_fastopen_queue`
/// is the kernel's passive-open queue depth — clients with TFO cookies
/// can carry data in their SYN, saving one RTT on connection setup.
///
/// **Linux only** today — the setsockopt path uses Linux's
/// `TCP_FASTOPEN` (option 23) which takes the queue size directly.
/// macOS supports TFO via a different optname value (261) and a
/// 0/1 enable flag, not exposed here. On non-Linux platforms the
/// option is silently dropped; the listener binds normally.
///
/// Operators also need `sysctl net.ipv4.tcp_fastopen=2` (or 3 for
/// both server + client roles) for the kernel to advertise TFO.
#[cfg(feature = "tokio")]
pub fn bind_reuseport_listener_with_options(
    addr: SocketAddr,
    tcp_fastopen_queue: Option<u32>,
) -> std::io::Result<tokio::net::TcpListener> {
    let socket = build_reuseport_socket(&addr)?;
    socket.bind(&addr.into())?;
    if let Some(queue) = tcp_fastopen_queue {
        apply_tcp_fastopen(&socket, queue)?;
    }
    socket.listen(ListenTuningConfig::default().backlog)?;
    let std_listener: std::net::TcpListener = socket.into();
    std_listener.set_nonblocking(true)?;
    tokio::net::TcpListener::from_std(std_listener)
}

#[cfg(all(feature = "tokio", target_os = "linux"))]
fn apply_tcp_fastopen(socket: &socket2::Socket, queue: u32) -> std::io::Result<()> {
    use std::os::unix::io::AsRawFd;
    // Linux constants — IPPROTO_TCP and TCP_FASTOPEN are stable
    // across kernels back to 3.7. Inlining avoids pulling libc as a
    // hard non-target-gated dep.
    const IPPROTO_TCP: i32 = 6;
    const TCP_FASTOPEN: i32 = 23;
    let value = queue as i32;
    unsafe extern "C" {
        fn setsockopt(
            sockfd: i32,
            level: i32,
            optname: i32,
            optval: *const core::ffi::c_void,
            optlen: u32,
        ) -> i32;
    }
    let ret = unsafe {
        setsockopt(
            socket.as_raw_fd(),
            IPPROTO_TCP,
            TCP_FASTOPEN,
            &value as *const _ as *const core::ffi::c_void,
            std::mem::size_of::<i32>() as u32,
        )
    };
    if ret != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(all(feature = "tokio", not(target_os = "linux")))]
fn apply_tcp_fastopen(_socket: &socket2::Socket, _queue: u32) -> std::io::Result<()> {
    // macOS / Windows / BSDs use different optname values + semantics;
    // not wired today. Accept the spec without erroring so operators
    // can target Linux + dev on macOS without conditional config.
    Ok(())
}

#[cfg(feature = "tls")]
fn attach_tls_to_spec(spec: &mut Value, tls: &proxima_tls::TlsConfig) {
    if !matches!(spec, Value::Object(_)) {
        *spec = Value::Object(serde_json::Map::new());
    }
    let Value::Object(table) = spec else {
        return;
    };
    table.insert(
        proxima_tls::SPEC_KEY.to_string(),
        proxima_tls::config_to_spec_value(tls),
    );
}

/// Composes TLS termination in FRONT of an inner [`ListenProtocol`] — the
/// on/off toggle for TLS is the PRESENCE of this wrapper, not a struct field
/// on [`ListenerSpec`]/[`Listener`]. A `tls: Option<TlsConfig>` field there
/// would make TLS a property of every protocol variant, forcing a protocol
/// × tls matrix as new wire protocols land; a decorator composes onto ANY
/// [`ListenProtocol`] uniformly, the same way [`Offload`](crate::Offload)
/// composes a background-pool hop onto any [`PipeHandle`] — no new
/// mechanism, same shape.
///
/// `serve` clones the spec it's handed, stamps the `__proxima_tls` marker
/// [`attach_tls_to_spec`] always used, and hands THAT to the wrapped
/// protocol's own `serve` — the wrapped protocol (e.g. `HttpListenProtocol`)
/// reads the identical marker key it always has; no change on that side.
/// `name()` delegates to the inner protocol so identity checks that key off
/// the wire name (e.g. the `is_http` SO_REUSEPORT-spread pick in
/// `Listener::run_with_runtime`) see straight through the wrapper.
#[cfg(feature = "tls")]
pub struct TlsListenProtocol {
    inner: Arc<dyn ListenProtocol>,
    tls: proxima_tls::TlsConfig,
}

#[cfg(feature = "tls")]
impl TlsListenProtocol {
    #[must_use]
    pub fn new(inner: Arc<dyn ListenProtocol>, tls: proxima_tls::TlsConfig) -> Self {
        Self { inner, tls }
    }
}

#[cfg(feature = "tls")]
impl ListenProtocol for TlsListenProtocol {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn serve(
        &self,
        bind: SocketAddr,
        dispatch: PipeHandle,
        spec: &Value,
        context: ServeContext,
        shutdown: oneshot::Receiver<()>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + '_>> {
        let mut spec_with_tls = spec.clone();
        attach_tls_to_spec(&mut spec_with_tls, &self.tls);
        let inner = self.inner.clone();
        Box::pin(async move {
            inner
                .serve(bind, dispatch, &spec_with_tls, context, shutdown)
                .await
        })
    }
}

fn attach_shutdown_to_spec(spec: &mut Value, policy: &ShutdownPolicy) {
    if !matches!(spec, Value::Object(_)) {
        *spec = Value::Object(serde_json::Map::new());
    }
    let Value::Object(table) = spec else {
        return;
    };
    match policy {
        ShutdownPolicy::Immediate => {
            table
                .entry("drain_timeout_ms".to_string())
                .or_insert(Value::Number(0.into()));
        }
        ShutdownPolicy::Drain { timeout } => {
            table
                .entry("drain_timeout_ms".to_string())
                .or_insert(Value::Number((timeout.as_millis() as u64).into()));
        }
        ShutdownPolicy::Quiesce { duration, then } => {
            table
                .entry("quiesce_duration_ms".to_string())
                .or_insert(Value::Number((duration.as_millis() as u64).into()));
            let drain_ms = match then.as_ref() {
                ShutdownPolicy::Drain { timeout } => timeout.as_millis() as u64,
                ShutdownPolicy::Immediate => 0,
                ShutdownPolicy::Quiesce { .. } => 30_000,
            };
            table
                .entry("drain_timeout_ms".to_string())
                .or_insert(Value::Number(drain_ms.into()));
        }
    }
}

async fn serve(
    protocol: Arc<dyn ListenProtocol>,
    bind: SocketAddr,
    dispatch: PipeHandle,
    spec: Value,
    context: ServeContext,
    shutdown: oneshot::Receiver<()>,
) -> Result<(), ProximaError> {
    protocol
        .serve(bind, dispatch, &spec, context, shutdown)
        .await
}

pub struct ListenerHandle {
    bind_addr: Option<SocketAddr>,
    shutdown: Option<oneshot::Sender<()>>,
    // Never populated with `Some` by any constructor in this crate today —
    // every path (`run_with_runtime`, `new_external`) sets `None`. Kept
    // (not deleted) because `stop()` reads it; gated on `tokio` since the
    // type itself is tokio's.
    #[cfg(feature = "tokio")]
    join: Option<JoinHandle<()>>,
    /// keep per-core runtime workers alive across `App` drop. None when
    /// the listener fell back to ambient `tokio::spawn`.
    _runtime: Option<Arc<dyn proxima_runtime::Runtime>>,
}

impl ListenerHandle {
    pub async fn stop(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        #[cfg(feature = "tokio")]
        if let Some(join) = self.join.take() {
            let _ = join.await;
        }
    }

    pub fn shutdown_signal(&mut self) -> Option<oneshot::Sender<()>> {
        self.shutdown.take()
    }

    pub fn shutdown(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
    }

    #[must_use]
    pub fn bind_addr(&self) -> Option<SocketAddr> {
        self.bind_addr
    }

    /// Construct a `ListenerHandle` for a serve path that bypasses
    /// `Listener::run_with_runtime` (e.g. prime-native serving, where
    /// the accept loop runs directly on prime's per-core executor
    /// with prime's `net::TcpListener` instead of the tokio-coupled
    /// `HttpListenProtocol` pipeline). The handle still owns the
    /// shutdown sender and keeps the runtime alive via `Arc`.
    #[must_use]
    pub fn new_external(
        bind_addr: SocketAddr,
        shutdown: oneshot::Sender<()>,
        runtime: Arc<dyn proxima_runtime::Runtime>,
    ) -> Self {
        Self {
            bind_addr: Some(bind_addr),
            shutdown: Some(shutdown),
            #[cfg(feature = "tokio")]
            join: None,
            _runtime: Some(runtime),
        }
    }
}

impl Drop for ListenerHandle {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
    }
}

#[derive(Debug, Clone)]
pub struct ListenerConfig {
    pub bind: SocketAddr,
    pub protocol: String,
    pub spec: Value,
}

impl ListenerConfig {
    #[must_use]
    pub fn http(bind: SocketAddr) -> Self {
        Self {
            bind,
            protocol: "http".into(),
            spec: Value::Null,
        }
    }

    #[must_use]
    pub fn with_spec(mut self, spec: Value) -> Self {
        self.spec = spec;
        self
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::any::Any;
    use std::sync::atomic::{AtomicBool, Ordering};

    use bytes::Bytes;
    use proxima_primitives::pipe::SendPipe;
    use proxima_primitives::pipe::handler::into_handle;
    use proxima_primitives::pipe::request::{Request, Response};
    use proxima_primitives::pipe::telemetry_surface::NoopTelemetry;
    use proxima_runtime::{BackgroundHandle, CoreId, Runtime, SpawnError};

    use super::*;

    #[test]
    fn http_helper_sets_defaults() {
        let bind: SocketAddr = "127.0.0.1:8080".parse().expect("address parses");
        let listener = ListenerConfig::http(bind);
        assert_eq!(listener.bind, bind);
        assert_eq!(listener.protocol, "http");
    }

    #[test]
    fn shutdown_policy_default_is_drain_30s() {
        let policy = ShutdownPolicy::drain_30s();
        match policy {
            ShutdownPolicy::Drain { timeout } => assert_eq!(timeout, Duration::from_secs(30)),
            _ => panic!("expected drain policy"),
        }
    }

    struct StubProto {
        registered_name: &'static str,
    }

    impl ListenProtocol for StubProto {
        fn name(&self) -> &str {
            self.registered_name
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

    /// Captures whatever `spec: &Value` it's handed, then answers `Ok(())`
    /// immediately — used to prove [`TlsListenProtocol::serve`] stamps the
    /// TLS marker into the spec it hands its wrapped inner protocol.
    #[cfg(feature = "tls")]
    struct SpecCapturingProto {
        captured: Arc<std::sync::Mutex<Option<Value>>>,
    }

    #[cfg(feature = "tls")]
    impl ListenProtocol for SpecCapturingProto {
        fn name(&self) -> &str {
            "captured"
        }

        fn serve(
            &self,
            _bind: SocketAddr,
            _dispatch: PipeHandle,
            spec: &Value,
            _context: ServeContext,
            _shutdown: oneshot::Receiver<()>,
        ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + '_>> {
            *self.captured.lock().expect("lock spec capture cell") = Some(spec.clone());
            Box::pin(async move { Ok(()) })
        }
    }

    #[cfg(feature = "tls")]
    #[test]
    fn tls_listen_protocol_name_delegates_to_inner() {
        let inner: Arc<dyn ListenProtocol> = Arc::new(StubProto {
            registered_name: "stub",
        });
        let wrapped = TlsListenProtocol::new(inner, proxima_tls::TlsConfig::self_signed());
        assert_eq!(wrapped.name(), "stub");
    }

    #[cfg(feature = "tls")]
    #[test]
    fn tls_listen_protocol_stamps_the_marker_into_the_spec_it_hands_the_inner_protocol() {
        let captured = Arc::new(std::sync::Mutex::new(None));
        let inner: Arc<dyn ListenProtocol> = Arc::new(SpecCapturingProto {
            captured: captured.clone(),
        });
        let wrapped = TlsListenProtocol::new(inner, proxima_tls::TlsConfig::self_signed());

        let bind: SocketAddr = "127.0.0.1:0".parse().expect("address parses");
        let context = ServeContext::new(Arc::new(NoopTelemetry));
        let (_shutdown_tx, shutdown_rx) = oneshot::channel();
        futures::executor::block_on(wrapped.serve(
            bind,
            null_dispatch(),
            &Value::Null,
            context,
            shutdown_rx,
        ))
        .expect("serve resolves");

        let spec = captured
            .lock()
            .expect("lock spec capture cell")
            .clone()
            .expect("inner protocol's serve was called");
        assert!(
            spec.get(proxima_tls::SPEC_KEY).is_some(),
            "expected {} in {spec:?}",
            proxima_tls::SPEC_KEY
        );
    }

    #[test]
    fn http_constructor_carries_no_protocol_and_falls_back_to_registry() {
        let bind: SocketAddr = "127.0.0.1:0".parse().expect("address parses");
        let spec = ListenerSpec::http(bind);
        assert_eq!(spec.protocol_name, "http");
        assert!(
            spec.protocol.is_none(),
            "http() is the registry-driven constructor; it must not carry an Arc"
        );
    }

    #[test]
    fn protocol_constructor_sets_name_and_carries_protocol() {
        let bind: SocketAddr = "127.0.0.1:0".parse().expect("address parses");
        let protocol: Arc<dyn ListenProtocol> = Arc::new(StubProto {
            registered_name: "stub",
        });
        let spec = ListenerSpec::protocol(bind, protocol);
        assert_eq!(spec.protocol_name, "stub");
        let carried = spec.protocol.as_ref().expect("protocol() must carry Arc");
        assert_eq!(carried.name(), "stub");
    }

    #[test]
    fn protocol_survives_attach_onto_listener() {
        let bind: SocketAddr = "127.0.0.1:0".parse().expect("address parses");
        let protocol: Arc<dyn ListenProtocol> = Arc::new(StubProto {
            registered_name: "stub",
        });
        let spec = ListenerSpec::protocol(bind, protocol);
        let listener = spec.attach(null_dispatch());
        assert_eq!(listener.protocol_name, "stub");
        let carried = listener
            .protocol
            .as_ref()
            .expect("attach() must carry the protocol through");
        assert_eq!(carried.name(), "stub");
    }

    /// Trivial `SendPipe` for the `dispatch` a `Listener` requires but the
    /// resolution tests below never actually invoke — the fake protocol's
    /// `serve` never calls into it.
    struct NullPipe;

    impl SendPipe for NullPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            async move { Ok(Response::new(200)) }
        }
    }

    fn null_dispatch() -> PipeHandle {
        into_handle(NullPipe)
    }

    /// Fake protocol whose `serve` signals readiness immediately and then
    /// blocks on the shutdown channel — no socket, no accept loop — so the
    /// `run_with_runtime` resolution test below stays a unit test instead
    /// of a real end-to-end bind.
    struct ReadySignalProtocol {
        served: Arc<AtomicBool>,
    }

    impl ListenProtocol for ReadySignalProtocol {
        fn name(&self) -> &str {
            "ready-signal-fake"
        }

        fn serve(
            &self,
            _bind: SocketAddr,
            _dispatch: PipeHandle,
            _spec: &Value,
            context: ServeContext,
            shutdown: oneshot::Receiver<()>,
        ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + '_>> {
            self.served.store(true, Ordering::SeqCst);
            Box::pin(async move {
                if let Some(sender) = context.ready_signal {
                    let _ = sender.send(());
                }
                let _ = shutdown.await;
                Ok(())
            })
        }
    }

    /// Every spawn hands its future to a real `std::thread` running a
    /// single-future executor — genuinely concurrent with the caller
    /// blocking on `run_with_runtime`'s ready-ack recv, without pulling in
    /// tokio or prime as a dev-dependency. Methods `run_with_runtime`
    /// never calls panic loudly instead of silently no-op'ing.
    struct ThreadRuntime;

    impl Runtime for ThreadRuntime {
        fn spawn_on_current_core(&self, _future: Pin<Box<dyn Future<Output = ()> + 'static>>) {
            unreachable!("run_with_runtime never calls spawn_on_current_core")
        }

        fn spawn_on_core(
            &self,
            _core_id: CoreId,
            future: Pin<Box<dyn Future<Output = ()> + Send + 'static>>,
        ) -> Result<(), SpawnError> {
            std::thread::spawn(move || futures::executor::block_on(future));
            Ok(())
        }

        fn spawn_factory_on_core(
            &self,
            _core_id: CoreId,
            factory: Box<
                dyn FnOnce() -> Pin<Box<dyn Future<Output = ()> + 'static>> + Send + 'static,
            >,
        ) -> Result<(), SpawnError> {
            std::thread::spawn(move || futures::executor::block_on(factory()));
            Ok(())
        }

        fn spawn_background_blocking(
            &self,
            _work: Box<dyn FnOnce() -> Result<Box<dyn Any + Send>, ProximaError> + Send>,
        ) -> BackgroundHandle<Box<dyn Any + Send>> {
            unreachable!("run_with_runtime never calls spawn_background_blocking")
        }

        fn timer_at(
            &self,
            _deadline: std::time::Instant,
        ) -> Pin<Box<dyn Future<Output = ()> + 'static>> {
            unreachable!("run_with_runtime never calls timer_at")
        }

        fn num_cores(&self) -> usize {
            1
        }

        fn current_core(&self) -> CoreId {
            CoreId(0)
        }
    }

    #[test]
    fn run_with_runtime_resolves_carried_protocol_without_registry_entry() {
        let served = Arc::new(AtomicBool::new(false));
        let bind: SocketAddr = "127.0.0.1:0".parse().expect("address parses");
        let protocol: Arc<dyn ListenProtocol> = Arc::new(ReadySignalProtocol {
            served: served.clone(),
        });
        let listener = ListenerSpec::protocol(bind, protocol)
            .attach(null_dispatch());

        // Empty registry — proves resolution never reaches `registry.get`.
        let registry = ListenRegistry::new();
        let telemetry: TelemetryHandle = Arc::new(NoopTelemetry);
        let runtime: Arc<dyn Runtime> = Arc::new(ThreadRuntime);

        let handle = listener
            .run_with_runtime(&registry, telemetry, Some(runtime), None, None)
            .expect("carried protocol resolves without a registry entry");
        assert!(served.load(Ordering::SeqCst), "carried protocol never served");
        futures::executor::block_on(handle.stop());
    }

    #[test]
    fn run_with_runtime_still_uses_registry_when_no_protocol_carried() {
        let bind: SocketAddr = "127.0.0.1:0".parse().expect("address parses");
        let listener = ListenerSpec::http(bind).attach(null_dispatch());
        assert!(listener.protocol.is_none());

        let registry = ListenRegistry::new();
        let telemetry: TelemetryHandle = Arc::new(NoopTelemetry);
        let runtime: Arc<dyn Runtime> = Arc::new(ThreadRuntime);

        let outcome = listener.run_with_runtime(&registry, telemetry, Some(runtime), None, None);
        let error = match outcome {
            Ok(_) => panic!("http() with nothing registered under \"http\" must still fail"),
            Err(error) => error,
        };
        let message = error.to_string();
        assert!(
            message.contains("http"),
            "expected the registry miss to name the protocol, got: {message}"
        );
    }
}
