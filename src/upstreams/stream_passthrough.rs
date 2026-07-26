//! Pumps the request body into a `StreamUpstream` and streams the
//! response back. Pair with `StreamListenerProtocol` for a full
//! byte-stream proxy that still composes with the substrate middleware.

use std::future::Future;
use std::net::SocketAddr;
#[cfg(all(feature = "unix", unix))]
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use bon::Builder;
use bytes::Bytes;
use conflaguration::{ConfigDisplay, Settings, Validate, ValidationMessage};
use futures::FutureExt;
use futures::io::{AsyncReadExt, AsyncWriteExt};
use futures::stream::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::body::ResponseStream;
use crate::error::ProximaError;
use crate::header_list::HeaderList;
use crate::listeners::stream_protocol::reader_to_byte_stream;
use proxima_primitives::pipe::SendPipe;

use crate::pipe::{PipeHandle, into_handle};
use crate::pipe_factory::PipeFactory;
use crate::request::{Request, Response};
use crate::stream::{StreamUpstream, StreamUpstreamExt};
#[cfg(all(feature = "tcp", feature = "tokio"))]
use crate::upstreams::tokio_stream::TokioTcpUpstream;
#[cfg(all(feature = "unix", unix, feature = "tokio"))]
use crate::upstreams::tokio_stream::TokioUnixUpstream;
// tokio-free TCP backend — the default (`tcp` no longer implies `tokio`,
// see the umbrella Cargo.toml doc on the `tcp` feature).
#[cfg(all(feature = "tcp", not(feature = "tokio")))]
use proxima_net::prime::PrimeTcpUpstream;
// prime-backed Unix upstream: the compiled-default fallback when no
// `RuntimeSelection` was installed ambiently (see `build_unix_stream_passthrough`).
#[cfg(all(
    feature = "unix",
    unix,
    not(feature = "tokio"),
    feature = "runtime-prime-inbox-alloc",
    any(target_os = "linux", target_os = "macos")
))]
use proxima_net::prime::PrimeUnixUpstream;

const DEFAULT_CHUNK_BYTES: usize = 64 * 1024;
const DEFAULT_TRANSPORT: &str = "tcp";
const DEFAULT_LABEL: &str = "stream";

/// Config and fluent-builder surface for the generic byte-stream upstream.
///
/// This is deliberately transport-shaped, not agent-shaped. The same pipe can
/// connect to a TCP socket today, a Unix socket when enabled, and later a
/// process stdio or PTY backend without changing the pipe contract.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings, ConfigDisplay)]
#[settings(prefix = "PROXIMA_STREAM")]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct StreamPassthroughSettings {
    /// Handler label used for metrics and `Handler::name`.
    #[setting(default_str = "stream")]
    #[serde(default = "default_label")]
    #[builder(default = default_label())]
    pub name: String,

    /// Backend transport. Supported values today: `tcp`, `unix`.
    #[setting(default_str = "tcp")]
    #[serde(default = "default_transport")]
    #[builder(default = default_transport())]
    pub transport: String,

    /// Transport address. For `tcp`, this is a socket address such as
    /// `127.0.0.1:9000`. For `unix`, this is a socket path.
    #[setting(default_str = "")]
    #[serde(default)]
    #[builder(default)]
    pub addr: String,

    /// Read chunk size for the response byte stream.
    #[setting(default = 65536)]
    #[serde(default = "default_chunk_bytes")]
    #[builder(default = default_chunk_bytes())]
    pub chunk_bytes: usize,
}

impl Default for StreamPassthroughSettings {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl StreamPassthroughSettings {
    /// Common fluent constructor for TCP-backed byte streams.
    #[must_use]
    pub fn tcp(addr: SocketAddr) -> Self {
        Self::builder()
            .transport("tcp")
            .addr(addr.to_string())
            .build()
    }

    /// Convert this settings object to the same discriminator shape accepted
    /// by the config loader.
    #[must_use]
    pub fn to_value(self) -> Value {
        let mut map = Map::new();
        map.insert("type".into(), Value::String("stream".into()));
        map.insert("name".into(), Value::String(self.name));
        map.insert("transport".into(), Value::String(self.transport));
        map.insert("addr".into(), Value::String(self.addr));
        map.insert("chunk_bytes".into(), Value::from(self.chunk_bytes));
        Value::Object(map)
    }
}

impl From<StreamPassthroughSettings> for crate::load::Spec {
    fn from(value: StreamPassthroughSettings) -> Self {
        Self::Inline(value.to_value())
    }
}

impl Validate for StreamPassthroughSettings {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.name.trim().is_empty() {
            errors.push(ValidationMessage::new("name", "must not be empty"));
        }
        if self.addr.trim().is_empty() {
            errors.push(ValidationMessage::new("addr", "must not be empty"));
        }
        if self.chunk_bytes == 0 {
            errors.push(ValidationMessage::new("chunk_bytes", "must be > 0"));
        }
        match self.transport.as_str() {
            "tcp" => {
                if !self.addr.trim().is_empty() && self.addr.parse::<SocketAddr>().is_err() {
                    errors.push(ValidationMessage::new(
                        "addr",
                        "must be a valid socket address for transport tcp",
                    ));
                }
            }
            "unix" => {}
            _ => errors.push(ValidationMessage::new(
                "transport",
                "must be one of: tcp, unix",
            )),
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

fn default_label() -> String {
    DEFAULT_LABEL.into()
}

fn default_transport() -> String {
    DEFAULT_TRANSPORT.into()
}

fn default_chunk_bytes() -> usize {
    DEFAULT_CHUNK_BYTES
}

pub struct StreamPassthroughUpstream<U: StreamUpstream + ?Sized> {
    upstream: Arc<U>,
    label: String,
    chunk_bytes: usize,
}

impl<U: StreamUpstream> StreamPassthroughUpstream<U> {
    pub fn new(upstream: U, label: impl Into<String>) -> Self {
        Self {
            upstream: Arc::new(upstream),
            label: label.into(),
            chunk_bytes: DEFAULT_CHUNK_BYTES,
        }
    }
}

impl<U: StreamUpstream + ?Sized> StreamPassthroughUpstream<U> {
    /// Build from an already-erased `Arc<dyn StreamUpstream<..>>` — the seam
    /// a `RuntimeSelection`-selected `UnixUpstreamFactory` dispatches
    /// through (`build_unix_stream_passthrough`), since the factory hands
    /// back one backend or the other behind the same trait-object type.
    pub fn from_arc(upstream: Arc<U>, label: impl Into<String>) -> Self {
        Self {
            upstream,
            label: label.into(),
            chunk_bytes: DEFAULT_CHUNK_BYTES,
        }
    }

    pub fn with_chunk_bytes(mut self, chunk_bytes: usize) -> Self {
        self.chunk_bytes = chunk_bytes.max(1);
        self
    }
}

impl<U: StreamUpstream + ?Sized + 'static> SendPipe for StreamPassthroughUpstream<U> {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let upstream = Arc::clone(&self.upstream);
        let chunk_bytes = self.chunk_bytes;
        let label = self.label.clone();
        async move {
            let cancel = request.context.child_signal();
            let conn = upstream
                .connect()
                .await
                .map_err(|err| ProximaError::Upstream(format!("{label} connect: {err}")))?;
            let (read_half, mut write_half) = conn.split();

            let pump_cancel = cancel.clone();
            let pump = async move {
                let mut request_stream = request.into_chunk_stream();
                loop {
                    let cancelled = pump_cancel.fired().fuse();
                    let next = request_stream.next().fuse();
                    futures::pin_mut!(cancelled, next);
                    futures::select_biased! {
                        () = cancelled => break,
                        chunk = next => match chunk {
                            Some(Ok(bytes)) => {
                                if write_half.write_all(&bytes).await.is_err() {
                                    break;
                                }
                            }
                            Some(Err(_)) | None => break,
                        },
                    }
                }
                let _ = write_half.close().await;
            };
            // `serve-prime` (rides `tcp`, see the umbrella Cargo.toml) puts
            // every connection task on a prime worker already, so this is
            // ambient-safe the same way `tokio::spawn` is ambient-safe on
            // the tokio path — both panic if called off their runtime.
            #[cfg(feature = "tokio")]
            tokio::spawn(pump);
            #[cfg(not(feature = "tokio"))]
            prime::os::core_shard::spawn_on_current_core(Box::pin(pump));

            let response_stream = reader_to_byte_stream(read_half, chunk_bytes);
            Ok(Response {
                status: 200,
                metadata: HeaderList::new(),
                payload: Bytes::new(),
                stream: Some(ResponseStream::new(response_stream)),
                upgrade: None,
            })
        }
    }
}

pub struct StreamPassthroughPipeFactory;

impl StreamPassthroughPipeFactory {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for StreamPassthroughPipeFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl PipeFactory for StreamPassthroughPipeFactory {
    fn name(&self) -> &str {
        "stream"
    }

    fn build(
        &self,
        spec: &Value,
        _inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        Box::pin(async move {
            let settings: StreamPassthroughSettings = serde_json::from_value(spec)
                .map_err(|err| ProximaError::Config(format!("stream upstream settings: {err}")))?;
            build_stream_passthrough(&settings)
        })
    }
}

fn build_stream_passthrough(
    settings: &StreamPassthroughSettings,
) -> Result<PipeHandle, ProximaError> {
    settings
        .validate()
        .map_err(|err| ProximaError::Config(format!("stream upstream settings: {err}")))?;
    match settings.transport.as_str() {
        "tcp" => build_tcp_stream_passthrough(settings),
        "unix" => build_unix_stream_passthrough(settings),
        other => Err(ProximaError::Config(format!(
            "unknown stream transport `{other}`"
        ))),
    }
}

#[cfg(all(feature = "tcp", feature = "tokio"))]
fn build_tcp_stream_passthrough(
    settings: &StreamPassthroughSettings,
) -> Result<PipeHandle, ProximaError> {
    let addr = settings.addr.parse::<SocketAddr>().map_err(|err| {
        ProximaError::Config(format!("stream tcp addr `{}`: {err}", settings.addr))
    })?;
    let upstream = TokioTcpUpstream::new(addr);
    let pipe = StreamPassthroughUpstream::new(upstream, settings.name.clone())
        .with_chunk_bytes(settings.chunk_bytes);
    Ok(into_handle(pipe))
}

#[cfg(all(feature = "tcp", not(feature = "tokio")))]
fn build_tcp_stream_passthrough(
    settings: &StreamPassthroughSettings,
) -> Result<PipeHandle, ProximaError> {
    let addr = settings.addr.parse::<SocketAddr>().map_err(|err| {
        ProximaError::Config(format!("stream tcp addr `{}`: {err}", settings.addr))
    })?;
    let upstream = PrimeTcpUpstream::new(addr);
    let pipe = StreamPassthroughUpstream::new(upstream, settings.name.clone())
        .with_chunk_bytes(settings.chunk_bytes);
    Ok(into_handle(pipe))
}

#[cfg(not(feature = "tcp"))]
fn build_tcp_stream_passthrough(
    _settings: &StreamPassthroughSettings,
) -> Result<PipeHandle, ProximaError> {
    Err(ProximaError::Config(
        "stream transport `tcp` requires the `tcp` feature".into(),
    ))
}

// Selects the unix upstream backend by VALUE: an ambiently-installed
// `RuntimeSelection` (e.g. from `#[proxima::main(runtime = "tokio"|"prime")]`)
// wins whenever one is present, so `unix`'s backend follows the caller's
// choice rather than which of `tokio`/prime happened to be linked (C2 —
// `unix` names no runtime in its own feature list any more, see the
// umbrella Cargo.toml). Falls back to a compiled default only when nothing
// was installed ambiently (e.g. `App::new()` called outside
// `#[proxima::main]`, or a test).
#[cfg(all(feature = "unix", unix))]
fn build_unix_stream_passthrough(
    settings: &StreamPassthroughSettings,
) -> Result<PipeHandle, ProximaError> {
    let path = PathBuf::from(settings.addr.clone());

    if let Some(selection) = crate::runtime::installed_runtime()
        && let Some(factory) = &selection.unix_upstream_factory
    {
        let upstream = factory.connect(path);
        let pipe = StreamPassthroughUpstream::from_arc(upstream, settings.name.clone())
            .with_chunk_bytes(settings.chunk_bytes);
        return Ok(into_handle(pipe));
    }

    build_unix_stream_passthrough_compiled_default(settings, path)
}

#[cfg(all(feature = "unix", unix, feature = "tokio"))]
fn build_unix_stream_passthrough_compiled_default(
    settings: &StreamPassthroughSettings,
    path: PathBuf,
) -> Result<PipeHandle, ProximaError> {
    let upstream = TokioUnixUpstream::new(path);
    let pipe = StreamPassthroughUpstream::new(upstream, settings.name.clone())
        .with_chunk_bytes(settings.chunk_bytes);
    Ok(into_handle(pipe))
}

#[cfg(all(
    feature = "unix",
    unix,
    not(feature = "tokio"),
    feature = "runtime-prime-inbox-alloc",
    any(target_os = "linux", target_os = "macos")
))]
fn build_unix_stream_passthrough_compiled_default(
    settings: &StreamPassthroughSettings,
    path: PathBuf,
) -> Result<PipeHandle, ProximaError> {
    let upstream = PrimeUnixUpstream::new(path);
    let pipe = StreamPassthroughUpstream::new(upstream, settings.name.clone())
        .with_chunk_bytes(settings.chunk_bytes);
    Ok(into_handle(pipe))
}

// neither backend is linked (`unix` compiled, but no runtime capable of
// dialing it) — a runtime `Config` error, not a compile failure: `unix`
// alone no longer implies a runtime (C3), so this is the expected shape for
// a caller who enabled `unix` without also naming a backend.
#[cfg(all(
    feature = "unix",
    unix,
    not(feature = "tokio"),
    not(all(
        feature = "runtime-prime-inbox-alloc",
        any(target_os = "linux", target_os = "macos")
    ))
))]
fn build_unix_stream_passthrough_compiled_default(
    _settings: &StreamPassthroughSettings,
    _path: PathBuf,
) -> Result<PipeHandle, ProximaError> {
    Err(ProximaError::Config(
        "stream transport `unix`: no runtime backend linked — enable `tokio`, the prime \
         `runtime-prime-inbox-alloc` bundle, or select one explicitly via \
         `RuntimeSelection`/`#[proxima::main(runtime = ...)]`"
            .into(),
    ))
}

#[cfg(not(all(feature = "unix", unix)))]
fn build_unix_stream_passthrough(
    _settings: &StreamPassthroughSettings,
) -> Result<PipeHandle, ProximaError> {
    Err(ProximaError::Config(
        "stream transport `unix` requires unix target support and the `unix` feature".into(),
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    #[cfg(feature = "tokio")]
    use crate::listeners::tokio_stream::TokioTcpListener;
    #[cfg(feature = "tokio")]
    use crate::stream::{StreamListener, StreamListenerExt};
    #[cfg(feature = "tokio")]
    use crate::upstreams::tokio_stream::TokioTcpUpstream;
    #[cfg(not(feature = "tokio"))]
    use core::future::poll_fn;
    use std::net::{Ipv4Addr, SocketAddr};
    // `PrimeTcpUpstream` itself is already in scope via `use super::*` above
    // (the module-level import under `all(feature = "tcp", not(feature =
    // "tokio"))`); only the acceptor side is test-only.
    #[cfg(not(feature = "tokio"))]
    use proxima_net::prime::PrimeAcceptorFactory;
    #[cfg(not(feature = "tokio"))]
    use proxima_primitives::stream::{AcceptorFactory, TcpBindOptions};

    fn passthrough_request() -> Request<Bytes> {
        Request {
            method: proxima_primitives::pipe::Method::from_bytes(b"STREAM"),
            path: Bytes::from_static(b"/"),
            query: HeaderList::new(),
            metadata: HeaderList::new(),
            payload: Bytes::from_static(b"hello over passthrough"),
            stream: None,
            context: crate::request::RequestContext::default(),
        }
    }

    #[cfg(feature = "tokio")]
    #[proxima::test]
    async fn passthrough_pumps_bytes_in_and_out() {
        // a tiny echo server backs the upstream side
        let server_listener = TokioTcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind");
        let server_addr = match server_listener.local_addr().expect("local_addr") {
            crate::stream::BindAddr::Tcp(addr) => addr,
            _ => panic!("expected tcp"),
        };
        tokio::spawn(async move {
            let conn = server_listener.accept().await.expect("accept");
            let (mut read, mut write) = futures::io::AsyncReadExt::split(conn);
            futures::io::copy(&mut read, &mut write)
                .await
                .expect("echo copy");
        });

        let upstream = TokioTcpUpstream::new(server_addr);
        let pipe = StreamPassthroughUpstream::new(upstream, "echo");

        let response = pipe.call(passthrough_request()).await.expect("call");
        let collected = response.collect_body().await.expect("collect");
        assert_eq!(&collected[..], b"hello over passthrough");
    }

    // prime-path sibling of `passthrough_pumps_bytes_in_and_out` — same
    // round trip, `PrimeAcceptorFactory`/`PrimeTcpUpstream` instead of the
    // tokio types, proving `StreamPassthroughUpstream::call`'s
    // `spawn_on_current_core` pump runs cleanly under the default
    // tokio-free `tcp` build (see the umbrella Cargo.toml's `tcp` doc).
    #[cfg(not(feature = "tokio"))]
    #[proxima::test]
    async fn passthrough_pumps_bytes_in_and_out() {
        let mut acceptor = PrimeAcceptorFactory
            .bind(
                SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
                TcpBindOptions::default(),
            )
            .expect("bind");
        let server_addr = acceptor.local_addr().expect("local_addr");

        prime::os::core_shard::spawn_on_current_core(Box::pin(async move {
            let conn = poll_fn(|cx| acceptor.poll_accept(cx))
                .await
                .expect("accept");
            let (mut read, mut write) = futures::io::AsyncReadExt::split(conn);
            futures::io::copy(&mut read, &mut write)
                .await
                .expect("echo copy");
        }));

        let upstream = PrimeTcpUpstream::new(server_addr);
        let pipe = StreamPassthroughUpstream::new(upstream, "echo");

        let response = pipe.call(passthrough_request()).await.expect("call");
        let collected = response.collect_body().await.expect("collect");
        assert_eq!(&collected[..], b"hello over passthrough");
    }

    #[test]
    fn settings_round_trip_between_builder_and_config_shape() {
        let from_builder = StreamPassthroughSettings::builder()
            .name("tcp-echo")
            .transport("tcp")
            .addr("127.0.0.1:9000")
            .chunk_bytes(4096)
            .build();
        let from_config: StreamPassthroughSettings = serde_json::from_value(serde_json::json!({
            "name": "tcp-echo",
            "transport": "tcp",
            "addr": "127.0.0.1:9000",
            "chunk_bytes": 4096
        }))
        .expect("settings parse");
        assert_eq!(from_config, from_builder);
        from_config.validate().expect("valid settings");
    }

    fn factory_request() -> Request<Bytes> {
        Request {
            method: proxima_primitives::pipe::Method::from_bytes(b"STREAM"),
            path: Bytes::from_static(b"/"),
            query: HeaderList::new(),
            metadata: HeaderList::new(),
            payload: Bytes::from_static(b"hello through factory"),
            stream: None,
            context: crate::request::RequestContext::default(),
        }
    }

    // the factory's `tcp` transport dispatch is itself gated on the `tcp`
    // feature (see the `build_tcp_stream_passthrough` arms above) —
    // without it, `factory.build(...)` returns a Config error.
    #[cfg(all(feature = "tcp", feature = "tokio"))]
    #[proxima::test]
    async fn factory_builds_tcp_stream_passthrough_from_settings() {
        let server_listener = TokioTcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind");
        let server_addr = match server_listener.local_addr().expect("local_addr") {
            crate::stream::BindAddr::Tcp(addr) => addr,
            _ => panic!("expected tcp"),
        };
        tokio::spawn(async move {
            let conn = server_listener.accept().await.expect("accept");
            let (mut read, mut write) = futures::io::AsyncReadExt::split(conn);
            futures::io::copy(&mut read, &mut write)
                .await
                .expect("echo copy");
        });

        let settings = StreamPassthroughSettings::builder()
            .transport("tcp")
            .addr(server_addr.to_string())
            .name("factory-echo")
            .chunk_bytes(32)
            .build();
        let factory = StreamPassthroughPipeFactory::new();
        let handle = factory
            .build(&settings.to_value(), None)
            .await
            .expect("build stream pipe");
        let response = SendPipe::call(&handle, factory_request())
            .await
            .expect("call");
        let collected = response.collect_body().await.expect("collect");
        assert_eq!(&collected[..], b"hello through factory");
    }

    // prime-path sibling of `factory_builds_tcp_stream_passthrough_from_settings`.
    #[cfg(all(feature = "tcp", not(feature = "tokio")))]
    #[proxima::test]
    async fn factory_builds_tcp_stream_passthrough_from_settings() {
        let mut acceptor = PrimeAcceptorFactory
            .bind(
                SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
                TcpBindOptions::default(),
            )
            .expect("bind");
        let server_addr = acceptor.local_addr().expect("local_addr");

        prime::os::core_shard::spawn_on_current_core(Box::pin(async move {
            let conn = poll_fn(|cx| acceptor.poll_accept(cx))
                .await
                .expect("accept");
            let (mut read, mut write) = futures::io::AsyncReadExt::split(conn);
            futures::io::copy(&mut read, &mut write)
                .await
                .expect("echo copy");
        }));

        let settings = StreamPassthroughSettings::builder()
            .transport("tcp")
            .addr(server_addr.to_string())
            .name("factory-echo")
            .chunk_bytes(32)
            .build();
        let factory = StreamPassthroughPipeFactory::new();
        let handle = factory
            .build(&settings.to_value(), None)
            .await
            .expect("build stream pipe");
        let response = SendPipe::call(&handle, factory_request())
            .await
            .expect("call");
        let collected = response.collect_body().await.expect("collect");
        assert_eq!(&collected[..], b"hello through factory");
    }

    #[test]
    fn settings_reject_unknown_transport() {
        let settings = StreamPassthroughSettings::builder()
            .transport("named-pipe")
            .addr("somewhere")
            .build();
        let error = settings.validate().expect_err("transport should reject");
        let message = format!("{error}");
        assert!(message.contains("transport"), "got: {message}");
    }
}
