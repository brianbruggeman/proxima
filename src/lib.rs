//! proxima is a config-driven proxy and middleware runtime. The
//! primitive is [`Pipe`] — an async `request → response` boundary
//! that every upstream, middleware, and composition unit implements.
//! Pipes compose recursively; the same spec drives the library, the
//! CLI, and an MCP control plane.
//!
//! For the teaching surface, start at the [`pipe`] module — its
//! `//!` rustdoc is the canonical introduction to Pipe-as-the-primitive
//! (composition, middleware, substrate primitives, recording, serving).
//! For the per-core runtime, see [`prime`]. Hot-swap, recording,
//! replay, and `causal explain` are substrate primitives — see
//! [`Replay`], [`Diff`], [`Isolate`], [`Causal`], [`SwappablePipe`],
//! [`WriteBack`], [`check_determinism`].

// `alloc` is referenced by the disciplined-component telemetry modules,
// which were originally no_std. brought in here so `alloc::vec::Vec`
// etc. resolve identically inside `crate::telemetry` regardless of
// whether the parent crate is std or no_std.
extern crate alloc;

#[cfg(test)]
extern crate self as proxima;

pub mod app;
pub mod app_builder;
pub mod app_config;
pub use proxima_core::buffer as buffer_pool;
// the no-runtime drive primitive; `proxima::runtime::block_on` is its
// runtime-holding sibling, and the `run*` edge drivers boot a runtime on top.
pub use proxima_primitives::block_on;
pub use proxima_primitives::pipe::body;
pub use proxima_recording::pipe::causality;
pub mod client;
pub use proxima_codec as codec;
pub use proxima_codec::factory as codec_factory;
pub use proxima_config as config_format;
pub use proxima_patterns::control_plane;
pub use proxima_patterns::middleware::context_inject;
pub mod daemon_control_plane;
pub mod determinism;
pub use proxima_core as error;
#[cfg(feature = "tcp")]
pub use proxima_protocols::json_framing as framing;
pub use proxima_http::http1::h1;
pub use proxima_http::http1::h1_body;
pub use proxima_http::http1::h1_connection;
pub use proxima_http::http1::h1_response;
#[cfg(feature = "http1")]
pub use proxima_http::http1::hyper_body;

// Composable, `Pipe`-shaped outbound HTTP/1.1 client — the production prime
// client path, usable as a transport STAGE in a pipe chain (e.g. a telemetry
// OTLP exporter terminal), not just the request-builder `Client` sugar. Pair
// `H1ClientUpstream` (a `Pipe`) with `PrimeTcpUpstream` (a `StreamUpstream`) and
// wrap via `pipe::into_handle` to get a `PipeHandle` to inject. Gated on
// `http-prime` (enables `proxima-h1/stream-client` + `proxima-net-prime`).
#[cfg(feature = "http-prime-deps")]
pub use proxima_http::http1::{
    H1ClientConfig, H1ClientUpstream, ResponseBodyMode, ResponseHandling, ResponseHandlingConfig,
    ResponseHeaderMode,
};
pub use proxima_patterns::kv;
pub use proxima_listen as listen;
// low-level serve-time plumbing (bind/spec/dispatch + per-core `run_with_runtime`;
// `Listener` itself lives here, see `listen_handle::Listener`). Named
// `listen_handle`, not `listener` — the bare `listener` name is this crate's
// own module (below), which composes this primitive into a fluent
// `Listener::builder()` surface via `ListenerBuilderEntry`, not a peer type.
pub use proxima_listen::handle as listen_handle;
pub mod listener;
#[cfg(all(
    feature = "http-prime-deps",
    any(target_os = "linux", target_os = "macos")
))]
pub use proxima_net::prime::PrimeTcpUpstream;
pub use proxima_primitives::pipe::header_list;
pub mod load;
pub use proxima_primitives::pipe::routing as mount;
pub use proxima_telemetry::log_buffer;
pub use proxima_net::packet;
pub use proxima_primitives::pipe::path_pattern;
#[cfg(feature = "amqp-listener")]
pub use proxima_protocols::amqp;
#[cfg(feature = "dns-substrate")]
pub use proxima_protocols::dns;
#[cfg(feature = "grpc-framing")]
pub use proxima_protocols::grpc_framing as grpc;
#[cfg(feature = "kafka-listener")]
pub use proxima_protocols::kafka;
#[cfg(feature = "memcached-listener")]
pub use proxima_protocols::memcached;
#[cfg(feature = "mqtt-listener")]
pub use proxima_protocols::mqtt;
#[cfg(feature = "protobuf-wire")]
pub use proxima_protocols::protobuf_wire as protobuf;
pub use proxima_protocols::proxy_protocol;
#[cfg(feature = "websocket-frame")]
pub use proxima_protocols::websocket_frame;
pub mod recording;
pub use proxima_primitives::pipe::request;

pub mod runtime;
pub mod scenarios;
pub use proxima_patterns::balancer::selection;
#[cfg(feature = "http2")]
pub use proxima_http::http2 as h2;
#[cfg(feature = "http3")]
pub use proxima_http::http3 as h3;
pub use proxima_primitives::pipe;
#[cfg(feature = "http3")]
pub use proxima_quic as quic;
pub use proxima_config::schema;
pub use proxima_primitives::sync::shutdown;
#[cfg(feature = "sync-wrappers")]
pub use proxima_primitives::sync;
pub use proxima_primitives::sync::task;
pub use proxima_core::time;
pub mod server;
pub mod settings;
pub use proxima_primitives::pipe::swap_registry as swap;
#[cfg(feature = "http1")]
pub use proxima_http::http1::shared_http;
pub use proxima_primitives::pipe::capture_surface;
pub use proxima_primitives::pipe::endpoint;
pub use proxima_primitives::pipe::telemetry_surface;
pub use proxima_config::store as state_store;
pub use proxima_primitives::stream;
pub use proxima_config::sugar;
pub use proxima_telemetry as telemetry;
// OTLP exporter face: `OtlpClient::http().endpoint(..).build()` (and the config
// path `exporter_pipe`/`recorder_from_config`) lower to a prime `OtlpHttpCodec
// -> transport` pipe chain. lives here (not the leaf) because it needs an HTTP
// client in scope; transport is a builder verb, never a type name.
#[cfg(all(
    feature = "otlp-http",
    feature = "http-prime",
    any(target_os = "linux", target_os = "macos")
))]
pub mod otlp;
pub use proxima_http::templates;
#[cfg(feature = "tls")]
pub use proxima_tls as tls;
pub mod tracing_init;
pub use proxima_patterns::balancer::upstream_ref;
pub use proxima_patterns::kv::write_back;
pub use proxima_primitives::pipe::pipe_factory;
pub use proxima_primitives::pipe::upgrade;

pub mod listeners;
pub mod middlewares;
// DAG-of-child-process-stages executor (`PipelineExecutor::run` fire-and-
// forgets each pipeline run via `tokio::spawn`) — a genuine tokio::process
// + tokio::spawn capability with no prime equivalent today, same as
// `scenarios::orchestrator`.
#[cfg(feature = "tokio")]
pub mod pipelines;
pub mod upstreams;
pub mod verify;

pub use app::{App, AppPipeBuilder, IntoMountTarget, MountTarget, RunConfig, Shutdown, offline_runtime};
pub use causality::{ByteRange, Causal, CausalEdge, CausalIndex};
pub use determinism::check_determinism;
#[cfg(feature = "rayon")]
pub use runtime::RayonBackgroundPool;
#[cfg(feature = "runtime-tokio")]
pub use runtime::TokioPerCoreRuntime;
pub use runtime::{BackgroundHandle, BackgroundPool, CoreId, Runtime};

/// The per-core runtime. See the [runtime discipline] doc for the why,
/// the [compatibility tradeoffs] doc for Tokio interop, and `PrimeRuntime::builder()`
/// or `PrimeConfig` for the entry points.
///
/// [runtime discipline]: https://github.com/brianbruggeman/proxima/blob/main/docs/runtime-prime/discipline.md
/// [compatibility tradeoffs]: https://github.com/brianbruggeman/proxima/blob/main/docs/runtime-prime/compat-mode-tradeoffs.md
#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
pub mod prime {
    pub use crate::runtime::{BackgroundHandle, BackgroundPool, CoreId, Runtime, SpawnError};
    pub use prime::config::{Affinity, Builder, CoreSelection, PoolKind, PrimeConfig};
    pub use prime::os::runtime::PrimeRuntime;
}

/// `#[proxima::test]` attribute — one test attribute that drives the body on
/// proxima's prime runtime (tokio fallback) and subsumes `#[rstest]` +
/// cassette record/replay. See `docs/proxima-test/`.
#[cfg(any(feature = "macros", test))]
pub use proxima_macros::test;

/// `#[proxima::main]` attribute — the production sibling of `#[proxima::test]`.
/// Turns `async fn main() -> R` into a sync entry point that boots a runtime
/// and drives the body via `proxima::runtime::run*` (adaptive default —
/// prime when compiled, else tokio).
#[cfg(any(feature = "macros", test))]
pub use proxima_macros::main;

/// `#[proxima::fixture]` — native rstest-style fixtures (no rstest dep),
/// consumed by `#[proxima::test]` parameters. See `docs/proxima-test/`.
#[cfg(any(feature = "macros", test))]
pub use proxima_macros::fixture;

/// `#[proxima::piped]` — generates a `Pipe`/`SendPipe`/`UnpinPipe`/
/// `UnpinSendPipe` impl from a plain function. See `proxima_macros::piped`.
#[cfg(any(feature = "macros", test))]
pub use proxima_macros::piped;

/// `#[proxima::instrument]` — the unified observability annotation: one
/// attribute yields a trace span, its duration histogram (behind
/// `instrument-metrics`), and a log line, all from the same expansion. Use
/// this on a handler instead of hand-rolling any of the three. See
/// `proxima_macros::instrument`.
#[cfg(any(feature = "macros", test))]
pub use proxima_macros::instrument;

/// `#[proxima::span]` — open a span without the metric/log framing
/// `#[proxima::instrument]` adds. See `proxima_macros::span`.
#[cfg(any(feature = "macros", test))]
pub use proxima_macros::span;

/// `pipe!(closure)` — the function-like leaf-lift sibling of
/// `#[proxima::piped]`, for an expression position. The attribute macro is
/// named `piped` (not `pipe`) precisely so this bang macro can keep the bare
/// `pipe` identifier — Rust's macro namespace does not distinguish `#[pipe]`
/// from `pipe!(..)` by invocation syntax, so the two would still collide
/// (E0428) if both were exported under it. See `proxima_macros::pipe`.
#[cfg(any(feature = "macros", test))]
pub use proxima_macros::pipe;

/// `filter!(predicate closure)` — lift a closure into the decision-pipe
/// shape (`In -> Result<In, Err>`) `pipe::filter`'s module doc describes.
/// See `proxima_macros::filter`.
#[cfg(any(feature = "macros", test))]
pub use proxima_macros::filter;

/// `fanout!(a, b, ..)` — variadic `FanOut` builder over closure and/or
/// pipe-expression arms. See `proxima_macros::fanout`.
#[cfg(any(feature = "macros", test))]
pub use proxima_macros::fanout;

/// `fanin!(a, b, ..)` — variadic `FanIn` builder over closure and/or
/// pipe-expression arms. See `proxima_macros::fanin`.
#[cfg(any(feature = "macros", test))]
pub use proxima_macros::fanin;

/// Runtime + cassette plumbing the `#[proxima::test]` expansion calls into.
#[cfg(any(test, feature = "test-support", feature = "macros"))]
pub mod test_support;

/// Cassette rot policy: rerecord/duplicate/staleness knobs + hooks.
#[cfg(any(test, feature = "test-support", feature = "macros"))]
pub mod cassette_config;

pub use app_builder::AppBuilder;
pub use body::{ChunkStream, RequestStream, ResponseStream};
pub use buffer_pool::{BufferPool, DEFAULT_BUFFER_BYTES, DEFAULT_POOL_PER_WORKER, PooledBuf};
pub use client::{
    Client, ClientProtocol, RequestBuilder as ClientRequestBuilder, Response as ClientResponse,
    Transport,
};
pub use codec::{BytesPassthrough, FrameCodec, JsonCodec, MessageCodec, StatefulCodec, WireCodec};
pub use codec_factory::{
    BytesPassthroughCodecFactory, BytesPassthroughDynCodec, CodecBuildFuture, CodecFactory,
    CodecRegistry, DynCodec, DynCodecFactory, DynCodecHandle, JsonCodecFactory, JsonDynCodec,
};
pub use config_format::{
    ConfigFormatFactory, ConfigFormatRegistry, DynConfigFormatFactory, Json5ConfigFormat,
    JsonConfigFormat, RonConfigFormat, TomlConfigFormat, XmlConfigFormat, YamlConfigFormat,
    default_config_format_registry,
};
pub use control_plane::{
    ControlPlane, ControlPlanePipe, DynControlPlane, PipeState, PipeStatus, StaticControlPlane,
};
pub use daemon_control_plane::{DaemonControlPlane, PipeConfig};
pub use error::{ProximaError, ProximaResult};
// the entire public surface is Request<Bytes>/Response<Bytes>; re-export Bytes
// so callers don't need a direct `bytes` dep to write `Bytes::from_static`.
pub use bytes::Bytes;
pub use header_list::{HeaderList, IntoHeaderBytes};
pub use kv::{CacheEntry, EvictionPolicy, KvCaps, KvHandle};
pub use listen::{
    ListenProtocol, ListenProtocolFluent, ListenRegistry, ServeBuilder, ServeContext,
    ThreadLocalListenProtocol, ThreadLocalListenRegistry,
};
pub use listen_handle::{Listener, ListenerHandle, ListenerSpec, ShutdownPolicy};
pub use listener::{ListenerBuilder, ListenerBuilderEntry};
#[cfg(feature = "tokio")]
pub use listeners::McpListenProtocol;
#[cfg(any(feature = "http1", feature = "http1-native"))]
pub use listeners::{HttpListenProtocol, serve_h1_connection};
pub use load::{LoadContext, Spec, load};
pub use log_buffer::{DEFAULT_LOG_BUFFER_CAPACITY, LiveTailReceiver, LogBuffer, LogBufferRegistry};
pub use middlewares::auth::{Auth, AuthFactory};
pub use middlewares::diff::{Diff, diff_handle};
pub use middlewares::isolate::{Isolate, IsolateFactory};
pub use middlewares::rate_limit::{
    ExceededAction, KeyExtractor, KeyOf, RateLimit, RateLimitCaps, RateLimitFactory,
    TokenBucketConfig,
};
pub use middlewares::retry::{Retry, RetryBudget, RetryFactory, RetryPredicate};
pub use middlewares::transform::{RequestOp, ResponseOp, Transform, TransformFactory};
pub use middlewares::write_back::{WriteBack, WriteBackTarget};
pub use mount::{MethodFilter, Mount, Router};
pub use packet::{Packet, PacketListener, PacketListenerExt};
pub use path_pattern::PathPattern;
// `Handler`/`ThreadLocalHandler` are the served-Pipe rename (proxima-pipe
// TARGET 2): a blanket impl over `SendPipe<In=Request<Bytes>,
// Out=Response<Bytes>, Err=ProximaError>`, replacing the old `Pipe`/
// `ThreadLocalPipe` traits' `name()`/`background_tasks()` methods (deleted,
// see TARGET 3 — the mount-site label + `App::source` replace them). Callers
// that only bound on the trait (`fn f<H: Handler>`), never implemented it
// explicitly (the blanket now covers every qualifying `SendPipe`), name the
// served trait directly as `Handler`/`ThreadLocalHandler`; `impl Handler for
// X {}` sites were deleted at the source (redundant against the blanket, and
// a coherence conflict if kept).
pub use pipe::{
    Handler, PipeHandle, ThreadLocalHandler, ThreadLocalPipeHandle, into_handle,
    into_thread_local_handle,
};
pub use proxima_patterns::middleware::context_inject::ContextInjector;
pub use proxima_primitives::pipe::SendPipe;
pub use recording::{
    AccumulatingSink, AppendFuture as RecordingAppendFuture, AppendLog, BinSource,
    BinSourceFactory, BoundedRecordingSink, CacheOutcome, DeferredRuntime, DropReason,
    DynRecordingSink, DynRecordingSource, DynRecordingSourceFactory, EventSource, EventTap,
    FailMode, FanOut, FormatKind, HttpEvent, INDEX_RECORD_BYTES, IndexReader, IndexRecord,
    IndexWriter, InteractionId, JsonlSource, JsonlSourceFactory, LazyFanOut, PipelineEvent,
    PipelineOutcome, ProcessEvent, ProtocolEvent, ProtocolRenderer, RECORD_DROP_METRIC,
    RECORDING_FORMAT_VERSION, RecordMeta, RecordingEvent, RecordingEventStream, RecordingSink,
    RecordingSource, RecordingSourceFactory, RecordingSourceRegistry, ReplayLog,
    RequestHeader as RecordingRequestHeader, SinkSpec, SourceBuildFuture, TerminalSignal,
    deferred_runtime,
};
pub use request::{Request, RequestBuilder, RequestContext, Response};
pub use proxima_primitives::pipe::RoutingPipe;
pub use scenarios::{CompareOp, Expectation, OrchestrationMode, Scenario, ScenarioPipeSpec, WorkloadSpec};
#[cfg(feature = "tokio")]
pub use scenarios::{ScenarioReport, run_scenario};
pub use schema::{
    EmptyResolver, EnumVariant, FieldFlags, PathSegment, Schema, SchemaRegistry, SchemaResolver,
    StringFormat, StructField, ValidationError,
};
pub use selection::{
    DispatchOutcome, DynSelection, Fallthrough, LeastConn, MissPolicy, MissReason, RoundRobin,
    Selection, SelectionHandle, ThreadLocalDynSelection, ThreadLocalSelection,
    ThreadLocalSelectionHandle, WeightedLeastConn, WeightedRoundRobin,
};
pub use swap::{SwapRegistry, SwappablePipe};

pub use pipe_factory::{DynPipeFactory, PipeFactory, PipeFactoryRegistry};
#[cfg(feature = "tokio")]
pub use pipelines::{
    DynPipelineControlPlane, FsPipelineControlPlane, InMemoryPipelineControlPlane,
    PipelineControlPlane, PipelineControlPlanePipe, PipelineExecutor, PipelineRunReport,
    PipelineSpec, PipelineStatus, PipelineSubmission, PipelineSummary, StageReport, StageSpec,
};
pub use settings::{BearerAuth, ClientAuth, Composable, HttpListener, HttpUpstream, OauthAuth};
#[cfg(feature = "http1")]
pub use shared_http::{SharedHttpClient, SharedHyperClient};
pub use stream::{
    Accept, BindAddr, Connect, PeerInfo, StreamConnection, StreamListener, StreamListenerExt,
    StreamUpstream, StreamUpstreamExt,
};
pub use sugar::desugar;
// the fluent builder seam + axis sugar traits — `use proxima::{ProtocolSugar,
// TransportSugar}` to light up `.http()`/`.tcp()`/… on any spec builder.
pub use sugar::{ProtocolSugar, SpecBuilder, TransportSugar};
pub use proxima_primitives::transport;
pub use proxima_primitives::transport::{
    DEFAULT_REPLAY_CAP_BYTES, Replay, ReplayEvent, tap_complete, tap_complete_with_size,
};
pub use telemetry::{
    HistogramSummary, Labels, Metrics, MetricsSnapshot, NoopTelemetry, Telemetry, TelemetryHandle,
};
pub use templates::{TemplateContext, expand as expand_template};
pub use tracing_init::{LogFormat, init_tracing, init_tracing_default};
pub use upgrade::{
    HijackStream, HijackedSocket, LocalHijackStream, LocalHijackedSocket, LocalUpgradeFuture,
    LocalUpgradeHandler, UpgradeFuture, UpgradeHandler,
};
pub use upstream_ref::{
    CallTracker, OutlierPolicy, ThreadLocalUpstreamRef, UpstreamMetrics, UpstreamRef,
};
pub use upstreams::{
    CallbackFn, CallbackFuture, CallbackPipeFactory, CallbackRegistry, CallbackUpstream,
    DynCallbackFn, KvCache, KvCacheFactory, KvFile, KvFileFactory, RecordPipeFactory,
    RecordUpstream, ReplayPipeFactory, ReplayUpstream, SynthPipeFactory, SynthUpstream,
};
#[cfg(feature = "tokio")]
pub use upstreams::{
    ProcessPipeFactory, ProcessRpcPipeFactory, ProcessRpcSpec, ProcessRpcUpstream, ProcessSpec,
    ProcessUpstream, ReadyProbe, RestartPolicy, ShutdownSignal,
};
pub use write_back::{WriteBackConditions, WriteBackRule};

/// Drive a future to completion — expression sugar over the `block_on` verb.
/// Two arities, both pointing down to a concrete `block_on`:
/// - `block_on!(fut)` — the no-runtime [`block_on`](crate::block_on) poll loop
///   (drives on the calling thread, no runtime, no reactor).
/// - `block_on!(rt, fut)` — the runtime-holding
///   [`block_on`](crate::runtime::block_on): drives `fut` on core 0 of the
///   runtime `rt` you already HOLD (foreign-thread entry — see its doc).
///
/// `run!` is the sibling that BOOTS a runtime first. Reach for `block_on!`
/// when you already hold a runtime (or need none); reach for `run!` at an edge.
#[macro_export]
macro_rules! block_on {
    ($rt:expr, $fut:expr) => {
        $crate::runtime::block_on(&$rt, $fut)
    };
    ($fut:expr) => {
        $crate::block_on($fut)
    };
}

/// Boot a runtime, then drive `fut` to completion on it — expression sugar for
/// the edge `run*` drivers (`#[proxima::main]` is the attribute form). Every
/// arm points down to a `crate::runtime::run*` driver:
/// - `run!(fut)` — adaptive [`run`](crate::runtime::run) (prime when compiled,
///   else tokio).
/// - `run!(prime, fut)` — [`run_prime`](crate::runtime::run_prime).
/// - `run!(tokio, fut)` — [`run_tokio`](crate::runtime::run_tokio), the
///   multi-thread tokio drive the adaptive path falls back to.
#[macro_export]
macro_rules! run {
    (prime, $fut:expr) => {
        $crate::runtime::run_prime($fut)
    };
    (tokio, $fut:expr) => {
        $crate::runtime::run_tokio(true, ::core::option::Option::None, $fut)
    };
    ($fut:expr) => {
        $crate::runtime::run($fut)
    };
}

#[cfg(test)]
#[allow(clippy::disallowed_methods, clippy::unwrap_used, clippy::expect_used)]
mod drive_macro_tests {
    #[test]
    fn block_on_bang_no_runtime_drives_to_completion() {
        let output: u32 = crate::block_on!(async { 40 + 2 });
        assert_eq!(output, 42);
    }

    #[test]
    fn run_bang_adaptive_boots_and_drives() {
        let output = crate::run!(async { 40 + 2 }).expect("adaptive run drives the body");
        assert_eq!(output, 42);
    }

    #[cfg(feature = "tokio")]
    #[test]
    fn run_bang_tokio_arm_boots_and_drives() {
        let output = crate::run!(tokio, async { 40 + 2 }).expect("tokio run drives the body");
        assert_eq!(output, 42);
    }

    #[cfg(all(
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    ))]
    #[test]
    fn run_bang_prime_arm_boots_and_drives() {
        let output = crate::run!(prime, async { 40 + 2 }).expect("prime run drives the body");
        assert_eq!(output, 42);
    }

    #[cfg(all(
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    ))]
    #[test]
    fn block_on_bang_runtime_arm_drives_on_held_runtime() {
        let runtime = crate::runtime::PrimeRuntime::new(1).expect("build a one-core prime runtime");
        let output = crate::block_on!(runtime, async { 40 + 2 })
            .expect("block_on drives the body on the held runtime");
        assert_eq!(output, 42);
    }
}
