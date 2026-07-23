#[cfg(all(
    feature = "amqp-client",
    any(target_os = "linux", target_os = "macos")
))]
pub mod amqp;
pub mod callback;
pub mod callback_registry;
#[cfg(all(feature = "dns-client", any(target_os = "linux", target_os = "macos")))]
pub mod dns;
pub mod fs;
#[cfg(all(
    feature = "http-prime",
    feature = "http2",
    any(target_os = "linux", target_os = "macos")
))]
pub mod grpc_h2;
#[cfg(all(
    feature = "kafka-client",
    any(target_os = "linux", target_os = "macos")
))]
pub mod kafka;
#[cfg(all(
    feature = "memcached-client",
    any(target_os = "linux", target_os = "macos")
))]
pub mod memcached;
#[cfg(all(feature = "mqtt-client", any(target_os = "linux", target_os = "macos")))]
pub mod mqtt;
#[cfg(all(
    feature = "pgwire-client",
    any(target_os = "linux", target_os = "macos")
))]
pub mod pgwire;
#[cfg(all(
    feature = "redis-client",
    any(target_os = "linux", target_os = "macos")
))]
pub mod redis;
#[cfg(feature = "http1")]
pub use proxima_http::http1::upstream as http;
pub mod kv_cache;
pub mod kv_file;
pub mod kv_upstream;
// child-process upstreams: tokio::process::Command supervision +
// tokio::spawn + tokio::sync (mpsc/watch/Mutex/JoinHandle) — a genuine
// tokio::process capability with no prime equivalent today.
#[cfg(feature = "tokio")]
pub mod process;
#[cfg(feature = "tokio")]
pub mod process_rpc;
pub mod record;
pub use proxima_recording::replay;
#[cfg(all(
    feature = "h3-native-upstream",
    any(target_os = "linux", target_os = "macos")
))]
pub mod h3_native;
#[cfg(any(feature = "tcp", feature = "unix"))]
pub mod stream_passthrough;
pub mod synth;
#[cfg(feature = "h3-upstream")]
pub use proxima_http::http3::upstream as h3;
#[cfg(any(feature = "tcp", feature = "unix"))]
pub use proxima_net::tokio::tokio_stream_upstream as tokio_stream;
#[cfg(feature = "websocket-upstream")]
pub use proxima_http::websocket::upstream as websocket;

#[cfg(all(
    feature = "amqp-client",
    any(target_os = "linux", target_os = "macos")
))]
pub use amqp::{AmqpClientProtocol, AmqpPipeFactory};
pub use callback::{CallbackPipeFactory, CallbackUpstream};
pub use callback_registry::{CallbackFn, CallbackFuture, CallbackRegistry, DynCallbackFn};
#[cfg(all(feature = "dns-client", any(target_os = "linux", target_os = "macos")))]
pub use dns::{DnsClientProtocol, DnsPipeFactory};
pub use fs::{FsPipeFactory, FsUpstream};
#[cfg(all(
    feature = "http-prime",
    feature = "http2",
    any(target_os = "linux", target_os = "macos")
))]
pub use grpc_h2::GrpcH2PipeFactory;
#[cfg(feature = "h3-upstream")]
pub use h3::Http3Upstream;
#[cfg(all(
    feature = "h3-native-upstream",
    any(target_os = "linux", target_os = "macos")
))]
pub use h3_native::H3NativeUpstreamFactory;
#[cfg(feature = "http1")]
pub use http::{HttpPipeFactory, HttpUpstream};
#[cfg(all(
    feature = "kafka-client",
    any(target_os = "linux", target_os = "macos")
))]
pub use kafka::{KafkaClientProtocol, KafkaPipeFactory};
pub use kv_cache::{KvCache, KvCacheFactory};
pub use kv_file::{KvFile, KvFileFactory};
pub use kv_upstream::KvUpstream;
#[cfg(all(
    feature = "memcached-client",
    any(target_os = "linux", target_os = "macos")
))]
pub use memcached::{MemcachedClientProtocol, MemcachedPipeFactory};
#[cfg(all(feature = "mqtt-client", any(target_os = "linux", target_os = "macos")))]
pub use mqtt::{MqttClientProtocol, MqttPipeFactory};
#[cfg(all(
    feature = "pgwire-client",
    any(target_os = "linux", target_os = "macos")
))]
pub use pgwire::{PgwireClientProtocol, PgwirePipeFactory};
#[cfg(feature = "tokio")]
pub use process::{
    ProcessPipeFactory, ProcessSpec, ProcessUpstream, ReadyProbe, RestartPolicy, ShutdownSignal,
};
#[cfg(feature = "tokio")]
pub use process_rpc::{ProcessRpcPipeFactory, ProcessRpcSpec, ProcessRpcUpstream};
pub use record::{RecordPipeFactory, RecordUpstream};
#[cfg(all(
    feature = "redis-client",
    any(target_os = "linux", target_os = "macos")
))]
pub use redis::{RedisClientProtocol, RedisPipeFactory};
pub use replay::{ReplayPipeFactory, ReplayUpstream};
#[cfg(any(feature = "tcp", feature = "unix"))]
pub use stream_passthrough::{
    StreamPassthroughPipeFactory, StreamPassthroughSettings, StreamPassthroughUpstream,
};
pub use synth::{SynthPipeFactory, SynthUpstream};
#[cfg(feature = "tcp")]
pub use tokio_stream::TokioTcpUpstream;
#[cfg(all(feature = "unix", unix))]
pub use tokio_stream::TokioUnixUpstream;
#[cfg(feature = "websocket-upstream")]
pub use websocket::WebSocketUpstream;
