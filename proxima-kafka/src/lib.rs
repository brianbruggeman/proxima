//! proxima's own Kafka client + broker-facade listener.
//!
//! The sans-IO wire envelope ([`proxima_protocols::kafka`]: the
//! `[i32 length][payload]` frame plus the `api_key`/`api_version`/
//! `correlation_id`/`client_id` request header) is lifted, not
//! reimplemented — see that module's docs. [`wire`] is this crate's own
//! addition: the Produce/Fetch/Metadata/ApiVersions v0 BODY codec
//! `proxima_protocols::kafka` stops short of (that module parses the
//! header and hands back an opaque body slice; nothing upstream decodes
//! what a given `api_key` means).
//!
//! The `client` feature adds the std client built on top: the async
//! [`client::KafkaClientUpstream`] Pipe and the blocking
//! [`client::KafkaClient`] driver, both driving the sans-IO
//! [`client::ClientSession`] over a pluggable transport (prime, tokio,
//! TLS-wrapped) — the same split `proxima-redis` uses between
//! `proxima-protocols::redis` and its own client.
//!
//! The `listen` feature adds the server side: [`connection`]'s
//! sans-IO-over-any-`futures::io`-stream driver, [`pipe::KafkaConnectionPipe`]
//! (the connection layer as a real `Pipe`), and
//! [`any_protocol::KafkaAnyProtocol`] — the `AnyProtocol` candidate that
//! mounts kafka into the open universal listener
//! (`Listener::builder().accept("kafka")`). [`broker::KafkaBroker`] is the
//! default Produce/Fetch/Metadata handler a caller plugs in as
//! `KafkaAnyProtocol::new(label, into_kafka_handle(broker))` — a
//! protocol-correct broker FACADE (in-memory per-topic-partition log,
//! batch-granularity offsets, a real if per-partition-sequential Fetch
//! long-poll), not a production Kafka broker. There is no standalone
//! `KafkaListenProtocol` bind+accept loop, mirroring redis's own shape.

#[cfg(feature = "client")]
pub mod client;

pub mod wire;

#[cfg(feature = "listen")]
pub mod any_protocol;
#[cfg(feature = "listen")]
pub mod broker;
#[cfg(feature = "listen")]
pub mod config;
#[cfg(feature = "listen")]
pub mod connection;
#[cfg(feature = "listen")]
pub mod error;
#[cfg(feature = "listen")]
pub mod pipe;
#[cfg(feature = "listen")]
pub mod pipes;

pub use wire::{
    ApiKey, ApiVersionRange, ApiVersionsResponse, FetchPartitionData, FetchPartitionResult,
    FetchRequest, FetchResponse, FetchTopicData, FetchTopicResult, MetadataBroker,
    MetadataPartition, MetadataRequest, MetadataResponse, MetadataTopic, ProducePartitionData,
    ProducePartitionResult, ProduceRequest, ProduceResponse, ProduceTopicData, ProduceTopicResult,
    RequestBody, ResponseBody, SUPPORTED_API_VERSIONS, WireError, error_code,
};

#[cfg(feature = "client")]
pub use client::{
    ClientError, ClientSession, KafkaClient, KafkaClientConfig, KafkaClientUpstream,
    KafkaConfigError, Step,
};

// the server-side surface a kafka handler builds against — re-exported so
// an engine author imports everything from proxima-kafka and never
// reaches past it into proxima-primitives/proxima-protocols internals
// (teaching surface, workspace principle 2), mirroring proxima-redis's own
// top-level re-export shape.
#[cfg(feature = "listen")]
pub use any_protocol::KafkaAnyProtocol;
#[cfg(feature = "listen")]
pub use broker::KafkaBroker;
#[cfg(feature = "listen")]
pub use config::KafkaServerConfig;
#[cfg(feature = "listen")]
pub use connection::serve_connection;
#[cfg(feature = "listen")]
pub use error::KafkaServeError;
#[cfg(feature = "listen")]
pub use pipe::KafkaConnectionPipe;
#[cfg(feature = "listen")]
pub use pipes::{KafkaPipeHandle, KafkaPipeReply, KafkaPipeRequest, into_kafka_handle};
