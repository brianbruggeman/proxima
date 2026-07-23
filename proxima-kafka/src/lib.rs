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
//! The `listen` feature adds the server side:
//! [`frame_codec::KafkaCodec`] (the sans-IO `FrameCodec` +
//! `OwnFrame`/`Incomplete` impl over [`wire`]'s Produce/Fetch/Metadata/
//! ApiVersions body codec), [`framed_app::KafkaFramedApp`] (the
//! business-handler pipe wired as `proxima_listen::any::FramedAny`'s
//! `App`), and [`any_protocol::KafkaAnyProtocol`] — the `AnyProtocol`
//! candidate that mounts kafka into the open universal listener
//! (`Listener::builder().accept("kafka")`) by building a `FramedAny`
//! internally. There is no bespoke per-connection I/O driver here
//! anymore (no `connection::serve_connection`, no
//! `pipe::KafkaConnectionPipe` CONNECT-and-upgrade indirection) —
//! `proxima_listen::any::FramedAny` is the ONE generic stateless
//! `AnyProtocol` driver every stateless request/reply wire shares; see
//! `framed_app`'s module doc for how Kafka's `ApiVersions`/violation
//! semantics map onto its `AsFrame` seam. [`broker::KafkaBroker`] is the
//! default Produce/Fetch/Metadata handler a caller plugs in as
//! `KafkaAnyProtocol::new(label, into_kafka_handle(broker))` — a
//! protocol-correct broker FACADE (in-memory per-topic-partition log,
//! batch-granularity offsets, a real if per-partition-sequential Fetch
//! long-poll), not a production Kafka broker. There is no standalone
//! `KafkaListenProtocol` bind+accept loop, mirroring redis's own shape.
//!
//! ## Scope
//!
//! **API version 0 only, record sets carried opaque.** [`wire`] decodes
//! exactly the v0 shape of Produce/Fetch/Metadata/ApiVersions
//! ([`wire::SUPPORTED_API_VERSIONS`] advertises `(api_key, 0, 0)` for every
//! key) — no compression, no transactional/idempotent-producer fields, no
//! v1+ schema evolution. `record_set` (the actual message batch bytes
//! inside a Produce/Fetch body) is carried and stored as an opaque
//! [`bytes::Bytes`] blob, never parsed into individual records
//! ([`broker::KafkaBroker`]'s doc: "one opaque `record_set` blob, appended
//! whole"); this facade routes and replays record sets, it does not
//! decode `RecordBatch` framing, per-record headers, or compression
//! codecs. Not a substitute for a real broker's replication, transactions,
//! or consumer-group coordination — those protocols are unimplemented.

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
pub mod frame_codec;
#[cfg(feature = "listen")]
pub mod framed_app;
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
pub use frame_codec::{KafkaCodec, KafkaCodecError, KafkaFrame, KafkaOwnedFrame, Violation};
#[cfg(feature = "listen")]
pub use framed_app::{KafkaAppError, KafkaFramedApp, KafkaOutcome};
#[cfg(feature = "listen")]
pub use pipes::{KafkaPipeHandle, KafkaPipeReply, KafkaPipeRequest, into_kafka_handle};
