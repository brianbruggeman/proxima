//! `KafkaBroker` — the default Produce/Fetch/Metadata handler every
//! [`crate::any_protocol::KafkaAnyProtocol`] this facade drives can plug
//! in as its [`crate::pipes::KafkaPipeHandle`].
//!
//! Unlike `proxima_redis::broker::RedisBroker` (protocol-level PUBLISH/
//! SUBSCRIBE bookkeeping sitting ALONGSIDE an arbitrary, separately-plugged
//! business-command handler), Kafka has no such split: Produce/Fetch/
//! Metadata already ARE the entire recognized wire surface, so the broker
//! itself is a [`proxima_primitives::pipe::SendPipe`] — an ordinary handler
//! (workspace principle 1: no bespoke "broker + handler" duality when one
//! pipe already expresses it) — that a caller wires in as
//! `KafkaAnyProtocol::new(label, into_kafka_handle(broker))`, or replaces
//! entirely with their own engine (forwarding to a real Kafka cluster, a
//! custom storage backend, ...).
//!
//! Two pieces of shared state, mirroring `RedisBroker`'s own split between
//! durable bookkeeping (its `channels`/`patterns` registries) and live
//! delivery (`KeyedFanOut::publish`):
//! - a durable per-topic-partition log ([`Live`]-backed copy-on-write
//!   `BTreeMap`, the same discipline [`KeyedFanOut`] itself uses for its
//!   registry) — batch-granularity offsets: one [`Self::produce`] call is
//!   one opaque `record_set` blob, appended whole; this facade never
//!   decodes individual records inside it, so the offset it hands back
//!   addresses the BATCH's position in the partition's log, not a
//!   record within it (documented simplification — a real Kafka broker's
//!   offsets are per-record).
//! - a [`KeyedFanOut`] of per-partition wake pings, published on every
//!   [`Self::produce`] — [`Self::fetch`] subscribes to the target
//!   partition's key and races it against a `max_wait_ms` timer when the
//!   log has nothing yet, giving Fetch's long-poll semantics real
//!   (if per-partition-sequential, not whole-request-fanned) teeth instead
//!   of a bare non-blocking read.

use std::sync::Arc;
use std::time::Duration;

use futures::FutureExt;
use futures::channel::mpsc;
use futures::stream::StreamExt;

use proxima_core::ProximaError;
use proxima_core::live::{Live, LiveControl, live};
use proxima_primitives::pipe::request::Response;
use proxima_primitives::pipe::{BestEffort, KeyedFanOut, SendPipe};

use crate::config::KafkaServerConfig;
use crate::pipes::{KafkaPipeReply, KafkaPipeRequest};
use crate::wire::{
    self, FetchPartitionResult, FetchResponse, FetchTopicResult, MetadataBroker, MetadataPartition,
    MetadataResponse, MetadataTopic, ProducePartitionResult, ProduceResponse, ProduceTopicResult,
    RequestBody, ResponseBody, error_code,
};

/// `topic\0partition` — the byte key both the durable log map and the wake
/// [`KeyedFanOut`] index by. NUL is a safe delimiter: real Kafka topic
/// names are restricted to `[a-zA-Z0-9._-]`.
fn partition_key(topic: &str, partition: i32) -> Vec<u8> {
    let mut key = Vec::with_capacity(topic.len() + 5);
    key.extend_from_slice(topic.as_bytes());
    key.push(0);
    key.extend_from_slice(&partition.to_be_bytes());
    key
}

type PartitionLog = std::collections::BTreeMap<Vec<u8>, Vec<bytes::Bytes>>;

/// A wake ping's sink half — a lightweight [`SendPipe`] over an unbounded
/// `()` channel, adapted the same way `RedisBroker::PushSink` adapts its
/// own `UnboundedSender` (workspace principle 1: a sink is an ordinary
/// pipe, no bespoke sink trait).
#[derive(Clone)]
struct WakeSink(mpsc::UnboundedSender<()>);

impl SendPipe for WakeSink {
    type In = ();
    type Out = ();
    type Err = ProximaError;

    fn call(&self, (): ()) -> impl core::future::Future<Output = Result<(), ProximaError>> + Send {
        let result = self.0.unbounded_send(());
        async move {
            result
                .map_err(|error| ProximaError::Upstream(format!("kafka wake sink closed: {error}")))
        }
    }
}

/// In-memory Produce/Fetch/Metadata broker facade. Construct once and share
/// via `Arc`, or wrap in [`crate::pipes::into_kafka_handle`] to plug it
/// directly into [`crate::any_protocol::KafkaAnyProtocol::new`].
pub struct KafkaBroker {
    logs: Live<PartitionLog>,
    logs_control: LiveControl<PartitionLog>,
    wakers: KeyedFanOut<WakeSink, BestEffort>,
    config: Arc<KafkaServerConfig>,
}

impl KafkaBroker {
    #[must_use]
    pub fn new(config: Arc<KafkaServerConfig>) -> Self {
        let (logs, logs_control) = live(PartitionLog::new());
        Self {
            logs,
            logs_control,
            wakers: KeyedFanOut::new(),
            config,
        }
    }

    /// Appends `record_set` to `topic`/`partition`'s log, returning the
    /// batch-granularity base offset (the log's length before the append).
    /// Wakes any [`Self::fetch`] currently long-polling this partition.
    pub async fn produce(&self, topic: &str, partition: i32, record_set: bytes::Bytes) -> i64 {
        let key = partition_key(topic, partition);
        // `LiveControl::update`'s closure must be `Fn` (it may be retried
        // under CAS contention) — a `Cell` lets the winning retry record
        // its own offset through a shared reference instead of needing a
        // captured `&mut i64`, which would force `FnMut`.
        let base_offset = core::cell::Cell::new(0_i64);
        self.logs_control.update(|current| {
            let mut next = current.clone();
            let entries = next.entry(key.clone()).or_default();
            base_offset.set(entries.len() as i64);
            entries.push(record_set.clone());
            next
        });
        // best-effort: no subscriber (nobody long-polling this partition
        // right now) is a harmless no-op, same as `RedisBroker::publish`
        // against an unsubscribed channel.
        let _ = self.wakers.publish(&key, ()).await;
        base_offset.get()
    }

    /// Batches at `fetch_offset` or later, concatenated (real Kafka's own
    /// wire format already allows back-to-back `RecordBatch`es in one
    /// `record_set`, so concatenation stays wire-plausible even though
    /// this facade never decodes individual records), plus the partition's
    /// current high watermark (its log length).
    fn read_from_offset(
        &self,
        topic: &str,
        partition: i32,
        fetch_offset: i64,
    ) -> (i64, bytes::Bytes) {
        let key = partition_key(topic, partition);
        self.logs.read(|logs| {
            let Some(entries) = logs.get(&key) else {
                return (0, bytes::Bytes::new());
            };
            let high_watermark = entries.len() as i64;
            let start = usize::try_from(fetch_offset.max(0)).unwrap_or(usize::MAX);
            if start >= entries.len() {
                return (high_watermark, bytes::Bytes::new());
            }
            let mut combined = Vec::new();
            for batch in &entries[start..] {
                combined.extend_from_slice(batch);
            }
            (high_watermark, bytes::Bytes::from(combined))
        })
    }

    /// Serves one partition's `Fetch`: an immediate read if data is
    /// already there, otherwise one bounded long-poll wait (races a wake
    /// ping from [`Self::produce`] against a `max_wait_ms` timer) before
    /// answering with whatever is available — real Kafka's own contract
    /// (never blocks past `max_wait_ms`, never guarantees `min_bytes` was
    /// actually reached).
    async fn fetch_partition(
        &self,
        topic: &str,
        partition: i32,
        fetch_offset: i64,
        max_wait_ms: i32,
    ) -> (i64, bytes::Bytes) {
        let (high_watermark, record_set) = self.read_from_offset(topic, partition, fetch_offset);
        if !record_set.is_empty() || max_wait_ms <= 0 {
            return (high_watermark, record_set);
        }

        let key = partition_key(topic, partition);
        let (tx, mut rx) = mpsc::unbounded();
        let subscription = self.wakers.subscribe(key.clone(), WakeSink(tx));
        let timeout = futures_timer::Delay::new(Duration::from_millis(max_wait_ms as u64));
        futures::select_biased! {
            _ = rx.next().fuse() => {}
            () = timeout.fuse() => {}
        }
        self.wakers.unsubscribe(&key, subscription);
        self.read_from_offset(topic, partition, fetch_offset)
    }

    /// This facade's own `Metadata` truth: it always answers as the single
    /// broker it is, and reports every partition a topic has EVER had
    /// data produced to (a topic nobody has produced to yet is simply
    /// absent — real Kafka's own auto-create-on-produce-or-metadata
    /// behavior, simplified to produce-only here).
    fn metadata(&self, requested: Option<&[String]>) -> MetadataResponse {
        let broker = MetadataBroker {
            node_id: self.config.broker_id,
            host: self.config.advertised_host.clone(),
            port: self.config.advertised_port,
        };
        let known_topics: Vec<(String, i32)> = self.logs.read(|logs| {
            logs.keys()
                .filter_map(|key| {
                    let nul = key.iter().position(|byte| *byte == 0)?;
                    let (topic_bytes, rest) = key.split_at(nul);
                    let partition_bytes = &rest[1..];
                    let partition = i32::from_be_bytes(partition_bytes.try_into().ok()?);
                    Some((String::from_utf8_lossy(topic_bytes).into_owned(), partition))
                })
                .collect()
        });

        let mut topics: std::collections::BTreeMap<String, Vec<i32>> =
            std::collections::BTreeMap::new();
        for (topic, partition) in known_topics {
            topics.entry(topic).or_default().push(partition);
        }

        let names: Vec<String> = match requested {
            Some(names) => names.to_vec(),
            None => topics.keys().cloned().collect(),
        };

        let topic_metadata = names
            .into_iter()
            .map(|topic| match topics.get(&topic) {
                Some(partitions) => MetadataTopic {
                    error_code: error_code::NONE,
                    topic: topic.clone(),
                    partitions: partitions
                        .iter()
                        .map(|&partition_id| MetadataPartition {
                            error_code: error_code::NONE,
                            partition_id,
                            leader: self.config.broker_id,
                            replicas: vec![self.config.broker_id],
                            isr: vec![self.config.broker_id],
                        })
                        .collect(),
                },
                None => MetadataTopic {
                    error_code: wire::error_code::UNKNOWN_SERVER_ERROR,
                    topic,
                    partitions: Vec::new(),
                },
            })
            .collect();

        MetadataResponse {
            brokers: vec![broker],
            topics: topic_metadata,
        }
    }
}

impl SendPipe for KafkaBroker {
    type In = KafkaPipeRequest;
    type Out = KafkaPipeReply;
    type Err = ProximaError;

    async fn call(&self, request: KafkaPipeRequest) -> Result<KafkaPipeReply, ProximaError> {
        match request.payload {
            RequestBody::Produce(produce) => {
                let mut topics = Vec::with_capacity(produce.topics.len());
                for topic_data in produce.topics {
                    let mut partitions = Vec::with_capacity(topic_data.partitions.len());
                    for partition_data in topic_data.partitions {
                        let base_offset = self
                            .produce(
                                &topic_data.topic,
                                partition_data.partition,
                                partition_data.record_set,
                            )
                            .await;
                        partitions.push(ProducePartitionResult {
                            partition: partition_data.partition,
                            error_code: error_code::NONE,
                            base_offset,
                        });
                    }
                    topics.push(ProduceTopicResult {
                        topic: topic_data.topic,
                        partitions,
                    });
                }
                Ok(Response::typed(
                    200,
                    ResponseBody::Produce(ProduceResponse { topics }),
                ))
            }
            RequestBody::Fetch(fetch) => {
                let mut topics = Vec::with_capacity(fetch.topics.len());
                for topic_data in fetch.topics {
                    let mut partitions = Vec::with_capacity(topic_data.partitions.len());
                    for partition_data in topic_data.partitions {
                        let (high_watermark, record_set) = self
                            .fetch_partition(
                                &topic_data.topic,
                                partition_data.partition,
                                partition_data.fetch_offset,
                                fetch.max_wait_ms,
                            )
                            .await;
                        partitions.push(FetchPartitionResult {
                            partition: partition_data.partition,
                            error_code: error_code::NONE,
                            high_watermark,
                            record_set,
                        });
                    }
                    topics.push(FetchTopicResult {
                        topic: topic_data.topic,
                        partitions,
                    });
                }
                Ok(Response::typed(
                    200,
                    ResponseBody::Fetch(FetchResponse { topics }),
                ))
            }
            RequestBody::Metadata(metadata) => {
                let requested = metadata.topics;
                let response = self.metadata(requested.as_deref());
                Ok(Response::typed(200, ResponseBody::Metadata(response)))
            }
            RequestBody::ApiVersions => {
                // protocol-level in `crate::framed_app::KafkaFramedApp` — a
                // handler should never see this variant, but answering it
                // here too keeps `KafkaBroker` correct standing alone
                // (e.g. driven directly in a unit test, bypassing the
                // connection driver).
                Ok(Response::typed(
                    200,
                    ResponseBody::ApiVersions(wire::ApiVersionsResponse::supported()),
                ))
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_primitives::pipe::method::Method;
    use proxima_primitives::pipe::request::{Request, RequestContext};

    fn broker() -> KafkaBroker {
        KafkaBroker::new(Arc::new(KafkaServerConfig::default()))
    }

    fn produce_request(topic: &str, partition: i32, record_set: &[u8]) -> KafkaPipeRequest {
        Request {
            method: Method::from_bytes(b"PRODUCE"),
            path: bytes::Bytes::new(),
            query: proxima_primitives::pipe::header_list::HeaderList::new(),
            metadata: proxima_primitives::pipe::header_list::HeaderList::new(),
            payload: RequestBody::Produce(wire::ProduceRequest {
                acks: 1,
                timeout_ms: 1000,
                topics: vec![wire::ProduceTopicData {
                    topic: topic.to_string(),
                    partitions: vec![wire::ProducePartitionData {
                        partition,
                        record_set: bytes::Bytes::copy_from_slice(record_set),
                    }],
                }],
            }),
            stream: None,
            context: RequestContext::default(),
        }
    }

    fn fetch_request(
        topic: &str,
        partition: i32,
        fetch_offset: i64,
        max_wait_ms: i32,
    ) -> KafkaPipeRequest {
        Request {
            method: Method::from_bytes(b"FETCH"),
            path: bytes::Bytes::new(),
            query: proxima_primitives::pipe::header_list::HeaderList::new(),
            metadata: proxima_primitives::pipe::header_list::HeaderList::new(),
            payload: RequestBody::Fetch(wire::FetchRequest {
                replica_id: -1,
                max_wait_ms,
                min_bytes: 1,
                topics: vec![wire::FetchTopicData {
                    topic: topic.to_string(),
                    partitions: vec![wire::FetchPartitionData {
                        partition,
                        fetch_offset,
                        max_bytes: 1_048_576,
                    }],
                }],
            }),
            stream: None,
            context: RequestContext::default(),
        }
    }

    #[proxima::test(runtime = "tokio")]
    async fn produce_then_fetch_from_offset_zero_returns_what_was_produced() {
        let broker = broker();
        let produced = broker
            .call(produce_request("orders", 0, b"hello"))
            .await
            .expect("produce");
        let ResponseBody::Produce(response) = produced.payload else {
            panic!("expected Produce");
        };
        assert_eq!(response.topics[0].partitions[0].base_offset, 0);

        let fetched = broker
            .call(fetch_request("orders", 0, 0, 0))
            .await
            .expect("fetch");
        let ResponseBody::Fetch(response) = fetched.payload else {
            panic!("expected Fetch");
        };
        assert_eq!(
            response.topics[0].partitions[0].record_set,
            bytes::Bytes::from_static(b"hello")
        );
        assert_eq!(response.topics[0].partitions[0].high_watermark, 1);
    }

    #[proxima::test(runtime = "tokio")]
    async fn second_produce_advances_the_base_offset_and_fetch_from_offset_one_sees_only_the_new_batch()
     {
        let broker = broker();
        broker
            .call(produce_request("orders", 0, b"first"))
            .await
            .expect("produce 1");
        let second = broker
            .call(produce_request("orders", 0, b"second"))
            .await
            .expect("produce 2");
        let ResponseBody::Produce(response) = second.payload else {
            panic!("expected Produce");
        };
        assert_eq!(response.topics[0].partitions[0].base_offset, 1);

        let fetched = broker
            .call(fetch_request("orders", 0, 1, 0))
            .await
            .expect("fetch");
        let ResponseBody::Fetch(response) = fetched.payload else {
            panic!("expected Fetch");
        };
        assert_eq!(
            response.topics[0].partitions[0].record_set,
            bytes::Bytes::from_static(b"second")
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn fetch_from_an_empty_partition_returns_an_empty_record_set_not_an_error() {
        let broker = broker();
        let fetched = broker
            .call(fetch_request("nobody-produced-here", 0, 0, 0))
            .await
            .expect("fetch");
        let ResponseBody::Fetch(response) = fetched.payload else {
            panic!("expected Fetch");
        };
        assert!(response.topics[0].partitions[0].record_set.is_empty());
        assert_eq!(response.topics[0].partitions[0].high_watermark, 0);
    }

    #[proxima::test(runtime = "tokio")]
    async fn long_poll_fetch_wakes_on_a_concurrent_produce_instead_of_waiting_the_full_timeout() {
        let broker = Arc::new(broker());
        let waiter = {
            let broker = Arc::clone(&broker);
            tokio::spawn(async move { broker.call(fetch_request("orders", 0, 0, 5_000)).await })
        };
        // give the fetch a moment to reach its long-poll wait before producing
        tokio::time::sleep(Duration::from_millis(50)).await;
        broker
            .call(produce_request("orders", 0, b"woke-me-up"))
            .await
            .expect("produce");

        let fetched = tokio::time::timeout(Duration::from_secs(2), waiter)
            .await
            .expect("fetch must return well before the 5s max_wait_ms timeout")
            .expect("join")
            .expect("fetch");
        let ResponseBody::Fetch(response) = fetched.payload else {
            panic!("expected Fetch");
        };
        assert_eq!(
            response.topics[0].partitions[0].record_set,
            bytes::Bytes::from_static(b"woke-me-up")
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn metadata_reports_only_topics_with_produced_data() {
        let broker = broker();
        broker
            .call(produce_request("orders", 0, b"x"))
            .await
            .expect("produce");

        let request = Request {
            method: Method::from_bytes(b"METADATA"),
            path: bytes::Bytes::new(),
            query: proxima_primitives::pipe::header_list::HeaderList::new(),
            metadata: proxima_primitives::pipe::header_list::HeaderList::new(),
            payload: RequestBody::Metadata(wire::MetadataRequest { topics: None }),
            stream: None,
            context: RequestContext::default(),
        };
        let response = broker.call(request).await.expect("metadata");
        let ResponseBody::Metadata(metadata) = response.payload else {
            panic!("expected Metadata");
        };
        assert_eq!(metadata.topics.len(), 1);
        assert_eq!(metadata.topics[0].topic, "orders");
        assert_eq!(metadata.brokers[0].port, 9092);
    }
}
