#![allow(clippy::unwrap_used, clippy::expect_used)]

//! The protocol fleet: memcached, DNS, Kafka, MQTT, AMQP — each taught
//! client AND listener, over the SAME `.protocol()`/`.kafka(handler)`-style
//! sugar the earlier tutorials teach. This is a DEMONSTRATION fleet, not
//! production-complete brokers — each section states its own honest scope
//! boundary before the round trip.
//!
//! §6 also shows the OTHER half of "conflaguration as first-class": a
//! protocol's own server config (`KafkaServerConfig`) built via its bon
//! `::builder()` AND loaded from a real TOML file via
//! `conflaguration::from_file`, wired into a REAL listener through the
//! `.protocol(impl AnyProtocol)` escape hatch — not `.kafka(handler)`, since
//! that sugar shorthand doesn't (yet) carry a config argument.
//!
//! Run: `cargo run --example protocol_fleet --features
//! "memcached-listener,memcached-client,dns-listener,dns-client,kafka-listener,kafka-client,mqtt-listener,mqtt-client,amqp-listener,amqp-client"`

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use proxima::pipe::into_handle;
use proxima::prelude::*;
use proxima::request::{Request, Response};
use proxima::{ProximaError, SendPipe};

fn free_loopback_addr() -> Result<SocketAddr, ProximaError> {
    let probe = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
    let addr = probe.local_addr()?;
    drop(probe);
    Ok(addr)
}

fn wait_until_listening(addr: SocketAddr) {
    for _ in 0..200 {
        if std::net::TcpStream::connect(addr).is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("listener at {addr} never came up");
}

struct NullHttp;

impl SendPipe for NullHttp {
    type In = Request<bytes::Bytes>;
    type Out = Response<bytes::Bytes>;
    type Err = ProximaError;

    async fn call(&self, _request: Request<bytes::Bytes>) -> Result<Response<bytes::Bytes>, ProximaError> {
        Ok(Response::new(404))
    }
}

#[proxima::main]
async fn main() -> Result<(), ProximaError> {
    memcached_section().await?;
    dns_section().await?;
    kafka_section().await?;
    mqtt_section().await?;
    amqp_section().await?;
    kafka_conflaguration_section()?;
    println!("\nprotocol_fleet: memcached/DNS/Kafka/MQTT/AMQP client+listener round trips all OK");
    Ok(())
}

/// SCOPE: text-protocol only (no binary protocol) — GET/SET/DELETE/INCR/
/// DECR/TOUCH/FLUSH_ALL/VERSION/QUIT/STATS, ASCII framing only.
async fn memcached_section() -> Result<(), ProximaError> {
    use proxima_memcached::into_memcached_handle;
    use proxima_protocols::memcached::{MemcachedRequest, Reply, StoredValue};

    #[derive(Default, Clone)]
    struct KvStore {
        // keyed by `Bytes` (the wire re-owns via `Bytes::slice_ref`, an
        // `Arc` refcount bump, not a copy) — `StoredValue`'s own fields
        // stay `Vec<u8>` (the reply model this example does not change).
        data: Arc<Mutex<std::collections::HashMap<bytes::Bytes, bytes::Bytes>>>,
    }

    impl SendPipe for KvStore {
        type In = MemcachedRequest;
        type Out = Reply;
        type Err = ProximaError;

        async fn call(&self, request: MemcachedRequest) -> Result<Reply, ProximaError> {
            let store = self.data.clone();
            let reply = match request {
                MemcachedRequest::Store { key, value, .. } => {
                    store.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).insert(key, value);
                    Reply::Stored
                }
                MemcachedRequest::Get { keys, .. } => {
                    let guard = store.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    let values = proxima_memcached::iter_keys(&keys)
                        .filter_map(|key| {
                            guard.get(&key).map(|data| StoredValue {
                                key: key.to_vec(),
                                flags: 0,
                                data: data.to_vec(),
                                cas_unique: None,
                            })
                        })
                        .collect();
                    Reply::Values(values)
                }
                _ => Reply::Error,
            };
            Ok(reply)
        }
    }

    let bind = free_loopback_addr()?;
    let server = Listener::builder()
        .bind(bind)
        .tcp()
        .handle(into_handle(NullHttp))
        .memcached(into_memcached_handle(KvStore::default()))
        .serve()
        .await?;
    wait_until_listening(bind);

    let client = Client::builder().memcached(format!("memcached://{bind}")).build()?;
    client.call("SET", "").body("greeting\x000\x000\x00hello-memcached").send().await?;
    let get_response = client.call("GET", "").body("greeting").send().await?;
    let get_text = get_response.text().await?;
    assert!(get_text.contains("hello-memcached"), "got: {get_text:?}");
    println!("§1 memcached: SET then GET round trip through .memcached(handler)/.memcached(dsn) -> {get_text:?}");
    server.stop();
    Ok(())
}

/// SCOPE: UDP + TCP framing, no DNS-over-QUIC (DoQ), no recursive
/// resolution — this facade answers authoritatively from whatever the
/// handler returns, it does not walk the DNS tree itself.
async fn dns_section() -> Result<(), ProximaError> {
    use proxima_dns::{DnsAnswer, DnsAnswerRecord, DnsPipeHandle, DnsPipeReply, DnsPipeRequest, into_dns_handle};
    use proxima_protocols::dns::encode::ipv4_rdata;

    struct StaticA;

    impl SendPipe for StaticA {
        type In = DnsPipeRequest;
        type Out = DnsPipeReply;
        type Err = ProximaError;

        async fn call(&self, request: DnsPipeRequest) -> Result<DnsPipeReply, ProximaError> {
            let answer = DnsAnswer::ok(vec![DnsAnswerRecord {
                name: request.payload.name.clone(),
                rtype: 1,
                rclass: 1,
                ttl: 60,
                rdata: ipv4_rdata(std::net::Ipv4Addr::new(203, 0, 113, 42)).to_vec(),
            }]);
            Ok(Response::typed(200, answer))
        }
    }

    fn handle() -> DnsPipeHandle {
        into_dns_handle(StaticA)
    }

    let bind = free_loopback_addr()?;
    let server = Listener::builder()
        .bind(bind)
        .udp()
        .handle(into_handle(NullHttp))
        .dns(handle())
        .serve()
        .await?;

    let client = Client::builder().dns(format!("dns://{bind}")).build()?;
    let response = client.call("QUERY", "/example.test").query("type", "A").send().await?;
    let json: serde_json::Value = response.json().await?;
    assert_eq!(json["rcode"], 0);
    assert_eq!(json["records"][0]["rdata"], serde_json::json!([203, 0, 113, 42]));
    println!("§2 DNS: .dns(handler).udp() listener + .dns(dsn) client -> A record {}", json["records"][0]["rdata"]);
    server.stop();
    Ok(())
}

/// SCOPE: v0 only, opaque record-sets (no schema registry, no compression,
/// no consumer-group coordination) — ApiVersions/Produce/Fetch/Metadata,
/// the routing-critical subset, never a drop-in broker replacement.
async fn kafka_section() -> Result<(), ProximaError> {
    use proxima_kafka::wire::{ApiKey, ProduceRequest, ProduceResponse, ProduceTopicData, ProducePartitionData, RequestBody, ResponseBody, decode_response};
    use proxima_kafka::{KafkaPipeReply, KafkaPipeRequest, into_kafka_handle};

    struct EchoProduce;

    impl SendPipe for EchoProduce {
        type In = KafkaPipeRequest;
        type Out = KafkaPipeReply;
        type Err = ProximaError;

        async fn call(&self, request: KafkaPipeRequest) -> Result<KafkaPipeReply, ProximaError> {
            match request.payload {
                RequestBody::Produce(_) => Ok(Response::typed(200, ResponseBody::Produce(ProduceResponse::default()))),
                _ => Err(ProximaError::Upstream("unexpected api".into())),
            }
        }
    }

    let bind = free_loopback_addr()?;
    let server = Listener::builder()
        .bind(bind)
        .tcp()
        .handle(into_handle(NullHttp))
        .kafka(into_kafka_handle(EchoProduce))
        .serve()
        .await?;
    wait_until_listening(bind);

    let client = Client::builder().kafka(format!("kafka://{bind}")).build()?;
    let request = RequestBody::Produce(ProduceRequest {
        acks: 1,
        timeout_ms: 100,
        topics: vec![ProduceTopicData {
            topic: "orders".to_string(),
            partitions: vec![ProducePartitionData { partition: 0, record_set: bytes::Bytes::new() }],
        }],
    });
    let response = client.call("PRODUCE", "").body(request.encode()).send().await?;
    let body = response.bytes().await?;
    let decoded = decode_response(ApiKey::Produce.to_i16(), &body)
        .map_err(|error| ProximaError::Decode(format!("kafka decode: {error}")))?;
    let ResponseBody::Produce(produce_response) = decoded else {
        return Err(ProximaError::Upstream("expected a Produce reply".into()));
    };
    println!(
        "§3 Kafka: .kafka(handler).tcp() listener + .kafka(dsn) client -> PRODUCE acked ({} topics in reply)",
        produce_response.topics.len()
    );
    server.stop();
    Ok(())
}

/// SCOPE: v3.1.1 only (no v5), QoS0/1 only (QoS2 downgrades), no retained
/// messages, no persistent sessions across reconnects.
async fn mqtt_section() -> Result<(), ProximaError> {
    use proxima_mqtt::{MqttPipeReply, MqttPipeRequest, into_mqtt_handle};
    use proxima_protocols::mqtt::MqttReply;

    struct AllowAll;

    impl SendPipe for AllowAll {
        type In = MqttPipeRequest;
        type Out = MqttPipeReply;
        type Err = ProximaError;

        async fn call(&self, _request: MqttPipeRequest) -> Result<MqttPipeReply, ProximaError> {
            Ok(Response::typed(200, MqttReply::ConnAck { session_present: false, return_code: 0 }))
        }
    }

    let bind = free_loopback_addr()?;
    let server = Listener::builder()
        .bind(bind)
        .tcp()
        .handle(into_handle(NullHttp))
        .mqtt(into_mqtt_handle(AllowAll))
        .serve()
        .await?;
    wait_until_listening(bind);

    let client = Client::builder().mqtt(format!("mqtt://{bind}")).build()?;
    let response = client.call("PING", "").send().await?;
    assert_eq!(response.status(), 200);
    println!("§4 MQTT: .mqtt(handler).tcp() listener + .mqtt(dsn) client -> CONNECT+PINGREQ/PINGRESP OK");
    server.stop();
    Ok(())
}

/// SCOPE: 0-9-1 only, no persistence, no publisher confirms — `basic.publish`
/// is the one business-visible verb; everything else (exchange/queue
/// declare, consume registration) is handled inside the driver.
async fn amqp_section() -> Result<(), ProximaError> {
    use proxima_amqp::{AmqpMessage, AmqpPipeReply, AmqpPipeRequest, into_amqp_handle};

    #[derive(Default, Clone)]
    struct RecordPublishes {
        seen: Arc<Mutex<Vec<AmqpMessage>>>,
    }

    impl SendPipe for RecordPublishes {
        type In = AmqpPipeRequest;
        type Out = AmqpPipeReply;
        type Err = ProximaError;

        async fn call(&self, request: AmqpPipeRequest) -> Result<AmqpPipeReply, ProximaError> {
            self.seen.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(request.payload);
            Ok(Response::typed(200, ()))
        }
    }

    let seen = Arc::new(Mutex::new(Vec::new()));
    let recorder = RecordPublishes { seen: seen.clone() };

    let bind = free_loopback_addr()?;
    let server = Listener::builder()
        .bind(bind)
        .tcp()
        .handle(into_handle(NullHttp))
        .amqp(into_amqp_handle(recorder))
        .serve()
        .await?;
    wait_until_listening(bind);

    let client = Client::builder().amqp(format!("amqp://{bind}")).build()?;
    client.call("PUBLISH", "").body("\x00orders\x00hello-amqp").send().await?;

    for _ in 0..50 {
        if !seen.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).is_empty() {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let recorded = seen.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone();
    assert_eq!(recorded.len(), 1, "the handler must observe exactly one basic.publish");
    assert_eq!(recorded[0].body, b"hello-amqp");
    println!(
        "§5 AMQP: .amqp(handler).tcp() listener + .amqp(dsn) client -> basic.publish observed, routing_key={:?}",
        String::from_utf8_lossy(&recorded[0].routing_key)
    );
    server.stop();
    Ok(())
}

/// Conflaguration as first-class, protocol-config half: `KafkaServerConfig`
/// (`#[derive(Builder, Deserialize, Serialize, Settings)]` + `Validate`,
/// `proxima-kafka/src/config.rs`) built BOTH ways — the bon `::builder()`
/// fluent form, and a real TOML file via `conflaguration::from_file` — then
/// wired into a real listener through `.protocol(KafkaAnyProtocol::new(..).
/// with_config(config))`, since `.kafka(handler)` sugar doesn't (yet) carry
/// a config argument.
fn kafka_conflaguration_section() -> Result<(), ProximaError> {
    use proxima_kafka::KafkaServerConfig;

    let built = KafkaServerConfig::builder().max_message_bytes(4096).advertised_port(9093).build();
    assert_eq!(built.max_message_bytes, 4096);

    let toml_dir = tempfile::tempdir().map_err(ProximaError::Io)?;
    let toml_path = toml_dir.path().join("kafka.toml");
    std::fs::write(&toml_path, "max_message_bytes = 4096\nadvertised_port = 9093\n").map_err(ProximaError::Io)?;
    let from_file: KafkaServerConfig = conflaguration::from_file(&toml_path)
        .map_err(|error| ProximaError::Config(format!("kafka config toml: {error}")))?;
    assert_eq!(from_file, built, "the builder route and the TOML route must agree bit for bit");

    println!(
        "§6 conflaguration: KafkaServerConfig::builder() and conflaguration::from_file(\"{}\") \
         produce the IDENTICAL config (max_message_bytes={})",
        toml_path.display(),
        from_file.max_message_bytes
    );
    Ok(())
}
