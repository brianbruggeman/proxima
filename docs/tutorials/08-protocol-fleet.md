# Listener on-ramp, part 5: the protocol fleet

**Prerequisites:** [part 4: composing the sugar](./07-sugar-composition.md). You should be comfortable with `.kafka(handler)`/`.dns(handler)` as protocol axes and `.kafka(dsn)`/`.dns(dsn)` as their client-side twins.

**You will:** wire up a real memcached, DNS, Kafka, MQTT, and AMQP endpoint — each one BOTH client and listener, dialing itself in-process — and learn the honest scope boundary of each before you reach for one in anger.

**New concepts (in order):** the five protocol crates (`proxima-memcached`, `proxima-dns`, `proxima-kafka`, `proxima-mqtt`, `proxima-amqp`) · each protocol's own typed request/reply pipe contract · why these are a DEMONSTRATION fleet, not drop-in broker replacements.

**Read this warning first, seriously:** none of the five protocols below is a production-complete implementation of the real wire spec. Each is a real, working, sans-IO implementation of a **routing-critical subset** — enough to classify, parse, and dispatch traffic through proxima's pipe algebra. If your production workload needs schema registries, exactly-once semantics, publisher confirms, or DNS-over-QUIC, this fleet is not there yet — the scope boundary for each protocol is stated before its round trip, not hidden in a footnote.

Every code block below is copied verbatim from `examples/protocol_fleet.rs`, and every printed line is the ACTUAL output of running it on this machine:

```sh
cargo run --example protocol_fleet --features "http1-native,memcached-listener,memcached-client,dns-listener,dns-client,kafka-listener,kafka-client,mqtt-listener,mqtt-client,amqp-listener,amqp-client"
```

## 1. memcached

**Scope:** text protocol ONLY (no binary protocol). `GET`/`GETS`/`SET`/`ADD`/`REPLACE`/`APPEND`/`PREPEND`/`CAS`/`DELETE`/`INCR`/`DECR`/`TOUCH`/`FLUSH_ALL`/`VERSION`/`QUIT`/`STATS` — the ASCII framing, not the binary one.

**Listener:** `.memcached(handler)` takes a `MemcachedPipeHandle` — a typed pipe over `Request<MemcachedRequest>` -> `Response<Reply>` (`proxima-memcached/src/pipes.rs`). No downcasting, no type erasure — the handler pattern-matches the SAME `MemcachedRequest` enum the wire parser produced:

```rust
impl SendPipe for KvStore {
    type In = MemcachedPipeRequest;
    type Out = MemcachedPipeReply;
    type Err = ProximaError;

    async fn call(&self, request: MemcachedPipeRequest) -> Result<MemcachedPipeReply, ProximaError> {
        let reply = match request.payload {
            MemcachedRequest::Store { key, value, .. } => {
                store.insert(key, value);
                Reply::Stored
            }
            MemcachedRequest::Get { keys, .. } => Reply::Values(/* ... */),
            _ => Reply::Error,
        };
        Ok(Response::typed(200, reply))
    }
}

let server = Listener::builder()
    .bind(bind)
    .tcp()
    .handle(into_handle(NullHttp))
    .memcached(into_memcached_handle(KvStore::default()))
    .serve()
    .await?;
```

**Client:** `.memcached(dsn)` on `Client::builder()`. The wire convention is a NUL-delimited body specific to each verb (`key\0flags\0exptime\0value` for `SET`, a bare key list for `GET`) — `src/upstreams/memcached.rs`'s own module doc has the full table:

```rust
let client = Client::builder().memcached(format!("memcached://{bind}")).build()?;
client.call("SET", "").body("greeting\x000\x000\x00hello-memcached").send().await?;
let get_response = client.call("GET", "").body("greeting").send().await?;
```

```
§1 memcached: SET then GET round trip through .memcached(handler)/.memcached(dsn) -> "VALUE greeting 0 15\r\nhello-memcached\r\nEND\r\n"
```

## 2. DNS

**Scope:** UDP + TCP framing (RFC 1035 §4.2.1/§4.2.2), no DNS-over-QUIC (DoQ — `.quic()` is a config error, [part 4](./07-sugar-composition.md) §5). This facade answers AUTHORITATIVELY from whatever the handler returns — it does not walk the DNS tree or perform recursive resolution itself; you are the authority for whatever zone you wire in.

**Listener:** `.dns(handler).udp()` (or `.tcp()` — [part 4](./07-sugar-composition.md) §5 covers the dual-transport branching). The handler answers a typed `DnsQuery` with a typed `DnsAnswer`:

```rust
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
```

**Client:** `.dns(dsn)` always dials UDP (`DnsClientUpstream` composes `PrimeDatagramFactory` directly — `src/upstreams/dns.rs`). The generic `Request<Bytes>`/`Response<Bytes>` boundary carries the query in the PATH and returns a JSON-encoded answer (this facade's own convention, since there is no wire-level JSON to borrow from):

```rust
let client = Client::builder().dns(format!("dns://{bind}")).build()?;
let response = client.call("QUERY", "/example.test").query("type", "A").send().await?;
let json: serde_json::Value = response.json().await?;
```

```
§2 DNS: .dns(handler).udp() listener + .dns(dsn) client -> A record [203,0,113,42]
```

## 3. Kafka

**Scope:** v0 API ONLY, opaque record-sets — `ApiVersions`/`Produce`/`Fetch`/`Metadata`, the routing-critical subset. No schema registry, no message compression, no consumer-group coordination. This is a wire-protocol facade you can route traffic through, never a drop-in Kafka broker replacement.

**Listener:** `.kafka(handler)` takes a `KafkaPipeHandle` over `Request<RequestBody>` -> `Response<ResponseBody>` — `ApiVersions` is answered protocol-level and never even reaches this handler (proven directly in `proxima-kafka/tests/any_protocol_end_to_end.rs`):

```rust
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
```

**Client:** `.kafka(dsn)`. The generic pipe boundary carries the body-only wire encoding — `Request.method` names the API (`PRODUCE`/`FETCH`/`METADATA`), `Request.payload` is `RequestBody::encode()`'s output, `Response.payload` is `ResponseBody::encode()`'s (`proxima-kafka/src/client/pipe.rs`'s own module doc has the full contract):

```rust
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
let decoded = decode_response(ApiKey::Produce.to_i16(), &response.bytes().await?)?;
```

```
§3 Kafka: .kafka(handler).tcp() listener + .kafka(dsn) client -> PRODUCE acked (0 topics in reply)
```

(`0 topics in reply` because the handler answers `ProduceResponse::default()` — an empty, well-formed acknowledgement — not because anything failed; see `examples/redis_server.rs`/`proxima-kafka/tests/any_protocol_end_to_end.rs` for a handler that actually threads topic data through.)

## 4. MQTT

**Scope:** v3.1.1 only (no v5), the routing-critical packet subset (`CONNECT`/`PUBLISH`/`SUBSCRIBE`/`UNSUBSCRIBE`/`PING`/`DISCONNECT`). QoS2 downgrades to a lower QoS rather than implementing the full four-packet handshake; no retained messages; no persistent sessions across reconnects.

**Listener:** `.mqtt(handler)` — the handler is dispatched ONLY for `CONNECT` (answer/reject the session); `PUBLISH`/`SUBSCRIBE`/`PING` are answered by the driver and broker directly, never reaching your pipe (proven in `proxima-mqtt/tests/pubsub_round_trip.rs`, which also proves a real two-connection pub/sub delivery):

```rust
impl SendPipe for AllowAll {
    type In = MqttPipeRequest;
    type Out = MqttPipeReply;
    type Err = ProximaError;

    async fn call(&self, _request: MqttPipeRequest) -> Result<MqttPipeReply, ProximaError> {
        Ok(Response::typed(200, MqttReply::ConnAck { session_present: false, return_code: 0 }))
    }
}
```

**Client:** `.mqtt(dsn)` — `CONNECT` runs automatically the first time the client dials (mirroring redis's own `HELLO`/`AUTH` handshake), never a caller-visible `Request`. The caller-visible verbs are `PUBLISH`/`SUBSCRIBE`/`UNSUBSCRIBE`/`PING`/`DISCONNECT`:

```rust
let client = Client::builder().mqtt(format!("mqtt://{bind}")).build()?;
let response = client.call("PING", "").send().await?;
```

```
§4 MQTT: .mqtt(handler).tcp() listener + .mqtt(dsn) client -> CONNECT+PINGREQ/PINGRESP OK
```

## 5. AMQP

**Scope:** 0-9-1 only, no persistence, no publisher confirms. `basic.publish` is the ONE business-visible verb — everything else (exchange/queue declare, consumer registration, channel lifecycle) is handled inside the driver against an in-process `AmqpBroker`, never surfaced to your handler.

**Listener:** `.amqp(handler)` — dispatched once per REASSEMBLED `basic.publish`, BEFORE the broker routes it, so a handler can observe, transform, persist, or reject a message (an `Err` drops it without routing). The reply carries no payload (`AmqpPipeReply = Response<()>`) — AMQP 0-9-1 has no synchronous per-publish acknowledgement outside publisher-confirms, which this facade does not implement:

```rust
impl SendPipe for RecordPublishes {
    type In = AmqpPipeRequest;
    type Out = AmqpPipeReply;
    type Err = ProximaError;

    async fn call(&self, request: AmqpPipeRequest) -> Result<AmqpPipeReply, ProximaError> {
        self.seen.lock().unwrap().push(request.payload);
        Ok(Response::typed(200, ()))
    }
}
```

**Client:** `.amqp(dsn)` — the ONE composable verb this generic boundary exposes is `PUBLISH` (`CONSUME` is the other, streaming deliveries via `Response.stream`); the body convention is `exchange\0routing_key\0body`:

```rust
let client = Client::builder().amqp(format!("amqp://{bind}")).build()?;
client.call("PUBLISH", "").body("\x00orders\x00hello-amqp").send().await?;
```

```
§5 AMQP: .amqp(handler).tcp() listener + .amqp(dsn) client -> basic.publish observed, routing_key="orders"
```

(Empty exchange `""` + routing key `"orders"` is AMQP's DEFAULT exchange convention — it routes directly to the queue named by the routing key, no `exchange.declare` required; see `proxima-amqp/src/broker.rs`'s `default_exchange_routes_directly_to_the_queue_named_by_routing_key` test.)

## What you built

Five real, working, sans-IO protocol implementations, each reached through the identical `Listener::builder()`/`Client::builder()` shape — `.<name>(handler)` on the listener, `.<name>(dsn)` on the client, both delegating to `.protocol()` underneath ([part 6](./09-extend-your-own-protocol.md) teaches that seam directly). None of the five needed a bespoke serve loop, a bespoke client type, or a fork of `App`/`Server`.

## Where to go next

- [Part 6: add your own protocol](./09-extend-your-own-protocol.md) — the SAME `.protocol()` mechanism every axis above delegates to, reachable from a crate that has never heard of proxima-kafka/proxima-mqtt/etc.
- [Part 7: conflaguration as first-class](./10-conflaguration.md) — `KafkaServerConfig`, built the SAME builder-and-TOML way as every config in this crate, wired into a listener through `.protocol(KafkaAnyProtocol::new(..).with_config(config))` (since `.kafka(handler)` sugar doesn't yet carry a config argument).
- `docs/protocol-gap/discipline.md` (if present in your checkout) and each protocol crate's own `src/config.rs` for the FULL knob set beyond what this page's round trips touched.
