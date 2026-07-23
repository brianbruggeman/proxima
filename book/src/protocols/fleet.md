# the protocol fleet — memcached, DNS, kafka, MQTT, AMQP

*(builds on: [dial/serve/run](../start/interfaces.md), [add your own protocol](../extend/protocol.md))*

Five real, sans-IO protocol implementations ship alongside proxima's HTTP
stack — each reached through the identical `Listener::builder()`/
`Client::builder()` shape every other chapter in this book uses.
`.<name>(handler)` on the listener, `.<name>(dsn)` on the client, both
delegating to `.protocol(impl AnyProtocol)` underneath — the SAME seam [add
your own protocol](../extend/protocol.md) teaches. This fleet is a
**demonstration** of that seam, not a fixed menu — see that chapter if the
protocol you need isn't here.

**Read the scope line for each protocol before reaching for it.** None of
these five is a production-complete implementation of the real wire spec;
each is a real, working implementation of the routing-critical subset.

The complete, compiled, runnable file below exercises all five, plus a
protocol config (`KafkaServerConfig`) built BOTH as a fluent `bon` builder
and loaded from a real TOML file — the "conflaguration as first-class" half
of this chapter:

```rust
{{#include ../../../examples/protocol_fleet.rs}}
```

Run it yourself:

```sh
cargo run --example protocol_fleet --features "http1-native,memcached-listener,memcached-client,dns-listener,dns-client,kafka-listener,kafka-client,mqtt-listener,mqtt-client,amqp-listener,amqp-client"
```

```
§1 memcached: SET then GET round trip through .memcached(handler)/.memcached(dsn) -> "VALUE greeting 0 15\r\nhello-memcached\r\nEND\r\n"
§2 DNS: .dns(handler).udp() listener + .dns(dsn) client -> A record [203,0,113,42]
§3 Kafka: .kafka(handler).tcp() listener + .kafka(dsn) client -> PRODUCE acked (0 topics in reply)
§4 MQTT: .mqtt(handler).tcp() listener + .mqtt(dsn) client -> CONNECT+PINGREQ/PINGRESP OK
§5 AMQP: .amqp(handler).tcp() listener + .amqp(dsn) client -> basic.publish observed, routing_key="orders"
§6 conflaguration: KafkaServerConfig::builder() and conflaguration::from_file("...") produce the IDENTICAL config (max_message_bytes=4096)

protocol_fleet: memcached/DNS/Kafka/MQTT/AMQP client+listener round trips all OK
```

## The five scope boundaries, at a glance

| protocol | scope |
|---|---|
| memcached | text (ASCII) protocol only — no binary protocol |
| DNS | UDP + TCP framing, no DNS-over-QUIC (DoQ); answers authoritatively from your handler, no recursive resolution |
| kafka | v0 API only (`ApiVersions`/`Produce`/`Fetch`/`Metadata`), opaque record-sets — no schema registry, no compression, no consumer-group coordination |
| MQTT | v3.1.1 only (no v5), QoS2 downgrades, no retained messages, no persistent sessions |
| AMQP | 0-9-1 only, no persistence, no publisher confirms — `basic.publish` is the one business-visible verb |

## Conflaguration as first-class, for a protocol's own config

`.kafka(handler)` sugar carries only the handler — not a `KafkaServerConfig`.
The config-carrying door is the SAME `.protocol()` escape hatch [add your own
protocol](../extend/protocol.md) teaches. Illustrative below — each piece is
verified against source individually; the composed snippet was not run as
its own example:

```rust
let configured = proxima_kafka::KafkaAnyProtocol::new("kafka", handler).with_config(built_config);
let server = Listener::builder()
    .bind(bind)
    .tcp()
    .handle(into_handle(dispatch))
    .any()
    .protocol(configured)
    .serve()
    .await?;
```

See [the config chapter](../configure/config.md) for the house
`#[derive(Builder, Deserialize, Serialize, Settings)]` + `Validate` pattern
this config type follows, and `docs/tutorials/10-conflaguration.md` for the
full builder-vs-TOML walkthrough, including a listener's own tuning config
(`ListenTuningConfig`) and the admission/blacklist knobs.

## What's next

- [add your own protocol](../extend/protocol.md) — the mechanism every
  protocol above is built on, if your wire isn't in this fleet.
- [the config chapter](../configure/config.md) — the house config pattern
  §6 above (`KafkaServerConfig`) follows.
- `docs/tutorials/08-protocol-fleet.md` — this chapter's prose companion,
  with each protocol's own worked example and scope boundary in full.
