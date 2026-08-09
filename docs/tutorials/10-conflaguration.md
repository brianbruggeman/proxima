# Listener on-ramp, part 7: conflaguration as first-class

**Prerequisites:** [part 3: production](./06-listener-production.md) §2/3 (`.deny(name, literal)` + `.blacklist(config)`) and §6 (`BlacklistConfig::layered().from_path`) · [part 5: the protocol fleet](./08-protocol-fleet.md) §3 (Kafka, both halves) · [part 6: add your own protocol](./09-extend-your-own-protocol.md) §3 (`.any().protocol(candidate)`).

**You will:** see proxima's ONE config pattern — `#[derive(Builder, Deserialize, Serialize, Settings)]` + `Validate` — recur, unchanged, across four real types: a from-scratch demo, a protocol's own server config, a listener's own tuning knobs, and the admission blacklist. Builder and TOML file always produce the identical value; that parity is a real, compiling assertion for every one of the four, not a claim taken on faith.

**New concepts (in order):** the house pattern, named · `conflaguration::from_file`, the load-from-TOML route · why `.kafka(handler)` sugar needs the `.protocol()` door for its config, and what a REAL round trip through that door looks like · `ListenTuningConfig`, and the honest door it does NOT have yet · the flat-vs-nested TOML gotcha, recapped from [part 3](./06-listener-production.md) §6.

## 1. The house pattern, named

Every config type in this crate — `ServerConfig` below, `KafkaServerConfig`, `ListenTuningConfig`, `BlacklistConfig` — follows the SAME shape (workspace principle 4: config is a first-class surface, not a lesser cousin of the fluent builder). `examples/config/main.rs` teaches it from zero, independent of any listener/protocol context:

```rust
#[derive(Debug, Clone, PartialEq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "EXAMPLE")]
#[builder(derive(Clone, Debug))]
struct ServerConfig {
    #[setting(default = "0.0.0.0")]
    #[serde(default = "default_host")]
    #[builder(default = default_host())]
    host: String,

    #[setting(default = 8080)]
    #[serde(default = "default_port")]
    #[builder(default = default_port())]
    port: u16,

    #[setting(default = 64)]
    #[serde(default = "default_max_connections")]
    #[builder(default = default_max_connections())]
    max_connections: usize,

    #[setting(default = 5000)]
    #[serde(default = "default_request_timeout_ms")]
    #[builder(default = default_request_timeout_ms())]
    request_timeout_ms: u64,
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8080
}

fn default_max_connections() -> usize {
    64
}

fn default_request_timeout_ms() -> u64 {
    5000
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Validate for ServerConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.host.is_empty() {
            errors.push(ValidationMessage::new("host", "must not be empty"));
        }
        if self.port == 0 {
            errors.push(ValidationMessage::new("port", "must be > 0"));
        }
        if self.max_connections == 0 {
            errors.push(ValidationMessage::new("max_connections", "must be > 0"));
        }
        if self.request_timeout_ms == 0 {
            errors.push(ValidationMessage::new("request_timeout_ms", "must be > 0"));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}
```
(`examples/config/main.rs:28-96`)

Four derives, one struct: `Builder` (from the `bon` crate) gives you `ServerConfig::builder().host("...").port(8080).build()`; `Deserialize`/`Serialize` give you the wire/file shape; `Settings` (from `conflaguration`) gives you an env-var surface (`EXAMPLE_HOST`, `EXAMPLE_PORT`, …) and `::from_env()` for free; `Validate` rejects bad values AFTER construction, from any source. One type, three doors in — builder, file, env — always the SAME struct. `cargo run --example config` runs the whole file end to end (defaults, a file overlay, an env overlay, `.with_*` call-order precedence, a serde round trip, and a rejected config) and prints, among other lines:

```
--- round 3: with_* after from_env wins (call-order precedence) ---
layered: env + explicit overrides: host=override.local port=8080 max_connections=256 request_timeout_ms=1500
--- round 5: Validate rejects an invalid config ---
  rejected: port: must be > 0
  rejected: max_connections: must be > 0
```

## 2. A protocol's own config: `KafkaServerConfig`, and the door `.kafka(handler)` doesn't have

`KafkaServerConfig` (`proxima-kafka/src/config.rs:64-95`) is a real, shipped instance of the identical four-derive-plus-`Validate` shape §1 just showed — same `#[derive(Debug, Clone, PartialEq, Eq, Builder, Serialize, Deserialize, Settings)]` + `#[settings(prefix = "KAFKA")]`, same `#[setting]`/`#[serde(default = ..)]`/`#[builder(default = ..)]` triple per field, same `impl Default`/`impl Validate` pair (`proxima-kafka/src/config.rs:97-124`) — over four fields instead of `ServerConfig`'s: the DoS-guarding `max_message_bytes` cap, the `broker_id` a `Metadata` reply reports, and the `advertised_host`/`advertised_port` a client uses to redial.

`examples/protocol_fleet.rs`'s §6 proves builder and TOML file agree bit for bit, THEN wires the built config into a REAL listener — not a hand-wave, an actual PRODUCE round trip through it. This is the honest gotcha worth stating plainly: `Listener::builder().kafka(handler)` — the sugar from [part 5](./08-protocol-fleet.md) — takes only a `KafkaPipeHandle`, not a `KafkaServerConfig` (`src/listener/handle.rs`'s own `ListenerProtocolExt::kafka` is `self.protocol(proxima_kafka::KafkaAnyProtocol::new("kafka", handler))` — no config parameter at all). The config-carrying door is one level down, through the SAME `.protocol()` escape hatch [part 6](./09-extend-your-own-protocol.md) teaches:

```rust
use proxima_kafka::wire::{
    ApiKey, ProducePartitionData, ProduceRequest, ProduceResponse, ProduceTopicData, RequestBody,
    ResponseBody, decode_response,
};
use proxima_kafka::{KafkaAnyProtocol, KafkaServerConfig, into_kafka_handle};

struct NullHttp;

impl SendPipe for NullHttp {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
        Ok(Response::new(404))
    }
}

async fn kafka_conflaguration_section(bind: SocketAddr) -> Result<(), ProximaError> {
    let built = KafkaServerConfig::builder()
        .max_message_bytes(4096)
        .advertised_port(9093)
        .build();
    assert_eq!(built.max_message_bytes, 4096);

    let toml_dir = tempfile::tempdir().map_err(ProximaError::Io)?;
    let toml_path = toml_dir.path().join("kafka.toml");
    std::fs::write(&toml_path, "max_message_bytes = 4096\nadvertised_port = 9093\n")
        .map_err(ProximaError::Io)?;
    let from_file: KafkaServerConfig = conflaguration::from_file(&toml_path)
        .map_err(|error| ProximaError::Config(format!("kafka config toml: {error}")))?;
    assert_eq!(
        from_file, built,
        "the builder route and the TOML route must agree bit for bit"
    );

    struct EchoProduce;

    impl SendPipe for EchoProduce {
        type In = RequestBody;
        type Out = ResponseBody;
        type Err = ProximaError;

        async fn call(&self, request: RequestBody) -> Result<ResponseBody, ProximaError> {
            match request {
                RequestBody::Produce(_) => Ok(ResponseBody::Produce(ProduceResponse::default())),
                _ => Err(ProximaError::Upstream("unexpected api".into())),
            }
        }
    }

    // `.kafka(handler)` has no config argument — `.protocol()` is the ONLY
    // door that carries `built` onto the wire.
    let configured =
        KafkaAnyProtocol::new("kafka", into_kafka_handle(EchoProduce)).with_config(built);
    let server = Listener::builder()
        .bind(bind)
        .tcp()
        .handle(into_handle(NullHttp))
        .any()
        .protocol(configured)
        .serve()
        .await?;

    let client = Client::builder().kafka(format!("kafka://{bind}")).build()?;
    let request = RequestBody::Produce(ProduceRequest {
        acks: 1,
        timeout_ms: 100,
        topics: vec![ProduceTopicData {
            topic: "orders".to_string(),
            partitions: vec![ProducePartitionData { partition: 0, record_set: Bytes::new() }],
        }],
    });
    let response = client.call("PRODUCE", "").body(request.encode()).send().await?;
    let body = response.bytes().await?;
    let decoded = decode_response(ApiKey::Produce.to_i16(), &body)
        .map_err(|error| ProximaError::Decode(format!("kafka decode: {error}")))?;
    let ResponseBody::Produce(_produce_response) = decoded else {
        return Err(ProximaError::Upstream("expected a Produce reply".into()));
    };
    server.stop();
    Ok(())
}
```
(`examples/protocol_fleet.rs`'s `kafka_conflaguration_section`, run as part of `cargo run --example protocol_fleet --features "http1-native,kafka-listener,kafka-client,..."`)

Running it produces exactly:

```
§6 conflaguration: KafkaServerConfig::builder() and conflaguration::from_file("/tmp/.../kafka.toml")
   produce the IDENTICAL config (max_message_bytes=4096)
§6 conflaguration: .protocol(KafkaAnyProtocol::new(..).with_config(config)) listener -> PRODUCE
   acked through the config-carrying door .kafka(handler) doesn't have
```

This is not a workaround or a missing feature bolted around — it's the SAME "sugar is a shorthand over the general seam" relationship every axis in this whole on-ramp series has. If you need the config knobs, you reach one door deeper, onto the door every protocol (first-party or yours) shares.

## 3. A listener's OWN tuning config, and where it doesn't (yet) reach

`ListenTuningConfig` (`proxima-listen/src/config.rs`) is the SAME pattern applied to the listener's own runtime knobs — `backlog` (the TCP `listen()` SYN queue depth), `drain_timeout_ms`, `http_handler_spread` — with a hand-rolled `.layered()` builder on top for call-order precedence (`.from_path`/`.from_env` override; `.underlay_path`/`.underlay_env` only fill still-unset fields; `.with_*` always wins at its call position). Its own crate doctest proves builder and TOML file agree, exactly like `ServerConfig` and `KafkaServerConfig` above:

```rust
use std::io::Write;

use proxima_listen::ListenTuningConfig;

let via_builder = ListenTuningConfig::builder()
    .backlog(2048)
    .drain_timeout_ms(10_000)
    .build();

let mut file = tempfile::Builder::new().suffix(".toml").tempfile().expect("tempfile");
write!(file, "backlog = 2048\ndrain_timeout_ms = 10000\n").expect("write toml");

let via_toml = ListenTuningConfig::layered()
    .from_path(file.path())
    .expect("load from toml")
    .build();

assert_eq!(via_builder, via_toml);
```
(`proxima-listen/src/config.rs:61-78`)

**A real, honest gap, not a teaching simplification:** `ListenTuningConfig` feeds `proxima_listen::handle::Listener::run_with_runtime` (`proxima-listen/src/handle.rs:157`) and `bind_reuseport_listener_with_options` (`proxima-listen/src/handle.rs:410`) directly — the LOWER-LEVEL primitives. It is NOT (yet) wired into the umbrella `Listener::builder()` this whole on-ramp series teaches — there is no `.tuning(config)` method on `ListenerBuilder` today (confirmed directly: `ListenerBuilder`'s own fields, `src/listener/handle.rs:82`, carry no `ListenTuningConfig`). If you need `backlog`/`drain_timeout_ms` tuning on a `Listener::builder()`-built listener, you don't have that door yet; you'd drop to `proxima_listen::handle::Listener` directly, the same lower-level primitive `Listener::builder()` itself composes ([`docs/tutorials/02-listener-builder.md`](./02-listener-builder.md) §2 shows exactly what that primitive looks like one layer down).

## 4. The knob you've already used: `BlacklistConfig`, recapped

[Part 3](./06-listener-production.md) §2/3 and §6 already proved this end to end for the admission/blacklist axis — a real `.deny(name, literal)` + `.blacklist(config)` listener banning a scanner's peer, and the flat-vs-nested TOML gotcha (a `[admission.blacklist]`-nested file loads without error but silently changes nothing, because the RUNTIME loader wants a flat file — only the BUILD-TIME `proxima-listen-core.toml` sizing floor nests under `[admission]`/`[admission.blacklist]`). Repeated here as the FOURTH instance of the identical house pattern, on a knob you've already composed through `.blacklist(config)`, via `BlacklistConfig`'s own crate doctest:

```rust
use std::io::Write as _;

use proxima_listen::BlacklistConfig;

let via_builder = BlacklistConfig::builder()
    .deny_strike_threshold(2)
    .ban_duration_ms(600_000)
    .build();

let mut file = tempfile::Builder::new().suffix(".toml").tempfile().expect("tempfile");
write!(file, "deny_strike_threshold = 2\nban_duration_ms = 600000\n").expect("write toml");

let via_toml = BlacklistConfig::layered()
    .from_path(file.path())
    .expect("load from toml")
    .build();

assert_eq!(via_builder, via_toml);
```
(`proxima-listen/src/admission/blacklist.rs:67-84`)

If you're tuning any config in this crate in production and a change silently doesn't take effect, [part 3](./06-listener-production.md) §6's gotcha is the first thing to check — for EVERY config here, not just `BlacklistConfig`: check whether you're handing the runtime loader a build-time-shaped file.

## What you built

Four real config surfaces (`ServerConfig`, `KafkaServerConfig`, `ListenTuningConfig`, `BlacklistConfig`), each proven builder-equals-file with a real, compiling assertion, plus the honest map of how deep each one is wired: `.blacklist(config)` reaches all the way to `Listener::builder()`; `KafkaServerConfig` needs one more door down, `.protocol(KafkaAnyProtocol::new(..).with_config(..))`, proven above with a real PRODUCE round trip; `ListenTuningConfig` doesn't have a `Listener::builder()` door at all yet.

## Where to go next

- [`docs/tutorials/06-listener-production.md`](./06-listener-production.md) §2/3 and §6 — the full real listener wiring for `BlacklistConfig`, and where the flat-vs-nested TOML gotcha was first proved.
- [`examples/config/main.rs`](../../examples/config/main.rs) (`cargo run --example config`) — the pattern taught from zero, independent of any listener/protocol context.
- [Part 5: the protocol fleet](./08-protocol-fleet.md) / [Part 6: add your own protocol](./09-extend-your-own-protocol.md) — the `.protocol()` seam §2 above used to wire a configured `KafkaAnyProtocol` in.
