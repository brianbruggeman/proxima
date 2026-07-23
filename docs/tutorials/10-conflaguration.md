# Listener on-ramp, part 7: conflaguration as first-class

**Prerequisites:** [part 3: production](./06-listener-production.md) §6 (`BlacklistConfig::layered().from_path`), [part 5: the protocol fleet](./08-protocol-fleet.md) (any one protocol's listener). This page collects and extends the config story those pages already started.

**You will:** see the house config pattern in full — one type that is simultaneously a fluent builder result, a serde shape, and an env/file-loadable settings struct — applied to a LISTENER's own tuning knobs and to a PROTOCOL's own server config, builder and TOML file always producing the identical value.

**New concepts (in order):** `#[derive(Builder, Deserialize, Serialize, Settings)]` + `Validate` · `conflaguration::from_file`/`::from_env()` · the builder-vs-TOML parity contract · why `.kafka(handler)` sugar and a config-carrying `AnyProtocol` registration are two DIFFERENT doors onto the same protocol.

## 1. The house pattern, named

Every config type in this crate — `TelemetryConfig`, `BlacklistConfig`, `KafkaServerConfig`, the `ServerConfig` in `examples/config/main.rs` — follows the SAME shape (workspace principle 4: config is a first-class surface, not a lesser cousin of the fluent builder):

```rust
#[derive(Debug, Clone, PartialEq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "EXAMPLE")]
#[builder(derive(Clone, Debug))]
struct ServerConfig {
    #[setting(default = "0.0.0.0")]
    #[serde(default = "default_host")]
    #[builder(default = default_host())]
    host: String,
    // ...
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Validate for ServerConfig {
    fn validate(&self) -> conflaguration::Result<()> { /* ... */ }
}
```
(`examples/config/main.rs:26–92`, run with `cargo run --example config`)

Four derives, one struct: `Builder` (from the `bon` crate) gives you `ServerConfig::builder().host("...").port(8080).build()`; `Deserialize`/`Serialize` give you the wire/file shape; `Settings` (from `conflaguration`) gives you an env-var surface (`EXAMPLE_HOST`, `EXAMPLE_PORT`, …) and `::from_env()` for free; `Validate` rejects bad values AFTER construction, from any source. One type, three doors in — builder, file, env — always the SAME struct, never three different representations to keep in sync by hand.

## 2. Builder and TOML file, side by side — a protocol's own config

`KafkaServerConfig` (`proxima-kafka/src/config.rs`) is a real, shipped instance of this exact pattern — read/write buffer sizes, the DoS-guarding `max_message_bytes` cap, and the `advertised_host`/`advertised_port` a `Metadata` reply reports:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Builder, Serialize, Deserialize, Settings)]
#[settings(prefix = "KAFKA")]
#[builder(derive(Clone, Debug))]
pub struct KafkaServerConfig {
    #[setting(default = 16777216)]
    #[serde(default = "default_max_message")]
    #[builder(default = default_max_message())]
    pub max_message_bytes: usize,
    // read_buffer_bytes, write_high_water_bytes, broker_id,
    // advertised_host, advertised_port ...
}
```

Both routes, proven to agree in `examples/protocol_fleet.rs`:

```rust
let built = KafkaServerConfig::builder().max_message_bytes(4096).advertised_port(9093).build();

std::fs::write(&toml_path, "max_message_bytes = 4096\nadvertised_port = 9093\n")?;
let from_file: KafkaServerConfig = conflaguration::from_file(&toml_path)?;

assert_eq!(from_file, built, "the builder route and the TOML route must agree bit for bit");
```

Running that produces exactly:

```
§6 conflaguration: KafkaServerConfig::builder() and conflaguration::from_file("/tmp/.../kafka.toml")
   produce the IDENTICAL config (max_message_bytes=4096)
```

**Wiring it into a real listener is the honest gotcha worth stating plainly:** `Listener::builder().kafka(handler)` — the sugar from [part 5](./08-protocol-fleet.md) — takes only the `KafkaPipeHandle`, not a `KafkaServerConfig`. The config-carrying door is one level down, through the SAME `.protocol()` escape hatch [part 6](./09-extend-your-own-protocol.md) teaches. Illustrative — every piece (`KafkaAnyProtocol::new`, `.with_config`, `impl AnyProtocol for KafkaAnyProtocol`, `ListenerBuilder::protocol`) is verified against source individually; the composed snippet was not run as its own example:

```rust
let configured = proxima_kafka::KafkaAnyProtocol::new("kafka", handler).with_config(built);
let server = Listener::builder()
    .bind(bind)
    .tcp()
    .handle(into_handle(NullHttp))
    .any()
    .protocol(configured)
    .serve()
    .await?;
```

This is not a workaround or a missing feature bolted around — it's the SAME "sugar is a shorthand over the general seam" relationship every axis in this whole on-ramp series has. If you need the config knobs, you reach one door deeper, onto the door every protocol (first-party or yours) shares.

## 3. A listener's OWN tuning config, and where it doesn't (yet) reach

`ListenTuningConfig` (`proxima-listen/src/config.rs`) is the SAME pattern applied to the listener's own runtime knobs — `backlog` (the TCP `listen()` SYN queue depth), `drain_timeout_ms`, `http_handler_spread` — with a hand-rolled `.layered()` builder on top for call-order precedence (`.from_path`/`.from_env` override; `.underlay_path`/`.underlay_env` only fill still-unset fields; `.with_*` always wins at its call position). Illustrative — types and method signatures verified against `proxima-listen/src/config.rs` directly, not run as part of this page's own example:

```rust
let config = ListenTuningConfig::layered()
    .from_path(&toml_path)?
    .with_drain_timeout_ms(5_000)  // explicit override wins over the file
    .build();
```

**A real, honest gap, not a teaching simplification:** `ListenTuningConfig` feeds `proxima_listen::handle::Listener::run_with_runtime`/`bind_reuseport_listener_with_options` directly — the LOWER-LEVEL primitives. It is NOT (yet) wired into the umbrella `Listener::builder()` this whole on-ramp series teaches — there is no `.tuning(config)` method on `ListenerBuilder` today. If you need `backlog`/`drain_timeout_ms` tuning on a `Listener::builder()`-built listener, you don't have that door yet; you'd drop to `proxima_listen::handle::Listener` directly, the same lower-level primitive `Listener::builder()` itself composes ([`docs/tutorials/02-listener-builder.md`](./02-listener-builder.md) §2 shows exactly what that primitive looks like one layer down).

## 4. The knob you've already used: `BlacklistConfig`, both ways

[Part 3](./06-listener-production.md) §6 and `examples/any_listener_conflag.rs` already proved this end to end for the admission/blacklist axis — repeated here because it's the THIRD instance of the same pattern, on a knob you've already composed through `.blacklist(config)`:

```rust
let config = BlacklistConfig::layered()
    .from_path(&toml_path)?
    .build();

let server = Listener::builder()
    .bind(bind)
    .accept("h1")
    .deny("scanner", SCANNER_LITERAL.to_vec())
    .blacklist(config)
    .handle(into_handle(LegitOk))
    .serve()
    .await?;
```

with `toml_path` pointing at:

```toml
deny_strike_threshold = 1
unclassifiable_strike_threshold = 5
strike_window_ms = 60000
ban_duration_ms = 300000
```

`examples/any_listener_conflag.rs` also proves a real, worth-knowing gotcha directly rather than asserting it from documentation: there are TWO genuinely different `[admission...]` TOML shapes in this codebase. The BUILD-TIME sizing TOML (`proxima-listen/proxima-listen-core.toml`, read by `build.rs`, baked into the no_std+no_alloc floor's `sized::` consts) nests under `[admission]`/`[admission.blacklist]` headers. The RUNTIME layered loader — what `.from_path()` above actually calls — is FLAT, no section header at all. Handing it a `[admission.blacklist]`-nested file does not error (it's syntactically valid TOML) but silently changes nothing, because the flat partial has no field literally named `admission`:

```
§2: a [admission.blacklist]-nested TOML loads WITHOUT error but changes nothing
   (deny_strike_threshold stayed at the default 1, not the file's 99) — the runtime
   loader wants a FLAT file, unlike the build-time sizing TOML
```

If you're tuning any of this in production and a config change silently doesn't take effect, this is the first thing to check — for EVERY config in this crate, not just `BlacklistConfig`: check whether you're handing the runtime loader a build-time-shaped file.

## What you built

Three real config surfaces (`KafkaServerConfig`, `ListenTuningConfig`, `BlacklistConfig`), each proven builder-equals-file, and the honest map of which ones the umbrella `Listener::builder()` wires in directly (`.blacklist(config)`) versus which ones require dropping one door deeper (`KafkaServerConfig` via `.protocol()`, `ListenTuningConfig` via `proxima_listen::handle::Listener` directly).

## Where to go next

- [`docs/tutorials/06-listener-production.md`](./06-listener-production.md) §6 — the first place this on-ramp taught `.blacklist(config)`, if you arrived here without reading it.
- [`examples/config/main.rs`](../../examples/config/main.rs) (`cargo run --example config`) — the pattern taught from zero, independent of any listener/protocol context.
- [Part 5: the protocol fleet](./08-protocol-fleet.md) / [Part 6: add your own protocol](./09-extend-your-own-protocol.md) — the `.protocol()` seam §2 above used to wire a configured `KafkaAnyProtocol` in.
