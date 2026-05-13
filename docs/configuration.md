# configuration

Two surfaces, one shape: typed Rust API and TOML files. Round-trip
via serde — anything expressible one way is expressible the other.

For the deeper architectural rationale see [`SHAPE.md`](../SHAPE.md);
for the design discipline behind it see [`principles.md`](principles.md).

## table of contents

- [the shapes](#the-shapes)
- [ProximaSettings — typed config](#proximasettings--typed-config)
- [tunables — env-overridable](#tunables--env-overridable)
- [typed listener configs](#typed-listener-configs)
- [typed upstream configs](#typed-upstream-configs)
- [typed middleware configs](#typed-middleware-configs)
- [composing chains with `.then()`](#composing-chains-with-then)
- [App lifecycle and Server handle](#app-lifecycle-and-server-handle)
- [round-trip property](#round-trip-property)
- [plugin extension](#plugin-extension)

---

## the shapes

```rust
use proxima::settings::{
    BearerAuth, Composable, HttpListener, HttpUpstream, ProximaSettings, RateLimit,
};
use proxima::{App, MountTarget};
use std::time::Duration;
```

Three things every config eventually produces:

1. A set of **named registry entries** (listeners, upstreams,
   middlewares, pipes) — typed structs with `bon::Builder`,
   serializable via serde.
2. An **App** that has those entries registered as `PipeHandle`s
   and mounted at routes.
3. A **`Server`** returned from `App::serve(...)` that drives the
   listener loop and exposes the operator `ControlPlane` API.

The fluent API and the TOML loader are both paths from (1) into (2)
and (3). They produce byte-identical runtime state.

## ProximaSettings — typed config

The top-level shape lives in `proxima::settings::ProximaSettings`.
Four map-keyed registries plus three nested tuning structs:

```rust
pub struct ProximaSettings {
    pub listeners: BTreeMap<String, RegistryEntry>,
    pub upstreams: BTreeMap<String, RegistryEntry>,
    pub middlewares: BTreeMap<String, RegistryEntry>,
    pub pipes: BTreeMap<String, RegistryEntry>,
    pub http: HttpTuning,
    pub zstd: ZstdTuning,
    pub buffer_pool: BufferPoolTuning,
}
```

`RegistryEntry` is the untyped form on the wire — `{ type: String,
spec: Value }`. Built-in factories deserialize their slice typed;
plugin factories register their own derived structs and bring their
own typed config. The top-level struct doesn't enumerate variants.

### loading from a file

```rust
let settings = ProximaSettings::from_path("proxima.toml")?;
```

Supports TOML, JSON, YAML, RON, JSON5, XML — format sniffed from the
file extension via the default config-format registry.

### applying to an App

```rust
let mut app = App::new()?;
app.apply_settings(&settings).await?;
```

Registers every `[upstreams.*]` entry as a named `PipeHandle`.
For every `[pipes.*]` entry, composes its declared chain
(middlewares in order, then the leaf upstream) and registers the
composed `PipeHandle`. Listeners are materialized at
`App::serve(...)` time; mounts are wired by the caller via
`App::mount(path, target)`.

## tunables — env-overridable

Tunables get free environment-variable overrides via the
`conflaguration::Settings` derive. Three groups today:

- `[http]` — framing-layer knobs (`response_buffer_bytes`,
  `read_buffer_bytes`, `headers_capacity`, `h1_max_headers`,
  `h2_max_frame_size`, `h2_max_concurrent_streams`,
  `h2_header_table_size`)
- `[zstd]` — `compression_level` for recording sinks
- `[buffer_pool]` — `depth_per_worker`, `buffer_bytes`

```toml
[http]
response_buffer_bytes = 16384

[zstd]
compression_level = 9
```

Or override at runtime:

```bash
PROXIMA_HTTP_RESPONSE_BUFFER_BYTES=32768 \
PROXIMA_ZSTD_COMPRESSION_LEVEL=19 \
    proxima serve --config proxima.toml
```

`#[setting(nested, override_prefix)]` on the parent struct
accumulates the parent prefix, so `PROXIMA_HTTP_*` rather than
the nested struct's standalone `HTTP_*`.

## typed listener configs

Three concrete shapes today:

```rust
// TCP HTTP/1.1 (with h2 prior-knowledge sniff on the listener side)
HttpListener::http("0.0.0.0:8080".parse()?)

// TCP + TLS termination + ALPN (h2 + http/1.1)
#[cfg(feature = "tls")]
HttpsListener::https(
    "0.0.0.0:8443".parse()?,
    PathBuf::from("cert.pem"),
    PathBuf::from("key.pem"),
)

// UDS-bound HTTP/1.1 (daemon control plane + local-only pipes)
#[cfg(unix)]
HttpUdsListener::local(PathBuf::from("/tmp/proxima.sock")) // mode 0o600
```

Each impls `Into<RunConfig>`, so `App::serve(impl Into<RunConfig>)`
accepts any of them directly. Equivalent TOML:

```toml
[listeners.public]
type = "http"
addr = "0.0.0.0:8443"
tls.cert = "cert.pem"
tls.key = "key.pem"

[listeners.admin]
type = "http"
path = "/tmp/proxima.sock"
mode = 0o600
```

## typed upstream configs

`HttpUpstream` covers the common upstream case:

```rust
HttpUpstream::builder()
    .url("https://backend.internal:8443")
    .timeout(Duration::from_secs(5))
    .build()
```

`Into<Spec>` emits the `{ "type": "http", ... }` shape the existing
factory registry dispatches on. Equivalent TOML:

```toml
[upstreams.backend]
type = "http"
url = "https://backend.internal:8443"
timeout = "5s"
```

Other upstream kinds (`kv`, `process`, `synth`, `replay`, `record`,
`fs`, `callback`) work through the same factory path; typed Rust
builders land as needed.

## typed middleware configs

Two middleware types today:

```rust
// Bearer-token auth — wraps an inner Pipe
BearerAuth::allow_tokens(["t-1", "t-2"])
// or with full options:
BearerAuth::builder()
    .allow(vec!["t-1".into(), "t-2".into()])
    .header("x-api-token")
    .realm("acme")
    .on_unauthorized_status(403)
    .strip_prefix("")  // disable the default "Bearer " prefix
    .build()

// Token-bucket rate-limit
RateLimit::token_bucket(100, 50)  // capacity, refill_per_sec
// or with key extractor:
RateLimit::builder()
    .capacity(100)
    .refill_per_sec(50)
    .key("header")
    .header_name("x-client-id")
    .retry_after_ms(2_000)
    .build()
```

Equivalent TOML:

```toml
[middlewares.auth]
type = "auth"
allow = ["t-1", "t-2"]

[middlewares.rl]
type = "rate_limit"
capacity = 100
refill_per_sec = 50
```

Other middlewares (`retry`, `transform`, `isolate`, `validate`,
`diff`, `tee`) work through the same factory path; typed Rust
builders land as needed.

## composing chains with `.then()`

```rust
use proxima::settings::Composable;

let chain = BearerAuth::allow_tokens(["t-1"])
    .then(RateLimit::token_bucket(100, 50))
    .then(HttpUpstream::url("https://backend.internal"));

app.pipe("api", chain).await?;
```

**Top-down code order = request execution order.** Auth fires
first; if it passes, the request hits rate-limit; if it passes,
the request goes to the http upstream. Same direction as Express,
axum route chains, every HTTP server framework draws.

`Composable` is implemented blanket-style for any `T: Into<Spec>`,
so every typed config gets `.then()` for free. The result is a
`Chain` whose `Into<Spec>` emits the
`{ <leaf_fields>, middleware: [...] }` shape the loader already
understands.

Equivalent TOML uses the registry-entry shape with a named chain:

```toml
[pipes.api]
mount = "/api/{*path}"
chain = ["auth", "rl", "backend"]
```

Where `auth`, `rl`, `backend` are pre-registered in `[middlewares.*]`
and `[upstreams.*]`. `App::apply_settings(&settings)` resolves the
chain by name at materialization time.

## App lifecycle and Server handle

`App::serve(...)` spawns the listener loop and returns a `Server`
handle. Three drive shapes:

```rust
// A: terminal future (await directly)
app.serve(listener).await?.await?;

// B: explicit method
let server = app.serve(listener).await?;
server.run_until_signal().await;

// C: clone-and-control (operators, tests, embedding code)
let server = app.serve(listener).await?;
let stopper = server.clone();
tokio::spawn(async move {
    on_some_event.await;
    stopper.stop();          // any clone can shutdown
});
let metrics = server.snapshot_metrics().await?;  // ControlPlane via Server
server.await?;
```

`Server: Clone` via internal `Arc`. Clones share control-plane
state; the listener-loop drive method is single-owner (first call
consumes the `Shutdown`, subsequent calls from clones are no-ops).
The `shutdown_notify` channel inside lets any clone wake the
driving awaiter.

`Server` impls the same `ControlPlane` trait the
`ControlPlanePipe` exposes over HTTP. One concept, two access
modes: in-process method calls on the trait, or remote HTTP routes
on the Pipe-shaped wrapper.

## round-trip property

Settings ⇄ TOML ⇄ Settings is identity. This is the load-bearing
invariant for the fluent ⇄ TOML claim: if they diverged,
"recording / replay / hot-swap key off the same named entries
either way" would be a lie. Property tests in
`rust/tests/settings_round_trip.rs` and
`rust/tests/settings_to_app.rs` gate the property.

The reverse direction — `Settings::from_app(&app)` extracting a
typed `ProximaSettings` from a running App — needs the App to store
original specs alongside `PipeHandle`s. That's a sidecar refactor
deferred to Phase 6.

## plugin extension

Plugins compose by registering their own `Factory` + typed config
struct at startup. The proxima factory registry (`PipeFactoryRegistry`)
keys off the `type` string in `RegistryEntry`; plugin code adds
entries to the registry, and the top-level `ProximaSettings`
doesn't need to know about them at compile time.

```rust
use companyx_proxima_auth::CompanyXAuthFactory;

App::builder()
    .register_factory::<CompanyXAuthFactory>()  // adds to PipeFactoryRegistry
    .build()?;
```

TOML for the same plugin works because the registry resolves
`type = "companyx-auth"` to `CompanyXAuthConfig`'s deserializer:

```toml
[middlewares.auth]
type = "companyx-auth"
okta_url = "https://acme.okta.com"
```

Each plugin's config struct gets conflaguration's nested env-var
support automatically. Built-ins (Auth, RateLimit, Retry, Transform,
Isolate, Validate, Diff, Tee, HttpUpstream, KvUpstream, ProcessRpc,
SynthUpstream, ReplayUpstream, etc.) ship registered in the default
factory registry.
