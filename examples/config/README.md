# config — typed config with `conflaguration`

## Builds on

[transform](../transform/README.md) — you configure the pipes you write.

## What it demonstrates

proxima is config-driven: every pipe, listener, and recorder in this repo is
built from a struct, not hand-wired at each call site. `conflaguration` is
the crate behind that struct, and proxima's house pattern stacks four traits
on one type:

- `bon::Builder` — explicit `.field(value)` construction, with per-field defaults.
- `serde::{Deserialize, Serialize}` — files and wire formats.
- `conflaguration::Settings` — resolves fields from prefixed env vars.
- `conflaguration::Validate` — rejects bad values after construction, collecting every failure, not just the first.

`ServerConfig` here is a small stand-in for a real config like
`proxima-telemetry`'s `TelemetryConfig` — four fields instead of a dozen, same
shape. On top of it sits a `layered()` fluent builder (mirroring
`TelemetryLayerBuilder`) that composes sources: `.from_path(...)` loads a
TOML file, `.from_env()` re-resolves from the environment, and `.with_*(...)`
sets a field directly.

The precedence rule is call order, not merge: `.from_path`/`.from_env` each
RE-RESOLVE THE WHOLE STRUCT from that source (touched fields from the source,
`#[setting]`/`#[serde(default)]` for everything else) — a value set by an
earlier call that the later source doesn't also set reverts to default.
`.with_*` only ever touches its own field, so it always wins when called
last. The example proves this with `assert_eq!` at every stage instead of
just printing and hoping.

The other half is the round trip: `toml::to_string` the built config,
`toml::from_str` it back, and `assert_eq!` the result against the original —
then reserialize the restored value and `assert_eq!` the bytes too, so the
proof covers both the struct and the wire format. Last, `Validate` rejects an
invalid config (`port = 0`, `max_connections = 0`) and reports both failures
at once.

## Run

```
cargo run --example config
```

## What you'll see

```
--- round 0: defaults ---
defaults: host=0.0.0.0 port=8080 max_connections=64 request_timeout_ms=5000

--- round 1: layered().from_path(...) overlays a file ---
from file: host=10.0.0.5 port=9000 max_connections=64 request_timeout_ms=5000

--- round 2: layered().from_env() re-resolves fresh from the environment ---
from env: host=0.0.0.0 port=7000 max_connections=256 request_timeout_ms=5000

--- round 3: with_* after from_env wins (call-order precedence) ---
layered: env + explicit overrides: host=override.local port=8080 max_connections=256 request_timeout_ms=1500

--- round 4: serialize <-> deserialize round trip ---
host = "override.local"
port = 8080
max_connections = 256
request_timeout_ms = 1500
round trip: OK (84 bytes, byte-identical)

--- round 5: Validate rejects an invalid config ---
  rejected: port: must be > 0
  rejected: max_connections: must be > 0
```

Round 1 sets `host`/`port` from a temp-file TOML and leaves
`max_connections`/`request_timeout_ms` at their serde defaults. Round 2
throws that file value away — `from_env` resolves every field fresh, so
`host` falls back to its `#[setting(default)]` even though round 1 set it;
only `EXAMPLE_PORT`/`EXAMPLE_MAX_CONNECTIONS` (the vars actually set) differ
from round 0. Round 3 layers one `.with_host`/`.with_request_timeout_ms`
after `.from_env()`: the env-sourced `max_connections` survives untouched,
while the two explicit fields win because they were set last. Round 4 proves
the round trip is exact, not just "close" — both the deserialized struct and
the reserialized bytes match. Round 5 proves `Validate` collects every
failing field in one pass, not just the first one found.
