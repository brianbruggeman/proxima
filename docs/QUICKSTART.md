# quickstart

A `Pipe` is the primitive — an async `request → response` boundary
that every upstream, middleware, and composition unit in proxima
implements. This quickstart gets you to a running Pipe in five
minutes. The canonical teaching surface for the primitive lives in
the `proxima::pipe` module rustdoc — open `cargo doc` and start
there, or read [proxima-primitives/src/pipe/mod.rs](../proxima-primitives/src/pipe/mod.rs)'s
module-level doc-comment.

proxima is config-first: the same Pipe spec drives both the library
face and `proxima.toml`. Sugar is a pure rewrite —
`proxima::desugar(spec)` shows you what it expands to.

## pipe is the primitive

Every example below builds a Pipe and runs it. The shapes differ
(config file, fluent Rust, CLI one-shot) but they all produce the
same `PipeHandle` under the hood. The substrate primitives (Tee,
Diff, Isolate, Causal, SwappablePipe, WriteBack, check_determinism)
that compose around any Pipe are documented in the `proxima::pipe`
module rustdoc.

## library

```rust
use proxima::Client;
use serde_json::json;

let client = Client::from_value(json!({ "http": "https://api.example.com" }))?;
let resp = client
    .call("POST", "/v1/chat/completions")
    .header("authorization", "Bearer sk-...")
    .json(&json!({"model": "gpt-4", "messages": [{"role": "user", "content": "ping"}]}))?
    .send()
    .await?;
let body: serde_json::Value = resp.json().await?;
```

### caching — sugar form

```rust
let client = Client::from_sugar(json!({
    "http": "https://api.example.com",
    "cache": true,
}))?;
```

### caching — primitive form (what `cache: true` desugars to)

```rust
let client = Client::from_value(json!({
    "name": "proxima",
    "upstreams": [
        { "name": "cache",  "kv": "cache", "max_entries": 1024 },
        { "name": "origin", "http": "https://api.example.com" },
    ],
    "select": { "algorithm": "fallthrough", "miss_on": ["no_data"] },
    "write_back": [["origin", "cache"]],
}))?;
```

That's the whole substrate: `upstreams` tried in order, `select`
decides when to fall through, `write_back` populates the cache.

## config

```toml
# proxima.toml — simplest
http = "https://api.example.com"
```

```sh
proxima call --config proxima.toml --method POST --path /v1/chat/completions
```

```toml
# proxima.toml — cached, primitive form
name = "proxima"

[[upstreams]]
name        = "cache"
kv          = "cache"
max_entries = 1024

[[upstreams]]
name = "origin"
http = "https://api.example.com"

[select]
algorithm = "fallthrough"
miss_on   = ["no_data"]

write_back = [["origin", "cache"]]
```

## sugar reference

| sugar | desugars to |
| --- | --- |
| `cache = true` | `kv:cache` upstream + fallthrough + write_back |
| `cache = { max_entries = N, ttl = "1h" }` | same with configured kv settings |
| `mock = "hello"` | `synth = { status = 200, body = "hello" }` |
| `mock = { status = 200, body = "..." }` | `synth = {...}` |
| `replay = "./fixture.jsonl"` | `replay = { source = "./fixture.jsonl", format = "jsonl" }` |
| `replay = "./fixture.bin"` | `replay = { source = "./fixture.bin", format = "bin" }` |

Sugar is the only sugar. New behaviour goes through the primitives.

## substrate

| substrate | config dispatch | examples |
| --- | --- | --- |
| listen protocol | `[[listen]] type = "..."` | http, direct_socket, mcp |
| upstream | `type = "..."` (or shorthand) | http, kv, synth, replay, callback, process |
| middleware | `[[middleware]] type = "..."` | retry, rate_limit, transform, auth |
| recording sink | `sink = { type = "..." }` | jsonl, bin |
| recording source | `source = { type = "..." }` | jsonl, bin |
| codec | `codec = { type = "..." }` | json, bytes |

Plugins register on these registries; see [PLUGINS.md](PLUGINS.md).

## next

- [PLUGINS.md](PLUGINS.md) — write your own factory
- `scenarios/` — runnable scenario fixtures
- `examples/plugin-skeleton/` — copy to start a plugin crate
