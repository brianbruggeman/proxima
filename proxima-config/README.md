# proxima-config

The config concern for proxima in one crate: a pluggable config-format registry, a desugar pass for pipe specs, a typed schema IR, and a desired-state file store.

Add it with `cargo add proxima-config`.

## What it provides

Four modules, each behind its own feature, none depending on the others:

- the crate root: a format registry. Parsers map source text to `serde_json::Value`; `ConfigFormatRegistry` dispatches by file extension, or by trial-parse sniffing when no hint is given. Formats: JSON (`JsonConfigFormat`), JSON5, TOML, YAML (via `serde_norway`), RON and XML. `ConfigFormatFactory` is the trait a new format implements.
- `sugar` (feature `sugar`): a pure rewrite from sugary specs to primitive specs (`cache = true`, `mock = { ... }`, `replay = "..."`, `mcp = { ... }`, `record = "..."`), plus the `SpecBuilder` seam the fluent builders share with it, so a fluent builder and a config file meet on one spec.
- `schema` (feature `schema`): a typed value-shape IR (`Describe`, `Schema`), a `validate` walk, and JSON Schema / OpenAPI / TOML emitters.
- `store` (feature `store`): the k8s-style desired-state file store. `PipeRecord` / `ListenerRecord` are the records; `StateStore` persists them as `pipes/<name>.toml` and `listeners/<name>.toml` under `${PROXIMA_STATE_DIR:-~/.proxima/state.d/}` with atomic writes.

## What it composes

- `proxima-core`: `ProximaError`, and `proxima-core/std` under `std`.
- `proxima-macros` (optional, feature `schema-derive`): the `#[derive(Schema)]` macro. The derive emits `::proxima_config::schema::...` paths, so a crate that derives `Schema` depends on this crate.
- External: `serde_json` always; `toml`, `json5`, `serde_norway`, `ron`, `quick-xml` and `arc-swap` under `std`.

Several of these modules were separate crates before (`proxima-sugar`, `proxima-schema`, `proxima-state-store`); they are folded in here.

## Features and tiers

Default feature: `std`. Nothing but the format registry is on by default; the other three modules are opt-in.

| feature | effect |
|---|---|
| `alloc` | `ConfigFormatFactory`, `DynConfigFormatFactory`, `JsonConfigFormat` on no_std + alloc |
| `std` (default) | implies `alloc`; adds `ConfigFormatRegistry` and the TOML, JSON5, YAML, RON and XML formats |
| `sugar` | the desugar pass (alloc tier) |
| `schema` | schema IR, `Describe`, JSON Schema / OpenAPI emit (alloc tier) |
| `schema-std` | adds `SchemaRegistry`, the TOML emitter and the `serde_value` escape hatch |
| `schema-derive` | adds `#[derive(Schema)]` |
| `store` | `PipeRecord` / `ListenerRecord` (alloc tier) |
| `store-std` | adds `StateStore` (filesystem, atomic write) |

On no_std, compose formats directly (`JsonConfigFormat.parse(raw)`) instead of going through the registry, which needs `arc-swap`.

## License

MIT OR Apache-2.0.

Repository: <https://github.com/brianbruggeman/proxima>
