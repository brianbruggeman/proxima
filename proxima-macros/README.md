# proxima-macros

The proc-macro crate behind proxima's test fixtures, runtime entrypoints, pipe lifts, telemetry spans, schema description and error derives.

Add it with `cargo add proxima-macros`. Most users reach these macros through the `proxima` facade (`#[proxima::test]`, `#[proxima::main]`, ...) rather than depending on this crate directly.

## What it provides

Attribute macros:

- `#[fixture]`: defines a fixture consumed by `#[test]` parameters, resolved by parameter name.
- `#[test]`: one test attribute that drives the body on proxima's prime runtime (tokio via `runtime = "tokio"`), with `#[case]` parameterization and scheduling metadata (`serial`, `group`, `process`).
- `#[main]`: the production sibling of `#[test]`; turns `async fn main()` into a sync `main` that boots a runtime and drives the body.
- `#[piped]`: generates a `Pipe` / `SendPipe` / `UnpinPipe` / `UnpinSendPipe` impl from a plain function or an inherent `impl` block.
- `#[span]` and `#[instrument]`: wrap a function body in a proxima span (`instrument` is the same expansion under the name for "make this observable").

Function-like macros that lift closures into pipes: `pipe!`, `filter!`, `fanout!`, `fanin!`.

Derives: `SpanCarrier`, `Error` (with `#[error]`, `#[source]`, `#[from]`), `Schema` (with `#[schema]`).

## What it composes

The pipe macros emit impls of the traits defined in `proxima-primitives` (`Pipe`, `SendPipe`, `UnpinPipe`, `UnpinSendPipe`; `FanOut` / `FanIn` for `fanout!` / `fanin!`) and add no new trait. The span macros emit calls into `proxima-telemetry`. The `Schema` derive emits paths into `proxima-config`'s `schema` module.

A proc-macro crate has no `$crate`, so it depends on none of those crates. Paths into `proxima-primitives`, `proxima-core` and `proxima-telemetry` are resolved with `proc-macro-crate` against what the invoking crate depends on: the owning crate itself, a direct dependency, or only the `proxima` facade. `#[derive(Schema)]` instead emits `::proxima_config::schema::...` directly, so the deriving crate must depend on `proxima-config` with its `schema` feature.

## Features and tiers

No cargo features. It is a `proc-macro = true` crate and runs at compile time on the host; the code it generates inherits the tier of the crate it expands in (the `Schema` derive names `alloc` rather than `std`, so it is usable from alloc-tier crates).

## License

MIT OR Apache-2.0.

Repository: <https://github.com/brianbruggeman/proxima>
