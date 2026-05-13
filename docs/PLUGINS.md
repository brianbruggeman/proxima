# plugin authoring

Plugins are normal Rust crates that depend on `proxima` and register
factories on an `AppBuilder` at startup. No fork, no FFI, no
dynamic loader.

## substrate registries

| substrate | trait | config dispatch |
| --- | --- | --- |
| listen protocol | `ListenProtocol` | `[[listen]] type = "..."` |
| pipe | `PipeFactory` | `type = "..."` (or shorthand) |
| recording sink | `RecordingSink` / `DynRecordingSink` | none — composed programmatically (`AccumulatingSink`, `LazyFanOut`, `EventTap`), no factory/registry/config dispatch |
| recording source | `RecordingSourceFactory` | `source = { type = "..." }` |
| codec | `CodecFactory` | `codec = { type = "..." }` |

Wrapping pipes (auth, retry, rate_limit, transform, validate, …) and
terminal pipes (http, synth, kv, fs, …) share `PipeFactory`. The
factory's `inner: Option<PipeHandle>` parameter distinguishes them at
build time: wrapping factories require `Some`, terminal factories ignore
`None`.

## composing a plugin into an app

```rust
use proxima::App;

let app = App::builder()
    .with_defaults()?
    .with_upstream_factory(Arc::new(my_plugin::JitterFactory))?
    .build()?;
```

Skip `with_defaults()` for an empty substrate.

## pipe factory skeleton

```rust
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use proxima::{
    ProximaError, Request, Response, Pipe, PipeFactory, PipeHandle, into_handle,
};
use serde_json::Value;

pub struct JitterFactory;

impl PipeFactory for JitterFactory {
    fn name(&self) -> &str { "jitter" }

    fn build(
        &self,
        spec: &Value,
        inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let max_us = spec.get("max_us").and_then(Value::as_u64).unwrap_or(500);
        Box::pin(async move {
            let inner = inner.ok_or_else(|| {
                ProximaError::Config("jitter requires an inner pipe".into())
            })?;
            Ok(into_handle(Jitter { inner, max_us }))
        })
    }
}

struct Jitter { inner: PipeHandle, max_us: u64 }

impl Pipe for Jitter {
    fn call(&self, request: Request)
        -> impl Future<Output = Result<Response, ProximaError>> + Send
    {
        let inner = self.inner.clone();
        let max = self.max_us;
        async move {
            tokio::time::sleep(std::time::Duration::from_micros(fastrand::u64(0..=max))).await;
            Pipe::call(&inner, request).await
        }
    }
}
```

Config:

```toml
[[pipe]]
name = "hello-name"
synth = { status = 200, body_template = "hello, {{body.name}}\n" }

[[pipe.middleware]]
type = "jitter"
max_us = 1500
```

The loader desugars each pipe's nested `[[pipe.middleware]]` array into a
pipe tree via `PipeFactory::build` (see `examples/config/04-hello-name/proxima.toml`
and `src/settings/chain.rs`). Terminal pipes use the same trait; their
factories ignore `inner`.

## recording sink wrapper skeleton

There is no `RecordingSinkFactory`/registry — recording sinks (unlike
recording *sources*) are composed directly in code by wrapping a
`DynRecordingSink` in another `RecordingSink` impl, the same way `EventTap`
wraps an inner sink to add live tailing (`proxima-recording/src/pipe/event_sink.rs`):

```rust
use proxima_recording::event::RecordingEvent;
use proxima_recording::pipe::{AppendFuture, DynRecordingSink, RecordingSink};

pub struct WrappingSink {
    inner: DynRecordingSink,
}

impl RecordingSink for WrappingSink {
    fn append<'lifetime>(&'lifetime self, event: RecordingEvent) -> AppendFuture<'lifetime> {
        Box::pin(async move {
            // wrapper-specific work goes here, then delegate
            self.inner.append(event).await
        })
    }

    fn flush<'lifetime>(&'lifetime self) -> AppendFuture<'lifetime> {
        self.inner.flush()
    }
}
```

Sink wrappers hold their inner `DynRecordingSink` directly and delegate —
there is no spec/registry indirection to resolve an `inner` config value at
build time the way `PipeFactory`/`RecordingSourceFactory` do.

## starter

Copy `examples/plugin-skeleton/` into a sibling crate and replace the
identifiers. The skeleton already wires `Cargo.toml`, a `register(builder)`
entry point, and unit tests.

## why static linking

Rust has no stable ABI and allocators don't cross dynamic boundaries
cleanly. Static link means type-safe registration, zero call overhead,
shared allocator, and panic-safety. The trade is one binary per
plugin combination — fine for this workload.
