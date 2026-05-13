# Build a plugin

**Prerequisites:** [Foundations](./00-foundations.md) — a wrapping `Pipe`, and `App::new()`.
**You will:** package a `Pipe` as a reusable crate others compose in one line — a `Pipe` + a `PipeFactory` + a `register(builder)` entry point.
**New concepts (in order):** a wrapping `Pipe` (`StampHeader`) · a `PipeFactory` (build the pipe from a config spec + an inner pipe) · `PluginRegistry` and the `register(builder)` convention · `App::builder()`, the builder form of `App::new()`.
**Answer key:** [`examples/plugin-skeleton/src/lib.rs`](../../examples/plugin-skeleton/src/lib.rs) — a plugin *crate*, composed by consumers via `my_plugin::register(App::builder().with_defaults()?)?.build()?`.

The example frames it, verbatim from its own module doc-comment (`plugin-skeleton/src/lib.rs:3-6`): *"Canonical plugin skeleton. Replace `StampHeader` with your own pipe and rename the crate. The composition pattern at the bottom (`register(builder) -> Result<AppBuilder>`) is the convention plugin crates expose so users get one-line composition."*

## 1. Your pipe — a wrapper

The plugin's pipe wraps an inner `PipeHandle` and adds behavior. `StampHeader` stamps a configurable header onto every response, copied verbatim from `plugin-skeleton/src/lib.rs:28-51`:

```rust
pub struct StampHeader {
    inner: PipeHandle,
    name: String,
    value: String,
}

impl SendPipe for StampHeader {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let inner = self.inner.clone();
        let header_name = self.name.clone();
        let header_value = self.value.clone();
        async move {
            let response = SendPipe::call(&inner, request).await?;
            Ok(response.with_header(header_name, header_value))
        }
    }
}
```

That is the entire pipe: one struct, one `SendPipe` impl, exactly the trait Foundations section 6 taught (this one climbs to the `SendPipe` tier because it is served HTTP traffic — section 13's requirement). There is no `name()` method, no extra trait to implement — `SendPipe` has none beyond `call` and `and_then`.

Same wrapping shape as the [gateway](./build-an-api-gateway.md)'s `Auth` or the [cache](./build-a-caching-reverse-proxy.md)'s `WriteBack` — a `Pipe` around a `Pipe`.

## 2. A factory — build the pipe from config

A `PipeFactory` builds your pipe from a `spec: &serde_json::Value` — arbitrary JSON config — and an optional inner pipe. This is config-as-composition: the config selects and parameterizes the pipe. Copied verbatim from `plugin-skeleton/src/lib.rs:69-100` (the trivial `StampHeaderFactory::new()`/`Default` boilerplate at lines 54-67 is omitted here for space):

```rust
impl PipeFactory for StampHeaderFactory {
    fn name(&self) -> &str {
        "stamp_header"
    }

    fn build(
        &self,
        spec: &Value,
        inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let header_name = spec
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("x-stamp")
            .to_string();
        let header_value = spec
            .get("value")
            .and_then(Value::as_str)
            .unwrap_or("set")
            .to_string();
        Box::pin(async move {
            let inner = inner.ok_or_else(|| {
                ProximaError::Config("stamp_header requires an inner pipe".into())
            })?;
            Ok(into_handle(StampHeader {
                inner,
                name: header_name,
                value: header_value,
            }))
        })
    }
}
```

Unlike `StampHeader::call` above, `PipeFactory` puts a `name()` method right on the trait itself (`proxima-primitives/src/pipe/pipe_factory.rs:15`) — that is where a factory's registered name comes from, not from the pipe it builds.

That `Box::pin` looks like it contradicts Foundations ("you never write `Box::pin` to make a pipe") — it does not, and here is why. Foundations' promise is about a pipe's own `call`: `StampHeader::call` in section 1 still returns `impl Future` directly, no boxing. `PipeFactory::build` is different because it is *type-erased* — the registry holds many factories for many different pipe types behind one trait object (`dyn PipeFactory`), so `build`'s return type cannot be a distinct `impl Future` per pipe; it has to be one concrete type every factory shares, and `Pin<Box<dyn Future<...>>>` is that shared type. This is the one place boxing appears in the plugin skeleton, and it lives in the factory, never in the pipe's own `call`.

The factory names the pipe (`"stamp_header"`) so config can reference it, reads its knobs from the spec, and requires an inner pipe — if one is missing it returns `ProximaError::Config(...)`, one of `ProximaError`'s named variants for a bad or missing configuration value.

## 3. `PluginRegistry` and `register(builder)` — the one-line entry point

A plugin needs somewhere to register its factory. That somewhere is a trait, not the concrete app type, so a plugin crate can add itself without depending on the whole `proxima` umbrella crate. `PluginRegistry` is that trait, copied verbatim from `proxima-primitives/src/pipe/plugin.rs:16-25`:

```rust
pub trait PluginRegistry: Sized {
    /// Register a pipe factory under the name embedded in the factory.
    /// Returns the builder with the factory registered. Errors if a
    /// factory with the same name is already registered.
    ///
    /// # Errors
    /// Returns `ProximaError` if registration fails (duplicate name,
    /// invalid factory shape, etc.).
    fn with_upstream_factory(self, factory: DynPipeFactory) -> Result<Self, ProximaError>;
}
```

One method, `with_upstream_factory(self, factory) -> Result<Self, ProximaError>`, that takes a factory and hands back the same builder with it registered (or an error, e.g. a duplicate name). `DynPipeFactory` is `Arc<dyn PipeFactory>` (`proxima-primitives/src/pipe/pipe_factory.rs:30`) — the type-erased form, since the registry holds many factories for many different pipe types side by side. The app's builder type, `AppBuilder`, implements `PluginRegistry` (`src/app_builder.rs:193-197`), so anything written against the trait works against a real app.

The convention plugin crates expose: a `register` function, generic over any `PluginRegistry`, that adds the factory to a builder and returns it. Copied verbatim from `plugin-skeleton/src/lib.rs:104-106`:

```rust
pub fn register<R: PluginRegistry>(builder: R) -> Result<R, ProximaError> {
    builder.with_upstream_factory(Arc::new(StampHeaderFactory::new()))
}
```

`Arc::new(...)` wraps the factory for shared ownership — the registry, and anything else holding a reference to it, can share the one factory instance safely.

Consumers compose it in one line — this is the crate's own module doc-comment, copied verbatim from `plugin-skeleton/src/lib.rs:9-13`:

```rust
use proxima::App;
let app = my_plugin::register(
    App::builder().with_defaults()?
)?
.build()?;
```

Read it left to right:

- `App::builder()` — a new concept this tutorial introduces: the builder form of `App::new()` (Foundations section 13). Same `App`, but assembled step by step so a plugin can add to it before it is built. `App::builder()` is defined at `src/app.rs:637`, and returns an `AppBuilder` (`src/app_builder.rs:48`).
- `.with_defaults()?` — fills in the builder's default pipes and settings (`src/app_builder.rs:96`); `?` propagates a `ProximaError` if that fails.
- `my_plugin::register(...)?` — hands the builder to your plugin, which registers its factory and returns the builder, or a `ProximaError` if registration failed.
- `.build()?` — turns the finished builder into a real `App` (`src/app_builder.rs:259`), ready to `mount` and serve.

## What you built

A reusable plugin crate from three pieces:

- **a wrapping `Pipe`** — your logic, `Request -> Response`, around an inner pipe.
- **a `PipeFactory`** — builds the pipe from a config spec + an inner pipe; config-as-composition.
- **`register(builder)`** — generic over any `PluginRegistry`, the one-line composition convention others depend on.

To ship your own: replace `StampHeader` with your pipe, rename the crate, keep the factory + `register` shape. A plugin is just a `Pipe` with a factory and an entry point.
