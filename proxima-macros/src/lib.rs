// proc-macros legitimately use unwrap/expect for malformed input —
// the panic is the error path (rustc surfaces it as a compile error
// at the macro invocation site, which is exactly the right shape).
// also allow type_complexity: proc-macro intermediate parse states
// have a wide tuple shape that's clearer than naming each sub-type.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity
)]

use proc_macro::TokenStream;

mod describe;
mod error_derive;
mod fixture_attr;
mod main_attr;
mod pipe_attr;
mod runtime_args;
mod span_attr;
mod span_carrier;
mod test_attr;

/// Defines a fixture consumed by `#[proxima::test]` parameters. Native
/// reimplementation of rstest's fixture model (no rstest dependency): generates
/// a `struct` with `async fn get/default/partial_N`, resolving dependency
/// fixtures by parameter name (`#[default(expr)]` / `#[from(path)]` override).
///
/// ```ignore
/// #[proxima::fixture]
/// fn port() -> u16 { 8080 }
///
/// #[proxima::fixture]
/// async fn client(port: u16) -> Client { Client::connect(port).await }
/// ```
#[proc_macro_attribute]
pub fn fixture(args: TokenStream, item: TokenStream) -> TokenStream {
    fixture_attr::expand(args.into(), item.into())
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

/// One test attribute that drives the body on proxima's prime runtime
/// (tokio fallback via `runtime = "tokio"`). Subsumes `#[tokio::test]`;
/// `#[rstest]` parameterization + cassette record/replay land in later slices.
///
/// ```ignore
/// #[proxima::test]
/// async fn round_trips() { assert_eq!(2 + 2, 4); }
///
/// #[proxima::test(runtime = "tokio")]
/// async fn on_tokio() { /* !Send-friendly body */ }
/// ```
#[proc_macro_attribute]
pub fn test(args: TokenStream, item: TokenStream) -> TokenStream {
    test_attr::expand(args.into(), item.into())
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

/// Production sibling of `#[proxima::test]`: turns `async fn main() -> R` into
/// a sync `fn main() -> R` that boots a runtime and drives the body to
/// completion via `proxima::runtime::run*`. Same runtime surface as the
/// test macro (adaptive default — prime when compiled, else tokio).
///
/// ```ignore
/// #[proxima::main]
/// async fn main() { /* adaptive: prime when compiled, else tokio */ }
///
/// #[proxima::main(runtime = "tokio", flavor = "multi_thread")]
/// async fn main() -> std::process::ExitCode { /* hyper/axum/TokioPerCore bin */ }
///
/// #[proxima::main(runtime = "prime")]
/// async fn main() -> Result<(), proxima::ProximaError> { /* prime serve path */ }
/// ```
#[proc_macro_attribute]
pub fn main(args: TokenStream, item: TokenStream) -> TokenStream {
    main_attr::expand(args.into(), item.into())
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

/// Generates a [`Pipe`]/`SendPipe`/`UnpinPipe`/`UnpinSendPipe` impl from a
/// plain function, removing the hand-written unit-struct-plus-impl
/// boilerplate. Picks exactly one of the four standalone tiers
/// (`proxima-primitives/src/pipe/primitives.rs`) — it adds no new noun to
/// the pipe algebra.
///
/// `sig.asyncness` decides the `Unpin` axis for free: `async fn` emits
/// [`Pipe`] (RPITIT passthrough); a plain `fn` emits `UnpinPipe`, wrapping
/// the call in `core::future::ready` (whose future IS `Unpin`). `Send` is
/// never inferred — only `#[proxima::pipe(send)]` climbs to `SendPipe` /
/// `UnpinSendPipe`. The generated struct is always fieldless, so it always
/// derives `Clone` unconditionally.
///
/// Also accepts a plain inherent `impl Foo { fn call(..) { .. } }` block, for
/// a STATEFUL pipe whose struct already carries its own fields — no struct
/// is generated there, `Foo` is relocated as-is into `impl #trait for Foo`.
///
/// # Arguments
///
/// - `send` — climb to the cross-core `SendPipe`/`UnpinSendPipe` form.
/// - `unpin` — asserts the (already-automatic) `Unpin` tier on a sync fn;
///   on an `async fn` this is a compile error (an async block's future is
///   never `Unpin`).
/// - `name = Ident` — give the pipe its own name instead of the fn's. The fn
///   then keeps its name and stays directly callable; by default the pipe
///   wears it, and the fn moves aside (both would live in the value
///   namespace). Does not apply to the impl-block form.
///
/// # Examples
///
/// ```ignore
/// #[proxima::pipe]
/// async fn double(input: u64) -> Result<u64, Infallible> { Ok(input * 2) }
/// // -> struct double; impl Pipe for double { .. }
///
/// #[proxima::pipe]
/// fn ring_pop(_: ()) -> Result<u8, Exhausted> { Ok(7) }
/// // -> struct ring_pop; impl UnpinPipe for ring_pop { .. }
///
/// #[proxima::pipe(send)]
/// async fn fetch(url: String) -> Result<Bytes, Error> { .. }
/// // -> struct fetch; impl SendPipe for fetch { .. }
///
/// // stateful form: `Client` already exists, with its own field.
/// struct Proxy { client: Client }
///
/// #[proxima::pipe(send)]
/// impl Proxy {
///     async fn call(&self, request: Request<Bytes>) -> Result<Response<Bytes>, Error> {
///         self.client.clone().call(request).await
///     }
/// }
/// // -> impl SendPipe for Proxy { .. }, no struct generated
/// ```
///
/// [`Pipe`]: https://docs.rs/proxima-primitives/latest/proxima_primitives/pipe/trait.Pipe.html
#[proc_macro_attribute]
pub fn pipe(args: TokenStream, item: TokenStream) -> TokenStream {
    pipe_attr::expand(args.into(), item.into())
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

/// Attribute macro for auto-spanning a function.
///
/// Wraps the function body in a proxima span so every call is recorded. With no
/// `recorder = ...`, the span resolves the process-wide ambient recorder via
/// `Recorder::current()` (installed by `set_default_recorder` /
/// `RecorderBuilder::install`) — zero wiring — and runs the body span-free when
/// none is installed, the same no-op contract as the `info!` / `debug!` macros.
///
/// # Arguments
///
/// - `name = "..."` — span name; defaults to the function name
/// - `level = "..."` — one of `trace`, `debug`, `info`, `warn`, `error`; defaults to `info`
/// - `recorder = <expr>` — expression resolving to `&Recorder`; defaults to the ambient recorder
/// - `parent = <expr>` — expression resolving to `Option<&[u8]>`, a W3C `traceparent`
///   (e.g. `RequestContext::traceparent()`, or bytes carried by hand from a caller's
///   own span). `Some` continues that trace (same `trace_id`, `parent_span_id` set);
///   `None`/absent opens a fresh root. Proxima carries span context as explicit
///   data — never an ambient/thread-local "current span" — so a caller wanting a
///   child span MUST pass this.
///
/// # Examples
///
/// ```ignore
/// #[span]
/// fn do_work(input: &str) -> usize { input.len() }
///
/// #[span(name = "explicit", level = "warn")]
/// async fn fetch(url: &str) -> Result<String, Error> { ... }
///
/// #[span(parent = request.context.traceparent())]
/// fn handle(request: &Request) { ... }
/// ```
#[proc_macro_attribute]
pub fn span(args: TokenStream, item: TokenStream) -> TokenStream {
    span_attr::expand(args.into(), item.into())
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

/// The unified observability annotation: the same expansion as [`span`], named
/// for what it produces — a unit of work made observable across pillars (the
/// trace span plus, behind `instrument-metrics`, its duration histogram). Use
/// this when you mean "instrument this function," `span` when you mean "open a
/// span"; they are one mechanism.
#[proc_macro_attribute]
pub fn instrument(args: TokenStream, item: TokenStream) -> TokenStream {
    span_attr::expand(args.into(), item.into())
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

/// Derive macro that implements `SpanCarrier` for a struct.
///
/// The struct must have either a field named `span_id` of type
/// `Option<SpanId>`, or exactly one field annotated `#[span_id]`.
///
/// # Examples
///
/// ```ignore
/// #[derive(SpanCarrier)]
/// struct Envelope {
///     span_id: Option<SpanId>,
///     payload: Vec<u8>,
/// }
///
/// #[derive(SpanCarrier)]
/// struct Request {
///     #[span_id]
///     trace_slot: Option<SpanId>,
///     body: Bytes,
/// }
/// ```
#[proc_macro_derive(SpanCarrier, attributes(span_id))]
pub fn derive_span_carrier(item: TokenStream) -> TokenStream {
    span_carrier::expand(item.into())
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

/// Derive macro that implements `Display` + `core::error::Error` on an
/// enum following the project's conventions (lowercase messages, no
/// trailing punctuation, typed `#[source]` only — no `Box<dyn Error>`).
///
/// Mirrors the most-used surface of `thiserror::Error`. Emits code that
/// compiles under `#![no_std]` with no `alloc` requirement, provided
/// the user's enum variants don't carry alloc-bearing payloads.
///
/// # Supported attributes
///
/// - `#[error("literal text")]` on a variant — emits `write!(f, "literal text")`
/// - `#[error("with {0}")]` / `#[error("with {field}")]` — positional /
///   named field interpolation in the Display message
/// - `#[error(transparent)]` — delegates Display + `source()` to the
///   single inner field (variant must have exactly one field)
/// - `#[source]` on a variant field — exposes the field via
///   `core::error::Error::source()`
/// - `#[from]` on a single tuple-variant field — additionally generates
///   `impl From<Inner> for Outer { fn from(v) -> Self { Self::Variant(v) } }`
///   and treats the field as a `#[source]`.
///
/// # Examples
///
/// ```ignore
/// use proxima_macros::Error;
///
/// #[derive(Error, Debug)]
/// pub enum DecodeError {
///     #[error("invalid magic byte: {0}")]
///     InvalidMagic(u8),
///
///     #[error("truncated frame")]
///     TruncatedFrame,
///
///     #[error("upstream error")]
///     Upstream(#[source] UpstreamError),
///
///     #[error(transparent)]
///     Wire(WireError),
/// }
/// ```
#[proc_macro_derive(Error, attributes(error, source, from))]
pub fn derive_error(item: TokenStream) -> TokenStream {
    error_derive::expand(item.into())
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

/// Derive macro that generates a `proxima_config::schema::Schema` from a
/// struct, so the typed shape is the single source of truth and the contract
/// cannot drift from the Rust type.
///
/// # Supported attributes
///
/// - `#[schema(rename = "wire_name")]` on a field — use a different name in the
///   schema (match a serde rename so the contract tracks the wire).
/// - `#[schema(skip)]` on a field — omit it from the schema.
///
/// `Option<T>` fields are marked optional (absent-allowed) automatically.
///
/// # Examples
///
/// ```ignore
/// use proxima_config::schema::{Schema, Describe};
///
/// #[derive(Schema)]
/// struct Memory {
///     id: String,
///     score: Option<f64>,
///     #[schema(rename = "type")]
///     kind: String,
/// }
/// ```
#[proc_macro_derive(Schema, attributes(schema))]
pub fn derive_schema(item: TokenStream) -> TokenStream {
    describe::expand(item.into())
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}
