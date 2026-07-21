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
mod fan_bang;
mod filter_bang;
mod fixture_attr;
mod main_attr;
mod pipe_attr;
mod pipe_bang;
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
/// boilerplate. Emits every tier in the downward closure the function's
/// shape qualifies for (`Tier::plan` in `proxima-macros/src/pipe_attr.rs`) —
/// never just one — because the higher tiers are additive constraints on
/// the same root contract, not a replacement for it. It adds no new noun to
/// the pipe algebra: still exactly four standalone traits
/// (`proxima-primitives/src/pipe/primitives.rs`).
///
/// `sig.asyncness` decides the `Unpin` axis for free: `async fn` reaches
/// [`Pipe`] (RPITIT passthrough), plus `SendPipe` under `send`; a plain `fn`
/// is wrapped in `core::future::ready` (whose future IS `Unpin`), reaching
/// `UnpinPipe` as well, plus `UnpinSendPipe` under `send`. `Send` is never
/// inferred — only `#[proxima::piped(send)]` climbs to `SendPipe` /
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
/// #[proxima::piped]
/// async fn double(input: u64) -> Result<u64, Infallible> { Ok(input * 2) }
/// // -> struct double; impl Pipe for double { .. }
///
/// #[proxima::piped]
/// fn ring_pop(_: ()) -> Result<u8, Exhausted> { Ok(7) }
/// // -> struct ring_pop; impl UnpinPipe for ring_pop { .. }
///
/// #[proxima::piped(send)]
/// async fn fetch(url: String) -> Result<Bytes, Error> { .. }
/// // -> struct fetch; impl SendPipe for fetch { .. }
///
/// // stateful form: `Client` already exists, with its own field.
/// struct Proxy { client: Client }
///
/// #[proxima::piped(send)]
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
pub fn piped(args: TokenStream, item: TokenStream) -> TokenStream {
    pipe_attr::expand(args.into(), item.into())
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

/// Function-like leaf-lift sibling of `#[proxima::piped]`: `pipe!(closure)`
/// mints a `Pipe`/`SendPipe`/`UnpinPipe`/`UnpinSendPipe` value INLINE from a
/// closure literal, at an expression position, instead of requiring a named
/// top-level `fn`. Its own name, `pipe!`: a bang macro and an attribute
/// macro can't share one identifier while both exist (`#[pipe]` vs
/// `pipe!(..)` still collide in Rust's macro namespace, E0428) — which is
/// exactly why the attribute macro above is `#[proxima::piped]`, freeing
/// `pipe!` for this one.
///
/// Same tier vocabulary as the attribute macro minus its `boxed` escape
/// hatch — this bridge is zero-box by construction: `send`/`unpin` as a
/// trailing comma-separated tail (`name = ..` does not apply — nothing
/// needs to move aside for a name). A plain closure reaches every tier; an
/// `async` closure reaches `Pipe` only, never `UnpinPipe` (would need
/// `Box::pin`) or `send` (see `pipe_bang`'s module doc — the latter is a
/// genuine stable-Rust limitation, not a missing feature). Either refusal
/// points at `#[proxima::piped(unpin, boxed)]`/`#[proxima::piped(send)]` on a
/// hand-written `async fn` as the escape hatch. Passing an expression that
/// is not a closure literal passes it through unchanged.
///
/// ```ignore
/// let doubled = pipe!(|input: u64| -> Result<u64, Infallible> { Ok(input * 2) });
/// let piped = pipe!(doubled).and_then(pipe!(|input: u64| -> Result<u64, Infallible> { Ok(input + 1) }));
/// ```
#[proc_macro]
pub fn pipe(input: TokenStream) -> TokenStream {
    pipe_bang::expand(input.into())
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

/// `filter!(predicate closure)` — lift a closure into the decision-pipe
/// shape `filter.rs`'s own module doc names as the point of that file:
/// `In -> Result<In, Err>` (`Ok` admits, returning the input unchanged;
/// `Err` rejects). The SAME leaf-lift bridge `pipe!` builds, with one
/// extra macro-time check: the closure's admit type must equal its input
/// type. No collision with an existing attribute macro, so this one keeps
/// its natural name.
///
/// ```ignore
/// let gate = filter!(|input: u64| -> Result<u64, &'static str> {
///     if input < 100 { Ok(input) } else { Err("too big") }
/// });
/// ```
#[proc_macro]
pub fn filter(input: TokenStream) -> TokenStream {
    filter_bang::expand(input.into())
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

/// `fanout!(a, b, ..)` — variadic: build a [`FanOut`](proxima_primitives::pipe::FanOut)
/// over N arms in one call. Each arm is either a closure literal (leaf-lifted
/// the same way `pipe!` does) or an already-built pipe expression,
/// passed through. Variadic arity is the whole point: N closures are N
/// distinct, unnameable types, reconciled into `FanOut`'s single homogeneous
/// sink type via a macro-generated enum (one variant per arm) — zero boxes,
/// see `fan_bang`'s module doc for the full mechanism.
///
/// ```ignore
/// let fan = fanout!(
///     |input: u64| -> Result<(), Infallible> { println!("a: {input}"); Ok(()) },
///     |input: u64| -> Result<(), Infallible> { println!("b: {input}"); Ok(()) },
/// );
/// ```
#[proc_macro]
pub fn fanout(input: TokenStream) -> TokenStream {
    fan_bang::expand_fanout(input.into())
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

/// `fanin!(a, b, ..)` — variadic: build a [`FanIn`](proxima_primitives::pipe::FanIn)
/// over N arms in one call, merged with [`Select::RoundRobin`](proxima_primitives::pipe::Select).
/// Same enum-of-arms mechanism as `fanout!`, with one extra restriction
/// `FanIn` itself imposes: each arm must be `UnpinPipe<In = (), Err =
/// Exhausted> + DropSafe` — a synchronous, never-suspending source — so a
/// closure-literal arm must be a plain (non-`async`) closure. An async
/// source can still participate: lift it first with
/// `#[proxima::piped(unpin, boxed)]` on a hand-written `async fn` and pass
/// the result in as a pass-through arm.
///
/// ```ignore
/// let merged = fanin!(
///     |(): ()| -> Result<u8, Exhausted> { Ok(1) },
///     |(): ()| -> Result<u8, Exhausted> { Ok(2) },
/// );
/// ```
#[proc_macro]
pub fn fanin(input: TokenStream) -> TokenStream {
    fan_bang::expand_fanin(input.into())
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
