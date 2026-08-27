// a proc-macro's error path IS a panic: rustc catches it and reports a compile
// error at the invocation site, so `expect` on a malformed-input invariant is
// the right shape rather than a swallowed Result.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proc_macro::TokenStream;

mod crate_path;
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
/// ```
/// #[proxima_macros::fixture]
/// fn port() -> u16 { 8080 }
///
/// // a parameter is resolved as the fixture of the same name, so `endpoint`
/// // gets `port`'s value with nothing wired by hand.
/// #[proxima_macros::fixture]
/// async fn endpoint(port: u16) -> String { format!("127.0.0.1:{port}") }
///
/// # async fn check() {
/// assert_eq!(endpoint::default().await, "127.0.0.1:8080");
/// assert_eq!(endpoint::partial_1(9000).await, "127.0.0.1:9000");
/// # }
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
/// ```
/// #[proxima_macros::test]
/// async fn round_trips() { assert_eq!(2 + 2, 4); }
///
/// #[proxima_macros::test(runtime = "tokio")]
/// async fn on_tokio() { tokio::task::yield_now().await; }
///
/// // parameterized: one `#[test]` per case, named for the case description.
/// #[proxima_macros::test]
/// #[case::small(1u32)]
/// #[case::large(4_000_000_000u32)]
/// async fn round_trips_each_case(#[case] value: u32) {
///     assert_eq!(u32::from_le_bytes(value.to_le_bytes()), value);
/// }
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
/// Any `?`-propagated error type works, not just `ProximaError` — the macro
/// never inspects `R`. A bare, non-`Send` `Box<dyn std::error::Error>` only
/// compiles under `runtime = "tokio"`: the prime backend moves the body's
/// output across a driver-core channel, which requires `Send`.
///
/// ```no_run
/// // adaptive: prime when compiled, else tokio.
/// #[proxima_macros::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
///     Ok(())
/// }
/// ```
///
/// The return type is preserved verbatim, so `()`, `Result<T, E>`, and
/// `ExitCode` all flow through:
///
/// ```no_run
/// #[proxima_macros::main(cores = 4)]
/// async fn main() -> std::process::ExitCode { std::process::ExitCode::SUCCESS }
/// ```
///
/// `runtime = "tokio"` / `flavor = "multi_thread"` / `worker_threads = N` select
/// an explicit tokio backend instead (they need the umbrella's `tokio` feature,
/// which is why they are not shown as a compiled example here — this crate's own
/// dev-dependencies deliberately do not pull that closure). `cores` / `affinity`
/// are the prime/adaptive path's vocabulary and are mutually exclusive with it.
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
/// ```
/// use std::convert::Infallible;
/// use proxima::pipe::{Exhausted, Pipe, SendPipe, UnpinPipe};
///
/// // -> struct double; impl Pipe for double { .. }
/// #[proxima_macros::piped]
/// async fn double(input: u64) -> Result<u64, Infallible> { Ok(input * 2) }
///
/// // a plain fn is wrapped in `core::future::ready`, so it also reaches Unpin.
/// // -> struct ring_pop; impl Pipe + UnpinPipe for ring_pop { .. }
/// #[proxima_macros::piped]
/// fn ring_pop(_: ()) -> Result<u8, Exhausted> { Ok(7) }
///
/// // `send` is never inferred — it is opted into.
/// // -> struct fetch; impl Pipe + SendPipe for fetch { .. }
/// #[proxima_macros::piped(send)]
/// async fn fetch(url: String) -> Result<usize, Infallible> { Ok(url.len()) }
///
/// // stateful form: the struct already carries its own fields, so no struct
/// // is generated — `Counter` is relocated into `impl SendPipe for Counter`.
/// #[derive(Clone)]
/// struct Counter { start: u64 }
///
/// #[proxima_macros::piped(send)]
/// impl Counter {
///     async fn call(&self, step: u64) -> Result<u64, Infallible> {
///         Ok(self.start + step)
///     }
/// }
///
/// # async fn check() {
/// assert_eq!(Pipe::call(&double, 21).await, Ok(42));
/// assert_eq!(UnpinPipe::call(&ring_pop, ()).await, Ok(7));
/// assert_eq!(SendPipe::call(&Counter { start: 40 }, 2).await, Ok(42));
/// # }
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
/// ```
/// use std::convert::Infallible;
/// use proxima::pipe::{Pipe, PipeExt};
/// use proxima_macros::pipe;
///
/// # async fn check() {
/// let doubled = pipe!(|input: u64| -> Result<u64, Infallible> { Ok(input * 2) });
/// let composed = doubled.and_then(pipe!(|input: u64| -> Result<u64, Infallible> { Ok(input + 1) }));
/// assert_eq!(Pipe::call(&composed, 20).await, Ok(41));
/// # }
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
/// ```
/// use proxima::pipe::Pipe;
/// use proxima_macros::filter;
///
/// # async fn check() {
/// let gate = filter!(|input: u64| -> Result<u64, &'static str> {
///     if input < 100 { Ok(input) } else { Err("too big") }
/// });
/// assert_eq!(Pipe::call(&gate, 7).await, Ok(7));
/// assert_eq!(Pipe::call(&gate, 900).await, Err("too big"));
/// # }
/// ```
#[proc_macro]
pub fn filter(input: TokenStream) -> TokenStream {
    filter_bang::expand(input.into())
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

/// `fanout!(a, b, ..)` — variadic: build a [`FanOut`] over N arms in one
/// call. Each arm is either a closure literal (leaf-lifted
/// the same way `pipe!` does) or an already-built pipe expression,
/// passed through. Variadic arity is the whole point: N closures are N
/// distinct, unnameable types, reconciled into `FanOut`'s single homogeneous
/// sink type via a macro-generated enum (one variant per arm) — zero boxes,
/// see `fan_bang`'s module doc for the full mechanism.
///
/// ```
/// use std::convert::Infallible;
/// use std::sync::atomic::{AtomicU64, Ordering};
/// use proxima::pipe::Pipe;
/// use proxima_macros::fanout;
///
/// static SEEN: AtomicU64 = AtomicU64::new(0);
///
/// # async fn check() {
/// let fan = fanout!(
///     |input: u64| -> Result<(), Infallible> { SEEN.fetch_add(input, Ordering::Relaxed); Ok(()) },
///     |input: u64| -> Result<(), Infallible> { SEEN.fetch_add(input, Ordering::Relaxed); Ok(()) },
/// );
/// Pipe::call(&fan, 21).await.expect("both arms admit");
/// assert_eq!(SEEN.load(Ordering::Relaxed), 42);
/// # }
/// ```
///
/// [`FanOut`]: https://docs.rs/proxima-primitives/latest/proxima_primitives/pipe/struct.FanOut.html
#[proc_macro]
pub fn fanout(input: TokenStream) -> TokenStream {
    fan_bang::expand_fanout(input.into())
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

/// `fanin!(a, b, ..)` — variadic: build a [`FanIn`] over N arms in one call,
/// merged with [`Select::RoundRobin`]. Same enum-of-arms mechanism as
/// `fanout!`, with one extra restriction
/// `FanIn` itself imposes: each arm must be `UnpinPipe<In = (), Err =
/// Exhausted> + DropSafe` — a synchronous, never-suspending source — so a
/// closure-literal arm must be a plain (non-`async`) closure. An async
/// source can still participate: lift it first with
/// `#[proxima::piped(unpin, boxed)]` on a hand-written `async fn` and pass
/// the result in as a pass-through arm.
///
/// ```
/// use proxima::pipe::{Exhausted, UnpinPipe};
/// use proxima_macros::fanin;
///
/// # async fn check() {
/// let merged = fanin!(
///     |(): ()| -> Result<u8, Exhausted> { Ok(1) },
///     |(): ()| -> Result<u8, Exhausted> { Ok(2) },
/// );
/// // RoundRobin: the first pull takes arm 0, the second arm 1.
/// assert_eq!(UnpinPipe::call(&merged, ()).await, Ok(1));
/// assert_eq!(UnpinPipe::call(&merged, ()).await, Ok(2));
/// # }
/// ```
///
/// [`FanIn`]: https://docs.rs/proxima-primitives/latest/proxima_primitives/pipe/struct.FanIn.html
/// [`Select::RoundRobin`]: https://docs.rs/proxima-primitives/latest/proxima_primitives/pipe/enum.Select.html
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
/// - `kind = "..."` — one of `internal`, `server`, `client`, `producer`, `consumer`;
///   defaults to `internal`
/// - `recorder = <expr>` — expression resolving to `&Recorder`; defaults to the ambient recorder
/// - `parent = <expr>` — expression resolving to `Option<&[u8]>`, a W3C `traceparent`
///   (e.g. `RequestContext::traceparent()`, or bytes carried by hand from a caller's
///   own span). `Some` continues that trace (same `trace_id`, `parent_span_id` set);
///   `None`/absent opens a fresh root. Proxima carries span context as explicit
///   data — never an ambient/thread-local "current span" — so a caller wanting a
///   child span MUST pass this.
/// - `fields(key = <expr>, "dotted.key" = <expr>, bare_ident)` — typed scalar tags on
///   the span. A bare identifier captures the argument of that name by value. Each
///   value must be `Into<ScalarValue>`; proxima tags are typed scalars, never `Debug`
///   strings, so a non-convertible expression is a compile error at the call site.
/// - `err` — run the body, and set the span's status to `Error` when it returns
///   `Err`. Requires a `Result` return; every return path (including an early
///   `return`) flows through the check.
/// - `budget = <ns>` — tail-sampling force-keep: if the span outruns this many
///   nanoseconds, the trace is kept regardless of the head sampling decision.
///
/// # Examples
///
/// ```
/// use proxima_macros::span;
///
/// #[span]
/// fn do_work(input: &str) -> usize { input.len() }
///
/// #[span(name = "explicit", level = "warn")]
/// async fn fetch(url: &str) -> Result<usize, &'static str> { Ok(url.len()) }
///
/// // a child span is opened by CARRYING the parent's traceparent, never by
/// // reading an ambient "current span".
/// #[span(parent = traceparent)]
/// fn handle(traceparent: Option<&[u8]>) -> u8 { 1 }
///
/// #[span(kind = "server", fields(component = "auth", attempt), err, budget = 5_000_000)]
/// fn authenticate(attempt: u64) -> Result<(), &'static str> { Ok(()) }
///
/// // with no recorder installed the body still runs, just span-free — the
/// // same no-op contract the `info!` / `debug!` macros have.
/// assert_eq!(do_work("proxima"), 7);
/// assert_eq!(handle(None), 1);
/// assert_eq!(authenticate(1), Ok(()));
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
/// ```
/// use proxima::telemetry::id::SpanId;
/// // the trait (type namespace) and the derive (macro namespace) share a name,
/// // so both imports are needed — the same shape as serde's `Serialize`.
/// use proxima::telemetry::trace::SpanCarrier;
/// use proxima_macros::SpanCarrier;
///
/// #[derive(SpanCarrier)]
/// struct Envelope {
///     span_id: Option<SpanId>,
///     payload: Vec<u8>,
/// }
///
/// #[derive(SpanCarrier)]
/// struct Message {
///     #[span_id]
///     trace_slot: Option<SpanId>,
///     body: String,
/// }
///
/// let mut envelope = Envelope { span_id: None, payload: vec![1, 2, 3] };
/// assert_eq!(envelope.span_id(), None);
///
/// let id = SpanId::from_bytes([1, 2, 3, 4, 5, 6, 7, 8]);
/// envelope.set_span_id(Some(id));
/// assert_eq!(envelope.span_id(), Some(id));
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
/// Message style is enforced, not merely documented: a literal that starts with
/// a capital or ends in `.`/`!`/`?` fails to expand.
///
/// ```
/// use core::error::Error as _;
/// use proxima_macros::Error;
///
/// #[derive(Debug)]
/// pub struct WireError;
/// impl core::fmt::Display for WireError {
///     fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
///         f.write_str("bad wire byte")
///     }
/// }
/// impl core::error::Error for WireError {}
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
///     Upstream(#[source] WireError),
///
///     #[error(transparent)]
///     Wire(WireError),
/// }
///
/// assert_eq!(DecodeError::InvalidMagic(0x7e).to_string(), "invalid magic byte: 126");
/// assert_eq!(DecodeError::TruncatedFrame.to_string(), "truncated frame");
/// // `transparent` forwards both Display and source() to the inner error.
/// assert_eq!(DecodeError::Wire(WireError).to_string(), "bad wire byte");
/// assert!(DecodeError::Upstream(WireError).source().is_some());
/// assert!(DecodeError::TruncatedFrame.source().is_none());
/// ```
///
/// ```compile_fail
/// # use proxima_macros::Error;
/// #[derive(Error, Debug)]
/// pub enum Shouty {
///     #[error("Invalid magic byte.")]
///     Invalid,
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
///   schema.
/// - `#[schema(skip)]` on a field — omit it from the schema.
///
/// Serde's own attributes are read too, so the schema tracks the wire with no
/// second annotation, and the contract's required-set matches what actually
/// deserializes:
///
/// - `#[serde(rename = "...")]` supplies the field name when `#[schema(rename)]`
///   is absent; `#[schema(rename)]` wins when both are present.
/// - `#[serde(default)]` on a field, or on the struct, marks it (or every field)
///   absent-tolerant.
///
/// `Option<T>` fields are marked optional (absent-allowed) automatically.
///
/// The emitted code holds at the alloc tier, so a `#![no_std]` consumer can
/// derive this — it names `alloc` through a local `extern crate alloc;` rather
/// than `std`, and never relies on `ToString` already being in scope.
///
/// # Examples
///
/// ```
/// use proxima_config::schema::{Describe, Schema as SchemaIr};
/// use proxima_macros::Schema;
///
/// #[derive(Schema)]
/// struct Memory {
///     id: String,
///     score: Option<f64>,
///     #[schema(rename = "type")]
///     kind: String,
/// }
///
/// let SchemaIr::Struct { name, fields } = Memory::schema() else {
///     panic!("a struct derives a struct schema");
/// };
/// assert_eq!(name, "Memory");
/// let described: Vec<(&str, bool)> = fields
///     .iter()
///     .map(|field| (field.name.as_str(), field.flags.optional))
///     .collect();
/// assert_eq!(described, [("id", false), ("score", true), ("type", false)]);
/// ```
#[proc_macro_derive(Schema, attributes(schema))]
pub fn derive_schema(item: TokenStream) -> TokenStream {
    describe::expand(item.into())
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}
